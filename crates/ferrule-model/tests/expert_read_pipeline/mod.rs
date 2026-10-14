//! Real CPU checkpoint worker + Standard materializer. No CUDA owner or fake
//! artifact constructor; owner windows below are not GPU overlap evidence.

mod lifecycle;

use super::*;
use crate::checkpoint::{CheckpointDType, CheckpointReadCounters, CheckpointTensorSlice};
use crate::nn::{ModulePath, ParameterDType, ParameterId, ParameterSpec};
use crate::support::TensorRole;
use crate::transformer::{
    BoundStateDict, ExactNameMapper, NameMapping, PreparedLinear, PreparedLinearStorage,
    StateDictBinder, StateDictMaterializer, StateDictSchema,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const IMAGE: u64 = 71;
const EXPERT_BYTES: u64 = 18;
const METADATA: u64 = 256 << 10;
const STACK: usize = 1 << 20;
static NEXT: AtomicU64 = AtomicU64::new(0);

pub(crate) struct Fixture {
    directory: PathBuf,
    path: PathBuf,
    pub(crate) bound: BoundStateDict,
    pub(crate) metadata: Vec<ExpertMetadata>,
}

impl Fixture {
    pub(crate) fn new(experts: usize) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "ferrule-expert-read-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("weights.bin");
        let mut data = Vec::new();
        let mut slices = Vec::new();
        let mut schema = StateDictSchema::builder();
        let mut mapper = ExactNameMapper::new();
        for expert in 0..experts {
            for (projection, role) in [
                TensorRole::RoutedExpertGate,
                TensorRole::RoutedExpertUp,
                TensorRole::RoutedExpertDown,
            ]
            .into_iter()
            .enumerate()
            {
                let name = format!("layer.0.expert.{expert}.projection.{projection}");
                let module = ModulePath::new(&name).unwrap();
                schema
                    .register_with_role(
                        ParameterSpec::new(
                            ParameterId::new((expert * 3 + projection + 1) as u64),
                            module.clone(),
                            ParameterDType::F8E4M3,
                            vec![2, 2],
                            ParameterResidency::expert(0, expert),
                        )
                        .unwrap()
                        .with_required_scale(ParameterDType::Bf16, vec![1, 1])
                        .unwrap(),
                        role.clone(),
                    )
                    .unwrap();
                mapper
                    .insert(
                        format!("{name}.weight"),
                        NameMapping::weight(module.clone()),
                    )
                    .unwrap();
                mapper
                    .insert(format!("{name}.scale"), NameMapping::scale(module))
                    .unwrap();
                slices.push(CheckpointTensorSlice {
                    name: format!("{name}.weight"),
                    role: role.clone(),
                    path: path.clone(),
                    offset: data.len() as u64,
                    bytes: 4,
                    dtype: CheckpointDType::F8E4M3,
                    shape: vec![2, 2],
                });
                data.extend([0x38 + projection as u8 + expert as u8; 4]);
                slices.push(CheckpointTensorSlice {
                    name: format!("{name}.scale"),
                    role,
                    path: path.clone(),
                    offset: data.len() as u64,
                    bytes: 2,
                    dtype: CheckpointDType::Bf16,
                    shape: vec![1, 1],
                });
                data.extend(half::bf16::ONE.to_bits().to_le_bytes());
            }
        }
        std::fs::write(&path, data).unwrap();
        let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
            .bind_slices(slices)
            .unwrap();
        let metadata = (0..experts)
            .map(|expert| {
                ExpertMetadata::new(
                    0,
                    expert,
                    bound.parameters()[expert * 3..expert * 3 + 3]
                        .to_vec()
                        .try_into()
                        .unwrap(),
                    None,
                )
                .unwrap()
            })
            .collect();
        Self {
            directory,
            path,
            bound,
            metadata,
        }
    }

    pub(crate) fn materializer(&self, schedule: ExpertReadSchedule) -> StateDictMaterializer {
        let mut result = StateDictMaterializer::new(128).unwrap();
        result
            .configure_expert_read_pipeline(options(schedule))
            .unwrap();
        result
    }

    fn sequence(&self, materializer: &StateDictMaterializer) -> ExpertReadSequence {
        materializer
            .expert_read_sequence(IMAGE, &self.metadata.iter().collect::<Vec<_>>())
            .unwrap()
            .unwrap()
    }

    fn replace(&self) {
        let replacement = self.directory.join("replacement.bin");
        std::fs::write(&replacement, std::fs::read(&self.path).unwrap()).unwrap();
        std::fs::rename(replacement, &self.path).unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

pub(crate) fn options(schedule: ExpertReadSchedule) -> ExpertReadPipelineOptions {
    ExpertReadPipelineOptions {
        schedule,
        max_host_bytes: 2 * EXPERT_BYTES + METADATA + STACK as u64,
        max_metadata_bytes: METADATA,
        worker_stack_bytes: STACK,
    }
}

fn prepared(materializer: &StateDictMaterializer, read: &CurrentExpertRead) -> PreparedSwiGlu {
    let linear = |index| {
        PreparedLinear::from_parameter(
            materializer
                .expert_parameter_from_read(read, index, &read.plan.parameters[index])
                .unwrap(),
            read.plan.parameters[index].role().clone(),
        )
        .unwrap()
    };
    PreparedSwiGlu::new(linear(0), linear(1), linear(2), None).unwrap()
}

fn bytes(expert: &PreparedSwiGlu) -> Vec<u8> {
    [expert.gate(), expert.up(), expert.down()]
        .into_iter()
        .flat_map(|linear| {
            let artifact = linear.numeric_fp8().unwrap();
            artifact
                .weight_bytes()
                .iter()
                .chain(artifact.scale_bytes())
                .copied()
        })
        .collect()
}

#[test]
fn canonical_three_projection_selection_is_payload_free_and_extent_ordered() {
    let f = Fixture::new(2);
    let reader = CheckpointTensorReader::new(128);
    let plans = ExpertReadSequence::select(
        options(ExpertReadSchedule::Blocking),
        IMAGE,
        &f.metadata.iter().collect::<Vec<_>>(),
        &reader,
    )
    .unwrap();
    assert_eq!(reader.read_counters(), CheckpointReadCounters::default());
    let plan = &plans[&(0, 1)];
    assert_eq!(plan.packed.storage_bytes(), EXPERT_BYTES);
    assert_eq!(plan.packed.source_files().len(), 1);
    assert_eq!(
        plan.packed
            .extents()
            .iter()
            .map(|extent| extent.offset())
            .collect::<Vec<_>>(),
        [18, 22, 24, 28, 30, 34]
    );
    assert_eq!(
        plan.ranges,
        [(0..4, 4..6), (6..10, 10..12), (12..16, 16..18)]
    );
    assert_eq!(
        plan.projections.each_ref().map(|read| read.local_shape()),
        [[2, 2]; 3]
    );
}

#[test]
fn opt_in_has_no_startup_reads_and_default_remains_unconfigured() {
    let mut materializer = StateDictMaterializer::new(128).unwrap();
    assert_eq!(materializer.expert_read_pipeline_options(), None);
    assert_eq!(ExpertReadSchedule::default(), ExpertReadSchedule::Blocking);
    materializer
        .configure_expert_read_pipeline(options(ExpertReadSchedule::Lookahead))
        .unwrap();
    assert_eq!(
        materializer.checkpoint_read_counters(),
        CheckpointReadCounters::default()
    );
    assert!(
        materializer
            .configure_expert_read_pipeline(options(ExpertReadSchedule::Blocking))
            .is_err()
    );
}

#[test]
fn double_buffer_admission_cap_minus_one_and_metadata_fail_before_io() {
    let f = Fixture::new(1);
    let reader = CheckpointTensorReader::new(128);
    let selected = [&f.metadata[0]];
    let good = options(ExpertReadSchedule::Blocking);
    assert!(ExpertReadSequence::select(good, IMAGE, &selected, &reader).is_ok());
    assert!(
        ExpertReadSequence::select(
            ExpertReadPipelineOptions {
                max_host_bytes: good.max_host_bytes - 1,
                ..good
            },
            IMAGE,
            &selected,
            &reader
        )
        .is_err()
    );
    assert!(
        ExpertReadSequence::select(
            ExpertReadPipelineOptions {
                max_metadata_bytes: 1,
                ..good
            },
            IMAGE,
            &selected,
            &reader
        )
        .is_err()
    );
    assert!(ExpertReadSequence::select(good, 0, &selected, &reader).is_err());
    assert!(
        ExpertReadSequence::select(good, IMAGE, &[&f.metadata[0], &f.metadata[0]], &reader)
            .is_err()
    );
    assert_eq!(reader.read_counters(), CheckpointReadCounters::default());
    assert!(
        ExpertReadSequence::select(good, IMAGE, &selected, &CheckpointTensorReader::new(6))
            .is_err()
    );
}

#[test]
fn both_schedules_use_identical_worker_limits_bytes_storage_and_custody() {
    let f = Fixture::new(3);
    let mut outputs = Vec::new();
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let materializer = f.materializer(schedule);
        let mut sequence = f.sequence(&materializer);
        assert_eq!(sequence.pipeline.options().worker_stack_bytes, STACK);
        assert_eq!(
            sequence.pipeline.budget().worker_stack.reserved_bytes,
            STACK
        );
        assert_eq!(
            sequence.options.limits().unwrap(),
            options(ExpertReadSchedule::Blocking).limits().unwrap()
        );
        let mut output = Vec::new();
        for expert in 0..3 {
            let value = prepared(&materializer, sequence.demand(0, expert).unwrap());
            sequence.track_artifacts(&value).unwrap();
            assert_eq!(sequence.pipeline.budget().reserved_bytes, EXPERT_BYTES);
            assert_eq!(sequence.artifact_bytes, EXPERT_BYTES);
            assert!(sequence.stats.peak_payload_bytes <= 2 * EXPERT_BYTES);
            assert!(sequence.stats.peak_host_charged_bytes <= options(schedule).max_host_bytes);
            output.push(bytes(&value));
            for linear in [value.gate(), value.up(), value.down()] {
                assert!(matches!(
                    linear.storage(),
                    PreparedLinearStorage::NumericFp8(_)
                ));
                assert_eq!(
                    linear.numeric_fp8().unwrap().decode_f32(16).unwrap().len(),
                    4
                );
            }
            drop(value); // Existing owner uploads release host payload before lookahead.
            sequence.lookahead((0, expert), (expert < 2).then_some((0, expert + 1)), false);
        }
        assert_eq!(sequence.stats.submitted_reads, 3);
        assert_eq!(sequence.stats.wasted_reads, 0);
        assert_eq!(
            materializer.checkpoint_read_counters().bytes_read,
            3 * EXPERT_BYTES
        );
        assert_eq!(materializer.checkpoint_read_counters().read_calls, 18);
        assert_eq!(
            sequence.stats.speculative_reads,
            if schedule == ExpertReadSchedule::Lookahead {
                2
            } else {
                0
            }
        );
        assert_eq!(sequence.pipeline.budget().reserved_bytes, 0);
        outputs.push(output);
    }
    assert_eq!(outputs[0], outputs[1]);
}

#[test]
fn verified_materializer_handoff_does_not_reread_and_matches_demand_artifacts() {
    let f = Fixture::new(1);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let mut sequence = f.sequence(&materializer);
    let read = sequence.demand(0, 0).unwrap();
    let before = materializer.checkpoint_read_counters();
    let value = prepared(&materializer, read);
    assert_eq!(materializer.checkpoint_read_counters(), before);
    let baseline = StateDictMaterializer::new(128).unwrap();
    for (index, linear) in [value.gate(), value.up(), value.down()]
        .into_iter()
        .enumerate()
    {
        let original = baseline
            .expert_parameter(0, 0, &f.bound.parameters()[index])
            .unwrap();
        assert_eq!(
            linear.numeric_fp8().unwrap(),
            original.numeric_fp8().unwrap()
        );
    }
    assert_eq!(before.bytes_read, EXPERT_BYTES);
}

#[test]
fn source_bound_identity_rejects_foreign_image_layer_expert_and_ticket() {
    let f = Fixture::new(2);
    let alien = Fixture::new(2);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let mut sequence = f.sequence(&materializer);
    let read = sequence.demand(0, 0).unwrap();
    assert!(
        materializer
            .expert_parameter_from_read(read, 0, &alien.bound.parameters()[0])
            .is_err()
    );
    assert!(
        materializer
            .expert_parameter_from_read(read, 0, &f.bound.parameters()[3])
            .is_err()
    );
    assert!(
        materializer
            .expert_parameter_from_read(read, 3, &f.bound.parameters()[0])
            .is_err()
    );
    let mut other = CheckpointReadPipeline::new(
        CheckpointTensorReader::new(128),
        options(ExpertReadSchedule::Blocking).limits().unwrap(),
    )
    .unwrap();
    let foreign = other
        .submit(SourceGeneration::new(IMAGE + 1), read.plan.packed.clone())
        .unwrap();
    other.wait(&foreign).unwrap();
    let foreign_completion = other.take(&foreign).unwrap();
    assert!(
        read.plan
            .validate_completion(&read.ticket, &foreign_completion)
            .is_err()
    );
    assert!(
        read.plan.projections[0]
            .materialize_verified_read_range(&read.completion, &foreign, 0..6)
            .is_err()
    );
}

#[test]
fn materializer_admits_one_sequence_and_drop_releases_only_proved_host_scope() {
    let f = Fixture::new(2);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let mut sequence = f.sequence(&materializer);
    sequence.lookahead((0, 0), Some((0, 1)), false);
    assert!(
        materializer
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .is_err()
    );
    drop(sequence); // Actual worker cancellation/join/drain.
    assert!(
        materializer
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .is_ok()
    );
}

#[test]
fn live_artifact_alias_prevents_read_credit_reuse_or_unaccounted_next_read() {
    let f = Fixture::new(2);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let mut sequence = f.sequence(&materializer);
    let value = prepared(&materializer, sequence.demand(0, 0).unwrap());
    sequence.track_artifacts(&value).unwrap();
    let PreparedLinearStorage::NumericFp8(artifact) = value.gate().storage() else {
        panic!()
    };
    let alias = Arc::clone(artifact);
    drop(value);
    sequence.lookahead((0, 0), Some((0, 1)), false);
    assert_eq!(sequence.stats.submitted_reads, 1);
    assert_eq!(sequence.pipeline.budget().reserved_bytes, EXPERT_BYTES);
    assert!(sequence.demand(0, 1).is_err());
    drop(alias);
    // The attempted scheduling violation is still delivered on demand, not
    // erased because a later release became possible.
    assert!(sequence.demand(0, 1).is_err());
}

#[test]
fn positive_hit_hint_skips_next_io_without_pin_or_route_reordering() {
    let f = Fixture::new(3);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let mut sequence = f.sequence(&materializer);
    let first = prepared(&materializer, sequence.demand(0, 0).unwrap());
    sequence.track_artifacts(&first).unwrap();
    drop(first);
    sequence.lookahead((0, 0), Some((0, 1)), true);
    assert!(sequence.pending.is_none());
    assert_eq!(sequence.stats.resident_skips, 1);
    // A later scratch eviction can turn the unpinned hint into a demand miss.
    let next = prepared(&materializer, sequence.demand(0, 1).unwrap());
    assert_eq!(
        next.gate().parameter().residency(),
        &ParameterResidency::expert(0, 1)
    );
    assert_eq!(
        materializer.checkpoint_read_counters().bytes_read,
        2 * EXPERT_BYTES
    );
}

#[test]
fn unused_speculation_is_discarded_accounted_and_does_not_change_error_order() {
    let f = Fixture::new(3);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let mut sequence = f.sequence(&materializer);
    sequence.lookahead((0, 0), Some((0, 1)), false);
    let ticket = sequence
        .pending
        .as_ref()
        .unwrap()
        .ticket
        .as_ref()
        .unwrap()
        .clone();
    sequence.pipeline.wait(&ticket).unwrap();
    assert_eq!(sequence.pipeline.budget().reserved_bytes, EXPERT_BYTES);
    // This bucket became a hit; the speculative read is not promoted to cache.
    sequence.lookahead((0, 1), Some((0, 2)), false);
    assert_eq!(sequence.stats.wasted_reads, 1);
    assert_eq!(sequence.stats.wasted_read_bytes, EXPERT_BYTES);
    let last = prepared(&materializer, sequence.demand(0, 2).unwrap());
    assert_eq!(
        last.gate().parameter().residency(),
        &ParameterResidency::expert(0, 2)
    );
    assert_eq!(
        materializer.checkpoint_read_counters().bytes_read,
        2 * EXPERT_BYTES
    );
}

#[test]
fn speculative_stale_failure_waits_for_demand_and_keeps_original_error_source() {
    let f = Fixture::new(2);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let mut sequence = f.sequence(&materializer);
    f.replace();
    sequence.lookahead((0, 0), Some((0, 1)), false); // no forward error here
    assert_eq!(sequence.stats.submitted_reads, 1);
    let error = sequence.demand(0, 1).unwrap_err();
    let Error::ModelSource { source } = &error else {
        panic!("original error source was lost")
    };
    assert!(
        std::error::Error::source(source.as_ref())
            .unwrap()
            .is::<Error>()
    );
    assert!(error.to_string().contains("stale"));
    assert_eq!(materializer.checkpoint_read_counters().bytes_read, 0);
    assert_eq!(sequence.pipeline.budget().reserved_bytes, EXPERT_BYTES);
    drop(error); // Failed completion charge remains until error custody drops.
    assert_eq!(sequence.pipeline.budget().reserved_bytes, 0);
}

#[test]
fn replacement_after_verified_read_rejects_before_artifact_publication() {
    let f = Fixture::new(1);
    let materializer = f.materializer(ExpertReadSchedule::Blocking);
    let mut sequence = f.sequence(&materializer);
    let read = sequence.demand(0, 0).unwrap();
    f.replace();
    assert!(
        materializer
            .expert_parameter_from_read(read, 0, &read.plan.parameters[0])
            .is_err()
    );
    assert_eq!(
        materializer.checkpoint_read_counters().bytes_read,
        EXPERT_BYTES
    );
}

#[test]
fn cancellation_retains_accepted_custody_until_actual_wait_take_drop() {
    let f = Fixture::new(2);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let mut sequence = f.sequence(&materializer);
    sequence.lookahead((0, 0), Some((0, 1)), false);
    let ticket = sequence
        .pending
        .as_ref()
        .unwrap()
        .ticket
        .as_ref()
        .unwrap()
        .clone();
    sequence.pipeline.cancel(&ticket).unwrap();
    assert_eq!(sequence.pipeline.budget().reserved_bytes, EXPERT_BYTES);
    sequence.pipeline.wait(&ticket).unwrap();
    assert_eq!(sequence.pipeline.budget().reserved_bytes, EXPERT_BYTES);
    sequence.discard_pending().unwrap();
    assert_eq!(sequence.pipeline.budget().reserved_bytes, 0);
    assert_eq!(sequence.stats.wasted_reads, 1);
}

#[test]
fn invalid_numeric_payload_fails_at_demand_without_reread_or_native_fallback() {
    let f = Fixture::new(1);
    let mut data = std::fs::read(&f.path).unwrap();
    data[0] = 0x7f;
    std::fs::write(&f.path, data).unwrap();
    // Bind the deliberately invalid bytes' current source generation.
    let rebound = StateDictBinder::new(
        &{
            let mut schema = StateDictSchema::builder();
            for binding in f.bound.parameters() {
                schema
                    .register_with_role(binding.spec().clone(), binding.role().clone())
                    .unwrap();
            }
            schema.build().unwrap()
        },
        &{
            let mut mapper = ExactNameMapper::new();
            for binding in f.bound.parameters() {
                mapper
                    .insert(
                        binding.weight().slice().name.clone(),
                        NameMapping::weight(binding.path().clone()),
                    )
                    .unwrap();
                mapper
                    .insert(
                        binding.scale().unwrap().slice().name.clone(),
                        NameMapping::scale(binding.path().clone()),
                    )
                    .unwrap();
            }
            mapper
        },
    )
    .bind_slices(f.bound.parameters().iter().flat_map(|binding| {
        [
            binding.weight().slice().clone(),
            binding.scale().unwrap().slice().clone(),
        ]
    }))
    .unwrap();
    let metadata = ExpertMetadata::new(
        0,
        0,
        rebound.parameters().to_vec().try_into().unwrap(),
        None,
    )
    .unwrap();
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let mut sequence = materializer
        .expert_read_sequence(IMAGE, &[&metadata])
        .unwrap()
        .unwrap();
    let read = sequence.demand(0, 0).unwrap();
    let error = materializer
        .expert_parameter_from_read(read, 0, &read.plan.parameters[0])
        .unwrap_err();
    assert!(error.to_string().contains("NaN"));
    assert_eq!(
        materializer.checkpoint_read_counters().bytes_read,
        EXPERT_BYTES
    );
}
