//! CPU layout/origin and preparation regressions; decoding is test oracle only.
use super::*;
use crate::TensorRole;
use crate::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use crate::nn::{ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec};
use crate::transformer::hybrid_parallel::{HybridExpertScope, HybridParallelPlan};
use crate::transformer::*;
use ferrule_common::{ParallelRankId, ParallelTopologyId};
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) fn layout(degree: usize, lane: usize) -> Arc<HybridRankLayout> {
    let norm = || RmsNorm::new(259, 1e-6).unwrap();
    let rope = RotaryEmbedding::new(
        64,
        10000.0,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 64 },
        RotaryScaling::None,
    )
    .unwrap();
    let layers = [
        Attention::GatedDeltaNet(
            GatedDeltaNetAttention::new(259, 4, 8, 65, 65, 3, 1e-6, false).unwrap(),
        ),
        Attention::Gqa(GqaAttention::new(259, 8, 2, 64, false, rope).unwrap()),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, attention)| {
        DecoderLayer::new(
            index,
            norm(),
            attention,
            Residual::Add,
            norm(),
            FeedForward::SwiGlu(SwiGlu::new(259, 259, false).unwrap()),
            Residual::Add,
        )
        .unwrap()
    })
    .collect();
    let spec = DecoderModelSpec::new(DecoderModelParts {
        architecture: "numeric-rank-test".into(),
        hidden_size: 259,
        vocab_size: 259,
        max_sequence_length: Some(16),
        token_embedding: Embedding::new(259, 259, None).unwrap(),
        layers,
        final_norm: norm(),
        output: Linear::new(259, 259, false).unwrap(),
        tie_word_embeddings: false,
    })
    .unwrap();
    Arc::new(
        HybridParallelPlan::new(
            &spec,
            0..2,
            HybridExpertScope {
                topology: ParallelTopologyId::new(81),
                replica: 2,
                stage: 0,
            },
            [7, 19, 31, 63][..degree]
                .iter()
                .copied()
                .map(ParallelRankId::new)
                .collect(),
            None,
        )
        .unwrap()
        .rank_layout(lane)
        .unwrap(),
    )
}

pub(crate) struct Fixture {
    directory: std::path::PathBuf,
    pub binding: BoundParameter,
    pub raw: Vec<u8>,
    pub scales: Vec<f32>,
}
impl Fixture {
    pub(crate) fn new(
        shape: [usize; 2],
        role: TensorRole,
        layer: Option<usize>,
        bf16: bool,
    ) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "numeric-rank-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let raw = (0..shape[0] * shape[1])
            .map(|i| [0x38, 0xb8, 0x30, 0x40][i % 4])
            .collect::<Vec<_>>();
        let scales = (0..shape[0].div_ceil(128) * shape[1].div_ceil(128))
            .map(|i| (i + 1) as f32 / 4.0)
            .collect::<Vec<_>>();
        let weight = CheckpointTensorSlice {
            name: "projection.weight".into(),
            role: role.clone(),
            path: directory.join("weight"),
            offset: 19,
            bytes: raw.len() as u64,
            shape: shape.to_vec(),
            dtype: CheckpointDType::F8E4M3,
        };
        let scale_bytes = scales
            .iter()
            .flat_map(|s| {
                if bf16 {
                    half::bf16::from_f32(*s).to_bits().to_le_bytes().to_vec()
                } else {
                    s.to_le_bytes().to_vec()
                }
            })
            .collect::<Vec<_>>();
        let scale = CheckpointTensorSlice {
            name: "projection.scale".into(),
            role: role.clone(),
            path: directory.join("scale"),
            offset: 7,
            bytes: scale_bytes.len() as u64,
            shape: shape.map(|n| n.div_ceil(128)).to_vec(),
            dtype: if bf16 {
                CheckpointDType::Bf16
            } else {
                CheckpointDType::F32
            },
        };
        std::fs::write(&weight.path, [vec![0xff; 19], raw.clone()].concat()).unwrap();
        std::fs::write(&scale.path, [vec![0xff; 7], scale_bytes].concat()).unwrap();
        let path = ModulePath::new("projection").unwrap();
        let spec = ParameterSpec::new(
            ParameterId::new(1),
            path.clone(),
            ParameterDType::F8E4M3,
            shape,
            layer.map_or(ParameterResidency::Static, |layer| {
                ParameterResidency::Layer { layer }
            }),
        )
        .unwrap()
        .with_required_scale(
            if bf16 {
                ParameterDType::Bf16
            } else {
                ParameterDType::F32
            },
            scale.shape.clone(),
        )
        .unwrap();
        let mut schema = StateDictSchema::builder();
        schema.register_with_role(spec, role).unwrap();
        let mut mapper = ExactNameMapper::new();
        mapper
            .insert(&weight.name, NameMapping::weight(path.clone()))
            .unwrap();
        mapper
            .insert(&scale.name, NameMapping::scale(path.clone()))
            .unwrap();
        let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
            .bind_slices(vec![weight, scale])
            .unwrap();
        Self {
            directory,
            binding: bound.get(&path).unwrap().clone(),
            raw,
            scales,
        }
    }
    fn expected(&self, rows: Range<usize>, columns: Range<usize>) -> Vec<f32> {
        let k = self.binding.weight().slice().shape[1];
        rows.flat_map(|r| {
            columns.clone().map(move |c| {
                crate::checkpoint::decode_fp8_e4m3fn_byte(self.raw[r * k + c])
                    * self.scales[r / 128 * k.div_ceil(128) + c / 128]
            })
        })
        .collect()
    }
    pub(crate) fn prepare(&self, degree: usize, lane: usize) -> PreparedNumericFp8RankLinear {
        let plan = NumericFp8RankPlan::new(&self.binding, layout(degree, lane)).unwrap();
        StateDictMaterializer::new(plan.budget().payload_bytes as u64)
            .unwrap()
            .prepare_numeric_fp8_rank_linear(&plan, plan.budget().allocation_grant_bytes)
            .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn numeric_rank_gdn_qkv_retains_global_origins_and_paired_scale_reads() {
    for bf16 in [false, true] {
        let f = Fixture::new([1040, 259], TensorRole::LinearAttentionQkv, Some(0), bf16);
        let plan = NumericFp8RankPlan::new(&f.binding, layout(2, 1)).unwrap();
        assert_eq!(plan.partition().rows, [130..260, 390..520, 780..1040]);
        assert_eq!(plan.local_shape(), [520, 259]);
        assert!(plan.budget().payload_bytes < f.raw.len());
        let reader = StateDictMaterializer::new(plan.budget().payload_bytes as u64).unwrap();
        let prepared = reader
            .prepare_numeric_fp8_rank_linear(&plan, plan.budget().allocation_grant_bytes)
            .unwrap();
        assert_eq!(
            reader.checkpoint_read_counters().bytes_read,
            plan.budget().payload_bytes as u64
        );
        assert_eq!(reader.checkpoint_read_counters().open_calls, 2);
        assert_eq!(reader.checkpoint_read_counters().source_checks, 4);
        assert_eq!(reader.tensor_reads().unwrap().len(), 6);
        let mut end = 0;
        for (segment, global) in prepared.segments().iter().zip(&plan.partition().rows) {
            let artifact = segment.artifact();
            assert_eq!(segment.output_columns(), end..end + global.len());
            end += global.len();
            assert_eq!(
                artifact.validated_payload().layout().row_origin,
                global.start
            );
            assert_eq!(artifact.provenance().weight_read().rows(), global.clone());
            assert_eq!(
                artifact.decode_f32(1 << 22).unwrap(),
                f.expected(global.clone(), 0..259)
            );
            let clone = artifact.clone();
            assert_eq!(
                clone.weight_bytes().as_ptr(),
                artifact.weight_bytes().as_ptr()
            );
            assert_eq!(
                clone.scale_bytes().as_ptr(),
                artifact.scale_bytes().as_ptr()
            );
        }
        assert_eq!(end, 520);
    }
}

#[test]
fn numeric_rank_row_column_and_kv_replica_partitions_are_local() {
    for bf16 in [false, true] {
        for (role, shape, layer) in [
            (TensorRole::DenseMlpGate, [259, 259], Some(0)),
            (TensorRole::DenseMlpUp, [259, 259], Some(0)),
            (TensorRole::DenseMlpDown, [259, 259], Some(0)),
            (TensorRole::AttentionOutput, [259, 520], Some(0)),
            (TensorRole::AttentionKey, [128, 259], Some(1)),
            (TensorRole::OutputHead, [259, 259], None),
        ] {
            let f = Fixture::new(shape, role, layer, bf16);
            for degree in [1, 2, 4] {
                for lane in 0..degree {
                    let p = f.prepare(degree, lane);
                    assert_eq!(p.segments().len(), 1);
                    let read = &p.plan().reads()[0];
                    let expected =
                        f.expected(read.weight_read().rows(), read.weight_read().columns());
                    assert_eq!(
                        p.segments()[0].artifact().decode_f32(1 << 22).unwrap(),
                        expected
                    );
                    assert_eq!(p.plan().layout().owner().get(), [7, 19, 31, 63][lane]);
                    assert_eq!(
                        read.scale_read().rows().start,
                        read.weight_read().rows().start / 128
                    );
                    assert_eq!(
                        read.scale_read().columns().start,
                        read.weight_read().columns().start / 128
                    );
                }
            }
        }
    }
}

#[test]
fn numeric_rank_ragged_row_partials_sum_to_full_matrix_oracle() {
    let f = Fixture::new([259, 259], TensorRole::DenseMlpDown, Some(0), false);
    let x = (0..259).map(|i| (i % 3) as f32 - 1.0).collect::<Vec<_>>();
    let mut sum = vec![0.0; 259];
    for lane in 0..4 {
        let prepared = f.prepare(4, lane);
        let columns = prepared.plan().partition().columns.clone();
        let w = prepared.segments()[0]
            .artifact()
            .decode_f32(1 << 22)
            .unwrap();
        for (r, row) in w.chunks_exact(columns.len()).enumerate() {
            sum[r] += row
                .iter()
                .zip(&x[columns.clone()])
                .map(|(a, b)| a * b)
                .sum::<f32>();
        }
    }
    let full = f.expected(0..259, 0..259);
    let expected = full
        .as_chunks::<259>()
        .0
        .iter()
        .map(|r| r.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>())
        .collect::<Vec<_>>();
    assert_eq!(sum, expected);
}

#[test]
fn numeric_rank_budget_checks_aggregate_and_separate_physical_scale_grants_before_io() {
    let f = Fixture::new([1040, 259], TensorRole::LinearAttentionQkv, Some(0), false);
    let plan = NumericFp8RankPlan::new(&f.binding, layout(2, 1)).unwrap();
    let budget = plan.budget();
    let expected = plan
        .reads()
        .iter()
        .map(|r| {
            grant(r.weight_read().read_plan().storage_bytes() as usize).unwrap()
                + grant(r.scale_read().read_plan().storage_bytes() as usize).unwrap()
        })
        .sum::<usize>();
    assert_eq!(budget.allocation_grant_bytes, expected);
    assert!(expected > budget.payload_bytes);
    for (payload, grants) in [
        (budget.payload_bytes - 1, expected),
        (budget.payload_bytes, expected - 1),
    ] {
        let reader = StateDictMaterializer::new(payload as u64).unwrap();
        assert!(
            reader
                .prepare_numeric_fp8_rank_linear(&plan, grants)
                .is_err()
        );
        assert_eq!(reader.checkpoint_read_counters().open_calls, 0);
        assert_eq!(reader.checkpoint_read_counters().read_calls, 0);
    }
    let reader = StateDictMaterializer::new(budget.payload_bytes as u64).unwrap();
    reader
        .prepare_numeric_fp8_rank_linear(&plan, expected)
        .unwrap();
    assert_eq!(
        reader
            .tensor_reads()
            .unwrap()
            .iter()
            .map(|r| r.bytes)
            .sum::<u64>(),
        budget.payload_bytes as u64
    );
}

#[test]
fn numeric_rank_stale_either_source_and_wrong_binding_or_lane_fail_closed() {
    for scale in [false, true] {
        let f = Fixture::new([259, 259], TensorRole::DenseMlpDown, Some(0), false);
        let plan = NumericFp8RankPlan::new(&f.binding, layout(2, 1)).unwrap();
        assert!(plan.validate_binding(&f.binding, &layout(2, 0)).is_err());
        let other = Fixture::new([259, 259], TensorRole::DenseMlpDown, Some(0), false);
        assert!(
            plan.validate_binding(&other.binding, &layout(2, 1))
                .is_err()
        );
        let path = if scale {
            f.binding.scale().unwrap().slice().path.clone()
        } else {
            f.binding.weight().slice().path.clone()
        };
        let bytes = std::fs::read(&path).unwrap();
        std::fs::rename(&path, path.with_extension("old")).unwrap();
        std::fs::write(&path, bytes).unwrap();
        let reader = StateDictMaterializer::new(plan.budget().payload_bytes as u64).unwrap();
        assert!(
            reader
                .prepare_numeric_fp8_rank_linear(&plan, usize::MAX)
                .is_err()
        );
        assert_eq!(reader.checkpoint_read_counters().bytes_read, 0);
        // Planning/identity comparison are pure metadata; no recaptured snapshot.
        assert!(NumericFp8RankPlan::new(&f.binding, layout(2, 1)).is_ok());
        assert!(plan.validate_binding(&f.binding, &layout(2, 1)).is_ok());
    }
}

#[test]
fn numeric_rank_arbitrary_rectangle_origins_preserve_minimal_scale_grid() {
    for bf16 in [false, true] {
        let f = Fixture::new([259, 259], TensorRole::DenseMlpDown, Some(0), bf16);
        for origin in [0, 1, 125, 127, 128, 129, 255, 256, 258] {
            for col in [0, 1, 125, 127, 128, 129, 255, 256, 258] {
                let p = HybridParameterPartition {
                    rows: std::iter::once(origin..259).collect(),
                    columns: col..259,
                    collective: None,
                };
                let plan = NumericFp8RankPlan::from_partition(&f.binding, layout(2, 1), p).unwrap();
                let read = &plan.reads()[0];
                let reader = crate::checkpoint::CheckpointTensorReader::new(
                    plan.budget().payload_bytes as u64,
                );
                let a = read.read(&reader).unwrap();
                assert_eq!(a.validated_payload().layout().row_origin, origin);
                assert_eq!(a.validated_payload().layout().column_origin, col);
                assert_eq!(
                    a.decode_f32(1 << 22).unwrap(),
                    f.expected(origin..259, col..259)
                );
                assert_eq!(a.storage_bytes() as usize, plan.budget().payload_bytes);
            }
        }
    }
}

#[test]
fn numeric_rank_rejects_forged_partition_before_payload_io_and_tp1_stays_full() {
    let f = Fixture::new([259, 259], TensorRole::DenseMlpDown, Some(0), false);
    let p = HybridParameterPartition {
        rows: std::iter::once(0..1).collect(),
        columns: 0..1,
        collective: None,
    };
    let forged = NumericFp8RankPlan::from_partition(&f.binding, layout(2, 1), p).unwrap();
    let reader = StateDictMaterializer::new(1 << 20).unwrap();
    assert!(
        reader
            .prepare_numeric_fp8_rank_linear(&forged, usize::MAX)
            .is_err()
    );
    assert_eq!(reader.checkpoint_read_counters().read_calls, 0);
    let full = reader
        .prepared_linear(&f.binding, f.binding.role().clone())
        .unwrap();
    assert!(matches!(
        full.storage(),
        PreparedLinearStorage::NumericFp8(_)
    ));
    assert_eq!(full.numeric_fp8().unwrap().local_shape(), [259, 259]);
    assert!(full.weight().is_err());
    assert!(full.parameter().values_f32().is_err());
    let one = f.prepare(1, 0);
    assert_eq!(one.segments()[0].artifact(), full.numeric_fp8().unwrap());
}

#[cfg(feature = "cuda")]
#[test]
fn numeric_rank_cuda_budget_without_a_device_counts_all_real_grants() {
    let f = Fixture::new([1040, 259], TensorRole::LinearAttentionQkv, Some(0), true);
    let plan = NumericFp8RankPlan::new(&f.binding, layout(2, 1)).unwrap();
    for rows in [1, 2, 9, 17] {
        for precision in [
            NumericFp8Precision::F32Tf32x3,
            NumericFp8Precision::Bf16RneF32Accumulate,
            NumericFp8Precision::W8A16Bf16F32Accumulate,
        ] {
            let budget = plan
                .cuda_execution_budget(rows, 1 << 20, precision)
                .unwrap();
            assert_eq!(
                budget.resident_grant_bytes,
                plan.budget().allocation_grant_bytes + grant(budget.workspace_bytes).unwrap()
            );
            assert_eq!(budget.input_grant_bytes, grant(rows * 259 * 4).unwrap());
            assert_eq!(
                budget.output_grant_bytes,
                grant(rows * 130 * 4).unwrap() * 2 + grant(rows * 260 * 4).unwrap()
            );
            assert_eq!(
                budget.total_grant_bytes,
                budget.resident_grant_bytes + budget.input_grant_bytes + budget.output_grant_bytes
            );
            if precision == NumericFp8Precision::W8A16Bf16F32Accumulate
                || (precision == NumericFp8Precision::F32Tf32x3 && rows == 1)
            {
                assert_eq!(budget.workspace_grant_bytes, 0);
            }
        }
    }
    assert!(
        plan.cuda_execution_budget(0, 1 << 20, NumericFp8Precision::F32Tf32x3)
            .is_err()
    );
    assert!(
        plan.cuda_execution_budget(2, 1, NumericFp8Precision::F32Tf32x3)
            .is_err()
    );
}
