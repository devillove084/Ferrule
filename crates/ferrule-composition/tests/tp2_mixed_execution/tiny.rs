//! Tiny actual Mixed worker fixture; all GPU tests require the separate owner grant.
use super::*;
use ferrule_model::{
    checkpoint::{CheckpointDType, CheckpointTensorReader, CheckpointTensorSlice},
    nn::{ModulePath, ParameterDType, ParameterId, ParameterSpec},
    transformer::*,
};
use std::sync::Arc;
#[allow(dead_code)]
#[path = "../../../ferrule-model/tests/support/hybrid_tp_ep_fixture.rs"]
pub(super) mod base;

pub(super) fn resources(f: &base::Fixture) -> BoundDecoderResources {
    let original = f.resources();
    let path = f.dir.join("runtime-mixed-native.bin");
    let reader = CheckpointTensorReader::new(1 << 20);
    let mut data = Vec::new();
    let mut slices = Vec::new();
    let mut schema = StateDictSchema::builder();
    let mut mapper = ExactNameMapper::new();
    for (index, binding) in original.state_dict().parameters().iter().enumerate() {
        let source = binding.weight().slice();
        let start = data.len();
        let native_f32 = matches!(
            binding.role(),
            ferrule_model::TensorRole::LinearAttentionNorm
                | ferrule_model::TensorRole::LinearAttentionALog
        );
        for value in reader.read_slice(source).unwrap().bytes.as_chunks::<4>().0 {
            let value = *value;
            if native_f32 {
                data.extend(value);
            } else {
                data.extend(
                    ferrule_model::execution::bf16_rne_word(f32::from_le_bytes(value))
                        .to_le_bytes(),
                );
            }
        }
        let name = ModulePath::new(&source.name).unwrap();
        schema
            .register_with_role(
                ParameterSpec::new(
                    ParameterId::new(index as u64 + 1),
                    name.clone(),
                    if native_f32 {
                        ParameterDType::F32
                    } else {
                        ParameterDType::Bf16
                    },
                    source.shape.clone(),
                    binding.residency().clone(),
                )
                .unwrap(),
                binding.role().clone(),
            )
            .unwrap();
        mapper
            .insert(&source.name, NameMapping::weight(name))
            .unwrap();
        slices.push(CheckpointTensorSlice {
            name: source.name.clone(),
            role: source.role.clone(),
            path: path.clone(),
            offset: start as u64,
            bytes: (data.len() - start) as u64,
            dtype: if native_f32 {
                CheckpointDType::F32
            } else {
                CheckpointDType::Bf16
            },
            shape: source.shape.clone(),
        });
    }
    std::fs::write(path, data).unwrap();
    let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
        .bind_slices(slices)
        .unwrap();
    BoundDecoderResources::new(original.spec().clone(), Arc::new(bound)).unwrap()
}

#[test]
#[ignore = "two-GPU owner only: real Mixed claim/worker state fork/release and local routed shards"]
fn mixed_claim_workers_prefill_fork_decode_release() {
    grant();
    let fixture = base::Fixture::new();
    let resources = resources(&fixture);
    let mut pipeline = build(
        resources,
        PipelineConfig {
            page_size: 2,
            max_pages: 16,
            max_positions: 32,
            max_batch_tokens: 2,
            max_parameter_bytes: 1 << 20,
            ..config()
        },
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let output = pipeline
            .forward(SessionId(1), &[1, 2], ForwardPhase::Prefill)
            .unwrap();
        assert_eq!(output.logits.width(), 11);
        assert!(output.logits.values().iter().all(|n| n.is_finite()));
        pipeline.fork_session(SessionId(1), SessionId(2)).unwrap();
        let child = pipeline
            .forward(SessionId(2), &[3], ForwardPhase::Decode)
            .unwrap();
        let parent = pipeline
            .forward(SessionId(1), &[3], ForwardPhase::Decode)
            .unwrap();
        assert_eq!(
            child.logits, parent.logits,
            "journal fork must be independent and reproducible"
        );
        for id in [1, 2] {
            pipeline.release_session(SessionId(id)).unwrap();
        }
        assert_eq!(pipeline.outstanding(), 0);
        assert!(
            pipeline
                .owner_stats()
                .unwrap()
                .iter()
                .all(|o| o.experts.is_empty() && o.sessions == 0)
        );
    }));
    pipeline.shutdown().unwrap();
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}

#[test]
fn tiny_mixed_claim_metadata_matches_two_owner_budgets_without_cuda() {
    let fixture = base::Fixture::new();
    let resources = resources(&fixture);
    let plan = ParallelExecutionPlan::describe(resources.spec(), &placement(), contract(), vec![])
        .unwrap();
    let config = PipelineConfig {
        page_size: 2,
        max_pages: 16,
        max_positions: 32,
        max_batch_tokens: 2,
        max_parameter_bytes: 1 << 20,
        ..config()
    };
    let limits = HostCollectiveLimits {
        max_ranks: 2,
        max_elements_per_rank: 16,
        max_host_bytes: 65536,
    };
    let inputs = ferrule_runtime::parallel::pipeline::MixedTensorExecutionPlan::new(
        resources.spec(),
        plan.topology(),
        plan.segments(),
        plan.layout(),
        config,
        limits,
    )
    .unwrap();
    for lane in 0..2 {
        let (estimate, context) = inputs
            .program(lane)
            .unwrap()
            .cuda_claim_inputs(
                &resources,
                32,
                lane,
                config.max_parameter_bytes,
                inputs.attention_shapes(),
            )
            .unwrap();
        let budget = ferrule_model::decoder::HybridCudaMemoryBudget {
            kv_pages: 32,
            state_bytes: estimate.per_sequence_state_bytes * 5,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        };
        context.validate_device_budget(lane, budget).unwrap();
        assert!(context.validate_device_budget(1 - lane, budget).is_err());
        assert!(
            context
                .validate_device_budget(
                    lane,
                    ferrule_model::decoder::HybridCudaMemoryBudget {
                        workspace_bytes: budget.workspace_bytes - 1,
                        ..budget
                    }
                )
                .is_err()
        );
    }
}
