use super::fixture::Fixture;
use ferrule_model::transformer::{ExpertCacheLimits, NumericFp8Precision as P};
use ferrule_model::{
    ModelFamily, WeightSource,
    decoder::{GenericDecoderOptions, HybridMemoryPlan},
    execution::ExecutionPrecisionPolicy,
};
const TILED: P = P::Bf16ModelTiledF32Accumulate;
const SLICED: P = P::Bf16ModelSlicedK2F32Accumulate;
pub(super) fn base(f: &Fixture, policy: ExecutionPrecisionPolicy) -> GenericDecoderOptions {
    GenericDecoderOptions::standard_cpu(
        f.resources.spec(),
        ModelFamily::Unknown("tiled-bf16-tiny".into()),
        WeightSource::Safetensors,
        2,
        32,
        8,
        1,
        1 << 20,
        policy,
    )
    .unwrap()
}
pub(super) fn limits() -> ExpertCacheLimits {
    ExpertCacheLimits {
        max_experts: 1,
        max_bytes: 64 << 10,
    }
}
#[test]
fn tiled_requires_explicit_bf16_policy_nonzero_scratch_and_rejects_legacy_selection() {
    let f = Fixture::numeric_profile();
    let bf16 = base(&f, ExecutionPrecisionPolicy::bf16_compatibility());
    let f32 = base(&f, ExecutionPrecisionPolicy::f32());
    assert!(
        bf16.clone()
            .with_hybrid_cuda_numeric_fp8_precision(limits(), 4096, TILED)
            .is_err()
    );
    assert!(
        f32.clone()
            .with_hybrid_cuda_numeric_fp8_precision(limits(), 4096, TILED)
            .is_err()
    );
    assert!(
        f32.with_hybrid_cuda_numeric_fp8_bf16_precision(limits(), 4096, TILED)
            .is_err()
    );
    for p in [
        P::F32Tf32x3,
        P::W8A16Bf16F32Accumulate,
        P::Bf16RneF32Accumulate,
    ] {
        assert!(
            bf16.clone()
                .with_hybrid_cuda_numeric_fp8_bf16_precision(limits(), 4096, p)
                .is_err()
        );
    }
    for (max_experts, max_bytes, scratch) in [(1, 65536, 0), (0, 65536, 4096), (1, 4096, 4096)] {
        assert!(
            bf16.clone()
                .with_hybrid_cuda_numeric_fp8_bf16_precision(
                    ExpertCacheLimits {
                        max_experts,
                        max_bytes
                    },
                    scratch,
                    TILED
                )
                .is_err()
        );
    }
    let configured = bf16
        .with_hybrid_cuda_numeric_fp8_bf16_precision(limits(), 4096, TILED)
        .unwrap();
    assert_eq!(configured.hybrid_cuda_numeric_fp8_precision(), Some(TILED));
    let sliced = base(&f, ExecutionPrecisionPolicy::bf16_compatibility())
        .with_hybrid_cuda_numeric_fp8_bf16_precision(limits(), 4096, SLICED)
        .unwrap();
    assert_eq!(sliced.hybrid_cuda_numeric_fp8_precision(), Some(SLICED));
    assert!(
        configured
            .with_hybrid_cuda_numeric_fp8_bf16_precision(limits(), 4096, TILED)
            .is_err()
    );
}
#[test]
fn tiled_all_projection_admission_rejects_underbudget_and_partial_depth() {
    let f = Fixture::numeric_profile();
    let base = base(&f, ExecutionPrecisionPolicy::bf16_compatibility());
    let tiled = base
        .clone()
        .with_hybrid_cuda_numeric_fp8_bf16_precision(limits(), 4096, TILED)
        .unwrap();
    let plan = HybridMemoryPlan::for_resources(&f.resources, &tiled, 32).unwrap();
    let fused = base
        .clone()
        .with_hybrid_cuda_numeric_fp8_bf16(limits(), 0)
        .unwrap();
    let prior = HybridMemoryPlan::for_resources(&f.resources, &fused, 32).unwrap();
    assert_eq!(
        plan.per_sequence_state_bytes,
        prior.per_sequence_state_bytes
    );
    assert_eq!(plan.kv_bytes, prior.kv_bytes);
    assert_eq!(
        prior.weight_bytes_upper_bound - plan.weight_bytes_upper_bound,
        4096
    );
    assert!(
        HybridMemoryPlan::for_resources(
            &f.resources,
            &tiled.clone().with_active_layers(1).unwrap(),
            32
        )
        .is_err()
    );
    let undersized = base
        .with_hybrid_cuda_numeric_fp8_bf16_precision(limits(), 1, TILED)
        .unwrap();
    assert!(HybridMemoryPlan::for_resources(&f.resources, &undersized, 32).is_err());
    #[cfg(feature = "cuda")]
    assert!(
        ferrule_model::decoder::HybridCudaMemoryEstimate::for_resources_with_routed_experts(
            &f.resources,
            &tiled,
            32
        )
        .is_err()
    );
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "sole GPU owner only; explicit tiled tiny using unchanged independent HF fixture and existing all-state verifier"]
fn numeric_fp8_bf16_tiled_full_state_decode_cow_abort_restore_matches_hf() {
    super::cuda::full_state_with_precision(Fixture::numeric_profile(), TILED);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "sole GPU owner; existing tiny full-state fixture with explicit sliced-K2 schedule (K16 tiny uses documented tiled branch)"]
fn numeric_fp8_bf16_sliced_k2_full_state_decode_cow_abort_restore_matches_hf() {
    super::cuda::full_state_with_precision(Fixture::numeric_profile(), SLICED);
}

#[test]
fn sliced_schedule_tiny_budget_equals_tiled_without_silent_default_change() {
    let f = Fixture::numeric_profile();
    let bf16 = base(&f, ExecutionPrecisionPolicy::bf16_compatibility());
    let tiled = bf16
        .clone()
        .with_hybrid_cuda_numeric_fp8_bf16_precision(limits(), 4096, TILED)
        .unwrap();
    let sliced = bf16
        .clone()
        .with_hybrid_cuda_numeric_fp8_bf16_precision(limits(), 4096, SLICED)
        .unwrap();
    assert_eq!(
        HybridMemoryPlan::for_resources(&f.resources, &tiled, 32).unwrap(),
        HybridMemoryPlan::for_resources(&f.resources, &sliced, 32).unwrap()
    );
    assert!(
        bf16.clone()
            .with_hybrid_cuda_numeric_fp8_bf16_precision(limits(), 0, SLICED)
            .is_err()
    );
    assert!(
        bf16.with_hybrid_cuda_numeric_fp8_precision(limits(), 4096, SLICED)
            .is_err()
    );
}
