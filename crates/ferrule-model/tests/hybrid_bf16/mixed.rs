use super::fixture::Fixture;
use ferrule_model::{
    decoder::HybridMemoryPlan, execution::ExecutionPrecisionPolicy,
    transformer::NumericFp8Precision as P,
};

#[test]
fn mixed_tiny_admission_reserves_maximal_scratch_and_rejects_partial_depth() {
    let f = Fixture::numeric_profile();
    let base = super::tiled::base(&f, ExecutionPrecisionPolicy::bf16_compatibility());
    let mixed = base
        .clone()
        .with_hybrid_cuda_numeric_fp8_bf16_precision(
            super::tiled::limits(),
            4096,
            P::Bf16ModelMixedF32Accumulate,
        )
        .unwrap();
    let tiled = base
        .clone()
        .with_hybrid_cuda_numeric_fp8_bf16_precision(
            super::tiled::limits(),
            4096,
            P::Bf16ModelTiledF32Accumulate,
        )
        .unwrap();
    let mixed_plan = HybridMemoryPlan::for_resources(&f.resources, &mixed, 32).unwrap();
    let mut tiled_plan = HybridMemoryPlan::for_resources(&f.resources, &tiled, 32).unwrap();
    let chunk = HybridMemoryPlan::bf16_chunk_workspace_bytes(&f.resources, &mixed).unwrap();
    assert!(chunk > 0);
    tiled_plan.workspace_bytes_upper_bound +=
        ferrule_backend::cuda_allocation_grant_bytes(chunk).unwrap();
    tiled_plan.workspace_bytes_upper_bound +=
        HybridMemoryPlan::bf16_attention_plan(&f.resources, &mixed)
            .unwrap()
            .unwrap()
            .total_charge_bytes()
            + HybridMemoryPlan::bf16_attention_metadata_bytes(&f.resources, &mixed).unwrap();
    let (projection, external) = HybridMemoryPlan::bf16_gemmex_bytes(&f.resources, &mixed).unwrap();
    tiled_plan.workspace_bytes_upper_bound +=
        ferrule_backend::cuda_allocation_grant_bytes(projection).unwrap() + external;
    assert_eq!(mixed_plan, tiled_plan);
    assert!(
        HybridMemoryPlan::for_resources(&f.resources, &mixed.with_active_layers(1).unwrap(), 32)
            .is_err()
    );
    let under = base
        .with_hybrid_cuda_numeric_fp8_bf16_precision(
            super::tiled::limits(),
            1,
            P::Bf16ModelMixedF32Accumulate,
        )
        .unwrap();
    assert!(HybridMemoryPlan::for_resources(&f.resources, &under, 32).is_err());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "sole GPU owner; Mixed chunk GDN plus existing projection schedule and routed baseline, unchanged HF all-state oracle"]
fn numeric_fp8_bf16_mixed_full_state_decode_cow_abort_restore_matches_hf() {
    super::cuda::full_state_with_precision(
        Fixture::numeric_profile(),
        P::Bf16ModelMixedF32Accumulate,
    );
}
