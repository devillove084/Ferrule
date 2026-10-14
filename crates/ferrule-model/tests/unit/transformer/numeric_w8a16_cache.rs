use super::*;
use crate::transformer::NumericFp8Precision;
use crate::transformer::standard::scratch::SwiGluScratchPlan;

#[test]
fn w8a16_cache_identity_cannot_hit_other_arithmetic_profiles() {
    let mut cache = Cache::new(ExpertCacheLimits {
        max_experts: 7,
        max_bytes: 2058,
    })
    .unwrap();
    for precision in [
        NumericFp8Precision::F32Tf32x3,
        NumericFp8Precision::Bf16RneF32Accumulate,
        NumericFp8Precision::W8A16Bf16F32Accumulate,
        NumericFp8Precision::Bf16ModelF32Accumulate,
        NumericFp8Precision::Bf16ModelTiledF32Accumulate,
        NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate,
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
    ] {
        let key = ExpertCacheKey {
            image_generation: 1,
            numeric_precision: Some(precision),
            layer: 0,
            expert: 0,
            parameters: [
                (ParameterId::new(1), TensorRole::RoutedExpertGate),
                (ParameterId::new(2), TensorRole::RoutedExpertUp),
                (ParameterId::new(3), TensorRole::RoutedExpertDown),
            ],
        };
        let lease = cache
            .acquire(key.clone(), 294, |_| panic!("distinct profiles fit"))
            .unwrap();
        cache.complete(&lease, true).unwrap();
        let hit = cache
            .acquire(key, 294, |_| panic!("same profile hits"))
            .unwrap();
        cache.complete(&hit, true).unwrap();
    }
    assert_eq!(cache.stats().uploads, 7);
    assert_eq!(cache.stats().hits, 7);
    assert_eq!(cache.stats().resident_bytes, 2058);
}

#[test]
fn w8a16_zero_numeric_scratch_still_charges_operation_and_compressed_experts() {
    let operation = 832;
    let compressed = 294;
    let scratch = SwiGluScratchPlan::reserved_bytes(operation, 0, 0).unwrap();
    let mut cache = Cache::new(ExpertCacheLimits {
        max_experts: 1,
        max_bytes: operation + compressed,
    })
    .unwrap();
    cache.preflight(compressed, scratch).unwrap();
    assert!(cache.preflight(compressed, scratch + 1).is_err());
    cache.scratch(scratch, |_| panic!()).unwrap();
    let lease = cache.acquire(0, compressed, |_| panic!()).unwrap();
    assert_eq!(cache.stats().charged_bytes(), operation + compressed);
    cache.complete(&lease, true).unwrap();
    cache.quarantine();
    let frozen = cache.stats();
    assert_eq!(frozen.unknown_quarantine_bytes, operation + compressed);
    assert!(cache.scratch(0, |_| panic!()).is_err());
    assert_eq!(cache.stats(), frozen);
}

#[test]
fn mixed_qkv_row_transition_keeps_cache_identity_and_exact_budget() {
    use ferrule_backend::cuda_plan::{NumericFp8Layout, NumericFp8ScaleType};
    let precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
    let layout = NumericFp8Layout {
        n: 8192,
        k: 2048,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::Bf16,
    };
    let scratch = 8 << 20;
    let compressed = 294;
    let key = ExpertCacheKey {
        image_generation: 1,
        numeric_precision: Some(precision),
        layer: 0,
        expert: 0,
        parameters: [
            (ParameterId::new(1), TensorRole::RoutedExpertGate),
            (ParameterId::new(2), TensorRole::RoutedExpertUp),
            (ParameterId::new(3), TensorRole::RoutedExpertDown),
        ],
    };
    let mut cache = Cache::new(ExpertCacheLimits {
        max_experts: 1,
        max_bytes: scratch + compressed,
    })
    .unwrap();
    for m in [1, 5, 9, 1] {
        let plan = precision
            .linear_plan(&TensorRole::LinearAttentionQkv, layout, m, scratch)
            .unwrap();
        let grant = plan.workspace_allocation_grant_bytes().unwrap();
        assert_eq!(grant, scratch);
        cache.preflight(compressed, grant).unwrap();
        assert!(cache.preflight(compressed, grant + 1).is_err());
        cache
            .scratch(grant, |_| panic!("row transition must not evict"))
            .unwrap();
        let lease = cache
            .acquire(key.clone(), compressed, |_| panic!("same image identity"))
            .unwrap();
        cache.complete(&lease, true).unwrap();
        assert_eq!(cache.stats().charged_bytes(), scratch + compressed);
    }
    assert_eq!(cache.stats().uploads, 1);
    assert_eq!(cache.stats().hits, 3);
    assert_eq!(cache.stats().evictions, 0);
}
