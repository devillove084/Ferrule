use super::*;

#[test]
fn tiled_model_precision_maps_to_measured_native_backend_not_fused() {
    use ferrule_backend::cuda_plan::NumericFp8Precision as Backend;
    let tiled = NumericFp8Precision::Bf16ModelTiledF32Accumulate;
    let sliced = NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate;
    assert!(sliced.has_bf16_model_boundaries());
    assert_eq!(
        sliced.hybrid_backend_name(),
        "cuda-hybrid-numeric-fp8-model-bf16-sliced-k2-f32-ssm"
    );
    assert_ne!(sliced.hybrid_backend_name(), tiled.hybrid_backend_name());
    assert!(tiled.has_bf16_model_boundaries());
    assert!(!tiled.allows_zero_scratch());
    assert_eq!(tiled.backend(), Backend::Bf16RneF32Accumulate);
    assert_eq!(
        NumericFp8Precision::Bf16ModelF32Accumulate.backend(),
        Backend::W8A16Bf16F32Accumulate
    );
    for legacy in [
        NumericFp8Precision::Bf16RneF32Accumulate,
        NumericFp8Precision::F32Tf32x3,
        NumericFp8Precision::W8A16Bf16F32Accumulate,
        NumericFp8Precision::Bf16ModelF32Accumulate,
    ] {
        assert_ne!(
            tiled.standard_backend_name(),
            legacy.standard_backend_name()
        );
        assert_ne!(tiled.hybrid_backend_name(), legacy.hybrid_backend_name());
    }
    assert_eq!(
        NumericFp8Precision::default(),
        NumericFp8Precision::Bf16RneF32Accumulate
    );
    let limits = ExpertCacheLimits {
        max_experts: 1,
        max_bytes: 8192,
    };
    let default = options(ExecutionPrecisionPolicy::bf16_compatibility())
        .with_hybrid_cuda_numeric_fp8_bf16(limits, 0)
        .unwrap();
    assert_eq!(
        default.hybrid_cuda_numeric_fp8_precision(),
        Some(NumericFp8Precision::Bf16ModelF32Accumulate)
    );
}
