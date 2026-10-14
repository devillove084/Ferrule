use super::*;
use ferrule_model::transformer::NumericFp8Precision as P;

#[test]
fn tiled_bf16_factory_rejects_legacy_arithmetic_cpu_and_f32_boundaries_before_io() {
    let config = AutoConfig::from_descriptor(tests::descriptor(
        ModelFamily::Qwen35Moe,
        "metadata-only",
        40,
    ));
    for (backend, model, precision) in [
        (
            BackendSelection::Cuda,
            Some(ModelPrecision::Bf16),
            P::F32Tf32x3,
        ),
        (
            BackendSelection::Cuda,
            Some(ModelPrecision::Bf16),
            P::W8A16Bf16F32Accumulate,
        ),
        (
            BackendSelection::Cuda,
            Some(ModelPrecision::Bf16),
            P::Bf16RneF32Accumulate,
        ),
        (
            BackendSelection::Cpu,
            Some(ModelPrecision::Bf16),
            P::Bf16ModelTiledF32Accumulate,
        ),
        (
            BackendSelection::Cuda,
            Some(ModelPrecision::F32),
            P::Bf16ModelTiledF32Accumulate,
        ),
        (BackendSelection::Cuda, None, P::Bf16ModelTiledF32Accumulate),
    ] {
        let err = ResidentModelPlanner
            .prepare_numeric_fp8_bf16_serving_with_precision(
                &config,
                backend,
                None,
                tests::options(),
                ServingModelOptions {
                    model_precision: model,
                    ..Default::default()
                },
                NumericStoragePolicy::E4M3FnBf16Scales,
                precision,
            )
            .err()
            .expect("reject before IO");
        assert!(err.to_string().contains("BF16"), "{err}");
    }
}
#[test]
fn tiled_factory_rejects_legacy_profile_and_parallel_policy_before_io() {
    let config = AutoConfig::from_descriptor(tests::descriptor(
        ModelFamily::Qwen35Moe,
        "metadata-only",
        40,
    ));
    let serving = ServingModelOptions {
        model_precision: Some(ModelPrecision::Bf16),
        ..Default::default()
    };
    let mut options = tests::options();
    options.numeric_fp8_profile = Some(NumericFp8Profile::F32);
    assert!(
        ResidentModelPlanner
            .prepare_numeric_fp8_bf16_serving_with_precision(
                &config,
                BackendSelection::Cuda,
                None,
                options,
                serving.clone(),
                NumericStoragePolicy::E4M3FnBf16Scales,
                P::Bf16ModelTiledF32Accumulate
            )
            .is_err()
    );
    assert!(
        ResidentModelPlanner
            .prepare_numeric_fp8_bf16_serving_with_precision(
                &config,
                BackendSelection::Cuda,
                None,
                tests::options(),
                ServingModelOptions {
                    parallel: Some(PipelineBuildOptions::default()),
                    ..serving
                },
                NumericStoragePolicy::E4M3FnBf16Scales,
                P::Bf16ModelTiledF32Accumulate
            )
            .is_err()
    );
}
