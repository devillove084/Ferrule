use super::*;
use ferrule_model::transformer::NumericFp8Precision as P;

#[test]
fn mixed_factory_rejects_wrong_boundary_backend_and_legacy_options_before_io() {
    let config = AutoConfig::from_descriptor(tests::descriptor(
        ModelFamily::Qwen35Moe,
        "must-not-read",
        40,
    ));
    for (backend, model) in [
        (BackendSelection::Cpu, Some(ModelPrecision::Bf16)),
        (BackendSelection::Cuda, Some(ModelPrecision::F32)),
        (BackendSelection::Cuda, None),
    ] {
        let result = ResidentModelPlanner.prepare_numeric_fp8_bf16_serving_with_precision(
            &config,
            backend,
            None,
            tests::options(),
            ServingModelOptions {
                model_precision: model,
                ..Default::default()
            },
            NumericStoragePolicy::E4M3FnBf16Scales,
            P::Bf16ModelMixedF32Accumulate,
        );
        assert!(result.err().unwrap().to_string().contains("BF16"));
    }
    let mut opts = tests::options();
    opts.numeric_fp8_profile = Some(NumericFp8Profile::W8A16Bf16);
    assert!(
        ResidentModelPlanner
            .prepare_numeric_fp8_bf16_serving_with_precision(
                &config,
                BackendSelection::Cuda,
                None,
                opts,
                ServingModelOptions {
                    model_precision: Some(ModelPrecision::Bf16),
                    ..Default::default()
                },
                NumericStoragePolicy::E4M3FnBf16Scales,
                P::Bf16ModelMixedF32Accumulate
            )
            .is_err()
    );
}

#[path = "bf16_chunk/admission.rs"]
mod chunk;
