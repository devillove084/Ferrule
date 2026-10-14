use super::*;
use crate::args::ServeModelPrecision;

#[test]
fn explicit_model_precision_is_forwarded_separately_from_numeric_profile() {
    let args = serve(&[
        "ferrule",
        "serve",
        "model",
        "--model-precision",
        "bf16",
        "--numeric-fp8-precision",
        "w8a16-bf16",
    ]);
    assert_eq!(args.model_precision, Some(ServeModelPrecision::Bf16));
    assert_eq!(
        args.cuda_expert_device.numeric_precision.as_deref(),
        Some("w8a16-bf16")
    );
    let serving = serving_options(&args).unwrap();
    assert_eq!(
        serving.model_precision,
        Some(ferrule_composition::ModelPrecision::Bf16)
    );
}

#[allow(dead_code)]
#[path = "../../../../ferrule-composition/tests/dense_bf16/fixture.rs"]
mod dense_fixture;

#[test]
fn native_bf16_cli_selects_executable_standard_profile_and_preserves_defaults() {
    use ferrule_composition::model_factory::{ArithmeticPrecision, StorageEncoding};
    let fixture = dense_fixture::Fixture::new();
    let args_for = |extra: &[&str]| {
        let mut flags = vec![
            "ferrule",
            "serve",
            fixture.dir.path().to_str().unwrap(),
            "--ctx-size",
            "16",
        ];
        flags.extend(extra);
        serve(&flags)
    };
    let default = args_for(&[]);
    assert_eq!(serving_options(&default).unwrap().model_precision, None);
    assert_eq!(
        model_factory_options(&default).unwrap().numeric_fp8_profile,
        None
    );
    let args = args_for(&["--model-precision", "bf16", "--devices", "0"]);
    // CLI's implicit resident expert-cache defaults must not turn this into an
    // always-reject flag; explicit expert options still fail closed below.
    assert_eq!(
        model_factory_options(&args)
            .unwrap()
            .expert_cache
            .host_entries,
        64
    );
    let plan = prepare_model(&args).unwrap();
    assert_eq!(plan.backend_profile(), "cuda-dense-bf16-serial");
    assert_eq!(plan.support().storage, StorageEncoding::NativeBf16);
    assert_eq!(
        plan.support().arithmetic,
        ArithmeticPrecision::Bf16Compatibility
    );
    assert!(plan.dense_bf16_memory_plan().unwrap().unwrap().weight_bytes > 0);
    assert_eq!(
        plan.effective_options().scheduler_config.max_decode_batch,
        1
    );
    for extra in [
        ["--numeric-fp8-precision", "w8a16-bf16"],
        ["--numeric-fp8-precision", "f32"],
        ["--kv-execution-policy", "host-kv-decode"],
        ["--tensor-parallel", "2"],
        ["--pipeline-parallel", "2"],
        ["--expert-host-cache-entries", "64"],
        ["--engine", "resident"],
    ] {
        let mut flags = vec!["--model-precision", "bf16"];
        flags.extend(extra);
        assert!(
            prepare_model(&args_for(&flags)).is_err(),
            "accepted {extra:?}"
        );
    }
}

mod mixed;
mod mixed_tp2;
