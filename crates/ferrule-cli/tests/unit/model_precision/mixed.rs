use super::*;
use crate::args::ServeModelProfile;
use ferrule_model::transformer::{NumericFp8Bf16HybridCudaProfile, NumericFp8Precision};

#[path = "../../../../ferrule-composition/tests/qwen35_memory_plan/mixed_fixture.rs"]
pub(crate) mod fixture;

fn mixed_args(model: &str, extra: &[&str]) -> ServeArgs {
    let mut flags = vec![
        "ferrule",
        "serve",
        model,
        "--model-profile",
        "qwen35-mixed-bf16",
        "--ctx-size",
        "128",
        "--max-active-sequences",
        "2",
        "--max-batch-tokens",
        "8",
        "--prefill-chunk-size",
        "8",
        "--kv-cache-mb",
        "16",
        "--cuda-expert-device-entries",
        "8",
        "--cuda-expert-device-bytes",
        "335544320",
        "--cuda-expert-scratch-mb",
        "8",
        "--cuda-workspace-mb",
        "2048",
        "--cuda-device-budget-mb",
        "7168",
    ];
    flags.extend(extra);
    serve(&flags)
}

#[test]
fn mixed_help_and_legacy_flags_are_unambiguous() {
    let args = mixed_args("model", &[]);
    assert_eq!(args.model_profile, Some(ServeModelProfile::Qwen35MixedBf16));
    assert_eq!(serving_options(&args).unwrap().model_precision, None);
    assert_eq!(
        model_factory_options(&args).unwrap().numeric_fp8_profile,
        None
    );
    assert_eq!(serve(&["ferrule", "serve", "model"]).model_profile, None);
    for flags in [
        vec!["--numeric-fp8-precision", "mixed-bf16"],
        vec!["--cuda-device-budget-mb", "7168"],
        vec!["--cuda-workspace-mb", "2048"],
        vec!["--cuda-expert-scratch-mb", "8"],
    ] {
        let mut argv = vec!["ferrule", "serve", "model"];
        argv.extend(flags);
        assert!(Cli::try_parse_from(argv).is_err());
    }
    let help = Cli::try_parse_from(["ferrule", "serve", "--help"])
        .err()
        .unwrap()
        .to_string();
    assert!(help.contains("qwen35-mixed-bf16"));
    assert!(help.contains("CUBLAS_WORKSPACE_CONFIG=:0:0"));
    assert!(help.contains("fresh quality is blocking"));
}

#[test]
fn mixed_cli_routes_metadata_to_existing_owner_and_accepts_lazy_confirmation() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_cli_routes_metadata_to_existing_owner_and_accepts_lazy_confirmation"
        ),
        || {
            let f = fixture::Fixture::new();
            for extra in [
                vec![],
                vec!["--expert-prewarm", "lazy", "--engine", "resident"],
            ] {
                let args = mixed_args(f.dir.path().to_str().unwrap(), &extra);
                let options = model_factory_options(&args).unwrap();
                assert!(options.qwen35_host_cache.is_none());
                let provenance = serving_provenance(&args);
                assert_eq!(provenance.expert_prewarm_workers, None);
                assert_eq!(
                    provenance.expert_prewarm,
                    if extra.is_empty() {
                        None
                    } else {
                        Some(ferrule_model::transformer::host_experts::ExpertPrewarmMode::Lazy)
                    }
                );
                let direct = ResidentModelPlanner::new()
                    .prepare_serving(
                        &AutoConfig::from_pretrained(&args.model).unwrap(),
                        ferrule_composition::BackendSelection::Cuda,
                        args.chat_template.as_deref(),
                        options,
                        serving_options(&args).unwrap(),
                    )
                    .unwrap();
                let plan = prepare_model(&args).unwrap();
                assert_eq!(
                    plan.resolution_report().to_string(),
                    direct.resolution_report().to_string()
                );
                assert_eq!(
                    plan.backend_profile(),
                    NumericFp8Bf16HybridCudaProfile::MIXED_NAME
                );
                assert_eq!(plan.numeric_fp8_profile(), None);
                assert!(plan.pipeline_options().is_none());
                let memory = plan.numeric_bf16_memory_plan().unwrap().unwrap();
                assert_eq!(
                    memory.numeric_precision(),
                    NumericFp8Precision::Bf16ModelMixedF32Accumulate
                );
                let limits = plan.qwen35_moe_capacity_limits().unwrap().unwrap();
                assert_eq!(
                    (
                        limits.max_positions,
                        limits.max_sequences,
                        limits.max_batch_tokens
                    ),
                    (128, 2, 8)
                );
                assert_eq!(
                    (
                        limits.device_bytes,
                        limits.workspace_bytes,
                        limits.scratch_bytes,
                        limits.kv_bytes
                    ),
                    (7 << 30, 2 << 30, 8 << 20, 16 << 20)
                );
                assert_eq!(limits.allocator_margin_bytes, 512 << 20);
                assert!(memory.total_device_bytes() <= limits.device_bytes);
                assert!(
                    !plan
                        .effective_options()
                        .driver_config
                        .enable_native_proposals
                );
            }
            let mut args = mixed_args(f.dir.path().to_str().unwrap(), &[]);
            args.cuda_device_budget_mb = Some(8192);
            assert_eq!(
                prepare_model(&args)
                    .unwrap()
                    .qwen35_moe_capacity_limits()
                    .unwrap()
                    .unwrap()
                    .device_bytes,
                8 << 30
            );
            args.cuda_device_budget_mb = Some(1);
            assert!(prepare_model(&args).is_err());
            args.cuda_device_budget_mb = Some(7168);
            args.max_layers = Some(1);
            assert!(prepare_model(&args).is_err());
        },
    );
}

// Policy comparisons use valid CLI syntax, nonzero common limits, CUDA selection
// and the required startup environment. CLI parsing/overflow and the production
// CPU-backend restriction intentionally remain separate from composition policy.
fn assert_shared_policy_error(args: &ServeArgs) {
    // CLI-only explicit cache provenance is rejected before composition lowering.
    if let Err(direct) = model_factory_options(args) {
        assert_eq!(
            prepare_model(args).err().unwrap().to_string(),
            direct.to_string()
        );
        return;
    }
    let direct = serving_options(args)
        .unwrap()
        .validate_before_model_io(
            ferrule_composition::BackendSelection::Cuda,
            &model_factory_options(args).unwrap(),
            &serving_provenance(args),
        )
        .unwrap_err();
    assert!(matches!(
        direct,
        ferrule_runtime::Error::InvalidRequest { .. }
    ));
    let cli = prepare_model(args)
        .err()
        .expect("policy must reject before model I/O");
    let source = cli.downcast_ref::<ferrule_runtime::Error>().unwrap();
    assert!(matches!(
        source,
        ferrule_runtime::Error::InvalidRequest { .. }
    ));
    assert_eq!(cli.to_string(), direct.to_string());
}

#[test]
fn mixed_cli_invalid_policy_rejects_before_model_io() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_cli_invalid_policy_rejects_before_model_io"
        ),
        || {
            for flags in [
                ["--engine", "pipeline"],
                ["--pipeline-parallel", "2"],
                ["--tensor-parallel", "2"],
                ["--expert-parallel", "2"],
                ["--devices", "0"],
                ["--rank-backend", "process"],
                ["--rank-restarts", "1"],
                ["--rank-timeout-ms", "1"],
                ["--expert-prewarm", "full"],
                ["--expert-prewarm-workers", "0"],
                ["--expert-prewarm-workers", "4"],
                ["--expert-host-cache-entries", "0"],
                ["--expert-host-cache-entries", "64"],
                ["--expert-host-cache-mb", "0"],
                ["--expert-host-cache-mb", "1024"],
                ["--expert-pinned-cache-entries", "0"],
                ["--expert-pinned-cache-entries", "16"],
                ["--expert-pinned-cache-mb", "0"],
                ["--expert-pinned-cache-mb", "256"],
                ["--moe-hotset-experts", "0"],
                ["--model-precision", "f32"],
                ["--model-precision", "bf16"],
                ["--numeric-fp8-precision", "f32"],
                ["--numeric-fp8-precision", "w8a16-bf16"],
                ["--kv-execution-policy", "host-kv-decode"],
            ] {
                assert_shared_policy_error(&mixed_args("must-not-read", &flags));
            }
            for change in 0..6 {
                let mut args = mixed_args("must-not-read", &[]);
                match change {
                    0 => args.max_batch_tokens = 33,
                    1 => args.ctx_size = 1,
                    2 => args.cuda_device_budget_mb = Some(0),
                    3 => args.cuda_workspace_mb = Some(0),
                    4 => args.cuda_expert_scratch_mb = Some(320),
                    _ => args.cuda_expert_device.entries = Some(0),
                }
                assert_shared_policy_error(&args);
            }
            let args = mixed_args("must-not-read", &["--backend", "cpu"]);
            let error = prepare_model(&args).err().unwrap();
            if cfg!(feature = "cpu-reference") {
                let direct = serving_options(&args)
                    .unwrap()
                    .validate_before_model_io(
                        ferrule_composition::BackendSelection::Cpu,
                        &model_factory_options(&args).unwrap(),
                        &serving_provenance(&args),
                    )
                    .unwrap_err();
                assert_eq!(error.to_string(), direct.to_string());
            } else {
                assert!(error.to_string().contains("CPU execution is test-only"));
            }
        },
    );
}

#[test]
fn mixed_cli_checks_unit_conversion_without_owning_budget_acceptance() {
    for change in 0..3 {
        let mut args = mixed_args("must-not-read", &[]);
        match change {
            0 => args.cuda_device_budget_mb = Some(usize::MAX),
            1 => args.cuda_workspace_mb = Some(usize::MAX),
            _ => args.cuda_expert_scratch_mb = Some(usize::MAX),
        }
        assert!(
            prepare_model(&args)
                .err()
                .unwrap()
                .to_string()
                .contains("overflows bytes")
        );
    }
    let mut args = mixed_args("must-not-read", &[]);
    args.cuda_device_budget_mb = Some(0);
    assert_eq!(
        model_factory_options(&args)
            .unwrap()
            .qwen35_moe_capacity
            .unwrap()
            .device_bytes,
        0
    );
}

#[test]
fn mixed_cli_valid_policy_reaches_model_io_without_rewriting_options() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_cli_valid_policy_reaches_model_io_without_rewriting_options"
        ),
        || {
            for extra in [vec![], vec!["--expert-prewarm", "lazy"]] {
                let args = mixed_args("must-not-read", &extra);
                let (backend, options, serving) = validate_args(&args).unwrap();
                assert!(options.qwen35_host_cache.is_none());
                assert_eq!(options.scheduler_config.max_batch_tokens, 8);
                serving
                    .validate_before_model_io(backend, &options, &serving_provenance(&args))
                    .unwrap();
                let error = prepare_model(&args).err().unwrap();
                assert!(
                    error.downcast_ref::<ferrule_runtime::Error>().is_none(),
                    "{error}"
                );
                assert!(error.to_string().contains("must-not-read"), "{error}");
            }
        },
    );
}

#[test]
fn mixed_startup_rejects_environment_before_model_io() {
    const CHILD: &str = "FERRULE_MIXED_STARTUP_TEST_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let args = mixed_args("must-not-read", &[]);
        assert_shared_policy_error(&args);
        let before = std::env::var_os("CUBLAS_WORKSPACE_CONFIG");
        let error = cmd_serve(args).unwrap_err().to_string();
        assert_eq!(std::env::var_os("CUBLAS_WORKSPACE_CONFIG"), before);
        assert!(error.contains("CUBLAS_WORKSPACE_CONFIG=:0:0"), "{error}");
        assert!(error.contains("restart"));
        return;
    }
    let name = concat!(
        module_path!(),
        "::mixed_startup_rejects_environment_before_model_io"
    )
    .split_once("::")
    .unwrap()
    .1;
    for value in [None, Some(""), Some(":4096:8"), Some(":0:0 ")] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        child
            .args(["--exact", name, "--nocapture"])
            .env(CHILD, "1")
            .env("CUDA_VISIBLE_DEVICES", "");
        if let Some(value) = value {
            child.env("CUBLAS_WORKSPACE_CONFIG", value);
        } else {
            child.env_remove("CUBLAS_WORKSPACE_CONFIG");
        }
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
    }
}

#[test]
fn mixed_smoke_configuration_selects_actual_public_model_options() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_smoke_configuration_selects_actual_public_model_options"
        ),
        || {
            let f = fixture::Fixture::new();
            let config: serde_json::Value =
                serde_json::from_str(include_str!("../../support/mixed_serve_smoke.json")).unwrap();
            let mut argv = vec!["ferrule", "serve", f.dir.path().to_str().unwrap()];
            for (flag, value) in config["serve_flags"].as_object().unwrap() {
                argv.push(flag);
                argv.push(value.as_str().unwrap());
            }
            let args = serve(&argv);
            let plan = prepare_model(&args).unwrap();
            assert_eq!(
                plan.backend_profile(),
                NumericFp8Bf16HybridCudaProfile::MIXED_NAME
            );
            let options = plan.effective_options();
            assert_eq!(options.driver_config.ctx_size, 128);
            assert_eq!(options.scheduler_config.max_active_sequences, 2);
            assert_eq!(options.scheduler_config.max_batch_tokens, 8);
            assert_eq!(options.scheduler_config.prefill_chunk_size, 8);
            assert_eq!(options.kv_cache_mebibytes, Some(16));
            assert!(!options.driver_config.enable_native_proposals);
            let limits = plan.qwen35_moe_capacity_limits().unwrap().unwrap();
            assert_eq!(
                (limits.max_experts, limits.max_bytes, limits.scratch_bytes),
                (8, 320 << 20, 8 << 20)
            );
            assert_eq!(
                (
                    limits.device_bytes,
                    limits.workspace_bytes,
                    limits.allocator_margin_bytes
                ),
                (7 << 30, 2 << 30, 512 << 20)
            );
            assert_eq!(
                plan.numeric_bf16_memory_plan()
                    .unwrap()
                    .unwrap()
                    .numeric_precision(),
                NumericFp8Precision::Bf16ModelMixedF32Accumulate
            );
            assert_eq!(
                plan.qwen35_host_cache_options().unwrap().mode,
                ferrule_model::transformer::host_experts::ExpertPrewarmMode::Lazy
            );
            assert!(plan.pipeline_options().is_none() && plan.qwen35_expert_placement().is_none());
            assert_eq!(
                worker_config(&args).unwrap().kv_execution_policy,
                ferrule_runtime::KvExecutionPolicy::Resident
            );
            assert_eq!(
                args.served_model_name.as_deref(),
                Some("qwen35-mixed-smoke")
            );
        },
    );
}
