//! CPU-only CLI/read-policy tests; no model payload or device owner.
use super::*;
use crate::args::ExpertReadArgs;
use ferrule_model::transformer::{ExpertReadPipelineOptions, ExpertReadSchedule};

#[path = "../../../ferrule-composition/tests/qwen35_memory_plan/mixed_fixture.rs"]
mod fixture;

#[test]
fn expert_read_disabled_is_not_an_implicit_blocking_pipeline() {
    let args = serve(&["ferrule", "serve", "not-opened"]);
    assert_eq!(args.expert_read.options().unwrap(), None);
    assert_eq!(serving_options(&args).unwrap().expert_read_pipeline, None);
    for fields in [
        ExpertReadArgs {
            schedule: Some("blocking".into()),
            ..Default::default()
        },
        ExpertReadArgs {
            metadata_mb: Some(8),
            ..Default::default()
        },
        ExpertReadArgs {
            worker_stack_mb: Some(2),
            ..Default::default()
        },
    ] {
        assert!(
            fields
                .options()
                .unwrap_err()
                .to_string()
                .contains("require --expert-read-host-mb")
        );
    }
}

#[test]
fn expert_read_flags_require_explicit_host_cap_and_validate_schedule() {
    for (flag, value) in [
        ("--expert-read-schedule", "blocking"),
        ("--expert-read-schedule", "lookahead"),
        ("--expert-read-metadata-mb", "8"),
        ("--expert-read-worker-stack-mb", "2"),
    ] {
        assert!(Cli::try_parse_from(["ferrule", "serve", "not-opened", flag, value]).is_err());
    }
    assert!(
        Cli::try_parse_from([
            "ferrule",
            "serve",
            "not-opened",
            "--expert-read-host-mb",
            "64",
            "--expert-read-schedule",
            "async",
        ])
        .is_err()
    );
}

#[test]
fn expert_read_schedules_share_host_limits_without_changing_device_budget() {
    for (schedule, expected) in [
        (None, ExpertReadSchedule::Blocking),
        (Some("blocking"), ExpertReadSchedule::Blocking),
        (Some("lookahead"), ExpertReadSchedule::Lookahead),
    ] {
        let mut argv = vec![
            "ferrule",
            "serve",
            "not-opened",
            "--expert-read-host-mb",
            "64",
        ];
        if let Some(schedule) = schedule {
            argv.extend(["--expert-read-schedule", schedule]);
        }
        let args = serve(&argv);
        let read = serving_options(&args)
            .unwrap()
            .expert_read_pipeline
            .unwrap();
        assert_eq!(read.schedule, expected);
        assert_eq!(read.max_host_bytes, 64 << 20);
        assert_eq!(read.max_metadata_bytes, 8 << 20);
        assert_eq!(read.worker_stack_bytes, 2 << 20);
        assert_eq!(
            model_factory_options(&args).unwrap().qwen35_moe_capacity,
            None
        );
        assert_eq!(
            model_factory_options(&args).unwrap().qwen35_host_cache,
            None
        );
    }
    let args = serve(&[
        "ferrule",
        "serve",
        "not-opened",
        "--expert-read-host-mb",
        "64",
        "--expert-read-metadata-mb",
        "4",
        "--expert-read-worker-stack-mb",
        "3",
        "--cuda-expert-device-bytes",
        "335544320",
    ]);
    let read = serving_options(&args)
        .unwrap()
        .expert_read_pipeline
        .unwrap();
    assert_eq!(
        (read.max_metadata_bytes, read.worker_stack_bytes),
        (4 << 20, 3 << 20)
    );
    assert_eq!(
        model_factory_options(&args)
            .unwrap()
            .qwen35_moe_capacity
            .unwrap()
            .max_bytes,
        335544320
    );
}

#[test]
fn expert_read_budget_zero_overflow_and_overhead_reject() {
    for flag in [
        "--expert-read-host-mb",
        "--expert-read-metadata-mb",
        "--expert-read-worker-stack-mb",
    ] {
        for value in ["0", "18446744073709551615"] {
            let mut argv = vec!["ferrule", "serve", "not-opened"];
            if flag != "--expert-read-host-mb" {
                argv.extend(["--expert-read-host-mb", "64"]);
            }
            argv.extend([flag, value]);
            assert!(serving_options(&serve(&argv)).is_err(), "{flag} {value}");
        }
    }
    for host in ["1", "10"] {
        assert!(
            serving_options(&serve(&[
                "ferrule",
                "serve",
                "not-opened",
                "--expert-read-host-mb",
                host,
            ]))
            .is_err()
        );
    }
}

#[test]
fn expert_read_unsupported_profiles_reject_before_model_io() {
    for flags in [
        vec![],
        vec!["--backend", "cpu"],
        vec!["--model-precision", "bf16"],
        vec!["--pipeline-parallel", "2"],
        vec![
            "--model-profile",
            "qwen35-mixed-bf16",
            "--tensor-parallel",
            "2",
        ],
        vec![
            "--model-profile",
            "qwen35-mixed-bf16",
            "--engine",
            "pipeline",
        ],
        vec![
            "--model-profile",
            "qwen35-mixed-bf16",
            "--kv-execution-policy",
            "host-kv-decode",
        ],
        vec![
            "--model-profile",
            "qwen35-mixed-bf16",
            "--expert-prewarm-workers",
            "1",
        ],
        vec![
            "--model-profile",
            "qwen35-mixed-bf16",
            "--expert-host-cache-entries",
            "1",
        ],
        vec![
            "--model-profile",
            "qwen35-mixed-bf16",
            "--numeric-fp8-precision",
            "w8a16-bf16",
        ],
        vec![
            "--model-profile",
            "qwen35-mixed-bf16",
            "--expert-prewarm",
            "full",
        ],
    ] {
        let mut argv = vec![
            "ferrule",
            "serve",
            "not-opened",
            "--expert-read-host-mb",
            "64",
            "--max-batch-tokens",
            "8",
        ];
        argv.extend(flags);
        let error = prepare_model(&serve(&argv))
            .err()
            .expect("unsupported before IO");
        assert!(!error.to_string().contains("No such file"), "{error}");
        assert!(
            error.to_string().contains("expert read")
                || error.to_string().contains("prewarm")
                || error.to_string().contains("expert options")
                || error.to_string().contains("CPU execution is test-only"),
            "{error}"
        );
    }
}

#[test]
fn expert_read_stack_and_metadata_sum_overflow_rejects_before_model_io() {
    let args = serve(&[
        "ferrule",
        "serve",
        "not-opened",
        "--expert-read-host-mb",
        "64",
        "--expert-read-metadata-mb",
        "17592186044415",
        "--expert-read-worker-stack-mb",
        "2",
    ]);
    let error = prepare_model(&args).err().expect("host overhead overflow");
    assert!(
        error.to_string().contains("host overhead overflows"),
        "{error}"
    );
}

fn storage_middle64_args(model: &str, schedule: Option<&str>) -> ServeArgs {
    let mut argv = vec![
        "ferrule",
        "serve",
        model,
        "--model-profile",
        "qwen35-mixed-bf16",
        "--expert-prewarm",
        "lazy",
        "--kv-execution-policy",
        "resident",
        "--ctx-size",
        "128",
        "--max-active-sequences",
        "2",
        "--max-batch-tokens",
        "8",
        "--prefill-chunk-size",
        "8",
        "--decode-cohort-target",
        "1",
        "--decode-cohort-max-deferrals",
        "0",
        "--kv-cache-mb",
        "16",
        "--max-tensor-mb",
        "1024",
        "--cuda-expert-device-entries",
        "64",
        "--cuda-expert-device-bytes",
        "335544320",
        "--cuda-expert-scratch-mb",
        "8",
        "--cuda-workspace-mb",
        "2048",
        "--cuda-device-budget-mb",
        "7168",
    ];
    if let Some(schedule) = schedule {
        argv.extend([
            "--expert-read-host-mb",
            "64",
            "--expert-read-schedule",
            schedule,
            "--expert-read-metadata-mb",
            "8",
            "--expert-read-worker-stack-mb",
            "2",
        ]);
    }
    serve(&argv)
}

#[test]
fn expert_read_v3_study_cohort_is_explicit_without_changing_cli_defaults() {
    let defaults = serve(&["ferrule", "serve", "not-opened"]);
    let scheduler = model_factory_options(&defaults).unwrap().scheduler_config;
    assert_eq!(defaults.decode_cohort_target, 4);
    assert_eq!(defaults.decode_cohort_max_deferrals, 3);
    assert_eq!(scheduler.decode_cohort_target, 4);
    assert_eq!(scheduler.decode_cohort_max_deferrals, 3);
    for schedule in [None, Some("blocking"), Some("lookahead")] {
        let args = storage_middle64_args("not-opened", schedule);
        let scheduler = model_factory_options(&args).unwrap().scheduler_config;
        assert_eq!(args.decode_cohort_target, 1);
        assert_eq!(args.decode_cohort_max_deferrals, 0);
        assert_eq!(scheduler.decode_cohort_target, 1);
        assert_eq!(scheduler.decode_cohort_max_deferrals, 0);
    }
}

#[test]
fn expert_read_cli_resource_report_rejects_stale_prepared_sources_before_owner() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::expert_read_cli_resource_report_rejects_stale_prepared_sources_before_owner"
        ),
        || {
            for schedule in ["blocking", "lookahead"] {
                for name in ["config.json", "model.safetensors"] {
                    let f = fixture::Fixture::new();
                    let args =
                        storage_middle64_args(f.dir.path().to_str().unwrap(), Some(schedule));
                    let plan = prepare_model(&args).unwrap();
                    assert!(serving_resource_report(&plan).unwrap().is_none());
                    let path = f.dir.path().join(name);
                    let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
                    file.set_len(file.metadata().unwrap().len() + 1).unwrap();
                    let error = serving_resource_report(&plan).unwrap_err();
                    assert!(
                        error.to_string().contains("stale"),
                        "{schedule} {name}: {error}"
                    );
                }
            }
        },
    );
}

#[test]
fn expert_read_cli_lowering_freezes_both_schedules_on_existing_storage_middle64_owner() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::expert_read_cli_lowering_freezes_both_schedules_on_existing_storage_middle64_owner"
        ),
        || {
            let f = fixture::Fixture::new();
            let model = f.dir.path().to_str().unwrap();
            let base_args = storage_middle64_args(model, None);
            let base = prepare_model(&base_args).unwrap();
            let base_memory = base.numeric_bf16_memory_plan().unwrap().unwrap();
            assert!(base.mixed_tp2_cache_requirements().unwrap().is_none());
            assert!(base.serving_resource_requirements().unwrap().is_none());
            assert!(serving_resource_report(&base).unwrap().is_none());
            assert_eq!(base.expert_read_pipeline_options(), None);
            assert_eq!(
                base_memory.runner_options().expert_read_pipeline_options(),
                None
            );
            assert!(base_memory.resources().host_experts().is_none());
            assert!(
                execution_report(&base, base_args.rank_timeout_ms)
                    .iter()
                    .all(|line| !line.contains("expert read host staging"))
            );
            let base_limits = base.qwen35_moe_capacity_limits().unwrap().unwrap();
            assert_eq!(
                (
                    base_limits.max_experts,
                    base_limits.max_bytes,
                    base_limits.scratch_bytes
                ),
                (64, 320 << 20, 8 << 20)
            );
            assert_eq!(
                (
                    base_limits.device_bytes,
                    base_limits.workspace_bytes,
                    base_limits.kv_bytes
                ),
                (7 << 30, 2 << 30, 16 << 20)
            );
            assert_eq!(base_limits.allocator_margin_bytes, 512 << 20);
            for (schedule, expected) in [
                ("blocking", ExpertReadSchedule::Blocking),
                ("lookahead", ExpertReadSchedule::Lookahead),
            ] {
                let read = ExpertReadPipelineOptions {
                    schedule: expected,
                    max_host_bytes: 64 << 20,
                    max_metadata_bytes: 8 << 20,
                    worker_stack_bytes: 2 << 20,
                };
                let args = storage_middle64_args(model, Some(schedule));
                assert_eq!(
                    serving_options(&args).unwrap().expert_read_pipeline,
                    Some(read)
                );
                assert!(
                    model_factory_options(&args)
                        .unwrap()
                        .qwen35_host_cache
                        .is_none()
                );
                let plan = prepare_model(&args).unwrap();
                let memory = plan.numeric_bf16_memory_plan().unwrap().unwrap();
                assert!(plan.mixed_tp2_cache_requirements().unwrap().is_none());
                assert!(plan.serving_resource_requirements().unwrap().is_none());
                assert!(serving_resource_report(&plan).unwrap().is_none());
                assert_eq!(
                    plan.selected_implementation().kind,
                    SelectedImplementationKind::Resident
                );
                assert_eq!(plan.backend_profile(), base.backend_profile());
                assert_eq!(plan.expert_read_pipeline_options(), Some(read));
                assert_eq!(memory.expert_read_pipeline_options(), Some(read));
                assert_eq!(
                    memory.runner_options().expert_read_pipeline_options(),
                    Some(read)
                );
                assert!(memory.resources().host_experts().is_none());
                assert_eq!(memory.model_budget(), base_memory.model_budget());
                assert_eq!(
                    memory.total_device_bytes(),
                    base_memory.total_device_bytes()
                );
                assert_eq!(
                    plan.qwen35_moe_capacity_limits().unwrap().unwrap(),
                    base_limits
                );
                assert_eq!(
                    plan.effective_options().scheduler_config,
                    base.effective_options().scheduler_config
                );
                let scheduler = plan.effective_options().scheduler_config;
                assert_eq!(scheduler.max_decode_batch, 2);
                assert_eq!(scheduler.decode_cohort_target, 1);
                assert_eq!(scheduler.decode_cohort_max_deferrals, 0);
                assert_eq!(
                    read.max_host_bytes - read.max_metadata_bytes - read.worker_stack_bytes as u64,
                    54 << 20
                );
                assert_eq!(
                    base_limits.device_bytes as u64 + read.max_host_bytes,
                    7_583_301_632
                );
                let report = execution_report(&plan, args.rank_timeout_ms);
                let expected_report = format!(
                    "expert read host staging: schedule={expected:?} host_cap_bytes=67108864 metadata_bytes=8388608 worker_stack_bytes=2097152; metadata and requested stack included in host cap; separate from CUDA cache/device budgets; OS/TLS/allocator headroom not included; no GPU-overlap claim"
                );
                assert_eq!(
                    report
                        .iter()
                        .filter(|line| line.starts_with("expert read host staging:"))
                        .collect::<Vec<_>>(),
                    vec![&expected_report]
                );
            }
        },
    );
}
