//! CPU-only CLI/read-policy tests; no model payload or device owner.
use super::*;
use crate::args::ExpertReadArgs;
use ferrule_model::transformer::ExpertReadSchedule;

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
                || error.to_string().contains("CPU execution is test-only"),
            "{error}"
        );
    }
}
