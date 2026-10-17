//! Dedicated CLI construction tests; no tokenizer/payload reads or CUDA owner.
use super::model_precision_tests::fixture;
use super::*;
use ferrule_composition::ServingExpertTransferPolicy;

#[test]
fn expert_transfer_cli_defaults_and_enum_values_are_unambiguous() {
    let args = serve(&["ferrule", "serve", "not-opened"]);
    assert_eq!(args.expert_transfer_policy, None);
    assert_eq!(serving_options(&args).unwrap().expert_transfer_policy, None);
    for (flag, policy) in [
        ("synchronous", ServingExpertTransferPolicy::Synchronous),
        (
            "grouped-synchronous",
            ServingExpertTransferPolicy::GroupedSynchronous,
        ),
    ] {
        let args = serve(&[
            "ferrule",
            "serve",
            "not-opened",
            "--expert-transfer-policy",
            flag,
        ]);
        assert_eq!(
            serving_options(&args).unwrap().expert_transfer_policy,
            Some(policy)
        );
    }
    for value in ["retained", "auto", "grouped_synchronous", "async"] {
        assert!(
            Cli::try_parse_from([
                "ferrule",
                "serve",
                "not-opened",
                "--expert-transfer-policy",
                value
            ])
            .is_err()
        );
    }
    let help = Cli::try_parse_from(["ferrule", "serve", "--help"])
        .err()
        .unwrap()
        .to_string();
    assert!(help.contains("--expert-transfer-policy"));
    assert!(help.contains("grouped-synchronous"));
    assert!(help.contains("omitted selects grouped"));
    assert!(help.contains("--expert-read-schedule lookahead"));
}

#[test]
fn expert_transfer_cli_unsupported_selections_fail_before_model_path_io() {
    for policy in ["synchronous", "grouped-synchronous"] {
        for extra in [
            vec![],
            vec!["--model-precision", "bf16"],
            vec![
                "--model-profile",
                "qwen35-mixed-bf16",
                "--tensor-parallel",
                "2",
                "--expert-read-host-mb",
                "64",
            ],
            vec![
                "--model-profile",
                "qwen35-mixed-bf16",
                "--engine",
                "pipeline",
                "--expert-read-host-mb",
                "64",
            ],
            vec![
                "--model-profile",
                "qwen35-mixed-bf16",
                "--kv-execution-policy",
                "host-kv-decode",
                "--expert-read-host-mb",
                "64",
            ],
        ] {
            let mut argv = vec![
                "ferrule",
                "serve",
                "not-opened",
                "--expert-transfer-policy",
                policy,
                "--max-batch-tokens",
                "8",
            ];
            argv.extend(extra);
            let error = prepare_model(&serve(&argv))
                .err()
                .expect("unsupported before IO");
            assert!(
                error.to_string().contains("expert transfer policy"),
                "{error}"
            );
            assert!(!error.to_string().contains("No such file"), "{error}");
        }
    }
    for reads in [
        vec![],
        vec!["--expert-read-host-mb", "64"],
        vec![
            "--expert-read-host-mb",
            "64",
            "--expert-read-schedule",
            "blocking",
        ],
    ] {
        let mut argv = vec![
            "ferrule",
            "serve",
            "not-opened",
            "--expert-transfer-policy",
            "grouped-synchronous",
            "--model-profile",
            "qwen35-mixed-bf16",
            "--max-batch-tokens",
            "8",
        ];
        argv.extend(reads);
        let error = prepare_model(&serve(&argv)).err().unwrap();
        assert!(
            error.to_string().contains("bounded expert read pipeline"),
            "{error}"
        );
        assert!(error.to_string().contains("Lookahead"), "{error}");
        assert!(!error.to_string().contains("No such file"), "{error}");
    }
}

#[test]
fn expert_transfer_cli_public_plan_reports_effective_policy_and_preserves_caps() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::expert_transfer_cli_public_plan_reports_effective_policy_and_preserves_caps"
        ),
        || {
            let fixture = fixture::Fixture::new();
            for schedule in ["blocking", "lookahead"] {
                let base = vec![
                    "ferrule",
                    "serve",
                    fixture.dir.path().to_str().unwrap(),
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
                    "--expert-read-host-mb",
                    "64",
                    "--expert-read-schedule",
                    schedule,
                ];
                let default = prepare_model(&serve(&base)).unwrap();
                let automatic = if schedule == "lookahead" {
                    ServingExpertTransferPolicy::GroupedSynchronous
                } else {
                    ServingExpertTransferPolicy::Synchronous
                };
                assert_eq!(default.expert_transfer_policy(), automatic);
                assert_eq!(default.requested_expert_transfer_policy(), None);
                let report = execution_report(&default, 30_000).join("\n");
                assert!(
                    report.contains(&format!("requested=None effective={}", automatic.as_str())),
                    "{report}"
                );
                for (flag, expected) in [
                    ("synchronous", ServingExpertTransferPolicy::Synchronous),
                    (
                        "grouped-synchronous",
                        ServingExpertTransferPolicy::GroupedSynchronous,
                    ),
                ] {
                    let mut argv = base.clone();
                    argv.extend(["--expert-transfer-policy", flag]);
                    if schedule == "blocking"
                        && expected == ServingExpertTransferPolicy::GroupedSynchronous
                    {
                        let error = prepare_model(&serve(&argv)).err().unwrap();
                        assert!(error.to_string().contains("Lookahead"), "{error}");
                        continue;
                    }
                    let plan = prepare_model(&serve(&argv)).unwrap();
                    assert_eq!(plan.expert_transfer_policy(), expected);
                    assert_eq!(plan.requested_expert_transfer_policy(), Some(expected));
                    assert_eq!(plan.backend_profile(), default.backend_profile());
                    assert_eq!(
                        plan.qwen35_moe_capacity_limits().unwrap(),
                        default.qwen35_moe_capacity_limits().unwrap()
                    );
                    assert_eq!(
                        plan.expert_read_pipeline_options(),
                        default.expert_read_pipeline_options()
                    );
                    let report = execution_report(&plan, 30_000).join("\n");
                    assert!(
                        report.contains(&format!("effective={}", expected.as_str())),
                        "{report}"
                    );
                    assert!(report.contains("host_cap_bytes=67108864"), "{report}");
                    assert!(
                        report.contains("unchanged host staging and device/cache budgets"),
                        "{report}"
                    );
                }
            }
        },
    );
}
