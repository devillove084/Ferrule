use super::*;
use ferrule_composition::model_factory::{
    LogicalTopology, ModelTopologyPlacement, PhysicalPlacement,
};

pub(super) fn assert_pipeline_topology(
    plan: &ResidentModelBuildPlan,
    pp: usize,
    tp: usize,
    ep: usize,
    rank_backend: PipelineRankBackend,
    devices: Option<Vec<usize>>,
) {
    assert_eq!(
        plan.selected_implementation().kind,
        SelectedImplementationKind::Pipeline
    );
    assert_eq!(
        plan.topology_placement(),
        ModelTopologyPlacement {
            rank_backend: Some(rank_backend),
            logical: LogicalTopology {
                pipeline_parallel: pp,
                tensor_parallel: tp,
                expert_parallel: ep,
                data_parallel: 1,
                sequence_parallel: 1,
                context_parallel: 1,
            },
            physical: PhysicalPlacement {
                devices,
                root_device: None,
                expert_devices: None,
                source_identity: None,
            },
        }
    );
}

#[test]
fn cli_preserves_explicit_policy_without_selecting_a_family() {
    let args = serve(&[
        "ferrule",
        "serve",
        "not-opened",
        "--engine",
        "resident",
        "--expert-parallel",
        "2",
        "--devices",
        "3,1",
        "--expert-host-cache-entries",
        "256",
        "--expert-host-cache-mb",
        "0",
        "--expert-pinned-cache-entries",
        "64",
        "--moe-hotset-experts",
        "0",
        "--max-tensor-mb",
        "128",
    ]);
    // The same request can be dedicated resident EP or a rejected generic
    // resident topology; only runtime model resolution may decide that.
    validate_args(&args).unwrap();
    let request = serving_options(&args).unwrap();
    assert_eq!(request.engine, ServingEngineSelection::Resident);
    assert_eq!(request.parallel.unwrap().devices, Some(vec![3, 1]));
    assert_eq!(request.expert_host_cache_entries, Some(256));
    assert_eq!(request.expert_host_cache_mebibytes, Some(0));
    assert_eq!(request.expert_pinned_cache_entries, Some(64));
    assert_eq!(request.moe_hotset_experts, Some(0));
    assert_eq!(request.max_tensor_mebibytes, Some(128));
    let omitted = serving_options(&serve(&["ferrule", "serve", "not-opened"])).unwrap();
    assert!(omitted.parallel.is_none());
    assert!(omitted.max_tensor_mebibytes.is_none());
    assert!(omitted.expert_host_cache_entries.is_none());
    assert!(omitted.moe_hotset_experts.is_none());
}

#[cfg(feature = "cpu-reference")]
#[test]
fn generic_pipeline_defaults_and_rejections_are_owned_by_runtime() {
    let fixture = PlanFixture::new(false);
    let args = fixture.reference_args(&["--pipeline-parallel", "2"]);
    let raw = model_factory_options(&args).unwrap();
    assert_eq!(raw.expert_cache.host_entries, 64);
    let plan = prepare_model(&args).unwrap();
    assert_pipeline_topology(&plan, 2, 1, 1, PipelineRankBackend::Thread, None);
    let report = execution_report(&plan, args.rank_timeout_ms).join("\n");
    assert!(
        report.contains("pipeline PP=2 TP=1 EP=1 cpu-reference Thread owners"),
        "{report}"
    );
    assert!(report.contains("precision=Bf16Compatibility"), "{report}");
    assert!(!report.contains("numeric"), "{report}");
    assert!(!report.contains("CUDA ordinals"), "{report}");
    assert_eq!(
        plan.requested_options().expert_cache,
        ExpertCacheOptions::default()
    );
    assert_eq!(
        plan.support().arithmetic,
        ferrule_composition::model_factory::ArithmeticPrecision::Bf16Compatibility
    );
    let explicit = fixture.reference_args(&[
        "--pipeline-parallel",
        "2",
        "--expert-host-cache-entries",
        "256",
    ]);
    assert!(serving_options(&explicit).is_ok());
    assert!(
        prepare_model(&explicit)
            .err()
            .unwrap()
            .to_string()
            .contains("--expert-host-cache-entries")
    );
}

#[test]
fn numeric_profile_cli_reaches_factory_without_a_cache_override() {
    let args = serve(&[
        "ferrule",
        "serve",
        "not-opened",
        "--numeric-fp8-precision",
        "w8a16-bf16",
        "--expert-parallel",
        "2",
        "--devices",
        "0,1",
        "--rank-backend",
        "process",
    ]);
    let options = model_factory_options(&args).unwrap();
    assert_eq!(
        options.numeric_fp8_profile,
        Some(ferrule_composition::model_factory::NumericFp8Profile::W8A16Bf16)
    );
    assert!(options.qwen35_moe_capacity.is_none());
    let selection = serving_options(&args).unwrap();
    assert_eq!(
        selection.parallel.unwrap().rank_backend,
        PipelineRankBackend::Process
    );
}

#[test]
fn qwen35_pp_ep_cli_selects_real_hybrid_factory_and_preserves_profile() {
    use ferrule_composition::model_factory::{NumericFp8Profile, SelectedImplementationKind};
    let fixture = qwen35_moe_fixture();
    for (flag, profile) in [
        (None, NumericFp8Profile::F32),
        (Some("w8a16-bf16"), NumericFp8Profile::W8A16Bf16),
    ] {
        let mut flags = vec![
            "--backend",
            "cuda",
            "--engine",
            "pipeline",
            "--pipeline-parallel",
            "2",
            "--expert-parallel",
            "2",
            "--devices",
            "0,1,0,1,0,1",
        ];
        if let Some(flag) = flag {
            flags.extend(["--numeric-fp8-precision", flag]);
        }
        let result = prepare_model(&fixture.args(&flags));
        if cfg!(feature = "cuda") {
            let plan = result.unwrap();
            assert_eq!(
                plan.selected_implementation().kind,
                SelectedImplementationKind::Pipeline
            );
            assert_pipeline_topology(
                &plan,
                2,
                1,
                2,
                PipelineRankBackend::Thread,
                Some(vec![0, 1, 0, 1, 0, 1]),
            );
            let report = execution_report(&plan, 1234).join("\n");
            assert!(
                report.contains("pipeline PP=2 TP=1 EP=2 cuda Thread owners"),
                "{report}"
            );
            assert!(
                report.contains("CUDA ordinals (PP, then stage EP owners): [0, 1, 0, 1, 0, 1]"),
                "{report}"
            );
            assert!(
                report.contains("full shared host prewarm and all-owned experts"),
                "{report}"
            );
            assert!(
                report.contains("physical root+expert colocation admission; no process fallback"),
                "{report}"
            );
            assert!(
                report.contains(profile.arithmetic_description()),
                "{report}"
            );
            assert_eq!(
                report.contains("W8A16 opt-in: quality-unaccepted"),
                profile == NumericFp8Profile::W8A16Bf16
            );
            assert!(!report.contains("resident root"), "{report}");
            assert_eq!(plan.numeric_fp8_profile(), Some(profile));
            assert_eq!(
                plan.pipeline_options().unwrap().devices,
                Some(vec![0, 1, 0, 1, 0, 1])
            );
            assert!(plan.qwen35_expert_placement().is_none());
            assert!(plan.qwen35_moe_capacity_limits().unwrap().is_none());
            assert!(plan.backend_profile().contains("pp2-ep"));
            assert!(plan.support().static_constraints.contains("no process"));
            assert!(
                plan.resolution_report()
                    .contains(profile.arithmetic_description())
            );
        } else {
            assert!(result.is_err());
        }
        for extra in [
            vec!["--rank-backend", "process"],
            vec!["--tensor-parallel", "2"],
            vec!["--engine", "resident"],
            vec!["--cuda-expert-device-entries", "1024"],
        ] {
            // Replace rather than repeat clap's single-valued engine argument.
            let mut invalid = flags.clone();
            if extra[0] == "--engine" {
                invalid[3] = "resident";
            } else {
                invalid.extend(extra);
            }
            assert!(prepare_model(&fixture.args(&invalid)).is_err());
        }
    }
}

#[test]
fn qwen35_tp_ep_cli_routes_pp1_and_pp2_to_pipeline_with_single_stage_attachments() {
    use ferrule_composition::model_factory::{NumericFp8Profile, SelectedImplementationKind};
    let fixture = qwen35_moe_fixture();
    for (pp, tp, devices) in [
        ("1", "2", "0,1,0,1"),
        ("2", "2", "0,1,2,3,4,5,6,7"),
        ("2", "4", "0,1,2,3,4,5,6,7,0,1,2,3"),
    ] {
        for (flag, profile) in [
            (None, NumericFp8Profile::F32),
            (Some("w8a16-bf16"), NumericFp8Profile::W8A16Bf16),
        ] {
            let mut flags = vec![
                "--pipeline-parallel",
                pp,
                "--tensor-parallel",
                tp,
                "--expert-parallel",
                "2",
                "--devices",
                devices,
                "--expert-prewarm",
                "full",
            ];
            if let Some(flag) = flag {
                flags.extend(["--numeric-fp8-precision", flag]);
            }
            let result = prepare_model(&fixture.args(&flags));
            if cfg!(feature = "cuda") {
                let plan = result.unwrap();
                assert_eq!(
                    plan.selected_implementation().kind,
                    SelectedImplementationKind::Pipeline
                );
                let expected_devices = devices
                    .split(',')
                    .map(|v| v.parse().unwrap())
                    .collect::<Vec<_>>();
                assert_pipeline_topology(
                    &plan,
                    pp.parse().unwrap(),
                    tp.parse().unwrap(),
                    2,
                    PipelineRankBackend::Thread,
                    Some(expected_devices.clone()),
                );
                let report = execution_report(&plan, 1234).join("\n");
                assert!(
                    report.contains(&format!("pipeline PP={pp} TP={tp} EP=2 cuda Thread owners")),
                    "{report}"
                );
                assert!(report.contains(&format!("CUDA ordinals (PP-major TP roots, then stage EP children): {expected_devices:?}")), "{report}");
                assert!(
                    report.contains(profile.arithmetic_description()),
                    "{report}"
                );
                assert_eq!(
                    report.contains("W8A16 opt-in: quality-unaccepted"),
                    profile == NumericFp8Profile::W8A16Bf16
                );
                assert!(!report.contains("resident root"), "{report}");
                assert_eq!(plan.numeric_fp8_profile(), Some(profile));
                assert!(
                    plan.qwen35_expert_placement().is_none(),
                    "PP1 TP must never route to resident EP"
                );
                assert!(plan.qwen35_moe_capacity_limits().unwrap().is_none());
                assert_eq!(
                    plan.pipeline_options()
                        .unwrap()
                        .devices
                        .as_ref()
                        .unwrap()
                        .len(),
                    pp.parse::<usize>().unwrap() * (tp.parse::<usize>().unwrap() + 2)
                );
                assert!(plan.backend_profile().contains("tp-ep"));
                if profile == NumericFp8Profile::W8A16Bf16 {
                    assert!(plan.backend_profile().contains("quality-unaccepted"));
                }
                assert!(plan.support().static_constraints.contains("no process"));
            } else {
                assert!(result.is_err());
            }
            for extra in [
                vec!["--rank-backend", "process"],
                vec!["--engine", "resident"],
                vec!["--backend", "cpu"],
                vec!["--rank-restarts", "1"],
            ] {
                let mut invalid = flags.clone();
                invalid.extend(extra);
                assert!(prepare_model(&fixture.args(&invalid)).is_err());
            }
        }
    }
}

#[test]
fn qwen35_tp_ep_cli_preserves_explicit_2048_mib_tensor_budget() {
    let fixture = qwen35_moe_fixture();
    for pp in ["1", "2"] {
        let devices = if pp == "1" {
            "0,1,2,3"
        } else {
            "0,1,2,3,4,5,6,7"
        };
        for (budget, expected) in [(None, 1024), (Some("1024"), 1024), (Some("2048"), 2048)] {
            let mut flags = vec![
                "--engine",
                "pipeline",
                "--pipeline-parallel",
                pp,
                "--tensor-parallel",
                "2",
                "--expert-parallel",
                "2",
                "--devices",
                devices,
            ];
            if let Some(budget) = budget {
                flags.extend(["--max-tensor-mb", budget]);
            }
            let args = fixture.args(&flags);
            assert_eq!(
                serving_options(&args).unwrap().max_tensor_mebibytes,
                budget.map(|_| expected)
            );
            let result = prepare_model(&args);
            if cfg!(feature = "cuda") {
                assert_eq!(
                    result.unwrap().effective_options().max_tensor_mebibytes,
                    expected
                );
            } else {
                assert!(result.is_err());
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn ordinary_cuda_pipeline_reports_effective_implicit_and_explicit_owner_rosters() {
    let fixture = PlanFixture::tensor();
    for (devices, expected) in [(None, vec![0, 1]), (Some("3,1"), vec![3, 1])] {
        let mut flags = vec![
            "--backend",
            "cuda",
            "--tensor-parallel",
            "2",
            "--ctx-size",
            "64",
        ];
        if let Some(devices) = devices {
            flags.extend(["--devices", devices]);
        }
        let args = fixture.args(&flags);
        let plan = prepare_model(&args).unwrap();
        assert_pipeline_topology(
            &plan,
            1,
            2,
            1,
            PipelineRankBackend::Thread,
            Some(expected.clone()),
        );
        let report = execution_report(&plan, args.rank_timeout_ms).join("\n");
        assert!(
            report.contains("pipeline PP=1 TP=2 EP=1 cuda Thread owners, precision=F32"),
            "{report}"
        );
        assert!(
            report.contains(&format!(
                "CUDA ordinals (PP-major TP roots, then stage EP children): {expected:?}"
            )),
            "{report}"
        );
        assert!(!report.contains("numeric hybrid"), "{report}");
        assert!(!report.contains("resident root"), "{report}");
    }
}
