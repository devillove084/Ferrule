use super::*;

#[cfg(unix)]
#[test]
fn process_ep_options_preserve_launch_and_reject_unfinished_combinations() {
    use ferrule_runtime::parallel::process::ProcessLaunch;
    use std::time::Duration;
    for degree in [2, 4, 8] {
        let valid = PipelineBuildOptions {
            parallelism: ferrule_common::ParallelismPlan {
                expert_parallel: degree,
                ..Default::default()
            },
            rank_backend: PipelineRankBackend::Process,
            process_launch: Some(
                ProcessLaunch::new("absent-child")
                    .arg("__rank-worker")
                    .env("EP_TEST", "preserved"),
            ),
            devices: Some((0..degree).rev().collect()),
            rank_timeout: Duration::from_millis(1234),
            ..Default::default()
        };
        let placement =
            Qwen35ExpertPlacement::from_options(&valid, BackendSelection::Auto).unwrap();
        assert_eq!(placement.rank_backend(), PipelineRankBackend::Process);
        assert_eq!(placement.rank_timeout, valid.rank_timeout);
        assert_eq!(placement.expert_devices(), valid.devices.as_ref().unwrap());
        assert_eq!(
            placement.process_launch.as_ref().unwrap().environment,
            valid.process_launch.as_ref().unwrap().environment
        );
        for variant in 0..12 {
            let mut bad = valid.clone();
            match variant {
                0 => bad.parallelism.pipeline_parallel = 2,
                1 => bad.parallelism.tensor_parallel = 2,
                2 => bad.parallelism.data_parallel = 2,
                3 => bad.rank_restarts = 1,
                4 => bad.process_launch = None,
                5 => bad.process_launch = Some(ProcessLaunch::new("")),
                6 => bad.rank_timeout = Duration::ZERO,
                7 => bad.rank_timeout = Duration::from_nanos(1),
                8 => bad.rank_timeout = Duration::MAX,
                9 => bad.rank_backend = PipelineRankBackend::Thread,
                10 => bad.parallelism.context_parallel = 2,
                _ => bad.parallelism.sequence_parallel = 2,
            }
            assert_ep_error(
                Qwen35ExpertPlacement::from_options(&bad, BackendSelection::Auto).unwrap_err(),
            );
        }
        assert_ep_error(
            Qwen35ExpertPlacement::from_options(&valid, BackendSelection::Cpu).unwrap_err(),
        );
    }
}

#[cfg(all(unix, feature = "cuda"))]
#[test]
fn process_ep_public_planner_forwards_real_options_to_resident_factory_without_payloads() {
    use ferrule_runtime::parallel::process::ProcessLaunch;
    for degree in [2, 4, 8] {
        let mut descriptor = crate::model_factory::tests::descriptor(
            ModelFamily::Qwen35Moe,
            "metadata-only-process-ep",
            40,
        );
        descriptor.spec.moe.num_experts = Some(256);
        descriptor.spec.moe.num_experts_per_tok = Some(8);
        let config = AutoConfig::from_descriptor(descriptor);
        let mut options = crate::model_factory::tests::options();
        options.max_tensor_mebibytes = 1024;
        options.driver_config.enable_native_proposals = false;
        let parallel = PipelineBuildOptions {
            parallelism: ferrule_common::ParallelismPlan {
                expert_parallel: degree,
                ..Default::default()
            },
            rank_backend: PipelineRankBackend::Process,
            process_launch: Some(
                ProcessLaunch::new("not-executed-metadata-only-child")
                    .arg("__rank-worker")
                    .arg("--max-frame-bytes")
                    .arg("8388608")
                    .env("EP_TEST", "kept"),
            ),
            devices: Some((0..degree).rev().collect()),
            rank_timeout: std::time::Duration::from_millis(6789),
            ..Default::default()
        };
        let plan = ResidentModelPlanner::new()
            .prepare_qwen35_expert_parallel(
                &config,
                BackendSelection::Auto,
                None,
                options,
                parallel.clone(),
            )
            .unwrap();
        assert!(plan.backend_profile().ends_with("qwen35-process-ep"));
        assert_eq!(
            plan.topology_placement().rank_backend,
            Some(PipelineRankBackend::Process)
        );
        assert_eq!(
            plan.requested_parallel_options().unwrap().rank_timeout,
            parallel.rank_timeout
        );
        assert_eq!(
            plan.selected_implementation().kind,
            SelectedImplementationKind::Resident
        );
        let expected: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> =
            qwen35::build_experts;
        assert!(std::ptr::fn_addr_eq(
            plan.request.execution.construction.builder(),
            expected
        ));
        assert!(
            plan.support()
                .static_constraints
                .contains("no TP/PP/restart")
        );
        assert!(plan.qwen35_moe_capacity_limits().unwrap().is_none());
        let expected = plan.qwen35_expert_placement().unwrap().clone();
        plan.build_with(|request| {
            assert!(request.resolved.pipeline.is_none());
            let placement = request
                .resolved
                .family_options
                .qwen35()
                .unwrap()
                .expert_placement()
                .unwrap();
            assert_eq!(placement, &expected);
            assert_eq!(
                placement.process_launch.as_ref().unwrap().args,
                parallel.process_launch.as_ref().unwrap().args
            );
            assert_eq!(
                placement.process_launch.as_ref().unwrap().environment,
                parallel.process_launch.as_ref().unwrap().environment
            );
            Ok(())
        })
        .unwrap();
    }
}
