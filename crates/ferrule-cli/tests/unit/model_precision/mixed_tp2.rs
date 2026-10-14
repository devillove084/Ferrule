use super::mixed::fixture;
use super::*;
use ferrule_composition::model_factory::ServingModelProfile;

fn args() -> ServeArgs {
    serve(&[
        "ferrule",
        "serve",
        "must-not-open-checkpoint",
        "--model-profile",
        "qwen35-mixed-bf16",
        "--tensor-parallel",
        "2",
        "--devices",
        "3,1",
        "--ctx-size",
        "128",
        "--max-active-sequences",
        "2",
        "--max-batch-tokens",
        "8",
        "--prefill-chunk-size",
        "8",
    ])
}

#[test]
fn mixed_tp2_cli_accepts_authorized_catalog_plan_without_starting_server() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_tp2_cli_accepts_authorized_catalog_plan_without_starting_server"
        ),
        || {
            let args = args();
            let serving = serving_options(&args).unwrap();
            assert_eq!(
                serving.model_profile,
                Some(ServingModelProfile::Qwen35MixedBf16)
            );
            assert_eq!(serving.parallel.unwrap().devices, Some(vec![3, 1]));
            let options = model_factory_options(&args).unwrap();
            assert_eq!(options.qwen35_moe_capacity.unwrap().device_bytes, 22 << 30);
            assert!(options.qwen35_host_cache.is_none());
            assert!(options.numeric_fp8_profile.is_none());
            let f = fixture::Fixture::new();
            let mut args = args;
            args.model = f.dir.path().to_string_lossy().into_owned();
            let plan = prepare_model(&args).unwrap();
            assert_eq!(
                plan.selected_implementation().kind,
                ferrule_composition::model_factory::SelectedImplementationKind::Pipeline
            );
            assert_eq!(
                plan.backend_profile(),
                "cuda-qwen35-mixed-bf16-tp2-local-shards-serial"
            );
            assert_eq!(plan.topology_placement().physical.devices, Some(vec![3, 1]));
            assert!(plan.qwen35_host_cache_options().is_none());
            assert_eq!(
                plan.qwen35_moe_capacity_limits()
                    .unwrap()
                    .unwrap()
                    .workspace_bytes,
                4 << 30
            );
            let cache = plan.mixed_tp2_cache_requirements().unwrap().unwrap();
            assert_eq!(cache.len(), 2);
            for owner in cache {
                assert!(owner.cache_enabled);
                assert_eq!(owner.cache_capacity_grants, 3 << 30);
                assert_eq!(owner.workspace_ceiling_bytes, 4 << 30);
                assert!(owner.required_workspace_bytes > 3 << 30);
                assert!(owner.required_workspace_bytes <= owner.workspace_ceiling_bytes as u64);
                assert!(owner.required_device_bytes <= 22 << 30);
            }
            assert_eq!(
                plan.qwen35_moe_capacity_limits()
                    .unwrap()
                    .unwrap()
                    .device_bytes,
                22 << 30
            );
            let help = Cli::try_parse_from(["ferrule", "serve", "--help"])
                .err()
                .unwrap()
                .to_string();
            assert!(help.contains("authorized full40 multistep/lifecycle gate"));
            assert!(help.contains("PER CARD"));
        },
    );
}

#[test]
fn mixed_tp2_cli_rejects_unvalidated_shapes_and_unused_tp1_flags() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_tp2_cli_rejects_unvalidated_shapes_and_unused_tp1_flags"
        ),
        || {
            let mutations: &[fn(&mut ServeArgs)] = &[
                |a| a.pipeline_parallel = 2,
                |a| a.tensor_parallel = 4,
                |a| a.expert_parallel = 2,
                |a| a.rank_backend = RankBackend::Process,
                |a| a.devices = None,
                |a| a.devices = Some(vec![1, 1]),
                |a| a.rank_restarts = 1,
                |a| a.engine = ServeEngine::Resident,
                |a| a.cuda_device_budget_mb = Some(22529),
                |a| a.cuda_workspace_mb = Some(0),
                |a| a.cuda_workspace_mb = Some(1024),
                |a| a.cuda_workspace_mb = Some(3072),
                |a| a.ctx_size = 129,
                |a| a.max_batch_tokens = 9,
                |a| a.max_active_sequences = 3,
                |a| a.cuda_expert_device.entries = Some(1024),
                |a| a.cuda_expert_device.bytes = Some(4 << 30),
                |a| a.cuda_expert_scratch_mb = Some(64),
                |a| a.expert_prewarm.mode = Some(crate::args::ExpertPrewarmArg::Lazy),
                |a| a.expert_prewarm.mode = Some(crate::args::ExpertPrewarmArg::Full),
                |a| a.cuda_expert_device.numeric_precision = Some("f32".into()),
                |a| a.cuda_expert_device.numeric_precision = Some("w8a16-bf16".into()),
            ];
            for mutate in mutations {
                let mut a = args();
                mutate(&mut a);
                let error = prepare_model(&a)
                    .err()
                    .expect("reject before I/O")
                    .to_string();
                assert!(!error.contains("must-not-open-checkpoint"), "{error}");
                assert!(
                    !error.contains("authorized full40"),
                    "must fail its policy first: {error}"
                );
            }
            let mut a = args();
            a.cuda_device_budget_mb = Some(21504);
            assert_eq!(
                model_factory_options(&a)
                    .unwrap()
                    .qwen35_moe_capacity
                    .unwrap()
                    .device_bytes,
                21 << 30
            );
            validate_args(&a).unwrap();
            let error = prepare_model(&a).err().unwrap().to_string();
            assert!(
                error.contains("must-not-open-checkpoint"),
                "valid policy must reach model I/O: {error}"
            );
        },
    );
}

#[test]
fn mixed_tp2_cli_explicit_cache_workspace_is_preserved_and_low_budget_fails_before_io() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_tp2_cli_explicit_cache_workspace_is_preserved_and_low_budget_fails_before_io"
        ),
        || {
            let mut a = args();
            a.cuda_workspace_mb = Some(4096);
            assert_eq!(
                model_factory_options(&a)
                    .unwrap()
                    .qwen35_moe_capacity
                    .unwrap()
                    .workspace_bytes,
                4 << 30
            );
            let error = prepare_model(&a).err().unwrap().to_string();
            assert!(error.contains("must-not-open-checkpoint"), "{error}");
            a.cuda_workspace_mb = Some(1024);
            let error = prepare_model(&a).err().unwrap().to_string();
            assert!(
                error.contains("cache-on") && error.contains("4096"),
                "{error}"
            );
            assert!(!error.contains("must-not-open-checkpoint"), "{error}");
        },
    );
}
