use super::*;
use crate::model_factory::tests::{descriptor, options};

#[test]
fn host_and_idle_kv_constructor_and_policy_are_frozen_on_resident_engine() {
    let config =
        AutoConfig::from_descriptor(descriptor(ModelFamily::Qwen3, "absent-checkpoint", 24));
    let mut options = options();
    options.scheduler_config.max_active_sequences = 1;
    options.scheduler_config.max_decode_batch = 1;
    for policy in [
        KvExecutionPolicy::HostKvDecode(Default::default()),
        KvExecutionPolicy::IdleReclaim(ferrule_model::decoder::IdleKvReclaimConfig {
            staging_page_limit: 1,
            device_bytes: 1 << 20,
            host_bytes: 1 << 20,
            pinned_bytes: 1 << 20,
        }),
    ] {
        let plan = ResidentModelPlanner
            .prepare_serving(
                &config,
                BackendSelection::Cuda,
                None,
                options.clone(),
                ServingModelOptions {
                    kv_execution_policy: policy,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            plan.selected_implementation().kind,
            SelectedImplementationKind::Resident
        );
        assert_eq!(plan.support().arithmetic, ArithmeticPrecision::F32);
        assert_eq!(plan.request.execution.kv_execution_policy, policy);
        let expected: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> = build;
        assert!(std::ptr::fn_addr_eq(
            plan.request.execution.construction.builder(),
            expected
        ));
        assert!(plan.pipeline_options().is_none());

        // The read-only conversion gate retains both explicit policy and builder.
        let error = plan.validate_pipeline_conversion().unwrap_err();
        assert!(error.to_string().contains("explicit HostKV/IdleReclaim"));
        assert_eq!(plan.request.execution.kv_execution_policy, policy);
        assert!(std::ptr::fn_addr_eq(
            plan.request.execution.construction.builder(),
            expected
        ));
        // TP would inspect absent checkpoint metadata if conversion got that far.
        // This must reject identically with and without the CUDA feature.
        let error = plan
            .with_pipeline(PipelineBuildOptions {
                parallelism: ferrule_common::ParallelismPlan {
                    tensor_parallel: 2,
                    ..Default::default()
                },
                ..Default::default()
            })
            .err()
            .expect("must not discard the explicit KV execution policy");
        assert!(error.to_string().contains("explicit HostKV/IdleReclaim"));
    }
}
