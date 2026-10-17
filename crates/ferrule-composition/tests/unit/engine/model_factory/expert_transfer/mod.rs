//! CPU-only admission/serialization tests; sparse headers never become an owner.
use super::mixed_serving::fixture;
use super::*;
use ferrule_model::transformer::{
    ExpertReadPipelineOptions, ExpertReadSchedule, ExpertTransferPolicy,
};

fn options() -> ModelFactoryOptions {
    let mut options = tests::options();
    options.max_layers = None;
    options.max_tensor_mebibytes = 1024;
    options.driver_config.ctx_size = 128;
    options.driver_config.enable_native_proposals = false;
    options.scheduler_config.max_batch_tokens = 8;
    options.scheduler_config.prefill_chunk_size = 8;
    options.scheduler_config.max_active_sequences = 2;
    options.scheduler_config.max_decode_batch = 1;
    options.scheduler_config.prefix_cache_capacity_pages = 0;
    options.kv_cache_mebibytes = Some(16);
    options.qwen35_moe_capacity = Some(Qwen35MoeCapacityLimits {
        max_experts: 8,
        max_bytes: 320 << 20,
        scratch_bytes: 8 << 20,
        workspace_bytes: 2 << 30,
        device_bytes: 7 << 30,
        ..Default::default()
    });
    options
}

fn serving(policy: Option<ServingExpertTransferPolicy>) -> ServingModelOptions {
    ServingModelOptions {
        expert_transfer_policy: policy,
        model_profile: Some(ServingModelProfile::Qwen35MixedBf16),
        expert_read_pipeline: Some(ExpertReadPipelineOptions {
            schedule: ExpertReadSchedule::Lookahead,
            max_host_bytes: 64 << 20,
            max_metadata_bytes: 8 << 20,
            worker_stack_bytes: 2 << 20,
        }),
        ..Default::default()
    }
}

#[test]
fn expert_transfer_serde_is_explicit_and_does_not_expose_retained() {
    for (policy, wire, leaf) in [
        (
            ServingExpertTransferPolicy::Synchronous,
            "\"synchronous\"",
            ExpertTransferPolicy::Synchronous,
        ),
        (
            ServingExpertTransferPolicy::GroupedSynchronous,
            "\"grouped_synchronous\"",
            ExpertTransferPolicy::GroupedSynchronous,
        ),
    ] {
        assert_eq!(serde_json::to_string(&policy).unwrap(), wire);
        assert_eq!(
            serde_json::from_str::<ServingExpertTransferPolicy>(wire).unwrap(),
            policy
        );
        assert_eq!(ExpertTransferPolicy::from(policy), leaf);
    }
    for wire in [
        "\"retained\"",
        "\"auto\"",
        "\"grouped-synchronous\"",
        "{}",
        "null",
    ] {
        assert!(serde_json::from_str::<ServingExpertTransferPolicy>(wire).is_err());
    }
    assert_eq!(
        ServingExpertTransferPolicy::default(),
        ServingExpertTransferPolicy::Synchronous
    );
    assert_eq!(ServingModelOptions::default().expert_transfer_policy, None);
}

#[test]
fn expert_transfer_omission_keeps_legacy_cpu_construction_without_io() {
    let mut descriptor = tests::descriptor(ModelFamily::Qwen35, "not-opened", 24);
    descriptor.path = "missing-transfer-checkpoint".into();
    let config = AutoConfig::from_descriptor(descriptor);
    let plan = ResidentModelPlanner
        .prepare_serving(
            &config,
            BackendSelection::Cpu,
            None,
            tests::options(),
            ServingModelOptions::default(),
        )
        .unwrap();
    assert_eq!(plan.requested_expert_transfer_policy(), None);
    assert_eq!(
        plan.expert_transfer_policy(),
        ServingExpertTransferPolicy::Synchronous
    );
    assert_eq!(plan.expert_read_pipeline_options(), None);
    assert!(
        plan.resolution_report()
            .contains("expert_transfer_policy=synchronous")
    );
    for policy in [
        ServingExpertTransferPolicy::Synchronous,
        ServingExpertTransferPolicy::GroupedSynchronous,
    ] {
        let error = ResidentModelPlanner
            .prepare_serving(
                &config,
                BackendSelection::Cpu,
                None,
                tests::options(),
                serving(Some(policy)),
            )
            .err()
            .expect("explicit unsupported policy must not fall back");
        assert!(
            error.to_string().contains("expert transfer policy"),
            "{error}"
        );
        assert!(!error.to_string().contains("No such file"), "{error}");
    }
}

#[test]
fn expert_transfer_rejects_backend_profile_topology_and_missing_reads_before_io() {
    let policy = Some(ServingExpertTransferPolicy::GroupedSynchronous);
    for (backend, request) in [
        (BackendSelection::Cpu, serving(policy)),
        (BackendSelection::Auto, serving(policy)),
        (
            BackendSelection::Cuda,
            ServingModelOptions {
                model_profile: None,
                ..serving(policy)
            },
        ),
        (
            BackendSelection::Cuda,
            ServingModelOptions {
                expert_read_pipeline: None,
                ..serving(policy)
            },
        ),
        (
            BackendSelection::Cuda,
            ServingModelOptions {
                expert_read_pipeline: Some(ExpertReadPipelineOptions {
                    schedule: ExpertReadSchedule::Blocking,
                    ..serving(policy).expert_read_pipeline.unwrap()
                }),
                ..serving(policy)
            },
        ),
        (
            BackendSelection::Cuda,
            ServingModelOptions {
                engine: ServingEngineSelection::Pipeline,
                ..serving(policy)
            },
        ),
        (
            BackendSelection::Cuda,
            ServingModelOptions {
                parallel: Some(PipelineBuildOptions::default()),
                ..serving(policy)
            },
        ),
        (
            BackendSelection::Cuda,
            ServingModelOptions {
                kv_execution_policy: ferrule_runtime::engine::KvExecutionPolicy::HostKvDecode(
                    Default::default(),
                ),
                ..serving(policy)
            },
        ),
    ] {
        let error = request
            .validate_before_model_io(backend, &options(), &Default::default())
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("expert transfer policy grouped_synchronous"),
            "{error}"
        );
    }
}

#[test]
fn expert_transfer_unsupported_provider_rejects_before_resolver_or_factory_calls() {
    fn resolve(_: &ModelDescriptor, _: BackendSelection) -> Result<ResolvedModelBackend> {
        panic!("unsupported transfer must reject before resolver/source inspection")
    }
    fn build(_: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
        panic!("unsupported transfer must reject before owner mutation")
    }
    fn select(
        entry: &ModelImplementation,
        _: BackendSelection,
        _: &SelectionRequest,
        _: &ModelFactoryOptions,
    ) -> Result<ResolvedStrategy> {
        let mut execution = ResolvedExecution::new(
            ModelExecutionBackend::Cuda,
            ResolvedConstruction::Resident(entry.build),
            StorageEncoding::NumericFp8E4m3fnBf16Scales,
            ArithmeticPrecision::Bf16Compatibility,
            "no transfer capability",
        );
        execution.numeric_bf16_precision =
            Some(ferrule_model::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate);
        Ok(ResolvedStrategy {
            execution,
            backend: ModelExecutionBackend::Cuda,
            catalog_backend: BackendSelection::Cuda,
            resolve: entry.resolve,
            policy: &RESIDENT_OPTIONS,
            parallel: None,
        })
    }
    let entry = ModelImplementation {
        family: ModelFamily::Unknown("no-transfer".into()),
        resolve,
        build,
        select,
    };
    let mut descriptor = tests::descriptor(entry.family.clone(), "no-transfer", 2);
    descriptor.path = "missing-transfer-checkpoint".into();
    for policy in [
        ServingExpertTransferPolicy::Synchronous,
        ServingExpertTransferPolicy::GroupedSynchronous,
    ] {
        let error = ResidentModelPlanner
            .prepare_catalog_serving(
                &entry,
                &AutoConfig::from_descriptor(descriptor.clone()),
                BackendSelection::Cuda,
                None,
                options(),
                SelectionRequest {
                    serving: serving(Some(policy)),
                    ..Default::default()
                },
            )
            .err()
            .expect("provider has not declared transfer capability");
        assert!(error.to_string().contains("selected provider"), "{error}");
    }
}

#[test]
fn expert_transfer_both_schedules_freeze_public_policy_without_budget_changes() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::expert_transfer_both_schedules_freeze_public_policy_without_budget_changes"
        ),
        || {
            let fixture = fixture::Fixture::new();
            let config = AutoConfig::from_pretrained(fixture.dir.path()).unwrap();
            for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
                let prepare = |policy| {
                    let mut request = serving(policy);
                    request.expert_read_pipeline.as_mut().unwrap().schedule = schedule;
                    ResidentModelPlanner.prepare_serving(
                        &config,
                        BackendSelection::Cuda,
                        None,
                        options(),
                        request,
                    )
                };
                let default = prepare(None).unwrap();
                let baseline = default.numeric_bf16_memory_plan().unwrap().unwrap();
                let automatic = match schedule {
                    ExpertReadSchedule::Blocking => ServingExpertTransferPolicy::Synchronous,
                    ExpertReadSchedule::Lookahead => {
                        ServingExpertTransferPolicy::GroupedSynchronous
                    }
                };
                assert_eq!(default.requested_expert_transfer_policy(), None);
                assert_eq!(default.expert_transfer_policy(), automatic);
                for requested in [
                    None,
                    Some(ServingExpertTransferPolicy::Synchronous),
                    Some(ServingExpertTransferPolicy::GroupedSynchronous),
                ] {
                    if schedule == ExpertReadSchedule::Blocking
                        && requested == Some(ServingExpertTransferPolicy::GroupedSynchronous)
                    {
                        let error = prepare(requested).err().unwrap();
                        assert!(error.to_string().contains("Lookahead"), "{error}");
                        continue;
                    }
                    let policy = requested.unwrap_or(automatic);
                    let plan = prepare(requested).unwrap();
                    let admitted = plan.numeric_bf16_memory_plan().unwrap().unwrap();
                    assert_eq!(plan.requested_expert_transfer_policy(), requested);
                    assert_eq!(plan.expert_transfer_policy(), policy);
                    assert!(
                        plan.resolution_report()
                            .contains(&format!("expert_transfer_policy={}", policy.as_str()))
                    );
                    assert_eq!(plan.backend_profile(), default.backend_profile());
                    assert_eq!(
                        plan.selected_implementation().kind,
                        default.selected_implementation().kind
                    );
                    assert_eq!(
                        plan.qwen35_moe_capacity_limits().unwrap(),
                        default.qwen35_moe_capacity_limits().unwrap()
                    );
                    assert_eq!(admitted.model_estimate(), baseline.model_estimate());
                    assert_eq!(admitted.model_budget(), baseline.model_budget());
                    assert_eq!(admitted.total_device_bytes(), baseline.total_device_bytes());
                    assert_eq!(
                        admitted.expert_read_pipeline_options(),
                        baseline.expert_read_pipeline_options()
                    );
                    assert_eq!(
                        same_final_request(&default.request, &plan.request),
                        requested.is_none()
                    );
                    let mut crossed = prepare(requested).unwrap();
                    if requested.is_some() {
                        crossed.request.execution.expert_transfer_policy = None;
                    } else {
                        crossed
                            .request
                            .execution
                            .expert_read_pipeline
                            .as_mut()
                            .unwrap()
                            .schedule = match schedule {
                            ExpertReadSchedule::Blocking => ExpertReadSchedule::Lookahead,
                            ExpertReadSchedule::Lookahead => ExpertReadSchedule::Blocking,
                        };
                        assert_ne!(crossed.expert_transfer_policy(), policy);
                    }
                    let called = std::cell::Cell::new(false);
                    assert!(
                        crossed
                            .build_with(|_| {
                                called.set(true);
                                Ok(())
                            })
                            .is_err()
                    );
                    assert!(
                        !called.get(),
                        "frozen source policy must reject before factory invocation"
                    );
                    plan.build_with(|request| {
                        assert_eq!(request.resolved.execution.expert_transfer_policy, requested);
                        assert_eq!(
                            expert_read_policy::effective_transfer_policy(
                                &request.resolved.execution
                            ),
                            policy
                        );
                        let expected: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> =
                            bf16_serving::build_numeric_bf16;
                        assert!(std::ptr::fn_addr_eq(
                            request.resolved.execution.construction.builder(),
                            expected
                        ));
                        assert_eq!(
                            request
                                .resolved
                                .execution
                                .expert_read_pipeline
                                .unwrap()
                                .max_host_bytes,
                            64 << 20
                        );
                        Ok(())
                    })
                    .unwrap();
                }
            }
        },
    );
}

#[test]
fn expert_transfer_automatic_selection_requires_the_complete_grouped_capability() {
    use ferrule_model::transformer::NumericFp8Precision;
    fn build(_: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
        panic!("metadata-only transfer selection must not construct an owner")
    }
    let mut execution = ResolvedExecution::new(
        ModelExecutionBackend::Cuda,
        ResolvedConstruction::Resident(build),
        StorageEncoding::NumericFp8E4m3fnBf16Scales,
        ArithmeticPrecision::Bf16Compatibility,
        "local numeric transfer fixture",
    );
    execution.supports_expert_transfer = true;
    execution.numeric_bf16_precision = Some(NumericFp8Precision::Bf16ModelMixedF32Accumulate);
    execution.expert_read_pipeline = serving(None).expert_read_pipeline;
    assert_eq!(
        expert_read_policy::effective_transfer_policy(&execution),
        ServingExpertTransferPolicy::GroupedSynchronous
    );
    let restrictions: &[fn(&mut ResolvedExecution)] = &[
        |e| e.supports_expert_transfer = false,
        |e| e.support.required_feature = None,
        |e| e.support.storage = StorageEncoding::NativeBf16,
        |e| e.numeric_bf16_precision = None,
        |e| e.numeric_bf16_precision = Some(NumericFp8Precision::Bf16ModelF32Accumulate),
        |e| e.numeric_bf16_precision = Some(NumericFp8Precision::Bf16ModelTiledF32Accumulate),
        |e| e.construction = ResolvedConstruction::Pipeline(build),
        |e| {
            e.kv_execution_policy =
                ferrule_runtime::engine::KvExecutionPolicy::HostKvDecode(Default::default())
        },
        |e| e.expert_read_pipeline = None,
        |e| e.expert_read_pipeline.as_mut().unwrap().schedule = ExpertReadSchedule::Blocking,
    ];
    for restrict in restrictions {
        let mut unsupported = execution;
        restrict(&mut unsupported);
        assert_eq!(unsupported.expert_transfer_policy, None);
        assert_eq!(
            expert_read_policy::effective_transfer_policy(&unsupported),
            ServingExpertTransferPolicy::Synchronous
        );
        expert_read_policy::validate_transfer_execution(&unsupported).unwrap();
        unsupported.expert_transfer_policy = Some(ServingExpertTransferPolicy::GroupedSynchronous);
        assert!(expert_read_policy::validate_transfer_execution(&unsupported).is_err());
    }
}

#[test]
fn expert_transfer_synchronous_does_not_require_a_read_worker() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::expert_transfer_synchronous_does_not_require_a_read_worker"
        ),
        || {
            let f = fixture::Fixture::new();
            let plan = ResidentModelPlanner
                .prepare_serving(
                    &AutoConfig::from_pretrained(f.dir.path()).unwrap(),
                    BackendSelection::Cuda,
                    None,
                    options(),
                    ServingModelOptions {
                        expert_read_pipeline: None,
                        ..serving(Some(ServingExpertTransferPolicy::Synchronous))
                    },
                )
                .unwrap();
            assert_eq!(
                plan.expert_transfer_policy(),
                ServingExpertTransferPolicy::Synchronous
            );
            assert!(plan.prepared_source_plan.is_none());
            assert_eq!(plan.expert_read_pipeline_options(), None);
        },
    );
}

#[test]
fn expert_transfer_unvalidated_numeric_precision_rejects_before_source_io() {
    use ferrule_model::transformer::NumericFp8Precision;
    let mut descriptor = tests::descriptor(
        ModelFamily::Qwen35Moe,
        "Qwen3_5MoeForConditionalGeneration",
        40,
    );
    descriptor.path = "missing-transfer-checkpoint".into();
    let config = AutoConfig::from_descriptor(descriptor);
    for precision in [
        NumericFp8Precision::Bf16ModelF32Accumulate,
        NumericFp8Precision::Bf16ModelTiledF32Accumulate,
    ] {
        let error = ResidentModelPlanner
            .prepare_numeric_fp8_bf16_serving_with_precision(
                &config,
                BackendSelection::Cuda,
                None,
                options(),
                ServingModelOptions {
                    model_profile: None,
                    model_precision: Some(ModelPrecision::Bf16),
                    ..serving(Some(ServingExpertTransferPolicy::GroupedSynchronous))
                },
                NumericStoragePolicy::E4M3FnBf16Scales,
                precision,
            )
            .err()
            .expect("grouped serving precision must be validated");
        assert!(
            error.to_string().contains("validated mixed BF16/F32"),
            "{error}"
        );
        assert!(!error.to_string().contains("No such file"), "{error}");
    }
}
