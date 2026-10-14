use super::tests::{descriptor, options};
use super::*;

fn parallel() -> PipelineBuildOptions {
    PipelineBuildOptions {
        parallelism: ferrule_common::ParallelismPlan {
            pipeline_parallel: 2,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn serving(parallel: Option<PipelineBuildOptions>) -> ServingModelOptions {
    ServingModelOptions {
        parallel,
        ..Default::default()
    }
}

#[test]
fn metadata_selection_and_precision_are_the_owner_build_inputs() {
    for (family, expected, arithmetic, expected_build) in [
        (
            ModelFamily::Qwen3,
            SelectedImplementationKind::Pipeline,
            ArithmeticPrecision::Bf16Compatibility,
            build_pipeline as fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine>,
        ),
        (
            ModelFamily::Qwen35,
            SelectedImplementationKind::Pipeline,
            ArithmeticPrecision::F32,
            build_hybrid_pipeline as fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine>,
        ),
    ] {
        // No checkpoint, child or device is needed to resolve this descriptor.
        let config = AutoConfig::from_descriptor(descriptor(family, "metadata-only", 24));
        let mut input = options();
        input.driver_config.ctx_size = 8;
        input.scheduler_config.max_batch_tokens = 512;
        let plan = ResidentModelPlanner
            .prepare_serving(
                &config,
                BackendSelection::Auto,
                None,
                input,
                serving(Some(parallel())),
            )
            .unwrap();
        let support = plan.support();
        let effective = plan.effective_options();
        assert_eq!(plan.selected_implementation().kind, expected);
        assert_eq!(support.arithmetic, arithmetic);
        assert_eq!(support.buildability, BuildabilityStatus::CatalogValidated);
        assert_eq!(
            support.admission,
            AdmissionBoundary::OwnerLiveAdmissionRequired
        );
        assert_eq!(
            plan.requested_options().scheduler_config.max_batch_tokens,
            512
        );
        assert_eq!(effective.scheduler_config.max_batch_tokens, 8);
        assert!(!effective.driver_config.enable_native_proposals);
        let precision = plan.request.execution.pipeline_precision();
        assert_eq!(
            precision == ferrule_model::execution::ExecutionPrecisionPolicy::f32(),
            arithmetic == ArithmeticPrecision::F32
        );
        let build = plan.request.execution.construction.builder();
        assert!(std::ptr::fn_addr_eq(build, expected_build));
        let error = plan
            .build_with(|request| -> Result<()> {
                assert_eq!(request.resolved.execution.construction.kind(), expected);
                assert_eq!(request.resolved.execution.support, support);
                assert_eq!(request.resolved.execution.pipeline_precision(), precision);
                assert_eq!(
                    request.resolved.scheduler_config,
                    effective.scheduler_config
                );
                Err(Error::RequestCapacity { limit: 7 })
            })
            .unwrap_err();
        assert!(matches!(error, Error::RequestCapacity { limit: 7 }));
    }
}

#[test]
fn legacy_with_pipeline_replaces_the_resident_selection_without_changing_request() {
    let config = AutoConfig::from_descriptor(descriptor(ModelFamily::Qwen35, "metadata", 24));
    let mut input = options();
    input.driver_config.enable_native_proposals = false;
    let resident = ResidentModelPlanner
        .prepare(&config, BackendSelection::Cpu, None, input)
        .unwrap();
    assert_eq!(
        resident.selected_implementation().kind,
        SelectedImplementationKind::Resident
    );
    let plan = resident.with_pipeline(parallel()).unwrap();
    assert_eq!(
        plan.selected_implementation().kind,
        SelectedImplementationKind::Pipeline
    );
    assert_eq!(plan.support().arithmetic, ArithmeticPrecision::F32);
    assert_eq!(plan.requested_options().max_tensor_mebibytes, 128);
    assert_eq!(plan.request.max_tensor_bytes, 128 << 20);
}

#[test]
fn serving_explicit_default_values_remain_constraints() {
    for family in [ModelFamily::Qwen3, ModelFamily::Qwen35] {
        for explicit in [0, 64, 256] {
            let config = AutoConfig::from_descriptor(descriptor(family.clone(), "metadata", 24));
            let mut request = serving(Some(parallel()));
            request.expert_host_cache_entries = Some(explicit);
            let error = ResidentModelPlanner
                .prepare_serving(&config, BackendSelection::Cpu, None, options(), request)
                .err()
                .unwrap();
            assert!(error.to_string().contains("--expert-host-cache-entries"));
        }
    }
    let config = AutoConfig::from_descriptor(descriptor(ModelFamily::Qwen35, "metadata", 24));
    let plan = ResidentModelPlanner
        .prepare_serving(
            &config,
            BackendSelection::Cpu,
            None,
            options(),
            serving(None),
        )
        .unwrap();
    assert_eq!(plan.effective_options().max_tensor_mebibytes, 1024);
    for explicit in [0, 16] {
        let mut request = serving(None);
        request.expert_pinned_cache_entries = Some(explicit);
        assert!(
            ResidentModelPlanner
                .prepare_serving(&config, BackendSelection::Cpu, None, options(), request,)
                .err()
                .unwrap()
                .to_string()
                .contains("explicit MoE")
        );
    }
    let mut request = serving(None);
    request.max_tensor_mebibytes = Some(128);
    let plan = ResidentModelPlanner
        .prepare_serving(&config, BackendSelection::Cpu, None, options(), request)
        .unwrap();
    assert_eq!(plan.effective_options().max_tensor_mebibytes, 128);
}

#[test]
fn unsupported_hybrid_compositions_and_resident_parallel_stay_rejected() {
    for family in [ModelFamily::Qwen3, ModelFamily::Qwen35] {
        let config = AutoConfig::from_descriptor(descriptor(family, "metadata", 24));
        let mut request = serving(Some(parallel()));
        request.engine = ServingEngineSelection::Resident;
        assert!(
            ResidentModelPlanner
                .prepare_serving(&config, BackendSelection::Cpu, None, options(), request,)
                .err()
                .unwrap()
                .to_string()
                .contains("parallel/rank options require")
        );
    }
    let config = AutoConfig::from_descriptor(descriptor(ModelFamily::Qwen35, "metadata", 24));
    for variant in 0..3 {
        let mut p = parallel();
        match variant {
            0 => p.parallelism.expert_parallel = 2,
            1 => p.parallelism.tensor_parallel = 2,
            _ => p.rank_backend = PipelineRankBackend::Process,
        }
        assert!(
            ResidentModelPlanner
                .prepare_serving(
                    &config,
                    BackendSelection::Cpu,
                    None,
                    options(),
                    serving(Some(p)),
                )
                .is_err()
        );
    }
}

#[test]
fn feature_support_is_not_a_live_admission_receipt() {
    let entry = implementation(&ModelFamily::Qwen35).unwrap();
    let resolved = (entry.select)(
        entry,
        BackendSelection::Cuda,
        &SelectionRequest::default(),
        &options(),
    )
    .unwrap()
    .execution;
    assert!(resolved.support.static_supported);
    assert_eq!(resolved.support.required_feature, Some("cuda"));
    assert_eq!(
        resolved.support.buildability,
        if cfg!(feature = "cuda") {
            BuildabilityStatus::CatalogValidated
        } else {
            BuildabilityStatus::FeatureUnavailable
        }
    );
    assert_eq!(
        resolved.support.admission,
        AdmissionBoundary::OwnerLiveAdmissionRequired
    );
    let config = AutoConfig::from_descriptor(descriptor(ModelFamily::Qwen35, "metadata", 24));
    let plan = ResidentModelPlanner.prepare_serving(
        &config,
        BackendSelection::Cuda,
        None,
        options(),
        serving(Some(parallel())),
    );
    assert_eq!(plan.is_ok(), cfg!(feature = "cuda"));
}

#[cfg(unix)]
#[test]
fn serving_process_ep_stays_resident_and_preserves_requested_launch() {
    let mut d = descriptor(ModelFamily::Qwen35Moe, "metadata-only", 40);
    d.spec.moe.num_experts = Some(256);
    d.spec.moe.num_experts_per_tok = Some(8);
    let config = AutoConfig::from_descriptor(d);
    let p = PipelineBuildOptions {
        parallelism: ferrule_common::ParallelismPlan {
            expert_parallel: 2,
            ..Default::default()
        },
        rank_backend: PipelineRankBackend::Process,
        process_launch: Some(
            ferrule_runtime::parallel::process::ProcessLaunch::new("must-not-launch")
                .arg("__rank-worker")
                .env("REQUEST", "retained"),
        ),
        rank_timeout: std::time::Duration::from_millis(6789),
        devices: Some(vec![3, 1]),
        ..Default::default()
    };
    for engine in [
        ServingEngineSelection::Auto,
        ServingEngineSelection::Resident,
        ServingEngineSelection::Pipeline,
    ] {
        let mut request = serving(Some(p.clone()));
        request.engine = engine;
        let result = ResidentModelPlanner.prepare_serving(
            &config,
            BackendSelection::Auto,
            None,
            options(),
            request,
        );
        if !cfg!(feature = "cuda") {
            assert!(result.is_err());
            continue;
        }
        let plan = result.unwrap();
        assert!(plan.pipeline_options().is_none());
        let expected: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> =
            qwen35::build_experts;
        assert!(std::ptr::fn_addr_eq(
            plan.request.execution.construction.builder(),
            expected
        ));
        assert_eq!(
            plan.selected_implementation().kind,
            SelectedImplementationKind::Resident
        );
        assert_eq!(plan.support().arithmetic, ArithmeticPrecision::F32Tf32x3);
        assert!(plan.qwen35_moe_capacity_limits().unwrap().is_none());
        let requested = plan.requested_parallel_options().unwrap();
        assert_eq!(requested.rank_timeout, p.rank_timeout);
        assert_eq!(
            requested.process_launch.as_ref().unwrap().environment,
            p.process_launch.as_ref().unwrap().environment
        );
        plan.build_with(|request| {
            assert_eq!(
                request.resolved.execution.construction.kind(),
                SelectedImplementationKind::Resident
            );
            assert!(request.resolved.pipeline.is_none());
            Ok(())
        })
        .unwrap();
    }
    for variant in 0..2 {
        let mut unsupported = p.clone();
        if variant == 0 {
            unsupported.parallelism.pipeline_parallel = 2;
        } else {
            unsupported.parallelism.tensor_parallel = 2;
        }
        let error = ResidentModelPlanner
            .prepare_serving(
                &config,
                BackendSelection::Auto,
                None,
                options(),
                serving(Some(unsupported)),
            )
            .err()
            .unwrap();
        let Error::Backend {
            source: ferrule_common::Error::ModelSource { source },
        } = error
        else {
            panic!("expected typed unsupported");
        };
        assert_eq!(
            source
                .downcast_ref::<ferrule_model::transformer::UnsupportedOperator>()
                .unwrap()
                .operator,
            "qwen35_moe_expert_parallel"
        );
    }
}

#[test]
fn synthetic_catalog_adapter_reuses_pipeline_strategy_without_family_dispatch() {
    fn resolve(d: &ModelDescriptor, _: BackendSelection) -> Result<ResolvedModelBackend> {
        Ok(ResolvedModelBackend {
            family: d.spec.family.clone(),
            model_name: "synthetic-catalog-adapter",
            backend: ModelExecutionBackend::Cpu,
            backend_profile: "synthetic-resident",
            default_chat_template: ChatTemplate::Qwen3,
        })
    }
    fn must_not_build(_: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
        panic!("metadata selection must not construct an owner")
    }
    let family = ModelFamily::Unknown("synthetic-pipeline-strategy".into());
    let entry = ModelImplementation {
        family: family.clone(),
        resolve,
        build: must_not_build,
        select: pipeline::select_dense_strategy,
    };
    let config = AutoConfig::from_descriptor(descriptor(family.clone(), "no-checkpoint", 24));
    let parallel = parallel();
    let serving = serving(Some(parallel.clone()));
    let strategy = (entry.select)(
        &entry,
        BackendSelection::Auto,
        &SelectionRequest::parallel(Some(parallel.clone())),
        &options(),
    )
    .unwrap();
    let options = strategy
        .resolve_options(&family, &serving, options())
        .unwrap();
    let plan = ResidentModelPlanner
        .prepare_strategy(
            &config,
            BackendSelection::Auto,
            None,
            options,
            Some(parallel),
            strategy,
        )
        .unwrap();
    assert_eq!(
        plan.selected_implementation().kind,
        SelectedImplementationKind::Pipeline
    );
    let expected: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> = build_pipeline;
    assert!(std::ptr::fn_addr_eq(
        plan.request.execution.construction.builder(),
        expected
    ));
    assert_eq!(
        plan.support().arithmetic,
        ArithmeticPrecision::Bf16Compatibility
    );
    assert_eq!(
        plan.effective_options().scheduler_config.max_decode_batch,
        1
    );
    assert!(
        !plan
            .effective_options()
            .driver_config
            .enable_native_proposals
    );
    assert_eq!(plan.request.family, family);
    plan.build_with(|request| {
        assert_eq!(
            request.resolved.execution.construction.kind(),
            SelectedImplementationKind::Pipeline
        );
        Ok(())
    })
    .unwrap();
}

#[test]
fn resident_numeric_constructor_and_arithmetic_are_frozen_independently_of_kind() {
    let entry = implementation(&ModelFamily::Qwen35Moe).unwrap();
    for (profile, arithmetic) in [
        (NumericFp8Profile::F32, ArithmeticPrecision::F32Tf32x3),
        (
            NumericFp8Profile::W8A16Bf16,
            ArithmeticPrecision::W8A16Bf16F32Accumulate,
        ),
    ] {
        let mut options = options();
        options.numeric_fp8_profile = Some(profile);
        let local = (entry.select)(
            entry,
            BackendSelection::Cuda,
            &SelectionRequest::default(),
            &options,
        )
        .unwrap();
        assert_eq!(
            local.execution.construction.kind(),
            SelectedImplementationKind::Resident
        );
        assert_eq!(local.execution.support.arithmetic, arithmetic);
        let local_builder: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> =
            qwen35::build;
        assert!(std::ptr::fn_addr_eq(
            local.execution.construction.builder(),
            local_builder
        ));
        let parallel = PipelineBuildOptions {
            parallelism: ferrule_common::ParallelismPlan {
                expert_parallel: 2,
                ..Default::default()
            },
            devices: Some(vec![3, 1]),
            ..Default::default()
        };
        let ep = (entry.select)(
            entry,
            BackendSelection::Cuda,
            &SelectionRequest::parallel(Some(parallel.clone())),
            &options,
        )
        .unwrap();
        assert_eq!(
            ep.execution.construction.kind(),
            SelectedImplementationKind::Resident
        );
        assert_eq!(ep.execution.support.arithmetic, arithmetic);
        let ep_builder: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> =
            qwen35::build_experts;
        assert!(std::ptr::fn_addr_eq(
            ep.execution.construction.builder(),
            ep_builder
        ));
    }
}

#[test]
fn explicit_model_precision_is_not_ignored_by_selected_strategy() {
    let config = AutoConfig::from_descriptor(descriptor(ModelFamily::Qwen3, "metadata", 24));
    for parallel in [None, Some(parallel())] {
        let error = ResidentModelPlanner
            .prepare_serving(
                &config,
                BackendSelection::Cpu,
                None,
                options(),
                ServingModelOptions {
                    model_precision: Some(ModelPrecision::F32),
                    parallel,
                    ..Default::default()
                },
            )
            .err()
            .expect("legacy BF16 compatibility is not explicit F32");
        assert!(error.to_string().contains("explicit F32"));
    }
    let config = AutoConfig::from_descriptor(descriptor(ModelFamily::Qwen35, "metadata", 24));
    let plan = ResidentModelPlanner
        .prepare_serving(
            &config,
            BackendSelection::Cpu,
            None,
            options(),
            ServingModelOptions {
                model_precision: Some(ModelPrecision::F32),
                parallel: Some(parallel()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(plan.support().arithmetic, ArithmeticPrecision::F32);
    assert_eq!(
        plan.selected_implementation().kind,
        SelectedImplementationKind::Pipeline
    );
}

#[test]
fn synthetic_adapter_builder_survives_normalization_and_is_invoked_by_build() {
    fn resolve(d: &ModelDescriptor, selection: BackendSelection) -> Result<ResolvedModelBackend> {
        assert_eq!(selection, BackendSelection::Cpu);
        Ok(ResolvedModelBackend {
            family: d.spec.family.clone(),
            model_name: "synthetic-sentinel",
            backend: ModelExecutionBackend::Cpu,
            backend_profile: "synthetic-resident",
            default_chat_template: ChatTemplate::Qwen3,
        })
    }
    fn sentinel(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
        assert_eq!(request.resolved.backend, ModelExecutionBackend::Cpu);
        assert_eq!(
            request.resolved.execution.construction.kind(),
            SelectedImplementationKind::Pipeline
        );
        assert_eq!(request.resolved.scheduler_config.max_decode_batch, 1);
        assert_eq!(request.resolved.scheduler_config.max_batch_tokens, 8);
        assert_eq!(request.resolved.scheduler_config.prefill_chunk_size, 8);
        assert!(!request.resolved.scheduler_config.allow_mixed_batches);
        assert!(!request.resolved.driver_config.enable_native_proposals);
        assert_eq!(
            request
                .resolved
                .pipeline
                .unwrap()
                .parallelism
                .pipeline_parallel,
            2
        );
        Err(Error::RequestCapacity { limit: 731 })
    }
    fn select(
        entry: &ModelImplementation,
        backend: BackendSelection,
        request: &SelectionRequest,
        options: &ModelFactoryOptions,
    ) -> Result<ResolvedStrategy> {
        let mut strategy = pipeline::select_dense_strategy(entry, backend, request, options)?;
        strategy.execution.construction = ResolvedConstruction::Pipeline(entry.build);
        Ok(strategy)
    }
    let family = ModelFamily::Unknown("synthetic-builder-strategy".into());
    let entry = ModelImplementation {
        family: family.clone(),
        resolve,
        build: sentinel,
        select,
    };
    let config = AutoConfig::from_descriptor(descriptor(family.clone(), "absent-checkpoint", 24));
    let parallel = parallel();
    let mut input = options();
    input.driver_config.ctx_size = 8;
    input.scheduler_config.max_batch_tokens = 512;
    let strategy = (entry.select)(
        &entry,
        BackendSelection::Auto,
        &SelectionRequest::parallel(Some(parallel.clone())),
        &input,
    )
    .unwrap();
    let expected: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> = sentinel;
    assert!(std::ptr::fn_addr_eq(
        strategy.execution.construction.builder(),
        expected
    ));
    assert_eq!(
        strategy.execution.construction.kind(),
        SelectedImplementationKind::Pipeline
    );
    let input = strategy
        .resolve_options(&family, &serving(Some(parallel.clone())), input)
        .unwrap();
    let plan = ResidentModelPlanner
        .prepare_strategy(
            &config,
            BackendSelection::Auto,
            None,
            input,
            Some(parallel),
            strategy,
        )
        .unwrap();
    assert!(std::ptr::fn_addr_eq(
        plan.request.execution.construction.builder(),
        expected
    ));
    assert_eq!(
        plan.selected_implementation().kind,
        SelectedImplementationKind::Pipeline
    );
    assert_eq!(
        plan.support().arithmetic,
        ArithmeticPrecision::Bf16Compatibility
    );
    assert_eq!(
        plan.requested_options().scheduler_config.max_batch_tokens,
        512
    );
    assert_eq!(
        plan.effective_options().scheduler_config.max_batch_tokens,
        8
    );
    // Exercise the actual frozen builder, not build_with's test injection seam.
    let error = plan.build().err().expect("sentinel must be invoked");
    assert!(matches!(error, Error::RequestCapacity { limit: 731 }));
}

#[cfg(feature = "cuda")]
#[test]
fn w8_pipeline_attachment_preserves_frozen_arithmetic_for_pp_and_tp() {
    let mut d = descriptor(ModelFamily::Qwen35Moe, "metadata-only", 40);
    d.spec.moe.num_experts = Some(256);
    d.spec.moe.num_experts_per_tok = Some(8);
    let config = AutoConfig::from_descriptor(d);
    for (pp, tp, ep) in [(2, 1, 2), (1, 2, 2), (2, 2, 2)] {
        let parallel = PipelineBuildOptions {
            parallelism: ferrule_common::ParallelismPlan {
                pipeline_parallel: pp,
                tensor_parallel: tp,
                expert_parallel: ep,
                ..Default::default()
            },
            devices: Some((0..pp * (tp + ep)).collect()),
            ..Default::default()
        };
        let mut input = options();
        input.numeric_fp8_profile = Some(NumericFp8Profile::W8A16Bf16);
        let plan = ResidentModelPlanner
            .prepare_serving(
                &config,
                BackendSelection::Cuda,
                None,
                input,
                ServingModelOptions {
                    parallel: Some(parallel),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            plan.selected_implementation().kind,
            SelectedImplementationKind::Pipeline
        );
        assert_eq!(
            plan.support().arithmetic,
            ArithmeticPrecision::W8A16Bf16F32Accumulate
        );
        assert_eq!(
            plan.numeric_fp8_profile(),
            Some(NumericFp8Profile::W8A16Bf16)
        );
        assert_eq!(
            plan.request.execution.pipeline_precision(),
            ferrule_model::execution::ExecutionPrecisionPolicy::f32()
        );
        let expected: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> =
            build_hybrid_pipeline;
        assert!(std::ptr::fn_addr_eq(
            plan.request.execution.construction.builder(),
            expected
        ));
        assert!(plan.qwen35_expert_placement().is_none());
        assert!(
            plan.support()
                .static_constraints
                .contains("quality-unaccepted")
        );
        let topology = plan.topology_placement();
        assert_eq!(topology.logical.pipeline_parallel, pp);
        assert_eq!(topology.logical.tensor_parallel, tp);
        assert_eq!(topology.logical.expert_parallel, ep);
        assert_eq!(topology.physical.devices.unwrap().len(), pp * (tp + ep));
        assert_eq!(
            plan.pipeline_options()
                .unwrap()
                .parallelism
                .pipeline_parallel,
            pp
        );
    }
}
