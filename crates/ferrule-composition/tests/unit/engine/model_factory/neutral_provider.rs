//! Bound semantic resources enter shared providers without catalog/family dispatch.
use super::tensor_tests::fixture::Fixture;
use super::*;
use super::{budget::*, experts::OwnerPlan, inputs::*, plan::Plan};
use crate::model_factory::*;
use ferrule_model::transformer::host_experts::{ExpertPrewarmMode, HostMemoryBudget};

fn fixture_request(f: &Fixture) -> (ResolvedModelRequest, NumericProviderInputs) {
    let family = ModelFamily::Unknown("bound-provider-fixture".into());
    let descriptor =
        crate::model_factory::tests::descriptor(family.clone(), "bound-provider-fixture", 4);
    let limits = NumericCapacityLimits {
        max_positions: 32,
        max_sequences: 2,
        max_batch_tokens: 4,
        ..Default::default()
    };
    let request = ResolvedModelRequest {
        execution: ResolvedExecution::new(
            ModelExecutionBackend::Cuda,
            ResolvedConstruction::Pipeline(unused_builder),
            StorageEncoding::NumericFp8E4m3fnBf16Scales,
            ArithmeticPrecision::F32Tf32x3,
            "synthetic provider metadata only",
        ),
        family: family.clone(),
        backend: ModelExecutionBackend::Cuda,
        model_path: f.dir.clone(),
        model_info: ferrule_model::ModelInfo::from_descriptor(&descriptor, "cuda"),
        family_options: ResolvedFamilyOptions::Generic,
        pipeline: Some(PipelineBuildOptions {
            parallelism: ferrule_common::ParallelismPlan {
                pipeline_parallel: 2,
                tensor_parallel: 2,
                expert_parallel: 2,
                ..Default::default()
            },
            devices: Some(vec![0, 1, 2, 3, 0, 1, 2, 3]),
            ..Default::default()
        }),
        max_layers: 4,
        max_tensor_bytes: 1 << 30,
        output_head_chunk_rows: 4096,
        expert_reader_max_tensor_bytes: 64 << 20,
        expert_memory_policy: ExpertMemoryPolicy::default(),
        moe_hotset_experts: 0,
        kv_cache_bytes: None,
        scheduler_config: ResidentSchedulerConfig {
            max_active_sequences: 2,
            max_batch_tokens: 4,
            prefill_chunk_size: 2,
            prefix_cache_capacity_pages: 0,
            ..Default::default()
        },
        driver_config: ResidentTopKDriverConfig {
            ctx_size: 32,
            enable_native_proposals: false,
            ..Default::default()
        },
    };
    let inputs = NumericProviderInputs {
        model_label: "bound-provider-fixture",
        family,
        limits,
        numeric_profile: NumericFp8Profile::F32,
        parallel_execution: ParallelExecutionContract::external_f32(),
        host_cache: HostExpertCacheOptions::default(),
        placement: None,
    };
    (request, inputs)
}
fn unused_builder(_: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    panic!("metadata test must not choose an engine")
}

#[test]
fn unknown_family_bound_resources_plan_without_adapter_or_catalog() {
    let f = Fixture::new();
    let (request, inputs) = fixture_request(&f);
    assert!(implementation(&request.family).is_err());
    let plan = Plan::new(&request, &f.resources, &inputs).unwrap();
    assert_eq!(plan.roots, 4);
    assert_eq!(plan.owners.len(), 2);
    assert_eq!(
        plan.owners[0]
            .group
            .members
            .iter()
            .map(|r| r.get())
            .collect::<Vec<_>>(),
        [4, 5]
    );
    #[cfg(feature = "cuda")]
    {
        let admission = Admission::new(&f.resources, plan).unwrap();
        assert_eq!(admission.budgets.len(), 8);
        assert!(
            admission.budgets[4..]
                .iter()
                .all(|b| b.kv == 0 && b.state == 0)
        );
    }
}

#[test]
fn neutral_options_preserve_profiles_and_identical_capacity() {
    let f = Fixture::new();
    let (request, mut inputs) = fixture_request(&f);
    for profile in [NumericFp8Profile::F32, NumericFp8Profile::W8A16Bf16] {
        inputs.numeric_profile = profile;
        let options = decoder_options(
            &f.resources,
            inputs.family.clone(),
            inputs.limits,
            request.max_tensor_bytes,
            profile,
        )
        .unwrap();
        assert_eq!(
            options.hybrid_cuda_numeric_fp8_precision(),
            Some(profile.precision())
        );
        let plan = Plan::new(&request, &f.resources, &inputs).unwrap();
        assert_eq!(plan.profile(), profile);
        assert_eq!(plan.config().max_positions, 32);
    }
}

#[test]
fn neutral_provider_rejects_stale_source_before_preparation() {
    let f = Fixture::new();
    let (request, inputs) = fixture_request(&f);
    Plan::new(&request, &f.resources, &inputs).unwrap();
    // Mutate only the fixture-owned backing file.
    std::fs::write(f.dir.join("weights.bin"), b"stale source identity").unwrap();
    assert!(Plan::new(&request, &f.resources, &inputs).is_err());
}

#[test]
fn neutral_provider_rejects_unconsumed_request_options() {
    let f = Fixture::new();
    let (request, inputs) = fixture_request(&f);
    for mutate in [
        |r: &mut ResolvedModelRequest| r.moe_hotset_experts = 1,
        |r: &mut ResolvedModelRequest| r.output_head_chunk_rows = 2,
        |r: &mut ResolvedModelRequest| r.driver_config.enable_native_proposals = true,
        |r: &mut ResolvedModelRequest| r.scheduler_config.prefix_cache_capacity_pages = 1,
    ] {
        let mut r = request.clone();
        mutate(&mut r);
        assert!(Plan::new(&r, &f.resources, &inputs).is_err());
    }
}

#[test]
fn neutral_provider_keeps_full_depth_and_separate_resident_placement() {
    let f = Fixture::new();
    let (mut request, mut inputs) = fixture_request(&f);
    request.max_layers = 2;
    assert!(Plan::new(&request, &f.resources, &inputs).is_err());
    request.max_layers = 4;
    inputs.placement = Some(ExpertPlacementConfig::new(2, vec![0, 1]).unwrap());
    assert!(Plan::new(&request, &f.resources, &inputs).is_err());
    inputs.placement = None;
    inputs.host_cache.mode = ExpertPrewarmMode::Lazy;
    assert!(Plan::new(&request, &f.resources, &inputs).is_err());
}

#[test]
fn neutral_external_root_uses_authoritative_estimator() {
    let f = Fixture::new();
    let (request, inputs) = fixture_request(&f);
    let options = decoder_options(
        &f.resources,
        inputs.family,
        inputs.limits,
        request.max_tensor_bytes,
        inputs.numeric_profile,
    )
    .unwrap();
    let estimate = ferrule_model::decoder::HybridMemoryPlan::for_resources_with_routed_experts(
        &f.resources,
        &options,
        8,
    )
    .unwrap();
    let root = ExpertRootMemory::for_resources(&f.resources, &options, 8).unwrap();
    assert_eq!(root.weights_bytes, estimate.weight_bytes_upper_bound);
    assert_eq!(root.state_bytes, estimate.per_sequence_state_bytes * 5);
    assert_eq!(root.kv_bytes, estimate.kv_bytes);
}

#[test]
fn neutral_payload_envelope_preserves_workspace_and_checks_overflow() {
    assert_eq!(
        expert_payload_envelope(123, 456).unwrap(),
        123 + (64 << 20) + 456
    );
    assert!(expert_payload_envelope(usize::MAX, 0).is_err());
    assert!(expert_payload_envelope(0, usize::MAX).is_err());
}

#[test]
fn neutral_physical_ledger_charges_both_root_contexts_and_exact_samples() {
    let placement = ExpertPlacementConfig::new(2, vec![3, 1]).unwrap();
    let plan = ExpertPhysicalPlan::new(
        &placement,
        ExpertRootMemory {
            weights_bytes: 100,
            state_bytes: 20,
            kv_bytes: 30,
            workspace_bytes: 40,
        },
        &[50, 60],
        70,
        10,
    )
    .unwrap();
    assert_eq!(plan.cards()[0].required_bytes, 330);
    assert_eq!(plan.cards()[0].contexts, 2);
    let mut samples = vec![
        PhysicalCardCapacity {
            ordinal: 3,
            free_bytes: 330,
            limit_bytes: 330,
        },
        PhysicalCardCapacity {
            ordinal: 1,
            free_bytes: 140,
            limit_bytes: 140,
        },
    ];
    plan.check_capacity(&samples).unwrap();
    samples[0].free_bytes -= 1;
    assert!(plan.check_capacity(&samples).is_err());
    samples[1].ordinal = 3;
    assert!(plan.check_capacity(&samples).is_err());
}

#[test]
fn neutral_host_ledger_is_not_a_reservation_or_rss_credit() {
    let f = Fixture::new();
    let (request, inputs) = fixture_request(&f);
    let base = StandardHostStartupMemory::for_resources(&f.resources, 32, request.max_tensor_bytes)
        .unwrap();
    let host = host_startup_options(
        &f.resources,
        inputs.host_cache,
        32,
        request.max_tensor_bytes,
    )
    .unwrap();
    assert_eq!(host.memory_budget, base.memory_budget().unwrap());
    assert_eq!(host.memory_budget.base_already_resident_bytes, 0);
}

#[test]
fn process_common_overhead_preserves_child_charge_and_overflow() {
    assert_eq!(
        process_child_overhead(100, 200, 3).unwrap(),
        (400 + (16 << 20) + 6400) * 3
    );
    assert!(process_child_overhead(u64::MAX, 0, 1).is_err());
    assert!(process_child_overhead(0, u64::MAX, 1).is_err());
}

#[test]
fn neutral_process_boot_plan_has_only_owned_partitions_without_starting_children() {
    #[cfg(unix)]
    {
        let f = Fixture::new();
        let (mut request, inputs) = fixture_request(&f);
        request.pipeline = None;
        let config = super::inputs::process_owner_config(std::time::Duration::from_millis(501));
        assert_eq!(
            config.startup_timeout,
            std::time::Duration::from_millis(501)
        );
        assert_eq!(config.command_timeout, config.startup_timeout);
        config.validate().unwrap();
        let mut placement = ExpertPlacementConfig::new(2, vec![3, 1]).unwrap();
        placement.rank_backend = PipelineRankBackend::Process;
        placement.process_launch = Some(ferrule_runtime::parallel::process::ProcessLaunch::new(
            "metadata-only-not-launched",
        ));
        let owners = OwnerPlan::new(&f.resources, &placement, 4).unwrap();
        let host = HostExpertCacheOptions {
            memory_budget: HostMemoryBudget {
                headroom_bytes: 0,
                ..Default::default()
            },
            ..inputs.host_cache
        };
        let plan = super::process::Plan::new(
            &request,
            &f.resources,
            &placement,
            &owners,
            host,
            inputs.numeric_profile,
        )
        .unwrap();
        assert_eq!(plan.boots.len(), 2);
        assert!(f.resources.host_experts().is_none());
        for (slot, (_, boot)) in plan.boots.iter().enumerate() {
            assert_eq!(boot.host_max_experts, owners.counts[slot]);
            assert_eq!(
                boot.device_bytes,
                expert_payload_envelope(owners.weights[slot], owners.activation_bytes).unwrap()
            );
            assert_eq!(boot.numeric_profile, inputs.numeric_profile.precision());
        }
    }
}

#[test]
fn sealed_arithmetic_cannot_crosswire_root_options_and_expert_startup() {
    let f = Fixture::new();
    for tp in [1, 2] {
        for (profile, replacement) in [
            (NumericFp8Profile::F32, NumericFp8Profile::W8A16Bf16),
            (NumericFp8Profile::W8A16Bf16, NumericFp8Profile::F32),
        ] {
            let (mut request, mut inputs) = fixture_request(&f);
            let parallel = request.pipeline.as_mut().unwrap();
            parallel.parallelism.tensor_parallel = tp;
            parallel.devices = Some((0..2 * (tp + 2)).map(|n| n % 4).collect());
            inputs.numeric_profile = profile;
            let plan = Plan::new(&request, &f.resources, &inputs).unwrap();
            #[cfg(feature = "cuda")]
            let reference = Admission::new(
                &f.resources,
                Plan::new(&request, &f.resources, &inputs).unwrap(),
            )
            .unwrap();

            // Previously these replacement inputs could be passed to admission
            // independently of the plan. Admission now has no such argument.
            inputs.numeric_profile = replacement;
            let root = plan.decoder_options(&f.resources).unwrap();
            assert_ne!(inputs.numeric_profile, plan.profile());
            assert_eq!(plan.profile(), profile);
            assert_eq!(
                root.hybrid_cuda_numeric_fp8_precision(),
                Some(plan.profile().precision())
            );
            #[cfg(feature = "cuda")]
            {
                // Metadata estimates only: never admit_live, prewarm or startup.
                let admission = Admission::new(&f.resources, plan).unwrap();
                assert_eq!(
                    admission.options.hybrid_cuda_numeric_fp8_precision(),
                    Some(profile.precision())
                );
                assert_eq!(admission.plan.profile(), profile);
                assert_eq!(admission.budgets, reference.budgets);
                assert_eq!(admission.plan.host(), reference.plan.host());
            }
        }
    }
}

fn reject_crosswired_capacity(mutate: impl FnOnce(&mut NumericCapacityLimits)) {
    let f = Fixture::new();
    let (request, mut inputs) = fixture_request(&f);
    mutate(&mut inputs.limits);
    let error = Plan::new(&request, &f.resources, &inputs)
        .err()
        .expect("mismatched capacity must not produce a plan");
    assert!(
        error
            .to_string()
            .contains("numeric provider limits differ from pipeline"),
        "{error}"
    );
}

#[test]
fn crosswired_context_limit_is_rejected_before_admission() {
    reject_crosswired_capacity(|limits| limits.max_positions += 1);
}

#[test]
fn crosswired_batch_limit_is_rejected_before_admission() {
    reject_crosswired_capacity(|limits| limits.max_batch_tokens += 1);
}

#[test]
fn crosswired_sequence_limit_is_rejected_before_admission() {
    reject_crosswired_capacity(|limits| limits.max_sequences += 1);
}

#[test]
fn sealed_limits_and_read_bound_do_not_follow_later_request_or_input_mutation() {
    let f = Fixture::new();
    let (mut request, mut inputs) = fixture_request(&f);
    let plan = Plan::new(&request, &f.resources, &inputs).unwrap();
    let expected = plan.decoder_options(&f.resources).unwrap();
    inputs.limits.max_positions = 64;
    inputs.limits.max_batch_tokens = 8;
    inputs.limits.max_sequences = 4;
    inputs.limits.max_experts = 1;
    inputs.limits.max_bytes = 1;
    inputs.limits.scratch_bytes = 1;
    request.driver_config.ctx_size = 128;
    request.scheduler_config.max_batch_tokens = 16;
    request.scheduler_config.max_active_sequences = 8;
    request.max_tensor_bytes = 1;
    assert!(Plan::new(&request, &f.resources, &inputs).is_err());
    let root = plan.decoder_options(&f.resources).unwrap();
    assert_eq!(root.max_positions(), expected.max_positions());
    assert_eq!(root.capabilities(), expected.capabilities());
    assert_eq!(root.max_parameter_bytes(), expected.max_parameter_bytes());
    assert_eq!(
        root.hybrid_cuda_numeric_fp8(),
        expected.hybrid_cuda_numeric_fp8()
    );
    assert_eq!(root.max_positions(), plan.config().max_positions);
    assert_eq!(
        root.capabilities().max_batch_tokens,
        plan.config().max_batch_tokens
    );
    assert_eq!(
        root.capabilities().max_sequences,
        plan.config().session_capacity
    );
    #[cfg(feature = "cuda")]
    {
        let admission = Admission::new(&f.resources, plan).unwrap();
        assert_eq!(admission.options.capabilities(), expected.capabilities());
        assert_eq!(
            admission.options.max_parameter_bytes(),
            expected.max_parameter_bytes()
        );
        assert_eq!(
            admission.options.hybrid_cuda_numeric_fp8(),
            expected.hybrid_cuda_numeric_fp8()
        );
    }
}

#[test]
fn sealed_host_policy_and_stage_placement_do_not_follow_replacement_inputs() {
    let f = Fixture::new();
    let (mut request, mut inputs) = fixture_request(&f);
    let plan = Plan::new(&request, &f.resources, &inputs).unwrap();
    let host = plan.host();
    let devices = plan.devices.clone();
    #[cfg(feature = "cuda")]
    let reference = Admission::new(
        &f.resources,
        Plan::new(&request, &f.resources, &inputs).unwrap(),
    )
    .unwrap();
    inputs.host_cache.mode = ExpertPrewarmMode::Lazy;
    inputs.host_cache.max_bytes = 1;
    inputs.host_cache.max_experts = 1;
    inputs.host_cache.workers = 1;
    inputs.host_cache.memory_budget.headroom_bytes = u64::MAX;
    assert!(Plan::new(&request, &f.resources, &inputs).is_err());
    inputs.host_cache = HostExpertCacheOptions::default();
    inputs.placement = Some(ExpertPlacementConfig::new(2, vec![7, 6]).unwrap());
    assert!(Plan::new(&request, &f.resources, &inputs).is_err());
    request.pipeline.as_mut().unwrap().devices = Some(vec![7; 8]);
    assert_eq!(plan.host(), host);
    assert_eq!(plan.devices, devices);
    #[cfg(feature = "cuda")]
    {
        // Compile-time contract: neither a request nor replacement inputs fit.
        let admit: fn(&BoundDecoderResources, Plan) -> Result<Admission> = Admission::new;
        let admission = admit(&f.resources, plan).unwrap();
        assert_eq!(admission.plan.host(), reference.plan.host());
        assert_eq!(admission.budgets, reference.budgets);
        assert_eq!(admission.plan.devices, devices);
    }
}
