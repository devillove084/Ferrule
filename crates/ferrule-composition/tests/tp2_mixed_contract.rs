//! Metadata-only capability probes. No checkpoint payloads, engines, or CUDA owners.
//! A generic topology capability is not a Mixed provider or quality acceptance.
#[allow(dead_code)]
#[path = "../src/model_factory/execution_policy.rs"]
mod execution_policy;
#[allow(dead_code)]
#[path = "parallel_composition_contract/support.rs"]
mod support;

use execution_policy::*;
use ferrule_common::ParallelismPlan;
use ferrule_composition::model_factory::*;
use ferrule_model::transformer::NumericFp8Precision;
use ferrule_model::transformer::hybrid_parallel::HybridParallelPlan;
use ferrule_model::{ModelExecutionBackend, ModelFamily};

fn parallel(ep: usize) -> PipelineBuildOptions {
    PipelineBuildOptions {
        parallelism: ParallelismPlan {
            tensor_parallel: 2,
            expert_parallel: ep,
            ..Default::default()
        },
        devices: Some(if ep == 1 {
            vec![0, 1]
        } else {
            vec![0, 1, 0, 1]
        }),
        ..Default::default()
    }
}

#[test]
fn explicit_mixed_metadata_preserves_owner_layers_and_does_not_enable_execution() {
    use ferrule_model::transformer::hybrid_parallel::{
        HybridCollectivePrecision, HybridRankPrecision, HybridRoutedExpertExecution,
    };
    let spec = support::spec();
    let external = support::topology(&spec, &support::PP1_TP2_EP2);
    let original = &external.stages()[0];
    for routed in [
        HybridRoutedExpertExecution::LocalTensorShards,
        HybridRoutedExpertExecution::ExternalExpertAttachment(
            original.expert_attachment().unwrap().clone(),
        ),
    ] {
        let local = routed.is_local_tensor_sharded();
        let plan = HybridParallelPlan::new_with_execution(
            &spec,
            0..40,
            original.scope(),
            original.tensor_members().to_vec(),
            routed,
            HybridRankPrecision::MixedBf16F32,
        )
        .unwrap();
        assert_eq!(plan.physical_members().len(), if local { 2 } else { 4 });
        assert_eq!(plan.expert_attachment().is_none(), local);
        assert_eq!(
            plan.precision().collective(),
            HybridCollectivePrecision::Bf16PartialsAndReduction
        );
        assert!(plan.validate_rank_execution().is_err());
        for lane in 0..2 {
            let layout = plan.rank_layout(lane).unwrap();
            let entries = layout.state_schema().entries();
            // TP partitions heads/channels, not the owner's layer count.
            assert_eq!(entries.len(), 40);
            for (layer, entry) in entries.iter().enumerate() {
                assert_eq!(entry.global_layer, layer);
                assert_eq!(entry.local_layer, layer);
                assert_eq!(entry.state_slot, layer);
            }
            let state = layout.state_budget(5).unwrap(); // 2*S+1 with S=2.
            assert_eq!(state.copies, 5);
            assert_eq!(state.total_grants, 5 * state.per_sequence_grants);
            assert!(state.copy_workspace_grants > 0);
            assert!(
                state.per_sequence_bytes
                    < original.rank_layout(lane).unwrap().state_bytes().unwrap()
            );
            assert!(layout.state_budget(0).is_err());
        }
    }
}

#[test]
fn neutral_plan_seals_local_tp2_without_external_owners_but_execution_stays_gated() {
    let options = PipelineBuildOptions {
        parallelism: ParallelismPlan {
            tensor_parallel: 2,
            expert_parallel: 1,
            ..Default::default()
        },
        devices: Some(vec![0, 1]),
        ..Default::default()
    };
    let spec = support::spec();
    let contract = ParallelExecutionContract::new(
        ferrule_model::transformer::hybrid_parallel::HybridRankPrecision::MixedBf16F32,
        ParallelExpertMode::LocalTensorShards,
    );
    let plan = ParallelExecutionPlan::describe(&spec, &options, contract, vec![]).unwrap();
    assert_eq!(plan.root_count(), 2);
    assert_eq!(plan.external_owner_count(), 0);
    assert!(
        plan.layout()
            .stages()
            .iter()
            .all(|stage| stage.expert_attachment().is_none())
    );
    assert!(ParallelExecutionPlan::prepare(&spec, &options, contract, vec![]).is_err());
    assert_eq!(plan.devices(), &[0, 1]);
    assert_eq!(plan.topology().world_size(), 2);
    assert_eq!(plan.contract(), contract);
    assert!(plan.validate_execution().is_err());
    let mut process = options.clone();
    process.rank_backend = PipelineRankBackend::Process;
    assert!(ParallelExecutionPlan::describe(&spec, &process, contract, vec![]).is_err());
    let external = support::topology(&spec, &support::PP1_TP2_EP2);
    let attachment = external.stages()[0].expert_attachment().unwrap().clone();
    assert!(
        ParallelExecutionPlan::describe(&spec, &options, contract, vec![attachment.clone()])
            .is_err()
    );
    let external_plan = ParallelExecutionPlan::prepare(
        &spec,
        &parallel(2),
        ParallelExecutionContract::external_f32(),
        vec![attachment],
    )
    .unwrap();
    assert_eq!(external_plan.root_count(), 2);
    assert_eq!(external_plan.external_owner_count(), 2);
    assert_eq!(external_plan.topology().world_size(), 2);
    assert_eq!(external_plan.devices(), &[0, 1, 0, 1]);
}

#[test]
fn generic_capability_slots_distinguish_two_roots_from_four_colocated_owners() {
    let selected = ExecutionCapabilities {
        ranks: RankCapability::ThreadOnly,
        tensor: TensorCapability::DeviceThreads { degrees: &[2] },
    };
    let placement = PlacementPolicy {
        selection: DeviceSelection::Explicit,
        sharing: DeviceSharing::Colocated,
    };
    for (ep, layout, owners) in [
        (1, OwnerLayout::StageRoots, 2),
        (2, OwnerLayout::StageRootsWithExperts, 4),
    ] {
        let input = parallel(ep);
        let topology = validate_topology(input.parallelism, layout).unwrap();
        assert_eq!(topology.logical_owners(), owners);
        assert_eq!(topology.device_slots(), owners);
        let placed = validate_placement(
            topology,
            PlacementTarget::Device,
            input.devices.as_deref(),
            placement,
        )
        .unwrap();
        assert_eq!(
            (0..owners)
                .map(|i| placed.device(i).unwrap())
                .collect::<Vec<_>>(),
            *input.devices.as_ref().unwrap()
        );
        assert_eq!(placed.device(owners), None);
        validate_execution_kind(
            topology,
            RankTransport::Thread,
            PlacementTarget::Device,
            selected,
        )
        .unwrap();
        assert_eq!(
            validate_execution_kind(
                topology,
                RankTransport::Thread,
                PlacementTarget::Host,
                selected
            ),
            Err(PolicyError::TensorRequiresDeviceThreads)
        );
        assert_eq!(
            validate_execution_kind(
                topology,
                RankTransport::Process,
                PlacementTarget::Device,
                selected
            ),
            Err(PolicyError::ThreadRanksRequired)
        );
        assert_eq!(
            validate_execution_kind(
                topology,
                RankTransport::Thread,
                PlacementTarget::Device,
                ExecutionCapabilities {
                    tensor: TensorCapability::Unsharded,
                    ..selected
                },
            ),
            Err(PolicyError::TensorUnsupported)
        );
        if ep == 2 {
            assert_eq!(
                validate_placement(topology, PlacementTarget::Device, Some(&[0, 1]), placement),
                Err(PolicyError::DeviceCount {
                    expected: 4,
                    actual: 2
                })
            );
        }
    }
}

#[test]
fn colocating_external_ep_does_not_turn_it_into_local_tensor_sharded_experts() {
    let spec = support::spec();
    let topology = support::topology(&spec, &support::PP1_TP2_EP2);
    let stage = &topology.stages()[0];
    assert_eq!(
        stage.tensor_members(),
        &[support::rank(0), support::rank(1)]
    );
    let ep = stage.expert_attachment().unwrap();
    assert_eq!(ep.source_rank, support::rank(0));
    assert_eq!(ep.members, vec![support::rank(2), support::rank(3)]);
    assert_eq!(topology.physical_members().len(), 4);
    for layer in 0..40 {
        for expert in 0..256 {
            assert_eq!(
                ep.placement.owner(layer, expert),
                Some(support::rank(2 + expert % 2))
            );
        }
    }
    // Removing EP must not silently select local MoE TP. The model requires
    // an explicit expert ownership mode for routed layers.
    let error = HybridParallelPlan::new(
        &spec,
        0..40,
        stage.scope(),
        stage.tensor_members().to_vec(),
        None,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("routed layer requires"),
        "{error}"
    );
    assert!(
        parallel(1)
            .hybrid_precision(&ModelFamily::Qwen35Moe, ModelExecutionBackend::Cuda)
            .is_err()
    );
    use ferrule_composition::model_factory::qwen35_composition::{
        HybridExecutionCapability, HybridExecutionProfile, Qwen35CompositionAdapter,
    };
    let adapter = Qwen35CompositionAdapter::new(&topology);
    let profile = HybridExecutionProfile {
        model: ferrule_model::models::qwen35::Qwen35Profile::Moe35BA3Bfp8,
        backend: ModelExecutionBackend::Cuda,
        rank_backend: PipelineRankBackend::Thread,
        parallelism: parallel(2).parallelism,
        numeric: NumericFp8Profile::F32,
    };
    assert_eq!(
        adapter.execution_capability(profile).unwrap(),
        HybridExecutionCapability::F32WithFullModelStrictEvidence
    );
    assert!(
        adapter
            .execution_capability(HybridExecutionProfile {
                parallelism: parallel(1).parallelism,
                ..profile
            })
            .is_err()
    );
}

#[test]
fn colocated_f32_catalog_geometry_is_not_live_capacity_or_mixed_acceptance() {
    let result = support::prepare(parallel(2), Some(NumericFp8Profile::F32));
    if cfg!(feature = "cuda") {
        let plan = result.unwrap();
        assert_eq!(
            plan.topology_placement().physical.devices,
            Some(vec![0, 1, 0, 1])
        );
        assert_eq!(
            plan.support().admission,
            AdmissionBoundary::OwnerLiveAdmissionRequired
        );
        assert_eq!(
            plan.support().buildability,
            BuildabilityStatus::CatalogValidated
        );
    } else {
        assert!(result.is_err(), "no CUDA-to-CPU fallback");
    }
}

#[test]
fn mixed_tp2_rejects_resident_engine_and_external_ep_before_checkpoint_io() {
    // The direct numeric request exercises topology rejection independently of
    // ambient CUBLAS_WORKSPACE_CONFIG, without mutating process environment.
    for ep in [1, 2] {
        for engine in [
            ServingEngineSelection::Resident,
            ServingEngineSelection::Pipeline,
        ] {
            // Authorized local TP2 pipeline is covered by the full catalog fixture
            // tests; metadata-only topology still cannot approve other paths.
            if ep == 1 && matches!(engine, ServingEngineSelection::Pipeline) {
                continue;
            }
            let error = ResidentModelPlanner
                .prepare_numeric_fp8_bf16_serving_with_precision(
                    &support::config(ModelFamily::Qwen35Moe),
                    BackendSelection::Cuda,
                    None,
                    support::options(None),
                    ServingModelOptions {
                        model_precision: Some(ModelPrecision::Bf16),
                        engine,
                        parallel: Some(parallel(ep)),
                        ..Default::default()
                    },
                    NumericStoragePolicy::E4M3FnBf16Scales,
                    NumericFp8Precision::Bf16ModelMixedF32Accumulate,
                )
                .err()
                .expect("Mixed parallel requests must reject");
            // Mixed TP2 catalog validation now precedes checkpoint I/O; the
            // old TP1-only blanket parallel rejection no longer applies here.
            let message = error.to_string();
            assert!(
                message.contains("Mixed TP2") || message.contains("Mixed BF16 TP2"),
                "{error}"
            );
            assert!(
                message.contains("auto/pipeline engine")
                    || message.contains("PP1 TP2 EP1")
                    || message.contains("initial envelope")
                    || message.contains("explicit CUDA/BF16"),
                "{error}"
            );
        }
    }
}

#[test]
fn mixed_requires_literal_launcher_workspace_configuration_without_mutation() {
    use std::ffi::OsStr;
    let profile = ServingModelProfile::Qwen35MixedBf16;
    profile
        .validate_startup_environment(Some(OsStr::new(":0:0")))
        .unwrap();
    for value in [None, Some(""), Some(":4096:8"), Some(":0:0 ")] {
        assert!(
            profile
                .validate_startup_environment(value.map(OsStr::new))
                .is_err()
        );
    }
    #[cfg(unix)]
    {
        // Inspect launch data only; never execute the test binary as a worker.
        let launch = ferrule_runtime::parallel::process::ProcessLaunch::new(
            std::env::current_exe().unwrap(),
        )
        .env("CUBLAS_WORKSPACE_CONFIG", ":0:0");
        assert_eq!(
            launch.environment,
            vec![("CUBLAS_WORKSPACE_CONFIG".into(), ":0:0".into())]
        );
    }
}

#[allow(dead_code)]
#[path = "qwen35_memory_plan/mixed_fixture.rs"]
mod mixed_headers;

#[test]
fn bound_local_storage_claims_are_per_owner_and_not_full_admission() {
    use ferrule_model::models::qwen35::Qwen35Adapter;
    use ferrule_model::transformer::hybrid_parallel::HybridRankPrecision;
    let fixture = mixed_headers::Fixture::new();
    let (_, resources) = Qwen35Adapter::bind_hf_metadata(fixture.dir.path()).unwrap();
    let contract = ParallelExecutionContract::new(
        HybridRankPrecision::MixedBf16F32,
        ParallelExpertMode::LocalTensorShards,
    );
    let mut options = parallel(1);
    options.devices = Some(vec![0, 0]);
    let plan =
        ParallelExecutionPlan::describe(resources.spec(), &options, contract, vec![]).unwrap();
    let claims = plan.storage_claims(&resources, 16, 32, 8, 2).unwrap();
    assert_eq!(claims.len(), 2);
    for (lane, claim) in claims.iter().enumerate() {
        assert_eq!(claim.owner, support::rank(lane));
        assert_eq!(claim.device, 0);
        assert_eq!(claim.storage.precision, HybridRankPrecision::MixedBf16F32);
        assert_eq!(claim.storage.state.copies, 5);
        assert_eq!(
            claim.storage.state.total_grants,
            5 * claim.storage.state.per_sequence_grants
        );
        assert_eq!(claim.storage.physical_pages, 8);
        assert!(claim.storage.weight_grants > 0);
        assert!(claim.storage.state.copy_workspace_grants > 0);
        assert!(claim.storage.kv_grants > 0);
        assert!(
            claim
                .storage
                .parameters
                .iter()
                .any(|p| p.role == ferrule_model::TensorRole::RoutedExpertGate)
        );
    }
    // No guessed operation/provider/pool/transport total is manufactured by this API.
    assert!(plan.validate_execution().is_err());
    assert!(plan.storage_claims(&resources, 16, 32, 8, 0).is_err());
    assert!(
        plan.storage_claims(&resources, 16, 32, 8, usize::MAX)
            .is_err()
    );
    let segment = ferrule_model::transformer::LayerSegmentPlan::new(40, 0..40, true, true).unwrap();
    let error = ferrule_runtime::parallel::pipeline::tensor_scope::translate_hybrid_schedule(
        &plan.layout().stages()[0],
        &segment,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("per-site payload precision"),
        "{error}"
    );
}

#[test]
fn mixed_collective_claims_bind_real_roots_and_charge_colocated_staging_separately() {
    use ferrule_common::{ParallelGroupId, ParallelRankId, ParallelTopologyId};
    use ferrule_model::transformer::hybrid_parallel::{
        HybridCollectiveKind, HybridCollectiveSemantic, HybridCollectiveShape,
        HybridCollectiveSite, HybridRankPrecision,
    };
    use ferrule_runtime::parallel::collective::{HostCollectiveKind, HostCollectiveLimits};
    use ferrule_runtime::parallel::pipeline::tensor_scope::{
        TensorCollectiveOperator, TensorCollectiveSemantic, TensorCollectiveShape,
        TensorCollectiveSite,
    };
    use ferrule_runtime::parallel::tensor::decoder_collective::MixedRankCollectivePlan;
    let spec = support::spec();
    let mut options = parallel(1);
    options.devices = Some(vec![0, 0]);
    let plan = ParallelExecutionPlan::describe(
        &spec,
        &options,
        ParallelExecutionContract::new(
            HybridRankPrecision::MixedBf16F32,
            ParallelExpertMode::LocalTensorShards,
        ),
        vec![],
    )
    .unwrap();
    let make_site = |layer| TensorCollectiveSite {
        site: 1,
        layer: Some(layer),
        operator: TensorCollectiveOperator::Protocol {
            version: 3,
            operation: TensorCollectiveSemantic::KvReplication,
            kind: HostCollectiveKind::AllGatherBf16,
            shape: TensorCollectiveShape::PerRow(2),
        },
    };
    let limits = HostCollectiveLimits {
        max_ranks: 2,
        max_elements_per_rank: 8,
        max_host_bytes: 96,
    };
    let prepare = |epoch, members, source, layer| {
        MixedRankCollectivePlan::new(
            epoch,
            ParallelGroupId::new(9),
            members,
            source,
            vec![make_site(layer)],
            limits,
            2,
            4,
        )
    };
    let members = vec![support::rank(0), support::rank(1)];
    let collective = prepare(
        plan.topology().topology_id(),
        members.clone(),
        support::rank(0),
        0,
    )
    .unwrap();
    let payload = collective.payload_budget();
    assert_eq!(payload.service_peak_bytes, 96); // 2*8*2 input + 2*16*2 output.
    assert_eq!(payload.input_bytes_per_owner, 16);
    assert_eq!(payload.output_bytes_per_owner, 32);
    assert_eq!(payload.staging_bytes_per_owner, 48);
    assert_eq!(payload.total_host_bytes, 192);
    let claims = plan.collective_claims(0, &collective).unwrap();
    assert_eq!(claims.source, support::rank(0));
    assert_eq!(claims.service_host_bytes, 96);
    assert_eq!(claims.total_host_bytes, 192);
    assert_eq!(claims.owners.len(), 2);
    for (index, owner) in claims.owners.iter().enumerate() {
        assert_eq!(owner.owner, support::rank(index));
        assert_eq!(owner.device, 0);
        assert_eq!(owner.host_input_bytes + owner.host_output_bytes, 48);
    }
    assert!(plan.validate_execution().is_err());
    assert!(plan.collective_claims(1, &collective).is_err());
    for foreign in [
        prepare(
            ParallelTopologyId::new(44),
            members.clone(),
            support::rank(0),
            0,
        )
        .unwrap(),
        prepare(
            plan.topology().topology_id(),
            vec![support::rank(2), support::rank(3)],
            support::rank(2),
            0,
        )
        .unwrap(),
        prepare(
            plan.topology().topology_id(),
            members.clone(),
            support::rank(0),
            40,
        )
        .unwrap(),
    ] {
        assert!(plan.collective_claims(0, &foreign).is_err());
    }
    assert!(
        prepare(
            plan.topology().topology_id(),
            members.clone(),
            ParallelRankId::new(2),
            0
        )
        .is_err()
    );
    assert!(
        MixedRankCollectivePlan::new(
            plan.topology().topology_id(),
            ParallelGroupId::new(9),
            members,
            support::rank(0),
            vec![make_site(0)],
            HostCollectiveLimits {
                max_host_bytes: 95,
                ..limits
            },
            2,
            4
        )
        .is_err()
    );

    let data_site = HybridCollectiveSite {
        ordinal: 7,
        global_layer: Some(0),
        semantic: HybridCollectiveSemantic::AttentionOutput,
        kind: HybridCollectiveKind::ReduceSum,
        shape: HybridCollectiveShape {
            fixed: 0,
            per_sequence: 0,
            per_row: 2,
        },
    };
    let translated = MixedRankCollectivePlan::from_hybrid_sites(
        &plan.layout().stages()[0],
        ParallelGroupId::new(10),
        support::rank(0),
        3,
        std::slice::from_ref(&data_site),
        limits,
        2,
        4,
    )
    .unwrap();
    assert_eq!(
        translated.sites()[0].kind(),
        HostCollectiveKind::AllReduceSumBf16
    );
    assert_eq!(translated.sites()[0].site, 7);
    assert_eq!(translated.sites()[0].elements(2, 4).unwrap(), 8);
    plan.collective_claims(0, &translated).unwrap();
    for semantic in [
        HybridCollectiveSemantic::Agreement,
        HybridCollectiveSemantic::Completion,
    ] {
        assert!(
            MixedRankCollectivePlan::from_hybrid_sites(
                &plan.layout().stages()[0],
                ParallelGroupId::new(10),
                support::rank(0),
                3,
                &[HybridCollectiveSite {
                    semantic,
                    ..data_site.clone()
                }],
                limits,
                2,
                4
            )
            .is_err()
        );
    }
}

fn resource_capacity() -> ParallelRankCapacity {
    ParallelRankCapacity {
        page_size: 16,
        max_positions: 1024,
        max_rows: 2,
        sessions: 2,
        logical_pages: 64,
    }
}
fn margins() -> ParallelOwnerMargins {
    ParallelOwnerMargins {
        context_bytes: 64 << 20,
        allocator_bytes: 64 << 20,
    }
}
fn local_contract() -> ParallelExecutionContract {
    ParallelExecutionContract::new(
        ferrule_model::transformer::hybrid_parallel::HybridRankPrecision::MixedBf16F32,
        ParallelExpertMode::LocalTensorShards,
    )
}
fn full_schedule(
    plan: &ParallelExecutionPlan,
) -> ferrule_runtime::parallel::tensor::decoder_collective::MixedRankCollectivePlan {
    use ferrule_runtime::parallel::collective::HostCollectiveLimits;
    plan.mixed_collective_plans(
        resource_capacity(),
        HostCollectiveLimits {
            max_ranks: 2,
            max_elements_per_rank: 248_320,
            max_host_bytes: 4 << 20,
        },
    )
    .unwrap()
    .remove(0)
}

fn test_pools() -> ParallelProviderPools {
    use ferrule_backend::cuda_plan::{
        bf16_attention::{Bf16AttentionPlan, Bf16AttentionShape},
        bf16_gemmex::Bf16GemmExPlan,
    };
    // Synthetic component budgets for ledger tests, NOT a full-model estimator.
    let small = Bf16GemmExPlan::new_resident(128, 128, 2, usize::MAX, usize::MAX).unwrap();
    let large = Bf16GemmExPlan::new_resident(4096, 2048, 2, usize::MAX, usize::MAX).unwrap();
    let attention = Bf16AttentionPlan::from_shapes(
        &[Bf16AttentionShape {
            rows: 2,
            tokens: 1024,
        }],
        2,
        2,
        8,
        1,
        256,
        usize::MAX,
        usize::MAX,
    )
    .unwrap();
    let pool = ParallelProviderPools::from_plans(&[small, large], std::slice::from_ref(&attention))
        .unwrap();
    assert_eq!(
        pool,
        ParallelProviderPools::from_plans(&[small, large].repeat(40), &vec![attention; 40])
            .unwrap()
    );
    assert_eq!(
        pool.gemmex_storage_grants(),
        large.storage_grant_bytes() as u64
    );
    assert_eq!(
        pool.gemmex_external_reserve() + pool.attention_external_reserve(),
        32 << 20
    );
    pool
}
fn test_estimate(
    layout: &ferrule_model::transformer::hybrid_parallel::HybridRankLayout,
    capacity: ParallelRankCapacity,
    _: &ferrule_model::transformer::hybrid_parallel::HybridRankStorageBudget,
) -> ferrule_composition::Result<ParallelOperatorEstimate> {
    let mut e = ParallelOperatorEstimate::pending(layout, capacity);
    e.providers = Some(test_pools());
    e.auxiliary = Some(ParallelAuxiliaryRequirements {
        derived_rope_grants: 1024,
        attention_metadata_grants: 2048,
        recurrent_workspace_grants: 4096,
        other_persistent_grants: 0,
    });
    e.operation_grants = Some(1 << 20);
    e.boundary = Some(ParallelBoundaryRequirements {
        pinned_host_bytes: 4096,
        host_metadata_bytes: 1024,
        device_grants: 8192,
    });
    Ok(e)
}
#[test]
fn aggregate_resource_accounting_pools_layers_and_charges_colocated_roots_without_enabling_execution()
 {
    let fixture = mixed_headers::Fixture::new();
    let (_, resources) =
        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(fixture.dir.path()).unwrap();
    for devices in [vec![0, 1], vec![0, 0]] {
        let mut options = parallel(1);
        options.devices = Some(devices.clone());
        let plan =
            ParallelExecutionPlan::describe(resources.spec(), &options, local_contract(), vec![])
                .unwrap();
        let schedule = full_schedule(&plan);
        // Dropping typed route sites cannot turn the control budget into zero.
        let no_routes =
            ferrule_runtime::parallel::tensor::decoder_collective::MixedRankCollectivePlan::new(
                schedule.epoch(),
                schedule.group(),
                schedule.members().to_vec(),
                schedule.source(),
                schedule.sites().to_vec(),
                schedule.limits(),
                schedule.max_sequences(),
                schedule.max_rows(),
            )
            .unwrap();
        assert!(
            plan.resource_requirements(
                &resources,
                resource_capacity(),
                margins(),
                &[no_routes],
                test_estimate
            )
            .is_err()
        );
        let host = schedule.payload_budget().total_host_bytes as u64;
        assert!(schedule.route_control_budget().unwrap().total_host_bytes > 0);
        let requirements = plan
            .resource_requirements(
                &resources,
                resource_capacity(),
                margins(),
                &[schedule],
                test_estimate,
            )
            .unwrap();
        assert!(requirements.resource_complete());
        assert_eq!(requirements.owners().len(), 2);
        assert_eq!(plan.external_owner_count(), 0);
        assert_eq!(
            requirements.known_host_bytes().unwrap(),
            host + 2 * (4096 + 1024)
        );
        let mut sum = 0;
        for owner in requirements.owners() {
            assert_eq!(owner.storage.physical_pages, 128);
            assert_eq!(owner.storage.state.copies, 5);
            let expected = owner.storage.storage_grants().unwrap() as u64
                + owner.storage.state.copy_workspace_grants as u64
                + test_pools().total().unwrap()
                + 1024
                + 2048
                + 4096
                + (1 << 20)
                + 8192
                + (128 << 20);
            assert_eq!(owner.device_bytes().unwrap(), Some(expected));
            sum += expected;
        }
        let capacity = if devices[0] == devices[1] {
            vec![(0, sum)]
        } else {
            vec![(0, sum / 2), (1, sum / 2)]
        };
        let admitted = requirements
            .admit(&capacity, requirements.known_host_bytes().unwrap())
            .unwrap();
        assert_eq!(admitted.iter().map(|p| p.required).sum::<u64>(), sum);
        let mut short = capacity.clone();
        short[0].1 -= 1;
        assert!(requirements.admit(&short, u64::MAX).is_err());
        assert!(
            requirements
                .admit(&capacity, requirements.known_host_bytes().unwrap() - 1)
                .is_err()
        );
        let entered = std::cell::Cell::new(false);
        assert!(
            requirements
                .prepare_internal(&capacity, u64::MAX, |_| {
                    entered.set(true);
                    Ok(())
                })
                .is_err()
        );
        assert!(!entered.get());
    }
}
#[test]
fn pending_model_estimates_never_become_zero_or_complete_and_sources_are_rechecked() {
    let fixture = mixed_headers::Fixture::new();
    let (_, resources) =
        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(fixture.dir.path()).unwrap();
    let plan =
        ParallelExecutionPlan::describe(resources.spec(), &parallel(1), local_contract(), vec![])
            .unwrap();
    let requirements = plan
        .resource_requirements(
            &resources,
            resource_capacity(),
            margins(),
            &[],
            |layout, capacity, _| Ok(ParallelOperatorEstimate::pending(layout, capacity)),
        )
        .unwrap();
    assert!(!requirements.resource_complete());
    for owner in requirements.owners() {
        assert_eq!(owner.missing().len(), 5);
        assert_eq!(owner.device_bytes().unwrap(), None);
    }
    let capacities = [(0, 22 << 30), (1, 22 << 30)];
    requirements
        .check_known_capacity(&capacities, u64::MAX)
        .unwrap();
    assert!(requirements.admit(&capacities, u64::MAX).is_err());
    assert!(
        requirements
            .prepare_internal::<()>(&capacities, u64::MAX, |_| panic!(
                "incomplete estimator entered owner"
            ))
            .is_err()
    );
    assert!(
        plan.resource_requirements(
            &resources,
            resource_capacity(),
            margins(),
            &[],
            |layout, mut capacity, _| {
                capacity.max_rows += 1;
                Ok(ParallelOperatorEstimate::pending(layout, capacity))
            }
        )
        .is_err()
    );
    let path = fixture.dir.path().join("model.safetensors");
    let len = std::fs::metadata(&path).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(len + 8)
        .unwrap();
    assert!(
        requirements
            .check_known_capacity(&capacities, u64::MAX)
            .is_err()
    );
}
#[test]
#[ignore = "real checkpoint headers only; explicit FERRULE_MIXED_HEADER_DIR and hidden GPUs"]
fn real_tp2_mixed_header_storage_probe() {
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
    let path = std::env::var("FERRULE_MIXED_HEADER_DIR").unwrap();
    let (_, resources) =
        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(std::path::Path::new(&path))
            .unwrap();
    let plan =
        ParallelExecutionPlan::describe(resources.spec(), &parallel(1), local_contract(), vec![])
            .unwrap();
    let schedule = full_schedule(&plan);
    let report = plan
        .resource_requirements(
            &resources,
            resource_capacity(),
            margins(),
            &[schedule],
            |layout, capacity, _| Ok(ParallelOperatorEstimate::pending(layout, capacity)),
        )
        .unwrap();
    assert_eq!(plan.devices(), &[0, 1]);
    assert_eq!(plan.root_count(), 2);
    assert_eq!(plan.external_owner_count(), 0);
    assert!(!report.resource_complete());
    for owner in report.owners() {
        println!(
            "REAL_TP2 root={} device={} weights={} state={} copy={} physical_kv={} known_device={} gaps={:?}",
            owner.owner.get(),
            owner.device,
            owner.storage.weight_grants,
            owner.storage.state.total_grants,
            owner.storage.state.copy_workspace_grants,
            owner.storage.kv_grants,
            owner.known_device_bytes().unwrap(),
            owner.missing()
        );
        assert_eq!(owner.storage.weight_grants, 18_390_764_032);
        assert_eq!(owner.storage.state.total_grants, 162_201_600);
        assert_eq!(owner.storage.kv_grants, 20_971_520);
        assert!(owner.device_bytes().unwrap().is_none());
    }
    println!("TYPED_HOST_BYTES {}", report.known_host_bytes().unwrap());
    let full = plan
        .resource_requirements_from_model(
            &resources,
            resource_capacity(),
            margins(),
            &[full_schedule(&plan)],
        )
        .unwrap();
    assert!(full.resource_complete());
    for owner in full.owners() {
        let pools = owner.operators.providers.unwrap();
        let aux = owner.operators.auxiliary.unwrap();
        let boundary = owner.operators.boundary.unwrap();
        println!(
            "REAL_TP2_FULL root={} device={} device_bytes={} gemmex={} gemmex_external={} attention={} attention_external={} rope={} metadata={} recurrent_workspace={} operations={} boundary_device={} pinned={} boundary_host={}",
            owner.owner.get(),
            owner.device,
            owner.device_bytes().unwrap().unwrap(),
            pools.gemmex_storage_grants(),
            pools.gemmex_external_reserve(),
            pools.attention_storage_grants(),
            pools.attention_external_reserve(),
            aux.derived_rope_grants,
            aux.attention_metadata_grants,
            aux.recurrent_workspace_grants,
            owner.operators.operation_grants.unwrap(),
            boundary.device_grants,
            boundary.pinned_host_bytes,
            boundary.host_metadata_bytes
        );
    }
    println!("FULL_HOST_BYTES {}", full.known_host_bytes().unwrap());
    let serving_capacity = ParallelRankCapacity {
        max_rows: 8,
        max_positions: 128,
        logical_pages: 32,
        ..resource_capacity()
    };
    let serving_schedules = plan
        .mixed_collective_plans(
            serving_capacity,
            ferrule_runtime::parallel::collective::HostCollectiveLimits {
                max_ranks: 2,
                max_elements_per_rank: 993_280,
                max_host_bytes: 64 << 20,
            },
        )
        .unwrap();
    let serving = plan
        .resource_requirements_from_model(
            &resources,
            serving_capacity,
            margins(),
            &serving_schedules,
        )
        .unwrap();
    for owner in serving.owners() {
        println!(
            "REAL_TP2_BATCH8_CTX128 root={} device={} device_bytes={} workspace={} boundary={:?}",
            owner.owner.get(),
            owner.device,
            owner.device_bytes().unwrap().unwrap(),
            owner.operators.operation_grants.unwrap(),
            owner.operators.boundary
        );
    }
    println!(
        "BATCH8_CTX128_HOST_BYTES {}",
        serving.known_host_bytes().unwrap()
    );
    serving
        .prepare_internal(&[(0, 22 << 30), (1, 22 << 30)], 256 << 20, |_| Ok(()))
        .unwrap();

    let capacities = full
        .owners()
        .iter()
        .map(|o| (o.device, o.device_bytes().unwrap().unwrap()))
        .collect::<Vec<_>>();
    full.admit(&capacities, full.known_host_bytes().unwrap())
        .unwrap();
    full.prepare_internal(&capacities, u64::MAX, |owners| {
        assert_eq!(owners.len(), 2);
        Ok(())
    })
    .unwrap();
    assert!(
        plan.validate_execution().is_err(),
        "experimental assembly must not advertise profile support"
    );
    let mut short = capacities;
    short[0].1 -= 1;
    short[1].1 += 1 << 30;
    assert!(full.admit(&short, u64::MAX).is_err());

    report
        .check_known_capacity(&[(0, 22 << 30), (1, 22 << 30)], u64::MAX)
        .unwrap();
    assert!(
        report
            .admit(&[(0, 22 << 30), (1, 22 << 30)], u64::MAX)
            .is_err()
    );
    let mut options = parallel(1);
    options.devices = Some(vec![0, 0]);
    let colocated =
        ParallelExecutionPlan::describe(resources.spec(), &options, local_contract(), vec![])
            .unwrap();
    let report = colocated
        .resource_requirements(
            &resources,
            resource_capacity(),
            margins(),
            &[],
            |layout, capacity, _| Ok(ParallelOperatorEstimate::pending(layout, capacity)),
        )
        .unwrap();
    assert!(
        report
            .check_known_capacity(&[(0, 24 << 30)], u64::MAX)
            .is_err(),
        "same-card roots exceed 24GiB even before missing operators"
    );
}

#[test]
fn full_program_metadata_includes_output_head_and_binds_actual_runtime_transport() {
    use ferrule_model::transformer::hybrid_parallel::{
        HybridBf16Collective, HybridCollectiveSemantic,
    };
    use ferrule_runtime::parallel::{
        collective::HostCollectiveKind,
        tensor::decoder_collective::{DecoderTensorCollective, MixedRankCollective},
    };
    let spec = support::spec();
    let plan =
        ParallelExecutionPlan::describe(&spec, &parallel(1), local_contract(), vec![]).unwrap();
    let programs = plan.mixed_programs(resource_capacity()).unwrap();
    assert_eq!(programs.len(), 2);
    let schedule = std::sync::Arc::new(full_schedule(&plan));
    assert_eq!(schedule.sites().len(), 81);
    assert_eq!(
        schedule.sites().last().unwrap().kind(),
        HostCollectiveKind::AllGatherBf16
    );
    assert_eq!(
        schedule.sites().last().unwrap().elements(2, 2).unwrap(),
        248_320
    );
    let endpoints = DecoderTensorCollective::new_group_from_members(
        schedule.members(),
        schedule.epoch(),
        schedule.group(),
        schedule.limits(),
        std::time::Duration::from_secs(1),
    )
    .unwrap();
    // Keep both endpoints alive through validation; no execution transaction is entered.
    let transports = endpoints
        .into_iter()
        .zip(&programs)
        .map(|(endpoint, program)| {
            MixedRankCollective::prepare(endpoint, program.layout().owner(), schedule.clone())
                .unwrap()
        })
        .collect::<Vec<_>>();
    for (program, transport) in programs.iter().zip(&transports) {
        assert!(program.segment().owns_embedding());
        assert!(program.segment().owns_output());
        assert_eq!(program.layers(), 0..40);
        assert_eq!(program.state_copies(), 5);
        assert_eq!(
            program.sites().last().unwrap().semantic,
            HybridCollectiveSemantic::OutputHead
        );
        assert_eq!(
            transport.sealed_binding().unwrap(),
            program.collective_binding()
        );
        program.validate_collective(transport).unwrap();
        assert!(
            program
                .validate_collective(&transports[1 - program.layout().lane()])
                .is_err()
        );
    }
    assert!(
        plan.validate_execution().is_err(),
        "prepared metadata is not execution permission"
    );
}

#[test]
fn full_resource_ledger_rejects_hidden_only_schedule_before_model_estimation() {
    use ferrule_model::transformer::hybrid_parallel::{
        HYBRID_AGREEMENT_VERSION, HybridMixedRankProgramPlan,
    };
    use ferrule_runtime::parallel::tensor::decoder_collective::MixedRankCollectivePlan;
    let fixture = mixed_headers::Fixture::new();
    let (_, resources) =
        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(fixture.dir.path()).unwrap();
    let plan =
        ParallelExecutionPlan::describe(resources.spec(), &parallel(1), local_contract(), vec![])
            .unwrap();
    let stage = &plan.layout().stages()[0];
    let model = HybridMixedRankProgramPlan::from_layout(
        std::sync::Arc::new(stage.rank_layout(0).unwrap()),
        2,
        2,
        1024,
        16,
        5,
    )
    .unwrap();
    let full = full_schedule(&plan);
    let hidden = MixedRankCollectivePlan::from_hybrid_sites(
        stage,
        full.group(),
        full.source(),
        HYBRID_AGREEMENT_VERSION,
        &model.sites(),
        full.limits(),
        2,
        2,
    )
    .unwrap()
    .with_route_sites(model.route_sites())
    .unwrap();
    assert!(
        plan.resource_requirements(
            &resources,
            resource_capacity(),
            margins(),
            &[hidden],
            |_, _, _| panic!("incomplete serving schedule reached estimator")
        )
        .is_err()
    );
}

#[test]
fn shared_runtime_scope_binding_handles_optional_ep_without_extra_kv_roots() {
    use ferrule_model::transformer::ExpertDispatchLimits;
    let spec = support::spec();
    let local =
        ParallelExecutionPlan::describe(&spec, &parallel(1), local_contract(), vec![]).unwrap();
    let scopes = local
        .layout()
        .bind_tensor_scopes(local.topology(), local.segments(), &[])
        .unwrap();
    assert_eq!(
        scopes
            .tensor_collective_participants(0, 0)
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![support::rank(0), support::rank(1)]
    );
    let external = support::topology(&spec, &support::PP1_TP2_EP2);
    let plan = ParallelExecutionPlan::prepare(
        &spec,
        &parallel(2),
        ParallelExecutionContract::external_f32(),
        vec![external.stages()[0].expert_attachment().unwrap().clone()],
    )
    .unwrap();
    let group = plan
        .layout()
        .expert_group(
            0,
            ExpertDispatchLimits {
                max_tokens: 2,
                max_bytes: 65536,
            },
        )
        .unwrap();
    let scopes = plan
        .layout()
        .bind_tensor_scopes(
            plan.topology(),
            plan.segments(),
            std::slice::from_ref(&group),
        )
        .unwrap();
    assert_eq!(
        scopes
            .tensor_collective_participants(0, 0)
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![support::rank(0), support::rank(1)]
    );
    assert!(
        plan.layout()
            .bind_tensor_scopes(plan.topology(), plan.segments(), &[])
            .is_err()
    );
    assert!(
        plan.layout()
            .bind_tensor_scopes(
                plan.topology(),
                plan.segments(),
                &[group.clone(), group.clone()]
            )
            .is_err()
    );
    assert!(
        local
            .layout()
            .bind_tensor_scopes(
                local.topology(),
                local.segments(),
                std::slice::from_ref(&group)
            )
            .is_err()
    );
    let mut wrong_source = group;
    wrong_source.source_rank = support::rank(1);
    assert!(
        plan.layout()
            .bind_tensor_scopes(plan.topology(), plan.segments(), &[wrong_source])
            .is_err()
    );
    let foreign = ferrule_common::ValidatedParallelTopology::new(
        ferrule_common::ParallelTopologyId::new(987),
        2,
        support::rank(0),
        parallel(1).parallelism,
    )
    .unwrap();
    assert!(
        local
            .layout()
            .bind_tensor_scopes(&foreign, local.segments(), &[])
            .is_err()
    );
}

#[test]
fn model_operation_estimate_is_consumed_without_storage_or_state_copy_double_charge() {
    let fixture = mixed_headers::Fixture::new();
    let (_, resources) =
        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(fixture.dir.path()).unwrap();
    let plan =
        ParallelExecutionPlan::describe(resources.spec(), &parallel(1), local_contract(), vec![])
            .unwrap();
    let report = plan
        .resource_requirements_from_model(
            &resources,
            resource_capacity(),
            margins(),
            &[full_schedule(&plan)],
        )
        .unwrap();
    let programs = plan.mixed_programs(resource_capacity()).unwrap();
    assert!(report.resource_complete());
    for (owner, program) in report.owners().iter().zip(&programs) {
        let estimate = program
            .operation_estimate(&resources, resource_capacity().physical_pages().unwrap())
            .unwrap();
        assert_eq!(
            owner.device_bytes().unwrap(),
            Some(estimate.total_device_grants().unwrap() as u64 + 128 * 1024 * 1024)
        );
        assert_eq!(
            owner.operators.providers.unwrap().gemmex_external_reserve(),
            16 << 20
        );
        assert_eq!(
            owner
                .operators
                .providers
                .unwrap()
                .attention_external_reserve(),
            16 << 20
        );
        assert_eq!(
            owner.operators.boundary.unwrap().pinned_host_bytes,
            estimate.pinned_host_bytes as u64
        );
        let mut capacity = resource_capacity();
        capacity.max_rows += 1;
        assert!(
            ParallelOperatorEstimate::from_program(&resources, program, capacity, &owner.storage)
                .is_err()
        );
        let mut storage = owner.storage.clone();
        storage.state.copy_workspace_grants += 256;
        assert!(
            ParallelOperatorEstimate::from_program(
                &resources,
                program,
                resource_capacity(),
                &storage
            )
            .is_err()
        );
    }
}

#[test]
fn shared_runtime_scope_binding_preserves_f32_tp2_tp4_external_stages() {
    use ferrule_model::transformer::ExpertDispatchLimits;
    let spec = support::spec();
    for case in [
        &support::PP1_TP2_EP2,
        &support::PP2_TP2_EP2,
        &support::PP2_TP4_EP2,
    ] {
        let layout = support::topology(&spec, case);
        let attachments = layout
            .stages()
            .iter()
            .map(|s| s.expert_attachment().unwrap().clone())
            .collect();
        let plan = ParallelExecutionPlan::prepare(
            &spec,
            &case.options(),
            ParallelExecutionContract::external_f32(),
            attachments,
        )
        .unwrap();
        let limits = ExpertDispatchLimits {
            max_tokens: 2,
            max_bytes: 65536,
        };
        let groups = (0..case.pp)
            .map(|i| plan.layout().expert_group(i, limits).unwrap())
            .collect::<Vec<_>>();
        let scopes = plan
            .layout()
            .bind_tensor_scopes(plan.topology(), plan.segments(), &groups)
            .unwrap();
        for (stage, expected) in case.roots.iter().enumerate() {
            assert_eq!(
                scopes
                    .tensor_collective_participants(stage as u32, 0)
                    .unwrap()
                    .iter()
                    .map(|r| r.get())
                    .collect::<Vec<_>>(),
                *expected
            );
        }
        let mut omitted = groups.clone();
        omitted.pop();
        assert!(
            plan.layout()
                .bind_tensor_scopes(plan.topology(), plan.segments(), &omitted)
                .is_err()
        );
        if groups.len() > 1 {
            let mut duplicate = groups.clone();
            duplicate[1] = duplicate[0].clone();
            assert!(
                plan.layout()
                    .bind_tensor_scopes(plan.topology(), plan.segments(), &duplicate)
                    .is_err()
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn experimental_protocol_binds_actual_batch_context_and_all_attention_views() {
    use ferrule_model::execution::ExecutionPrecisionPolicy;
    use ferrule_runtime::parallel::{
        collective::HostCollectiveLimits,
        pipeline::{MixedTensorExecutionPlan, PipelineConfig},
    };
    let spec = support::spec();
    let plan =
        ParallelExecutionPlan::describe(&spec, &parallel(1), local_contract(), vec![]).unwrap();
    let config = PipelineConfig {
        page_size: 16,
        max_pages: 32,
        max_positions: 128,
        max_batch_tokens: 8,
        session_capacity: 2,
        max_parameter_bytes: 2 << 30,
        precision: ExecutionPrecisionPolicy::bf16_compatibility(),
        max_ack_polls: 8,
    };
    let limits = HostCollectiveLimits {
        max_ranks: 2,
        max_elements_per_rank: 8 * 124_160,
        max_host_bytes: 32 << 20,
    };
    let make = |config, limits| {
        MixedTensorExecutionPlan::new(
            &spec,
            plan.topology(),
            plan.segments(),
            plan.layout(),
            config,
            limits,
        )
    };
    let inputs = make(config, limits).unwrap();
    assert_eq!(inputs.attention_shapes().len(), 996);
    assert_eq!(inputs.program(0).unwrap().collective_binding().max_rows, 8);
    assert_eq!(
        inputs
            .collective()
            .sites()
            .last()
            .unwrap()
            .elements(2, 8)
            .unwrap(),
        993_280
    );
    for rows in 1..=8 {
        for tokens in rows..=128 {
            assert!(inputs.attention_shapes().contains(
                &ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape { rows, tokens }
            ));
        }
    }
    assert!(
        make(
            PipelineConfig {
                precision: ExecutionPrecisionPolicy::f32(),
                ..config
            },
            limits
        )
        .is_err()
    );
    assert!(
        make(
            PipelineConfig {
                max_batch_tokens: 33,
                ..config
            },
            limits
        )
        .is_err()
    );
    assert!(
        make(
            PipelineConfig {
                max_positions: 1025,
                ..config
            },
            limits
        )
        .is_err()
    );
    assert!(
        make(
            config,
            HostCollectiveLimits {
                max_elements_per_rank: 1,
                ..limits
            }
        )
        .is_err()
    );
    assert!(inputs.program(2).is_err());
}
