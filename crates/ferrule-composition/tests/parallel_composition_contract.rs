//! Public metadata contracts only. These tests never build an engine or read weights.
//! Runtime transaction/fault tests remain in ferrule-runtime; no test coordinator here.
#[path = "parallel_composition_contract/support.rs"]
mod support;

use ferrule_common::ParallelismPlan;
use ferrule_composition::model_factory::qwen35_composition::{
    HybridCapabilityError, HybridExecutionCapability, HybridExecutionProfile,
    Qwen35CompositionAdapter,
};
use ferrule_composition::model_factory::*;
use ferrule_model::models::qwen35::Qwen35Profile;
use ferrule_model::transformer::ExpertDispatchLimits;
use ferrule_model::transformer::hybrid_parallel::{HybridLocalAttention, HybridParallelPlan};
use ferrule_model::{ModelExecutionBackend, ModelFamily};
use ferrule_runtime::parallel::hybrid_topology::{HybridExpertTopology, HybridTopologyError};
use ferrule_runtime::parallel::pipeline::{HybridDeviceBudget, HybridStageBudget};
use support::*;

fn profile(case: &Case, numeric: NumericFp8Profile) -> HybridExecutionProfile {
    HybridExecutionProfile {
        model: Qwen35Profile::Moe35BA3Bfp8,
        backend: ModelExecutionBackend::Cuda,
        rank_backend: PipelineRankBackend::Thread,
        parallelism: case.parallelism(),
        numeric,
    }
}

fn check_case(case: &Case) {
    let spec = spec();
    let topology = topology(&spec, case);
    topology
        .validate_external_expert_roster(case.parallelism())
        .unwrap();
    let expected_owners: Vec<_> = case
        .roots
        .iter()
        .chain(case.experts)
        .flat_map(|r| r.iter().copied())
        .collect();
    assert_eq!(
        topology
            .physical_members()
            .iter()
            .map(|r| r.get())
            .collect::<Vec<_>>(),
        expected_owners
    );
    let full_kv = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39];
    for (index, stage) in topology.stages().iter().enumerate() {
        let range = match (case.pp, index) {
            (1, 0) => 0..40,
            (2, 0) => 0..20,
            (2, 1) => 20..40,
            _ => panic!("unexpected acceptance case"),
        };
        assert_eq!(stage.schema().layer_range(), range);
        assert_eq!(
            stage
                .tensor_members()
                .iter()
                .map(|r| r.get())
                .collect::<Vec<_>>(),
            case.roots[index]
        );
        let group = topology
            .expert_group(
                index,
                ExpertDispatchLimits {
                    max_tokens: 8,
                    max_bytes: 65536,
                },
            )
            .unwrap();
        assert_eq!(group.source_rank.get(), case.roots[index][0]);
        assert_eq!(
            group.members.iter().map(|r| r.get()).collect::<Vec<_>>(),
            case.experts[index]
        );
        assert_eq!(group.layers, range);
        assert!(!group.members.contains(&group.source_rank));
        for lane in 0..case.tp {
            let local = stage.rank_layout(lane).unwrap();
            assert_eq!(local.owner().get(), case.roots[index][lane]);
            assert_eq!(local.state_schema().entries().len(), range.len());
            let mut kv = Vec::new();
            for entry in local.state_schema().entries() {
                assert_eq!(entry.global_layer, range.start + entry.local_layer);
                assert_eq!(entry.state_slot, entry.local_layer);
                if let Some(global_kv) = entry.global_kv_layer {
                    kv.push((entry.global_layer, global_kv, entry.local_kv_layer.unwrap()));
                } else {
                    assert_eq!(entry.local_kv_layer, None);
                    assert!(!full_kv.contains(&entry.global_layer));
                }
            }
            let expected_kv: Vec<_> = full_kv
                .iter()
                .enumerate()
                .filter(|(_, layer)| range.contains(layer))
                .enumerate()
                .map(|(local, (global, layer))| (*layer, global, local))
                .collect();
            assert_eq!(kv, expected_kv);
            let HybridLocalAttention::Gqa {
                kv_heads,
                kv_owners,
                ..
            } = local.attention(range.end - 1).unwrap()
            else {
                panic!("last layer is GQA")
            };
            match case.tp {
                1 => {
                    assert_eq!(kv_heads.clone(), 0..2);
                    assert_eq!(kv_owners, &vec![local.owner(); 2]);
                }
                2 => {
                    assert_eq!(kv_heads.clone(), lane..lane + 1);
                    assert_eq!(kv_owners, &vec![local.owner()]);
                }
                4 => {
                    assert_eq!(kv_heads.clone(), lane / 2..lane / 2 + 1);
                    assert_eq!(kv_owners, &vec![rank(index * 4 + lane / 2 * 2)]);
                }
                _ => unreachable!(),
            }
        }
    }
    let plan = prepare(case.options(), None);
    if cfg!(feature = "cuda") {
        let plan = plan.unwrap();
        assert_eq!(
            plan.selected_implementation().kind,
            SelectedImplementationKind::Pipeline
        );
        assert_eq!(plan.backend(), ModelExecutionBackend::Cuda);
        assert_eq!(plan.numeric_fp8_profile(), Some(NumericFp8Profile::F32));
        assert_eq!(plan.support().arithmetic, ArithmeticPrecision::F32Tf32x3);
        assert_eq!(
            plan.support().storage,
            StorageEncoding::NumericFp8E4m3fnBf16Scales
        );
        assert_eq!(
            plan.support().admission,
            AdmissionBoundary::OwnerLiveAdmissionRequired
        );
        assert_eq!(
            plan.support().buildability,
            BuildabilityStatus::CatalogValidated
        );
        assert_eq!(plan.observability().max_layers, 40);
        assert_eq!(plan.effective_options().max_tensor_mebibytes, 2048);
        let placement = plan.topology_placement();
        assert_eq!(placement.rank_backend, Some(PipelineRankBackend::Thread));
        assert_eq!(
            (
                placement.logical.pipeline_parallel,
                placement.logical.tensor_parallel,
                placement.logical.expert_parallel
            ),
            (case.pp, case.tp, case.ep)
        );
        assert_eq!(placement.physical.devices, Some(case.devices.to_vec()));
        assert!(placement.physical.source_identity.is_none());
        assert!(plan.qwen35_expert_placement().is_none());
    } else {
        assert!(
            plan.is_err(),
            "CUDA serving must not fall back to a CPU owner"
        );
    }
}

#[test]
fn pp2_tp1_ep2_full40_metadata_contract() {
    check_case(&PP2_TP1_EP2);
}
#[test]
fn pp2_tp1_ep4_full40_metadata_contract() {
    check_case(&PP2_TP1_EP4);
}
#[test]
fn pp1_tp2_ep2_full40_metadata_contract() {
    check_case(&PP1_TP2_EP2);
}
#[test]
fn pp2_tp2_ep2_full40_metadata_contract() {
    check_case(&PP2_TP2_EP2);
}
#[test]
fn pp2_tp4_ep2_metadata_does_not_claim_hardware_acceptance() {
    check_case(&PP2_TP4_EP2);
}

#[test]
fn arithmetic_and_quality_evidence_are_not_interchangeable() {
    let spec = spec();
    for (case, expected) in [
        (
            &PP1_TP2_EP2,
            HybridExecutionCapability::F32WithFullModelStrictEvidence,
        ),
        (
            &PP2_TP2_EP2,
            HybridExecutionCapability::F32WithFullModelStrictEvidence,
        ),
        (&PP2_TP4_EP2, HybridExecutionCapability::F32FactoryOnly),
    ] {
        let topology = topology(&spec, case);
        let adapter = Qwen35CompositionAdapter::new(&topology);
        assert_eq!(
            adapter
                .execution_capability(profile(case, NumericFp8Profile::F32))
                .unwrap(),
            expected
        );
        adapter
            .validate_production_execution(profile(case, NumericFp8Profile::F32))
            .unwrap();
        let w8 = profile(case, NumericFp8Profile::W8A16Bf16);
        assert_eq!(
            adapter.execution_capability(w8).unwrap(),
            HybridExecutionCapability::W8A16QualityUnaccepted
        );
        let error = adapter.validate_production_execution(w8).unwrap_err();
        assert_eq!(
            ferrule_common::error::ErrorChain::new(&error).downcast_ref::<HybridCapabilityError>(),
            Some(&HybridCapabilityError::W8A16QualityUnaccepted)
        );
        if cfg!(feature = "cuda") {
            let plan = prepare(case.options(), Some(NumericFp8Profile::W8A16Bf16)).unwrap();
            assert_eq!(
                plan.numeric_fp8_profile(),
                Some(NumericFp8Profile::W8A16Bf16)
            );
            assert_eq!(
                plan.support().arithmetic,
                ArithmeticPrecision::W8A16Bf16F32Accumulate
            );
        }
    }
}

#[test]
fn source_and_external_owner_roster_mismatches_fail_closed() {
    let spec = spec();
    let original = topology(&spec, &PP2_TP2_EP2);
    // A real TP member is valid in the model-neutral API, but not the catalog's
    // canonical first-lane source. The rejection must inspect the supplied plan.
    for change_source in [true, false] {
        let mut stages = original.stages().to_vec();
        let first = &stages[0];
        let mut attachment = first.expert_attachment().unwrap().clone();
        if change_source {
            attachment.source_rank = rank(1);
        } else {
            attachment.members.swap(0, 1);
        }
        stages[0] = HybridParallelPlan::new(
            &spec,
            0..20,
            first.scope(),
            first.tensor_members().to_vec(),
            Some(attachment),
        )
        .unwrap();
        let invalid = HybridExpertTopology::new(&spec, stages).unwrap();
        assert_eq!(
            invalid.validate_external_expert_roster(PP2_TP2_EP2.parallelism()),
            Err(HybridTopologyError::ExpertOwners)
        );
        let error = Qwen35CompositionAdapter::new(&invalid)
            .execution_capability(profile(&PP2_TP2_EP2, NumericFp8Profile::F32))
            .unwrap_err();
        assert_eq!(
            ferrule_common::error::ErrorChain::new(&error).downcast_ref::<HybridCapabilityError>(),
            Some(&HybridCapabilityError::ExpertOwners)
        );
    }
    let first = &original.stages()[0];
    let mut attachment = first.expert_attachment().unwrap().clone();
    attachment.source_rank = rank(4); // EP child cannot authenticate as a TP source.
    assert!(
        HybridParallelPlan::new(
            &spec,
            0..20,
            first.scope(),
            first.tensor_members().to_vec(),
            Some(attachment)
        )
        .is_err()
    );
}

#[test]
fn colocated_roots_and_experts_charge_one_physical_capacity_sample() {
    // Explicit synthetic byte ledger, not a claim about private CUDA allocator grants.
    let mut owners = Vec::new();
    for _ in 0..4 {
        owners.push(HybridStageBudget {
            device: 7,
            weights: 100,
            state: 10,
            kv: 20,
            workspace: 30,
            margin: 5,
        });
    }
    for _ in 0..4 {
        owners.push(HybridStageBudget {
            device: 7,
            weights: 40,
            state: 0,
            kv: 0,
            workspace: 10,
            margin: 5,
        });
    }
    assert_eq!(
        HybridDeviceBudget::admit(&owners, &[(7, 880)]).unwrap(),
        vec![HybridDeviceBudget {
            device: 7,
            required: 880,
            capacity: 880
        }]
    );
    for samples in [
        vec![(7, 879)],
        vec![(7, 880), (7, 880)],
        vec![],
        vec![(7, 880), (8, 1)],
    ] {
        assert!(HybridDeviceBudget::admit(&owners, &samples).is_err());
    }
    owners[0].weights = u64::MAX;
    assert!(HybridDeviceBudget::admit(&owners, &[(7, u64::MAX)]).is_err());
}

#[test]
fn unsupported_transport_degrees_and_placement_do_not_get_a_plan() {
    let spec = spec();
    let topology = topology(&spec, &PP2_TP2_EP2);
    let adapter = Qwen35CompositionAdapter::new(&topology);
    let mut process = profile(&PP2_TP2_EP2, NumericFp8Profile::F32);
    process.rank_backend = PipelineRankBackend::Process;
    let error = adapter.execution_capability(process).unwrap_err();
    assert_eq!(
        ferrule_common::error::ErrorChain::new(&error).downcast_ref::<HybridCapabilityError>(),
        Some(&HybridCapabilityError::RankBackend)
    );
    for mutate in [
        (|p: &mut ParallelismPlan| p.pipeline_parallel = 3) as fn(&mut ParallelismPlan),
        |p| p.tensor_parallel = 3,
        |p| p.expert_parallel = 3,
        |p| p.data_parallel = 2,
        |p| p.sequence_parallel = 2,
        |p| p.context_parallel = 2,
    ] {
        let mut p = profile(&PP2_TP2_EP2, NumericFp8Profile::F32);
        mutate(&mut p.parallelism);
        assert!(adapter.execution_capability(p).is_err());
        let mut input = PP2_TP2_EP2.options();
        input.parallelism = p.parallelism;
        assert!(prepare(input, None).is_err());
    }
    for case in [&PP2_TP1_EP2, &PP2_TP1_EP4, &PP1_TP2_EP2, &PP2_TP2_EP2] {
        let mut process = case.options();
        process.rank_backend = PipelineRankBackend::Process;
        #[cfg(unix)]
        {
            process.process_launch = Some(ferrule_runtime::parallel::process::ProcessLaunch::new(
                std::env::current_exe().unwrap(),
            ));
        }
        assert!(prepare(process, None).is_err());
        let mut short = case.options();
        short.devices.as_mut().unwrap().pop();
        assert!(prepare(short, None).is_err());
    }
}

#[test]
fn resident_ep_thread_and_process_keep_root_identity_and_distinct_catalog_kind() {
    for transport in [PipelineRankBackend::Thread, PipelineRankBackend::Process] {
        if transport == PipelineRankBackend::Process && !cfg!(unix) {
            continue;
        }
        let mut parallel = PipelineBuildOptions {
            parallelism: ParallelismPlan {
                expert_parallel: 8,
                ..Default::default()
            },
            rank_backend: transport,
            devices: Some(vec![3, 2, 1, 0, 7, 6, 5, 4]),
            ..Default::default()
        };
        #[cfg(unix)]
        if transport == PipelineRankBackend::Process {
            parallel.process_launch = Some(ferrule_runtime::parallel::process::ProcessLaunch::new(
                std::env::current_exe().unwrap(),
            ));
        }
        let result = ResidentModelPlanner.prepare_serving(
            &config(ModelFamily::Qwen35Moe),
            BackendSelection::Cuda,
            None,
            options(None),
            ServingModelOptions {
                parallel: Some(parallel),
                ..Default::default()
            },
        );
        if cfg!(feature = "cuda") {
            let plan = result.unwrap();
            assert_eq!(
                plan.selected_implementation().kind,
                SelectedImplementationKind::Resident
            );
            let placement = plan.topology_placement();
            assert_eq!(placement.rank_backend, Some(transport));
            assert_eq!(placement.logical.expert_parallel, 8);
            assert_eq!(placement.physical.root_device, Some(3));
            assert_eq!(placement.physical.source_identity, Some(rank(0)));
            assert_eq!(
                placement.physical.expert_devices,
                Some(vec![3, 2, 1, 0, 7, 6, 5, 4])
            );
            assert_eq!(
                plan.qwen35_expert_placement().unwrap().rank_backend(),
                transport
            );
            assert!(plan.pipeline_options().is_none());
        } else {
            assert!(result.is_err());
        }
    }
}

#[test]
fn ordinary_dense_pipeline_does_not_inherit_numeric_moe_contracts() {
    let plan = ResidentModelPlanner
        .prepare_serving(
            &config(ModelFamily::Qwen3),
            BackendSelection::Cpu,
            None,
            options(None),
            ServingModelOptions {
                parallel: Some(PipelineBuildOptions {
                    parallelism: ParallelismPlan {
                        pipeline_parallel: 2,
                        ..Default::default()
                    },
                    ..Default::default()
                }),
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
        ArithmeticPrecision::Bf16Compatibility
    );
    assert_eq!(plan.support().storage, StorageEncoding::ProfileDefined);
    assert_eq!(plan.numeric_fp8_profile(), None);
    assert_eq!(plan.topology_placement().logical.expert_parallel, 1);
    assert_eq!(plan.topology_placement().physical.expert_devices, None);
    assert!(plan.qwen35_expert_placement().is_none());
}
