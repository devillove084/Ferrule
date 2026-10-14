#[cfg(feature = "cuda")]
use super::assembly::start_admitted as executor;
use super::budget::{ExpertBudgetRepresentation, expert_budgets, grant, wire_bytes};
use super::*;
use crate::model_factory::qwen35::Qwen35ResolvedAdapter;
use crate::model_factory::qwen35_composition::{HybridExecutionProfile, Qwen35CompositionAdapter};
use crate::model_factory::*;
use ferrule_model::ModelExecutionBackend;
#[cfg(feature = "cuda")]
use ferrule_model::transformer::CudaHybridRankMemory;
use ferrule_runtime::parallel::pipeline::PhysicalKvLayout;
#[path = "parallel_tensor_fixture.rs"]
pub(super) mod fixture;
use crate::model_factory::qwen35_composition::{HybridCapabilityError, HybridExecutionCapability};
use ferrule_model::models::qwen35::Qwen35Profile;
use fixture::Fixture;

fn execution_profile(r: &ResolvedModelRequest) -> HybridExecutionProfile {
    let pipeline = r.pipeline.as_ref().unwrap();
    HybridExecutionProfile {
        model: Qwen35Profile::Moe35BA3Bfp8,
        backend: r.backend,
        rank_backend: pipeline.rank_backend,
        parallelism: pipeline.parallelism,
        numeric: r
            .family_options
            .qwen35()
            .unwrap()
            .moe()
            .unwrap()
            .numeric_profile,
    }
}
fn assert_capability_error(error: ferrule_common::Error, expected: HybridCapabilityError) {
    assert_eq!(
        ferrule_common::error::ErrorChain::new(&error).downcast_ref::<HybridCapabilityError>(),
        Some(&expected),
    );
}

#[test]
fn tp_ep_capability_matches_factory_without_promoting_tp4_or_w8_quality() {
    let f = Fixture::new();
    for pp in [1, 2] {
        for tp in [2, 4] {
            for numeric in [NumericFp8Profile::F32, NumericFp8Profile::W8A16Bf16] {
                let mut r = request(&f, pp, tp);
                let Qwen35ResolvedAdapter::Moe { options, .. } =
                    r.family_options.qwen35_mut().unwrap()
                else {
                    unreachable!()
                };
                options.numeric_profile = numeric;
                let plan =
                    crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).unwrap();
                let profile = execution_profile(&r);
                let expected = match (numeric, tp) {
                    (NumericFp8Profile::F32, 2) => {
                        HybridExecutionCapability::F32WithFullModelStrictEvidence
                    }
                    (NumericFp8Profile::F32, _) => HybridExecutionCapability::F32FactoryOnly,
                    (NumericFp8Profile::W8A16Bf16, _) => {
                        HybridExecutionCapability::W8A16QualityUnaccepted
                    }
                };
                assert_eq!(
                    Qwen35CompositionAdapter::new(&plan.layout)
                        .execution_capability(profile)
                        .unwrap(),
                    expected
                );
                if numeric == NumericFp8Profile::F32 {
                    Qwen35CompositionAdapter::new(&plan.layout)
                        .validate_production_execution(profile)
                        .unwrap();
                } else {
                    assert_capability_error(
                        Qwen35CompositionAdapter::new(&plan.layout)
                            .validate_production_execution(profile)
                            .unwrap_err(),
                        HybridCapabilityError::W8A16QualityUnaccepted,
                    );
                }
            }
        }
    }
}

#[test]
fn tp_ep_capability_and_factory_reject_unsupported_execution_profiles() {
    let f = Fixture::new();
    let base = request(&f, 2, 2);
    let plan = crate::model_factory::qwen35::binding::pipeline_plan(&base, &f.resources).unwrap();
    let invalid: [fn(&mut ResolvedModelRequest); 10] = [
        |r| r.backend = ModelExecutionBackend::Cpu,
        |r| r.pipeline.as_mut().unwrap().rank_backend = PipelineRankBackend::Process,
        |r| r.pipeline.as_mut().unwrap().parallelism.pipeline_parallel = 3,
        |r| r.pipeline.as_mut().unwrap().parallelism.tensor_parallel = 1,
        |r| r.pipeline.as_mut().unwrap().parallelism.tensor_parallel = 3,
        |r| r.pipeline.as_mut().unwrap().parallelism.tensor_parallel = usize::MAX,
        |r| r.pipeline.as_mut().unwrap().parallelism.expert_parallel = 4,
        |r| r.pipeline.as_mut().unwrap().parallelism.data_parallel = 2,
        |r| r.pipeline.as_mut().unwrap().parallelism.sequence_parallel = 2,
        |r| r.pipeline.as_mut().unwrap().parallelism.context_parallel = 2,
    ];
    for (index, mutate) in invalid.into_iter().enumerate() {
        let mut r = request(&f, 2, 2);
        mutate(&mut r);
        assert!(crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).is_err());
        let expected = match index {
            0 => HybridCapabilityError::Backend,
            1 => HybridCapabilityError::RankBackend,
            _ => HybridCapabilityError::Parallelism,
        };
        assert_capability_error(
            Qwen35CompositionAdapter::new(&plan.layout)
                .validate_production_execution(execution_profile(&r))
                .unwrap_err(),
            expected,
        );
    }
    let mut profile = execution_profile(&base);
    profile.model = Qwen35Profile::Dense08Bbf16;
    assert_capability_error(
        Qwen35CompositionAdapter::new(&plan.layout)
            .validate_production_execution(profile)
            .unwrap_err(),
        HybridCapabilityError::ModelProfile,
    );
    let mut r = request(&f, 2, 2);
    r.family = ModelFamily::Qwen35;
    assert!(crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).is_err());
}

#[test]
fn tp_ep_capability_rejects_generic_sparse_kv_and_nonfactory_expert_scopes() {
    use ferrule_model::transformer::expert_parallel::ExpertPlacement;
    let f = Fixture::new();
    let r = request(&f, 2, 2);
    let plan = crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).unwrap();
    // Each mutation remains valid general/reference geometry, but not the
    // factory's KV cohort, external EP roster or per-stage CUDA KV schema.
    for (case, expected) in [
        (0, HybridCapabilityError::KvOwners),
        (1, HybridCapabilityError::KvOwners),
        (2, HybridCapabilityError::ExpertOwners),
        (3, HybridCapabilityError::ExpertOwners),
        (4, HybridCapabilityError::ExpertOwners),
        (5, HybridCapabilityError::KvOwners),
    ] {
        let stages = plan
            .layout
            .stages()
            .iter()
            .map(|original| {
                let mut roots = original.tensor_members().to_vec();
                let mut ep = original.expert_attachment().unwrap().clone();
                let mut scope = original.scope();
                let layers = original.schema().layer_range();
                match case {
                    0 => {
                        roots = roots
                            .iter()
                            .map(|r| ParallelRankId::new(r.get() + 20))
                            .collect();
                        ep.source_rank = roots[0];
                    }
                    1 => roots.reverse(),
                    2 => ep.source_rank = roots[1],
                    3 => {
                        ep.members = ep
                            .members
                            .iter()
                            .map(|r| ParallelRankId::new(r.get() + 20))
                            .collect()
                    }
                    4 => ep.members[0] = roots[0],
                    5 => scope.replica = 1,

                    _ => unreachable!(),
                }
                ep.placement = ExpertPlacement::new(layers.clone().flat_map(|layer| {
                    let members = &ep.members;
                    (0..4).map(move |expert| (layer, expert, members[expert % 2]))
                }))
                .unwrap();
                HybridParallelPlan::new(f.resources.spec(), layers, scope, roots, Some(ep)).unwrap()
            })
            .collect();
        let layout = HybridExpertTopology::new(f.resources.spec(), stages).unwrap();
        assert_capability_error(
            Qwen35CompositionAdapter::new(&layout)
                .validate_production_execution(execution_profile(&r))
                .unwrap_err(),
            expected,
        );
    }
    // A recurrent-only stage is rejected by HybridParallelPlan itself; it must
    // not be widened merely to reach the CUDA capability query.
    let original = plan.layout.stages()[0].clone();
    let mut recurrent = original.expert_attachment().unwrap().clone();
    recurrent.placement =
        ExpertPlacement::new((0..4).map(|expert| (0, expert, recurrent.members[expert % 2])))
            .unwrap();
    assert!(
        HybridParallelPlan::new(
            f.resources.spec(),
            0..1,
            original.scope(),
            original.tensor_members().to_vec(),
            Some(recurrent),
        )
        .is_err()
    );
}

fn request(f: &Fixture, pp: usize, tp: usize) -> ResolvedModelRequest {
    let mut r = f.request();
    r.max_layers = 4;
    r.model_info.num_layers = 4;
    r.model_info.hidden_size = 8;
    r.model_info.vocab_size = 11;
    r.model_info.num_experts = 4;
    if let Qwen35ResolvedAdapter::Moe {
        expert_placement, ..
    } = r.family_options.qwen35_mut().unwrap()
    {
        *expert_placement = None;
    }
    r.pipeline = Some(PipelineBuildOptions {
        parallelism: ferrule_common::ParallelismPlan {
            pipeline_parallel: pp,
            tensor_parallel: tp,
            expert_parallel: 2,
            ..Default::default()
        },
        devices: Some((0..pp * (tp + 2)).map(|i| i % 2).collect()),
        ..Default::default()
    });
    r
}

#[test]
fn tp_ep_admission_scoped_owners_replicated_kv_and_full_vocab_wire() {
    use ferrule_common::execution::KvLayoutSchema;
    let f = Fixture::new();
    for pp in [1, 2] {
        for tp in [2, 4] {
            let r = request(&f, pp, tp);
            let p = crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).unwrap();
            assert_eq!(p.profile(), NumericFp8Profile::F32);
            assert_eq!(p.segments.len(), pp);
            let topology = ferrule_common::ValidatedParallelTopology::new(
                ParallelTopologyId::new(1),
                (pp * tp) as u32,
                ParallelRankId::new(0),
                r.pipeline.as_ref().unwrap().parallelism,
            )
            .unwrap();
            assert_eq!(topology.world_size() as usize, pp * tp);
            assert_eq!(p.layout.physical_members().len(), pp * (tp + 2));
            let global = HybridStateSchema::from_spec(f.resources.spec(), 4)
                .unwrap()
                .kv_planes(16, 32)
                .unwrap()
                .checked_page_bytes()
                .unwrap();
            assert_eq!(
                p.kv.configured_bytes.unwrap(),
                (global * tp * p.kv.configured_pages) as u64
            );
            assert_eq!(
                p.kv.full_capacity_pages * 2,
                p.config()
                    .physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cuda))
                    .unwrap()
            );
            for (stage, owner) in p.owners.iter().enumerate() {
                assert_eq!(owner.group.source_rank.get() as usize, stage * tp);
                assert_eq!(owner.group.layers, stage * (4 / pp)..(stage + 1) * (4 / pp));
                assert_eq!(
                    owner
                        .group
                        .members
                        .iter()
                        .map(|r| r.get() as usize)
                        .collect::<Vec<_>>(),
                    [pp * tp + stage * 2, pp * tp + stage * 2 + 1]
                );
            }
            assert!(
                wire_bytes(32, 2048, 248320, &p.owners, pp * tp).unwrap()
                    >= (32 * 248320 * 4 * 8 * pp * tp) as u64
            );
            assert!(wire_bytes(usize::MAX, 2048, 248320, &p.owners, pp * tp).is_err());
            assert!(p.host().memory_budget.base_resident_bytes > p.wire_bytes);
            let service = ferrule_runtime::parallel::collective::HostCollectiveGroup::new(
                ParallelTopologyId::new(1),
                ferrule_common::ParallelGroupId::new(1),
                p.layout.stages()[0].tensor_members().to_vec(),
                p.collective,
            )
            .unwrap();
            let peak = service
                .memory_for(
                    ferrule_runtime::parallel::collective::HostCollectiveKind::AllGatherF32,
                    p.collective.max_elements_per_rank,
                )
                .unwrap();
            assert_eq!(peak.peak_bytes, p.collective.max_host_bytes);
            let child = expert_budgets(
                &f.resources,
                &p,
                ExpertBudgetRepresentation::PhysicalAllocationGrants,
            )
            .unwrap();
            assert_eq!(child.len(), 2 * pp);
            assert!(
                child
                    .iter()
                    .all(|b| b.weights > p.owners[0].weights[0] as u64
                        && b.state == 0
                        && b.kv == 0)
            );
        }
    }
}

#[test]
fn tp_ep_admission_rejects_process_stale_seal_and_insufficient_kv() {
    let f = Fixture::new();
    for mutate in [
        |r: &mut ResolvedModelRequest| {
            r.pipeline.as_mut().unwrap().rank_backend = PipelineRankBackend::Process
        },
        |r: &mut ResolvedModelRequest| r.pipeline.as_mut().unwrap().parallelism.tensor_parallel = 3,
        |r: &mut ResolvedModelRequest| {
            r.pipeline.as_mut().unwrap().parallelism.pipeline_parallel = 3
        },
        |r: &mut ResolvedModelRequest| r.pipeline.as_mut().unwrap().parallelism.expert_parallel = 4,
        |r: &mut ResolvedModelRequest| r.pipeline.as_mut().unwrap().devices = Some(vec![0; 16]),
        |r: &mut ResolvedModelRequest| r.pipeline.as_mut().unwrap().devices = None,
        |r: &mut ResolvedModelRequest| r.pipeline.as_mut().unwrap().rank_restarts = 1,
        |r: &mut ResolvedModelRequest| r.backend = ModelExecutionBackend::Cpu,
        |r: &mut ResolvedModelRequest| r.max_layers = 2,
        |r: &mut ResolvedModelRequest| r.kv_cache_bytes = Some(1),
    ] {
        let mut r = request(&f, 2, 2);
        mutate(&mut r);
        assert!(crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).is_err());
    }
    let mut r = request(&f, 2, 2);
    let p = crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).unwrap();
    r.kv_cache_bytes = Some(p.kv.configured_bytes.unwrap());
    assert_eq!(
        crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources)
            .unwrap()
            .kv
            .configured_pages,
        p.kv.configured_pages
    );
    r.kv_cache_bytes = Some(p.kv.configured_bytes.unwrap() - 1);
    assert!(crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).is_err());
    r.kv_cache_bytes = None;
    std::fs::write(f.dir.join("weights.bin"), b"changed seal").unwrap();
    assert!(crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).is_err());
}

#[test]
fn tp_ep_physical_grants_round_every_expert_allocation_and_aggregate_colocation() {
    let f = Fixture::new();
    let mut r = request(&f, 2, 2);
    r.pipeline.as_mut().unwrap().devices = Some(vec![0; 8]);
    let p = crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).unwrap();
    let mut budgets = expert_budgets(
        &f.resources,
        &p,
        ExpertBudgetRepresentation::PhysicalAllocationGrants,
    )
    .unwrap();
    // Each owner has four experts, each three separately rounded weight+scale pairs.
    assert!(budgets.iter().all(|b| b.weights == 4 * 3 * 2 * 256));
    for _ in 0..4 {
        budgets.push(HybridStageBudget {
            device: 0,
            weights: 100,
            state: 200,
            kv: 300,
            workspace: 400,
            margin: MARGIN,
        });
    }
    let total = budgets.iter().map(|b| b.total().unwrap()).sum::<u64>();
    assert_eq!(
        HybridDeviceBudget::admit(&budgets, &[(0, total)]).unwrap()[0].required,
        total
    );
    assert!(HybridDeviceBudget::admit(&budgets, &[(0, total - 1)]).is_err());
    assert!(grant(u64::MAX).is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn tp_ep_cuda_admission_hidden_devices_seals_profile_state_and_kv_grants() {
    let f = Fixture::new();
    for pp in [1, 2] {
        for tp in [2, 4] {
            for profile in [NumericFp8Profile::F32, NumericFp8Profile::W8A16Bf16] {
                let mut r = request(&f, pp, tp);
                if let Qwen35ResolvedAdapter::Moe { options, .. } =
                    r.family_options.qwen35_mut().unwrap()
                {
                    options.numeric_profile = profile;
                }
                let a = crate::model_factory::qwen35::binding::pipeline_admission(&r, &f.resources)
                    .unwrap();
                assert_eq!(a.plan.profile(), profile);
                assert_eq!(
                    a.options.hybrid_cuda_numeric_fp8_precision(),
                    Some(profile.precision())
                );
                assert_eq!(a.budgets.len(), pp * (tp + 2));
                assert!(
                    a.budgets[pp * tp..]
                        .iter()
                        .all(|b| b.kv == 0 && b.state == 0)
                );
                let mut kv_grants = 0;
                for (stage, parallel) in a.plan.layout.stages().iter().enumerate() {
                    for lane in 0..tp {
                        let m = CudaHybridRankMemory::for_rank(
                            &f.resources,
                            &a.plan.segments[stage],
                            parallel,
                            &parallel.rank_layout(lane).unwrap(),
                            &a.options,
                            a.plan.kv.configured_pages,
                        )
                        .unwrap();
                        let b = &a.budgets[stage * tp + lane];
                        assert_eq!(
                            b.state as usize,
                            (2 * a.plan.config().session_capacity + 1)
                                * m.per_sequence_state_grants
                        );
                        assert_eq!(
                            b.total().unwrap() - b.margin,
                            m.total_device_grants(a.plan.config().session_capacity)
                                .unwrap() as u64
                        );
                        kv_grants += b.kv;
                    }
                }
                assert!(kv_grants >= a.plan.kv.configured_bytes.unwrap());
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires two CUDA GPUs; real TP1/2/4 numeric roots and stage EP workers"]
fn tp_ep_gpu_tiny_matches_tp1_ep_full_logits_and_custody() {
    use ferrule_common::execution::ForwardPhase;
    use ferrule_model::runner::{MultiSessionRunner, ResidentModelRunner};
    use ferrule_runtime::SessionId;
    for profile in [NumericFp8Profile::F32, NumericFp8Profile::W8A16Bf16] {
        let f = Fixture::new();
        for (degree, tp) in [(1, 2), (2, 2), (1, 4), (2, 4)] {
            let mut r = request(&f, degree, tp);
            if let Qwen35ResolvedAdapter::Moe { options, .. } =
                r.family_options.qwen35_mut().unwrap()
            {
                options.numeric_profile = profile;
            }
            let mut reference = r.clone();
            reference.pipeline = None;
            reference
                .family_options
                .qwen35_mut()
                .unwrap()
                .set_expert_placement(Qwen35ExpertPlacement::new(2, vec![0, 1]).unwrap());
            let mut root = crate::model_factory::qwen35::ep::prepare_bound(
                &reference,
                f.resources.clone(),
                TokenizerHandle::load(&f.dir).unwrap(),
            )
            .unwrap();
            let mut state = root.runner.create_sequence_state().unwrap();
            let mut pages = vec![Vec::new()];
            let mut expected = Vec::new();
            for (index, (tokens, phase)) in [
                (vec![1, 3, 2], ForwardPhase::Prefill),
                (vec![5], ForwardPhase::Decode),
                (vec![7], ForwardPhase::Decode),
            ]
            .into_iter()
            .enumerate()
            {
                expected.push(crate::model_factory::qwen35::ep::strict::full_logits_step(
                    &mut root.runner,
                    std::slice::from_mut(&mut state),
                    &mut pages,
                    &tokens,
                    phase,
                    index as u64 + 1,
                ));
            }
            root.runner.try_release_sequence_state(state).unwrap();
            root.runner.shutdown_physical().unwrap();
            drop(root);
            let a = crate::model_factory::qwen35::binding::pipeline_admission(&r, &f.resources)
                .unwrap();
            a.admit_live(&f.resources).unwrap();
            let bytes = a.plan.kv.configured_bytes.unwrap();
            let mut pp = executor(&r, f.resources.clone(), a).unwrap();
            assert_eq!(pp.physical_kv_bytes().unwrap(), bytes);
            for (index, (tokens, phase)) in [
                (vec![1, 3, 2], ForwardPhase::Prefill),
                (vec![5], ForwardPhase::Decode),
                (vec![7], ForwardPhase::Decode),
            ]
            .into_iter()
            .enumerate()
            {
                let actual = pp.forward(SessionId(17), &tokens, phase).unwrap();
                for (row, expected) in expected[index].iter().enumerate() {
                    let actual = actual.logits.row(row).unwrap();
                    assert_eq!(actual.len(), expected.len());
                    for (&a, &b) in actual.iter().zip(expected) {
                        assert!(
                            a.is_finite()
                                && b.is_finite()
                                && (a - b).abs() <= 3e-4 + 3e-4 * b.abs(),
                            "profile={profile:?}: {a} != {b}"
                        );
                    }
                }
            }
            pp.fork_session(SessionId(17), SessionId(29)).unwrap();
            let cancelled = std::sync::atomic::AtomicBool::new(true);
            assert!(
                pp.forward_cancellable(SessionId(17), &[3], ForwardPhase::Decode, &cancelled)
                    .is_err()
            );
            assert_eq!(pp.coordinator().publication_count(), 3);
            let left = pp
                .forward(SessionId(17), &[3], ForwardPhase::Decode)
                .unwrap();
            let right = pp
                .forward(SessionId(29), &[3], ForwardPhase::Decode)
                .unwrap();
            assert_eq!(left.logits.row(0).unwrap(), right.logits.row(0).unwrap());
            for session in [SessionId(17), SessionId(29)] {
                pp.release_session(session).unwrap();
            }
            assert_eq!(pp.page_manager().allocated_pages(), 0);
            pp.shutdown().unwrap();
        }
    }
}

#[cfg(feature = "cuda")]
fn full35b(pp: usize, profile: Option<NumericFp8Profile>) {
    use ferrule_runtime::{GenerateRequest, InferenceShutdownProgress, RequestId, SessionId};
    let path = std::env::var("FERRULE_QWEN35_TP_EP_MODEL")
        .expect("set strict full35B checkpoint directory");
    let devices = std::env::var("FERRULE_QWEN35_TP_EP_DEVICES")
        .expect("set roots-then-stage-EP CUDA ordinals")
        .split(',')
        .map(|s| s.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        devices.len(),
        pp * 4,
        "PP1 TP2 EP2 needs 4 owners; PP2 needs 8"
    );
    let config = ferrule_model::AutoConfig::from_pretrained(&path).unwrap();
    let mut input = crate::model_factory::tests::options();
    input.max_tensor_mebibytes = 2048;
    input.numeric_fp8_profile = profile;
    input.driver_config.enable_native_proposals = false;
    input.driver_config.ctx_size = 32;
    input.scheduler_config.max_active_sequences = 2;
    input.scheduler_config.max_batch_tokens = 4;
    input.scheduler_config.prefill_chunk_size = 4;
    let plan = ResidentModelPlanner
        .prepare_serving(
            &config,
            BackendSelection::Cuda,
            None,
            input,
            ServingModelOptions {
                // Replicated BF16 embedding expands to F32; the 1024 MiB default
                // cannot cover it. Serving intent must be explicit or the resolver
                // restores that default over ModelFactoryOptions.
                max_tensor_mebibytes: Some(2048),
                engine: ServingEngineSelection::Pipeline,
                parallel: Some(PipelineBuildOptions {
                    parallelism: ferrule_common::ParallelismPlan {
                        pipeline_parallel: pp,
                        tensor_parallel: 2,
                        expert_parallel: 2,
                        ..Default::default()
                    },
                    devices: Some(devices),
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
    let expected: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> =
        build_hybrid_pipeline;
    assert!(std::ptr::fn_addr_eq(
        plan.request.execution.construction.builder(),
        expected
    ));
    assert_eq!(
        plan.numeric_fp8_profile(),
        Some(profile.unwrap_or_default())
    );
    assert_eq!(plan.observability().max_layers, 40);
    assert_eq!(plan.request.max_tensor_bytes, 2048 << 20);
    let tokenizer = TokenizerHandle::load(std::path::Path::new(&path)).unwrap();
    let tokens = tokenizer.encode("Hello").unwrap();
    let mut engine = plan.build().unwrap();
    engine.retain_session(SessionId(1)).unwrap();
    engine
        .try_submit(GenerateRequest {
            id: RequestId(1),
            session_id: Some(SessionId(1)),
            prompt_tokens: tokens,
            max_new_tokens: 2,
            stop: vec![],
            ignore_eos: true,
        })
        .unwrap();
    let mut emitted = 0;
    let mut finished = false;
    for _ in 0..128 {
        engine
            .step(&mut |_| {
                emitted += 1;
                Ok(())
            })
            .unwrap();
        if engine.take_request_terminal(RequestId(1)).is_some() {
            finished = true;
            break;
        }
    }
    assert!(
        finished && emitted == 2,
        "full35B real pipeline must generate two tokens"
    );
    assert!(engine.retained_session_position(SessionId(1)).unwrap() > 0);
    engine.reset_session(SessionId(1)).unwrap();
    engine
        .try_submit(GenerateRequest {
            id: RequestId(2),
            session_id: Some(SessionId(2)),
            prompt_tokens: vec![1],
            max_new_tokens: 2,
            stop: vec![],
            ignore_eos: true,
        })
        .unwrap();
    engine.cancel_request(RequestId(2)).unwrap();
    for _ in 0..128 {
        if matches!(
            engine.shutdown().unwrap(),
            InferenceShutdownProgress::Complete
        ) {
            return;
        }
        std::thread::yield_now();
    }
    panic!("full35B PPxEP shutdown failed to retire custody");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires strict35B checkpoint and explicit FERRULE_QWEN35_TP_EP_MODEL/DEVICES; real GPU factory, no CPU fallback"]
fn tp_ep_gpu_full35b_public_factory_pp1_f32_default() {
    full35b(1, None);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires strict35B checkpoint and explicit FERRULE_QWEN35_TP_EP_MODEL/DEVICES; real GPU factory, no CPU fallback"]
fn tp_ep_gpu_full35b_public_factory_pp1_w8a16_opt_in() {
    full35b(1, Some(NumericFp8Profile::W8A16Bf16));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires strict35B checkpoint and explicit FERRULE_QWEN35_TP_EP_MODEL/DEVICES; real GPU factory, no CPU fallback"]
fn tp_ep_gpu_full35b_public_factory_pp2_f32_default() {
    full35b(2, None);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires strict35B checkpoint and explicit FERRULE_QWEN35_TP_EP_MODEL/DEVICES; real GPU factory, no CPU fallback"]
fn tp_ep_gpu_full35b_public_factory_pp2_w8a16_opt_in() {
    full35b(2, Some(NumericFp8Profile::W8A16Bf16));
}

#[test]
fn tp_ep_serving_tensor_budget_requires_explicit_intent_without_changing_default() {
    for (explicit, expected) in [(None, 1024), (Some(1024), 1024), (Some(2048), 2048)] {
        let mut input = crate::model_factory::tests::options();
        input.max_tensor_mebibytes = 2048;
        let serving = ServingModelOptions {
            max_tensor_mebibytes: explicit,
            engine: ServingEngineSelection::Pipeline,
            parallel: Some(PipelineBuildOptions {
                parallelism: ferrule_common::ParallelismPlan {
                    tensor_parallel: 2,
                    expert_parallel: 2,
                    ..Default::default()
                },
                devices: Some(vec![0, 1, 2, 3]),
                ..Default::default()
            }),
            ..Default::default()
        };
        let resolved = qwen35_composition::moe_pipeline_options(
            &ModelFamily::Qwen35Moe,
            &serving,
            input,
            ResolvedConstruction::Pipeline(build_hybrid_pipeline),
        )
        .unwrap();
        assert_eq!(resolved.max_tensor_mebibytes, expected);
    }
}

#[cfg(feature = "cuda")]
#[path = "full40_strict/tp.rs"]
mod full40_strict;
