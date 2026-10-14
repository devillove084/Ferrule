#[cfg(feature = "cuda")]
use super::assembly::start_admitted as executor;
use super::budget::wire_bytes;
use super::*;
use crate::model_factory::qwen35::Qwen35ResolvedAdapter;
use crate::model_factory::qwen35::ep::tests::Fixture;
use crate::model_factory::*;
use ferrule_common::{ParallelTopologyId, ValidatedParallelTopology};
use ferrule_runtime::parallel::pipeline::{HybridDeviceBudget, HybridStageBudget};

fn request(f: &Fixture) -> ResolvedModelRequest {
    let mut r = f.request();
    r.max_layers = 4;
    r.model_info.num_layers = 4;
    r.model_info.hidden_size = 8;
    r.model_info.vocab_size = 11;
    r.model_info.num_experts = 4;
    if let Some(Qwen35ResolvedAdapter::Moe {
        expert_placement, ..
    }) = r.family_options.qwen35_mut()
    {
        *expert_placement = None;
    }
    r.pipeline = Some(PipelineBuildOptions {
        parallelism: ferrule_common::ParallelismPlan {
            pipeline_parallel: 2,
            expert_parallel: 2,
            ..Default::default()
        },
        devices: Some(vec![0, 1, 0, 1, 0, 1]),
        ..Default::default()
    });
    r
}

#[test]
fn pp_ep_admission_seals_global_ranges_actual_owners_and_full_vocab_wire() {
    let f = Fixture::with_layers(4);
    let r = request(&f);
    let p = crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).unwrap();
    assert_eq!(
        p.segments.iter().map(|p| p.layers()).collect::<Vec<_>>(),
        [0..2, 2..4]
    );
    assert_eq!(p.segment_layout.as_ref().unwrap().kv_layers(), 2);
    assert_eq!(
        p.config()
            .physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cuda))
            .unwrap(),
        p.kv.configured_pages
    );
    assert_eq!(p.devices, [0, 1, 0, 1, 0, 1]);
    assert_eq!(p.profile(), NumericFp8Profile::F32);
    for (stage, owners) in p.owners.iter().enumerate() {
        assert_eq!(owners.group.source_rank.get() as usize, stage);
        assert_eq!(owners.group.layers, stage * 2..stage * 2 + 2);
        assert_eq!(owners.counts, [4, 4]);
        assert_eq!(owners.weights, [4 * 294, 4 * 294]);
        assert_eq!(
            owners
                .group
                .members
                .iter()
                .map(|r| r.get())
                .collect::<Vec<_>>(),
            [2 + stage as u32 * 2, 3 + stage as u32 * 2]
        );
    }
    let topology = ValidatedParallelTopology::new(
        ParallelTopologyId::new(1),
        2,
        ParallelRankId::new(0),
        r.pipeline.as_ref().unwrap().parallelism,
    )
    .unwrap();
    assert_eq!(topology.world_size(), 2, "EP is not a PP mesh multiplier");
    let full = wire_bytes(32, 2048, 248320, &p.owners, 1).unwrap();
    assert!(full >= 32 * 248320 * 4 * 8);
    assert!(full > wire_bytes(32, 2048, 2048, &p.owners, 1).unwrap());
    assert!(wire_bytes(usize::MAX, 2048, 248320, &p.owners, 1).is_err());
    assert!(wire_bytes(32, 0, 248320, &p.owners, 1).is_err());
    assert!(p.host().memory_budget.base_resident_bytes > p.wire_bytes);
    let warm = HostExpertWarmPlan::new(
        f.resources.state_dict(),
        p.host(),
        p.config().max_parameter_bytes,
    )
    .unwrap();
    assert!(
        warm.check_available_bytes(p.host().memory_budget.base_resident_bytes)
            .is_err()
    );
    warm.check_available_bytes(u64::MAX).unwrap();
}

#[test]
fn pp_ep_admission_rejects_unconsumed_policies_and_stale_checkpoint() {
    let f = Fixture::with_layers(4);
    for mutate in [
        |r: &mut ResolvedModelRequest| {
            r.pipeline.as_mut().unwrap().rank_backend = PipelineRankBackend::Process
        },
        |r: &mut ResolvedModelRequest| r.pipeline.as_mut().unwrap().parallelism.tensor_parallel = 2,
        |r: &mut ResolvedModelRequest| {
            r.pipeline.as_mut().unwrap().devices = Some(vec![0, 1, 2, 3])
        },
        |r: &mut ResolvedModelRequest| r.max_layers = 2,
        |r: &mut ResolvedModelRequest| r.backend = ModelExecutionBackend::Cpu,
        |r: &mut ResolvedModelRequest| r.pipeline.as_mut().unwrap().rank_restarts = 1,
    ] {
        let mut r = request(&f);
        mutate(&mut r);
        assert!(crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).is_err());
    }
    let mut r = request(&f);
    if let Qwen35ResolvedAdapter::Moe { options, .. } = r.family_options.qwen35_mut().unwrap() {
        options.numeric_profile = NumericFp8Profile::W8A16Bf16;
    }
    assert_eq!(
        crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources)
            .unwrap()
            .profile(),
        NumericFp8Profile::W8A16Bf16
    );
    std::fs::write(f.dir.join("weights.bin"), b"changed after metadata seal").unwrap();
    assert!(crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).is_err());
}

#[test]
fn pp_ep_colocated_roots_and_all_children_consume_one_physical_card() {
    let f = Fixture::with_layers(4);
    let r = request(&f);
    let p = crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).unwrap();
    let mut budgets = (0..2)
        .map(|stage| {
            p.segment_layout
                .as_ref()
                .unwrap()
                .stage_budget(
                    stage,
                    p.config(),
                    0,
                    100,
                    200,
                    64,
                    ModelExecutionBackend::Cuda,
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    for owner in &p.owners {
        for &weights in &owner.weights {
            budgets.push(HybridStageBudget {
                device: 0,
                weights: weights as u64,
                state: 0,
                kv: 0,
                workspace: owner.activation_bytes as u64 + 64,
                margin: 64,
            });
        }
    }
    let sum = budgets.iter().map(|b| b.total().unwrap()).sum::<u64>();
    assert_eq!(
        HybridDeviceBudget::admit(&budgets, &[(0, sum)]).unwrap()[0].required,
        sum
    );
    assert!(HybridDeviceBudget::admit(&budgets, &[(0, sum - 1)]).is_err());
    assert!(HybridDeviceBudget::admit(&budgets, &[(0, sum), (1, sum)]).is_err());
    let mut groups = p.owners.iter().map(|o| o.group.clone()).collect::<Vec<_>>();
    groups[1].source_rank = ParallelRankId::new(0);
    assert!(HybridPipelineLayout::with_routed_experts(&f.resources, &p.segments, groups).is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn pp_ep_cuda_admission_without_devices_keeps_numeric_profile_and_exact_kv() {
    let f = Fixture::with_layers(4);
    let mut r = request(&f);
    for profile in [NumericFp8Profile::F32, NumericFp8Profile::W8A16Bf16] {
        if let Qwen35ResolvedAdapter::Moe { options, .. } = r.family_options.qwen35_mut().unwrap() {
            options.numeric_profile = profile;
        }
        let a =
            crate::model_factory::qwen35::binding::pipeline_admission(&r, &f.resources).unwrap();
        assert_eq!(
            a.options.hybrid_cuda_numeric_fp8_precision(),
            Some(profile.precision())
        );
        assert_eq!(a.budgets.len(), 6);
        assert_eq!(
            a.budgets.iter().map(|b| b.kv).sum::<u64>(),
            a.plan.kv.configured_bytes.unwrap()
        );
        assert!(a.budgets[2..].iter().all(|b| b.kv == 0 && b.state == 0));
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires two CUDA GPUs; real numeric root and four stage-scoped expert owners"]
fn pp_ep_gpu_tiny_matches_monolithic_ep_full_logits_and_custody() {
    use ferrule_common::execution::ForwardPhase;
    use ferrule_model::runner::{MultiSessionRunner, ResidentModelRunner};
    use ferrule_runtime::SessionId;
    for profile in [NumericFp8Profile::F32, NumericFp8Profile::W8A16Bf16] {
        let f = Fixture::with_layers(4);
        let mut r = request(&f);
        if let Qwen35ResolvedAdapter::Moe { options, .. } = r.family_options.qwen35_mut().unwrap() {
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
        let a =
            crate::model_factory::qwen35::binding::pipeline_admission(&r, &f.resources).unwrap();
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
                        a.is_finite() && b.is_finite() && (a - b).abs() <= 3e-4 + 3e-4 * b.abs(),
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

#[cfg(feature = "cuda")]
fn full35b(profile: Option<NumericFp8Profile>) {
    use ferrule_runtime::{GenerateRequest, InferenceShutdownProgress, RequestId, SessionId};
    let path = std::env::var("FERRULE_QWEN35_PP_EP_MODEL")
        .expect("set strict full35B checkpoint directory");
    let devices = std::env::var("FERRULE_QWEN35_PP_EP_DEVICES")
        .expect("set roots-then-stage-EP CUDA ordinals")
        .split(',')
        .map(|s| s.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    assert!(devices.len() >= 6 && devices.len().is_multiple_of(2));
    let degree = (devices.len() - 2) / 2;
    assert!(matches!(degree, 2 | 4 | 8));
    let config = ferrule_model::AutoConfig::from_pretrained(&path).unwrap();
    let mut input = crate::model_factory::tests::options();
    input.max_tensor_mebibytes = 1024;
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
                engine: ServingEngineSelection::Pipeline,
                parallel: Some(PipelineBuildOptions {
                    parallelism: ferrule_common::ParallelismPlan {
                        pipeline_parallel: 2,
                        expert_parallel: degree,
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
#[ignore = "requires full35B numeric FP8 checkpoint, sufficient RAM/VRAM, and explicit FERRULE_QWEN35_PP_EP_MODEL/DEVICES"]
fn pp_ep_gpu_full35b_public_factory_f32_default() {
    full35b(None);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires full35B numeric FP8 checkpoint, sufficient RAM/VRAM, and explicit FERRULE_QWEN35_PP_EP_MODEL/DEVICES"]
fn pp_ep_gpu_full35b_public_factory_w8a16_opt_in() {
    full35b(Some(NumericFp8Profile::W8A16Bf16));
}

#[test]
fn pp_ep_four_owners_per_stage_are_disjoint_not_extra_kv_ranks() {
    let f = Fixture::with_layers(4);
    let mut r = request(&f);
    let parallel = r.pipeline.as_mut().unwrap();
    parallel.parallelism.expert_parallel = 4;
    parallel.devices = Some(vec![0; 10]);
    let p = crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).unwrap();
    assert_eq!(p.owners[0].counts, [2; 4]);
    assert_eq!(p.owners[1].counts, [2; 4]);
    assert_eq!(p.segment_layout.as_ref().unwrap().kv_layers(), 2);
    let mut groups = p.owners.iter().map(|o| o.group.clone()).collect::<Vec<_>>();
    groups[1].members = groups[0].members.clone();
    groups[1].placement = ferrule_model::transformer::expert_parallel::ExpertPlacement::new(
        (2..4).flat_map(|layer| {
            (0..4).map(move |expert| (layer, expert, ParallelRankId::new(2 + expert as u32)))
        }),
    )
    .unwrap();
    assert!(HybridPipelineLayout::with_routed_experts(&f.resources, &p.segments, groups).is_err());
}

#[cfg(feature = "cuda")]
#[path = "full40_strict/pp.rs"]
mod full40_strict;
