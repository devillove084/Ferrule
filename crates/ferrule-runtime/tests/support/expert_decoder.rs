//! Production PP/EP tests sharing only the checkpoint/oracle/backend fixtures.
//! No test threads, channels, dispatch loops, layer math or transaction protocol.
use super::*;

use std::collections::BTreeSet;
use std::ops::Range;

use ferrule_model::moe::ExpertId;
use ferrule_model::nn::ParameterResidency;
use ferrule_model::transformer::expert_parallel::{
    ExpertDispatchContext, ExpertDispatchLimits, ExpertPlacement, ExpertResult, ExpertToken,
    ExpertTokenBucket,
};
use ferrule_runtime::parallel::expert::{
    ExpertGroup, ExpertParallelExecutor, ExpertRank, ExpertRankWorker, ExpertRankWorkers,
};

fn group(layers: Range<usize>) -> ExpertGroup {
    // Deliberately noncontiguous, non-sorted global IDs, unrelated to PP slots.
    let base = 100 + layers.start as u32 * 100;
    let members = vec![rank(base + 9), rank(base + 2)];
    ExpertGroup {
        source_rank: members[1],
        placement: ExpertPlacement::new(
            layers
                .clone()
                .flat_map(|layer| [(layer, 0, members[1]), (layer, 1, members[0])]),
        )
        .unwrap(),
        members,
        layers,
        limits: ExpertDispatchLimits {
            max_tokens: MAX * 2,
            max_bytes: MAX * 2 * 4 * 4,
        },
    }
}

struct HiddenWeights(Vec<(PathBuf, PathBuf)>);
impl Drop for HiddenWeights {
    fn drop(&mut self) {
        for (original, hidden) in &self.0 {
            std::fs::rename(hidden, original).unwrap();
        }
    }
}

fn strict_worker(path: &Path, owner: ExpertRank, group: &ExpertGroup) -> Result<ExpertRankWorker> {
    let resources = load(path)?;
    let mut hidden = HiddenWeights(Vec::new());
    // Factories are serialized by production DP initialization. Remove every
    // unowned payload (including all non-expert weights) AFTER metadata binding,
    // proving preparation does not read or retain another owner's weights.
    let paths = resources.state_dict().parameters().iter().filter(|parameter| {
        !matches!(parameter.residency(), ParameterResidency::Expert { layer, expert }
            if group.layers.contains(layer) && group.placement.owner(*layer, *expert) == Some(owner.owner))
    }).map(|parameter| parameter.weight().slice().path.clone()).collect::<BTreeSet<_>>();
    for original in paths {
        let parked = original.with_extension("hidden");
        std::fs::rename(&original, &parked)?;
        hidden.0.push((original, parked));
    }
    ExpertRankWorker::prepare_cpu(&resources, owner, group, 4096)
}

fn pipeline(
    fixture: &Fixture,
    cfg: PipelineConfig,
    spy: Arc<Spy>,
    configure: fn(&mut ExpertGroup),
) -> Result<PipelineParallelExecutor> {
    let path = fixture.0.clone();
    PipelineParallelExecutor::new(topology(2), plans(2), cfg, move |pp, plan| {
        let resources = load(&path)?;
        let mut group = group(plan.layers());
        configure(&mut group);
        let expert_path = path.clone();
        let experts = ExpertParallelExecutor::new(group, move |owner, group| {
            strict_worker(&expert_path, owner, group)
        })?;
        let stage = PipelineStage::prepare_cpu_with_experts(&resources, plan, cfg, experts)?;
        Ok(stage.map_backend(|inner| SpyBackend::new(inner, pp.local.get() as usize, spy)))
    })
}

fn remove_expert_weights(fixture: &Fixture) {
    let resources = load(&fixture.0).unwrap();
    for layer in 0..2 {
        let first = resources
            .experts()
            .require(layer, 0, TensorRole::RoutedExpertGate)
            .unwrap();
        let second = resources
            .experts()
            .require(layer, 1, TensorRole::RoutedExpertGate)
            .unwrap();
        assert_ne!(
            std::fs::read(&first.weight().slice().path).unwrap(),
            std::fs::read(&second.weight().slice().path).unwrap()
        );
    }
    let paths = resources
        .state_dict()
        .parameters()
        .iter()
        .filter(|parameter| matches!(parameter.residency(), ParameterResidency::Expert { .. }))
        .map(|parameter| parameter.weight().slice().path.clone())
        .collect::<BTreeSet<_>>();
    for path in paths {
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn pp2_ep2_topk2_full_prefill_decode_is_bitwise_local_without_checkpoint_expert_payloads() {
    for tied in [false, true] {
        let fixture = Fixture::with_top_k(tied, 2);
        let cfg = config(ExecutionPrecisionPolicy::f32());
        let steps = [
            (&[1, 2, 3][..], ForwardPhase::Prefill),
            (&[4, 5][..], ForwardPhase::Prefill),
            (&[6][..], ForwardPhase::Decode),
            (&[1][..], ForwardPhase::Decode),
        ];
        let mut oracle = Oracle::new(&fixture, cfg.precision);
        let expected = steps
            .iter()
            .map(|(tokens, phase)| oracle.forward(tokens, *phase))
            .collect::<Vec<_>>();
        drop(oracle);
        let spy = Spy::new();
        let mut pipeline = pipeline(&fixture, cfg, Arc::clone(&spy), |_| {}).unwrap();
        let scopes = pipeline.execution_scopes();
        assert_eq!(scopes.plan().expert_parallel, 1);
        assert_eq!(
            scopes.kv_participants().iter().collect::<Vec<_>>(),
            [rank(0), rank(1)]
        );
        for stage in 0..2 {
            let attached = scopes.expert_dispatch_members(stage, 0).unwrap().unwrap();
            assert_eq!(
                attached.iter().collect::<Vec<_>>(),
                group(stage as usize..stage as usize + 1).members
            );
            assert!(
                attached
                    .iter()
                    .all(|member| !scopes.kv_participants().contains(member))
            );
        }
        let initial = pipeline.owner_stats().unwrap();
        let mut threads = vec![thread::current().id()];
        for (layer, pp) in initial.iter().enumerate() {
            assert!(!threads.contains(&pp.thread));
            threads.push(pp.thread);
            assert_eq!(pp.experts.len(), 2);
            for (slot, expert) in pp.experts.iter().enumerate() {
                assert!(!threads.contains(&expert.thread));
                threads.push(expert.thread);
                assert_eq!(expert.rank.slot, rank(slot as u32));
                assert_eq!(expert.rank.owner, group(layer..layer + 1).members[slot]);
                assert_eq!(expert.owned_experts, [ExpertId::new(layer, 1 - slot)]);
                assert_eq!(expert.calls, 0);
            }
        }
        assert_eq!(threads.len(), 7); // caller + 2 PP owners + 4 EP owners
        remove_expert_weights(&fixture);
        let session = SessionId(912);
        let mut position = 0;
        for (step, ((tokens, phase), expected)) in steps.iter().zip(&expected).enumerate() {
            let actual = pipeline.forward(session, tokens, *phase).unwrap();
            logits(&actual.logits, expected); // every row, every vocabulary element
            position += tokens.len();
            let table = pipeline
                .page_manager()
                .block_table(pipeline.session_slot(session).unwrap())
                .unwrap();
            assert_eq!(table.committed_tokens(), position);
            let stats = pipeline.owner_stats().unwrap();
            for (layer, pp) in stats.iter().enumerate() {
                assert_eq!(pp.thread, initial[layer].thread);
                assert_eq!(pp.expert_outstanding, 0);
                assert_eq!(*spy.ranks[layer].last_pages.lock().unwrap(), table.pages());
                for (slot, expert) in pp.experts.iter().enumerate() {
                    assert_eq!(expert.thread, initial[layer].experts[slot].thread);
                    assert_eq!(expert.calls, step + 1);
                    assert_eq!(expert.tokens, position);
                    assert_eq!(
                        expert.last_context,
                        Some(ExpertDispatchContext {
                            transaction: actual.transaction,
                            source_rank: group(layer..layer + 1).source_rank,
                            layer,
                        })
                    );
                }
            }
            assert_eq!(pipeline.coordinator().publication_count(), step + 1);
            drained(&pipeline);
        }
        pipeline.release_session(session).unwrap();
        zero(&pipeline);
        pipeline.shutdown().unwrap();
        for pp in &spy.ranks {
            assert_eq!(pp.constructed.load(Ordering::Acquire), 1);
            assert_eq!(pp.dropped.load(Ordering::Acquire), 1);
            assert_eq!(pp.installs.load(Ordering::Acquire), steps.len());
            assert_eq!(pp.finishes.load(Ordering::Acquire), steps.len());
        }
    }
}

fn context() -> ExpertDispatchContext {
    ExpertDispatchContext {
        transaction: tx(901),
        source_rank: group(0..1).source_rank,
        layer: 0,
    }
}
fn bucket() -> ExpertTokenBucket {
    let context = context();
    ExpertTokenBucket {
        owner_rank: group(0..1).placement.owner(0, 0).unwrap(),
        tokens: vec![ExpertToken {
            transaction: context.transaction,
            sequence: 81,
            source_rank: context.source_rank,
            source_row: 0,
            route_slot: 1,
            expert: ExpertId::new(0, 0),
            weight: 0.25,
            payload: vec![1.0, -0.5, 0.2, 2.0],
        }],
    }
}
fn active(id: ExecutionTransactionId) -> Result<()> {
    if id == context().transaction {
        Ok(())
    } else {
        Err(fault("unknown outer transaction"))
    }
}
fn workers(fixture: &Fixture, group: ExpertGroup) -> ExpertRankWorkers {
    let path = fixture.0.clone();
    ExpertRankWorkers::new(group, move |owner, group| {
        strict_worker(&path, owner, group)
    })
    .unwrap()
}

#[test]
fn expert_results_are_unweighted_and_only_echo_activation_identity() {
    let fixture = Fixture::with_top_k(false, 2);
    let mut workers = workers(&fixture, group(0..1));
    remove_expert_weights(&fixture);
    let mut request = bucket();
    request.tokens[0].weight = 0.0;
    let zero_weight = workers
        .execute_bucket(context(), request.clone(), &mut active)
        .unwrap();
    request.tokens[0].weight = 100.0;
    let large_weight = workers
        .execute_bucket(context(), request, &mut active)
        .unwrap();
    assert_eq!(zero_weight, large_weight);
    // Exhaustive destructuring: no route weight, prepared weight, provider or KV
    // handle can be returned in the production reply type.
    let ExpertResult {
        transaction,
        sequence,
        source_rank,
        source_row,
        route_slot,
        expert,
        owner_rank,
        output,
    } = &large_weight[0];
    assert_eq!(*transaction, context().transaction);
    assert_eq!(*sequence, 81);
    assert_eq!(*source_rank, context().source_rank);
    assert_eq!((*source_row, *route_slot), (0, 1));
    assert_eq!(*expert, ExpertId::new(0, 0));
    assert_eq!(*owner_rank, bucket().owner_rank);
    assert_eq!(output.len(), 4);
    assert!(output.iter().any(|&value| value != 0.0));
    assert_eq!(workers.outstanding(), 0);
    workers.shutdown().unwrap();
    workers.shutdown().unwrap();
    assert!(
        workers
            .execute_bucket(context(), bucket(), &mut active)
            .is_err()
    );
}

#[test]
fn expert_admission_rejects_unknown_cancelled_misowned_and_oversized_work_before_compute() {
    let fixture = Fixture::with_top_k(false, 2);
    let mut group = group(0..1);
    group.limits = ExpertDispatchLimits {
        max_tokens: 1,
        max_bytes: 16,
    };
    let mut workers = workers(&fixture, group);
    let mut unknown = context();
    unknown.transaction = tx(902);
    assert!(
        workers
            .execute_bucket(unknown, bucket(), &mut active)
            .is_err()
    );
    assert!(
        workers
            .execute_bucket(context(), bucket(), &mut |_| Err(fault("cancelled")))
            .is_err()
    );
    for kind in 0..8 {
        let mut request = bucket();
        match kind {
            0 => request.owner_rank = rank(999),
            1 => request.owner_rank = group_owner_for_expert_one(),
            2 => request.tokens[0].transaction = tx(902),
            3 => request.tokens[0].expert = ExpertId::new(0, 99),
            4 => request.tokens[0].expert = ExpertId::new(1, 0),
            5 => request.tokens[0].source_rank = rank(0),
            6 => request.tokens.push(request.tokens[0].clone()),
            7 => request.tokens[0].payload.push(0.0),
            _ => unreachable!(),
        }
        assert!(
            workers
                .execute_bucket(context(), request, &mut active)
                .is_err(),
            "case {kind}"
        );
    }
    let empty = ExpertTokenBucket {
        owner_rank: bucket().owner_rank,
        tokens: vec![],
    };
    assert!(
        workers
            .execute_bucket(unknown, empty.clone(), &mut active)
            .is_err()
    );
    assert!(
        workers
            .execute_bucket(
                context(),
                ExpertTokenBucket {
                    owner_rank: rank(999),
                    tokens: vec![]
                },
                &mut active
            )
            .is_err()
    );
    assert!(
        workers
            .execute_bucket(context(), empty, &mut active)
            .unwrap()
            .is_empty()
    );
    assert!(
        workers
            .owner_stats()
            .unwrap()
            .iter()
            .all(|owner| owner.calls == 0)
    );
    assert_eq!(workers.outstanding(), 0);
    assert_eq!(
        workers
            .execute_bucket(context(), bucket(), &mut active)
            .unwrap()
            .len(),
        1
    );
    workers.shutdown().unwrap();
}

fn group_owner_for_expert_one() -> ParallelRankId {
    group(0..1).placement.owner(0, 1).unwrap()
}

#[test]
fn cancellation_after_admission_drains_the_real_owner_and_keeps_outer_id_separate_from_serial() {
    let fixture = Fixture::with_top_k(false, 2);
    let mut workers = workers(&fixture, group(0..1));
    let mut checks = 0;
    let result = workers.execute_bucket(context(), bucket(), &mut |id| {
        active(id)?;
        checks += 1;
        // First: entry; second: transport preflight; third: post-admission.
        if checks >= 3 {
            Err(fault("cancelled after admission"))
        } else {
            Ok(())
        }
    });
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("cancelled after admission")
    );
    assert_eq!(workers.outstanding(), 0);
    // Same outer identity can dispatch another bucket; private DP serials, not
    // this transaction ID, supply the pool's strictly increasing admission IDs.
    let results = workers
        .execute_bucket(context(), bucket(), &mut active)
        .unwrap();
    assert_eq!(results[0].transaction, context().transaction);
    assert_eq!(workers.outstanding(), 0);
}

#[test]
fn ep_limit_error_and_cancellation_roll_back_pp_kv_without_independent_expert_publication() {
    for cancel in [false, true] {
        let fixture = Fixture::with_top_k(false, 2);
        let cfg = config(ExecutionPrecisionPolicy::f32());
        let mut oracle = Oracle::new(&fixture, cfg.precision);
        let first = oracle.forward(&[1], ForwardPhase::Prefill);
        let next = oracle.forward(&[3], ForwardPhase::Decode);
        let spy = Spy::new();
        let mut pipeline = pipeline(&fixture, cfg, Arc::clone(&spy), |group| {
            group.limits.max_tokens = 2;
        })
        .unwrap();
        remove_expert_weights(&fixture);
        let session = SessionId(79);
        logits(
            &pipeline
                .forward(session, &[1], ForwardPhase::Prefill)
                .unwrap()
                .logits,
            &first,
        );
        let failed = if cancel {
            pipeline.forward_observed(
                session,
                &[2],
                ForwardPhase::Decode,
                &spy.cancellation,
                |progress| {
                    if progress.state == TransactionState::Preparing {
                        assert_eq!(progress.publications, 1);
                        assert_eq!(progress.committed_tokens, 1);
                        spy.cancellation.store(true, Ordering::Release);
                    }
                },
            )
        } else {
            pipeline.forward(session, &[2, 4], ForwardPhase::Prefill)
        };
        assert!(failed.is_err());
        drained(&pipeline);
        assert_eq!(pipeline.coordinator().publication_count(), 1);
        assert_eq!(pipeline.page_manager().stats().committed_tokens, 1);
        assert_eq!(pipeline.page_manager().allocated_pages(), 1);
        for pp in &spy.ranks {
            assert_eq!(pp.installs.load(Ordering::Acquire), 1);
        }
        for pp in pipeline.owner_stats().unwrap() {
            for expert in pp.experts {
                assert_eq!(expert.calls, if cancel { 2 } else { 1 });
            }
        }
        spy.cancellation.store(false, Ordering::Release);
        logits(
            &pipeline
                .forward(session, &[3], ForwardPhase::Decode)
                .unwrap()
                .logits,
            &next,
        );
        assert_eq!(pipeline.coordinator().publication_count(), 2);
        pipeline.release_session(session).unwrap();
        zero(&pipeline);
        pipeline.shutdown().unwrap();
    }
}

#[test]
fn ep_results_do_not_bypass_pp_install_ack_publication_gate() {
    let fixture = Fixture::with_top_k(false, 2);
    let cfg = config(ExecutionPrecisionPolicy::f32());
    let expected = Oracle::new(&fixture, cfg.precision).forward(&[1, 2], ForwardPhase::Prefill);
    let spy = Spy::new();
    let mut pipeline = pipeline(&fixture, cfg, Arc::clone(&spy), |_| {}).unwrap();
    spy.hold_install.store(true, Ordering::Release);
    let mut saw_pending = false;
    let output = pipeline
        .forward_observed(
            SessionId(9),
            &[1, 2],
            ForwardPhase::Prefill,
            &spy.cancellation,
            |progress| {
                if progress.state == TransactionState::Decided(Decision::Commit) {
                    assert_eq!(progress.publications, 0);
                    assert_eq!(progress.committed_tokens, 0);
                    if !saw_pending && spy.ranks[1].polls.load(Ordering::Acquire) > 0 {
                        assert!(progress.pending_ranks.contains(&rank(1)));
                        saw_pending = true;
                        spy.hold_install.store(false, Ordering::Release);
                    }
                }
            },
        )
        .unwrap();
    assert!(saw_pending);
    logits(&output.logits, &expected);
    assert_eq!(pipeline.coordinator().publication_count(), 1);
    drained(&pipeline);
}

#[test]
fn ep_precision_membership_placement_and_rank_namespace_are_not_dummy_configuration() {
    let fixture = Fixture::with_top_k(false, 2);
    assert!(
        pipeline(
            &fixture,
            config(ExecutionPrecisionPolicy::bf16_compatibility()),
            Spy::new(),
            |_| {}
        )
        .is_err()
    );
    for configure in [
        (|group: &mut ExpertGroup| group.source_rank = rank(999)) as fn(&mut ExpertGroup),
        |group| group.members[0] = group.members[1],
        |group| {
            group.placement =
                ExpertPlacement::new([(group.layers.start, 0, group.members[0])]).unwrap()
        },
        |group| group.members.push(rank(0)), // collides with PP/KV namespace
        |group| group.members.push(rank(999)), // collides across the two EP groups
    ] {
        assert!(
            pipeline(
                &fixture,
                config(ExecutionPrecisionPolicy::f32()),
                Spy::new(),
                configure
            )
            .is_err()
        );
    }
    // EP metadata constrains explicit groups but never expands the mesh world.
    let attached = ValidatedParallelTopology::new(
        ParallelTopologyId::new(71),
        2,
        rank(0),
        ParallelismPlan::validated(1, 1, 2, 1, 1, 2).unwrap(),
    )
    .unwrap();
    assert_eq!(attached.world_size(), 2);
    assert!(
        ValidatedParallelTopology::new(attached.topology_id(), 4, rank(0), attached.plan(),)
            .is_err()
    );
    let dispatch = group(0..1).dispatch_members(&attached, 0, 0).unwrap();
    assert_eq!(dispatch.iter().collect::<Vec<_>>(), group(0..1).members);
    let mismatch = ValidatedParallelTopology::new(
        attached.topology_id(),
        2,
        rank(0),
        ParallelismPlan::validated(1, 1, 3, 1, 1, 2).unwrap(),
    )
    .unwrap();
    assert!(group(0..1).dispatch_members(&mismatch, 0, 0).is_err());
}
