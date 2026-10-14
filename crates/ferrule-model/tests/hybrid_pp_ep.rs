//! PP boundary and scoped EP math using independent local physical transactions.
//! Global commit/rollback/Unknown belongs to runtime's sealed factory tests.
#![cfg(feature = "cpu-reference")]
#[path = "support/hybrid_pp_ep_fixture.rs"]
mod fixture;
#[path = "hybrid_parallel/model_contract.rs"]
mod support;
use ferrule_model::{decoder::*, transformer::*};
use support::*;

#[test]
fn split_pp_ep_logits_recurrent_and_kv_match_unsplit() {
    let fixture = fixture::Fixture::new();
    let resources = fixture.resources();
    let mut oracle = Oracle::new(&resources);
    let mut stages = (0..2)
        .map(|stage| {
            let owner = rank(if stage == 0 { 7 } else { 19 });
            let p = parallel(
                &resources,
                stage * 2..stage * 2 + 2,
                vec![owner],
                0,
                stage as u32,
            );
            Tensor::new(&resources, p, 0, endpoints_for(&[owner]).pop().unwrap())
        })
        .collect::<Vec<_>>();
    for (i, tokens) in [&[1, 2, 3][..], &[4], &[6], &[2]].into_iter().enumerate() {
        let id = i as u64 + 1;
        let (expected, full) = oracle.step(id, tokens);
        let mut a = stages[0].enter(id, tokens);
        let boundary = stages[0].execute(&mut a, SegmentInput::Tokens).unwrap();
        let mut b = stages[1].enter(id, tokens);
        let actual = stages[1].execute(&mut b, hidden(boundary)).unwrap();
        close(&logits(actual), &expected);
        for (stage, work) in [(&stages[0], &a), (&stages[1], &b)] {
            compare_local(
                stage.module.parallel_plan(),
                0,
                &stage.snapshot(work),
                &full,
            );
            assert_routes(stage, id, tokens.len());
        }
        stages[0].commit_local(a);
        stages[1].commit_local(b);
    }
    for stage in &mut stages {
        stage.release();
    }
    oracle.release();
}
#[test]
fn transaction_and_sequence_mismatch_reject_before_mutation_or_dispatch() {
    let fixture = fixture::Fixture::new();
    let resources = fixture.resources();
    for mode in 0..2 {
        let p = parallel(&resources, 0..2, vec![rank(7)], 0, 0);
        let mut stage = Tensor::new(&resources, p, 0, endpoints_for(&[rank(7)]).pop().unwrap());
        let mut work = stage.enter(1, &[1, 2, 3]);
        let before = work.states.clone();
        let binding = stage
            .module
            .bind_expert_call(
                stage.module.parallel_plan().scope(),
                stage.module.parallel_plan().expert_attachment().unwrap(),
                stage.experts.as_mut().unwrap(),
            )
            .unwrap();
        let result = stage.module.execute_in_transaction(
            if mode == 0 { tx(99) } else { tx(1) },
            &work.batch,
            SegmentInput::Tokens,
            &mut work.states,
            &mut work.view,
            if mode == 0 { &[73] } else { &[] },
            Some(binding),
            &mut |_| Ok(()),
        );
        assert!(result.is_err());
        assert_eq!(work.states, before);
        assert!(stage.experts.as_ref().unwrap().calls.is_empty());
        assert_eq!(stage.local.backend.capacity().active_transactions, 1);
        stage.rollback_local(work);
        stage.release();
    }
}
#[test]
fn stage_and_collective_owner_mismatch_reject_during_prepare() {
    use ferrule_model::transformer::hybrid_parallel::*;
    let fixture = fixture::Fixture::new();
    let resources = fixture.resources();
    for (range, owner) in [(2..4, rank(7)), (0..2, rank(19))] {
        let p = parallel(&resources, 0..2, vec![rank(7)], 0, 0);
        assert!(
            HybridTensorCpuSegment::prepare(
                &resources,
                LayerSegmentPlan::new(4, range, false, false).unwrap(),
                p,
                0,
                Box::new(endpoints_for(&[owner]).pop().unwrap()),
                32,
                1 << 20
            )
            .is_err()
        );
    }
}
