use super::*;
use crate::parallel::pipeline::PhysicalKvLayout;
use ferrule_model::ModelExecutionBackend;

#[test]
fn predecision_cancel_rolls_back_all_owners_and_new_step_matches_oracle() {
    let fixture = Arc::new(fixture::Fixture::new());
    let observed = Arc::new(Probe::default());
    let oracle = Arc::new(Probe::default());
    let mut split = pipeline(fixture.clone(), 2, 2, observed.clone());
    let mut full = pipeline(fixture, 1, 1, oracle.clone());
    split
        .forward(SessionId(73), &[1, 2, 3], ForwardPhase::Prefill)
        .unwrap();
    full.forward(SessionId(73), &[1, 2, 3], ForwardPhase::Prefill)
        .unwrap();
    for tokens in [[5], [6]] {
        let cancel = AtomicBool::new(false);
        assert!(
            split
                .forward_observed(
                    SessionId(73),
                    &tokens,
                    ForwardPhase::Decode,
                    &cancel,
                    |progress| {
                        if progress.state == crate::TransactionState::Preparing {
                            cancel.store(true, Ordering::Release);
                        }
                    }
                )
                .is_err()
        );
        assert_eq!(split.coordinator().publication_count(), 1);
        assert!(!split.is_quarantined());
    }
    let a = split
        .forward(SessionId(73), &[7], ForwardPhase::Decode)
        .unwrap();
    let b = full
        .forward(SessionId(73), &[7], ForwardPhase::Decode)
        .unwrap();
    close(a.logits.values(), b.logits.values());
    let snapshots = observed.snapshots.lock().unwrap();
    for rank in 0..4 {
        for id in [3, 4] {
            let a = &snapshots[&(rank, 2, true)];
            let b = &snapshots[&(rank, id, true)];
            assert_eq!(a.recurrent, b.recurrent);
            for (layer, kv) in &a.kv {
                assert_eq!(kv.key, b.kv[layer].key);
                assert_eq!(kv.value, b.kv[layer].value);
            }
        }
    }
    drop(snapshots);
    split.shutdown().unwrap();
    full.shutdown().unwrap();
}

#[test]
fn late_failure_never_partially_publishes_and_unknown_retains_parent_custody() {
    for unknown in [false, true] {
        let fixture = Arc::new(fixture::Fixture::new());
        let probe = Arc::new(Probe::default());
        let mut pp = pipeline(fixture, 2, 2, probe.clone());
        pp.forward(SessionId(73), &[1, 2, 3], ForwardPhase::Prefill)
            .unwrap();
        probe
            .fault
            .store(if unknown { 2 } else { 1 }, Ordering::Release);
        assert!(
            pp.forward(SessionId(73), &[5], ForwardPhase::Decode)
                .is_err()
        );
        assert_eq!(pp.coordinator().publication_count(), 1);
        if unknown {
            assert!(pp.is_quarantined());
            assert!(pp.owner_stats().is_err());
            assert!(pp.shutdown().is_err());
            let participants = pp
                .coordinator()
                .participants(ExecutionTransactionId::new(2).unwrap())
                .unwrap();
            assert_eq!(participants.len(), 4);
            assert!(participants.iter().all(|r| r.get() < 4));
            assert_eq!(probe.stopped.load(Ordering::Relaxed), 0);
        } else {
            let stats = pp.owner_stats().unwrap();
            assert!(
                stats
                    .iter()
                    .all(|s| s.kv.active_transactions == 0 && s.expert_outstanding == 0)
            );
            pp.shutdown().unwrap();
            assert_eq!(probe.expert_shutdowns.load(Ordering::Relaxed), 2);
        }
    }
}

#[test]
fn fork_cow_and_new_page_abort_preserve_the_source_and_physical_capacity() {
    let fixture = Arc::new(fixture::Fixture::new());
    let observed = Arc::new(Probe::default());
    let oracle = Arc::new(Probe::default());
    let mut pp = pipeline(fixture.clone(), 2, 2, observed);
    let mut full = pipeline(fixture.clone(), 1, 1, oracle);
    let expected_bytes = layout(fixture.resources().spec(), 2, 2)
        .stages()
        .iter()
        .flat_map(|stage| (0..2).map(move |lane| stage.rank_layout(lane).unwrap()))
        .map(|local| {
            local
                .kv_planes(2, 32)
                .unwrap()
                .checked_page_bytes()
                .unwrap()
                * config()
                    .physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cpu))
                    .unwrap()
        })
        .sum::<usize>();
    assert_eq!(pp.physical_kv_bytes().unwrap(), expected_bytes as u64);
    pp.forward(SessionId(73), &[1, 2, 3], ForwardPhase::Prefill)
        .unwrap();
    full.forward(SessionId(73), &[1, 2, 3], ForwardPhase::Prefill)
        .unwrap();
    pp.fork_session(SessionId(73), SessionId(74)).unwrap();
    full.fork_session(SessionId(73), SessionId(74)).unwrap();
    let a = pp
        .forward(SessionId(74), &[5], ForwardPhase::Decode)
        .unwrap();
    let b = full
        .forward(SessionId(74), &[5], ForwardPhase::Decode)
        .unwrap();
    close(a.logits.values(), b.logits.values());
    let old = pp
        .owner_stats()
        .unwrap()
        .into_iter()
        .map(|s| s.kv)
        .collect::<Vec<_>>();
    let cancel = AtomicBool::new(false);
    assert!(
        pp.forward_observed(
            SessionId(74),
            &[6],
            ForwardPhase::Decode,
            &cancel,
            |progress| {
                if progress.state == crate::TransactionState::Preparing {
                    cancel.store(true, Ordering::Release);
                }
            }
        )
        .is_err()
    );
    assert_eq!(
        pp.owner_stats()
            .unwrap()
            .into_iter()
            .map(|s| s.kv)
            .collect::<Vec<_>>(),
        old
    );
    for (session, token) in [(SessionId(74), 7), (SessionId(73), 4)] {
        let a = pp.forward(session, &[token], ForwardPhase::Decode).unwrap();
        let b = full
            .forward(session, &[token], ForwardPhase::Decode)
            .unwrap();
        close(a.logits.values(), b.logits.values());
    }
    pp.shutdown().unwrap();
    full.shutdown().unwrap();
}

#[test]
fn typed_unknown_retains_full_cohort_credits_new_page_and_frontier_across_cleanup_attempts() {
    let fixture = Arc::new(fixture::Fixture::new());
    let probe = Arc::new(Probe::default());
    let mut pp = pipeline(fixture, 2, 2, probe.clone());
    let record = protocol::tap(&mut pp);
    pp.forward(SessionId(73), &[1, 2], ForwardPhase::Prefill)
        .unwrap();
    let slot = pp.session_slot(SessionId(73)).unwrap();
    let source = pp.page_manager().stats();
    probe.fault.store(3, Ordering::Release);
    let failure = pp
        .forward(SessionId(73), &[5], ForwardPhase::Decode)
        .unwrap_err();
    assert!(completion_boundary::is_unknown(&failure));
    assert!(
        ferrule_common::error::ErrorChain::new(&failure)
            .downcast_ref::<ferrule_model::decoder::KvPrepareQuiescenceUnknown>()
            .is_some()
    );
    let id = ExecutionTransactionId::new(2).unwrap();
    assert_eq!(
        pp.coordinator()
            .participants(id)
            .unwrap()
            .iter()
            .map(|r| r.get())
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(pp.coordinator().pending_ranks(id).unwrap().len(), 4);
    for _ in 0..3 {
        assert_eq!(pp.coordinator().retained_transaction_count(), 1);
        assert_eq!(pp.coordinator().in_use_credits(), 4);
        assert_eq!(pp.coordinator().publication_count(), 1);
        assert_eq!(
            pp.page_manager().stats().allocated_pages,
            source.allocated_pages + 1
        );
        assert_eq!(
            pp.page_manager()
                .block_table(slot)
                .unwrap()
                .committed_tokens(),
            2
        );
        assert_eq!(
            pp.page_manager().stats().committed_tokens,
            source.committed_tokens
        );
        assert!(pp.shutdown().is_err());
        assert!(pp.release_session(SessionId(73)).is_err());
        assert!(
            pp.forward(SessionId(73), &[7], ForwardPhase::Decode)
                .is_err()
        );
    }
    let events = record.events.lock().unwrap();
    assert_eq!(events.iter().filter(|(op, _)| *op == "publish").count(), 4);
    assert!(!events.iter().any(|(op, _)| *op == "rollback"));
    assert_eq!(probe.stopped.load(Ordering::Relaxed), 0);
}
