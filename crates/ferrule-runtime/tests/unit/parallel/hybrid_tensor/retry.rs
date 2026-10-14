use super::*;
use crate::parallel::expert::cpu_reference_faults::{self, Event};

/// Uses the public factory and a real EP owner compute hook, not a replacement
/// program. The injected event occurs after model/backend enter on txn 2.
pub(super) fn child_fault_pipeline(
    fixture: Arc<fixture::Fixture>,
    tp: usize,
    mode: usize,
    cancel: Arc<AtomicBool>,
) -> PipelineParallelExecutor {
    let spec = fixture.resources().spec().clone();
    PipelineParallelExecutor::new_hybrid_cpu_tensor(
        topology(2, tp),
        &spec,
        layout(&spec, 2, tp),
        config(),
        options(),
        move |rank| {
            let target = rank.global.get() as usize == 2 * tp - 1;
            EXPERT_STARTUP.with_borrow_mut(|slot| {
                *slot = Some(Arc::new(move |expert| {
                    let cancel = cancel.clone();
                    cpu_reference_faults::HOOK.with_borrow_mut(|hook| {
                        *hook = Some(Arc::new(move |event| {
                            if let Event::Compute(context) = event
                                && target
                                && expert.slot.get() == 1
                                && context.transaction.get() == 2
                            {
                                match mode {
                                    0 => cancel.store(true, Ordering::Release),
                                    1 => return Err(error("secret EP checkpoint failure")),
                                    2 => std::panic::panic_any(PanicQuiescence::Unknown),
                                    _ => (),
                                }
                            }
                            Ok(())
                        }))
                    });
                    Ok(())
                }))
            });
            Ok(fixture.resources())
        },
    )
    .unwrap()
}

#[test]
fn tp1_real_in_compute_cancel_and_ep_error_retry_after_journal_rollback_tp2_stays_poisoned() {
    for tp in [1, 2] {
        for mode in [0, 1] {
            let fixture = Arc::new(fixture::Fixture::new());
            let cancel = Arc::new(AtomicBool::new(false));
            let mut pp = child_fault_pipeline(fixture.clone(), tp, mode, cancel.clone());
            let record = protocol::tap(&mut pp);
            let mut oracle = pipeline(fixture, 1, 1, Arc::new(Probe::default()));
            pp.forward(SessionId(73), &[1, 2], ForwardPhase::Prefill)
                .unwrap();
            oracle
                .forward(SessionId(73), &[1, 2], ForwardPhase::Prefill)
                .unwrap();
            let before = pp.page_manager().stats();
            let failure = pp
                .forward_cancellable(SessionId(73), &[5], ForwardPhase::Decode, &cancel)
                .unwrap_err();
            assert!(!completion_boundary::is_unknown(&failure));
            assert_eq!(pp.coordinator().publication_count(), 1);
            assert_eq!(pp.coordinator().retained_transaction_count(), 0);
            assert_eq!(pp.coordinator().in_use_credits(), 0);
            assert_eq!(
                pp.page_manager().stats().allocated_pages,
                before.allocated_pages
            );
            assert_eq!(pp.page_manager().stats().committed_tokens, 2);
            let events = record.events.lock().unwrap();
            let rollbacks = events
                .iter()
                .filter(|(op, _)| *op == "rollback")
                .map(|(_, rank)| *rank)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(rollbacks, (0..(2 * tp) as u32).collect());
            drop(events);
            if tp == 1 {
                assert!(!pp.is_quarantined());
                let actual = pp
                    .forward(SessionId(73), &[7], ForwardPhase::Decode)
                    .unwrap();
                let expected = oracle
                    .forward(SessionId(73), &[7], ForwardPhase::Decode)
                    .unwrap();
                assert_eq!(actual.transaction.get(), 3);
                close(actual.logits.values(), expected.logits.values());
                assert_eq!(pp.page_manager().stats().committed_tokens, 3);
            } else {
                assert!(pp.is_quarantined());
                assert!(
                    pp.forward(SessionId(73), &[7], ForwardPhase::Decode)
                        .is_err()
                );
            }
            pp.shutdown().unwrap();
            oracle.shutdown().unwrap();
        }
    }
}

#[test]
fn singleton_unknown_or_pending_rollback_never_releases_step_admission() {
    for unknown in [false, true] {
        let fixture = Arc::new(fixture::Fixture::new());
        let cancel = Arc::new(AtomicBool::new(false));
        let mut pp = child_fault_pipeline(fixture, 1, if unknown { 2 } else { 1 }, cancel);
        let record = protocol::tap(&mut pp);
        pp.forward(SessionId(73), &[1, 2], ForwardPhase::Prefill)
            .unwrap();
        record.pending_rollback.store(!unknown, Ordering::Release);
        let failure = pp
            .forward(SessionId(73), &[5], ForwardPhase::Decode)
            .unwrap_err();
        if unknown {
            assert!(completion_boundary::is_unknown(&failure));
        }
        let retained = pp.page_manager().stats().allocated_pages;
        assert_eq!(retained, 2);
        assert!(pp.is_quarantined());
        assert_eq!(pp.coordinator().retained_transaction_count(), 1);
        assert_eq!(pp.coordinator().in_use_credits(), 2);
        for _ in 0..2 {
            assert!(
                pp.forward(SessionId(73), &[7], ForwardPhase::Decode)
                    .is_err()
            );
            assert!(pp.release_session(SessionId(73)).is_err());
            assert!(pp.shutdown().is_err());
            assert_eq!(pp.page_manager().stats().allocated_pages, retained);
            assert_eq!(pp.coordinator().publication_count(), 1);
            assert_eq!(pp.coordinator().retained_transaction_count(), 1);
            assert_eq!(pp.coordinator().in_use_credits(), 2);
        }
    }
}
