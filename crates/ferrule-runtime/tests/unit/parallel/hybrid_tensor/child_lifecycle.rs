use super::*;
use crate::parallel::expert::cpu_reference_faults::{self, Event};

#[test]
fn real_child_cancel_failure_and_unknown_propagate_through_parent_custody() {
    for mode in 0..3 {
        let fixture = Arc::new(fixture::Fixture::new());
        let spec = fixture.resources().spec().clone();
        let l = layout(&spec, 2, 2);
        let owner_layout = l.clone();
        let probe = Arc::new(Probe::default());
        let observed = probe.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let child_cancel = cancel.clone();
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let child_shutdowns = shutdowns.clone();
        let mut pp = build(
            topology(2, 2),
            &spec,
            l,
            config(),
            options(),
            move |rank| {
                let target = rank.global.get() == 3;
                EXPERT_STARTUP.with_borrow_mut(|slot| {
                    *slot = Some(Arc::new(move |expert| {
                        let cancel = child_cancel.clone();
                        let shutdowns = child_shutdowns.clone();
                        cpu_reference_faults::HOOK.with_borrow_mut(|hook| {
                            *hook = Some(Arc::new(move |event| {
                                match event {
                                    Event::Compute(context)
                                        if target
                                            && expert.slot.get() == 1
                                            && context.transaction.get() == 2 =>
                                    {
                                        match mode {
                                            0 => cancel.store(true, Ordering::Release),
                                            1 => {
                                                return Err(error(
                                                    "injected child compute failure",
                                                ));
                                            }
                                            _ => std::panic::panic_any(PanicQuiescence::Unknown),
                                        }
                                    }
                                    Event::Shutdown => {
                                        shutdowns.fetch_add(1, Ordering::Relaxed);
                                    }
                                    _ => (),
                                }
                                Ok(())
                            }))
                        });
                        Ok(())
                    }))
                });
                Ok(fixture.resources())
            },
            move |rank, stage| {
                let index = rank.global.get() as usize;
                Ok(stage.map_program(|inner| Observed {
                    inner,
                    parallel: owner_layout.stages()[index / 2].clone(),
                    lane: index % 2,
                    probe: observed,
                }))
            },
        )
        .unwrap();
        pp.forward(SessionId(73), &[1, 2, 3], ForwardPhase::Prefill)
            .unwrap();
        let source = pp.page_manager().stats();
        assert!(
            pp.forward_cancellable(SessionId(73), &[5], ForwardPhase::Decode, &cancel)
                .is_err()
        );
        assert_eq!(pp.coordinator().publication_count(), 1);
        if mode == 2 {
            assert!(pp.owner_stats().is_err());
            assert!(pp.shutdown().is_err());
            assert_eq!(
                pp.coordinator()
                    .participants(ExecutionTransactionId::new(2).unwrap())
                    .unwrap()
                    .len(),
                4
            );
            assert!(pp.page_manager().stats().allocated_pages >= source.allocated_pages);
            assert_eq!(probe.stopped.load(Ordering::Relaxed), 0);
        } else {
            assert_eq!(
                pp.page_manager().stats().committed_tokens,
                source.committed_tokens
            );
            assert!(
                pp.owner_stats()
                    .unwrap()
                    .iter()
                    .all(|s| s.kv.active_transactions == 0 && s.expert_outstanding == 0)
            );
            pp.shutdown().unwrap();
            assert_eq!(shutdowns.load(Ordering::Relaxed), 4);
            assert_eq!(probe.stopped.load(Ordering::Relaxed), 4);
        }
    }
}

#[test]
fn child_shutdown_unknown_never_becomes_a_parent_shutdown_ack() {
    let fixture = Arc::new(fixture::Fixture::new());
    let spec = fixture.resources().spec().clone();
    let l = layout(&spec, 2, 2);
    let mut pp = PipelineParallelExecutor::new_hybrid_cpu_tensor(
        topology(2, 2),
        &spec,
        l,
        config(),
        options(),
        move |rank| {
            if rank.global.get() == 3 {
                EXPERT_STARTUP.with_borrow_mut(|slot| {
                    *slot = Some(Arc::new(move |expert| {
                        if expert.slot.get() == 1 {
                            cpu_reference_faults::HOOK.with_borrow_mut(|hook| {
                                *hook = Some(Arc::new(|event| {
                                    if matches!(event, Event::Shutdown) {
                                        std::panic::panic_any(PanicQuiescence::Unknown);
                                    }
                                    Ok(())
                                }))
                            });
                        }
                        Ok(())
                    }))
                });
            }
            Ok(fixture.resources())
        },
    )
    .unwrap();
    pp.forward(SessionId(73), &[1, 2, 3], ForwardPhase::Prefill)
        .unwrap();
    let failure = pp.shutdown().unwrap_err();
    assert!(completion_boundary::is_unknown(&failure));
    assert!(pp.is_quarantined());
    assert!(pp.shutdown().is_err());
    assert!(pp.owner_stats().is_err());
}
