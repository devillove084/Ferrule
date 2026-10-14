use super::*;
use std::cell::Cell;
use std::rc::Rc;
use std::sync::mpsc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Case {
    Success,
    Cancel,
    Deadline,
    Unknown,
    FinalCancel,
    FinalDeadline,
}

struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        dispatch_test::HOOK.with_borrow_mut(|hook| *hook = None);
        dispatch_test::NOW.with(|now| now.set(None));
    }
}

struct GatedWorker {
    inner: ExpertRankWorker,
    entered: mpsc::SyncSender<()>,
    gate: mpsc::Receiver<()>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    unknown: bool,
}

impl ReplicaWorker<Command> for GatedWorker {
    type Output = Reply;
    type Error = Error;

    fn execute(&mut self, request: WorkRequest<Command>) -> Result<Reply> {
        self.entered.send(()).unwrap();
        self.gate.recv_timeout(Duration::from_secs(5)).unwrap();
        self.cancelled
            .store(request.cancellation.is_requested(), Ordering::SeqCst);
        if self.unknown {
            std::panic::panic_any(PanicQuiescence::Unknown);
        }
        self.inner.execute(request)
    }

    fn shutdown(&mut self) -> Result<()> {
        self.inner.shutdown()
    }
}

#[test]
fn completion_wait_preserves_deadline_cancel_unknown_and_final_callback_custody() {
    let fixture = Fixture::new(false);
    for case in [
        Case::Success,
        Case::Cancel,
        Case::Deadline,
        Case::Unknown,
        Case::FinalCancel,
        Case::FinalDeadline,
    ] {
        let _reset = Reset;
        let start = Instant::now();
        let timeout = Duration::from_secs(5);
        dispatch_test::NOW.with(|now| now.set(Some(start)));
        let group = Arc::new(group(1));
        let factory_group = Arc::clone(&group);
        let resources = Arc::clone(&fixture.resources);
        let (entered, entry) = mpsc::sync_channel(1);
        let (release, gate) = mpsc::sync_channel(1);
        let gate = Arc::new(Mutex::new(Some(gate)));
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let observed_cancel = Arc::clone(&cancelled);
        let pool = DataParallelExecutor::new(
            DataParallelConfig {
                replicas: 1,
                max_outstanding_per_replica: 1,
                session_capacity: 1,
            },
            move |slot| {
                Ok(GatedWorker {
                    inner: ExpertRankWorker::prepare_cpu(
                        &resources,
                        ExpertRank {
                            slot,
                            owner: factory_group.members[0],
                        },
                        &factory_group,
                        4096,
                    )?,
                    entered,
                    gate: gate.lock().unwrap().take().unwrap(),
                    cancelled: observed_cancel,
                    unknown: case == Case::Unknown,
                })
            },
        )
        .unwrap();
        let mut workers = ExpertRankWorkers {
            group,
            pool: Some(pool),
            next_serial: 1,
            closed: false,
            unknown: false,
            timeout,
            tickets: BTreeMap::new(),
            processes: None,
        };
        let phase = Rc::new(Cell::new(dispatch_test::Phase::Poll));
        let observed_phase = Rc::clone(&phase);
        let cancel = Rc::new(Cell::new(false));
        let request_cancel = Rc::clone(&cancel);
        let waits = Rc::new(Cell::new(0));
        let observed_waits = Rc::clone(&waits);
        let mut first_poll = true;
        let mut release = Some(release);
        dispatch_test::HOOK.with_borrow_mut(|hook| {
            *hook = Some(Box::new(move |workers, phase| {
                use dispatch_test::Phase;
                observed_phase.set(phase);
                match phase {
                    Phase::Poll if first_poll => {
                        first_poll = false;
                        entry.recv_timeout(Duration::from_secs(5)).unwrap();
                    }
                    Phase::BeforeWait => {
                        observed_waits.set(observed_waits.get() + 1);
                        assert_eq!(workers.outstanding(), 1);
                        assert_eq!(workers.pool().outstanding(), 1);
                        assert!(!workers.tickets.values().next().unwrap().completed);
                        if (case != Case::Cancel || request_cancel.get())
                            && let Some(release) = release.take()
                        {
                            release.send(()).unwrap();
                            // Fixed enqueue phase, not a worker-return guess.
                            let shutdown = workers.pool().shutdown();
                            assert_eq!(shutdown.is_err(), case == Case::Unknown);
                        }
                        if case == Case::Deadline {
                            dispatch_test::NOW.with(|now| now.set(Some(start + timeout)));
                        }
                    }
                    Phase::AfterWait if case == Case::Cancel => {
                        request_cancel.set(true);
                    }
                    Phase::BeforePublish => {
                        assert_eq!(workers.outstanding(), 1);
                        assert_eq!(workers.pool().outstanding(), 0);
                        assert!(workers.tickets.values().next().unwrap().completed);
                    }
                    _ => {}
                }
            }));
        });
        let mut final_calls = 0;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            workers.call_many(vec![(rank(0), Command::Stats)], &mut || {
                if cancel.get() {
                    return Err(error("cancel after bounded empty wait"));
                }
                if phase.get() == dispatch_test::Phase::BeforePublish {
                    final_calls += 1;
                    if case == Case::FinalCancel {
                        return Err(error("final callback cancellation"));
                    }
                    if case == Case::FinalDeadline {
                        dispatch_test::NOW.with(|now| now.set(Some(start + timeout)));
                    }
                }
                Ok(())
            })
        }));
        assert!(waits.get() >= 1, "{case:?} bypassed the empty wait");
        match case {
            Case::Deadline | Case::Unknown | Case::FinalDeadline => {
                assert_eq!(
                    result.unwrap_err().downcast_ref(),
                    Some(&PanicQuiescence::Unknown)
                );
                assert!(workers.unknown);
                assert_eq!(workers.outstanding(), 1);
                assert!(
                    workers
                        .call_many(vec![(rank(0), Command::Stats)], &mut || Ok(()))
                        .is_err()
                );
                assert_eq!(
                    workers.pool().outstanding(),
                    usize::from(case == Case::Deadline)
                );
            }
            Case::Cancel | Case::FinalCancel => {
                assert!(result.unwrap().is_err());
                assert!(!workers.unknown);
                assert_eq!(workers.outstanding(), 0);
                assert_eq!(workers.pool().outstanding(), 0);
            }
            Case::Success => {
                assert!(matches!(
                    result.unwrap().unwrap().as_slice(),
                    [Reply::Stats(_)]
                ));
                assert_eq!(workers.outstanding(), 0);
            }
        }
        assert_eq!(
            final_calls,
            usize::from(matches!(
                case,
                Case::Success | Case::FinalCancel | Case::FinalDeadline
            ))
        );
        if case == Case::Cancel {
            assert!(cancelled.load(Ordering::SeqCst));
            assert_eq!(waits.get(), 2);
        }
        eprintln!(
            "{case:?}: bounded wait calls={}, fixed 50us sleeps=0",
            waits.get()
        );
        // Test owners are joined at the gate above. Production Unknown still
        // retains its pool/tickets; reclaim only this test's host pool here.
        drop(workers.pool.take());
    }
}
