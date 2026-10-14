use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::{Command, Stdio};

struct Callback(Box<dyn Fn() + Send + Sync>);
impl Wake for Callback {
    fn wake(self: Arc<Self>) {
        (self.0)();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        (self.0)();
    }
}

// Deadlocks, self-join panics and double-panic aborts must not wedge the test
// runner. Each actual regression executes in its own externally timed child.
fn isolated(name: &str, test: impl FnOnce()) {
    let full = format!("checkpoint::read_pipeline::tests::reentrant::{name}");
    if std::env::var("FERRULE_REENTRANT_CHILD").ok().as_deref() == Some(&full) {
        test();
        return;
    }
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &full, "--nocapture", "--test-threads=1"])
        .env("FERRULE_REENTRANT_CHILD", &full)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            let output = child.wait_with_output().unwrap();
            assert!(
                status.success(),
                "{full} failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        if start.elapsed() > Duration::from_secs(10) {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!(
                "{full} exceeded subprocess deadline:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        thread::sleep(Duration::from_millis(5));
    }
}

fn arm(
    pipeline: &CheckpointReadPipeline,
    action: impl Fn() + Send + Sync + 'static,
) -> CompletionListener {
    let mut listener = pipeline.listen();
    let waker = Waker::from(Arc::new(Callback(Box::new(action))));
    assert!(
        Pin::new(&mut listener)
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    listener
}

#[test]
fn worker_waker_drop_retains_until_off_worker_join() {
    isolated("worker_waker_drop_retains_until_off_worker_join", || {
        let fixture = Fixture::new();
        let reader = CheckpointTensorReader::new(32);
        let gate = Gate::new(TestPhase::AfterPublish);
        let pipeline = fixture.pipeline(&reader, Some(gate.action.clone()));
        let handoff = pipeline.join_handoff();
        let slot = Arc::new(Mutex::new(Some(pipeline)));
        let callback_slot = Arc::clone(&slot);
        let callback_handoff = handoff.clone();
        let (tx, rx) = mpsc::sync_channel(1);
        let listener = arm(slot.lock().unwrap().as_ref().unwrap(), move || {
            let pipeline = callback_slot.lock().unwrap().take();
            if let Some(pipeline) = pipeline {
                drop(pipeline);
                tx.send(callback_handoff.join()).unwrap();
            }
        });
        let ticket = slot
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .submit(GENERATION, fixture.plan.clone())
            .unwrap();
        let pending = rx.recv_timeout(DEADLINE).unwrap();
        gate.entered();
        assert!(pending.join_pending);
        assert!(!pending.join_failed);
        assert!(pending.join_handoff.is_some());
        assert_eq!(pending.budget.reserved_bytes, 8);
        assert_eq!(
            pending.budget.worker_stack.reserved_bytes,
            DEFAULT_READ_WORKER_STACK_BYTES
        );
        assert!(matches!(
            handoff.shared.lock().poll(&ticket).unwrap(),
            ReadPoll::Ready(_)
        ));
        assert!(handoff.shared.join_lock().deferred.is_some());
        gate.release();
        let joined = handoff.join();
        assert!(!joined.join_pending && !joined.join_failed);
        assert_eq!(joined.budget.reserved_bytes, 0);
        assert_eq!(joined.budget.worker_stack.reserved_bytes, 0);
        assert!(handoff.shared.join_lock().retained.is_none());
        assert!(handoff.shared.join_lock().deferred.is_none());
        drop(listener);
    });
}

#[test]
fn worker_waker_shutdown_returns_handoff_without_consuming_results() {
    isolated(
        "worker_waker_shutdown_returns_handoff_without_consuming_results",
        || {
            let fixture = Fixture::new();
            let reader = CheckpointTensorReader::new(32);
            let gate = Gate::new(TestPhase::AfterPublish);
            let pipeline = fixture.pipeline(&reader, Some(gate.action.clone()));
            let slot = Arc::new(Mutex::new(Some(pipeline)));
            let callback_slot = Arc::clone(&slot);
            let (tx, rx) = mpsc::sync_channel(1);
            let listener = arm(slot.lock().unwrap().as_ref().unwrap(), move || {
                let pipeline = callback_slot.lock().unwrap().take();
                if let Some(mut pipeline) = pipeline {
                    let report = pipeline.shutdown();
                    *callback_slot.lock().unwrap() = Some(pipeline);
                    tx.send(report).unwrap();
                }
            });
            let ticket = slot
                .lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .submit(GENERATION, fixture.plan.clone())
                .unwrap();
            let pending = rx.recv_timeout(DEADLINE).unwrap();
            gate.entered();
            assert!(pending.join_pending && !pending.join_failed);
            assert_eq!(pending.budget.reserved_bytes, 8);
            gate.release();
            let mut pipeline = slot.lock().unwrap().take().unwrap();
            let joined = pipeline.shutdown();
            assert!(!joined.join_pending && !joined.join_failed);
            assert_eq!(joined.budget.reserved_bytes, 8);
            assert_eq!(joined.budget.worker_stack.reserved_bytes, 0);
            drop(pipeline.take(&ticket).unwrap());
            assert_eq!(pipeline.budget().reserved_bytes, 0);
            drop(listener);
        },
    );
}

#[test]
fn worker_unwind_waker_drop_does_not_self_join_or_double_panic() {
    isolated(
        "worker_unwind_waker_drop_does_not_self_join_or_double_panic",
        || {
            let fixture = Fixture::new();
            let reader = CheckpointTensorReader::new(32);
            let gate = Gate::new(TestPhase::SessionOpened);
            let gated = gate.action.clone();
            let pipeline = fixture.pipeline(
                &reader,
                Some(hook(move |phase, ticket| {
                    gated(phase, ticket);
                    if phase == TestPhase::PayloadRead {
                        panic!("injected worker unwind");
                    }
                })),
            );
            let handoff = pipeline.join_handoff();
            let slot = Arc::new(Mutex::new(Some(pipeline)));
            let callback_slot = Arc::clone(&slot);
            let callback_handoff = handoff.clone();
            let (tx, rx) = mpsc::sync_channel(1);
            let listener = arm(slot.lock().unwrap().as_ref().unwrap(), move || {
                let pipeline = callback_slot.lock().unwrap().take();
                if let Some(pipeline) = pipeline {
                    drop(pipeline);
                    tx.send(callback_handoff.join()).unwrap();
                }
            });
            slot.lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .submit(GENERATION, fixture.plan.clone())
                .unwrap();
            gate.entered();
            gate.release();
            let pending = rx.recv_timeout(DEADLINE).unwrap();
            assert!(pending.join_pending);
            assert_eq!(pending.budget.unknown_bytes, 8);
            let joined = handoff.join();
            assert!(!joined.join_pending && joined.join_failed);
            assert_eq!(joined.budget.unknown_bytes, 8);
            assert_eq!(
                joined.budget.worker_stack.unknown_bytes,
                DEFAULT_READ_WORKER_STACK_BYTES
            );
            assert_eq!(joined.unknown[0].reason, ReadUnknownReason::WorkerPanicked);
            drop(listener);
        },
    );
}

#[test]
fn poisoned_lock_close_can_reenter_poll_and_drop_on_worker() {
    isolated(
        "poisoned_lock_close_can_reenter_poll_and_drop_on_worker",
        || {
            let fixture = Fixture::new();
            let reader = CheckpointTensorReader::new(32);
            let gate = Gate::new(TestPhase::SessionOpened);
            let shared_slot = Arc::new(Mutex::new(None::<Arc<Shared>>));
            let hook_slot = Arc::clone(&shared_slot);
            let gated = gate.action.clone();
            let pipeline = fixture.pipeline(
                &reader,
                Some(hook(move |phase, ticket| {
                    gated(phase, ticket);
                    if phase == TestPhase::PayloadRead {
                        let shared = hook_slot.lock().unwrap().as_ref().unwrap().clone();
                        assert!(
                            catch_unwind(AssertUnwindSafe(|| {
                                let _state = shared.state.lock().unwrap();
                                panic!("injected poisoned state");
                            }))
                            .is_err()
                        );
                        assert!(!shared.lock().accepting);
                    }
                })),
            );
            *shared_slot.lock().unwrap() = Some(Arc::clone(&pipeline.shared));
            let handoff = pipeline.join_handoff();
            let ticket_slot = Arc::new(Mutex::new(None::<ReadTicket>));
            let callback_ticket = Arc::clone(&ticket_slot);
            let callback_handoff = handoff.clone();
            let slot = Arc::new(Mutex::new(Some(pipeline)));
            let callback_slot = Arc::clone(&slot);
            let (tx, rx) = mpsc::sync_channel(1);
            let listener = arm(slot.lock().unwrap().as_ref().unwrap(), move || {
                let ticket = callback_ticket.lock().unwrap().as_ref().unwrap().clone();
                assert!(matches!(
                    callback_handoff.shared.lock().poll(&ticket).unwrap(),
                    ReadPoll::Unknown {
                        reason: ReadUnknownReason::StatePoisoned,
                        ..
                    }
                ));
                let pipeline = callback_slot.lock().unwrap().take();
                if let Some(pipeline) = pipeline {
                    drop(pipeline);
                    tx.send(callback_handoff.join()).unwrap();
                }
            });
            let ticket = slot
                .lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .submit(GENERATION, fixture.plan.clone())
                .unwrap();
            gate.entered();
            *ticket_slot.lock().unwrap() = Some(ticket);
            gate.release();
            let pending = rx.recv_timeout(DEADLINE).unwrap();
            assert!(pending.join_pending);
            let joined = handoff.join();
            assert!(!joined.join_pending && !joined.join_failed);
            assert_eq!(joined.budget.unknown_bytes, 8);
            assert_eq!(joined.unknown[0].reason, ReadUnknownReason::StatePoisoned);
            assert_eq!(joined.budget.worker_stack.reserved_bytes, 0);
            drop(listener);
        },
    );
}

#[test]
fn condvar_poison_close_can_reenter_state_without_holding_guard() {
    isolated(
        "condvar_poison_close_can_reenter_state_without_holding_guard",
        || {
            let fixture = Fixture::new();
            let reader = CheckpointTensorReader::new(32);
            let gate = Gate::new(TestPhase::SessionOpened);
            let mut pipeline = fixture.pipeline(&reader, Some(gate.action.clone()));
            let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
            gate.entered();
            let shared = Arc::clone(&pipeline.shared);
            let callback_shared = Arc::clone(&shared);
            let callback_ticket = ticket.clone();
            let (wake_tx, wake_rx) = mpsc::sync_channel(1);
            let listener = arm(&pipeline, move || {
                let result = callback_shared.lock().poll(&callback_ticket).unwrap();
                wake_tx.send(result).unwrap();
            });
            let (waiting_tx, waiting_rx) = mpsc::sync_channel(0);
            let waiter_shared = Arc::clone(&shared);
            let waiter = thread::spawn(move || {
                let guard = waiter_shared.state.lock().unwrap();
                waiting_tx.send(()).unwrap();
                let guard = waiter_shared.wait(guard);
                assert!(!guard.accepting);
            });
            waiting_rx.recv_timeout(DEADLINE).unwrap();
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    let _state = shared.state.lock().unwrap();
                    panic!("injected poison while condvar waiter sleeps");
                }))
                .is_err()
            );
            shared.changed.notify_all();
            assert!(matches!(
                wake_rx.recv_timeout(DEADLINE).unwrap(),
                ReadPoll::Unknown {
                    reason: ReadUnknownReason::StatePoisoned,
                    ..
                }
            ));
            waiter.join().unwrap();
            gate.release();
            let report = pipeline.shutdown();
            assert!(!report.join_pending && !report.join_failed);
            assert_eq!(report.budget.unknown_bytes, 8);
            assert_eq!(report.unknown[0].reason, ReadUnknownReason::StatePoisoned);
            drop(listener);
        },
    );
}

#[test]
fn notification_panic_retains_queued_custody_without_killing_worker_owner() {
    isolated(
        "notification_panic_retains_queued_custody_without_killing_worker_owner",
        || {
            let fixture = Fixture::new();
            let reader = CheckpointTensorReader::new(32);
            let gate = Gate::new(TestPhase::SessionOpened);
            let mut pipeline = fixture.pipeline(&reader, Some(gate.action.clone()));
            let first = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
            gate.entered();
            let listener = arm(&pipeline, || panic!("injected user waker panic"));
            let second = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
            gate.release();
            ready(&pipeline, &first);
            assert!(matches!(
                pipeline.wait(&second).unwrap(),
                ReadPoll::Unknown {
                    reason: ReadUnknownReason::NotificationPanicked,
                    ..
                }
            ));
            let report = pipeline.shutdown();
            assert!(!report.join_pending && !report.join_failed);
            assert_eq!(report.budget.unknown_bytes, 8);
            assert_eq!(report.budget.reserved_bytes, 16);
            assert_eq!(reader.read_counters().bytes_read, 8);
            drop(pipeline.take(&first).unwrap());
            assert_eq!(pipeline.budget().reserved_bytes, 8);
            assert!(pipeline.drain().is_empty());
            drop(listener);
        },
    );
}

#[test]
fn user_waker_panic_during_worker_unwind_is_contained() {
    isolated("user_waker_panic_during_worker_unwind_is_contained", || {
        let fixture = Fixture::new();
        let reader = CheckpointTensorReader::new(32);
        let gate = Gate::new(TestPhase::SessionOpened);
        let gated = gate.action.clone();
        let mut pipeline = fixture.pipeline(
            &reader,
            Some(hook(move |phase, ticket| {
                gated(phase, ticket);
                if phase == TestPhase::PayloadRead {
                    panic!("injected worker panic");
                }
            })),
        );
        let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
        gate.entered();
        let listener = arm(&pipeline, || panic!("injected panic during unwind close"));
        gate.release();
        assert!(matches!(
            pipeline.wait(&ticket).unwrap(),
            ReadPoll::Unknown {
                reason: ReadUnknownReason::WorkerPanicked,
                ..
            }
        ));
        let report = pipeline.shutdown();
        assert!(!report.join_pending && report.join_failed);
        assert_eq!(report.budget.unknown_bytes, 8);
        assert_eq!(
            report.budget.worker_stack.unknown_bytes,
            DEFAULT_READ_WORKER_STACK_BYTES
        );
        drop(listener);
    });
}

#[test]
fn lost_handoff_does_not_detach_or_refund_finished_worker() {
    isolated(
        "lost_handoff_does_not_detach_or_refund_finished_worker",
        || {
            let fixture = Fixture::new();
            let reader = CheckpointTensorReader::new(32);
            let gate = Gate::new(TestPhase::AfterPublish);
            let pipeline = fixture.pipeline(&reader, Some(gate.action.clone()));
            let weak = Arc::downgrade(&pipeline.shared);
            let slot = Arc::new(Mutex::new(Some(pipeline)));
            let callback_slot = Arc::clone(&slot);
            let (tx, rx) = mpsc::sync_channel(1);
            let listener = arm(slot.lock().unwrap().as_ref().unwrap(), move || {
                let pipeline = callback_slot.lock().unwrap().take();
                if let Some(pipeline) = pipeline {
                    // Deliberately keep no returned public handoff: this test checks
                    // conservative custody, then uses private access for cleanup.
                    drop(pipeline);
                    tx.send(()).unwrap();
                }
            });
            slot.lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .submit(GENERATION, fixture.plan.clone())
                .unwrap();
            rx.recv_timeout(DEADLINE).unwrap();
            gate.entered();
            gate.release();
            let deadline = Instant::now() + DEADLINE;
            loop {
                let shared = weak
                    .upgrade()
                    .expect("unjoined worker custody was detached/dropped");
                if shared.join_lock().deferred.as_ref().unwrap().is_finished() {
                    break;
                }
                assert!(Instant::now() < deadline, "worker failed to actually exit");
                thread::sleep(Duration::from_millis(1));
            }
            let retained = weak.upgrade().unwrap();
            assert_eq!(retained.lock().budget.reserved_bytes, 8);
            assert_eq!(
                retained.lock().budget.worker_stack.reserved_bytes,
                DEFAULT_READ_WORKER_STACK_BYTES
            );
            assert!(retained.join_lock().retained.is_some());
            let report = retained.shutdown_snapshot();
            assert!(report.join_pending);
            let joined = report.join_handoff.as_ref().unwrap().join();
            assert!(!joined.join_pending && !joined.join_failed);
            assert_eq!(joined.budget.reserved_bytes, 0);
            assert_eq!(joined.budget.worker_stack.reserved_bytes, 0);
            assert!(retained.join_lock().retained.is_none());
            drop(listener);
        },
    );
}
