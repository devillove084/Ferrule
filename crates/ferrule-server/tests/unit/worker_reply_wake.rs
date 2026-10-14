use super::*;

// Capture the ledger inside oneshot::send's synchronous wake, before the
// command handler can return. This is the deterministic negative control
// for relying on command drop after send (no scheduler sleeps required).
struct ReplyLedgerWake {
    admission: ServerAdmission,
    snapshots: Mutex<Vec<ServerAdmissionSnapshot>>,
    observer: Mutex<Option<ServerAdmissionPermit>>,
}

impl std::task::Wake for ReplyLedgerWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        // Model the caller dropping its observation during a reply race.
        drop(self.observer.lock().unwrap().take());
        self.snapshots
            .lock()
            .unwrap()
            .push(self.admission.snapshot());
    }
}

fn watch_reply<T>(
    receiver: &mut oneshot::Receiver<T>,
    admission: &ServerAdmission,
    observer: Option<ServerAdmissionPermit>,
) -> Arc<ReplyLedgerWake> {
    let wake = Arc::new(ReplyLedgerWake {
        admission: admission.clone(),
        snapshots: Mutex::new(vec![]),
        observer: Mutex::new(observer),
    });
    let waker = std::task::Waker::from(wake.clone());
    assert!(
        std::future::Future::poll(
            std::pin::Pin::new(receiver),
            &mut std::task::Context::from_waker(&waker),
        )
        .is_pending()
    );
    wake
}

fn assert_reply_ledger(wake: &ReplyLedgerWake, requests: usize) {
    let snapshots = wake.snapshots.lock().unwrap();
    assert_eq!(snapshots.len(), 1, "one reply wake");
    assert_eq!(snapshots[0].held_requests, requests);
    assert_eq!(snapshots[0].held_prompt_bytes, 0);
}

#[test]
fn unadmitted_rejections_release_permit_before_reply_wake() {
    for mode in 0..3 {
        for disconnect_at_reply in [false, true] {
            let control = ready_control();
            let mut engine = Pr13Engine {
                encode_error: mode == 0,
                encode_empty: mode == 1,
                reject: mode == 2,
                ..Default::default()
            };
            let mut active = HashMap::new();
            let permit = control.admission.try_acquire(5).unwrap();
            let observer = disconnect_at_reply.then(|| permit.observer());
            let (events, _stream) = mpsc::channel(2);
            let (accepted, mut reply) = oneshot::channel();
            let wake = watch_reply(&mut reply, &control.admission, observer);
            handle_command(
                WorkerCommand::Submit(SubmitCommand {
                    request_id: RequestId(1),
                    enqueued_at: Instant::now(),
                    request: request(),
                    permit,
                    events,
                    accepted,
                    cancellation: Arc::new(AtomicBool::new(false)),
                }),
                &mut engine,
                &mut active,
                &control,
            );
            assert_reply_ledger(&wake, 0);
            let error = reply.try_recv().unwrap().unwrap_err();
            match (mode, error.as_ref()) {
                (0, WorkerExecutionError::AdmissionTokenization { .. })
                | (1, WorkerExecutionError::AdmissionEmptyPromptTokens)
                | (2, WorkerExecutionError::Runtime { .. }) => {}
                _ => panic!("wrong rejection: {error:?}"),
            }
            assert!(active.is_empty());
            assert_eq!(engine.submits, 0);
            pr13_assert_counts(&control.admission, 0, 0);
        }
    }
}

#[test]
fn tokenize_success_and_error_release_permit_before_reply_wake() {
    for encode_error in [false, true] {
        let control = ready_control();
        let mut engine = Pr13Engine {
            encode_error,
            ..Default::default()
        };
        let (response, mut reply) = oneshot::channel();
        let wake = watch_reply(&mut reply, &control.admission, None);
        handle_command(
            WorkerCommand::Tokenize(TokenizeCommand {
                prompt: "hello".into(),
                permit: control.admission.try_acquire(5).unwrap(),
                response,
            }),
            &mut engine,
            &mut HashMap::new(),
            &control,
        );
        assert_reply_ledger(&wake, 0);
        assert_eq!(reply.try_recv().unwrap().is_err(), encode_error);
        pr13_assert_counts(&control.admission, 0, 0);
    }
}

#[test]
fn admitted_reply_and_duplicate_keep_only_physical_owner_until_cleanup() {
    for disconnected in [false, true] {
        let control = ready_control();
        let mut engine = Pr13Engine::default();
        let mut active = HashMap::new();
        let mut streams = vec![];
        for duplicate in [false, true] {
            let permit = control.admission.try_acquire(5).unwrap();
            let (events, stream) = mpsc::channel(2);
            let (accepted, mut reply) = oneshot::channel();
            let wake = watch_reply(&mut reply, &control.admission, Some(permit.observer()));
            handle_command(
                WorkerCommand::Submit(SubmitCommand {
                    request_id: RequestId(1),
                    enqueued_at: Instant::now(),
                    request: request(),
                    permit,
                    events,
                    accepted,
                    cancellation: Arc::new(AtomicBool::new(false)),
                }),
                &mut engine,
                &mut active,
                &control,
            );
            assert_reply_ledger(&wake, 1);
            assert_eq!(reply.try_recv().unwrap().is_err(), duplicate);
            streams.push(stream);
        }
        assert_eq!(engine.submits, 1);
        assert_eq!(active.len(), 1);
        if disconnected {
            streams.clear();
            active
                .get_mut(&RequestId(1))
                .unwrap()
                .cancellation
                .store(true, Ordering::Release);
            cancel_disconnected(&mut engine, &mut active, &mut vec![], &mut vec![], &control);
            assert_eq!(engine.cancel_calls, 1);
            pr13_assert_counts(&control.admission, 1, 0);
        }
        let request = engine.requests.remove(&RequestId(1)).unwrap();
        let mut state = SequenceState::from_request(&request, request.session_id.unwrap());
        if disconnected {
            state.finish_reason = Some(SequenceFinishReason::Cancelled);
            engine.cancelled.push(state);
        } else {
            engine.failed.push(state);
        }
        drain_terminal(&mut engine, &mut active, &control);
        assert!(active.is_empty());
        if !disconnected {
            assert!(matches!(
                streams[0].try_recv(),
                Ok(WorkerEvent::Failed { .. })
            ));
            assert!(streams[0].try_recv().is_err());
            assert!(streams[1].try_recv().is_err());
        }
        // Even terminal delivery is not the physical cleanup receipt.
        pr13_assert_counts(&control.admission, 1, 0);
        engine.cleanup.remove(&RequestId(1)).unwrap().release();
        drain_terminal(&mut engine, &mut active, &control);
        pr13_assert_counts(&control.admission, 0, 0);
    }
}
