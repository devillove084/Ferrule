use super::*;
use std::cell::RefCell;
use std::sync::atomic::AtomicUsize;

thread_local! {
    static REGISTERED: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
}

pub(super) fn registered() {
    let hook = REGISTERED.with_borrow_mut(Option::take);
    if let Some(hook) = hook {
        hook();
    }
}

struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        REGISTERED.with_borrow_mut(|hook| *hook = None);
    }
}

struct Worker;
impl ReplicaWorker<Box<dyn FnOnce(Cancellation) + Send>> for Worker {
    type Output = ();
    type Error = ();

    fn execute(
        &mut self,
        request: WorkRequest<Box<dyn FnOnce(Cancellation) + Send>>,
    ) -> Result<(), ()> {
        (request.input)(request.cancellation);
        Ok(())
    }
}

type Pool = DataParallelExecutor<Box<dyn FnOnce(Cancellation) + Send>, (), ()>;

fn pool(capacity: usize) -> Pool {
    DataParallelExecutor::new(
        DataParallelConfig {
            replicas: 1,
            max_outstanding_per_replica: capacity,
            session_capacity: 1,
        },
        |_| Ok(Worker),
    )
    .unwrap()
}

fn submit(pool: &mut Pool, id: u64, work: impl FnOnce(Cancellation) + Send + 'static) {
    assert!(
        pool.try_submit(
            SessionId(1),
            ExecutionTransactionId::new(id).unwrap(),
            Box::new(work)
        )
        .is_ok()
    );
}

fn assert_notified(signal: &CompletionSignal) {
    let notified = signal.notified.lock().unwrap();
    let (notified, _) = signal
        .ready
        .wait_timeout_while(notified, Duration::from_secs(5), |notified| !*notified)
        .unwrap();
    assert!(*notified, "registered consumer was not notified");
}

#[test]
fn empty_to_registered_wake_preserves_credit_until_consumption() {
    let _reset = Reset;
    let mut pool = pool(1);
    let (release, gate) = mpsc::sync_channel(1);
    submit(&mut pool, 1, move |_| {
        gate.recv_timeout(Duration::from_secs(5)).unwrap()
    });
    let mut listener = pool.completion_listener();
    assert!(pool.poll().is_none());
    let signal = Arc::clone(&pool.completion_wait.signal);
    REGISTERED.with_borrow_mut(|hook| {
        *hook = Some(Box::new(move || {
            release.send(()).unwrap();
            assert_notified(&signal);
        }));
    });
    pool.wait_for_completion(&mut listener, Instant::now() + Duration::from_secs(5));
    assert!(REGISTERED.with_borrow(|hook| hook.is_none()));
    assert_eq!(pool.completion_hub.epoch(), 1);
    assert_eq!(pool.outstanding(), 1);
    assert!(matches!(
        pool.release_session(SessionId(1)),
        Err(ReleaseSessionError::Busy { .. })
    ));
    assert!(matches!(
        pool.try_recv().unwrap().outcome,
        CompletionOutcome::Success(())
    ));
    assert_eq!(pool.outstanding(), 0);
    pool.shutdown().unwrap();
    assert_eq!(pool.completion_hub.epoch(), 2); // enqueue + owner exit, no polling producer
    eprintln!(
        "empty->wake: completion notifications=1, consumer wait calls=1, fixed 50us sleeps=0"
    );
}

#[test]
fn burst_before_registration_is_coalesced_without_releasing_credits() {
    let mut pool = pool(8);
    let calls = Arc::new(AtomicUsize::new(0));
    for id in 1..=8 {
        let calls = Arc::clone(&calls);
        submit(&mut pool, id, move |_| {
            calls.fetch_add(1, Ordering::Relaxed);
        });
    }
    let mut listener = pool.completion_listener();
    // Join proves all accepted requests have an enqueued outcome, even those
    // cancelled before entry. No completion was consumed by shutdown.
    pool.shutdown().unwrap();
    assert_eq!(pool.completion_hub.epoch(), 9);
    pool.wait_for_completion(&mut listener, Instant::now());
    assert_eq!(pool.outstanding(), 8);
    for id in 1..=8 {
        let completion = pool.try_recv().unwrap();
        assert_eq!(completion.transaction.get(), id);
        assert!(matches!(
            completion.outcome,
            CompletionOutcome::Success(()) | CompletionOutcome::Cancelled
        ));
    }
    assert!(pool.poll().is_none());
    assert_eq!(pool.outstanding(), 0);
    assert!(calls.load(Ordering::Relaxed) <= 8);
    eprintln!(
        "burst: enqueue notifications=8, exit notifications=1, consumer wait calls=1, fixed 50us sleeps=0"
    );
}

#[test]
fn expired_deadline_and_spurious_signal_are_not_completions() {
    let _reset = Reset;
    let mut pool = pool(1);
    let (release, gate) = mpsc::sync_channel(1);
    submit(&mut pool, 1, move |_| {
        gate.recv_timeout(Duration::from_secs(5)).unwrap()
    });
    let mut listener = pool.completion_listener();
    assert!(pool.try_recv().is_none());
    let signal = Arc::clone(&pool.completion_wait.signal);
    REGISTERED.with_borrow_mut(|hook| {
        *hook = Some(Box::new(move || signal.ready.notify_one()));
    });
    let deadline = Instant::now();
    pool.wait_for_completion(&mut listener, deadline);
    assert_eq!(pool.completion_hub.epoch(), 0);
    assert!(pool.poll().is_none());
    assert_eq!(pool.outstanding(), 1);
    release.send(()).unwrap();
    pool.shutdown().unwrap();
    assert!(pool.try_recv().is_some());
}

#[test]
fn cancellation_wait_is_bounded_without_a_completion_producer() {
    let mut pool = pool(1);
    let (entered, entry) = mpsc::sync_channel(1);
    let (release, gate) = mpsc::sync_channel(1);
    let (observed, observation) = mpsc::sync_channel(1);
    submit(&mut pool, 1, move |cancel| {
        entered.send(()).unwrap();
        gate.recv_timeout(Duration::from_secs(5)).unwrap();
        observed.send(cancel.is_requested()).unwrap();
    });
    entry.recv_timeout(Duration::from_secs(5)).unwrap();
    let mut listener = pool.completion_listener();
    let start = Instant::now();
    pool.wait_for_completion(&mut listener, start + Duration::from_secs(30));
    assert!(start.elapsed() < Duration::from_secs(5));
    assert_eq!(pool.completion_hub.epoch(), 0);
    assert!(pool.poll().is_none());
    pool.cancel(ExecutionTransactionId::new(1).unwrap())
        .unwrap();
    assert_eq!(pool.outstanding(), 1);
    release.send(()).unwrap();
    assert!(observation.recv_timeout(Duration::from_secs(5)).unwrap());
    pool.shutdown().unwrap();
    let completion = pool.try_recv().unwrap();
    assert!(completion.cancellation_requested);
    assert!(matches!(completion.outcome, CompletionOutcome::Success(())));
}

#[test]
fn unknown_panic_notifies_without_replay_or_safe_completion() {
    let mut pool = pool(1);
    let (entered, entry) = mpsc::sync_channel(1);
    let (release, gate) = mpsc::sync_channel(1);
    submit(&mut pool, 1, move |_| {
        entered.send(()).unwrap();
        gate.recv_timeout(Duration::from_secs(5)).unwrap();
        std::panic::panic_any(PanicQuiescence::Unknown);
    });
    entry.recv_timeout(Duration::from_secs(5)).unwrap();
    let mut listener = pool.completion_listener();
    assert!(pool.try_recv().is_none());
    release.send(()).unwrap();
    assert!(pool.shutdown().is_err());
    pool.wait_for_completion(&mut listener, Instant::now());
    assert_eq!(pool.completion_hub.epoch(), 2);
    assert_eq!(pool.outstanding(), 1);
    assert!(matches!(
        pool.try_recv().unwrap().outcome,
        CompletionOutcome::QuiescenceUnknown
    ));
    assert!(pool.poll().is_none());
    assert!(
        pool.try_submit(
            SessionId(1),
            ExecutionTransactionId::new(2).unwrap(),
            Box::new(|_| panic!("replay"))
        )
        .is_err()
    );
}

#[test]
fn owner_exit_notifies_an_empty_pool_and_shutdown_still_requires_join() {
    let mut pool = pool(1);
    let mut listener = pool.completion_listener();
    assert!(pool.poll().is_none());
    pool.begin_shutdown();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !pool.shutdown_ready() {
        assert!(Instant::now() < deadline);
        pool.wait_for_completion(&mut listener, deadline);
        listener = pool.completion_listener();
    }
    pool.shutdown().unwrap();
    assert_eq!(pool.completion_hub.epoch(), 1);
    assert!(pool.poll().is_none());
}

#[test]
fn poisoned_adapter_lock_cannot_panic_owner_exit_notification() {
    let wait = CompletionWait::new();
    let signal = Arc::clone(&wait.signal);
    assert!(
        std::panic::catch_unwind(move || {
            let _guard = signal.notified.lock().unwrap();
            panic!("poison only the notification lock");
        })
        .is_err()
    );
    let hub = CompletionHub::new();
    let mut listener = hub.listen();
    let mut context = Context::from_waker(&wait.waker);
    assert!(Pin::new(&mut listener).poll(&mut context).is_pending());
    let alive = Arc::new(AtomicBool::new(true));
    drop(OwnerExit(Arc::clone(&alive), hub.clone()));
    assert!(!alive.load(Ordering::Acquire));
    assert_eq!(hub.epoch(), 1);
    wait.wait(&mut listener, Instant::now());
}
