//! CPU-only owner/routing and scoped transaction integration tests.
//! Routing delivers host evidence; the existing distributed coordinator alone
//! owns communication accounting, transaction decisions, and publication.

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{
    ParallelRankId, ParallelTopologyId, ParallelismPlan, ParticipantSet, ValidatedParallelTopology,
};
use ferrule_runtime::distributed::PendingCompletion;
use ferrule_runtime::parallel::data::{
    BuildError, CompletionOutcome, ConfigError, DataParallelConfig, DataParallelExecutor,
    HostCompletion, OwnerFailureKind, PanicQuiescence, ReleaseSessionError, ReplicaWorker,
    SubmitErrorKind, WorkRequest,
};
use ferrule_runtime::{
    Decision, DistributedTransaction, DistributedTransactionError, FinalizeOutcome, SessionId,
    TransactionState,
};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

const LIMIT: Duration = Duration::from_secs(10);

fn tx(value: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(value).unwrap()
}

fn rank(value: u32) -> ParallelRankId {
    ParallelRankId::new(value)
}

fn config(replicas: usize, outstanding: usize, sessions: usize) -> DataParallelConfig {
    DataParallelConfig {
        replicas,
        max_outstanding_per_replica: outstanding,
        session_capacity: sessions,
    }
}

#[derive(Debug)]
struct Gate {
    entered: SyncSender<()>,
    release: Receiver<()>,
}

#[derive(Debug)]
struct Input {
    value: u64,
    gate: Option<Gate>,
    panic: bool,
    fail: bool,
}

fn input(value: u64) -> Input {
    Input {
        value,
        gate: None,
        panic: false,
        fail: false,
    }
}

fn gated(value: u64) -> (Input, Receiver<()>, SyncSender<()>) {
    let (entered, started) = mpsc::sync_channel(1);
    let (release, wait) = mpsc::sync_channel(1);
    let mut request = input(value);
    request.gate = Some(Gate {
        entered,
        release: wait,
    });
    (request, started, release)
}

#[derive(Debug)]
struct Reply {
    value: u64,
    invocation: usize,
    cancelled: bool,
    owner: ThreadId,
}

#[derive(Debug)]
struct Dropped {
    rank: ParallelRankId,
    owner: ThreadId,
    executed: usize,
}

struct LocalWorker {
    rank: ParallelRankId,
    owner: ThreadId,
    // Compile-time proof that a worker need not be Send or Sync.
    calls: Rc<Cell<usize>>,
    dropped: SyncSender<Dropped>,
    fail_shutdown: bool,
    panic_shutdown: bool,
    quiescence_gate: Option<Gate>,
}

impl LocalWorker {
    fn new(rank: ParallelRankId, dropped: SyncSender<Dropped>) -> Self {
        Self {
            rank,
            owner: thread::current().id(),
            calls: Rc::new(Cell::new(0)),
            dropped,
            fail_shutdown: false,
            panic_shutdown: false,
            quiescence_gate: None,
        }
    }
}

impl ReplicaWorker<Input> for LocalWorker {
    type Output = Reply;
    type Error = &'static str;

    fn execute(&mut self, request: WorkRequest<Input>) -> Result<Reply, &'static str> {
        assert_eq!(thread::current().id(), self.owner);
        assert_eq!(request.rank, self.rank);
        assert!(request.transaction.get() > 0);
        assert!(request.session.0 > 0);
        self.calls.set(self.calls.get() + 1);
        if let Some(gate) = request.input.gate {
            gate.entered.send(()).unwrap();
            gate.release
                .recv_timeout(LIMIT)
                .expect("test gate was not released");
        }
        assert!(!request.input.panic, "injected execute panic");
        if request.input.fail {
            return Err("execute failed");
        }
        Ok(Reply {
            value: request.input.value,
            invocation: self.calls.get(),
            cancelled: request.cancellation.is_requested(),
            owner: self.owner,
        })
    }

    fn panic_quiescence(&mut self) -> PanicQuiescence {
        assert_eq!(thread::current().id(), self.owner);
        if let Some(gate) = self.quiescence_gate.take() {
            gate.entered.send(()).unwrap();
            gate.release
                .recv_timeout(LIMIT)
                .expect("quiescence gate not released");
        }
        // This worker has only synchronous CPU work, no DMA or asynchronous users.
        PanicQuiescence::Quiescent
    }

    fn shutdown(&mut self) -> Result<(), &'static str> {
        assert_eq!(thread::current().id(), self.owner);
        assert!(!self.panic_shutdown, "injected shutdown panic");
        if self.fail_shutdown {
            Err("shutdown failed")
        } else {
            Ok(())
        }
    }
}

impl Drop for LocalWorker {
    fn drop(&mut self) {
        assert_eq!(thread::current().id(), self.owner);
        self.dropped
            .try_send(Dropped {
                rank: self.rank,
                owner: self.owner,
                executed: self.calls.get(),
            })
            .expect("bounded drop observer has one slot per worker");
    }
}

type Pool = DataParallelExecutor<Input, Reply, &'static str>;
type Completion = HostCompletion<Reply, &'static str>;

fn pool(config: DataParallelConfig) -> (Pool, Receiver<Dropped>) {
    let (dropped, observed) = mpsc::sync_channel(config.replicas);
    let host = thread::current().id();
    let pool = DataParallelExecutor::new(config, move |rank| {
        assert_ne!(host, thread::current().id(), "factory must run on owner");
        Ok(LocalWorker::new(rank, dropped.clone()))
    })
    .unwrap();
    (pool, observed)
}

// Gates establish causality; the deadline only bounds failures, never determines
// correctness. There are no sleeps or assumptions about relative thread speed.
fn completion(pool: &mut Pool) -> Completion {
    let deadline = Instant::now() + LIMIT;
    loop {
        if let Some(completion) = pool.poll() {
            return completion;
        }
        assert!(Instant::now() < deadline, "completion did not arrive");
        thread::yield_now();
    }
}

fn success(completion: Completion, transaction: u64, expected_rank: u32) -> Reply {
    assert_eq!(completion.transaction, tx(transaction));
    assert_eq!(completion.rank, rank(expected_rank));
    match completion.outcome {
        CompletionOutcome::Success(reply) => reply,
        other => panic!("expected success, got {other:?}"),
    }
}

#[test]
fn rejects_all_zero_configuration_before_invoking_factory() {
    for (config, expected) in [
        (config(0, 1, 1), ConfigError::ZeroReplicas),
        (config(1, 0, 1), ConfigError::ZeroOutstanding),
        (config(1, 1, 0), ConfigError::ZeroSessionCapacity),
    ] {
        assert_eq!(config.validate(), Err(expected));
        let result = Pool::new(config, |_| -> Result<LocalWorker, &'static str> {
            panic!("invalid config must not invoke factory")
        });
        assert!(matches!(result, Err(BuildError::Config(error)) if error == expected));
    }
    if let Some(too_many) = (u32::MAX as usize).checked_add(2) {
        assert_eq!(
            config(too_many, 1, 1).validate(),
            Err(ConfigError::TooManyReplicas)
        );
    }
}

#[test]
fn independent_replicas_and_sticky_rank_local_non_send_workers() {
    let (mut pool, drops) = pool(config(2, 2, 3));
    // Explicit route binding is the integration seam for begin_scoped; it sends
    // no execution and consumes no outstanding credit.
    assert_eq!(pool.route_session(SessionId(10)).unwrap(), rank(0));
    assert_eq!(pool.route_session(SessionId(10)).unwrap(), rank(0));
    assert_eq!(pool.outstanding(), 0);
    let (blocked, entered, release) = gated(100);
    assert_eq!(
        pool.try_submit(SessionId(10), tx(1), blocked).unwrap(),
        rank(0)
    );
    entered.recv_timeout(LIMIT).unwrap();
    assert_eq!(
        pool.try_submit(SessionId(20), tx(2), input(200)).unwrap(),
        rank(1)
    );
    let first = success(completion(&mut pool), 2, 1);
    assert_eq!(first.value, 200);
    assert_eq!(first.invocation, 1);
    // Rank 0 has not been released: rank 1 must make progress independently.
    assert_eq!(
        pool.try_submit(SessionId(20), tx(3), input(201)).unwrap(),
        rank(1)
    );
    let second = success(completion(&mut pool), 3, 1);
    assert_eq!(second.owner, first.owner);
    assert_eq!(second.invocation, 2);
    assert_eq!(pool.session_rank(SessionId(10)), Some(rank(0)));
    assert_eq!(pool.session_rank(SessionId(20)), Some(rank(1)));
    release.send(()).unwrap();
    let blocked = success(completion(&mut pool), 1, 0);
    assert_ne!(blocked.owner, first.owner);
    pool.shutdown().unwrap();
    let owners = [drops.try_recv().unwrap(), drops.try_recv().unwrap()];
    assert!(
        owners
            .iter()
            .any(|owner| owner.rank == rank(0) && owner.owner == blocked.owner)
    );
    assert!(
        owners
            .iter()
            .any(|owner| owner.rank == rank(1) && owner.owner == first.owner)
    );
}

#[test]
fn completed_but_unconsumed_work_retains_credits_and_sticky_does_not_spill() {
    let (mut pool, _drops) = pool(config(2, 2, 2));
    pool.try_submit(SessionId(1), tx(1), input(1)).unwrap();
    let (blocked, entered, release) = gated(2);
    pool.try_submit(SessionId(1), tx(2), blocked).unwrap();
    // Entering request 2 proves request 1's completion was sent on this owner.
    entered.recv_timeout(LIMIT).unwrap();
    assert_eq!(pool.outstanding(), 2);
    let rejected = pool.try_submit(SessionId(1), tx(3), input(3)).unwrap_err();
    assert_eq!(rejected.kind, SubmitErrorKind::Backpressure);
    assert_eq!(rejected.rank, Some(rank(0)));
    assert_eq!(rejected.session, SessionId(1));
    assert_eq!(rejected.transaction, tx(3));
    assert_eq!(
        pool.release_session(SessionId(1)),
        Err(ReleaseSessionError::Busy {
            session: SessionId(1),
            rank: rank(0),
            transaction: tx(1),
        })
    );
    success(completion(&mut pool), 1, 0);
    // Rejected ID 3 is still admissible; rank 1 being idle did not migrate us.
    assert_eq!(
        pool.try_submit(SessionId(1), tx(3), rejected.input)
            .unwrap(),
        rank(0)
    );
    release.send(()).unwrap();
    success(completion(&mut pool), 2, 0);
    success(completion(&mut pool), 3, 0);
    assert_eq!(pool.outstanding(), 0);
    pool.shutdown().unwrap();
}

#[test]
fn bounded_sessions_explicit_release_and_constant_space_terminal_id_policy() {
    let (mut pool, _drops) = pool(config(1, 2, 1));
    let (blocked, entered, release) = gated(10);
    pool.try_submit(SessionId(1), tx(10), blocked).unwrap();
    entered.recv_timeout(LIMIT).unwrap();
    let duplicate = pool
        .try_submit(SessionId(2), tx(10), input(99))
        .unwrap_err();
    assert_eq!(duplicate.kind, SubmitErrorKind::DuplicateTransaction);
    assert_eq!(duplicate.rank, Some(rank(0)));
    let capacity = pool.route_session(SessionId(2)).unwrap_err();
    assert_eq!(capacity.kind, SubmitErrorKind::SessionCapacity);
    assert_eq!(capacity.session, SessionId(2));
    assert_eq!(capacity.rank, None);
    assert_eq!(
        pool.try_submit(SessionId(2), tx(100), input(100))
            .unwrap_err()
            .kind,
        SubmitErrorKind::SessionCapacity
    );
    assert_eq!(pool.session_rank(SessionId(2)), None);
    release.send(()).unwrap();
    success(completion(&mut pool), 10, 0);
    assert_eq!(pool.release_session(SessionId(1)), Ok(rank(0)));
    assert_eq!(
        pool.release_session(SessionId(1)),
        Err(ReleaseSessionError::Unknown {
            session: SessionId(1)
        })
    );
    assert_eq!(pool.session_rank(SessionId(1)), None);
    for old in [9, 10] {
        assert_eq!(
            pool.try_submit(SessionId(2), tx(old), input(old))
                .unwrap_err()
                .kind,
            SubmitErrorKind::RetiredOrOutOfOrder {
                high_watermark: tx(10)
            }
        );
    }
    // Rejected ID 100 did not burn IDs or retain a phantom route.
    pool.try_submit(SessionId(2), tx(11), input(11)).unwrap();
    success(completion(&mut pool), 11, 0);
    pool.shutdown().unwrap();
}

#[test]
fn cancel_skips_queued_work_and_exposes_in_flight_intent_without_early_credit_release() {
    let (mut pool, drops) = pool(config(1, 2, 1));
    let (blocked, entered, release) = gated(1);
    pool.try_submit(SessionId(1), tx(1), blocked).unwrap();
    entered.recv_timeout(LIMIT).unwrap();
    pool.try_submit(SessionId(1), tx(2), input(2)).unwrap();
    pool.cancel(tx(2)).unwrap();
    pool.cancel(tx(2)).unwrap();
    pool.cancel(tx(1)).unwrap();
    assert_eq!(pool.outstanding(), 2);
    assert_eq!(
        pool.try_submit(SessionId(1), tx(3), input(3))
            .unwrap_err()
            .kind,
        SubmitErrorKind::Backpressure
    );
    release.send(()).unwrap();
    let running = completion(&mut pool);
    assert!(running.cancellation_requested);
    assert!(success(running, 1, 0).cancelled);
    let queued = completion(&mut pool);
    assert_eq!(queued.transaction, tx(2));
    assert_eq!(queued.rank, rank(0));
    assert_eq!(queued.session, SessionId(1));
    assert!(queued.cancellation_requested);
    assert!(matches!(queued.outcome, CompletionOutcome::Cancelled));
    assert_eq!(pool.cancel(tx(2)).unwrap_err().transaction, tx(2));
    assert_eq!(pool.outstanding(), 0);
    pool.shutdown().unwrap();
    assert_eq!(drops.try_recv().unwrap().executed, 1);
}

#[test]
fn ordinary_worker_error_is_not_owner_loss() {
    let (mut pool, _drops) = pool(config(1, 1, 1));
    let mut bad = input(1);
    bad.fail = true;
    pool.try_submit(SessionId(1), tx(1), bad).unwrap();
    let failed = completion(&mut pool);
    assert_eq!(failed.transaction, tx(1));
    assert_eq!(failed.rank, rank(0));
    assert!(matches!(
        failed.outcome,
        CompletionOutcome::Failed("execute failed")
    ));
    pool.try_submit(SessionId(1), tx(2), input(2)).unwrap();
    assert_eq!(success(completion(&mut pool), 2, 0).invocation, 2);
    pool.shutdown().unwrap();
}

#[test]
fn execute_panic_does_not_replay_migrate_or_stop_another_replica() {
    let (mut pool, drops) = pool(config(2, 2, 2));
    let (mut bad, entered, release) = gated(1);
    bad.panic = true;
    pool.try_submit(SessionId(1), tx(1), bad).unwrap();
    entered.recv_timeout(LIMIT).unwrap();
    pool.try_submit(SessionId(1), tx(2), input(2)).unwrap();
    pool.try_submit(SessionId(2), tx(3), input(3)).unwrap();
    success(completion(&mut pool), 3, 1);
    release.send(()).unwrap();
    let panicked = completion(&mut pool);
    assert_eq!(panicked.transaction, tx(1));
    assert_eq!(panicked.rank, rank(0));
    assert!(matches!(panicked.outcome, CompletionOutcome::Panicked));
    let lost = completion(&mut pool);
    assert_eq!(lost.transaction, tx(2));
    assert_eq!(lost.rank, rank(0));
    assert!(matches!(
        lost.outcome,
        CompletionOutcome::ReplicaUnavailable
    ));
    assert_eq!(pool.session_rank(SessionId(1)), Some(rank(0)));
    let rejected = pool.try_submit(SessionId(1), tx(4), input(4)).unwrap_err();
    assert_eq!(rejected.kind, SubmitErrorKind::ReplicaUnavailable);
    assert_eq!(rejected.rank, Some(rank(0)));
    let route = pool.route_session(SessionId(1)).unwrap_err();
    assert_eq!(route.kind, SubmitErrorKind::ReplicaUnavailable);
    assert_eq!(route.rank, Some(rank(0)));
    pool.try_submit(SessionId(2), tx(4), input(4)).unwrap();
    assert_eq!(success(completion(&mut pool), 4, 1).invocation, 2);
    let failure = pool.shutdown().unwrap_err();
    assert_eq!(failure.failures.len(), 1);
    assert_eq!(failure.failures[0].rank, rank(0));
    assert_eq!(failure.failures[0].transaction, Some(tx(1)));
    assert!(matches!(
        failure.failures[0].kind,
        OwnerFailureKind::Panicked
    ));
    let owners = [drops.try_recv().unwrap(), drops.try_recv().unwrap()];
    assert!(
        owners
            .iter()
            .any(|owner| owner.rank == rank(0) && owner.executed == 1)
    );
    assert!(
        owners
            .iter()
            .any(|owner| owner.rank == rank(1) && owner.executed == 2)
    );
    pool.shutdown().unwrap();
}

#[test]
fn shutdown_joins_every_owner_despite_errors_panics_and_undrained_completions() {
    let (dropped, drops) = mpsc::sync_channel(3);
    let mut pool = Pool::new(config(3, 1, 3), move |rank_id| {
        let mut worker = LocalWorker::new(rank_id, dropped.clone());
        worker.fail_shutdown = rank_id == rank(0);
        worker.panic_shutdown = rank_id == rank(1);
        Ok(worker)
    })
    .unwrap();
    let mut releases = Vec::new();
    for n in 1..=3 {
        let (request, entered, release) = gated(n);
        assert_eq!(
            pool.try_submit(SessionId(n), tx(n), request).unwrap(),
            rank((n - 1) as u32)
        );
        entered.recv_timeout(LIMIT).unwrap();
        releases.push(release);
    }
    for release in releases {
        release.send(()).unwrap();
    }
    let error = pool.shutdown().unwrap_err();
    assert_eq!(error.failures.len(), 2);
    assert_eq!(error.failures[0].rank, rank(0));
    assert!(matches!(
        error.failures[0].kind,
        OwnerFailureKind::Shutdown("shutdown failed")
    ));
    assert_eq!(error.failures[1].rank, rank(1));
    assert!(matches!(error.failures[1].kind, OwnerFailureKind::Panicked));
    // Nonblocking reads AFTER shutdown prove even the last healthy owner joined.
    let mut ranks = (0..3)
        .map(|_| drops.try_recv().unwrap().rank)
        .collect::<Vec<_>>();
    ranks.sort();
    assert_eq!(ranks, vec![rank(0), rank(1), rank(2)]);
    assert_eq!(pool.outstanding(), 3);
    for n in 1..=3 {
        let result = pool.try_recv().unwrap();
        assert!(result.cancellation_requested);
        success(result, n, (n - 1) as u32);
    }
    assert_eq!(pool.outstanding(), 0);
    assert!(pool.try_recv().is_none());
    assert_eq!(
        pool.try_submit(SessionId(1), tx(4), input(4))
            .unwrap_err()
            .kind,
        SubmitErrorKind::Closed
    );
    assert_eq!(
        pool.route_session(SessionId(4)).unwrap_err().kind,
        SubmitErrorKind::Closed
    );
    pool.shutdown().unwrap();
}

#[test]
fn initialization_error_or_panic_joins_previously_created_workers() {
    for factory_failure in 0..3 {
        let (dropped, drops) = mpsc::sync_channel(1);
        let result = Pool::new(config(3, 1, 1), move |rank_id| {
            if rank_id == rank(1) {
                match factory_failure {
                    0 => return Err("initialization failed"),
                    1 => panic!("ordinary factory panic"),
                    2 => std::panic::panic_any(PanicQuiescence::Unknown),
                    _ => unreachable!(),
                }
            }
            assert_eq!(rank_id, rank(0), "must stop creating after failure");
            let mut worker = LocalWorker::new(rank_id, dropped.clone());
            worker.fail_shutdown = true;
            Ok(worker)
        });
        let failures = match result {
            Err(BuildError::Owners(failures)) => failures,
            _ => panic!("construction must fail"),
        };
        assert_eq!(failures.len(), 2);
        assert_eq!(failures[0].rank, rank(1));
        assert_eq!(failures[0].transaction, None);
        match factory_failure {
            0 => assert!(matches!(
                failures[0].kind,
                OwnerFailureKind::Initialization("initialization failed")
            )),
            1 => assert!(matches!(failures[0].kind, OwnerFailureKind::Panicked)),
            2 => assert!(matches!(
                failures[0].kind,
                OwnerFailureKind::QuiescenceUnknown
            )),
            _ => unreachable!(),
        }
        assert_eq!(failures[1].rank, rank(0));
        assert!(matches!(
            failures[1].kind,
            OwnerFailureKind::Shutdown("shutdown failed")
        ));
        assert_eq!(drops.try_recv().unwrap().rank, rank(0));
    }
}

#[test]
fn drop_joins_and_retired_ids_are_not_reset_by_session_release() {
    let (mut pool, drops) = pool(config(1, 1, 1));
    pool.try_submit(SessionId(1), tx(u64::MAX), input(1))
        .unwrap();
    success(completion(&mut pool), u64::MAX, 0);
    pool.release_session(SessionId(1)).unwrap();
    assert_eq!(
        pool.try_submit(SessionId(2), tx(u64::MAX), input(2))
            .unwrap_err()
            .kind,
        SubmitErrorKind::RetiredOrOutOfOrder {
            high_watermark: tx(u64::MAX)
        }
    );
    drop(pool);
    assert_eq!(drops.try_recv().unwrap().executed, 1);
}

fn scoped_requests_progress_independently(cancel_slow: bool) {
    let (mut pool, _drops) = pool(config(2, 1, 2));
    let topology = ValidatedParallelTopology::new(
        ParallelTopologyId::new(1),
        2,
        rank(0),
        ParallelismPlan::validated(2, 1, 1, 1, 1, 1).unwrap(),
    )
    .unwrap();
    let slow_rank = pool.route_session(SessionId(10)).unwrap();
    let fast_rank = pool.route_session(SessionId(20)).unwrap();
    assert_eq!((slow_rank, fast_rank), (rank(0), rank(1)));
    let slow_scope = ParticipantSet::new(&topology, [slow_rank]).unwrap();
    let fast_scope = ParticipantSet::new(&topology, [fast_rank]).unwrap();
    let mut coordinator = DistributedTransaction::new(topology, 3);
    let slow = tx(1);
    let fast = tx(2);
    coordinator.begin_scoped(slow, slow_scope.clone()).unwrap();
    coordinator.begin_scoped(fast, fast_scope.clone()).unwrap();
    assert_eq!(coordinator.participants(slow), Ok(&slow_scope));
    assert_eq!(coordinator.participants(fast), Ok(&fast_scope));
    coordinator.reserve(slow, 1).unwrap();
    let slow_operations = coordinator.communicate_all(slow).unwrap();
    let fast_operations = coordinator.communicate_all(fast).unwrap();
    assert_eq!(slow_operations.len(), 1);
    assert_eq!(fast_operations.len(), 1);
    let slow_operation = slow_operations[0];
    let fast_operation = fast_operations[0];
    assert_eq!(
        (slow_operation.transaction, slow_operation.rank),
        (slow, slow_rank)
    );
    assert_eq!(
        (fast_operation.transaction, fast_operation.rank),
        (fast, fast_rank)
    );
    assert_ne!(slow_operation.identity, fast_operation.identity);
    assert_eq!(coordinator.in_use_credits(), 3);

    let (blocked, entered, release) = gated(100);
    assert_eq!(
        pool.try_submit(SessionId(10), slow, blocked).unwrap(),
        slow_rank
    );
    entered.recv_timeout(LIMIT).unwrap();
    if cancel_slow {
        pool.cancel(slow).unwrap();
        // Routing cancellation is intent only, not a second transaction decision.
        assert_eq!(coordinator.state(slow), Ok(TransactionState::Preparing));
        assert_eq!(coordinator.in_use_credits(), 3);
        assert_eq!(pool.outstanding(), 1);
    }

    // The slow rank has not even prepared, and its worker remains gated.
    coordinator.prepare(fast, fast_rank).unwrap();
    assert_eq!(
        pool.try_submit(SessionId(20), fast, input(200)).unwrap(),
        fast_rank
    );
    let ready = completion(&mut pool);
    assert_eq!(ready.session, SessionId(20));
    assert!(!ready.cancellation_requested);
    let reply = success(ready, fast.get(), fast_rank.get());
    assert_eq!(reply.value, 200);
    assert!(!reply.cancelled);
    assert_eq!(pool.outstanding(), 1);
    // Consuming a routing completion neither completes communication nor commits.
    assert_eq!(coordinator.state(fast), Ok(TransactionState::Preparing));
    assert_eq!(
        coordinator.commit_decision(fast),
        Err(DistributedTransactionError::InvalidState)
    );
    coordinator
        .complete(fast_operation, FinalizeOutcome::Success)
        .unwrap();
    assert_eq!(
        coordinator.commit_decision(fast),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(coordinator.drain(), 1);
    assert_eq!(
        coordinator.commit_decision(fast),
        Err(DistributedTransactionError::InvalidState)
    );
    coordinator
        .prepare_vote(fast, fast_rank, FinalizeOutcome::Success)
        .unwrap();
    assert_eq!(coordinator.commit_decision(fast), Ok(Decision::Commit));
    assert_eq!(
        coordinator.publish(fast),
        Err(DistributedTransactionError::InvalidState)
    );
    coordinator
        .finalize(fast, fast_rank, FinalizeOutcome::Success)
        .unwrap();
    coordinator.publish(fast).unwrap();
    assert_eq!(coordinator.state(fast), Ok(TransactionState::Published));
    assert_eq!(coordinator.publication_count(), 1);
    assert_eq!(coordinator.in_use_credits(), 2);
    assert_eq!(
        coordinator.pending_completions(),
        vec![PendingCompletion {
            operation: slow_operation,
            outcome: None
        }]
    );
    assert_eq!(coordinator.pending_ranks(slow).unwrap(), vec![slow_rank]);
    assert_eq!(coordinator.state(slow), Ok(TransactionState::Preparing));
    assert!(pool.poll().is_none());

    // Release only AFTER the other request has published: no world barrier.
    release.send(()).unwrap();
    let late = completion(&mut pool);
    assert_eq!(late.session, SessionId(10));
    assert_eq!(late.cancellation_requested, cancel_slow);
    let reply = success(late, slow.get(), slow_rank.get());
    assert_eq!(reply.value, 100);
    assert_eq!(reply.cancelled, cancel_slow);
    assert_eq!(pool.outstanding(), 0);
    coordinator
        .complete(slow_operation, FinalizeOutcome::Success)
        .unwrap();
    // Cancellation cannot reclaim an operation credit before late completion drain.
    assert_eq!(coordinator.in_use_credits(), 2);
    assert_eq!(coordinator.drain(), 1);
    if cancel_slow {
        assert_eq!(coordinator.state(slow), Ok(TransactionState::Preparing));
        coordinator
            .prepare_vote(slow, slow_rank, FinalizeOutcome::Failure)
            .unwrap();
        assert_eq!(coordinator.commit_decision(slow), Ok(Decision::Abort));
        assert_eq!(
            coordinator.retire(slow),
            Err(DistributedTransactionError::InvalidState)
        );
        coordinator
            .finalize(slow, slow_rank, FinalizeOutcome::Success)
            .unwrap();
        coordinator.cancel(slow).unwrap();
        assert_eq!(coordinator.state(slow), Ok(TransactionState::Cancelled));
        assert_eq!(
            coordinator.commit_decision(slow),
            Err(DistributedTransactionError::InvalidState)
        );
        assert_eq!(
            coordinator.publish(slow),
            Err(DistributedTransactionError::InvalidState)
        );
        assert_eq!(coordinator.publication_count(), 1);
    } else {
        coordinator.prepare(slow, slow_rank).unwrap();
        coordinator
            .prepare_vote(slow, slow_rank, FinalizeOutcome::Success)
            .unwrap();
        assert_eq!(coordinator.commit_decision(slow), Ok(Decision::Commit));
        coordinator
            .finalize(slow, slow_rank, FinalizeOutcome::Success)
            .unwrap();
        coordinator.publish(slow).unwrap();
        assert_eq!(coordinator.state(slow), Ok(TransactionState::Published));
        assert_eq!(coordinator.publication_count(), 2);
    }
    assert_eq!(coordinator.state(fast), Ok(TransactionState::Published));
    assert_eq!(coordinator.in_use_credits(), 0);
    assert!(coordinator.pending_completions().is_empty());
    coordinator.retire(fast).unwrap();
    coordinator.retire(slow).unwrap();
    assert_eq!(coordinator.retained_transaction_count(), 0);
    pool.shutdown().unwrap();
}

#[test]
fn scoped_transaction_publishes_while_another_replica_is_gated() {
    scoped_requests_progress_independently(false);
}

#[test]
fn cancelling_gated_transaction_does_not_block_other_replica_publication() {
    scoped_requests_progress_independently(true);
}

#[test]
fn panic_notification_waits_for_quiescence_hook_before_releasing_host_credit() {
    let (dropped, drops) = mpsc::sync_channel(1);
    let (entered, started) = mpsc::sync_channel(1);
    let (release, wait) = mpsc::sync_channel(1);
    // Move the non-Clone receiver into the one owner's factory via a one-shot slot.
    let gate = std::sync::Arc::new(std::sync::Mutex::new(Some(Gate {
        entered,
        release: wait,
    })));
    let mut pool = Pool::new(config(1, 1, 1), move |rank| {
        let mut worker = LocalWorker::new(rank, dropped.clone());
        worker.quiescence_gate = gate.lock().unwrap().take();
        Ok(worker)
    })
    .unwrap();
    let mut bad = input(1);
    bad.panic = true;
    pool.try_submit(SessionId(1), tx(1), bad).unwrap();
    started.recv_timeout(LIMIT).unwrap();
    assert!(
        pool.poll().is_none(),
        "panic notification preceded its fence"
    );
    assert_eq!(pool.outstanding(), 1);
    assert!(drops.try_recv().is_err());
    release.send(()).unwrap();
    assert!(matches!(
        completion(&mut pool).outcome,
        CompletionOutcome::Panicked
    ));
    assert_eq!(pool.outstanding(), 0);
    let error = pool.shutdown().unwrap_err();
    assert!(matches!(error.failures[0].kind, OwnerFailureKind::Panicked));
    assert_eq!(drops.try_recv().unwrap().executed, 1);
}

// Deliberately omit panic_quiescence: the no-op default shutdown must not be
// mistaken for a fence. The contained CPU worker's safe hook is NOT delegated.
struct DefaultQuiescenceWorker(LocalWorker);
impl ReplicaWorker<Input> for DefaultQuiescenceWorker {
    type Output = Reply;
    type Error = &'static str;
    fn execute(&mut self, request: WorkRequest<Input>) -> Result<Reply, Self::Error> {
        self.0.execute(request)
    }
}

struct PanickingQuiescenceWorker(LocalWorker);
impl ReplicaWorker<Input> for PanickingQuiescenceWorker {
    type Output = Reply;
    type Error = &'static str;
    fn execute(&mut self, request: WorkRequest<Input>) -> Result<Reply, Self::Error> {
        self.0.execute(request)
    }
    fn panic_quiescence(&mut self) -> PanicQuiescence {
        assert_eq!(thread::current().id(), self.0.owner);
        panic!("quiescence hook panicked");
    }
}

#[test]
fn default_unknown_and_hook_panic_quarantine_without_drop_or_replay() {
    for hook_panics in [false, true] {
        let (dropped, drops) = mpsc::sync_channel(2);
        let mut pool = if hook_panics {
            Pool::new(config(2, 2, 2), move |rank| {
                Ok(PanickingQuiescenceWorker(LocalWorker::new(
                    rank,
                    dropped.clone(),
                )))
            })
        } else {
            Pool::new(config(2, 2, 2), move |rank| {
                Ok(DefaultQuiescenceWorker(LocalWorker::new(
                    rank,
                    dropped.clone(),
                )))
            })
        }
        .unwrap();
        let (mut bad, entered, release) = gated(1);
        bad.panic = true;
        pool.try_submit_to(SessionId(1), rank(0), tx(1), bad)
            .unwrap();
        entered.recv_timeout(LIMIT).unwrap();
        pool.try_submit_to(SessionId(1), rank(0), tx(2), input(2))
            .unwrap();
        release.send(()).unwrap();
        let unknown = completion(&mut pool);
        assert_eq!(unknown.transaction, tx(1));
        assert!(matches!(
            unknown.outcome,
            CompletionOutcome::QuiescenceUnknown
        ));
        let queued = completion(&mut pool);
        assert_eq!(queued.transaction, tx(2));
        assert!(matches!(
            queued.outcome,
            CompletionOutcome::ReplicaUnavailable
        ));
        assert_eq!(pool.outstanding(), 0); // Delivery drained, not a GPU fence.
        assert!(pool.poll().is_none());
        assert_eq!(
            pool.route_session_to(SessionId(1), rank(0))
                .unwrap_err()
                .kind,
            SubmitErrorKind::ReplicaUnavailable
        );
        pool.try_submit_to(SessionId(2), rank(1), tx(3), input(3))
            .unwrap();
        success(completion(&mut pool), 3, 1);
        let error = pool.shutdown().unwrap_err();
        assert_eq!(error.failures.len(), 1);
        assert_eq!(error.failures[0].transaction, Some(tx(1)));
        assert!(matches!(
            error.failures[0].kind,
            OwnerFailureKind::QuiescenceUnknown
        ));
        // The healthy owner joined and dropped; the quarantined owner joined
        // without running the inner worker's destructor, even when the hook panics.
        assert_eq!(drops.try_recv().unwrap().rank, rank(1));
        assert!(drops.try_recv().is_err());
        pool.shutdown().unwrap();
        drop(pool);
        assert!(drops.try_recv().is_err(), "quarantine reclaimed a worker");
    }
}
