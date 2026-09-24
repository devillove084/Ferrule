//! Production TP orchestration with CPU-only, owner-local mock workers.
//! No test-owned TP threads, channels, collective or transaction coordinator.

use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use ferrule_common::{
    ParallelGroupId, ParallelRankId, ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology,
};
use ferrule_model::transformer::parallel::{
    TensorParallelLinearPartition as Partition, TensorParallelLinearPlan,
};
use ferrule_runtime::parallel::collective::{HostCollectiveError, HostCollectiveLimits};
use ferrule_runtime::parallel::data::{
    BuildError, CompletionOutcome, DataParallelConfig, DataParallelExecutor, OwnerFailureKind,
    PanicQuiescence, ReplicaWorker, SubmitErrorKind, WorkRequest,
};
use ferrule_runtime::parallel::tensor::{
    TensorCommand, TensorParallelError, TensorParallelExecutor, TensorRank, TensorWork,
};
use ferrule_runtime::{
    DistributedTransactionError, ExecutionTransactionId, SessionId, TransactionState,
};

const INPUT: usize = 7;
const OUTPUT: usize = 5;
const ROWS: usize = 3;
const WAIT: Duration = Duration::from_secs(5);

fn rank(value: usize) -> ParallelRankId {
    ParallelRankId::new(value as u32)
}

fn tx(value: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(value).unwrap()
}

fn topology(degree: usize) -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(11),
        (2 * degree) as u32,
        rank(0),
        ParallelismPlan {
            data_parallel: 2,
            tensor_parallel: degree,
            ..ParallelismPlan::default()
        },
    )
    .unwrap()
}

fn config(degree: usize) -> DataParallelConfig {
    DataParallelConfig {
        replicas: degree,
        max_outstanding_per_replica: 1,
        session_capacity: degree,
    }
}

fn limits(degree: usize) -> HostCollectiveLimits {
    HostCollectiveLimits {
        max_ranks: degree,
        max_elements_per_rank: ROWS * OUTPUT,
        max_host_bytes: 4 * degree * (ROWS * OUTPUT + degree * ROWS * OUTPUT),
    }
}

fn weights() -> Vec<f32> {
    (0..OUTPUT * INPUT)
        .map(|i| (i as i32 % 9 - 4) as f32 / 8.0)
        .collect()
}

fn input(rows: usize) -> Arc<[f32]> {
    (0..rows * INPUT)
        .map(|i| (i + 1) as f32 / 4.0)
        .collect::<Vec<_>>()
        .into()
}

fn oracle(rows: usize) -> Vec<f32> {
    let input = input(rows);
    let weights = weights();
    let mut result = vec![0.0; rows * OUTPUT];
    for row in 0..rows {
        for output in 0..OUTPUT {
            for feature in 0..INPUT {
                result[row * OUTPUT + output] +=
                    input[row * INPUT + feature] * weights[output * INPUT + feature];
            }
            // An observable Apply, distinct from collective output.
            result[row * OUTPUT + output] *= 2.0;
        }
    }
    result
}

fn wait(mut ready: impl FnMut() -> bool) {
    let until = Instant::now() + WAIT;
    while !ready() {
        assert!(Instant::now() < until, "mock worker progress timed out");
        thread::yield_now();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    Compute,
    Apply,
    BadPartial,
    BadApply,
    Panic,
    PanicUnknown,
    HookPanic,
    ApplyPanicUnknown,
    ApplyHookPanic,
    ErrorThenUnknown,
    CancelThenUnknown,
    FatalFence,
    CancelCompute,
    CancelApply,
    Init,
    Shutdown,
}

struct Observations {
    created: Vec<AtomicUsize>,
    dropped: Vec<AtomicUsize>,
    shutdown: Vec<AtomicUsize>,
    panic_hooks: Vec<AtomicUsize>,
    computed: AtomicUsize,
    applied: AtomicUsize,
    turn: AtomicUsize,
    cancellation_seen: AtomicUsize,
    cancel: AtomicBool,
    events: Mutex<Vec<(u64, usize, bool)>>,
}

impl Observations {
    fn new(degree: usize) -> Arc<Self> {
        Arc::new(Self {
            created: (0..degree).map(|_| AtomicUsize::new(0)).collect(),
            dropped: (0..degree).map(|_| AtomicUsize::new(0)).collect(),
            shutdown: (0..degree).map(|_| AtomicUsize::new(0)).collect(),
            panic_hooks: (0..degree).map(|_| AtomicUsize::new(0)).collect(),
            computed: AtomicUsize::new(0),
            applied: AtomicUsize::new(0),
            turn: AtomicUsize::new(0),
            cancellation_seen: AtomicUsize::new(0),
            cancel: AtomicBool::new(false),
            events: Mutex::new(Vec::new()),
        })
    }

    fn reset(&self) {
        self.computed.store(0, Ordering::Release);
        self.applied.store(0, Ordering::Release);
        self.turn.store(0, Ordering::Release);
        self.cancel.store(false, Ordering::Release);
    }

    fn joined(&self, degree: usize) {
        for index in 0..degree {
            assert_eq!(self.created[index].load(Ordering::Acquire), 1);
            assert_eq!(self.dropped[index].load(Ordering::Acquire), 1);
        }
    }
}

struct LinearWorker {
    // Deliberately !Send/!Sync: construction, execution and Drop must stay local.
    owner: Rc<ThreadId>,
    rank: TensorRank,
    plan: TensorParallelLinearPlan,
    weight: Vec<f32>,
    observations: Arc<Observations>,
    order: Vec<usize>,
    fault: Fault,
}

impl ReplicaWorker<TensorWork> for LinearWorker {
    type Output = Vec<f32>;
    type Error = &'static str;

    fn execute(&mut self, request: WorkRequest<TensorWork>) -> Result<Vec<f32>, Self::Error> {
        assert_eq!(*self.owner, thread::current().id());
        assert_eq!(request.rank, self.rank.local);
        assert_eq!(request.input.rank, self.rank);
        let local = self.rank.local.get() as usize;
        let degree = self.plan.ranks();
        let id = request.input.transaction.get();
        let fault = if id == 1 { self.fault } else { Fault::None };
        let o = &self.observations;
        match request.input.command {
            TensorCommand::Compute { input, rows } => {
                o.computed.fetch_add(1, Ordering::AcqRel);
                if matches!(fault, Fault::ErrorThenUnknown | Fault::CancelThenUnknown) {
                    // All requests entered before failure/cancel. Rank one can
                    // only panic after the coordinator has already seen intent.
                    wait(|| o.computed.load(Ordering::Acquire) == degree);
                    if local == 0 && fault == Fault::ErrorThenUnknown {
                        return Err("ordinary error before peer's unknown panic");
                    }
                    if local == 1 {
                        if fault == Fault::CancelThenUnknown {
                            o.cancel.store(true, Ordering::Release);
                        }
                        wait(|| request.cancellation.is_requested());
                        panic!("late panic after error/cancel");
                    }
                }
                if matches!(fault, Fault::CancelCompute) {
                    wait(|| o.computed.load(Ordering::Acquire) == degree);
                    o.cancel.store(true, Ordering::Release);
                    wait(|| request.cancellation.is_requested());
                    o.cancellation_seen.fetch_add(1, Ordering::AcqRel);
                    return Err("cancelled compute");
                }
                if local == 0 {
                    match fault {
                        Fault::Compute => return Err("compute failed"),
                        Fault::Panic | Fault::PanicUnknown | Fault::HookPanic => {
                            panic!("mock compute panic")
                        }
                        Fault::FatalFence => std::panic::panic_any(PanicQuiescence::Unknown),
                        _ => {}
                    }
                }
                let input = self
                    .plan
                    .shard_input(self.rank.local, &input, rows)
                    .unwrap();
                let (out, features) = self.plan.local_shape(self.rank.local).unwrap();
                let mut partial = vec![0.0; rows * out];
                for row in 0..rows {
                    for output in 0..out {
                        for feature in 0..features {
                            partial[row * out + output] += input[row * features + feature]
                                * self.weight[output * features + feature];
                        }
                    }
                }
                if !self.order.is_empty() {
                    let position = self.order.iter().position(|&r| r == local).unwrap();
                    wait(|| o.turn.load(Ordering::Acquire) == position);
                }
                o.events.lock().unwrap().push((id, local, false));
                if !self.order.is_empty() {
                    o.turn.fetch_add(1, Ordering::AcqRel);
                }
                if local == 0 && fault == Fault::BadPartial {
                    partial.pop();
                }
                Ok(partial)
            }
            TensorCommand::Apply { mut values, rows } => {
                assert_eq!(values.len(), rows * OUTPUT);
                o.applied.fetch_add(1, Ordering::AcqRel);
                if matches!(fault, Fault::ApplyPanicUnknown | Fault::ApplyHookPanic) {
                    wait(|| o.applied.load(Ordering::Acquire) == degree);
                    if local == 0 {
                        panic!("mock Apply panic after collective Ready");
                    }
                }
                if fault == Fault::CancelApply {
                    wait(|| o.applied.load(Ordering::Acquire) == degree);
                    o.cancel.store(true, Ordering::Release);
                    wait(|| request.cancellation.is_requested());
                    o.cancellation_seen.fetch_add(1, Ordering::AcqRel);
                    return Err("cancelled apply");
                }
                if local == 0 && fault == Fault::Apply {
                    return Err("apply failed");
                }
                for value in &mut values {
                    *value *= 2.0;
                }
                o.events.lock().unwrap().push((id, local, true));
                if local == 0 && fault == Fault::BadApply {
                    values.pop();
                }
                Ok(values)
            }
        }
    }

    fn panic_quiescence(&mut self) -> PanicQuiescence {
        assert_eq!(*self.owner, thread::current().id());
        self.observations.panic_hooks[self.rank.local.get() as usize]
            .fetch_add(1, Ordering::AcqRel);
        match self.fault {
            Fault::HookPanic | Fault::ApplyHookPanic => panic!("panic in quiescence hook"),
            Fault::PanicUnknown
            | Fault::ApplyPanicUnknown
            | Fault::ErrorThenUnknown
            | Fault::CancelThenUnknown => PanicQuiescence::Unknown,
            // Pure CPU worker. FatalFence intentionally tests that the explicit
            // Unknown signal cannot be downgraded by this normally safe hook.
            _ => PanicQuiescence::Quiescent,
        }
    }

    fn shutdown(&mut self) -> Result<(), Self::Error> {
        assert_eq!(*self.owner, thread::current().id());
        self.observations.shutdown[self.rank.local.get() as usize].fetch_add(1, Ordering::AcqRel);
        if self.fault == Fault::Shutdown {
            Err("shutdown failed")
        } else {
            Ok(())
        }
    }
}

impl Drop for LinearWorker {
    fn drop(&mut self) {
        assert_eq!(*self.owner, thread::current().id());
        self.observations.dropped[self.rank.local.get() as usize].fetch_add(1, Ordering::AcqRel);
    }
}

fn executor(
    degree: usize,
    partition: Partition,
    order: Vec<usize>,
    fault: Fault,
    observations: Arc<Observations>,
) -> Result<TensorParallelExecutor<LinearWorker>, TensorParallelError<&'static str>> {
    let plan = TensorParallelLinearPlan::new(OUTPUT, INPUT, degree, partition).unwrap();
    let worker_plan = plan.clone();
    let caller = thread::current().id();
    TensorParallelExecutor::new(
        topology(degree),
        1, // Nonzero replica catches accidental local/global-rank conflation.
        plan,
        config(degree),
        move |rank: TensorRank| {
            assert_ne!(caller, thread::current().id());
            assert_eq!(rank.global.get(), degree as u32 + rank.local.get());
            if fault == Fault::Init && rank.local.get() == 1 {
                return Err("initialization failed");
            }
            observations.created[rank.local.get() as usize].fetch_add(1, Ordering::AcqRel);
            Ok(LinearWorker {
                owner: Rc::new(thread::current().id()),
                rank,
                weight: worker_plan.shard_weight(rank.local, &weights()).unwrap().0,
                plan: worker_plan,
                observations,
                order,
                fault,
            })
        },
        ParallelGroupId::new(9),
        limits(degree),
    )
}

fn drained(executor: &TensorParallelExecutor<LinearWorker>) {
    assert_eq!(executor.outstanding(), 0);
    assert_eq!(executor.coordinator().in_use_credits(), 0);
    assert_eq!(executor.coordinator().retained_transaction_count(), 0);
    assert_eq!(executor.coordinator().retained_operation_count(), 0);
    assert!(executor.coordinator().pending_completions().is_empty());
    assert!(executor.collective().active_descriptor().is_none());
    assert_eq!(executor.collective().owned_host_bytes(), 0);
    assert_eq!(executor.collective().reserved_host_bytes(), 0);
}

#[test]
fn ragged_column_and_row_multiple_rows_and_return_orders() {
    for degree in [2, 3, 4] {
        for partition in [Partition::Column, Partition::Row] {
            for order in [(0..degree).collect::<Vec<_>>(), (0..degree).rev().collect()] {
                let observations = Observations::new(degree);
                let mut executor = executor(
                    degree,
                    partition,
                    order.clone(),
                    Fault::None,
                    Arc::clone(&observations),
                )
                .unwrap();
                assert_eq!(
                    executor.collective().members(),
                    &(degree..2 * degree).map(rank).collect::<Vec<_>>()
                );
                for (id, rows) in [(10, 1), (20, ROWS)] {
                    observations.reset();
                    // Only TP routing slots exist. Different bases prove routes
                    // are released rather than accumulating across transactions.
                    let output = executor
                        .execute(tx(id), SessionId(id * 10), input(rows), rows)
                        .unwrap();
                    assert_eq!(output.transaction, tx(id));
                    assert_eq!(output.ranks.len(), degree);
                    for (local, (rank, values)) in output.ranks.iter().enumerate() {
                        assert_eq!(rank.local.get() as usize, local);
                        assert_eq!(rank.global.get() as usize, degree + local);
                        assert_eq!(*values, oracle(rows));
                    }
                    let returned: Vec<_> = observations
                        .events
                        .lock()
                        .unwrap()
                        .iter()
                        .filter(|&&(transaction, _, apply)| transaction == id && !apply)
                        .map(|&(_, rank, _)| rank)
                        .collect();
                    assert_eq!(returned, order);
                    assert_eq!(observations.applied.load(Ordering::Acquire), degree);
                    drained(&executor);
                }
                assert_eq!(executor.coordinator().publication_count(), 2);
                executor.shutdown().unwrap();
                executor.shutdown().unwrap();
                observations.joined(degree);
                assert!(
                    observations
                        .shutdown
                        .iter()
                        .all(|n| n.load(Ordering::Acquire) == 1)
                );
            }
        }
    }
}

#[test]
fn failure_and_bad_descriptors_drain_retire_and_allow_next_transaction() {
    for partition in [Partition::Column, Partition::Row] {
        for fault in [
            Fault::Compute,
            Fault::Apply,
            Fault::BadPartial,
            Fault::BadApply,
        ] {
            let observations = Observations::new(3);
            let mut executor =
                executor(3, partition, vec![], fault, Arc::clone(&observations)).unwrap();
            let error = executor
                .execute(tx(1), SessionId(0), input(ROWS), ROWS)
                .unwrap_err();
            match fault {
                Fault::Compute | Fault::Apply => assert!(matches!(error,
                    TensorParallelError::Worker { rank: TensorRank { local, global }, .. }
                    if local == rank(0) && global == rank(3))),
                Fault::BadPartial | Fault::BadApply => {
                    assert!(matches!(error, TensorParallelError::Shape(_)))
                }
                _ => unreachable!(),
            }
            drained(&executor);
            // An Abort decision was published, never a partial successful result.
            assert_eq!(executor.coordinator().publication_count(), 1);
            if fault == Fault::Compute || fault == Fault::BadPartial {
                assert_eq!(observations.applied.load(Ordering::Acquire), 0);
            }
            assert!(matches!(
                executor.execute(tx(1), SessionId(0), input(ROWS), ROWS),
                Err(TensorParallelError::Transaction(
                    DistributedTransactionError::InvalidTransaction
                ))
            ));
            observations.reset();
            let output = executor
                .execute(tx(2), SessionId(100), input(ROWS), ROWS)
                .unwrap();
            assert!(
                output
                    .ranks
                    .iter()
                    .all(|(_, values)| *values == oracle(ROWS))
            );
            drained(&executor);
            executor.shutdown().unwrap();
            observations.joined(3);
        }
    }
}

#[test]
fn cooperative_cancel_in_compute_and_apply_drains_every_accepted_completion() {
    for fault in [Fault::CancelCompute, Fault::CancelApply] {
        let observations = Observations::new(3);
        let mut executor = executor(
            3,
            Partition::Column,
            vec![],
            fault,
            Arc::clone(&observations),
        )
        .unwrap();
        let result = executor.execute_cancellable(
            tx(1),
            SessionId(0),
            input(ROWS),
            ROWS,
            &observations.cancel,
        );
        assert!(matches!(result, Err(TensorParallelError::Cancelled)));
        assert_eq!(observations.cancellation_seen.load(Ordering::Acquire), 3);
        assert_eq!(executor.coordinator().publication_count(), 0);
        if fault == Fault::CancelApply {
            // Collective Ready was reached; cancellation must still win over Commit.
            assert_eq!(observations.applied.load(Ordering::Acquire), 3);
            assert_eq!(executor.collective().sequence_watermark(), Some(1));
        }
        drained(&executor);
        assert!(matches!(
            executor.execute(tx(1), SessionId(0), input(ROWS), ROWS),
            Err(TensorParallelError::Transaction(
                DistributedTransactionError::InvalidTransaction
            ))
        ));
        observations.reset();
        executor
            .execute(tx(2), SessionId(0), input(ROWS), ROWS)
            .unwrap();
        drained(&executor);
        drop(executor);
        observations.joined(3);
    }
}

#[test]
fn precancel_shape_limits_id_exhaustion_and_closed_pool() {
    let observations = Observations::new(2);
    let mut executor = executor(
        2,
        Partition::Row,
        vec![],
        Fault::None,
        Arc::clone(&observations),
    )
    .unwrap();
    assert!(matches!(
        executor.execute(tx(1), SessionId(u64::MAX), input(1), 1),
        Err(TensorParallelError::IdentityExhausted)
    ));
    assert!(matches!(
        executor.execute(tx(1), SessionId(0), input(1), 0),
        Err(TensorParallelError::Shape(_))
    ));
    assert!(matches!(
        executor.execute(tx(1), SessionId(0), input(ROWS + 1), ROWS + 1),
        Err(TensorParallelError::Collective(
            HostCollectiveError::CountLimit
        ))
    ));
    // Preflight rejections consume no outer ID.
    assert!(matches!(
        executor.execute_cancellable(tx(1), SessionId(0), input(1), 1, &AtomicBool::new(true)),
        Err(TensorParallelError::Cancelled)
    ));
    assert_eq!(observations.computed.load(Ordering::Acquire), 0);
    drained(&executor);
    executor
        .execute(tx(u64::MAX), SessionId(0), input(1), 1)
        .unwrap();
    assert!(matches!(
        executor.execute(tx(u64::MAX), SessionId(0), input(1), 1),
        Err(TensorParallelError::Transaction(
            DistributedTransactionError::InvalidTransaction
        ))
    ));
    drained(&executor);
    executor.shutdown().unwrap();
    assert!(matches!(
        executor.execute(tx(2), SessionId(0), input(1), 1),
        Err(TensorParallelError::Route {
            kind: SubmitErrorKind::Closed,
            ..
        })
    ));
    observations.joined(2);
}

#[test]
fn panic_shutdown_failure_initialization_failure_and_drop_join_all_owners() {
    for fault in [Fault::Panic, Fault::Shutdown, Fault::None] {
        let observations = Observations::new(3);
        let mut executor =
            executor(3, Partition::Row, vec![], fault, Arc::clone(&observations)).unwrap();
        let result = executor.execute(tx(1), SessionId(0), input(ROWS), ROWS);
        if fault == Fault::Panic {
            assert!(matches!(
                result,
                Err(TensorParallelError::WorkerPanicked(_))
            ));
            assert!(matches!(
                executor.execute(tx(2), SessionId(0), input(ROWS), ROWS),
                Err(TensorParallelError::Route {
                    kind: SubmitErrorKind::ReplicaUnavailable,
                    ..
                })
            ));
        } else {
            result.unwrap();
        }
        drained(&executor);
        if fault != Fault::None {
            let error = executor.shutdown().unwrap_err();
            assert_eq!(
                error.failures.len(),
                if fault == Fault::Shutdown { 3 } else { 1 }
            );
            executor.shutdown().unwrap();
        }
        drop(executor);
        observations.joined(3);
    }
    let observations = Observations::new(3);
    assert!(matches!(
        executor(
            3,
            Partition::Row,
            vec![],
            Fault::Init,
            Arc::clone(&observations)
        ),
        Err(TensorParallelError::Pool(BuildError::Owners(_)))
    ));
    assert_eq!(observations.created[0].load(Ordering::Acquire), 1);
    assert_eq!(observations.dropped[0].load(Ordering::Acquire), 1);
    assert_eq!(observations.created[1].load(Ordering::Acquire), 0);
    assert_eq!(observations.created[2].load(Ordering::Acquire), 0);
}

struct Echo;
impl ReplicaWorker<bool> for Echo {
    type Output = bool;
    type Error = ();
    fn execute(&mut self, request: WorkRequest<bool>) -> Result<bool, ()> {
        assert!(!request.input, "explicit route owner panic");
        Ok(request.input)
    }
    fn panic_quiescence(&mut self) -> PanicQuiescence {
        // Synchronous CPU-only echo, no asynchronous resource accesses.
        PanicQuiescence::Quiescent
    }
}

#[test]
fn explicit_routes_validate_rank_stickiness_liveness_and_admit_atomically() {
    let mut pool = DataParallelExecutor::new(config(2), |_| Ok(Echo)).unwrap();
    let first = SessionId(10);
    let fresh = SessionId(11);
    assert_eq!(pool.route_session_to(first, rank(1)).unwrap(), rank(1));
    assert_eq!(pool.session_rank(first), None); // Preflight installs nothing.
    assert_eq!(
        pool.route_session_to(first, rank(2)).unwrap_err().kind,
        SubmitErrorKind::InvalidRank
    );
    assert_eq!(
        pool.try_submit_to(first, rank(2), tx(1), false)
            .unwrap_err()
            .kind,
        SubmitErrorKind::InvalidRank
    );
    assert_eq!(pool.session_rank(first), None);
    pool.try_submit_to(first, rank(1), tx(1), false).unwrap();
    assert_eq!(pool.session_rank(first), Some(rank(1)));
    assert_eq!(
        pool.route_session_to(first, rank(0)).unwrap_err().kind,
        SubmitErrorKind::StickyMismatch { bound: rank(1) }
    );
    assert_eq!(
        pool.try_submit_to(first, rank(0), tx(2), false)
            .unwrap_err()
            .kind,
        SubmitErrorKind::StickyMismatch { bound: rank(1) }
    );
    // Even a ready but unconsumed result owns capacity. Rank zero is idle, but
    // explicit routing must reject, never silently choose that least-loaded rank.
    assert_eq!(
        pool.try_submit_to(fresh, rank(1), tx(2), false)
            .unwrap_err()
            .kind,
        SubmitErrorKind::Backpressure
    );
    assert_eq!(pool.session_rank(fresh), None);
    wait(|| pool.poll().is_some());
    assert_eq!(
        pool.try_submit_to(fresh, rank(0), tx(1), false)
            .unwrap_err()
            .kind,
        SubmitErrorKind::RetiredOrOutOfOrder {
            high_watermark: tx(1)
        }
    );
    assert_eq!(pool.session_rank(fresh), None);
    pool.try_submit_to(fresh, rank(0), tx(2), true).unwrap();
    wait(|| {
        pool.poll()
            .is_some_and(|completion| matches!(completion.outcome, CompletionOutcome::Panicked))
    });
    assert_eq!(
        pool.route_session_to(fresh, rank(0)).unwrap_err().kind,
        SubmitErrorKind::ReplicaUnavailable
    );
    assert_eq!(
        pool.try_submit_to(fresh, rank(0), tx(3), false)
            .unwrap_err()
            .kind,
        SubmitErrorKind::ReplicaUnavailable
    );
    pool.release_session(first).unwrap();
    pool.release_session(fresh).unwrap();
    // The original implicit route still binds immediately and ignores dead ranks.
    assert_eq!(pool.route_session(SessionId(20)).unwrap(), rank(1));
    assert_eq!(pool.session_rank(SessionId(20)), Some(rank(1)));
    pool.shutdown().unwrap_err();
    assert_eq!(
        pool.route_session_to(SessionId(21), rank(1))
            .unwrap_err()
            .kind,
        SubmitErrorKind::Closed
    );
    assert_eq!(pool.outstanding(), 0);
}

#[test]
fn unknown_quiescence_retains_operation_custody_and_closes_executor() {
    for partition in [Partition::Column, Partition::Row] {
        for fault in [
            Fault::PanicUnknown,
            Fault::HookPanic,
            Fault::ApplyPanicUnknown,
            Fault::ApplyHookPanic,
            Fault::ErrorThenUnknown,
            Fault::CancelThenUnknown,
            Fault::FatalFence,
        ] {
            let degree = 3;
            let failed_local =
                if matches!(fault, Fault::ErrorThenUnknown | Fault::CancelThenUnknown) {
                    1
                } else {
                    0
                };
            let failed = TensorRank {
                local: rank(failed_local),
                global: rank(degree + failed_local),
            };
            let observations = Observations::new(degree);
            let mut executor =
                executor(degree, partition, vec![], fault, Arc::clone(&observations)).unwrap();
            let result = executor.execute_cancellable(
                tx(1),
                SessionId(0),
                input(ROWS),
                ROWS,
                &observations.cancel,
            );
            assert!(
                matches!(result, Err(TensorParallelError::QuiescenceUnknown(rank)) if rank == failed)
            );
            assert!(executor.is_quarantined());
            assert_eq!(
                executor.outstanding(),
                0,
                "all host notifications must be consumed"
            );
            assert_eq!(
                executor.coordinator().state(tx(1)).unwrap(),
                TransactionState::Preparing
            );
            assert_eq!(executor.coordinator().publication_count(), 0);
            assert_eq!(executor.coordinator().retained_transaction_count(), 1);
            assert_eq!(executor.coordinator().retained_operation_count(), degree);
            assert_eq!(executor.coordinator().in_use_credits(), 1);
            let pending = executor.coordinator().pending_completions();
            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0].operation.rank, failed.global);
            assert_eq!(pending[0].operation.transaction, tx(1));
            assert_eq!(
                pending[0].outcome, None,
                "Unknown is NOT a completed Failure"
            );
            assert_eq!(
                executor.coordinator().pending_ranks(tx(1)).unwrap().len(),
                degree
            );
            assert!(executor.collective().active_descriptor().is_none());
            assert_eq!(executor.collective().owned_host_bytes(), 0);
            if matches!(fault, Fault::ApplyPanicUnknown | Fault::ApplyHookPanic) {
                assert_eq!(observations.applied.load(Ordering::Acquire), degree);
                assert_eq!(executor.collective().sequence_watermark(), Some(1));
            }
            // execute already joined every owner: healthy peers ran shutdown and
            // Drop, but the unknown owner ran its hook once and never ran either.
            for local in 0..degree {
                assert_eq!(observations.created[local].load(Ordering::Acquire), 1);
                assert_eq!(
                    observations.panic_hooks[local].load(Ordering::Acquire),
                    usize::from(local == failed_local)
                );
                let safe = usize::from(local != failed_local);
                assert_eq!(observations.shutdown[local].load(Ordering::Acquire), safe);
                assert_eq!(observations.dropped[local].load(Ordering::Acquire), safe);
            }
            assert!(matches!(
                executor.execute(tx(2), SessionId(100), input(ROWS), ROWS),
                Err(TensorParallelError::Quarantined)
            ));
            let error = executor.shutdown().unwrap_err();
            assert_eq!(error.failures.len(), 1);
            assert_eq!(error.failures[0].rank, failed.local); // DP owner rank is local.
            assert!(matches!(
                error.failures[0].kind,
                OwnerFailureKind::QuiescenceUnknown
            ));
            executor.shutdown().unwrap();
            assert_eq!(executor.coordinator().pending_completions(), pending);
            assert_eq!(executor.coordinator().in_use_credits(), 1);
            assert_eq!(executor.coordinator().retained_transaction_count(), 1);
            assert_eq!(executor.coordinator().retained_operation_count(), degree);
            assert_eq!(executor.coordinator().publication_count(), 0);
            assert!(matches!(
                executor.execute_cancellable(
                    tx(3),
                    SessionId(0),
                    input(ROWS),
                    ROWS,
                    &AtomicBool::new(true)
                ),
                Err(TensorParallelError::Quarantined)
            ));
            drop(executor);
            assert_eq!(
                observations.dropped[failed_local].load(Ordering::Acquire),
                0
            );
        }
    }
}

fn composed_topology() -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(81),
        4,
        rank(0),
        ParallelismPlan::validated(1, 2, 2, 1, 1, 2).unwrap(),
    )
    .unwrap()
}

#[test]
fn pp2_tp2_ep2_tensor_stage_uses_global_members_not_kv_ownership() {
    for stage in 0..2 {
        let observations = Observations::new(2);
        let worker_observations = Arc::clone(&observations);
        let plan = TensorParallelLinearPlan::new(OUTPUT, INPUT, 2, Partition::Row).unwrap();
        let worker_plan = plan.clone();
        let mut executor = TensorParallelExecutor::new_at_stage(
            composed_topology(),
            0,
            stage,
            plan,
            config(2),
            move |tensor_rank: TensorRank| {
                assert_eq!(
                    tensor_rank.global.get(),
                    stage * 2 + tensor_rank.local.get()
                );
                worker_observations.created[tensor_rank.local.get() as usize]
                    .fetch_add(1, Ordering::AcqRel);
                Ok::<_, &'static str>(LinearWorker {
                    owner: Rc::new(thread::current().id()),
                    rank: tensor_rank,
                    weight: worker_plan
                        .shard_weight(tensor_rank.local, &weights())
                        .unwrap()
                        .0,
                    plan: worker_plan,
                    observations: worker_observations,
                    order: vec![1, 0],
                    fault: Fault::None,
                })
            },
            ParallelGroupId::new(9),
            limits(2),
        )
        .unwrap();
        let scopes = executor.execution_scopes();
        assert_eq!(
            scopes.kv_participants().iter().collect::<Vec<_>>(),
            [rank(0), rank(1), rank(2), rank(3)]
        );
        assert_eq!(
            executor.collective_participants().as_participants(),
            &composed_topology()
                .tensor_stage_participants(0, stage)
                .unwrap()
        );
        assert_eq!(
            scopes.validate_kv_participants(executor.collective_participants().as_participants()),
            Err(ferrule_common::ParallelTopologyError::ScopeKindMismatch)
        );
        assert_eq!(
            executor.collective().members(),
            &[rank(stage as usize * 2), rank(stage as usize * 2 + 1)]
        );
        let output = executor
            .execute(tx(10), SessionId(90), input(ROWS), ROWS)
            .unwrap();
        for (tensor_rank, values) in output.ranks {
            assert_eq!(
                tensor_rank.global.get(),
                stage * 2 + tensor_rank.local.get()
            );
            assert_eq!(values, oracle(ROWS));
        }
        assert_eq!(executor.coordinator().publication_count(), 1);
        drained(&executor);
        executor.shutdown().unwrap();
        observations.joined(2);
    }
}

#[test]
fn pp2_tp2_ep2_kv_transaction_waits_for_all_mesh_owners_not_expert_dispatch() {
    use ferrule_model::decoder::KvCommitBinding;
    use ferrule_model::transformer::expert_parallel::{ExpertDispatchLimits, ExpertPlacement};
    use ferrule_runtime::parallel::expert::ExpertGroup;
    use ferrule_runtime::{Decision, DistributedTransaction, FinalizeOutcome};

    let topology = composed_topology();
    let mut scopes = topology.execution_scopes(0).unwrap();
    for stage in 0..2 {
        let members = vec![rank(101 + stage * 10), rank(100 + stage * 10)];
        let group = ExpertGroup {
            source_rank: members[1],
            members: members.clone(),
            layers: stage..stage + 1,
            placement: ExpertPlacement::new([(stage, 0, members[0]), (stage, 1, members[1])])
                .unwrap(),
            limits: ExpertDispatchLimits {
                max_tokens: 4,
                max_bytes: 64,
            },
        };
        scopes
            .attach_expert_dispatch_members(
                group.dispatch_members(&topology, 0, stage as u32).unwrap(),
            )
            .unwrap();
        let attached = scopes
            .expert_dispatch_members(stage as u32, 0)
            .unwrap()
            .unwrap();
        assert_eq!(attached.iter().collect::<Vec<_>>(), members);
        assert_eq!(attached.source_rank(), group.source_rank);
    }
    let kv = scopes.kv_participants().as_participants().clone();
    let binding = KvCommitBinding::new(tx(42), topology.topology_id(), kv.clone(), 1).unwrap();
    let mut coordinator = DistributedTransaction::new(topology.clone(), 4);
    coordinator.begin_scoped(tx(42), kv.clone()).unwrap();
    assert_eq!(coordinator.participants(tx(42)), Ok(binding.participants()));
    assert_eq!(
        coordinator.pending_ranks(tx(42)).unwrap(),
        [rank(0), rank(1), rank(2), rank(3)]
    );
    for outsider in [100, 101, 110, 111, 4] {
        assert_eq!(
            coordinator.prepare(tx(42), rank(outsider)),
            Err(DistributedTransactionError::InvalidRank)
        );
        assert_eq!(
            coordinator.communicate(tx(42), rank(outsider)),
            Err(DistributedTransactionError::InvalidRank)
        );
        assert_eq!(
            coordinator.prepare_vote(tx(42), rank(outsider), FinalizeOutcome::Success),
            Err(DistributedTransactionError::InvalidRank)
        );
    }
    for stage in 0..2 {
        let tp = scopes.tensor_collective_participants(stage, 0).unwrap();
        for owner in tp.iter() {
            coordinator.prepare(tx(42), owner).unwrap();
            coordinator
                .prepare_vote(tx(42), owner, FinalizeOutcome::Success)
                .unwrap();
        }
        if stage == 0 {
            assert_eq!(
                coordinator.commit_decision(tx(42)),
                Err(DistributedTransactionError::InvalidState)
            );
        }
    }
    assert_eq!(coordinator.commit_decision(tx(42)), Ok(Decision::Commit));
    for owner in scopes.tensor_collective_participants(0, 0).unwrap().iter() {
        coordinator
            .finalize(tx(42), owner, FinalizeOutcome::Success)
            .unwrap();
    }
    assert_eq!(
        coordinator.publish(tx(42)),
        Err(DistributedTransactionError::InvalidState)
    );
    assert_eq!(
        coordinator.pending_ranks(tx(42)).unwrap(),
        [rank(2), rank(3)]
    );
    // Stage one's local slots 0/1 are NOT owners 2/3. Full coordinates resolve them.
    for tensor in 0..2 {
        let coordinate = topology.coordinate(0, 1, tensor).unwrap();
        let owner = scopes.kv_owner(coordinate).unwrap();
        assert_ne!(owner, rank(tensor as usize));
        coordinator
            .finalize(tx(42), owner, FinalizeOutcome::Success)
            .unwrap();
    }
    for stage in 0..2 {
        for expert in scopes
            .expert_dispatch_members(stage, 0)
            .unwrap()
            .unwrap()
            .iter()
        {
            assert_eq!(
                coordinator.finalize(tx(42), expert, FinalizeOutcome::Success),
                Err(DistributedTransactionError::InvalidRank)
            );
        }
    }
    coordinator.publish(tx(42)).unwrap();
    assert_eq!(coordinator.publication_count(), 1);
    coordinator.retire(tx(42)).unwrap();
    assert_eq!(coordinator.retained_transaction_count(), 0);
    // Changed EP metadata is a different transaction identity, despite identical KV IDs.
    let foreign = ValidatedParallelTopology::new(
        topology.topology_id(),
        4,
        rank(0),
        ParallelismPlan::validated(1, 2, 3, 1, 1, 2).unwrap(),
    )
    .unwrap();
    assert_eq!(
        coordinator.begin_scoped(
            tx(43),
            foreign
                .execution_scopes(0)
                .unwrap()
                .kv_participants()
                .as_participants()
                .clone()
        ),
        Err(DistributedTransactionError::InvalidScope)
    );
    coordinator.begin_scoped(tx(43), kv).unwrap();
}
