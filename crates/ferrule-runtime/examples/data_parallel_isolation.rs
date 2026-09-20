//! CPU-only acceptance check for the release replica worker unwind boundary.
//! Run: cargo run --locked --release -p ferrule-runtime --example data_parallel_isolation
//! The injected panic still prints through Rust's default panic hook.

use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;
use std::time::{Duration, Instant};

use ferrule_runtime::parallel::data::{
    CompletionOutcome, DataParallelConfig, DataParallelExecutor, HostCompletion, OwnerFailureKind,
    PanicQuiescence, ReplicaWorker, WorkRequest,
};
use ferrule_runtime::{ExecutionTransactionId, ParallelRankId, SessionId};

const TIMEOUT: Duration = Duration::from_secs(10);

type Executor = DataParallelExecutor<Input, u32, &'static str>;
type Completion = HostCompletion<u32, &'static str>;

#[derive(Debug)]
enum Input {
    Panic {
        entered: SyncSender<()>,
        release: Receiver<()>,
    },
    Value(u32),
}

struct Worker {
    rank: ParallelRankId,
    calls: u32,
    stopped: bool,
    dropped: SyncSender<(u32, u32, bool)>,
}

impl ReplicaWorker<Input> for Worker {
    type Output = u32;
    type Error = &'static str;

    fn execute(&mut self, request: WorkRequest<Input>) -> Result<u32, Self::Error> {
        self.calls += 1;
        match request.input {
            Input::Panic { entered, release } => {
                entered
                    .try_send(())
                    .map_err(|_| "gate notification failed")?;
                release
                    .recv_timeout(TIMEOUT)
                    .map_err(|_| "panic gate was not released in time")?;
                panic!("intentional replica worker panic: isolation acceptance check");
            }
            Input::Value(value) => Ok(value),
        }
    }

    fn panic_quiescence(&mut self) -> PanicQuiescence {
        // Pure CPU work: no device or other asynchronous users can outlive execute.
        PanicQuiescence::Quiescent
    }

    fn shutdown(&mut self) -> Result<(), Self::Error> {
        self.stopped = true;
        Ok(())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // One slot per owner; never block or panic during unwinding/cleanup.
        let _ = self
            .dropped
            .try_send((self.rank.get(), self.calls, self.stopped));
    }
}

fn transaction(value: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(value).expect("example uses nonzero transaction IDs")
}

fn completion(executor: &mut Executor, id: u64, rank: u32) -> Result<Completion, String> {
    let deadline = Instant::now() + TIMEOUT;
    while Instant::now() < deadline {
        if let Some(completion) = executor.poll() {
            if completion.transaction != transaction(id)
                || completion.rank != ParallelRankId::new(rank)
                || completion.session != SessionId(u64::from(rank) + 1)
                || completion.cancellation_requested
            {
                return Err(format!("unexpected completion identity: {completion:?}"));
            }
            return Ok(completion);
        }
        thread::yield_now();
    }
    Err(format!(
        "timed out waiting for transaction {id} on rank {rank}"
    ))
}

fn exercise(executor: &mut Executor) -> Result<(), String> {
    let (entered, started) = mpsc::sync_channel(1);
    let (release, wait) = mpsc::sync_channel(1);
    let panic_rank = executor
        .try_submit(
            SessionId(1),
            transaction(1),
            Input::Panic {
                entered,
                release: wait,
            },
        )
        .map_err(|error| format!("panic request admission failed: {error:?}"))?;
    if panic_rank != ParallelRankId::new(0) {
        return Err(format!("unexpected panic rank: {panic_rank:?}"));
    }
    started
        .recv_timeout(TIMEOUT)
        .map_err(|error| format!("worker did not enter panic gate: {error}"))?;

    // The gate guarantees this request is queued BEFORE the owner panics.
    let queued_rank = executor
        .try_submit(SessionId(1), transaction(2), Input::Value(99))
        .map_err(|error| format!("queued request admission failed: {error:?}"))?;
    if queued_rank != panic_rank {
        return Err("queued request migrated away from its sticky rank".into());
    }
    release
        .try_send(())
        .map_err(|error| format!("cannot release panic gate: {error}"))?;
    let panicked = completion(executor, 1, 0)?;
    if !matches!(panicked.outcome, CompletionOutcome::Panicked) {
        return Err(format!("expected Panicked, got {panicked:?}"));
    }
    let queued = completion(executor, 2, 0)?;
    if !matches!(queued.outcome, CompletionOutcome::ReplicaUnavailable) {
        return Err(format!(
            "expected queued ReplicaUnavailable, got {queued:?}"
        ));
    }

    // Admit new work only AFTER observing the other owner's failure.
    let healthy_rank = executor
        .try_submit(SessionId(2), transaction(3), Input::Value(42))
        .map_err(|error| format!("healthy request admission failed: {error:?}"))?;
    if healthy_rank != ParallelRankId::new(1) {
        return Err(format!("unexpected healthy rank: {healthy_rank:?}"));
    }
    let healthy = completion(executor, 3, 1)?;
    if !matches!(healthy.outcome, CompletionOutcome::Success(42)) {
        return Err(format!("expected healthy Success(42), got {healthy:?}"));
    }
    if executor.outstanding() != 0 || executor.poll().is_some() {
        return Err("outstanding or duplicate completions remain".into());
    }
    Ok(())
}

fn main() -> Result<(), String> {
    if !cfg!(panic = "unwind") {
        return Err(
            "replica isolation requires panic=unwind; do not override the release profile".into(),
        );
    }
    let (dropped, drops) = mpsc::sync_channel(2);
    let mut executor = Executor::new(
        DataParallelConfig {
            replicas: 2,
            max_outstanding_per_replica: 2,
            session_capacity: 2,
        },
        move |rank| {
            Ok(Worker {
                rank,
                calls: 0,
                stopped: false,
                dropped: dropped.clone(),
            })
        },
    )
    .map_err(|error| format!("executor initialization failed: {error:?}"))?;

    let result = exercise(&mut executor);
    // Join every owner even if an acceptance check above failed.
    let shutdown = executor.shutdown();
    result?;
    let error = shutdown.expect_err("shutdown must report the isolated worker panic");
    if error.failures.len() != 1
        || error.failures[0].rank != ParallelRankId::new(0)
        || error.failures[0].transaction != Some(transaction(1))
        || !matches!(error.failures[0].kind, OwnerFailureKind::Panicked)
    {
        return Err(format!("unexpected shutdown failure report: {error:?}"));
    }
    let mut owners = Vec::with_capacity(2);
    for _ in 0..2 {
        owners.push(
            drops
                .try_recv()
                .map_err(|error| format!("owner was not dropped: {error}"))?,
        );
    }
    owners.sort_unstable();
    // The failed worker never executes its queued request or shutdown hook.
    if owners != [(0, 1, false), (1, 1, true)] || drops.try_recv().is_ok() {
        return Err(format!("unexpected owner lifecycle: {owners:?}"));
    }
    executor
        .shutdown()
        .map_err(|error| format!("repeated shutdown was not a no-op: {error:?}"))?;
    println!(
        "PASS: panic isolated, queued work failed, other replica succeeded, all owners joined"
    );
    Ok(())
}
