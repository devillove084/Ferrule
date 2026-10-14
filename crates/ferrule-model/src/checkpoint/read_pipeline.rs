//! One positioned-read worker, one queued request, and bounded host custody.
//!
//! `submit` captures the original plan, never fresh source snapshots. The worker
//! opens a `VerifiedReadSession`, reads its ordered extents, and runs its mandatory
//! post boundary before publishing. `poll` does no I/O. An owner can submit the
//! next read, execute current compute, then wait/take; this is not an asynchronous
//! runtime forward API. A blocking baseline uses the same pipeline and waits
//! immediately after submission.
//!
//! Slots and requested bytes remain charged until a taken completion is dropped,
//! including failed and cancelled operations. Cancellation cannot interrupt an OS
//! read. Unproved worker exits or mismatched completions quarantine their charge;
//! neither cancellation nor shutdown refunds it. There is no payload cache.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::{self, JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use ferrule_common::async_wake::{CompletionHub, CompletionListener};
use ferrule_common::{Error, OperationId, SourceGeneration};

use super::{CheckpointReadPlan, CheckpointTensorReader};

static NEXT_OPERATION: AtomicU64 = AtomicU64::new(1);

/// Explicit default for both blocking and overlapped schedules, not the OS default.
pub const DEFAULT_READ_WORKER_STACK_BYTES: usize = 1024 * 1024;

/// Limits on the configured request to `Builder::stack_size`, not measured RSS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadPipelineOptions {
    pub worker_stack_bytes: usize,
    pub max_worker_stack_bytes: usize,
}

impl Default for ReadPipelineOptions {
    fn default() -> Self {
        Self {
            worker_stack_bytes: DEFAULT_READ_WORKER_STACK_BYTES,
            max_worker_stack_bytes: DEFAULT_READ_WORKER_STACK_BYTES,
        }
    }
}

/// Requested stack reservation. OS minimums, rounding, guard pages, TLS and
/// runtime/allocator overhead are unobserved; this is not an RSS guarantee.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReadWorkerStackSnapshot {
    pub configured_bytes: usize,
    pub limit_bytes: usize,
    pub reserved_bytes: usize,
    pub peak_bytes: usize,
    pub unknown_bytes: usize,
    pub os_overhead_bytes: Option<u64>,
}

/// Payload-byte and operation-slot bounds, including results held by the caller.
/// Plan metadata is bounded by slot count and the caller's already-built plans;
/// the byte limit covers packed read buffers, not downstream decoded copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadPipelineLimits {
    pub max_host_bytes: u64,
    pub max_read_slots: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum ReadPipelineError {
    #[error("invalid checkpoint read pipeline configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("checkpoint read pipeline is closed")]
    Closed,
    #[error("checkpoint read worker's depth-one queue is full")]
    QueueFull,
    #[error(
        "checkpoint read budget exhausted: requested {requested_bytes} bytes, available {available_bytes} bytes and {available_slots} slots"
    )]
    BudgetExhausted {
        requested_bytes: u64,
        available_bytes: u64,
        available_slots: usize,
    },
    #[error("checkpoint read size {bytes} exceeds reader/host limit {limit}")]
    SizeLimit { bytes: u64, limit: u64 },
    #[error("checkpoint read operation IDs exhausted")]
    IdExhausted,
    #[error("checkpoint read ticket does not match an owned operation")]
    TicketMismatch,
    #[error("checkpoint read operation is not complete")]
    NotReady,
    #[error("checkpoint read completion has already been taken")]
    AlreadyTaken,
    #[error("checkpoint read custody is unproved and remains charged: {0:?}")]
    UnknownCustody(ReadUnknownReason),
    #[error("spawn checkpoint read worker: {0}")]
    Spawn(std::io::Error),
}

/// Exact operation, source-image generation, catalog snapshots and extent order.
/// IDs are process-wide monotonic, never recycled, including rejected submissions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadTicket {
    operation: OperationId,
    source_generation: SourceGeneration,
    plan: CheckpointReadPlan,
}

impl ReadTicket {
    pub const fn operation(&self) -> OperationId {
        self.operation
    }

    pub const fn source_generation(&self) -> SourceGeneration {
        self.source_generation
    }

    pub fn read_plan(&self) -> &CheckpointReadPlan {
        &self.plan
    }
}

/// All instants use the same process monotonic clock as caller compute windows.
/// The I/O interval encloses allocation and positioned read syscalls, not opening
/// handles or source checks. A failed/partial read has no proved byte count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadReceipt {
    pub operation: OperationId,
    pub source_generation: SourceGeneration,
    pub requested_bytes: u64,
    /// Successfully packed bytes; not a partial-read syscall counter.
    pub io_bytes: Option<u64>,
    /// Bytes published in this result (zero on cancellation/failure/unknown).
    pub payload_bytes: u64,
    pub submitted_at: Instant,
    pub worker_started_at: Option<Instant>,
    pub io_started_at: Option<Instant>,
    pub io_finished_at: Option<Instant>,
    pub source_verified_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub taken_at: Option<Instant>,
    pub cancellation_requested: bool,
}

impl ReadReceipt {
    /// Host-clock overlap of the actual read envelope with a caller window.
    /// This alone is not proof of GPU execution; use correlated GPU events.
    pub fn io_overlap(&self, start: Instant, end: Instant) -> Option<Duration> {
        if end < start {
            return None;
        }
        let left = self.io_started_at?.max(start);
        let right = self.io_finished_at?.min(end);
        Some(right.saturating_duration_since(left))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadStage {
    Queued,
    Reading,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadUnknownReason {
    WorkerPanicked,
    WorkerStopped,
    CompletionIdentityMismatch,
    JoinFailed,
    StatePoisoned,
    NotificationPanicked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadPoll {
    Pending(ReadStage),
    Ready(ReadReceipt),
    Taken,
    Unknown {
        receipt: ReadReceipt,
        reason: ReadUnknownReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadCancellation {
    BeforeStart,
    AfterStart,
    AfterCompletion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadCancelStatus {
    Requested,
    Suppressed,
    AlreadyTerminal,
    RetainedUnknown,
}

#[derive(Debug)]
pub enum ReadOutcome {
    SourceVerified(Vec<u8>),
    Failed(Error),
    Cancelled(ReadCancellation),
}

/// Non-cloneable host custody. Payloads cannot be moved out independently of the
/// charge. Keep this guard through upload/consumption; downstream copies require
/// their own budget. Field order drops the payload before releasing its slot.
#[derive(Debug)]
pub struct ReadCompletion {
    outcome: ReadOutcome,
    ticket: ReadTicket,
    receipt: ReadReceipt,
    _custody: Custody,
}

impl ReadCompletion {
    pub fn outcome(&self) -> &ReadOutcome {
        &self.outcome
    }

    pub fn ticket(&self) -> &ReadTicket {
        &self.ticket
    }

    pub const fn receipt(&self) -> &ReadReceipt {
        &self.receipt
    }

    // The proof stays private to checkpoint consumers: public raw bytes cannot
    // be relabelled as a verified read. No I/O, charge transfer or release here.
    pub(super) fn verified_packed_range(
        &self,
        expected: &ReadTicket,
        plan: &CheckpointReadPlan,
        range: std::ops::Range<usize>,
    ) -> ferrule_common::Result<&[u8]> {
        let invalid = |message: &str| Error::Model {
            message: format!("checkpoint verified packed read: {message}"),
        };
        if self.ticket != *expected
            || self.receipt.operation != expected.operation
            || self.receipt.source_generation != expected.source_generation
        {
            return Err(invalid("operation/plan/source generation ticket mismatch"));
        }
        let bytes = self
            .payload()
            .ok_or_else(|| invalid("completion is not source verified"))?;
        if self.receipt.source_verified_at.is_none()
            || self.receipt.requested_bytes != expected.plan.storage_bytes()
            || bytes.len() as u64 != self.receipt.requested_bytes
            || self.receipt.io_bytes != Some(bytes.len() as u64)
            || self.receipt.payload_bytes != bytes.len() as u64
        {
            return Err(invalid(
                "completion payload length or source-boundary proof mismatch",
            ));
        }
        if range.start >= range.end
            || range.end > bytes.len()
            || (range.end - range.start) as u64 != plan.storage_bytes()
            || plan
                .source_files()
                .iter()
                .any(|source| !expected.plan.source_files().contains(source))
        {
            return Err(invalid(
                "packed range/source snapshots do not match paired plan",
            ));
        }
        let mut cursor = 0usize;
        let mut start = None;
        let mut end = None;
        for (index, extent) in expected.plan.extents().iter().enumerate() {
            if cursor == range.start {
                start = Some(index);
            }
            cursor = cursor
                .checked_add(
                    usize::try_from(extent.bytes())
                        .map_err(|_| invalid("extent does not fit host address space"))?,
                )
                .ok_or_else(|| invalid("packed extent offset overflows host address space"))?;
            if cursor == range.end {
                end = Some(index + 1);
                break;
            }
            if cursor > range.end {
                break;
            }
        }
        let (Some(start), Some(end)) = (start, end) else {
            return Err(invalid("packed range must follow exact extent boundaries"));
        };
        if expected.plan.extents()[start..end] != *plan.extents() {
            return Err(invalid("packed range has mismatched ordered extents"));
        }
        Ok(&bytes[range])
    }

    pub fn payload(&self) -> Option<&[u8]> {
        match &self.outcome {
            ReadOutcome::SourceVerified(bytes) => Some(bytes),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReadBudgetSnapshot {
    pub reserved_bytes: u64,
    pub reserved_slots: usize,
    pub peak_bytes: u64,
    pub peak_slots: usize,
    pub unknown_bytes: u64,
    pub unknown_slots: usize,
    pub worker_stack: ReadWorkerStackSnapshot,
}

impl ReadBudgetSnapshot {
    /// Sum of payload and requested-stack reservations without u64 overflow.
    /// This excludes unobserved OS/allocator overhead and separately staged artifacts.
    pub const fn requested_host_reservation_bytes(&self) -> u128 {
        self.reserved_bytes as u128 + self.worker_stack.reserved_bytes as u128
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadUnknownOperation {
    pub ticket: ReadTicket,
    pub receipt: ReadReceipt,
    pub reason: ReadUnknownReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadShutdown {
    pub join_failed: bool,
    pub join_pending: bool,
    pub join_handoff: Option<ReadJoinHandoff>,
    pub budget: ReadBudgetSnapshot,
    pub unknown: Vec<ReadUnknownOperation>,
}

/// Retains teardown custody if an inline waker destroys/shuts down its pipeline
/// on the read worker. Capture before arming callbacks; join it OFF that worker.
/// Discarding every handoff does not detach/refund an unjoined worker: custody
/// remains quarantined until an authorized off-worker join can reclaim it.
#[derive(Clone)]
pub struct ReadJoinHandoff {
    shared: Arc<Shared>,
}

impl std::fmt::Debug for ReadJoinHandoff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadJoinHandoff")
            .field("worker", &self.shared.worker_id.get())
            .finish_non_exhaustive()
    }
}
impl PartialEq for ReadJoinHandoff {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.shared, &other.shared)
    }
}
impl Eq for ReadJoinHandoff {}

impl ReadJoinHandoff {
    /// Join a deferred shutdown. Calling from the read worker or while another
    /// thread is joining returns `join_pending`, without waiting or refunding.
    /// Before shutdown/drop there is no deferred handle, so it is also pending.
    pub fn join(&self) -> ReadShutdown {
        self.shared.join_deferred();
        self.shared.shutdown_snapshot()
    }

    pub fn budget(&self) -> ReadBudgetSnapshot {
        self.shared.lock().snapshot()
    }
}

#[derive(Default)]
struct JoinCustody {
    deferred: Option<JoinHandle<()>>,
    joining: bool,
    joined: bool,
    failed: bool,
    owner_dropped: bool,
    // An explicit retained cycle protects the handle if the owner is destroyed
    // on the worker. Only an actual off-worker join clears this ownership.
    retained: Option<Arc<Shared>>,
}

impl std::fmt::Debug for JoinCustody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JoinCustody")
            .field("deferred", &self.deferred.is_some())
            .field("joining", &self.joining)
            .field("joined", &self.joined)
            .field("failed", &self.failed)
            .field("owner_dropped", &self.owner_dropped)
            .field("retained", &self.retained.is_some())
            .finish()
    }
}

#[derive(Debug)]
enum Phase {
    Queued,
    Reading,
    Complete(ReadOutcome),
    Taken,
    Unknown(ReadUnknownReason),
}

#[derive(Debug)]
struct Entry {
    ticket: ReadTicket,
    receipt: ReadReceipt,
    phase: Phase,
}

#[derive(Debug)]
struct State {
    entries: BTreeMap<OperationId, Entry>,
    budget: ReadBudgetSnapshot,
    accepting: bool,
}

impl State {
    fn quarantine_pending(&mut self, reason: ReadUnknownReason) {
        self.accepting = false;
        for entry in self.entries.values_mut() {
            if matches!(entry.phase, Phase::Queued | Phase::Reading) {
                entry.phase = Phase::Unknown(reason);
                entry.receipt.payload_bytes = 0;
                entry.receipt.completed_at = Some(Instant::now());
            }
        }
    }

    fn snapshot(&self) -> ReadBudgetSnapshot {
        let mut snapshot = self.budget;
        for entry in self.entries.values() {
            if matches!(entry.phase, Phase::Unknown(_)) {
                snapshot.unknown_bytes += entry.receipt.requested_bytes;
                snapshot.unknown_slots += 1;
            }
        }
        snapshot
    }

    fn matching(&self, ticket: &ReadTicket) -> Result<&Entry, ReadPipelineError> {
        self.entries
            .get(&ticket.operation)
            .filter(|entry| entry.ticket == *ticket)
            .ok_or(ReadPipelineError::TicketMismatch)
    }

    fn poll(&self, ticket: &ReadTicket) -> Result<ReadPoll, ReadPipelineError> {
        let entry = self.matching(ticket)?;
        Ok(match entry.phase {
            Phase::Queued => ReadPoll::Pending(ReadStage::Queued),
            Phase::Reading => ReadPoll::Pending(ReadStage::Reading),
            Phase::Complete(_) => ReadPoll::Ready(entry.receipt),
            Phase::Taken => ReadPoll::Taken,
            Phase::Unknown(reason) => ReadPoll::Unknown {
                receipt: entry.receipt,
                reason,
            },
        })
    }

    fn release(&mut self, operation: OperationId) {
        if let Some(entry) = self.entries.remove(&operation) {
            self.budget.reserved_bytes -= entry.receipt.requested_bytes;
            self.budget.reserved_slots -= 1;
        }
    }
}

#[derive(Debug)]
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
    wake: CompletionHub,
    worker_id: OnceLock<ThreadId>,
    join: Mutex<JoinCustody>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        loop {
            match self.state.lock() {
                Ok(guard) => return guard,
                Err(poisoned) => self.recover_poison(poisoned.into_inner()),
            }
        }
    }

    fn wait<'a>(&'a self, state: MutexGuard<'a, State>) -> MutexGuard<'a, State> {
        match self.changed.wait(state) {
            Ok(guard) => guard,
            Err(poisoned) => {
                self.recover_poison(poisoned.into_inner());
                self.lock()
            }
        }
    }

    fn recover_poison(&self, mut guard: MutexGuard<'_, State>) {
        guard.quarantine_pending(ReadUnknownReason::StatePoisoned);
        self.state.clear_poison();
        drop(guard);
        // AtomicWaker invokes arbitrary user code inline; NEVER retain a state
        // or join mutex across close/notify, including poison recovery.
        self.changed.notify_all();
        self.close_wake();
    }

    fn on_worker(&self) -> bool {
        self.worker_id
            .get()
            .is_some_and(|id| *id == thread::current().id())
    }

    fn join_lock(&self) -> MutexGuard<'_, JoinCustody> {
        self.join
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn hand_off(self: &Arc<Self>, handle: JoinHandle<()>) {
        {
            let mut join = self.join_lock();
            join.deferred = Some(handle);
            join.retained = Some(Arc::clone(self));
        }
        if !self.on_worker() {
            self.join_deferred();
        }
    }

    fn join_deferred(self: &Arc<Self>) {
        if self.on_worker() {
            return;
        }
        let handle = {
            let mut join = self.join_lock();
            if join.joining {
                return;
            }
            let Some(handle) = join.deferred.take() else {
                return;
            };
            join.joining = true;
            handle
        };
        // Neither mutex is held: the exiting worker may synchronously reenter
        // through arbitrary wake callbacks, including another shutdown/join.
        let failed = handle.join().is_err();
        {
            let mut state = self.lock();
            if failed {
                state.quarantine_pending(ReadUnknownReason::JoinFailed);
                state.budget.worker_stack.unknown_bytes = state.budget.worker_stack.reserved_bytes;
            } else {
                state.budget.worker_stack.reserved_bytes = 0;
            }
        }
        let dispose = {
            let mut join = self.join_lock();
            join.joining = false;
            join.joined = true;
            join.failed = failed;
            join.retained.take();
            join.owner_dropped
        };
        if dispose {
            self.discard_proved();
        }
        self.changed.notify_all();
        self.close_wake();
    }

    fn discard_proved(&self) {
        let mut state = self.lock();
        let ready = state
            .entries
            .iter()
            .filter_map(|(id, entry)| matches!(entry.phase, Phase::Complete(_)).then_some(*id))
            .collect::<Vec<_>>();
        for id in ready {
            state.release(id);
        }
    }

    fn shutdown_snapshot(self: &Arc<Self>) -> ReadShutdown {
        let (join_pending, join_failed) = {
            let join = self.join_lock();
            (!join.joined, join.failed)
        };
        let state = self.lock();
        ReadShutdown {
            join_failed,
            join_pending,
            join_handoff: join_pending.then(|| ReadJoinHandoff {
                shared: Arc::clone(self),
            }),
            budget: state.snapshot(),
            unknown: state
                .entries
                .values()
                .filter_map(|entry| {
                    if let Phase::Unknown(reason) = entry.phase {
                        Some(ReadUnknownOperation {
                            ticket: entry.ticket.clone(),
                            receipt: entry.receipt,
                            reason,
                        })
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }

    fn notify(&self) {
        self.deliver_wake(false);
    }
    fn close_wake(&self) {
        self.deliver_wake(true);
    }

    fn deliver_wake(&self, close: bool) {
        self.changed.notify_all();
        // A user waker can panic, including while WorkerExit is unwinding. Do
        // not permit that to turn a retained read failure into a process abort.
        let notified = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if close {
                self.wake.close();
            } else {
                self.wake.notify();
            }
        }));
        if notified.is_err() {
            self.lock()
                .quarantine_pending(ReadUnknownReason::NotificationPanicked);
            self.changed.notify_all();
            if !close {
                self.close_wake();
            }
        }
    }
}

#[derive(Debug)]
struct Custody {
    shared: Arc<Shared>,
    operation: OperationId,
}

impl Drop for Custody {
    fn drop(&mut self) {
        self.shared.lock().release(self.operation);
        self.shared.notify();
    }
}

#[derive(Debug)]
struct WorkerExit(Arc<Shared>);

impl Drop for WorkerExit {
    fn drop(&mut self) {
        let reason = if thread::panicking() {
            ReadUnknownReason::WorkerPanicked
        } else {
            ReadUnknownReason::WorkerStopped
        };
        self.0.lock().quarantine_pending(reason);
        self.0.changed.notify_all();
        self.0.close_wake();
    }
}

#[derive(Debug)]
struct ReadWork {
    ticket: ReadTicket,
    dispatch: tracing::Dispatch,
}

/// Single owner; only the worker performs checkpoint I/O. A bounded request
/// channel has depth one, independent of the configurable custody-slot bound.
/// `wait` and `shutdown` block; `poll`, `take`, `cancel` never perform disk I/O.
#[derive(Debug)]
pub struct CheckpointReadPipeline {
    limits: ReadPipelineLimits,
    max_read_bytes: u64,
    options: ReadPipelineOptions,
    shared: Arc<Shared>,
    sender: Option<SyncSender<ReadWork>>,
    worker: Option<JoinHandle<()>>,
}

impl CheckpointReadPipeline {
    pub fn new(
        reader: CheckpointTensorReader,
        limits: ReadPipelineLimits,
    ) -> Result<Self, ReadPipelineError> {
        Self::start(
            reader,
            limits,
            #[cfg(test)]
            None,
        )
    }

    /// Use identical options for both schedules. Payload limits remain separate
    /// from the configured worker-stack cap, preserving the existing `new` API.
    pub fn with_options(
        reader: CheckpointTensorReader,
        limits: ReadPipelineLimits,
        options: ReadPipelineOptions,
    ) -> Result<Self, ReadPipelineError> {
        Self::start_configured(
            reader,
            limits,
            options,
            #[cfg(test)]
            None,
        )
    }

    fn start(
        reader: CheckpointTensorReader,
        limits: ReadPipelineLimits,
        #[cfg(test)] hook: Option<TestHook>,
    ) -> Result<Self, ReadPipelineError> {
        Self::start_configured(
            reader,
            limits,
            ReadPipelineOptions::default(),
            #[cfg(test)]
            hook,
        )
    }

    fn start_configured(
        reader: CheckpointTensorReader,
        limits: ReadPipelineLimits,
        options: ReadPipelineOptions,
        #[cfg(test)] hook: Option<TestHook>,
    ) -> Result<Self, ReadPipelineError> {
        if options.worker_stack_bytes == 0
            || options.max_worker_stack_bytes == 0
            || options.worker_stack_bytes > options.max_worker_stack_bytes
            || options.worker_stack_bytes > isize::MAX as usize
        {
            return Err(ReadPipelineError::InvalidConfig(
                "worker stack request must be nonzero and within its configured cap and host address space",
            ));
        }
        if limits.max_host_bytes == 0 || limits.max_read_slots == 0 {
            return Err(ReadPipelineError::InvalidConfig("bounds must be nonzero"));
        }
        let max_read_bytes = reader.max_tensor_bytes().min(isize::MAX as u64);
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                entries: BTreeMap::new(),
                budget: ReadBudgetSnapshot {
                    worker_stack: ReadWorkerStackSnapshot {
                        configured_bytes: options.worker_stack_bytes,
                        limit_bytes: options.max_worker_stack_bytes,
                        reserved_bytes: options.worker_stack_bytes,
                        peak_bytes: options.worker_stack_bytes,
                        unknown_bytes: 0,
                        os_overhead_bytes: None,
                    },
                    ..ReadBudgetSnapshot::default()
                },
                accepting: true,
            }),
            changed: Condvar::new(),
            wake: CompletionHub::new(),
            worker_id: OnceLock::new(),
            join: Mutex::new(JoinCustody::default()),
        });
        let (sender, receiver) = mpsc::sync_channel::<ReadWork>(1);
        let worker_shared = Arc::clone(&shared);
        let spawned = thread::Builder::new()
            .name("checkpoint-read".into())
            .stack_size(options.worker_stack_bytes)
            .spawn(move || {
                let _exit = WorkerExit(Arc::clone(&worker_shared));
                while let Ok(work) = receiver.recv() {
                    // Capture per submission, not at worker spawn: embedded
                    // callers and scoped test subscribers retain their owner.
                    let published = tracing::dispatcher::with_default(&work.dispatch, || {
                        read_one(
                            &reader,
                            &worker_shared,
                            work.ticket,
                            #[cfg(test)]
                            hook.as_ref(),
                        )
                    });
                    if !published {
                        break;
                    }
                }
            });
        let worker = finish_worker_start(&shared, spawned)?;
        let _ = shared.worker_id.set(worker.thread().id());
        Ok(Self {
            limits,
            max_read_bytes,
            options,
            shared,
            sender: Some(sender),
            worker: Some(worker),
        })
    }

    /// The generation labels the caller's source image, not a freshly captured
    /// snapshot. The complete immutable plan additionally participates in exact
    /// ticket matching, so a reused generation cannot bless changed sources.
    pub fn submit(
        &mut self,
        source_generation: SourceGeneration,
        plan: CheckpointReadPlan,
    ) -> Result<ReadTicket, ReadPipelineError> {
        if source_generation.is_zero() {
            return Err(ReadPipelineError::InvalidConfig(
                "source generation must be nonzero",
            ));
        }
        let mut state = self.shared.lock();
        if !state.accepting {
            return Err(ReadPipelineError::Closed);
        }
        let Some(sender) = self.sender.as_ref() else {
            state.accepting = false;
            return Err(ReadPipelineError::Closed);
        };
        let bytes = plan.storage_bytes();
        if bytes > self.max_read_bytes {
            return Err(ReadPipelineError::SizeLimit {
                bytes,
                limit: self.max_read_bytes,
            });
        }
        let available_bytes = self.limits.max_host_bytes - state.budget.reserved_bytes;
        let available_slots = self.limits.max_read_slots - state.budget.reserved_slots;
        // Compare before adding; even a u64::MAX configured budget cannot wrap.
        if bytes > available_bytes || available_slots == 0 {
            return Err(ReadPipelineError::BudgetExhausted {
                requested_bytes: bytes,
                available_bytes,
                available_slots,
            });
        }
        let operation = allocate_operation(&NEXT_OPERATION)?;
        let ticket = ReadTicket {
            operation,
            source_generation,
            plan,
        };
        state.entries.insert(
            ticket.operation,
            Entry {
                ticket: ticket.clone(),
                receipt: ReadReceipt {
                    operation: ticket.operation,
                    source_generation,
                    requested_bytes: bytes,
                    io_bytes: None,
                    payload_bytes: 0,
                    submitted_at: Instant::now(),
                    worker_started_at: None,
                    io_started_at: None,
                    io_finished_at: None,
                    source_verified_at: None,
                    completed_at: None,
                    taken_at: None,
                    cancellation_requested: false,
                },
                phase: Phase::Queued,
            },
        );
        state.budget.reserved_bytes += bytes;
        state.budget.reserved_slots += 1;
        let work = ReadWork {
            ticket: ticket.clone(),
            dispatch: tracing::dispatcher::get_default(Clone::clone),
        };
        match sender.try_send(work) {
            Ok(()) => {
                state.budget.peak_bytes = state.budget.peak_bytes.max(state.budget.reserved_bytes);
                state.budget.peak_slots = state.budget.peak_slots.max(state.budget.reserved_slots);
                Ok(ticket)
            }
            Err(error) => {
                // The channel proves this particular request was never accepted.
                state.release(ticket.operation);
                match error {
                    TrySendError::Full(_) => Err(ReadPipelineError::QueueFull),
                    TrySendError::Disconnected(_) => {
                        state.quarantine_pending(ReadUnknownReason::WorkerStopped);
                        drop(state);
                        self.shared.changed.notify_all();
                        self.shared.close_wake();
                        Err(ReadPipelineError::Closed)
                    }
                }
            }
        }
    }

    /// Capture before an inline callback can destroy the pipeline on its worker.
    pub fn join_handoff(&self) -> ReadJoinHandoff {
        ReadJoinHandoff {
            shared: Arc::clone(&self.shared),
        }
    }

    pub const fn options(&self) -> ReadPipelineOptions {
        self.options
    }

    pub fn budget(&self) -> ReadBudgetSnapshot {
        self.shared.lock().snapshot()
    }

    /// Arm BEFORE polling/draining; await only if the owner found no ready work.
    /// One owner listener at a time, following CompletionHub's existing contract.
    pub fn listen(&self) -> CompletionListener {
        self.shared.wake.listen()
    }

    pub fn poll(&self, ticket: &ReadTicket) -> Result<ReadPoll, ReadPipelineError> {
        self.shared.lock().poll(ticket)
    }

    pub fn wait(&self, ticket: &ReadTicket) -> Result<ReadPoll, ReadPipelineError> {
        let mut state = self.shared.lock();
        loop {
            let result = state.poll(ticket)?;
            if !matches!(result, ReadPoll::Pending(_)) {
                return Ok(result);
            }
            state = self.shared.wait(state);
        }
    }

    pub fn take(&mut self, ticket: &ReadTicket) -> Result<ReadCompletion, ReadPipelineError> {
        let mut state = self.shared.lock();
        match state.poll(ticket)? {
            ReadPoll::Pending(_) => return Err(ReadPipelineError::NotReady),
            ReadPoll::Taken => return Err(ReadPipelineError::AlreadyTaken),
            ReadPoll::Unknown { reason, .. } => {
                return Err(ReadPipelineError::UnknownCustody(reason));
            }
            ReadPoll::Ready(_) => {}
        }
        let entry = state
            .entries
            .get_mut(&ticket.operation)
            .expect("matched ticket");
        let Phase::Complete(outcome) = std::mem::replace(&mut entry.phase, Phase::Taken) else {
            unreachable!("ready completion under owner lock");
        };
        entry.receipt.taken_at = Some(Instant::now());
        Ok(ReadCompletion {
            outcome,
            ticket: ticket.clone(),
            receipt: entry.receipt,
            _custody: Custody {
                shared: Arc::clone(&self.shared),
                operation: ticket.operation,
            },
        })
    }

    pub fn cancel(&mut self, ticket: &ReadTicket) -> Result<ReadCancelStatus, ReadPipelineError> {
        let mut state = self.shared.lock();
        state.matching(ticket)?;
        let entry = state
            .entries
            .get_mut(&ticket.operation)
            .expect("matched ticket");
        if matches!(entry.phase, Phase::Taken) {
            return Err(ReadPipelineError::AlreadyTaken);
        }
        entry.receipt.cancellation_requested = true;
        let status = match &entry.phase {
            Phase::Queued | Phase::Reading => ReadCancelStatus::Requested,
            Phase::Complete(ReadOutcome::SourceVerified(_)) => {
                entry.phase =
                    Phase::Complete(ReadOutcome::Cancelled(ReadCancellation::AfterCompletion));
                entry.receipt.payload_bytes = 0;
                ReadCancelStatus::Suppressed
            }
            Phase::Complete(_) => ReadCancelStatus::AlreadyTerminal,
            Phase::Unknown(_) => ReadCancelStatus::RetainedUnknown,
            Phase::Taken => unreachable!(),
        };
        drop(state);
        self.shared.notify();
        Ok(status)
    }

    /// Take all currently proved completions, without waiting or refunding them.
    /// Unknown operations remain in custody and are reported by shutdown/poll.
    pub fn drain(&mut self) -> Vec<ReadCompletion> {
        let tickets = self
            .shared
            .lock()
            .entries
            .values()
            .filter(|entry| matches!(entry.phase, Phase::Complete(_)))
            .map(|entry| entry.ticket.clone())
            .collect::<Vec<_>>();
        tickets
            .iter()
            .map(|ticket| self.take(ticket).expect("single owner ready completion"))
            .collect()
    }

    /// Stop admissions, cancel queued/in-flight work, and join the worker. OS
    /// positioned reads cannot be forcibly interrupted, so joining may block.
    /// Proved results remain takeable; unknown charges are never refunded. On
    /// the read worker, return a retained `join_pending` handoff instead of
    /// self-joining. Finish that handoff on a supervising non-worker thread.
    pub fn shutdown(&mut self) -> ReadShutdown {
        {
            let mut state = self.shared.lock();
            state.accepting = false;
            for entry in state.entries.values_mut() {
                if matches!(entry.phase, Phase::Queued | Phase::Reading) {
                    entry.receipt.cancellation_requested = true;
                }
            }
        }
        // Closing the queue precedes all callbacks and joins. A worker-side
        // callback cannot self-join; ownership is handed off without detaching.
        self.sender.take();
        self.shared.changed.notify_all();
        if let Some(worker) = self.worker.take() {
            self.shared.hand_off(worker);
        } else {
            self.shared.join_deferred();
        }
        self.shared.changed.notify_all();
        self.shared.close_wake();
        self.shared.shutdown_snapshot()
    }
}

impl Drop for CheckpointReadPipeline {
    fn drop(&mut self) {
        self.shared.join_lock().owner_dropped = true;
        let shutdown = self.shutdown();
        if !shutdown.join_pending {
            self.shared.discard_proved();
        }
        // A deferred self-drop keeps all custody until the off-worker join.
        // Unknown and caller-held results are never discarded as proved work.
    }
}

fn finish_worker_start(
    shared: &Shared,
    spawned: std::io::Result<JoinHandle<()>>,
) -> Result<JoinHandle<()>, ReadPipelineError> {
    spawned.map_err(|error| {
        let mut state = shared.lock();
        state.accepting = false;
        // A spawn error proves no worker started and no operation was admitted.
        state.budget.worker_stack.reserved_bytes = 0;
        drop(state);
        shared.close_wake();
        ReadPipelineError::Spawn(error)
    })
}

fn allocate_operation(next: &AtomicU64) -> Result<OperationId, ReadPipelineError> {
    next.try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map(OperationId::new)
        .map_err(|_| ReadPipelineError::IdExhausted)
}

fn read_one(
    reader: &CheckpointTensorReader,
    shared: &Arc<Shared>,
    ticket: ReadTicket,
    #[cfg(test)] hook: Option<&TestHook>,
) -> bool {
    #[cfg(test)]
    if !call_hook(hook, TestPhase::Dequeued, &mut ticket.clone()) {
        return false;
    }
    let mut receipt = {
        let mut state = shared.lock();
        let Some(entry) = state.entries.get_mut(&ticket.operation) else {
            return false;
        };
        if entry.ticket != ticket || !matches!(entry.phase, Phase::Queued) {
            state.quarantine_pending(ReadUnknownReason::CompletionIdentityMismatch);
            return false;
        }
        if !entry.receipt.cancellation_requested {
            entry.phase = Phase::Reading;
            entry.receipt.worker_started_at = Some(Instant::now());
        }
        entry.receipt
    };
    let outcome = if receipt.cancellation_requested {
        ReadOutcome::Cancelled(ReadCancellation::BeforeStart)
    } else {
        let result = reader
            .verified_read_session(ticket.read_plan())
            .and_then(|session| {
                #[cfg(test)]
                call_hook(hook, TestPhase::SessionOpened, &mut ticket.clone());
                session.read_incrementally(|session| {
                    receipt.io_started_at = Some(Instant::now());
                    #[cfg(test)]
                    call_hook(hook, TestPhase::IoStarted, &mut ticket.clone());
                    let result = {
                        // Actual worker allocation/pread envelope, excluding
                        // session opening and pre/post source verification.
                        let _range = tracing::debug_span!(target: "ferrule_profile", "read_subset",
                            operation = ticket.operation.get(),
                            source_generation = ticket.source_generation.get(),
                            requested_bytes = ticket.read_plan().storage_bytes())
                        .entered();
                        session.read_subset(ticket.read_plan())
                    };
                    receipt.io_finished_at = Some(Instant::now());
                    if let Ok(bytes) = &result {
                        receipt.io_bytes = Some(bytes.len() as u64);
                    }
                    #[cfg(test)]
                    call_hook(hook, TestPhase::PayloadRead, &mut ticket.clone());
                    result
                })
            });
        match result {
            Ok(bytes) => {
                receipt.source_verified_at = Some(Instant::now());
                receipt.payload_bytes = bytes.len() as u64;
                ReadOutcome::SourceVerified(bytes)
            }
            Err(error) => ReadOutcome::Failed(error),
        }
    };
    let observed_ticket = ticket.clone();
    #[cfg(test)]
    let observed_ticket = {
        let mut observed = observed_ticket;
        call_hook(hook, TestPhase::BeforePublish, &mut observed);
        observed
    };
    let published = publish(shared, &ticket, &observed_ticket, receipt, outcome);
    #[cfg(test)]
    call_hook(hook, TestPhase::AfterPublish, &mut ticket.clone());
    published
}

fn publish(
    shared: &Shared,
    expected: &ReadTicket,
    observed: &ReadTicket,
    mut receipt: ReadReceipt,
    mut outcome: ReadOutcome,
) -> bool {
    let mut state = shared.lock();
    let Some(entry) = state.entries.get_mut(&expected.operation) else {
        state.quarantine_pending(ReadUnknownReason::CompletionIdentityMismatch);
        return false;
    };
    if matches!(entry.phase, Phase::Unknown(_)) {
        return false;
    }
    let expected_payload_bytes = match &outcome {
        ReadOutcome::SourceVerified(bytes) => bytes.len() as u64,
        _ => 0,
    };
    if entry.ticket != *expected
        || *observed != *expected
        || receipt.operation != expected.operation
        || receipt.source_generation != expected.source_generation
        || receipt.requested_bytes != expected.plan.storage_bytes()
        || receipt.submitted_at != entry.receipt.submitted_at
        || receipt.payload_bytes != expected_payload_bytes
        || (matches!(outcome, ReadOutcome::SourceVerified(_))
            && (receipt.source_verified_at.is_none()
                || expected_payload_bytes != receipt.requested_bytes))
        || !matches!(entry.phase, Phase::Queued | Phase::Reading)
    {
        entry.phase = Phase::Unknown(ReadUnknownReason::CompletionIdentityMismatch);
        entry.receipt.payload_bytes = 0;
        entry.receipt.completed_at = Some(Instant::now());
        state.accepting = false;
        drop(state);
        shared.notify();
        return false;
    }
    receipt.cancellation_requested = entry.receipt.cancellation_requested;
    if receipt.cancellation_requested && matches!(outcome, ReadOutcome::SourceVerified(_)) {
        // Assignment destroys the unpublished buffer before terminal publication.
        outcome = ReadOutcome::Cancelled(ReadCancellation::AfterStart);
        receipt.payload_bytes = 0;
    }
    receipt.completed_at = Some(Instant::now());
    entry.receipt = receipt;
    entry.phase = Phase::Complete(outcome);
    drop(state);
    shared.notify();
    true
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TestPhase {
    Dequeued,
    SessionOpened,
    IoStarted,
    PayloadRead,
    BeforePublish,
    AfterPublish,
}

#[cfg(test)]
type TestHook = Arc<dyn Fn(TestPhase, &mut ReadTicket) -> bool + Send + Sync>;

#[cfg(test)]
fn call_hook(hook: Option<&TestHook>, phase: TestPhase, ticket: &mut ReadTicket) -> bool {
    hook.is_none_or(|hook| hook(phase, ticket))
}

#[cfg(test)]
#[path = "../../tests/checkpoint_read_pipeline/mod.rs"]
mod tests;
