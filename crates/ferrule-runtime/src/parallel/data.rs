//! Bounded host-side data-parallel routing, not a transaction protocol.
//!
//! Each rank owns one long-lived worker, constructed and dropped on its thread.
//! A worker whose device quiescence is unknown is quarantined and leaked instead.
//! Only host requests/results and cancellation flags cross threads: callers must
//! not put device handles or shared CUDA state in these payloads. There is no
//! collective, retry, session migration, or Prepare/Commit state machine here.
//!
//! Accepted transaction IDs must be strictly increasing across the whole pool.
//! A constant-space high-watermark rejects both active duplicates and retired
//! IDs, including after `release_session`. Rejected submissions do not advance
//! it. Out-of-order fresh IDs are also rejected; allocate IDs at admission, and
//! do not reuse IDs across pool lifetimes in the enclosing transaction system.
//!
//! The coordinator resolves the sticky rank, begins its existing
//! [`crate::DistributedTransaction`] with a singleton
//! [`ferrule_common::ParticipantSet`] via `begin_scoped`, then submits the same ID.
//! Rejected admission must be handled explicitly by that coordinator.
//! Completions are host evidence, NOT a commit/publication decision; routing
//! retains only bounded admission/delivery bookkeeping and cancellation intent.
//!
//! Panic isolation requires `panic=unwind`, which the workspace release profile
//! sets for this replica worker boundary. Shutdown/Drop join every owner, but
//! cannot preempt a factory, worker, destructor, or device operation that does
//! not return/cooperate.

use std::collections::{BTreeMap, HashMap};
use std::mem::{self, ManuallyDrop};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::thread::{self, JoinHandle};

use ferrule_common::ParallelRankId;
use ferrule_common::execution::ExecutionTransactionId;

use crate::SessionId;

#[derive(Debug, Clone, Copy)]
pub struct DataParallelConfig {
    pub replicas: usize,
    /// Queued + executing + completed but not consumed, per rank.
    pub max_outstanding_per_replica: usize,
    /// Sticky routes are never evicted automatically.
    pub session_capacity: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    ZeroReplicas,
    ZeroOutstanding,
    ZeroSessionCapacity,
    TooManyReplicas,
}

impl DataParallelConfig {
    pub fn validate(self) -> Result<(), ConfigError> {
        if self.replicas == 0 {
            return Err(ConfigError::ZeroReplicas);
        }
        if self.max_outstanding_per_replica == 0 {
            return Err(ConfigError::ZeroOutstanding);
        }
        if self.session_capacity == 0 {
            return Err(ConfigError::ZeroSessionCapacity);
        }
        if u32::try_from(self.replicas - 1).is_err() {
            return Err(ConfigError::TooManyReplicas);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Cancellation(Arc<AtomicBool>);

impl Cancellation {
    pub fn is_requested(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
pub struct WorkRequest<I> {
    pub rank: ParallelRankId,
    pub session: SessionId,
    pub transaction: ExecutionTransactionId,
    pub input: I,
    pub cancellation: Cancellation,
}

/// Evidence about all asynchronous work owned by this worker, not merely host
/// thread exit. Unknown must never be converted into a GPU completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanicQuiescence {
    Quiescent,
    Unknown,
}

/// No Send bound: factory invocation, execution, quiescence and destruction are
/// all rank-local. Ordinary Ok/Err returns MUST prove all asynchronous accesses
/// to request/worker resources finished. Failure to fence is NOT an ordinary Err:
/// retain affected resources in the worker and use
/// `std::panic::panic_any(PanicQuiescence::Unknown)` to force quarantine. This
/// typed fatal signal cannot be downgraded even by a successful panic hook.
///
/// Panic unwinds execute's stack before the hook runs. Workers must therefore
/// retain device/DMA-referenced resources in worker-owned storage or protect
/// temporaries with their own fence/leak guards; this hook cannot undo an unsafe
/// drop during unwinding. Factories have the same cleanup responsibility before
/// returning a worker. This boundary cannot make an arbitrary GPU worker safe.
pub trait ReplicaWorker<I> {
    type Output;
    type Error;

    fn execute(&mut self, request: WorkRequest<I>) -> Result<Self::Output, Self::Error>;

    /// Called on the owner after execute (or shutdown) panics; itself caught.
    /// Only return Quiescent after proving ALL device/DMA accesses have stopped.
    /// Pure CPU workers with no asynchronous users may explicitly return it.
    /// Unknown or a hook panic leaks the worker without shutdown/Drop. Never
    /// delegate to default shutdown: its no-op is not a quiescence guarantee.
    fn panic_quiescence(&mut self) -> PanicQuiescence {
        PanicQuiescence::Unknown
    }

    /// Ok and Err must both leave resources safe to drop, just like execute.
    fn shutdown(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[derive(Debug)]
pub enum CompletionOutcome<O, E> {
    Success(O),
    Failed(E),
    /// Cancellation observed before entering the worker.
    Cancelled,
    /// Execute panicked, but the owner hook proved quiescence.
    Panicked,
    /// No proof that asynchronous accesses stopped. Consuming this notification
    /// releases ONLY host queue bookkeeping, never device/transaction custody.
    /// The owner is disabled and its worker intentionally leaked.
    QuiescenceUnknown,
    /// Owner exited without executing this request, or after proving quiescence
    /// but before delivering its result. Never replayed.
    ReplicaUnavailable,
}

#[derive(Debug)]
pub struct HostCompletion<O, E> {
    pub rank: ParallelRankId,
    pub session: SessionId,
    pub transaction: ExecutionTransactionId,
    /// Intent at consumption time; it does not rewrite a successful result.
    pub cancellation_requested: bool,
    pub outcome: CompletionOutcome<O, E>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitErrorKind {
    InvalidRank,
    StickyMismatch {
        bound: ParallelRankId,
    },
    Closed,
    DuplicateTransaction,
    RetiredOrOutOfOrder {
        high_watermark: ExecutionTransactionId,
    },
    SessionCapacity,
    Backpressure,
    ReplicaUnavailable,
}

/// Rejected payloads are returned to the caller; there is no hidden retry.
#[derive(Debug)]
pub struct SubmitError<I> {
    pub rank: Option<ParallelRankId>,
    pub session: SessionId,
    pub transaction: ExecutionTransactionId,
    pub kind: SubmitErrorKind,
    pub input: I,
}

/// Routing can fail with Closed, SessionCapacity, or ReplicaUnavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteError {
    pub rank: Option<ParallelRankId>,
    pub session: SessionId,
    pub kind: SubmitErrorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CancelError {
    pub transaction: ExecutionTransactionId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseSessionError {
    Unknown {
        session: SessionId,
    },
    Busy {
        session: SessionId,
        rank: ParallelRankId,
        transaction: ExecutionTransactionId,
    },
}

#[derive(Debug)]
pub enum OwnerFailureKind<E> {
    Spawn(std::io::Error),
    Initialization(E),
    Panicked,
    /// Worker was leaked because its panic hook could not prove quiescence.
    QuiescenceUnknown,
    Shutdown(E),
}

#[derive(Debug)]
pub struct OwnerFailure<E> {
    pub rank: ParallelRankId,
    /// Present for execute panics; initialization/shutdown have no transaction.
    pub transaction: Option<ExecutionTransactionId>,
    pub kind: OwnerFailureKind<E>,
}

#[derive(Debug)]
pub enum BuildError<E> {
    Config(ConfigError),
    /// Includes failures encountered while joining already-created owners.
    Owners(Vec<OwnerFailure<E>>),
}

#[derive(Debug)]
pub struct ShutdownError<E> {
    pub failures: Vec<OwnerFailure<E>>,
}

struct Pending {
    replica: usize,
    session: SessionId,
    cancellation: Cancellation,
}

type Delivery<O, E> = (ExecutionTransactionId, CompletionOutcome<O, E>);

struct Replica<I, O, E> {
    sender: Option<SyncSender<WorkRequest<I>>>,
    receiver: Receiver<Delivery<O, E>>,
    owner: Option<JoinHandle<Result<(), OwnerFailure<E>>>>,
    alive: Arc<AtomicBool>,
    // Nonzero until an entered command has proven quiescence. Also protects the
    // disconnected-channel fallback from inventing a safe failure on owner loss.
    in_flight: Arc<AtomicU64>,
    outstanding: usize,
}

/// Single host coordinator; mutable admission serializes routing and ID checks.
/// Memory is bounded by session_capacity + replicas * outstanding capacity
/// (payload sizes themselves remain the caller's responsibility).
pub struct DataParallelExecutor<I: Send + 'static, O: Send + 'static, E: Send + 'static> {
    config: DataParallelConfig,
    replicas: Vec<Replica<I, O, E>>,
    sessions: HashMap<SessionId, usize>,
    pending: BTreeMap<ExecutionTransactionId, Pending>,
    high_watermark: Option<ExecutionTransactionId>,
    next_route: usize,
    next_poll: usize,
    closed: bool,
}

impl<I: Send + 'static, O: Send + 'static, E: Send + 'static> DataParallelExecutor<I, O, E> {
    /// The Send factory is cloned on the host, but invoked INSIDE each thread.
    /// Neither W nor its device-local state needs to implement Send or Sync.
    /// Returns only after every created owner acknowledges initialization.
    pub fn new<F, W>(config: DataParallelConfig, factory: F) -> Result<Self, BuildError<E>>
    where
        F: FnOnce(ParallelRankId) -> Result<W, E> + Clone + Send + 'static,
        W: ReplicaWorker<I, Output = O, Error = E> + 'static,
    {
        config.validate().map_err(BuildError::Config)?;
        let mut pool = Self {
            config,
            replicas: Vec::new(),
            sessions: HashMap::new(),
            pending: BTreeMap::new(),
            high_watermark: None,
            next_route: 0,
            next_poll: 0,
            closed: false,
        };
        for index in 0..config.replicas {
            let rank = rank_id(index);
            let (sender, requests) = mpsc::sync_channel(config.max_outstanding_per_replica);
            let (results, receiver) = mpsc::sync_channel(config.max_outstanding_per_replica);
            let (ready_tx, ready_rx) = mpsc::sync_channel(1);
            let alive = Arc::new(AtomicBool::new(true));
            let owner_alive = Arc::clone(&alive);
            let in_flight = Arc::new(AtomicU64::new(0));
            let owner_in_flight = Arc::clone(&in_flight);
            let make_worker = factory.clone();
            let owner = thread::Builder::new()
                .name(format!("dp-owner-{index}"))
                .spawn(move || {
                    let _exit = OwnerExit(Arc::clone(&owner_alive));
                    let initialized = catch_unwind(AssertUnwindSafe(|| make_worker(rank)));
                    let worker = match initialized {
                        Ok(Ok(worker)) => worker,
                        other => {
                            let kind = match other {
                                Ok(Err(error)) => OwnerFailureKind::Initialization(error),
                                Err(payload) => match panic_payload_quiescence(payload) {
                                    Some(PanicQuiescence::Unknown) => {
                                        OwnerFailureKind::QuiescenceUnknown
                                    }
                                    // No worker was constructed, so no hook can run.
                                    // An ordinary factory panic is a factory failure,
                                    // not proof of worker/device quiescence.
                                    Some(PanicQuiescence::Quiescent) | None => {
                                        OwnerFailureKind::Panicked
                                    }
                                },
                                Ok(Ok(_)) => unreachable!(),
                            };
                            let _ = ready_tx.send(Err(OwnerFailure {
                                rank,
                                transaction: None,
                                kind,
                            }));
                            return Ok(());
                        }
                    };
                    if ready_tx.send(Ok(())).is_err() {
                        return Ok(());
                    }
                    run_owner(
                        rank,
                        worker,
                        requests,
                        results,
                        &owner_alive,
                        &owner_in_flight,
                    )
                });
            let failure = match owner {
                Ok(owner) => {
                    pool.replicas.push(Replica {
                        sender: Some(sender),
                        receiver,
                        owner: Some(owner),
                        alive,
                        in_flight,
                        outstanding: 0,
                    });
                    match ready_rx.recv() {
                        Ok(Ok(())) => None,
                        Ok(Err(failure)) => Some(failure),
                        Err(_) => Some(OwnerFailure {
                            rank,
                            transaction: None,
                            kind: OwnerFailureKind::Panicked,
                        }),
                    }
                }
                Err(error) => Some(OwnerFailure {
                    rank,
                    transaction: None,
                    kind: OwnerFailureKind::Spawn(error),
                }),
            };
            if let Some(failure) = failure {
                let mut failures = vec![failure];
                if let Err(error) = pool.shutdown() {
                    failures.extend(error.failures);
                }
                return Err(BuildError::Owners(failures));
            }
        }
        Ok(pool)
    }

    pub fn session_rank(&self, session: SessionId) -> Option<ParallelRankId> {
        self.sessions.get(&session).copied().map(rank_id)
    }

    pub fn outstanding(&self) -> usize {
        self.pending.len()
    }

    /// Explicitly bind a session before beginning a scoped transaction. This
    /// reserves only a bounded routing slot, not an outstanding-work credit.
    /// Begin/reserve in the existing coordinator before try_submit; on rejection
    /// the coordinator must abort/release its own reservation. The route remains
    /// sticky until explicit release, including if its owner subsequently fails.
    pub fn route_session(&mut self, session: SessionId) -> Result<ParallelRankId, RouteError> {
        let sticky = self.sessions.get(&session).copied();
        let reject = |kind, index: Option<usize>| RouteError {
            rank: index.map(rank_id),
            session,
            kind,
        };
        if self.closed {
            return Err(reject(SubmitErrorKind::Closed, sticky));
        }
        if sticky.is_none() && self.sessions.len() >= self.config.session_capacity {
            return Err(reject(SubmitErrorKind::SessionCapacity, None));
        }
        let index = self
            .route_candidate(session)
            .ok_or_else(|| reject(SubmitErrorKind::ReplicaUnavailable, None))?;
        if !self.replicas[index].alive.load(Ordering::Acquire) {
            return Err(reject(SubmitErrorKind::ReplicaUnavailable, Some(index)));
        }
        self.sessions.insert(session, index);
        if sticky.is_none() {
            self.next_route = (index + 1) % self.replicas.len();
        }
        Ok(rank_id(index))
    }

    /// Validate an explicit pool-local rank without installing a binding.
    /// Unlike `route_session`, this is a preflight only: `try_submit_to` installs
    /// a sticky route only after successful admission. No least-loaded fallback.
    pub fn route_session_to(
        &self,
        session: SessionId,
        rank: ParallelRankId,
    ) -> Result<ParallelRankId, RouteError> {
        let reject = |kind| RouteError {
            rank: Some(rank),
            session,
            kind,
        };
        if self.closed {
            return Err(reject(SubmitErrorKind::Closed));
        }
        let index = rank.get() as usize;
        let Some(replica) = self.replicas.get(index) else {
            return Err(reject(SubmitErrorKind::InvalidRank));
        };
        if let Some(&bound) = self.sessions.get(&session) {
            if bound != index {
                return Err(reject(SubmitErrorKind::StickyMismatch {
                    bound: rank_id(bound),
                }));
            }
        } else if self.sessions.len() >= self.config.session_capacity {
            return Err(reject(SubmitErrorKind::SessionCapacity));
        }
        if !replica.alive.load(Ordering::Acquire) {
            return Err(reject(SubmitErrorKind::ReplicaUnavailable));
        }
        Ok(rank)
    }

    /// Nonblocking explicit admission, with the same ID/queue/session bounds as
    /// `try_submit`. Rejection never creates or changes a sticky binding.
    pub fn try_submit_to(
        &mut self,
        session: SessionId,
        rank: ParallelRankId,
        transaction: ExecutionTransactionId,
        input: I,
    ) -> Result<ParallelRankId, SubmitError<I>> {
        if let Err(error) = self.route_session_to(session, rank) {
            return Err(SubmitError {
                rank: error.rank,
                session,
                transaction,
                kind: error.kind,
                input,
            });
        }
        self.submit_on(session, transaction, input, Some(rank))
    }

    fn route_candidate(&self, session: SessionId) -> Option<usize> {
        self.sessions.get(&session).copied().or_else(|| {
            (0..self.replicas.len())
                .map(|offset| (self.next_route + offset) % self.replicas.len())
                .filter(|&index| self.replicas[index].alive.load(Ordering::Acquire))
                .min_by_key(|&index| self.replicas[index].outstanding)
        })
    }

    /// Nonblocking admission. Sticky sessions never fall back to another rank,
    /// even when their owner is full or dead. Fresh sessions use the least-loaded
    /// live rank, with rotating tie breaking. Routes are installed only on success.
    pub fn try_submit(
        &mut self,
        session: SessionId,
        transaction: ExecutionTransactionId,
        input: I,
    ) -> Result<ParallelRankId, SubmitError<I>> {
        self.submit_on(session, transaction, input, None)
    }

    fn submit_on(
        &mut self,
        session: SessionId,
        transaction: ExecutionTransactionId,
        input: I,
        explicit_rank: Option<ParallelRankId>,
    ) -> Result<ParallelRankId, SubmitError<I>> {
        let sticky = self.sessions.get(&session).copied();
        let reject = |kind, replica: Option<usize>, input| SubmitError {
            rank: replica.map(rank_id),
            session,
            transaction,
            kind,
            input,
        };
        if self.closed {
            return Err(reject(SubmitErrorKind::Closed, sticky, input));
        }
        if let Some(pending) = self.pending.get(&transaction) {
            return Err(reject(
                SubmitErrorKind::DuplicateTransaction,
                Some(pending.replica),
                input,
            ));
        }
        if let Some(high_watermark) = self.high_watermark {
            if transaction <= high_watermark {
                return Err(reject(
                    SubmitErrorKind::RetiredOrOutOfOrder { high_watermark },
                    sticky,
                    input,
                ));
            }
        }
        if sticky.is_none() && self.sessions.len() >= self.config.session_capacity {
            return Err(reject(SubmitErrorKind::SessionCapacity, None, input));
        }
        let index = explicit_rank
            .map(|rank| rank.get() as usize)
            .or_else(|| self.route_candidate(session));
        let Some(index) = index else {
            return Err(reject(SubmitErrorKind::ReplicaUnavailable, None, input));
        };
        let replica = &mut self.replicas[index];
        if !replica.alive.load(Ordering::Acquire) {
            return Err(reject(
                SubmitErrorKind::ReplicaUnavailable,
                Some(index),
                input,
            ));
        }
        if replica.outstanding >= self.config.max_outstanding_per_replica {
            return Err(reject(SubmitErrorKind::Backpressure, Some(index), input));
        }
        let cancellation = Cancellation(Arc::new(AtomicBool::new(false)));
        let request = WorkRequest {
            rank: rank_id(index),
            session,
            transaction,
            input,
            cancellation: cancellation.clone(),
        };
        match replica
            .sender
            .as_ref()
            .expect("open pool has senders")
            .try_send(request)
        {
            Ok(()) => {}
            Err(TrySendError::Full(request)) => {
                return Err(reject(
                    SubmitErrorKind::Backpressure,
                    Some(index),
                    request.input,
                ));
            }
            Err(TrySendError::Disconnected(request)) => {
                replica.alive.store(false, Ordering::Release);
                return Err(reject(
                    SubmitErrorKind::ReplicaUnavailable,
                    Some(index),
                    request.input,
                ));
            }
        }
        replica.outstanding += 1;
        self.pending.insert(
            transaction,
            Pending {
                replica: index,
                session,
                cancellation,
            },
        );
        self.sessions.insert(session, index);
        self.high_watermark = Some(transaction);
        if sticky.is_none() {
            self.next_route = (index + 1) % self.replicas.len();
        }
        Ok(rank_id(index))
    }

    /// Set intent without adding a message or releasing a credit. Queued work is
    /// skipped if intent is observed before worker entry. In-flight work must
    /// check its token at safe points and report its own outcome; no rollback is
    /// implied. Even a ready-but-unconsumed completion may receive intent.
    pub fn cancel(&mut self, transaction: ExecutionTransactionId) -> Result<(), CancelError> {
        let pending = self
            .pending
            .get(&transaction)
            .ok_or(CancelError { transaction })?;
        pending.cancellation.0.store(true, Ordering::Release);
        Ok(())
    }

    /// Release routing only after all completions for the session are consumed.
    /// First submit/complete any worker-specific KV cleanup as ordinary work.
    /// Reusing a released session ID explicitly starts a NEW routing lifetime;
    /// this never transfers retained model state or silently migrates a session.
    pub fn release_session(
        &mut self,
        session: SessionId,
    ) -> Result<ParallelRankId, ReleaseSessionError> {
        let index = *self
            .sessions
            .get(&session)
            .ok_or(ReleaseSessionError::Unknown { session })?;
        if let Some((&transaction, _)) = self
            .pending
            .iter()
            .find(|(_, pending)| pending.session == session)
        {
            return Err(ReleaseSessionError::Busy {
                session,
                rank: rank_id(index),
                transaction,
            });
        }
        self.sessions.remove(&session);
        Ok(rank_id(index))
    }

    /// Fair, nonblocking host completion consumption. This is the ONLY normal
    /// host queue credit-release point, NOT proof of device quiescence. Unknown
    /// notifications transfer unresolved custody to the caller. Owner loss
    /// synthesizes failures only after draining already-sent completions; an
    /// entered command without proof remains QuiescenceUnknown.
    pub fn try_recv(&mut self) -> Option<HostCompletion<O, E>> {
        for offset in 0..self.replicas.len() {
            let index = (self.next_poll + offset) % self.replicas.len();
            let delivery = match self.replicas[index].receiver.try_recv() {
                Ok(delivery) => {
                    if matches!(
                        &delivery.1,
                        CompletionOutcome::Panicked | CompletionOutcome::QuiescenceUnknown
                    ) {
                        self.replicas[index].alive.store(false, Ordering::Release);
                    }
                    Some(delivery)
                }
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => {
                    // Channel closure can precede owner-local destruction.
                    self.replicas[index].alive.store(false, Ordering::Release);
                    self.pending
                        .iter()
                        .find(|(_, pending)| pending.replica == index)
                        .map(|(&transaction, _)| {
                            let outcome = if self.replicas[index].in_flight.load(Ordering::Acquire)
                                == transaction.get()
                            {
                                CompletionOutcome::QuiescenceUnknown
                            } else {
                                CompletionOutcome::ReplicaUnavailable
                            };
                            (transaction, outcome)
                        })
                }
            };
            if let Some((transaction, outcome)) = delivery {
                let pending = self
                    .pending
                    .remove(&transaction)
                    .expect("one completion per accepted request");
                self.replicas[index].outstanding -= 1;
                self.next_poll = (index + 1) % self.replicas.len();
                return Some(HostCompletion {
                    rank: rank_id(index),
                    session: pending.session,
                    transaction,
                    cancellation_requested: pending.cancellation.is_requested(),
                    outcome,
                });
            }
        }
        None
    }

    pub fn poll(&mut self) -> Option<HostCompletion<O, E>> {
        self.try_recv()
    }

    /// Close inputs and signal cancellation without joining owners or retiring
    /// any ticket. Poll shutdown_ready, then shutdown, for a bounded host wait.
    pub fn begin_shutdown(&mut self) {
        self.closed = true;
        for pending in self.pending.values() {
            pending.cancellation.0.store(true, Ordering::Release);
        }
        for replica in &mut self.replicas {
            replica.sender.take();
        }
    }

    /// Nonblocking join preflight, not a device fence. Consume completions and
    /// inspect shutdown failures before treating external work as complete.
    pub fn shutdown_ready(&self) -> bool {
        self.closed
            && self
                .replicas
                .iter()
                .all(|r| r.owner.as_ref().is_none_or(JoinHandle::is_finished))
    }

    /// Close ALL inputs, request cancellation, then join ALL owners even if some
    /// fail. Completion channels hold at least the entire outstanding budget, so
    /// joining never requires concurrent host draining. Results/credits remain
    /// available through try_recv after shutdown. Repeated shutdown is a no-op;
    /// failures are reported once. Drop performs the same joins but discards errors.
    /// A joined quarantined owner is NOT a device fence: its worker stays leaked,
    /// and the caller must retain custody for QuiescenceUnknown notifications.
    pub fn shutdown(&mut self) -> Result<(), ShutdownError<E>> {
        self.begin_shutdown();
        let mut failures = Vec::new();
        for (index, replica) in self.replicas.iter_mut().enumerate() {
            if let Some(owner) = replica.owner.take() {
                match owner.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(failure)) => failures.push(failure),
                    Err(payload) => {
                        mem::forget(payload);
                        let active = replica.in_flight.load(Ordering::Acquire);
                        failures.push(OwnerFailure {
                            rank: rank_id(index),
                            transaction: ExecutionTransactionId::new(active).ok(),
                            kind: if active == 0 {
                                OwnerFailureKind::Panicked
                            } else {
                                OwnerFailureKind::QuiescenceUnknown
                            },
                        });
                    }
                }
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(ShutdownError { failures })
        }
    }
}

impl<I: Send + 'static, O: Send + 'static, E: Send + 'static> Drop
    for DataParallelExecutor<I, O, E>
{
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn rank_id(index: usize) -> ParallelRankId {
    ParallelRankId::new(u32::try_from(index).expect("validated replica count"))
}

struct OwnerExit(Arc<AtomicBool>);

impl Drop for OwnerExit {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

// Do not drop panic payloads here: arbitrary payload destructors can themselves
// panic and bypass quarantine/delivery. These allocations are panic-path only.
fn panic_payload_quiescence(payload: Box<dyn std::any::Any + Send>) -> Option<PanicQuiescence> {
    let quiescence = payload.downcast_ref::<PanicQuiescence>().copied();
    // Panic payload destructors are arbitrary application code. Never run them
    // on this failure boundary; factory/worker guards own their resources.
    mem::forget(payload);
    quiescence
}

fn quiesce_after_panic<I, W: ReplicaWorker<I>>(
    worker: &mut W,
    payload: Box<dyn std::any::Any + Send>,
) -> PanicQuiescence {
    let forced_unknown = panic_payload_quiescence(payload) == Some(PanicQuiescence::Unknown);
    let quiescence = match catch_unwind(AssertUnwindSafe(|| worker.panic_quiescence())) {
        Ok(quiescence) => quiescence,
        Err(payload) => {
            let _ = panic_payload_quiescence(payload);
            PanicQuiescence::Unknown
        }
    };
    if forced_unknown {
        PanicQuiescence::Unknown
    } else {
        quiescence
    }
}

fn run_owner<I, W: ReplicaWorker<I>>(
    rank: ParallelRankId,
    worker: W,
    requests: Receiver<WorkRequest<I>>,
    results: SyncSender<Delivery<W::Output, W::Error>>,
    alive: &AtomicBool,
    in_flight: &AtomicU64,
) -> Result<(), OwnerFailure<W::Error>> {
    // Keep the allocation as well as its fields alive on quarantine. Suppressing
    // Drop for a stack W alone would let thread exit reclaim its inline storage.
    // No reference/pointer into W may survive its factory-to-owner move.
    let mut worker = ManuallyDrop::new(Box::new(worker));
    for request in requests {
        let transaction = request.transaction;
        let outcome = if request.cancellation.is_requested() {
            CompletionOutcome::Cancelled
        } else {
            in_flight.store(transaction.get(), Ordering::Release);
            let result = catch_unwind(AssertUnwindSafe(|| worker.execute(request)));
            match result {
                Ok(result) => {
                    // Both ordinary return paths promise quiescence by contract.
                    in_flight.store(0, Ordering::Release);
                    match result {
                        Ok(output) => CompletionOutcome::Success(output),
                        Err(error) => CompletionOutcome::Failed(error),
                    }
                }
                Err(payload) => {
                    alive.store(false, Ordering::Release);
                    let quiescence = quiesce_after_panic::<I, W>(&mut **worker, payload);
                    let (outcome, kind) = match quiescence {
                        PanicQuiescence::Quiescent => {
                            in_flight.store(0, Ordering::Release);
                            (CompletionOutcome::Panicked, OwnerFailureKind::Panicked)
                        }
                        PanicQuiescence::Unknown => (
                            CompletionOutcome::QuiescenceUnknown,
                            OwnerFailureKind::QuiescenceUnknown,
                        ),
                    };
                    let _ = results.send((transaction, outcome));
                    if quiescence == PanicQuiescence::Quiescent {
                        drop(ManuallyDrop::into_inner(worker));
                    }
                    return Err(OwnerFailure {
                        rank,
                        transaction: Some(transaction),
                        kind,
                    });
                }
            }
        };
        if results.send((transaction, outcome)).is_err() {
            break;
        }
    }
    let result = match catch_unwind(AssertUnwindSafe(|| worker.shutdown())) {
        Ok(result) => result.map_err(|error| OwnerFailure {
            rank,
            transaction: None,
            kind: OwnerFailureKind::Shutdown(error),
        }),
        Err(payload) => {
            alive.store(false, Ordering::Release);
            if quiesce_after_panic::<I, W>(&mut **worker, payload) == PanicQuiescence::Unknown {
                return Err(OwnerFailure {
                    rank,
                    transaction: None,
                    kind: OwnerFailureKind::QuiescenceUnknown,
                });
            }
            Err(OwnerFailure {
                rank,
                transaction: None,
                kind: OwnerFailureKind::Panicked,
            })
        }
    };
    drop(ManuallyDrop::into_inner(worker));
    result
}
