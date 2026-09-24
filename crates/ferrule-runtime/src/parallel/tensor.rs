//! Rank-aware Compute -> host collective -> Apply orchestration.
//!
//! One executor owns one TP replica, one existing DP worker pool, one existing
//! host collective and one existing distributed coordinator. Execution is
//! synchronous and serial; there is no second commit/publication protocol.
//! Workers and resident weights are constructed on their owner threads. Only
//! CPU-owned payloads may cross this boundary. Compute and Apply must fence all
//! device work (including on error) before returning. A failed fence must use
//! `panic_any(data::PanicQuiescence::Unknown)`, not an ordinary Err; see the
//! ReplicaWorker resource-retention contract. Apply is provisional, not
//! an externally visible commit. Only a successful `execute` publishes results.
//!
//! Successfully begun outer IDs must strictly increase, including after failure
//! or cancellation. Quiescent transactions are drained and retired before return.
//! Unknown quiescence instead preserves unresolved operation credits/metadata,
//! closes and joins the pool, and rejects all later execution. Joining a host
//! thread is NOT proof that its GPU stopped; the quarantined worker stays leaked.
//! Private, strictly increasing DP admission IDs identify individual commands;
//! `TensorWork::transaction` is the end-to-end identity, NOT WorkRequest's ID.
//! Neither identity space is reset by retirement. Reconfiguration/new lifetimes
//! require caller-managed epoch/ID uniqueness. Cancellation cannot preempt an
//! uncooperative worker. Shutdown and Drop reuse DP's join-all-owner contract.

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ferrule_common::{
    ParallelExecutionScopes, ParallelGroupId, ParallelRankId, TensorCollectiveParticipants,
    ValidatedParallelTopology,
};
use ferrule_model::transformer::parallel::{TensorParallelCollective, TensorParallelStagePlan};

use super::collective::{
    HostCollectiveDescriptor, HostCollectiveError, HostCollectiveGroup, HostCollectiveKind,
    HostCollectiveLimits, HostCollectivePoll,
};
use super::data::{
    BuildError, CompletionOutcome, DataParallelConfig, DataParallelExecutor, ReplicaWorker,
    ShutdownError, SubmitErrorKind,
};
use crate::distributed::DistributedTransactionLimits;
use crate::{
    Decision, DistributedTransaction, DistributedTransactionError, ExecutionTransactionId,
    FinalizeOutcome, SessionId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorRank {
    /// Pool-local rank and the index passed to the stage plan: 0..TP.
    pub local: ParallelRankId,
    /// Topology/collective rank: ((replica * PP) + stage) * TP + local.
    /// Neither this collective role nor `local` grants KV ownership.
    pub global: ParallelRankId,
}

#[derive(Debug)]
pub enum TensorCommand {
    /// Full row-major input. The worker uses its plan/local rank to shard it.
    Compute { input: Arc<[f32]>, rows: usize },
    /// Full row-major collective result, with Column padding already removed.
    /// Return a full [rows, out_features] result after all uploads/work finish.
    Apply { values: Vec<f32>, rows: usize },
}

/// Implement `ReplicaWorker<TensorWork, Output = Vec<f32>>`. The factory may
/// capture resident weights and a plan; workers need not implement Send/Sync.
#[derive(Debug)]
pub struct TensorWork {
    pub transaction: ExecutionTransactionId,
    pub rank: TensorRank,
    pub command: TensorCommand,
}

#[derive(Debug)]
pub enum TensorParallelError<E> {
    Configuration(&'static str),
    Pool(BuildError<E>),
    Shape(String),
    IdentityExhausted,
    Transaction(DistributedTransactionError),
    Collective(HostCollectiveError),
    Route {
        rank: TensorRank,
        kind: SubmitErrorKind,
    },
    Worker {
        rank: TensorRank,
        error: E,
    },
    WorkerPanicked(TensorRank),
    /// The worker panic hook could not prove device/DMA quiescence. The outer
    /// transaction remains Preparing; unproven operations keep their credits.
    QuiescenceUnknown(TensorRank),
    WorkerUnavailable(TensorRank),
    /// This executor permanently retains unresolved operation custody and rejects
    /// new work. Constructing a replacement does not reclaim leaked resources or
    /// prove device safety; external recovery must establish a safe new lifetime.
    Quarantined,
    Cancelled,
}

impl<E> From<DistributedTransactionError> for TensorParallelError<E> {
    fn from(error: DistributedTransactionError) -> Self {
        Self::Transaction(error)
    }
}

impl<E> From<HostCollectiveError> for TensorParallelError<E> {
    fn from(error: HostCollectiveError) -> Self {
        Self::Collective(error)
    }
}

/// Successful, published outputs in TP-local rank order. All ranks are returned;
/// the executor neither assumes identical Apply results nor picks a leader.
#[derive(Debug)]
pub struct TensorParallelOutput {
    pub transaction: ExecutionTransactionId,
    pub ranks: Vec<(TensorRank, Vec<f32>)>,
}

type WorkerPool<E> = DataParallelExecutor<TensorWork, Vec<f32>, E>;

pub struct TensorParallelExecutor<T>
where
    T: ReplicaWorker<TensorWork, Output = Vec<f32>> + 'static,
    T::Error: Send + 'static,
{
    pool: WorkerPool<T::Error>,
    coordinator: DistributedTransaction,
    scopes: ParallelExecutionScopes,
    participants: TensorCollectiveParticipants,
    ranks: Vec<TensorRank>,
    plan: TensorParallelStagePlan,
    group_id: ParallelGroupId,
    group: HostCollectiveGroup,
    next_worker_id: u64,
    quarantined: bool,
    // Preserve automatic join failures for the caller's explicit shutdown.
    shutdown_failure: Option<ShutdownError<T::Error>>,
    worker: PhantomData<fn() -> T>,
}

impl<T> TensorParallelExecutor<T>
where
    T: ReplicaWorker<TensorWork, Output = Vec<f32>> + 'static,
    T::Error: Send + 'static,
{
    /// `config.replicas` must equal TP, not the topology's DP or world size.
    /// The group is created with the selected replica's GLOBAL members, ordered
    /// by LOCAL rank. The outer coordinator retains at most one transaction/TP
    /// operations; host-buffer budgets are independent of transaction credits.
    pub fn new<F>(
        topology: ValidatedParallelTopology,
        replica: u32,
        plan: impl Into<TensorParallelStagePlan>,
        config: DataParallelConfig,
        worker_factory: F,
        group_id: ParallelGroupId,
        limits: HostCollectiveLimits,
    ) -> Result<Self, TensorParallelError<T::Error>>
    where
        F: FnOnce(TensorRank) -> Result<T, T::Error> + Clone + Send + 'static,
    {
        Self::new_at_stage(
            topology,
            replica,
            0,
            plan,
            config,
            worker_factory,
            group_id,
            limits,
        )
    }

    /// Run a stage-local collective using the existing CPU-owned orchestration.
    /// This does not run a PP pipeline or acquire KV custody. Its transaction
    /// tracks only collective work, not the replica's full KV commit cohort.
    pub fn new_at_stage<F>(
        topology: ValidatedParallelTopology,
        replica: u32,
        stage: u32,
        plan: impl Into<TensorParallelStagePlan>,
        config: DataParallelConfig,
        worker_factory: F,
        group_id: ParallelGroupId,
        limits: HostCollectiveLimits,
    ) -> Result<Self, TensorParallelError<T::Error>>
    where
        F: FnOnce(TensorRank) -> Result<T, T::Error> + Clone + Send + 'static,
    {
        let plan = plan.into();
        let scopes = topology
            .execution_scopes(replica)
            .map_err(|_| TensorParallelError::Configuration("replica outside topology"))?;
        let participants = scopes
            .tensor_collective_participants(stage, replica)
            .map_err(|_| TensorParallelError::Configuration("stage outside topology"))?
            .clone();
        let degree = participants.len();
        if plan.ranks() != degree || config.replicas != degree {
            return Err(TensorParallelError::Configuration(
                "plan/pool/topology TP mismatch",
            ));
        }
        if config.session_capacity < degree {
            return Err(TensorParallelError::Configuration(
                "session capacity must cover TP",
            ));
        }
        let ranks: Vec<_> = participants
            .iter()
            .enumerate()
            .map(|(local, global)| TensorRank {
                local: ParallelRankId::new(local as u32),
                global,
            })
            .collect();
        let group = HostCollectiveGroup::new(
            topology.topology_id(),
            group_id,
            participants.iter().collect(),
            limits,
        )?;
        let factory_ranks = ranks.clone();
        let pool = DataParallelExecutor::new(config, move |local: ParallelRankId| {
            worker_factory(factory_ranks[local.get() as usize])
        })
        .map_err(TensorParallelError::Pool)?;
        Ok(Self {
            pool,
            coordinator: DistributedTransaction::new_with_limits(
                topology,
                degree,
                DistributedTransactionLimits {
                    max_transactions: 1,
                    max_operations: degree,
                },
            ),
            scopes,
            participants,
            ranks,
            plan,
            group_id,
            group,
            next_worker_id: 1,
            quarantined: false,
            shutdown_failure: None,
            worker: PhantomData,
        })
    }

    pub fn execution_scopes(&self) -> &ParallelExecutionScopes {
        &self.scopes
    }
    pub fn collective_participants(&self) -> &TensorCollectiveParticipants {
        &self.participants
    }
    pub fn coordinator(&self) -> &DistributedTransaction {
        &self.coordinator
    }
    pub fn collective(&self) -> &HostCollectiveGroup {
        &self.group
    }
    pub fn outstanding(&self) -> usize {
        self.pool.outstanding()
    }

    /// Host completion delivery is separate from unresolved device custody.
    pub fn is_quarantined(&self) -> bool {
        self.quarantined
    }

    /// Weights are resident worker state, initialized by `worker_factory`.
    /// Reserves sessions [session_base, session_base + TP) for this call only;
    /// routes are explicitly checked and released after draining, on every path.
    pub fn execute(
        &mut self,
        id: ExecutionTransactionId,
        session_base: SessionId,
        input: Arc<[f32]>,
        rows: usize,
    ) -> Result<TensorParallelOutput, TensorParallelError<T::Error>> {
        self.execute_cancellable(id, session_base, input, rows, &AtomicBool::new(false))
    }

    /// External callers may set this flag while execution blocks. Once observed,
    /// cancellation is latched, propagated to accepted DP commands, and all their
    /// notifications are consumed before cancelling/retiring a quiescent outer
    /// transaction. Unknown quiescence takes precedence over cancellation.
    /// A request racing with publication may lose to successful completion.
    pub fn execute_cancellable(
        &mut self,
        id: ExecutionTransactionId,
        session_base: SessionId,
        input: Arc<[f32]>,
        rows: usize,
        cancellation: &AtomicBool,
    ) -> Result<TensorParallelOutput, TensorParallelError<T::Error>> {
        if self.quarantined {
            return Err(TensorParallelError::Quarantined);
        }
        let output_len = elements(rows, self.plan.out_features())
            .ok_or_else(|| TensorParallelError::Shape("invalid output shape".into()))?;
        if elements(rows, self.plan.in_features()) != Some(input.len()) {
            return Err(TensorParallelError::Shape("invalid input shape".into()));
        }
        session_base
            .0
            .checked_add(self.ranks.len() as u64 - 1)
            .ok_or(TensorParallelError::IdentityExhausted)?;
        // Reserve enough identity space for both phases before beginning.
        self.next_worker_id
            .checked_add(2 * self.ranks.len() as u64)
            .ok_or(TensorParallelError::IdentityExhausted)?;
        let kind = match self.plan.collective() {
            TensorParallelCollective::AllGather => HostCollectiveKind::AllGatherF32,
            TensorParallelCollective::Sum => HostCollectiveKind::AllReduceSumF32,
        };
        let count = self.plan.collective_count(rows).map_err(shape)?;
        self.group.memory_for(kind, count)?;
        for &rank in &self.ranks {
            self.pool
                .route_session_to(session(session_base, rank), rank.local)
                .map_err(|error| TensorParallelError::Route {
                    rank,
                    kind: error.kind,
                })?;
        }
        self.coordinator
            .begin_scoped(id, self.participants.as_participants().clone())?;
        let descriptor = HostCollectiveDescriptor {
            epoch: self.coordinator.topology().topology_id(),
            group: self.group_id,
            transaction: id,
            sequence: id.get(),
            kind,
            count,
        };
        // All fallible work after begin is enclosed so cleanup cannot be skipped.
        let mut operations = Vec::new();
        let mut unproven = vec![false; self.ranks.len()];
        let work = (|| {
            for &rank in &self.ranks {
                self.coordinator.prepare(id, rank.global)?;
            }
            operations = self.coordinator.communicate_all(id)?;
            let commands = self
                .ranks
                .iter()
                .map(|_| TensorCommand::Compute {
                    input: Arc::clone(&input),
                    rows,
                })
                .collect();
            let plan = &self.plan;
            let group = &mut self.group;
            run_phase(
                &mut self.pool,
                &mut self.next_worker_id,
                &self.ranks,
                id,
                session_base,
                commands,
                cancellation,
                &mut unproven,
                |rank, partial| {
                    let payload = plan
                        .pack_local_output(rank.local, &partial, rows)
                        .map_err(shape)?;
                    group
                        .submit(rank.global, descriptor, payload)
                        .map_err(|error| TensorParallelError::Collective(error.error))?;
                    Ok(Vec::new())
                },
            )?;
            let mut commands = Vec::with_capacity(self.ranks.len());
            for rank in &self.ranks {
                let HostCollectivePoll::Ready(result) = self.group.take_result(rank.global)? else {
                    return Err(TensorParallelError::Collective(
                        HostCollectiveError::NoOperation,
                    ));
                };
                if result.descriptor != descriptor {
                    return Err(TensorParallelError::Collective(
                        HostCollectiveError::DescriptorMismatch,
                    ));
                }
                let values = self
                    .plan
                    .unpack_collective_output(&result.values, rows)
                    .map_err(shape)?;
                commands.push(TensorCommand::Apply { values, rows });
            }
            // Host Ready is not completion: all outer credits remain held until
            // every accepted Apply has returned its synchronized host evidence.
            run_phase(
                &mut self.pool,
                &mut self.next_worker_id,
                &self.ranks,
                id,
                session_base,
                commands,
                cancellation,
                &mut unproven,
                |_, values| {
                    if values.len() != output_len {
                        return Err(TensorParallelError::Shape(
                            "invalid Apply output length".into(),
                        ));
                    }
                    Ok(values)
                },
            )
        })();
        let cancelled = matches!(&work, Err(TensorParallelError::Cancelled))
            || cancellation.load(Ordering::Acquire);
        let outcome = if work.is_ok() && !cancelled {
            FinalizeOutcome::Success
        } else {
            FinalizeOutcome::Failure
        };
        // run_phase never returns with accepted commands outstanding, even when
        // submission, packing, descriptors, workers, or cancellation fail.
        debug_assert_eq!(self.pool.outstanding(), 0);
        if let Some(active) = self.group.active_descriptor() {
            self.group
                .abort(active)
                .expect("exclusive collective identity");
        }
        for rank in &self.ranks {
            let session = session(session_base, *rank);
            if self.pool.session_rank(session).is_some() {
                self.pool
                    .release_session(session)
                    .expect("all accepted completions consumed");
            }
        }
        // Queue notifications are not GPU fences. Only proven ranks may return
        // communication credits; unknown operations remain owned by the existing
        // coordinator. Keep all transaction metadata and make no decision.
        for operation in operations {
            let local = self
                .ranks
                .iter()
                .position(|rank| rank.global == operation.rank)
                .expect("scoped operation rank");
            if !unproven[local] {
                self.coordinator
                    .complete(operation, outcome)
                    .expect("owned unfinished operation");
            }
        }
        self.coordinator.drain();
        if let Some(local) = unproven.iter().position(|&unknown| unknown) {
            self.quarantined = true;
            self.shutdown_failure = self.pool.shutdown().err();
            return Err(TensorParallelError::QuiescenceUnknown(self.ranks[local]));
        }
        for rank in &self.ranks {
            self.coordinator
                .prepare_vote(id, rank.global, outcome)
                .expect("one ready vote after completed work");
        }
        let decision = self
            .coordinator
            .commit_decision(id)
            .expect("all ready votes and communication drained");
        debug_assert_eq!(
            decision,
            if outcome == FinalizeOutcome::Success {
                Decision::Commit
            } else {
                Decision::Abort
            }
        );
        // Linear/SwiGLU stages have no mutable KV to install or roll back. Every accepted
        // command is quiescent and collective/session cleanup above is complete,
        // even when the business result failed or cancellation was requested.
        for rank in &self.ranks {
            self.coordinator
                .finalize(id, rank.global, FinalizeOutcome::Success)
                .expect("post-decision cleanup ACK");
        }
        if cancelled {
            self.coordinator.cancel(id).expect("Abort cleanup complete");
        } else {
            self.coordinator
                .publish(id)
                .expect("all cleanup ACKs present");
        }
        self.coordinator.retire(id).expect("terminal and drained");
        if cancelled {
            return Err(TensorParallelError::Cancelled);
        }
        let outputs = work?;
        Ok(TensorParallelOutput {
            transaction: id,
            ranks: self.ranks.iter().copied().zip(outputs).collect(),
        })
    }

    /// Join every owner, including after one owner's failure. Reports automatic
    /// quarantine-join failures once. Never releases unproven operation custody,
    /// and repeated shutdown does not make a quarantined executor reusable.
    /// Drop also joins, via the owned DP executor; leaked workers stay leaked.
    pub fn shutdown(&mut self) -> Result<(), ShutdownError<T::Error>> {
        let result = self.pool.shutdown();
        if let Some(mut previous) = self.shutdown_failure.take() {
            if let Err(error) = result {
                previous.failures.extend(error.failures);
            }
            Err(previous)
        } else {
            result
        }
    }
}

fn session(base: SessionId, rank: TensorRank) -> SessionId {
    SessionId(base.0 + u64::from(rank.local.get()))
}

fn shape<E>(error: ferrule_common::Error) -> TensorParallelError<E> {
    TensorParallelError::Shape(error.to_string())
}

fn elements(rows: usize, width: usize) -> Option<usize> {
    if rows == 0 || width == 0 {
        return None;
    }
    rows.checked_mul(width).filter(|count| {
        count
            .checked_mul(std::mem::size_of::<f32>())
            .is_some_and(|bytes| bytes <= isize::MAX as usize)
    })
}

/// Bounded per-phase delivery bookkeeping only. Lifecycle, queues, cancellation
/// and panic/owner-loss completion synthesis all belong to DataParallelExecutor.
/// Preserve error evidence and consume every accepted notification on all paths.
/// Unproven rank custody is recorded independently so a prior error/cancellation
/// cannot hide it and accidentally release its outer operation.
fn run_phase<E: Send + 'static>(
    pool: &mut WorkerPool<E>,
    next_id: &mut u64,
    ranks: &[TensorRank],
    transaction: ExecutionTransactionId,
    session_base: SessionId,
    commands: Vec<TensorCommand>,
    cancellation: &AtomicBool,
    unproven: &mut [bool],
    mut consume: impl FnMut(TensorRank, Vec<f32>) -> Result<Vec<f32>, TensorParallelError<E>>,
) -> Result<Vec<Vec<f32>>, TensorParallelError<E>> {
    let mut pending = vec![None; ranks.len()];
    let mut outputs: Vec<_> = (0..ranks.len()).map(|_| Vec::new()).collect();
    let mut error = None;
    for (&rank, command) in ranks.iter().zip(commands) {
        if cancellation.load(Ordering::Acquire) {
            error = Some(TensorParallelError::Cancelled);
            break;
        }
        let admission = ExecutionTransactionId::new(*next_id).expect("preflight ID space");
        *next_id += 1;
        match pool.try_submit_to(
            session(session_base, rank),
            rank.local,
            admission,
            TensorWork {
                transaction,
                rank,
                command,
            },
        ) {
            Ok(_) => pending[rank.local.get() as usize] = Some(admission),
            Err(rejected) => {
                error = Some(TensorParallelError::Route {
                    rank,
                    kind: rejected.kind,
                });
                break;
            }
        }
    }
    while pool.outstanding() != 0 {
        if cancellation.load(Ordering::Acquire) {
            error = Some(TensorParallelError::Cancelled);
        }
        if error.is_some() {
            for &id in pending.iter().flatten() {
                pool.cancel(id).expect("accepted unconsumed command");
            }
        }
        let Some(completion) = pool.poll() else {
            // Synchronous baseline: bounded sleep avoids a hot host spin loop.
            std::thread::sleep(Duration::from_micros(50));
            continue;
        };
        let index = completion.rank.get() as usize;
        let rank = ranks[index];
        assert_eq!(pending[index].take(), Some(completion.transaction));
        assert_eq!(completion.session, session(session_base, rank));
        let result = match completion.outcome {
            CompletionOutcome::Success(values) if error.is_none() => consume(rank, values),
            CompletionOutcome::Success(_) => Ok(Vec::new()),
            CompletionOutcome::Failed(error) => Err(TensorParallelError::Worker { rank, error }),
            CompletionOutcome::Cancelled => Err(TensorParallelError::Cancelled),
            CompletionOutcome::Panicked => Err(TensorParallelError::WorkerPanicked(rank)),
            CompletionOutcome::QuiescenceUnknown => {
                unproven[index] = true;
                Err(TensorParallelError::QuiescenceUnknown(rank))
            }
            CompletionOutcome::ReplicaUnavailable => {
                Err(TensorParallelError::WorkerUnavailable(rank))
            }
        };
        match result {
            Ok(values) => outputs[index] = values,
            Err(failure) if error.is_none() => error = Some(failure),
            Err(_) => {}
        }
    }
    match error {
        Some(error) => Err(error),
        None => Ok(outputs),
    }
}

/// Bounded host collective endpoints and shared abort/wake control.
#[path = "tensor_collective.rs"]
pub mod decoder_collective;
