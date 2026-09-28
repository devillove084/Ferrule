//! Host domain protocol shared by thread and process adapters.
//!
//! These are owned domain values, NOT serde for private KV handles. An IPC codec
//! must bound its frames and reconstruct/validate domain values before dispatch.
//! `PipelineCommandKey` is addressing metadata; the non-cloneable physical token
//! remains in `PipelineStageWorker`. A cancellation flag is call-local and must
//! be reconstructed in the child, never serialized as an address.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ferrule_common::execution::{
    ExecutionBatch, ExecutionTransactionId, KvPageId, KvReservationView,
};
use ferrule_common::{Error, Result};
use ferrule_model::decoder::{
    DecoderKvPageSnapshot, KvCommitBinding, KvCommitProjection, KvEndProgress,
    LogicalExecutionIdentity, PackedDecoderBatch,
};
use ferrule_model::transformer::{SegmentInput, SegmentOutput};

use super::{PipelineOwnerStats, PipelineRank, PipelineStageDescription, error};
use crate::SessionId;
use crate::parallel::data::{CompletionOutcome, DataParallelExecutor};

/// Address of one active physical operation, never authority to reconstruct it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineCommandKey {
    pub(super) binding: KvCommitBinding,
    pub(super) rank: PipelineRank,
    pub(super) session: SessionId,
}

impl PipelineCommandKey {
    pub fn new(binding: KvCommitBinding, rank: PipelineRank, session: SessionId) -> Result<Self> {
        if !binding
            .participants()
            .iter()
            .any(|member| member == rank.global)
        {
            return Err(error("pipeline key rank is not a KV participant"));
        }
        Ok(Self {
            binding,
            rank,
            session,
        })
    }

    pub fn binding(&self) -> &KvCommitBinding {
        &self.binding
    }
    pub fn rank(&self) -> PipelineRank {
        self.rank
    }
    pub fn session(&self) -> SessionId {
        self.session
    }
    pub fn into_parts(self) -> (KvCommitBinding, PipelineRank, SessionId) {
        (self.binding, self.rank, self.session)
    }
}

/// All stage operations are dispatched through the same owner journal. Prepare
/// lowers and reserves only; Execute enters that exact reservation once.
#[derive(Debug)]
pub enum PipelineCommand {
    Describe,
    Stats,
    Create {
        session: SessionId,
    },
    Fork {
        source: SessionId,
        target: SessionId,
    },
    Release {
        session: SessionId,
        pages: Vec<KvPageId>,
    },
    Prepare {
        key: PipelineCommandKey,
        batch: Box<ExecutionBatch>,
        reservation: KvReservationView,
    },
    Execute {
        key: PipelineCommandKey,
        input: SegmentInput,
        cancellation: Arc<AtomicBool>,
    },
    Ready(PipelineCommandKey),
    Install {
        key: PipelineCommandKey,
        generation: u64,
    },
    PollInstall {
        key: PipelineCommandKey,
        generation: u64,
    },
    Abort {
        key: PipelineCommandKey,
        generation: u64,
    },
    Rollback(PipelineCommandKey),
    Publish(PipelineCommandKey),
    CheckRetirement {
        key: PipelineCommandKey,
        pages: Vec<KvPageId>,
    },
    Retire {
        key: PipelineCommandKey,
        generation: u64,
        pages: Vec<KvPageId>,
    },
    Finish(PipelineCommandKey),
}

/// Portable preparation projection: no sequence topology, private KV token,
/// device handle, or backend transaction is encoded. An IPC codec may convert
/// these public host values to its bounded wire DTO. Decoded values are untrusted
/// until `into_commit_projection` checks the exact original parent request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelinePreparedProjection {
    pub batch: ExecutionBatch,
    pub reservation: KvReservationView,
    pub page_size: usize,
    pub page_statuses: Vec<DecoderKvPageSnapshot>,
}

impl PipelinePreparedProjection {
    /// Called by the physical owner after model lowering, with snapshots from
    /// that prepare boundary (not post-allocation page status).
    pub fn from_packed(
        batch: &PackedDecoderBatch,
        page_statuses: Vec<DecoderKvPageSnapshot>,
    ) -> Result<Self> {
        if batch.sequences().len() != 1 || batch.cow_replacements().len() > 1 {
            return Err(error("pipeline projection requires one standard sequence"));
        }
        let sequence = &batch.sequences()[0];
        Ok(Self {
            batch: batch.source_batch().clone(),
            reservation: KvReservationView {
                state_slot: sequence.page_state_slot(),
                execution_state_slot: batch.source_batch().sequences()[0].state_slot,
                positions: sequence.context_len()..sequence.sequence_len(),
                newly_allocated: batch.new_pages().to_vec(),
                generation: sequence.page_generation(),
                execution_generation: sequence.execution_generation(),
                cow_replacement: batch.cow_replacements().first().copied(),
            },
            page_size: batch.page_size(),
            page_statuses,
        })
    }

    /// Revalidate only logical commit metadata. No synthetic state or executable batch is made.
    pub fn into_commit_projection(
        self,
        expected_batch: &ExecutionBatch,
        expected_reservation: &KvReservationView,
        description: &PipelineStageDescription,
    ) -> Result<KvCommitProjection> {
        let statuses = self.validate_metadata(expected_batch, expected_reservation, description)?;
        let projection = KvCommitProjection::validate(
            &self.batch,
            std::slice::from_ref(&self.reservation),
            &description.execution_capabilities()?,
            self.page_size,
            &|page| {
                statuses
                    .get(&page)
                    .copied()
                    .unwrap_or(ferrule_model::decoder::DecoderKvPageStatus::Preempted)
            },
        )?;
        self.validate_protected(&projection, &statuses)?;
        Ok(projection)
    }

    /// Strong collective identity is derived from the original full request, never commit equality.
    pub fn into_execution_identity(
        self,
        expected_batch: &ExecutionBatch,
        expected_reservation: &KvReservationView,
        description: &PipelineStageDescription,
        sessions: &[u64],
    ) -> Result<LogicalExecutionIdentity> {
        let statuses = self.validate_metadata(expected_batch, expected_reservation, description)?;
        let identity = LogicalExecutionIdentity::validate(
            &self.batch,
            std::slice::from_ref(&self.reservation),
            &description.execution_capabilities()?,
            self.page_size,
            &|page| {
                statuses
                    .get(&page)
                    .copied()
                    .unwrap_or(ferrule_model::decoder::DecoderKvPageStatus::Preempted)
            },
            sessions,
        )?;
        self.validate_protected(&identity.commit_projection(), &statuses)?;
        Ok(identity)
    }

    fn validate_metadata(
        &self,
        expected_batch: &ExecutionBatch,
        expected_reservation: &KvReservationView,
        description: &PipelineStageDescription,
    ) -> Result<std::collections::BTreeMap<KvPageId, ferrule_model::decoder::DecoderKvPageStatus>>
    {
        description.validate()?;
        if &self.batch != expected_batch
            || &self.reservation != expected_reservation
            || self.page_size != description.config.page_size
            || self.batch.sequences().len() != 1
            || self.reservation.positions.end > description.config.max_positions
            || self.page_statuses.len() > description.config.max_pages
            || self
                .batch
                .token_ids()
                .iter()
                .any(|&token| token as usize >= description.vocabulary)
        {
            return Err(error(
                "pipeline preparation projection differs from parent request",
            ));
        }
        let statuses = self
            .page_statuses
            .iter()
            .map(|s| (s.page, s.status))
            .collect::<std::collections::BTreeMap<_, _>>();
        if statuses.len() != self.page_statuses.len() {
            return Err(error("duplicate pipeline projection page snapshot"));
        }
        Ok(statuses)
    }

    fn validate_protected(
        &self,
        projection: &KvCommitProjection,
        statuses: &std::collections::BTreeMap<
            KvPageId,
            ferrule_model::decoder::DecoderKvPageStatus,
        >,
    ) -> Result<()> {
        if projection
            .protected_pages()
            .iter()
            .copied()
            .ne(statuses.keys().copied())
        {
            return Err(error(
                "pipeline projection snapshots do not cover exact protected pages",
            ));
        }
        Ok(())
    }
}

/// No physical handle or device allocation appears in a reply.
#[derive(Debug)]
pub enum PipelineReply {
    Description(PipelineStageDescription),
    Stats(PipelineOwnerStats),
    Generation(u64),
    Prepared {
        projection: PipelinePreparedProjection,
    },
    Executed {
        output: SegmentOutput,
    },
    Ack(KvEndProgress),
}

/// Serial production call contract. There is no transaction/decision registry in
/// a transport. Never replay an accepted command on timeout or loss of an ACK.
/// `poll` must be called while waiting so the caller can mirror cancellation into
/// the Execute flag; cancellation still drains admitted work before rollback.
///
/// Thread calls and healthy shutdown are COOPERATIVE and may wait indefinitely
/// for a kernel. A deadline cannot preempt a thread or prove GPU quiescence.
/// Process adapters must inject their `ProcessRankOwner::call` deadline here,
/// invalidate lost owners, and use bounded host proxies/reaping, not join a CUDA
/// worker in the parent. Parent boot must not initialize CUDA.
pub trait PipelineTransport {
    fn call(&mut self, rank: PipelineRank, command: PipelineCommand) -> Result<PipelineReply> {
        self.call_observed(rank, command, &mut || {})
    }

    fn call_observed(
        &mut self,
        rank: PipelineRank,
        command: PipelineCommand,
        poll: &mut dyn FnMut(),
    ) -> Result<PipelineReply>;

    /// Admit every member before waiting for any reply. The default explicitly
    /// rejects multi-owner calls: process transports do not support TP yet.
    /// Errors must drain admitted commands; they never stand in for KV ACKs.
    fn call_batch_observed(
        &mut self,
        commands: Vec<(PipelineRank, PipelineCommand)>,
        poll: &mut dyn FnMut(),
    ) -> Result<Vec<PipelineReply>> {
        if commands.len() != 1 {
            return Err(error("transport does not support concurrent TP dispatch"));
        }
        let (rank, command) = commands.into_iter().next().expect("one command");
        self.call_observed(rank, command, poll)
            .map(|reply| vec![reply])
    }

    /// Failure notification wakes in-flight collective peers BEFORE draining.
    /// It carries no completion evidence and must not issue any KV ACK.
    fn call_batch_observed_with_failure(
        &mut self,
        commands: Vec<(PipelineRank, PipelineCommand)>,
        poll: &mut dyn FnMut(),
        on_failure: &mut dyn FnMut(),
    ) -> Result<Vec<PipelineReply>> {
        let result = self.call_batch_observed(commands, poll);
        if result.is_err() {
            on_failure();
        }
        result
    }

    fn outstanding(&self) -> usize;

    /// Attempt ALL owners even when one shutdown fails. Healthy thread shutdown
    /// joins; process implementations must preserve their bounded lifecycle.
    fn shutdown(&mut self) -> Result<()>;

    /// Nonblocking fail-closed teardown. Must retain unknown custody and must NOT
    /// join an unresponsive GPU thread. Process adapters invalidate/escalate via
    /// their bounded proxy. This is not physical retirement evidence.
    fn quarantine(&mut self);
}

impl<T: PipelineTransport + ?Sized> PipelineTransport for Box<T> {
    fn call(&mut self, rank: PipelineRank, command: PipelineCommand) -> Result<PipelineReply> {
        (**self).call(rank, command)
    }
    fn call_observed(
        &mut self,
        rank: PipelineRank,
        command: PipelineCommand,
        poll: &mut dyn FnMut(),
    ) -> Result<PipelineReply> {
        (**self).call_observed(rank, command, poll)
    }
    fn call_batch_observed(
        &mut self,
        commands: Vec<(PipelineRank, PipelineCommand)>,
        poll: &mut dyn FnMut(),
    ) -> Result<Vec<PipelineReply>> {
        (**self).call_batch_observed(commands, poll)
    }
    fn call_batch_observed_with_failure(
        &mut self,
        commands: Vec<(PipelineRank, PipelineCommand)>,
        poll: &mut dyn FnMut(),
        on_failure: &mut dyn FnMut(),
    ) -> Result<Vec<PipelineReply>> {
        (**self).call_batch_observed_with_failure(commands, poll, on_failure)
    }
    fn outstanding(&self) -> usize {
        (**self).outstanding()
    }

    fn shutdown(&mut self) -> Result<()> {
        (**self).shutdown()
    }
    fn quarantine(&mut self) {
        (**self).quarantine()
    }
}

pub(super) type Pool = DataParallelExecutor<PipelineCommand, PipelineReply, Error>;

/// Default thread transport. Unknown custody intentionally leaks its pool rather
/// than entering DataParallelExecutor's potentially unbounded Drop/join path.
pub struct DataPoolPipelineTransport {
    pool: Option<Pool>,
    next_id: u64,
    unavailable: bool,
}

impl DataPoolPipelineTransport {
    pub(super) fn new(pool: Pool) -> Self {
        Self {
            pool: Some(pool),
            next_id: 1,
            unavailable: false,
        }
    }
}

impl PipelineTransport for DataPoolPipelineTransport {
    fn call_observed(
        &mut self,
        rank: PipelineRank,
        command: PipelineCommand,
        poll: &mut dyn FnMut(),
    ) -> Result<PipelineReply> {
        if self.unavailable {
            return Err(error("pipeline transport is quarantined"));
        }
        let pool = self
            .pool
            .as_mut()
            .ok_or_else(|| error("pipeline transport is closed"))?;
        let id = ExecutionTransactionId::new(self.next_id)?;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| error("pipeline command identity exhausted"))?;
        let route = SessionId(u64::from(rank.local.get()) + 1);
        pool.try_submit_to(route, rank.local, id, command)
            .map_err(|submit| error(format!("pipeline owner admission: {:?}", submit.kind)))?;
        let completion = loop {
            poll();
            if let Some(completion) = pool.try_recv() {
                break completion;
            }
            std::thread::sleep(Duration::from_micros(50));
        };
        if completion.transaction != id
            || completion.rank != rank.local
            || completion.session != route
        {
            self.unavailable = true;
            return Err(error("pipeline completion identity mismatch"));
        }
        match completion.outcome {
            CompletionOutcome::Success(reply) => Ok(reply),
            CompletionOutcome::Failed(source) => Err(source),
            other => {
                self.unavailable = true;
                Err(error(format!("pipeline owner unavailable: {other:?}")))
            }
        }
    }

    fn call_batch_observed(
        &mut self,
        commands: Vec<(PipelineRank, PipelineCommand)>,
        poll: &mut dyn FnMut(),
    ) -> Result<Vec<PipelineReply>> {
        self.call_batch_observed_with_failure(commands, poll, &mut || {})
    }

    fn call_batch_observed_with_failure(
        &mut self,
        commands: Vec<(PipelineRank, PipelineCommand)>,
        poll: &mut dyn FnMut(),
        on_failure: &mut dyn FnMut(),
    ) -> Result<Vec<PipelineReply>> {
        if self.unavailable || commands.is_empty() {
            return Err(error("empty batch or unavailable pipeline transport"));
        }
        let pool = self
            .pool
            .as_mut()
            .ok_or_else(|| error("pipeline transport is closed"))?;
        if pool.outstanding() != 0 {
            return Err(error("pipeline batch requires an idle transport"));
        }
        let count = commands.len();
        let end = self
            .next_id
            .checked_add(count as u64)
            .ok_or_else(|| error("pipeline command identity exhausted"))?;
        let mut ranks = std::collections::BTreeSet::new();
        let mut cancellation = Vec::new();
        for (rank, command) in &commands {
            if !ranks.insert(rank.local) {
                return Err(error("duplicate pipeline batch owner"));
            }
            if let PipelineCommand::Execute {
                cancellation: flag, ..
            } = command
            {
                cancellation.push(Arc::clone(flag));
            }
        }
        let start = self.next_id;
        self.next_id = end;
        let mut pending = std::collections::BTreeMap::new();
        let mut replies: Vec<Option<PipelineReply>> = (0..count).map(|_| None).collect();
        let mut failures = Vec::new();
        for (index, (rank, command)) in commands.into_iter().enumerate() {
            let id = ExecutionTransactionId::new(start + index as u64)?;
            let route = SessionId(u64::from(rank.local.get()) + 1);
            match pool.try_submit_to(route, rank.local, id, command) {
                Ok(_) => {
                    pending.insert(id, (index, rank, route));
                }
                Err(rejected) => {
                    on_failure();
                    failures.push(error(format!(
                        "pipeline batch admission: {:?}",
                        rejected.kind
                    )));
                    break;
                }
            }
        }
        while pool.outstanding() != 0 {
            poll();
            if !failures.is_empty() {
                // Wake collective waiters, not fake-complete a physical command.
                for flag in &cancellation {
                    flag.store(true, Ordering::Release);
                }
            }
            let Some(completion) = pool.try_recv() else {
                std::thread::sleep(Duration::from_micros(50));
                continue;
            };
            let expected = pending.remove(&completion.transaction);
            let Some((index, rank, route)) = expected else {
                on_failure();
                self.unavailable = true;
                failures.push(error("unknown pipeline batch completion"));
                continue;
            };
            if completion.rank != rank.local || completion.session != route {
                on_failure();
                self.unavailable = true;
                failures.push(error("pipeline batch completion identity mismatch"));
                continue;
            }
            match completion.outcome {
                CompletionOutcome::Success(reply) => replies[index] = Some(reply),
                CompletionOutcome::Failed(source) => {
                    on_failure();
                    failures.push(source)
                }
                other => {
                    on_failure();
                    self.unavailable = true;
                    failures.push(error(format!(
                        "pipeline batch owner unavailable: {other:?}"
                    )));
                }
            }
        }
        if !failures.is_empty() {
            for flag in &cancellation {
                flag.store(true, Ordering::Release);
            }
            return Err(super::errors("pipeline batch dispatch", failures));
        }
        replies
            .into_iter()
            .map(|reply| reply.ok_or_else(|| error("missing pipeline batch reply")))
            .collect()
    }

    fn outstanding(&self) -> usize {
        self.pool.as_ref().map_or(0, Pool::outstanding)
    }

    fn shutdown(&mut self) -> Result<()> {
        match self.pool.as_mut() {
            Some(pool) => pool
                .shutdown()
                .map_err(|failure| super::pool_shutdown_error(failure)),
            None => Ok(()),
        }
    }

    fn quarantine(&mut self) {
        self.unavailable = true;
        if let Some(pool) = self.pool.take() {
            std::mem::forget(pool);
        }
    }
}

impl Drop for DataPoolPipelineTransport {
    fn drop(&mut self) {
        if self.unavailable {
            self.quarantine();
        }
    }
}
