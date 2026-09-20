//! Host domain protocol shared by thread and process adapters.
//!
//! These are owned domain values, NOT serde for private KV handles. An IPC codec
//! must bound its frames and reconstruct/validate domain values before dispatch.
//! `PipelineCommandKey` is addressing metadata; the non-cloneable physical token
//! remains in `PipelineStageWorker`. A cancellation flag is call-local and must
//! be reconstructed in the child, never serialized as an address.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use ferrule_common::execution::{
    ExecutionBatch, ExecutionTransactionId, KvPageId, KvReservationView, StateSlot,
};
use ferrule_common::{Error, Result};
use ferrule_model::decoder::{
    DecoderKvPageSnapshot, GenericDecoderSequenceState, KvCommitBinding, KvEndProgress,
    PackedDecoderBatch,
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
/// until `into_commit_batch` checks the exact original parent request.
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

    /// Revalidate the logical cohort projection without reconstructing an
    /// owner-local sequence identity. This batch is for RemotePhysicalOwner's
    /// `commit_batch` ONLY; never pass it to a physical backend's `enter`.
    ///
    /// The model cohort compares logical page slots/generations, COW and row
    /// projection, deliberately excluding rank-local execution slots/topologies.
    /// Bind a temporary generation-zero sequence to a DIFFERENT execution slot
    /// than the logical page slot. This preserves the real logical generation
    /// (including arbitrarily deep forks) without serializing private sequence
    /// state or creating a second parent sequence/transaction registry. Physical
    /// readiness still requires an ACK from the original owner and real token.
    pub fn into_commit_batch(
        self,
        expected_batch: &ExecutionBatch,
        expected_reservation: &KvReservationView,
        description: &PipelineStageDescription,
    ) -> Result<PackedDecoderBatch> {
        description.validate()?;
        if &self.batch != expected_batch
            || &self.reservation != expected_reservation
            || self.page_size != description.config.page_size
            || self.batch.sequences().len() != 1
            || self.reservation.positions.end > description.config.max_positions
            || self.page_statuses.len() > description.config.max_pages
        {
            return Err(error(
                "pipeline preparation projection differs from parent request",
            ));
        }
        let statuses = self
            .page_statuses
            .iter()
            .map(|snapshot| (snapshot.page, snapshot.status))
            .collect::<std::collections::BTreeMap<_, _>>();
        if statuses.len() != self.page_statuses.len() {
            return Err(error("duplicate pipeline projection page snapshot"));
        }
        let execution_slot = StateSlot::new(if self.reservation.state_slot == StateSlot::new(0) {
            1
        } else {
            0
        });
        let mut sequences = self.batch.sequences().to_vec();
        sequences[0].state_slot = execution_slot;
        let normalized = ExecutionBatch::new(
            self.batch.mode(),
            self.batch.token_ids().to_vec(),
            self.batch.positions().to_vec(),
            self.batch.kv_write_slots().to_vec(),
            self.batch.logits().to_vec(),
            sequences,
            self.batch.kv_block_ids().to_vec(),
        )
        .with_intent(self.batch.intent());
        let mut reservation = self.reservation;
        reservation.execution_state_slot = execution_slot;
        reservation.execution_generation = 0;
        let states = [
            GenericDecoderSequenceState::with_position(reservation.positions.start, (), ()),
            GenericDecoderSequenceState::with_position(reservation.positions.start, (), ()),
        ];
        let packed = PackedDecoderBatch::lower(
            &normalized,
            &[reservation],
            &states,
            &description.execution_capabilities()?,
            self.page_size,
            &|page| {
                statuses
                    .get(&page)
                    .copied()
                    .unwrap_or(ferrule_model::decoder::DecoderKvPageStatus::Preempted)
            },
        )?;
        if packed
            .protected_pages()
            .iter()
            .copied()
            .ne(statuses.keys().copied())
        {
            return Err(error(
                "pipeline projection snapshots do not cover exact protected pages",
            ));
        }
        Ok(packed)
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
