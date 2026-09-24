//! One physical custody journal for CPU/GPU and thread/process owners.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ferrule_common::execution::{ExecutionBatch, KvPageId, KvReservationView, StateSlot};
use ferrule_model::decoder::{
    DecoderKvBackend, DecoderKvCapacity, DecoderKvCommitBackend, DecoderKvPageSnapshot,
    DecoderKvPrepare, DecoderKvSequenceCustody, GenericDecoderSequenceState, KvEndProgress,
    KvPrepareQuiescenceUnknown, PackedDecoderBatch,
};
use ferrule_model::transformer::SegmentInput;

use super::transport::{
    PipelineCommand as Command, PipelineCommandKey as WireKey, PipelineReply as Reply,
};
use super::{
    BoxedPipelineStageWorker, CpuPipelineStageProgram, PipelineExecutionContext,
    PipelinePreparedProjection, PipelineRank, PipelineStage, PipelineStageBoot,
    PipelineStageDescription, PipelineStageProgram, Result, StageWorkerDispatch, error,
};
use crate::SessionId;
use crate::parallel::data::{PanicQuiescence, ReplicaWorker, WorkRequest};
use crate::parallel::expert::ExpertOwnerStats;

// One bounded owner operation journal, not a decision authority. It binds ACK
// retries to the original physical handle and prevents transport redispatch.
struct Active<T> {
    key: WireKey,
    physical: Option<T>,
    entered: bool,
    working: GenericDecoderSequenceState,
    packed: PackedDecoderBatch,
    execute_dispatched: bool,
    executed: bool,
    ready: bool,
    install_generation: Option<u64>,
    installed: bool,
    published: bool,
    // Cache completion only with the exact arguments acknowledged by the backend.
    retired: Option<(u64, Vec<KvPageId>)>,
    aborted: Option<u64>,
}

/// One owner journal for every device/transport. Construct and dispatch on the
/// owner thread or process child; physical tokens never leave this value.
pub struct PipelineStageWorker<B: DecoderKvBackend, P = CpuPipelineStageProgram> {
    rank: PipelineRank,
    stage: PipelineStage<B, P>,
    sessions: BTreeMap<SessionId, GenericDecoderSequenceState>,
    active: Option<Active<B::Transaction>>,
    executions: usize,
}

#[derive(Debug, Clone)]
pub struct PipelineOwnerStats {
    pub rank: PipelineRank,
    pub layers: std::ops::Range<usize>,
    pub thread: std::thread::ThreadId,
    pub sessions: usize,
    pub executions: usize,
    pub kv: DecoderKvCapacity,
    pub experts: Vec<ExpertOwnerStats>,
    pub expert_outstanding: usize,
}

impl<B, P> PipelineStageWorker<B, P>
where
    B: DecoderKvCommitBackend<SequenceState = GenericDecoderSequenceState>,
    P: PipelineStageProgram<KvView = B::KvView>,
{
    /// Shared thread/process stage bootstrap. Reject mismatched boot metadata on
    /// the owner and explicitly shut down the returned program and backend.
    pub fn new(boot: &PipelineStageBoot, mut stage: PipelineStage<B, P>) -> Result<Self> {
        let validation = (|| {
            boot.validate()?;
            stage.validate()?;
            if stage.description.plan != boot.plan || stage.description.config != boot.config {
                return Err(error("owner factory returned a different plan or capacity"));
            }
            Ok(())
        })();
        if let Err(source) = validation {
            return Err(ferrule_common::Error::with_cleanup(
                "pipeline owner startup",
                source,
                stage.shutdown(),
            ));
        }
        Ok(Self {
            rank: boot.rank,
            stage,
            sessions: BTreeMap::new(),
            active: None,
            executions: 0,
        })
    }

    pub fn boot_description(&self) -> &PipelineStageDescription {
        &self.stage.description
    }

    /// Logical custody that must survive a fenced call/error. This is NOT a
    /// device-quiescence test: ordinary errors retain the worker for rollback;
    /// unknown quiescence requires transport quarantine, not a fabricated ACK.
    pub fn has_active_custody(&self) -> bool {
        self.active.is_some() || self.stage.backend.capacity().active_transactions != 0
    }

    pub fn boxed(self) -> BoxedPipelineStageWorker
    where
        B: 'static,
    {
        BoxedPipelineStageWorker(Box::new(self))
    }

    fn check(&self, key: &WireKey) -> Result<()> {
        if key.rank != self.rank || self.active.as_ref().is_none_or(|active| active.key != *key) {
            return Err(error("stale pipeline owner capability"));
        }
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.physical.is_none())
        {
            return Err(KvPrepareQuiescenceUnknown::wrap(
                key.binding.transaction(),
                error("owner prepare retained custody without a cleanup handle/ACK"),
            ));
        }
        Ok(())
    }

    fn prepare_segment(
        &mut self,
        key: WireKey,
        batch: ExecutionBatch,
        reservation: KvReservationView,
    ) -> Result<Reply> {
        if self.has_active_custody() || key.rank != self.rank {
            return Err(error("pipeline owner is busy or rank-mismatched"));
        }
        if batch.sequences().len() != 1
            || batch.sequences()[0].state_slot != StateSlot::new(0)
            || reservation.positions.end > self.stage.description.config.max_positions
            || batch
                .token_ids()
                .iter()
                .any(|&id| id as usize >= self.stage.description.vocabulary)
        {
            return Err(error("invalid pipeline stage batch bounds"));
        }
        batch.validate(1, &self.stage.description.execution_capabilities()?)?;
        let source = self
            .sessions
            .get(&key.session)
            .ok_or_else(|| error("unknown pipeline owner session"))?;
        let packed = PackedDecoderBatch::lower(
            &batch,
            std::slice::from_ref(&reservation),
            std::slice::from_ref(source),
            &self.stage.capabilities,
            self.stage.description.config.page_size,
            &|page| self.stage.backend.page_status(page),
        )?;
        let sequence = &packed.sequences()[0];
        let custody = [DecoderKvSequenceCustody {
            source_index: 0,
            topology_id: source.topology_id(),
            page_state_slot: sequence.page_state_slot(),
            page_generation: sequence.page_generation(),
            execution_generation: sequence.execution_generation(),
            context_len: sequence.context_len(),
            query_len: sequence.query_len(),
        }];
        let statuses = packed
            .protected_pages()
            .iter()
            .map(|&page| DecoderKvPageSnapshot {
                page,
                status: self.stage.backend.page_status(page),
            })
            .collect::<Vec<_>>();
        let mut working = source.transaction_working_copy()?;
        let step = working.core().begin_step()?;
        working.core_mut().commit_step(step, packed.len())?;
        let transaction = key.binding.transaction();
        // Prepare may submit device work and then fail before issuing a handle.
        // Journal custody first; absence of a handle is not a rollback fence.
        self.active = Some(Active {
            key,
            physical: None,
            entered: false,
            working,
            packed: packed.clone(),
            execute_dispatched: false,
            executed: false,
            ready: false,
            install_generation: None,
            installed: false,
            published: false,
            retired: None,
            aborted: None,
        });
        match self.stage.backend.prepare(DecoderKvPrepare {
            transaction,
            sequences: &custody,
            new_pages: packed.new_pages(),
            writable_pages: packed.writable_pages(),
            cow_replacements: packed.cow_replacements(),
            protected_pages: packed.protected_pages(),
            capacity: self.stage.backend.capacity(),
            page_statuses: &statuses,
        }) {
            Ok(physical) => {
                self.active.as_mut().expect("prepare journal").physical = Some(physical)
            }
            Err(source) => {
                if KvPrepareQuiescenceUnknown::from_error(&source).is_some()
                    || self.stage.backend.capacity().active_transactions != 0
                {
                    return Err(KvPrepareQuiescenceUnknown::wrap(transaction, source));
                }
                // Rejected before work, or cleanup positively proved quiescent.
                self.active = None;
                return Err(source);
            }
        }
        Ok(Reply::Prepared {
            projection: PipelinePreparedProjection::from_packed(&packed, statuses)?,
        })
    }

    fn execute_segment(
        &mut self,
        key: WireKey,
        input: SegmentInput,
        cancellation: Arc<AtomicBool>,
    ) -> Result<Reply> {
        self.check(&key)?;
        let active = self.active.as_mut().expect("stored owner custody");
        if active.execute_dispatched
            || active.aborted.is_some()
            || active.install_generation.is_some()
        {
            return Err(error("duplicate or invalid stage execution"));
        }
        active.execute_dispatched = true;
        if cancellation.load(Ordering::Acquire) {
            return Err(error("pipeline cancelled before stage execution"));
        }
        let packed = &active.packed;
        self.stage.program.validate_execution(
            &active.key.binding,
            packed,
            &[active.key.session.0],
        )?;
        self.stage
            .description
            .validate_input(&input, packed.len())?;
        let state = self
            .sessions
            .get_mut(&active.key.session)
            .expect("checked owner session");
        self.stage.backend.enter(
            active.physical.as_mut().expect("stored physical custody"),
            packed,
            std::slice::from_mut(state),
        )?;
        active.entered = true;
        let result = {
            let mut view = self
                .stage
                .backend
                .active_view(active.physical.as_mut().expect("stored physical custody"))?;
            let transaction = active.key.binding.transaction();
            let sequence_ids = [active.key.session.0];
            self.stage.program.execute(
                packed,
                input,
                &mut view,
                PipelineExecutionContext {
                    transaction,
                    sequence_ids: &sequence_ids,
                    experts: None,
                    check_active: &mut |id| {
                        if id != transaction {
                            Err(error("unknown pipeline transaction"))
                        } else if cancellation.load(Ordering::Acquire) {
                            Err(error("pipeline transaction cancelled"))
                        } else {
                            Ok(())
                        }
                    },
                },
            )
        };
        self.stage
            .backend
            .leave(active.physical.as_mut().expect("stored physical custody"))?;
        active.entered = false;
        let output = result?;
        if cancellation.load(Ordering::Acquire) {
            return Err(error("pipeline cancelled during stage execution"));
        }
        self.stage
            .description
            .validate_output(&output, packed.len())?;
        active.executed = true;
        self.executions = self.executions.saturating_add(1);
        Ok(Reply::Executed { output })
    }

    pub fn dispatch(&mut self, command: Command) -> Result<Reply> {
        match command {
            Command::Describe => Ok(Reply::Description(self.stage.description.clone())),
            Command::Stats => Ok(Reply::Stats(PipelineOwnerStats {
                rank: self.rank,
                layers: self.stage.program.plan().layers(),
                thread: std::thread::current().id(),
                sessions: self.sessions.len(),
                executions: self.executions,
                kv: self.stage.backend.capacity(),
                expert_outstanding: self.stage.program.expert_outstanding(),
                experts: self.stage.program.owner_stats()?,
            })),
            command @ (Command::Create { .. } | Command::Fork { .. }) => {
                let (session, source) = match command {
                    Command::Create { session } => (session, None),
                    Command::Fork { source, target } => (target, Some(source)),
                    _ => unreachable!(),
                };
                if self.has_active_custody()
                    || self.sessions.contains_key(&session)
                    || self.sessions.len() >= self.stage.description.config.session_capacity
                {
                    return Err(error("owner session capacity or lifetime conflict"));
                }
                let state = match source {
                    Some(source) => self
                        .sessions
                        .get(&source)
                        .ok_or_else(|| error("unknown fork source"))?
                        .logical_fork()?,
                    None => GenericDecoderSequenceState::new((), ()),
                };
                let generation = state.core().generation();
                self.sessions.insert(session, state);
                Ok(Reply::Generation(generation))
            }
            Command::Release { session, pages } => {
                if self.has_active_custody() {
                    return Err(error("cannot release an owner with active custody"));
                }
                if !self.sessions.contains_key(&session) {
                    return Err(error("unknown owner session"));
                }
                if pages.len() > self.stage.description.config.max_pages
                    || pages.iter().collect::<BTreeSet<_>>().len() != pages.len()
                {
                    return Err(error("invalid pipeline release pages"));
                }
                self.stage.backend.release(&pages)?;
                self.sessions
                    .remove(&session)
                    .ok_or_else(|| error("unknown owner session"))?;
                Ok(Reply::Ack(KvEndProgress::Complete))
            }
            Command::Prepare {
                key,
                batch,
                reservation,
            } => self.prepare_segment(key, *batch, reservation),
            Command::Execute {
                key,
                input,
                cancellation,
            } => self.execute_segment(key, input, cancellation),
            command => {
                let key = match &command {
                    Command::Ready(key)
                    | Command::Rollback(key)
                    | Command::Publish(key)
                    | Command::Finish(key)
                    | Command::Install { key, .. }
                    | Command::PollInstall { key, .. }
                    | Command::Abort { key, .. }
                    | Command::CheckRetirement { key, .. }
                    | Command::Retire { key, .. } => key,
                    _ => unreachable!(),
                };
                if key.rank != self.rank {
                    return Err(error("incorrect pipeline owner key rank"));
                }
                if matches!(&command, Command::Rollback(_)) && self.active.is_none() {
                    if self.has_active_custody() {
                        return Err(KvPrepareQuiescenceUnknown::wrap(
                            key.binding.transaction(),
                            error("backend retains custody without an owner cleanup ACK"),
                        ));
                    }
                    return Ok(Reply::Ack(KvEndProgress::Complete));
                }
                self.check(key)?;
                self.commit_command(command)
            }
        }
    }

    fn commit_command(&mut self, command: Command) -> Result<Reply> {
        let active = self.active.as_mut().expect("checked active owner custody");
        let backend = &mut self.stage.backend;
        let physical = active
            .physical
            .as_mut()
            .ok_or_else(|| error("owner physical handle absent"))?;
        let progress = match command {
            Command::Ready(key) => {
                if !active.executed
                    || active.install_generation.is_some()
                    || active.aborted.is_some()
                {
                    return Err(error("readiness after physical dispatch"));
                }
                let source = self
                    .sessions
                    .get(&key.session)
                    .ok_or_else(|| error("owner source disappeared"))?;
                backend.preflight_commit_ready(
                    physical,
                    &key.binding,
                    self.rank.global,
                    std::slice::from_ref(source),
                )?;
                active.ready = true;
                KvEndProgress::Complete
            }
            Command::Install { generation, .. } => {
                if !active.ready
                    || active.install_generation.is_some()
                    || active.aborted.is_some()
                    || generation == 0
                {
                    return Err(error("duplicate or invalid physical install"));
                }
                // Set the dispatch marker before the backend call. An Err may
                // represent a lost ACK and therefore is poll-only afterward.
                active.install_generation = Some(generation);
                let progress = backend.install_commit(physical, generation)?;
                active.installed = progress == KvEndProgress::Complete;
                progress
            }
            Command::PollInstall { generation, .. } => {
                if active.install_generation != Some(generation) {
                    return Err(error("stale physical install poll"));
                }
                if active.installed {
                    KvEndProgress::Complete
                } else {
                    let progress = backend.poll_install_ack(physical, generation)?;
                    active.installed = progress == KvEndProgress::Complete;
                    progress
                }
            }
            Command::Abort { generation, .. } => {
                if active.install_generation.is_some() {
                    return Err(error("cannot abort after physical install"));
                }
                if let Some(completed_generation) = active.aborted {
                    if completed_generation != generation {
                        return Err(error("abort retry does not match the completed generation"));
                    }
                    KvEndProgress::Complete
                } else {
                    let progress = backend.abort_prepared(physical, generation)?;
                    if progress == KvEndProgress::Complete {
                        active.aborted = Some(generation);
                    }
                    progress
                }
            }
            Command::Rollback(_) => {
                if active.install_generation.is_some() {
                    return Err(error("cannot rollback after physical install"));
                }
                if active.entered {
                    backend.leave(physical)?;
                    active.entered = false;
                }
                let progress = backend.rollback(&mut active.physical)?;
                if progress == KvEndProgress::Complete {
                    if active.physical.is_some() {
                        return Err(error("rollback retained physical custody"));
                    }
                    self.active = None;
                }
                progress
            }
            Command::Publish(key) => {
                if !active.installed || active.published {
                    return Err(error("invalid owner publication"));
                }
                backend.publish_committed(physical);
                *self
                    .sessions
                    .get_mut(&key.session)
                    .expect("owner source remains allocated") = active.working.clone();
                active.published = true;
                KvEndProgress::Complete
            }
            Command::CheckRetirement { pages, .. } => {
                backend.preflight_retirement(physical, &pages)?;
                KvEndProgress::Complete
            }
            Command::Retire {
                generation, pages, ..
            } => {
                if active.install_generation != Some(generation) || !active.published {
                    return Err(error("retirement before publication"));
                }
                if let Some((completed_generation, completed_pages)) = &active.retired {
                    if *completed_generation != generation || completed_pages != &pages {
                        return Err(error(
                            "retirement retry does not match the completed generation or pages",
                        ));
                    }
                    KvEndProgress::Complete
                } else {
                    let progress = backend.retire_prepared(physical, generation, &pages)?;
                    if progress == KvEndProgress::Complete {
                        active.retired = Some((generation, pages));
                    }
                    progress
                }
            }
            Command::Finish(_) => {
                if active.aborted.is_none() && active.retired.is_none() {
                    return Err(error("owner custody is unfinished"));
                }
                backend.finish_prepared(
                    active
                        .physical
                        .take()
                        .ok_or_else(|| error("owner finish handle absent"))?,
                );
                self.active = None;
                KvEndProgress::Complete
            }
            _ => unreachable!(),
        };
        Ok(Reply::Ack(progress))
    }
}

impl<B, P> ReplicaWorker<Command> for PipelineStageWorker<B, P>
where
    B: DecoderKvCommitBackend<SequenceState = GenericDecoderSequenceState>,
    P: PipelineStageProgram<KvView = B::KvView>,
{
    type Output = Reply;
    type Error = ferrule_common::Error;

    fn execute(&mut self, request: WorkRequest<Command>) -> Result<Reply> {
        if request.rank != self.rank.local {
            return Err(error("incorrect pipeline owner rank"));
        }
        self.dispatch(request.input)
    }

    fn panic_quiescence(&mut self) -> PanicQuiescence {
        // An empty KV journal is not proof that an arbitrary device program's
        // statistics/shutdown panic left all asynchronous resources quiescent.
        PanicQuiescence::Unknown
    }

    fn shutdown(&mut self) -> Result<()> {
        if self.has_active_custody() {
            std::panic::panic_any(PanicQuiescence::Unknown);
        }
        self.stage.shutdown()
    }
}

impl<B, P> StageWorkerDispatch for PipelineStageWorker<B, P>
where
    B: DecoderKvCommitBackend<SequenceState = GenericDecoderSequenceState>,
    P: PipelineStageProgram<KvView = B::KvView>,
{
    fn dispatch(&mut self, command: Command) -> Result<Reply> {
        self.dispatch(command)
    }
    fn boot_description(&self) -> &PipelineStageDescription {
        self.boot_description()
    }
    fn has_active_custody(&self) -> bool {
        self.has_active_custody()
    }
}
