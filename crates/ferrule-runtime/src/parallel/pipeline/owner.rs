//! One physical custody journal for CPU/GPU and thread/process owners.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ferrule_common::execution::{ExecutionBatch, KvPageId, KvReservationView, StateSlot};
use ferrule_model::decoder::{
    DecoderKvBackend, DecoderKvCapacity, DecoderKvPageSnapshot, DecoderKvPrepare,
    DecoderKvSequenceCustody, DecoderSequence, DecoderSequenceCheckout, DecoderSequenceLifecycle,
    KvCommitParticipant, KvEndProgress, KvPrepareQuiescenceUnknown, PackedDecoderBatch,
};
use ferrule_model::transformer::SegmentInput;

use super::transport::{
    PipelineCommand as Command, PipelineCommandKey as WireKey, PipelineReply as Reply,
};
use super::{
    BoxedPipelineStageWorker, CpuPipelineStageProgram, PipelineExecutionContext,
    PipelinePreparedProjection, PipelineRank, PipelineStage, PipelineStageBoot,
    PipelineStageDescription, PipelineStateProgram, Result, StageWorkerDispatch, error,
};
use crate::SessionId;
use crate::parallel::data::{PanicQuiescence, ReplicaWorker, WorkRequest};
use crate::parallel::expert::ExpertOwnerStats;

// One bounded owner operation journal, not a decision authority. It binds ACK
// retries to the original physical handle and prevents transport redispatch.
struct Active<T, S> {
    key: WireKey,
    physical: Option<T>,
    entered: bool,
    working: Option<S>,
    displaced: Option<S>,
    step: ferrule_model::execution::SequenceStepBinding,
    rollback_complete: bool,
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
pub struct PipelineStageWorker<
    B: DecoderKvBackend,
    P: PipelineStateProgram = CpuPipelineStageProgram,
> {
    rank: PipelineRank,
    stage: PipelineStage<B, P>,
    sessions: BTreeMap<SessionId, P::SequenceState>,
    lifecycle: P::Lifecycle,
    release_pending: Option<(SessionId, Vec<KvPageId>, P::SequenceState)>,
    active: Option<Active<B::Transaction, P::SequenceState>>,
    executions: usize,
    #[cfg(feature = "cuda")]
    last_trace_publication: Option<WireKey>,
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
    B: DecoderKvBackend<SequenceState = P::SequenceState>
        + 'static
        + KvCommitParticipant<
            SequenceState = P::SequenceState,
            Transaction = <B as DecoderKvBackend>::Transaction,
        >,
    P: PipelineStateProgram<KvView = <B as DecoderKvBackend>::KvView>,
{
    /// Shared thread/process stage bootstrap. Reject mismatched boot metadata on
    /// the owner and explicitly shut down the returned program and backend.
    pub fn new(boot: &PipelineStageBoot, mut stage: PipelineStage<B, P>) -> Result<Self> {
        let validation = (|| {
            boot.validate()?;
            stage.validate()?;
            if stage.description.plan != boot.plan
                || stage.description.config != boot.config
                || stage.backend.capacity().physical_pages != boot.physical_pages
                || stage.backend.capacity().free_pages != boot.physical_pages
            {
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
        let lifecycle = match stage.program.new_lifecycle() {
            Ok(lifecycle) => lifecycle,
            Err(source) => {
                return Err(ferrule_common::Error::with_cleanup(
                    "pipeline lifecycle startup",
                    source,
                    stage.shutdown(),
                ));
            }
        };
        Ok(Self {
            lifecycle,
            release_pending: None,
            rank: boot.rank,
            stage,
            sessions: BTreeMap::new(),
            active: None,
            executions: 0,
            #[cfg(feature = "cuda")]
            last_trace_publication: None,
        })
    }

    pub fn boot_description(&self) -> &PipelineStageDescription {
        &self.stage.description
    }

    /// Logical custody that must survive a fenced call/error. This is NOT a
    /// device-quiescence test: ordinary errors retain the worker for rollback;
    /// unknown quiescence requires transport quarantine, not a fabricated ACK.
    pub fn has_active_custody(&self) -> bool {
        self.active.is_some()
            || self.release_pending.is_some()
            || self.stage.backend.capacity().active_transactions != 0
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
            .is_some_and(|active| active.physical.is_none() && !active.rollback_complete)
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
        let step = source.core().begin_step()?;
        let transaction = key.binding.transaction();
        let working = self
            .lifecycle
            .checkout(DecoderSequenceCheckout::new(transaction, 0, 0), source)?;
        let checkout_valid =
            working.core() == source.core() && working.topology_id() == source.topology_id();
        // Prepare may submit device work and then fail before issuing a handle.
        // Journal custody first; absence of a handle is not a rollback fence.
        self.active = Some(Active {
            key,
            physical: None,
            entered: false,
            working: Some(working),
            displaced: None,
            step,
            rollback_complete: false,
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
        if !checkout_valid {
            self.active.as_mut().unwrap().rollback_complete = true;
            let cleanup = self.release_working();
            return Err(ferrule_common::Error::with_cleanup(
                "pipeline checkout identity",
                error("checkout changed source core/topology"),
                cleanup,
            ));
        }
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
                self.active.as_mut().unwrap().rollback_complete = true;
                let cleanup = self.release_working();
                return Err(ferrule_common::Error::with_cleanup(
                    "pipeline prepare",
                    source,
                    cleanup,
                ));
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
        if active.rollback_complete
            || active.execute_dispatched
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
        let state = active.working.as_mut().expect("working custody");
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
            self.stage.program.execute_state(
                packed,
                input,
                std::slice::from_mut(state),
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
        let leave = self
            .stage
            .backend
            .leave(active.physical.as_mut().expect("stored physical custody"));
        if leave.is_ok() {
            active.entered = false;
        }
        let output = match result {
            Ok(output) => {
                leave?;
                output
            }
            Err(source) => {
                return Err(ferrule_common::Error::with_cleanup(
                    "pipeline execute",
                    source,
                    leave,
                ));
            }
        };
        if cancellation.load(Ordering::Acquire) {
            return Err(error("pipeline cancelled during stage execution"));
        }
        self.stage
            .description
            .validate_output(&output, packed.len())?;
        let state = active.working.as_mut().expect("working custody");
        let source = self
            .sessions
            .get(&active.key.session)
            .expect("source custody");
        if state.core() != source.core() || state.topology_id() != source.topology_id() {
            return Err(error("program changed working core/topology"));
        }
        self.stage
            .program
            .validate_frontier(packed, std::slice::from_ref(state))?;
        state.core_mut().commit_step(active.step, packed.len())?;
        active.executed = true;
        self.executions = self.executions.saturating_add(1);
        Ok(Reply::Executed { output })
    }

    pub fn dispatch(&mut self, command: Command) -> Result<Reply> {
        match command {
            #[cfg(feature = "cuda")]
            Command::TraceCommitted(request) => {
                if self.last_trace_publication.as_ref() != Some(&request.key)
                    || request.key.rank != self.rank
                    || self.has_active_custody()
                {
                    return Err(error(
                        "committed trace before publication/retirement or wrong identity",
                    ));
                }
                let state = self
                    .sessions
                    .get(&request.key.session)
                    .ok_or_else(|| error("trace session missing"))?;
                if state.core().generation() != request.source.generation
                    || state.core().position() != request.source.committed_tokens
                {
                    return Err(error("committed trace source generation/frontier differs"));
                }
                if !self.stage.program.poll_quiescence()? {
                    return Err(error("trace owner not quiescent"));
                }
                self.stage.program.trace_committed(
                    &mut self.stage.backend,
                    request.key.binding.transaction(),
                    &request.source,
                )?;
                Ok(Reply::Ack(KvEndProgress::Complete))
            }
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
                    Some(source) => {
                        let source = self
                            .sessions
                            .get(&source)
                            .ok_or_else(|| error("unknown fork source"))?;
                        self.lifecycle
                            .logical_fork(source, source.core().position())?
                    }
                    None => self.lifecycle.create()?,
                };
                let generation = state.core().generation();
                self.sessions.insert(session, state);
                Ok(Reply::Generation(generation))
            }
            Command::Release { session, pages } => {
                if self.active.is_some() || self.stage.backend.capacity().active_transactions != 0 {
                    return Err(error("cannot release an owner with active custody"));
                }
                if let Some((pending_session, pending_pages, _)) = &self.release_pending {
                    if *pending_session != session || *pending_pages != pages {
                        return Err(error("release retry differs from pending custody"));
                    }
                } else {
                    if !self.sessions.contains_key(&session) {
                        return Err(error("unknown owner session"));
                    }
                    if pages.len() > self.stage.description.config.max_pages
                        || pages.iter().collect::<BTreeSet<_>>().len() != pages.len()
                    {
                        return Err(error("invalid pipeline release pages"));
                    }
                    self.stage.backend.release(&pages)?;
                    let state = self.sessions.remove(&session).expect("checked session");
                    self.release_pending = Some((session, pages, state));
                }
                let (session, pages, state) = self.release_pending.take().unwrap();
                if let Err(failure) = self.lifecycle.try_release(state) {
                    let (source, state) = failure.into_parts();
                    self.release_pending = Some((session, pages, state));
                    return Err(source);
                }
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

    fn release_working(&mut self) -> Result<()> {
        let active = self.active.as_mut().expect("cleanup custody");
        release_state(&mut self.lifecycle, &mut active.working)?;
        self.active = None;
        Ok(())
    }

    fn commit_command(&mut self, command: Command) -> Result<Reply> {
        if matches!(
            command,
            Command::Rollback(_) | Command::Abort { .. } | Command::Ready(_)
        ) && !self.stage.program.poll_quiescence()?
        {
            return Ok(Reply::Ack(KvEndProgress::Pending));
        }
        if matches!(command, Command::Rollback(_)) {
            let active = self.active.as_mut().expect("checked active owner custody");
            if active.install_generation.is_some() {
                return Err(error("cannot rollback after physical install"));
            }
            if !active.rollback_complete {
                if active.entered {
                    self.stage
                        .backend
                        .leave(active.physical.as_mut().expect("physical custody"))?;
                    active.entered = false;
                }
                let progress = self.stage.backend.rollback(&mut active.physical)?;
                if progress != KvEndProgress::Complete {
                    return Ok(Reply::Ack(progress));
                }
                if active.physical.is_some() {
                    return Err(error("rollback retained physical custody"));
                }
                active.rollback_complete = true;
            }
            self.release_working()?;
            return Ok(Reply::Ack(KvEndProgress::Complete));
        }
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
                let progress = if let Some(completed_generation) = active.aborted {
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
                };
                if progress == KvEndProgress::Complete {
                    release_state(&mut self.lifecycle, &mut active.working)?;
                }
                progress
            }
            Command::Publish(key) => {
                if !active.installed || active.published {
                    return Err(error("invalid owner publication"));
                }
                backend.publish_committed(physical);
                active.displaced = Some(std::mem::replace(
                    self.sessions
                        .get_mut(&key.session)
                        .expect("owner source remains allocated"),
                    active.working.take().expect("working custody"),
                ));
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
                let progress =
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
                    };
                if progress == KvEndProgress::Complete {
                    release_state(&mut self.lifecycle, &mut active.displaced)?;
                }
                progress
            }
            Command::Finish(_) => {
                if active.aborted.is_none() && active.retired.is_none() {
                    return Err(error("owner custody is unfinished"));
                }
                release_state(&mut self.lifecycle, &mut active.working)?;
                release_state(&mut self.lifecycle, &mut active.displaced)?;
                backend.finish_prepared(
                    active
                        .physical
                        .take()
                        .ok_or_else(|| error("owner finish handle absent"))?,
                );
                #[cfg(feature = "cuda")]
                if active.published {
                    self.last_trace_publication = Some(active.key.clone());
                }
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
    B: DecoderKvBackend<SequenceState = P::SequenceState>
        + 'static
        + KvCommitParticipant<
            SequenceState = P::SequenceState,
            Transaction = <B as DecoderKvBackend>::Transaction,
        >,
    P: PipelineStateProgram<KvView = <B as DecoderKvBackend>::KvView>,
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
        while let Some((id, state)) = self.sessions.pop_first() {
            if let Err(failure) = self.lifecycle.try_release(state) {
                let (_, state) = failure.into_parts();
                self.sessions.insert(id, state);
                std::panic::panic_any(PanicQuiescence::Unknown);
            }
        }
        self.stage.shutdown()
    }
}

impl<B, P> StageWorkerDispatch for PipelineStageWorker<B, P>
where
    B: DecoderKvBackend<SequenceState = P::SequenceState>
        + 'static
        + KvCommitParticipant<
            SequenceState = P::SequenceState,
            Transaction = <B as DecoderKvBackend>::Transaction,
        >,
    P: PipelineStateProgram<KvView = <B as DecoderKvBackend>::KvView>,
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
    fn has_hybrid_state(&self) -> bool {
        self.stage.program.hybrid_schema().is_some()
    }
}

fn release_state<S, L: DecoderSequenceLifecycle<S>>(
    lifecycle: &mut L,
    state: &mut Option<S>,
) -> Result<()> {
    if let Some(owned) = state.take()
        && let Err(failure) = lifecycle.try_release(owned)
    {
        let (source, owned) = failure.into_parts();
        *state = Some(owned);
        return Err(source);
    }
    Ok(())
}
