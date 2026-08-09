use super::{
    DecoderCancelProgress, DecoderComposition, DecoderControlPlane, DecoderForwardExecutor,
    DecoderForwardProgress, DecoderForwardResumeProgress, DecoderKvBackend, DecoderKvPageSnapshot,
    DecoderKvPrepare, DecoderKvSequenceCustody, DecoderProposalExecutor,
    DecoderProvisionalRetainContext, DecoderResolverHandle, DecoderSequence,
    DecoderSequenceCheckout, DecoderSequenceLifecycle, DecoderTransactionContext, KvEndProgress,
    PackedDecoderBatch,
};
use crate::execution::{SequenceStepBinding, SequenceTopologyId};
#[cfg(test)]
use ferrule_common::CompletionHub;
use ferrule_common::execution::{
    ExecutionBatch, ExecutionOutput, ExecutionTransactionId, KvPageId,
};
use ferrule_common::{ContinuationId, Error, ResidencyLeaseSet, Result};
use std::collections::{BTreeSet, HashMap, HashSet};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderTransactionPhase {
    Transitioning,
    Prepared,
    Waiting,
    Executed,
    FailedActive,
    FailedQuiescent,
    Publishing,
    Aborting,
    Finished,
    ConsumedRejected,
}
#[derive(Debug, Clone, PartialEq)]
pub enum DecoderTransactionProgress {
    Waiting {
        continuation: ContinuationId,
        wait: super::DecoderWait,
    },
    Complete(ExecutionOutput),
}
#[derive(Debug, Clone, Copy)]
struct DecoderSequenceOwner {
    source_index: usize,
    working_index: usize,
    topology_id: SequenceTopologyId,
    binding: SequenceStepBinding,
    rows: usize,
}
struct DecoderWaiting<C> {
    continuation: C,
    wait: super::DecoderContinuationWait,
}
enum DecoderTransactionState<C, G> {
    Transitioning,
    Prepared,
    Waiting(DecoderWaiting<C>),
    Executed(G),
    FailedActive(C),
    FailedQuiescent(Option<G>),
    Publishing(G),
    Aborting(Option<G>),
    Finished,
    ConsumedRejected,
}
/// Owns each decoder transaction and its exactly-once publication or rollback.
/// Custody includes working states, KV, continuations, leases, and terminal guard.
pub struct DecoderTransaction<D>
where
    D: DecoderComposition,
{
    transaction: ExecutionTransactionId,
    batch: PackedDecoderBatch,
    owners: Box<[DecoderSequenceOwner]>,
    working_states: Vec<D::SequenceState>,
    kv_transaction: Option<<D::KvBackend as DecoderKvBackend>::Transaction>,
    kv_entered: bool,
    held_resume_leases: Vec<ResidencyLeaseSet>,
    continuation_id: Option<ContinuationId>,
    resolver: DecoderResolverHandle,
    publication_staged: bool,
    state: DecoderTransactionState<D::Continuation, D::TerminalGuard>,
    #[cfg(test)]
    fail_next_wait_rebuild: bool,
}
impl<D> DecoderTransaction<D>
where
    D: DecoderComposition,
{
    pub fn prepare(
        transaction: ExecutionTransactionId,
        mut batch: PackedDecoderBatch,
        states: &[D::SequenceState],
        backend: &mut D::KvBackend,
        lifecycle: &mut D::SequenceLifecycle,
        resolver: DecoderResolverHandle,
    ) -> Result<Self> {
        batch.validate_states(states)?;
        let mut owners = Vec::with_capacity(batch.sequences().len());
        let mut working_states = Vec::with_capacity(batch.sequences().len());
        let mut sequence_custody = Vec::with_capacity(batch.sequences().len());
        for (working_index, sequence) in batch.sequences().iter().enumerate() {
            let source = states.get(sequence.state_index()).ok_or_else(|| {
                execution_error(format!(
                    "decoder transaction {} state slot {} no longer exists",
                    transaction.get(),
                    sequence.state_index()
                ))
            })?;
            let binding = source.core().begin_step()?;
            let working = lifecycle.checkout(
                DecoderSequenceCheckout::new(transaction, working_index, sequence.state_index()),
                source,
            )?;
            if working.topology_id() != source.topology_id() || working.core() != source.core() {
                return Err(execution_error(format!(
                    "decoder transaction {} sequence checkout {} changed committed identity or core",
                    transaction.get(),
                    working_index
                )));
            }
            owners.push(DecoderSequenceOwner {
                source_index: sequence.state_index(),
                working_index,
                topology_id: source.topology_id(),
                binding,
                rows: sequence.query_len(),
            });
            sequence_custody.push(DecoderKvSequenceCustody {
                source_index: sequence.state_index(),
                topology_id: source.topology_id(),
                page_state_slot: sequence.page_state_slot(),
                page_generation: sequence.page_generation(),
                execution_generation: sequence.execution_generation(),
                context_len: sequence.context_len(),
                query_len: sequence.query_len(),
            });
            working_states.push(working);
        }
        let page_statuses = batch
            .protected_pages()
            .iter()
            .copied()
            .chain(batch.new_pages().iter().copied())
            .chain(batch.writable_pages().iter().copied())
            .chain(
                batch
                    .cow_replacements()
                    .iter()
                    .flat_map(|cow| [cow.source, cow.replacement]),
            )
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|page| DecoderKvPageSnapshot {
                page,
                status: backend.page_status(page),
            })
            .collect::<Vec<_>>();
        let kv_transaction = backend.prepare(DecoderKvPrepare {
            transaction,
            sequences: &sequence_custody,
            new_pages: batch.new_pages(),
            writable_pages: batch.writable_pages(),
            cow_replacements: batch.cow_replacements(),
            protected_pages: batch.protected_pages(),
            capacity: backend.capacity(),
            page_statuses: &page_statuses,
        })?;
        batch.remap_state_indices_in_sequence_order();
        Ok(Self {
            transaction,
            batch,
            owners: owners.into_boxed_slice(),
            working_states,
            kv_transaction: Some(kv_transaction),
            kv_entered: false,
            held_resume_leases: Vec::new(),
            continuation_id: None,
            resolver,
            publication_staged: false,
            state: DecoderTransactionState::Prepared,
            #[cfg(test)]
            fail_next_wait_rebuild: false,
        })
    }
    pub const fn batch(&self) -> &PackedDecoderBatch {
        &self.batch
    }
    #[cfg(test)]
    pub fn held_resume_lease_count(&self) -> usize {
        self.held_resume_leases.len()
    }
    pub fn phase(&self) -> DecoderTransactionPhase {
        phase_of(&self.state)
    }
    pub fn validate_sources(&self, states: &[D::SequenceState]) -> Result<()> {
        for owner in &self.owners {
            let state = states.get(owner.source_index).ok_or_else(|| {
                execution_error(format!(
                    "decoder transaction {} state slot {} no longer exists",
                    self.transaction.get(),
                    owner.source_index
                ))
            })?;
            if state.topology_id() != owner.topology_id {
                return Err(execution_error(format!(
                    "decoder transaction {} state slot {} changed topology identity from {} to {}",
                    self.transaction.get(),
                    owner.source_index,
                    owner.topology_id.get(),
                    state.topology_id().get()
                )));
            }
            let binding = state.core().begin_step()?;
            if binding != owner.binding {
                return Err(execution_error(format!(
                    "decoder transaction {} state slot {} changed generation or committed position",
                    self.transaction.get(),
                    owner.source_index
                )));
            }
        }
        Ok(())
    }
    pub fn execute(
        &mut self,
        backend: &mut D::KvBackend,
        exec: &mut D::ForwardExecutor,
        control: &DecoderControlPlane,
    ) -> Result<DecoderTransactionProgress> {
        let state = self.take_state();
        if !matches!(state, DecoderTransactionState::Prepared) {
            let phase = phase_of(&state);
            self.restore_state(state);
            return Err(execution_error(format!(
                "decoder transaction {} cannot execute from phase {phase:?}",
                self.transaction.get()
            )));
        }
        let continuation_id = match control.continuations().allocate() {
            Ok(id) => id,
            Err(error) => {
                self.restore_state(DecoderTransactionState::Prepared);
                return Err(error);
            }
        };
        self.continuation_id = Some(continuation_id);
        let kv_transaction = match self.kv_transaction.as_mut() {
            Some(kv_transaction) => kv_transaction,
            None => {
                self.restore_state(DecoderTransactionState::Prepared);
                return Err(internal_error(format!(
                    "decoder transaction {} lost its KV transaction before execution",
                    self.transaction.get()
                )));
            }
        };
        if let Err(error) = backend.enter(kv_transaction, &self.batch, &mut self.working_states) {
            self.restore_state(DecoderTransactionState::FailedQuiescent(None));
            return Err(error);
        }
        self.kv_entered = true;
        let mut kv = match backend.active_view(kv_transaction) {
            Ok(kv) => kv,
            Err(error) => {
                let leave = self.leave_kv(backend);
                self.restore_state(DecoderTransactionState::FailedQuiescent(None));
                return match leave {
                    Ok(()) => Err(error),
                    Err(leave_error) => Err(combined_error(
                        "decoder KV view construction and KV leave both failed",
                        error,
                        leave_error,
                    )),
                };
            }
        };
        let mut ctx = self.take_forward_context(control, Some(continuation_id), false);
        let progress = exec.start_forward(&mut ctx, &self.batch, &mut self.working_states, &mut kv);
        drop(kv);
        self.restore_forward_context(ctx);
        match progress {
            Ok(progress) => self.finish_forward_start(progress, continuation_id, backend),
            Err(error) => {
                let leave = self.leave_kv(backend);
                self.restore_state(DecoderTransactionState::FailedQuiescent(None));
                match leave {
                    Ok(()) => Err(error),
                    Err(leave_error) => Err(combined_error(
                        "decoder forward start and KV leave both failed",
                        error,
                        leave_error,
                    )),
                }
            }
        }
    }
    pub fn resume(
        &mut self,
        continuation_id: ContinuationId,
        leases: ResidencyLeaseSet,
        backend: &mut D::KvBackend,
        exec: &mut D::ForwardExecutor,
        control: &DecoderControlPlane,
    ) -> Result<DecoderTransactionProgress> {
        let state = self.take_state();
        let DecoderTransactionState::Waiting(mut waiting) = state else {
            let phase = phase_of(&state);
            self.restore_state(state);
            return Err(execution_error(format!(
                "decoder transaction {} cannot resume from phase {phase:?}",
                self.transaction.get()
            )));
        };
        if waiting.wait.continuation_id() != continuation_id {
            let expected = waiting.wait.continuation_id();
            self.restore_state(DecoderTransactionState::Waiting(waiting));
            return Err(execution_error(format!(
                "decoder transaction {} continuation mismatch: expected {}, got {}",
                self.transaction.get(),
                expected.get(),
                continuation_id.get()
            )));
        }
        if let Err(error) = waiting.wait.validate_resume_leases(&leases) {
            self.restore_state(DecoderTransactionState::Waiting(waiting));
            return Err(error);
        }
        // Custody transfers before every subsequent fallible operation. A view
        // construction or forward error therefore cannot drop a non-cloneable lease.
        self.held_resume_leases.push(leases);
        let kv_transaction = match self.kv_transaction.as_mut() {
            Some(kv_transaction) => kv_transaction,
            None => {
                self.restore_state(DecoderTransactionState::Waiting(waiting));
                return Err(internal_error(format!(
                    "decoder transaction {} lost its KV transaction while waiting",
                    self.transaction.get()
                )));
            }
        };
        let mut kv = match backend.active_view(kv_transaction) {
            Ok(kv) => kv,
            Err(error) => {
                self.restore_state(DecoderTransactionState::Waiting(waiting));
                return Err(error);
            }
        };
        let mut ctx = self.take_forward_context(control, Some(continuation_id), true);
        let progress = exec.resume_forward(
            &mut ctx,
            &self.batch,
            &mut self.working_states,
            &mut kv,
            &mut waiting.continuation,
        );
        drop(kv);
        self.restore_forward_context(ctx);
        match progress {
            Ok(DecoderForwardResumeProgress::Waiting(wait)) => {
                let wait = match self.rebuild_wait(waiting.wait.continuation_id(), wait) {
                    Ok(wait) => wait,
                    Err(error) => {
                        self.restore_state(DecoderTransactionState::FailedActive(
                            waiting.continuation,
                        ));
                        return Err(error);
                    }
                };
                waiting.wait = wait;
                let progress = waiting_progress(&waiting);
                self.restore_state(DecoderTransactionState::Waiting(waiting));
                Ok(progress)
            }
            Ok(DecoderForwardResumeProgress::Complete {
                logits,
                terminal_guard,
            }) => {
                drop(waiting);
                self.finish_forward_complete(logits, terminal_guard, backend)
            }
            Ok(DecoderForwardResumeProgress::FailedActive(error)) | Err(error) => {
                self.restore_state(DecoderTransactionState::FailedActive(waiting.continuation));
                Err(error)
            }
            Ok(DecoderForwardResumeProgress::FailedQuiescent(error)) => {
                drop(waiting);
                let leave = self.leave_kv(backend);
                self.restore_state(DecoderTransactionState::FailedQuiescent(None));
                match leave {
                    Ok(()) => Err(error),
                    Err(leave_error) => Err(combined_error(
                        "decoder forward resume and KV leave both failed",
                        error,
                        leave_error,
                    )),
                }
            }
        }
    }
    pub fn retain_provisional(
        &mut self,
        sources: &[D::SequenceState],
        branches: &[D::SequenceState],
        executed_rows: &[usize],
        retained_rows: &[usize],
        backend: &mut D::KvBackend,
        proposal: &mut D::Proposal,
    ) -> Result<()> {
        self.validate_sources(branches)?;
        let state = self.take_state();
        let DecoderTransactionState::Executed(terminal_guard) = state else {
            let phase = phase_of(&state);
            self.restore_state(state);
            return Err(execution_error(format!(
                "decoder transaction {} cannot retain a provisional prefix from phase {phase:?}",
                self.transaction.get()
            )));
        };
        let restore_guard = |shell: &mut Self, guard| {
            shell.restore_state(DecoderTransactionState::Executed(guard));
        };
        let sequence_count = self.owners.len();
        if sequence_count == 0
            || sources.len() != sequence_count
            || branches.len() != sequence_count
            || executed_rows.len() != sequence_count
            || retained_rows.len() != sequence_count
        {
            restore_guard(self, terminal_guard);
            return Err(execution_error(format!(
                "decoder provisional cohort shape mismatch: sources={} branches={} executed={} retained={} transaction={}",
                sources.len(),
                branches.len(),
                executed_rows.len(),
                retained_rows.len(),
                self.transaction.get()
            )));
        }
        let mut committed_cores = Vec::with_capacity(sequence_count);
        for sequence in 0..sequence_count {
            let owner = self.owners[sequence];
            let executed = executed_rows[sequence];
            let retained = retained_rows[sequence];
            if executed == 0 || retained == 0 || retained > executed || executed != owner.rows {
                restore_guard(self, terminal_guard);
                return Err(execution_error(format!(
                    "invalid decoder retained prefix for sequence {sequence}: retained={retained} executed={executed} prepared_rows={}",
                    owner.rows
                )));
            }
            let mut committed = self.working_states[owner.working_index].core().clone();
            if let Err(error) = committed.commit_step(owner.binding, retained) {
                restore_guard(self, terminal_guard);
                return Err(error);
            }
            committed_cores.push(committed);
        }
        let kv_transaction = match self.kv_transaction.as_mut() {
            Some(kv_transaction) => kv_transaction,
            None => {
                restore_guard(self, terminal_guard);
                return Err(internal_error(format!(
                    "decoder transaction {} lost its KV transaction before provisional retain",
                    self.transaction.get()
                )));
            }
        };
        let result = proposal.retain_provisional(&mut DecoderProvisionalRetainContext::new(
            self.transaction,
            sources,
            &mut self.working_states,
            &self.batch,
            executed_rows,
            retained_rows,
            backend,
            kv_transaction,
        ));
        if let Err(error) = result {
            restore_guard(self, terminal_guard);
            return Err(error);
        }
        for (((owner, &retained), working), committed_core) in self
            .owners
            .iter_mut()
            .zip(retained_rows)
            .zip(&mut self.working_states)
            .zip(committed_cores)
        {
            *working.core_mut() = committed_core;
            owner.rows = retained;
        }
        self.publication_staged = true;
        restore_guard(self, terminal_guard);
        Ok(())
    }
    pub fn publish(
        &mut self,
        states: &mut [D::SequenceState],
        backend: &mut D::KvBackend,
    ) -> Result<KvEndProgress> {
        self.validate_sources(states)?;
        let state = self.take_state();
        let guard = match state {
            DecoderTransactionState::Executed(guard) => {
                if let Err(error) = self.stage_publication() {
                    self.restore_state(DecoderTransactionState::Executed(guard));
                    return Err(error);
                }
                guard
            }
            DecoderTransactionState::Publishing(guard) => guard,
            other => {
                let phase = phase_of(&other);
                self.restore_state(other);
                return Err(execution_error(format!(
                    "decoder transaction {} cannot publish from phase {phase:?}",
                    self.transaction.get()
                )));
            }
        };
        let progress = match backend.commit(&mut self.kv_transaction) {
            Ok(progress) => progress,
            Err(error) => {
                if self.kv_transaction.is_some() {
                    self.restore_state(DecoderTransactionState::Publishing(guard));
                    return Err(error);
                }
                self.discard_consumed_work();
                self.restore_state(DecoderTransactionState::ConsumedRejected);
                return Err(consumed_kv_transaction_error(
                    "commit",
                    self.transaction,
                    error,
                ));
            }
        };
        if let Err(error) = self.validate_kv_end_transaction(progress) {
            if self.kv_transaction.is_some() {
                self.restore_state(DecoderTransactionState::Publishing(guard));
            } else {
                self.discard_consumed_work();
                self.restore_state(DecoderTransactionState::ConsumedRejected);
            }
            return Err(error);
        }
        match progress {
            KvEndProgress::Pending => {
                self.restore_state(DecoderTransactionState::Publishing(guard));
                Ok(KvEndProgress::Pending)
            }
            KvEndProgress::Complete => {
                self.publish_working_states(states);
                self.held_resume_leases.clear();
                drop(guard);
                self.restore_state(DecoderTransactionState::Finished);
                Ok(KvEndProgress::Complete)
            }
            KvEndProgress::ConsumedRejected => {
                self.discard_consumed_work();
                drop(guard);
                self.restore_state(DecoderTransactionState::ConsumedRejected);
                Ok(KvEndProgress::ConsumedRejected)
            }
        }
    }
    pub fn abort(
        &mut self,
        backend: &mut D::KvBackend,
        exec: &mut D::ForwardExecutor,
        control: &DecoderControlPlane,
    ) -> Result<KvEndProgress> {
        let state = self.take_state();
        match state {
            DecoderTransactionState::Waiting(waiting) => {
                self.cancel_active(waiting.continuation, backend, exec, control)
            }
            DecoderTransactionState::FailedActive(continuation) => {
                self.cancel_active(continuation, backend, exec, control)
            }
            DecoderTransactionState::Prepared => self.finish_abort(None, backend),
            DecoderTransactionState::Executed(guard) => self.finish_abort(Some(guard), backend),
            DecoderTransactionState::FailedQuiescent(guard) => self.finish_abort(guard, backend),
            DecoderTransactionState::Aborting(guard) => self.finish_abort(guard, backend),
            DecoderTransactionState::Publishing(guard) => {
                self.restore_state(DecoderTransactionState::Publishing(guard));
                Err(execution_error(format!(
                    "decoder transaction {} cannot reverse an in-progress publish into abort",
                    self.transaction.get()
                )))
            }
            DecoderTransactionState::Finished | DecoderTransactionState::ConsumedRejected => {
                let phase = phase_of(&state);
                self.restore_state(state);
                Err(execution_error(format!(
                    "decoder transaction {} is already terminal in phase {phase:?}",
                    self.transaction.get()
                )))
            }
            DecoderTransactionState::Transitioning => {
                self.restore_state(DecoderTransactionState::Transitioning);
                Err(internal_error(format!(
                    "decoder transaction {} is already transitioning",
                    self.transaction.get()
                )))
            }
        }
    }
    fn cancel_active(
        &mut self,
        mut continuation: D::Continuation,
        backend: &mut D::KvBackend,
        exec: &mut D::ForwardExecutor,
        control: &DecoderControlPlane,
    ) -> Result<KvEndProgress> {
        let kv_transaction = match self.kv_transaction.as_mut() {
            Some(kv_transaction) => kv_transaction,
            None => {
                self.restore_state(DecoderTransactionState::FailedActive(continuation));
                return Err(internal_error(format!(
                    "decoder transaction {} lost its KV transaction during cancellation",
                    self.transaction.get()
                )));
            }
        };
        let mut kv = match backend.active_view(kv_transaction) {
            Ok(kv) => kv,
            Err(error) => {
                self.restore_state(DecoderTransactionState::FailedActive(continuation));
                return Err(error);
            }
        };
        let mut ctx = self.take_forward_context(control, self.continuation_id, false);
        let cancel = exec.cancel_forward(
            &mut ctx,
            &self.batch,
            &mut self.working_states,
            &mut kv,
            &mut continuation,
        );
        drop(kv);
        self.restore_forward_context(ctx);
        match cancel {
            Ok(DecoderCancelProgress::Waiting) => {
                self.restore_state(DecoderTransactionState::FailedActive(continuation));
                Ok(KvEndProgress::Pending)
            }
            Ok(DecoderCancelProgress::Complete) => {
                drop(continuation);
                self.finish_abort(None, backend)
            }
            Err(error) => {
                self.restore_state(DecoderTransactionState::FailedActive(continuation));
                Err(error)
            }
        }
    }
    fn finish_abort(
        &mut self,
        guard: Option<D::TerminalGuard>,
        backend: &mut D::KvBackend,
    ) -> Result<KvEndProgress> {
        if let Err(error) = self.leave_kv(backend) {
            self.restore_state(DecoderTransactionState::FailedQuiescent(guard));
            return Err(error);
        }
        let progress = match backend.rollback(&mut self.kv_transaction) {
            Ok(progress) => progress,
            Err(error) => {
                if self.kv_transaction.is_some() {
                    self.restore_state(DecoderTransactionState::Aborting(guard));
                    return Err(error);
                }
                self.discard_consumed_work();
                self.restore_state(DecoderTransactionState::ConsumedRejected);
                return Err(consumed_kv_transaction_error(
                    "rollback",
                    self.transaction,
                    error,
                ));
            }
        };
        if let Err(error) = self.validate_kv_end_transaction(progress) {
            if self.kv_transaction.is_some() {
                self.restore_state(DecoderTransactionState::Aborting(guard));
            } else {
                self.discard_consumed_work();
                self.restore_state(DecoderTransactionState::ConsumedRejected);
            }
            return Err(error);
        }
        match progress {
            KvEndProgress::Pending => {
                self.restore_state(DecoderTransactionState::Aborting(guard));
                Ok(KvEndProgress::Pending)
            }
            KvEndProgress::Complete => {
                self.discard_consumed_work();
                drop(guard);
                self.restore_state(DecoderTransactionState::Finished);
                Ok(KvEndProgress::Complete)
            }
            KvEndProgress::ConsumedRejected => {
                self.discard_consumed_work();
                drop(guard);
                self.restore_state(DecoderTransactionState::ConsumedRejected);
                Ok(KvEndProgress::ConsumedRejected)
            }
        }
    }
    fn finish_forward_start(
        &mut self,
        progress: DecoderForwardProgress<D::Continuation, D::TerminalGuard>,
        continuation_id: ContinuationId,
        backend: &mut D::KvBackend,
    ) -> Result<DecoderTransactionProgress> {
        match progress {
            DecoderForwardProgress::Waiting { continuation, wait } => {
                let wait = match self.rebuild_wait(continuation_id, wait) {
                    Ok(wait) => wait,
                    Err(error) => {
                        self.restore_state(DecoderTransactionState::FailedActive(continuation));
                        return Err(error);
                    }
                };
                let waiting = DecoderWaiting { continuation, wait };
                let progress = waiting_progress(&waiting);
                self.restore_state(DecoderTransactionState::Waiting(waiting));
                Ok(progress)
            }
            DecoderForwardProgress::Complete {
                logits,
                terminal_guard,
            } => self.finish_forward_complete(logits, terminal_guard, backend),
            DecoderForwardProgress::FailedActive {
                continuation,
                error,
            } => {
                self.restore_state(DecoderTransactionState::FailedActive(continuation));
                Err(error)
            }
            DecoderForwardProgress::FailedQuiescent(error) => {
                let leave = self.leave_kv(backend);
                self.restore_state(DecoderTransactionState::FailedQuiescent(None));
                match leave {
                    Ok(()) => Err(error),
                    Err(leave_error) => Err(combined_error(
                        "decoder forward and KV leave both failed",
                        error,
                        leave_error,
                    )),
                }
            }
        }
    }
    fn finish_forward_complete(
        &mut self,
        logits: super::DecoderLogits,
        terminal_guard: D::TerminalGuard,
        backend: &mut D::KvBackend,
    ) -> Result<DecoderTransactionProgress> {
        if let Err(error) = self.leave_kv(backend) {
            self.restore_state(DecoderTransactionState::FailedQuiescent(Some(
                terminal_guard,
            )));
            return Err(error);
        }
        let output = match logits.lower(self.batch.logits_plan()) {
            Ok(output) => output,
            Err(error) => {
                self.restore_state(DecoderTransactionState::FailedQuiescent(Some(
                    terminal_guard,
                )));
                return Err(error);
            }
        };
        self.restore_state(DecoderTransactionState::Executed(terminal_guard));
        Ok(DecoderTransactionProgress::Complete(output))
    }
    fn leave_kv(&mut self, backend: &mut D::KvBackend) -> Result<()> {
        if !self.kv_entered {
            return Ok(());
        }
        let kv_transaction = self.kv_transaction.as_mut().ok_or_else(|| {
            internal_error(format!(
                "decoder transaction {} lost an entered KV transaction",
                self.transaction.get()
            ))
        })?;
        backend.leave(kv_transaction)?;
        self.kv_entered = false;
        Ok(())
    }
    fn stage_publication(&mut self) -> Result<()> {
        if self.publication_staged {
            return Ok(());
        }
        for owner in &self.owners {
            let working = self
                .working_states
                .get_mut(owner.working_index)
                .ok_or_else(|| {
                    internal_error(format!(
                        "decoder transaction {} lost working state {}",
                        self.transaction.get(),
                        owner.working_index
                    ))
                })?;
            working.core_mut().commit_step(owner.binding, owner.rows)?;
        }
        self.publication_staged = true;
        Ok(())
    }
    fn publish_working_states(&mut self, states: &mut [D::SequenceState]) {
        debug_assert_eq!(self.working_states.len(), self.owners.len());
        debug_assert!(
            self.owners
                .iter()
                .all(|owner| owner.source_index < states.len())
        );
        let working_states = std::mem::take(&mut self.working_states);
        for (owner, working) in self.owners.iter().zip(working_states) {
            states[owner.source_index] = working;
        }
    }
    fn discard_consumed_work(&mut self) {
        self.working_states.clear();
        self.held_resume_leases.clear();
    }
    fn take_forward_context(
        &mut self,
        control: &DecoderControlPlane,
        continuation_id: Option<ContinuationId>,
        has_current_resume_lease: bool,
    ) -> DecoderTransactionContext {
        DecoderTransactionContext::with_services(
            self.transaction,
            control.clone(),
            std::mem::take(&mut self.held_resume_leases),
            has_current_resume_lease,
            continuation_id,
            self.resolver.clone(),
        )
    }
    fn restore_forward_context(&mut self, context: DecoderTransactionContext) {
        debug_assert!(self.held_resume_leases.is_empty());
        self.held_resume_leases = context.into_leases();
    }
    fn validate_kv_end_transaction(&self, progress: KvEndProgress) -> Result<()> {
        let kv_transaction_present = self.kv_transaction.is_some();
        match progress {
            KvEndProgress::Pending if kv_transaction_present => Ok(()),
            KvEndProgress::Complete | KvEndProgress::ConsumedRejected
                if !kv_transaction_present =>
            {
                Ok(())
            }
            KvEndProgress::Pending => Err(internal_error(format!(
                "decoder transaction {} KV backend consumed a pending KV transaction",
                self.transaction.get()
            ))),
            KvEndProgress::Complete | KvEndProgress::ConsumedRejected => {
                Err(internal_error(format!(
                    "decoder transaction {} KV backend reported terminal progress without consuming its KV transaction",
                    self.transaction.get()
                )))
            }
        }
    }
    fn rebuild_wait(
        &mut self,
        continuation_id: ContinuationId,
        wait: super::DecoderWait,
    ) -> Result<super::DecoderContinuationWait> {
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_wait_rebuild) {
            return Err(execution_error("injected decoder wait rebuild failure"));
        }
        super::DecoderContinuationWait::from_wait(continuation_id, wait)
    }
    fn take_state(&mut self) -> DecoderTransactionState<D::Continuation, D::TerminalGuard> {
        std::mem::replace(&mut self.state, DecoderTransactionState::Transitioning)
    }
    fn restore_state(&mut self, state: DecoderTransactionState<D::Continuation, D::TerminalGuard>) {
        debug_assert!(matches!(self.state, DecoderTransactionState::Transitioning));
        self.state = state;
    }
}
fn phase_of<C, G>(state: &DecoderTransactionState<C, G>) -> DecoderTransactionPhase {
    match state {
        DecoderTransactionState::Transitioning => DecoderTransactionPhase::Transitioning,
        DecoderTransactionState::Prepared => DecoderTransactionPhase::Prepared,
        DecoderTransactionState::Waiting(_) => DecoderTransactionPhase::Waiting,
        DecoderTransactionState::Executed(_) => DecoderTransactionPhase::Executed,
        DecoderTransactionState::FailedActive(_) => DecoderTransactionPhase::FailedActive,
        DecoderTransactionState::FailedQuiescent(_) => DecoderTransactionPhase::FailedQuiescent,
        DecoderTransactionState::Publishing(_) => DecoderTransactionPhase::Publishing,
        DecoderTransactionState::Aborting(_) => DecoderTransactionPhase::Aborting,
        DecoderTransactionState::Finished => DecoderTransactionPhase::Finished,
        DecoderTransactionState::ConsumedRejected => DecoderTransactionPhase::ConsumedRejected,
    }
}
fn waiting_progress<C>(waiting: &DecoderWaiting<C>) -> DecoderTransactionProgress {
    DecoderTransactionProgress::Waiting {
        continuation: waiting.wait.continuation_id(),
        wait: waiting.wait.wait().clone(),
    }
}
/// Ownership registry for independent packed decoder transactions.
pub struct PackedTransactionRegistry<D>
where
    D: DecoderComposition,
{
    error_context: &'static str,
    control: DecoderControlPlane,
    resolver: DecoderResolverHandle,
    transactions: HashMap<ExecutionTransactionId, DecoderTransaction<D>>,
    sequence_owners: HashMap<SequenceTopologyId, ExecutionTransactionId>,
    page_readers: HashMap<KvPageId, BTreeSet<ExecutionTransactionId>>,
    page_writers: HashMap<KvPageId, ExecutionTransactionId>,
}
impl<D> PackedTransactionRegistry<D>
where
    D: DecoderComposition,
{
    #[cfg(test)]
    pub fn new(error_context: &'static str) -> Self {
        Self::with_control(
            error_context,
            DecoderControlPlane::new(CompletionHub::new()),
        )
    }
    pub fn with_control(error_context: &'static str, control: DecoderControlPlane) -> Self {
        Self {
            error_context,
            control,
            resolver: DecoderResolverHandle::default(),
            transactions: HashMap::new(),
            sequence_owners: HashMap::new(),
            page_readers: HashMap::new(),
            page_writers: HashMap::new(),
        }
    }
    pub fn set_resolver(&mut self, resolver: DecoderResolverHandle) -> Result<()> {
        if !self.transactions.is_empty() {
            return Err(execution_error(
                "cannot replace decoder resolver while transactions are active",
            ));
        }
        self.resolver = resolver;
        Ok(())
    }
    #[cfg(test)]
    pub fn prepare(
        &mut self,
        transaction: ExecutionTransactionId,
        batch: PackedDecoderBatch,
        states: &[D::SequenceState],
        backend: &mut D::KvBackend,
    ) -> Result<()>
    where
        D::SequenceLifecycle: Default,
    {
        self.prepare_with_lifecycle(
            transaction,
            batch,
            states,
            backend,
            &mut D::SequenceLifecycle::default(),
        )
    }
    pub fn prepare_with_lifecycle(
        &mut self,
        transaction: ExecutionTransactionId,
        batch: PackedDecoderBatch,
        states: &[D::SequenceState],
        backend: &mut D::KvBackend,
        lifecycle: &mut D::SequenceLifecycle,
    ) -> Result<()> {
        self.validate_prepare(transaction, &batch, states)?;
        let mutation_pages = batch
            .new_pages()
            .iter()
            .copied()
            .chain(batch.writable_pages().iter().copied())
            .chain(batch.cow_replacements().iter().map(|cow| cow.replacement))
            .collect::<BTreeSet<_>>();
        let protected_pages = batch
            .protected_pages()
            .iter()
            .copied()
            .filter(|page| !mutation_pages.contains(page))
            .collect::<Vec<_>>();
        let shell = DecoderTransaction::<D>::prepare(
            transaction,
            batch,
            states,
            backend,
            lifecycle,
            self.resolver.clone(),
        )?;
        for owner in &shell.owners {
            self.sequence_owners.insert(owner.topology_id, transaction);
        }
        for page in protected_pages {
            self.page_readers
                .entry(page)
                .or_default()
                .insert(transaction);
        }
        for page in mutation_pages {
            self.page_writers.insert(page, transaction);
        }
        let replaced = self.transactions.insert(transaction, shell);
        debug_assert!(replaced.is_none());
        Ok(())
    }
    pub fn execute_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &[D::SequenceState],
        batch: &ExecutionBatch,
        backend: &mut D::KvBackend,
        exec: &mut D::ForwardExecutor,
    ) -> Result<DecoderTransactionProgress> {
        let shell = self.shell(transaction)?;
        shell.batch().validate_source_batch(batch)?;
        shell.validate_sources(states)?;
        let control = self.control.clone();
        self.shell_mut(transaction)?
            .execute(backend, exec, &control)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn resume_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &[D::SequenceState],
        batch: &ExecutionBatch,
        continuation: ContinuationId,
        leases: ResidencyLeaseSet,
        backend: &mut D::KvBackend,
        exec: &mut D::ForwardExecutor,
    ) -> Result<DecoderTransactionProgress> {
        let shell = self.shell(transaction)?;
        shell.batch().validate_source_batch(batch)?;
        shell.validate_sources(states)?;
        let control = self.control.clone();
        self.shell_mut(transaction)?
            .resume(continuation, leases, backend, exec, &control)
    }
    pub fn retain_provisional(
        &mut self,
        transaction: ExecutionTransactionId,
        sources: &[D::SequenceState],
        branches: &[D::SequenceState],
        executed_rows: &[usize],
        retained_rows: &[usize],
        backend: &mut D::KvBackend,
        proposal: &mut D::Proposal,
    ) -> Result<()> {
        self.shell_mut(transaction)?.retain_provisional(
            sources,
            branches,
            executed_rows,
            retained_rows,
            backend,
            proposal,
        )
    }
    pub fn publish(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [D::SequenceState],
        backend: &mut D::KvBackend,
    ) -> Result<KvEndProgress> {
        let result = self.shell_mut(transaction)?.publish(states, backend);
        match result {
            Ok(progress) => {
                if matches!(
                    progress,
                    KvEndProgress::Complete | KvEndProgress::ConsumedRejected
                ) {
                    self.finish(transaction);
                }
                Ok(progress)
            }
            Err(error) => {
                if self
                    .transactions
                    .get(&transaction)
                    .is_some_and(|shell| shell.phase() == DecoderTransactionPhase::ConsumedRejected)
                {
                    self.finish(transaction);
                }
                Err(error)
            }
        }
    }
    pub fn abort(
        &mut self,
        transaction: ExecutionTransactionId,
        backend: &mut D::KvBackend,
        exec: &mut D::ForwardExecutor,
    ) -> Result<KvEndProgress> {
        let control = self.control.clone();
        let result = self.shell_mut(transaction)?.abort(backend, exec, &control);
        match result {
            Ok(progress) => {
                if matches!(
                    progress,
                    KvEndProgress::Complete | KvEndProgress::ConsumedRejected
                ) {
                    self.finish(transaction);
                }
                Ok(progress)
            }
            Err(error) => {
                if self
                    .transactions
                    .get(&transaction)
                    .is_some_and(|shell| shell.phase() == DecoderTransactionPhase::ConsumedRejected)
                {
                    self.finish(transaction);
                }
                Err(error)
            }
        }
    }
    #[cfg(test)]
    pub fn phase(&self, transaction: ExecutionTransactionId) -> Result<DecoderTransactionPhase> {
        Ok(self.shell(transaction)?.phase())
    }
    #[cfg(test)]
    pub fn held_resume_lease_count(&self, transaction: ExecutionTransactionId) -> Result<usize> {
        Ok(self.shell(transaction)?.held_resume_lease_count())
    }
    #[cfg(test)]
    pub fn take_kv_transaction(
        &mut self,
        transaction: ExecutionTransactionId,
    ) -> Result<<D::KvBackend as DecoderKvBackend>::Transaction> {
        self.shell_mut(transaction)?
            .kv_transaction
            .take()
            .ok_or_else(|| internal_error("decoder test fault injection found no KV transaction"))
    }
    #[cfg(test)]
    pub fn restore_kv_transaction(
        &mut self,
        transaction: ExecutionTransactionId,
        kv_transaction: <D::KvBackend as DecoderKvBackend>::Transaction,
    ) -> Result<()> {
        let shell = self.shell_mut(transaction)?;
        if shell.kv_transaction.is_some() {
            return Err(internal_error(
                "decoder test fault injection would replace a KV transaction",
            ));
        }
        shell.kv_transaction = Some(kv_transaction);
        Ok(())
    }
    #[cfg(test)]
    pub fn fail_next_wait_rebuild(&mut self, transaction: ExecutionTransactionId) -> Result<()> {
        self.shell_mut(transaction)?.fail_next_wait_rebuild = true;
        Ok(())
    }
    pub fn contains(&self, transaction: ExecutionTransactionId) -> bool {
        self.transactions.contains_key(&transaction)
    }
    pub fn len(&self) -> usize {
        self.transactions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.transactions.is_empty()
    }
    pub fn ensure_sequence_available(
        &self,
        state: &D::SequenceState,
        operation: &str,
    ) -> Result<()> {
        if let Some(transaction) = self.sequence_owners.get(&state.topology_id()) {
            return Err(execution_error(format!(
                "cannot {operation}: {} sequence topology {} is owned by transaction {}",
                self.error_context,
                state.topology_id().get(),
                transaction.get()
            )));
        }
        Ok(())
    }
    pub fn ensure_pages_available(&self, pages: &[KvPageId], operation: &str) -> Result<()> {
        for page in pages {
            if let Some(readers) = self
                .page_readers
                .get(page)
                .filter(|readers| !readers.is_empty())
            {
                return Err(execution_error(format!(
                    "cannot {operation} {} KV page {} while transaction(s) {:?} retain read custody",
                    self.error_context, page.0, readers
                )));
            }
            if let Some(writer) = self.page_writers.get(page) {
                return Err(execution_error(format!(
                    "cannot {operation} {} KV page {} while transaction {} retains write custody",
                    self.error_context,
                    page.0,
                    writer.get()
                )));
            }
        }
        Ok(())
    }
    fn validate_prepare(
        &self,
        transaction: ExecutionTransactionId,
        batch: &PackedDecoderBatch,
        states: &[D::SequenceState],
    ) -> Result<()> {
        batch.validate_states(states)?;
        if self.transactions.contains_key(&transaction) {
            return Err(execution_error(format!(
                "{} decoder transaction {} is already registered",
                self.error_context,
                transaction.get()
            )));
        }
        for page in batch
            .new_pages()
            .iter()
            .copied()
            .chain(batch.writable_pages().iter().copied())
            .chain(batch.cow_replacements().iter().map(|cow| cow.replacement))
        {
            if let Some(readers) = self
                .page_readers
                .get(&page)
                .filter(|readers| !readers.is_empty())
            {
                return Err(execution_error(format!(
                    "{} decoder transaction {} cannot mutate KV page {} while transaction(s) {:?} retain custody",
                    self.error_context,
                    transaction.get(),
                    page.0,
                    readers
                )));
            }
            if let Some(writer) = self.page_writers.get(&page) {
                return Err(execution_error(format!(
                    "{} decoder transaction {} cannot mutate KV page {} while transaction {} already mutates it",
                    self.error_context,
                    transaction.get(),
                    page.0,
                    writer.get()
                )));
            }
        }
        for page in batch.protected_pages() {
            if let Some(writer) = self.page_writers.get(page) {
                return Err(execution_error(format!(
                    "{} decoder transaction {} cannot read KV page {} while transaction {} mutates it",
                    self.error_context,
                    transaction.get(),
                    page.0,
                    writer.get()
                )));
            }
        }
        let mut topologies = HashSet::new();
        for sequence in batch.sequences() {
            let state = states.get(sequence.state_index()).ok_or_else(|| {
                execution_error(format!(
                    "{} decoder state slot {} is missing",
                    self.error_context,
                    sequence.state_index()
                ))
            })?;
            if !topologies.insert(state.topology_id()) {
                return Err(execution_error(format!(
                    "{} decoder transaction {} references topology {} more than once",
                    self.error_context,
                    transaction.get(),
                    state.topology_id().get()
                )));
            }
            if let Some(active) = self.sequence_owners.get(&state.topology_id()) {
                return Err(execution_error(format!(
                    "{} decoder sequence topology {} is already owned by transaction {}; transaction {} conflicts",
                    self.error_context,
                    state.topology_id().get(),
                    active.get(),
                    transaction.get()
                )));
            }
        }
        Ok(())
    }
    fn shell(&self, transaction: ExecutionTransactionId) -> Result<&DecoderTransaction<D>> {
        self.transactions.get(&transaction).ok_or_else(|| {
            execution_error(format!(
                "{} decoder transaction {} is not registered",
                self.error_context,
                transaction.get()
            ))
        })
    }
    fn shell_mut(
        &mut self,
        transaction: ExecutionTransactionId,
    ) -> Result<&mut DecoderTransaction<D>> {
        self.transactions.get_mut(&transaction).ok_or_else(|| {
            execution_error(format!(
                "{} decoder transaction {} is not registered",
                self.error_context,
                transaction.get()
            ))
        })
    }
    fn finish(&mut self, transaction: ExecutionTransactionId) {
        let Some(shell) = self.transactions.remove(&transaction) else {
            debug_assert!(false, "terminal decoder transaction lost its registry slot");
            return;
        };
        for owner in &shell.owners {
            if self.sequence_owners.get(&owner.topology_id) == Some(&transaction) {
                self.sequence_owners.remove(&owner.topology_id);
            }
        }
        for page in shell.batch.protected_pages() {
            if let Some(readers) = self.page_readers.get_mut(page) {
                readers.remove(&transaction);
                if readers.is_empty() {
                    self.page_readers.remove(page);
                }
            }
        }
        for page in shell
            .batch
            .new_pages()
            .iter()
            .copied()
            .chain(shell.batch.writable_pages().iter().copied())
            .chain(
                shell
                    .batch
                    .cow_replacements()
                    .iter()
                    .map(|cow| cow.replacement),
            )
        {
            if self.page_writers.get(&page) == Some(&transaction) {
                self.page_writers.remove(&page);
            }
        }
    }
}
fn consumed_kv_transaction_error(
    operation: &str,
    transaction: ExecutionTransactionId,
    source: Error,
) -> Error {
    Error::Context {
        operation: format!(
            "internal invariant: decoder transaction {} KV backend consumed its KV transaction before {operation} failed",
            transaction.get()
        ),
        source: Box::new(source),
    }
}
fn combined_error(context: &str, first: Error, second: Error) -> Error {
    Error::Execution {
        message: format!("{context}: primary={first}; cleanup={second}"),
    }
}
fn execution_error(message: impl Into<String>) -> Error {
    Error::Execution {
        message: message.into(),
    }
}
fn internal_error(message: impl Into<String>) -> Error {
    Error::Internal {
        message: message.into(),
    }
}
