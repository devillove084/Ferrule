//! Driver-local trace observation. It observes committed values but never owns
//! scheduler, session, KV, prefix, or physical custody.
use super::*;
use crate::engine::kv_trace::{
    LookupOutcome, RemovalReason, RuntimeKvRecorder, RuntimeKvTraceConfig,
    RuntimeKvTraceKind as Kind, RuntimeKvTraceSnapshot, TerminalStatus,
};

/// Narrow prefix-side observation seam. Prefix lifecycle state never stores a
/// recorder or exposes its maps; the driver remains the sole observer owner.
pub(super) trait PrefixTraceObserver {
    fn enabled(&self) -> bool;
    fn prefix_entry_removed(
        &mut self,
        tokens: &[u32],
        charge_pages: usize,
        generation: u64,
        reason: RemovalReason,
    );
    fn prefix_entry_published(
        &mut self,
        entry: u64,
        tokens: &[u32],
        charge_pages: usize,
        generation: u64,
        replaced: bool,
    );
    fn prefix_unpin_ack(&mut self, entry: u64);
}

/// Owns only observational correlation and bounded recording, never execution
/// authority. Callers can report values, but cannot borrow the recorder or maps.
#[derive(Default)]
pub(super) struct TraceObserver {
    recorder: Option<RuntimeKvRecorder>,
    disabled_logged: bool,
    overflow_logged: bool,
}

impl TraceObserver {
    fn install(&mut self, recorder: RuntimeKvRecorder) {
        self.recorder = Some(recorder);
        self.disabled_logged = false;
        self.overflow_logged = false;
    }

    fn note_disabled(&mut self, reason: &'static str) {
        if !self.disabled_logged {
            tracing::debug!(
                component = "runtime_kv_trace",
                state = "disabled",
                reason,
                "driver trace observation is disabled"
            );
            self.disabled_logged = true;
        }
    }

    fn note_snapshot(&mut self, snapshot: &RuntimeKvTraceSnapshot) {
        if snapshot.status.overflow && !self.overflow_logged {
            tracing::warn!(
                component = "runtime_kv_trace",
                state = "overflow",
                reason = "recorder_overflow",
                events = snapshot.usage.events,
                observations = snapshot.usage.observations,
                token_bytes = snapshot.usage.token_bytes,
                "driver trace observation overflowed"
            );
            self.overflow_logged = true;
        }
        // The recorder returns a final snapshot only once; it owns that state.
        if snapshot.finalized {
            tracing::debug!(
                component = "runtime_kv_trace",
                state = "finalized",
                reason = if snapshot.complete {
                    "complete"
                } else {
                    "incomplete"
                },
                complete = snapshot.complete,
                events = snapshot.usage.events,
                observations = snapshot.usage.observations,
                token_bytes = snapshot.usage.token_bytes,
                "driver trace observation finalized"
            );
        }
    }

    fn drain(&mut self, finalized: bool) -> Option<RuntimeKvTraceSnapshot> {
        let Some(recorder) = self.recorder.as_mut() else {
            self.note_disabled(if finalized { "finalize" } else { "drain" });
            return None;
        };
        let snapshot = recorder.drain(finalized)?;
        // Observe recorder status only at export boundaries, without a second
        // hot-path budget or an extra drain that would change trace watermarks.
        self.note_snapshot(&snapshot);
        Some(snapshot)
    }

    fn finalize(&mut self, live: bool) -> Option<RuntimeKvTraceSnapshot> {
        if live {
            self.unsupported();
        }
        self.drain(true)
    }

    fn unsupported(&mut self) {
        if let Some(recorder) = &mut self.recorder {
            recorder.unsupported();
        }
    }

    fn observation(&mut self, session: u64) -> Option<u64> {
        self.recorder.as_mut()?.observation(session)
    }

    fn record(&mut self, kind: Kind) {
        if let Some(recorder) = &mut self.recorder {
            recorder.record(kind);
        }
    }

    fn record_tokens(
        &mut self,
        tokens: &[u32],
        make: impl FnOnce(crate::engine::kv_trace::TokenSpan) -> Kind,
    ) {
        if let Some(recorder) = &mut self.recorder {
            recorder.record_tokens(tokens, make);
        }
    }

    fn demand(&mut self, request: u64, session: u64, tokens: &[u32]) {
        if let Some(recorder) = &mut self.recorder {
            recorder.demand(request, session, tokens);
        }
    }

    fn admitted(&mut self, session: u64, reused: usize, page_size: Option<usize>, bypass: bool) {
        let Some(observation) = self.observation(session) else {
            return;
        };
        if bypass {
            self.record(Kind::Lookup {
                observation: Some(observation),
                outcome: LookupOutcome::Bypass,
                entry: None,
                matched_tokens: 0,
                generation: 0,
            });
        }
        self.record(Kind::AdmissionPublished {
            observation,
            actual_tokens: reused,
            actual_pages: reused.div_ceil(page_size.expect("traced admission has a page manager")),
        });
    }

    fn terminal(&mut self, request: u64, status: TerminalStatus) {
        if let Some(recorder) = &mut self.recorder {
            recorder.terminal(request, status);
        }
    }

    pub(super) fn cancellation(&mut self, request: u64) {
        if let Some(recorder) = &mut self.recorder {
            recorder.cancellation(request);
        }
    }

    fn reservation(
        &mut self,
        session: Option<u64>,
        reservation: u64,
        allocated_pages: usize,
        cow: Option<ferrule_common::execution::KvCowReplacement>,
    ) {
        let observation = session.and_then(|session| self.observation(session));
        if observation.is_none() {
            self.unsupported();
        }
        self.record(Kind::ReserveSucceeded {
            observation,
            reservation,
            allocated_pages,
            cow: cow.is_some(),
        });
        if let Some(cow) = cow {
            self.record(Kind::CowReplacement {
                reservation,
                logical_page: cow.logical_page,
                source: cow.source.0,
                replacement: cow.replacement.0,
            });
        }
    }

    fn input(&mut self, session: u64, reservation: u64, start: usize, tokens: &[u32]) {
        if let Some(recorder) = &mut self.recorder {
            recorder.input(session, reservation, start, tokens);
        }
    }

    fn finish_execution(&mut self, session: u64, reservation: u64, committed: Option<usize>) {
        if let Some(recorder) = &mut self.recorder {
            recorder.finish_execution(session, reservation, committed);
        }
    }

    pub(super) fn lookup(
        &mut self,
        session: u64,
        outcome: LookupOutcome,
        entry: Option<u64>,
        matched_tokens: usize,
        generation: u64,
    ) {
        let observation = self.observation(session);
        self.record(Kind::Lookup {
            observation,
            outcome,
            entry,
            matched_tokens,
            generation,
        });
    }

    pub(super) fn snapshot_captured(
        &mut self,
        snapshot: u64,
        committed_tokens: usize,
        pages: usize,
    ) {
        self.record(Kind::SnapshotCaptured {
            snapshot,
            committed_tokens,
            pages,
        });
    }

    pub(super) fn snapshot_released(&mut self, snapshot: u64, retiring_pages: usize) {
        self.record(Kind::SnapshotReleased {
            snapshot,
            retiring_pages,
        });
    }

    pub(super) fn snapshot_fork_published(&mut self, snapshot: u64, session: u64) {
        let Some(observation) = self.observation(session) else {
            return;
        };
        self.record(Kind::SnapshotForkPublished {
            snapshot,
            observation,
        });
    }

    pub(super) fn backend_release_ack(&mut self, pages: usize) {
        if pages != 0 {
            self.record(Kind::BackendReleaseAck { pages });
        }
    }

    pub(super) fn retirement_confirmed(&mut self, pages: usize) {
        if pages != 0 {
            self.record(Kind::RetirementConfirmed { pages });
        }
    }
}

impl PrefixTraceObserver for TraceObserver {
    fn enabled(&self) -> bool {
        self.recorder.is_some()
    }

    fn prefix_entry_removed(
        &mut self,
        tokens: &[u32],
        charge_pages: usize,
        generation: u64,
        reason: RemovalReason,
    ) {
        self.record_tokens(tokens, |tokens| Kind::PrefixEntryRemoved {
            tokens,
            charge_pages,
            generation,
            reason,
        });
    }

    fn prefix_entry_published(
        &mut self,
        entry: u64,
        tokens: &[u32],
        charge_pages: usize,
        generation: u64,
        replaced: bool,
    ) {
        self.record_tokens(tokens, |tokens| Kind::PrefixEntryPublished {
            entry,
            tokens,
            charge_pages,
            generation,
            replaced,
        });
    }

    fn prefix_unpin_ack(&mut self, entry: u64) {
        self.record(Kind::UnpinAck { entry });
    }
}

impl<R: MultiSessionRunner, C: SequenceSlotPool> ResidentTopKDriver<R, C> {
    /// Enable before the first admission on a fresh owner. Buffers are opt-in.
    pub fn enable_kv_trace(&mut self, config: RuntimeKvTraceConfig) -> Result<()> {
        if self.admission_started
            || self.trace.enabled()
            || self.next_admission_session != Some(1)
            || !self.request_identities.is_empty()
            || !self.prefix.is_empty()
            || self
                .page_manager
                .as_ref()
                .is_none_or(|m| m.allocated_pages() != 0)
        {
            return Err(Error::InvalidRequest {
                message: "KV trace requires a fresh paged owner and may only be enabled once"
                    .into(),
            });
        }
        let Some(manager) = self.page_manager.as_ref() else {
            return Err(Error::InvalidRequest {
                message: "KV tracing requires an authoritative page manager".into(),
            });
        };
        let placement = self.materialization_resolver.placement();
        let plan = self.executor.runner().prefix_cache_plan_identity();
        let mut recorder = RuntimeKvRecorder::new(
            config,
            manager.page_size(),
            self.prefix.capacity(),
            [
                placement.model().get(),
                placement.backend().get(),
                placement.device().get(),
                plan,
                manager.owner_identity(),
            ],
        )
        .map_err(|e| Error::InvalidRequest {
            message: e.to_string(),
        })?;
        if plan == 0 {
            recorder.unsupported();
        }
        self.trace.install(recorder);
        Ok(())
    }

    pub fn drain_kv_trace(&mut self) -> Option<RuntimeKvTraceSnapshot> {
        self.trace.drain(false)
    }

    pub fn finalize_kv_trace(&mut self) -> Option<RuntimeKvTraceSnapshot> {
        let live = self.has_live_transactions()
            || !self.kv.pending_abort_empty()
            || !self.kv.pending_retirements_empty()
            || !self.prefix.pending_cleanups_empty()
            || self.sessions.cleanup_count() != 0;
        self.trace.finalize(live)
    }

    pub(super) fn trace_unsupported(&mut self) {
        self.trace.unsupported();
    }

    pub(super) fn trace_demand(
        &mut self,
        request: &GenerateRequest,
        session: SessionId,
        position: usize,
    ) {
        if position != 0 {
            self.trace.unsupported();
        }
        self.trace
            .demand(request.id.0, session.0, &request.prompt_tokens);
    }

    pub(super) fn trace_admitted(&mut self, session: SessionId, reused: usize, bypass: bool) {
        if !self.trace.enabled() {
            return;
        }
        let page_size = self.page_manager.as_ref().map(KvPageManager::page_size);
        self.trace.admitted(session.0, reused, page_size, bypass);
    }

    pub(super) fn trace_terminal(&mut self, sequence: &SequenceState, status: TerminalStatus) {
        if let Some(request) = sequence.request_id {
            self.trace.terminal(request.0, status);
        } else {
            self.trace.unsupported();
        }
    }

    pub(super) fn trace_reservation(&mut self, reservation: &KvReservation) {
        if !self.trace.enabled() {
            return;
        }
        let session = self.sessions.page_slot_ids().find(|session| {
            self.sessions.page_slot(session) == Some(&reservation.view().state_slot)
        });
        self.trace.reservation(
            session.map(|session| session.0),
            reservation.id().get(),
            reservation.view().newly_allocated.len(),
            reservation.view().cow_replacement,
        );
    }

    pub(super) fn trace_batch_input(
        &mut self,
        scheduled: &ScheduledBatch,
        transaction: ExecutionTransactionId,
    ) {
        if !self.trace.enabled() {
            return;
        }
        for (sequence, execution) in scheduled
            .sequences
            .iter()
            .zip(scheduled.execution().sequences())
        {
            let range = execution.query.start as usize..execution.query.end as usize;
            self.trace.input(
                sequence.session_id.0,
                transaction.get(),
                execution.context_len as usize,
                &scheduled.execution().token_ids()[range],
            );
        }
    }

    pub(super) fn trace_speculative_input(
        &mut self,
        actions: &[DecodeAction],
        batch: &ferrule_common::execution::ExecutionBatch,
        transaction: ExecutionTransactionId,
    ) {
        if !self.trace.enabled() {
            return;
        }
        for (action, execution) in actions.iter().zip(batch.sequences()) {
            let range = execution.query.start as usize..execution.query.end as usize;
            self.trace.input(
                action.session_id.0,
                transaction.get(),
                execution.context_len as usize,
                &batch.token_ids()[range],
            );
        }
    }

    pub(super) fn trace_speculative_end(
        &mut self,
        actions: &[DecodeAction],
        transaction: ExecutionTransactionId,
        publish: bool,
    ) {
        if !self.trace.enabled() {
            return;
        }
        for action in actions {
            let frontier = if publish {
                self.sessions
                    .page_slot(&action.session_id)
                    .and_then(|slot| self.page_manager.as_ref()?.block_table(*slot))
                    .map(|b| b.committed_tokens())
            } else {
                None
            };
            if publish && frontier.is_none() {
                self.trace.unsupported();
            }
            self.trace
                .finish_execution(action.session_id.0, transaction.get(), frontier);
        }
    }

    pub(super) fn trace_action_end(
        &mut self,
        action: &SchedulerAction,
        transaction: ExecutionTransactionId,
        publish: bool,
    ) {
        if !self.trace.enabled() {
            return;
        }
        for_each_input(action, |session, _, _| {
            let frontier = if publish {
                self.sessions
                    .page_slot(&session)
                    .and_then(|slot| self.page_manager.as_ref()?.block_table(*slot))
                    .map(|b| b.committed_tokens())
            } else {
                None
            };
            if publish && frontier.is_none() {
                self.trace.unsupported();
            }
            self.trace
                .finish_execution(session.0, transaction.get(), frontier);
        });
    }
}

fn for_each_input(action: &SchedulerAction, mut visit: impl FnMut(SessionId, usize, &[u32])) {
    match action {
        SchedulerAction::Execute { prefills, decodes } => {
            for p in prefills {
                visit(p.session_id, p.position_start, &p.tokens);
            }
            for d in decodes {
                visit(d.session_id, d.position, std::slice::from_ref(&d.token_id));
            }
        }
        SchedulerAction::PrefillChunk(p) => visit(p.session_id, p.position_start, &p.tokens),
        SchedulerAction::DecodeBatch(ds) => {
            for d in ds {
                visit(d.session_id, d.position, std::slice::from_ref(&d.token_id));
            }
        }
        _ => {}
    }
}
