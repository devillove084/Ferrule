use std::collections::HashMap;
use std::num::NonZeroU32;

use ferrule_common::execution::{ExecutionOutput, KvBindingMode, KvReservation, StateSlot};
use ferrule_common::{Error, Result};
use ferrule_model::MultiSessionRunner;
use tracing;

use crate::cache::{KvPageManager, PreemptedKvState, SequenceSlotPool};
use crate::generation::matched_stop;
use crate::scheduling::resident::{SuspendedSequenceSchedule, greedy_candidate};
use crate::scheduling::{
    CancelRequestResult, DecodeAction, GenerateRequest, RequestId, ResidentScheduler,
    ResidentSchedulerConfig, ScheduledBatch, SchedulerAction, SequenceFinishReason, SequenceState,
    SessionId,
};

use super::NativeMultiSessionExecutor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentTopKDriverConfig {
    pub ctx_size: usize,
    pub stop_at_eos: bool,
    pub append_eos_to_session: bool,
    /// Safety valve for `run_until_blocked` so an unhealthy backend cannot spin forever.
    pub max_steps_per_run: usize,
}

impl Default for ResidentTopKDriverConfig {
    fn default() -> Self {
        Self {
            ctx_size: 4096,
            stop_at_eos: true,
            append_eos_to_session: true,
            max_steps_per_run: 16_384,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidentTopKDriverStats {
    pub actions: usize,
    pub prefill_chunks: usize,
    pub prefill_tokens: usize,
    pub decode_steps: usize,
    pub emitted_tokens: usize,
    pub staged_tokens: usize,
    pub finished_sequences: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResidentTokenEvent {
    pub session_id: SessionId,
    pub request_id: Option<crate::scheduling::RequestId>,
    pub index: usize,
    pub token: u32,
    pub logit: Option<f32>,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidentDriverStep {
    /// No waiting, active, or ready work remains.
    Idle,
    /// Work exists but no action could be produced, usually because KV admission is blocked.
    Blocked,
    /// One scheduler action was executed and committed.
    Executed {
        action_kind: ResidentActionKind,
        rows: usize,
        staged: usize,
        finished: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentActionKind {
    Prefill,
    Decode,
    Mixed,
    Finish,
    Cancel,
}

/// Synchronous resident driver over scheduler + KV + native multi-session executor.
///
/// This is the end-to-end serving-shaped loop in runtime. It remains concrete
/// and synchronous: no async server, no trait-object framework, and no concrete
/// model ownership. The driver connects: request admission, chunked prefill,
/// decode, stop policy, token streaming, and KV/session finish lifecycle
/// through typed runtime values.
///
/// The driver requires `R: MultiSessionRunner`, so each sequence's state is
/// explicitly managed and swapped into the runner during execution.
struct SuspendedDriverSequence<S> {
    model_state: S,
    page_slot: StateSlot,
    kv_state: PreemptedKvState,
    schedule: SuspendedSequenceSchedule,
}

pub struct ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner,
    C: SequenceSlotPool,
{
    scheduler: ResidentScheduler,
    slot_pool: C,
    executor: NativeMultiSessionExecutor<R>,
    /// Per-session sequence states forked from the runner's default session.
    sequence_states: HashMap<SessionId, R::SequenceState>,
    /// Sessions explicitly retained across completed request turns, with their
    /// last committed logical position.
    retained_sessions: HashMap<SessionId, usize>,
    /// Default top-k used for batch lowering.
    top_k: NonZeroU32,
    page_manager: Option<KvPageManager>,
    page_slots: HashMap<SessionId, StateSlot>,
    suspended_sequences: HashMap<SessionId, SuspendedDriverSequence<R::SequenceState>>,
    next_page_slot: u32,
    config: ResidentTopKDriverConfig,
    stats: ResidentTopKDriverStats,
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner,
    C: SequenceSlotPool,
{
    pub fn new(runner: R, slot_pool: C) -> Self {
        Self::with_parts(
            ResidentScheduler::default(),
            slot_pool,
            NativeMultiSessionExecutor::new(runner),
            default_top_k(),
            ResidentTopKDriverConfig::default(),
        )
    }

    pub fn with_configs(
        runner: R,
        slot_pool: C,
        scheduler_config: ResidentSchedulerConfig,
        top_k: NonZeroU32,
        driver_config: ResidentTopKDriverConfig,
    ) -> Self {
        Self::with_parts(
            ResidentScheduler::new(scheduler_config),
            slot_pool,
            NativeMultiSessionExecutor::new(runner),
            top_k,
            driver_config,
        )
    }

    fn with_parts(
        scheduler: ResidentScheduler,
        slot_pool: C,
        executor: NativeMultiSessionExecutor<R>,
        top_k: NonZeroU32,
        config: ResidentTopKDriverConfig,
    ) -> Self {
        Self {
            scheduler,
            slot_pool,
            executor,
            sequence_states: HashMap::new(),
            retained_sessions: HashMap::new(),
            top_k,
            page_manager: None,
            page_slots: HashMap::new(),
            suspended_sequences: HashMap::new(),
            next_page_slot: 0,
            config,
            stats: ResidentTopKDriverStats::default(),
        }
    }

    pub fn scheduler(&self) -> &ResidentScheduler {
        &self.scheduler
    }

    pub fn slot_pool(&self) -> &C {
        &self.slot_pool
    }

    pub fn executor(&self) -> &NativeMultiSessionExecutor<R> {
        &self.executor
    }

    pub fn with_page_manager(mut self, page_manager: KvPageManager) -> Self {
        self.page_manager = Some(page_manager);
        self
    }

    /// Install the authoritative runtime page manager and configure a backend
    /// physical pool with the same bounded page capacity.
    pub fn try_with_page_manager(mut self, page_manager: KvPageManager) -> Result<Self> {
        let max_pages = page_manager.max_pages();
        if max_pages == 0 {
            return Err(Error::Execution(
                "a physical KV backend requires a bounded non-zero page capacity".into(),
            ));
        }
        self.executor.configure_kv_page_capacity(max_pages)?;
        self.page_manager = Some(page_manager);
        Ok(self)
    }

    pub fn page_manager(&self) -> Option<&KvPageManager> {
        self.page_manager.as_ref()
    }

    pub fn suspended_len(&self) -> usize {
        self.suspended_sequences.len()
    }

    /// Suspend one active session and move its exclusively owned physical pages
    /// out of backend device residency.
    pub fn preempt_session(&mut self, session_id: SessionId) -> Result<()> {
        if self.suspended_sequences.contains_key(&session_id) {
            return Err(Error::Execution(format!(
                "session {session_id:?} is already suspended"
            )));
        }
        if !self.sequence_states.contains_key(&session_id) {
            return Err(Error::Internal(format!(
                "session {session_id:?} has no model sequence state"
            )));
        }
        let schedule = self.scheduler.suspend_sequence(session_id)?;
        let slot = match self.page_slots.get(&session_id).copied() {
            Some(slot) => slot,
            None => {
                self.scheduler.restore_suspended(schedule)?;
                return Err(Error::Execution(format!(
                    "session {session_id:?} has no authoritative page slot"
                )));
            }
        };
        let kv_state = match self.page_manager.as_mut() {
            Some(manager) => match manager.preempt_sequence(slot) {
                Ok(state) => state,
                Err(error) => {
                    self.scheduler.restore_suspended(schedule)?;
                    return Err(error);
                }
            },
            None => {
                self.scheduler.restore_suspended(schedule)?;
                return Err(Error::Execution(
                    "session preemption requires an authoritative KvPageManager".into(),
                ));
            }
        };
        if let Err(error) = self.executor.preempt_kv_pages(kv_state.evicted_pages()) {
            self.page_manager
                .as_mut()
                .expect("checked above")
                .restore_sequence(slot, kv_state)?;
            self.scheduler.restore_suspended(schedule)?;
            return Err(error);
        }
        let model_state = self
            .sequence_states
            .remove(&session_id)
            .expect("model state was validated before preemption");
        self.page_slots.remove(&session_id);
        self.suspended_sequences.insert(
            session_id,
            SuspendedDriverSequence {
                model_state,
                page_slot: slot,
                kv_state,
                schedule,
            },
        );
        Ok(())
    }

    /// Restore a previously suspended session and its exact physical page contents.
    pub fn restore_session(&mut self, session_id: SessionId) -> Result<()> {
        let suspended = self
            .suspended_sequences
            .remove(&session_id)
            .ok_or_else(|| Error::Execution(format!("session {session_id:?} is not suspended")))?;
        if let Err(error) = self
            .executor
            .restore_kv_pages(suspended.kv_state.evicted_pages())
        {
            self.suspended_sequences.insert(session_id, suspended);
            return Err(error);
        }
        let page_restore = self
            .page_manager
            .as_mut()
            .ok_or_else(|| Error::Execution("KvPageManager was removed while suspended".into()))?
            .restore_sequence(suspended.page_slot, suspended.kv_state.clone());
        if let Err(error) = page_restore {
            let _ = self
                .executor
                .preempt_kv_pages(suspended.kv_state.evicted_pages());
            self.suspended_sequences.insert(session_id, suspended);
            return Err(error);
        }
        let schedule_backup = suspended.schedule.clone();
        if let Err(error) = self.scheduler.restore_suspended(suspended.schedule) {
            let kv_state = self
                .page_manager
                .as_mut()
                .expect("checked above")
                .preempt_sequence(suspended.page_slot)?;
            let _ = self.executor.preempt_kv_pages(kv_state.evicted_pages());
            self.suspended_sequences.insert(
                session_id,
                SuspendedDriverSequence {
                    model_state: suspended.model_state,
                    page_slot: suspended.page_slot,
                    kv_state,
                    schedule: schedule_backup,
                },
            );
            return Err(error);
        }
        self.page_slots.insert(session_id, suspended.page_slot);
        self.sequence_states
            .insert(session_id, suspended.model_state);
        Ok(())
    }

    pub fn into_runner(self) -> Result<R> {
        if !self.scheduler.is_idle()
            || !self.suspended_sequences.is_empty()
            || !self.sequence_states.is_empty()
        {
            return Err(Error::Execution(
                "cannot extract resident runner while session state is still retained or active"
                    .into(),
            ));
        }
        self.executor.into_runner()
    }

    /// Keep a session's model and KV state resident after each request finishes.
    /// Subsequent requests with this explicit session ID append at the last
    /// committed position instead of creating a fresh sequence.
    pub fn retain_session(&mut self, session_id: SessionId) -> Result<()> {
        if !self.scheduler.is_idle() || self.suspended_sequences.contains_key(&session_id) {
            return Err(Error::Execution(format!(
                "cannot retain session {session_id:?} while scheduler work is active or suspended"
            )));
        }
        self.retained_sessions.entry(session_id).or_insert(0);
        Ok(())
    }

    /// Return the committed position of an explicitly retained session.
    pub fn retained_session_position(&self, session_id: SessionId) -> Option<usize> {
        self.retained_sessions.get(&session_id).copied()
    }

    /// Release an idle retained session and all model/KV state owned by it.
    pub fn release_session(&mut self, session_id: SessionId) -> Result<()> {
        if !self.scheduler.is_idle() || self.suspended_sequences.contains_key(&session_id) {
            return Err(Error::Execution(format!(
                "cannot release session {session_id:?} while scheduler work is active or suspended"
            )));
        }
        self.retained_sessions.remove(&session_id);
        self.release_sequence_state(session_id);
        Ok(())
    }

    /// Reset an idle retained session to an empty position while preserving its
    /// retained lifecycle registration for future request turns.
    pub fn reset_session(&mut self, session_id: SessionId) -> Result<()> {
        if !self.retained_sessions.contains_key(&session_id) {
            return Err(Error::Execution(format!(
                "session {session_id:?} is not retained"
            )));
        }
        self.release_session(session_id)?;
        self.retained_sessions.insert(session_id, 0);
        Ok(())
    }

    pub fn stats(&self) -> &ResidentTopKDriverStats {
        &self.stats
    }

    /// Validate scheduler policy against the truthful capabilities of the native
    /// multi-session executor before any queue entry is consumed.
    pub fn validate_configuration(&self) -> Result<()> {
        let capabilities = self.executor.capabilities();
        let scheduler = self.scheduler.config();
        if scheduler.max_active_sequences > capabilities.max_sequences {
            return Err(Error::Execution(format!(
                "resident driver config allows {} active sequences, but its executor supports {}",
                scheduler.max_active_sequences, capabilities.max_sequences
            )));
        }
        if scheduler.max_decode_batch > capabilities.max_sequences {
            return Err(Error::Execution(format!(
                "resident driver config allows decode batch {}, but its executor supports {} sequence",
                scheduler.max_decode_batch, capabilities.max_sequences
            )));
        }
        if capabilities
            .max_top_k
            .is_some_and(|maximum| self.top_k > maximum)
        {
            return Err(Error::Execution(format!(
                "resident driver requests top-k {}, exceeding executor capability",
                self.top_k.get()
            )));
        }
        Ok(())
    }

    pub fn submit(&mut self, request: GenerateRequest) {
        let retained_position = request
            .session_id
            .and_then(|session_id| self.retained_sessions.get(&session_id).copied());
        if let Some(position) = retained_position {
            self.scheduler.submit_at_position(request, position);
        } else {
            self.scheduler.submit(request);
        }
    }

    /// Submit a request at a specific position. This is used for testing and
    /// for single-runner backends where the caller knows the runner's current
    /// position.
    pub fn submit_at_position(&mut self, request: GenerateRequest, position_start: usize) {
        self.scheduler.submit_at_position(request, position_start);
    }

    /// Cancel a waiting or active request without executing another model step.
    ///
    /// Active cancellation releases the scheduler slot, model sequence state,
    /// authoritative paged KV metadata, and physical KV pages. Cleanup failures
    /// are returned but do not poison the executor.
    pub fn cancel_request(&mut self, request_id: RequestId) -> Result<CancelRequestResult> {
        let result = self
            .scheduler
            .cancel_request(request_id, &mut self.slot_pool)?;
        if let CancelRequestResult::Active { session_id, .. } = result {
            if let Some(position) = self.retained_sessions.get_mut(&session_id) {
                *position = 0;
            }
            self.try_release_sequence_state(session_id)?;
        }
        Ok(result)
    }

    /// Fork an active session from exactly its currently committed paged prefix.
    ///
    /// `target_request.prompt_tokens` is the suffix for the target branch; the
    /// target starts at `expected_committed_position` and never re-executes the
    /// shared prefix. Scheduler, model, and page-table state are all prepared
    /// before any target becomes visible.
    pub fn fork_session_exact(
        &mut self,
        source_session_id: SessionId,
        target_request: GenerateRequest,
        expected_committed_position: usize,
    ) -> Result<SessionId> {
        if self.executor.is_poisoned() {
            return Err(Error::Execution(
                "cannot fork a session while the native executor is poisoned".into(),
            ));
        }
        let target_session_id = target_request.session_id.ok_or_else(|| {
            Error::Execution("exact fork target request requires an explicit session ID".into())
        })?;
        if self.suspended_sequences.contains_key(&source_session_id) {
            return Err(Error::Execution(
                "cannot fork from a suspended source session".into(),
            ));
        }
        if self.suspended_sequences.contains_key(&target_session_id)
            || self.sequence_states.contains_key(&target_session_id)
            || self.page_slots.contains_key(&target_session_id)
        {
            return Err(Error::Execution(format!(
                "fork target session {target_session_id:?} already exists"
            )));
        }
        let source_page_slot = *self.page_slots.get(&source_session_id).ok_or_else(|| {
            Error::Execution(format!(
                "fork source session {source_session_id:?} has no authoritative page slot"
            ))
        })?;
        let target_page_slot = StateSlot::new(self.next_page_slot);
        let next_page_slot = self.next_page_slot.checked_add(1).ok_or_else(|| {
            Error::Execution("driver page slot generation overflow during fork".into())
        })?;
        let kv_handle = self.slot_pool.alloc_slot()?;

        let prepared_schedule = match self.scheduler.prepare_fork_session_exact(
            source_session_id,
            target_session_id,
            &target_request,
            expected_committed_position,
            kv_handle,
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                let _ = self.slot_pool.free_slot(kv_handle);
                return Err(error);
            }
        };
        debug_assert_eq!(prepared_schedule.target_session_id(), target_session_id);

        let prepared_pages = match self.page_manager.as_ref() {
            Some(manager) => match manager.prepare_fork_sequence_exact(
                source_page_slot,
                target_page_slot,
                0,
                expected_committed_position,
            ) {
                Ok(prepared) => prepared,
                Err(error) => {
                    let _ = self.slot_pool.free_slot(kv_handle);
                    return Err(error);
                }
            },
            None => {
                let _ = self.slot_pool.free_slot(kv_handle);
                return Err(Error::Execution(
                    "exact-prefix fork requires an authoritative KvPageManager".into(),
                ));
            }
        };

        let prepared_model = {
            let source = self.sequence_states.get(&source_session_id).ok_or_else(|| {
                Error::Execution(format!(
                    "fork source session {source_session_id:?} has no model state"
                ))
            });
            match source.and_then(|source| {
                self.executor
                    .fork_sequence_state_from(source, expected_committed_position)
            }) {
                Ok(state) => state,
                Err(error) => {
                    let _ = self.slot_pool.free_slot(kv_handle);
                    return Err(error);
                }
            }
        };

        if let Err(error) = self
            .page_manager
            .as_mut()
            .expect("page manager was validated during fork prepare")
            .publish_fork_sequence_exact(prepared_pages)
        {
            let release_error = self.executor.release_sequence_state(prepared_model).err();
            let slot_error = self.slot_pool.free_slot(kv_handle).err();
            return match (release_error, slot_error) {
                (None, None) => Err(error),
                (release, slot) => Err(Error::Internal(format!(
                    "fork page publish failed ({error}); cleanup model={release:?}, slot={slot:?}"
                ))),
            };
        }

        self.scheduler.publish_fork_session_exact(prepared_schedule);
        let previous = self
            .sequence_states
            .insert(target_session_id, prepared_model);
        debug_assert!(
            previous.is_none(),
            "prepared model target must remain absent"
        );
        self.page_slots.insert(target_session_id, target_page_slot);
        self.next_page_slot = next_page_slot;
        Ok(target_session_id)
    }

    pub fn drain_finished(&mut self) -> Vec<SequenceState> {
        self.scheduler.drain_finished()
    }

    pub fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        self.scheduler.drain_cancelled()
    }

    pub fn drain_failed(&mut self) -> Vec<SequenceState> {
        self.scheduler.drain_failed()
    }

    /// Admit waiting sequences from the scheduler, creating a forked sequence
    /// state for each newly admitted session.
    fn admit_new_sequences(&mut self) -> Result<()> {
        // Don't admit new sequences if the executor is poisoned.
        if self.executor.is_poisoned() {
            return Ok(());
        }
        let old_active = self.scheduler.active_len();
        self.scheduler.admit_waiting(&mut self.slot_pool)?;
        let new_active = self.scheduler.active_len();
        if new_active > old_active {
            // Fork sequence states for newly admitted sessions.
            for session_id in self.scheduler.active_session_ids() {
                if !self.sequence_states.contains_key(&session_id) {
                    let state = self.executor.create_sequence_state()?;
                    self.sequence_states.insert(session_id, state);
                    if let Some(manager) = &mut self.page_manager {
                        let slot = StateSlot::new(self.next_page_slot);
                        self.next_page_slot =
                            self.next_page_slot.checked_add(1).ok_or_else(|| {
                                Error::Execution("driver page slot generation overflow".into())
                            })?;
                        if let Err(error) = manager.alloc_sequence(slot, 0) {
                            self.sequence_states.remove(&session_id);
                            let _ = self
                                .scheduler
                                .fail_sequence(session_id, &mut self.slot_pool);
                            return Err(error);
                        }
                        self.page_slots.insert(session_id, slot);
                    }
                }
            }
        }
        Ok(())
    }

    /// Preserve a retained session at its committed position, or release a
    /// normal one immediately after the request turn finishes.
    fn finalize_sequence_state(&mut self, session_id: SessionId, position: usize) {
        if let Some(retained_position) = self.retained_sessions.get_mut(&session_id) {
            *retained_position = position;
        } else {
            self.release_sequence_state(session_id);
        }
    }

    /// Release the sequence state for a finished or cancelled session.
    fn release_sequence_state(&mut self, session_id: SessionId) {
        if let Err(error) = self.try_release_sequence_state(session_id) {
            tracing::warn!("failed to release state for session {session_id:?}: {error}");
        }
    }

    /// Attempt every resource release and report cleanup errors without changing
    /// executor poison state. This is used directly by explicit cancellation.
    fn try_release_sequence_state(&mut self, session_id: SessionId) -> Result<()> {
        let mut errors = Vec::new();
        if let Some(slot) = self.page_slots.remove(&session_id) {
            match self.page_manager.as_mut() {
                Some(manager) => match manager.free_sequence_pages(slot) {
                    Ok(pages) => {
                        if let Err(error) = self.executor.release_kv_pages(&pages) {
                            errors.push(format!("physical KV release failed: {error}"));
                        }
                    }
                    Err(error) => errors.push(format!("paged KV release failed: {error}")),
                },
                None => errors.push("authoritative page manager is missing".into()),
            }
        }
        if let Some(state) = self.sequence_states.remove(&session_id) {
            if let Err(error) = self.executor.release_sequence_state(state) {
                errors.push(format!("model sequence-state release failed: {error}"));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(Error::Internal(format!(
                "failed to release cancelled session {session_id:?}: {}",
                errors.join("; ")
            )))
        }
    }

    fn feed_session_token(&mut self, session_id: SessionId, token_id: u32) -> Result<()> {
        let mut state = self.sequence_states.remove(&session_id).ok_or_else(|| {
            Error::Internal(format!(
                "cannot feed token for session {session_id:?} without sequence state"
            ))
        })?;
        let result = self.executor.feed_sequence_token(&mut state, token_id);
        self.sequence_states.insert(session_id, state);
        result
    }

    fn reserve_batch_pages(&mut self, batch: &ScheduledBatch) -> Result<Vec<KvReservation>> {
        let Some(manager) = &mut self.page_manager else {
            return Ok(Vec::new());
        };
        let mut reservations = Vec::with_capacity(batch.sequences.len());
        for (scheduled, execution) in batch.sequences.iter().zip(batch.execution().sequences()) {
            let slot = *self.page_slots.get(&scheduled.session_id).ok_or_else(|| {
                Error::Internal(format!(
                    "no page slot for active session {:?}",
                    scheduled.session_id
                ))
            })?;
            let token_count = usize::try_from(execution.query.end - execution.query.start)
                .map_err(|_| Error::Execution("query length exceeds usize".into()))?;
            match manager.reserve(slot, 0, token_count) {
                Ok(reservation) => reservations.push(reservation),
                Err(error) => {
                    for reservation in reservations.drain(..) {
                        let _ = manager.rollback(reservation);
                    }
                    return Err(error);
                }
            }
        }
        Ok(reservations)
    }

    fn bind_reserved_pages(
        &self,
        batch: &mut ScheduledBatch,
        reservations: &[KvReservation],
    ) -> Result<()> {
        if self.executor.capabilities().kv_binding_mode != KvBindingMode::Paged {
            return Ok(());
        }
        let manager = self.page_manager.as_ref().ok_or_else(|| {
            Error::Execution("paged executor requires a runtime KvPageManager".into())
        })?;
        let bindings = reservations
            .iter()
            .map(|reservation| manager.reservation_bindings(reservation))
            .collect::<Result<Vec<_>>>()?;
        batch.bind_paged_kv(&bindings)
    }

    fn rollback_page_reservations(&mut self, reservations: Vec<KvReservation>) {
        if let Some(manager) = &mut self.page_manager {
            for reservation in reservations {
                if let Err(error) = manager.rollback(reservation) {
                    tracing::warn!("failed to rollback KV reservation: {error}");
                }
            }
        }
    }

    fn commit_page_reservations(
        &mut self,
        reservations: Vec<KvReservation>,
    ) -> Result<Vec<ferrule_common::execution::KvPageId>> {
        match &mut self.page_manager {
            Some(manager) => manager.commit_batch_with_freed(reservations),
            None => Ok(Vec::new()),
        }
    }

    pub fn step<F>(&mut self, on_token: &mut F) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        self.validate_configuration()?;

        // If the executor is poisoned, reject all further work.
        if self.executor.is_poisoned() {
            return Err(Error::Execution(
                "native executor is poisoned; reset before executing again".into(),
            ));
        }

        // Admit waiting sequences before planning the next action. Each newly
        // admitted sequence gets a forked sequence state from the runner.
        self.admit_new_sequences()?;

        let Some(mut action) = self.scheduler.next_action(&mut self.slot_pool)? else {
            return Ok(if self.scheduler.is_idle() {
                ResidentDriverStep::Idle
            } else {
                ResidentDriverStep::Blocked
            });
        };

        let action_kind = action_kind(&action);
        let rows = action_rows(&action);
        let mut scheduled = match ScheduledBatch::from_action(&mut action, self.top_k) {
            Ok(scheduled) => scheduled,
            Err(error) => return Err(self.abort_action(&action, error, false, "batch lowering")),
        };

        let mut page_reservations = match scheduled.as_ref() {
            Some(batch) => match self.reserve_batch_pages(batch) {
                Ok(reservations) => reservations,
                Err(error) => {
                    return Err(self.abort_action(&action, error, false, "KV reserve"));
                }
            },
            None => Vec::new(),
        };
        if let Some(batch) = scheduled.as_mut() {
            if let Err(error) = self.bind_reserved_pages(batch, &page_reservations) {
                self.rollback_page_reservations(std::mem::take(&mut page_reservations));
                return Err(self.abort_action(&action, error, false, "KV binding"));
            }
        }

        // Collect the sequence states referenced by this batch into a dense
        // slice ordered by state slot index. The executor uses state_slot to
        // index into this slice.
        let output = match scheduled.as_ref() {
            Some(batch) => {
                let state_count = batch.sequences.len();
                // Collect the sequence states referenced by this batch into a
                // dense slice ordered by state slot index. The executor uses
                // state_slot to index into this slice.
                let mut states_flat: Vec<R::SequenceState> = Vec::with_capacity(state_count);
                let mut missing_session = None;
                for scheduled_seq in &batch.sequences {
                    match self.sequence_states.remove(&scheduled_seq.session_id) {
                        Some(state) => states_flat.push(state),
                        None => {
                            missing_session = Some(scheduled_seq.session_id);
                            break;
                        }
                    }
                }
                if let Some(session_id) = missing_session {
                    for (scheduled_seq, state) in batch.sequences.iter().zip(states_flat) {
                        self.sequence_states.insert(scheduled_seq.session_id, state);
                    }
                    self.rollback_page_reservations(std::mem::take(&mut page_reservations));
                    let error = Error::Internal(format!(
                        "sequence state disappeared for session {session_id:?}"
                    ));
                    return Err(self.abort_action(
                        &action,
                        error,
                        false,
                        "sequence state collection",
                    ));
                }

                let exec_result = self.executor.execute_batch_with_kv(
                    &mut states_flat,
                    batch.execution(),
                    &page_reservations,
                );

                // Move every state back in linear time, including after execution failure.
                debug_assert_eq!(batch.sequences.len(), states_flat.len());
                for (scheduled_seq, state) in batch.sequences.iter().zip(states_flat) {
                    self.sequence_states.insert(scheduled_seq.session_id, state);
                }

                match exec_result {
                    Ok(output) => Some(output),
                    Err(execution_error) => {
                        self.rollback_page_reservations(std::mem::take(&mut page_reservations));
                        let poisoned = self.executor.is_poisoned();
                        return Err(self.abort_action(
                            &action,
                            execution_error,
                            poisoned,
                            "model execution",
                        ));
                    }
                }
            }
            None => None,
        };

        if let (Some(batch), Some(output)) = (scheduled.as_ref(), output.as_ref()) {
            if let Err(contract_error) = batch.validate_output(output) {
                self.rollback_page_reservations(std::mem::take(&mut page_reservations));
                let _ = self.executor.rollback_prepared_batch();
                return Err(self.abort_action(
                    &action,
                    contract_error,
                    true,
                    "model output contract",
                ));
            }
        } else if scheduled.is_some() != output.is_some() {
            self.rollback_page_reservations(std::mem::take(&mut page_reservations));
            let _ = self.executor.rollback_prepared_batch();
            let error =
                Error::Internal("scheduled execution and model output presence diverged".into());
            return Err(self.abort_action(&action, error, true, "model output presence"));
        }

        let freed_pages = match self
            .commit_page_reservations(std::mem::take(&mut page_reservations))
        {
            Ok(pages) => pages,
            Err(error) => {
                let rollback_error = self.executor.rollback_prepared_batch().err();
                let error = match rollback_error {
                    Some(rollback) => Error::Internal(format!(
                        "runtime KV commit failed ({error}); backend rollback also failed ({rollback})"
                    )),
                    None => error,
                };
                self.executor.poison("KV commit", &error);
                return Err(self.abort_action(&action, error, true, "KV commit"));
            }
        };
        if let Err(error) = self.executor.commit_prepared_batch() {
            return Err(self.abort_action(&action, error, true, "backend KV commit"));
        }
        if let Err(error) = self.executor.release_kv_pages(&freed_pages) {
            return Err(self.abort_action(&action, error, true, "backend KV release"));
        }

        let terminal_action = match &action {
            SchedulerAction::Finish { session_id, .. } => self
                .scheduler
                .active_sequence(*session_id)
                .map(|sequence| (*session_id, sequence.position, true)),
            SchedulerAction::Cancel { session_id, .. } => self
                .scheduler
                .active_sequence(*session_id)
                .map(|sequence| (*session_id, sequence.position, false)),
            _ => None,
        };
        if let Err(commit_error) = self.scheduler.commit_action(&action) {
            return Err(self.abort_action(&action, commit_error, true, "runtime commit"));
        }
        if let Some((session_id, position, retain_on_finish)) = terminal_action {
            if retain_on_finish {
                self.finalize_sequence_state(session_id, position);
            } else {
                self.release_sequence_state(session_id);
            }
        }
        self.stats.actions += 1;
        match &action {
            SchedulerAction::Execute { prefills, decodes } => {
                self.stats.prefill_chunks += prefills.len();
                self.stats.prefill_tokens += prefills
                    .iter()
                    .map(|action| action.token_range.len())
                    .sum::<usize>();
                self.stats.decode_steps += decodes.len();
                if let Err(error) = self.emit_committed_decode_tokens(decodes, on_token) {
                    return Err(self.abort_action(&action, error, true, "token emission"));
                }
            }
            SchedulerAction::PrefillChunk(prefill) => {
                self.stats.prefill_chunks += 1;
                self.stats.prefill_tokens += prefill.token_range.len();
            }
            SchedulerAction::DecodeBatch(actions) => {
                self.stats.decode_steps += actions.len();
                if let Err(error) = self.emit_committed_decode_tokens(actions, on_token) {
                    return Err(self.abort_action(&action, error, true, "token emission"));
                }
            }
            SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => {}
        }

        let action_finish = match self.finish_after_decode_action(&action) {
            Ok(outcome) => outcome,
            Err(error) => {
                return Err(self.abort_action(&action, error, true, "sequence finish"));
            }
        };
        let mut finished = action_finish.finished;
        let staged = match (scheduled.as_ref(), output.as_ref()) {
            (Some(batch), Some(output)) => {
                let outcome =
                    match self.apply_execution_output(batch, output, &action_finish.session_ids) {
                        Ok(outcome) => outcome,
                        Err(error) => {
                            return Err(self.abort_action(
                                &action,
                                error,
                                true,
                                "output application",
                            ));
                        }
                    };
                finished += outcome.finished;
                outcome.staged
            }
            (None, None) => 0,
            _ => unreachable!("scheduled/output presence was validated before commit"),
        };

        Ok(ResidentDriverStep::Executed {
            action_kind,
            rows,
            staged,
            finished,
        })
    }

    fn abort_action(
        &mut self,
        action: &SchedulerAction,
        error: Error,
        poison_executor: bool,
        stage: &'static str,
    ) -> Error {
        if poison_executor && !self.executor.is_poisoned() {
            self.executor.poison(stage, &error);
        }
        // Collect session IDs from the action before failing.
        let session_ids: Vec<SessionId> = match action {
            SchedulerAction::Execute { prefills, decodes } => prefills
                .iter()
                .map(|action| action.session_id)
                .chain(decodes.iter().map(|action| action.session_id))
                .collect(),
            SchedulerAction::PrefillChunk(prefill) => vec![prefill.session_id],
            SchedulerAction::DecodeBatch(actions) => actions.iter().map(|a| a.session_id).collect(),
            SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => Vec::new(),
        };
        match self.scheduler.fail_action(action, &mut self.slot_pool) {
            Ok(_) => {
                for session_id in &session_ids {
                    self.release_sequence_state(*session_id);
                }
                error
            }
            Err(cleanup_error) => {
                for session_id in &session_ids {
                    self.release_sequence_state(*session_id);
                }
                Error::Internal(format!(
                    "{stage} failed ({error}); error-state cleanup also failed ({cleanup_error})"
                ))
            }
        }
    }

    pub fn run_until_blocked<F>(&mut self, mut on_token: F) -> Result<ResidentTopKDriverStats>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        for _ in 0..self.config.max_steps_per_run {
            match self.step(&mut on_token)? {
                ResidentDriverStep::Executed { .. } => {}
                ResidentDriverStep::Idle | ResidentDriverStep::Blocked => {
                    return Ok(self.stats.clone());
                }
            }
        }
        Err(Error::Internal(format!(
            "resident top-k driver exceeded max_steps_per_run={} without becoming idle or blocked",
            self.config.max_steps_per_run
        )))
    }

    fn emit_committed_decode_tokens<F>(
        &mut self,
        actions: &[DecodeAction],
        on_token: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        for action in actions {
            let runner = self.executor.runner();
            let sequence = self
                .scheduler
                .active_sequence_mut(action.session_id)
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "cannot emit token for inactive session {:?}",
                        action.session_id
                    ))
                })?;
            let text = runner
                .decode_incremental(action.token_id, &mut sequence.incremental_decode)?
                .unwrap_or_default();
            sequence.append_generated_text(&text);
            let index = sequence.generated.saturating_sub(1);
            let event = ResidentTokenEvent {
                session_id: sequence.session_id,
                request_id: sequence.request_id,
                index,
                token: action.token_id,
                logit: action.logit,
                text,
            };
            self.stats.emitted_tokens += 1;
            on_token(&event)?;
        }
        Ok(())
    }

    fn finish_after_decode_action(
        &mut self,
        action: &SchedulerAction,
    ) -> Result<ActionFinishOutcome> {
        let actions: &[DecodeAction] = match action {
            SchedulerAction::Execute { decodes, .. } => decodes,
            SchedulerAction::DecodeBatch(actions) => actions,
            _ => return Ok(ActionFinishOutcome::default()),
        };

        let mut outcome = ActionFinishOutcome::default();
        for action in actions {
            let Some(sequence) = self.scheduler.active_sequence(action.session_id) else {
                continue;
            };
            let reason = if sequence.generated >= sequence.max_new_tokens {
                Some(SequenceFinishReason::MaxTokens)
            } else if matched_stop(&sequence.generated_text, &sequence.stop).is_some() {
                Some(SequenceFinishReason::StopString)
            } else {
                None
            };
            if let Some(reason) = reason {
                let position = sequence.position;
                self.scheduler
                    .finish_sequence(action.session_id, reason, &mut self.slot_pool)?;
                self.finalize_sequence_state(action.session_id, position);
                self.stats.finished_sequences += 1;
                outcome.finished += 1;
                outcome.session_ids.push(action.session_id);
            }
        }
        Ok(outcome)
    }

    fn apply_execution_output(
        &mut self,
        scheduled: &ScheduledBatch,
        output: &ExecutionOutput,
        action_finished_sessions: &[SessionId],
    ) -> Result<OutputOutcome> {
        let mut outcome = OutputOutcome::default();
        for row in &output.logits {
            let correlation = scheduled
                .sequence_for_input_row(row.input_row)
                .copied()
                .ok_or_else(|| {
                    Error::Execution(format!(
                        "output input row {} has no scheduled sequence",
                        row.input_row
                    ))
                })?;
            let execution_sequence = scheduled
                .execution()
                .sequences()
                .iter()
                .find(|sequence| sequence.query.contains(&row.input_row))
                .ok_or_else(|| {
                    Error::Execution(format!(
                        "output input row {} has no execution sequence span",
                        row.input_row
                    ))
                })?;
            let session_id = correlation.session_id;
            let Some(sequence) = self.scheduler.active_sequence(session_id) else {
                if action_finished_sessions.contains(&session_id) {
                    // The just-committed token ended the sequence (for example via
                    // a stop string), so its already-computed next-token logits are
                    // intentionally discarded after successful correlation.
                    continue;
                }
                return Err(Error::Execution(format!(
                    "output for input row {} references inactive session {:?}",
                    row.input_row, session_id
                )));
            };
            if sequence.request_id != correlation.request_id {
                return Err(Error::Execution(format!(
                    "output correlation request mismatch for session {:?}: active {:?}, scheduled {:?}",
                    session_id, sequence.request_id, correlation.request_id
                )));
            }
            if sequence.kv_handle != correlation.kv_handle {
                return Err(Error::Execution(format!(
                    "output correlation KV mismatch for session {:?}: active {:?}, scheduled {:?}",
                    session_id, sequence.kv_handle, correlation.kv_handle
                )));
            }
            if sequence.position != execution_sequence.sequence_len as usize {
                return Err(Error::Execution(format!(
                    "output correlation position mismatch for session {:?}: active {}, executed {}",
                    session_id, sequence.position, execution_sequence.sequence_len
                )));
            }
            if sequence.generated >= sequence.max_new_tokens {
                let position = sequence.position;
                self.scheduler.finish_sequence(
                    session_id,
                    SequenceFinishReason::MaxTokens,
                    &mut self.slot_pool,
                )?;
                self.finalize_sequence_state(session_id, position);
                self.stats.finished_sequences += 1;
                outcome.finished += 1;
                continue;
            }
            if sequence.position >= self.config.ctx_size {
                let position = sequence.position;
                self.scheduler.finish_sequence(
                    session_id,
                    SequenceFinishReason::Context,
                    &mut self.slot_pool,
                )?;
                self.finalize_sequence_state(session_id, position);
                self.stats.finished_sequences += 1;
                outcome.finished += 1;
                continue;
            }

            let Some(candidate) = greedy_candidate(&row.logits) else {
                let position = sequence.position;
                self.scheduler.finish_sequence(
                    session_id,
                    SequenceFinishReason::NoCandidate,
                    &mut self.slot_pool,
                )?;
                self.finalize_sequence_state(session_id, position);
                self.stats.finished_sequences += 1;
                outcome.finished += 1;
                continue;
            };

            if self.config.stop_at_eos
                && !sequence.ignore_eos
                && self.executor.runner().eos_token_id() == Some(candidate.token_id)
            {
                if self.config.append_eos_to_session {
                    if let Err(execution_error) =
                        self.feed_session_token(session_id, candidate.token_id)
                    {
                        if let Err(cleanup_error) = self
                            .scheduler
                            .fail_sequence(session_id, &mut self.slot_pool)
                        {
                            return Err(Error::Internal(format!(
                                "EOS state update failed ({execution_error}); error-state cleanup also failed ({cleanup_error})"
                            )));
                        }
                        return Err(execution_error);
                    }
                    if let Some(sequence) = self.scheduler.active_sequence_mut(session_id) {
                        sequence.advance_position(1);
                    }
                }
                let position = self
                    .scheduler
                    .active_sequence(session_id)
                    .map_or(0, |sequence| sequence.position);
                self.scheduler.finish_sequence(
                    session_id,
                    SequenceFinishReason::Eos,
                    &mut self.slot_pool,
                )?;
                self.finalize_sequence_state(session_id, position);
                self.stats.finished_sequences += 1;
                outcome.finished += 1;
                continue;
            }

            self.scheduler.stage_decode_candidate(
                session_id,
                candidate.token_id,
                Some(candidate.logit),
            )?;
            self.stats.staged_tokens += 1;
            outcome.staged += 1;
        }
        Ok(outcome)
    }
}

#[derive(Default)]
struct ActionFinishOutcome {
    finished: usize,
    session_ids: Vec<SessionId>,
}

#[derive(Default)]
struct OutputOutcome {
    staged: usize,
    finished: usize,
}

fn default_top_k() -> NonZeroU32 {
    NonZeroU32::new(1).expect("1 is non-zero")
}

fn action_kind(action: &SchedulerAction) -> ResidentActionKind {
    match action {
        SchedulerAction::Execute { prefills, decodes } => {
            match (prefills.is_empty(), decodes.is_empty()) {
                (false, false) => ResidentActionKind::Mixed,
                (false, true) => ResidentActionKind::Prefill,
                (true, false) => ResidentActionKind::Decode,
                (true, true) => ResidentActionKind::Mixed,
            }
        }
        SchedulerAction::PrefillChunk(_) => ResidentActionKind::Prefill,
        SchedulerAction::DecodeBatch(_) => ResidentActionKind::Decode,
        SchedulerAction::Finish { .. } => ResidentActionKind::Finish,
        SchedulerAction::Cancel { .. } => ResidentActionKind::Cancel,
    }
}

fn action_rows(action: &SchedulerAction) -> usize {
    match action {
        SchedulerAction::Execute { prefills, decodes } => prefills
            .iter()
            .map(|action| action.token_range.len())
            .sum::<usize>()
            .saturating_add(decodes.len()),
        SchedulerAction::PrefillChunk(prefill) => prefill.token_range.len(),
        SchedulerAction::DecodeBatch(actions) => actions.len(),
        SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => 0,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use ferrule_common::execution::{KvLayoutSchema, KvPlaneDescriptor};
    use ferrule_common::{Error, Result};
    use ferrule_model::{
        ModelInfo, ModelRunner, MultiSessionRunner, PrefillMode, TokenLogit, TopKModelRunner,
    };

    use crate::cache::{FixedSequenceSlotPool, PagedSequenceKvCache};
    use crate::sampling::SamplingConfig;
    use crate::scheduling::{RequestId, SequenceStatus};

    use super::*;

    #[derive(Debug)]
    struct DriverTestKvSchema;

    static DRIVER_TEST_PLANE: KvPlaneDescriptor = KvPlaneDescriptor {
        name: "test",
        elements_per_token: 1,
        layer_count: 1,
    };

    impl KvLayoutSchema for DriverTestKvSchema {
        fn planes(&self) -> &[KvPlaneDescriptor] {
            std::slice::from_ref(&DRIVER_TEST_PLANE)
        }

        fn page_size(&self) -> usize {
            4
        }

        fn max_sequence_len(&self) -> usize {
            64
        }
    }

    #[derive(Debug)]
    struct MockTopKRunner {
        position: usize,
        eos: Option<u32>,
        outputs: VecDeque<Vec<TokenLogit>>,
        fed: Vec<u32>,
        prefills: Vec<Vec<u32>>,
        fail_next_mutation: bool,
        mutation_calls: usize,
        released_sequence_states: usize,
        released_kv_pages: Vec<ferrule_common::execution::KvPageId>,
    }

    impl MockTopKRunner {
        fn new(outputs: Vec<Vec<TokenLogit>>) -> Self {
            Self {
                position: 0,
                eos: None,
                outputs: outputs.into(),
                fed: Vec::new(),
                prefills: Vec::new(),
                fail_next_mutation: false,
                mutation_calls: 0,
                released_sequence_states: 0,
                released_kv_pages: Vec::new(),
            }
        }

        fn failing_next_mutation(mut self) -> Self {
            self.fail_next_mutation = true;
            self
        }

        fn complete_mutation<T>(&mut self, value: T) -> Result<T> {
            self.mutation_calls += 1;
            if std::mem::take(&mut self.fail_next_mutation) {
                Err(Error::Model(
                    "simulated failure after partial runner mutation".into(),
                ))
            } else {
                Ok(value)
            }
        }

        fn with_eos(mut self, eos: u32) -> Self {
            self.eos = Some(eos);
            self
        }

        fn next_output(&mut self) -> Vec<TokenLogit> {
            self.outputs.pop_front().unwrap_or_default()
        }
    }

    impl ModelRunner for MockTopKRunner {
        fn model_info(&self) -> ModelInfo {
            ModelInfo {
                family: ferrule_model::ModelFamily::Unknown("mock".into()),
                architecture: Some("mock".into()),
                attention: ferrule_model::AttentionKind::Unknown("mock".into()),
                weight_source: ferrule_model::WeightSource::Unknown,
                hidden_size: 1,
                num_layers: 1,
                num_experts: 0,
                num_experts_per_tok: 0,
                vocab_size: 256,
                backend: "mock",
            }
        }

        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(&self, tokens: &[u32]) -> Result<String> {
            Ok(tokens
                .iter()
                .map(|token| char::from_u32(*token).unwrap_or('?'))
                .collect())
        }

        fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
            self.position += tokens.len();
            Ok(vec![0.0])
        }

        fn decode_token(&mut self, token: u32) -> Result<Vec<f32>> {
            self.fed.push(token);
            self.position += 1;
            Ok(vec![0.0])
        }

        fn reset_session(&mut self) -> Result<()> {
            self.position = 0;
            self.fed.clear();
            self.prefills.clear();
            Ok(())
        }

        fn eos_token_id(&self) -> Option<u32> {
            self.eos
        }
    }

    impl TopKModelRunner for MockTopKRunner {
        fn position(&self) -> usize {
            self.position
        }

        fn feed_token(&mut self, token_id: u32) -> Result<()> {
            self.fed.push(token_id);
            self.position += 1;
            self.complete_mutation(())
        }

        fn prefill_topk(
            &mut self,
            token_ids: &[u32],
            _top_k: usize,
            _mode: PrefillMode,
        ) -> Result<Vec<TokenLogit>> {
            self.prefills.push(token_ids.to_vec());
            self.position += token_ids.len();
            let output = self.next_output();
            self.complete_mutation(output)
        }

        fn decode_topk(&mut self, token_id: u32, _top_k: usize) -> Result<Vec<TokenLogit>> {
            self.feed_token(token_id)?;
            Ok(self.next_output())
        }
    }

    /// Per-sequence state for the mock runner. Tracks position and fed tokens.
    #[derive(Debug)]
    struct MockSequenceState {
        position: usize,
        fed: Vec<u32>,
        prefills: Vec<Vec<u32>>,
        outputs: VecDeque<Vec<TokenLogit>>,
        fail_next_mutation: bool,
        mutation_calls: usize,
    }

    impl MockSequenceState {
        fn new(position: usize, outputs: &VecDeque<Vec<TokenLogit>>) -> Self {
            Self {
                position,
                fed: Vec::new(),
                prefills: Vec::new(),
                outputs: outputs.clone(),
                fail_next_mutation: false,
                mutation_calls: 0,
            }
        }
    }

    impl MultiSessionRunner for MockTopKRunner {
        type SequenceState = MockSequenceState;

        fn with_sequence_state<T>(
            &mut self,
            state: &mut Self::SequenceState,
            execute: impl FnOnce(&mut Self) -> Result<T>,
        ) -> Result<T> {
            // Swap position, outputs, fail flag, and mutation_calls between
            // the runner and the state.
            let saved_position = self.position;
            let saved_outputs = std::mem::take(&mut self.outputs);
            let saved_fed = std::mem::take(&mut self.fed);
            let saved_fail = self.fail_next_mutation;
            let saved_calls = self.mutation_calls;

            self.position = state.position;
            self.outputs = std::mem::take(&mut state.outputs);
            self.fed = std::mem::take(&mut state.fed);
            self.fail_next_mutation = state.fail_next_mutation;
            self.mutation_calls = state.mutation_calls;

            let result = execute(self);

            // Swap back, preserving any state changes the runner made.
            state.position = self.position;
            state.outputs = std::mem::take(&mut self.outputs);
            state.fed = std::mem::take(&mut self.fed);
            state.fail_next_mutation = self.fail_next_mutation;
            state.mutation_calls = self.mutation_calls;
            state.prefills.extend(self.prefills.drain(..));

            self.position = saved_position;
            self.outputs = saved_outputs;
            self.fed = saved_fed;
            self.fail_next_mutation = saved_fail;
            self.mutation_calls = saved_calls;

            result
        }

        fn fork_sequence_state(&mut self) -> Result<Self::SequenceState> {
            let mut state = MockSequenceState::new(0, &self.outputs);
            state.fail_next_mutation = self.fail_next_mutation;
            Ok(state)
        }

        fn fork_sequence_state_from(
            &mut self,
            source: &Self::SequenceState,
            expected_position: usize,
        ) -> Result<Self::SequenceState> {
            if source.position != expected_position {
                return Err(Error::Execution(format!(
                    "mock fork expected position {expected_position}, source is at {}",
                    source.position
                )));
            }
            if source.fail_next_mutation {
                return Err(Error::Model("simulated model fork prepare failure".into()));
            }
            Ok(MockSequenceState {
                position: source.position,
                fed: source.fed.clone(),
                prefills: Vec::new(),
                outputs: source.outputs.clone(),
                fail_next_mutation: source.fail_next_mutation,
                mutation_calls: source.mutation_calls,
            })
        }

        fn reset_sequence_state(&mut self, state: &mut Self::SequenceState) -> Result<()> {
            state.position = 0;
            state.fed.clear();
            state.prefills.clear();
            state.mutation_calls = 0;
            Ok(())
        }

        fn release_sequence_state(&mut self, _state: Self::SequenceState) -> Result<()> {
            self.released_sequence_states += 1;
            Ok(())
        }

        fn release_kv_pages(
            &mut self,
            pages: &[ferrule_common::execution::KvPageId],
        ) -> Result<()> {
            self.released_kv_pages.extend_from_slice(pages);
            Ok(())
        }

        fn multi_session_capabilities(&self) -> ferrule_common::execution::ExecutionCapabilities {
            ferrule_common::execution::ExecutionCapabilities {
                max_batch_tokens: 1024,
                max_sequences: 4,
                max_prefill_query_tokens_per_sequence: 1024,
                max_decode_query_tokens_per_sequence: 1,
                max_top_k: NonZeroU32::new(40),
                supports_prefill: true,
                supports_decode: true,
                supports_mixed: true,
                full_logits_width: None,
                kv_binding_mode: ferrule_common::execution::KvBindingMode::None,
                logits_row_policy: ferrule_common::execution::LogitsRowPolicy::LastPerSequence,
            }
        }
    }

    fn top(token_id: u32) -> Vec<TokenLogit> {
        vec![TokenLogit {
            token_id,
            logit: token_id as f32,
        }]
    }

    fn request(
        id: u64,
        prompt: &[u32],
        max_new_tokens: usize,
        stop: Vec<String>,
    ) -> GenerateRequest {
        GenerateRequest {
            id: RequestId(id),
            session_id: None,
            prompt_tokens: prompt.to_vec(),
            sampling: SamplingConfig::greedy(),
            max_new_tokens,
            stop,
            ignore_eos: false,
        }
    }

    fn driver_from_runner(
        runner: MockTopKRunner,
    ) -> ResidentTopKDriver<MockTopKRunner, PagedSequenceKvCache> {
        ResidentTopKDriver::with_configs(
            runner,
            PagedSequenceKvCache::new(1, 1, 8, 1),
            ResidentSchedulerConfig {
                prefill_chunk_size: 2,
                max_active_sequences: 1,
                max_decode_batch: 1,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig {
                ctx_size: 16,
                stop_at_eos: true,
                append_eos_to_session: true,
                max_steps_per_run: 64,
            },
        )
    }

    fn driver_with_outputs(
        outputs: Vec<Vec<TokenLogit>>,
    ) -> ResidentTopKDriver<MockTopKRunner, PagedSequenceKvCache> {
        driver_from_runner(MockTopKRunner::new(outputs))
    }

    fn batched_driver_from_runner(
        runner: MockTopKRunner,
    ) -> ResidentTopKDriver<MockTopKRunner, PagedSequenceKvCache> {
        ResidentTopKDriver::with_configs(
            runner,
            PagedSequenceKvCache::new(1, 1, 8, 2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 2,
                max_decode_batch: 2,
                max_batch_tokens: 16,
                allow_mixed_batches: true,
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
    }

    #[test]
    fn driver_runs_request_to_max_tokens_and_frees_kv() {
        let mut driver = driver_with_outputs(vec![top(b'a' as u32), top(b'b' as u32)]);
        driver.submit(request(1, &[1, 2], 2, Vec::new()));
        let mut events = Vec::new();
        let stats = driver
            .run_until_blocked(|event| {
                events.push(event.clone());
                Ok(())
            })
            .unwrap();

        assert_eq!(
            events.iter().map(|event| event.token).collect::<Vec<_>>(),
            vec![b'a' as u32, b'b' as u32]
        );
        assert_eq!(
            events
                .iter()
                .map(|event| event.text.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert_eq!(stats.prefill_chunks, 1);
        assert_eq!(stats.prefill_tokens, 2);
        assert_eq!(stats.decode_steps, 2);
        assert_eq!(stats.emitted_tokens, 2);
        assert_eq!(stats.finished_sequences, 1);
        assert_eq!(driver.slot_pool().active_count(), 0);

        let finished = driver.drain_finished();
        assert_eq!(finished.len(), 1);
        assert_eq!(
            finished[0].finish_reason,
            Some(SequenceFinishReason::MaxTokens)
        );
        assert_eq!(finished[0].generated_text, "ab");
        assert_eq!(finished[0].status, SequenceStatus::Finished);
    }

    #[test]
    fn retained_session_continues_across_turns_and_can_reset() {
        let session_id = SessionId(42);
        let mut driver = driver_with_outputs(vec![top(b'a' as u32), top(b'b' as u32)]);
        driver.retain_session(session_id).unwrap();

        let mut first = request(10, &[1], 1, Vec::new());
        first.session_id = Some(session_id);
        driver.submit(first);
        driver.run_until_blocked(|_| Ok(())).unwrap();
        assert_eq!(driver.retained_session_position(session_id), Some(2));
        let _ = driver.drain_finished();

        let mut second = request(11, &[2], 1, Vec::new());
        second.session_id = Some(session_id);
        driver.submit(second);
        let mut events = Vec::new();
        driver
            .run_until_blocked(|event| {
                events.push(event.token);
                Ok(())
            })
            .unwrap();
        assert_eq!(events, vec![b'b' as u32]);
        assert_eq!(driver.retained_session_position(session_id), Some(4));
        let _ = driver.drain_finished();

        driver.reset_session(session_id).unwrap();
        assert_eq!(driver.retained_session_position(session_id), Some(0));
        assert_eq!(driver.slot_pool().active_count(), 0);

        driver.release_session(session_id).unwrap();
        assert_eq!(driver.retained_session_position(session_id), None);
    }

    #[test]
    fn driver_stops_on_stop_string_after_committed_token() {
        let mut driver =
            driver_with_outputs(vec![top(b'a' as u32), top(b'b' as u32), top(b'c' as u32)]);
        driver.submit(request(2, &[9], 8, vec!["ab".into()]));
        let mut text = String::new();
        driver
            .run_until_blocked(|event| {
                text.push_str(&event.text);
                Ok(())
            })
            .unwrap();

        let finished = driver.drain_finished();
        assert_eq!(text, "ab");
        assert_eq!(
            finished[0].finish_reason,
            Some(SequenceFinishReason::StopString)
        );
        assert_eq!(finished[0].generated_text, "ab");
    }

    #[test]
    fn driver_finishes_on_eos_without_emitting_token() {
        let runner = MockTopKRunner::new(vec![top(2)]).with_eos(2);
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            PagedSequenceKvCache::new(1, 1, 4, 1),
            ResidentSchedulerConfig::default(),
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        );
        driver.submit(request(3, &[1], 4, Vec::new()));
        let mut events = Vec::new();
        driver
            .run_until_blocked(|event| {
                events.push(event.clone());
                Ok(())
            })
            .unwrap();

        assert!(events.is_empty());
        assert!(
            driver.executor().runner().fed.is_empty(),
            "EOS must be applied to the explicit session state, not the runner default"
        );
        let finished = driver.drain_finished();
        assert_eq!(finished[0].position, 2);
        assert_eq!(finished[0].finish_reason, Some(SequenceFinishReason::Eos));
        assert_eq!(finished[0].tokens, vec![1]);
    }

    #[test]
    fn mixed_requests_isolate_ignore_eos_policy() {
        let eos = 2;
        let mut driver = ResidentTopKDriver::with_configs(
            MockTopKRunner::new(vec![top(eos)]).with_eos(eos),
            PagedSequenceKvCache::new(1, 1, 4, 2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 4,
                max_active_sequences: 2,
                max_decode_batch: 2,
                max_batch_tokens: 8,
                allow_mixed_batches: true,
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        );
        let stop_at_eos = request(40, &[1], 1, Vec::new());
        let mut ignore_eos = request(41, &[3], 1, Vec::new());
        ignore_eos.ignore_eos = true;
        driver.submit(stop_at_eos);
        driver.submit(ignore_eos);

        let mut events = Vec::new();
        driver
            .run_until_blocked(|event| {
                events.push((event.request_id, event.token));
                Ok(())
            })
            .unwrap();

        assert_eq!(events, vec![(Some(RequestId(41)), eos)]);
        let finished = driver.drain_finished();
        let stopped = finished
            .iter()
            .find(|sequence| sequence.request_id == Some(RequestId(40)))
            .unwrap();
        let ignored = finished
            .iter()
            .find(|sequence| sequence.request_id == Some(RequestId(41)))
            .unwrap();
        assert_eq!(stopped.finish_reason, Some(SequenceFinishReason::Eos));
        assert!(!stopped.ignore_eos);
        assert_eq!(stopped.tokens, vec![1]);
        assert_eq!(ignored.finish_reason, Some(SequenceFinishReason::MaxTokens));
        assert!(ignored.ignore_eos);
        assert_eq!(ignored.tokens, vec![3, eos]);
    }

    #[test]
    fn final_decode_skips_next_logits() {
        let mut driver = driver_with_outputs(vec![top(b'a' as u32)]);
        driver.submit(request(4, &[1], 1, Vec::new()));
        driver.run_until_blocked(|_| Ok(())).unwrap();

        let finished = driver.drain_finished();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].position, 2);
        assert_eq!(
            finished[0].finish_reason,
            Some(SequenceFinishReason::MaxTokens)
        );
    }

    #[test]
    fn max_new_zero_finishes_after_prefill() {
        let mut driver = driver_with_outputs(vec![top(b'a' as u32)]);
        driver.submit(request(5, &[1], 0, Vec::new()));
        driver.run_until_blocked(|_| Ok(())).unwrap();

        let finished = driver.drain_finished();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].position, 1);
        assert_eq!(
            finished[0].finish_reason,
            Some(SequenceFinishReason::MaxTokens)
        );
    }

    #[test]
    fn submit_submits_fresh_sessions_and_finishes() {
        let mut driver = driver_with_outputs(vec![top(b'a' as u32), top(b'b' as u32)]);
        driver.submit(request(7, &[1], 1, Vec::new()));
        driver.run_until_blocked(|_| Ok(())).unwrap();
        let first = driver.drain_finished();
        assert_eq!(first[0].position, 2);

        driver.submit(request(8, &[2], 1, Vec::new()));
        driver.run_until_blocked(|_| Ok(())).unwrap();
        let second = driver.drain_finished();
        assert_eq!(second[0].prompt_tokens_for_range(0..1).unwrap(), &[2]);
        assert_eq!(second[0].position, 2);
    }

    #[test]
    fn into_runner_allows_clean_driver_rebuild_after_warmup() {
        let mut driver = driver_with_outputs(vec![top(b'w' as u32), top(b'm' as u32)]);
        driver.submit(request(9, &[1], 1, Vec::new()));
        driver.run_until_blocked(|_| Ok(())).unwrap();
        let warmup = driver.drain_finished();
        assert_eq!(warmup[0].position, 2);

        let runner = driver.into_runner().unwrap();
        let mut driver = driver_from_runner(runner);
        driver.submit(request(10, &[2], 1, Vec::new()));
        driver.run_until_blocked(|_| Ok(())).unwrap();
        let measured = driver.drain_finished();

        assert_eq!(measured[0].prompt_tokens_for_range(0..1).unwrap(), &[2]);
        assert_eq!(measured[0].position, 2);
    }

    #[test]
    fn driver_moves_partially_executed_sequence_to_error_state() {
        let runner = MockTopKRunner::new(vec![top(b'a' as u32)]).failing_next_mutation();
        let mut driver = driver_from_runner(runner);
        driver.submit(request(11, &[1, 2], 1, Vec::new()));

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(format!("{error}").contains("simulated failure"));
        assert!(driver.executor().is_poisoned());
        assert_eq!(driver.scheduler().active_len(), 0);
        assert_eq!(driver.scheduler().failed_len(), 1);
        assert_eq!(driver.slot_pool().active_count(), 0);

        let failed = driver.drain_failed();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].status, SequenceStatus::Error);
        assert_eq!(failed[0].position, 0, "runtime metadata was not committed");
        assert_eq!(failed[0].kv_handle, None);

        driver.submit(request(12, &[3], 1, Vec::new()));
        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(format!("{error}").contains("native executor is poisoned"));
        // The second request was never admitted because the executor is poisoned.
        assert_eq!(driver.scheduler().waiting_len(), 1);
    }

    #[test]
    fn batch_execution_writes_back_every_sequence_state_on_success() {
        let mut driver = batched_driver_from_runner(MockTopKRunner::new(vec![top(b'a' as u32)]));
        driver.submit(request(50, &[1], 2, Vec::new()));
        driver.submit(request(51, &[2], 2, Vec::new()));

        let step = driver.step(&mut |_| Ok(())).unwrap();
        assert!(matches!(step, ResidentDriverStep::Executed { rows: 2, .. }));
        assert_eq!(driver.sequence_states.len(), 2);
        assert_eq!(driver.sequence_states[&SessionId(1)].position, 1);
        assert_eq!(driver.sequence_states[&SessionId(2)].position, 1);
    }

    #[test]
    fn batch_execution_writes_back_every_sequence_state_on_failure() {
        let runner = MockTopKRunner::new(vec![top(b'a' as u32)]).failing_next_mutation();
        let mut driver = batched_driver_from_runner(runner);
        driver.submit(request(52, &[1], 2, Vec::new()));
        driver.submit(request(53, &[2], 2, Vec::new()));

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(format!("{error}").contains("simulated failure"));
        assert_eq!(driver.scheduler().failed_len(), 2);
        assert!(driver.sequence_states.is_empty());
        assert_eq!(driver.executor().runner().released_sequence_states, 2);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn lowering_failure_moves_dequeued_sequence_to_failed_state() {
        let mut runner = MockTopKRunner::new(Vec::new());
        runner.position = u32::MAX as usize + 1;
        let mut driver = driver_from_runner(runner);
        // Submit at the overflow position to trigger a lowering failure.
        driver.submit_at_position(request(13, &[1], 1, Vec::new()), u32::MAX as usize + 1);

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(format!("{error}").contains("neutral u32 ABI"));
        assert!(!driver.executor().is_poisoned());
        assert_eq!(driver.scheduler().active_len(), 0);
        assert_eq!(driver.scheduler().failed_len(), 1);
        assert_eq!(driver.slot_pool().active_count(), 0);
        assert_eq!(driver.executor().runner().mutation_calls, 0);
    }

    #[test]
    fn callback_failure_poison_fails_committed_sequence_and_blocks_runner_extraction() {
        let mut driver = driver_with_outputs(vec![top(b'a' as u32), top(b'b' as u32)]);
        driver.submit(request(14, &[1], 3, Vec::new()));

        let error = driver
            .run_until_blocked(|_| Err(Error::Internal("simulated callback failure".into())))
            .unwrap_err();
        assert!(format!("{error}").contains("callback failure"));
        assert!(driver.executor().is_poisoned());
        assert_eq!(driver.scheduler().active_len(), 0);
        assert_eq!(driver.scheduler().failed_len(), 1);
        assert_eq!(driver.slot_pool().active_count(), 0);
        assert!(driver.into_runner().is_err());
    }

    #[test]
    fn driver_rejects_exceeding_executor_capabilities_before_consuming_work() {
        // The mock executor supports max_sequences=4. Set max_active_sequences=5
        // to trigger a validation failure before any work is consumed.
        let mut driver = ResidentTopKDriver::with_configs(
            MockTopKRunner::new(Vec::new()),
            PagedSequenceKvCache::new(1, 1, 4, 2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 2,
                max_active_sequences: 5,
                max_decode_batch: 2,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        );
        driver.submit(request(13, &[1, 2], 1, Vec::new()));

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(format!("{error}").contains("allows 5 active sequences"));
        assert_eq!(driver.scheduler().waiting_len(), 1);
        assert_eq!(driver.scheduler().active_len(), 0);
        assert_eq!(driver.slot_pool().active_count(), 0);
    }

    #[test]
    fn driver_cancel_request_reports_waiting_and_unknown() {
        let mut driver = driver_with_outputs(Vec::new());
        driver.submit(request(18, &[1], 1, Vec::new()));

        assert_eq!(
            driver.cancel_request(RequestId(18)).unwrap(),
            CancelRequestResult::Waiting {
                request_id: RequestId(18),
                session_id: SessionId(1),
            }
        );
        assert_eq!(
            driver.cancel_request(RequestId(99)).unwrap(),
            CancelRequestResult::NotFound {
                request_id: RequestId(99),
            }
        );
        assert!(!driver.executor().is_poisoned());
        let cancelled = driver.drain_cancelled();
        assert_eq!(cancelled.len(), 1);
        assert_eq!(cancelled[0].request_id, Some(RequestId(18)));
        assert_eq!(cancelled[0].status, SequenceStatus::Cancelled);
    }

    #[test]
    fn driver_active_cancel_releases_all_resources_and_allows_followup() {
        let manager = KvPageManager::new(Box::new(DriverTestKvSchema), 16);
        let mut driver = driver_with_outputs(vec![top(b'a' as u32), top(b'b' as u32)])
            .with_page_manager(manager);
        driver.submit(request(19, &[1, 2, 3], 2, Vec::new()));
        driver.step(&mut |_| Ok(())).unwrap();

        assert_eq!(driver.scheduler().active_len(), 1);
        assert_eq!(driver.slot_pool().active_count(), 1);
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 1);
        assert!(driver.page_manager().unwrap().allocated_pages() > 0);
        assert!(driver.sequence_states.contains_key(&SessionId(1)));

        assert_eq!(
            driver.cancel_request(RequestId(19)).unwrap(),
            CancelRequestResult::Active {
                request_id: RequestId(19),
                session_id: SessionId(1),
            }
        );
        assert_eq!(driver.scheduler().active_len(), 0);
        assert_eq!(driver.slot_pool().active_count(), 0);
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 0);
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert!(!driver.page_slots.contains_key(&SessionId(1)));
        assert_eq!(driver.executor().runner().released_sequence_states, 1);
        assert_eq!(driver.executor().runner().released_kv_pages.len(), 1);
        assert!(!driver.executor().is_poisoned());

        let cancelled = driver.drain_cancelled();
        assert_eq!(cancelled.len(), 1);
        assert_eq!(cancelled[0].request_id, Some(RequestId(19)));
        assert_eq!(cancelled[0].status, SequenceStatus::Cancelled);

        driver.submit(request(20, &[9], 1, Vec::new()));
        let mut events = Vec::new();
        driver
            .run_until_blocked(|event| {
                events.push(event.token);
                Ok(())
            })
            .unwrap();
        assert_eq!(events, vec![b'a' as u32]);
        let finished = driver.drain_finished();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].request_id, Some(RequestId(20)));
        assert_eq!(driver.slot_pool().active_count(), 0);
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 0);
        assert!(!driver.executor().is_poisoned());
    }

    #[test]
    fn driver_page_manager_reserves_commits_and_releases_with_sequence() {
        let manager = KvPageManager::new(Box::new(DriverTestKvSchema), 16);
        let mut driver = driver_with_outputs(vec![top(b'a' as u32)]).with_page_manager(manager);
        driver.submit(request(20, &[1, 2, 3], 1, Vec::new()));
        driver.run_until_blocked(|_| Ok(())).unwrap();
        let finished = driver.drain_finished();
        assert_eq!(finished.len(), 1);
        let manager = driver.page_manager().unwrap();
        assert_eq!(manager.active_sequences(), 0);
        assert_eq!(manager.allocated_pages(), 0);
    }

    #[test]
    fn driver_preempt_restore_preserves_sequence_and_continues_exactly() {
        let manager = KvPageManager::new(Box::new(DriverTestKvSchema), 16);
        let mut driver = driver_with_outputs(vec![top(b'a' as u32), top(b'b' as u32)])
            .with_page_manager(manager);
        driver.submit(request(21, &[1, 2], 2, Vec::new()));

        let first = driver.step(&mut |_| Ok(())).unwrap();
        assert!(matches!(first, ResidentDriverStep::Executed { .. }));
        let session_id = SessionId(1);
        let before = driver
            .page_manager()
            .unwrap()
            .block_table(StateSlot::new(0))
            .unwrap()
            .clone();

        driver.preempt_session(session_id).unwrap();
        assert_eq!(driver.suspended_len(), 1);
        assert_eq!(driver.scheduler().active_len(), 0);
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 0);

        driver.restore_session(session_id).unwrap();
        assert_eq!(driver.suspended_len(), 0);
        assert_eq!(driver.scheduler().active_len(), 1);
        let after = driver
            .page_manager()
            .unwrap()
            .block_table(StateSlot::new(0))
            .unwrap();
        assert_eq!(after.pages(), before.pages());
        assert_eq!(after.committed_tokens(), before.committed_tokens());

        let mut emitted = Vec::new();
        driver
            .run_until_blocked(|event| {
                emitted.push(event.token);
                Ok(())
            })
            .unwrap();
        assert_eq!(emitted, vec![b'a' as u32, b'b' as u32]);
        let finished = driver.drain_finished();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].generated_text, "ab");
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
    }

    fn fork_driver() -> ResidentTopKDriver<MockTopKRunner, PagedSequenceKvCache> {
        ResidentTopKDriver::with_configs(
            MockTopKRunner::new(vec![top(b'a' as u32), top(b'b' as u32)]),
            PagedSequenceKvCache::new(1, 1, 8, 2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 2,
                max_decode_batch: 2,
                max_batch_tokens: 8,
                allow_mixed_batches: true,
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16))
    }

    #[test]
    fn exact_fork_executes_only_suffix_and_clears_source_candidate() {
        let mut driver = fork_driver();
        driver.submit(request(30, &[1, 2, 3], 4, Vec::new()));
        driver.step(&mut |_| Ok(())).unwrap();
        let source = SessionId(1);
        assert_eq!(
            driver
                .scheduler()
                .active_sequence(source)
                .unwrap()
                .next_decode_token,
            Some(b'a' as u32)
        );

        let mut target_request = request(31, &[9, 10], 2, Vec::new());
        target_request.session_id = Some(SessionId(2));
        let target = driver
            .fork_session_exact(source, target_request, 3)
            .unwrap();
        assert_eq!(target, SessionId(2));
        assert_eq!(
            driver
                .scheduler()
                .active_sequence(source)
                .unwrap()
                .next_decode_token,
            None
        );
        let target_schedule = driver.scheduler().active_sequence(target).unwrap();
        assert_eq!(target_schedule.position, 3);
        assert_eq!(target_schedule.remaining_prompt_tokens(), 2);
        let source_page = driver
            .page_manager()
            .unwrap()
            .block_table(StateSlot::new(0))
            .unwrap()
            .pages()[0];
        assert_eq!(
            driver
                .page_manager()
                .unwrap()
                .block_table(StateSlot::new(1))
                .unwrap()
                .pages()[0],
            source_page
        );

        driver.step(&mut |_| Ok(())).unwrap();
        let target_model = driver.sequence_states.get(&target).unwrap();
        assert_eq!(target_model.prefills, vec![vec![9, 10]]);
        assert_eq!(target_model.position, 5);
        assert_ne!(
            driver
                .page_manager()
                .unwrap()
                .block_table(StateSlot::new(1))
                .unwrap()
                .pages()[0],
            source_page,
            "partial shared tail append must publish runtime COW"
        );
        assert_eq!(
            driver
                .page_manager()
                .unwrap()
                .block_table(StateSlot::new(0))
                .unwrap()
                .pages()[0],
            source_page
        );
    }

    #[test]
    fn exact_fork_prepare_failure_leaves_source_and_target_unchanged() {
        let mut driver = fork_driver();
        driver.submit(request(32, &[1, 2, 3], 4, Vec::new()));
        driver.step(&mut |_| Ok(())).unwrap();
        let source = SessionId(1);
        let source_candidate = driver
            .scheduler()
            .active_sequence(source)
            .unwrap()
            .next_decode_token;
        let source_pages = driver
            .page_manager()
            .unwrap()
            .block_table(StateSlot::new(0))
            .unwrap()
            .pages()
            .to_vec();

        let mut target_request = request(33, &[9], 1, Vec::new());
        target_request.session_id = Some(SessionId(2));
        let error = driver
            .fork_session_exact(source, target_request, 2)
            .unwrap_err();
        assert!(error.to_string().contains("expected committed position"));
        assert_eq!(driver.scheduler().active_len(), 1);
        assert!(driver.scheduler().active_sequence(SessionId(2)).is_none());
        assert!(!driver.sequence_states.contains_key(&SessionId(2)));
        assert_eq!(driver.slot_pool().active_count(), 1);
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 1);
        assert_eq!(
            driver
                .scheduler()
                .active_sequence(source)
                .unwrap()
                .next_decode_token,
            source_candidate
        );
        assert_eq!(
            driver
                .page_manager()
                .unwrap()
                .block_table(StateSlot::new(0))
                .unwrap()
                .pages(),
            source_pages
        );
        assert!(
            source_pages
                .iter()
                .all(|page| driver.page_manager().unwrap().page_refcount(*page) == 1)
        );
    }

    #[test]
    fn exact_fork_model_prepare_failure_rolls_back_all_provisional_state() {
        let mut driver = fork_driver();
        driver.submit(request(34, &[1, 2, 3], 4, Vec::new()));
        driver.step(&mut |_| Ok(())).unwrap();
        let source = SessionId(1);
        driver
            .sequence_states
            .get_mut(&source)
            .unwrap()
            .fail_next_mutation = true;
        let source_candidate = driver
            .scheduler()
            .active_sequence(source)
            .unwrap()
            .next_decode_token;
        let source_page = driver
            .page_manager()
            .unwrap()
            .block_table(StateSlot::new(0))
            .unwrap()
            .pages()[0];

        let mut target_request = request(35, &[9], 1, Vec::new());
        target_request.session_id = Some(SessionId(2));
        let error = driver
            .fork_session_exact(source, target_request, 3)
            .unwrap_err();
        assert!(error.to_string().contains("model fork prepare failure"));
        assert_eq!(driver.scheduler().active_len(), 1);
        assert!(driver.scheduler().active_sequence(SessionId(2)).is_none());
        assert!(!driver.sequence_states.contains_key(&SessionId(2)));
        assert_eq!(driver.slot_pool().active_count(), 1);
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 1);
        assert_eq!(driver.page_manager().unwrap().page_refcount(source_page), 1);
        assert_eq!(
            driver
                .scheduler()
                .active_sequence(source)
                .unwrap()
                .next_decode_token,
            source_candidate
        );
    }

    #[test]
    fn driver_blocks_when_kv_cannot_admit_waiting_request() {
        let mut driver = ResidentTopKDriver::with_configs(
            MockTopKRunner::new(Vec::new()),
            FixedSequenceSlotPool::new(0),
            ResidentSchedulerConfig::default(),
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        );
        driver.submit(request(6, &[1], 1, Vec::new()));
        let step = driver.step(&mut |_| Ok(())).unwrap();
        assert_eq!(step, ResidentDriverStep::Blocked);
        assert_eq!(driver.scheduler().waiting_len(), 1);
    }
}
