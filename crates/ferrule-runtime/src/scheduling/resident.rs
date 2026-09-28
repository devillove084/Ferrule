//! Resident request/session scheduler.
//!
//! This module ties together the runtime vocabulary that was previously adjacent
//! but not connected: `GenerateRequest`, `SequenceState`, `SequenceSlotPool`,
//! `SchedulerAction`, neutral execution lowering, and runtime output correlation.
//!
//! It is intentionally synchronous and single-process. It does not execute a
//! model and does not know concrete model families; it only decides which
//! resident sequence should prefill/decode next and owns slot allocation lifecycle.

use std::collections::{HashMap, VecDeque};

use ahash::RandomState;

use crate::{Error, Result};
use ferrule_common::execution::{LogitsOutput, TokenLogit};

use super::actions::{DecodeAction, PrefillChunkAction, SchedulerAction, plan_prefill_chunk};
use super::session::{GenerateRequest, RequestId, SequenceFinishReason, SequenceState, SessionId};
use super::{KvHandle, SequenceSlotPool};

#[derive(Debug, Clone)]
struct WaitingRequest {
    request: GenerateRequest,
    position_start: Option<usize>,
}

impl WaitingRequest {
    fn new(request: GenerateRequest) -> Self {
        Self {
            request,
            position_start: None,
        }
    }

    fn at_position(request: GenerateRequest, position_start: usize) -> Self {
        Self {
            request,
            position_start: Some(position_start),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentSchedulerConfig {
    pub prefill_chunk_size: usize,
    pub max_active_sequences: usize,
    pub max_decode_batch: usize,
    /// Desired number of ready decode sequences before dispatch. Normalized to
    /// `1..=max_decode_batch`.
    pub decode_cohort_target: usize,
    /// Maximum consecutive cohort-building prefill-only decisions allowed while
    /// a non-empty decode cohort is below target. Zero preserves eager dispatch
    /// except for the token-budget fairness turn described below.
    pub decode_cohort_max_deferrals: usize,
    /// Maximum total packed tokens in one execution batch. Zero means no limit.
    /// This bounds the combined prefill + decode token count per batch.
    pub max_batch_tokens: usize,
    /// Tokens reserved for runnable prefill when decode can exhaust a bounded
    /// batch. Defaults to one; zero disables token-budget fairness. Mixed batches
    /// retain at least one decode token, even if the requested reserve is larger.
    /// With a one-token budget or mixed batches disabled, an exhausted decode
    /// batch instead earns prefill one turn before another decode batch.
    /// Below budget, existing packing and decode cohort policy are unchanged.
    pub prefill_reserve_tokens: usize,
    /// When true, the scheduler may combine prefill and decode sequences into
    /// one mixed execution batch. When false, prefill and decode are dispatched
    /// as separate batches.
    pub allow_mixed_batches: bool,
    /// Maximum number of retained KV pages charged to committed prompt prefixes.
    /// Zero disables prefix reuse without affecting ordinary resident execution.
    pub prefix_cache_capacity_pages: usize,
}

impl Default for ResidentSchedulerConfig {
    fn default() -> Self {
        Self {
            prefill_chunk_size: super::actions::DEFAULT_CHUNK_SIZE,
            max_active_sequences: 1,
            max_decode_batch: 1,
            decode_cohort_target: 1,
            decode_cohort_max_deferrals: 0,
            max_batch_tokens: super::actions::DEFAULT_CHUNK_SIZE,
            prefill_reserve_tokens: 1,
            allow_mixed_batches: true,
            prefix_cache_capacity_pages: 0,
        }
    }
}

/// Outcome of cancelling a resident generation request by request ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelRequestResult {
    /// The request was removed before admission and never owned runtime resources.
    Waiting {
        request_id: RequestId,
        session_id: SessionId,
    },
    /// The request was active; its scheduler slot was released.
    Active {
        request_id: RequestId,
        session_id: SessionId,
    },
    /// No waiting or active request had this ID.
    NotFound { request_id: RequestId },
}

/// Exactly one terminal request outcome removed from scheduler ownership.
#[derive(Debug, Clone)]
pub enum RequestTerminal {
    Finished(SequenceState),
    Cancelled(SequenceState),
    Failed(SequenceState),
}

impl RequestTerminal {
    pub const fn sequence(&self) -> &SequenceState {
        match self {
            Self::Finished(sequence) | Self::Cancelled(sequence) | Self::Failed(sequence) => {
                sequence
            }
        }
    }
}

impl ResidentSchedulerConfig {
    fn normalized(self) -> Self {
        let max_decode_batch = self.max_decode_batch.max(1);
        Self {
            prefill_chunk_size: self.prefill_chunk_size.max(1),
            max_active_sequences: self.max_active_sequences.max(1),
            max_decode_batch,
            decode_cohort_target: self.decode_cohort_target.clamp(1, max_decode_batch),
            decode_cohort_max_deferrals: self.decode_cohort_max_deferrals,
            max_batch_tokens: self.max_batch_tokens,
            prefill_reserve_tokens: self.prefill_reserve_tokens,
            allow_mixed_batches: self.allow_mixed_batches,
            prefix_cache_capacity_pages: self.prefix_cache_capacity_pages,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SuspendedSequenceSchedule {
    sequence: SequenceState,
    was_prefill_ready: bool,
    was_decode_ready: bool,
}

/// Scheduler admission prepared with an owned logical slot but not yet visible.
///
/// The driver must either publish this token after model/KV preparation succeeds
/// or abort it to restore the waiting request and release the slot.
#[derive(Debug)]
pub(crate) struct PreparedWaitingAdmission {
    waiting: WaitingRequest,
    sequence: SequenceState,
}

impl PreparedWaitingAdmission {
    pub(crate) const fn session_id(&self) -> SessionId {
        self.sequence.session_id
    }

    pub(crate) fn is_fresh_prompt(&self) -> bool {
        self.waiting.position_start.is_none() && self.sequence.position == 0
    }

    pub(crate) fn prompt_tokens(&self) -> &[u32] {
        self.sequence.current_prompt_tokens()
    }

    pub(crate) fn apply_committed_prefix(&mut self, matched_tokens: usize) -> Result<()> {
        if !self.is_fresh_prompt() {
            return Err(Error::Invariant {
                message: "prefix reuse requires a fresh prompt admission".into(),
            });
        }
        if matched_tokens >= self.sequence.prompt_len {
            return Err(Error::Invariant {
                message: format!(
                    "prefix reuse must leave the final prompt token executable: matched {matched_tokens} of {}",
                    self.sequence.prompt_len
                ),
            });
        }
        self.sequence.position = matched_tokens;
        self.sequence.prompt_cursor = matched_tokens;
        Ok(())
    }
}

/// Scheduler half of an exact-prefix fork, validated but not yet visible.
#[derive(Debug)]
pub(crate) struct PreparedSequenceFork {
    source_session_id: SessionId,
    target: SequenceState,
}

impl PreparedSequenceFork {
    pub(crate) fn target_session_id(&self) -> SessionId {
        self.target.session_id
    }
}

impl SuspendedSequenceSchedule {
    pub(crate) fn request_id(&self) -> Option<RequestId> {
        self.sequence.request_id
    }
    pub fn session_id(&self) -> SessionId {
        self.sequence.session_id
    }
}

#[derive(Debug)]
pub struct ResidentScheduler {
    config: ResidentSchedulerConfig,
    waiting: VecDeque<WaitingRequest>,
    active: HashMap<SessionId, SequenceState, RandomState>,
    prefill_queue: VecDeque<SessionId>,
    decode_ready: VecDeque<SessionId>,
    decode_cohort_deferrals: usize,
    prefill_turn_due: bool,
    finished: Vec<SequenceState>,
    cancelled: Vec<SequenceState>,
    failed: Vec<SequenceState>,
    next_session_id: u64,
    total_submitted: u64,
}

impl Default for ResidentScheduler {
    fn default() -> Self {
        Self::new(ResidentSchedulerConfig::default())
    }
}

impl ResidentScheduler {
    pub fn new(config: ResidentSchedulerConfig) -> Self {
        Self {
            config: config.normalized(),
            waiting: VecDeque::new(),
            active: HashMap::default(),
            prefill_queue: VecDeque::new(),
            decode_ready: VecDeque::new(),
            decode_cohort_deferrals: 0,
            prefill_turn_due: false,
            finished: Vec::new(),
            cancelled: Vec::new(),
            failed: Vec::new(),
            next_session_id: 1,
            total_submitted: 0,
        }
    }

    pub fn config(&self) -> ResidentSchedulerConfig {
        self.config
    }

    pub fn submit(&mut self, request: GenerateRequest) {
        self.total_submitted = self.total_submitted.saturating_add(1);
        self.waiting.push_back(WaitingRequest::new(request));
    }

    /// Submit a request turn whose prompt should append to an already-resident
    /// backend session at `position_start`.
    ///
    /// The runner owns the physical session/KV state, while the scheduler owns
    /// request-turn accounting and model-neutral slot lifecycle.
    pub fn submit_at_position(&mut self, request: GenerateRequest, position_start: usize) {
        self.total_submitted = self.total_submitted.saturating_add(1);
        self.waiting
            .push_back(WaitingRequest::at_position(request, position_start));
    }

    pub(crate) fn contains_request_identity(&self, id: RequestId) -> bool {
        self.waiting.iter().any(|waiting| waiting.request.id == id)
            || self
                .active
                .values()
                .chain(&self.finished)
                .chain(&self.cancelled)
                .chain(&self.failed)
                .any(|sequence| sequence.request_id == Some(id))
    }

    pub(crate) fn contains_session_identity(&self, id: SessionId) -> bool {
        self.waiting
            .iter()
            .any(|waiting| waiting.request.session_id == Some(id))
            || self
                .active
                .values()
                .chain(&self.finished)
                .chain(&self.cancelled)
                .chain(&self.failed)
                .any(|sequence| sequence.session_id == id)
    }

    pub(crate) fn identity_sessions(&self) -> Vec<SessionId> {
        self.waiting
            .iter()
            .filter_map(|waiting| waiting.request.session_id)
            .chain(
                self.active
                    .values()
                    .chain(&self.finished)
                    .chain(&self.cancelled)
                    .chain(&self.failed)
                    .map(|sequence| sequence.session_id),
            )
            .collect()
    }

    pub fn total_submitted(&self) -> u64 {
        self.total_submitted
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    pub(crate) fn active_session_for_request(&self, request_id: RequestId) -> Option<SessionId> {
        self.active.iter().find_map(|(session_id, sequence)| {
            (sequence.request_id == Some(request_id)).then_some(*session_id)
        })
    }

    pub(crate) fn request_ids(&self) -> Vec<RequestId> {
        let mut requests = self
            .waiting
            .iter()
            .map(|waiting| waiting.request.id)
            .chain(
                self.active
                    .values()
                    .filter_map(|sequence| sequence.request_id),
            )
            .collect::<Vec<_>>();
        requests.sort_unstable_by_key(|request| request.0);
        requests.dedup();
        requests
    }

    /// Restore exact decode actions removed for a suspended transaction.
    ///
    /// Validation is failure-atomic: no queue entry is published unless every
    /// survivor still matches its staged token, position, request, and KV handle.
    pub(crate) fn requeue_decode_actions_front(&mut self, actions: &[DecodeAction]) -> Result<()> {
        for (index, action) in actions.iter().enumerate() {
            if actions[..index]
                .iter()
                .any(|queued| queued.session_id == action.session_id)
            {
                return Err(Error::Invariant {
                    message: format!(
                        "duplicate suspended decode action for session {:?}",
                        action.session_id
                    ),
                });
            }
            if self.decode_ready.contains(&action.session_id) {
                return Err(Error::Invariant {
                    message: format!(
                        "suspended decode session {:?} is already ready",
                        action.session_id
                    ),
                });
            }
            let sequence = self
                .active
                .get(&action.session_id)
                .ok_or_else(|| Error::Invariant {
                    message: format!(
                        "cannot restore decode action for inactive session {:?}",
                        action.session_id
                    ),
                })?;
            let token_id = sequence.next_decode_token.ok_or_else(|| Error::Invariant {
                message: format!(
                    "cannot restore decode action without a staged token for session {:?}",
                    action.session_id
                ),
            })?;
            let current = DecodeAction::from_sequence(sequence, token_id);
            if current != *action {
                return Err(Error::Invariant {
                    message: format!(
                        "suspended decode action no longer matches session {:?}",
                        action.session_id
                    ),
                });
            }
        }
        for action in actions.iter().rev() {
            self.decode_ready.push_front(action.session_id);
        }
        Ok(())
    }

    pub fn prefill_queue_len(&self) -> usize {
        self.prefill_queue.len()
    }

    pub fn decode_ready_len(&self) -> usize {
        self.decode_ready.len()
    }

    pub fn finished_len(&self) -> usize {
        self.finished.len()
    }

    pub fn cancelled_len(&self) -> usize {
        self.cancelled.len()
    }

    pub fn failed_len(&self) -> usize {
        self.failed.len()
    }

    pub fn active_sequence(&self, session_id: SessionId) -> Option<&SequenceState> {
        self.active.get(&session_id)
    }

    pub fn active_sequence_mut(&mut self, session_id: SessionId) -> Option<&mut SequenceState> {
        self.active.get_mut(&session_id)
    }

    /// Returns the session IDs of all active sequences in arbitrary order.
    pub fn active_session_ids(&self) -> Vec<SessionId> {
        self.active.keys().copied().collect()
    }

    /// Validate and build scheduler metadata for an exact-prefix fork without
    /// publishing the target or changing the source candidate.
    pub(crate) fn prepare_fork_session_exact(
        &self,
        source_session_id: SessionId,
        target_session_id: SessionId,
        request: &GenerateRequest,
        expected_position: usize,
        kv_handle: KvHandle,
    ) -> Result<PreparedSequenceFork> {
        if source_session_id == target_session_id {
            return Err(Error::InvalidRequest {
                message: "fork source and target sessions must differ".into(),
            });
        }
        if request.session_id != Some(target_session_id) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "fork target request must name target session {target_session_id:?}"
                ),
            });
        }
        if self.active.contains_key(&target_session_id)
            || self
                .waiting
                .iter()
                .any(|waiting| waiting.request.session_id == Some(target_session_id))
        {
            return Err(Error::InvalidRequest {
                message: format!("fork target session {target_session_id:?} already exists"),
            });
        }
        if self.active.len() >= self.config.max_active_sequences {
            return Err(Error::InvalidRequest {
                message: "resident scheduler has no active-sequence capacity for fork target"
                    .into(),
            });
        }
        let source = self
            .active
            .get(&source_session_id)
            .ok_or_else(|| Error::InvalidRequest {
                message: format!("fork source session {source_session_id:?} is not active"),
            })?;
        let mut target = source.fork_exact(target_session_id, request, expected_position)?;
        target.bind_kv(kv_handle);
        Ok(PreparedSequenceFork {
            source_session_id,
            target,
        })
    }

    /// Publish a fully prepared fork. All fallible validation happens in prepare.
    pub(crate) fn publish_fork_session_exact(&mut self, prepared: PreparedSequenceFork) {
        if let Some(source) = self.active.get_mut(&prepared.source_session_id) {
            source.next_decode_token = None;
            source.next_decode_logit = None;
        }
        self.remove_from_queue(prepared.source_session_id, QueueKind::Decode);
        let target_session_id = prepared.target.session_id;
        if !prepared.target.prompt_prefill_done() {
            self.prefill_queue.push_back(target_session_id);
        }
        let previous = self.active.insert(target_session_id, prepared.target);
        debug_assert!(
            previous.is_none(),
            "prepared fork target must remain absent"
        );
    }

    /// Remove a sequence from runnable queues without finishing or releasing it.
    pub fn suspend_sequence(&mut self, session_id: SessionId) -> Result<SuspendedSequenceSchedule> {
        let was_prefill_ready = self.prefill_queue.contains(&session_id);
        let was_decode_ready = self.decode_ready.contains(&session_id);
        let sequence = self.remove_active_sequence(session_id)?;
        Ok(SuspendedSequenceSchedule {
            sequence,
            was_prefill_ready,
            was_decode_ready,
        })
    }

    /// Return a suspended sequence to the same runnable queue it occupied.
    pub fn restore_suspended(&mut self, suspended: SuspendedSequenceSchedule) -> Result<()> {
        let session_id = suspended.sequence.session_id;
        if self.active.contains_key(&session_id) {
            return Err(Error::Invariant {
                message: format!("cannot restore already-active resident session {session_id:?}"),
            });
        }
        if suspended.was_prefill_ready {
            self.prefill_queue.push_back(session_id);
        }
        if suspended.was_decode_ready {
            self.decode_ready.push_back(session_id);
        }

        self.active.insert(session_id, suspended.sequence);
        Ok(())
    }

    /// Cancel a waiting or active request by its service-level request ID.
    ///
    /// Waiting requests are converted directly to terminal cancellation records.
    /// Active requests additionally release their scheduler-owned sequence slot.
    pub fn cancel_request<C>(
        &mut self,
        request_id: RequestId,
        slot_pool: &mut C,
    ) -> Result<CancelRequestResult>
    where
        C: SequenceSlotPool,
    {
        if let Some(session_id) = self.active.iter().find_map(|(session_id, sequence)| {
            (sequence.request_id == Some(request_id)).then_some(*session_id)
        }) {
            self.cancel_sequence(session_id, slot_pool)?;
            return Ok(CancelRequestResult::Active {
                request_id,
                session_id,
            });
        }

        if let Some(index) = self
            .waiting
            .iter()
            .position(|waiting| waiting.request.id == request_id)
        {
            let waiting = self
                .waiting
                .remove(index)
                .expect("waiting request index was just found");
            let session_id = self.resolve_session_id(waiting.request.session_id);
            let mut sequence = SequenceState::from_request(&waiting.request, session_id);
            if let Some(position_start) = waiting.position_start {
                sequence.position = position_start;
            }
            sequence.mark_cancelled();
            self.cancelled.push(sequence);
            return Ok(CancelRequestResult::Waiting {
                request_id,
                session_id,
            });
        }

        if let Some(session_id) =
            self.cancelled
                .iter()
                .chain(self.failed.iter())
                .find_map(|sequence| {
                    (sequence.request_id == Some(request_id)).then_some(sequence.session_id)
                })
        {
            return Ok(CancelRequestResult::Active {
                request_id,
                session_id,
            });
        }

        Ok(CancelRequestResult::NotFound { request_id })
    }

    pub fn take_request_terminal(&mut self, request_id: RequestId) -> Option<RequestTerminal> {
        fn take(
            sequences: &mut Vec<SequenceState>,
            request_id: RequestId,
        ) -> Option<SequenceState> {
            let index = sequences
                .iter()
                .position(|sequence| sequence.request_id == Some(request_id))?;
            Some(sequences.remove(index))
        }

        take(&mut self.finished, request_id)
            .map(RequestTerminal::Finished)
            .or_else(|| take(&mut self.cancelled, request_id).map(RequestTerminal::Cancelled))
            .or_else(|| {
                let index = self.failed.iter().position(|sequence| {
                    sequence.request_id == Some(request_id) && sequence.kv_handle.is_none()
                })?;
                Some(RequestTerminal::Failed(self.failed.remove(index)))
            })
    }

    pub fn drain_finished(&mut self) -> Vec<SequenceState> {
        std::mem::take(&mut self.finished)
    }

    pub fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        std::mem::take(&mut self.cancelled)
    }

    pub fn drain_failed(&mut self) -> Vec<SequenceState> {
        let (drained, retained): (Vec<_>, Vec<_>) = std::mem::take(&mut self.failed)
            .into_iter()
            .partition(|sequence| sequence.kv_handle.is_none());
        self.failed = retained;
        drained
    }

    pub(crate) fn retry_failed_slot_releases<C>(&mut self, slot_pool: &mut C) -> Result<usize>
    where
        C: SequenceSlotPool,
    {
        let mut released = 0;
        let mut first_error = None;
        for sequence in &mut self.failed {
            let Some(handle) = sequence.kv_handle else {
                continue;
            };
            match slot_pool.free_slot(handle) {
                Ok(()) => {
                    sequence.clear_kv();
                    released += 1;
                }
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        first_error.map_or(Ok(released), Err)
    }

    pub(crate) fn failed_slot_ownership(&self) -> usize {
        self.failed
            .iter()
            .filter(|sequence| sequence.kv_handle.is_some())
            .count()
    }

    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty()
            && self.active.is_empty()
            && self.prefill_queue.is_empty()
            && self.decode_ready.is_empty()
    }

    /// Reserve the scheduler-owned slot for the front waiting request without
    /// publishing it to active/runnable state.
    pub(crate) fn prepare_waiting_admission<C>(
        &mut self,
        slot_pool: &mut C,
    ) -> Result<Option<PreparedWaitingAdmission>>
    where
        C: SequenceSlotPool,
    {
        if self.active.len() >= self.config.max_active_sequences {
            return Ok(None);
        }
        let Some(waiting) = self.waiting.pop_front() else {
            return Ok(None);
        };
        let session_id = self.resolve_session_id(waiting.request.session_id);
        if self.active.contains_key(&session_id) {
            self.waiting.push_front(waiting);
            return Err(Error::Invariant {
                message: format!("session {:?} is already active", session_id),
            });
        }
        let kv_handle = match slot_pool.alloc_slot() {
            Ok(handle) => handle,
            Err(_) => {
                self.waiting.push_front(waiting);
                return Ok(None);
            }
        };
        let mut sequence = SequenceState::from_request(&waiting.request, session_id);
        if let Some(position_start) = waiting.position_start {
            sequence.position = position_start;
        }
        sequence.bind_kv(kv_handle);
        Ok(Some(PreparedWaitingAdmission { waiting, sequence }))
    }

    /// Publish a fully prepared admission. No fallible work remains after driver
    /// model/KV state has become visible.
    pub(crate) fn publish_waiting_admission(&mut self, prepared: PreparedWaitingAdmission) {
        let session_id = prepared.sequence.session_id;
        assert!(
            self.active.len() < self.config.max_active_sequences,
            "prepared admission capacity must remain reserved"
        );
        assert!(
            !self.active.contains_key(&session_id),
            "prepared admission target must remain absent"
        );
        if !prepared.sequence.prompt_prefill_done() {
            self.prefill_queue.push_back(session_id);
        }
        self.active.insert(session_id, prepared.sequence);
    }

    /// Abort a prepared admission. A successful slot release restores the exact
    /// request at the queue front. If release fails, the failed sequence retains
    /// its handle so ownership is explicit and the request is not duplicated.
    pub(crate) fn abort_waiting_admission<C>(
        &mut self,
        mut prepared: PreparedWaitingAdmission,
        slot_pool: &mut C,
    ) -> Result<()>
    where
        C: SequenceSlotPool,
    {
        let handle = prepared
            .sequence
            .kv_handle
            .expect("prepared admission always owns a logical slot");
        match slot_pool.free_slot(handle) {
            Ok(()) => {
                prepared.sequence.clear_kv();
                self.waiting.push_front(prepared.waiting);
                Ok(())
            }
            Err(error) => {
                prepared.sequence.mark_error();
                self.failed.push(prepared.sequence);
                Err(error)
            }
        }
    }

    pub fn admit_waiting<C>(&mut self, slot_pool: &mut C) -> Result<usize>
    where
        C: SequenceSlotPool,
    {
        let mut admitted = 0;
        while let Some(prepared) = self.prepare_waiting_admission(slot_pool)? {
            self.publish_waiting_admission(prepared);
            admitted += 1;
        }
        Ok(admitted)
    }

    pub fn next_prefill_action<C>(&mut self, slot_pool: &mut C) -> Result<Option<SchedulerAction>>
    where
        C: SequenceSlotPool,
    {
        if self.prefill_queue.is_empty() {
            self.admit_waiting(slot_pool)?;
        }

        while let Some(session_id) = self.prefill_queue.front().copied() {
            let Some(sequence) = self.active.get(&session_id) else {
                self.prefill_queue.pop_front();
                continue;
            };
            if sequence.prompt_prefill_done() {
                self.prefill_queue.pop_front();
                continue;
            }
            let action = plan_prefill_chunk(sequence, self.config.prefill_chunk_size)?;

            return Ok(action);
        }

        Ok(None)
    }

    /// Pick the next executable action, preferring ready decode work without
    /// starving prefill at a saturated token budget. Callers needing prefill-first
    /// behavior can keep using `next_prefill_action` and `next_decode_action`
    /// directly.
    pub fn next_action<C>(&mut self, slot_pool: &mut C) -> Result<Option<SchedulerAction>>
    where
        C: SequenceSlotPool,
    {
        self.next_action_policy(slot_pool, self.config.allow_mixed_batches)
    }

    pub(crate) fn next_action_policy<C>(
        &mut self,
        slot_pool: &mut C,
        allow_mixed_batches: bool,
    ) -> Result<Option<SchedulerAction>>
    where
        C: SequenceSlotPool,
    {
        self.admit_waiting(slot_pool)?;
        self.next_admitted_action_policy(allow_mixed_batches)
    }

    pub(crate) fn next_admitted_action_policy(
        &mut self,
        allow_mixed_batches: bool,
    ) -> Result<Option<SchedulerAction>> {
        let token_budget = if self.config.max_batch_tokens == 0 {
            usize::MAX
        } else {
            self.config.max_batch_tokens
        };
        let has_prefill = self.prefill_queue.iter().any(|session_id| {
            self.active
                .get(session_id)
                .is_some_and(|sequence| !sequence.prompt_prefill_done())
        });
        let fairness_enabled = self.config.prefill_reserve_tokens > 0
            && self.config.max_batch_tokens != 0
            && has_prefill;
        let decode_saturates_budget =
            self.decode_ready.len().min(self.config.max_decode_batch) >= token_budget;
        let separate_prefill_turn = !allow_mixed_batches || token_budget == 1;
        let fair_prefill_turn = fairness_enabled && separate_prefill_turn && self.prefill_turn_due;
        let defer_decode = fair_prefill_turn
            || (!allow_mixed_batches
                && !self.decode_ready.is_empty()
                && self.decode_ready.len() < self.config.decode_cohort_target
                && has_prefill
                && self.decode_cohort_deferrals < self.config.decode_cohort_max_deferrals);
        // Leave decode at least one row. If the two kinds cannot share a batch,
        // repay an exhausted decode-only decision with one prefill turn instead.
        let prefill_reserve =
            if fairness_enabled && !separate_prefill_turn && decode_saturates_budget {
                self.config.prefill_reserve_tokens.min(token_budget - 1)
            } else {
                0
            };
        let decode_budget = token_budget - prefill_reserve;
        let mut remaining = token_budget;
        let mut decodes = Vec::new();
        let decode_candidates = if defer_decode {
            0
        } else {
            self.decode_ready.len()
        };
        for _ in 0..decode_candidates {
            if remaining == 0
                || decodes.len() >= self.config.max_decode_batch
                || decodes.len() >= decode_budget
            {
                break;
            }
            let Some(session_id) = self.decode_ready.pop_front() else {
                break;
            };
            let Some(sequence) = self.active.get(&session_id) else {
                continue;
            };
            let Some(token_id) = sequence.next_decode_token else {
                continue;
            };
            decodes.push(DecodeAction::from_sequence(sequence, token_id));
            remaining -= 1;
        }

        let mut prefills = Vec::new();
        if allow_mixed_batches || decodes.is_empty() {
            let candidates = self.prefill_queue.len();
            for _ in 0..candidates {
                if remaining == 0 {
                    break;
                }
                let Some(session_id) = self.prefill_queue.pop_front() else {
                    break;
                };
                let Some(sequence) = self.active.get(&session_id) else {
                    continue;
                };
                if sequence.prompt_prefill_done() {
                    continue;
                }
                let chunk = self.config.prefill_chunk_size.min(remaining);
                let Some(action) = PrefillChunkAction::from_sequence(sequence, chunk)? else {
                    continue;
                };
                self.prefill_queue.push_back(session_id);
                remaining -= action.tokens.len();
                prefills.push(action);
                if !allow_mixed_batches {
                    break;
                }
            }
        }

        self.prefill_turn_due = fairness_enabled
            && separate_prefill_turn
            && prefills.is_empty()
            && decodes.len() == token_budget;
        if !decodes.is_empty() || self.decode_ready.is_empty() {
            self.decode_cohort_deferrals = 0;
        } else if !prefills.is_empty() && defer_decode {
            // A fairness turn also counts toward cohort-building deferrals; it
            // must not buy an additional run of prefill-only cohort decisions.
            self.decode_cohort_deferrals = self.decode_cohort_deferrals.saturating_add(1);
        }

        if prefills.is_empty() && decodes.is_empty() {
            Ok(None)
        } else if !allow_mixed_batches {
            if decodes.is_empty() {
                Ok(prefills.pop().map(SchedulerAction::PrefillChunk))
            } else {
                Ok(Some(SchedulerAction::DecodeBatch(decodes)))
            }
        } else {
            Ok(Some(SchedulerAction::Execute { prefills, decodes }))
        }
    }

    pub fn commit_prefill_action(&mut self, action: &PrefillChunkAction) -> Result<()> {
        let sequence = self
            .active
            .get_mut(&action.session_id)
            .ok_or_else(|| Error::Invariant {
                message: format!(
                    "cannot commit prefill for inactive session {:?}",
                    action.session_id
                ),
            })?;
        action.commit(sequence)?;
        if sequence.prompt_prefill_done() {
            self.remove_from_queue(action.session_id, QueueKind::Prefill);
        }
        Ok(())
    }

    pub fn stage_decode_token(&mut self, session_id: SessionId, token_id: u32) -> Result<()> {
        self.stage_decode_candidate(session_id, token_id, None)
    }

    pub fn stage_decode_candidate(
        &mut self,
        session_id: SessionId,
        token_id: u32,
        logit: Option<f32>,
    ) -> Result<()> {
        let sequence = self
            .active
            .get_mut(&session_id)
            .ok_or_else(|| Error::Invariant {
                message: format!(
                    "cannot stage decode token for inactive session {:?}",
                    session_id
                ),
            })?;
        sequence.stage_decode_candidate(token_id, logit)?;
        if !self.decode_ready.iter().any(|queued| *queued == session_id) {
            self.decode_ready.push_back(session_id);
        }
        Ok(())
    }

    /// Stage a greedy candidate after runtime has explicitly correlated neutral
    /// output back to a service-level session.
    pub fn stage_greedy_decode_from_logits(
        &mut self,
        session_id: SessionId,
        logits: &LogitsOutput,
    ) -> Result<bool> {
        let Some(candidate) = greedy_candidate(logits) else {
            return Ok(false);
        };
        self.stage_decode_candidate(session_id, candidate.token_id, Some(candidate.logit))?;
        Ok(true)
    }

    pub fn next_decode_action(&mut self) -> Result<Option<SchedulerAction>> {
        let mut actions = Vec::new();
        while actions.len() < self.config.max_decode_batch {
            let Some(session_id) = self.decode_ready.pop_front() else {
                break;
            };
            let Some(sequence) = self.active.get(&session_id) else {
                continue;
            };
            let Some(token_id) = sequence.next_decode_token else {
                continue;
            };
            actions.push(DecodeAction::from_sequence(sequence, token_id));
        }

        if actions.is_empty() {
            if self.decode_ready.is_empty() {
                self.decode_cohort_deferrals = 0;
            }
            Ok(None)
        } else {
            self.decode_cohort_deferrals = 0;
            Ok(Some(SchedulerAction::DecodeBatch(actions)))
        }
    }

    pub fn commit_decode_action(&mut self, action: &DecodeAction) -> Result<()> {
        let sequence = self
            .active
            .get_mut(&action.session_id)
            .ok_or_else(|| Error::Invariant {
                message: format!(
                    "cannot commit decode for inactive session {:?}",
                    action.session_id
                ),
            })?;
        Self::validate_decode_commit(sequence, action)?;
        sequence.commit_staged_decode_token(action.token_id)
    }

    fn validate_decode_commit(sequence: &SequenceState, action: &DecodeAction) -> Result<()> {
        if sequence.position != action.position {
            return Err(Error::Invariant {
                message: format!(
                    "decode position mismatch for session {:?}: sequence at {}, action at {}",
                    action.session_id, sequence.position, action.position
                ),
            });
        }
        if sequence.kv_handle != action.kv_handle {
            return Err(Error::Invariant {
                message: format!(
                    "decode KV handle mismatch for session {:?}: sequence {:?}, action {:?}",
                    action.session_id, sequence.kv_handle, action.kv_handle
                ),
            });
        }
        match sequence.next_decode_token {
            Some(expected) if expected == action.token_id => Ok(()),
            Some(expected) => Err(Error::Invariant {
                message: format!(
                    "decode token mismatch: staged {expected}, committed {}",
                    action.token_id
                ),
            }),
            None => Err(Error::Invariant {
                message: "cannot commit decode token without a staged token".into(),
            }),
        }
    }

    pub fn commit_decode_batch(&mut self, actions: &[DecodeAction]) -> Result<usize> {
        for action in actions {
            self.commit_decode_action(action)?;
        }
        Ok(actions.len())
    }

    /// Commit scheduler-owned state after an action has been executed by a backend.
    ///
    /// This updates prefill cursors or committed decode tokens; logits/output
    /// staging remains explicit after runtime correlation so sampling policy stays
    /// separate from state commits.
    pub fn commit_action(&mut self, action: &SchedulerAction) -> Result<usize> {
        self.preflight_action_commit(action)?;
        match action {
            SchedulerAction::Execute { prefills, decodes } => {
                let mut committed = 0;
                for prefill in prefills {
                    self.commit_prefill_action(prefill)?;
                    committed += prefill.token_range.len();
                }
                committed += self.commit_decode_batch(decodes)?;
                Ok(committed)
            }
            SchedulerAction::PrefillChunk(prefill) => {
                self.commit_prefill_action(prefill)?;
                Ok(prefill.token_range.len())
            }
            SchedulerAction::DecodeBatch(actions) => self.commit_decode_batch(actions),
            SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => Ok(0),
        }
    }

    fn preflight_action_commit(&self, action: &SchedulerAction) -> Result<()> {
        let mut sequences = HashMap::<SessionId, SequenceState, RandomState>::default();
        match action {
            SchedulerAction::Execute { prefills, decodes } => {
                for prefill in prefills {
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        sequences.entry(prefill.session_id)
                    {
                        let sequence = self.active.get(&prefill.session_id).ok_or_else(|| {
                            Error::Invariant {
                                message: format!(
                                    "cannot commit prefill for inactive session {:?}",
                                    prefill.session_id
                                ),
                            }
                        })?;
                        entry.insert(sequence.clone());
                    }
                    prefill.commit(
                        sequences
                            .get_mut(&prefill.session_id)
                            .expect("prefill preflight inserted its sequence"),
                    )?;
                }
                for decode in decodes {
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        sequences.entry(decode.session_id)
                    {
                        let sequence = self.active.get(&decode.session_id).ok_or_else(|| {
                            Error::Invariant {
                                message: format!(
                                    "cannot commit decode for inactive session {:?}",
                                    decode.session_id
                                ),
                            }
                        })?;
                        entry.insert(sequence.clone());
                    }
                    let sequence = sequences
                        .get_mut(&decode.session_id)
                        .expect("decode preflight inserted its sequence");
                    Self::validate_decode_commit(sequence, decode)?;
                    sequence.commit_staged_decode_token(decode.token_id)?;
                }
            }
            SchedulerAction::PrefillChunk(prefill) => {
                let sequence =
                    self.active
                        .get(&prefill.session_id)
                        .ok_or_else(|| Error::Invariant {
                            message: format!(
                                "cannot commit prefill for inactive session {:?}",
                                prefill.session_id
                            ),
                        })?;
                let mut sequence = sequence.clone();
                prefill.commit(&mut sequence)?;
            }
            SchedulerAction::DecodeBatch(decodes) => {
                let mut sequences = HashMap::<SessionId, SequenceState, RandomState>::default();
                for decode in decodes {
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        sequences.entry(decode.session_id)
                    {
                        let sequence = self.active.get(&decode.session_id).ok_or_else(|| {
                            Error::Invariant {
                                message: format!(
                                    "cannot commit decode for inactive session {:?}",
                                    decode.session_id
                                ),
                            }
                        })?;
                        entry.insert(sequence.clone());
                    }
                    let sequence = sequences
                        .get_mut(&decode.session_id)
                        .expect("decode preflight inserted its sequence");
                    Self::validate_decode_commit(sequence, decode)?;
                    sequence.commit_staged_decode_token(decode.token_id)?;
                }
            }
            SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => {}
        }
        Ok(())
    }

    pub fn finish_sequence<C>(
        &mut self,
        session_id: SessionId,
        reason: SequenceFinishReason,
        slot_pool: &mut C,
    ) -> Result<SchedulerAction>
    where
        C: SequenceSlotPool,
    {
        let mut sequence = self.remove_active_sequence(session_id)?;
        if let Some(handle) = sequence.kv_handle {
            if let Err(error) = slot_pool.free_slot(handle) {
                sequence.mark_error();
                self.failed.push(sequence);
                return Err(error);
            }
            sequence.clear_kv();
        }
        sequence.mark_finished(reason);
        let action = SchedulerAction::Finish {
            request_id: sequence.request_id,
            session_id,
            reason,
        };
        self.finished.push(sequence);
        Ok(action)
    }

    pub(crate) fn cancel_suspended<C>(
        &mut self,
        suspended: SuspendedSequenceSchedule,
        slot_pool: &mut C,
    ) -> Result<SchedulerAction>
    where
        C: SequenceSlotPool,
    {
        let mut sequence = suspended.sequence;
        let session_id = sequence.session_id;
        if let Some(handle) = sequence.kv_handle {
            if let Err(error) = slot_pool.free_slot(handle) {
                sequence.mark_error();
                self.failed.push(sequence);
                return Err(error);
            }
            sequence.clear_kv();
        }
        sequence.mark_cancelled();
        let action = SchedulerAction::Cancel {
            request_id: sequence.request_id,
            session_id,
        };
        self.cancelled.push(sequence);
        Ok(action)
    }

    pub fn cancel_sequence<C>(
        &mut self,
        session_id: SessionId,
        slot_pool: &mut C,
    ) -> Result<SchedulerAction>
    where
        C: SequenceSlotPool,
    {
        let mut sequence = self.remove_active_sequence(session_id)?;
        if let Some(handle) = sequence.kv_handle {
            if let Err(error) = slot_pool.free_slot(handle) {
                sequence.mark_error();
                self.failed.push(sequence);
                return Err(error);
            }
            sequence.clear_kv();
        }
        sequence.mark_cancelled();
        let action = SchedulerAction::Cancel {
            request_id: sequence.request_id,
            session_id,
        };
        self.cancelled.push(sequence);
        Ok(action)
    }

    /// Remove one sequence from scheduling after backend execution may have
    /// partially committed state. Its slot is released when possible; a failed
    /// release leaves the handle attached to the terminal error record.
    pub fn fail_sequence<C>(&mut self, session_id: SessionId, slot_pool: &mut C) -> Result<()>
    where
        C: SequenceSlotPool,
    {
        let mut sequence = self.remove_active_sequence(session_id)?;
        sequence.mark_error();
        let release_result = if let Some(handle) = sequence.kv_handle {
            match slot_pool.free_slot(handle) {
                Ok(()) => {
                    sequence.clear_kv();
                    Ok(())
                }
                Err(error) => Err(error),
            }
        } else {
            Ok(())
        };
        self.failed.push(sequence);
        release_result
    }

    /// Fail every sequence referenced by an action. Cleanup is attempted for all
    /// rows even if one slot release fails.
    pub fn fail_action<C>(&mut self, action: &SchedulerAction, slot_pool: &mut C) -> Result<usize>
    where
        C: SequenceSlotPool,
    {
        let session_ids: Vec<SessionId> = match action {
            SchedulerAction::Execute { prefills, decodes } => prefills
                .iter()
                .map(|action| action.session_id)
                .chain(decodes.iter().map(|action| action.session_id))
                .collect(),
            SchedulerAction::PrefillChunk(prefill) => vec![prefill.session_id],
            SchedulerAction::DecodeBatch(actions) => {
                actions.iter().map(|action| action.session_id).collect()
            }
            SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => Vec::new(),
        };

        let mut failed = 0;
        let mut seen = Vec::new();
        let mut first_error = None;
        for session_id in session_ids {
            if seen.contains(&session_id) {
                continue;
            }
            seen.push(session_id);
            if !self.active.contains_key(&session_id) {
                continue;
            }
            match self.fail_sequence(session_id, slot_pool) {
                Ok(()) => failed += 1,
                Err(error) => {
                    failed += 1;
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }

        match first_error {
            Some(error) => Err(error),
            None => Ok(failed),
        }
    }

    fn resolve_session_id(&mut self, requested: Option<SessionId>) -> SessionId {
        if let Some(session_id) = requested {
            return session_id;
        }
        loop {
            let session_id = SessionId(self.next_session_id);
            self.next_session_id = self.next_session_id.saturating_add(1);
            if !self.active.contains_key(&session_id) {
                return session_id;
            }
        }
    }

    fn remove_active_sequence(&mut self, session_id: SessionId) -> Result<SequenceState> {
        self.remove_from_queue(session_id, QueueKind::Prefill);
        self.remove_from_queue(session_id, QueueKind::Decode);
        self.active
            .remove(&session_id)
            .ok_or_else(|| Error::Invariant {
                message: format!("cannot remove inactive resident session {:?}", session_id),
            })
    }

    fn remove_from_queue(&mut self, session_id: SessionId, queue: QueueKind) {
        let target = match queue {
            QueueKind::Prefill => &mut self.prefill_queue,
            QueueKind::Decode => &mut self.decode_ready,
        };
        target.retain(|queued| *queued != session_id);
        if self.prefill_queue.is_empty() {
            self.prefill_turn_due = false;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueueKind {
    Prefill,
    Decode,
}

pub(crate) fn greedy_candidate(logits: &LogitsOutput) -> Option<TokenLogit> {
    match logits {
        LogitsOutput::TopK(topk) => topk.first().copied(),
        LogitsOutput::Full(logits) => logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, logit)| TokenLogit {
                token_id: index as u32,
                logit,
            }),
    }
}

#[cfg(test)]
mod tests {
    use ferrule_common::execution::LogitsOutput;
    use ferrule_model::TokenLogit;

    use crate::scheduling::RequestId;
    use crate::scheduling::{FixedSequenceSlotPool, KvHandle, SequenceSlotPool};

    use super::*;

    fn request(id: u64, tokens: Vec<u32>) -> GenerateRequest {
        GenerateRequest {
            id: RequestId(id),
            session_id: None,
            prompt_tokens: tokens,
            max_new_tokens: 16,
            stop: Vec::new(),
            ignore_eos: false,
        }
    }

    fn fairness_fixture(
        config: ResidentSchedulerConfig,
        decode_count: usize,
        prompts: &[Vec<u32>],
    ) -> (ResidentScheduler, FixedSequenceSlotPool) {
        let capacity = decode_count + prompts.len();
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            max_active_sequences: capacity,
            ..config
        });
        let mut slots = FixedSequenceSlotPool::new(capacity);
        for index in 0..capacity {
            let tokens = if index < decode_count {
                vec![index as u32]
            } else {
                prompts[index - decode_count].clone()
            };
            let mut request = request(index as u64 + 1, tokens);
            request.max_new_tokens = 1024;
            scheduler.submit(request);
        }
        assert_eq!(scheduler.admit_waiting(&mut slots).unwrap(), capacity);
        for index in 0..decode_count {
            let action = scheduler.next_prefill_action(&mut slots).unwrap().unwrap();
            scheduler.commit_action(&action).unwrap();
            scheduler
                .stage_decode_token(SessionId(index as u64 + 1), 99)
                .unwrap();
        }
        (scheduler, slots)
    }

    fn execution_parts(action: &SchedulerAction) -> (&[PrefillChunkAction], &[DecodeAction]) {
        match action {
            SchedulerAction::Execute { prefills, decodes } => (prefills, decodes),
            SchedulerAction::PrefillChunk(prefill) => (std::slice::from_ref(prefill), &[]),
            SchedulerAction::DecodeBatch(decodes) => (&[], decodes),
            _ => panic!("expected executable action"),
        }
    }

    fn assert_bounded_fairness_action(scheduler: &ResidentScheduler, action: &SchedulerAction) {
        let (prefills, decodes) = execution_parts(action);
        let tokens = prefills.iter().map(|p| p.tokens.len()).sum::<usize>() + decodes.len();
        assert!(tokens > 0);
        assert!(
            scheduler.config.max_batch_tokens == 0 || tokens <= scheduler.config.max_batch_tokens
        );
        assert!(decodes.len() <= scheduler.config.max_decode_batch);
        let mut sessions = std::collections::HashSet::new();
        let mut requests = std::collections::HashSet::new();
        let mut handles = std::collections::HashSet::new();
        for (session_id, request_id, handle, position) in prefills
            .iter()
            .map(|p| (p.session_id, p.request_id, p.kv_handle, p.position_start))
            .chain(
                decodes
                    .iter()
                    .map(|d| (d.session_id, d.request_id, d.kv_handle, d.position)),
            )
        {
            assert!(sessions.insert(session_id), "duplicate session in tick");
            assert!(
                requests.insert(request_id.unwrap()),
                "duplicate request in tick"
            );
            assert!(handles.insert(handle.unwrap()), "aliased slot in tick");
            let sequence = scheduler.active_sequence(session_id).unwrap();
            assert_eq!(sequence.request_id, request_id);
            assert_eq!(sequence.kv_handle, handle);
            assert_eq!(sequence.position, position);
        }
        for prefill in prefills {
            let sequence = scheduler.active_sequence(prefill.session_id).unwrap();
            assert_eq!(prefill.token_range.start, sequence.prompt_cursor);
            assert_eq!(prefill.tokens.len(), prefill.token_range.len());
            assert!(prefill.tokens.len() <= scheduler.config.prefill_chunk_size);
        }
    }

    fn commit_and_restage_decodes(scheduler: &mut ResidentScheduler, action: &SchedulerAction) {
        assert_bounded_fairness_action(scheduler, action);
        scheduler.commit_action(action).unwrap();
        for decode in execution_parts(action).1 {
            scheduler
                .stage_decode_token(decode.session_id, decode.token_id)
                .unwrap();
        }
    }

    struct FailingFreeKvCache;

    impl SequenceSlotPool for FailingFreeKvCache {
        fn alloc_slot(&mut self) -> Result<KvHandle> {
            Ok(KvHandle(0))
        }

        fn free_slot(&mut self, _handle: KvHandle) -> Result<()> {
            Err(Error::Invariant {
                message: "simulated slot free failure".into(),
            })
        }
    }

    #[test]
    fn resident_scheduler_admits_prefills_decodes_and_finishes() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 2,
            max_active_sequences: 2,
            max_decode_batch: 2,
            ..Default::default()
        });
        let mut kv = FixedSequenceSlotPool::new(2);
        scheduler.submit(request(10, vec![1, 2, 3]));

        let SchedulerAction::PrefillChunk(first) = scheduler
            .next_prefill_action(&mut kv)
            .unwrap()
            .expect("first prefill action")
        else {
            panic!("expected prefill action");
        };
        assert_eq!(first.request_id, Some(RequestId(10)));
        assert_eq!(first.session_id, SessionId(1));
        assert_eq!(first.tokens, vec![1, 2]);
        assert_eq!(first.position_start, 0);
        assert_eq!(first.kv_handle, Some(KvHandle(0)));
        scheduler.commit_prefill_action(&first).unwrap();
        assert_eq!(scheduler.active_sequence(SessionId(1)).unwrap().position, 2);

        let SchedulerAction::PrefillChunk(last) = scheduler
            .next_prefill_action(&mut kv)
            .unwrap()
            .expect("last prefill action")
        else {
            panic!("expected prefill action");
        };
        assert_eq!(last.tokens, vec![3]);
        assert_eq!(last.position_start, 2);
        scheduler.commit_prefill_action(&last).unwrap();
        assert_eq!(scheduler.prefill_queue_len(), 0);
        assert_eq!(scheduler.active_sequence(SessionId(1)).unwrap().position, 3);

        let logits = LogitsOutput::TopK(vec![TokenLogit {
            token_id: 99,
            logit: 1.0,
        }]);
        assert!(
            scheduler
                .stage_greedy_decode_from_logits(SessionId(1), &logits)
                .unwrap()
        );
        assert_eq!(scheduler.decode_ready_len(), 1);

        let SchedulerAction::DecodeBatch(actions) = scheduler
            .next_decode_action()
            .unwrap()
            .expect("decode action")
        else {
            panic!("expected decode batch");
        };
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].token_id, 99);
        assert_eq!(actions[0].position, 3);
        scheduler.commit_decode_action(&actions[0]).unwrap();
        let seq = scheduler.active_sequence(SessionId(1)).unwrap();
        assert_eq!(seq.tokens, vec![1, 2, 3, 99]);
        assert_eq!(seq.position, 4);
        assert_eq!(seq.next_decode_token, None);

        let finish = scheduler
            .finish_sequence(SessionId(1), SequenceFinishReason::MaxTokens, &mut kv)
            .unwrap();
        assert!(matches!(finish, SchedulerAction::Finish { .. }));
        assert_eq!(scheduler.active_len(), 0);
        assert_eq!(scheduler.finished_len(), 1);
        assert_eq!(kv.active_count(), 0);
    }

    #[test]
    fn resident_scheduler_batches_ready_decode_rows() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 4,
            max_active_sequences: 2,
            max_decode_batch: 2,
            ..Default::default()
        });
        let mut kv = FixedSequenceSlotPool::new(2);
        scheduler.submit(request(1, vec![1]));
        scheduler.submit(request(2, vec![2]));
        assert_eq!(scheduler.admit_waiting(&mut kv).unwrap(), 2);

        for session_id in [SessionId(1), SessionId(2)] {
            let action = scheduler.next_prefill_action(&mut kv).unwrap().unwrap();
            let SchedulerAction::PrefillChunk(prefill) = action else {
                panic!("expected prefill");
            };
            assert_eq!(prefill.session_id, session_id);
            scheduler.commit_prefill_action(&prefill).unwrap();
        }

        scheduler.stage_decode_token(SessionId(1), 10).unwrap();
        scheduler.stage_decode_token(SessionId(2), 20).unwrap();
        let Some(SchedulerAction::DecodeBatch(actions)) = scheduler.next_decode_action().unwrap()
        else {
            panic!("expected decode batch");
        };
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0].token_id, 10);
        assert_eq!(actions[1].token_id, 20);
    }

    #[test]
    fn requeue_decode_actions_front_restores_exact_order_and_is_failure_atomic() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 4,
            max_active_sequences: 2,
            max_decode_batch: 2,
            ..Default::default()
        });
        let mut kv = FixedSequenceSlotPool::new(2);
        scheduler.submit(request(1, vec![1]));
        scheduler.submit(request(2, vec![2]));
        assert_eq!(scheduler.admit_waiting(&mut kv).unwrap(), 2);
        for session_id in [SessionId(1), SessionId(2)] {
            let SchedulerAction::PrefillChunk(prefill) =
                scheduler.next_prefill_action(&mut kv).unwrap().unwrap()
            else {
                panic!("expected prefill");
            };
            assert_eq!(prefill.session_id, session_id);
            scheduler.commit_prefill_action(&prefill).unwrap();
        }
        scheduler.stage_decode_token(SessionId(1), 10).unwrap();
        scheduler.stage_decode_token(SessionId(2), 20).unwrap();
        let SchedulerAction::DecodeBatch(actions) =
            scheduler.next_decode_action().unwrap().unwrap()
        else {
            panic!("expected decode batch");
        };

        scheduler.requeue_decode_actions_front(&actions).unwrap();
        let SchedulerAction::DecodeBatch(restored) =
            scheduler.next_decode_action().unwrap().unwrap()
        else {
            panic!("expected restored decode batch");
        };
        assert_eq!(restored, actions);

        let mut invalid = restored;
        invalid[1].position += 1;
        assert!(scheduler.requeue_decode_actions_front(&invalid).is_err());
        assert!(scheduler.next_decode_action().unwrap().is_none());
    }

    #[test]
    fn resident_scheduler_defers_decode_to_form_target_cohort() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 4,
            max_active_sequences: 2,
            max_decode_batch: 2,
            decode_cohort_target: 2,
            decode_cohort_max_deferrals: 1,
            allow_mixed_batches: false,
            ..Default::default()
        });
        let mut kv = FixedSequenceSlotPool::new(2);
        scheduler.submit(request(1, vec![1]));
        scheduler.submit(request(2, vec![2]));

        let first = scheduler.next_action(&mut kv).unwrap().unwrap();
        let SchedulerAction::PrefillChunk(first_prefill) = &first else {
            panic!("expected first prefill");
        };
        assert_eq!(first_prefill.session_id, SessionId(1));
        scheduler.commit_action(&first).unwrap();
        scheduler.stage_decode_token(SessionId(1), 10).unwrap();

        let second = scheduler.next_action(&mut kv).unwrap().unwrap();
        let SchedulerAction::PrefillChunk(second_prefill) = &second else {
            panic!("expected cohort-forming prefill");
        };
        assert_eq!(second_prefill.session_id, SessionId(2));
        assert_eq!(scheduler.decode_cohort_deferrals, 1);
        scheduler.commit_action(&second).unwrap();
        scheduler.stage_decode_token(SessionId(2), 20).unwrap();

        let action = scheduler.next_action(&mut kv).unwrap().unwrap();
        let SchedulerAction::DecodeBatch(decodes) = action else {
            panic!("expected decode cohort");
        };
        assert_eq!(decodes.len(), 2);
        assert_eq!(
            decodes
                .iter()
                .map(|decode| decode.session_id)
                .collect::<Vec<_>>(),
            vec![SessionId(1), SessionId(2)]
        );
        assert_eq!(scheduler.decode_cohort_deferrals, 0);
    }

    #[test]
    fn resident_scheduler_decode_cohort_max_deferrals_forces_progress() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 1,
            max_active_sequences: 2,
            max_decode_batch: 2,
            decode_cohort_target: 2,
            decode_cohort_max_deferrals: 1,
            allow_mixed_batches: false,
            ..Default::default()
        });
        let mut kv = FixedSequenceSlotPool::new(2);
        scheduler.submit(request(1, vec![1]));
        scheduler.submit(request(2, vec![2, 3, 4]));

        let first = scheduler.next_action(&mut kv).unwrap().unwrap();
        scheduler.commit_action(&first).unwrap();
        scheduler.stage_decode_token(SessionId(1), 10).unwrap();

        let deferred = scheduler.next_action(&mut kv).unwrap().unwrap();
        let SchedulerAction::PrefillChunk(prefill) = &deferred else {
            panic!("expected one deferred prefill");
        };
        assert_eq!(prefill.session_id, SessionId(2));
        scheduler.commit_action(&deferred).unwrap();
        assert_eq!(scheduler.prefill_queue_len(), 1);

        let action = scheduler.next_action(&mut kv).unwrap().unwrap();
        let SchedulerAction::DecodeBatch(decodes) = action else {
            panic!("expected forced singleton decode");
        };
        assert_eq!(decodes.len(), 1);
        assert_eq!(decodes[0].session_id, SessionId(1));
        assert_eq!(scheduler.decode_cohort_deferrals, 0);
    }

    #[test]
    fn resident_scheduler_default_decode_cohort_policy_is_eager() {
        let defaults = ResidentSchedulerConfig::default();
        assert_eq!(defaults.decode_cohort_target, 1);
        assert_eq!(defaults.decode_cohort_max_deferrals, 0);

        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 4,
            max_active_sequences: 2,
            max_decode_batch: 2,
            allow_mixed_batches: false,
            ..defaults
        });
        let mut kv = FixedSequenceSlotPool::new(2);
        scheduler.submit(request(1, vec![1]));
        scheduler.submit(request(2, vec![2]));

        let first = scheduler.next_action(&mut kv).unwrap().unwrap();
        scheduler.commit_action(&first).unwrap();
        scheduler.stage_decode_token(SessionId(1), 10).unwrap();

        let action = scheduler.next_action(&mut kv).unwrap().unwrap();
        let SchedulerAction::DecodeBatch(decodes) = action else {
            panic!("expected eager decode");
        };
        assert_eq!(decodes.len(), 1);
        assert_eq!(decodes[0].session_id, SessionId(1));
        assert_eq!(scheduler.prefill_queue_len(), 1);
    }

    #[test]
    fn resident_scheduler_normalizes_decode_cohort_target() {
        let lower = ResidentScheduler::new(ResidentSchedulerConfig {
            max_decode_batch: 2,
            decode_cohort_target: 0,
            ..Default::default()
        });
        assert_eq!(lower.config().decode_cohort_target, 1);

        let upper = ResidentScheduler::new(ResidentSchedulerConfig {
            max_decode_batch: 2,
            decode_cohort_target: 3,
            ..Default::default()
        });
        assert_eq!(upper.config().decode_cohort_target, 2);
    }

    #[test]
    fn native_scheduler_enforces_token_budget_and_mixes_decode_with_ragged_prefill() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 8,
            max_active_sequences: 2,
            max_decode_batch: 2,
            max_batch_tokens: 3,
            allow_mixed_batches: true,
            ..Default::default()
        });
        let mut kv = FixedSequenceSlotPool::new(2);
        scheduler.submit(request(1, vec![1]));
        scheduler.submit(request(2, vec![2, 3, 4, 5, 6]));
        scheduler.admit_waiting(&mut kv).unwrap();

        let first = scheduler.next_prefill_action(&mut kv).unwrap().unwrap();
        let SchedulerAction::PrefillChunk(first_prefill) = &first else {
            panic!("expected initial prefill");
        };
        let decode_session = first_prefill.session_id;
        scheduler.commit_action(&first).unwrap();
        scheduler.stage_decode_token(decode_session, 99).unwrap();

        let action = scheduler.next_action(&mut kv).unwrap().unwrap();
        let SchedulerAction::Execute { prefills, decodes } = action else {
            panic!("expected native execution action");
        };
        assert_eq!(decodes.len(), 1);
        assert_eq!(prefills.len(), 1);
        assert_eq!(prefills[0].tokens.len(), 2);
        assert_eq!(decodes.len() + prefills[0].tokens.len(), 3);
    }

    #[test]
    fn scheduler_fairness_saturated_mixed_batches_progress_both_queues_in_bounded_ticks() {
        for budget in [2, 3, 32] {
            for decode_count in [budget, budget + 8] {
                for reserve in [1, 8, usize::MAX] {
                    let (mut scheduler, slots) = fairness_fixture(
                        ResidentSchedulerConfig {
                            prefill_chunk_size: 8,
                            max_decode_batch: decode_count,
                            max_batch_tokens: budget,
                            prefill_reserve_tokens: reserve,
                            ..Default::default()
                        },
                        decode_count,
                        &[vec![10; 3], vec![20; 3], vec![30; 3]],
                    );
                    let mut expected_decodes = scheduler.decode_ready.clone();
                    // Each tick consumes at least one of the nine prompt tokens.
                    // Decodes stay ready throughout, rather than draining away.
                    for _ in 0..9 {
                        if scheduler.prefill_queue.is_empty() {
                            break;
                        }
                        let expected_prefill = scheduler.prefill_queue[0];
                        let action = scheduler
                            .next_admitted_action_policy(true)
                            .unwrap()
                            .unwrap();
                        let (prefills, decodes) = execution_parts(&action);
                        assert_eq!(prefills[0].session_id, expected_prefill);
                        assert!(!decodes.is_empty());
                        assert_eq!(decodes.len(), budget - reserve.min(budget - 1));
                        for decode in decodes {
                            assert_eq!(Some(decode.session_id), expected_decodes.pop_front());
                            expected_decodes.push_back(decode.session_id);
                        }
                        commit_and_restage_decodes(&mut scheduler, &action);
                        assert_eq!(scheduler.decode_ready, expected_decodes);
                        assert_eq!(slots.active_count(), decode_count + 3);
                        assert_eq!(scheduler.active_len(), decode_count + 3);
                    }
                    assert!(scheduler.prefill_queue.is_empty());
                    for id in decode_count + 1..=decode_count + 3 {
                        let sequence = scheduler.active_sequence(SessionId(id as u64)).unwrap();
                        assert_eq!(sequence.prompt_cursor, 3);
                        assert_eq!(sequence.position, 3);
                    }
                    let action = scheduler
                        .next_admitted_action_policy(true)
                        .unwrap()
                        .unwrap();
                    assert_eq!(execution_parts(&action).1.len(), budget);
                    assert_bounded_fairness_action(&scheduler, &action);
                }
            }
        }
    }

    #[test]
    fn scheduler_fairness_decode_only_keeps_full_budget_and_fifo() {
        for mixed in [false, true] {
            let (mut scheduler, _) = fairness_fixture(
                ResidentSchedulerConfig {
                    max_decode_batch: 40,
                    max_batch_tokens: 32,
                    prefill_reserve_tokens: usize::MAX,
                    ..Default::default()
                },
                40,
                &[],
            );
            for tick in 0..3 {
                let action = scheduler
                    .next_admitted_action_policy(mixed)
                    .unwrap()
                    .unwrap();
                let (prefills, decodes) = execution_parts(&action);
                assert!(prefills.is_empty());
                assert_eq!(decodes.len(), 32);
                for (row, decode) in decodes.iter().enumerate() {
                    assert_eq!(
                        decode.session_id,
                        SessionId(((tick * 32 + row) % 40 + 1) as u64)
                    );
                }
                commit_and_restage_decodes(&mut scheduler, &action);
            }
        }
    }

    #[test]
    fn scheduler_fairness_prefill_only_visits_each_request_once_per_tick() {
        for mixed in [false, true] {
            let (mut scheduler, _) = fairness_fixture(
                ResidentSchedulerConfig {
                    prefill_chunk_size: 2,
                    max_batch_tokens: 32,
                    ..Default::default()
                },
                0,
                &[vec![1; 5], vec![2; 3]],
            );
            for _ in 0..5 {
                if scheduler.prefill_queue.is_empty() {
                    break;
                }
                let front = scheduler.prefill_queue[0];
                let action = scheduler
                    .next_admitted_action_policy(mixed)
                    .unwrap()
                    .unwrap();
                let (prefills, decodes) = execution_parts(&action);
                assert!(decodes.is_empty());
                assert_eq!(prefills[0].session_id, front);
                assert!(prefills.len() <= if mixed { 2 } else { 1 });
                commit_and_restage_decodes(&mut scheduler, &action);
            }
            assert!(scheduler.prefill_queue.is_empty());
            assert!(
                scheduler
                    .next_admitted_action_policy(mixed)
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn scheduler_fairness_preserves_under_budget_and_opt_out_packing() {
        assert_eq!(ResidentSchedulerConfig::default().prefill_reserve_tokens, 1);
        // A larger reserve does not affect unsaturated batches, decode batch
        // limits, or the existing unlimited-budget convention.
        for (budget, max_decodes, reserve, expected_decode, expected_prefill) in [
            (4, 4, 3, 2, 2),
            (2, 1, 9, 1, 1),
            (2, 2, 0, 2, 0),
            (0, 2, usize::MAX, 2, 8),
        ] {
            let (mut scheduler, _) = fairness_fixture(
                ResidentSchedulerConfig {
                    prefill_chunk_size: 8,
                    max_decode_batch: max_decodes,
                    max_batch_tokens: budget,
                    prefill_reserve_tokens: reserve,
                    ..Default::default()
                },
                2,
                &[vec![3; 10]],
            );
            let action = scheduler
                .next_admitted_action_policy(true)
                .unwrap()
                .unwrap();
            assert_bounded_fairness_action(&scheduler, &action);
            let (prefills, decodes) = execution_parts(&action);
            assert_eq!(decodes.len(), expected_decode);
            assert_eq!(
                prefills.iter().map(|p| p.tokens.len()).sum::<usize>(),
                expected_prefill
            );
        }
    }

    #[test]
    fn scheduler_fairness_alternates_when_work_cannot_share_a_batch() {
        for (mixed, budget) in [(true, 1), (false, 1), (false, 32)] {
            let decode_count = budget + 2;
            let (mut scheduler, _) = fairness_fixture(
                ResidentSchedulerConfig {
                    prefill_chunk_size: 1,
                    max_decode_batch: decode_count,
                    max_batch_tokens: budget,
                    ..Default::default()
                },
                decode_count,
                &[vec![10; 3], vec![20; 3]],
            );
            for tick in 0..12 {
                let action = scheduler
                    .next_admitted_action_policy(mixed)
                    .unwrap()
                    .unwrap();
                let (prefills, decodes) = execution_parts(&action);
                if tick % 2 == 0 {
                    assert!(prefills.is_empty());
                    assert_eq!(decodes.len(), budget);
                } else {
                    assert!(decodes.is_empty());
                    assert_eq!(prefills.len(), 1);
                    assert_eq!(
                        prefills[0].session_id,
                        SessionId((decode_count + 1 + (tick / 2) % 2) as u64)
                    );
                }
                commit_and_restage_decodes(&mut scheduler, &action);
            }
            assert!(scheduler.prefill_queue.is_empty());
            assert!(!scheduler.prefill_turn_due);
        }
    }

    #[test]
    fn scheduler_fairness_mixed_cancellation_and_decode_requeue_keep_exact_ownership() {
        let (mut scheduler, mut slots) = fairness_fixture(
            ResidentSchedulerConfig {
                prefill_chunk_size: 4,
                max_decode_batch: 4,
                max_batch_tokens: 2,
                decode_cohort_target: 4,
                decode_cohort_max_deferrals: 3,
                ..Default::default()
            },
            3,
            &[vec![10; 3], vec![20; 3]],
        );
        scheduler.submit(request(6, vec![6]));
        assert_eq!(scheduler.admit_waiting(&mut slots).unwrap(), 0);
        let action = scheduler
            .next_admitted_action_policy(true)
            .unwrap()
            .unwrap();
        assert_bounded_fairness_action(&scheduler, &action);
        let (prefills, decodes) = execution_parts(&action);
        assert_eq!(prefills[0].request_id, Some(RequestId(4)));
        assert_eq!(decodes[0].request_id, Some(RequestId(1)));
        assert_eq!(scheduler.decode_cohort_deferrals, 0);
        // An unexecuted mixed batch can lose its prefill request while its
        // decode actions are restored with their original token/position/slot.
        scheduler.cancel_request(RequestId(4), &mut slots).unwrap();
        scheduler.requeue_decode_actions_front(decodes).unwrap();
        let action = scheduler
            .next_admitted_action_policy(true)
            .unwrap()
            .unwrap();
        let (prefills, restored) = execution_parts(&action);
        assert_eq!(restored, decodes);
        assert_eq!(prefills[0].request_id, Some(RequestId(5)));
        commit_and_restage_decodes(&mut scheduler, &action);
        scheduler.cancel_request(RequestId(2), &mut slots).unwrap();
        assert_eq!(slots.active_count(), 3);
        assert_eq!(scheduler.admit_waiting(&mut slots).unwrap(), 1);
        for _ in 0..3 {
            let action = scheduler
                .next_admitted_action_policy(true)
                .unwrap()
                .unwrap();
            assert_bounded_fairness_action(&scheduler, &action);
            let (prefills, decodes) = execution_parts(&action);
            assert!(!prefills.is_empty());
            assert_eq!(decodes.len(), 1);
            assert!(decodes.iter().all(|d| d.request_id != Some(RequestId(2))));
            assert!(prefills.iter().all(|p| p.request_id != Some(RequestId(4))));
            commit_and_restage_decodes(&mut scheduler, &action);
        }
        assert!(scheduler.prefill_queue.is_empty());
        assert_eq!(slots.active_count(), 4);
        assert_eq!(scheduler.active_len(), 4);
    }

    #[test]
    fn scheduler_fairness_turn_counts_toward_cohort_deferrals() {
        let (mut scheduler, _) = fairness_fixture(
            ResidentSchedulerConfig {
                prefill_chunk_size: 1,
                max_decode_batch: 3,
                max_batch_tokens: 1,
                decode_cohort_target: 3,
                decode_cohort_max_deferrals: 1,
                ..Default::default()
            },
            2,
            &[vec![3; 8]],
        );
        for tick in 0..6 {
            let action = scheduler
                .next_admitted_action_policy(false)
                .unwrap()
                .unwrap();
            let (prefills, decodes) = execution_parts(&action);
            if tick % 2 == 0 {
                assert_eq!(prefills.len(), 1);
                assert!(decodes.is_empty());
                assert_eq!(scheduler.decode_cohort_deferrals, 1);
            } else {
                assert!(prefills.is_empty());
                assert_eq!(decodes.len(), 1);
                assert_eq!(scheduler.decode_cohort_deferrals, 0);
            }
            commit_and_restage_decodes(&mut scheduler, &action);
        }
    }

    #[test]
    fn scheduler_fairness_cancellation_clears_turn_and_preserves_capacity() {
        for mixed in [false, true] {
            let (mut scheduler, mut slots) = fairness_fixture(
                ResidentSchedulerConfig {
                    max_decode_batch: 2,
                    max_batch_tokens: 1,
                    ..Default::default()
                },
                2,
                &[vec![3; 3]],
            );
            let action = scheduler
                .next_admitted_action_policy(mixed)
                .unwrap()
                .unwrap();
            commit_and_restage_decodes(&mut scheduler, &action);
            assert!(scheduler.prefill_turn_due);
            let decode_order = scheduler.decode_ready.clone();
            scheduler.submit(request(4, vec![4]));
            assert_eq!(scheduler.admit_waiting(&mut slots).unwrap(), 0);
            scheduler.cancel_request(RequestId(3), &mut slots).unwrap();
            assert!(!scheduler.prefill_turn_due);
            assert_eq!(slots.active_count(), 2);
            assert_eq!(scheduler.decode_ready, decode_order);
            assert_eq!(scheduler.admit_waiting(&mut slots).unwrap(), 1);
            let action = scheduler
                .next_admitted_action_policy(mixed)
                .unwrap()
                .unwrap();
            assert_eq!(execution_parts(&action).1[0].session_id, decode_order[0]);
            commit_and_restage_decodes(&mut scheduler, &action);
            scheduler.cancel_request(RequestId(1), &mut slots).unwrap();
            let action = scheduler
                .next_admitted_action_policy(mixed)
                .unwrap()
                .unwrap();
            assert_eq!(execution_parts(&action).0[0].request_id, Some(RequestId(4)));
            commit_and_restage_decodes(&mut scheduler, &action);
            scheduler.cancel_request(RequestId(2), &mut slots).unwrap();
            scheduler.cancel_request(RequestId(4), &mut slots).unwrap();
            assert_eq!(slots.active_count(), 0);
            assert!(scheduler.is_idle());
            assert!(
                scheduler
                    .next_admitted_action_policy(mixed)
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn resident_scheduler_cancel_frees_kv_and_drains_cancelled() {
        let mut scheduler = ResidentScheduler::default();
        let mut kv = FixedSequenceSlotPool::new(1);
        scheduler.submit(request(7, vec![1]));
        assert!(scheduler.next_prefill_action(&mut kv).unwrap().is_some());
        assert_eq!(kv.active_count(), 1);

        let action = scheduler.cancel_sequence(SessionId(1), &mut kv).unwrap();
        assert!(matches!(action, SchedulerAction::Cancel { .. }));
        assert_eq!(kv.active_count(), 0);
        assert_eq!(scheduler.cancelled_len(), 1);
        let cancelled = scheduler.drain_cancelled();
        assert_eq!(
            cancelled[0].finish_reason,
            Some(SequenceFinishReason::Cancelled)
        );
        assert_eq!(cancelled[0].kv_handle, None);
        assert!(scheduler.is_idle());
    }

    #[test]
    fn take_request_terminal_removes_only_the_exact_request_and_preserves_outcome() {
        let mut scheduler = ResidentScheduler::default();
        let mut finished = SequenceState::from_request(&request(1, vec![1]), SessionId(1));
        finished.mark_finished(SequenceFinishReason::MaxTokens);
        let mut cancelled = SequenceState::from_request(&request(2, vec![2]), SessionId(2));
        cancelled.mark_cancelled();
        let mut failed = SequenceState::from_request(&request(3, vec![3]), SessionId(3));
        failed.mark_error();
        scheduler.finished.push(finished);
        scheduler.cancelled.push(cancelled);
        scheduler.failed.push(failed);

        assert!(matches!(
            scheduler.take_request_terminal(RequestId(2)),
            Some(RequestTerminal::Cancelled(sequence))
                if sequence.request_id == Some(RequestId(2))
        ));
        assert!(scheduler.take_request_terminal(RequestId(99)).is_none());
        assert!(matches!(
            scheduler.take_request_terminal(RequestId(1)),
            Some(RequestTerminal::Finished(sequence))
                if sequence.request_id == Some(RequestId(1))
        ));
        assert!(matches!(
            scheduler.take_request_terminal(RequestId(3)),
            Some(RequestTerminal::Failed(sequence))
                if sequence.request_id == Some(RequestId(3))
        ));
        assert!(scheduler.finished.is_empty());
        assert!(scheduler.cancelled.is_empty());
        assert!(scheduler.failed.is_empty());
    }

    #[test]
    fn cancel_request_distinguishes_waiting_active_and_unknown() {
        let mut scheduler = ResidentScheduler::default();
        let mut kv = FixedSequenceSlotPool::new(1);
        scheduler.submit(request(1, vec![1]));
        scheduler.submit(request(2, vec![2]));
        assert_eq!(scheduler.admit_waiting(&mut kv).unwrap(), 1);

        assert_eq!(
            scheduler.cancel_request(RequestId(2), &mut kv).unwrap(),
            CancelRequestResult::Waiting {
                request_id: RequestId(2),
                session_id: SessionId(2),
            }
        );
        assert_eq!(scheduler.waiting_len(), 0);
        assert_eq!(scheduler.active_len(), 1);
        assert_eq!(kv.active_count(), 1);

        assert_eq!(
            scheduler.cancel_request(RequestId(1), &mut kv).unwrap(),
            CancelRequestResult::Active {
                request_id: RequestId(1),
                session_id: SessionId(1),
            }
        );
        assert_eq!(scheduler.active_len(), 0);
        assert_eq!(kv.active_count(), 0);
        assert_eq!(
            scheduler.cancel_request(RequestId(99), &mut kv).unwrap(),
            CancelRequestResult::NotFound {
                request_id: RequestId(99),
            }
        );

        let cancelled = scheduler.drain_cancelled();
        assert_eq!(cancelled.len(), 2);
        assert_eq!(cancelled[0].request_id, Some(RequestId(2)));
        assert_eq!(cancelled[1].request_id, Some(RequestId(1)));
        assert!(cancelled.iter().all(|sequence| {
            sequence.status == super::super::session::SequenceStatus::Cancelled
                && sequence.finish_reason == Some(SequenceFinishReason::Cancelled)
        }));

        scheduler.submit(request(3, vec![3]));
        assert_eq!(scheduler.admit_waiting(&mut kv).unwrap(), 1);
        assert_eq!(scheduler.active_len(), 1);
    }

    #[test]
    fn greedy_decode_stages_full_logits_argmax() {
        let mut scheduler = ResidentScheduler::default();
        let mut kv = FixedSequenceSlotPool::new(1);
        scheduler.submit(request(1, vec![1]));
        let SchedulerAction::PrefillChunk(prefill) = scheduler
            .next_prefill_action(&mut kv)
            .unwrap()
            .expect("prefill")
        else {
            panic!("expected prefill");
        };
        scheduler.commit_prefill_action(&prefill).unwrap();

        let logits = LogitsOutput::Full(vec![0.1, 2.0, 1.5]);
        assert!(
            scheduler
                .stage_greedy_decode_from_logits(SessionId(1), &logits)
                .unwrap()
        );
        let Some(SchedulerAction::DecodeBatch(actions)) = scheduler.next_decode_action().unwrap()
        else {
            panic!("expected decode batch");
        };
        assert_eq!(actions[0].token_id, 1);
    }

    #[test]
    fn prepared_admission_is_invisible_and_abort_restores_the_request() {
        let mut scheduler = ResidentScheduler::default();
        let mut slots = FixedSequenceSlotPool::new(1);
        let mut submitted = request(30, vec![1, 2]);
        submitted.session_id = Some(SessionId(30));
        scheduler.submit(submitted);

        let prepared = scheduler
            .prepare_waiting_admission(&mut slots)
            .unwrap()
            .expect("waiting request should reserve a slot");
        assert_eq!(prepared.session_id(), SessionId(30));
        assert_eq!(scheduler.waiting_len(), 0);
        assert_eq!(scheduler.active_len(), 0);
        assert_eq!(slots.active_count(), 1);

        scheduler
            .abort_waiting_admission(prepared, &mut slots)
            .unwrap();
        assert_eq!(scheduler.waiting_len(), 1);
        assert_eq!(scheduler.active_len(), 0);
        assert_eq!(slots.active_count(), 0);

        assert_eq!(scheduler.admit_waiting(&mut slots).unwrap(), 1);
        assert!(scheduler.active_sequence(SessionId(30)).is_some());
    }

    #[test]
    fn failed_prepared_admission_abort_retains_slot_ownership() {
        let mut scheduler = ResidentScheduler::default();
        let mut slots = FailingFreeKvCache;
        scheduler.submit(request(31, vec![1]));

        let prepared = scheduler
            .prepare_waiting_admission(&mut slots)
            .unwrap()
            .expect("waiting request should reserve a slot");
        let error = scheduler
            .abort_waiting_admission(prepared, &mut slots)
            .unwrap_err();

        assert!(error.to_string().contains("slot free failure"));
        assert_eq!(scheduler.waiting_len(), 0);
        assert_eq!(scheduler.active_len(), 0);
        assert_eq!(scheduler.failed_len(), 1);
        assert_eq!(scheduler.failed[0].kv_handle, Some(KvHandle(0)));
        assert!(scheduler.drain_failed().is_empty());
        assert_eq!(scheduler.failed_slot_ownership(), 1);
    }

    #[test]
    fn finish_and_cancel_preserve_sequence_ownership_when_slot_free_fails() {
        for cancel in [false, true] {
            let mut scheduler = ResidentScheduler::default();
            let mut kv = FailingFreeKvCache;
            scheduler.submit(request(if cancel { 21 } else { 20 }, vec![1]));
            assert!(scheduler.next_prefill_action(&mut kv).unwrap().is_some());

            let result = if cancel {
                scheduler.cancel_sequence(SessionId(1), &mut kv)
            } else {
                scheduler.finish_sequence(SessionId(1), SequenceFinishReason::MaxTokens, &mut kv)
            };
            assert!(format!("{}", result.unwrap_err()).contains("slot free failure"));
            assert_eq!(scheduler.active_len(), 0);
            assert_eq!(scheduler.failed_len(), 1);
            assert_eq!(
                scheduler.failed[0].status,
                super::super::session::SequenceStatus::Error
            );
            assert_eq!(scheduler.failed[0].kv_handle, Some(KvHandle(0)));
            assert!(scheduler.drain_failed().is_empty());
            assert_eq!(scheduler.failed_slot_ownership(), 1);
        }
    }

    #[test]
    fn resident_scheduler_waits_when_kv_is_full() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 4,
            max_active_sequences: 2,
            max_decode_batch: 2,
            ..Default::default()
        });
        let mut kv = FixedSequenceSlotPool::new(1);
        scheduler.submit(request(1, vec![1]));
        scheduler.submit(request(2, vec![2]));
        assert_eq!(scheduler.admit_waiting(&mut kv).unwrap(), 1);
        assert_eq!(scheduler.waiting_len(), 1);
        assert_eq!(scheduler.active_len(), 1);
    }
    #[test]
    fn pr12_identity_queries_cover_waiting_active_and_all_terminals() {
        for kind in ["finished", "cancelled", "failed"] {
            let mut scheduler = ResidentScheduler::default();
            let mut slots = FixedSequenceSlotPool::new(1);
            let mut request = request(12, vec![1]);
            request.session_id = Some(SessionId(12));
            scheduler.submit(request);
            assert!(scheduler.contains_request_identity(RequestId(12)));
            assert!(scheduler.contains_session_identity(SessionId(12)));
            scheduler.admit_waiting(&mut slots).unwrap();
            assert!(scheduler.contains_request_identity(RequestId(12)));
            match kind {
                "finished" => {
                    scheduler
                        .finish_sequence(SessionId(12), SequenceFinishReason::MaxTokens, &mut slots)
                        .unwrap();
                }
                "cancelled" => {
                    scheduler
                        .cancel_sequence(SessionId(12), &mut slots)
                        .unwrap();
                }
                _ => scheduler.fail_sequence(SessionId(12), &mut slots).unwrap(),
            }
            assert!(scheduler.contains_request_identity(RequestId(12)));
            assert!(scheduler.contains_session_identity(SessionId(12)));
            assert_eq!(scheduler.identity_sessions(), vec![SessionId(12)]);
            assert!(scheduler.take_request_terminal(RequestId(12)).is_some());
            assert!(!scheduler.contains_request_identity(RequestId(12)));
            assert!(!scheduler.contains_session_identity(SessionId(12)));
        }
    }
}
