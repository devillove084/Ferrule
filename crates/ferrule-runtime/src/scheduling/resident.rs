//! Resident request/session scheduler.
//!
//! This module ties together the runtime vocabulary that was previously adjacent
//! but not connected: `GenerateRequest`, `SequenceState`, `SequenceSlotPool`,
//! `SchedulerAction`, neutral execution lowering, and runtime output correlation.
//!
//! It is intentionally synchronous and single-process. It does not execute a
//! model and does not know concrete model families; it only decides which
//! resident sequence should prefill/decode next and owns slot allocation lifecycle.

use std::collections::{BTreeMap, VecDeque};

use ferrule_common::execution::{LogitsOutput, TokenLogit};
use ferrule_common::{Error, Result};

use super::actions::{DecodeAction, PrefillChunkAction, SchedulerAction, plan_prefill_chunk};
use super::expert_io::{
    ExpertIoAdvisor, ExpertIoBatchUsage, ExpertIoBudget, ExpertIoCandidate, ExpertIoDecisionTrace,
    ExpertIoPhase, ExpertIoQueueClass, ZeroExpertIoAdvisor, classify_admitted,
};
use super::session::{GenerateRequest, RequestId, SequenceFinishReason, SequenceState, SessionId};
use super::{KvHandle, SequenceSlotPool};

#[derive(Debug, Clone)]
struct WaitingRequest {
    request: GenerateRequest,
    position_start: Option<usize>,
}

struct BlockedExpertCandidate<T, A> {
    action: T,
    admission: A,
    trace_index: usize,
}

enum ExpertIoCandidateDecision<A> {
    Admitted,
    Blocked { admission: A, trace_index: usize },
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
    /// Maximum consecutive prefill-only decisions allowed while a non-empty
    /// decode cohort is below target. Zero preserves eager decode dispatch.
    pub decode_cohort_max_deferrals: usize,
    /// Maximum total packed tokens in one execution batch. Zero means no limit.
    /// This bounds the combined prefill + decode token count per batch.
    pub max_batch_tokens: usize,
    /// When true, the scheduler may combine prefill and decode sequences into
    /// one mixed execution batch. When false, prefill and decode are dispatched
    /// as separate batches.
    pub allow_mixed_batches: bool,
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
            allow_mixed_batches: true,
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
            allow_mixed_batches: self.allow_mixed_batches,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SuspendedSequenceSchedule {
    sequence: SequenceState,
    was_prefill_ready: bool,
    was_decode_ready: bool,
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
    pub fn session_id(&self) -> SessionId {
        self.sequence.session_id
    }
}

#[derive(Debug)]
pub struct ResidentScheduler {
    config: ResidentSchedulerConfig,
    waiting: VecDeque<WaitingRequest>,
    active: BTreeMap<SessionId, SequenceState>,
    prefill_queue: VecDeque<SessionId>,
    decode_ready: VecDeque<SessionId>,
    decode_cohort_deferrals: usize,
    finished: Vec<SequenceState>,
    cancelled: Vec<SequenceState>,
    failed: Vec<SequenceState>,
    expert_io_trace: Vec<ExpertIoDecisionTrace>,
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
            active: BTreeMap::new(),
            prefill_queue: VecDeque::new(),
            decode_ready: VecDeque::new(),
            decode_cohort_deferrals: 0,
            finished: Vec::new(),
            cancelled: Vec::new(),
            failed: Vec::new(),
            expert_io_trace: Vec::new(),
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

    pub fn total_submitted(&self) -> u64 {
        self.total_submitted
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn active_len(&self) -> usize {
        self.active.len()
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

    /// Expert-I/O decisions made while constructing the most recent batch.
    pub fn expert_io_trace(&self) -> &[ExpertIoDecisionTrace] {
        &self.expert_io_trace
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
            return Err(Error::Execution(
                "fork source and target sessions must differ".into(),
            ));
        }
        if request.session_id != Some(target_session_id) {
            return Err(Error::Execution(format!(
                "fork target request must name target session {target_session_id:?}"
            )));
        }
        if self.active.contains_key(&target_session_id)
            || self
                .waiting
                .iter()
                .any(|waiting| waiting.request.session_id == Some(target_session_id))
        {
            return Err(Error::Execution(format!(
                "fork target session {target_session_id:?} already exists"
            )));
        }
        if self.active.len() >= self.config.max_active_sequences {
            return Err(Error::Execution(
                "resident scheduler has no active-sequence capacity for fork target".into(),
            ));
        }
        let source = self.active.get(&source_session_id).ok_or_else(|| {
            Error::Execution(format!(
                "fork source session {source_session_id:?} is not active"
            ))
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
            return Err(Error::Internal(format!(
                "cannot restore already-active resident session {session_id:?}"
            )));
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

        Ok(CancelRequestResult::NotFound { request_id })
    }

    pub fn drain_finished(&mut self) -> Vec<SequenceState> {
        std::mem::take(&mut self.finished)
    }

    pub fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        std::mem::take(&mut self.cancelled)
    }

    pub fn drain_failed(&mut self) -> Vec<SequenceState> {
        std::mem::take(&mut self.failed)
    }

    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty()
            && self.active.is_empty()
            && self.prefill_queue.is_empty()
            && self.decode_ready.is_empty()
    }

    pub fn admit_waiting<C>(&mut self, slot_pool: &mut C) -> Result<usize>
    where
        C: SequenceSlotPool,
    {
        let mut admitted = 0;
        while self.active.len() < self.config.max_active_sequences {
            let Some(waiting) = self.waiting.pop_front() else {
                break;
            };
            let session_id = self.resolve_session_id(waiting.request.session_id);
            if self.active.contains_key(&session_id) {
                self.waiting.push_front(waiting);
                return Err(Error::Internal(format!(
                    "session {:?} is already active",
                    session_id
                )));
            }

            let kv_handle = match slot_pool.alloc_slot() {
                Ok(handle) => handle,
                Err(_) => {
                    self.waiting.push_front(waiting);
                    break;
                }
            };

            let mut sequence = SequenceState::from_request(&waiting.request, session_id);
            if let Some(position_start) = waiting.position_start {
                sequence.position = position_start;
            }
            sequence.bind_kv(kv_handle);
            if !sequence.prompt_prefill_done() {
                self.prefill_queue.push_back(sequence.session_id);
            }
            self.active.insert(sequence.session_id, sequence);
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
            return plan_prefill_chunk(sequence, self.config.prefill_chunk_size);
        }

        Ok(None)
    }

    /// Pick the next executable action, preferring ready decode work over new
    /// prefill chunks for token latency. Callers that need stricter prefill-first
    /// behavior can keep using `next_prefill_action` and `next_decode_action`
    /// directly.
    pub fn next_action<C>(&mut self, slot_pool: &mut C) -> Result<Option<SchedulerAction>>
    where
        C: SequenceSlotPool,
    {
        self.next_action_with_expert_io(
            slot_pool,
            &mut ZeroExpertIoAdvisor,
            ExpertIoBudget::unbounded(),
        )
    }

    /// Build the next request-centric batch under both token and expert-I/O
    /// budgets. The advisor is model-neutral; a zero-cost advisor exactly
    /// preserves the ordinary scheduler policy.
    pub fn next_action_with_expert_io<C, A>(
        &mut self,
        slot_pool: &mut C,
        advisor: &mut A,
        expert_budget: ExpertIoBudget,
    ) -> Result<Option<SchedulerAction>>
    where
        C: SequenceSlotPool,
        A: ExpertIoAdvisor,
    {
        self.expert_io_trace.clear();
        if !A::ENABLED {
            return self.next_admitted_action(slot_pool, advisor, expert_budget);
        }

        advisor.begin_batch();
        let decode_ready = self.decode_ready.clone();
        let prefill_queue = self.prefill_queue.clone();
        match self.next_admitted_action(slot_pool, advisor, expert_budget) {
            Ok(action) => Ok(action),
            Err(error) => {
                self.decode_ready = decode_ready;
                self.prefill_queue = prefill_queue;
                Err(error)
            }
        }
    }

    fn admit_expert_io_candidate<A: ExpertIoAdvisor>(
        &mut self,
        advisor: &mut A,
        budget: ExpertIoBudget,
        usage: &mut ExpertIoBatchUsage,
        candidate: ExpertIoCandidate<'_>,
    ) -> Result<ExpertIoCandidateDecision<A::Admission>> {
        if !A::ENABLED {
            return Ok(ExpertIoCandidateDecision::Admitted);
        }
        let (estimate, admission) = advisor.estimate(candidate)?;
        let rejection = usage.inspect(budget, estimate);
        let admitted = rejection.is_none();
        let trace_index = self.expert_io_trace.len();
        self.expert_io_trace.push(ExpertIoDecisionTrace {
            session_id: candidate.session_id,
            phase: candidate.phase,
            queue: if admitted {
                classify_admitted(candidate.phase, estimate)
            } else {
                ExpertIoQueueClass::MissBlocked
            },
            admitted,
            forced_progress: false,
            estimate,
            rejection,
        });
        if admitted {
            usage.admit(estimate);
            advisor.admit(admission);
            Ok(ExpertIoCandidateDecision::Admitted)
        } else {
            Ok(ExpertIoCandidateDecision::Blocked {
                admission,
                trace_index,
            })
        }
    }

    fn next_admitted_action<C, A>(
        &mut self,
        slot_pool: &mut C,
        advisor: &mut A,
        expert_budget: ExpertIoBudget,
    ) -> Result<Option<SchedulerAction>>
    where
        C: SequenceSlotPool,
        A: ExpertIoAdvisor,
    {
        self.admit_waiting(slot_pool)?;
        let defer_decode = !self.config.allow_mixed_batches
            && !self.decode_ready.is_empty()
            && self.decode_ready.len() < self.config.decode_cohort_target
            && !self.prefill_queue.is_empty()
            && self.decode_cohort_deferrals < self.config.decode_cohort_max_deferrals;
        let token_budget = if self.config.max_batch_tokens == 0 {
            usize::MAX
        } else {
            self.config.max_batch_tokens
        };
        let mut remaining = token_budget;
        let mut expert_usage = ExpertIoBatchUsage::default();
        let mut decodes = Vec::new();
        let mut blocked_decode = None;
        let decode_candidates = if defer_decode {
            0
        } else {
            self.decode_ready.len()
        };
        for _ in 0..decode_candidates {
            if remaining == 0 || decodes.len() >= self.config.max_decode_batch {
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
            let action = DecodeAction::from_sequence(sequence, token_id);
            let decision = match self.admit_expert_io_candidate(
                advisor,
                expert_budget,
                &mut expert_usage,
                ExpertIoCandidate {
                    session_id,
                    phase: ExpertIoPhase::Decode,
                    token_ids: std::slice::from_ref(&token_id),
                },
            ) {
                Ok(decision) => decision,
                Err(error) => {
                    self.decode_ready.push_front(session_id);
                    return Err(error);
                }
            };
            if let ExpertIoCandidateDecision::Blocked {
                admission,
                trace_index,
            } = decision
            {
                self.decode_ready.push_back(session_id);
                if blocked_decode.is_none() {
                    blocked_decode = Some(BlockedExpertCandidate {
                        action,
                        admission,
                        trace_index,
                    });
                }
                continue;
            }
            decodes.push(action);
            remaining -= 1;
        }

        let mut prefills = Vec::new();
        let mut blocked_prefill = None;
        if self.config.allow_mixed_batches || decodes.is_empty() {
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
                let decision = match self.admit_expert_io_candidate(
                    advisor,
                    expert_budget,
                    &mut expert_usage,
                    ExpertIoCandidate {
                        session_id,
                        phase: ExpertIoPhase::Prefill,
                        token_ids: &action.tokens,
                    },
                ) {
                    Ok(decision) => decision,
                    Err(error) => {
                        self.prefill_queue.push_front(session_id);
                        return Err(error);
                    }
                };
                self.prefill_queue.push_back(session_id);
                if let ExpertIoCandidateDecision::Blocked {
                    admission,
                    trace_index,
                } = decision
                {
                    if blocked_prefill.is_none() {
                        blocked_prefill = Some(BlockedExpertCandidate {
                            action,
                            admission,
                            trace_index,
                        });
                    }
                    continue;
                }
                remaining -= action.tokens.len();
                prefills.push(action);
                if !self.config.allow_mixed_batches {
                    break;
                }
            }
        }

        if prefills.is_empty()
            && decodes.is_empty()
            && A::ENABLED
            && expert_budget.allow_singleton_overflow
        {
            if let Some(blocked) = blocked_decode {
                self.remove_from_queue(blocked.action.session_id, QueueKind::Decode);
                advisor.admit(blocked.admission);
                let trace = &mut self.expert_io_trace[blocked.trace_index];
                trace.admitted = true;
                trace.forced_progress = true;
                decodes.push(blocked.action);
            } else if let Some(blocked) = blocked_prefill {
                advisor.admit(blocked.admission);
                let trace = &mut self.expert_io_trace[blocked.trace_index];
                trace.admitted = true;
                trace.forced_progress = true;
                prefills.push(blocked.action);
            }
        }

        if prefills.is_empty() && decodes.is_empty() {
            if self.decode_ready.is_empty() {
                self.decode_cohort_deferrals = 0;
            }
            Ok(None)
        } else if !self.config.allow_mixed_batches {
            if decodes.is_empty() {
                let action = prefills.pop().map(SchedulerAction::PrefillChunk);
                if action.is_some() && defer_decode {
                    self.decode_cohort_deferrals = self.decode_cohort_deferrals.saturating_add(1);
                } else if self.decode_ready.is_empty() {
                    self.decode_cohort_deferrals = 0;
                }
                Ok(action)
            } else {
                self.decode_cohort_deferrals = 0;
                Ok(Some(SchedulerAction::DecodeBatch(decodes)))
            }
        } else {
            Ok(Some(SchedulerAction::Execute { prefills, decodes }))
        }
    }

    pub fn commit_prefill_action(&mut self, action: &PrefillChunkAction) -> Result<()> {
        let sequence = self.active.get_mut(&action.session_id).ok_or_else(|| {
            Error::Internal(format!(
                "cannot commit prefill for inactive session {:?}",
                action.session_id
            ))
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
        let sequence = self.active.get_mut(&session_id).ok_or_else(|| {
            Error::Internal(format!(
                "cannot stage decode token for inactive session {:?}",
                session_id
            ))
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
        let sequence = self.active.get_mut(&action.session_id).ok_or_else(|| {
            Error::Internal(format!(
                "cannot commit decode for inactive session {:?}",
                action.session_id
            ))
        })?;
        if sequence.position != action.position {
            return Err(Error::Internal(format!(
                "decode position mismatch for session {:?}: sequence at {}, action at {}",
                action.session_id, sequence.position, action.position
            )));
        }
        if sequence.kv_handle != action.kv_handle {
            return Err(Error::Internal(format!(
                "decode KV handle mismatch for session {:?}: sequence {:?}, action {:?}",
                action.session_id, sequence.kv_handle, action.kv_handle
            )));
        }
        sequence.commit_staged_decode_token(action.token_id)
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
        self.active.remove(&session_id).ok_or_else(|| {
            Error::Internal(format!(
                "cannot remove inactive resident session {:?}",
                session_id
            ))
        })
    }

    fn remove_from_queue(&mut self, session_id: SessionId, queue: QueueKind) {
        let target = match queue {
            QueueKind::Prefill => &mut self.prefill_queue,
            QueueKind::Decode => &mut self.decode_ready,
        };
        target.retain(|queued| *queued != session_id);
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

    struct FailingFreeKvCache;

    struct TokenCostAdvisor;

    #[derive(Default)]
    struct FailSecondAdvisor {
        calls: usize,
    }

    #[derive(Default)]
    struct UnionAdvisor {
        admitted_tokens: Vec<u32>,
        begin_calls: usize,
    }

    impl ExpertIoAdvisor for UnionAdvisor {
        type Admission = u32;

        fn begin_batch(&mut self) {
            self.begin_calls += 1;
            self.admitted_tokens.clear();
        }

        fn estimate(
            &mut self,
            candidate: ExpertIoCandidate<'_>,
        ) -> Result<(super::super::expert_io::ExpertIoEstimate, Self::Admission)> {
            let token = candidate.token_ids[0];
            let incremental = if self.admitted_tokens.contains(&token) {
                0
            } else {
                64
            };
            Ok((
                super::super::expert_io::ExpertIoEstimate {
                    incremental_unique_bytes: incremental,
                    predicted_cold_bytes: incremental,
                    ..Default::default()
                },
                token,
            ))
        }

        fn admit(&mut self, token: Self::Admission) {
            if !self.admitted_tokens.contains(&token) {
                self.admitted_tokens.push(token);
            }
        }
    }

    impl ExpertIoAdvisor for FailSecondAdvisor {
        type Admission = ();

        fn begin_batch(&mut self) {
            self.calls = 0;
        }

        fn estimate(
            &mut self,
            _candidate: ExpertIoCandidate<'_>,
        ) -> Result<(super::super::expert_io::ExpertIoEstimate, Self::Admission)> {
            self.calls += 1;
            if self.calls == 2 {
                Err(Error::Internal("simulated advisor failure".into()))
            } else {
                Ok((Default::default(), ()))
            }
        }

        fn admit(&mut self, _admission: Self::Admission) {}
    }

    impl ExpertIoAdvisor for TokenCostAdvisor {
        type Admission = ();

        fn begin_batch(&mut self) {}

        fn estimate(
            &mut self,
            candidate: ExpertIoCandidate<'_>,
        ) -> Result<(super::super::expert_io::ExpertIoEstimate, Self::Admission)> {
            let cold = candidate.token_ids.first().copied() == Some(10);
            Ok((
                super::super::expert_io::ExpertIoEstimate {
                    incremental_unique_bytes: if cold { 128 } else { 0 },
                    predicted_cold_bytes: if cold { 128 } else { 0 },
                    ..Default::default()
                },
                (),
            ))
        }

        fn admit(&mut self, _admission: Self::Admission) {}
    }

    impl SequenceSlotPool for FailingFreeKvCache {
        fn alloc_slot(&mut self) -> Result<KvHandle> {
            Ok(KvHandle(0))
        }

        fn free_slot(&mut self, _handle: KvHandle) -> Result<()> {
            Err(Error::Internal("simulated slot free failure".into()))
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
        assert!(scheduler.expert_io_trace().is_empty());
    }

    #[test]
    fn expert_io_budget_skips_blocked_decode_without_stalling_resident_work() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 4,
            max_active_sequences: 2,
            max_decode_batch: 2,
            max_batch_tokens: 2,
            allow_mixed_batches: true,
            ..Default::default()
        });
        let mut kv = FixedSequenceSlotPool::new(2);
        scheduler.submit(request(1, vec![1]));
        scheduler.submit(request(2, vec![2]));
        scheduler.admit_waiting(&mut kv).unwrap();
        for _ in 0..2 {
            let action = scheduler.next_prefill_action(&mut kv).unwrap().unwrap();
            scheduler.commit_action(&action).unwrap();
        }
        scheduler.stage_decode_token(SessionId(1), 10).unwrap();
        scheduler.stage_decode_token(SessionId(2), 20).unwrap();

        let action = scheduler
            .next_action_with_expert_io(
                &mut kv,
                &mut TokenCostAdvisor,
                ExpertIoBudget {
                    max_incremental_expert_bytes: 0,
                    ..ExpertIoBudget::unbounded()
                },
            )
            .unwrap()
            .unwrap();
        let SchedulerAction::Execute { decodes, prefills } = action else {
            panic!("expected native execution action");
        };
        assert!(prefills.is_empty());
        assert_eq!(decodes.len(), 1);
        assert_eq!(decodes[0].token_id, 20);
        assert_eq!(scheduler.expert_io_trace().len(), 2);
        assert_eq!(
            scheduler.expert_io_trace()[0].queue,
            ExpertIoQueueClass::MissBlocked
        );
        assert_eq!(
            scheduler.expert_io_trace()[1].queue,
            ExpertIoQueueClass::ResidentReady
        );
        assert_eq!(scheduler.decode_ready_len(), 1);
    }

    #[test]
    fn expert_io_advisor_error_restores_all_runnable_queues() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 4,
            max_active_sequences: 2,
            max_decode_batch: 2,
            max_batch_tokens: 2,
            allow_mixed_batches: true,
            ..Default::default()
        });
        let mut kv = FixedSequenceSlotPool::new(2);
        scheduler.submit(request(1, vec![1]));
        scheduler.submit(request(2, vec![2]));
        scheduler.admit_waiting(&mut kv).unwrap();
        for _ in 0..2 {
            let action = scheduler.next_prefill_action(&mut kv).unwrap().unwrap();
            scheduler.commit_action(&action).unwrap();
        }
        scheduler.stage_decode_token(SessionId(1), 10).unwrap();
        scheduler.stage_decode_token(SessionId(2), 20).unwrap();

        let error = scheduler
            .next_action_with_expert_io(
                &mut kv,
                &mut FailSecondAdvisor::default(),
                ExpertIoBudget::unbounded(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("simulated advisor failure"));
        assert_eq!(scheduler.decode_ready_len(), 2);

        let action = scheduler.next_action(&mut kv).unwrap().unwrap();
        let SchedulerAction::Execute { decodes, .. } = action else {
            panic!("expected restored decode batch");
        };
        assert_eq!(
            decodes
                .iter()
                .map(|decode| decode.session_id)
                .collect::<Vec<_>>(),
            vec![SessionId(1), SessionId(2)]
        );
    }

    #[test]
    fn expert_io_advisor_commits_union_only_for_admitted_candidates() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 4,
            max_active_sequences: 2,
            max_decode_batch: 2,
            max_batch_tokens: 2,
            allow_mixed_batches: true,
            ..Default::default()
        });
        let mut kv = FixedSequenceSlotPool::new(2);
        scheduler.submit(request(1, vec![1]));
        scheduler.submit(request(2, vec![2]));
        scheduler.admit_waiting(&mut kv).unwrap();
        for _ in 0..2 {
            let action = scheduler.next_prefill_action(&mut kv).unwrap().unwrap();
            scheduler.commit_action(&action).unwrap();
        }
        scheduler.stage_decode_token(SessionId(1), 10).unwrap();
        scheduler.stage_decode_token(SessionId(2), 10).unwrap();

        let mut advisor = UnionAdvisor::default();
        let action = scheduler
            .next_action_with_expert_io(
                &mut kv,
                &mut advisor,
                ExpertIoBudget {
                    max_incremental_expert_bytes: 64,
                    allow_singleton_overflow: false,
                    ..ExpertIoBudget::unbounded()
                },
            )
            .unwrap()
            .unwrap();
        let SchedulerAction::Execute { decodes, .. } = action else {
            panic!("expected native execution action");
        };
        assert_eq!(decodes.len(), 2);
        assert_eq!(advisor.begin_calls, 1);
        assert_eq!(advisor.admitted_tokens, vec![10]);
        assert_eq!(
            scheduler.expert_io_trace()[0]
                .estimate
                .incremental_unique_bytes,
            64
        );
        assert_eq!(
            scheduler.expert_io_trace()[1]
                .estimate
                .incremental_unique_bytes,
            0
        );
    }

    #[test]
    fn expert_io_budget_forces_singleton_progress_when_all_work_is_blocked() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            max_batch_tokens: 1,
            ..Default::default()
        });
        let mut kv = FixedSequenceSlotPool::new(1);
        scheduler.submit(request(1, vec![10]));

        let action = scheduler
            .next_action_with_expert_io(
                &mut kv,
                &mut TokenCostAdvisor,
                ExpertIoBudget {
                    max_incremental_expert_bytes: 0,
                    ..ExpertIoBudget::unbounded()
                },
            )
            .unwrap();
        assert!(action.is_some());
        assert_eq!(scheduler.expert_io_trace().len(), 1);
        assert!(scheduler.expert_io_trace()[0].admitted);
        assert!(scheduler.expert_io_trace()[0].forced_progress);
        assert_eq!(
            scheduler.expert_io_trace()[0].queue,
            ExpertIoQueueClass::MissBlocked
        );
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
            let failed = scheduler.drain_failed();
            assert_eq!(
                failed[0].status,
                super::super::session::SequenceStatus::Error
            );
            assert_eq!(failed[0].kv_handle, Some(KvHandle(0)));
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
}
