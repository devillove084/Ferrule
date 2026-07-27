use std::collections::{HashMap, HashSet, VecDeque};
use std::num::NonZeroU32;
use std::time::Instant;

use ferrule_common::execution::{
    ExecutionOutput, ExecutionTransactionId, KvBindingMode, KvReservationView, StateSlot,
};
use ferrule_common::{Error, Result};
use ferrule_model::{
    BatchContinuationCancelOutcome, MultiSessionRunner, NativeProposal, NativeProposalProgress,
    NativeProposalSource, PendingModelProgress, ResidentModelRunner,
};
use tracing;

use crate::cache::{
    KvPageManager, KvReservation, KvReservationCommit, KvRetirement, PreemptedKvState,
    PreparedKvCommit,
};
use crate::scheduling::resident::{SuspendedSequenceSchedule, greedy_candidate};
use crate::scheduling::{
    BrokerExpertIoResourceControl, CancelRequestResult, DecodeAction, ExpertIoBudget,
    GenerateRequest, ModelExpertIoAdvisor, RequestId, ResidentScheduler, ResidentSchedulerConfig,
    ScheduledBatch, SchedulerAction, SequenceFinishReason, SequenceSlotPool, SequenceState,
    SessionId,
};
use crate::speculation::{
    PendingSpeculativeVerificationCohort, SpeculativeCohortProgress, SpeculativeCycleResult,
    SpeculativeMetrics, SpeculativeVerificationItem, TargetFrontier,
    begin_resumable_speculative_verification_cohort,
    cancel_resumable_speculative_verification_cohort,
    resume_resumable_speculative_verification_cohort,
};

use super::observability::{ResidentDriverObservability, ResidentTopKDriverStats};
use super::{NativeBatchExecutionProgress, NativeMultiSessionExecutor};

fn matched_stop(text: &str, stop: &[String]) -> bool {
    stop.iter()
        .any(|candidate| !candidate.is_empty() && text.ends_with(candidate))
}

fn proposal_confidence_probability(logit: f32) -> f32 {
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exponential = logit.exp();
        exponential / (1.0 + exponential)
    }
}

fn confident_proposal_prefix_length(logits: &[f32], threshold: f32) -> Result<usize> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(Error::Execution(format!(
            "proposal confidence threshold must be finite and within [0, 1], got {threshold}"
        )));
    }
    if threshold == 0.0 {
        return Ok(logits.len());
    }
    Ok(logits
        .iter()
        .position(|logit| proposal_confidence_probability(*logit) < threshold)
        .unwrap_or(logits.len()))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidentTopKDriverConfig {
    pub ctx_size: usize,
    pub stop_at_eos: bool,
    /// Static per-position confidence threshold used until the calibrated,
    /// batch-wide hardware scheduler is available. Zero disables truncation.
    pub proposal_confidence_threshold: f32,
}

impl Default for ResidentTopKDriverConfig {
    fn default() -> Self {
        Self {
            ctx_size: 4096,
            stop_at_eos: true,
            proposal_confidence_threshold: 0.2,
        }
    }
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
    /// Model work is suspended on one or more owned asynchronous continuations.
    WaitingForModelProgress(Vec<PendingModelProgress>),
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

enum PendingResidentKv {
    Reserved(Vec<KvReservation>),
    Prepared(PreparedKvCommit),
}

enum PendingKvRetirement {
    BackendRelease(KvRetirement),
    LogicalConfirmation(KvRetirement),
}

enum PendingSequenceCleanup<S> {
    Deferred,
    Owned {
        retirement: Option<PendingKvRetirement>,
        model_state: Option<S>,
    },
}

struct PendingResidentBatch<S> {
    transaction: ExecutionTransactionId,
    action: SchedulerAction,
    scheduled: ScheduledBatch,
    kv: PendingResidentKv,
    states: Vec<S>,
    schedules: Vec<SuspendedSequenceSchedule>,
    pending_progress: Option<PendingModelProgress>,
    cancelling: Option<RequestId>,
}

enum PendingSpeculativeDriverCohort<S> {
    Proposing(PendingNativeProposalCohort<S>),
    Verifying(Box<PendingSpeculativeVerificationDriverCohort<S>>),
}

impl<S> PendingSpeculativeDriverCohort<S> {
    fn actions(&self) -> &[DecodeAction] {
        match self {
            Self::Proposing(pending) => &pending.actions,
            Self::Verifying(pending) => &pending.actions,
        }
    }

    fn has_pending_progress(&self) -> bool {
        match self {
            Self::Proposing(pending) => pending
                .slots
                .iter()
                .any(|slot| matches!(&slot.status, NativeProposalSlotStatus::Waiting(_))),
            Self::Verifying(_) => true,
        }
    }

    fn extend_pending_progress(&self, output: &mut Vec<PendingModelProgress>) {
        match self {
            Self::Proposing(pending) => {
                output.extend(pending.slots.iter().filter_map(|slot| match &slot.status {
                    NativeProposalSlotStatus::Waiting(progress) => Some(progress.clone()),
                    NativeProposalSlotStatus::NotStarted
                    | NativeProposalSlotStatus::Complete { .. } => None,
                }));
            }
            Self::Verifying(pending) => {
                output.push(pending.verification.pending_progress().clone());
            }
        }
    }
}

struct PendingNativeProposalCohort<S> {
    transaction: ExecutionTransactionId,
    cohort_start: Instant,
    actions: Vec<DecodeAction>,
    proposal_source: NativeProposalSource,
    source_states: Vec<S>,
    schedules: Vec<SuspendedSequenceSchedule>,
    slots: Vec<PendingNativeProposalSlot>,
    cancellation_request: Option<RequestId>,
    cancellation_error: Option<String>,
}

struct PendingSpeculativeVerificationDriverCohort<S> {
    transaction: ExecutionTransactionId,
    cohort_start: Instant,
    actions: Vec<DecodeAction>,
    prepared: Vec<PreparedSpeculativeAction>,
    source_states: Vec<S>,
    schedules: Vec<SuspendedSequenceSchedule>,
    verification: PendingSpeculativeVerificationCohort<S>,
    cancellation_request: Option<RequestId>,
}

struct PendingNativeProposalSlot {
    sequence: SequenceState,
    page_slot: StateSlot,
    max_drafts: usize,
    proposal_start: Option<Instant>,
    status: NativeProposalSlotStatus,
}

enum NativeProposalSlotStatus {
    NotStarted,
    Waiting(PendingModelProgress),
    Complete {
        proposal: NativeProposal,
        proposal_time_us: u64,
        prepared: Box<Option<PreparedSpeculativeAction>>,
    },
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
    observability: ResidentDriverObservability,
    next_transaction_id: u64,
    resident_transactions: HashMap<ExecutionTransactionId, PendingResidentBatch<R::SequenceState>>,
    speculative_transactions:
        HashMap<ExecutionTransactionId, PendingSpeculativeDriverCohort<R::SequenceState>>,
    session_owner: HashMap<SessionId, ExecutionTransactionId>,
    ready_transactions: VecDeque<ExecutionTransactionId>,
    pending_kv_retirements: VecDeque<PendingKvRetirement>,
    pending_sequence_cleanups: HashMap<SessionId, PendingSequenceCleanup<R::SequenceState>>,
    committed_token_outbox: VecDeque<ResidentTokenEvent>,
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner,
    C: SequenceSlotPool,
{
    pub fn new(runner: R, slot_pool: C) -> Self
    where
        R: ResidentModelRunner,
    {
        Self::with_parts(
            ResidentScheduler::default(),
            slot_pool,
            Self::executor_with_expert_io_resources(runner)
                .expect("resident model reported an invalid expert-I/O resource topology"),
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
    ) -> Self
    where
        R: ResidentModelRunner,
    {
        Self::with_parts(
            ResidentScheduler::new(scheduler_config),
            slot_pool,
            Self::executor_with_expert_io_resources(runner)
                .expect("resident model reported an invalid expert-I/O resource topology"),
            top_k,
            driver_config,
        )
    }

    fn executor_with_expert_io_resources(mut runner: R) -> Result<NativeMultiSessionExecutor<R>>
    where
        R: ResidentModelRunner,
    {
        let limits = runner.expert_io_resource_limits()?.validate()?;
        let (control, handle) =
            BrokerExpertIoResourceControl::new(limits, runner.completion_hub())?;
        runner.install_expert_io_resource_control(Box::new(control))?;
        Ok(NativeMultiSessionExecutor::new(runner).with_expert_io_resources(handle))
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
            observability: ResidentDriverObservability::default(),
            next_transaction_id: 1,
            resident_transactions: HashMap::new(),
            speculative_transactions: HashMap::new(),
            session_owner: HashMap::new(),
            ready_transactions: VecDeque::new(),
            pending_kv_retirements: VecDeque::new(),
            pending_sequence_cleanups: HashMap::new(),
            committed_token_outbox: VecDeque::new(),
        }
    }

    pub fn scheduler(&self) -> &ResidentScheduler {
        &self.scheduler
    }

    pub fn slot_pool(&self) -> &C {
        &self.slot_pool
    }

    #[cfg(test)]
    pub(crate) fn executor(&self) -> &NativeMultiSessionExecutor<R> {
        &self.executor
    }

    #[cfg(test)]
    pub(crate) fn executor_mut(&mut self) -> &mut NativeMultiSessionExecutor<R> {
        &mut self.executor
    }

    pub fn model_info(&self) -> ferrule_model::ModelInfo {
        self.executor.runner().model_info()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.executor.runner().encode(text)
    }

    pub fn bound_layer_count(&self) -> Option<usize> {
        self.executor.runner().bound_layer_count()
    }

    pub fn expert_report(&self) -> Option<String> {
        self.executor.runner().expert_report()
    }

    pub fn model_observability_snapshot(&self) -> R::ObservabilitySnapshot
    where
        R: ResidentModelRunner,
    {
        self.executor.runner().observability_snapshot()
    }

    pub(crate) fn completion_hub(&self) -> ferrule_common::CompletionHub
    where
        R: ResidentModelRunner,
    {
        self.executor.runner().completion_hub()
    }

    pub(crate) fn take_completion_reactors(&mut self) -> Vec<ferrule_model::ModelCompletionReactor>
    where
        R: ResidentModelRunner,
    {
        self.executor.runner_mut().take_completion_reactors()
    }

    pub fn has_pending_async_work(&self) -> bool {
        !self.resident_transactions.is_empty()
            || !self.speculative_transactions.is_empty()
            || !self.pending_kv_retirements.is_empty()
            || !self.pending_sequence_cleanups.is_empty()
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

    fn ensure_no_suspended_execution(&self, operation: &str) -> Result<()> {
        if self.has_pending_async_work() || self.executor.has_transactions() {
            return Err(Error::Execution(format!(
                "cannot {operation} while execution transactions are live"
            )));
        }
        Ok(())
    }

    fn take_transaction_id(&mut self) -> Result<ExecutionTransactionId> {
        let value = self.next_transaction_id;
        self.next_transaction_id = value.checked_add(1).ok_or_else(|| {
            Error::Execution("resident execution transaction ID space is exhausted".into())
        })?;
        ExecutionTransactionId::new(value)
    }

    fn executor_has_transaction(&self, transaction: ExecutionTransactionId) -> bool {
        self.executor
            .transaction_ids()
            .any(|active| active == transaction)
    }

    fn enqueue_transaction(&mut self, transaction: ExecutionTransactionId) {
        if !self.ready_transactions.contains(&transaction) {
            self.ready_transactions.push_back(transaction);
        }
    }

    fn remove_transaction_from_queue(&mut self, transaction: ExecutionTransactionId) {
        self.ready_transactions
            .retain(|queued| *queued != transaction);
    }

    /// Suspend one active session and move its exclusively owned physical pages
    /// out of backend device residency.
    pub fn preempt_session(&mut self, session_id: SessionId) -> Result<()> {
        self.ensure_no_suspended_execution("preempt a session")?;
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
        self.ensure_no_suspended_execution("restore a session")?;
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

    pub fn try_into_runner(mut self) -> std::result::Result<R, Box<(Error, Self)>>
    where
        R: ResidentModelRunner,
    {
        if self.has_pending_async_work()
            || !self.session_owner.is_empty()
            || !self.committed_token_outbox.is_empty()
        {
            return Err(Box::new((
                Error::Execution(
                    "cannot extract resident runner with live execution transactions or undelivered committed events"
                        .into(),
                ),
                self,
            )));
        }
        if !self.scheduler.is_idle()
            || !self.suspended_sequences.is_empty()
            || !self.sequence_states.is_empty()
        {
            return Err(Box::new((
                Error::Execution(
                    "cannot extract resident runner while session state is still retained or active"
                        .into(),
                ),
                self,
            )));
        }
        if let Some(error) = self.executor.runner_extraction_error() {
            return Err(Box::new((error, self)));
        }
        if let Err(error) = self
            .executor
            .runner_mut()
            .uninstall_expert_io_resource_control()
        {
            return Err(Box::new((error, self)));
        }
        match self.executor.try_into_runner() {
            Ok(runner) => Ok(runner),
            Err(failure) => {
                let (error, executor) = *failure;
                self.executor = executor;
                Err(Box::new((error, self)))
            }
        }
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
        self.ensure_no_suspended_execution("release a session")?;
        if !self.scheduler.is_idle() || self.suspended_sequences.contains_key(&session_id) {
            return Err(Error::Execution(format!(
                "cannot release session {session_id:?} while scheduler work is active or suspended"
            )));
        }
        self.release_sequence_state(session_id)?;
        self.retained_sessions.remove(&session_id);
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
        self.observability.stats()
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

    /// Cancel a waiting or active request without disturbing unrelated transactions.
    pub fn cancel_request(&mut self, request_id: RequestId) -> Result<CancelRequestResult>
    where
        R: ResidentModelRunner,
    {
        if let Some(transaction) = self.transaction_for_request(request_id) {
            self.cancel_transaction(transaction, request_id)?;
        }
        self.cancel_scheduled_request(request_id)
    }

    fn cancel_scheduled_request(&mut self, request_id: RequestId) -> Result<CancelRequestResult> {
        let result = self
            .scheduler
            .cancel_request(request_id, &mut self.slot_pool)?;
        if let CancelRequestResult::Active { session_id, .. } = result {
            if let Some(position) = self.retained_sessions.get_mut(&session_id) {
                *position = 0;
            }
            self.release_sequence_state(session_id)?;
        }
        Ok(result)
    }

    pub(crate) fn request_has_pending_model_progress(&self, request_id: RequestId) -> bool {
        let Some(transaction) = self.transaction_for_request(request_id) else {
            return false;
        };
        self.resident_transactions
            .get(&transaction)
            .is_some_and(|pending| pending.pending_progress.is_some())
            || self
                .speculative_transactions
                .get(&transaction)
                .is_some_and(PendingSpeculativeDriverCohort::has_pending_progress)
    }

    fn transaction_for_request(&self, request_id: RequestId) -> Option<ExecutionTransactionId> {
        self.resident_transactions
            .iter()
            .find_map(|(transaction, pending)| {
                action_contains_request(&pending.action, request_id).then_some(*transaction)
            })
            .or_else(|| {
                self.speculative_transactions
                    .iter()
                    .find_map(|(transaction, pending)| {
                        pending
                            .actions()
                            .iter()
                            .any(|action| action.request_id == Some(request_id))
                            .then_some(*transaction)
                    })
            })
    }

    fn cancel_transaction(
        &mut self,
        transaction: ExecutionTransactionId,
        request_id: RequestId,
    ) -> Result<()>
    where
        R: ResidentModelRunner,
    {
        self.remove_transaction_from_queue(transaction);
        if let Some(pending) = self.resident_transactions.remove(&transaction) {
            return self.cancel_resident_transaction(pending, request_id);
        }
        let pending = self
            .speculative_transactions
            .remove(&transaction)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "transaction {transaction:?} has no driver ownership"
                ))
            })?;
        match pending {
            PendingSpeculativeDriverCohort::Proposing(pending) => {
                self.cancel_native_proposal_cohort(pending, Some(request_id), None)
            }
            PendingSpeculativeDriverCohort::Verifying(pending) => {
                self.cancel_speculative_verification_cohort(*pending, request_id)
            }
        }
    }

    fn cancel_resident_transaction(
        &mut self,
        mut pending: PendingResidentBatch<R::SequenceState>,
        request_id: RequestId,
    ) -> Result<()> {
        let transaction = pending.transaction;
        pending.cancelling = Some(request_id);
        let cancellation = match pending.pending_progress.as_ref() {
            Some(progress) => self.executor.cancel_resumable_batch(
                transaction,
                &mut pending.states,
                progress.continuation(),
            ),
            None if self.executor_has_transaction(transaction) => self
                .executor
                .rollback_prepared_batch(transaction, &mut pending.states),
            None => Ok(()),
        };
        if let Err(error) = cancellation {
            if let Some(progress) = self.executor.pending_model_progress(transaction).cloned() {
                pending.pending_progress = Some(progress);
                self.resident_transactions.insert(transaction, pending);
                self.enqueue_transaction(transaction);
                return Err(error);
            }
            if self.executor_has_transaction(transaction) {
                pending.pending_progress = None;
                self.resident_transactions.insert(transaction, pending);
                self.enqueue_transaction(transaction);
                return Err(error);
            }
            let cleanup = self.finish_resident_rollback(pending);
            return match cleanup {
                Ok(()) => Err(error),
                Err(cleanup) => Err(Error::Internal(format!(
                    "transaction {transaction:?} cancellation failed ({error}); cleanup also failed ({cleanup})"
                ))),
            };
        }
        self.finish_resident_rollback(pending)
    }

    fn cancel_native_proposal_cohort(
        &mut self,
        mut pending: PendingNativeProposalCohort<R::SequenceState>,
        request_id: Option<RequestId>,
        failure: Option<Error>,
    ) -> Result<()>
    where
        R: ResidentModelRunner,
    {
        if let Some(request_id) = request_id {
            pending.cancellation_request = Some(request_id);
        }
        if let Some(failure) = failure {
            let failure = failure.to_string();
            pending.cancellation_error = Some(match pending.cancellation_error.take() {
                Some(previous) => format!("{previous}; {failure}"),
                None => failure,
            });
        }

        let mut still_active = Vec::new();
        for slot_index in 0..pending.slots.len() {
            let continuation = match &pending.slots[slot_index].status {
                NativeProposalSlotStatus::Waiting(progress) => progress.continuation(),
                NativeProposalSlotStatus::NotStarted
                | NativeProposalSlotStatus::Complete { .. } => continue,
            };
            let transaction = pending.transaction;
            let cancellation = self
                .executor
                .with_sequence_state(&mut pending.source_states[slot_index], |runner| {
                    Ok(runner.cancel_native_proposal(transaction, continuation))
                });
            match cancellation {
                Ok(BatchContinuationCancelOutcome::Cancelled) => {
                    pending.slots[slot_index].status = NativeProposalSlotStatus::NotStarted;
                }
                Ok(BatchContinuationCancelOutcome::Quiesced(error)) => {
                    pending.slots[slot_index].status = NativeProposalSlotStatus::NotStarted;
                    let error = error.to_string();
                    pending.cancellation_error = Some(match pending.cancellation_error.take() {
                        Some(previous) => format!("{previous}; {error}"),
                        None => error,
                    });
                }
                Ok(BatchContinuationCancelOutcome::StillActive(error)) | Err(error) => {
                    still_active.push(error.to_string());
                }
            }
        }

        if !still_active.is_empty() {
            let error = Error::Execution(format!(
                "speculative proposal cancellation left active continuations: {}",
                still_active.join("; ")
            ));
            let transaction = pending.transaction;
            self.speculative_transactions.insert(
                transaction,
                PendingSpeculativeDriverCohort::Proposing(pending),
            );
            self.enqueue_transaction(transaction);
            return Err(error);
        }

        let actions = pending.actions;
        self.restore_transaction_sessions(pending.schedules, pending.source_states)?;
        let requeue = self.scheduler.requeue_decode_actions_front(&actions);
        match (pending.cancellation_error, requeue) {
            (None, Ok(())) => Ok(()),
            (Some(error), Ok(())) => Err(Error::Execution(error)),
            (None, Err(error)) => Err(error),
            (Some(error), Err(requeue)) => Err(Error::Internal(format!(
                "speculative proposal cancellation failed ({error}); restoring scheduler actions also failed ({requeue})"
            ))),
        }
    }

    fn cancel_speculative_verification_cohort(
        &mut self,
        pending: PendingSpeculativeVerificationDriverCohort<R::SequenceState>,
        request_id: RequestId,
    ) -> Result<()> {
        let PendingSpeculativeVerificationDriverCohort {
            transaction,
            cohort_start,
            actions,
            prepared,
            source_states,
            schedules,
            verification,
            cancellation_request: _,
        } = pending;
        let page_manager = match self.page_manager.as_mut() {
            Some(page_manager) => page_manager,
            None => {
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::Verifying(Box::new(
                        PendingSpeculativeVerificationDriverCohort {
                            transaction,
                            cohort_start,
                            actions,
                            prepared,
                            source_states,
                            schedules,
                            verification,
                            cancellation_request: Some(request_id),
                        },
                    )),
                );
                self.enqueue_transaction(transaction);
                return Err(Error::Internal(
                    "authoritative KvPageManager disappeared while cancelling a suspended speculative cohort"
                        .into(),
                ));
            }
        };
        let cancellation = cancel_resumable_speculative_verification_cohort(
            &mut self.executor,
            page_manager,
            verification,
        );
        match cancellation {
            Ok(()) => {
                self.restore_transaction_sessions(schedules, source_states)?;
                self.scheduler.requeue_decode_actions_front(&actions)
            }
            Err(error) => {
                let (error, verification) = error.into_parts();
                if let Some(verification) = verification {
                    self.speculative_transactions.insert(
                        transaction,
                        PendingSpeculativeDriverCohort::Verifying(Box::new(
                            PendingSpeculativeVerificationDriverCohort {
                                transaction,
                                cohort_start,
                                actions,
                                prepared,
                                source_states,
                                schedules,
                                verification: *verification,
                                cancellation_request: Some(request_id),
                            },
                        )),
                    );
                    self.enqueue_transaction(transaction);
                    Err(error)
                } else {
                    self.restore_transaction_sessions(schedules, source_states)?;
                    match self.scheduler.requeue_decode_actions_front(&actions) {
                        Ok(()) => Err(error),
                        Err(requeue) => Err(Error::Internal(format!(
                            "speculative cancellation cleanup failed ({error}); restoring scheduler actions also failed ({requeue})"
                        ))),
                    }
                }
            }
        }
    }

    fn claim_transaction_sessions(
        &mut self,
        transaction: ExecutionTransactionId,
        session_ids: &[SessionId],
    ) -> Result<(Vec<SuspendedSequenceSchedule>, Vec<R::SequenceState>)> {
        let mut unique = HashSet::with_capacity(session_ids.len());
        for session_id in session_ids {
            if !unique.insert(*session_id) {
                return Err(Error::Internal(format!(
                    "transaction {transaction:?} contains duplicate session {session_id:?}"
                )));
            }
            if let Some(owner) = self.session_owner.get(session_id) {
                return Err(Error::Execution(format!(
                    "session {session_id:?} is already owned by transaction {owner:?}"
                )));
            }
        }

        let mut schedules = Vec::with_capacity(session_ids.len());
        let mut states = Vec::with_capacity(session_ids.len());
        for session_id in session_ids {
            let schedule = match self.scheduler.suspend_sequence(*session_id) {
                Ok(schedule) => schedule,
                Err(error) => {
                    self.restore_transaction_sessions(schedules, states)?;
                    return Err(error);
                }
            };
            let Some(state) = self.sequence_states.remove(session_id) else {
                self.scheduler.restore_suspended(schedule)?;
                self.restore_transaction_sessions(schedules, states)?;
                return Err(Error::Internal(format!(
                    "session {session_id:?} has no model sequence state"
                )));
            };
            self.session_owner.insert(*session_id, transaction);
            schedules.push(schedule);
            states.push(state);
        }
        Ok((schedules, states))
    }

    fn restore_transaction_sessions(
        &mut self,
        schedules: Vec<SuspendedSequenceSchedule>,
        states: Vec<R::SequenceState>,
    ) -> Result<()> {
        if schedules.len() != states.len() {
            return Err(Error::Internal(format!(
                "transaction schedule/state mismatch: schedules={} states={}",
                schedules.len(),
                states.len()
            )));
        }
        for (schedule, state) in schedules.into_iter().zip(states) {
            let session_id = schedule.session_id();
            self.session_owner.remove(&session_id);
            let previous = self.sequence_states.insert(session_id, state);
            if previous.is_some() {
                return Err(Error::Internal(format!(
                    "session {session_id:?} model state was already published"
                )));
            }
            self.scheduler.restore_suspended(schedule)?;
        }
        Ok(())
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
        self.ensure_no_suspended_execution("fork a session")?;
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
    fn finalize_sequence_state(&mut self, session_id: SessionId, position: usize) -> Result<()> {
        if let Some(retained_position) = self.retained_sessions.get_mut(&session_id) {
            *retained_position = position;
            Ok(())
        } else {
            self.release_sequence_state(session_id)
        }
    }

    /// Release sequence/KV ownership now, or retain it in a driver-owned cleanup
    /// record until every packed backend transaction is quiescent.
    fn release_sequence_state(&mut self, session_id: SessionId) -> Result<()> {
        self.pending_sequence_cleanups
            .entry(session_id)
            .or_insert(PendingSequenceCleanup::Deferred);
        if self.executor.has_transactions() {
            return Ok(());
        }
        self.progress_sequence_cleanup(session_id)
    }

    fn progress_pending_cleanups(&mut self) -> Result<()> {
        if self.executor.has_transactions() {
            return Ok(());
        }
        while let Some(retirement) = self.pending_kv_retirements.pop_front() {
            if let Err((error, retirement)) = self.progress_kv_retirement(retirement) {
                self.pending_kv_retirements.push_front(retirement);
                return Err(error);
            }
        }
        let sessions = self
            .pending_sequence_cleanups
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for session_id in sessions {
            self.progress_sequence_cleanup(session_id)?;
        }
        Ok(())
    }

    fn progress_sequence_cleanup(&mut self, session_id: SessionId) -> Result<()> {
        if self.executor.has_transactions() {
            return Ok(());
        }
        let Some(cleanup) = self.pending_sequence_cleanups.remove(&session_id) else {
            return Ok(());
        };
        let cleanup = match cleanup {
            PendingSequenceCleanup::Deferred => {
                let retirement = match self.page_slots.get(&session_id).copied() {
                    Some(slot) => {
                        let Some(manager) = self.page_manager.as_mut() else {
                            self.pending_sequence_cleanups
                                .insert(session_id, PendingSequenceCleanup::Deferred);
                            return Err(Error::Internal(format!(
                                "session {session_id:?} has a page slot without an authoritative page manager"
                            )));
                        };
                        let retirement = match manager.free_sequence_pages(slot) {
                            Ok(retirement) => retirement,
                            Err(error) => {
                                self.pending_sequence_cleanups
                                    .insert(session_id, PendingSequenceCleanup::Deferred);
                                return Err(error);
                            }
                        };
                        self.page_slots.remove(&session_id);
                        Some(PendingKvRetirement::BackendRelease(retirement))
                    }
                    None => None,
                };
                PendingSequenceCleanup::Owned {
                    retirement,
                    model_state: self.sequence_states.remove(&session_id),
                }
            }
            cleanup @ PendingSequenceCleanup::Owned { .. } => cleanup,
        };
        let PendingSequenceCleanup::Owned {
            retirement,
            model_state,
        } = cleanup
        else {
            unreachable!("deferred sequence cleanup was converted to owned state")
        };
        if let Some(retirement) = retirement
            && let Err((error, retirement)) = self.progress_kv_retirement(retirement)
        {
            self.pending_sequence_cleanups.insert(
                session_id,
                PendingSequenceCleanup::Owned {
                    retirement: Some(retirement),
                    model_state,
                },
            );
            return Err(error);
        }
        if let Some(state) = model_state {
            self.executor.release_sequence_state(state)?;
        }
        Ok(())
    }

    fn progress_kv_retirement(
        &mut self,
        retirement: PendingKvRetirement,
    ) -> std::result::Result<(), (Error, PendingKvRetirement)> {
        if self.executor.has_transactions() {
            return Err((
                Error::Execution(
                    "cannot progress KV retirement while packed transactions are live".into(),
                ),
                retirement,
            ));
        }
        let retirement = match retirement {
            PendingKvRetirement::BackendRelease(retirement) => {
                if !retirement.is_empty()
                    && let Err(error) = self
                        .executor
                        .runner_mut()
                        .release_kv_pages(retirement.pages())
                {
                    return Err((error, PendingKvRetirement::BackendRelease(retirement)));
                }
                retirement
            }
            PendingKvRetirement::LogicalConfirmation(retirement) => retirement,
        };
        let Some(manager) = self.page_manager.as_mut() else {
            return Err((
                Error::Internal("retiring KV pages have no authoritative page manager".into()),
                PendingKvRetirement::LogicalConfirmation(retirement),
            ));
        };
        match manager.confirm_page_retirement(retirement) {
            Ok(()) => Ok(()),
            Err(error) => {
                let (error, retirement) = error.into_parts();
                Err((error, PendingKvRetirement::LogicalConfirmation(retirement)))
            }
        }
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
                    let cleanup = manager
                        .abort_reservations(reservations)
                        .map_err(|abort| abort.into_parts().0)
                        .and_then(|retirement| {
                            manager
                                .confirm_page_retirement(retirement)
                                .map_err(|confirm| confirm.into_parts().0)
                        });
                    return match cleanup {
                        Ok(()) => Err(error),
                        Err(cleanup) => Err(Error::Internal(format!(
                            "KV reserve failed ({error}); reservation cleanup also failed ({cleanup})"
                        ))),
                    };
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
        let views = manager.reservation_views(reservations)?;
        let bindings = views
            .iter()
            .map(|reservation| manager.reservation_bindings(reservation))
            .collect::<Result<Vec<_>>>()?;
        batch.bind_paged_kv(&bindings)
    }

    fn abort_quiesced_resident_kv(&mut self, kv: PendingResidentKv) -> Result<()> {
        let Some(manager) = &mut self.page_manager else {
            return match kv {
                PendingResidentKv::Reserved(reservations) if reservations.is_empty() => Ok(()),
                PendingResidentKv::Reserved(_) | PendingResidentKv::Prepared(_) => {
                    Err(Error::Internal(
                        "KV transaction exists without an authoritative page manager".into(),
                    ))
                }
            };
        };
        let retirement = match kv {
            PendingResidentKv::Reserved(reservations) => manager
                .abort_reservations(reservations)
                .map_err(|error| error.into_parts().0)?,
            PendingResidentKv::Prepared(prepared) => manager.abort_prepared_commit(prepared),
        };
        self.release_and_confirm_retirement(retirement)
    }

    fn release_and_confirm_retirement(&mut self, retirement: KvRetirement) -> Result<()> {
        let retirement = PendingKvRetirement::BackendRelease(retirement);
        if self.executor.has_transactions() {
            self.pending_kv_retirements.push_back(retirement);
            return Ok(());
        }
        match self.progress_kv_retirement(retirement) {
            Ok(()) => Ok(()),
            Err((error, retirement)) => {
                self.pending_kv_retirements.push_back(retirement);
                Err(error)
            }
        }
    }

    fn prepare_step(&mut self) -> Result<()> {
        self.validate_configuration()?;
        if self.executor.is_poisoned() {
            return Err(Error::Execution(
                "native executor is poisoned; reset before executing again".into(),
            ));
        }
        self.admit_new_sequences()
    }

    fn no_action_step(&self) -> ResidentDriverStep {
        if self.scheduler.is_idle() {
            ResidentDriverStep::Idle
        } else {
            ResidentDriverStep::Blocked
        }
    }

    fn execute_planned_action<F>(
        &mut self,
        mut action: SchedulerAction,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let Some(mut scheduled) = ScheduledBatch::from_action(&mut action, self.top_k)
            .map_err(|error| self.abort_action(&action, error, false, "batch lowering"))?
        else {
            self.scheduler.commit_action(&action)?;
            return Ok(ResidentDriverStep::Executed {
                action_kind: action_kind(&action),
                rows: 0,
                staged: 0,
                finished: 0,
            });
        };

        let transaction = self.take_transaction_id()?;
        let session_ids = scheduled
            .sequences
            .iter()
            .map(|sequence| sequence.session_id)
            .collect::<Vec<_>>();
        let (schedules, mut states) = self
            .claim_transaction_sessions(transaction, &session_ids)
            .map_err(|error| self.abort_action(&action, error, false, "session claim"))?;

        let mut page_reservations = match self.reserve_batch_pages(&scheduled) {
            Ok(reservations) => reservations,
            Err(error) => {
                self.restore_transaction_sessions(schedules, states)?;
                return Err(self.abort_action(&action, error, false, "KV reserve"));
            }
        };
        if let Err(error) = self.bind_reserved_pages(&mut scheduled, &page_reservations) {
            let rollback = self.abort_quiesced_resident_kv(PendingResidentKv::Reserved(
                std::mem::take(&mut page_reservations),
            ));
            self.restore_transaction_sessions(schedules, states)?;
            return Err(match rollback {
                Ok(_) => self.abort_action(&action, error, false, "KV binding"),
                Err(cleanup) => Error::Internal(format!(
                    "KV binding failed ({error}); logical rollback also failed ({cleanup})"
                )),
            });
        }
        let reservation_views = match &self.page_manager {
            Some(manager) => manager.reservation_views(&page_reservations),
            None if page_reservations.is_empty() => Ok(Vec::<KvReservationView>::new()),
            None => Err(Error::Internal(
                "KV reservations exist without an authoritative page manager".into(),
            )),
        };
        let reservation_views = match reservation_views {
            Ok(views) => views,
            Err(error) => {
                let cleanup =
                    self.abort_quiesced_resident_kv(PendingResidentKv::Reserved(page_reservations));
                self.restore_transaction_sessions(schedules, states)?;
                let error = self.abort_action(&action, error, false, "KV reservation view");
                return Err(match cleanup {
                    Ok(()) => error,
                    Err(cleanup) => Error::Internal(format!(
                        "KV reservation view failed ({error}); KV cleanup also failed ({cleanup})"
                    )),
                });
            }
        };

        let progress = self.executor.begin_resumable_batch_with_kv(
            transaction,
            &mut states,
            scheduled.execution(),
            &reservation_views,
        );
        let mut pending = PendingResidentBatch {
            transaction,
            action,
            scheduled,
            kv: PendingResidentKv::Reserved(page_reservations),
            states,
            schedules,
            pending_progress: None,
            cancelling: None,
        };
        match progress {
            Ok(NativeBatchExecutionProgress::Complete(output)) => {
                self.finish_resident_transaction(pending, output, on_token)
            }
            Ok(NativeBatchExecutionProgress::Waiting(progress)) => {
                pending.pending_progress = Some(progress);
                self.resident_transactions.insert(transaction, pending);
                self.enqueue_transaction(transaction);
                Ok(ResidentDriverStep::WaitingForModelProgress(
                    self.pending_model_progresses(),
                ))
            }
            Err(error) => {
                if let Some(progress) = self.executor.pending_model_progress(transaction).cloned() {
                    pending.pending_progress = Some(progress);
                    self.resident_transactions.insert(transaction, pending);
                    self.enqueue_transaction(transaction);
                    return Err(error);
                }
                if self.executor_has_transaction(transaction) {
                    self.resident_transactions.insert(transaction, pending);
                    return Err(error);
                }
                let cleanup = self.abort_quiesced_resident_kv(pending.kv);
                self.restore_transaction_sessions(pending.schedules, pending.states)?;
                let error = self.abort_action(&pending.action, error, false, "model execution");
                Err(match cleanup {
                    Ok(()) => error,
                    Err(cleanup) => Error::Internal(format!(
                        "model execution failed ({error}); KV cleanup also failed ({cleanup})"
                    )),
                })
            }
        }
    }

    fn resume_resident_transaction<F>(
        &mut self,
        transaction: ExecutionTransactionId,
        on_token: &mut F,
    ) -> Result<Option<ResidentDriverStep>>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let mut pending = self
            .resident_transactions
            .remove(&transaction)
            .ok_or_else(|| {
                Error::Internal(format!("resident transaction {transaction:?} disappeared"))
            })?;
        if let Some(request_id) = pending.cancelling {
            return match self.cancel_resident_transaction(pending, request_id) {
                Ok(()) => {
                    self.cancel_scheduled_request(request_id)?;
                    Ok(Some(ResidentDriverStep::Executed {
                        action_kind: ResidentActionKind::Cancel,
                        rows: 0,
                        staged: 0,
                        finished: 0,
                    }))
                }
                Err(_) if self.resident_transactions.contains_key(&transaction) => Ok(None),
                Err(error) => Err(error),
            };
        }
        let continuation = match pending.pending_progress.as_ref() {
            Some(progress) => progress.continuation(),
            None => {
                self.resident_transactions.insert(transaction, pending);
                return Err(Error::Execution(format!(
                    "resident transaction {transaction:?} is quarantined"
                )));
            }
        };
        match self.executor.resume_resumable_batch(
            transaction,
            &mut pending.states,
            pending.scheduled.execution(),
            continuation,
        ) {
            Ok(NativeBatchExecutionProgress::Complete(output)) => {
                pending.pending_progress = None;
                self.finish_resident_transaction(pending, output, on_token)
                    .map(Some)
            }
            Ok(NativeBatchExecutionProgress::Waiting(progress)) => {
                pending.pending_progress = Some(progress);
                self.resident_transactions.insert(transaction, pending);
                self.enqueue_transaction(transaction);
                Ok(None)
            }
            Err(error) => {
                if let Some(progress) = self.executor.pending_model_progress(transaction).cloned() {
                    pending.pending_progress = Some(progress);
                    self.resident_transactions.insert(transaction, pending);
                    self.enqueue_transaction(transaction);
                    return Err(error);
                }
                if self.executor_has_transaction(transaction) {
                    pending.pending_progress = None;
                    self.resident_transactions.insert(transaction, pending);
                    return Err(error);
                }
                let cleanup = self.abort_quiesced_resident_kv(pending.kv);
                self.restore_transaction_sessions(pending.schedules, pending.states)?;
                let error =
                    self.abort_action(&pending.action, error, false, "resumable model execution");
                Err(match cleanup {
                    Ok(()) => error,
                    Err(cleanup) => Error::Internal(format!(
                        "resumable model execution failed ({error}); KV cleanup also failed ({cleanup})"
                    )),
                })
            }
        }
    }

    fn finish_resident_transaction<F>(
        &mut self,
        mut pending: PendingResidentBatch<R::SequenceState>,
        output: ExecutionOutput,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transaction = pending.transaction;
        if let Err(error) = pending.scheduled.validate_output(&output) {
            return Err(self.rollback_and_fail_resident(pending, error, "model output contract"));
        }

        let reserved = match std::mem::replace(
            &mut pending.kv,
            PendingResidentKv::Reserved(Vec::new()),
        ) {
            PendingResidentKv::Reserved(reservations) => reservations,
            PendingResidentKv::Prepared(prepared) => {
                pending.kv = PendingResidentKv::Prepared(prepared);
                self.resident_transactions.insert(transaction, pending);
                return Err(Error::Internal(format!(
                    "transaction {transaction:?} reached model completion with an existing prepared logical commit"
                )));
            }
        };
        let prepared = match self.page_manager.as_mut() {
            Some(manager) => {
                let commits = reserved
                    .into_iter()
                    .map(|reservation| {
                        let rows = reservation.view().positions.len();
                        KvReservationCommit::new(reservation, rows)
                    })
                    .collect();
                match manager.prepare_commit(commits) {
                    Ok(prepared) => Some(prepared),
                    Err(error) => {
                        let (error, commits) = error.into_parts();
                        pending.kv = PendingResidentKv::Reserved(
                            commits
                                .into_iter()
                                .map(|commit| commit.reservation)
                                .collect(),
                        );
                        return Err(self.rollback_and_fail_resident(
                            pending,
                            error,
                            "logical KV prepare",
                        ));
                    }
                }
            }
            None if reserved.is_empty() => None,
            None => {
                pending.kv = PendingResidentKv::Reserved(reserved);
                return Err(self.rollback_and_fail_resident(
                    pending,
                    Error::Internal(
                        "KV reservations exist without an authoritative page manager".into(),
                    ),
                    "logical KV prepare",
                ));
            }
        };
        if let Some(prepared) = prepared {
            pending.kv = PendingResidentKv::Prepared(prepared);
        }

        if let Err(error) = self
            .executor
            .commit_prepared_batch(transaction, &mut pending.states)
        {
            return Err(self.rollback_and_fail_resident(pending, error, "backend KV commit"));
        }
        let retirement =
            match std::mem::replace(&mut pending.kv, PendingResidentKv::Reserved(Vec::new())) {
                PendingResidentKv::Prepared(prepared) => Some(
                    self.page_manager
                        .as_mut()
                        .expect("prepared logical commit requires a page manager")
                        .publish_commit(prepared),
                ),
                PendingResidentKv::Reserved(reservations) if reservations.is_empty() => None,
                PendingResidentKv::Reserved(_) => {
                    unreachable!("backend committed while logical reservations remained unprepared")
                }
            };
        if let Some(retirement) = retirement {
            self.release_and_confirm_retirement(retirement)?;
        }

        let action_kind = action_kind(&pending.action);
        let rows = action_rows(&pending.action);
        self.restore_transaction_sessions(pending.schedules, pending.states)?;
        if let Err(error) = self.scheduler.commit_action(&pending.action) {
            return Err(self.abort_action(&pending.action, error, false, "scheduler publish"));
        }
        self.observability.stats.actions += 1;
        match &pending.action {
            SchedulerAction::Execute { prefills, decodes } => {
                self.observability.stats.prefill_chunks += prefills.len();
                self.observability.stats.prefill_tokens += prefills
                    .iter()
                    .map(|action| action.token_range.len())
                    .sum::<usize>();
                self.observability.stats.decode_steps += decodes.len();
                self.enqueue_committed_decode_tokens(decodes)?;
            }
            SchedulerAction::PrefillChunk(prefill) => {
                self.observability.stats.prefill_chunks += 1;
                self.observability.stats.prefill_tokens += prefill.token_range.len();
            }
            SchedulerAction::DecodeBatch(actions) => {
                self.observability.stats.decode_steps += actions.len();
                self.enqueue_committed_decode_tokens(actions)?;
            }
            SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => {}
        }

        let action_finish = self.finish_after_decode_action(&pending.action)?;
        let mut finished = action_finish.finished;
        let output_outcome =
            self.apply_execution_output(&pending.scheduled, &output, &action_finish.session_ids)?;
        finished += output_outcome.finished;
        self.flush_committed_token_outbox(on_token)?;
        Ok(ResidentDriverStep::Executed {
            action_kind,
            rows,
            staged: output_outcome.staged,
            finished,
        })
    }

    fn rollback_and_fail_resident(
        &mut self,
        mut pending: PendingResidentBatch<R::SequenceState>,
        error: Error,
        stage: &'static str,
    ) -> Error {
        let transaction = pending.transaction;
        let rollback = self
            .executor
            .rollback_prepared_batch(transaction, &mut pending.states);
        if let Err(rollback_error) = rollback {
            pending.pending_progress = None;
            self.resident_transactions.insert(transaction, pending);
            return Error::Internal(format!(
                "{stage} failed ({error}); backend rollback also failed ({rollback_error})"
            ));
        }
        let logical = self.abort_quiesced_resident_kv(pending.kv);
        let restore = self.restore_transaction_sessions(pending.schedules, pending.states);
        match (logical, restore) {
            (Ok(()), Ok(())) => self.abort_action(&pending.action, error, false, stage),
            (logical, restore) => Error::Internal(format!(
                "{stage} failed ({error}); logical cleanup={:?}; session restore={:?}",
                logical.err(),
                restore.err()
            )),
        }
    }

    fn finish_resident_rollback(
        &mut self,
        pending: PendingResidentBatch<R::SequenceState>,
    ) -> Result<()> {
        self.abort_quiesced_resident_kv(pending.kv)?;
        let action = pending.action;
        self.restore_transaction_sessions(pending.schedules, pending.states)?;
        self.scheduler
            .requeue_decode_actions_front(action_decode_actions(&action))
    }

    fn pending_model_progresses(&self) -> Vec<PendingModelProgress> {
        let mut progresses = Vec::new();
        let mut visited = HashSet::new();
        for transaction in &self.ready_transactions {
            if !visited.insert(*transaction) {
                continue;
            }
            if let Some(pending) = self.resident_transactions.get(transaction) {
                if let Some(progress) = &pending.pending_progress {
                    progresses.push(progress.clone());
                }
            } else if let Some(pending) = self.speculative_transactions.get(transaction) {
                pending.extend_pending_progress(&mut progresses);
            }
        }
        progresses
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
        let session_ids = action_session_ids(action);
        let scheduler_cleanup = self
            .scheduler
            .fail_action(action, &mut self.slot_pool)
            .err();
        let mut state_cleanup = Vec::new();
        for session_id in &session_ids {
            if let Err(cleanup) = self.release_sequence_state(*session_id) {
                state_cleanup.push(format!("session {session_id:?}: {cleanup}"));
            }
        }
        match (scheduler_cleanup, state_cleanup.is_empty()) {
            (None, true) => error,
            (scheduler_cleanup, state_clean) => Error::Internal(format!(
                "{stage} failed ({error}); scheduler cleanup={scheduler_cleanup:?}; state cleanup={:?}",
                (!state_clean).then_some(state_cleanup)
            )),
        }
    }

    fn flush_committed_token_outbox<F>(&mut self, on_token: &mut F) -> Result<()>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        while let Some(event) = self.committed_token_outbox.front() {
            on_token(event)?;
            self.committed_token_outbox.pop_front();
            self.observability.stats.emitted_tokens += 1;
        }
        Ok(())
    }

    fn enqueue_committed_decode_tokens(&mut self, actions: &[DecodeAction]) -> Result<()> {
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
            self.committed_token_outbox.push_back(event);
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
            } else if matched_stop(&sequence.generated_text, &sequence.stop) {
                Some(SequenceFinishReason::StopString)
            } else {
                None
            };
            if let Some(reason) = reason {
                let position = sequence.position;
                self.scheduler
                    .finish_sequence(action.session_id, reason, &mut self.slot_pool)?;
                self.finalize_sequence_state(action.session_id, position)?;
                self.observability.stats.finished_sequences += 1;
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
                self.finalize_sequence_state(session_id, position)?;
                self.observability.stats.finished_sequences += 1;
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
                self.finalize_sequence_state(session_id, position)?;
                self.observability.stats.finished_sequences += 1;
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
                self.finalize_sequence_state(session_id, position)?;
                self.observability.stats.finished_sequences += 1;
                outcome.finished += 1;
                continue;
            };

            if self.config.stop_at_eos
                && !sequence.ignore_eos
                && self.executor.runner().eos_token_id() == Some(candidate.token_id)
            {
                let position = self
                    .scheduler
                    .active_sequence(session_id)
                    .map_or(0, |sequence| sequence.position);
                self.scheduler.finish_sequence(
                    session_id,
                    SequenceFinishReason::Eos,
                    &mut self.slot_pool,
                )?;
                self.finalize_sequence_state(session_id, position)?;
                self.observability.stats.finished_sequences += 1;
                outcome.finished += 1;
                continue;
            }

            self.scheduler.stage_decode_candidate(
                session_id,
                candidate.token_id,
                Some(candidate.logit),
            )?;
            self.observability.stats.staged_tokens += 1;
            outcome.staged += 1;
        }
        Ok(outcome)
    }
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: ResidentModelRunner,
    C: SequenceSlotPool,
{
    /// Execute the resident model path selected by model capabilities.
    ///
    /// All models use the same packed target executor. A model that reports a
    /// checkpoint-native proposal capability may add proposal + verification for
    /// decode; models without it continue through target-only packed decode.
    pub fn step_with_model_expert_io<F>(
        &mut self,
        on_token: &mut F,
        expert_budget: ExpertIoBudget,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        self.flush_committed_token_outbox(on_token)?;
        self.progress_pending_cleanups()?;
        let proposal_enabled = self.executor.runner().native_proposal_source()?.is_some();

        let requires_page_manager = proposal_enabled
            || self.executor.capabilities().kv_binding_mode == KvBindingMode::Paged;
        if requires_page_manager && self.page_manager.is_none() {
            return Err(Error::Execution(
                "paged or proposal-enabled resident execution requires an authoritative KvPageManager"
                    .into(),
            ));
        }
        let resumable = self.ready_transactions.len();
        for _ in 0..resumable {
            let Some(transaction) = self.ready_transactions.pop_front() else {
                break;
            };
            if self.resident_transactions.contains_key(&transaction) {
                if let Some(step) = self.resume_resident_transaction(transaction, on_token)? {
                    return Ok(step);
                }
            } else if self.speculative_transactions.contains_key(&transaction)
                && let Some(step) = self.resume_speculative_transaction(transaction, on_token)?
            {
                return Ok(step);
            }
            if proposal_enabled {
                let pending = self.pending_model_progresses();
                if !pending.is_empty() {
                    return Ok(ResidentDriverStep::WaitingForModelProgress(pending));
                }
            }
        }

        self.prepare_step()?;
        let allow_mixed_batches = !proposal_enabled && self.scheduler.config().allow_mixed_batches;
        loop {
            let mut advisor =
                ModelExpertIoAdvisor::new(self.executor.runner(), &self.sequence_states);
            let action = self.scheduler.next_action_with_expert_io_policy(
                &mut self.slot_pool,
                &mut advisor,
                expert_budget,
                allow_mixed_batches,
            )?;
            drop(advisor);
            let Some(action) = action else {
                let pending = self.pending_model_progresses();
                return if pending.is_empty() {
                    Ok(self.no_action_step())
                } else {
                    Ok(ResidentDriverStep::WaitingForModelProgress(pending))
                };
            };
            let step = match action {
                SchedulerAction::DecodeBatch(actions) if proposal_enabled => {
                    self.execute_speculative_decode_batch(actions, on_token)?
                }
                SchedulerAction::Execute { .. } if proposal_enabled => {
                    return Err(Error::Internal(
                        "proposal-enabled scheduler produced a mixed action while mixed dispatch is disabled"
                            .into(),
                    ));
                }
                action => self.execute_planned_action(action, on_token)?,
            };
            if matches!(step, ResidentDriverStep::WaitingForModelProgress(_)) {
                if proposal_enabled {
                    return Ok(step);
                }
                continue;
            }
            return Ok(step);
        }
    }

    fn execute_speculative_decode_batch<F>(
        &mut self,
        actions: Vec<DecodeAction>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        if actions.is_empty() {
            return Err(Error::Internal(
                "production speculative decode batch cannot be empty".into(),
            ));
        }

        match self.try_execute_speculative_decode_batch(&actions, on_token) {
            Ok(step) => Ok(step),
            Err(error) => {
                let first = &actions[0];
                tracing::error!(
                    target: "ferrule_speculative_cycle",
                    event = "speculative_cohort_failed",
                    request_id = first.request_id.map_or(0, |request_id| request_id.0),
                    has_request_id = first.request_id.is_some(),
                    session_id = first.session_id.0,
                    position = first.position,
                    anchor_token = first.token_id,
                    cohort_size = actions.len(),
                    error = %error,
                    "production speculative cohort failed"
                );
                Err(error)
            }
        }
    }

    fn try_execute_speculative_decode_batch<F>(
        &mut self,
        actions: &[DecodeAction],
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transaction = self.take_transaction_id()?;
        let cohort_start = Instant::now();
        let proposal_source = match self.executor.runner().native_proposal_source() {
            Ok(Some(proposal_source)) => proposal_source,
            Ok(None) => {
                return Err(self.abort_speculative_decode_batch(
                    actions,
                    Error::Execution(
                        "resident decode entered proposal verification without model capability"
                            .into(),
                    ),
                    "production speculative initialization",
                ));
            }
            Err(error) => {
                return Err(self.abort_speculative_decode_batch(
                    actions,
                    error,
                    "production speculative initialization",
                ));
            }
        };
        if let Err(error) = proposal_source.validate() {
            return Err(self.abort_speculative_decode_batch(
                actions,
                error,
                "production speculative initialization",
            ));
        }

        let slots = match self.prepare_native_proposal_slots(actions) {
            Ok(slots) => slots,
            Err(error) => {
                return Err(self.abort_speculative_decode_batch(
                    actions,
                    error,
                    "production speculative proposal preparation",
                ));
            }
        };
        let session_ids = actions
            .iter()
            .map(|action| action.session_id)
            .collect::<Vec<_>>();
        let (schedules, source_states) =
            match self.claim_transaction_sessions(transaction, &session_ids) {
                Ok(ownership) => ownership,
                Err(error) => {
                    return Err(self.abort_speculative_decode_batch(
                        actions,
                        error,
                        "production speculative state collection",
                    ));
                }
            };
        self.advance_native_proposal_cohort(
            PendingNativeProposalCohort {
                transaction,
                cohort_start,
                actions: actions.to_vec(),
                proposal_source,
                source_states,
                schedules,
                slots,
                cancellation_request: None,
                cancellation_error: None,
            },
            on_token,
        )
    }

    fn advance_native_proposal_cohort<F>(
        &mut self,
        mut pending: PendingNativeProposalCohort<R::SequenceState>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        debug_assert_eq!(pending.actions.len(), pending.source_states.len());
        debug_assert_eq!(pending.actions.len(), pending.slots.len());
        let mut first_error = None;
        for slot_index in 0..pending.slots.len() {
            if matches!(
                &pending.slots[slot_index].status,
                NativeProposalSlotStatus::NotStarted
            ) {
                pending.slots[slot_index].proposal_start = Some(Instant::now());
            }
            let progress =
                match &pending.slots[slot_index].status {
                    NativeProposalSlotStatus::NotStarted => {
                        let anchor_token = pending.actions[slot_index].token_id;
                        Some(self.executor.with_sequence_state(
                            &mut pending.source_states[slot_index],
                            |runner| {
                                runner.begin_native_proposal(pending.transaction, anchor_token)
                            },
                        ))
                    }
                    NativeProposalSlotStatus::Waiting(waiting) => {
                        let continuation = waiting.continuation();
                        Some(self.executor.with_sequence_state(
                            &mut pending.source_states[slot_index],
                            |runner| {
                                runner.resume_native_proposal(pending.transaction, continuation)
                            },
                        ))
                    }
                    NativeProposalSlotStatus::Complete { .. } => None,
                };
            if let Some(progress) = progress {
                match progress {
                    Ok(NativeProposalProgress::Complete(proposal)) => {
                        pending.slots[slot_index].status = NativeProposalSlotStatus::Complete {
                            proposal,
                            proposal_time_us: pending.slots[slot_index]
                                .proposal_start
                                .as_ref()
                                .expect("a completed native proposal was started")
                                .elapsed()
                                .as_micros() as u64,
                            prepared: Box::new(None),
                        };
                    }
                    Ok(NativeProposalProgress::Waiting(waiting)) => {
                        pending.slots[slot_index].status =
                            NativeProposalSlotStatus::Waiting(waiting);
                    }
                    Err(error) => {
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                }
            }
            if let Err(error) = self.prepare_completed_native_proposal_slot(
                &mut pending.slots[slot_index],
                pending.proposal_source,
            ) && first_error.is_none()
            {
                first_error = Some(error);
            }
        }

        if let Some(error) = first_error {
            return match self.cancel_native_proposal_cohort(pending, None, Some(error)) {
                Ok(()) => Err(Error::Internal(
                    "speculative proposal failure was lost during cancellation".into(),
                )),
                Err(error) => Err(error),
            };
        }

        let waiting = pending
            .slots
            .iter()
            .filter_map(|slot| match &slot.status {
                NativeProposalSlotStatus::Waiting(waiting) => Some(waiting.clone()),
                NativeProposalSlotStatus::NotStarted
                | NativeProposalSlotStatus::Complete { .. } => None,
            })
            .collect::<Vec<_>>();
        if !waiting.is_empty() {
            let transaction = pending.transaction;
            self.speculative_transactions.insert(
                transaction,
                PendingSpeculativeDriverCohort::Proposing(pending),
            );
            self.enqueue_transaction(transaction);
            return Ok(ResidentDriverStep::WaitingForModelProgress(
                self.pending_model_progresses(),
            ));
        }

        let PendingNativeProposalCohort {
            transaction,
            cohort_start,
            actions,
            source_states,
            schedules,
            slots,
            ..
        } = pending;
        let prepared = match Self::take_prepared_speculative_actions(slots) {
            Ok(prepared) => prepared,
            Err(error) => {
                self.restore_transaction_sessions(schedules, source_states)?;
                return Err(self.abort_speculative_decode_batch(
                    &actions,
                    error,
                    "production speculative proposal completion",
                ));
            }
        };
        self.begin_speculative_verification_cohort(
            transaction,
            cohort_start,
            actions,
            prepared,
            source_states,
            schedules,
            on_token,
        )
    }

    fn begin_speculative_verification_cohort<F>(
        &mut self,
        transaction: ExecutionTransactionId,
        cohort_start: Instant,
        actions: Vec<DecodeAction>,
        prepared: Vec<PreparedSpeculativeAction>,
        mut source_states: Vec<R::SequenceState>,
        schedules: Vec<SuspendedSequenceSchedule>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let verification_items = actions
            .iter()
            .zip(&prepared)
            .map(|(action, prepared)| SpeculativeVerificationItem {
                state_slot: prepared.page_slot,
                generation: 0,
                proposal: &prepared.proposal,
                frontier: TargetFrontier {
                    position: action.position,
                    top1: ferrule_common::execution::TokenLogit::new(
                        action.token_id,
                        action.logit.unwrap_or(0.0),
                    ),
                },
            })
            .collect::<Vec<_>>();
        let progress = match self.page_manager.as_mut() {
            Some(page_manager) => begin_resumable_speculative_verification_cohort(
                &mut self.executor,
                page_manager,
                transaction,
                &mut source_states,
                &verification_items,
                self.top_k,
            ),
            None => {
                drop(verification_items);
                self.restore_transaction_sessions(schedules, source_states)?;
                return Err(self.abort_speculative_decode_batch(
                    &actions,
                    Error::Internal(
                        "authoritative KvPageManager disappeared during speculative cohort".into(),
                    ),
                    "production speculative verification",
                ));
            }
        };
        drop(verification_items);

        match progress {
            Ok(SpeculativeCohortProgress::Complete(cohort)) => {
                self.restore_transaction_sessions(schedules, source_states)?;
                let publication_actions = actions.clone();
                match self.publish_speculative_decode_cohort(
                    cohort_start,
                    actions,
                    prepared,
                    cohort,
                    on_token,
                ) {
                    Ok(step) => Ok(step),
                    Err(error) => Err(self.abort_speculative_decode_batch(
                        &publication_actions,
                        error,
                        "production speculative publication",
                    )),
                }
            }
            Ok(SpeculativeCohortProgress::Waiting(verification)) => {
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::Verifying(Box::new(
                        PendingSpeculativeVerificationDriverCohort {
                            transaction,
                            cohort_start,
                            actions,
                            prepared,
                            source_states,
                            schedules,
                            verification: *verification,
                            cancellation_request: None,
                        },
                    )),
                );
                self.enqueue_transaction(transaction);
                Ok(ResidentDriverStep::WaitingForModelProgress(
                    self.pending_model_progresses(),
                ))
            }
            Err(error) => {
                let (error, verification) = error.into_parts();
                if let Some(verification) = verification {
                    self.speculative_transactions.insert(
                        transaction,
                        PendingSpeculativeDriverCohort::Verifying(Box::new(
                            PendingSpeculativeVerificationDriverCohort {
                                transaction,
                                cohort_start,
                                actions,
                                prepared,
                                source_states,
                                schedules,
                                verification: *verification,
                                cancellation_request: None,
                            },
                        )),
                    );
                    self.enqueue_transaction(transaction);
                    Err(error)
                } else {
                    self.restore_transaction_sessions(schedules, source_states)?;
                    Err(self.abort_speculative_decode_batch(
                        &actions,
                        error,
                        "production speculative verification",
                    ))
                }
            }
        }
    }

    fn resume_speculative_transaction<F>(
        &mut self,
        transaction: ExecutionTransactionId,
        on_token: &mut F,
    ) -> Result<Option<ResidentDriverStep>>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let pending = self
            .speculative_transactions
            .remove(&transaction)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "speculative transaction {transaction:?} disappeared"
                ))
            })?;
        match pending {
            PendingSpeculativeDriverCohort::Proposing(pending)
                if pending.cancellation_request.is_some()
                    || pending.cancellation_error.is_some() =>
            {
                let cancellation_request = pending.cancellation_request;
                match self.cancel_native_proposal_cohort(pending, None, None) {
                    Ok(()) => {
                        if let Some(request_id) = cancellation_request {
                            self.cancel_scheduled_request(request_id)?;
                        }
                        Ok(Some(ResidentDriverStep::Executed {
                            action_kind: ResidentActionKind::Cancel,
                            rows: 0,
                            staged: 0,
                            finished: 0,
                        }))
                    }
                    Err(_) if self.speculative_transactions.contains_key(&transaction) => Ok(None),
                    Err(error) => Err(error),
                }
            }
            PendingSpeculativeDriverCohort::Proposing(pending) => {
                let step = self.advance_native_proposal_cohort(pending, on_token)?;
                Ok(
                    (!matches!(step, ResidentDriverStep::WaitingForModelProgress(_)))
                        .then_some(step),
                )
            }
            PendingSpeculativeDriverCohort::Verifying(pending)
                if pending.cancellation_request.is_some() =>
            {
                let request_id = pending
                    .cancellation_request
                    .expect("guarded speculative cancellation request");
                match self.cancel_speculative_verification_cohort(*pending, request_id) {
                    Ok(()) => {
                        self.cancel_scheduled_request(request_id)?;
                        Ok(Some(ResidentDriverStep::Executed {
                            action_kind: ResidentActionKind::Cancel,
                            rows: 0,
                            staged: 0,
                            finished: 0,
                        }))
                    }
                    Err(_) if self.speculative_transactions.contains_key(&transaction) => Ok(None),
                    Err(error) => Err(error),
                }
            }
            PendingSpeculativeDriverCohort::Verifying(pending) => {
                self.resume_speculative_verification_transaction(*pending, on_token)
            }
        }
    }

    fn resume_speculative_verification_transaction<F>(
        &mut self,
        pending: PendingSpeculativeVerificationDriverCohort<R::SequenceState>,
        on_token: &mut F,
    ) -> Result<Option<ResidentDriverStep>>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let PendingSpeculativeVerificationDriverCohort {
            transaction,
            cohort_start,
            actions,
            prepared,
            mut source_states,
            schedules,
            verification,
            cancellation_request,
        } = pending;
        let progress = match self.page_manager.as_mut() {
            Some(page_manager) => resume_resumable_speculative_verification_cohort(
                &mut self.executor,
                page_manager,
                &mut source_states,
                verification,
            ),
            None => {
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::Verifying(Box::new(
                        PendingSpeculativeVerificationDriverCohort {
                            transaction,
                            cohort_start,
                            actions,
                            prepared,
                            source_states,
                            schedules,
                            verification,
                            cancellation_request,
                        },
                    )),
                );
                self.enqueue_transaction(transaction);
                return Err(Error::Internal(
                    "authoritative KvPageManager disappeared while speculative verification was suspended"
                        .into(),
                ));
            }
        };

        match progress {
            Ok(SpeculativeCohortProgress::Complete(cohort)) => {
                self.restore_transaction_sessions(schedules, source_states)?;
                let publication_actions = actions.clone();
                match self.publish_speculative_decode_cohort(
                    cohort_start,
                    actions,
                    prepared,
                    cohort,
                    on_token,
                ) {
                    Ok(step) => Ok(Some(step)),
                    Err(error) => Err(self.abort_speculative_decode_batch(
                        &publication_actions,
                        error,
                        "production speculative resume publication",
                    )),
                }
            }
            Ok(SpeculativeCohortProgress::Waiting(verification)) => {
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::Verifying(Box::new(
                        PendingSpeculativeVerificationDriverCohort {
                            transaction,
                            cohort_start,
                            actions,
                            prepared,
                            source_states,
                            schedules,
                            verification: *verification,
                            cancellation_request,
                        },
                    )),
                );
                self.enqueue_transaction(transaction);
                Ok(None)
            }
            Err(error) => {
                let (error, verification) = error.into_parts();
                if let Some(verification) = verification {
                    self.speculative_transactions.insert(
                        transaction,
                        PendingSpeculativeDriverCohort::Verifying(Box::new(
                            PendingSpeculativeVerificationDriverCohort {
                                transaction,
                                cohort_start,
                                actions,
                                prepared,
                                source_states,
                                schedules,
                                verification: *verification,
                                cancellation_request,
                            },
                        )),
                    );
                    self.enqueue_transaction(transaction);
                    Err(error)
                } else {
                    self.restore_transaction_sessions(schedules, source_states)?;
                    Err(self.abort_speculative_decode_batch(
                        &actions,
                        error,
                        "production speculative resume",
                    ))
                }
            }
        }
    }

    fn publish_speculative_decode_cohort<F>(
        &mut self,
        cohort_start: Instant,
        actions: Vec<DecodeAction>,
        prepared: Vec<PreparedSpeculativeAction>,
        cohort: crate::speculation::SpeculativeCohortResult,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        if cohort.results.len() != actions.len() {
            return Err(Error::Internal(format!(
                "speculative cohort returned {} results for {} actions",
                cohort.results.len(),
                actions.len()
            )));
        }

        let cohort_transaction_time_us = cohort.transaction_time_us;
        let cohort_verify_time_us = cohort.verify_time_us;
        let eos_token_id = self.executor.runner().eos_token_id();
        let mut rows = 0usize;
        let mut staged = 0usize;
        let mut finished = 0usize;

        for ((action, prepared), result) in actions.iter().zip(prepared).zip(cohort.results) {
            let externally_committed = result.accepted.len().checked_add(1).ok_or_else(|| {
                Error::Internal("speculative external token count overflow".into())
            })?;
            if result.accounting.externally_committed_tokens != externally_committed {
                return Err(Error::Internal(format!(
                    "speculative transaction committed {} rows but returned {} external tokens",
                    result.accounting.externally_committed_tokens, externally_committed
                )));
            }

            self.scheduler.commit_decode_action(action)?;
            self.scheduler
                .active_sequence_mut(action.session_id)
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "speculative session {:?} disappeared after anchor commit",
                        action.session_id
                    ))
                })?
                .extend_generated(&result.accepted);
            let runtime_emitted_tokens =
                self.enqueue_speculative_committed_tokens(action, &result.accepted)?;
            if result.accounting.externally_committed_tokens != runtime_emitted_tokens {
                return Err(Error::Internal(format!(
                    "speculative transaction committed {} tokens but invoked {runtime_emitted_tokens} runtime token callbacks",
                    result.accounting.externally_committed_tokens
                )));
            }

            let mut finish_reason = {
                let sequence = self
                    .scheduler
                    .active_sequence(action.session_id)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "speculative session {:?} disappeared before frontier staging",
                            action.session_id
                        ))
                    })?;
                if sequence.generated >= sequence.max_new_tokens {
                    Some(SequenceFinishReason::MaxTokens)
                } else if matched_stop(&sequence.generated_text, &sequence.stop) {
                    Some(SequenceFinishReason::StopString)
                } else if sequence.position >= self.config.ctx_size {
                    Some(SequenceFinishReason::Context)
                } else {
                    None
                }
            };

            let mut action_staged = 0usize;
            if finish_reason.is_none() {
                finish_reason = match result.target_next {
                    None => Some(SequenceFinishReason::NoCandidate),
                    Some(next)
                        if self.config.stop_at_eos
                            && !prepared.sequence.ignore_eos
                            && eos_token_id == Some(next.token_id) =>
                    {
                        Some(SequenceFinishReason::Eos)
                    }
                    Some(next) => {
                        self.scheduler.stage_decode_candidate(
                            action.session_id,
                            next.token_id,
                            Some(next.logit),
                        )?;
                        self.observability.stats.staged_tokens += 1;
                        action_staged = 1;
                        None
                    }
                };
            }

            let mut action_finished = 0usize;
            if let Some(reason) = finish_reason {
                let position = self.scheduler.active_sequence(action.session_id).map_or(
                    action.position.saturating_add(externally_committed),
                    |sequence| sequence.position,
                );
                self.scheduler
                    .finish_sequence(action.session_id, reason, &mut self.slot_pool)?;
                self.finalize_sequence_state(action.session_id, position)?;
                self.observability.stats.finished_sequences += 1;
                action_finished = 1;
            }

            let verified_rows = result.accounting.verified_rows;
            rows = rows.saturating_add(verified_rows);
            staged += action_staged;
            finished += action_finished;
            record_speculative_sequence_metrics(
                &mut self.observability.stats.speculative,
                &result,
                prepared.proposal_time_us,
                runtime_emitted_tokens,
            );
        }

        let complete_cohort_time_us = cohort_start.elapsed().as_micros() as u64;
        record_speculative_cohort_metrics(
            &mut self.observability.stats.speculative,
            cohort_transaction_time_us,
            cohort_verify_time_us,
            complete_cohort_time_us,
        );
        self.observability.stats.actions += 1;
        self.observability.stats.decode_steps += actions.len();

        let metrics = &self.observability.stats.speculative;
        if metrics.cycles <= actions.len() || metrics.cycles.is_multiple_of(64) {
            tracing::info!(
                cycles = metrics.cycles,
                cohort_size = actions.len(),
                proposed_tokens = metrics.proposed_tokens,
                verified_rows = metrics.verified_rows,
                accepted_draft_tokens = metrics.accepted_draft_tokens,
                runtime_emitted_tokens = metrics.runtime_emitted_tokens,
                acceptance = metrics.acceptance_rate(),
                verify_us = cohort_verify_time_us,
                transaction_us = cohort_transaction_time_us,
                cohort_us = complete_cohort_time_us,
                "production speculative cohort"
            );
        }

        self.flush_committed_token_outbox(on_token)?;
        Ok(ResidentDriverStep::Executed {
            action_kind: ResidentActionKind::Decode,
            rows,
            staged,
            finished,
        })
    }

    fn prepare_native_proposal_slots(
        &self,
        actions: &[DecodeAction],
    ) -> Result<Vec<PendingNativeProposalSlot>> {
        let mut slots = Vec::with_capacity(actions.len());

        for action in actions {
            let sequence = self
                .scheduler
                .active_sequence(action.session_id)
                .cloned()
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "cannot execute speculative for inactive session {:?}",
                        action.session_id
                    ))
                })?;
            if sequence.request_id != action.request_id
                || sequence.kv_handle != action.kv_handle
                || sequence.position != action.position
                || sequence.next_decode_token != Some(action.token_id)
            {
                return Err(Error::Internal(format!(
                    "speculative action no longer matches session {:?}: action(request={:?}, kv={:?}, position={}, token={}), sequence(request={:?}, kv={:?}, position={}, token={:?})",
                    action.session_id,
                    action.request_id,
                    action.kv_handle,
                    action.position,
                    action.token_id,
                    sequence.request_id,
                    sequence.kv_handle,
                    sequence.position,
                    sequence.next_decode_token,
                )));
            }

            let remaining_output = sequence.max_new_tokens.saturating_sub(sequence.generated);
            let remaining_context = self.config.ctx_size.saturating_sub(sequence.position);
            let commit_capacity = remaining_output.min(remaining_context);
            if commit_capacity == 0 {
                return Err(Error::Internal(format!(
                    "speculative decode action for session {:?} has no output/context capacity",
                    action.session_id
                )));
            }
            let max_drafts = commit_capacity.saturating_sub(1);
            let page_slot = *self.page_slots.get(&action.session_id).ok_or_else(|| {
                Error::Internal(format!(
                    "speculative session {:?} has no authoritative page slot",
                    action.session_id
                ))
            })?;

            let status = if max_drafts == 0 {
                NativeProposalSlotStatus::Complete {
                    proposal: NativeProposal {
                        token_ids: Vec::new(),
                        confidence_logits: Vec::new(),
                    },
                    proposal_time_us: 0,
                    prepared: Box::new(None),
                }
            } else {
                NativeProposalSlotStatus::NotStarted
            };
            slots.push(PendingNativeProposalSlot {
                sequence,
                page_slot,
                max_drafts,
                proposal_start: None,
                status,
            });
        }

        Ok(slots)
    }

    fn prepare_completed_native_proposal_slot(
        &self,
        slot: &mut PendingNativeProposalSlot,
        proposal_source: NativeProposalSource,
    ) -> Result<()> {
        let NativeProposalSlotStatus::Complete {
            proposal,
            proposal_time_us,
            prepared,
        } = &mut slot.status
        else {
            return Ok(());
        };
        let prepared = prepared.as_mut();
        if prepared.is_some() {
            return Ok(());
        }

        if slot.max_drafts != 0 {
            proposal.validate_for_source(proposal_source)?;
        } else {
            proposal.validate()?;
        }
        let anchor_token_id = slot.sequence.next_decode_token.ok_or_else(|| {
            Error::Internal(format!(
                "speculative sequence {:?} lost its validated anchor token",
                slot.sequence.session_id
            ))
        })?;
        let mut proposal_tokens = std::mem::take(&mut proposal.token_ids);
        let mut confidence_logits = std::mem::take(&mut proposal.confidence_logits);
        let capacity_width = proposal_tokens.len().min(slot.max_drafts);
        proposal_tokens.truncate(capacity_width);
        confidence_logits.truncate(capacity_width);
        proposal_tokens = self.truncate_native_proposal_at_output_boundary(
            &slot.sequence,
            anchor_token_id,
            proposal_tokens,
        )?;
        confidence_logits.truncate(proposal_tokens.len());
        let confidence_width = confident_proposal_prefix_length(
            &confidence_logits,
            self.config.proposal_confidence_threshold,
        )?;
        proposal_tokens.truncate(confidence_width);
        *prepared = Some(PreparedSpeculativeAction {
            sequence: slot.sequence.clone(),
            page_slot: slot.page_slot,
            proposal: proposal_tokens,
            proposal_time_us: *proposal_time_us,
        });
        Ok(())
    }

    fn take_prepared_speculative_actions(
        slots: Vec<PendingNativeProposalSlot>,
    ) -> Result<Vec<PreparedSpeculativeAction>> {
        let mut prepared_actions = Vec::with_capacity(slots.len());
        for slot in slots {
            match slot.status {
                NativeProposalSlotStatus::Complete { prepared, .. } => match *prepared {
                    Some(prepared) => prepared_actions.push(prepared),
                    None => {
                        return Err(Error::Internal(
                            "speculative proposal cohort completed with an unfinished slot".into(),
                        ));
                    }
                },
                NativeProposalSlotStatus::NotStarted | NativeProposalSlotStatus::Waiting(_) => {
                    return Err(Error::Internal(
                        "speculative proposal cohort completed with an unfinished slot".into(),
                    ));
                }
            }
        }
        Ok(prepared_actions)
    }

    fn truncate_native_proposal_at_output_boundary(
        &self,
        sequence: &SequenceState,
        anchor_token_id: u32,
        proposal: Vec<u32>,
    ) -> Result<Vec<u32>> {
        let eos_token_id = self.executor.runner().eos_token_id();
        if self.config.stop_at_eos && !sequence.ignore_eos && eos_token_id == Some(anchor_token_id)
        {
            return Err(Error::Internal(
                "an EOS token must not be staged as a speculative anchor".into(),
            ));
        }

        let mut decode_state = sequence.incremental_decode.clone();
        let mut generated_text = sequence.generated_text.clone();
        let anchor_text = self
            .executor
            .runner()
            .decode_incremental(anchor_token_id, &mut decode_state)?
            .unwrap_or_default();
        generated_text.push_str(&anchor_text);
        if matched_stop(&generated_text, &sequence.stop) {
            return Ok(Vec::new());
        }

        let mut admitted = Vec::with_capacity(proposal.len());
        for token_id in proposal {
            if self.config.stop_at_eos && !sequence.ignore_eos && eos_token_id == Some(token_id) {
                break;
            }
            let text = self
                .executor
                .runner()
                .decode_incremental(token_id, &mut decode_state)?
                .unwrap_or_default();
            generated_text.push_str(&text);
            admitted.push(token_id);
            if matched_stop(&generated_text, &sequence.stop) {
                break;
            }
        }
        Ok(admitted)
    }

    fn enqueue_speculative_committed_tokens(
        &mut self,
        action: &DecodeAction,
        accepted: &[u32],
    ) -> Result<usize> {
        let mut tokens = Vec::with_capacity(accepted.len() + 1);
        tokens.push((action.token_id, action.logit));
        tokens.extend(accepted.iter().copied().map(|token| (token, None)));
        let emitted_tokens = tokens.len();
        let runner = self.executor.runner();
        let sequence = self
            .scheduler
            .active_sequence_mut(action.session_id)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "cannot emit speculative block for inactive session {:?}",
                    action.session_id
                ))
            })?;
        let start_index = sequence
            .generated
            .checked_sub(tokens.len())
            .ok_or_else(|| {
                Error::Internal(
                    "speculative emitted block exceeds committed generation count".into(),
                )
            })?;
        for (offset, (token, logit)) in tokens.into_iter().enumerate() {
            let text = runner
                .decode_incremental(token, &mut sequence.incremental_decode)?
                .unwrap_or_default();
            sequence.append_generated_text(&text);
            let event = ResidentTokenEvent {
                session_id: sequence.session_id,
                request_id: sequence.request_id,
                index: start_index + offset,
                token,
                logit,
                text,
            };
            self.committed_token_outbox.push_back(event);
        }
        Ok(emitted_tokens)
    }

    fn abort_speculative_decode_batch(
        &mut self,
        actions: &[DecodeAction],
        error: Error,
        stage: &'static str,
    ) -> Error {
        let mut cleanup_errors = Vec::new();
        for action in actions {
            if self.scheduler.active_sequence(action.session_id).is_some() {
                if let Err(cleanup) = self
                    .scheduler
                    .fail_sequence(action.session_id, &mut self.slot_pool)
                {
                    cleanup_errors.push(format!(
                        "session {:?} scheduler cleanup failed: {cleanup}",
                        action.session_id
                    ));
                }
                if let Err(cleanup) = self.release_sequence_state(action.session_id) {
                    cleanup_errors.push(format!(
                        "session {:?} state cleanup failed: {cleanup}",
                        action.session_id
                    ));
                }
            }
        }
        if cleanup_errors.is_empty() {
            error
        } else {
            Error::Internal(format!(
                "{stage} failed ({error}); {}",
                cleanup_errors.join("; ")
            ))
        }
    }
}

struct PreparedSpeculativeAction {
    sequence: SequenceState,
    page_slot: StateSlot,
    proposal: Vec<u32>,
    proposal_time_us: u64,
}

fn record_speculative_sequence_metrics(
    metrics: &mut SpeculativeMetrics,
    result: &SpeculativeCycleResult,
    proposal_time_us: u64,
    runtime_emitted_tokens: usize,
) {
    let accounting = result.accounting;
    metrics.cycles = metrics.cycles.saturating_add(1);
    metrics.proposed_tokens = metrics
        .proposed_tokens
        .saturating_add(accounting.proposed_tokens);
    metrics.verified_rows = metrics
        .verified_rows
        .saturating_add(accounting.verified_rows);
    metrics.accepted_draft_tokens = metrics
        .accepted_draft_tokens
        .saturating_add(accounting.accepted_draft_tokens);
    metrics.correction_tokens = metrics
        .correction_tokens
        .saturating_add(accounting.correction_tokens);
    metrics.externally_committed_tokens = metrics
        .externally_committed_tokens
        .saturating_add(accounting.externally_committed_tokens);
    metrics.rolled_back_rows = metrics
        .rolled_back_rows
        .saturating_add(accounting.rolled_back_rows);
    metrics.rejected_tokens = metrics
        .rejected_tokens
        .saturating_add(result.rejected.is_some() as usize);
    if metrics.accepted_prefix_histogram.len() <= accounting.accepted_draft_tokens {
        metrics
            .accepted_prefix_histogram
            .resize(accounting.accepted_draft_tokens + 1, 0);
    }
    metrics.accepted_prefix_histogram[accounting.accepted_draft_tokens] =
        metrics.accepted_prefix_histogram[accounting.accepted_draft_tokens].saturating_add(1);
    metrics.total_proposal_time_us = metrics
        .total_proposal_time_us
        .saturating_add(proposal_time_us);
    metrics.record_runtime_emitted_tokens(runtime_emitted_tokens);
}

fn record_speculative_cohort_metrics(
    metrics: &mut SpeculativeMetrics,
    transaction_time_us: u64,
    verify_time_us: u64,
    complete_cohort_time_us: u64,
) {
    metrics.total_transaction_time_us = metrics
        .total_transaction_time_us
        .saturating_add(transaction_time_us);
    metrics.total_verify_time_us = metrics.total_verify_time_us.saturating_add(verify_time_us);
    metrics.total_cycle_time_us = metrics
        .total_cycle_time_us
        .saturating_add(complete_cohort_time_us);
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

fn action_session_ids(action: &SchedulerAction) -> Vec<SessionId> {
    let mut sessions = match action {
        SchedulerAction::Execute { prefills, decodes } => prefills
            .iter()
            .map(|action| action.session_id)
            .chain(decodes.iter().map(|action| action.session_id))
            .collect::<Vec<_>>(),
        SchedulerAction::PrefillChunk(prefill) => vec![prefill.session_id],
        SchedulerAction::DecodeBatch(actions) => {
            actions.iter().map(|action| action.session_id).collect()
        }
        SchedulerAction::Finish { session_id, .. } | SchedulerAction::Cancel { session_id, .. } => {
            vec![*session_id]
        }
    };
    sessions.sort_unstable_by_key(|session| session.0);
    sessions.dedup();
    sessions
}

fn action_contains_request(action: &SchedulerAction, request_id: RequestId) -> bool {
    match action {
        SchedulerAction::Execute { prefills, decodes } => {
            prefills
                .iter()
                .any(|action| action.request_id == Some(request_id))
                || decodes
                    .iter()
                    .any(|action| action.request_id == Some(request_id))
        }
        SchedulerAction::PrefillChunk(prefill) => prefill.request_id == Some(request_id),
        SchedulerAction::DecodeBatch(actions) => actions
            .iter()
            .any(|action| action.request_id == Some(request_id)),
        SchedulerAction::Finish {
            request_id: owner, ..
        }
        | SchedulerAction::Cancel {
            request_id: owner, ..
        } => *owner == Some(request_id),
    }
}

fn action_decode_actions(action: &SchedulerAction) -> &[DecodeAction] {
    match action {
        SchedulerAction::Execute { decodes, .. } | SchedulerAction::DecodeBatch(decodes) => decodes,
        SchedulerAction::PrefillChunk(_)
        | SchedulerAction::Finish { .. }
        | SchedulerAction::Cancel { .. } => &[],
    }
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
    use std::collections::{HashMap, VecDeque};

    use ferrule_common::execution::{
        ExecutionIntent, ForwardPhase, KvLayoutSchema, KvPlaneDescriptor, LogitsOutput,
        LogitsRequest, LogitsRow,
    };
    use ferrule_common::{Error, Result};
    use ferrule_model::{
        BatchContinuationCancelOutcome, BatchContinuationId, ExpertIoModelRunner, ModelInfo,
        ModelRunner, MultiSessionBatchProgress, MultiSessionRunner, NativeProposal,
        NativeProposalProgress, NativeProposalSource, PendingExpertLoad, PendingModelProgress,
        ResidentModelRunner, TokenLogit,
    };

    use crate::scheduling::FixedSequenceSlotPool;
    use crate::scheduling::{RequestId, SequenceStatus};

    use super::*;

    impl<R, C> ResidentTopKDriver<R, C>
    where
        R: ResidentModelRunner,
        C: SequenceSlotPool,
    {
        fn drive_ready_test_work<F>(&mut self, mut on_token: F) -> Result<ResidentTopKDriverStats>
        where
            F: FnMut(&ResidentTokenEvent) -> Result<()>,
        {
            loop {
                match self.step_with_model_expert_io(&mut on_token, ExpertIoBudget::unbounded())? {
                    ResidentDriverStep::Idle => return Ok(self.stats().clone()),
                    ResidentDriverStep::Executed { .. } => {}
                    ResidentDriverStep::Blocked => {
                        return Err(Error::Execution(
                            "test driver blocked without asynchronous progress".into(),
                        ));
                    }
                    ResidentDriverStep::WaitingForModelProgress(progress) => {
                        return Err(Error::Execution(format!(
                            "test driver requires completion owner for {} pending operation(s)",
                            progress.len()
                        )));
                    }
                }
            }
        }
    }

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
    struct MockPendingProposal {
        proposal: NativeProposal,
        waits_remaining: usize,
    }

    #[derive(Debug)]
    struct MockTopKRunner {
        completion_hub: ferrule_common::CompletionHub,
        position: usize,
        eos: Option<u32>,
        outputs: VecDeque<Vec<TokenLogit>>,
        fed: Vec<u32>,
        prefills: Vec<Vec<u32>>,
        fail_next_mutation: bool,
        mutation_calls: usize,
        released_sequence_states: usize,
        released_kv_pages: Vec<ferrule_common::execution::KvPageId>,
        native_proposals: VecDeque<NativeProposal>,
        native_proposal_enabled: bool,
        packed_predictions: VecDeque<Vec<TokenLogit>>,
        packed_committed_calls: usize,
        packed_verification_calls: usize,
        proposal_waits: VecDeque<usize>,
        active_native_proposals: HashMap<BatchContinuationId, MockPendingProposal>,
        next_native_proposal_continuation: u64,
        native_proposal_begin_calls: usize,
        native_proposal_resume_calls: usize,
        native_proposal_cancel_calls: usize,
        native_proposal_resume_errors_remaining: usize,
        native_proposal_cancel_still_active_remaining: usize,
        cancelled_native_proposals: Vec<BatchContinuationId>,
        committed_resumable_armed: bool,
        resumable_wait_scripts: VecDeque<usize>,
        active_resumable_waits: HashMap<ExecutionTransactionId, usize>,
        resume_errors_remaining: usize,
        active_resumable_predictions: HashMap<ExecutionTransactionId, Vec<TokenLogit>>,
        resumable_cancel_still_active_remaining: usize,
        cancelled_continuations: Vec<BatchContinuationId>,
        reject_topology_mutation_while_packed: bool,
        topology_mutation_attempts_while_packed: usize,
        sequence_state_fork_calls: usize,
        prepared_batches: usize,
        committed_batches: usize,
        rolled_back_batches: usize,
        paged: bool,
        expert_io_resource_control_installed: bool,
    }

    impl MockTopKRunner {
        fn new(outputs: Vec<Vec<TokenLogit>>) -> Self {
            Self {
                completion_hub: ferrule_common::CompletionHub::new(),
                position: 0,
                eos: None,
                outputs: outputs.into(),
                fed: Vec::new(),
                prefills: Vec::new(),
                fail_next_mutation: false,
                mutation_calls: 0,
                released_sequence_states: 0,
                released_kv_pages: Vec::new(),
                native_proposals: VecDeque::new(),
                native_proposal_enabled: false,
                packed_predictions: VecDeque::new(),
                packed_committed_calls: 0,
                packed_verification_calls: 0,
                proposal_waits: VecDeque::new(),
                active_native_proposals: HashMap::new(),
                next_native_proposal_continuation: 100,
                native_proposal_begin_calls: 0,
                native_proposal_resume_calls: 0,
                native_proposal_cancel_calls: 0,
                native_proposal_resume_errors_remaining: 0,
                native_proposal_cancel_still_active_remaining: 0,
                cancelled_native_proposals: Vec::new(),
                committed_resumable_armed: false,
                resumable_wait_scripts: VecDeque::new(),
                active_resumable_waits: HashMap::new(),
                resume_errors_remaining: 0,
                active_resumable_predictions: HashMap::new(),
                resumable_cancel_still_active_remaining: 0,
                cancelled_continuations: Vec::new(),
                reject_topology_mutation_while_packed: false,
                topology_mutation_attempts_while_packed: 0,
                sequence_state_fork_calls: 0,
                prepared_batches: 0,
                committed_batches: 0,
                rolled_back_batches: 0,
                paged: false,
                expert_io_resource_control_installed: false,
            }
        }

        fn with_speculative_cycle(
            self,
            proposal: NativeProposal,
            target_row_top1: Vec<TokenLogit>,
        ) -> Self {
            self.with_speculative_cohort(vec![proposal], target_row_top1)
        }

        fn with_speculative_cohort(
            mut self,
            proposals: Vec<NativeProposal>,
            target_row_top1: Vec<TokenLogit>,
        ) -> Self {
            self.native_proposals.extend(proposals);
            self.native_proposal_enabled = true;
            self.packed_predictions.push_back(target_row_top1);
            self.paged = true;
            self
        }

        fn with_resumable_wait_scripts(mut self, waits: impl IntoIterator<Item = usize>) -> Self {
            self.committed_resumable_armed = true;
            self.resumable_wait_scripts.extend(waits);
            self
        }

        fn with_resumable_cancel_still_active(mut self, attempts: usize) -> Self {
            self.resumable_cancel_still_active_remaining = attempts;
            self
        }

        fn with_proposal_waits(mut self, waits: Vec<usize>) -> Self {
            self.proposal_waits = waits.into();
            self
        }

        fn with_proposal_resume_errors(mut self, errors: usize) -> Self {
            self.native_proposal_resume_errors_remaining = errors;
            self
        }

        fn with_proposal_cancel_still_active(mut self, attempts: usize) -> Self {
            self.native_proposal_cancel_still_active_remaining = attempts;
            self
        }

        fn with_committed_resumable_batch(
            mut self,
            waits: usize,
            predictions: Vec<TokenLogit>,
        ) -> Self {
            assert!(waits > 0);
            self.committed_resumable_armed = true;
            self.resumable_wait_scripts.push_back(waits);
            self.packed_predictions.push_back(predictions);
            self.paged = true;
            self
        }

        fn with_resume_errors(mut self, errors: usize) -> Self {
            self.resume_errors_remaining = errors;
            self
        }

        fn with_packed_topology_guard(mut self) -> Self {
            self.reject_topology_mutation_while_packed = true;
            self
        }

        fn ensure_packed_topology_quiescent(&mut self, operation: &str) -> Result<()> {
            if self.reject_topology_mutation_while_packed
                && !self.active_resumable_predictions.is_empty()
            {
                self.topology_mutation_attempts_while_packed += 1;
                return Err(Error::Execution(format!(
                    "cannot {operation} while a mock packed transaction is outstanding"
                )));
            }
            Ok(())
        }

        fn failing_next_mutation(mut self) -> Self {
            self.fail_next_mutation = true;
            self
        }

        fn with_eos(mut self, eos: u32) -> Self {
            self.eos = Some(eos);
            self
        }

        fn pending_resumable_progress(transaction: ExecutionTransactionId) -> PendingModelProgress {
            let continuation = BatchContinuationId::new(transaction.get()).unwrap();
            PendingModelProgress::new(
                transaction,
                continuation,
                vec![PendingExpertLoad::new(transaction.get(), 0, 0).unwrap()],
            )
            .unwrap()
        }

        fn pending_native_proposal(
            transaction: ExecutionTransactionId,
            continuation: BatchContinuationId,
        ) -> PendingModelProgress {
            PendingModelProgress::new(transaction, continuation, Vec::new()).unwrap()
        }

        fn complete_packed_batch(
            &mut self,
            states: &mut [MockSequenceState],
            batch: &ferrule_common::execution::ExecutionBatch,
            predictions: Vec<TokenLogit>,
        ) -> Result<ExecutionOutput> {
            if states.len() != batch.sequences().len() {
                return Err(Error::Internal(format!(
                    "mock packed batch state/sequence mismatch: states={} sequences={}",
                    states.len(),
                    batch.sequences().len()
                )));
            }
            if predictions.len() != batch.token_ids().len() {
                return Err(Error::Model(format!(
                    "mock packed predictions {} do not match {} input rows",
                    predictions.len(),
                    batch.token_ids().len()
                )));
            }
            for (state, sequence) in states.iter_mut().zip(batch.sequences()) {
                let query_start = sequence.query.start as usize;
                let query_end = sequence.query.end as usize;
                state.position = state
                    .position
                    .checked_add(query_end - query_start)
                    .ok_or_else(|| Error::Internal("mock packed position overflow".into()))?;
                state
                    .fed
                    .extend_from_slice(&batch.token_ids()[query_start..query_end]);
            }
            let logits = predictions
                .into_iter()
                .enumerate()
                .filter(|(row, _)| matches!(batch.logits()[*row], LogitsRequest::TopK(_)))
                .map(|(row, top1)| LogitsRow::new(row as u32, LogitsOutput::TopK(vec![top1])))
                .collect();
            Ok(ExecutionOutput::new(logits))
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

    impl ResidentModelRunner for MockTopKRunner {
        type ObservabilitySnapshot = ();

        fn observability_snapshot(&self) -> Self::ObservabilitySnapshot {}

        fn completion_hub(&self) -> ferrule_common::CompletionHub {
            self.completion_hub.clone()
        }

        fn take_completion_reactors(&mut self) -> Vec<ferrule_model::ModelCompletionReactor> {
            Vec::new()
        }

        fn native_proposal_source(&self) -> Result<Option<NativeProposalSource>> {
            Ok(self
                .native_proposal_enabled
                .then_some(NativeProposalSource {
                    implementation: "mock-speculative-v1",
                    prepared_plan_id: 0xfeed,
                    native_width: 2,
                }))
        }

        fn begin_native_proposal(
            &mut self,
            transaction: ExecutionTransactionId,
            _anchor_token_id: u32,
        ) -> Result<NativeProposalProgress> {
            self.native_proposal_begin_calls += 1;
            let proposal = self
                .native_proposals
                .pop_front()
                .ok_or_else(|| Error::Model("mock speculative proposal queue is empty".into()))?;
            let waits = self.proposal_waits.pop_front().unwrap_or(0);
            if waits == 0 {
                return Ok(NativeProposalProgress::Complete(proposal));
            }
            let continuation = BatchContinuationId::new(self.next_native_proposal_continuation)?;
            self.next_native_proposal_continuation = self
                .next_native_proposal_continuation
                .checked_add(1)
                .ok_or_else(|| Error::Internal("mock proposal continuation overflow".into()))?;
            let replaced = self.active_native_proposals.insert(
                continuation,
                MockPendingProposal {
                    proposal,
                    waits_remaining: waits - 1,
                },
            );
            debug_assert!(replaced.is_none());
            Ok(NativeProposalProgress::Waiting(
                Self::pending_native_proposal(transaction, continuation),
            ))
        }

        fn resume_native_proposal(
            &mut self,
            transaction: ExecutionTransactionId,
            continuation: BatchContinuationId,
        ) -> Result<NativeProposalProgress> {
            self.native_proposal_resume_calls += 1;
            if self.native_proposal_resume_errors_remaining > 0 {
                self.native_proposal_resume_errors_remaining -= 1;
                return Err(Error::Model(
                    "simulated native proposal resume failure".into(),
                ));
            }
            let mut pending = self
                .active_native_proposals
                .remove(&continuation)
                .ok_or_else(|| {
                    Error::Execution("mock received an unknown proposal continuation".into())
                })?;
            if pending.waits_remaining > 0 {
                pending.waits_remaining -= 1;
                self.active_native_proposals.insert(continuation, pending);
                return Ok(NativeProposalProgress::Waiting(
                    Self::pending_native_proposal(transaction, continuation),
                ));
            }
            Ok(NativeProposalProgress::Complete(pending.proposal))
        }

        fn cancel_native_proposal(
            &mut self,
            _transaction: ExecutionTransactionId,
            continuation: BatchContinuationId,
        ) -> BatchContinuationCancelOutcome {
            self.native_proposal_cancel_calls += 1;
            if !self.active_native_proposals.contains_key(&continuation) {
                return BatchContinuationCancelOutcome::StillActive(Error::Execution(
                    "mock received an unknown proposal continuation".into(),
                ));
            }
            if self.native_proposal_cancel_still_active_remaining > 0 {
                self.native_proposal_cancel_still_active_remaining -= 1;
                return BatchContinuationCancelOutcome::StillActive(Error::Model(
                    "simulated active native proposal cancellation".into(),
                ));
            }
            self.active_native_proposals.remove(&continuation);
            self.cancelled_native_proposals.push(continuation);
            BatchContinuationCancelOutcome::Cancelled
        }
    }

    impl ExpertIoModelRunner for MockTopKRunner {
        type ExpertIoBatchState = ();
        type ExpertIoAdmission = ();

        fn expert_io_resource_limits(
            &self,
        ) -> Result<ferrule_common::expert_io::ExpertIoResourceLimits> {
            Ok(ferrule_common::expert_io::ExpertIoResourceLimits::default())
        }

        fn install_expert_io_resource_control(
            &mut self,
            _control: Box<dyn ferrule_common::expert_io::ExpertIoResourceControl>,
        ) -> Result<()> {
            if self.expert_io_resource_control_installed {
                return Err(Error::Execution(
                    "mock expert-I/O resource control is already installed".into(),
                ));
            }
            self.expert_io_resource_control_installed = true;
            Ok(())
        }

        fn uninstall_expert_io_resource_control(&mut self) -> Result<()> {
            if !self.expert_io_resource_control_installed {
                return Err(Error::Execution(
                    "mock expert-I/O resource control is not installed".into(),
                ));
            }
            self.expert_io_resource_control_installed = false;
            Ok(())
        }

        fn begin_expert_io_batch(&self) -> Self::ExpertIoBatchState {}

        fn estimate_expert_io(
            &self,
            _batch: &mut Self::ExpertIoBatchState,
            _sequence: &Self::SequenceState,
            _phase: ferrule_common::expert_io::ExpertIoPhase,
            _token_ids: &[u32],
        ) -> Result<(
            ferrule_common::expert_io::ExpertIoEstimate,
            Self::ExpertIoAdmission,
        )> {
            Ok((ferrule_common::expert_io::ExpertIoEstimate::default(), ()))
        }

        fn admit_expert_io(
            &self,
            _batch: &mut Self::ExpertIoBatchState,
            _admission: Self::ExpertIoAdmission,
        ) {
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

    fn complete_mock_committed_batch(
        states: &mut [MockSequenceState],
        batch: &ferrule_common::execution::ExecutionBatch,
    ) -> Result<ExecutionOutput> {
        let mut output_rows = Vec::new();
        for sequence in batch.sequences() {
            let state_index = sequence
                .state_slot
                .try_as_usize()
                .map_err(|_| Error::Internal("mock packed state slot exceeds usize".into()))?;
            let state = states.get_mut(state_index).ok_or_else(|| {
                Error::Internal(format!(
                    "mock packed state slot {state_index} is out of range"
                ))
            })?;
            let query_start = usize::try_from(sequence.query.start)
                .map_err(|_| Error::Internal("mock packed query start exceeds usize".into()))?;
            let query_end = usize::try_from(sequence.query.end)
                .map_err(|_| Error::Internal("mock packed query end exceeds usize".into()))?;
            let token_ids = &batch.token_ids()[query_start..query_end];
            state.position = state
                .position
                .checked_add(token_ids.len())
                .ok_or_else(|| Error::Internal("mock packed position overflow".into()))?;
            match sequence.phase {
                ForwardPhase::Prefill => state.prefills.push(token_ids.to_vec()),
                ForwardPhase::Decode => state.fed.extend_from_slice(token_ids),
            }
            state.mutation_calls += 1;
            if std::mem::take(&mut state.fail_next_mutation) {
                return Err(Error::Model(
                    "simulated failure after partial runner mutation".into(),
                ));
            }
            for row in query_start..query_end {
                if matches!(batch.logits()[row], LogitsRequest::TopK(_)) {
                    let logits = state.outputs.pop_front().unwrap_or_default();
                    output_rows.push(LogitsRow::new(
                        u32::try_from(row)
                            .map_err(|_| Error::Internal("mock packed row exceeds u32".into()))?,
                        LogitsOutput::TopK(logits),
                    ));
                }
            }
        }
        Ok(ExecutionOutput::new(output_rows))
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
            state.prefills.append(&mut self.prefills);

            self.position = saved_position;
            self.outputs = saved_outputs;
            self.fed = saved_fed;
            self.fail_next_mutation = saved_fail;
            self.mutation_calls = saved_calls;

            result
        }

        fn create_sequence_state(&mut self) -> Result<Self::SequenceState> {
            let mut state = MockSequenceState::new(0, &self.outputs);
            state.fail_next_mutation = self.fail_next_mutation;
            Ok(state)
        }

        fn fork_sequence_state(&mut self) -> Result<Self::SequenceState> {
            self.ensure_packed_topology_quiescent("fork the active sequence")?;
            self.sequence_state_fork_calls += 1;
            let mut state = MockSequenceState::new(0, &self.outputs);
            state.fail_next_mutation = self.fail_next_mutation;
            Ok(state)
        }

        fn fork_sequence_state_from(
            &mut self,
            source: &Self::SequenceState,
            expected_position: usize,
        ) -> Result<Self::SequenceState> {
            self.ensure_packed_topology_quiescent("fork explicit sequence state")?;
            self.sequence_state_fork_calls += 1;
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
            self.ensure_packed_topology_quiescent("reset sequence state")?;
            state.position = 0;
            state.fed.clear();
            state.prefills.clear();
            state.mutation_calls = 0;
            Ok(())
        }

        fn release_sequence_state(&mut self, _state: Self::SequenceState) -> Result<()> {
            self.ensure_packed_topology_quiescent("release sequence state")?;
            self.released_sequence_states += 1;
            Ok(())
        }

        fn configure_kv_page_capacity(&mut self, _max_pages: usize) -> Result<()> {
            Ok(())
        }

        fn release_kv_pages(
            &mut self,
            pages: &[ferrule_common::execution::KvPageId],
        ) -> Result<()> {
            self.ensure_packed_topology_quiescent("release KV pages")?;
            self.released_kv_pages.extend_from_slice(pages);
            Ok(())
        }

        fn preempt_kv_pages(
            &mut self,
            _pages: &[ferrule_common::execution::KvPageId],
        ) -> Result<()> {
            Ok(())
        }

        fn restore_kv_pages(
            &mut self,
            _pages: &[ferrule_common::execution::KvPageId],
        ) -> Result<()> {
            Ok(())
        }

        fn prepare_multi_session_batch(
            &mut self,
            _transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
            _batch: &ferrule_common::execution::ExecutionBatch,
            _kv_reservations: &[KvReservationView],
        ) -> Result<()> {
            self.prepared_batches += 1;
            Ok(())
        }

        fn commit_multi_session_batch(
            &mut self,
            _transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
        ) -> Result<()> {
            self.committed_batches += 1;
            Ok(())
        }

        fn rollback_multi_session_batch(
            &mut self,
            _transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
        ) -> Result<()> {
            self.rolled_back_batches += 1;
            Ok(())
        }

        fn retain_provisional_prefixes(
            &mut self,
            _transaction: ExecutionTransactionId,
            sources: &[Self::SequenceState],
            branches: &mut [Self::SequenceState],
            executed_rows: &[usize],
            retained_rows: &[usize],
        ) -> Result<()> {
            if sources.len() != branches.len()
                || sources.len() != executed_rows.len()
                || sources.len() != retained_rows.len()
            {
                return Err(Error::Internal(
                    "mock speculative provisional prefix shape mismatch".into(),
                ));
            }
            for (sequence, ((source, branch), (&executed, &retained))) in sources
                .iter()
                .zip(branches.iter())
                .zip(executed_rows.iter().zip(retained_rows))
                .enumerate()
            {
                let executed_position = source.position.checked_add(executed).ok_or_else(|| {
                    Error::Internal("mock speculative executed position overflow".into())
                })?;
                let executed_fed = source.fed.len().checked_add(executed).ok_or_else(|| {
                    Error::Internal("mock speculative executed token count overflow".into())
                })?;
                if retained == 0
                    || retained > executed
                    || branch.position != executed_position
                    || branch.fed.len() != executed_fed
                {
                    return Err(Error::Internal(format!(
                        "mock speculative invalid provisional prefix for sequence {sequence}"
                    )));
                }
            }
            for ((source, branch), &retained) in
                sources.iter().zip(branches.iter_mut()).zip(retained_rows)
            {
                branch.position = source.position + retained;
                branch.fed.truncate(source.fed.len() + retained);
            }
            Ok(())
        }

        fn execute_multi_session_batch_progress(
            &mut self,
            transaction: ExecutionTransactionId,
            states: &mut [Self::SequenceState],
            batch: &ferrule_common::execution::ExecutionBatch,
        ) -> Result<MultiSessionBatchProgress> {
            if batch.intent() == ExecutionIntent::Committed {
                self.packed_committed_calls += 1;
            }
            let resumable = batch.intent() == ExecutionIntent::ProvisionalVerification
                || self.committed_resumable_armed;
            let waits = self
                .active_resumable_waits
                .entry(transaction)
                .or_insert_with(|| self.resumable_wait_scripts.pop_front().unwrap_or(0));
            if resumable && *waits > 0 {
                self.packed_verification_calls += 1;
                if let Some(predictions) = self.packed_predictions.pop_front() {
                    self.active_resumable_predictions
                        .insert(transaction, predictions);
                }
                *waits -= 1;
                return Ok(MultiSessionBatchProgress::Waiting(
                    Self::pending_resumable_progress(transaction),
                ));
            }
            if batch.intent() == ExecutionIntent::ProvisionalVerification {
                self.packed_verification_calls += 1;
                let predictions = self
                    .packed_predictions
                    .pop_front()
                    .ok_or_else(|| Error::Model("mock packed prediction queue is empty".into()))?;
                return self
                    .complete_packed_batch(states, batch, predictions)
                    .map(MultiSessionBatchProgress::Complete);
            }
            complete_mock_committed_batch(states, batch).map(MultiSessionBatchProgress::Complete)
        }

        fn resume_multi_session_batch(
            &mut self,
            transaction: ExecutionTransactionId,
            states: &mut [Self::SequenceState],
            batch: &ferrule_common::execution::ExecutionBatch,
            continuation: BatchContinuationId,
        ) -> Result<MultiSessionBatchProgress> {
            if continuation != Self::pending_resumable_progress(transaction).continuation() {
                return Err(Error::Execution(
                    "mock received an unknown batch continuation".into(),
                ));
            }
            if self.resume_errors_remaining > 0 {
                self.resume_errors_remaining -= 1;
                return Err(Error::Model("simulated resumable batch failure".into()));
            }
            let waits = self
                .active_resumable_waits
                .get_mut(&transaction)
                .ok_or_else(|| Error::Execution("mock has no active batch continuation".into()))?;
            if *waits > 0 {
                *waits -= 1;
                return Ok(MultiSessionBatchProgress::Waiting(
                    Self::pending_resumable_progress(transaction),
                ));
            }
            self.active_resumable_waits.remove(&transaction);
            let output = match self.active_resumable_predictions.remove(&transaction) {
                Some(predictions) => self.complete_packed_batch(states, batch, predictions)?,
                None => complete_mock_committed_batch(states, batch)?,
            };
            self.committed_resumable_armed = !self.resumable_wait_scripts.is_empty();
            Ok(MultiSessionBatchProgress::Complete(output))
        }

        fn cancel_multi_session_batch(
            &mut self,
            transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
            continuation: BatchContinuationId,
        ) -> BatchContinuationCancelOutcome {
            if continuation != Self::pending_resumable_progress(transaction).continuation() {
                return BatchContinuationCancelOutcome::StillActive(Error::Execution(
                    "mock received an unknown batch continuation".into(),
                ));
            }
            if self.resumable_cancel_still_active_remaining > 0 {
                self.resumable_cancel_still_active_remaining -= 1;
                return BatchContinuationCancelOutcome::StillActive(Error::Model(
                    "simulated active packed cancellation".into(),
                ));
            }
            self.active_resumable_predictions.remove(&transaction);
            self.active_resumable_waits.remove(&transaction);
            self.committed_resumable_armed = !self.resumable_wait_scripts.is_empty();
            self.cancelled_continuations.push(continuation);
            BatchContinuationCancelOutcome::Cancelled
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
                kv_binding_mode: if self.paged {
                    ferrule_common::execution::KvBindingMode::Paged
                } else {
                    ferrule_common::execution::KvBindingMode::None
                },
                logits_row_policy: if self.paged {
                    ferrule_common::execution::LogitsRowPolicy::Any
                } else {
                    ferrule_common::execution::LogitsRowPolicy::LastPerSequence
                },
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
            max_new_tokens,
            stop,
            ignore_eos: false,
        }
    }

    fn driver_from_runner(
        runner: MockTopKRunner,
    ) -> ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool> {
        ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(1),
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
                proposal_confidence_threshold: 0.2,
            },
        )
    }

    fn driver_with_outputs(
        outputs: Vec<Vec<TokenLogit>>,
    ) -> ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool> {
        driver_from_runner(MockTopKRunner::new(outputs))
    }

    fn batched_driver_from_runner(
        runner: MockTopKRunner,
    ) -> ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool> {
        ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 2,
                max_decode_batch: 2,
                max_batch_tokens: 16,
                allow_mixed_batches: true,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
    }

    fn concurrent_transaction_driver(
        runner: MockTopKRunner,
    ) -> ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool> {
        ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 1,
                max_active_sequences: 2,
                max_decode_batch: 1,
                max_batch_tokens: 1,
                allow_mixed_batches: false,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16))
    }

    fn speculative_driver_from_runner(
        runner: MockTopKRunner,
        capacity: usize,
    ) -> ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool> {
        ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(capacity),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: capacity,
                max_decode_batch: capacity,
                max_batch_tokens: 16,
                allow_mixed_batches: false,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig {
                ctx_size: 16,
                stop_at_eos: true,
                proposal_confidence_threshold: 0.2,
            },
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16))
    }

    fn ready_speculative_decode_actions(
        driver: &mut ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool>,
        request_ids: &[u64],
    ) -> Vec<DecodeAction> {
        for &id in request_ids {
            let mut submitted = request(id, &[id as u32], 4, Vec::new());
            submitted.session_id = Some(SessionId(id));
            driver.submit(submitted);
        }
        driver.prepare_step().unwrap();
        for _ in request_ids {
            let action = driver
                .scheduler
                .next_prefill_action(&mut driver.slot_pool)
                .unwrap()
                .unwrap();
            driver
                .execute_planned_action(action, &mut |_| Ok(()))
                .unwrap();
        }
        let SchedulerAction::DecodeBatch(actions) =
            driver.scheduler.next_decode_action().unwrap().unwrap()
        else {
            panic!("expected speculative decode batch");
        };
        actions
    }

    #[test]
    fn proposal_confidence_threshold_selects_a_causal_prefix() {
        let logits = [2.0, 0.0, -2.0, 4.0];
        assert_eq!(confident_proposal_prefix_length(&logits, 0.0).unwrap(), 4);
        assert_eq!(confident_proposal_prefix_length(&logits, 0.2).unwrap(), 2);
        assert_eq!(confident_proposal_prefix_length(&logits, 0.6).unwrap(), 1);
        assert!(confident_proposal_prefix_length(&logits, f32::NAN).is_err());
    }

    #[test]
    fn model_without_native_proposal_resumes_only_packed_target_execution() {
        let runner = MockTopKRunner::new(vec![top(10)]).with_resumable_wait_scripts([1, 1]);
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(1),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 1,
                max_decode_batch: 1,
                allow_mixed_batches: false,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig {
                ctx_size: 16,
                stop_at_eos: true,
                proposal_confidence_threshold: 0.2,
            },
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        let mut request = request(76, &[1], 1, Vec::new());
        request.session_id = Some(SessionId(76));
        driver.submit(request);
        let mut events = Vec::new();

        let step = |driver: &mut ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool>,
                    events: &mut Vec<ResidentTokenEvent>| {
            driver.step_with_model_expert_io(
                &mut |event| {
                    events.push(event.clone());
                    Ok(())
                },
                ExpertIoBudget::unbounded(),
            )
        };

        assert!(matches!(
            step(&mut driver, &mut events).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(ref pending) if pending.len() == 1
        ));
        assert!(events.is_empty());
        assert_eq!(driver.executor().runner().native_proposal_begin_calls, 0);

        assert!(matches!(
            step(&mut driver, &mut events).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                rows: 1,
                ..
            }
        ));
        assert!(events.is_empty());

        assert!(matches!(
            step(&mut driver, &mut events).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(ref pending) if pending.len() == 1
        ));
        assert!(events.is_empty());
        assert_eq!(driver.executor().runner().native_proposal_begin_calls, 0);

        assert!(matches!(
            step(&mut driver, &mut events).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                rows: 1,
                staged: 0,
                finished: 1,
            }
        ));

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].token, 10);
        assert_eq!(events[0].session_id, SessionId(76));
        assert_eq!(driver.executor().runner().packed_committed_calls, 2);
        assert_eq!(driver.executor().runner().prepared_batches, 2);
        assert_eq!(driver.executor().runner().committed_batches, 2);
        assert_eq!(driver.executor().runner().rolled_back_batches, 0);
        assert_eq!(driver.executor().runner().native_proposal_begin_calls, 0);
        assert_eq!(driver.stats().speculative.cycles, 0);
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        let finished = driver.drain_finished();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].position, 2);
        assert_eq!(finished[0].generated, 1);
        assert_eq!(
            finished[0].finish_reason,
            Some(SequenceFinishReason::MaxTokens)
        );
    }

    #[test]
    fn production_speculative_zero_accept_commits_correction_frontier_and_metrics() {
        let runner = MockTopKRunner::new(vec![top(10)]).with_speculative_cycle(
            NativeProposal {
                token_ids: vec![11, 12],
                confidence_logits: vec![0.75, -0.5],
            },
            vec![
                TokenLogit::new(99, 9.0),
                TokenLogit::new(98, 8.0),
                TokenLogit::new(97, 7.0),
            ],
        );
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(1),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 1,
                max_decode_batch: 1,
                allow_mixed_batches: true,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig {
                ctx_size: 16,
                stop_at_eos: true,
                proposal_confidence_threshold: 0.2,
            },
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        let mut request = request(77, &[1], 3, Vec::new());
        request.session_id = Some(SessionId(77));
        driver.submit(request);
        let mut events = Vec::new();

        let prefill = driver
            .step_with_model_expert_io(
                &mut |event| {
                    events.push(event.clone());
                    Ok(())
                },
                ExpertIoBudget::unbounded(),
            )
            .unwrap();
        assert!(matches!(
            prefill,
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));

        let decode = driver
            .step_with_model_expert_io(
                &mut |event| {
                    events.push(event.clone());
                    Ok(())
                },
                ExpertIoBudget::unbounded(),
            )
            .unwrap();
        assert!(matches!(
            decode,
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                rows: 3,
                ..
            }
        ));
        assert_eq!(driver.executor().runner().packed_verification_calls, 1);
        assert_eq!(driver.executor().runner().prepared_batches, 2);
        assert_eq!(driver.executor().runner().committed_batches, 2);
        assert_eq!(driver.executor().runner().rolled_back_batches, 0);

        assert_eq!(
            events
                .iter()
                .map(|event| (event.session_id, event.token))
                .collect::<Vec<_>>(),
            vec![(SessionId(77), 10)]
        );
        let sequence = driver
            .scheduler()
            .active_sequence(SessionId(77))
            .expect("speculative sequence should remain active");
        assert_eq!(sequence.position, 2);
        assert_eq!(sequence.generated, 1);
        assert_eq!(sequence.next_decode_token, Some(99));
        assert_eq!(
            driver
                .page_manager()
                .unwrap()
                .block_table(StateSlot::new(0))
                .unwrap()
                .committed_tokens(),
            2
        );

        let metrics = &driver.stats().speculative;
        assert_eq!(metrics.cycles, 1);
        assert_eq!(metrics.proposed_tokens, 2);
        assert_eq!(metrics.verified_rows, 3);
        assert_eq!(metrics.accepted_draft_tokens, 0);
        assert_eq!(metrics.correction_tokens, 1);
        assert_eq!(metrics.externally_committed_tokens, 1);
        assert_eq!(metrics.runtime_emitted_tokens, 1);
        assert_eq!(metrics.rolled_back_rows, 2);
        assert_eq!(metrics.rejected_tokens, 1);
        assert_eq!(metrics.accepted_prefix_histogram, vec![1]);
    }

    #[test]
    fn proposal_wait_does_not_block_later_slot_and_reports_empty_head_progress() {
        let runner = MockTopKRunner::new(vec![top(10)])
            .with_speculative_cohort(
                vec![
                    NativeProposal {
                        token_ids: vec![11, 12],
                        confidence_logits: vec![1.0, 1.0],
                    },
                    NativeProposal {
                        token_ids: vec![21, 22],
                        confidence_logits: vec![1.0, 1.0],
                    },
                ],
                vec![
                    TokenLogit::new(11, 9.0),
                    TokenLogit::new(99, 8.0),
                    TokenLogit::new(98, 7.0),
                    TokenLogit::new(21, 6.0),
                    TokenLogit::new(97, 5.0),
                    TokenLogit::new(96, 4.0),
                ],
            )
            .with_proposal_waits(vec![1, 0]);
        let mut driver = speculative_driver_from_runner(runner, 2);
        let actions = ready_speculative_decode_actions(&mut driver, &[1, 2]);

        let first = driver
            .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
            .unwrap();
        let ResidentDriverStep::WaitingForModelProgress(waiting) = first else {
            panic!("expected native proposal wait");
        };
        assert_eq!(waiting.len(), 1);
        assert_eq!(
            waiting[0].continuation(),
            BatchContinuationId::new(100).unwrap()
        );
        assert!(waiting[0].expert_loads().is_empty());
        assert_eq!(driver.executor().runner().native_proposal_begin_calls, 2);
        assert_eq!(driver.executor().runner().native_proposal_resume_calls, 0);
        assert_eq!(driver.executor().runner().active_native_proposals.len(), 1);
        assert_eq!(driver.executor().runner().packed_verification_calls, 0);

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            }
        ));
        assert_eq!(driver.executor().runner().native_proposal_begin_calls, 2);
        assert_eq!(driver.executor().runner().native_proposal_resume_calls, 1);
        assert_eq!(driver.executor().runner().packed_verification_calls, 1);
        assert!(
            driver
                .executor()
                .runner()
                .active_native_proposals
                .is_empty()
        );
    }

    #[test]
    fn repeated_proposal_wakes_resume_once_without_reproposal_and_verify_once() {
        let runner = MockTopKRunner::new(vec![top(10)])
            .with_speculative_cycle(
                NativeProposal {
                    token_ids: vec![11, 12],
                    confidence_logits: vec![1.0, 1.0],
                },
                vec![
                    TokenLogit::new(11, 9.0),
                    TokenLogit::new(99, 8.0),
                    TokenLogit::new(98, 7.0),
                ],
            )
            .with_proposal_waits(vec![3]);
        let mut driver = speculative_driver_from_runner(runner, 1);
        let actions = ready_speculative_decode_actions(&mut driver, &[1]);
        assert!(matches!(
            driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        let Some(PendingSpeculativeDriverCohort::Proposing(pending)) =
            driver.speculative_transactions.values_mut().next()
        else {
            panic!("expected pending proposal cohort");
        };
        pending.slots[0].proposal_start =
            Instant::now().checked_sub(std::time::Duration::from_millis(10));

        for expected_resumes in [1, 2] {
            assert!(matches!(
                driver
                    .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                    .unwrap(),
                ResidentDriverStep::WaitingForModelProgress(_)
            ));
            assert_eq!(driver.executor().runner().native_proposal_begin_calls, 1);
            assert_eq!(
                driver.executor().runner().native_proposal_resume_calls,
                expected_resumes
            );
            assert_eq!(driver.executor().runner().packed_verification_calls, 0);
        }

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            }
        ));
        assert_eq!(driver.executor().runner().native_proposal_begin_calls, 1);
        assert_eq!(driver.executor().runner().native_proposal_resume_calls, 3);
        assert_eq!(driver.executor().runner().packed_verification_calls, 1);
        assert_eq!(driver.stats().speculative.cycles, 1);
        assert!(driver.stats().speculative.total_proposal_time_us >= 10_000);
    }

    #[test]
    fn zero_draft_capacity_completes_empty_without_beginning_a_proposal() {
        let runner = MockTopKRunner::new(vec![top(10)]).with_speculative_cycle(
            NativeProposal {
                token_ids: vec![11, 12],
                confidence_logits: vec![1.0, 1.0],
            },
            vec![TokenLogit::new(99, 9.0)],
        );
        let mut driver = speculative_driver_from_runner(runner, 1);
        let mut submitted = request(1, &[1], 1, Vec::new());
        submitted.session_id = Some(SessionId(1));
        driver.submit(submitted);
        driver.prepare_step().unwrap();
        let prefill = driver
            .scheduler
            .next_prefill_action(&mut driver.slot_pool)
            .unwrap()
            .unwrap();
        driver
            .execute_planned_action(prefill, &mut |_| Ok(()))
            .unwrap();
        let SchedulerAction::DecodeBatch(actions) =
            driver.scheduler.next_decode_action().unwrap().unwrap()
        else {
            panic!("expected zero-draft decode batch");
        };

        assert!(matches!(
            driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                rows: 1,
                ..
            }
        ));
        assert_eq!(driver.executor().runner().native_proposal_begin_calls, 0);
        assert_eq!(driver.executor().runner().native_proposal_resume_calls, 0);
        assert_eq!(driver.executor().runner().native_proposals.len(), 1);
        assert_eq!(driver.executor().runner().packed_verification_calls, 1);
    }

    #[test]
    fn proposal_cancel_still_active_retains_cohort_and_sequence_ownership() {
        let runner = MockTopKRunner::new(vec![top(10)])
            .with_speculative_cycle(
                NativeProposal {
                    token_ids: vec![11, 12],
                    confidence_logits: vec![1.0, 1.0],
                },
                vec![
                    TokenLogit::new(11, 9.0),
                    TokenLogit::new(99, 8.0),
                    TokenLogit::new(98, 7.0),
                ],
            )
            .with_proposal_waits(vec![1])
            .with_proposal_cancel_still_active(1);
        let mut driver = speculative_driver_from_runner(runner, 1);
        let actions = ready_speculative_decode_actions(&mut driver, &[1]);
        assert!(matches!(
            driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));

        let error = driver.cancel_request(RequestId(1)).unwrap_err();
        assert!(error.to_string().contains("active continuations"));
        let Some(PendingSpeculativeDriverCohort::Proposing(pending)) =
            driver.speculative_transactions.values().next()
        else {
            panic!("StillActive must retain the proposing cohort");
        };
        assert_eq!(pending.cancellation_request, Some(RequestId(1)));
        assert_eq!(pending.source_states.len(), 1);
        assert!(matches!(
            &pending.slots[0].status,
            NativeProposalSlotStatus::Waiting(_)
        ));
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert_eq!(driver.executor().runner().active_native_proposals.len(), 1);
        assert_eq!(driver.executor().runner().native_proposal_cancel_calls, 1);

        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            CancelRequestResult::Active {
                request_id: RequestId(1),
                session_id: SessionId(1),
            }
        );
        assert!(driver.speculative_transactions.is_empty());
        assert!(
            driver
                .executor()
                .runner()
                .active_native_proposals
                .is_empty()
        );
        assert_eq!(driver.executor().runner().native_proposal_cancel_calls, 2);
        assert_eq!(
            driver.executor().runner().cancelled_native_proposals.len(),
            1
        );
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
    }

    #[test]
    fn proposal_resume_error_cancels_before_releasing_owned_state() {
        let runner = MockTopKRunner::new(vec![top(10)])
            .with_speculative_cycle(
                NativeProposal {
                    token_ids: vec![11, 12],
                    confidence_logits: vec![1.0, 1.0],
                },
                vec![
                    TokenLogit::new(11, 9.0),
                    TokenLogit::new(99, 8.0),
                    TokenLogit::new(98, 7.0),
                ],
            )
            .with_proposal_waits(vec![1])
            .with_proposal_resume_errors(1)
            .with_proposal_cancel_still_active(1);
        let mut driver = speculative_driver_from_runner(runner, 1);
        let actions = ready_speculative_decode_actions(&mut driver, &[1]);
        assert!(matches!(
            driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));

        assert!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .is_err()
        );
        assert!(matches!(
            driver.speculative_transactions.values().next(),
            Some(PendingSpeculativeDriverCohort::Proposing(_))
        ));
        assert_eq!(driver.executor().runner().native_proposal_resume_calls, 1);
        assert_eq!(driver.executor().runner().native_proposal_cancel_calls, 1);
        assert_eq!(driver.executor().runner().active_native_proposals.len(), 1);
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));

        assert!(driver.cancel_request(RequestId(1)).is_err());
        assert!(driver.speculative_transactions.is_empty());
        assert!(
            driver
                .executor()
                .runner()
                .active_native_proposals
                .is_empty()
        );
        assert_eq!(driver.executor().runner().native_proposal_cancel_calls, 2);
        assert!(driver.sequence_states.contains_key(&SessionId(1)));
        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            CancelRequestResult::Active {
                request_id: RequestId(1),
                session_id: SessionId(1),
            }
        );
    }

    #[test]
    fn production_speculative_waits_and_resumes_without_reproposal_or_early_publication() {
        let runner = MockTopKRunner::new(vec![top(10)])
            .with_speculative_cycle(
                NativeProposal {
                    token_ids: vec![11, 12],
                    confidence_logits: vec![1.0, 1.0],
                },
                vec![
                    TokenLogit::new(11, 9.0),
                    TokenLogit::new(99, 8.0),
                    TokenLogit::new(98, 7.0),
                ],
            )
            .with_resumable_wait_scripts([0, 2]);
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(1),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 1,
                max_decode_batch: 1,
                allow_mixed_batches: false,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig {
                ctx_size: 16,
                stop_at_eos: true,
                proposal_confidence_threshold: 0.2,
            },
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        let mut request = request(77, &[1], 4, Vec::new());
        request.session_id = Some(SessionId(77));
        driver.submit(request);
        let events = std::cell::RefCell::new(Vec::new());
        let mut emit = |event: &ResidentTokenEvent| {
            events.borrow_mut().push(event.clone());
            Ok(())
        };

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut emit, ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
        let first_wait = driver
            .step_with_model_expert_io(&mut emit, ExpertIoBudget::unbounded())
            .unwrap();
        assert!(matches!(
            first_wait,
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert!(!driver.speculative_transactions.is_empty());
        assert!(!driver.sequence_states.contains_key(&SessionId(77)));
        assert!(events.borrow().is_empty());
        assert_eq!(driver.stats().speculative.cycles, 0);
        assert_eq!(driver.executor().runner().native_proposal_begin_calls, 1);
        assert_eq!(driver.executor().runner().packed_verification_calls, 1);

        let second_wait = driver
            .step_with_model_expert_io(&mut emit, ExpertIoBudget::unbounded())
            .unwrap();
        assert!(matches!(
            second_wait,
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert!(events.borrow().is_empty());
        assert_eq!(driver.executor().runner().native_proposal_begin_calls, 1);
        assert_eq!(driver.executor().runner().packed_verification_calls, 1);

        let completed = driver
            .step_with_model_expert_io(&mut emit, ExpertIoBudget::unbounded())
            .unwrap();
        assert!(matches!(
            completed,
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                rows: 3,
                ..
            }
        ));
        assert!(driver.speculative_transactions.is_empty());
        assert!(driver.sequence_states.contains_key(&SessionId(77)));
        assert_eq!(
            events
                .borrow()
                .iter()
                .map(|event| event.token)
                .collect::<Vec<_>>(),
            vec![10, 11]
        );
        assert_eq!(driver.stats().speculative.cycles, 1);
        assert_eq!(
            driver.stats().speculative.externally_committed_tokens,
            events.borrow().len()
        );
        assert_eq!(
            driver.stats().speculative.runtime_emitted_tokens,
            events.borrow().len()
        );
        assert_eq!(
            driver
                .page_manager()
                .unwrap()
                .block_table(StateSlot::new(0))
                .unwrap()
                .committed_tokens(),
            3
        );
        assert_eq!(driver.executor().runner().native_proposal_begin_calls, 1);
        assert_eq!(driver.executor().runner().packed_verification_calls, 1);
    }

    #[test]
    fn c2_suspended_packed_verification_preserves_sequence_and_kv_ownership() {
        let runner = MockTopKRunner::new(vec![top(10)])
            .with_speculative_cycle(
                NativeProposal {
                    token_ids: vec![11, 12],
                    confidence_logits: vec![1.0, 1.0],
                },
                vec![
                    TokenLogit::new(11, 9.0),
                    TokenLogit::new(99, 8.0),
                    TokenLogit::new(98, 7.0),
                ],
            )
            .with_speculative_cycle(
                NativeProposal {
                    token_ids: vec![21, 22],
                    confidence_logits: vec![1.0, 1.0],
                },
                vec![
                    TokenLogit::new(21, 6.0),
                    TokenLogit::new(97, 5.0),
                    TokenLogit::new(96, 4.0),
                ],
            )
            .with_resumable_wait_scripts([0, 0, 2, 0])
            .with_packed_topology_guard();
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 2,
                max_decode_batch: 1,
                max_batch_tokens: 1,
                allow_mixed_batches: false,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig {
                ctx_size: 16,
                stop_at_eos: true,
                proposal_confidence_threshold: 0.2,
            },
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        for id in [1, 2] {
            let mut submitted = request(id, &[id as u32], 4, Vec::new());
            submitted.session_id = Some(SessionId(id));
            driver.submit(submitted);
        }

        driver.prepare_step().unwrap();
        for _ in 0..2 {
            let prefill = driver
                .scheduler
                .next_prefill_action(&mut driver.slot_pool)
                .unwrap()
                .expect("both c2 sessions should require prefill");
            assert!(matches!(
                driver
                    .execute_planned_action(prefill, &mut |_| Ok(()))
                    .unwrap(),
                ResidentDriverStep::Executed {
                    action_kind: ResidentActionKind::Prefill,
                    ..
                }
            ));
        }
        let page_tables_before = [SessionId(1), SessionId(2)].map(|session_id| {
            let slot = driver.page_slots[&session_id];
            driver
                .page_manager()
                .unwrap()
                .block_table(slot)
                .unwrap()
                .pages()
                .to_vec()
        });
        let allocated_pages_before = driver.page_manager().unwrap().allocated_pages();

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(ref pending) if pending.len() == 1
        ));
        assert_eq!(driver.speculative_transactions.len(), 1);
        assert_eq!(driver.session_owner.len(), 1);
        let owned_session = *driver.session_owner.keys().next().unwrap();
        let runnable_session = if owned_session == SessionId(1) {
            SessionId(2)
        } else {
            SessionId(1)
        };
        assert!(!driver.sequence_states.contains_key(&owned_session));
        assert!(driver.sequence_states.contains_key(&runnable_session));
        assert_eq!(driver.executor().runner().sequence_state_fork_calls, 1);
        assert_eq!(
            driver
                .executor()
                .runner()
                .topology_mutation_attempts_while_packed,
            0
        );
        assert_eq!(driver.executor().runner().released_sequence_states, 0);
        assert!(driver.executor().runner().released_kv_pages.is_empty());
        assert_eq!(
            [SessionId(1), SessionId(2)].map(|session_id| {
                let slot = driver.page_slots[&session_id];
                driver
                    .page_manager()
                    .unwrap()
                    .block_table(slot)
                    .unwrap()
                    .pages()
                    .to_vec()
            }),
            page_tables_before
        );
        assert_eq!(
            driver.page_manager().unwrap().allocated_pages(),
            allocated_pages_before
        );

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert_eq!(driver.speculative_transactions.len(), 1);
        assert_eq!(driver.executor().runner().sequence_state_fork_calls, 1);
        assert_eq!(
            driver
                .executor()
                .runner()
                .topology_mutation_attempts_while_packed,
            0
        );
        assert_eq!(driver.executor().runner().released_sequence_states, 0);
        assert!(driver.executor().runner().released_kv_pages.is_empty());

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            }
        ));
        assert!(driver.speculative_transactions.is_empty());
        assert!(driver.session_owner.is_empty());
        assert!(driver.sequence_states.contains_key(&owned_session));
        assert!(driver.sequence_states.contains_key(&runnable_session));

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            }
        ));
        assert_eq!(driver.executor().runner().sequence_state_fork_calls, 2);
        assert_eq!(
            driver
                .executor()
                .runner()
                .topology_mutation_attempts_while_packed,
            0
        );
        assert_eq!(driver.executor().runner().packed_verification_calls, 2);
    }

    #[test]
    fn cancelling_pending_speculative_cohort_restores_surviving_decode_order() {
        let runner = MockTopKRunner::new(vec![top(10)])
            .with_speculative_cohort(
                vec![
                    NativeProposal {
                        token_ids: vec![11, 12],
                        confidence_logits: vec![1.0, 1.0],
                    },
                    NativeProposal {
                        token_ids: vec![21, 22],
                        confidence_logits: vec![1.0, 1.0],
                    },
                ],
                vec![
                    TokenLogit::new(11, 9.0),
                    TokenLogit::new(99, 8.0),
                    TokenLogit::new(98, 7.0),
                    TokenLogit::new(21, 6.0),
                    TokenLogit::new(22, 5.0),
                    TokenLogit::new(23, 4.0),
                ],
            )
            .with_resumable_wait_scripts([0, 0, 1]);
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 2,
                max_decode_batch: 2,
                max_batch_tokens: 16,
                allow_mixed_batches: false,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig {
                ctx_size: 16,
                stop_at_eos: true,
                proposal_confidence_threshold: 0.2,
            },
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        for id in [1, 2] {
            let mut request = request(id, &[id as u32], 4, Vec::new());
            request.session_id = Some(SessionId(id));
            driver.submit(request);
        }
        driver.prepare_step().unwrap();
        for _ in 0..2 {
            let action = driver
                .scheduler
                .next_prefill_action(&mut driver.slot_pool)
                .unwrap()
                .unwrap();
            driver
                .execute_planned_action(action, &mut |_| Ok(()))
                .unwrap();
        }
        let SchedulerAction::DecodeBatch(actions) =
            driver.scheduler.next_decode_action().unwrap().unwrap()
        else {
            panic!("expected decode batch");
        };
        assert!(matches!(
            driver
                .execute_speculative_decode_batch(actions.clone(), &mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));

        let cancelled = driver.cancel_request(RequestId(1)).unwrap();
        assert_eq!(
            cancelled,
            CancelRequestResult::Active {
                request_id: RequestId(1),
                session_id: SessionId(1),
            }
        );
        assert!(driver.speculative_transactions.is_empty());
        assert_eq!(driver.executor().runner().cancelled_continuations.len(), 1);
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert!(driver.sequence_states.contains_key(&SessionId(2)));
        let SchedulerAction::DecodeBatch(restored) =
            driver.scheduler.next_decode_action().unwrap().unwrap()
        else {
            panic!("expected surviving decode action");
        };
        assert_eq!(restored, vec![actions[1]]);
    }

    #[test]
    fn production_speculative_batches_two_ragged_sessions_in_one_provisional_execution() {
        let runner = MockTopKRunner::new(vec![top(10)]).with_speculative_cohort(
            vec![
                NativeProposal {
                    token_ids: vec![11, 12],
                    confidence_logits: vec![1.0, 1.0],
                },
                NativeProposal {
                    token_ids: vec![21, 22],
                    confidence_logits: vec![1.0, -2.0],
                },
            ],
            vec![
                TokenLogit::new(11, 9.0),
                TokenLogit::new(99, 8.0),
                TokenLogit::new(98, 7.0),
                TokenLogit::new(21, 6.0),
                TokenLogit::new(22, 5.0),
            ],
        );
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 2,
                max_decode_batch: 2,
                max_batch_tokens: 16,
                allow_mixed_batches: false,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig {
                ctx_size: 16,
                stop_at_eos: true,
                proposal_confidence_threshold: 0.2,
            },
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));

        for id in [1, 2] {
            let mut request = request(id, &[id as u32], 4, Vec::new());
            request.session_id = Some(SessionId(id));
            driver.submit(request);
        }
        driver.prepare_step().unwrap();
        let mut events = Vec::new();
        for _ in 0..2 {
            let action = driver
                .scheduler
                .next_prefill_action(&mut driver.slot_pool)
                .unwrap()
                .unwrap();
            driver
                .execute_planned_action(action, &mut |event| {
                    events.push(event.clone());
                    Ok(())
                })
                .unwrap();
        }
        let SchedulerAction::DecodeBatch(actions) = driver
            .scheduler
            .next_decode_action()
            .unwrap()
            .expect("both sessions should be decode-ready")
        else {
            panic!("expected a decode batch");
        };
        assert_eq!(actions.len(), 2);
        let expected_events = vec![
            (actions[0].session_id, actions[0].token_id),
            (actions[0].session_id, 11),
            (actions[1].session_id, actions[1].token_id),
            (actions[1].session_id, 21),
        ];

        let decode = driver
            .execute_speculative_decode_batch(actions, &mut |event| {
                events.push(event.clone());
                Ok(())
            })
            .unwrap();
        assert_eq!(
            decode,
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                rows: 5,
                staged: 2,
                finished: 0,
            }
        );
        assert_eq!(driver.executor().runner().packed_verification_calls, 1);
        assert_eq!(driver.executor().runner().prepared_batches, 3);
        assert_eq!(driver.executor().runner().committed_batches, 3);
        assert_eq!(driver.executor().runner().rolled_back_batches, 0);
        assert_eq!(
            events
                .iter()
                .map(|event| (event.session_id, event.token))
                .collect::<Vec<_>>(),
            expected_events
        );

        let metrics = &driver.stats().speculative;
        assert_eq!(metrics.cycles, 2);
        assert_eq!(metrics.proposed_tokens, 3);
        assert_eq!(metrics.verified_rows, 5);
        assert_eq!(metrics.accepted_draft_tokens, 2);
        assert_eq!(metrics.correction_tokens, 1);
        assert_eq!(metrics.externally_committed_tokens, 4);
        assert_eq!(metrics.runtime_emitted_tokens, 4);
        assert_eq!(metrics.rolled_back_rows, 1);
        assert_eq!(metrics.rejected_tokens, 1);
        assert_eq!(metrics.accepted_prefix_histogram, vec![0, 2]);
        assert!(metrics.total_verify_time_us <= metrics.total_transaction_time_us);
        assert!(metrics.total_transaction_time_us <= metrics.total_cycle_time_us);
        for slot in [0, 1] {
            assert_eq!(
                driver
                    .page_manager()
                    .unwrap()
                    .block_table(StateSlot::new(slot))
                    .unwrap()
                    .committed_tokens(),
                3
            );
        }
    }

    #[test]
    fn ordinary_resumable_prefill_waits_resumes_and_commits_once() {
        let runner = MockTopKRunner::new(Vec::new()).with_committed_resumable_batch(
            2,
            vec![TokenLogit::new(7, 1.0), TokenLogit::new(8, 2.0)],
        );
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(1),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 1,
                max_decode_batch: 1,
                max_batch_tokens: 8,
                allow_mixed_batches: false,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        let mut submitted = request(1, &[1, 2], 2, Vec::new());
        submitted.session_id = Some(SessionId(1));
        driver.submit(submitted);
        let mut events = Vec::new();

        assert!(matches!(
            driver
                .step_with_model_expert_io(
                    &mut |event| {
                        events.push(event.clone());
                        Ok(())
                    },
                    ExpertIoBudget::unbounded()
                )
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert!(!driver.resident_transactions.is_empty());
        assert!(driver.speculative_transactions.is_empty());
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert!(driver.executor().has_transactions());
        assert_eq!(
            driver
                .page_manager()
                .unwrap()
                .block_table(StateSlot::new(0))
                .unwrap()
                .committed_tokens(),
            0
        );
        assert_eq!(driver.executor().runner().prepared_batches, 1);
        assert_eq!(driver.executor().runner().committed_batches, 0);
        assert_eq!(driver.executor().runner().rolled_back_batches, 0);
        assert_eq!(driver.executor().runner().packed_verification_calls, 1);
        assert!(events.is_empty());

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert_eq!(driver.executor().runner().prepared_batches, 1);
        assert_eq!(driver.executor().runner().committed_batches, 0);
        assert_eq!(driver.executor().runner().packed_verification_calls, 1);
        assert_eq!(
            driver
                .page_manager()
                .unwrap()
                .block_table(StateSlot::new(0))
                .unwrap()
                .committed_tokens(),
            0
        );

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                rows: 2,
                ..
            }
        ));
        assert!(driver.resident_transactions.is_empty());
        assert_eq!(driver.sequence_states[&SessionId(1)].position, 2);
        assert_eq!(
            driver
                .scheduler()
                .active_sequence(SessionId(1))
                .unwrap()
                .position,
            2
        );
        assert_eq!(
            driver
                .page_manager()
                .unwrap()
                .block_table(StateSlot::new(0))
                .unwrap()
                .committed_tokens(),
            2
        );
        assert_eq!(driver.executor().runner().prepared_batches, 1);
        assert_eq!(driver.executor().runner().committed_batches, 1);
        assert_eq!(driver.executor().runner().rolled_back_batches, 0);
        assert_eq!(driver.executor().runner().packed_verification_calls, 1);
    }

    #[test]
    fn ordinary_resumable_resume_error_retains_pending_ownership() {
        let runner = MockTopKRunner::new(Vec::new())
            .with_committed_resumable_batch(1, vec![TokenLogit::new(7, 1.0)])
            .with_resume_errors(1);
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(1),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 1,
                max_decode_batch: 1,
                allow_mixed_batches: false,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        let mut submitted = request(1, &[1], 2, Vec::new());
        submitted.session_id = Some(SessionId(1));
        driver.submit(submitted);

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .is_err()
        );
        assert!(!driver.resident_transactions.is_empty());
        assert!(driver.executor().has_transactions());
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert_eq!(driver.executor().runner().committed_batches, 0);
        assert_eq!(driver.executor().runner().rolled_back_batches, 0);
        assert!(!driver.executor().is_poisoned());

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                rows: 1,
                ..
            }
        ));
        assert!(driver.resident_transactions.is_empty());
        assert_eq!(driver.executor().runner().prepared_batches, 1);
        assert_eq!(driver.executor().runner().committed_batches, 1);
        assert_eq!(driver.executor().runner().rolled_back_batches, 0);
    }

    #[test]
    fn cancelling_unrelated_request_does_not_quiesce_pending_transaction() {
        let runner = MockTopKRunner::new(vec![top(9)]).with_resumable_wait_scripts([1, 0]);
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 1,
                max_active_sequences: 2,
                max_decode_batch: 1,
                max_batch_tokens: 1,
                allow_mixed_batches: false,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        for id in [1, 2] {
            let mut submitted = request(id, &[id as u32], 2, Vec::new());
            submitted.session_id = Some(SessionId(id));
            driver.submit(submitted);
        }

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                rows: 1,
                ..
            }
        ));
        assert_eq!(driver.resident_transactions.len(), 1);
        assert!(driver.executor().has_transactions());
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert!(driver.sequence_states.contains_key(&SessionId(2)));

        let cancelled = driver.cancel_request(RequestId(2)).unwrap();
        assert_eq!(
            cancelled,
            CancelRequestResult::Active {
                request_id: RequestId(2),
                session_id: SessionId(2),
            }
        );
        assert_eq!(driver.resident_transactions.len(), 1);
        assert!(driver.executor().has_transactions());
        assert!(
            driver
                .executor()
                .runner()
                .cancelled_continuations
                .is_empty()
        );
        assert_eq!(driver.executor().runner().committed_batches, 1);
        assert_eq!(driver.executor().runner().rolled_back_batches, 0);
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert!(driver.sequence_states.contains_key(&SessionId(2)));
        assert!(driver.pending_sequence_cleanups.contains_key(&SessionId(2)));
        assert!(driver.scheduler().active_sequence(SessionId(1)).is_none());
        assert!(driver.scheduler().active_sequence(SessionId(2)).is_none());
        assert!(driver.session_owner.contains_key(&SessionId(1)));
        assert!(!driver.session_owner.contains_key(&SessionId(2)));
        assert_eq!(driver.slot_pool().active_count(), 1);
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 2);
        assert!(!driver.executor().is_poisoned());

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                rows: 1,
                ..
            }
        ));
        assert!(driver.resident_transactions.is_empty());
        assert!(!driver.executor().has_transactions());
        assert_eq!(driver.executor().runner().committed_batches, 2);
        assert!(driver.pending_sequence_cleanups.contains_key(&SessionId(2)));

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            }
        ));
        assert!(!driver.pending_sequence_cleanups.contains_key(&SessionId(2)));
        assert!(!driver.sequence_states.contains_key(&SessionId(2)));
        assert!(!driver.page_slots.contains_key(&SessionId(2)));
        assert_eq!(
            driver.page_manager().unwrap().active_sequences(),
            usize::from(driver.page_slots.contains_key(&SessionId(1)))
        );
    }

    #[test]
    fn concurrent_transactions_wait_and_complete_in_reverse_order() {
        let runner = MockTopKRunner::new(vec![top(9)]).with_resumable_wait_scripts([2, 1]);
        let mut driver = concurrent_transaction_driver(runner);
        for id in [1, 2] {
            let mut submitted = request(id, &[id as u32], 2, Vec::new());
            submitted.session_id = Some(SessionId(id));
            driver.submit(submitted);
        }

        let ResidentDriverStep::WaitingForModelProgress(progress) = driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap()
        else {
            panic!("both transactions must suspend");
        };
        assert_eq!(progress.len(), 2);
        let transaction_a = driver.session_owner[&SessionId(1)];
        let transaction_b = driver.session_owner[&SessionId(2)];
        assert_ne!(transaction_a, transaction_b);
        assert!(
            progress
                .iter()
                .any(|wait| wait.transaction() == transaction_a)
        );
        assert!(
            progress
                .iter()
                .any(|wait| wait.transaction() == transaction_b)
        );
        assert_eq!(driver.resident_transactions.len(), 2);

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
        assert!(driver.resident_transactions.contains_key(&transaction_a));
        assert!(!driver.resident_transactions.contains_key(&transaction_b));
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert!(driver.sequence_states.contains_key(&SessionId(2)));
        assert_eq!(driver.executor().runner().committed_batches, 1);

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
        assert!(driver.resident_transactions.is_empty());
        assert!(driver.session_owner.is_empty());
        assert!(driver.sequence_states.contains_key(&SessionId(1)));
        assert_eq!(driver.executor().runner().committed_batches, 2);
        assert_eq!(driver.executor().runner().rolled_back_batches, 0);
    }

    #[test]
    fn c4_sibling_cancellation_defers_topology_cleanup_until_packed_owner_completes() {
        let runner = MockTopKRunner::new(vec![top(9)])
            .with_committed_resumable_batch(1, vec![TokenLogit::new(7, 1.0)])
            .with_resumable_wait_scripts([0])
            .with_packed_topology_guard();
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(4),
            ResidentSchedulerConfig {
                prefill_chunk_size: 1,
                max_active_sequences: 4,
                max_decode_batch: 1,
                max_batch_tokens: 1,
                allow_mixed_batches: false,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        for id in 1..=4 {
            let mut submitted = request(id, &[id as u32], 2, Vec::new());
            submitted.session_id = Some(SessionId(id));
            driver.submit(submitted);
        }

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
        assert_eq!(driver.resident_transactions.len(), 1);
        let packed_owner = *driver.session_owner.keys().next().unwrap();
        let siblings = (1..=4)
            .map(SessionId)
            .filter(|session_id| *session_id != packed_owner)
            .collect::<Vec<_>>();
        let sibling_pages = siblings
            .iter()
            .flat_map(|session_id| {
                let slot = driver.page_slots[session_id];
                driver
                    .page_manager()
                    .unwrap()
                    .block_table(slot)
                    .unwrap()
                    .pages()
                    .to_vec()
            })
            .collect::<Vec<_>>();

        for session_id in &siblings {
            let request_id = RequestId(session_id.0);
            assert_eq!(
                driver.cancel_request(request_id).unwrap(),
                CancelRequestResult::Active {
                    request_id,
                    session_id: *session_id,
                }
            );
        }
        assert_eq!(driver.pending_sequence_cleanups.len(), 3);
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 4);
        for session_id in &siblings {
            assert!(driver.page_slots.contains_key(session_id));
            assert!(driver.sequence_states.contains_key(session_id));
        }
        assert_eq!(
            driver
                .executor()
                .runner()
                .topology_mutation_attempts_while_packed,
            0
        );
        assert_eq!(driver.executor().runner().released_sequence_states, 0);
        assert!(driver.executor().runner().released_kv_pages.is_empty());

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
        assert!(driver.resident_transactions.is_empty());
        assert_eq!(driver.pending_sequence_cleanups.len(), 3);

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            }
        ));
        assert!(driver.pending_sequence_cleanups.is_empty());
        assert!(driver.pending_kv_retirements.is_empty());
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 1);
        for session_id in &siblings {
            assert!(!driver.page_slots.contains_key(session_id));
            assert!(!driver.sequence_states.contains_key(session_id));
        }
        assert_eq!(driver.executor().runner().released_sequence_states, 3);
        assert_eq!(driver.executor().runner().released_kv_pages, sibling_pages);
        assert_eq!(
            driver
                .executor()
                .runner()
                .topology_mutation_attempts_while_packed,
            0
        );
    }

    #[test]
    fn target_only_concurrent_completion_defers_cleanup_until_all_transactions_quiesce() {
        let runner = MockTopKRunner::new(Vec::new())
            .with_committed_resumable_batch(2, vec![TokenLogit::new(7, 1.0)])
            .with_committed_resumable_batch(1, vec![TokenLogit::new(8, 2.0)])
            .with_packed_topology_guard();
        let mut driver = concurrent_transaction_driver(runner);
        for id in [1, 2] {
            let mut submitted = request(id, &[id as u32], 0, Vec::new());
            submitted.session_id = Some(SessionId(id));
            driver.submit(submitted);
        }

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(ref pending) if pending.len() == 2
        ));
        assert_eq!(driver.resident_transactions.len(), 2);

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                finished: 1,
                ..
            }
        ));
        assert_eq!(driver.resident_transactions.len(), 1);
        assert_eq!(driver.pending_kv_retirements.len(), 1);
        assert_eq!(driver.pending_sequence_cleanups.len(), 1);
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 2);
        assert_eq!(driver.executor().runner().released_sequence_states, 0);
        assert!(driver.executor().runner().released_kv_pages.is_empty());
        assert_eq!(
            driver
                .executor()
                .runner()
                .topology_mutation_attempts_while_packed,
            0
        );

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                finished: 1,
                ..
            }
        ));
        assert!(driver.resident_transactions.is_empty());
        assert_eq!(driver.pending_sequence_cleanups.len(), 1);
        assert_eq!(driver.executor().runner().released_sequence_states, 1);

        assert_eq!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Idle
        );
        assert!(driver.pending_kv_retirements.is_empty());
        assert!(driver.pending_sequence_cleanups.is_empty());
        assert!(driver.sequence_states.is_empty());
        assert!(driver.page_slots.is_empty());
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 0);
        assert_eq!(driver.executor().runner().released_sequence_states, 2);
        assert_eq!(
            driver
                .executor()
                .runner()
                .topology_mutation_attempts_while_packed,
            0
        );
    }

    #[test]
    fn cancelling_one_waiting_transaction_does_not_disturb_the_other() {
        let runner = MockTopKRunner::new(Vec::new())
            .with_committed_resumable_batch(2, vec![TokenLogit::new(7, 1.0)])
            .with_committed_resumable_batch(2, vec![TokenLogit::new(8, 2.0)])
            .with_packed_topology_guard();
        let mut driver = concurrent_transaction_driver(runner);
        for id in [1, 2] {
            let mut submitted = request(id, &[id as u32], 2, Vec::new());
            submitted.session_id = Some(SessionId(id));
            driver.submit(submitted);
        }
        assert!(matches!(
            driver.step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded()).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(progress) if progress.len() == 2
        ));
        let transaction_b = driver.session_owner[&SessionId(2)];

        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            CancelRequestResult::Active {
                request_id: RequestId(1),
                session_id: SessionId(1),
            }
        );
        assert_eq!(driver.resident_transactions.len(), 1);
        assert!(driver.resident_transactions.contains_key(&transaction_b));
        assert_eq!(driver.executor().runner().cancelled_continuations.len(), 1);
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert_eq!(driver.executor().runner().committed_batches, 0);
        assert!(!driver.session_owner.contains_key(&SessionId(1)));
        assert_eq!(driver.session_owner[&SessionId(2)], transaction_b);
        assert_eq!(driver.pending_kv_retirements.len(), 1);
        assert!(driver.pending_sequence_cleanups.contains_key(&SessionId(1)));
        assert!(driver.sequence_states.contains_key(&SessionId(1)));
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 2);
        assert_eq!(
            driver
                .executor()
                .runner()
                .topology_mutation_attempts_while_packed,
            0
        );
        assert_eq!(driver.executor().runner().released_sequence_states, 0);
        assert!(driver.executor().runner().released_kv_pages.is_empty());

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
        assert!(driver.resident_transactions.is_empty());
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert_eq!(driver.executor().runner().committed_batches, 1);
        assert!(driver.pending_sequence_cleanups.contains_key(&SessionId(1)));

        let _ = driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap();
        assert!(driver.pending_kv_retirements.is_empty());
        assert!(!driver.pending_sequence_cleanups.contains_key(&SessionId(1)));
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert!(!driver.page_slots.contains_key(&SessionId(1)));
        assert_eq!(
            driver
                .executor()
                .runner()
                .topology_mutation_attempts_while_packed,
            0
        );
    }

    #[test]
    fn still_active_cancellation_keeps_ownership_while_another_transaction_commits() {
        let runner = MockTopKRunner::new(vec![top(9)])
            .with_resumable_wait_scripts([1, 1])
            .with_resumable_cancel_still_active(1);
        let mut driver = concurrent_transaction_driver(runner);
        for id in [1, 2] {
            let mut submitted = request(id, &[id as u32], 2, Vec::new());
            submitted.session_id = Some(SessionId(id));
            driver.submit(submitted);
        }
        driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap();
        let transaction_a = driver.session_owner[&SessionId(1)];
        let transaction_b = driver.session_owner[&SessionId(2)];

        let error = driver.cancel_request(RequestId(1)).unwrap_err();
        assert!(error.to_string().contains("did not confirm quiescence"));
        assert_eq!(
            driver.resident_transactions[&transaction_a].cancelling,
            Some(RequestId(1))
        );
        assert!(driver.resident_transactions.contains_key(&transaction_b));
        assert_eq!(driver.executor().runner().rolled_back_batches, 0);

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
        assert!(driver.resident_transactions.contains_key(&transaction_a));
        assert!(!driver.resident_transactions.contains_key(&transaction_b));
        assert_eq!(driver.executor().runner().committed_batches, 1);
        assert_eq!(driver.executor().runner().rolled_back_batches, 0);

        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Cancel,
                ..
            }
        ));
        assert!(!driver.resident_transactions.contains_key(&transaction_a));
        assert_eq!(driver.executor().runner().committed_batches, 1);
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert_eq!(driver.executor().runner().cancelled_continuations.len(), 1);
        assert!(driver.scheduler().active_sequence(SessionId(1)).is_none());
        assert!(driver.scheduler().active_sequence(SessionId(2)).is_some());
    }

    #[test]
    fn ordinary_pending_batch_blocks_session_topology_mutation_and_runner_extraction() {
        let runner = MockTopKRunner::new(Vec::new())
            .with_committed_resumable_batch(1, vec![TokenLogit::new(7, 1.0)]);
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 1,
                max_decode_batch: 1,
                allow_mixed_batches: false,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        driver.retain_session(SessionId(1)).unwrap();
        let mut submitted = request(1, &[1], 2, Vec::new());
        submitted.session_id = Some(SessionId(1));
        driver.submit(submitted);
        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));

        let release = driver.release_session(SessionId(1)).unwrap_err();
        assert!(
            release
                .to_string()
                .contains("execution transactions are live")
        );
        assert_eq!(driver.retained_session_position(SessionId(1)), Some(0));
        let preempt = driver.preempt_session(SessionId(1)).unwrap_err();
        assert!(
            preempt
                .to_string()
                .contains("execution transactions are live")
        );
        let mut fork_target = request(2, &[2], 1, Vec::new());
        fork_target.session_id = Some(SessionId(2));
        let fork = driver
            .fork_session_exact(SessionId(1), fork_target, 0)
            .unwrap_err();
        assert!(fork.to_string().contains("execution transactions are live"));
        assert_eq!(driver.slot_pool().active_count(), 1);
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 1);

        let Err(failure) = driver.try_into_runner() else {
            panic!("runner extraction must reject a suspended resident batch");
        };
        let (error, mut driver) = *failure;
        assert!(error.to_string().contains("live execution transactions"));
        assert!(!driver.resident_transactions.is_empty());
        driver.cancel_request(RequestId(1)).unwrap();
        assert!(driver.resident_transactions.is_empty());
    }

    #[test]
    fn driver_runs_request_to_max_tokens_and_frees_kv() {
        let mut driver = driver_with_outputs(vec![top(b'a' as u32), top(b'b' as u32)]);
        driver.submit(request(1, &[1, 2], 2, Vec::new()));
        let mut events = Vec::new();
        let stats = driver
            .drive_ready_test_work(|event| {
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
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        assert_eq!(driver.retained_session_position(session_id), Some(2));
        let _ = driver.drain_finished();

        let mut second = request(11, &[2], 1, Vec::new());
        second.session_id = Some(session_id);
        driver.submit(second);
        let mut events = Vec::new();
        driver
            .drive_ready_test_work(|event| {
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
            .drive_ready_test_work(|event| {
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
    fn driver_finishes_on_eos_without_appending_or_emitting_candidate() {
        let runner = MockTopKRunner::new(vec![top(2)]).with_eos(2);
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(1),
            ResidentSchedulerConfig::default(),
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        );
        driver.submit(request(3, &[1], 4, Vec::new()));
        let mut events = Vec::new();
        driver
            .drive_ready_test_work(|event| {
                events.push(event.clone());
                Ok(())
            })
            .unwrap();

        assert!(events.is_empty());
        assert!(driver.executor().runner().fed.is_empty());
        let finished = driver.drain_finished();
        assert_eq!(finished[0].position, 1);
        assert_eq!(finished[0].finish_reason, Some(SequenceFinishReason::Eos));
        assert_eq!(finished[0].tokens, vec![1]);
    }

    #[test]
    fn mixed_requests_isolate_ignore_eos_policy() {
        let eos = 2;
        let mut driver = ResidentTopKDriver::with_configs(
            MockTopKRunner::new(vec![top(eos)]).with_eos(eos),
            FixedSequenceSlotPool::new(2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 4,
                max_active_sequences: 2,
                max_decode_batch: 2,
                max_batch_tokens: 8,
                allow_mixed_batches: true,
                ..Default::default()
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
            .drive_ready_test_work(|event| {
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
        driver.drive_ready_test_work(|_| Ok(())).unwrap();

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
        driver.drive_ready_test_work(|_| Ok(())).unwrap();

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
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        let first = driver.drain_finished();
        assert_eq!(first[0].position, 2);

        driver.submit(request(8, &[2], 1, Vec::new()));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        let second = driver.drain_finished();
        assert_eq!(second[0].prompt_tokens_for_range(0..1).unwrap(), &[2]);
        assert_eq!(second[0].position, 2);
    }

    #[test]
    fn into_runner_allows_clean_driver_rebuild_after_warmup() {
        let mut driver = driver_with_outputs(vec![top(b'w' as u32), top(b'm' as u32)]);
        driver.submit(request(9, &[1], 1, Vec::new()));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        let warmup = driver.drain_finished();
        assert_eq!(warmup[0].position, 2);

        let runner = driver
            .try_into_runner()
            .map_err(|failure| failure.0)
            .unwrap();
        assert!(!runner.expert_io_resource_control_installed);
        let mut driver = driver_from_runner(runner);
        assert!(
            driver
                .executor()
                .runner()
                .expert_io_resource_control_installed
        );
        driver.submit(request(10, &[2], 1, Vec::new()));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        let measured = driver.drain_finished();

        assert_eq!(measured[0].prompt_tokens_for_range(0..1).unwrap(), &[2]);
        assert_eq!(measured[0].position, 2);
    }

    #[test]
    fn driver_moves_partially_executed_sequence_to_error_state() {
        let runner = MockTopKRunner::new(vec![top(b'a' as u32)]).failing_next_mutation();
        let mut driver = driver_from_runner(runner);
        driver.submit(request(11, &[1, 2], 1, Vec::new()));

        let error = driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap_err();
        assert!(format!("{error}").contains("simulated failure"));
        assert!(!driver.executor().is_poisoned());
        assert!(driver.resident_transactions.is_empty());
        assert!(driver.session_owner.is_empty());
        assert_eq!(driver.scheduler().active_len(), 0);
        assert_eq!(driver.scheduler().failed_len(), 1);
        assert_eq!(driver.slot_pool().active_count(), 0);

        let failed = driver.drain_failed();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].status, SequenceStatus::Error);
        assert_eq!(failed[0].position, 0, "runtime metadata was not committed");
        assert_eq!(failed[0].kv_handle, None);

        driver.executor_mut().runner_mut().fail_next_mutation = false;
        driver.submit(request(12, &[3], 1, Vec::new()));
        assert!(matches!(
            driver
                .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
                .unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
    }

    #[test]
    fn batch_execution_writes_back_every_sequence_state_on_success() {
        let mut driver = batched_driver_from_runner(MockTopKRunner::new(vec![top(b'a' as u32)]));
        driver.submit(request(50, &[1], 2, Vec::new()));
        driver.submit(request(51, &[2], 2, Vec::new()));

        let step = driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap();
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

        let error = driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap_err();
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

        let error = driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap_err();
        assert!(format!("{error}").contains("neutral u32 ABI"));
        assert!(!driver.executor().is_poisoned());
        assert_eq!(driver.scheduler().active_len(), 0);
        assert_eq!(driver.scheduler().failed_len(), 1);
        assert_eq!(driver.slot_pool().active_count(), 0);
        assert_eq!(driver.executor().runner().mutation_calls, 0);
    }

    #[test]
    fn callback_failure_retains_committed_event_without_poisoning_execution() {
        let mut driver = driver_with_outputs(vec![top(b'a' as u32), top(b'b' as u32)]);
        driver.submit(request(14, &[1], 3, Vec::new()));

        let error = driver
            .drive_ready_test_work(|_| Err(Error::Internal("simulated callback failure".into())))
            .unwrap_err();
        assert!(format!("{error}").contains("callback failure"));
        assert!(!driver.executor().is_poisoned());
        assert!(!driver.executor().has_transactions());
        assert_eq!(driver.scheduler().active_len(), 1);
        assert_eq!(driver.scheduler().failed_len(), 0);
        assert_eq!(driver.slot_pool().active_count(), 1);
        assert_eq!(driver.committed_token_outbox.len(), 1);

        let mut delivered = Vec::new();
        driver
            .flush_committed_token_outbox(&mut |event| {
                delivered.push(event.clone());
                Ok(())
            })
            .unwrap();
        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].token, b'a' as u32);
        assert!(driver.committed_token_outbox.is_empty());
        assert_eq!(driver.stats().emitted_tokens, 1);

        driver.cancel_request(RequestId(14)).unwrap();
        driver.drain_cancelled();
        assert!(driver.try_into_runner().is_ok());
    }

    #[test]
    fn driver_rejects_exceeding_executor_capabilities_before_consuming_work() {
        // The mock executor supports max_sequences=4. Set max_active_sequences=5
        // to trigger a validation failure before any work is consumed.
        let mut driver = ResidentTopKDriver::with_configs(
            MockTopKRunner::new(Vec::new()),
            FixedSequenceSlotPool::new(2),
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

        let error = driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap_err();
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
        driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap();

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
            .drive_ready_test_work(|event| {
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
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
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

        let first = driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap();
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
            .drive_ready_test_work(|event| {
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

    fn fork_driver() -> ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool> {
        ResidentTopKDriver::with_configs(
            MockTopKRunner::new(vec![top(b'a' as u32), top(b'b' as u32)]),
            FixedSequenceSlotPool::new(2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 2,
                max_decode_batch: 2,
                max_batch_tokens: 8,
                allow_mixed_batches: true,
                ..Default::default()
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
        driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap();
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

        driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap();
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
        driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap();
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
        driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap();
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
        let step = driver
            .step_with_model_expert_io(&mut |_| Ok(()), ExpertIoBudget::unbounded())
            .unwrap();
        assert_eq!(step, ResidentDriverStep::Blocked);
        assert_eq!(driver.scheduler().waiting_len(), 1);
    }
}
