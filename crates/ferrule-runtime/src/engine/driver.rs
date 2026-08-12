use std::collections::{HashMap, HashSet, VecDeque};
use std::num::NonZeroU32;
use std::time::Instant;

use crate::{CleanupStep, Error, Result};

use ferrule_common::execution::{
    ExecutionOutput, ExecutionTransactionId, KvBindingMode, KvPageId, KvReservationView, StateSlot,
};
use ferrule_common::io_protocol::{
    BackendId, CancellationReason, ContinuationId, DependencySetEpoch, DeviceId, ModelInstanceId,
    RequestGeneration, RetirementReason, WaiterId,
};
use ferrule_model::{
    MaterializationPlacement, MaterializationResolver, MultiSessionBatchProgress,
    MultiSessionRunner, NativeProposal, NativeProposalProgress, NativeProposalSource,
    PendingModelProgress, PhysicalMaterializationTopology, ResidentModelRunner,
    TransactionEndIntent, TransactionEndProgress,
};
use tracing;

use crate::cache::{
    KvPageManager, KvPrefixSnapshot, KvReservation, KvReservationCommit, KvRetirement,
    PreemptedKvState, PrefixCacheNamespace, PrefixLookupLimit, PreparedKvCommit,
    PreparedKvSnapshotFork, RadixPrefixCache, RemovedPrefixEntry,
};
use crate::io::{
    FailedContinuation, FairQueueConfig, LoadRegistry, OutputTokenId, ResumeDisposition,
    RuntimeMaterializationProvider, RuntimeMaterializationResolver,
    RuntimeMaterializationResolverStats, SharedMaterializationProvider, TransactionCustodyOutcome,
    UnavailableMaterializationProvider,
};
use crate::scheduling::resident::{
    PreparedWaitingAdmission, SuspendedSequenceSchedule, greedy_candidate,
};
use crate::scheduling::{
    CancelRequestResult, DecodeAction, ExecutionPhase, ExecutionPhaseSet, GenerateRequest,
    PhysicalResourceBroker, PhysicalResourceClaim, PhysicalResourceGrant, PhysicalResourceLimit,
    PrefillChunkAction, RequestId, ResidentScheduler, ResidentSchedulerConfig, ResourceKind,
    ScheduledBatch, SchedulerAction, SequenceFinishReason, SequenceSlotPool, SequenceState,
    SessionId,
};
use crate::speculation::{
    PendingSpeculativeVerificationCohort, PreparedSpeculativeCohort,
    QuiescedSpeculativeAbortProgress, QuiescedSpeculativePublishProgress, SpeculativeCohortFailure,
    SpeculativeCohortProgress, SpeculativeCohortTransaction, SpeculativeCycleResult,
    SpeculativeMetrics, SpeculativeVerificationItem, TargetFrontier,
    abort_quiesced_speculative_transaction, begin_prepared_speculative_verification,
    prepare_speculative_verification_transaction, publish_quiesced_speculative_cohort,
    resume_resumable_speculative_verification_cohort,
};

use super::NativeMultiSessionExecutor;
use super::observability::{
    ResidentDriverObservability, ResidentPrefixCacheStats, ResidentTopKDriverStats,
};

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

fn prefix_cache_error(operation: &str, error: impl std::fmt::Display) -> Error {
    Error::Invariant {
        message: format!("{operation}: {error}"),
    }
}

fn confident_proposal_prefix_length(logits: &[f32], threshold: f32) -> Result<usize> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(Error::InvalidRequest {
            message: format!(
                "proposal confidence threshold must be finite and within [0, 1], got {threshold}"
            ),
        });
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
    /// Whether a model-provided proposal capability may replace target-only decode.
    pub enable_native_proposals: bool,
    /// Static per-position confidence threshold used until the calibrated,
    /// batch-wide hardware scheduler is available. Zero disables truncation.
    pub proposal_confidence_threshold: f32,
}

impl Default for ResidentTopKDriverConfig {
    fn default() -> Self {
        Self {
            ctx_size: 4096,
            stop_at_eos: true,
            enable_native_proposals: true,
            proposal_confidence_threshold: 0.2,
        }
    }
}

/// Runtime-owned limits that are not reported by the physical materialization provider.
/// KV capacity starts at zero and is replaced by the exact page-manager capacity
/// when `try_with_page_manager` installs that owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentRuntimeResourceLimits {
    pub arena_slots: u64,
    pub kv_pages: u64,
    pub continuations: u64,
    pub waiters: u64,
    pub load_operations: u64,
    pub ready_cohorts: u64,
}

impl ResidentRuntimeResourceLimits {
    pub fn for_scheduler(config: ResidentSchedulerConfig) -> Result<Self> {
        let active = u64::try_from(config.max_active_sequences.max(1)).map_err(|_| {
            Error::InvalidRequest {
                message: "resident scheduler concurrency exceeds runtime resource range".into(),
            }
        })?;
        Ok(Self {
            arena_slots: active,
            kv_pages: 0,
            continuations: active,
            waiters: active,
            // The physical stage capacities remain the effective default bound.
            load_operations: u64::MAX,
            ready_cohorts: active,
        })
    }

    fn validate(self) -> Result<Self> {
        for (name, value) in [
            ("arena slots", self.arena_slots),
            ("continuations", self.continuations),
            ("waiters", self.waiters),
            ("load operations", self.load_operations),
            ("ready cohorts", self.ready_cohorts),
        ] {
            if value == 0 {
                return Err(Error::InvalidRequest {
                    message: format!("resident runtime {name} limit must be non-zero"),
                });
            }
        }
        Ok(self)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentCancelProgress {
    Pending,
    Complete(CancelRequestResult),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentShutdownProgress {
    Pending,
    Complete(ResidentDriverShutdownReport),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentDriverShutdownReport {
    pub registry: crate::io::ShutdownReport,
    pub executor_transactions: usize,
    pub kv_page_grants: usize,
    pub pending_kv_retirements: usize,
}

/// Synchronous resident driver over scheduler + KV + native multi-session executor.
///
/// This is the end-to-end resident workload loop in runtime. It remains concrete
/// and synchronous: no async frontend, no trait-object framework, and no concrete
/// model ownership. The driver connects request admission, scheduled execution,
/// output policy, event delivery, and resource/session lifecycle through typed
/// runtime values.
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
    Retiring(PendingKvRetirement),
}

enum PendingKvRetirement {
    BackendRelease(KvRetirement),
    LogicalConfirmation(KvRetirement),
}

enum PendingSequenceCleanup<S> {
    Deferred,
    Suspended {
        kv_state: Option<PreemptedKvState>,
        retirement: Option<PendingKvRetirement>,
        model_state: Option<S>,
    },
    Owned {
        retirement: Option<PendingKvRetirement>,
        model_state: Option<S>,
    },
}

struct ResidentPrefixPayload<S> {
    model_state: S,
    snapshot: KvPrefixSnapshot,
}

struct PreparedPrefixAdmission<S> {
    model_state: S,
    pages: PreparedKvSnapshotFork,
    matched_tokens: usize,
}

struct PendingPrefixCleanup<S> {
    snapshot: Option<KvPrefixSnapshot>,
    retirement: Option<PendingKvRetirement>,
    model_state: Option<S>,
}

impl<S> PendingPrefixCleanup<S> {
    fn from_payload(payload: ResidentPrefixPayload<S>) -> Self {
        Self {
            snapshot: Some(payload.snapshot),
            retirement: None,
            model_state: Some(payload.model_state),
        }
    }

    fn model_only(model_state: S) -> Self {
        Self {
            snapshot: None,
            retirement: None,
            model_state: Some(model_state),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PendingRequestCancellation {
    session_id: SessionId,
    transaction: Option<ExecutionTransactionId>,
    ready: bool,
    result: Option<CancelRequestResult>,
}

struct RegisteredModelContinuation {
    transaction: ExecutionTransactionId,
    dependencies: ferrule_common::DependencySet,
}

#[derive(Debug)]
struct PendingMaterializationFailure {
    failed: FailedContinuation,
    transaction: Option<ExecutionTransactionId>,
}

struct ResidentRuntimeParts<R: MultiSessionRunner> {
    executor: NativeMultiSessionExecutor<R>,
    completion_hub: ferrule_common::CompletionHub,
    registry: LoadRegistry<Box<dyn RuntimeMaterializationProvider>>,
    resolver: RuntimeMaterializationResolver,
    warmup_requests: Vec<ferrule_model::MaterializationRequest>,
    uninstalled_resolver: Option<RuntimeMaterializationResolver>,
}

fn physical_materialization_resources(
    topology: PhysicalMaterializationTopology,
    runtime: ResidentRuntimeResourceLimits,
) -> Result<PhysicalResourceBroker> {
    let limits = topology.stage_limits().validate()?;
    let runtime = runtime.validate()?;
    let capacity = limits.capacity;
    let reserve = limits.execution_reserve;
    let lease_capacity = topology
        .residency_lease_slots_per_continuation()
        .checked_mul(runtime.continuations)
        .ok_or_else(|| Error::InvalidRequest {
            message: "residency lease capacity overflow".into(),
        })?;
    let physical_load_capacity = runtime.load_operations;
    let physical_load_reserve = 0;
    let resources = [
        PhysicalResourceLimit::new(
            ResourceKind::ReadSlot,
            capacity.read_slots,
            reserve.read_slots,
        ),
        PhysicalResourceLimit::new(
            ResourceKind::PinnedHostBytes,
            capacity.pinned_host_bytes,
            reserve.pinned_host_bytes,
        ),
        PhysicalResourceLimit::new(
            ResourceKind::StorageReadBytes,
            capacity.storage_read_bytes,
            reserve.storage_read_bytes,
        ),
        PhysicalResourceLimit::new(
            ResourceKind::UploadSlot,
            capacity.upload_slots,
            reserve.upload_slots,
        ),
        PhysicalResourceLimit::new(
            ResourceKind::UploadBytes,
            capacity.h2d_bytes,
            reserve.h2d_bytes,
        ),
        PhysicalResourceLimit::new(
            ResourceKind::InstallSlot,
            capacity.install_slots,
            reserve.install_slots,
        ),
        PhysicalResourceLimit::new(
            ResourceKind::DeviceInstallBytes,
            capacity.device_install_bytes,
            reserve.device_install_bytes,
        ),
        PhysicalResourceLimit::new(
            ResourceKind::ResidentBytes,
            topology.resident_capacity_bytes(),
            0,
        ),
        PhysicalResourceLimit::new(ResourceKind::ResidencyLease, lease_capacity, 0),
        PhysicalResourceLimit::new(ResourceKind::Arena, runtime.arena_slots, 0),
        PhysicalResourceLimit::new(ResourceKind::KvPage, runtime.kv_pages, 0),
        PhysicalResourceLimit::new(ResourceKind::Continuation, runtime.continuations, 0),
        PhysicalResourceLimit::new(ResourceKind::Waiter, runtime.waiters, 0),
        PhysicalResourceLimit::new(
            ResourceKind::LoadOperation,
            physical_load_capacity,
            physical_load_reserve,
        ),
        PhysicalResourceLimit::new(ResourceKind::ReadyCohort, runtime.ready_cohorts, 0),
    ];
    PhysicalResourceBroker::new(resources).map_err(Error::from)
}

struct PendingResidentBatch<S> {
    transaction: ExecutionTransactionId,
    action: SchedulerAction,
    scheduled: ScheduledBatch,
    kv: Option<PendingResidentKv>,
    states: Vec<S>,
    schedules: Vec<SuspendedSequenceSchedule>,
    phase: ResidentTransactionPhase,
}

enum ResidentTransactionPhase {
    Executing(Option<PendingModelProgress>),
    Publishing {
        output: ExecutionOutput,
        started_ns: u64,
        cancellation_request: Option<RequestId>,
    },
    Aborting {
        request: Option<RequestId>,
        custody: TransactionCustodyOutcome,
        continuation: Option<ContinuationId>,
        failure: Option<(Error, &'static str)>,
    },
    BackendCommittedPendingPublish {
        output: ExecutionOutput,
        custody: Option<TransactionCustodyOutcome>,
        cancellation_request: Option<RequestId>,
    },
    BackendAbortedPendingCleanup {
        request: Option<RequestId>,
        custody: Option<TransactionCustodyOutcome>,
        continuation: Option<ContinuationId>,
        failure: Option<(Error, &'static str)>,
        decode_requeued: bool,
    },
}

impl ResidentTransactionPhase {
    fn pending_progress(&self) -> Option<&PendingModelProgress> {
        match self {
            Self::Executing(pending) => pending.as_ref(),
            Self::Publishing { .. }
            | Self::Aborting { .. }
            | Self::BackendCommittedPendingPublish { .. }
            | Self::BackendAbortedPendingCleanup { .. } => None,
        }
    }

    const fn is_ending(&self) -> bool {
        !matches!(self, Self::Executing(_))
    }
}

enum PendingSpeculativeDriverCohort<S> {
    Proposing(Box<PendingNativeProposalCohort<S>>),
    Verifying(Box<PendingSpeculativeVerificationDriverCohort<S>>),
    Ending(Box<PendingSpeculativeEndingDriverCohort<S>>),
}

impl<S> PendingSpeculativeDriverCohort<S> {
    fn actions(&self) -> &[DecodeAction] {
        match self {
            Self::Proposing(pending) => &pending.actions,
            Self::Verifying(pending) => &pending.actions,
            Self::Ending(pending) => &pending.actions,
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
            Self::Ending(_) => {}
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
    abort_cause: Option<Error>,
    backend_aborted: bool,
    custody: Option<TransactionCustodyOutcome>,
    decode_requeued: bool,
}

enum SpeculativeEnding<S> {
    Publish(PreparedSpeculativeCohort<S>),
    Abort {
        transaction: SpeculativeCohortTransaction<S>,
        failure: Option<(Error, &'static str)>,
    },
    BackendCommittedPendingPublish {
        progress: QuiescedSpeculativePublishProgress<S>,
        retirements: VecDeque<PendingKvRetirement>,
        custody: Option<TransactionCustodyOutcome>,
    },
    BackendAbortedPendingCleanup {
        progress: QuiescedSpeculativeAbortProgress<S>,
        retirements: VecDeque<PendingKvRetirement>,
        custody: Option<TransactionCustodyOutcome>,
        failure: Option<(Error, &'static str)>,
        decode_requeued: bool,
    },
}

struct PendingSpeculativeEndingDriverCohort<S> {
    transaction: ExecutionTransactionId,
    cohort_start: Instant,
    started_ns: u64,
    actions: Vec<DecodeAction>,
    prepared: Vec<PreparedSpeculativeAction>,
    source_states: Vec<S>,
    schedules: Vec<SuspendedSequenceSchedule>,
    ending: SpeculativeEnding<S>,
    request_id: Option<RequestId>,
    continuation: Option<ContinuationId>,
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
    kv_page_grants: HashMap<KvPageId, PhysicalResourceGrant>,
    page_slots: HashMap<SessionId, StateSlot>,
    prefix_cache: RadixPrefixCache<ResidentPrefixPayload<R::SequenceState>>,
    prefix_cache_sessions: HashSet<SessionId>,
    suspended_sequences: HashMap<SessionId, SuspendedDriverSequence<R::SequenceState>>,
    next_page_slot: u32,
    config: ResidentTopKDriverConfig,
    observability: ResidentDriverObservability,
    next_transaction_id: u64,
    resident_transactions: HashMap<ExecutionTransactionId, PendingResidentBatch<R::SequenceState>>,
    speculative_transactions:
        HashMap<ExecutionTransactionId, PendingSpeculativeDriverCohort<R::SequenceState>>,
    session_owner: HashMap<SessionId, ExecutionTransactionId>,
    completion_hub: ferrule_common::CompletionHub,
    load_registry: LoadRegistry<Box<dyn RuntimeMaterializationProvider>>,
    materialization_resolver: RuntimeMaterializationResolver,
    uninstalled_materialization_resolver: Option<RuntimeMaterializationResolver>,
    continuations: HashMap<ContinuationId, RegisteredModelContinuation>,
    transaction_continuations: HashMap<ExecutionTransactionId, HashSet<ContinuationId>>,
    ready_transactions: VecDeque<ExecutionTransactionId>,
    queued_transactions: HashSet<ExecutionTransactionId>,
    ready_continuations: HashSet<ContinuationId>,
    pending_materialization_failures: VecDeque<PendingMaterializationFailure>,
    warmup_requests: Vec<ferrule_model::MaterializationRequest>,
    warmup_started_ns: Option<u64>,
    warmup_last_report_ns: u64,

    pending_registry_detaches: HashMap<ContinuationId, CancellationReason>,
    next_dependency_epoch: u64,
    runtime_clock: Instant,
    runtime_tick: u64,
    next_output_token_id: u64,
    pending_kv_retirements: VecDeque<PendingKvRetirement>,
    pending_resident_kv_aborts: VecDeque<PendingResidentKv>,
    pending_sequence_cleanups: HashMap<SessionId, PendingSequenceCleanup<R::SequenceState>>,
    pending_request_cancellations: HashMap<RequestId, PendingRequestCancellation>,
    pending_prefix_cleanups: VecDeque<PendingPrefixCleanup<R::SequenceState>>,
    committed_token_outbox: VecDeque<ResidentTokenEvent>,
    shutting_down: bool,
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
        Self::try_new(runner, slot_pool)
            .expect("resident model reported an invalid runtime resource topology")
    }

    pub fn try_new(runner: R, slot_pool: C) -> Result<Self>
    where
        R: ResidentModelRunner,
    {
        Self::try_with_configs(
            runner,
            slot_pool,
            ResidentSchedulerConfig::default(),
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
        Self::try_with_configs(runner, slot_pool, scheduler_config, top_k, driver_config)
            .expect("resident model reported an invalid runtime resource topology")
    }

    pub fn try_with_configs(
        runner: R,
        slot_pool: C,
        scheduler_config: ResidentSchedulerConfig,
        top_k: NonZeroU32,
        driver_config: ResidentTopKDriverConfig,
    ) -> Result<Self>
    where
        R: ResidentModelRunner,
    {
        let runtime_limits = ResidentRuntimeResourceLimits::for_scheduler(scheduler_config)?;
        Self::try_with_configs_and_runtime_limits(
            runner,
            slot_pool,
            scheduler_config,
            top_k,
            driver_config,
            runtime_limits,
        )
    }

    pub fn try_with_configs_and_runtime_limits(
        runner: R,
        slot_pool: C,
        scheduler_config: ResidentSchedulerConfig,
        top_k: NonZeroU32,
        driver_config: ResidentTopKDriverConfig,
        runtime_limits: ResidentRuntimeResourceLimits,
    ) -> Result<Self>
    where
        R: ResidentModelRunner,
    {
        Self::with_parts(
            ResidentScheduler::new(scheduler_config),
            slot_pool,
            Self::runtime_parts(runner, runtime_limits)?,
            top_k,
            driver_config,
        )
    }

    fn runtime_parts(
        mut runner: R,
        runtime_limits: ResidentRuntimeResourceLimits,
    ) -> Result<ResidentRuntimeParts<R>>
    where
        R: ResidentModelRunner,
    {
        let requirements = runner.expert_residency_requirements();
        if runner.materialization_resolver_installed() {
            return Err(Error::InvalidRequest {
                message:
                    "runner already owns a materialization resolver outside the runtime registry"
                        .into(),
            });
        }
        if let Some(requirements) = requirements.as_ref()
            && !runner.expert_residency_control_installed()
        {
            let control = crate::expert_residency::ExpertResidencyController::with_requirements(
                requirements.clone(),
            )?;
            runner.install_expert_residency_control(Box::new(control))?;
        }

        let warmup_requests = runner.take_warmup_requests()?;

        // Materialization capability is independent of model topology. Expert
        // residency requirements add slot-policy validation but do not gate dense
        // parameter or mutable-state streaming.
        let physical = runner.take_materialization_provider();
        if requirements.is_some() && physical.is_none() {
            return Err(Error::InvalidRequest { message:
                "runner reports expert residency requirements but provides no physical materialization provider"
                    .into(),
             });
        }
        let completion_hub = runner.completion_hub();
        let has_physical = physical.is_some();
        let (placement, registry, installed_physical) = match physical {
            Some(physical) => {
                let provider = SharedMaterializationProvider::new(physical);
                let placement = provider.placement();
                if let Some(requirements) = requirements.as_ref()
                    && placement.model().get() != requirements.model_instance
                {
                    return Err(Error::InvalidRequest {
                        message: format!(
                            "physical materialization provider model namespace {} does not match runner expert-residency namespace {}",
                            placement.model().get(),
                            requirements.model_instance
                        ),
                    });
                }
                let topology = provider.resource_topology()?;
                let physical_limits = topology.stage_limits().validate()?;
                let fairness = FairQueueConfig::for_production(physical_limits)?;
                let resources = physical_materialization_resources(topology, runtime_limits)?;
                let registry = LoadRegistry::new(
                    Box::new(provider.clone()) as Box<dyn RuntimeMaterializationProvider>,
                    resources,
                    fairness,
                )?;
                (placement, registry, Some(provider))
            }
            None => {
                let placement = MaterializationPlacement::new(
                    ModelInstanceId::new(1),
                    BackendId::new(1),
                    DeviceId::new(0),
                )?;
                let resources = physical_materialization_resources(
                    PhysicalMaterializationTopology::new(
                        ferrule_common::materialization_io::MaterializationResourceLimits::default(
                        ),
                        0,
                        0,
                    )?,
                    runtime_limits,
                )?;
                let registry = LoadRegistry::new(
                    Box::new(UnavailableMaterializationProvider)
                        as Box<dyn RuntimeMaterializationProvider>,
                    resources,
                    FairQueueConfig::default(),
                )?;
                (placement, registry, None)
            }
        };

        // Build the model-facing resolver only after the registry owns provider command
        // and hard-resource authority, then install it back into any runner that
        // exposes physical materialization capability.
        let resolver = RuntimeMaterializationResolver::new(placement, installed_physical);
        let uninstalled_resolver = if has_physical {
            runner.install_materialization_resolver(Box::new(resolver.clone()))?;
            None
        } else {
            Some(resolver.clone())
        };

        Ok(ResidentRuntimeParts {
            executor: NativeMultiSessionExecutor::new(runner),
            completion_hub,
            registry,
            resolver,
            warmup_requests,
            uninstalled_resolver,
        })
    }

    fn with_parts(
        scheduler: ResidentScheduler,
        slot_pool: C,
        runtime: ResidentRuntimeParts<R>,
        top_k: NonZeroU32,
        config: ResidentTopKDriverConfig,
    ) -> Result<Self> {
        let ResidentRuntimeParts {
            executor,
            completion_hub,
            registry,
            resolver,
            warmup_requests,
            uninstalled_resolver,
        } = runtime;
        let prefix_cache = RadixPrefixCache::new(scheduler.config().prefix_cache_capacity_pages);
        let driver = Self {
            scheduler,
            slot_pool,
            executor,
            sequence_states: HashMap::new(),
            retained_sessions: HashMap::new(),
            top_k,
            page_manager: None,
            kv_page_grants: HashMap::new(),
            page_slots: HashMap::new(),
            prefix_cache,
            prefix_cache_sessions: HashSet::new(),
            suspended_sequences: HashMap::new(),
            next_page_slot: 0,
            config,
            observability: ResidentDriverObservability::default(),
            next_transaction_id: 1,
            resident_transactions: HashMap::new(),
            speculative_transactions: HashMap::new(),
            session_owner: HashMap::new(),
            completion_hub,
            load_registry: registry,
            materialization_resolver: resolver,
            uninstalled_materialization_resolver: uninstalled_resolver,
            continuations: HashMap::new(),
            transaction_continuations: HashMap::new(),
            ready_transactions: VecDeque::new(),
            queued_transactions: HashSet::new(),
            ready_continuations: HashSet::new(),
            pending_materialization_failures: VecDeque::new(),
            warmup_requests,
            warmup_started_ns: None,
            warmup_last_report_ns: 0,
            pending_registry_detaches: HashMap::new(),
            next_dependency_epoch: 1,
            runtime_clock: Instant::now(),
            runtime_tick: 0,
            next_output_token_id: 1,
            pending_kv_retirements: VecDeque::new(),
            pending_resident_kv_aborts: VecDeque::new(),
            pending_sequence_cleanups: HashMap::new(),
            pending_request_cancellations: HashMap::new(),
            pending_prefix_cleanups: VecDeque::new(),
            committed_token_outbox: VecDeque::new(),
            shutting_down: false,
        };
        Ok(driver)
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
        Ok(self.executor.runner().encode(text)?)
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

    pub(crate) fn completion_hub(&self) -> ferrule_common::CompletionHub {
        self.completion_hub.clone()
    }

    pub(crate) fn take_completion_reactors(&mut self) -> Vec<ferrule_model::ModelCompletionReactor>
    where
        R: ResidentModelRunner,
    {
        self.executor.runner_mut().take_completion_reactors()
    }

    pub fn warmup_pending(&self) -> bool {
        !self.warmup_requests.is_empty()
            || self
                .load_registry
                .prefetch_active(crate::io::PrefetchOwner::ModelWarmup)
            || self
                .load_registry
                .prefetch_failed(crate::io::PrefetchOwner::ModelWarmup)
    }

    pub(crate) fn start_background_work(&mut self) -> Result<()> {
        const INITIAL_BACKGROUND_TRANSITIONS: usize = 64;

        self.begin_warmup()?;
        let now_ns = self.runtime_now_ns();
        self.load_registry
            .drive(now_ns, INITIAL_BACKGROUND_TRANSITIONS)?;
        self.check_warmup()?;
        self.update_hard_resource_observability();
        Ok(())
    }

    fn begin_warmup(&mut self) -> Result<()> {
        let owner = crate::io::PrefetchOwner::ModelWarmup;
        if self.load_registry.prefetch_active(owner) || self.warmup_requests.is_empty() {
            return Ok(());
        }
        let requests = std::mem::take(&mut self.warmup_requests);
        match self.prefetch(
            owner,
            crate::scheduling::ResourceDemand::ModelWarmup,
            requests.clone(),
        ) {
            Ok(_) => {
                self.warmup_started_ns = Some(self.runtime_now_ns());
                self.warmup_last_report_ns = 0;
                Ok(())
            }
            Err(error) => {
                self.warmup_requests = requests;
                Err(error)
            }
        }
    }

    fn check_warmup(&mut self) -> Result<()> {
        let owner = crate::io::PrefetchOwner::ModelWarmup;
        let Some(reason) = self.load_registry.take_prefetch_failure(owner) else {
            if !self.load_registry.prefetch_active(owner) && self.warmup_requests.is_empty() {
                self.warmup_started_ns = None;
            }
            return Ok(());
        };
        match reason {
            RetirementReason::Failed(source) => Err(Error::WarmupMaterialization { source }),
            RetirementReason::Cancelled(reason) => {
                Err(Error::WarmupMaterializationCancelled { reason })
            }
            RetirementReason::Stale(reason) => Err(Error::WarmupMaterializationStale { reason }),
            RetirementReason::ResidentOwnershipTransferred => Ok(()),
            RetirementReason::Drained
            | RetirementReason::OrphanCompletion
            | RetirementReason::OwnerShutdown => Err(Error::WarmupMaterializationNotPublished),
        }
    }

    pub fn has_background_work(&self) -> bool {
        self.warmup_pending()
    }

    fn has_pending_non_prefix_async_work(&self) -> bool {
        !self.resident_transactions.is_empty()
            || !self.speculative_transactions.is_empty()
            || !self.continuations.is_empty()
            || !self.pending_materialization_failures.is_empty()
            || !self.pending_registry_detaches.is_empty()
            || self.load_registry.active_prefetches() != 0
            || self.load_registry.active_operations() != 0
            || self.load_registry.has_pending_owner_work()
            || !self.pending_kv_retirements.is_empty()
            || !self.pending_resident_kv_aborts.is_empty()
            || !self.pending_sequence_cleanups.is_empty()
            || !self.pending_request_cancellations.is_empty()
            || self.scheduler.failed_slot_ownership() != 0
    }

    pub fn has_pending_async_work(&self) -> bool {
        // Prefix cleanup is owner-thread synchronous work. It has no completion
        // producer and is retried by step/shutdown/runner extraction, so reporting
        // it here would let an event-driven owner sleep forever.
        self.has_pending_non_prefix_async_work()
    }

    pub const fn is_shutting_down(&self) -> bool {
        self.shutting_down
    }

    /// Replaces the materialization provider before any execution suspends.
    pub fn try_with_materialization_provider<B>(
        mut self,
        provider: B,
        resources: crate::scheduling::PhysicalResourceBroker,
        fairness: FairQueueConfig,
    ) -> Result<Self>
    where
        B: RuntimeMaterializationProvider + 'static,
    {
        self.ensure_no_suspended_execution("replace the materialization provider")?;
        if let Some(resolver) = self.uninstalled_materialization_resolver.take() {
            self.executor
                .runner_mut()
                .install_materialization_resolver(Box::new(resolver))?;
        }
        self.load_registry = LoadRegistry::new(
            Box::new(provider) as Box<dyn RuntimeMaterializationProvider>,
            resources,
            fairness,
        )?;
        Ok(self)
    }

    #[cfg(test)]
    pub fn with_materialization_provider<B>(self, provider: B) -> Result<Self>
    where
        B: RuntimeMaterializationProvider + 'static,
    {
        self.try_with_materialization_provider(
            provider,
            crate::scheduling::PhysicalResourceBroker::testing_default(),
            FairQueueConfig::default(),
        )
    }

    pub fn load_registry(&self) -> &LoadRegistry<Box<dyn RuntimeMaterializationProvider>> {
        &self.load_registry
    }

    pub fn prefetch(
        &mut self,
        owner: crate::io::PrefetchOwner,
        demand: crate::scheduling::ResourceDemand,
        requests: impl IntoIterator<Item = ferrule_model::MaterializationRequest>,
    ) -> Result<crate::io::PrefetchReport> {
        if !demand.is_prefetch() {
            return Err(Error::InvalidRequest {
                message: format!(
                    "prefetch requires a non-blocking resource demand, got {demand:?}"
                ),
            });
        }
        if self.shutting_down {
            return Err(Error::InvalidRequest {
                message: "cannot submit prefetch after driver shutdown begins".into(),
            });
        }
        let mut prepared = Vec::new();
        for request in requests {
            let key = match self.materialization_resolver.prepare_prefetch(request) {
                Ok(key) => key,
                Err(error) => {
                    let cleanup = self
                        .load_registry
                        .discard_preparations(prepared.iter().copied());
                    return Err(Error::with_cleanup(
                        "prefetch preparation",
                        error,
                        cleanup.map_err(Error::from),
                    ));
                }
            };
            prepared.push(key);
        }
        let loads = match prepared
            .iter()
            .copied()
            .map(|key| self.load_registry.prepare_prefetch_request(key, demand))
            .collect::<std::result::Result<Vec<_>, _>>()
        {
            Ok(loads) => loads,
            Err(error) => {
                let cleanup = self
                    .load_registry
                    .discard_preparations(prepared.iter().copied());
                return Err(Error::with_cleanup(
                    "prefetch request validation",
                    error,
                    cleanup.map_err(Error::from),
                ));
            }
        };
        match self
            .load_registry
            .prefetch(owner, loads, self.runtime_now_ns())
        {
            Ok(report) => {
                if !report.created.is_empty() {
                    self.completion_hub.notify();
                }
                Ok(report)
            }
            Err(error) => {
                let cleanup = self.load_registry.discard_preparations(prepared);
                Err(Error::with_cleanup(
                    "prefetch admission",
                    error,
                    cleanup.map_err(Error::from),
                ))
            }
        }
    }

    pub fn cancel_prefetch(&mut self, owner: crate::io::PrefetchOwner) -> Result<()> {
        self.load_registry
            .cancel_prefetch(owner, self.runtime_now_ns())
            .map_err(Error::from)
    }

    pub fn materialization_resolver_stats(&self) -> RuntimeMaterializationResolverStats {
        self.materialization_resolver.stats()
    }

    /// Installs the page manager, panicking on an invalid test topology.
    pub fn with_page_manager(mut self, page_manager: KvPageManager) -> Self {
        let explicit_kv_limit = self
            .load_registry
            .resources()
            .snapshots()
            .find(|snapshot| snapshot.kind == ResourceKind::KvPage)
            .expect("hard resource catalog contains KV pages")
            .capacity;
        if explicit_kv_limit == 0 {
            return self
                .try_with_page_manager(page_manager)
                .expect("test page-manager topology must be valid");
        }
        self.executor
            .configure_kv_page_capacity(page_manager.max_pages())
            .expect("test backend accepts page-manager capacity");
        self.page_manager = Some(page_manager);
        self
    }

    /// Install the authoritative runtime page manager and configure a backend
    /// physical pool with the same bounded page capacity.
    pub fn try_with_page_manager(mut self, page_manager: KvPageManager) -> Result<Self> {
        if self.page_manager.is_some() {
            return Err(Error::InvalidRequest {
                message: "the authoritative KV page manager is already installed".into(),
            });
        }
        let max_pages = page_manager.max_pages();
        if max_pages == 0 {
            return Err(Error::InvalidRequest {
                message: "a physical KV backend requires a bounded non-zero page capacity".into(),
            });
        }
        self.ensure_no_suspended_execution("install a page manager")?;
        self.executor.configure_kv_page_capacity(max_pages)?;
        self.load_registry.resources_mut().reconfigure_limit(
            ResourceKind::KvPage,
            u64::try_from(max_pages).map_err(|_| Error::InvalidRequest {
                message: "KV page capacity exceeds runtime resource range".into(),
            })?,
            0,
        )?;
        self.page_manager = Some(page_manager);
        Ok(self)
    }

    pub fn page_manager(&self) -> Option<&KvPageManager> {
        self.page_manager.as_ref()
    }

    fn prefix_cache_namespace(&self) -> Option<PrefixCacheNamespace> {
        if self.prefix_cache.capacity() == 0 {
            return None;
        }
        let manager = self.page_manager.as_ref()?;
        let placement = self.materialization_resolver.placement();
        let plan = self.executor.runner().prefix_cache_plan_identity();
        if plan == 0 {
            return None;
        }
        Some(PrefixCacheNamespace::for_placement(
            placement.model().get(),
            placement.backend().get(),
            placement.device().get(),
            plan,
            manager.owner_identity(),
        ))
    }

    fn queue_removed_prefix(
        &mut self,
        removed: RemovedPrefixEntry<ResidentPrefixPayload<R::SequenceState>>,
    ) {
        self.pending_prefix_cleanups
            .push_back(PendingPrefixCleanup::from_payload(removed.into_payload()));
    }

    fn queue_prefix_payload_cleanup(&mut self, payload: ResidentPrefixPayload<R::SequenceState>) {
        self.pending_prefix_cleanups
            .push_back(PendingPrefixCleanup::from_payload(payload));
    }

    fn progress_prefix_cleanup(
        &mut self,
        mut cleanup: PendingPrefixCleanup<R::SequenceState>,
    ) -> std::result::Result<(), (Error, PendingPrefixCleanup<R::SequenceState>)> {
        if self.has_live_transactions() {
            return Err((
                Error::InvalidRequest {
                    message: "cannot release a cached prefix while packed transactions are live"
                        .into(),
                },
                cleanup,
            ));
        }
        if let Some(snapshot) = cleanup.snapshot {
            let retirement = match self.page_manager.as_mut() {
                Some(manager) => match manager.release_prefix_snapshot(snapshot) {
                    Ok(retirement) => retirement,
                    Err(error) => return Err((error, cleanup)),
                },
                None => {
                    return Err((
                        Error::Invariant {
                            message: "cached prefix snapshot has no authoritative page manager"
                                .into(),
                        },
                        cleanup,
                    ));
                }
            };
            cleanup.snapshot = None;
            cleanup.retirement = Some(PendingKvRetirement::BackendRelease(retirement));
        }
        if let Some(retirement) = cleanup.retirement.take()
            && let Err((error, retirement)) = self.progress_kv_retirement(retirement)
        {
            cleanup.retirement = Some(retirement);
            return Err((error, cleanup));
        }
        if let Some(model_state) = cleanup.model_state.take()
            && let Err(failure) = self.executor.try_release_sequence_state(model_state)
        {
            let (error, model_state) = failure.into_parts();
            cleanup.model_state = Some(model_state);
            return Err((error.into(), cleanup));
        }
        Ok(())
    }

    fn progress_prefix_cleanups(&mut self) -> Result<()> {
        if self.has_live_transactions() {
            return Ok(());
        }
        while let Some(cleanup) = self.pending_prefix_cleanups.pop_front() {
            if let Err((error, cleanup)) = self.progress_prefix_cleanup(cleanup) {
                self.pending_prefix_cleanups.push_front(cleanup);
                return Err(error);
            }
        }
        Ok(())
    }

    fn drain_prefix_cache(&mut self) -> Result<()> {
        let removed = self
            .prefix_cache
            .drain()
            .map_err(|error| prefix_cache_error("prefix cache drain", error))?;
        for entry in removed {
            self.queue_removed_prefix(entry);
        }
        self.progress_prefix_cleanups()
    }

    fn available_kv_page_credits(&self) -> usize {
        self.load_registry
            .resources()
            .snapshots()
            .find(|snapshot| snapshot.kind == ResourceKind::KvPage)
            .map_or(0, |snapshot| {
                usize::try_from(snapshot.capacity.saturating_sub(snapshot.in_use))
                    .unwrap_or(usize::MAX)
            })
    }

    fn evict_prefixes_for_kv_pages(&mut self, required: usize) -> Result<()> {
        if required == 0 || self.available_kv_page_credits() >= required {
            return Ok(());
        }
        if self.has_live_transactions() {
            return Ok(());
        }
        while self.available_kv_page_credits() < required {
            let Some(removed) = self.prefix_cache.evict_lru() else {
                break;
            };
            self.queue_removed_prefix(removed);
            self.progress_prefix_cleanups()?;
        }
        Ok(())
    }

    pub fn suspended_len(&self) -> usize {
        self.suspended_sequences.len()
    }

    fn ensure_no_suspended_execution(&self, operation: &str) -> Result<()> {
        if self.has_pending_async_work() {
            return Err(Error::InvalidRequest {
                message: format!("cannot {operation} while execution transactions are live"),
            });
        }
        Ok(())
    }

    fn take_transaction_id(&mut self) -> Result<ExecutionTransactionId> {
        let value = self.next_transaction_id;
        self.next_transaction_id = value.checked_add(1).ok_or_else(|| Error::InvalidRequest {
            message: "resident execution transaction ID space is exhausted".into(),
        })?;
        Ok(ExecutionTransactionId::new(value)?)
    }

    #[cfg(test)]
    fn owns_transaction(&self, transaction: ExecutionTransactionId) -> bool {
        self.resident_transactions.contains_key(&transaction)
            || self.speculative_transactions.contains_key(&transaction)
    }

    fn has_live_transactions(&self) -> bool {
        !self.resident_transactions.is_empty() || !self.speculative_transactions.is_empty()
    }

    fn runtime_now_ns(&self) -> u64 {
        self.runtime_clock
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64
    }

    /// Feeds one synchronous compute span to the critical-path ledger so that
    /// shared dependency waits overlapping it are reported as covered rather
    /// than uncovered critical-path time.
    fn record_runnable_work_span(&mut self, started_ns: u64) -> Result<()> {
        self.load_registry
            .record_runnable_work(started_ns, self.runtime_now_ns())
            .map_err(Error::from)
    }

    fn transaction_is_cancelling(&self, transaction: ExecutionTransactionId) -> bool {
        self.resident_transactions
            .get(&transaction)
            .is_some_and(|pending| pending.phase.is_ending())
            || self.speculative_transactions.get(&transaction).is_some_and(
                |pending| match pending {
                    PendingSpeculativeDriverCohort::Proposing(pending) => {
                        pending.cancellation_request.is_some() || pending.abort_cause.is_some()
                    }
                    PendingSpeculativeDriverCohort::Verifying(pending) => {
                        pending.cancellation_request.is_some()
                    }
                    PendingSpeculativeDriverCohort::Ending(_) => true,
                },
            )
    }

    fn finish_speculative_transaction_custody(
        &mut self,
        transaction: ExecutionTransactionId,
        outcome: TransactionCustodyOutcome,
    ) -> Result<()> {
        let now_ns = self.runtime_now_ns();
        self.load_registry
            .finish_transaction_custody(transaction, outcome, now_ns)
            .map_err(Error::from)
    }

    fn action_demand(action: &SchedulerAction) -> crate::scheduling::ResourceDemand {
        let phases = match action {
            SchedulerAction::PrefillChunk(_) => ExecutionPhaseSet::one(ExecutionPhase::Prefill),
            SchedulerAction::DecodeBatch(_) => ExecutionPhaseSet::one(ExecutionPhase::Decode),
            SchedulerAction::Execute { prefills, decodes } => {
                let phases = ExecutionPhaseSet::one(if prefills.is_empty() {
                    ExecutionPhase::Decode
                } else {
                    ExecutionPhase::Prefill
                });
                if decodes.is_empty() {
                    phases
                } else {
                    phases.with(ExecutionPhase::Decode)
                }
            }
            SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => {
                unreachable!("terminal scheduler actions never lower to execution batches")
            }
        };
        crate::scheduling::ResourceDemand::required_phases(phases)
    }

    fn enqueue_transaction(&mut self, transaction: ExecutionTransactionId) {
        if self.queued_transactions.insert(transaction) {
            self.ready_transactions.push_back(transaction);
        }
    }

    fn pop_ready_transaction(&mut self) -> Option<ExecutionTransactionId> {
        let transaction = self.ready_transactions.pop_front()?;
        self.queued_transactions.remove(&transaction);
        Some(transaction)
    }

    fn remove_transaction_from_queue(&mut self, transaction: ExecutionTransactionId) {
        self.ready_transactions
            .retain(|queued| *queued != transaction);
        self.queued_transactions.remove(&transaction);
    }

    fn ready_continuation_for(
        &self,
        transaction: ExecutionTransactionId,
    ) -> Option<ContinuationId> {
        self.transaction_continuations
            .get(&transaction)
            .into_iter()
            .flatten()
            .filter(|continuation| self.ready_continuations.contains(continuation))
            .copied()
            .min_by_key(|continuation| continuation.get())
    }

    fn declare_transaction_prefetch(
        &mut self,
        transaction: ExecutionTransactionId,
        required: crate::scheduling::ResourceDemand,
        batch: &ferrule_common::execution::ExecutionBatch,
    ) -> Result<()> {
        let phases = required.phases().ok_or_else(|| Error::Invariant {
            message: "transaction execution demand has no execution phase".into(),
        })?;
        let demand = crate::scheduling::ResourceDemand::prefetch_phases(phases);
        let requests = self
            .executor
            .runner()
            .transaction_prefetch_requests(transaction, batch)?;
        if requests.is_empty() {
            return Ok(());
        }
        let owner = crate::io::PrefetchOwner::transaction(transaction, phases);
        let report = self.prefetch(owner, demand, requests)?;
        let operations = report.created.len().saturating_add(report.joined.len());
        if operations == 0 {
            return Ok(());
        }
        let transition_budget = operations.saturating_mul(2).max(1);
        let now_ns = self.runtime_now_ns();
        self.load_registry
            .drive_foreground(now_ns, transition_budget)?;
        Ok(())
    }

    fn register_pending_progress(
        &mut self,
        progress: &PendingModelProgress,
        demand: crate::scheduling::ResourceDemand,
    ) -> Result<()> {
        progress.dependencies().validate()?;
        let continuation = progress.continuation();
        let transaction = progress.transaction();
        if self.continuations.contains_key(&continuation) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "model continuation {} is already registered",
                    continuation.get()
                ),
            });
        }
        let epoch = DependencySetEpoch::new(self.next_dependency_epoch);
        self.next_dependency_epoch =
            self.next_dependency_epoch
                .checked_add(1)
                .ok_or_else(|| Error::InvalidRequest {
                    message: "dependency-set epoch space is exhausted".into(),
                })?;
        let waiter = WaiterId::new(transaction, RequestGeneration::new(1), epoch, continuation)?;
        let keys = progress
            .dependencies()
            .iter()
            .filter_map(|dependency| dependency.materialization_key())
            .collect::<Vec<_>>();
        if keys.len() != progress.resources().len()
            || progress
                .resources()
                .iter()
                .zip(&keys)
                .any(|(resource, key)| resource.key() != *key)
        {
            return Err(Error::InvalidRequest { message:
                "pending model progress resource custody does not exactly match its dependencies"
                    .into(),
             });
        }
        let requests = match progress
            .resources()
            .iter()
            .map(|resource| {
                self.load_registry.prepare_execution_request(
                    resource.key(),
                    demand,
                    resource.retention(),
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()
        {
            Ok(requests) => requests,
            Err(error) => {
                let cleanup = self
                    .load_registry
                    .discard_preparations(keys.iter().copied());
                return Err(Error::with_cleanup(
                    "materialization request validation",
                    error,
                    cleanup.map_err(Error::from),
                ));
            }
        };
        if let Err(error) =
            self.load_registry
                .attach_waiter(waiter, demand, requests, self.runtime_now_ns())
        {
            let cleanup = self
                .load_registry
                .discard_preparations(keys.iter().copied());
            return Err(Error::with_cleanup(
                "materialization attachment",
                error,
                cleanup.map_err(Error::from),
            ));
        }

        self.continuations.insert(
            continuation,
            RegisteredModelContinuation {
                transaction,
                dependencies: progress.dependencies().clone(),
            },
        );
        self.transaction_continuations
            .entry(transaction)
            .or_default()
            .insert(continuation);

        if !keys.is_empty() {
            // The completion listener is armed before model execution. Newly
            // attached registry work is owner-local and has no provider event yet,
            // so explicitly schedule the next owner step that submits it.
            self.completion_hub.notify();
        }
        Ok(())
    }

    fn snapshot_transaction_outputs(
        &mut self,
        transaction: ExecutionTransactionId,
        externally_committed_tokens: usize,
    ) -> Result<()> {
        let token_count =
            u64::try_from(externally_committed_tokens).map_err(|_| Error::InvalidRequest {
                message: "externally committed token count exceeds u64".into(),
            })?;
        let next = self
            .next_output_token_id
            .checked_add(token_count)
            .ok_or_else(|| Error::InvalidRequest {
                message: "output token identity space is exhausted".into(),
            })?;
        let captured_at_ns = self.runtime_now_ns();
        for token in self.next_output_token_id..next {
            self.load_registry.snapshot_transaction_output(
                transaction,
                OutputTokenId::new(token),
                externally_committed_tokens,
                captured_at_ns,
            )?;
        }
        self.next_output_token_id = next;
        Ok(())
    }

    fn update_hard_resource_observability(&mut self) {
        self.observability.stats.hard_resource_high_water = self
            .load_registry
            .resources()
            .snapshots()
            .map(|snapshot| (snapshot.kind, snapshot.high_water))
            .collect();
    }

    fn retry_pending_continuation_cleanups(&mut self) -> Result<()> {
        let mut continuations = self
            .pending_registry_detaches
            .keys()
            .copied()
            .collect::<Vec<_>>();
        continuations.sort_unstable();
        continuations.dedup();
        let mut first_error = None;
        for continuation in continuations {
            if let Some(reason) = self.pending_registry_detaches.get(&continuation).cloned() {
                match self.load_registry.detach_continuation(
                    continuation,
                    reason,
                    self.runtime_now_ns(),
                ) {
                    Ok(()) => {
                        self.pending_registry_detaches.remove(&continuation);
                    }
                    Err(error) => {
                        if first_error.is_none() {
                            first_error = Some(error.into());
                        }
                        continue;
                    }
                }
            }
            if !self.pending_registry_detaches.contains_key(&continuation) {
                self.unregister_continuation(continuation);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn collect_materialization_failures(&mut self) {
        while let Some(failed) = self.load_registry.pop_failed() {
            let transaction = self
                .continuations
                .get(&failed.continuation)
                .map(|registered| registered.transaction);
            self.pending_materialization_failures
                .push_back(PendingMaterializationFailure {
                    failed,
                    transaction,
                });
        }
    }

    fn cleanup_materialization_failures(&mut self, report_business_error: bool) -> Result<()>
    where
        R: ResidentModelRunner,
    {
        self.collect_materialization_failures();
        if self.pending_materialization_failures.is_empty() {
            return Ok(());
        }
        self.load_registry.finish_pending_lease_releases()?;

        let mut failures = Vec::<(Option<ExecutionTransactionId>, Error)>::new();
        while let Some(failure) = self.pending_materialization_failures.pop_front() {
            let continuation = failure.failed.continuation;
            self.unregister_continuation(continuation);
            let error = Error::InvalidRequest {
                message: format!(
                    "materialization for continuation {} failed ({:?}); transaction={:?}",
                    continuation.get(),
                    failure.failed.failure,
                    failure.transaction
                ),
            };
            if let Some(index) = failures
                .iter()
                .position(|(transaction, _)| *transaction == failure.transaction)
            {
                let (transaction, previous) = failures.remove(index);
                failures.insert(
                    index,
                    (
                        transaction,
                        Error::combine("transaction materialization", previous, error),
                    ),
                );
            } else {
                failures.push((failure.transaction, error));
            }
        }

        let mut first_error = None;
        for (transaction, error) in failures {
            let terminal_error = match transaction {
                Some(transaction)
                    if self.resident_transactions.contains_key(&transaction)
                        || self.speculative_transactions.contains_key(&transaction) =>
                {
                    self.abort_materialization_failed_transaction(transaction, error)
                }
                _ => Some(error),
            };
            if first_error.is_none() {
                first_error = terminal_error;
            }
        }

        if report_business_error {
            first_error.map_or(Ok(()), Err)
        } else {
            Ok(())
        }
    }

    fn abort_materialization_failed_transaction(
        &mut self,
        transaction: ExecutionTransactionId,
        error: Error,
    ) -> Option<Error>
    where
        R: ResidentModelRunner,
    {
        self.remove_transaction_from_queue(transaction);
        if let Some(pending) = self.resident_transactions.remove(&transaction) {
            return self
                .abort_failed_resident(pending, error, "model materialization")
                .err();
        }

        let Some(pending) = self.speculative_transactions.remove(&transaction) else {
            return Some(Error::Invariant {
                message: format!(
                    "materialization failure lost execution transaction {transaction:?} ownership"
                ),
            });
        };
        match pending {
            PendingSpeculativeDriverCohort::Proposing(pending) => self
                .cancel_native_proposal_cohort(*pending, None, Some(error))
                .err(),
            PendingSpeculativeDriverCohort::Verifying(pending) => {
                let PendingSpeculativeVerificationDriverCohort {
                    transaction,
                    cohort_start,
                    actions,
                    prepared,
                    source_states,
                    schedules,
                    verification,
                    cancellation_request,
                } = *pending;
                let ending = PendingSpeculativeEndingDriverCohort {
                    transaction,
                    cohort_start,
                    started_ns: self.runtime_now_ns(),
                    actions,
                    prepared,
                    source_states,
                    schedules,
                    ending: SpeculativeEnding::Abort {
                        transaction: verification.into_transaction(),
                        failure: Some((error, "model materialization")),
                    },
                    request_id: cancellation_request,
                    continuation: None,
                };
                self.drive_speculative_ending(ending, &mut |_| Ok(())).err()
            }
            PendingSpeculativeDriverCohort::Ending(pending) => {
                self.speculative_transactions
                    .insert(transaction, PendingSpeculativeDriverCohort::Ending(pending));
                Some(Error::Invariant {
                    message: format!(
                        "transaction {transaction:?} reported materialization failure while terminalizing"
                    ),
                })
            }
        }
    }

    fn foreground_lifecycle_active(&self) -> bool {
        !self.scheduler.is_idle()
            || !self.resident_transactions.is_empty()
            || !self.speculative_transactions.is_empty()
            || !self.continuations.is_empty()
            || !self.pending_sequence_cleanups.is_empty()
            || !self.pending_materialization_failures.is_empty()
    }

    fn materialization_transition_budget(&self) -> usize {
        const ACTIVE_TRANSITIONS_PER_SLICE: usize = 2;
        const IDLE_MINIMUM_TRANSITIONS: usize = 4_096;
        const IDLE_MAXIMUM_TRANSITIONS: usize = 32_768;
        const FOREGROUND_TRANSITIONS: usize = 512;

        if self.foreground_lifecycle_active() || !self.warmup_pending() {
            FOREGROUND_TRANSITIONS
        } else {
            self.load_registry
                .active_operations()
                .saturating_mul(ACTIVE_TRANSITIONS_PER_SLICE)
                .clamp(IDLE_MINIMUM_TRANSITIONS, IDLE_MAXIMUM_TRANSITIONS)
        }
    }

    fn progress_materialization(&mut self) -> Result<()>
    where
        R: ResidentModelRunner,
    {
        self.retry_pending_continuation_cleanups()?;
        self.cleanup_materialization_failures(true)?;

        let now_ns = self.runtime_now_ns();
        let transition_budget = self.materialization_transition_budget();
        let progressed = if self.foreground_lifecycle_active() {
            self.load_registry
                .drive_foreground(now_ns, transition_budget)?
        } else {
            self.load_registry.drive(now_ns, transition_budget)?
        };
        // The inference owner arms its listener before entering this method. If
        // owner-side work consumes the whole slice, publish a local wake so a
        // runnable transition left at the budget boundary cannot wait forever
        // for a provider completion that may never be needed.
        if progressed == transition_budget {
            self.completion_hub.notify();
        }
        const WARMUP_REPORT_INTERVAL_NS: u64 = 1_000_000_000;
        let trace_enabled = std::env::var_os("FERRULE_IO_TRACE").is_some();
        let report_warmup = self.warmup_pending()
            && now_ns.saturating_sub(self.warmup_last_report_ns) >= WARMUP_REPORT_INTERVAL_NS;
        if trace_enabled && report_warmup {
            self.warmup_last_report_ns = now_ns;
            let stages = self.load_registry.stage_counts().collect::<Vec<_>>();
            let resources = self
                .load_registry
                .resources()
                .snapshots()
                .filter(|snapshot| {
                    matches!(
                        snapshot.kind,
                        ResourceKind::ReadSlot
                            | ResourceKind::PinnedHostBytes
                            | ResourceKind::StorageReadBytes
                            | ResourceKind::UploadSlot
                            | ResourceKind::UploadBytes
                            | ResourceKind::InstallSlot
                            | ResourceKind::DeviceInstallBytes
                            | ResourceKind::ResidentBytes
                            | ResourceKind::ResidencyLease
                            | ResourceKind::LoadOperation
                    )
                })
                .map(|snapshot| (snapshot.kind, snapshot.in_use, snapshot.capacity))
                .collect::<Vec<_>>();
            let resident = self.load_registry.resident_entries();
            let resident_bytes = self.load_registry.resident_bytes();
            let elapsed_seconds = self
                .warmup_started_ns
                .map(|started| now_ns.saturating_sub(started) as f64 / 1_000_000_000.0)
                .unwrap_or_default();
            let gib_per_second = if elapsed_seconds > 0.0 {
                resident_bytes as f64 / elapsed_seconds / (1u64 << 30) as f64
            } else {
                0.0
            };
            let total = resident.saturating_add(
                self.load_registry
                    .prefetch_operations(crate::io::PrefetchOwner::ModelWarmup)
                    .count(),
            );
            let eta_seconds = if resident != 0 && elapsed_seconds > 0.0 {
                elapsed_seconds * total.saturating_sub(resident) as f64 / resident as f64
            } else {
                0.0
            };
            eprintln!(
                "[ferrule-io] tick={} progressed={} active={} physical={} runnable={} completions={} resident={}/{} resident_bytes={} elapsed_s={elapsed_seconds:.2} gib_s={gib_per_second:.2} eta_s={eta_seconds:.1} stages={stages:?} resources={resources:?}",
                self.runtime_tick,
                progressed,
                self.load_registry.active_operations(),
                self.load_registry.pending_physical_operations(),
                self.load_registry.runnable_actions(),
                self.load_registry.pending_completions(),
                resident,
                total,
                resident_bytes,
            );
        }
        self.cleanup_materialization_failures(true)?;
        let mut newly_ready = Vec::new();
        while let Some(continuation) = self.load_registry.pop_ready(now_ns)? {
            let transaction = self
                .continuations
                .get(&continuation)
                .ok_or_else(|| Error::Invariant {
                    message: format!(
                        "ready continuation {} has no transaction registration",
                        continuation.get()
                    ),
                })?
                .transaction;
            self.ready_continuations.insert(continuation);
            newly_ready.push((continuation, transaction));
        }
        newly_ready.sort_unstable_by_key(|(continuation, transaction)| {
            (
                self.transaction_is_cancelling(*transaction),
                transaction.get(),
                continuation.get(),
            )
        });
        for (_, transaction) in newly_ready {
            self.enqueue_transaction(transaction);
        }
        Ok(())
    }

    fn prepare_resume_lease(
        &mut self,
        continuation: ContinuationId,
    ) -> Result<crate::io::ResumeLease> {
        let dependencies = self
            .continuations
            .get(&continuation)
            .ok_or_else(|| Error::InvalidRequest {
                message: format!(
                    "continuation {} is not registered for resume",
                    continuation.get()
                ),
            })?
            .dependencies
            .clone();
        self.load_registry
            .prepare_resume(continuation, &dependencies)
            .map_err(Error::from)
    }

    fn finish_resume_lease(
        &mut self,
        continuation: ContinuationId,
        mut lease: crate::io::ResumeLease,
        disposition: ResumeDisposition,
        started_ns: u64,
    ) -> Result<()> {
        debug_assert_eq!(lease.continuation(), continuation);
        self.load_registry.finish_resume(
            &mut lease,
            disposition,
            started_ns,
            self.runtime_now_ns(),
        )?;
        if disposition == ResumeDisposition::Consumed {
            self.unregister_continuation(continuation);
        }
        Ok(())
    }

    fn detach_registered_continuation(
        &mut self,
        continuation: ContinuationId,
        reason: CancellationReason,
    ) -> Result<()> {
        if !self.continuations.contains_key(&continuation) {
            return Ok(());
        }
        self.load_registry.detach_continuation(
            continuation,
            reason.clone(),
            self.runtime_now_ns(),
        )?;
        self.unregister_continuation(continuation);
        Ok(())
    }

    fn unregister_continuation(&mut self, continuation: ContinuationId) {
        self.ready_continuations.remove(&continuation);
        self.pending_registry_detaches.remove(&continuation);
        let Some(registered) = self.continuations.remove(&continuation) else {
            return;
        };
        if let Some(continuations) = self
            .transaction_continuations
            .get_mut(&registered.transaction)
        {
            continuations.remove(&continuation);
            if continuations.is_empty() {
                self.transaction_continuations
                    .remove(&registered.transaction);
            }
        }
    }

    /// Suspend one active session and move its exclusively owned physical pages
    /// out of backend device residency.
    pub fn preempt_session(&mut self, session_id: SessionId) -> Result<()> {
        self.ensure_no_suspended_execution("preempt a session")?;
        if self.suspended_sequences.contains_key(&session_id) {
            return Err(Error::InvalidRequest {
                message: format!("session {session_id:?} is already suspended"),
            });
        }
        if !self.sequence_states.contains_key(&session_id) {
            return Err(Error::Invariant {
                message: format!("session {session_id:?} has no model sequence state"),
            });
        }
        let schedule = self.scheduler.suspend_sequence(session_id)?;
        let slot = match self.page_slots.get(&session_id).copied() {
            Some(slot) => slot,
            None => {
                self.scheduler.restore_suspended(schedule)?;
                return Err(Error::InvalidRequest {
                    message: format!("session {session_id:?} has no authoritative page slot"),
                });
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
                return Err(Error::InvalidRequest {
                    message: "session preemption requires an authoritative KvPageManager".into(),
                });
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
            .ok_or_else(|| Error::InvalidRequest {
                message: format!("session {session_id:?} is not suspended"),
            })?;
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
            .ok_or_else(|| Error::InvalidRequest {
                message: "KvPageManager was removed while suspended".into(),
            })?
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
        if let Err(error) = self
            .scheduler
            .retry_failed_slot_releases(&mut self.slot_pool)
        {
            return Err(Box::new((error, self)));
        }
        if self.has_pending_non_prefix_async_work()
            || self.warmup_pending()
            || !self.session_owner.is_empty()
            || !self.committed_token_outbox.is_empty()
        {
            return Err(Box::new((
                Error::InvalidRequest { message:
                    "cannot extract resident runner with live execution transactions or undelivered committed events"
                        .into(),
                 },
                self,
            )));
        }
        if !self.scheduler.is_idle()
            || !self.suspended_sequences.is_empty()
            || !self.sequence_states.is_empty()
            || !self.retained_sessions.is_empty()
            || !self.prefix_cache_sessions.is_empty()
        {
            return Err(Box::new((
                Error::InvalidRequest { message:
                    "cannot extract resident runner while session state is still retained or active"
                        .into(),
                 },
                self,
            )));
        }

        if let Err(error) = self.drain_prefix_cache() {
            return Err(Box::new((error, self)));
        }
        let retiring_pages = self
            .page_manager
            .as_ref()
            .map_or(0, |manager| manager.stats().retiring_pages);
        let active_page_sequences = self
            .page_manager
            .as_ref()
            .map_or(0, KvPageManager::active_sequences);
        if !self.pending_prefix_cleanups.is_empty()
            || !self.pending_kv_retirements.is_empty()
            || !self.kv_page_grants.is_empty()
            || retiring_pages != 0
            || active_page_sequences != 0
        {
            return Err(Box::new((
                Error::Invariant {
                    message: format!(
                        "cannot extract resident runner with retained KV ownership: prefix_cleanups={} retirements={} grants={} retiring_pages={retiring_pages} active_page_sequences={active_page_sequences}",
                        self.pending_prefix_cleanups.len(),
                        self.pending_kv_retirements.len(),
                        self.kv_page_grants.len(),
                    ),
                },
                self,
            )));
        }
        if let Err(error) = self.load_registry.shutdown(self.runtime_now_ns(), 0) {
            let error = Error::from(error);
            return Err(Box::new((error, self)));
        }

        Ok(self.executor.into_runner())
    }

    /// Keep a session's model and KV state resident after each request finishes.
    /// Subsequent requests with this explicit session ID append at the last
    /// committed position instead of creating a fresh sequence.
    pub fn retain_session(&mut self, session_id: SessionId) -> Result<()> {
        if !self.scheduler.is_idle() || self.suspended_sequences.contains_key(&session_id) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "cannot retain session {session_id:?} while scheduler work is active or suspended"
                ),
            });
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
            return Err(Error::InvalidRequest {
                message: format!(
                    "cannot release session {session_id:?} while scheduler work is active or suspended"
                ),
            });
        }
        self.release_sequence_state(session_id)?;
        self.retained_sessions.remove(&session_id);
        Ok(())
    }

    /// Reset an idle retained session to an empty position while preserving its
    /// retained lifecycle registration for future request turns.
    pub fn reset_session(&mut self, session_id: SessionId) -> Result<()> {
        if !self.retained_sessions.contains_key(&session_id) {
            return Err(Error::InvalidRequest {
                message: format!("session {session_id:?} is not retained"),
            });
        }
        self.release_session(session_id)?;
        self.retained_sessions.insert(session_id, 0);
        Ok(())
    }

    pub fn stats(&self) -> &ResidentTopKDriverStats {
        self.observability.stats()
    }

    pub fn prefix_cache_stats(&self) -> &ResidentPrefixCacheStats {
        self.observability.prefix_cache_stats()
    }

    pub fn prefix_hits(&self) -> usize {
        self.observability.prefix_hits()
    }

    pub fn prefix_misses(&self) -> usize {
        self.observability.prefix_misses()
    }

    /// Validate scheduler policy against the truthful capabilities of the native
    /// multi-session executor before any queue entry is consumed.
    pub fn validate_configuration(&self) -> Result<()> {
        let capabilities = self.executor.capabilities();
        let scheduler = self.scheduler.config();
        if scheduler.max_active_sequences > capabilities.max_sequences {
            return Err(Error::InvalidRequest {
                message: format!(
                    "resident driver config allows {} active sequences, but its executor supports {}",
                    scheduler.max_active_sequences, capabilities.max_sequences
                ),
            });
        }
        if scheduler.max_decode_batch > capabilities.max_sequences {
            return Err(Error::InvalidRequest {
                message: format!(
                    "resident driver config allows decode batch {}, but its executor supports {} sequence",
                    scheduler.max_decode_batch, capabilities.max_sequences
                ),
            });
        }
        if capabilities
            .max_top_k
            .is_some_and(|maximum| self.top_k > maximum)
        {
            return Err(Error::InvalidRequest {
                message: format!(
                    "resident driver requests top-k {}, exceeding executor capability",
                    self.top_k.get()
                ),
            });
        }
        Ok(())
    }

    pub fn try_submit(&mut self, request: GenerateRequest) -> Result<()> {
        if self.shutting_down {
            return Err(Error::InvalidRequest {
                message: "resident driver admission is closed during shutdown".into(),
            });
        }
        let retained_position = request
            .session_id
            .and_then(|session_id| self.retained_sessions.get(&session_id).copied());
        if let Some(position) = retained_position {
            self.scheduler.submit_at_position(request, position);
        } else {
            self.scheduler.submit(request);
        }
        Ok(())
    }

    pub fn submit(&mut self, request: GenerateRequest) {
        if let Err(error) = self.try_submit(request) {
            tracing::warn!(error = %error, "resident request rejected");
        }
    }

    /// Submit a request at a specific position. This is used for testing and
    /// for single-runner backends where the caller knows the runner's current
    /// position.
    pub fn try_submit_at_position(
        &mut self,
        request: GenerateRequest,
        position_start: usize,
    ) -> Result<()> {
        if self.shutting_down {
            return Err(Error::InvalidRequest {
                message: "resident driver admission is closed during shutdown".into(),
            });
        }
        self.scheduler.submit_at_position(request, position_start);
        Ok(())
    }

    pub fn submit_at_position(&mut self, request: GenerateRequest, position_start: usize) {
        if let Err(error) = self.try_submit_at_position(request, position_start) {
            tracing::warn!(error = %error, "resident positioned request rejected");
        }
    }

    /// Request cancellation without releasing resources before backend quiescence.
    /// Active transactions are driven automatically by subsequent owner ticks.
    pub fn cancel_request(&mut self, request_id: RequestId) -> Result<ResidentCancelProgress>
    where
        R: ResidentModelRunner,
    {
        if let Some(pending) = self.pending_request_cancellations.get(&request_id).copied() {
            if let Some(transaction) = pending.transaction {
                return self.request_transaction_abort(transaction, request_id);
            }
            return self
                .progress_request_cancellation(request_id)
                .map(|result| match result {
                    Some(result) => ResidentCancelProgress::Complete(result),
                    None => ResidentCancelProgress::Pending,
                });
        }
        if let Some(transaction) = self.transaction_for_request(request_id) {
            let session_id = self
                .session_for_transaction_request(transaction, request_id)
                .ok_or_else(|| Error::Invariant {
                    message: format!(
                        "transaction {transaction:?} lost request {request_id:?} session ownership"
                    ),
                })?;
            self.register_transaction_cancellation(transaction, request_id, session_id)?;
            return self.request_transaction_abort(transaction, request_id);
        }
        self.cancel_scheduled_request(request_id)
            .map(|result| match result {
                Some(result) => ResidentCancelProgress::Complete(result),
                None => ResidentCancelProgress::Pending,
            })
    }

    fn cancel_scheduled_request(
        &mut self,
        request_id: RequestId,
    ) -> Result<Option<CancelRequestResult>> {
        if !self.pending_request_cancellations.contains_key(&request_id) {
            if let Some(session_id) = self.scheduler.active_session_for_request(request_id) {
                self.pending_request_cancellations.insert(
                    request_id,
                    PendingRequestCancellation {
                        session_id,
                        transaction: None,
                        ready: true,
                        result: None,
                    },
                );
            } else {
                return self
                    .scheduler
                    .cancel_request(request_id, &mut self.slot_pool)
                    .map(Some);
            }
        }
        self.progress_request_cancellation(request_id)
    }

    fn progress_request_cancellation(
        &mut self,
        request_id: RequestId,
    ) -> Result<Option<CancelRequestResult>> {
        let mut pending = self
            .pending_request_cancellations
            .remove(&request_id)
            .ok_or_else(|| Error::Invariant {
                message: format!("request {request_id:?} has no pending cancellation owner"),
            })?;
        if !pending.ready {
            self.pending_request_cancellations
                .insert(request_id, pending);
            return Ok(None);
        }
        if pending.result.is_none() {
            match self
                .scheduler
                .cancel_request(request_id, &mut self.slot_pool)
            {
                Ok(_) => {
                    pending.result = Some(CancelRequestResult::Active {
                        request_id,
                        session_id: pending.session_id,
                    });
                }
                Err(error) => {
                    pending.result = Some(CancelRequestResult::Active {
                        request_id,
                        session_id: pending.session_id,
                    });
                    self.pending_request_cancellations
                        .insert(request_id, pending);
                    return Err(error);
                }
            }
        }
        if let Err(error) = self
            .scheduler
            .retry_failed_slot_releases(&mut self.slot_pool)
        {
            self.pending_request_cancellations
                .insert(request_id, pending);
            return Err(error);
        }
        if let Some(position) = self.retained_sessions.get_mut(&pending.session_id) {
            *position = 0;
        }
        if let Err(error) = self.release_sequence_state(pending.session_id) {
            self.pending_request_cancellations
                .insert(request_id, pending);
            return Err(error);
        }
        if self
            .pending_sequence_cleanups
            .contains_key(&pending.session_id)
        {
            self.pending_request_cancellations
                .insert(request_id, pending);
            return Ok(None);
        }
        let result = pending
            .result
            .expect("completed cancellation retains its scheduler result");
        Ok(Some(result))
    }

    fn register_transaction_cancellation(
        &mut self,
        transaction: ExecutionTransactionId,
        request_id: RequestId,
        session_id: SessionId,
    ) -> Result<()> {
        if let Some(existing) = self.pending_request_cancellations.get(&request_id) {
            if existing.session_id != session_id || existing.transaction != Some(transaction) {
                return Err(Error::Invariant {
                    message: format!(
                        "request {request_id:?} cancellation owner changed from session {:?} transaction {:?} to session {session_id:?} transaction {transaction:?}",
                        existing.session_id, existing.transaction
                    ),
                });
            }
            return Ok(());
        }
        self.pending_request_cancellations.insert(
            request_id,
            PendingRequestCancellation {
                session_id,
                transaction: Some(transaction),
                ready: false,
                result: None,
            },
        );
        Ok(())
    }

    fn mark_transaction_cancellations_ready(&mut self, transaction: ExecutionTransactionId) {
        for pending in self.pending_request_cancellations.values_mut() {
            if pending.transaction == Some(transaction) {
                pending.transaction = None;
                pending.ready = true;
            }
        }
    }

    fn retry_pending_request_cancellations(&mut self) -> Result<()> {
        let requests = self
            .pending_request_cancellations
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for request_id in requests {
            let _ = self.progress_request_cancellation(request_id)?;
        }
        Ok(())
    }

    fn finish_transaction_cancellations(
        &mut self,
        transaction: ExecutionTransactionId,
        requested: Option<RequestId>,
    ) -> Result<Option<CancelRequestResult>> {
        let requests = self
            .pending_request_cancellations
            .iter()
            .filter_map(|(request_id, pending)| {
                (pending.transaction == Some(transaction)).then_some(*request_id)
            })
            .collect::<Vec<_>>();
        self.mark_transaction_cancellations_ready(transaction);
        let mut requested_result = None;
        for request_id in requests {
            let result = self.progress_request_cancellation(request_id)?;
            if requested == Some(request_id) {
                requested_result = result;
            }
        }
        Ok(requested_result)
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

    fn session_for_transaction_request(
        &self,
        transaction: ExecutionTransactionId,
        request_id: RequestId,
    ) -> Option<SessionId> {
        self.resident_transactions
            .get(&transaction)
            .and_then(|pending| action_session_for_request(&pending.action, request_id))
            .or_else(|| {
                self.speculative_transactions
                    .get(&transaction)
                    .and_then(|pending| {
                        pending.actions().iter().find_map(|action| {
                            (action.request_id == Some(request_id)).then_some(action.session_id)
                        })
                    })
            })
    }

    fn request_for_transaction(&self, transaction: ExecutionTransactionId) -> Option<RequestId> {
        self.resident_transactions
            .get(&transaction)
            .and_then(|pending| action_request_id(&pending.action))
            .or_else(|| {
                self.speculative_transactions
                    .get(&transaction)
                    .and_then(|pending| {
                        pending
                            .actions()
                            .iter()
                            .find_map(|action| action.request_id)
                    })
            })
    }

    fn request_transaction_abort(
        &mut self,
        transaction: ExecutionTransactionId,
        request_id: RequestId,
    ) -> Result<ResidentCancelProgress>
    where
        R: ResidentModelRunner,
    {
        if !self.pending_request_cancellations.contains_key(&request_id) {
            let session_id = self
                .session_for_transaction_request(transaction, request_id)
                .ok_or_else(|| Error::Invariant {
                    message: format!(
                        "transaction {transaction:?} lost request {request_id:?} session ownership"
                    ),
                })?;
            self.register_transaction_cancellation(transaction, request_id, session_id)?;
        }
        self.remove_transaction_from_queue(transaction);
        if let Some(mut pending) = self.resident_transactions.remove(&transaction) {
            match &mut pending.phase {
                ResidentTransactionPhase::Aborting { request, .. } => {
                    if request.is_none() {
                        *request = Some(request_id);
                    }
                }
                ResidentTransactionPhase::BackendAbortedPendingCleanup { request, .. } => {
                    if request.is_none() {
                        *request = Some(request_id);
                    }
                    return self.finish_resident_abort(pending, Some(request_id)).map(
                        |(_, cancellation)| match cancellation {
                            Some(result) => ResidentCancelProgress::Complete(result),
                            None => ResidentCancelProgress::Pending,
                        },
                    );
                }
                ResidentTransactionPhase::BackendCommittedPendingPublish {
                    cancellation_request,
                    ..
                }
                | ResidentTransactionPhase::Publishing {
                    cancellation_request,
                    ..
                } => {
                    if cancellation_request.is_none() {
                        *cancellation_request = Some(request_id);
                    }
                    self.resident_transactions.insert(transaction, pending);
                    self.enqueue_transaction(transaction);
                    return Ok(ResidentCancelProgress::Pending);
                }
                ResidentTransactionPhase::Executing(_) => {
                    let continuation = pending
                        .phase
                        .pending_progress()
                        .map(PendingModelProgress::continuation);
                    pending.phase = ResidentTransactionPhase::Aborting {
                        request: Some(request_id),
                        custody: TransactionCustodyOutcome::Cancelled,
                        continuation,
                        failure: None,
                    };
                }
            }
            match self.executor.end_transaction(
                transaction,
                &mut pending.states,
                TransactionEndIntent::Abort,
            ) {
                Ok(TransactionEndProgress::Pending) => {
                    self.resident_transactions.insert(transaction, pending);
                    self.enqueue_transaction(transaction);
                    return Ok(ResidentCancelProgress::Pending);
                }
                Err(error) => {
                    self.resident_transactions.insert(transaction, pending);
                    return Err(error);
                }
                Ok(TransactionEndProgress::Complete) => {
                    let ResidentTransactionPhase::Aborting {
                        request,
                        custody,
                        continuation,
                        failure,
                    } = std::mem::replace(
                        &mut pending.phase,
                        ResidentTransactionPhase::Executing(None),
                    )
                    else {
                        unreachable!("resident cancellation must own an abort phase")
                    };
                    pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
                        request,
                        custody: Some(custody),
                        continuation,
                        failure,
                        decode_requeued: false,
                    };
                    return self.finish_resident_abort(pending, Some(request_id)).map(
                        |(_, cancellation)| match cancellation {
                            Some(result) => ResidentCancelProgress::Complete(result),
                            None => ResidentCancelProgress::Pending,
                        },
                    );
                }
            }
        }
        let pending = self
            .speculative_transactions
            .remove(&transaction)
            .ok_or_else(|| Error::Invariant {
                message: format!("transaction {transaction:?} has no driver ownership"),
            })?;
        let request_session = pending.actions().iter().find_map(|action| {
            (action.request_id == Some(request_id)).then_some(action.session_id)
        });
        let (cancellation, scheduler_finished) = match pending {
            PendingSpeculativeDriverCohort::Proposing(pending) => {
                let result = self.cancel_native_proposal_cohort(*pending, Some(request_id), None);
                if let Some(retained) = self.speculative_transactions.get(&transaction) {
                    let backend_aborted = matches!(
                        retained,
                        PendingSpeculativeDriverCohort::Proposing(pending)
                            if pending.backend_aborted
                    );
                    return match result {
                        Ok(TransactionEndProgress::Pending) | Err(_) if backend_aborted => {
                            Ok(ResidentCancelProgress::Pending)
                        }
                        Ok(TransactionEndProgress::Pending) => Ok(ResidentCancelProgress::Pending),
                        Ok(TransactionEndProgress::Complete) => Err(Error::Invariant {
                            message: format!(
                                "proposal transaction {transaction:?} reported complete but retained ownership"
                            ),
                        }),
                        Err(error) => Err(error),
                    };
                }
                (result, false)
            }
            PendingSpeculativeDriverCohort::Verifying(pending) => (
                self.cancel_speculative_verification_cohort(*pending, request_id),
                true,
            ),
            PendingSpeculativeDriverCohort::Ending(mut pending) => {
                let post_terminal = matches!(
                    pending.ending,
                    SpeculativeEnding::BackendCommittedPendingPublish { .. }
                        | SpeculativeEnding::BackendAbortedPendingCleanup { .. }
                );
                if pending.request_id.is_none() {
                    pending.request_id = Some(request_id);
                }
                if !post_terminal {
                    self.speculative_transactions
                        .insert(transaction, PendingSpeculativeDriverCohort::Ending(pending));
                    return Ok(ResidentCancelProgress::Pending);
                }
                self.drive_speculative_ending(*pending, &mut |_| Ok(()))?;
                if self.speculative_transactions.contains_key(&transaction) {
                    return Ok(ResidentCancelProgress::Pending);
                }
                if self.pending_request_cancellations.contains_key(&request_id) {
                    return Ok(ResidentCancelProgress::Pending);
                }
                let session_id = request_session.ok_or_else(|| Error::Invariant {
                    message: format!(
                        "speculative transaction {transaction:?} lost request {request_id:?} ownership"
                    ),
                })?;
                return Ok(ResidentCancelProgress::Complete(
                    CancelRequestResult::Active {
                        request_id,
                        session_id,
                    },
                ));
            }
        };
        if self.speculative_transactions.contains_key(&transaction) {
            return match cancellation {
                Ok(TransactionEndProgress::Pending) => Ok(ResidentCancelProgress::Pending),
                Ok(TransactionEndProgress::Complete) => Err(Error::Invariant {
                    message: format!(
                        "speculative transaction {transaction:?} reported complete but retained ownership"
                    ),
                }),
                Err(error) => Err(error),
            };
        }
        let terminal = cancellation;
        let cancellation = self.finish_transaction_cancellations(transaction, Some(request_id));
        match (terminal, cancellation) {
            (Ok(_), Ok(Some(result))) => Ok(ResidentCancelProgress::Complete(result)),
            (Ok(_), Ok(None))
                if scheduler_finished
                    && !self.pending_request_cancellations.contains_key(&request_id) =>
            {
                let session_id = request_session.ok_or_else(|| Error::Invariant {
                    message: format!(
                        "speculative transaction {transaction:?} lost request {request_id:?} ownership"
                    ),
                })?;
                Ok(ResidentCancelProgress::Complete(
                    CancelRequestResult::Active {
                        request_id,
                        session_id,
                    },
                ))
            }
            (Ok(_), Ok(None)) => Ok(ResidentCancelProgress::Pending),
            (Err(error), Ok(_)) | (Ok(_), Err(error)) => Err(error),
            (Err(error), Err(cleanup)) => Err(Error::cleanup(
                "speculative request cancellation",
                error,
                cleanup,
            )),
        }
    }

    fn cancel_native_proposal_cohort(
        &mut self,
        mut pending: PendingNativeProposalCohort<R::SequenceState>,
        request_id: Option<RequestId>,
        failure: Option<Error>,
    ) -> Result<TransactionEndProgress>
    where
        R: ResidentModelRunner,
    {
        if let Some(request_id) = request_id {
            pending.cancellation_request = Some(request_id);
        }
        if let Some(failure) = failure {
            pending.abort_cause = Some(match pending.abort_cause.take() {
                Some(previous) => {
                    Error::combine("speculative proposal cancellation", previous, failure)
                }
                None => failure,
            });
        }

        let transaction = pending.transaction;
        if !pending.backend_aborted {
            match self.executor.end_transaction(
                transaction,
                &mut pending.source_states,
                TransactionEndIntent::Abort,
            ) {
                Ok(TransactionEndProgress::Pending) => {
                    self.speculative_transactions.insert(
                        transaction,
                        PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
                    );
                    return Ok(TransactionEndProgress::Pending);
                }
                Ok(TransactionEndProgress::Complete) => {
                    pending.backend_aborted = true;
                    pending.custody = Some(if pending.cancellation_request.is_some() {
                        TransactionCustodyOutcome::Cancelled
                    } else {
                        TransactionCustodyOutcome::RolledBack
                    });
                }
                Err(error) => {
                    self.speculative_transactions.insert(
                        transaction,
                        PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
                    );
                    return Err(error);
                }
            }
        }
        let continuations = pending
            .slots
            .iter()
            .filter_map(|slot| match &slot.status {
                NativeProposalSlotStatus::Waiting(progress) => Some(progress.continuation()),
                NativeProposalSlotStatus::NotStarted
                | NativeProposalSlotStatus::Complete { .. } => None,
            })
            .collect::<Vec<_>>();
        for continuation in continuations {
            if let Err(error) = self
                .detach_registered_continuation(continuation, CancellationReason::ExternalRequest)
            {
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
                );
                return Err(error);
            }
        }
        if let Some(outcome) = pending.custody.take()
            && let Err(error) = self.finish_speculative_transaction_custody(transaction, outcome)
        {
            pending.custody = Some(outcome);
            self.speculative_transactions.insert(
                transaction,
                PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
            );
            return Err(error);
        }
        if let Err(error) = self.progress_transaction_session_restore(
            &mut pending.schedules,
            &mut pending.source_states,
        ) {
            self.speculative_transactions.insert(
                transaction,
                PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
            );
            return Err(error);
        }
        if !pending.decode_requeued {
            if let Err(error) = self
                .scheduler
                .requeue_decode_actions_front(&pending.actions)
            {
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
                );
                return Err(error);
            }
            pending.decode_requeued = true;
        }
        match pending.abort_cause {
            Some(error) => Err(error),
            None => Ok(TransactionEndProgress::Complete),
        }
    }

    fn cancel_speculative_verification_cohort(
        &mut self,
        pending: PendingSpeculativeVerificationDriverCohort<R::SequenceState>,
        request_id: RequestId,
    ) -> Result<TransactionEndProgress>
    where
        R: ResidentModelRunner,
    {
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
        let continuation = verification.pending_progress().continuation();
        let ending = PendingSpeculativeEndingDriverCohort {
            transaction,
            cohort_start,
            started_ns: self.runtime_now_ns(),
            actions,
            prepared,
            source_states,
            schedules,
            ending: SpeculativeEnding::Abort {
                transaction: verification.into_transaction(),
                failure: None,
            },
            request_id: Some(request_id),
            continuation: Some(continuation),
        };
        self.drive_speculative_ending(ending, &mut |_| Ok(()))
            .map(|step| {
                if matches!(step, ResidentDriverStep::Blocked) {
                    TransactionEndProgress::Pending
                } else {
                    TransactionEndProgress::Complete
                }
            })
    }

    fn claim_transaction_sessions(
        &mut self,
        transaction: ExecutionTransactionId,
        session_ids: &[SessionId],
    ) -> Result<(Vec<SuspendedSequenceSchedule>, Vec<R::SequenceState>)> {
        let mut unique = HashSet::with_capacity(session_ids.len());
        for session_id in session_ids {
            if !unique.insert(*session_id) {
                return Err(Error::Invariant {
                    message: format!(
                        "transaction {transaction:?} contains duplicate session {session_id:?}"
                    ),
                });
            }
            if let Some(owner) = self.session_owner.get(session_id) {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "session {session_id:?} is already owned by transaction {owner:?}"
                    ),
                });
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
                return Err(Error::Invariant {
                    message: format!("session {session_id:?} has no model sequence state"),
                });
            };
            self.session_owner.insert(*session_id, transaction);
            schedules.push(schedule);
            states.push(state);
        }
        Ok((schedules, states))
    }

    fn restore_transaction_sessions(
        &mut self,
        mut schedules: Vec<SuspendedSequenceSchedule>,
        mut states: Vec<R::SequenceState>,
    ) -> Result<()> {
        self.progress_transaction_session_restore(&mut schedules, &mut states)
    }

    fn progress_transaction_session_restore(
        &mut self,
        schedules: &mut Vec<SuspendedSequenceSchedule>,
        states: &mut Vec<R::SequenceState>,
    ) -> Result<()> {
        if schedules.len() != states.len() {
            return Err(Error::Invariant {
                message: format!(
                    "transaction schedule/state mismatch: schedules={} states={}",
                    schedules.len(),
                    states.len()
                ),
            });
        }
        let mut unique = HashSet::with_capacity(schedules.len());
        for schedule in schedules.iter() {
            let session_id = schedule.session_id();
            if !unique.insert(session_id) {
                return Err(Error::Invariant {
                    message: format!(
                        "transaction restore contains duplicate session {session_id:?}"
                    ),
                });
            }
            if self.sequence_states.contains_key(&session_id) {
                return Err(Error::Invariant {
                    message: format!("session {session_id:?} model state was already published"),
                });
            }
            if !self.session_owner.contains_key(&session_id) {
                return Err(Error::Invariant {
                    message: format!(
                        "session {session_id:?} lost transaction ownership before restoration"
                    ),
                });
            }
            if self.scheduler.active_sequence(session_id).is_some() {
                return Err(Error::Invariant {
                    message: format!(
                        "cannot restore already-active resident session {session_id:?}"
                    ),
                });
            }
        }
        while let Some(schedule) = schedules.first() {
            let session_id = schedule.session_id();
            self.scheduler
                .restore_suspended(schedule.clone())
                .expect("transaction session restore was preflighted");
            let schedule = schedules.remove(0);
            debug_assert_eq!(schedule.session_id(), session_id);
            let state = states.remove(0);
            self.session_owner.remove(&session_id);
            let previous = self.sequence_states.insert(session_id, state);
            debug_assert!(
                previous.is_none(),
                "transaction session restore was preflighted"
            );
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

        let target_session_id = target_request
            .session_id
            .ok_or_else(|| Error::InvalidRequest {
                message: "exact fork target request requires an explicit session ID".into(),
            })?;
        if self.suspended_sequences.contains_key(&source_session_id) {
            return Err(Error::InvalidRequest {
                message: "cannot fork from a suspended source session".into(),
            });
        }
        if self.suspended_sequences.contains_key(&target_session_id)
            || self.sequence_states.contains_key(&target_session_id)
            || self.page_slots.contains_key(&target_session_id)
        {
            return Err(Error::InvalidRequest {
                message: format!("fork target session {target_session_id:?} already exists"),
            });
        }
        let source_page_slot =
            *self
                .page_slots
                .get(&source_session_id)
                .ok_or_else(|| Error::InvalidRequest {
                    message: format!(
                        "fork source session {source_session_id:?} has no authoritative page slot"
                    ),
                })?;
        let target_page_slot = StateSlot::new(self.next_page_slot);
        let next_page_slot =
            self.next_page_slot
                .checked_add(1)
                .ok_or_else(|| Error::InvalidRequest {
                    message: "driver page slot generation overflow during fork".into(),
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
                return Err(Error::InvalidRequest {
                    message: "exact-prefix fork requires an authoritative KvPageManager".into(),
                });
            }
        };

        let prepared_model = {
            let source =
                self.sequence_states
                    .get(&source_session_id)
                    .ok_or_else(|| Error::InvalidRequest {
                        message: format!(
                            "fork source session {source_session_id:?} has no model state"
                        ),
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
                (release, slot) => Err(Error::Invariant {
                    message: format!(
                        "fork page publish failed ({error}); cleanup model={release:?}, slot={slot:?}"
                    ),
                }),
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

    pub fn take_request_terminal(
        &mut self,
        request_id: RequestId,
    ) -> Option<crate::scheduling::RequestTerminal> {
        self.scheduler.take_request_terminal(request_id)
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

    fn prepare_prefix_admission(
        &mut self,
        prepared: &mut PreparedWaitingAdmission,
        target_slot: StateSlot,
    ) -> Result<Option<PreparedPrefixAdmission<R::SequenceState>>> {
        let Some(namespace) = self.prefix_cache_namespace() else {
            return Ok(None);
        };
        if self.has_live_transactions()
            || !prepared.is_fresh_prompt()
            || prepared.prompt_tokens().is_empty()
        {
            return Ok(None);
        }
        let prompt = prepared.prompt_tokens().to_vec();
        let lease = self
            .prefix_cache
            .lookup_longest_prefix(namespace, &prompt, PrefixLookupLimit::BeforeLastPromptToken)
            .map_err(|error| prefix_cache_error("prefix lookup", error))?;
        let Some(lease) = lease else {
            return Ok(None);
        };
        let matched_tokens = lease.matched_tokens();

        let (model_state, snapshot) = match self
            .prefix_cache
            .payload(&lease)
            .map_err(|error| prefix_cache_error("pinned prefix payload", error))
            .and_then(|payload| {
                self.executor
                    .fork_sequence_state_from(&payload.model_state, matched_tokens)
                    .map(|model_state| (model_state, payload.snapshot))
            }) {
            Ok(prepared) => prepared,
            Err(error) => {
                let cleanup = self
                    .prefix_cache
                    .unpin(lease)
                    .map_err(|error| prefix_cache_error("prefix lookup unpin", error));
                return Err(Error::with_cleanup("prefix model fork", error, cleanup));
            }
        };

        let pages = match self.page_manager.as_ref() {
            Some(manager) => {
                manager.prepare_fork_prefix_snapshot(snapshot, target_slot, 0, matched_tokens)
            }
            None => Err(Error::Invariant {
                message: "prefix cache hit has no authoritative page manager".into(),
            }),
        };
        let pages = match pages {
            Ok(pages) => pages,
            Err(error) => {
                let mut cleanup = Vec::new();
                if let Err(source) = self.executor.release_sequence_state(model_state) {
                    cleanup.push(CleanupStep::new("prefix model fork release", source));
                }
                if let Err(source) = self
                    .prefix_cache
                    .unpin(lease)
                    .map_err(|error| prefix_cache_error("prefix lookup unpin", error))
                {
                    cleanup.push(CleanupStep::new("prefix lookup unpin", source));
                }
                return Err(Error::with_cleanup_batch(
                    "prefix page snapshot fork",
                    error,
                    cleanup,
                ));
            }
        };

        if let Err(error) = prepared.apply_committed_prefix(matched_tokens) {
            let mut cleanup = Vec::new();
            if let Err(source) = self.executor.release_sequence_state(model_state) {
                cleanup.push(CleanupStep::new("prefix model fork release", source));
            }
            if let Err(source) = self
                .prefix_cache
                .unpin(lease)
                .map_err(|error| prefix_cache_error("prefix lookup unpin", error))
            {
                cleanup.push(CleanupStep::new("prefix lookup unpin", source));
            }
            return Err(Error::with_cleanup_batch(
                "prefix scheduler frontier prepare",
                error,
                cleanup,
            ));
        }
        if let Err(error) = self
            .prefix_cache
            .unpin(lease)
            .map_err(|error| prefix_cache_error("prefix lookup unpin", error))
        {
            let cleanup = self.executor.release_sequence_state(model_state);
            return Err(Error::with_cleanup("prefix lookup unpin", error, cleanup));
        }
        Ok(Some(PreparedPrefixAdmission {
            model_state,
            pages,
            matched_tokens,
        }))
    }

    /// Admit waiting sequences only after their model and logical KV ownership is
    /// fully prepared. No fallible work remains once the scheduler publishes the
    /// sequence as active.
    fn admit_new_sequences(&mut self) -> Result<()> {
        if self.shutting_down {
            return Ok(());
        }
        while let Some(mut prepared) = self
            .scheduler
            .prepare_waiting_admission(&mut self.slot_pool)?
        {
            let session_id = prepared.session_id();
            if self.pending_sequence_cleanups.contains_key(&session_id) {
                self.scheduler
                    .abort_waiting_admission(prepared, &mut self.slot_pool)?;
                break;
            }
            if self.sequence_states.contains_key(&session_id) {
                self.scheduler.publish_waiting_admission(prepared);
                continue;
            }

            let cache_eligible = self.prefix_cache_namespace().is_some()
                && !self.has_live_transactions()
                && !self.retained_sessions.contains_key(&session_id)
                && prepared.is_fresh_prompt()
                && !prepared.prompt_tokens().is_empty();
            let (page_slot, next_page_slot) = if self.page_manager.is_some() {
                let slot = StateSlot::new(self.next_page_slot);
                let next_page_slot = match self.next_page_slot.checked_add(1) {
                    Some(next_page_slot) => next_page_slot,
                    None => {
                        let error = Error::InvalidRequest {
                            message: "driver page slot generation overflow".into(),
                        };
                        let cleanup = self
                            .scheduler
                            .abort_waiting_admission(prepared, &mut self.slot_pool);
                        return Err(Error::with_cleanup(
                            "resident admission slot generation",
                            error,
                            cleanup,
                        ));
                    }
                };
                (Some(slot), Some(next_page_slot))
            } else {
                (None, None)
            };

            if cache_eligible {
                let target_slot = page_slot.expect("cache eligibility requires a page manager");
                match self.prepare_prefix_admission(&mut prepared, target_slot) {
                    Ok(Some(prefix)) => {
                        debug_assert!(prefix.matched_tokens > 0);
                        if let Err(error) = self
                            .page_manager
                            .as_mut()
                            .expect("prefix preparation requires a page manager")
                            .publish_fork_prefix_snapshot(prefix.pages)
                        {
                            let mut cleanup = Vec::new();
                            if let Err(source) =
                                self.executor.release_sequence_state(prefix.model_state)
                            {
                                cleanup.push(CleanupStep::new("prefix model fork release", source));
                            }
                            if let Err(source) = self
                                .scheduler
                                .abort_waiting_admission(prepared, &mut self.slot_pool)
                            {
                                cleanup.push(CleanupStep::new(
                                    "prefix scheduler admission cleanup",
                                    source,
                                ));
                            }
                            return Err(Error::with_cleanup_batch(
                                "prefix page fork publish",
                                error,
                                cleanup,
                            ));
                        }
                        self.sequence_states.insert(session_id, prefix.model_state);
                        self.page_slots.insert(session_id, target_slot);
                        self.next_page_slot =
                            next_page_slot.expect("page-manager admission reserves the next slot");
                        self.prefix_cache_sessions.insert(session_id);
                        self.scheduler.publish_waiting_admission(prepared);
                        self.observability.prefix_cache.hits =
                            self.observability.prefix_cache.hits.saturating_add(1);
                        continue;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        let cleanup = self
                            .scheduler
                            .abort_waiting_admission(prepared, &mut self.slot_pool);
                        return Err(Error::with_cleanup(
                            "prefix cache admission",
                            error,
                            cleanup,
                        ));
                    }
                }
            }

            if let Some(slot) = page_slot
                && let Err(error) = self
                    .page_manager
                    .as_mut()
                    .expect("page-manager presence was checked")
                    .alloc_sequence(slot, 0)
            {
                let cleanup = self
                    .scheduler
                    .abort_waiting_admission(prepared, &mut self.slot_pool);
                return Err(Error::with_cleanup(
                    "resident admission logical KV preparation",
                    error,
                    cleanup,
                ));
            }
            if let Some(next_page_slot) = next_page_slot {
                self.next_page_slot = next_page_slot;
            }

            let state = match self.executor.create_sequence_state() {
                Ok(state) => state,
                Err(error) => {
                    let mut cleanup = Vec::new();
                    if let Some(slot) = page_slot {
                        let previous = self.page_slots.insert(session_id, slot);
                        debug_assert!(
                            previous.is_none(),
                            "fresh admission page slot must remain absent"
                        );
                        if let Err(source) = self.release_sequence_state(session_id) {
                            cleanup.push(CleanupStep::new(
                                format!("session {session_id:?} logical KV cleanup"),
                                source,
                            ));
                        }
                    }
                    if let Err(source) = self
                        .scheduler
                        .abort_waiting_admission(prepared, &mut self.slot_pool)
                    {
                        cleanup.push(CleanupStep::new(
                            format!("session {session_id:?} scheduler admission cleanup"),
                            source,
                        ));
                    }
                    return Err(Error::with_cleanup_batch(
                        "resident admission model preparation",
                        error,
                        cleanup,
                    ));
                }
            };

            let previous = self.sequence_states.insert(session_id, state);
            debug_assert!(
                previous.is_none(),
                "prepared admission model target must remain absent"
            );
            if let Some(slot) = page_slot {
                let previous = self.page_slots.insert(session_id, slot);
                debug_assert!(
                    previous.is_none(),
                    "prepared admission page target must remain absent"
                );
            }
            if cache_eligible {
                self.prefix_cache_sessions.insert(session_id);
            }
            self.scheduler.publish_waiting_admission(prepared);
            if cache_eligible {
                self.observability.prefix_cache.misses =
                    self.observability.prefix_cache.misses.saturating_add(1);
            }
        }
        Ok(())
    }

    fn capture_committed_prefill_prefix(&mut self, action: &PrefillChunkAction) -> Result<()> {
        if !self.prefix_cache_sessions.contains(&action.session_id) || self.has_live_transactions()
        {
            // Retained sessions, speculative work, and concurrent packed topology
            // are deliberately not cached until their ownership is independently
            // proven safe.
            return Ok(());
        }
        let Some(namespace) = self.prefix_cache_namespace() else {
            return Ok(());
        };
        let sequence = self
            .scheduler
            .active_sequence(action.session_id)
            .ok_or_else(|| Error::Invariant {
                message: format!(
                    "committed prefill session {:?} is no longer active",
                    action.session_id
                ),
            })?;
        let frontier = sequence.position;
        if frontier == 0
            || sequence.prompt_cursor != frontier
            || frontier > sequence.prompt_len
            || action.token_range.end != sequence.prompt_cursor
        {
            return Err(Error::Invariant {
                message: format!(
                    "committed prefix frontier mismatch for session {:?}: position={frontier} cursor={} prompt={} action_end={}",
                    action.session_id,
                    sequence.prompt_cursor,
                    sequence.prompt_len,
                    action.token_range.end
                ),
            });
        }
        let tokens = sequence.current_prompt_tokens()[..frontier].to_vec();
        let page_slot =
            *self
                .page_slots
                .get(&action.session_id)
                .ok_or_else(|| Error::Invariant {
                    message: format!(
                        "committed prefix session {:?} has no page slot",
                        action.session_id
                    ),
                })?;
        let cached_model = {
            let source = self
                .sequence_states
                .get(&action.session_id)
                .ok_or_else(|| Error::Invariant {
                    message: format!(
                        "committed prefix session {:?} has no model state",
                        action.session_id
                    ),
                })?;
            match self.executor.fork_sequence_state_from(source, frontier) {
                Ok(state) => state,
                Err(error) => {
                    tracing::warn!(
                        session_id = action.session_id.0,
                        frontier,
                        error = %error,
                        "committed prefix model capture bypassed"
                    );
                    return Ok(());
                }
            }
        };
        let snapshot = match self
            .page_manager
            .as_mut()
            .expect("prefix namespace requires a page manager")
            .capture_prefix_snapshot(page_slot, frontier)
        {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.pending_prefix_cleanups
                    .push_back(PendingPrefixCleanup::model_only(cached_model));
                let cleanup = self.progress_prefix_cleanups();
                if cleanup.is_ok() {
                    tracing::warn!(
                        session_id = action.session_id.0,
                        frontier,
                        error = %error,
                        "committed prefix KV snapshot capture bypassed"
                    );
                }
                return cleanup;
            }
        };
        let charge = snapshot.page_count();
        let payload = ResidentPrefixPayload {
            model_state: cached_model,
            snapshot,
        };
        match self
            .prefix_cache
            .insert(namespace, &tokens, payload, charge)
        {
            Ok(outcome) => {
                let (_, replaced, evicted) = outcome.into_parts();
                if let Some(replaced) = replaced {
                    self.queue_removed_prefix(replaced);
                }
                for entry in evicted {
                    self.queue_removed_prefix(entry);
                }
                self.progress_prefix_cleanups()
            }
            Err(error) => {
                let (kind, payload) = error.into_parts();
                self.queue_prefix_payload_cleanup(payload);
                let cleanup = self.progress_prefix_cleanups();
                if cleanup.is_ok() {
                    tracing::warn!(
                        session_id = action.session_id.0,
                        frontier,
                        error = %kind,
                        "committed prefix cache insertion bypassed"
                    );
                }
                cleanup
            }
        }
    }

    fn capture_committed_prefill_prefixes(&mut self, action: &SchedulerAction) -> Result<()> {
        match action {
            SchedulerAction::Execute { prefills, .. } => {
                for prefill in prefills {
                    self.capture_committed_prefill_prefix(prefill)?;
                }
            }
            SchedulerAction::PrefillChunk(prefill) => {
                self.capture_committed_prefill_prefix(prefill)?;
            }
            SchedulerAction::DecodeBatch(_)
            | SchedulerAction::Finish { .. }
            | SchedulerAction::Cancel { .. } => {}
        }
        Ok(())
    }

    /// Preserve a retained session at its committed position, or release a
    /// normal one immediately after the request turn finishes.
    fn finalize_sequence_state(&mut self, session_id: SessionId, position: usize) -> Result<()> {
        self.prefix_cache_sessions.remove(&session_id);
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
        self.prefix_cache_sessions.remove(&session_id);
        self.pending_sequence_cleanups
            .entry(session_id)
            .or_insert(PendingSequenceCleanup::Deferred);
        if self.has_live_transactions() {
            return Ok(());
        }
        self.progress_sequence_cleanup(session_id)
    }

    fn progress_pending_cleanups(&mut self) -> Result<()> {
        if self.has_live_transactions() {
            return Ok(());
        }
        while let Some(kv) = self.pending_resident_kv_aborts.pop_front() {
            if let Err((error, kv)) = self.progress_aborted_resident_kv(kv) {
                self.pending_resident_kv_aborts.push_front(kv);
                return Err(error);
            }
        }
        while let Some(retirement) = self.pending_kv_retirements.pop_front() {
            if let Err((error, retirement)) = self.progress_kv_retirement(retirement) {
                self.pending_kv_retirements.push_front(retirement);
                return Err(error);
            }
        }
        self.scheduler
            .retry_failed_slot_releases(&mut self.slot_pool)?;
        self.progress_prefix_cleanups()?;
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
        if self.has_live_transactions() {
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
                            return Err(Error::Invariant {
                                message: format!(
                                    "session {session_id:?} has a page slot without an authoritative page manager"
                                ),
                            });
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
            cleanup @ PendingSequenceCleanup::Suspended { .. }
            | cleanup @ PendingSequenceCleanup::Owned { .. } => cleanup,
        };
        let cleanup = if let PendingSequenceCleanup::Suspended {
            mut kv_state,
            mut retirement,
            model_state,
        } = cleanup
        {
            if let Some(state) = kv_state.take() {
                let Some(manager) = self.page_manager.as_mut() else {
                    self.pending_sequence_cleanups.insert(
                        session_id,
                        PendingSequenceCleanup::Suspended {
                            kv_state: Some(state),
                            retirement,
                            model_state,
                        },
                    );
                    return Err(Error::Invariant {
                        message: "suspended KV state has no authoritative page manager".into(),
                    });
                };
                match manager.release_preempted_pages(state) {
                    Ok(released) => {
                        retirement = Some(PendingKvRetirement::BackendRelease(released));
                    }
                    Err(failure) => {
                        let (error, state) = failure.into_parts();
                        self.pending_sequence_cleanups.insert(
                            session_id,
                            PendingSequenceCleanup::Suspended {
                                kv_state: Some(state),
                                retirement,
                                model_state,
                            },
                        );
                        return Err(error);
                    }
                }
            }
            PendingSequenceCleanup::Owned {
                retirement,
                model_state,
            }
        } else {
            cleanup
        };
        let PendingSequenceCleanup::Owned {
            retirement,
            model_state,
        } = cleanup
        else {
            unreachable!("sequence cleanup was converted to owned state")
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
        if let Some(state) = model_state
            && let Err(failure) = self.executor.try_release_sequence_state(state)
        {
            let (error, state) = failure.into_parts();
            self.pending_sequence_cleanups.insert(
                session_id,
                PendingSequenceCleanup::Owned {
                    retirement: None,
                    model_state: Some(state),
                },
            );
            return Err(error.into());
        }
        Ok(())
    }

    fn progress_kv_retirement(
        &mut self,
        retirement: PendingKvRetirement,
    ) -> std::result::Result<(), (Error, PendingKvRetirement)> {
        if self.has_live_transactions() {
            return Err((
                Error::InvalidRequest {
                    message: "cannot progress KV retirement while packed transactions are live"
                        .into(),
                },
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
                    return Err((
                        error.into(),
                        PendingKvRetirement::BackendRelease(retirement),
                    ));
                }
                retirement
            }
            PendingKvRetirement::LogicalConfirmation(retirement) => retirement,
        };
        let Some(manager) = self.page_manager.as_mut() else {
            return Err((
                Error::Invariant {
                    message: "retiring KV pages have no authoritative page manager".into(),
                },
                PendingKvRetirement::LogicalConfirmation(retirement),
            ));
        };
        let retired_pages = retirement.pages().to_vec();
        if let Some(untracked) = retired_pages
            .iter()
            .find(|page| !self.kv_page_grants.contains_key(page))
        {
            return Err((
                Error::Invariant {
                    message: format!(
                        "retiring KV page {} has no exact hard-credit grant",
                        untracked.0
                    ),
                },
                PendingKvRetirement::LogicalConfirmation(retirement),
            ));
        }
        match manager.confirm_page_retirement(retirement) {
            Ok(()) => {
                for page in retired_pages {
                    let mut grant = self
                        .kv_page_grants
                        .remove(&page)
                        .expect("retirement hard-credit ownership was preflighted");
                    self.load_registry
                        .release_hard_resources(&mut grant)
                        .expect("KV page hard grant matches its registry broker");
                }
                Ok(())
            }
            Err(error) => {
                let (error, retirement) = error.into_parts();
                Err((error, PendingKvRetirement::LogicalConfirmation(retirement)))
            }
        }
    }

    fn release_unbound_kv_page_grants(&mut self, grants: Vec<PhysicalResourceGrant>) {
        for mut grant in grants {
            self.load_registry
                .release_hard_resources(&mut grant)
                .expect("unbound KV page hard grant matches its registry broker");
        }
    }

    fn track_kv_page_grants(
        &mut self,
        physical_pages: Vec<KvPageId>,
        grants: Vec<PhysicalResourceGrant>,
    ) -> Result<()> {
        if physical_pages.len() != grants.len() {
            let page_count = physical_pages.len();
            let grant_count = grants.len();
            self.release_unbound_kv_page_grants(grants);
            return Err(Error::Invariant {
                message: format!(
                    "cannot track {grant_count} KV page hard grants for {page_count} physical pages"
                ),
            });
        }
        let mut unique = HashSet::with_capacity(physical_pages.len());
        if let Some(duplicate) = physical_pages
            .iter()
            .copied()
            .find(|page| !unique.insert(*page) || self.kv_page_grants.contains_key(page))
        {
            self.release_unbound_kv_page_grants(grants);
            return Err(Error::Invariant {
                message: format!(
                    "KV page {} acquired hard credit more than once",
                    duplicate.0
                ),
            });
        }
        for (page, grant) in physical_pages.into_iter().zip(grants) {
            let previous = self.kv_page_grants.insert(page, grant);
            debug_assert!(previous.is_none(), "KV grant preflight rejected duplicates");
        }
        Ok(())
    }

    fn reserve_batch_pages(
        &mut self,
        transaction: ExecutionTransactionId,
        demand: crate::scheduling::ResourceDemand,
        batch: &ScheduledBatch,
        execution_generations: &[u64],
    ) -> Result<Vec<KvReservation>> {
        if self.page_manager.is_none() {
            return Ok(Vec::new());
        }
        if execution_generations.len() != batch.sequences.len() {
            return Err(Error::Invariant {
                message: format!(
                    "KV execution generation count {} does not match scheduled sequence count {}",
                    execution_generations.len(),
                    batch.sequences.len()
                ),
            });
        }
        let mut reservations = Vec::with_capacity(batch.sequences.len());
        for ((scheduled, execution), execution_generation) in batch
            .sequences
            .iter()
            .zip(batch.execution().sequences())
            .zip(execution_generations)
        {
            let slot =
                *self
                    .page_slots
                    .get(&scheduled.session_id)
                    .ok_or_else(|| Error::Invariant {
                        message: format!(
                            "no page slot for active session {:?}",
                            scheduled.session_id
                        ),
                    })?;
            let token_count = usize::try_from(execution.query.end - execution.query.start)
                .map_err(|_| Error::InvalidRequest {
                    message: "query length exceeds usize".into(),
                })?;
            let page_generation = self
                .page_manager
                .as_ref()
                .expect("page manager presence checked above")
                .sequence_generation(slot)?;
            let required_before_eviction = self
                .page_manager
                .as_ref()
                .expect("page manager presence checked above")
                .required_physical_pages(slot, page_generation, token_count)?;
            self.evict_prefixes_for_kv_pages(required_before_eviction)?;
            let required = self
                .page_manager
                .as_ref()
                .expect("page manager presence checked above")
                .required_physical_pages(slot, page_generation, token_count)?;
            let mut grants = Vec::with_capacity(required);
            for _ in 0..required {
                match self.load_registry.acquire_hard_resources(
                    transaction.get(),
                    demand,
                    [PhysicalResourceClaim::new(ResourceKind::KvPage, 1)],
                ) {
                    Ok(grant) => grants.push(grant),
                    Err(error) => {
                        for mut grant in grants {
                            self.load_registry
                                .release_hard_resources(&mut grant)
                                .expect("unsubmitted KV page grant is releasable");
                        }
                        let cleanup = self
                            .abort_quiesced_resident_kv(PendingResidentKv::Reserved(reservations));
                        return Err(Error::with_cleanup("KV hard admission", error, cleanup));
                    }
                }
            }

            let mut reservation = match self
                .page_manager
                .as_mut()
                .expect("page manager presence checked above")
                .reserve(slot, page_generation, token_count)
            {
                Ok(reservation) => reservation,
                Err(error) => {
                    for mut grant in grants {
                        self.load_registry
                            .release_hard_resources(&mut grant)
                            .expect("unsubmitted KV page grant is releasable");
                    }
                    let cleanup =
                        self.abort_quiesced_resident_kv(PendingResidentKv::Reserved(reservations));
                    return Err(Error::with_cleanup("KV reservation", error, cleanup));
                }
            };
            if let Err(error) = self
                .page_manager
                .as_mut()
                .expect("page manager presence checked above")
                .bind_reservation_execution(
                    &mut reservation,
                    execution.state_slot,
                    *execution_generation,
                )
            {
                for mut grant in grants {
                    self.load_registry
                        .release_hard_resources(&mut grant)
                        .expect("unsubmitted KV page grant is releasable");
                }
                let mut cleanup_reservations = reservations;
                cleanup_reservations.push(reservation);
                let cleanup = self
                    .abort_quiesced_resident_kv(PendingResidentKv::Reserved(cleanup_reservations));
                return Err(Error::with_cleanup(
                    "KV execution-generation binding",
                    error,
                    cleanup,
                ));
            }
            let mut physical_pages = reservation.view().newly_allocated.clone();
            if let Some(cow) = reservation.view().cow_replacement {
                physical_pages.push(cow.replacement);
            }
            if physical_pages.len() != grants.len() {
                for mut grant in grants {
                    self.load_registry
                        .release_hard_resources(&mut grant)
                        .expect("unsubmitted KV page grant is releasable");
                }
                let mut cleanup_reservations = reservations;
                cleanup_reservations.push(reservation);
                self.abort_quiesced_resident_kv(PendingResidentKv::Reserved(cleanup_reservations))?;
                return Err(Error::Invariant {
                    message: format!(
                        "KV page manager reserved {} physical pages after exact hard admission for {}",
                        physical_pages.len(),
                        required
                    ),
                });
            }
            if let Err(error) = self.track_kv_page_grants(physical_pages, grants) {
                let mut cleanup_reservations = reservations;
                cleanup_reservations.push(reservation);
                let cleanup = self
                    .abort_quiesced_resident_kv(PendingResidentKv::Reserved(cleanup_reservations));
                return Err(Error::with_cleanup(
                    "KV hard-credit tracking",
                    error,
                    cleanup,
                ));
            }
            reservations.push(reservation);
        }
        Ok(reservations)
    }

    fn reserve_speculative_pages(
        &mut self,
        transaction: ExecutionTransactionId,
        items: &[SpeculativeVerificationItem<'_>],
    ) -> Result<Vec<KvReservation>> {
        let mut reservations = Vec::with_capacity(items.len());
        for item in items {
            let token_count =
                item.proposal
                    .len()
                    .checked_add(1)
                    .ok_or_else(|| Error::InvalidRequest {
                        message: "speculative verification row count overflow".into(),
                    })?;
            let required = self
                .page_manager
                .as_ref()
                .ok_or_else(|| Error::InvalidRequest {
                    message: "speculative KV reservation requires an authoritative page manager"
                        .into(),
                })?
                .required_physical_pages(item.state_slot, item.generation, token_count)?;
            // Speculative proposal/verification owns model topology outside the
            // ordinary resident reserve path, so prefix eviction is deliberately
            // bypassed here until that ownership can be quiesced independently.
            let mut grants = Vec::with_capacity(required);
            for _ in 0..required {
                match self.load_registry.acquire_hard_resources(
                    transaction.get(),
                    crate::scheduling::ResourceDemand::required(
                        ExecutionPhase::SpeculativeVerification,
                    ),
                    [PhysicalResourceClaim::new(ResourceKind::KvPage, 1)],
                ) {
                    Ok(grant) => grants.push(grant),
                    Err(error) => {
                        for mut grant in grants {
                            self.load_registry
                                .release_hard_resources(&mut grant)
                                .expect("unsubmitted speculative KV page grant is releasable");
                        }
                        self.abort_quiesced_resident_kv(PendingResidentKv::Reserved(reservations))?;
                        return Err(error.into());
                    }
                }
            }

            let reservation = match self
                .page_manager
                .as_mut()
                .expect("page manager presence checked above")
                .reserve(item.state_slot, item.generation, token_count)
            {
                Ok(reservation) => reservation,
                Err(error) => {
                    for mut grant in grants {
                        self.load_registry
                            .release_hard_resources(&mut grant)
                            .expect("unsubmitted speculative KV page grant is releasable");
                    }
                    self.abort_quiesced_resident_kv(PendingResidentKv::Reserved(reservations))?;
                    return Err(error);
                }
            };
            let mut physical_pages = reservation.view().newly_allocated.clone();
            if let Some(cow) = reservation.view().cow_replacement {
                physical_pages.push(cow.replacement);
            }
            if physical_pages.len() != grants.len() {
                for mut grant in grants {
                    self.load_registry
                        .release_hard_resources(&mut grant)
                        .expect("unsubmitted speculative KV page grant is releasable");
                }
                reservations.push(reservation);
                self.abort_quiesced_resident_kv(PendingResidentKv::Reserved(reservations))?;
                return Err(Error::Invariant {
                    message: format!(
                        "speculative KV manager reserved {} physical pages after exact hard admission for {required}",
                        physical_pages.len()
                    ),
                });
            }
            if let Err(error) = self.track_kv_page_grants(physical_pages, grants) {
                reservations.push(reservation);
                let cleanup =
                    self.abort_quiesced_resident_kv(PendingResidentKv::Reserved(reservations));
                return Err(Error::with_cleanup(
                    "speculative KV hard-credit tracking",
                    error,
                    cleanup,
                ));
            }
            reservations.push(reservation);
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
        let manager = self
            .page_manager
            .as_ref()
            .ok_or_else(|| Error::InvalidRequest {
                message: "paged executor requires a runtime KvPageManager".into(),
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
                PendingResidentKv::Reserved(_)
                | PendingResidentKv::Prepared(_)
                | PendingResidentKv::Retiring(_) => Err(Error::Invariant {
                    message: "KV transaction exists without an authoritative page manager".into(),
                }),
            };
        };
        let retirement = match kv {
            PendingResidentKv::Reserved(reservations) => {
                match manager.abort_reservations(reservations) {
                    Ok(retirement) => retirement,
                    Err(error) => {
                        let (error, reservations) = error.into_parts();
                        self.pending_resident_kv_aborts
                            .push_back(PendingResidentKv::Reserved(reservations));
                        return Err(error);
                    }
                }
            }
            PendingResidentKv::Prepared(prepared) => manager.abort_prepared_commit(prepared),
            PendingResidentKv::Retiring(retirement) => {
                return match self.progress_kv_retirement(retirement) {
                    Ok(()) => Ok(()),
                    Err((error, retirement)) => {
                        self.pending_kv_retirements.push_back(retirement);
                        Err(error)
                    }
                };
            }
        };
        self.release_and_confirm_retirement(retirement)
    }

    /// Retires every page owned by one sequence slot and confirms the retirement.
    pub fn retire_sequence_pages(
        &mut self,
        state_slot: ferrule_common::execution::StateSlot,
    ) -> Result<()> {
        let retirement = self
            .page_manager
            .as_mut()
            .ok_or_else(|| Error::InvalidRequest {
                message: "the authoritative KV page manager is not installed".into(),
            })?
            .free_sequence_pages(state_slot)?;
        self.release_and_confirm_retirement(retirement)
    }

    fn release_and_confirm_retirement(&mut self, retirement: KvRetirement) -> Result<()> {
        let retirement = PendingKvRetirement::BackendRelease(retirement);
        if self.has_live_transactions() {
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
        self.admit_new_sequences()
    }

    fn no_action_step(&self) -> ResidentDriverStep {
        if self.scheduler.is_idle() && !self.warmup_pending() {
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

        let resource_demand = Self::action_demand(&action);
        let execution_generations = states
            .iter()
            .map(|state| self.executor.runner().sequence_generation(state))
            .collect::<Vec<_>>();
        let mut page_reservations = match self.reserve_batch_pages(
            transaction,
            resource_demand,
            &scheduled,
            &execution_generations,
        ) {
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
            let error = self.abort_action(&action, error, false, "KV binding");
            return Err(Error::with_cleanup("KV binding", error, rollback));
        }
        let reservation_views = match &self.page_manager {
            Some(manager) => manager.reservation_views(&page_reservations),
            None if page_reservations.is_empty() => Ok(Vec::<KvReservationView>::new()),
            None => Err(Error::Invariant {
                message: "KV reservations exist without an authoritative page manager".into(),
            }),
        };
        let reservation_views = match reservation_views {
            Ok(views) => views,
            Err(error) => {
                let cleanup =
                    self.abort_quiesced_resident_kv(PendingResidentKv::Reserved(page_reservations));
                self.restore_transaction_sessions(schedules, states)?;
                let error = self.abort_action(&action, error, false, "KV reservation view");
                return Err(Error::with_cleanup("KV reservation view", error, cleanup));
            }
        };

        if let Err(error) =
            self.declare_transaction_prefetch(transaction, resource_demand, scheduled.execution())
        {
            let now_ns = self.runtime_now_ns();
            let materialization_cleanup = self
                .load_registry
                .finish_transaction_custody(
                    transaction,
                    TransactionCustodyOutcome::RolledBack,
                    now_ns,
                )
                .map_err(Error::from);
            let kv_cleanup =
                self.abort_quiesced_resident_kv(PendingResidentKv::Reserved(page_reservations));
            self.restore_transaction_sessions(schedules, states)?;
            let error = self.abort_action(&action, error, false, "transaction prefetch");
            let error = Error::with_cleanup(
                "transaction prefetch materialization",
                error,
                materialization_cleanup,
            );
            return Err(Error::with_cleanup(
                "transaction prefetch KV",
                error,
                kv_cleanup,
            ));
        }

        let execution_started_ns = self.runtime_now_ns();
        if let Err(error) = self.executor.prepare_batch_with_kv(
            transaction,
            &mut states,
            scheduled.execution(),
            &reservation_views,
        ) {
            let now_ns = self.runtime_now_ns();
            let materialization_cleanup = self
                .load_registry
                .finish_transaction_custody(
                    transaction,
                    TransactionCustodyOutcome::RolledBack,
                    now_ns,
                )
                .map_err(Error::from);
            let kv_cleanup =
                self.abort_quiesced_resident_kv(PendingResidentKv::Reserved(page_reservations));
            self.restore_transaction_sessions(schedules, states)?;
            let error = self.abort_action(&action, error, false, "backend prepare");
            let error = Error::with_cleanup(
                "backend prepare materialization",
                error,
                materialization_cleanup,
            );
            return Err(Error::with_cleanup("backend prepare KV", error, kv_cleanup));
        }
        let progress =
            self.executor
                .execute_prepared_batch(transaction, &mut states, scheduled.execution());
        let mut pending = PendingResidentBatch {
            transaction,
            action,
            scheduled,
            kv: Some(PendingResidentKv::Reserved(page_reservations)),
            states,
            schedules,
            phase: ResidentTransactionPhase::Executing(None),
        };
        if let Err(error) = self.record_runnable_work_span(execution_started_ns) {
            return self.abort_failed_resident(pending, error, "model execution observation");
        }
        match progress {
            Ok(MultiSessionBatchProgress::Complete(output)) => {
                self.finish_resident_transaction(pending, output, on_token)
            }
            Ok(MultiSessionBatchProgress::Waiting(progress)) => {
                if let Err(error) = self.register_pending_progress(&progress, resource_demand) {
                    return self.abort_failed_resident(
                        pending,
                        error,
                        "model continuation registration",
                    );
                }
                pending.phase = ResidentTransactionPhase::Executing(Some(progress));
                self.resident_transactions.insert(transaction, pending);
                Ok(ResidentDriverStep::WaitingForModelProgress(
                    self.pending_model_progresses(),
                ))
            }
            Err(error) => self.abort_failed_resident(pending, error, "model execution"),
        }
    }

    fn resume_resident_transaction<F>(
        &mut self,
        transaction: ExecutionTransactionId,
        ready_continuation: ContinuationId,
        on_token: &mut F,
    ) -> Result<Option<ResidentDriverStep>>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let mut pending = self
            .resident_transactions
            .remove(&transaction)
            .ok_or_else(|| Error::Invariant {
                message: format!("resident transaction {transaction:?} disappeared"),
            })?;
        let owned_continuation = pending
            .phase
            .pending_progress()
            .map(PendingModelProgress::continuation);
        let continuation = match owned_continuation {
            Some(continuation) if continuation == ready_continuation => continuation,
            Some(continuation) => {
                let error = Error::InvalidRequest {
                    message: format!(
                        "resident transaction {transaction:?} owns continuation {}, not ready continuation {}",
                        continuation.get(),
                        ready_continuation.get()
                    ),
                };
                self.resident_transactions.insert(transaction, pending);
                return Err(error);
            }
            None => {
                self.resident_transactions.insert(transaction, pending);
                return Err(Error::InvalidRequest {
                    message: format!("resident transaction {transaction:?} is quarantined"),
                });
            }
        };
        let mut resume_lease = match self.prepare_resume_lease(continuation) {
            Ok(lease) => lease,
            Err(error) => {
                self.resident_transactions.insert(transaction, pending);
                return Err(error);
            }
        };
        let leases = match resume_lease.take() {
            Ok(leases) => leases,
            Err(error) => {
                self.resident_transactions.insert(transaction, pending);
                return Err(error.into());
            }
        };
        let resume_started = self.runtime_now_ns();
        let progress = self.executor.resume_prepared_batch(
            transaction,
            &mut pending.states,
            pending.scheduled.execution(),
            continuation,
            leases,
        );
        let disposition = if progress.is_err() {
            ResumeDisposition::StillActive
        } else {
            ResumeDisposition::Consumed
        };
        if let Err(error) =
            self.finish_resume_lease(continuation, resume_lease, disposition, resume_started)
        {
            return self
                .abort_failed_resident(pending, error, "model resume lease completion")
                .map(Some);
        }
        if let Err(error) = self.record_runnable_work_span(resume_started) {
            return self
                .abort_failed_resident(pending, error, "model resume observation")
                .map(Some);
        }
        match progress {
            Ok(MultiSessionBatchProgress::Complete(output)) => {
                pending.phase = ResidentTransactionPhase::Executing(None);
                self.finish_resident_transaction(pending, output, on_token)
                    .map(Some)
            }
            Ok(MultiSessionBatchProgress::Waiting(progress)) => {
                pending.phase = ResidentTransactionPhase::Executing(None);
                if let Err(error) =
                    self.register_pending_progress(&progress, Self::action_demand(&pending.action))
                {
                    return self
                        .abort_failed_resident(pending, error, "model continuation registration")
                        .map(Some);
                }
                pending.phase = ResidentTransactionPhase::Executing(Some(progress));
                self.resident_transactions.insert(transaction, pending);
                Ok(None)
            }
            Err(error) => self
                .abort_failed_resident(pending, error, "resumable model execution")
                .map(Some),
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
            return self.abort_failed_resident(pending, error, "model output contract");
        }

        let reserved = match pending
            .kv
            .take()
            .expect("executing resident transaction owns logical KV state")
        {
            PendingResidentKv::Reserved(reservations) => reservations,
            PendingResidentKv::Prepared(prepared) => {
                pending.kv = Some(PendingResidentKv::Prepared(prepared));
                self.resident_transactions.insert(transaction, pending);
                return Err(Error::Invariant {
                    message: format!(
                        "transaction {transaction:?} reached model completion with an existing prepared logical commit"
                    ),
                });
            }
            PendingResidentKv::Retiring(retirement) => {
                pending.kv = Some(PendingResidentKv::Retiring(retirement));
                self.resident_transactions.insert(transaction, pending);
                return Err(Error::Invariant {
                    message: format!(
                        "transaction {transaction:?} reached model completion while logical KV retirement was already active"
                    ),
                });
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
                        pending.kv = Some(PendingResidentKv::Reserved(
                            commits
                                .into_iter()
                                .map(|commit| commit.reservation)
                                .collect(),
                        ));
                        return self.abort_failed_resident(pending, error, "logical KV prepare");
                    }
                }
            }
            None if reserved.is_empty() => None,
            None => {
                pending.kv = Some(PendingResidentKv::Reserved(reserved));
                return self.abort_failed_resident(
                    pending,
                    Error::Invariant {
                        message: "KV reservations exist without an authoritative page manager"
                            .into(),
                    },
                    "logical KV prepare",
                );
            }
        };
        if let Some(prepared) = prepared {
            pending.kv = Some(PendingResidentKv::Prepared(prepared));
        }

        pending.phase = ResidentTransactionPhase::Publishing {
            output,
            started_ns: self.runtime_now_ns(),
            cancellation_request: None,
        };
        self.drive_resident_ending(pending, on_token)
    }

    fn publish_resident_transaction<F>(
        &mut self,
        mut pending: PendingResidentBatch<R::SequenceState>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transaction = pending.transaction;
        let ResidentTransactionPhase::BackendCommittedPendingPublish {
            output,
            mut custody,
            cancellation_request,
        } = std::mem::replace(
            &mut pending.phase,
            ResidentTransactionPhase::Executing(None),
        )
        else {
            unreachable!("resident publish progress requires a backend-committed phase")
        };

        if let Some(outcome) = custody
            && let Err(error) = self.load_registry.finish_transaction_custody(
                transaction,
                outcome,
                self.runtime_now_ns(),
            )
        {
            custody = Some(outcome);
            pending.phase = ResidentTransactionPhase::BackendCommittedPendingPublish {
                output,
                custody,
                cancellation_request,
            };
            return self.retain_resident_ending_error(pending, error.into());
        }

        if let Some(kv) = pending.kv.take()
            && let Err((error, kv)) = self.progress_committed_resident_kv(kv)
        {
            pending.kv = Some(kv);
            pending.phase = ResidentTransactionPhase::BackendCommittedPendingPublish {
                output,
                custody: None,
                cancellation_request,
            };
            return self.retain_resident_ending_error(pending, error);
        }

        if let Err(error) =
            self.progress_transaction_session_restore(&mut pending.schedules, &mut pending.states)
        {
            pending.phase = ResidentTransactionPhase::BackendCommittedPendingPublish {
                output,
                custody: None,
                cancellation_request,
            };
            return self.retain_resident_ending_error(pending, error);
        }

        let step = match self.publish_restored_resident_transaction(pending, output) {
            Ok(step) => step,
            Err(error) => {
                let cleanup = self
                    .finish_transaction_cancellations(transaction, cancellation_request)
                    .map(|_| ());
                return Err(Error::with_cleanup(
                    "committed resident cancellation",
                    error,
                    cleanup,
                ));
            }
        };
        self.finish_transaction_cancellations(transaction, cancellation_request)?;
        self.flush_committed_token_outbox(on_token)?;
        Ok(step)
    }

    fn progress_committed_resident_kv(
        &mut self,
        kv: PendingResidentKv,
    ) -> std::result::Result<(), (Error, PendingResidentKv)> {
        let kv = match kv {
            PendingResidentKv::Prepared(prepared) => {
                let Some(manager) = self.page_manager.as_mut() else {
                    return Err((
                        Error::Invariant {
                            message: "prepared logical commit has no authoritative page manager"
                                .into(),
                        },
                        PendingResidentKv::Prepared(prepared),
                    ));
                };
                PendingResidentKv::Retiring(PendingKvRetirement::BackendRelease(
                    manager.publish_commit(prepared),
                ))
            }
            PendingResidentKv::Reserved(reservations) if reservations.is_empty() => return Ok(()),
            PendingResidentKv::Reserved(reservations) => {
                return Err((
                    Error::Invariant {
                        message: "backend committed while logical reservations remained unprepared"
                            .into(),
                    },
                    PendingResidentKv::Reserved(reservations),
                ));
            }
            kv @ PendingResidentKv::Retiring(_) => kv,
        };
        let PendingResidentKv::Retiring(retirement) = kv else {
            unreachable!("committed logical KV was converted to retirement")
        };
        self.progress_or_defer_resident_retirement(retirement)
            .map_err(|(error, retirement)| (error, PendingResidentKv::Retiring(retirement)))
    }

    fn progress_or_defer_resident_retirement(
        &mut self,
        retirement: PendingKvRetirement,
    ) -> std::result::Result<(), (Error, PendingKvRetirement)> {
        if self.has_live_transactions() {
            self.pending_kv_retirements.push_back(retirement);
            Ok(())
        } else {
            self.progress_kv_retirement(retirement)
        }
    }

    fn publish_restored_resident_transaction(
        &mut self,
        pending: PendingResidentBatch<R::SequenceState>,
        output: ExecutionOutput,
    ) -> Result<ResidentDriverStep> {
        let transaction = pending.transaction;
        debug_assert!(pending.kv.is_none());
        debug_assert!(pending.states.is_empty());
        debug_assert!(pending.schedules.is_empty());
        let action_kind = action_kind(&pending.action);
        let rows = action_rows(&pending.action);
        let publication = (|| -> Result<(usize, usize)> {
            self.scheduler.commit_action(&pending.action)?;
            self.capture_committed_prefill_prefixes(&pending.action)?;
            self.observability.stats.actions += 1;
            let externally_committed_tokens = match &pending.action {
                SchedulerAction::Execute { prefills, decodes } => {
                    self.observability.stats.prefill_chunks += prefills.len();
                    self.observability.stats.prefill_tokens += prefills
                        .iter()
                        .map(|action| action.token_range.len())
                        .sum::<usize>();
                    self.observability.stats.decode_steps += decodes.len();
                    self.enqueue_committed_decode_tokens(decodes)?
                }
                SchedulerAction::PrefillChunk(prefill) => {
                    self.observability.stats.prefill_chunks += 1;
                    self.observability.stats.prefill_tokens += prefill.token_range.len();
                    0
                }
                SchedulerAction::DecodeBatch(actions) => {
                    self.observability.stats.decode_steps += actions.len();
                    self.enqueue_committed_decode_tokens(actions)?
                }
                SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => 0,
            };
            let expected_external = action_decode_actions(&pending.action).len();
            if externally_committed_tokens != expected_external {
                return Err(Error::Invariant {
                    message: format!(
                        "resident transaction committed {expected_external} external tokens but queued {externally_committed_tokens}"
                    ),
                });
            }
            self.snapshot_transaction_outputs(transaction, externally_committed_tokens)?;

            let action_finish = self.finish_after_decode_action(&pending.action)?;
            let mut finished = action_finish.finished;
            let output_outcome = self.apply_execution_output(
                &pending.scheduled,
                &output,
                &action_finish.session_ids,
            )?;
            finished += output_outcome.finished;
            Ok((output_outcome.staged, finished))
        })();
        let (staged, finished) = match publication {
            Ok(outcome) => outcome,
            Err(error) => {
                return Err(self.abort_action(
                    &pending.action,
                    error,
                    false,
                    "committed resident publication",
                ));
            }
        };
        Ok(ResidentDriverStep::Executed {
            action_kind,
            rows,
            staged,
            finished,
        })
    }

    fn drive_speculative_endings<F>(
        &mut self,
        on_token: &mut F,
    ) -> Result<Option<ResidentDriverStep>>
    where
        R: ResidentModelRunner,
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transactions = self
            .speculative_transactions
            .iter()
            .filter_map(|(transaction, pending)| match pending {
                PendingSpeculativeDriverCohort::Proposing(pending)
                    if pending.cancellation_request.is_some() || pending.abort_cause.is_some() =>
                {
                    Some(*transaction)
                }
                PendingSpeculativeDriverCohort::Ending(_) => Some(*transaction),
                PendingSpeculativeDriverCohort::Proposing(_)
                | PendingSpeculativeDriverCohort::Verifying(_) => None,
            })
            .collect::<Vec<_>>();
        for transaction in transactions {
            let pending = self
                .speculative_transactions
                .remove(&transaction)
                .expect("ending speculative transaction was collected above");
            match pending {
                PendingSpeculativeDriverCohort::Proposing(pending) => {
                    let cancellation_request = pending.cancellation_request;
                    match self.cancel_native_proposal_cohort(*pending, None, None) {
                        Ok(TransactionEndProgress::Pending) => {}
                        Ok(TransactionEndProgress::Complete) => {
                            self.finish_transaction_cancellations(
                                transaction,
                                cancellation_request,
                            )?;
                            return Ok(Some(ResidentDriverStep::Executed {
                                action_kind: ResidentActionKind::Cancel,
                                rows: 0,
                                staged: 0,
                                finished: 0,
                            }));
                        }
                        Err(error) => return Err(error),
                    }
                }
                PendingSpeculativeDriverCohort::Ending(pending) => {
                    let step = self.drive_speculative_ending(*pending, on_token)?;
                    if !matches!(step, ResidentDriverStep::Blocked) {
                        return Ok(Some(step));
                    }
                }
                PendingSpeculativeDriverCohort::Verifying(_) => {
                    unreachable!("collected speculative transaction must be ending")
                }
            }
        }
        Ok(None)
    }

    fn drive_speculative_ending<F>(
        &mut self,
        mut pending: PendingSpeculativeEndingDriverCohort<R::SequenceState>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        R: ResidentModelRunner,
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transaction = pending.transaction;
        let terminal = match &mut pending.ending {
            SpeculativeEnding::Publish(prepared) => Some(self.executor.end_transaction(
                transaction,
                prepared.transaction_mut().states_mut(),
                TransactionEndIntent::Publish,
            )),
            SpeculativeEnding::Abort { transaction, .. } => Some(self.executor.end_transaction(
                transaction.id(),
                transaction.states_mut(),
                TransactionEndIntent::Abort,
            )),
            SpeculativeEnding::BackendCommittedPendingPublish { .. }
            | SpeculativeEnding::BackendAbortedPendingCleanup { .. } => None,
        };
        if let Some(terminal) = terminal {
            match terminal {
                Ok(TransactionEndProgress::Pending) => {
                    self.speculative_transactions.insert(
                        transaction,
                        PendingSpeculativeDriverCohort::Ending(Box::new(pending)),
                    );
                    return Ok(ResidentDriverStep::Blocked);
                }
                Err(error) => {
                    self.speculative_transactions.insert(
                        transaction,
                        PendingSpeculativeDriverCohort::Ending(Box::new(pending)),
                    );
                    return Err(error);
                }
                Ok(TransactionEndProgress::Complete) => {
                    pending.ending = match pending.ending {
                        SpeculativeEnding::Publish(prepared) => {
                            SpeculativeEnding::BackendCommittedPendingPublish {
                                progress: QuiescedSpeculativePublishProgress::from_prepared(
                                    prepared,
                                ),
                                retirements: VecDeque::new(),
                                custody: Some(TransactionCustodyOutcome::Committed {
                                    started_ns: pending.started_ns,
                                    finished_ns: self.runtime_now_ns(),
                                }),
                            }
                        }
                        SpeculativeEnding::Abort {
                            transaction,
                            failure,
                        } => SpeculativeEnding::BackendAbortedPendingCleanup {
                            progress: QuiescedSpeculativeAbortProgress::from_transaction(
                                transaction,
                            ),
                            retirements: VecDeque::new(),
                            custody: Some(if pending.request_id.is_some() {
                                TransactionCustodyOutcome::Cancelled
                            } else {
                                TransactionCustodyOutcome::RolledBack
                            }),
                            failure,
                            decode_requeued: false,
                        },
                        SpeculativeEnding::BackendCommittedPendingPublish { .. }
                        | SpeculativeEnding::BackendAbortedPendingCleanup { .. } => {
                            unreachable!("backend terminalization matched a pre-terminal ending")
                        }
                    };
                }
            }
        }

        match pending.ending {
            SpeculativeEnding::BackendCommittedPendingPublish { .. } => {
                self.publish_quiesced_speculative_ending(pending, on_token)
            }
            SpeculativeEnding::BackendAbortedPendingCleanup { .. } => {
                self.cleanup_quiesced_speculative_ending(pending)
            }
            SpeculativeEnding::Publish(_) | SpeculativeEnding::Abort { .. } => {
                unreachable!("completed speculative ending was converted to post-terminal progress")
            }
        }
    }

    fn publish_quiesced_speculative_ending<F>(
        &mut self,
        mut pending: PendingSpeculativeEndingDriverCohort<R::SequenceState>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        R: ResidentModelRunner,
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transaction = pending.transaction;
        let mut newly_retired = Vec::new();
        let progress_result = {
            let SpeculativeEnding::BackendCommittedPendingPublish { progress, .. } =
                &mut pending.ending
            else {
                unreachable!("speculative publish requires post-terminal publish progress")
            };
            publish_quiesced_speculative_cohort(
                &mut self.executor,
                self.page_manager
                    .as_mut()
                    .expect("speculative transaction retains its page manager"),
                &mut pending.source_states,
                progress,
                &mut newly_retired,
            )
        };
        if let SpeculativeEnding::BackendCommittedPendingPublish { retirements, .. } =
            &mut pending.ending
        {
            retirements.extend(
                newly_retired
                    .into_iter()
                    .map(PendingKvRetirement::BackendRelease),
            );
        }
        if let Err(error) = progress_result {
            return self.retain_speculative_ending_error(pending, error);
        }
        if let Err(error) = self.progress_speculative_ending_retirements(&mut pending) {
            return self.retain_speculative_ending_error(pending, error);
        }

        let custody = match &mut pending.ending {
            SpeculativeEnding::BackendCommittedPendingPublish { custody, .. } => custody.take(),
            _ => unreachable!("speculative publish retained its post-terminal phase"),
        };
        if let Some(outcome) = custody
            && let Err(error) = self.finish_speculative_transaction_custody(transaction, outcome)
        {
            if let SpeculativeEnding::BackendCommittedPendingPublish { custody, .. } =
                &mut pending.ending
            {
                *custody = Some(outcome);
            }
            return self.retain_speculative_ending_error(pending, error);
        }
        if let Some(continuation) = pending.continuation
            && let Err(error) = self
                .detach_registered_continuation(continuation, CancellationReason::ExternalRequest)
        {
            return self.retain_speculative_ending_error(pending, error);
        }
        pending.continuation = None;
        if let Err(error) = self.progress_transaction_session_restore(
            &mut pending.schedules,
            &mut pending.source_states,
        ) {
            return self.retain_speculative_ending_error(pending, error);
        }
        let cohort = match &mut pending.ending {
            SpeculativeEnding::BackendCommittedPendingPublish { progress, .. } => progress
                .take_result()
                .expect("completed speculative publish retains its result"),
            _ => unreachable!("speculative publish retained its post-terminal phase"),
        };
        let cleanup_actions = pending.actions.clone();
        let cancellation_request = pending.request_id;
        let step = match self.publish_speculative_decode_cohort(
            transaction,
            pending.cohort_start,
            pending.actions,
            pending.prepared,
            cohort,
        ) {
            Ok(step) => step,
            Err(error) => {
                let error = self.abort_speculative_decode_batch(
                    &cleanup_actions,
                    error,
                    "committed speculative publication",
                );
                let cleanup = self
                    .finish_transaction_cancellations(transaction, cancellation_request)
                    .map(|_| ());
                return Err(Error::with_cleanup(
                    "committed speculative cancellation",
                    error,
                    cleanup,
                ));
            }
        };
        self.finish_transaction_cancellations(transaction, cancellation_request)?;
        self.flush_committed_token_outbox(on_token)?;
        Ok(step)
    }

    fn cleanup_quiesced_speculative_ending(
        &mut self,
        mut pending: PendingSpeculativeEndingDriverCohort<R::SequenceState>,
    ) -> Result<ResidentDriverStep>
    where
        R: ResidentModelRunner,
    {
        let transaction = pending.transaction;
        let mut newly_retired = Vec::new();
        let progress_result = {
            let SpeculativeEnding::BackendAbortedPendingCleanup { progress, .. } =
                &mut pending.ending
            else {
                unreachable!("speculative abort requires post-terminal abort progress")
            };
            abort_quiesced_speculative_transaction(
                &mut self.executor,
                self.page_manager
                    .as_mut()
                    .expect("speculative transaction retains its page manager"),
                progress,
                &mut newly_retired,
            )
        };
        if let SpeculativeEnding::BackendAbortedPendingCleanup { retirements, .. } =
            &mut pending.ending
        {
            retirements.extend(
                newly_retired
                    .into_iter()
                    .map(PendingKvRetirement::BackendRelease),
            );
        }
        if let Err(error) = progress_result {
            return self.retain_speculative_ending_error(pending, error);
        }
        if let Err(error) = self.progress_speculative_ending_retirements(&mut pending) {
            return self.retain_speculative_ending_error(pending, error);
        }

        let custody = match &mut pending.ending {
            SpeculativeEnding::BackendAbortedPendingCleanup { custody, .. } => custody.take(),
            _ => unreachable!("speculative abort retained its post-terminal phase"),
        };
        if let Some(outcome) = custody
            && let Err(error) = self.finish_speculative_transaction_custody(transaction, outcome)
        {
            if let SpeculativeEnding::BackendAbortedPendingCleanup { custody, .. } =
                &mut pending.ending
            {
                *custody = Some(outcome);
            }
            return self.retain_speculative_ending_error(pending, error);
        }
        if let Some(continuation) = pending.continuation
            && let Err(error) = self
                .detach_registered_continuation(continuation, CancellationReason::ExternalRequest)
        {
            return self.retain_speculative_ending_error(pending, error);
        }
        pending.continuation = None;
        if let Err(error) = self.progress_transaction_session_restore(
            &mut pending.schedules,
            &mut pending.source_states,
        ) {
            return self.retain_speculative_ending_error(pending, error);
        }

        let (failure, mut decode_requeued) = match &mut pending.ending {
            SpeculativeEnding::BackendAbortedPendingCleanup {
                failure,
                decode_requeued,
                ..
            } => (failure.take(), *decode_requeued),
            _ => unreachable!("speculative abort retained its post-terminal phase"),
        };
        if let Some((error, stage)) = failure {
            let error = self.abort_speculative_decode_batch(&pending.actions, error, stage);
            let cleanup = self
                .finish_transaction_cancellations(transaction, pending.request_id)
                .map(|_| ());
            return Err(Error::with_cleanup(
                "speculative abort cancellation",
                error,
                cleanup,
            ));
        }
        if !decode_requeued {
            if let Err(error) = self
                .scheduler
                .requeue_decode_actions_front(&pending.actions)
            {
                return self.retain_speculative_ending_error(pending, error);
            }
            decode_requeued = true;
            if let SpeculativeEnding::BackendAbortedPendingCleanup {
                decode_requeued: retained,
                ..
            } = &mut pending.ending
            {
                *retained = true;
            }
        }
        let requested = pending.request_id;
        if let Err(error) = self.finish_transaction_cancellations(transaction, requested) {
            debug_assert!(decode_requeued);
            return self.retain_speculative_ending_error(pending, error);
        }
        Ok(ResidentDriverStep::Executed {
            action_kind: ResidentActionKind::Cancel,
            rows: 0,
            staged: 0,
            finished: 0,
        })
    }

    fn progress_speculative_ending_retirements(
        &mut self,
        pending: &mut PendingSpeculativeEndingDriverCohort<R::SequenceState>,
    ) -> Result<()> {
        let retirements = match &mut pending.ending {
            SpeculativeEnding::BackendCommittedPendingPublish { retirements, .. }
            | SpeculativeEnding::BackendAbortedPendingCleanup { retirements, .. } => retirements,
            _ => unreachable!("speculative retirement progress requires a post-terminal phase"),
        };
        while let Some(retirement) = retirements.pop_front() {
            if let Err((error, retirement)) = self.progress_or_defer_resident_retirement(retirement)
            {
                retirements.push_front(retirement);
                return Err(error);
            }
        }
        Ok(())
    }

    fn retain_speculative_ending_error<T>(
        &mut self,
        pending: PendingSpeculativeEndingDriverCohort<R::SequenceState>,
        error: Error,
    ) -> Result<T> {
        self.speculative_transactions.insert(
            pending.transaction,
            PendingSpeculativeDriverCohort::Ending(Box::new(pending)),
        );
        Err(error)
    }

    #[allow(clippy::too_many_arguments)]
    fn drive_quiesced_speculative_failure<F>(
        &mut self,
        transaction: ExecutionTransactionId,
        cohort_start: Instant,
        started_ns: u64,
        actions: Vec<DecodeAction>,
        prepared: Vec<PreparedSpeculativeAction>,
        source_states: Vec<R::SequenceState>,
        schedules: Vec<SuspendedSequenceSchedule>,
        cleanup: QuiescedSpeculativeAbortProgress<R::SequenceState>,
        retirements: Vec<KvRetirement>,
        error: Error,
        stage: &'static str,
        request_id: Option<RequestId>,
        continuation: Option<ContinuationId>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        R: ResidentModelRunner,
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        self.drive_speculative_ending(
            PendingSpeculativeEndingDriverCohort {
                transaction,
                cohort_start,
                started_ns,
                actions,
                prepared,
                source_states,
                schedules,
                ending: SpeculativeEnding::BackendAbortedPendingCleanup {
                    progress: cleanup,
                    retirements: retirements
                        .into_iter()
                        .map(PendingKvRetirement::BackendRelease)
                        .collect(),
                    custody: Some(if request_id.is_some() {
                        TransactionCustodyOutcome::Cancelled
                    } else {
                        TransactionCustodyOutcome::RolledBack
                    }),
                    failure: Some((error, stage)),
                    decode_requeued: false,
                },
                request_id,
                continuation,
            },
            on_token,
        )
    }

    fn drive_resident_endings<F>(&mut self, on_token: &mut F) -> Result<Option<ResidentDriverStep>>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transactions = self
            .resident_transactions
            .iter()
            .filter_map(|(transaction, pending)| pending.phase.is_ending().then_some(*transaction))
            .collect::<Vec<_>>();
        for transaction in transactions {
            let pending = self
                .resident_transactions
                .remove(&transaction)
                .expect("ending transaction was collected above");
            let step = self.drive_resident_ending(pending, on_token)?;
            if !matches!(step, ResidentDriverStep::Blocked) {
                return Ok(Some(step));
            }
        }
        Ok(None)
    }

    fn drive_resident_ending<F>(
        &mut self,
        mut pending: PendingResidentBatch<R::SequenceState>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transaction = pending.transaction;
        let intent = match &pending.phase {
            ResidentTransactionPhase::Publishing { .. } => Some(TransactionEndIntent::Publish),
            ResidentTransactionPhase::Aborting { .. } => Some(TransactionEndIntent::Abort),
            ResidentTransactionPhase::BackendCommittedPendingPublish { .. } => {
                return self.publish_resident_transaction(pending, on_token);
            }
            ResidentTransactionPhase::BackendAbortedPendingCleanup { .. } => {
                return self
                    .finish_resident_abort(pending, None)
                    .map(|(step, _)| step);
            }
            ResidentTransactionPhase::Executing(_) => {
                self.resident_transactions.insert(transaction, pending);
                return Err(Error::Invariant {
                    message: format!("transaction {transaction:?} is not ending"),
                });
            }
        }
        .expect("backend ending phase has an end intent");
        match self
            .executor
            .end_transaction(transaction, &mut pending.states, intent)
        {
            Ok(TransactionEndProgress::Pending) => {
                self.resident_transactions.insert(transaction, pending);
                Ok(ResidentDriverStep::Blocked)
            }
            Err(error) => {
                self.resident_transactions.insert(transaction, pending);
                Err(error)
            }
            Ok(TransactionEndProgress::Complete) => {
                let phase = std::mem::replace(
                    &mut pending.phase,
                    ResidentTransactionPhase::Executing(None),
                );
                match phase {
                    ResidentTransactionPhase::Publishing {
                        output,
                        started_ns,
                        cancellation_request,
                    } => {
                        pending.phase = ResidentTransactionPhase::BackendCommittedPendingPublish {
                            output,
                            custody: Some(TransactionCustodyOutcome::Committed {
                                started_ns,
                                finished_ns: self.runtime_now_ns(),
                            }),
                            cancellation_request,
                        };
                        self.publish_resident_transaction(pending, on_token)
                    }
                    ResidentTransactionPhase::Aborting {
                        request,
                        custody,
                        continuation,
                        failure,
                    } => {
                        pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
                            request,
                            custody: Some(custody),
                            continuation,
                            failure,
                            decode_requeued: false,
                        };
                        self.finish_resident_abort(pending, None)
                            .map(|(step, _)| step)
                    }
                    ResidentTransactionPhase::Executing(_)
                    | ResidentTransactionPhase::BackendCommittedPendingPublish { .. }
                    | ResidentTransactionPhase::BackendAbortedPendingCleanup { .. } => {
                        unreachable!("matched pre-terminal ending phase")
                    }
                }
            }
        }
    }

    fn finish_resident_abort(
        &mut self,
        mut pending: PendingResidentBatch<R::SequenceState>,
        requested: Option<RequestId>,
    ) -> Result<(ResidentDriverStep, Option<CancelRequestResult>)> {
        let transaction = pending.transaction;
        let ResidentTransactionPhase::BackendAbortedPendingCleanup {
            request,
            mut custody,
            mut continuation,
            failure,
            mut decode_requeued,
        } = std::mem::replace(
            &mut pending.phase,
            ResidentTransactionPhase::Executing(None),
        )
        else {
            unreachable!("resident abort progress requires a backend-aborted phase")
        };

        if let Some(outcome) = custody
            && let Err(error) = self.load_registry.finish_transaction_custody(
                transaction,
                outcome,
                self.runtime_now_ns(),
            )
        {
            custody = Some(outcome);
            pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
                request,
                custody,
                continuation,
                failure,
                decode_requeued,
            };
            return self.retain_resident_ending_error(pending, error.into());
        }

        if let Some(continuation_id) = continuation {
            if let Err(error) = self.detach_registered_continuation(
                continuation_id,
                CancellationReason::ExternalRequest,
            ) {
                continuation = Some(continuation_id);
                pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
                    request,
                    custody: None,
                    continuation,
                    failure,
                    decode_requeued,
                };
                return self.retain_resident_ending_error(pending, error);
            }
            continuation = None;
        }

        if let Some(kv) = pending.kv.take()
            && let Err((error, kv)) = self.progress_aborted_resident_kv(kv)
        {
            pending.kv = Some(kv);
            pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
                request,
                custody: None,
                continuation,
                failure,
                decode_requeued,
            };
            return self.retain_resident_ending_error(pending, error);
        }

        if let Err(error) =
            self.progress_transaction_session_restore(&mut pending.schedules, &mut pending.states)
        {
            pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
                request,
                custody: None,
                continuation,
                failure,
                decode_requeued,
            };
            return self.retain_resident_ending_error(pending, error);
        }

        if let Some((error, stage)) = failure {
            let error = self.abort_action(&pending.action, error, false, stage);
            let cleanup = self
                .finish_transaction_cancellations(transaction, request)
                .map(|_| ());
            return Err(Error::with_cleanup(
                "resident abort cancellation",
                error,
                cleanup,
            ));
        }
        if !decode_requeued {
            if let Err(error) = self
                .scheduler
                .requeue_decode_actions_front(action_decode_actions(&pending.action))
            {
                pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
                    request,
                    custody: None,
                    continuation: None,
                    failure: None,
                    decode_requeued,
                };
                return self.retain_resident_ending_error(pending, error);
            }
            decode_requeued = true;
        }
        let cancellation =
            match self.finish_transaction_cancellations(transaction, requested.or(request)) {
                Ok(result) => result,
                Err(error) => {
                    pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
                        request,
                        custody: None,
                        continuation: None,
                        failure: None,
                        decode_requeued,
                    };
                    return self
                        .retain_resident_ending_error(pending, error)
                        .map(|step| (step, None));
                }
            };
        Ok((
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Cancel,
                rows: 0,
                staged: 0,
                finished: 0,
            },
            cancellation,
        ))
    }

    fn progress_aborted_resident_kv(
        &mut self,
        kv: PendingResidentKv,
    ) -> std::result::Result<(), (Error, PendingResidentKv)> {
        let kv = match kv {
            PendingResidentKv::Reserved(reservations) if reservations.is_empty() => return Ok(()),
            PendingResidentKv::Reserved(reservations) => {
                let Some(manager) = self.page_manager.as_mut() else {
                    return Err((
                        Error::Invariant {
                            message: "KV reservations have no authoritative page manager".into(),
                        },
                        PendingResidentKv::Reserved(reservations),
                    ));
                };
                let retirement = match manager.abort_reservations(reservations) {
                    Ok(retirement) => retirement,
                    Err(error) => {
                        let (error, reservations) = error.into_parts();
                        return Err((error, PendingResidentKv::Reserved(reservations)));
                    }
                };
                PendingResidentKv::Retiring(PendingKvRetirement::BackendRelease(retirement))
            }
            PendingResidentKv::Prepared(prepared) => {
                let Some(manager) = self.page_manager.as_mut() else {
                    return Err((
                        Error::Invariant {
                            message: "prepared logical commit has no authoritative page manager"
                                .into(),
                        },
                        PendingResidentKv::Prepared(prepared),
                    ));
                };
                PendingResidentKv::Retiring(PendingKvRetirement::BackendRelease(
                    manager.abort_prepared_commit(prepared),
                ))
            }
            kv @ PendingResidentKv::Retiring(_) => kv,
        };
        let PendingResidentKv::Retiring(retirement) = kv else {
            unreachable!("aborted logical KV was converted to retirement")
        };
        self.progress_or_defer_resident_retirement(retirement)
            .map_err(|(error, retirement)| (error, PendingResidentKv::Retiring(retirement)))
    }

    fn retain_resident_ending_error<T>(
        &mut self,
        pending: PendingResidentBatch<R::SequenceState>,
        error: Error,
    ) -> Result<T> {
        self.resident_transactions
            .insert(pending.transaction, pending);
        Err(error)
    }

    fn abort_failed_resident(
        &mut self,
        mut pending: PendingResidentBatch<R::SequenceState>,
        error: Error,
        stage: &'static str,
    ) -> Result<ResidentDriverStep> {
        if let ResidentTransactionPhase::Aborting { failure, .. } = &mut pending.phase {
            *failure = Some(match failure.take() {
                Some((previous, previous_stage)) => (
                    Error::combine("resident transaction abort", previous, error),
                    previous_stage,
                ),
                None => (error, stage),
            });
        } else {
            let continuation = pending
                .phase
                .pending_progress()
                .map(PendingModelProgress::continuation);
            pending.phase = ResidentTransactionPhase::Aborting {
                request: None,
                custody: TransactionCustodyOutcome::RolledBack,
                continuation,
                failure: Some((error, stage)),
            };
        }
        self.drive_resident_ending(pending, &mut |_| Ok(()))
    }

    fn pending_model_progresses(&self) -> Vec<PendingModelProgress> {
        let mut progresses = self
            .resident_transactions
            .values()
            .filter_map(|pending| pending.phase.pending_progress().cloned())
            .collect::<Vec<_>>();
        for pending in self.speculative_transactions.values() {
            pending.extend_pending_progress(&mut progresses);
        }
        progresses.sort_unstable_by_key(|progress| {
            (progress.transaction().get(), progress.continuation().get())
        });
        progresses.dedup_by_key(|progress| progress.continuation());
        progresses
    }

    fn abort_action(
        &mut self,
        action: &SchedulerAction,
        error: Error,
        poison_executor: bool,
        stage: &'static str,
    ) -> Error {
        let _ = poison_executor;
        let session_ids = action_session_ids(action);
        let mut cleanup = Vec::new();
        if let Err(source) = self.scheduler.fail_action(action, &mut self.slot_pool) {
            cleanup.push(CleanupStep::new("scheduler action cleanup", source));
        }
        for session_id in &session_ids {
            self.retained_sessions.remove(session_id);
            if let Err(source) = self.release_sequence_state(*session_id) {
                cleanup.push(CleanupStep::new(
                    format!("session {session_id:?} state cleanup"),
                    source,
                ));
            }
        }
        Error::with_cleanup_batch(stage, error, cleanup)
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

    fn enqueue_committed_decode_tokens(&mut self, actions: &[DecodeAction]) -> Result<usize> {
        for action in actions {
            let runner = self.executor.runner();
            let sequence = self
                .scheduler
                .active_sequence_mut(action.session_id)
                .ok_or_else(|| Error::Invariant {
                    message: format!(
                        "cannot emit token for inactive session {:?}",
                        action.session_id
                    ),
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
        Ok(actions.len())
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
                .ok_or_else(|| Error::InvalidRequest {
                    message: format!(
                        "output input row {} has no scheduled sequence",
                        row.input_row
                    ),
                })?;
            let execution_sequence = scheduled
                .execution()
                .sequences()
                .iter()
                .find(|sequence| sequence.query.contains(&row.input_row))
                .ok_or_else(|| Error::InvalidRequest {
                    message: format!(
                        "output input row {} has no execution sequence span",
                        row.input_row
                    ),
                })?;
            let session_id = correlation.session_id;
            let Some(sequence) = self.scheduler.active_sequence(session_id) else {
                if action_finished_sessions.contains(&session_id) {
                    // The just-committed token ended the sequence (for example via
                    // a stop string), so its already-computed next-token logits are
                    // intentionally discarded after successful correlation.
                    continue;
                }
                return Err(Error::InvalidRequest {
                    message: format!(
                        "output for input row {} references inactive session {:?}",
                        row.input_row, session_id
                    ),
                });
            };
            if sequence.request_id != correlation.request_id {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "output correlation request mismatch for session {:?}: active {:?}, scheduled {:?}",
                        session_id, sequence.request_id, correlation.request_id
                    ),
                });
            }
            if sequence.kv_handle != correlation.kv_handle {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "output correlation KV mismatch for session {:?}: active {:?}, scheduled {:?}",
                        session_id, sequence.kv_handle, correlation.kv_handle
                    ),
                });
            }
            if sequence.position != execution_sequence.sequence_len as usize {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "output correlation position mismatch for session {:?}: active {}, executed {}",
                        session_id, sequence.position, execution_sequence.sequence_len
                    ),
                });
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
                && self.executor.runner().is_eos_token(candidate.token_id)
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
    fn progress_ending_transactions<F>(&mut self, on_token: &mut F) -> Result<()>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let resident = self
            .resident_transactions
            .iter()
            .filter_map(|(transaction, pending)| pending.phase.is_ending().then_some(*transaction))
            .collect::<Vec<_>>();
        for transaction in resident {
            let pending = self
                .resident_transactions
                .remove(&transaction)
                .expect("post-terminal resident transaction was collected above");
            self.drive_resident_ending(pending, on_token)?;
        }

        let speculative = self
            .speculative_transactions
            .iter()
            .filter_map(|(transaction, pending)| match pending {
                PendingSpeculativeDriverCohort::Ending(_) => Some(*transaction),
                _ => None,
            })
            .collect::<Vec<_>>();
        for transaction in speculative {
            let PendingSpeculativeDriverCohort::Ending(pending) = self
                .speculative_transactions
                .remove(&transaction)
                .expect("ending speculative transaction was collected above")
            else {
                unreachable!("collected speculative transaction was ending")
            };
            self.drive_speculative_ending(*pending, on_token)?;
        }
        Ok(())
    }

    fn progress_post_terminal_transaction_cleanups<F>(&mut self, on_token: &mut F) -> Result<()>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let resident = self
            .resident_transactions
            .iter()
            .filter_map(|(transaction, pending)| {
                matches!(
                    pending.phase,
                    ResidentTransactionPhase::BackendCommittedPendingPublish { .. }
                        | ResidentTransactionPhase::BackendAbortedPendingCleanup { .. }
                )
                .then_some(*transaction)
            })
            .collect::<Vec<_>>();
        for transaction in resident {
            let pending = self
                .resident_transactions
                .remove(&transaction)
                .expect("post-terminal resident transaction was collected above");
            self.drive_resident_ending(pending, on_token)?;
        }

        let speculative = self
            .speculative_transactions
            .iter()
            .filter_map(|(transaction, pending)| match pending {
                PendingSpeculativeDriverCohort::Proposing(pending) if pending.backend_aborted => {
                    Some(*transaction)
                }
                PendingSpeculativeDriverCohort::Ending(pending)
                    if matches!(
                        pending.ending,
                        SpeculativeEnding::BackendCommittedPendingPublish { .. }
                            | SpeculativeEnding::BackendAbortedPendingCleanup { .. }
                    ) =>
                {
                    Some(*transaction)
                }
                PendingSpeculativeDriverCohort::Proposing(_)
                | PendingSpeculativeDriverCohort::Verifying(_)
                | PendingSpeculativeDriverCohort::Ending(_) => None,
            })
            .collect::<Vec<_>>();
        for transaction in speculative {
            let pending = self
                .speculative_transactions
                .remove(&transaction)
                .expect("post-terminal speculative transaction was collected above");
            match pending {
                PendingSpeculativeDriverCohort::Proposing(pending) => {
                    let cancellation_request = pending.cancellation_request;
                    let terminal = self.cancel_native_proposal_cohort(*pending, None, None);
                    if self.speculative_transactions.contains_key(&transaction) {
                        return match terminal {
                            Ok(TransactionEndProgress::Pending) => Err(Error::Invariant {
                                message: format!(
                                    "post-terminal proposal transaction {transaction:?} returned pending"
                                ),
                            }),
                            Ok(TransactionEndProgress::Complete) => Err(Error::Invariant {
                                message: format!(
                                    "post-terminal proposal transaction {transaction:?} reported complete but retained ownership"
                                ),
                            }),
                            Err(error) => Err(error),
                        };
                    }
                    let cleanup = self
                        .finish_transaction_cancellations(transaction, cancellation_request)
                        .map(|_| ());
                    match terminal {
                        Ok(TransactionEndProgress::Complete) => cleanup?,
                        Ok(TransactionEndProgress::Pending) => {
                            return Err(Error::with_cleanup(
                                "post-terminal proposal cancellation",
                                Error::Invariant {
                                    message: format!(
                                        "post-terminal proposal transaction {transaction:?} returned pending without ownership"
                                    ),
                                },
                                cleanup,
                            ));
                        }
                        Err(error) => {
                            return Err(Error::with_cleanup(
                                "post-terminal proposal cancellation",
                                error,
                                cleanup,
                            ));
                        }
                    }
                }
                PendingSpeculativeDriverCohort::Ending(pending) => {
                    self.drive_speculative_ending(*pending, on_token)?;
                }
                PendingSpeculativeDriverCohort::Verifying(_) => {
                    unreachable!("post-terminal speculative transaction cannot be verifying")
                }
            }
        }
        Ok(())
    }

    pub fn shutdown_progress<F>(&mut self, on_token: &mut F) -> Result<ResidentShutdownProgress>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        const COMPLETIONS_PER_TICK: usize = 512;
        loop {
            match self.shutdown(on_token, COMPLETIONS_PER_TICK) {
                Ok(report) => return Ok(ResidentShutdownProgress::Complete(report)),
                Err(Error::Registry { source })
                    if matches!(*source, crate::io::RegistryError::ShutdownIncomplete { .. })
                        && (self.load_registry.pending_completions() != 0
                            || self.load_registry.has_pending_owner_work()) =>
                {
                    continue;
                }
                Err(Error::Registry { source })
                    if matches!(
                        *source,
                        crate::io::RegistryError::ShutdownIncomplete { .. }
                            | crate::io::RegistryError::LostCompletion { .. }
                    ) =>
                {
                    return Ok(ResidentShutdownProgress::Pending);
                }
                Err(Error::ShutdownIncomplete { .. }) => {
                    return Ok(ResidentShutdownProgress::Pending);
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Stop admission, quiesce model transactions, release all session/KV
    /// ownership, and drain provider completions. An error retains any ownership
    /// that could not yet be proven quiescent so the caller may retry.
    pub fn shutdown<F>(
        &mut self,
        on_token: &mut F,
        maximum_completions: usize,
    ) -> Result<ResidentDriverShutdownReport>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        self.shutting_down = true;
        self.flush_committed_token_outbox(on_token)?;
        self.progress_ending_transactions(on_token)?;

        let mut transactions = self
            .resident_transactions
            .keys()
            .chain(self.speculative_transactions.keys())
            .copied()
            .filter(|transaction| {
                !self
                    .resident_transactions
                    .get(transaction)
                    .is_some_and(|pending| pending.phase.is_ending())
                    && !matches!(
                        self.speculative_transactions.get(transaction),
                        Some(PendingSpeculativeDriverCohort::Ending(_))
                    )
            })
            .collect::<Vec<_>>();
        transactions.sort_unstable();
        transactions.dedup();
        for transaction in transactions {
            let request_id =
                self.request_for_transaction(transaction)
                    .ok_or_else(|| Error::InvalidRequest {
                        message: format!(
                            "cannot quiesce transaction {transaction:?} without a request owner"
                        ),
                    })?;
            self.request_transaction_abort(transaction, request_id)?;
        }
        // A cancellation request can retain post-terminal synchronous cleanup
        // while reporting Pending to the request API. Retry only that owner-local
        // work before classifying pre-terminal backend progress as wake-dependent.
        self.progress_post_terminal_transaction_cleanups(on_token)?;
        if !self.resident_transactions.is_empty()
            || !self.speculative_transactions.is_empty()
            || self.has_live_transactions()
        {
            return Err(Error::ShutdownIncomplete {
                message: "driver could not quiesce every execution transaction".into(),
            });
        }
        self.retry_pending_continuation_cleanups()?;
        self.cleanup_materialization_failures(false)?;
        self.drain_prefix_cache()?;

        for request_id in self.scheduler.request_ids() {
            self.cancel_scheduled_request(request_id)?;
        }

        let suspended = self.suspended_sequences.keys().copied().collect::<Vec<_>>();
        for session_id in suspended {
            let SuspendedDriverSequence {
                model_state,
                page_slot: _,
                kv_state,
                schedule,
            } = self
                .suspended_sequences
                .remove(&session_id)
                .expect("suspended session identity was collected above");
            self.pending_sequence_cleanups.insert(
                session_id,
                PendingSequenceCleanup::Suspended {
                    kv_state: Some(kv_state),
                    retirement: None,
                    model_state: Some(model_state),
                },
            );
            self.prefix_cache_sessions.remove(&session_id);
            self.scheduler
                .cancel_suspended(schedule, &mut self.slot_pool)?;
            self.progress_sequence_cleanup(session_id)?;
        }

        for session_id in self.scheduler.active_session_ids() {
            let cancellation = self
                .scheduler
                .cancel_sequence(session_id, &mut self.slot_pool);
            let cleanup = self.release_sequence_state(session_id);
            match (cancellation, cleanup) {
                (Ok(_), Ok(())) => {}
                (Err(error), Ok(())) | (Ok(_), Err(error)) => return Err(error),
                (Err(error), Err(cleanup)) => {
                    return Err(Error::cleanup("active session shutdown", error, cleanup));
                }
            }
        }
        let retained = self.sequence_states.keys().copied().collect::<Vec<_>>();
        for session_id in retained {
            self.release_sequence_state(session_id)?;
        }
        self.retained_sessions.clear();
        self.prefix_cache_sessions.clear();
        self.progress_pending_cleanups()?;
        self.retry_pending_request_cancellations()?;

        let registry = self
            .load_registry
            .shutdown(self.runtime_now_ns(), maximum_completions)?;

        self.progress_pending_cleanups()?;
        self.update_hard_resource_observability();

        let executor_transactions =
            self.resident_transactions.len() + self.speculative_transactions.len();
        let report = ResidentDriverShutdownReport {
            registry,
            executor_transactions,
            kv_page_grants: self.kv_page_grants.len(),
            pending_kv_retirements: self.pending_kv_retirements.len(),
        };
        let retiring_pages = self
            .page_manager
            .as_ref()
            .map_or(0, |manager| manager.stats().retiring_pages);
        if report.executor_transactions != 0
            || report.kv_page_grants != 0
            || report.pending_kv_retirements != 0
            || !self.pending_resident_kv_aborts.is_empty()
            || retiring_pages != 0
            || !self.continuations.is_empty()
            || !self.transaction_continuations.is_empty()
            || !self.pending_materialization_failures.is_empty()
            || !self.pending_registry_detaches.is_empty()
            || !self.session_owner.is_empty()
            || !self.pending_sequence_cleanups.is_empty()
            || !self.pending_request_cancellations.is_empty()
            || !self.pending_prefix_cleanups.is_empty()
            || !self.prefix_cache.is_empty()
            || !self.prefix_cache_sessions.is_empty()
            || !self.sequence_states.is_empty()
            || !self.suspended_sequences.is_empty()
            || self.scheduler.failed_slot_ownership() != 0
            || !self.scheduler.is_idle()
        {
            return Err(Error::Invariant {
                message: format!(
                    "driver shutdown retained ownership: {report:?}, resident_kv_aborts={}, retiring_pages={retiring_pages}, continuations={}, transaction_continuations={}, session_owners={}, cleanups={}, request_cancellations={}, prefix_cleanups={}, cached_prefixes={}, prefix_sessions={}, sequence_states={}, suspended={}, failed_slots={}, scheduler_idle={}",
                    self.pending_resident_kv_aborts.len(),
                    self.continuations.len(),
                    self.transaction_continuations.len(),
                    self.session_owner.len(),
                    self.pending_sequence_cleanups.len(),
                    self.pending_request_cancellations.len(),
                    self.pending_prefix_cleanups.len(),
                    self.prefix_cache.len(),
                    self.prefix_cache_sessions.len(),
                    self.sequence_states.len(),
                    self.suspended_sequences.len(),
                    self.scheduler.failed_slot_ownership(),
                    self.scheduler.is_idle(),
                ),
            });
        }
        Ok(report)
    }

    /// Execute the resident model path selected by model capabilities.
    ///
    /// All models use the same packed target executor. A model that reports a
    /// checkpoint-native proposal capability may add proposal + verification for
    /// decode; models without it continue through target-only packed decode.
    pub fn step<F>(&mut self, on_token: &mut F) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        if self.shutting_down {
            return Err(Error::InvalidRequest {
                message: "resident driver is shutting down; use shutdown() to drain ownership"
                    .into(),
            });
        }
        self.runtime_tick = self.runtime_tick.saturating_add(1);
        if let Some(step) = self.drive_resident_endings(on_token)? {
            return Ok(step);
        }
        if let Some(step) = self.drive_speculative_endings(on_token)? {
            return Ok(step);
        }
        self.progress_pending_cleanups()?;
        self.retry_pending_request_cancellations()?;
        self.flush_committed_token_outbox(on_token)?;
        self.begin_warmup()?;
        self.progress_materialization()?;
        self.check_warmup()?;
        self.update_hard_resource_observability();
        let proposal_enabled = self.config.enable_native_proposals
            && self.prefix_cache.capacity() == 0
            && self.executor.runner().native_proposal_source()?.is_some();

        // Prefix eviction cannot run while speculative topology is owned. Until
        // that pressure protocol is transactional, a configured prefix cache
        // selects the safe target-only path rather than risking request failure.
        let requires_page_manager = proposal_enabled
            || self.executor.capabilities().kv_binding_mode == KvBindingMode::Paged;
        if requires_page_manager && self.page_manager.is_none() {
            return Err(Error::InvalidRequest { message:
                "paged or proposal-enabled resident execution requires an authoritative KvPageManager"
                    .into(),
             });
        }
        let resumable = self.ready_transactions.len();
        for _ in 0..resumable {
            let Some(transaction) = self.pop_ready_transaction() else {
                break;
            };
            let ready_continuation = self.ready_continuation_for(transaction).or_else(|| {
                self.transaction_continuations
                    .get(&transaction)
                    .and_then(|continuations| continuations.iter().copied().next())
            });
            let Some(ready_continuation) = ready_continuation else {
                continue;
            };
            let completed = if self.resident_transactions.contains_key(&transaction) {
                self.resume_resident_transaction(transaction, ready_continuation, on_token)?
            } else if self.speculative_transactions.contains_key(&transaction) {
                self.resume_speculative_transaction(transaction, ready_continuation, on_token)?
            } else {
                None
            };
            if self.ready_continuation_for(transaction).is_some() {
                self.enqueue_transaction(transaction);
            }
            if let Some(step) = completed {
                return Ok(step);
            }
        }

        self.prepare_step()?;
        let allow_mixed_batches = self.scheduler.config().allow_mixed_batches;
        loop {
            let action = self
                .scheduler
                .next_admitted_action_policy(allow_mixed_batches)?;
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
                SchedulerAction::Execute { prefills, decodes }
                    if proposal_enabled && prefills.is_empty() =>
                {
                    self.execute_speculative_decode_batch(decodes, on_token)?
                }
                action => self.execute_planned_action(action, on_token)?,
            };
            if matches!(step, ResidentDriverStep::WaitingForModelProgress(_)) {
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
            return Err(Error::Invariant {
                message: "production speculative decode batch cannot be empty".into(),
            });
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
                    Error::InvalidRequest {
                        message:
                            "resident decode entered proposal verification without model capability"
                                .into(),
                    },
                    "production speculative initialization",
                ));
            }
            Err(error) => {
                return Err(self.abort_speculative_decode_batch(
                    actions,
                    error.into(),
                    "production speculative initialization",
                ));
            }
        };
        if let Err(error) = proposal_source.validate() {
            return Err(self.abort_speculative_decode_batch(
                actions,
                error.into(),
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
                abort_cause: None,
                backend_aborted: false,
                custody: None,
                decode_requeued: false,
            },
            on_token,
            None,
        )
    }

    fn advance_native_proposal_cohort<F>(
        &mut self,
        mut pending: PendingNativeProposalCohort<R::SequenceState>,
        on_token: &mut F,
        mut resume: Option<(ContinuationId, crate::io::ResumeLease)>,
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
            let progress = match &pending.slots[slot_index].status {
                NativeProposalSlotStatus::NotStarted => {
                    let anchor_token = pending.actions[slot_index].token_id;
                    let proposal_started_ns = self.runtime_now_ns();
                    let progress = self
                        .executor
                        .with_sequence_state(&mut pending.source_states[slot_index], |runner| {
                            runner.begin_native_proposal(pending.transaction, anchor_token)
                        });
                    self.record_runnable_work_span(proposal_started_ns)?;
                    Some(progress)
                }
                NativeProposalSlotStatus::Waiting(waiting) => {
                    let continuation = waiting.continuation();
                    if resume
                        .as_ref()
                        .is_none_or(|(ready, _)| *ready != continuation)
                    {
                        None
                    } else {
                        let (_, mut resume_lease) = resume
                            .take()
                            .expect("matching proposal resume lease exists");
                        let leases = resume_lease.take()?;
                        let resume_started = self.runtime_now_ns();
                        let progress = self.executor.with_sequence_state(
                            &mut pending.source_states[slot_index],
                            |runner| {
                                runner.resume_native_proposal(
                                    pending.transaction,
                                    continuation,
                                    leases,
                                )
                            },
                        );
                        let disposition = if progress.is_err() {
                            ResumeDisposition::StillActive
                        } else {
                            ResumeDisposition::Consumed
                        };
                        self.finish_resume_lease(
                            continuation,
                            resume_lease,
                            disposition,
                            resume_started,
                        )?;
                        self.record_runnable_work_span(resume_started)?;
                        Some(progress)
                    }
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
                        let registration = self.register_pending_progress(
                            &waiting,
                            crate::scheduling::ResourceDemand::required(
                                ExecutionPhase::SpeculativeProposal,
                            ),
                        );
                        pending.slots[slot_index].status =
                            NativeProposalSlotStatus::Waiting(waiting);
                        if let Err(error) = registration
                            && first_error.is_none()
                        {
                            first_error = Some(error);
                        }
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
                Ok(TransactionEndProgress::Pending) => Ok(ResidentDriverStep::Blocked),
                Ok(TransactionEndProgress::Complete) => Err(Error::Invariant {
                    message: "speculative proposal failure was lost during cancellation".into(),
                }),
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
                PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
            );
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
        source_states: Vec<R::SequenceState>,
        schedules: Vec<SuspendedSequenceSchedule>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let verification_items = {
            let page_manager = self
                .page_manager
                .as_ref()
                .expect("speculative verification requires a page manager");
            actions
                .iter()
                .zip(&prepared)
                .map(|(action, prepared)| {
                    Ok(SpeculativeVerificationItem {
                        state_slot: prepared.page_slot,
                        generation: page_manager.sequence_generation(prepared.page_slot)?,
                        proposal: &prepared.proposal,
                        frontier: TargetFrontier {
                            position: action.position,
                            top1: ferrule_common::execution::TokenLogit::new(
                                action.token_id,
                                action.logit.unwrap_or(0.0),
                            ),
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()
        };
        let verification_items = match verification_items {
            Ok(items) => items,
            Err(error) => {
                self.finish_speculative_transaction_custody(
                    transaction,
                    TransactionCustodyOutcome::RolledBack,
                )?;
                self.restore_transaction_sessions(schedules, source_states)?;
                return Err(self.abort_speculative_decode_batch(
                    &actions,
                    error,
                    "production speculative KV generation",
                ));
            }
        };
        let reservations = match self.reserve_speculative_pages(transaction, &verification_items) {
            Ok(reservations) => reservations,
            Err(error) => {
                drop(verification_items);
                self.finish_speculative_transaction_custody(
                    transaction,
                    TransactionCustodyOutcome::RolledBack,
                )?;
                self.restore_transaction_sessions(schedules, source_states)?;
                return Err(self.abort_speculative_decode_batch(
                    &actions,
                    error,
                    "production speculative KV reserve",
                ));
            }
        };
        let verification_started_ns = self.runtime_now_ns();
        let verification = match self.page_manager.as_mut() {
            Some(page_manager) => prepare_speculative_verification_transaction(
                &mut self.executor,
                page_manager,
                transaction,
                &source_states,
                &verification_items,
                reservations,
                self.top_k,
            ),
            None => unreachable!("speculative page reservations require a page manager"),
        };
        drop(verification_items);
        let verification = match verification {
            Ok(verification) => verification,
            Err(SpeculativeCohortFailure::Quiesced { error, cleanup }) => {
                return self.drive_quiesced_speculative_failure(
                    transaction,
                    cohort_start,
                    verification_started_ns,
                    actions,
                    prepared,
                    source_states,
                    schedules,
                    *cleanup,
                    Vec::new(),
                    error,
                    "production speculative verification preparation",
                    None,
                    None,
                    on_token,
                );
            }
            Err(SpeculativeCohortFailure::Active { .. }) => {
                unreachable!("verification preparation cannot activate backend ownership")
            }
        };
        let demand =
            crate::scheduling::ResourceDemand::required(ExecutionPhase::SpeculativeVerification);
        if let Err(error) =
            self.declare_transaction_prefetch(transaction, demand, verification.batch())
        {
            return self.drive_quiesced_speculative_failure(
                transaction,
                cohort_start,
                verification_started_ns,
                actions,
                prepared,
                source_states,
                schedules,
                QuiescedSpeculativeAbortProgress::from_transaction(verification),
                Vec::new(),
                error,
                "speculative verification prefetch",
                None,
                None,
                on_token,
            );
        }
        let progress = match self.page_manager.as_mut() {
            Some(page_manager) => begin_prepared_speculative_verification(
                &mut self.executor,
                page_manager,
                &source_states,
                verification,
            ),
            None => unreachable!("speculative page reservations require a page manager"),
        };
        self.record_runnable_work_span(verification_started_ns)?;

        match progress {
            Ok(SpeculativeCohortProgress::Ready(prepared_cohort)) => self.drive_speculative_ending(
                PendingSpeculativeEndingDriverCohort {
                    transaction,
                    cohort_start,
                    started_ns: verification_started_ns,
                    actions,
                    prepared,
                    source_states,
                    schedules,
                    ending: SpeculativeEnding::Publish(*prepared_cohort),
                    request_id: None,
                    continuation: None,
                },
                on_token,
            ),
            Ok(SpeculativeCohortProgress::Waiting(verification)) => {
                if let Err(error) = self.register_pending_progress(
                    verification.pending_progress(),
                    crate::scheduling::ResourceDemand::required(
                        ExecutionPhase::SpeculativeVerification,
                    ),
                ) {
                    return self.drive_speculative_ending(
                        PendingSpeculativeEndingDriverCohort {
                            transaction,
                            cohort_start,
                            started_ns: verification_started_ns,
                            actions,
                            prepared,
                            source_states,
                            schedules,
                            ending: SpeculativeEnding::Abort {
                                transaction: verification.into_transaction(),
                                failure: Some((error, "speculative continuation registration")),
                            },
                            request_id: None,
                            continuation: None,
                        },
                        on_token,
                    );
                }
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
                Ok(ResidentDriverStep::WaitingForModelProgress(
                    self.pending_model_progresses(),
                ))
            }
            Err(SpeculativeCohortFailure::Active {
                error,
                transaction: backend,
            }) => self.drive_speculative_ending(
                PendingSpeculativeEndingDriverCohort {
                    transaction,
                    cohort_start,
                    started_ns: verification_started_ns,
                    actions,
                    prepared,
                    source_states,
                    schedules,
                    ending: SpeculativeEnding::Abort {
                        transaction: *backend,
                        failure: Some((error, "production speculative verification")),
                    },
                    request_id: None,
                    continuation: None,
                },
                on_token,
            ),
            Err(SpeculativeCohortFailure::Quiesced { error, cleanup }) => self
                .drive_quiesced_speculative_failure(
                    transaction,
                    cohort_start,
                    verification_started_ns,
                    actions,
                    prepared,
                    source_states,
                    schedules,
                    *cleanup,
                    Vec::new(),
                    error,
                    "production speculative verification",
                    None,
                    None,
                    on_token,
                ),
        }
    }

    fn resume_speculative_transaction<F>(
        &mut self,
        transaction: ExecutionTransactionId,
        ready_continuation: ContinuationId,
        on_token: &mut F,
    ) -> Result<Option<ResidentDriverStep>>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let pending = self
            .speculative_transactions
            .remove(&transaction)
            .ok_or_else(|| Error::Invariant {
                message: format!("speculative transaction {transaction:?} disappeared"),
            })?;
        match pending {
            PendingSpeculativeDriverCohort::Proposing(pending) => {
                let lease = match self.prepare_resume_lease(ready_continuation) {
                    Ok(lease) => lease,
                    Err(error) => {
                        self.speculative_transactions.insert(
                            transaction,
                            PendingSpeculativeDriverCohort::Proposing(pending),
                        );
                        return Err(error);
                    }
                };
                let step = self.advance_native_proposal_cohort(
                    *pending,
                    on_token,
                    Some((ready_continuation, lease)),
                )?;
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
                    Ok(TransactionEndProgress::Pending) => Ok(None),
                    Ok(TransactionEndProgress::Complete) => {
                        Ok(Some(ResidentDriverStep::Executed {
                            action_kind: ResidentActionKind::Cancel,
                            rows: 0,
                            staged: 0,
                            finished: 0,
                        }))
                    }
                    Err(error) => Err(error),
                }
            }
            PendingSpeculativeDriverCohort::Verifying(pending) => self
                .resume_speculative_verification_transaction(
                    *pending,
                    ready_continuation,
                    on_token,
                ),
            PendingSpeculativeDriverCohort::Ending(pending) => {
                self.speculative_transactions
                    .insert(transaction, PendingSpeculativeDriverCohort::Ending(pending));
                Ok(None)
            }
        }
    }

    fn resume_speculative_verification_transaction<F>(
        &mut self,
        pending: PendingSpeculativeVerificationDriverCohort<R::SequenceState>,
        ready_continuation: ContinuationId,
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
            source_states,
            schedules,
            verification,
            cancellation_request,
        } = pending;
        if self.page_manager.is_none() {
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
            return Err(Error::Invariant { message:
                "authoritative KvPageManager disappeared while speculative verification was suspended"
                    .into(),
             });
        }
        if verification.pending_progress().continuation() != ready_continuation {
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
            return Err(Error::InvalidRequest {
                message: format!(
                    "speculative verification transaction {} owns continuation {}, not ready continuation {}",
                    transaction.get(),
                    self.speculative_transactions
                        .get(&transaction)
                        .and_then(|pending| match pending {
                            PendingSpeculativeDriverCohort::Verifying(pending) => {
                                Some(pending.verification.pending_progress().continuation().get())
                            }
                            PendingSpeculativeDriverCohort::Proposing(_)
                            | PendingSpeculativeDriverCohort::Ending(_) => None,
                        })
                        .unwrap_or_default(),
                    ready_continuation.get()
                ),
            });
        }
        let mut resume_lease = match self.prepare_resume_lease(ready_continuation) {
            Ok(lease) => lease,
            Err(error) => {
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
                return Err(error);
            }
        };
        let leases = match resume_lease.take() {
            Ok(leases) => leases,
            Err(error) => {
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
                return Err(error.into());
            }
        };
        let resume_started = self.runtime_now_ns();
        let progress = resume_resumable_speculative_verification_cohort(
            &mut self.executor,
            self.page_manager
                .as_mut()
                .expect("page manager was checked above"),
            &source_states,
            verification,
            leases,
        );
        let disposition = if matches!(progress, Err(SpeculativeCohortFailure::Active { .. })) {
            ResumeDisposition::StillActive
        } else {
            ResumeDisposition::Consumed
        };
        self.finish_resume_lease(
            ready_continuation,
            resume_lease,
            disposition,
            resume_started,
        )?;
        self.record_runnable_work_span(resume_started)?;

        match progress {
            Ok(SpeculativeCohortProgress::Ready(prepared_cohort)) => self
                .drive_speculative_ending(
                    PendingSpeculativeEndingDriverCohort {
                        transaction,
                        cohort_start,
                        started_ns: resume_started,
                        actions,
                        prepared,
                        source_states,
                        schedules,
                        ending: SpeculativeEnding::Publish(*prepared_cohort),
                        request_id: cancellation_request,
                        continuation: None,
                    },
                    on_token,
                )
                .map(Some),
            Ok(SpeculativeCohortProgress::Waiting(verification)) => {
                if let Err(error) = self.register_pending_progress(
                    verification.pending_progress(),
                    crate::scheduling::ResourceDemand::required(
                        ExecutionPhase::SpeculativeVerification,
                    ),
                ) {
                    return self
                        .drive_speculative_ending(
                            PendingSpeculativeEndingDriverCohort {
                                transaction,
                                cohort_start,
                                started_ns: resume_started,
                                actions,
                                prepared,
                                source_states,
                                schedules,
                                ending: SpeculativeEnding::Abort {
                                    transaction: verification.into_transaction(),
                                    failure: Some((error, "speculative continuation registration")),
                                },
                                request_id: cancellation_request,
                                continuation: None,
                            },
                            on_token,
                        )
                        .map(Some);
                }
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
                Ok(None)
            }
            Err(SpeculativeCohortFailure::Active {
                error,
                transaction: backend,
            }) => self
                .drive_speculative_ending(
                    PendingSpeculativeEndingDriverCohort {
                        transaction,
                        cohort_start,
                        started_ns: resume_started,
                        actions,
                        prepared,
                        source_states,
                        schedules,
                        ending: SpeculativeEnding::Abort {
                            transaction: *backend,
                            failure: Some((error, "production speculative resume")),
                        },
                        request_id: cancellation_request,
                        continuation: Some(ready_continuation),
                    },
                    on_token,
                )
                .map(Some),
            Err(SpeculativeCohortFailure::Quiesced { error, cleanup }) => self
                .drive_quiesced_speculative_failure(
                    transaction,
                    cohort_start,
                    resume_started,
                    actions,
                    prepared,
                    source_states,
                    schedules,
                    *cleanup,
                    Vec::new(),
                    error,
                    "production speculative resume",
                    cancellation_request,
                    None,
                    on_token,
                )
                .map(Some),
        }
    }

    fn publish_speculative_decode_cohort(
        &mut self,
        transaction: ExecutionTransactionId,
        cohort_start: Instant,
        actions: Vec<DecodeAction>,
        prepared: Vec<PreparedSpeculativeAction>,
        cohort: crate::speculation::SpeculativeCohortResult,
    ) -> Result<ResidentDriverStep> {
        if cohort.results.len() != actions.len() {
            return Err(Error::Invariant {
                message: format!(
                    "speculative cohort returned {} results for {} actions",
                    cohort.results.len(),
                    actions.len()
                ),
            });
        }

        let cohort_transaction_time_us = cohort.transaction_time_us;
        let cohort_verify_time_us = cohort.verify_time_us;
        let mut rows = 0usize;
        let mut staged = 0usize;
        let mut finished = 0usize;
        let mut externally_committed_tokens = 0usize;

        for ((action, prepared), result) in actions.iter().zip(prepared).zip(cohort.results) {
            let externally_committed =
                result
                    .accepted
                    .len()
                    .checked_add(1)
                    .ok_or_else(|| Error::Invariant {
                        message: "speculative external token count overflow".into(),
                    })?;
            if result.accounting.externally_committed_tokens != externally_committed {
                return Err(Error::Invariant {
                    message: format!(
                        "speculative transaction committed {} rows but returned {} external tokens",
                        result.accounting.externally_committed_tokens, externally_committed
                    ),
                });
            }

            self.scheduler.commit_decode_action(action)?;
            self.scheduler
                .active_sequence_mut(action.session_id)
                .ok_or_else(|| Error::Invariant {
                    message: format!(
                        "speculative session {:?} disappeared after anchor commit",
                        action.session_id
                    ),
                })?
                .extend_generated(&result.accepted);
            let runtime_emitted_tokens =
                self.enqueue_speculative_committed_tokens(action, &result.accepted)?;
            if result.accounting.externally_committed_tokens != runtime_emitted_tokens {
                return Err(Error::Invariant {
                    message: format!(
                        "speculative transaction committed {} tokens but invoked {runtime_emitted_tokens} runtime token callbacks",
                        result.accounting.externally_committed_tokens
                    ),
                });
            }
            externally_committed_tokens = externally_committed_tokens
                .checked_add(runtime_emitted_tokens)
                .ok_or_else(|| Error::Invariant {
                    message: "speculative external token count overflow".into(),
                })?;

            let mut finish_reason = {
                let sequence = self
                    .scheduler
                    .active_sequence(action.session_id)
                    .ok_or_else(|| Error::Invariant {
                        message: format!(
                            "speculative session {:?} disappeared before frontier staging",
                            action.session_id
                        ),
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
                            && self.executor.runner().is_eos_token(next.token_id) =>
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
        self.snapshot_transaction_outputs(transaction, externally_committed_tokens)?;
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
                .ok_or_else(|| Error::Invariant {
                    message: format!(
                        "cannot execute speculative for inactive session {:?}",
                        action.session_id
                    ),
                })?;
            if sequence.request_id != action.request_id
                || sequence.kv_handle != action.kv_handle
                || sequence.position != action.position
                || sequence.next_decode_token != Some(action.token_id)
            {
                return Err(Error::Invariant {
                    message: format!(
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
                    ),
                });
            }

            let remaining_output = sequence.max_new_tokens.saturating_sub(sequence.generated);
            let remaining_context = self.config.ctx_size.saturating_sub(sequence.position);
            let commit_capacity = remaining_output.min(remaining_context);
            if commit_capacity == 0 {
                return Err(Error::Invariant {
                    message: format!(
                        "speculative decode action for session {:?} has no output/context capacity",
                        action.session_id
                    ),
                });
            }
            let max_drafts = commit_capacity.saturating_sub(1);
            let page_slot =
                *self
                    .page_slots
                    .get(&action.session_id)
                    .ok_or_else(|| Error::Invariant {
                        message: format!(
                            "speculative session {:?} has no authoritative page slot",
                            action.session_id
                        ),
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
        let anchor_token_id = slot
            .sequence
            .next_decode_token
            .ok_or_else(|| Error::Invariant {
                message: format!(
                    "speculative sequence {:?} lost its validated anchor token",
                    slot.sequence.session_id
                ),
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
                        return Err(Error::Invariant {
                            message:
                                "speculative proposal cohort completed with an unfinished slot"
                                    .into(),
                        });
                    }
                },
                NativeProposalSlotStatus::NotStarted | NativeProposalSlotStatus::Waiting(_) => {
                    return Err(Error::Invariant {
                        message: "speculative proposal cohort completed with an unfinished slot"
                            .into(),
                    });
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
        if self.config.stop_at_eos
            && !sequence.ignore_eos
            && self.executor.runner().is_eos_token(anchor_token_id)
        {
            return Err(Error::Invariant {
                message: "an EOS token must not be staged as a speculative anchor".into(),
            });
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
            if self.config.stop_at_eos
                && !sequence.ignore_eos
                && self.executor.runner().is_eos_token(token_id)
            {
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
            .ok_or_else(|| Error::Invariant {
                message: format!(
                    "cannot emit speculative block for inactive session {:?}",
                    action.session_id
                ),
            })?;
        let start_index = sequence
            .generated
            .checked_sub(tokens.len())
            .ok_or_else(|| Error::Invariant {
                message: "speculative emitted block exceeds committed generation count".into(),
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
        let mut cleanup = Vec::new();
        let mut sessions = actions
            .iter()
            .map(|action| action.session_id)
            .collect::<Vec<_>>();
        sessions.sort_unstable_by_key(|session| session.0);
        sessions.dedup();
        for session_id in sessions {
            if self.scheduler.active_sequence(session_id).is_some()
                && let Err(source) = self
                    .scheduler
                    .fail_sequence(session_id, &mut self.slot_pool)
            {
                cleanup.push(CleanupStep::new(
                    format!("session {session_id:?} scheduler cleanup"),
                    source,
                ));
            }
            self.retained_sessions.remove(&session_id);
            if let Err(source) = self.release_sequence_state(session_id) {
                cleanup.push(CleanupStep::new(
                    format!("session {session_id:?} state cleanup"),
                    source,
                ));
            }
        }
        Error::with_cleanup_batch(stage, error, cleanup)
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

fn action_request_id(action: &SchedulerAction) -> Option<RequestId> {
    match action {
        SchedulerAction::Execute { prefills, decodes } => prefills
            .iter()
            .find_map(|action| action.request_id)
            .or_else(|| decodes.iter().find_map(|action| action.request_id)),
        SchedulerAction::PrefillChunk(prefill) => prefill.request_id,
        SchedulerAction::DecodeBatch(actions) => {
            actions.iter().find_map(|action| action.request_id)
        }
        SchedulerAction::Finish { request_id, .. } | SchedulerAction::Cancel { request_id, .. } => {
            *request_id
        }
    }
}

fn action_session_for_request(
    action: &SchedulerAction,
    request_id: RequestId,
) -> Option<SessionId> {
    match action {
        SchedulerAction::Execute { prefills, decodes } => prefills
            .iter()
            .find_map(|action| (action.request_id == Some(request_id)).then_some(action.session_id))
            .or_else(|| {
                decodes.iter().find_map(|action| {
                    (action.request_id == Some(request_id)).then_some(action.session_id)
                })
            }),
        SchedulerAction::PrefillChunk(prefill) => {
            (prefill.request_id == Some(request_id)).then_some(prefill.session_id)
        }
        SchedulerAction::DecodeBatch(actions) => actions.iter().find_map(|action| {
            (action.request_id == Some(request_id)).then_some(action.session_id)
        }),
        SchedulerAction::Finish {
            request_id: owner,
            session_id,
            ..
        }
        | SchedulerAction::Cancel {
            request_id: owner,
            session_id,
        } => (*owner == Some(request_id)).then_some(*session_id),
    }
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
        ExecutionIntent, ForwardPhase, KvElementType, KvLayoutSchema, KvPlaneDescriptor,
        LogitsOutput, LogitsRequest, LogitsRow,
    };
    use ferrule_common::{
        CompletionOutcome, ContentHash, DependencySet, Error as ModelError, ExpertId,
        FailureReason, LayerId, LoadStage, LogicalDependency, MaterializedResourceId, OperationId,
        PayloadEncodingId, ResidencyLeaseSet, Result as ModelResult, SourceGeneration,
        SourceIdentityHash,
    };
    use ferrule_model::{
        ContinuationId, MaterializationProvider, MaterializationRequest, MaterializationResolver,
        ModelInfo, ModelRunner, MultiSessionBatchProgress, MultiSessionRunner, NativeProposal,
        NativeProposalProgress, NativeProposalSource, PendingModelProgress, ResidentModelRunner,
        ResourceSource, TokenLogit, TransactionEndIntent, TransactionEndProgress,
    };

    use crate::io::testing::FakeMaterializationProvider;
    use crate::io::testing::{MockPhysicalCommand, MockPhysicalProvider};
    use crate::io::{CohortId, FairQueueConfig};
    use crate::scheduling::{
        FixedSequenceSlotPool, PhysicalResourceBroker, PhysicalResourceLimit, RequestId,
        SequenceStatus,
    };

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
                match self.step(&mut on_token)? {
                    ResidentDriverStep::Idle => return Ok(self.stats().clone()),
                    ResidentDriverStep::Executed { .. } => {}
                    ResidentDriverStep::Blocked => {
                        return Err(Error::InvalidRequest {
                            message: "test driver blocked without asynchronous progress".into(),
                        });
                    }
                    ResidentDriverStep::WaitingForModelProgress(progress) => {
                        return Err(Error::InvalidRequest {
                            message: format!(
                                "test driver requires completion owner for {} pending operation(s)",
                                progress.len()
                            ),
                        });
                    }
                }
            }
        }
    }

    #[derive(Debug)]
    struct FailOnceSequenceSlotPool {
        inner: FixedSequenceSlotPool,
        free_failures_remaining: usize,
    }

    impl FailOnceSequenceSlotPool {
        fn new(capacity: usize, free_failures: usize) -> Self {
            Self {
                inner: FixedSequenceSlotPool::new(capacity),
                free_failures_remaining: free_failures,
            }
        }

        fn active_count(&self) -> usize {
            self.inner.active_count()
        }
    }

    impl SequenceSlotPool for FailOnceSequenceSlotPool {
        fn alloc_slot(&mut self) -> Result<crate::scheduling::KvHandle> {
            self.inner.alloc_slot()
        }

        fn free_slot(&mut self, handle: crate::scheduling::KvHandle) -> Result<()> {
            if self.free_failures_remaining > 0 {
                self.free_failures_remaining -= 1;
                return Err(Error::Invariant {
                    message: "simulated sequence slot release failure".into(),
                });
            }
            self.inner.free_slot(handle)
        }
    }

    #[derive(Debug)]
    struct DriverTestKvSchema;

    static DRIVER_TEST_PLANE: KvPlaneDescriptor =
        KvPlaneDescriptor::new("test", 1, 1, KvElementType::F32);

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

    struct MockTopKRunner {
        completion_hub: ferrule_common::CompletionHub,
        position: usize,
        eos: Option<u32>,
        additional_eos: Vec<u32>,
        outputs: VecDeque<Vec<TokenLogit>>,
        fed: Vec<u32>,
        prefills: Vec<Vec<u32>>,
        fail_next_mutation: bool,
        mutation_calls: usize,
        released_sequence_states: usize,
        release_failures_remaining: usize,
        kv_release_failures_remaining: usize,
        released_kv_pages: Vec<ferrule_common::execution::KvPageId>,
        native_proposals: VecDeque<NativeProposal>,
        native_proposal_enabled: bool,
        packed_predictions: VecDeque<Vec<TokenLogit>>,
        packed_committed_calls: usize,
        packed_verification_calls: usize,
        proposal_waits: VecDeque<usize>,
        active_native_proposals: HashMap<ContinuationId, MockPendingProposal>,
        next_native_proposal_continuation: u64,
        native_proposal_begin_calls: usize,
        native_proposal_resume_calls: usize,
        native_proposal_cancel_calls: usize,
        native_proposal_resume_errors_remaining: usize,
        native_proposal_cancel_still_active_remaining: usize,
        cancelled_native_proposals: Vec<ContinuationId>,
        committed_resumable_armed: bool,
        resumable_wait_scripts: VecDeque<usize>,
        active_resumable_waits: HashMap<ExecutionTransactionId, usize>,
        resume_errors_remaining: usize,
        active_resumable_predictions: HashMap<ExecutionTransactionId, Vec<TokenLogit>>,
        resumable_cancel_still_active_remaining: usize,
        cancelled_continuations: Vec<ContinuationId>,
        reject_topology_mutation_while_packed: bool,
        topology_mutation_attempts_while_packed: usize,
        sequence_state_fork_calls: usize,
        prepared_batches: usize,
        committed_batches: usize,
        rolled_back_batches: usize,
        rollback_failures_remaining: usize,
        empty_top_k_output: bool,
        paged: bool,
        expert_residency_requirements:
            Option<ferrule_common::expert_residency::ExpertResidencyRequirements>,
        expert_residency_control_installed: bool,
        expert_residency_control_install_calls: usize,
        materialization_provider: Option<Box<dyn MaterializationProvider>>,
        materialization_provider_take_calls: usize,
        materialization_resolver: Option<Box<dyn MaterializationResolver>>,
        materialization_resolver_install_calls: usize,
        materialization_request: Option<MaterializationRequest>,
        transaction_prefetch_request: Option<MaterializationRequest>,
        warmup_requests: Vec<MaterializationRequest>,
        materialization_retention: ferrule_model::ResourceRetention,
    }

    impl std::fmt::Debug for MockTopKRunner {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("MockTopKRunner")
                .field("position", &self.position)
                .field("prepared_batches", &self.prepared_batches)
                .field("committed_batches", &self.committed_batches)
                .field("rolled_back_batches", &self.rolled_back_batches)
                .field(
                    "materialization_resolver_installed",
                    &self.materialization_resolver.is_some(),
                )
                .field(
                    "materialization_provider",
                    &self.materialization_provider.is_some(),
                )
                .finish_non_exhaustive()
        }
    }

    impl MockTopKRunner {
        fn new(outputs: Vec<Vec<TokenLogit>>) -> Self {
            Self {
                completion_hub: ferrule_common::CompletionHub::new(),
                position: 0,
                eos: None,
                additional_eos: Vec::new(),
                outputs: outputs.into(),
                fed: Vec::new(),
                prefills: Vec::new(),
                fail_next_mutation: false,
                mutation_calls: 0,
                released_sequence_states: 0,
                release_failures_remaining: 0,
                kv_release_failures_remaining: 0,
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
                rollback_failures_remaining: 0,
                empty_top_k_output: false,
                paged: false,
                expert_residency_requirements: None,
                expert_residency_control_installed: false,
                expert_residency_control_install_calls: 0,
                materialization_provider: None,
                materialization_provider_take_calls: 0,
                materialization_resolver: None,
                materialization_resolver_install_calls: 0,
                materialization_request: None,
                transaction_prefetch_request: None,
                warmup_requests: Vec::new(),
                materialization_retention: ferrule_model::ResourceRetention::ThroughStage,
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

        fn with_rollback_failures(mut self, failures: usize) -> Self {
            self.rollback_failures_remaining = failures;
            self
        }

        fn with_empty_top_k_output(mut self) -> Self {
            self.empty_top_k_output = true;
            self
        }

        fn with_materialization_request(mut self, request: MaterializationRequest) -> Self {
            self.materialization_request = Some(request);
            self
        }

        fn with_transaction_prefetch(mut self, request: MaterializationRequest) -> Self {
            self.transaction_prefetch_request = Some(request);
            self
        }

        fn with_materialization_retention(
            mut self,
            retention: ferrule_model::ResourceRetention,
        ) -> Self {
            self.materialization_retention = retention;
            self
        }

        fn with_expert_requirements(mut self) -> Self {
            self.expert_residency_requirements = Some(
                ferrule_common::expert_residency::ExpertResidencyRequirements::new(17, vec![1]),
            );
            self
        }

        fn with_materialization_provider(
            mut self,
            provider: Box<dyn MaterializationProvider>,
        ) -> Self {
            self.materialization_provider = Some(provider);
            self
        }

        fn with_warmup(mut self, request: MaterializationRequest) -> Self {
            self.warmup_requests.push(request);
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

        fn ensure_packed_topology_quiescent(&mut self, operation: &str) -> ModelResult<()> {
            if self.reject_topology_mutation_while_packed
                && !self.active_resumable_predictions.is_empty()
            {
                self.topology_mutation_attempts_while_packed += 1;
                return Err(ModelError::Execution {
                    message: format!(
                        "cannot {operation} while a mock packed transaction is outstanding"
                    ),
                });
            }
            Ok(())
        }

        fn failing_next_mutation(mut self) -> Self {
            self.fail_next_mutation = true;
            self
        }

        fn with_release_failures(mut self, failures: usize) -> Self {
            self.release_failures_remaining = failures;
            self
        }

        fn with_kv_release_failures(mut self, failures: usize) -> Self {
            self.kv_release_failures_remaining = failures;
            self
        }

        fn with_eos(mut self, eos: u32) -> Self {
            self.eos = Some(eos);
            self.additional_eos.clear();
            self
        }

        fn with_eos_tokens(mut self, eos_tokens: impl IntoIterator<Item = u32>) -> Self {
            let mut eos_tokens = eos_tokens.into_iter();
            self.eos = eos_tokens.next();
            self.additional_eos = eos_tokens.collect();
            self
        }

        fn pending_dependencies(continuation: ContinuationId) -> DependencySet {
            DependencySet::new([LogicalDependency::operation_retired(OperationId::new(
                continuation.get(),
            ))
            .unwrap()])
            .unwrap()
        }

        fn pending_resumable_progress(
            &mut self,
            transaction: ExecutionTransactionId,
        ) -> ModelResult<PendingModelProgress> {
            let continuation = ContinuationId::new(transaction.get());
            match self.materialization_request {
                Some(request) => {
                    let key = self.materialization_resolver()?.resolve(request)?;
                    Ok(materialization_progress_with_retention(
                        transaction,
                        continuation,
                        [(request, key)],
                        self.materialization_retention,
                    ))
                }
                None => PendingModelProgress::new(
                    transaction,
                    continuation,
                    Self::pending_dependencies(continuation),
                ),
            }
        }

        fn pending_native_proposal(
            &mut self,
            transaction: ExecutionTransactionId,
            continuation: ContinuationId,
        ) -> ModelResult<PendingModelProgress> {
            match self.materialization_request {
                Some(request) => {
                    let key = self.materialization_resolver()?.resolve(request)?;
                    Ok(materialization_progress_with_retention(
                        transaction,
                        continuation,
                        [(request, key)],
                        self.materialization_retention,
                    ))
                }
                None => PendingModelProgress::new(
                    transaction,
                    continuation,
                    Self::pending_dependencies(continuation),
                ),
            }
        }

        fn complete_packed_batch(
            &mut self,
            states: &mut [MockSequenceState],
            batch: &ferrule_common::execution::ExecutionBatch,
            predictions: Vec<TokenLogit>,
        ) -> ModelResult<ExecutionOutput> {
            if states.len() != batch.sequences().len() {
                return Err(ModelError::Internal {
                    message: format!(
                        "mock packed batch state/sequence mismatch: states={} sequences={}",
                        states.len(),
                        batch.sequences().len()
                    ),
                });
            }
            if predictions.len() != batch.token_ids().len() {
                return Err(ModelError::Model {
                    message: format!(
                        "mock packed predictions {} do not match {} input rows",
                        predictions.len(),
                        batch.token_ids().len()
                    ),
                });
            }
            for (state, sequence) in states.iter_mut().zip(batch.sequences()) {
                let query_start = sequence.query.start as usize;
                let query_end = sequence.query.end as usize;
                state.position = state
                    .position
                    .checked_add(query_end - query_start)
                    .ok_or_else(|| ModelError::Internal {
                        message: "mock packed position overflow".into(),
                    })?;
                state
                    .fed
                    .extend_from_slice(&batch.token_ids()[query_start..query_end]);
            }
            let logits = predictions
                .into_iter()
                .enumerate()
                .filter(|(row, _)| matches!(batch.logits()[*row], LogitsRequest::TopK(_)))
                .map(|(row, top1)| {
                    let top_k = if self.empty_top_k_output {
                        Vec::new()
                    } else {
                        vec![top1]
                    };
                    LogitsRow::new(row as u32, LogitsOutput::TopK(top_k))
                })
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

        fn encode(&self, text: &str) -> ModelResult<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(&self, tokens: &[u32]) -> ModelResult<String> {
            Ok(tokens
                .iter()
                .map(|token| char::from_u32(*token).unwrap_or('?'))
                .collect())
        }

        fn reset_session(&mut self) -> ModelResult<()> {
            self.position = 0;
            self.fed.clear();
            self.prefills.clear();
            Ok(())
        }

        fn eos_token_id(&self) -> Option<u32> {
            self.eos
        }

        fn is_eos_token(&self, token_id: u32) -> bool {
            self.eos == Some(token_id) || self.additional_eos.contains(&token_id)
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

        fn native_proposal_source(&self) -> ModelResult<Option<NativeProposalSource>> {
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
        ) -> ModelResult<NativeProposalProgress> {
            self.native_proposal_begin_calls += 1;
            let proposal = self
                .native_proposals
                .pop_front()
                .ok_or_else(|| ModelError::Model {
                    message: "mock speculative proposal queue is empty".into(),
                })?;
            let waits = self.proposal_waits.pop_front().unwrap_or(0);
            if waits == 0 {
                return Ok(NativeProposalProgress::Complete(proposal));
            }
            let continuation = ContinuationId::new(self.next_native_proposal_continuation);
            self.next_native_proposal_continuation = self
                .next_native_proposal_continuation
                .checked_add(1)
                .ok_or_else(|| ModelError::Internal {
                    message: "mock proposal continuation overflow".into(),
                })?;
            let replaced = self.active_native_proposals.insert(
                continuation,
                MockPendingProposal {
                    proposal,
                    waits_remaining: waits - 1,
                },
            );
            debug_assert!(replaced.is_none());
            Ok(NativeProposalProgress::Waiting(
                self.pending_native_proposal(transaction, continuation)?,
            ))
        }

        fn resume_native_proposal(
            &mut self,
            transaction: ExecutionTransactionId,
            continuation: ContinuationId,
            _leases: ResidencyLeaseSet,
        ) -> ModelResult<NativeProposalProgress> {
            self.native_proposal_resume_calls += 1;
            if self.native_proposal_resume_errors_remaining > 0 {
                self.native_proposal_resume_errors_remaining -= 1;
                return Err(ModelError::Model {
                    message: "simulated native proposal resume failure".into(),
                });
            }
            let mut pending = self
                .active_native_proposals
                .remove(&continuation)
                .ok_or_else(|| ModelError::Execution {
                    message: "mock received an unknown proposal continuation".into(),
                })?;
            if pending.waits_remaining > 0 {
                pending.waits_remaining -= 1;
                self.active_native_proposals.insert(continuation, pending);
                return Ok(NativeProposalProgress::Waiting(
                    self.pending_native_proposal(transaction, continuation)?,
                ));
            }
            Ok(NativeProposalProgress::Complete(pending.proposal))
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
    ) -> ModelResult<ExecutionOutput> {
        let mut output_rows = Vec::new();
        for sequence in batch.sequences() {
            let state_index =
                sequence
                    .state_slot
                    .try_as_usize()
                    .map_err(|_| ModelError::Internal {
                        message: "mock packed state slot exceeds usize".into(),
                    })?;
            let state = states
                .get_mut(state_index)
                .ok_or_else(|| ModelError::Internal {
                    message: format!("mock packed state slot {state_index} is out of range"),
                })?;
            let query_start =
                usize::try_from(sequence.query.start).map_err(|_| ModelError::Internal {
                    message: "mock packed query start exceeds usize".into(),
                })?;
            let query_end =
                usize::try_from(sequence.query.end).map_err(|_| ModelError::Internal {
                    message: "mock packed query end exceeds usize".into(),
                })?;
            let token_ids = &batch.token_ids()[query_start..query_end];
            state.position = state.position.checked_add(token_ids.len()).ok_or_else(|| {
                ModelError::Internal {
                    message: "mock packed position overflow".into(),
                }
            })?;
            match sequence.phase {
                ForwardPhase::Prefill => state.prefills.push(token_ids.to_vec()),
                ForwardPhase::Decode => state.fed.extend_from_slice(token_ids),
            }
            state.mutation_calls += 1;
            if std::mem::take(&mut state.fail_next_mutation) {
                return Err(ModelError::Model {
                    message: "simulated failure after partial runner mutation".into(),
                });
            }
            for row in query_start..query_end {
                if matches!(batch.logits()[row], LogitsRequest::TopK(_)) {
                    let logits = state.outputs.pop_front().unwrap_or_default();
                    output_rows.push(LogitsRow::new(
                        u32::try_from(row).map_err(|_| ModelError::Internal {
                            message: "mock packed row exceeds u32".into(),
                        })?,
                        LogitsOutput::TopK(logits),
                    ));
                }
            }
        }
        Ok(ExecutionOutput::new(output_rows))
    }

    impl MultiSessionRunner for MockTopKRunner {
        type SequenceState = MockSequenceState;

        fn sequence_generation(&self, _state: &Self::SequenceState) -> u64 {
            0
        }

        fn prefix_cache_plan_identity(&self) -> u64 {
            0xcafe
        }

        fn expert_residency_requirements(
            &self,
        ) -> Option<ferrule_common::expert_residency::ExpertResidencyRequirements> {
            self.expert_residency_requirements.clone()
        }

        fn expert_residency_control_installed(&self) -> bool {
            self.expert_residency_control_installed
        }

        fn install_expert_residency_control(
            &mut self,
            _control: Box<dyn ferrule_common::expert_residency::ExpertResidencyControl>,
        ) -> ModelResult<()> {
            self.expert_residency_control_install_calls += 1;
            if self.expert_residency_control_installed {
                return Err(ModelError::Execution {
                    message: "mock expert residency control is already installed".into(),
                });
            }
            self.expert_residency_control_installed = true;
            Ok(())
        }

        fn take_materialization_provider(&mut self) -> Option<Box<dyn MaterializationProvider>> {
            self.materialization_provider_take_calls += 1;
            self.materialization_provider.take()
        }

        fn take_warmup_requests(&mut self) -> ModelResult<Vec<MaterializationRequest>> {
            Ok(std::mem::take(&mut self.warmup_requests))
        }

        fn transaction_prefetch_requests(
            &self,
            _transaction: ExecutionTransactionId,
            _batch: &ferrule_common::execution::ExecutionBatch,
        ) -> ModelResult<Vec<MaterializationRequest>> {
            Ok(self.transaction_prefetch_request.into_iter().collect())
        }

        fn materialization_resolver_installed(&self) -> bool {
            self.materialization_resolver.is_some()
        }

        fn install_materialization_resolver(
            &mut self,
            resolver: Box<dyn MaterializationResolver>,
        ) -> ModelResult<()> {
            self.materialization_resolver_install_calls += 1;
            if self.materialization_resolver.is_some() {
                return Err(ModelError::Execution {
                    message: "mock materialization resolver is already installed".into(),
                });
            }
            self.materialization_resolver = Some(resolver);
            Ok(())
        }

        fn materialization_resolver(
            &mut self,
        ) -> ModelResult<&mut (dyn MaterializationResolver + '_)> {
            match self.materialization_resolver.as_mut() {
                Some(resolver) => Ok(resolver.as_mut()),
                None => Err(ModelError::Execution {
                    message: "mock materialization resolver is not installed".into(),
                }),
            }
        }

        fn with_sequence_state<T>(
            &mut self,
            state: &mut Self::SequenceState,
            execute: impl FnOnce(&mut Self) -> ModelResult<T>,
        ) -> ModelResult<T> {
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

        fn create_sequence_state(&mut self) -> ModelResult<Self::SequenceState> {
            let mut state = MockSequenceState::new(0, &self.outputs);
            state.fail_next_mutation = self.fail_next_mutation;
            Ok(state)
        }

        fn fork_sequence_state(&mut self) -> ModelResult<Self::SequenceState> {
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
        ) -> ModelResult<Self::SequenceState> {
            self.ensure_packed_topology_quiescent("fork explicit sequence state")?;
            self.sequence_state_fork_calls += 1;
            if source.position != expected_position {
                return Err(ModelError::Execution {
                    message: format!(
                        "mock fork expected position {expected_position}, source is at {}",
                        source.position
                    ),
                });
            }
            if source.fail_next_mutation {
                return Err(ModelError::Model {
                    message: "simulated model fork prepare failure".into(),
                });
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

        fn reset_sequence_state(&mut self, state: &mut Self::SequenceState) -> ModelResult<()> {
            self.ensure_packed_topology_quiescent("reset sequence state")?;
            state.position = 0;
            state.fed.clear();
            state.prefills.clear();
            state.mutation_calls = 0;
            Ok(())
        }

        fn try_release_sequence_state(
            &mut self,
            state: Self::SequenceState,
        ) -> std::result::Result<(), ferrule_model::SequenceStateReleaseError<Self::SequenceState>>
        {
            if let Err(error) = self.ensure_packed_topology_quiescent("release sequence state") {
                return Err(ferrule_model::SequenceStateReleaseError::new(error, state));
            }
            if self.release_failures_remaining > 0 {
                self.release_failures_remaining -= 1;
                return Err(ferrule_model::SequenceStateReleaseError::new(
                    ModelError::Execution {
                        message: "simulated sequence-state release failure".into(),
                    },
                    state,
                ));
            }
            self.released_sequence_states += 1;
            Ok(())
        }

        fn configure_kv_page_capacity(&mut self, _max_pages: usize) -> ModelResult<()> {
            Ok(())
        }

        fn release_kv_pages(
            &mut self,
            pages: &[ferrule_common::execution::KvPageId],
        ) -> ModelResult<()> {
            self.ensure_packed_topology_quiescent("release KV pages")?;
            if self.kv_release_failures_remaining > 0 {
                self.kv_release_failures_remaining -= 1;
                return Err(ModelError::Execution {
                    message: "simulated KV page release failure".into(),
                });
            }
            self.released_kv_pages.extend_from_slice(pages);
            Ok(())
        }

        fn preempt_kv_pages(
            &mut self,
            _pages: &[ferrule_common::execution::KvPageId],
        ) -> ModelResult<()> {
            Ok(())
        }

        fn restore_kv_pages(
            &mut self,
            _pages: &[ferrule_common::execution::KvPageId],
        ) -> ModelResult<()> {
            Ok(())
        }

        fn prepare_multi_session_batch(
            &mut self,
            _transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
            _batch: &ferrule_common::execution::ExecutionBatch,
            _kv_reservations: &[KvReservationView],
        ) -> ModelResult<()> {
            self.prepared_batches += 1;
            Ok(())
        }

        fn end_transaction(
            &mut self,
            transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
            intent: TransactionEndIntent,
        ) -> ModelResult<TransactionEndProgress> {
            match intent {
                TransactionEndIntent::Publish => {
                    self.committed_batches += 1;
                    Ok(TransactionEndProgress::Complete)
                }
                TransactionEndIntent::Abort => {
                    self.rolled_back_batches += 1;
                    if self.rollback_failures_remaining > 0 {
                        self.rollback_failures_remaining -= 1;
                        return Err(ModelError::Model {
                            message: "simulated backend rollback failure".into(),
                        });
                    }
                    if self.native_proposal_cancel_still_active_remaining > 0 {
                        self.native_proposal_cancel_calls += 1;
                        self.native_proposal_cancel_still_active_remaining -= 1;
                        return Ok(TransactionEndProgress::Pending);
                    }
                    let proposal_continuations = self
                        .active_native_proposals
                        .keys()
                        .copied()
                        .collect::<Vec<_>>();
                    if !proposal_continuations.is_empty() {
                        self.native_proposal_cancel_calls += 1;
                    }
                    for continuation in proposal_continuations {
                        self.active_native_proposals.remove(&continuation);
                        self.cancelled_native_proposals.push(continuation);
                    }
                    if self.resumable_cancel_still_active_remaining > 0 {
                        self.resumable_cancel_still_active_remaining -= 1;
                        return Ok(TransactionEndProgress::Pending);
                    }
                    if self.active_resumable_waits.remove(&transaction).is_some() {
                        self.cancelled_continuations
                            .push(ContinuationId::new(transaction.get()));
                    }
                    self.active_resumable_predictions.remove(&transaction);
                    self.committed_resumable_armed = !self.resumable_wait_scripts.is_empty();
                    Ok(TransactionEndProgress::Complete)
                }
            }
        }

        fn retain_provisional_prefixes(
            &mut self,
            _transaction: ExecutionTransactionId,
            sources: &[Self::SequenceState],
            branches: &mut [Self::SequenceState],
            executed_rows: &[usize],
            retained_rows: &[usize],
        ) -> ModelResult<()> {
            if sources.len() != branches.len()
                || sources.len() != executed_rows.len()
                || sources.len() != retained_rows.len()
            {
                return Err(ModelError::Internal {
                    message: "mock speculative provisional prefix shape mismatch".into(),
                });
            }
            for (sequence, ((source, branch), (&executed, &retained))) in sources
                .iter()
                .zip(branches.iter())
                .zip(executed_rows.iter().zip(retained_rows))
                .enumerate()
            {
                let executed_position =
                    source
                        .position
                        .checked_add(executed)
                        .ok_or_else(|| ModelError::Internal {
                            message: "mock speculative executed position overflow".into(),
                        })?;
                let executed_fed =
                    source
                        .fed
                        .len()
                        .checked_add(executed)
                        .ok_or_else(|| ModelError::Internal {
                            message: "mock speculative executed token count overflow".into(),
                        })?;
                if retained == 0
                    || retained > executed
                    || branch.position != executed_position
                    || branch.fed.len() != executed_fed
                {
                    return Err(ModelError::Internal {
                        message: format!(
                            "mock speculative invalid provisional prefix for sequence {sequence}"
                        ),
                    });
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
        ) -> ModelResult<MultiSessionBatchProgress> {
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
                let pending = self.pending_resumable_progress(transaction)?;
                return Ok(MultiSessionBatchProgress::Waiting(pending));
            }
            if batch.intent() == ExecutionIntent::ProvisionalVerification {
                self.packed_verification_calls += 1;
                let predictions =
                    self.packed_predictions
                        .pop_front()
                        .ok_or_else(|| ModelError::Model {
                            message: "mock packed prediction queue is empty".into(),
                        })?;
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
            continuation: ContinuationId,
            _leases: ResidencyLeaseSet,
        ) -> ModelResult<MultiSessionBatchProgress> {
            if continuation != ContinuationId::new(transaction.get()) {
                return Err(ModelError::Execution {
                    message: "mock received an unknown batch continuation".into(),
                });
            }
            if self.resume_errors_remaining > 0 {
                self.resume_errors_remaining -= 1;
                return Err(ModelError::Model {
                    message: "simulated resumable batch failure".into(),
                });
            }
            let waits = self
                .active_resumable_waits
                .get_mut(&transaction)
                .ok_or_else(|| ModelError::Execution {
                    message: "mock has no active batch continuation".into(),
                })?;
            if *waits > 0 {
                *waits -= 1;
                let pending = self.pending_resumable_progress(transaction)?;
                return Ok(MultiSessionBatchProgress::Waiting(pending));
            }
            self.active_resumable_waits.remove(&transaction);
            let output = match self.active_resumable_predictions.remove(&transaction) {
                Some(predictions) => self.complete_packed_batch(states, batch, predictions)?,
                None => complete_mock_committed_batch(states, batch)?,
            };
            self.committed_resumable_armed = !self.resumable_wait_scripts.is_empty();
            Ok(MultiSessionBatchProgress::Complete(output))
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

    fn materialization_request(seed: u8) -> MaterializationRequest {
        let artifact = ResourceSource::new(
            SourceIdentityHash::new([seed.max(1); 32]),
            ContentHash::new([seed.saturating_add(1).max(1); 32]),
            PayloadEncodingId::new(1),
            SourceGeneration::new(1),
        )
        .unwrap();
        MaterializationRequest::new(
            ModelInstanceId::new(17),
            artifact,
            MaterializedResourceId::routed_expert(
                LayerId::new(u32::from(seed)),
                ExpertId::new(u32::from(seed)),
            ),
            BackendId::new(4),
            DeviceId::new(2),
        )
        .unwrap()
    }

    fn materialization_progress(
        transaction: ExecutionTransactionId,
        continuation: ContinuationId,
        resources: impl IntoIterator<
            Item = (MaterializationRequest, ferrule_common::MaterializationKey),
        >,
    ) -> PendingModelProgress {
        materialization_progress_with_retention(
            transaction,
            continuation,
            resources,
            ferrule_model::ResourceRetention::ThroughStage,
        )
    }

    fn materialization_progress_with_retention(
        transaction: ExecutionTransactionId,
        continuation: ContinuationId,
        resources: impl IntoIterator<
            Item = (MaterializationRequest, ferrule_common::MaterializationKey),
        >,
        retention: ferrule_model::ResourceRetention,
    ) -> PendingModelProgress {
        let resolved = resources
            .into_iter()
            .map(|(request, key)| {
                let resource_use = ferrule_model::StageResourceUse::new(
                    request.resource(),
                    ferrule_model::ResourceAccess::Read,
                    retention,
                );
                let request =
                    ferrule_model::StageMaterializationRequest::new(resource_use, request).unwrap();
                ferrule_model::ResolvedStageResource::new(request, key).unwrap()
            })
            .collect::<Vec<_>>();
        let stage =
            ferrule_model::ResolvedStage::new(resolved, ferrule_model::WorkspaceClaim::NONE)
                .unwrap();
        PendingModelProgress::for_resolved_stage(transaction, continuation, stage).unwrap()
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
                enable_native_proposals: true,
                proposal_confidence_threshold: 0.2,
            },
        )
    }

    fn driver_with_outputs(
        outputs: Vec<Vec<TokenLogit>>,
    ) -> ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool> {
        driver_from_runner(MockTopKRunner::new(outputs))
    }

    fn prefix_cache_driver(
        outputs: Vec<Vec<TokenLogit>>,
        prefix_capacity_pages: usize,
        kv_capacity_pages: usize,
    ) -> ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool> {
        let mut runner = MockTopKRunner::new(outputs);
        runner.paged = true;
        ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(1),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 1,
                max_decode_batch: 1,
                max_batch_tokens: 8,
                allow_mixed_batches: false,
                prefix_cache_capacity_pages: prefix_capacity_pages,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(KvPageManager::new(
            Box::new(DriverTestKvSchema),
            kv_capacity_pages,
        ))
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

    fn concurrent_fake_driver(
        runner: MockTopKRunner,
    ) -> ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool> {
        let (physical, _) = MockPhysicalProvider::automatic();
        concurrent_transaction_driver(runner.with_materialization_provider(Box::new(physical)))
    }

    fn hard_resources_with_kv_capacity(capacity: u64) -> PhysicalResourceBroker {
        PhysicalResourceBroker::new(ResourceKind::ALL.map(|kind| {
            let kind_capacity = if kind == ResourceKind::KvPage {
                capacity
            } else if matches!(
                kind,
                ResourceKind::StorageReadBytes | ResourceKind::UploadBytes
            ) {
                1 << 40
            } else {
                1 << 20
            };
            PhysicalResourceLimit::new(kind, kind_capacity, 0)
        }))
        .unwrap()
    }

    fn speculative_shared_tail_credit_driver(
        kv_capacity: u64,
    ) -> (
        ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool>,
        StateSlot,
        StateSlot,
        KvPageId,
    ) {
        let mut manager = KvPageManager::new(Box::new(DriverTestKvSchema), 8);
        let source = StateSlot::new(0);
        let target = StateSlot::new(1);
        manager.alloc_sequence(source, 0).unwrap();
        let reservation = manager.reserve(source, 0, 3).unwrap();
        let source_page = reservation.view().newly_allocated[0];
        let prepared = manager
            .prepare_commit(vec![KvReservationCommit::new(reservation, 3)])
            .unwrap();
        let empty_retirement = manager.publish_commit(prepared);
        assert!(empty_retirement.is_empty());
        manager.confirm_page_retirement(empty_retirement).unwrap();
        let fork = manager
            .prepare_fork_sequence_exact(source, target, 0, 3)
            .unwrap();
        manager.publish_fork_sequence_exact(fork).unwrap();

        let mut driver = driver_from_runner(MockTopKRunner::new(Vec::new()))
            .try_with_materialization_provider(
                FakeMaterializationProvider::new(),
                hard_resources_with_kv_capacity(kv_capacity),
                FairQueueConfig::default(),
            )
            .unwrap()
            .with_page_manager(manager);
        let source_grant = driver
            .load_registry
            .acquire_hard_resources(
                1,
                crate::scheduling::ResourceDemand::required(ExecutionPhase::Decode),
                [PhysicalResourceClaim::new(ResourceKind::KvPage, 1)],
            )
            .unwrap();
        assert!(
            driver
                .kv_page_grants
                .insert(source_page, source_grant)
                .is_none()
        );
        (driver, source, target, source_page)
    }

    fn retire_test_sequence(
        driver: &mut ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool>,
        slot: StateSlot,
    ) {
        let retirement = driver
            .page_manager
            .as_mut()
            .unwrap()
            .free_sequence_pages(slot)
            .unwrap();
        driver.release_and_confirm_retirement(retirement).unwrap();
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
                enable_native_proposals: true,
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
    fn physical_bridge_driver_prepare_rejects_missing_provider() {
        let runner = MockTopKRunner::new(Vec::new()).with_expert_requirements();
        let error = match ResidentTopKDriver::try_new(runner, FixedSequenceSlotPool::new(1)) {
            Ok(_) => panic!("MoE driver preparation unexpectedly accepted a missing backend"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("provides no physical materialization provider")
        );
    }

    #[test]
    fn physical_bridge_driver_installs_expert_policy_and_materialization_provider() {
        let (physical, handle) = MockPhysicalProvider::manual();
        let runner = MockTopKRunner::new(Vec::new())
            .with_expert_requirements()
            .with_materialization_provider(Box::new(physical));
        let driver = match ResidentTopKDriver::try_new(runner, FixedSequenceSlotPool::new(1)) {
            Ok(driver) => driver,
            Err(error) => panic!("physical driver preparation failed: {error}"),
        };
        assert!(
            driver
                .executor()
                .runner()
                .expert_residency_control_installed
        );
        assert!(
            driver
                .executor()
                .runner()
                .materialization_resolver
                .is_some()
        );
        assert_eq!(
            driver
                .executor()
                .runner()
                .expert_residency_control_install_calls,
            1
        );
        assert_eq!(
            driver
                .executor()
                .runner()
                .materialization_provider_take_calls,
            1
        );
        assert_eq!(
            driver
                .executor()
                .runner()
                .materialization_resolver_install_calls,
            1
        );
        assert!(
            driver
                .executor()
                .runner()
                .materialization_provider
                .is_none()
        );
        let limits = handle.limits();
        let capacity = |kind| {
            driver
                .load_registry()
                .resources()
                .snapshots()
                .find(|snapshot| snapshot.kind == kind)
                .unwrap()
                .capacity
        };
        assert_eq!(capacity(ResourceKind::ReadSlot), limits.capacity.read_slots);
        assert_eq!(
            capacity(ResourceKind::PinnedHostBytes),
            limits.capacity.pinned_host_bytes
        );
        assert_eq!(
            capacity(ResourceKind::ResidentBytes),
            limits.capacity.device_install_bytes
        );
    }

    #[test]
    fn foreground_lifecycle_uses_the_bounded_materialization_slice() {
        let warmup = materialization_request(1);
        let (physical, _) = MockPhysicalProvider::manual();
        let runner = MockTopKRunner::new(Vec::new())
            .with_materialization_provider(Box::new(physical))
            .with_warmup(warmup);
        let mut driver =
            ResidentTopKDriver::try_new(runner, FixedSequenceSlotPool::new(1)).unwrap();

        assert!(driver.warmup_pending());
        assert_eq!(driver.materialization_transition_budget(), 4_096);
        driver.submit(request(1, &[1], 1, Vec::new()));
        assert!(driver.foreground_lifecycle_active());
        assert!(driver.warmup_pending());
        assert_eq!(driver.materialization_transition_budget(), 512);
    }

    #[test]
    fn idle_owner_fills_the_planned_residency_high_water_before_idling() {
        let (physical, _) = MockPhysicalProvider::automatic();
        let runner = MockTopKRunner::new(Vec::new())
            .with_materialization_provider(Box::new(physical))
            .with_warmup(materialization_request(1))
            .with_warmup(materialization_request(2));
        let mut driver =
            ResidentTopKDriver::try_new(runner, FixedSequenceSlotPool::new(1)).unwrap();

        let mut reached_idle = false;
        for _ in 0..16 {
            let step = driver.step(&mut |_| Ok(())).unwrap();
            if matches!(step, ResidentDriverStep::Idle) {
                assert!(!driver.warmup_pending());
                reached_idle = true;
                break;
            }
            assert!(driver.warmup_pending());
        }

        assert!(
            reached_idle,
            "idle background fill did not reach its high water"
        );
        assert_eq!(driver.load_registry().resident_entries(), 2);
        assert_eq!(driver.load_registry().active_operations(), 0);
        assert_eq!(driver.load_registry().active_prefetches(), 0);
    }

    #[test]
    fn warmup_does_not_block_request_execution_or_stop_background_io() {
        let warmup = materialization_request(1);
        let (physical, handle) = MockPhysicalProvider::manual();
        let runner = MockTopKRunner::new(vec![top(b'a' as u32)])
            .with_materialization_provider(Box::new(physical))
            .with_warmup(warmup);
        let mut driver =
            ResidentTopKDriver::try_new(runner, FixedSequenceSlotPool::new(1)).unwrap();
        driver.start_background_work().unwrap();
        driver.submit(request(1, &[1], 1, Vec::new()));

        let mut events = Vec::new();
        for _ in 0..8 {
            let step = driver
                .step(&mut |event| {
                    events.push(event.token);
                    Ok(())
                })
                .unwrap();
            if !events.is_empty() {
                assert!(matches!(step, ResidentDriverStep::Executed { .. }));
                break;
            }
        }

        assert_eq!(events, [b'a' as u32]);
        assert!(driver.warmup_pending());
        assert_eq!(driver.load_registry().active_prefetches(), 1);
        assert_eq!(driver.load_registry().active_operations(), 1);
        assert_eq!(
            handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
            1,
            "the startup wave must remain submitted while foreground execution blocks new warmup commands"
        );
    }

    #[test]
    fn warmup_failure_is_typed_and_fatal() {
        let request = materialization_request(1);
        let (physical, handle) = MockPhysicalProvider::automatic();
        handle.script_outcome(
            LoadStage::ReadSubmitted,
            CompletionOutcome::Failed(FailureReason::StorageUnavailable),
        );
        let runner = MockTopKRunner::new(Vec::new())
            .with_materialization_provider(Box::new(physical))
            .with_warmup(request);
        let mut driver =
            ResidentTopKDriver::try_new(runner, FixedSequenceSlotPool::new(1)).unwrap();

        let error = (0..16)
            .find_map(|_| driver.step(&mut |_| Ok(())).err())
            .expect("warmup failure must become terminal within the bounded fake pipeline");
        assert!(matches!(
            error,
            Error::WarmupMaterialization {
                source: FailureReason::StorageUnavailable
            }
        ));
        assert!(!driver.warmup_pending());
        assert!(driver.resident_transactions.is_empty());
        assert!(driver.speculative_transactions.is_empty());
    }

    #[test]
    fn dense_runner_can_install_and_resolve_through_materialization_provider() {
        let (physical, _) = MockPhysicalProvider::manual();
        let runner =
            MockTopKRunner::new(Vec::new()).with_materialization_provider(Box::new(physical));
        let mut driver =
            ResidentTopKDriver::try_new(runner, FixedSequenceSlotPool::new(1)).unwrap();

        assert!(
            !driver
                .executor()
                .runner()
                .expert_residency_control_installed
        );
        assert_eq!(
            driver
                .executor()
                .runner()
                .expert_residency_control_install_calls,
            0
        );
        assert!(
            driver
                .executor()
                .runner()
                .materialization_resolver
                .is_some()
        );
        assert_eq!(
            driver
                .executor()
                .runner()
                .materialization_resolver_install_calls,
            1
        );
        let base = materialization_request(1);
        let parameter_request = MaterializationRequest::new(
            base.model(),
            base.source(),
            ferrule_common::MaterializedResourceId::new(
                ferrule_common::MaterializedResourceKind::Parameter,
                0,
                0,
            ),
            base.backend(),
            base.device(),
        )
        .unwrap();
        let key = driver
            .executor
            .runner_mut()
            .materialization_resolver()
            .unwrap()
            .resolve(parameter_request)
            .unwrap();
        assert_eq!(
            key.resource().kind(),
            ferrule_common::MaterializedResourceKind::Parameter
        );
    }

    #[test]
    fn transaction_prefetch_submits_io_without_a_waiter_and_drains_after_terminal() {
        let request = materialization_request(7);
        let (physical, handle) = MockPhysicalProvider::manual();
        let runner = MockTopKRunner::new(Vec::new())
            .with_transaction_prefetch(request)
            .with_materialization_provider(Box::new(physical));
        let mut driver =
            ResidentTopKDriver::try_new(runner, FixedSequenceSlotPool::new(1)).unwrap();
        let transaction = ExecutionTransactionId::new(17).unwrap();
        let batch = ferrule_common::execution::ExecutionBatch::new(
            ferrule_common::execution::ForwardMode::Prefill,
            vec![11],
            vec![0],
            vec![None],
            vec![ferrule_common::execution::LogitsRequest::None],
            vec![ferrule_common::execution::ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Prefill,
                0..1,
                0,
                1,
                0..0,
            )],
            Vec::new(),
        );
        let phases = ExecutionPhaseSet::one(ExecutionPhase::Prefill);

        driver
            .declare_transaction_prefetch(
                transaction,
                crate::scheduling::ResourceDemand::required_phases(phases),
                &batch,
            )
            .unwrap();

        assert_eq!(driver.load_registry().active_prefetches(), 1);
        assert!(driver.load_registry().waiters().is_empty());
        assert_eq!(
            driver
                .load_registry()
                .resources()
                .in_use(ResourceKind::Continuation),
            0
        );
        assert_eq!(
            handle.command_count(|command| matches!(command, MockPhysicalCommand::Prepare(_))),
            1
        );
        assert_eq!(
            handle.command_count(|command| matches!(command, MockPhysicalCommand::Reserve(..))),
            1
        );
        assert_eq!(
            handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
            1
        );
        let operation = driver
            .load_registry()
            .prefetch_operations(crate::io::PrefetchOwner::transaction(transaction, phases))
            .next()
            .unwrap();
        assert_eq!(
            driver
                .load_registry()
                .operation(operation)
                .unwrap()
                .demand(),
            crate::scheduling::ResourceDemand::prefetch_phases(phases)
        );

        let now_ns = driver.runtime_now_ns();
        driver
            .load_registry
            .finish_transaction_custody(transaction, TransactionCustodyOutcome::RolledBack, now_ns)
            .unwrap();
        assert_eq!(driver.load_registry().active_prefetches(), 0);
        assert!(driver.load_registry().operation(operation).is_some());
        assert_eq!(
            handle.command_count(|command| matches!(command, MockPhysicalCommand::Cancel(..))),
            1
        );

        driver.load_registry.collect_provider_completions(1);
        driver.load_registry.process_one_completion().unwrap();
        assert!(driver.load_registry().operation(operation).is_none());
        assert_eq!(driver.load_registry().resources().active_grants(), 0);
    }

    #[test]
    fn driver_prefetch_is_explicit_and_creates_no_execution_owner() {
        let (physical, handle) = MockPhysicalProvider::manual();
        let runner =
            MockTopKRunner::new(Vec::new()).with_materialization_provider(Box::new(physical));
        let mut driver =
            ResidentTopKDriver::try_new(runner, FixedSequenceSlotPool::new(1)).unwrap();
        let owner = crate::io::PrefetchOwner::external(std::num::NonZeroU64::new(1).unwrap());

        let report = driver
            .prefetch(
                owner,
                crate::scheduling::ResourceDemand::prefetch(ExecutionPhase::Prefill),
                [materialization_request(1)],
            )
            .unwrap();
        assert_eq!(report.created.len(), 1);
        assert!(driver.has_pending_async_work());
        assert_eq!(driver.load_registry().active_prefetches(), 1);
        assert!(driver.load_registry().waiters().is_empty());
        assert_eq!(
            driver
                .load_registry()
                .transaction_operations(ExecutionTransactionId::new(1).unwrap())
                .count(),
            0
        );
        assert_eq!(
            handle.command_count(|command| matches!(command, MockPhysicalCommand::Prepare(_))),
            1
        );
        assert_eq!(
            handle.command_count(|command| matches!(
                command,
                MockPhysicalCommand::PromoteToExecution(_)
            )),
            0
        );

        driver.cancel_prefetch(owner).unwrap();
        assert_eq!(driver.load_registry().active_prefetches(), 0);
        assert_eq!(driver.load_registry().active_operations(), 0);
        assert!(!driver.has_pending_async_work());
    }

    #[test]
    fn physical_bridge_attach_accepts_large_resource_and_wakes_owner() {
        const EXPERT_BYTES: u64 = 13_369_344;

        let (physical, handle) = MockPhysicalProvider::manual();
        let limits = ferrule_common::materialization_io::MaterializationResourceLimits {
            capacity: ferrule_common::materialization_io::MaterializationResourceRequirements {
                read_slots: 1,
                storage_read_bytes: EXPERT_BYTES,
                pinned_host_bytes: EXPERT_BYTES,
                upload_slots: 1,
                h2d_bytes: EXPERT_BYTES,
                install_slots: 1,
                device_install_bytes: EXPERT_BYTES,
            },
            execution_reserve:
                ferrule_common::materialization_io::MaterializationResourceRequirements::default(),
        }
        .validate()
        .unwrap();
        handle.set_bytes_and_limits(EXPERT_BYTES, limits);
        let runner =
            MockTopKRunner::new(Vec::new()).with_materialization_provider(Box::new(physical));
        let mut driver =
            ResidentTopKDriver::try_new(runner, FixedSequenceSlotPool::new(1)).unwrap();
        let key = driver
            .executor
            .runner_mut()
            .materialization_resolver()
            .unwrap()
            .resolve(materialization_request(1))
            .unwrap();
        let progress = materialization_progress(
            ExecutionTransactionId::new(1).unwrap(),
            ContinuationId::new(1),
            [(materialization_request(1), key)],
        );

        let mut listener = driver.completion_hub().listen();
        driver
            .register_pending_progress(
                &progress,
                crate::scheduling::ResourceDemand::required(ExecutionPhase::Prefill),
            )
            .unwrap();
        assert_eq!(driver.load_registry().stats().operations_created, 1);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(matches!(
            std::future::Future::poll(std::pin::Pin::new(&mut listener), &mut context),
            std::task::Poll::Ready(ferrule_common::CompletionWake::Progress(_))
        ));
    }

    #[test]
    fn physical_bridge_dense_driver_defaults_to_unavailable_provider() {
        let driver = match ResidentTopKDriver::try_new(
            MockTopKRunner::new(Vec::new()),
            FixedSequenceSlotPool::new(1),
        ) {
            Ok(driver) => driver,
            Err(error) => panic!("dense driver preparation failed: {error}"),
        };
        assert!(matches!(
            driver.load_registry().prepare_execution_request(
                materialization_request(1)
                    .materialization_key(ferrule_common::DestinationGeneration::new(1))
                    .unwrap(),
                crate::scheduling::ResourceDemand::required(ExecutionPhase::Decode),
                ferrule_model::ResourceRetention::ThroughStage,
            ),
            Err(crate::io::RegistryError::Provider {
                source: ferrule_common::FailureReason::DeviceUnavailable
            })
        ));
    }

    #[test]
    fn physical_bridge_real_and_runtime_limits_map_without_testing_defaults() {
        let (_, handle) = MockPhysicalProvider::manual();
        let physical = handle.limits();
        let runtime = ResidentRuntimeResourceLimits {
            arena_slots: 2,
            kv_pages: 3,
            continuations: 4,
            waiters: 5,
            load_operations: 3,
            ready_cohorts: 6,
        };
        let topology = PhysicalMaterializationTopology::new(
            physical,
            physical.capacity.device_install_bytes,
            physical.capacity.install_slots,
        )
        .unwrap();
        let broker = physical_materialization_resources(topology, runtime).unwrap();
        let snapshot = |kind| {
            broker
                .snapshots()
                .find(|snapshot| snapshot.kind == kind)
                .unwrap()
        };
        assert_eq!(
            snapshot(ResourceKind::ReadSlot).capacity,
            physical.capacity.read_slots
        );
        assert_eq!(
            snapshot(ResourceKind::PinnedHostBytes).capacity,
            physical.capacity.pinned_host_bytes
        );
        assert_eq!(
            snapshot(ResourceKind::StorageReadBytes).capacity,
            physical.capacity.storage_read_bytes
        );
        assert_eq!(
            snapshot(ResourceKind::UploadSlot).capacity,
            physical.capacity.upload_slots
        );
        assert_eq!(
            snapshot(ResourceKind::UploadBytes).capacity,
            physical.capacity.h2d_bytes
        );
        assert_eq!(
            snapshot(ResourceKind::ResidentBytes).capacity,
            physical.capacity.device_install_bytes
        );
        assert_eq!(
            snapshot(ResourceKind::ResidencyLease).capacity,
            physical.capacity.install_slots * runtime.continuations
        );
        assert_eq!(snapshot(ResourceKind::Arena).capacity, 2);
        assert_eq!(snapshot(ResourceKind::KvPage).capacity, 3);
        assert_eq!(snapshot(ResourceKind::Continuation).capacity, 4);
        assert_eq!(snapshot(ResourceKind::Waiter).capacity, 5);
        assert_eq!(snapshot(ResourceKind::LoadOperation).capacity, 3);
        assert_eq!(snapshot(ResourceKind::ReadyCohort).capacity, 6);
    }

    #[test]
    fn physical_bridge_continuation_rejects_unresolved_synthesized_key() {
        let mut driver = match ResidentTopKDriver::try_new(
            MockTopKRunner::new(Vec::new()),
            FixedSequenceSlotPool::new(1),
        ) {
            Ok(driver) => driver,
            Err(error) => panic!("dense driver preparation failed: {error}"),
        };
        let transaction = ExecutionTransactionId::new(1).unwrap();
        let continuation = ContinuationId::new(1);
        let key = materialization_request(1)
            .materialization_key(ferrule_common::DestinationGeneration::new(1))
            .unwrap();
        let progress = materialization_progress(
            transaction,
            continuation,
            [(materialization_request(1), key)],
        );
        let error = driver
            .register_pending_progress(
                &progress,
                crate::scheduling::ResourceDemand::required(ExecutionPhase::Decode),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Registry { source }
                if matches!(
                    *source,
                    crate::io::RegistryError::Provider {
                        source: ferrule_common::FailureReason::DeviceUnavailable,
                    }
                )
        ));
        assert!(driver.continuations.is_empty());
        assert_eq!(driver.load_registry().active_operations(), 0);
    }

    #[test]
    fn driver_fake_provider_c2_waiting_a_does_not_block_b() {
        let runner = MockTopKRunner::new(vec![top(10)])
            .with_resumable_wait_scripts([1, 0])
            .with_materialization_request(materialization_request(1));
        let mut driver = concurrent_fake_driver(runner);
        for id in [1, 2] {
            let mut submitted = request(id, &[id as u32], 2, Vec::new());
            submitted.session_id = Some(SessionId(id));
            driver.submit(submitted);
        }

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
        assert_eq!(driver.resident_transactions.len(), 1);
        assert!(driver.session_owner.contains_key(&SessionId(1)));
        assert!(driver.sequence_states.contains_key(&SessionId(2)));
        assert_eq!(driver.executor().runner().committed_batches, 1);
        assert_eq!(driver.load_registry().stats().operations_created, 1);

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
        assert!(driver.resident_transactions.is_empty());
        assert_eq!(driver.executor().runner().committed_batches, 2);
    }

    #[test]
    fn materialization_failure_aborts_waiting_transaction_before_reporting_error() {
        let materialization = materialization_request(7);
        let (physical, handle) = MockPhysicalProvider::automatic();
        handle.script_outcome(
            LoadStage::ReadSubmitted,
            CompletionOutcome::Failed(FailureReason::StorageUnavailable),
        );
        let runner = MockTopKRunner::new(Vec::new())
            .with_resumable_wait_scripts([1])
            .with_materialization_request(materialization)
            .with_materialization_provider(Box::new(physical));
        let mut driver = concurrent_transaction_driver(runner);
        let mut submitted = request(1, &[1], 2, Vec::new());
        submitted.session_id = Some(SessionId(1));
        driver.submit(submitted);

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert_eq!(driver.resident_transactions.len(), 1);
        assert_eq!(driver.continuations.len(), 1);
        assert_eq!(driver.executor().runner().rolled_back_batches, 0);

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("materialization for continuation")
        );
        assert!(error.to_string().contains("StorageUnavailable"));
        assert!(driver.resident_transactions.is_empty());
        assert!(driver.continuations.is_empty());
        assert!(driver.transaction_continuations.is_empty());
        assert!(driver.session_owner.is_empty());
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert_eq!(driver.scheduler().failed_len(), 1);
        for kind in [
            ResourceKind::Continuation,
            ResourceKind::Waiter,
            ResourceKind::Arena,
            ResourceKind::ResidencyLease,
        ] {
            assert_eq!(driver.load_registry().resources().in_use(kind), 0);
        }
    }

    #[test]
    fn driver_compute_spans_cover_overlapping_materialization_wait() {
        // Session 1's decode suspends on one expert load while session 2's
        // prefill executes: the session 2 compute span must cover part of the
        // session 1 dependency wait in the critical-path ledger.
        let runner = MockTopKRunner::new(vec![top(10)])
            .with_resumable_wait_scripts([0, 1, 0])
            .with_materialization_request(materialization_request(1));
        let mut driver = concurrent_fake_driver(runner);
        for id in [1, 2] {
            let mut submitted = request(id, &[id as u32], 2, Vec::new());
            submitted.session_id = Some(SessionId(id));
            driver.submit(submitted);
        }

        // Session 1 prefills, then its decode suspends inside the same step
        // that executes session 2's prefill.
        for _ in 0..2 {
            assert!(matches!(
                driver.step(&mut |_| Ok(())).unwrap(),
                ResidentDriverStep::Executed {
                    action_kind: ResidentActionKind::Prefill,
                    ..
                }
            ));
        }
        assert_eq!(driver.resident_transactions.len(), 1);
        // Session 1 resumes once the load resolves and commits first; session
        // 2's decode commits afterwards.
        for _ in 0..2 {
            assert!(matches!(
                driver.step(&mut |_| Ok(())).unwrap(),
                ResidentDriverStep::Executed {
                    action_kind: ResidentActionKind::Decode,
                    ..
                }
            ));
        }
        assert!(driver.resident_transactions.is_empty());

        // The first committed decode token belongs to session 1, which waited
        // while session 2 executed: part of that wait must be covered.
        let delayed = driver
            .load_registry()
            .ledger()
            .output(OutputTokenId::new(1))
            .expect("first committed decode token has a snapshot");
        assert!(delayed.wait_ns > 0);
        assert!(delayed.covered_wait_ns > 0);
        assert!(delayed.covered_wait_ns < delayed.wait_ns);
        assert_eq!(
            delayed.covered_wait_ns + delayed.uncovered_wait_ns,
            delayed.wait_ns
        );

        // Session 2's decode transaction never waited on materialization.
        let undelayed = driver
            .load_registry()
            .ledger()
            .output(OutputTokenId::new(2))
            .expect("second committed decode token has a snapshot");
        assert_eq!(undelayed.wait_ns, 0);
        assert_eq!(undelayed.covered_wait_ns, 0);
    }

    #[test]
    fn driver_fake_provider_c4_same_key_single_flight_and_shutdown_no_leak() {
        let runner = MockTopKRunner::new(vec![top(10)])
            .with_resumable_wait_scripts([1, 1])
            .with_materialization_request(materialization_request(2));
        let (physical, handle) = MockPhysicalProvider::automatic();
        let mut driver =
            concurrent_transaction_driver(runner.with_materialization_provider(Box::new(physical)));
        for id in [1, 2] {
            let mut submitted = request(id, &[id as u32], 2, Vec::new());
            submitted.session_id = Some(SessionId(id));
            driver.submit(submitted);
        }

        assert!(matches!(
            driver
                .step(&mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(ref pending) if pending.len() == 2
        ));
        assert_eq!(driver.load_registry().stats().operations_created, 1);
        assert_eq!(driver.load_registry().stats().single_flight_joins, 1);
        assert_eq!(
            handle.command_count(|command| matches!(
                command,
                crate::io::testing::MockPhysicalCommand::Prepare(_)
            )),
            2
        );

        for _ in 0..2 {
            assert!(matches!(
                driver.step(&mut |_| Ok(())).unwrap(),
                ResidentDriverStep::Executed {
                    action_kind: ResidentActionKind::Prefill,
                    ..
                }
            ));
        }
        assert_eq!(
            handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
            1
        );
        let key = materialization_request(2)
            .materialization_key(ferrule_common::DestinationGeneration::new(37))
            .unwrap();
        assert_eq!(
            driver.load_registry().residency_binding(key),
            Some(handle.binding(key))
        );

        let report = driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert!(report.registry.drained);
        assert_eq!(report.registry.active_grants, 0);
        assert_eq!(report.executor_transactions, 0);
        assert_eq!(report.kv_page_grants, 0);
        assert_eq!(report.pending_kv_retirements, 0);
    }

    #[test]
    fn terminal_failed_stale_and_cancelled_continuations_release_resident_siblings() {
        let outcomes = [
            ferrule_common::CompletionOutcome::Failed(
                ferrule_common::FailureReason::StorageUnavailable,
            ),
            ferrule_common::CompletionOutcome::Stale(
                ferrule_common::StaleReason::SourceIdentityChanged,
            ),
            ferrule_common::CompletionOutcome::Cancelled(CancellationReason::ExternalRequest),
        ];
        for (index, outcome) in outcomes.into_iter().enumerate() {
            let (physical, handle) = MockPhysicalProvider::automatic();
            handle.set_resident(true);
            let runner =
                MockTopKRunner::new(Vec::new()).with_materialization_provider(Box::new(physical));
            let mut driver = concurrent_transaction_driver(runner);
            let resident_request = materialization_request(10 + index as u8);
            let resident_key = driver
                .executor
                .runner_mut()
                .materialization_resolver()
                .unwrap()
                .resolve(resident_request)
                .unwrap();
            handle.set_resident(false);
            let failing_request = materialization_request(20 + index as u8);
            let failing_key = driver
                .executor
                .runner_mut()
                .materialization_resolver()
                .unwrap()
                .resolve(failing_request)
                .unwrap();
            handle.script_outcome(ferrule_common::LoadStage::ReadSubmitted, outcome);
            let continuations = [
                ContinuationId::new(100 + (index as u64 * 2)),
                ContinuationId::new(101 + (index as u64 * 2)),
            ];
            for continuation in continuations {
                let transaction = ExecutionTransactionId::new(continuation.get()).unwrap();
                let progress = materialization_progress(
                    transaction,
                    continuation,
                    [
                        (resident_request, resident_key),
                        (failing_request, failing_key),
                    ],
                );
                driver
                    .register_pending_progress(
                        &progress,
                        crate::scheduling::ResourceDemand::required(ExecutionPhase::Decode),
                    )
                    .unwrap();
            }

            let error = driver.progress_materialization().unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("materialization for continuation")
            );
            assert!(
                continuations
                    .iter()
                    .all(|continuation| !driver.continuations.contains_key(continuation))
            );
            assert!(driver.pending_materialization_failures.is_empty());
            assert_eq!(
                handle.command_count(|command| matches!(
                    command,
                    MockPhysicalCommand::ReleaseExecutionLease(key) if *key == resident_key
                )),
                1
            );
            let retried = driver
                .executor
                .runner_mut()
                .materialization_resolver()
                .unwrap()
                .resolve(failing_request)
                .unwrap();
            assert_eq!(retried, failing_key);
            assert_eq!(
                handle.command_count(|command| matches!(
                    command,
                    MockPhysicalCommand::Prepare(request) if *request == failing_request
                )),
                2
            );
        }
    }

    #[test]
    fn terminal_cleanup_release_failure_is_retained_and_retried_before_business_error() {
        let (physical, handle) = MockPhysicalProvider::automatic();
        handle.set_resident(true);
        let runner =
            MockTopKRunner::new(Vec::new()).with_materialization_provider(Box::new(physical));
        let mut driver = concurrent_transaction_driver(runner);
        let resident_key = driver
            .executor
            .runner_mut()
            .materialization_resolver()
            .unwrap()
            .resolve(materialization_request(30))
            .unwrap();
        handle.set_resident(false);
        let failing_key = driver
            .executor
            .runner_mut()
            .materialization_resolver()
            .unwrap()
            .resolve(materialization_request(31))
            .unwrap();
        handle.script_outcome(
            ferrule_common::LoadStage::ReadSubmitted,
            ferrule_common::CompletionOutcome::Failed(
                ferrule_common::FailureReason::StorageUnavailable,
            ),
        );
        handle.fail_next_release(ferrule_common::FailureReason::DeviceUnavailable);
        let transaction = ExecutionTransactionId::new(200).unwrap();
        let continuation = ContinuationId::new(200);
        let progress = materialization_progress(
            transaction,
            continuation,
            [
                (materialization_request(30), resident_key),
                (materialization_request(31), failing_key),
            ],
        );
        driver
            .register_pending_progress(
                &progress,
                crate::scheduling::ResourceDemand::required(ExecutionPhase::Decode),
            )
            .unwrap();

        let cleanup_error = driver.progress_materialization().unwrap_err();
        assert!(matches!(
            cleanup_error,
            Error::Registry { source }
                if matches!(
                    *source,
                    crate::io::RegistryError::Provider {
                        source: ferrule_common::FailureReason::DeviceUnavailable,
                    }
                )
        ));
        assert!(driver.continuations.contains_key(&continuation));
        assert!(driver.pending_materialization_failures.is_empty());
        assert_eq!(
            driver.load_registry().residency_binding(resident_key),
            Some(handle.binding(resident_key))
        );
        assert!(
            driver
                .load_registry()
                .residency_binding(failing_key)
                .is_none()
        );

        let business_error = driver.progress_materialization().unwrap_err();
        assert!(
            business_error
                .to_string()
                .contains("materialization for continuation")
        );
        assert!(!driver.continuations.contains_key(&continuation));
        assert!(driver.pending_materialization_failures.is_empty());
        assert_eq!(
            handle.command_count(|command| matches!(
                command,
                MockPhysicalCommand::ReleaseExecutionLease(key) if *key == resident_key
            )),
            2
        );
    }

    #[test]
    fn driver_shutdown_cancels_waiters_drains_submitted_ops_and_releases_all_ownership() {
        let runner = MockTopKRunner::new(Vec::new())
            .with_resumable_wait_scripts([2])
            .with_materialization_request(materialization_request(3));
        let (physical, handle) = MockPhysicalProvider::automatic();
        handle.lose_next(ferrule_common::io_protocol::LoadStage::ReadSubmitted);
        let runner = runner.with_materialization_provider(Box::new(physical));
        let mut driver = concurrent_transaction_driver(runner);
        let mut submitted = request(1, &[1], 2, Vec::new());
        submitted.session_id = Some(SessionId(1));
        driver.submit(submitted);

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert_eq!(driver.load_registry().pending_physical_operations(), 1);

        let report = driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert!(report.registry.drained);
        assert_eq!(report.registry.active_grants, 0);
        assert_eq!(report.executor_transactions, 0);
        assert_eq!(report.kv_page_grants, 0);
        assert_eq!(report.pending_kv_retirements, 0);
        assert_eq!(driver.load_registry().stats().cancellations_requested, 1);
        assert!(driver.load_registry().waiters().is_empty());
        assert!(driver.continuations.is_empty());
        assert!(driver.transaction_continuations.is_empty());
        assert!(driver.session_owner.is_empty());
        assert!(driver.pending_sequence_cleanups.is_empty());
        assert!(driver.scheduler().is_idle());
        assert_eq!(driver.load_registry().resident_entries(), 0);
        assert_eq!(driver.load_registry().active_operations(), 0);
        assert_eq!(
            handle.command_count(|command| matches!(command, MockPhysicalCommand::PhysicalDropped)),
            0
        );

        let runner = match driver.try_into_runner() {
            Ok(runner) => runner,
            Err(failure) => panic!("drained driver did not extract its runner: {}", failure.0),
        };
        assert_eq!(
            handle.command_count(|command| matches!(command, MockPhysicalCommand::PhysicalDropped)),
            0
        );
        drop(runner);
        let commands = handle.commands();
        let cancel = commands
            .iter()
            .position(|command| matches!(command, MockPhysicalCommand::Cancel(..)))
            .expect("registry shutdown must cancel submitted physical work");
        let dropped = commands
            .iter()
            .position(|command| matches!(command, MockPhysicalCommand::PhysicalDropped))
            .expect("physical authority must be dropped with the extracted runner");
        assert!(cancel < dropped);
    }

    #[test]
    fn speculative_kv_cow_reservation_and_retirement_release_exact_hard_credit() {
        let (mut driver, source, target, source_page) = speculative_shared_tail_credit_driver(2);
        assert_eq!(
            driver
                .load_registry()
                .resources()
                .in_use(ResourceKind::KvPage),
            1
        );

        let proposal = [];
        let item = SpeculativeVerificationItem {
            state_slot: target,
            generation: 0,
            proposal: &proposal,
            frontier: TargetFrontier {
                position: 3,
                top1: TokenLogit::new(9, 1.0),
            },
        };
        let reservations = driver
            .reserve_speculative_pages(ExecutionTransactionId::new(90).unwrap(), &[item])
            .unwrap();
        let cow = reservations[0]
            .view()
            .cow_replacement
            .expect("shared partial tail must reserve a COW replacement");
        assert_eq!(cow.source, source_page);
        assert_ne!(cow.replacement, source_page);
        assert_eq!(driver.kv_page_grants.len(), 2);
        assert_eq!(
            driver
                .load_registry()
                .resources()
                .in_use(ResourceKind::KvPage),
            2
        );

        driver
            .abort_quiesced_resident_kv(PendingResidentKv::Reserved(reservations))
            .unwrap();
        assert!(driver.kv_page_grants.contains_key(&source_page));
        assert!(!driver.kv_page_grants.contains_key(&cow.replacement));
        assert_eq!(driver.kv_page_grants.len(), 1);
        assert_eq!(
            driver
                .load_registry()
                .resources()
                .in_use(ResourceKind::KvPage),
            1
        );
        assert_eq!(driver.page_manager().unwrap().stats().retiring_pages, 0);

        retire_test_sequence(&mut driver, target);
        assert_eq!(driver.kv_page_grants.len(), 1);
        retire_test_sequence(&mut driver, source);
        assert!(driver.kv_page_grants.is_empty());
        assert_eq!(
            driver
                .load_registry()
                .resources()
                .in_use(ResourceKind::KvPage),
            0
        );
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
    }

    #[test]
    fn speculative_kv_credit_exhaustion_is_failure_atomic_and_leak_free() {
        let (mut driver, source, target, source_page) = speculative_shared_tail_credit_driver(1);
        let proposal = [];
        let item = SpeculativeVerificationItem {
            state_slot: target,
            generation: 0,
            proposal: &proposal,
            frontier: TargetFrontier {
                position: 3,
                top1: TokenLogit::new(9, 1.0),
            },
        };

        let error = driver
            .reserve_speculative_pages(ExecutionTransactionId::new(91).unwrap(), &[item])
            .unwrap_err();
        assert!(error.to_string().contains("KvPage"));
        assert_eq!(driver.kv_page_grants.len(), 1);
        assert!(driver.kv_page_grants.contains_key(&source_page));
        assert_eq!(
            driver
                .page_manager()
                .unwrap()
                .required_physical_pages(target, 0, 1)
                .unwrap(),
            1
        );
        assert_eq!(
            driver
                .load_registry()
                .resources()
                .in_use(ResourceKind::KvPage),
            1
        );
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 1);

        retire_test_sequence(&mut driver, target);
        retire_test_sequence(&mut driver, source);
        assert!(driver.kv_page_grants.is_empty());
        assert_eq!(
            driver
                .load_registry()
                .resources()
                .in_use(ResourceKind::KvPage),
            0
        );
    }

    #[test]
    fn resident_external_commit_creates_one_snapshot_per_output_token() {
        let mut driver = driver_with_outputs(vec![top(65)]);
        driver.submit(request(1, &[1], 1, Vec::new()));
        let mut events = Vec::new();
        driver
            .drive_ready_test_work(|event| {
                events.push(event.clone());
                Ok(())
            })
            .unwrap();
        assert_eq!(events.len(), 1);
        let snapshot = driver
            .load_registry()
            .ledger()
            .output(OutputTokenId::new(1))
            .unwrap();
        assert!(snapshot.cohort_phases.contains_key(&CohortId::new(2)));
        assert_eq!(snapshot.externally_committed_tokens, 1);
        assert!(
            driver
                .load_registry()
                .ledger()
                .output(OutputTokenId::new(2))
                .is_none()
        );
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
    fn policy_can_disable_native_proposal_and_resume_packed_target_execution() {
        let mut runner = MockTopKRunner::new(vec![top(10)]).with_resumable_wait_scripts([1, 1]);
        runner.native_proposal_enabled = true;
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
                enable_native_proposals: false,
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
            driver.step(&mut |event| {
                events.push(event.clone());
                Ok(())
            })
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
    fn production_speculative_stops_at_secondary_eos_without_emitting_it() {
        let secondary_eos = 3;
        let runner = MockTopKRunner::new(vec![top(10)])
            .with_eos_tokens([2, secondary_eos])
            .with_speculative_cycle(
                NativeProposal {
                    token_ids: vec![11, secondary_eos],
                    confidence_logits: vec![1.0, 1.0],
                },
                vec![
                    TokenLogit::new(11, 9.0),
                    TokenLogit::new(secondary_eos, 8.0),
                ],
            );
        let mut driver = speculative_driver_from_runner(runner, 1);
        let mut submitted = request(77, &[1], 3, Vec::new());
        submitted.session_id = Some(SessionId(77));
        driver.submit(submitted);
        let mut events = Vec::new();

        driver
            .drive_ready_test_work(|event| {
                events.push(event.clone());
                Ok(())
            })
            .unwrap();

        assert_eq!(
            events.iter().map(|event| event.token).collect::<Vec<_>>(),
            vec![10, 11]
        );
        assert!(!events.iter().any(|event| event.token == secondary_eos));
        assert_eq!(driver.executor().runner().packed_verification_calls, 1);
        assert_eq!(driver.stats().speculative.proposed_tokens, 1);
        assert_eq!(driver.stats().speculative.accepted_draft_tokens, 1);
        assert_eq!(driver.stats().speculative.runtime_emitted_tokens, 2);

        let finished = driver.drain_finished();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].finish_reason, Some(SequenceFinishReason::Eos));
        assert_eq!(finished[0].generated, 2);
        assert_eq!(finished[0].tokens, vec![1, 10, 11]);
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
                enable_native_proposals: true,
                proposal_confidence_threshold: 0.2,
            },
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        let mut request = request(77, &[1], 3, Vec::new());
        request.session_id = Some(SessionId(77));
        driver.submit(request);
        let mut events = Vec::new();

        let prefill = driver
            .step(&mut |event| {
                events.push(event.clone());
                Ok(())
            })
            .unwrap();
        assert!(matches!(
            prefill,
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));

        let decode = driver
            .step(&mut |event| {
                events.push(event.clone());
                Ok(())
            })
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
        let snapshot = driver
            .load_registry()
            .ledger()
            .output(OutputTokenId::new(1))
            .unwrap();
        assert!(snapshot.cohort_phases.contains_key(&CohortId::new(2)));
        assert_eq!(snapshot.externally_committed_tokens, 1);
        assert!(
            driver
                .load_registry()
                .ledger()
                .output(OutputTokenId::new(2))
                .is_none()
        );
        assert_eq!(
            driver
                .load_registry()
                .resources()
                .in_use(ResourceKind::KvPage),
            driver.page_manager().unwrap().allocated_pages() as u64
        );
        assert!(
            driver
                .load_registry()
                .resources()
                .in_use(ResourceKind::KvPage)
                > 0
        );
        let report = driver.shutdown(&mut |_| Ok(()), 0).unwrap();
        assert_eq!(report.kv_page_grants, 0);
        assert_eq!(report.registry.active_grants, 0);
    }

    #[test]
    fn speculative_post_commit_source_release_failure_retries_without_recommit() {
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
            .with_release_failures(1);
        let mut driver = speculative_driver_from_runner(runner, 1);
        let actions = ready_speculative_decode_actions(&mut driver, &[1]);

        let error = driver
            .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("simulated sequence-state release failure")
        );
        let Some(PendingSpeculativeDriverCohort::Ending(pending)) =
            driver.speculative_transactions.values().next()
        else {
            panic!("post-commit source release failure must retain the ending cohort");
        };
        assert!(matches!(
            pending.ending,
            SpeculativeEnding::BackendCommittedPendingPublish { .. }
        ));
        assert_eq!(pending.source_states.len(), 1);
        assert_eq!(pending.schedules.len(), 1);
        assert!(driver.session_owner.contains_key(&SessionId(1)));
        let committed_batches = driver.executor().runner().committed_batches;

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            }
        ));
        assert!(driver.speculative_transactions.is_empty());
        assert!(driver.session_owner.is_empty());
        assert_eq!(
            driver.executor().runner().committed_batches,
            committed_batches
        );
        assert_eq!(driver.executor().runner().released_sequence_states, 1);
    }

    #[test]
    fn speculative_post_abort_branch_release_failure_retries_without_reabort() {
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
            .with_resumable_wait_scripts([0, 1])
            .with_resume_errors(1)
            .with_release_failures(1);
        let mut driver = speculative_driver_from_runner(runner, 1);
        let actions = ready_speculative_decode_actions(&mut driver, &[1]);
        assert!(matches!(
            driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));

        let cleanup_error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            cleanup_error
                .to_string()
                .contains("simulated sequence-state release failure")
        );
        let Some(PendingSpeculativeDriverCohort::Ending(pending)) =
            driver.speculative_transactions.values().next()
        else {
            panic!("post-abort branch release failure must retain the ending cohort");
        };
        assert!(matches!(
            pending.ending,
            SpeculativeEnding::BackendAbortedPendingCleanup { .. }
        ));
        assert_eq!(pending.source_states.len(), 1);
        assert_eq!(pending.schedules.len(), 1);
        assert!(driver.session_owner.contains_key(&SessionId(1)));
        let rolled_back_batches = driver.executor().runner().rolled_back_batches;

        let business_error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            business_error
                .to_string()
                .contains("simulated resumable batch failure")
        );
        assert!(driver.speculative_transactions.is_empty());
        assert!(driver.session_owner.is_empty());
        assert!(driver.continuations.is_empty());
        assert!(driver.transaction_continuations.is_empty());
        assert_eq!(
            driver.executor().runner().rolled_back_batches,
            rolled_back_batches
        );
        assert_eq!(driver.executor().runner().released_sequence_states, 2);
        assert_eq!(driver.scheduler().failed_len(), 1);
    }

    #[test]
    fn speculative_rollback_failure_quarantines_all_ownership_until_retry() {
        let request = materialization_request(9);
        let key = request
            .materialization_key(ferrule_common::DestinationGeneration::new(37))
            .unwrap();
        let (physical, handle) = MockPhysicalProvider::automatic();
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
            .with_resumable_wait_scripts([0, 1])
            .with_materialization_request(request)
            .with_materialization_retention(ferrule_model::ResourceRetention::ThroughTransaction)
            .with_empty_top_k_output()
            .with_rollback_failures(1)
            .with_materialization_provider(Box::new(physical));
        let mut driver = speculative_driver_from_runner(runner, 1);
        let actions = ready_speculative_decode_actions(&mut driver, &[1]);

        assert!(matches!(
            driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(ref pending) if pending.len() == 1
        ));
        let transaction = *driver
            .speculative_transactions
            .keys()
            .next()
            .expect("speculative verification should own its transaction");
        assert!(driver.owns_transaction(transaction));
        assert_eq!(driver.session_owner[&SessionId(1)], transaction);
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("simulated backend rollback failure")
        );
        let Some(PendingSpeculativeDriverCohort::Ending(pending)) =
            driver.speculative_transactions.get(&transaction)
        else {
            panic!("terminal failure must retain the complete speculative cohort");
        };
        assert_eq!(pending.source_states.len(), 1);
        assert_eq!(pending.schedules.len(), 1);
        assert_eq!(pending.actions.len(), 1);
        assert_eq!(pending.request_id, None);
        assert!(matches!(pending.ending, SpeculativeEnding::Abort { .. }));
        assert!(driver.owns_transaction(transaction));
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert_eq!(driver.session_owner[&SessionId(1)], transaction);
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert!(driver.page_manager().unwrap().allocated_pages() > 0);
        assert_eq!(
            handle.command_count(|command| matches!(
                command,
                MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
            )),
            0
        );

        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Pending
        );
        assert!(driver.owns_transaction(transaction));
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert_eq!(driver.session_owner[&SessionId(1)], transaction);
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert!(driver.page_manager().unwrap().allocated_pages() > 0);
        assert_eq!(
            handle.command_count(|command| matches!(
                command,
                MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
            )),
            0
        );

        driver
            .load_registry
            .finish_pending_lease_releases()
            .unwrap();
        assert_eq!(
            handle.command_count(|command| matches!(
                command,
                MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
            )),
            0,
            "fatal backend termination must retain physical ownership"
        );
    }

    #[test]
    fn proposal_post_abort_provider_cancel_failure_retries_without_reabort() {
        let materialization = materialization_request(19);
        let (physical, handle) = MockPhysicalProvider::manual();
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
            .with_materialization_request(materialization)
            .with_materialization_retention(ferrule_model::ResourceRetention::ThroughTransaction)
            .with_materialization_provider(Box::new(physical));
        let mut driver = speculative_driver_from_runner(runner, 1);
        let actions = ready_speculative_decode_actions(&mut driver, &[1]);

        assert!(matches!(
            driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        driver.progress_materialization().unwrap();
        handle.fail_next_cancel(FailureReason::DeviceUnavailable);

        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Pending
        );
        let Some(PendingSpeculativeDriverCohort::Proposing(pending)) =
            driver.speculative_transactions.values().next()
        else {
            panic!("post-abort provider cleanup failure must retain the proposal cohort");
        };
        assert!(pending.backend_aborted);
        assert!(pending.custody.is_some());
        assert_eq!(pending.source_states.len(), 1);
        assert_eq!(pending.schedules.len(), 1);
        assert!(driver.session_owner.contains_key(&SessionId(1)));
        let rolled_back_batches = driver.executor().runner().rolled_back_batches;
        assert_eq!(
            handle.command_count(|command| matches!(command, MockPhysicalCommand::Cancel(..))),
            1
        );

        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Complete(CancelRequestResult::Active {
                request_id: RequestId(1),
                session_id: SessionId(1),
            })
        );
        assert!(driver.speculative_transactions.is_empty());
        assert!(driver.session_owner.is_empty());
        assert!(driver.continuations.is_empty());
        assert!(driver.transaction_continuations.is_empty());
        assert_eq!(
            driver.executor().runner().rolled_back_batches,
            rolled_back_batches
        );

        assert_eq!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Idle
        );
        assert_eq!(
            handle.command_count(|command| matches!(command, MockPhysicalCommand::Cancel(..))),
            2
        );
        for kind in [
            ResourceKind::Continuation,
            ResourceKind::Waiter,
            ResourceKind::Arena,
            ResourceKind::ResidencyLease,
        ] {
            assert_eq!(driver.load_registry().resources().in_use(kind), 0);
        }
    }

    #[test]
    fn verification_cancellation_reports_pending_until_backend_abort_completes() {
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
            .with_resumable_wait_scripts([0, 1])
            .with_resumable_cancel_still_active(1);
        let mut driver = speculative_driver_from_runner(runner, 1);
        let actions = ready_speculative_decode_actions(&mut driver, &[1]);
        assert!(matches!(
            driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));

        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Pending
        );
        assert!(matches!(
            driver.speculative_transactions.values().next(),
            Some(PendingSpeculativeDriverCohort::Ending(pending))
                if matches!(pending.ending, SpeculativeEnding::Abort { .. })
        ));
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);

        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Pending
        );
        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Cancel,
                ..
            }
        ));
        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Complete(CancelRequestResult::Active {
                request_id: RequestId(1),
                session_id: SessionId(1),
            })
        );
        assert!(driver.speculative_transactions.is_empty());
        assert_eq!(driver.executor().runner().rolled_back_batches, 2);
    }

    #[test]
    fn shutdown_finishes_proposal_cancellation_before_returning_saved_error() {
        let materialization = materialization_request(20);
        let (physical, handle) = MockPhysicalProvider::manual();
        let runner = MockTopKRunner::new(vec![top(10)])
            .with_speculative_cycle(
                NativeProposal {
                    token_ids: vec![11],
                    confidence_logits: vec![1.0],
                },
                vec![TokenLogit::new(11, 9.0), TokenLogit::new(99, 8.0)],
            )
            .with_proposal_waits(vec![1])
            .with_materialization_request(materialization)
            .with_materialization_retention(ferrule_model::ResourceRetention::ThroughTransaction)
            .with_materialization_provider(Box::new(physical));
        let mut driver = speculative_driver_from_runner(runner, 1);
        let actions = ready_speculative_decode_actions(&mut driver, &[1]);
        assert!(matches!(
            driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        driver.progress_materialization().unwrap();
        let PendingSpeculativeDriverCohort::Proposing(pending) = driver
            .speculative_transactions
            .values_mut()
            .next()
            .expect("proposal wait retains its cohort")
        else {
            panic!("expected a proposing cohort");
        };
        pending.abort_cause = Some(Error::InvalidRequest {
            message: "saved proposal business failure".into(),
        });
        handle.fail_next_cancel(FailureReason::DeviceUnavailable);

        let error = driver.shutdown(&mut |_| Ok(()), 32).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("saved proposal business failure")
        );
        assert!(driver.speculative_transactions.is_empty());
        assert!(driver.pending_request_cancellations.is_empty());
        assert!(driver.session_owner.is_empty());
        assert!(driver.continuations.is_empty());
        assert!(driver.transaction_continuations.is_empty());
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert!(
            driver
                .shutdown(&mut |_| Ok(()), 32)
                .unwrap()
                .registry
                .drained
        );
    }

    #[test]
    fn proposal_continuation_registration_failure_restores_cohort_ownership() {
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
            .with_proposal_waits(vec![1]);
        let mut driver = speculative_driver_from_runner(runner, 1);
        let actions = ready_speculative_decode_actions(&mut driver, &[1]);
        driver.next_dependency_epoch = u64::MAX;

        let error = driver
            .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("dependency-set epoch space is exhausted")
        );
        assert!(driver.speculative_transactions.is_empty());
        assert!(driver.continuations.is_empty());
        assert!(driver.transaction_continuations.is_empty());
        assert!(driver.session_owner.is_empty());
        assert!(driver.sequence_states.contains_key(&SessionId(1)));
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert!(
            driver
                .executor()
                .runner()
                .active_native_proposals
                .is_empty()
        );
        assert_eq!(driver.scheduler().failed_len(), 0);
        assert!(driver.scheduler.next_decode_action().unwrap().is_some());
        for kind in [
            ResourceKind::Continuation,
            ResourceKind::Waiter,
            ResourceKind::Arena,
            ResourceKind::ResidencyLease,
        ] {
            assert_eq!(driver.load_registry().resources().in_use(kind), 0);
        }
    }

    #[test]
    fn verification_continuation_registration_failure_rolls_back_transaction() {
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
            .with_resumable_wait_scripts([0, 1]);
        let mut driver = speculative_driver_from_runner(runner, 1);
        let actions = ready_speculative_decode_actions(&mut driver, &[1]);
        driver.next_dependency_epoch = u64::MAX;

        let error = driver
            .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("dependency-set epoch space is exhausted")
        );
        assert!(driver.speculative_transactions.is_empty());
        assert!(driver.continuations.is_empty());
        assert!(driver.transaction_continuations.is_empty());
        assert!(driver.session_owner.is_empty());
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert_eq!(driver.scheduler().failed_len(), 1);
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        for kind in [
            ResourceKind::Continuation,
            ResourceKind::Waiter,
            ResourceKind::Arena,
            ResourceKind::ResidencyLease,
        ] {
            assert_eq!(driver.load_registry().resources().in_use(kind), 0);
        }
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
        assert_eq!(waiting[0].continuation(), ContinuationId::new(100));
        assert_eq!(waiting[0].dependencies().len(), 1);
        assert_eq!(driver.executor().runner().native_proposal_begin_calls, 2);
        assert_eq!(driver.executor().runner().native_proposal_resume_calls, 0);
        assert_eq!(driver.executor().runner().active_native_proposals.len(), 1);
        assert_eq!(driver.executor().runner().packed_verification_calls, 0);

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
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
                driver.step(&mut |_| Ok(())).unwrap(),
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
            driver.step(&mut |_| Ok(())).unwrap(),
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
    fn verification_resume_error_keeps_continuation_until_abort_completes() {
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
            .with_resumable_wait_scripts([0, 1])
            .with_resume_errors(1)
            .with_resumable_cancel_still_active(1);
        let mut driver = speculative_driver_from_runner(runner, 1);
        let actions = ready_speculative_decode_actions(&mut driver, &[1]);
        assert!(matches!(
            driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        let continuation = *driver
            .continuations
            .keys()
            .next()
            .expect("verification wait owns a continuation");

        assert_eq!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Blocked
        );
        let Some(PendingSpeculativeDriverCohort::Ending(pending)) =
            driver.speculative_transactions.values().next()
        else {
            panic!("active resume failure must retain an ending cohort");
        };
        assert_eq!(pending.continuation, Some(continuation));
        assert!(driver.continuations.contains_key(&continuation));
        assert!(
            driver
                .transaction_continuations
                .values()
                .any(|continuations| continuations.contains(&continuation))
        );
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert_eq!(
            driver
                .load_registry()
                .resources()
                .in_use(ResourceKind::Continuation),
            1
        );

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("simulated resumable batch failure")
        );
        assert!(driver.speculative_transactions.is_empty());
        assert!(driver.continuations.is_empty());
        assert!(driver.transaction_continuations.is_empty());
        assert!(driver.session_owner.is_empty());
        assert_eq!(driver.executor().runner().rolled_back_batches, 2);
        for kind in [
            ResourceKind::Continuation,
            ResourceKind::Waiter,
            ResourceKind::Arena,
            ResourceKind::ResidencyLease,
        ] {
            assert_eq!(driver.load_registry().resources().in_use(kind), 0);
        }
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

        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Pending
        );
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

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Cancel,
                ..
            }
        ));
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
    fn packed_proposal_retains_every_cancellation_while_abort_is_pending() {
        let runner = MockTopKRunner::new(vec![top(10)])
            .with_speculative_cycle(
                NativeProposal {
                    token_ids: vec![11],
                    confidence_logits: vec![1.0],
                },
                vec![TokenLogit::new(11, 9.0), TokenLogit::new(99, 8.0)],
            )
            .with_speculative_cycle(
                NativeProposal {
                    token_ids: vec![12],
                    confidence_logits: vec![1.0],
                },
                vec![TokenLogit::new(12, 9.0), TokenLogit::new(98, 8.0)],
            )
            .with_proposal_waits(vec![1, 1])
            .with_proposal_cancel_still_active(1);
        let mut driver = speculative_driver_from_runner(runner, 2);
        let actions = ready_speculative_decode_actions(&mut driver, &[1, 2]);
        assert!(matches!(
            driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));

        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Pending
        );
        assert_eq!(
            driver.cancel_request(RequestId(2)).unwrap(),
            ResidentCancelProgress::Complete(CancelRequestResult::Active {
                request_id: RequestId(2),
                session_id: SessionId(2),
            })
        );
        assert!(driver.speculative_transactions.is_empty());
        assert!(driver.pending_request_cancellations.is_empty());
        assert!(driver.pending_sequence_cleanups.is_empty());
        assert!(driver.session_owner.is_empty());
        assert_eq!(driver.executor().runner().rolled_back_batches, 2);
        for request_id in [RequestId(1), RequestId(2)] {
            let session_id = SessionId(request_id.0);
            assert_eq!(
                driver.cancel_request(request_id).unwrap(),
                ResidentCancelProgress::Complete(CancelRequestResult::Active {
                    request_id,
                    session_id,
                })
            );
        }
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

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Blocked
        ));
        assert!(matches!(
            driver.speculative_transactions.values().next(),
            Some(PendingSpeculativeDriverCohort::Proposing(_))
        ));
        assert_eq!(driver.executor().runner().native_proposal_resume_calls, 1);
        assert_eq!(driver.executor().runner().native_proposal_cancel_calls, 1);
        assert_eq!(driver.executor().runner().active_native_proposals.len(), 1);
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(matches!(
            error,
            Error::Backend {
                source: ferrule_common::Error::Model { .. },
            }
        ));
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
            ResidentCancelProgress::Complete(CancelRequestResult::Active {
                request_id: RequestId(1),
                session_id: SessionId(1),
            })
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
                enable_native_proposals: true,
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
            driver.step(&mut emit).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
        let first_wait = driver.step(&mut emit).unwrap();
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

        let second_wait = driver.step(&mut emit).unwrap();
        assert!(matches!(
            second_wait,
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert!(events.borrow().is_empty());
        assert_eq!(driver.executor().runner().native_proposal_begin_calls, 1);
        assert_eq!(driver.executor().runner().packed_verification_calls, 1);

        let completed = driver.step(&mut emit).unwrap();
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
                vec![TokenLogit::new(21, 6.0)],
            )
            .with_resumable_wait_scripts([0, 0, 2, 0]);
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
                enable_native_proposals: true,
                proposal_confidence_threshold: 0.2,
            },
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        for id in [1, 2] {
            let max_new_tokens = if id == 1 { 4 } else { 1 };
            let mut submitted = request(id, &[id as u32], max_new_tokens, Vec::new());
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
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            }
        ));
        assert_eq!(driver.pending_model_progresses().len(), 1);
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
        assert_eq!(driver.executor().runner().sequence_state_fork_calls, 2);
        assert_eq!(
            driver
                .executor()
                .runner()
                .topology_mutation_attempts_while_packed,
            0
        );
        assert_eq!(driver.executor().runner().released_sequence_states, 1);
        let owned_index = usize::from(owned_session == SessionId(2));
        let owned_slot = driver.page_slots[&owned_session];
        assert_eq!(
            driver
                .page_manager()
                .unwrap()
                .block_table(owned_slot)
                .unwrap()
                .pages(),
            page_tables_before[owned_index]
        );
        assert_eq!(
            driver.page_manager().unwrap().allocated_pages(),
            allocated_pages_before
        );

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert_eq!(driver.speculative_transactions.len(), 1);
        assert_eq!(driver.executor().runner().sequence_state_fork_calls, 2);
        assert_eq!(
            driver
                .executor()
                .runner()
                .topology_mutation_attempts_while_packed,
            0
        );
        assert_eq!(driver.executor().runner().released_sequence_states, 1);

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            }
        ));
        assert!(driver.speculative_transactions.is_empty());
        assert!(driver.session_owner.is_empty());
        assert!(driver.sequence_states.contains_key(&owned_session));
        assert!(driver.sequence_states.contains_key(&runnable_session));
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
                enable_native_proposals: true,
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
            ResidentCancelProgress::Complete(CancelRequestResult::Active {
                request_id: RequestId(1),
                session_id: SessionId(1),
            })
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
                enable_native_proposals: true,
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
        for token in 1..=4 {
            let snapshot = driver
                .load_registry()
                .ledger()
                .output(OutputTokenId::new(token))
                .unwrap();
            assert_eq!(snapshot.externally_committed_tokens, 4);
        }
        assert!(
            driver
                .load_registry()
                .ledger()
                .output(OutputTokenId::new(5))
                .is_none()
        );
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
                .step(&mut |event| {
                    events.push(event.clone());
                    Ok(())
                },)
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert!(!driver.resident_transactions.is_empty());
        assert!(driver.speculative_transactions.is_empty());
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert!(driver.has_live_transactions());
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
            driver.step(&mut |_| Ok(())).unwrap(),
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
            driver.step(&mut |_| Ok(())).unwrap(),
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
    fn initial_continuation_registration_failure_aborts_active_transaction() {
        let runner = MockTopKRunner::new(Vec::new())
            .with_committed_resumable_batch(1, vec![TokenLogit::new(7, 1.0)]);
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
        driver.next_dependency_epoch = u64::MAX;
        let mut submitted = request(1, &[1], 2, Vec::new());
        submitted.session_id = Some(SessionId(1));
        driver.submit(submitted);

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("dependency-set epoch space is exhausted")
        );
        assert!(driver.resident_transactions.is_empty());
        assert!(driver.continuations.is_empty());
        assert!(driver.transaction_continuations.is_empty());
        assert!(driver.session_owner.is_empty());
        assert_eq!(driver.executor().runner().prepared_batches, 1);
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert_eq!(driver.scheduler().failed_len(), 1);
        assert_eq!(
            driver
                .load_registry()
                .resources()
                .in_use(ResourceKind::Continuation),
            0
        );
        assert_eq!(
            driver
                .load_registry()
                .resources()
                .in_use(ResourceKind::Waiter),
            0
        );
    }

    #[test]
    fn resumed_continuation_registration_failure_aborts_active_transaction() {
        let runner = MockTopKRunner::new(Vec::new())
            .with_committed_resumable_batch(2, vec![TokenLogit::new(7, 1.0)]);
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
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        driver.next_dependency_epoch = u64::MAX;
        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("dependency-set epoch space is exhausted")
        );
        assert!(driver.resident_transactions.is_empty());
        assert!(driver.continuations.is_empty());
        assert!(driver.transaction_continuations.is_empty());
        assert!(driver.session_owner.is_empty());
        assert_eq!(driver.executor().runner().prepared_batches, 1);
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert_eq!(driver.scheduler().failed_len(), 1);
        for kind in [
            ResourceKind::Continuation,
            ResourceKind::Waiter,
            ResourceKind::Arena,
            ResourceKind::ResidencyLease,
        ] {
            assert_eq!(driver.load_registry().resources().in_use(kind), 0);
        }
    }

    #[test]
    fn resident_post_abort_kv_release_failure_retries_without_reending_backend() {
        let runner = MockTopKRunner::new(Vec::new())
            .with_committed_resumable_batch(1, vec![TokenLogit::new(7, 1.0)])
            .with_resume_errors(1)
            .with_kv_release_failures(1);
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
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        let cleanup_error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            cleanup_error
                .to_string()
                .contains("simulated KV page release failure")
        );
        let pending = driver
            .resident_transactions
            .values()
            .next()
            .expect("failed post-abort cleanup retains the resident transaction");
        assert!(matches!(
            pending.phase,
            ResidentTransactionPhase::BackendAbortedPendingCleanup { .. }
        ));
        assert!(matches!(pending.kv, Some(PendingResidentKv::Retiring(_))));
        assert!(driver.session_owner.contains_key(&SessionId(1)));
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);

        let business_error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            business_error
                .to_string()
                .contains("simulated resumable batch failure")
        );
        assert!(driver.resident_transactions.is_empty());
        assert!(driver.session_owner.is_empty());
        assert!(driver.continuations.is_empty());
        assert!(driver.transaction_continuations.is_empty());
        assert!(driver.kv_page_grants.is_empty());
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert_eq!(driver.executor().runner().released_kv_pages.len(), 1);
        assert_eq!(driver.scheduler().failed_len(), 1);
        for kind in [
            ResourceKind::KvPage,
            ResourceKind::Continuation,
            ResourceKind::Waiter,
            ResourceKind::Arena,
            ResourceKind::ResidencyLease,
        ] {
            assert_eq!(driver.load_registry().resources().in_use(kind), 0);
        }
    }

    #[test]
    fn resident_terminal_slot_release_failure_is_retried_by_owner_tick() {
        let runner = MockTopKRunner::new(Vec::new())
            .with_committed_resumable_batch(1, vec![TokenLogit::new(7, 1.0)])
            .with_resume_errors(1);
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FailOnceSequenceSlotPool::new(1, 1),
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
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("simulated resumable batch failure")
        );
        assert!(driver.resident_transactions.is_empty());
        assert_eq!(driver.scheduler.failed_slot_ownership(), 1);
        assert_eq!(driver.slot_pool().active_count(), 1);
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);

        assert_eq!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Idle
        );
        assert_eq!(driver.scheduler.failed_slot_ownership(), 0);
        assert_eq!(driver.slot_pool().active_count(), 0);
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
    }

    #[test]
    fn ordinary_resumable_resume_error_retains_ownership_until_abort_completes() {
        let runner = MockTopKRunner::new(Vec::new())
            .with_committed_resumable_batch(1, vec![TokenLogit::new(7, 1.0)])
            .with_resume_errors(1)
            .with_resumable_cancel_still_active(1);
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
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert_eq!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Blocked
        );
        assert!(!driver.resident_transactions.is_empty());
        assert!(driver.has_live_transactions());
        assert!(!driver.continuations.is_empty());
        assert!(!driver.transaction_continuations.is_empty());
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert_eq!(driver.executor().runner().committed_batches, 0);
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("simulated resumable batch failure")
        );
        assert!(driver.resident_transactions.is_empty());
        assert_eq!(driver.executor().runner().prepared_batches, 1);
        assert_eq!(driver.executor().runner().committed_batches, 0);
        assert_eq!(driver.executor().runner().rolled_back_batches, 2);
        assert_eq!(driver.scheduler().failed_len(), 1);
        assert!(driver.continuations.is_empty());
        assert!(driver.transaction_continuations.is_empty());
        assert!(driver.session_owner.is_empty());
        for kind in [
            ResourceKind::Continuation,
            ResourceKind::Waiter,
            ResourceKind::Arena,
            ResourceKind::ResidencyLease,
        ] {
            assert_eq!(driver.load_registry().resources().in_use(kind), 0);
        }
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
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                rows: 1,
                ..
            }
        ));
        assert_eq!(driver.resident_transactions.len(), 1);
        assert!(driver.has_live_transactions());
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert!(driver.sequence_states.contains_key(&SessionId(2)));

        let cancelled = driver.cancel_request(RequestId(2)).unwrap();
        assert_eq!(cancelled, ResidentCancelProgress::Pending);
        assert!(
            driver
                .pending_request_cancellations
                .contains_key(&RequestId(2))
        );
        assert_eq!(driver.resident_transactions.len(), 1);
        assert!(driver.has_live_transactions());
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

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                rows: 1,
                ..
            }
        ));
        assert!(driver.resident_transactions.is_empty());
        assert!(!driver.has_live_transactions());
        assert_eq!(driver.executor().runner().committed_batches, 2);
        assert!(driver.pending_sequence_cleanups.contains_key(&SessionId(2)));

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            }
        ));
        assert!(!driver.pending_sequence_cleanups.contains_key(&SessionId(2)));
        assert!(driver.pending_request_cancellations.is_empty());
        assert!(!driver.sequence_states.contains_key(&SessionId(2)));
        assert!(!driver.page_slots.contains_key(&SessionId(2)));
        assert_eq!(
            driver.cancel_request(RequestId(2)).unwrap(),
            ResidentCancelProgress::Complete(CancelRequestResult::Active {
                request_id: RequestId(2),
                session_id: SessionId(2),
            })
        );
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

        let ResidentDriverStep::WaitingForModelProgress(progress) =
            driver.step(&mut |_| Ok(())).unwrap()
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
            driver.step(&mut |_| Ok(())).unwrap(),
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
            driver.step(&mut |_| Ok(())).unwrap(),
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
            driver.step(&mut |_| Ok(())).unwrap(),
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
                ResidentCancelProgress::Pending
            );
        }
        assert_eq!(driver.pending_request_cancellations.len(), 3);
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
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
        assert!(driver.resident_transactions.is_empty());
        assert_eq!(driver.pending_sequence_cleanups.len(), 3);

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            }
        ));
        assert!(driver.pending_sequence_cleanups.is_empty());
        assert!(driver.pending_request_cancellations.is_empty());
        assert!(driver.pending_kv_retirements.is_empty());
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 1);
        for session_id in &siblings {
            assert!(!driver.page_slots.contains_key(session_id));
            assert!(!driver.sequence_states.contains_key(session_id));
        }
        assert_eq!(driver.executor().runner().released_sequence_states, 3);
        assert_eq!(driver.executor().runner().released_kv_pages, sibling_pages);
        for session_id in &siblings {
            let request_id = RequestId(session_id.0);
            assert_eq!(
                driver.cancel_request(request_id).unwrap(),
                ResidentCancelProgress::Complete(CancelRequestResult::Active {
                    request_id,
                    session_id: *session_id,
                })
            );
        }
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
                .step(&mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(ref pending) if pending.len() == 2
        ));
        assert_eq!(driver.resident_transactions.len(), 2);

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
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
            driver.step(&mut |_| Ok(())).unwrap(),
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
            driver.step(&mut |_| Ok(())).unwrap(),
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
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(progress) if progress.len() == 2
        ));
        let transaction_b = driver.session_owner[&SessionId(2)];

        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Pending
        );
        assert!(
            driver
                .pending_request_cancellations
                .contains_key(&RequestId(1))
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
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
        assert!(driver.resident_transactions.is_empty());
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert_eq!(driver.executor().runner().committed_batches, 1);
        assert!(driver.pending_sequence_cleanups.contains_key(&SessionId(1)));

        let _ = driver.step(&mut |_| Ok(())).unwrap();
        assert!(driver.pending_kv_retirements.is_empty());
        assert!(!driver.pending_sequence_cleanups.contains_key(&SessionId(1)));
        assert!(driver.pending_request_cancellations.is_empty());
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert!(!driver.page_slots.contains_key(&SessionId(1)));
        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Complete(CancelRequestResult::Active {
                request_id: RequestId(1),
                session_id: SessionId(1),
            })
        );
        assert_eq!(
            driver
                .executor()
                .runner()
                .topology_mutation_attempts_while_packed,
            0
        );
    }

    #[test]
    fn shutdown_does_not_poll_existing_resident_abort_twice_without_a_wake() {
        let runner = MockTopKRunner::new(Vec::new())
            .with_committed_resumable_batch(1, vec![TokenLogit::new(7, 1.0)])
            .with_resumable_cancel_still_active(2);
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
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Pending
        );
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);

        let error = driver.shutdown(&mut |_| Ok(()), 32).unwrap_err();

        assert!(error.to_string().contains("could not quiesce"));
        assert_eq!(driver.executor().runner().rolled_back_batches, 2);
        assert!(matches!(
            driver
                .resident_transactions
                .values()
                .next()
                .map(|pending| &pending.phase),
            Some(ResidentTransactionPhase::Aborting { .. })
        ));
        assert!(
            driver
                .shutdown(&mut |_| Ok(()), 32)
                .unwrap()
                .registry
                .drained
        );
        assert_eq!(driver.executor().runner().rolled_back_batches, 3);
    }

    #[test]
    fn repeated_shutdown_preserves_resident_abort_continuation() {
        let runner = MockTopKRunner::new(Vec::new())
            .with_committed_resumable_batch(1, vec![TokenLogit::new(7, 1.0)])
            .with_resumable_cancel_still_active(1);
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
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        let continuation = *driver
            .continuations
            .keys()
            .next()
            .expect("resident wait owns a continuation");

        let error = driver.shutdown(&mut |_| Ok(()), 32).unwrap_err();
        assert!(error.to_string().contains("could not quiesce"));
        let pending = driver
            .resident_transactions
            .values()
            .next()
            .expect("pending abort retains resident transaction");
        let ResidentTransactionPhase::Aborting {
            continuation: retained,
            ..
        } = pending.phase
        else {
            panic!("shutdown must retain the resident abort phase");
        };
        assert_eq!(retained, Some(continuation));
        assert!(driver.continuations.contains_key(&continuation));
        assert_eq!(
            driver
                .load_registry()
                .resources()
                .in_use(ResourceKind::Continuation),
            1
        );

        let report = driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert!(report.registry.drained);
        assert!(driver.resident_transactions.is_empty());
        assert!(driver.continuations.is_empty());
        assert!(driver.transaction_continuations.is_empty());
        assert!(driver.session_owner.is_empty());
        assert_eq!(report.registry.active_grants, 0);
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
        driver.step(&mut |_| Ok(())).unwrap();
        let transaction_a = driver.session_owner[&SessionId(1)];
        let transaction_b = driver.session_owner[&SessionId(2)];

        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Pending
        );
        assert!(matches!(
            driver.resident_transactions[&transaction_a].phase,
            ResidentTransactionPhase::Aborting {
                request: Some(RequestId(1)),
                ..
            }
        ));
        assert!(driver.resident_transactions.contains_key(&transaction_b));
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Cancel,
                ..
            }
        ));
        assert!(!driver.resident_transactions.contains_key(&transaction_a));
        assert!(driver.resident_transactions.contains_key(&transaction_b));
        assert_eq!(driver.executor().runner().committed_batches, 0);
        assert_eq!(driver.executor().runner().rolled_back_batches, 2);

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
        assert!(!driver.resident_transactions.contains_key(&transaction_b));
        assert_eq!(driver.executor().runner().committed_batches, 1);
        assert_eq!(driver.executor().runner().rolled_back_batches, 2);
        assert_eq!(driver.executor().runner().cancelled_continuations.len(), 1);
        assert!(driver.scheduler().active_sequence(SessionId(1)).is_none());
        assert!(driver.scheduler().active_sequence(SessionId(2)).is_some());
    }

    #[test]
    fn packed_resident_cancellation_returns_the_callers_request() {
        let runner = MockTopKRunner::new(Vec::new())
            .with_resumable_wait_scripts([1])
            .with_resumable_cancel_still_active(1);
        let mut driver = batched_driver_from_runner(runner)
            .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        for id in [1, 2] {
            let mut submitted = request(id, &[id as u32], 2, Vec::new());
            submitted.session_id = Some(SessionId(id));
            driver.submit(submitted);
        }
        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        let transaction = driver.session_owner[&SessionId(1)];
        assert_eq!(driver.session_owner[&SessionId(2)], transaction);

        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Pending
        );
        assert_eq!(
            driver.cancel_request(RequestId(2)).unwrap(),
            ResidentCancelProgress::Complete(CancelRequestResult::Active {
                request_id: RequestId(2),
                session_id: SessionId(2),
            })
        );
        assert!(driver.resident_transactions.is_empty());
        assert!(driver.pending_request_cancellations.is_empty());
        assert_eq!(driver.executor().runner().rolled_back_batches, 2);
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
            driver.step(&mut |_| Ok(())).unwrap(),
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
    fn secondary_eos_wins_over_max_tokens_without_becoming_visible() {
        let secondary_eos = 3;
        let runner =
            MockTopKRunner::new(vec![top(secondary_eos)]).with_eos_tokens([2, secondary_eos]);
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(1),
            ResidentSchedulerConfig::default(),
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        );
        driver.submit(request(39, &[1], 1, Vec::new()));
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
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].finish_reason, Some(SequenceFinishReason::Eos));
        assert_eq!(finished[0].generated, 0);
        assert_eq!(finished[0].position, 1);
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
    fn prefix_cleanup_release_failure_retains_state_without_async_wait() {
        let mut driver =
            driver_from_runner(MockTopKRunner::new(Vec::new()).with_release_failures(1));
        let mut state = driver.executor_mut().create_sequence_state().unwrap();
        state.position = 37;
        driver
            .pending_prefix_cleanups
            .push_back(PendingPrefixCleanup::model_only(state));

        assert!(!driver.has_pending_async_work());
        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("simulated sequence-state release failure")
        );
        assert_eq!(driver.pending_prefix_cleanups.len(), 1);
        assert_eq!(
            driver
                .pending_prefix_cleanups
                .front()
                .and_then(|cleanup| cleanup.model_state.as_ref())
                .map(|state| state.position),
            Some(37)
        );
        assert_eq!(driver.executor().runner().released_sequence_states, 0);
        assert!(!driver.has_pending_async_work());

        assert_eq!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Idle
        );
        assert!(driver.pending_prefix_cleanups.is_empty());
        assert_eq!(driver.executor().runner().released_sequence_states, 1);
    }

    #[test]
    fn shutdown_retries_prefix_cleanup_before_runner_extraction() {
        let mut driver = prefix_cache_driver(Vec::new(), 2, 2);
        driver.submit(request(64, &[1, 2, 3], 0, Vec::new()));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        driver.drain_finished();
        assert!(!driver.prefix_cache.is_empty());
        assert_eq!(driver.executor().runner().released_sequence_states, 1);

        driver
            .executor_mut()
            .runner_mut()
            .release_failures_remaining = 1;
        let error = driver.shutdown(&mut |_| Ok(()), 32).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("simulated sequence-state release failure")
        );
        assert!(driver.prefix_cache.is_empty());
        assert_eq!(driver.pending_prefix_cleanups.len(), 1);
        assert_eq!(
            driver
                .pending_prefix_cleanups
                .front()
                .and_then(|cleanup| cleanup.model_state.as_ref())
                .map(|state| state.position),
            Some(3)
        );
        assert!(!driver.has_pending_async_work());

        let report = driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert!(report.registry.drained);
        assert!(driver.pending_prefix_cleanups.is_empty());
        let runner = driver
            .try_into_runner()
            .map_err(|failure| failure.0)
            .unwrap();
        assert_eq!(runner.released_sequence_states, 2);
    }

    #[test]
    fn prefix_cache_pressure_uses_target_only_when_proposals_are_available() {
        let mut driver = prefix_cache_driver(vec![top(42)], 2, 2);
        driver.executor_mut().runner_mut().native_proposal_enabled = true;
        let namespace = driver.prefix_cache_namespace().unwrap();

        driver.submit(request(65, &[1, 2, 3, 4], 0, Vec::new()));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        driver.drain_finished();
        assert!(driver.prefix_cache.contains_exact(namespace, &[1, 2, 3, 4]));
        assert_eq!(driver.available_kv_page_credits(), 1);

        let mut events = Vec::new();
        driver.submit(request(66, &[9, 10, 11, 12, 13], 1, Vec::new()));
        driver
            .drive_ready_test_work(|event| {
                events.push(event.token);
                Ok(())
            })
            .unwrap();
        let finished = driver.drain_finished();

        assert_eq!(events, vec![42]);
        assert_eq!(finished[0].tokens, vec![9, 10, 11, 12, 13, 42]);
        assert!(!driver.prefix_cache.contains_exact(namespace, &[1, 2, 3, 4]));
        assert_eq!(driver.executor().runner().native_proposal_begin_calls, 0);
        assert_eq!(driver.executor().runner().packed_verification_calls, 0);
        assert_eq!(driver.stats().speculative.cycles, 0);

        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
    }

    #[test]
    fn prefix_lookup_is_not_counted_until_admission_publish() {
        let mut driver = prefix_cache_driver(Vec::new(), 4, 8);
        let namespace = driver.prefix_cache_namespace().unwrap();
        driver.submit(request(67, &[1, 2, 3], 0, Vec::new()));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        driver.drain_finished();
        assert!(driver.prefix_cache.contains_exact(namespace, &[1, 2, 3]));
        assert_eq!(
            (
                driver.prefix_cache_stats().hits,
                driver.prefix_cache_stats().misses
            ),
            (0, 1)
        );

        let target_slot = StateSlot::new(driver.next_page_slot);
        driver
            .page_manager
            .as_mut()
            .unwrap()
            .alloc_sequence(target_slot, 0)
            .unwrap();
        driver.submit(request(68, &[1, 2, 3, 4], 0, Vec::new()));
        let error = driver.prepare_step().unwrap_err();

        assert!(error.to_string().contains("already allocated"));
        assert_eq!(driver.executor().runner().sequence_state_fork_calls, 2);
        assert_eq!(
            (
                driver.prefix_cache_stats().hits,
                driver.prefix_cache_stats().misses
            ),
            (0, 1)
        );
        assert_eq!(driver.prefix_hits(), driver.prefix_cache_stats().hits);
        assert_eq!(driver.prefix_misses(), driver.prefix_cache_stats().misses);
        assert_eq!(driver.scheduler().waiting_len(), 1);
        assert_eq!(driver.scheduler().active_len(), 0);

        let retirement = driver
            .page_manager
            .as_mut()
            .unwrap()
            .free_sequence_pages(target_slot)
            .unwrap();
        driver.release_and_confirm_retirement(retirement).unwrap();
        driver.cancel_request(RequestId(68)).unwrap();
        driver.drain_cancelled();
        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
    }

    #[test]
    fn prefix_cache_hit_restores_committed_frontier_and_executes_only_suffix() {
        let mut driver = prefix_cache_driver(vec![top(b'a' as u32), top(b'b' as u32)], 4, 8);
        let namespace = driver
            .prefix_cache_namespace()
            .expect("configured cache has a namespace");
        assert_eq!(namespace.model(), 1);
        assert_eq!(namespace.backend(), 1);
        assert_eq!(namespace.device(), 0);
        assert_eq!(namespace.plan(), 0xcafe);
        assert_eq!(
            namespace.layout(),
            driver.page_manager().unwrap().owner_identity()
        );

        driver.submit(request(60, &[1, 2, 3], 0, Vec::new()));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        let first = driver.drain_finished();
        assert_eq!(first[0].tokens, vec![1, 2, 3]);
        assert_eq!(driver.prefix_hits(), 0);
        assert_eq!(driver.prefix_misses(), 1);
        assert!(driver.prefix_cache.contains_exact(namespace, &[1, 2, 3]));
        assert_eq!(driver.prefix_cache.used_capacity(), 1);
        assert_eq!(driver.executor().runner().released_sequence_states, 1);

        driver.submit(request(61, &[1, 2, 3, 4, 5], 1, Vec::new()));
        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                rows: 2,
                ..
            }
        ));

        let sequence = driver.scheduler().active_sequence(SessionId(2)).unwrap();
        assert_eq!(sequence.tokens, vec![1, 2, 3, 4, 5]);
        assert_eq!(sequence.position, 5);
        assert_eq!(sequence.prompt_cursor, 5);
        let model = &driver.sequence_states[&SessionId(2)];
        assert_eq!(model.position, 5);
        assert_eq!(model.prefills, vec![vec![4, 5]]);
        assert_eq!(driver.prefix_hits(), 1);
        assert_eq!(driver.prefix_misses(), 1);
        assert!(
            driver
                .prefix_cache
                .contains_exact(namespace, &[1, 2, 3, 4, 5])
        );
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 1);

        let report = driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert!(report.registry.drained);
        assert!(driver.prefix_cache.is_empty());
        assert!(driver.pending_prefix_cleanups.is_empty());
        assert!(driver.prefix_cache_sessions.is_empty());
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 0);
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        assert_eq!(driver.executor().runner().released_sequence_states, 4);
    }

    #[test]
    fn retained_session_explicitly_bypasses_prefix_cache() {
        let mut driver = prefix_cache_driver(Vec::new(), 4, 8);
        driver.retain_session(SessionId(70)).unwrap();
        let mut submitted = request(70, &[1, 2, 3], 0, Vec::new());
        submitted.session_id = Some(SessionId(70));
        driver.submit(submitted);
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        driver.drain_finished();

        assert_eq!(driver.prefix_hits(), 0);
        assert_eq!(driver.prefix_misses(), 0);
        assert!(driver.prefix_cache.is_empty());
        assert!(!driver.prefix_cache_sessions.contains(&SessionId(70)));
        assert_eq!(driver.retained_session_position(SessionId(70)), Some(3));

        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
    }

    #[test]
    fn kv_reserve_evicts_cached_prefix_before_hard_admission() {
        let mut driver = prefix_cache_driver(Vec::new(), 2, 2);
        let namespace = driver.prefix_cache_namespace().unwrap();

        driver.submit(request(62, &[1, 2, 3, 4], 0, Vec::new()));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        driver.drain_finished();
        assert!(driver.prefix_cache.contains_exact(namespace, &[1, 2, 3, 4]));
        assert_eq!(driver.available_kv_page_credits(), 1);

        driver.submit(request(63, &[9, 10, 11, 12, 13], 0, Vec::new()));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        let finished = driver.drain_finished();
        assert_eq!(finished[0].tokens, vec![9, 10, 11, 12, 13]);
        assert!(!driver.prefix_cache.contains_exact(namespace, &[1, 2, 3, 4]));
        assert!(
            driver
                .prefix_cache
                .contains_exact(namespace, &[9, 10, 11, 12, 13])
        );
        assert_eq!(driver.prefix_cache.used_capacity(), 2);
        assert_eq!(driver.prefix_hits(), 0);
        assert_eq!(driver.prefix_misses(), 2);
        assert_eq!(driver.executor().runner().released_sequence_states, 3);
        assert!(!driver.executor().runner().released_kv_pages.is_empty());

        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        assert_eq!(driver.executor().runner().released_sequence_states, 4);
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
    fn into_runner_rejects_unstarted_warmup_and_retained_session_state() {
        let (physical, _) = MockPhysicalProvider::automatic();
        let warmup_driver = driver_from_runner(
            MockTopKRunner::new(Vec::new())
                .with_materialization_provider(Box::new(physical))
                .with_warmup(materialization_request(1)),
        );
        let Err(warmup_failure) = warmup_driver.try_into_runner() else {
            panic!("runner extraction must retain unstarted warmup ownership");
        };
        assert!(warmup_failure.0.to_string().contains("cannot extract"));

        let mut retained_driver = driver_with_outputs(Vec::new());
        retained_driver.retain_session(SessionId(7)).unwrap();
        let Err(retained_failure) = retained_driver.try_into_runner() else {
            panic!("runner extraction must retain explicit session state");
        };
        assert!(retained_failure.0.to_string().contains("session state"));
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
        let mut driver = driver_from_runner(runner);
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

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(format!("{error}").contains("simulated failure"));
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
            driver.step(&mut |_| Ok(())).unwrap(),
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
            .drive_ready_test_work(|_| {
                Err(Error::Invariant {
                    message: "simulated callback failure".into(),
                })
            })
            .unwrap_err();
        assert!(format!("{error}").contains("callback failure"));
        assert!(!driver.has_live_transactions());
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
            ResidentCancelProgress::Complete(CancelRequestResult::Waiting {
                request_id: RequestId(18),
                session_id: SessionId(1),
            })
        );
        assert_eq!(
            driver.cancel_request(RequestId(99)).unwrap(),
            ResidentCancelProgress::Complete(CancelRequestResult::NotFound {
                request_id: RequestId(99),
            })
        );
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
            ResidentCancelProgress::Complete(CancelRequestResult::Active {
                request_id: RequestId(19),
                session_id: SessionId(1),
            })
        );
        assert_eq!(driver.scheduler().active_len(), 0);
        assert_eq!(driver.slot_pool().active_count(), 0);
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 0);
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        assert!(!driver.sequence_states.contains_key(&SessionId(1)));
        assert!(!driver.page_slots.contains_key(&SessionId(1)));
        assert_eq!(driver.executor().runner().released_sequence_states, 1);
        assert_eq!(driver.executor().runner().released_kv_pages.len(), 1);

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
    fn shutdown_retries_suspended_cleanup_after_slot_release_failure() {
        let manager = KvPageManager::new(Box::new(DriverTestKvSchema), 16);
        let mut driver = ResidentTopKDriver::with_configs(
            MockTopKRunner::new(vec![top(b'a' as u32), top(b'b' as u32)]),
            FailOnceSequenceSlotPool::new(1, 1),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 1,
                max_decode_batch: 1,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(manager);
        driver.submit(request(22, &[1, 2], 2, Vec::new()));
        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed { .. }
        ));
        let session_id = SessionId(1);
        driver.preempt_session(session_id).unwrap();

        let error = driver.shutdown(&mut |_| Ok(()), 16).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("simulated sequence slot release failure")
        );
        assert!(driver.suspended_sequences.is_empty());
        assert!(matches!(
            driver.pending_sequence_cleanups.get(&session_id),
            Some(PendingSequenceCleanup::Suspended { .. })
        ));
        assert_eq!(driver.scheduler.failed_slot_ownership(), 1);
        assert_eq!(driver.slot_pool.active_count(), 1);
        assert_eq!(driver.executor.runner().released_sequence_states, 0);

        let report = driver.shutdown(&mut |_| Ok(()), 16).unwrap();
        assert!(report.registry.drained);
        assert!(driver.pending_sequence_cleanups.is_empty());
        assert_eq!(driver.scheduler.failed_slot_ownership(), 0);
        assert_eq!(driver.slot_pool.active_count(), 0);
        assert!(driver.kv_page_grants.is_empty());
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        assert_eq!(driver.executor.runner().released_sequence_states, 1);
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
    fn driver_admission_restores_waiting_request_when_logical_kv_prepare_fails() {
        let mut manager = KvPageManager::new(Box::new(DriverTestKvSchema), 8);
        manager.alloc_sequence(StateSlot::new(0), 0).unwrap();
        let mut driver = ResidentTopKDriver::with_configs(
            MockTopKRunner::new(Vec::new()),
            FixedSequenceSlotPool::new(1),
            ResidentSchedulerConfig::default(),
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(manager);
        driver.submit(request(5, &[1], 1, Vec::new()));

        let error = driver.prepare_step().unwrap_err();
        assert!(error.to_string().contains("already allocated"));
        assert_eq!(driver.scheduler().waiting_len(), 1);
        assert_eq!(driver.scheduler().active_len(), 0);
        assert_eq!(driver.slot_pool().active_count(), 0);
        assert!(driver.sequence_states.is_empty());
        assert!(driver.page_slots.is_empty());
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 1);
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
