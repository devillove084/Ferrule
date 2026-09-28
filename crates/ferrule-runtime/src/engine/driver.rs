use std::collections::{HashMap, HashSet, VecDeque};
use std::num::NonZeroU32;
use std::time::Instant;

use crate::{CleanupStep, Error, Result};

use ferrule_common::execution::{
    ExecutionOutput, ExecutionTransactionId, KvBindingMode, KvPageId, KvReservationView, StateSlot,
};
use ferrule_common::io_protocol::{
    BackendId, CancellationReason, ContinuationId, DeviceId, ModelInstanceId,
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
    FairQueueConfig, LoadRegistry, OutputTokenId, ResumeDisposition,
    RuntimeMaterializationProvider, RuntimeMaterializationResolver, SharedMaterializationProvider,
    TransactionCustodyOutcome, UnavailableMaterializationProvider,
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

mod admission;
mod continuations;
mod kv_lifecycle;
mod output;
mod prefix;
mod resident;
mod sessions;
mod shutdown;
mod speculative;

use admission::{AdmissionSessions, RuntimeRequestIdentity};
use continuations::ContinuationRegistry;
use kv_lifecycle::{KvLifecycleState, PendingKvRetirement, PendingResidentKv};
use output::{OutputArbiter, TerminalDecision, arbitrate_terminal, matched_stop};
use prefix::{PendingPrefixCleanup, PrefixLifecycleState, ResidentPrefixPayload};
use resident::{PendingResidentBatch, ResidentCancellationRoute, ResidentTransactionPhase};
use sessions::{PendingSequenceCleanup, SessionCustody};
#[cfg(test)]
use speculative::{NativeProposalSlotStatus, confident_proposal_prefix_length};
use speculative::{
    PendingSpeculativeDriverCohort, PendingSpeculativeEndingDriverCohort,
    PendingSpeculativeVerificationDriverCohort, PreparedSpeculativeAction, SpeculativeEnding,
    record_speculative_cohort_metrics, record_speculative_sequence_metrics,
};

use super::observability::{
    ResidentDriverObservability, ResidentPrefixCacheStats, ResidentTopKDriverStats,
};
use super::{InferenceRequestCleanup, NativeMultiSessionExecutor, RequestCleanupOwner};

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

/// Admission records are independent of HTTP permits and physical grants.
/// Waiting is a subset of request identities, not a second permit charge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeAdmissionOptions {
    pub max_waiting_requests: usize,
    pub max_request_identities: usize,
    pub max_session_identities: usize,
}
impl Default for RuntimeAdmissionOptions {
    fn default() -> Self {
        Self {
            max_waiting_requests: 1024,
            max_request_identities: 4096,
            max_session_identities: 4096,
        }
    }
}
impl RuntimeAdmissionOptions {
    pub fn validate(self) -> Result<Self> {
        if self.max_waiting_requests == 0
            || self.max_request_identities == 0
            || self.max_session_identities == 0
            || self.max_waiting_requests > self.max_request_identities
        {
            return Err(RuntimeAdmissionError::InvalidOptions.into());
        }
        Ok(self)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeAdmissionResource {
    WaitingRequests,
    RequestIdentities,
    SessionIdentities,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, snafu::Snafu)]
pub enum RuntimeAdmissionError {
    #[snafu(display("runtime admission is closed"))]
    Closed,
    #[snafu(display("runtime {resource:?} capacity {limit} exhausted (held {held})"))]
    Capacity {
        resource: RuntimeAdmissionResource,
        limit: usize,
        held: usize,
    },
    #[snafu(display("request {request_id:?} still owns runtime identity"))]
    DuplicateRequest { request_id: RequestId },
    #[snafu(display("session {session_id:?} still owns a turn, terminal or cleanup"))]
    SessionBusy { session_id: SessionId },
    #[snafu(display(
        "admission limits must be nonzero and waiting must not exceed request identities"
    ))]
    InvalidOptions,
    #[snafu(display(
        "cannot reduce admission limits below currently held identities or waiting requests"
    ))]
    OptionsInUse,
    #[snafu(display(
        "prompt position {position} plus {prompt_tokens} tokens exceeds context {context}"
    ))]
    InvalidPosition {
        position: usize,
        prompt_tokens: usize,
        context: usize,
    },
    #[snafu(display(
        "explicit position {supplied} differs from retained session position {expected}"
    ))]
    RetainedPositionMismatch { supplied: usize, expected: usize },
    #[snafu(display("automatic runtime session identity space exhausted"))]
    SessionIdentityExhausted,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeAdmissionSnapshot {
    pub limits: RuntimeAdmissionOptions,
    pub waiting_requests: usize,
    pub request_identities_held: usize,
    pub session_identities_held: usize,
    pub closed: bool,
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

/// A selected output token. With native proposals disabled, delivery follows
/// the selecting forward's commit, before this token's own KV append. Successful
/// request completion (not delivery) guarantees a fully retained KV frontier.
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
    /// The session is suspended; restore it or shut down its owner before retrying.
    RequiresRestoreOrShutdown {
        request_id: RequestId,
        session_id: SessionId,
    },
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

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DriverTickEvent {
    ResidentEndings,
    SpeculativeEndings,
    PendingCleanups,
    RequestCancellations,
    Outbox,
    Warmup,
    Materialization,
    WarmupCheck,
    Observability,
    ReadySnapshot(usize),
    Resume(ExecutionTransactionId),
    ContinuationReady(ContinuationId),
    TokenAcknowledged(SessionId, u32),
    Admitted(SessionId),
    CancelComplete(RequestId),
    CleanupDeferred,
    Bookkeeping,
    PrepareStep,
    Admission,
}

pub struct ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner,
    C: SequenceSlotPool,
{
    scheduler: ResidentScheduler,
    admission_options: RuntimeAdmissionOptions,
    request_identities: HashMap<RequestId, RuntimeRequestIdentity>,
    next_admission_session: Option<u64>,
    admission_closed: bool,
    slot_pool: C,
    executor: NativeMultiSessionExecutor<R>,
    sessions: SessionCustody<R::SequenceState>,
    /// Default top-k used for batch lowering.
    top_k: NonZeroU32,
    page_manager: Option<KvPageManager>,
    kv: KvLifecycleState,
    prefix: PrefixLifecycleState<R::SequenceState>,
    next_page_slot: u32,
    config: ResidentTopKDriverConfig,
    observability: ResidentDriverObservability,
    next_transaction_id: u64,
    resident_transactions: HashMap<ExecutionTransactionId, PendingResidentBatch<R::SequenceState>>,
    speculative_transactions:
        HashMap<ExecutionTransactionId, PendingSpeculativeDriverCohort<R::SequenceState>>,
    completion_hub: ferrule_common::CompletionHub,
    load_registry: LoadRegistry<Box<dyn RuntimeMaterializationProvider>>,
    materialization_resolver: RuntimeMaterializationResolver,
    uninstalled_materialization_resolver: Option<RuntimeMaterializationResolver>,
    continuations: ContinuationRegistry,
    warmup_requests: Vec<ferrule_model::MaterializationRequest>,
    warmup_started_ns: Option<u64>,
    warmup_last_report_ns: u64,

    runtime_clock: Instant,
    runtime_tick: u64,
    next_output_token_id: u64,
    output: OutputArbiter,
    shutting_down: bool,
    physical_shutdown_report: Option<ResidentDriverShutdownReport>,
    #[cfg(test)]
    stage_faults: VecDeque<&'static str>,
    #[cfg(test)]
    tick_trace: Vec<DriverTickEvent>,
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
        let prefix = PrefixLifecycleState::new(scheduler.config().prefix_cache_capacity_pages);
        let driver = Self {
            scheduler,
            admission_options: RuntimeAdmissionOptions::default(),
            request_identities: HashMap::new(),
            next_admission_session: Some(1),
            admission_closed: false,
            slot_pool,
            executor,
            sessions: SessionCustody::default(),
            top_k,
            page_manager: None,
            kv: KvLifecycleState::default(),
            prefix,
            next_page_slot: 0,
            config,
            observability: ResidentDriverObservability::default(),
            next_transaction_id: 1,
            resident_transactions: HashMap::new(),
            speculative_transactions: HashMap::new(),
            completion_hub,
            load_registry: registry,
            materialization_resolver: resolver,
            uninstalled_materialization_resolver: uninstalled_resolver,
            continuations: ContinuationRegistry::default(),
            warmup_requests,
            warmup_started_ns: None,
            warmup_last_report_ns: 0,
            runtime_clock: Instant::now(),
            runtime_tick: 0,
            next_output_token_id: 1,
            output: OutputArbiter::default(),
            shutting_down: false,
            physical_shutdown_report: None,
            #[cfg(test)]
            stage_faults: VecDeque::new(),
            #[cfg(test)]
            tick_trace: Vec::new(),
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

    fn has_pending_non_prefix_async_work(&self) -> bool {
        !self.resident_transactions.is_empty()
            || !self.speculative_transactions.is_empty()
            || !self.continuations.is_empty()
            || self.continuations.has_pending_failures()
            || self.continuations.has_pending_detaches()
            || self.load_registry.active_prefetches() != 0
            || self.load_registry.active_operations() != 0
            || self.load_registry.has_pending_owner_work()
            || !self.kv.pending_retirements_empty()
            || !self.kv.pending_abort_empty()
            || self.sessions.cleanup_count() != 0
            || !self.output.cancellations_empty()
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

    #[cfg(test)]
    fn fail_stage(&mut self, stage: &'static str) -> Result<()> {
        if self.stage_faults.front() == Some(&stage) {
            self.stage_faults.pop_front();
            return Err(Error::Invariant {
                message: format!("injected {stage}"),
            });
        }
        Ok(())
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
                    PendingSpeculativeDriverCohort::Bookkeeping(_) => true,
                    PendingSpeculativeDriverCohort::Ending(_)
                    | PendingSpeculativeDriverCohort::AdmissionRollback(_) => true,
                },
            )
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
        #[cfg(test)]
        self.tick_trace.push(DriverTickEvent::Observability);
        self.observability.stats.hard_resource_high_water = self
            .load_registry
            .resources()
            .snapshots()
            .map(|snapshot| (snapshot.kind, snapshot.high_water))
            .collect();
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

    /// Request cancellation without releasing resources before backend quiescence.
    /// Active transactions are driven automatically by subsequent owner ticks.
    pub fn cancel_request(&mut self, request_id: RequestId) -> Result<ResidentCancelProgress>
    where
        R: ResidentModelRunner,
    {
        if let Some((session_id, _)) = self
            .sessions
            .suspended_entries()
            .find(|(_, pending)| pending.schedule.request_id() == Some(request_id))
        {
            return Ok(ResidentCancelProgress::RequiresRestoreOrShutdown {
                request_id,
                session_id: *session_id,
            });
        }
        if let Some(pending) = self.output.cancellation(request_id) {
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
        if !self.output.cancellation_accepted(Some(request_id)) {
            if let Some(session_id) = self.scheduler.active_session_for_request(request_id) {
                self.output
                    .accept_scheduled_cancellation(request_id, session_id);
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
        let mut pending =
            self.output
                .take_cancellation(request_id)
                .ok_or_else(|| Error::Invariant {
                    message: format!("request {request_id:?} has no pending cancellation owner"),
                })?;
        if !pending.ready {
            self.output.retry_cancellation(request_id, pending);
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
                    self.output.retry_cancellation(request_id, pending);
                    return Err(error);
                }
            }
        }
        if let Err(error) = self
            .scheduler
            .retry_failed_slot_releases(&mut self.slot_pool)
        {
            self.output.retry_cancellation(request_id, pending);
            return Err(error);
        }
        if let Some(position) = self.sessions.retained_position_mut(&pending.session_id) {
            *position = 0;
        }
        if let Err(error) = self.release_sequence_state(pending.session_id) {
            self.output.retry_cancellation(request_id, pending);
            return Err(error);
        }
        if self.sessions.has_cleanup(&pending.session_id) {
            self.output.retry_cancellation(request_id, pending);
            return Ok(None);
        }
        let result = pending
            .result
            .expect("completed cancellation retains its scheduler result");
        #[cfg(test)]
        self.tick_trace
            .push(DriverTickEvent::CancelComplete(request_id));
        Ok(Some(result))
    }

    fn register_transaction_cancellation(
        &mut self,
        transaction: ExecutionTransactionId,
        request_id: RequestId,
        session_id: SessionId,
    ) -> Result<()> {
        self.output
            .accept_transaction_cancellation(transaction, request_id, session_id)
    }

    fn retry_pending_request_cancellations(&mut self) -> Result<()> {
        #[cfg(test)]
        self.tick_trace.push(DriverTickEvent::RequestCancellations);
        let requests = self.output.cancellation_requests();
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
        let requests = self.output.release_transaction_cancellations(transaction);
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
        if !self.output.cancellation_accepted(Some(request_id)) {
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
            match pending.phase.accept_cancellation(request_id) {
                ResidentCancellationRoute::Cleanup => {
                    return self.finish_resident_abort(pending, Some(request_id)).map(
                        |(_, cancellation)| match cancellation {
                            Some(result) => ResidentCancelProgress::Complete(result),
                            None => ResidentCancelProgress::Pending,
                        },
                    );
                }
                ResidentCancellationRoute::Publish => {
                    self.resident_transactions.insert(transaction, pending);
                    self.enqueue_transaction(transaction);
                    return Ok(ResidentCancelProgress::Pending);
                }
                ResidentCancellationRoute::Abort => {}
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
                    pending.phase = pending.phase.backend_completed(self.runtime_now_ns());
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
            PendingSpeculativeDriverCohort::AdmissionRollback(mut pending) => {
                pending.cancellation_request = Some(request_id);
                self.progress_speculative_admission_rollback(*pending)?;
                unreachable!("rejected admission reports its retained primary failure")
            }
            PendingSpeculativeDriverCohort::Bookkeeping(mut pending) => {
                pending.next.accept_cancellation(request_id);
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::Bookkeeping(pending),
                );
                return Ok(ResidentCancelProgress::Pending);
            }
            PendingSpeculativeDriverCohort::Proposing(pending) => {
                let result = self.cancel_native_proposal_cohort(*pending, Some(request_id), None);
                if let Some(retained) = self.speculative_transactions.get(&transaction) {
                    let backend_aborted = matches!(
                        retained,
                        PendingSpeculativeDriverCohort::Proposing(pending)
                            if pending.backend_aborted()
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
                let post_terminal = pending.ending.is_post_terminal();
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
                if self.output.cancellation_accepted(Some(request_id)) {
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
                if scheduler_finished && !self.output.cancellation_accepted(Some(request_id)) =>
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

    fn capture_committed_prefill_prefix(&mut self, action: &PrefillChunkAction) -> Result<()> {
        if !self.prefix.has_cached_session(&action.session_id) || self.has_live_transactions() {
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
                .sessions
                .page_slot(&action.session_id)
                .ok_or_else(|| Error::Invariant {
                    message: format!(
                        "committed prefix session {:?} has no page slot",
                        action.session_id
                    ),
                })?;
        let cached_model = {
            let source = self
                .sessions
                .sequence_state(&action.session_id)
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
                self.prefix
                    .queue_cleanup(PendingPrefixCleanup::model_only(cached_model));
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
            .prefix
            .insert_and_queue_cleanup(namespace, &tokens, payload, charge)
        {
            Ok(()) => self.progress_prefix_cleanups(),
            Err(kind) => {
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

    pub fn submit(&mut self, request: GenerateRequest) {
        if let Err(error) = self.try_submit(request) {
            tracing::warn!(%error, "resident request rejected");
        }
    }

    pub fn submit_at_position(&mut self, request: GenerateRequest, position_start: usize) {
        if let Err(error) = self.try_submit_at_position(request, position_start) {
            tracing::warn!(%error, "resident positioned request rejected");
        }
    }

    fn progress_pending_cleanups(&mut self) -> Result<()> {
        #[cfg(test)]
        self.tick_trace.push(DriverTickEvent::PendingCleanups);
        if self.has_live_transactions() {
            #[cfg(test)]
            self.tick_trace.push(DriverTickEvent::CleanupDeferred);
            return Ok(());
        }
        while let Some(kv) = self.kv.pop_abort() {
            if let Err((error, kv)) = self.progress_aborted_resident_kv(kv) {
                self.kv.retry_abort(kv);
                return Err(error);
            }
        }
        while let Some(retirement) = self.kv.pop_retirement() {
            if let Err((error, retirement)) = self.progress_kv_retirement(retirement) {
                self.kv.retry_retirement(retirement);
                return Err(error);
            }
        }
        let mut admission_failures = Vec::new();
        let admissions = self
            .sessions
            .cleanups()
            .filter_map(|(id, cleanup)| {
                matches!(cleanup, PendingSequenceCleanup::Admission { .. }).then_some(*id)
            })
            .collect::<Vec<_>>();
        for id in admissions {
            if let Err(error) = self.cleanup_prepared_admission(id) {
                admission_failures.push(CleanupStep::new("prepared admission undo", error));
            }
        }
        if let Err(error) = self
            .scheduler
            .retry_failed_slot_releases(&mut self.slot_pool)
        {
            admission_failures.push(CleanupStep::new("scheduler slot undo", error));
        }
        if !admission_failures.is_empty() {
            return Err(Error::with_cleanup_batch(
                "admission cleanup",
                Error::Invariant {
                    message: "admission cleanup incomplete".into(),
                },
                admission_failures,
            ));
        }
        self.progress_prefix_cleanups()?;
        let sessions = self.sessions.cleanup_ids().collect::<Vec<_>>();
        for session_id in sessions {
            self.progress_sequence_cleanup(session_id)?;
        }
        self.reap_request_identities();
        Ok(())
    }

    fn prepare_step(&mut self) -> Result<()> {
        #[cfg(test)]
        self.tick_trace.push(DriverTickEvent::PrepareStep);
        self.validate_configuration()?;
        self.admit_new_sequences()
    }

    fn no_action_step(&self) -> ResidentDriverStep {
        // Ending transactions still own suspended sequences even when the
        // scheduler has no runnable work. Pending Publish is not engine idle.
        if self.scheduler.is_idle() && !self.has_live_transactions() && !self.warmup_pending() {
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
        let Some(scheduled) = ScheduledBatch::from_action(&mut action, self.top_k)
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

        let (mut pending, execution_started_ns) =
            self.prepare_resident_transaction(action, scheduled)?;
        let transaction = pending.transaction;
        let resource_demand = Self::action_demand(&pending.action);
        let progress = self.executor.execute_prepared_batch(
            transaction,
            &mut pending.states,
            pending.scheduled.execution(),
        );
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

    fn drive_speculative_endings<F>(
        &mut self,
        on_token: &mut F,
    ) -> Result<Option<ResidentDriverStep>>
    where
        R: ResidentModelRunner,
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        #[cfg(test)]
        self.tick_trace.push(DriverTickEvent::SpeculativeEndings);
        let transactions = self
            .speculative_transactions
            .iter()
            .filter_map(|(transaction, pending)| match pending {
                PendingSpeculativeDriverCohort::Proposing(pending)
                    if pending.cancellation_request.is_some() || pending.abort_cause.is_some() =>
                {
                    Some(*transaction)
                }
                PendingSpeculativeDriverCohort::Ending(_)
                | PendingSpeculativeDriverCohort::AdmissionRollback(_) => Some(*transaction),
                PendingSpeculativeDriverCohort::Bookkeeping(_)
                | PendingSpeculativeDriverCohort::Proposing(_)
                | PendingSpeculativeDriverCohort::Verifying(_) => None,
            })
            .collect::<Vec<_>>();
        for transaction in transactions {
            let pending = self
                .speculative_transactions
                .remove(&transaction)
                .expect("ending speculative transaction was collected above");
            match pending {
                PendingSpeculativeDriverCohort::AdmissionRollback(pending) => {
                    self.progress_speculative_admission_rollback(*pending)?;
                }
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
                PendingSpeculativeDriverCohort::Bookkeeping(_)
                | PendingSpeculativeDriverCohort::Verifying(_) => {
                    unreachable!("collected speculative transaction must be ending")
                }
            }
        }
        Ok(None)
    }

    fn drive_resident_endings<F>(&mut self, on_token: &mut F) -> Result<Option<ResidentDriverStep>>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        #[cfg(test)]
        self.tick_trace.push(DriverTickEvent::ResidentEndings);
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
                PendingSpeculativeDriverCohort::Ending(_)
                | PendingSpeculativeDriverCohort::AdmissionRollback(_) => Some(*transaction),
                _ => None,
            })
            .collect::<Vec<_>>();
        for transaction in speculative {
            match self
                .speculative_transactions
                .remove(&transaction)
                .expect("ending speculative transaction was collected above")
            {
                PendingSpeculativeDriverCohort::Ending(pending) => {
                    self.drive_speculative_ending(*pending, on_token)?;
                }
                PendingSpeculativeDriverCohort::AdmissionRollback(pending) => {
                    self.progress_speculative_admission_rollback(*pending)?;
                }
                _ => unreachable!("collected speculative transaction was ending"),
            }
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
                pending.phase.is_post_terminal().then_some(*transaction)
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
                PendingSpeculativeDriverCohort::AdmissionRollback(_) => Some(*transaction),
                PendingSpeculativeDriverCohort::Proposing(pending) if pending.backend_aborted() => {
                    Some(*transaction)
                }
                PendingSpeculativeDriverCohort::Ending(pending)
                    if pending.ending.is_post_terminal() =>
                {
                    Some(*transaction)
                }
                PendingSpeculativeDriverCohort::Bookkeeping(_)
                | PendingSpeculativeDriverCohort::Proposing(_)
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
                PendingSpeculativeDriverCohort::AdmissionRollback(pending) => {
                    self.progress_speculative_admission_rollback(*pending)?;
                }
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
                PendingSpeculativeDriverCohort::Bookkeeping(_)
                | PendingSpeculativeDriverCohort::Verifying(_) => {
                    unreachable!("post-terminal speculative transaction cannot be verifying")
                }
            }
        }
        Ok(())
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
        if let Some(transaction) = self
            .speculative_transactions
            .iter()
            .find_map(|(id, pending)| {
                matches!(pending, PendingSpeculativeDriverCohort::Bookkeeping(_)).then_some(*id)
            })
        {
            return self.drive_speculative_bookkeeping(transaction, on_token);
        }
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
            && self.prefix.capacity() == 0
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
        let resumable = self.continuations.ready_snapshot_len();
        #[cfg(test)]
        self.tick_trace
            .push(DriverTickEvent::ReadySnapshot(resumable));
        for _ in 0..resumable {
            let Some(transaction) = self.pop_ready_transaction() else {
                break;
            };
            let ready_continuation = self
                .ready_continuation_for(transaction)
                .or_else(|| self.continuations.continuation_for(transaction));
            let Some(ready_continuation) = ready_continuation else {
                continue;
            };
            #[cfg(test)]
            self.tick_trace.push(DriverTickEvent::Resume(transaction));
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

            let cancellation_accepted = self.cancellation_accepted(action.request_id);
            let mut action_staged = 0usize;
            if arbitrate_terminal(cancellation_accepted, finish_reason)
                == TerminalDecision::Continue
            {
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
            if let TerminalDecision::Finish(reason) =
                arbitrate_terminal(cancellation_accepted, finish_reason)
            {
                action_finished =
                    usize::from(self.finish_successful_sequence(action.session_id, reason)?);
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

    // A synchronous busy-loop cannot be interrupted by an async timeout. Build
    // the non-Send driver on a separate thread so regressions fail in bounded time.
    fn assert_shutdown_returns(test: impl FnOnce() + Send + 'static) {
        let (done, completion) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            test();
            done.send(()).unwrap();
        });
        completion
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("shutdown must return without spinning");
        thread.join().unwrap();
    }

    fn inject_global_unknown(
        driver: &mut ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool>,
        handle: &crate::io::testing::MockPhysicalHandle,
    ) -> ferrule_common::ProviderFault {
        let fault = ferrule_common::ProviderFault {
            scope: None,
            failure: FailureReason::StorageUnavailable,
            quiescence: ferrule_common::QuiescenceEvidence::Unknown,
        };
        handle.push_fault(fault.clone());
        assert_eq!(driver.load_registry.collect_provider_completions(1), 1);
        assert!(matches!(
            driver.load_registry.process_one_completion(),
            Err(crate::io::RegistryError::Provider {
                source: FailureReason::StorageUnavailable,
            })
        ));
        fault
    }

    #[test]
    fn shutdown_global_unknown_empty_returns_pending_without_spinning() {
        assert_shutdown_returns(|| {
            let (physical, handle) = MockPhysicalProvider::manual();
            let mut driver = driver_from_runner(
                MockTopKRunner::new(Vec::new()).with_materialization_provider(Box::new(physical)),
            );
            let fault = inject_global_unknown(&mut driver, &handle);
            for _ in 0..3 {
                assert_eq!(driver.load_registry.active_operations(), 0);
                assert_eq!(driver.load_registry.pending_completions(), 0);
                assert_eq!(driver.load_registry.resources().active_grants(), 0);
                assert!(driver.load_registry.has_pending_owner_work());
                let wake_epoch = driver.completion_hub().epoch();
                assert_eq!(
                    driver.shutdown_progress(&mut |_| Ok(())).unwrap(),
                    ResidentShutdownProgress::Pending
                );
                assert_eq!(driver.completion_hub().epoch(), wake_epoch);
                assert_eq!(driver.load_registry.provider_faults(), &[fault.clone()]);
                assert!(matches!(
                    driver.shutdown(&mut |_| Ok(()), 512),
                    Err(Error::Registry { source }) if matches!(
                        *source,
                        crate::io::RegistryError::ShutdownIncomplete {
                            pending_operations: 0,
                            pending_completions: 0,
                            active_grants: 0,
                        }
                    )
                ));
            }
        });
    }

    #[test]
    fn shutdown_global_unknown_retries_pending_completions_and_preserves_credit_until_proof() {
        assert_shutdown_returns(|| {
            let (physical, handle) = MockPhysicalProvider::manual();
            handle.defer_cancellation_completion();
            let runner = MockTopKRunner::new(Vec::new())
                .with_resumable_wait_scripts([2])
                .with_materialization_request(materialization_request(3))
                .with_materialization_provider(Box::new(physical));
            let mut driver = concurrent_transaction_driver(runner);
            driver.submit(request(1, &[1], 2, Vec::new()));
            for _ in 0..2 {
                assert!(matches!(
                    driver.step(&mut |_| Ok(())).unwrap(),
                    ResidentDriverStep::WaitingForModelProgress(_)
                ));
            }
            let (operation, key) = handle
                .commands()
                .iter()
                .find_map(|command| match command {
                    MockPhysicalCommand::SubmitRead(operation, key, _) => Some((*operation, *key)),
                    _ => None,
                })
                .expect("a real read must own physical credit");
            let fault = inject_global_unknown(&mut driver, &handle);
            let read_credit = driver
                .load_registry
                .resources()
                .in_use(ResourceKind::ReadSlot);
            assert_eq!(read_credit, 1);
            assert_eq!(
                driver.shutdown_progress(&mut |_| Ok(())).unwrap(),
                ResidentShutdownProgress::Pending
            );
            assert_eq!(driver.load_registry.active_operations(), 1);
            assert_eq!(
                driver
                    .load_registry
                    .resources()
                    .in_use(ResourceKind::ReadSlot),
                read_credit
            );
            assert!(driver.load_registry.retirement(operation).is_none());

            // Stale completions fill the first 512-event batch. The actual
            // cancellation proof is queued after it and must still be consumed
            // in this call even though the global fault remains Unknown.
            for _ in 0..512 {
                handle.push_outcome(
                    OperationId::new(u64::MAX),
                    key,
                    LoadStage::ReadSubmitted,
                    CompletionOutcome::Cancelled(CancellationReason::OwnerShutdown),
                );
            }
            handle.push_outcome(
                operation,
                key,
                LoadStage::ReadSubmitted,
                CompletionOutcome::Cancelled(CancellationReason::OwnerShutdown),
            );
            assert_eq!(
                driver.shutdown_progress(&mut |_| Ok(())).unwrap(),
                ResidentShutdownProgress::Pending
            );
            assert_eq!(driver.load_registry.rejected_completions().len(), 512);
            assert_eq!(driver.load_registry.pending_completions(), 0);
            assert_eq!(driver.load_registry.active_operations(), 0);
            assert_eq!(driver.load_registry.resources().active_grants(), 0);
            assert!(driver.load_registry.retirement(operation).is_some());
            assert!(driver.load_registry.has_pending_owner_work());
            assert_eq!(driver.load_registry.provider_faults(), &[fault]);
            assert!(!driver.physical_shutdown_complete());
            assert!(driver.executor.runner().physical_owned);
            assert_eq!(driver.executor.runner().physical_shutdown_calls, 0);
        });
    }

    #[test]
    fn shutdown_global_unknown_close_and_extraction_preserve_unknown_custody() {
        assert_shutdown_returns(|| {
            use crate::engine::{
                InferenceEngine, InferenceShutdownProgress, ResidentInferenceEngine,
            };
            let (physical, handle) = MockPhysicalProvider::manual();
            let mut driver = driver_from_runner(
                MockTopKRunner::new(Vec::new()).with_materialization_provider(Box::new(physical)),
            );
            let fault = inject_global_unknown(&mut driver, &handle);
            for _ in 0..2 {
                assert_eq!(
                    driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
                    ResidentShutdownProgress::Pending
                );
                assert!(!driver.physical_shutdown_complete());
                assert!(driver.executor.runner().physical_owned);
                assert_eq!(driver.executor.runner().physical_shutdown_calls, 0);
            }
            let mut driver = match driver.try_into_runner() {
                Ok(_) => panic!("Unknown custody must prevent runner extraction"),
                Err(failure) => {
                    let (error, driver) = *failure;
                    assert!(matches!(error, Error::InvalidRequest { .. }));
                    driver
                }
            };
            assert_eq!(driver.load_registry.provider_faults(), &[fault]);
            assert!(driver.load_registry.has_pending_owner_work());
            assert_eq!(
                driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
                ResidentShutdownProgress::Pending
            );
            let mut engine = ResidentInferenceEngine::new(driver);
            assert_eq!(
                engine.shutdown().unwrap(),
                InferenceShutdownProgress::Pending
            );
            assert!(!engine.driver().physical_shutdown_complete());
            assert!(engine.driver().executor.runner().physical_owned);
            assert_eq!(engine.driver().executor.runner().physical_shutdown_calls, 0);
            assert_eq!(
                handle.command_count(|command| matches!(
                    command,
                    MockPhysicalCommand::PhysicalDropped
                )),
                0
            );
        });
    }

    #[test]
    fn physical_shutdown_error_propagates_after_zero_logical_pages_and_retains_runner() {
        use crate::engine::{InferenceEngine, InferenceShutdownProgress, ResidentInferenceEngine};
        let mut driver = driver_with_outputs(Vec::new())
            .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        driver.executor.runner_mut().physical_shutdown_failures = 1;
        let mut engine = ResidentInferenceEngine::new(driver);
        let error = engine.shutdown().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("injected physical shutdown failure")
        );
        let driver = engine.driver();
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        assert!(driver.scheduler().is_idle());
        assert!(!driver.physical_shutdown_complete());
        assert!(driver.executor.runner().physical_owned);
        assert_eq!(driver.executor.runner().physical_shutdown_calls, 1);

        assert_eq!(
            engine.shutdown().unwrap(),
            InferenceShutdownProgress::Complete
        );
        assert!(engine.driver().physical_shutdown_complete());
        assert!(!engine.driver().executor.runner().physical_owned);
        assert_eq!(engine.driver().executor.runner().physical_shutdown_calls, 2);
        assert_eq!(
            engine.shutdown().unwrap(),
            InferenceShutdownProgress::Complete
        );
        assert_eq!(engine.driver().executor.runner().physical_shutdown_calls, 2);
    }

    #[test]
    fn physical_shutdown_unknown_retry_does_not_clear_quarantine_or_release_custody() {
        use crate::engine::{InferenceEngine, ResidentInferenceEngine};
        let mut driver = driver_with_outputs(Vec::new())
            .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        driver.executor.runner_mut().physical_shutdown_failures = 1;
        driver.executor.runner_mut().physical_shutdown_unknown = true;
        let mut engine = ResidentInferenceEngine::new(driver);
        assert!(
            engine
                .shutdown()
                .unwrap_err()
                .to_string()
                .contains("injected physical")
        );
        for _ in 0..2 {
            assert!(
                engine
                    .shutdown()
                    .unwrap_err()
                    .to_string()
                    .contains("remains quarantined")
            );
            assert_eq!(engine.driver().page_manager().unwrap().allocated_pages(), 0);
            assert!(!engine.driver().physical_shutdown_complete());
            assert!(engine.driver().executor.runner().physical_owned);
            assert!(engine.driver().executor.runner().physical_quarantined);
        }
        assert_eq!(engine.driver().executor.runner().physical_shutdown_calls, 3);
    }

    #[test]
    fn physical_shutdown_is_not_attempted_before_logical_custody_drains() {
        let mut driver = driver_with_outputs(vec![top(b'a' as u32)])
            .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        driver.submit(request(1, &[1], 3, Vec::new()));
        driver.step(&mut |_| Ok(())).unwrap();
        driver.executor.runner_mut().release_failures_remaining = 1;
        assert!(driver.shutdown_and_close_progress(&mut |_| Ok(())).is_err());
        assert_eq!(driver.executor.runner().physical_shutdown_calls, 0);
        assert!(!driver.physical_shutdown_complete());
        assert!(matches!(
            driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
            ResidentShutdownProgress::Complete(_)
        ));
        assert_eq!(driver.executor.runner().physical_shutdown_calls, 1);
    }

    #[tokio::test]
    async fn physical_shutdown_local_concrete_owner_propagates_and_can_retry() {
        let mut driver = driver_with_outputs(Vec::new());
        driver.executor.runner_mut().physical_shutdown_failures = 1;
        let mut engine = crate::engine::LocalResidentInferenceEngine::new(driver);
        assert!(
            engine
                .shutdown()
                .await
                .unwrap_err()
                .to_string()
                .contains("injected physical")
        );
        assert!(engine.driver().executor.runner().physical_owned);
        engine.shutdown().await.unwrap();
        engine.shutdown().await.unwrap();
        assert_eq!(engine.driver().executor.runner().physical_shutdown_calls, 2);
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
        preempt_calls: usize,
        restore_calls: usize,
        preempt_errors: usize,
        restore_errors: usize,
        physical_shutdown_calls: usize,
        physical_shutdown_failures: usize,
        physical_shutdown_unknown: bool,
        physical_quarantined: bool,
        physical_owned: bool,
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
        packed_resume_calls: usize,
        proposal_waits: VecDeque<usize>,
        active_native_proposals: HashMap<ContinuationId, MockPendingProposal>,
        next_native_proposal_continuation: u64,
        reject_proposal_swap: bool,
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
        publish_progress: VecDeque<ModelResult<TransactionEndProgress>>,
        end_intents: Vec<(ExecutionTransactionId, TransactionEndIntent)>,
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
                preempt_calls: 0,
                restore_calls: 0,
                preempt_errors: 0,
                restore_errors: 0,
                physical_shutdown_calls: 0,
                physical_shutdown_failures: 0,
                physical_shutdown_unknown: false,
                physical_quarantined: false,
                physical_owned: true,
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
                packed_resume_calls: 0,
                proposal_waits: VecDeque::new(),
                active_native_proposals: HashMap::new(),
                next_native_proposal_continuation: 100,
                reject_proposal_swap: false,
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
                publish_progress: VecDeque::new(),
                end_intents: Vec::new(),
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

        fn shutdown_physical(&mut self) -> ModelResult<()> {
            self.physical_shutdown_calls += 1;
            if self.physical_quarantined {
                return Err(ModelError::Execution {
                    message: "physical owner remains quarantined".into(),
                });
            }
            if self.physical_shutdown_failures != 0 {
                self.physical_shutdown_failures -= 1;
                self.physical_quarantined = self.physical_shutdown_unknown;
                return Err(ModelError::Execution {
                    message: "injected physical shutdown failure".into(),
                });
            }
            self.physical_owned = false;
            Ok(())
        }

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
            _: ExecutionTransactionId,
            _: u32,
        ) -> ModelResult<NativeProposalProgress> {
            Err(ModelError::Execution {
                message: "PR14 driver must use explicit proposal begin".into(),
            })
        }

        fn resume_native_proposal(
            &mut self,
            _: ExecutionTransactionId,
            _: ContinuationId,
            _: ResidencyLeaseSet,
        ) -> ModelResult<NativeProposalProgress> {
            Err(ModelError::Execution {
                message: "PR14 driver must use explicit proposal resume".into(),
            })
        }

        fn begin_native_proposal_for(
            &mut self,
            _state: &mut Self::SequenceState,
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

        fn resume_native_proposal_for(
            &mut self,
            _state: &mut Self::SequenceState,
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
            if self.reject_proposal_swap {
                return Err(ModelError::Execution {
                    message: "PR14 proposal owner rejects the legacy sequence swap".into(),
                });
            }
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
            self.preempt_calls += 1;
            if self.preempt_errors > 0 {
                self.preempt_errors -= 1;
                return Err(ModelError::Execution {
                    message: "physical preempt unknown".into(),
                });
            }
            Ok(())
        }

        fn restore_kv_pages(
            &mut self,
            _pages: &[ferrule_common::execution::KvPageId],
        ) -> ModelResult<()> {
            self.restore_calls += 1;
            if self.restore_errors > 0 {
                self.restore_errors -= 1;
                return Err(ModelError::Execution {
                    message: "physical restore unknown".into(),
                });
            }
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
            self.end_intents.push((transaction, intent));
            match intent {
                TransactionEndIntent::Publish => {
                    match self.publish_progress.pop_front().transpose()? {
                        Some(TransactionEndProgress::Pending) => {
                            self.completion_hub.notify();
                            return Ok(TransactionEndProgress::Pending);
                        }
                        Some(TransactionEndProgress::Complete) | None => {}
                    }
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
            self.packed_resume_calls += 1;
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
        driver
            .track_kv_page_grants(vec![source_page], vec![source_grant])
            .unwrap();
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
        assert!(driver.sessions.is_owned(&SessionId(1)));
        assert!(driver.sessions.contains_sequence_state(&SessionId(2)));
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
        assert!(driver.continuations.transaction_count() == 0);
        assert!(driver.sessions.owner_count() == 0);
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
                    .all(|continuation| !driver.continuations.contains(*continuation))
            );
            assert!(!driver.continuations.has_pending_failures());
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
        assert!(driver.continuations.contains(continuation));
        assert!(!driver.continuations.has_pending_failures());
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
        assert!(!driver.continuations.contains(continuation));
        assert!(!driver.continuations.has_pending_failures());
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
        assert!(driver.continuations.transaction_count() == 0);
        assert!(driver.sessions.owner_count() == 0);
        assert!(driver.sessions.cleanup_count() == 0);
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
        assert_eq!(driver.kv.grant_count(), 2);
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
        assert!(driver.kv.has_grant(&source_page));
        assert!(!driver.kv.has_grant(&cow.replacement));
        assert_eq!(driver.kv.grant_count(), 1);
        assert_eq!(
            driver
                .load_registry()
                .resources()
                .in_use(ResourceKind::KvPage),
            1
        );
        assert_eq!(driver.page_manager().unwrap().stats().retiring_pages, 0);

        retire_test_sequence(&mut driver, target);
        assert_eq!(driver.kv.grant_count(), 1);
        retire_test_sequence(&mut driver, source);
        assert!(driver.kv.grants_empty());
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
        assert_eq!(driver.kv.grant_count(), 1);
        assert!(driver.kv.has_grant(&source_page));
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
        assert!(driver.kv.grants_empty());
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
        assert_eq!(events.len(), 1);

        assert!(matches!(
            step(&mut driver, &mut events).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(ref pending) if pending.len() == 1
        ));
        assert_eq!(events.len(), 1);
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
        assert!(driver.sessions.is_owned(&SessionId(1)));
        let committed_batches = driver.executor().runner().committed_batches;

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            }
        ));
        assert!(driver.speculative_transactions.is_empty());
        assert!(driver.sessions.owner_count() == 0);
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
        assert!(driver.sessions.is_owned(&SessionId(1)));
        let rolled_back_batches = driver.executor().runner().rolled_back_batches;

        let business_error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            business_error
                .to_string()
                .contains("simulated resumable batch failure")
        );
        assert!(driver.speculative_transactions.is_empty());
        assert!(driver.sessions.owner_count() == 0);
        assert!(driver.continuations.is_empty());
        assert!(driver.continuations.transaction_count() == 0);
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
        assert_eq!(driver.sessions.owner(&SessionId(1)).unwrap(), transaction);
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));

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
        assert_eq!(driver.sessions.owner(&SessionId(1)).unwrap(), transaction);
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));
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
        assert_eq!(driver.sessions.owner(&SessionId(1)).unwrap(), transaction);
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));
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
        assert!(pending.backend_aborted());
        assert!(pending.has_pending_custody());
        assert_eq!(pending.source_states.len(), 1);
        assert_eq!(pending.schedules.len(), 1);
        assert!(driver.sessions.is_owned(&SessionId(1)));
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
        assert!(driver.sessions.owner_count() == 0);
        assert!(driver.continuations.is_empty());
        assert!(driver.continuations.transaction_count() == 0);
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
        assert!(driver.output.cancellations_empty());
        assert!(driver.sessions.owner_count() == 0);
        assert!(driver.continuations.is_empty());
        assert!(driver.continuations.transaction_count() == 0);
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
        driver.continuations.exhaust_dependency_epochs();

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
        assert!(driver.continuations.transaction_count() == 0);
        assert!(driver.sessions.owner_count() == 0);
        assert!(driver.sessions.contains_sequence_state(&SessionId(1)));
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
        driver.continuations.exhaust_dependency_epochs();

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
        assert!(driver.continuations.transaction_count() == 0);
        assert!(driver.sessions.owner_count() == 0);
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));
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
    fn pr14_driver_explicit_proposal_route_bypasses_legacy_swap_gate() {
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
        driver.executor.runner_mut().reject_proposal_swap = true;
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
        let continuation = driver
            .continuations
            .continuation_for(driver.sessions.owner(&SessionId(1)).unwrap())
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
        assert!(driver.continuations.contains(continuation));
        assert!(
            driver
                .continuations
                .continuation_for(driver.sessions.owner(&SessionId(1)).unwrap())
                == Some(continuation)
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
        assert!(driver.continuations.transaction_count() == 0);
        assert!(driver.sessions.owner_count() == 0);
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
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));
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
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));
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
        assert!(driver.output.cancellations_empty());
        assert!(driver.sessions.cleanup_count() == 0);
        assert!(driver.sessions.owner_count() == 0);
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
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));

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
        assert!(driver.sessions.contains_sequence_state(&SessionId(1)));
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
        assert!(!driver.sessions.contains_sequence_state(&SessionId(77)));
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
        assert!(driver.sessions.contains_sequence_state(&SessionId(77)));
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
            let slot = *driver.sessions.page_slot(&session_id).unwrap();
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
        assert_eq!(driver.sessions.owner_count(), 1);
        let owned_session = driver.sessions.owner_ids().next().unwrap();
        let runnable_session = if owned_session == SessionId(1) {
            SessionId(2)
        } else {
            SessionId(1)
        };
        assert!(!driver.sessions.contains_sequence_state(&owned_session));
        assert!(driver.sessions.contains_sequence_state(&runnable_session));
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
        let owned_slot = *driver.sessions.page_slot(&owned_session).unwrap();
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
        assert!(driver.sessions.owner_count() == 0);
        assert!(driver.sessions.contains_sequence_state(&owned_session));
        assert!(driver.sessions.contains_sequence_state(&runnable_session));
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
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));
        assert!(driver.sessions.contains_sequence_state(&SessionId(2)));
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
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));
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
        assert_eq!(
            driver
                .sessions
                .sequence_state(&SessionId(1))
                .unwrap()
                .position,
            2
        );
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
        driver.continuations.exhaust_dependency_epochs();
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
        assert!(driver.continuations.transaction_count() == 0);
        assert!(driver.sessions.owner_count() == 0);
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
        driver.continuations.exhaust_dependency_epochs();
        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("dependency-set epoch space is exhausted")
        );
        assert!(driver.resident_transactions.is_empty());
        assert!(driver.continuations.is_empty());
        assert!(driver.continuations.transaction_count() == 0);
        assert!(driver.sessions.owner_count() == 0);
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
        assert!(driver.sessions.is_owned(&SessionId(1)));
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);

        let business_error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            business_error
                .to_string()
                .contains("simulated resumable batch failure")
        );
        assert!(driver.resident_transactions.is_empty());
        assert!(driver.sessions.owner_count() == 0);
        assert!(driver.continuations.is_empty());
        assert!(driver.continuations.transaction_count() == 0);
        assert!(driver.kv.grants_empty());
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
        assert!(driver.continuations.transaction_count() != 0);
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));
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
        assert!(driver.continuations.transaction_count() == 0);
        assert!(driver.sessions.owner_count() == 0);
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
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));
        assert!(driver.sessions.contains_sequence_state(&SessionId(2)));

        let cancelled = driver.cancel_request(RequestId(2)).unwrap();
        assert_eq!(cancelled, ResidentCancelProgress::Pending);
        assert!(driver.output.cancellation_accepted(Some(RequestId(2))));
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
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));
        assert!(driver.sessions.contains_sequence_state(&SessionId(2)));
        assert!(driver.sessions.has_cleanup(&SessionId(2)));
        assert!(driver.scheduler().active_sequence(SessionId(1)).is_none());
        assert!(driver.scheduler().active_sequence(SessionId(2)).is_none());
        assert!(driver.sessions.is_owned(&SessionId(1)));
        assert!(!driver.sessions.is_owned(&SessionId(2)));
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
        assert!(driver.sessions.has_cleanup(&SessionId(2)));

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            }
        ));
        assert!(!driver.sessions.has_cleanup(&SessionId(2)));
        assert!(driver.output.cancellations_empty());
        assert!(!driver.sessions.contains_sequence_state(&SessionId(2)));
        assert!(!driver.sessions.has_page_slot(&SessionId(2)));
        assert_eq!(
            driver.cancel_request(RequestId(2)).unwrap(),
            ResidentCancelProgress::Complete(CancelRequestResult::Active {
                request_id: RequestId(2),
                session_id: SessionId(2),
            })
        );
        assert_eq!(
            driver.page_manager().unwrap().active_sequences(),
            usize::from(driver.sessions.has_page_slot(&SessionId(1)))
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
        let transaction_a = driver.sessions.owner(&SessionId(1)).unwrap();
        let transaction_b = driver.sessions.owner(&SessionId(2)).unwrap();
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
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));
        assert!(driver.sessions.contains_sequence_state(&SessionId(2)));
        assert_eq!(driver.executor().runner().committed_batches, 1);

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Prefill,
                ..
            }
        ));
        assert!(driver.resident_transactions.is_empty());
        assert!(driver.sessions.owner_count() == 0);
        assert!(driver.sessions.contains_sequence_state(&SessionId(1)));
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
        let packed_owner = driver.sessions.owner_ids().next().unwrap();
        let siblings = (1..=4)
            .map(SessionId)
            .filter(|session_id| *session_id != packed_owner)
            .collect::<Vec<_>>();
        let sibling_pages = siblings
            .iter()
            .flat_map(|session_id| {
                let slot = *driver.sessions.page_slot(session_id).unwrap();
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
        assert_eq!(driver.output.cancellation_count(), 3);
        assert_eq!(driver.sessions.cleanup_count(), 3);
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 4);
        for session_id in &siblings {
            assert!(driver.sessions.has_page_slot(session_id));
            assert!(driver.sessions.contains_sequence_state(session_id));
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
        assert_eq!(driver.sessions.cleanup_count(), 3);

        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            }
        ));
        assert!(driver.sessions.cleanup_count() == 0);
        assert!(driver.output.cancellations_empty());
        assert!(driver.kv.pending_retirements_empty());
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 1);
        for session_id in &siblings {
            assert!(!driver.sessions.has_page_slot(session_id));
            assert!(!driver.sessions.contains_sequence_state(session_id));
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
        assert_eq!(driver.kv.pending_retirement_count(), 1);
        assert_eq!(driver.sessions.cleanup_count(), 1);
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
        assert_eq!(driver.sessions.cleanup_count(), 1);
        assert_eq!(driver.executor().runner().released_sequence_states, 1);

        assert_eq!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Idle
        );
        assert!(driver.kv.pending_retirements_empty());
        assert!(driver.sessions.cleanup_count() == 0);
        assert!(driver.sessions.sequence_state_count() == 0);
        assert!(driver.sessions.page_slot_count() == 0);
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
        let transaction_b = driver.sessions.owner(&SessionId(2)).unwrap();

        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Pending
        );
        assert!(driver.output.cancellation_accepted(Some(RequestId(1))));
        assert_eq!(driver.resident_transactions.len(), 1);
        assert!(driver.resident_transactions.contains_key(&transaction_b));
        assert_eq!(driver.executor().runner().cancelled_continuations.len(), 1);
        assert_eq!(driver.executor().runner().rolled_back_batches, 1);
        assert_eq!(driver.executor().runner().committed_batches, 0);
        assert!(!driver.sessions.is_owned(&SessionId(1)));
        assert_eq!(driver.sessions.owner(&SessionId(2)).unwrap(), transaction_b);
        assert_eq!(driver.kv.pending_retirement_count(), 1);
        assert!(driver.sessions.has_cleanup(&SessionId(1)));
        assert!(driver.sessions.contains_sequence_state(&SessionId(1)));
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
        assert!(driver.sessions.has_cleanup(&SessionId(1)));

        let _ = driver.step(&mut |_| Ok(())).unwrap();
        assert!(driver.kv.pending_retirements_empty());
        assert!(!driver.sessions.has_cleanup(&SessionId(1)));
        assert!(driver.output.cancellations_empty());
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));
        assert!(!driver.sessions.has_page_slot(&SessionId(1)));
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
        let continuation = driver
            .continuations
            .continuation_for(driver.sessions.owner(&SessionId(1)).unwrap())
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
        assert!(driver.continuations.contains(continuation));
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
        assert!(driver.continuations.transaction_count() == 0);
        assert!(driver.sessions.owner_count() == 0);
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
        let transaction_a = driver.sessions.owner(&SessionId(1)).unwrap();
        let transaction_b = driver.sessions.owner(&SessionId(2)).unwrap();

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
        let transaction = driver.sessions.owner(&SessionId(1)).unwrap();
        assert_eq!(driver.sessions.owner(&SessionId(2)).unwrap(), transaction);

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
        assert!(driver.output.cancellations_empty());
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

    fn early_driver(
        outputs: Vec<Vec<TokenLogit>>,
    ) -> ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool> {
        let mut driver = driver_with_outputs(outputs);
        driver.config.enable_native_proposals = false;
        driver
    }

    fn early_pending_publish_driver(
        outputs: Vec<Vec<TokenLogit>>,
    ) -> ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool> {
        let mut driver = early_driver(outputs)
            .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        driver.executor.runner_mut().publish_progress = VecDeque::from([
            Ok(TransactionEndProgress::Complete), // prefill
            Ok(TransactionEndProgress::Pending),  // final token append
        ]);
        driver
    }

    #[test]
    fn early_pending_publish_final_token_cancel_wins_over_max_tokens() {
        let session = SessionId(1);
        let mut driver = early_pending_publish_driver(vec![top(65), top(66)]);
        driver.retain_session(session).unwrap();
        let mut req = request(1, &[9], 1, vec![]);
        req.session_id = Some(session);
        driver.submit(req);
        let mut events = vec![];
        driver
            .step(&mut |e| {
                events.push(e.clone());
                Ok(())
            })
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(driver.stats().emitted_tokens, 1);
        assert_eq!(
            driver
                .step(&mut |_| panic!("delivered token replayed"))
                .unwrap(),
            ResidentDriverStep::Blocked
        );
        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Pending
        );
        assert!(driver.drain_finished().is_empty());
        assert!(driver.drain_cancelled().is_empty());
        assert!(driver.has_live_transactions());
        assert_eq!(driver.slot_pool().active_count(), 1);
        assert!(driver.page_manager().unwrap().allocated_pages() > 0);
        assert_eq!(driver.executor.runner().released_sequence_states, 0);
        driver
            .step(&mut |_| panic!("delivered token replayed"))
            .unwrap();
        assert!(
            driver.drain_finished().is_empty(),
            "accepted cancellation must not become Finished"
        );
        let cancelled = driver.drain_cancelled();
        assert_eq!(cancelled.len(), 1);
        assert_eq!(
            cancelled[0].finish_reason,
            Some(SequenceFinishReason::Cancelled)
        );
        assert_eq!(cancelled[0].tokens, vec![9, 65]);
        assert_eq!(cancelled[0].position, 2);
        assert_eq!(cancelled[0].generated, 1);
        assert_eq!(cancelled[0].generated_text, "A");
        assert_eq!(driver.retained_session_position(session), Some(0));
        assert_eq!(driver.stats().decode_steps, 1);
        assert_eq!(driver.stats().finished_sequences, 0);
        assert_eq!(driver.stats().emitted_tokens, 1);
        assert_eq!(driver.executor.runner().committed_batches, 2);
        assert_eq!(driver.executor.runner().rolled_back_batches, 0);
        assert!(
            driver
                .executor
                .runner()
                .end_intents
                .iter()
                .all(|(_, intent)| *intent == TransactionEndIntent::Publish)
        );
        assert!(driver.output.cancellations_empty());
        assert!(driver.output.early_empty());
        assert!(driver.output.outbox_empty());
        assert_eq!(driver.slot_pool().active_count(), 0);
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        assert_eq!(
            driver
                .step(&mut |_| panic!("replayed after cancellation"))
                .unwrap(),
            ResidentDriverStep::Idle
        );
        assert!(driver.drain_finished().is_empty());
        assert!(driver.drain_cancelled().is_empty());
        driver.reset_session(session).unwrap();
        let mut next = request(2, &[10], 1, vec![]);
        next.session_id = Some(session);
        driver.submit(next);
        driver
            .drive_ready_test_work(|e| {
                assert_eq!(e.index, 0);
                Ok(())
            })
            .unwrap();
        assert_eq!(driver.retained_session_position(session), Some(2));
    }

    #[test]
    fn early_pending_publish_session_engine_cancel_beats_all_output_terminals() {
        use crate::engine::{
            InferenceCancelProgress, InferenceEngine, ResidentInferenceEngine,
            SessionInferenceEngine,
        };
        for (limit, stop, next, eos, context, prefill_only) in [
            (1, vec![], top(66), None, 16, false),
            (3, vec!["A".into()], top(66), None, 16, false),
            (3, vec![], top(66), Some(66), 16, false),
            (3, vec![], top(66), None, 2, false),
            (3, vec![], vec![], None, 16, false),
            (0, vec![], top(66), None, 16, true),
        ] {
            let mut driver = early_pending_publish_driver(vec![top(65), next]);
            driver.config.ctx_size = context;
            driver.executor.runner_mut().eos = eos;
            if prefill_only {
                driver.executor.runner_mut().publish_progress.pop_front();
            }
            // Exercise the existing object-safe session engine, not a new engine.
            let mut engine: Box<dyn SessionInferenceEngine> =
                Box::new(ResidentInferenceEngine::new(driver));
            engine.retain_session(SessionId(1)).unwrap();
            let mut req = request(1, &[9], limit, stop);
            req.session_id = Some(SessionId(1));
            if 1 + limit > context {
                let admission = engine.admission_snapshot();
                let capacity = engine.capacity_snapshot();
                let stats = engine.observability_snapshot().driver;
                let position = engine.retained_session_position(SessionId(1));
                assert!(matches!(
                    engine.try_submit(req.clone()),
                    Err(Error::Admission {
                        source: RuntimeAdmissionError::InvalidPosition {
                            position: 0,
                            prompt_tokens: 1,
                            context: 0,
                        }
                    })
                ));
                assert_eq!(engine.admission_snapshot(), admission);
                assert_eq!(engine.capacity_snapshot(), capacity);
                assert_eq!(engine.observability_snapshot().driver, stats);
                assert_eq!(engine.retained_session_position(SessionId(1)), position);
                assert!(matches!(
                    engine.request_cleanup(RequestId(1)),
                    crate::engine::InferenceRequestCleanup::Unavailable
                ));
                assert!(engine.take_request_terminal(RequestId(1)).is_none());
                assert!(engine.drain_finished().is_empty());
                assert!(engine.drain_cancelled().is_empty());
                assert!(engine.drain_failed().is_empty());
                assert_eq!(
                    engine
                        .step(&mut |_| panic!("rejected request emitted output"))
                        .unwrap(),
                    ResidentDriverStep::Idle
                );
                assert_eq!(engine.admission_snapshot(), admission);
                assert_eq!(engine.capacity_snapshot(), capacity);
                assert_eq!(engine.observability_snapshot().driver, stats);

                // Rejection must not consume the request/session identity.
                req.max_new_tokens = context - req.prompt_tokens.len();
                engine.try_submit(req).unwrap();
                assert_eq!(
                    engine.cancel_request(RequestId(1)).unwrap(),
                    InferenceCancelProgress::Complete(CancelRequestResult::Waiting {
                        request_id: RequestId(1),
                        session_id: SessionId(1),
                    })
                );
                assert_eq!(engine.drain_cancelled().len(), 1);
                assert_eq!(engine.admission_snapshot(), admission);
                assert_eq!(engine.capacity_snapshot(), capacity);
                assert_eq!(engine.retained_session_position(SessionId(1)), position);
                continue;
            }
            engine.try_submit(req).unwrap();
            let mut events = vec![];
            if !prefill_only {
                engine
                    .step(&mut |e| {
                        events.push(e.clone());
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(events.len(), 1);
            }
            assert_eq!(
                engine.step(&mut |_| panic!("unexpected delivery")).unwrap(),
                ResidentDriverStep::Blocked
            );
            assert_eq!(
                engine.cancel_request(RequestId(1)).unwrap(),
                InferenceCancelProgress::Pending
            );
            assert!(engine.take_request_terminal(RequestId(1)).is_none());
            engine
                .step(&mut |_| panic!("cancelled output replayed"))
                .unwrap();
            assert!(engine.drain_finished().is_empty());
            let Some(crate::scheduling::RequestTerminal::Cancelled(sequence)) =
                engine.take_request_terminal(RequestId(1))
            else {
                panic!("pending cancellation lost: limit={limit}, prefill_only={prefill_only}");
            };
            assert_eq!(sequence.generated, usize::from(!prefill_only));
            assert_eq!(sequence.position, 1 + usize::from(!prefill_only));
            assert_eq!(engine.retained_session_position(SessionId(1)), Some(0));
            assert!(engine.drain_cancelled().is_empty());
            assert!(engine.take_request_terminal(RequestId(1)).is_none());
            assert!(engine.drain_failed().is_empty());
            assert_eq!(engine.observability_snapshot().driver.finished_sequences, 0);
            engine.reset_session(SessionId(1)).unwrap();
            assert_eq!(
                engine.step(&mut |_| panic!("replay")).unwrap(),
                ResidentDriverStep::Idle
            );
        }
    }

    #[test]
    fn early_pending_publish_error_keeps_cancellation_and_custody_until_complete() {
        let mut driver = early_pending_publish_driver(vec![top(65), top(66)]);
        driver.executor.runner_mut().publish_progress.extend([
            Err(ModelError::Execution {
                message: "publish custody unknown".into(),
            }),
            Ok(TransactionEndProgress::Pending),
            Ok(TransactionEndProgress::Complete),
        ]);
        driver.retain_session(SessionId(1)).unwrap();
        let mut req = request(1, &[9], 1, vec![]);
        req.session_id = Some(SessionId(1));
        driver.submit(req);
        driver.step(&mut |_| Ok(())).unwrap();
        assert_eq!(
            driver.step(&mut |_| panic!("replay")).unwrap(),
            ResidentDriverStep::Blocked
        );
        assert_eq!(
            driver.cancel_request(RequestId(1)).unwrap(),
            ResidentCancelProgress::Pending
        );
        let pages = driver.page_manager().unwrap().allocated_pages();
        for unknown in [true, false] {
            let step = driver.step(&mut |_| panic!("replay"));
            if unknown {
                assert!(
                    step.unwrap_err()
                        .to_string()
                        .contains("publish custody unknown")
                );
            } else {
                assert_eq!(step.unwrap(), ResidentDriverStep::Blocked);
                assert!(driver.has_pending_async_work());
            }
            assert!(driver.take_request_terminal(RequestId(1)).is_none());
            assert_eq!(driver.resident_transactions.len(), 1);
            assert_eq!(driver.output.cancellation_count(), 1);
            assert_eq!(driver.sessions.owner_count(), 1);
            assert_eq!(driver.slot_pool().active_count(), 1);
            assert_eq!(driver.page_manager().unwrap().allocated_pages(), pages);
            assert_eq!(driver.executor.runner().committed_batches, 1);
            assert_eq!(driver.executor.runner().released_sequence_states, 0);
            assert!(driver.executor.runner().released_kv_pages.is_empty());
            assert!(driver.reset_session(SessionId(1)).is_err());
        }
        driver.step(&mut |_| panic!("replay")).unwrap();
        assert!(driver.drain_finished().is_empty());
        assert!(driver.drain_failed().is_empty());
        assert_eq!(driver.drain_cancelled().len(), 1);
        assert!(driver.drain_cancelled().is_empty());
        assert_eq!(driver.retained_session_position(SessionId(1)), Some(0));
        assert_eq!(driver.executor.runner().committed_batches, 2);
        assert_eq!(driver.executor.runner().rolled_back_batches, 0);
        let intents = &driver.executor.runner().end_intents;
        assert_eq!(intents.len(), 5);
        assert!(
            intents
                .iter()
                .all(|(_, intent)| *intent == TransactionEndIntent::Publish)
        );
        assert!(intents[1..].iter().all(|(id, _)| id.get() == 2));
        assert!(driver.output.cancellations_empty());
        assert!(!driver.has_live_transactions());
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        assert_eq!(driver.stats().emitted_tokens, 1);
    }

    #[test]
    fn early_pending_publish_completed_before_cancel_keeps_successful_frontier() {
        use crate::engine::{
            InferenceCancelProgress, InferenceEngine, ResidentInferenceEngine,
            SessionInferenceEngine,
        };
        let driver = early_pending_publish_driver(vec![top(65), top(66)]);
        let mut engine = ResidentInferenceEngine::new(driver);
        engine.retain_session(SessionId(1)).unwrap();
        let mut req = request(1, &[9], 1, vec![]);
        req.session_id = Some(SessionId(1));
        engine.try_submit(req).unwrap();
        engine.step(&mut |_| Ok(())).unwrap();
        assert_eq!(
            engine.step(&mut |_| panic!("replay")).unwrap(),
            ResidentDriverStep::Blocked
        );
        engine.step(&mut |_| panic!("replay")).unwrap();
        assert_eq!(engine.retained_session_position(SessionId(1)), Some(2));
        assert_eq!(
            engine.cancel_request(RequestId(1)).unwrap(),
            InferenceCancelProgress::Complete(CancelRequestResult::NotFound {
                request_id: RequestId(1)
            })
        );
        let finished = engine.drain_finished();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].tokens, vec![9, 65]);
        assert_eq!(
            finished[0].finish_reason,
            Some(SequenceFinishReason::MaxTokens)
        );
        assert!(engine.drain_cancelled().is_empty());
        assert!(engine.take_request_terminal(RequestId(1)).is_none());
        assert_eq!(engine.retained_session_position(SessionId(1)), Some(2));
        assert_eq!(engine.driver().executor.runner().committed_batches, 2);
        assert_eq!(engine.driver().executor.runner().rolled_back_batches, 0);
        engine.reset_session(SessionId(1)).unwrap();
        assert_eq!(engine.retained_session_position(SessionId(1)), Some(0));
    }

    #[test]
    fn early_stream_profiles_selection_commit_not_next_forward_and_keeps_final_kv() {
        let mut driver = early_driver(vec![top(65), top(66), top(67), top(68)]);
        driver.retain_session(SessionId(1)).unwrap();
        let mut req = request(1, &[9], 3, vec![]);
        req.session_id = Some(SessionId(1));
        driver.submit(req);
        let mut events = Vec::new();
        for forward in 1..=4 {
            driver
                .step(&mut |event| {
                    events.push(event.clone());
                    Ok(())
                })
                .unwrap();
            assert_eq!(driver.executor().runner().committed_batches, forward);
            assert_eq!(events.len(), forward.min(3));
            assert_eq!(driver.stats().emitted_tokens, forward.min(3));
            if forward < 4 {
                assert!(driver.drain_finished().is_empty());
                let sequence = driver.scheduler().active_sequence(SessionId(1)).unwrap();
                assert_eq!(sequence.position, forward);
                assert_eq!(sequence.generated, forward - 1);
                assert_eq!(sequence.tokens.len(), forward);
                let mut fork = request(3, &[10], 1, vec![]);
                fork.session_id = Some(SessionId(3));
                assert!(
                    driver
                        .fork_session_exact(SessionId(1), fork, forward)
                        .unwrap_err()
                        .to_string()
                        .contains("KV append")
                );
            }
        }
        assert_eq!(
            events
                .iter()
                .map(|e| (e.index, e.token, e.text.as_str()))
                .collect::<Vec<_>>(),
            vec![(0, 65, "A"), (1, 66, "B"), (2, 67, "C")]
        );
        let finished = driver.drain_finished();
        assert_eq!(finished[0].tokens, vec![9, 65, 66, 67]);
        assert_eq!(finished[0].generated_text, "ABC");
        assert_eq!(
            finished[0].finish_reason,
            Some(SequenceFinishReason::MaxTokens)
        );
        assert_eq!(driver.retained_session_position(SessionId(1)), Some(4));
        assert_eq!(driver.stats().prefill_chunks, 1);
        assert_eq!(driver.stats().decode_steps, 3);
        assert_eq!(driver.stats().staged_tokens, 3);
        assert!(driver.output.early_empty());
        for token in 1..=3 {
            let snapshot = driver
                .load_registry()
                .ledger()
                .output(OutputTokenId::new(token))
                .unwrap();
            assert!(snapshot.cohort_phases.contains_key(&CohortId::new(token)));
            assert_eq!(snapshot.externally_committed_tokens, 1);
        }
        assert!(
            driver
                .load_registry()
                .ledger()
                .output(OutputTokenId::new(4))
                .is_none()
        );
        let mut next = request(2, &[10], 1, vec![]);
        next.session_id = Some(SessionId(1));
        driver.submit(next);
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        assert_eq!(driver.retained_session_position(SessionId(1)), Some(6));
    }

    #[test]
    fn early_chunked_prefill_emits_only_after_final_prompt_commit() {
        let mut driver = early_driver(vec![top(65), top(66)]);
        driver.submit(request(1, &[7, 8, 9], 1, vec![]));
        let mut events = vec![];
        driver
            .step(&mut |e| {
                events.push(e.clone());
                Ok(())
            })
            .unwrap();
        assert!(events.is_empty());
        driver
            .step(&mut |e| {
                events.push(e.clone());
                Ok(())
            })
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(driver.stats().prefill_tokens, 3);
        assert_eq!(driver.stats().decode_steps, 0);
        driver
            .drive_ready_test_work(|e| {
                events.push(e.clone());
                Ok(())
            })
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(driver.drain_finished()[0].position, 4);
    }

    #[test]
    fn early_suspended_decode_cancellation_never_replays_delivered_token() {
        let runner = MockTopKRunner::new(vec![top(65)]).with_resumable_wait_scripts([0, 1]);
        let mut driver = driver_from_runner(runner)
            .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        driver.config.enable_native_proposals = false;
        driver.retain_session(SessionId(1)).unwrap();
        let mut req = request(1, &[9], 1, vec![]);
        req.session_id = Some(SessionId(1));
        driver.submit(req);
        let mut events = vec![];
        driver
            .step(&mut |e| {
                events.push(e.clone());
                Ok(())
            })
            .unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            driver
                .step(&mut |e| {
                    events.push(e.clone());
                    Ok(())
                })
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        driver.cancel_request(RequestId(1)).unwrap();
        driver
            .drive_ready_test_work(|e| {
                events.push(e.clone());
                Ok(())
            })
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(driver.stats().emitted_tokens, 1);
        assert_eq!(driver.stats().decode_steps, 0);
        assert_eq!(driver.drain_cancelled().len(), 1);
        assert_eq!(driver.retained_session_position(SessionId(1)), Some(0));
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        assert!(driver.output.early_empty());
    }

    #[test]
    fn early_stream_stop_eos_context_and_zero_limit_keep_committed_frontiers() {
        for (limit, stop, eos, ignore_eos, context, expected, reason) in [
            (
                8,
                vec!["AB".into()],
                None,
                false,
                16,
                vec![65, 66],
                Some(SequenceFinishReason::StopString),
            ),
            (
                8,
                vec![],
                Some(66),
                false,
                16,
                vec![65],
                Some(SequenceFinishReason::Eos),
            ),
            (
                8,
                vec![],
                Some(65),
                false,
                16,
                vec![],
                Some(SequenceFinishReason::Eos),
            ),
            (
                2,
                vec![],
                Some(66),
                true,
                16,
                vec![65, 66],
                Some(SequenceFinishReason::MaxTokens),
            ),
            (
                0,
                vec![],
                None,
                false,
                16,
                vec![],
                Some(SequenceFinishReason::MaxTokens),
            ),
            (8, vec![], None, false, 2, vec![], None),
        ] {
            let mut driver = early_driver(vec![top(65), top(66), top(67)]);
            driver.executor_mut().runner_mut().eos = eos;
            driver.config.ctx_size = context;
            driver.retain_session(SessionId(1)).unwrap();
            let mut req = request(1, &[9], limit, stop);
            req.session_id = Some(SessionId(1));
            req.ignore_eos = ignore_eos;
            if reason.is_none() {
                driver.update_hard_resource_observability();
                let admission = driver.admission_snapshot();
                let stats = driver.stats().clone();
                let ownership = (
                    driver.scheduler.total_submitted(),
                    driver.slot_pool.active_count(),
                    driver.sessions.sequence_state_count(),
                    driver.sessions.page_slot_count(),
                    driver.sessions.owner_count(),
                    driver.sessions.cleanup_count(),
                    driver.kv.grant_count(),
                    driver.request_identities.len(),
                    driver.next_admission_session,
                    driver.retained_session_position(SessionId(1)),
                );
                assert!(matches!(
                    driver.try_submit(req.clone()),
                    Err(Error::Admission {
                        source: RuntimeAdmissionError::InvalidPosition {
                            position: 0,
                            prompt_tokens: 1,
                            context: 0,
                        }
                    })
                ));
                assert_eq!(driver.admission_snapshot(), admission);
                assert_eq!(driver.stats(), &stats);
                assert_eq!(
                    (
                        driver.scheduler.total_submitted(),
                        driver.slot_pool.active_count(),
                        driver.sessions.sequence_state_count(),
                        driver.sessions.page_slot_count(),
                        driver.sessions.owner_count(),
                        driver.sessions.cleanup_count(),
                        driver.kv.grant_count(),
                        driver.request_identities.len(),
                        driver.next_admission_session,
                        driver.retained_session_position(SessionId(1)),
                    ),
                    ownership
                );
                assert!(driver.take_request_terminal(RequestId(1)).is_none());
                assert!(driver.drain_finished().is_empty());
                assert!(driver.drain_cancelled().is_empty());
                assert!(driver.drain_failed().is_empty());
                assert!(driver.output.early_empty());
                assert_eq!(
                    driver
                        .step(&mut |_| panic!("rejected request emitted output"))
                        .unwrap(),
                    ResidentDriverStep::Idle
                );
                assert_eq!(driver.admission_snapshot(), admission);
                assert_eq!(driver.stats(), &stats);
                req.max_new_tokens = context - req.prompt_tokens.len();
                driver.try_submit(req).unwrap();
                assert_eq!(
                    driver.cancel_request(RequestId(1)).unwrap(),
                    ResidentCancelProgress::Complete(CancelRequestResult::Waiting {
                        request_id: RequestId(1),
                        session_id: SessionId(1),
                    })
                );
                assert_eq!(driver.drain_cancelled().len(), 1);
                assert_eq!(driver.admission_snapshot(), admission);
                assert_eq!(driver.retained_session_position(SessionId(1)), ownership.9);
                continue;
            }
            driver.try_submit(req).unwrap();
            let mut events = vec![];
            driver
                .drive_ready_test_work(|e| {
                    events.push(e.token);
                    Ok(())
                })
                .unwrap();
            assert_eq!(events, expected);
            let finished = driver.drain_finished();
            assert_eq!(finished[0].finish_reason, reason);
            assert_eq!(finished[0].generated, expected.len());
            assert_eq!(finished[0].position, 1 + expected.len());
            assert_eq!(
                driver.retained_session_position(SessionId(1)),
                Some(1 + expected.len())
            );
            assert_eq!(driver.stats().decode_steps, expected.len());
            assert_eq!(driver.stats().emitted_tokens, expected.len());
            assert!(driver.output.early_empty());
        }
    }

    #[test]
    fn early_callback_retry_is_exactly_once_without_replaying_prefill_or_final_token() {
        let mut driver = early_driver(vec![top(65), top(66)]);
        driver.submit(request(1, &[9], 1, vec![]));
        assert!(
            driver
                .step(&mut |_| Err(Error::Invariant {
                    message: "subscriber rejected event".into()
                }))
                .is_err()
        );
        assert_eq!(driver.executor().runner().committed_batches, 1);
        assert_eq!(driver.stats().emitted_tokens, 0);
        assert_eq!(driver.output.queued_count(), 1);
        let mut events = vec![];
        driver
            .drive_ready_test_work(|e| {
                events.push(e.clone());
                Ok(())
            })
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!((events[0].index, events[0].token), (0, 65));
        assert_eq!(driver.executor().runner().committed_batches, 2);
        assert_eq!(driver.stats().emitted_tokens, 1);
        assert_eq!(driver.drain_finished()[0].tokens, vec![9, 65]);
        assert!(driver.output.outbox_empty());
        assert!(driver.output.early_empty());
    }

    #[test]
    fn early_cancel_clears_pending_delivery_and_invalidates_retained_session() {
        for accepted in [false, true] {
            let mut driver = early_driver(vec![top(65), top(66)]);
            driver.retain_session(SessionId(1)).unwrap();
            let mut req = request(1, &[9], 3, vec![]);
            req.session_id = Some(SessionId(1));
            driver.submit(req);
            let mut events = vec![];
            let result = driver.step(&mut |e| {
                if accepted {
                    events.push(e.token);
                    Ok(())
                } else {
                    Err(Error::Invariant {
                        message: "not accepted".into(),
                    })
                }
            });
            assert_eq!(result.is_ok(), accepted);
            driver.cancel_request(RequestId(1)).unwrap();
            assert_eq!(driver.retained_session_position(SessionId(1)), Some(0));
            assert_eq!(driver.drain_cancelled().len(), 1);
            assert!(driver.output.outbox_empty());
            assert!(driver.output.early_empty());
            assert_eq!(driver.executor().runner().committed_batches, 1);
            driver
                .drive_ready_test_work(|e| {
                    events.push(e.token);
                    Ok(())
                })
                .unwrap();
            assert_eq!(events.len(), usize::from(accepted));
            driver.reset_session(SessionId(1)).unwrap();
            let mut next = request(2, &[10], 1, vec![]);
            next.session_id = Some(SessionId(1));
            driver.submit(next);
            driver
                .drive_ready_test_work(|e| {
                    assert_eq!(e.index, 0);
                    Ok(())
                })
                .unwrap();
            assert_eq!(driver.retained_session_position(SessionId(1)), Some(2));
        }
    }

    #[test]
    fn early_append_failure_does_not_reemit_or_retain_incomplete_session() {
        let mut driver = early_driver(vec![top(65), top(66)]);
        driver.retain_session(SessionId(1)).unwrap();
        let mut req = request(1, &[9], 3, vec![]);
        req.session_id = Some(SessionId(1));
        driver.submit(req);
        let mut events = vec![];
        driver
            .step(&mut |e| {
                events.push(e.token);
                Ok(())
            })
            .unwrap();
        driver
            .sessions
            .sequence_state_mut(&SessionId(1))
            .unwrap()
            .fail_next_mutation = true;
        assert!(
            driver
                .step(&mut |e| {
                    events.push(e.token);
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(events, vec![65]);
        assert_eq!(driver.drain_failed().len(), 1);
        assert_eq!(driver.retained_session_position(SessionId(1)), None);
        assert!(driver.output.early_empty());
        assert!(driver.output.outbox_empty());
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
            .prefix
            .queue_cleanup(PendingPrefixCleanup::model_only(state));

        assert!(!driver.has_pending_async_work());
        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("simulated sequence-state release failure")
        );
        assert_eq!(driver.prefix.pending_cleanup_count(), 1);
        assert_eq!(
            driver
                .prefix
                .front_cleanup()
                .and_then(|cleanup| cleanup.model_state())
                .map(|state| state.position),
            Some(37)
        );
        assert_eq!(driver.executor().runner().released_sequence_states, 0);
        assert!(!driver.has_pending_async_work());

        assert_eq!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Idle
        );
        assert!(driver.prefix.pending_cleanups_empty());
        assert_eq!(driver.executor().runner().released_sequence_states, 1);
    }

    #[test]
    fn shutdown_retries_prefix_cleanup_before_runner_extraction() {
        let mut driver = prefix_cache_driver(Vec::new(), 2, 2);
        driver.submit(request(64, &[1, 2, 3], 0, Vec::new()));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        driver.drain_finished();
        assert!(!driver.prefix.is_empty());
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
        assert!(driver.prefix.is_empty());
        assert_eq!(driver.prefix.pending_cleanup_count(), 1);
        assert_eq!(
            driver
                .prefix
                .front_cleanup()
                .and_then(|cleanup| cleanup.model_state())
                .map(|state| state.position),
            Some(3)
        );
        assert!(!driver.has_pending_async_work());

        let report = driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert!(report.registry.drained);
        assert!(driver.prefix.pending_cleanups_empty());
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
        assert!(driver.prefix.contains_exact(namespace, &[1, 2, 3, 4]));
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
        assert!(!driver.prefix.contains_exact(namespace, &[1, 2, 3, 4]));
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
        assert!(driver.prefix.contains_exact(namespace, &[1, 2, 3]));
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
        assert!(driver.prefix.contains_exact(namespace, &[1, 2, 3]));
        assert_eq!(driver.prefix.used_capacity(), 1);
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
        let model = driver.sessions.sequence_state(&SessionId(2)).unwrap();
        assert_eq!(model.position, 5);
        assert_eq!(model.prefills, vec![vec![4, 5]]);
        assert_eq!(driver.prefix_hits(), 1);
        assert_eq!(driver.prefix_misses(), 1);
        assert!(driver.prefix.contains_exact(namespace, &[1, 2, 3, 4, 5]));
        assert_eq!(driver.page_manager().unwrap().active_sequences(), 1);

        let report = driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert!(report.registry.drained);
        assert!(driver.prefix.is_empty());
        assert!(driver.prefix.pending_cleanups_empty());
        assert!(driver.prefix.cached_sessions_empty());
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
        assert!(driver.prefix.is_empty());
        assert!(!driver.prefix.has_cached_session(&SessionId(70)));
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
        assert!(driver.prefix.contains_exact(namespace, &[1, 2, 3, 4]));
        assert_eq!(driver.available_kv_page_credits(), 1);

        driver.submit(request(63, &[9, 10, 11, 12, 13], 0, Vec::new()));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        let finished = driver.drain_finished();
        assert_eq!(finished[0].tokens, vec![9, 10, 11, 12, 13]);
        assert!(!driver.prefix.contains_exact(namespace, &[1, 2, 3, 4]));
        assert!(
            driver
                .prefix
                .contains_exact(namespace, &[9, 10, 11, 12, 13])
        );
        assert_eq!(driver.prefix.used_capacity(), 2);
        assert_eq!(driver.prefix_hits(), 0);
        assert_eq!(driver.prefix_misses(), 2);
        assert_eq!(driver.executor().runner().released_sequence_states, 3);
        assert!(!driver.executor().runner().released_kv_pages.is_empty());

        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        assert_eq!(driver.executor().runner().released_sequence_states, 4);
    }

    fn assert_resident_batch_eviction_rollback(fail_rollback: bool) {
        let mut runner = MockTopKRunner::new(Vec::new());
        runner.paged = true;
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 2,
                max_decode_batch: 2,
                max_batch_tokens: 16,
                allow_mixed_batches: true,
                prefix_cache_capacity_pages: 1,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 2));
        let namespace = driver.prefix_cache_namespace().unwrap();
        driver.submit(request(60, &[1, 2, 3, 4], 0, Vec::new()));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        assert_eq!(driver.drain_finished().len(), 1);
        assert_eq!(driver.available_kv_page_credits(), 1);

        // The first request shares the cached page, so eviction reaches model
        // release without freeing that page or consuming the KV-release fault.
        driver.submit(request(61, &[1, 2, 3, 4, 5], 0, Vec::new()));
        driver.submit(request(62, &[9, 10], 0, Vec::new()));
        driver.admit_new_sequences().unwrap();
        assert_eq!(driver.prefix_hits(), 1);
        let mut action = driver
            .scheduler
            .next_admitted_action_policy(true)
            .unwrap()
            .unwrap();
        let batch = ScheduledBatch::from_action(&mut action, driver.top_k)
            .unwrap()
            .unwrap();
        assert_eq!(
            batch
                .sequences
                .iter()
                .map(|sequence| sequence.request_id)
                .collect::<Vec<_>>(),
            vec![Some(RequestId(61)), Some(RequestId(62))]
        );
        let first_slot = *driver
            .sessions
            .page_slot(&batch.sequences[0].session_id)
            .unwrap();
        let second_slot = *driver
            .sessions
            .page_slot(&batch.sequences[1].session_id)
            .unwrap();
        let shared_page = driver
            .page_manager()
            .unwrap()
            .block_table(first_slot)
            .unwrap()
            .pages()[0];
        let prepared = driver.executor.runner().prepared_batches;
        let intents = driver.executor.runner().end_intents.clone();
        let released_pages = driver.executor.runner().released_kv_pages.len();
        driver.executor.runner_mut().release_failures_remaining = 1;
        driver.executor.runner_mut().kv_release_failures_remaining = usize::from(fail_rollback);
        let error = driver
            .reserve_batch_pages(
                ExecutionTransactionId::new(993).unwrap(),
                crate::scheduling::ResourceDemand::required(ExecutionPhase::Prefill),
                &batch,
                &[0, 0],
            )
            .unwrap_err();
        if fail_rollback {
            let Error::Cleanup {
                source: primary,
                cleanup,
                ..
            } = &error
            else {
                panic!("both eviction and reservation rollback errors must survive: {error:?}");
            };
            assert!(
                primary
                    .to_string()
                    .contains("sequence-state release failure")
            );
            assert!(cleanup.to_string().contains("KV page release failure"));
        } else {
            assert!(error.to_string().contains("sequence-state release failure"));
        }
        assert!(!driver.prefix.contains_exact(namespace, &[1, 2, 3, 4]));
        assert_eq!(driver.prefix.pending_cleanup_count(), 1);
        assert_eq!(
            driver.kv.pending_retirement_count(),
            usize::from(fail_rollback)
        );
        assert!(driver.kv.pending_abort_empty());
        assert_eq!(driver.kv.grant_count(), 1 + usize::from(fail_rollback));
        assert!(driver.kv.has_grant(&shared_page));
        assert_eq!(
            driver
                .load_registry
                .resources()
                .in_use(ResourceKind::KvPage),
            1 + u64::from(fail_rollback)
        );
        assert_eq!(
            driver.available_kv_page_credits(),
            usize::from(!fail_rollback)
        );
        assert_eq!(
            driver.page_manager().unwrap().stats().retiring_pages,
            usize::from(fail_rollback)
        );
        assert_eq!(
            driver.page_manager().unwrap().allocated_pages(),
            1 + usize::from(fail_rollback)
        );
        assert_eq!(
            driver
                .page_manager()
                .unwrap()
                .block_table(first_slot)
                .unwrap()
                .pages(),
            &[shared_page]
        );
        assert!(
            driver
                .page_manager()
                .unwrap()
                .block_table(second_slot)
                .unwrap()
                .pages()
                .is_empty()
        );
        // The first token reached abort, not Drop: its slot can reserve again.
        assert_eq!(
            driver
                .page_manager()
                .unwrap()
                .required_physical_pages(first_slot, 0, 1)
                .unwrap(),
            1
        );
        assert_eq!(
            driver.executor.runner().released_kv_pages.len(),
            released_pages + usize::from(!fail_rollback)
        );
        assert_eq!(driver.executor.runner().prepared_batches, prepared);
        assert_eq!(driver.executor.runner().end_intents, intents);

        driver.progress_pending_cleanups().unwrap();
        assert!(driver.prefix.pending_cleanups_empty());
        assert!(driver.kv.pending_retirements_empty());
        assert!(driver.kv.pending_abort_empty());
        assert_eq!(driver.kv.grant_count(), 1);
        assert_eq!(driver.available_kv_page_credits(), 1);
        assert_eq!(
            driver.executor.runner().released_kv_pages.len(),
            released_pages + 1
        );
        for request_id in [RequestId(61), RequestId(62)] {
            assert!(matches!(
                driver.cancel_request(request_id).unwrap(),
                ResidentCancelProgress::Complete(_)
            ));
        }
        assert_eq!(driver.drain_cancelled().len(), 2);
        driver
            .shutdown(&mut |_| panic!("failed admission emitted output"), 32)
            .unwrap();
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        assert!(driver.kv.grants_empty());
        assert_eq!(
            driver
                .load_registry
                .resources()
                .in_use(ResourceKind::KvPage),
            0
        );
        assert_eq!(driver.slot_pool.active_count(), 0);
        assert_eq!(driver.executor.runner().prepared_batches, prepared);
        assert_eq!(driver.executor.runner().end_intents, intents);
    }

    #[test]
    fn pr18_resident_batch_eviction_failure_aborts_prior_reservation() {
        assert_resident_batch_eviction_rollback(false);
    }

    #[test]
    fn pr18_resident_batch_eviction_and_rollback_failure_retain_credit_until_retry() {
        assert_resident_batch_eviction_rollback(true);
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
        assert!(driver.sessions.owner_count() == 0);
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
        assert_eq!(driver.sessions.sequence_state_count(), 2);
        assert_eq!(
            driver
                .sessions
                .sequence_state(&SessionId(1))
                .unwrap()
                .position,
            1
        );
        assert_eq!(
            driver
                .sessions
                .sequence_state(&SessionId(2))
                .unwrap()
                .position,
            1
        );
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
        assert!(driver.sessions.sequence_state_count() == 0);
        assert_eq!(driver.executor().runner().released_sequence_states, 2);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn lowering_failure_moves_dequeued_sequence_to_failed_state() {
        let mut runner = MockTopKRunner::new(Vec::new());
        runner.position = u32::MAX as usize + 1;
        let mut driver = driver_from_runner(runner);
        // Admission now rejects invalid positions without allocating. Bypass
        // it here to retain the independent lowering cleanup regression.
        driver
            .scheduler
            .submit_at_position(request(13, &[1], 1, Vec::new()), u32::MAX as usize + 1);

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
        assert_eq!(driver.output.queued_count(), 1);

        let mut delivered = Vec::new();
        driver
            .flush_committed_token_outbox(&mut |event| {
                delivered.push(event.clone());
                Ok(())
            })
            .unwrap();
        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].token, b'a' as u32);
        assert!(driver.output.outbox_empty());
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
        let rejection = driver
            .try_submit(request(13, &[1, 2], 1, Vec::new()))
            .unwrap_err();
        assert!(rejection.to_string().contains("allows 5 active sequences"));

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(format!("{error}").contains("allows 5 active sequences"));
        assert_eq!(driver.scheduler().waiting_len(), 0);
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
        assert!(driver.sessions.contains_sequence_state(&SessionId(1)));

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
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));
        assert!(!driver.sessions.has_page_slot(&SessionId(1)));
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
        assert!(driver.sessions.suspended_count() == 0);
        assert!(matches!(
            driver.sessions.cleanup(&session_id),
            Some(PendingSequenceCleanup::Suspended { .. })
        ));
        assert_eq!(driver.scheduler.failed_slot_ownership(), 1);
        assert_eq!(driver.slot_pool.active_count(), 1);
        assert_eq!(driver.executor.runner().released_sequence_states, 0);

        let report = driver.shutdown(&mut |_| Ok(()), 16).unwrap();
        assert!(report.registry.drained);
        assert!(driver.sessions.cleanup_count() == 0);
        assert_eq!(driver.scheduler.failed_slot_ownership(), 0);
        assert_eq!(driver.slot_pool.active_count(), 0);
        assert!(driver.kv.grants_empty());
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

    #[test]
    fn pr08_restore_fault_retains_every_owner_before_retry() {
        for stage in ["restore logical", "restore scheduler"] {
            let mut driver = fork_driver();
            driver.submit(request(81, &[1, 2], 4, vec![]));
            driver.step(&mut |_| Ok(())).unwrap();
            driver.preempt_session(SessionId(1)).unwrap();
            let pages = driver.page_manager().unwrap().allocated_pages();
            driver.stage_faults.push_back(stage);
            let error = driver.restore_session(SessionId(1)).unwrap_err();
            eprintln!(
                "PR08 stage={stage} error={error} suspended={} active={} slots={} pages={} grants={} physical_restore={} physical_preempt={}",
                driver.suspended_len(),
                driver.scheduler.active_len(),
                driver.slot_pool.active_count(),
                driver.page_manager().unwrap().allocated_pages(),
                driver.kv.grant_count(),
                driver.executor.runner().restore_calls,
                driver.executor.runner().preempt_calls
            );
            assert_eq!(
                driver.suspended_len(),
                1,
                "source/model/schedule must survive {stage}"
            );
            assert_eq!(driver.scheduler.active_len(), 0);
            assert_eq!(driver.slot_pool.active_count(), 1);
            assert_eq!(driver.page_manager().unwrap().allocated_pages(), pages);
            driver.restore_session(SessionId(1)).unwrap();
            assert_eq!(driver.scheduler.active_len(), 1);
            driver.shutdown(&mut |_| Ok(()), 32).unwrap();
            assert_eq!(driver.slot_pool.active_count(), 0);
            assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        }
    }

    #[test]
    fn pr08_physical_error_is_quarantined_not_replayed() {
        let mut driver = fork_driver();
        driver.submit(request(82, &[1, 2], 4, vec![]));
        driver.step(&mut |_| Ok(())).unwrap();
        driver.executor.runner_mut().preempt_errors = 1;
        assert!(driver.preempt_session(SessionId(1)).is_err());
        assert_eq!(driver.suspended_len(), 1);
        assert!(driver.restore_session(SessionId(1)).is_err());
        assert!(driver.shutdown(&mut |_| Ok(()), 32).is_err());
        assert_eq!(driver.executor.runner().preempt_calls, 1);
        assert_eq!(driver.executor.runner().restore_calls, 0);
        assert_eq!(driver.slot_pool.active_count(), 1);
        assert_eq!(driver.executor.runner().released_sequence_states, 0);
        assert!(driver.try_into_runner().is_err());
    }

    #[test]
    fn pr09_fork_prepare_slot_failure_retains_identity_until_cleanup() {
        let mut driver = ResidentTopKDriver::with_configs(
            MockTopKRunner::new(vec![top(65)]),
            FailOnceSequenceSlotPool::new(2, 1),
            ResidentSchedulerConfig {
                max_active_sequences: 2,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        driver.submit(request(91, &[1, 2], 4, vec![]));
        driver.step(&mut |_| Ok(())).unwrap();
        let mut target = request(92, &[3], 1, vec![]);
        target.session_id = Some(SessionId(2));
        let error = driver
            .fork_session_exact(SessionId(1), target.clone(), 999)
            .unwrap_err();
        eprintln!(
            "PR09 prepare error={error} slots={} pending={} active={} pages={} grants={}",
            driver.slot_pool.active_count(),
            driver.sessions.cleanup_count(),
            driver.scheduler.active_len(),
            driver.page_manager().unwrap().allocated_pages(),
            driver.kv.grant_count()
        );
        assert!(format!("{error:?}").contains("sequence slot release failure"));
        assert!(driver.sessions.has_cleanup(&SessionId(2)));
        assert_eq!(driver.slot_pool.active_count(), 2);
        assert!(driver.fork_session_exact(SessionId(1), target, 2).is_err());
        driver.progress_pending_cleanups().unwrap();
        assert!(!driver.sessions.has_cleanup(&SessionId(2)));
        assert_eq!(driver.slot_pool.active_count(), 1);
        assert_eq!(driver.scheduler.active_len(), 1);
        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert_eq!(driver.slot_pool.active_count(), 0);
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
    }

    #[test]
    fn pr10_proposal_progress_survives_observation_failure_without_replay() {
        for waits in [0, 1] {
            let runner = MockTopKRunner::new(vec![top(10)])
                .with_speculative_cycle(
                    NativeProposal {
                        token_ids: vec![11, 12],
                        confidence_logits: vec![1.0, 1.0],
                    },
                    vec![
                        TokenLogit::new(11, 9.0),
                        TokenLogit::new(12, 8.0),
                        TokenLogit::new(13, 7.0),
                    ],
                )
                .with_proposal_waits(vec![waits]);
            let mut driver = speculative_driver_from_runner(runner, 1);
            let actions = ready_speculative_decode_actions(&mut driver, &[1]);
            driver.stage_faults.push_back("spec observation");
            let error = driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap_err();
            eprintln!(
                "PR10 waits={waits} primary={error} owners={} sessions={} slots={} pages={} grants={} proposals={}",
                driver.speculative_transactions.len(),
                driver.sessions.owner_count(),
                driver.slot_pool.active_count(),
                driver.page_manager().unwrap().allocated_pages(),
                driver.kv.grant_count(),
                driver.executor.runner().native_proposal_begin_calls
            );
            assert_eq!(driver.speculative_transactions.len(), 1);
            assert_eq!(driver.sessions.owner_count(), 1);
            assert_eq!(driver.executor.runner().native_proposal_begin_calls, 1);
            driver.step(&mut |_| Ok(())).unwrap();
            assert_eq!(driver.executor.runner().native_proposal_begin_calls, 1);
            driver.shutdown(&mut |_| Ok(()), 32).unwrap();
            assert_eq!(driver.slot_pool.active_count(), 0);
            assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        }
    }

    #[test]
    fn pr11_speculative_pending_publish_cancel_beats_output_terminals() {
        for reason in [
            SequenceFinishReason::MaxTokens,
            SequenceFinishReason::StopString,
            SequenceFinishReason::Context,
            SequenceFinishReason::Eos,
        ] {
            for accepted_cancel in [false, true] {
                let mut runner = MockTopKRunner::new(vec![top(65)]).with_speculative_cycle(
                    NativeProposal {
                        token_ids: vec![66, 67],
                        confidence_logits: vec![-100.0, -100.0],
                    },
                    vec![TokenLogit::new(99, 8.0)],
                );
                runner.eos = Some(99);
                let mut driver = speculative_driver_from_runner(runner, 1);
                let actions = ready_speculative_decode_actions(&mut driver, &[1]);
                let sequence = driver.scheduler.active_sequence_mut(SessionId(1)).unwrap();
                if reason == SequenceFinishReason::MaxTokens {
                    sequence.max_new_tokens = 1;
                }
                if reason == SequenceFinishReason::StopString {
                    sequence.stop = vec!["A".into()];
                }
                if reason == SequenceFinishReason::Context {
                    driver.config.ctx_size = 2;
                }
                driver.executor.runner_mut().publish_progress = VecDeque::from([
                    Ok(TransactionEndProgress::Pending),
                    Ok(TransactionEndProgress::Complete),
                ]);
                let before_commits = driver.executor.runner().committed_batches;
                assert_eq!(
                    driver
                        .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                        .unwrap(),
                    ResidentDriverStep::Blocked
                );
                assert!(driver.take_request_terminal(RequestId(1)).is_none());
                if accepted_cancel {
                    assert_eq!(
                        driver.cancel_request(RequestId(1)).unwrap(),
                        ResidentCancelProgress::Pending
                    );
                }
                driver.step(&mut |_| Ok(())).unwrap();
                let terminal = driver.take_request_terminal(RequestId(1)).unwrap();
                eprintln!(
                    "PR11 reason={reason:?} accepted={accepted_cancel} terminal={terminal:?} commits={} aborts={} slots={} pages={} grants={}",
                    driver.executor.runner().committed_batches,
                    driver.executor.runner().rolled_back_batches,
                    driver.slot_pool.active_count(),
                    driver.page_manager().unwrap().allocated_pages(),
                    driver.kv.grant_count()
                );
                if accepted_cancel {
                    assert!(matches!(
                        terminal,
                        crate::scheduling::RequestTerminal::Cancelled(_)
                    ));
                } else {
                    assert!(
                        matches!(terminal, crate::scheduling::RequestTerminal::Finished(ref sequence) if sequence.finish_reason == Some(reason))
                    );
                }
                assert!(driver.take_request_terminal(RequestId(1)).is_none());
                assert_eq!(
                    driver.executor.runner().committed_batches,
                    before_commits + 1
                );
                assert_eq!(driver.executor.runner().rolled_back_batches, 0);
                assert_eq!(driver.slot_pool.active_count(), 0);
                assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
                assert!(driver.kv.grants_empty());
                driver.step(&mut |_| Ok(())).unwrap();
                assert_eq!(
                    driver.executor.runner().committed_batches,
                    before_commits + 1
                );
            }
        }
    }

    #[test]
    fn pr08_transition_double_fault_matrix_preserves_phase_and_retry() {
        let cases = [
            (false, "preempt scheduler", None),
            (false, "preempt logical", None),
            (false, "preempt logical", Some("restore scheduler")),
            (false, "preempt physical", Some("restore logical")),
            (false, "preempt physical", Some("restore scheduler")),
            (true, "restore physical", None),
            (true, "restore logical", Some("rollback physical")),
            (true, "restore scheduler", Some("rollback logical")),
            (true, "restore scheduler", Some("rollback physical")),
        ];
        for (restoring, primary, cleanup) in cases {
            let mut driver = fork_driver();
            driver.submit(request(81, &[1, 2], 4, vec![]));
            driver.step(&mut |_| Ok(())).unwrap();
            let original_model_position = driver
                .sessions
                .sequence_state(&SessionId(1))
                .unwrap()
                .position;
            let original_pages = driver
                .page_manager()
                .unwrap()
                .block_table(StateSlot::new(0))
                .unwrap()
                .pages()
                .to_vec();
            if restoring {
                driver.preempt_session(SessionId(1)).unwrap();
            }
            driver.stage_faults.push_back(primary);
            driver.stage_faults.extend(cleanup);
            let error = if restoring {
                driver.restore_session(SessionId(1))
            } else {
                driver.preempt_session(SessionId(1))
            }
            .unwrap_err();
            assert!(format!("{error:?}").contains(primary));
            if let Some(cleanup) = cleanup {
                assert!(format!("{error:?}").contains(cleanup));
            }
            let calls = (
                driver.executor.runner().preempt_calls,
                driver.executor.runner().restore_calls,
            );
            eprintln!(
                "PR08 primary={primary} cleanup={cleanup:?} error={error:?} phase={:?} slots={} pages={} grants={} physical={calls:?}",
                driver
                    .sessions
                    .suspended(&SessionId(1))
                    .map(|p| (p.phase, p.rolling_back)),
                driver.slot_pool.active_count(),
                driver.page_manager().unwrap().allocated_pages(),
                driver.kv.grant_count()
            );
            assert_eq!(driver.slot_pool.active_count(), 1);
            assert_eq!(
                driver.page_manager().unwrap().allocated_pages(),
                original_pages.len()
            );
            if driver.suspended_len() != 0 {
                if let Some(pending) = driver.sessions.suspended(&SessionId(1)) {
                    assert_eq!(pending.model_state.position, original_model_position);
                    assert_eq!(pending.schedule.session_id(), SessionId(1));
                }
                driver.restore_session(SessionId(1)).unwrap();
            }
            let expected = if restoring {
                if primary == "restore physical" {
                    (1, 1)
                } else {
                    (2, 2)
                }
            } else {
                (0, 0)
            };
            assert_eq!(
                (
                    driver.executor.runner().preempt_calls,
                    driver.executor.runner().restore_calls
                ),
                expected,
                "completed physical stage must not replay: {primary}/{cleanup:?}"
            );
            assert_eq!(driver.scheduler.active_len(), 1);
            assert_eq!(
                driver
                    .page_manager()
                    .unwrap()
                    .block_table(StateSlot::new(0))
                    .unwrap()
                    .pages(),
                original_pages
            );
            driver.shutdown(&mut |_| Ok(()), 32).unwrap();
            assert_eq!(driver.slot_pool.active_count(), 0);
            assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
            assert!(driver.kv.grants_empty());
        }
    }

    #[test]
    fn pr09_fork_publish_model_and_slot_undo_are_independent() {
        for (model_failures, slot_failures) in [(1, 1), (1, 0), (0, 1)] {
            let mut driver = ResidentTopKDriver::with_configs(
                MockTopKRunner::new(vec![top(65)]),
                FailOnceSequenceSlotPool::new(2, slot_failures),
                ResidentSchedulerConfig {
                    max_active_sequences: 2,
                    ..Default::default()
                },
                NonZeroU32::new(1).unwrap(),
                ResidentTopKDriverConfig::default(),
            )
            .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
            driver.submit(request(91, &[1, 2], 4, vec![]));
            driver.step(&mut |_| Ok(())).unwrap();
            let source_pages = driver
                .page_manager()
                .unwrap()
                .block_table(StateSlot::new(0))
                .unwrap()
                .pages()
                .to_vec();
            let source_candidate = driver
                .scheduler
                .active_sequence(SessionId(1))
                .unwrap()
                .next_decode_token;
            driver.executor.runner_mut().release_failures_remaining = model_failures;
            driver.stage_faults.push_back("fork publish");
            let mut target = request(92, &[3], 1, vec![]);
            target.session_id = Some(SessionId(2));
            let error = driver
                .fork_session_exact(SessionId(1), target.clone(), 2)
                .unwrap_err();
            eprintln!(
                "PR09 publish model_failures={model_failures} slot_failures={slot_failures} primary+cleanup={error:?} slots={} pages={} grants={} released_models={}",
                driver.slot_pool.active_count(),
                driver.page_manager().unwrap().allocated_pages(),
                driver.kv.grant_count(),
                driver.executor.runner().released_sequence_states
            );
            assert!(format!("{error:?}").contains("fork publish"));
            assert!(driver.sessions.has_cleanup(&SessionId(2)));
            assert_eq!(driver.slot_pool.active_count(), 1 + slot_failures);
            assert_eq!(
                driver.executor.runner().released_sequence_states,
                1 - model_failures
            );
            assert_eq!(driver.scheduler.active_len(), 1);
            assert_eq!(
                driver
                    .scheduler
                    .active_sequence(SessionId(1))
                    .unwrap()
                    .next_decode_token,
                source_candidate
            );
            assert!(
                source_pages
                    .iter()
                    .all(|page| driver.page_manager().unwrap().page_refcount(*page) == 1)
            );
            assert!(driver.fork_session_exact(SessionId(1), target, 2).is_err());
            driver.progress_pending_cleanups().unwrap();
            driver.progress_pending_cleanups().unwrap();
            assert_eq!(driver.executor.runner().released_sequence_states, 1);
            assert_eq!(driver.slot_pool.active_count(), 1);
            assert!(driver.sessions.cleanup_count() == 0);
            driver.shutdown(&mut |_| Ok(()), 32).unwrap();
            assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        }
    }

    #[test]
    fn pr09_prefix_prepare_unpin_and_model_failure_retain_pin_identity() {
        let mut driver = prefix_cache_driver(vec![top(65)], 8, 16);
        driver.submit(request(91, &[1, 2], 0, vec![]));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        driver.drain_finished();
        assert_eq!(driver.prefix.len(), 1);
        let released = driver.executor.runner().released_sequence_states;
        driver.executor.runner_mut().release_failures_remaining = 1;
        driver
            .stage_faults
            .extend(["prefix prepare", "prefix unpin"]);
        let mut target = request(92, &[1, 2, 3], 1, vec![]);
        target.session_id = Some(SessionId(92));
        driver.submit(target);
        let error = driver.prepare_step().unwrap_err();
        eprintln!(
            "PR09 prefix primary+cleanup={error:?} slots={} pages={} grants={} pending={} released_models={}",
            driver.slot_pool.active_count(),
            driver.page_manager().unwrap().allocated_pages(),
            driver.kv.grant_count(),
            driver.sessions.cleanup_count(),
            driver.executor.runner().released_sequence_states
        );
        let Some(PendingSequenceCleanup::Admission {
            lease,
            model_state: Some(_),
            ..
        }) = driver.sessions.cleanup(&SessionId(92))
        else {
            panic!("model and pin must remain owned");
        };
        assert_eq!(driver.prefix.admission_pin_count(lease), Some(1));
        assert_eq!(driver.executor.runner().released_sequence_states, released);
        assert_eq!(driver.slot_pool.active_count(), 0);
        assert_eq!(driver.scheduler.active_len(), 0);
        // Even an explicit admission retry cannot publish the retained identity.
        driver.admit_new_sequences().unwrap();
        assert_eq!(driver.scheduler.active_len(), 0);
        driver.progress_pending_cleanups().unwrap();
        assert_eq!(
            driver.executor.runner().released_sequence_states,
            released + 1
        );
        assert!(driver.sessions.cleanup_count() == 0);
        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert!(driver.prefix.is_empty());
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
    }

    #[test]
    fn pr10_finish_resume_fault_keeps_lease_and_progress_without_reproposal() {
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
        assert!(matches!(
            driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        let before_resume = driver.executor.runner().native_proposal_resume_calls;
        driver.stage_faults.push_back("spec finish resume");
        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        eprintln!(
            "PR10 finish_resume error={error:?} owners={} continuations={} resume_calls={} resources={{continuation={},lease={},arena={}}}",
            driver.speculative_transactions.len(),
            driver.continuations.len(),
            driver.executor.runner().native_proposal_resume_calls,
            driver
                .load_registry
                .resources()
                .in_use(ResourceKind::Continuation),
            driver
                .load_registry
                .resources()
                .in_use(ResourceKind::ResidencyLease),
            driver.load_registry.resources().in_use(ResourceKind::Arena)
        );
        assert!(format!("{error:?}").contains("spec finish resume"));
        assert_eq!(driver.speculative_transactions.len(), 1);
        assert_eq!(
            driver.executor.runner().native_proposal_resume_calls,
            before_resume + 1
        );
        assert_eq!(
            driver
                .load_registry
                .resources()
                .in_use(ResourceKind::Continuation),
            1
        );
        driver.step(&mut |_| Ok(())).unwrap();
        assert_eq!(
            driver.executor.runner().native_proposal_resume_calls,
            before_resume + 1
        );
        assert!(driver.speculative_transactions.is_empty());
        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert_eq!(driver.slot_pool.active_count(), 0);
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
    }

    #[test]
    fn pr10_verification_observation_matrix_keeps_progress_before_error() {
        for (waits, execution_error) in [(0, false), (1, false), (0, true)] {
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
                .with_resumable_wait_scripts([0, waits]);
            let mut driver = speculative_driver_from_runner(runner, 1);
            let actions = ready_speculative_decode_actions(&mut driver, &[1]);
            driver.executor.runner_mut().empty_top_k_output = execution_error;
            driver.stage_faults.push_back("verification observation");
            let error = driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap_err();
            let Some(PendingSpeculativeDriverCohort::Bookkeeping(pending)) =
                driver.speculative_transactions.values().next()
            else {
                panic!("verification progress must remain owned");
            };
            let kind = match &pending.next {
                PendingSpeculativeDriverCohort::Ending(p)
                    if matches!(p.ending, SpeculativeEnding::Publish(_)) =>
                {
                    "Ready"
                }
                PendingSpeculativeDriverCohort::Verifying(_) => "Waiting",
                PendingSpeculativeDriverCohort::Ending(p)
                    if matches!(
                        p.ending,
                        SpeculativeEnding::Abort {
                            failure: Some(_),
                            ..
                        }
                    ) =>
                {
                    "ActiveFailure"
                }
                _ => panic!("unexpected verification handoff"),
            };
            eprintln!(
                "PR10 verification kind={kind} observation={error:?} owners={} sessions={} slots={} pages={} grants={} executions={}",
                driver.speculative_transactions.len(),
                driver.sessions.owner_count(),
                driver.slot_pool.active_count(),
                driver.page_manager().unwrap().allocated_pages(),
                driver.kv.grant_count(),
                driver.executor.runner().packed_verification_calls
            );
            let retry = driver.step(&mut |_| Ok(()));
            if execution_error {
                assert!(retry.unwrap_err().to_string().contains("empty top-k"));
            } else {
                retry.unwrap();
            }
            assert_eq!(driver.executor.runner().packed_verification_calls, 1);
            assert_eq!(driver.executor.runner().native_proposal_begin_calls, 1);
            driver.shutdown(&mut |_| Ok(()), 32).unwrap();
            assert_eq!(driver.slot_pool.active_count(), 0);
            assert!(driver.kv.grants_empty());
            assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        }
    }

    #[test]
    fn pr10_resume_handoff_retries_original_grant_and_never_reexecutes() {
        for proposal in [false, true] {
            for (waits, execution_error) in [(1, false), (2, false), (1, true)] {
                let mut runner = MockTopKRunner::new(vec![top(10)]).with_speculative_cycle(
                    NativeProposal {
                        token_ids: vec![11, 12],
                        confidence_logits: vec![1.0, 1.0],
                    },
                    vec![
                        TokenLogit::new(11, 9.0),
                        TokenLogit::new(99, 8.0),
                        TokenLogit::new(98, 7.0),
                    ],
                );
                if proposal {
                    runner = runner.with_proposal_waits(vec![waits]);
                } else {
                    runner = runner.with_resumable_wait_scripts([0, waits]);
                }
                let mut driver = speculative_driver_from_runner(runner, 1);
                let actions = ready_speculative_decode_actions(&mut driver, &[1]);
                assert!(matches!(
                    driver
                        .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                        .unwrap(),
                    ResidentDriverStep::WaitingForModelProgress(_)
                ));
                if proposal {
                    driver
                        .executor
                        .runner_mut()
                        .native_proposal_resume_errors_remaining = usize::from(execution_error);
                } else {
                    driver.executor.runner_mut().resume_errors_remaining =
                        usize::from(execution_error);
                }
                driver.stage_faults.extend([
                    "spec finish resume",
                    "spec finish resume",
                    "spec observation",
                ]);
                let first = driver.step(&mut |_| Ok(())).unwrap_err();
                let transaction = *driver.speculative_transactions.keys().next().unwrap();
                let Some(PendingSpeculativeDriverCohort::Bookkeeping(pending)) =
                    driver.speculative_transactions.get(&transaction)
                else {
                    panic!("original resume lease must be retained");
                };
                let continuation = pending.finish.as_ref().unwrap().0;
                let calls = (
                    driver.executor.runner().native_proposal_resume_calls,
                    driver.executor.runner().packed_resume_calls,
                );
                assert_eq!(calls, if proposal { (1, 0) } else { (0, 1) });
                assert_eq!(
                    driver.load_registry.resources().in_use(ResourceKind::Arena),
                    1
                );
                let second = driver.step(&mut |_| Ok(())).unwrap_err();
                assert!(format!("{second:?}").contains("spec finish resume"));
                let Some(PendingSpeculativeDriverCohort::Bookkeeping(pending)) =
                    driver.speculative_transactions.get(&transaction)
                else {
                    unreachable!()
                };
                assert_eq!(pending.finish.as_ref().unwrap().0, continuation);
                assert_eq!(
                    driver.load_registry.resources().in_use(ResourceKind::Arena),
                    1
                );
                let observation = driver.step(&mut |_| Ok(())).unwrap_err();
                assert!(format!("{observation:?}").contains("spec observation"));
                assert_eq!(
                    driver.load_registry.resources().in_use(ResourceKind::Arena),
                    0
                );
                let Some(PendingSpeculativeDriverCohort::Bookkeeping(pending)) =
                    driver.speculative_transactions.get(&transaction)
                else {
                    unreachable!()
                };
                assert!(
                    pending.finish.is_none(),
                    "accepted finish cannot be replayed"
                );
                eprintln!(
                    "PR10 proposal={proposal} waits={waits} execution_error={execution_error} primary={first:?} retry={second:?} observation={observation:?} original_continuation={continuation:?} resume_calls={calls:?} slots={} pages={} grants={} arena={} lease={}",
                    driver.slot_pool.active_count(),
                    driver.page_manager().unwrap().allocated_pages(),
                    driver.kv.grant_count(),
                    driver.load_registry.resources().in_use(ResourceKind::Arena),
                    driver
                        .load_registry
                        .resources()
                        .in_use(ResourceKind::ResidencyLease)
                );
                let retry = driver.step(&mut |_| Ok(()));
                if execution_error {
                    assert!(retry.is_err());
                } else {
                    retry.unwrap();
                }
                assert_eq!(
                    (
                        driver.executor.runner().native_proposal_resume_calls,
                        driver.executor.runner().packed_resume_calls
                    ),
                    calls
                );
                driver.shutdown(&mut |_| Ok(()), 32).unwrap();
                assert!(driver.continuations.is_empty());
                assert_eq!(driver.slot_pool.active_count(), 0);
                assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
                for kind in [
                    ResourceKind::Arena,
                    ResourceKind::ResidencyLease,
                    ResourceKind::Continuation,
                    ResourceKind::Waiter,
                    ResourceKind::KvPage,
                ] {
                    assert_eq!(driver.load_registry.resources().in_use(kind), 0);
                }
            }
        }
    }

    #[test]
    fn pr09_prefix_publish_model_and_scheduler_slot_failure_are_independent() {
        let mut runner = MockTopKRunner::new(vec![top(65)]);
        runner.paged = true;
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FailOnceSequenceSlotPool::new(1, 0),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 1,
                max_decode_batch: 1,
                prefix_cache_capacity_pages: 8,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        driver.submit(request(91, &[1, 2], 0, vec![]));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        driver.drain_finished();
        let released = driver.executor.runner().released_sequence_states;
        driver.executor.runner_mut().release_failures_remaining = 1;
        driver.slot_pool.free_failures_remaining = 2;
        driver.stage_faults.push_back("prefix publish");
        let mut target = request(92, &[1, 2, 3], 1, vec![]);
        target.session_id = Some(SessionId(92));
        driver.submit(target);
        let error = driver.prepare_step().unwrap_err();
        eprintln!(
            "PR09 prefix publish primary+cleanup={error:?} slots={} failed_slots={} active={} pages={} grants={} model_releases={}",
            driver.slot_pool.active_count(),
            driver.scheduler.failed_slot_ownership(),
            driver.scheduler.active_len(),
            driver.page_manager().unwrap().allocated_pages(),
            driver.kv.grant_count(),
            driver.executor.runner().released_sequence_states
        );
        let text = format!("{error:?}");
        for stage in [
            "prefix publish",
            "sequence-state release failure",
            "sequence slot release failure",
        ] {
            assert!(text.contains(stage));
        }
        assert_eq!(driver.scheduler.active_len(), 0);
        assert_eq!(driver.scheduler.failed_slot_ownership(), 1);
        assert_eq!(driver.executor.runner().released_sequence_states, released);
        assert!(driver.progress_pending_cleanups().is_err());
        assert_eq!(
            driver.executor.runner().released_sequence_states,
            released + 1,
            "slot failure must not block independent model release"
        );
        assert!(driver.sessions.has_cleanup(&SessionId(92)));
        assert_eq!(driver.scheduler.failed_slot_ownership(), 1);
        driver.progress_pending_cleanups().unwrap();
        driver.progress_pending_cleanups().unwrap();
        assert_eq!(
            driver.executor.runner().released_sequence_states,
            released + 1
        );
        assert!(driver.sessions.cleanup_count() == 0);
        assert_eq!(driver.slot_pool.active_count(), 0);
        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
    }

    #[test]
    fn pr11_no_candidate_publication_uses_the_same_terminal_arbitration() {
        // Current packed verifier always produces target_next=Some. Exercise
        // the optional-result publication contract directly, not a fake backend
        // claim that an empty TopK is a successful physical verification.
        for accepted in [false, true] {
            let mut driver = speculative_driver_from_runner(MockTopKRunner::new(vec![top(65)]), 1);
            let actions = ready_speculative_decode_actions(&mut driver, &[1]);
            let sequence = driver
                .scheduler
                .active_sequence(SessionId(1))
                .unwrap()
                .clone();
            let transaction = driver.take_transaction_id().unwrap();
            if accepted {
                driver
                    .register_transaction_cancellation(transaction, RequestId(1), SessionId(1))
                    .unwrap();
            }
            let prepared = vec![PreparedSpeculativeAction {
                sequence,
                page_slot: *driver.sessions.page_slot(&SessionId(1)).unwrap(),
                proposal: vec![],
                proposal_time_us: 0,
            }];
            let cohort = crate::speculation::SpeculativeCohortResult {
                results: vec![SpeculativeCycleResult {
                    accepted: vec![],
                    rejected: None,
                    target_correction: None,
                    target_next: None,
                    accounting: crate::speculation::SpeculativeCycleAccounting {
                        verified_rows: 1,
                        externally_committed_tokens: 1,
                        ..Default::default()
                    },
                    transaction_time_us: 0,
                    verify_time_us: 0,
                }],
                transaction_time_us: 0,
                verify_time_us: 0,
            };
            driver
                .publish_speculative_decode_cohort(
                    transaction,
                    Instant::now(),
                    actions,
                    prepared,
                    cohort,
                )
                .unwrap();
            driver
                .finish_transaction_cancellations(transaction, accepted.then_some(RequestId(1)))
                .unwrap();
            let terminal = driver.take_request_terminal(RequestId(1)).unwrap();
            eprintln!(
                "PR11 synthetic NoCandidate accepted={accepted} terminal={terminal:?} slots={} pages={} grants={}",
                driver.slot_pool.active_count(),
                driver.page_manager().unwrap().allocated_pages(),
                driver.kv.grant_count()
            );
            if accepted {
                assert!(matches!(
                    terminal,
                    crate::scheduling::RequestTerminal::Cancelled(_)
                ));
            } else {
                assert!(
                    matches!(terminal, crate::scheduling::RequestTerminal::Finished(ref s) if s.finish_reason == Some(SequenceFinishReason::NoCandidate))
                );
            }
            assert!(driver.take_request_terminal(RequestId(1)).is_none());
            driver.shutdown(&mut |_| Ok(()), 32).unwrap();
            assert_eq!(driver.slot_pool.active_count(), 0);
            assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        }
    }

    #[test]
    fn pr12_duplicate_waiting_rejection_is_failure_atomic() {
        let mut driver = driver_with_outputs(vec![top(65)]);
        let submitted = request(12, &[1], 1, vec![]);
        driver.try_submit(submitted.clone()).unwrap();
        let error = driver
            .try_submit(submitted.clone())
            .expect_err("duplicate waiting request must be rejected");
        eprintln!(
            "PR12 waiting error={error:?} waiting={} submitted={} slots={}",
            driver.scheduler.waiting_len(),
            driver.scheduler.total_submitted(),
            driver.slot_pool.active_count()
        );
        driver.submit(submitted);
        assert_eq!(driver.scheduler.waiting_len(), 1);
        assert_eq!(driver.scheduler.total_submitted(), 1);
        assert!(driver.take_request_terminal(RequestId(12)).is_none());
    }

    #[test]
    fn pr12_unconsumed_terminal_prevents_request_reuse() {
        let mut driver = driver_with_outputs(vec![top(65)]);
        let submitted = request(12, &[1], 1, vec![]);
        driver.try_submit(submitted.clone()).unwrap();
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        assert!(
            driver.try_submit(submitted.clone()).is_err(),
            "unconsumed terminal owns its identity"
        );
        assert_eq!(driver.drain_finished().len(), 1);
        driver.try_submit(submitted).unwrap();
    }

    #[test]
    fn pr12_suspended_identity_and_cancel_are_not_unknown() {
        let mut driver = fork_driver();
        let mut submitted = request(12, &[1, 2], 4, vec![]);
        submitted.session_id = Some(SessionId(12));
        driver.try_submit(submitted.clone()).unwrap();
        driver.step(&mut |_| Ok(())).unwrap();
        driver.preempt_session(SessionId(12)).unwrap();
        let cancel = driver.cancel_request(RequestId(12)).unwrap();
        assert!(
            !matches!(
                cancel,
                ResidentCancelProgress::Complete(CancelRequestResult::NotFound { .. })
            ),
            "suspended cancellation must return a typed disposition"
        );
        assert!(driver.try_submit(submitted).is_err());
        assert_eq!(driver.suspended_len(), 1);
        assert_eq!(driver.slot_pool.active_count(), 1);
    }

    #[test]
    fn pr12_invalid_position_rejects_without_enqueuing() {
        let mut driver = driver_with_outputs(vec![top(65)]);
        assert!(
            driver
                .try_submit_at_position(request(12, &[1], 1, vec![]), usize::MAX)
                .is_err()
        );
        assert_eq!(driver.scheduler.waiting_len(), 0);
        assert_eq!(driver.scheduler.total_submitted(), 0);
        assert!(driver.take_request_terminal(RequestId(12)).is_none());
    }

    #[test]
    fn pr12_waiting_capacity_and_closed_admission_are_failure_atomic() {
        let mut driver = driver_with_outputs(vec![top(65)]);
        driver
            .set_admission_options(RuntimeAdmissionOptions {
                max_waiting_requests: 1,
                max_request_identities: 2,
                max_session_identities: 2,
            })
            .unwrap();
        driver.try_submit(request(20, &[1], 1, vec![])).unwrap();
        let error = driver.try_submit(request(21, &[2], 1, vec![])).unwrap_err();
        assert!(
            matches!(error, Error::Admission { source } if matches!(source, RuntimeAdmissionError::Capacity { resource: RuntimeAdmissionResource::WaitingRequests, .. }))
        );
        assert_eq!(driver.scheduler.waiting_len(), 1);
        assert_eq!(driver.admission_snapshot().request_identities_held, 1);
        driver.close_admission();
        assert!(
            matches!(driver.try_submit(request(22, &[3], 1, vec![])), Err(Error::Admission { source }) if matches!(source, RuntimeAdmissionError::Closed))
        );
        assert_eq!(driver.scheduler.waiting_len(), 1);
        assert_eq!(driver.admission_snapshot().request_identities_held, 1);
    }

    #[test]
    fn pr12_retained_session_allows_only_sequential_turns() {
        let mut driver = driver_with_outputs(vec![top(65), top(66)]);
        let session = SessionId(42);
        driver.retain_session(session).unwrap();
        let mut first = request(42, &[1], 1, vec![]);
        first.session_id = Some(session);
        driver.try_submit(first).unwrap();
        let mut concurrent = request(43, &[2], 1, vec![]);
        concurrent.session_id = Some(session);
        assert!(
            matches!(driver.try_submit(concurrent), Err(Error::Admission { source }) if matches!(source, RuntimeAdmissionError::SessionBusy { .. }))
        );
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        assert!(driver.take_request_terminal(RequestId(42)).is_some());
        let mut second = request(44, &[3], 1, vec![]);
        second.session_id = Some(session);
        driver.try_submit(second).unwrap();
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        assert!(driver.take_request_terminal(RequestId(44)).is_some());
        assert!(driver.retained_session_position(session).is_some());
    }

    #[test]
    fn pr13_receipts_survive_reaping_without_a_permanent_identity_map() {
        let mut driver = fork_driver();
        driver
            .set_admission_options(RuntimeAdmissionOptions {
                max_waiting_requests: 2,
                max_request_identities: 2,
                max_session_identities: 2,
            })
            .unwrap();
        let mut previous: Option<crate::RequestCleanupReceipt> = None;
        for _ in 0..10 {
            let mut turn = request(1, &[1], 1, vec![]);
            turn.session_id = Some(SessionId(1));
            driver.try_submit(turn).unwrap();
            let crate::InferenceRequestCleanup::Tracked(receipt) =
                driver.request_cleanup(RequestId(1))
            else {
                panic!("missing receipt");
            };
            assert!(!receipt.is_released());
            if let Some(old) = previous.take() {
                assert!(old.is_released());
            }
            driver.drive_ready_test_work(|_| Ok(())).unwrap();
            assert!(
                !receipt.is_released(),
                "unconsumed terminal still owns identity"
            );
            assert_eq!(driver.drain_finished().len(), 1);
            assert!(receipt.is_released());
            assert!(driver.request_identities.is_empty());
            assert_eq!(driver.admission_snapshot().request_identities_held, 0);
            for _ in 0..3 {
                assert!(driver.drain_finished().is_empty());
                assert!(receipt.is_released());
            }
            previous = Some(receipt);
        }
    }

    #[test]
    fn pr12_terminal_consumed_but_cleanup_pending_keeps_identity_reserved() {
        let mut driver = fork_driver();
        let session = SessionId(42);
        let mut submitted = request(42, &[1], 4, vec![]);
        submitted.session_id = Some(session);
        driver.try_submit(submitted.clone()).unwrap();
        let crate::InferenceRequestCleanup::Tracked(receipt) =
            driver.request_cleanup(RequestId(42))
        else {
            panic!("missing receipt")
        };
        assert!(!receipt.is_released());
        driver.step(&mut |_| Ok(())).unwrap();
        driver.executor.runner_mut().release_failures_remaining = 2;
        assert!(driver.cancel_request(RequestId(42)).is_err());
        assert!(matches!(
            driver.take_request_terminal(RequestId(42)),
            Some(crate::scheduling::RequestTerminal::Cancelled(_))
        ));
        assert!(driver.sessions.has_cleanup(&session));
        assert!(
            !receipt.is_released(),
            "consumed terminal is not cleanup proof"
        );
        let mut unrelated = request(44, &[2], 4, vec![]);
        unrelated.session_id = Some(SessionId(44));
        driver.try_submit(unrelated).unwrap();
        assert!(driver.try_submit(submitted.clone()).is_err());
        let mut new_request = submitted.clone();
        new_request.id = RequestId(43);
        assert!(matches!(
            driver.try_submit(new_request),
            Err(Error::Admission {
                source: RuntimeAdmissionError::SessionBusy { .. }
            })
        ));
        let snap = driver.admission_snapshot();
        eprintln!(
            "PR12 terminal consumed; cleanup retained: {snap:?} slots={} pages={} grants={}",
            driver.slot_pool.active_count(),
            driver.page_manager().unwrap().allocated_pages(),
            driver.kv.grant_count()
        );
        assert_eq!(snap.request_identities_held, 2);
        assert_eq!(driver.slot_pool.active_count(), 0);
        assert!(driver.cancel_request(RequestId(42)).is_err());
        assert!(driver.try_submit(submitted.clone()).is_err());
        driver.cancel_request(RequestId(42)).unwrap();
        assert!(driver.drain_cancelled().is_empty());
        assert!(receipt.is_released(), "A must not wait for active B");
        assert!(!driver.request_identities.contains_key(&RequestId(42)));
        assert!(driver.take_request_terminal(RequestId(42)).is_none());
        driver.try_submit(submitted).unwrap();
        let crate::InferenceRequestCleanup::Tracked(next) = driver.request_cleanup(RequestId(42))
        else {
            panic!("missing new generation")
        };
        assert!(!next.is_released());
        assert!(receipt.is_released());
        assert_eq!(driver.admission_snapshot().request_identities_held, 2);
    }

    fn assert_pr12_duplicate_protection(
        driver: &mut ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool>,
        label: &str,
    ) {
        let before = driver.admission_snapshot();
        let submitted = driver.scheduler.total_submitted();
        let slots = driver.slot_pool.active_count();
        let pages = driver.page_manager().unwrap().allocated_pages();
        let terminal_counts = (
            driver.scheduler.finished_len(),
            driver.scheduler.cancelled_len(),
            driver.scheduler.failed_len(),
        );
        let mut duplicate = request(1, &[2], 1, vec![]);
        duplicate.session_id = Some(SessionId(99));
        assert!(
            matches!(
                driver.try_submit(duplicate.clone()),
                Err(Error::Admission {
                    source: RuntimeAdmissionError::DuplicateRequest {
                        request_id: RequestId(1)
                    }
                })
            ),
            "{label}"
        );
        driver.submit(duplicate);
        let mut concurrent = request(99, &[2], 1, vec![]);
        concurrent.session_id = Some(SessionId(1));
        assert!(
            matches!(
                driver.try_submit(concurrent),
                Err(Error::Admission {
                    source: RuntimeAdmissionError::SessionBusy {
                        session_id: SessionId(1)
                    }
                })
            ),
            "{label}"
        );
        assert_eq!(driver.admission_snapshot(), before);
        assert_eq!(driver.scheduler.total_submitted(), submitted);
        assert_eq!(driver.slot_pool.active_count(), slots);
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), pages);
        assert_eq!(
            (
                driver.scheduler.finished_len(),
                driver.scheduler.cancelled_len(),
                driver.scheduler.failed_len()
            ),
            terminal_counts
        );
        eprintln!(
            "PR12 container={label} snapshot={before:?} slots={slots} pages={pages} grants={} terminals={terminal_counts:?}",
            driver.kv.grant_count()
        );
    }

    #[test]
    fn pr12_identity_all_owner_containers_matrix() {
        for label in [
            "waiting",
            "active",
            "suspended",
            "quarantined",
            "resident transaction",
            "finished",
            "cancelled",
            "failed",
        ] {
            let runner = if label == "resident transaction" {
                MockTopKRunner::new(vec![top(65)]).with_resumable_wait_scripts([1])
            } else {
                MockTopKRunner::new(vec![top(65)])
            };
            let mut driver = driver_from_runner(runner)
                .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
            let mut submitted = request(1, &[1], 4, vec![]);
            submitted.session_id = Some(SessionId(1));
            driver.try_submit(submitted).unwrap();
            if label != "waiting" {
                driver.step(&mut |_| Ok(())).unwrap();
            }
            match label {
                "suspended" => driver.preempt_session(SessionId(1)).unwrap(),
                "quarantined" => {
                    driver.executor.runner_mut().preempt_errors = 1;
                    assert!(driver.preempt_session(SessionId(1)).is_err());
                }
                "finished" => {
                    driver
                        .scheduler
                        .active_sequence_mut(SessionId(1))
                        .unwrap()
                        .max_new_tokens = 1;
                    driver.drive_ready_test_work(|_| Ok(())).unwrap();
                }
                "cancelled" => {
                    driver.cancel_request(RequestId(1)).unwrap();
                }
                "failed" => {
                    driver
                        .scheduler
                        .fail_sequence(SessionId(1), &mut driver.slot_pool)
                        .unwrap();
                    driver.release_sequence_state(SessionId(1)).unwrap();
                }
                _ => {}
            }
            assert_pr12_duplicate_protection(&mut driver, label);
            if label != "quarantined" {
                driver.shutdown(&mut |_| Ok(()), 32).unwrap();
            }
        }
        for label in [
            "proposal",
            "verification",
            "bookkeeping",
            "publish pending",
            "post-publish cleanup",
        ] {
            let mut runner = MockTopKRunner::new(vec![top(10)]).with_speculative_cycle(
                NativeProposal {
                    token_ids: vec![11, 12],
                    confidence_logits: vec![1.0, 1.0],
                },
                vec![
                    TokenLogit::new(11, 9.0),
                    TokenLogit::new(99, 8.0),
                    TokenLogit::new(98, 7.0),
                ],
            );
            if label == "proposal" {
                runner = runner.with_proposal_waits(vec![1]);
            }
            if label == "verification" {
                runner = runner.with_resumable_wait_scripts([0, 1]);
            }
            let mut driver = speculative_driver_from_runner(runner, 1);
            let actions = ready_speculative_decode_actions(&mut driver, &[1]);
            if label == "bookkeeping" {
                driver.stage_faults.push_back("spec observation");
            }
            if label == "publish pending" {
                driver
                    .executor
                    .runner_mut()
                    .publish_progress
                    .push_back(Ok(TransactionEndProgress::Pending));
            }
            if label == "post-publish cleanup" {
                driver.executor.runner_mut().release_failures_remaining = 1;
            }
            let result = driver.execute_speculative_decode_batch(actions, &mut |_| Ok(()));
            if label == "bookkeeping" || label == "post-publish cleanup" {
                assert!(result.is_err());
            } else {
                result.unwrap();
            }
            assert_pr12_duplicate_protection(&mut driver, label);
            driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        }
    }

    #[test]
    fn pr12_identity_capacity_moves_without_reacquiring_and_waiting_is_subset() {
        let mut driver = fork_driver();
        let limits = RuntimeAdmissionOptions {
            max_waiting_requests: 1,
            max_request_identities: 2,
            max_session_identities: 2,
        };
        driver.set_admission_options(limits).unwrap();
        driver.try_submit(request(1, &[1], 4, vec![])).unwrap();
        assert_eq!(driver.admission_snapshot().waiting_requests, 1);
        driver.step(&mut |_| Ok(())).unwrap();
        assert_eq!(driver.admission_snapshot().waiting_requests, 0);
        assert_eq!(driver.admission_snapshot().request_identities_held, 1);
        driver.preempt_session(SessionId(1)).unwrap();
        driver.try_submit(request(2, &[2], 1, vec![])).unwrap();
        let snapshot = driver.admission_snapshot();
        assert_eq!(snapshot.request_identities_held, 2);
        assert_eq!(snapshot.waiting_requests, 1);
        assert!(matches!(
            driver.try_submit(request(3, &[3], 1, vec![])),
            Err(Error::Admission {
                source: RuntimeAdmissionError::Capacity {
                    resource: RuntimeAdmissionResource::RequestIdentities,
                    ..
                }
            })
        ));
        driver.cancel_request(RequestId(2)).unwrap();
        assert_eq!(driver.admission_snapshot().waiting_requests, 0);
        assert!(
            driver.try_submit(request(3, &[3], 1, vec![])).is_err(),
            "unconsumed terminal still holds permit"
        );
        assert_eq!(driver.drain_cancelled().len(), 1);
        driver.try_submit(request(3, &[3], 1, vec![])).unwrap();
        assert_eq!(driver.admission_snapshot().request_identities_held, 2);
        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert_eq!(driver.admission_snapshot().request_identities_held, 2);
        assert_eq!(driver.drain_cancelled().len(), 2);
        assert_eq!(driver.admission_snapshot().request_identities_held, 0);
        assert_eq!(driver.admission_snapshot().session_identities_held, 0);
        assert!(matches!(
            driver.try_submit(request(1, &[1], 1, vec![])),
            Err(Error::Admission {
                source: RuntimeAdmissionError::Closed
            })
        ));
    }

    #[test]
    fn pr12_options_position_and_auto_session_validation_are_atomic() {
        let mut driver = fork_driver();
        let limits = RuntimeAdmissionOptions {
            max_waiting_requests: 2,
            max_request_identities: 2,
            max_session_identities: 2,
        };
        driver.set_admission_options(limits).unwrap();
        for options in [
            RuntimeAdmissionOptions {
                max_waiting_requests: 0,
                ..limits
            },
            RuntimeAdmissionOptions {
                max_waiting_requests: 3,
                ..limits
            },
        ] {
            assert!(matches!(
                driver.set_admission_options(options),
                Err(Error::Admission {
                    source: RuntimeAdmissionError::InvalidOptions
                })
            ));
            assert_eq!(driver.admission_options(), limits);
        }
        driver.retain_session(SessionId(1)).unwrap();
        driver.try_submit(request(1, &[1], 1, vec![])).unwrap();
        assert_eq!(
            driver.request_identities[&RequestId(1)].session_id,
            SessionId(2)
        );
        let before = driver.admission_snapshot();
        assert!(matches!(driver.retain_session(SessionId(3)), Err(_)));
        assert!(matches!(
            driver.set_admission_options(RuntimeAdmissionOptions {
                max_session_identities: 1,
                ..limits
            }),
            Err(Error::Admission {
                source: RuntimeAdmissionError::OptionsInUse
            })
        ));
        assert_eq!(driver.admission_snapshot(), before);
        let mut turn = request(2, &[1], 1, vec![]);
        turn.session_id = Some(SessionId(1));
        assert!(matches!(
            driver.try_submit_at_position(turn, 99),
            Err(Error::Admission {
                source: RuntimeAdmissionError::RetainedPositionMismatch { .. }
            })
        ));
        assert_eq!(driver.admission_snapshot(), before);
        driver.cancel_request(RequestId(1)).unwrap();
        driver.drain_cancelled();
        driver.release_session(SessionId(1)).unwrap();
        driver.next_admission_session = Some(u64::MAX);
        driver.retain_session(SessionId(u64::MAX)).unwrap();
        let before = driver.admission_snapshot();
        assert!(matches!(
            driver.try_submit(request(3, &[1], 1, vec![])),
            Err(Error::Admission {
                source: RuntimeAdmissionError::SessionIdentityExhausted
            })
        ));
        assert_eq!(driver.admission_snapshot(), before);
    }

    #[test]
    fn pr12_failed_admission_fork_keeps_request_and_session_until_slot_release() {
        let mut driver = ResidentTopKDriver::with_configs(
            MockTopKRunner::new(vec![top(65)]),
            FailOnceSequenceSlotPool::new(2, 1),
            ResidentSchedulerConfig {
                max_active_sequences: 2,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        let mut source = request(1, &[1], 4, vec![]);
        source.session_id = Some(SessionId(1));
        driver.try_submit(source).unwrap();
        driver.step(&mut |_| Ok(())).unwrap();
        let mut target = request(2, &[2], 1, vec![]);
        target.session_id = Some(SessionId(2));
        assert!(
            driver
                .fork_session_exact(SessionId(1), target.clone(), 999)
                .is_err()
        );
        assert_eq!(driver.admission_snapshot().request_identities_held, 2);
        let mut same_id = target.clone();
        same_id.session_id = Some(SessionId(99));
        assert!(matches!(
            driver.try_submit(same_id),
            Err(Error::Admission {
                source: RuntimeAdmissionError::DuplicateRequest {
                    request_id: RequestId(2)
                }
            })
        ));
        let mut same_session = target.clone();
        same_session.id = RequestId(99);
        assert!(matches!(
            driver.try_submit(same_session),
            Err(Error::Admission {
                source: RuntimeAdmissionError::SessionBusy {
                    session_id: SessionId(2)
                }
            })
        ));
        assert!(
            driver.take_request_terminal(RequestId(2)).is_none(),
            "rejected fork has no business terminal"
        );
        assert_eq!(driver.slot_pool.active_count(), 2);
        driver.progress_pending_cleanups().unwrap();
        assert_eq!(driver.admission_snapshot().request_identities_held, 1);
        driver.fork_session_exact(SessionId(1), target, 1).unwrap();
        assert_eq!(driver.admission_snapshot().request_identities_held, 2);
        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        driver.drain_cancelled();
        assert_eq!(driver.admission_snapshot().request_identities_held, 0);
    }

    #[test]
    fn pr12_failed_scheduler_slot_terminal_is_not_consumed_early() {
        let mut driver = ResidentTopKDriver::with_configs(
            MockTopKRunner::new(vec![top(65)]),
            FailOnceSequenceSlotPool::new(1, 1),
            ResidentSchedulerConfig::default(),
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        );
        let mut request = request(1, &[1], 4, vec![]);
        request.session_id = Some(SessionId(1));
        driver.try_submit(request.clone()).unwrap();
        driver.step(&mut |_| Ok(())).unwrap();
        assert!(driver.cancel_request(RequestId(1)).is_err());
        assert!(driver.drain_failed().is_empty());
        assert!(driver.take_request_terminal(RequestId(1)).is_none());
        assert_eq!(driver.slot_pool.active_count(), 1);
        assert!(driver.try_submit(request.clone()).is_err());
        driver.cancel_request(RequestId(1)).unwrap();
        assert!(
            driver.try_submit(request.clone()).is_err(),
            "unconsumed failure retains ID after slot cleanup"
        );
        assert_eq!(driver.drain_failed().len(), 1);
        driver.try_submit(request).unwrap();
    }

    #[test]
    fn pr12_fork_respects_identity_capacity_duplicates_and_closed_admission() {
        let mut driver = fork_driver();
        driver
            .set_admission_options(RuntimeAdmissionOptions {
                max_waiting_requests: 1,
                max_request_identities: 1,
                max_session_identities: 2,
            })
            .unwrap();
        let mut source = request(1, &[1], 4, vec![]);
        source.session_id = Some(SessionId(1));
        driver.try_submit(source).unwrap();
        driver.step(&mut |_| Ok(())).unwrap();
        let mut target = request(2, &[2], 1, vec![]);
        target.session_id = Some(SessionId(2));
        assert!(matches!(
            driver.fork_session_exact(SessionId(1), target.clone(), 1),
            Err(Error::Admission {
                source: RuntimeAdmissionError::Capacity {
                    resource: RuntimeAdmissionResource::RequestIdentities,
                    ..
                }
            })
        ));
        target.id = RequestId(1);
        assert!(matches!(
            driver.fork_session_exact(SessionId(1), target.clone(), 1),
            Err(Error::Admission {
                source: RuntimeAdmissionError::DuplicateRequest { .. }
            })
        ));
        driver.close_admission();
        assert!(matches!(
            driver.fork_session_exact(SessionId(1), target, 1),
            Err(Error::Admission {
                source: RuntimeAdmissionError::Closed
            })
        ));
        assert_eq!(driver.slot_pool.active_count(), 1);
        assert_eq!(driver.executor.runner().sequence_state_fork_calls, 0);
    }

    #[test]
    fn pr12_inference_box_forwards_capacity_and_suspended_disposition() {
        use crate::engine::{
            BoxedSessionInferenceEngine, InferenceCancelProgress, InferenceEngine,
            ResidentInferenceEngine,
        };
        let mut driver = fork_driver();
        let mut request = request(1, &[1], 4, vec![]);
        request.session_id = Some(SessionId(1));
        driver.try_submit(request.clone()).unwrap();
        driver.step(&mut |_| Ok(())).unwrap();
        driver.preempt_session(SessionId(1)).unwrap();
        let mut engine: BoxedSessionInferenceEngine =
            Box::new(ResidentInferenceEngine::new(driver));
        let options = RuntimeAdmissionOptions {
            max_waiting_requests: 1,
            max_request_identities: 1,
            max_session_identities: 1,
        };
        engine.set_admission_options(options).unwrap();
        assert_eq!(
            engine.admission_snapshot().unwrap().request_identities_held,
            1
        );
        assert_eq!(
            engine.cancel_request(RequestId(1)).unwrap(),
            InferenceCancelProgress::RequiresRestoreOrShutdown {
                request_id: RequestId(1),
                session_id: SessionId(1)
            }
        );
        assert!(engine.take_request_terminal(RequestId(1)).is_none());
        assert!(engine.try_submit(request.clone()).is_err());
        engine.close_admission().unwrap();
        assert!(engine.admission_snapshot().unwrap().closed);
        assert!(matches!(
            engine.try_submit(request),
            Err(Error::Admission {
                source: RuntimeAdmissionError::Closed
            })
        ));
        engine.shutdown().unwrap();
        assert!(engine.take_request_terminal(RequestId(1)).is_some());
        assert_eq!(
            engine.admission_snapshot().unwrap().request_identities_held,
            0
        );
    }

    #[test]
    fn pr12_session_capacity_counts_idle_retained_sessions() {
        let mut driver = fork_driver();
        driver
            .set_admission_options(RuntimeAdmissionOptions {
                max_waiting_requests: 1,
                max_request_identities: 2,
                max_session_identities: 1,
            })
            .unwrap();
        driver.retain_session(SessionId(1)).unwrap();
        let before = driver.admission_snapshot();
        assert!(matches!(
            driver.retain_session(SessionId(2)),
            Err(Error::Admission {
                source: RuntimeAdmissionError::Capacity {
                    resource: RuntimeAdmissionResource::SessionIdentities,
                    ..
                }
            })
        ));
        assert!(matches!(
            driver.try_submit(request(2, &[2], 1, vec![])),
            Err(Error::Admission {
                source: RuntimeAdmissionError::Capacity {
                    resource: RuntimeAdmissionResource::SessionIdentities,
                    ..
                }
            })
        ));
        assert_eq!(driver.admission_snapshot(), before);
        driver.release_session(SessionId(1)).unwrap();
        driver.try_submit(request(2, &[2], 1, vec![])).unwrap();
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
        let target_model = driver.sessions.sequence_state(&target).unwrap();
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
        assert!(!driver.sessions.contains_sequence_state(&SessionId(2)));
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
            .sessions
            .sequence_state_mut(&source)
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
        assert!(!driver.sessions.contains_sequence_state(&SessionId(2)));
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
        assert!(driver.sessions.sequence_state_count() == 0);
        assert!(driver.sessions.page_slot_count() == 0);
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

    // These assertions observe actual owner methods and completed operations, not
    // a second scheduler. Runner/provider counters and custody assertions below
    // distinguish entering a phase from successfully performing its work.
    fn pr18_trace_prefix() -> Vec<DriverTickEvent> {
        use DriverTickEvent::*;
        vec![
            ResidentEndings,
            SpeculativeEndings,
            PendingCleanups,
            RequestCancellations,
            Outbox,
            Warmup,
            Materialization,
            WarmupCheck,
            Observability,
        ]
    }

    #[test]
    fn pr18_tick_cleanup_failure_then_cancel_ack_warmup_and_admission() {
        use DriverTickEvent::*;
        let (physical, handle) = MockPhysicalProvider::manual();
        let runner = MockTopKRunner::new(vec![top(65), top(66)])
            .with_materialization_provider(Box::new(physical))
            .with_warmup(materialization_request(1));
        let mut driver = concurrent_transaction_driver(runner);
        driver.config.enable_native_proposals = false;
        driver.start_background_work().unwrap();
        for id in [1, 2] {
            let mut req = request(id, &[id as u32], 2, vec![]);
            req.session_id = Some(SessionId(id));
            driver.submit(req);
        }
        driver.prepare_step().unwrap();
        // Produce a committed, unacknowledged event through the real executor.
        for id in [1, 2] {
            let action = driver
                .scheduler
                .next_prefill_action(&mut driver.slot_pool)
                .unwrap()
                .unwrap();
            let result = driver.execute_planned_action(action, &mut |_| {
                if id == 2 {
                    Err(Error::InvalidRequest {
                        message: "callback not accepted".into(),
                    })
                } else {
                    Ok(())
                }
            });
            assert_eq!(result.is_err(), id == 2);
        }
        assert_eq!(driver.output.queued_count(), 1);
        driver.executor.runner_mut().release_failures_remaining = 2;
        assert!(driver.cancel_request(RequestId(1)).is_err());
        assert!(driver.sessions.has_cleanup(&SessionId(1)));
        assert!(driver.output.cancellation_accepted(Some(RequestId(1))));
        let mut req = request(3, &[3], 0, vec![]);
        req.session_id = Some(SessionId(3));
        driver.submit(req);
        let committed = driver.executor.runner().committed_batches;
        driver.tick_trace.clear();
        assert!(
            driver
                .step(&mut |_| panic!("cleanup failure must precede callback"))
                .is_err()
        );
        assert_eq!(
            driver.tick_trace,
            [ResidentEndings, SpeculativeEndings, PendingCleanups]
        );
        assert_eq!(driver.executor.runner().committed_batches, committed);
        assert!(!driver.sessions.contains_sequence_state(&SessionId(3)));
        assert_eq!(driver.output.queued_count(), 1);

        driver.tick_trace.clear();
        let mut delivered = vec![];
        driver
            .step(&mut |event| {
                delivered.push(event.clone());
                Ok(())
            })
            .unwrap();
        let mut expected = pr18_trace_prefix();
        expected.insert(4, CancelComplete(RequestId(1)));
        expected.insert(6, TokenAcknowledged(SessionId(2), 65));
        expected.extend([
            ReadySnapshot(0),
            PrepareStep,
            Admission,
            Admitted(SessionId(3)),
        ]);
        assert!(
            driver.tick_trace.starts_with(&expected),
            "{:?}",
            driver.tick_trace
        );
        assert_eq!(delivered[0].request_id, Some(RequestId(2)));
        assert_eq!(delivered.iter().filter(|e| e.index == 0).count(), 1);
        assert!(driver.sessions.cleanup_count() == 0);
        assert!(driver.output.cancellations_empty());
        assert!(driver.scheduler.active_sequence(SessionId(3)).is_some());
        assert!(driver.warmup_pending());
        assert!(driver.warmup_requests.is_empty());
        assert_eq!(
            handle.command_count(|c| matches!(c, MockPhysicalCommand::SubmitRead(..))),
            1
        );
        assert!(!driver.stats().hard_resource_high_water.is_empty());
        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
    }

    #[test]
    fn pr18_tick_outbox_nack_blocks_warmup_snapshot_and_admission() {
        use DriverTickEvent::*;
        let (physical, handle) = MockPhysicalProvider::manual();
        let runner = MockTopKRunner::new(vec![top(65)])
            .with_materialization_provider(Box::new(physical))
            .with_warmup(materialization_request(1));
        let mut driver = driver_from_runner(runner);
        driver.config.enable_native_proposals = false;
        driver.submit(request(1, &[1], 1, vec![]));
        let reject = || Error::InvalidRequest {
            message: "not acknowledged".into(),
        };
        assert!(driver.step(&mut |_| Err(reject())).is_err());
        assert_eq!(driver.executor.runner().committed_batches, 1);
        // First step has already started warmup; its blocked physical work must
        // not advance on a subsequent callback failure.
        let operations = driver.load_registry.stats().operations_created;
        let commands = handle.command_count(|_| true);
        driver.submit(request(2, &[2], 1, vec![]));
        driver.tick_trace.clear();
        assert!(driver.step(&mut |_| Err(reject())).is_err());
        assert_eq!(
            driver.tick_trace,
            [
                ResidentEndings,
                SpeculativeEndings,
                PendingCleanups,
                RequestCancellations,
                Outbox
            ]
        );
        assert_eq!(driver.load_registry.stats().operations_created, operations);
        assert_eq!(handle.command_count(|_| true), commands);
        assert_eq!(driver.executor.runner().committed_batches, 1);
        assert_eq!(driver.stats().emitted_tokens, 0);
        assert_eq!(driver.output.queued_count(), 1);
        assert!(!driver.sessions.contains_sequence_state(&SessionId(2)));
        driver.tick_trace.clear();
        let mut delivered = vec![];
        driver
            .step(&mut |e| {
                delivered.push(e.token);
                Ok(())
            })
            .unwrap();
        assert_eq!(delivered, [65]);
        assert_eq!(driver.stats().emitted_tokens, 1);
        assert!(
            driver
                .tick_trace
                .contains(&TokenAcknowledged(SessionId(1), 65))
        );
        assert!(driver.tick_trace.contains(&ReadySnapshot(0)));
        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
    }

    #[test]
    fn pr18_tick_pending_and_complete_endings_preserve_early_return() {
        use DriverTickEvent::*;
        for speculative in [false, true] {
            let runner = MockTopKRunner::new(vec![top(65), top(66)]).with_speculative_cycle(
                NativeProposal {
                    token_ids: vec![66, 67],
                    confidence_logits: vec![1.0, -10.0],
                },
                vec![TokenLogit::new(66, 9.0), TokenLogit::new(67, 9.0)],
            );
            let mut driver = speculative_driver_from_runner(runner, 1);
            driver.executor.runner_mut().native_proposal_enabled = speculative;
            driver.executor.runner_mut().publish_progress = VecDeque::from([
                Ok(TransactionEndProgress::Complete),
                Ok(TransactionEndProgress::Pending),
                Ok(TransactionEndProgress::Pending),
                Ok(TransactionEndProgress::Complete),
            ]);
            driver.submit(request(1, &[1], 3, vec![]));
            driver.step(&mut |_| Ok(())).unwrap();
            assert_eq!(
                driver.step(&mut |_| panic!("publish pending")).unwrap(),
                ResidentDriverStep::Blocked
            );
            driver.submit(request(2, &[2], 0, vec![]));
            driver.tick_trace.clear();
            let calls = driver.executor.runner().end_intents.len();
            assert_eq!(
                driver
                    .step(&mut |_| panic!("publish still pending"))
                    .unwrap(),
                ResidentDriverStep::Blocked
            );
            assert_eq!(driver.executor.runner().end_intents.len(), calls + 1);
            let mut expected = pr18_trace_prefix();
            expected.insert(3, CleanupDeferred);
            expected.extend([ReadySnapshot(0), PrepareStep, Admission]);
            assert_eq!(driver.tick_trace, expected);
            assert!(driver.has_live_transactions());
            assert!(!driver.sessions.contains_sequence_state(&SessionId(2)));
            driver.tick_trace.clear();
            let mut events = vec![];
            assert!(matches!(
                driver
                    .step(&mut |e| {
                        events.push(e.token);
                        Ok(())
                    })
                    .unwrap(),
                ResidentDriverStep::Executed {
                    action_kind: ResidentActionKind::Decode,
                    ..
                }
            ));
            let mut expected = vec![ResidentEndings];
            if speculative {
                expected.push(SpeculativeEndings);
            }
            expected.extend([Outbox, TokenAcknowledged(SessionId(1), 65)]);
            if speculative {
                expected.push(TokenAcknowledged(SessionId(1), 66));
            }
            assert_eq!(driver.tick_trace, expected);
            assert_eq!(events, if speculative { vec![65, 66] } else { vec![65] });
            assert_eq!(driver.executor.runner().committed_batches, 2);
            assert!(!driver.has_live_transactions());
            assert!(!driver.sessions.contains_sequence_state(&SessionId(2)));
            assert_eq!(
                driver.executor.runner().packed_verification_calls,
                usize::from(speculative)
            );
            driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        }
    }

    #[test]
    fn pr18_tick_bookkeeping_fault_retries_before_endings_without_reexecution() {
        use DriverTickEvent::*;
        let runner = MockTopKRunner::new(vec![top(65)]).with_speculative_cycle(
            NativeProposal {
                token_ids: vec![66, 67],
                confidence_logits: vec![1.0, -10.0],
            },
            vec![TokenLogit::new(66, 9.0), TokenLogit::new(67, 9.0)],
        );
        let mut driver = speculative_driver_from_runner(runner, 1);
        driver.submit(request(1, &[1], 3, vec![]));
        driver.step(&mut |_| Ok(())).unwrap();
        driver.stage_faults.push_back("spec observation");
        assert!(
            driver
                .step(&mut |_| panic!("unpublished proposal"))
                .unwrap_err()
                .to_string()
                .contains("spec observation")
        );
        assert!(matches!(
            driver.speculative_transactions.values().next(),
            Some(PendingSpeculativeDriverCohort::Bookkeeping(_))
        ));
        assert_eq!(driver.executor.runner().native_proposal_begin_calls, 1);
        assert_eq!(driver.executor.runner().packed_verification_calls, 0);
        driver.tick_trace.clear();
        driver.step(&mut |_| Ok(())).unwrap();
        assert_eq!(
            driver.tick_trace,
            [
                Bookkeeping,
                Outbox,
                TokenAcknowledged(SessionId(1), 65),
                TokenAcknowledged(SessionId(1), 66)
            ]
        );
        assert_eq!(driver.executor.runner().native_proposal_begin_calls, 1);
        assert_eq!(driver.executor.runner().packed_verification_calls, 1);
        assert!(driver.speculative_transactions.is_empty());
        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
    }

    #[test]
    fn pr18_tick_ready_snapshot_deduplicates_shared_transaction_and_bounds_requeue() {
        use DriverTickEvent::*;
        let (physical, handle) = MockPhysicalProvider::automatic();
        let runner = MockTopKRunner::new(vec![top(10)])
            .with_speculative_cohort(
                vec![
                    NativeProposal {
                        token_ids: vec![11, 12],
                        confidence_logits: vec![1.0, -10.0],
                    },
                    NativeProposal {
                        token_ids: vec![21, 22],
                        confidence_logits: vec![1.0, -10.0],
                    },
                ],
                vec![
                    TokenLogit::new(11, 9.0),
                    TokenLogit::new(99, 9.0),
                    TokenLogit::new(21, 9.0),
                    TokenLogit::new(98, 9.0),
                ],
            )
            .with_proposal_waits(vec![2, 1])
            .with_materialization_request(materialization_request(1))
            .with_materialization_provider(Box::new(physical));
        let mut driver = speculative_driver_from_runner(runner, 2);
        let actions = ready_speculative_decode_actions(&mut driver, &[1, 2]);
        assert!(matches!(
            driver
                .execute_speculative_decode_batch(actions, &mut |_| Ok(()))
                .unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        let transaction = driver.sessions.owner(&SessionId(1)).unwrap();
        assert_eq!(driver.sessions.owner(&SessionId(2)).unwrap(), transaction);
        for resumes in [1, 2] {
            driver.tick_trace.clear();
            assert!(matches!(
                driver
                    .step(&mut |_| panic!("not all proposals ready"))
                    .unwrap(),
                ResidentDriverStep::WaitingForModelProgress(_)
            ));
            assert!(
                driver.tick_trace.contains(&ReadySnapshot(1)),
                "{:?}",
                driver.tick_trace
            );
            assert_eq!(
                driver
                    .tick_trace
                    .iter()
                    .filter(|e| matches!(e, Resume(_)))
                    .count(),
                1
            );
            assert!(driver.tick_trace.contains(&Resume(transaction)));
            assert_eq!(
                driver.executor.runner().native_proposal_resume_calls,
                resumes
            );
            assert_eq!(
                driver.continuations.ready_snapshot_len(),
                1,
                "already-ready sibling is requeued, not resumed again within this tick"
            );
            assert_eq!(driver.executor.runner().native_proposal_begin_calls, 2);
            assert_eq!(driver.executor.runner().packed_verification_calls, 0);
        }
        assert_eq!(
            handle.command_count(|c| matches!(c, MockPhysicalCommand::SubmitRead(..))),
            1
        );
        driver.tick_trace.clear();
        driver.step(&mut |_| Ok(())).unwrap();
        assert!(driver.tick_trace.contains(&ReadySnapshot(1)));
        assert_eq!(
            driver
                .tick_trace
                .iter()
                .filter(|e| matches!(e, Resume(_)))
                .count(),
            1
        );
        assert!(
            !driver.tick_trace.contains(&PrepareStep),
            "completed resume returns before admission"
        );
        assert_eq!(driver.executor.runner().native_proposal_resume_calls, 3);
        assert_eq!(driver.executor.runner().packed_verification_calls, 1);
        assert_eq!(driver.continuations.ready_snapshot_len(), 0);
        assert!(driver.continuations.is_empty());
        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
    }

    #[test]
    fn pr18_tick_materialization_failure_precedes_snapshot_and_new_admission() {
        use DriverTickEvent::*;
        let (physical, handle) = MockPhysicalProvider::automatic();
        handle.script_outcome(
            LoadStage::ReadSubmitted,
            CompletionOutcome::Failed(FailureReason::StorageUnavailable),
        );
        let runner = MockTopKRunner::new(vec![])
            .with_resumable_wait_scripts([1])
            .with_materialization_request(materialization_request(7))
            .with_materialization_provider(Box::new(physical));
        let mut driver = concurrent_transaction_driver(runner);
        driver.submit(request(1, &[1], 2, vec![]));
        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::WaitingForModelProgress(_)
        ));
        driver.submit(request(2, &[2], 2, vec![]));
        driver.tick_trace.clear();
        assert!(
            driver
                .step(&mut |_| panic!("failed load cannot publish"))
                .unwrap_err()
                .to_string()
                .contains("StorageUnavailable")
        );
        assert_eq!(
            driver.tick_trace,
            [
                ResidentEndings,
                SpeculativeEndings,
                PendingCleanups,
                CleanupDeferred,
                RequestCancellations,
                Outbox,
                Warmup,
                Materialization
            ]
        );
        assert_eq!(driver.executor.runner().rolled_back_batches, 1);
        assert_eq!(driver.executor.runner().committed_batches, 0);
        assert!(!driver.sessions.contains_sequence_state(&SessionId(2)));
        assert!(driver.continuations.is_empty());
        assert_eq!(driver.continuations.ready_snapshot_len(), 0);
        assert_eq!(
            handle.command_count(|c| matches!(c, MockPhysicalCommand::SubmitRead(..))),
            1
        );
        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
    }

    #[test]
    fn pr18_tick_warmup_failure_and_missing_page_manager_stop_before_snapshot() {
        use DriverTickEvent::*;
        let (physical, handle) = MockPhysicalProvider::automatic();
        handle.script_outcome(
            LoadStage::ReadSubmitted,
            CompletionOutcome::Failed(FailureReason::StorageUnavailable),
        );
        let runner = MockTopKRunner::new(vec![])
            .with_materialization_provider(Box::new(physical))
            .with_warmup(materialization_request(1));
        let mut driver = driver_from_runner(runner);
        assert!(matches!(
            driver.step(&mut |_| Ok(())).unwrap_err(),
            Error::WarmupMaterialization { .. }
        ));
        assert_eq!(driver.tick_trace, pr18_trace_prefix()[..8]);
        assert_eq!(
            handle.command_count(|c| matches!(c, MockPhysicalCommand::SubmitRead(..))),
            1
        );
        assert_eq!(driver.executor.runner().prepared_batches, 0);

        let mut runner = MockTopKRunner::new(vec![top(65)]);
        runner.native_proposal_enabled = true;
        let mut driver = driver_from_runner(runner);
        driver.submit(request(1, &[1], 1, vec![]));
        assert!(
            driver
                .step(&mut |_| Ok(()))
                .unwrap_err()
                .to_string()
                .contains("KvPageManager")
        );
        assert_eq!(driver.tick_trace, pr18_trace_prefix());
        assert_eq!(driver.executor.runner().prepared_batches, 0);
        assert!(driver.sessions.sequence_state_count() == 0);
        assert!(!driver.tick_trace.contains(&Admission));
    }

    fn pr18_matrix_driver(
        enabled: bool,
        capability: bool,
        prefix_pages: usize,
        limit: usize,
    ) -> ResidentTopKDriver<MockTopKRunner, FixedSequenceSlotPool> {
        let predictions = if limit == 1 {
            vec![TokenLogit::new(66, 9.0)]
        } else {
            vec![
                TokenLogit::new(66, 9.0),
                TokenLogit::new(67, 9.0),
                TokenLogit::new(68, 9.0),
            ]
        };
        let mut runner = MockTopKRunner::new(vec![top(65), top(66), top(67), top(68)])
            .with_speculative_cycle(
                NativeProposal {
                    token_ids: vec![66, 67],
                    confidence_logits: vec![1.0, 1.0],
                },
                predictions,
            );
        runner.native_proposal_enabled = capability;
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(1),
            ResidentSchedulerConfig {
                prefill_chunk_size: 8,
                max_active_sequences: 1,
                max_decode_batch: 1,
                max_batch_tokens: 8,
                allow_mixed_batches: false,
                prefix_cache_capacity_pages: prefix_pages,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig {
                enable_native_proposals: enabled,
                ..Default::default()
            },
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        driver.retain_session(SessionId(1)).unwrap();
        let mut req = request(1, &[9], limit, vec![]);
        req.session_id = Some(SessionId(1));
        driver.submit(req);
        driver
    }

    #[test]
    fn pr18_early_delivery_policy_capability_prefix_and_draft_capacity_matrix() {
        for enabled in [false, true] {
            for capability in [false, true] {
                for prefix_pages in [0, 2] {
                    for limit in [1, 3] {
                        let case = (enabled, capability, prefix_pages, limit);
                        let mut driver =
                            pr18_matrix_driver(enabled, capability, prefix_pages, limit);
                        driver.executor.runner_mut().publish_progress = VecDeque::from([
                            Ok(TransactionEndProgress::Pending),
                            Ok(TransactionEndProgress::Complete),
                            Ok(TransactionEndProgress::Pending),
                            Ok(TransactionEndProgress::Complete),
                        ]);
                        let mut events = vec![];
                        assert_eq!(
                            driver
                                .step(&mut |_| panic!("uncommitted prefill: {case:?}"))
                                .unwrap(),
                            ResidentDriverStep::Blocked
                        );
                        assert_eq!(driver.executor.runner().committed_batches, 0);
                        driver
                            .step(&mut |e| {
                                events.push(e.clone());
                                Ok(())
                            })
                            .unwrap();
                        assert_eq!(
                            events.len(),
                            usize::from(!enabled),
                            "prefill delivery: {case:?}"
                        );
                        assert_eq!(driver.executor.runner().committed_batches, 1);
                        assert_eq!(
                            driver
                                .page_manager()
                                .unwrap()
                                .block_table(StateSlot::new(0))
                                .unwrap()
                                .committed_tokens(),
                            1
                        );
                        assert_eq!(
                            driver
                                .scheduler
                                .active_sequence(SessionId(1))
                                .unwrap()
                                .generated,
                            0
                        );
                        assert_eq!(driver.output.early_count(), usize::from(!enabled));
                        assert_eq!(
                            driver
                                .step(&mut |_| panic!("uncommitted decode: {case:?}"))
                                .unwrap(),
                            ResidentDriverStep::Blocked
                        );
                        let proposal = enabled && capability && prefix_pages == 0;
                        assert_eq!(
                            driver.executor.runner().packed_verification_calls,
                            usize::from(proposal),
                            "{case:?}"
                        );
                        assert_eq!(
                            driver.executor.runner().native_proposal_begin_calls,
                            usize::from(proposal && limit > 1),
                            "{case:?}"
                        );
                        driver
                            .drive_ready_test_work(|e| {
                                events.push(e.clone());
                                Ok(())
                            })
                            .unwrap();
                        assert_eq!(
                            events.iter().map(|e| e.token).collect::<Vec<_>>(),
                            (65..65 + limit as u32).collect::<Vec<_>>(),
                            "{case:?}"
                        );
                        assert_eq!(
                            events.iter().map(|e| e.index).collect::<Vec<_>>(),
                            (0..limit).collect::<Vec<_>>(),
                            "{case:?}"
                        );
                        assert_eq!(driver.stats().emitted_tokens, limit);
                        assert_eq!(
                            driver.retained_session_position(SessionId(1)),
                            Some(1 + limit)
                        );
                        assert_eq!(
                            driver
                                .page_manager()
                                .unwrap()
                                .block_table(StateSlot::new(0))
                                .unwrap()
                                .committed_tokens(),
                            1 + limit
                        );
                        assert_eq!(driver.drain_finished().len(), 1);
                        assert!(driver.output.early_empty());
                        assert!(driver.output.outbox_empty());
                        assert_eq!(
                            driver.executor.runner().packed_verification_calls,
                            usize::from(proposal),
                            "publish retry must not reverify: {case:?}"
                        );
                        driver.release_session(SessionId(1)).unwrap();
                        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
                    }
                }
            }
        }
    }

    #[test]
    fn pr18_early_delivery_matrix_callback_retry_keeps_committed_output_exactly_once() {
        for enabled in [false, true] {
            for capability in [false, true] {
                for prefix_pages in [0, 2] {
                    let case = (enabled, capability, prefix_pages);
                    let mut driver = pr18_matrix_driver(enabled, capability, prefix_pages, 3);
                    let mut rejected = None;
                    for _ in 0..4 {
                        let result = driver.step(&mut |event| {
                            assert!(rejected.is_none());
                            rejected = Some(event.clone());
                            Err(Error::InvalidRequest {
                                message: "retry callback".into(),
                            })
                        });
                        if result.is_err() {
                            break;
                        }
                    }
                    let rejected = rejected.expect("matrix must reach a callback");
                    assert_eq!(rejected.token, 65, "{case:?}");
                    assert_eq!(driver.output.front(), Some(&rejected));
                    assert_eq!(driver.stats().emitted_tokens, 0);
                    let committed = driver.executor.runner().committed_batches;
                    assert_eq!(committed, if enabled { 2 } else { 1 }, "{case:?}");
                    driver.tick_trace.clear();
                    let mut accepted = vec![];
                    driver
                        .drive_ready_test_work(|e| {
                            accepted.push(e.clone());
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(accepted[0], rejected);
                    assert_eq!(
                        accepted.iter().map(|e| e.token).collect::<Vec<_>>(),
                        [65, 66, 67],
                        "{case:?}"
                    );
                    assert_eq!(driver.stats().emitted_tokens, 3);
                    let proposal = enabled && capability && prefix_pages == 0;
                    assert_eq!(
                        driver.executor.runner().committed_batches,
                        if proposal { 2 } else { 4 },
                        "no execution replay: {case:?}"
                    );
                    assert_eq!(driver.retained_session_position(SessionId(1)), Some(4));
                    assert_eq!(driver.drain_finished().len(), 1);
                    driver.release_session(SessionId(1)).unwrap();
                    driver.shutdown(&mut |_| Ok(()), 32).unwrap();
                }
            }
        }
    }

    #[test]
    fn pr18_continuation_owner_register_resume_and_detach_preserve_siblings() {
        let (physical, handle) = MockPhysicalProvider::automatic();
        handle.set_resident(true);
        let runner = MockTopKRunner::new(vec![]).with_materialization_provider(Box::new(physical));
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            FixedSequenceSlotPool::new(3),
            ResidentSchedulerConfig {
                prefill_chunk_size: 1,
                max_active_sequences: 3,
                max_decode_batch: 3,
                max_batch_tokens: 3,
                allow_mixed_batches: false,
                ..Default::default()
            },
            NonZeroU32::new(1).unwrap(),
            ResidentTopKDriverConfig::default(),
        )
        .with_page_manager(KvPageManager::new(Box::new(DriverTestKvSchema), 16));
        let first = ExecutionTransactionId::new(400).unwrap();
        let second = ExecutionTransactionId::new(401).unwrap();
        let demand = crate::scheduling::ResourceDemand::required(ExecutionPhase::Decode);
        for (seed, transaction, id) in [(80, first, 401), (81, first, 400), (82, second, 402)] {
            let request = materialization_request(seed);
            let key = driver
                .executor
                .runner_mut()
                .materialization_resolver()
                .unwrap()
                .resolve(request)
                .unwrap();
            let progress =
                materialization_progress(transaction, ContinuationId::new(id), [(request, key)]);
            driver.register_pending_progress(&progress, demand).unwrap();
            driver.continuations.assert_consistent();
            let before = handle.commands();
            assert!(
                driver
                    .register_pending_progress(&progress, demand)
                    .unwrap_err()
                    .to_string()
                    .contains("already registered")
            );
            assert_eq!(
                handle.commands(),
                before,
                "duplicate must not attach/discard physical custody"
            );
        }
        assert_eq!(driver.continuations.len(), 3);
        assert_eq!(driver.continuations.transaction_count(), 2);
        driver.progress_materialization().unwrap();
        assert_eq!(
            driver.ready_continuation_for(first),
            Some(ContinuationId::new(400))
        );
        assert_eq!(
            driver.ready_continuation_for(second),
            Some(ContinuationId::new(402))
        );
        assert_eq!(driver.continuations.ready_snapshot_len(), 2);
        assert_eq!(driver.pop_ready_transaction(), Some(first));
        assert_eq!(driver.pop_ready_transaction(), Some(second));
        assert_eq!(driver.pop_ready_transaction(), None);
        let continuation = ContinuationId::new(400);
        let mut lease = driver.prepare_resume_lease(continuation).unwrap();
        let _leases = lease.take().unwrap();
        let started = driver.runtime_now_ns();
        driver
            .finish_resume_lease(continuation, lease, ResumeDisposition::Consumed, started)
            .unwrap();
        driver.continuations.assert_consistent();
        assert_eq!(driver.continuations.transaction(continuation), None);
        assert_eq!(driver.continuations.transaction_count(), 2);
        assert_eq!(
            driver.ready_continuation_for(first),
            Some(ContinuationId::new(401))
        );
        assert!(
            driver
                .prepare_resume_lease(continuation)
                .unwrap_err()
                .to_string()
                .contains("not registered")
        );
        driver
            .detach_registered_continuation(
                ContinuationId::new(401),
                CancellationReason::ExternalRequest,
            )
            .unwrap();
        assert_eq!(driver.continuations.continuation_for(first), None);
        assert_eq!(driver.continuations.transaction_count(), 1);
        assert_eq!(
            driver.ready_continuation_for(second),
            Some(ContinuationId::new(402))
        );
        driver.continuations.assert_consistent();
        driver
            .detach_registered_continuation(
                ContinuationId::new(402),
                CancellationReason::ExternalRequest,
            )
            .unwrap();
        driver.continuations.assert_consistent();
        assert!(driver.continuations.is_empty());
        assert_eq!(driver.continuations.transaction_count(), 0);
        assert!(!driver.continuations.has_pending_detaches());
        assert_eq!(
            driver
                .shutdown(&mut |_| Ok(()), 32)
                .unwrap()
                .registry
                .active_grants,
            0
        );
    }

    #[test]
    fn pr18_continuation_owner_failed_detach_retains_indexes_until_registry_ack() {
        let (physical, handle) = MockPhysicalProvider::manual();
        let runner = MockTopKRunner::new(vec![]).with_materialization_provider(Box::new(physical));
        let mut driver = concurrent_transaction_driver(runner);
        let transaction = ExecutionTransactionId::new(410).unwrap();
        let continuation = ContinuationId::new(410);
        let request = materialization_request(83);
        let key = driver
            .executor
            .runner_mut()
            .materialization_resolver()
            .unwrap()
            .resolve(request)
            .unwrap();
        let progress = materialization_progress(transaction, continuation, [(request, key)]);
        let demand = crate::scheduling::ResourceDemand::required(ExecutionPhase::Decode);
        driver.register_pending_progress(&progress, demand).unwrap();
        driver.progress_materialization().unwrap();
        handle.fail_next_cancel(FailureReason::DeviceUnavailable);
        assert!(
            driver
                .detach_registered_continuation(continuation, CancellationReason::ExternalRequest)
                .is_err()
        );
        driver.continuations.assert_consistent();
        assert!(driver.continuations.contains(continuation));
        assert_eq!(
            driver.continuations.continuation_for(transaction),
            Some(continuation)
        );
        assert!(driver.continuations.has_pending_detaches());
        assert!(driver.has_pending_async_work());
        assert!(driver.load_registry.active_operations() > 0);
        let before = handle.commands();
        assert!(driver.register_pending_progress(&progress, demand).is_err());
        assert_eq!(handle.commands(), before);
        // Registry acknowledgement removes only logical continuation custody.
        // Submitted physical cleanup is still retained by the existing LoadRegistry.
        driver.retry_pending_continuation_cleanups().unwrap();
        driver.continuations.assert_consistent();
        assert!(!driver.continuations.contains(continuation));
        assert_eq!(driver.continuations.continuation_for(transaction), None);
        assert!(!driver.continuations.has_pending_detaches());
        assert_eq!(driver.load_registry.active_operations(), 1);
        assert!(driver.has_pending_async_work());
        assert!(
            driver
                .load_registry
                .resources()
                .in_use(ResourceKind::LoadOperation)
                > 0
        );
        driver.progress_materialization().unwrap();
        let commands = handle.commands();
        let cancellations = commands
            .iter()
            .filter_map(|command| match command {
                MockPhysicalCommand::Cancel(_, k, _, reason) if *k == key => Some(reason),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(cancellations, vec![&CancellationReason::ExternalRequest; 2]);
        assert_eq!(driver.load_registry.active_operations(), 0);
        assert_eq!(
            driver
                .shutdown(&mut |_| Ok(()), 32)
                .unwrap()
                .registry
                .active_grants,
            0
        );
    }

    #[test]
    fn pr18_continuation_owner_partial_attach_failure_keeps_physical_undo() {
        let (physical, handle) = MockPhysicalProvider::automatic();
        let runner = MockTopKRunner::new(vec![]).with_materialization_provider(Box::new(physical));
        let mut driver = concurrent_transaction_driver(runner);
        let transaction = ExecutionTransactionId::new(420).unwrap();
        let continuation = ContinuationId::new(420);
        let request = materialization_request(84);
        // A prefetched preparation must actually promote during attachment;
        // an execution-prepared key already holds its execution lease.
        let key = driver
            .materialization_resolver
            .prepare_prefetch(request)
            .unwrap();
        let progress = materialization_progress(transaction, continuation, [(request, key)]);
        let demand = crate::scheduling::ResourceDemand::required(ExecutionPhase::Decode);
        handle.fail_next_promotion(key, FailureReason::StorageUnavailable);
        handle.fail_next_discard(key, FailureReason::DeviceUnavailable);
        let error = driver
            .register_pending_progress(&progress, demand)
            .unwrap_err();
        assert!(format!("{error:?}").contains("StorageUnavailable"));
        assert!(format!("{error:?}").contains("DeviceUnavailable"));
        driver.continuations.assert_consistent();
        assert!(driver.continuations.is_empty());
        assert_eq!(driver.continuations.transaction_count(), 0);
        assert_eq!(driver.continuations.ready_snapshot_len(), 0);
        assert_eq!(driver.load_registry.active_operations(), 1);
        assert!(driver.load_registry.has_pending_owner_work());
        assert!(driver.has_pending_async_work());
        // The failed promotion is still an unsubmitted admission attempt: the
        // operation and undo record remain, while LoadOperation credit has not
        // yet been claimed by the scheduler.
        assert_eq!(
            driver
                .load_registry
                .resources()
                .in_use(ResourceKind::LoadOperation),
            0
        );
        assert!(
            format!(
                "{:?}",
                driver
                    .register_pending_progress(&progress, demand)
                    .unwrap_err()
            )
            .contains("AdmissionCleanupPending")
        );
        assert_eq!(
            handle.command_count(
                |c| matches!(c,MockPhysicalCommand::DiscardPreparation(k) if *k==key)
            ),
            1
        );
        driver.progress_materialization().unwrap();
        assert_eq!(driver.load_registry.active_operations(), 0);
        assert_eq!(
            driver
                .load_registry
                .resources()
                .in_use(ResourceKind::LoadOperation),
            0
        );
        assert_eq!(
            handle.command_count(
                |c| matches!(c,MockPhysicalCommand::DiscardPreparation(k) if *k==key)
            ),
            2
        );
        // Retry after physical undo uses a fresh waiter epoch, without inheriting
        // a partial forward/reverse edge from either failed attachment.
        let key = driver
            .executor
            .runner_mut()
            .materialization_resolver()
            .unwrap()
            .resolve(request)
            .unwrap();
        let progress = materialization_progress(transaction, continuation, [(request, key)]);
        driver.register_pending_progress(&progress, demand).unwrap();
        driver.continuations.assert_consistent();
        assert_eq!(
            driver.continuations.continuation_for(transaction),
            Some(continuation)
        );
        driver
            .detach_registered_continuation(continuation, CancellationReason::ExternalRequest)
            .unwrap();
        assert_eq!(
            driver
                .shutdown(&mut |_| Ok(()), 32)
                .unwrap()
                .registry
                .active_grants,
            0
        );
    }

    #[test]
    fn pr18_continuation_owner_terminal_failure_retains_indexes_until_lease_cleanup() {
        let (physical, handle) = MockPhysicalProvider::automatic();
        handle.set_resident(true);
        let runner = MockTopKRunner::new(vec![]).with_materialization_provider(Box::new(physical));
        let mut driver = concurrent_transaction_driver(runner);
        let resident_request = materialization_request(85);
        let resident = driver
            .executor
            .runner_mut()
            .materialization_resolver()
            .unwrap()
            .resolve(resident_request)
            .unwrap();
        handle.set_resident(false);
        let failed_request = materialization_request(86);
        let failed = driver
            .executor
            .runner_mut()
            .materialization_resolver()
            .unwrap()
            .resolve(failed_request)
            .unwrap();
        let transaction = ExecutionTransactionId::new(430).unwrap();
        let continuation = ContinuationId::new(430);
        handle.script_outcome(
            LoadStage::ReadSubmitted,
            CompletionOutcome::Failed(FailureReason::StorageUnavailable),
        );
        handle.fail_next_release(FailureReason::DeviceUnavailable);
        let progress = materialization_progress(
            transaction,
            continuation,
            [(resident_request, resident), (failed_request, failed)],
        );
        driver
            .register_pending_progress(
                &progress,
                crate::scheduling::ResourceDemand::required(ExecutionPhase::Decode),
            )
            .unwrap();
        assert!(driver.progress_materialization().is_err());
        handle.fail_next_release(FailureReason::DeviceUnavailable);
        assert!(driver.cleanup_materialization_failures(true).is_err());
        driver.continuations.assert_consistent();
        assert!(driver.continuations.has_pending_failures());
        assert!(driver.continuations.contains(continuation));
        assert_eq!(
            driver.continuations.continuation_for(transaction),
            Some(continuation)
        );
        let error = driver.cleanup_materialization_failures(true).unwrap_err();
        assert!(error.to_string().contains("StorageUnavailable"));
        driver.continuations.assert_consistent();
        assert!(!driver.continuations.has_pending_failures());
        assert!(!driver.continuations.contains(continuation));
        assert_eq!(driver.continuations.continuation_for(transaction), None);
        assert_eq!(
            handle.command_count(
                |c| matches!(c,MockPhysicalCommand::ReleaseExecutionLease(k) if *k==resident)
            ),
            3
        );
        assert_eq!(
            driver
                .shutdown(&mut |_| Ok(()), 32)
                .unwrap()
                .registry
                .active_grants,
            0
        );
    }
    #[test]
    fn pr18_session_custody_claim_restore_preflight_preserves_both_sides() {
        let mut driver = concurrent_fake_driver(MockTopKRunner::new(Vec::new()));
        driver.submit(request(1, &[1], 2, vec![]));
        driver.submit(request(2, &[2], 2, vec![]));
        driver.prepare_step().unwrap();
        let transaction = ExecutionTransactionId::new(500).unwrap();
        assert!(
            driver
                .claim_transaction_sessions(transaction, &[SessionId(1), SessionId(1)])
                .is_err()
        );
        driver.sessions.assert_consistent();
        assert_eq!(driver.sessions.sequence_state_count(), 2);
        assert_eq!(driver.scheduler.active_len(), 2);
        // A later scheduler failure restores the already-claimed sibling.
        assert!(
            driver
                .claim_transaction_sessions(transaction, &[SessionId(1), SessionId(99)])
                .is_err()
        );
        driver.sessions.assert_consistent();
        assert_eq!(driver.sessions.owner_count(), 0);
        assert_eq!(driver.scheduler.active_len(), 2);
        let (mut schedules, mut states) = driver
            .claim_transaction_sessions(transaction, &[SessionId(1), SessionId(2)])
            .unwrap();
        driver.sessions.assert_consistent();
        assert_eq!(driver.sessions.sequence_state_count(), 0);
        assert_eq!(driver.sessions.owner_count(), 2);
        let saved = states.pop().unwrap();
        assert!(
            driver
                .progress_transaction_session_restore(&mut schedules, &mut states)
                .is_err()
        );
        assert_eq!(schedules.len(), 2);
        assert_eq!(states.len(), 1);
        assert_eq!(driver.sessions.owner_count(), 2);
        assert_eq!(driver.scheduler.active_len(), 0);
        states.push(saved);
        driver
            .progress_transaction_session_restore(&mut schedules, &mut states)
            .unwrap();
        assert!(schedules.is_empty() && states.is_empty());
        driver.sessions.assert_consistent();
        assert_eq!(driver.sessions.owner_count(), 0);
        assert_eq!(driver.sessions.sequence_state_count(), 2);
        assert_eq!(driver.scheduler.active_len(), 2);
    }

    #[test]
    fn pr18_session_custody_unknown_preempt_never_republishes_or_replays() {
        let mut driver = concurrent_fake_driver(MockTopKRunner::new(vec![top(7)]));
        driver.submit(request(1, &[1], 2, vec![]));
        driver.step(&mut |_| Ok(())).unwrap();
        driver.executor.runner_mut().preempt_errors = 1;
        assert!(driver.preempt_session(SessionId(1)).is_err());
        driver.sessions.assert_consistent();
        assert_eq!(driver.sessions.suspended_count(), 1);
        assert!(!driver.sessions.contains_sequence_state(&SessionId(1)));
        assert!(!driver.sessions.has_page_slot(&SessionId(1)));
        assert!(driver.restore_session(SessionId(1)).is_err());
        driver.sessions.assert_consistent();
        assert_eq!(driver.executor.runner().preempt_calls, 1);
        assert_eq!(driver.executor.runner().restore_calls, 0);
        assert_eq!(driver.sessions.suspended_count(), 1);
    }
    #[test]
    fn pr18_kv_grants_reject_partial_registration_without_losing_credit() {
        let (mut driver, source, target, page) = speculative_shared_tail_credit_driver(8);
        for pages in [vec![page], vec![KvPageId(90), KvPageId(90)], vec![]] {
            let count = if pages.is_empty() { 1 } else { pages.len() };
            let grants = (0..count)
                .map(|_| {
                    driver
                        .load_registry
                        .acquire_hard_resources(
                            900,
                            crate::scheduling::ResourceDemand::required(ExecutionPhase::Decode),
                            [PhysicalResourceClaim::new(ResourceKind::KvPage, 1)],
                        )
                        .unwrap()
                })
                .collect();
            assert!(driver.track_kv_page_grants(pages, grants).is_err());
            assert_eq!(driver.kv.grant_count(), 1);
            assert!(driver.kv.has_grant(&page));
            assert_eq!(driver.available_kv_page_credits(), 7);
        }
        driver.retire_sequence_pages(target).unwrap();
        driver.retire_sequence_pages(source).unwrap();
        assert!(driver.kv.grants_empty());
        assert_eq!(driver.available_kv_page_credits(), 8);
    }

    #[test]
    fn pr18_kv_logical_confirmation_failure_retains_grants_without_backend_replay() {
        let (mut driver, source, target, page) = speculative_shared_tail_credit_driver(8);
        driver.retire_sequence_pages(target).unwrap();
        let retirement = driver
            .page_manager
            .as_mut()
            .unwrap()
            .free_sequence_pages(source)
            .unwrap();
        let manager = driver.page_manager.take().unwrap();
        assert!(driver.release_and_confirm_retirement(retirement).is_err());
        assert_eq!(driver.kv.pending_retirement_count(), 1);
        assert!(driver.kv.has_grant(&page));
        assert_eq!(driver.available_kv_page_credits(), 7);
        assert_eq!(driver.executor.runner().released_kv_pages, vec![page]);
        driver.page_manager = Some(manager);
        driver.progress_pending_cleanups().unwrap();
        assert!(driver.kv.grants_empty());
        assert!(driver.kv.pending_retirements_empty());
        assert_eq!(driver.available_kv_page_credits(), 8);
        assert_eq!(driver.executor.runner().released_kv_pages, vec![page]);
    }
    #[test]
    fn pr18_prefix_pin_blocks_drain_until_admission_undo_ack() {
        let mut driver = prefix_cache_driver(vec![top(65)], 8, 16);
        driver.submit(request(91, &[1, 2], 0, vec![]));
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        driver.drain_finished();
        let namespace = driver.prefix_cache_namespace().unwrap();
        driver
            .stage_faults
            .extend(["prefix prepare", "prefix unpin"]);
        let mut target = request(92, &[1, 2, 3], 1, vec![]);
        target.session_id = Some(SessionId(92));
        driver.submit(target);
        assert!(driver.prepare_step().is_err());
        let Some(PendingSequenceCleanup::Admission { lease, .. }) =
            driver.sessions.cleanup(&SessionId(92))
        else {
            panic!("pin undo must survive");
        };
        assert_eq!(driver.prefix.admission_pin_count(lease), Some(1));
        assert!(driver.drain_prefix_cache().is_err());
        assert!(driver.prefix.contains_exact(namespace, &[1, 2]));
        assert!(driver.prefix.pending_cleanups_empty());
        assert!(driver.sessions.has_cleanup(&SessionId(92)));
        driver.progress_pending_cleanups().unwrap();
        assert!(!driver.sessions.has_cleanup(&SessionId(92)));
        driver.drain_prefix_cache().unwrap();
        assert!(driver.prefix.is_empty());
        assert!(driver.prefix.pending_cleanups_empty());
        assert!(driver.kv.grants_empty());
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
    }

    #[test]
    fn pr18_prefix_replacement_retains_removed_payload_before_release_retry() {
        let mut driver = prefix_cache_driver(vec![top(65)], 8, 16);
        driver.submit(request(91, &[1, 2], 2, vec![]));
        driver.step(&mut |_| Ok(())).unwrap();
        let namespace = driver.prefix_cache_namespace().unwrap();
        assert!(driver.prefix.contains_exact(namespace, &[1, 2]));
        let slot = *driver.sessions.page_slot(&SessionId(1)).unwrap();
        let model = driver
            .executor
            .fork_sequence_state_from(driver.sessions.sequence_state(&SessionId(1)).unwrap(), 2)
            .unwrap();
        let snapshot = driver
            .page_manager
            .as_mut()
            .unwrap()
            .capture_prefix_snapshot(slot, 2)
            .unwrap();
        driver
            .prefix
            .insert_and_queue_cleanup(
                namespace,
                &[1, 2],
                ResidentPrefixPayload {
                    model_state: model,
                    snapshot,
                },
                snapshot.page_count(),
            )
            .unwrap();
        assert_eq!(driver.prefix.len(), 1);
        assert_eq!(driver.prefix.pending_cleanup_count(), 1);
        let released = driver.executor.runner().released_sequence_states;
        driver.executor.runner_mut().release_failures_remaining = 1;
        assert!(driver.progress_prefix_cleanups().is_err());
        assert!(driver.prefix.contains_exact(namespace, &[1, 2]));
        assert_eq!(driver.prefix.pending_cleanup_count(), 1);
        assert_eq!(driver.executor.runner().released_sequence_states, released);
        driver.progress_prefix_cleanups().unwrap();
        assert_eq!(
            driver.executor.runner().released_sequence_states,
            released + 1
        );
        assert!(driver.prefix.pending_cleanups_empty());
        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        assert!(driver.kv.grants_empty());
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
    }

    #[test]
    fn pr18_transaction_admission_failed_compensation_retains_primary_and_claims() {
        for fail_restore in [false, true] {
            let mut driver = fork_driver();
            driver.submit(request(91, &[1, 2], 2, vec![]));
            driver.stage_faults.push_back("transaction prepare");
            if fail_restore {
                driver.stage_faults.push_back("transaction session restore");
            } else {
                driver.executor.runner_mut().kv_release_failures_remaining = 1;
            }
            let error = driver
                .step(&mut |_| panic!("rejected admission emitted output"))
                .unwrap_err();
            assert!(format!("{error:?}").contains(if fail_restore {
                "transaction session restore"
            } else {
                "KV page release failure"
            }));
            assert_eq!(driver.resident_transactions.len(), 1);
            let pending = driver.resident_transactions.values().next().unwrap();
            let ResidentTransactionPhase::BackendAbortedPendingCleanup {
                failure: Some((primary, stage)),
                ..
            } = &pending.phase
            else {
                panic!("failed compensation must retain the original admission failure");
            };
            assert_eq!(*stage, "backend prepare");
            assert!(format!("{primary:?}").contains("transaction prepare"));
            assert_eq!(pending.states.len(), 1);
            assert_eq!(pending.schedules.len(), 1);
            assert_eq!(driver.sessions.owner_count(), 1);
            assert_eq!(driver.slot_pool.active_count(), 1);
            assert_eq!(driver.admission_snapshot().request_identities_held, 1);
            assert!(driver.take_request_terminal(RequestId(91)).is_none());
            assert_eq!(driver.executor.runner().prepared_batches, 0);
            assert!(driver.executor.runner().end_intents.is_empty());
            let error = driver
                .step(&mut |_| panic!("rollback emitted output"))
                .unwrap_err();
            assert!(format!("{error:?}").contains("transaction prepare"));
            assert!(driver.resident_transactions.is_empty());
            assert_eq!(driver.sessions.owner_count(), 0);
            assert_eq!(driver.slot_pool.active_count(), 0);
            assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
            assert!(driver.kv.grants_empty());
            assert_eq!(driver.executor.runner().released_sequence_states, 1);
            assert!(driver.executor.runner().end_intents.is_empty());
            assert!(matches!(
                driver.take_request_terminal(RequestId(91)),
                Some(crate::scheduling::RequestTerminal::Failed(_))
            ));
            assert!(driver.take_request_terminal(RequestId(91)).is_none());
            driver
                .step(&mut |_| panic!("failed admission was replayed"))
                .unwrap();
            assert_eq!(driver.executor.runner().prepared_batches, 0);
            driver.shutdown(&mut |_| Ok(()), 32).unwrap();
        }
    }

    #[test]
    fn pr18_terminal_arbiter_cancel_owner_is_failure_atomic_and_released_once() {
        let mut output = OutputArbiter::default();
        let first = ExecutionTransactionId::new(900).unwrap();
        let second = ExecutionTransactionId::new(901).unwrap();
        output
            .accept_transaction_cancellation(first, RequestId(1), SessionId(1))
            .unwrap();
        output
            .accept_transaction_cancellation(first, RequestId(1), SessionId(1))
            .unwrap();
        output
            .accept_transaction_cancellation(second, RequestId(2), SessionId(2))
            .unwrap();
        assert!(
            output
                .accept_transaction_cancellation(second, RequestId(1), SessionId(1))
                .is_err()
        );
        assert!(
            output
                .accept_transaction_cancellation(first, RequestId(1), SessionId(2))
                .is_err()
        );
        assert_eq!(output.cancellation_count(), 2);
        assert_eq!(
            output.cancellation(RequestId(1)).unwrap().transaction,
            Some(first)
        );
        assert!(!output.cancellation(RequestId(1)).unwrap().ready);
        assert_eq!(
            output.release_transaction_cancellations(first),
            vec![RequestId(1)]
        );
        assert!(output.release_transaction_cancellations(first).is_empty());
        assert!(output.cancellation(RequestId(1)).unwrap().ready);
        assert!(!output.cancellation(RequestId(2)).unwrap().ready);
        for reason in [
            None,
            Some(SequenceFinishReason::MaxTokens),
            Some(SequenceFinishReason::Context),
            Some(SequenceFinishReason::Eos),
            Some(SequenceFinishReason::NoCandidate),
            Some(SequenceFinishReason::StopString),
        ] {
            assert_eq!(
                arbitrate_terminal(output.cancellation_accepted(Some(RequestId(1))), reason),
                TerminalDecision::DeferForCancellation
            );
            assert_eq!(
                arbitrate_terminal(false, reason),
                reason.map_or(TerminalDecision::Continue, TerminalDecision::Finish)
            );
        }
    }

    #[test]
    fn pr18_publish_cancel_cleanup_and_callback_retry_are_exactly_once() {
        for speculative in [false, true] {
            for cleanup_failure in [false, true] {
                let limit = if speculative { 3 } else { 1 };
                let mut driver = pr18_matrix_driver(true, speculative, 0, limit);
                driver
                    .step(&mut |_| panic!("target prefill must not deliver early"))
                    .unwrap();
                driver.executor.runner_mut().publish_progress = VecDeque::from([
                    Ok(TransactionEndProgress::Pending),
                    Ok(TransactionEndProgress::Complete),
                ]);
                assert_eq!(
                    driver
                        .step(&mut |_| panic!("uncommitted publication"))
                        .unwrap(),
                    ResidentDriverStep::Blocked
                );
                for _ in 0..2 {
                    assert_eq!(
                        driver.cancel_request(RequestId(1)).unwrap(),
                        ResidentCancelProgress::Pending
                    );
                }
                if cleanup_failure {
                    driver.executor.runner_mut().release_failures_remaining = 1;
                }
                let mut rejected = None;
                let mut nack = |event: &ResidentTokenEvent| {
                    assert!(rejected.is_none());
                    rejected = Some(event.clone());
                    Err(Error::InvalidRequest {
                        message: "publication callback NACK".into(),
                    })
                };
                assert!(driver.step(&mut nack).is_err());
                let backend_intents = driver.executor.runner().end_intents.clone();
                if rejected.is_none() {
                    assert!(cleanup_failure);
                    assert!(
                        driver
                            .step(&mut |event| {
                                rejected = Some(event.clone());
                                Err(Error::InvalidRequest {
                                    message: "publication callback NACK".into(),
                                })
                            })
                            .is_err()
                    );
                }
                assert_eq!(driver.executor.runner().end_intents, backend_intents);
                assert_eq!(driver.executor.runner().committed_batches, 2);
                assert_eq!(driver.executor.runner().rolled_back_batches, 0);
                assert!(matches!(
                    driver.take_request_terminal(RequestId(1)),
                    Some(crate::scheduling::RequestTerminal::Cancelled(_))
                ));
                assert!(driver.take_request_terminal(RequestId(1)).is_none());
                assert!(driver.drain_finished().is_empty());
                assert_eq!(driver.stats().finished_sequences, 0);
                let rejected =
                    rejected.expect("committed output remains deliverable after cancellation");
                let mut delivered = Vec::new();
                driver
                    .drive_ready_test_work(|event| {
                        delivered.push(event.clone());
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(delivered.first(), Some(&rejected));
                assert_eq!(
                    delivered
                        .iter()
                        .map(|event| event.index)
                        .collect::<Vec<_>>(),
                    (0..limit).collect::<Vec<_>>()
                );
                assert_eq!(driver.stats().emitted_tokens, limit);
                assert_eq!(driver.executor.runner().end_intents, backend_intents);
                assert!(driver.output.cancellations_empty());
                assert!(driver.output.outbox_empty());
                assert!(driver.take_request_terminal(RequestId(1)).is_none());
                driver.shutdown(&mut |_| Ok(()), 32).unwrap();
                assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
                assert!(driver.kv.grants_empty());
            }
        }
    }

    #[test]
    fn pr18_spec_admission_fault_matrix_retains_custody_without_proposal_or_backend_replay() {
        for waits in [0, 1] {
            for primary in [
                "spec admission proposals",
                "spec admission generation",
                "spec admission reserve",
            ] {
                for cleanup in ["spec admission custody", "transaction session restore"] {
                    let runner = MockTopKRunner::new(vec![top(65)])
                        .with_speculative_cycle(
                            NativeProposal {
                                token_ids: vec![66, 67],
                                confidence_logits: vec![1.0, 1.0],
                            },
                            vec![
                                TokenLogit::new(66, 9.0),
                                TokenLogit::new(67, 8.0),
                                TokenLogit::new(68, 7.0),
                            ],
                        )
                        .with_proposal_waits(vec![waits]);
                    let mut driver = speculative_driver_from_runner(runner, 1);
                    let actions = ready_speculative_decode_actions(&mut driver, &[1]);
                    let pages = driver.page_manager().unwrap().allocated_pages();
                    let intents = driver.executor.runner().end_intents.clone();
                    if waits != 0 {
                        driver.stage_faults.push_back("spec finish resume");
                    }
                    driver.stage_faults.extend([primary, cleanup]);
                    let mut result = driver.execute_speculative_decode_batch(actions, &mut |_| {
                        panic!("failed admission emitted")
                    });
                    if waits != 0 {
                        assert!(matches!(
                            result,
                            Ok(ResidentDriverStep::WaitingForModelProgress(_))
                        ));
                        let mut observed_finish_fault = false;
                        for _ in 0..8 {
                            result = driver.step(&mut |_| panic!("failed admission emitted"));
                            if let Err(error) = &result {
                                assert!(format!("{error:?}").contains("spec finish resume"));
                                observed_finish_fault = true;
                                break;
                            }
                        }
                        assert!(observed_finish_fault);
                        assert!(matches!(
                            driver.speculative_transactions.values().next(),
                            Some(PendingSpeculativeDriverCohort::Bookkeeping(_))
                        ));
                        assert_eq!(driver.sessions.owner_count(), 1);
                        result = driver.step(&mut |_| panic!("failed admission emitted"));
                    }
                    let error = result.unwrap_err();
                    assert!(
                        format!("{error:?}").contains(cleanup),
                        "{waits}/{primary}/{cleanup}: {error:?}"
                    );
                    let Some(PendingSpeculativeDriverCohort::AdmissionRollback(pending)) =
                        driver.speculative_transactions.values().next()
                    else {
                        panic!("failed admission undo must retain its custody");
                    };
                    assert!(format!("{:?}", pending.primary_error()).contains(primary));
                    assert_eq!(pending.retained_sessions(), (1, 1));
                    assert_eq!(
                        pending.registry_pending(),
                        cleanup == "spec admission custody"
                    );
                    assert_eq!(driver.sessions.owner_count(), 1);
                    assert_eq!(driver.page_manager().unwrap().allocated_pages(), pages);
                    assert_eq!(driver.slot_pool.active_count(), 1);
                    assert!(driver.take_request_terminal(RequestId(1)).is_none());
                    assert!(driver.pending_model_progresses().is_empty());
                    let calls = (
                        driver.executor.runner().native_proposal_begin_calls,
                        driver.executor.runner().native_proposal_resume_calls,
                    );
                    assert_eq!(calls, (1, waits));
                    assert_eq!(driver.executor.runner().packed_verification_calls, 0);
                    assert_eq!(driver.executor.runner().end_intents, intents);
                    // Registry ACK is consumed once even if the following restore failed.
                    if cleanup == "transaction session restore" {
                        driver.stage_faults.push_back("spec admission custody");
                    }
                    let error = driver
                        .step(&mut |_| panic!("rollback emitted"))
                        .unwrap_err();
                    assert!(format!("{error:?}").contains(primary));
                    if cleanup == "transaction session restore" {
                        assert_eq!(
                            driver.stage_faults.pop_front(),
                            Some("spec admission custody")
                        );
                    }
                    assert!(driver.speculative_transactions.is_empty());
                    assert_eq!(driver.sessions.owner_count(), 0);
                    assert_eq!(driver.executor.runner().end_intents, intents);
                    assert_eq!(
                        (
                            driver.executor.runner().native_proposal_begin_calls,
                            driver.executor.runner().native_proposal_resume_calls
                        ),
                        calls
                    );
                    assert!(matches!(
                        driver.take_request_terminal(RequestId(1)),
                        Some(crate::scheduling::RequestTerminal::Failed(_))
                    ));
                    assert!(driver.take_request_terminal(RequestId(1)).is_none());
                    assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
                    assert_eq!(driver.slot_pool.active_count(), 0);
                    assert!(driver.kv.grants_empty());
                    assert!(driver.continuations.is_empty());
                    assert!(matches!(
                        driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
                        ResidentShutdownProgress::Complete(_)
                    ));
                }
            }
        }
    }

    #[test]
    fn pr18_shutdown_admission_rollback_must_drain_before_physical_proof() {
        let mut driver = pr18_matrix_driver(true, true, 0, 3);
        driver.step(&mut |_| Ok(())).unwrap();
        driver
            .stage_faults
            .extend(["spec admission reserve", "transaction session restore"]);
        assert!(
            driver
                .step(&mut |_| panic!("rejected verification emitted"))
                .is_err()
        );
        let intents = driver.executor.runner().end_intents.clone();
        let pages = driver.page_manager().unwrap().allocated_pages();
        driver.stage_faults.push_back("transaction session restore");
        let error = driver
            .shutdown_and_close_progress(&mut |_| Ok(()))
            .unwrap_err();
        assert!(format!("{error:?}").contains("transaction session restore"));
        assert!(driver.is_shutting_down());
        assert_eq!(driver.sessions.owner_count(), 1);
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), pages);
        assert_eq!(driver.executor.runner().physical_shutdown_calls, 0);
        assert!(!driver.physical_shutdown_complete());
        let error = driver
            .shutdown_and_close_progress(&mut |_| Ok(()))
            .unwrap_err();
        assert!(format!("{error:?}").contains("spec admission reserve"));
        assert!(driver.speculative_transactions.is_empty());
        assert_eq!(driver.sessions.owner_count(), 0);
        assert_eq!(driver.executor.runner().physical_shutdown_calls, 0);
        assert_eq!(driver.executor.runner().end_intents, intents);
        assert!(matches!(
            driver.take_request_terminal(RequestId(1)),
            Some(crate::scheduling::RequestTerminal::Failed(_))
        ));
        assert!(matches!(
            driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
            ResidentShutdownProgress::Complete(_)
        ));
        assert!(matches!(
            driver
                .shutdown_and_close_progress(&mut |_| panic!("closed driver delivered twice"))
                .unwrap(),
            ResidentShutdownProgress::Complete(_)
        ));
        assert_eq!(driver.executor.runner().physical_shutdown_calls, 1);
        assert_eq!(driver.executor.runner().end_intents, intents);
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
        assert!(driver.kv.grants_empty());
    }

    #[test]
    fn pr18_shutdown_proof_requires_callback_ack_and_never_caches_failed_close() {
        for unknown in [false, true] {
            let mut driver = pr18_matrix_driver(true, true, 0, 3);
            driver.step(&mut |_| Ok(())).unwrap();
            assert!(
                driver
                    .step(&mut |_| Err(Error::InvalidRequest {
                        message: "unacknowledged delivery".into()
                    }))
                    .is_err()
            );
            let committed = driver.executor.runner().committed_batches;
            assert!(matches!(
                driver.take_request_terminal(RequestId(1)),
                Some(crate::scheduling::RequestTerminal::Finished(_))
            ));
            assert!(
                driver
                    .shutdown_and_close_progress(&mut |_| Err(Error::InvalidRequest {
                        message: "still unacknowledged".into()
                    }))
                    .is_err()
            );
            assert_eq!(driver.executor.runner().physical_shutdown_calls, 0);
            assert!(!driver.physical_shutdown_complete());
            driver.executor.runner_mut().physical_shutdown_failures = 1;
            driver.executor.runner_mut().physical_shutdown_unknown = unknown;
            let mut delivered = Vec::new();
            let error = driver
                .shutdown_and_close_progress(&mut |event| {
                    delivered.push(event.index);
                    Ok(())
                })
                .unwrap_err();
            assert!(format!("{error:?}").contains("injected physical shutdown failure"));
            assert_eq!(delivered, vec![0, 1, 2]);
            assert_eq!(driver.stats().emitted_tokens, 3);
            assert!(!driver.physical_shutdown_complete());
            assert!(driver.executor.runner().physical_owned);
            assert_eq!(driver.executor.runner().committed_batches, committed);
            assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
            for _ in 0..2 {
                let result =
                    driver.shutdown_and_close_progress(&mut |_| panic!("ACKed event replayed"));
                if unknown {
                    assert!(format!("{:?}", result.unwrap_err()).contains("remains quarantined"));
                    assert!(!driver.physical_shutdown_complete());
                    assert!(driver.executor.runner().physical_owned);
                } else {
                    assert!(matches!(
                        result.unwrap(),
                        ResidentShutdownProgress::Complete(_)
                    ));
                    assert!(driver.physical_shutdown_complete());
                    assert!(!driver.executor.runner().physical_owned);
                    assert_eq!(driver.executor.runner().physical_shutdown_calls, 2);
                }
                assert_eq!(driver.executor.runner().committed_batches, committed);
                assert_eq!(driver.stats().emitted_tokens, 3);
            }
        }
    }

    #[test]
    fn pr18_spec_admission_partial_reservation_keeps_primary_cleanup_and_credit() {
        for stale_generation in [false, true] {
            let capacity = if stale_generation { 8 } else { 2 };
            let (mut driver, source, target, source_page) =
                speculative_shared_tail_credit_driver(capacity);
            let transaction = ExecutionTransactionId::new(991).unwrap();
            let item = |slot, generation| SpeculativeVerificationItem {
                state_slot: slot,
                generation,
                proposal: &[],
                frontier: TargetFrontier {
                    position: 3,
                    top1: TokenLogit::new(9, 1.0),
                },
            };
            driver.executor.runner_mut().kv_release_failures_remaining = 1;
            let error = driver
                .reserve_speculative_pages(
                    transaction,
                    &[item(target, 0), item(source, u64::from(stale_generation))],
                )
                .unwrap_err();
            let Error::Cleanup {
                source: primary,
                cleanup,
                ..
            } = &error
            else {
                panic!("partial reservation must preserve both errors: {error:?}");
            };
            assert!(primary.to_string().contains(if stale_generation {
                "stale generation"
            } else {
                "KvPage"
            }));
            assert!(cleanup.to_string().contains("KV page release failure"));
            assert_eq!(driver.kv.pending_retirement_count(), 1);
            assert_eq!(driver.kv.grant_count(), 2);
            assert!(driver.kv.has_grant(&source_page));
            assert_eq!(
                driver
                    .load_registry
                    .resources()
                    .in_use(ResourceKind::KvPage),
                2
            );
            assert_eq!(
                driver
                    .page_manager()
                    .unwrap()
                    .block_table(source)
                    .unwrap()
                    .pages(),
                &[source_page]
            );
            assert_eq!(
                driver
                    .page_manager()
                    .unwrap()
                    .block_table(target)
                    .unwrap()
                    .pages(),
                &[source_page]
            );
            assert!(driver.executor.runner().end_intents.is_empty());
            driver.progress_pending_cleanups().unwrap();
            assert!(driver.kv.pending_retirements_empty());
            assert_eq!(driver.kv.grant_count(), 1);
            assert_eq!(
                driver
                    .load_registry
                    .resources()
                    .in_use(ResourceKind::KvPage),
                1
            );
            // The first reservation was really aborted, not merely forgotten.
            let reservations = driver
                .reserve_speculative_pages(transaction, &[item(target, 0)])
                .unwrap();
            driver
                .abort_quiesced_resident_kv(PendingResidentKv::Reserved(reservations))
                .unwrap();
            retire_test_sequence(&mut driver, target);
            retire_test_sequence(&mut driver, source);
            assert!(driver.kv.grants_empty());
            assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
            assert!(matches!(
                driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
                ResidentShutdownProgress::Complete(_)
            ));
        }
    }

    #[test]
    fn pr18_spec_admission_cancel_during_failed_undo_keeps_one_terminal() {
        let mut driver = pr18_matrix_driver(true, true, 0, 3);
        driver.step(&mut |_| Ok(())).unwrap();
        driver
            .stage_faults
            .extend(["spec admission generation", "spec admission custody"]);
        assert!(
            driver
                .step(&mut |_| panic!("rejected admission emitted"))
                .is_err()
        );
        let intents = driver.executor.runner().end_intents.clone();
        for _ in 0..2 {
            driver.stage_faults.push_back("spec admission custody");
            let error = driver.cancel_request(RequestId(1)).unwrap_err();
            assert!(format!("{error:?}").contains("spec admission custody"));
            assert!(driver.output.cancellation_accepted(Some(RequestId(1))));
            assert_eq!(driver.output.cancellation_count(), 1);
            assert_eq!(driver.sessions.owner_count(), 1);
            assert!(driver.take_request_terminal(RequestId(1)).is_none());
            assert_eq!(driver.executor.runner().end_intents, intents);
        }
        let error = driver
            .step(&mut |_| panic!("failed admission emitted"))
            .unwrap_err();
        assert!(format!("{error:?}").contains("spec admission generation"));
        assert!(driver.output.cancellations_empty());
        assert!(driver.speculative_transactions.is_empty());
        // Cancellation cannot manufacture a second terminal for a failed admission.
        assert!(matches!(
            driver.take_request_terminal(RequestId(1)),
            Some(crate::scheduling::RequestTerminal::Failed(_))
        ));
        assert!(driver.take_request_terminal(RequestId(1)).is_none());
        assert!(driver.drain_cancelled().is_empty());
        assert!(driver.drain_finished().is_empty());
        assert_eq!(driver.executor.runner().end_intents, intents);
        assert!(matches!(
            driver.shutdown_and_close_progress(&mut |_| Ok(())).unwrap(),
            ResidentShutdownProgress::Complete(_)
        ));
        assert!(driver.kv.grants_empty());
    }

    #[test]
    fn pr18_admission_missing_page_manager_keeps_reservation_until_owner_returns() {
        let (mut driver, source, target, _) = speculative_shared_tail_credit_driver(8);
        let item = SpeculativeVerificationItem {
            state_slot: target,
            generation: 0,
            proposal: &[],
            frontier: TargetFrontier {
                position: 3,
                top1: TokenLogit::new(9, 1.0),
            },
        };
        let reservations = driver
            .reserve_speculative_pages(ExecutionTransactionId::new(992).unwrap(), &[item])
            .unwrap();
        let manager = driver.page_manager.take().unwrap();
        assert!(
            driver
                .abort_quiesced_resident_kv(PendingResidentKv::Reserved(reservations))
                .is_err()
        );
        assert_eq!(driver.kv.pending_abort_count(), 1);
        assert_eq!(driver.kv.grant_count(), 2);
        assert!(driver.executor.runner().released_kv_pages.is_empty());
        driver.page_manager = Some(manager);
        driver.progress_pending_cleanups().unwrap();
        assert!(driver.kv.pending_abort_empty());
        assert_eq!(driver.kv.grant_count(), 1);
        retire_test_sequence(&mut driver, target);
        retire_test_sequence(&mut driver, source);
        assert!(driver.kv.grants_empty());
        assert_eq!(driver.page_manager().unwrap().allocated_pages(), 0);
    }
}
