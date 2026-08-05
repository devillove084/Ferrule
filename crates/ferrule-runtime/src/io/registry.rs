//! Runtime-owned global single-flight materialization registry.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::num::NonZeroU64;

use ahash::RandomState;
use snafu::Snafu;

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::io_protocol::{
    BackendId, CancellationReason, CompletionEvent, CompletionExpectation, CompletionGeneration,
    CompletionOutcome, CompletionTimestamp, ContinuationId, DependencySet, DestinationSlotId,
    DeviceId, DispatchFenceContract, FailureReason, FenceId, IoProtocolError, LoadStage,
    MappingEpoch, MaterializationKey, ModelInstanceId, OperationId, ResidencyBinding,
    ResidencyLeaseSet, RetirementReason, RetirementRecord, RetirementToken, StaleReason,
    ValidatedResidencyBinding, WaiterId,
};

use ferrule_common::materialization_io::{
    MaterializationResourceError, MaterializationResourcePlan,
};
use ferrule_model::{MaterializationPreparation, ResourceRetention};

use crate::io::fairness::{
    FairQueue, FairQueueBand, FairQueueConfig, FairQueueEntryState, FairQueueError,
};
use crate::io::ledger::{CohortId, CriticalPathLedger, CriticalPhase, LedgerError, TimeSpan};
use crate::io::provider::{MaterializationOperationReservation, RuntimeMaterializationProvider};
use crate::io::waiters::{WaiterIndex, WaiterIndexError};
use crate::scheduling::{
    ExecutionPhaseSet, PhysicalResourceBroker, PhysicalResourceClaim, PhysicalResourceError,
    PhysicalResourceGrant, ResourceDemand, ResourceKind,
};

const READ_CUSTODY: &[ResourceKind] = &[
    ResourceKind::ReadSlot,
    ResourceKind::PinnedHostBytes,
    ResourceKind::StorageReadBytes,
];
const READ_RELEASE: &[ResourceKind] = &[ResourceKind::ReadSlot, ResourceKind::StorageReadBytes];
const UPLOAD_CUSTODY: &[ResourceKind] = &[
    ResourceKind::UploadSlot,
    ResourceKind::UploadBytes,
    ResourceKind::ResidentBytes,
];
const UPLOAD_RELEASE: &[ResourceKind] = &[ResourceKind::UploadSlot, ResourceKind::UploadBytes];
const INSTALL_DESTINATION_CUSTODY: &[ResourceKind] = &[ResourceKind::ResidentBytes];
const INSTALL_CUSTODY: &[ResourceKind] =
    &[ResourceKind::InstallSlot, ResourceKind::DeviceInstallBytes];
const RESIDENCY_CLAIMS: &[ResourceKind] = &[ResourceKind::ResidentBytes];

const fn fair_queue_band(demand: ResourceDemand) -> FairQueueBand {
    match demand {
        ResourceDemand::ModelWarmup => FairQueueBand::Background,
        ResourceDemand::Prefetch(_) => FairQueueBand::Prefetch,
        ResourceDemand::Required(_) => FairQueueBand::Required,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadRequest {
    pub preparation: MaterializationPreparation,
    pub key: MaterializationKey,
    pub plan: MaterializationResourcePlan,
    pub demand: ResourceDemand,
    pub retention: ResourceRetention,
}

impl LoadRequest {
    pub const fn new(
        preparation: MaterializationPreparation,
        plan: MaterializationResourcePlan,
        demand: ResourceDemand,
        retention: ResourceRetention,
    ) -> Self {
        Self {
            key: preparation.key(),
            preparation,
            plan,
            demand,
            retention,
        }
    }
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum RegistryError {
    #[snafu(transparent)]
    Protocol { source: IoProtocolError },
    #[snafu(transparent)]
    Resources { source: PhysicalResourceError },
    #[snafu(transparent)]
    Fairness { source: FairQueueError },
    #[snafu(transparent)]
    Waiters { source: WaiterIndexError },
    #[snafu(transparent)]
    Ledger { source: LedgerError },
    #[snafu(transparent)]
    Provider { source: FailureReason },
    #[snafu(display("materialization request for {key:?} has an invalid resource plan: {source}"))]
    ResourcePlan {
        key: Box<MaterializationKey>,
        source: MaterializationResourceError,
    },
    #[snafu(display("materialization dependency {key:?} is duplicated"))]
    DuplicateDependency { key: Box<MaterializationKey> },
    #[snafu(display(
        "materialization resource plan mismatch for {key:?}: active {expected:?}, requested {requested:?}"
    ))]
    ResourcePlanMismatch {
        key: Box<MaterializationKey>,
        expected: Box<MaterializationResourcePlan>,
        requested: Box<MaterializationResourcePlan>,
    },
    #[snafu(display("materialization operation identity space is exhausted"))]
    OperationIdExhausted,
    #[snafu(display("materialization operation {operation:?} is unknown"))]
    UnknownOperation { operation: OperationId },
    #[snafu(display("materialization continuation {continuation:?} is unknown"))]
    UnknownContinuation { continuation: ContinuationId },
    #[snafu(display("materialization continuation {continuation:?} is already ready"))]
    ContinuationAlreadyReady { continuation: ContinuationId },
    #[snafu(display(
        "continuation {continuation:?} belongs to transaction {expected:?}, not {requested:?}"
    ))]
    ContinuationTransactionMismatch {
        continuation: ContinuationId,
        expected: ExecutionTransactionId,
        requested: ExecutionTransactionId,
    },
    #[snafu(display("materialization registry is shutting down"))]
    RegistryShuttingDown,
    #[snafu(display(
        "materialization shutdown is incomplete: {pending_operations} operations, {pending_completions} completions, {active_grants} grants remain"
    ))]
    ShutdownIncomplete {
        pending_operations: usize,
        pending_completions: usize,
        active_grants: usize,
    },
    #[snafu(display(
        "materialization operation {operation:?} lost its {stage:?} completion; {pending_operations} operations and {active_grants} grants remain"
    ))]
    LostCompletion {
        operation: OperationId,
        stage: LoadStage,
        pending_operations: usize,
        active_grants: usize,
    },
    #[snafu(display("published residency conflicts with materialization key {key:?}"))]
    PublishedResidencyConflict { key: Box<MaterializationKey> },
    #[snafu(display("materialization key {key:?} has no published residency"))]
    MissingResidency { key: Box<MaterializationKey> },
    #[snafu(display("resume lease for continuation {continuation:?} was already taken"))]
    ResumeLeaseAlreadyTaken { continuation: ContinuationId },
    #[snafu(display("transaction {transaction:?} already finished materialization custody"))]
    TransactionCustodyAlreadyFinished { transaction: ExecutionTransactionId },
    #[snafu(display("persistent materialization custody for {key:?} was not found"))]
    PersistentCustodyNotFound { key: Box<MaterializationKey> },
    #[snafu(display("prefetch owner {owner:?} is already registered"))]
    DuplicatePrefetch { owner: PrefetchOwner },
    #[snafu(display("prefetch owner {owner:?} is unknown"))]
    UnknownPrefetch { owner: PrefetchOwner },
    #[snafu(display("execution materialization {key:?} must have Required demand"))]
    InvalidExecutionDemand { key: Box<MaterializationKey> },

    #[snafu(display("prefetch materialization {key:?} must have non-required demand"))]
    InvalidPrefetchDemand { key: Box<MaterializationKey> },

    #[snafu(display("materialization waiter must have Required demand, got {demand:?}"))]
    InvalidWaiterDemand { demand: ResourceDemand },
    #[snafu(display(
        "cancelled materialization {key:?} still has submitted physical work to drain"
    ))]
    CancelledOperationStillDraining { key: Box<MaterializationKey> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrefetchOwner {
    ModelWarmup,
    Transaction {
        transaction: ExecutionTransactionId,
        phases: ExecutionPhaseSet,
    },
    External(NonZeroU64),
}

impl PrefetchOwner {
    pub const fn external(value: NonZeroU64) -> Self {
        Self::External(value)
    }

    pub const fn transaction(
        transaction: ExecutionTransactionId,
        phases: ExecutionPhaseSet,
    ) -> Self {
        Self::Transaction {
            transaction,
            phases,
        }
    }

    pub fn belongs_to_transaction(self, expected: ExecutionTransactionId) -> bool {
        matches!(
            self,
            Self::Transaction { transaction, .. } if transaction == expected
        )
    }
}

#[derive(Debug)]
struct PrefetchState {
    demand: ResourceDemand,
    operations: BTreeSet<OperationId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchReport {
    pub created: Vec<OperationId>,
    pub joined: Vec<OperationId>,
    pub already_resident: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachReport {
    pub created: Vec<OperationId>,
    pub joined: Vec<OperationId>,
    pub already_resident: usize,
    pub continuation_ready: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContinuationFailure {
    Failed(FailureReason),
    Stale(StaleReason),
    Cancelled(CancellationReason),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedContinuation {
    pub continuation: ContinuationId,
    pub failure: ContinuationFailure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionRejectionReason {
    UnknownOperation,
    RetiredOperation,
    UnexpectedOwnerStage(LoadStage),
    Protocol(IoProtocolError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionRejection {
    pub event: CompletionEvent,
    pub reason: CompletionRejectionReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionDisposition {
    Applied {
        operation: OperationId,
        key: MaterializationKey,
        stage: LoadStage,
    },
    Rejected(CompletionRejectionReason),
    QueueEmpty,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RegistryStats {
    pub operations_created: u64,
    pub single_flight_joins: u64,
    pub physical_completions: u64,
    pub rejected_completions: u64,
    pub publications: u64,
    pub retirements: u64,
    pub cancellations_requested: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadActionKind {
    Reserve,
    SubmitRead,
    SubmitUpload,
    PollInstall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LoadAction {
    operation: OperationId,
    kind: LoadActionKind,
}

/// Owner-side active load operation. Physical reservation, stage grants, and
/// retirement authority are deliberately non-cloneable.
#[derive(Debug)]
pub struct LoadOp {
    operation: OperationId,
    key: MaterializationKey,
    expected_binding: ResidencyBinding,
    plan: MaterializationResourcePlan,
    demand: ResourceDemand,
    execution_required: bool,
    stage: LoadStage,
    reservation: Option<MaterializationOperationReservation>,
    logical_grant: Option<PhysicalResourceGrant>,
    read_grant: Option<PhysicalResourceGrant>,
    upload_grant: Option<PhysicalResourceGrant>,
    install_grant: Option<PhysicalResourceGrant>,
    retirement: Option<RetirementToken>,
    cancellation: Option<OperationCancellation>,
    install_submitted: bool,
    replaced: Option<MaterializationKey>,
    stage_started_ns: u64,
}

impl LoadOp {
    pub const fn operation(&self) -> OperationId {
        self.operation
    }

    pub const fn key(&self) -> MaterializationKey {
        self.key
    }

    pub const fn plan(&self) -> MaterializationResourcePlan {
        self.plan
    }

    pub const fn demand(&self) -> ResourceDemand {
        self.demand
    }

    pub const fn stage(&self) -> LoadStage {
        self.stage
    }

    pub const fn cancellation_requested(&self) -> bool {
        self.cancellation.is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionLeaseState {
    Held,
    ReleasePending,
    Released,
}

#[derive(Debug)]
struct ResidentEntry {
    binding: ResidencyBinding,
    grant: PhysicalResourceGrant,
    lease: ExecutionLeaseState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ResidencyLocation {
    model: ModelInstanceId,
    backend: BackendId,
    device: DeviceId,
    slot: DestinationSlotId,
}

impl From<ResidencyBinding> for ResidencyLocation {
    fn from(binding: ResidencyBinding) -> Self {
        Self {
            model: binding.model,
            backend: binding.backend,
            device: binding.device,
            slot: binding.slot,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PublicationPreflight {
    Ready {
        replaced: Option<MaterializationKey>,
    },
    Blocked,
}

#[derive(Debug, Clone)]
enum OperationCancellation {
    Pending(CancellationReason),
    Submitted {
        stage: LoadStage,
        reason: CancellationReason,
    },
}

impl OperationCancellation {
    fn reason(&self) -> &CancellationReason {
        match self {
            Self::Pending(reason) | Self::Submitted { reason, .. } => reason,
        }
    }
}

#[derive(Debug)]
enum PostCompletion {
    Keep,
    Publish,
    Terminal {
        retirement: RetirementReason,
        failure: Option<ContinuationFailure>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShutdownReport {
    pub drained: bool,
    pub pending_operations: usize,
    pub pending_completions: usize,
    pub active_grants: usize,
}

/// Runtime-owned hard-credit guard paired with the exact lease set passed to one
/// model resume. The lease set can be taken once; the guard must then be returned
/// to [`LoadRegistry::finish_resume`] on every success or error path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeDisposition {
    Consumed,
    StillActive,
}

#[derive(Debug)]
pub struct ResumeLease {
    continuation: ContinuationId,
    leases: Option<ResidencyLeaseSet>,
    grant: Option<PhysicalResourceGrant>,
}

impl ResumeLease {
    pub const fn continuation(&self) -> ContinuationId {
        self.continuation
    }

    pub fn take(&mut self) -> Result<ResidencyLeaseSet, RegistryError> {
        self.leases
            .take()
            .ok_or(RegistryError::ResumeLeaseAlreadyTaken {
                continuation: self.continuation,
            })
    }
}

/// Per-continuation runtime state: exact execution demand, grants, and ready admission.
/// Consolidated from three parallel BTreeMaps.
#[derive(Debug)]
struct ContinuationState {
    transaction: ExecutionTransactionId,
    grant: PhysicalResourceGrant,
    demand: ResourceDemand,
    ready_grant: Option<PhysicalResourceGrant>,
    owns_stage_custody: bool,
}

/// Per-transaction tracking: operations and cohorts.
/// Consolidated from two parallel BTreeMaps.
#[derive(Debug, Default)]
struct TransactionState {
    operations: BTreeSet<OperationId>,
    cohorts: BTreeSet<CohortId>,
    owns_materialization_custody: bool,
    materialization_custody_finished: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum MaterializationCustodyOwner {
    Stage(ContinuationId),
    Transaction(ExecutionTransactionId),
    Persistent(MaterializationKey),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionCustodyOutcome {
    Committed { started_ns: u64, finished_ns: u64 },
    RolledBack,
    Cancelled,
}

/// Runtime-wide authoritative owner of materialization, waiters, credits,
/// completion validation, publication, and retirement.
#[derive(Debug)]
pub struct LoadRegistry<P: RuntimeMaterializationProvider> {
    provider: P,
    resources: PhysicalResourceBroker,
    key_to_operation: HashMap<MaterializationKey, OperationId, RandomState>,
    operation_to_key: HashMap<OperationId, MaterializationKey, RandomState>,
    operations: HashMap<OperationId, LoadOp, RandomState>,
    retirements: HashMap<OperationId, RetirementRecord, RandomState>,
    stage_history: HashMap<OperationId, Vec<LoadStage>, RandomState>,
    residencies: HashMap<MaterializationKey, ResidentEntry, RandomState>,
    residency_locations: HashMap<ResidencyLocation, MaterializationKey, RandomState>,
    pending_publications: VecDeque<OperationId>,
    queued_publications: BTreeSet<OperationId>,
    pending_cleanups: VecDeque<OperationId>,
    queued_cleanups: BTreeSet<OperationId>,
    pending_lease_releases: BTreeSet<MaterializationKey>,
    waiters: WaiterIndex,
    prefetches: HashMap<PrefetchOwner, PrefetchState, RandomState>,
    operation_prefetches: HashMap<OperationId, BTreeSet<PrefetchOwner>, RandomState>,
    prefetch_failures: HashMap<PrefetchOwner, RetirementReason, RandomState>,
    waiter_grants: HashMap<WaiterId, PhysicalResourceGrant, RandomState>,
    waiter_demands: HashMap<WaiterId, ResourceDemand, RandomState>,
    ready_waiters: HashMap<ContinuationId, BTreeSet<WaiterId>, RandomState>,
    continuations: HashMap<ContinuationId, ContinuationState, RandomState>,
    custody: HashMap<MaterializationKey, BTreeSet<MaterializationCustodyOwner>, RandomState>,
    owner_custody: HashMap<MaterializationCustodyOwner, BTreeSet<MaterializationKey>, RandomState>,
    failed_continuations: VecDeque<FailedContinuation>,
    failed_set: BTreeSet<ContinuationId>,
    runnable: FairQueue<LoadAction>,
    completions: VecDeque<CompletionEvent>,
    rejected_completions: Vec<CompletionRejection>,
    wait_started: HashMap<(WaiterId, OperationId), u64, RandomState>,
    transactions: HashMap<ExecutionTransactionId, TransactionState, RandomState>,
    ledger: CriticalPathLedger,
    next_operation: u64,
    next_dispatch: u64,
    stats: RegistryStats,
    shutting_down: bool,
}

impl<P: RuntimeMaterializationProvider> LoadRegistry<P> {
    pub fn new(
        provider: P,
        resources: PhysicalResourceBroker,
        fairness: FairQueueConfig,
    ) -> Result<Self, RegistryError> {
        Ok(Self {
            provider,
            resources,
            key_to_operation: HashMap::default(),
            operation_to_key: HashMap::default(),
            operations: HashMap::default(),
            retirements: HashMap::default(),
            stage_history: HashMap::default(),
            residencies: HashMap::default(),
            residency_locations: HashMap::default(),
            pending_publications: VecDeque::new(),
            queued_publications: BTreeSet::new(),
            pending_cleanups: VecDeque::new(),
            queued_cleanups: BTreeSet::new(),
            pending_lease_releases: BTreeSet::new(),
            waiters: WaiterIndex::new(),
            prefetches: HashMap::default(),
            operation_prefetches: HashMap::default(),
            prefetch_failures: HashMap::default(),
            waiter_grants: HashMap::default(),
            waiter_demands: HashMap::default(),
            ready_waiters: HashMap::default(),
            continuations: HashMap::default(),
            custody: HashMap::default(),
            owner_custody: HashMap::default(),
            failed_continuations: VecDeque::new(),
            failed_set: BTreeSet::new(),
            runnable: FairQueue::new(fairness)?,
            completions: VecDeque::new(),
            rejected_completions: Vec::new(),
            wait_started: HashMap::default(),
            transactions: HashMap::default(),
            ledger: CriticalPathLedger::new(),
            next_operation: 1,
            next_dispatch: 1,
            stats: RegistryStats::default(),
            shutting_down: false,
        })
    }

    #[cfg(test)]
    pub fn with_testing_resources(provider: P) -> Result<Self, RegistryError> {
        Self::new(
            provider,
            PhysicalResourceBroker::testing_default(),
            FairQueueConfig::default(),
        )
    }

    pub const fn stats(&self) -> RegistryStats {
        self.stats
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }

    pub fn provider_mut(&mut self) -> &mut P {
        &mut self.provider
    }

    pub(crate) fn resources_mut(&mut self) -> &mut PhysicalResourceBroker {
        &mut self.resources
    }

    pub fn resources(&self) -> &PhysicalResourceBroker {
        &self.resources
    }

    pub fn acquire_hard_resources(
        &mut self,
        owner: u64,
        demand: ResourceDemand,
        claims: impl IntoIterator<Item = PhysicalResourceClaim>,
    ) -> Result<PhysicalResourceGrant, RegistryError> {
        Ok(self.resources.acquire(owner, demand, claims)?)
    }

    pub fn release_hard_resources(
        &mut self,
        grant: &mut PhysicalResourceGrant,
    ) -> Result<(), RegistryError> {
        Ok(self.resources.release_all_held(grant)?)
    }

    pub fn ledger(&self) -> &CriticalPathLedger {
        &self.ledger
    }

    pub fn ledger_mut(&mut self) -> &mut CriticalPathLedger {
        &mut self.ledger
    }

    pub fn waiters(&self) -> &WaiterIndex {
        &self.waiters
    }

    pub fn active_operations(&self) -> usize {
        self.operations.len()
    }

    pub fn active_prefetches(&self) -> usize {
        self.prefetches.len()
    }

    pub fn prefetch_active(&self, prefetch: PrefetchOwner) -> bool {
        self.prefetches.contains_key(&prefetch)
    }

    pub fn prefetch_failed(&self, prefetch: PrefetchOwner) -> bool {
        self.prefetch_failures.contains_key(&prefetch)
    }

    pub fn take_prefetch_failure(&mut self, prefetch: PrefetchOwner) -> Option<RetirementReason> {
        self.prefetch_failures.remove(&prefetch)
    }

    pub fn prefetch_operations(
        &self,
        prefetch: PrefetchOwner,
    ) -> impl Iterator<Item = OperationId> + '_ {
        self.prefetches
            .get(&prefetch)
            .into_iter()
            .flat_map(|state| state.operations.iter().copied())
    }

    pub fn operation_has_prefetch_owner(&self, operation: OperationId) -> bool {
        self.operation_prefetches
            .get(&operation)
            .is_some_and(|owners| !owners.is_empty())
    }

    pub fn runnable_actions(&self) -> usize {
        self.runnable.len()
    }

    pub fn pending_completions(&self) -> usize {
        self.completions.len()
    }

    pub fn resident_entries(&self) -> usize {
        self.residencies.len()
    }

    pub fn resident_bytes(&self) -> u64 {
        self.residencies
            .values()
            .flat_map(|resident| resident.grant.claims())
            .filter(|claim| claim.kind == ResourceKind::ResidentBytes)
            .fold(0u64, |total, claim| total.saturating_add(claim.amount))
    }

    pub fn stage_counts(&self) -> impl Iterator<Item = (LoadStage, usize)> + '_ {
        LoadStage::ALL.into_iter().map(|stage| {
            let count = self
                .operations
                .values()
                .filter(|operation| operation.stage == stage)
                .count();
            (stage, count)
        })
    }

    pub const fn is_shutting_down(&self) -> bool {
        self.shutting_down
    }

    pub fn pending_physical_operations(&self) -> usize {
        self.operations
            .values()
            .filter(|operation| {
                matches!(
                    operation.stage,
                    LoadStage::ReadSubmitted | LoadStage::UploadSubmitted
                ) || (operation.stage == LoadStage::Installing && operation.install_submitted)
            })
            .count()
    }

    pub fn operation_for_key(&self, key: MaterializationKey) -> Option<OperationId> {
        self.key_to_operation.get(&key).copied()
    }

    pub fn key_for_operation(&self, operation: OperationId) -> Option<MaterializationKey> {
        self.operation_to_key.get(&operation).copied()
    }

    pub fn operation(&self, operation: OperationId) -> Option<&LoadOp> {
        self.operations.get(&operation)
    }

    pub fn retirement(&self, operation: OperationId) -> Option<&RetirementRecord> {
        self.retirements.get(&operation)
    }

    pub fn stage_history(&self, operation: OperationId) -> Option<&[LoadStage]> {
        self.stage_history.get(&operation).map(Vec::as_slice)
    }

    pub fn residency_binding(
        &self,
        key: MaterializationKey,
    ) -> Option<ferrule_common::io_protocol::ResidencyBinding> {
        self.residencies.get(&key).map(|entry| entry.binding)
    }

    pub fn validated_residency(
        &self,
        key: MaterializationKey,
    ) -> Result<ValidatedResidencyBinding, RegistryError> {
        let binding = self
            .residency_binding(key)
            .ok_or_else(|| RegistryError::MissingResidency { key: Box::new(key) })?;
        Ok(ValidatedResidencyBinding::new(key, binding)?)
    }

    pub fn prepare_execution_request(
        &self,
        key: MaterializationKey,
        demand: ResourceDemand,
        retention: ResourceRetention,
    ) -> Result<LoadRequest, RegistryError> {
        if !demand.is_required() {
            return Err(RegistryError::InvalidExecutionDemand { key: Box::new(key) });
        }
        self.prepare_request(key, demand, retention)
    }

    pub fn prepare_prefetch_request(
        &self,
        key: MaterializationKey,
        demand: ResourceDemand,
    ) -> Result<LoadRequest, RegistryError> {
        if !demand.is_prefetch() {
            return Err(RegistryError::InvalidPrefetchDemand { key: Box::new(key) });
        }
        self.prepare_request(key, demand, ResourceRetention::ThroughStage)
    }

    fn prepare_request(
        &self,
        key: MaterializationKey,
        demand: ResourceDemand,
        retention: ResourceRetention,
    ) -> Result<LoadRequest, RegistryError> {
        let preparation = self.provider.preparation(key)?;
        if preparation.key() != key {
            return Err(RegistryError::Provider {
                source: FailureReason::ContractViolation {
                    message: "materialization provider returned a different prepared key".into(),
                },
            });
        }
        let plan = self
            .provider
            .materialization_plan(key)?
            .validate()
            .map_err(|source| RegistryError::Provider {
                source: FailureReason::Resources { source },
            })?;
        Ok(LoadRequest::new(preparation, plan, demand, retention))
    }

    fn adopt_prepared_residency(
        &mut self,
        request: LoadRequest,
        binding: ferrule_common::io_protocol::ResidencyBinding,
        lease: ExecutionLeaseState,
    ) -> Result<(), RegistryError> {
        let validated = ValidatedResidencyBinding::new(request.key, binding)?;
        if let Some(existing) = self.residencies.get(&request.key) {
            return if existing.binding == validated.binding() {
                Ok(())
            } else {
                Err(RegistryError::PublishedResidencyConflict {
                    key: Box::new(request.key),
                })
            };
        }
        if self.key_to_operation.contains_key(&request.key) {
            return Err(RegistryError::PublishedResidencyConflict {
                key: Box::new(request.key),
            });
        }
        let location = ResidencyLocation::from(validated.binding());
        if let Some(existing) = self.residency_locations.get(&location) {
            return Err(RegistryError::PublishedResidencyConflict {
                key: Box::new(*existing),
            });
        }
        let grant = self.resources.acquire(
            request.key.destination_generation().get(),
            request.demand,
            [PhysicalResourceClaim::new(
                ResourceKind::ResidentBytes,
                request.plan.resident_bytes,
            )],
        )?;
        self.residencies.insert(
            request.key,
            ResidentEntry {
                binding: validated.binding(),
                grant,
                lease,
            },
        );
        self.residency_locations.insert(location, request.key);
        Ok(())
    }

    pub fn discard_preparations(
        &mut self,
        keys: impl IntoIterator<Item = MaterializationKey>,
    ) -> Result<(), RegistryError> {
        for key in keys {
            if self.key_to_operation.contains_key(&key) {
                continue;
            }
            if let Some(resident) = self.residencies.get(&key)
                && resident.lease != ExecutionLeaseState::Released
            {
                continue;
            }
            self.provider.discard_preparation(key)?;
        }
        Ok(())
    }

    pub fn rejected_completions(&self) -> &[CompletionRejection] {
        &self.rejected_completions
    }

    fn acquire_custody(&mut self, key: MaterializationKey, owner: MaterializationCustodyOwner) {
        if self.custody.entry(key).or_default().insert(owner) {
            self.owner_custody.entry(owner).or_default().insert(key);
        }
        if let Some(resident) = self.residencies.get_mut(&key) {
            resident.lease = ExecutionLeaseState::Held;
            self.pending_lease_releases.remove(&key);
        }
    }

    fn release_custody_owner(&mut self, owner: MaterializationCustodyOwner) -> bool {
        let Some(keys) = self.owner_custody.remove(&owner) else {
            return false;
        };
        for key in keys {
            let last_owner = if let Some(owners) = self.custody.get_mut(&key) {
                owners.remove(&owner);
                owners.is_empty()
            } else {
                false
            };
            if !last_owner {
                continue;
            }
            self.custody.remove(&key);
            if let Some(resident) = self.residencies.get_mut(&key)
                && resident.lease == ExecutionLeaseState::Held
            {
                resident.lease = ExecutionLeaseState::ReleasePending;
                self.pending_lease_releases.insert(key);
            }
        }
        true
    }

    fn has_custody(&self, key: MaterializationKey) -> bool {
        self.custody
            .get(&key)
            .is_some_and(|owners| !owners.is_empty())
    }

    pub fn retire_persistent_custody(
        &mut self,
        key: MaterializationKey,
    ) -> Result<(), RegistryError> {
        if !self.release_custody_owner(MaterializationCustodyOwner::Persistent(key)) {
            return Err(RegistryError::PersistentCustodyNotFound { key: Box::new(key) });
        }
        Ok(())
    }

    pub fn prefetch(
        &mut self,
        owner: PrefetchOwner,
        requests: impl IntoIterator<Item = LoadRequest>,
        now_ns: u64,
    ) -> Result<PrefetchReport, RegistryError> {
        if self.shutting_down {
            return Err(RegistryError::RegistryShuttingDown);
        }
        if self.prefetches.contains_key(&owner) || self.prefetch_failures.contains_key(&owner) {
            return Err(RegistryError::DuplicatePrefetch { owner });
        }

        let mut requests = requests.into_iter().collect::<Vec<_>>();
        requests.sort_unstable_by_key(|request| request.key);
        for request in &requests {
            if !request.demand.is_prefetch() {
                return Err(RegistryError::InvalidPrefetchDemand {
                    key: Box::new(request.key),
                });
            }
            self.validate_request(request)?;
        }
        let demand = requests
            .iter()
            .fold(ResourceDemand::ModelWarmup, |demand, request| {
                demand.merge(request.demand)
            });
        if let Some(duplicate) = requests
            .windows(2)
            .find(|window| window[0].key == window[1].key)
        {
            return Err(RegistryError::DuplicateDependency {
                key: Box::new(duplicate[0].key),
            });
        }

        let mut joined = Vec::new();
        let mut resident_preparations = Vec::new();
        let mut new_requests = Vec::new();
        let mut already_resident = 0;
        for request in &requests {
            if let Some(resident) = self.residencies.get(&request.key) {
                if resident.binding != request.preparation.binding()
                    || !matches!(request.preparation, MaterializationPreparation::Resident(_))
                {
                    return Err(RegistryError::PublishedResidencyConflict {
                        key: Box::new(request.key),
                    });
                }
                already_resident += 1;
                continue;
            }
            if let Some(operation) = self.key_to_operation.get(&request.key).copied() {
                let active = self
                    .operations
                    .get(&operation)
                    .expect("active key index must point to an operation");
                self.validate_join(active, request)?;
                if active.cancellation.as_ref().is_some_and(|cancellation| {
                    matches!(cancellation, OperationCancellation::Submitted { .. })
                }) {
                    return Err(RegistryError::CancelledOperationStillDraining {
                        key: Box::new(request.key),
                    });
                }
                joined.push(operation);
                continue;
            }
            match request.preparation {
                MaterializationPreparation::Resident(_) => {
                    resident_preparations.push(*request);
                    already_resident += 1;
                }
                MaterializationPreparation::Transfer(_) => new_requests.push(*request),
            }
        }

        let resident_bytes = resident_preparations
            .iter()
            .try_fold(0_u64, |total, request| {
                total.checked_add(request.plan.resident_bytes).ok_or(
                    PhysicalResourceError::ClaimOverflow {
                        kind: ResourceKind::ResidentBytes,
                    },
                )
            })?;
        if resident_bytes != 0 {
            self.resources.can_acquire(
                demand,
                [PhysicalResourceClaim::new(
                    ResourceKind::ResidentBytes,
                    resident_bytes,
                )],
            )?;
        }

        let mut adopted = Vec::new();
        for request in &resident_preparations {
            if let Err(error) = self.adopt_prepared_residency(
                *request,
                request.preparation.binding(),
                ExecutionLeaseState::Released,
            ) {
                for key in adopted.into_iter().rev() {
                    self.rollback_adopted_residency(key)?;
                }
                return Err(error);
            }
            adopted.push(request.key);
        }

        let mut created = Vec::new();
        for request in new_requests {
            match self.create_operation(request, now_ns) {
                Ok(operation) => created.push(operation),
                Err(error) => {
                    while let Some(operation) = created.pop() {
                        self.rollback_reserved_operation(operation, now_ns)?;
                    }
                    for key in adopted.into_iter().rev() {
                        self.rollback_adopted_residency(key)?;
                    }
                    return Err(error);
                }
            }
        }

        let operations = joined
            .iter()
            .chain(&created)
            .copied()
            .collect::<BTreeSet<_>>();
        if !operations.is_empty() {
            for operation in &operations {
                self.operation_prefetches
                    .entry(*operation)
                    .or_default()
                    .insert(owner);
                if let Some(active) = self.operations.get_mut(operation)
                    && matches!(active.cancellation, Some(OperationCancellation::Pending(_)))
                {
                    active.cancellation = None;
                }
            }
            self.prefetches.insert(
                owner,
                PrefetchState {
                    demand,
                    operations: operations.clone(),
                },
            );
            for operation in operations {
                self.refresh_operation_demand(operation)?;
            }
        }
        self.stats.operations_created = self
            .stats
            .operations_created
            .saturating_add(created.len() as u64);
        self.stats.single_flight_joins = self
            .stats
            .single_flight_joins
            .saturating_add(joined.len() as u64);
        Ok(PrefetchReport {
            created,
            joined,
            already_resident,
        })
    }

    pub fn cancel_prefetch(
        &mut self,
        owner: PrefetchOwner,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        self.release_prefetch(owner, CancellationReason::PrefetchCancelled, now_ns)
    }

    pub fn attach_waiter(
        &mut self,
        waiter: WaiterId,
        demand: ResourceDemand,
        requests: impl IntoIterator<Item = LoadRequest>,
        now_ns: u64,
    ) -> Result<AttachReport, RegistryError> {
        if !demand.is_required() {
            return Err(RegistryError::InvalidWaiterDemand { demand });
        }
        if self.shutting_down {
            return Err(RegistryError::RegistryShuttingDown);
        }
        waiter.validate()?;
        if self.waiters.has_seen_waiter(waiter) {
            return Err(WaiterIndexError::DuplicateWaiter { waiter }.into());
        }
        let continuation = waiter.continuation();
        if self
            .transactions
            .get(&waiter.transaction())
            .is_some_and(|state| state.materialization_custody_finished)
        {
            return Err(RegistryError::TransactionCustodyAlreadyFinished {
                transaction: waiter.transaction(),
            });
        }
        if let Some(state) = self.continuations.get(&continuation) {
            if state.transaction != waiter.transaction() {
                return Err(RegistryError::ContinuationTransactionMismatch {
                    continuation,
                    expected: state.transaction,
                    requested: waiter.transaction(),
                });
            }
            if self.waiters.unresolved_for(continuation).is_none() {
                return Err(RegistryError::ContinuationAlreadyReady { continuation });
            }
        }

        let mut requests: Vec<_> = requests.into_iter().collect();
        requests.sort_unstable_by_key(|request| request.key);
        for request in &requests {
            if !request.demand.is_required() {
                return Err(RegistryError::InvalidExecutionDemand {
                    key: Box::new(request.key),
                });
            }
            self.validate_request(request)?;
        }
        if let Some(duplicate) = requests
            .windows(2)
            .find(|window| window[0].key == window[1].key)
        {
            return Err(RegistryError::DuplicateDependency {
                key: Box::new(duplicate[0].key),
            });
        }
        let demand = requests
            .iter()
            .fold(demand, |demand, request| demand.merge(request.demand));

        let mut joined = Vec::new();
        let mut resident_preparations = Vec::new();
        let mut new_requests = Vec::new();
        let mut already_resident = 0;
        for request in &requests {
            if let Some(resident) = self.residencies.get(&request.key) {
                if resident.binding != request.preparation.binding()
                    || !matches!(request.preparation, MaterializationPreparation::Resident(_))
                {
                    return Err(RegistryError::PublishedResidencyConflict {
                        key: Box::new(request.key),
                    });
                }
                already_resident += 1;
                continue;
            }
            if let Some(operation) = self.key_to_operation.get(&request.key).copied() {
                let active = self
                    .operations
                    .get(&operation)
                    .expect("active key index must point to an operation");
                self.validate_join(active, request)?;
                if active.cancellation.as_ref().is_some_and(|cancellation| {
                    matches!(cancellation, OperationCancellation::Submitted { .. })
                }) {
                    return Err(RegistryError::CancelledOperationStillDraining {
                        key: Box::new(request.key),
                    });
                }
                joined.push(operation);
            } else {
                match request.preparation {
                    MaterializationPreparation::Resident(_) => {
                        resident_preparations.push(*request);
                        already_resident += 1;
                    }
                    MaterializationPreparation::Transfer(_) => new_requests.push(*request),
                }
            }
        }

        let continuation_is_new = !self.continuations.contains_key(&continuation);
        let mut preflight = vec![PhysicalResourceClaim::new(ResourceKind::Waiter, 1)];
        if continuation_is_new {
            preflight.push(PhysicalResourceClaim::new(ResourceKind::Continuation, 1));
        }
        if !resident_preparations.is_empty() {
            let resident_bytes =
                resident_preparations
                    .iter()
                    .try_fold(0_u64, |total, request| {
                        total.checked_add(request.plan.resident_bytes).ok_or(
                            PhysicalResourceError::ClaimOverflow {
                                kind: ResourceKind::ResidentBytes,
                            },
                        )
                    })?;
            preflight.push(PhysicalResourceClaim::new(
                ResourceKind::ResidentBytes,
                resident_bytes,
            ));
        }
        self.resources.can_acquire(demand, preflight)?;

        let mut adopted = Vec::new();
        for request in &resident_preparations {
            if let Err(error) = self.adopt_prepared_residency(
                *request,
                request.preparation.binding(),
                ExecutionLeaseState::Held,
            ) {
                for key in adopted.into_iter().rev() {
                    self.rollback_adopted_residency(key)?;
                }
                return Err(error);
            }
            adopted.push(request.key);
        }
        let mut created = Vec::new();
        for request in new_requests {
            match self.create_operation(request, now_ns) {
                Ok(operation) => created.push(operation),
                Err(error) => {
                    while let Some(operation) = created.pop() {
                        self.rollback_reserved_operation(operation, now_ns)?;
                    }
                    for key in adopted.into_iter().rev() {
                        self.rollback_adopted_residency(key)?;
                    }
                    return Err(error);
                }
            }
        }

        if continuation_is_new {
            let grant = self.resources.acquire(
                continuation.get(),
                demand,
                [PhysicalResourceClaim::new(ResourceKind::Continuation, 1)],
            )?;
            self.continuations.insert(
                continuation,
                ContinuationState {
                    transaction: waiter.transaction(),
                    grant,
                    demand,
                    ready_grant: None,
                    owns_stage_custody: requests
                        .iter()
                        .any(|request| request.retention == ResourceRetention::ThroughStage),
                },
            );
        } else {
            let state = self
                .continuations
                .get_mut(&continuation)
                .expect("continuation state is created together with grant");
            let merged = state.demand.merge(demand);
            if merged != state.demand {
                self.resources.reclassify(&state.grant, merged)?;
                state.demand = merged;
            }
            state.owns_stage_custody |= requests
                .iter()
                .any(|request| request.retention == ResourceRetention::ThroughStage);
        }
        let waiter_grant = self.resources.acquire(
            continuation.get(),
            demand,
            [PhysicalResourceClaim::new(ResourceKind::Waiter, 1)],
        )?;

        let operations: Vec<_> = joined.iter().chain(&created).copied().collect();
        let joined_keys = joined
            .iter()
            .map(|operation| {
                self.operations
                    .get(operation)
                    .expect("joined operation remains active")
                    .key
            })
            .collect::<BTreeSet<_>>();
        let mut waiter_grant = waiter_grant;
        let mut promoted_joined = Vec::new();
        for request in &requests {
            let promotion = match self.provider.promote_to_execution(request.key) {
                Ok(promotion) => promotion,
                Err(error) => {
                    for key in promoted_joined.into_iter().rev() {
                        self.provider.release_execution_lease(key)?;
                    }
                    self.resources.release_all_held(&mut waiter_grant)?;
                    if continuation_is_new {
                        self.release_continuation(continuation)?;
                    }
                    while let Some(operation) = created.pop() {
                        self.rollback_reserved_operation(operation, now_ns)?;
                    }
                    for key in adopted.into_iter().rev() {
                        self.rollback_adopted_residency(key)?;
                    }
                    return Err(RegistryError::Provider { source: error });
                }
            };
            let preparation = promotion.preparation();
            if preparation.key() != request.key
                || preparation.binding() != request.preparation.binding()
            {
                if promotion.changed() && joined_keys.contains(&request.key) {
                    self.provider.release_execution_lease(request.key)?;
                }
                for key in promoted_joined.into_iter().rev() {
                    self.provider.release_execution_lease(key)?;
                }
                self.resources.release_all_held(&mut waiter_grant)?;
                if continuation_is_new {
                    self.release_continuation(continuation)?;
                }
                while let Some(operation) = created.pop() {
                    self.rollback_reserved_operation(operation, now_ns)?;
                }
                for key in adopted.into_iter().rev() {
                    self.rollback_adopted_residency(key)?;
                }
                return Err(RegistryError::PublishedResidencyConflict {
                    key: Box::new(request.key),
                });
            }
            if promotion.changed() && joined_keys.contains(&request.key) {
                promoted_joined.push(request.key);
            }
        }
        for operation in &joined {
            self.promote_operation_for_execution(*operation, demand)?;
        }

        let continuation_ready = self.waiters.register(waiter, operations.clone())?;
        self.waiter_grants.insert(waiter, waiter_grant);
        self.waiter_demands.insert(waiter, demand);
        self.transactions
            .entry(waiter.transaction())
            .or_default()
            .operations
            .extend(operations.iter().copied());
        self.transactions
            .entry(waiter.transaction())
            .or_default()
            .cohorts
            .insert(CohortId::new(continuation.get()));
        if continuation_ready {
            self.ready_waiters
                .entry(continuation)
                .or_default()
                .insert(waiter);
        } else {
            for operation in &operations {
                self.wait_started.insert((waiter, *operation), now_ns);
            }
        }

        self.stats.operations_created = self
            .stats
            .operations_created
            .saturating_add(created.len() as u64);
        self.stats.single_flight_joins = self
            .stats
            .single_flight_joins
            .saturating_add(joined.len() as u64);
        for request in &requests {
            let owner = match request.retention {
                ResourceRetention::ThroughStage => MaterializationCustodyOwner::Stage(continuation),
                ResourceRetention::ThroughTransaction => {
                    self.transactions
                        .entry(waiter.transaction())
                        .or_default()
                        .owns_materialization_custody = true;
                    MaterializationCustodyOwner::Transaction(waiter.transaction())
                }
                ResourceRetention::Persistent => {
                    MaterializationCustodyOwner::Persistent(request.key)
                }
            };
            self.acquire_custody(request.key, owner);
        }
        Ok(AttachReport {
            created,
            joined,
            already_resident,
            continuation_ready,
        })
    }

    pub fn detach_waiter(
        &mut self,
        waiter: WaiterId,
        reason: CancellationReason,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        let operations = self
            .waiters
            .loads_for(waiter)
            .cloned()
            .ok_or(WaiterIndexError::UnknownWaiter { waiter })?;
        for operation in &operations {
            self.close_wait(waiter, *operation, now_ns)?;
        }
        let detached = self.waiters.detach_waiter(waiter)?;
        if let Some(mut grant) = self.waiter_grants.remove(&waiter) {
            self.resources.release_all_held(&mut grant)?;
        }
        self.waiter_demands.remove(&waiter);
        if let Some(continuation) = detached.continuation_became_empty {
            self.release_continuation(continuation)?;
        }
        self.mark_unowned_for_cleanup(
            detached.operations_losing_last_waiter.iter().copied(),
            &reason,
        );
        for operation in detached.operations_losing_last_waiter {
            self.release_execution_if_prefetch_only(operation)?;
            self.handle_unowned_operation(operation, reason.clone(), now_ns)?;
        }
        Ok(())
    }

    /// Schedules one hard-feasible owner transition. Stage claims are admitted
    /// only when their action reaches the head of the fair queue.
    pub fn schedule_one(&mut self, now_ns: u64) -> Result<bool, RegistryError> {
        self.schedule_one_transition(now_ns, true)
    }

    fn schedule_one_transition(
        &mut self,
        now_ns: u64,
        allow_model_warmup_reserve: bool,
    ) -> Result<bool, RegistryError> {
        let operations = &self.operations;
        let resources = &self.resources;
        let Some(action) = self.runnable.pop_next_by(now_ns, |action| {
            let Some(operation) = operations.get(&action.operation) else {
                return FairQueueEntryState::Stale;
            };
            if !action_matches(operation, action.kind) {
                return FairQueueEntryState::Stale;
            }
            if !allow_model_warmup_reserve
                && operation.demand == ResourceDemand::ModelWarmup
                && action.kind == LoadActionKind::Reserve
            {
                return FairQueueEntryState::Blocked;
            }

            let feasible = match action.kind {
                LoadActionKind::Reserve => resources
                    .can_acquire(
                        operation.demand,
                        std::iter::once(PhysicalResourceClaim::new(ResourceKind::LoadOperation, 1))
                            .chain(read_claims(operation.plan)),
                    )
                    .is_ok(),
                LoadActionKind::SubmitUpload => resources
                    .can_acquire(operation.demand, upload_claims(operation.plan))
                    .is_ok(),
                LoadActionKind::PollInstall => resources
                    .can_acquire(operation.demand, install_claims(operation.plan))
                    .is_ok(),
                LoadActionKind::SubmitRead => true,
            };
            if feasible {
                FairQueueEntryState::Ready
            } else {
                FairQueueEntryState::Blocked
            }
        }) else {
            return Ok(false);
        };
        let mut operation = self
            .operations
            .remove(&action.operation)
            .expect("retained runnable action must reference an active operation");
        let result = match action.kind {
            LoadActionKind::Reserve => self.reserve(&mut operation, now_ns),
            LoadActionKind::SubmitRead => self.submit_read(&mut operation, now_ns).map(|()| None),
            LoadActionKind::SubmitUpload => {
                self.submit_upload(&mut operation, now_ns).map(|()| None)
            }
            LoadActionKind::PollInstall => self.poll_install(&mut operation, now_ns).map(|()| None),
        };
        match result {
            Ok(Some(failure)) => {
                let operation_id = operation.operation;
                if let Err(error) = self.retire_owned(
                    &mut operation,
                    RetirementReason::Failed(failure.clone()),
                    now_ns,
                ) {
                    self.operations.insert(operation_id, operation);
                    return Err(error);
                }
                self.fail_operation_waiters(
                    operation_id,
                    ContinuationFailure::Failed(failure),
                    now_ns,
                )?;
            }
            Ok(None) => {
                self.operations.insert(operation.operation, operation);
            }
            Err(error) => {
                self.operations.insert(operation.operation, operation);
                return Err(error);
            }
        }
        Ok(true)
    }

    pub fn enqueue_completion(&mut self, event: CompletionEvent) {
        self.completions.push_back(event);
    }

    pub fn collect_provider_completions(&mut self, maximum: usize) -> usize {
        let mut collected = 0;
        while collected < maximum {
            let Some(event) = self.provider.next_completion() else {
                break;
            };
            self.completions.push_back(event);
            collected += 1;
        }
        collected
    }

    pub fn process_one_completion(&mut self) -> Result<CompletionDisposition, RegistryError> {
        self.process_one_completion_with_observation(None)
    }

    pub(crate) fn process_one_completion_at(
        &mut self,
        observed_ns: u64,
    ) -> Result<CompletionDisposition, RegistryError> {
        self.process_one_completion_with_observation(Some(observed_ns))
    }

    fn process_one_completion_with_observation(
        &mut self,
        observed_ns: Option<u64>,
    ) -> Result<CompletionDisposition, RegistryError> {
        let Some(event) = self.completions.pop_front() else {
            return Ok(CompletionDisposition::QueueEmpty);
        };
        let operation_id = event.operation;
        let Some(active) = self.operations.get(&operation_id) else {
            let reason = if self.retirements.contains_key(&operation_id) {
                CompletionRejectionReason::RetiredOperation
            } else {
                CompletionRejectionReason::UnknownOperation
            };
            self.reject_completion(event, reason.clone());
            return Ok(CompletionDisposition::Rejected(reason));
        };
        if !active.stage.is_submitted_completion_stage()
            || (active.stage == LoadStage::Installing && !active.install_submitted)
        {
            let reason = CompletionRejectionReason::UnexpectedOwnerStage(active.stage);
            self.reject_completion(event, reason.clone());
            return Ok(CompletionDisposition::Rejected(reason));
        }
        let completion_bytes =
            active
                .plan
                .completion_bytes(active.stage)
                .ok_or(RegistryError::Protocol {
                    source: IoProtocolError::InvalidCompletionStage {
                        stage: active.stage,
                    },
                })?;
        let expectation = CompletionExpectation::new(
            active.operation,
            active.key,
            active.stage,
            completion_bytes,
        )?;
        if let Err(error) = event.validate(&expectation) {
            let reason = CompletionRejectionReason::Protocol(error);
            self.reject_completion(event, reason.clone());
            return Ok(CompletionDisposition::Rejected(reason));
        }

        let mut operation = self
            .operations
            .remove(&operation_id)
            .expect("completion was validated against an active operation");
        let completed_stage = operation.stage;
        let completed_key = operation.key;
        // Provider timestamps belong to a foreign monotonic domain (the
        // physical backend stamps a per-backend counter, not runtime
        // nanoseconds). Ledger spans must close in the runtime clock domain,
        // so the observation time of the drive step that collected this event
        // is the lower bound for every downstream phase/wait span.
        let timestamp = event.timestamp.as_nanos().max(observed_ns.unwrap_or(0));
        let post = match self.apply_completion(&mut operation, &event, observed_ns) {
            Ok(post) => post,
            Err(error) => {
                self.operations.insert(operation_id, operation);
                return Err(error);
            }
        };
        self.stats.physical_completions = self.stats.physical_completions.saturating_add(1);
        match post {
            PostCompletion::Keep => {
                let retry_cancellation = self
                    .operation_is_unowned(operation_id)
                    .then(|| {
                        operation
                            .cancellation
                            .as_ref()
                            .map(OperationCancellation::reason)
                            .cloned()
                    })
                    .flatten();
                self.operations.insert(operation_id, operation);
                if let Some(reason) = retry_cancellation {
                    self.handle_unowned_operation(operation_id, reason, timestamp)?;
                }
            }
            PostCompletion::Publish => {
                self.operations.insert(operation_id, operation);
                if !self.publish(operation_id, timestamp)? {
                    self.enqueue_publication(operation_id);
                }
            }
            PostCompletion::Terminal {
                retirement,
                failure,
            } => {
                if let Err(error) = self.retire_owned(&mut operation, retirement, timestamp) {
                    self.operations.insert(operation_id, operation);
                    return Err(error);
                }
                if let Some(failure) = failure {
                    self.fail_operation_waiters(operation_id, failure, timestamp)?;
                }
            }
        }
        Ok(CompletionDisposition::Applied {
            operation: operation_id,
            key: completed_key,
            stage: completed_stage,
        })
    }

    /// Drives owner commands and physical completions until no immediate work
    /// remains or the transition budget is exhausted.
    pub fn drive(&mut self, now_ns: u64, maximum: usize) -> Result<usize, RegistryError> {
        self.drive_with_warmup_admission(now_ns, maximum, true)
    }

    /// Drives foreground and exact-demand work without reserving new model-warmup
    /// operations. Every operation already admitted by the startup or idle owner
    /// still advances through read, upload, install, publication, and cleanup.
    /// This avoids H2D/install contention while preserving one physical executor.
    pub fn drive_foreground(
        &mut self,
        now_ns: u64,
        maximum: usize,
    ) -> Result<usize, RegistryError> {
        self.drive_with_warmup_admission(now_ns, maximum, false)
    }

    fn drive_with_warmup_admission(
        &mut self,
        now_ns: u64,
        maximum: usize,
        allow_model_warmup_reserve: bool,
    ) -> Result<usize, RegistryError> {
        let mut progressed = 0;
        while progressed < maximum {
            let round_start = progressed;
            while progressed < maximum && self.retry_one_lease_release()?.is_some() {
                progressed += 1;
            }
            while progressed < maximum && self.publish_one_queued(now_ns)?.is_some() {
                progressed += 1;
            }
            while progressed < maximum && self.cleanup_one_queued(now_ns)?.is_some() {
                progressed += 1;
            }
            if progressed < maximum {
                self.collect_provider_completions(maximum - progressed);
            }
            while progressed < maximum && !self.completions.is_empty() {
                let _ = self.process_one_completion_at(now_ns)?;
                progressed += 1;
            }
            while progressed < maximum
                && self.schedule_one_transition(now_ns, allow_model_warmup_reserve)?
            {
                progressed += 1;
            }
            if progressed == round_start {
                break;
            }
        }
        Ok(progressed)
    }

    pub fn pop_ready(&mut self, _now_ns: u64) -> Result<Option<ContinuationId>, RegistryError> {
        let Some(continuation) = self.waiters.ready_front() else {
            return Ok(None);
        };
        self.try_admit_ready(continuation)?;
        let has_ready = self
            .continuations
            .get(&continuation)
            .is_some_and(|state| state.ready_grant.is_some());
        if !has_ready {
            return Ok(None);
        }
        let popped = self
            .waiters
            .pop_ready()
            .expect("ready front remains queued until admission");
        let mut ready = self
            .continuations
            .get_mut(&popped)
            .expect("admitted ready continuation owns state")
            .ready_grant
            .take()
            .expect("admitted ready continuation owns a cohort grant");
        self.resources.release_all_held(&mut ready)?;
        self.failed_set.remove(&popped);
        Ok(Some(popped))
    }

    pub fn prepare_resume(
        &mut self,
        continuation: ContinuationId,
        dependencies: &DependencySet,
    ) -> Result<ResumeLease, RegistryError> {
        if !self.continuations.contains_key(&continuation) {
            return Err(RegistryError::UnknownContinuation { continuation });
        }
        let required = dependencies
            .iter()
            .filter_map(|dependency| dependency.materialization_key())
            .collect::<Vec<_>>();
        let mut bindings = Vec::with_capacity(required.len());
        for key in &required {
            bindings.push(self.validated_residency(*key)?);
        }
        let (backend, device) = match bindings.first() {
            Some(binding) => {
                let raw = binding.binding();
                for other in &bindings[1..] {
                    let candidate = other.binding();
                    if candidate.backend != raw.backend || candidate.device != raw.device {
                        return Err(RegistryError::Protocol {
                            source: IoProtocolError::ResidencyMismatch {
                                field: "resume lease backend/device",
                            },
                        });
                    }
                }
                (raw.backend, raw.device)
            }
            None => (BackendId::new(0), DeviceId::new(0)),
        };
        let dispatch = self.next_dispatch;
        self.next_dispatch = dispatch
            .checked_add(1)
            .ok_or(RegistryError::OperationIdExhausted)?;
        let leases = ResidencyLeaseSet::new(
            required.iter().copied(),
            bindings,
            MappingEpoch::new(dispatch),
            DispatchFenceContract::new(
                OperationId::new(dispatch),
                FenceId::new(dispatch),
                backend,
                device,
            ),
        )?;
        let demand = self
            .continuations
            .get(&continuation)
            .map(|state| state.demand)
            .ok_or(RegistryError::UnknownContinuation { continuation })?;
        let grant = self.resources.acquire(
            continuation.get(),
            demand,
            [
                PhysicalResourceClaim::new(ResourceKind::Arena, 1),
                PhysicalResourceClaim::new(ResourceKind::ResidencyLease, required.len() as u64),
            ],
        )?;
        Ok(ResumeLease {
            continuation,
            leases: Some(leases),
            grant: Some(grant),
        })
    }

    pub fn finish_resume(
        &mut self,
        lease: &mut ResumeLease,
        disposition: ResumeDisposition,
        started_ns: u64,
        finished_ns: u64,
    ) -> Result<(), RegistryError> {
        let continuation = lease.continuation;
        if lease.leases.is_some() {
            return Err(RegistryError::ResumeLeaseAlreadyTaken { continuation });
        }
        if let Some(grant) = lease.grant.as_ref() {
            self.resources.can_release_all_held(grant)?;
        }
        self.ledger.record_cohort_phase(
            CohortId::new(continuation.get()),
            CriticalPhase::Resume,
            started_ns,
            finished_ns.max(started_ns),
        )?;
        if let Some(grant) = lease.grant.as_mut() {
            self.resources.release_all_held(grant)?;
        }
        lease.grant = None;
        if disposition == ResumeDisposition::StillActive {
            return Ok(());
        }
        self.release_ready_waiters(continuation)?;
        self.release_continuation(continuation)
    }

    pub fn detach_continuation(
        &mut self,
        continuation: ContinuationId,
        reason: CancellationReason,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        let detached = self.waiters.detach_continuation(continuation);
        let mut lost_last = detached.operations_losing_last_waiter;
        for waiter in detached.removed_waiters {
            self.close_all_waits(waiter, now_ns)?;
            if let Some(mut grant) = self.waiter_grants.remove(&waiter) {
                self.resources.release_all_held(&mut grant)?;
            }
        }
        if let Some(waiters) = self.ready_waiters.remove(&continuation) {
            for waiter in waiters {
                if let Some(mut grant) = self.waiter_grants.remove(&waiter) {
                    self.resources.release_all_held(&mut grant)?;
                }
            }
        }
        self.release_continuation(continuation)?;
        self.mark_unowned_for_cleanup(lost_last.iter().copied(), &reason);
        for operation in std::mem::take(&mut lost_last) {
            self.release_execution_if_prefetch_only(operation)?;
            self.handle_unowned_operation(operation, reason.clone(), now_ns)?;
        }
        Ok(())
    }

    pub fn finish_transaction_custody(
        &mut self,
        transaction: ExecutionTransactionId,
        outcome: TransactionCustodyOutcome,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        if self
            .transactions
            .get(&transaction)
            .is_some_and(|state| state.materialization_custody_finished)
        {
            return Err(RegistryError::TransactionCustodyAlreadyFinished { transaction });
        }
        if let TransactionCustodyOutcome::Committed {
            started_ns,
            finished_ns,
        } = outcome
        {
            let cohort = CohortId::new(transaction.get());
            self.ledger.record_cohort_phase(
                cohort,
                CriticalPhase::Commit,
                started_ns,
                finished_ns.max(started_ns),
            )?;
            self.transactions
                .entry(transaction)
                .or_default()
                .cohorts
                .insert(cohort);
        }
        let prefetches = self
            .prefetches
            .keys()
            .copied()
            .filter(|owner| owner.belongs_to_transaction(transaction))
            .collect::<Vec<_>>();
        for prefetch in prefetches {
            self.release_prefetch(prefetch, CancellationReason::PrefetchCancelled, now_ns)?;
        }
        self.prefetch_failures
            .retain(|owner, _| !owner.belongs_to_transaction(transaction));

        let state = self.transactions.entry(transaction).or_default();
        let owned = state.owns_materialization_custody;
        state.owns_materialization_custody = false;
        state.materialization_custody_finished = true;
        if owned {
            self.release_custody_owner(MaterializationCustodyOwner::Transaction(transaction));
        }
        Ok(())
    }

    /// Records useful independent execution that legally covers dependency
    /// wait. The scheduler-facing owner feeds synchronous compute spans here so
    /// shared waits overlapping them are reported as covered, not uncovered.
    pub fn record_runnable_work(
        &mut self,
        started_ns: u64,
        finished_ns: u64,
    ) -> Result<(), RegistryError> {
        self.ledger
            .record_runnable_work(started_ns, finished_ns.max(started_ns))?;
        Ok(())
    }

    pub fn transaction_operations(
        &self,
        transaction: ExecutionTransactionId,
    ) -> impl Iterator<Item = OperationId> + '_ {
        self.transactions
            .get(&transaction)
            .into_iter()
            .flat_map(|state| state.operations.iter().copied())
    }

    pub fn snapshot_transaction_output(
        &mut self,
        transaction: ExecutionTransactionId,
        token: crate::io::OutputTokenId,
        externally_committed_tokens: usize,
        captured_at_ns: u64,
    ) -> Result<&crate::io::OutputTokenSnapshot, RegistryError> {
        let operations = self
            .transactions
            .get(&transaction)
            .map(|state| state.operations.clone())
            .unwrap_or_default();
        let mut cohorts = self
            .transactions
            .get(&transaction)
            .map(|state| state.cohorts.clone())
            .unwrap_or_default();
        cohorts.insert(CohortId::new(transaction.get()));
        Ok(self.ledger.snapshot_output(
            token,
            externally_committed_tokens,
            operations,
            cohorts,
            captured_at_ns,
        )?)
    }

    pub fn pop_failed(&mut self) -> Option<FailedContinuation> {
        let failed = self.failed_continuations.pop_front()?;
        self.failed_set.remove(&failed.continuation);
        Some(failed)
    }

    fn release_residency_accounting(
        &mut self,
        key: MaterializationKey,
    ) -> Result<(), RegistryError> {
        let Some(resident) = self.residencies.get(&key) else {
            return Ok(());
        };
        self.resources.can_release_all_held(&resident.grant)?;
        if resident.lease != ExecutionLeaseState::Released {
            self.provider.release_execution_lease(key)?;
        }
        let mut resident = self
            .residencies
            .remove(&key)
            .expect("validated residency remains registered");
        self.residency_locations
            .remove(&ResidencyLocation::from(resident.binding));
        self.pending_lease_releases.remove(&key);
        self.resources.release_all_held(&mut resident.grant)?;
        Ok(())
    }

    pub fn begin_shutdown(&mut self, now_ns: u64) -> Result<(), RegistryError> {
        self.shutting_down = true;
        let prefetches = self.prefetches.keys().copied().collect::<Vec<_>>();
        for prefetch in prefetches {
            self.release_prefetch(prefetch, CancellationReason::OwnerShutdown, now_ns)?;
        }
        let waiters: Vec<_> = self.waiters.active_waiters().collect();
        for waiter in waiters {
            self.detach_waiter(waiter, CancellationReason::OwnerShutdown, now_ns)?;
        }
        let operations: Vec<_> = self.operations.keys().copied().collect();
        for operation in operations {
            if self.operation_is_unowned(operation) {
                self.handle_unowned_operation(
                    operation,
                    CancellationReason::OwnerShutdown,
                    now_ns,
                )?;
            }
        }
        let continuations: Vec<_> = self.continuations.keys().copied().collect();
        for continuation in continuations {
            self.detach_continuation(continuation, CancellationReason::OwnerShutdown, now_ns)?;
        }
        let owners = self.owner_custody.keys().copied().collect::<Vec<_>>();
        for owner in owners {
            self.release_custody_owner(owner);
        }
        let resident: Vec<_> = self.residencies.keys().copied().collect();
        for key in resident {
            self.release_residency_accounting(key)?;
        }
        self.failed_continuations.clear();
        self.failed_set.clear();
        Ok(())
    }

    pub fn shutdown(
        &mut self,
        now_ns: u64,
        maximum_completions: usize,
    ) -> Result<ShutdownReport, RegistryError> {
        self.begin_shutdown(now_ns)?;
        let mut processed = 0;
        while processed < maximum_completions {
            self.collect_provider_completions(maximum_completions - processed);
            if self.completions.is_empty() {
                break;
            }
            let _ = self.process_one_completion()?;
            processed += 1;
        }
        let pending_operations = self.operations.len();
        let pending_completions = self.completions.len();
        let active_grants = self.resources.active_grants();
        if processed < maximum_completions
            && pending_completions == 0
            && let Some(operation) = self.operations.values().find(|operation| {
                matches!(
                    operation.stage,
                    LoadStage::ReadSubmitted | LoadStage::UploadSubmitted
                ) || (operation.stage == LoadStage::Installing && operation.install_submitted)
            })
        {
            return Err(RegistryError::LostCompletion {
                operation: operation.operation,
                stage: operation.stage,
                pending_operations,
                active_grants,
            });
        }
        let drained = pending_operations == 0
            && pending_completions == 0
            && active_grants == 0
            && self.waiters.is_empty()
            && self.prefetches.is_empty()
            && self.operation_prefetches.is_empty();
        if !drained {
            return Err(RegistryError::ShutdownIncomplete {
                pending_operations,
                pending_completions,
                active_grants,
            });
        }
        Ok(ShutdownReport {
            drained,
            pending_operations,
            pending_completions,
            active_grants,
        })
    }

    fn validate_request(&self, request: &LoadRequest) -> Result<(), RegistryError> {
        request.key.validate()?;
        request
            .plan
            .validate()
            .map_err(|source| RegistryError::ResourcePlan {
                key: Box::new(request.key),
                source,
            })?;
        let cost = request.plan.transition_cost();
        if cost > self.runnable.config().max_transition_cost {
            return Err(FairQueueError::TransitionTooLarge {
                cost,
                maximum: self.runnable.config().max_transition_cost,
            }
            .into());
        }
        self.validate_operation_capacity(request.plan)
    }

    fn validate_join(&self, active: &LoadOp, request: &LoadRequest) -> Result<(), RegistryError> {
        if active.plan != request.plan {
            return Err(RegistryError::ResourcePlanMismatch {
                key: Box::new(request.key),
                expected: Box::new(active.plan),
                requested: Box::new(request.plan),
            });
        }
        if !matches!(request.preparation, MaterializationPreparation::Transfer(_))
            || request.preparation.binding() != active.expected_binding
        {
            return Err(RegistryError::PublishedResidencyConflict {
                key: Box::new(request.key),
            });
        }
        Ok(())
    }

    fn promote_operation_for_execution(
        &mut self,
        operation: OperationId,
        demand: ResourceDemand,
    ) -> Result<(), RegistryError> {
        let merged = {
            let active = self
                .operations
                .get_mut(&operation)
                .expect("validated operation remains active");
            active.execution_required = true;
            if matches!(active.cancellation, Some(OperationCancellation::Pending(_))) {
                active.cancellation = None;
            }
            active.demand.merge(demand)
        };
        self.set_operation_demand(operation, merged)
    }

    fn release_execution_if_prefetch_only(
        &mut self,
        operation: OperationId,
    ) -> Result<(), RegistryError> {
        if self.waiters.waiter_count(operation) != 0
            || !self.operation_has_prefetch_owner(operation)
        {
            return Ok(());
        }
        let key = match self.operations.get(&operation) {
            Some(active) if active.execution_required => active.key,
            Some(_) | None => return Ok(()),
        };
        self.provider.release_execution_lease(key)?;
        self.operations
            .get_mut(&operation)
            .expect("operation remains active while releasing execution")
            .execution_required = false;
        self.refresh_operation_demand(operation)
    }

    fn prefetch_demand_for_operation(&self, operation: OperationId) -> Option<ResourceDemand> {
        self.operation_prefetches
            .get(&operation)
            .into_iter()
            .flatten()
            .filter_map(|owner| self.prefetches.get(owner).map(|state| state.demand))
            .reduce(ResourceDemand::merge)
    }

    fn waiter_demand_for_operation(&self, operation: OperationId) -> Option<ResourceDemand> {
        self.waiters
            .waiters_for(operation)
            .filter_map(|waiter| self.waiter_demands.get(&waiter).copied())
            .reduce(ResourceDemand::merge)
    }

    fn refresh_operation_demand(&mut self, operation: OperationId) -> Result<(), RegistryError> {
        let demand = match (
            self.prefetch_demand_for_operation(operation),
            self.waiter_demand_for_operation(operation),
        ) {
            (Some(prefetch), Some(required)) => prefetch.merge(required),
            (Some(prefetch), None) => prefetch,
            (None, Some(required)) => required,
            (None, None) => return Ok(()),
        };
        self.set_operation_demand(operation, demand)
    }

    fn set_operation_demand(
        &mut self,
        operation: OperationId,
        demand: ResourceDemand,
    ) -> Result<(), RegistryError> {
        let active = self
            .operations
            .get_mut(&operation)
            .expect("demand update requires an active operation");
        if active.demand == demand {
            return Ok(());
        }
        for grant in [
            active.logical_grant.as_ref(),
            active.read_grant.as_ref(),
            active.upload_grant.as_ref(),
            active.install_grant.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            self.resources.reclassify(grant, demand)?;
        }
        active.demand = demand;
        self.runnable
            .reclassify_where(fair_queue_band(demand), |action| {
                action.operation == operation
            });
        Ok(())
    }

    fn operation_is_unowned(&self, operation: OperationId) -> bool {
        self.waiters.waiter_count(operation) == 0 && !self.operation_has_prefetch_owner(operation)
    }

    fn validate_operation_capacity(
        &self,
        plan: MaterializationResourcePlan,
    ) -> Result<(), RegistryError> {
        for claim in read_claims(plan)
            .into_iter()
            .chain(upload_claims(plan))
            .chain(install_claims(plan))
        {
            let capacity = self
                .resources
                .snapshots()
                .find(|snapshot| snapshot.kind == claim.kind)
                .expect("hard resource catalog contains every resource kind")
                .capacity;
            if claim.amount > capacity {
                return Err(PhysicalResourceError::ExceedsCapacity {
                    kind: claim.kind,
                    requested: claim.amount,
                    capacity,
                }
                .into());
            }
        }
        Ok(())
    }

    fn create_operation(
        &mut self,
        request: LoadRequest,
        now_ns: u64,
    ) -> Result<OperationId, RegistryError> {
        let operation = OperationId::new(self.next_operation);
        if operation.is_zero() {
            return Err(RegistryError::OperationIdExhausted);
        }
        self.next_operation = self
            .next_operation
            .checked_add(1)
            .ok_or(RegistryError::OperationIdExhausted)?;
        let retirement = RetirementToken::new(operation, request.key)?;
        self.key_to_operation.insert(request.key, operation);
        self.operation_to_key.insert(operation, request.key);
        self.stage_history
            .insert(operation, vec![LoadStage::Reserved]);
        self.operations.insert(
            operation,
            LoadOp {
                operation,
                key: request.key,
                expected_binding: request.preparation.binding(),
                plan: request.plan,
                demand: request.demand,
                execution_required: request.demand.is_required(),
                stage: LoadStage::Reserved,
                reservation: None,
                logical_grant: None,
                read_grant: None,
                upload_grant: None,
                install_grant: None,
                retirement: Some(retirement),
                cancellation: None,
                install_submitted: false,
                replaced: match request.preparation {
                    MaterializationPreparation::Transfer(transfer) => transfer.evicted(),
                    MaterializationPreparation::Resident(_) => None,
                },
                stage_started_ns: now_ns,
            },
        );
        self.runnable.push(
            LoadAction {
                operation,
                kind: LoadActionKind::Reserve,
            },
            fair_queue_band(request.demand),
            1,
            now_ns,
        )?;
        Ok(operation)
    }

    fn rollback_adopted_residency(&mut self, key: MaterializationKey) -> Result<(), RegistryError> {
        let Some(resident) = self.residencies.get(&key) else {
            return Ok(());
        };
        self.resources.can_release_all_held(&resident.grant)?;
        if resident.lease != ExecutionLeaseState::Released {
            self.provider.release_execution_lease(key)?;
        }
        let mut resident = self
            .residencies
            .remove(&key)
            .expect("validated adopted residency remains registered");
        self.residency_locations
            .remove(&ResidencyLocation::from(resident.binding));
        self.pending_lease_releases.remove(&key);
        self.resources.release_all_held(&mut resident.grant)?;
        Ok(())
    }

    fn rollback_reserved_operation(
        &mut self,
        operation: OperationId,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        let Some(mut operation) = self.operations.remove(&operation) else {
            return Ok(());
        };
        if operation.reservation.is_some() {
            self.provider.cancel(
                operation.operation,
                operation.key,
                LoadStage::Reserved,
                CancellationReason::Superseded,
            )?;
        } else {
            self.provider.discard_preparation(operation.key)?;
        }
        self.retire_owned(
            &mut operation,
            RetirementReason::Cancelled(CancellationReason::Superseded),
            now_ns,
        )
    }

    fn reserve(
        &mut self,
        operation: &mut LoadOp,
        now_ns: u64,
    ) -> Result<Option<FailureReason>, RegistryError> {
        debug_assert_eq!(operation.stage, LoadStage::Reserved);
        debug_assert!(operation.reservation.is_none());
        let logical_grant = self.resources.acquire(
            operation.operation.get(),
            operation.demand,
            [PhysicalResourceClaim::new(ResourceKind::LoadOperation, 1)],
        )?;
        operation.logical_grant = Some(logical_grant);
        let mut grant = match self.resources.acquire(
            operation.operation.get(),
            operation.demand,
            read_claims(operation.plan),
        ) {
            Ok(grant) => grant,
            Err(error) => {
                let mut logical_grant = operation
                    .logical_grant
                    .take()
                    .expect("reserve acquired load-operation credit");
                self.resources.release_all_held(&mut logical_grant)?;
                return Err(error.into());
            }
        };
        let reservation =
            match self
                .provider
                .reserve(operation.operation, operation.key, operation.plan)
            {
                Ok(reservation) if reservation.binding() == operation.expected_binding => {
                    reservation
                }
                Ok(mut reservation) => {
                    let _ = reservation.retire_slabs();
                    self.resources.release_all_held(&mut grant)?;
                    return Ok(Some(FailureReason::ContractViolation {
                        message: "physical provider reserved a different residency binding".into(),
                    }));
                }
                Err(failure) => {
                    self.resources.release_all_held(&mut grant)?;
                    return Ok(Some(failure));
                }
            };
        operation.reservation = Some(reservation);
        operation.read_grant = Some(grant);
        operation.stage_started_ns = now_ns;
        self.runnable.push(
            LoadAction {
                operation: operation.operation,
                kind: LoadActionKind::SubmitRead,
            },
            fair_queue_band(operation.demand),
            operation.plan.requirements.storage_read_bytes,
            now_ns,
        )?;
        Ok(None)
    }

    fn submit_read(&mut self, operation: &mut LoadOp, now_ns: u64) -> Result<(), RegistryError> {
        debug_assert_eq!(operation.stage, LoadStage::Reserved);
        let rejection = self
            .provider
            .submit_read(
                operation.operation,
                operation.key,
                operation
                    .reservation
                    .as_ref()
                    .expect("submit-read action requires a physical reservation"),
                operation.plan,
            )
            .err();
        operation
            .reservation
            .as_mut()
            .expect("submit-read action requires a physical reservation")
            .mark_read_submitted()?;
        self.resources.mark_submitted(
            operation
                .read_grant
                .as_mut()
                .expect("reserved operation owns a read grant"),
            READ_CUSTODY,
        )?;
        self.transition(operation, LoadStage::ReadSubmitted)?;
        operation.stage_started_ns = now_ns;
        if let Some(reason) = rejection {
            self.enqueue_provider_rejection(operation, LoadStage::ReadSubmitted, reason, now_ns);
        }
        Ok(())
    }

    fn submit_upload(&mut self, operation: &mut LoadOp, now_ns: u64) -> Result<(), RegistryError> {
        debug_assert_eq!(operation.stage, LoadStage::HostReady);
        let grant = self.resources.acquire(
            operation.operation.get(),
            operation.demand,
            upload_claims(operation.plan),
        )?;
        operation.upload_grant = Some(grant);
        let rejection = self
            .provider
            .submit_upload(
                operation.operation,
                operation.key,
                operation
                    .reservation
                    .as_ref()
                    .expect("host-ready operation owns a physical reservation"),
                operation.plan,
            )
            .err();
        operation
            .reservation
            .as_mut()
            .expect("host-ready operation owns a physical reservation")
            .mark_upload_submitted()?;
        self.resources.mark_submitted(
            operation
                .read_grant
                .as_mut()
                .expect("host-ready operation owns pinned-host credit"),
            &[ResourceKind::PinnedHostBytes],
        )?;
        self.resources.mark_submitted(
            operation
                .upload_grant
                .as_mut()
                .expect("upload-submitted operation owns an upload grant"),
            UPLOAD_CUSTODY,
        )?;
        self.transition(operation, LoadStage::UploadSubmitted)?;
        operation.stage_started_ns = now_ns;
        if let Some(reason) = rejection {
            self.enqueue_provider_rejection(operation, LoadStage::UploadSubmitted, reason, now_ns);
        }
        Ok(())
    }

    fn poll_install(&mut self, operation: &mut LoadOp, now_ns: u64) -> Result<(), RegistryError> {
        debug_assert_eq!(operation.stage, LoadStage::Installing);
        debug_assert!(!operation.install_submitted);
        let grant = self.resources.acquire(
            operation.operation.get(),
            operation.demand,
            install_claims(operation.plan),
        )?;
        operation.install_grant = Some(grant);
        let rejection = self
            .provider
            .poll_install(
                operation.operation,
                operation.key,
                operation
                    .reservation
                    .as_ref()
                    .expect("installing operation owns a physical reservation"),
                operation.plan,
            )
            .err();
        self.resources.mark_submitted(
            operation
                .upload_grant
                .as_mut()
                .expect("installing operation owns a destination grant"),
            INSTALL_DESTINATION_CUSTODY,
        )?;
        self.resources.mark_submitted(
            operation
                .install_grant
                .as_mut()
                .expect("installing operation owns an install-slot grant"),
            INSTALL_CUSTODY,
        )?;
        operation.install_submitted = true;
        operation.stage_started_ns = now_ns;
        if let Some(reason) = rejection {
            self.enqueue_provider_rejection(operation, LoadStage::Installing, reason, now_ns);
        }
        Ok(())
    }

    fn apply_completion(
        &mut self,
        operation: &mut LoadOp,
        event: &CompletionEvent,
        observed_ns: Option<u64>,
    ) -> Result<PostCompletion, RegistryError> {
        let next = event.next_stage()?;
        let timestamp = event
            .timestamp
            .as_nanos()
            .max(operation.stage_started_ns)
            .max(observed_ns.unwrap_or(0));
        match event.stage {
            LoadStage::ReadSubmitted => {
                match event.outcome {
                    CompletionOutcome::Succeeded => {
                        operation
                            .reservation
                            .as_mut()
                            .expect("read-submitted operation owns a physical reservation")
                            .mark_host_ready()?;
                    }
                    _ => operation
                        .reservation
                        .as_mut()
                        .expect("read-submitted operation owns a physical reservation")
                        .mark_read_returned_without_artifact()?,
                }
                let grant = operation
                    .read_grant
                    .as_mut()
                    .expect("read-submitted operation owns a read grant");
                self.resources.mark_returned(grant, READ_CUSTODY)?;
                self.ledger.record_operation_phase(
                    operation.operation,
                    CriticalPhase::Read,
                    operation.stage_started_ns,
                    timestamp,
                )?;
                if next == LoadStage::HostReady {
                    self.resources.release_held(grant, READ_RELEASE)?;
                }
                self.transition(operation, next)?;
                if next == LoadStage::HostReady {
                    self.runnable.push(
                        LoadAction {
                            operation: operation.operation,
                            kind: LoadActionKind::SubmitUpload,
                        },
                        fair_queue_band(operation.demand),
                        operation.plan.requirements.h2d_bytes,
                        timestamp,
                    )?;
                    Ok(PostCompletion::Keep)
                } else {
                    terminal_post(&event.outcome)
                }
            }
            LoadStage::UploadSubmitted => {
                operation
                    .reservation
                    .as_mut()
                    .expect("upload-submitted operation owns a physical reservation")
                    .mark_upload_fence(event.timestamp)?;
                let read_grant = operation
                    .read_grant
                    .as_mut()
                    .expect("upload-submitted operation owns pinned-slab credit");
                self.resources
                    .mark_returned(read_grant, &[ResourceKind::PinnedHostBytes])?;
                self.resources.release_all_held(read_grant)?;
                operation.read_grant = None;
                let upload_grant = operation
                    .upload_grant
                    .as_mut()
                    .expect("upload-submitted operation owns an upload grant");
                self.resources.mark_returned(upload_grant, UPLOAD_CUSTODY)?;
                self.resources.release_held(upload_grant, UPLOAD_RELEASE)?;
                operation
                    .reservation
                    .as_mut()
                    .expect("upload-submitted operation owns a physical reservation")
                    .retire_slabs()?;
                self.ledger.record_operation_phase(
                    operation.operation,
                    CriticalPhase::Upload,
                    operation.stage_started_ns,
                    timestamp,
                )?;
                self.transition(operation, next)?;
                if next == LoadStage::Installing {
                    operation.install_submitted = false;
                    self.runnable.push(
                        LoadAction {
                            operation: operation.operation,
                            kind: LoadActionKind::PollInstall,
                        },
                        fair_queue_band(operation.demand),
                        1,
                        timestamp,
                    )?;
                    Ok(PostCompletion::Keep)
                } else {
                    terminal_post(&event.outcome)
                }
            }
            LoadStage::Installing => {
                let destination_grant = operation
                    .upload_grant
                    .as_mut()
                    .expect("installing operation owns a destination grant");
                self.resources
                    .mark_returned(destination_grant, INSTALL_DESTINATION_CUSTODY)?;
                let mut install_grant = operation
                    .install_grant
                    .take()
                    .expect("installing operation owns an install-stage grant");
                self.resources
                    .mark_returned(&mut install_grant, INSTALL_CUSTODY)?;
                self.resources.release_all_held(&mut install_grant)?;
                operation.install_submitted = false;
                self.ledger.record_operation_phase(
                    operation.operation,
                    CriticalPhase::Publish,
                    operation.stage_started_ns,
                    timestamp,
                )?;
                self.transition(operation, next)?;
                if next == LoadStage::Resident {
                    Ok(PostCompletion::Publish)
                } else {
                    terminal_post(&event.outcome)
                }
            }
            stage => Err(IoProtocolError::InvalidCompletionStage { stage }.into()),
        }
    }

    fn publication_preflight(
        &self,
        operation: &LoadOp,
    ) -> Result<PublicationPreflight, RegistryError> {
        let binding = operation
            .reservation
            .as_ref()
            .expect("resident operation owns a physical reservation")
            .binding();
        ValidatedResidencyBinding::new(operation.key, binding)?;
        if self.residencies.contains_key(&operation.key) {
            return Err(RegistryError::PublishedResidencyConflict {
                key: Box::new(operation.key),
            });
        }

        let location = ResidencyLocation::from(binding);
        let Some(replaced) = self.residency_locations.get(&location).copied() else {
            if let Some(evicted) = operation.replaced
                && (self.residencies.contains_key(&evicted)
                    || evicted.model() != binding.model
                    || evicted.backend() != binding.backend
                    || evicted.device() != binding.device
                    || evicted.destination_generation() >= binding.generation)
            {
                return Err(RegistryError::PublishedResidencyConflict {
                    key: Box::new(evicted),
                });
            }
            return Ok(PublicationPreflight::Ready { replaced: None });
        };
        if operation.replaced != Some(replaced) {
            return Err(RegistryError::PublishedResidencyConflict {
                key: Box::new(replaced),
            });
        }
        let resident = self
            .residencies
            .get(&replaced)
            .expect("located residency remains registered");
        if resident.binding.generation >= binding.generation {
            return Err(RegistryError::PublishedResidencyConflict {
                key: Box::new(replaced),
            });
        }
        if resident.lease != ExecutionLeaseState::Released || self.has_custody(replaced) {
            return Ok(PublicationPreflight::Blocked);
        }
        self.resources.can_release_all_held(&resident.grant)?;
        Ok(PublicationPreflight::Ready {
            replaced: Some(replaced),
        })
    }

    fn enqueue_publication(&mut self, operation: OperationId) {
        if self.queued_publications.insert(operation) {
            self.pending_publications.push_back(operation);
        }
    }

    fn publish_one_queued(
        &mut self,
        timestamp: u64,
    ) -> Result<Option<MaterializationKey>, RegistryError> {
        let queued = self.pending_publications.len();
        for _ in 0..queued {
            let operation = self
                .pending_publications
                .pop_front()
                .expect("publication queue length is stable during one pass");
            self.queued_publications.remove(&operation);
            let Some(active) = self.operations.get(&operation) else {
                continue;
            };
            if active.stage != LoadStage::Resident {
                continue;
            }
            let key = active.key;
            if self.publish(operation, timestamp)? {
                return Ok(Some(key));
            }
            self.enqueue_publication(operation);
        }
        Ok(None)
    }

    fn preflight_retirement(&self, operation: &LoadOp) -> Result<(), RegistryError> {
        operation.stage.validate_transition(LoadStage::Retired)?;
        if operation
            .retirement
            .as_ref()
            .expect("active operation owns retirement authority")
            .is_consumed()
        {
            return Err(IoProtocolError::RetirementAlreadyConsumed.into());
        }
        for grant in [
            &operation.read_grant,
            &operation.upload_grant,
            &operation.install_grant,
            &operation.logical_grant,
        ] {
            if let Some(grant) = grant.as_ref() {
                self.resources.can_release_all_held(grant)?;
            }
        }
        Ok(())
    }

    fn preflight_operation_waiters(
        &self,
        operation: OperationId,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        for waiter in self.waiters.waiters_for(operation) {
            if let Some(start_ns) = self.wait_started.get(&(waiter, operation)) {
                TimeSpan::new(*start_ns, now_ns.max(*start_ns))?;
            }
        }
        Ok(())
    }

    fn publish(
        &mut self,
        operation_id: OperationId,
        timestamp: u64,
    ) -> Result<bool, RegistryError> {
        let preflight = {
            let operation =
                self.operations
                    .get(&operation_id)
                    .ok_or(RegistryError::UnknownOperation {
                        operation: operation_id,
                    })?;
            let publication = self.publication_preflight(operation)?;
            if matches!(publication, PublicationPreflight::Ready { .. }) {
                self.preflight_retirement(operation)?;
                self.preflight_operation_waiters(operation_id, timestamp)?;
            }
            publication
        };
        let PublicationPreflight::Ready { replaced } = preflight else {
            return Ok(false);
        };
        let operation = self
            .operations
            .get(&operation_id)
            .expect("publication preflight retained active operation");
        let release = operation
            .upload_grant
            .as_ref()
            .expect("resident operation owns its destination grant")
            .claims()
            .map(|claim| claim.kind)
            .filter(|kind| !RESIDENCY_CLAIMS.contains(kind))
            .collect::<Vec<_>>();
        self.resources.can_release_held(
            operation
                .upload_grant
                .as_ref()
                .expect("resident operation owns its destination grant"),
            &release,
        )?;
        for grant in [
            &operation.read_grant,
            &operation.install_grant,
            &operation.logical_grant,
        ] {
            if let Some(grant) = grant.as_ref() {
                self.resources.can_release_all_held(grant)?;
            }
        }

        let binding = operation
            .reservation
            .as_ref()
            .expect("resident operation owns a physical reservation")
            .binding();
        if let Some(replaced) = replaced {
            let mut resident = self
                .residencies
                .remove(&replaced)
                .expect("publication preflight retained replaced residency");
            self.residency_locations
                .remove(&ResidencyLocation::from(resident.binding));
            self.pending_lease_releases.remove(&replaced);
            self.resources.release_all_held(&mut resident.grant)?;
        }

        let mut operation = self
            .operations
            .remove(&operation_id)
            .expect("publication preflight retained active operation");
        let mut grant = operation
            .upload_grant
            .take()
            .expect("resident operation owns its destination grant");
        self.resources.release_held(&mut grant, &release)?;
        let lease = if !operation.execution_required {
            ExecutionLeaseState::Released
        } else if self.has_custody(operation.key) {
            ExecutionLeaseState::Held
        } else {
            self.pending_lease_releases.insert(operation.key);
            ExecutionLeaseState::ReleasePending
        };
        self.residencies.insert(
            operation.key,
            ResidentEntry {
                binding,
                grant,
                lease,
            },
        );
        self.residency_locations
            .insert(ResidencyLocation::from(binding), operation.key);
        self.retire_owned(
            &mut operation,
            RetirementReason::ResidentOwnershipTransferred,
            timestamp,
        )?;
        self.resolve_operation_waiters(operation_id, timestamp)?;
        self.stats.publications = self.stats.publications.saturating_add(1);
        Ok(true)
    }

    fn resolve_operation_waiters(
        &mut self,
        operation: OperationId,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        let waiters: Vec<_> = self.waiters.waiters_for(operation).collect();
        for waiter in &waiters {
            self.close_wait(*waiter, operation, now_ns)?;
        }
        let resolution = self.waiters.satisfy_operation(operation);
        for waiter in resolution.completed_waiters {
            self.ready_waiters
                .entry(waiter.continuation())
                .or_default()
                .insert(waiter);
        }
        Ok(())
    }

    fn fail_operation_waiters(
        &mut self,
        operation: OperationId,
        failure: ContinuationFailure,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        let continuations = self.waiters.continuations_waiting_on(operation);
        let mut lost_last = BTreeSet::new();
        for continuation in continuations {
            let detached = self.waiters.detach_continuation(continuation);
            for waiter in detached.removed_waiters {
                self.close_all_waits(waiter, now_ns)?;
                if let Some(mut grant) = self.waiter_grants.remove(&waiter) {
                    self.resources.release_all_held(&mut grant)?;
                }
            }
            lost_last.extend(detached.operations_losing_last_waiter);
            self.release_continuation(continuation)?;
            if self.failed_set.insert(continuation) {
                self.failed_continuations.push_back(FailedContinuation {
                    continuation,
                    failure: failure.clone(),
                });
            }
        }
        lost_last.remove(&operation);
        self.mark_unowned_for_cleanup(lost_last.iter().copied(), &CancellationReason::Superseded);
        for other in lost_last {
            self.handle_unowned_operation(other, CancellationReason::Superseded, now_ns)?;
        }
        Ok(())
    }

    fn mark_unowned_for_cleanup(
        &mut self,
        operations: impl IntoIterator<Item = OperationId>,
        reason: &CancellationReason,
    ) {
        for operation in operations {
            if self.operation_is_unowned(operation)
                && let Some(active) = self.operations.get_mut(&operation)
                && active.cancellation.is_none()
            {
                active.cancellation = Some(OperationCancellation::Pending(reason.clone()));
                self.enqueue_cleanup(operation);
            }
        }
    }

    fn release_prefetch(
        &mut self,
        owner: PrefetchOwner,
        reason: CancellationReason,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        let state = self
            .prefetches
            .remove(&owner)
            .ok_or(RegistryError::UnknownPrefetch { owner })?;
        for operation in state.operations {
            let remove_reverse = if let Some(owners) = self.operation_prefetches.get_mut(&operation)
            {
                owners.remove(&owner);
                owners.is_empty()
            } else {
                false
            };
            if remove_reverse {
                self.operation_prefetches.remove(&operation);
            }
            if self.operation_is_unowned(operation) {
                self.mark_unowned_for_cleanup([operation], &reason);
                self.handle_unowned_operation(operation, reason.clone(), now_ns)?;
            } else {
                self.refresh_operation_demand(operation)?;
            }
        }
        Ok(())
    }

    fn record_prefetch_failure(&mut self, operation: OperationId, reason: &RetirementReason) {
        if matches!(reason, RetirementReason::ResidentOwnershipTransferred) {
            return;
        }
        let Some(owners) = self.operation_prefetches.get(&operation) else {
            return;
        };
        for owner in owners {
            self.prefetch_failures
                .entry(*owner)
                .or_insert_with(|| reason.clone());
        }
    }

    fn forget_operation_prefetch_owners(&mut self, operation: OperationId) {
        let Some(owners) = self.operation_prefetches.remove(&operation) else {
            return;
        };
        for owner in owners {
            let remove_owner = if let Some(state) = self.prefetches.get_mut(&owner) {
                state.operations.remove(&operation);
                state.operations.is_empty()
            } else {
                false
            };
            if remove_owner {
                self.prefetches.remove(&owner);
            }
        }
    }

    fn enqueue_cleanup(&mut self, operation: OperationId) {
        if self.queued_cleanups.insert(operation) {
            self.pending_cleanups.push_back(operation);
        }
    }

    fn cleanup_one_queued(
        &mut self,
        now_ns: u64,
    ) -> Result<Option<MaterializationKey>, RegistryError> {
        while let Some(operation) = self.pending_cleanups.pop_front() {
            self.queued_cleanups.remove(&operation);
            let Some(active) = self.operations.get(&operation) else {
                continue;
            };
            if !self.operation_is_unowned(operation) {
                continue;
            }
            let Some(reason) =
                active
                    .cancellation
                    .as_ref()
                    .and_then(|cancellation| match cancellation {
                        OperationCancellation::Pending(reason) => Some(reason.clone()),
                        OperationCancellation::Submitted { .. } => None,
                    })
            else {
                continue;
            };
            let key = active.key;
            if let Err(error) = self.handle_unowned_operation(operation, reason, now_ns) {
                self.enqueue_cleanup(operation);
                return Err(error);
            }
            return Ok(Some(key));
        }
        Ok(None)
    }

    fn submit_operation_cancellation(
        &mut self,
        operation: &mut LoadOp,
        reason: CancellationReason,
    ) -> Result<(), RegistryError> {
        if matches!(
            operation.cancellation,
            Some(OperationCancellation::Submitted { stage, .. }) if stage == operation.stage
        ) {
            return Ok(());
        }
        if let Err(error) = self.provider.cancel(
            operation.operation,
            operation.key,
            operation.stage,
            reason.clone(),
        ) {
            operation.cancellation = Some(OperationCancellation::Pending(reason));
            return Err(RegistryError::Provider { source: error });
        }
        operation.cancellation = Some(OperationCancellation::Submitted {
            stage: operation.stage,
            reason,
        });
        self.stats.cancellations_requested = self.stats.cancellations_requested.saturating_add(1);
        Ok(())
    }

    fn handle_unowned_operation(
        &mut self,
        operation_id: OperationId,
        reason: CancellationReason,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        if !self.operation_is_unowned(operation_id) {
            if let Some(operation) = self.operations.get_mut(&operation_id)
                && matches!(
                    operation.cancellation,
                    Some(OperationCancellation::Pending(_))
                )
            {
                operation.cancellation = None;
            }
            return Ok(());
        }
        if self
            .operations
            .get(&operation_id)
            .is_some_and(|operation| operation.stage == LoadStage::Resident)
        {
            if !self.publish(operation_id, now_ns)? {
                self.enqueue_publication(operation_id);
            }
            return Ok(());
        }
        let Some(mut operation) = self.operations.remove(&operation_id) else {
            return Ok(());
        };
        let cleanup = (|| match operation.stage {
            LoadStage::Reserved => {
                if operation.reservation.is_some() {
                    self.submit_operation_cancellation(&mut operation, reason.clone())?;
                } else {
                    self.provider.discard_preparation(operation.key)?;
                }
                self.retire_owned(&mut operation, RetirementReason::Cancelled(reason), now_ns)?;
                Ok(false)
            }
            LoadStage::HostReady => {
                self.submit_operation_cancellation(&mut operation, reason.clone())?;
                operation
                    .reservation
                    .as_mut()
                    .expect("host-ready operation owns a physical reservation")
                    .retire_slabs()?;
                self.transition(&mut operation, LoadStage::Stale)?;
                self.retire_owned(&mut operation, RetirementReason::Cancelled(reason), now_ns)?;
                Ok(false)
            }
            LoadStage::Installing if !operation.install_submitted => {
                self.submit_operation_cancellation(&mut operation, reason.clone())?;
                self.transition(&mut operation, LoadStage::Stale)?;
                self.retire_owned(&mut operation, RetirementReason::Cancelled(reason), now_ns)?;
                Ok(false)
            }
            LoadStage::ReadSubmitted | LoadStage::UploadSubmitted | LoadStage::Installing => {
                self.submit_operation_cancellation(&mut operation, reason)?;
                Ok(true)
            }
            LoadStage::Failed | LoadStage::Stale | LoadStage::Draining => {
                self.retire_owned(&mut operation, RetirementReason::Cancelled(reason), now_ns)?;
                Ok(false)
            }
            LoadStage::Retired => Ok(false),
            LoadStage::Resident => unreachable!("resident operations publish in place"),
        })();
        match cleanup {
            Ok(retain) => {
                if retain {
                    self.operations.insert(operation_id, operation);
                }
                Ok(())
            }
            Err(error) => {
                self.operations.insert(operation_id, operation);
                Err(error)
            }
        }
    }

    fn retire_owned(
        &mut self,
        operation: &mut LoadOp,
        reason: RetirementReason,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        self.preflight_retirement(operation)?;
        for grant in [
            &mut operation.read_grant,
            &mut operation.upload_grant,
            &mut operation.install_grant,
            &mut operation.logical_grant,
        ] {
            if let Some(grant) = grant.as_mut() {
                self.resources.release_all_held(grant)?;
            }
            *grant = None;
        }
        self.transition(operation, LoadStage::Retired)?;
        let record = operation
            .retirement
            .as_mut()
            .expect("active operation owns retirement authority")
            .consume(reason, CompletionTimestamp::from_nanos(now_ns))?;
        if self.key_to_operation.get(&operation.key) == Some(&operation.operation) {
            self.key_to_operation.remove(&operation.key);
        }
        self.record_prefetch_failure(operation.operation, &record.reason);
        self.forget_operation_prefetch_owners(operation.operation);
        self.retirements.insert(operation.operation, record);
        self.stats.retirements = self.stats.retirements.saturating_add(1);
        Ok(())
    }

    fn release_ready_waiters(&mut self, continuation: ContinuationId) -> Result<(), RegistryError> {
        if let Some(waiters) = self.ready_waiters.remove(&continuation) {
            for waiter in waiters {
                if let Some(mut grant) = self.waiter_grants.remove(&waiter) {
                    self.resources.release_all_held(&mut grant)?;
                }
            }
        }
        Ok(())
    }

    fn release_continuation(&mut self, continuation: ContinuationId) -> Result<(), RegistryError> {
        let Some(mut state) = self.continuations.remove(&continuation) else {
            return Ok(());
        };
        if state.owns_stage_custody {
            self.release_custody_owner(MaterializationCustodyOwner::Stage(continuation));
            state.owns_stage_custody = false;
        }
        if let Some(mut ready) = state.ready_grant {
            self.resources.release_all_held(&mut ready)?;
        }
        self.resources.release_all_held(&mut state.grant)?;
        Ok(())
    }

    pub(crate) fn finish_pending_lease_releases(&mut self) -> Result<(), RegistryError> {
        while self.retry_one_lease_release()?.is_some() {}
        Ok(())
    }

    fn retry_one_lease_release(&mut self) -> Result<Option<MaterializationKey>, RegistryError> {
        let Some(key) = self.pending_lease_releases.first().copied() else {
            return Ok(None);
        };
        let Some(resident) = self.residencies.get_mut(&key) else {
            self.pending_lease_releases.remove(&key);
            return Ok(Some(key));
        };
        if resident.lease != ExecutionLeaseState::ReleasePending {
            self.pending_lease_releases.remove(&key);
            return Ok(Some(key));
        }
        self.provider.release_execution_lease(key)?;
        resident.lease = ExecutionLeaseState::Released;
        self.pending_lease_releases.remove(&key);
        Ok(Some(key))
    }

    fn try_admit_ready(&mut self, continuation: ContinuationId) -> Result<bool, RegistryError> {
        let state = self
            .continuations
            .get(&continuation)
            .ok_or(RegistryError::UnknownContinuation { continuation })?;
        if state.ready_grant.is_some() {
            return Ok(true);
        }
        let demand = state.demand;
        match self.resources.acquire(
            continuation.get(),
            demand,
            [PhysicalResourceClaim::new(ResourceKind::ReadyCohort, 1)],
        ) {
            Ok(grant) => {
                self.continuations
                    .get_mut(&continuation)
                    .expect("continuation state was just read")
                    .ready_grant = Some(grant);
                Ok(true)
            }
            Err(
                PhysicalResourceError::TemporarilyUnavailable { .. }
                | PhysicalResourceError::ExecutionReserve { .. }
                | PhysicalResourceError::ExceedsCapacity { .. },
            ) => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    fn close_wait(
        &mut self,
        waiter: WaiterId,
        operation: OperationId,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        if let Some(start_ns) = self.wait_started.remove(&(waiter, operation)) {
            self.ledger.record_shared_wait(
                operation,
                CohortId::new(waiter.continuation().get()),
                start_ns,
                now_ns.max(start_ns),
            )?;
        }
        Ok(())
    }

    fn close_all_waits(&mut self, waiter: WaiterId, now_ns: u64) -> Result<(), RegistryError> {
        let operations: Vec<_> = self
            .wait_started
            .keys()
            .filter_map(|(candidate, operation)| (*candidate == waiter).then_some(*operation))
            .collect();
        for operation in operations {
            self.close_wait(waiter, operation, now_ns)?;
        }
        Ok(())
    }

    fn enqueue_provider_rejection(
        &mut self,
        operation: &LoadOp,
        stage: LoadStage,
        reason: FailureReason,
        now_ns: u64,
    ) {
        self.completions.push_back(CompletionEvent::new(
            operation.operation,
            operation.key,
            stage,
            CompletionOutcome::Failed(reason),
            0,
            CompletionGeneration::for_key(operation.key),
            CompletionTimestamp::from_nanos(now_ns),
        ));
    }

    fn transition(&mut self, operation: &mut LoadOp, next: LoadStage) -> Result<(), RegistryError> {
        operation.stage.validate_transition(next)?;
        operation.stage = next;
        self.stage_history
            .entry(operation.operation)
            .or_default()
            .push(next);
        Ok(())
    }

    fn reject_completion(&mut self, event: CompletionEvent, reason: CompletionRejectionReason) {
        self.stats.rejected_completions = self.stats.rejected_completions.saturating_add(1);
        self.rejected_completions
            .push(CompletionRejection { event, reason });
    }
}

fn action_matches(operation: &LoadOp, action: LoadActionKind) -> bool {
    matches!(
        (
            operation.stage,
            operation.reservation.is_some(),
            operation.install_submitted,
            action,
        ),
        (LoadStage::Reserved, false, _, LoadActionKind::Reserve)
            | (LoadStage::Reserved, true, _, LoadActionKind::SubmitRead)
            | (LoadStage::HostReady, true, _, LoadActionKind::SubmitUpload)
            | (
                LoadStage::Installing,
                true,
                false,
                LoadActionKind::PollInstall
            )
    )
}

fn read_claims(plan: MaterializationResourcePlan) -> [PhysicalResourceClaim; 3] {
    [
        PhysicalResourceClaim::new(ResourceKind::ReadSlot, plan.requirements.read_slots),
        PhysicalResourceClaim::new(
            ResourceKind::PinnedHostBytes,
            plan.requirements.pinned_host_bytes,
        ),
        PhysicalResourceClaim::new(
            ResourceKind::StorageReadBytes,
            plan.requirements.storage_read_bytes,
        ),
    ]
}

fn upload_claims(plan: MaterializationResourcePlan) -> [PhysicalResourceClaim; 3] {
    [
        PhysicalResourceClaim::new(ResourceKind::UploadSlot, plan.requirements.upload_slots),
        PhysicalResourceClaim::new(ResourceKind::UploadBytes, plan.requirements.h2d_bytes),
        PhysicalResourceClaim::new(ResourceKind::ResidentBytes, plan.resident_bytes),
    ]
}

fn install_claims(plan: MaterializationResourcePlan) -> [PhysicalResourceClaim; 2] {
    [
        PhysicalResourceClaim::new(ResourceKind::InstallSlot, plan.requirements.install_slots),
        PhysicalResourceClaim::new(
            ResourceKind::DeviceInstallBytes,
            plan.requirements.device_install_bytes,
        ),
    ]
}

fn terminal_post(outcome: &CompletionOutcome) -> Result<PostCompletion, RegistryError> {
    match outcome {
        CompletionOutcome::Succeeded => Err(RegistryError::Protocol {
            source: IoProtocolError::InvalidCompletionStage {
                stage: LoadStage::Resident,
            },
        }),
        CompletionOutcome::Failed(reason) => Ok(PostCompletion::Terminal {
            retirement: RetirementReason::Failed(reason.clone()),
            failure: Some(ContinuationFailure::Failed(reason.clone())),
        }),
        CompletionOutcome::Cancelled(reason) => Ok(PostCompletion::Terminal {
            retirement: RetirementReason::Cancelled(reason.clone()),
            failure: Some(ContinuationFailure::Cancelled(reason.clone())),
        }),
        CompletionOutcome::Stale(reason) => Ok(PostCompletion::Terminal {
            retirement: RetirementReason::Stale(reason.clone()),
            failure: Some(ContinuationFailure::Stale(reason.clone())),
        }),
    }
}
