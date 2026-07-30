//! Runtime-owned global single-flight materialization registry.

use std::collections::{BTreeSet, HashMap, VecDeque};

use ahash::RandomState;

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::io_protocol::{
    BackendId, CancellationReason, CompletionEvent, CompletionExpectation, CompletionGeneration,
    CompletionOutcome, CompletionTimestamp, ContinuationId, DependencySet, DeviceId,
    DispatchFenceContract, ExpertLeaseSet, FailureReason, FenceId, IoProtocolError, LoadKey,
    LoadStage, MappingEpoch, OperationId, RetirementReason, RetirementRecord, RetirementToken,
    StaleReason, ValidatedResidencyBinding, WaiterId,
};

use crate::io::backend::{MaterializationBackend, MaterializationReservation};
use crate::io::fairness::{FairQueue, FairQueueConfig, FairQueueError};
use crate::io::ledger::{CohortId, CriticalPathLedger, CriticalPhase, LedgerError};
use crate::io::waiters::{WaiterIndex, WaiterIndexError};
use crate::scheduling::{
    HardResourceBroker, HardResourceClaim, HardResourceError, HardResourceGrant, ResourceClass,
    ResourceKind,
};

const READ_CUSTODY: &[ResourceKind] = &[
    ResourceKind::Sqe,
    ResourceKind::PinnedSlab,
    ResourceKind::ReadBytes,
];
const READ_RELEASE: &[ResourceKind] = &[ResourceKind::Sqe, ResourceKind::ReadBytes];
const UPLOAD_CUSTODY: &[ResourceKind] = &[
    ResourceKind::UploadSlot,
    ResourceKind::UploadBytes,
    ResourceKind::ExpertFrame,
];
const UPLOAD_RELEASE: &[ResourceKind] = &[ResourceKind::UploadSlot, ResourceKind::UploadBytes];
const INSTALL_CUSTODY: &[ResourceKind] = &[ResourceKind::ExpertFrame];
const RESIDENCY_CLAIMS: &[ResourceKind] = &[ResourceKind::ExpertFrame];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadRequest {
    pub key: LoadKey,
    pub bytes: u64,
    pub class: ResourceClass,
}

impl LoadRequest {
    pub const fn new(key: LoadKey, bytes: u64, class: ResourceClass) -> Self {
        Self { key, bytes, class }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    Protocol(Box<IoProtocolError>),
    Resources(HardResourceError),
    Fairness(FairQueueError),
    Waiters(WaiterIndexError),
    Ledger(LedgerError),
    EmptyDependencySet,
    ZeroByteLoad(Box<LoadKey>),
    DuplicateDependency(Box<LoadKey>),
    ByteExpectationMismatch {
        key: Box<LoadKey>,
        expected: u64,
        requested: u64,
    },
    OperationIdExhausted,
    Backend(FailureReason),
    UnknownOperation(OperationId),
    UnknownContinuation(ContinuationId),
    ContinuationAlreadyReady(ContinuationId),
    RegistryShuttingDown,
    ShutdownIncomplete {
        pending_operations: usize,
        pending_completions: usize,
        active_grants: usize,
    },
    LostCompletion {
        operation: OperationId,
        stage: LoadStage,
        pending_operations: usize,
        active_grants: usize,
    },
    PublishedResidencyConflict(Box<LoadKey>),
    MissingResidency(Box<LoadKey>),
    ResumeLeaseAlreadyTaken(ContinuationId),
    ResumeContinuationMismatch {
        expected: ContinuationId,
        actual: ContinuationId,
    },
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "load registry error: {self:?}")
    }
}

impl std::error::Error for RegistryError {}

impl From<IoProtocolError> for RegistryError {
    fn from(value: IoProtocolError) -> Self {
        Self::Protocol(Box::new(value))
    }
}

impl From<HardResourceError> for RegistryError {
    fn from(value: HardResourceError) -> Self {
        Self::Resources(value)
    }
}

impl From<FairQueueError> for RegistryError {
    fn from(value: FairQueueError) -> Self {
        Self::Fairness(value)
    }
}

impl From<WaiterIndexError> for RegistryError {
    fn from(value: WaiterIndexError) -> Self {
        Self::Waiters(value)
    }
}

impl From<LedgerError> for RegistryError {
    fn from(value: LedgerError) -> Self {
        Self::Ledger(value)
    }
}

impl From<FailureReason> for RegistryError {
    fn from(value: FailureReason) -> Self {
        Self::Backend(value)
    }
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
        key: LoadKey,
        stage: LoadStage,
    },
    Rejected(CompletionRejectionReason),
    QueueEmpty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RegistryDriveStep {
    Idle,
    Progressed { key: Option<LoadKey> },
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
    key: LoadKey,
    bytes: u64,
    class: ResourceClass,
    stage: LoadStage,
    reservation: Option<MaterializationReservation>,
    logical_grant: Option<HardResourceGrant>,
    read_grant: Option<HardResourceGrant>,
    upload_grant: Option<HardResourceGrant>,
    retirement: Option<RetirementToken>,
    cancellation_reason: Option<CancellationReason>,
    install_submitted: bool,
    stage_started_ns: u64,
}

impl LoadOp {
    pub const fn operation(&self) -> OperationId {
        self.operation
    }

    pub const fn key(&self) -> LoadKey {
        self.key
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub const fn class(&self) -> ResourceClass {
        self.class
    }

    pub const fn stage(&self) -> LoadStage {
        self.stage
    }

    pub const fn cancellation_requested(&self) -> bool {
        self.cancellation_reason.is_some()
    }
}

#[derive(Debug)]
struct ResidentEntry {
    binding: ferrule_common::io_protocol::ResidencyBinding,
    grant: HardResourceGrant,
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
#[derive(Debug)]
pub struct ResumeLease {
    continuation: ContinuationId,
    leases: Option<ExpertLeaseSet>,
    grant: Option<HardResourceGrant>,
}

impl ResumeLease {
    pub const fn continuation(&self) -> ContinuationId {
        self.continuation
    }

    pub fn take(&mut self) -> Result<ExpertLeaseSet, RegistryError> {
        self.leases
            .take()
            .ok_or(RegistryError::ResumeLeaseAlreadyTaken(self.continuation))
    }
}

/// Per-continuation runtime state: resource grant, priority class, and ready admission.
/// Consolidated from three parallel BTreeMaps.
#[derive(Debug)]
struct ContinuationState {
    grant: HardResourceGrant,
    class: ResourceClass,
    ready_grant: Option<HardResourceGrant>,
}

/// Per-transaction tracking: operations and cohorts.
/// Consolidated from two parallel BTreeMaps.
#[derive(Debug, Default)]
struct TransactionState {
    operations: BTreeSet<OperationId>,
    cohorts: BTreeSet<CohortId>,
}

/// Runtime-wide authoritative owner of materialization, waiters, credits,
/// completion validation, publication, and retirement.
#[derive(Debug)]
pub struct LoadRegistry<B: MaterializationBackend> {
    backend: B,
    resources: HardResourceBroker,
    key_to_operation: HashMap<LoadKey, OperationId, RandomState>,
    operation_to_key: HashMap<OperationId, LoadKey, RandomState>,
    operations: HashMap<OperationId, LoadOp, RandomState>,
    retirements: HashMap<OperationId, RetirementRecord, RandomState>,
    stage_history: HashMap<OperationId, Vec<LoadStage>, RandomState>,
    residencies: HashMap<LoadKey, ResidentEntry, RandomState>,
    waiters: WaiterIndex,
    waiter_grants: HashMap<WaiterId, HardResourceGrant, RandomState>,
    ready_waiters: HashMap<ContinuationId, BTreeSet<WaiterId>, RandomState>,
    continuations: HashMap<ContinuationId, ContinuationState, RandomState>,
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

impl<B: MaterializationBackend> LoadRegistry<B> {
    pub fn new(
        backend: B,
        resources: HardResourceBroker,
        fairness: FairQueueConfig,
    ) -> Result<Self, RegistryError> {
        Ok(Self {
            backend,
            resources,
            key_to_operation: HashMap::default(),
            operation_to_key: HashMap::default(),
            operations: HashMap::default(),
            retirements: HashMap::default(),
            stage_history: HashMap::default(),
            residencies: HashMap::default(),
            waiters: WaiterIndex::new(),
            waiter_grants: HashMap::default(),
            ready_waiters: HashMap::default(),
            continuations: HashMap::default(),
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
    pub fn with_testing_resources(backend: B) -> Result<Self, RegistryError> {
        Self::new(
            backend,
            HardResourceBroker::testing_default(),
            FairQueueConfig::default(),
        )
    }

    pub const fn stats(&self) -> RegistryStats {
        self.stats
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    pub(crate) fn resources_mut(&mut self) -> &mut HardResourceBroker {
        &mut self.resources
    }

    pub fn resources(&self) -> &HardResourceBroker {
        &self.resources
    }

    pub fn acquire_hard_resources(
        &mut self,
        owner: u64,
        class: ResourceClass,
        claims: impl IntoIterator<Item = HardResourceClaim>,
    ) -> Result<HardResourceGrant, RegistryError> {
        Ok(self.resources.acquire(owner, class, claims)?)
    }

    pub fn release_hard_resources(
        &mut self,
        grant: &mut HardResourceGrant,
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

    pub fn runnable_actions(&self) -> usize {
        self.runnable.len()
    }

    pub fn pending_completions(&self) -> usize {
        self.completions.len()
    }

    pub fn resident_entries(&self) -> usize {
        self.residencies.len()
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

    pub fn operation_for_key(&self, key: LoadKey) -> Option<OperationId> {
        self.key_to_operation.get(&key).copied()
    }

    pub fn key_for_operation(&self, operation: OperationId) -> Option<LoadKey> {
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
        key: LoadKey,
    ) -> Option<ferrule_common::io_protocol::ResidencyBinding> {
        self.residencies.get(&key).map(|entry| entry.binding)
    }

    pub fn validated_residency(
        &self,
        key: LoadKey,
    ) -> Result<ValidatedResidencyBinding, RegistryError> {
        let binding = self
            .residency_binding(key)
            .ok_or_else(|| RegistryError::MissingResidency(Box::new(key)))?;
        Ok(ValidatedResidencyBinding::new(key, binding)?)
    }

    pub fn load_request(
        &self,
        key: LoadKey,
        class: ResourceClass,
    ) -> Result<LoadRequest, RegistryError> {
        let bytes = self.backend.materialization_bytes(key)?;
        Ok(LoadRequest::new(key, bytes, class))
    }

    /// Adopt an exact binding already selected as resident by the shared physical
    /// authority. This accounts the physical frame in the same registry used for
    /// newly materialized publications, so resume leases never bypass hard limits.
    pub fn adopt_residency(
        &mut self,
        request: LoadRequest,
        binding: ferrule_common::io_protocol::ResidencyBinding,
    ) -> Result<(), RegistryError> {
        if self.shutting_down {
            return Err(RegistryError::RegistryShuttingDown);
        }
        request.key.validate()?;
        if request.bytes == 0 {
            return Err(RegistryError::ZeroByteLoad(Box::new(request.key)));
        }
        let validated = ValidatedResidencyBinding::new(request.key, binding)?;
        if let Some(existing) = self.residencies.get(&request.key) {
            return if existing.binding == validated.binding() {
                Ok(())
            } else {
                Err(RegistryError::PublishedResidencyConflict(Box::new(
                    request.key,
                )))
            };
        }
        if self.key_to_operation.contains_key(&request.key) {
            return Err(RegistryError::PublishedResidencyConflict(Box::new(
                request.key,
            )));
        }
        let grant = self.resources.acquire(
            request.key.destination_generation().get(),
            request.class,
            [HardResourceClaim::new(
                ResourceKind::ExpertFrame,
                request.bytes,
            )],
        )?;
        self.residencies.insert(
            request.key,
            ResidentEntry {
                binding: validated.binding(),
                grant,
            },
        );
        Ok(())
    }

    pub fn rejected_completions(&self) -> &[CompletionRejection] {
        &self.rejected_completions
    }

    pub fn attach_waiter(
        &mut self,
        waiter: WaiterId,
        requests: impl IntoIterator<Item = LoadRequest>,
        now_ns: u64,
    ) -> Result<AttachReport, RegistryError> {
        if self.shutting_down {
            return Err(RegistryError::RegistryShuttingDown);
        }
        waiter.validate()?;
        if self.waiters.has_seen_waiter(waiter) {
            return Err(WaiterIndexError::DuplicateWaiter(waiter).into());
        }
        let continuation = waiter.continuation();
        if self.continuations.contains_key(&continuation)
            && self.waiters.unresolved_for(continuation).is_none()
        {
            return Err(RegistryError::ContinuationAlreadyReady(continuation));
        }

        let mut requests: Vec<_> = requests.into_iter().collect();
        requests.sort_unstable_by_key(|request| request.key);
        for request in &requests {
            request.key.validate()?;
            if request.bytes == 0 {
                return Err(RegistryError::ZeroByteLoad(Box::new(request.key)));
            }
            if request.bytes > self.runnable.config().max_transition_cost {
                return Err(FairQueueError::TransitionTooLarge {
                    cost: request.bytes,
                    maximum: self.runnable.config().max_transition_cost,
                }
                .into());
            }
            self.validate_operation_capacity(request.bytes)?;
        }
        if let Some(duplicate) = requests
            .windows(2)
            .find(|window| window[0].key == window[1].key)
        {
            return Err(RegistryError::DuplicateDependency(Box::new(
                duplicate[0].key,
            )));
        }
        let class = requests
            .iter()
            .map(|request| request.class)
            .max()
            .unwrap_or(ResourceClass::Decode);

        let mut joined = Vec::new();
        let mut new_requests = Vec::new();
        let mut already_resident = 0;
        for request in &requests {
            if self.residencies.contains_key(&request.key) {
                already_resident += 1;
                continue;
            }
            if let Some(operation) = self.key_to_operation.get(&request.key).copied() {
                let active = self
                    .operations
                    .get(&operation)
                    .expect("active key index must point to an operation");
                if active.bytes != request.bytes {
                    return Err(RegistryError::ByteExpectationMismatch {
                        key: Box::new(request.key),
                        expected: active.bytes,
                        requested: request.bytes,
                    });
                }
                joined.push(operation);
            } else {
                new_requests.push(*request);
            }
        }

        let continuation_is_new = !self.continuations.contains_key(&continuation);
        let mut preflight = vec![HardResourceClaim::new(ResourceKind::Waiter, 1)];
        if continuation_is_new {
            preflight.push(HardResourceClaim::new(ResourceKind::Continuation, 1));
        }
        self.resources.can_acquire(class, preflight)?;

        for operation in &joined {
            let active = self
                .operations
                .get_mut(operation)
                .expect("joined operation remains active");
            if class > active.class {
                if let Some(grant) = active.logical_grant.as_ref() {
                    self.resources.promote(grant, class)?;
                }
                if let Some(grant) = active.read_grant.as_ref() {
                    self.resources.promote(grant, class)?;
                }
                if let Some(grant) = active.upload_grant.as_ref() {
                    self.resources.promote(grant, class)?;
                }
                active.class = class;
            }
        }

        let mut created = Vec::new();
        for request in new_requests {
            match self.create_operation(request, now_ns) {
                Ok(operation) => created.push(operation),
                Err(error) => {
                    while let Some(operation) = created.pop() {
                        self.rollback_reserved_operation(operation, now_ns)?;
                    }
                    return Err(error);
                }
            }
        }

        if continuation_is_new {
            let grant = self.resources.acquire(
                continuation.get(),
                class,
                [HardResourceClaim::new(ResourceKind::Continuation, 1)],
            )?;
            self.continuations.insert(
                continuation,
                ContinuationState {
                    grant,
                    class,
                    ready_grant: None,
                },
            );
        } else {
            let state = self
                .continuations
                .get_mut(&continuation)
                .expect("continuation state is created together with grant");
            if class > state.class {
                self.resources.promote(&state.grant, class)?;
                state.class = class;
            }
        }
        let waiter_grant = self.resources.acquire(
            continuation.get(),
            class,
            [HardResourceClaim::new(ResourceKind::Waiter, 1)],
        )?;

        let operations: Vec<_> = joined.iter().chain(&created).copied().collect();
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
        let continuation_ready = self.waiters.register(waiter, operations.clone())?;
        self.waiter_grants.insert(waiter, waiter_grant);
        if continuation_ready {
            self.ready_waiters
                .entry(continuation)
                .or_default()
                .insert(waiter);
            self.try_admit_ready(continuation)?;
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
            .ok_or(WaiterIndexError::UnknownWaiter(waiter))?;
        for operation in &operations {
            self.close_wait(waiter, *operation, now_ns)?;
        }
        let detached = self.waiters.detach_waiter(waiter)?;
        if let Some(mut grant) = self.waiter_grants.remove(&waiter) {
            self.resources.release_all_held(&mut grant)?;
        }
        if let Some(continuation) = detached.continuation_became_empty {
            self.release_continuation(continuation)?;
        }
        for operation in detached.operations_losing_last_waiter {
            self.handle_last_waiter(operation, reason.clone(), now_ns)?;
        }
        Ok(())
    }

    /// Schedules one hard-feasible owner transition. Stage claims are admitted
    /// only when their action reaches the head of the fair queue.
    pub fn schedule_one(&mut self, now_ns: u64) -> Result<bool, RegistryError> {
        Ok(self.schedule_one_with_key(now_ns)?.is_some())
    }

    fn schedule_one_with_key(&mut self, now_ns: u64) -> Result<Option<LoadKey>, RegistryError> {
        let operations = &self.operations;
        self.runnable.retain(|action| {
            operations
                .get(&action.operation)
                .is_some_and(|operation| action_matches(operation, action.kind))
        });
        let operations = &self.operations;
        let resources = &self.resources;
        let Some(action) = self.runnable.pop_next(now_ns, |action| {
            let Some(operation) = operations.get(&action.operation) else {
                return false;
            };
            match action.kind {
                LoadActionKind::Reserve => resources
                    .can_acquire(
                        operation.class,
                        std::iter::once(HardResourceClaim::new(ResourceKind::LoadOperation, 1))
                            .chain(read_claims(operation.bytes)),
                    )
                    .is_ok(),
                LoadActionKind::SubmitUpload => resources
                    .can_acquire(operation.class, upload_claims(operation.bytes))
                    .is_ok(),
                LoadActionKind::SubmitRead | LoadActionKind::PollInstall => true,
            }
        }) else {
            return Ok(None);
        };
        let mut operation = self
            .operations
            .remove(&action.operation)
            .expect("retained runnable action must reference an active operation");
        let key = operation.key;
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
                self.fail_operation_waiters(
                    operation.operation,
                    ContinuationFailure::Failed(failure.clone()),
                    now_ns,
                )?;
                self.retire_owned(operation, RetirementReason::Failed(failure), now_ns)?;
            }
            Ok(None) => {
                self.operations.insert(operation.operation, operation);
            }
            Err(error) => {
                self.operations.insert(operation.operation, operation);
                return Err(error);
            }
        }
        Ok(Some(key))
    }

    pub fn enqueue_completion(&mut self, event: CompletionEvent) {
        self.completions.push_back(event);
    }

    pub fn collect_backend_completions(&mut self, maximum: usize) -> usize {
        let mut collected = 0;
        while collected < maximum {
            let Some(event) = self.backend.next_completion() else {
                break;
            };
            self.completions.push_back(event);
            collected += 1;
        }
        collected
    }

    pub fn process_one_completion(&mut self) -> Result<CompletionDisposition, RegistryError> {
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
        let expectation =
            CompletionExpectation::new(active.operation, active.key, active.stage, active.bytes)?;
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
        let timestamp = event.timestamp.as_nanos();
        let post = match self.apply_completion(&mut operation, &event) {
            Ok(post) => post,
            Err(error) => {
                self.operations.insert(operation_id, operation);
                return Err(error);
            }
        };
        self.stats.physical_completions = self.stats.physical_completions.saturating_add(1);
        match post {
            PostCompletion::Keep => {
                let retry_cancellation = (self.waiters.waiter_count(operation_id) == 0)
                    .then(|| operation.cancellation_reason.clone())
                    .flatten();
                self.operations.insert(operation_id, operation);
                if let Some(reason) = retry_cancellation {
                    self.handle_last_waiter(operation_id, reason, timestamp)?;
                }
            }
            PostCompletion::Publish => {
                self.publish(operation, timestamp)?;
            }
            PostCompletion::Terminal {
                retirement,
                failure,
            } => {
                if let Some(failure) = failure {
                    self.fail_operation_waiters(operation_id, failure, timestamp)?;
                }
                self.retire_owned(operation, retirement, timestamp)?;
            }
        }
        Ok(CompletionDisposition::Applied {
            operation: operation_id,
            key: completed_key,
            stage: completed_stage,
        })
    }

    pub(crate) fn drive_one(&mut self, now_ns: u64) -> Result<RegistryDriveStep, RegistryError> {
        if let Some(key) = self.schedule_one_with_key(now_ns)? {
            return Ok(RegistryDriveStep::Progressed { key: Some(key) });
        }
        self.collect_backend_completions(1);
        if self.completions.is_empty() {
            return Ok(RegistryDriveStep::Idle);
        }
        let key = match self.process_one_completion()? {
            CompletionDisposition::Applied { key, .. } => Some(key),
            CompletionDisposition::Rejected(_) | CompletionDisposition::QueueEmpty => None,
        };
        Ok(RegistryDriveStep::Progressed { key })
    }

    /// Drives commands and completions deterministically until no immediate work
    /// remains or the transition budget is exhausted.
    pub fn drive(&mut self, now_ns: u64, maximum: usize) -> Result<usize, RegistryError> {
        let mut progressed = 0;
        while progressed < maximum {
            let mut made_progress = false;
            if self.schedule_one(now_ns)? {
                progressed += 1;
                made_progress = true;
                if progressed == maximum {
                    break;
                }
            }
            self.collect_backend_completions(maximum - progressed);
            if !self.completions.is_empty() {
                let _ = self.process_one_completion()?;
                progressed += 1;
                made_progress = true;
            }
            if !made_progress {
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
            return Err(RegistryError::UnknownContinuation(continuation));
        }
        let required = dependencies
            .iter()
            .filter_map(|dependency| dependency.load_key())
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
                        return Err(RegistryError::Protocol(Box::new(
                            IoProtocolError::ResidencyMismatch {
                                field: "resume lease backend/device",
                            },
                        )));
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
        let leases = ExpertLeaseSet::new(
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
        let class = self
            .continuations
            .get(&continuation)
            .map(|state| state.class)
            .ok_or(RegistryError::UnknownContinuation(continuation))?;
        let grant = self.resources.acquire(
            continuation.get(),
            class,
            [
                HardResourceClaim::new(ResourceKind::Arena, 1),
                HardResourceClaim::new(ResourceKind::Lease, required.len() as u64),
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
        continuation: ContinuationId,
        mut lease: ResumeLease,
        started_ns: u64,
        finished_ns: u64,
    ) -> Result<(), RegistryError> {
        if lease.continuation != continuation {
            return Err(RegistryError::ResumeContinuationMismatch {
                expected: lease.continuation,
                actual: continuation,
            });
        }
        if lease.leases.is_some() {
            return Err(RegistryError::ResumeLeaseAlreadyTaken(continuation));
        }
        if let Some(mut grant) = lease.grant.take() {
            self.resources.release_all_held(&mut grant)?;
        }
        self.ledger.record_cohort_phase(
            CohortId::new(continuation.get()),
            CriticalPhase::Resume,
            started_ns,
            finished_ns.max(started_ns),
        )?;
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
        for operation in std::mem::take(&mut lost_last) {
            self.handle_last_waiter(operation, reason.clone(), now_ns)?;
        }
        Ok(())
    }

    pub fn record_commit(
        &mut self,
        transaction: ExecutionTransactionId,
        started_ns: u64,
        finished_ns: u64,
    ) -> Result<(), RegistryError> {
        let cohort = CohortId::new(transaction.get());
        self.transactions
            .entry(transaction)
            .or_default()
            .cohorts
            .insert(cohort);
        self.ledger.record_cohort_phase(
            cohort,
            CriticalPhase::Commit,
            started_ns,
            finished_ns.max(started_ns),
        )?;
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

    pub fn evict(&mut self, key: LoadKey) -> Result<bool, RegistryError> {
        let Some(mut resident) = self.residencies.remove(&key) else {
            return Ok(false);
        };
        self.resources.release_all_held(&mut resident.grant)?;
        Ok(true)
    }

    pub fn begin_shutdown(&mut self, now_ns: u64) -> Result<(), RegistryError> {
        self.shutting_down = true;
        let waiters: Vec<_> = self.waiters.active_waiters().collect();
        for waiter in waiters {
            self.detach_waiter(waiter, CancellationReason::OwnerShutdown, now_ns)?;
        }
        let operations: Vec<_> = self.operations.keys().copied().collect();
        for operation in operations {
            if self.waiters.waiter_count(operation) == 0 {
                self.handle_last_waiter(operation, CancellationReason::OwnerShutdown, now_ns)?;
            }
        }
        let continuations: Vec<_> = self.continuations.keys().copied().collect();
        for continuation in continuations {
            self.detach_continuation(continuation, CancellationReason::OwnerShutdown, now_ns)?;
        }
        let resident: Vec<_> = self.residencies.keys().copied().collect();
        for key in resident {
            self.evict(key)?;
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
            self.collect_backend_completions(maximum_completions - processed);
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
            && self.waiters.is_empty();
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

    fn validate_operation_capacity(&self, bytes: u64) -> Result<(), RegistryError> {
        for claim in read_claims(bytes).into_iter().chain(upload_claims(bytes)) {
            let capacity = self
                .resources
                .snapshots()
                .find(|snapshot| snapshot.kind == claim.kind)
                .expect("hard resource catalog contains every resource kind")
                .capacity;
            if claim.amount > capacity {
                return Err(HardResourceError::ExceedsCapacity {
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
                bytes: request.bytes,
                class: request.class,
                stage: LoadStage::Reserved,
                reservation: None,
                logical_grant: None,
                read_grant: None,
                upload_grant: None,
                retirement: Some(retirement),
                cancellation_reason: None,
                install_submitted: false,
                stage_started_ns: now_ns,
            },
        );
        self.runnable.push(
            LoadAction {
                operation,
                kind: LoadActionKind::Reserve,
            },
            request.class,
            request.bytes,
            now_ns,
        )?;
        Ok(operation)
    }

    fn rollback_reserved_operation(
        &mut self,
        operation: OperationId,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        let Some(operation) = self.operations.remove(&operation) else {
            return Ok(());
        };
        if operation.reservation.is_some() {
            self.backend.cancel(
                operation.operation,
                operation.key,
                LoadStage::Reserved,
                CancellationReason::Superseded,
            )?;
        }
        self.retire_owned(
            operation,
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
            operation.class,
            [HardResourceClaim::new(ResourceKind::LoadOperation, 1)],
        )?;
        operation.logical_grant = Some(logical_grant);
        let mut grant = match self.resources.acquire(
            operation.operation.get(),
            operation.class,
            read_claims(operation.bytes),
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
                .backend
                .reserve(operation.operation, operation.key, operation.bytes)
            {
                Ok(reservation) => reservation,
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
            operation.class,
            operation.bytes,
            now_ns,
        )?;
        Ok(None)
    }

    fn submit_read(&mut self, operation: &mut LoadOp, now_ns: u64) -> Result<(), RegistryError> {
        debug_assert_eq!(operation.stage, LoadStage::Reserved);
        let rejection = self
            .backend
            .submit_read(
                operation.operation,
                operation.key,
                operation
                    .reservation
                    .as_ref()
                    .expect("submit-read action requires a physical reservation"),
                operation.bytes,
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
            operation.class,
            upload_claims(operation.bytes),
        )?;
        operation.upload_grant = Some(grant);
        let rejection = self
            .backend
            .submit_upload(
                operation.operation,
                operation.key,
                operation
                    .reservation
                    .as_ref()
                    .expect("host-ready operation owns a physical reservation"),
                operation.bytes,
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
                .expect("host-ready operation owns pinned-slab credit"),
            &[ResourceKind::PinnedSlab],
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
        let rejection = self
            .backend
            .poll_install(
                operation.operation,
                operation.key,
                operation
                    .reservation
                    .as_ref()
                    .expect("installing operation owns a physical reservation"),
                operation.bytes,
            )
            .err();
        self.resources.mark_submitted(
            operation
                .upload_grant
                .as_mut()
                .expect("installing operation owns a destination grant"),
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
    ) -> Result<PostCompletion, RegistryError> {
        let next = event.next_stage()?;
        let timestamp = event.timestamp.as_nanos().max(operation.stage_started_ns);
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
                        operation.class,
                        operation.bytes,
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
                    .mark_returned(read_grant, &[ResourceKind::PinnedSlab])?;
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
                        operation.class,
                        1,
                        timestamp,
                    )?;
                    Ok(PostCompletion::Keep)
                } else {
                    terminal_post(&event.outcome)
                }
            }
            LoadStage::Installing => {
                let grant = operation
                    .upload_grant
                    .as_mut()
                    .expect("installing operation owns a destination grant");
                self.resources.mark_returned(grant, INSTALL_CUSTODY)?;
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

    fn publish(&mut self, mut operation: LoadOp, timestamp: u64) -> Result<(), RegistryError> {
        let binding = operation
            .reservation
            .as_ref()
            .expect("resident operation owns a physical reservation")
            .binding();
        let _validated =
            ferrule_common::io_protocol::ValidatedResidencyBinding::new(operation.key, binding)?;
        if self.residencies.contains_key(&operation.key) {
            return Err(RegistryError::PublishedResidencyConflict(Box::new(
                operation.key,
            )));
        }
        let mut grant = operation
            .upload_grant
            .take()
            .expect("resident operation owns its destination grant");
        let release: Vec<_> = grant
            .claims()
            .map(|claim| claim.kind)
            .filter(|kind| !RESIDENCY_CLAIMS.contains(kind))
            .collect();
        self.resources.release_held(&mut grant, &release)?;
        self.residencies
            .insert(operation.key, ResidentEntry { binding, grant });
        self.resolve_operation_waiters(operation.operation, timestamp)?;
        self.stats.publications = self.stats.publications.saturating_add(1);
        self.retire_owned(
            operation,
            RetirementReason::ResidentOwnershipTransferred,
            timestamp,
        )
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
        for continuation in resolution.ready_continuations {
            self.try_admit_ready(continuation)?;
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
        for other in lost_last {
            if other != operation {
                self.handle_last_waiter(other, CancellationReason::Superseded, now_ns)?;
            }
        }
        Ok(())
    }

    fn handle_last_waiter(
        &mut self,
        operation_id: OperationId,
        reason: CancellationReason,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        if self.waiters.waiter_count(operation_id) != 0 {
            return Ok(());
        }
        let Some(mut operation) = self.operations.remove(&operation_id) else {
            return Ok(());
        };
        match operation.stage {
            LoadStage::Reserved => {
                if operation.reservation.is_some() {
                    self.backend.cancel(
                        operation.operation,
                        operation.key,
                        LoadStage::Reserved,
                        reason.clone(),
                    )?;
                }
                self.retire_owned(operation, RetirementReason::Cancelled(reason), now_ns)
            }
            LoadStage::HostReady => {
                self.backend.cancel(
                    operation.operation,
                    operation.key,
                    LoadStage::HostReady,
                    reason.clone(),
                )?;
                operation
                    .reservation
                    .as_mut()
                    .expect("host-ready operation owns a physical reservation")
                    .retire_slabs()?;
                self.transition(&mut operation, LoadStage::Stale)?;
                self.retire_owned(operation, RetirementReason::Cancelled(reason), now_ns)
            }
            LoadStage::Installing if !operation.install_submitted => {
                self.backend.cancel(
                    operation.operation,
                    operation.key,
                    LoadStage::Installing,
                    reason.clone(),
                )?;
                self.transition(&mut operation, LoadStage::Stale)?;
                self.retire_owned(operation, RetirementReason::Cancelled(reason), now_ns)
            }
            LoadStage::ReadSubmitted | LoadStage::UploadSubmitted | LoadStage::Installing => {
                if operation.cancellation_reason.is_none() {
                    if let Err(error) = self.backend.cancel(
                        operation.operation,
                        operation.key,
                        operation.stage,
                        reason.clone(),
                    ) {
                        self.operations.insert(operation_id, operation);
                        return Err(RegistryError::Backend(error));
                    }
                    operation.cancellation_reason = Some(reason);
                    self.stats.cancellations_requested =
                        self.stats.cancellations_requested.saturating_add(1);
                }
                self.operations.insert(operation_id, operation);
                Ok(())
            }
            LoadStage::Failed | LoadStage::Stale | LoadStage::Draining => {
                self.retire_owned(operation, RetirementReason::Cancelled(reason), now_ns)
            }
            LoadStage::Resident | LoadStage::Retired => {
                self.operations.insert(operation_id, operation);
                Ok(())
            }
        }
    }

    fn retire_owned(
        &mut self,
        mut operation: LoadOp,
        reason: RetirementReason,
        now_ns: u64,
    ) -> Result<(), RegistryError> {
        for grant in [
            &mut operation.read_grant,
            &mut operation.upload_grant,
            &mut operation.logical_grant,
        ] {
            if let Some(mut grant) = grant.take() {
                self.resources.release_all_held(&mut grant)?;
            }
        }
        self.transition(&mut operation, LoadStage::Retired)?;
        let record = operation
            .retirement
            .as_mut()
            .expect("active operation owns retirement authority")
            .consume(reason, CompletionTimestamp::from_nanos(now_ns))?;
        if self.key_to_operation.get(&operation.key) == Some(&operation.operation) {
            self.key_to_operation.remove(&operation.key);
        }
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
        if let Some(state) = self.continuations.remove(&continuation) {
            if let Some(mut ready) = state.ready_grant {
                self.resources.release_all_held(&mut ready)?;
            }
            let mut grant = state.grant;
            self.resources.release_all_held(&mut grant)?;
        }
        Ok(())
    }

    fn try_admit_ready(&mut self, continuation: ContinuationId) -> Result<bool, RegistryError> {
        let state = self
            .continuations
            .get(&continuation)
            .ok_or(RegistryError::UnknownContinuation(continuation))?;
        if state.ready_grant.is_some() {
            return Ok(true);
        }
        let class = state.class;
        match self.resources.acquire(
            continuation.get(),
            class,
            [HardResourceClaim::new(ResourceKind::ReadyCohort, 1)],
        ) {
            Ok(grant) => {
                self.continuations
                    .get_mut(&continuation)
                    .expect("continuation state was just read")
                    .ready_grant = Some(grant);
                Ok(true)
            }
            Err(
                HardResourceError::TemporarilyUnavailable { .. }
                | HardResourceError::DemandReserve { .. }
                | HardResourceError::ExceedsCapacity { .. },
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

fn read_claims(bytes: u64) -> [HardResourceClaim; 3] {
    [
        HardResourceClaim::new(ResourceKind::Sqe, 1),
        HardResourceClaim::new(ResourceKind::PinnedSlab, bytes),
        HardResourceClaim::new(ResourceKind::ReadBytes, bytes),
    ]
}

fn upload_claims(bytes: u64) -> [HardResourceClaim; 3] {
    [
        HardResourceClaim::new(ResourceKind::UploadSlot, 1),
        HardResourceClaim::new(ResourceKind::UploadBytes, bytes),
        HardResourceClaim::new(ResourceKind::ExpertFrame, bytes),
    ]
}

fn terminal_post(outcome: &CompletionOutcome) -> Result<PostCompletion, RegistryError> {
    match outcome {
        CompletionOutcome::Succeeded => Err(RegistryError::Protocol(Box::new(
            IoProtocolError::InvalidCompletionStage {
                stage: LoadStage::Resident,
            },
        ))),
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
