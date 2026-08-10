//! Deterministic materialization providers, physical mocks, and fixture
//! constructors shared by the crate's own tests and downstream integration
//! tests. Everything here is side-effect-free and scheduling-explicit.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::io_protocol::{
    BackendId, CancellationReason, CompletionEvent, CompletionGeneration, CompletionOutcome,
    CompletionTimestamp, ContentHash, ContinuationId, DependencySetEpoch, DestinationGeneration,
    DestinationSlotId, DeviceId, ExpertId, FailureReason, FenceId, LayerId, LoadStage,
    MaterializationKey, MaterializedResourceId, ModelInstanceId, OperationId, PayloadEncodingId,
    RegisteredPinnedAlignedSlabLease, RegisteredPinnedAlignedSlabLeaseDescriptor, RegistrationId,
    RequestGeneration, ResidencyBinding, SlabId, SourceGeneration, SourceIdentityHash,
    UploadFenceContract, WaiterId,
};
use ferrule_common::materialization_io::{
    MaterializationResourceLimits, MaterializationResourcePlan, MaterializationResourceRequirements,
};
use ferrule_model::{
    MaterializationPlacement, MaterializationPreparation, MaterializationProvider,
    MaterializationPurpose, MaterializationRequest, MaterializationResident,
    MaterializationTransfer, PhysicalMaterializationOperationReservation,
    PhysicalMaterializationTopology, ResourceSource,
};

use super::{
    CompletionDisposition, ExecutionPromotion, FairQueueConfig, LoadRegistry, LoadRequest,
    MaterializationOperationReservation, RuntimeMaterializationProvider,
    SharedMaterializationProvider,
};
use crate::scheduling::{
    ExecutionPhase, PhysicalResourceBroker, PhysicalResourceLimit, ResourceDemand, ResourceKind,
};

pub const BYTES: u64 = 4096;

pub fn retained_request(
    preparation: MaterializationPreparation,
    plan: MaterializationResourcePlan,
    demand: ResourceDemand,
    retention: ferrule_model::ResourceRetention,
) -> LoadRequest {
    LoadRequest::new(preparation, plan, demand, retention)
}

pub fn stage_request(
    preparation: MaterializationPreparation,
    plan: MaterializationResourcePlan,
    demand: ResourceDemand,
) -> LoadRequest {
    retained_request(
        preparation,
        plan,
        demand,
        ferrule_model::ResourceRetention::ThroughStage,
    )
}
pub const DESTINATION_GENERATION: u64 = 37;
pub const SLOT: u32 = 9;
pub const SLAB_ID_OFFSET: u64 = 1000;
pub const REGISTRATION_ID_OFFSET: u64 = 2000;
pub const FENCE_ID_OFFSET: u64 = 3000;

pub fn uniform_plan() -> MaterializationResourcePlan {
    MaterializationResourcePlan::uniform_payload(BYTES)
        .expect("mock uniform materialization plan must be valid")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockPhysicalCommand {
    Prepare(MaterializationRequest),
    Prepared(MaterializationKey),
    PromoteToExecution(MaterializationKey),
    DiscardPreparation(MaterializationKey),
    MaterializationPlan(MaterializationKey),
    ReleaseExecutionLease(MaterializationKey),
    Reserve(OperationId, MaterializationKey, MaterializationResourcePlan),
    SubmitRead(OperationId, MaterializationKey, MaterializationResourcePlan),
    SubmitUpload(OperationId, MaterializationKey, MaterializationResourcePlan),
    PollInstall(OperationId, MaterializationKey, MaterializationResourcePlan),
    Cancel(
        OperationId,
        MaterializationKey,
        LoadStage,
        CancellationReason,
    ),
    PhysicalDropped,
}

#[derive(Debug, Clone, Copy)]
pub struct MockPreparation {
    key: MaterializationKey,
    binding: ResidencyBinding,
    evicted: Option<MaterializationKey>,
}

#[derive(Debug)]
pub struct MockPhysicalState {
    placement: MaterializationPlacement,
    limits: MaterializationResourceLimits,
    plan: MaterializationResourcePlan,
    generation: DestinationGeneration,
    automatic: bool,
    resident: bool,
    resolve_failure: Option<FailureReason>,
    reserve_failure: Option<FailureReason>,
    promotion_failures: BTreeMap<MaterializationKey, VecDeque<FailureReason>>,
    release_failures: VecDeque<FailureReason>,
    reservation_key_override: Option<MaterializationKey>,
    reservation_operation_override: Option<OperationId>,
    reservation_slot_override: Option<DestinationSlotId>,
    next_preparation: Option<(
        DestinationGeneration,
        DestinationSlotId,
        Option<MaterializationKey>,
    )>,
    resolved: BTreeMap<MaterializationRequest, MockPreparation>,
    resident_keys: BTreeSet<MaterializationKey>,
    commands: Vec<MockPhysicalCommand>,
    completions: VecDeque<CompletionEvent>,
    scripted_outcomes: BTreeMap<LoadStage, VecDeque<CompletionOutcome>>,
    lost_completions: BTreeMap<LoadStage, usize>,
    clock_ns: u64,
}

impl MockPhysicalState {
    pub fn new(automatic: bool) -> Self {
        let placement = MaterializationPlacement::new(
            ModelInstanceId::new(17),
            BackendId::new(4),
            DeviceId::new(2),
        )
        .unwrap();
        let operation_capacity = 4;
        let byte_capacity = BYTES * operation_capacity;
        let operation_reserve = 1;
        Self {
            placement,
            limits: MaterializationResourceLimits {
                capacity: MaterializationResourceRequirements {
                    read_slots: operation_capacity,
                    storage_read_bytes: byte_capacity,
                    pinned_host_bytes: byte_capacity,
                    upload_slots: operation_capacity,
                    h2d_bytes: byte_capacity,
                    install_slots: operation_capacity,
                    device_install_bytes: byte_capacity,
                },
                execution_reserve: MaterializationResourceRequirements {
                    read_slots: operation_reserve,
                    storage_read_bytes: BYTES,
                    pinned_host_bytes: BYTES,
                    upload_slots: operation_reserve,
                    h2d_bytes: BYTES,
                    install_slots: operation_reserve,
                    device_install_bytes: BYTES,
                },
            },
            plan: uniform_plan(),
            generation: DestinationGeneration::new(DESTINATION_GENERATION),
            automatic,
            resident: false,
            resolve_failure: None,
            reserve_failure: None,
            promotion_failures: BTreeMap::new(),
            release_failures: VecDeque::new(),
            reservation_key_override: None,
            reservation_operation_override: None,
            reservation_slot_override: None,
            next_preparation: None,
            resolved: BTreeMap::new(),
            resident_keys: BTreeSet::new(),
            commands: Vec::new(),
            completions: VecDeque::new(),
            scripted_outcomes: BTreeMap::new(),
            lost_completions: BTreeMap::new(),
            clock_ns: 1,
        }
    }

    pub fn binding_for(&self, key: MaterializationKey) -> ResidencyBinding {
        let resource = key.resource();
        let slot = resource
            .group()
            .wrapping_mul(1024)
            .wrapping_add(resource.item())
            .wrapping_add(SLOT);
        ResidencyBinding::new(
            key.model(),
            resource,
            key.backend(),
            key.device(),
            DestinationSlotId::new(slot),
            key.destination_generation(),
        )
    }

    pub fn emit(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        stage: LoadStage,
        plan: MaterializationResourcePlan,
        default_outcome: CompletionOutcome,
    ) {
        if let Some(remaining) = self.lost_completions.get_mut(&stage)
            && *remaining != 0
        {
            *remaining -= 1;
            return;
        }
        let scripted = self
            .scripted_outcomes
            .get_mut(&stage)
            .and_then(VecDeque::pop_front);
        if !self.automatic && scripted.is_none() {
            return;
        }
        let outcome = scripted.unwrap_or(default_outcome);
        if stage == LoadStage::Installing && matches!(outcome, CompletionOutcome::Succeeded) {
            let evicted = self
                .resolved
                .values()
                .find(|preparation| preparation.key == key)
                .and_then(|preparation| preparation.evicted);
            if let Some(evicted) = evicted {
                self.resident_keys.remove(&evicted);
            }
            self.resident_keys.insert(key);
        }
        let bytes = if matches!(outcome, CompletionOutcome::Succeeded) {
            plan.completion_bytes(stage)
                .expect("mock success completion requires a submitted stage")
        } else {
            0
        };
        let timestamp = CompletionTimestamp::from_nanos(self.clock_ns);
        self.clock_ns = self.clock_ns.saturating_add(1);
        self.completions.push_back(CompletionEvent::new(
            operation,
            key,
            stage,
            outcome,
            bytes,
            CompletionGeneration::for_key(key),
            timestamp,
        ));
    }
}

#[derive(Debug)]
pub struct MockPhysicalProvider {
    state: Arc<Mutex<MockPhysicalState>>,
}

#[derive(Debug, Clone)]
pub struct MockPhysicalHandle {
    state: Arc<Mutex<MockPhysicalState>>,
}

impl Drop for MockPhysicalProvider {
    fn drop(&mut self) {
        self.lock()
            .commands
            .push(MockPhysicalCommand::PhysicalDropped);
    }
}

impl MockPhysicalProvider {
    pub fn automatic() -> (Self, MockPhysicalHandle) {
        Self::new(true)
    }

    pub fn manual() -> (Self, MockPhysicalHandle) {
        Self::new(false)
    }

    pub fn new(automatic: bool) -> (Self, MockPhysicalHandle) {
        let state = Arc::new(Mutex::new(MockPhysicalState::new(automatic)));
        (
            Self {
                state: Arc::clone(&state),
            },
            MockPhysicalHandle { state },
        )
    }

    pub fn lock(&self) -> MutexGuard<'_, MockPhysicalState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl MockPhysicalHandle {
    pub fn lock(&self) -> MutexGuard<'_, MockPhysicalState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn placement(&self) -> MaterializationPlacement {
        self.lock().placement
    }

    pub fn limits(&self) -> MaterializationResourceLimits {
        self.lock().limits
    }

    pub fn set_bytes_and_limits(&self, bytes: u64, limits: MaterializationResourceLimits) {
        self.set_plan_and_limits(
            MaterializationResourcePlan::uniform_payload(bytes)
                .expect("mock uniform materialization plan must be valid"),
            limits,
        );
    }

    pub fn set_plan_and_limits(
        &self,
        plan: MaterializationResourcePlan,
        limits: MaterializationResourceLimits,
    ) {
        let mut state = self.lock();
        state.plan = plan;
        state.limits = limits;
    }

    pub fn command_count(&self, predicate: impl Fn(&MockPhysicalCommand) -> bool) -> usize {
        self.lock()
            .commands
            .iter()
            .filter(|command| predicate(command))
            .count()
    }

    pub fn commands(&self) -> Vec<MockPhysicalCommand> {
        self.lock().commands.clone()
    }

    pub fn binding(&self, key: MaterializationKey) -> ResidencyBinding {
        let state = self.lock();
        state
            .resolved
            .values()
            .find(|preparation| preparation.key == key)
            .map(|preparation| preparation.binding)
            .unwrap_or_else(|| state.binding_for(key))
    }

    pub fn configure_next_preparation(
        &self,
        generation: u64,
        slot: DestinationSlotId,
        evicted: Option<MaterializationKey>,
    ) {
        self.lock().next_preparation =
            Some((DestinationGeneration::new(generation), slot, evicted));
    }

    pub fn set_resident(&self, resident: bool) {
        self.lock().resident = resident;
    }

    pub fn fail_next_resolve(&self, failure: FailureReason) {
        self.lock().resolve_failure = Some(failure);
    }

    pub fn fail_next_reserve(&self, failure: FailureReason) {
        self.lock().reserve_failure = Some(failure);
    }

    pub fn fail_next_promotion(&self, key: MaterializationKey, failure: FailureReason) {
        self.lock()
            .promotion_failures
            .entry(key)
            .or_default()
            .push_back(failure);
    }

    pub fn fail_next_release(&self, failure: FailureReason) {
        self.lock().release_failures.push_back(failure);
    }

    pub fn override_reservation_key(&self, key: MaterializationKey) {
        self.lock().reservation_key_override = Some(key);
    }

    pub fn override_reservation_operation(&self, operation: OperationId) {
        self.lock().reservation_operation_override = Some(operation);
    }

    pub fn override_reservation_slot(&self, slot: DestinationSlotId) {
        self.lock().reservation_slot_override = Some(slot);
    }

    pub fn lose_next(&self, stage: LoadStage) {
        *self.lock().lost_completions.entry(stage).or_default() += 1;
    }

    pub fn script_outcome(&self, stage: LoadStage, outcome: CompletionOutcome) {
        self.lock()
            .scripted_outcomes
            .entry(stage)
            .or_default()
            .push_back(outcome);
    }

    pub fn push_completion(&self, completion: CompletionEvent) {
        self.lock().completions.push_back(completion);
    }

    pub fn push_outcome(
        &self,
        operation: OperationId,
        key: MaterializationKey,
        stage: LoadStage,
        outcome: CompletionOutcome,
    ) -> CompletionEvent {
        let mut state = self.lock();
        let bytes = if matches!(outcome, CompletionOutcome::Succeeded) {
            state
                .plan
                .completion_bytes(stage)
                .expect("mock success completion requires a submitted stage")
        } else {
            0
        };
        let event = CompletionEvent::new(
            operation,
            key,
            stage,
            outcome,
            bytes,
            CompletionGeneration::for_key(key),
            CompletionTimestamp::from_nanos(state.clock_ns),
        );
        state.clock_ns = state.clock_ns.saturating_add(1);
        state.completions.push_back(event.clone());
        event
    }
}

impl MaterializationProvider for MockPhysicalProvider {
    fn placement(&self) -> MaterializationPlacement {
        self.lock().placement
    }

    fn resource_topology(&self) -> ferrule_common::Result<PhysicalMaterializationTopology> {
        let limits = self.lock().limits;
        PhysicalMaterializationTopology::new(
            limits,
            limits.capacity.device_install_bytes,
            limits.capacity.install_slots,
        )
    }

    fn prepare(
        &mut self,
        request: MaterializationRequest,
        _intent: MaterializationPurpose,
    ) -> Result<MaterializationPreparation, FailureReason> {
        let mut state = self.lock();
        state.commands.push(MockPhysicalCommand::Prepare(request));
        if let Some(failure) = state.resolve_failure.take() {
            return Err(failure);
        }
        if request.model() != state.placement.model()
            || request.backend() != state.placement.backend()
            || request.device() != state.placement.device()
        {
            return Err(FailureReason::ContractViolation {
                message: "mock request does not match physical placement".into(),
            });
        }
        let preparation = match state.resolved.get(&request).copied() {
            Some(preparation) => preparation,
            None => {
                let configured = state.next_preparation.take();
                let generation = configured
                    .map(|(generation, _, _)| generation)
                    .unwrap_or(state.generation);
                let key = request.materialization_key(generation).map_err(|error| {
                    FailureReason::ContractViolation {
                        message: error.to_string(),
                    }
                })?;
                let default_binding = state.binding_for(key);
                let binding = configured
                    .map(|(_, slot, _)| {
                        ResidencyBinding::new(
                            key.model(),
                            key.resource(),
                            key.backend(),
                            key.device(),
                            slot,
                            key.destination_generation(),
                        )
                    })
                    .unwrap_or(default_binding);
                let preparation = MockPreparation {
                    key,
                    binding,
                    evicted: configured.and_then(|(_, _, evicted)| evicted),
                };
                state.resolved.insert(request, preparation);
                preparation
            }
        };
        if state.resident {
            state.resident_keys.insert(preparation.key);
            MaterializationResident::new(preparation.key, preparation.binding)
                .map(MaterializationPreparation::Resident)
                .map_err(|error| FailureReason::ContractViolation {
                    message: error.to_string(),
                })
        } else {
            MaterializationTransfer::new(preparation.key, preparation.binding, preparation.evicted)
                .map(MaterializationPreparation::Transfer)
                .map_err(|error| FailureReason::ContractViolation {
                    message: error.to_string(),
                })
        }
    }

    fn prepared(
        &mut self,
        key: MaterializationKey,
    ) -> Result<MaterializationPreparation, FailureReason> {
        let mut state = self.lock();
        state.commands.push(MockPhysicalCommand::Prepared(key));
        let preparation = state
            .resolved
            .values()
            .find(|preparation| preparation.key == key)
            .copied()
            .ok_or_else(|| FailureReason::ContractViolation {
                message: "mock has no provider preparation".into(),
            })?;
        if state.resident_keys.contains(&preparation.key) {
            MaterializationResident::new(preparation.key, preparation.binding)
                .map(MaterializationPreparation::Resident)
                .map_err(|error| FailureReason::ContractViolation {
                    message: error.to_string(),
                })
        } else {
            MaterializationTransfer::new(preparation.key, preparation.binding, preparation.evicted)
                .map(MaterializationPreparation::Transfer)
                .map_err(|error| FailureReason::ContractViolation {
                    message: error.to_string(),
                })
        }
    }

    fn promote_to_execution(
        &mut self,
        key: MaterializationKey,
    ) -> Result<MaterializationPreparation, FailureReason> {
        {
            let mut state = self.lock();
            state
                .commands
                .push(MockPhysicalCommand::PromoteToExecution(key));
            if let Some(failure) = state
                .promotion_failures
                .get_mut(&key)
                .and_then(VecDeque::pop_front)
            {
                return Err(failure);
            }
        }
        self.prepared(key)
    }

    fn discard_preparation(&mut self, key: MaterializationKey) -> Result<(), FailureReason> {
        let mut state = self.lock();
        state
            .commands
            .push(MockPhysicalCommand::DiscardPreparation(key));
        let request = state
            .resolved
            .iter()
            .find_map(|(request, preparation)| (preparation.key == key).then_some(*request))
            .ok_or_else(|| FailureReason::ContractViolation {
                message: "mock cannot discard unknown preparation".into(),
            })?;
        state.resolved.remove(&request);
        state.resident_keys.remove(&key);
        Ok(())
    }

    fn materialization_plan(
        &self,
        key: MaterializationKey,
    ) -> Result<MaterializationResourcePlan, FailureReason> {
        let mut state = self.lock();
        state
            .commands
            .push(MockPhysicalCommand::MaterializationPlan(key));
        Ok(state.plan)
    }

    fn release_execution_lease(&mut self, key: MaterializationKey) -> Result<(), FailureReason> {
        let mut state = self.lock();
        state
            .commands
            .push(MockPhysicalCommand::ReleaseExecutionLease(key));
        if let Some(failure) = state.release_failures.pop_front() {
            Err(failure)
        } else {
            Ok(())
        }
    }

    fn reserve(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        plan: MaterializationResourcePlan,
    ) -> Result<PhysicalMaterializationOperationReservation, FailureReason> {
        let mut state = self.lock();
        state
            .commands
            .push(MockPhysicalCommand::Reserve(operation, key, plan));
        if let Some(failure) = state.reserve_failure.take() {
            return Err(failure);
        }
        if plan != state.plan {
            return Err(FailureReason::ContractViolation {
                message: "mock physical resource plan expectation mismatch".into(),
            });
        }
        let reservation_key = state.reservation_key_override.take().unwrap_or(key);
        let reservation_operation = state
            .reservation_operation_override
            .take()
            .unwrap_or(operation);
        let binding = match state.reservation_slot_override.take() {
            Some(slot) => ResidencyBinding::new(
                reservation_key.model(),
                reservation_key.resource(),
                reservation_key.backend(),
                reservation_key.device(),
                slot,
                reservation_key.destination_generation(),
            ),
            None => state
                .resolved
                .values()
                .find(|preparation| preparation.key == reservation_key)
                .map(|preparation| preparation.binding)
                .unwrap_or_else(|| state.binding_for(reservation_key)),
        };
        let identity = reservation_operation.get();
        let descriptor = RegisteredPinnedAlignedSlabLeaseDescriptor::new(
            reservation_operation,
            SlabId::new(identity.saturating_add(SLAB_ID_OFFSET)),
            RegistrationId::new(identity.saturating_add(REGISTRATION_ID_OFFSET)),
            0x10000,
            plan.requirements.pinned_host_bytes,
            0,
            plan.requirements.pinned_host_bytes,
            4096,
            reservation_key.source_generation(),
            reservation_key.destination_generation(),
        )
        .map_err(|error| FailureReason::ContractViolation {
            message: error.to_string(),
        })?;
        PhysicalMaterializationOperationReservation::new(
            reservation_key,
            binding,
            [descriptor],
            UploadFenceContract::new(
                reservation_operation,
                FenceId::new(identity.saturating_add(FENCE_ID_OFFSET)),
                reservation_key.destination_generation(),
            ),
        )
        .map_err(|error| FailureReason::ContractViolation {
            message: error.to_string(),
        })
    }

    fn submit_read(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &PhysicalMaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        if reservation.key() != key || reservation.upload_fence().operation != operation {
            return Err(FailureReason::ContractViolation {
                message: "mock received a mismatched read reservation".into(),
            });
        }
        let mut state = self.lock();
        state
            .commands
            .push(MockPhysicalCommand::SubmitRead(operation, key, plan));
        if plan != state.plan {
            return Err(FailureReason::ContractViolation {
                message: "mock read resource plan expectation mismatch".into(),
            });
        }
        state.emit(
            operation,
            key,
            LoadStage::ReadSubmitted,
            plan,
            CompletionOutcome::Succeeded,
        );
        Ok(())
    }

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &PhysicalMaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        if reservation.key() != key || reservation.upload_fence().operation != operation {
            return Err(FailureReason::ContractViolation {
                message: "mock received a mismatched upload reservation".into(),
            });
        }
        let mut state = self.lock();
        state
            .commands
            .push(MockPhysicalCommand::SubmitUpload(operation, key, plan));
        if plan != state.plan {
            return Err(FailureReason::ContractViolation {
                message: "mock upload resource plan expectation mismatch".into(),
            });
        }
        state.emit(
            operation,
            key,
            LoadStage::UploadSubmitted,
            plan,
            CompletionOutcome::Succeeded,
        );
        Ok(())
    }

    fn poll_install(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &PhysicalMaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        if reservation.key() != key || reservation.binding() != state_binding(self, key) {
            return Err(FailureReason::ContractViolation {
                message: "mock received a mismatched install reservation".into(),
            });
        }
        let mut state = self.lock();
        state
            .commands
            .push(MockPhysicalCommand::PollInstall(operation, key, plan));
        if plan != state.plan {
            return Err(FailureReason::ContractViolation {
                message: "mock install resource plan expectation mismatch".into(),
            });
        }
        state.emit(
            operation,
            key,
            LoadStage::Installing,
            plan,
            CompletionOutcome::Succeeded,
        );
        Ok(())
    }

    fn cancel(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        stage: LoadStage,
        reason: CancellationReason,
    ) -> Result<(), FailureReason> {
        let mut state = self.lock();
        state.commands.push(MockPhysicalCommand::Cancel(
            operation,
            key,
            stage,
            reason.clone(),
        ));
        state
            .completions
            .retain(|event| !(event.operation == operation && event.stage == stage));
        if stage.is_submitted_completion_stage() {
            let timestamp = CompletionTimestamp::from_nanos(state.clock_ns);
            state.clock_ns = state.clock_ns.saturating_add(1);
            state.completions.push_back(CompletionEvent::new(
                operation,
                key,
                stage,
                CompletionOutcome::Cancelled(reason),
                0,
                CompletionGeneration::for_key(key),
                timestamp,
            ));
        }
        Ok(())
    }

    fn next_completion(&mut self) -> Option<CompletionEvent> {
        self.lock().completions.pop_front()
    }
}

pub fn state_binding(backend: &MockPhysicalProvider, key: MaterializationKey) -> ResidencyBinding {
    let state = backend.lock();
    state
        .resolved
        .values()
        .find(|preparation| preparation.key == key)
        .map(|preparation| preparation.binding)
        .unwrap_or_else(|| state.binding_for(key))
}

pub fn request(seed: u8) -> MaterializationRequest {
    let artifact = ResourceSource::new(
        SourceIdentityHash::new([seed.max(1); 32]),
        ContentHash::new([seed.saturating_add(1).max(1); 32]),
        PayloadEncodingId::new(1),
        SourceGeneration::new(5),
    )
    .unwrap();
    MaterializationRequest::for_placement(
        MaterializationPlacement::new(
            ModelInstanceId::new(17),
            BackendId::new(4),
            DeviceId::new(2),
        )
        .unwrap(),
        artifact,
        MaterializedResourceId::routed_expert(
            LayerId::new(u32::from(seed)),
            ExpertId::new(u32::from(seed)),
        ),
    )
    .unwrap()
}

pub fn key(seed: u8) -> MaterializationKey {
    request(seed)
        .materialization_key(DestinationGeneration::new(DESTINATION_GENERATION))
        .unwrap()
}

pub fn waiter(transaction: u64, continuation: u64) -> WaiterId {
    WaiterId::new(
        ExecutionTransactionId::new(transaction).unwrap(),
        RequestGeneration::new(1),
        DependencySetEpoch::new(1),
        ContinuationId::new(continuation),
    )
    .unwrap()
}

pub fn physical_resources() -> PhysicalResourceBroker {
    bounded_physical_resources(4, 4, 4, 4)
}

pub fn bounded_physical_resources(
    load_operations: u64,
    sqe: u64,
    pinned_operations: u64,
    upload_slots: u64,
) -> PhysicalResourceBroker {
    PhysicalResourceBroker::new(ResourceKind::ALL.map(|kind| {
        let capacity = match kind {
            ResourceKind::ReadSlot => sqe,
            ResourceKind::PinnedHostBytes => BYTES * pinned_operations,
            ResourceKind::StorageReadBytes => BYTES * sqe,
            ResourceKind::UploadSlot => upload_slots,
            ResourceKind::UploadBytes => BYTES * upload_slots,
            ResourceKind::InstallSlot => upload_slots,
            ResourceKind::DeviceInstallBytes => BYTES * upload_slots,
            ResourceKind::ResidentBytes => BYTES * load_operations,
            ResourceKind::ResidencyLease | ResourceKind::LoadOperation => load_operations,
            ResourceKind::Arena
            | ResourceKind::KvPage
            | ResourceKind::Continuation
            | ResourceKind::Waiter
            | ResourceKind::ReadyCohort => 64,
        };
        PhysicalResourceLimit::new(kind, capacity, 0)
    }))
    .unwrap()
}

pub fn resolved_provider() -> (
    SharedMaterializationProvider,
    MockPhysicalHandle,
    MaterializationKey,
) {
    let (physical, handle) = MockPhysicalProvider::manual();
    let backend = SharedMaterializationProvider::new(Box::new(physical));
    let preparation = backend
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    assert!(matches!(
        preparation,
        MaterializationPreparation::Transfer(_)
    ));
    (backend, handle, preparation.key())
}

pub fn registry(
    automatic: bool,
) -> (
    LoadRegistry<SharedMaterializationProvider>,
    MockPhysicalHandle,
) {
    let (physical, handle) = if automatic {
        MockPhysicalProvider::automatic()
    } else {
        MockPhysicalProvider::manual()
    };
    let backend = SharedMaterializationProvider::new(Box::new(physical));
    for seed in 1..=4 {
        backend
            .prepare(request(seed), MaterializationPurpose::Execution)
            .unwrap();
    }
    (
        LoadRegistry::new(backend, physical_resources(), FairQueueConfig::default()).unwrap(),
        handle,
    )
}

pub fn prefetch_registry(
    automatic: bool,
) -> (
    LoadRegistry<SharedMaterializationProvider>,
    MockPhysicalHandle,
    MaterializationKey,
) {
    let (physical, handle) = if automatic {
        MockPhysicalProvider::automatic()
    } else {
        MockPhysicalProvider::manual()
    };
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let preparation = provider
        .prepare(request(1), MaterializationPurpose::Prefetch)
        .unwrap();
    let key = preparation.key();
    (
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap(),
        handle,
        key,
    )
}

pub fn load_request(
    registry: &LoadRegistry<SharedMaterializationProvider>,
    key: MaterializationKey,
    plan: MaterializationResourcePlan,
    demand: ResourceDemand,
) -> LoadRequest {
    stage_request(registry.provider().preparation(key).unwrap(), plan, demand)
}

pub fn attach(
    registry: &mut LoadRegistry<SharedMaterializationProvider>,
    waiter: WaiterId,
    key: MaterializationKey,
    now_ns: u64,
) -> OperationId {
    let request = load_request(
        registry,
        key,
        uniform_plan(),
        ResourceDemand::required(ExecutionPhase::Prefill),
    );
    registry
        .attach_waiter(
            waiter,
            ResourceDemand::required(ExecutionPhase::Prefill),
            [request],
            now_ns,
        )
        .unwrap()
        .created[0]
}

pub fn manual_at_read() -> (
    LoadRegistry<SharedMaterializationProvider>,
    MockPhysicalHandle,
    MaterializationKey,
    OperationId,
) {
    let (mut registry, handle) = registry(false);
    let key = key(1);
    let operation = attach(&mut registry, waiter(1, 1), key, 1);
    assert!(registry.schedule_one(2).unwrap());
    assert!(registry.schedule_one(3).unwrap());
    (registry, handle, key, operation)
}

pub fn apply_physical(
    registry: &mut LoadRegistry<SharedMaterializationProvider>,
    maximum: usize,
) -> CompletionDisposition {
    assert_eq!(registry.collect_provider_completions(maximum), 1);
    registry.process_one_completion().unwrap()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FakeMaterializationCommand {
    Reserve(OperationId),
    SubmitRead(OperationId),
    SubmitUpload(OperationId),
    PollInstall(OperationId),
    Cancel(OperationId, LoadStage),
}

/// One deterministic completion override. Omitted fields retain exact command
/// identity, making wrong-ID/generation/stage tests explicit and readable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FakeCompletionSpec {
    pub outcome: CompletionOutcome,
    pub operation: Option<OperationId>,
    pub key: Option<MaterializationKey>,
    pub stage: Option<LoadStage>,
    pub bytes: Option<u64>,
    pub generation: Option<CompletionGeneration>,
    pub timestamp: Option<CompletionTimestamp>,
}

impl FakeCompletionSpec {
    pub fn success() -> Self {
        Self {
            outcome: CompletionOutcome::Succeeded,
            operation: None,
            key: None,
            stage: None,
            bytes: None,
            generation: None,
            timestamp: None,
        }
    }

    pub fn failed(reason: FailureReason) -> Self {
        Self {
            outcome: CompletionOutcome::Failed(reason),
            ..Self::success()
        }
    }
}

/// Deterministic no-thread/no-sleep provider. Commands append scripted completion
/// events to a FIFO; tests may also inject arbitrary events directly.
#[derive(Debug)]
pub struct FakeMaterializationProvider {
    automatic: bool,
    clock_ns: u64,
    commands: Vec<FakeMaterializationCommand>,
    completions: VecDeque<CompletionEvent>,
    scripts: BTreeMap<LoadStage, VecDeque<FakeCompletionSpec>>,
    lost: BTreeMap<LoadStage, usize>,
    rejected: BTreeMap<LoadStage, VecDeque<FailureReason>>,
    reserve_failures: VecDeque<FailureReason>,
    reads: usize,
    uploads: usize,
    installs: usize,
    cancellations: usize,
}

impl Default for FakeMaterializationProvider {
    fn default() -> Self {
        Self {
            automatic: true,
            clock_ns: 1,
            commands: Vec::new(),
            completions: VecDeque::new(),
            scripts: BTreeMap::new(),
            lost: BTreeMap::new(),
            rejected: BTreeMap::new(),
            reserve_failures: VecDeque::new(),
            reads: 0,
            uploads: 0,
            installs: 0,
            cancellations: 0,
        }
    }
}

impl FakeMaterializationProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn manual() -> Self {
        Self {
            automatic: false,
            ..Self::default()
        }
    }

    pub fn set_automatic(&mut self, automatic: bool) {
        self.automatic = automatic;
    }

    pub fn script_next(&mut self, stage: LoadStage, completion: FakeCompletionSpec) {
        self.scripts.entry(stage).or_default().push_back(completion);
    }

    pub fn lose_next(&mut self, stage: LoadStage) {
        *self.lost.entry(stage).or_default() += 1;
    }

    pub fn reject_next(&mut self, stage: LoadStage, reason: FailureReason) {
        self.rejected.entry(stage).or_default().push_back(reason);
    }

    pub fn fail_next_reserve(&mut self, reason: FailureReason) {
        self.reserve_failures.push_back(reason);
    }

    pub fn push_completion(&mut self, completion: CompletionEvent) {
        self.completions.push_back(completion);
    }

    pub fn commands(&self) -> &[FakeMaterializationCommand] {
        &self.commands
    }

    pub const fn physical_reads(&self) -> usize {
        self.reads
    }

    pub const fn physical_uploads(&self) -> usize {
        self.uploads
    }

    pub const fn physical_installs(&self) -> usize {
        self.installs
    }

    pub const fn cancellations(&self) -> usize {
        self.cancellations
    }

    pub fn queued_completions(&self) -> usize {
        self.completions.len()
    }

    pub fn command_result(&mut self, stage: LoadStage) -> Result<(), FailureReason> {
        match self.rejected.get_mut(&stage).and_then(VecDeque::pop_front) {
            Some(reason) => Err(reason),
            None => Ok(()),
        }
    }

    pub fn emit(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        stage: LoadStage,
        bytes: u64,
    ) {
        if let Some(lost) = self.lost.get_mut(&stage)
            && *lost != 0
        {
            *lost -= 1;
            return;
        }
        if !self.automatic && !self.scripts.contains_key(&stage) {
            return;
        }
        let spec = self
            .scripts
            .get_mut(&stage)
            .and_then(VecDeque::pop_front)
            .unwrap_or_else(FakeCompletionSpec::success);
        let timestamp = spec.timestamp.unwrap_or_else(|| {
            let timestamp = CompletionTimestamp::from_nanos(self.clock_ns);
            self.clock_ns = self.clock_ns.saturating_add(1);
            timestamp
        });
        self.completions.push_back(CompletionEvent::new(
            spec.operation.unwrap_or(operation),
            spec.key.unwrap_or(key),
            spec.stage.unwrap_or(stage),
            spec.outcome,
            spec.bytes.unwrap_or(bytes),
            spec.generation
                .unwrap_or_else(|| CompletionGeneration::for_key(key)),
            timestamp,
        ));
    }
}

pub fn fake_slot_for_key(key: MaterializationKey) -> DestinationSlotId {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    let slot = (hasher.finish() as u32).max(1);
    DestinationSlotId::new(slot)
}

impl RuntimeMaterializationProvider for FakeMaterializationProvider {
    fn preparation(
        &self,
        key: MaterializationKey,
    ) -> Result<MaterializationPreparation, FailureReason> {
        let binding = ResidencyBinding::new(
            key.model(),
            key.resource(),
            key.backend(),
            key.device(),
            fake_slot_for_key(key),
            key.destination_generation(),
        );
        ferrule_model::MaterializationTransfer::new(key, binding, None)
            .map(MaterializationPreparation::Transfer)
            .map_err(|source| FailureReason::Protocol { source })
    }

    fn promote_to_execution(
        &mut self,
        key: MaterializationKey,
    ) -> Result<ExecutionPromotion, FailureReason> {
        self.preparation(key)
            .map(ExecutionPromotion::AlreadyExecution)
    }

    fn discard_preparation(&mut self, _key: MaterializationKey) -> Result<(), FailureReason> {
        Ok(())
    }

    fn release_execution_lease(&mut self, _key: MaterializationKey) -> Result<(), FailureReason> {
        Ok(())
    }

    fn materialization_plan(
        &self,
        _key: MaterializationKey,
    ) -> Result<MaterializationResourcePlan, FailureReason> {
        MaterializationResourcePlan::new(
            ferrule_common::materialization_io::MaterializationResourceRequirements {
                read_slots: 1,
                storage_read_bytes: 4096,
                pinned_host_bytes: 4096,
                upload_slots: 1,
                h2d_bytes: 4096,
                install_slots: 1,
                device_install_bytes: 4096,
            },
            4096,
        )
        .map_err(|source| FailureReason::Resources { source })
    }

    fn reserve(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        plan: MaterializationResourcePlan,
    ) -> Result<MaterializationOperationReservation, FailureReason> {
        self.commands
            .push(FakeMaterializationCommand::Reserve(operation));
        let bytes = plan.requirements.pinned_host_bytes;
        if let Some(reason) = self.reserve_failures.pop_front() {
            return Err(reason);
        }
        let identity = operation.get().max(1);
        let base_address = 0x1000usize.saturating_add(
            usize::try_from(identity.saturating_mul(0x10)).unwrap_or(usize::MAX - 0x1000),
        );
        let descriptor = RegisteredPinnedAlignedSlabLeaseDescriptor::new(
            operation,
            SlabId::new(identity),
            RegistrationId::new(identity),
            base_address,
            bytes,
            0,
            bytes,
            1,
            key.source_generation(),
            key.destination_generation(),
        )
        .map_err(|source| FailureReason::Protocol { source })?;
        let slot = fake_slot_for_key(key);
        Ok(MaterializationOperationReservation::new(
            vec![RegisteredPinnedAlignedSlabLease::new(descriptor)].into_boxed_slice(),
            ResidencyBinding::new(
                key.model(),
                key.resource(),
                key.backend(),
                key.device(),
                slot,
                key.destination_generation(),
            ),
            UploadFenceContract::new(
                operation,
                FenceId::new(identity),
                key.destination_generation(),
            ),
            None,
        ))
    }

    fn submit_read(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        _reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        self.command_result(LoadStage::ReadSubmitted)?;
        self.commands
            .push(FakeMaterializationCommand::SubmitRead(operation));
        self.reads += 1;
        self.emit(
            operation,
            key,
            LoadStage::ReadSubmitted,
            plan.requirements.storage_read_bytes,
        );
        Ok(())
    }

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        _reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        self.command_result(LoadStage::UploadSubmitted)?;
        self.commands
            .push(FakeMaterializationCommand::SubmitUpload(operation));
        self.uploads += 1;
        self.emit(
            operation,
            key,
            LoadStage::UploadSubmitted,
            plan.requirements.h2d_bytes,
        );
        Ok(())
    }

    fn poll_install(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        _reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        self.command_result(LoadStage::Installing)?;
        self.commands
            .push(FakeMaterializationCommand::PollInstall(operation));
        self.installs += 1;
        self.emit(
            operation,
            key,
            LoadStage::Installing,
            plan.requirements.device_install_bytes,
        );
        Ok(())
    }

    fn cancel(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        stage: LoadStage,
        reason: CancellationReason,
    ) -> Result<(), FailureReason> {
        self.command_result(stage)?;
        self.commands
            .push(FakeMaterializationCommand::Cancel(operation, stage));
        self.cancellations += 1;
        self.completions
            .retain(|event| !(event.operation == operation && event.stage == stage));
        if self.automatic && stage.is_submitted_completion_stage() {
            let timestamp = CompletionTimestamp::from_nanos(self.clock_ns);
            self.clock_ns = self.clock_ns.saturating_add(1);
            self.completions.push_back(CompletionEvent::new(
                operation,
                key,
                stage,
                CompletionOutcome::Cancelled(reason),
                0,
                CompletionGeneration::for_key(key),
                timestamp,
            ));
        }
        Ok(())
    }

    fn next_completion(&mut self) -> Option<CompletionEvent> {
        self.completions.pop_front()
    }
}
