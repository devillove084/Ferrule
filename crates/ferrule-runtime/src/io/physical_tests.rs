use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::io_protocol::{
    BackendId, CancellationReason, CompletionEvent, CompletionGeneration, CompletionOutcome,
    CompletionTimestamp, ContentHash, ContinuationId, DependencySetEpoch, DestinationGeneration,
    DestinationSlotId, DeviceId, ExpertId, FailureReason, FenceId, LayerId, LoadStage,
    MaterializationKey, MaterializedResourceId, ModelInstanceId, OperationId, PayloadEncodingId,
    RegisteredPinnedAlignedSlabLeaseDescriptor, RegistrationId, RequestGeneration,
    ResidencyBinding, SlabId, SourceGeneration, SourceIdentityHash, StaleReason,
    UploadFenceContract, WaiterId,
};
use ferrule_common::materialization_io::{
    MaterializationResourceLimits, MaterializationResourcePlan, MaterializationResourceRequirements,
};
use ferrule_model::{
    MaterializationPlacement, MaterializationPreparation, MaterializationProvider,
    MaterializationPurpose, MaterializationRequest, MaterializationResident,
    MaterializationResolver, MaterializationTransfer, PhysicalMaterializationOperationReservation,
    PhysicalMaterializationTopology, ResourceSource,
};

use super::{
    CompletionDisposition, ContinuationFailure, FairQueueConfig, LoadRegistry, LoadRequest,
    RuntimeMaterializationProvider, RuntimeMaterializationResolver, SharedMaterializationProvider,
};
use crate::scheduling::{
    PhysicalResourceBroker, PhysicalResourceLimit, ResourceClass, ResourceKind,
};

const BYTES: u64 = 4096;

fn retained_request(
    preparation: MaterializationPreparation,
    plan: MaterializationResourcePlan,
    class: ResourceClass,
    retention: ferrule_model::ResourceRetention,
) -> LoadRequest {
    LoadRequest::new(
        preparation,
        plan,
        class,
        retention,
        if class == ResourceClass::Prefetch {
            MaterializationPurpose::Prefetch
        } else {
            MaterializationPurpose::Execution
        },
    )
}

fn stage_request(
    preparation: MaterializationPreparation,
    plan: MaterializationResourcePlan,
    class: ResourceClass,
) -> LoadRequest {
    retained_request(
        preparation,
        plan,
        class,
        ferrule_model::ResourceRetention::ThroughStage,
    )
}
const DESTINATION_GENERATION: u64 = 37;
const SLOT: u32 = 9;
const SLAB_ID_OFFSET: u64 = 1000;
const REGISTRATION_ID_OFFSET: u64 = 2000;
const FENCE_ID_OFFSET: u64 = 3000;

fn uniform_plan() -> MaterializationResourcePlan {
    MaterializationResourcePlan::uniform_payload(BYTES)
        .expect("mock uniform materialization plan must be valid")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MockPhysicalCommand {
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
struct MockPreparation {
    key: MaterializationKey,
    binding: ResidencyBinding,
    evicted: Option<MaterializationKey>,
}

#[derive(Debug)]
struct MockPhysicalState {
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
    fn new(automatic: bool) -> Self {
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

    fn binding_for(&self, key: MaterializationKey) -> ResidencyBinding {
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

    fn emit(
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
pub(crate) struct MockPhysicalProvider {
    state: Arc<Mutex<MockPhysicalState>>,
}

#[derive(Debug, Clone)]
pub(crate) struct MockPhysicalHandle {
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
    pub(crate) fn automatic() -> (Self, MockPhysicalHandle) {
        Self::new(true)
    }

    pub(crate) fn manual() -> (Self, MockPhysicalHandle) {
        Self::new(false)
    }

    fn new(automatic: bool) -> (Self, MockPhysicalHandle) {
        let state = Arc::new(Mutex::new(MockPhysicalState::new(automatic)));
        (
            Self {
                state: Arc::clone(&state),
            },
            MockPhysicalHandle { state },
        )
    }

    fn lock(&self) -> MutexGuard<'_, MockPhysicalState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl MockPhysicalHandle {
    fn lock(&self) -> MutexGuard<'_, MockPhysicalState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn placement(&self) -> MaterializationPlacement {
        self.lock().placement
    }

    pub(crate) fn limits(&self) -> MaterializationResourceLimits {
        self.lock().limits
    }

    pub(crate) fn set_bytes_and_limits(&self, bytes: u64, limits: MaterializationResourceLimits) {
        self.set_plan_and_limits(
            MaterializationResourcePlan::uniform_payload(bytes)
                .expect("mock uniform materialization plan must be valid"),
            limits,
        );
    }

    fn set_plan_and_limits(
        &self,
        plan: MaterializationResourcePlan,
        limits: MaterializationResourceLimits,
    ) {
        let mut state = self.lock();
        state.plan = plan;
        state.limits = limits;
    }

    pub(crate) fn command_count(&self, predicate: impl Fn(&MockPhysicalCommand) -> bool) -> usize {
        self.lock()
            .commands
            .iter()
            .filter(|command| predicate(command))
            .count()
    }

    pub(crate) fn commands(&self) -> Vec<MockPhysicalCommand> {
        self.lock().commands.clone()
    }

    pub(crate) fn binding(&self, key: MaterializationKey) -> ResidencyBinding {
        let state = self.lock();
        state
            .resolved
            .values()
            .find(|preparation| preparation.key == key)
            .map(|preparation| preparation.binding)
            .unwrap_or_else(|| state.binding_for(key))
    }

    fn configure_next_preparation(
        &self,
        generation: u64,
        slot: DestinationSlotId,
        evicted: Option<MaterializationKey>,
    ) {
        self.lock().next_preparation =
            Some((DestinationGeneration::new(generation), slot, evicted));
    }

    pub(crate) fn set_resident(&self, resident: bool) {
        self.lock().resident = resident;
    }

    fn fail_next_resolve(&self, failure: FailureReason) {
        self.lock().resolve_failure = Some(failure);
    }

    fn fail_next_reserve(&self, failure: FailureReason) {
        self.lock().reserve_failure = Some(failure);
    }

    fn fail_next_promotion(&self, key: MaterializationKey, failure: FailureReason) {
        self.lock()
            .promotion_failures
            .entry(key)
            .or_default()
            .push_back(failure);
    }

    pub(crate) fn fail_next_release(&self, failure: FailureReason) {
        self.lock().release_failures.push_back(failure);
    }

    fn override_reservation_key(&self, key: MaterializationKey) {
        self.lock().reservation_key_override = Some(key);
    }

    fn override_reservation_operation(&self, operation: OperationId) {
        self.lock().reservation_operation_override = Some(operation);
    }

    fn override_reservation_slot(&self, slot: DestinationSlotId) {
        self.lock().reservation_slot_override = Some(slot);
    }

    pub(crate) fn lose_next(&self, stage: LoadStage) {
        *self.lock().lost_completions.entry(stage).or_default() += 1;
    }

    pub(crate) fn script_outcome(&self, stage: LoadStage, outcome: CompletionOutcome) {
        self.lock()
            .scripted_outcomes
            .entry(stage)
            .or_default()
            .push_back(outcome);
    }

    fn push_completion(&self, completion: CompletionEvent) {
        self.lock().completions.push_back(completion);
    }

    fn push_outcome(
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

fn state_binding(backend: &MockPhysicalProvider, key: MaterializationKey) -> ResidencyBinding {
    let state = backend.lock();
    state
        .resolved
        .values()
        .find(|preparation| preparation.key == key)
        .map(|preparation| preparation.binding)
        .unwrap_or_else(|| state.binding_for(key))
}

fn request(seed: u8) -> MaterializationRequest {
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

fn key(seed: u8) -> MaterializationKey {
    request(seed)
        .materialization_key(DestinationGeneration::new(DESTINATION_GENERATION))
        .unwrap()
}

fn waiter(transaction: u64, continuation: u64) -> WaiterId {
    WaiterId::new(
        ExecutionTransactionId::new(transaction).unwrap(),
        RequestGeneration::new(1),
        DependencySetEpoch::new(1),
        ContinuationId::new(continuation),
    )
    .unwrap()
}

fn physical_resources() -> PhysicalResourceBroker {
    bounded_physical_resources(4, 4, 4, 4)
}

fn bounded_physical_resources(
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

fn resolved_provider() -> (
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

fn registry(
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

fn prefetch_registry(
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

fn load_request(
    registry: &LoadRegistry<SharedMaterializationProvider>,
    key: MaterializationKey,
    plan: MaterializationResourcePlan,
    class: ResourceClass,
) -> LoadRequest {
    stage_request(registry.provider().preparation(key).unwrap(), plan, class)
}

fn attach(
    registry: &mut LoadRegistry<SharedMaterializationProvider>,
    waiter: WaiterId,
    key: MaterializationKey,
    now_ns: u64,
) -> OperationId {
    let request = load_request(registry, key, uniform_plan(), ResourceClass::Throughput);
    registry
        .attach_waiter(waiter, [request], now_ns)
        .unwrap()
        .created[0]
}

fn manual_at_read() -> (
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

fn apply_physical(
    registry: &mut LoadRegistry<SharedMaterializationProvider>,
    maximum: usize,
) -> CompletionDisposition {
    assert_eq!(registry.collect_provider_completions(maximum), 1);
    registry.process_one_completion().unwrap()
}

#[test]
fn physical_bridge_reservation_keeps_exact_key() {
    let (mut backend, _, key) = resolved_provider();
    let reservation = backend
        .reserve(OperationId::new(7), key, uniform_plan())
        .unwrap();
    assert_eq!(
        reservation.binding().generation,
        key.destination_generation()
    );
}

#[test]
fn physical_bridge_reservation_keeps_exact_binding() {
    let (mut backend, handle, key) = resolved_provider();
    let reservation = backend
        .reserve(OperationId::new(7), key, uniform_plan())
        .unwrap();
    assert_eq!(reservation.binding(), handle.binding(key));
}

#[test]
fn physical_bridge_reservation_keeps_exact_slab_descriptor() {
    let (mut backend, _, key) = resolved_provider();
    let operation = OperationId::new(7);
    let reservation = backend.reserve(operation, key, uniform_plan()).unwrap();
    let descriptor = reservation.slabs()[0].descriptor();
    assert_eq!(descriptor.operation(), operation);
    assert_eq!(descriptor.slab(), SlabId::new(7 + SLAB_ID_OFFSET));
    assert_eq!(
        descriptor.registration(),
        RegistrationId::new(7 + REGISTRATION_ID_OFFSET)
    );
    assert_eq!(descriptor.address().get(), 0x10000);
}

#[test]
fn physical_bridge_reservation_keeps_exact_upload_fence() {
    let (mut backend, _, key) = resolved_provider();
    let operation = OperationId::new(7);
    let reservation = backend.reserve(operation, key, uniform_plan()).unwrap();
    assert_eq!(reservation.upload_fence().operation, operation);
    assert_eq!(
        reservation.upload_fence().fence,
        FenceId::new(7 + FENCE_ID_OFFSET)
    );
}

#[test]
fn physical_bridge_rejects_reservation_for_different_key() {
    let (mut backend, handle, canonical_key) = resolved_provider();
    handle.override_reservation_key(key(2));
    assert!(matches!(
        backend.reserve(OperationId::new(7), canonical_key, uniform_plan()),
        Err(FailureReason::ContractViolation { message: _ })
    ));
}

#[test]
fn physical_bridge_rejects_reservation_for_different_operation() {
    let (mut backend, handle, key) = resolved_provider();
    handle.override_reservation_operation(OperationId::new(99));
    assert!(matches!(
        backend.reserve(OperationId::new(7), key, uniform_plan()),
        Err(FailureReason::ContractViolation { message: _ })
    ));
}

#[test]
fn physical_bridge_rejects_binding_changed_after_resolve() {
    let (mut backend, handle, key) = resolved_provider();
    handle.override_reservation_slot(DestinationSlotId::new(SLOT + 1));
    assert!(matches!(
        backend.reserve(OperationId::new(7), key, uniform_plan()),
        Err(FailureReason::ContractViolation { message: _ })
    ));
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::Cancel(_, _, LoadStage::Reserved, _)
        )),
        1
    );
}

#[test]
fn physical_bridge_forwards_materialization_plan() {
    let (physical, handle) = MockPhysicalProvider::manual();
    let backend = SharedMaterializationProvider::new(Box::new(physical));
    backend
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    assert_eq!(
        backend.materialization_plan(key(1)).unwrap(),
        uniform_plan()
    );
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::MaterializationPlan(_)
        )),
        1
    );
}

#[test]
fn physical_bridge_forwards_read_upload_install_commands() {
    let (mut registry, handle) = registry(true);
    attach(&mut registry, waiter(1, 1), key(1), 1);
    registry.drive(100, 32).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
        1
    );
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitUpload(..))),
        1
    );
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::PollInstall(..))),
        1
    );
}

#[test]
fn physical_bridge_completion_enters_registry_unchanged() {
    let (mut registry, handle, key, operation) = manual_at_read();
    let mut completion = CompletionEvent::new(
        operation,
        key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Succeeded,
        BYTES,
        CompletionGeneration::for_key(key),
        CompletionTimestamp::from_nanos(88),
    );
    completion.generation.destination = DestinationGeneration::new(999);
    handle.push_completion(completion.clone());
    assert!(matches!(
        apply_physical(&mut registry, 4),
        CompletionDisposition::Rejected(_)
    ));
    assert_eq!(registry.rejected_completions()[0].event, completion);
}

#[test]
fn resolver_uses_provider_generation_without_logical_cache() {
    let (physical, handle) = MockPhysicalProvider::manual();
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let mut resolver = RuntimeMaterializationResolver::new(handle.placement(), Some(provider));
    let first = resolver.resolve(request(1)).unwrap();
    let second = resolver.resolve(request(1)).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.destination_generation(),
        DestinationGeneration::new(DESTINATION_GENERATION)
    );
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Prepare(_))),
        2
    );
    assert_eq!(resolver.stats().resolves, 2);
}

#[test]
fn registry_adopts_resident_preparation_without_a_read() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = preparation.key();
    let load = stage_request(preparation, uniform_plan(), ResourceClass::Throughput);
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    let report = registry.attach_waiter(waiter(1, 1), [load], 1).unwrap();
    assert_eq!(report.already_resident, 1);
    assert!(report.created.is_empty());
    assert_eq!(registry.residency_binding(key), Some(handle.binding(key)));
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
        0
    );
    registry
        .detach_continuation(ContinuationId::new(1), CancellationReason::Superseded, 2)
        .unwrap();
    registry.shutdown(3, 0).unwrap();
}

#[test]
fn registry_releases_provider_lease_after_last_continuation() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let first = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = first.key();
    let mut registry = LoadRegistry::new(
        provider.clone(),
        physical_resources(),
        FairQueueConfig::default(),
    )
    .unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            [stage_request(
                first,
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            1,
        )
        .unwrap();
    let second = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    registry
        .attach_waiter(
            waiter(2, 2),
            [stage_request(
                second,
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            2,
        )
        .unwrap();

    registry
        .detach_continuation(ContinuationId::new(1), CancellationReason::Superseded, 3)
        .unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        0
    );
    registry
        .detach_continuation(ContinuationId::new(2), CancellationReason::Superseded, 4)
        .unwrap();
    registry.drive(5, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    registry.shutdown(5, 0).unwrap();
}

fn consume_resident_resume(
    registry: &mut LoadRegistry<SharedMaterializationProvider>,
    continuation: ContinuationId,
    key: MaterializationKey,
    disposition: super::ResumeDisposition,
    now_ns: u64,
) {
    assert_eq!(registry.pop_ready(now_ns).unwrap(), Some(continuation));
    let dependencies =
        ferrule_common::DependencySet::new([ferrule_common::LogicalDependency::resource_resident(
            key,
        )
        .unwrap()])
        .unwrap();
    let mut resume = registry
        .prepare_resume(continuation, &dependencies)
        .unwrap();
    let leases = resume.take().unwrap();
    assert_eq!(leases.len(), 1);
    registry
        .finish_resume(&mut resume, disposition, now_ns + 1, now_ns + 2)
        .unwrap();
}

#[test]
fn still_active_stage_keeps_provider_lease_until_quiescent_detach() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = preparation.key();
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            [stage_request(
                preparation,
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            1,
        )
        .unwrap();

    consume_resident_resume(
        &mut registry,
        ContinuationId::new(1),
        key,
        super::ResumeDisposition::StillActive,
        2,
    );
    registry.drive(5, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        0
    );
    registry
        .detach_continuation(ContinuationId::new(1), CancellationReason::Superseded, 6)
        .unwrap();
    registry.drive(7, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    registry.shutdown(8, 0).unwrap();
}

#[test]
fn transaction_custody_survives_resume_and_releases_on_commit_or_rollback() {
    for (seed, outcome) in [
        (
            1,
            super::TransactionCustodyOutcome::Committed {
                started_ns: 10,
                finished_ns: 11,
            },
        ),
        (2, super::TransactionCustodyOutcome::RolledBack),
    ] {
        let (physical, handle) = MockPhysicalProvider::manual();
        handle.set_resident(true);
        let provider = SharedMaterializationProvider::new(Box::new(physical));
        let preparation = provider
            .prepare(request(seed), MaterializationPurpose::Execution)
            .unwrap();
        let key = preparation.key();
        let mut registry =
            LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
        registry
            .attach_waiter(
                waiter(seed as u64, seed as u64),
                [retained_request(
                    preparation,
                    uniform_plan(),
                    ResourceClass::Throughput,
                    ferrule_model::ResourceRetention::ThroughTransaction,
                )],
                1,
            )
            .unwrap();
        consume_resident_resume(
            &mut registry,
            ContinuationId::new(seed as u64),
            key,
            super::ResumeDisposition::Consumed,
            2,
        );
        registry.drive(5, 1).unwrap();
        assert_eq!(
            handle.command_count(|command| matches!(
                command,
                MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
            )),
            0
        );

        let transaction = ExecutionTransactionId::new(seed as u64).unwrap();
        registry
            .finish_transaction_custody(transaction, outcome)
            .unwrap();
        assert!(matches!(
            registry.finish_transaction_custody(transaction, outcome),
            Err(super::RegistryError::TransactionCustodyAlreadyFinished { transaction: candidate })
                if candidate == transaction
        ));
        registry.drive(6, 1).unwrap();
        assert_eq!(
            handle.command_count(|command| matches!(
                command,
                MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
            )),
            1
        );
        registry.shutdown(7, 0).unwrap();
    }
}

#[test]
fn persistent_custody_survives_transaction_terminal_until_explicit_retirement() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = preparation.key();
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            [retained_request(
                preparation,
                uniform_plan(),
                ResourceClass::Throughput,
                ferrule_model::ResourceRetention::Persistent,
            )],
            1,
        )
        .unwrap();
    consume_resident_resume(
        &mut registry,
        ContinuationId::new(1),
        key,
        super::ResumeDisposition::Consumed,
        2,
    );
    registry
        .finish_transaction_custody(
            ExecutionTransactionId::new(1).unwrap(),
            super::TransactionCustodyOutcome::Committed {
                started_ns: 5,
                finished_ns: 6,
            },
        )
        .unwrap();
    registry.drive(7, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        0
    );

    registry.retire_persistent_custody(key).unwrap();
    assert!(matches!(
        registry.retire_persistent_custody(key),
        Err(super::RegistryError::PersistentCustodyNotFound { key: candidate }) if *candidate == key
    ));
    registry.drive(8, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    registry.shutdown(9, 0).unwrap();
}

#[test]
fn shared_key_releases_only_after_last_transaction_owner() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let first = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = first.key();
    let second = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    for (preparation, id) in [(first, 1), (second, 2)] {
        registry
            .attach_waiter(
                waiter(id, id),
                [retained_request(
                    preparation,
                    uniform_plan(),
                    ResourceClass::Throughput,
                    ferrule_model::ResourceRetention::ThroughTransaction,
                )],
                id,
            )
            .unwrap();
        consume_resident_resume(
            &mut registry,
            ContinuationId::new(id),
            key,
            super::ResumeDisposition::Consumed,
            id + 2,
        );
    }
    registry
        .finish_transaction_custody(
            ExecutionTransactionId::new(1).unwrap(),
            super::TransactionCustodyOutcome::Committed {
                started_ns: 10,
                finished_ns: 11,
            },
        )
        .unwrap();
    registry.drive(12, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        0
    );
    registry
        .finish_transaction_custody(
            ExecutionTransactionId::new(2).unwrap(),
            super::TransactionCustodyOutcome::RolledBack,
        )
        .unwrap();
    registry.drive(13, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    registry.shutdown(14, 0).unwrap();
}

#[test]
fn new_attach_cancels_pending_provider_lease_release() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let first = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = first.key();
    let mut registry = LoadRegistry::new(
        provider.clone(),
        physical_resources(),
        FairQueueConfig::default(),
    )
    .unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            [stage_request(
                first,
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            1,
        )
        .unwrap();
    registry
        .detach_continuation(ContinuationId::new(1), CancellationReason::Superseded, 2)
        .unwrap();

    let second = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    registry
        .attach_waiter(
            waiter(2, 2),
            [stage_request(
                second,
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            3,
        )
        .unwrap();
    registry.drive(4, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        0
    );
    registry
        .detach_continuation(ContinuationId::new(2), CancellationReason::Superseded, 5)
        .unwrap();
    registry.drive(6, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    registry.shutdown(7, 0).unwrap();
}

#[test]
fn registry_retries_failed_provider_lease_release_without_replay() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = preparation.key();
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    registry
        .attach_waiter(
            waiter(9, 9),
            [stage_request(
                preparation,
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            1,
        )
        .unwrap();
    handle.fail_next_release(FailureReason::DeviceUnavailable);
    registry
        .detach_continuation(ContinuationId::new(9), CancellationReason::Superseded, 2)
        .unwrap();
    assert!(registry.drive(3, 1).is_err());
    assert_eq!(registry.residency_binding(key), Some(handle.binding(key)));
    registry.drive(4, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        2
    );
    registry.shutdown(4, 0).unwrap();
}

#[test]
fn queued_install_success_after_last_detach_releases_lease_once() {
    let (mut registry, handle) = registry(false);
    let key = key(1);
    let operation = attach(&mut registry, waiter(1, 1), key, 1);
    registry.schedule_one(2).unwrap();
    handle.script_outcome(LoadStage::ReadSubmitted, CompletionOutcome::Succeeded);
    registry.schedule_one(3).unwrap();
    apply_physical(&mut registry, 1);
    handle.script_outcome(LoadStage::UploadSubmitted, CompletionOutcome::Succeeded);
    registry.schedule_one(4).unwrap();
    apply_physical(&mut registry, 1);
    handle.script_outcome(LoadStage::Installing, CompletionOutcome::Succeeded);
    registry.schedule_one(5).unwrap();
    assert_eq!(registry.collect_provider_completions(1), 1);

    registry
        .detach_continuation(ContinuationId::new(1), CancellationReason::Superseded, 6)
        .unwrap();
    registry.process_one_completion_at(7).unwrap();
    assert!(registry.operation(operation).is_none());
    assert_eq!(registry.residency_binding(key), Some(handle.binding(key)));
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        0
    );
    registry.drive(8, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    registry.drive(9, 1).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    registry.shutdown(10, 0).unwrap();
}

#[test]
fn replacement_commit_swaps_exact_slot_generation_and_bytes() {
    let (physical, handle) = MockPhysicalProvider::automatic();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let old_preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let old_key = old_preparation.key();
    let slot = old_preparation.binding().slot;
    let mut registry = LoadRegistry::new(
        provider.clone(),
        physical_resources(),
        FairQueueConfig::default(),
    )
    .unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            [stage_request(
                old_preparation,
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            1,
        )
        .unwrap();
    registry
        .detach_continuation(ContinuationId::new(1), CancellationReason::Superseded, 2)
        .unwrap();
    registry.drive(3, 1).unwrap();

    handle.set_resident(false);
    handle.configure_next_preparation(DESTINATION_GENERATION + 1, slot, Some(old_key));
    let new_preparation = provider
        .prepare(request(2), MaterializationPurpose::Execution)
        .unwrap();
    let new_key = new_preparation.key();
    registry
        .attach_waiter(
            waiter(2, 2),
            [stage_request(
                new_preparation,
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            4,
        )
        .unwrap();
    registry.drive(100, 32).unwrap();

    assert!(registry.residency_binding(old_key).is_none());
    assert_eq!(
        registry.residency_binding(new_key),
        Some(handle.binding(new_key))
    );
    assert_eq!(registry.resident_entries(), 1);
    assert_eq!(
        registry.resources().in_use(ResourceKind::ResidentBytes),
        BYTES
    );
    assert_eq!(
        registry.pop_ready(101).unwrap(),
        Some(ContinuationId::new(2))
    );
    registry
        .detach_continuation(ContinuationId::new(2), CancellationReason::Superseded, 102)
        .unwrap();
    registry.shutdown(103, 0).unwrap();
}

#[test]
fn replacement_waits_for_old_logical_owner_without_losing_operation() {
    let (physical, handle) = MockPhysicalProvider::automatic();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let old_preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let old_key = old_preparation.key();
    let slot = old_preparation.binding().slot;
    let mut registry = LoadRegistry::new(
        provider.clone(),
        physical_resources(),
        FairQueueConfig::default(),
    )
    .unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            [stage_request(
                old_preparation,
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            1,
        )
        .unwrap();

    handle.set_resident(false);
    handle.configure_next_preparation(DESTINATION_GENERATION + 1, slot, Some(old_key));
    let new_preparation = provider
        .prepare(request(2), MaterializationPurpose::Execution)
        .unwrap();
    let new_key = new_preparation.key();
    let operation = registry
        .attach_waiter(
            waiter(2, 2),
            [stage_request(
                new_preparation,
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            2,
        )
        .unwrap()
        .created[0];
    registry.drive(100, 32).unwrap();

    assert_eq!(
        registry.residency_binding(old_key),
        Some(handle.binding(old_key))
    );
    assert!(registry.residency_binding(new_key).is_none());
    assert_eq!(
        registry
            .operation(operation)
            .map(|operation| operation.stage()),
        Some(LoadStage::Resident)
    );
    assert!(registry.retirement(operation).is_none());
    assert_eq!(
        registry.pop_ready(101).unwrap(),
        Some(ContinuationId::new(1))
    );

    registry
        .detach_continuation(ContinuationId::new(1), CancellationReason::Superseded, 102)
        .unwrap();
    registry.drive(103, 8).unwrap();
    assert!(registry.residency_binding(old_key).is_none());
    assert_eq!(
        registry.residency_binding(new_key),
        Some(handle.binding(new_key))
    );
    assert!(registry.operation(operation).is_none());
    assert!(registry.retirement(operation).is_some());
    registry
        .detach_continuation(ContinuationId::new(2), CancellationReason::Superseded, 104)
        .unwrap();
    registry.shutdown(105, 0).unwrap();
}

#[test]
fn replacement_cannot_evict_key_from_another_slot() {
    let (physical, handle) = MockPhysicalProvider::automatic();
    handle.set_resident(true);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let old_preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let old_key = old_preparation.key();
    let wrong_slot = DestinationSlotId::new(old_preparation.binding().slot.get() + 1);
    let mut registry = LoadRegistry::new(
        provider.clone(),
        physical_resources(),
        FairQueueConfig::default(),
    )
    .unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            [stage_request(
                old_preparation,
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            1,
        )
        .unwrap();

    handle.set_resident(false);
    handle.configure_next_preparation(DESTINATION_GENERATION + 1, wrong_slot, Some(old_key));
    let new_preparation = provider
        .prepare(request(2), MaterializationPurpose::Execution)
        .unwrap();
    let new_key = new_preparation.key();
    registry
        .attach_waiter(
            waiter(2, 2),
            [stage_request(
                new_preparation,
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            2,
        )
        .unwrap();
    assert!(matches!(
        registry.drive(100, 32),
        Err(super::RegistryError::PublishedResidencyConflict { key: _ })
    ));
    assert_eq!(
        registry.residency_binding(old_key),
        Some(handle.binding(old_key))
    );
    assert!(registry.residency_binding(new_key).is_none());
    assert_eq!(registry.resident_entries(), 1);
}

#[test]
fn pre_reserve_cancellation_discards_provider_preparation() {
    let (physical, handle) = MockPhysicalProvider::manual();
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = preparation.key();
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    registry
        .attach_waiter(
            waiter(11, 11),
            [stage_request(
                preparation,
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            1,
        )
        .unwrap();
    registry
        .detach_continuation(ContinuationId::new(11), CancellationReason::Superseded, 2)
        .unwrap();
    assert_eq!(registry.active_operations(), 0);
    assert!(registry.provider().preparation(key).is_err());
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::DiscardPreparation(candidate) if *candidate == key
        )),
        1
    );
}

#[test]
fn resolver_surfaces_provider_prepare_failure() {
    let (physical, handle) = MockPhysicalProvider::manual();
    handle.fail_next_resolve(FailureReason::StorageUnavailable);
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let mut resolver = RuntimeMaterializationResolver::new(handle.placement(), Some(provider));
    assert!(resolver.resolve(request(1)).is_err());
}

#[test]
fn physical_bridge_registry_surfaces_reserve_failure_without_credit_leak() {
    let (mut registry, handle) = registry(false);
    handle.fail_next_reserve(FailureReason::DeviceUnavailable);
    let report = registry
        .attach_waiter(
            waiter(1, 1),
            [load_request(
                &registry,
                key(1),
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            1,
        )
        .unwrap();
    assert_eq!(report.created.len(), 1);
    assert!(registry.schedule_one(2).unwrap());
    assert!(matches!(
        registry.pop_failed().unwrap().failure,
        ContinuationFailure::Failed(FailureReason::DeviceUnavailable)
    ));
    assert_eq!(registry.resources().active_grants(), 0);
}

#[test]
fn physical_bridge_execution_promotes_prefetch_exactly_once_and_can_release_back() {
    let (mut registry, handle, key) = prefetch_registry(true);
    let binding = registry.provider().preparation(key).unwrap().binding();
    let prefetch = registry
        .prefetch(
            super::PrefetchId::new(1).unwrap(),
            [stage_request(
                registry.provider().preparation(key).unwrap(),
                uniform_plan(),
                ResourceClass::Prefetch,
            )],
            1,
        )
        .unwrap();
    let operation = prefetch.created[0];
    let joined = registry
        .attach_waiter(
            waiter(1, 1),
            [stage_request(
                registry.provider().preparation(key).unwrap(),
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            2,
        )
        .unwrap();

    assert_eq!(joined.joined, [operation]);
    assert_eq!(registry.operation(operation).unwrap().key(), key);
    assert_eq!(
        registry.provider().preparation(key).unwrap().binding(),
        binding
    );
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::PromoteToExecution(candidate) if *candidate == key
        )),
        1
    );

    registry
        .detach_waiter(waiter(1, 1), CancellationReason::ExternalRequest, 3)
        .unwrap();
    assert!(registry.operation_has_prefetch_owner(operation));
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == key
        )),
        1
    );
    assert_eq!(
        registry.operation(operation).unwrap().class(),
        ResourceClass::Prefetch
    );
    assert_eq!(
        registry.provider().preparation(key).unwrap().binding(),
        binding
    );
}

#[test]
fn failed_multi_key_promotion_restores_prefetch_and_reclaims_new_work() {
    let (physical, handle) = MockPhysicalProvider::manual();
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let mut preparations = [
        provider
            .prepare(request(1), MaterializationPurpose::Prefetch)
            .unwrap(),
        provider
            .prepare(request(2), MaterializationPurpose::Prefetch)
            .unwrap(),
    ];
    preparations.sort_unstable_by_key(|preparation| preparation.key());
    let prefetch_key = preparations[0].key();
    let failing_key = preparations[1].key();
    let owner = super::PrefetchId::new(41).unwrap();
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    let prefetched = registry
        .prefetch(
            owner,
            [stage_request(
                preparations[0],
                uniform_plan(),
                ResourceClass::Prefetch,
            )],
            1,
        )
        .unwrap();
    let prefetch_operation = prefetched.created[0];
    let baseline_grants = registry.resources().active_grants();
    handle.fail_next_promotion(failing_key, FailureReason::DeviceUnavailable);

    let execution_waiter = waiter(51, 51);
    let error = registry
        .attach_waiter(
            execution_waiter,
            preparations.map(|preparation| {
                stage_request(preparation, uniform_plan(), ResourceClass::Throughput)
            }),
            2,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        super::RegistryError::Provider {
            source: FailureReason::DeviceUnavailable
        }
    ));
    assert_eq!(registry.active_operations(), 1);
    assert_eq!(registry.active_prefetches(), 1);
    assert_eq!(
        registry.operation_for_key(prefetch_key),
        Some(prefetch_operation)
    );
    assert_eq!(registry.operation_for_key(failing_key), None);
    assert!(registry.operation_has_prefetch_owner(prefetch_operation));
    assert_eq!(
        registry.operation(prefetch_operation).unwrap().class(),
        ResourceClass::Prefetch
    );
    assert_eq!(registry.resources().active_grants(), baseline_grants);
    assert_eq!(registry.waiters().active_waiters().count(), 0);
    assert!(registry.provider().preparation(failing_key).is_err());
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::ReleaseExecutionLease(candidate) if *candidate == prefetch_key
        )),
        1
    );
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::DiscardPreparation(candidate) if *candidate == failing_key
        )),
        1
    );

    let retry = registry
        .attach_waiter(
            execution_waiter,
            [stage_request(
                registry.provider().preparation(prefetch_key).unwrap(),
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            3,
        )
        .unwrap();
    assert_eq!(retry.joined, [prefetch_operation]);
    registry
        .detach_waiter(execution_waiter, CancellationReason::ExternalRequest, 4)
        .unwrap();
    assert!(registry.operation_has_prefetch_owner(prefetch_operation));
}

#[test]
fn physical_bridge_single_flight_issues_one_physical_read() {
    let (mut registry, handle) = registry(true);
    let key = key(1);
    let operation = attach(&mut registry, waiter(1, 1), key, 1);
    let joined_request = load_request(&registry, key, uniform_plan(), ResourceClass::Throughput);
    let joined = registry
        .attach_waiter(waiter(2, 2), [joined_request], 2)
        .unwrap();
    assert_eq!(joined.joined, vec![operation]);
    registry.drive(100, 32).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
        1
    );
}

#[test]
fn physical_bridge_reverse_completion_wakes_only_target() {
    let (mut registry, handle) = registry(false);
    let first_key = key(1);
    let second_key = key(2);
    let first = attach(&mut registry, waiter(1, 11), first_key, 1);
    let second = attach(&mut registry, waiter(2, 22), second_key, 2);
    for now in 3..7 {
        assert!(registry.schedule_one(now).unwrap());
    }
    handle.push_outcome(
        second,
        second_key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Succeeded,
    );
    apply_physical(&mut registry, 4);
    assert!(registry.schedule_one(5).unwrap());
    handle.push_outcome(
        second,
        second_key,
        LoadStage::UploadSubmitted,
        CompletionOutcome::Succeeded,
    );
    apply_physical(&mut registry, 4);
    assert!(registry.schedule_one(6).unwrap());
    handle.push_outcome(
        second,
        second_key,
        LoadStage::Installing,
        CompletionOutcome::Succeeded,
    );
    apply_physical(&mut registry, 4);
    assert_eq!(
        registry.pop_ready(7).unwrap(),
        Some(ContinuationId::new(22))
    );
    assert_eq!(registry.pop_ready(7).unwrap(), None);
    assert_eq!(
        registry.operation(first).unwrap().stage(),
        LoadStage::ReadSubmitted
    );
}

#[test]
fn physical_bridge_cancel_one_waiter_retains_shared_operation() {
    let (mut registry, handle) = registry(false);
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    let joined_request = load_request(&registry, key, uniform_plan(), ResourceClass::Throughput);
    registry
        .attach_waiter(waiter(2, 2), [joined_request], 2)
        .unwrap();
    registry.schedule_one(3).unwrap();
    registry
        .detach_waiter(waiter(1, 1), CancellationReason::ExternalRequest, 4)
        .unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Cancel(..))),
        0
    );
}

#[test]
fn physical_bridge_cancel_last_queued_skips_physical_cancel() {
    let (mut registry, handle) = registry(false);
    attach(&mut registry, waiter(1, 1), key(1), 1);
    registry
        .detach_waiter(waiter(1, 1), CancellationReason::ExternalRequest, 2)
        .unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Reserve(..))),
        0
    );
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Cancel(..))),
        0
    );
    assert_eq!(registry.resources().active_grants(), 0);
}

#[test]
fn physical_bridge_cancel_last_reserved_calls_physical_cancel() {
    let (mut registry, handle) = registry(false);
    attach(&mut registry, waiter(1, 1), key(1), 1);
    assert!(registry.schedule_one(2).unwrap());
    registry
        .detach_waiter(waiter(1, 1), CancellationReason::ExternalRequest, 3)
        .unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::Cancel(_, _, LoadStage::Reserved, _)
        )),
        1
    );
    assert_eq!(registry.resources().active_grants(), 0);
}

#[test]
fn physical_bridge_cancel_last_submitted_drains_completion() {
    let (mut registry, handle, _, operation) = manual_at_read();
    registry
        .detach_waiter(waiter(1, 1), CancellationReason::ExternalRequest, 3)
        .unwrap();
    assert!(registry.operation(operation).is_some());
    apply_physical(&mut registry, 4);
    assert!(registry.operation(operation).is_none());
    assert_eq!(registry.resources().active_grants(), 0);
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::Cancel(_, _, LoadStage::ReadSubmitted, _)
        )),
        1
    );
}

#[test]
fn physical_bridge_cancel_host_ready_calls_physical_cancel() {
    let (mut registry, handle, key, operation) = manual_at_read();
    handle.push_outcome(
        operation,
        key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Succeeded,
    );
    apply_physical(&mut registry, 4);
    registry
        .detach_waiter(waiter(1, 1), CancellationReason::ExternalRequest, 4)
        .unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::Cancel(_, _, LoadStage::HostReady, _)
        )),
        1
    );
    assert!(registry.operation(operation).is_none());
}

#[test]
fn physical_bridge_cancel_install_ready_calls_physical_cancel() {
    let (mut registry, handle, key, operation) = manual_at_read();
    handle.push_outcome(
        operation,
        key,
        LoadStage::ReadSubmitted,
        CompletionOutcome::Succeeded,
    );
    apply_physical(&mut registry, 4);
    registry.schedule_one(4).unwrap();
    handle.push_outcome(
        operation,
        key,
        LoadStage::UploadSubmitted,
        CompletionOutcome::Succeeded,
    );
    apply_physical(&mut registry, 4);
    registry
        .detach_waiter(waiter(1, 1), CancellationReason::ExternalRequest, 5)
        .unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::Cancel(_, _, LoadStage::Installing, _)
        )),
        1
    );
    assert!(registry.operation(operation).is_none());
}

#[test]
fn physical_bridge_read_failure_fails_all_waiters() {
    let (mut registry, handle) = registry(true);
    handle.script_outcome(
        LoadStage::ReadSubmitted,
        CompletionOutcome::Failed(FailureReason::StorageUnavailable),
    );
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    let joined_request = load_request(&registry, key, uniform_plan(), ResourceClass::Throughput);
    registry
        .attach_waiter(waiter(2, 2), [joined_request], 2)
        .unwrap();
    registry.drive(100, 16).unwrap();
    let failures = [
        registry.pop_failed().unwrap(),
        registry.pop_failed().unwrap(),
    ];
    assert!(failures.iter().all(|failure| matches!(
        failure.failure,
        ContinuationFailure::Failed(FailureReason::StorageUnavailable)
    )));
}

#[test]
fn physical_bridge_upload_failure_never_publishes() {
    let (mut registry, handle) = registry(true);
    handle.script_outcome(
        LoadStage::UploadSubmitted,
        CompletionOutcome::Failed(FailureReason::UploadRejected),
    );
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    registry.drive(100, 16).unwrap();
    assert!(registry.residency_binding(key).is_none());
    assert!(matches!(
        registry.pop_failed().unwrap().failure,
        ContinuationFailure::Failed(FailureReason::UploadRejected)
    ));
}

#[test]
fn physical_bridge_install_failure_never_publishes() {
    let (mut registry, handle) = registry(true);
    handle.script_outcome(
        LoadStage::Installing,
        CompletionOutcome::Failed(FailureReason::InstallationRejected),
    );
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    registry.drive(100, 16).unwrap();
    assert!(registry.residency_binding(key).is_none());
    assert!(registry.pop_failed().is_some());
}

#[test]
fn physical_bridge_read_stale_never_publishes() {
    let (mut registry, handle) = registry(true);
    handle.script_outcome(
        LoadStage::ReadSubmitted,
        CompletionOutcome::Stale(StaleReason::SourceIdentityChanged),
    );
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    registry.drive(100, 16).unwrap();
    assert!(registry.residency_binding(key).is_none());
    assert!(matches!(
        registry.pop_failed().unwrap().failure,
        ContinuationFailure::Stale(StaleReason::SourceIdentityChanged)
    ));
}

#[test]
fn physical_bridge_upload_stale_never_publishes() {
    let (mut registry, handle) = registry(true);
    handle.script_outcome(
        LoadStage::UploadSubmitted,
        CompletionOutcome::Stale(StaleReason::DestinationReused),
    );
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    registry.drive(100, 16).unwrap();
    assert!(registry.residency_binding(key).is_none());
}

#[test]
fn physical_bridge_install_stale_never_publishes() {
    let (mut registry, handle) = registry(true);
    handle.script_outcome(
        LoadStage::Installing,
        CompletionOutcome::Stale(StaleReason::SupersededOperation),
    );
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    registry.drive(100, 16).unwrap();
    assert!(registry.residency_binding(key).is_none());
}

#[test]
fn registry_publication_uses_provider_binding_without_reconciliation() {
    let (physical, handle) = MockPhysicalProvider::automatic();
    let provider = SharedMaterializationProvider::new(Box::new(physical));
    let preparation = provider
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    let key = preparation.key();
    let mut registry =
        LoadRegistry::new(provider, physical_resources(), FairQueueConfig::default()).unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            [stage_request(
                preparation,
                uniform_plan(),
                ResourceClass::Throughput,
            )],
            1,
        )
        .unwrap();
    registry.drive(100, 32).unwrap();
    assert_eq!(registry.residency_binding(key), Some(handle.binding(key)));
}

#[test]
fn physical_bridge_shutdown_cancels_and_drains_submitted_work() {
    let (mut registry, handle, _, _) = manual_at_read();
    let report = registry.shutdown(10, 8).unwrap();
    assert!(report.drained);
    assert_eq!(report.active_grants, 0);
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::Cancel(_, _, LoadStage::ReadSubmitted, _)
        )),
        1
    );
}

#[test]
fn forty_dependencies_attach_and_complete_with_qd_two() {
    const DEPENDENCIES: u8 = 40;
    const QD: u64 = 2;

    let (physical, handle) = MockPhysicalProvider::manual();
    let backend = SharedMaterializationProvider::new(Box::new(physical));
    for seed in 1..=DEPENDENCIES {
        backend
            .prepare(request(seed), MaterializationPurpose::Execution)
            .unwrap();
    }
    let mut resources = bounded_physical_resources(u64::from(DEPENDENCIES), QD, QD, QD);
    resources
        .reconfigure_limit(ResourceKind::LoadOperation, QD, 0)
        .unwrap();
    let mut registry = LoadRegistry::new(backend, resources, FairQueueConfig::default()).unwrap();
    let requests = (1..=DEPENDENCIES)
        .map(|seed| {
            load_request(
                &registry,
                key(seed),
                uniform_plan(),
                ResourceClass::Throughput,
            )
        })
        .collect::<Vec<_>>();

    let report = registry.attach_waiter(waiter(1, 1), requests, 1).unwrap();
    assert_eq!(report.created.len(), usize::from(DEPENDENCIES));
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Reserve(..))),
        0
    );
    assert_eq!(registry.resources().in_use(ResourceKind::ReadSlot), 0);
    assert_eq!(registry.resources().in_use(ResourceKind::LoadOperation), 0);

    registry.drive(10, 128).unwrap();
    let submitted = report
        .created
        .iter()
        .copied()
        .filter(|operation| {
            registry.operation(*operation).unwrap().stage() == LoadStage::ReadSubmitted
        })
        .collect::<Vec<_>>();
    assert_eq!(submitted.len(), QD as usize);
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
        QD as usize
    );

    for _ in QD..u64::from(DEPENDENCIES) {
        handle.script_outcome(LoadStage::ReadSubmitted, CompletionOutcome::Succeeded);
    }
    for _ in 0..u64::from(DEPENDENCIES) {
        handle.script_outcome(LoadStage::UploadSubmitted, CompletionOutcome::Succeeded);
        handle.script_outcome(LoadStage::Installing, CompletionOutcome::Succeeded);
    }
    for operation in submitted {
        handle.push_outcome(
            operation,
            registry.key_for_operation(operation).unwrap(),
            LoadStage::ReadSubmitted,
            CompletionOutcome::Succeeded,
        );
    }

    registry.drive(100, 2048).unwrap();
    assert_eq!(registry.stats().publications, u64::from(DEPENDENCIES));
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
        usize::from(DEPENDENCIES)
    );
    let sqe = registry
        .resources()
        .snapshots()
        .find(|snapshot| snapshot.kind == ResourceKind::ReadSlot)
        .unwrap();
    assert_eq!(sqe.capacity, QD);
    assert_eq!(sqe.high_water, QD);
    assert_eq!(sqe.in_use, 0);
    let load_operations = registry
        .resources()
        .snapshots()
        .find(|snapshot| snapshot.kind == ResourceKind::LoadOperation)
        .unwrap();
    assert_eq!(load_operations.capacity, QD);
    assert_eq!(load_operations.high_water, QD);
    assert_eq!(load_operations.in_use, 0);

    let shutdown = registry.shutdown(200, 128).unwrap();
    assert!(shutdown.drained);
    assert_eq!(shutdown.active_grants, 0);
}

#[test]
fn physical_bridge_resource_high_water_uses_real_bytes() {
    let (mut registry, _) = registry(true);
    let key = key(1);
    attach(&mut registry, waiter(1, 1), key, 1);
    registry.drive(100, 32).unwrap();
    let high_water = |kind| {
        registry
            .resources()
            .snapshots()
            .find(|snapshot| snapshot.kind == kind)
            .unwrap()
            .high_water
    };
    assert_eq!(high_water(ResourceKind::PinnedHostBytes), BYTES);
    assert_eq!(high_water(ResourceKind::StorageReadBytes), BYTES);
    assert_eq!(high_water(ResourceKind::UploadBytes), BYTES);
    assert_eq!(high_water(ResourceKind::DeviceInstallBytes), BYTES);
    assert_eq!(high_water(ResourceKind::ResidentBytes), BYTES);
    assert_eq!(high_water(ResourceKind::ReadSlot), 1);
    assert_eq!(high_water(ResourceKind::InstallSlot), 1);
    assert_eq!(high_water(ResourceKind::LoadOperation), 1);
}

#[test]
fn physical_bridge_preserves_nonuniform_resource_plan() {
    let plan = MaterializationResourcePlan::new(
        MaterializationResourceRequirements {
            read_slots: 2,
            storage_read_bytes: 4096,
            pinned_host_bytes: 8192,
            upload_slots: 3,
            h2d_bytes: 12_288,
            install_slots: 4,
            device_install_bytes: 16_384,
        },
        20_480,
    )
    .unwrap();
    let limits = MaterializationResourceLimits {
        capacity: MaterializationResourceRequirements {
            device_install_bytes: plan.resident_bytes,
            ..plan.requirements
        },
        execution_reserve: MaterializationResourceRequirements::default(),
    }
    .validate()
    .unwrap();
    let (physical, handle) = MockPhysicalProvider::automatic();
    handle.set_plan_and_limits(plan, limits);
    let mut backend = SharedMaterializationProvider::new(Box::new(physical));
    let resolved = backend
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    assert!(matches!(resolved, MaterializationPreparation::Transfer(_)));
    let key = resolved.key();
    assert_eq!(backend.materialization_plan(key).unwrap(), plan);

    let direct_operation = OperationId::new(77);
    let reservation = backend.reserve(direct_operation, key, plan).unwrap();
    assert_eq!(
        reservation.slabs()[0].descriptor().len(),
        plan.requirements.pinned_host_bytes
    );
    backend
        .submit_read(direct_operation, key, &reservation, plan)
        .unwrap();
    backend
        .submit_upload(direct_operation, key, &reservation, plan)
        .unwrap();
    backend
        .poll_install(direct_operation, key, &reservation, plan)
        .unwrap();
    let completions = [
        backend.next_completion().unwrap(),
        backend.next_completion().unwrap(),
        backend.next_completion().unwrap(),
    ];
    assert_eq!(
        completions.map(|event| (event.stage, event.bytes)),
        [
            (
                LoadStage::ReadSubmitted,
                plan.requirements.storage_read_bytes
            ),
            (LoadStage::UploadSubmitted, plan.requirements.h2d_bytes),
            (
                LoadStage::Installing,
                plan.requirements.device_install_bytes
            ),
        ]
    );
    let commands = handle.commands();
    assert!(commands.contains(&MockPhysicalCommand::Reserve(direct_operation, key, plan,)));
    assert!(commands.contains(&MockPhysicalCommand::SubmitRead(
        direct_operation,
        key,
        plan,
    )));
    assert!(commands.contains(&MockPhysicalCommand::SubmitUpload(
        direct_operation,
        key,
        plan,
    )));
    assert!(commands.contains(&MockPhysicalCommand::PollInstall(
        direct_operation,
        key,
        plan,
    )));
    drop(reservation);

    let (registry_physical, registry_handle) = MockPhysicalProvider::automatic();
    registry_handle.set_plan_and_limits(plan, limits);
    let registry_backend = SharedMaterializationProvider::new(Box::new(registry_physical));
    let registry_resolved = registry_backend
        .prepare(request(1), MaterializationPurpose::Execution)
        .unwrap();
    assert!(matches!(
        registry_resolved,
        MaterializationPreparation::Transfer(_)
    ));
    let registry_key = registry_resolved.key();
    assert_eq!(registry_key, key);

    let resources = PhysicalResourceBroker::new(ResourceKind::ALL.map(|kind| {
        let capacity = match kind {
            ResourceKind::ReadSlot => plan.requirements.read_slots,
            ResourceKind::PinnedHostBytes => plan.requirements.pinned_host_bytes,
            ResourceKind::StorageReadBytes => plan.requirements.storage_read_bytes,
            ResourceKind::UploadSlot => plan.requirements.upload_slots,
            ResourceKind::UploadBytes => plan.requirements.h2d_bytes,
            ResourceKind::InstallSlot => plan.requirements.install_slots,
            ResourceKind::DeviceInstallBytes => plan.requirements.device_install_bytes,
            ResourceKind::ResidentBytes => plan.resident_bytes,
            ResourceKind::ResidencyLease | ResourceKind::LoadOperation => 1,
            ResourceKind::Arena
            | ResourceKind::KvPage
            | ResourceKind::Continuation
            | ResourceKind::Waiter
            | ResourceKind::ReadyCohort => 64,
        };
        PhysicalResourceLimit::new(kind, capacity, 0)
    }))
    .unwrap();
    let mut registry =
        LoadRegistry::new(registry_backend, resources, FairQueueConfig::default()).unwrap();
    registry
        .attach_waiter(
            waiter(1, 1),
            [stage_request(
                registry_resolved,
                plan,
                ResourceClass::Throughput,
            )],
            1,
        )
        .unwrap();
    for now_ns in 100..=108 {
        registry.drive(now_ns, 32).unwrap();
    }
    assert!(registry.residency_binding(registry_key).is_some());

    let high_water = |kind| {
        registry
            .resources()
            .snapshots()
            .find(|snapshot| snapshot.kind == kind)
            .unwrap()
            .high_water
    };
    assert_eq!(
        high_water(ResourceKind::StorageReadBytes),
        plan.requirements.storage_read_bytes
    );
    assert_eq!(
        high_water(ResourceKind::PinnedHostBytes),
        plan.requirements.pinned_host_bytes
    );
    assert_eq!(
        high_water(ResourceKind::UploadBytes),
        plan.requirements.h2d_bytes
    );
    assert_eq!(
        high_water(ResourceKind::DeviceInstallBytes),
        plan.requirements.device_install_bytes
    );
    assert_eq!(high_water(ResourceKind::ResidentBytes), plan.resident_bytes);
    assert_eq!(
        high_water(ResourceKind::ReadSlot),
        plan.requirements.read_slots
    );
    assert_eq!(
        high_water(ResourceKind::UploadSlot),
        plan.requirements.upload_slots
    );
    assert_eq!(
        high_water(ResourceKind::InstallSlot),
        plan.requirements.install_slots
    );
}
