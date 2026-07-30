use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use ferrule_common::CompletionHub;
use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::expert_io::{ExpertIoResourceDemand, ExpertIoResourceLimits};
use ferrule_common::io_protocol::{
    ArtifactFormat, BackendId, CancellationReason, CompletionEvent, CompletionGeneration,
    CompletionOutcome, CompletionTimestamp, ContentHash, ContinuationId, DependencySetEpoch,
    DestinationGeneration, DestinationSlotId, DeviceId, ExpertId, FailureReason, FenceId, LayerId,
    LoadKey, LoadStage, ModelInstanceId, OperationId, RegisteredPinnedAlignedSlabLeaseDescriptor,
    RegistrationId, RequestGeneration, ResidencyBinding, SlabId, SourceGeneration,
    SourceIdentityHash, StaleReason, UploadFenceContract, ValidatedResidencyBinding, WaiterId,
};
use ferrule_model::{
    ExpertArtifactIdentity, ExpertDependencyResolution, ExpertMaterializationAdapter,
    ExpertMaterializationPlacement, ExpertMaterializationRequest,
    PhysicalExpertMaterializationBackend, PhysicalExpertOperationReservation,
    PhysicalExpertReservation, PhysicalExpertReservationDescriptor, PhysicalExpertResourceTopology,
};

use super::{
    CompletionDisposition, ContinuationFailure, FairQueueConfig, LoadRegistry, LoadRequest,
    MaterializationBackend, RunnerMaterializationBackend, RuntimeMaterializationControl,
};
use crate::scheduling::{HardResourceBroker, HardResourceLimit, ResourceClass, ResourceKind};

const BYTES: u64 = 4096;
const DESTINATION_GENERATION: u64 = 37;
const SLOT: u32 = 9;
const SLAB_ID_OFFSET: u64 = 1000;
const REGISTRATION_ID_OFFSET: u64 = 2000;
const FENCE_ID_OFFSET: u64 = 3000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MockPhysicalCommand {
    Resolve(ExpertMaterializationRequest),
    MaterializationBytes(LoadKey),
    ReleaseSelected(LoadKey),
    Reserve(OperationId, LoadKey, u64),
    SubmitRead(OperationId, LoadKey, u64),
    SubmitUpload(OperationId, LoadKey, u64),
    PollInstall(OperationId, LoadKey, u64),
    Cancel(OperationId, LoadKey, LoadStage, CancellationReason),
    PhysicalDropped,
}

#[derive(Debug)]
struct MockPhysicalState {
    placement: ExpertMaterializationPlacement,
    limits: ExpertIoResourceLimits,
    bytes: u64,
    generation: DestinationGeneration,
    slot: DestinationSlotId,
    automatic: bool,
    resident: bool,
    resolve_failure: Option<FailureReason>,
    reserve_failure: Option<FailureReason>,
    release_failures: VecDeque<FailureReason>,
    reservation_key_override: Option<LoadKey>,
    reservation_operation_override: Option<OperationId>,
    reservation_slot_override: Option<DestinationSlotId>,
    resolved: BTreeMap<ExpertMaterializationRequest, (LoadKey, ResidencyBinding)>,
    commands: Vec<MockPhysicalCommand>,
    completions: VecDeque<CompletionEvent>,
    scripted_outcomes: BTreeMap<LoadStage, VecDeque<CompletionOutcome>>,
    lost_completions: BTreeMap<LoadStage, usize>,
    clock_ns: u64,
}

impl MockPhysicalState {
    fn new(automatic: bool) -> Self {
        let placement = ExpertMaterializationPlacement::new(
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
            limits: ExpertIoResourceLimits {
                capacity: ExpertIoResourceDemand {
                    read_slots: operation_capacity,
                    storage_read_bytes: byte_capacity,
                    pinned_host_bytes: byte_capacity,
                    upload_slots: operation_capacity,
                    h2d_bytes: byte_capacity,
                    install_slots: operation_capacity,
                    device_install_bytes: byte_capacity,
                },
                demand_reserve: ExpertIoResourceDemand {
                    read_slots: operation_reserve,
                    storage_read_bytes: BYTES,
                    pinned_host_bytes: BYTES,
                    upload_slots: operation_reserve,
                    h2d_bytes: BYTES,
                    install_slots: operation_reserve,
                    device_install_bytes: BYTES,
                },
            },
            bytes: BYTES,
            generation: DestinationGeneration::new(DESTINATION_GENERATION),
            slot: DestinationSlotId::new(SLOT),
            automatic,
            resident: false,
            resolve_failure: None,
            reserve_failure: None,
            release_failures: VecDeque::new(),
            reservation_key_override: None,
            reservation_operation_override: None,
            reservation_slot_override: None,
            resolved: BTreeMap::new(),
            commands: Vec::new(),
            completions: VecDeque::new(),
            scripted_outcomes: BTreeMap::new(),
            lost_completions: BTreeMap::new(),
            clock_ns: 1,
        }
    }

    fn binding_for(&self, key: LoadKey) -> ResidencyBinding {
        ResidencyBinding::new(
            key.model(),
            key.layer(),
            key.expert(),
            key.backend(),
            key.device(),
            self.slot,
            key.destination_generation(),
        )
    }

    fn emit(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        stage: LoadStage,
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
        let bytes = if matches!(outcome, CompletionOutcome::Succeeded) {
            self.bytes
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
pub(crate) struct MockPhysicalBackend {
    state: Arc<Mutex<MockPhysicalState>>,
}

#[derive(Debug, Clone)]
pub(crate) struct MockPhysicalHandle {
    state: Arc<Mutex<MockPhysicalState>>,
}

impl Drop for MockPhysicalBackend {
    fn drop(&mut self) {
        self.lock()
            .commands
            .push(MockPhysicalCommand::PhysicalDropped);
    }
}

impl MockPhysicalBackend {
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

    pub(crate) fn placement(&self) -> ExpertMaterializationPlacement {
        self.lock().placement
    }

    pub(crate) fn limits(&self) -> ExpertIoResourceLimits {
        self.lock().limits
    }

    pub(crate) fn set_bytes_and_limits(&self, bytes: u64, limits: ExpertIoResourceLimits) {
        let mut state = self.lock();
        state.bytes = bytes;
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

    pub(crate) fn binding(&self, key: LoadKey) -> ResidencyBinding {
        self.lock().binding_for(key)
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

    pub(crate) fn fail_next_release(&self, failure: FailureReason) {
        self.lock().release_failures.push_back(failure);
    }

    fn override_reservation_key(&self, key: LoadKey) {
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
        key: LoadKey,
        stage: LoadStage,
        outcome: CompletionOutcome,
    ) -> CompletionEvent {
        let mut state = self.lock();
        let bytes = if matches!(outcome, CompletionOutcome::Succeeded) {
            state.bytes
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

impl PhysicalExpertMaterializationBackend for MockPhysicalBackend {
    fn placement(&self) -> ExpertMaterializationPlacement {
        self.lock().placement
    }

    fn resource_topology(&self) -> ferrule_common::Result<PhysicalExpertResourceTopology> {
        let limits = self.lock().limits;
        PhysicalExpertResourceTopology::new(
            limits,
            limits.capacity.device_install_bytes,
            limits.capacity.install_slots,
        )
    }

    fn resolve_or_reserve(
        &mut self,
        request: ExpertMaterializationRequest,
    ) -> Result<PhysicalExpertReservation, FailureReason> {
        let mut state = self.lock();
        state.commands.push(MockPhysicalCommand::Resolve(request));
        if let Some(failure) = state.resolve_failure.take() {
            return Err(failure);
        }
        if request.model() != state.placement.model()
            || request.backend() != state.placement.backend()
            || request.device() != state.placement.device()
        {
            return Err(FailureReason::ProtocolViolation(
                "mock request does not match physical placement".into(),
            ));
        }
        let (key, binding) = match state.resolved.get(&request).copied() {
            Some(resolved) => resolved,
            None => {
                let key = request
                    .load_key(state.generation)
                    .map_err(|error| FailureReason::ProtocolViolation(error.to_string()))?;
                let binding = state.binding_for(key);
                state.resolved.insert(request, (key, binding));
                (key, binding)
            }
        };
        if state.resident {
            Ok(PhysicalExpertReservation::Resident(
                ValidatedResidencyBinding::new(key, binding)
                    .map_err(|error| FailureReason::ProtocolViolation(error.to_string()))?,
            ))
        } else {
            Ok(PhysicalExpertReservation::Reserved(
                PhysicalExpertReservationDescriptor::new(key, binding, None)
                    .map_err(|error| FailureReason::ProtocolViolation(error.to_string()))?,
            ))
        }
    }

    fn materialization_bytes(&self, key: LoadKey) -> Result<u64, FailureReason> {
        let mut state = self.lock();
        state
            .commands
            .push(MockPhysicalCommand::MaterializationBytes(key));
        Ok(state.bytes)
    }

    fn release_selected(&mut self, key: LoadKey) -> Result<(), FailureReason> {
        let mut state = self.lock();
        state
            .commands
            .push(MockPhysicalCommand::ReleaseSelected(key));
        if let Some(failure) = state.release_failures.pop_front() {
            Err(failure)
        } else {
            Ok(())
        }
    }

    fn reserve(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        bytes: u64,
    ) -> Result<PhysicalExpertOperationReservation, FailureReason> {
        let mut state = self.lock();
        state
            .commands
            .push(MockPhysicalCommand::Reserve(operation, key, bytes));
        if let Some(failure) = state.reserve_failure.take() {
            return Err(failure);
        }
        if bytes != state.bytes {
            return Err(FailureReason::ProtocolViolation(
                "mock physical byte expectation mismatch".into(),
            ));
        }
        let reservation_key = state.reservation_key_override.take().unwrap_or(key);
        let reservation_operation = state
            .reservation_operation_override
            .take()
            .unwrap_or(operation);
        let binding = match state.reservation_slot_override.take() {
            Some(slot) => ResidencyBinding::new(
                reservation_key.model(),
                reservation_key.layer(),
                reservation_key.expert(),
                reservation_key.backend(),
                reservation_key.device(),
                slot,
                reservation_key.destination_generation(),
            ),
            None => state.binding_for(reservation_key),
        };
        let identity = reservation_operation.get();
        let descriptor = RegisteredPinnedAlignedSlabLeaseDescriptor::new(
            reservation_operation,
            SlabId::new(identity.saturating_add(SLAB_ID_OFFSET)),
            RegistrationId::new(identity.saturating_add(REGISTRATION_ID_OFFSET)),
            0x10000,
            bytes,
            0,
            bytes,
            4096,
            reservation_key.source_generation(),
            reservation_key.destination_generation(),
        )
        .map_err(|error| FailureReason::ProtocolViolation(error.to_string()))?;
        PhysicalExpertOperationReservation::new(
            reservation_key,
            binding,
            [descriptor],
            UploadFenceContract::new(
                reservation_operation,
                FenceId::new(identity.saturating_add(FENCE_ID_OFFSET)),
                reservation_key.destination_generation(),
            ),
        )
        .map_err(|error| FailureReason::ProtocolViolation(error.to_string()))
    }

    fn submit_read(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &PhysicalExpertOperationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason> {
        if reservation.key() != key || reservation.upload_fence().operation != operation {
            return Err(FailureReason::ProtocolViolation(
                "mock received a mismatched read reservation".into(),
            ));
        }
        let mut state = self.lock();
        state
            .commands
            .push(MockPhysicalCommand::SubmitRead(operation, key, bytes));
        state.emit(
            operation,
            key,
            LoadStage::ReadSubmitted,
            CompletionOutcome::Succeeded,
        );
        Ok(())
    }

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &PhysicalExpertOperationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason> {
        if reservation.key() != key || reservation.upload_fence().operation != operation {
            return Err(FailureReason::ProtocolViolation(
                "mock received a mismatched upload reservation".into(),
            ));
        }
        let mut state = self.lock();
        state
            .commands
            .push(MockPhysicalCommand::SubmitUpload(operation, key, bytes));
        state.emit(
            operation,
            key,
            LoadStage::UploadSubmitted,
            CompletionOutcome::Succeeded,
        );
        Ok(())
    }

    fn poll_install(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &PhysicalExpertOperationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason> {
        if reservation.key() != key || reservation.binding() != state_binding(self, key) {
            return Err(FailureReason::ProtocolViolation(
                "mock received a mismatched install reservation".into(),
            ));
        }
        let mut state = self.lock();
        state
            .commands
            .push(MockPhysicalCommand::PollInstall(operation, key, bytes));
        state.emit(
            operation,
            key,
            LoadStage::Installing,
            CompletionOutcome::Succeeded,
        );
        Ok(())
    }

    fn cancel(
        &mut self,
        operation: OperationId,
        key: LoadKey,
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

fn state_binding(backend: &MockPhysicalBackend, key: LoadKey) -> ResidencyBinding {
    backend.lock().binding_for(key)
}

fn request(seed: u8) -> ExpertMaterializationRequest {
    let artifact = ExpertArtifactIdentity::new(
        SourceIdentityHash::new([seed.max(1); 32]),
        ContentHash::new([seed.saturating_add(1).max(1); 32]),
        ArtifactFormat::new(1),
        SourceGeneration::new(5),
    )
    .unwrap();
    ExpertMaterializationRequest::for_placement(
        ExpertMaterializationPlacement::new(
            ModelInstanceId::new(17),
            BackendId::new(4),
            DeviceId::new(2),
        )
        .unwrap(),
        artifact,
        LayerId::new(u32::from(seed)),
        ExpertId::new(u32::from(seed)),
    )
    .unwrap()
}

fn key(seed: u8) -> LoadKey {
    request(seed)
        .load_key(DestinationGeneration::new(DESTINATION_GENERATION))
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

fn physical_resources() -> HardResourceBroker {
    bounded_physical_resources(4, 4, 4, 4)
}

fn bounded_physical_resources(
    load_operations: u64,
    sqe: u64,
    pinned_operations: u64,
    upload_slots: u64,
) -> HardResourceBroker {
    HardResourceBroker::new(ResourceKind::ALL.map(|kind| {
        let capacity = match kind {
            ResourceKind::Sqe => sqe,
            ResourceKind::PinnedSlab => BYTES * pinned_operations,
            ResourceKind::ReadBytes => BYTES * sqe,
            ResourceKind::UploadSlot => upload_slots,
            ResourceKind::UploadBytes => BYTES * upload_slots,
            ResourceKind::ExpertFrame => BYTES * load_operations,
            ResourceKind::Lease | ResourceKind::LoadOperation => load_operations,
            ResourceKind::Arena
            | ResourceKind::KvPage
            | ResourceKind::Continuation
            | ResourceKind::Waiter
            | ResourceKind::ReadyCohort => 64,
        };
        HardResourceLimit::new(kind, capacity, 0)
    }))
    .unwrap()
}

fn resolved_backend() -> (RunnerMaterializationBackend, MockPhysicalHandle, LoadKey) {
    let (physical, handle) = MockPhysicalBackend::manual();
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    let PhysicalExpertReservation::Reserved(reservation) =
        backend.resolve_or_reserve(request(1)).unwrap()
    else {
        panic!("mock physical backend must reserve a canonical key");
    };
    (backend, handle, reservation.key())
}

fn registry(
    automatic: bool,
) -> (
    LoadRegistry<RunnerMaterializationBackend>,
    MockPhysicalHandle,
) {
    let (physical, handle) = if automatic {
        MockPhysicalBackend::automatic()
    } else {
        MockPhysicalBackend::manual()
    };
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    for seed in 1..=4 {
        backend.resolve_or_reserve(request(seed)).unwrap();
    }
    (
        LoadRegistry::new(backend, physical_resources(), FairQueueConfig::default()).unwrap(),
        handle,
    )
}

fn attach(
    registry: &mut LoadRegistry<RunnerMaterializationBackend>,
    waiter: WaiterId,
    key: LoadKey,
    now_ns: u64,
) -> OperationId {
    registry
        .attach_waiter(
            waiter,
            [LoadRequest::new(key, BYTES, ResourceClass::Decode)],
            now_ns,
        )
        .unwrap()
        .created[0]
}

fn manual_at_read() -> (
    LoadRegistry<RunnerMaterializationBackend>,
    MockPhysicalHandle,
    LoadKey,
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
    registry: &mut LoadRegistry<RunnerMaterializationBackend>,
    maximum: usize,
) -> CompletionDisposition {
    assert_eq!(registry.collect_backend_completions(maximum), 1);
    registry.process_one_completion().unwrap()
}

#[test]
fn physical_bridge_reservation_keeps_exact_key() {
    let (mut backend, _, key) = resolved_backend();
    let reservation = backend.reserve(OperationId::new(7), key, BYTES).unwrap();
    assert_eq!(
        reservation.binding().generation,
        key.destination_generation()
    );
}

#[test]
fn physical_bridge_reservation_keeps_exact_binding() {
    let (mut backend, handle, key) = resolved_backend();
    let reservation = backend.reserve(OperationId::new(7), key, BYTES).unwrap();
    assert_eq!(reservation.binding(), handle.lock().binding_for(key));
}

#[test]
fn physical_bridge_reservation_keeps_exact_slab_descriptor() {
    let (mut backend, _, key) = resolved_backend();
    let operation = OperationId::new(7);
    let reservation = backend.reserve(operation, key, BYTES).unwrap();
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
    let (mut backend, _, key) = resolved_backend();
    let operation = OperationId::new(7);
    let reservation = backend.reserve(operation, key, BYTES).unwrap();
    assert_eq!(reservation.upload_fence().operation, operation);
    assert_eq!(
        reservation.upload_fence().fence,
        FenceId::new(7 + FENCE_ID_OFFSET)
    );
}

#[test]
fn physical_bridge_rejects_reservation_for_different_key() {
    let (mut backend, handle, canonical_key) = resolved_backend();
    handle.override_reservation_key(key(2));
    assert!(matches!(
        backend.reserve(OperationId::new(7), canonical_key, BYTES),
        Err(FailureReason::ProtocolViolation(_))
    ));
}

#[test]
fn physical_bridge_rejects_reservation_for_different_operation() {
    let (mut backend, handle, key) = resolved_backend();
    handle.override_reservation_operation(OperationId::new(99));
    assert!(matches!(
        backend.reserve(OperationId::new(7), key, BYTES),
        Err(FailureReason::ProtocolViolation(_))
    ));
}

#[test]
fn physical_bridge_rejects_binding_changed_after_resolve() {
    let (mut backend, handle, key) = resolved_backend();
    handle.override_reservation_slot(DestinationSlotId::new(SLOT + 1));
    assert!(matches!(
        backend.reserve(OperationId::new(7), key, BYTES),
        Err(FailureReason::ProtocolViolation(_))
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
fn physical_bridge_forwards_materialization_bytes() {
    let (physical, handle) = MockPhysicalBackend::manual();
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    assert_eq!(backend.materialization_bytes(key(1)).unwrap(), BYTES);
    assert_eq!(
        handle.command_count(|command| matches!(
            command,
            MockPhysicalCommand::MaterializationBytes(_)
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
fn physical_bridge_adapter_uses_physical_destination_generation() {
    let (physical, handle) = MockPhysicalBackend::manual();
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    let (mut adapter, _) =
        RuntimeMaterializationControl::new(handle.placement(), Some(backend), CompletionHub::new());
    let ExpertDependencyResolution::Waiting(key) = adapter.resolve(request(1)).unwrap() else {
        panic!("expected physical reservation");
    };
    assert_eq!(
        key.destination_generation(),
        DestinationGeneration::new(DESTINATION_GENERATION)
    );
}

#[test]
fn physical_bridge_adapter_caches_canonical_resolution() {
    let (physical, handle) = MockPhysicalBackend::manual();
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    let (mut adapter, _) =
        RuntimeMaterializationControl::new(handle.placement(), Some(backend), CompletionHub::new());
    let first = adapter.resolve(request(1)).unwrap();
    let second = adapter.resolve(request(1)).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Resolve(_))),
        1
    );
}

#[test]
fn physical_bridge_adapter_preserves_resident_binding() {
    let (physical, handle) = MockPhysicalBackend::manual();
    handle.set_resident(true);
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    let (mut adapter, _) =
        RuntimeMaterializationControl::new(handle.placement(), Some(backend), CompletionHub::new());
    let ExpertDependencyResolution::Resident(binding) = adapter.resolve(request(1)).unwrap() else {
        panic!("expected resident physical resolution");
    };
    assert_eq!(binding.binding(), handle.lock().binding_for(binding.key()));
}

#[test]
fn physical_resident_resolution_is_adopted_without_a_second_read() {
    let (physical, handle) = MockPhysicalBackend::manual();
    handle.set_resident(true);
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    let (mut adapter, control) = RuntimeMaterializationControl::new(
        handle.placement(),
        Some(backend.clone()),
        CompletionHub::new(),
    );
    let ExpertDependencyResolution::Resident(resident) = adapter.resolve(request(1)).unwrap()
    else {
        panic!("expected resident physical resolution");
    };
    let key = resident.key();
    let load = LoadRequest::new(key, BYTES, ResourceClass::Decode);
    let mut registry =
        LoadRegistry::new(backend, physical_resources(), FairQueueConfig::default()).unwrap();
    registry.adopt_residency(load, resident.binding()).unwrap();
    let report = registry.attach_waiter(waiter(1, 1), [load], 1).unwrap();
    assert_eq!(report.already_resident, 1);
    assert!(report.created.is_empty());
    assert_eq!(registry.residency_binding(key), Some(handle.binding(key)));
    control.attach(ContinuationId::new(1), key).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::SubmitRead(..))),
        0
    );
    adapter.detach(ContinuationId::new(1), key).unwrap();
    registry.shutdown(2, 0).unwrap();
    control.forget_all().unwrap();
}

#[test]
fn physical_selected_lease_releases_only_after_last_logical_demand_detaches() {
    let (physical, handle) = MockPhysicalBackend::manual();
    handle.set_resident(true);
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    let completion_hub = CompletionHub::new();
    let (mut adapter, control) = RuntimeMaterializationControl::new(
        handle.placement(),
        Some(backend),
        completion_hub.clone(),
    );
    let ExpertDependencyResolution::Resident(binding) = adapter.resolve(request(1)).unwrap() else {
        panic!("expected resident physical resolution");
    };
    let key = binding.key();
    control.attach(ContinuationId::new(1), key).unwrap();
    control.attach(ContinuationId::new(2), key).unwrap();

    let initial_epoch = completion_hub.epoch();
    adapter.detach(ContinuationId::new(1), key).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::ReleaseSelected(_))),
        0
    );
    assert_eq!(completion_hub.epoch(), initial_epoch);
    adapter.detach(ContinuationId::new(2), key).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::ReleaseSelected(_))),
        1
    );
    assert_eq!(completion_hub.epoch(), initial_epoch + 1);
}

#[test]
fn physical_selected_lease_supports_same_continuation_key_on_a_second_edge() {
    let (physical, handle) = MockPhysicalBackend::manual();
    handle.set_resident(true);
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    let (mut adapter, control) =
        RuntimeMaterializationControl::new(handle.placement(), Some(backend), CompletionHub::new());
    let ExpertDependencyResolution::Resident(first) = adapter.resolve(request(1)).unwrap() else {
        panic!("expected resident physical resolution");
    };
    let continuation = ContinuationId::new(7);
    control.attach(continuation, first.key()).unwrap();
    adapter.detach(continuation, first.key()).unwrap();

    let ExpertDependencyResolution::Resident(second) = adapter.resolve(request(1)).unwrap() else {
        panic!("expected resident physical resolution on the second edge");
    };
    assert_eq!(second.key(), first.key());
    control.attach(continuation, second.key()).unwrap();
    adapter.detach(continuation, second.key()).unwrap();

    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::ReleaseSelected(_))),
        2
    );
    assert_eq!(control.active_attachment_count(first.key()), 0);
}

#[test]
fn zero_demand_publish_releases_selected_and_retries_release_failure() {
    let (physical, handle) = MockPhysicalBackend::manual();
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    let (mut adapter, control) =
        RuntimeMaterializationControl::new(handle.placement(), Some(backend), CompletionHub::new());
    let ExpertDependencyResolution::Waiting(key) = adapter.resolve(request(1)).unwrap() else {
        panic!("expected pending physical reservation");
    };
    handle.fail_next_release(FailureReason::DeviceUnavailable);
    assert!(control.record_resident(key, handle.binding(key)).is_err());
    assert_eq!(control.resident_binding(key), Some(handle.binding(key)));
    control.record_resident(key, handle.binding(key)).unwrap();
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::ReleaseSelected(_))),
        2
    );
}

#[test]
fn forget_release_failure_restores_exact_attachment_for_retry() {
    let (physical, handle) = MockPhysicalBackend::manual();
    handle.set_resident(true);
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    let (mut adapter, control) =
        RuntimeMaterializationControl::new(handle.placement(), Some(backend), CompletionHub::new());
    let ExpertDependencyResolution::Resident(binding) = adapter.resolve(request(1)).unwrap() else {
        panic!("expected resident physical resolution");
    };
    let continuation = ContinuationId::new(9);
    control.attach(continuation, binding.key()).unwrap();
    handle.fail_next_release(FailureReason::DeviceUnavailable);
    assert!(control.forget(binding.key()).is_err());
    assert_eq!(control.active_attachment_count(binding.key()), 1);
    adapter.detach(continuation, binding.key()).unwrap();
    assert_eq!(control.active_attachment_count(binding.key()), 0);
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::ReleaseSelected(_))),
        2
    );
}

#[test]
fn terminal_idle_key_is_forgotten_before_request_retry() {
    let (physical, handle) = MockPhysicalBackend::manual();
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    let (mut adapter, control) =
        RuntimeMaterializationControl::new(handle.placement(), Some(backend), CompletionHub::new());
    let ExpertDependencyResolution::Waiting(key) = adapter.resolve(request(1)).unwrap() else {
        panic!("expected pending physical reservation");
    };
    let continuation = ContinuationId::new(11);
    control.attach(continuation, key).unwrap();
    adapter.detach(continuation, key).unwrap();
    assert!(control.forget_if_idle(key).unwrap());
    assert!(matches!(
        adapter.resolve(request(1)).unwrap(),
        ExpertDependencyResolution::Waiting(_)
    ));
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Resolve(_))),
        2
    );
}

#[test]
fn physical_bridge_adapter_surfaces_resolve_failure() {
    let (physical, handle) = MockPhysicalBackend::manual();
    handle.fail_next_resolve(FailureReason::StorageUnavailable);
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    let (mut adapter, _) =
        RuntimeMaterializationControl::new(handle.placement(), Some(backend), CompletionHub::new());
    assert!(adapter.resolve(request(1)).is_err());
}

#[test]
fn physical_bridge_registry_surfaces_reserve_failure_without_credit_leak() {
    let (mut registry, handle) = registry(false);
    handle.fail_next_reserve(FailureReason::DeviceUnavailable);
    let report = registry
        .attach_waiter(
            waiter(1, 1),
            [LoadRequest::new(key(1), BYTES, ResourceClass::Decode)],
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
fn physical_bridge_single_flight_issues_one_physical_read() {
    let (mut registry, handle) = registry(true);
    let key = key(1);
    let operation = attach(&mut registry, waiter(1, 1), key, 1);
    let joined = registry
        .attach_waiter(
            waiter(2, 2),
            [LoadRequest::new(key, BYTES, ResourceClass::Decode)],
            2,
        )
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
    registry
        .attach_waiter(
            waiter(2, 2),
            [LoadRequest::new(key, BYTES, ResourceClass::Decode)],
            2,
        )
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
    registry
        .attach_waiter(
            waiter(2, 2),
            [LoadRequest::new(key, BYTES, ResourceClass::Decode)],
            2,
        )
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
fn physical_bridge_registry_and_adapter_retain_identical_binding() {
    let (physical, handle) = MockPhysicalBackend::automatic();
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    let (mut adapter, control) = RuntimeMaterializationControl::new(
        handle.placement(),
        Some(backend.clone()),
        CompletionHub::new(),
    );
    let ExpertDependencyResolution::Waiting(key) = adapter.resolve(request(1)).unwrap() else {
        panic!("expected physical reservation");
    };
    let mut registry =
        LoadRegistry::new(backend, physical_resources(), FairQueueConfig::default()).unwrap();
    attach(&mut registry, waiter(1, 1), key, 1);
    registry.drive(100, 32).unwrap();
    let binding = registry.residency_binding(key).unwrap();
    assert_eq!(binding, handle.binding(key));
    control.record_resident(key, binding).unwrap();
    assert_eq!(control.resident_binding(key), Some(binding));
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

    let (physical, handle) = MockPhysicalBackend::manual();
    let backend = RunnerMaterializationBackend::new(Box::new(physical));
    for seed in 1..=DEPENDENCIES {
        backend.resolve_or_reserve(request(seed)).unwrap();
    }
    let mut resources = bounded_physical_resources(u64::from(DEPENDENCIES), QD, QD, QD);
    resources
        .reconfigure_limit(ResourceKind::LoadOperation, QD, 0)
        .unwrap();
    let mut registry = LoadRegistry::new(backend, resources, FairQueueConfig::default()).unwrap();
    let requests =
        (1..=DEPENDENCIES).map(|seed| LoadRequest::new(key(seed), BYTES, ResourceClass::Decode));

    let report = registry.attach_waiter(waiter(1, 1), requests, 1).unwrap();
    assert_eq!(report.created.len(), usize::from(DEPENDENCIES));
    assert_eq!(
        handle.command_count(|command| matches!(command, MockPhysicalCommand::Reserve(..))),
        0
    );
    assert_eq!(registry.resources().in_use(ResourceKind::Sqe), 0);
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
        .find(|snapshot| snapshot.kind == ResourceKind::Sqe)
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
    assert_eq!(high_water(ResourceKind::PinnedSlab), BYTES);
    assert_eq!(high_water(ResourceKind::ReadBytes), BYTES);
    assert_eq!(high_water(ResourceKind::UploadBytes), BYTES);
    assert_eq!(high_water(ResourceKind::ExpertFrame), BYTES);
    assert_eq!(high_water(ResourceKind::Sqe), 1);
    assert_eq!(high_water(ResourceKind::LoadOperation), 1);
}
