//! Materialization-provider boundary and deterministic fake implementation.

#[cfg(test)]
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use ferrule_common::io_protocol::{
    CancellationReason, CompletionEvent, CompletionTimestamp, FailureReason, LoadStage,
    MaterializationKey, OperationId, RegisteredPinnedAlignedSlabLease, ResidencyBinding,
    UploadFenceContract,
};
#[cfg(test)]
use ferrule_common::io_protocol::{
    CompletionGeneration, CompletionOutcome, DestinationSlotId, FenceId,
    RegisteredPinnedAlignedSlabLeaseDescriptor, RegistrationId, SlabId,
};
use ferrule_common::materialization_io::MaterializationResourcePlan;
use ferrule_model::{
    MaterializationPlacement, MaterializationPreparation, MaterializationProvider,
    MaterializationRequest, PhysicalMaterializationOperationReservation,
    PhysicalMaterializationTopology,
};

/// Owner-held physical reservation. Providers receive only immutable descriptors;
/// the runtime retains and advances the owning pinned lease.
#[derive(Debug)]
pub struct MaterializationOperationReservation {
    pub(crate) slabs: Box<[RegisteredPinnedAlignedSlabLease]>,
    binding: ResidencyBinding,
    upload_fence: UploadFenceContract,
    physical: Option<PhysicalMaterializationOperationReservation>,
}

impl MaterializationOperationReservation {
    pub fn slabs(&self) -> &[RegisteredPinnedAlignedSlabLease] {
        &self.slabs
    }

    pub const fn binding(&self) -> ResidencyBinding {
        self.binding
    }

    pub const fn upload_fence(&self) -> UploadFenceContract {
        self.upload_fence
    }

    fn physical(&self) -> Result<&PhysicalMaterializationOperationReservation, FailureReason> {
        self.physical.as_ref().ok_or_else(|| {
            FailureReason::ProtocolViolation(
                "runner materialization command received a non-physical reservation".into(),
            )
        })
    }

    pub(crate) fn mark_read_submitted(&mut self) -> Result<(), ferrule_common::IoProtocolError> {
        for slab in &mut self.slabs {
            slab.mark_read_submitted()?;
        }
        Ok(())
    }

    pub(crate) fn mark_host_ready(&mut self) -> Result<(), ferrule_common::IoProtocolError> {
        for slab in &mut self.slabs {
            slab.mark_host_ready(slab.descriptor().len())?;
        }
        Ok(())
    }

    pub(crate) fn mark_read_returned_without_artifact(
        &mut self,
    ) -> Result<(), ferrule_common::IoProtocolError> {
        for slab in &mut self.slabs {
            slab.mark_read_returned_without_artifact()?;
        }
        Ok(())
    }

    pub(crate) fn mark_upload_submitted(&mut self) -> Result<(), ferrule_common::IoProtocolError> {
        for slab in &mut self.slabs {
            slab.mark_upload_submitted(self.upload_fence)?;
        }
        Ok(())
    }

    pub(crate) fn mark_upload_fence(
        &mut self,
        timestamp: CompletionTimestamp,
    ) -> Result<(), ferrule_common::IoProtocolError> {
        for slab in &mut self.slabs {
            slab.mark_upload_fence(self.upload_fence.observation(timestamp))?;
        }
        Ok(())
    }

    pub(crate) fn retire_slabs(&mut self) -> Result<(), ferrule_common::IoProtocolError> {
        for slab in &mut self.slabs {
            slab.retire()?;
        }
        Ok(())
    }
}

/// Runtime/provider command interface. Command acceptance never changes owner
/// state by itself; physical progress is observed only through `CompletionEvent`.
pub trait RuntimeMaterializationProvider: std::fmt::Debug + Send {
    /// Exact provider-owned preparation fixed before registry admission.
    fn preparation(
        &self,
        key: MaterializationKey,
    ) -> Result<MaterializationPreparation, FailureReason>;

    /// Roll back a preparation that never became registry-owned work.
    fn discard_preparation(&mut self, key: MaterializationKey) -> Result<(), FailureReason>;

    /// Release provider execution custody after the last registry logical owner.
    fn release_execution_lease(&mut self, key: MaterializationKey) -> Result<(), FailureReason>;

    /// Exact transient stage demand and persistent residency bytes fixed before
    /// runtime hard admission.
    fn materialization_plan(
        &self,
        key: MaterializationKey,
    ) -> Result<MaterializationResourcePlan, FailureReason>;

    fn reserve(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        plan: MaterializationResourcePlan,
    ) -> Result<MaterializationOperationReservation, FailureReason>;

    fn submit_read(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason>;

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason>;

    fn poll_install(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason>;

    fn cancel(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        stage: LoadStage,
        reason: CancellationReason,
    ) -> Result<(), FailureReason>;

    fn next_completion(&mut self) -> Option<CompletionEvent>;
}

/// Shared runtime handle around the physical provider transferred once from a
/// model runner. Clones address the same model-owned residency authority; they
/// do not duplicate provider streams, pinned operations, tickets, or publication.
pub struct SharedMaterializationProvider {
    state: Arc<Mutex<SharedMaterializationProviderState>>,
}

struct SharedMaterializationProviderState {
    provider: Box<dyn MaterializationProvider>,
}

impl std::fmt::Debug for SharedMaterializationProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SharedMaterializationProvider")
            .field("placement", &self.placement())
            .finish_non_exhaustive()
    }
}

impl Clone for SharedMaterializationProvider {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl SharedMaterializationProvider {
    pub fn new(provider: Box<dyn MaterializationProvider>) -> Self {
        Self {
            state: Arc::new(Mutex::new(SharedMaterializationProviderState { provider })),
        }
    }

    pub fn placement(&self) -> MaterializationPlacement {
        self.lock().provider.placement()
    }

    pub fn resource_topology(&self) -> ferrule_common::Result<PhysicalMaterializationTopology> {
        self.lock().provider.resource_topology()
    }

    pub fn prepare(
        &self,
        request: MaterializationRequest,
    ) -> Result<MaterializationPreparation, FailureReason> {
        let preparation = self.lock().provider.prepare(request)?;
        request
            .validate_key(preparation.key())
            .map_err(|error| FailureReason::ProtocolViolation(error.to_string()))?;
        Ok(preparation)
    }

    pub fn prepared(
        &self,
        key: MaterializationKey,
    ) -> Result<MaterializationPreparation, FailureReason> {
        self.lock().provider.prepared(key)
    }

    pub fn discard_preparation(&self, key: MaterializationKey) -> Result<(), FailureReason> {
        self.lock().provider.discard_preparation(key)
    }

    fn lock(&self) -> MutexGuard<'_, SharedMaterializationProviderState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl RuntimeMaterializationProvider for SharedMaterializationProvider {
    fn preparation(
        &self,
        key: MaterializationKey,
    ) -> Result<MaterializationPreparation, FailureReason> {
        self.prepared(key)
    }

    fn discard_preparation(&mut self, key: MaterializationKey) -> Result<(), FailureReason> {
        SharedMaterializationProvider::discard_preparation(self, key)
    }

    fn release_execution_lease(&mut self, key: MaterializationKey) -> Result<(), FailureReason> {
        self.lock().provider.release_execution_lease(key)
    }

    fn materialization_plan(
        &self,
        key: MaterializationKey,
    ) -> Result<MaterializationResourcePlan, FailureReason> {
        self.lock().provider.materialization_plan(key)
    }

    fn reserve(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        plan: MaterializationResourcePlan,
    ) -> Result<MaterializationOperationReservation, FailureReason> {
        let mut state = self.lock();
        let expected = state.provider.prepared(key)?;
        let expected_binding = expected.binding();
        if !matches!(expected, MaterializationPreparation::Transfer(_)) {
            return Err(FailureReason::ProtocolViolation(
                "physical operation reserve requires a prepared transfer".into(),
            ));
        }
        let physical = state.provider.reserve(operation, key, plan)?;
        let binding = physical.binding();
        let upload_fence = physical.upload_fence();
        let violation = if physical.key() != key {
            Some("physical provider reserved a different load key")
        } else if binding != expected_binding {
            Some("physical provider reserved a different residency binding")
        } else if upload_fence.operation != operation {
            Some("physical provider returned an upload fence for a different operation")
        } else if physical
            .slabs()
            .iter()
            .any(|slab| slab.operation() != operation)
        {
            Some("physical provider returned a slab for a different operation")
        } else {
            None
        };
        if let Some(violation) = violation {
            let cleanup = state.provider.cancel(
                operation,
                key,
                LoadStage::Reserved,
                CancellationReason::Superseded,
            );
            let message = match cleanup {
                Ok(()) => violation.to_owned(),
                Err(cleanup) => format!("{violation}; physical cleanup failed: {cleanup:?}"),
            };
            return Err(FailureReason::ProtocolViolation(message));
        }
        let slabs = physical
            .slabs()
            .iter()
            .copied()
            .map(RegisteredPinnedAlignedSlabLease::new)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(MaterializationOperationReservation {
            slabs,
            binding,
            upload_fence,
            physical: Some(physical),
        })
    }

    fn submit_read(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        self.lock()
            .provider
            .submit_read(operation, key, reservation.physical()?, plan)
    }

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        self.lock()
            .provider
            .submit_upload(operation, key, reservation.physical()?, plan)
    }

    fn poll_install(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        self.lock()
            .provider
            .poll_install(operation, key, reservation.physical()?, plan)
    }

    fn cancel(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        stage: LoadStage,
        reason: CancellationReason,
    ) -> Result<(), FailureReason> {
        self.lock().provider.cancel(operation, key, stage, reason)
    }

    fn next_completion(&mut self) -> Option<CompletionEvent> {
        self.lock().provider.next_completion()
    }
}

#[cfg(test)]
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
#[cfg(test)]
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

#[cfg(test)]
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
#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

    fn command_result(&mut self, stage: LoadStage) -> Result<(), FailureReason> {
        match self.rejected.get_mut(&stage).and_then(VecDeque::pop_front) {
            Some(reason) => Err(reason),
            None => Ok(()),
        }
    }

    fn emit(
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

#[cfg(test)]
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
            DestinationSlotId::new(1),
            key.destination_generation(),
        );
        ferrule_model::MaterializationTransfer::new(key, binding, None)
            .map(MaterializationPreparation::Transfer)
            .map_err(|error| FailureReason::ProtocolViolation(error.to_string()))
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
            ferrule_common::materialization_io::MaterializationResourceDemand {
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
        .map_err(|error| FailureReason::ProtocolViolation(error.to_string()))
    }

    fn reserve(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        plan: MaterializationResourcePlan,
    ) -> Result<MaterializationOperationReservation, FailureReason> {
        self.commands
            .push(FakeMaterializationCommand::Reserve(operation));
        let bytes = plan.demand.pinned_host_bytes;
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
        .map_err(|error| FailureReason::ProtocolViolation(error.to_string()))?;
        let slot = DestinationSlotId::new(identity.min(u32::MAX as u64) as u32);
        Ok(MaterializationOperationReservation {
            slabs: vec![RegisteredPinnedAlignedSlabLease::new(descriptor)].into_boxed_slice(),
            binding: ResidencyBinding::new(
                key.model(),
                key.resource(),
                key.backend(),
                key.device(),
                slot,
                key.destination_generation(),
            ),
            upload_fence: UploadFenceContract::new(
                operation,
                FenceId::new(identity),
                key.destination_generation(),
            ),
            physical: None,
        })
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
            plan.demand.storage_read_bytes,
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
            plan.demand.h2d_bytes,
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
            plan.demand.device_install_bytes,
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

impl<T> RuntimeMaterializationProvider for Box<T>
where
    T: RuntimeMaterializationProvider + ?Sized,
{
    fn preparation(
        &self,
        key: MaterializationKey,
    ) -> Result<MaterializationPreparation, FailureReason> {
        (**self).preparation(key)
    }

    fn discard_preparation(&mut self, key: MaterializationKey) -> Result<(), FailureReason> {
        (**self).discard_preparation(key)
    }

    fn release_execution_lease(&mut self, key: MaterializationKey) -> Result<(), FailureReason> {
        (**self).release_execution_lease(key)
    }

    fn materialization_plan(
        &self,
        key: MaterializationKey,
    ) -> Result<MaterializationResourcePlan, FailureReason> {
        (**self).materialization_plan(key)
    }

    fn reserve(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        plan: MaterializationResourcePlan,
    ) -> Result<MaterializationOperationReservation, FailureReason> {
        (**self).reserve(operation, key, plan)
    }

    fn submit_read(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        (**self).submit_read(operation, key, reservation, plan)
    }

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        (**self).submit_upload(operation, key, reservation, plan)
    }

    fn poll_install(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &MaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        (**self).poll_install(operation, key, reservation, plan)
    }

    fn cancel(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        stage: LoadStage,
        reason: CancellationReason,
    ) -> Result<(), FailureReason> {
        (**self).cancel(operation, key, stage, reason)
    }

    fn next_completion(&mut self) -> Option<CompletionEvent> {
        (**self).next_completion()
    }
}

/// Fail-closed placeholder used until a real physical provider is installed.
#[derive(Debug, Default)]
pub struct UnavailableMaterializationProvider;

impl RuntimeMaterializationProvider for UnavailableMaterializationProvider {
    fn preparation(
        &self,
        _key: MaterializationKey,
    ) -> Result<MaterializationPreparation, FailureReason> {
        Err(FailureReason::DeviceUnavailable)
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
        Err(FailureReason::DeviceUnavailable)
    }

    fn reserve(
        &mut self,
        _operation: OperationId,
        _key: MaterializationKey,
        _plan: MaterializationResourcePlan,
    ) -> Result<MaterializationOperationReservation, FailureReason> {
        Err(FailureReason::DeviceUnavailable)
    }

    fn submit_read(
        &mut self,
        _operation: OperationId,
        _key: MaterializationKey,
        _reservation: &MaterializationOperationReservation,
        _plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        Err(FailureReason::DeviceUnavailable)
    }

    fn submit_upload(
        &mut self,
        _operation: OperationId,
        _key: MaterializationKey,
        _reservation: &MaterializationOperationReservation,
        _plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        Err(FailureReason::DeviceUnavailable)
    }

    fn poll_install(
        &mut self,
        _operation: OperationId,
        _key: MaterializationKey,
        _reservation: &MaterializationOperationReservation,
        _plan: MaterializationResourcePlan,
    ) -> Result<(), FailureReason> {
        Err(FailureReason::DeviceUnavailable)
    }

    fn cancel(
        &mut self,
        _operation: OperationId,
        _key: MaterializationKey,
        _stage: LoadStage,
        _reason: CancellationReason,
    ) -> Result<(), FailureReason> {
        Ok(())
    }

    fn next_completion(&mut self) -> Option<CompletionEvent> {
        None
    }
}
