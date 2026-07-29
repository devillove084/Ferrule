//! Materialization-provider boundary and deterministic fake implementation.

use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

use ferrule_common::io_protocol::{
    CancellationReason, CompletionEvent, CompletionTimestamp, FailureReason, LoadKey, LoadStage,
    OperationId, RegisteredPinnedAlignedSlabLease, ResidencyBinding, UploadFenceContract,
};
#[cfg(test)]
use ferrule_common::io_protocol::{
    CompletionGeneration, CompletionOutcome, DestinationSlotId, FenceId,
    RegisteredPinnedAlignedSlabLeaseDescriptor, RegistrationId, SlabId,
};
use ferrule_model::{
    ExpertMaterializationPlacement, ExpertMaterializationRequest,
    PhysicalExpertMaterializationBackend, PhysicalExpertOperationReservation,
    PhysicalExpertReservation, PhysicalExpertResourceTopology,
};

/// Owner-held physical reservation. Providers receive only immutable descriptors;
/// the runtime retains and advances the owning pinned lease.
#[derive(Debug)]
pub struct MaterializationReservation {
    pub(crate) slabs: Box<[RegisteredPinnedAlignedSlabLease]>,
    binding: ResidencyBinding,
    upload_fence: UploadFenceContract,
    physical: Option<PhysicalExpertOperationReservation>,
}

impl MaterializationReservation {
    pub fn slabs(&self) -> &[RegisteredPinnedAlignedSlabLease] {
        &self.slabs
    }

    pub const fn binding(&self) -> ResidencyBinding {
        self.binding
    }

    pub const fn upload_fence(&self) -> UploadFenceContract {
        self.upload_fence
    }

    fn physical(&self) -> Result<&PhysicalExpertOperationReservation, FailureReason> {
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
pub trait MaterializationBackend: std::fmt::Debug + Send {
    /// Exact physical payload bytes charged to hard read/upload credits.
    fn materialization_bytes(&self, key: LoadKey) -> Result<u64, FailureReason>;

    fn reserve(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        bytes: u64,
    ) -> Result<MaterializationReservation, FailureReason>;

    fn submit_read(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &MaterializationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason>;

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &MaterializationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason>;

    fn poll_install(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &MaterializationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason>;

    fn cancel(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        stage: LoadStage,
        reason: CancellationReason,
    ) -> Result<(), FailureReason>;

    fn next_completion(&mut self) -> Option<CompletionEvent>;
}

/// Shared runtime handle around the physical backend transferred once from a
/// model runner. Clones address the same model-owned slot/frame authority; they
/// do not duplicate CUDA streams, pinned operations, tickets, or publication.
pub struct RunnerMaterializationBackend {
    state: Arc<Mutex<RunnerMaterializationState>>,
}

struct RunnerMaterializationState {
    physical: Box<dyn PhysicalExpertMaterializationBackend>,
    canonical_bindings: BTreeMap<LoadKey, ResidencyBinding>,
}

impl std::fmt::Debug for RunnerMaterializationBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RunnerMaterializationBackend")
            .field("placement", &self.placement())
            .finish_non_exhaustive()
    }
}

impl Clone for RunnerMaterializationBackend {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl RunnerMaterializationBackend {
    pub fn new(physical: Box<dyn PhysicalExpertMaterializationBackend>) -> Self {
        Self {
            state: Arc::new(Mutex::new(RunnerMaterializationState {
                physical,
                canonical_bindings: BTreeMap::new(),
            })),
        }
    }

    pub fn placement(&self) -> ExpertMaterializationPlacement {
        self.lock().physical.placement()
    }

    pub fn resource_topology(&self) -> ferrule_common::Result<PhysicalExpertResourceTopology> {
        self.lock().physical.resource_topology()
    }

    pub fn resolve_or_reserve(
        &self,
        request: ExpertMaterializationRequest,
    ) -> Result<PhysicalExpertReservation, FailureReason> {
        let mut state = self.lock();
        let resolution = state.physical.resolve_or_reserve(request)?;
        match &resolution {
            PhysicalExpertReservation::Resident(binding) => {
                state
                    .canonical_bindings
                    .insert(binding.key(), binding.binding());
            }
            PhysicalExpertReservation::Reserved(reservation) => {
                state
                    .canonical_bindings
                    .insert(reservation.key(), reservation.binding());
            }
        }
        Ok(resolution)
    }

    pub fn release_selected(&self, key: LoadKey) -> Result<(), FailureReason> {
        self.lock().physical.release_selected(key)
    }

    fn lock(&self) -> MutexGuard<'_, RunnerMaterializationState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl MaterializationBackend for RunnerMaterializationBackend {
    fn materialization_bytes(&self, key: LoadKey) -> Result<u64, FailureReason> {
        self.lock().physical.materialization_bytes(key)
    }

    fn reserve(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        bytes: u64,
    ) -> Result<MaterializationReservation, FailureReason> {
        let mut state = self.lock();
        let expected_binding = state.canonical_bindings.get(&key).copied().ok_or_else(|| {
            FailureReason::ProtocolViolation(
                "physical operation reserve was not preceded by canonical resolve/reserve".into(),
            )
        })?;
        let physical = state.physical.reserve(operation, key, bytes)?;
        let binding = physical.binding();
        let upload_fence = physical.upload_fence();
        let violation = if physical.key() != key {
            Some("physical backend reserved a different load key")
        } else if binding != expected_binding {
            Some("physical backend reserved a different residency binding")
        } else if upload_fence.operation != operation {
            Some("physical backend returned an upload fence for a different operation")
        } else if physical
            .slabs()
            .iter()
            .any(|slab| slab.operation() != operation)
        {
            Some("physical backend returned a slab for a different operation")
        } else {
            None
        };
        if let Some(violation) = violation {
            let cleanup = state.physical.cancel(
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
        Ok(MaterializationReservation {
            slabs,
            binding,
            upload_fence,
            physical: Some(physical),
        })
    }

    fn submit_read(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &MaterializationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason> {
        self.lock()
            .physical
            .submit_read(operation, key, reservation.physical()?, bytes)
    }

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &MaterializationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason> {
        self.lock()
            .physical
            .submit_upload(operation, key, reservation.physical()?, bytes)
    }

    fn poll_install(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &MaterializationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason> {
        self.lock()
            .physical
            .poll_install(operation, key, reservation.physical()?, bytes)
    }

    fn cancel(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        stage: LoadStage,
        reason: CancellationReason,
    ) -> Result<(), FailureReason> {
        self.lock().physical.cancel(operation, key, stage, reason)
    }

    fn next_completion(&mut self) -> Option<CompletionEvent> {
        self.lock().physical.next_completion()
    }
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FakeCommand {
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
    pub key: Option<LoadKey>,
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

/// Deterministic no-thread/no-sleep backend. Commands append scripted completion
/// events to a FIFO; tests may also inject arbitrary events directly.
#[cfg(test)]
#[derive(Debug)]
pub struct FakeBackend {
    automatic: bool,
    clock_ns: u64,
    commands: Vec<FakeCommand>,
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
impl Default for FakeBackend {
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
impl FakeBackend {
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

    pub fn commands(&self) -> &[FakeCommand] {
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

    fn emit(&mut self, operation: OperationId, key: LoadKey, stage: LoadStage, bytes: u64) {
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
impl MaterializationBackend for FakeBackend {
    fn materialization_bytes(&self, _key: LoadKey) -> Result<u64, FailureReason> {
        Ok(4096)
    }

    fn reserve(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        bytes: u64,
    ) -> Result<MaterializationReservation, FailureReason> {
        self.commands.push(FakeCommand::Reserve(operation));
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
        Ok(MaterializationReservation {
            slabs: vec![RegisteredPinnedAlignedSlabLease::new(descriptor)].into_boxed_slice(),
            binding: ResidencyBinding::new(
                key.model(),
                key.layer(),
                key.expert(),
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
        key: LoadKey,
        _reservation: &MaterializationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason> {
        self.command_result(LoadStage::ReadSubmitted)?;
        self.commands.push(FakeCommand::SubmitRead(operation));
        self.reads += 1;
        self.emit(operation, key, LoadStage::ReadSubmitted, bytes);
        Ok(())
    }

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        _reservation: &MaterializationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason> {
        self.command_result(LoadStage::UploadSubmitted)?;
        self.commands.push(FakeCommand::SubmitUpload(operation));
        self.uploads += 1;
        self.emit(operation, key, LoadStage::UploadSubmitted, bytes);
        Ok(())
    }

    fn poll_install(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        _reservation: &MaterializationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason> {
        self.command_result(LoadStage::Installing)?;
        self.commands.push(FakeCommand::PollInstall(operation));
        self.installs += 1;
        self.emit(operation, key, LoadStage::Installing, bytes);
        Ok(())
    }

    fn cancel(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        stage: LoadStage,
        reason: CancellationReason,
    ) -> Result<(), FailureReason> {
        self.command_result(stage)?;
        self.commands.push(FakeCommand::Cancel(operation, stage));
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

impl<T> MaterializationBackend for Box<T>
where
    T: MaterializationBackend + ?Sized,
{
    fn materialization_bytes(&self, key: LoadKey) -> Result<u64, FailureReason> {
        (**self).materialization_bytes(key)
    }

    fn reserve(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        bytes: u64,
    ) -> Result<MaterializationReservation, FailureReason> {
        (**self).reserve(operation, key, bytes)
    }

    fn submit_read(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &MaterializationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason> {
        (**self).submit_read(operation, key, reservation, bytes)
    }

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &MaterializationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason> {
        (**self).submit_upload(operation, key, reservation, bytes)
    }

    fn poll_install(
        &mut self,
        operation: OperationId,
        key: LoadKey,
        reservation: &MaterializationReservation,
        bytes: u64,
    ) -> Result<(), FailureReason> {
        (**self).poll_install(operation, key, reservation, bytes)
    }

    fn cancel(
        &mut self,
        operation: OperationId,
        key: LoadKey,
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
pub struct UnavailableBackend;

impl MaterializationBackend for UnavailableBackend {
    fn materialization_bytes(&self, _key: LoadKey) -> Result<u64, FailureReason> {
        Err(FailureReason::DeviceUnavailable)
    }

    fn reserve(
        &mut self,
        _operation: OperationId,
        _key: LoadKey,
        _bytes: u64,
    ) -> Result<MaterializationReservation, FailureReason> {
        Err(FailureReason::DeviceUnavailable)
    }

    fn submit_read(
        &mut self,
        _operation: OperationId,
        _key: LoadKey,
        _reservation: &MaterializationReservation,
        _bytes: u64,
    ) -> Result<(), FailureReason> {
        Err(FailureReason::DeviceUnavailable)
    }

    fn submit_upload(
        &mut self,
        _operation: OperationId,
        _key: LoadKey,
        _reservation: &MaterializationReservation,
        _bytes: u64,
    ) -> Result<(), FailureReason> {
        Err(FailureReason::DeviceUnavailable)
    }

    fn poll_install(
        &mut self,
        _operation: OperationId,
        _key: LoadKey,
        _reservation: &MaterializationReservation,
        _bytes: u64,
    ) -> Result<(), FailureReason> {
        Err(FailureReason::DeviceUnavailable)
    }

    fn cancel(
        &mut self,
        _operation: OperationId,
        _key: LoadKey,
        _stage: LoadStage,
        _reason: CancellationReason,
    ) -> Result<(), FailureReason> {
        Ok(())
    }

    fn next_completion(&mut self) -> Option<CompletionEvent> {
        None
    }
}
