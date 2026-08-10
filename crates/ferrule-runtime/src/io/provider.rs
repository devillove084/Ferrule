//! Materialization-provider boundary and deterministic fake implementation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use ferrule_common::io_protocol::{
    CancellationReason, CompletionEvent, CompletionTimestamp, FailureReason, LoadStage,
    MaterializationKey, OperationId, RegisteredPinnedAlignedSlabLease, ResidencyBinding,
    UploadFenceContract,
};
use ferrule_common::materialization_io::MaterializationResourcePlan;
use ferrule_model::{
    MaterializationPlacement, MaterializationPreparation, MaterializationProvider,
    MaterializationPurpose, MaterializationRequest, PhysicalMaterializationOperationReservation,
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
    /// Creates one operation reservation. Provider implementations and test
    /// doubles construct reservations through this boundary.
    pub fn new(
        slabs: Box<[RegisteredPinnedAlignedSlabLease]>,
        binding: ResidencyBinding,
        upload_fence: UploadFenceContract,
        physical: Option<PhysicalMaterializationOperationReservation>,
    ) -> Self {
        Self {
            slabs,
            binding,
            upload_fence,
            physical,
        }
    }

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
        self.physical
            .as_ref()
            .ok_or_else(|| FailureReason::ContractViolation {
                message: "runner materialization command received a non-physical reservation"
                    .into(),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPromotion {
    AlreadyExecution(MaterializationPreparation),
    Promoted(MaterializationPreparation),
}

impl ExecutionPromotion {
    pub const fn preparation(self) -> MaterializationPreparation {
        match self {
            Self::AlreadyExecution(preparation) | Self::Promoted(preparation) => preparation,
        }
    }

    pub const fn changed(self) -> bool {
        matches!(self, Self::Promoted(_))
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

    /// Explicitly promote a frozen prefetch to execution custody without changing
    /// its key, binding, or physical operation.
    fn promote_to_execution(
        &mut self,
        key: MaterializationKey,
    ) -> Result<ExecutionPromotion, FailureReason>;

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

#[derive(Debug, Clone, Copy)]
struct FrozenPreparation {
    preparation: MaterializationPreparation,
    purpose: MaterializationPurpose,
}

struct SharedMaterializationProviderState {
    provider: Box<dyn MaterializationProvider>,
    preparations: HashMap<MaterializationKey, FrozenPreparation>,
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
            state: Arc::new(Mutex::new(SharedMaterializationProviderState {
                provider,
                preparations: HashMap::new(),
            })),
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
        intent: MaterializationPurpose,
    ) -> Result<MaterializationPreparation, FailureReason> {
        let mut state = self.lock();
        let preparation = state.provider.prepare(request, intent)?;
        request
            .validate_key(preparation.key())
            .map_err(|source| FailureReason::Protocol { source })?;
        state
            .preparations
            .entry(preparation.key())
            .and_modify(|frozen| frozen.preparation = preparation)
            .or_insert(FrozenPreparation {
                preparation,
                purpose: intent,
            });
        Ok(preparation)
    }

    pub fn prepared(
        &self,
        key: MaterializationKey,
    ) -> Result<MaterializationPreparation, FailureReason> {
        let mut state = self.lock();
        let preparation = state.provider.prepared(key)?;
        if preparation.key() != key {
            return Err(FailureReason::ContractViolation {
                message: "physical provider returned a different prepared key".into(),
            });
        }
        let frozen =
            state
                .preparations
                .get_mut(&key)
                .ok_or_else(|| FailureReason::ContractViolation {
                    message: "provider observation has no frozen preparation purpose".into(),
                })?;
        frozen.preparation = preparation;
        Ok(preparation)
    }

    pub fn promote_to_execution(
        &self,
        key: MaterializationKey,
    ) -> Result<ExecutionPromotion, FailureReason> {
        let mut state = self.lock();
        let expected = state.preparations.get(&key).copied().ok_or_else(|| {
            FailureReason::ContractViolation {
                message: "execution promotion has no frozen provider preparation".into(),
            }
        })?;
        if expected.purpose == MaterializationPurpose::Execution {
            return Ok(ExecutionPromotion::AlreadyExecution(expected.preparation));
        }
        let preparation = state.provider.promote_to_execution(key)?;
        if preparation.key() != key || preparation.binding() != expected.preparation.binding() {
            return Err(FailureReason::ContractViolation {
                message: "execution promotion changed the frozen key or binding".into(),
            });
        }
        state.preparations.insert(
            key,
            FrozenPreparation {
                preparation,
                purpose: MaterializationPurpose::Execution,
            },
        );
        Ok(ExecutionPromotion::Promoted(preparation))
    }

    pub fn discard_preparation(&self, key: MaterializationKey) -> Result<(), FailureReason> {
        let mut state = self.lock();
        state.provider.discard_preparation(key)?;
        state.preparations.remove(&key);
        Ok(())
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

    fn promote_to_execution(
        &mut self,
        key: MaterializationKey,
    ) -> Result<ExecutionPromotion, FailureReason> {
        SharedMaterializationProvider::promote_to_execution(self, key)
    }

    fn discard_preparation(&mut self, key: MaterializationKey) -> Result<(), FailureReason> {
        SharedMaterializationProvider::discard_preparation(self, key)
    }

    fn release_execution_lease(&mut self, key: MaterializationKey) -> Result<(), FailureReason> {
        let mut state = self.lock();
        state.provider.release_execution_lease(key)?;
        if let Some(frozen) = state.preparations.get_mut(&key) {
            frozen.purpose = MaterializationPurpose::Prefetch;
        }
        Ok(())
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
        let expected = state.preparations.get(&key).copied().ok_or_else(|| {
            FailureReason::ContractViolation {
                message: "physical operation has no frozen provider preparation".into(),
            }
        })?;
        if !matches!(
            expected.preparation,
            MaterializationPreparation::Transfer(_)
        ) {
            return Err(FailureReason::ContractViolation {
                message: "physical operation reserve requires a prepared transfer".into(),
            });
        }
        let expected_binding = expected.preparation.binding();
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
            return Err(match cleanup {
                Ok(()) => FailureReason::ContractViolation {
                    message: violation.into(),
                },
                Err(cleanup) => FailureReason::ContractCleanup {
                    message: violation.into(),
                    cleanup: Box::new(cleanup),
                },
            });
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

    fn promote_to_execution(
        &mut self,
        key: MaterializationKey,
    ) -> Result<ExecutionPromotion, FailureReason> {
        (**self).promote_to_execution(key)
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

    fn promote_to_execution(
        &mut self,
        _key: MaterializationKey,
    ) -> Result<ExecutionPromotion, FailureReason> {
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
