//! DeepSeek-V4 CUDA materialization provider.
//!
//! This module owns one prepared-generation source catalog and the physical CUDA
//! install authority. Routed experts currently implement the full asynchronous
//! path; static tensor bundles remain fail-closed until the generic pinned
//! checkpoint transport is connected. Runtime code owns logical demand, hard
//! credits, operation IDs, completion validation, publication, and retirement.

#![cfg(feature = "cuda")]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use ferrule_common::materialization_io::{
    MaterializationResourceLimits, MaterializationResourcePlan,
};
use ferrule_common::{
    CancellationReason, CompletionEvent, CompletionGeneration, CompletionOutcome,
    CompletionTimestamp, DestinationGeneration, DestinationSlotId, Error,
    ExpertInstallActivationOutcome, ExpertInstallIntent, ExpertInstallPrepareOutcome,
    ExpertInstallReason, ExpertKey, ExpertLease, ExpertResidencyControl, ExpertResidencyStats,
    ExpertSlotBinding, FailureReason, FenceId, LoadStage, MaterializationKey, OperationId,
    PreparedExpertInstall, ResidencyBinding, ResidencyLeaseSet, Result, StaleReason,
    UploadFenceContract, ValidatedResidencyBinding,
};

use crate::checkpoint::CheckpointReadPlan;
use crate::materialization::{
    MaterializationPlacement, MaterializationPreparation, MaterializationProvider,
    MaterializationRequest, MaterializationResident, MaterializationSourceCatalog,
    MaterializationTransfer, PhysicalMaterializationOperationReservation,
    PhysicalMaterializationTopology,
};
use crate::moe::streaming::{
    ExpertId, ExpertLinearFormat, ExpertLoadSource, ExpertMatrixKind, ExpertStreamingReader,
    PinnedExpertArtifactPayload, PinnedExpertLoadPlan, PinnedExpertReadPoll,
    PinnedExpertReadTicket, infer_expert_linear_format,
};
use crate::runner::completion_notify_callback;

use super::prepared::DeepSeekV4InstallDescriptor;

#[derive(Clone)]
pub(super) struct DeepSeekV4SharedExpertSubsystem {
    inner: Arc<Mutex<DeepSeekV4ExpertSubsystemState>>,
}

impl std::fmt::Debug for DeepSeekV4SharedExpertSubsystem {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.lock();
        formatter
            .debug_struct("DeepSeekV4SharedExpertSubsystem")
            .field("has_residency_control", &state.residency.is_some())
            .field("resident_experts", &state.experts.len())
            .field("slot_tables", &state.tables.len())
            .field("poisoned_layers", &state.poisoned_layers)
            .finish()
    }
}

impl DeepSeekV4SharedExpertSubsystem {
    fn new(
        tables: BTreeMap<usize, ferrule_cuda::context::CudaExpertSlotTable>,
        frame_capacity: usize,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(DeepSeekV4ExpertSubsystemState {
                residency: None,
                tables,
                experts: BTreeMap::new(),
                free_frames: Vec::new(),
                frame_capacity,
                frames_allocated: 0,
                poisoned_layers: BTreeSet::new(),
                resident_keys: HashMap::new(),
            })),
        }
    }

    fn lock(&self) -> MutexGuard<'_, DeepSeekV4ExpertSubsystemState> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(super) fn install_residency_control(
        &self,
        control: Box<dyn ExpertResidencyControl>,
    ) -> Result<()> {
        let mut state = self.lock();
        if state.residency.is_some() {
            return Err(Error::Execution(
                "DeepSeek-V4 physical residency control is already installed".into(),
            ));
        }
        state.residency = Some(control);
        Ok(())
    }

    pub(super) fn resident_stats_for_layer(&self, layer: usize) -> (usize, u64) {
        let state = self.lock();
        state
            .experts
            .iter()
            .filter(|(expert, _)| expert.layer == layer)
            .fold((0usize, 0u64), |(count, bytes), (_, frame)| {
                (count.saturating_add(1), bytes.saturating_add(frame.bytes))
            })
    }

    pub(super) fn resident_experts_for_layer(&self, layer: usize) -> Result<BTreeSet<usize>> {
        let state = self.lock();
        if state.poisoned_layers.contains(&layer) {
            return Err(Error::Execution(format!(
                "DeepSeek-V4 published expert layer {layer} is poisoned"
            )));
        }
        if !state.tables.contains_key(&layer) {
            return Err(Error::Execution(format!(
                "DeepSeek-V4 published expert table is missing layer {layer}"
            )));
        }
        Ok(state
            .experts
            .keys()
            .filter_map(|expert| (expert.layer == layer).then_some(expert.expert))
            .collect())
    }

    pub(super) fn layer_slot_capacity(&self, layer: usize) -> Result<usize> {
        let state = self.lock();
        if state.poisoned_layers.contains(&layer) {
            return Err(Error::Execution(format!(
                "DeepSeek-V4 published expert layer {layer} is poisoned"
            )));
        }
        state
            .tables
            .get(&layer)
            .map(|table| table.host().slot_capacity())
            .ok_or_else(|| {
                Error::Execution(format!(
                    "DeepSeek-V4 published expert table is missing layer {layer}"
                ))
            })
    }

    /// Execute against one published layer table after proving that the runtime
    /// lease set exactly covers the selected expert window. No frame, pointer, or
    /// table escapes this owner and unpublished reservations are never visible.
    pub(super) fn with_validated_published_experts<T>(
        &self,
        layer: usize,
        selected: &[usize],
        leases: &ResidencyLeaseSet,
        execute: impl FnOnce(&ferrule_cuda::context::CudaExpertSlotTable, &CudaExpertFrame) -> Result<T>,
    ) -> Result<T> {
        let state = self.lock();
        if state.poisoned_layers.contains(&layer) {
            return Err(Error::Execution(format!(
                "DeepSeek-V4 published expert layer {layer} is poisoned"
            )));
        }
        let table = state.tables.get(&layer).ok_or_else(|| {
            Error::Execution(format!(
                "DeepSeek-V4 published expert table is missing layer {layer}"
            ))
        })?;
        validate_published_lease_window(layer, selected, leases, |expert_index| {
            let expert = ExpertId::new(layer, expert_index);
            let layer_u32 = u32::try_from(layer).ok()?;
            let expert_u32 = u32::try_from(expert_index).ok()?;
            let key = state
                .resident_keys
                .values()
                .find(|key| {
                    key.resource().routed_expert_coordinates().is_some_and(
                        |(key_layer, key_expert)| {
                            key_layer.get() == layer_u32 && key_expert.get() == expert_u32
                        },
                    )
                })
                .copied()?;
            let physical = table.host().binding(expert_index)?;
            Some(PublishedExpertBinding {
                key,
                slot: DestinationSlotId::new(physical.slot.try_into().ok()?),
                generation: DestinationGeneration::new(physical.generation.try_into().ok()?),
                frame_published: state.experts.contains_key(&expert),
            })
        })?;

        let first = ExpertId::new(layer, selected[0]);
        let frame = state.experts.get(&first).ok_or_else(|| {
            Error::Execution(format!(
                "DeepSeek-V4 published expert frame is unavailable for {}:{}",
                first.layer, first.expert
            ))
        })?;
        execute(table, frame)
    }
}

#[derive(Debug, Clone, Copy)]
struct PublishedExpertBinding {
    key: MaterializationKey,
    slot: DestinationSlotId,
    generation: DestinationGeneration,
    frame_published: bool,
}

fn validate_published_lease_window(
    layer: usize,
    selected: &[usize],
    leases: &ResidencyLeaseSet,
    mut published: impl FnMut(usize) -> Option<PublishedExpertBinding>,
) -> Result<()> {
    if selected.is_empty() || leases.len() != selected.len() {
        return Err(Error::Execution(format!(
            "DeepSeek-V4 expert lease window mismatch: selected={} leases={}",
            selected.len(),
            leases.len()
        )));
    }
    let layer_u32 = u32::try_from(layer)
        .map_err(|_| Error::Execution("expert layer exceeds u32 ABI".into()))?;
    let mut selected_set = BTreeSet::new();
    for &expert_index in selected {
        if !selected_set.insert(expert_index) {
            return Err(Error::Execution(format!(
                "DeepSeek-V4 selected expert window contains duplicate {layer}:{expert_index}"
            )));
        }
    }
    for candidate in leases.bindings() {
        let key = candidate.key();
        let (key_layer, key_expert) =
            key.resource().routed_expert_coordinates().ok_or_else(|| {
                Error::Execution(
                    "DeepSeek-V4 expert lease window received a non-routed-expert resource".into(),
                )
            })?;
        let expert_index = usize::try_from(key_expert.get())
            .map_err(|_| Error::Execution("expert index exceeds usize".into()))?;
        if key_layer.get() != layer_u32 || !selected_set.contains(&expert_index) {
            return Err(Error::Execution(format!(
                "DeepSeek-V4 lease binding {}:{} is outside the selected expert window",
                key_layer.get(),
                key_expert.get()
            )));
        }
    }
    for &expert_index in selected {
        let expert_u32 = u32::try_from(expert_index)
            .map_err(|_| Error::Execution("expert index exceeds u32 ABI".into()))?;
        let mut candidates = leases.bindings().iter().filter(|candidate| {
            candidate
                .key()
                .resource()
                .routed_expert_coordinates()
                .is_some_and(|(key_layer, key_expert)| {
                    key_layer.get() == layer_u32 && key_expert.get() == expert_u32
                })
        });
        let candidate = candidates.next().ok_or_else(|| {
            Error::Execution(format!(
                "DeepSeek-V4 selected expert {layer}:{expert_index} has no lease binding"
            ))
        })?;
        if candidates.next().is_some() {
            return Err(Error::Execution(format!(
                "DeepSeek-V4 selected expert {layer}:{expert_index} has duplicate lease bindings"
            )));
        }
        let published = published(expert_index).ok_or_else(|| {
            Error::Execution(format!(
                "DeepSeek-V4 expert {layer}:{expert_index} is not published"
            ))
        })?;
        let binding = candidate.binding();
        if !published.frame_published
            || candidate.key() != published.key
            || binding.slot != published.slot
            || binding.generation != published.generation
        {
            return Err(Error::Execution(format!(
                "DeepSeek-V4 expert {layer}:{expert_index} lease does not match the published key/slot/generation"
            )));
        }
    }
    Ok(())
}

type ResidentExpertKeyIndex = HashMap<ExpertKey, MaterializationKey>;

struct DeepSeekV4ExpertSubsystemState {
    residency: Option<Box<dyn ExpertResidencyControl>>,
    tables: BTreeMap<usize, ferrule_cuda::context::CudaExpertSlotTable>,
    experts: BTreeMap<ExpertId, CudaExpertFrame>,
    free_frames: Vec<CudaExpertFrame>,
    frame_capacity: usize,
    frames_allocated: usize,
    poisoned_layers: BTreeSet<usize>,
    resident_keys: ResidentExpertKeyIndex,
}

pub(super) struct CudaExpertFrame {
    pub(super) gate: ferrule_cuda::context::CudaArtifactLinearHandle,
    pub(super) up: ferrule_cuda::context::CudaArtifactLinearHandle,
    pub(super) down: ferrule_cuda::context::CudaArtifactLinearHandle,
    pub(super) bytes: u64,
}

struct SlotReservation {
    request: MaterializationRequest,
    expert: ExpertId,
    read_plan: CheckpointReadPlan,
    pinned_plan: PinnedExpertLoadPlan,
    resource_plan: MaterializationResourcePlan,
    prepared: PreparedExpertInstall,
    binding: ResidencyBinding,
    evicted: Option<MaterializationKey>,
}

struct MaterializationOperation {
    key: MaterializationKey,
    request: MaterializationRequest,
    expert: ExpertId,
    read_plan: CheckpointReadPlan,
    resource_plan: MaterializationResourcePlan,
    prepared: Option<PreparedExpertInstall>,
    binding: ResidencyBinding,
    evicted: Option<MaterializationKey>,
    pending_terminal: Option<CompletionOutcome>,
    state: Option<CudaMaterializationOperationState>,
}

#[derive(Debug, Clone, Copy)]
enum SelectedLeaseState {
    Pending,
    ReleaseRequested,
    Active(ExpertLease),
}

#[derive(Debug, Clone, Copy)]
struct SelectedLeaseOwnership {
    operation: Option<OperationId>,
    state: SelectedLeaseState,
}

#[derive(Debug, Default)]
struct SelectedLeaseTracker {
    ownership: HashMap<MaterializationKey, SelectedLeaseOwnership>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectedLeasePublication {
    Retained,
    ReleaseImmediately,
}

#[derive(Debug, Clone, Copy)]
enum SelectedLeaseRelease {
    None,
    Deferred,
    Active {
        lease: ExpertLease,
        operation: Option<OperationId>,
    },
}

impl SelectedLeaseTracker {
    fn begin_pending(&mut self, key: MaterializationKey) -> std::result::Result<(), FailureReason> {
        if self.ownership.contains_key(&key) {
            return Err(FailureReason::ProtocolViolation(
                "selected expert already has physical execution ownership".into(),
            ));
        }
        self.ownership.insert(
            key,
            SelectedLeaseOwnership {
                operation: None,
                state: SelectedLeaseState::Pending,
            },
        );
        Ok(())
    }

    fn bind_operation(
        &mut self,
        key: MaterializationKey,
        operation: OperationId,
    ) -> std::result::Result<(), FailureReason> {
        let ownership = self.ownership.get_mut(&key).ok_or_else(|| {
            FailureReason::ProtocolViolation(
                "prepared selected expert has no physical execution ownership".into(),
            )
        })?;
        if ownership.operation.is_some() || matches!(ownership.state, SelectedLeaseState::Active(_))
        {
            return Err(FailureReason::ProtocolViolation(
                "selected expert execution ownership already has an operation".into(),
            ));
        }
        ownership.operation = Some(operation);
        Ok(())
    }

    fn begin_active(
        &mut self,
        key: MaterializationKey,
        lease: ExpertLease,
    ) -> std::result::Result<(), FailureReason> {
        if self.ownership.contains_key(&key) {
            return Err(FailureReason::ProtocolViolation(
                "selected resident expert already has physical execution ownership".into(),
            ));
        }
        self.ownership.insert(
            key,
            SelectedLeaseOwnership {
                operation: None,
                state: SelectedLeaseState::Active(lease),
            },
        );
        Ok(())
    }

    fn publish(
        &mut self,
        key: MaterializationKey,
        operation: OperationId,
        lease: ExpertLease,
    ) -> std::result::Result<SelectedLeasePublication, FailureReason> {
        let ownership = self.ownership.get_mut(&key).ok_or_else(|| {
            FailureReason::ProtocolViolation(
                "selected expert publication has no pending execution ownership".into(),
            )
        })?;
        if ownership.operation != Some(operation) {
            return Err(FailureReason::ProtocolViolation(
                "selected expert publication operation is stale".into(),
            ));
        }
        match ownership.state {
            SelectedLeaseState::Pending => {
                ownership.state = SelectedLeaseState::Active(lease);
                Ok(SelectedLeasePublication::Retained)
            }
            SelectedLeaseState::ReleaseRequested => {
                ownership.state = SelectedLeaseState::Active(lease);
                Ok(SelectedLeasePublication::ReleaseImmediately)
            }
            SelectedLeaseState::Active(_) => Err(FailureReason::ProtocolViolation(
                "selected expert publication replaced an active execution lease".into(),
            )),
        }
    }

    fn request_release(&mut self, key: MaterializationKey) -> SelectedLeaseRelease {
        let Some(ownership) = self.ownership.get(&key).copied() else {
            return SelectedLeaseRelease::None;
        };
        match ownership.state {
            SelectedLeaseState::Pending => {
                self.ownership
                    .get_mut(&key)
                    .expect("selected ownership was read above")
                    .state = SelectedLeaseState::ReleaseRequested;
                SelectedLeaseRelease::Deferred
            }
            SelectedLeaseState::ReleaseRequested => SelectedLeaseRelease::Deferred,
            SelectedLeaseState::Active(lease) => {
                self.ownership.remove(&key);
                SelectedLeaseRelease::Active {
                    lease,
                    operation: ownership.operation,
                }
            }
        }
    }

    fn restore_active(
        &mut self,
        key: MaterializationKey,
        lease: ExpertLease,
        operation: Option<OperationId>,
    ) {
        let previous = self.ownership.insert(
            key,
            SelectedLeaseOwnership {
                operation,
                state: SelectedLeaseState::Active(lease),
            },
        );
        debug_assert!(previous.is_none());
    }

    fn cancel_operation(&mut self, key: MaterializationKey, operation: OperationId) {
        if self.ownership.get(&key).is_some_and(|ownership| {
            ownership.operation == Some(operation)
                && !matches!(ownership.state, SelectedLeaseState::Active(_))
        }) {
            self.ownership.remove(&key);
        }
    }

    fn cancel_unbound(&mut self, key: MaterializationKey) {
        if self.ownership.get(&key).is_some_and(|ownership| {
            ownership.operation.is_none()
                && !matches!(ownership.state, SelectedLeaseState::Active(_))
        }) {
            self.ownership.remove(&key);
        }
    }

    fn active_owned_by(&self, key: MaterializationKey, operation: OperationId) -> bool {
        self.ownership.get(&key).is_some_and(|ownership| {
            ownership.operation == Some(operation)
                && matches!(ownership.state, SelectedLeaseState::Active(_))
        })
    }

    fn contains(&self, key: MaterializationKey) -> bool {
        self.ownership.contains_key(&key)
    }

    fn active_count(&self) -> usize {
        self.ownership
            .values()
            .filter(|ownership| matches!(ownership.state, SelectedLeaseState::Active(_)))
            .count()
    }

    fn pending_count(&self) -> usize {
        self.ownership.len().saturating_sub(self.active_count())
    }
}

type CudaMaterializationOperationState = MaterializationOperationState<
    PinnedExpertReadTicket,
    PinnedExpertBundle,
    CudaExpertUploadTicket,
    CudaExpertFrame,
>;

enum MaterializationOperationState<Read, Host, Upload, Frame> {
    Reserved(Read),
    ReadSubmitted(Read),
    HostReady(Host),
    UploadSubmitted(Upload),
    UploadReady(Frame),
    Installing(Frame),
}

impl<Read, Host, Upload, Frame> MaterializationOperationState<Read, Host, Upload, Frame> {
    const fn owner_stage(&self) -> LoadStage {
        match self {
            Self::Reserved(_) => LoadStage::Reserved,
            Self::ReadSubmitted(_) => LoadStage::ReadSubmitted,
            Self::HostReady(_) => LoadStage::HostReady,
            Self::UploadSubmitted(_) => LoadStage::UploadSubmitted,
            Self::UploadReady(_) | Self::Installing(_) => LoadStage::Installing,
        }
    }
}

fn cancel_observation_matches_physical_state<Read, Host, Upload, Frame>(
    observed: LoadStage,
    physical: &MaterializationOperationState<Read, Host, Upload, Frame>,
) -> bool {
    observed == physical.owner_stage()
        || matches!(
            (observed, physical),
            (
                LoadStage::ReadSubmitted,
                MaterializationOperationState::HostReady(_)
            ) | (
                LoadStage::UploadSubmitted,
                MaterializationOperationState::UploadReady(_)
            )
        )
}

struct PinnedExpertLinear {
    matrix: ExpertMatrixKind,
    format: ExpertLinearFormat,
    weight: ferrule_cuda::context::CudaPinnedU8HostBuffer,
    scale: ferrule_cuda::context::CudaPinnedU8HostBuffer,
}

struct PinnedExpertBundle {
    expert: ExpertId,
    gate: PinnedExpertLinear,
    up: PinnedExpertLinear,
    down: PinnedExpertLinear,
    bytes: u64,
}

struct CudaExpertUploadTicket {
    frame: Option<CudaExpertFrame>,
    gate: Option<ferrule_cuda::context::CudaArtifactLinearAsyncOverwrite>,
    up: Option<ferrule_cuda::context::CudaArtifactLinearAsyncOverwrite>,
    down: Option<ferrule_cuda::context::CudaArtifactLinearAsyncOverwrite>,
    event: ferrule_cuda::context::CudaUploadEvent,
}

impl CudaExpertUploadTicket {
    fn is_complete(&self) -> Result<bool> {
        self.event.is_complete()
    }

    fn drain_into_frame(mut self) -> Result<CudaExpertFrame> {
        self.event.synchronize()?;
        self.gate.take();
        self.up.take();
        self.down.take();
        self.frame
            .take()
            .ok_or_else(|| Error::Internal("CUDA expert upload lost its frame".into()))
    }
}

/// Runner-owned factory result for the single CUDA materialization authority.
///
/// The operator cache receives the expert execution handle while the provider is
/// transferred exactly once to the runtime registry. Both point at the same
/// frame/table/residency state.
pub(super) struct DeepSeekV4CudaMaterializationOwner {
    placement: MaterializationPlacement,
    shared: DeepSeekV4SharedExpertSubsystem,
    provider: Option<DeepSeekV4CudaMaterializationProvider>,
}

pub struct DeepSeekV4CudaMaterializationProvider {
    placement: MaterializationPlacement,
    topology: PhysicalMaterializationTopology,
    sources: Arc<MaterializationSourceCatalog<DeepSeekV4InstallDescriptor>>,
    reader: ExpertStreamingReader,
    ops: ferrule_cuda::context::CudaArtifactOperatorContext,
    completion_hub: ferrule_common::CompletionHub,
    shared: DeepSeekV4SharedExpertSubsystem,
    request_keys: BTreeMap<MaterializationRequest, MaterializationKey>,
    resource_plans: BTreeMap<MaterializationKey, MaterializationResourcePlan>,
    selected_lease_ownership: SelectedLeaseTracker,
    reservations: BTreeMap<MaterializationKey, SlotReservation>,
    operations: BTreeMap<OperationId, MaterializationOperation>,
    operation_order: VecDeque<OperationId>,
    seen_operations: HashSet<OperationId>,
    terminal_operations: HashSet<OperationId>,
    completions: VecDeque<CompletionEvent>,
    clock_ns: u64,
}

impl std::fmt::Debug for DeepSeekV4CudaMaterializationProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeepSeekV4CudaMaterializationProvider")
            .field("placement", &self.placement)
            .field("source_entries", &self.sources.len())
            .field("reservations", &self.reservations.len())
            .field(
                "active_selected_leases",
                &self.selected_lease_ownership.active_count(),
            )
            .field(
                "pending_selected_ownership",
                &self.selected_lease_ownership.pending_count(),
            )
            .field("operations", &self.operations.len())
            .field("terminal_operations", &self.terminal_operations.len())
            .field("shared", &self.shared)
            .finish()
    }
}

impl DeepSeekV4CudaMaterializationOwner {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create(
        placement: MaterializationPlacement,
        limits: MaterializationResourceLimits,
        sources: Arc<MaterializationSourceCatalog<DeepSeekV4InstallDescriptor>>,
        reader: ExpertStreamingReader,
        expert_capacity: usize,
        layer_slot_capacities: &[(usize, usize)],
    ) -> Result<Self> {
        let provider = DeepSeekV4CudaMaterializationProvider::new(
            placement,
            limits,
            sources,
            reader,
            expert_capacity,
            layer_slot_capacities,
        )?;
        let shared = provider.shared.clone();
        Ok(Self {
            placement,
            shared,
            provider: Some(provider),
        })
    }

    pub(super) fn handle(&self) -> DeepSeekV4SharedExpertSubsystem {
        self.shared.clone()
    }

    pub(super) fn residency_control_installed(&self) -> bool {
        self.shared.lock().residency.is_some()
    }

    pub(super) fn install_residency_control(
        &self,
        control: Box<dyn ExpertResidencyControl>,
    ) -> Result<()> {
        if control.requirements().model_instance != self.placement.model().get() {
            return Err(Error::Execution(
                "DeepSeek-V4 physical residency model namespace mismatch".into(),
            ));
        }
        self.shared.install_residency_control(control)
    }

    pub(super) fn residency_stats(&self) -> ExpertResidencyStats {
        self.shared
            .lock()
            .residency
            .as_ref()
            .map_or_else(ExpertResidencyStats::default, |control| control.stats())
    }

    pub(super) fn take_provider(&mut self) -> Option<Box<dyn MaterializationProvider>> {
        self.provider
            .take()
            .map(|provider| Box::new(provider) as Box<dyn MaterializationProvider>)
    }
}

impl DeepSeekV4CudaMaterializationProvider {
    #[allow(clippy::too_many_arguments)]
    fn new(
        placement: MaterializationPlacement,
        limits: MaterializationResourceLimits,
        sources: Arc<MaterializationSourceCatalog<DeepSeekV4InstallDescriptor>>,
        reader: ExpertStreamingReader,
        expert_capacity: usize,
        layer_slot_capacities: &[(usize, usize)],
    ) -> Result<Self> {
        let limits = limits.validate()?;
        let ops = ferrule_cuda::context::CudaArtifactOperatorContext::new()?;
        let mut routed_frame_bytes = BTreeMap::<usize, u64>::new();
        for entry in sources.iter() {
            match entry.descriptor() {
                DeepSeekV4InstallDescriptor::StaticTensorBundle(_) => {
                    if entry.resource().routed_expert_coordinates().is_some() {
                        return Err(Error::Model(
                            "DeepSeek-V4 static tensor-bundle descriptor is bound to a routed-expert resource"
                                .into(),
                        ));
                    }
                }
                DeepSeekV4InstallDescriptor::RoutedExpert(source) => {
                    let (layer, _) =
                        entry.resource().routed_expert_coordinates().ok_or_else(|| {
                            Error::Model(
                                "DeepSeek-V4 routed-expert install descriptor is bound to a non-expert resource"
                                    .into(),
                            )
                        })?;
                    let layer = usize::try_from(layer.get()).map_err(|_| {
                        Error::Model("DeepSeek-V4 routed-expert layer exceeds usize".into())
                    })?;
                    let bytes = source.bytes();
                    if bytes == 0 || bytes != entry.read_plan().storage_bytes() {
                        return Err(Error::Model(format!(
                            "DeepSeek-V4 routed-expert source/read size mismatch at layer {layer}: source={bytes} read={}",
                            entry.read_plan().storage_bytes()
                        )));
                    }
                    routed_frame_bytes
                        .entry(layer)
                        .and_modify(|maximum| *maximum = (*maximum).max(bytes))
                        .or_insert(bytes);
                }
            }
        }

        let mut tables = BTreeMap::new();
        let mut resident_capacity = 0usize;
        let mut resident_frame_bytes = 0u64;
        let mut maximum_frame_bytes = 0u64;
        let mut residency_lease_slots_per_continuation = 0u64;
        for &(layer, slots) in layer_slot_capacities {
            if slots == 0 || tables.contains_key(&layer) {
                return Err(Error::Model(format!(
                    "invalid or duplicate DeepSeek-V4 physical expert layer capacity {layer}:{slots}"
                )));
            }
            let layer_frame_bytes = routed_frame_bytes.get(&layer).copied().ok_or_else(|| {
                Error::Model(format!(
                    "DeepSeek-V4 CUDA materialization catalog has no routed experts for layer {layer}"
                ))
            })?;
            let slots_u64 = u64::try_from(slots)
                .map_err(|_| Error::Model("physical expert slot capacity exceeds u64".into()))?;
            resident_frame_bytes = resident_frame_bytes
                .checked_add(layer_frame_bytes.checked_mul(slots_u64).ok_or_else(|| {
                    Error::Model("physical expert resident byte capacity overflow".into())
                })?)
                .ok_or_else(|| {
                    Error::Model("physical expert resident byte capacity overflow".into())
                })?;
            maximum_frame_bytes = maximum_frame_bytes.max(layer_frame_bytes);
            residency_lease_slots_per_continuation =
                residency_lease_slots_per_continuation.max(slots_u64);
            resident_capacity = resident_capacity
                .checked_add(slots)
                .ok_or_else(|| Error::Model("physical expert frame capacity overflow".into()))?;
            tables.insert(layer, ops.expert_slot_table(expert_capacity, slots)?);
        }
        if let Some(layer) = routed_frame_bytes
            .keys()
            .find(|layer| !tables.contains_key(layer))
        {
            return Err(Error::Model(format!(
                "DeepSeek-V4 CUDA materialization catalog has no slot capacity for routed-expert layer {layer}"
            )));
        }
        let shadow = usize::try_from(limits.capacity.upload_slots.max(1))
            .map_err(|_| Error::Model("physical expert upload slots exceed usize".into()))?;
        let shadow_u64 = u64::try_from(shadow)
            .map_err(|_| Error::Model("physical expert shadow capacity exceeds u64".into()))?;
        let resident_capacity_bytes = resident_frame_bytes
            .checked_add(maximum_frame_bytes.checked_mul(shadow_u64).ok_or_else(|| {
                Error::Model("physical expert shadow byte capacity overflow".into())
            })?)
            .ok_or_else(|| Error::Model("physical expert frame byte capacity overflow".into()))?;
        let frame_capacity = resident_capacity
            .checked_add(shadow)
            .ok_or_else(|| Error::Model("physical expert frame capacity overflow".into()))?;
        let topology = PhysicalMaterializationTopology::new(
            limits,
            resident_capacity_bytes,
            residency_lease_slots_per_continuation,
        )?;
        let completion_hub = reader.completion_hub();
        Ok(Self {
            placement,
            topology,
            sources,
            reader,
            ops,
            completion_hub,
            shared: DeepSeekV4SharedExpertSubsystem::new(tables, frame_capacity),
            request_keys: BTreeMap::new(),
            resource_plans: BTreeMap::new(),
            selected_lease_ownership: SelectedLeaseTracker::default(),
            reservations: BTreeMap::new(),
            operations: BTreeMap::new(),
            operation_order: VecDeque::new(),
            seen_operations: HashSet::new(),
            terminal_operations: HashSet::new(),
            completions: VecDeque::new(),
            clock_ns: 1,
        })
    }

    fn source_for_request(
        &self,
        request: MaterializationRequest,
    ) -> std::result::Result<(ExpertId, CheckpointReadPlan, ExpertLoadSource), FailureReason> {
        request
            .validate_key(
                request
                    .materialization_key(DestinationGeneration::new(1))
                    .map_err(protocol_failure)?,
            )
            .map_err(protocol_failure)?;
        if request.model() != self.placement.model()
            || request.backend() != self.placement.backend()
            || request.device() != self.placement.device()
        {
            return Err(FailureReason::ProtocolViolation(
                "DeepSeek-V4 request placement mismatch".into(),
            ));
        }
        let entry = self.sources.resolve(request)?;
        let install_source = match entry.descriptor() {
            DeepSeekV4InstallDescriptor::RoutedExpert(source) => source.clone(),
            DeepSeekV4InstallDescriptor::StaticTensorBundle(_) => {
                return Err(FailureReason::ProtocolViolation(
                    "DeepSeek-V4 static CUDA tensor-bundle installation is not implemented".into(),
                ));
            }
        };
        let (protocol_layer, protocol_expert) =
            request.routed_expert_coordinates().ok_or_else(|| {
                FailureReason::ProtocolViolation(
                    "DeepSeek-V4 routed-expert descriptor is bound to a non-expert resource".into(),
                )
            })?;
        let layer = usize::try_from(protocol_layer.get())
            .map_err(|_| FailureReason::ProtocolViolation("expert layer exceeds usize".into()))?;
        let expert_index = usize::try_from(protocol_expert.get())
            .map_err(|_| FailureReason::ProtocolViolation("expert index exceeds usize".into()))?;
        Ok((
            ExpertId::new(layer, expert_index),
            entry.read_plan().clone(),
            install_source,
        ))
    }

    fn source_for_key(
        &self,
        key: MaterializationKey,
    ) -> std::result::Result<
        (
            MaterializationRequest,
            ExpertId,
            CheckpointReadPlan,
            ExpertLoadSource,
        ),
        FailureReason,
    > {
        key.validate().map_err(|error| {
            FailureReason::ProtocolViolation(format!("invalid physical load key: {error}"))
        })?;
        let request = self
            .request_keys
            .iter()
            .find_map(|(request, candidate)| (*candidate == key).then_some(*request))
            .ok_or_else(|| {
                FailureReason::ProtocolViolation(
                    "physical MaterializationKey cannot be recovered to a source request".into(),
                )
            })?;
        request.validate_key(key).map_err(protocol_failure)?;
        let (expert, read_plan, install_source) = self.source_for_request(request)?;
        Ok((request, expert, read_plan, install_source))
    }

    fn protocol_binding(
        &self,
        request: MaterializationRequest,
        slot: ExpertSlotBinding,
    ) -> std::result::Result<ResidencyBinding, FailureReason> {
        let generation = DestinationGeneration::new(u64::from(slot.generation.get()));
        if generation.is_zero() {
            return Err(FailureReason::ProtocolViolation(
                "physical expert slot generation is zero".into(),
            ));
        }
        Ok(ResidencyBinding::new(
            request.model(),
            request.resource(),
            request.backend(),
            request.device(),
            DestinationSlotId::new(slot.slot.get()),
            generation,
        ))
    }

    fn remove_request_key(&mut self, request: MaterializationRequest, key: MaterializationKey) {
        if self.request_keys.get(&request) == Some(&key) {
            self.request_keys.remove(&request);
        }
        self.resource_plans.remove(&key);
        self.reservations.remove(&key);
    }

    fn remove_key(&mut self, key: MaterializationKey) {
        if let Some(request) = self
            .request_keys
            .iter()
            .find_map(|(request, candidate)| (*candidate == key).then_some(*request))
        {
            self.remove_request_key(request, key);
        } else {
            self.resource_plans.remove(&key);
            self.reservations.remove(&key);
        }
    }

    fn remove_operation_order(&mut self, operation: OperationId) {
        self.operation_order
            .retain(|candidate| *candidate != operation);
    }

    fn cancel_prepared(&self, prepared: PreparedExpertInstall) -> Result<()> {
        let mut shared = self.shared.lock();
        shared
            .residency
            .as_mut()
            .ok_or_else(|| Error::Execution("physical residency control is not installed".into()))?
            .cancel_install(prepared)
    }

    fn rollback_slot_reservation(
        &mut self,
        key: MaterializationKey,
        request: MaterializationRequest,
        prepared: PreparedExpertInstall,
        primary: FailureReason,
    ) -> FailureReason {
        let cleanup = self.cancel_prepared(prepared);
        self.selected_lease_ownership.cancel_unbound(key);
        self.remove_request_key(request, key);
        match cleanup {
            Ok(()) => primary,
            Err(error) => FailureReason::ProtocolViolation(format!(
                "physical reservation failed ({primary:?}); prepared slot rollback also failed ({error})"
            )),
        }
    }

    fn emit(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        stage: LoadStage,
        outcome: CompletionOutcome,
        bytes: u64,
    ) {
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
        self.completion_hub.notify();
    }

    fn recycle_frame(&self, mut frame: CudaExpertFrame) {
        frame.bytes = 0;
        self.shared.lock().free_frames.push(frame);
    }

    fn allocate_frame(&self, bundle: &PinnedExpertBundle) -> Result<CudaExpertFrame> {
        let mut shared = self.shared.lock();
        if let Some(mut frame) = shared.free_frames.pop() {
            frame.bytes = bundle.bytes;
            return Ok(frame);
        }
        if shared.frames_allocated >= shared.frame_capacity {
            return Err(Error::Execution(format!(
                "DeepSeek-V4 physical expert frame pool exhausted at {} frames",
                shared.frame_capacity
            )));
        }
        let gate_shape = pinned_linear_shape(&bundle.gate)?;
        let up_shape = pinned_linear_shape(&bundle.up)?;
        let down_shape = pinned_linear_shape(&bundle.down)?;
        let frame = CudaExpertFrame {
            gate: self.ops.allocate_artifact_linear_device(gate_shape)?,
            up: self.ops.allocate_artifact_linear_device(up_shape)?,
            down: self.ops.allocate_artifact_linear_device(down_shape)?,
            bytes: bundle.bytes,
        };
        shared.frames_allocated = shared.frames_allocated.saturating_add(1);
        Ok(frame)
    }

    fn submit_bundle_upload(&self, bundle: PinnedExpertBundle) -> Result<CudaExpertUploadTicket> {
        let mut frame = self.allocate_frame(&bundle)?;
        let submitted = (|| {
            let gate = self.ops.overwrite_artifact_linear_from_pinned_async(
                &mut frame.gate,
                pinned_linear_shape(&bundle.gate)?,
                bundle.gate.weight.clone(),
                Some(bundle.gate.scale.clone()),
            )?;
            let up = self.ops.overwrite_artifact_linear_from_pinned_async(
                &mut frame.up,
                pinned_linear_shape(&bundle.up)?,
                bundle.up.weight.clone(),
                Some(bundle.up.scale.clone()),
            )?;
            let down = self.ops.overwrite_artifact_linear_from_pinned_async(
                &mut frame.down,
                pinned_linear_shape(&bundle.down)?,
                bundle.down.weight.clone(),
                Some(bundle.down.scale.clone()),
            )?;
            let event = self.ops.record_upload_event()?;
            self.ops
                .notify_upload_stream(completion_notify_callback(self.completion_hub.clone()))?;
            Ok((gate, up, down, event))
        })();
        match submitted {
            Ok((gate, up, down, event)) => Ok(CudaExpertUploadTicket {
                frame: Some(frame),
                gate: Some(gate),
                up: Some(up),
                down: Some(down),
                event,
            }),
            Err(error) => {
                if let Err(sync_error) = self.ops.sync_upload_stream() {
                    return Err(Error::Internal(format!(
                        "expert upload submission failed ({error}); upload drain also failed ({sync_error})"
                    )));
                }
                self.recycle_frame(frame);
                Err(error)
            }
        }
    }

    fn operation_identity_outcome(
        operation: &MaterializationOperation,
    ) -> Option<CompletionOutcome> {
        if let Err(error) = operation.request.validate_key(operation.key) {
            return Some(CompletionOutcome::Failed(protocol_failure(error)));
        }
        if let Err(error) = ValidatedResidencyBinding::new(operation.key, operation.binding) {
            return Some(CompletionOutcome::Failed(FailureReason::ProtocolViolation(
                error.to_string(),
            )));
        }
        let Some(prepared) = operation.prepared else {
            return Some(CompletionOutcome::Failed(FailureReason::ProtocolViolation(
                "physical operation lost prepared slot ownership".into(),
            )));
        };
        let expected = prepared.binding();
        let Some((layer, expert)) = operation.request.routed_expert_coordinates() else {
            return Some(CompletionOutcome::Failed(FailureReason::ProtocolViolation(
                "DeepSeek-V4 expert operation received a non-routed-expert resource".into(),
            )));
        };
        let expected_key =
            ExpertKey::new(operation.request.model().get(), layer.get(), expert.get());
        if expected.key != expected_key
            || expected.slot.get() != operation.binding.slot.get()
            || u64::from(expected.generation.get()) != operation.key.destination_generation().get()
            || operation.expert.layer != usize::try_from(layer.get()).unwrap_or(usize::MAX)
            || operation.expert.expert != usize::try_from(expert.get()).unwrap_or(usize::MAX)
        {
            return Some(CompletionOutcome::Failed(FailureReason::ProtocolViolation(
                "prepared slot identity/generation diverged from the canonical operation".into(),
            )));
        }
        operation
            .read_plan
            .validate_source_identity()
            .err()
            .map(CompletionOutcome::Stale)
    }

    fn mark_source_stale(operation: &mut MaterializationOperation) {
        if operation.pending_terminal.is_none()
            && operation.read_plan.validate_source_identity().is_err()
        {
            operation.pending_terminal =
                Some(CompletionOutcome::Stale(StaleReason::SourceIdentityChanged));
        }
    }

    fn activate_physical_install(
        &self,
        operation: &MaterializationOperation,
    ) -> std::result::Result<ExpertInstallActivationOutcome, CompletionOutcome> {
        let prepared = operation
            .prepared
            .ok_or_else(|| CompletionOutcome::Failed(FailureReason::InstallationRejected))?;
        let mut shared = self.shared.lock();
        if shared.poisoned_layers.contains(&operation.expert.layer) {
            return Err(CompletionOutcome::Failed(
                FailureReason::InstallationRejected,
            ));
        }
        shared
            .residency
            .as_mut()
            .ok_or_else(|| CompletionOutcome::Failed(FailureReason::InstallationRejected))?
            .activate_install(prepared)
            .map_err(|error| CompletionOutcome::Failed(protocol_failure(error)))
    }

    fn install_frame(
        &mut self,
        operation_id: OperationId,
        operation: &mut MaterializationOperation,
        frame: CudaExpertFrame,
    ) -> std::result::Result<ValidatedResidencyBinding, CompletionOutcome> {
        if let Some(outcome) = Self::operation_identity_outcome(operation) {
            self.shared.lock().free_frames.push(frame);
            return Err(outcome);
        }
        let Some(prepared) = operation.prepared else {
            self.shared.lock().free_frames.push(frame);
            return Err(CompletionOutcome::Failed(
                FailureReason::InstallationRejected,
            ));
        };
        let binding = match ValidatedResidencyBinding::new(operation.key, operation.binding) {
            Ok(binding) => binding,
            Err(error) => {
                self.shared.lock().free_frames.push(frame);
                return Err(CompletionOutcome::Failed(FailureReason::ProtocolViolation(
                    error.to_string(),
                )));
            }
        };
        let expected = prepared.binding();
        let pointers = match self
            .ops
            .expert_slot_pointers(&frame.gate, &frame.up, &frame.down)
        {
            Ok(pointers) => pointers,
            Err(error) => {
                self.shared.lock().free_frames.push(frame);
                return Err(CompletionOutcome::Failed(protocol_failure(error)));
            }
        };

        let mut shared = self.shared.lock();
        if shared.poisoned_layers.contains(&operation.expert.layer) {
            shared.free_frames.push(frame);
            return Err(CompletionOutcome::Failed(
                FailureReason::InstallationRejected,
            ));
        }
        let eviction = (|| -> Result<Option<(ExpertId, ExpertSlotBinding, MaterializationKey)>> {
            let table = shared.tables.get(&operation.expert.layer).ok_or_else(|| {
                Error::Internal(format!(
                    "physical expert slot table missing layer {}",
                    operation.expert.layer
                ))
            })?;
            if let Some(evicted_key) = prepared.evicted_key() {
                let evicted =
                    ExpertId::new(evicted_key.layer as usize, evicted_key.expert as usize);
                if evicted.layer != operation.expert.layer {
                    return Err(Error::Internal(
                        "prepared eviction crossed physical expert layers".into(),
                    ));
                }
                let old = prepared.evicted_binding().ok_or_else(|| {
                    Error::Internal("prepared eviction lost transaction binding".into())
                })?;
                let physical = table.host().binding(evicted.expert).ok_or_else(|| {
                    Error::Internal("prepared eviction lost physical table binding".into())
                })?;
                let old_key = shared
                    .resident_keys
                    .get(&evicted_key)
                    .copied()
                    .ok_or_else(|| {
                        Error::Internal(
                            "prepared eviction lost exact resident MaterializationKey".into(),
                        )
                    })?;
                if old.slot != expected.slot
                    || old.generation.get().checked_add(1) != Some(expected.generation.get())
                    || physical.slot != i32::try_from(old.slot.get()).unwrap_or(-1)
                    || physical.generation != i32::try_from(old.generation.get()).unwrap_or(-1)
                    || old_key.destination_generation().get() != u64::from(old.generation.get())
                    || !shared.experts.contains_key(&evicted)
                {
                    return Err(Error::Execution(
                        "prepared eviction source/slot generation is stale".into(),
                    ));
                }
                Ok(Some((evicted, old, old_key)))
            } else {
                if table.host().binding(operation.expert.expert).is_some()
                    || shared.experts.contains_key(&operation.expert)
                {
                    return Err(Error::Execution(
                        "prepared empty slot already has a physical expert owner".into(),
                    ));
                }
                Ok(None)
            }
        })();
        let eviction = match eviction {
            Ok(eviction) => eviction,
            Err(error) => {
                shared.free_frames.push(frame);
                return Err(CompletionOutcome::Failed(protocol_failure(error)));
            }
        };

        if operation.read_plan.validate_source_identity().is_err() {
            shared.free_frames.push(frame);
            return Err(CompletionOutcome::Stale(StaleReason::SourceIdentityChanged));
        }
        if let Some((evicted, old, _)) = eviction
            && let Err(error) = self.ops.evict_expert_slot_binding(
                shared
                    .tables
                    .get_mut(&operation.expert.layer)
                    .ok_or_else(|| Error::Internal("physical expert slot table disappeared".into()))
                    .map_err(|error| CompletionOutcome::Failed(protocol_failure(error)))?,
                evicted.expert,
                old.slot.get(),
                old.generation.get(),
            )
        {
            shared.free_frames.push(frame);
            return Err(CompletionOutcome::Failed(protocol_failure(error)));
        }
        if let Err(error) = self.ops.install_expert_slot_at(
            shared
                .tables
                .get_mut(&operation.expert.layer)
                .ok_or_else(|| Error::Internal("physical expert slot table disappeared".into()))
                .map_err(|error| CompletionOutcome::Failed(protocol_failure(error)))?,
            operation.expert.expert,
            expected.slot.get(),
            expected.generation.get(),
            pointers,
        ) {
            if eviction.is_some() {
                shared.poisoned_layers.insert(operation.expert.layer);
            }
            if self.ops.sync_stream().is_ok() {
                shared.free_frames.push(frame);
            }
            return Err(CompletionOutcome::Failed(protocol_failure(error)));
        }
        if let Err(error) = self.ops.sync_stream() {
            shared.poisoned_layers.insert(operation.expert.layer);
            shared.experts.insert(operation.expert, frame);
            return Err(CompletionOutcome::Failed(protocol_failure(error)));
        }

        let physical_matches = shared
            .tables
            .get(&operation.expert.layer)
            .and_then(|table| table.host().binding(operation.expert.expert))
            .is_some_and(|physical| {
                physical.slot == i32::try_from(expected.slot.get()).unwrap_or(-1)
                    && physical.generation == i32::try_from(expected.generation.get()).unwrap_or(-1)
            });
        if !physical_matches || operation.read_plan.validate_source_identity().is_err() {
            shared.poisoned_layers.insert(operation.expert.layer);
            shared.experts.insert(operation.expert, frame);
            return Err(if physical_matches {
                CompletionOutcome::Stale(StaleReason::SourceIdentityChanged)
            } else {
                CompletionOutcome::Failed(FailureReason::ProtocolViolation(
                    "physical slot generation changed before controller publication".into(),
                ))
            });
        }

        let grant = match shared
            .residency
            .as_mut()
            .ok_or_else(|| Error::Execution("physical residency control is not installed".into()))
            .and_then(|residency| residency.publish_install(prepared))
        {
            Ok(grant) => {
                operation.prepared = None;
                grant
            }
            Err(error) => {
                shared.poisoned_layers.insert(operation.expert.layer);
                shared.experts.insert(operation.expert, frame);
                return Err(CompletionOutcome::Failed(protocol_failure(error)));
            }
        };
        let controller_matches = shared
            .residency
            .as_ref()
            .and_then(|residency| residency.binding(expected.key).ok().flatten())
            == Some(expected);
        let physical_matches = shared
            .tables
            .get(&operation.expert.layer)
            .and_then(|table| table.host().binding(operation.expert.expert))
            .is_some_and(|physical| {
                physical.slot == i32::try_from(expected.slot.get()).unwrap_or(-1)
                    && physical.generation == i32::try_from(expected.generation.get()).unwrap_or(-1)
            });
        let lease = grant.lease();
        let source_is_current = operation.read_plan.validate_source_identity().is_ok();
        if grant.binding() != expected
            || grant.reason() != ExpertInstallReason::Selected
            || lease.is_none()
            || !controller_matches
            || !physical_matches
            || !source_is_current
        {
            if let Some(lease) = lease
                && let Some(residency) = shared.residency.as_mut()
            {
                let _ = residency.release(lease);
            }
            shared.poisoned_layers.insert(operation.expert.layer);
            shared.experts.insert(operation.expert, frame);
            return Err(if source_is_current {
                CompletionOutcome::Failed(FailureReason::ProtocolViolation(
                    "physical/controller slot identity diverged after publication".into(),
                ))
            } else {
                CompletionOutcome::Stale(StaleReason::SourceIdentityChanged)
            });
        }
        let Some(lease) = lease else {
            shared.poisoned_layers.insert(operation.expert.layer);
            shared.experts.insert(operation.expert, frame);
            return Err(CompletionOutcome::Failed(
                FailureReason::InstallationRejected,
            ));
        };

        if let Some((evicted, _, old_key)) = eviction {
            if let Some(old_frame) = shared.experts.remove(&evicted) {
                shared.free_frames.push(old_frame);
            }
            shared.resident_keys.retain(|_, key| *key != old_key);
        }
        shared.experts.insert(operation.expert, frame);
        shared.resident_keys.insert(expected.key, operation.key);
        drop(shared);
        let publication =
            match self
                .selected_lease_ownership
                .publish(operation.key, operation_id, lease)
            {
                Ok(publication) => publication,
                Err(reason) => {
                    if let Some(residency) = self.shared.lock().residency.as_mut() {
                        let _ = residency.release(lease);
                    }
                    return Err(CompletionOutcome::Failed(reason));
                }
            };
        if publication == SelectedLeasePublication::ReleaseImmediately {
            self.release_execution_lease(operation.key)
                .map_err(CompletionOutcome::Failed)?;
        }
        Ok(binding)
    }

    fn cleanup_operation(
        &mut self,
        operation_id: OperationId,
        mut operation: MaterializationOperation,
        state: Option<CudaMaterializationOperationState>,
        reader_consumed: bool,
    ) -> Result<()> {
        let mut first_error = None;
        match state {
            Some(MaterializationOperationState::Reserved(ticket)) => {
                record_first_error(
                    &mut first_error,
                    self.reader.detach_load_source_pinned(ticket),
                );
            }
            Some(MaterializationOperationState::ReadSubmitted(ticket)) => {
                if !reader_consumed {
                    record_first_error(
                        &mut first_error,
                        self.reader.detach_load_source_pinned(ticket),
                    );
                }
            }
            Some(MaterializationOperationState::UploadSubmitted(ticket)) => {
                match ticket.drain_into_frame() {
                    Ok(frame) => self.recycle_frame(frame),
                    Err(error) => first_error = Some(error),
                }
            }
            Some(MaterializationOperationState::UploadReady(frame))
            | Some(MaterializationOperationState::Installing(frame)) => {
                self.recycle_frame(frame);
            }
            Some(MaterializationOperationState::HostReady(bundle)) => drop(bundle),
            None => {}
        }
        if let Some(prepared) = operation.prepared.take() {
            record_first_error(&mut first_error, self.cancel_prepared(prepared));
        }
        if self
            .selected_lease_ownership
            .active_owned_by(operation.key, operation_id)
        {
            record_first_error(
                &mut first_error,
                self.release_execution_lease(operation.key)
                    .map_err(|reason| {
                        Error::Execution(format!(
                            "terminal selected-expert lease release failed: {reason:?}"
                        ))
                    }),
            );
        }
        self.selected_lease_ownership
            .cancel_operation(operation.key, operation_id);
        self.remove_request_key(operation.request, operation.key);
        first_error.map_or(Ok(()), Err)
    }

    fn finish_terminal_operation(
        &mut self,
        operation_id: OperationId,
        operation: MaterializationOperation,
        state: Option<CudaMaterializationOperationState>,
        completion: Option<(LoadStage, CompletionOutcome, u64)>,
        reader_consumed: bool,
    ) -> Result<()> {
        self.remove_operation_order(operation_id);
        let first_terminal = self.terminal_operations.insert(operation_id);
        let key = operation.key;
        let cleanup = self.cleanup_operation(operation_id, operation, state, reader_consumed);
        if first_terminal && let Some((stage, mut outcome, bytes)) = completion {
            if let Err(error) = &cleanup {
                outcome = CompletionOutcome::Failed(protocol_failure(Error::Internal(format!(
                    "terminal resource cleanup failed: {error}"
                ))));
            }
            self.emit(operation_id, key, stage, outcome, bytes);
        }
        cleanup
    }

    fn finish_successful_install(
        &mut self,
        operation_id: OperationId,
        operation: MaterializationOperation,
    ) {
        self.remove_operation_order(operation_id);
        if self.terminal_operations.insert(operation_id) {
            self.emit(
                operation_id,
                operation.key,
                LoadStage::Installing,
                CompletionOutcome::Succeeded,
                operation.resource_plan.demand.device_install_bytes,
            );
        }
    }

    fn keep_operation(
        &mut self,
        operation_id: OperationId,
        mut operation: MaterializationOperation,
        state: CudaMaterializationOperationState,
    ) {
        operation.state = Some(state);
        self.operations.insert(operation_id, operation);
        self.operation_order.push_back(operation_id);
    }

    fn progress_operation(
        &mut self,
        operation_id: OperationId,
        mut operation: MaterializationOperation,
    ) {
        let Some(state) = operation.state.take() else {
            let _ = self.finish_terminal_operation(
                operation_id,
                operation,
                None,
                Some((
                    LoadStage::Installing,
                    CompletionOutcome::Failed(FailureReason::ProtocolViolation(
                        "active physical operation had no resource owner".into(),
                    )),
                    0,
                )),
                false,
            );
            return;
        };
        match state {
            MaterializationOperationState::Reserved(ticket) => {
                self.keep_operation(
                    operation_id,
                    operation,
                    MaterializationOperationState::Reserved(ticket),
                );
            }
            MaterializationOperationState::HostReady(bundle) => {
                self.keep_operation(
                    operation_id,
                    operation,
                    MaterializationOperationState::HostReady(bundle),
                );
            }
            MaterializationOperationState::UploadReady(frame) => {
                self.keep_operation(
                    operation_id,
                    operation,
                    MaterializationOperationState::UploadReady(frame),
                );
            }
            MaterializationOperationState::ReadSubmitted(ticket) => {
                Self::mark_source_stale(&mut operation);
                if matches!(
                    operation.pending_terminal,
                    Some(CompletionOutcome::Stale(_))
                ) {
                    let _ = self.reader.cancel_load_source_pinned(ticket);
                }
                match self.reader.poll_load_source_pinned(ticket, 64) {
                    Ok(PinnedExpertReadPoll::Pending) => self.keep_operation(
                        operation_id,
                        operation,
                        MaterializationOperationState::ReadSubmitted(ticket),
                    ),
                    Ok(PinnedExpertReadPoll::Ready(payload)) => {
                        if let Some(outcome) = operation.pending_terminal.take() {
                            let _ = self.finish_terminal_operation(
                                operation_id,
                                operation,
                                None,
                                Some((LoadStage::ReadSubmitted, outcome, 0)),
                                true,
                            );
                            return;
                        }
                        Self::mark_source_stale(&mut operation);
                        if let Some(outcome) = operation.pending_terminal.take() {
                            let _ = self.finish_terminal_operation(
                                operation_id,
                                operation,
                                None,
                                Some((LoadStage::ReadSubmitted, outcome, 0)),
                                true,
                            );
                            return;
                        }
                        match PinnedExpertBundle::from_payload(payload) {
                            Ok(bundle)
                                if bundle.expert == operation.expert
                                    && bundle.bytes == operation.resource_plan.demand.h2d_bytes =>
                            {
                                self.emit(
                                    operation_id,
                                    operation.key,
                                    LoadStage::ReadSubmitted,
                                    CompletionOutcome::Succeeded,
                                    operation.resource_plan.demand.storage_read_bytes,
                                );
                                self.keep_operation(
                                    operation_id,
                                    operation,
                                    MaterializationOperationState::HostReady(bundle),
                                );
                            }
                            Ok(_) => {
                                let _ = self.finish_terminal_operation(
                                    operation_id,
                                    operation,
                                    None,
                                    Some((
                                        LoadStage::ReadSubmitted,
                                        CompletionOutcome::Failed(
                                            FailureReason::ProtocolViolation(
                                                "pinned payload expert identity or byte count mismatch"
                                                    .into(),
                                            ),
                                        ),
                                        0,
                                    )),
                                    true,
                                );
                            }
                            Err(error) => {
                                let _ = self.finish_terminal_operation(
                                    operation_id,
                                    operation,
                                    None,
                                    Some((
                                        LoadStage::ReadSubmitted,
                                        CompletionOutcome::Failed(protocol_failure(error)),
                                        0,
                                    )),
                                    true,
                                );
                            }
                        }
                    }
                    Ok(PinnedExpertReadPoll::Cancelled) => {
                        let outcome = operation.pending_terminal.take().unwrap_or(
                            CompletionOutcome::Cancelled(CancellationReason::LastWaiterDetached),
                        );
                        let _ = self.finish_terminal_operation(
                            operation_id,
                            operation,
                            None,
                            Some((LoadStage::ReadSubmitted, outcome, 0)),
                            true,
                        );
                    }
                    Ok(PinnedExpertReadPoll::Failed(error)) => {
                        Self::mark_source_stale(&mut operation);
                        let outcome = operation
                            .pending_terminal
                            .take()
                            .unwrap_or_else(|| CompletionOutcome::Failed(protocol_failure(error)));
                        let _ = self.finish_terminal_operation(
                            operation_id,
                            operation,
                            None,
                            Some((LoadStage::ReadSubmitted, outcome, 0)),
                            true,
                        );
                    }
                    Err(error) => {
                        Self::mark_source_stale(&mut operation);
                        let outcome = operation
                            .pending_terminal
                            .take()
                            .unwrap_or_else(|| CompletionOutcome::Failed(protocol_failure(error)));
                        let _ = self.finish_terminal_operation(
                            operation_id,
                            operation,
                            Some(MaterializationOperationState::ReadSubmitted(ticket)),
                            Some((LoadStage::ReadSubmitted, outcome, 0)),
                            false,
                        );
                    }
                }
            }
            MaterializationOperationState::UploadSubmitted(ticket) => {
                Self::mark_source_stale(&mut operation);
                match ticket.is_complete() {
                    Ok(false) => self.keep_operation(
                        operation_id,
                        operation,
                        MaterializationOperationState::UploadSubmitted(ticket),
                    ),
                    completion => {
                        let query_error = completion.err();
                        match ticket.drain_into_frame() {
                            Ok(frame) => {
                                Self::mark_source_stale(&mut operation);
                                if let Some(outcome) = operation.pending_terminal.take() {
                                    let _ = self.finish_terminal_operation(
                                        operation_id,
                                        operation,
                                        Some(MaterializationOperationState::UploadReady(frame)),
                                        Some((LoadStage::UploadSubmitted, outcome, 0)),
                                        true,
                                    );
                                } else if let Some(error) = query_error {
                                    let _ = self.finish_terminal_operation(
                                        operation_id,
                                        operation,
                                        Some(MaterializationOperationState::UploadReady(frame)),
                                        Some((
                                            LoadStage::UploadSubmitted,
                                            CompletionOutcome::Failed(protocol_failure(error)),
                                            0,
                                        )),
                                        true,
                                    );
                                } else if let Some(outcome) =
                                    Self::operation_identity_outcome(&operation)
                                {
                                    let _ = self.finish_terminal_operation(
                                        operation_id,
                                        operation,
                                        Some(MaterializationOperationState::UploadReady(frame)),
                                        Some((LoadStage::UploadSubmitted, outcome, 0)),
                                        true,
                                    );
                                } else {
                                    self.emit(
                                        operation_id,
                                        operation.key,
                                        LoadStage::UploadSubmitted,
                                        CompletionOutcome::Succeeded,
                                        operation.resource_plan.demand.h2d_bytes,
                                    );
                                    self.keep_operation(
                                        operation_id,
                                        operation,
                                        MaterializationOperationState::UploadReady(frame),
                                    );
                                }
                            }
                            Err(error) => {
                                let outcome =
                                    operation.pending_terminal.take().unwrap_or_else(|| {
                                        CompletionOutcome::Failed(protocol_failure(error))
                                    });
                                let _ = self.finish_terminal_operation(
                                    operation_id,
                                    operation,
                                    None,
                                    Some((LoadStage::UploadSubmitted, outcome, 0)),
                                    true,
                                );
                            }
                        }
                    }
                }
            }
            MaterializationOperationState::Installing(frame) => {
                Self::mark_source_stale(&mut operation);
                if let Some(outcome) = operation.pending_terminal.take() {
                    let _ = self.finish_terminal_operation(
                        operation_id,
                        operation,
                        Some(MaterializationOperationState::Installing(frame)),
                        Some((LoadStage::Installing, outcome, 0)),
                        true,
                    );
                    return;
                }
                match self.activate_physical_install(&operation) {
                    Ok(ExpertInstallActivationOutcome::BlockedByLeases) => {
                        self.keep_operation(
                            operation_id,
                            operation,
                            MaterializationOperationState::Installing(frame),
                        );
                        return;
                    }
                    Ok(ExpertInstallActivationOutcome::Activated) => {}
                    Err(outcome) => {
                        let _ = self.finish_terminal_operation(
                            operation_id,
                            operation,
                            Some(MaterializationOperationState::Installing(frame)),
                            Some((LoadStage::Installing, outcome, 0)),
                            true,
                        );
                        return;
                    }
                }
                match self.install_frame(operation_id, &mut operation, frame) {
                    Ok(_) => self.finish_successful_install(operation_id, operation),
                    Err(outcome) => {
                        let _ = self.finish_terminal_operation(
                            operation_id,
                            operation,
                            None,
                            Some((LoadStage::Installing, outcome, 0)),
                            true,
                        );
                    }
                }
            }
        }
    }

    fn progress_one(&mut self) -> Option<CompletionEvent> {
        if let Some(completion) = self.completions.pop_front() {
            return Some(completion);
        }
        let (operation_id, operation) = loop {
            let operation_id = self.operation_order.pop_front()?;
            if let Some(operation) = self.operations.remove(&operation_id) {
                break (operation_id, operation);
            }
        };
        self.progress_operation(operation_id, operation);
        self.completions.pop_front()
    }
}

impl MaterializationProvider for DeepSeekV4CudaMaterializationProvider {
    fn placement(&self) -> MaterializationPlacement {
        self.placement
    }

    fn resource_topology(&self) -> Result<PhysicalMaterializationTopology> {
        Ok(self.topology)
    }

    fn prepare(
        &mut self,
        request: MaterializationRequest,
    ) -> std::result::Result<MaterializationPreparation, FailureReason> {
        let (expert, read_plan, install_source) = self.source_for_request(request)?;
        read_plan.validate_source_identity().map_err(|_| {
            FailureReason::ProtocolViolation("checkpoint source identity is already stale".into())
        })?;
        if let Some(existing) = self.request_keys.get(&request).copied() {
            if let Some(reservation) = self.reservations.get(&existing) {
                return Ok(MaterializationPreparation::Transfer(
                    MaterializationTransfer::new(
                        existing,
                        reservation.binding,
                        reservation.evicted,
                    )
                    .map_err(protocol_failure)?,
                ));
            }
            if let Some(active) = self
                .operations
                .values()
                .find(|operation| operation.key == existing)
            {
                return Ok(MaterializationPreparation::Transfer(
                    MaterializationTransfer::new(existing, active.binding, active.evicted)
                        .map_err(protocol_failure)?,
                ));
            }
            let (layer, expert_index) = request.routed_expert_coordinates().ok_or_else(|| {
                FailureReason::ProtocolViolation(
                    "DeepSeek-V4 routed-expert preparation received a non-expert resource".into(),
                )
            })?;
            let expert_key = ExpertKey::new(request.model().get(), layer.get(), expert_index.get());
            if self.selected_lease_ownership.contains(existing) {
                return self.prepared(existing);
            }
            let (raw, lease) = {
                let mut shared = self.shared.lock();
                let Some(grant) = shared
                    .residency
                    .as_mut()
                    .ok_or(FailureReason::InstallationRejected)?
                    .acquire_selected(expert_key)
                    .map_err(protocol_failure)?
                else {
                    drop(shared);
                    self.request_keys.remove(&request);
                    return self.prepare(request);
                };
                let lease = grant.lease().ok_or_else(|| {
                    FailureReason::ProtocolViolation(
                        "selected resident expert did not return a lease".into(),
                    )
                })?;
                let raw = match self.protocol_binding(request, grant.binding()) {
                    Ok(raw) => raw,
                    Err(reason) => {
                        if let Some(residency) = shared.residency.as_mut() {
                            let _ = residency.release(lease);
                        }
                        return Err(reason);
                    }
                };
                let physical = shared
                    .tables
                    .get(&expert.layer)
                    .and_then(|table| table.host().binding(expert.expert));
                let valid = grant.reason() == ExpertInstallReason::Selected
                    && read_plan.validate_source_identity().is_ok()
                    && shared.experts.contains_key(&expert)
                    && shared.resident_keys.get(&expert_key) == Some(&existing)
                    && physical.is_some_and(|physical| {
                        physical.slot == i32::try_from(raw.slot.get()).unwrap_or(-1)
                            && physical.generation
                                == i32::try_from(grant.binding().generation.get()).unwrap_or(-1)
                    });
                if !valid {
                    if let Some(residency) = shared.residency.as_mut() {
                        let _ = residency.release(lease);
                    }
                    return Err(FailureReason::ProtocolViolation(
                        "controller resident binding/source is not physically current".into(),
                    ));
                }
                (raw, lease)
            };
            let validated = match ValidatedResidencyBinding::new(existing, raw) {
                Ok(validated) => validated,
                Err(error) => {
                    if let Some(residency) = self.shared.lock().residency.as_mut() {
                        let _ = residency.release(lease);
                    }
                    return Err(FailureReason::ProtocolViolation(error.to_string()));
                }
            };
            if let Err(reason) = self.selected_lease_ownership.begin_active(existing, lease) {
                if let Some(residency) = self.shared.lock().residency.as_mut() {
                    let _ = residency.release(lease);
                }
                return Err(reason);
            }
            return MaterializationResident::new(existing, validated.binding())
                .map(MaterializationPreparation::Resident)
                .map_err(protocol_failure);
        }

        let (layer, expert_index) = request.routed_expert_coordinates().ok_or_else(|| {
            FailureReason::ProtocolViolation(
                "DeepSeek-V4 routed-expert preparation received a non-expert resource".into(),
            )
        })?;
        let expert_key = ExpertKey::new(request.model().get(), layer.get(), expert_index.get());
        let pinned_plan = self
            .reader
            .plan_checkpoint_source_pinned(expert, &read_plan, &install_source)
            .map_err(protocol_failure)?
            .ok_or(FailureReason::StorageUnavailable)?;
        let resource_plan = MaterializationResourcePlan::new(
            pinned_plan.demand(),
            pinned_plan.demand().device_install_bytes,
        )
        .map_err(protocol_failure)?;
        let mut shared = self.shared.lock();
        if shared.poisoned_layers.contains(&expert.layer) {
            return Err(FailureReason::InstallationRejected);
        }
        let outcome = shared
            .residency
            .as_mut()
            .ok_or(FailureReason::InstallationRejected)?
            .prepare_install(ExpertInstallIntent::selected(expert_key))
            .map_err(protocol_failure)?;
        match outcome {
            ExpertInstallPrepareOutcome::Resident(grant) => {
                let lease = grant.lease().ok_or_else(|| {
                    FailureReason::ProtocolViolation(
                        "selected resident expert did not return a lease".into(),
                    )
                })?;
                let raw = match self.protocol_binding(request, grant.binding()) {
                    Ok(raw) => raw,
                    Err(reason) => {
                        if let Some(residency) = shared.residency.as_mut() {
                            let _ = residency.release(lease);
                        }
                        return Err(reason);
                    }
                };
                let key = match request
                    .materialization_key(raw.generation)
                    .map_err(protocol_failure)
                {
                    Ok(key) => key,
                    Err(reason) => {
                        if let Some(residency) = shared.residency.as_mut() {
                            let _ = residency.release(lease);
                        }
                        return Err(reason);
                    }
                };
                let physical = shared
                    .tables
                    .get(&expert.layer)
                    .and_then(|table| table.host().binding(expert.expert));
                if grant.reason() != ExpertInstallReason::Selected
                    || shared.resident_keys.get(&expert_key) != Some(&key)
                    || !shared.experts.contains_key(&expert)
                    || !physical.is_some_and(|physical| {
                        physical.slot == i32::try_from(raw.slot.get()).unwrap_or(-1)
                            && physical.generation
                                == i32::try_from(grant.binding().generation.get()).unwrap_or(-1)
                    })
                    || read_plan.validate_source_identity().is_err()
                {
                    if let Some(residency) = shared.residency.as_mut() {
                        let _ = residency.release(lease);
                    }
                    return Err(FailureReason::ProtocolViolation(
                        "controller resident binding/source is not physically current".into(),
                    ));
                }
                let validated = match ValidatedResidencyBinding::new(key, raw) {
                    Ok(validated) => validated,
                    Err(error) => {
                        if let Some(residency) = shared.residency.as_mut() {
                            let _ = residency.release(lease);
                        }
                        return Err(FailureReason::ProtocolViolation(error.to_string()));
                    }
                };
                drop(shared);
                if let Err(reason) = self.selected_lease_ownership.begin_active(key, lease) {
                    if let Some(residency) = self.shared.lock().residency.as_mut() {
                        let _ = residency.release(lease);
                    }
                    return Err(reason);
                }
                self.request_keys.insert(request, key);
                self.resource_plans.insert(key, resource_plan);
                MaterializationResident::new(key, validated.binding())
                    .map(MaterializationPreparation::Resident)
                    .map_err(protocol_failure)
            }
            ExpertInstallPrepareOutcome::Prepared(prepared) => {
                let raw = match self.protocol_binding(request, prepared.binding()) {
                    Ok(raw) => raw,
                    Err(reason) => {
                        if let Some(residency) = shared.residency.as_mut() {
                            let _ = residency.cancel_install(prepared);
                        }
                        return Err(reason);
                    }
                };
                let key = match request
                    .materialization_key(raw.generation)
                    .map_err(protocol_failure)
                {
                    Ok(key) => key,
                    Err(reason) => {
                        if let Some(residency) = shared.residency.as_mut() {
                            let _ = residency.cancel_install(prepared);
                        }
                        return Err(reason);
                    }
                };
                let evicted = prepared
                    .evicted_key()
                    .and_then(|evicted| shared.resident_keys.get(&evicted).copied());
                let descriptor = match MaterializationTransfer::new(key, raw, evicted)
                    .map_err(protocol_failure)
                {
                    Ok(descriptor) => descriptor,
                    Err(reason) => {
                        if let Some(residency) = shared.residency.as_mut() {
                            let _ = residency.cancel_install(prepared);
                        }
                        return Err(reason);
                    }
                };
                drop(shared);
                if let Err(reason) = self.selected_lease_ownership.begin_pending(key) {
                    let _ = self.cancel_prepared(prepared);
                    return Err(reason);
                }
                self.request_keys.insert(request, key);
                self.resource_plans.insert(key, resource_plan);
                self.reservations.insert(
                    key,
                    SlotReservation {
                        request,
                        expert,
                        read_plan,
                        pinned_plan,
                        resource_plan,
                        prepared,
                        binding: raw,
                        evicted,
                    },
                );
                Ok(MaterializationPreparation::Transfer(descriptor))
            }
            ExpertInstallPrepareOutcome::CapacityAllLeased => {
                Err(FailureReason::InstallationRejected)
            }
        }
    }

    fn prepared(
        &self,
        key: MaterializationKey,
    ) -> std::result::Result<MaterializationPreparation, FailureReason> {
        let (request, expert, read_plan, _) = self.source_for_key(key)?;
        read_plan.validate_source_identity().map_err(|_| {
            FailureReason::ProtocolViolation("checkpoint source identity is stale".into())
        })?;
        if let Some(reservation) = self.reservations.get(&key) {
            return MaterializationTransfer::new(key, reservation.binding, reservation.evicted)
                .map(MaterializationPreparation::Transfer)
                .map_err(protocol_failure);
        }
        if let Some(active) = self
            .operations
            .values()
            .find(|operation| operation.key == key)
        {
            return MaterializationTransfer::new(key, active.binding, active.evicted)
                .map(MaterializationPreparation::Transfer)
                .map_err(protocol_failure);
        }
        if !self.selected_lease_ownership.contains(key) {
            return Err(FailureReason::ProtocolViolation(
                "materialization key has no provider-owned preparation".into(),
            ));
        }
        let shared = self.shared.lock();
        let (layer, expert_index) = request.routed_expert_coordinates().ok_or_else(|| {
            FailureReason::ProtocolViolation(
                "DeepSeek-V4 provider received a non-routed-expert resource".into(),
            )
        })?;
        let expert_key = ExpertKey::new(request.model().get(), layer.get(), expert_index.get());
        if shared.resident_keys.get(&expert_key) != Some(&key)
            || !shared.experts.contains_key(&expert)
        {
            return Err(FailureReason::ProtocolViolation(
                "prepared resident expert is not physically published".into(),
            ));
        }
        let physical = shared
            .tables
            .get(&expert.layer)
            .and_then(|table| table.host().binding(expert.expert))
            .ok_or_else(|| {
                FailureReason::ProtocolViolation(
                    "prepared resident expert has no physical table binding".into(),
                )
            })?;
        let slot = u32::try_from(physical.slot).map_err(|_| {
            FailureReason::ProtocolViolation("resident expert slot exceeds protocol ABI".into())
        })?;
        let generation = u64::try_from(physical.generation).map_err(|_| {
            FailureReason::ProtocolViolation(
                "resident expert generation exceeds protocol ABI".into(),
            )
        })?;
        let binding = ResidencyBinding::new(
            request.model(),
            request.resource(),
            request.backend(),
            request.device(),
            DestinationSlotId::new(slot),
            DestinationGeneration::new(generation),
        );
        MaterializationResident::new(key, binding)
            .map(MaterializationPreparation::Resident)
            .map_err(protocol_failure)
    }

    fn discard_preparation(
        &mut self,
        key: MaterializationKey,
    ) -> std::result::Result<(), FailureReason> {
        if self
            .operations
            .values()
            .any(|operation| operation.key == key)
        {
            return Err(FailureReason::ProtocolViolation(
                "cannot discard a preparation claimed by an active operation".into(),
            ));
        }
        if let Some(reservation) = self.reservations.remove(&key) {
            self.cancel_prepared(reservation.prepared)
                .map_err(protocol_failure)?;
            self.selected_lease_ownership.cancel_unbound(key);
            self.remove_key(key);
            return Ok(());
        }
        if self.selected_lease_ownership.contains(key) {
            self.release_execution_lease(key)?;
            self.remove_key(key);
            return Ok(());
        }
        Err(FailureReason::ProtocolViolation(
            "cannot discard an unknown materialization preparation".into(),
        ))
    }

    fn materialization_plan(
        &self,
        key: MaterializationKey,
    ) -> std::result::Result<MaterializationResourcePlan, FailureReason> {
        self.source_for_key(key)?;
        self.resource_plans.get(&key).copied().ok_or_else(|| {
            FailureReason::ProtocolViolation(
                "DeepSeek-V4 materialization key has no frozen physical resource plan".into(),
            )
        })
    }

    fn release_execution_lease(
        &mut self,
        key: MaterializationKey,
    ) -> std::result::Result<(), FailureReason> {
        let SelectedLeaseRelease::Active { lease, operation } =
            self.selected_lease_ownership.request_release(key)
        else {
            return Ok(());
        };
        let release = self
            .shared
            .lock()
            .residency
            .as_mut()
            .ok_or(FailureReason::InstallationRejected)
            .and_then(|residency| residency.release(lease).map_err(protocol_failure));
        if let Err(error) = release {
            self.selected_lease_ownership
                .restore_active(key, lease, operation);
            return Err(error);
        }
        Ok(())
    }

    fn reserve(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        plan: MaterializationResourcePlan,
    ) -> std::result::Result<PhysicalMaterializationOperationReservation, FailureReason> {
        if operation.is_zero() || self.seen_operations.contains(&operation) {
            return Err(FailureReason::ProtocolViolation(
                "duplicate or zero physical materialization operation".into(),
            ));
        }
        if self.operations.values().any(|active| active.key == key) {
            return Err(FailureReason::ProtocolViolation(
                "prepared MaterializationKey already has a physical operation owner".into(),
            ));
        }
        if self.materialization_plan(key)? != plan {
            return Err(FailureReason::ProtocolViolation(
                "physical materialization resource plan expectation mismatch".into(),
            ));
        }
        let reservation = self.reservations.remove(&key).ok_or_else(|| {
            FailureReason::ProtocolViolation(
                "MaterializationKey has no prepared slot reservation".into(),
            )
        })?;
        if reservation.read_plan.validate_source_identity().is_err() {
            return Err(self.rollback_slot_reservation(
                key,
                reservation.request,
                reservation.prepared,
                FailureReason::ProtocolViolation(
                    "checkpoint source identity changed before reserve".into(),
                ),
            ));
        }
        if reservation.resource_plan != plan {
            return Err(self.rollback_slot_reservation(
                key,
                reservation.request,
                reservation.prepared,
                FailureReason::ProtocolViolation(
                    "prepared slot resource plan diverged from the canonical plan".into(),
                ),
            ));
        }
        let read =
            match self
                .reader
                .reserve_load_source_pinned(reservation.pinned_plan, operation, key)
            {
                Ok(read) => read,
                Err(error) => {
                    return Err(self.rollback_slot_reservation(
                        key,
                        reservation.request,
                        reservation.prepared,
                        protocol_failure(error),
                    ));
                }
            };
        let fence = UploadFenceContract::new(
            operation,
            FenceId::new(operation.get()),
            key.destination_generation(),
        );
        let descriptor = match PhysicalMaterializationOperationReservation::new(
            key,
            reservation.binding,
            read.slabs,
            fence,
        ) {
            Ok(descriptor) => descriptor,
            Err(error) => {
                let primary = protocol_failure(error);
                let cleanup = self.reader.detach_load_source_pinned(read.ticket);
                let reason = self.rollback_slot_reservation(
                    key,
                    reservation.request,
                    reservation.prepared,
                    primary,
                );
                return Err(match cleanup {
                    Ok(()) => reason,
                    Err(error) => FailureReason::ProtocolViolation(format!(
                        "physical reservation rollback failed ({reason:?}); read reservation detach also failed ({error})"
                    )),
                });
            }
        };
        if let Err(reason) = self.selected_lease_ownership.bind_operation(key, operation) {
            let cleanup = self.reader.detach_load_source_pinned(read.ticket);
            let reason = self.rollback_slot_reservation(
                key,
                reservation.request,
                reservation.prepared,
                reason,
            );
            return Err(match cleanup {
                Ok(()) => reason,
                Err(error) => FailureReason::ProtocolViolation(format!(
                    "physical lease bind rollback failed ({reason:?}); read reservation detach also failed ({error})"
                )),
            });
        }
        self.operations.insert(
            operation,
            MaterializationOperation {
                key,
                request: reservation.request,
                expert: reservation.expert,
                read_plan: reservation.read_plan,
                resource_plan: reservation.resource_plan,
                prepared: Some(reservation.prepared),
                binding: reservation.binding,
                evicted: reservation.evicted,
                pending_terminal: None,
                state: Some(MaterializationOperationState::Reserved(read.ticket)),
            },
        );
        self.seen_operations.insert(operation);
        self.operation_order.push_back(operation);
        Ok(descriptor)
    }

    fn submit_read(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &PhysicalMaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> std::result::Result<(), FailureReason> {
        let mut active = self
            .operations
            .remove(&operation)
            .ok_or_else(|| FailureReason::ProtocolViolation("unknown read operation".into()))?;
        let state = active.state.take();
        if let Err(reason) =
            validate_operation_reservation(&active, operation, key, reservation, plan)
        {
            let _ = self.finish_terminal_operation(operation, active, state, None, false);
            return Err(reason);
        }
        let Some(MaterializationOperationState::Reserved(ticket)) = state else {
            let _ = self.finish_terminal_operation(operation, active, state, None, false);
            return Err(FailureReason::ReadRejected);
        };
        if let Some(outcome) = Self::operation_identity_outcome(&active) {
            let _ = self.finish_terminal_operation(
                operation,
                active,
                Some(MaterializationOperationState::Reserved(ticket)),
                Some((LoadStage::ReadSubmitted, outcome, 0)),
                false,
            );
            return Ok(());
        }
        if let Err(error) = self.reader.submit_reserved_load_source_pinned(ticket) {
            let reason = protocol_failure(error);
            let _ = self.finish_terminal_operation(
                operation,
                active,
                Some(MaterializationOperationState::Reserved(ticket)),
                None,
                false,
            );
            return Err(reason);
        }
        active.state = Some(MaterializationOperationState::ReadSubmitted(ticket));
        self.operations.insert(operation, active);
        Ok(())
    }

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &PhysicalMaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> std::result::Result<(), FailureReason> {
        let mut active = self
            .operations
            .remove(&operation)
            .ok_or_else(|| FailureReason::ProtocolViolation("unknown upload operation".into()))?;
        let state = active.state.take();
        if let Err(reason) =
            validate_operation_reservation(&active, operation, key, reservation, plan)
        {
            let _ = self.finish_terminal_operation(operation, active, state, None, true);
            return Err(reason);
        }
        let Some(MaterializationOperationState::HostReady(bundle)) = state else {
            let _ = self.finish_terminal_operation(operation, active, state, None, true);
            return Err(FailureReason::UploadRejected);
        };
        if let Some(outcome) = Self::operation_identity_outcome(&active) {
            let _ = self.finish_terminal_operation(
                operation,
                active,
                Some(MaterializationOperationState::HostReady(bundle)),
                Some((LoadStage::UploadSubmitted, outcome, 0)),
                true,
            );
            return Ok(());
        }
        match self.submit_bundle_upload(bundle) {
            Ok(ticket) => {
                active.state = Some(MaterializationOperationState::UploadSubmitted(ticket));
                self.operations.insert(operation, active);
                Ok(())
            }
            Err(error) => {
                let reason = protocol_failure(error);
                let _ = self.finish_terminal_operation(operation, active, None, None, true);
                Err(reason)
            }
        }
    }

    fn poll_install(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &PhysicalMaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> std::result::Result<(), FailureReason> {
        let mut active = self
            .operations
            .remove(&operation)
            .ok_or_else(|| FailureReason::ProtocolViolation("unknown install operation".into()))?;
        let state = active.state.take();
        if let Err(reason) =
            validate_operation_reservation(&active, operation, key, reservation, plan)
        {
            let _ = self.finish_terminal_operation(operation, active, state, None, true);
            return Err(reason);
        }
        let Some(MaterializationOperationState::UploadReady(frame)) = state else {
            let _ = self.finish_terminal_operation(operation, active, state, None, true);
            return Err(FailureReason::InstallationRejected);
        };
        if let Some(outcome) = Self::operation_identity_outcome(&active) {
            let _ = self.finish_terminal_operation(
                operation,
                active,
                Some(MaterializationOperationState::UploadReady(frame)),
                Some((LoadStage::Installing, outcome, 0)),
                true,
            );
            return Ok(());
        }
        active.state = Some(MaterializationOperationState::Installing(frame));
        self.operations.insert(operation, active);
        Ok(())
    }

    fn cancel(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        stage: LoadStage,
        reason: CancellationReason,
    ) -> std::result::Result<(), FailureReason> {
        let Some(mut active) = self.operations.remove(&operation) else {
            if self
                .selected_lease_ownership
                .active_owned_by(key, operation)
            {
                return self.release_execution_lease(key);
            }
            return Ok(());
        };
        let Some(state) = active.state.take() else {
            let _ = self.finish_terminal_operation(operation, active, None, None, false);
            return Err(FailureReason::ProtocolViolation(
                "cancel found an ownerless physical operation".into(),
            ));
        };
        if active.key != key {
            active.state = Some(state);
            self.operations.insert(operation, active);
            return Err(FailureReason::ProtocolViolation(
                "cancel key does not match physical operation identity".into(),
            ));
        }
        if active.pending_terminal.is_some() {
            active.state = Some(state);
            self.operations.insert(operation, active);
            return Ok(());
        }
        if !cancel_observation_matches_physical_state(stage, &state) {
            let actual_stage = state.owner_stage();
            active.state = Some(state);
            self.operations.insert(operation, active);
            return Err(FailureReason::ProtocolViolation(format!(
                "cancel observed stage {stage:?} is incompatible with physical operation stage {actual_stage:?}"
            )));
        }
        match state {
            MaterializationOperationState::Reserved(ticket) => self
                .finish_terminal_operation(
                    operation,
                    active,
                    Some(MaterializationOperationState::Reserved(ticket)),
                    None,
                    false,
                )
                .map_err(protocol_failure),
            MaterializationOperationState::HostReady(bundle) => self
                .finish_terminal_operation(
                    operation,
                    active,
                    Some(MaterializationOperationState::HostReady(bundle)),
                    None,
                    true,
                )
                .map_err(protocol_failure),
            MaterializationOperationState::UploadReady(frame) => self
                .finish_terminal_operation(
                    operation,
                    active,
                    Some(MaterializationOperationState::UploadReady(frame)),
                    None,
                    true,
                )
                .map_err(protocol_failure),
            MaterializationOperationState::ReadSubmitted(ticket) => {
                if let Err(error) = self.reader.cancel_load_source_pinned(ticket) {
                    active.state = Some(MaterializationOperationState::ReadSubmitted(ticket));
                    self.operations.insert(operation, active);
                    return Err(protocol_failure(error));
                }
                active.pending_terminal = Some(CompletionOutcome::Cancelled(reason));
                active.state = Some(MaterializationOperationState::ReadSubmitted(ticket));
                self.operations.insert(operation, active);
                Ok(())
            }
            MaterializationOperationState::UploadSubmitted(ticket) => {
                active.pending_terminal = Some(CompletionOutcome::Cancelled(reason));
                active.state = Some(MaterializationOperationState::UploadSubmitted(ticket));
                self.operations.insert(operation, active);
                Ok(())
            }
            MaterializationOperationState::Installing(frame) => self
                .finish_terminal_operation(
                    operation,
                    active,
                    Some(MaterializationOperationState::Installing(frame)),
                    Some((
                        LoadStage::Installing,
                        CompletionOutcome::Cancelled(reason),
                        0,
                    )),
                    true,
                )
                .map_err(protocol_failure),
        }
    }

    fn next_completion(&mut self) -> Option<CompletionEvent> {
        self.progress_one()
    }
}

impl Drop for DeepSeekV4CudaMaterializationProvider {
    fn drop(&mut self) {
        let operation_count = self.operations.len();
        let reservation_count = self.reservations.len();
        if operation_count != 0 || reservation_count != 0 {
            tracing::error!(
                operations = operation_count,
                reservations = reservation_count,
                "DeepSeek-V4 CUDA materialization provider dropped before runtime registry drain"
            );
        }

        let operations = std::mem::take(&mut self.operations);
        for (operation_id, mut operation) in operations {
            let state = operation.state.take();
            if let Err(error) =
                self.finish_terminal_operation(operation_id, operation, state, None, false)
            {
                tracing::error!(
                    operation = operation_id.get(),
                    error = %error,
                    "failed to drain a DeepSeek-V4 physical operation during drop"
                );
            }
        }
        self.operation_order.clear();

        let reservations = std::mem::take(&mut self.reservations);
        for (key, reservation) in reservations {
            self.remove_request_key(reservation.request, key);
            self.selected_lease_ownership.cancel_unbound(key);
            if let Err(error) = self.cancel_prepared(reservation.prepared) {
                tracing::error!(
                    error = %error,
                    "failed to cancel a DeepSeek-V4 slot reservation during drop"
                );
            }
        }

        let selected_ownership = std::mem::take(&mut self.selected_lease_ownership.ownership);
        if !selected_ownership.is_empty() {
            let mut shared = self.shared.lock();
            for (_, ownership) in selected_ownership {
                let SelectedLeaseState::Active(lease) = ownership.state else {
                    tracing::error!(
                        operation = ?ownership.operation.map(OperationId::get),
                        "DeepSeek-V4 pending selected ownership survived backend drain"
                    );
                    continue;
                };
                let release = shared
                    .residency
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Execution("physical residency control is not installed".into())
                    })
                    .and_then(|residency| residency.release(lease));
                if let Err(error) = release {
                    tracing::error!(
                        error = %error,
                        "failed to release a DeepSeek-V4 selected lease during drop"
                    );
                }
            }
        }
    }
}

impl PinnedExpertBundle {
    fn from_payload(payload: PinnedExpertArtifactPayload) -> Result<Self> {
        let expert = payload.expert;
        let mut grouped = BTreeMap::<
            ExpertMatrixKind,
            Vec<crate::moe::io_uring_reader::PinnedExpertTensorPayload>,
        >::new();
        for tensor in payload.tensors {
            if tensor.slice.key.expert != expert {
                return Err(Error::Model(
                    "physical pinned expert payload identity mismatch".into(),
                ));
            }
            grouped
                .entry(tensor.slice.key.matrix)
                .or_default()
                .push(tensor);
        }
        let gate = PinnedExpertLinear::from_tensors(
            expert,
            ExpertMatrixKind::Gate,
            grouped.remove(&ExpertMatrixKind::Gate).unwrap_or_default(),
        )?;
        let up = PinnedExpertLinear::from_tensors(
            expert,
            ExpertMatrixKind::Up,
            grouped.remove(&ExpertMatrixKind::Up).unwrap_or_default(),
        )?;
        let down = PinnedExpertLinear::from_tensors(
            expert,
            ExpertMatrixKind::Down,
            grouped.remove(&ExpertMatrixKind::Down).unwrap_or_default(),
        )?;
        let bytes = [
            gate.weight.len(),
            gate.scale.len(),
            up.weight.len(),
            up.scale.len(),
            down.weight.len(),
            down.scale.len(),
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            total.checked_add(bytes as u64).ok_or_else(|| {
                Error::Model("physical pinned expert payload byte total overflow".into())
            })
        })?;
        Ok(Self {
            expert,
            gate,
            up,
            down,
            bytes,
        })
    }
}

impl PinnedExpertLinear {
    fn from_tensors(
        expert: ExpertId,
        matrix: ExpertMatrixKind,
        tensors: Vec<crate::moe::io_uring_reader::PinnedExpertTensorPayload>,
    ) -> Result<Self> {
        let mut weight = None;
        let mut scale = None;
        for tensor in tensors {
            if tensor.slice.key.expert != expert || tensor.slice.key.matrix != matrix {
                return Err(Error::Model(
                    "physical pinned expert tensor identity mismatch".into(),
                ));
            }
            match tensor.slice.component {
                crate::moe::streaming::ExpertTensorComponent::Weight => {
                    if weight.replace(tensor).is_some() {
                        return Err(Error::Model("duplicate physical expert weight".into()));
                    }
                }
                crate::moe::streaming::ExpertTensorComponent::Scale => {
                    if scale.replace(tensor).is_some() {
                        return Err(Error::Model("duplicate physical expert scale".into()));
                    }
                }
                crate::moe::streaming::ExpertTensorComponent::Other(component) => {
                    return Err(Error::Model(format!(
                        "unsupported physical expert tensor component {component}"
                    )));
                }
            }
        }
        let weight = weight.ok_or_else(|| Error::Model("missing physical expert weight".into()))?;
        let scale = scale.ok_or_else(|| Error::Model("missing physical expert scale".into()))?;
        let format = infer_expert_linear_format(
            &weight.slice,
            weight.bytes.len(),
            Some((&scale.slice, scale.bytes.len())),
        )?;
        Ok(Self {
            matrix,
            format,
            weight: weight.bytes,
            scale: scale.bytes,
        })
    }
}

fn pinned_linear_shape(
    linear: &PinnedExpertLinear,
) -> Result<ferrule_cuda::context::CudaArtifactLinearShape> {
    let ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
        out_features,
        in_features,
        block_size: 32,
    } = linear.format
    else {
        return Err(Error::Model(format!(
            "CUDA physical expert {:?} requires FP4 E2M1/E8M0 block_size=32",
            linear.matrix
        )));
    };
    Ok(
        ferrule_cuda::context::CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features,
            in_features,
        },
    )
}

fn validate_operation_reservation(
    active: &MaterializationOperation,
    operation: OperationId,
    key: MaterializationKey,
    reservation: &PhysicalMaterializationOperationReservation,
    plan: MaterializationResourcePlan,
) -> std::result::Result<(), FailureReason> {
    if active.key != key
        || active.resource_plan != plan
        || reservation.key() != key
        || reservation.binding() != active.binding
        || reservation.upload_fence().operation != operation
        || reservation.upload_fence().destination_generation != key.destination_generation()
    {
        return Err(FailureReason::ProtocolViolation(
            "physical operation reservation identity mismatch".into(),
        ));
    }
    Ok(())
}

fn record_first_error(first: &mut Option<Error>, result: Result<()>) {
    if let Err(error) = result {
        first.get_or_insert(error);
    }
}

fn protocol_failure(error: Error) -> FailureReason {
    FailureReason::ProtocolViolation(error.to_string())
}

#[cfg(test)]
mod tests {
    use ferrule_common::{
        BackendId, ContentHash, DeviceId, DispatchFenceContract, ExpertId as ProtocolExpertId,
        LayerId, MappingEpoch, MaterializedResourceId, ModelInstanceId, PayloadEncodingId,
        SourceGeneration, SourceIdentityHash,
    };

    use super::*;

    fn lease_key(expert: u32, generation: u64) -> MaterializationKey {
        MaterializationKey::new(
            ModelInstanceId::new(17),
            SourceIdentityHash::new([1; 32]),
            ContentHash::new([expert as u8; 32]),
            MaterializedResourceId::routed_expert(LayerId::new(3), ProtocolExpertId::new(expert)),
            PayloadEncodingId::new(1),
            BackendId::new(4),
            DeviceId::new(0),
            SourceGeneration::new(2),
            DestinationGeneration::new(generation),
        )
        .unwrap()
    }

    fn lease_set(entries: &[(MaterializationKey, u32)]) -> ResidencyLeaseSet {
        ResidencyLeaseSet::new(
            entries.iter().map(|(key, _)| *key),
            entries.iter().map(|(key, slot)| {
                ValidatedResidencyBinding::new(
                    *key,
                    ResidencyBinding::new(
                        key.model(),
                        key.resource(),
                        key.backend(),
                        key.device(),
                        DestinationSlotId::new(*slot),
                        key.destination_generation(),
                    ),
                )
                .unwrap()
            }),
            MappingEpoch::new(1),
            DispatchFenceContract::new(
                OperationId::new(91),
                FenceId::new(92),
                BackendId::new(4),
                DeviceId::new(0),
            ),
        )
        .unwrap()
    }

    fn published(key: MaterializationKey, slot: u32) -> PublishedExpertBinding {
        PublishedExpertBinding {
            key,
            slot: DestinationSlotId::new(slot),
            generation: key.destination_generation(),
            frame_published: true,
        }
    }

    #[test]
    fn published_expert_window_accepts_complete_exact_leases() {
        let first = lease_key(1, 7);
        let second = lease_key(2, 8);
        let leases = lease_set(&[(first, 4), (second, 5)]);
        let published = BTreeMap::from([(1, published(first, 4)), (2, published(second, 5))]);

        validate_published_lease_window(3, &[1, 2], &leases, |expert| {
            published.get(&expert).copied()
        })
        .unwrap();
    }

    #[test]
    fn unpublished_expert_window_is_rejected() {
        let key = lease_key(1, 7);
        let leases = lease_set(&[(key, 4)]);
        let error = validate_published_lease_window(3, &[1], &leases, |_| None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("is not published"));
    }

    #[test]
    fn wrong_published_generation_is_rejected() {
        let key = lease_key(1, 7);
        let leases = lease_set(&[(key, 4)]);
        let mut stale = published(key, 4);
        stale.generation = DestinationGeneration::new(8);
        let error = validate_published_lease_window(3, &[1], &leases, |_| Some(stale))
            .unwrap_err()
            .to_string();
        assert!(error.contains("key/slot/generation"));
    }

    #[test]
    fn incomplete_expert_lease_window_is_rejected() {
        let key = lease_key(1, 7);
        let leases = lease_set(&[(key, 4)]);
        let error = validate_published_lease_window(3, &[1, 2], &leases, |expert| {
            (expert == 1).then_some(published(key, 4))
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("selected=2 leases=1"));
    }

    #[test]
    fn resident_expert_key_index_does_not_require_ord() {
        let expert = ExpertKey::new(17, 3, 1);
        let key = lease_key(1, 7);
        let mut resident_keys = ResidentExpertKeyIndex::new();
        resident_keys.insert(expert, key);
        assert_eq!(resident_keys.get(&expert), Some(&key));
    }

    fn leased_resident(
        expert: ExpertKey,
    ) -> (ferrule_common::ExpertResidencyCoordinator, ExpertLease) {
        let mut residency = ferrule_common::ExpertResidencyCoordinator::new(1).unwrap();
        let prepared = residency
            .try_prepare_install(expert, ExpertInstallReason::Selected)
            .unwrap()
            .unwrap();
        assert_eq!(
            residency.activate_install(prepared).unwrap(),
            ExpertInstallActivationOutcome::Activated
        );
        let (_, lease) = residency.publish_install_leased(prepared).unwrap();
        (residency, lease)
    }

    #[test]
    fn release_racing_publication_is_deferred_and_released_exactly_once() {
        let key = lease_key(1, 7);
        let expert = ExpertKey::new(17, 3, 1);
        let operation = OperationId::new(41);
        let (mut residency, lease) = leased_resident(expert);
        let mut selected = SelectedLeaseTracker::default();
        selected.begin_pending(key).unwrap();
        selected.bind_operation(key, operation).unwrap();

        assert!(matches!(
            selected.request_release(key),
            SelectedLeaseRelease::Deferred
        ));
        assert_eq!(
            selected.publish(key, operation, lease).unwrap(),
            SelectedLeasePublication::ReleaseImmediately
        );
        let SelectedLeaseRelease::Active { lease, .. } = selected.request_release(key) else {
            panic!("published release request must expose the exact lease");
        };
        residency.release(lease).unwrap();
        assert_eq!(residency.stats().active_leases, 0);
        assert!(matches!(
            selected.request_release(key),
            SelectedLeaseRelease::None
        ));

        let replacement = ExpertKey::new(17, 3, 2);
        assert!(
            residency
                .try_prepare_install(replacement, ExpertInstallReason::Selected)
                .unwrap()
                .is_some(),
            "the second turn must be able to evict the released resident slot"
        );
    }

    #[test]
    fn stale_operation_cannot_publish_or_cancel_another_selected_owner() {
        let key = lease_key(1, 7);
        let expert = ExpertKey::new(17, 3, 1);
        let owner = OperationId::new(51);
        let stale = OperationId::new(52);
        let (mut residency, lease) = leased_resident(expert);
        let mut selected = SelectedLeaseTracker::default();
        selected.begin_pending(key).unwrap();
        selected.bind_operation(key, owner).unwrap();

        assert!(selected.publish(key, stale, lease).is_err());
        assert_eq!(selected.pending_count(), 1);
        selected.cancel_operation(key, stale);
        assert_eq!(selected.pending_count(), 1);
        selected.cancel_operation(key, owner);
        assert_eq!(selected.pending_count(), 0);
        residency.release(lease).unwrap();
        assert_eq!(residency.stats().active_leases, 0);
    }

    type MockState = MaterializationOperationState<u64, u64, u64, u64>;

    struct MockOperation {
        key: u64,
        source_identity: u64,
        slot_generation: u64,
        pending_terminal: Option<CompletionOutcome>,
        state: Option<MockState>,
    }

    #[derive(Debug, Default, PartialEq, Eq)]
    struct MockResources {
        read_slabs: usize,
        host_bundles: usize,
        upload_events: usize,
        frames: usize,
        prepared_slots: usize,
        resident_frames: usize,
    }

    impl MockResources {
        fn transient_is_empty(&self) -> bool {
            self.read_slabs == 0
                && self.host_bundles == 0
                && self.upload_events == 0
                && self.frames == 0
                && self.prepared_slots == 0
        }
    }

    #[derive(Default)]
    struct HostMockProvider {
        operations: HashMap<u64, MockOperation>,
        seen_operations: HashSet<u64>,
        terminal_operations: HashSet<u64>,
        completions: Vec<(u64, LoadStage, CompletionOutcome)>,
        source_identities: HashMap<u64, u64>,
        slot_generations: HashMap<u64, u64>,
        resources: MockResources,
    }

    impl HostMockProvider {
        fn reserve(&mut self, operation: u64, key: u64) -> bool {
            if operation == 0
                || self.seen_operations.contains(&operation)
                || self.operations.values().any(|active| active.key == key)
            {
                return false;
            }
            let source_identity = *self.source_identities.entry(key).or_insert(1);
            let slot_generation = *self.slot_generations.entry(key).or_insert(1);
            self.resources.read_slabs += 1;
            self.resources.prepared_slots += 1;
            self.seen_operations.insert(operation);
            self.operations.insert(
                operation,
                MockOperation {
                    key,
                    source_identity,
                    slot_generation,
                    pending_terminal: None,
                    state: Some(MockState::Reserved(operation)),
                },
            );
            true
        }

        fn transition(&mut self, operation: u64, expected: LoadStage, next: MockState) -> bool {
            let Some(mut active) = self.operations.remove(&operation) else {
                return false;
            };
            let Some(state) = active.state.take() else {
                self.operations.insert(operation, active);
                return false;
            };
            if state.owner_stage() != expected {
                active.state = Some(state);
                self.operations.insert(operation, active);
                return false;
            }
            active.state = Some(next);
            self.operations.insert(operation, active);
            true
        }

        fn submit_read(&mut self, operation: u64) -> bool {
            self.transition(
                operation,
                LoadStage::Reserved,
                MockState::ReadSubmitted(operation),
            )
        }

        fn complete_read(&mut self, operation: u64, failed: bool) -> bool {
            let Some(mut active) = self.operations.remove(&operation) else {
                return false;
            };
            let ticket = match active.state.take() {
                Some(MockState::ReadSubmitted(ticket)) => ticket,
                state => {
                    active.state = state;
                    self.operations.insert(operation, active);
                    return false;
                }
            };
            self.resources.read_slabs -= 1;
            if let Some(outcome) = active.pending_terminal.take() {
                self.finish(
                    operation,
                    active,
                    None,
                    Some((LoadStage::ReadSubmitted, outcome)),
                );
            } else if self.source_identities.get(&active.key) != Some(&active.source_identity) {
                self.finish(
                    operation,
                    active,
                    None,
                    Some((
                        LoadStage::ReadSubmitted,
                        CompletionOutcome::Stale(StaleReason::SourceIdentityChanged),
                    )),
                );
            } else if failed {
                self.finish(
                    operation,
                    active,
                    None,
                    Some((
                        LoadStage::ReadSubmitted,
                        CompletionOutcome::Failed(FailureReason::StorageUnavailable),
                    )),
                );
            } else {
                self.resources.host_bundles += 1;
                active.state = Some(MockState::HostReady(ticket));
                self.operations.insert(operation, active);
                self.completions.push((
                    operation,
                    LoadStage::ReadSubmitted,
                    CompletionOutcome::Succeeded,
                ));
            }
            true
        }

        fn submit_upload(&mut self, operation: u64) -> bool {
            let Some(mut active) = self.operations.remove(&operation) else {
                return false;
            };
            let bundle = match active.state.take() {
                Some(MockState::HostReady(bundle)) => bundle,
                state => {
                    active.state = state;
                    self.operations.insert(operation, active);
                    return false;
                }
            };
            if self.source_identities.get(&active.key) != Some(&active.source_identity) {
                self.finish(
                    operation,
                    active,
                    Some(MockState::HostReady(bundle)),
                    Some((
                        LoadStage::UploadSubmitted,
                        CompletionOutcome::Stale(StaleReason::SourceIdentityChanged),
                    )),
                );
                return true;
            }
            self.resources.upload_events += 1;
            self.resources.frames += 1;
            active.state = Some(MockState::UploadSubmitted(bundle));
            self.operations.insert(operation, active);
            true
        }

        fn complete_upload(&mut self, operation: u64, failed: bool) -> bool {
            let Some(mut active) = self.operations.remove(&operation) else {
                return false;
            };
            let ticket = match active.state.take() {
                Some(MockState::UploadSubmitted(ticket)) => ticket,
                state => {
                    active.state = state;
                    self.operations.insert(operation, active);
                    return false;
                }
            };
            self.resources.upload_events -= 1;
            self.resources.host_bundles -= 1;
            let outcome = active.pending_terminal.take().or_else(|| {
                (self.source_identities.get(&active.key) != Some(&active.source_identity))
                    .then_some(CompletionOutcome::Stale(StaleReason::SourceIdentityChanged))
            });
            if let Some(outcome) = outcome {
                self.finish(
                    operation,
                    active,
                    Some(MockState::UploadReady(ticket)),
                    Some((LoadStage::UploadSubmitted, outcome)),
                );
            } else if failed {
                self.finish(
                    operation,
                    active,
                    Some(MockState::UploadReady(ticket)),
                    Some((
                        LoadStage::UploadSubmitted,
                        CompletionOutcome::Failed(FailureReason::DeviceUnavailable),
                    )),
                );
            } else {
                active.state = Some(MockState::UploadReady(ticket));
                self.operations.insert(operation, active);
                self.completions.push((
                    operation,
                    LoadStage::UploadSubmitted,
                    CompletionOutcome::Succeeded,
                ));
            }
            true
        }

        fn submit_install(&mut self, operation: u64) -> bool {
            let Some(mut active) = self.operations.remove(&operation) else {
                return false;
            };
            let frame = match active.state.take() {
                Some(MockState::UploadReady(frame)) => frame,
                state => {
                    active.state = state;
                    self.operations.insert(operation, active);
                    return false;
                }
            };
            let outcome =
                if self.source_identities.get(&active.key) != Some(&active.source_identity) {
                    Some(CompletionOutcome::Stale(StaleReason::SourceIdentityChanged))
                } else if self.slot_generations.get(&active.key) != Some(&active.slot_generation) {
                    Some(CompletionOutcome::Stale(StaleReason::DestinationReused))
                } else {
                    None
                };
            if let Some(outcome) = outcome {
                self.finish(
                    operation,
                    active,
                    Some(MockState::UploadReady(frame)),
                    Some((LoadStage::Installing, outcome)),
                );
            } else {
                active.state = Some(MockState::Installing(frame));
                self.operations.insert(operation, active);
            }
            true
        }

        fn complete_install(&mut self, operation: u64, post_source_identity: Option<u64>) -> bool {
            self.complete_install_after(operation, post_source_identity, None)
        }

        fn complete_install_after(
            &mut self,
            operation: u64,
            post_source_identity: Option<u64>,
            post_slot_generation: Option<u64>,
        ) -> bool {
            let Some(mut active) = self.operations.remove(&operation) else {
                return false;
            };
            let frame = match active.state.take() {
                Some(MockState::Installing(frame)) => frame,
                state => {
                    active.state = state;
                    self.operations.insert(operation, active);
                    return false;
                }
            };
            if let Some(outcome) = active.pending_terminal.take() {
                self.finish(
                    operation,
                    active,
                    Some(MockState::Installing(frame)),
                    Some((LoadStage::Installing, outcome)),
                );
                return true;
            }
            let source_before = self.source_identities.get(&active.key).copied();
            let generation_before = self.slot_generations.get(&active.key).copied();
            if source_before != Some(active.source_identity) {
                self.finish(
                    operation,
                    active,
                    Some(MockState::Installing(frame)),
                    Some((
                        LoadStage::Installing,
                        CompletionOutcome::Stale(StaleReason::SourceIdentityChanged),
                    )),
                );
                return true;
            }
            if generation_before != Some(active.slot_generation) {
                self.finish(
                    operation,
                    active,
                    Some(MockState::Installing(frame)),
                    Some((
                        LoadStage::Installing,
                        CompletionOutcome::Stale(StaleReason::DestinationReused),
                    )),
                );
                return true;
            }
            if let Some(identity) = post_source_identity {
                self.source_identities.insert(active.key, identity);
            }
            if let Some(generation) = post_slot_generation {
                self.slot_generations.insert(active.key, generation);
            }
            let source_after = self.source_identities.get(&active.key).copied();
            let generation_after = self.slot_generations.get(&active.key).copied();
            if source_after != Some(active.source_identity)
                || generation_after != Some(active.slot_generation)
            {
                let outcome = if source_after != Some(active.source_identity) {
                    CompletionOutcome::Stale(StaleReason::SourceIdentityChanged)
                } else {
                    CompletionOutcome::Stale(StaleReason::DestinationReused)
                };
                self.finish(
                    operation,
                    active,
                    Some(MockState::Installing(frame)),
                    Some((LoadStage::Installing, outcome)),
                );
                return true;
            }
            self.resources.frames -= 1;
            self.resources.prepared_slots -= 1;
            self.resources.resident_frames += 1;
            if self.terminal_operations.insert(operation) {
                self.completions.push((
                    operation,
                    LoadStage::Installing,
                    CompletionOutcome::Succeeded,
                ));
            }
            true
        }

        fn fail_install(&mut self, operation: u64) -> bool {
            let Some(mut active) = self.operations.remove(&operation) else {
                return false;
            };
            let frame = match active.state.take() {
                Some(MockState::Installing(frame)) => frame,
                state => {
                    active.state = state;
                    self.operations.insert(operation, active);
                    return false;
                }
            };
            self.finish(
                operation,
                active,
                Some(MockState::Installing(frame)),
                Some((
                    LoadStage::Installing,
                    CompletionOutcome::Failed(FailureReason::InstallationRejected),
                )),
            );
            true
        }

        fn physical_cancel(&mut self, operation: u64, reason: CancellationReason) -> bool {
            let Some(mut active) = self.operations.remove(&operation) else {
                return false;
            };
            let Some(state) = active.state.take() else {
                self.operations.insert(operation, active);
                return false;
            };
            match state {
                MockState::Reserved(_) | MockState::HostReady(_) | MockState::UploadReady(_) => {
                    self.finish(operation, active, Some(state), None);
                }
                MockState::ReadSubmitted(_) | MockState::UploadSubmitted(_) => {
                    active.pending_terminal = Some(CompletionOutcome::Cancelled(reason));
                    active.state = Some(state);
                    self.operations.insert(operation, active);
                }
                MockState::Installing(_) => self.finish(
                    operation,
                    active,
                    Some(state),
                    Some((LoadStage::Installing, CompletionOutcome::Cancelled(reason))),
                ),
            }
            true
        }

        fn release_state(&mut self, state: MockState) {
            match state {
                MockState::Reserved(_) | MockState::ReadSubmitted(_) => {
                    self.resources.read_slabs -= 1;
                }
                MockState::HostReady(_) => {
                    self.resources.host_bundles -= 1;
                }
                MockState::UploadSubmitted(_) => {
                    self.resources.upload_events -= 1;
                    self.resources.host_bundles -= 1;
                    self.resources.frames -= 1;
                }
                MockState::UploadReady(_) | MockState::Installing(_) => {
                    self.resources.frames -= 1;
                }
            }
        }

        fn finish(
            &mut self,
            operation: u64,
            _active: MockOperation,
            state: Option<MockState>,
            completion: Option<(LoadStage, CompletionOutcome)>,
        ) {
            if let Some(state) = state {
                self.release_state(state);
            }
            self.resources.prepared_slots -= 1;
            if self.terminal_operations.insert(operation)
                && let Some((stage, outcome)) = completion
            {
                self.completions.push((operation, stage, outcome));
            }
        }

        fn change_source(&mut self, key: u64) {
            let current = self.source_identities.entry(key).or_insert(1);
            *current += 1;
        }

        fn reuse_slot(&mut self, key: u64) {
            let current = self.slot_generations.entry(key).or_insert(1);
            *current += 1;
        }

        fn shutdown(&mut self) {
            let operations = std::mem::take(&mut self.operations);
            for (operation, mut active) in operations {
                let state = active.state.take();
                self.finish(operation, active, state, None);
            }
        }

        fn terminal_completion_count(&self, operation: u64) -> usize {
            self.completions
                .iter()
                .filter(|(candidate, _, outcome)| {
                    *candidate == operation && !matches!(outcome, CompletionOutcome::Succeeded)
                })
                .count()
        }
    }

    fn advance_to_host(mock: &mut HostMockProvider, operation: u64) {
        assert!(mock.submit_read(operation));
        assert!(mock.complete_read(operation, false));
    }

    fn advance_to_upload(mock: &mut HostMockProvider, operation: u64) {
        advance_to_host(mock, operation);
        assert!(mock.submit_upload(operation));
    }

    fn advance_to_frame(mock: &mut HostMockProvider, operation: u64) {
        advance_to_upload(mock, operation);
        assert!(mock.complete_upload(operation, false));
    }

    #[test]
    fn host_mock_success_visits_every_submitted_stage_once() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(1, 11));
        advance_to_frame(&mut mock, 1);
        assert!(mock.submit_install(1));
        assert!(mock.complete_install(1, None));
        assert!(!mock.complete_install(1, None));
        let stages = mock
            .completions
            .iter()
            .map(|(_, stage, _)| *stage)
            .collect::<Vec<_>>();
        assert_eq!(
            stages,
            vec![
                LoadStage::ReadSubmitted,
                LoadStage::UploadSubmitted,
                LoadStage::Installing,
            ]
        );
        assert_eq!(mock.resources.resident_frames, 1);
        assert!(mock.resources.transient_is_empty());
        assert_eq!(mock.terminal_operations.len(), 1);
    }

    #[test]
    fn host_mock_cancel_drains_every_owner_stage() {
        let mut mock = HostMockProvider::default();

        assert!(mock.reserve(1, 1));
        assert!(mock.physical_cancel(1, CancellationReason::ExternalRequest));
        assert!(mock.resources.transient_is_empty());

        assert!(mock.reserve(2, 2));
        assert!(mock.submit_read(2));
        assert!(mock.physical_cancel(2, CancellationReason::ExternalRequest));
        assert_eq!(mock.resources.read_slabs, 1);
        assert!(mock.complete_read(2, false));
        assert!(mock.resources.transient_is_empty());

        assert!(mock.reserve(3, 3));
        advance_to_host(&mut mock, 3);
        assert!(mock.physical_cancel(3, CancellationReason::ExternalRequest));
        assert!(mock.resources.transient_is_empty());

        assert!(mock.reserve(4, 4));
        advance_to_upload(&mut mock, 4);
        assert!(mock.physical_cancel(4, CancellationReason::ExternalRequest));
        assert_eq!(mock.resources.upload_events, 1);
        assert_eq!(mock.resources.host_bundles, 1);
        assert_eq!(mock.resources.frames, 1);
        assert!(mock.complete_upload(4, false));
        assert!(mock.resources.transient_is_empty());

        assert!(mock.reserve(5, 5));
        advance_to_frame(&mut mock, 5);
        assert!(mock.physical_cancel(5, CancellationReason::ExternalRequest));
        assert!(mock.resources.transient_is_empty());

        assert!(mock.reserve(6, 6));
        advance_to_frame(&mut mock, 6);
        assert!(mock.submit_install(6));
        assert!(mock.physical_cancel(6, CancellationReason::ExternalRequest));
        assert!(mock.resources.transient_is_empty());
        assert!(!mock.complete_install(6, None));
        assert_eq!(mock.terminal_completion_count(6), 1);
        assert_eq!(mock.terminal_operations.len(), 6);
    }

    #[test]
    fn host_mock_one_waiter_detach_keeps_owner_until_last_waiter_cancel() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(7, 70));
        assert!(mock.submit_read(7));

        let mut waiters = 2;
        waiters -= 1;
        assert_eq!(waiters, 1);
        assert!(mock.operations.contains_key(&7));
        assert_eq!(mock.resources.read_slabs, 1);

        waiters -= 1;
        assert_eq!(waiters, 0);
        assert!(mock.physical_cancel(7, CancellationReason::LastWaiterDetached));
        assert_eq!(mock.resources.read_slabs, 1);
        assert!(mock.complete_read(7, false));
        assert!(!mock.complete_read(7, false));
        assert!(mock.resources.transient_is_empty());
        assert_eq!(mock.terminal_completion_count(7), 1);
    }

    #[test]
    fn host_mock_failure_stale_and_slot_reuse_allow_new_operation_retry() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(8, 80));
        assert!(mock.submit_read(8));
        assert!(mock.complete_read(8, true));
        assert!(matches!(
            mock.completions.last(),
            Some((8, LoadStage::ReadSubmitted, CompletionOutcome::Failed(_)))
        ));
        assert!(mock.reserve(9, 80));
        advance_to_upload(&mut mock, 9);
        mock.change_source(80);
        assert!(mock.complete_upload(9, false));
        assert!(matches!(
            mock.completions.last(),
            Some((
                9,
                LoadStage::UploadSubmitted,
                CompletionOutcome::Stale(StaleReason::SourceIdentityChanged)
            ))
        ));
        assert!(mock.reserve(10, 80));
        advance_to_frame(&mut mock, 10);
        assert!(mock.submit_install(10));
        mock.reuse_slot(80);
        assert!(mock.complete_install(10, None));
        assert!(matches!(
            mock.completions.last(),
            Some((
                10,
                LoadStage::Installing,
                CompletionOutcome::Stale(StaleReason::DestinationReused)
            ))
        ));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_checks_source_identity_before_and_after_install() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(11, 110));
        advance_to_frame(&mut mock, 11);
        assert!(mock.submit_install(11));
        assert!(mock.complete_install(11, Some(2)));
        assert!(matches!(
            mock.completions.last(),
            Some((
                11,
                LoadStage::Installing,
                CompletionOutcome::Stale(StaleReason::SourceIdentityChanged)
            ))
        ));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_checks_slot_generation_before_and_after_install() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(12, 120));
        advance_to_frame(&mut mock, 12);
        assert!(mock.submit_install(12));
        assert!(mock.complete_install_after(12, None, Some(2)));
        assert!(matches!(
            mock.completions.last(),
            Some((
                12,
                LoadStage::Installing,
                CompletionOutcome::Stale(StaleReason::DestinationReused)
            ))
        ));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_duplicate_completion_is_ignored_after_exact_terminal() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(12, 120));
        assert!(mock.submit_read(12));
        assert!(mock.physical_cancel(12, CancellationReason::LastWaiterDetached));
        assert!(mock.complete_read(12, false));
        assert!(!mock.complete_read(12, false));
        assert_eq!(mock.terminal_operations.len(), 1);
        assert_eq!(mock.terminal_completion_count(12), 1);
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_shutdown_drains_all_states_without_leaks() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(21, 201));

        assert!(mock.reserve(22, 202));
        assert!(mock.submit_read(22));

        assert!(mock.reserve(23, 203));
        advance_to_host(&mut mock, 23);

        assert!(mock.reserve(24, 204));
        advance_to_upload(&mut mock, 24);

        assert!(mock.reserve(25, 205));
        advance_to_frame(&mut mock, 25);

        assert!(mock.reserve(26, 206));
        advance_to_frame(&mut mock, 26);
        assert!(mock.submit_install(26));

        mock.shutdown();
        assert!(mock.operations.is_empty());
        assert!(mock.resources.transient_is_empty());
        assert_eq!(mock.terminal_operations.len(), 6);

        mock.shutdown();
        assert!(mock.operations.is_empty());
        assert!(mock.resources.transient_is_empty());
        assert_eq!(mock.terminal_operations.len(), 6);
    }

    #[test]
    fn host_mock_reserved_cancel_releases_slot_and_slabs_immediately() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(31, 301));
        assert!(mock.physical_cancel(31, CancellationReason::ExternalRequest));
        assert!(mock.resources.transient_is_empty());
        assert!(mock.operations.is_empty());
        assert!(mock.terminal_operations.contains(&31));
    }

    #[test]
    fn host_mock_read_cancel_retains_slabs_until_cqe() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(32, 302));
        assert!(mock.submit_read(32));
        assert!(mock.physical_cancel(32, CancellationReason::DeadlineExceeded));
        assert_eq!(mock.resources.read_slabs, 1);
        assert!(mock.complete_read(32, false));
        assert!(mock.resources.transient_is_empty());
        assert_eq!(mock.terminal_completion_count(32), 1);
    }

    #[test]
    fn host_mock_host_ready_cancel_releases_pinned_bundle_immediately() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(33, 303));
        advance_to_host(&mut mock, 33);
        assert_eq!(mock.resources.host_bundles, 1);
        assert!(mock.physical_cancel(33, CancellationReason::Superseded));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_upload_cancel_waits_for_event_before_recycling_frame() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(34, 304));
        advance_to_upload(&mut mock, 34);
        assert!(mock.physical_cancel(34, CancellationReason::ExternalRequest));
        assert_eq!(mock.resources.upload_events, 1);
        assert_eq!(mock.resources.frames, 1);
        assert!(mock.complete_upload(34, false));
        assert!(mock.resources.transient_is_empty());
        assert_eq!(mock.terminal_completion_count(34), 1);
    }

    #[test]
    fn host_mock_upload_ready_cancel_recycles_completed_frame() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(35, 305));
        advance_to_frame(&mut mock, 35);
        assert_eq!(mock.resources.frames, 1);
        assert!(mock.physical_cancel(35, CancellationReason::ExternalRequest));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_install_cancel_drains_frame_and_slot_immediately() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(36, 306));
        advance_to_frame(&mut mock, 36);
        assert!(mock.submit_install(36));
        assert!(mock.physical_cancel(36, CancellationReason::OwnerShutdown));
        assert!(mock.resources.transient_is_empty());
        assert!(!mock.complete_install(36, None));
        assert_eq!(mock.terminal_completion_count(36), 1);
    }

    #[test]
    fn host_mock_read_failure_releases_all_transient_owners() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(41, 401));
        assert!(mock.submit_read(41));
        assert!(mock.complete_read(41, true));
        assert!(matches!(
            mock.completions.last(),
            Some((41, LoadStage::ReadSubmitted, CompletionOutcome::Failed(_)))
        ));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_upload_failure_waits_for_event_then_releases_every_owner() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(42, 402));
        advance_to_upload(&mut mock, 42);
        assert!(mock.complete_upload(42, true));
        assert!(matches!(
            mock.completions.last(),
            Some((42, LoadStage::UploadSubmitted, CompletionOutcome::Failed(_)))
        ));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_install_failure_recycles_frame_and_cancels_slot() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(43, 403));
        advance_to_frame(&mut mock, 43);
        assert!(mock.submit_install(43));
        assert!(mock.fail_install(43));
        assert!(matches!(
            mock.completions.last(),
            Some((43, LoadStage::Installing, CompletionOutcome::Failed(_)))
        ));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_source_stale_during_read_never_reaches_host_ready() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(51, 501));
        assert!(mock.submit_read(51));
        mock.change_source(501);
        assert!(mock.complete_read(51, false));
        assert!(matches!(
            mock.completions.last(),
            Some((
                51,
                LoadStage::ReadSubmitted,
                CompletionOutcome::Stale(StaleReason::SourceIdentityChanged)
            ))
        ));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_source_stale_at_host_ready_never_submits_upload() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(52, 502));
        advance_to_host(&mut mock, 52);
        mock.change_source(502);
        assert!(mock.submit_upload(52));
        assert!(matches!(
            mock.completions.last(),
            Some((
                52,
                LoadStage::UploadSubmitted,
                CompletionOutcome::Stale(StaleReason::SourceIdentityChanged)
            ))
        ));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_source_stale_during_upload_waits_for_event_and_never_installs() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(53, 503));
        advance_to_upload(&mut mock, 53);
        mock.change_source(503);
        assert!(mock.complete_upload(53, false));
        assert!(matches!(
            mock.completions.last(),
            Some((
                53,
                LoadStage::UploadSubmitted,
                CompletionOutcome::Stale(StaleReason::SourceIdentityChanged)
            ))
        ));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_source_stale_before_install_never_publishes() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(54, 504));
        advance_to_frame(&mut mock, 54);
        mock.change_source(504);
        assert!(mock.submit_install(54));
        assert!(matches!(
            mock.completions.last(),
            Some((
                54,
                LoadStage::Installing,
                CompletionOutcome::Stale(StaleReason::SourceIdentityChanged)
            ))
        ));
        assert_eq!(mock.resources.resident_frames, 0);
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_slot_reuse_before_install_never_publishes() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(55, 505));
        advance_to_frame(&mut mock, 55);
        mock.reuse_slot(505);
        assert!(mock.submit_install(55));
        assert!(matches!(
            mock.completions.last(),
            Some((
                55,
                LoadStage::Installing,
                CompletionOutcome::Stale(StaleReason::DestinationReused)
            ))
        ));
        assert_eq!(mock.resources.resident_frames, 0);
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_failed_key_accepts_new_operation_id_retry() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(61, 601));
        assert!(mock.submit_read(61));
        assert!(mock.complete_read(61, true));
        assert!(mock.reserve(62, 601));
        assert!(mock.physical_cancel(62, CancellationReason::Superseded));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_stale_key_accepts_new_operation_id_retry() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(63, 603));
        assert!(mock.submit_read(63));
        mock.change_source(603);
        assert!(mock.complete_read(63, false));
        assert!(mock.reserve(64, 603));
        assert!(mock.physical_cancel(64, CancellationReason::Superseded));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_cancelled_key_accepts_new_operation_id_retry() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(65, 605));
        assert!(mock.physical_cancel(65, CancellationReason::ExternalRequest));
        assert!(mock.reserve(66, 605));
        assert!(mock.physical_cancel(66, CancellationReason::Superseded));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_operation_id_is_never_reused_after_terminal() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(71, 701));
        assert!(mock.physical_cancel(71, CancellationReason::ExternalRequest));
        assert!(!mock.reserve(71, 702));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_materialization_key_has_exactly_one_active_operation_owner() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(72, 702));
        assert!(!mock.reserve(73, 702));
        assert_eq!(mock.operations.len(), 1);
        assert!(mock.physical_cancel(72, CancellationReason::ExternalRequest));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_out_of_order_completion_is_rejected_without_state_mutation() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(73, 703));
        assert!(!mock.complete_read(73, false));
        assert!(!mock.complete_upload(73, false));
        assert!(!mock.complete_install(73, None));
        assert_eq!(mock.resources.read_slabs, 1);
        assert_eq!(
            mock.operations
                .get(&73)
                .and_then(|operation| operation.state.as_ref())
                .map(MaterializationOperationState::owner_stage),
            Some(LoadStage::Reserved)
        );
        assert!(mock.physical_cancel(73, CancellationReason::ExternalRequest));
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn host_mock_terminal_failure_completion_is_exactly_once() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(74, 704));
        assert!(mock.submit_read(74));
        assert!(mock.complete_read(74, true));
        assert!(!mock.complete_read(74, true));
        assert_eq!(mock.terminal_completion_count(74), 1);
        assert!(mock.resources.transient_is_empty());
    }

    #[test]
    fn cancel_observation_only_accepts_exact_or_unapplied_completion_edges() {
        let states = [
            MockState::Reserved(1),
            MockState::ReadSubmitted(2),
            MockState::HostReady(3),
            MockState::UploadSubmitted(4),
            MockState::UploadReady(5),
            MockState::Installing(6),
        ];
        let observed_stages = [
            LoadStage::Reserved,
            LoadStage::ReadSubmitted,
            LoadStage::HostReady,
            LoadStage::UploadSubmitted,
            LoadStage::Installing,
        ];

        for observed in observed_stages {
            for physical in &states {
                let expected = observed == physical.owner_stage()
                    || matches!(
                        (observed, physical),
                        (LoadStage::ReadSubmitted, MockState::HostReady(_))
                            | (LoadStage::UploadSubmitted, MockState::UploadReady(_))
                    );
                assert_eq!(
                    cancel_observation_matches_physical_state(observed, physical),
                    expected,
                    "unexpected cancellation compatibility for observed={observed:?} physical={:?}",
                    physical.owner_stage()
                );
            }
        }
    }

    #[test]
    fn host_mock_state_owner_stage_mapping_is_total() {
        assert_eq!(MockState::Reserved(1).owner_stage(), LoadStage::Reserved);
        assert_eq!(
            MockState::ReadSubmitted(1).owner_stage(),
            LoadStage::ReadSubmitted
        );
        assert_eq!(MockState::HostReady(1).owner_stage(), LoadStage::HostReady);
        assert_eq!(
            MockState::UploadSubmitted(1).owner_stage(),
            LoadStage::UploadSubmitted
        );
        assert_eq!(
            MockState::UploadReady(1).owner_stage(),
            LoadStage::Installing
        );
        assert_eq!(
            MockState::Installing(1).owner_stage(),
            LoadStage::Installing
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_reservation_descriptor_and_upload_event_progression_compile() {
        use ferrule_common::{
            BackendId, ContentHash, DeviceId, ExpertId as ProtocolExpertId, LayerId,
            MaterializedResourceId, ModelInstanceId, PayloadEncodingId,
            RegisteredPinnedAlignedSlabLeaseDescriptor, RegistrationId, SlabId, SourceGeneration,
            SourceIdentityHash,
        };

        let operation = OperationId::new(81);
        let source_generation = SourceGeneration::new(7);
        let destination_generation = DestinationGeneration::new(9);
        let key = MaterializationKey::new(
            ModelInstanceId::new(1),
            SourceIdentityHash::new([1; 32]),
            ContentHash::new([2; 32]),
            MaterializedResourceId::routed_expert(LayerId::new(3), ProtocolExpertId::new(4)),
            PayloadEncodingId::new(5),
            BackendId::new(6),
            DeviceId::new(0),
            source_generation,
            destination_generation,
        )
        .unwrap();
        let binding = ResidencyBinding::new(
            ModelInstanceId::new(1),
            MaterializedResourceId::routed_expert(LayerId::new(3), ProtocolExpertId::new(4)),
            BackendId::new(6),
            DeviceId::new(0),
            DestinationSlotId::new(2),
            destination_generation,
        );
        let slab = RegisteredPinnedAlignedSlabLeaseDescriptor::new(
            operation,
            SlabId::new(11),
            RegistrationId::new(12),
            4096,
            4096,
            0,
            4096,
            4096,
            source_generation,
            destination_generation,
        )
        .unwrap();
        let fence = UploadFenceContract::new(operation, FenceId::new(13), destination_generation);
        let reservation =
            PhysicalMaterializationOperationReservation::new(key, binding, [slab], fence).unwrap();

        assert_eq!(reservation.key(), key);
        assert_eq!(reservation.binding(), binding);
        assert_eq!(reservation.slabs(), &[slab]);
        assert_eq!(reservation.upload_fence(), fence);

        let _query: fn(&CudaExpertUploadTicket) -> Result<bool> =
            CudaExpertUploadTicket::is_complete;
        let _drain: fn(CudaExpertUploadTicket) -> Result<CudaExpertFrame> =
            CudaExpertUploadTicket::drain_into_frame;
    }
}
