//! Model-independent CUDA routed-expert materialization.
//!
//! This module owns one prepared-generation expert source catalog and the
//! physical CUDA install authority. Runtime code owns logical demand, hard
//! credits, operation IDs, completion validation, publication, and retirement.

#![cfg(feature = "cuda")]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use ferrule_backend::cuda::context::{
    CudaPreparedRoutedExpert, CudaRoutedExpertArena, CudaRoutedExpertMaterialization,
    CudaRoutedExpertShape,
};
use ferrule_common::materialization_io::{
    MaterializationResourceLimits, MaterializationResourcePlan,
};
use ferrule_common::{
    CancellationReason, CompletionEvent, CompletionGeneration, CompletionOutcome,
    CompletionTimestamp, DestinationGeneration, DestinationSlotId, Error,
    ExpertInstallActivationOutcome, ExpertInstallIntent, ExpertInstallPrepareOutcome,
    ExpertInstallReason, ExpertKey, ExpertLease, ExpertResidencyControl, ExpertResidencyStats,
    ExpertSlotBinding, FailureReason, FenceId, LoadStage, MaterializationKey,
    MaterializationPurpose, OperationId, PreparedExpertInstall, ResidencyBinding,
    ResidencyLeaseSet, Result, StaleReason, UploadFenceContract, ValidatedResidencyBinding,
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

#[derive(Clone)]
pub(crate) struct CudaSharedExpertSubsystem {
    inner: Arc<Mutex<CudaExpertSubsystemState>>,
}

impl std::fmt::Debug for CudaSharedExpertSubsystem {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.lock();
        formatter
            .debug_struct("CudaSharedExpertSubsystem")
            .field("has_residency_control", &state.residency.is_some())
            .field("resident_experts", &state.experts.len())
            .field("slot_tables", &state.tables.len())
            .field("poisoned_layers", &state.poisoned_layers)
            .finish()
    }
}

impl CudaSharedExpertSubsystem {
    fn new(
        tables: BTreeMap<usize, ferrule_backend::cuda::context::CudaExpertSlotTable>,
        expert_capacity: usize,
        expert_arenas: BTreeMap<CudaRoutedExpertShapeKey, CudaRoutedExpertArena>,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CudaExpertSubsystemState {
                residency: None,
                tables,
                expert_capacity,
                expert_arenas,
                experts: BTreeMap::new(),
                free_frames: Vec::new(),
                poisoned_layers: BTreeSet::new(),
                resident_keys: HashMap::new(),
            })),
        }
    }

    fn lock(&self) -> MutexGuard<'_, CudaExpertSubsystemState> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn install_residency_control(
        &self,
        control: Box<dyn ExpertResidencyControl>,
    ) -> Result<()> {
        let mut state = self.lock();
        if state.residency.is_some() {
            return Err(Error::Execution {
                message: "routed-expert physical residency control is already installed".into(),
            });
        }
        state.residency = Some(control);
        Ok(())
    }

    pub(crate) fn resident_stats_for_layer(&self, layer: usize) -> (usize, u64) {
        let state = self.lock();
        state
            .experts
            .iter()
            .filter(|(expert, _)| expert.layer == layer)
            .fold((0usize, 0u64), |(count, bytes), (_, frame)| {
                (
                    count.saturating_add(1),
                    bytes.saturating_add(frame.physical_bytes()),
                )
            })
    }

    pub(crate) fn resident_experts_for_layer(&self, layer: usize) -> Result<BTreeSet<usize>> {
        let state = self.lock();
        if state.poisoned_layers.contains(&layer) {
            return Err(Error::Execution {
                message: format!("published expert layer {layer} is poisoned"),
            });
        }
        if !state.tables.contains_key(&layer) {
            return Err(Error::Execution {
                message: format!("published expert table is missing layer {layer}"),
            });
        }
        Ok(state
            .experts
            .keys()
            .filter_map(|expert| (expert.layer == layer).then_some(expert.expert))
            .collect())
    }

    pub(crate) fn layer_slot_capacity(&self, layer: usize) -> Result<usize> {
        let state = self.lock();
        if state.poisoned_layers.contains(&layer) {
            return Err(Error::Execution {
                message: format!("published expert layer {layer} is poisoned"),
            });
        }
        state
            .tables
            .get(&layer)
            .map(|table| table.host().slot_capacity())
            .ok_or_else(|| Error::Execution {
                message: format!("published expert table is missing layer {layer}"),
            })
    }

    /// Execute against one published layer table after proving that the runtime
    /// lease set exactly covers the selected expert window. No frame, pointer, or
    /// table escapes this owner and unpublished reservations are never visible.
    pub(crate) fn with_validated_published_experts<T>(
        &self,
        layer: usize,
        selected: &[usize],
        leases: &ResidencyLeaseSet,
        execute: impl FnOnce(
            &ferrule_backend::cuda::context::CudaExpertSlotTable,
            &CudaExpertFrame,
        ) -> Result<T>,
    ) -> Result<T> {
        let state = self.lock();
        if state.poisoned_layers.contains(&layer) {
            return Err(Error::Execution {
                message: format!("published expert layer {layer} is poisoned"),
            });
        }
        let table = state.tables.get(&layer).ok_or_else(|| Error::Execution {
            message: format!("published expert table is missing layer {layer}"),
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
            let local_slot = u32::try_from(physical.slot).ok()?;
            Some(PublishedExpertBinding {
                key,
                slot: device_expert_slot_id(layer, local_slot, state.expert_capacity).ok()?,
                generation: DestinationGeneration::new(physical.generation.try_into().ok()?),
                frame_published: state.experts.contains_key(&expert),
            })
        })?;

        let first = ExpertId::new(layer, selected[0]);
        let frame = state.experts.get(&first).ok_or_else(|| Error::Execution {
            message: format!(
                "published expert frame is unavailable for {}:{}",
                first.layer, first.expert
            ),
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
        return Err(Error::Execution {
            message: format!(
                "expert lease window mismatch: selected={} leases={}",
                selected.len(),
                leases.len()
            ),
        });
    }
    let layer_u32 = u32::try_from(layer).map_err(|_| Error::Execution {
        message: "expert layer exceeds u32 ABI".into(),
    })?;
    let mut selected_set = BTreeSet::new();
    for &expert_index in selected {
        if !selected_set.insert(expert_index) {
            return Err(Error::Execution {
                message: format!(
                    "routed-expert selected expert window contains duplicate {layer}:{expert_index}"
                ),
            });
        }
    }
    for candidate in leases.bindings() {
        let key = candidate.key();
        let (key_layer, key_expert) =
            key.resource()
                .routed_expert_coordinates()
                .ok_or_else(|| Error::Execution {
                    message: "expert lease window received a non-routed-expert resource".into(),
                })?;
        let expert_index = usize::try_from(key_expert.get()).map_err(|_| Error::Execution {
            message: "expert index exceeds usize".into(),
        })?;
        if key_layer.get() != layer_u32 || !selected_set.contains(&expert_index) {
            return Err(Error::Execution {
                message: format!(
                    "routed-expert lease binding {}:{} is outside the selected expert window",
                    key_layer.get(),
                    key_expert.get()
                ),
            });
        }
    }
    for &expert_index in selected {
        let expert_u32 = u32::try_from(expert_index).map_err(|_| Error::Execution {
            message: "expert index exceeds u32 ABI".into(),
        })?;
        let mut candidates = leases.bindings().iter().filter(|candidate| {
            candidate
                .key()
                .resource()
                .routed_expert_coordinates()
                .is_some_and(|(key_layer, key_expert)| {
                    key_layer.get() == layer_u32 && key_expert.get() == expert_u32
                })
        });
        let candidate = candidates.next().ok_or_else(|| Error::Execution {
            message: format!(
                "routed-expert selected expert {layer}:{expert_index} has no lease binding"
            ),
        })?;
        if candidates.next().is_some() {
            return Err(Error::Execution {
                message: format!(
                    "routed-expert selected expert {layer}:{expert_index} has duplicate lease bindings"
                ),
            });
        }
        let published = published(expert_index).ok_or_else(|| Error::Execution {
            message: format!("expert {layer}:{expert_index} is not published"),
        })?;
        let binding = candidate.binding();
        if !published.frame_published
            || candidate.key() != published.key
            || binding.slot != published.slot
            || binding.generation != published.generation
        {
            return Err(Error::Execution {
                message: format!(
                    "expert {layer}:{expert_index} lease does not match the published key/slot/generation"
                ),
            });
        }
    }
    Ok(())
}

type ResidentExpertKeyIndex = HashMap<ExpertKey, MaterializationKey>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CudaRoutedExpertShapeKey {
    input: usize,
    intermediate: usize,
    output: usize,
}

impl From<CudaRoutedExpertShape> for CudaRoutedExpertShapeKey {
    fn from(shape: CudaRoutedExpertShape) -> Self {
        Self {
            input: shape.input,
            intermediate: shape.intermediate,
            output: shape.output,
        }
    }
}

struct CudaExpertSubsystemState {
    residency: Option<Box<dyn ExpertResidencyControl>>,
    tables: BTreeMap<usize, ferrule_backend::cuda::context::CudaExpertSlotTable>,
    expert_capacity: usize,
    expert_arenas: BTreeMap<CudaRoutedExpertShapeKey, CudaRoutedExpertArena>,
    experts: BTreeMap<ExpertId, CudaExpertFrame>,
    free_frames: Vec<CudaExpertFrame>,
    poisoned_layers: BTreeSet<usize>,
    resident_keys: ResidentExpertKeyIndex,
}

pub(crate) struct CudaExpertFrame {
    expert: CudaPreparedRoutedExpert,
    logical_payload_bytes: u64,
}

impl CudaExpertFrame {
    fn matches(&self, shape: CudaRoutedExpertShape) -> bool {
        self.expert.matches(shape)
    }

    pub(crate) fn input_size(&self) -> usize {
        self.expert.shape().input
    }

    pub(crate) fn intermediate_size(&self) -> usize {
        self.expert.shape().intermediate
    }

    pub(crate) fn output_size(&self) -> usize {
        self.expert.shape().output
    }

    fn physical_bytes(&self) -> u64 {
        self.expert.physical_bytes() as u64
    }

    fn expert_slot_pointers(
        &self,
    ) -> Result<ferrule_backend::cuda::context::CudaExpertSlotPointers> {
        self.expert.expert_slot_pointers()
    }
}

struct SlotReservation {
    request: MaterializationRequest,
    expert: ExpertId,
    read_plan: CheckpointReadPlan,
    pinned_plan: PinnedExpertLoadPlan,
    resource_plan: MaterializationResourcePlan,
    prepared: PreparedExpertInstall,
    purpose: MaterializationPurpose,
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
    purpose: MaterializationPurpose,
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
            return Err(FailureReason::ContractViolation {
                message: "selected expert already has physical execution ownership".into(),
            });
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
        let ownership =
            self.ownership
                .get_mut(&key)
                .ok_or_else(|| FailureReason::ContractViolation {
                    message: "prepared selected expert has no physical execution ownership".into(),
                })?;
        if ownership.operation.is_some() || matches!(ownership.state, SelectedLeaseState::Active(_))
        {
            return Err(FailureReason::ContractViolation {
                message: "selected expert execution ownership already has an operation".into(),
            });
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
            return Err(FailureReason::ContractViolation {
                message: "selected resident expert already has physical execution ownership".into(),
            });
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
        let ownership =
            self.ownership
                .get_mut(&key)
                .ok_or_else(|| FailureReason::ContractViolation {
                    message: "selected expert publication has no pending execution ownership"
                        .into(),
                })?;
        if ownership.operation != Some(operation) {
            return Err(FailureReason::ContractViolation {
                message: "selected expert publication operation is stale".into(),
            });
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
            SelectedLeaseState::Active(_) => Err(FailureReason::ContractViolation {
                message: "selected expert publication replaced an active execution lease".into(),
            }),
        }
    }

    fn retain_execution(
        &mut self,
        key: MaterializationKey,
    ) -> std::result::Result<bool, FailureReason> {
        let Some(ownership) = self.ownership.get_mut(&key) else {
            return Ok(false);
        };
        if matches!(ownership.state, SelectedLeaseState::ReleaseRequested) {
            ownership.state = SelectedLeaseState::Pending;
        }
        Ok(true)
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
    CudaExpertInstallTicket,
>;

enum MaterializationOperationState<Read, Host, Upload, Frame, Install> {
    Reserved(Read),
    ReadSubmitted(Read),
    HostReady(Host),
    UploadSubmitted(Upload),
    UploadReady(Frame),
    InstallQueued(Frame),
    Installing(Install),
}

impl<Read, Host, Upload, Frame, Install>
    MaterializationOperationState<Read, Host, Upload, Frame, Install>
{
    const fn owner_stage(&self) -> LoadStage {
        match self {
            Self::Reserved(_) => LoadStage::Reserved,
            Self::ReadSubmitted(_) => LoadStage::ReadSubmitted,
            Self::HostReady(_) => LoadStage::HostReady,
            Self::UploadSubmitted(_) => LoadStage::UploadSubmitted,
            Self::UploadReady(_) | Self::InstallQueued(_) | Self::Installing(_) => {
                LoadStage::Installing
            }
        }
    }

    const fn pollable(&self) -> bool {
        matches!(
            self,
            Self::ReadSubmitted(_)
                | Self::UploadSubmitted(_)
                | Self::InstallQueued(_)
                | Self::Installing(_)
        )
    }
}

fn cancel_observation_matches_physical_state<Read, Host, Upload, Frame, Install>(
    observed: LoadStage,
    physical: &MaterializationOperationState<Read, Host, Upload, Frame, Install>,
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
    weight: ferrule_backend::cuda::context::CudaPinnedU8HostBuffer,
    scale: ferrule_backend::cuda::context::CudaPinnedU8HostBuffer,
}

struct PinnedExpertBundle {
    expert: ExpertId,
    gate: PinnedExpertLinear,
    up: PinnedExpertLinear,
    down: PinnedExpertLinear,
    bytes: u64,
}

struct CudaExpertUploadTicket {
    materialization: CudaRoutedExpertMaterialization,
    frame: Option<CudaExpertFrame>,
}

impl CudaExpertUploadTicket {
    fn is_complete(&self) -> Result<bool> {
        self.materialization.is_complete()
    }

    fn drain_into_frame(mut self) -> Result<CudaExpertFrame> {
        self.materialization.synchronize()?;
        self.frame.take().ok_or_else(|| Error::Internal {
            message: "CUDA expert upload lost its frame".into(),
        })
    }
}

impl Drop for CudaExpertUploadTicket {
    fn drop(&mut self) {
        if self.frame.is_some()
            && !matches!(self.materialization.is_complete(), Ok(true))
            && self.materialization.synchronize().is_err()
            && let Some(frame) = self.frame.take()
        {
            std::mem::forget(frame);
        }
    }
}

struct CudaExpertInstallTicket {
    frame: CudaExpertFrame,
    physical: ferrule_backend::cuda::context::CudaExpertSlotInstallTicket,
    eviction: Option<(ExpertId, ExpertSlotBinding, MaterializationKey)>,
}

impl CudaExpertInstallTicket {
    fn is_complete(&self) -> Result<bool> {
        self.physical.is_complete()
    }
}

/// Runner-owned factory result for the single CUDA materialization authority.
///
/// The operator cache receives the expert execution handle while the provider is
/// transferred exactly once to the runtime registry. Both point at the same
/// frame/table/residency state.
pub(crate) struct CudaExpertMaterializationOwner {
    placement: MaterializationPlacement,
    shared: CudaSharedExpertSubsystem,
    provider: Option<CudaExpertMaterializationProvider>,
}

pub struct CudaExpertMaterializationProvider {
    placement: MaterializationPlacement,
    topology: PhysicalMaterializationTopology,
    expert_capacity: usize,
    sources: Arc<MaterializationSourceCatalog<ExpertLoadSource>>,
    reader: ExpertStreamingReader,
    ops: ferrule_backend::cuda::context::CudaArtifactOperatorContext,
    consumer_compute: ferrule_backend::cuda::context::CudaComputeStreamAuthority,
    completion_hub: ferrule_common::CompletionHub,
    shared: CudaSharedExpertSubsystem,
    request_keys: BTreeMap<MaterializationRequest, MaterializationKey>,
    resource_plans: BTreeMap<MaterializationKey, MaterializationResourcePlan>,
    selected_lease_ownership: SelectedLeaseTracker,
    reservations: BTreeMap<MaterializationKey, SlotReservation>,
    operations: BTreeMap<OperationId, MaterializationOperation>,
    execution_order: VecDeque<OperationId>,
    prefetch_order: VecDeque<OperationId>,
    seen_operations: HashSet<OperationId>,
    terminal_operations: HashSet<OperationId>,
    completions: VecDeque<CompletionEvent>,
    clock_ns: u64,
}

impl std::fmt::Debug for CudaExpertMaterializationProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaExpertMaterializationProvider")
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

impl CudaExpertMaterializationOwner {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create(
        placement: MaterializationPlacement,
        limits: MaterializationResourceLimits,
        sources: Arc<MaterializationSourceCatalog<ExpertLoadSource>>,
        reader: ExpertStreamingReader,
        expert_capacity: usize,
        layer_slot_capacities: &[(usize, usize)],
        consumer_compute: ferrule_backend::cuda::context::CudaComputeStreamAuthority,
    ) -> Result<Self> {
        let provider = CudaExpertMaterializationProvider::new(
            placement,
            limits,
            sources,
            reader,
            expert_capacity,
            layer_slot_capacities,
            consumer_compute,
        )?;
        let shared = provider.shared.clone();
        Ok(Self {
            placement,
            shared,
            provider: Some(provider),
        })
    }

    pub(crate) const fn placement(&self) -> MaterializationPlacement {
        self.placement
    }

    pub(crate) fn handle(&self) -> CudaSharedExpertSubsystem {
        self.shared.clone()
    }

    pub(crate) fn residency_control_installed(&self) -> bool {
        self.shared.lock().residency.is_some()
    }

    pub(crate) fn install_residency_control(
        &self,
        control: Box<dyn ExpertResidencyControl>,
    ) -> Result<()> {
        if control.requirements().model_instance != self.placement.model().get() {
            return Err(Error::Execution {
                message: "routed-expert physical residency model namespace mismatch".into(),
            });
        }
        self.shared.install_residency_control(control)
    }

    pub(crate) fn residency_stats(&self) -> ExpertResidencyStats {
        self.shared
            .lock()
            .residency
            .as_ref()
            .map_or_else(ExpertResidencyStats::default, |control| control.stats())
    }

    pub(crate) fn take_provider(&mut self) -> Option<Box<dyn MaterializationProvider>> {
        self.provider
            .take()
            .map(|provider| Box::new(provider) as Box<dyn MaterializationProvider>)
    }
}

impl CudaExpertMaterializationProvider {
    #[allow(clippy::too_many_arguments)]
    fn new(
        placement: MaterializationPlacement,
        limits: MaterializationResourceLimits,
        sources: Arc<MaterializationSourceCatalog<ExpertLoadSource>>,
        reader: ExpertStreamingReader,
        expert_capacity: usize,
        layer_slot_capacities: &[(usize, usize)],
        consumer_compute: ferrule_backend::cuda::context::CudaComputeStreamAuthority,
    ) -> Result<Self> {
        let mut limits = limits.validate()?;
        if expert_capacity == 0 {
            return Err(Error::Model {
                message: "expert capacity must be non-zero".into(),
            });
        }
        let ops = ferrule_backend::cuda::context::CudaArtifactOperatorContext::new()?;
        let mut routed_layer_shapes = BTreeMap::<usize, CudaRoutedExpertShape>::new();
        let mut routed_layer_raw_bytes = BTreeMap::<usize, u64>::new();
        for entry in sources.iter() {
            let source = entry.descriptor();
            let (layer, _) = entry
                .resource()
                .routed_expert_coordinates()
                .ok_or_else(|| Error::Model {
                    message: "routed-expert source is bound to a non-expert resource".into(),
                })?;
            let layer = usize::try_from(layer.get()).map_err(|_| Error::Model {
                message: "routed-expert layer exceeds usize".into(),
            })?;
            let raw_bytes = source.bytes();
            if raw_bytes == 0 || raw_bytes != entry.read_plan().storage_bytes() {
                return Err(Error::Model {
                    message: format!(
                        "routed-expert source/read size mismatch at layer {layer}: source={raw_bytes} read={}",
                        entry.read_plan().storage_bytes()
                    ),
                });
            }
            let shape = source_routed_expert_shape(source)?;
            if let Some(existing) = routed_layer_shapes.insert(layer, shape)
                && existing != shape
            {
                return Err(Error::Model {
                    message: format!("routed-expert shape mismatch within layer {layer}"),
                });
            }
            if let Some(existing) = routed_layer_raw_bytes.insert(layer, raw_bytes)
                && existing != raw_bytes
            {
                return Err(Error::Model {
                    message: format!("routed-expert raw byte mismatch within layer {layer}"),
                });
            }
        }

        let mut tables = BTreeMap::new();
        let mut resident_capacity_by_shape =
            BTreeMap::<CudaRoutedExpertShapeKey, (CudaRoutedExpertShape, usize)>::new();
        let mut resident_frame_bytes = 0u64;
        let mut maximum_physical_expansion = 0u64;
        let mut residency_lease_slots_per_continuation = 0u64;
        for &(layer, slots) in layer_slot_capacities {
            if slots == 0 || slots > expert_capacity || tables.contains_key(&layer) {
                return Err(Error::Model {
                    message: format!(
                        "invalid or duplicate routed-expert physical expert layer capacity {layer}:{slots}"
                    ),
                });
            }
            let maximum_local_slot = u32::try_from(slots - 1).map_err(|_| Error::Model {
                message: format!(
                    "routed-expert physical expert slot capacity exceeds protocol ABI at layer {layer}"
                ),
            })?;
            device_expert_slot_id(layer, maximum_local_slot, expert_capacity).map_err(|reason| {
                Error::Model {
                    message: format!(
                        "routed-expert device expert slot namespace is invalid at layer {layer}: {reason}"
                    ),
                }
            })?;
            let shape = routed_layer_shapes.get(&layer).copied().ok_or_else(|| {
                Error::Model { message: format!(
                    "routed-expert CUDA materialization catalog has no routed experts for layer {layer}"
                ) }
            })?;
            let layer_frame_bytes =
                u64::try_from(shape.physical_bytes()?).map_err(|_| Error::Model {
                    message: "physical expert frame storage exceeds u64".into(),
                })?;
            let slots_u64 = u64::try_from(slots).map_err(|_| Error::Model {
                message: "physical expert slot capacity exceeds u64".into(),
            })?;
            resident_frame_bytes = resident_frame_bytes
                .checked_add(layer_frame_bytes.checked_mul(slots_u64).ok_or_else(|| {
                    Error::Model {
                        message: "physical expert resident byte capacity overflow".into(),
                    }
                })?)
                .ok_or_else(|| Error::Model {
                    message: "physical expert resident byte capacity overflow".into(),
                })?;
            let layer_raw_bytes = routed_layer_raw_bytes.get(&layer).copied().ok_or_else(|| {
                Error::Model {
                    message: format!(
                        "routed-expert CUDA materialization catalog has no raw storage for layer {layer}"
                    ),
                }
            })?;
            maximum_physical_expansion = maximum_physical_expansion.max(
                layer_frame_bytes
                    .checked_sub(layer_raw_bytes)
                    .ok_or_else(|| Error::Model {
                        message: "physical expert storage is smaller than its source".into(),
                    })?,
            );
            residency_lease_slots_per_continuation =
                residency_lease_slots_per_continuation.max(slots_u64);
            let shape_key = shape.into();
            if let Some((_, capacity)) = resident_capacity_by_shape.get_mut(&shape_key) {
                *capacity = capacity.checked_add(slots).ok_or_else(|| Error::Model {
                    message: "physical expert arena frame capacity overflow".into(),
                })?;
            } else {
                resident_capacity_by_shape.insert(shape_key, (shape, slots));
            }
            tables.insert(layer, ops.expert_slot_table(expert_capacity, slots)?);
        }
        if let Some(layer) = routed_layer_shapes
            .keys()
            .find(|layer| !tables.contains_key(layer))
        {
            return Err(Error::Model {
                message: format!(
                    "routed-expert CUDA materialization catalog has no slot capacity for routed-expert layer {layer}"
                ),
            });
        }
        let shadow =
            usize::try_from(limits.capacity.upload_slots.max(1)).map_err(|_| Error::Model {
                message: "physical expert upload slots exceed usize".into(),
            })?;
        let shadow_u64 = u64::try_from(shadow).map_err(|_| Error::Model {
            message: "physical expert shadow capacity exceeds u64".into(),
        })?;
        let mut resident_capacity_bytes = resident_frame_bytes;
        let mut expert_arenas = BTreeMap::new();
        for (shape_key, (shape, resident_frames)) in resident_capacity_by_shape {
            let frame_capacity =
                resident_frames
                    .checked_add(shadow)
                    .ok_or_else(|| Error::Model {
                        message: "physical expert arena frame capacity overflow".into(),
                    })?;
            let shadow_bytes = u64::try_from(shape.physical_bytes()?)
                .map_err(|_| Error::Model {
                    message: "physical expert frame storage exceeds u64".into(),
                })?
                .checked_mul(shadow_u64)
                .ok_or_else(|| Error::Model {
                    message: "physical expert shadow byte capacity overflow".into(),
                })?;
            resident_capacity_bytes = resident_capacity_bytes
                .checked_add(shadow_bytes)
                .ok_or_else(|| Error::Model {
                    message: "physical expert frame byte capacity overflow".into(),
                })?;
            expert_arenas.insert(
                shape_key,
                ops.allocate_routed_expert_arena(shape, frame_capacity)?,
            );
        }
        let capacity_expansion = maximum_physical_expansion
            .checked_mul(limits.capacity.install_slots)
            .ok_or_else(|| Error::Model {
                message: "physical expert device install capacity overflow".into(),
            })?;
        limits.capacity.device_install_bytes = limits
            .capacity
            .device_install_bytes
            .checked_add(capacity_expansion)
            .ok_or_else(|| Error::Model {
                message: "physical expert device install capacity overflow".into(),
            })?;
        let reserve_expansion = maximum_physical_expansion
            .checked_mul(limits.execution_reserve.install_slots)
            .ok_or_else(|| Error::Model {
                message: "physical expert execution reserve overflow".into(),
            })?;
        limits.execution_reserve.device_install_bytes = limits
            .execution_reserve
            .device_install_bytes
            .checked_add(reserve_expansion)
            .ok_or_else(|| Error::Model {
                message: "physical expert execution reserve overflow".into(),
            })?;
        let limits = limits.validate()?;
        let topology = PhysicalMaterializationTopology::new(
            limits,
            resident_capacity_bytes,
            residency_lease_slots_per_continuation,
        )?;
        let completion_hub = reader.completion_hub();
        Ok(Self {
            placement,
            topology,
            expert_capacity,
            sources,
            reader,
            ops,
            consumer_compute,
            completion_hub,
            shared: CudaSharedExpertSubsystem::new(tables, expert_capacity, expert_arenas),
            request_keys: BTreeMap::new(),
            resource_plans: BTreeMap::new(),
            selected_lease_ownership: SelectedLeaseTracker::default(),
            reservations: BTreeMap::new(),
            operations: BTreeMap::new(),
            execution_order: VecDeque::new(),
            prefetch_order: VecDeque::new(),
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
            return Err(FailureReason::ContractViolation {
                message: "routed-expert request placement mismatch".into(),
            });
        }
        let entry = self.sources.resolve(request)?;
        let install_source = entry.descriptor().clone();
        let (protocol_layer, protocol_expert) =
            request.routed_expert_coordinates().ok_or_else(|| {
                FailureReason::ContractViolation {
                    message: "routed-expert descriptor is bound to a non-expert resource".into(),
                }
            })?;
        let layer = usize::try_from(protocol_layer.get()).map_err(|_| {
            FailureReason::ContractViolation {
                message: "expert layer exceeds usize".into(),
            }
        })?;
        let expert_index = usize::try_from(protocol_expert.get()).map_err(|_| {
            FailureReason::ContractViolation {
                message: "expert index exceeds usize".into(),
            }
        })?;
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
        key.validate()
            .map_err(|error| FailureReason::ContractViolation {
                message: format!("invalid physical load key: {error}"),
            })?;
        let request = self
            .request_keys
            .iter()
            .find_map(|(request, candidate)| (*candidate == key).then_some(*request))
            .ok_or_else(|| FailureReason::ContractViolation {
                message: "physical MaterializationKey cannot be recovered to a source request"
                    .into(),
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
            return Err(FailureReason::ContractViolation {
                message: "physical expert slot generation is zero".into(),
            });
        }
        let (layer, _) = request.routed_expert_coordinates().ok_or_else(|| {
            FailureReason::ContractViolation {
                message: "physical expert slot is bound to a non-routed-expert resource".into(),
            }
        })?;
        let layer = usize::try_from(layer.get()).map_err(|_| FailureReason::ContractViolation {
            message: "expert layer exceeds usize".into(),
        })?;
        Ok(ResidencyBinding::new(
            request.model(),
            request.resource(),
            request.backend(),
            request.device(),
            device_expert_slot_id(layer, slot.slot.get(), self.expert_capacity)?,
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

    fn enqueue_operation(&mut self, operation: OperationId, purpose: MaterializationPurpose) {
        match purpose {
            MaterializationPurpose::Execution => self.execution_order.push_back(operation),
            MaterializationPurpose::Prefetch => self.prefetch_order.push_back(operation),
        }
    }

    fn remove_operation_order(&mut self, operation: OperationId) {
        self.execution_order
            .retain(|candidate| *candidate != operation);
        self.prefetch_order
            .retain(|candidate| *candidate != operation);
    }

    fn promote_operation_order(&mut self, operation: OperationId) {
        self.remove_operation_order(operation);
        if self
            .operations
            .get(&operation)
            .and_then(|operation| operation.state.as_ref())
            .is_some_and(MaterializationOperationState::pollable)
        {
            self.execution_order.push_back(operation);
        }
    }

    fn cancel_prepared(&self, prepared: PreparedExpertInstall) -> Result<()> {
        let mut shared = self.shared.lock();
        shared
            .residency
            .as_mut()
            .ok_or_else(|| Error::Execution {
                message: "physical residency control is not installed".into(),
            })?
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
            Err(error) => FailureReason::ContractViolation {
                message: format!(
                    "physical reservation failed ({primary:?}); prepared slot rollback also failed ({error})"
                ),
            },
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
        frame.logical_payload_bytes = 0;
        self.shared.lock().free_frames.push(frame);
    }

    fn allocate_frame(&self, bundle: &PinnedExpertBundle) -> Result<CudaExpertFrame> {
        let shape = pinned_routed_expert_shape(bundle)?;
        let mut shared = self.shared.lock();
        if let Some(index) = shared
            .free_frames
            .iter()
            .rposition(|frame| frame.matches(shape))
        {
            let mut frame = shared.free_frames.swap_remove(index);
            frame.logical_payload_bytes = bundle.bytes;
            return Ok(frame);
        }
        let arena = shared
            .expert_arenas
            .get_mut(&shape.into())
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "physical expert arena is missing shape input={} intermediate={} output={}",
                    shape.input, shape.intermediate, shape.output
                ),
            })?;
        let expert = arena.allocate_frame().map_err(|error| Error::Execution {
            message: format!(
                "routed-expert physical expert frame pool exhausted for input={} intermediate={} output={}: {error}",
                shape.input, shape.intermediate, shape.output
            ),
        })?;
        Ok(CudaExpertFrame {
            expert,
            logical_payload_bytes: bundle.bytes,
        })
    }

    fn submit_bundle_upload(&self, bundle: PinnedExpertBundle) -> Result<CudaExpertUploadTicket> {
        let mut frame = self.allocate_frame(&bundle)?;
        let PinnedExpertBundle {
            gate,
            up,
            down,
            bytes: _,
            expert: _,
        } = bundle;
        let submitted = (|| {
            let materialization = self.ops.materialize_routed_expert_from_pinned_async(
                &mut frame.expert,
                gate.weight,
                gate.scale,
                up.weight,
                up.scale,
                down.weight,
                down.scale,
            )?;
            self.ops
                .notify_upload_stream(completion_notify_callback(self.completion_hub.clone()))?;
            Ok(materialization)
        })();
        match submitted {
            Ok(materialization) => Ok(CudaExpertUploadTicket {
                materialization,
                frame: Some(frame),
            }),
            Err(error) => {
                self.recycle_frame(frame);
                Err(error)
            }
        }
    }

    fn operation_identity_outcome(
        operation: &MaterializationOperation,
        expert_capacity: usize,
    ) -> Option<CompletionOutcome> {
        if let Err(error) = operation.request.validate_key(operation.key) {
            return Some(CompletionOutcome::Failed(protocol_failure(error)));
        }
        if let Err(error) = ValidatedResidencyBinding::new(operation.key, operation.binding) {
            return Some(CompletionOutcome::Failed(
                FailureReason::ContractViolation {
                    message: error.to_string(),
                },
            ));
        }
        let Some(prepared) = operation.prepared else {
            return Some(CompletionOutcome::Failed(
                FailureReason::ContractViolation {
                    message: "physical operation lost prepared slot ownership".into(),
                },
            ));
        };
        let expected = prepared.binding();
        let Some((layer, expert)) = operation.request.routed_expert_coordinates() else {
            return Some(CompletionOutcome::Failed(
                FailureReason::ContractViolation {
                    message: "expert operation received a non-routed-expert resource".into(),
                },
            ));
        };
        let expected_key =
            ExpertKey::new(operation.request.model().get(), layer.get(), expert.get());
        let protocol_slot = match device_expert_slot_id(
            operation.expert.layer,
            expected.slot.get(),
            expert_capacity,
        ) {
            Ok(slot) => slot,
            Err(reason) => return Some(CompletionOutcome::Failed(reason)),
        };
        if expected.key != expected_key
            || protocol_slot != operation.binding.slot
            || u64::from(expected.generation.get()) != operation.key.destination_generation().get()
            || operation.expert.layer != usize::try_from(layer.get()).unwrap_or(usize::MAX)
            || operation.expert.expert != usize::try_from(expert.get()).unwrap_or(usize::MAX)
        {
            return Some(CompletionOutcome::Failed(
                FailureReason::ContractViolation {
                    message:
                        "prepared slot identity/generation diverged from the canonical operation"
                            .into(),
                },
            ));
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

    fn submit_physical_install(
        &mut self,
        operation: &MaterializationOperation,
        frame: CudaExpertFrame,
    ) -> std::result::Result<CudaExpertInstallTicket, CompletionOutcome> {
        if let Some(outcome) = Self::operation_identity_outcome(operation, self.expert_capacity) {
            self.recycle_frame(frame);
            return Err(outcome);
        }
        let Some(prepared) = operation.prepared else {
            self.recycle_frame(frame);
            return Err(CompletionOutcome::Failed(
                FailureReason::InstallationRejected,
            ));
        };
        let expected = prepared.binding();
        let pointers = match frame.expert_slot_pointers() {
            Ok(pointers) => pointers,
            Err(error) => {
                self.recycle_frame(frame);
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
            let table =
                shared
                    .tables
                    .get(&operation.expert.layer)
                    .ok_or_else(|| Error::Internal {
                        message: format!(
                            "physical expert slot table missing layer {}",
                            operation.expert.layer
                        ),
                    })?;
            if let Some(evicted_key) = prepared.evicted_key() {
                let evicted =
                    ExpertId::new(evicted_key.layer as usize, evicted_key.expert as usize);
                if evicted.layer != operation.expert.layer {
                    return Err(Error::Internal {
                        message: "prepared eviction crossed physical expert layers".into(),
                    });
                }
                let old = prepared.evicted_binding().ok_or_else(|| Error::Internal {
                    message: "prepared eviction lost transaction binding".into(),
                })?;
                let physical =
                    table
                        .host()
                        .binding(evicted.expert)
                        .ok_or_else(|| Error::Internal {
                            message: "prepared eviction lost physical table binding".into(),
                        })?;
                let old_key = shared
                    .resident_keys
                    .get(&evicted_key)
                    .copied()
                    .ok_or_else(|| Error::Internal {
                        message: "prepared eviction lost exact resident MaterializationKey".into(),
                    })?;
                if old.slot != expected.slot
                    || old.generation.get().checked_add(1) != Some(expected.generation.get())
                    || physical.slot != i32::try_from(old.slot.get()).unwrap_or(-1)
                    || physical.generation != i32::try_from(old.generation.get()).unwrap_or(-1)
                    || old_key.destination_generation().get() != u64::from(old.generation.get())
                    || !shared.experts.contains_key(&evicted)
                {
                    return Err(Error::Execution {
                        message: "prepared eviction source/slot generation is stale".into(),
                    });
                }
                Ok(Some((evicted, old, old_key)))
            } else {
                if table.host().binding(operation.expert.expert).is_some()
                    || shared.experts.contains_key(&operation.expert)
                {
                    return Err(Error::Execution {
                        message: "prepared empty slot already has a physical expert owner".into(),
                    });
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
        let target = if let Some((evicted, old, _)) = eviction {
            let consumer_quiescence = match self.consumer_compute.record_event() {
                Ok(event) => event,
                Err(error) => {
                    shared.free_frames.push(frame);
                    return Err(CompletionOutcome::Failed(protocol_failure(error)));
                }
            };
            ferrule_backend::cuda::context::CudaExpertSlotInstallTarget::Replacement {
                previous_expert: evicted.expert,
                previous_binding: ferrule_backend::cuda::context::CudaExpertSlotBinding {
                    slot: i32::try_from(old.slot.get()).unwrap_or(-1),
                    generation: i32::try_from(old.generation.get()).unwrap_or(-1),
                },
                consumer_quiescence,
            }
        } else {
            ferrule_backend::cuda::context::CudaExpertSlotInstallTarget::Empty
        };
        let physical = self.ops.submit_expert_slot_install(
            shared
                .tables
                .get_mut(&operation.expert.layer)
                .ok_or_else(|| Error::Internal {
                    message: "physical expert slot table disappeared".into(),
                })
                .map_err(|error| CompletionOutcome::Failed(protocol_failure(error)))?,
            target,
            operation.expert.expert,
            expected.slot.get(),
            expected.generation.get(),
            pointers,
        );
        let physical = match physical {
            Ok(physical) => physical,
            Err(error) => {
                if shared
                    .tables
                    .get(&operation.expert.layer)
                    .is_some_and(|table| table.is_poisoned())
                {
                    shared.poisoned_layers.insert(operation.expert.layer);
                }
                shared.free_frames.push(frame);
                return Err(CompletionOutcome::Failed(protocol_failure(error)));
            }
        };
        drop(shared);
        if self
            .ops
            .notify_upload_stream(completion_notify_callback(self.completion_hub.clone()))
            .is_err()
        {
            self.completion_hub.notify();
        }
        Ok(CudaExpertInstallTicket {
            frame,
            physical,
            eviction,
        })
    }

    fn complete_physical_install(
        &mut self,
        operation_id: OperationId,
        operation: &mut MaterializationOperation,
        ticket: CudaExpertInstallTicket,
    ) -> std::result::Result<ValidatedResidencyBinding, CompletionOutcome> {
        let Some(prepared) = operation.prepared else {
            self.shared
                .lock()
                .poisoned_layers
                .insert(operation.expert.layer);
            return Err(CompletionOutcome::Failed(
                FailureReason::InstallationRejected,
            ));
        };
        let binding = ValidatedResidencyBinding::new(operation.key, operation.binding)
            .map_err(|error| CompletionOutcome::Failed(protocol_failure(error)))?;
        let expected = prepared.binding();
        let CudaExpertInstallTicket {
            frame,
            physical,
            eviction,
        } = ticket;
        let mut shared = self.shared.lock();
        let physical_binding = physical
            .complete(
                shared
                    .tables
                    .get_mut(&operation.expert.layer)
                    .ok_or_else(|| Error::Internal {
                        message: "physical expert slot table disappeared".into(),
                    })
                    .map_err(|error| CompletionOutcome::Failed(protocol_failure(error)))?,
            )
            .map_err(|error| {
                shared.poisoned_layers.insert(operation.expert.layer);
                CompletionOutcome::Failed(protocol_failure(error))
            })?;
        if physical_binding.slot != i32::try_from(expected.slot.get()).unwrap_or(-1)
            || physical_binding.generation != i32::try_from(expected.generation.get()).unwrap_or(-1)
        {
            shared.poisoned_layers.insert(operation.expert.layer);
            shared.experts.insert(operation.expert, frame);
            return Err(CompletionOutcome::Failed(
                FailureReason::ContractViolation {
                    message: "completed physical slot identity differs from prepared publication"
                        .into(),
                },
            ));
        }

        let grant = match shared
            .residency
            .as_mut()
            .ok_or_else(|| Error::Execution {
                message: "physical residency control is not installed".into(),
            })
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
        let lease = grant.lease();
        let expected_reason = match operation.purpose {
            MaterializationPurpose::Execution => ExpertInstallReason::Selected,
            MaterializationPurpose::Prefetch => ExpertInstallReason::Prefetch,
        };
        if grant.binding() != expected
            || grant.reason() != expected_reason
            || (operation.purpose == MaterializationPurpose::Execution) != lease.is_some()
            || !controller_matches
        {
            if let Some(lease) = lease
                && let Some(residency) = shared.residency.as_mut()
            {
                let _ = residency.release(lease);
            }
            shared.poisoned_layers.insert(operation.expert.layer);
            shared.experts.insert(operation.expert, frame);
            return Err(CompletionOutcome::Failed(
                FailureReason::ContractViolation {
                    message: "physical/controller slot identity diverged after publication".into(),
                },
            ));
        }
        if let Some((evicted, _, old_key)) = eviction {
            if let Some(old_frame) = shared.experts.remove(&evicted) {
                shared.free_frames.push(old_frame);
            }
            shared.resident_keys.retain(|_, key| *key != old_key);
        }
        shared.experts.insert(operation.expert, frame);
        shared.resident_keys.insert(expected.key, operation.key);
        drop(shared);
        if let Some(lease) = lease {
            let publication = self
                .selected_lease_ownership
                .publish(operation.key, operation_id, lease)
                .map_err(CompletionOutcome::Failed)?;
            if publication == SelectedLeasePublication::ReleaseImmediately {
                self.release_execution_lease(operation.key)
                    .map_err(CompletionOutcome::Failed)?;
            }
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
            | Some(MaterializationOperationState::InstallQueued(frame)) => {
                self.recycle_frame(frame);
            }
            Some(MaterializationOperationState::Installing(ticket)) => {
                std::mem::forget(ticket);
                first_error = Some(Error::Internal {
                    message:
                        "attempted to clean up a submitted CUDA expert install before publication"
                            .into(),
                });
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
                    .map_err(|reason| Error::Execution {
                        message: format!(
                            "terminal selected-expert lease release failed: {reason:?}"
                        ),
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
                outcome = CompletionOutcome::Failed(protocol_failure(Error::Internal {
                    message: format!("terminal resource cleanup failed: {error}"),
                }));
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
                operation.resource_plan.requirements.device_install_bytes,
            );
        }
    }

    fn keep_operation(
        &mut self,
        operation_id: OperationId,
        mut operation: MaterializationOperation,
        state: CudaMaterializationOperationState,
    ) {
        let pollable = state.pollable();
        operation.state = Some(state);
        let purpose = operation.purpose;
        self.operations.insert(operation_id, operation);
        if pollable {
            self.enqueue_operation(operation_id, purpose);
        }
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
                    CompletionOutcome::Failed(FailureReason::ContractViolation {
                        message: "active physical operation had no resource owner".into(),
                    }),
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
                                    && bundle.bytes
                                        == operation.resource_plan.requirements.h2d_bytes =>
                            {
                                self.emit(
                                    operation_id,
                                    operation.key,
                                    LoadStage::ReadSubmitted,
                                    CompletionOutcome::Succeeded,
                                    operation.resource_plan.requirements.storage_read_bytes,
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
                                            FailureReason::ContractViolation { message:
                                                "pinned payload expert identity or byte count mismatch"
                                                    .into(),
                                             },
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
                                } else if let Some(outcome) = Self::operation_identity_outcome(
                                    &operation,
                                    self.expert_capacity,
                                ) {
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
                                        operation.resource_plan.requirements.h2d_bytes,
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
            MaterializationOperationState::InstallQueued(frame) => {
                Self::mark_source_stale(&mut operation);
                if let Some(outcome) = operation.pending_terminal.take() {
                    let _ = self.finish_terminal_operation(
                        operation_id,
                        operation,
                        Some(MaterializationOperationState::InstallQueued(frame)),
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
                            MaterializationOperationState::InstallQueued(frame),
                        );
                    }
                    Ok(ExpertInstallActivationOutcome::Activated) => {
                        match self.submit_physical_install(&operation, frame) {
                            Ok(ticket) => self.keep_operation(
                                operation_id,
                                operation,
                                MaterializationOperationState::Installing(ticket),
                            ),
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
                    Err(outcome) => {
                        let _ = self.finish_terminal_operation(
                            operation_id,
                            operation,
                            Some(MaterializationOperationState::InstallQueued(frame)),
                            Some((LoadStage::Installing, outcome, 0)),
                            true,
                        );
                    }
                }
            }
            MaterializationOperationState::Installing(ticket) => match ticket.is_complete() {
                Ok(false) => self.keep_operation(
                    operation_id,
                    operation,
                    MaterializationOperationState::Installing(ticket),
                ),
                Ok(true) => {
                    match self.complete_physical_install(operation_id, &mut operation, ticket) {
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
                Err(error) => {
                    std::mem::forget(ticket);
                    self.shared
                        .lock()
                        .poisoned_layers
                        .insert(operation.expert.layer);
                    let _ = self.finish_terminal_operation(
                        operation_id,
                        operation,
                        None,
                        Some((
                            LoadStage::Installing,
                            CompletionOutcome::Failed(protocol_failure(error)),
                            0,
                        )),
                        true,
                    );
                }
            },
        }
    }

    fn progress_ordered_operation(&mut self, execution: bool) -> Option<CompletionEvent> {
        let operation_id = if execution {
            self.execution_order.pop_front()
        } else {
            self.prefetch_order.pop_front()
        }?;
        if let Some(operation) = self.operations.remove(&operation_id) {
            self.progress_operation(operation_id, operation);
        }
        self.completions.pop_front()
    }

    fn progress_one(&mut self) -> Option<CompletionEvent> {
        if let Some(completion) = self.completions.pop_front() {
            return Some(completion);
        }

        let execution_count = self.execution_order.len();
        let prefetch_count = self.prefetch_order.len();
        for _ in 0..execution_count {
            if let Some(completion) = self.progress_ordered_operation(true) {
                return Some(completion);
            }
        }
        for _ in 0..prefetch_count {
            if let Some(completion) = self.progress_ordered_operation(false) {
                return Some(completion);
            }
        }
        None
    }
}

impl MaterializationProvider for CudaExpertMaterializationProvider {
    fn placement(&self) -> MaterializationPlacement {
        self.placement
    }

    fn resource_topology(&self) -> Result<PhysicalMaterializationTopology> {
        Ok(self.topology)
    }

    fn prepare(
        &mut self,
        request: MaterializationRequest,
        purpose: MaterializationPurpose,
    ) -> std::result::Result<MaterializationPreparation, FailureReason> {
        let (expert, read_plan, install_source) = self.source_for_request(request)?;
        read_plan
            .validate_source_identity()
            .map_err(|_| FailureReason::ContractViolation {
                message: "checkpoint source identity is already stale".into(),
            })?;
        if let Some(existing) = self.request_keys.get(&request).copied() {
            return self.prepared(existing);
        }

        let (layer, expert_index) = request.routed_expert_coordinates().ok_or_else(|| {
            FailureReason::ContractViolation {
                message: "routed-expert preparation received a non-expert resource".into(),
            }
        })?;
        let expert_key = ExpertKey::new(request.model().get(), layer.get(), expert_index.get());
        let pinned_plan = self
            .reader
            .plan_checkpoint_source_pinned(expert, &read_plan, &install_source)
            .map_err(protocol_failure)?
            .ok_or(FailureReason::StorageUnavailable)?;
        let shape = source_routed_expert_shape(&install_source).map_err(protocol_failure)?;
        let physical_bytes = shape.physical_bytes().map_err(protocol_failure)?;
        let physical_bytes =
            u64::try_from(physical_bytes).map_err(|_| FailureReason::ContractViolation {
                message: "physical expert byte size exceeds u64".into(),
            })?;
        let mut requirements = pinned_plan.requirements();
        requirements.device_install_bytes = physical_bytes;
        let resource_plan = MaterializationResourcePlan::new(requirements, physical_bytes)
            .map_err(protocol_failure)?;
        let mut shared = self.shared.lock();
        if shared.poisoned_layers.contains(&expert.layer) {
            return Err(FailureReason::InstallationRejected);
        }
        let install_intent = match purpose {
            MaterializationPurpose::Execution => ExpertInstallIntent::selected(expert_key),
            MaterializationPurpose::Prefetch => ExpertInstallIntent::prefetch(expert_key),
        };
        let outcome = shared
            .residency
            .as_mut()
            .ok_or(FailureReason::InstallationRejected)?
            .prepare_install(install_intent)
            .map_err(protocol_failure)?;
        match outcome {
            ExpertInstallPrepareOutcome::Resident(grant) => {
                let expected_reason = match purpose {
                    MaterializationPurpose::Execution => ExpertInstallReason::Selected,
                    MaterializationPurpose::Prefetch => ExpertInstallReason::Prefetch,
                };
                let lease = grant.lease();
                if grant.reason() != expected_reason
                    || (purpose == MaterializationPurpose::Execution) != lease.is_some()
                {
                    if let Some(lease) = lease
                        && let Some(residency) = shared.residency.as_mut()
                    {
                        let _ = residency.release(lease);
                    }
                    return Err(FailureReason::ContractViolation {
                        message: "expert resident grant does not match materialization purpose"
                            .into(),
                    });
                }
                let raw = match self.protocol_binding(request, grant.binding()) {
                    Ok(raw) => raw,
                    Err(reason) => {
                        if let Some(lease) = lease
                            && let Some(residency) = shared.residency.as_mut()
                        {
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
                        if let Some(lease) = lease
                            && let Some(residency) = shared.residency.as_mut()
                        {
                            let _ = residency.release(lease);
                        }
                        return Err(reason);
                    }
                };
                let physical = shared
                    .tables
                    .get(&expert.layer)
                    .and_then(|table| table.host().binding(expert.expert));
                if grant.reason() != expected_reason
                    || shared.resident_keys.get(&expert_key) != Some(&key)
                    || !shared.experts.contains_key(&expert)
                    || !physical.is_some_and(|physical| {
                        physical.slot == i32::try_from(grant.binding().slot.get()).unwrap_or(-1)
                            && physical.generation
                                == i32::try_from(grant.binding().generation.get()).unwrap_or(-1)
                    })
                    || read_plan.validate_source_identity().is_err()
                {
                    if let Some(lease) = lease
                        && let Some(residency) = shared.residency.as_mut()
                    {
                        let _ = residency.release(lease);
                    }
                    return Err(FailureReason::ContractViolation {
                        message: "controller resident binding/source is not physically current"
                            .into(),
                    });
                }
                let validated = match ValidatedResidencyBinding::new(key, raw) {
                    Ok(validated) => validated,
                    Err(error) => {
                        if let Some(lease) = lease
                            && let Some(residency) = shared.residency.as_mut()
                        {
                            let _ = residency.release(lease);
                        }
                        return Err(FailureReason::ContractViolation {
                            message: error.to_string(),
                        });
                    }
                };
                drop(shared);
                if let Some(lease) = lease
                    && let Err(reason) = self.selected_lease_ownership.begin_active(key, lease)
                {
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
                if purpose == MaterializationPurpose::Execution
                    && let Err(reason) = self.selected_lease_ownership.begin_pending(key)
                {
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
                        purpose,
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
        &mut self,
        key: MaterializationKey,
    ) -> std::result::Result<MaterializationPreparation, FailureReason> {
        let (request, expert, read_plan, _) = self.source_for_key(key)?;
        read_plan
            .validate_source_identity()
            .map_err(|_| FailureReason::ContractViolation {
                message: "checkpoint source identity is stale".into(),
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

        let (layer, expert_index) = request.routed_expert_coordinates().ok_or_else(|| {
            FailureReason::ContractViolation {
                message: "routed-expert provider received a non-routed-expert resource".into(),
            }
        })?;
        let expert_key = ExpertKey::new(request.model().get(), layer.get(), expert_index.get());
        let shared = self.shared.lock();
        if shared.resident_keys.get(&expert_key) != Some(&key)
            || !shared.experts.contains_key(&expert)
        {
            return Err(FailureReason::ContractViolation {
                message: "prepared resident expert is not physically published".into(),
            });
        }
        let physical = shared
            .tables
            .get(&expert.layer)
            .and_then(|table| table.host().binding(expert.expert))
            .ok_or_else(|| FailureReason::ContractViolation {
                message: "prepared resident expert has no physical table binding".into(),
            })?;
        let local_slot =
            u32::try_from(physical.slot).map_err(|_| FailureReason::ContractViolation {
                message: "resident expert slot exceeds protocol ABI".into(),
            })?;
        let binding = ResidencyBinding::new(
            request.model(),
            request.resource(),
            request.backend(),
            request.device(),
            device_expert_slot_id(expert.layer, local_slot, self.expert_capacity)?,
            DestinationGeneration::new(u64::try_from(physical.generation).map_err(|_| {
                FailureReason::ContractViolation {
                    message: "resident expert generation exceeds protocol ABI".into(),
                }
            })?),
        );
        MaterializationResident::new(key, binding)
            .map(MaterializationPreparation::Resident)
            .map_err(protocol_failure)
    }

    fn promote_to_execution(
        &mut self,
        key: MaterializationKey,
    ) -> std::result::Result<MaterializationPreparation, FailureReason> {
        let observed = self.prepared(key)?;
        if self.selected_lease_ownership.retain_execution(key)? {
            return Ok(observed);
        }

        if let Some(reservation) = self.reservations.get(&key) {
            let prepared = reservation.prepared;
            self.selected_lease_ownership.begin_pending(key)?;
            let promoted = self
                .shared
                .lock()
                .residency
                .as_mut()
                .ok_or(FailureReason::InstallationRejected)
                .and_then(|residency| {
                    residency
                        .promote_install(prepared)
                        .map_err(protocol_failure)
                });
            let promoted = match promoted {
                Ok(promoted) => promoted,
                Err(reason) => {
                    self.selected_lease_ownership.cancel_unbound(key);
                    return Err(reason);
                }
            };
            let reservation = self
                .reservations
                .get_mut(&key)
                .expect("promoted reservation remains provider-owned");
            reservation.prepared = promoted;
            reservation.purpose = MaterializationPurpose::Execution;
            return Ok(observed);
        }

        if let Some((operation_id, active)) = self
            .operations
            .iter()
            .find(|(_, operation)| operation.key == key)
        {
            let operation_id = *operation_id;
            let prepared = active
                .prepared
                .ok_or_else(|| FailureReason::ContractViolation {
                    message: "active prefetch has no promotable install ownership".into(),
                })?;
            self.selected_lease_ownership.begin_pending(key)?;
            if let Err(reason) = self
                .selected_lease_ownership
                .bind_operation(key, operation_id)
            {
                self.selected_lease_ownership.cancel_unbound(key);
                return Err(reason);
            }
            let promoted = self
                .shared
                .lock()
                .residency
                .as_mut()
                .ok_or(FailureReason::InstallationRejected)
                .and_then(|residency| {
                    residency
                        .promote_install(prepared)
                        .map_err(protocol_failure)
                });
            let promoted = match promoted {
                Ok(promoted) => promoted,
                Err(reason) => {
                    self.selected_lease_ownership
                        .cancel_operation(key, operation_id);
                    return Err(reason);
                }
            };
            let active = self
                .operations
                .get_mut(&operation_id)
                .expect("promoted physical operation remains active");
            active.prepared = Some(promoted);
            active.purpose = MaterializationPurpose::Execution;
            self.promote_operation_order(operation_id);
            return Ok(observed);
        }

        let (request, _, _, _) = self.source_for_key(key)?;
        let (layer, expert_index) = request.routed_expert_coordinates().ok_or_else(|| {
            FailureReason::ContractViolation {
                message: "routed-expert provider received a non-routed-expert resource".into(),
            }
        })?;
        let expert_key = ExpertKey::new(request.model().get(), layer.get(), expert_index.get());
        let lease = {
            let mut shared = self.shared.lock();
            let grant = shared
                .residency
                .as_mut()
                .ok_or(FailureReason::InstallationRejected)?
                .acquire_selected(expert_key)
                .map_err(protocol_failure)?
                .ok_or(FailureReason::InstallationRejected)?;
            let lease = grant
                .lease()
                .ok_or_else(|| FailureReason::ContractViolation {
                    message: "selected resident expert did not return an execution lease".into(),
                })?;
            if grant.reason() != ExpertInstallReason::Selected
                || self.protocol_binding(request, grant.binding())? != observed.binding()
            {
                if let Some(residency) = shared.residency.as_mut() {
                    let _ = residency.release(lease);
                }
                return Err(FailureReason::ContractViolation {
                    message: "selected resident lease does not match the published binding".into(),
                });
            }
            lease
        };
        if let Err(reason) = self.selected_lease_ownership.begin_active(key, lease) {
            if let Some(residency) = self.shared.lock().residency.as_mut() {
                let _ = residency.release(lease);
            }
            return Err(reason);
        }
        Ok(observed)
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
            return Err(FailureReason::ContractViolation {
                message: "cannot discard a preparation claimed by an active operation".into(),
            });
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
        Err(FailureReason::ContractViolation {
            message: "cannot discard an unknown materialization preparation".into(),
        })
    }

    fn materialization_plan(
        &self,
        key: MaterializationKey,
    ) -> std::result::Result<MaterializationResourcePlan, FailureReason> {
        self.source_for_key(key)?;
        self.resource_plans
            .get(&key)
            .copied()
            .ok_or_else(|| FailureReason::ContractViolation {
                message: "routed-expert materialization key has no frozen physical resource plan"
                    .into(),
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
            return Err(FailureReason::ContractViolation {
                message: "duplicate or zero physical materialization operation".into(),
            });
        }
        if self.operations.values().any(|active| active.key == key) {
            return Err(FailureReason::ContractViolation {
                message: "prepared MaterializationKey already has a physical operation owner"
                    .into(),
            });
        }
        if self.materialization_plan(key)? != plan {
            return Err(FailureReason::ContractViolation {
                message: "physical materialization resource plan expectation mismatch".into(),
            });
        }
        let reservation =
            self.reservations
                .remove(&key)
                .ok_or_else(|| FailureReason::ContractViolation {
                    message: "MaterializationKey has no prepared slot reservation".into(),
                })?;
        if reservation.read_plan.validate_source_identity().is_err() {
            return Err(self.rollback_slot_reservation(
                key,
                reservation.request,
                reservation.prepared,
                FailureReason::ContractViolation {
                    message: "checkpoint source identity changed before reserve".into(),
                },
            ));
        }
        if reservation.resource_plan != plan {
            return Err(self.rollback_slot_reservation(
                key,
                reservation.request,
                reservation.prepared,
                FailureReason::ContractViolation {
                    message: "prepared slot resource plan diverged from the canonical plan".into(),
                },
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
                    Err(error) => FailureReason::ContractViolation {
                        message: format!(
                            "physical reservation rollback failed ({reason:?}); read reservation detach also failed ({error})"
                        ),
                    },
                });
            }
        };
        if reservation.purpose == MaterializationPurpose::Execution
            && let Err(reason) = self.selected_lease_ownership.bind_operation(key, operation)
        {
            let cleanup = self.reader.detach_load_source_pinned(read.ticket);
            let reason = self.rollback_slot_reservation(
                key,
                reservation.request,
                reservation.prepared,
                reason,
            );
            return Err(match cleanup {
                Ok(()) => reason,
                Err(error) => FailureReason::ContractViolation {
                    message: format!(
                        "physical lease bind rollback failed ({reason:?}); read reservation detach also failed ({error})"
                    ),
                },
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
                purpose: reservation.purpose,
                binding: reservation.binding,
                evicted: reservation.evicted,
                pending_terminal: None,
                state: Some(MaterializationOperationState::Reserved(read.ticket)),
            },
        );
        self.seen_operations.insert(operation);
        Ok(descriptor)
    }

    fn submit_read(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &PhysicalMaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> std::result::Result<(), FailureReason> {
        let mut active =
            self.operations
                .remove(&operation)
                .ok_or_else(|| FailureReason::ContractViolation {
                    message: "unknown read operation".into(),
                })?;
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
        if let Some(outcome) = Self::operation_identity_outcome(&active, self.expert_capacity) {
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
        let purpose = active.purpose;
        self.operations.insert(operation, active);
        self.enqueue_operation(operation, purpose);
        Ok(())
    }

    fn submit_upload(
        &mut self,
        operation: OperationId,
        key: MaterializationKey,
        reservation: &PhysicalMaterializationOperationReservation,
        plan: MaterializationResourcePlan,
    ) -> std::result::Result<(), FailureReason> {
        let mut active =
            self.operations
                .remove(&operation)
                .ok_or_else(|| FailureReason::ContractViolation {
                    message: "unknown upload operation".into(),
                })?;
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
        if let Some(outcome) = Self::operation_identity_outcome(&active, self.expert_capacity) {
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
                let purpose = active.purpose;
                self.operations.insert(operation, active);
                self.enqueue_operation(operation, purpose);
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
        let mut active =
            self.operations
                .remove(&operation)
                .ok_or_else(|| FailureReason::ContractViolation {
                    message: "unknown install operation".into(),
                })?;
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
        if let Some(outcome) = Self::operation_identity_outcome(&active, self.expert_capacity) {
            let _ = self.finish_terminal_operation(
                operation,
                active,
                Some(MaterializationOperationState::UploadReady(frame)),
                Some((LoadStage::Installing, outcome, 0)),
                true,
            );
            return Ok(());
        }
        active.state = Some(MaterializationOperationState::InstallQueued(frame));
        let purpose = active.purpose;
        self.operations.insert(operation, active);
        self.enqueue_operation(operation, purpose);
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
            return Err(FailureReason::ContractViolation {
                message: "cancel found an ownerless physical operation".into(),
            });
        };
        if active.key != key {
            active.state = Some(state);
            self.operations.insert(operation, active);
            return Err(FailureReason::ContractViolation {
                message: "cancel key does not match physical operation identity".into(),
            });
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
            return Err(FailureReason::ContractViolation {
                message: format!(
                    "cancel observed stage {stage:?} is incompatible with physical operation stage {actual_stage:?}"
                ),
            });
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
            MaterializationOperationState::InstallQueued(frame) => self
                .finish_terminal_operation(
                    operation,
                    active,
                    Some(MaterializationOperationState::InstallQueued(frame)),
                    Some((
                        LoadStage::Installing,
                        CompletionOutcome::Cancelled(reason),
                        0,
                    )),
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
            MaterializationOperationState::Installing(ticket) => {
                self.release_execution_lease(key)?;
                active.pending_terminal = Some(CompletionOutcome::Cancelled(reason));
                active.state = Some(MaterializationOperationState::Installing(ticket));
                self.operations.insert(operation, active);
                Ok(())
            }
        }
    }

    fn next_completion(&mut self) -> Option<CompletionEvent> {
        self.progress_one()
    }
}

impl Drop for CudaExpertMaterializationProvider {
    fn drop(&mut self) {
        let operation_count = self.operations.len();
        let reservation_count = self.reservations.len();
        if operation_count != 0 || reservation_count != 0 {
            tracing::error!(
                operations = operation_count,
                reservations = reservation_count,
                "routed-expert CUDA materialization provider dropped before runtime registry drain"
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
                    "failed to drain a routed-expert physical operation during drop"
                );
            }
        }
        self.execution_order.clear();
        self.prefetch_order.clear();

        let reservations = std::mem::take(&mut self.reservations);
        for (key, reservation) in reservations {
            self.remove_request_key(reservation.request, key);
            self.selected_lease_ownership.cancel_unbound(key);
            if let Err(error) = self.cancel_prepared(reservation.prepared) {
                tracing::error!(
                    error = %error,
                    "failed to cancel a routed-expert slot reservation during drop"
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
                        "routed-expert pending selected ownership survived backend drain"
                    );
                    continue;
                };
                let release = shared
                    .residency
                    .as_mut()
                    .ok_or_else(|| Error::Execution {
                        message: "physical residency control is not installed".into(),
                    })
                    .and_then(|residency| residency.release(lease));
                if let Err(error) = release {
                    tracing::error!(
                        error = %error,
                        "failed to release a routed-expert selected lease during drop"
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
                return Err(Error::Model {
                    message: "physical pinned expert payload identity mismatch".into(),
                });
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
            let bytes = u64::try_from(bytes).map_err(|_| Error::Model {
                message: "physical pinned expert payload component exceeds u64".into(),
            })?;
            total.checked_add(bytes).ok_or_else(|| Error::Model {
                message: "physical pinned expert payload byte total overflow".into(),
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
                return Err(Error::Model {
                    message: "physical pinned expert tensor identity mismatch".into(),
                });
            }
            match tensor.slice.component {
                crate::moe::streaming::ExpertTensorComponent::Weight => {
                    if weight.replace(tensor).is_some() {
                        return Err(Error::Model {
                            message: "duplicate physical expert weight".into(),
                        });
                    }
                }
                crate::moe::streaming::ExpertTensorComponent::Scale => {
                    if scale.replace(tensor).is_some() {
                        return Err(Error::Model {
                            message: "duplicate physical expert scale".into(),
                        });
                    }
                }
                crate::moe::streaming::ExpertTensorComponent::Other(component) => {
                    return Err(Error::Model {
                        message: format!(
                            "unsupported physical expert tensor component {component}"
                        ),
                    });
                }
            }
        }
        let weight = weight.ok_or_else(|| Error::Model {
            message: "missing physical expert weight".into(),
        })?;
        let scale = scale.ok_or_else(|| Error::Model {
            message: "missing physical expert scale".into(),
        })?;
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

fn pinned_linear_dimensions(linear: &PinnedExpertLinear) -> Result<(usize, usize)> {
    let ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
        out_features,
        in_features,
        block_size: 32,
    } = linear.format
    else {
        return Err(Error::Model {
            message: format!(
                "CUDA physical expert {:?} requires FP4 E2M1/E8M0 block_size=32",
                linear.matrix
            ),
        });
    };
    validate_mxfp4_linear_storage(
        out_features,
        in_features,
        linear.weight.len(),
        linear.scale.len(),
    )?;
    Ok((out_features, in_features))
}

fn pinned_routed_expert_shape(bundle: &PinnedExpertBundle) -> Result<CudaRoutedExpertShape> {
    let (gate_out, gate_in) = pinned_linear_dimensions(&bundle.gate)?;
    let (up_out, up_in) = pinned_linear_dimensions(&bundle.up)?;
    let (down_out, down_in) = pinned_linear_dimensions(&bundle.down)?;
    if (up_out, up_in) != (gate_out, gate_in) || down_in != gate_out {
        return Err(Error::Model {
            message: format!(
                "inconsistent CUDA routed-expert projection dimensions: gate={gate_out}x{gate_in} up={up_out}x{up_in} down={down_out}x{down_in}"
            ),
        });
    }
    CudaRoutedExpertShape::new(gate_in, gate_out, down_out)
}

fn validate_mxfp4_linear_storage(
    out_features: usize,
    in_features: usize,
    weight_bytes: usize,
    scale_bytes: usize,
) -> Result<()> {
    if out_features == 0
        || in_features == 0
        || !in_features.is_multiple_of(32)
        || !in_features.is_multiple_of(2)
    {
        return Err(Error::Model {
            message: format!(
                "invalid CUDA physical expert FP4 shape: out={out_features} in={in_features}"
            ),
        });
    }
    let expected_weight =
        out_features
            .checked_mul(in_features / 2)
            .ok_or_else(|| Error::Model {
                message: "physical expert FP4 weight storage overflow".into(),
            })?;
    let expected_scale =
        out_features
            .checked_mul(in_features / 32)
            .ok_or_else(|| Error::Model {
                message: "physical expert linear scale storage overflow".into(),
            })?;
    if weight_bytes != expected_weight || scale_bytes != expected_scale {
        return Err(Error::Model {
            message: format!(
                "physical expert FP4 storage mismatch: weight={weight_bytes}/{expected_weight} scale={scale_bytes}/{expected_scale}"
            ),
        });
    }
    Ok(())
}

fn source_routed_expert_shape(source: &ExpertLoadSource) -> Result<CudaRoutedExpertShape> {
    let tensors = match source {
        ExpertLoadSource::LocalTensorSet { tensors }
        | ExpertLoadSource::HfLocalTensorSet { tensors, .. } => tensors,
        _ => {
            return Err(Error::Model {
                message: "CUDA physical expert source does not expose tensor shapes".into(),
            });
        }
    };
    let mut dimensions = BTreeMap::new();
    for matrix in [
        ExpertMatrixKind::Gate,
        ExpertMatrixKind::Up,
        ExpertMatrixKind::Down,
    ] {
        let mut weight = None;
        let mut scale = None;
        for tensor in tensors.iter().filter(|tensor| tensor.key.matrix == matrix) {
            match tensor.component {
                crate::moe::streaming::ExpertTensorComponent::Weight => {
                    if weight.replace(tensor).is_some() {
                        return Err(Error::Model {
                            message: "duplicate physical expert source weight".into(),
                        });
                    }
                }
                crate::moe::streaming::ExpertTensorComponent::Scale => {
                    if scale.replace(tensor).is_some() {
                        return Err(Error::Model {
                            message: "duplicate physical expert source scale".into(),
                        });
                    }
                }
                crate::moe::streaming::ExpertTensorComponent::Other(_) => {}
            }
        }
        let weight = weight.ok_or_else(|| Error::Model {
            message: "missing physical expert source weight".into(),
        })?;
        let scale = scale.ok_or_else(|| Error::Model {
            message: "missing physical expert source scale".into(),
        })?;
        let weight_bytes = usize::try_from(weight.bytes).map_err(|_| Error::Model {
            message: "physical expert source weight exceeds usize".into(),
        })?;
        let scale_bytes = usize::try_from(scale.bytes).map_err(|_| Error::Model {
            message: "physical expert source scale exceeds usize".into(),
        })?;
        let format = infer_expert_linear_format(weight, weight_bytes, Some((scale, scale_bytes)))?;
        let ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
            out_features,
            in_features,
            block_size: 32,
        } = format
        else {
            return Err(Error::Model {
                message: "CUDA physical expert source requires FP4 E2M1/E8M0 block_size=32".into(),
            });
        };
        validate_mxfp4_linear_storage(out_features, in_features, weight_bytes, scale_bytes)?;
        dimensions.insert(matrix, (out_features, in_features));
    }
    let (gate_out, gate_in) = dimensions[&ExpertMatrixKind::Gate];
    let (up_out, up_in) = dimensions[&ExpertMatrixKind::Up];
    let (down_out, down_in) = dimensions[&ExpertMatrixKind::Down];
    if (up_out, up_in) != (gate_out, gate_in) || down_in != gate_out {
        return Err(Error::Model {
            message: format!(
                "inconsistent CUDA routed-expert source dimensions: gate={gate_out}x{gate_in} up={up_out}x{up_in} down={down_out}x{down_in}"
            ),
        });
    }
    CudaRoutedExpertShape::new(gate_in, gate_out, down_out)
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
        return Err(FailureReason::ContractViolation {
            message: "physical operation reservation identity mismatch".into(),
        });
    }
    Ok(())
}

fn record_first_error(first: &mut Option<Error>, result: Result<()>) {
    if let Err(error) = result {
        first.get_or_insert(error);
    }
}

fn protocol_failure(error: impl std::fmt::Display) -> FailureReason {
    FailureReason::ContractViolation {
        message: error.to_string(),
    }
}

/// Convert a layer-local physical slot into the device-global residency namespace.
/// Physical CUDA tables remain layer-local; only runtime publication uses this ID.
fn device_expert_slot_id(
    layer: usize,
    local_slot: u32,
    expert_capacity: usize,
) -> std::result::Result<DestinationSlotId, FailureReason> {
    if expert_capacity == 0 {
        return Err(FailureReason::ContractViolation {
            message: "expert capacity must be non-zero".into(),
        });
    }
    let local_slot = usize::try_from(local_slot).map_err(|_| FailureReason::ContractViolation {
        message: "local expert slot exceeds usize".into(),
    })?;
    if local_slot >= expert_capacity {
        return Err(FailureReason::ContractViolation {
            message: format!(
                "local expert slot {local_slot} exceeds layer capacity {expert_capacity}"
            ),
        });
    }
    let device_slot = layer
        .checked_mul(expert_capacity)
        .and_then(|base| base.checked_add(local_slot))
        .ok_or_else(|| FailureReason::ContractViolation {
            message: "device expert slot identity overflow".into(),
        })?;
    let device_slot = u32::try_from(device_slot).map_err(|_| FailureReason::ContractViolation {
        message: "device expert slot exceeds protocol ABI".into(),
    })?;
    Ok(DestinationSlotId::new(device_slot))
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
    fn device_expert_slots_are_unique_across_layers() {
        let layer_zero = device_expert_slot_id(0, 5, 256).unwrap();
        let layer_one = device_expert_slot_id(1, 5, 256).unwrap();
        let final_slot = device_expert_slot_id(45, 255, 256).unwrap();

        assert_eq!(layer_zero.get(), 5);
        assert_eq!(layer_one.get(), 261);
        assert_eq!(final_slot.get(), 11_775);
        assert_ne!(layer_zero, layer_one);
    }

    #[test]
    fn invalid_device_expert_slots_are_rejected() {
        assert!(device_expert_slot_id(0, 0, 0).is_err());
        assert!(device_expert_slot_id(0, 256, 256).is_err());
        assert!(device_expert_slot_id(usize::MAX, 0, 256).is_err());
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

    type MockState = MaterializationOperationState<u64, u64, u64, u64, u64>;

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
        install_events: usize,
        frames: usize,
        prepared_slots: usize,
        resident_frames: usize,
    }

    impl MockResources {
        fn transient_is_empty(&self) -> bool {
            self.read_slabs == 0
                && self.host_bundles == 0
                && self.upload_events == 0
                && self.install_events == 0
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

        fn queue_install(&mut self, operation: u64) -> bool {
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
                active.state = Some(MockState::InstallQueued(frame));
                self.operations.insert(operation, active);
            }
            true
        }

        fn submit_physical_install(&mut self, operation: u64) -> bool {
            let Some(mut active) = self.operations.remove(&operation) else {
                return false;
            };
            let frame = match active.state.take() {
                Some(MockState::InstallQueued(frame)) => frame,
                state => {
                    active.state = state;
                    self.operations.insert(operation, active);
                    return false;
                }
            };
            self.resources.install_events += 1;
            active.state = Some(MockState::Installing(frame));
            self.operations.insert(operation, active);
            true
        }

        fn submit_install(&mut self, operation: u64) -> bool {
            if !self.queue_install(operation) {
                return false;
            }
            if !self.operations.contains_key(&operation) {
                return true;
            }
            self.submit_physical_install(operation)
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
            active.pending_terminal = None;
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
            self.resources.install_events -= 1;
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
                MockState::Reserved(_)
                | MockState::HostReady(_)
                | MockState::UploadReady(_)
                | MockState::InstallQueued(_) => {
                    self.finish(operation, active, Some(state), None);
                }
                MockState::ReadSubmitted(_) | MockState::UploadSubmitted(_) => {
                    active.pending_terminal = Some(CompletionOutcome::Cancelled(reason));
                    active.state = Some(state);
                    self.operations.insert(operation, active);
                }
                MockState::Installing(_) => {
                    active.pending_terminal = Some(CompletionOutcome::Cancelled(reason));
                    active.state = Some(state);
                    self.operations.insert(operation, active);
                }
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
                MockState::UploadReady(_) | MockState::InstallQueued(_) => {
                    self.resources.frames -= 1;
                }
                MockState::Installing(_) => {
                    self.resources.install_events -= 1;
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
            let submitted = self
                .operations
                .iter()
                .filter_map(|(operation, active)| {
                    matches!(active.state, Some(MockState::Installing(_))).then_some(*operation)
                })
                .collect::<Vec<_>>();
            for operation in submitted {
                let _ = self.complete_install(operation, None);
            }
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
        assert_eq!(mock.resources.install_events, 1);
        assert_eq!(mock.resources.frames, 1);
        assert_eq!(mock.resources.prepared_slots, 1);
        assert!(mock.operations.contains_key(&6));
        assert!(mock.complete_install(6, None));
        assert!(mock.resources.transient_is_empty());
        assert_eq!(mock.resources.resident_frames, 1);
        assert!(!mock.complete_install(6, None));
        assert_eq!(mock.terminal_completion_count(6), 0);
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
    fn host_mock_install_cancel_retains_frame_until_event_then_publishes_once() {
        let mut mock = HostMockProvider::default();
        assert!(mock.reserve(36, 306));
        advance_to_frame(&mut mock, 36);
        assert!(mock.submit_install(36));
        assert!(mock.physical_cancel(36, CancellationReason::OwnerShutdown));
        assert_eq!(mock.resources.install_events, 1);
        assert_eq!(mock.resources.frames, 1);
        assert_eq!(mock.resources.prepared_slots, 1);
        assert_eq!(mock.resources.resident_frames, 0);
        assert!(mock.operations.contains_key(&36));

        assert!(mock.complete_install(36, None));
        assert!(mock.resources.transient_is_empty());
        assert_eq!(mock.resources.resident_frames, 1);
        assert!(!mock.complete_install(36, None));
        assert_eq!(mock.terminal_completion_count(36), 0);
        assert_eq!(
            mock.completions
                .iter()
                .filter(|(operation, stage, outcome)| {
                    *operation == 36
                        && *stage == LoadStage::Installing
                        && matches!(outcome, CompletionOutcome::Succeeded)
                })
                .count(),
            1
        );
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
            MockState::InstallQueued(6),
            MockState::Installing(7),
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
            MockState::InstallQueued(1).owner_stage(),
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
