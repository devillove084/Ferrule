//! DeepSeek-V4 runner: ModelRunner implementation.

#[cfg(feature = "cuda")]
use std::collections::BTreeMap;
#[cfg(any(feature = "cuda", test))]
use std::collections::{BTreeSet, HashMap};

#[cfg(any(feature = "cuda", test))]
use std::num::NonZeroU64;

#[cfg(any(feature = "cuda", test))]
use std::ops::Range;
use std::path::Path;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use crate::execution::ExecutionShapeKey;
use crate::execution::ModelExecutionBackend;
#[cfg(any(feature = "cuda", test))]
use crate::execution::SequenceStepBinding;
#[cfg(feature = "cuda")]
use crate::execution::{
    OwnedArenaCheckout, PersistentArenaPool, ResourceAccess, ResourceRetention,
    StageMaterializationRequest, StageResourceUse, WorkspaceClaim,
};
#[cfg(any(feature = "cuda", test))]
use crate::materialization::ContinuationDependencyState;
#[cfg(feature = "cuda")]
use crate::materialization::{MaterializationPlacement, resolve_stage_resources};
use crate::materialization::{
    MaterializationProvider, MaterializationRequest, MaterializationResolver,
};
#[cfg(feature = "cuda")]
use crate::moe::cuda_materialization::CudaExpertMaterializationOwner;
use crate::moe::prediction::ExpertHotsetPredictor;

#[cfg(all(feature = "cuda", target_os = "linux"))]
use crate::moe::streaming::ExpertIoPlan;
use crate::moe::streaming::ExpertStreamingReader;
#[cfg(feature = "cuda")]
use crate::runner::NativeProposal;
use crate::runner::{
    BatchContinuationId, ModelCompletionReactor, ModelInfo, ModelRunner, MultiSessionBatchProgress,
    MultiSessionRunner, NativeProposalProgress, NativeProposalSource, ResidentModelRunner,
    TransactionEndIntent, TransactionEndProgress,
};
#[cfg(feature = "cuda")]
use crate::runner::{PendingModelProgress, TokenLogit};
use ferrule_common::execution::{ExecutionBatch, ExecutionTransactionId};
#[cfg(feature = "cuda")]
use ferrule_common::execution::{
    ExecutionIntent, ExecutionOutput, LogitsOutput, LogitsRequest, LogitsRow,
    TokenLogit as ExecutionTokenLogit,
};
#[cfg(any(feature = "cuda", test))]
use ferrule_common::execution::{ForwardMode, ForwardPhase, KvPageId};

#[cfg(any(feature = "cuda", test))]
use ferrule_common::DependencySet;
#[cfg(any(feature = "cuda", test))]
use ferrule_common::LogicalDependency;
#[cfg(feature = "cuda")]
use ferrule_common::ModelInstanceId;
use ferrule_common::expert_residency::{ExpertResidencyControl, ExpertResidencyRequirements};
#[cfg(any(feature = "cuda", test))]
use ferrule_common::{
    BackendId, DeviceId, DispatchFenceContract, FenceId, MappingEpoch, OperationId,
};
use ferrule_common::{CompletionHub, Error, ResidencyLeaseSet, Result};

use super::checkpoint::DeepSeekV4Checkpoint;

#[cfg(feature = "cuda")]
use super::cuda_cache::{
    DeepSeekV4CudaComputeQuiescence, DeepSeekV4DecodeBuffers, DeepSeekV4OutputHeadTopKDownload,
};
#[cfg(feature = "cuda")]
use super::cuda_cache::{
    DeepSeekV4ProposalAttentionBuffers, DeepSeekV4ProposalHeadBuffers,
    DeepSeekV4ProposalMainBuffers,
};

#[cfg(feature = "cuda")]
use super::layer::{
    DeepSeekV4LayerArena, DeepSeekV4ProposalLayerContinuation, DeepSeekV4ProposalLayerProgress,
};
#[cfg(feature = "cuda")]
use super::layer::{
    DeepSeekV4LayerArenaVariants, DeepSeekV4PackedLayerContinuation, DeepSeekV4PackedLayerProgress,
};
use super::layer::{DeepSeekV4LayerExpertRuntime, DeepSeekV4LayerState};
use super::operators::{
    DeepSeekV4AttentionProfileStats, DeepSeekV4LayerProfileStats, DeepSeekV4OperatorContext,
    DeepSeekV4OperatorRuntimeCounters,
};
pub use super::prepared::DeepSeekV4PrepareOptions;
#[cfg(feature = "cuda")]
use super::prepared::DeepSeekV4PreparedResources;
use super::prepared::{
    DeepSeekV4ExecutionPolicy, DeepSeekV4PrepareProfile, DeepSeekV4PreparedLayerExperts,
    DeepSeekV4PreparedModelPlan, prepare_with_policy,
};
use super::proposal_attachment::DeepSeekV4ProposalAttachment;
#[cfg(feature = "cuda")]
use super::proposal_attachment::DeepSeekV4ProposalProtocol;
use super::sequence::DeepSeekV4SequenceExecutionState;
#[cfg(any(feature = "cuda", test))]
use super::sequence::DeepSeekV4SequenceTopologyId;
#[cfg(feature = "cuda")]
use super::sequence::{DeepSeekV4PagedKvBinding, DeepSeekV4SequenceMoeAccessEvent};
#[cfg(feature = "cuda")]
use crate::moe::residency::{
    DeviceResidencyPlan, DeviceResidencyPlanError, ExpertLayerInventory, plan_device_residency,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4LayerRuntimeStats {
    pub layer: usize,
    pub window_kv_len: usize,
    pub compressed_kv_len: usize,
    pub indexer_compressed_kv_len: usize,
    pub resident_experts: usize,
    pub resident_expert_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4OutputProfileStats {
    pub packed_prefill_batches: u64,
    pub packed_prefill_rows: u64,
    pub packed_decode_batches: u64,
    pub packed_decode_rows: u64,
    pub packed_mixed_batches: u64,
    pub packed_mixed_rows: u64,
    pub final_hc_head_calls: u64,
    pub final_hc_head_us: u64,
    pub final_norm_calls: u64,
    pub final_norm_us: u64,
    pub lm_head_topk_calls: u64,
    pub lm_head_topk_us: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4LoadProfile {
    pub checkpoint_us: u64,
    pub prepare: DeepSeekV4PrepareProfile,
    pub cuda_context_us: u64,
    pub sequence_state_us: u64,
    pub reader_us: u64,
    pub static_image_globals_us: u64,
    pub static_image_embedding_us: u64,
    pub static_image_output_head_us: u64,
    pub static_image_target_layers_us: u64,
    pub static_image_attachment_us: u64,
    pub static_image_total_us: u64,
    pub residency_us: u64,
    pub total_us: u64,
}

#[derive(Debug, Clone)]
pub struct DeepSeekV4ObservabilitySnapshot {
    pub position: usize,
    pub load: DeepSeekV4LoadProfile,
    pub operator: DeepSeekV4OperatorRuntimeCounters,
    pub layers: Vec<DeepSeekV4LayerProfileStats>,
    pub attention: Vec<DeepSeekV4AttentionProfileStats>,
    pub output: DeepSeekV4OutputProfileStats,
    pub layer_runtime: Vec<DeepSeekV4LayerRuntimeStats>,
}

#[cfg(feature = "cuda")]
static NEXT_DSV4_MODEL_INSTANCE: AtomicU64 = AtomicU64::new(1);

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum DeepSeekV4LayerArenaRowLayout {
    IndependentRows,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct DeepSeekV4LayerArenaPoolKey {
    shape: ExecutionShapeKey,
    row_layout: DeepSeekV4LayerArenaRowLayout,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4LayerArenaPoolKey {
    pub(super) const fn new(
        shape: ExecutionShapeKey,
        row_layout: DeepSeekV4LayerArenaRowLayout,
    ) -> Self {
        Self { shape, row_layout }
    }
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DeepSeekV4PackedSequenceOwner {
    state_index: usize,
    topology_id: DeepSeekV4SequenceTopologyId,
    binding: SequenceStepBinding,
}

#[cfg(any(feature = "cuda", test))]
impl DeepSeekV4PackedSequenceOwner {
    pub(super) fn capture(
        state_index: usize,
        state: &DeepSeekV4SequenceExecutionState,
    ) -> Result<Self> {
        Ok(Self {
            state_index,
            topology_id: state.topology_id(),
            binding: state.begin_step()?,
        })
    }
}

#[cfg(any(feature = "cuda", test))]
struct DeepSeekV4PackedTransactionSlot<T> {
    owners: Vec<DeepSeekV4PackedSequenceOwner>,
    protected_pages: BTreeSet<KvPageId>,
    payload: Option<T>,
}

/// Host-side ownership index for packed transaction topology.
///
/// Payloads may be taken temporarily while one transaction is progressed, but
/// sequence/page ownership remains registered until an exactly-once terminal
/// `finish`. No mutex or runner-wide execution lease is used.
#[cfg(any(feature = "cuda", test))]
pub(super) struct DeepSeekV4PackedTransactionRegistry<T> {
    transactions: HashMap<ExecutionTransactionId, DeepSeekV4PackedTransactionSlot<T>>,
    sequence_owners: HashMap<DeepSeekV4SequenceTopologyId, ExecutionTransactionId>,
    page_readers: HashMap<KvPageId, BTreeSet<ExecutionTransactionId>>,
}

#[cfg(any(feature = "cuda", test))]
impl<T> Default for DeepSeekV4PackedTransactionRegistry<T> {
    fn default() -> Self {
        Self {
            transactions: HashMap::new(),
            sequence_owners: HashMap::new(),
            page_readers: HashMap::new(),
        }
    }
}

#[cfg(any(feature = "cuda", test))]
impl<T> DeepSeekV4PackedTransactionRegistry<T> {
    fn validate_insert(
        &self,
        transaction: ExecutionTransactionId,
        owners: &[DeepSeekV4PackedSequenceOwner],
    ) -> Result<()> {
        if self.transactions.contains_key(&transaction) {
            return Err(Error::Execution {
                message: format!(
                    "DeepSeek-V4 transaction {} is already registered",
                    transaction.get()
                ),
            });
        }
        let mut unique = BTreeSet::new();
        for owner in owners {
            if !unique.insert(owner.topology_id.get()) {
                return Err(Error::Execution {
                    message: format!(
                        "DeepSeek-V4 transaction {} references sequence topology {} more than once",
                        transaction.get(),
                        owner.topology_id.get()
                    ),
                });
            }
            if let Some(active) = self.sequence_owners.get(&owner.topology_id) {
                return Err(Error::Execution {
                    message: format!(
                        "DeepSeek-V4 sequence topology {} is already owned by transaction {}; transaction {} conflicts",
                        owner.topology_id.get(),
                        active.get(),
                        transaction.get()
                    ),
                });
            }
        }
        Ok(())
    }

    fn insert(
        &mut self,
        transaction: ExecutionTransactionId,
        owners: Vec<DeepSeekV4PackedSequenceOwner>,
        protected_pages: impl IntoIterator<Item = KvPageId>,
        payload: T,
    ) -> Result<()> {
        self.validate_insert(transaction, &owners)?;
        let protected_pages = protected_pages.into_iter().collect::<BTreeSet<_>>();
        for owner in &owners {
            self.sequence_owners.insert(owner.topology_id, transaction);
        }
        for page in &protected_pages {
            self.page_readers
                .entry(*page)
                .or_default()
                .insert(transaction);
        }
        self.transactions.insert(
            transaction,
            DeepSeekV4PackedTransactionSlot {
                owners,
                protected_pages,
                payload: Some(payload),
            },
        );
        Ok(())
    }

    fn take_payload(&mut self, transaction: ExecutionTransactionId) -> Result<T> {
        self.transactions
            .get_mut(&transaction)
            .ok_or_else(|| Error::Execution {
                message: format!(
                    "DeepSeek-V4 transaction {} is not registered",
                    transaction.get()
                ),
            })?
            .payload
            .take()
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "DeepSeek-V4 transaction {} topology is already being progressed",
                    transaction.get()
                ),
            })
    }

    fn put_payload(&mut self, transaction: ExecutionTransactionId, payload: T) -> Result<()> {
        let slot = self
            .transactions
            .get_mut(&transaction)
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "DeepSeek-V4 transaction {} lost its ownership slot",
                    transaction.get()
                ),
            })?;
        if slot.payload.is_some() {
            return Err(Error::Internal {
                message: format!(
                    "DeepSeek-V4 transaction {} topology was restored more than once",
                    transaction.get()
                ),
            });
        }
        slot.payload = Some(payload);
        Ok(())
    }

    fn validate_states(
        &self,
        transaction: ExecutionTransactionId,
        states: &[DeepSeekV4SequenceExecutionState],
    ) -> Result<()> {
        let slot = self
            .transactions
            .get(&transaction)
            .ok_or_else(|| Error::Execution {
                message: format!(
                    "DeepSeek-V4 transaction {} is not registered",
                    transaction.get()
                ),
            })?;
        for owner in &slot.owners {
            let state = states
                .get(owner.state_index)
                .ok_or_else(|| Error::Execution {
                    message: format!(
                        "DeepSeek-V4 transaction {} state slot {} no longer exists",
                        transaction.get(),
                        owner.state_index
                    ),
                })?;
            if state.topology_id() != owner.topology_id {
                return Err(Error::Execution {
                    message: format!(
                        "DeepSeek-V4 transaction {} state slot {} changed topology identity from {} to {}",
                        transaction.get(),
                        owner.state_index,
                        owner.topology_id.get(),
                        state.topology_id().get()
                    ),
                });
            }
            let current = state.begin_step()?;
            if current != owner.binding {
                return Err(Error::Execution {
                    message: format!(
                        "DeepSeek-V4 transaction {} has a stale sequence generation/position at state slot {}: expected {}/{}, got {}/{}",
                        transaction.get(),
                        owner.state_index,
                        owner.binding.generation(),
                        owner.binding.committed_position(),
                        current.generation(),
                        current.committed_position()
                    ),
                });
            }
        }
        Ok(())
    }

    fn ensure_sequence_available(
        &self,
        state: &DeepSeekV4SequenceExecutionState,
        operation: &str,
    ) -> Result<()> {
        if let Some(transaction) = self.sequence_owners.get(&state.topology_id()) {
            return Err(Error::Execution {
                message: format!(
                    "cannot {operation}: DeepSeek-V4 sequence topology {} is owned by transaction {}",
                    state.topology_id().get(),
                    transaction.get()
                ),
            });
        }
        Ok(())
    }

    fn ensure_pages_available(&self, pages: &[KvPageId], operation: &str) -> Result<()> {
        if let Some((page, readers)) = pages.iter().find_map(|page| {
            self.page_readers
                .get(page)
                .filter(|readers| !readers.is_empty())
                .map(|readers| (*page, readers))
        }) {
            return Err(Error::Execution {
                message: format!(
                    "cannot {operation} DeepSeek-V4 KV page {} while transaction(s) {:?} reference it",
                    page.0, readers
                ),
            });
        }
        Ok(())
    }

    fn finalize_checkout(&mut self, transaction: ExecutionTransactionId) {
        let slot = self
            .transactions
            .remove(&transaction)
            .expect("checked-out packed transaction lost its registry slot");
        debug_assert!(slot.payload.is_none());
        self.release_ownership(transaction, &slot);
    }

    fn release_ownership(
        &mut self,
        transaction: ExecutionTransactionId,
        slot: &DeepSeekV4PackedTransactionSlot<T>,
    ) {
        for owner in &slot.owners {
            if self.sequence_owners.get(&owner.topology_id) == Some(&transaction) {
                self.sequence_owners.remove(&owner.topology_id);
            }
        }
        for page in &slot.protected_pages {
            if let Some(readers) = self.page_readers.get_mut(page) {
                readers.remove(&transaction);
                if readers.is_empty() {
                    self.page_readers.remove(page);
                }
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.transactions.is_empty()
    }

    fn contains(&self, transaction: ExecutionTransactionId) -> bool {
        self.transactions.contains_key(&transaction)
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.transactions.len()
    }

    #[cfg(test)]
    pub(super) fn owner_of(
        &self,
        state: &DeepSeekV4SequenceExecutionState,
    ) -> Option<ExecutionTransactionId> {
        self.sequence_owners.get(&state.topology_id()).copied()
    }
}

#[cfg(feature = "cuda")]
enum DeepSeekV4PackedOutputHeadState {
    NotStarted,
    Downloading(DeepSeekV4OutputHeadTopKDownload),
    Ready(Vec<Vec<TokenLogit>>),
}

#[cfg(feature = "cuda")]
struct DeepSeekV4PackedCudaContinuation {
    transaction: ExecutionTransactionId,
    id: BatchContinuationId,
    batch: ExecutionBatch,
    metadata: PackedBatchMetadata,
    sequence_step_bindings: Vec<SequenceStepBinding>,
    paged_bindings: Vec<DeepSeekV4PagedKvBinding>,
    sequence_phases: Vec<ForwardPhase>,
    positions: Vec<usize>,
    max_top_k: usize,
    arena_checkout:
        Option<OwnedArenaCheckout<DeepSeekV4LayerArenaPoolKey, DeepSeekV4LayerArenaVariants>>,
    decode_buffers: Option<DeepSeekV4DecodeBuffers>,
    #[cfg(feature = "cuda")]
    proposal_main_buffers: Option<DeepSeekV4ProposalMainBuffers>,
    initialized: bool,
    next_layer_index: usize,
    current_layer: Option<DeepSeekV4PackedLayerContinuation>,
    output_head: DeepSeekV4PackedOutputHeadState,
    cancel_quiescence: Option<DeepSeekV4CudaComputeQuiescence>,
    dependencies: Option<ContinuationDependencyState>,
    moe_access_events: Vec<DeepSeekV4SequenceMoeAccessEvent>,
    expert_leases: Vec<ResidencyLeaseSet>,
    paged_bindings_active: bool,
    provisional_checkpoints: bool,
    failed: bool,
}

#[cfg(feature = "cuda")]
enum DeepSeekV4PackedTransactionExecution {
    Prepared,
    Waiting(DeepSeekV4PackedCudaContinuation),
    Complete(Vec<ResidencyLeaseSet>),
}

#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
struct DeepSeekV4PackedTransactionTopology {
    batch: ExecutionBatch,
    metadata: PackedBatchMetadata,
    committed_state_indices: Vec<usize>,
    working_states: Vec<DeepSeekV4SequenceExecutionState>,
    execution: DeepSeekV4PackedTransactionExecution,
}

#[cfg(feature = "cuda")]
enum DeepSeekV4PackedCudaProgress {
    Waiting(DeepSeekV4PackedCudaContinuation, PendingModelProgress),
    Complete(ExecutionOutput, Vec<ResidencyLeaseSet>),
}

#[cfg(feature = "cuda")]
enum DeepSeekV4PackedCudaStep {
    Waiting(PendingModelProgress),
    Complete(ExecutionOutput),
}

struct DeepSeekV4RunnerObservability {
    load: DeepSeekV4LoadProfile,
    output: DeepSeekV4OutputProfileStats,
}

impl DeepSeekV4RunnerObservability {
    fn new() -> Self {
        Self {
            load: DeepSeekV4LoadProfile::default(),
            output: DeepSeekV4OutputProfileStats::default(),
        }
    }
}

pub struct DeepSeekV4Runner {
    plan: DeepSeekV4PreparedModelPlan,
    operators: DeepSeekV4OperatorContext,
    /// CPU/reference planner and handle stores. CUDA consumes immutable prepared
    /// catalogs while runtime owns logical slots, generations, leases, and policy.
    cpu_expert_runtimes: Option<Box<[DeepSeekV4LayerExpertRuntime]>>,
    materialization_resolver: Option<Box<dyn MaterializationResolver>>,

    #[cfg(feature = "cuda")]
    cuda_materialization_owner: Option<CudaExpertMaterializationOwner>,
    #[cfg(feature = "cuda")]
    materialization_provider_transferred: bool,
    #[cfg(feature = "cuda")]
    expert_residency_requirements: Option<ExpertResidencyRequirements>,
    #[cfg(feature = "cuda")]
    warmup_requests: Vec<MaterializationRequest>,
    #[cfg(feature = "cuda")]
    device_residency_plan: Option<DeviceResidencyPlan>,
    #[cfg(feature = "cuda")]
    layer_arena_pool:
        PersistentArenaPool<DeepSeekV4LayerArenaPoolKey, DeepSeekV4LayerArenaVariants>,
    #[cfg(feature = "cuda")]
    packed_transactions: DeepSeekV4PackedTransactionRegistry<DeepSeekV4PackedTransactionTopology>,
    #[cfg(feature = "cuda")]
    next_packed_cuda_continuation_id: NonZeroU64,
    #[cfg(feature = "cuda")]
    proposal_arena_pool: Vec<DeepSeekV4ProposalArena>,
    #[cfg(feature = "cuda")]
    proposal_continuations: HashMap<BatchContinuationId, DeepSeekV4ProposalContinuation>,
    proposal_source: Option<NativeProposalSource>,
    /// E3: per-sequence state. The runner wraps one default sequence.
    sequence: DeepSeekV4SequenceExecutionState,
    observability: DeepSeekV4RunnerObservability,
    completion_hub: CompletionHub,
    expert_completion_reactors: Vec<ModelCompletionReactor>,
    #[cfg(feature = "cuda")]
    expert_io_resource_limits: ferrule_common::materialization_io::MaterializationResourceLimits,
    shutdown: bool,
}

#[cfg(feature = "cuda")]
struct DeepSeekV4ProposalArena {
    stages: Box<[DeepSeekV4LayerArena]>,
    attention: DeepSeekV4ProposalAttentionBuffers,
    head: DeepSeekV4ProposalHeadBuffers,
    hc_state: ferrule_backend::cuda::context::CudaF32Buffer,
}

#[cfg(feature = "cuda")]
struct DeepSeekV4ProposalContinuation {
    transaction: ExecutionTransactionId,
    id: BatchContinuationId,
    anchor_token_id: u32,
    token_ids: Vec<u32>,
    sequence_tokens: usize,
    paged_binding: Option<DeepSeekV4PagedKvBinding>,
    arena: Option<DeepSeekV4ProposalArena>,
    initialized: bool,
    stage: usize,
    current_layer: Option<DeepSeekV4ProposalLayerContinuation>,
    moe_access_events: Vec<DeepSeekV4SequenceMoeAccessEvent>,
    expert_leases: Vec<ResidencyLeaseSet>,
    head_download: Option<ferrule_backend::cuda::context::CudaI32HostDownload>,
    cancel_quiescence: Option<DeepSeekV4CudaComputeQuiescence>,
    paged_binding_active: bool,
    callback_armed: bool,
    dependencies: Option<ContinuationDependencyState>,
    failed: bool,
}

#[cfg(feature = "cuda")]
enum DeepSeekV4ProposalStep {
    Waiting(PendingModelProgress),
    Complete(NativeProposal),
}

#[cfg(feature = "cuda")]
fn plan_cuda_expert_residency(
    resources: &DeepSeekV4PreparedResources,
    free_device_bytes: u64,
) -> Result<DeviceResidencyPlan> {
    let mut layers = BTreeMap::<usize, (usize, u64)>::new();
    for entry in resources.expert_materialization_sources().iter() {
        let source = entry.descriptor();
        let (layer, _) = entry
            .resource()
            .routed_expert_coordinates()
            .ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 routed-expert source has a non-expert resource identity"
                    .into(),
            })?;
        let layer = usize::try_from(layer.get()).map_err(|_| Error::Model {
            message: "DeepSeek-V4 routed-expert execution layer exceeds usize".into(),
        })?;
        let state = layers.entry(layer).or_default();
        state.0 = state.0.checked_add(1).ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 routed-expert count overflow".into(),
        })?;
        state.1 = state.1.max(source.bytes());
    }
    let expected_experts = resources.model().config.num_routed_experts;
    let inventory = layers
        .into_iter()
        .map(|(execution_layer, (expert_count, frame_bytes))| {
            if expert_count != expected_experts {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 execution layer {execution_layer} has {expert_count} routed-expert sources, expected {expected_experts}"
                    ),
                });
            }
            Ok(ExpertLayerInventory {
                execution_layer,
                expert_count,
                frame_bytes,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let manual_slot_cap = (resources.prepare_options().moe_hotset_experts != 0)
        .then_some(resources.prepare_options().moe_hotset_experts);
    plan_device_residency(
        free_device_bytes,
        resources.prepare_options().reserved_device_bytes,
        resources.model().config.num_experts_per_tok,
        manual_slot_cap,
        inventory,
    )
    .map_err(|source| Error::ModelSource {
        source: Box::new(source),
    })
}

#[cfg(feature = "cuda")]
fn residency_requirements(
    model_instance: u64,
    plan: &DeviceResidencyPlan,
) -> Result<ExpertResidencyRequirements> {
    let mut capacities = Vec::with_capacity(plan.layer_slot_capacities.len());
    for (expected_layer, &(execution_layer, capacity)) in
        plan.layer_slot_capacities.iter().enumerate()
    {
        if execution_layer != expected_layer {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 routed-expert execution layers must be contiguous: expected {expected_layer}, found {execution_layer}"
                ),
            });
        }
        capacities.push(capacity);
    }
    Ok(ExpertResidencyRequirements::new(model_instance, capacities))
}

#[cfg(feature = "cuda")]
fn planned_residency_warmup_requests(
    resources: &DeepSeekV4PreparedResources,
    placement: MaterializationPlacement,
    plan: &DeviceResidencyPlan,
) -> Result<Vec<MaterializationRequest>> {
    let mut remaining = plan
        .layer_slot_capacities
        .iter()
        .copied()
        .collect::<BTreeMap<_, _>>();
    let planned_slots = remaining
        .values()
        .copied()
        .try_fold(0usize, |total, slots| {
            total
                .checked_add(slots)
                .ok_or(DeviceResidencyPlanError::WarmupSlotCountOverflow)
        })
        .map_err(|source| Error::ModelSource {
            source: Box::new(source),
        })?;
    let mut requests = Vec::with_capacity(planned_slots);

    for entry in resources.expert_materialization_sources().iter() {
        let (layer, _) = entry
            .resource()
            .routed_expert_coordinates()
            .ok_or_else(|| Error::Model {
                message: "expert materialization catalog contains a non-expert resource".into(),
            })?;
        let layer = usize::try_from(layer.get()).map_err(|_| Error::ModelSource {
            source: Box::new(DeviceResidencyPlanError::WarmupLayerOutOfRange {
                layer: layer.get(),
            }),
        })?;
        let slots = remaining
            .get_mut(&layer)
            .ok_or_else(|| Error::ModelSource {
                source: Box::new(DeviceResidencyPlanError::WarmupUnplannedLayer {
                    execution_layer: layer,
                }),
            })?;
        if *slots == 0 {
            continue;
        }
        requests.push(MaterializationRequest::for_placement(
            placement,
            entry.source(),
            entry.resource(),
        )?);
        *slots -= 1;
    }

    if let Some((&layer, &missing)) = remaining.iter().find(|(_, missing)| **missing != 0) {
        return Err(Error::ModelSource {
            source: Box::new(DeviceResidencyPlanError::WarmupCapacityUnfilled {
                execution_layer: layer,
                missing_slots: missing,
            }),
        });
    }
    Ok(requests)
}

#[cfg(feature = "cuda")]
const PROPOSAL_IMPLEMENTATION: &str = "deepseek-v4-proposal-cuda-cutlass-v1";

fn prepared_proposal_source(
    plan: &DeepSeekV4PreparedModelPlan,
    backend: ModelExecutionBackend,
) -> Result<Option<NativeProposalSource>> {
    let Some(attachment) = plan.resources().proposal_attachment() else {
        return Ok(None);
    };
    if backend != ModelExecutionBackend::Cuda {
        return Err(Error::Model { message:
            "DeepSeek-V4 checkpoint contains a native proposal attachment that requires the CUDA backend"
                .into(),
         });
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = attachment;
        Err(Error::Model { message:
            "DeepSeek-V4 checkpoint contains a native proposal attachment that requires building with CUDA and CUTLASS support"
                .into(),
         })
    }
    #[cfg(feature = "cuda")]
    {
        if attachment.prediction_heads.is_none() {
            return Err(Error::Model {
                message: "DeepSeek-V4 Proposal source is missing prediction heads".into(),
            });
        }
        let source = NativeProposalSource {
            implementation: PROPOSAL_IMPLEMENTATION,
            prepared_plan_id: plan.generation(),
            native_width: attachment.config.block_size,
        };
        source.validate()?;
        Ok(Some(source))
    }
}

fn build_cpu_expert_runtimes(
    backend: ModelExecutionBackend,
    layers: &[DeepSeekV4PreparedLayerExperts],
) -> Option<Box<[DeepSeekV4LayerExpertRuntime]>> {
    (backend == ModelExecutionBackend::Cpu).then(|| {
        layers
            .iter()
            .map(|layer| {
                DeepSeekV4LayerExpertRuntime::from_catalog(
                    std::sync::Arc::clone(layer.source_catalog()),
                    layer.streaming_policy().clone(),
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    })
}

/// Model-local lowering of model-neutral packed execution metadata.
///
/// Rows remain in packed query order. `sequences` supplies the mutable state and
/// query range for each sequence, while `row_to_sequence` makes row-owned CUDA
/// work independent of the aggregate forward mode.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PackedBatchMetadata {
    pub(super) mode: ForwardMode,
    pub(super) sequences: Vec<PackedSequenceMetadata>,
    pub(super) row_to_sequence: Vec<usize>,
    pub(super) sequence_major_rows: Vec<usize>,
    pub(super) max_query_tokens: usize,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PackedSequenceMetadata {
    pub(super) state_index: usize,
    pub(super) phase: ForwardPhase,
    pub(super) query: Range<usize>,
}

#[cfg(any(feature = "cuda", test))]
fn take_packed_cuda_continuation_id(next: &mut NonZeroU64) -> Result<BatchContinuationId> {
    let value = next.get();
    let following = value
        .checked_add(1)
        .and_then(NonZeroU64::new)
        .ok_or_else(|| Error::Execution {
            message: "DeepSeek-V4 packed continuation ID space is exhausted".into(),
        })?;
    let id = BatchContinuationId::new(value);
    *next = following;
    Ok(id)
}

#[cfg(any(feature = "cuda", test))]
fn empty_expert_lease_set(continuation: BatchContinuationId) -> Result<ResidencyLeaseSet> {
    ResidencyLeaseSet::new(
        [],
        [],
        MappingEpoch::new(continuation.get()),
        DispatchFenceContract::new(
            OperationId::new(continuation.get()),
            FenceId::new(continuation.get()),
            BackendId::new(0),
            DeviceId::new(0),
        ),
    )
    .map_err(|error| Error::Internal {
        message: format!("failed to construct an empty DeepSeek-V4 route-poll lease set: {error}"),
    })
}

#[cfg(any(feature = "cuda", test))]
fn take_cuda_backend_once<T, B>(
    backend: ModelExecutionBackend,
    owner: Option<&mut T>,
    take: impl FnOnce(&mut T) -> Option<B>,
) -> Option<B> {
    if backend == ModelExecutionBackend::Cuda {
        owner.and_then(take)
    } else {
        None
    }
}

#[cfg(any(feature = "cuda", test))]
fn install_residency_control_on_owner<T, C>(
    owner: Option<&T>,
    control: C,
    install: impl FnOnce(&T, C) -> Result<()>,
) -> Result<()> {
    let owner = owner.ok_or_else(|| Error::Internal {
        message: "DeepSeek-V4 CUDA expert subsystem owner is unavailable".into(),
    })?;
    install(owner, control)
}

#[cfg(any(feature = "cuda", test))]
fn validate_cuda_backend_shutdown_ownership(
    backend: ModelExecutionBackend,
    owner_present: bool,
    backend_transferred: bool,
    runtime_adapter_installed: bool,
) -> Result<()> {
    if backend != ModelExecutionBackend::Cuda {
        return Ok(());
    }
    if !owner_present {
        return Err(Error::Internal {
            message: "DeepSeek-V4 CUDA runner lost its shared expert subsystem owner".into(),
        });
    }
    if backend_transferred && !runtime_adapter_installed {
        return Err(Error::Execution { message:
            "cannot shut down the DeepSeek-V4 CUDA runner while its transferred expert backend is outside the quiesced runtime adapter"
                .into(),
         });
    }
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn prepared_physical_expert_io_resource_limits(
    resources: &DeepSeekV4PreparedResources,
    reader: &ExpertStreamingReader,
    io_plan: ExpertIoPlan,
) -> Result<ferrule_common::materialization_io::MaterializationResourceLimits> {
    use ferrule_common::materialization_io::{
        MaterializationResourceLimits, MaterializationResourceRequirements,
    };

    let reader_capacity = reader
        .physical_resource_capacity()?
        .ok_or_else(|| Error::Model {
            message:
                "DeepSeek-V4 resident inference requires CUDA-pinned io_uring resource topology"
                    .into(),
        })?;
    let maximum_expert_bytes = resources
        .layer_experts()
        .iter()
        .chain(resources.proposal_stage_experts())
        .flat_map(|layer| {
            layer
                .source_catalog()
                .iter()
                .map(|(_, source)| source.bytes())
        })
        .max()
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 prepared plan has no physical expert-I/O demand".into(),
        })?;
    let upload_slots = u64::try_from(
        resources
            .policy()
            .expert_upload_inflight()
            .checked_add(1)
            .ok_or_else(|| Error::Model {
                message: "expert upload slot capacity overflow".into(),
            })?,
    )
    .map_err(|_| Error::Model {
        message: "expert upload slot capacity exceeds u64".into(),
    })?;
    let transfer_bytes = maximum_expert_bytes
        .checked_mul(upload_slots)
        .ok_or_else(|| Error::Model {
            message: "expert transfer byte capacity overflow".into(),
        })?;
    let capacity = MaterializationResourceRequirements {
        upload_slots,
        h2d_bytes: transfer_bytes,
        install_slots: upload_slots,
        device_install_bytes: transfer_bytes,
        ..reader_capacity
    };
    let execution_reserve = io_plan.execution_reserve(
        maximum_expert_bytes,
        resources.model().config.num_experts_per_tok,
    )?;
    Ok(MaterializationResourceLimits {
        capacity,
        execution_reserve,
    }
    .validate()?)
}

#[cfg(any(feature = "cuda", test))]
fn record_continuation_dependencies(
    state: &mut Option<ContinuationDependencyState>,
    continuation: BatchContinuationId,
    dependencies: DependencySet,
) -> Result<()> {
    match state {
        Some(state) => state.replace_unresolved(dependencies),
        None => {
            *state = Some(ContinuationDependencyState::new(
                continuation,
                dependencies,
            )?);
            Ok(())
        }
    }
}

#[cfg(any(feature = "cuda", test))]
fn internal_continuation_dependency(continuation: BatchContinuationId) -> Result<DependencySet> {
    let dependency = LogicalDependency::operation_retired(OperationId::new(continuation.get()))?;
    Ok(DependencySet::new([dependency])?)
}

#[cfg(any(feature = "cuda", test))]
fn validate_packed_cuda_resume_identity(
    expected_id: BatchContinuationId,
    expected_batch: &ExecutionBatch,
    actual_id: BatchContinuationId,
    actual_batch: &ExecutionBatch,
) -> Result<()> {
    if actual_id != expected_id {
        return Err(Error::Execution {
            message: format!(
                "DeepSeek-V4 packed continuation ID mismatch: expected {}, got {}",
                expected_id.get(),
                actual_id.get()
            ),
        });
    }
    if actual_batch != expected_batch {
        return Err(Error::Execution {
            message:
                "DeepSeek-V4 packed continuation batch does not exactly match the suspended batch"
                    .into(),
        });
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
impl PackedBatchMetadata {
    pub(super) fn lower(batch: &ExecutionBatch, state_count: usize) -> Result<Self> {
        let mut sequences = Vec::with_capacity(batch.sequences().len());
        let mut row_to_sequence = vec![usize::MAX; batch.len()];
        let mut sequence_major_rows = Vec::with_capacity(batch.len());
        let mut expected_start = 0usize;
        let mut max_query_tokens = 0usize;
        let mut state_indices = BTreeSet::new();

        for (sequence_index, sequence) in batch.sequences().iter().enumerate() {
            let state_index = sequence
                .state_slot
                .try_as_usize()
                .map_err(|_| Error::Model {
                    message: "DeepSeek-V4 state slot exceeds usize".into(),
                })?;
            if state_index >= state_count {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 state slot {state_index} is missing from {state_count} states"
                    ),
                });
            }
            if !state_indices.insert(state_index) {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 state slot {state_index} is referenced more than once"
                    ),
                });
            }
            let start = usize::try_from(sequence.query.start).map_err(|_| Error::Model {
                message: "DeepSeek-V4 query start exceeds usize".into(),
            })?;
            let end = usize::try_from(sequence.query.end).map_err(|_| Error::Model {
                message: "DeepSeek-V4 query end exceeds usize".into(),
            })?;
            if start != expected_start || start >= end || end > batch.len() {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 sequence {sequence_index} query range {start}..{end} does not densely cover packed rows from {expected_start}"
                    ),
                });
            }
            for row in start..end {
                row_to_sequence[row] = sequence_index;
                sequence_major_rows.push(row);
            }
            max_query_tokens = max_query_tokens.max(end - start);
            expected_start = end;
            sequences.push(PackedSequenceMetadata {
                state_index,
                phase: sequence.phase,
                query: start..end,
            });
        }
        if expected_start != batch.len() {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 sequence queries cover {expected_start} of {} packed rows",
                    batch.len()
                ),
            });
        }

        Ok(Self {
            mode: batch.mode(),
            sequences,
            row_to_sequence,
            sequence_major_rows,
            max_query_tokens,
        })
    }

    /// Native CUDA keeps row-owned projection/HC/MoE work packed while mutable
    /// recurrent state is advanced once per sequence in query order.
    ///
    /// Single-sequence multi-row batches (Proposal verification: 1 sequence × V
    /// candidate rows) use the same packed path as multi-session batches.
    #[cfg(feature = "cuda")]
    pub(super) fn supports_native_cuda(&self) -> bool {
        !self.row_to_sequence.is_empty()
            && self.sequence_major_rows.len() == self.row_to_sequence.len()
            && self
                .row_to_sequence
                .iter()
                .all(|sequence| *sequence < self.sequences.len())
    }
}

#[cfg(any(feature = "cuda", test))]
pub(super) fn begin_packed_sequence_steps(
    states: &[DeepSeekV4SequenceExecutionState],
    metadata: &PackedBatchMetadata,
) -> Result<Vec<crate::execution::SequenceStepBinding>> {
    metadata
        .sequences
        .iter()
        .map(|sequence| states[sequence.state_index].begin_step())
        .collect()
}

#[cfg(any(feature = "cuda", test))]
pub(super) fn poison_packed_sequence_steps(
    states: &mut [DeepSeekV4SequenceExecutionState],
    metadata: &PackedBatchMetadata,
    bindings: &[crate::execution::SequenceStepBinding],
) {
    for (sequence, binding) in metadata.sequences.iter().zip(bindings.iter().copied()) {
        states[sequence.state_index].poison_step(binding);
    }
}

#[cfg(any(feature = "cuda", test))]
pub(super) fn commit_packed_sequence_steps(
    states: &mut [DeepSeekV4SequenceExecutionState],
    metadata: &PackedBatchMetadata,
    bindings: Vec<crate::execution::SequenceStepBinding>,
) -> Result<()> {
    for (sequence, binding) in metadata.sequences.iter().zip(bindings) {
        states[sequence.state_index].commit_step(binding, sequence.query.len())?;
    }
    Ok(())
}

impl DeepSeekV4Runner {
    pub fn new_with_operator_backend(
        model: DeepSeekV4Checkpoint,
        options: DeepSeekV4PrepareOptions,
        operator_backend: ModelExecutionBackend,
    ) -> Result<Self> {
        Self::new_with_operator_backend_profiled(
            model,
            options,
            operator_backend,
            DeepSeekV4LoadProfile::default(),
        )
    }

    fn new_with_operator_backend_profiled(
        model: DeepSeekV4Checkpoint,
        options: DeepSeekV4PrepareOptions,
        operator_backend: ModelExecutionBackend,
        mut load_profile: DeepSeekV4LoadProfile,
    ) -> Result<Self> {
        let total_start = Instant::now();
        if operator_backend != ModelExecutionBackend::Cuda {
            return Err(Error::Model {
                message:
                    "DeepSeek-V4 resident inference requires the CUDA packed execution backend"
                        .into(),
            });
        }
        let completion_hub = CompletionHub::new();
        let policy_start = Instant::now();
        let policy = DeepSeekV4ExecutionPolicy::resolve()?;
        let policy_resolution_us = duration_us(policy_start.elapsed());
        #[cfg(feature = "cuda")]
        let (plan_result, cuda_result) = std::thread::scope(|scope| {
            let cuda_completion_hub = completion_hub.clone();
            let cuda_policy = policy.clone();
            let cuda_start = Instant::now();
            let cuda = scope.spawn(move || {
                super::cuda_cache::DeepSeekV4CudaOperatorCache::new_with_completion_hub(
                    &cuda_policy,
                    options.expert_memory_policy,
                    cuda_completion_hub,
                )
                .map(|cuda| (cuda, duration_us(cuda_start.elapsed())))
            });
            let plan = prepare_with_policy(model, options, policy.clone(), policy_resolution_us);
            let cuda = cuda.join().map_err(|_| Error::Internal {
                message: "DeepSeek-V4 CUDA initialization thread panicked".into(),
            });
            (plan, cuda)
        });
        #[cfg(feature = "cuda")]
        let (plan, (cuda, cuda_context_us)) = match (plan_result, cuda_result) {
            (Ok(plan), Ok(Ok(cuda))) => (plan, cuda),
            (Err(plan_error), Ok(Err(cuda_error))) => {
                return Err(Error::FailureBatch {
                    operation: "DeepSeek-V4 parallel preparation".into(),
                    failures: vec![plan_error, cuda_error],
                });
            }
            (Err(error), _) | (_, Err(error)) => return Err(error),
            (Ok(_), Ok(Err(error))) => return Err(error),
        };
        #[cfg(not(feature = "cuda"))]
        let plan = prepare_with_policy(model, options, policy.clone(), policy_resolution_us)?;
        load_profile.prepare = plan.resources().prepare_profile();
        #[cfg(feature = "cuda")]
        {
            load_profile.cuda_context_us = cuda_context_us;
        }
        let proposal_source = prepared_proposal_source(&plan, operator_backend)?;
        let options = *plan.resources().prepare_options();
        let policy = plan.resources().policy();
        let model = plan.resources().model();
        let mut operators = DeepSeekV4OperatorContext::from_preinitialized(
            operator_backend,
            policy,
            #[cfg(feature = "cuda")]
            Some(cuda),
        );
        let phase_start = Instant::now();
        let mut layer_states = Vec::with_capacity(options.max_layers);
        for layer_idx in 0..options.max_layers {
            let state_start = operators.profile_start();
            layer_states.push(model.new_layer_sequence_state(layer_idx)?);
            if let Some(state_start) = state_start {
                operators.record_layer_state_init(layer_idx, duration_us(state_start.elapsed()));
            }
        }
        let proposal_stage_states = plan
            .resources()
            .proposal_attachment()
            .map(|attachment| {
                attachment
                    .layers
                    .iter()
                    .map(|stage| DeepSeekV4LayerState::new(stage.transformer.attention.config))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let cpu_expert_runtimes =
            build_cpu_expert_runtimes(operator_backend, plan.resources().layer_experts());
        load_profile.sequence_state_us = duration_us(phase_start.elapsed());

        let sequence = DeepSeekV4SequenceExecutionState::new(
            layer_states,
            proposal_stage_states,
            model.config.num_routed_experts,
        );

        #[cfg(feature = "cuda")]
        let phase_start = Instant::now();
        #[cfg(feature = "cuda")]
        let expert_reader_max_tensor_bytes = options.expert_reader_max_tensor_bytes;
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let (expert_reader, expert_io_plan) = if operator_backend == ModelExecutionBackend::Cuda {
            let allocator = operators
                .cuda
                .as_ref()
                .expect("CUDA backend initialized above")
                .pinned_host_allocator();
            let io_plan = ExpertIoPlan::for_checkpoint_plans(
                plan.resources()
                    .expert_materialization_sources()
                    .iter()
                    .map(|entry| entry.read_plan()),
            )?;
            let (reader, io_plan) = ExpertStreamingReader::from_env_with_cuda_pinned(
                expert_reader_max_tensor_bytes,
                allocator,
                completion_hub.clone(),
                io_plan,
            )?;
            (reader, Some(io_plan))
        } else {
            (
                ExpertStreamingReader::from_env_with_completion_hub(
                    expert_reader_max_tensor_bytes,
                    completion_hub.clone(),
                )?,
                None,
            )
        };
        #[cfg(all(feature = "cuda", not(target_os = "linux")))]
        let expert_reader = if operator_backend == ModelExecutionBackend::Cuda {
            return Err(Error::Model { message:
                "DeepSeek-V4 CUDA resident expert materialization is unsupported on non-Linux platforms; Linux pinned io_uring is required"
                    .into(),
             });
        } else {
            ExpertStreamingReader::from_env_with_completion_hub(
                expert_reader_max_tensor_bytes,
                completion_hub.clone(),
            )?
        };
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let expert_io_resource_limits = match expert_io_plan {
            Some(io_plan) => prepared_physical_expert_io_resource_limits(
                plan.resources(),
                &expert_reader,
                io_plan,
            )?,
            None => ferrule_common::materialization_io::MaterializationResourceLimits::default(),
        };
        #[cfg(all(feature = "cuda", not(target_os = "linux")))]
        let expert_io_resource_limits =
            ferrule_common::materialization_io::MaterializationResourceLimits::default();
        #[cfg(feature = "cuda")]
        let expert_completion_reactors = expert_reader.take_completion_reactors();
        #[cfg(feature = "cuda")]
        {
            load_profile.reader_us = duration_us(phase_start.elapsed());
        }
        #[cfg(not(feature = "cuda"))]
        let expert_completion_reactors = ExpertStreamingReader::new_with_completion_hub(
            options.expert_reader_max_tensor_bytes,
            completion_hub.clone(),
        )
        .take_completion_reactors();

        #[cfg(feature = "cuda")]
        let model_instance = NEXT_DSV4_MODEL_INSTANCE.fetch_add(1, Ordering::Relaxed);
        let mut runner = Self {
            plan,
            operators,
            cpu_expert_runtimes,
            materialization_resolver: None,

            #[cfg(feature = "cuda")]
            cuda_materialization_owner: None,
            #[cfg(feature = "cuda")]
            materialization_provider_transferred: false,
            #[cfg(feature = "cuda")]
            expert_residency_requirements: None,
            #[cfg(feature = "cuda")]
            warmup_requests: Vec::new(),
            #[cfg(feature = "cuda")]
            device_residency_plan: None,
            #[cfg(feature = "cuda")]
            layer_arena_pool: PersistentArenaPool::new(),
            #[cfg(feature = "cuda")]
            packed_transactions: DeepSeekV4PackedTransactionRegistry::default(),
            #[cfg(feature = "cuda")]
            next_packed_cuda_continuation_id: NonZeroU64::new(1).expect("one is non-zero"),
            #[cfg(feature = "cuda")]
            proposal_arena_pool: Vec::new(),
            #[cfg(feature = "cuda")]
            proposal_continuations: HashMap::new(),
            proposal_source,
            sequence,
            observability: DeepSeekV4RunnerObservability {
                load: load_profile,
                ..DeepSeekV4RunnerObservability::new()
            },
            completion_hub,
            expert_completion_reactors,
            #[cfg(feature = "cuda")]
            expert_io_resource_limits,
            shutdown: false,
        };
        #[cfg(feature = "cuda")]
        if operator_backend == ModelExecutionBackend::Cuda {
            // Static bundle descriptors share the provider catalog, but their generic
            // transport/install path is not connected yet. Compile them first so the
            // residency budget is based on the actual remaining device memory.
            let image_profile = runner
                .operators
                .compile_cuda_execution_image(runner.plan.generation(), runner.plan.resources())?;
            runner.observability.load.static_image_globals_us = image_profile.globals_us;
            runner.observability.load.static_image_embedding_us = image_profile.embedding_us;
            runner.observability.load.static_image_output_head_us = image_profile.output_head_us;
            runner.observability.load.static_image_target_layers_us =
                image_profile.target_layers_us;
            runner.observability.load.static_image_attachment_us = image_profile.attachment_us;
            runner.observability.load.static_image_total_us = image_profile.total_us;
            let residency_start = Instant::now();
            let (free_device_bytes, total_device_bytes) = runner.operators.cuda_memory_info()?;
            let free_device_bytes = u64::try_from(free_device_bytes).map_err(|_| Error::Model {
                message: "CUDA free-memory byte count exceeds u64".into(),
            })?;
            let total_device_bytes =
                u64::try_from(total_device_bytes).map_err(|_| Error::Model {
                    message: "CUDA total-memory byte count exceeds u64".into(),
                })?;
            let residency_plan =
                plan_cuda_expert_residency(runner.plan.resources(), free_device_bytes)?;
            let requirements = residency_requirements(model_instance, &residency_plan)?;
            let limits = runner.expert_io_resource_limits;
            let placement = MaterializationPlacement::new(
                ModelInstanceId::new(model_instance),
                BackendId::new(1),
                DeviceId::new(0),
            )?;
            let warmup_requests = planned_residency_warmup_requests(
                runner.plan.resources(),
                placement,
                &residency_plan,
            )?;
            let materialization_sources =
                std::sync::Arc::clone(runner.plan.resources().expert_materialization_sources());
            let expert_capacity = runner.plan.resources().model().config.num_routed_experts;
            let consumer_compute = runner.operators.cuda_compute_stream_authority()?;
            let owner = CudaExpertMaterializationOwner::create(
                placement,
                limits,
                materialization_sources,
                expert_reader,
                expert_capacity,
                &residency_plan.layer_slot_capacities,
                consumer_compute,
            )?;
            runner
                .operators
                .configure_expert_subsystem(owner.handle())?;
            tracing::info!(
                device = placement.device().get(),
                free_device_bytes,
                total_device_bytes,
                reserved_dynamic_bytes = residency_plan.reserved_dynamic_bytes,
                expert_budget_bytes = residency_plan.expert_budget_bytes,
                planned_expert_bytes = residency_plan.planned_expert_bytes,
                expert_slots = residency_plan
                    .layer_slot_capacities
                    .iter()
                    .map(|(_, capacity)| *capacity)
                    .sum::<usize>(),
                fully_resident = residency_plan.fully_resident,
                warmup_resources = warmup_requests.len(),
                "planned DeepSeek-V4 device residency"
            );
            runner.expert_residency_requirements = Some(requirements);
            runner.warmup_requests = warmup_requests;
            runner.device_residency_plan = Some(residency_plan);
            runner.cuda_materialization_owner = Some(owner);
            runner.observability.load.residency_us = duration_us(residency_start.elapsed());
        }

        runner.observability.load.total_us = runner
            .observability
            .load
            .checkpoint_us
            .saturating_add(duration_us(total_start.elapsed()));
        Ok(runner)
    }

    pub fn load_hf_with_options(
        model_dir: &Path,
        max_tensor_bytes: u64,
        options: DeepSeekV4PrepareOptions,
    ) -> Result<Self> {
        Self::new_with_operator_backend(
            DeepSeekV4Checkpoint::load_hf_with_limit(model_dir, max_tensor_bytes)?,
            options,
            ModelExecutionBackend::Cuda,
        )
    }

    pub fn load_hf_with_options_and_backend(
        model_dir: &Path,
        max_tensor_bytes: u64,
        options: DeepSeekV4PrepareOptions,
        operator_backend: ModelExecutionBackend,
    ) -> Result<Self> {
        let checkpoint_start = Instant::now();
        let model = DeepSeekV4Checkpoint::load_hf_with_limit(model_dir, max_tensor_bytes)?;
        let load_profile = DeepSeekV4LoadProfile {
            checkpoint_us: duration_us(checkpoint_start.elapsed()),
            ..DeepSeekV4LoadProfile::default()
        };
        Self::new_with_operator_backend_profiled(model, options, operator_backend, load_profile)
    }

    pub fn model(&self) -> &DeepSeekV4Checkpoint {
        self.plan.resources().model()
    }

    pub fn proposal_attachment(&self) -> Option<&DeepSeekV4ProposalAttachment> {
        self.plan.resources().proposal_attachment()
    }

    #[cfg(feature = "cuda")]
    fn build_proposal_arena(&mut self) -> Result<DeepSeekV4ProposalArena> {
        let resources = self.plan.resources();
        let attachment = resources
            .proposal_attachment()
            .ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 Proposal attachment is missing".into(),
            })?;
        let rows = ferrule_backend::cuda::cutlass::PROPOSAL_ROWS;
        let mut stages = Vec::with_capacity(attachment.layers.len());
        for layer in &attachment.layers {
            stages.push(DeepSeekV4LayerArena::new(
                &layer.transformer,
                rows,
                true,
                true,
                &mut self.operators,
            )?);
        }
        let attention = self
            .operators
            .cuda_mut()?
            .allocate_proposal_attention_buffers()?;
        let head = self
            .operators
            .cuda_mut()?
            .allocate_proposal_head_buffers()?;
        let hc_state = self.operators.cuda_mut()?.ops.zero_f32_buffer(
            rows.checked_mul(resources.model().config.hc_config().hc_hidden_size())
                .ok_or_else(|| Error::Model {
                    message: "Proposal HC size overflow".into(),
                })?,
        )?;
        Ok(DeepSeekV4ProposalArena {
            stages: stages.into_boxed_slice(),
            attention,
            head,
            hc_state,
        })
    }

    #[cfg(feature = "cuda")]
    fn begin_proposal_continuation(
        &mut self,
        transaction: ExecutionTransactionId,
        anchor_token_id: u32,
    ) -> Result<DeepSeekV4ProposalContinuation> {
        if self.operators.backend() != ModelExecutionBackend::Cuda {
            return Err(Error::Model {
                message: "DeepSeek-V4 Proposal production proposal requires CUDA".into(),
            });
        }
        if !self
            .cuda_materialization_owner
            .as_ref()
            .is_some_and(CudaExpertMaterializationOwner::residency_control_installed)
        {
            return Err(Error::Execution {
                message: "runtime expert residency controller is not installed".into(),
            });
        }
        let (protocol, stage_count) = {
            let attachment =
                self.plan
                    .resources()
                    .proposal_attachment()
                    .ok_or_else(|| Error::Model {
                        message: "DeepSeek-V4 Proposal attachment is missing".into(),
                    })?;
            (
                DeepSeekV4ProposalProtocol::try_from(&attachment.config)?,
                attachment.layers.len(),
            )
        };
        let token_ids = protocol.draft_input_ids(anchor_token_id);
        let sequence_tokens = self.sequence.core.position();
        if sequence_tokens == 0 {
            return Err(Error::Model {
                message: "DeepSeek-V4 Proposal requires committed target context".into(),
            });
        }
        let paged_binding = self.sequence.paged_kv_binding.clone();
        let id = take_packed_cuda_continuation_id(&mut self.next_packed_cuda_continuation_id)?;
        let arena = match self.proposal_arena_pool.pop() {
            Some(arena) => arena,
            None => self.build_proposal_arena()?,
        };
        let arena_stage_count = arena.stages.len();
        if arena_stage_count != stage_count
            || stage_count != self.plan.resources().proposal_stage_experts().len()
        {
            self.proposal_arena_pool.push(arena);
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 Proposal arena mismatch: arenas={arena_stage_count} stages={stage_count} expert_catalogs={}",
                    self.plan.resources().proposal_stage_experts().len()
                ),
            });
        }
        Ok(DeepSeekV4ProposalContinuation {
            transaction,
            id,
            anchor_token_id,
            token_ids,
            sequence_tokens,
            paged_binding,
            arena: Some(arena),
            initialized: false,
            stage: 0,
            current_layer: None,
            moe_access_events: Vec::with_capacity(stage_count),
            expert_leases: Vec::new(),
            head_download: None,
            cancel_quiescence: None,
            paged_binding_active: false,
            callback_armed: false,
            dependencies: None,
            failed: false,
        })
    }

    #[cfg(feature = "cuda")]
    fn activate_proposal_binding(
        &mut self,
        continuation: &mut DeepSeekV4ProposalContinuation,
    ) -> Result<()> {
        self.operators.cuda_mut()?.activate_paged_binding(
            continuation.transaction,
            continuation.paged_binding.as_ref(),
        )?;
        continuation.paged_binding_active = true;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn deactivate_proposal_binding(
        &mut self,
        continuation: &mut DeepSeekV4ProposalContinuation,
    ) -> Result<()> {
        if continuation.paged_binding_active {
            self.operators
                .cuda_mut()?
                .activate_paged_binding(continuation.transaction, None)?;
            continuation.paged_binding_active = false;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn pending_proposal_progress(
        &mut self,
        continuation: &mut DeepSeekV4ProposalContinuation,
    ) -> Result<PendingModelProgress> {
        let pending = continuation
            .current_layer
            .as_ref()
            .map(DeepSeekV4ProposalLayerContinuation::pending_experts)
            .unwrap_or_default();
        let resolved_stage = if pending.is_empty() {
            None
        } else {
            let prepared = self
                .plan
                .resources()
                .proposal_stage_experts()
                .get(continuation.stage)
                .ok_or_else(|| Error::Internal {
                    message: format!(
                        "DeepSeek-V4 Proposal stage {} has no prepared expert catalog",
                        continuation.stage
                    ),
                })?;
            let requests = pending
                .into_iter()
                .map(|operation| {
                    let layer = u32::try_from(operation.layer).map_err(|_| Error::Execution {
                        message: format!(
                            "DeepSeek-V4 pending expert layer {} exceeds u32 ABI",
                            operation.layer
                        ),
                    })?;
                    let expert = u32::try_from(operation.expert).map_err(|_| Error::Execution {
                        message: format!(
                            "DeepSeek-V4 pending expert {} exceeds u32 ABI",
                            operation.expert
                        ),
                    })?;
                    let resource_source = prepared.source_catalog().require_resource_source(
                        crate::moe::streaming::ExpertId::new(operation.layer, operation.expert),
                    )?;
                    Ok((resource_source, layer, expert))
                })
                .collect::<Result<Vec<_>>>()?;
            let resolver = self
                .materialization_resolver
                .as_deref_mut()
                .ok_or_else(|| {
                    Error::Execution { message:
                    "DeepSeek-V4 exact resource wait requires an installed materialization resolver"
                        .into(),
                 }
                })?;
            let placement = resolver.placement();
            let resources = requests
                .into_iter()
                .map(|(resource_source, layer, expert)| {
                    let resource = ferrule_common::MaterializedResourceId::routed_expert(
                        ferrule_common::LayerId::new(layer),
                        ferrule_common::ExpertId::new(expert),
                    );
                    StageMaterializationRequest::new(
                        StageResourceUse::new(
                            resource,
                            ResourceAccess::Read,
                            ResourceRetention::ThroughTransaction,
                        ),
                        MaterializationRequest::for_placement(
                            placement,
                            resource_source,
                            resource,
                        )?,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            Some(resolve_stage_resources(
                resources,
                WorkspaceClaim::NONE,
                resolver,
            )?)
        };
        let dependencies = match &resolved_stage {
            Some(stage) => stage
                .dependencies()
                .expect("selected experts produce residency dependencies")
                .clone(),
            None => internal_continuation_dependency(continuation.id)?,
        };
        record_continuation_dependencies(
            &mut continuation.dependencies,
            continuation.id,
            dependencies.clone(),
        )?;
        match resolved_stage {
            Some(stage) => PendingModelProgress::for_resolved_stage(
                continuation.transaction,
                continuation.id,
                stage,
            ),
            None => {
                PendingModelProgress::new(continuation.transaction, continuation.id, dependencies)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn poll_proposal_route_cancel_ready(
        &mut self,
        continuation: &mut DeepSeekV4ProposalContinuation,
    ) -> Result<bool> {
        let Some(layer_continuation) = continuation.current_layer.as_mut() else {
            return Ok(true);
        };
        let stage = continuation.stage;
        let layer = self
            .plan
            .resources()
            .proposal_attachment()
            .and_then(|attachment| attachment.layers.get(stage))
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "current Proposal stage {stage} is outside the prepared attachment"
                ),
            })?;
        layer
            .transformer
            .poll_proposal_cancel_ready(layer_continuation, &mut self.operators)
    }

    #[cfg(feature = "cuda")]
    fn arm_proposal_completion(&mut self, continuation: &mut DeepSeekV4ProposalContinuation) {
        let notify = crate::runner::completion_notify_callback(self.completion_hub.clone());
        continuation.callback_armed = self
            .operators
            .cuda_mut()
            .and_then(|cuda| cuda.ops.notify_control_stream(notify))
            .is_ok();
        if !continuation.callback_armed {
            self.completion_hub.notify();
        }
    }

    #[cfg(feature = "cuda")]
    fn progress_proposal(
        &mut self,
        continuation: &mut DeepSeekV4ProposalContinuation,
        resume_leases: Option<ResidencyLeaseSet>,
    ) -> Result<DeepSeekV4ProposalStep> {
        if continuation.failed {
            return Err(Error::Execution {
                message: format!(
                    "DeepSeek-V4 native proposal continuation {} previously failed and must be aborted",
                    continuation.id.get()
                ),
            });
        }
        let stage_count = self
            .plan
            .resources()
            .proposal_attachment()
            .ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 Proposal attachment is missing".into(),
            })?
            .layers
            .len();
        if !continuation.initialized {
            let arena = continuation.arena.as_mut().ok_or_else(|| Error::Internal {
                message: "DeepSeek-V4 Proposal arena is unavailable".into(),
            })?;
            self.operators
                .cuda_mut()?
                .proposal_input_device_into(continuation.anchor_token_id, &mut arena.hc_state)?;
            continuation.initialized = true;
        }

        let mut resume_leases = resume_leases;
        while continuation.stage < stage_count {
            let stage = continuation.stage;
            let leases = if continuation.current_layer.is_some() {
                resume_leases.take().ok_or_else(|| Error::Execution {
                    message: format!(
                        "DeepSeek-V4 Proposal continuation {} has no lease set for its current layer",
                        continuation.id.get()
                    ),
                })?
            } else {
                let attachment = self
                    .plan
                    .resources()
                    .proposal_attachment()
                    .expect("validated Proposal attachment above");
                let layer = &attachment.layers[stage].transformer;
                let arena = continuation.arena.as_mut().ok_or_else(|| Error::Internal {
                    message: "DeepSeek-V4 Proposal arena is unavailable".into(),
                })?;
                let stage_arena = arena.stages.get_mut(stage).ok_or_else(|| Error::Internal {
                    message: format!("DeepSeek-V4 Proposal stage arena {stage} is unavailable"),
                })?;
                match layer.begin_proposal_block_device_hc_device(
                    stage,
                    continuation.sequence_tokens,
                    stage_arena,
                    &mut arena.hc_state,
                    &continuation.token_ids,
                    &mut self.operators,
                    &mut arena.attention,
                )? {
                    DeepSeekV4ProposalLayerProgress::Waiting(next) => {
                        continuation.current_layer = Some(next);
                        if continuation
                            .current_layer
                            .as_ref()
                            .is_some_and(|current| !current.pending_experts().is_empty())
                        {
                            return Ok(DeepSeekV4ProposalStep::Waiting(
                                self.pending_proposal_progress(continuation)?,
                            ));
                        }
                        empty_expert_lease_set(continuation.id)?
                    }
                    DeepSeekV4ProposalLayerProgress::Complete { events, leases } => {
                        continuation.moe_access_events.extend(events);
                        continuation.expert_leases.extend(leases);
                        continuation.stage += 1;
                        continue;
                    }
                }
            };

            let attachment = self
                .plan
                .resources()
                .proposal_attachment()
                .expect("validated Proposal attachment above");
            let layer = &attachment.layers[stage].transformer;
            let arena = continuation.arena.as_mut().ok_or_else(|| Error::Internal {
                message: "DeepSeek-V4 Proposal arena is unavailable".into(),
            })?;
            let stage_arena = arena.stages.get_mut(stage).ok_or_else(|| Error::Internal {
                message: format!("DeepSeek-V4 Proposal stage arena {stage} is unavailable"),
            })?;
            let layer_continuation = continuation
                .current_layer
                .take()
                .expect("Proposal layer continuation was created above");
            match layer.resume_proposal_block_device_hc_device(
                layer_continuation,
                leases,
                stage_arena,
                &mut arena.hc_state,
                &mut self.operators,
            )? {
                DeepSeekV4ProposalLayerProgress::Waiting(next) => {
                    continuation.current_layer = Some(next);
                    return Ok(DeepSeekV4ProposalStep::Waiting(
                        self.pending_proposal_progress(continuation)?,
                    ));
                }
                DeepSeekV4ProposalLayerProgress::Complete { events, leases } => {
                    continuation.moe_access_events.extend(events);
                    continuation.expert_leases.extend(leases);
                    continuation.stage += 1;
                }
            }
        }
        if let Some(unused) = resume_leases {
            if !unused.is_empty() {
                return Err(Error::Execution {
                    message: format!(
                        "DeepSeek-V4 Proposal continuation {} received expert leases without a current layer",
                        continuation.id.get()
                    ),
                });
            }
        }

        if continuation.head_download.is_none() {
            let arena = continuation.arena.as_mut().ok_or_else(|| Error::Internal {
                message: "DeepSeek-V4 Proposal arena is unavailable".into(),
            })?;
            self.operators.cuda_mut()?.proposal_head_device_into(
                continuation.anchor_token_id,
                &arena.hc_state,
                &mut arena.head,
            )?;
            continuation.head_download = Some(
                self.operators
                    .cuda_mut()?
                    .begin_proposal_head_result_download(&mut arena.head)?,
            );
            self.arm_proposal_completion(continuation);
            let dependencies = internal_continuation_dependency(continuation.id)?;
            record_continuation_dependencies(
                &mut continuation.dependencies,
                continuation.id,
                dependencies.clone(),
            )?;
            return Ok(DeepSeekV4ProposalStep::Waiting(PendingModelProgress::new(
                continuation.transaction,
                continuation.id,
                dependencies,
            )?));
        }

        let compact = {
            let arena = continuation.arena.as_mut().ok_or_else(|| Error::Internal {
                message: "DeepSeek-V4 Proposal arena is unavailable".into(),
            })?;
            let download = continuation
                .head_download
                .as_ref()
                .expect("checked Proposal head download");
            self.operators
                .cuda_mut()?
                .poll_proposal_head_result(&mut arena.head, download)?
        };
        let Some(compact) = compact else {
            if !continuation.callback_armed {
                self.arm_proposal_completion(continuation);
            }
            let dependencies = internal_continuation_dependency(continuation.id)?;
            record_continuation_dependencies(
                &mut continuation.dependencies,
                continuation.id,
                dependencies.clone(),
            )?;
            return Ok(DeepSeekV4ProposalStep::Waiting(PendingModelProgress::new(
                continuation.transaction,
                continuation.id,
                dependencies,
            )?));
        };
        continuation.head_download = None;
        let (token_ids, confidence_logits) = self
            .operators
            .cuda_mut()?
            .decode_proposal_head_result(compact)?;
        let proposal = NativeProposal {
            token_ids,
            confidence_logits,
        };
        let source = self.proposal_source.ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 has no checkpoint-native proposal capability".into(),
        })?;
        proposal.validate_for_source(source)?;

        Ok(DeepSeekV4ProposalStep::Complete(proposal))
    }

    #[cfg(feature = "cuda")]
    fn release_proposal_arena(
        &mut self,
        continuation: &mut DeepSeekV4ProposalContinuation,
    ) -> Result<()> {
        let arena = continuation.arena.take().ok_or_else(|| Error::Internal {
            message: "DeepSeek-V4 Proposal arena was already released".into(),
        })?;
        self.proposal_arena_pool.push(arena);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cleanup_unpublished_proposal(
        &mut self,
        continuation: &mut DeepSeekV4ProposalContinuation,
        error: Error,
    ) -> Error {
        let mut cleanup_errors = Vec::new();
        if let Some(layer_continuation) = continuation.current_layer.take() {
            let stage = continuation.stage;
            match self
                .plan
                .resources()
                .proposal_attachment()
                .and_then(|attachment| attachment.layers.get(stage))
            {
                Some(layer) => {
                    if let Err(cancel_error) =
                        layer.transformer.cancel_proposal_block_device_hc_device(
                            layer_continuation,
                            &mut self.operators,
                        )
                    {
                        cleanup_errors.push(Error::context(
                            format!("stage {stage} MoE cancel"),
                            cancel_error,
                        ));
                    }
                }
                None => cleanup_errors.push(Error::Internal {
                    message: format!(
                        "current Proposal stage {stage} is outside the prepared attachment"
                    ),
                }),
            }
        }
        if let Err(deactivate_error) = self.deactivate_proposal_binding(continuation) {
            cleanup_errors.push(Error::context(
                "paged binding deactivation",
                deactivate_error,
            ));
        }
        self.discard_continuation_dependencies(&mut continuation.dependencies);
        if let Some(arena) = continuation.arena.take() {
            self.proposal_arena_pool.push(arena);
        }
        Error::with_cleanup(
            "DeepSeek-V4 native proposal",
            error,
            Error::failures("DeepSeek-V4 native proposal cleanup", cleanup_errors),
        )
    }

    pub fn proposal_context_kv_lengths(&self) -> Vec<usize> {
        self.sequence
            .proposal_stages
            .iter()
            .map(|state| state.kv.len())
            .collect()
    }

    pub fn prepare_options(&self) -> &DeepSeekV4PrepareOptions {
        self.plan.resources().prepare_options()
    }

    pub fn execution_policy(&self) -> &DeepSeekV4ExecutionPolicy {
        self.plan.resources().policy()
    }

    pub fn kv_layout_schema(&self) -> &super::prepared::DeepSeekV4KvLayoutSchema {
        self.plan.resources().kv_layout()
    }

    pub fn operator_backend(&self) -> ModelExecutionBackend {
        self.operators.backend()
    }

    fn ensure_sequence_topology_available(
        &self,
        state: &DeepSeekV4SequenceExecutionState,
        operation: &str,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        self.packed_transactions
            .ensure_sequence_available(state, operation)?;
        let _ = (state, operation);
        Ok(())
    }

    fn ensure_kv_pages_available(
        &self,
        pages: &[ferrule_common::execution::KvPageId],
        operation: &str,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        self.packed_transactions
            .ensure_pages_available(pages, operation)?;
        let _ = (pages, operation);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn ensure_no_proposal_continuations(&self, operation: &str) -> Result<()> {
        if !self.proposal_continuations.is_empty() {
            return Err(Error::Execution {
                message: format!(
                    "cannot {operation} while {} DeepSeek-V4 native proposal continuation(s) are outstanding",
                    self.proposal_continuations.len()
                ),
            });
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_failpoints(&self) -> Result<&ferrule_backend::cuda::CudaFailpoints> {
        self.operators.cuda_failpoints()
    }

    pub fn operator_runtime_counters(&self) -> DeepSeekV4OperatorRuntimeCounters {
        let mut counters = self.operators.runtime_counters();
        #[cfg(feature = "cuda")]
        if let Some(owner) = self.cuda_materialization_owner.as_ref() {
            counters.expert_residency_stats = owner.residency_stats();
        }
        counters.expert_predictor_stats = self.sequence.predictor.stats();
        counters
    }

    pub fn layer_profile_stats(&self) -> Vec<DeepSeekV4LayerProfileStats> {
        self.operators.layer_profile_stats()
    }

    pub fn attention_profile_stats(&self) -> Vec<DeepSeekV4AttentionProfileStats> {
        self.operators.attention_profile_stats()
    }

    pub fn output_profile_stats(&self) -> DeepSeekV4OutputProfileStats {
        self.observability.output
    }

    pub fn position(&self) -> usize {
        self.sequence.core.position()
    }

    pub fn observability_snapshot(&self) -> DeepSeekV4ObservabilitySnapshot {
        DeepSeekV4ObservabilitySnapshot {
            position: self.position(),
            load: self.observability.load,
            operator: self.operator_runtime_counters(),
            layers: self.layer_profile_stats(),
            attention: self.attention_profile_stats(),
            output: self.output_profile_stats(),
            layer_runtime: self.layer_runtime_stats(),
        }
    }

    /// Construct a fresh serving sequence without cloning default-session KV.
    pub fn create_sequence_state(&self) -> Result<DeepSeekV4SequenceExecutionState> {
        let resources = self.plan.resources();
        let mut layers = Vec::with_capacity(resources.prepare_options().max_layers);
        for layer in 0..resources.prepare_options().max_layers {
            layers.push(resources.model().new_layer_sequence_state(layer)?);
        }
        let proposal_stages = resources
            .proposal_attachment()
            .map(|attachment| {
                attachment
                    .layers
                    .iter()
                    .map(|stage| DeepSeekV4LayerState::new(stage.transformer.attention.config))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Ok(DeepSeekV4SequenceExecutionState::new(
            layers,
            proposal_stages,
            resources.model().config.num_routed_experts,
        ))
    }

    /// Fork the default runner sequence as a runtime-shared paged prefix.
    pub fn fork_sequence_state(&mut self) -> Result<DeepSeekV4SequenceExecutionState> {
        self.ensure_sequence_topology_available(&self.sequence, "fork the active sequence")?;
        let position = self.sequence.position();
        Self::fork_sequence_state_from_explicit(&self.sequence, position, &self.operators)
    }

    fn fork_sequence_state_from_explicit(
        source: &DeepSeekV4SequenceExecutionState,
        expected_position: usize,
        operators: &DeepSeekV4OperatorContext,
    ) -> Result<DeepSeekV4SequenceExecutionState> {
        source.begin_step()?;
        if source.position() != expected_position {
            return Err(Error::Execution {
                message: format!(
                    "DeepSeek-V4 exact fork expected committed position {expected_position}, source is at {}",
                    source.position()
                ),
            });
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(source.layers.len())
            .map_err(|error| Error::Model {
                message: format!(
                    "DeepSeek-V4 paged fork layer metadata allocation failed: {error}"
                ),
            })?;
        for layer in &source.layers {
            layers.push(layer.fork_paged_prefix_metadata(operators)?);
        }
        let proposal_stages = source
            .proposal_stages
            .iter()
            .map(|stage| stage.fork_paged_prefix_metadata(operators))
            .collect::<Result<Vec<_>>>()?;
        Ok(DeepSeekV4SequenceExecutionState {
            #[cfg(any(feature = "cuda", test))]
            topology_id: DeepSeekV4SequenceTopologyId::take(),
            core: source.core.forked()?,
            layers,
            proposal_stages,
            predictor: source.predictor.clone(),
            #[cfg(feature = "cuda")]
            paged_kv_binding: None,
        })
    }

    #[cfg(feature = "cuda")]
    fn checkout_transaction_sequence_topology(
        source: &DeepSeekV4SequenceExecutionState,
        operators: &DeepSeekV4OperatorContext,
    ) -> Result<DeepSeekV4SequenceExecutionState> {
        source.begin_step()?;
        let layers = source
            .layers
            .iter()
            .map(|layer| layer.fork_paged_prefix_metadata(operators))
            .collect::<Result<Vec<_>>>()?;
        let proposal_stages = source
            .proposal_stages
            .iter()
            .map(|stage| stage.fork_paged_prefix_metadata(operators))
            .collect::<Result<Vec<_>>>()?;
        Ok(DeepSeekV4SequenceExecutionState {
            topology_id: source.topology_id(),
            core: source.core.clone(),
            layers,
            proposal_stages,
            predictor: source.predictor.clone(),
            paged_kv_binding: source.paged_kv_binding.clone(),
        })
    }

    /// Execute serially with an explicit sequence while retaining the runner's
    /// prepared layers, weights, expert residency, and backend scratch resources.
    pub fn with_sequence_state<T>(
        &mut self,
        state: &mut DeepSeekV4SequenceExecutionState,
        execute: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        if self.shutdown {
            return Err(Error::Model {
                message: "DeepSeek-V4 runner is shut down".into(),
            });
        }

        if state.max_layers() != self.sequence.max_layers() {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 sequence layer count {} does not match runner layer count {}",
                    state.max_layers(),
                    self.sequence.max_layers()
                ),
            });
        }
        if state.proposal_stage_count() != self.sequence.proposal_stage_count() {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 sequence Proposal stage count {} does not match runner stage count {}",
                    state.proposal_stage_count(),
                    self.sequence.proposal_stage_count()
                ),
            });
        }
        self.ensure_sequence_topology_available(state, "execute serial sequence work")?;
        state.begin_step()?;
        self.discard_pending_moe_access_events();
        std::mem::swap(&mut self.sequence, state);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(self)));
        match &result {
            Ok(Ok(_)) => self.observe_pending_moe_access_events(),
            Ok(Err(_)) | Err(_) => self.discard_pending_moe_access_events(),
        }
        std::mem::swap(&mut self.sequence, state);
        match result {
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    pub fn release_sequence_state(
        &mut self,
        mut state: DeepSeekV4SequenceExecutionState,
    ) -> Result<()> {
        self.ensure_sequence_topology_available(&state, "release sequence state")?;
        state.release_capacity();
        Ok(())
    }

    /// Reset a sequence state for reuse with a new logical sequence.
    pub fn reset_sequence_state(
        &mut self,
        state: &mut DeepSeekV4SequenceExecutionState,
    ) -> Result<()> {
        self.ensure_sequence_topology_available(state, "reset sequence state")?;
        state.reset_for_reuse();
        Ok(())
    }

    pub fn shutdown(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        if !self.packed_transactions.is_empty() {
            return Err(Error::Execution { message:
                "cannot shut down the DeepSeek-V4 runner before packed transactions are quiesced"
                    .into(),
             });
        }
        #[cfg(feature = "cuda")]
        self.ensure_no_proposal_continuations("shut down the runner")?;
        if self.shutdown {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        validate_cuda_backend_shutdown_ownership(
            self.operators.backend(),
            self.cuda_materialization_owner.is_some(),
            self.materialization_provider_transferred,
            self.materialization_resolver.is_some(),
        )?;

        self.sequence.release_capacity();
        self.cpu_expert_runtimes = None;
        #[cfg(feature = "cuda")]
        {
            self.layer_arena_pool.clear();
            #[cfg(feature = "cuda")]
            {
                self.proposal_arena_pool.clear();
                self.proposal_continuations.clear();
            }
        }
        self.operators.shutdown()?;
        self.materialization_resolver = None;
        #[cfg(feature = "cuda")]
        if self.operators.backend() == ModelExecutionBackend::Cuda {
            let mut owner = self.cuda_materialization_owner.take().ok_or_else(|| {
                Error::Internal { message:
                    "DeepSeek-V4 CUDA runner lost its shared expert subsystem owner during shutdown"
                        .into(),
                 }
            })?;
            drop(owner.take_provider());
        }
        self.shutdown = true;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        if self.shutdown {
            return Err(Error::Model {
                message: "DeepSeek-V4 runner is shut down".into(),
            });
        }
        self.ensure_sequence_topology_available(&self.sequence, "reset the runner sequence")?;
        self.sequence.reset_for_reuse();
        Ok(())
    }

    fn observe_pending_moe_access_events(&mut self) {
        for event in self.operators.drain_moe_access_events() {
            self.sequence.predictor.observe_batch(event);
        }
    }

    fn discard_pending_moe_access_events(&mut self) {
        self.operators.drain_moe_access_events();
    }

    pub fn bound_layer_count(&self) -> usize {
        self.plan.resources().layers().len()
    }

    pub fn layer_runtime_stats(&self) -> Vec<DeepSeekV4LayerRuntimeStats> {
        let mut stats = Vec::new();
        for layer_idx in 0..self.plan.resources().prepare_options().max_layers {
            let layer = &self.plan.resources().layers()[layer_idx];
            let state = &self.sequence.layers[layer_idx];
            let cpu_expert_runtime = self
                .cpu_expert_runtimes
                .as_deref()
                .and_then(|runtimes| runtimes.get(layer_idx));
            let index_head_dim = layer.attention.config.index_head_dim;
            let (resident_experts, resident_expert_bytes) = {
                #[cfg(feature = "cuda")]
                {
                    if let Some(cache) = self.operators.cuda.as_ref() {
                        cache.resident_expert_stats_for_layer(layer_idx)
                    } else {
                        cpu_expert_runtime
                            .map(|runtime| {
                                (
                                    runtime.expert_handles.len(),
                                    runtime.expert_handles.total_bytes(),
                                )
                            })
                            .unwrap_or((0, 0))
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    cpu_expert_runtime
                        .map(|runtime| {
                            (
                                runtime.expert_handles.len(),
                                runtime.expert_handles.total_bytes(),
                            )
                        })
                        .unwrap_or((0, 0))
                }
            };
            stats.push(DeepSeekV4LayerRuntimeStats {
                layer: layer_idx,
                window_kv_len: state.kv.len(),
                compressed_kv_len: state.kv.compressed_len(),
                indexer_compressed_kv_len: state.kv.indexer_compressed_len(index_head_dim),
                resident_experts,
                resident_expert_bytes,
            });
        }
        stats
    }

    #[cfg(feature = "cuda")]
    fn packed_cuda_cleanup_result(context: &str, errors: Vec<Error>) -> Result<()> {
        Error::failures(context, errors)
    }

    #[cfg(feature = "cuda")]
    fn packed_cuda_error_with_cleanup(context: &str, error: Error, cleanup: Result<()>) -> Error {
        Error::with_cleanup(context, error, cleanup)
    }

    #[cfg(feature = "cuda")]
    fn restore_cuda_decode_buffers(&mut self, buffers: DeepSeekV4DecodeBuffers) {
        self.operators
            .cuda
            .as_mut()
            .expect("CUDA cache exists for CUDA decode")
            .restore_decode_buffers(buffers);
    }

    #[cfg(feature = "cuda")]
    fn release_packed_cuda_resources(
        &mut self,
        continuation: &mut DeepSeekV4PackedCudaContinuation,
        deactivate_before_checkin: bool,
        disable_provisional_checkpoints: bool,
    ) -> Result<()> {
        if matches!(
            continuation.output_head,
            DeepSeekV4PackedOutputHeadState::Downloading(_)
        ) {
            return Err(Error::Execution { message:
                "DeepSeek-V4 packed resources cannot be released while output-head D2H is active"
                    .into(),
             });
        }
        if continuation.cancel_quiescence.is_some() {
            return Err(Error::Execution {
                message:
                    "DeepSeek-V4 packed resources cannot be released before cancellation quiesces"
                        .into(),
            });
        }
        let mut errors = Vec::new();
        if deactivate_before_checkin && continuation.paged_bindings_active {
            match self
                .operators
                .cuda_mut()
                .and_then(|cuda| cuda.activate_paged_binding(continuation.transaction, None))
            {
                Ok(()) => continuation.paged_bindings_active = false,
                Err(error) => errors.push(Error::context("paged binding deactivation", error)),
            }
        }

        if continuation.provisional_checkpoints {
            match self.operators.cuda.as_mut() {
                Some(cuda) => {
                    if let Err(error) =
                        cuda.deactivate_provisional_prefix_checkpoints(continuation.transaction)
                    {
                        errors.push(Error::context("provisional checkpoint deactivation", error));
                    }
                    if disable_provisional_checkpoints {
                        cuda.discard_provisional_prefix_checkpoints(continuation.transaction);
                        continuation.provisional_checkpoints = false;
                    }
                }
                None => errors.push(Error::Internal {
                    message: "provisional checkpoint cache is unavailable".into(),
                }),
            }
        }

        if let Some(arena_checkout) = continuation.arena_checkout.take() {
            if let Err(error) = self.layer_arena_pool.checkin(arena_checkout) {
                errors.push(Error::context("arena checkin", error));
            }
        }

        if !deactivate_before_checkin && continuation.paged_bindings_active {
            match self
                .operators
                .cuda_mut()
                .and_then(|cuda| cuda.activate_paged_binding(continuation.transaction, None))
            {
                Ok(()) => continuation.paged_bindings_active = false,
                Err(error) => errors.push(Error::context("paged binding deactivation", error)),
            }
        }

        if self.operators.cuda.is_some() {
            if let Some(decode_buffers) = continuation.decode_buffers.take() {
                self.restore_cuda_decode_buffers(decode_buffers);
            }
            #[cfg(feature = "cuda")]
            if let Some(proposal) = continuation.proposal_main_buffers.take() {
                self.operators
                    .cuda
                    .as_mut()
                    .expect("checked CUDA cache availability")
                    .restore_proposal_main_buffers(continuation.batch.len(), proposal);
            }
        } else {
            if continuation.decode_buffers.is_some() {
                errors.push(Error::Internal {
                    message: "decode buffer cache is unavailable".into(),
                });
            }
            #[cfg(feature = "cuda")]
            if continuation.proposal_main_buffers.is_some() {
                errors.push(Error::Internal {
                    message: "Proposal main buffer cache is unavailable".into(),
                });
            }
        }

        Self::packed_cuda_cleanup_result("DeepSeek-V4 packed CUDA resource", errors)
    }

    #[cfg(feature = "cuda")]
    fn discard_continuation_dependencies(
        &mut self,
        state: &mut Option<ContinuationDependencyState>,
    ) {
        if let Some(state) = state.as_mut() {
            state.discard_unresolved();
        }
    }

    #[cfg(feature = "cuda")]
    fn poll_packed_cuda_cancel_ready(
        &mut self,
        continuation: &mut DeepSeekV4PackedCudaContinuation,
    ) -> Result<bool> {
        if let Some(layer_continuation) = continuation.current_layer.as_mut() {
            let layer_index = continuation.next_layer_index;
            let layer = self
                .plan
                .resources()
                .layers()
                .get(layer_index)
                .ok_or_else(|| Error::Internal {
                    message: format!(
                        "current packed layer {layer_index} is outside the prepared layer set"
                    ),
                })?;
            if !layer.poll_packed_rows_cancel_ready(layer_continuation, &mut self.operators)? {
                return Ok(false);
            }
        }

        if let DeepSeekV4PackedOutputHeadState::Downloading(download) =
            &mut continuation.output_head
        {
            if !self
                .operators
                .cuda_mut()?
                .poll_output_head_topk_cancel_ready(download)?
            {
                return Ok(false);
            }
            continuation.output_head = DeepSeekV4PackedOutputHeadState::Ready(Vec::new());
        }

        if continuation.cancel_quiescence.is_none() {
            continuation.cancel_quiescence =
                Some(self.operators.cuda_mut()?.begin_compute_quiescence()?);
            return Ok(false);
        }
        let ready = self.operators.cuda_mut()?.poll_compute_quiescence(
            continuation
                .cancel_quiescence
                .as_mut()
                .expect("cancellation quiescence initialized above"),
        )?;
        if ready {
            continuation.cancel_quiescence = None;
        }
        Ok(ready)
    }

    #[cfg(feature = "cuda")]
    fn finish_packed_cuda_cancellation(
        &mut self,
        states: &mut [DeepSeekV4SequenceExecutionState],
        mut continuation: DeepSeekV4PackedCudaContinuation,
    ) -> (DeepSeekV4PackedCudaContinuation, Result<()>) {
        let mut errors = Vec::new();
        if let Some(layer_continuation) = continuation.current_layer.take() {
            let layer_index = continuation.next_layer_index;
            match self.plan.resources().layers().get(layer_index) {
                Some(layer) => {
                    if let Err(error) = layer.cancel_packed_rows_device_hc_device(
                        layer_continuation,
                        &mut self.operators,
                    ) {
                        errors.push(Error::context(
                            format!("layer {layer_index} continuation"),
                            error,
                        ));
                    }
                }
                None => errors.push(Error::Internal {
                    message: format!(
                        "current packed layer {layer_index} is outside the prepared layer set"
                    ),
                }),
            }
        }

        if let Err(error) = self.release_packed_cuda_resources(&mut continuation, true, true) {
            errors.push(error);
        }
        continuation.moe_access_events.clear();
        continuation.expert_leases.clear();
        self.discard_pending_moe_access_events();
        if continuation
            .metadata
            .sequences
            .iter()
            .all(|sequence| sequence.state_index < states.len())
        {
            poison_packed_sequence_steps(
                states,
                &continuation.metadata,
                &continuation.sequence_step_bindings,
            );
        } else {
            for (sequence, binding) in continuation
                .metadata
                .sequences
                .iter()
                .zip(continuation.sequence_step_bindings.iter().copied())
            {
                match states.get_mut(sequence.state_index) {
                    Some(state) => state.poison_step(binding),
                    None => errors.push(Error::Internal {
                        message: format!(
                            "saved state slot {} is missing during cancellation",
                            sequence.state_index
                        ),
                    }),
                }
            }
        }
        let result =
            Self::packed_cuda_cleanup_result("DeepSeek-V4 packed CUDA cancellation", errors);
        (continuation, result)
    }

    #[cfg(feature = "cuda")]
    fn abort_packed_cuda_continuation(
        &mut self,
        states: &mut [DeepSeekV4SequenceExecutionState],
        mut continuation: DeepSeekV4PackedCudaContinuation,
    ) -> (DeepSeekV4PackedCudaContinuation, Result<bool>) {
        self.discard_continuation_dependencies(&mut continuation.dependencies);
        match self.poll_packed_cuda_cancel_ready(&mut continuation) {
            Ok(false) => (continuation, Ok(false)),
            Err(error) => {
                self.completion_hub.notify();
                (continuation, Err(error))
            }
            Ok(true) => {
                let (continuation, cleanup) =
                    self.finish_packed_cuda_cancellation(states, continuation);
                (continuation, cleanup.map(|()| true))
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn begin_cuda_packed_batch_continuation(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [DeepSeekV4SequenceExecutionState],
        batch: &ExecutionBatch,
        metadata: PackedBatchMetadata,
    ) -> Result<DeepSeekV4PackedCudaContinuation> {
        let rows = batch.len();
        if !metadata.supports_native_cuda() {
            return Err(Error::Model {
                message: "DeepSeek-V4 CUDA packed shell does not support this batch shape yet"
                    .into(),
            });
        }
        for (sequence_index, sequence) in metadata.sequences.iter().enumerate() {
            let state = &states[sequence.state_index];
            for (offset, row) in sequence.query.clone().enumerate() {
                let expected =
                    state
                        .position()
                        .checked_add(offset)
                        .ok_or_else(|| Error::Model {
                            message: "DeepSeek-V4 packed sequence position overflow".into(),
                        })?;
                if expected != batch.positions()[row] as usize {
                    return Err(Error::Model {
                        message: format!(
                            "DeepSeek-V4 packed sequence {sequence_index} row {row} position mismatch: expected={expected} batch={}",
                            batch.positions()[row]
                        ),
                    });
                }
            }
            if state.paged_kv_binding.is_none() {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 packed sequence {sequence_index} has no prepared paged binding"
                    ),
                });
            }
        }
        if !self
            .cuda_materialization_owner
            .as_ref()
            .is_some_and(CudaExpertMaterializationOwner::residency_control_installed)
        {
            return Err(Error::Execution {
                message: "runtime expert residency controller is not installed".into(),
            });
        }
        let id = take_packed_cuda_continuation_id(&mut self.next_packed_cuda_continuation_id)?;

        let max_top_k = batch
            .logits()
            .iter()
            .filter_map(|request| match request {
                LogitsRequest::TopK(k) => Some(k.get() as usize),
                LogitsRequest::None | LogitsRequest::Full => None,
            })
            .max()
            .unwrap_or(0);
        let sequence_step_bindings = begin_packed_sequence_steps(states, &metadata)?;
        let paged_bindings = metadata
            .sequences
            .iter()
            .map(|sequence| {
                states[sequence.state_index]
                    .paged_kv_binding
                    .clone()
                    .expect("validated packed paged binding")
            })
            .collect::<Vec<_>>();
        let sequence_phases = metadata
            .sequences
            .iter()
            .map(|sequence| sequence.phase)
            .collect::<Vec<_>>();
        let positions = batch
            .positions()
            .iter()
            .map(|position| *position as usize)
            .collect::<Vec<_>>();
        let max_layers = self.plan.resources().prepare_options().max_layers;
        if batch.intent() == ExecutionIntent::ProvisionalVerification {
            let sequence_shapes = metadata
                .sequences
                .iter()
                .map(|sequence| {
                    (
                        states[sequence.state_index].position(),
                        sequence.query.len(),
                    )
                })
                .collect::<Vec<_>>();
            if let Err(error) = self
                .operators
                .cuda_mut()?
                .begin_provisional_prefix_checkpoints(transaction, &sequence_shapes, max_layers)
            {
                self.operators
                    .cuda_mut()?
                    .discard_provisional_prefix_checkpoints(transaction);
                return Err(error);
            }
        } else {
            self.operators
                .cuda_mut()?
                .discard_provisional_prefix_checkpoints(transaction);
        }
        let hidden_size = self.plan.resources().model().config.hidden_size;
        let hc_row_size = self
            .plan
            .resources()
            .model()
            .config
            .hc_config()
            .hc_hidden_size();
        let Some(hc_len) = rows.checked_mul(hc_row_size) else {
            self.operators
                .cuda_mut()?
                .discard_provisional_prefix_checkpoints(transaction);
            return Err(Error::Model {
                message: "DeepSeek-V4 packed HC size overflow".into(),
            });
        };
        let Some(hidden_len) = rows.checked_mul(hidden_size) else {
            self.operators
                .cuda_mut()?
                .discard_provisional_prefix_checkpoints(transaction);
            return Err(Error::Model {
                message: "DeepSeek-V4 packed hidden size overflow".into(),
            });
        };

        if let Err(error) = self.operators.check_cuda_arena_acquire() {
            self.operators
                .cuda_mut()?
                .discard_provisional_prefix_checkpoints(transaction);
            return Err(error);
        }
        let shape = match ExecutionShapeKey::from_batch(batch) {
            Ok(shape) => shape,
            Err(error) => {
                self.operators
                    .cuda_mut()?
                    .discard_provisional_prefix_checkpoints(transaction);
                return Err(error);
            }
        };
        let arena_key =
            DeepSeekV4LayerArenaPoolKey::new(shape, DeepSeekV4LayerArenaRowLayout::IndependentRows);
        let arena_checkout = {
            let layers = self.plan.resources().layers();
            let operators = &mut self.operators;
            self.layer_arena_pool.checkout(arena_key, || {
                DeepSeekV4LayerArenaVariants::try_build_for_packed_mode(layers, rows, operators)
            })
        };
        let arena_checkout = match arena_checkout {
            Ok(arena) => arena,
            Err(error) => {
                self.operators
                    .cuda_mut()?
                    .discard_provisional_prefix_checkpoints(transaction);
                return Err(error);
            }
        };

        let mut continuation = DeepSeekV4PackedCudaContinuation {
            transaction,
            id,
            batch: batch.clone(),
            metadata,
            sequence_step_bindings,
            paged_bindings,
            sequence_phases,
            positions,
            max_top_k,
            arena_checkout: Some(arena_checkout),
            decode_buffers: None,
            #[cfg(feature = "cuda")]
            proposal_main_buffers: None,
            initialized: false,
            next_layer_index: 0,
            current_layer: None,
            output_head: DeepSeekV4PackedOutputHeadState::NotStarted,
            cancel_quiescence: None,
            dependencies: None,
            moe_access_events: Vec::with_capacity(
                max_layers.saturating_mul(batch.sequences().len()),
            ),
            expert_leases: Vec::new(),
            paged_bindings_active: false,
            provisional_checkpoints: batch.intent() == ExecutionIntent::ProvisionalVerification,
            failed: false,
        };

        continuation.decode_buffers = match self
            .operators
            .cuda_mut()?
            .take_decode_buffers(hc_len, hidden_len, hidden_len)
        {
            Ok(buffers) => Some(buffers),
            Err(error) => {
                let cleanup = self.release_packed_cuda_resources(&mut continuation, false, true);
                return Err(Self::packed_cuda_error_with_cleanup(
                    "DeepSeek-V4 packed decode buffer checkout",
                    error,
                    cleanup,
                ));
            }
        };

        #[cfg(feature = "cuda")]
        {
            let capture = self
                .plan
                .resources()
                .proposal_attachment()
                .is_some_and(|attachment| {
                    !attachment.config.target_layer_ids.is_empty()
                        && attachment
                            .config
                            .target_layer_ids
                            .iter()
                            .all(|layer| *layer < max_layers)
                });
            if capture {
                continuation.proposal_main_buffers =
                    match self.operators.cuda_mut()?.take_proposal_main_buffers(rows) {
                        Ok(buffers) => Some(buffers),
                        Err(error) => {
                            let cleanup =
                                self.release_packed_cuda_resources(&mut continuation, false, true);
                            return Err(Self::packed_cuda_error_with_cleanup(
                                "DeepSeek-V4 packed Proposal buffer checkout",
                                error,
                                cleanup,
                            ));
                        }
                    };
            }
        }

        Ok(continuation)
    }

    #[cfg(feature = "cuda")]
    fn validate_packed_cuda_resume_states(
        states: &[DeepSeekV4SequenceExecutionState],
        continuation: &DeepSeekV4PackedCudaContinuation,
    ) -> Result<()> {
        if continuation
            .metadata
            .sequences
            .iter()
            .any(|sequence| sequence.state_index >= states.len())
        {
            return Err(Error::Execution {
                message: "DeepSeek-V4 packed continuation state slots no longer exist".into(),
            });
        }
        let current = begin_packed_sequence_steps(states, &continuation.metadata)?;
        if current != continuation.sequence_step_bindings {
            return Err(Error::Execution {
                message:
                    "DeepSeek-V4 packed continuation sequence step bindings changed while suspended"
                        .into(),
            });
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn pending_model_progress_for_packed_cuda(
        &mut self,
        continuation: &mut DeepSeekV4PackedCudaContinuation,
    ) -> Result<PendingModelProgress> {
        let resolved_stage = if matches!(
            continuation.output_head,
            DeepSeekV4PackedOutputHeadState::Downloading(_)
        ) {
            if continuation.current_layer.is_some() {
                return Err(Error::Internal { message:
                    "DeepSeek-V4 packed continuation cannot wait on a layer and output head simultaneously"
                        .into(),
                 });
            }
            None
        } else {
            let layer_continuation =
                continuation
                    .current_layer
                    .as_ref()
                    .ok_or_else(|| {
                        Error::Internal { message:
                    "DeepSeek-V4 waiting packed continuation has no current layer continuation"
                        .into(),
                 }
                    })?;
            let pending = layer_continuation.pending_experts();
            if pending.is_empty() {
                None
            } else {
                let requests = pending
                    .into_iter()
                    .map(|operation| {
                        let layer = u32::try_from(operation.layer).map_err(|_| {
                            Error::Execution { message: format!(
                                "DeepSeek-V4 pending expert layer {} exceeds u32 ABI",
                                operation.layer
                            ) }
                        })?;
                        let expert = u32::try_from(operation.expert).map_err(|_| {
                            Error::Execution { message: format!(
                                "DeepSeek-V4 pending expert {} exceeds u32 ABI",
                                operation.expert
                            ) }
                        })?;
                        let prepared = self
                            .plan
                            .resources()
                            .layer_experts()
                            .get(operation.layer)
                            .ok_or_else(|| {
                                Error::Internal { message: format!(
                                    "DeepSeek-V4 pending expert layer {} has no prepared source catalog",
                                    operation.layer
                                ) }
                            })?;
                        let resource_source = prepared.source_catalog().require_resource_source(
                            crate::moe::streaming::ExpertId::new(
                                operation.layer,
                                operation.expert,
                            ),
                        )?;
                        Ok((resource_source, layer, expert))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let resolver = self.materialization_resolver.as_deref_mut().ok_or_else(|| {
                    Error::Execution { message:
                        "DeepSeek-V4 exact resource wait requires an installed materialization resolver"
                            .into(),
                     }
                })?;
                let placement = resolver.placement();
                let resources = requests
                    .into_iter()
                    .map(|(resource_source, layer, expert)| {
                        let resource = ferrule_common::MaterializedResourceId::routed_expert(
                            ferrule_common::LayerId::new(layer),
                            ferrule_common::ExpertId::new(expert),
                        );
                        StageMaterializationRequest::new(
                            StageResourceUse::new(
                                resource,
                                ResourceAccess::Read,
                                ResourceRetention::ThroughTransaction,
                            ),
                            MaterializationRequest::for_placement(
                                placement,
                                resource_source,
                                resource,
                            )?,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                Some(resolve_stage_resources(
                    resources,
                    WorkspaceClaim::NONE,
                    resolver,
                )?)
            }
        };
        let dependencies = match &resolved_stage {
            Some(stage) => stage
                .dependencies()
                .expect("selected experts produce residency dependencies")
                .clone(),
            None => internal_continuation_dependency(continuation.id)?,
        };
        record_continuation_dependencies(
            &mut continuation.dependencies,
            continuation.id,
            dependencies.clone(),
        )?;
        match resolved_stage {
            Some(stage) => PendingModelProgress::for_resolved_stage(
                continuation.transaction,
                continuation.id,
                stage,
            ),
            None => {
                PendingModelProgress::new(continuation.transaction, continuation.id, dependencies)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn progress_cuda_packed_batch(
        &mut self,
        states: &mut [DeepSeekV4SequenceExecutionState],
        mut continuation: DeepSeekV4PackedCudaContinuation,
        resume_leases: Option<ResidencyLeaseSet>,
    ) -> std::result::Result<DeepSeekV4PackedCudaProgress, (Error, DeepSeekV4PackedCudaContinuation)>
    {
        if continuation.provisional_checkpoints {
            if let Err(error) = self.operators.cuda_mut().and_then(|cuda| {
                cuda.activate_provisional_prefix_checkpoints(continuation.transaction)
            }) {
                continuation.failed = true;
                return Err((error, continuation));
            }
        }
        if !continuation.paged_bindings_active {
            let binding_refs = continuation.paged_bindings.iter().collect::<Vec<_>>();
            if let Err(error) = self.operators.cuda_mut().and_then(|cuda| {
                cuda.activate_paged_bindings_for_rows(
                    continuation.transaction,
                    &binding_refs,
                    &continuation.metadata.row_to_sequence,
                )
            }) {
                continuation.failed = true;
                return Err((error, continuation));
            }
            continuation.paged_bindings_active = true;
        }

        let progress =
            self.progress_cuda_packed_batch_inner(states, &mut continuation, resume_leases);
        match progress {
            Ok(DeepSeekV4PackedCudaStep::Waiting(pending)) => {
                if let Err(error) = self
                    .operators
                    .cuda_mut()
                    .and_then(|cuda| cuda.activate_paged_binding(continuation.transaction, None))
                {
                    continuation.failed = true;
                    return Err((error, continuation));
                }
                continuation.paged_bindings_active = false;
                if continuation.provisional_checkpoints
                    && let Err(error) = self.operators.cuda_mut().and_then(|cuda| {
                        cuda.deactivate_provisional_prefix_checkpoints(continuation.transaction)
                    })
                {
                    continuation.failed = true;
                    return Err((error, continuation));
                }
                Ok(DeepSeekV4PackedCudaProgress::Waiting(continuation, pending))
            }
            Ok(DeepSeekV4PackedCudaStep::Complete(output)) => {
                let expert_leases = std::mem::take(&mut continuation.expert_leases);
                Ok(DeepSeekV4PackedCudaProgress::Complete(
                    output,
                    expert_leases,
                ))
            }
            Err(error) => {
                let deactivate = if continuation.paged_bindings_active {
                    self.operators.cuda_mut().and_then(|cuda| {
                        cuda.activate_paged_binding(continuation.transaction, None)
                    })
                } else {
                    Ok(())
                };
                if deactivate.is_ok() {
                    continuation.paged_bindings_active = false;
                }
                continuation.failed = true;
                let error = match deactivate {
                    Ok(()) => error,
                    Err(deactivate_error) => Error::Execution {
                        message: format!(
                            "DeepSeek-V4 packed progress failed ({error}); paged binding deactivation also failed ({deactivate_error})"
                        ),
                    },
                };
                Err((error, continuation))
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn progress_cuda_packed_batch_inner(
        &mut self,
        states: &mut [DeepSeekV4SequenceExecutionState],
        continuation: &mut DeepSeekV4PackedCudaContinuation,
        resume_leases: Option<ResidencyLeaseSet>,
    ) -> Result<DeepSeekV4PackedCudaStep> {
        if continuation.failed {
            return Err(Error::Execution {
                message: format!(
                    "DeepSeek-V4 packed continuation {} previously failed and must be cancelled",
                    continuation.id.get()
                ),
            });
        }
        if !continuation.initialized {
            let decode_buffers =
                continuation
                    .decode_buffers
                    .as_mut()
                    .ok_or_else(|| Error::Internal {
                        message: "DeepSeek-V4 packed decode buffers are unavailable".into(),
                    })?;
            self.operators.cuda_mut()?.resident_embedding_hc_rows_into(
                continuation.batch.token_ids(),
                &mut decode_buffers.hc_input,
            )?;
            #[cfg(feature = "cuda")]
            if let Some(proposal) = continuation.proposal_main_buffers.as_mut() {
                let positions_i32 = continuation
                    .positions
                    .iter()
                    .map(|position| {
                        i32::try_from(*position).map_err(|_| Error::Model {
                            message: "DeepSeek-V4 packed Proposal position exceeds i32".into(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.operators
                    .cuda_mut()?
                    .ops
                    .overwrite_i32_buffer(&positions_i32, &mut proposal.positions)?;
            }
            continuation.initialized = true;
        }
        let max_layers = self.plan.resources().prepare_options().max_layers;
        let mut resume_leases = resume_leases;
        while continuation.next_layer_index < max_layers {
            let layer_idx = continuation.next_layer_index;
            let layer_progress = if let Some(layer_continuation) = continuation.current_layer.take()
            {
                let leases = resume_leases.take().ok_or_else(|| Error::Execution {
                    message: format!(
                        "DeepSeek-V4 packed continuation {} has no lease set for its current layer",
                        continuation.id.get()
                    ),
                })?;
                let arena = continuation
                    .arena_checkout
                    .as_mut()
                    .and_then(|checkout| checkout.get_mut().get_for_layer_mut(layer_idx))
                    .ok_or_else(|| Error::Internal {
                        message: format!(
                            "DeepSeek-V4 packed layer arena {layer_idx} is unavailable"
                        ),
                    })?;
                let hc_state = &mut continuation
                    .decode_buffers
                    .as_mut()
                    .ok_or_else(|| Error::Internal {
                        message: "DeepSeek-V4 packed decode buffers are unavailable".into(),
                    })?
                    .hc_input;
                self.plan.resources().layers()[layer_idx].resume_packed_rows_device_hc_device(
                    layer_continuation,
                    leases,
                    arena,
                    hc_state,
                    &mut self.operators,
                )?
            } else {
                let requested_states = continuation
                    .metadata
                    .sequences
                    .iter()
                    .map(|sequence| sequence.state_index)
                    .collect::<BTreeSet<_>>();
                let mut available_states = states
                    .iter_mut()
                    .enumerate()
                    .filter(|(state_index, _)| requested_states.contains(state_index))
                    .collect::<BTreeMap<_, _>>();
                let mut layer_states = continuation
                    .metadata
                    .sequences
                    .iter()
                    .map(|sequence| {
                        available_states
                            .remove(&sequence.state_index)
                            .map(|state| &mut state.layers[layer_idx])
                            .ok_or_else(|| Error::Model {
                                message: format!(
                                    "DeepSeek-V4 state slot {} is referenced more than once",
                                    sequence.state_index
                                ),
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let arena = continuation
                    .arena_checkout
                    .as_mut()
                    .and_then(|checkout| checkout.get_mut().get_for_layer_mut(layer_idx))
                    .ok_or_else(|| Error::Internal {
                        message: format!(
                            "DeepSeek-V4 packed layer arena {layer_idx} is unavailable"
                        ),
                    })?;
                let hc_state = &mut continuation
                    .decode_buffers
                    .as_mut()
                    .ok_or_else(|| Error::Internal {
                        message: "DeepSeek-V4 packed decode buffers are unavailable".into(),
                    })?
                    .hc_input;
                match self.plan.resources().layers()[layer_idx].begin_packed_rows_device_hc_device(
                    &mut layer_states,
                    &continuation.metadata.row_to_sequence,
                    &continuation.metadata.sequence_major_rows,
                    &continuation.sequence_phases,
                    &continuation.paged_bindings,
                    arena,
                    hc_state,
                    continuation.batch.token_ids(),
                    &continuation.positions,
                    &mut self.operators,
                )? {
                    DeepSeekV4PackedLayerProgress::Waiting(next)
                        if !next.pending_experts().is_empty() =>
                    {
                        continuation.current_layer = Some(next);
                        let pending = self.pending_model_progress_for_packed_cuda(continuation)?;
                        return Ok(DeepSeekV4PackedCudaStep::Waiting(pending));
                    }
                    DeepSeekV4PackedLayerProgress::Waiting(next) => self.plan.resources().layers()
                        [layer_idx]
                        .resume_packed_rows_device_hc_device(
                            next,
                            empty_expert_lease_set(continuation.id)?,
                            arena,
                            hc_state,
                            &mut self.operators,
                        )?,
                    complete @ DeepSeekV4PackedLayerProgress::Complete { .. } => complete,
                }
            };

            match layer_progress {
                DeepSeekV4PackedLayerProgress::Waiting(next) => {
                    continuation.current_layer = Some(next);
                    let pending = self.pending_model_progress_for_packed_cuda(continuation)?;
                    return Ok(DeepSeekV4PackedCudaStep::Waiting(pending));
                }
                DeepSeekV4PackedLayerProgress::Complete { events, leases } => {
                    let hc_state = &continuation
                        .decode_buffers
                        .as_ref()
                        .ok_or_else(|| Error::Internal {
                            message: "DeepSeek-V4 packed decode buffers are unavailable".into(),
                        })?
                        .hc_input;
                    #[cfg(feature = "cuda")]
                    if std::env::var("FERRULE_DEBUG_HC_LAYER")
                        .ok()
                        .and_then(|value| value.parse::<usize>().ok())
                        == Some(layer_idx)
                    {
                        let values = self
                            .operators
                            .cuda_mut()?
                            .ops
                            .download_f32_buffer(hc_state)?;
                        let sum = values.iter().copied().sum::<f32>();
                        let sumsq = values.iter().map(|value| value * value).sum::<f32>();
                        let absmax = values
                            .iter()
                            .map(|value| value.abs())
                            .fold(0.0f32, f32::max);
                        eprintln!(
                            "hc layer={} sum={} sumsq={} absmax={} samples={:?}",
                            layer_idx,
                            sum,
                            sumsq,
                            absmax,
                            &values[..values.len().min(4)]
                        );
                    }
                    #[cfg(feature = "cuda")]
                    if let Some(proposal) = continuation.proposal_main_buffers.as_mut() {
                        self.operators
                            .cuda_mut()?
                            .capture_proposal_target_tap_from_device(
                                layer_idx,
                                hc_state,
                                continuation.batch.len(),
                                &mut proposal.target_taps,
                            )?;
                    }
                    continuation.moe_access_events.extend(events);
                    continuation.expert_leases.extend(leases);
                    continuation.next_layer_index += 1;
                }
            }
        }
        if let Some(unused) = resume_leases {
            if !unused.is_empty() {
                return Err(Error::Execution {
                    message: format!(
                        "DeepSeek-V4 packed continuation {} received expert leases without a current layer",
                        continuation.id.get()
                    ),
                });
            }
        }

        if !self.progress_cuda_packed_output_head(states, continuation)? {
            let pending = self.pending_model_progress_for_packed_cuda(continuation)?;
            return Ok(DeepSeekV4PackedCudaStep::Waiting(pending));
        }
        let output = self.complete_cuda_packed_batch(states, continuation)?;
        Ok(DeepSeekV4PackedCudaStep::Complete(output))
    }

    #[cfg(feature = "cuda")]
    fn progress_cuda_packed_output_head(
        &mut self,
        states: &[DeepSeekV4SequenceExecutionState],
        continuation: &mut DeepSeekV4PackedCudaContinuation,
    ) -> Result<bool> {
        #[cfg(not(feature = "cuda"))]
        let _ = states;
        if matches!(
            continuation.output_head,
            DeepSeekV4PackedOutputHeadState::NotStarted
        ) {
            let rows = continuation.batch.len();
            #[cfg(feature = "cuda")]
            if let Some(proposal) = continuation.proposal_main_buffers.as_mut() {
                self.operators
                    .cuda_mut()?
                    .proposal_main_project_norm_device_into(rows, proposal)?;
                let attachment =
                    self.plan
                        .resources()
                        .proposal_attachment()
                        .ok_or_else(|| Error::Model {
                            message: "DeepSeek-V4 packed Proposal context has no attachment".into(),
                        })?;
                let stage_count = attachment.layers.len();
                for sequence in &continuation.metadata.sequences {
                    let state = &states[sequence.state_index];
                    if state.proposal_stages.len() != stage_count {
                        return Err(Error::Model {
                            message: format!(
                                "DeepSeek-V4 packed Proposal context mismatch for state {}: states={} stages={stage_count}",
                                sequence.state_index,
                                state.proposal_stages.len()
                            ),
                        });
                    }
                }
                let max_position =
                    continuation
                        .positions
                        .iter()
                        .copied()
                        .max()
                        .ok_or_else(|| Error::Model {
                            message: "DeepSeek-V4 packed Proposal positions are empty".into(),
                        })?;
                for stage in 0..stage_count {
                    let config = attachment.layers[stage].transformer.attention.config;
                    self.operators
                        .cuda_mut()?
                        .proposal_context_kv_stage_packed_device_into(
                            stage,
                            config,
                            rows,
                            max_position,
                            proposal,
                        )?;
                }
            }

            if continuation.max_top_k == 0 {
                continuation.output_head =
                    DeepSeekV4PackedOutputHeadState::Ready(vec![Vec::new(); rows]);
                return Ok(true);
            }
            let decode_buffers =
                continuation
                    .decode_buffers
                    .as_mut()
                    .ok_or_else(|| Error::Internal {
                        message: "DeepSeek-V4 packed decode buffers are unavailable".into(),
                    })?;
            self.operators.cuda_mut()?.hc_head_from_device_into(
                &decode_buffers.hc_input,
                rows,
                &mut decode_buffers.final_hidden,
            )?;
            self.operators
                .cuda_mut()?
                .rms_norm_output_rows_device_into(
                    &decode_buffers.final_hidden,
                    rows,
                    self.plan.resources().model().config.norm_eps,
                    &mut decode_buffers.topk_row,
                )?;
            let download = self
                .operators
                .cuda_mut()?
                .begin_output_head_topk_rows_from_execution_image(
                    &decode_buffers.topk_row,
                    rows,
                    continuation.max_top_k,
                )?;
            continuation.output_head = DeepSeekV4PackedOutputHeadState::Downloading(download);
            return Ok(false);
        }

        if let DeepSeekV4PackedOutputHeadState::Downloading(download) =
            &mut continuation.output_head
        {
            let Some(logits) = self
                .operators
                .cuda_mut()?
                .poll_output_head_topk_rows_from_execution_image(download)?
            else {
                return Ok(false);
            };
            continuation.output_head = DeepSeekV4PackedOutputHeadState::Ready(logits);
        }
        Ok(matches!(
            continuation.output_head,
            DeepSeekV4PackedOutputHeadState::Ready(_)
        ))
    }

    #[cfg(feature = "cuda")]
    fn complete_cuda_packed_batch(
        &mut self,
        states: &mut [DeepSeekV4SequenceExecutionState],
        continuation: &mut DeepSeekV4PackedCudaContinuation,
    ) -> Result<ExecutionOutput> {
        let rows = continuation.batch.len();
        let all_row_logits = match &continuation.output_head {
            DeepSeekV4PackedOutputHeadState::Ready(logits) if logits.len() == rows => logits,
            DeepSeekV4PackedOutputHeadState::Ready(logits) => {
                return Err(Error::Internal {
                    message: format!(
                        "DeepSeek-V4 packed output-head row mismatch: expected {rows}, got {}",
                        logits.len()
                    ),
                });
            }
            DeepSeekV4PackedOutputHeadState::NotStarted => {
                return Err(Error::Internal { message:
                    "DeepSeek-V4 packed output publication started before output-head submission"
                        .into(),
                 });
            }
            DeepSeekV4PackedOutputHeadState::Downloading(_) => {
                return Err(Error::Internal { message:
                    "DeepSeek-V4 packed output publication started while output-head D2H is active"
                        .into(),
                 });
            }
        };
        let mut output_rows = Vec::new();
        for (row, request) in continuation.batch.logits().iter().copied().enumerate() {
            if let LogitsRequest::TopK(k) = request {
                output_rows.push(LogitsRow::new(
                    row as u32,
                    LogitsOutput::TopK(
                        all_row_logits[row]
                            .iter()
                            .take(k.get() as usize)
                            .map(|item| ExecutionTokenLogit {
                                token_id: item.token_id,
                                logit: item.logit,
                            })
                            .collect(),
                    ),
                ));
            }
        }
        let output = ExecutionOutput::new(output_rows);
        output
            .validate_with_capabilities(&continuation.batch, &self.multi_session_capabilities())?;
        if continuation
            .moe_access_events
            .iter()
            .any(|event| event.sequence_index >= continuation.metadata.sequences.len())
        {
            return Err(Error::Internal {
                message: "DeepSeek-V4 packed MoE event references an unknown sequence".into(),
            });
        }
        #[cfg(feature = "cuda")]
        let proposal_stage_count = if continuation.proposal_main_buffers.is_some() {
            Some(
                self.plan
                    .resources()
                    .proposal_attachment()
                    .ok_or_else(|| Error::Model {
                        message: "DeepSeek-V4 packed Proposal context has no attachment".into(),
                    })?
                    .layers
                    .len(),
            )
        } else {
            None
        };
        self.release_packed_cuda_resources(continuation, false, false)?;
        commit_packed_sequence_steps(
            states,
            &continuation.metadata,
            continuation.sequence_step_bindings.clone(),
        )?;
        #[cfg(feature = "cuda")]
        if let Some(stage_count) = proposal_stage_count {
            for sequence in &continuation.metadata.sequences {
                let state = &mut states[sequence.state_index];
                for stage in 0..stage_count {
                    state.proposal_stages[stage]
                        .kv
                        .window
                        .record_device_rows(sequence.query.len());
                }
            }
        }
        for attributed in continuation.moe_access_events.drain(..) {
            let state_index =
                continuation.metadata.sequences[attributed.sequence_index].state_index;
            states[state_index]
                .predictor
                .observe_batch(attributed.event);
        }
        match continuation.metadata.mode {
            ForwardMode::Prefill => {
                self.observability.output.packed_prefill_batches = self
                    .observability
                    .output
                    .packed_prefill_batches
                    .saturating_add(1);
                self.observability.output.packed_prefill_rows = self
                    .observability
                    .output
                    .packed_prefill_rows
                    .saturating_add(rows as u64);
            }
            ForwardMode::Decode => {
                self.observability.output.packed_decode_batches = self
                    .observability
                    .output
                    .packed_decode_batches
                    .saturating_add(1);
                self.observability.output.packed_decode_rows = self
                    .observability
                    .output
                    .packed_decode_rows
                    .saturating_add(rows as u64);
            }
            ForwardMode::Mixed => {
                self.observability.output.packed_mixed_batches = self
                    .observability
                    .output
                    .packed_mixed_batches
                    .saturating_add(1);
                self.observability.output.packed_mixed_rows = self
                    .observability
                    .output
                    .packed_mixed_rows
                    .saturating_add(rows as u64);
            }
        }
        Ok(output)
    }
}

impl DeepSeekV4Runner {
    #[cfg(feature = "cuda")]
    fn abort_native_proposals(
        &mut self,
        transaction: ExecutionTransactionId,
    ) -> Result<TransactionEndProgress> {
        loop {
            let Some(continuation_id) =
                self.proposal_continuations
                    .iter()
                    .find_map(|(id, continuation)| {
                        (continuation.transaction == transaction).then_some(*id)
                    })
            else {
                return Ok(TransactionEndProgress::Complete);
            };
            let mut continuation = self
                .proposal_continuations
                .remove(&continuation_id)
                .expect("native proposal continuation was selected above");
            self.discard_continuation_dependencies(&mut continuation.dependencies);

            if !self.poll_proposal_route_cancel_ready(&mut continuation)? {
                self.proposal_continuations
                    .insert(continuation_id, continuation);
                return Ok(TransactionEndProgress::Pending);
            }

            if let Some(download) = continuation.head_download.as_ref() {
                let result = continuation
                    .arena
                    .as_mut()
                    .ok_or_else(|| Error::Internal {
                        message: "DeepSeek-V4 Proposal arena is unavailable".into(),
                    })
                    .and_then(|arena| {
                        self.operators
                            .cuda_mut()?
                            .poll_proposal_head_result(&mut arena.head, download)
                    })?;
                if result.is_none() {
                    if !continuation.callback_armed {
                        self.arm_proposal_completion(&mut continuation);
                    }
                    self.proposal_continuations
                        .insert(continuation_id, continuation);
                    return Ok(TransactionEndProgress::Pending);
                }
                continuation.head_download = None;
            }

            if continuation.cancel_quiescence.is_none() {
                continuation.cancel_quiescence =
                    Some(self.operators.cuda_mut()?.begin_compute_quiescence()?);
                self.proposal_continuations
                    .insert(continuation_id, continuation);
                return Ok(TransactionEndProgress::Pending);
            }
            if !self.operators.cuda_mut()?.poll_compute_quiescence(
                continuation
                    .cancel_quiescence
                    .as_mut()
                    .expect("proposal cancellation quiescence was initialized above"),
            )? {
                self.proposal_continuations
                    .insert(continuation_id, continuation);
                return Ok(TransactionEndProgress::Pending);
            }
            continuation.cancel_quiescence = None;

            if let Some(layer_continuation) = continuation.current_layer.take() {
                let stage = continuation.stage;
                let layer = self
                    .plan
                    .resources()
                    .proposal_attachment()
                    .and_then(|attachment| attachment.layers.get(stage))
                    .ok_or_else(|| Error::Internal {
                        message: format!(
                            "current Proposal stage {stage} is outside the prepared attachment"
                        ),
                    })?;
                layer.transformer.cancel_proposal_block_device_hc_device(
                    layer_continuation,
                    &mut self.operators,
                )?;
            }
            self.deactivate_proposal_binding(&mut continuation)?;
            continuation.moe_access_events.clear();
            continuation.expert_leases.clear();
            self.release_proposal_arena(&mut continuation)?;
        }
    }
}

fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

impl ModelRunner for DeepSeekV4Runner {
    fn model_info(&self) -> ModelInfo {
        let mut info = self.plan.resources().model().model_info();
        info.backend = self.operator_backend().as_str();
        info
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.plan.resources().model().tokenizer.encode(text)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.plan.resources().model().tokenizer.decode(tokens)
    }

    fn reset_session(&mut self) -> Result<()> {
        self.reset()
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.plan.resources().model().tokenizer.eos_token_id()
    }

    fn bound_layer_count(&self) -> Option<usize> {
        Some(DeepSeekV4Runner::bound_layer_count(self))
    }

    fn expert_report(&self) -> Option<String> {
        let stats = self.layer_runtime_stats();
        if stats.is_empty() {
            return Some("DeepSeek-V4 layers are not bound yet.\n".into());
        }
        let mut report = String::new();
        for stat in stats {
            report.push_str(&format!(
                "L{:>2}: window_kv={} compressed_kv={} indexer_kv={} resident_experts={} resident_bytes={}\n",
                stat.layer,
                stat.window_kv_len,
                stat.compressed_kv_len,
                stat.indexer_compressed_kv_len,
                stat.resident_experts,
                stat.resident_expert_bytes
            ));
        }
        Some(report)
    }
}

impl ResidentModelRunner for DeepSeekV4Runner {
    type ObservabilitySnapshot = DeepSeekV4ObservabilitySnapshot;

    fn observability_snapshot(&self) -> Self::ObservabilitySnapshot {
        DeepSeekV4Runner::observability_snapshot(self)
    }

    fn completion_hub(&self) -> CompletionHub {
        self.completion_hub.clone()
    }

    fn take_completion_reactors(&mut self) -> Vec<ModelCompletionReactor> {
        std::mem::take(&mut self.expert_completion_reactors)
    }

    fn native_proposal_source(&self) -> Result<Option<NativeProposalSource>> {
        Ok(self.proposal_source)
    }

    fn begin_native_proposal(
        &mut self,
        transaction: ExecutionTransactionId,
        anchor_token_id: u32,
    ) -> Result<NativeProposalProgress> {
        let source = self.proposal_source.ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 has no checkpoint-native proposal capability".into(),
        })?;
        #[cfg(feature = "cuda")]
        {
            source.validate()?;
            let mut continuation =
                self.begin_proposal_continuation(transaction, anchor_token_id)?;
            if let Err(error) = self.activate_proposal_binding(&mut continuation) {
                return Err(self.cleanup_unpublished_proposal(&mut continuation, error));
            }
            let progress = self.progress_proposal(&mut continuation, None);
            match progress {
                Ok(DeepSeekV4ProposalStep::Waiting(pending)) => {
                    if let Err(error) = self.deactivate_proposal_binding(&mut continuation) {
                        continuation.failed = true;
                        self.completion_hub.notify();
                        let continuation_id = continuation.id;
                        self.proposal_continuations
                            .insert(continuation_id, continuation);
                        return Err(Error::context(
                            "paged binding deactivation after proposal begin",
                            error,
                        ));
                    }
                    let replaced = self
                        .proposal_continuations
                        .insert(continuation.id, continuation);
                    debug_assert!(replaced.is_none(), "native proposal continuation ID reused");
                    Ok(NativeProposalProgress::Waiting(pending))
                }
                Ok(DeepSeekV4ProposalStep::Complete(proposal)) => {
                    if let Err(error) = self.deactivate_proposal_binding(&mut continuation) {
                        return Err(self.cleanup_unpublished_proposal(&mut continuation, error));
                    }
                    self.release_proposal_arena(&mut continuation)?;
                    Ok(NativeProposalProgress::Complete(proposal))
                }
                Err(error) => Err(self.cleanup_unpublished_proposal(&mut continuation, error)),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (transaction, anchor_token_id, source);
            Err(Error::Model {
                message: "DeepSeek-V4 checkpoint-native proposal requires CUDA + CUTLASS".into(),
            })
        }
    }

    fn resume_native_proposal(
        &mut self,
        transaction: ExecutionTransactionId,
        continuation_id: BatchContinuationId,
        leases: ResidencyLeaseSet,
    ) -> Result<NativeProposalProgress> {
        #[cfg(feature = "cuda")]
        {
            let mut continuation = self
                .proposal_continuations
                .remove(&continuation_id)
                .ok_or_else(|| Error::Execution {
                    message: format!(
                        "DeepSeek-V4 has no native proposal continuation {}",
                        continuation_id.get()
                    ),
                })?;
            if continuation.transaction != transaction {
                let owner = continuation.transaction;
                self.proposal_continuations
                    .insert(continuation_id, continuation);
                return Err(Error::Execution {
                    message: format!(
                        "DeepSeek-V4 native proposal continuation {} belongs to transaction {}, not {}",
                        continuation_id.get(),
                        owner.get(),
                        transaction.get()
                    ),
                });
            }
            if continuation.failed {
                let error = Error::Execution {
                    message: format!(
                        "DeepSeek-V4 native proposal continuation {} previously failed and must be aborted",
                        continuation_id.get()
                    ),
                };
                self.proposal_continuations
                    .insert(continuation_id, continuation);
                return Err(error);
            }
            let resume_validation = continuation
                .dependencies
                .as_mut()
                .ok_or_else(|| {
                    Error::Execution { message: format!(
                        "DeepSeek-V4 native proposal continuation {} has no declared dependency set",
                        continuation_id.get()
                    ) }
                })
                .and_then(|dependencies| {
                    dependencies.validate_resume(continuation_id, &leases)
                });
            if let Err(error) = resume_validation {
                self.proposal_continuations
                    .insert(continuation_id, continuation);
                return Err(error);
            }
            if let Err(error) = self.activate_proposal_binding(&mut continuation) {
                self.proposal_continuations
                    .insert(continuation_id, continuation);
                return Err(error);
            }
            let progress = self.progress_proposal(&mut continuation, Some(leases));
            let deactivate = self.deactivate_proposal_binding(&mut continuation);
            match (progress, deactivate) {
                (Ok(DeepSeekV4ProposalStep::Waiting(pending)), Ok(())) => {
                    self.proposal_continuations
                        .insert(continuation_id, continuation);
                    Ok(NativeProposalProgress::Waiting(pending))
                }
                (Ok(DeepSeekV4ProposalStep::Complete(proposal)), Ok(())) => {
                    if let Err(error) = self.release_proposal_arena(&mut continuation) {
                        continuation.failed = true;
                        self.proposal_continuations
                            .insert(continuation_id, continuation);
                        return Err(error);
                    }
                    Ok(NativeProposalProgress::Complete(proposal))
                }
                (progress, deactivate) => {
                    let error = match (progress, deactivate) {
                        (Err(error), Ok(())) => error,
                        (Ok(_), Err(error)) => Error::context("paged binding deactivation", error),
                        (Err(error), Err(deactivate_error)) => Error::with_cleanup(
                            "native proposal progress",
                            error,
                            Err(deactivate_error),
                        ),
                        (Ok(_), Ok(())) => unreachable!("handled successful proposal progress"),
                    };
                    continuation.failed = true;
                    self.proposal_continuations
                        .insert(continuation_id, continuation);
                    Err(error)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (transaction, continuation_id, leases);
            Err(Error::Execution {
                message:
                    "DeepSeek-V4 was built without CUDA + CUTLASS native proposal continuations"
                        .into(),
            })
        }
    }
}

impl MultiSessionRunner for DeepSeekV4Runner {
    type SequenceState = DeepSeekV4SequenceExecutionState;

    fn sequence_generation(&self, state: &Self::SequenceState) -> u64 {
        state.generation()
    }

    fn expert_residency_requirements(&self) -> Option<ExpertResidencyRequirements> {
        #[cfg(feature = "cuda")]
        {
            return self.expert_residency_requirements.clone();
        }
        #[cfg(not(feature = "cuda"))]
        None
    }

    fn expert_residency_control_installed(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            return self
                .cuda_materialization_owner
                .as_ref()
                .is_some_and(CudaExpertMaterializationOwner::residency_control_installed);
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    fn install_expert_residency_control(
        &mut self,
        control: Box<dyn ExpertResidencyControl>,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if !self.packed_transactions.is_empty() {
                return Err(Error::Execution { message:
                    "cannot replace DeepSeek-V4 expert residency control while packed transactions own expert work"
                        .into(),
                 });
            }
            let expected =
                self.expert_residency_requirements()
                    .ok_or_else(|| Error::Execution {
                        message:
                            "DeepSeek-V4 CPU runner does not accept an expert residency controller"
                                .into(),
                    })?;
            if control.requirements() != expected {
                return Err(Error::Execution {
                    message: format!(
                        "DeepSeek-V4 expert residency requirements mismatch: expected {:?}, got {:?}",
                        expected,
                        control.requirements()
                    ),
                });
            }
            return install_residency_control_on_owner(
                self.cuda_materialization_owner.as_ref(),
                control,
                CudaExpertMaterializationOwner::install_residency_control,
            );
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = control;
            Err(Error::Execution {
                message: "DeepSeek-V4 was built without CUDA expert residency support".into(),
            })
        }
    }

    fn take_materialization_provider(&mut self) -> Option<Box<dyn MaterializationProvider>> {
        #[cfg(feature = "cuda")]
        {
            let execution_backend = self.operators.backend();
            let physical = take_cuda_backend_once(
                execution_backend,
                self.cuda_materialization_owner.as_mut(),
                CudaExpertMaterializationOwner::take_provider,
            );
            if physical.is_some() {
                self.materialization_provider_transferred = true;
            }
            return physical;
        }
        #[cfg(not(feature = "cuda"))]
        None
    }

    fn take_warmup_requests(&mut self) -> Result<Vec<MaterializationRequest>> {
        #[cfg(feature = "cuda")]
        {
            return Ok(std::mem::take(&mut self.warmup_requests));
        }
        #[cfg(not(feature = "cuda"))]
        Ok(Vec::new())
    }

    fn transaction_prefetch_requests(
        &self,
        _transaction: ExecutionTransactionId,
        batch: &ExecutionBatch,
    ) -> Result<Vec<MaterializationRequest>> {
        #[cfg(feature = "cuda")]
        {
            if self.operators.backend() != ModelExecutionBackend::Cuda {
                return Ok(Vec::new());
            }
            let Some(owner) = self.cuda_materialization_owner.as_ref() else {
                return Ok(Vec::new());
            };
            let resources = self.plan.resources();
            let expert_limit = resources.model().config.num_routed_experts;
            let mut requests = Vec::new();
            for (layer_index, layer) in resources.layers().iter().enumerate() {
                let prepared = resources.layer_experts().get(layer_index).ok_or_else(|| {
                    Error::Internal {
                        message: format!(
                            "DeepSeek-V4 execution layer {layer_index} has no prepared expert catalog"
                        ),
                    }
                })?;
                let Some(experts) = layer.router.hash_expert_union_for_tokens(
                    batch.token_ids(),
                    layer.router_policy.top_k,
                    expert_limit,
                )?
                else {
                    continue;
                };
                for expert in experts {
                    let layer_id = u32::try_from(layer.layer).map_err(|_| Error::Model {
                        message: format!(
                            "DeepSeek-V4 execution layer {} exceeds the materialization ABI",
                            layer.layer
                        ),
                    })?;
                    let expert_id = u32::try_from(expert).map_err(|_| Error::Model {
                        message: format!(
                            "DeepSeek-V4 routed expert {expert} exceeds the materialization ABI"
                        ),
                    })?;
                    let source = prepared.source_catalog().require_resource_source(
                        crate::moe::streaming::ExpertId::new(layer.layer, expert),
                    )?;
                    let resource = ferrule_common::MaterializedResourceId::routed_expert(
                        ferrule_common::LayerId::new(layer_id),
                        ferrule_common::ExpertId::new(expert_id),
                    );
                    requests.push(MaterializationRequest::for_placement(
                        owner.placement(),
                        source,
                        resource,
                    )?);
                }
            }
            return Ok(requests);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = batch;
            Ok(Vec::new())
        }
    }

    fn materialization_resolver_installed(&self) -> bool {
        self.materialization_resolver.is_some()
    }

    fn install_materialization_resolver(
        &mut self,
        resolver: Box<dyn MaterializationResolver>,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if !self.packed_transactions.is_empty() {
            return Err(Error::Execution { message:
                "cannot replace the DeepSeek-V4 materialization resolver while packed transactions are outstanding"
                    .into(),
             });
        }
        if self.materialization_resolver.is_some() {
            return Err(Error::Execution {
                message: "DeepSeek-V4 materialization resolver is already installed".into(),
            });
        }
        self.materialization_resolver = Some(resolver);
        Ok(())
    }

    fn materialization_resolver(&mut self) -> Result<&mut (dyn MaterializationResolver + '_)> {
        match self.materialization_resolver.as_mut() {
            Some(resolver) => Ok(resolver.as_mut()),
            None => Err(Error::Execution {
                message: "DeepSeek-V4 materialization resolver is not installed".into(),
            }),
        }
    }

    fn configure_kv_page_capacity(&mut self, max_pages: usize) -> Result<()> {
        #[cfg(feature = "cuda")]
        if !self.packed_transactions.is_empty() {
            return Err(Error::Execution { message:
                "cannot reconfigure DeepSeek-V4 KV capacity while packed transactions own reservations"
                    .into(),
             });
        }
        #[cfg(feature = "cuda")]
        if self.operators.backend() == ModelExecutionBackend::Cuda {
            let schema = self.plan.resources().kv_layout().clone();
            self.operators
                .cuda_mut()?
                .configure_kv_page_pool(&schema, max_pages)?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = max_pages;
        Ok(())
    }

    fn release_kv_pages(&mut self, pages: &[ferrule_common::execution::KvPageId]) -> Result<()> {
        self.ensure_kv_pages_available(pages, "release")?;
        #[cfg(feature = "cuda")]
        if self.operators.cuda.is_some() {
            self.operators.cuda_mut()?.release_kv_pages(pages)?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = pages;
        Ok(())
    }

    fn preempt_kv_pages(&mut self, pages: &[ferrule_common::execution::KvPageId]) -> Result<()> {
        self.ensure_kv_pages_available(pages, "preempt")?;
        #[cfg(feature = "cuda")]
        if self.operators.cuda.is_some() {
            self.operators.cuda_mut()?.preempt_kv_pages(pages)?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = pages;
        Ok(())
    }

    fn restore_kv_pages(&mut self, pages: &[ferrule_common::execution::KvPageId]) -> Result<()> {
        self.ensure_kv_pages_available(pages, "restore")?;
        #[cfg(feature = "cuda")]
        if self.operators.cuda.is_some() {
            self.operators.cuda_mut()?.restore_kv_pages(pages)?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = pages;
        Ok(())
    }

    fn prepare_multi_session_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        batch: &ferrule_common::execution::ExecutionBatch,
        kv_reservations: &[ferrule_common::execution::KvReservationView],
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if self.operators.backend() != ModelExecutionBackend::Cuda {
                return Err(Error::Execution {
                    message: "DeepSeek-V4 multi-session prepare requires the CUDA packed backend"
                        .into(),
                });
            }
            if !self
                .operators
                .cuda
                .as_ref()
                .is_some_and(|cuda| cuda.has_kv_page_pool())
            {
                return Err(Error::Execution {
                    message:
                        "DeepSeek-V4 multi-session prepare requires a configured CUDA KV page pool"
                            .into(),
                });
            }
            if kv_reservations.len() != batch.sequences().len() {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 paged batch has {} sequences but {} KV reservations",
                        batch.sequences().len(),
                        kv_reservations.len()
                    ),
                });
            }

            let metadata = PackedBatchMetadata::lower(batch, states.len())?;
            for ((sequence, lowered), reservation) in batch
                .sequences()
                .iter()
                .zip(&metadata.sequences)
                .zip(kv_reservations)
            {
                if reservation.execution_state_slot != sequence.state_slot {
                    return Err(Error::Execution {
                        message: format!(
                            "DeepSeek-V4 transaction {} KV reservation is bound to execution state slot {}, not {} (page owner slot {})",
                            transaction.get(),
                            reservation.execution_state_slot.get(),
                            sequence.state_slot.get(),
                            reservation.state_slot.get()
                        ),
                    });
                }
                let generation = states[lowered.state_index].generation();
                if reservation.execution_generation != generation {
                    return Err(Error::Execution {
                        message: format!(
                            "DeepSeek-V4 transaction {} has stale KV reservation generation {} for state slot {}; current generation is {}",
                            transaction.get(),
                            reservation.execution_generation,
                            lowered.state_index,
                            generation
                        ),
                    });
                }
            }

            let owners = metadata
                .sequences
                .iter()
                .map(|sequence| {
                    DeepSeekV4PackedSequenceOwner::capture(
                        sequence.state_index,
                        &states[sequence.state_index],
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            self.packed_transactions
                .validate_insert(transaction, &owners)?;

            let committed_state_indices = metadata
                .sequences
                .iter()
                .map(|sequence| sequence.state_index)
                .collect::<Vec<_>>();
            let mut working_states = committed_state_indices
                .iter()
                .map(|state_index| {
                    Self::checkout_transaction_sequence_topology(
                        &states[*state_index],
                        &self.operators,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let mut working_metadata = metadata;
            for (working_index, sequence) in working_metadata.sequences.iter_mut().enumerate() {
                sequence.state_index = working_index;
            }

            let physical = kv_reservations
                .iter()
                .map(|reservation| {
                    (
                        reservation.newly_allocated.clone(),
                        reservation.cow_replacement,
                    )
                })
                .collect::<Vec<_>>();
            self.operators
                .cuda_mut()?
                .prepare_kv_pages(transaction, &physical)?;

            let lowered = batch
                .sequences()
                .iter()
                .map(|sequence| {
                    let block_start =
                        usize::try_from(sequence.block_table.start).map_err(|_| Error::Model {
                            message: "KV block range exceeds usize".into(),
                        })?;
                    let block_end =
                        usize::try_from(sequence.block_table.end).map_err(|_| Error::Model {
                            message: "KV block range exceeds usize".into(),
                        })?;
                    self.operators.cuda_mut()?.lower_paged_binding(
                        transaction,
                        &batch.kv_block_ids()[block_start..block_end],
                        sequence.sequence_len as usize,
                    )
                })
                .collect::<Result<Vec<_>>>();
            let lowered = match lowered {
                Ok(lowered) => lowered,
                Err(error) => {
                    let cleanup = self
                        .operators
                        .cuda_mut()
                        .and_then(|cuda| cuda.rollback_kv_pages(transaction));
                    return Err(Self::packed_cuda_error_with_cleanup(
                        "DeepSeek-V4 paged binding lowering",
                        error,
                        cleanup,
                    ));
                }
            };
            for (state, binding) in working_states.iter_mut().zip(lowered) {
                state.paged_kv_binding = Some(binding);
            }

            let mut protected_pages = batch
                .kv_block_ids()
                .iter()
                .map(|block| KvPageId(block.get()))
                .collect::<BTreeSet<_>>();
            for reservation in kv_reservations {
                protected_pages.extend(reservation.newly_allocated.iter().copied());
                if let Some(cow) = reservation.cow_replacement {
                    protected_pages.insert(cow.source);
                    protected_pages.insert(cow.replacement);
                }
            }
            let topology = DeepSeekV4PackedTransactionTopology {
                batch: batch.clone(),
                metadata: working_metadata,
                committed_state_indices,
                working_states,
                execution: DeepSeekV4PackedTransactionExecution::Prepared,
            };
            if let Err(error) =
                self.packed_transactions
                    .insert(transaction, owners, protected_pages, topology)
            {
                let cleanup = self
                    .operators
                    .cuda_mut()
                    .and_then(|cuda| cuda.rollback_kv_pages(transaction));
                return Err(Self::packed_cuda_error_with_cleanup(
                    "DeepSeek-V4 packed transaction registration",
                    error,
                    cleanup,
                ));
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (transaction, states, batch, kv_reservations);
            Err(Error::Execution {
                message: "DeepSeek-V4 multi-session prepare requires CUDA support".into(),
            })
        }
    }

    fn end_transaction(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        intent: TransactionEndIntent,
    ) -> Result<TransactionEndProgress> {
        #[cfg(feature = "cuda")]
        {
            match intent {
                TransactionEndIntent::Publish => {
                    if self
                        .proposal_continuations
                        .values()
                        .any(|continuation| continuation.transaction == transaction)
                    {
                        return Err(Error::Execution {
                            message: format!(
                                "cannot publish DeepSeek-V4 transaction {} with active proposal continuations",
                                transaction.get()
                            ),
                        });
                    }
                    self.packed_transactions
                        .validate_states(transaction, states)?;
                    let mut topology = self.packed_transactions.take_payload(transaction)?;
                    if !matches!(
                        &topology.execution,
                        DeepSeekV4PackedTransactionExecution::Complete(_)
                    ) {
                        let error = Error::Execution {
                            message: format!(
                                "cannot publish DeepSeek-V4 transaction {} before packed execution completes",
                                transaction.get()
                            ),
                        };
                        self.packed_transactions
                            .put_payload(transaction, topology)?;
                        return Err(error);
                    }
                    if let Err(error) = self
                        .operators
                        .cuda_mut()
                        .and_then(|cuda| cuda.commit_kv_pages(transaction))
                    {
                        self.packed_transactions
                            .put_payload(transaction, topology)?;
                        return Err(error);
                    }
                    let DeepSeekV4PackedTransactionExecution::Complete(expert_leases) =
                        std::mem::replace(
                            &mut topology.execution,
                            DeepSeekV4PackedTransactionExecution::Prepared,
                        )
                    else {
                        unreachable!("packed execution was validated as complete")
                    };
                    for (state_index, working_state) in topology
                        .committed_state_indices
                        .iter()
                        .copied()
                        .zip(topology.working_states.drain(..))
                    {
                        states[state_index] = working_state;
                    }
                    drop(expert_leases);
                    self.packed_transactions.finalize_checkout(transaction);
                    Ok(TransactionEndProgress::Complete)
                }
                TransactionEndIntent::Abort => {
                    if self.abort_native_proposals(transaction)? == TransactionEndProgress::Pending
                    {
                        return Ok(TransactionEndProgress::Pending);
                    }
                    if !self.packed_transactions.contains(transaction) {
                        return Ok(TransactionEndProgress::Complete);
                    }
                    self.packed_transactions
                        .validate_states(transaction, states)?;
                    let mut topology = self.packed_transactions.take_payload(transaction)?;
                    let execution = std::mem::replace(
                        &mut topology.execution,
                        DeepSeekV4PackedTransactionExecution::Prepared,
                    );
                    if let DeepSeekV4PackedTransactionExecution::Waiting(continuation) = execution {
                        let (continuation, progress) = self.abort_packed_cuda_continuation(
                            &mut topology.working_states,
                            continuation,
                        );
                        match progress {
                            Ok(false) => {
                                topology.execution =
                                    DeepSeekV4PackedTransactionExecution::Waiting(continuation);
                                self.packed_transactions
                                    .put_payload(transaction, topology)?;
                                return Ok(TransactionEndProgress::Pending);
                            }
                            Err(error) => {
                                topology.execution =
                                    DeepSeekV4PackedTransactionExecution::Waiting(continuation);
                                self.packed_transactions
                                    .put_payload(transaction, topology)?;
                                return Err(error);
                            }
                            Ok(true) => {}
                        }
                    }
                    if let Err(error) = self
                        .operators
                        .cuda_mut()
                        .and_then(|cuda| cuda.rollback_kv_pages(transaction))
                    {
                        self.packed_transactions
                            .put_payload(transaction, topology)?;
                        return Err(error);
                    }
                    self.packed_transactions.finalize_checkout(transaction);
                    Ok(TransactionEndProgress::Complete)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (transaction, states, intent);
            Err(Error::Execution {
                message: "DeepSeek-V4 transaction terminalization requires CUDA support".into(),
            })
        }
    }

    fn retain_provisional_prefixes(
        &mut self,
        transaction: ExecutionTransactionId,
        sources: &[Self::SequenceState],
        branches: &mut [Self::SequenceState],
        executed_rows: &[usize],
        retained_rows: &[usize],
    ) -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (transaction, sources, branches, executed_rows, retained_rows);
            Err(Error::Execution {
                message: "DeepSeek-V4 provisional prefix retention requires CUDA support".into(),
            })
        }

        #[cfg(feature = "cuda")]
        {
            if self.operators.backend() != ModelExecutionBackend::Cuda {
                return Err(Error::Execution {
                    message:
                        "DeepSeek-V4 provisional prefix retention requires the CUDA packed backend"
                            .into(),
                });
            }
            let sequence_count = sources.len();
            if sequence_count == 0
                || branches.len() != sequence_count
                || executed_rows.len() != sequence_count
                || retained_rows.len() != sequence_count
            {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 provisional cohort shape mismatch: sources={} branches={} executed={} retained={}",
                        sequence_count,
                        branches.len(),
                        executed_rows.len(),
                        retained_rows.len()
                    ),
                });
            }
            self.packed_transactions
                .validate_states(transaction, branches)?;
            let mut topology = self.packed_transactions.take_payload(transaction)?;
            let result = (|| -> Result<()> {
                if !matches!(
                    &topology.execution,
                    DeepSeekV4PackedTransactionExecution::Complete(_)
                ) {
                    return Err(Error::Execution {
                        message: format!(
                            "cannot retain a DeepSeek-V4 provisional prefix before transaction {} completes execution",
                            transaction.get()
                        ),
                    });
                }
                if topology.working_states.len() != sequence_count {
                    return Err(Error::Internal {
                        message: format!(
                            "DeepSeek-V4 transaction {} working topology has {} sequences, expected {}",
                            transaction.get(),
                            topology.working_states.len(),
                            sequence_count
                        ),
                    });
                }

                for sequence_index in 0..sequence_count {
                    let source = &sources[sequence_index];
                    let branch = &topology.working_states[sequence_index];
                    let executed = executed_rows[sequence_index];
                    let retained = retained_rows[sequence_index];
                    if executed == 0 || retained == 0 || retained > executed {
                        return Err(Error::Model {
                            message: format!(
                                "invalid DeepSeek-V4 retained prefix for sequence {sequence_index}: retained={retained} executed={executed}"
                            ),
                        });
                    }
                    let expected_branch_position = source
                        .position()
                        .checked_add(executed)
                        .ok_or_else(|| Error::Model {
                            message: "DeepSeek-V4 branch position overflow".into(),
                        })?;
                    if branch.position() != expected_branch_position
                        || source.layers.len() != branch.layers.len()
                        || source.proposal_stages.len() != branch.proposal_stages.len()
                        || branch.paged_kv_binding.is_none()
                    {
                        return Err(Error::Model {
                            message: format!(
                                "DeepSeek-V4 provisional branch mismatch for sequence {sequence_index}: source_position={} branch_position={} expected={} source_layers={} branch_layers={} source_stages={} branch_stages={} paged_binding={}",
                                source.position(),
                                branch.position(),
                                expected_branch_position,
                                source.layers.len(),
                                branch.layers.len(),
                                source.proposal_stages.len(),
                                branch.proposal_stages.len(),
                                branch.paged_kv_binding.is_some()
                            ),
                        });
                    }
                    if retained < executed
                        && !self.operators.cuda.as_ref().is_some_and(|cuda| {
                            cuda.provisional_prefix_matches(
                                transaction,
                                sequence_index,
                                source.position(),
                                executed,
                                branch.layers.len(),
                            )
                        })
                    {
                        return Err(Error::Execution {
                            message: format!(
                                "DeepSeek-V4 provisional checkpoints do not cover sequence {sequence_index} for transaction {}",
                                transaction.get()
                            ),
                        });
                    }
                }

                for sequence_index in 0..sequence_count {
                    let source = &sources[sequence_index];
                    let branch = &mut topology.working_states[sequence_index];
                    let executed = executed_rows[sequence_index];
                    let retained = retained_rows[sequence_index];
                    if retained == executed {
                        continue;
                    }

                    for (layer, (source_state, state)) in
                        source.layers.iter().zip(&mut branch.layers).enumerate()
                    {
                        let metadata = self
                            .operators
                            .cuda_mut()?
                            .restore_provisional_prefix_checkpoint(
                                transaction,
                                sequence_index,
                                layer,
                                retained,
                                state.kv.window.cuda_state_mut(),
                            )?;
                        if let Some(metadata) = metadata {
                            state.kv.restore_provisional_prefix_metadata(metadata)?;
                        } else {
                            state
                                .kv
                                .restore_uncompressed_prefix_from(&source_state.kv, retained)?;
                        }
                    }
                    for (source_stage, branch_stage) in source
                        .proposal_stages
                        .iter()
                        .zip(&mut branch.proposal_stages)
                    {
                        branch_stage
                            .kv
                            .restore_uncompressed_prefix_from(&source_stage.kv, retained)?;
                    }

                    let final_position =
                        source
                            .position()
                            .checked_add(retained)
                            .ok_or_else(|| Error::Model {
                                message: "DeepSeek-V4 retained position overflow".into(),
                            })?;
                    branch
                        .paged_kv_binding
                        .as_mut()
                        .expect("validated provisional paged binding")
                        .retain_sequence_len(final_position)?;
                    branch.predictor = source.predictor.clone();
                    branch.core.restore_from(&source.core);
                    let binding = branch.core.begin_step()?;
                    branch.core.commit_step(binding, retained)?;
                }
                Ok(())
            })();
            let restore = self.packed_transactions.put_payload(transaction, topology);
            match (result, restore) {
                (result, Ok(())) => result,
                (Ok(()), Err(error)) => Err(error),
                (Err(error), Err(restore_error)) => Err(Error::Internal {
                    message: format!(
                        "DeepSeek-V4 provisional prefix retention failed ({error}); registry restore also failed ({restore_error})"
                    ),
                }),
            }
        }
    }

    fn execute_multi_session_batch_progress(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        batch: &ExecutionBatch,
    ) -> Result<MultiSessionBatchProgress> {
        #[cfg(feature = "cuda")]
        {
            if self.operators.backend() != ModelExecutionBackend::Cuda {
                return Err(Error::Execution {
                    message: "DeepSeek-V4 multi-session execution requires the CUDA packed backend"
                        .into(),
                });
            }
            if !self
                .operators
                .cuda
                .as_ref()
                .is_some_and(|cuda| cuda.has_kv_page_pool())
            {
                return Err(Error::Execution { message:
                    "DeepSeek-V4 multi-session execution requires a configured CUDA KV page pool"
                        .into(),
                 });
            }
            if batch
                .logits()
                .iter()
                .any(|request| matches!(request, LogitsRequest::Full))
            {
                return Err(Error::Execution {
                    message: "DeepSeek-V4 packed execution does not support full-vocabulary logits"
                        .into(),
                });
            }

            self.packed_transactions
                .validate_states(transaction, states)?;
            let mut topology = self.packed_transactions.take_payload(transaction)?;
            if topology.batch != *batch {
                self.packed_transactions
                    .put_payload(transaction, topology)?;
                return Err(Error::Execution {
                    message: format!(
                        "DeepSeek-V4 transaction {} execution batch does not match its prepared topology",
                        transaction.get()
                    ),
                });
            }
            if !topology.metadata.supports_native_cuda() {
                self.packed_transactions
                    .put_payload(transaction, topology)?;
                return Err(Error::Execution {
                    message: "DeepSeek-V4 batch shape is not supported by the CUDA packed executor"
                        .into(),
                });
            }
            if !matches!(
                &topology.execution,
                DeepSeekV4PackedTransactionExecution::Prepared
            ) {
                self.packed_transactions
                    .put_payload(transaction, topology)?;
                return Err(Error::Execution {
                    message: format!(
                        "DeepSeek-V4 transaction {} packed execution has already started",
                        transaction.get()
                    ),
                });
            }

            let continuation = match self.begin_cuda_packed_batch_continuation(
                transaction,
                &mut topology.working_states,
                batch,
                topology.metadata.clone(),
            ) {
                Ok(continuation) => continuation,
                Err(error) => {
                    self.packed_transactions
                        .put_payload(transaction, topology)?;
                    return Err(error);
                }
            };
            let progress = match self.progress_cuda_packed_batch(
                &mut topology.working_states,
                continuation,
                None,
            ) {
                Ok(DeepSeekV4PackedCudaProgress::Waiting(continuation, pending)) => {
                    topology.execution =
                        DeepSeekV4PackedTransactionExecution::Waiting(continuation);
                    Ok(MultiSessionBatchProgress::Waiting(pending))
                }
                Ok(DeepSeekV4PackedCudaProgress::Complete(output, expert_leases)) => {
                    topology.execution =
                        DeepSeekV4PackedTransactionExecution::Complete(expert_leases);
                    Ok(MultiSessionBatchProgress::Complete(output))
                }
                Err((error, continuation)) => {
                    topology.execution =
                        DeepSeekV4PackedTransactionExecution::Waiting(continuation);
                    Err(error)
                }
            };
            self.packed_transactions
                .put_payload(transaction, topology)?;
            progress
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (transaction, states, batch);
            Err(Error::Execution {
                message: "DeepSeek-V4 multi-session execution requires CUDA support".into(),
            })
        }
    }

    fn resume_multi_session_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        batch: &ExecutionBatch,
        continuation_id: BatchContinuationId,
        leases: ResidencyLeaseSet,
    ) -> Result<MultiSessionBatchProgress> {
        #[cfg(feature = "cuda")]
        {
            self.packed_transactions
                .validate_states(transaction, states)?;
            let mut topology = self.packed_transactions.take_payload(transaction)?;
            if topology.batch != *batch {
                self.packed_transactions
                    .put_payload(transaction, topology)?;
                return Err(Error::Execution {
                    message: format!(
                        "DeepSeek-V4 transaction {} resume batch does not match its prepared topology",
                        transaction.get()
                    ),
                });
            }
            let execution = std::mem::replace(
                &mut topology.execution,
                DeepSeekV4PackedTransactionExecution::Prepared,
            );
            let DeepSeekV4PackedTransactionExecution::Waiting(mut continuation) = execution else {
                topology.execution = execution;
                self.packed_transactions
                    .put_payload(transaction, topology)?;
                return Err(Error::Execution {
                    message: format!(
                        "DeepSeek-V4 transaction {} has no resumable packed continuation {}",
                        transaction.get(),
                        continuation_id.get()
                    ),
                });
            };
            if let Err(error) = validate_packed_cuda_resume_identity(
                continuation.id,
                &continuation.batch,
                continuation_id,
                batch,
            ) {
                topology.execution = DeepSeekV4PackedTransactionExecution::Waiting(continuation);
                self.packed_transactions
                    .put_payload(transaction, topology)?;
                return Err(error);
            }
            if let Err(error) =
                Self::validate_packed_cuda_resume_states(&topology.working_states, &continuation)
            {
                topology.execution = DeepSeekV4PackedTransactionExecution::Waiting(continuation);
                self.packed_transactions
                    .put_payload(transaction, topology)?;
                return Err(error);
            }
            let resume_validation = continuation
                .dependencies
                .as_mut()
                .ok_or_else(|| Error::Execution {
                    message: format!(
                        "DeepSeek-V4 packed continuation {} has no declared dependency set",
                        continuation_id.get()
                    ),
                })
                .and_then(|dependencies| dependencies.validate_resume(continuation_id, &leases));
            if let Err(error) = resume_validation {
                topology.execution = DeepSeekV4PackedTransactionExecution::Waiting(continuation);
                self.packed_transactions
                    .put_payload(transaction, topology)?;
                return Err(error);
            }

            let progress = match self.progress_cuda_packed_batch(
                &mut topology.working_states,
                continuation,
                Some(leases),
            ) {
                Ok(DeepSeekV4PackedCudaProgress::Waiting(continuation, pending)) => {
                    topology.execution =
                        DeepSeekV4PackedTransactionExecution::Waiting(continuation);
                    Ok(MultiSessionBatchProgress::Waiting(pending))
                }
                Ok(DeepSeekV4PackedCudaProgress::Complete(output, expert_leases)) => {
                    topology.execution =
                        DeepSeekV4PackedTransactionExecution::Complete(expert_leases);
                    Ok(MultiSessionBatchProgress::Complete(output))
                }
                Err((error, continuation)) => {
                    topology.execution =
                        DeepSeekV4PackedTransactionExecution::Waiting(continuation);
                    Err(error)
                }
            };
            self.packed_transactions
                .put_payload(transaction, topology)?;
            progress
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (transaction, states, batch, continuation_id, leases);
            Err(Error::Execution {
                message: "DeepSeek-V4 was built without CUDA packed continuation support".into(),
            })
        }
    }

    fn create_sequence_state(&mut self) -> Result<Self::SequenceState> {
        DeepSeekV4Runner::create_sequence_state(self)
    }

    fn with_sequence_state<T>(
        &mut self,
        state: &mut Self::SequenceState,
        execute: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        DeepSeekV4Runner::with_sequence_state(self, state, execute)
    }

    fn fork_sequence_state(&mut self) -> Result<Self::SequenceState> {
        DeepSeekV4Runner::fork_sequence_state(self)
    }

    fn fork_sequence_state_from(
        &mut self,
        source: &Self::SequenceState,
        expected_position: usize,
    ) -> Result<Self::SequenceState> {
        self.ensure_sequence_topology_available(source, "fork explicit sequence state")?;
        Self::fork_sequence_state_from_explicit(source, expected_position, &self.operators)
    }

    fn reset_sequence_state(&mut self, state: &mut Self::SequenceState) -> Result<()> {
        DeepSeekV4Runner::reset_sequence_state(self, state)
    }

    fn release_sequence_state(&mut self, state: Self::SequenceState) -> Result<()> {
        DeepSeekV4Runner::release_sequence_state(self, state)
    }

    fn multi_session_capabilities(&self) -> ferrule_common::execution::ExecutionCapabilities {
        let max_packed_rows = usize::try_from(u32::MAX).unwrap_or(usize::MAX);

        ferrule_common::execution::ExecutionCapabilities {
            max_batch_tokens: max_packed_rows,
            max_sequences: usize::MAX,
            max_prefill_query_tokens_per_sequence: max_packed_rows,
            max_decode_query_tokens_per_sequence: 1,
            max_top_k: std::num::NonZeroU32::new(40),
            supports_prefill: true,
            supports_decode: true,
            supports_mixed: true,
            full_logits_width: None,
            kv_binding_mode: ferrule_common::execution::KvBindingMode::Paged,
            logits_row_policy: {
                #[cfg(feature = "cuda")]
                {
                    if self
                        .operators
                        .cuda
                        .as_ref()
                        .is_some_and(|cuda| cuda.has_kv_page_pool())
                    {
                        ferrule_common::execution::LogitsRowPolicy::Any
                    } else {
                        ferrule_common::execution::LogitsRowPolicy::LastPerSequence
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    ferrule_common::execution::LogitsRowPolicy::LastPerSequence
                }
            },
        }
    }
}

#[cfg(test)]
mod packed_continuation_tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    };

    use ferrule_common::execution::{
        ExecutionSequence, ForwardMode, ForwardPhase, LogitsRequest, StateSlot,
    };
    use ferrule_common::{
        ContentHash, DestinationGeneration, DestinationSlotId, ExpertId as ProtocolExpertId,
        LayerId, MaterializationKey, MaterializedResourceId, ModelInstanceId, PayloadEncodingId,
        ResidencyBinding, SourceGeneration, SourceIdentityHash, ValidatedResidencyBinding,
    };

    use super::*;

    fn one_row_batch(token_id: u32, state_slot: u32) -> ExecutionBatch {
        ExecutionBatch::new(
            ForwardMode::Decode,
            vec![token_id],
            vec![0],
            vec![None],
            vec![LogitsRequest::None],
            vec![ExecutionSequence::new(
                StateSlot::new(state_slot),
                ForwardPhase::Decode,
                0..1,
                0,
                1,
                0..0,
            )],
            Vec::new(),
        )
    }

    fn shared_published_expert_lease(continuation: BatchContinuationId) -> ResidencyLeaseSet {
        let key = MaterializationKey::new(
            ModelInstanceId::new(17),
            SourceIdentityHash::new([1; 32]),
            ContentHash::new([2; 32]),
            MaterializedResourceId::routed_expert(LayerId::new(3), ProtocolExpertId::new(5)),
            PayloadEncodingId::new(1),
            BackendId::new(4),
            DeviceId::new(0),
            SourceGeneration::new(2),
            DestinationGeneration::new(7),
        )
        .unwrap();
        let binding = ValidatedResidencyBinding::new(
            key,
            ResidencyBinding::new(
                key.model(),
                key.resource(),
                key.backend(),
                key.device(),
                DestinationSlotId::new(9),
                key.destination_generation(),
            ),
        )
        .unwrap();
        ResidencyLeaseSet::new(
            [key],
            [binding],
            MappingEpoch::new(continuation.get()),
            DispatchFenceContract::new(
                OperationId::new(continuation.get()),
                FenceId::new(continuation.get()),
                key.backend(),
                key.device(),
            ),
        )
        .unwrap()
    }

    #[test]
    fn packed_continuation_ids_remain_unique_and_nonzero_at_exhaustion() {
        let mut next = NonZeroU64::new(u64::MAX - 1).unwrap();
        assert_eq!(
            take_packed_cuda_continuation_id(&mut next).unwrap().get(),
            u64::MAX - 1
        );
        assert_eq!(next.get(), u64::MAX);
        let error = take_packed_cuda_continuation_id(&mut next)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ID space is exhausted"));
        assert_eq!(next.get(), u64::MAX);
    }

    #[test]
    fn cuda_moe_materialization_backend_can_only_be_taken_once() {
        let mut owner = Some("physical-backend");
        assert_eq!(
            take_cuda_backend_once(ModelExecutionBackend::Cuda, Some(&mut owner), Option::take,),
            Some("physical-backend")
        );
        assert_eq!(
            take_cuda_backend_once(ModelExecutionBackend::Cuda, Some(&mut owner), Option::take,),
            None
        );
    }

    #[test]
    fn cuda_moe_materialization_backend_is_none_when_owner_is_missing() {
        let missing: Option<&mut Option<&'static str>> = None;
        assert_eq!(
            take_cuda_backend_once(ModelExecutionBackend::Cuda, missing, Option::take),
            None
        );
    }

    #[test]
    fn cpu_runner_never_transfers_a_materialization_backend() {
        let mut owner = Some("physical-backend");
        assert_eq!(
            take_cuda_backend_once(ModelExecutionBackend::Cpu, Some(&mut owner), Option::take),
            None
        );
        assert_eq!(owner, Some("physical-backend"));
    }

    #[test]
    fn residency_controller_is_forwarded_to_one_owner_only() {
        struct MockOwner {
            installed: std::cell::Cell<bool>,
        }

        let owner = MockOwner {
            installed: std::cell::Cell::new(false),
        };
        let install = |owner: &MockOwner, ()| {
            if owner.installed.replace(true) {
                Err(Error::Execution {
                    message: "mock residency controller is already installed".into(),
                })
            } else {
                Ok(())
            }
        };
        install_residency_control_on_owner(Some(&owner), (), install).unwrap();
        let duplicate = install_residency_control_on_owner(Some(&owner), (), install)
            .unwrap_err()
            .to_string();
        assert!(duplicate.contains("already installed"));

        let missing = install_residency_control_on_owner::<MockOwner, _>(None, (), install)
            .unwrap_err()
            .to_string();
        assert!(missing.contains("owner is unavailable"));
    }

    #[test]
    fn cuda_backend_shutdown_accepts_local_or_runtime_owned_backend() {
        validate_cuda_backend_shutdown_ownership(ModelExecutionBackend::Cuda, true, false, false)
            .unwrap();
        validate_cuda_backend_shutdown_ownership(ModelExecutionBackend::Cuda, true, true, true)
            .unwrap();
    }

    #[test]
    fn cuda_backend_shutdown_rejects_lost_or_unbridged_backend() {
        let lost = validate_cuda_backend_shutdown_ownership(
            ModelExecutionBackend::Cuda,
            false,
            true,
            true,
        )
        .unwrap_err()
        .to_string();
        assert!(lost.contains("lost its shared expert subsystem owner"));

        let unbridged = validate_cuda_backend_shutdown_ownership(
            ModelExecutionBackend::Cuda,
            true,
            true,
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(unbridged.contains("outside the quiesced runtime adapter"));
    }

    #[test]
    fn route_download_resume_edge_cannot_be_replayed() {
        let continuation = BatchContinuationId::new(19);
        let mut state = None;
        record_continuation_dependencies(
            &mut state,
            continuation,
            internal_continuation_dependency(continuation).unwrap(),
        )
        .unwrap();
        state
            .as_mut()
            .unwrap()
            .validate_resume(continuation, &empty_expert_lease_set(continuation).unwrap())
            .unwrap();

        let error = state
            .as_mut()
            .unwrap()
            .validate_resume(continuation, &empty_expert_lease_set(continuation).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("no unresolved dependency set; refusing resume replay"));
    }

    #[test]
    fn packed_resume_identity_requires_the_exact_id_and_batch() {
        let expected_id = BatchContinuationId::new(7);
        let batch = one_row_batch(11, 0);
        validate_packed_cuda_resume_identity(expected_id, &batch, expected_id, &batch).unwrap();

        let wrong_id = validate_packed_cuda_resume_identity(
            expected_id,
            &batch,
            BatchContinuationId::new(8),
            &batch,
        )
        .unwrap_err()
        .to_string();
        assert!(wrong_id.contains("continuation ID mismatch"));

        let wrong_batch = validate_packed_cuda_resume_identity(
            expected_id,
            &batch,
            expected_id,
            &one_row_batch(12, 0),
        )
        .unwrap_err()
        .to_string();
        assert!(wrong_batch.contains("does not exactly match"));

        let wrong_slot = validate_packed_cuda_resume_identity(
            expected_id,
            &batch,
            expected_id,
            &one_row_batch(11, 1),
        )
        .unwrap_err()
        .to_string();
        assert!(wrong_slot.contains("does not exactly match"));
    }

    fn transaction(value: u64) -> ExecutionTransactionId {
        ExecutionTransactionId::new(value).unwrap()
    }

    fn sequence_state() -> DeepSeekV4SequenceExecutionState {
        DeepSeekV4SequenceExecutionState::new(Vec::new(), Vec::new(), 8)
    }

    fn owner(
        state_index: usize,
        state: &DeepSeekV4SequenceExecutionState,
    ) -> DeepSeekV4PackedSequenceOwner {
        DeepSeekV4PackedSequenceOwner::capture(state_index, state).unwrap()
    }

    struct TransactionLeasePayload {
        leases: Vec<ResidencyLeaseSet>,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for TransactionLeasePayload {
        fn drop(&mut self) {
            self.drops.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[test]
    fn transaction_retains_expert_leases_through_terminal_checkout() {
        let states = [sequence_state()];
        let active = transaction(91);
        let drops = Arc::new(AtomicUsize::new(0));
        let mut registry = DeepSeekV4PackedTransactionRegistry::default();
        registry
            .insert(
                active,
                vec![owner(0, &states[0])],
                [],
                TransactionLeasePayload {
                    leases: vec![empty_expert_lease_set(BatchContinuationId::new(91)).unwrap()],
                    drops: Arc::clone(&drops),
                },
            )
            .unwrap();

        let payload = registry.take_payload(active).unwrap();
        assert_eq!(payload.leases.len(), 1);
        assert_eq!(drops.load(AtomicOrdering::Relaxed), 0);
        registry.put_payload(active, payload).unwrap();
        assert_eq!(drops.load(AtomicOrdering::Relaxed), 0);

        let payload = registry.take_payload(active).unwrap();
        registry.finalize_checkout(active);
        assert_eq!(payload.leases.len(), 1);
        assert_eq!(drops.load(AtomicOrdering::Relaxed), 0);
        drop(payload);
        assert_eq!(drops.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn two_transactions_retain_independent_leases_for_one_published_frame() {
        let states = [sequence_state(), sequence_state()];
        let first = transaction(99);
        let second = transaction(100);
        let first_lease = shared_published_expert_lease(BatchContinuationId::new(99));
        let second_lease = shared_published_expert_lease(BatchContinuationId::new(100));
        assert_eq!(first_lease.bindings(), second_lease.bindings());
        assert_ne!(
            first_lease.completion_contract(),
            second_lease.completion_contract()
        );

        let mut registry = DeepSeekV4PackedTransactionRegistry::default();
        registry
            .insert(first, vec![owner(0, &states[0])], [], first_lease)
            .unwrap();
        registry
            .insert(second, vec![owner(1, &states[1])], [], second_lease)
            .unwrap();

        let second_lease = registry.take_payload(second).unwrap();
        registry.finalize_checkout(second);
        let first_lease = registry.take_payload(first).unwrap();
        registry.finalize_checkout(first);
        assert_eq!(first_lease.bindings(), second_lease.bindings());
        assert!(registry.is_empty());
    }

    #[test]
    fn c2_packed_transactions_complete_in_reverse_order_without_cross_release() {
        let states = [sequence_state(), sequence_state()];
        let first = transaction(101);
        let second = transaction(102);
        let mut registry = DeepSeekV4PackedTransactionRegistry::default();
        registry
            .insert(first, vec![owner(0, &states[0])], [KvPageId(1)], "first")
            .unwrap();
        registry
            .insert(second, vec![owner(1, &states[1])], [KvPageId(2)], "second")
            .unwrap();

        let suspended = registry.take_payload(first).unwrap();
        registry.put_payload(first, suspended).unwrap();
        let second_payload = registry.take_payload(second).unwrap();
        registry.finalize_checkout(second);
        assert_eq!(second_payload, "second");
        assert_eq!(registry.owner_of(&states[0]), Some(first));
        let first_payload = registry.take_payload(first).unwrap();
        registry.finalize_checkout(first);
        assert_eq!(first_payload, "first");
        assert!(registry.is_empty());
    }

    #[test]
    fn packed_transactions_reject_the_same_sequence() {
        let states = [sequence_state()];
        let first = transaction(201);
        let second = transaction(202);
        let mut registry = DeepSeekV4PackedTransactionRegistry::default();
        registry
            .insert(first, vec![owner(0, &states[0])], [], ())
            .unwrap();

        let error = registry
            .insert(second, vec![owner(0, &states[0])], [], ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("already owned by transaction 201"));
        assert_eq!(registry.len(), 1);
        registry.take_payload(first).unwrap();
        registry.finalize_checkout(first);
    }

    #[test]
    fn unrelated_topology_mutation_is_not_globally_blocked() {
        let states = [sequence_state(), sequence_state()];
        let active = transaction(301);
        let mut registry = DeepSeekV4PackedTransactionRegistry::default();
        registry
            .insert(active, vec![owner(0, &states[0])], [KvPageId(11)], ())
            .unwrap();

        assert!(
            registry
                .ensure_sequence_available(&states[0], "reset")
                .is_err()
        );
        registry
            .ensure_sequence_available(&states[1], "reset")
            .unwrap();
        assert!(
            registry
                .ensure_pages_available(&[KvPageId(11)], "release")
                .is_err()
        );
        registry
            .ensure_pages_available(&[KvPageId(12)], "release")
            .unwrap();
        registry.take_payload(active).unwrap();
        registry.finalize_checkout(active);
    }

    #[test]
    fn c4_sibling_cancellation_releases_only_the_packed_owner() {
        let states = [
            sequence_state(),
            sequence_state(),
            sequence_state(),
            sequence_state(),
        ];
        let packed_owner = transaction(401);
        let mut registry = DeepSeekV4PackedTransactionRegistry::default();
        registry
            .insert(
                packed_owner,
                vec![owner(0, &states[0])],
                [KvPageId(21)],
                "cancelled",
            )
            .unwrap();

        for sibling in &states[1..] {
            registry
                .ensure_sequence_available(sibling, "release sibling")
                .unwrap();
        }
        let cancelled = registry.take_payload(packed_owner).unwrap();
        registry.finalize_checkout(packed_owner);
        assert_eq!(cancelled, "cancelled");
        assert!(
            states
                .iter()
                .all(|state| registry.owner_of(state).is_none())
        );
    }

    #[test]
    fn packed_transaction_rejects_stale_identity_and_generation() {
        let mut states = [sequence_state()];
        let active = transaction(501);
        let mut registry = DeepSeekV4PackedTransactionRegistry::default();
        registry
            .insert(active, vec![owner(0, &states[0])], [], ())
            .unwrap();

        let original = std::mem::replace(&mut states[0], sequence_state());
        let identity_error = registry
            .validate_states(active, &states)
            .unwrap_err()
            .to_string();
        assert!(identity_error.contains("changed topology identity"));

        states[0] = original;
        states[0].reset_for_reuse();
        let generation_error = registry
            .validate_states(active, &states)
            .unwrap_err()
            .to_string();
        assert!(generation_error.contains("stale sequence generation/position"));
        registry.take_payload(active).unwrap();
        registry.finalize_checkout(active);
    }

    #[test]
    fn packed_transaction_failure_restores_payload_for_retry() {
        let states = [sequence_state()];
        let active = transaction(601);
        let mut registry = DeepSeekV4PackedTransactionRegistry::default();
        registry
            .insert(
                active,
                vec![owner(0, &states[0])],
                [KvPageId(31)],
                String::from("model-and-kv-owned"),
            )
            .unwrap();

        let payload = registry.take_payload(active).unwrap();
        assert_eq!(payload, "model-and-kv-owned");
        assert_eq!(registry.owner_of(&states[0]), Some(active));
        assert!(
            registry
                .ensure_pages_available(&[KvPageId(31)], "release")
                .is_err()
        );
        registry.put_payload(active, payload).unwrap();

        let retried = registry.take_payload(active).unwrap();
        assert_eq!(retried, "model-and-kv-owned");
        drop(retried);
        registry.finalize_checkout(active);
    }

    #[test]
    fn packed_transaction_quiescence_releases_all_indexes_without_leaks() {
        let states = [sequence_state(), sequence_state()];
        let first = transaction(701);
        let second = transaction(702);
        let shared_page = KvPageId(41);
        let mut registry = DeepSeekV4PackedTransactionRegistry::default();
        registry
            .insert(
                first,
                vec![owner(0, &states[0])],
                [shared_page],
                String::from("first"),
            )
            .unwrap();
        registry
            .insert(
                second,
                vec![owner(1, &states[1])],
                [shared_page],
                String::from("second"),
            )
            .unwrap();

        let first_payload = registry.take_payload(first).unwrap();
        drop(first_payload);
        registry.finalize_checkout(first);
        assert!(registry.contains(second));
        assert!(
            registry
                .ensure_pages_available(&[shared_page], "release")
                .is_err()
        );
        let payload = registry.take_payload(second).unwrap();
        assert_eq!(payload, "second");
        drop(payload);
        registry.finalize_checkout(second);

        assert!(registry.is_empty());
        assert!(
            states
                .iter()
                .all(|state| registry.owner_of(state).is_none())
        );
        registry
            .ensure_pages_available(&[shared_page], "release")
            .unwrap();
    }
}

#[cfg(test)]
mod expert_runtime_tests {
    use std::sync::Arc;

    use crate::moe::streaming::{
        ExpertId, ExpertLoadSource, ExpertSourceCatalog, ExpertStreamingPolicy,
    };

    use super::*;

    fn prepared_layer() -> DeepSeekV4PreparedLayerExperts {
        DeepSeekV4PreparedLayerExperts::new(
            Arc::new(ExpertSourceCatalog::from_sources([(
                ExpertId::new(0, 0),
                ExpertLoadSource::LocalShard {
                    path: "expert.safetensors".into(),
                    offset: 0,
                    bytes: 16,
                },
            )])),
            ExpertStreamingPolicy::quality_first(1),
        )
    }

    #[test]
    fn runner_allocates_layer_expert_runtime_only_for_cpu() {
        let layers = [prepared_layer()];
        let cpu = build_cpu_expert_runtimes(ModelExecutionBackend::Cpu, &layers);
        let cuda = build_cpu_expert_runtimes(ModelExecutionBackend::Cuda, &layers);

        assert_eq!(cpu.as_deref().map(<[_]>::len), Some(1));
        assert!(cuda.is_none());
        assert!(Arc::ptr_eq(
            cpu.as_ref().unwrap()[0].expert_planner.source_catalog(),
            layers[0].source_catalog(),
        ));
    }
}
