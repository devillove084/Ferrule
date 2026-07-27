//! DeepSeek-V4 runner: ModelRunner implementation.

#[cfg(feature = "cuda")]
use std::collections::BTreeMap;
#[cfg(any(feature = "cuda", test))]
use std::collections::BTreeSet;
#[cfg(feature = "cuda")]
use std::collections::HashMap;

#[cfg(any(feature = "cuda", test))]
use std::num::NonZeroU64;

#[cfg(any(feature = "cuda", test))]
use std::ops::Range;
use std::path::Path;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[cfg(feature = "cuda")]
use crate::execution::ExecutionShapeKey;
use crate::execution::ModelExecutionBackend;
#[cfg(feature = "cuda")]
use crate::execution::{OwnedArenaCheckout, PersistentArenaPool, SequenceStepBinding};
use crate::moe::prediction::ExpertHotsetPredictor;
use crate::moe::streaming::ExpertStreamingReader;
#[cfg(all(feature = "cuda", feature = "cutlass"))]
use crate::runner::NativeProposal;
use crate::runner::{
    BatchContinuationCancelOutcome, BatchContinuationId, ModelCompletionReactor, ModelInfo,
    ModelRunner, MultiSessionBatchProgress, MultiSessionRunner, NativeProposalProgress,
    NativeProposalSource, ResidentModelRunner,
};
#[cfg(feature = "cuda")]
use crate::runner::{PendingExpertLoad, PendingModelProgress, TokenLogit};
use ferrule_common::execution::{ExecutionBatch, ExecutionTransactionId};
#[cfg(feature = "cuda")]
use ferrule_common::execution::{
    ExecutionIntent, ExecutionOutput, LogitsOutput, LogitsRequest, LogitsRow,
    TokenLogit as ExecutionTokenLogit,
};
#[cfg(any(feature = "cuda", test))]
use ferrule_common::execution::{ForwardMode, ForwardPhase};

use ferrule_common::expert_residency::{ExpertResidencyControl, ExpertResidencyRequirements};
use ferrule_common::{CompletionHub, Error, Result};

use super::artifact::DeepSeekV4ArtifactModel;

#[cfg(feature = "cuda")]
use super::cuda_cache::{
    DeepSeekV4CudaComputeQuiescence, DeepSeekV4DecodeBuffers, DeepSeekV4OutputHeadTopKDownload,
};
#[cfg(all(feature = "cuda", feature = "cutlass"))]
use super::cuda_cache::{
    DeepSeekV4DsparkAttentionBuffers, DeepSeekV4DsparkMainBuffers,
    DeepSeekV4DsparkProposalHeadBuffers,
};
use super::expert_io::{DeepSeekV4ExpertIoLayerSnapshot, DeepSeekV4ExpertIoSnapshot};
#[cfg(all(feature = "cuda", feature = "cutlass"))]
use super::layer::{
    DeepSeekV4DsparkLayerContinuation, DeepSeekV4DsparkLayerProgress, DeepSeekV4LayerArena,
};
#[cfg(feature = "cuda")]
use super::layer::{
    DeepSeekV4LayerArenaVariants, DeepSeekV4PackedLayerContinuation, DeepSeekV4PackedLayerProgress,
};
use super::layer::{DeepSeekV4LayerExpertRuntime, DeepSeekV4LayerState};
#[cfg(all(feature = "cuda", feature = "cutlass"))]
use super::mtp::DeepSeekV4DsparkProtocol;
use super::mtp::DeepSeekV4MtpModel;
use super::operators::{
    DeepSeekV4AttentionProfileStats, DeepSeekV4LayerProfileStats, DeepSeekV4OperatorContext,
    DeepSeekV4OperatorRuntimeCounters,
};
pub use super::prepared::DeepSeekV4PrepareOptions;
#[cfg(feature = "cuda")]
use super::prepared::DeepSeekV4PreparedResources;
use super::prepared::{
    DeepSeekV4ExecutionPolicy, DeepSeekV4PreparedLayerExperts, DeepSeekV4PreparedModelPlan, prepare,
};
use super::sequence::DeepSeekV4SequenceExecutionState;
#[cfg(feature = "cuda")]
use super::sequence::{DeepSeekV4PagedKvBinding, DeepSeekV4SequenceMoeAccessEvent};

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

#[derive(Debug, Clone)]
pub struct DeepSeekV4ObservabilitySnapshot {
    pub position: usize,
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
    #[cfg(feature = "cutlass")]
    dspark_main_buffers: Option<DeepSeekV4DsparkMainBuffers>,
    initialized: bool,
    next_layer_index: usize,
    current_layer: Option<DeepSeekV4PackedLayerContinuation>,
    output_head: DeepSeekV4PackedOutputHeadState,
    cancel_quiescence: Option<DeepSeekV4CudaComputeQuiescence>,
    moe_access_events: Vec<DeepSeekV4SequenceMoeAccessEvent>,
    paged_bindings_active: bool,
    provisional_checkpoints: bool,
    failed: bool,
}

#[cfg(feature = "cuda")]
enum DeepSeekV4PackedCudaProgress {
    Waiting(DeepSeekV4PackedCudaContinuation, PendingModelProgress),
    Complete(ExecutionOutput),
}

#[cfg(feature = "cuda")]
enum DeepSeekV4PackedCudaStep {
    Waiting(PendingModelProgress),
    Complete(ExecutionOutput),
}

struct DeepSeekV4RunnerObservability {
    output: DeepSeekV4OutputProfileStats,
}

impl DeepSeekV4RunnerObservability {
    fn new() -> Self {
        Self {
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
    #[cfg(feature = "cuda")]
    model_instance: u64,
    #[cfg(feature = "cuda")]
    expert_residency: Option<Box<dyn ExpertResidencyControl>>,
    #[cfg(feature = "cuda")]
    layer_arena_pool:
        PersistentArenaPool<DeepSeekV4LayerArenaPoolKey, DeepSeekV4LayerArenaVariants>,
    #[cfg(feature = "cuda")]
    packed_cuda_continuations: HashMap<ExecutionTransactionId, DeepSeekV4PackedCudaContinuation>,
    #[cfg(feature = "cuda")]
    prepared_paged_bindings:
        HashMap<ExecutionTransactionId, Vec<(usize, Option<DeepSeekV4PagedKvBinding>)>>,
    #[cfg(feature = "cuda")]
    next_packed_cuda_continuation_id: NonZeroU64,
    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    dspark_proposal_arena_pool: Vec<DeepSeekV4DsparkProposalArena>,
    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    dspark_proposal_continuations:
        HashMap<BatchContinuationId, DeepSeekV4DsparkProposalContinuation>,
    dspark_proposal_source: Option<NativeProposalSource>,
    /// E3: per-sequence state. The runner wraps one default sequence.
    sequence: DeepSeekV4SequenceExecutionState,
    observability: DeepSeekV4RunnerObservability,
    completion_hub: CompletionHub,
    expert_reader: ExpertStreamingReader,
    shutdown: bool,
}

#[cfg(all(feature = "cuda", feature = "cutlass"))]
struct DeepSeekV4DsparkProposalArena {
    stages: Box<[DeepSeekV4LayerArena]>,
    attention: DeepSeekV4DsparkAttentionBuffers,
    head: DeepSeekV4DsparkProposalHeadBuffers,
    hc_state: ferrule_cuda::context::CudaF32Buffer,
}

#[cfg(all(feature = "cuda", feature = "cutlass"))]
struct DeepSeekV4DsparkProposalContinuation {
    transaction: ExecutionTransactionId,
    id: BatchContinuationId,
    anchor_token_id: u32,
    token_ids: Vec<u32>,
    sequence_tokens: usize,
    paged_binding: Option<DeepSeekV4PagedKvBinding>,
    arena: Option<DeepSeekV4DsparkProposalArena>,
    initialized: bool,
    stage: usize,
    current_layer: Option<DeepSeekV4DsparkLayerContinuation>,
    moe_access_events: Vec<DeepSeekV4SequenceMoeAccessEvent>,
    head_download: Option<ferrule_cuda::context::CudaI32HostDownload>,
    paged_binding_active: bool,
    callback_armed: bool,
    failed: Option<String>,
}

#[cfg(all(feature = "cuda", feature = "cutlass"))]
enum DeepSeekV4DsparkProposalStep {
    Waiting(PendingModelProgress),
    Complete(NativeProposal),
}

#[cfg(feature = "cuda")]
fn expert_residency_layer_capacities(resources: &DeepSeekV4PreparedResources) -> Vec<usize> {
    let mut capacities = resources
        .layer_experts()
        .iter()
        .map(DeepSeekV4PreparedLayerExperts::resident_capacity)
        .collect::<Vec<_>>();
    if capacities.len() == resources.model().config.num_layers {
        if let Some(mtp) = resources.mtp() {
            debug_assert_eq!(mtp.layers.len(), resources.mtp_layer_experts().len());
            for (stage, experts) in mtp.layers.iter().zip(resources.mtp_layer_experts()) {
                debug_assert_eq!(stage.execution_layer, capacities.len());
                capacities.push(experts.resident_capacity());
            }
        }
    }
    capacities
}

#[cfg(all(feature = "cuda", feature = "cutlass"))]
const DSPARK_PROPOSAL_IMPLEMENTATION: &str = "deepseek-v4-dspark-cuda-cutlass-v1";

fn prepared_dspark_proposal_source(
    plan: &DeepSeekV4PreparedModelPlan,
    backend: ModelExecutionBackend,
) -> Result<Option<NativeProposalSource>> {
    let Some(mtp) = plan.resources().mtp() else {
        return Ok(None);
    };
    if backend != ModelExecutionBackend::Cuda {
        return Err(Error::Model(
            "DeepSeek-V4 checkpoint contains a native proposal attachment that requires the CUDA backend"
                .into(),
        ));
    }
    #[cfg(not(all(feature = "cuda", feature = "cutlass")))]
    {
        let _ = mtp;
        Err(Error::Model(
            "DeepSeek-V4 checkpoint contains a native proposal attachment that requires building with CUDA and CUTLASS support"
                .into(),
        ))
    }
    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    {
        if mtp.prediction_heads.is_none() {
            return Err(Error::Model(
                "DeepSeek-V4 DSpark proposal source is missing prediction heads".into(),
            ));
        }
        let source = NativeProposalSource {
            implementation: DSPARK_PROPOSAL_IMPLEMENTATION,
            prepared_plan_id: plan.generation(),
            native_width: mtp.config.block_size,
        };
        source.validate()?;
        Ok(Some(source))
    }
}

fn cold_expert_residency(
    expert_counts: &[usize],
) -> Vec<Box<[crate::moe::prediction::ExpertResidency]>> {
    expert_counts
        .iter()
        .map(|&count| vec![crate::moe::prediction::ExpertResidency::Cold; count].into_boxed_slice())
        .collect()
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
        .ok_or_else(|| {
            Error::Execution("DeepSeek-V4 packed continuation ID space is exhausted".into())
        })?;
    let id = BatchContinuationId::new(value)?;
    *next = following;
    Ok(id)
}

#[cfg(any(feature = "cuda", test))]
fn validate_packed_cuda_resume_identity(
    expected_id: BatchContinuationId,
    expected_batch: &ExecutionBatch,
    actual_id: BatchContinuationId,
    actual_batch: &ExecutionBatch,
) -> Result<()> {
    if actual_id != expected_id {
        return Err(Error::Execution(format!(
            "DeepSeek-V4 packed continuation ID mismatch: expected {}, got {}",
            expected_id.get(),
            actual_id.get()
        )));
    }
    if actual_batch != expected_batch {
        return Err(Error::Execution(
            "DeepSeek-V4 packed continuation batch does not exactly match the suspended batch"
                .into(),
        ));
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
                .map_err(|_| Error::Model("DeepSeek-V4 state slot exceeds usize".into()))?;
            if state_index >= state_count {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 state slot {state_index} is missing from {state_count} states"
                )));
            }
            if !state_indices.insert(state_index) {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 state slot {state_index} is referenced more than once"
                )));
            }
            let start = usize::try_from(sequence.query.start)
                .map_err(|_| Error::Model("DeepSeek-V4 query start exceeds usize".into()))?;
            let end = usize::try_from(sequence.query.end)
                .map_err(|_| Error::Model("DeepSeek-V4 query end exceeds usize".into()))?;
            if start != expected_start || start >= end || end > batch.len() {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 sequence {sequence_index} query range {start}..{end} does not densely cover packed rows from {expected_start}"
                )));
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
            return Err(Error::Model(format!(
                "DeepSeek-V4 sequence queries cover {expected_start} of {} packed rows",
                batch.len()
            )));
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
    /// Single-sequence multi-row batches (DSpark verification: 1 sequence × V
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
    pub(super) fn physical_expert_io_resource_limits(
        &self,
    ) -> Result<ferrule_common::expert_io::ExpertIoResourceLimits> {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            use ferrule_common::expert_io::{ExpertIoResourceDemand, ExpertIoResourceLimits};

            let reader_capacity = self
                .expert_reader
                .physical_resource_capacity()?
                .ok_or_else(|| {
                    Error::Model(
                        "DeepSeek-V4 resident inference requires CUDA-pinned io_uring resource topology"
                            .into(),
                    )
                })?;
            let mut max_operation = ExpertIoResourceDemand::default();
            for layer in self
                .plan
                .resources()
                .layer_experts()
                .iter()
                .chain(self.plan.resources().mtp_layer_experts())
            {
                for (expert, source) in layer.source_catalog().iter() {
                    let demand = self
                        .expert_reader
                        .plan_load_source_pinned(*expert, source)?
                        .ok_or_else(|| {
                            Error::Model(
                                "DeepSeek-V4 expert resource planning requires pinned io_uring"
                                    .into(),
                            )
                        })?
                        .demand();
                    max_operation.read_slots = max_operation.read_slots.max(demand.read_slots);
                    max_operation.storage_read_bytes = max_operation
                        .storage_read_bytes
                        .max(demand.storage_read_bytes);
                    max_operation.pinned_host_bytes = max_operation
                        .pinned_host_bytes
                        .max(demand.pinned_host_bytes);
                    max_operation.upload_slots =
                        max_operation.upload_slots.max(demand.upload_slots);
                    max_operation.h2d_bytes = max_operation.h2d_bytes.max(demand.h2d_bytes);
                    max_operation.install_slots =
                        max_operation.install_slots.max(demand.install_slots);
                    max_operation.device_install_bytes = max_operation
                        .device_install_bytes
                        .max(demand.device_install_bytes);
                }
            }
            if max_operation.is_empty() {
                return Err(Error::Model(
                    "DeepSeek-V4 prepared plan has no physical expert-I/O demand".into(),
                ));
            }
            ExpertIoResourceDemand {
                read_slots: max_operation.read_slots,
                storage_read_bytes: max_operation.storage_read_bytes,
                pinned_host_bytes: max_operation.pinned_host_bytes,
                ..ExpertIoResourceDemand::default()
            }
            .validate_within(reader_capacity, "DeepSeek-V4 pinned read")?;
            let upload_slots = u64::try_from(
                self.plan
                    .resources()
                    .policy()
                    .expert_upload_inflight()
                    .checked_add(1)
                    .ok_or_else(|| Error::Model("expert upload slot capacity overflow".into()))?,
            )
            .map_err(|_| Error::Model("expert upload slot capacity exceeds u64".into()))?;
            let transfer_bytes = max_operation
                .h2d_bytes
                .checked_mul(upload_slots)
                .ok_or_else(|| Error::Model("expert transfer byte capacity overflow".into()))?;
            let capacity = ExpertIoResourceDemand {
                upload_slots,
                h2d_bytes: transfer_bytes,
                install_slots: upload_slots,
                device_install_bytes: transfer_bytes,
                ..reader_capacity
            };
            let demand_reserve = max_operation;
            return ExpertIoResourceLimits {
                capacity,
                demand_reserve,
            }
            .validate();
        }
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            Ok(ferrule_common::expert_io::ExpertIoResourceLimits::default())
        }
    }

    pub(super) fn install_physical_expert_io_resource_control(
        &mut self,
        control: Box<dyn ferrule_common::expert_io::ExpertIoResourceControl>,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            return self
                .operators
                .cuda_mut()?
                .install_expert_io_resource_control(control);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = control;
            Err(Error::Model(
                "DeepSeek-V4 expert-I/O resource control requires CUDA".into(),
            ))
        }
    }

    pub(super) fn uninstall_physical_expert_io_resource_control(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            return self
                .operators
                .cuda_mut()?
                .uninstall_expert_io_resource_control();
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::Model(
                "DeepSeek-V4 expert-I/O resource control requires CUDA".into(),
            ))
        }
    }

    pub fn new_with_operator_backend(
        model: DeepSeekV4ArtifactModel,
        options: DeepSeekV4PrepareOptions,
        operator_backend: ModelExecutionBackend,
    ) -> Result<Self> {
        if operator_backend != ModelExecutionBackend::Cuda {
            return Err(Error::Model(
                "DeepSeek-V4 resident inference requires the CUDA packed execution backend".into(),
            ));
        }
        let plan = prepare(model, options)?;
        let dspark_proposal_source = prepared_dspark_proposal_source(&plan, operator_backend)?;
        let options = *plan.resources().prepare_options();
        let policy = plan.resources().policy();
        let model = plan.resources().model();
        let completion_hub = CompletionHub::new();
        let mut operators = DeepSeekV4OperatorContext::new_with_completion_hub(
            operator_backend,
            policy,
            options.expert_memory_policy,
            completion_hub.clone(),
        )?;
        #[cfg(feature = "cuda")]
        if operator_backend == ModelExecutionBackend::Cuda {
            let resources = plan.resources();
            let mut layer_slot_capacities = resources
                .layer_experts()
                .iter()
                .enumerate()
                .map(|(layer, experts)| (layer, experts.resident_capacity()))
                .collect::<Vec<_>>();
            if let Some(mtp) = resources.mtp() {
                if mtp.layers.len() != resources.mtp_layer_experts().len() {
                    return Err(Error::Model(format!(
                        "DeepSeek-V4 DSpark stage/expert capacity mismatch: stages={} capacities={}",
                        mtp.layers.len(),
                        resources.mtp_layer_experts().len()
                    )));
                }
                layer_slot_capacities.extend(
                    mtp.layers
                        .iter()
                        .zip(resources.mtp_layer_experts())
                        .map(|(stage, experts)| {
                            (stage.execution_layer, experts.resident_capacity())
                        }),
                );
            }
            operators.configure_expert_frame_pool(
                model.config.num_routed_experts,
                &layer_slot_capacities,
                model.config.hidden_size,
                model.config.moe_intermediate_size,
            )?;
            operators.compile_cuda_execution_image(plan.generation(), plan.resources())?;
        }

        let mut layer_states = Vec::with_capacity(options.max_layers);
        for layer_idx in 0..options.max_layers {
            let state_start = operators.profile_start();
            layer_states.push(model.new_layer_sequence_state(layer_idx)?);
            if let Some(state_start) = state_start {
                operators.record_layer_state_init(layer_idx, duration_us(state_start.elapsed()));
            }
        }
        let dspark_stage_states = plan
            .resources()
            .mtp()
            .map(|mtp| {
                mtp.layers
                    .iter()
                    .map(|stage| DeepSeekV4LayerState::new(stage.transformer.attention.config))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let cpu_expert_runtimes =
            build_cpu_expert_runtimes(operator_backend, plan.resources().layer_experts());

        let sequence = DeepSeekV4SequenceExecutionState::new(
            layer_states,
            dspark_stage_states,
            model.config.num_routed_experts,
        );

        let expert_reader_max_tensor_bytes = options.expert_reader_max_tensor_bytes;
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let expert_reader = if operator_backend == ModelExecutionBackend::Cuda {
            let allocator = operators
                .cuda
                .as_ref()
                .expect("CUDA backend initialized above")
                .pinned_host_allocator();
            ExpertStreamingReader::from_env_with_cuda_pinned(
                expert_reader_max_tensor_bytes,
                allocator,
                completion_hub.clone(),
            )?
        } else {
            ExpertStreamingReader::from_env_with_completion_hub(
                expert_reader_max_tensor_bytes,
                completion_hub.clone(),
            )?
        };
        #[cfg(all(feature = "cuda", not(target_os = "linux")))]
        let expert_reader = if operator_backend == ModelExecutionBackend::Cuda {
            return Err(Error::Model(
                "DeepSeek-V4 CUDA resident expert materialization is unsupported on non-Linux platforms; Linux pinned io_uring is required"
                    .into(),
            ));
        } else {
            ExpertStreamingReader::from_env_with_completion_hub(
                expert_reader_max_tensor_bytes,
                completion_hub.clone(),
            )?
        };
        #[cfg(not(feature = "cuda"))]
        let expert_reader = ExpertStreamingReader::from_env_with_completion_hub(
            expert_reader_max_tensor_bytes,
            completion_hub.clone(),
        )?;

        Ok(Self {
            plan,
            operators,
            cpu_expert_runtimes,
            #[cfg(feature = "cuda")]
            model_instance: NEXT_DSV4_MODEL_INSTANCE.fetch_add(1, Ordering::Relaxed),
            #[cfg(feature = "cuda")]
            expert_residency: None,
            #[cfg(feature = "cuda")]
            layer_arena_pool: PersistentArenaPool::new(),
            #[cfg(feature = "cuda")]
            packed_cuda_continuations: HashMap::new(),
            #[cfg(feature = "cuda")]
            prepared_paged_bindings: HashMap::new(),
            #[cfg(feature = "cuda")]
            next_packed_cuda_continuation_id: NonZeroU64::new(1).expect("one is non-zero"),
            #[cfg(all(feature = "cuda", feature = "cutlass"))]
            dspark_proposal_arena_pool: Vec::new(),
            #[cfg(all(feature = "cuda", feature = "cutlass"))]
            dspark_proposal_continuations: HashMap::new(),
            dspark_proposal_source,
            sequence,
            observability: DeepSeekV4RunnerObservability::new(),
            completion_hub,
            expert_reader,
            shutdown: false,
        })
    }

    pub fn load_hf_with_options(
        model_dir: &Path,
        max_tensor_bytes: u64,
        options: DeepSeekV4PrepareOptions,
    ) -> Result<Self> {
        Self::new_with_operator_backend(
            DeepSeekV4ArtifactModel::load_hf_with_limit(model_dir, max_tensor_bytes)?,
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
        Self::new_with_operator_backend(
            DeepSeekV4ArtifactModel::load_hf_with_limit(model_dir, max_tensor_bytes)?,
            options,
            operator_backend,
        )
    }

    pub fn model(&self) -> &DeepSeekV4ArtifactModel {
        self.plan.resources().model()
    }

    pub fn mtp(&self) -> Option<&DeepSeekV4MtpModel> {
        self.plan.resources().mtp()
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    fn build_dspark_proposal_arena(&mut self) -> Result<DeepSeekV4DsparkProposalArena> {
        let resources = self.plan.resources();
        let mtp = resources
            .mtp()
            .ok_or_else(|| Error::Model("DeepSeek-V4 DSpark attachment is missing".into()))?;
        let rows = ferrule_cuda::cutlass::DSPARK_PROPOSAL_ROWS;
        let mut stages = Vec::with_capacity(mtp.layers.len());
        for layer in &mtp.layers {
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
            .allocate_dspark_attention_buffers()?;
        let head = self
            .operators
            .cuda_mut()?
            .allocate_dspark_proposal_head_buffers()?;
        let hc_state = self.operators.cuda_mut()?.ops.zero_f32_buffer(
            rows.checked_mul(resources.model().config.hc_config().hc_hidden_size())
                .ok_or_else(|| Error::Model("DSpark proposal HC size overflow".into()))?,
        )?;
        Ok(DeepSeekV4DsparkProposalArena {
            stages: stages.into_boxed_slice(),
            attention,
            head,
            hc_state,
        })
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    fn begin_dspark_proposal_continuation(
        &mut self,
        transaction: ExecutionTransactionId,
        anchor_token_id: u32,
    ) -> Result<DeepSeekV4DsparkProposalContinuation> {
        if self.operators.backend() != ModelExecutionBackend::Cuda {
            return Err(Error::Model(
                "DeepSeek-V4 DSpark production proposal requires CUDA".into(),
            ));
        }
        if self.expert_residency.is_none() {
            return Err(Error::Execution(
                "runtime expert residency controller is not installed".into(),
            ));
        }
        let (protocol, stage_count) = {
            let mtp =
                self.plan.resources().mtp().ok_or_else(|| {
                    Error::Model("DeepSeek-V4 DSpark attachment is missing".into())
                })?;
            (
                DeepSeekV4DsparkProtocol::try_from(&mtp.config)?,
                mtp.layers.len(),
            )
        };
        let token_ids = protocol.draft_input_ids(anchor_token_id);
        let sequence_tokens = self.sequence.core.position();
        if sequence_tokens == 0 {
            return Err(Error::Model(
                "DeepSeek-V4 DSpark proposal requires committed target context".into(),
            ));
        }
        let paged_binding = self.sequence.paged_kv_binding.clone();
        let id = take_packed_cuda_continuation_id(&mut self.next_packed_cuda_continuation_id)?;
        let arena = match self.dspark_proposal_arena_pool.pop() {
            Some(arena) => arena,
            None => self.build_dspark_proposal_arena()?,
        };
        let arena_stage_count = arena.stages.len();
        if arena_stage_count != stage_count
            || stage_count != self.plan.resources().mtp_layer_experts().len()
        {
            self.dspark_proposal_arena_pool.push(arena);
            return Err(Error::Model(format!(
                "DeepSeek-V4 DSpark proposal arena mismatch: arenas={arena_stage_count} stages={stage_count} expert_catalogs={}",
                self.plan.resources().mtp_layer_experts().len()
            )));
        }
        Ok(DeepSeekV4DsparkProposalContinuation {
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
            head_download: None,
            paged_binding_active: false,
            callback_armed: false,
            failed: None,
        })
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    fn activate_dspark_proposal_binding(
        &mut self,
        continuation: &mut DeepSeekV4DsparkProposalContinuation,
    ) -> Result<()> {
        self.operators.cuda_mut()?.activate_paged_binding(
            continuation.transaction,
            continuation.paged_binding.as_ref(),
        )?;
        continuation.paged_binding_active = true;
        Ok(())
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    fn deactivate_dspark_proposal_binding(
        &mut self,
        continuation: &mut DeepSeekV4DsparkProposalContinuation,
    ) -> Result<()> {
        if continuation.paged_binding_active {
            self.operators
                .cuda_mut()?
                .activate_paged_binding(continuation.transaction, None)?;
            continuation.paged_binding_active = false;
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    fn pending_dspark_proposal_progress(
        continuation: &DeepSeekV4DsparkProposalContinuation,
    ) -> Result<PendingModelProgress> {
        let operations = continuation
            .current_layer
            .as_ref()
            .map(DeepSeekV4DsparkLayerContinuation::pending_operations)
            .unwrap_or_default()
            .into_iter()
            .map(|operation| {
                let layer = u32::try_from(operation.layer).map_err(|_| {
                    Error::Execution(format!(
                        "DeepSeek-V4 pending expert layer {} exceeds u32 ABI",
                        operation.layer
                    ))
                })?;
                let expert = u32::try_from(operation.expert).map_err(|_| {
                    Error::Execution(format!(
                        "DeepSeek-V4 pending expert {} exceeds u32 ABI",
                        operation.expert
                    ))
                })?;
                PendingExpertLoad::new(operation.operation_id, layer, expert)
            })
            .collect::<Result<Vec<_>>>()?;
        PendingModelProgress::new(continuation.transaction, continuation.id, operations)
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    fn poll_dspark_route_cancel_ready(
        &mut self,
        continuation: &mut DeepSeekV4DsparkProposalContinuation,
    ) -> Result<bool> {
        let Some(layer_continuation) = continuation.current_layer.as_mut() else {
            return Ok(true);
        };
        let stage = continuation.stage;
        let layer = self
            .plan
            .resources()
            .mtp()
            .and_then(|mtp| mtp.layers.get(stage))
            .ok_or_else(|| {
                Error::Internal(format!(
                    "current DSpark stage {stage} is outside the prepared attachment"
                ))
            })?;
        layer
            .transformer
            .poll_dspark_proposal_cancel_ready(layer_continuation, &mut self.operators)
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    fn arm_dspark_proposal_completion(
        &mut self,
        continuation: &mut DeepSeekV4DsparkProposalContinuation,
    ) {
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

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    fn progress_dspark_proposal(
        &mut self,
        continuation: &mut DeepSeekV4DsparkProposalContinuation,
    ) -> Result<DeepSeekV4DsparkProposalStep> {
        if let Some(failed) = continuation.failed.as_deref() {
            return Err(Error::Execution(format!(
                "DeepSeek-V4 native proposal continuation {} previously failed: {failed}",
                continuation.id.get()
            )));
        }
        let stage_count = self
            .plan
            .resources()
            .mtp()
            .ok_or_else(|| Error::Model("DeepSeek-V4 DSpark attachment is missing".into()))?
            .layers
            .len();
        if !continuation.initialized {
            let arena = continuation.arena.as_mut().ok_or_else(|| {
                Error::Internal("DeepSeek-V4 DSpark proposal arena is unavailable".into())
            })?;
            self.operators
                .cuda_mut()?
                .dspark_proposal_input_device_into(
                    continuation.anchor_token_id,
                    &mut arena.hc_state,
                )?;
            continuation.initialized = true;
        }

        while continuation.stage < stage_count {
            let stage = continuation.stage;
            if continuation.current_layer.is_none() {
                let mtp = self
                    .plan
                    .resources()
                    .mtp()
                    .expect("validated DSpark attachment above");
                let layer = &mtp.layers[stage].transformer;
                let prepared = &self.plan.resources().mtp_layer_experts()[stage];
                let residency = self.expert_residency.as_deref_mut().ok_or_else(|| {
                    Error::Execution("runtime expert residency controller is not installed".into())
                })?;
                let arena = continuation.arena.as_mut().ok_or_else(|| {
                    Error::Internal("DeepSeek-V4 DSpark proposal arena is unavailable".into())
                })?;
                let stage_arena = arena.stages.get_mut(stage).ok_or_else(|| {
                    Error::Internal(format!(
                        "DeepSeek-V4 DSpark stage arena {stage} is unavailable"
                    ))
                })?;
                continuation.current_layer =
                    Some(layer.begin_dspark_proposal_block_device_hc_device(
                        stage,
                        continuation.sequence_tokens,
                        residency,
                        prepared.source_catalog().as_ref(),
                        prepared.prefetch_capacity(),
                        stage_arena,
                        &mut arena.hc_state,
                        &continuation.token_ids,
                        &[],
                        &self.expert_reader,
                        &mut self.operators,
                        &mut arena.attention,
                    )?);
            }

            let mtp = self
                .plan
                .resources()
                .mtp()
                .expect("validated DSpark attachment above");
            let layer = &mtp.layers[stage].transformer;
            let prepared = &self.plan.resources().mtp_layer_experts()[stage];
            let residency = self.expert_residency.as_deref_mut().ok_or_else(|| {
                Error::Execution("runtime expert residency controller is not installed".into())
            })?;
            let arena = continuation.arena.as_mut().ok_or_else(|| {
                Error::Internal("DeepSeek-V4 DSpark proposal arena is unavailable".into())
            })?;
            let stage_arena = arena.stages.get_mut(stage).ok_or_else(|| {
                Error::Internal(format!(
                    "DeepSeek-V4 DSpark stage arena {stage} is unavailable"
                ))
            })?;
            let layer_continuation = continuation
                .current_layer
                .take()
                .expect("DSpark layer continuation was created above");
            match layer.resume_dspark_proposal_block_device_hc_device(
                layer_continuation,
                residency,
                prepared.source_catalog().as_ref(),
                stage_arena,
                &mut arena.hc_state,
                &self.expert_reader,
                &mut self.operators,
            )? {
                DeepSeekV4DsparkLayerProgress::Waiting(next) => {
                    continuation.current_layer = Some(next);
                    return Ok(DeepSeekV4DsparkProposalStep::Waiting(
                        Self::pending_dspark_proposal_progress(continuation)?,
                    ));
                }
                DeepSeekV4DsparkLayerProgress::Complete(events) => {
                    continuation.moe_access_events.extend(events);
                    continuation.stage += 1;
                }
            }
        }

        if continuation.head_download.is_none() {
            let arena = continuation.arena.as_mut().ok_or_else(|| {
                Error::Internal("DeepSeek-V4 DSpark proposal arena is unavailable".into())
            })?;
            self.operators
                .cuda_mut()?
                .dspark_proposal_head_device_into(
                    continuation.anchor_token_id,
                    &arena.hc_state,
                    &mut arena.head,
                )?;
            continuation.head_download = Some(
                self.operators
                    .cuda_mut()?
                    .begin_dspark_proposal_head_result_download(&mut arena.head)?,
            );
            self.arm_dspark_proposal_completion(continuation);
            return Ok(DeepSeekV4DsparkProposalStep::Waiting(
                PendingModelProgress::new(continuation.transaction, continuation.id, Vec::new())?,
            ));
        }

        let compact = {
            let arena = continuation.arena.as_mut().ok_or_else(|| {
                Error::Internal("DeepSeek-V4 DSpark proposal arena is unavailable".into())
            })?;
            let download = continuation
                .head_download
                .as_ref()
                .expect("checked DSpark proposal head download");
            self.operators
                .cuda_mut()?
                .poll_dspark_proposal_head_result_download(&mut arena.head, download)?
        };
        let Some(compact) = compact else {
            if !continuation.callback_armed {
                self.arm_dspark_proposal_completion(continuation);
            }
            return Ok(DeepSeekV4DsparkProposalStep::Waiting(
                PendingModelProgress::new(continuation.transaction, continuation.id, Vec::new())?,
            ));
        };
        continuation.head_download = None;
        let (token_ids, confidence_logits) = self
            .operators
            .cuda_mut()?
            .decode_dspark_proposal_head_result(compact)?;
        let proposal = NativeProposal {
            token_ids,
            confidence_logits,
        };
        let source = self.dspark_proposal_source.ok_or_else(|| {
            Error::Model("DeepSeek-V4 has no checkpoint-native proposal capability".into())
        })?;
        proposal.validate_for_source(source)?;

        Ok(DeepSeekV4DsparkProposalStep::Complete(proposal))
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    fn release_dspark_proposal_arena(
        &mut self,
        continuation: &mut DeepSeekV4DsparkProposalContinuation,
    ) -> Result<()> {
        let arena = continuation.arena.take().ok_or_else(|| {
            Error::Internal("DeepSeek-V4 DSpark proposal arena was already released".into())
        })?;
        self.dspark_proposal_arena_pool.push(arena);
        Ok(())
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    fn cleanup_unpublished_dspark_proposal(
        &mut self,
        continuation: &mut DeepSeekV4DsparkProposalContinuation,
        error: Error,
    ) -> Error {
        let mut cleanup_errors = Vec::new();
        if let Some(layer_continuation) = continuation.current_layer.take() {
            let stage = continuation.stage;
            match (
                self.plan
                    .resources()
                    .mtp()
                    .and_then(|mtp| mtp.layers.get(stage)),
                self.expert_residency.as_deref_mut(),
            ) {
                (Some(layer), Some(residency)) => {
                    if let Err(cancel_error) = layer
                        .transformer
                        .cancel_dspark_proposal_block_device_hc_device(
                            layer_continuation,
                            residency,
                            &mut self.operators,
                        )
                    {
                        cleanup_errors.push(format!("stage {stage} MoE cancel: {cancel_error}"));
                    }
                }
                (None, _) => cleanup_errors.push(format!(
                    "current DSpark stage {stage} is outside the prepared attachment"
                )),
                (_, None) => cleanup_errors.push(
                    "runtime expert residency controller is unavailable during cancellation".into(),
                ),
            }
        }
        if let Err(deactivate_error) = self.deactivate_dspark_proposal_binding(continuation) {
            cleanup_errors.push(format!("paged binding deactivation: {deactivate_error}"));
        }
        if let Some(arena) = continuation.arena.take() {
            self.dspark_proposal_arena_pool.push(arena);
        }
        if cleanup_errors.is_empty() {
            error
        } else {
            Error::Execution(format!(
                "DeepSeek-V4 native proposal failed ({error}); cleanup also failed ({})",
                cleanup_errors.join("; ")
            ))
        }
    }

    pub fn dspark_context_kv_lengths(&self) -> Vec<usize> {
        self.sequence
            .dspark_stages
            .iter()
            .map(|state| state.kv.len())
            .collect()
    }

    pub fn prepare_options(&self) -> &DeepSeekV4PrepareOptions {
        self.plan.resources().prepare_options()
    }

    pub fn expert_io_snapshot(&self) -> DeepSeekV4ExpertIoSnapshot {
        let prepared_layers = self.plan.resources().layer_experts();
        let expert_counts = prepared_layers
            .iter()
            .map(|layer| layer.source_bytes().len())
            .collect::<Vec<_>>();
        #[cfg(feature = "cuda")]
        let mut residency = self
            .operators
            .cuda
            .as_ref()
            .map(|cache| cache.expert_io_residency_snapshot(&expert_counts))
            .unwrap_or_else(|| cold_expert_residency(&expert_counts));
        #[cfg(not(feature = "cuda"))]
        let mut residency = cold_expert_residency(&expert_counts);
        let layers = prepared_layers
            .iter()
            .zip(residency.drain(..))
            .map(|(prepared, residency)| {
                DeepSeekV4ExpertIoLayerSnapshot::with_source_order(
                    std::sync::Arc::clone(prepared.source_bytes()),
                    std::sync::Arc::clone(prepared.source_order()),
                    residency,
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        DeepSeekV4ExpertIoSnapshot::new(
            layers,
            self.model()
                .config
                .num_experts_per_tok
                .min(self.model().config.num_routed_experts),
        )
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

    fn ensure_no_packed_cuda_continuation(&self, operation: &str) -> Result<()> {
        #[cfg(feature = "cuda")]
        if !self.packed_cuda_continuations.is_empty() {
            return Err(Error::Execution(format!(
                "cannot {operation} while {} DeepSeek-V4 packed transaction(s) are outstanding",
                self.packed_cuda_continuations.len()
            )));
        }
        let _ = operation;
        Ok(())
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    fn ensure_no_dspark_proposal_continuations(&self, operation: &str) -> Result<()> {
        if !self.dspark_proposal_continuations.is_empty() {
            return Err(Error::Execution(format!(
                "cannot {operation} while {} DeepSeek-V4 native proposal continuation(s) are outstanding",
                self.dspark_proposal_continuations.len()
            )));
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_failpoints(&self) -> Result<&ferrule_cuda::CudaFailpoints> {
        self.operators.cuda_failpoints()
    }

    pub fn operator_runtime_counters(&self) -> DeepSeekV4OperatorRuntimeCounters {
        let mut counters = self.operators.runtime_counters();
        #[cfg(feature = "cuda")]
        if let Some(residency) = self.expert_residency.as_ref() {
            counters.expert_residency_stats = residency.stats();
        }
        let io = self.expert_reader.io_stats();
        counters.expert_io_submitted_extents = io.submitted_extents;
        counters.expert_io_completed_extents = io.completed_extents;
        counters.expert_io_failed_extents = io.failed_extents;
        counters.expert_io_requested_bytes = io.requested_bytes;
        counters.expert_io_aligned_bytes = io.aligned_bytes;
        counters.expert_io_coalesced_slices = io.coalesced_slices;
        counters.expert_io_fixed_file_registrations = io.fixed_file_registrations;
        counters.expert_io_slab_exhaustions = io.slab_exhaustions;
        counters.expert_io_peak_queue_depth = io.peak_queue_depth;
        counters.expert_io_read_us = io.read_us;
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
        let dspark_stages = resources
            .mtp()
            .map(|mtp| {
                mtp.layers
                    .iter()
                    .map(|stage| DeepSeekV4LayerState::new(stage.transformer.attention.config))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Ok(DeepSeekV4SequenceExecutionState::new(
            layers,
            dspark_stages,
            resources.model().config.num_routed_experts,
        ))
    }

    /// Fork the default runner sequence as a runtime-shared paged prefix.
    pub fn fork_sequence_state(&mut self) -> Result<DeepSeekV4SequenceExecutionState> {
        self.ensure_no_packed_cuda_continuation("fork the active sequence")?;
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
            return Err(Error::Execution(format!(
                "DeepSeek-V4 exact fork expected committed position {expected_position}, source is at {}",
                source.position()
            )));
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(source.layers.len())
            .map_err(|error| {
                Error::Model(format!(
                    "DeepSeek-V4 paged fork layer metadata allocation failed: {error}"
                ))
            })?;
        for layer in &source.layers {
            layers.push(layer.fork_paged_prefix_metadata(operators)?);
        }
        let dspark_stages = source
            .dspark_stages
            .iter()
            .map(|stage| stage.fork_paged_prefix_metadata(operators))
            .collect::<Result<Vec<_>>>()?;
        Ok(DeepSeekV4SequenceExecutionState {
            core: source.core.forked()?,
            layers,
            dspark_stages,
            predictor: source.predictor.clone(),
            #[cfg(feature = "cuda")]
            paged_kv_binding: None,
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
            return Err(Error::Model("DeepSeek-V4 runner is shut down".into()));
        }

        if state.max_layers() != self.sequence.max_layers() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 sequence layer count {} does not match runner layer count {}",
                state.max_layers(),
                self.sequence.max_layers()
            )));
        }
        if state.dspark_stage_count() != self.sequence.dspark_stage_count() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 sequence DSpark stage count {} does not match runner stage count {}",
                state.dspark_stage_count(),
                self.sequence.dspark_stage_count()
            )));
        }
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
        state.release_capacity();
        Ok(())
    }

    /// Reset a sequence state for reuse with a new logical sequence.
    pub fn reset_sequence_state(
        &mut self,
        state: &mut DeepSeekV4SequenceExecutionState,
    ) -> Result<()> {
        self.ensure_no_packed_cuda_continuation("reset sequence state")?;
        state.reset_for_reuse();
        Ok(())
    }

    pub fn shutdown(&mut self) -> Result<()> {
        self.ensure_no_packed_cuda_continuation("shut down the runner")?;
        #[cfg(all(feature = "cuda", feature = "cutlass"))]
        self.ensure_no_dspark_proposal_continuations("shut down the runner")?;
        if self.shutdown {
            return Ok(());
        }
        self.sequence.release_capacity();
        self.cpu_expert_runtimes = None;
        #[cfg(feature = "cuda")]
        {
            self.layer_arena_pool.clear();
            #[cfg(feature = "cutlass")]
            {
                self.dspark_proposal_arena_pool.clear();
                self.dspark_proposal_continuations.clear();
            }
            self.operators
                .shutdown(self.expert_residency.as_deref_mut())?;
        }
        #[cfg(not(feature = "cuda"))]
        self.operators.shutdown()?;
        self.shutdown = true;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        if self.shutdown {
            return Err(Error::Model("DeepSeek-V4 runner is shut down".into()));
        }
        self.ensure_no_packed_cuda_continuation("reset the runner sequence")?;
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

    #[cfg(feature = "cuda")]
    pub fn prewarm_predicted_experts(&mut self) -> Result<usize> {
        self.ensure_no_packed_cuda_continuation("prewarm predicted experts")?;
        if self.operators.backend() != ModelExecutionBackend::Cuda {
            return Ok(0);
        }
        let count = self
            .plan
            .resources()
            .prepare_options()
            .moe_prefetch_experts
            .max(self.plan.resources().prepare_options().moe_hotset_experts)
            .min(self.plan.resources().model().config.num_routed_experts);
        if count == 0 {
            return Ok(0);
        }
        let mut warmed = 0usize;
        let residency = self.expert_residency.as_deref_mut().ok_or_else(|| {
            Error::Execution("runtime expert residency controller is not installed".into())
        })?;
        for (layer_idx, prepared) in self.plan.resources().layer_experts().iter().enumerate() {
            let predicted = prepared
                .source_catalog()
                .iter()
                .map(|(expert, _)| expert.expert)
                .take(count)
                .collect::<Vec<_>>();
            if predicted.is_empty() {
                continue;
            }
            warmed = warmed.saturating_add(self.operators.prewarm_experts(
                layer_idx,
                &predicted,
                residency,
                prepared.source_catalog().as_ref(),
                prepared.prefetch_capacity(),
                &self.expert_reader,
            )?);
        }
        Ok(warmed)
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
    fn packed_cuda_cleanup_result(context: &str, errors: Vec<String>) -> Result<()> {
        if errors.is_empty() {
            Ok(())
        } else {
            Err(Error::Execution(format!(
                "{context} cleanup failed ({})",
                errors.join("; ")
            )))
        }
    }

    #[cfg(feature = "cuda")]
    fn packed_cuda_error_with_cleanup(context: &str, error: Error, cleanup: Result<()>) -> Error {
        match cleanup {
            Ok(()) => Error::Execution(format!("{context} failed ({error})")),
            Err(cleanup_error) => Error::Execution(format!(
                "{context} failed ({error}); cleanup also failed ({cleanup_error})"
            )),
        }
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
            return Err(Error::Execution(
                "DeepSeek-V4 packed resources cannot be released while output-head D2H is active"
                    .into(),
            ));
        }
        if continuation.cancel_quiescence.is_some() {
            return Err(Error::Execution(
                "DeepSeek-V4 packed resources cannot be released before cancellation quiesces"
                    .into(),
            ));
        }
        let mut errors = Vec::new();
        if deactivate_before_checkin && continuation.paged_bindings_active {
            match self
                .operators
                .cuda_mut()
                .and_then(|cuda| cuda.activate_paged_binding(continuation.transaction, None))
            {
                Ok(()) => continuation.paged_bindings_active = false,
                Err(error) => errors.push(format!("paged binding deactivation: {error}")),
            }
        }

        if continuation.provisional_checkpoints {
            match self.operators.cuda.as_mut() {
                Some(cuda) => {
                    if let Err(error) =
                        cuda.deactivate_provisional_prefix_checkpoints(continuation.transaction)
                    {
                        errors.push(format!("provisional checkpoint deactivation: {error}"));
                    }
                    if disable_provisional_checkpoints {
                        cuda.discard_provisional_prefix_checkpoints(continuation.transaction);
                    }
                }
                None => errors.push("provisional checkpoint cache is unavailable".into()),
            }
        }

        if let Some(arena_checkout) = continuation.arena_checkout.take() {
            if let Err(error) = self.layer_arena_pool.checkin(arena_checkout) {
                errors.push(format!("arena checkin: {error}"));
            }
        }

        if !deactivate_before_checkin && continuation.paged_bindings_active {
            match self
                .operators
                .cuda_mut()
                .and_then(|cuda| cuda.activate_paged_binding(continuation.transaction, None))
            {
                Ok(()) => continuation.paged_bindings_active = false,
                Err(error) => errors.push(format!("paged binding deactivation: {error}")),
            }
        }

        if self.operators.cuda.is_some() {
            if let Some(decode_buffers) = continuation.decode_buffers.take() {
                self.restore_cuda_decode_buffers(decode_buffers);
            }
            #[cfg(feature = "cutlass")]
            if let Some(dspark) = continuation.dspark_main_buffers.take() {
                self.operators
                    .cuda
                    .as_mut()
                    .expect("checked CUDA cache availability")
                    .restore_dspark_main_buffers(continuation.batch.len(), dspark);
            }
        } else {
            if continuation.decode_buffers.is_some() {
                errors.push("decode buffer cache is unavailable".into());
            }
            #[cfg(feature = "cutlass")]
            if continuation.dspark_main_buffers.is_some() {
                errors.push("DSpark main buffer cache is unavailable".into());
            }
        }

        Self::packed_cuda_cleanup_result("DeepSeek-V4 packed CUDA resource", errors)
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
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "current packed layer {layer_index} is outside the prepared layer set"
                    ))
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
    fn cancel_packed_cuda_continuation_owned(
        &mut self,
        states: &mut [DeepSeekV4SequenceExecutionState],
        mut continuation: DeepSeekV4PackedCudaContinuation,
    ) -> Result<()> {
        match self.poll_packed_cuda_cancel_ready(&mut continuation) {
            Ok(true) => {}
            Ok(false) => {
                let transaction = continuation.transaction;
                let continuation_id = continuation.id;
                self.packed_cuda_continuations
                    .insert(transaction, continuation);
                return Err(Error::Execution(format!(
                    "DeepSeek-V4 packed continuation {} GPU work is still active",
                    continuation_id.get()
                )));
            }
            Err(error) => {
                let transaction = continuation.transaction;
                self.packed_cuda_continuations
                    .insert(transaction, continuation);
                self.completion_hub.notify();
                return Err(error);
            }
        }

        let mut errors = Vec::new();
        if let Some(layer_continuation) = continuation.current_layer.take() {
            let layer_index = continuation.next_layer_index;
            match (
                self.plan.resources().layers().get(layer_index),
                self.expert_residency.as_deref_mut(),
            ) {
                (Some(layer), Some(residency)) => {
                    if let Err(error) = layer.cancel_packed_rows_device_hc_device(
                        layer_continuation,
                        residency,
                        &mut self.operators,
                    ) {
                        errors.push(format!("layer {layer_index} continuation: {error}"));
                    }
                }
                (None, _) => errors.push(format!(
                    "current packed layer {layer_index} is outside the prepared layer set"
                )),
                (_, None) => errors.push(
                    "runtime expert residency controller is not installed during cancellation"
                        .into(),
                ),
            }
        }

        if let Err(error) = self.release_packed_cuda_resources(&mut continuation, true, true) {
            errors.push(error.to_string());
        }
        continuation.moe_access_events.clear();
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
                    None => errors.push(format!(
                        "saved state slot {} is missing during cancellation",
                        sequence.state_index
                    )),
                }
            }
        }
        Self::packed_cuda_cleanup_result("DeepSeek-V4 packed CUDA cancellation", errors)
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
            return Err(Error::Model(
                "DeepSeek-V4 CUDA packed shell does not support this batch shape yet".into(),
            ));
        }
        for (sequence_index, sequence) in metadata.sequences.iter().enumerate() {
            let state = &states[sequence.state_index];
            for (offset, row) in sequence.query.clone().enumerate() {
                let expected = state.position().checked_add(offset).ok_or_else(|| {
                    Error::Model("DeepSeek-V4 packed sequence position overflow".into())
                })?;
                if expected != batch.positions()[row] as usize {
                    return Err(Error::Model(format!(
                        "DeepSeek-V4 packed sequence {sequence_index} row {row} position mismatch: expected={expected} batch={}",
                        batch.positions()[row]
                    )));
                }
            }
            if state.paged_kv_binding.is_none() {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 packed sequence {sequence_index} has no prepared paged binding"
                )));
            }
        }
        if self.expert_residency.is_none() {
            return Err(Error::Execution(
                "runtime expert residency controller is not installed".into(),
            ));
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
            return Err(Error::Model("DeepSeek-V4 packed HC size overflow".into()));
        };
        let Some(hidden_len) = rows.checked_mul(hidden_size) else {
            self.operators
                .cuda_mut()?
                .discard_provisional_prefix_checkpoints(transaction);
            return Err(Error::Model(
                "DeepSeek-V4 packed hidden size overflow".into(),
            ));
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
            #[cfg(feature = "cutlass")]
            dspark_main_buffers: None,
            initialized: false,
            next_layer_index: 0,
            current_layer: None,
            output_head: DeepSeekV4PackedOutputHeadState::NotStarted,
            cancel_quiescence: None,
            moe_access_events: Vec::with_capacity(
                max_layers.saturating_mul(batch.sequences().len()),
            ),
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

        #[cfg(feature = "cutlass")]
        {
            let capture = self.plan.resources().mtp().is_some_and(|mtp| {
                !mtp.config.target_layer_ids.is_empty()
                    && mtp
                        .config
                        .target_layer_ids
                        .iter()
                        .all(|layer| *layer < max_layers)
            });
            if capture {
                continuation.dspark_main_buffers =
                    match self.operators.cuda_mut()?.take_dspark_main_buffers(rows) {
                        Ok(buffers) => Some(buffers),
                        Err(error) => {
                            let cleanup =
                                self.release_packed_cuda_resources(&mut continuation, false, true);
                            return Err(Self::packed_cuda_error_with_cleanup(
                                "DeepSeek-V4 packed DSpark buffer checkout",
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
            return Err(Error::Execution(
                "DeepSeek-V4 packed continuation state slots no longer exist".into(),
            ));
        }
        let current = begin_packed_sequence_steps(states, &continuation.metadata)?;
        if current != continuation.sequence_step_bindings {
            return Err(Error::Execution(
                "DeepSeek-V4 packed continuation sequence step bindings changed while suspended"
                    .into(),
            ));
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn pending_model_progress_for_packed_cuda(
        continuation: &DeepSeekV4PackedCudaContinuation,
    ) -> Result<PendingModelProgress> {
        if matches!(
            continuation.output_head,
            DeepSeekV4PackedOutputHeadState::Downloading(_)
        ) {
            if continuation.current_layer.is_some() {
                return Err(Error::Internal(
                    "DeepSeek-V4 packed continuation cannot wait on a layer and output head simultaneously"
                        .into(),
                ));
            }
            return PendingModelProgress::new(
                continuation.transaction,
                continuation.id,
                Vec::new(),
            );
        }
        let layer_continuation = continuation.current_layer.as_ref().ok_or_else(|| {
            Error::Internal(
                "DeepSeek-V4 waiting packed continuation has no current layer continuation".into(),
            )
        })?;
        let operations = layer_continuation
            .pending_operations()
            .into_iter()
            .map(|operation| {
                let layer = u32::try_from(operation.layer).map_err(|_| {
                    Error::Execution(format!(
                        "DeepSeek-V4 pending expert layer {} exceeds u32 ABI",
                        operation.layer
                    ))
                })?;
                let expert = u32::try_from(operation.expert).map_err(|_| {
                    Error::Execution(format!(
                        "DeepSeek-V4 pending expert {} exceeds u32 ABI",
                        operation.expert
                    ))
                })?;
                PendingExpertLoad::new(operation.operation_id, layer, expert)
            })
            .collect::<Result<Vec<_>>>()?;
        PendingModelProgress::new(continuation.transaction, continuation.id, operations)
    }

    #[cfg(feature = "cuda")]
    fn progress_cuda_packed_batch(
        &mut self,
        states: &mut [DeepSeekV4SequenceExecutionState],
        mut continuation: DeepSeekV4PackedCudaContinuation,
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

        let progress = self.progress_cuda_packed_batch_inner(states, &mut continuation);
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
                Ok(DeepSeekV4PackedCudaProgress::Complete(output))
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
                    Err(deactivate_error) => Error::Execution(format!(
                        "DeepSeek-V4 packed progress failed ({error}); paged binding deactivation also failed ({deactivate_error})"
                    )),
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
    ) -> Result<DeepSeekV4PackedCudaStep> {
        if continuation.failed {
            return Err(Error::Execution(format!(
                "DeepSeek-V4 packed continuation {} previously failed and must be cancelled",
                continuation.id.get()
            )));
        }
        if !continuation.initialized {
            let decode_buffers = continuation.decode_buffers.as_mut().ok_or_else(|| {
                Error::Internal("DeepSeek-V4 packed decode buffers are unavailable".into())
            })?;
            self.operators.cuda_mut()?.resident_embedding_hc_rows_into(
                continuation.batch.token_ids(),
                &mut decode_buffers.hc_input,
            )?;
            #[cfg(feature = "cutlass")]
            if let Some(dspark) = continuation.dspark_main_buffers.as_mut() {
                let positions_i32 = continuation
                    .positions
                    .iter()
                    .map(|position| {
                        i32::try_from(*position).map_err(|_| {
                            Error::Model("DeepSeek-V4 packed DSpark position exceeds i32".into())
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.operators
                    .cuda_mut()?
                    .ops
                    .overwrite_i32_buffer(&positions_i32, &mut dspark.positions)?;
            }
            continuation.initialized = true;
        }
        let max_layers = self.plan.resources().prepare_options().max_layers;
        while continuation.next_layer_index < max_layers {
            let layer_idx = continuation.next_layer_index;
            if continuation.current_layer.is_none() {
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
                            .ok_or_else(|| {
                                Error::Model(format!(
                                    "DeepSeek-V4 state slot {} is referenced more than once",
                                    sequence.state_index
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let arena = continuation
                    .arena_checkout
                    .as_mut()
                    .and_then(|checkout| checkout.get_mut().get_for_layer_mut(layer_idx))
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "DeepSeek-V4 packed layer arena {layer_idx} is unavailable"
                        ))
                    })?;
                let hc_state = &mut continuation
                    .decode_buffers
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Internal("DeepSeek-V4 packed decode buffers are unavailable".into())
                    })?
                    .hc_input;
                let residency = self.expert_residency.as_deref_mut().ok_or_else(|| {
                    Error::Execution("runtime expert residency controller is not installed".into())
                })?;
                let prepared = &self.plan.resources().layer_experts()[layer_idx];
                continuation.current_layer = Some(
                    self.plan.resources().layers()[layer_idx].begin_packed_rows_device_hc_device(
                        &mut layer_states,
                        &continuation.metadata.row_to_sequence,
                        &continuation.metadata.sequence_major_rows,
                        &continuation.sequence_phases,
                        &continuation.paged_bindings,
                        residency,
                        prepared.source_catalog().as_ref(),
                        prepared.prefetch_capacity(),
                        arena,
                        hc_state,
                        continuation.batch.token_ids(),
                        &continuation.positions,
                        &[],
                        &self.expert_reader,
                        &mut self.operators,
                    )?,
                );
            }

            let arena = continuation
                .arena_checkout
                .as_mut()
                .and_then(|checkout| checkout.get_mut().get_for_layer_mut(layer_idx))
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "DeepSeek-V4 packed layer arena {layer_idx} is unavailable"
                    ))
                })?;
            let hc_state = &mut continuation
                .decode_buffers
                .as_mut()
                .ok_or_else(|| {
                    Error::Internal("DeepSeek-V4 packed decode buffers are unavailable".into())
                })?
                .hc_input;
            let residency = self.expert_residency.as_deref_mut().ok_or_else(|| {
                Error::Execution("runtime expert residency controller is not installed".into())
            })?;
            let prepared = &self.plan.resources().layer_experts()[layer_idx];
            let layer_continuation = continuation
                .current_layer
                .take()
                .expect("packed layer continuation was created above");
            match self.plan.resources().layers()[layer_idx].resume_packed_rows_device_hc_device(
                layer_continuation,
                residency,
                prepared.source_catalog().as_ref(),
                arena,
                hc_state,
                &self.expert_reader,
                &mut self.operators,
            )? {
                DeepSeekV4PackedLayerProgress::Waiting(next) => {
                    continuation.current_layer = Some(next);
                    let pending = Self::pending_model_progress_for_packed_cuda(continuation)?;
                    return Ok(DeepSeekV4PackedCudaStep::Waiting(pending));
                }
                DeepSeekV4PackedLayerProgress::Complete(layer_events) => {
                    #[cfg(feature = "cutlass")]
                    if let Some(dspark) = continuation.dspark_main_buffers.as_mut() {
                        self.operators
                            .cuda_mut()?
                            .capture_dspark_target_tap_from_device(
                                layer_idx,
                                hc_state,
                                continuation.batch.len(),
                                &mut dspark.target_taps,
                            )?;
                    }
                    continuation.moe_access_events.extend(layer_events);
                    continuation.next_layer_index += 1;
                }
            }
        }

        if !self.progress_cuda_packed_output_head(states, continuation)? {
            let pending = Self::pending_model_progress_for_packed_cuda(continuation)?;
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
        #[cfg(not(feature = "cutlass"))]
        let _ = states;
        if matches!(
            continuation.output_head,
            DeepSeekV4PackedOutputHeadState::NotStarted
        ) {
            let rows = continuation.batch.len();
            #[cfg(feature = "cutlass")]
            if let Some(dspark) = continuation.dspark_main_buffers.as_mut() {
                self.operators
                    .cuda_mut()?
                    .dspark_main_project_norm_device_into(rows, dspark)?;
                let mtp = self.plan.resources().mtp().ok_or_else(|| {
                    Error::Model("DeepSeek-V4 packed DSpark context has no attachment".into())
                })?;
                let stage_count = mtp.layers.len();
                for sequence in &continuation.metadata.sequences {
                    let state = &states[sequence.state_index];
                    if state.dspark_stages.len() != stage_count {
                        return Err(Error::Model(format!(
                            "DeepSeek-V4 packed DSpark context mismatch for state {}: states={} stages={stage_count}",
                            sequence.state_index,
                            state.dspark_stages.len()
                        )));
                    }
                }
                let max_position =
                    continuation
                        .positions
                        .iter()
                        .copied()
                        .max()
                        .ok_or_else(|| {
                            Error::Model("DeepSeek-V4 packed DSpark positions are empty".into())
                        })?;
                for stage in 0..stage_count {
                    let config = mtp.layers[stage].transformer.attention.config;
                    self.operators
                        .cuda_mut()?
                        .dspark_context_kv_stage_packed_device_into(
                            stage,
                            config,
                            rows,
                            max_position,
                            dspark,
                        )?;
                }
            }

            if continuation.max_top_k == 0 {
                continuation.output_head =
                    DeepSeekV4PackedOutputHeadState::Ready(vec![Vec::new(); rows]);
                return Ok(true);
            }
            let decode_buffers = continuation.decode_buffers.as_mut().ok_or_else(|| {
                Error::Internal("DeepSeek-V4 packed decode buffers are unavailable".into())
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
                return Err(Error::Internal(format!(
                    "DeepSeek-V4 packed output-head row mismatch: expected {rows}, got {}",
                    logits.len()
                )));
            }
            DeepSeekV4PackedOutputHeadState::NotStarted => {
                return Err(Error::Internal(
                    "DeepSeek-V4 packed output publication started before output-head submission"
                        .into(),
                ));
            }
            DeepSeekV4PackedOutputHeadState::Downloading(_) => {
                return Err(Error::Internal(
                    "DeepSeek-V4 packed output publication started while output-head D2H is active"
                        .into(),
                ));
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
            return Err(Error::Internal(
                "DeepSeek-V4 packed MoE event references an unknown sequence".into(),
            ));
        }
        #[cfg(feature = "cutlass")]
        let dspark_stage_count = if continuation.dspark_main_buffers.is_some() {
            Some(
                self.plan
                    .resources()
                    .mtp()
                    .ok_or_else(|| {
                        Error::Model("DeepSeek-V4 packed DSpark context has no attachment".into())
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
        #[cfg(feature = "cutlass")]
        if let Some(stage_count) = dspark_stage_count {
            for sequence in &continuation.metadata.sequences {
                let state = &mut states[sequence.state_index];
                for stage in 0..stage_count {
                    state.dspark_stages[stage]
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
        self.expert_reader.take_completion_reactors()
    }

    fn native_proposal_source(&self) -> Result<Option<NativeProposalSource>> {
        Ok(self.dspark_proposal_source)
    }

    fn begin_native_proposal(
        &mut self,
        transaction: ExecutionTransactionId,
        anchor_token_id: u32,
    ) -> Result<NativeProposalProgress> {
        let source = self.dspark_proposal_source.ok_or_else(|| {
            Error::Model("DeepSeek-V4 has no checkpoint-native proposal capability".into())
        })?;
        #[cfg(all(feature = "cuda", feature = "cutlass"))]
        {
            source.validate()?;
            let mut continuation =
                self.begin_dspark_proposal_continuation(transaction, anchor_token_id)?;
            if let Err(error) = self.activate_dspark_proposal_binding(&mut continuation) {
                return Err(self.cleanup_unpublished_dspark_proposal(&mut continuation, error));
            }
            let progress = self.progress_dspark_proposal(&mut continuation);
            match progress {
                Ok(DeepSeekV4DsparkProposalStep::Waiting(pending)) => {
                    if let Err(error) = self.deactivate_dspark_proposal_binding(&mut continuation) {
                        continuation.failed = Some(format!(
                            "paged binding deactivation after begin failed: {error}"
                        ));
                        self.completion_hub.notify();
                    }
                    let replaced = self
                        .dspark_proposal_continuations
                        .insert(continuation.id, continuation);
                    debug_assert!(replaced.is_none(), "native proposal continuation ID reused");
                    Ok(NativeProposalProgress::Waiting(pending))
                }
                Ok(DeepSeekV4DsparkProposalStep::Complete(proposal)) => {
                    if let Err(error) = self.deactivate_dspark_proposal_binding(&mut continuation) {
                        return Err(
                            self.cleanup_unpublished_dspark_proposal(&mut continuation, error)
                        );
                    }
                    self.release_dspark_proposal_arena(&mut continuation)?;
                    Ok(NativeProposalProgress::Complete(proposal))
                }
                Err(error) => {
                    Err(self.cleanup_unpublished_dspark_proposal(&mut continuation, error))
                }
            }
        }
        #[cfg(not(all(feature = "cuda", feature = "cutlass")))]
        {
            let _ = (transaction, anchor_token_id, source);
            Err(Error::Model(
                "DeepSeek-V4 checkpoint-native proposal requires CUDA + CUTLASS".into(),
            ))
        }
    }

    fn resume_native_proposal(
        &mut self,
        transaction: ExecutionTransactionId,
        continuation_id: BatchContinuationId,
    ) -> Result<NativeProposalProgress> {
        #[cfg(all(feature = "cuda", feature = "cutlass"))]
        {
            let mut continuation = self
                .dspark_proposal_continuations
                .remove(&continuation_id)
                .ok_or_else(|| {
                    Error::Execution(format!(
                        "DeepSeek-V4 has no native proposal continuation {}",
                        continuation_id.get()
                    ))
                })?;
            if continuation.transaction != transaction {
                let owner = continuation.transaction;
                self.dspark_proposal_continuations
                    .insert(continuation_id, continuation);
                return Err(Error::Execution(format!(
                    "DeepSeek-V4 native proposal continuation {} belongs to transaction {}, not {}",
                    continuation_id.get(),
                    owner.get(),
                    transaction.get()
                )));
            }
            if let Some(failed) = continuation.failed.as_deref() {
                let error = Error::Execution(format!(
                    "DeepSeek-V4 native proposal continuation {} previously failed: {failed}",
                    continuation_id.get()
                ));
                self.dspark_proposal_continuations
                    .insert(continuation_id, continuation);
                return Err(error);
            }
            if let Err(error) = self.activate_dspark_proposal_binding(&mut continuation) {
                self.dspark_proposal_continuations
                    .insert(continuation_id, continuation);
                return Err(error);
            }
            let progress = self.progress_dspark_proposal(&mut continuation);
            let deactivate = self.deactivate_dspark_proposal_binding(&mut continuation);
            match (progress, deactivate) {
                (Ok(DeepSeekV4DsparkProposalStep::Waiting(pending)), Ok(())) => {
                    self.dspark_proposal_continuations
                        .insert(continuation_id, continuation);
                    Ok(NativeProposalProgress::Waiting(pending))
                }
                (Ok(DeepSeekV4DsparkProposalStep::Complete(proposal)), Ok(())) => {
                    if let Err(error) = self.release_dspark_proposal_arena(&mut continuation) {
                        let message = error.to_string();
                        continuation.failed = Some(message.clone());
                        self.dspark_proposal_continuations
                            .insert(continuation_id, continuation);
                        return Err(Error::Execution(message));
                    }
                    Ok(NativeProposalProgress::Complete(proposal))
                }
                (progress, deactivate) => {
                    let message = match (progress, deactivate) {
                        (Err(error), Ok(())) => error.to_string(),
                        (Ok(_), Err(error)) => {
                            format!("paged binding deactivation failed: {error}")
                        }
                        (Err(error), Err(deactivate_error)) => format!(
                            "proposal progress failed ({error}); paged binding deactivation also failed ({deactivate_error})"
                        ),
                        (Ok(_), Ok(())) => unreachable!("handled successful proposal progress"),
                    };
                    continuation.failed = Some(message.clone());
                    self.dspark_proposal_continuations
                        .insert(continuation_id, continuation);
                    Err(Error::Execution(message))
                }
            }
        }
        #[cfg(not(all(feature = "cuda", feature = "cutlass")))]
        {
            let _ = (transaction, continuation_id);
            Err(Error::Execution(
                "DeepSeek-V4 was built without CUDA + CUTLASS native proposal continuations".into(),
            ))
        }
    }

    fn cancel_native_proposal(
        &mut self,
        transaction: ExecutionTransactionId,
        continuation_id: BatchContinuationId,
    ) -> BatchContinuationCancelOutcome {
        #[cfg(all(feature = "cuda", feature = "cutlass"))]
        {
            let Some(owner) = self
                .dspark_proposal_continuations
                .get(&continuation_id)
                .map(|continuation| continuation.transaction)
            else {
                return BatchContinuationCancelOutcome::Quiesced(Error::Execution(format!(
                    "DeepSeek-V4 has no native proposal continuation {}",
                    continuation_id.get()
                )));
            };
            if owner != transaction {
                return BatchContinuationCancelOutcome::StillActive(Error::Execution(format!(
                    "DeepSeek-V4 native proposal continuation {} belongs to transaction {}, not {}",
                    continuation_id.get(),
                    owner.get(),
                    transaction.get()
                )));
            }
            let mut continuation = self
                .dspark_proposal_continuations
                .remove(&continuation_id)
                .expect("validated native proposal continuation owner");

            match self.poll_dspark_route_cancel_ready(&mut continuation) {
                Ok(true) => {}
                Ok(false) => {
                    self.dspark_proposal_continuations
                        .insert(continuation_id, continuation);
                    return BatchContinuationCancelOutcome::StillActive(Error::Execution(format!(
                        "DeepSeek-V4 native proposal continuation {} route download is still active",
                        continuation_id.get()
                    )));
                }
                Err(error) => {
                    self.dspark_proposal_continuations
                        .insert(continuation_id, continuation);
                    return BatchContinuationCancelOutcome::StillActive(error);
                }
            }

            if let Some(download) = continuation.head_download.as_ref() {
                let poll = continuation
                    .arena
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Internal("DeepSeek-V4 DSpark proposal arena is unavailable".into())
                    })
                    .and_then(|arena| {
                        self.operators
                            .cuda_mut()?
                            .poll_dspark_proposal_head_result_download(&mut arena.head, download)
                    });
                match poll {
                    Ok(Some(_)) => continuation.head_download = None,
                    Ok(None) => {
                        if !continuation.callback_armed {
                            self.arm_dspark_proposal_completion(&mut continuation);
                        }
                        self.dspark_proposal_continuations
                            .insert(continuation_id, continuation);
                        return BatchContinuationCancelOutcome::StillActive(Error::Execution(
                            format!(
                                "DeepSeek-V4 native proposal continuation {} head download is still active",
                                continuation_id.get()
                            ),
                        ));
                    }
                    Err(error) => {
                        if !continuation.callback_armed {
                            self.arm_dspark_proposal_completion(&mut continuation);
                        }
                        self.dspark_proposal_continuations
                            .insert(continuation_id, continuation);
                        return BatchContinuationCancelOutcome::StillActive(error);
                    }
                }
            }

            let mut errors = Vec::new();
            if continuation.current_layer.is_some() {
                let stage = continuation.stage;
                let Some(layer) = self
                    .plan
                    .resources()
                    .mtp()
                    .and_then(|mtp| mtp.layers.get(stage))
                else {
                    let error = Error::Internal(format!(
                        "current DSpark stage {stage} is outside the prepared attachment"
                    ));
                    self.dspark_proposal_continuations
                        .insert(continuation_id, continuation);
                    return BatchContinuationCancelOutcome::StillActive(error);
                };
                let Some(residency) = self.expert_residency.as_deref_mut() else {
                    let error = Error::Execution(
                        "runtime expert residency controller is unavailable during cancellation"
                            .into(),
                    );
                    self.dspark_proposal_continuations
                        .insert(continuation_id, continuation);
                    return BatchContinuationCancelOutcome::StillActive(error);
                };
                let layer_continuation = continuation
                    .current_layer
                    .take()
                    .expect("checked native proposal layer continuation");
                if let Err(error) = layer
                    .transformer
                    .cancel_dspark_proposal_block_device_hc_device(
                        layer_continuation,
                        residency,
                        &mut self.operators,
                    )
                {
                    errors.push(format!("stage {stage} MoE cancel: {error}"));
                }
            }
            if let Err(error) = self.deactivate_dspark_proposal_binding(&mut continuation) {
                continuation.failed = Some(format!(
                    "paged binding deactivation during cancellation failed: {error}"
                ));
                self.dspark_proposal_continuations
                    .insert(continuation_id, continuation);
                return BatchContinuationCancelOutcome::StillActive(error);
            }
            continuation.moe_access_events.clear();
            if let Err(error) = self.release_dspark_proposal_arena(&mut continuation) {
                errors.push(format!("arena release: {error}"));
            }
            if errors.is_empty() {
                BatchContinuationCancelOutcome::Cancelled
            } else {
                BatchContinuationCancelOutcome::Quiesced(Error::Execution(format!(
                    "DeepSeek-V4 native proposal cancellation failed after quiescence: {}",
                    errors.join("; ")
                )))
            }
        }
        #[cfg(not(all(feature = "cuda", feature = "cutlass")))]
        {
            let _ = (transaction, continuation_id);
            BatchContinuationCancelOutcome::Quiesced(Error::Execution(
                "DeepSeek-V4 was built without CUDA + CUTLASS native proposal continuations".into(),
            ))
        }
    }
}

impl MultiSessionRunner for DeepSeekV4Runner {
    type SequenceState = DeepSeekV4SequenceExecutionState;

    fn expert_residency_requirements(&self) -> Option<ExpertResidencyRequirements> {
        #[cfg(feature = "cuda")]
        if self.operators.backend() == ModelExecutionBackend::Cuda {
            return Some(ExpertResidencyRequirements::new(
                self.model_instance,
                expert_residency_layer_capacities(self.plan.resources()),
            ));
        }
        None
    }

    fn expert_residency_control_installed(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            return self.expert_residency.is_some();
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    fn install_expert_residency_control(
        &mut self,
        control: Box<dyn ExpertResidencyControl>,
    ) -> Result<()> {
        self.ensure_no_packed_cuda_continuation("install expert residency control")?;
        #[cfg(feature = "cuda")]
        {
            let expected = self.expert_residency_requirements().ok_or_else(|| {
                Error::Execution(
                    "DeepSeek-V4 CPU runner does not accept an expert residency controller".into(),
                )
            })?;
            if control.requirements() != expected {
                return Err(Error::Execution(format!(
                    "DeepSeek-V4 expert residency requirements mismatch: expected {:?}, got {:?}",
                    expected,
                    control.requirements()
                )));
            }
            if self.expert_residency.is_some() {
                return Err(Error::Execution(
                    "DeepSeek-V4 expert residency controller is already installed".into(),
                ));
            }
            self.expert_residency = Some(control);
            return Ok(());
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = control;
            Err(Error::Execution(
                "DeepSeek-V4 was built without CUDA expert residency support".into(),
            ))
        }
    }

    fn configure_kv_page_capacity(&mut self, max_pages: usize) -> Result<()> {
        self.ensure_no_packed_cuda_continuation("configure KV page capacity")?;
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
        self.ensure_no_packed_cuda_continuation("release KV pages")?;
        #[cfg(feature = "cuda")]
        if self.operators.cuda.is_some() {
            self.operators.cuda_mut()?.release_kv_pages(pages)?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = pages;
        Ok(())
    }

    fn preempt_kv_pages(&mut self, pages: &[ferrule_common::execution::KvPageId]) -> Result<()> {
        self.ensure_no_packed_cuda_continuation("preempt KV pages")?;
        #[cfg(feature = "cuda")]
        if self.operators.cuda.is_some() {
            self.operators.cuda_mut()?.preempt_kv_pages(pages)?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = pages;
        Ok(())
    }

    fn restore_kv_pages(&mut self, pages: &[ferrule_common::execution::KvPageId]) -> Result<()> {
        self.ensure_no_packed_cuda_continuation("restore KV pages")?;
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
            if self.packed_cuda_continuations.contains_key(&transaction) {
                return Err(Error::Execution(format!(
                    "DeepSeek-V4 transaction {} already owns a packed continuation",
                    transaction.get()
                )));
            }
            if self.operators.backend() != ModelExecutionBackend::Cuda {
                return Err(Error::Execution(
                    "DeepSeek-V4 multi-session prepare requires the CUDA packed backend".into(),
                ));
            }
            if !self
                .operators
                .cuda
                .as_ref()
                .is_some_and(|cuda| cuda.has_kv_page_pool())
            {
                return Err(Error::Execution(
                    "DeepSeek-V4 multi-session prepare requires a configured CUDA KV page pool"
                        .into(),
                ));
            }
            if kv_reservations.len() != batch.sequences().len() {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 paged batch has {} sequences but {} KV reservations",
                    batch.sequences().len(),
                    kv_reservations.len()
                )));
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
            let lowered = (|| -> Result<Vec<_>> {
                batch
                    .sequences()
                    .iter()
                    .map(|sequence| {
                        let state_index = sequence.state_slot.try_as_usize().map_err(|_| {
                            Error::Model("DeepSeek-V4 state slot exceeds usize".into())
                        })?;
                        if state_index >= states.len() {
                            return Err(Error::Model(format!(
                                "DeepSeek-V4 state slot {state_index} is missing during paged prepare"
                            )));
                        }
                        let block_start = usize::try_from(sequence.block_table.start)
                            .map_err(|_| Error::Model("KV block range exceeds usize".into()))?;
                        let block_end = usize::try_from(sequence.block_table.end)
                            .map_err(|_| Error::Model("KV block range exceeds usize".into()))?;
                        let binding = self.operators.cuda_mut()?.lower_paged_binding(
                            transaction,
                            &batch.kv_block_ids()[block_start..block_end],
                            sequence.sequence_len as usize,
                        )?;
                        Ok((state_index, binding))
                    })
                    .collect()
            })();
            let lowered = match lowered {
                Ok(lowered) => lowered,
                Err(error) => {
                    let _ = self.operators.cuda_mut()?.rollback_kv_pages(transaction);
                    return Err(error);
                }
            };
            let previous = lowered
                .iter()
                .map(|(state_index, _)| {
                    (*state_index, states[*state_index].paged_kv_binding.clone())
                })
                .collect::<Vec<_>>();
            for (state_index, binding) in lowered {
                states[state_index].paged_kv_binding = Some(binding);
            }
            let replaced = self.prepared_paged_bindings.insert(transaction, previous);
            if replaced.is_some() {
                let _ = self.operators.cuda_mut()?.rollback_kv_pages(transaction);
                return Err(Error::Execution(format!(
                    "DeepSeek-V4 transaction {} replaced prepared paged bindings",
                    transaction.get()
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (transaction, states, batch, kv_reservations);
            Err(Error::Execution(
                "DeepSeek-V4 multi-session prepare requires CUDA support".into(),
            ))
        }
    }

    fn commit_multi_session_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if self.packed_cuda_continuations.contains_key(&transaction) {
                return Err(Error::Execution(format!(
                    "cannot commit DeepSeek-V4 transaction {} with a live packed continuation",
                    transaction.get()
                )));
            }
            let prepared = self
                .prepared_paged_bindings
                .get(&transaction)
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "DeepSeek-V4 transaction {} has no prepared paged bindings",
                        transaction.get()
                    ))
                })?;
            if prepared
                .iter()
                .any(|(state_index, _)| *state_index >= states.len())
            {
                return Err(Error::Internal(format!(
                    "DeepSeek-V4 transaction {} commit state slice changed",
                    transaction.get()
                )));
            }
            self.operators.cuda_mut()?.commit_kv_pages(transaction)?;
            self.prepared_paged_bindings.remove(&transaction);
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (transaction, states);
            Err(Error::Execution(
                "DeepSeek-V4 multi-session commit requires CUDA support".into(),
            ))
        }
    }

    fn rollback_multi_session_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if self.packed_cuda_continuations.contains_key(&transaction) {
                return Err(Error::Execution(format!(
                    "cannot roll back DeepSeek-V4 transaction {} with a live packed continuation",
                    transaction.get()
                )));
            }
            let previous = self
                .prepared_paged_bindings
                .remove(&transaction)
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "DeepSeek-V4 transaction {} has no prepared paged bindings",
                        transaction.get()
                    ))
                })?;
            if previous
                .iter()
                .any(|(state_index, _)| *state_index >= states.len())
            {
                self.prepared_paged_bindings.insert(transaction, previous);
                return Err(Error::Internal(format!(
                    "DeepSeek-V4 transaction {} rollback state slice changed",
                    transaction.get()
                )));
            }
            let backend = self.operators.cuda_mut()?.rollback_kv_pages(transaction);
            for (state_index, binding) in previous {
                states[state_index].paged_kv_binding = binding;
            }
            backend?;
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (transaction, states);
            Err(Error::Execution(
                "DeepSeek-V4 multi-session rollback requires CUDA support".into(),
            ))
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
        #[cfg(feature = "cuda")]
        if self.packed_cuda_continuations.contains_key(&transaction) {
            return Err(Error::Execution(format!(
                "cannot retain a provisional prefix for transaction {} with a live packed continuation",
                transaction.get()
            )));
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (transaction, sources, branches, executed_rows, retained_rows);
            Err(Error::Execution(
                "DeepSeek-V4 provisional prefix retention requires CUDA support".into(),
            ))
        }

        #[cfg(feature = "cuda")]
        {
            if self.operators.backend() != ModelExecutionBackend::Cuda {
                return Err(Error::Execution(
                    "DeepSeek-V4 provisional prefix retention requires the CUDA packed backend"
                        .into(),
                ));
            }
            let sequence_count = sources.len();
            if sequence_count == 0
                || branches.len() != sequence_count
                || executed_rows.len() != sequence_count
                || retained_rows.len() != sequence_count
            {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 provisional cohort shape mismatch: sources={} branches={} executed={} retained={}",
                    sequence_count,
                    branches.len(),
                    executed_rows.len(),
                    retained_rows.len()
                )));
            }

            for sequence_index in 0..sequence_count {
                let source = &sources[sequence_index];
                let branch = &branches[sequence_index];
                let executed = executed_rows[sequence_index];
                let retained = retained_rows[sequence_index];
                if executed == 0 || retained == 0 || retained > executed {
                    return Err(Error::Model(format!(
                        "invalid DeepSeek-V4 retained prefix for sequence {sequence_index}: retained={retained} executed={executed}"
                    )));
                }
                let expected_branch_position = source
                    .position()
                    .checked_add(executed)
                    .ok_or_else(|| Error::Model("DeepSeek-V4 branch position overflow".into()))?;
                if branch.position() != expected_branch_position
                    || source.layers.len() != branch.layers.len()
                    || source.dspark_stages.len() != branch.dspark_stages.len()
                    || branch.paged_kv_binding.is_none()
                {
                    return Err(Error::Model(format!(
                        "DeepSeek-V4 provisional branch mismatch for sequence {sequence_index}: source_position={} branch_position={} expected={} source_layers={} branch_layers={} source_stages={} branch_stages={} paged_binding={}",
                        source.position(),
                        branch.position(),
                        expected_branch_position,
                        source.layers.len(),
                        branch.layers.len(),
                        source.dspark_stages.len(),
                        branch.dspark_stages.len(),
                        branch.paged_kv_binding.is_some()
                    )));
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
                    return Err(Error::Execution(format!(
                        "DeepSeek-V4 provisional checkpoints do not cover sequence {sequence_index} for transaction {}",
                        transaction.get()
                    )));
                }
            }

            for sequence_index in 0..sequence_count {
                let source = &sources[sequence_index];
                let branch = &mut branches[sequence_index];
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
                for (source_stage, branch_stage) in
                    source.dspark_stages.iter().zip(&mut branch.dspark_stages)
                {
                    branch_stage
                        .kv
                        .restore_uncompressed_prefix_from(&source_stage.kv, retained)?;
                }

                let final_position = source
                    .position()
                    .checked_add(retained)
                    .ok_or_else(|| Error::Model("DeepSeek-V4 retained position overflow".into()))?;
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
        }
    }

    fn execute_multi_session_batch_progress(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        batch: &ExecutionBatch,
    ) -> Result<MultiSessionBatchProgress> {
        #[cfg(feature = "cuda")]
        if self.packed_cuda_continuations.contains_key(&transaction) {
            return Err(Error::Execution(format!(
                "DeepSeek-V4 transaction {} already has a packed continuation",
                transaction.get()
            )));
        }
        #[cfg(feature = "cuda")]
        {
            if self.operators.backend() != ModelExecutionBackend::Cuda {
                return Err(Error::Execution(
                    "DeepSeek-V4 multi-session execution requires the CUDA packed backend".into(),
                ));
            }
            if !self
                .operators
                .cuda
                .as_ref()
                .is_some_and(|cuda| cuda.has_kv_page_pool())
            {
                return Err(Error::Execution(
                    "DeepSeek-V4 multi-session execution requires a configured CUDA KV page pool"
                        .into(),
                ));
            }
            if batch
                .logits()
                .iter()
                .any(|request| matches!(request, LogitsRequest::Full))
            {
                return Err(Error::Execution(
                    "DeepSeek-V4 packed execution does not support full-vocabulary logits".into(),
                ));
            }
            let metadata = PackedBatchMetadata::lower(batch, states.len())?;
            if !metadata.supports_native_cuda() {
                return Err(Error::Execution(
                    "DeepSeek-V4 batch shape is not supported by the CUDA packed executor".into(),
                ));
            }
            let continuation =
                self.begin_cuda_packed_batch_continuation(transaction, states, batch, metadata)?;
            return match self.progress_cuda_packed_batch(states, continuation) {
                Ok(DeepSeekV4PackedCudaProgress::Waiting(continuation, pending)) => {
                    let replaced = self
                        .packed_cuda_continuations
                        .insert(transaction, continuation);
                    debug_assert!(replaced.is_none(), "packed transaction ID reused");
                    Ok(MultiSessionBatchProgress::Waiting(pending))
                }
                Ok(DeepSeekV4PackedCudaProgress::Complete(output)) => {
                    Ok(MultiSessionBatchProgress::Complete(output))
                }
                Err((error, continuation)) => {
                    let continuation_id = continuation.id;
                    let cleanup = self.cancel_packed_cuda_continuation_owned(states, continuation);
                    if self.packed_cuda_continuations.contains_key(&transaction) {
                        self.completion_hub.notify();
                        Ok(MultiSessionBatchProgress::Waiting(
                            PendingModelProgress::new(transaction, continuation_id, Vec::new())?,
                        ))
                    } else {
                        Err(Self::packed_cuda_error_with_cleanup(
                            "DeepSeek-V4 resumable packed CUDA begin",
                            error,
                            cleanup,
                        ))
                    }
                }
            };
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (transaction, states, batch);
            Err(Error::Execution(
                "DeepSeek-V4 multi-session execution requires CUDA support".into(),
            ))
        }
    }

    fn resume_multi_session_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        batch: &ExecutionBatch,
        continuation_id: BatchContinuationId,
    ) -> Result<MultiSessionBatchProgress> {
        #[cfg(feature = "cuda")]
        {
            let continuation = self
                .packed_cuda_continuations
                .remove(&transaction)
                .ok_or_else(|| {
                    Error::Execution(format!(
                        "DeepSeek-V4 transaction {} has no outstanding packed continuation {}",
                        transaction.get(),
                        continuation_id.get()
                    ))
                })?;
            if let Err(error) = validate_packed_cuda_resume_identity(
                continuation.id,
                &continuation.batch,
                continuation_id,
                batch,
            ) {
                self.packed_cuda_continuations
                    .insert(transaction, continuation);
                return Err(error);
            }
            if let Err(error) = Self::validate_packed_cuda_resume_states(states, &continuation) {
                self.packed_cuda_continuations
                    .insert(transaction, continuation);
                return Err(error);
            }
            return match self.progress_cuda_packed_batch(states, continuation) {
                Ok(DeepSeekV4PackedCudaProgress::Waiting(continuation, pending)) => {
                    self.packed_cuda_continuations
                        .insert(transaction, continuation);
                    Ok(MultiSessionBatchProgress::Waiting(pending))
                }
                Ok(DeepSeekV4PackedCudaProgress::Complete(output)) => {
                    Ok(MultiSessionBatchProgress::Complete(output))
                }
                Err((error, continuation)) => {
                    self.packed_cuda_continuations
                        .insert(transaction, continuation);
                    Err(error)
                }
            };
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (transaction, states, batch, continuation_id);
            Err(Error::Execution(
                "DeepSeek-V4 was built without CUDA packed continuation support".into(),
            ))
        }
    }

    fn cancel_multi_session_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        continuation_id: BatchContinuationId,
    ) -> BatchContinuationCancelOutcome {
        #[cfg(feature = "cuda")]
        {
            let Some(expected) = self
                .packed_cuda_continuations
                .get(&transaction)
                .map(|continuation| continuation.id)
            else {
                return BatchContinuationCancelOutcome::Quiesced(Error::Execution(format!(
                    "DeepSeek-V4 transaction {} has no outstanding packed continuation {}",
                    transaction.get(),
                    continuation_id.get()
                )));
            };
            if expected != continuation_id {
                return BatchContinuationCancelOutcome::StillActive(Error::Execution(format!(
                    "DeepSeek-V4 transaction {} continuation mismatch: expected {}, got {}",
                    transaction.get(),
                    expected.get(),
                    continuation_id.get()
                )));
            }
            let continuation = self
                .packed_cuda_continuations
                .remove(&transaction)
                .expect("validated outstanding packed transaction");
            return match self.cancel_packed_cuda_continuation_owned(states, continuation) {
                Ok(()) => BatchContinuationCancelOutcome::Cancelled,
                Err(error) if self.packed_cuda_continuations.contains_key(&transaction) => {
                    BatchContinuationCancelOutcome::StillActive(error)
                }
                Err(error) => BatchContinuationCancelOutcome::Quiesced(error),
            };
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (transaction, states, continuation_id);
            BatchContinuationCancelOutcome::Quiesced(Error::Execution(
                "DeepSeek-V4 was built without CUDA packed continuation support".into(),
            ))
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
        self.ensure_no_packed_cuda_continuation("fork explicit sequence state")?;
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
    use ferrule_common::execution::{
        ExecutionSequence, ForwardMode, ForwardPhase, LogitsRequest, StateSlot,
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
    fn packed_resume_identity_requires_the_exact_id_and_batch() {
        let expected_id = BatchContinuationId::new(7).unwrap();
        let batch = one_row_batch(11, 0);
        validate_packed_cuda_resume_identity(expected_id, &batch, expected_id, &batch).unwrap();

        let wrong_id = validate_packed_cuda_resume_identity(
            expected_id,
            &batch,
            BatchContinuationId::new(8).unwrap(),
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
