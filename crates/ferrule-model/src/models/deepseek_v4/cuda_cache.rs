//! DeepSeek-V4 CUDA operator cache: device-resident weights, KV cache, MoE handles.

#![cfg(feature = "cuda")]

use std::collections::{BTreeSet, HashMap};
#[cfg(feature = "cuda")]
use std::fs::File;
#[cfg(feature = "cuda")]
use std::io::{Read, Seek, SeekFrom};

use std::time::{Duration, Instant};

use ferrule_backend::plan::{ExecutionMode, KernelOperation, ModelKernelPlan};
#[cfg(feature = "cuda")]
use ferrule_backend::plan::{KernelProviderId, LaunchDescriptor};
use ferrule_common::execution::{
    ExecutionTransactionId, ForwardPhase, KvCowReplacement, KvLayoutSchema, KvPageId,
};
use ferrule_common::{CompletionHub, Error, ResidencyLeaseSet, Result};

use crate::attention_backend::SparseAttentionSpec;
#[cfg(feature = "cuda")]
use crate::checkpoint::tensor::{CheckpointDType, CheckpointMatrixSlice};
#[cfg(feature = "cuda")]
use crate::checkpoint::weight::ActivationQuantization;
use crate::checkpoint::weight::{LinearWeight, LinearWeightFormat};
use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::{HyperConnectionConfig, HyperConnectionWeights};
use crate::moe::cuda_materialization::{CudaExpertFrame, CudaSharedExpertSubsystem};
use crate::moe::routing::RouterWeights;

use crate::moe::prediction::{ExpertAccessPhase, ExpertBatchAccessEvent};

use crate::moe::routing::{
    ExpertRoute, ExpertRouterPolicy, RouterScoreFunction, RouterSelectionPolicy,
};
use crate::moe::streaming::{ExpertId, ExpertMemoryPolicy, ExpertStreamingStep};
use crate::runner::{TokenLogit, completion_notify_callback};

use super::attention::DeepSeekV4CompressorPayload;
use super::config::{DeepSeekV4AttentionConfig, DeepSeekV4RopeParams};

use super::helpers::yarn_frequency;
use super::layer::DeepSeekV4Layer;

use super::operators::DeepSeekV4OperatorRuntimeCounters;
use super::prepared::{
    DeepSeekV4ExecutionPolicy, DeepSeekV4KvLayoutSchema, DeepSeekV4PreparedResources,
};
use super::sequence::{DeepSeekV4PagedKvBinding, DeepSeekV4SequenceMoeAccessEvent};

const DSV4_ROPE_TABLE_MIN_CAPACITY: usize = 4096;
const DSV4_EXPERT_TABLE_CAPACITY: usize = 512;

struct CudaRopeTable {
    rope_dim: usize,
    rope: DeepSeekV4RopeParams,
    capacity: usize,
    cos: ferrule_backend::cuda::context::CudaF32Buffer,
    sin: ferrule_backend::cuda::context::CudaF32Buffer,
}

fn validate_rope_table_request(
    name: &str,
    rope_dim: usize,
    rope: DeepSeekV4RopeParams,
    required_positions: usize,
) -> Result<()> {
    if rope_dim == 0 || !rope_dim.is_multiple_of(2) {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 RoPE table '{name}' requires a positive even dimension, got {rope_dim}"
            ),
        });
    }
    if !rope.theta.is_finite() || rope.theta <= 0.0 {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 RoPE table '{name}' requires finite positive theta, got {}",
                rope.theta
            ),
        });
    }
    if !rope.factor.is_finite() || rope.factor <= 0.0 {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 RoPE table '{name}' requires finite positive factor, got {}",
                rope.factor
            ),
        });
    }
    if required_positions == 0 {
        return Err(Error::Model {
            message: format!("DeepSeek-V4 RoPE table '{name}' requires at least one position"),
        });
    }
    let max_addressable_positions = u64::from(u32::MAX) + 1;
    if u64::try_from(required_positions).unwrap_or(u64::MAX) > max_addressable_positions {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 RoPE table '{name}' requires {required_positions} positions, exceeding the CUDA u32 position limit of {max_addressable_positions}"
            ),
        });
    }
    Ok(())
}

fn validate_rope_table_identity(
    name: &str,
    table: &CudaRopeTable,
    rope_dim: usize,
    rope: DeepSeekV4RopeParams,
) -> Result<()> {
    if table.rope_dim != rope_dim || table.rope != rope {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 RoPE table '{name}' identity mismatch: cached dim={} params={:?}, requested dim={rope_dim} params={rope:?}",
                table.rope_dim, table.rope
            ),
        });
    }
    Ok(())
}

fn validate_rope_table_capacity(
    name: &str,
    table: &CudaRopeTable,
    required_positions: usize,
) -> Result<()> {
    if table.capacity < required_positions {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 RoPE table '{name}' capacity {} is smaller than required position count {required_positions}; prepare/grow it before launching CUDA work",
                table.capacity
            ),
        });
    }
    Ok(())
}

fn rope_table_capacity(required_positions: usize) -> Result<usize> {
    required_positions
        .max(DSV4_ROPE_TABLE_MIN_CAPACITY)
        .checked_next_power_of_two()
        .ok_or_else(|| Error::Model {
            message: format!(
                "DeepSeek-V4 RoPE table capacity overflow for {required_positions} positions"
            ),
        })
}

/// Device-resident recurrent compressor state owned by one DeepSeek-V4 layer.
///
/// Historical KV values live exclusively in the runtime-managed page pool.
#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct DeepSeekV4CudaSequenceKvState {
    pub(crate) main_compressor_recurrent:
        Option<ferrule_backend::cuda::CudaCompressorRecurrentState>,
    pub(crate) main_compressor_needs_reset: bool,
    pub(crate) indexer_compressor_recurrent:
        Option<ferrule_backend::cuda::CudaCompressorRecurrentState>,
    pub(crate) indexer_compressor_needs_reset: bool,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4CudaSequenceKvState {
    fn fork_paged_prefix(
        &self,
        operators: &ferrule_backend::cuda::CudaArtifactOperatorContext,
    ) -> Result<Self> {
        Ok(Self {
            main_compressor_recurrent: self
                .main_compressor_recurrent
                .as_ref()
                .map(|state| operators.clone_compressor_recurrent_state(state))
                .transpose()?,
            main_compressor_needs_reset: self.main_compressor_needs_reset,
            indexer_compressor_recurrent: self
                .indexer_compressor_recurrent
                .as_ref()
                .map(|state| operators.clone_compressor_recurrent_state(state))
                .transpose()?,
            indexer_compressor_needs_reset: self.indexer_compressor_needs_reset,
        })
    }

    pub(crate) fn reset_for_reuse(&mut self) {
        self.main_compressor_needs_reset = self.main_compressor_recurrent.is_some();
        self.indexer_compressor_needs_reset = self.indexer_compressor_recurrent.is_some();
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeepSeekV4CudaSequenceKvState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepSeekV4CudaSequenceKvState")
            .field(
                "has_main_compressor_recurrent",
                &self.main_compressor_recurrent.is_some(),
            )
            .field(
                "has_indexer_compressor_recurrent",
                &self.indexer_compressor_recurrent.is_some(),
            )
            .finish()
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeepSeekV4CudaAttentionPrefixMetadata {
    pub(crate) window_len: usize,
    pub(crate) compressed_rows: usize,
    pub(crate) indexer_compressed_rows: usize,
    pub(crate) main_compressor_needs_reset: bool,
    pub(crate) indexer_compressor_needs_reset: bool,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct DeepSeekV4CudaLayerPrefixCheckpoints {
    main: Option<ferrule_backend::cuda::CudaCompressorRecurrentCheckpointSlab>,
    indexer: Option<ferrule_backend::cuda::CudaCompressorRecurrentCheckpointSlab>,
    metadata: Vec<Option<DeepSeekV4CudaAttentionPrefixMetadata>>,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct DeepSeekV4CudaSequencePrefixCheckpoints {
    start_position: usize,
    executed_rows: usize,
    layers: Vec<DeepSeekV4CudaLayerPrefixCheckpoints>,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct DeepSeekV4CudaProvisionalPrefixCheckpoints {
    active: bool,
    sequences: Vec<DeepSeekV4CudaSequencePrefixCheckpoints>,
    row_to_sequence: Vec<usize>,
    row_to_local: Vec<usize>,
}

#[cfg(feature = "cuda")]
fn capture_recurrent_checkpoint(
    operators: &ferrule_backend::cuda::CudaArtifactOperatorContext,
    source: Option<&ferrule_backend::cuda::CudaCompressorRecurrentState>,
    checkpoints: &mut Option<ferrule_backend::cuda::CudaCompressorRecurrentCheckpointSlab>,
    slots: usize,
    slot: usize,
) -> Result<()> {
    let Some(source) = source else {
        *checkpoints = None;
        return Ok(());
    };
    if !checkpoints
        .as_ref()
        .is_some_and(|checkpoints| checkpoints.supports(source, slots))
    {
        *checkpoints = Some(operators.create_compressor_recurrent_checkpoint_slab(source, slots)?);
    }
    operators.capture_compressor_recurrent_checkpoint(
        source,
        checkpoints.as_mut().expect("created above"),
        slot,
    )
}

#[cfg(feature = "cuda")]
fn restore_recurrent_checkpoint(
    operators: &ferrule_backend::cuda::CudaArtifactOperatorContext,
    checkpoints: Option<&ferrule_backend::cuda::CudaCompressorRecurrentCheckpointSlab>,
    slot: usize,
    destination: &mut Option<ferrule_backend::cuda::CudaCompressorRecurrentState>,
) -> Result<()> {
    match (checkpoints, destination.as_mut()) {
        (Some(checkpoints), Some(destination)) => {
            operators.restore_compressor_recurrent_checkpoint(checkpoints, slot, destination)
        }
        (None, None) => Ok(()),
        _ => Err(Error::Model {
            message: "DeepSeek-V4 provisional recurrent checkpoint presence mismatch".into(),
        }),
    }
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct DeepSeekV4CudaMetrics {
    output_head_calls: u64,
    output_head_rows: u64,
    output_head_topk_us: u64,
    moe_router_us: u64,
    moe_routing_us: u64,
    moe_plan_us: u64,
    moe_shared_us: u64,
    moe_workspace_us: u64,
    moe_compute_submit_us: u64,
    moe_commit_us: u64,
    expert_selected: u64,
    expert_unique_selected: u64,
    expert_selected_load_requests: u64,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4CudaOperatorCache {
    pub(crate) ops: ferrule_backend::cuda::context::CudaArtifactOperatorContext,
    completion_hub: CompletionHub,
    profile: bool,
    metrics: DeepSeekV4CudaMetrics,
    managed_experts: bool,
    expert_subsystem: Option<CudaSharedExpertSubsystem>,
    kv_page_pool: Option<ferrule_backend::cuda::CudaKvPagePool>,
    pending_kv_reservations:
        HashMap<ExecutionTransactionId, Vec<ferrule_backend::cuda::KvPoolReservation>>,
    provisional_prefix_checkpoints:
        HashMap<ExecutionTransactionId, DeepSeekV4CudaProvisionalPrefixCheckpoints>,
    active_provisional_prefix_transaction: Option<ExecutionTransactionId>,
    active_paged_kv_owner: Option<ExecutionTransactionId>,
    active_paged_kv: Option<ActivePagedKvBinding>,
    cached_paged_kv: HashMap<(usize, usize, usize, usize), ActivePagedKvBinding>,
    moe_access_events: Vec<ExpertBatchAccessEvent>,
    decode_arena: DeepSeekV4DecodeArena,
    /// Immutable, context-bound model resources compiled before the runner is published.
    execution_image: Option<DeepSeekV4CudaExecutionImage>,
    /// Packed input token IDs reuse one device allocation for each row count.
    #[cfg(feature = "cuda")]
    embedding_token_ids: HashMap<usize, ferrule_backend::cuda::context::CudaI32HostMirror>,
    /// Token ids are persistent by packed batch shape. The host mirror prevents
    /// repeated H2D copies as every routed layer sees the same batch/step ids.
    router_token_ids: HashMap<usize, ferrule_backend::cuda::context::CudaDsv4RouterTokenIds>,
    /// Reusable interleaved i32/f32-bit mirrors. A mirror is removed from this
    /// pool while its control-stream D2H is owned by a model continuation.
    compact_i32_mirrors: HashMap<usize, Vec<ferrule_backend::cuda::context::CudaI32HostMirror>>,
    /// Typed precomputed RoPE tables keyed by their stable layer/resource name.
    /// Each entry records its parameters and growable position capacity so a
    /// same-name shape/configuration mismatch cannot be silently reused.
    rope_tables: HashMap<String, CudaRopeTable>,

    /// Reusable maximum-width provider scratch keyed by packed row count.
    hybrid_mla_explicit_selection_workspaces:
        HashMap<usize, ferrule_backend::cuda::context::CudaHybridMlaExplicitSelectionWorkspace>,
    /// Exact-shape logical/selector scratch for combined-ring paged attention.
    /// DSV4 compression boundaries make adjacent layers alternate between a few
    /// top-k widths; caching by element count avoids synchronizing `cuMemFree`
    /// twice per layer when those widths differ.
    paged_topk_buffers: HashMap<
        usize,
        (
            ferrule_backend::cuda::context::CudaI32Buffer,
            ferrule_backend::cuda::context::CudaI32Buffer,
        ),
    >,
    #[cfg(feature = "cuda")]
    output_head_logits: HashMap<(usize, usize), ferrule_backend::cuda::context::CudaF32Buffer>,
    #[cfg(feature = "cuda")]
    output_head_linear_workspaces:
        HashMap<usize, ferrule_backend::cuda::context::CudaArtifactLinearWorkspace>,
    #[cfg(feature = "cuda")]
    output_head_indices: HashMap<usize, ferrule_backend::cuda::context::CudaI32Buffer>,
    #[cfg(feature = "cuda")]
    output_head_values: HashMap<usize, ferrule_backend::cuda::context::CudaF32Buffer>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeepSeekV4CudaHcStage {
    Attention,
    FeedForward,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeepSeekV4CudaLayerNorm {
    Query,
    KeyValue,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeepSeekV4CudaCompressor {
    Main,
    Indexer,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeepSeekV4CudaLinear {
    #[cfg(feature = "cuda")]
    QueryA,
    QueryB,
    #[cfg(feature = "cuda")]
    KeyValue,
    MainCompressorKv,
    MainCompressorGate,
    IndexerCompressorKv,
    IndexerCompressorGate,
    IndexerQuery,
    IndexerWeights,
    Router,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4CudaLinear {
    const fn operation(self) -> KernelOperation {
        match self {
            #[cfg(feature = "cuda")]
            Self::QueryA => KernelOperation::MlaQueryA,
            Self::QueryB => KernelOperation::MlaQueryB,
            #[cfg(feature = "cuda")]
            Self::KeyValue => KernelOperation::MlaKeyValue,
            Self::MainCompressorKv | Self::MainCompressorGate => {
                KernelOperation::MainCompressorProjection
            }
            Self::IndexerCompressorKv | Self::IndexerCompressorGate => {
                KernelOperation::IndexerCompressorProjection
            }
            Self::IndexerQuery => KernelOperation::IndexerQuery,
            Self::IndexerWeights => KernelOperation::IndexerWeights,
            Self::Router => KernelOperation::Router,
        }
    }
}

#[cfg(feature = "cuda")]
struct HcDeviceWeights {
    function_row_major: ferrule_backend::cuda::context::CudaF32Buffer,
    scale: ferrule_backend::cuda::context::CudaF32Buffer,
    base: ferrule_backend::cuda::context::CudaF32Buffer,
}

#[cfg(feature = "cuda")]
struct DeepSeekV4CudaPreparedLinear {
    handle: ferrule_backend::cuda::context::CudaArtifactLinearHandle,
    #[cfg(feature = "cuda")]
    activation_quantization: Option<ActivationQuantization>,
}

#[cfg(feature = "cuda")]
struct DeepSeekV4CudaPreparedCompressor {
    ape: ferrule_backend::cuda::context::CudaF32Buffer,
    norm: ferrule_backend::cuda::context::CudaF32Buffer,
    kv: DeepSeekV4CudaPreparedLinear,
    gate: DeepSeekV4CudaPreparedLinear,
}

#[cfg(feature = "cuda")]
struct DeepSeekV4CudaPreparedLayer {
    #[cfg(feature = "cuda")]
    query_a: DeepSeekV4CudaPreparedLinear,
    query_b: DeepSeekV4CudaPreparedLinear,
    #[cfg(feature = "cuda")]
    key_value: DeepSeekV4CudaPreparedLinear,
    #[cfg(feature = "cuda")]
    output_a: DeepSeekV4CudaPreparedLinear,
    #[cfg(feature = "cuda")]
    output_b: DeepSeekV4CudaPreparedLinear,
    indexer_query: Option<DeepSeekV4CudaPreparedLinear>,
    indexer_weights: Option<DeepSeekV4CudaPreparedLinear>,
    router: DeepSeekV4CudaPreparedLinear,
    shared_gate: DeepSeekV4CudaPreparedLinear,
    shared_up: DeepSeekV4CudaPreparedLinear,
    shared_down: DeepSeekV4CudaPreparedLinear,
    attention_norm: ferrule_backend::cuda::context::CudaF32Buffer,
    feed_forward_norm: ferrule_backend::cuda::context::CudaF32Buffer,
    query_norm: ferrule_backend::cuda::context::CudaF32Buffer,
    key_value_norm: ferrule_backend::cuda::context::CudaF32Buffer,
    attention_hc: HcDeviceWeights,
    feed_forward_hc: HcDeviceWeights,
    attention_sink: ferrule_backend::cuda::context::CudaF32Buffer,
    main_compressor: Option<DeepSeekV4CudaPreparedCompressor>,
    indexer_compressor: Option<DeepSeekV4CudaPreparedCompressor>,
    router_bias: Option<ferrule_backend::cuda::context::CudaF32Buffer>,
    router_hash_table: Option<ferrule_backend::cuda::context::CudaDsv4RouterHashTable>,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4CudaPreparedLayer {
    fn linear(&self, binding: DeepSeekV4CudaLinear) -> Option<&DeepSeekV4CudaPreparedLinear> {
        match binding {
            #[cfg(feature = "cuda")]
            DeepSeekV4CudaLinear::QueryA => Some(&self.query_a),
            DeepSeekV4CudaLinear::QueryB => Some(&self.query_b),
            #[cfg(feature = "cuda")]
            DeepSeekV4CudaLinear::KeyValue => Some(&self.key_value),
            DeepSeekV4CudaLinear::MainCompressorKv => {
                self.main_compressor.as_ref().map(|value| &value.kv)
            }
            DeepSeekV4CudaLinear::MainCompressorGate => {
                self.main_compressor.as_ref().map(|value| &value.gate)
            }
            DeepSeekV4CudaLinear::IndexerCompressorKv => {
                self.indexer_compressor.as_ref().map(|value| &value.kv)
            }
            DeepSeekV4CudaLinear::IndexerCompressorGate => {
                self.indexer_compressor.as_ref().map(|value| &value.gate)
            }
            DeepSeekV4CudaLinear::IndexerQuery => self.indexer_query.as_ref(),
            DeepSeekV4CudaLinear::IndexerWeights => self.indexer_weights.as_ref(),
            DeepSeekV4CudaLinear::Router => Some(&self.router),
        }
    }
}

/// Context-bound immutable resources for one prepared model generation.
///
/// Runtime caches and routed-expert frames remain outside this image. Stable,
/// typed layer slots are intentionally used here so future fused projection
/// groups and CUDA graphs do not depend on string identities or hash lookups.
#[cfg(feature = "cuda")]
struct DeepSeekV4CudaExecutionImage {
    generation: u64,
    hc_config: HyperConnectionConfig,
    hc_head: HcHeadDeviceWeights,
    output_norm: ferrule_backend::cuda::context::CudaF32Buffer,
    #[cfg(feature = "cuda")]
    embedding: ferrule_backend::cuda::context::CudaArtifactLinearHandle,
    #[cfg(feature = "cuda")]
    output_head: ferrule_backend::cuda::context::CudaArtifactLinearHandle,
    layers: Box<[DeepSeekV4CudaPreparedLayer]>,
    kernel_plan: ModelKernelPlan,
    mtp: Option<DeepSeekV4CudaPreparedProposalAttachment>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DeepSeekV4CudaImageProfile {
    pub globals_us: u64,
    pub embedding_us: u64,
    pub output_head_us: u64,
    pub target_layers_us: u64,
    pub attachment_us: u64,
    pub total_us: u64,
}

/// Immutable Proposal resources are prepared before publication even though the
/// execution methods are connected in a subsequent R1 step. Holding them in the
/// image gives every pointer the same generation and CUDA-context lifetime.
#[allow(dead_code)]
struct DeepSeekV4CudaPreparedProposalAttachment {
    block_size: usize,
    noise_token_id: u32,
    target_layer_ids: Box<[usize]>,
    layers: Box<[DeepSeekV4CudaPreparedProposalStage]>,
    heads: DeepSeekV4CudaPreparedProposalHeads,
    transformer_kernel_plan: ModelKernelPlan,
}

#[allow(dead_code)]
struct DeepSeekV4CudaPreparedProposalStage {
    execution_layer: usize,
    transformer: DeepSeekV4CudaPreparedLayer,
    main_proj: Option<DeepSeekV4CudaPreparedLinear>,
    main_norm: Option<ferrule_backend::cuda::context::CudaF32Buffer>,
}

#[allow(dead_code)]
struct DeepSeekV4CudaPreparedProposalHeads {
    hc_head: HcHeadDeviceWeights,
    norm: ferrule_backend::cuda::context::CudaF32Buffer,
    markov_w1: DeepSeekV4CudaPreparedLinear,
    markov_w2: DeepSeekV4CudaPreparedLinear,
    confidence_proj: DeepSeekV4CudaPreparedLinear,
}

#[cfg(feature = "cuda")]
struct HcHeadDeviceWeights {
    /// HC head has only four rows; its original row-major streams are faster.
    function_row_major: ferrule_backend::cuda::context::CudaF32Buffer,
    scale: ferrule_backend::cuda::context::CudaF32Buffer,
    base: ferrule_backend::cuda::context::CudaF32Buffer,
}

#[cfg(feature = "cuda")]
struct ActivePagedKvBinding {
    physical_block_slots: Vec<i32>,
    block_slots_device: ferrule_backend::cuda::context::CudaI32HostMirror,
    block_offsets_device: ferrule_backend::cuda::context::CudaI32HostMirror,
    kv_len_device: ferrule_backend::cuda::context::CudaI32HostMirror,
    second_kv_len_device: ferrule_backend::cuda::context::CudaI32HostMirror,
    row_sequence_ids_device: ferrule_backend::cuda::context::CudaI32HostMirror,
    page_tokens: usize,
    layer_count: usize,
    sequence_count: usize,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaRouterHashTableIdentity {
    host: Vec<usize>,
    hash_rows: usize,
    hash_cols: usize,
    experts: usize,
    top_k: usize,
}

#[cfg(test)]
impl CudaRouterHashTableIdentity {
    fn new(
        layer: usize,
        table: &[usize],
        hash_rows: usize,
        hash_cols: usize,
        experts: usize,
        top_k: usize,
    ) -> Result<Self> {
        validate_router_hash_table_shape(layer, table, hash_rows, hash_cols)?;
        Ok(Self {
            host: table.to_vec(),
            hash_rows,
            hash_cols,
            experts,
            top_k,
        })
    }

    fn validate_request(
        &self,
        layer: usize,
        table: &[usize],
        hash_rows: usize,
        hash_cols: usize,
        experts: usize,
        top_k: usize,
    ) -> Result<()> {
        validate_router_hash_table_shape(layer, table, hash_rows, hash_cols)?;
        if self.host != table
            || self.hash_rows != hash_rows
            || self.hash_cols != hash_cols
            || self.experts != experts
            || self.top_k != top_k
        {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 layer {layer} router hash table changed after device upload"
                ),
            });
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn validate_router_hash_table_shape(
    layer: usize,
    table: &[usize],
    hash_rows: usize,
    hash_cols: usize,
) -> Result<()> {
    let expected = hash_rows.checked_mul(hash_cols).ok_or_else(|| {
        Error::Model { message: format!(
            "DeepSeek-V4 layer {layer} router hash table shape overflows usize: rows={hash_rows} cols={hash_cols}"
        ) }
    })?;
    if table.len() != expected {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 layer {layer} router hash table shape mismatch: rows={hash_rows} cols={hash_cols} require {expected} entries, got {}",
                table.len()
            ),
        });
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct DeepSeekV4DecodeArena {
    #[cfg(feature = "cuda")]
    proposal_main: HashMap<usize, DeepSeekV4ProposalMainBuffers>,
    hc_inputs: HashMap<usize, ferrule_backend::cuda::context::CudaF32Buffer>,
    final_hiddens: HashMap<usize, ferrule_backend::cuda::context::CudaF32Buffer>,
    final_norms: HashMap<usize, ferrule_backend::cuda::context::CudaF32Buffer>,
    topk_rows: HashMap<usize, ferrule_backend::cuda::context::CudaF32Buffer>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4ProposalMainBuffers {
    pub(crate) target_taps: ferrule_backend::cuda::context::CudaF32Buffer,
    pub(crate) positions: ferrule_backend::cuda::context::CudaI32Buffer,
    activation: ferrule_backend::cuda::context::CudaFp8ActivationPack,
    inv_rms: ferrule_backend::cuda::context::CudaF32Buffer,
    pub(crate) main_x: ferrule_backend::cuda::context::CudaF32Buffer,
    context_kv_raw: ferrule_backend::cuda::context::CudaF32Buffer,
    context_kv: ferrule_backend::cuda::context::CudaF32Buffer,
    context_linear_workspace: ferrule_backend::cuda::context::CudaArtifactLinearWorkspace,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4ProposalAttentionBuffers {
    workspace: ferrule_backend::cuda::context::CudaHybridMlaAttentionWorkspace,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4ProposalHeadBuffers {
    workspace: ferrule_backend::cuda::context::CudaProposalHeadWorkspace,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4DecodeBuffers {
    pub(crate) hc_input: ferrule_backend::cuda::context::CudaF32Buffer,
    pub(crate) final_hidden: ferrule_backend::cuda::context::CudaF32Buffer,
    pub(crate) final_norm: ferrule_backend::cuda::context::CudaF32Buffer,
    pub(crate) topk_row: ferrule_backend::cuda::context::CudaF32Buffer,
}

#[cfg(feature = "cuda")]
struct CudaCompactI32Download {
    compact_len: usize,
    mirror: Option<ferrule_backend::cuda::context::CudaI32HostMirror>,
    download: ferrule_backend::cuda::context::CudaI32HostDownload,
    callback_armed: bool,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4OutputHeadTopKDownload {
    rows: usize,
    top_k: usize,
    vocab: usize,
    compact: CudaCompactI32Download,
    topk_start: Option<Instant>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4CudaComputeQuiescence {
    event: ferrule_backend::cuda::context::CudaComputeEvent,
    callback_armed: bool,
}

#[cfg(feature = "cuda")]
enum CudaPackedMoeRouteState {
    RoutesPending(CudaCompactI32Download),
    RoutesReady,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CudaCompactDownloadPollAction {
    Wait,
    ArmAndWait,
    Consume,
}

#[cfg(feature = "cuda")]
const fn compact_download_poll_action(
    ready: bool,
    callback_armed: bool,
) -> CudaCompactDownloadPollAction {
    if ready {
        CudaCompactDownloadPollAction::Consume
    } else if callback_armed {
        CudaCompactDownloadPollAction::Wait
    } else {
        CudaCompactDownloadPollAction::ArmAndWait
    }
}

#[cfg(feature = "cuda")]
fn decode_compact_router_routes(
    compact: &[i32],
    tokens: usize,
    top_k: usize,
) -> Result<Vec<Vec<ExpertRoute>>> {
    let route_count = tokens.checked_mul(top_k).ok_or_else(|| Error::Internal {
        message: "CUDA router route count overflow".into(),
    })?;
    let expected = route_count.checked_mul(2).ok_or_else(|| Error::Internal {
        message: "CUDA compact route size overflow".into(),
    })?;
    if compact.len() != expected {
        return Err(Error::Internal {
            message: format!(
                "CUDA compact route length mismatch: got {} expected {expected}",
                compact.len()
            ),
        });
    }

    let mut routes_by_token = Vec::with_capacity(tokens);
    for token in 0..tokens {
        let mut routes = Vec::with_capacity(top_k);
        for slot in 0..top_k {
            let index = token * top_k + slot;
            let expert = usize::try_from(compact[index * 2]).map_err(|_| Error::Internal {
                message: format!(
                    "CUDA compact route contains negative expert index {} at route {index}",
                    compact[index * 2]
                ),
            })?;
            routes.push(ExpertRoute {
                expert,
                weight: f32::from_bits(compact[index * 2 + 1] as u32),
                score: 0.0,
                selection_score: 0.0,
            });
        }
        routes_by_token.push(routes);
    }
    Ok(routes_by_token)
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeepSeekV4CudaPackedMoePendingExpert {
    pub(crate) layer: usize,
    pub(crate) expert: usize,
}

#[cfg(feature = "cuda")]
struct CudaPackedMoeChunk {
    selected: Vec<usize>,
    unresolved_experts: Vec<usize>,
    streaming: ExpertStreamingStep,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4CudaPackedMoeContinuation {
    layer: usize,
    tokens: usize,
    hidden_size: usize,
    routes_per_token: usize,
    route_count: usize,
    expected_route_output: usize,
    route_state: CudaPackedMoeRouteState,
    routes_by_token: Vec<Vec<ExpertRoute>>,
    unique_experts: Vec<usize>,
    sequence_phases: Option<Vec<ExpertAccessPhase>>,
    row_to_sequence: Option<Vec<usize>>,
    swiglu_limit: f32,
    resident_slot_capacity: Option<usize>,
    route_plan_capacity: usize,
    next_chunk_start: usize,
    current_chunk: Option<CudaPackedMoeChunk>,
    retained_leases: Vec<ResidencyLeaseSet>,
    input_prepared: bool,
    expected_intermediate: Option<usize>,
    streaming_steps: Vec<ExpertStreamingStep>,
    compute_start: Option<Instant>,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4CudaPackedMoeContinuation {
    pub(crate) fn routes_pending(&self) -> bool {
        matches!(&self.route_state, CudaPackedMoeRouteState::RoutesPending(_))
    }

    pub(crate) fn pending_experts(&self) -> Vec<DeepSeekV4CudaPackedMoePendingExpert> {
        self.current_chunk
            .as_ref()
            .into_iter()
            .flat_map(|chunk| chunk.unresolved_experts.iter().copied())
            .map(|expert| DeepSeekV4CudaPackedMoePendingExpert {
                layer: self.layer,
                expert,
            })
            .collect()
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::large_enum_variant)]
pub(crate) enum DeepSeekV4CudaPackedMoeProgress {
    Waiting(DeepSeekV4CudaPackedMoeContinuation),
    Complete {
        events: Vec<DeepSeekV4SequenceMoeAccessEvent>,
        leases: Vec<ResidencyLeaseSet>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CudaPackedMoeResumeAction {
    PollSubmittedRoute,
    SubmitExpertWindow,
}

fn packed_moe_resume_action(
    routes_pending: bool,
    expert_window_pending: bool,
    lease_count: usize,
) -> Result<CudaPackedMoeResumeAction> {
    if routes_pending {
        if lease_count != 0 {
            return Err(Error::Execution { message:
                "DeepSeek-V4 packed MoE received expert leases before its route dependency completed"
                    .into(),
             });
        }
        return Ok(CudaPackedMoeResumeAction::PollSubmittedRoute);
    }
    if !expert_window_pending {
        return Err(Error::Execution {
            message:
                "DeepSeek-V4 packed MoE has no declared expert dependency; refusing resume replay"
                    .into(),
        });
    }
    Ok(CudaPackedMoeResumeAction::SubmitExpertWindow)
}

fn duration_us(d: Duration) -> u64 {
    d.as_micros().min(u128::from(u64::MAX)) as u64
}

#[inline]
fn profile_start(enabled: bool) -> Option<Instant> {
    enabled.then(Instant::now)
}

#[inline]
fn record_profile_duration(stat: &mut u64, start: Option<Instant>) {
    if let Some(start) = start {
        *stat = stat.saturating_add(duration_us(start.elapsed()));
    }
}

fn decode_output_head_topk_rows(
    compact: &[i32],
    batch_rows: usize,
    top_k: usize,
    vocab: usize,
) -> Result<Vec<Vec<TokenLogit>>> {
    if top_k == 0 {
        if compact.is_empty() {
            return Ok((0..batch_rows).map(|_| Vec::new()).collect());
        }
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 output-head compact result must be empty for k=0, got {} values",
                compact.len()
            ),
        });
    }
    let expected_pairs = batch_rows.checked_mul(top_k).ok_or_else(|| Error::Model {
        message: "DeepSeek-V4 output-head top-k size overflow".into(),
    })?;
    let expected_len = expected_pairs.checked_mul(2).ok_or_else(|| Error::Model {
        message: "DeepSeek-V4 compact output-head size overflow".into(),
    })?;
    if compact.len() != expected_len {
        return Err(Error::Model {
            message: format!(
                "DeepSeek-V4 output-head compact result mismatch: rows={batch_rows} k={top_k} expected={expected_len} got={}",
                compact.len()
            ),
        });
    }

    compact
        .chunks_exact(top_k * 2)
        .map(|row| {
            row.chunks_exact(2)
                .map(|pair| {
                    let token = usize::try_from(pair[0]).map_err(|_| Error::Model {
                        message: format!(
                            "DeepSeek-V4 output-head returned negative token index {}",
                            pair[0]
                        ),
                    })?;
                    if token >= vocab {
                        return Err(Error::Model {
                            message: format!(
                                "DeepSeek-V4 output-head token index {token} exceeds vocab {vocab}"
                            ),
                        });
                    }
                    Ok(TokenLogit {
                        token_id: u32::try_from(token).map_err(|_| Error::Model {
                            message: format!(
                                "DeepSeek-V4 output-head token index {token} exceeds u32"
                            ),
                        })?,
                        logit: f32::from_bits(pair[1] as u32),
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect()
}

#[cfg(feature = "cuda")]
impl DeepSeekV4CudaOperatorCache {
    pub(crate) fn pinned_host_allocator(
        &self,
    ) -> ferrule_backend::cuda::context::CudaPinnedHostAllocator {
        self.ops.pinned_host_allocator()
    }

    pub(crate) fn memory_info(&self) -> Result<(usize, usize)> {
        self.ops.memory_info()
    }

    pub(crate) fn compute_stream_authority(
        &self,
    ) -> ferrule_backend::cuda::context::CudaComputeStreamAuthority {
        self.ops.compute_stream_authority()
    }

    #[cfg(test)]
    pub(crate) fn new(
        policy: &DeepSeekV4ExecutionPolicy,
        expert_memory_policy: ExpertMemoryPolicy,
    ) -> Result<Self> {
        Self::new_with_completion_hub(policy, expert_memory_policy, CompletionHub::new())
    }

    pub(super) fn new_with_completion_hub(
        policy: &DeepSeekV4ExecutionPolicy,
        _expert_memory_policy: ExpertMemoryPolicy,
        completion_hub: CompletionHub,
    ) -> Result<Self> {
        Ok(Self {
            ops: ferrule_backend::cuda::context::CudaArtifactOperatorContext::new()?,
            completion_hub,
            profile: policy.profile_enabled(),
            metrics: DeepSeekV4CudaMetrics::default(),
            managed_experts: policy.managed_experts(),
            expert_subsystem: None,
            kv_page_pool: None,
            pending_kv_reservations: HashMap::new(),
            provisional_prefix_checkpoints: HashMap::new(),
            active_provisional_prefix_transaction: None,
            active_paged_kv_owner: None,
            active_paged_kv: None,
            cached_paged_kv: HashMap::new(),
            moe_access_events: Vec::new(),
            decode_arena: DeepSeekV4DecodeArena::default(),
            execution_image: None,
            #[cfg(feature = "cuda")]
            embedding_token_ids: HashMap::new(),
            router_token_ids: HashMap::new(),
            compact_i32_mirrors: HashMap::new(),
            rope_tables: HashMap::new(),

            hybrid_mla_explicit_selection_workspaces: HashMap::new(),
            paged_topk_buffers: HashMap::new(),
            #[cfg(feature = "cuda")]
            output_head_logits: HashMap::new(),
            #[cfg(feature = "cuda")]
            output_head_linear_workspaces: HashMap::new(),
            #[cfg(feature = "cuda")]
            output_head_indices: HashMap::new(),
            #[cfg(feature = "cuda")]
            output_head_values: HashMap::new(),
        })
    }

    pub(crate) fn configure_kv_page_pool(
        &mut self,
        schema: &DeepSeekV4KvLayoutSchema,
        max_pages: usize,
    ) -> Result<()> {
        if !self.pending_kv_reservations.is_empty() {
            return Err(Error::Model {
                message: "cannot reconfigure DeepSeek-V4 KV pool with a pending batch".into(),
            });
        }
        let data_planes = schema.planes().get(..3).ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 KV schema is missing token-scaled data planes".into(),
        })?;
        self.kv_page_pool = Some(ferrule_backend::cuda::CudaKvPagePool::new(
            &self.ops,
            data_planes,
            schema.page_size(),
            max_pages,
        )?);
        Ok(())
    }

    pub(crate) fn has_kv_page_pool(&self) -> bool {
        self.kv_page_pool.is_some()
    }

    pub(crate) fn fork_sequence_kv_state(
        &self,
        source: &DeepSeekV4CudaSequenceKvState,
    ) -> Result<DeepSeekV4CudaSequenceKvState> {
        source.fork_paged_prefix(&self.ops)
    }

    pub(crate) fn begin_provisional_prefix_checkpoints(
        &mut self,
        transaction: ExecutionTransactionId,
        sequence_shapes: &[(usize, usize)],
        layer_count: usize,
    ) -> Result<()> {
        if sequence_shapes.is_empty()
            || layer_count == 0
            || sequence_shapes.iter().any(|(_, rows)| *rows == 0)
        {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 provisional checkpoint shape is empty: sequences={} layers={layer_count}",
                    sequence_shapes.len()
                ),
            });
        }
        let total_rows = sequence_shapes
            .iter()
            .try_fold(0usize, |total, (_, rows)| {
                total.checked_add(*rows).ok_or_else(|| Error::Model {
                    message: "DeepSeek-V4 provisional checkpoint row count overflow".into(),
                })
            })?;
        if self
            .provisional_prefix_checkpoints
            .contains_key(&transaction)
        {
            return Err(Error::Execution {
                message: format!(
                    "DeepSeek-V4 transaction {} already has provisional checkpoints",
                    transaction.get()
                ),
            });
        }
        let mut checkpoints = DeepSeekV4CudaProvisionalPrefixCheckpoints::default();
        checkpoints.active = true;
        checkpoints.sequences.resize_with(
            sequence_shapes.len(),
            DeepSeekV4CudaSequencePrefixCheckpoints::default,
        );
        checkpoints.sequences.truncate(sequence_shapes.len());
        checkpoints.row_to_sequence.clear();
        checkpoints.row_to_sequence.reserve(total_rows);
        checkpoints.row_to_local.clear();
        checkpoints.row_to_local.reserve(total_rows);

        for (sequence_index, ((start_position, executed_rows), sequence)) in sequence_shapes
            .iter()
            .copied()
            .zip(&mut checkpoints.sequences)
            .enumerate()
        {
            sequence.start_position = start_position;
            sequence.executed_rows = executed_rows;
            if sequence.layers.len() < layer_count {
                sequence
                    .layers
                    .resize_with(layer_count, DeepSeekV4CudaLayerPrefixCheckpoints::default);
            }
            let checkpoint_slots = executed_rows - 1;
            for layer in sequence.layers.iter_mut().take(layer_count) {
                layer.metadata.clear();
                layer.metadata.resize(checkpoint_slots, None);
            }
            checkpoints
                .row_to_sequence
                .extend(std::iter::repeat_n(sequence_index, executed_rows));
            checkpoints.row_to_local.extend(0..executed_rows);
        }
        self.provisional_prefix_checkpoints
            .insert(transaction, checkpoints);
        self.active_provisional_prefix_transaction = Some(transaction);
        Ok(())
    }

    pub(crate) fn activate_provisional_prefix_checkpoints(
        &mut self,
        transaction: ExecutionTransactionId,
    ) -> Result<()> {
        if !self
            .provisional_prefix_checkpoints
            .contains_key(&transaction)
        {
            return Err(Error::Execution {
                message: format!(
                    "DeepSeek-V4 transaction {} has no provisional checkpoints",
                    transaction.get()
                ),
            });
        }
        self.active_provisional_prefix_transaction = Some(transaction);
        Ok(())
    }

    pub(crate) fn deactivate_provisional_prefix_checkpoints(
        &mut self,
        transaction: ExecutionTransactionId,
    ) -> Result<()> {
        match self.active_provisional_prefix_transaction {
            Some(owner) if owner == transaction => {
                self.active_provisional_prefix_transaction = None;
                Ok(())
            }
            Some(owner) => Err(Error::Execution {
                message: format!(
                    "cannot deactivate provisional transaction {}; active owner is {}",
                    transaction.get(),
                    owner.get()
                ),
            }),
            None => Ok(()),
        }
    }

    pub(crate) fn discard_provisional_prefix_checkpoints(
        &mut self,
        transaction: ExecutionTransactionId,
    ) {
        self.provisional_prefix_checkpoints.remove(&transaction);
        if self.active_provisional_prefix_transaction == Some(transaction) {
            self.active_provisional_prefix_transaction = None;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_provisional_prefix_checkpoint(
        &mut self,
        layer: usize,
        row: usize,
        state: &DeepSeekV4CudaSequenceKvState,
        window_len: usize,
        compressed_rows: usize,
        indexer_compressed_rows: usize,
    ) -> Result<()> {
        let Some(transaction) = self.active_provisional_prefix_transaction else {
            return Ok(());
        };
        let checkpoints = self
            .provisional_prefix_checkpoints
            .get_mut(&transaction)
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "active provisional transaction {} has no checkpoint state",
                    transaction.get()
                ),
            })?;
        let sequence_index = checkpoints
            .row_to_sequence
            .get(row)
            .copied()
            .ok_or_else(|| Error::Model {
                message: format!(
                    "DeepSeek-V4 provisional checkpoint row {row} is outside the packed cohort"
                ),
            })?;
        let local_row = checkpoints.row_to_local[row];
        let sequence = checkpoints
            .sequences
            .get_mut(sequence_index)
            .ok_or_else(|| Error::Model {
                message: format!(
                    "DeepSeek-V4 provisional checkpoint sequence {sequence_index} is missing"
                ),
            })?;
        if local_row + 1 >= sequence.executed_rows {
            return Ok(());
        }
        let checkpoint_slots = sequence.executed_rows - 1;
        let layer_checkpoints = sequence.layers.get_mut(layer).ok_or_else(|| {
            Error::Model { message: format!(
                "DeepSeek-V4 provisional checkpoint layer {layer} is not prepared for sequence {sequence_index}"
            ) }
        })?;
        capture_recurrent_checkpoint(
            &self.ops,
            state.main_compressor_recurrent.as_ref(),
            &mut layer_checkpoints.main,
            checkpoint_slots,
            local_row,
        )?;
        capture_recurrent_checkpoint(
            &self.ops,
            state.indexer_compressor_recurrent.as_ref(),
            &mut layer_checkpoints.indexer,
            checkpoint_slots,
            local_row,
        )?;
        layer_checkpoints.metadata[local_row] = Some(DeepSeekV4CudaAttentionPrefixMetadata {
            window_len,
            compressed_rows,
            indexer_compressed_rows,
            main_compressor_needs_reset: state.main_compressor_needs_reset,
            indexer_compressor_needs_reset: state.indexer_compressor_needs_reset,
        });
        Ok(())
    }

    pub(crate) fn provisional_prefix_matches(
        &self,
        transaction: ExecutionTransactionId,
        sequence_index: usize,
        start_position: usize,
        executed_rows: usize,
        layer_count: usize,
    ) -> bool {
        self.provisional_prefix_checkpoints
            .get(&transaction)
            .is_some_and(|checkpoints| {
                checkpoints.active
                    && checkpoints
                        .sequences
                        .get(sequence_index)
                        .is_some_and(|sequence| {
                            sequence.start_position == start_position
                                && sequence.executed_rows == executed_rows
                                && sequence.layers.len() >= layer_count
                        })
            })
    }

    pub(crate) fn restore_provisional_prefix_checkpoint(
        &self,
        transaction: ExecutionTransactionId,
        sequence_index: usize,
        layer: usize,
        retained_rows: usize,
        state: &mut DeepSeekV4CudaSequenceKvState,
    ) -> Result<Option<DeepSeekV4CudaAttentionPrefixMetadata>> {
        let checkpoints = self
            .provisional_prefix_checkpoints
            .get(&transaction)
            .ok_or_else(|| Error::Model {
                message: format!(
                    "DeepSeek-V4 transaction {} has no provisional checkpoints",
                    transaction.get()
                ),
            })?;
        let sequence = checkpoints
            .sequences
            .get(sequence_index)
            .ok_or_else(|| Error::Model {
                message: format!(
                    "DeepSeek-V4 provisional checkpoint sequence {sequence_index} is missing"
                ),
            })?;
        if !checkpoints.active || retained_rows == 0 || retained_rows >= sequence.executed_rows {
            return Err(Error::Model {
                message: format!(
                    "invalid DeepSeek-V4 provisional prefix restore: sequence={sequence_index} active={} retained={retained_rows} executed={}",
                    checkpoints.active, sequence.executed_rows
                ),
            });
        }
        let slot = retained_rows - 1;
        let layer_checkpoints = sequence.layers.get(layer).ok_or_else(|| {
            Error::Model { message: format!(
                "DeepSeek-V4 provisional checkpoint layer {layer} is missing for sequence {sequence_index}"
            ) }
        })?;
        let Some(metadata) = layer_checkpoints
            .metadata
            .get(slot)
            .and_then(|metadata| *metadata)
        else {
            return Ok(None);
        };
        restore_recurrent_checkpoint(
            &self.ops,
            layer_checkpoints.main.as_ref(),
            slot,
            &mut state.main_compressor_recurrent,
        )?;
        restore_recurrent_checkpoint(
            &self.ops,
            layer_checkpoints.indexer.as_ref(),
            slot,
            &mut state.indexer_compressor_recurrent,
        )?;
        state.main_compressor_needs_reset = metadata.main_compressor_needs_reset;
        state.indexer_compressor_needs_reset = metadata.indexer_compressor_needs_reset;
        Ok(Some(metadata))
    }

    pub(crate) fn prepare_kv_pages(
        &mut self,
        transaction: ExecutionTransactionId,
        reservations: &[(Vec<KvPageId>, Option<KvCowReplacement>)],
    ) -> Result<()> {
        if self.pending_kv_reservations.contains_key(&transaction) {
            return Err(Error::Execution {
                message: format!(
                    "DeepSeek-V4 CUDA transaction {} is already prepared",
                    transaction.get()
                ),
            });
        }
        let pool = self.kv_page_pool.as_mut().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA physical KV pool is not configured".into(),
        })?;
        let mut prepared = Vec::with_capacity(reservations.len());
        for (pages, cow) in reservations {
            match pool.reserve(&self.ops, pages, *cow) {
                Ok(reservation) => prepared.push(reservation),
                Err(error) => {
                    for reservation in prepared {
                        let _ = pool.rollback(&self.ops, reservation);
                    }
                    return Err(error);
                }
            }
        }
        self.pending_kv_reservations.insert(transaction, prepared);
        Ok(())
    }

    pub(crate) fn lower_paged_binding(
        &self,
        transaction: ExecutionTransactionId,
        block_ids: &[ferrule_common::execution::KvBlockId],
        sequence_len: usize,
    ) -> Result<DeepSeekV4PagedKvBinding> {
        let pool = self.kv_page_pool.as_ref().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA physical KV pool is not configured".into(),
        })?;
        let mut physical_block_slots = Vec::with_capacity(block_ids.len());
        for block in block_ids {
            let page = KvPageId(block.get());
            let slot = pool.physical_slot(page).or_else(|| {
                self.pending_kv_reservations
                    .get(&transaction)
                    .and_then(|reservations| {
                        reservations
                            .iter()
                            .find_map(|reservation| pool.pending_slot(reservation, page))
                    })
            });
            let slot = slot.ok_or_else(|| Error::Model {
                message: format!(
                    "DeepSeek-V4 KV page {} has no committed or provisional physical slot",
                    page.0
                ),
            })?;
            physical_block_slots.push(i32::try_from(slot).map_err(|_| Error::Model {
                message: "DeepSeek-V4 physical KV slot exceeds i32 ABI".into(),
            })?);
        }
        Ok(DeepSeekV4PagedKvBinding {
            physical_block_slots,
            sequence_len,
            page_tokens: pool.page_tokens(),
            layer_count: pool.planes().first().map_or(0, |plane| plane.layer_count),
        })
    }

    pub(crate) fn activate_paged_binding(
        &mut self,
        transaction: ExecutionTransactionId,
        binding: Option<&DeepSeekV4PagedKvBinding>,
    ) -> Result<()> {
        match binding {
            Some(binding) => self.activate_paged_bindings(transaction, &[binding]),
            None => {
                if let Some(owner) = self.active_paged_kv_owner
                    && owner != transaction
                {
                    return Err(Error::Execution {
                        message: format!(
                            "cannot deactivate paged binding for transaction {}; active owner is {}",
                            transaction.get(),
                            owner.get()
                        ),
                    });
                }
                self.active_paged_kv_owner = None;
                if let Some(active) = self.active_paged_kv.take() {
                    let shape = (
                        active.block_slots_device.len(),
                        active.block_offsets_device.len(),
                        active.kv_len_device.len(),
                        active.row_sequence_ids_device.len(),
                    );
                    self.cached_paged_kv.insert(shape, active);
                }
                Ok(())
            }
        }
    }

    /// Activate one flattened ragged page table with an identity row selector.
    pub(crate) fn activate_paged_bindings(
        &mut self,
        transaction: ExecutionTransactionId,
        bindings: &[&DeepSeekV4PagedKvBinding],
    ) -> Result<()> {
        let row_sequence_ids = (0..bindings.len()).collect::<Vec<_>>();
        self.activate_paged_bindings_for_rows(transaction, bindings, &row_sequence_ids)
    }

    /// Activate sequence-owned page tables plus an independent packed-row selector.
    pub(crate) fn activate_paged_bindings_for_rows(
        &mut self,
        transaction: ExecutionTransactionId,
        bindings: &[&DeepSeekV4PagedKvBinding],
        row_sequence_ids: &[usize],
    ) -> Result<()> {
        if bindings.is_empty() {
            return self.activate_paged_binding(transaction, None);
        }
        if let Some(owner) = self.active_paged_kv_owner
            && owner != transaction
        {
            return Err(Error::Execution {
                message: format!(
                    "cannot activate paged transaction {}; active owner is {}",
                    transaction.get(),
                    owner.get()
                ),
            });
        }
        let page_tokens = bindings[0].page_tokens;
        let layer_count = bindings[0].layer_count;
        if page_tokens == 0 || layer_count == 0 {
            return Err(Error::Model {
                message: "DeepSeek-V4 paged binding has invalid page/layer metadata".into(),
            });
        }

        let total_blocks = bindings.iter().try_fold(0usize, |total, binding| {
            if binding.page_tokens != page_tokens || binding.layer_count != layer_count {
                return Err(Error::Model {
                    message: "DeepSeek-V4 packed paged bindings use incompatible layouts".into(),
                });
            }
            total
                .checked_add(binding.physical_block_slots.len())
                .ok_or_else(|| Error::Model {
                    message: "DeepSeek-V4 packed block table overflow".into(),
                })
        })?;
        let mut physical_block_slots = Vec::with_capacity(total_blocks);
        let mut block_offsets = Vec::with_capacity(bindings.len() + 1);
        let mut kv_lens = Vec::with_capacity(bindings.len());
        block_offsets.push(0);
        for binding in bindings {
            physical_block_slots.extend_from_slice(&binding.physical_block_slots);
            block_offsets.push(i32::try_from(physical_block_slots.len()).map_err(|_| {
                Error::Model {
                    message: "DeepSeek-V4 packed block table exceeds i32 ABI".into(),
                }
            })?);
            kv_lens.push(
                i32::try_from(binding.sequence_len).map_err(|_| Error::Model {
                    message: "DeepSeek-V4 sequence length exceeds i32 ABI".into(),
                })?,
            );
        }

        if row_sequence_ids.is_empty()
            || row_sequence_ids
                .iter()
                .any(|sequence| *sequence >= bindings.len())
        {
            return Err(Error::Model {
                message: "DeepSeek-V4 packed row selector references a missing sequence".into(),
            });
        }
        let second_kv_lens = vec![0i32; kv_lens.len()];
        let row_sequence_ids = row_sequence_ids
            .iter()
            .map(|sequence| {
                i32::try_from(*sequence).map_err(|_| Error::Model {
                    message: "DeepSeek-V4 row sequence ID exceeds i32 ABI".into(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let shape = (
            physical_block_slots.len(),
            block_offsets.len(),
            kv_lens.len(),
            row_sequence_ids.len(),
        );
        if self.active_paged_kv.as_ref().is_some_and(|active| {
            (
                active.block_slots_device.len(),
                active.block_offsets_device.len(),
                active.kv_len_device.len(),
                active.row_sequence_ids_device.len(),
            ) != shape
        }) {
            if let Some(active) = self.active_paged_kv.take() {
                let old_shape = (
                    active.block_slots_device.len(),
                    active.block_offsets_device.len(),
                    active.kv_len_device.len(),
                    active.row_sequence_ids_device.len(),
                );
                self.cached_paged_kv.insert(old_shape, active);
            }
        }
        if self.active_paged_kv.is_none() {
            self.active_paged_kv = self.cached_paged_kv.remove(&shape);
        }
        let same_shape = self.active_paged_kv.is_some();
        if same_shape {
            let active = self
                .active_paged_kv
                .as_mut()
                .expect("same-shape active paged binding exists");
            self.ops
                .update_i32_host_mirror(&physical_block_slots, &mut active.block_slots_device)?;
            self.ops
                .update_i32_host_mirror(&block_offsets, &mut active.block_offsets_device)?;
            self.ops
                .update_i32_host_mirror(&kv_lens, &mut active.kv_len_device)?;
            self.ops
                .update_i32_host_mirror(&second_kv_lens, &mut active.second_kv_len_device)?;
            self.ops
                .update_i32_host_mirror(&row_sequence_ids, &mut active.row_sequence_ids_device)?;
            active.physical_block_slots = physical_block_slots;
            active.page_tokens = page_tokens;
            active.layer_count = layer_count;
            active.sequence_count = bindings.len();
        } else {
            self.active_paged_kv = Some(ActivePagedKvBinding {
                block_slots_device: self.ops.i32_host_mirror(&physical_block_slots)?,
                block_offsets_device: self.ops.i32_host_mirror(&block_offsets)?,
                kv_len_device: self.ops.i32_host_mirror(&kv_lens)?,
                second_kv_len_device: self.ops.i32_host_mirror(&second_kv_lens)?,
                row_sequence_ids_device: self.ops.i32_host_mirror(&row_sequence_ids)?,
                physical_block_slots,
                page_tokens,
                layer_count,
                sequence_count: bindings.len(),
            });
        }
        self.active_paged_kv_owner = Some(transaction);
        Ok(())
    }

    pub(crate) fn commit_kv_pages(&mut self, transaction: ExecutionTransactionId) -> Result<()> {
        let pool = self.kv_page_pool.as_mut().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA physical KV pool is not configured".into(),
        })?;
        let reservations = self
            .pending_kv_reservations
            .remove(&transaction)
            .ok_or_else(|| Error::Execution {
                message: format!(
                    "DeepSeek-V4 CUDA transaction {} is not prepared",
                    transaction.get()
                ),
            })?;
        let result = pool.commit_many(reservations);
        if result.is_ok() {
            self.discard_provisional_prefix_checkpoints(transaction);
        }
        result
    }

    pub(crate) fn rollback_kv_pages(&mut self, transaction: ExecutionTransactionId) -> Result<()> {
        let reservations = self
            .pending_kv_reservations
            .remove(&transaction)
            .ok_or_else(|| Error::Execution {
                message: format!(
                    "DeepSeek-V4 CUDA transaction {} is not prepared",
                    transaction.get()
                ),
            })?;
        let result = if let Some(pool) = self.kv_page_pool.as_mut() {
            let mut first_error = None;
            for reservation in reservations {
                if let Err(error) = pool.rollback(&self.ops, reservation) {
                    first_error.get_or_insert(error);
                }
            }
            match first_error {
                Some(error) => Err(error),
                None => Ok(()),
            }
        } else {
            Err(Error::Model {
                message: "DeepSeek-V4 pending KV pages have no physical pool".into(),
            })
        };
        self.discard_provisional_prefix_checkpoints(transaction);
        result
    }

    pub(crate) fn release_kv_pages(&mut self, pages: &[KvPageId]) -> Result<()> {
        let Some(pool) = self.kv_page_pool.as_mut() else {
            return Ok(());
        };
        for page in pages {
            if pool.physical_slot(*page).is_some() || pool.has_snapshot(*page) {
                pool.release(&self.ops, *page)?;
            }
        }
        Ok(())
    }

    pub(crate) fn preempt_kv_pages(&mut self, pages: &[KvPageId]) -> Result<()> {
        if pages.is_empty() {
            return Ok(());
        }
        let pool = self.kv_page_pool.as_mut().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA physical KV pool is not configured".into(),
        })?;
        pool.preempt(&self.ops, pages).map(|_| ())
    }

    pub(crate) fn restore_kv_pages(&mut self, pages: &[KvPageId]) -> Result<()> {
        if pages.is_empty() {
            return Ok(());
        }
        let pool = self.kv_page_pool.as_mut().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA physical KV pool is not configured".into(),
        })?;
        pool.restore(&self.ops, pages)
    }

    pub(crate) fn take_decode_buffers(
        &mut self,
        hc_len: usize,
        hidden_len: usize,
        topk_row_len: usize,
    ) -> Result<DeepSeekV4DecodeBuffers> {
        let hc_input = match self.decode_arena.hc_inputs.remove(&hc_len) {
            Some(buffer) => buffer,
            None => self.ops.zero_f32_buffer(hc_len)?,
        };
        let final_hidden = match self.decode_arena.final_hiddens.remove(&hidden_len) {
            Some(buffer) => buffer,
            None => self.ops.zero_f32_buffer(hidden_len)?,
        };
        let final_norm = match self.decode_arena.final_norms.remove(&hidden_len) {
            Some(buffer) => buffer,
            None => self.ops.zero_f32_buffer(hidden_len)?,
        };
        let topk_row = match self.decode_arena.topk_rows.remove(&topk_row_len) {
            Some(buffer) => buffer,
            None => self.ops.zero_f32_buffer(topk_row_len)?,
        };
        Ok(DeepSeekV4DecodeBuffers {
            hc_input,
            final_hidden,
            final_norm,
            topk_row,
        })
    }

    pub(crate) fn restore_decode_buffers(&mut self, buffers: DeepSeekV4DecodeBuffers) {
        self.decode_arena
            .hc_inputs
            .insert(buffers.hc_input.len(), buffers.hc_input);
        self.decode_arena
            .final_hiddens
            .insert(buffers.final_hidden.len(), buffers.final_hidden);
        self.decode_arena
            .final_norms
            .insert(buffers.final_norm.len(), buffers.final_norm);
        self.decode_arena
            .topk_rows
            .insert(buffers.topk_row.len(), buffers.topk_row);
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn take_proposal_main_buffers(
        &mut self,
        rows: usize,
    ) -> Result<DeepSeekV4ProposalMainBuffers> {
        if rows == 0 {
            return Err(Error::Model {
                message: "DeepSeek-V4 Proposal main buffers require at least one row".into(),
            });
        }
        if let Some(buffers) = self.decode_arena.proposal_main.remove(&rows) {
            return Ok(buffers);
        }
        let (tap_count, input_size, output_size, context_kv_size) = {
            let mtp = self
                .prepared_image()?
                .mtp
                .as_ref()
                .ok_or_else(|| Error::Model {
                    message: "DeepSeek-V4 CUDA Proposal image is missing".into(),
                })?;
            let stage_zero = mtp.layers.first().ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 CUDA Proposal image has no stages".into(),
            })?;
            let projection = stage_zero.main_proj.as_ref().ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 CUDA Proposal stage zero main projection is missing".into(),
            })?;
            let output_size = projection.handle.shape().out_features();
            let context_kv_size = stage_zero
                .transformer
                .key_value
                .handle
                .shape()
                .out_features();
            for (stage, layer) in mtp.layers.iter().enumerate() {
                let key_value = &layer.transformer.key_value.handle;
                if key_value.shape().in_features() != output_size
                    || key_value.shape().out_features() != context_kv_size
                    || layer.transformer.key_value_norm.len() != context_kv_size
                {
                    return Err(Error::Model {
                        message: format!(
                            "DeepSeek-V4 CUDA Proposal stage {stage} context-KV shape mismatch: wkv={:?} norm={} main_x={output_size} expected_kv={context_kv_size}",
                            key_value.shape(),
                            layer.transformer.key_value_norm.len()
                        ),
                    });
                }
            }
            (
                mtp.target_layer_ids.len(),
                projection.handle.shape().in_features(),
                output_size,
                context_kv_size,
            )
        };
        if tap_count == 0 || input_size % tap_count != 0 {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 CUDA Proposal tap layout is invalid: taps={tap_count} input={input_size}"
                ),
            });
        }
        let target_taps_len = rows.checked_mul(input_size).ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA Proposal target-tap size overflow".into(),
        })?;
        let main_x_len = rows.checked_mul(output_size).ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA Proposal main-x size overflow".into(),
        })?;
        let context_kv_len = rows
            .checked_mul(context_kv_size)
            .ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 CUDA Proposal context-KV size overflow".into(),
            })?;
        Ok(DeepSeekV4ProposalMainBuffers {
            target_taps: self.ops.zero_f32_buffer(target_taps_len)?,
            positions: self.ops.zero_i32_buffer(rows)?,
            activation: self.ops.fp8_activation_pack(rows, input_size)?,
            inv_rms: self.ops.zero_f32_buffer(rows)?,
            main_x: self.ops.zero_f32_buffer(main_x_len)?,
            context_kv_raw: self.ops.zero_f32_buffer(context_kv_len)?,
            context_kv: self.ops.zero_f32_buffer(context_kv_len)?,
            context_linear_workspace: self.ops.artifact_linear_workspace(rows, output_size)?,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn restore_proposal_main_buffers(
        &mut self,
        rows: usize,
        buffers: DeepSeekV4ProposalMainBuffers,
    ) {
        self.decode_arena.proposal_main.insert(rows, buffers);
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn allocate_proposal_attention_buffers(
        &self,
    ) -> Result<DeepSeekV4ProposalAttentionBuffers> {
        let mtp = self
            .prepared_image()?
            .mtp
            .as_ref()
            .ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 CUDA Proposal image is missing".into(),
            })?;
        if mtp.block_size != ferrule_backend::cuda::cutlass::PROPOSAL_ROWS {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 CUDA Proposal attention requires {} proposal rows, checkpoint declares {}",
                    ferrule_backend::cuda::cutlass::PROPOSAL_ROWS,
                    mtp.block_size
                ),
            });
        }
        Ok(DeepSeekV4ProposalAttentionBuffers {
            workspace: self.ops.hybrid_mla_attention_workspace()?,
        })
    }

    pub(crate) fn resident_embedding_hc_rows_into(
        &mut self,
        token_ids: &[u32],
        output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (token_ids, output);
            Err(Error::Model { message:
                "DeepSeek-V4 packed CUDA resident embedding gather requires building with CUTLASS support; host artifact fallback is disabled"
                    .into(),
             })
        }
        #[cfg(feature = "cuda")]
        {
            let (vocab, hc_mult) = {
                let image = self.prepared_image()?;
                (
                    image.embedding.shape().out_features(),
                    image.hc_config.hc_mult,
                )
            };
            if token_ids.is_empty() {
                return Err(Error::Model {
                    message: "DeepSeek-V4 packed CUDA embedding gather requires at least one token"
                        .into(),
                });
            }
            let device_values = token_ids
                .iter()
                .copied()
                .map(|token_id| {
                    if token_id as usize >= vocab {
                        return Err(Error::Model { message: format!(
                            "DeepSeek-V4 token id {token_id} exceeds resident embedding vocab {vocab}"
                        ) });
                    }
                    i32::try_from(token_id).map_err(|_| {
                        Error::Model { message: format!(
                            "DeepSeek-V4 token id {token_id} exceeds the CUDA i32 token ABI"
                        ) }
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let rows = token_ids.len();
            if let Some(cached) = self.embedding_token_ids.get_mut(&rows) {
                self.ops.update_i32_host_mirror(&device_values, cached)?;
            } else {
                self.embedding_token_ids
                    .insert(rows, self.ops.i32_host_mirror(&device_values)?);
            }
            let image = self
                .execution_image
                .as_ref()
                .ok_or_else(|| Error::Internal {
                    message:
                        "DeepSeek-V4 CUDA execution image was not compiled before embedding gather"
                            .into(),
                })?;
            let token_ids = self
                .embedding_token_ids
                .get(&rows)
                .expect("embedding token buffer initialized above");
            self.ops.resident_embedding_hc_bf16_into(
                &image.embedding,
                token_ids.device(),
                rows,
                hc_mult,
                output,
            )
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn proposal_input_device_into(
        &self,
        anchor_token_id: u32,
        output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        let image = self.prepared_image()?;
        let mtp = image.mtp.as_ref().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA Proposal image is missing".into(),
        })?;
        self.ops.proposal_embedding_hc_from_resident_bf16_into(
            &image.embedding,
            anchor_token_id,
            mtp.noise_token_id,
            mtp.block_size,
            image.hc_config.hc_mult,
            output,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn allocate_proposal_head_buffers(&self) -> Result<DeepSeekV4ProposalHeadBuffers> {
        const PARTIAL_CAPACITY: usize = 64;
        let image = self.prepared_image()?;
        let mtp = image.mtp.as_ref().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA Proposal image is missing".into(),
        })?;
        let output_shape = image.output_head.shape();
        let vocab = output_shape.out_features();
        let hidden = output_shape.in_features();
        let markov_rank = mtp.heads.markov_w1.handle.shape().in_features();
        if mtp.block_size != ferrule_backend::cuda::cutlass::PROPOSAL_ROWS
            || hidden != image.hc_config.hidden_size
            || mtp.heads.markov_w1.handle.shape().out_features() != vocab
            || mtp.heads.markov_w2.handle.shape().out_features() != vocab
            || mtp.heads.markov_w2.handle.shape().in_features() != markov_rank
            || mtp.heads.confidence_proj.handle.shape().out_features() != 1
            || mtp.heads.confidence_proj.handle.shape().in_features()
                != hidden.saturating_add(markov_rank)
        {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 Proposal-head shape mismatch: output={output_shape:?} block={} w1={:?} w2={:?} confidence={:?}",
                    mtp.block_size,
                    mtp.heads.markov_w1.handle.shape(),
                    mtp.heads.markov_w2.handle.shape(),
                    mtp.heads.confidence_proj.handle.shape()
                ),
            });
        }
        Ok(DeepSeekV4ProposalHeadBuffers {
            workspace: self.ops.proposal_head_workspace(
                mtp.block_size,
                hidden,
                vocab,
                PARTIAL_CAPACITY,
            )?,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn capture_proposal_target_tap_from_device(
        &self,
        target_layer: usize,
        hc_state: &ferrule_backend::cuda::context::CudaF32Buffer,
        rows: usize,
        target_taps: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<bool> {
        let image = self.prepared_image()?;
        let mtp = image.mtp.as_ref().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA Proposal image is missing".into(),
        })?;
        let Some(tap_slot) = mtp
            .target_layer_ids
            .iter()
            .position(|layer| *layer == target_layer)
        else {
            return Ok(false);
        };
        self.ops.hc_mean_scatter_from_device_into(
            hc_state,
            rows,
            image.hc_config.hc_mult,
            image.hc_config.hidden_size,
            tap_slot,
            mtp.target_layer_ids.len(),
            target_taps,
        )?;
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn proposal_main_project_norm_device_into(
        &self,
        rows: usize,
        buffers: &mut DeepSeekV4ProposalMainBuffers,
    ) -> Result<()> {
        let descriptor =
            self.require_mtp_inference_descriptor(0, KernelOperation::MainProjectNorm)?;
        if descriptor.kernel.provider != KernelProviderId::CUDA_CUTLASS
            || !descriptor.is_provider_managed()
        {
            return Err(Error::Model {
                message: format!(
                    "invalid Proposal main-project/norm provider binding: {:?}",
                    descriptor.kernel
                ),
            });
        }
        let image = self.prepared_image()?;
        let stage_zero = image
            .mtp
            .as_ref()
            .and_then(|mtp| mtp.layers.first())
            .ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 CUDA Proposal stage zero is missing".into(),
            })?;
        let projection = stage_zero.main_proj.as_ref().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA Proposal stage-zero projection is missing".into(),
        })?;
        let norm = stage_zero.main_norm.as_ref().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA Proposal stage-zero norm is missing".into(),
        })?;
        self.ops.artifact_main_project_norm_into(
            &projection.handle,
            norm,
            &buffers.target_taps,
            rows,
            image.hc_config.norm_eps,
            &mut buffers.activation,
            &mut buffers.inv_rms,
            &mut buffers.main_x,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn proposal_head_device_into(
        &self,
        anchor_token_id: u32,
        hc_state: &ferrule_backend::cuda::context::CudaF32Buffer,
        buffers: &mut DeepSeekV4ProposalHeadBuffers,
    ) -> Result<()> {
        let descriptor = self.require_mtp_inference_descriptor(0, KernelOperation::ProposalHead)?;
        if descriptor.kernel.provider != KernelProviderId::CUDA_CUTLASS
            || !descriptor.is_provider_managed()
        {
            return Err(Error::Model {
                message: format!(
                    "invalid Proposal-head provider binding: {:?}",
                    descriptor.kernel
                ),
            });
        }
        let image = self.prepared_image()?;
        let mtp = image.mtp.as_ref().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA Proposal image is missing".into(),
        })?;
        let output_shape = image.output_head.shape();
        let markov_rank = mtp.heads.markov_w1.handle.shape().in_features();
        self.ops.artifact_proposal_head_into(
            hc_state,
            &mtp.heads.hc_head.function_row_major,
            &mtp.heads.hc_head.scale,
            &mtp.heads.hc_head.base,
            &mtp.heads.norm,
            &image.output_head,
            &mtp.heads.markov_w1.handle,
            &mtp.heads.markov_w2.handle,
            &mtp.heads.confidence_proj.handle,
            anchor_token_id,
            ferrule_backend::cuda::cutlass::ProposalHeadLayout {
                rows: mtp.block_size,
                hc: image.hc_config.hc_mult,
                hidden: output_shape.in_features(),
                vocab: output_shape.out_features(),
                markov_rank,
                partial_capacity: 64,
                hc_eps: image.hc_config.eps,
                norm_eps: image.hc_config.norm_eps,
            },
            &mut buffers.workspace,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn begin_proposal_head_result_download(
        &self,
        buffers: &mut DeepSeekV4ProposalHeadBuffers,
    ) -> Result<ferrule_backend::cuda::context::CudaI32HostDownload> {
        self.ops
            .begin_proposal_head_result_download(&mut buffers.workspace)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn poll_proposal_head_result(
        &self,
        buffers: &mut DeepSeekV4ProposalHeadBuffers,
        download: &ferrule_backend::cuda::context::CudaI32HostDownload,
    ) -> Result<Option<Vec<i32>>> {
        self.ops
            .poll_proposal_head_result(&mut buffers.workspace, download)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn decode_proposal_head_result(
        &self,
        compact: Vec<i32>,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let rows = ferrule_backend::cuda::cutlass::PROPOSAL_ROWS;
        if compact.len() != 1 + 2 * rows || compact[0] != 0 {
            return Err(Error::Execution {
                message: format!(
                    "DeepSeek-V4 Proposal compact proposal-head result is invalid: {compact:?}"
                ),
            });
        }
        let token_ids = compact[1..1 + rows]
            .iter()
            .copied()
            .map(|token| {
                u32::try_from(token).map_err(|_| Error::Execution {
                    message: format!("DeepSeek-V4 Proposal emitted invalid token {token}"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let confidence = compact[1 + rows..]
            .iter()
            .map(|bits| f32::from_bits(*bits as u32))
            .collect();
        Ok((token_ids, confidence))
    }

    /// Execute the official Proposal hybrid attention against one stage's
    /// committed paged context and the read-only five-row proposal KV block.
    /// The stage is resolved only through the prepared MTP image; target-layer
    /// bindings are never consulted.
    #[cfg(feature = "cuda")]
    pub(crate) fn proposal_hybrid_attention_device_into(
        &self,
        stage: usize,
        expected_execution_layer: usize,
        config: DeepSeekV4AttentionConfig,
        sequence_tokens: usize,
        query: &ferrule_backend::cuda::context::CudaF32Buffer,
        block_kv: &ferrule_backend::cuda::context::CudaF32Buffer,
        output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        buffers: &mut DeepSeekV4ProposalAttentionBuffers,
    ) -> Result<()> {
        let descriptor =
            self.require_mtp_inference_descriptor(stage, KernelOperation::HybridMlaAttention)?;
        if descriptor.kernel.provider != KernelProviderId::CUDA_CUTLASS
            || !descriptor.is_provider_managed()
        {
            return Err(Error::Model {
                message: format!(
                    "invalid Proposal hybrid-attention provider binding at stage {stage}: {:?}",
                    descriptor.kernel
                ),
            });
        }
        let prepared = self.prepared_mtp_stage(stage)?;
        let execution_layer = prepared.execution_layer;
        if execution_layer != expected_execution_layer {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 Proposal stage/execution-layer mismatch: stage={stage} prepared={execution_layer} requested={expected_execution_layer}"
                ),
            });
        }
        if config.num_heads != ferrule_backend::cuda::cutlass::HYBRID_MLA_ATTENTION_HEADS
            || config.head_dim != ferrule_backend::cuda::cutlass::HYBRID_MLA_ATTENTION_HEAD_DIM
            || config.window_size != ferrule_backend::cuda::cutlass::HYBRID_MLA_ATTENTION_WINDOW
            || config.compress_ratio != 0
        {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 Proposal hybrid-attention shape mismatch at stage {stage}: heads={} head_dim={} window={} compress_ratio={}",
                    config.num_heads, config.head_dim, config.window_size, config.compress_ratio
                ),
            });
        }
        let active = self.active_paged_kv.as_ref().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 Proposal hybrid attention has no active paged binding".into(),
        })?;
        if active.sequence_count != 1
            || sequence_tokens == 0
            || sequence_tokens
                > active
                    .physical_block_slots
                    .len()
                    .saturating_mul(active.page_tokens)
            || execution_layer >= active.layer_count
        {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 Proposal hybrid-attention binding mismatch: stage={stage} execution_layer={execution_layer} sequence_tokens={sequence_tokens} sequences={} slots={} page_tokens={} layer_count={}",
                    active.sequence_count,
                    active.physical_block_slots.len(),
                    active.page_tokens,
                    active.layer_count
                ),
            });
        }
        let pool = self.kv_page_pool.as_ref().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA physical KV pool is not configured".into(),
        })?;
        let context_plane = pool.plane_storage(0).ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 Proposal window KV plane is missing".into(),
        })?;
        let plane = pool.planes().first().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 Proposal window KV descriptor is missing".into(),
        })?;
        if active.page_tokens != ferrule_backend::cuda::cutlass::HYBRID_MLA_ATTENTION_PAGE_TOKENS
            || plane.elements_per_token != config.head_dim
            || plane.layer_count != active.layer_count
        {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 Proposal hybrid-attention plane mismatch: page_tokens={} elements_per_token={} layer_count={} expected_page_tokens={} expected_head_dim={} active_layers={}",
                    active.page_tokens,
                    plane.elements_per_token,
                    plane.layer_count,
                    ferrule_backend::cuda::cutlass::HYBRID_MLA_ATTENTION_PAGE_TOKENS,
                    config.head_dim,
                    active.layer_count
                ),
            });
        }
        self.ops.hybrid_mla_attention_into(
            query,
            context_plane,
            block_kv,
            &active.block_slots_device,
            &prepared.transformer.attention_sink,
            ferrule_backend::cuda::cutlass::HybridMlaAttentionLayout {
                sequence_tokens,
                page_tokens: active.page_tokens,
                elements_per_token: plane.elements_per_token,
                layer_index: execution_layer,
                layer_count: active.layer_count,
                block_slot_offset: 0,
                block_slot_count: active.physical_block_slots.len(),
                softmax_scale: (config.head_dim as f32).powf(-0.5),
            },
            output,
            &mut buffers.workspace,
        )
    }

    /// Packed variant of the official Proposal context branch. Target rows may
    /// belong to multiple sequences, so RoPE and paged publication use the
    /// authoritative per-row positions and active row-to-sequence bindings.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn proposal_context_kv_stage_packed_device_into(
        &mut self,
        stage: usize,
        config: DeepSeekV4AttentionConfig,
        rows: usize,
        max_position: usize,
        buffers: &mut DeepSeekV4ProposalMainBuffers,
    ) -> Result<()> {
        const ROPE_NAMES: [&str; 8] = [
            "rope_proposal_stage_0",
            "rope_proposal_stage_1",
            "rope_proposal_stage_2",
            "rope_proposal_stage_3",
            "rope_proposal_stage_4",
            "rope_proposal_stage_5",
            "rope_proposal_stage_6",
            "rope_proposal_stage_7",
        ];
        if rows == 0 || config.compress_ratio != 0 || buffers.positions.len() != rows {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 packed Proposal stage {stage} context-KV shape mismatch: rows={rows} positions={} compress_ratio={}",
                    buffers.positions.len(),
                    config.compress_ratio
                ),
            });
        }
        let rope_name = ROPE_NAMES.get(stage).copied().ok_or_else(|| Error::Model {
            message: format!(
                "DeepSeek-V4 Proposal stage {stage} exceeds the prepared RoPE identity table"
            ),
        })?;
        let execution_layer = self.prepared_mtp_stage(stage)?.execution_layer;
        if buffers.main_x.len() != rows.saturating_mul(config.hidden_size)
            || buffers.context_kv_raw.len() != rows.saturating_mul(config.head_dim)
            || buffers.context_kv.len() != rows.saturating_mul(config.head_dim)
        {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 packed Proposal stage {stage} context-KV buffer mismatch: main_x={}/{} raw={}/{} kv={}/{}",
                    buffers.main_x.len(),
                    rows.saturating_mul(config.hidden_size),
                    buffers.context_kv_raw.len(),
                    rows.saturating_mul(config.head_dim),
                    buffers.context_kv.len(),
                    rows.saturating_mul(config.head_dim)
                ),
            });
        }

        {
            let prepared = self.prepared_mtp_stage(stage)?;
            self.ops
                .artifact_linear_rows_from_device_into_with_scratch(
                    &prepared.transformer.key_value.handle,
                    &buffers.main_x,
                    rows,
                    &mut buffers.context_kv_raw,
                    &mut buffers.context_linear_workspace,
                )?;
        }
        {
            let prepared = self.prepared_mtp_stage(stage)?;
            self.ops.rms_norm_rows_from_device_into(
                &buffers.context_kv_raw,
                rows,
                &prepared.transformer.key_value_norm,
                config.norm_eps,
                &mut buffers.context_kv,
            )?;
        }
        let required_positions = max_position.checked_add(1).ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 Proposal context position overflow".into(),
        })?;
        self.ensure_rope_tables_with_params(
            rope_name,
            config.rope_head_dim,
            config.rope_params(),
            required_positions,
        )?;
        self.rope_tail_rows_indexed_from_device(
            rope_name,
            &mut buffers.context_kv,
            &buffers.positions,
            max_position,
            1,
            u32::try_from(config.head_dim).map_err(|_| Error::Model {
                message: "DeepSeek-V4 Proposal context head dimension exceeds u32".into(),
            })?,
            u32::try_from(config.rope_head_dim).map_err(|_| Error::Model {
                message: "DeepSeek-V4 Proposal context RoPE dimension exceeds u32".into(),
            })?,
            false,
        )?;
        self.ops.fp8_attention_kv_qat_quantize_buffer_in_place(
            &mut buffers.context_kv,
            config.head_dim,
            config.rope_head_dim,
        )?;
        self.paged_scatter_rows_from_device(
            0,
            execution_layer,
            &buffers.context_kv,
            &buffers.positions,
            None,
            config.head_dim,
        )
    }

    pub(crate) fn drain_moe_access_events(&mut self) -> Vec<ExpertBatchAccessEvent> {
        std::mem::take(&mut self.moe_access_events)
    }

    pub(super) fn configure_expert_subsystem(
        &mut self,
        subsystem: CudaSharedExpertSubsystem,
    ) -> Result<()> {
        if self.expert_subsystem.is_some() {
            return Err(Error::Execution {
                message: "DeepSeek-V4 shared expert subsystem is already attached".into(),
            });
        }
        self.expert_subsystem = Some(subsystem);
        Ok(())
    }

    pub(crate) fn resident_expert_stats_for_layer(&self, layer: usize) -> (usize, u64) {
        self.expert_subsystem.as_ref().map_or((0, 0), |subsystem| {
            subsystem.resident_stats_for_layer(layer)
        })
    }

    pub(crate) fn shutdown(&mut self) -> Result<()> {
        self.ops.sync_stream()?;
        self.decode_arena = DeepSeekV4DecodeArena::default();

        self.hybrid_mla_explicit_selection_workspaces.clear();
        self.paged_topk_buffers.clear();
        #[cfg(feature = "cuda")]
        {
            self.output_head_logits.clear();
            self.output_head_linear_workspaces.clear();
            self.output_head_indices.clear();
            self.output_head_values.clear();
            self.embedding_token_ids.clear();
        }
        self.execution_image = None;
        self.router_token_ids.clear();
        self.compact_i32_mirrors.clear();
        self.rope_tables.clear();
        self.expert_subsystem = None;
        Ok(())
    }

    pub(crate) fn runtime_counters(&self) -> DeepSeekV4OperatorRuntimeCounters {
        let cuda = self.ops.counters();
        DeepSeekV4OperatorRuntimeCounters {
            kernel_launches: cuda.kernel_launches,
            host_to_device_copies: cuda.host_to_device_copies,
            host_to_device_bytes: cuda.host_to_device_bytes,
            device_to_host_copies: cuda.device_to_host_copies,
            device_to_host_bytes: cuda.device_to_host_bytes,
            artifact_uploads: cuda.artifact_uploads,
            artifact_upload_bytes: cuda.artifact_upload_bytes,
            device_allocation_attempts: cuda.device_allocation_attempts,
            device_allocations: cuda.device_allocations,
            device_allocation_failures: cuda.device_allocation_failures,
            device_allocation_bytes: cuda.device_allocation_bytes,
            stream_wide_syncs: cuda.stream_wide_syncs,
            stream_wide_sync_failures: cuda.stream_wide_sync_failures,
            moe_calls: cuda.moe_calls,
            moe_total_us: cuda.moe_total_us,
            moe_input_prepare_us: cuda.moe_input_prepare_us,
            moe_gate_up_us: cuda.moe_gate_up_us,
            moe_swiglu_us: cuda.moe_swiglu_us,
            moe_down_us: cuda.moe_down_us,
            output_head_calls: self.metrics.output_head_calls,
            output_head_rows: self.metrics.output_head_rows,
            output_head_topk_us: self.metrics.output_head_topk_us,
            moe_router_us: self.metrics.moe_router_us,
            moe_routing_us: self.metrics.moe_routing_us,
            moe_plan_us: self.metrics.moe_plan_us,
            moe_shared_us: self.metrics.moe_shared_us,
            moe_workspace_us: self.metrics.moe_workspace_us,
            moe_compute_submit_us: self.metrics.moe_compute_submit_us,
            moe_commit_us: self.metrics.moe_commit_us,
            expert_selected: self.metrics.expert_selected,
            expert_unique_selected: self.metrics.expert_unique_selected,
            expert_selected_load_requests: self.metrics.expert_selected_load_requests,
            arena_hits: cuda.arena_hits,
            arena_misses: cuda.arena_misses,
            arena_grows: cuda.arena_grows,
            arena_reuses: cuda.arena_reuses,
            ..DeepSeekV4OperatorRuntimeCounters::default()
        }
    }

    #[cfg(feature = "cuda")]
    fn readonly_prepared_linear(
        &self,
        layer: usize,
        linear: DeepSeekV4CudaLinear,
        input_len: usize,
    ) -> Result<&DeepSeekV4CudaPreparedLinear> {
        let prepared = self.prepared_linear(layer, linear)?;
        let in_features = prepared.handle.shape().in_features();
        if input_len != in_features {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 layer {layer} CUDA readonly {linear:?} input length mismatch: expected {in_features}, got {input_len}"
                ),
            });
        }
        if prepared.activation_quantization.is_some() {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 layer {layer} CUDA readonly {linear:?} cannot skip activation quantization"
                ),
            });
        }
        if !matches!(
            prepared.handle.shape(),
            ferrule_backend::cuda::context::CudaArtifactLinearShape::F32 { .. }
                | ferrule_backend::cuda::context::CudaArtifactLinearShape::Bf16Bytes { .. }
        ) {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 layer {layer} CUDA readonly {linear:?} requires F32/BF16 weights, got {:?}",
                    prepared.handle.shape()
                ),
            });
        }
        Ok(prepared)
    }

    pub(crate) fn linear_pair_matvec_readonly_from_device_into(
        &self,
        layer: usize,
        first: DeepSeekV4CudaLinear,
        second: DeepSeekV4CudaLinear,
        input: &ferrule_backend::cuda::context::CudaF32Buffer,
        first_output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        second_output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (layer, first, second, input, first_output, second_output);
            return Err(Error::Model {
                message: "CUDA compressor execution requires the `cutlass` feature".into(),
            });
        }
        #[cfg(feature = "cuda")]
        {
            let operation = match (first, second) {
                (
                    DeepSeekV4CudaLinear::MainCompressorKv,
                    DeepSeekV4CudaLinear::MainCompressorGate,
                ) => KernelOperation::MainCompressorProjection,
                (
                    DeepSeekV4CudaLinear::IndexerCompressorKv,
                    DeepSeekV4CudaLinear::IndexerCompressorGate,
                ) => KernelOperation::IndexerCompressorProjection,
                _ => {
                    return Err(Error::Model {
                        message: format!("BF16 bundle does not bind {first:?}+{second:?}"),
                    });
                }
            };
            self.require_inference_operation(layer, operation)?;
            let first = self.readonly_prepared_linear(layer, first, input.len())?;
            let second = self.readonly_prepared_linear(layer, second, input.len())?;
            self.ops.artifact_bf16_compressor_into(
                &first.handle,
                &second.handle,
                input,
                1,
                first_output,
                second_output,
            )
        }
    }

    fn upload_hc_device_weights(
        &self,
        weights: &HyperConnectionWeights,
        config: HyperConnectionConfig,
    ) -> Result<HcDeviceWeights> {
        weights.validate(config)?;
        Ok(HcDeviceWeights {
            function_row_major: self.ops.upload_f32_buffer(&weights.function)?,
            scale: self.ops.upload_f32_buffer(&weights.scale)?,
            base: self.ops.upload_f32_buffer(&weights.base)?,
        })
    }

    #[cfg(feature = "cuda")]
    fn upload_resident_bf16_matrix(
        &self,
        matrix: &CheckpointMatrixSlice,
    ) -> Result<ferrule_backend::cuda::context::CudaArtifactLinearHandle> {
        const CHUNK_BYTES: usize = 16 * 1024 * 1024;
        if matrix.slice.dtype != CheckpointDType::Bf16 {
            return Err(Error::Model {
                message: format!(
                    "resident BF16 matrix '{}' has dtype {:?}",
                    matrix.slice.name, matrix.slice.dtype
                ),
            });
        }
        let expected_bytes = matrix
            .rows
            .checked_mul(matrix.cols)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Error::Model {
                message: "resident BF16 matrix size overflow".into(),
            })?;
        if matrix.slice.bytes != expected_bytes as u64 {
            return Err(Error::Model {
                message: format!(
                    "resident BF16 matrix '{}' byte mismatch: slice={} expected={expected_bytes}",
                    matrix.slice.name, matrix.slice.bytes
                ),
            });
        }
        let shape = ferrule_backend::cuda::context::CudaArtifactLinearShape::Bf16Bytes {
            out_features: matrix.rows,
            in_features: matrix.cols,
        };
        let mut handle = self.ops.allocate_artifact_linear_device(shape)?;
        let mut file = File::open(&matrix.slice.path).map_err(|error| Error::Model {
            message: format!(
                "open resident BF16 matrix '{}': {error}",
                matrix.slice.path.display()
            ),
        })?;
        file.seek(SeekFrom::Start(matrix.slice.offset))
            .map_err(|error| Error::Model {
                message: format!(
                    "seek resident BF16 matrix '{}': {error}",
                    matrix.slice.path.display()
                ),
            })?;
        let mut chunk = vec![0u8; CHUNK_BYTES.min(expected_bytes)];
        let mut offset = 0usize;
        while offset < expected_bytes {
            let bytes = chunk.len().min(expected_bytes - offset);
            file.read_exact(&mut chunk[..bytes])
                .map_err(|error| Error::Model {
                    message: format!(
                        "read resident BF16 matrix '{}' at {offset}: {error}",
                        matrix.slice.name
                    ),
                })?;
            self.ops.overwrite_artifact_linear_weight_range(
                &mut handle,
                offset,
                &chunk[..bytes],
            )?;
            offset += bytes;
        }
        Ok(handle)
    }

    fn upload_prepared_linear(
        &self,
        linear: &LinearWeight,
    ) -> Result<DeepSeekV4CudaPreparedLinear> {
        Ok(DeepSeekV4CudaPreparedLinear {
            handle: self.upload_linear(linear)?,
            #[cfg(feature = "cuda")]
            activation_quantization: linear.execution.activation_quantization,
        })
    }

    fn upload_compressor_resources(
        &self,
        payload: &DeepSeekV4CompressorPayload,
    ) -> Result<DeepSeekV4CudaPreparedCompressor> {
        Ok(DeepSeekV4CudaPreparedCompressor {
            ape: self.ops.upload_f32_buffer(&payload.ape)?,
            norm: self.upload_norm_weight(&payload.norm)?,
            kv: self.upload_prepared_linear(&payload.wkv)?,
            gate: self.upload_prepared_linear(&payload.wgate)?,
        })
    }

    fn upload_prepared_layer(
        &self,
        layer: &DeepSeekV4Layer,
    ) -> Result<DeepSeekV4CudaPreparedLayer> {
        let layer_id = layer.layer;
        #[cfg(feature = "cuda")]
        let query_a = self.upload_prepared_linear(&layer.attention.payload.query_a)?;
        let query_b = self.upload_prepared_linear(&layer.attention.payload.query_b)?;
        #[cfg(feature = "cuda")]
        let key_value = self.upload_prepared_linear(&layer.attention.payload.key_value)?;
        #[cfg(feature = "cuda")]
        let output_a = self.upload_prepared_linear(&layer.attention.payload.output_a)?;
        #[cfg(feature = "cuda")]
        let output_b = self.upload_prepared_linear(&layer.attention.payload.output_b)?;
        let main_compressor = layer
            .attention
            .compressed
            .as_ref()
            .map(|compressed| self.upload_compressor_resources(&compressed.compressor))
            .transpose()?;
        let indexer_compressor = layer
            .attention
            .compressed
            .as_ref()
            .and_then(|compressed| compressed.indexer.as_ref())
            .map(|indexer| self.upload_compressor_resources(&indexer.compressor))
            .transpose()?;
        let indexer_query = layer
            .attention
            .compressed
            .as_ref()
            .and_then(|compressed| compressed.indexer.as_ref())
            .map(|indexer| self.upload_prepared_linear(&indexer.wq_b))
            .transpose()?;
        let indexer_weights = layer
            .attention
            .compressed
            .as_ref()
            .and_then(|compressed| compressed.indexer.as_ref())
            .map(|indexer| self.upload_prepared_linear(&indexer.weights_proj))
            .transpose()?;
        let router = self.upload_prepared_linear(&layer.router.weight)?;
        let shared_gate = self.upload_prepared_linear(&layer.shared_ffn.gate)?;
        let shared_up = self.upload_prepared_linear(&layer.shared_ffn.up)?;
        let shared_down = self.upload_prepared_linear(&layer.shared_ffn.down)?;
        let experts = layer.router.weight.format.out_features();
        let router_bias = layer
            .router
            .bias
            .as_deref()
            .map(|bias| {
                if bias.len() != experts {
                    return Err(Error::Model { message: format!(
                        "DeepSeek-V4 layer {layer_id} router bias length mismatch: got {} expected {experts}",
                        bias.len()
                    ) });
                }
                self.ops.upload_f32_buffer(bias)
            })
            .transpose()?;
        let router_hash_table = match layer.router_policy.selection {
            RouterSelectionPolicy::ScoreTopK => None,
            RouterSelectionPolicy::Hash => {
                let table = layer
                    .router
                    .hash_table
                    .as_deref()
                    .ok_or_else(|| Error::Model {
                        message: format!(
                            "DeepSeek-V4 layer {layer_id} hash router is missing its hash table"
                        ),
                    })?;
                validate_router_hash_table_shape(
                    layer_id,
                    table,
                    layer.router.hash_rows,
                    layer.router.hash_cols,
                )?;
                Some(self.ops.upload_dsv4_router_hash_table(
                    table,
                    layer.router.hash_rows,
                    layer.router.hash_cols,
                    experts,
                    layer.router_policy.top_k,
                )?)
            }
        };
        Ok(DeepSeekV4CudaPreparedLayer {
            #[cfg(feature = "cuda")]
            query_a,
            query_b,
            #[cfg(feature = "cuda")]
            key_value,
            #[cfg(feature = "cuda")]
            output_a,
            #[cfg(feature = "cuda")]
            output_b,
            indexer_query,
            indexer_weights,
            router,
            shared_gate,
            shared_up,
            shared_down,
            attention_norm: self.upload_norm_weight(&layer.attn_norm)?,
            feed_forward_norm: self.upload_norm_weight(&layer.ffn_norm)?,
            query_norm: self.upload_norm_weight(&layer.attention.payload.query_norm)?,
            key_value_norm: self.upload_norm_weight(&layer.attention.payload.key_value_norm)?,
            attention_hc: self.upload_hc_device_weights(&layer.hc_attention, layer.hc_config)?,
            feed_forward_hc: self
                .upload_hc_device_weights(&layer.hc_feed_forward, layer.hc_config)?,
            attention_sink: self
                .ops
                .upload_f32_buffer(&layer.attention.payload.attention_sink)?,
            main_compressor,
            indexer_compressor,
            router_bias,
            router_hash_table,
        })
    }

    fn upload_prepared_proposal_attachment(
        &self,
        resources: &DeepSeekV4PreparedResources,
        hc_config: HyperConnectionConfig,
    ) -> Result<Option<DeepSeekV4CudaPreparedProposalAttachment>> {
        let Some(mtp) = resources.mtp() else {
            return Ok(None);
        };
        let transformer_kernel_plan =
            resources
                .proposal_transformer_kernel_plan()
                .ok_or_else(|| Error::Model {
                    message: "DeepSeek-V4 MTP transformer plan is missing".into(),
                })?;
        if transformer_kernel_plan.layers.len() != mtp.layers.len() {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 MTP kernel plan has {} layers for {} prepared stages",
                    transformer_kernel_plan.layers.len(),
                    mtp.layers.len()
                ),
            });
        }

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(mtp.layers.len())
            .map_err(|error| Error::Model {
                message: format!(
                    "DeepSeek-V4 CUDA MTP image allocation failed for {} stages: {error}",
                    mtp.layers.len()
                ),
            })?;
        for (stage, layer) in mtp.layers.iter().enumerate() {
            if layer.mtp_index != stage {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 CUDA MTP stage identity mismatch: slot={stage} checkpoint_stage={}",
                        layer.mtp_index
                    ),
                });
            }
            if layer.transformer.layer != layer.execution_layer {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 CUDA MTP execution identity mismatch: stage={stage} transformer_layer={} execution_layer={}",
                        layer.transformer.layer, layer.execution_layer
                    ),
                });
            }
            layers.push(DeepSeekV4CudaPreparedProposalStage {
                execution_layer: layer.execution_layer,
                transformer: self.upload_prepared_layer(&layer.transformer)?,
                main_proj: layer
                    .main_proj
                    .as_ref()
                    .map(|linear| self.upload_prepared_linear(linear))
                    .transpose()?,
                main_norm: layer
                    .main_norm
                    .as_deref()
                    .map(|norm| self.upload_norm_weight(norm))
                    .transpose()?,
            });
        }

        let prediction_heads = mtp.prediction_heads.as_ref().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA MTP image requires prediction heads".into(),
        })?;
        prediction_heads.hc_head.validate(hc_config)?;
        let heads = DeepSeekV4CudaPreparedProposalHeads {
            hc_head: HcHeadDeviceWeights {
                function_row_major: self
                    .ops
                    .upload_f32_buffer(&prediction_heads.hc_head.function)?,
                scale: self
                    .ops
                    .upload_f32_buffer(&prediction_heads.hc_head.scale)?,
                base: self.ops.upload_f32_buffer(&prediction_heads.hc_head.base)?,
            },
            norm: self.upload_norm_weight(&prediction_heads.norm)?,
            markov_w1: self.upload_prepared_linear(&prediction_heads.markov_w1)?,
            markov_w2: self.upload_prepared_linear(&prediction_heads.markov_w2)?,
            confidence_proj: self.upload_prepared_linear(&prediction_heads.confidence_proj)?,
        };
        let noise_token_id = mtp.config.noise_token_id.ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA MTP image requires a noise token id".into(),
        })?;

        Ok(Some(DeepSeekV4CudaPreparedProposalAttachment {
            block_size: mtp.config.block_size,
            noise_token_id,
            target_layer_ids: mtp.config.target_layer_ids.clone().into_boxed_slice(),
            layers: layers.into_boxed_slice(),
            heads,
            transformer_kernel_plan: transformer_kernel_plan.clone(),
        }))
    }

    /// Compile immutable model CUDA resources before any model execution.
    ///
    /// The image is built transactionally in a temporary owner and published
    /// only after every upload succeeds. Dynamic expert and KV resources remain
    /// runtime-owned; embedding and output-head weights live in the image.
    pub(crate) fn compile_execution_image(
        &mut self,
        generation: u64,
        resources: &DeepSeekV4PreparedResources,
    ) -> Result<DeepSeekV4CudaImageProfile> {
        let model = resources.model();
        let layers = resources.layers();
        let kernel_plan = resources.kernel_plan();
        if kernel_plan.layers.len() != layers.len() {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 kernel plan has {} layers for {} prepared layers",
                    kernel_plan.layers.len(),
                    layers.len()
                ),
            });
        }
        if let Some(image) = self.execution_image.as_ref() {
            let image_mtp_layers = image.mtp.as_ref().map_or(0, |mtp| mtp.layers.len());
            let requested_mtp_layers = resources.mtp().map_or(0, |mtp| mtp.layers.len());
            if image.generation == generation
                && image.layers.len() == layers.len()
                && image.kernel_plan.layers.len() == kernel_plan.layers.len()
                && image_mtp_layers == requested_mtp_layers
            {
                return Ok(DeepSeekV4CudaImageProfile::default());
            }
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 CUDA execution image is already compiled for generation {} with {} layers",
                    image.generation,
                    image.layers.len()
                ),
            });
        }

        let total_start = Instant::now();
        let phase_start = Instant::now();
        let hc_config = model.config.hc_config();
        model.hc_head.validate(hc_config)?;
        let hc_head = HcHeadDeviceWeights {
            function_row_major: self.ops.upload_f32_buffer(&model.hc_head.function)?,
            scale: self.ops.upload_f32_buffer(&model.hc_head.scale)?,
            base: self.ops.upload_f32_buffer(&model.hc_head.base)?,
        };
        let output_norm = self.upload_norm_weight(&model.output_norm)?;
        let globals_us = duration_us(phase_start.elapsed());
        #[cfg(feature = "cuda")]
        let phase_start = Instant::now();
        #[cfg(feature = "cuda")]
        let embedding = self.upload_resident_bf16_matrix(&model.embedding)?;
        #[cfg(feature = "cuda")]
        let embedding_us = duration_us(phase_start.elapsed());
        #[cfg(feature = "cuda")]
        let phase_start = Instant::now();
        #[cfg(feature = "cuda")]
        let output_head = self.upload_resident_bf16_matrix(&model.output_head)?;
        #[cfg(feature = "cuda")]
        let output_head_us = duration_us(phase_start.elapsed());

        let phase_start = Instant::now();
        let mut prepared = Vec::new();
        prepared
            .try_reserve_exact(layers.len())
            .map_err(|error| Error::Model {
                message: format!(
                    "DeepSeek-V4 CUDA execution image allocation failed for {} layers: {error}",
                    layers.len()
                ),
            })?;
        for (layer_index, layer) in layers.iter().enumerate() {
            if layer.layer != layer_index {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 CUDA execution image layer identity mismatch: slot={layer_index} layer={}",
                        layer.layer
                    ),
                });
            }
            prepared.push(self.upload_prepared_layer(layer)?);
        }
        let target_layers_us = duration_us(phase_start.elapsed());

        let phase_start = Instant::now();
        let prepared_mtp = self.upload_prepared_proposal_attachment(resources, hc_config)?;
        let attachment_us = duration_us(phase_start.elapsed());
        self.execution_image = Some(DeepSeekV4CudaExecutionImage {
            generation,
            hc_config,
            hc_head,
            output_norm,
            #[cfg(feature = "cuda")]
            embedding,
            #[cfg(feature = "cuda")]
            output_head,
            layers: prepared.into_boxed_slice(),
            kernel_plan: kernel_plan.clone(),
            mtp: prepared_mtp,
        });
        Ok(DeepSeekV4CudaImageProfile {
            globals_us,
            embedding_us,
            output_head_us,
            target_layers_us,
            attachment_us,
            total_us: duration_us(total_start.elapsed()),
        })
    }

    fn prepared_image(&self) -> Result<&DeepSeekV4CudaExecutionImage> {
        self.execution_image
            .as_ref()
            .ok_or_else(|| Error::Internal {
                message: "DeepSeek-V4 CUDA execution image was not compiled before execution"
                    .into(),
            })
    }

    #[cfg(feature = "cuda")]
    fn require_inference_descriptor(
        &self,
        execution_layer: usize,
        operation: KernelOperation,
    ) -> Result<LaunchDescriptor> {
        let image = self.prepared_image()?;
        let descriptor = image
            .kernel_plan
            .layer(execution_layer)
            .and_then(|plan| plan.operation(operation, ExecutionMode::Inference))
            .copied()
            .or_else(|| {
                let mtp = image.mtp.as_ref()?;
                let stage = mtp
                    .layers
                    .iter()
                    .position(|layer| layer.execution_layer == execution_layer)?;
                mtp.transformer_kernel_plan
                    .layer(stage)
                    .and_then(|plan| plan.operation(operation, ExecutionMode::Inference))
                    .copied()
            });
        descriptor.ok_or_else(|| Error::Model {
            message: format!(
                "inference operation {operation:?} is missing for execution layer={execution_layer}"
            ),
        })
    }

    #[cfg(feature = "cuda")]
    fn require_mtp_inference_descriptor(
        &self,
        stage: usize,
        operation: KernelOperation,
    ) -> Result<LaunchDescriptor> {
        self.prepared_image()?
            .mtp
            .as_ref()
            .and_then(|mtp| mtp.transformer_kernel_plan.layer(stage))
            .and_then(|plan| plan.operation(operation, ExecutionMode::Inference))
            .copied()
            .ok_or_else(|| Error::Model {
                message: format!(
                    "DeepSeek-V4 CUDA MTP stage {stage} is missing inference operation {operation:?}"
                ),
            })
    }

    #[cfg(feature = "cuda")]
    fn require_inference_operation(&self, layer: usize, operation: KernelOperation) -> Result<()> {
        let descriptor = self.require_inference_descriptor(layer, operation)?;
        if descriptor.kernel.provider != KernelProviderId::CUDA_CUTLASS
            || descriptor.kernel.operation != operation
            || descriptor.kernel.mode != ExecutionMode::Inference
            || !descriptor.is_provider_managed()
        {
            return Err(Error::Model {
                message: format!(
                    "invalid CUDA inference binding for layer={layer} operation={operation:?}: {:?}",
                    descriptor.kernel
                ),
            });
        }
        Ok(())
    }

    fn prepared_layer(&self, execution_layer: usize) -> Result<&DeepSeekV4CudaPreparedLayer> {
        let image = self.prepared_image()?;
        if let Some(layer) = image.layers.get(execution_layer) {
            return Ok(layer);
        }
        image
            .mtp
            .as_ref()
            .and_then(|mtp| {
                mtp.layers
                    .iter()
                    .find(|layer| layer.execution_layer == execution_layer)
            })
            .map(|layer| &layer.transformer)
            .ok_or_else(|| Error::Model {
                message: format!(
                    "DeepSeek-V4 CUDA execution layer {execution_layer} is not prepared"
                ),
            })
    }

    #[cfg(feature = "cuda")]
    fn prepared_mtp_stage(&self, stage: usize) -> Result<&DeepSeekV4CudaPreparedProposalStage> {
        self.prepared_image()?
            .mtp
            .as_ref()
            .and_then(|mtp| mtp.layers.get(stage))
            .ok_or_else(|| Error::Model {
                message: format!("DeepSeek-V4 CUDA Proposal stage {stage} is out of range"),
            })
    }

    fn prepared_attention_sink(
        &self,
        layer: usize,
    ) -> Result<&ferrule_backend::cuda::context::CudaF32Buffer> {
        Ok(&self.prepared_layer(layer)?.attention_sink)
    }

    fn prepared_compressor(
        &self,
        layer: usize,
        compressor: DeepSeekV4CudaCompressor,
    ) -> Result<&DeepSeekV4CudaPreparedCompressor> {
        let prepared = self.prepared_layer(layer)?;
        let resources = match compressor {
            DeepSeekV4CudaCompressor::Main => prepared.main_compressor.as_ref(),
            DeepSeekV4CudaCompressor::Indexer => prepared.indexer_compressor.as_ref(),
        };
        resources.ok_or_else(|| Error::Model {
            message: format!(
                "DeepSeek-V4 layer {layer} CUDA {compressor:?} compressor resources are unavailable"
            ),
        })
    }

    fn prepared_linear(
        &self,
        layer: usize,
        linear: DeepSeekV4CudaLinear,
    ) -> Result<&DeepSeekV4CudaPreparedLinear> {
        let prepared = self.prepared_layer(layer)?;
        let handle = prepared.linear(linear);
        handle.ok_or_else(|| Error::Model {
            message: format!(
                "DeepSeek-V4 layer {layer} CUDA {linear:?} ({:?}) linear is unavailable",
                linear.operation()
            ),
        })
    }

    /// Upload a norm weight once for reuse with `rms_norm_from_device`.
    pub(crate) fn upload_norm_weight(
        &self,
        weight: &[f32],
    ) -> Result<ferrule_backend::cuda::context::CudaF32Buffer> {
        self.ops.upload_norm_weight(weight)
    }

    pub(crate) fn rms_norm_layer_rows_device_into(
        &self,
        layer: usize,
        norm: DeepSeekV4CudaLayerNorm,
        input: &ferrule_backend::cuda::context::CudaF32Buffer,
        rows: usize,
        eps: f32,
        output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        let prepared = self.prepared_layer(layer)?;
        let weight = match norm {
            DeepSeekV4CudaLayerNorm::Query => &prepared.query_norm,
            DeepSeekV4CudaLayerNorm::KeyValue => &prepared.key_value_norm,
        };
        if rows == 0 || input.len() != rows * weight.len() || output.len() != input.len() {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 layer {layer} CUDA RMS rows length mismatch: rows={rows} input={} output={} weight={}",
                    input.len(),
                    output.len(),
                    weight.len()
                ),
            });
        }
        self.ops
            .rms_norm_rows_from_device_into(input, rows, weight, eps, output)
    }

    pub(crate) fn rms_norm_compressor_rows_device_into(
        &self,
        layer: usize,
        compressor: DeepSeekV4CudaCompressor,
        input: &ferrule_backend::cuda::context::CudaF32Buffer,
        rows: usize,
        eps: f32,
        output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        let weight = &self.prepared_compressor(layer, compressor)?.norm;
        if rows == 0 || input.len() != rows * weight.len() || output.len() != input.len() {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 layer {layer} CUDA {compressor:?} compressor RMS rows length mismatch: rows={rows} input={} output={} weight={}",
                    input.len(),
                    output.len(),
                    weight.len()
                ),
            });
        }
        self.ops
            .rms_norm_rows_from_device_into(input, rows, weight, eps, output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hc_pre_rmsnorm_fp8_into<'a>(
        &self,
        layer: usize,
        stage: DeepSeekV4CudaHcStage,
        state: &ferrule_backend::cuda::context::CudaF32Buffer,
        tokens: usize,
        config: HyperConnectionConfig,
        hidden: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        normalized: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        mix: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        workspace: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        pre: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        post: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        comb: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        packed: &'a mut ferrule_backend::cuda::context::CudaFp8ActivationPack,
    ) -> Result<ferrule_backend::cuda::context::CudaPreparedFp8Activation<'a>> {
        #[cfg(feature = "cuda")]
        self.require_inference_operation(
            layer,
            match stage {
                DeepSeekV4CudaHcStage::Attention => KernelOperation::AttentionHcPre,
                DeepSeekV4CudaHcStage::FeedForward => KernelOperation::FeedForwardHcPre,
            },
        )?;
        let prepared = self.prepared_layer(layer)?;
        let (hw, rms_weight) = match stage {
            DeepSeekV4CudaHcStage::Attention => (&prepared.attention_hc, &prepared.attention_norm),
            DeepSeekV4CudaHcStage::FeedForward => {
                (&prepared.feed_forward_hc, &prepared.feed_forward_norm)
            }
        };
        self.ops.hc_pre_rmsnorm_fp8_into(
            state,
            &hw.function_row_major,
            &hw.scale,
            &hw.base,
            rms_weight,
            mix,
            workspace,
            tokens,
            config.hc_mult,
            config.hidden_size,
            config.sinkhorn_iters,
            config.eps,
            config.norm_eps,
            config.norm_eps,
            hidden,
            normalized,
            pre,
            post,
            comb,
            packed,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hc_post_from_device_into(
        &self,
        hidden: &ferrule_backend::cuda::context::CudaF32Buffer,
        residual: &ferrule_backend::cuda::context::CudaF32Buffer,
        split_post: &ferrule_backend::cuda::context::CudaF32Buffer,
        split_comb: &ferrule_backend::cuda::context::CudaF32Buffer,
        tokens: usize,
        config: HyperConnectionConfig,
        output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.ops.hc_post_from_device_into(
            hidden,
            residual,
            split_post,
            split_comb,
            tokens,
            config.hc_mult,
            config.hidden_size,
            output,
        )
    }

    pub(crate) fn hc_head_from_device_into(
        &self,
        state: &ferrule_backend::cuda::context::CudaF32Buffer,
        tokens: usize,
        output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        let image = self.prepared_image()?;
        self.ops.hc_head_from_device_into(
            state,
            &image.hc_head.function_row_major,
            &image.hc_head.scale,
            &image.hc_head.base,
            tokens,
            image.hc_config.hc_mult,
            image.hc_config.hidden_size,
            image.hc_config.eps,
            image.hc_config.norm_eps,
            output,
        )
    }

    /// Multi-row output RMS norm.  `input` must contain `rows * hidden_size`
    /// elements; `output` is written in-place with the same layout.
    pub(crate) fn rms_norm_output_rows_device_into(
        &self,
        input: &ferrule_backend::cuda::context::CudaF32Buffer,
        rows: usize,
        eps: f32,
        output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        let image = self.prepared_image()?;
        let weight = &image.output_norm;
        self.ops
            .rms_norm_rows_from_device_into(input, rows, weight, eps, output)
    }

    fn ensure_router_token_ids(&mut self, token_ids: &[u32], hash_rows: usize) -> Result<()> {
        match self.router_token_ids.get_mut(&token_ids.len()) {
            Some(cached) => {
                self.ops
                    .update_dsv4_router_token_ids(token_ids, hash_rows, cached)?;
            }
            None => {
                let device = self.ops.dsv4_router_token_ids(token_ids, hash_rows)?;
                self.router_token_ids.insert(token_ids.len(), device);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dsv4_router_topk_routes_from_device_logits(
        &mut self,
        layer: usize,
        logits: &ferrule_backend::cuda::context::CudaF32Buffer,
        token_ids: &[u32],
        router: &RouterWeights,
        router_policy: &ExpertRouterPolicy,
        indices_dev: &mut ferrule_backend::cuda::context::CudaI32Buffer,
        weights_dev: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        if router_policy.score_function != RouterScoreFunction::SqrtSoftplus
            || !router_policy.normalize_non_softmax_weights
        {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 CUDA router does not support policy {:?}",
                    router_policy
                ),
            });
        }
        let tokens = token_ids.len();
        let experts = router.weight.format.out_features();
        let top_k = router_policy.top_k;
        if tokens == 0 || top_k == 0 || top_k > 64 || top_k > experts || experts > 512 {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 CUDA router unsupported shape: tokens={tokens} experts={experts} top_k={top_k}"
                ),
            });
        }
        if !router_policy.route_scale.is_finite() {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 CUDA router route_scale must be finite, got {}",
                    router_policy.route_scale
                ),
            });
        }

        match router_policy.selection {
            RouterSelectionPolicy::ScoreTopK => {
                let bias = self.prepared_layer(layer)?.router_bias.as_ref();
                self.ops
                    .dsv4_router_topk_sqrt_softplus_rows_from_device_into(
                        logits,
                        bias,
                        tokens,
                        experts,
                        top_k,
                        router_policy.route_scale,
                        indices_dev,
                        weights_dev,
                    )?;
            }
            RouterSelectionPolicy::Hash => {
                self.ensure_router_token_ids(token_ids, router.hash_rows)?;
                let hash_table = self
                    .prepared_layer(layer)?
                    .router_hash_table
                    .as_ref()
                    .ok_or_else(|| Error::Internal {
                        message: format!(
                            "DeepSeek-V4 layer {layer} prepared hash router table is missing"
                        ),
                    })?;
                let token_ids_dev = self
                    .router_token_ids
                    .get(&tokens)
                    .expect("token ids inserted above");
                self.ops
                    .dsv4_router_hash_sqrt_softplus_rows_from_device_into(
                        logits,
                        token_ids_dev,
                        hash_table,
                        tokens,
                        experts,
                        top_k,
                        router_policy.route_scale,
                        indices_dev,
                        weights_dev,
                    )?;
            }
        }

        Ok(())
    }

    fn take_compact_i32_mirror(
        &mut self,
        compact_len: usize,
    ) -> Result<ferrule_backend::cuda::context::CudaI32HostMirror> {
        if let Some(buffer) = self
            .compact_i32_mirrors
            .get_mut(&compact_len)
            .and_then(Vec::pop)
        {
            return Ok(buffer);
        }
        self.ops.i32_host_mirror(&vec![0; compact_len])
    }

    fn restore_compact_i32_mirror(
        &mut self,
        compact_len: usize,
        buffer: ferrule_backend::cuda::context::CudaI32HostMirror,
    ) {
        self.compact_i32_mirrors
            .entry(compact_len)
            .or_default()
            .push(buffer);
    }

    fn arm_compact_download_completion(&mut self, pending: &mut CudaCompactI32Download) {
        if pending.callback_armed {
            return;
        }
        pending.callback_armed = self
            .ops
            .notify_control_stream(completion_notify_callback(self.completion_hub.clone()))
            .is_ok();
        if !pending.callback_armed {
            self.completion_hub.notify();
        }
    }

    fn begin_dsv4_router_route_download(
        &mut self,
        indices_dev: &ferrule_backend::cuda::context::CudaI32Buffer,
        weights_dev: &ferrule_backend::cuda::context::CudaF32Buffer,
        tokens: usize,
        top_k: usize,
    ) -> Result<CudaCompactI32Download> {
        let route_count = tokens.checked_mul(top_k).ok_or_else(|| Error::Internal {
            message: "CUDA router route count overflow".into(),
        })?;
        let compact_len = route_count.checked_mul(2).ok_or_else(|| Error::Internal {
            message: "CUDA compact route size overflow".into(),
        })?;
        let mut mirror = self.take_compact_i32_mirror(compact_len)?;
        self.ops.pack_i32_f32_pairs_into(
            indices_dev,
            weights_dev,
            mirror.device_mut_invalidate_host(),
            route_count,
        )?;
        let produced = self.ops.record_compute_event()?;
        let download = self
            .ops
            .begin_i32_host_mirror_download_after(&mut mirror, &produced)?;
        let mut pending = CudaCompactI32Download {
            compact_len,
            mirror: Some(mirror),
            download,
            callback_armed: false,
        };
        self.arm_compact_download_completion(&mut pending);
        Ok(pending)
    }

    fn poll_compact_i32_download(
        &mut self,
        pending: &mut CudaCompactI32Download,
    ) -> Result<Option<Vec<i32>>> {
        let mirror = pending.mirror.as_mut().ok_or_else(|| Error::Internal {
            message: "CUDA compact download mirror was already restored".into(),
        })?;
        let compact = self
            .ops
            .poll_i32_host_mirror_download(mirror, &pending.download)?;
        match compact_download_poll_action(compact.is_some(), pending.callback_armed) {
            CudaCompactDownloadPollAction::Consume => {}
            CudaCompactDownloadPollAction::Wait => return Ok(None),
            CudaCompactDownloadPollAction::ArmAndWait => {
                self.arm_compact_download_completion(pending);
                return Ok(None);
            }
        }
        Ok(compact)
    }

    fn restore_compact_i32_download(&mut self, pending: &mut CudaCompactI32Download) -> Result<()> {
        let mirror = pending.mirror.take().ok_or_else(|| Error::Internal {
            message: "CUDA compact download mirror was already restored".into(),
        })?;
        self.restore_compact_i32_mirror(pending.compact_len, mirror);
        Ok(())
    }

    pub(crate) fn begin_output_head_topk_rows_from_execution_image(
        &mut self,
        hidden: &ferrule_backend::cuda::context::CudaF32Buffer,
        batch_rows: usize,
        top_k: usize,
    ) -> Result<DeepSeekV4OutputHeadTopKDownload> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (hidden, batch_rows, top_k);
            Err(Error::Model { message:
                "DeepSeek-V4 packed CUDA resident output head requires building with CUTLASS support; host artifact fallback is disabled"
                    .into(),
             })
        }
        #[cfg(feature = "cuda")]
        {
            let (vocab, hidden_width) = {
                let image = self.prepared_image()?;
                let ferrule_backend::cuda::context::CudaArtifactLinearShape::Bf16Bytes {
                    out_features,
                    in_features,
                } = image.output_head.shape()
                else {
                    return Err(Error::Model {
                        message: format!(
                            "DeepSeek-V4 resident output head requires BF16 storage, got {:?}",
                            image.output_head.shape()
                        ),
                    });
                };
                (out_features, in_features)
            };
            let expected_hidden =
                batch_rows
                    .checked_mul(hidden_width)
                    .ok_or_else(|| Error::Model {
                        message: "DeepSeek-V4 output-head batch input size overflow".into(),
                    })?;
            if hidden.len() != expected_hidden {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 output-head input mismatch: expected {batch_rows}x{hidden_width}={expected_hidden}, got {}",
                        hidden.len()
                    ),
                });
            }
            if batch_rows == 0 || top_k == 0 || top_k > vocab || top_k > 40 {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 output-head top-k requires rows>0 and k in 1..={}, got rows={batch_rows} k={top_k}",
                        vocab.min(40)
                    ),
                });
            }
            if vocab > i32::MAX as usize {
                return Err(Error::Model {
                    message: format!(
                        "DeepSeek-V4 output-head vocab {vocab} exceeds the device i32 token-id limit"
                    ),
                });
            }

            let logits_key = (batch_rows, vocab);
            if !self.output_head_logits.contains_key(&logits_key) {
                let logits_len = batch_rows.checked_mul(vocab).ok_or_else(|| Error::Model {
                    message: "DeepSeek-V4 full-vocab logits workspace overflow".into(),
                })?;
                self.output_head_logits
                    .insert(logits_key, self.ops.zero_f32_buffer(logits_len)?);
            }
            let output_len = batch_rows.checked_mul(top_k).ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 output-head top-k workspace overflow".into(),
            })?;
            if !self.output_head_indices.contains_key(&output_len) {
                self.output_head_indices
                    .insert(output_len, self.ops.zero_i32_buffer(output_len)?);
                self.output_head_values
                    .insert(output_len, self.ops.zero_f32_buffer(output_len)?);
            }
            if !self
                .output_head_linear_workspaces
                .contains_key(&hidden.len())
            {
                self.output_head_linear_workspaces.insert(
                    hidden.len(),
                    self.ops
                        .artifact_linear_workspace(batch_rows, hidden_width)?,
                );
            }

            let compact_len = output_len.checked_mul(2).ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 compact output-head workspace overflow".into(),
            })?;
            let mut mirror = self.take_compact_i32_mirror(compact_len)?;
            self.metrics.output_head_calls = self.metrics.output_head_calls.saturating_add(1);
            self.metrics.output_head_rows = self
                .metrics
                .output_head_rows
                .saturating_add(batch_rows as u64);
            let topk_start = profile_start(self.profile);
            let image =
                self.execution_image.as_ref().ok_or_else(|| {
                    Error::Internal { message:
                    "DeepSeek-V4 CUDA execution image was not compiled before output-head execution"
                        .into(),
                 }
                })?;
            let logits = self
                .output_head_logits
                .get_mut(&logits_key)
                .expect("output-head logits workspace initialized above");
            let indices = self
                .output_head_indices
                .get_mut(&output_len)
                .expect("output-head indices workspace initialized above");
            let values = self
                .output_head_values
                .get_mut(&output_len)
                .expect("output-head values workspace initialized above");
            let linear_workspace = self
                .output_head_linear_workspaces
                .get_mut(&hidden.len())
                .expect("output-head linear workspace initialized above");
            self.ops
                .artifact_linear_rows_from_device_into_with_scratch(
                    &image.output_head,
                    hidden,
                    batch_rows,
                    logits,
                    linear_workspace,
                )?;
            self.ops.topk_vocab_rows_from_device_into(
                logits, batch_rows, vocab, top_k, indices, values,
            )?;
            self.ops.pack_i32_f32_pairs_into(
                indices,
                values,
                mirror.device_mut_invalidate_host(),
                output_len,
            )?;
            let produced = self.ops.record_compute_event()?;
            let download = self
                .ops
                .begin_i32_host_mirror_download_after(&mut mirror, &produced)?;
            let mut compact = CudaCompactI32Download {
                compact_len,
                mirror: Some(mirror),
                download,
                callback_armed: false,
            };
            self.arm_compact_download_completion(&mut compact);
            Ok(DeepSeekV4OutputHeadTopKDownload {
                rows: batch_rows,
                top_k,
                vocab,
                compact,
                topk_start,
            })
        }
    }

    pub(crate) fn poll_output_head_topk_rows_from_execution_image(
        &mut self,
        pending: &mut DeepSeekV4OutputHeadTopKDownload,
    ) -> Result<Option<Vec<Vec<TokenLogit>>>> {
        let Some(compact) = self.poll_compact_i32_download(&mut pending.compact)? else {
            return Ok(None);
        };
        self.restore_compact_i32_download(&mut pending.compact)?;
        record_profile_duration(
            &mut self.metrics.output_head_topk_us,
            pending.topk_start.take(),
        );
        decode_output_head_topk_rows(&compact, pending.rows, pending.top_k, pending.vocab).map(Some)
    }

    pub(crate) fn poll_output_head_topk_cancel_ready(
        &mut self,
        pending: &mut DeepSeekV4OutputHeadTopKDownload,
    ) -> Result<bool> {
        if pending.compact.mirror.is_none() {
            return Ok(true);
        }
        if self
            .poll_compact_i32_download(&mut pending.compact)?
            .is_none()
        {
            return Ok(false);
        }
        self.restore_compact_i32_download(&mut pending.compact)?;
        Ok(true)
    }

    fn arm_compute_quiescence_completion(&mut self, pending: &mut DeepSeekV4CudaComputeQuiescence) {
        if pending.callback_armed {
            return;
        }
        pending.callback_armed = self
            .ops
            .notify_compute_stream(completion_notify_callback(self.completion_hub.clone()))
            .is_ok();
        if !pending.callback_armed {
            self.completion_hub.notify();
        }
    }

    pub(crate) fn begin_compute_quiescence(&mut self) -> Result<DeepSeekV4CudaComputeQuiescence> {
        let mut pending = DeepSeekV4CudaComputeQuiescence {
            event: self.ops.record_compute_event()?,
            callback_armed: false,
        };
        self.arm_compute_quiescence_completion(&mut pending);
        Ok(pending)
    }

    pub(crate) fn poll_compute_quiescence(
        &mut self,
        pending: &mut DeepSeekV4CudaComputeQuiescence,
    ) -> Result<bool> {
        if pending.event.is_complete()? {
            return Ok(true);
        }
        self.arm_compute_quiescence_completion(pending);
        Ok(false)
    }

    pub(crate) fn upload_linear(
        &self,
        linear: &LinearWeight,
    ) -> Result<ferrule_backend::cuda::context::CudaArtifactLinearHandle> {
        match linear.format {
            LinearWeightFormat::F32 {
                out_features,
                in_features,
            } => self
                .ops
                .upload_f32_linear(&linear.weight.bytes, out_features, in_features),
            LinearWeightFormat::Bf16 {
                out_features,
                in_features,
            } => self
                .ops
                .upload_bf16_linear(&linear.weight.bytes, out_features, in_features),
            LinearWeightFormat::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m,
                block_k,
            } => {
                let scale = linear.scale.as_ref().ok_or_else(|| Error::Model {
                    message: format!(
                        "artifact linear {:?} CUDA FP8 weight is missing E8M0 scale tensor",
                        linear.role
                    ),
                })?;
                self.ops.upload_fp8_e4m3_e8m0_linear(
                    &linear.weight.bytes,
                    &scale.bytes,
                    out_features,
                    in_features,
                    block_m,
                    block_k,
                )
            }
            LinearWeightFormat::Fp4E2M1PackedWithE8M0Scale {
                out_features,
                in_features,
                block_size: 32,
            } => {
                let scale = linear.scale.as_ref().ok_or_else(|| Error::Model {
                    message: format!(
                        "artifact linear {:?} CUDA FP4 weight is missing E8M0 scale tensor",
                        linear.role
                    ),
                })?;
                self.ops.upload_fp4_e2m1_e8m0_linear(
                    &linear.weight.bytes,
                    &scale.bytes,
                    out_features,
                    in_features,
                    self.managed_experts,
                )
            }
            LinearWeightFormat::Fp4E2M1PackedWithE8M0Scale { block_size, .. } => {
                Err(Error::Model {
                    message: format!(
                        "artifact linear {:?} CUDA FP4 block_size {block_size} is unsupported (expected 32)",
                        linear.role
                    ),
                })
            }
        }
    }

    pub(crate) fn rms_norm_heads_from_device_into(
        &self,
        input: &ferrule_backend::cuda::context::CudaF32Buffer,
        heads: usize,
        head_dim: usize,
        eps: f32,
        output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.ops
            .rms_norm_heads_from_device_into(input, heads, head_dim, eps, output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_routed_moe_prefill_batch_from_device_into(
        &mut self,
        layer: usize,
        input_dev: &ferrule_backend::cuda::context::CudaF32Buffer,
        shared_input_fp8: &ferrule_backend::cuda::context::CudaPreparedFp8Activation<'_>,
        shared_hidden_f32: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        shared_hidden_fp8: &mut ferrule_backend::cuda::context::CudaFp8ActivationPack,
        token_ids: &[u32],
        row_to_sequence: Option<&[usize]>,
        sequence_phases: Option<&[ForwardPhase]>,
        router: &RouterWeights,
        router_policy: &ExpertRouterPolicy,
        shared_expert: &SwiGluFfnPayload,
        router_logits_dev: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        router_indices_dev: &mut ferrule_backend::cuda::context::CudaI32Buffer,
        router_weights_dev: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        linear_workspace: &mut ferrule_backend::cuda::context::CudaArtifactLinearWorkspace,
        output_dev: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<DeepSeekV4CudaPackedMoeContinuation> {
        let tokens = token_ids.len();
        if tokens == 0 {
            return Err(Error::Model {
                message: "CUDA routed MoE prefill batch requires at least one token".into(),
            });
        }
        let (row_to_sequence, sequence_phases) = match (row_to_sequence, sequence_phases) {
            (None, None) => (None, None),
            (Some(sequences), Some(phases))
                if sequences.len() == tokens
                    && !phases.is_empty()
                    && sequences.iter().all(|sequence| *sequence < phases.len()) =>
            {
                (
                    Some(sequences.to_vec()),
                    Some(
                        phases
                            .iter()
                            .map(|phase| match phase {
                                ForwardPhase::Prefill => ExpertAccessPhase::Prefill,
                                ForwardPhase::Decode => ExpertAccessPhase::Decode,
                            })
                            .collect(),
                    ),
                )
            }
            _ => {
                return Err(Error::Model {
                    message: "CUDA routed MoE packed row/sequence metadata mismatch".into(),
                });
            }
        };
        let hidden_size = router.weight.format.in_features();
        let expected_input = tokens
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Model {
                message: "CUDA routed MoE prefill input size overflow".into(),
            })?;
        if input_dev.len() != expected_input {
            return Err(Error::Model {
                message: format!(
                    "CUDA routed MoE prefill device input length mismatch: input={} expected {} tokens x {} hidden",
                    input_dev.len(),
                    tokens,
                    hidden_size
                ),
            });
        }
        if output_dev.len() != input_dev.len() {
            return Err(Error::Model {
                message: format!(
                    "CUDA routed MoE prefill output length mismatch: output={} expected {}",
                    output_dev.len(),
                    input_dev.len()
                ),
            });
        }

        let stage_start = profile_start(self.profile);
        self.linear_rows_from_device_into(
            layer,
            DeepSeekV4CudaLinear::Router,
            input_dev,
            tokens,
            router_logits_dev,
            linear_workspace,
        )?;
        self.dsv4_router_topk_routes_from_device_logits(
            layer,
            router_logits_dev,
            token_ids,
            router,
            router_policy,
            router_indices_dev,
            router_weights_dev,
        )?;
        let route_download = self.begin_dsv4_router_route_download(
            router_indices_dev,
            router_weights_dev,
            tokens,
            router_policy.top_k,
        )?;
        record_profile_duration(&mut self.metrics.moe_router_us, stage_start);

        let stage_start = profile_start(self.profile);
        #[cfg(feature = "cuda")]
        self.require_inference_operation(layer, KernelOperation::SharedFfn)?;
        let prepared = self.prepared_layer(layer)?;
        self.ops.artifact_shared_ffn_into(
            &prepared.shared_gate.handle,
            &prepared.shared_up.handle,
            &prepared.shared_down.handle,
            shared_input_fp8,
            shared_hidden_f32,
            shared_hidden_fp8,
            tokens,
            1.0,
            shared_expert.swiglu_limit,
            output_dev,
            false,
        )?;
        record_profile_duration(&mut self.metrics.moe_shared_us, stage_start);

        #[cfg(feature = "cuda")]
        self.require_inference_operation(layer, KernelOperation::GroupedFp4Moe)?;
        let routes_per_token = router_policy.top_k;
        let route_count = tokens
            .checked_mul(routes_per_token)
            .ok_or_else(|| Error::Internal {
                message: "CUDA routed MoE route count overflow".into(),
            })?;
        if route_count > i32::MAX as usize || tokens > i32::MAX as usize {
            return Err(Error::Internal {
                message: format!(
                    "CUDA routed MoE exceeds i32 metadata ABI: tokens={tokens} routes={route_count}"
                ),
            });
        }
        let expected_route_output =
            route_count
                .checked_mul(hidden_size)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA routed MoE route output overflow".into(),
                })?;

        Ok(DeepSeekV4CudaPackedMoeContinuation {
            layer,
            tokens,
            hidden_size,
            routes_per_token,
            route_count,
            expected_route_output,
            route_state: CudaPackedMoeRouteState::RoutesPending(route_download),
            routes_by_token: Vec::new(),
            unique_experts: Vec::new(),
            sequence_phases,
            row_to_sequence,
            swiglu_limit: shared_expert.swiglu_limit,
            resident_slot_capacity: None,
            route_plan_capacity: route_count,
            next_chunk_start: 0,
            current_chunk: None,
            retained_leases: Vec::new(),
            input_prepared: false,
            expected_intermediate: None,
            streaming_steps: Vec::new(),
            compute_start: profile_start(self.profile),
        })
    }

    /// Poll the already-submitted router exactly once and expose the first logical
    /// expert window when the compact route result is ready. This never submits
    /// expert compute and therefore requires no lease authority.
    pub(crate) fn prime_routed_moe_prefill_batch(
        &mut self,
        continuation: &mut DeepSeekV4CudaPackedMoeContinuation,
    ) -> Result<()> {
        if self.poll_packed_moe_routes(continuation)? {
            self.prepare_next_packed_moe_chunk(continuation)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resume_routed_moe_prefill_batch_from_device_into(
        &mut self,
        mut continuation: DeepSeekV4CudaPackedMoeContinuation,
        leases: ResidencyLeaseSet,
        input_dev: &ferrule_backend::cuda::context::CudaF32Buffer,
        router_indices: &ferrule_backend::cuda::context::CudaI32Buffer,
        router_weights: &ferrule_backend::cuda::context::CudaF32Buffer,
        route_plan: &mut Option<ferrule_backend::cuda::context::CudaExpertGroupRoutePlan>,
        route_output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        output_dev: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<DeepSeekV4CudaPackedMoeProgress> {
        match packed_moe_resume_action(
            continuation.routes_pending(),
            continuation.current_chunk.is_some(),
            leases.len(),
        )? {
            CudaPackedMoeResumeAction::PollSubmittedRoute => {
                self.prime_routed_moe_prefill_batch(&mut continuation)?;
                return Ok(DeepSeekV4CudaPackedMoeProgress::Waiting(continuation));
            }
            CudaPackedMoeResumeAction::SubmitExpertWindow => {}
        }

        let progress = self.resume_routed_moe_prefill_inner(
            &mut continuation,
            leases,
            input_dev,
            router_indices,
            router_weights,
            route_plan,
            route_output,
            output_dev,
        );
        match progress {
            Ok(Some(events)) => Ok(DeepSeekV4CudaPackedMoeProgress::Complete {
                events,
                leases: std::mem::take(&mut continuation.retained_leases),
            }),
            Ok(None) => {
                debug_assert!(
                    !continuation.pending_experts().is_empty(),
                    "a waiting packed MoE continuation must expose expert work"
                );
                Ok(DeepSeekV4CudaPackedMoeProgress::Waiting(continuation))
            }
            Err(error) => Err(error),
        }
    }

    pub(crate) fn poll_routed_moe_cancel_ready(
        &mut self,
        continuation: &mut DeepSeekV4CudaPackedMoeContinuation,
    ) -> Result<bool> {
        self.poll_packed_moe_routes(continuation)
    }

    pub(crate) fn cancel_routed_moe_prefill_batch(
        &mut self,
        mut continuation: DeepSeekV4CudaPackedMoeContinuation,
    ) -> Result<()> {
        if continuation.routes_pending() {
            return Err(Error::Execution {
                message: "CUDA packed MoE route download is still active during cancellation"
                    .into(),
            });
        }
        continuation.current_chunk = None;
        continuation.retained_leases.clear();
        Ok(())
    }

    fn poll_packed_moe_routes(
        &mut self,
        continuation: &mut DeepSeekV4CudaPackedMoeContinuation,
    ) -> Result<bool> {
        let state = std::mem::replace(
            &mut continuation.route_state,
            CudaPackedMoeRouteState::RoutesReady,
        );
        let CudaPackedMoeRouteState::RoutesPending(mut pending) = state else {
            return Ok(true);
        };
        let compact = match self.poll_compact_i32_download(&mut pending) {
            Ok(Some(compact)) => compact,
            Ok(None) => {
                continuation.route_state = CudaPackedMoeRouteState::RoutesPending(pending);
                return Ok(false);
            }
            Err(error) => {
                continuation.route_state = CudaPackedMoeRouteState::RoutesPending(pending);
                return Err(error);
            }
        };
        self.restore_compact_i32_download(&mut pending)?;

        let routes_by_token = decode_compact_router_routes(
            &compact,
            continuation.tokens,
            continuation.routes_per_token,
        )?;
        if std::env::var("FERRULE_DEBUG_ROUTER_LAYER")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            == Some(continuation.layer)
        {
            for (token, routes) in routes_by_token.iter().enumerate() {
                eprintln!(
                    "router layer={} token={} routes={:?}",
                    continuation.layer,
                    token,
                    routes
                        .iter()
                        .map(|route| (route.expert, route.weight))
                        .collect::<Vec<_>>()
                );
            }
        }
        for routes in &routes_by_token {
            self.metrics.expert_selected = self
                .metrics
                .expert_selected
                .saturating_add(routes.len() as u64);
        }
        let mut unique_experts = routes_by_token
            .iter()
            .flat_map(|routes| routes.iter().map(|route| route.expert))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if unique_experts.is_empty() {
            return Err(Error::Internal {
                message: "CUDA routed MoE selected no experts".into(),
            });
        }
        unique_experts.shrink_to_fit();
        self.metrics.expert_unique_selected = self
            .metrics
            .expert_unique_selected
            .saturating_add(unique_experts.len() as u64);
        continuation.routes_by_token = routes_by_token;
        continuation.unique_experts = unique_experts;
        Ok(true)
    }

    fn prepare_next_packed_moe_chunk(
        &mut self,
        continuation: &mut DeepSeekV4CudaPackedMoeContinuation,
    ) -> Result<()> {
        if continuation.routes_pending()
            || continuation.current_chunk.is_some()
            || continuation.next_chunk_start == continuation.unique_experts.len()
        {
            return Ok(());
        }
        let subsystem = self
            .expert_subsystem
            .as_ref()
            .ok_or_else(|| Error::Execution {
                message: "DeepSeek-V4 shared expert subsystem is not attached".into(),
            })?;
        if continuation.resident_slot_capacity.is_none() {
            let resident_slot_capacity = subsystem
                .layer_slot_capacity(continuation.layer)?
                .clamp(1, DSV4_EXPERT_TABLE_CAPACITY);
            let resident_experts = subsystem.resident_experts_for_layer(continuation.layer)?;
            continuation
                .unique_experts
                .sort_by_key(|expert| (!resident_experts.contains(expert), *expert));
            continuation.resident_slot_capacity = Some(resident_slot_capacity);
        }
        let resident_slot_capacity = continuation
            .resident_slot_capacity
            .expect("resident slot capacity initialized above");
        let end = continuation
            .next_chunk_start
            .saturating_add(resident_slot_capacity)
            .min(continuation.unique_experts.len());
        let selected = continuation.unique_experts[continuation.next_chunk_start..end].to_vec();
        let selected_ids = selected
            .iter()
            .copied()
            .map(|expert| ExpertId::new(continuation.layer, expert))
            .collect();
        continuation.current_chunk = Some(CudaPackedMoeChunk {
            selected: selected.clone(),
            unresolved_experts: selected,
            streaming: ExpertStreamingStep {
                layer: continuation.layer,
                selected: selected_ids,
                prefetched: Vec::new(),
                loads: Vec::new(),
                evictions: Vec::new(),
            },
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn resume_routed_moe_prefill_inner(
        &mut self,
        continuation: &mut DeepSeekV4CudaPackedMoeContinuation,
        leases: ResidencyLeaseSet,
        input_dev: &ferrule_backend::cuda::context::CudaF32Buffer,
        router_indices: &ferrule_backend::cuda::context::CudaI32Buffer,
        router_weights: &ferrule_backend::cuda::context::CudaF32Buffer,
        route_plan: &mut Option<ferrule_backend::cuda::context::CudaExpertGroupRoutePlan>,
        route_output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        output_dev: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<Option<Vec<DeepSeekV4SequenceMoeAccessEvent>>> {
        if input_dev.len() != continuation.tokens * continuation.hidden_size
            || output_dev.len() != input_dev.len()
            || route_output.len() != continuation.expected_route_output
        {
            return Err(Error::Model {
                message: "CUDA resumable routed MoE buffers changed while suspended".into(),
            });
        }
        let subsystem = self
            .expert_subsystem
            .clone()
            .ok_or_else(|| Error::Execution {
                message: "DeepSeek-V4 shared expert subsystem is not attached".into(),
            })?;
        let mut chunk =
            continuation.current_chunk.take().ok_or_else(|| {
                Error::Execution { message:
                "DeepSeek-V4 packed MoE has no declared expert dependency; refusing resume replay"
                    .into(),
             }
            })?;
        let dispatch = subsystem.with_validated_published_experts(
            continuation.layer,
            &chunk.selected,
            &leases,
            |table, first_frame| {
                self.submit_packed_moe_chunk(
                    continuation,
                    &chunk,
                    table,
                    first_frame,
                    input_dev,
                    router_indices,
                    router_weights,
                    route_plan,
                    route_output,
                )
            },
        );
        if let Err(error) = dispatch {
            continuation.current_chunk = Some(chunk);
            return Err(error);
        }

        chunk.unresolved_experts.clear();
        continuation.next_chunk_start = continuation
            .next_chunk_start
            .saturating_add(chunk.selected.len());
        continuation.streaming_steps.push(chunk.streaming);
        continuation.retained_leases.push(leases);

        if continuation.next_chunk_start == continuation.unique_experts.len() {
            self.ops.reduce_moe_route_outputs_ranked(
                route_output,
                continuation.tokens,
                continuation.routes_per_token,
                continuation.hidden_size,
                output_dev,
            )?;
            record_profile_duration(
                &mut self.metrics.moe_compute_submit_us,
                continuation.compute_start.take(),
            );
            return Ok(Some(self.finish_packed_moe_events(continuation)));
        }

        self.prepare_next_packed_moe_chunk(continuation)?;
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_packed_moe_chunk(
        &mut self,
        continuation: &mut DeepSeekV4CudaPackedMoeContinuation,
        chunk: &CudaPackedMoeChunk,
        table: &ferrule_backend::cuda::context::CudaExpertSlotTable,
        first_frame: &CudaExpertFrame,
        input_dev: &ferrule_backend::cuda::context::CudaF32Buffer,
        router_indices: &ferrule_backend::cuda::context::CudaI32Buffer,
        router_weights: &ferrule_backend::cuda::context::CudaF32Buffer,
        route_plan: &mut Option<ferrule_backend::cuda::context::CudaExpertGroupRoutePlan>,
        route_output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        if chunk.selected.is_empty() {
            return Err(Error::Internal {
                message: "CUDA routed MoE empty expert window".into(),
            });
        }
        let intermediate_size = first_frame.intermediate_size();
        if first_frame.input_size() != continuation.hidden_size
            || first_frame.output_size() != continuation.hidden_size
        {
            return Err(Error::Model {
                message: format!(
                    "CUDA grouped MoE hidden mismatch: expert input={} output={} expected={}",
                    first_frame.input_size(),
                    first_frame.output_size(),
                    continuation.hidden_size
                ),
            });
        }
        if let Some(expected) = continuation.expected_intermediate {
            if intermediate_size != expected {
                return Err(Error::Model {
                    message: format!(
                        "CUDA routed MoE intermediate mismatch: got {intermediate_size} expected {expected}"
                    ),
                });
            }
        } else {
            continuation.expected_intermediate = Some(intermediate_size);
        }
        let resident_slot_capacity = continuation
            .resident_slot_capacity
            .expect("resident slot capacity initialized before expert execution");
        let plan_needs_init = route_plan
            .as_ref()
            .map(|plan| {
                !plan.matches(
                    resident_slot_capacity,
                    continuation.route_plan_capacity,
                    continuation.tokens,
                    continuation.hidden_size,
                    intermediate_size,
                    continuation.hidden_size,
                )
            })
            .unwrap_or(true);
        if plan_needs_init {
            *route_plan = Some(self.ops.expert_group_route_plan(
                resident_slot_capacity,
                continuation.route_plan_capacity,
                continuation.tokens,
                continuation.hidden_size,
                intermediate_size,
                continuation.hidden_size,
            )?);
            continuation.input_prepared = false;
        }
        if !continuation.input_prepared {
            let plan = route_plan
                .as_mut()
                .expect("expert group route plan initialized above");
            self.ops.prepare_expert_group_route_input_from_device(
                input_dev,
                continuation.tokens,
                continuation.hidden_size,
                plan,
            )?;
            self.ops.begin_expert_group_route_invocation(
                continuation.routes_per_token,
                plan,
                route_output,
            )?;
            continuation.input_prepared = true;
        }
        let plan = route_plan
            .as_mut()
            .expect("expert group route plan initialized above");
        self.ops.prepare_expert_group_route_plan(
            table,
            router_indices,
            router_weights,
            continuation.route_count,
            continuation.routes_per_token,
            plan,
        )?;
        self.ops.grouped_fp4_moe_from_prepared_plan(
            table,
            continuation.routes_per_token,
            continuation.swiglu_limit,
            plan,
            route_output,
        )
    }

    fn finish_packed_moe_events(
        &mut self,
        continuation: &DeepSeekV4CudaPackedMoeContinuation,
    ) -> Vec<DeepSeekV4SequenceMoeAccessEvent> {
        let Some(row_to_sequence) = continuation.row_to_sequence.as_deref() else {
            self.moe_access_events
                .push(ExpertBatchAccessEvent::from_routes_by_token(
                    continuation.layer,
                    ExpertAccessPhase::Prefill,
                    continuation.tokens,
                    &continuation.routes_by_token,
                    &continuation.streaming_steps,
                ));
            return Vec::new();
        };
        ExpertBatchAccessEvent::from_packed_routes_by_sequence(
            continuation.layer,
            continuation
                .sequence_phases
                .as_deref()
                .expect("packed phases validated at begin"),
            row_to_sequence,
            &continuation.routes_by_token,
            &continuation.streaming_steps,
        )
        .into_iter()
        .map(|(sequence_index, event)| DeepSeekV4SequenceMoeAccessEvent {
            sequence_index,
            event,
        })
        .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn paged_window_sparse_attention_rows_into(
        &mut self,
        query: &ferrule_backend::cuda::context::CudaF32Buffer,
        row_kv_lens: &ferrule_backend::cuda::context::CudaI32Buffer,
        topk: &mut ferrule_backend::cuda::context::CudaI32Buffer,
        rows: usize,
        layer: usize,
        spec: SparseAttentionSpec,
        output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        if row_kv_lens.len() != rows {
            return Err(Error::Model {
                message: format!(
                    "packed window attention row metadata mismatch: visible_lens={} rows={rows}",
                    row_kv_lens.len()
                ),
            });
        }
        self.ops
            .fill_recent_rows_into(row_kv_lens, rows, spec.topk, topk)?;
        let active = self.active_paged_kv.as_ref().ok_or_else(|| Error::Model {
            message: "packed window attention requires an active paged binding".into(),
        })?;
        if active.row_sequence_ids_device.len() != rows {
            return Err(Error::Model {
                message: format!(
                    "packed window attention selector rows mismatch: got {} expected {rows}",
                    active.row_sequence_ids_device.len()
                ),
            });
        }
        let pool = self.kv_page_pool.as_ref().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA physical KV pool is not configured".into(),
        })?;
        let plane = pool.plane_storage(0).ok_or_else(|| Error::Model {
            message: "window KV plane is missing".into(),
        })?;
        let descriptor = &pool.planes()[0];
        let layout = ferrule_backend::cuda::PagedSparseAttentionLayout {
            batch_size: rows,
            tokens_per_sequence: 1,
            heads: spec.heads,
            head_dim: spec.head_dim,
            topk: spec.topk,
            page_tokens: active.page_tokens,
            elements_per_token: descriptor.elements_per_token,
            layer_index: layer,
            layer_count: active.layer_count,
            softmax_scale: spec.softmax_scale,
        };
        let explicit_selection_layout = layout.explicit_selection_layout(true)?;
        let mut workspace = match self.hybrid_mla_explicit_selection_workspaces.remove(&rows) {
            Some(workspace) => workspace,
            None => self.ops.hybrid_mla_explicit_selection_workspace(
                ferrule_backend::cuda::cutlass::HybridMlaExplicitSelectionLayout {
                    selected_width:
                        ferrule_backend::cuda::cutlass::HYBRID_MLA_EXPLICIT_SELECTION_MAXIMUM_WIDTH,
                    ..explicit_selection_layout
                },
            )?,
        };
        let result = self
            .ops
            .paged_sparse_attention_selected_rows_from_device_into(
                query,
                plane,
                &active.block_slots_device,
                &active.block_offsets_device,
                &active.kv_len_device,
                &active.row_sequence_ids_device,
                row_kv_lens,
                topk,
                self.prepared_attention_sink(layer)?,
                layout,
                &mut workspace,
                output,
            );
        self.hybrid_mla_explicit_selection_workspaces
            .insert(rows, workspace);
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dual_plane_paged_sparse_attention_rows_into(
        &mut self,
        query: &ferrule_backend::cuda::context::CudaF32Buffer,
        topk: &ferrule_backend::cuda::context::CudaI32Buffer,
        selectors: &ferrule_backend::cuda::context::CudaI32Buffer,
        row_kv_lens: &ferrule_backend::cuda::context::CudaI32Buffer,
        row_second_kv_lens: &ferrule_backend::cuda::context::CudaI32Buffer,
        second_sequence_kv_lens: &[i32],
        rows: usize,
        layer: usize,
        spec: SparseAttentionSpec,
        output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        {
            let active = self.active_paged_kv.as_mut().ok_or_else(|| Error::Model {
                message: "packed sparse attention requires an active paged binding".into(),
            })?;
            if active.row_sequence_ids_device.len() != rows
                || row_kv_lens.len() != rows
                || row_second_kv_lens.len() != rows
                || second_sequence_kv_lens.len() != active.sequence_count
            {
                return Err(Error::Model {
                    message: format!(
                        "packed sparse attention metadata mismatch: selectors={} first_row_lens={} second_row_lens={} second_sequence_lens={} expected_rows={rows} expected_sequences={}",
                        active.row_sequence_ids_device.len(),
                        row_kv_lens.len(),
                        row_second_kv_lens.len(),
                        second_sequence_kv_lens.len(),
                        active.sequence_count,
                    ),
                });
            }
            self.ops.update_i32_host_mirror(
                second_sequence_kv_lens,
                &mut active.second_kv_len_device,
            )?;
        }
        let active = self
            .active_paged_kv
            .as_ref()
            .expect("validated active paged binding remains installed");
        let pool = self.kv_page_pool.as_ref().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA physical KV pool is not configured".into(),
        })?;
        let first = pool.plane_storage(0).ok_or_else(|| Error::Model {
            message: "window KV plane is missing".into(),
        })?;
        let second = pool.plane_storage(1).ok_or_else(|| Error::Model {
            message: "compressed KV plane is missing".into(),
        })?;
        let first_descriptor = &pool.planes()[0];
        let second_descriptor = &pool.planes()[1];
        let layout = ferrule_backend::cuda::DualPlanePagedSparseAttentionLayout {
            base: ferrule_backend::cuda::PagedSparseAttentionLayout {
                batch_size: rows,
                tokens_per_sequence: 1,
                heads: spec.heads,
                head_dim: spec.head_dim,
                topk: spec.topk,
                page_tokens: active.page_tokens,
                elements_per_token: first_descriptor.elements_per_token,
                layer_index: layer,
                layer_count: active.layer_count,
                softmax_scale: spec.softmax_scale,
            },
            second_elements_per_token: second_descriptor.elements_per_token,
        };
        let explicit_selection_layout = layout.explicit_selection_layout(true)?;
        let mut workspace = match self.hybrid_mla_explicit_selection_workspaces.remove(&rows) {
            Some(workspace) => workspace,
            None => self.ops.hybrid_mla_explicit_selection_workspace(
                ferrule_backend::cuda::cutlass::HybridMlaExplicitSelectionLayout {
                    selected_width:
                        ferrule_backend::cuda::cutlass::HYBRID_MLA_EXPLICIT_SELECTION_MAXIMUM_WIDTH,
                    ..explicit_selection_layout
                },
            )?,
        };
        let result = self
            .ops
            .dual_plane_paged_sparse_attention_selected_rows_from_device_into(
                query,
                first,
                second,
                &active.block_slots_device,
                &active.block_offsets_device,
                &active.kv_len_device,
                &active.second_kv_len_device,
                &active.row_sequence_ids_device,
                row_kv_lens,
                row_second_kv_lens,
                topk,
                selectors,
                self.prepared_attention_sink(layer)?,
                layout,
                &mut workspace,
                output,
            );
        self.hybrid_mla_explicit_selection_workspaces
            .insert(rows, workspace);
        result
    }

    /// One-launch grouped output-A -> BF16 latent -> output-B MLA transaction.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mla_output_rows_from_device_into(
        &self,
        context: &ferrule_backend::cuda::context::CudaF32Buffer,
        rows: usize,
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
        latent: &mut ferrule_backend::cuda::context::CudaBf16Buffer,
        workspace: &mut ferrule_backend::cuda::context::CudaArtifactLinearWorkspace,
        output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (context, rows, cfg, layer, latent, workspace, output);
            return Err(Error::Model {
                message: "CUDA MLA output execution requires the `cutlass` feature".into(),
            });
        }
        #[cfg(feature = "cuda")]
        {
            self.require_inference_operation(layer, KernelOperation::MlaOutput)?;
            let prepared = self.prepared_layer(layer)?;
            self.ops.artifact_mla_output_into(
                context,
                rows,
                &prepared.output_a.handle,
                &prepared.output_b.handle,
                cfg.o_groups,
                cfg.output_group_input_dim(),
                cfg.o_lora_rank,
                latent,
                workspace,
                output,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn query_a_kv_from_prepared_fp8_into(
        &self,
        layer: usize,
        activation: &ferrule_backend::cuda::context::CudaPreparedFp8Activation<'_>,
        query_a_output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        key_value_output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (layer, activation, query_a_output, key_value_output);
            return Err(Error::Model {
                message: "CUDA QueryA+KV execution requires the `cutlass` feature".into(),
            });
        }
        #[cfg(feature = "cuda")]
        {
            let query_a = self.prepared_linear(layer, DeepSeekV4CudaLinear::QueryA)?;
            let key_value = self.prepared_linear(layer, DeepSeekV4CudaLinear::KeyValue)?;
            self.require_inference_operation(layer, KernelOperation::MlaQueryAKv)?;
            self.ops.artifact_fp8_query_a_kv_into(
                &query_a.handle,
                &key_value.handle,
                activation,
                query_a_output,
                key_value_output,
            )
        }
    }

    pub(crate) fn compressor_rows_from_device_into(
        &self,
        layer: usize,
        compressor: DeepSeekV4CudaCompressor,
        input: &ferrule_backend::cuda::context::CudaF32Buffer,
        rows: usize,
        kv_output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        gate_output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (layer, compressor, input, rows, kv_output, gate_output);
            return Err(Error::Model {
                message: "CUDA compressor execution requires the `cutlass` feature".into(),
            });
        }
        #[cfg(feature = "cuda")]
        {
            let operation = match compressor {
                DeepSeekV4CudaCompressor::Main => KernelOperation::MainCompressorProjection,
                DeepSeekV4CudaCompressor::Indexer => KernelOperation::IndexerCompressorProjection,
            };
            self.require_inference_operation(layer, operation)?;
            let prepared = self.prepared_compressor(layer, compressor)?;
            self.ops.artifact_bf16_compressor_into(
                &prepared.kv.handle,
                &prepared.gate.handle,
                input,
                rows,
                kv_output,
                gate_output,
            )
        }
    }

    pub(crate) fn linear_rows_from_device_into(
        &self,
        layer: usize,
        linear: DeepSeekV4CudaLinear,
        input: &ferrule_backend::cuda::context::CudaF32Buffer,
        rows: usize,
        output: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        workspace: &mut ferrule_backend::cuda::context::CudaArtifactLinearWorkspace,
    ) -> Result<()> {
        let prepared = self.prepared_linear(layer, linear)?;
        let in_features = prepared.handle.shape().in_features();
        if rows == 0 || input.len() != rows * in_features {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 layer {layer} CUDA {linear:?} rows input length mismatch: rows={rows} expected {}, got {}",
                    rows * in_features,
                    input.len()
                ),
            });
        }

        #[cfg(feature = "cuda")]
        if linear == DeepSeekV4CudaLinear::QueryB {
            self.require_inference_operation(layer, KernelOperation::MlaQueryB)?;
            return self
                .ops
                .artifact_fp8_projection_rows_from_device_into_with_scratch(
                    &prepared.handle,
                    input,
                    rows,
                    output,
                    workspace,
                );
        }
        self.ops.artifact_linear_rows_from_device_into_with_scratch(
            &prepared.handle,
            input,
            rows,
            output,
            workspace,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compressor_recurrent_append_into(
        &self,
        layer: usize,
        compressor: DeepSeekV4CudaCompressor,
        state: &mut Option<ferrule_backend::cuda::CudaCompressorRecurrentState>,
        needs_reset: &mut bool,
        projected_kv: &ferrule_backend::cuda::context::CudaF32Buffer,
        projected_score: &ferrule_backend::cuda::context::CudaF32Buffer,
        position: usize,
        ratio: usize,
        head_dim: usize,
        out_dim: usize,
        overlap: bool,
        compressed: &mut ferrule_backend::cuda::context::CudaF32Buffer,
    ) -> Result<bool> {
        let ape = &self.prepared_compressor(layer, compressor)?.ape;
        if state.is_none() {
            *state = Some(
                self.ops
                    .create_compressor_recurrent_state(ratio, head_dim, out_dim, overlap)?,
            );
        }
        let state = state.as_mut().expect("created above");
        if *needs_reset {
            self.ops.reset_compressor_recurrent_state(state)?;
            *needs_reset = false;
        }
        let boundary = self.ops.compressor_recurrent_append_projected(
            state,
            projected_kv,
            projected_score,
            ape,
            position,
        )?;
        if boundary {
            self.ops
                .compressor_recurrent_boundary_into(state, compressed)?;
        }
        Ok(boundary)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_topk_indices_paged_indexer_rows_into(
        &mut self,
        query: &ferrule_backend::cuda::context::CudaF32Buffer,
        weights: &ferrule_backend::cuda::context::CudaF32Buffer,
        positions: &ferrule_backend::cuda::context::CudaI32Buffer,
        window_lens: &ferrule_backend::cuda::context::CudaI32Buffer,
        compressed_lens: &ferrule_backend::cuda::context::CudaI32Buffer,
        layer: usize,
        window_size: usize,
        index_topk: usize,
        compress_ratio: usize,
        direct_compressed: bool,
        index_heads: usize,
        index_head_dim: usize,
        weight_scale: f32,
        logical_indices: &mut ferrule_backend::cuda::context::CudaI32Buffer,
        plane_selectors: &mut ferrule_backend::cuda::context::CudaI32Buffer,
    ) -> Result<()> {
        let rows = positions.len();
        if window_lens.len() != rows || compressed_lens.len() != rows {
            return Err(Error::Model {
                message: "packed decode top-k row metadata is inconsistent".into(),
            });
        }
        let active = self.active_paged_kv.as_ref().ok_or_else(|| Error::Model {
            message: "packed decode top-k requires an active paged binding".into(),
        })?;
        if active.row_sequence_ids_device.len() != rows {
            return Err(Error::Model {
                message: "packed decode top-k selector row count is inconsistent".into(),
            });
        }
        let pool = self.kv_page_pool.as_ref().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA physical KV pool is not configured".into(),
        })?;
        let indexer_plane = pool.plane_storage(2).ok_or_else(|| Error::Model {
            message: "indexer KV plane is missing".into(),
        })?;
        self.ops
            .dsv4_decode_topk_indices_paged_indexer_rows_from_device_into(
                query,
                weights,
                indexer_plane,
                &active.block_slots_device,
                &active.block_offsets_device,
                &active.row_sequence_ids_device,
                positions,
                window_lens,
                compressed_lens,
                rows,
                window_size,
                index_topk,
                compress_ratio,
                direct_compressed,
                index_heads,
                index_head_dim,
                active.page_tokens,
                layer,
                active.layer_count,
                weight_scale,
                logical_indices,
                plane_selectors,
            )
    }

    pub(crate) fn paged_scatter_rows_from_device(
        &mut self,
        plane: usize,
        layer: usize,
        values: &ferrule_backend::cuda::context::CudaF32Buffer,
        positions: &ferrule_backend::cuda::context::CudaI32Buffer,
        mask: Option<&ferrule_backend::cuda::context::CudaI32Buffer>,
        row_dim: usize,
    ) -> Result<()> {
        let active = self.active_paged_kv.as_ref().ok_or_else(|| Error::Model {
            message: "packed paged KV scatter requires an active binding".into(),
        })?;
        if active.row_sequence_ids_device.len() != positions.len()
            || values.len() != positions.len() * row_dim
        {
            return Err(Error::Model {
                message: "packed paged KV scatter row metadata is inconsistent".into(),
            });
        }
        if mask.is_some_and(|mask| mask.len() != positions.len()) {
            return Err(Error::Model {
                message: "packed paged KV scatter mask length is inconsistent".into(),
            });
        }
        let pool = self.kv_page_pool.as_mut().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 CUDA physical KV pool is not configured".into(),
        })?;
        let descriptor = pool.planes().get(plane).ok_or_else(|| Error::Model {
            message: format!("paged KV plane {plane} is missing"),
        })?;
        if descriptor.elements_per_token != row_dim || descriptor.layer_count != active.layer_count
        {
            return Err(Error::Model {
                message: format!(
                    "paged KV plane {plane} layout mismatch: row_dim={row_dim}/{} layers={}/{}",
                    descriptor.elements_per_token, active.layer_count, descriptor.layer_count
                ),
            });
        }
        let layout = ferrule_backend::cuda::PagedPlaneLayout {
            page_tokens: active.page_tokens,
            elements_per_token: row_dim,
            layer_index: layer,
            layer_count: active.layer_count,
        };
        self.ops.paged_plane_scatter_selected_rows_from_device(
            values,
            positions,
            &active.block_slots_device,
            &active.block_offsets_device,
            &active.row_sequence_ids_device,
            mask,
            pool.plane_storage_mut(plane)
                .expect("validated paged KV plane"),
            layout,
        )
    }

    pub(crate) fn ensure_rope_tables_with_params(
        &mut self,
        name: &str,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        required_positions: usize,
    ) -> Result<()> {
        validate_rope_table_request(name, rope_dim, rope, required_positions)?;
        if let Some(table) = self.rope_tables.get(name) {
            validate_rope_table_identity(name, table, rope_dim, rope)?;
            if table.capacity >= required_positions {
                return Ok(());
            }
        }

        let capacity = rope_table_capacity(required_positions)?;
        let rd2 = rope_dim / 2;
        let elements = capacity.checked_mul(rd2).ok_or_else(|| {
            Error::Model { message: format!(
                "DeepSeek-V4 RoPE table '{name}' element count overflow: capacity={capacity} rope_dim={rope_dim}"
            ) }
        })?;
        let mut cos = Vec::new();
        cos.try_reserve_exact(elements).map_err(|error| {
            Error::Model { message: format!(
                "DeepSeek-V4 RoPE cosine table '{name}' host allocation failed for {elements} elements: {error}"
            ) }
        })?;
        cos.resize(elements, 0.0f32);
        let mut sin = Vec::new();
        sin.try_reserve_exact(elements).map_err(|error| {
            Error::Model { message: format!(
                "DeepSeek-V4 RoPE sine table '{name}' host allocation failed for {elements} elements: {error}"
            ) }
        })?;
        sin.resize(elements, 0.0f32);
        for position in 0..capacity {
            for pair in 0..rd2 {
                let freq = yarn_frequency(pair, rope_dim, rope);
                let angle = position as f32 * freq;
                let (s, c) = angle.sin_cos();
                cos[position * rd2 + pair] = c;
                sin[position * rd2 + pair] = s;
            }
        }

        let cos = self.ops.upload_f32_buffer(&cos)?;
        let sin = self.ops.upload_f32_buffer(&sin)?;
        self.rope_tables.insert(
            name.to_string(),
            CudaRopeTable {
                rope_dim,
                rope,
                capacity,
                cos,
                sin,
            },
        );
        Ok(())
    }

    pub(crate) fn require_rope_tables(
        &self,
        name: &str,
        rope_dim: usize,
        required_positions: usize,
    ) -> Result<()> {
        if required_positions == 0 {
            return Err(Error::Model {
                message: format!("DeepSeek-V4 RoPE table '{name}' requires at least one position"),
            });
        }
        let table = self.rope_tables.get(name).ok_or_else(|| Error::Model {
            message: format!("DeepSeek-V4 RoPE table '{name}' is not prepared"),
        })?;
        if table.rope_dim != rope_dim {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 RoPE table '{name}' dimension mismatch: cached={} requested={rope_dim}",
                    table.rope_dim
                ),
            });
        }
        validate_rope_table_capacity(name, table, required_positions)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope_tail_rows_from_device(
        &mut self,
        name: &str,
        qk: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        start_position: u32,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        self.rope_tail_rows_strided_from_device(
            name,
            qk,
            start_position,
            1,
            rows,
            heads,
            head_dim,
            rope_dim,
            inverse,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope_tail_rows_indexed_from_device(
        &mut self,
        name: &str,
        qk: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        positions: &ferrule_backend::cuda::context::CudaI32Buffer,
        max_position: usize,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        if positions.is_empty() || heads == 0 || rope_dim == 0 {
            return Ok(());
        }
        let required_positions = max_position.checked_add(1).ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 indexed RoPE position overflow".into(),
        })?;
        self.require_rope_tables(name, rope_dim as usize, required_positions)?;
        let table = self
            .rope_tables
            .get(name)
            .expect("rope tables required immediately above");
        self.ops.rope_tail_rows_indexed_from_device(
            qk,
            &table.cos,
            &table.sin,
            positions,
            positions.len() as u32,
            heads,
            head_dim,
            rope_dim,
            inverse,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope_tail_rows_strided_from_device(
        &mut self,
        name: &str,
        qk: &mut ferrule_backend::cuda::context::CudaF32Buffer,
        start_position: u32,
        position_stride: u32,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        if rows == 0 || heads == 0 || rope_dim == 0 {
            return Ok(());
        }
        let last_offset = (rows as usize - 1)
            .checked_mul(position_stride as usize)
            .ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 RoPE row-stride overflow".into(),
            })?;
        let required_positions = (start_position as usize)
            .checked_add(last_offset)
            .and_then(|position| position.checked_add(1))
            .ok_or_else(|| Error::Model {
                message: "DeepSeek-V4 RoPE position overflow".into(),
            })?;
        self.require_rope_tables(name, rope_dim as usize, required_positions)?;
        let table = self
            .rope_tables
            .get(name)
            .expect("rope tables required immediately above");
        self.ops.rope_tail_rows_strided_from_device(
            qk,
            &table.cos,
            &table.sin,
            start_position,
            position_stride,
            rows,
            heads,
            head_dim,
            rope_dim,
            inverse,
        )
    }
}

#[cfg(test)]
mod tests {

    use super::super::helpers::apply_rotary_tail_scaled;
    use super::*;

    #[test]
    fn router_hash_table_identity_rejects_same_values_with_different_shape() {
        let table = vec![0, 1, 2, 3, 4, 5];
        let identity = CudaRouterHashTableIdentity::new(7, &table, 2, 3, 8, 2).unwrap();

        identity.validate_request(7, &table, 2, 3, 8, 2).unwrap();
        let error = identity
            .validate_request(7, &table, 3, 2, 8, 2)
            .expect_err("same flattened table with a different shape must be rejected");
        assert!(error.to_string().contains("changed after device upload"));
    }

    #[test]
    fn router_hash_table_identity_validates_flattened_shape() {
        let error = CudaRouterHashTableIdentity::new(2, &[0, 1, 2], 2, 2, 4, 1)
            .expect_err("malformed flattened shape must be rejected");
        assert!(error.to_string().contains("shape mismatch"));

        let error = validate_router_hash_table_shape(2, &[], usize::MAX, 2)
            .expect_err("overflowing shape must be rejected");
        assert!(error.to_string().contains("shape overflows usize"));
    }

    #[test]
    fn compact_download_state_machine_waits_without_blocking() {
        assert_eq!(
            compact_download_poll_action(false, false),
            CudaCompactDownloadPollAction::ArmAndWait
        );
        assert_eq!(
            compact_download_poll_action(false, true),
            CudaCompactDownloadPollAction::Wait
        );
        assert_eq!(
            compact_download_poll_action(true, false),
            CudaCompactDownloadPollAction::Consume
        );
        assert_eq!(
            compact_download_poll_action(true, true),
            CudaCompactDownloadPollAction::Consume
        );
    }

    #[test]
    fn compact_router_routes_decode_after_async_completion() {
        let first_weight = 0.25f32;
        let second_weight = -1.5f32;
        let compact = [
            3,
            first_weight.to_bits() as i32,
            1,
            second_weight.to_bits() as i32,
        ];
        let routes = decode_compact_router_routes(&compact, 2, 1).unwrap();

        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0][0].expert, 3);
        assert_eq!(routes[0][0].weight, first_weight);
        assert_eq!(routes[1][0].expert, 1);
        assert_eq!(routes[1][0].weight, second_weight);
    }

    #[test]
    fn compact_output_head_rows_decode_after_async_completion() {
        let first = 1.25f32;
        let second = -0.0f32;
        let third = f32::NEG_INFINITY;
        let fourth = 9.5f32;
        let compact = [
            3,
            first.to_bits() as i32,
            1,
            second.to_bits() as i32,
            0,
            third.to_bits() as i32,
            4,
            fourth.to_bits() as i32,
        ];

        let rows = decode_output_head_topk_rows(&compact, 2, 2, 5).unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0].token_id, 3);
        assert_eq!(rows[0][0].logit.to_bits(), first.to_bits());
        assert_eq!(rows[0][1].token_id, 1);
        assert_eq!(rows[0][1].logit.to_bits(), second.to_bits());
        assert_eq!(rows[1][0].token_id, 0);
        assert_eq!(rows[1][0].logit, third);
        assert_eq!(rows[1][1].token_id, 4);
        assert_eq!(rows[1][1].logit, fourth);
    }

    #[test]
    fn compact_output_head_rows_reject_invalid_payloads() {
        let length_error = decode_output_head_topk_rows(&[0, 0], 2, 1, 4)
            .expect_err("truncated compact output-head payload must be rejected");
        assert!(length_error.to_string().contains("result mismatch"));

        let negative_error = decode_output_head_topk_rows(&[-1, 0], 1, 1, 4)
            .expect_err("negative token ids must be rejected");
        assert!(negative_error.to_string().contains("negative token index"));

        let bounds_error = decode_output_head_topk_rows(&[4, 0], 1, 1, 4)
            .expect_err("out-of-vocab token ids must be rejected");
        assert!(bounds_error.to_string().contains("exceeds vocab"));

        let zero_k_error = decode_output_head_topk_rows(&[0, 0], 1, 0, 4)
            .expect_err("k=0 must reject a non-empty compact payload");
        assert!(zero_k_error.to_string().contains("must be empty"));
        assert_eq!(decode_output_head_topk_rows(&[], 2, 0, 4).unwrap().len(), 2);
    }

    #[test]
    fn compact_router_routes_reject_invalid_pending_payloads() {
        let length_error = decode_compact_router_routes(&[0, 0], 2, 1)
            .expect_err("truncated compact route payload must be rejected");
        assert!(length_error.to_string().contains("length mismatch"));

        let expert_error = decode_compact_router_routes(&[-1, 0], 1, 1)
            .expect_err("negative expert indices must be rejected");
        assert!(expert_error.to_string().contains("negative expert index"));
    }

    #[test]
    fn packed_moe_resume_rejects_replay_without_a_declared_dependency() {
        assert_eq!(
            packed_moe_resume_action(true, false, 0).unwrap(),
            CudaPackedMoeResumeAction::PollSubmittedRoute
        );
        assert_eq!(
            packed_moe_resume_action(false, true, 1).unwrap(),
            CudaPackedMoeResumeAction::SubmitExpertWindow
        );
        let early = packed_moe_resume_action(true, false, 1)
            .unwrap_err()
            .to_string();
        assert!(early.contains("before its route dependency completed"));
        let replay = packed_moe_resume_action(false, false, 0)
            .unwrap_err()
            .to_string();
        assert!(replay.contains("refusing resume replay"));
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn packed_paged_scatter_rows_handles_mixed_compressor_and_page_boundaries() -> Result<()> {
        let mut operators = DeepSeekV4CudaOperatorCache::new(
            &DeepSeekV4ExecutionPolicy::default(),
            ExpertMemoryPolicy::default(),
        )?;
        let planes = [
            ferrule_common::execution::KvPlaneDescriptor {
                name: "window".into(),
                elements_per_token: 2,
                layer_count: 1,
            },
            ferrule_common::execution::KvPlaneDescriptor {
                name: "main".into(),
                elements_per_token: 2,
                layer_count: 1,
            },
            ferrule_common::execution::KvPlaneDescriptor {
                name: "indexer".into(),
                elements_per_token: 1,
                layer_count: 1,
            },
        ];
        operators.kv_page_pool = Some(ferrule_backend::cuda::CudaKvPagePool::new(
            &operators.ops,
            &planes,
            2,
            4,
        )?);
        let block_slots = vec![0, 1, 2, 3];
        let block_offsets = vec![0, 2, 4];
        let kv_lens = vec![2, 3];
        operators.active_paged_kv = Some(ActivePagedKvBinding {
            physical_block_slots: block_slots.clone(),
            block_slots_device: operators.ops.i32_host_mirror(&block_slots)?,
            block_offsets_device: operators.ops.i32_host_mirror(&block_offsets)?,
            kv_len_device: operators.ops.i32_host_mirror(&kv_lens)?,
            second_kv_len_device: operators.ops.i32_host_mirror(&[0, 0])?,
            row_sequence_ids_device: operators.ops.i32_host_mirror(&[0, 1])?,
            page_tokens: 2,
            layer_count: 1,
            sequence_count: 2,
        });

        let window = operators.ops.upload_f32_buffer(&[10.0, 11.0, 20.0, 21.0])?;
        let window_positions = operators.ops.upload_i32_buffer(&[1, 2])?;
        operators.paged_scatter_rows_from_device(0, 0, &window, &window_positions, None, 2)?;
        let main = operators.ops.upload_f32_buffer(&[30.0, 31.0, 40.0, 41.0])?;
        let compressed_positions = operators.ops.upload_i32_buffer(&[0, 0])?;
        let main_mask = operators.ops.upload_i32_buffer(&[1, 0])?;
        operators.paged_scatter_rows_from_device(
            1,
            0,
            &main,
            &compressed_positions,
            Some(&main_mask),
            2,
        )?;
        let indexer = operators.ops.upload_f32_buffer(&[50.0, 60.0])?;
        let indexer_mask = operators.ops.upload_i32_buffer(&[0, 1])?;
        operators.paged_scatter_rows_from_device(
            2,
            0,
            &indexer,
            &compressed_positions,
            Some(&indexer_mask),
            1,
        )?;
        operators.ops.sync_stream()?;

        let pool = operators.kv_page_pool.as_ref().expect("configured pool");
        let window = operators
            .ops
            .download_f32_buffer(pool.plane_storage(0).expect("window plane"))?;
        let main = operators
            .ops
            .download_f32_buffer(pool.plane_storage(1).expect("main plane"))?;
        let indexer = operators
            .ops
            .download_f32_buffer(pool.plane_storage(2).expect("indexer plane"))?;
        assert_eq!(&window[2..4], &[10.0, 11.0]);
        assert_eq!(&window[12..14], &[20.0, 21.0]);
        assert_eq!(&main[0..2], &[30.0, 31.0]);
        assert_eq!(&main[8..10], &[0.0, 0.0]);
        assert_eq!(indexer[0], 0.0);
        assert_eq!(indexer[4], 60.0);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn rope_table_grows_across_4096_boundary() -> Result<()> {
        let mut operators = DeepSeekV4CudaOperatorCache::new(
            &DeepSeekV4ExecutionPolicy::default(),
            ExpertMemoryPolicy::default(),
        )?;
        let name = "rope_boundary_test";
        let rope_dim = 4;
        let head_dim = 6;
        let rope = DeepSeekV4RopeParams {
            theta: 10_000.0,
            original_seq_len: 4096,
            factor: 2.0,
            beta_fast: 32,
            beta_slow: 1,
        };

        operators.ensure_rope_tables_with_params(name, rope_dim, rope, 4096)?;
        assert_eq!(operators.rope_tables[name].capacity, 4096);
        operators.ensure_rope_tables_with_params(name, rope_dim, rope, 4097)?;
        assert_eq!(operators.rope_tables[name].capacity, 8192);

        for position in [4095usize, 4096] {
            let input = [0.25f32, -0.5, 1.0, 2.0, -3.0, 4.0];
            let mut expected = input;
            apply_rotary_tail_scaled(&mut expected, 1, head_dim, rope_dim, position, rope, false)?;

            let mut actual_device = operators.ops.upload_f32_buffer(&input)?;
            operators.require_rope_tables(name, rope_dim, position + 1)?;
            let table = operators
                .rope_tables
                .get(name)
                .expect("table prepared above");
            operators.ops.rope_tail_from_device(
                &mut actual_device,
                &table.cos,
                &table.sin,
                position as u32,
                1,
                head_dim as u32,
                rope_dim as u32,
                false,
            )?;
            let actual = operators.ops.download_f32_buffer(&actual_device)?;
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= 1.0e-5,
                    "RoPE boundary mismatch at position={position} element={index}: actual={actual} expected={expected}"
                );
            }
        }

        let mismatch = operators
            .ensure_rope_tables_with_params(
                name,
                rope_dim,
                DeepSeekV4RopeParams {
                    theta: 20_000.0,
                    ..rope
                },
                4097,
            )
            .expect_err("same-name RoPE parameter mismatch must be rejected");
        assert!(mismatch.to_string().contains("identity mismatch"));

        Ok(())
    }
}
