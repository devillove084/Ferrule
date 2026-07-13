//! DeepSeek-V4 CUDA operator cache: device-resident weights, KV cache, MoE handles.

#![cfg(feature = "cuda")]

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ferrule_common::execution::{KvCowReplacement, KvLayoutSchema, KvPageId};
use ferrule_common::{
    Error, ExpertInstallIntent, ExpertInstallPrepareOutcome, ExpertKey, ExpertLease,
    ExpertResidencyControl, ExpertResidencyGrant, ExpertSlotBinding, MemoryPoolLimits,
    MemoryPoolStats, OwnerMemoryLru, PreparedExpertInstall, Result,
};

use crate::TensorRole;
use crate::artifact::binding::RouterArtifactPayload;
use crate::artifact::linear::{
    ArtifactActivationQuantization, ArtifactLinearFormat, ArtifactLinearPayload,
};
use crate::artifact::linear::{artifact_linear_cache_key, artifact_linear_row_cache_key};
use crate::artifact::tensor::{ArtifactTensorReader, ArtifactTensorSlice};
use crate::attention_backend::SparseAttentionSpec;
use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::{
    HyperConnectionConfig, HyperConnectionHeadWeights, HyperConnectionWeights,
};

use crate::moe::prediction::{ExpertAccessPhase, ExpertBatchAccessEvent};
use crate::moe::routed::RoutedMoeStepOutput;
use crate::moe::routing::{
    ExpertRoute, ExpertRouterPolicy, RouterScoreFunction, RouterSelectionPolicy,
};
use crate::moe::streaming::{
    AsyncHostStagedExpertLoader, ExpertComputeBundle, ExpertEvictRequest, ExpertId,
    ExpertLinearFormat, ExpertLinearPayload, ExpertLoadReason, ExpertLoadRequest, ExpertMatrixKind,
    ExpertMemoryPolicy, ExpertSourceCatalog, ExpertStorageTier, ExpertStreamingReader,
    ExpertStreamingStep, HostStagedExpertCache, read_experts_concurrent,
};
use crate::runner::TokenLogit;

use super::config::{DeepSeekV4AttentionConfig, DeepSeekV4RopeParams};
use super::helpers::{check_linear, rank_logits_desc, yarn_frequency};
use super::operators::DeepSeekV4OperatorRuntimeCounters;
use super::prepared::{DeepSeekV4ExecutionPolicy, DeepSeekV4KvLayoutSchema};
use super::sequence::DeepSeekV4PagedKvBinding;

const DSV4_ROPE_TABLE_MIN_CAPACITY: usize = 4096;
const DSV4_EXPERT_TABLE_CAPACITY: usize = 512;

struct CudaRopeTable {
    rope_dim: usize,
    rope: DeepSeekV4RopeParams,
    capacity: usize,
    cos: ferrule_cuda::context::CudaF32Buffer,
    sin: ferrule_cuda::context::CudaF32Buffer,
}

fn validate_rope_table_request(
    name: &str,
    rope_dim: usize,
    rope: DeepSeekV4RopeParams,
    required_positions: usize,
) -> Result<()> {
    if rope_dim == 0 || !rope_dim.is_multiple_of(2) {
        return Err(Error::Model(format!(
            "DeepSeek-V4 RoPE table '{name}' requires a positive even dimension, got {rope_dim}"
        )));
    }
    if !rope.theta.is_finite() || rope.theta <= 0.0 {
        return Err(Error::Model(format!(
            "DeepSeek-V4 RoPE table '{name}' requires finite positive theta, got {}",
            rope.theta
        )));
    }
    if !rope.factor.is_finite() || rope.factor <= 0.0 {
        return Err(Error::Model(format!(
            "DeepSeek-V4 RoPE table '{name}' requires finite positive factor, got {}",
            rope.factor
        )));
    }
    if required_positions == 0 {
        return Err(Error::Model(format!(
            "DeepSeek-V4 RoPE table '{name}' requires at least one position"
        )));
    }
    let max_addressable_positions = u64::from(u32::MAX) + 1;
    if u64::try_from(required_positions).unwrap_or(u64::MAX) > max_addressable_positions {
        return Err(Error::Model(format!(
            "DeepSeek-V4 RoPE table '{name}' requires {required_positions} positions, exceeding the CUDA u32 position limit of {max_addressable_positions}"
        )));
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
        return Err(Error::Model(format!(
            "DeepSeek-V4 RoPE table '{name}' identity mismatch: cached dim={} params={:?}, requested dim={rope_dim} params={rope:?}",
            table.rope_dim, table.rope
        )));
    }
    Ok(())
}

fn validate_rope_table_capacity(
    name: &str,
    table: &CudaRopeTable,
    required_positions: usize,
) -> Result<()> {
    if table.capacity < required_positions {
        return Err(Error::Model(format!(
            "DeepSeek-V4 RoPE table '{name}' capacity {} is smaller than required position count {required_positions}; prepare/grow it before launching CUDA work",
            table.capacity
        )));
    }
    Ok(())
}

fn rope_table_capacity(required_positions: usize) -> Result<usize> {
    required_positions
        .max(DSV4_ROPE_TABLE_MIN_CAPACITY)
        .checked_next_power_of_two()
        .ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 RoPE table capacity overflow for {required_positions} positions"
            ))
        })
}

/// Device-resident recurrent compressor state owned by one DeepSeek-V4 layer.
///
/// Historical KV values live exclusively in the runtime-managed page pool.
#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct DeepSeekV4CudaSequenceKvState {
    pub(crate) main_compressor_recurrent: Option<ferrule_cuda::CudaCompressorRecurrentState>,
    pub(crate) main_compressor_needs_reset: bool,
    pub(crate) indexer_compressor_recurrent: Option<ferrule_cuda::CudaCompressorRecurrentState>,
    pub(crate) indexer_compressor_needs_reset: bool,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4CudaSequenceKvState {
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
pub(crate) struct DeepSeekV4CudaOperatorCache {
    pub(crate) ops: ferrule_cuda::context::CudaArtifactOperatorContext,
    profile: bool,
    managed_experts: bool,
    expert_upload_inflight: usize,
    kv_page_pool: Option<ferrule_cuda::CudaKvPagePool>,
    pending_kv_reservations: Vec<ferrule_cuda::KvPoolReservation>,
    active_paged_kv: Option<ActivePagedKvBinding>,
    cached_paged_kv: HashMap<(usize, usize, usize, usize), ActivePagedKvBinding>,
    linears: HashMap<String, ferrule_cuda::context::CudaArtifactLinearHandle>,
    experts: HashMap<ExpertId, CudaFp4ExpertHandles>,
    expert_slot_tables: HashMap<usize, ferrule_cuda::context::CudaExpertSlotTable>,
    retired_experts: Vec<CudaRetiredExpert>,
    abandoned_uploads: Vec<CudaExpertUploadTicket>,
    uploading_experts: HashMap<ExpertId, CudaPendingExpertInstall>,
    poisoned_expert_layers: BTreeSet<usize>,
    moe_access_events: Vec<ExpertBatchAccessEvent>,
    decode_arena: DeepSeekV4DecodeArena,
    host_staged_cache: HostStagedExpertCache,
    unretained_host_experts: HashMap<ExpertId, Arc<ExpertComputeBundle>>,
    pinned_host_expert_cache: CudaPinnedExpertCache,
    async_host_stager: AsyncHostStagedExpertLoader,
    norm_weights: HashMap<String, ferrule_cuda::context::CudaF32Buffer>,
    compressor_ape_buffers: HashMap<String, ferrule_cuda::context::CudaF32Buffer>,
    /// Cached HC weights: function, scale, base — uploaded once per layer.
    hc_weights: HashMap<String, HcDeviceWeights>,
    /// Cached HC head weights: function, scale, base — uploaded once for the
    /// terminal HC head projection.
    hc_head_weights: Option<HcDeviceWeights>,
    /// Cached attention sink buffers, keyed by layer tag — uploaded once per
    /// layer and reused across decode steps.
    pub(crate) sink_buffers: HashMap<String, ferrule_cuda::context::CudaF32Buffer>,
    /// Cached router bias buffers, keyed by layer tag — uploaded once per layer.
    router_bias_buffers: HashMap<String, ferrule_cuda::context::CudaF32Buffer>,
    /// Validated hash tables are converted from host usize and uploaded once per layer.
    router_hash_tables: HashMap<usize, CudaRouterHashTableCache>,
    /// Token ids are persistent by packed batch shape. The host mirror prevents
    /// repeated H2D copies as every routed layer sees the same batch/step ids.
    router_token_ids: HashMap<usize, ferrule_cuda::context::CudaDsv4RouterTokenIds>,
    /// Cached dequantized f32 weights for grouped output_a, uploaded once.
    grouped_wo_a_weights: HashMap<String, ferrule_cuda::context::CudaF32Buffer>,

    /// Typed precomputed RoPE tables keyed by their stable layer/resource name.
    /// Each entry records its parameters and growable position capacity so a
    /// same-name shape/configuration mismatch cannot be silently reused.
    rope_tables: HashMap<String, CudaRopeTable>,
    /// Exact-shape top-k index buffers for device-resident sparse attention.
    /// Packed prefill and decode use different row counts and must not evict each
    /// other's warm buffer.
    topk_buffers: HashMap<usize, ferrule_cuda::context::CudaI32Buffer>,
    paged_topk_logical: Option<ferrule_cuda::context::CudaI32Buffer>,
    paged_topk_selectors: Option<ferrule_cuda::context::CudaI32Buffer>,
    output_head_logits: HashMap<(usize, usize), ferrule_cuda::context::CudaF32Buffer>,
    output_head_linear_workspaces:
        HashMap<usize, ferrule_cuda::context::CudaArtifactLinearWorkspace>,
    output_head_indices: HashMap<usize, ferrule_cuda::context::CudaF32Buffer>,
    output_head_values: HashMap<usize, ferrule_cuda::context::CudaF32Buffer>,
    output_head_calls: u64,
    output_head_chunks: u64,
    output_head_rows: u64,
    output_head_cache_hits: u64,
    output_head_cache_misses: u64,
    output_head_hidden_uploads: u64,
    output_head_hidden_upload_us: u64,
    output_head_read_us: u64,
    output_head_upload_us: u64,
    output_head_topk_us: u64,
    output_head_merge_us: u64,
    moe_router_us: u64,
    moe_routing_us: u64,
    moe_plan_us: u64,
    moe_predicted_experts: u64,
    moe_prefetch_loads: u64,
    moe_prefetch_enqueued: u64,
    moe_prefetch_skipped_cached_or_inflight: u64,
    moe_prefetch_resident: u64,
    moe_prefetch_materializing: u64,
    moe_prefetch_host_staged: u64,
    moe_prefetch_in_flight: u64,
    moe_prefetch_cold: u64,
    expert_selected_resident_hits: u64,
    expert_selected_upload_hits: u64,
    expert_selected_host_staged_hits: u64,
    expert_selected_host_staging_waits: u64,
    expert_selected_host_staging_hits: u64,
    expert_selected_host_staging_wait_us: u64,
    expert_selected_cold_misses: u64,
    expert_upload_prefetch_submitted: u64,
    expert_upload_prefetch_completed: u64,
    expert_selected_upload_waits: u64,
    expert_selected_upload_wait_us: u64,
    expert_async_upload_bytes: u64,
    expert_lookahead_prefetch_calls: u64,
    expert_lookahead_prefetch_experts: u64,
    expert_lookahead_prefetch_enqueued: u64,
    expert_lookahead_prefetch_us: u64,
    moe_cache_lookup_us: u64,
    moe_expert_read_us: u64,
    moe_expert_upload_us: u64,
    moe_shared_us: u64,
    moe_workspace_us: u64,
    moe_compute_submit_us: u64,
    moe_commit_us: u64,
    expert_selected: u64,
    expert_selected_load_requests: u64,
    expert_loads: u64,
    expert_load_bytes: u64,
    expert_evictions: u64,
}

#[cfg(feature = "cuda")]
struct HcDeviceWeights {
    function: ferrule_cuda::context::CudaF32Buffer,
    scale: ferrule_cuda::context::CudaF32Buffer,
    base: ferrule_cuda::context::CudaF32Buffer,
}

#[cfg(feature = "cuda")]
struct ActivePagedKvBinding {
    physical_block_slots: Vec<i32>,
    block_slots_device: ferrule_cuda::context::CudaI32Buffer,
    block_offsets_device: ferrule_cuda::context::CudaI32Buffer,
    kv_len_device: ferrule_cuda::context::CudaI32Buffer,
    row_sequence_ids_device: ferrule_cuda::context::CudaI32Buffer,
    page_tokens: usize,
    layer_count: usize,
    sequence_count: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaRouterHashTableIdentity {
    host: Vec<usize>,
    hash_rows: usize,
    hash_cols: usize,
    experts: usize,
    top_k: usize,
}

#[cfg(feature = "cuda")]
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
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} router hash table changed after device upload"
            )));
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
        Error::Model(format!(
            "DeepSeek-V4 layer {layer} router hash table shape overflows usize: rows={hash_rows} cols={hash_cols}"
        ))
    })?;
    if table.len() != expected {
        return Err(Error::Model(format!(
            "DeepSeek-V4 layer {layer} router hash table shape mismatch: rows={hash_rows} cols={hash_cols} require {expected} entries, got {}",
            table.len()
        )));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
struct CudaRouterHashTableCache {
    identity: CudaRouterHashTableIdentity,
    device: ferrule_cuda::context::CudaDsv4RouterHashTable,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct DeepSeekV4DecodeArena {
    hidden: Option<ferrule_cuda::context::CudaF32Buffer>,
    hc_inputs: HashMap<usize, ferrule_cuda::context::CudaF32Buffer>,
    final_hiddens: HashMap<usize, ferrule_cuda::context::CudaF32Buffer>,
    final_norms: HashMap<usize, ferrule_cuda::context::CudaF32Buffer>,
    topk_rows: HashMap<usize, ferrule_cuda::context::CudaF32Buffer>,
    /// Reusable routed MoE scratch/pointer buffers keyed by exact execution shape.
    moe_workspaces:
        HashMap<(usize, usize, usize, usize), ferrule_cuda::context::CudaMoeBatchedWorkspace>,
    /// Per-layer stable-slot route resolution scratch. Keeping this persistent is
    /// required for allocation-free steady decode.
    moe_resolve_workspaces: HashMap<usize, ferrule_cuda::context::CudaExpertRouteResolveWorkspace>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4DecodeBuffers {
    pub(crate) hc_input: ferrule_cuda::context::CudaF32Buffer,
    pub(crate) final_hidden: ferrule_cuda::context::CudaF32Buffer,
    pub(crate) final_norm: ferrule_cuda::context::CudaF32Buffer,
    pub(crate) topk_row: ferrule_cuda::context::CudaF32Buffer,
}

#[cfg(feature = "cuda")]
struct CudaFp4ExpertHandles {
    gate: ferrule_cuda::context::CudaArtifactLinearHandle,
    up: ferrule_cuda::context::CudaArtifactLinearHandle,
    down: ferrule_cuda::context::CudaArtifactLinearHandle,
    bytes: u64,
    upload_guard: Option<CudaExpertUploadGuard>,
}

#[cfg(feature = "cuda")]
struct CudaExpertUploadGuard {
    _staging: CudaPinnedExpertBundle,
    event: ferrule_cuda::context::CudaUploadEvent,
}

#[cfg(feature = "cuda")]
struct CudaRetiredExpert {
    _handles: CudaFp4ExpertHandles,
    event: ferrule_cuda::context::CudaComputeEvent,
}

#[cfg(feature = "cuda")]
struct CudaExpertUploadTicket {
    gate: Option<ferrule_cuda::context::CudaArtifactLinearAsyncUpload>,
    up: Option<ferrule_cuda::context::CudaArtifactLinearAsyncUpload>,
    down: Option<ferrule_cuda::context::CudaArtifactLinearAsyncUpload>,
    bytes: u64,
    staging: Option<CudaPinnedExpertBundle>,
    event: Option<ferrule_cuda::context::CudaUploadEvent>,
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
struct CudaPinnedExpertLinear {
    matrix: ExpertMatrixKind,
    format: ExpertLinearFormat,
    weight: ferrule_cuda::context::CudaPinnedU8HostBuffer,
    scale: ferrule_cuda::context::CudaPinnedU8HostBuffer,
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
struct CudaPinnedExpertBundle {
    expert: ExpertId,
    gate: CudaPinnedExpertLinear,
    up: CudaPinnedExpertLinear,
    down: CudaPinnedExpertLinear,
    bytes: u64,
}

#[cfg(feature = "cuda")]
struct CudaPinnedExpertCache {
    cache: OwnerMemoryLru<ExpertId, CudaPinnedExpertBundle>,
}

#[cfg(feature = "cuda")]
struct CudaMoeRoutes {
    routes: Vec<ExpertRoute>,
    selected: Vec<usize>,
}

#[cfg(feature = "cuda")]
struct CudaMoeMaterialization {
    streaming: ExpertStreamingStep,
    leases: Vec<ExpertLease>,
}

#[cfg(feature = "cuda")]
struct CudaPendingExpertInstall {
    prepared: PreparedExpertInstall,
    load_source: crate::moe::streaming::ExpertLoadSource,
    ticket: Option<CudaExpertUploadTicket>,
}

#[cfg(feature = "cuda")]
impl CudaExpertUploadTicket {
    fn event(&self) -> &ferrule_cuda::context::CudaUploadEvent {
        self.event
            .as_ref()
            .expect("live upload ticket has an event")
    }

    fn is_complete(&self) -> Result<bool> {
        self.event().is_complete()
    }

    fn synchronize(&self) -> Result<()> {
        self.event().synchronize()
    }

    fn bytes(&self) -> u64 {
        self.bytes
    }

    fn into_handles(mut self) -> CudaFp4ExpertHandles {
        let gate = self
            .gate
            .take()
            .expect("live upload ticket has gate upload");
        let up = self.up.take().expect("live upload ticket has up upload");
        let down = self
            .down
            .take()
            .expect("live upload ticket has down upload");
        let staging = self
            .staging
            .take()
            .expect("live upload ticket has staging resources");
        let event = self.event.take().expect("live upload ticket has an event");
        CudaFp4ExpertHandles {
            gate: gate.into_handle(),
            up: up.into_handle(),
            down: down.into_handle(),
            bytes: self.bytes,
            upload_guard: Some(CudaExpertUploadGuard {
                _staging: staging,
                event,
            }),
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaExpertUploadTicket {
    fn drop(&mut self) {
        if let Some(event) = self.event.as_ref()
            && !matches!(event.is_complete(), Ok(true))
        {
            let _ = event.synchronize();
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaExpertUploadGuard {
    fn drop(&mut self) {
        if !matches!(self.event.is_complete(), Ok(true)) {
            let _ = self.event.synchronize();
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaRetiredExpert {
    fn drop(&mut self) {
        if !matches!(self.event.is_complete(), Ok(true)) {
            let _ = self.event.synchronize();
        }
    }
}

#[cfg(feature = "cuda")]
impl CudaPinnedExpertLinear {
    fn pin(
        ops: &ferrule_cuda::context::CudaArtifactOperatorContext,
        linear: &ExpertLinearPayload,
    ) -> Result<Self> {
        let scale = linear.scale.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "CUDA routed expert {:?} is missing E8M0 scale payload",
                linear.matrix
            ))
        })?;
        Ok(Self {
            matrix: linear.matrix,
            format: linear.format.clone(),
            weight: ops.pin_u8_host_buffer(&linear.weight.bytes)?,
            scale: ops.pin_u8_host_buffer(&scale.bytes)?,
        })
    }
}

#[cfg(feature = "cuda")]
impl CudaPinnedExpertBundle {
    fn pin(
        ops: &ferrule_cuda::context::CudaArtifactOperatorContext,
        bundle: &ExpertComputeBundle,
    ) -> Result<Self> {
        Ok(Self {
            expert: bundle.expert,
            gate: CudaPinnedExpertLinear::pin(ops, &bundle.gate)?,
            up: CudaPinnedExpertLinear::pin(ops, &bundle.up)?,
            down: CudaPinnedExpertLinear::pin(ops, &bundle.down)?,
            bytes: bundle.total_bytes(),
        })
    }
}

#[cfg(feature = "cuda")]
impl CudaPinnedExpertCache {
    fn new(limits: MemoryPoolLimits) -> Self {
        Self {
            cache: OwnerMemoryLru::new(limits),
        }
    }

    fn get(&mut self, expert: ExpertId) -> Option<CudaPinnedExpertBundle> {
        self.cache.get_cloned(expert)
    }

    fn insert(&mut self, bundle: CudaPinnedExpertBundle) -> bool {
        let expert = bundle.expert;
        let bytes = bundle.bytes;
        self.cache.insert(expert, bundle, bytes)
    }

    fn stats(&self) -> MemoryPoolStats {
        self.cache.stats()
    }
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

fn validate_output_head_rows_request(
    shape: &[usize],
    hidden_len: usize,
    batch_rows: usize,
    top_k: usize,
    chunk_rows: usize,
) -> Result<(usize, usize)> {
    if shape.len() != 2 {
        return Err(Error::Model(format!(
            "DeepSeek-V4 CUDA output head expects 2D slice, got {shape:?}"
        )));
    }
    if chunk_rows == 0 {
        return Err(Error::Model(
            "DeepSeek-V4 CUDA output-head chunk_rows must be > 0".into(),
        ));
    }
    if top_k > 40 {
        return Err(Error::Model(format!(
            "DeepSeek-V4 CUDA output-head top-k supports k<=40, got {top_k}"
        )));
    }
    let vocab_rows = shape[0];
    let hidden_cols = shape[1];
    let expected_hidden = batch_rows.checked_mul(hidden_cols).ok_or_else(|| {
        Error::Model("DeepSeek-V4 CUDA output-head batch input size overflow".into())
    })?;
    if hidden_len != expected_hidden {
        return Err(Error::Model(format!(
            "DeepSeek-V4 CUDA output-head rows input mismatch: expected {batch_rows}x{hidden_cols}={expected_hidden}, got {hidden_len}"
        )));
    }
    Ok((vocab_rows, hidden_cols))
}

fn fixed_eight_segment_capacity(route_count: usize, resident_slots: usize) -> Result<usize> {
    let populated_slots = route_count.min(resident_slots);
    let padding = populated_slots.checked_mul(7).ok_or_else(|| {
        Error::Internal(format!(
            "CUDA segmented MoE fixed-eight capacity overflow: routes={route_count} resident_slots={resident_slots}"
        ))
    })?;
    let padded_routes = route_count.checked_add(padding).ok_or_else(|| {
        Error::Internal(format!(
            "CUDA segmented MoE fixed-eight capacity overflow: routes={route_count} resident_slots={resident_slots}"
        ))
    })?;
    let segment_capacity = (padded_routes / 8).max(1);
    if segment_capacity > u16::MAX as usize {
        return Err(Error::Internal(format!(
            "CUDA segmented MoE fixed-eight segment capacity {segment_capacity} exceeds the u16 limit 65535: routes={route_count} resident_slots={resident_slots}"
        )));
    }
    Ok(segment_capacity)
}

fn merge_output_head_chunk(
    top_by_row: &mut [Vec<TokenLogit>],
    indices: &[f32],
    values: &[f32],
    chunk_k: usize,
    token_offset: usize,
    top_k: usize,
) -> Result<()> {
    let expected = top_by_row
        .len()
        .checked_mul(chunk_k)
        .ok_or_else(|| Error::Model("DeepSeek-V4 output-head top-k merge size overflow".into()))?;
    if indices.len() != expected || values.len() != expected {
        return Err(Error::Model(format!(
            "DeepSeek-V4 output-head top-k merge shape mismatch: rows={} k={chunk_k} indices={} values={}",
            top_by_row.len(),
            indices.len(),
            values.len()
        )));
    }
    for (row, top) in top_by_row.iter_mut().enumerate() {
        let row_start = row * chunk_k;
        for slot in 0..chunk_k {
            let local_token = indices[row_start + slot] as usize;
            let token_id =
                u32::try_from(token_offset.checked_add(local_token).ok_or_else(|| {
                    Error::Model("DeepSeek-V4 output-head token id overflow".into())
                })?)
                .map_err(|_| Error::Model("DeepSeek-V4 output-head token id exceeds u32".into()))?;
            top.push(TokenLogit {
                token_id,
                logit: values[row_start + slot],
            });
        }
        top.sort_by(rank_logits_desc);
        top.truncate(top_k);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
impl DeepSeekV4CudaOperatorCache {
    pub(crate) fn new(
        policy: &DeepSeekV4ExecutionPolicy,
        expert_memory_policy: ExpertMemoryPolicy,
    ) -> Result<Self> {
        Ok(Self {
            ops: ferrule_cuda::context::CudaArtifactOperatorContext::new()?,
            profile: policy.profile_enabled(),
            managed_experts: policy.managed_experts(),
            expert_upload_inflight: policy.expert_upload_inflight(),
            kv_page_pool: None,
            pending_kv_reservations: Vec::new(),
            active_paged_kv: None,
            cached_paged_kv: HashMap::new(),
            linears: HashMap::new(),
            experts: HashMap::new(),
            expert_slot_tables: HashMap::new(),
            retired_experts: Vec::new(),
            abandoned_uploads: Vec::new(),
            uploading_experts: HashMap::new(),
            poisoned_expert_layers: BTreeSet::new(),
            moe_access_events: Vec::new(),
            decode_arena: DeepSeekV4DecodeArena::default(),
            host_staged_cache: HostStagedExpertCache::with_limits(expert_memory_policy.host_staged),
            unretained_host_experts: HashMap::new(),
            pinned_host_expert_cache: CudaPinnedExpertCache::new(expert_memory_policy.pinned_host),
            async_host_stager: AsyncHostStagedExpertLoader::default(),
            norm_weights: HashMap::new(),
            compressor_ape_buffers: HashMap::new(),
            hc_weights: HashMap::new(),
            hc_head_weights: None,
            sink_buffers: HashMap::new(),
            router_bias_buffers: HashMap::new(),
            router_hash_tables: HashMap::new(),
            router_token_ids: HashMap::new(),
            grouped_wo_a_weights: HashMap::new(),
            rope_tables: HashMap::new(),
            topk_buffers: HashMap::new(),
            paged_topk_logical: None,
            paged_topk_selectors: None,
            output_head_logits: HashMap::new(),
            output_head_linear_workspaces: HashMap::new(),
            output_head_indices: HashMap::new(),
            output_head_values: HashMap::new(),
            output_head_calls: 0,
            output_head_chunks: 0,
            output_head_rows: 0,
            output_head_cache_hits: 0,
            output_head_cache_misses: 0,
            output_head_hidden_uploads: 0,
            output_head_hidden_upload_us: 0,
            output_head_read_us: 0,
            output_head_upload_us: 0,
            output_head_topk_us: 0,
            output_head_merge_us: 0,
            moe_router_us: 0,
            moe_routing_us: 0,
            moe_plan_us: 0,
            moe_predicted_experts: 0,
            moe_prefetch_loads: 0,
            moe_prefetch_enqueued: 0,
            moe_prefetch_skipped_cached_or_inflight: 0,
            moe_prefetch_resident: 0,
            moe_prefetch_materializing: 0,
            moe_prefetch_host_staged: 0,
            moe_prefetch_in_flight: 0,
            moe_prefetch_cold: 0,
            expert_selected_resident_hits: 0,
            expert_selected_upload_hits: 0,
            expert_selected_host_staged_hits: 0,
            expert_selected_host_staging_waits: 0,
            expert_selected_host_staging_hits: 0,
            expert_selected_host_staging_wait_us: 0,
            expert_selected_cold_misses: 0,
            expert_upload_prefetch_submitted: 0,
            expert_upload_prefetch_completed: 0,
            expert_selected_upload_waits: 0,
            expert_selected_upload_wait_us: 0,
            expert_async_upload_bytes: 0,
            expert_lookahead_prefetch_calls: 0,
            expert_lookahead_prefetch_experts: 0,
            expert_lookahead_prefetch_enqueued: 0,
            expert_lookahead_prefetch_us: 0,
            moe_cache_lookup_us: 0,
            moe_expert_read_us: 0,
            moe_expert_upload_us: 0,
            moe_shared_us: 0,
            moe_workspace_us: 0,
            moe_compute_submit_us: 0,
            moe_commit_us: 0,
            expert_selected: 0,
            expert_selected_load_requests: 0,
            expert_loads: 0,
            expert_load_bytes: 0,
            expert_evictions: 0,
        })
    }

    pub(crate) fn configure_kv_page_pool(
        &mut self,
        schema: &DeepSeekV4KvLayoutSchema,
        max_pages: usize,
    ) -> Result<()> {
        if !self.pending_kv_reservations.is_empty() {
            return Err(Error::Model(
                "cannot reconfigure DeepSeek-V4 KV pool with a pending batch".into(),
            ));
        }
        let data_planes = schema.planes().get(..3).ok_or_else(|| {
            Error::Model("DeepSeek-V4 KV schema is missing token-scaled data planes".into())
        })?;
        self.kv_page_pool = Some(ferrule_cuda::CudaKvPagePool::new(
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

    pub(crate) fn prepare_kv_pages(
        &mut self,
        reservations: &[(Vec<KvPageId>, Option<KvCowReplacement>)],
    ) -> Result<()> {
        if !self.pending_kv_reservations.is_empty() {
            return Err(Error::Model(
                "DeepSeek-V4 CUDA KV batch is already pending".into(),
            ));
        }
        let pool = self.kv_page_pool.as_mut().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
        })?;
        for (pages, cow) in reservations {
            match pool.reserve(&self.ops, pages, *cow) {
                Ok(reservation) => self.pending_kv_reservations.push(reservation),
                Err(error) => {
                    for reservation in self.pending_kv_reservations.drain(..) {
                        let _ = pool.rollback(&self.ops, reservation);
                    }
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn lower_paged_binding(
        &self,
        block_ids: &[ferrule_common::execution::KvBlockId],
        sequence_len: usize,
    ) -> Result<DeepSeekV4PagedKvBinding> {
        let pool = self.kv_page_pool.as_ref().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
        })?;
        let mut physical_block_slots = Vec::with_capacity(block_ids.len());
        for block in block_ids {
            let page = KvPageId(block.get());
            let slot = pool.physical_slot(page).or_else(|| {
                self.pending_kv_reservations
                    .iter()
                    .find_map(|reservation| pool.pending_slot(reservation, page))
            });
            let slot = slot.ok_or_else(|| {
                Error::Model(format!(
                    "DeepSeek-V4 KV page {} has no committed or provisional physical slot",
                    page.0
                ))
            })?;
            physical_block_slots.push(i32::try_from(slot).map_err(|_| {
                Error::Model("DeepSeek-V4 physical KV slot exceeds i32 ABI".into())
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
        binding: Option<&DeepSeekV4PagedKvBinding>,
    ) -> Result<()> {
        match binding {
            Some(binding) => self.activate_paged_bindings(&[binding]),
            None => {
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
        bindings: &[&DeepSeekV4PagedKvBinding],
    ) -> Result<()> {
        let row_sequence_ids = (0..bindings.len()).collect::<Vec<_>>();
        self.activate_paged_bindings_for_rows(bindings, &row_sequence_ids)
    }

    /// Activate sequence-owned page tables plus an independent packed-row selector.
    pub(crate) fn activate_paged_bindings_for_rows(
        &mut self,
        bindings: &[&DeepSeekV4PagedKvBinding],
        row_sequence_ids: &[usize],
    ) -> Result<()> {
        if bindings.is_empty() {
            return self.activate_paged_binding(None);
        }
        let page_tokens = bindings[0].page_tokens;
        let layer_count = bindings[0].layer_count;
        if page_tokens == 0 || layer_count == 0 {
            return Err(Error::Model(
                "DeepSeek-V4 paged binding has invalid page/layer metadata".into(),
            ));
        }

        let total_blocks = bindings.iter().try_fold(0usize, |total, binding| {
            if binding.page_tokens != page_tokens || binding.layer_count != layer_count {
                return Err(Error::Model(
                    "DeepSeek-V4 packed paged bindings use incompatible layouts".into(),
                ));
            }
            total
                .checked_add(binding.physical_block_slots.len())
                .ok_or_else(|| Error::Model("DeepSeek-V4 packed block table overflow".into()))
        })?;
        let mut physical_block_slots = Vec::with_capacity(total_blocks);
        let mut block_offsets = Vec::with_capacity(bindings.len() + 1);
        let mut kv_lens = Vec::with_capacity(bindings.len());
        block_offsets.push(0);
        for binding in bindings {
            physical_block_slots.extend_from_slice(&binding.physical_block_slots);
            block_offsets.push(i32::try_from(physical_block_slots.len()).map_err(|_| {
                Error::Model("DeepSeek-V4 packed block table exceeds i32 ABI".into())
            })?);
            kv_lens.push(
                i32::try_from(binding.sequence_len).map_err(|_| {
                    Error::Model("DeepSeek-V4 sequence length exceeds i32 ABI".into())
                })?,
            );
        }

        if row_sequence_ids.is_empty()
            || row_sequence_ids
                .iter()
                .any(|sequence| *sequence >= bindings.len())
        {
            return Err(Error::Model(
                "DeepSeek-V4 packed row selector references a missing sequence".into(),
            ));
        }
        let row_sequence_ids = row_sequence_ids
            .iter()
            .map(|sequence| {
                i32::try_from(*sequence)
                    .map_err(|_| Error::Model("DeepSeek-V4 row sequence ID exceeds i32 ABI".into()))
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
                .overwrite_i32_buffer(&physical_block_slots, &mut active.block_slots_device)?;
            self.ops
                .overwrite_i32_buffer(&block_offsets, &mut active.block_offsets_device)?;
            self.ops
                .overwrite_i32_buffer(&kv_lens, &mut active.kv_len_device)?;
            self.ops
                .overwrite_i32_buffer(&row_sequence_ids, &mut active.row_sequence_ids_device)?;
            active.physical_block_slots = physical_block_slots;
            active.page_tokens = page_tokens;
            active.layer_count = layer_count;
            active.sequence_count = bindings.len();
        } else {
            self.active_paged_kv = Some(ActivePagedKvBinding {
                block_slots_device: self.ops.upload_i32_buffer(&physical_block_slots)?,
                block_offsets_device: self.ops.upload_i32_buffer(&block_offsets)?,
                kv_len_device: self.ops.upload_i32_buffer(&kv_lens)?,
                row_sequence_ids_device: self.ops.upload_i32_buffer(&row_sequence_ids)?,
                physical_block_slots,
                page_tokens,
                layer_count,
                sequence_count: bindings.len(),
            });
        }
        Ok(())
    }

    pub(crate) fn commit_kv_pages(&mut self) -> Result<()> {
        let pool = self.kv_page_pool.as_mut().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
        })?;
        pool.commit_many(std::mem::take(&mut self.pending_kv_reservations))
    }

    pub(crate) fn rollback_kv_pages(&mut self) -> Result<()> {
        let Some(pool) = self.kv_page_pool.as_mut() else {
            if self.pending_kv_reservations.is_empty() {
                return Ok(());
            }
            return Err(Error::Model(
                "DeepSeek-V4 pending KV pages have no physical pool".into(),
            ));
        };
        let mut first_error = None;
        for reservation in self.pending_kv_reservations.drain(..) {
            if let Err(error) = pool.rollback(&self.ops, reservation) {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
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
        let pool = self.kv_page_pool.as_mut().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
        })?;
        pool.preempt(&self.ops, pages).map(|_| ())
    }

    pub(crate) fn restore_kv_pages(&mut self, pages: &[KvPageId]) -> Result<()> {
        if pages.is_empty() {
            return Ok(());
        }
        let pool = self.kv_page_pool.as_mut().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
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

    pub(crate) fn drain_moe_access_events(&mut self) -> Vec<ExpertBatchAccessEvent> {
        std::mem::take(&mut self.moe_access_events)
    }

    pub(crate) fn resident_experts_for_layer(&self, layer: usize) -> Vec<usize> {
        let mut experts = self
            .experts
            .keys()
            .filter(|expert| expert.layer == layer)
            .map(|expert| expert.expert)
            .collect::<Vec<_>>();
        experts.sort_unstable();
        experts
    }

    pub(crate) fn materializing_experts_for_layer(&self, layer: usize) -> Vec<usize> {
        let mut experts = self
            .uploading_experts
            .keys()
            .filter(|expert| expert.layer == layer)
            .map(|expert| expert.expert)
            .collect::<Vec<_>>();
        experts.sort_unstable();
        experts
    }

    pub(crate) fn host_staged_experts_for_layer(&self, layer: usize) -> Vec<usize> {
        self.host_staged_cache.expert_ids_for_layer(layer)
    }

    pub(crate) fn resident_expert_stats_for_layer(&self, layer: usize) -> (usize, u64) {
        let mut entries = 0usize;
        let mut bytes = 0u64;
        for (expert, handles) in &self.experts {
            if expert.layer == layer {
                entries = entries.saturating_add(1);
                bytes = bytes.saturating_add(handles.bytes);
            }
        }
        (entries, bytes)
    }

    fn resident_expert_totals(&self) -> (usize, u64) {
        let mut entries = 0usize;
        let mut bytes = 0u64;
        for expert in self.experts.values() {
            entries = entries.saturating_add(1);
            bytes = bytes.saturating_add(expert.bytes);
        }
        (entries, bytes)
    }

    pub(crate) fn clear_expert_residency(
        &mut self,
        mut residency: Option<&mut (dyn ExpertResidencyControl + '_)>,
    ) -> Result<()> {
        for (_, mut pending) in self.uploading_experts.drain() {
            if let Some(ticket) = pending.ticket.take() {
                ticket.synchronize()?;
            }
            let control = residency.as_deref_mut().ok_or_else(|| {
                Error::Internal(
                    "cannot clear pending CUDA expert installs without residency control".into(),
                )
            })?;
            control.cancel_install(pending.prepared)?;
        }
        for table in self.expert_slot_tables.values_mut() {
            self.ops.clear_expert_slot_table(table)?;
        }
        for retired in &self.retired_experts {
            retired.event.synchronize()?;
        }
        for upload in &self.abandoned_uploads {
            upload.synchronize()?;
        }
        self.experts.clear();
        self.retired_experts.clear();
        self.abandoned_uploads.clear();
        self.poisoned_expert_layers.clear();
        Ok(())
    }

    pub(crate) fn shutdown(
        &mut self,
        residency: Option<&mut (dyn ExpertResidencyControl + '_)>,
    ) -> Result<()> {
        self.clear_expert_residency(residency)?;
        self.ops.sync_upload_stream()?;
        self.ops.sync_stream()?;
        self.decode_arena = DeepSeekV4DecodeArena::default();
        self.topk_buffers.clear();
        self.linears.clear();
        self.norm_weights.clear();
        self.compressor_ape_buffers.clear();
        self.hc_weights.clear();
        self.hc_head_weights = None;
        self.sink_buffers.clear();
        self.router_bias_buffers.clear();
        self.router_hash_tables.clear();
        self.router_token_ids.clear();
        self.grouped_wo_a_weights.clear();
        self.rope_tables.clear();
        Ok(())
    }

    fn expert_key(model_instance: u64, expert: ExpertId) -> Result<ExpertKey> {
        Ok(ExpertKey::new(
            model_instance,
            u32::try_from(expert.layer)
                .map_err(|_| Error::Model(format!("expert layer {} exceeds u32", expert.layer)))?,
            u32::try_from(expert.expert)
                .map_err(|_| Error::Model(format!("expert index {} exceeds u32", expert.expert)))?,
        ))
    }

    fn expert_id(key: ExpertKey) -> Result<ExpertId> {
        Ok(ExpertId::new(
            usize::try_from(key.layer)
                .map_err(|_| Error::Model("expert layer exceeds usize".into()))?,
            usize::try_from(key.expert)
                .map_err(|_| Error::Model("expert index exceeds usize".into()))?,
        ))
    }

    fn ensure_expert_layer_healthy(&self, layer: usize) -> Result<()> {
        if self.poisoned_expert_layers.contains(&layer) {
            return Err(Error::Internal(format!(
                "DeepSeek-V4 CUDA expert table for layer {layer} is poisoned after a physical/controller residency divergence"
            )));
        }
        Ok(())
    }

    fn ensure_expert_slot_table(
        &mut self,
        layer: usize,
        expert_capacity: usize,
        slot_capacity: usize,
    ) -> Result<()> {
        self.ensure_expert_layer_healthy(layer)?;
        if slot_capacity == 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 CUDA expert residency capacity is zero for layer {layer}"
            )));
        }
        if let Some(table) = self.expert_slot_tables.get(&layer) {
            if table.host().expert_capacity() != expert_capacity
                || table.host().slot_capacity() != slot_capacity
            {
                return Err(Error::Internal(format!(
                    "DeepSeek-V4 CUDA expert table capacity changed for layer {layer}: experts={} slots={}, requested experts={expert_capacity} slots={slot_capacity}",
                    table.host().expert_capacity(),
                    table.host().slot_capacity(),
                )));
            }
            return Ok(());
        }
        self.expert_slot_tables.insert(
            layer,
            self.ops.expert_slot_table(expert_capacity, slot_capacity)?,
        );
        Ok(())
    }

    fn poison_expert_layer(&mut self, layer: usize, cause: impl std::fmt::Display) -> Error {
        self.poisoned_expert_layers.insert(layer);
        let clear_error = self
            .expert_slot_tables
            .get_mut(&layer)
            .and_then(|table| self.ops.clear_expert_slot_table(table).err());
        let resident = self
            .experts
            .keys()
            .copied()
            .filter(|expert| expert.layer == layer)
            .collect::<Vec<_>>();
        for expert in resident {
            self.experts.remove(&expert);
        }
        match clear_error {
            Some(clear_error) => Error::Internal(format!(
                "DeepSeek-V4 CUDA expert residency diverged for layer {layer}: {cause}; physical table clear also failed: {clear_error}; layer poisoned"
            )),
            None => Error::Internal(format!(
                "DeepSeek-V4 CUDA expert residency diverged for layer {layer}: {cause}; physical table cleared and layer poisoned"
            )),
        }
    }

    fn cancel_prepared_after_failure(
        residency: &mut dyn ExpertResidencyControl,
        prepared: PreparedExpertInstall,
        failure: Error,
    ) -> Error {
        match residency.cancel_install(prepared) {
            Ok(()) => failure,
            Err(cancel_error) => Error::Internal(format!(
                "expert backend operation failed ({failure}); canceling prepared install also failed ({cancel_error})"
            )),
        }
    }

    fn drain_async_host_staging(&mut self) -> usize {
        let completed = self.async_host_stager.drain_into(
            &mut self.host_staged_cache,
            &mut self.unretained_host_experts,
        );
        let uploading_experts = &self.uploading_experts;
        self.unretained_host_experts
            .retain(|expert, _| uploading_experts.contains_key(expert));
        completed
    }

    fn retire_completed_expert_resources(&mut self) -> Result<()> {
        for handles in self.experts.values_mut() {
            let completed = handles
                .upload_guard
                .as_ref()
                .map(|guard| guard.event.is_complete())
                .transpose()?
                .unwrap_or(false);
            if completed {
                handles.upload_guard = None;
            }
        }
        for retired in &mut self.retired_experts {
            let completed = retired
                ._handles
                .upload_guard
                .as_ref()
                .map(|guard| guard.event.is_complete())
                .transpose()?
                .unwrap_or(false);
            if completed {
                retired._handles.upload_guard = None;
            }
        }
        let mut index = 0;
        while index < self.retired_experts.len() {
            if self.retired_experts[index].event.is_complete()? {
                self.retired_experts.swap_remove(index);
            } else {
                index += 1;
            }
        }
        let mut index = 0;
        while index < self.abandoned_uploads.len() {
            if self.abandoned_uploads[index].is_complete()? {
                self.abandoned_uploads.swap_remove(index);
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    fn limit_prefetch_reservations_for_selected(
        &mut self,
        layer: usize,
        selected: &[ExpertId],
        resident_capacity: usize,
        residency: &mut dyn ExpertResidencyControl,
    ) -> Result<()> {
        let selected = selected.iter().copied().collect::<BTreeSet<_>>();
        let keep = resident_capacity.saturating_sub(selected.len());
        let mut pending = self
            .uploading_experts
            .keys()
            .copied()
            .filter(|expert| expert.layer == layer && !selected.contains(expert))
            .collect::<Vec<_>>();
        pending.sort_unstable();
        while pending.len() > keep {
            let expert = pending.pop().expect("pending prefetch count checked");
            let mut install = self
                .uploading_experts
                .remove(&expert)
                .expect("pending prefetch key came from map");
            self.unretained_host_experts.remove(&expert);
            residency.cancel_install(install.prepared)?;
            if let Some(ticket) = install.ticket.take() {
                self.abandoned_uploads.push(ticket);
            }
        }
        Ok(())
    }

    fn upload_prefetches_in_flight(&self) -> usize {
        self.uploading_experts
            .values()
            .filter(|pending| pending.ticket.is_some())
            .count()
    }

    fn max_upload_prefetch_in_flight(&self) -> usize {
        self.expert_upload_inflight
    }

    fn pinned_bundle_for_upload(
        &mut self,
        bundle: &ExpertComputeBundle,
    ) -> Result<CudaPinnedExpertBundle> {
        if let Some(pinned) = self.pinned_host_expert_cache.get(bundle.expert) {
            return Ok(pinned);
        }
        let pinned = CudaPinnedExpertBundle::pin(&self.ops, bundle)?;
        self.pinned_host_expert_cache.insert(pinned.clone());
        Ok(pinned)
    }

    fn prefetch_upload_ticket(
        &mut self,
        bundle: &ExpertComputeBundle,
    ) -> Result<CudaExpertUploadTicket> {
        let pinned = self.pinned_bundle_for_upload(bundle)?;
        let ticket = self.upload_pinned_expert_bundle_async(&pinned)?;
        self.expert_async_upload_bytes = self
            .expert_async_upload_bytes
            .saturating_add(ticket.bytes());
        self.expert_upload_prefetch_submitted =
            self.expert_upload_prefetch_submitted.saturating_add(1);
        Ok(ticket)
    }

    fn physical_binding_matches(&self, layer: usize, binding: ExpertSlotBinding) -> Result<bool> {
        let expert = Self::expert_id(binding.key)?;
        if expert.layer != layer || !self.experts.contains_key(&expert) {
            return Ok(false);
        }
        let Some(table) = self.expert_slot_tables.get(&layer) else {
            return Ok(false);
        };
        let Some(physical) = table.host().binding(expert.expert) else {
            return Ok(false);
        };
        Ok(
            physical.slot == i32::try_from(binding.slot.get()).unwrap_or(-1)
                && physical.generation == i32::try_from(binding.generation.get()).unwrap_or(-1),
        )
    }

    fn install_prepared_handles(
        &mut self,
        residency: &mut dyn ExpertResidencyControl,
        prepared: PreparedExpertInstall,
        handles: CudaFp4ExpertHandles,
        expert_capacity: usize,
    ) -> Result<ExpertResidencyGrant> {
        let binding = prepared.binding();
        let expert = Self::expert_id(binding.key)?;
        let requirements = residency.requirements();
        let slot_capacity = requirements
            .layer_capacity(binding.key.layer)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "expert residency controller has no capacity for layer {}",
                    binding.key.layer
                ))
            })?;
        self.ensure_expert_slot_table(expert.layer, expert_capacity, slot_capacity)?;

        if self.ops.failpoints().check_expert_upload() {
            let failure = Error::Internal(format!(
                "deterministic failpoint: expert upload install layer {} expert {}",
                expert.layer, expert.expert
            ));
            return Err(Self::cancel_prepared_after_failure(
                residency, prepared, failure,
            ));
        }

        let pointers = self
            .ops
            .expert_slot_pointers(&handles.gate, &handles.up, &handles.down)?;
        let mut physical_changed = false;
        if let Some(evicted_key) = prepared.evicted_key() {
            let evicted = Self::expert_id(evicted_key)?;
            let old_binding = residency.binding(evicted_key)?.ok_or_else(|| {
                Error::Internal(format!(
                    "controller prepared eviction of missing resident expert {:?}",
                    evicted_key
                ))
            })?;
            if old_binding.slot != binding.slot
                || old_binding.generation.get().checked_add(1) != Some(binding.generation.get())
                || !self.physical_binding_matches(expert.layer, old_binding)?
            {
                let failure = Error::Internal(format!(
                    "prepared CUDA eviction does not match physical/controller binding for layer {} expert {}",
                    evicted.layer, evicted.expert
                ));
                let failure = Self::cancel_prepared_after_failure(residency, prepared, failure);
                return Err(self.poison_expert_layer(expert.layer, failure));
            }
            let table = self
                .expert_slot_tables
                .get_mut(&expert.layer)
                .expect("expert table ensured above");
            if let Err(error) = self.ops.evict_expert_slot_binding(
                table,
                evicted.expert,
                old_binding.slot.get(),
                old_binding.generation.get(),
            ) {
                let failure = Self::cancel_prepared_after_failure(residency, prepared, error);
                return Err(self.poison_expert_layer(expert.layer, failure));
            }
            physical_changed = true;
            let retirement_event = match self.ops.record_compute_event() {
                Ok(event) => event,
                Err(error) => {
                    let failure = Self::cancel_prepared_after_failure(residency, prepared, error);
                    return Err(self.poison_expert_layer(expert.layer, failure));
                }
            };
            let old_handles = self.experts.remove(&evicted).ok_or_else(|| {
                self.poison_expert_layer(
                    expert.layer,
                    format!(
                        "physical eviction removed layer {} expert {} but its CUDA handles were missing",
                        evicted.layer, evicted.expert
                    ),
                )
            })?;
            self.retired_experts.push(CudaRetiredExpert {
                _handles: old_handles,
                event: retirement_event,
            });
            self.expert_evictions = self.expert_evictions.saturating_add(1);
        }

        let table = self
            .expert_slot_tables
            .get_mut(&expert.layer)
            .expect("expert table ensured above");
        if let Err(error) = self.ops.install_expert_slot_at(
            table,
            expert.expert,
            binding.slot.get(),
            binding.generation.get(),
            pointers,
        ) {
            let failure = Self::cancel_prepared_after_failure(residency, prepared, error);
            if physical_changed || table.is_poisoned() {
                return Err(self.poison_expert_layer(expert.layer, failure));
            }
            return Err(failure);
        }
        self.experts.insert(expert, handles);
        let grant = match residency.publish_install(prepared) {
            Ok(grant) if grant.binding() == binding => grant,
            Ok(grant) => {
                return Err(self.poison_expert_layer(
                    expert.layer,
                    format!(
                        "controller published binding {:?}, expected {:?}",
                        grant.binding(),
                        binding
                    ),
                ));
            }
            Err(error) => {
                let _ = residency.cancel_install(prepared);
                return Err(self.poison_expert_layer(expert.layer, error));
            }
        };
        self.expert_loads = self.expert_loads.saturating_add(1);
        self.expert_load_bytes = self
            .expert_load_bytes
            .saturating_add(self.experts[&expert].bytes);
        Ok(grant)
    }

    fn load_selected_bundle(
        &mut self,
        expert: ExpertId,
        load_source: &crate::moe::streaming::ExpertLoadSource,
        reader: &ExpertStreamingReader,
    ) -> Result<Arc<ExpertComputeBundle>> {
        if let Some(bundle) = self.unretained_host_experts.remove(&expert) {
            self.expert_selected_host_staging_hits =
                self.expert_selected_host_staging_hits.saturating_add(1);
            return Ok(bundle);
        }
        if self.async_host_stager.is_in_flight(expert) {
            self.expert_selected_host_staging_waits =
                self.expert_selected_host_staging_waits.saturating_add(1);
            let wait_start = profile_start(self.profile);
            let staged = self.async_host_stager.wait_for_into(
                expert,
                &mut self.host_staged_cache,
                &mut self.unretained_host_experts,
            )?;
            record_profile_duration(&mut self.expert_selected_host_staging_wait_us, wait_start);
            if let Some(bundle) = staged {
                self.expert_selected_host_staging_hits =
                    self.expert_selected_host_staging_hits.saturating_add(1);
                return Ok(bundle);
            }
        }
        if let Some(bundle) = self.host_staged_cache.get(expert) {
            self.expert_selected_host_staged_hits =
                self.expert_selected_host_staged_hits.saturating_add(1);
            return Ok(bundle);
        }
        self.expert_selected_cold_misses = self.expert_selected_cold_misses.saturating_add(1);
        let stage_start = profile_start(self.profile);
        let payload = read_experts_concurrent(reader, &[(expert, load_source.clone())])?
            .into_iter()
            .next()
            .ok_or_else(|| {
                Error::Internal(format!(
                    "expert read returned no payload for layer {} expert {}",
                    expert.layer, expert.expert
                ))
            })?;
        record_profile_duration(&mut self.moe_expert_read_us, stage_start);
        let bundle = Arc::new(ExpertComputeBundle::from_artifact_payload(payload)?);
        self.host_staged_cache.insert_shared(Arc::clone(&bundle));
        Ok(bundle)
    }

    fn finish_pending_selected(
        &mut self,
        expert: ExpertId,
        mut pending: CudaPendingExpertInstall,
        residency: &mut dyn ExpertResidencyControl,
        expert_capacity: usize,
        reader: &ExpertStreamingReader,
    ) -> Result<ExpertResidencyGrant> {
        let handles = if let Some(ticket) = pending.ticket.take() {
            let handles = match self.queue_selected_upload_wait(ticket) {
                Ok(handles) => handles,
                Err(error) => {
                    return Err(Self::cancel_prepared_after_failure(
                        residency,
                        pending.prepared,
                        error,
                    ));
                }
            };
            self.expert_upload_prefetch_completed =
                self.expert_upload_prefetch_completed.saturating_add(1);
            handles
        } else {
            let bundle = match self.load_selected_bundle(expert, &pending.load_source, reader) {
                Ok(bundle) => bundle,
                Err(error) => {
                    return Err(Self::cancel_prepared_after_failure(
                        residency,
                        pending.prepared,
                        error,
                    ));
                }
            };
            let ticket = match self.selected_upload_ticket(&bundle) {
                Ok(ticket) => ticket,
                Err(error) => {
                    return Err(Self::cancel_prepared_after_failure(
                        residency,
                        pending.prepared,
                        error,
                    ));
                }
            };
            match self.queue_selected_upload_wait(ticket) {
                Ok(handles) => handles,
                Err(error) => {
                    return Err(Self::cancel_prepared_after_failure(
                        residency,
                        pending.prepared,
                        error,
                    ));
                }
            }
        };
        self.install_prepared_handles(residency, pending.prepared, handles, expert_capacity)
    }

    fn progress_prefetch_installs(
        &mut self,
        residency: &mut dyn ExpertResidencyControl,
        expert_capacity: usize,
        reader: &ExpertStreamingReader,
    ) -> Result<usize> {
        self.drain_async_host_staging();
        let waiting = self
            .uploading_experts
            .iter()
            .filter_map(|(expert, pending)| pending.ticket.is_none().then_some(*expert))
            .collect::<Vec<_>>();
        for expert in waiting {
            if self.upload_prefetches_in_flight() >= self.max_upload_prefetch_in_flight() {
                break;
            }
            let bundle = self
                .unretained_host_experts
                .remove(&expert)
                .or_else(|| self.host_staged_cache.get(expert));
            if let Some(bundle) = bundle {
                let ticket = match self.prefetch_upload_ticket(&bundle) {
                    Ok(ticket) => ticket,
                    Err(error) => {
                        let pending = self
                            .uploading_experts
                            .remove(&expert)
                            .expect("pending prefetch exists");
                        return Err(Self::cancel_prepared_after_failure(
                            residency,
                            pending.prepared,
                            error,
                        ));
                    }
                };
                self.uploading_experts
                    .get_mut(&expert)
                    .expect("pending prefetch exists")
                    .ticket = Some(ticket);
            } else if !self.async_host_stager.is_in_flight(expert) {
                let source = self
                    .uploading_experts
                    .get(&expert)
                    .expect("pending prefetch exists")
                    .load_source
                    .clone();
                let _ = self.async_host_stager.enqueue(expert, source, reader);
            }
        }

        let mut completed = Vec::new();
        for (expert, pending) in &self.uploading_experts {
            if let Some(ticket) = &pending.ticket {
                match ticket.is_complete() {
                    Ok(true) => completed.push(*expert),
                    Ok(false) => {}
                    Err(error) => {
                        let expert = *expert;
                        let pending = self
                            .uploading_experts
                            .remove(&expert)
                            .expect("pending prefetch exists");
                        return Err(Self::cancel_prepared_after_failure(
                            residency,
                            pending.prepared,
                            error,
                        ));
                    }
                }
            }
        }

        let mut published = 0usize;
        for expert in completed {
            let mut pending = self
                .uploading_experts
                .remove(&expert)
                .expect("completed prefetch exists");
            let ticket = pending.ticket.take().expect("completed ticket exists");
            let grant = self.install_prepared_handles(
                residency,
                pending.prepared,
                ticket.into_handles(),
                expert_capacity,
            )?;
            if grant.reason() != ferrule_common::ExpertInstallReason::Prefetch
                || grant.lease().is_some()
            {
                return Err(self.poison_expert_layer(
                    expert.layer,
                    "completed asynchronous prefetch published with non-prefetch intent or a lease",
                ));
            }
            self.expert_upload_prefetch_completed =
                self.expert_upload_prefetch_completed.saturating_add(1);
            published = published.saturating_add(1);
        }
        Ok(published)
    }

    fn release_expert_leases(
        residency: &mut dyn ExpertResidencyControl,
        leases: Vec<ExpertLease>,
    ) -> Result<()> {
        let mut first_error = None;
        for lease in leases {
            if let Err(error) = residency.release(lease) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn selected_streaming_step(
        layer: usize,
        selected: Vec<ExpertId>,
        prefetched: Vec<ExpertId>,
        loads: Vec<ExpertLoadRequest>,
        evictions: Vec<ExpertEvictRequest>,
    ) -> ExpertStreamingStep {
        ExpertStreamingStep {
            layer,
            selected,
            prefetched,
            loads,
            evictions,
        }
    }

    fn materialize_selected_experts(
        &mut self,
        layer: usize,
        selected: &[usize],
        predicted_experts: &[usize],
        residency: &mut dyn ExpertResidencyControl,
        source_catalog: &ExpertSourceCatalog,
        prefetch_capacity: usize,
        reader: &ExpertStreamingReader,
    ) -> Result<CudaMoeMaterialization> {
        self.ensure_expert_layer_healthy(layer)?;
        self.retire_completed_expert_resources()?;
        let requirements = residency.requirements();
        let model_instance = requirements.model_instance;
        self.progress_prefetch_installs(residency, source_catalog.count(), reader)?;

        let mut selected_ids = selected
            .iter()
            .copied()
            .map(|expert| ExpertId::new(layer, expert))
            .collect::<Vec<_>>();
        selected_ids.sort_unstable();
        selected_ids.dedup();
        let resident_capacity = requirements
            .layer_capacity(
                u32::try_from(layer)
                    .map_err(|_| Error::Model(format!("expert layer {layer} exceeds u32")))?,
            )
            .ok_or_else(|| {
                Error::Internal(format!(
                    "expert residency controller has no capacity for layer {layer}"
                ))
            })?;
        self.limit_prefetch_reservations_for_selected(
            layer,
            &selected_ids,
            resident_capacity,
            residency,
        )?;
        let mut leases = Vec::with_capacity(selected_ids.len());
        let mut misses = Vec::new();
        let mut loads = Vec::new();
        let mut evictions = Vec::new();

        for expert in &selected_ids {
            let key = Self::expert_key(model_instance, *expert)?;
            match residency.acquire_selected(key)? {
                Some(grant) => {
                    let lease = grant.lease().ok_or_else(|| {
                        Error::Internal("selected resident grant did not carry a lease".into())
                    })?;
                    if !self.physical_binding_matches(layer, grant.binding())? {
                        let _ = residency.release(lease);
                        return Err(self.poison_expert_layer(
                            layer,
                            format!(
                                "controller selected binding {:?} is not physically installed",
                                grant.binding()
                            ),
                        ));
                    }
                    self.expert_selected_resident_hits =
                        self.expert_selected_resident_hits.saturating_add(1);
                    leases.push(lease);
                }
                None => misses.push(*expert),
            }
        }

        let materialize_result = (|| -> Result<()> {
            for expert in misses {
                let key = Self::expert_key(model_instance, expert)?;
                let grant = if let Some(pending) = self.uploading_experts.remove(&expert) {
                    self.expert_selected_upload_hits =
                        self.expert_selected_upload_hits.saturating_add(1);
                    let grant = self.finish_pending_selected(
                        expert,
                        pending,
                        residency,
                        source_catalog.count(),
                        reader,
                    )?;
                    let lease = residency
                        .acquire_selected(key)?
                        .and_then(ExpertResidencyGrant::lease)
                        .ok_or_else(|| {
                            Error::Internal(format!(
                                "prefetched selected expert {:?} was not acquirable after publication",
                                key
                            ))
                        })?;
                    leases.push(lease);
                    grant
                } else {
                    match residency.prepare_install(ExpertInstallIntent::selected(key))? {
                        ExpertInstallPrepareOutcome::Resident(grant) => {
                            let lease = grant.lease().ok_or_else(|| {
                                Error::Internal(
                                    "selected prepare returned resident grant without lease".into(),
                                )
                            })?;
                            if !self.physical_binding_matches(layer, grant.binding())? {
                                let _ = residency.release(lease);
                                return Err(self.poison_expert_layer(
                                    layer,
                                    "selected prepare returned a physically missing binding",
                                ));
                            }
                            leases.push(lease);
                            grant
                        }
                        ExpertInstallPrepareOutcome::Prepared(prepared) => {
                            let source = source_catalog.source(expert).ok_or_else(|| {
                                Error::Model(format!(
                                    "expert source catalog missing layer {} expert {}",
                                    expert.layer, expert.expert
                                ))
                            })?;
                            let bundle = match self.load_selected_bundle(expert, source, reader) {
                                Ok(bundle) => bundle,
                                Err(error) => {
                                    return Err(Self::cancel_prepared_after_failure(
                                        residency, prepared, error,
                                    ));
                                }
                            };
                            let ticket = match self.selected_upload_ticket(&bundle) {
                                Ok(ticket) => ticket,
                                Err(error) => {
                                    return Err(Self::cancel_prepared_after_failure(
                                        residency, prepared, error,
                                    ));
                                }
                            };
                            let handles = match self.queue_selected_upload_wait(ticket) {
                                Ok(handles) => handles,
                                Err(error) => {
                                    return Err(Self::cancel_prepared_after_failure(
                                        residency, prepared, error,
                                    ));
                                }
                            };
                            if let Some(evicted) = prepared.evicted_key() {
                                let evicted = Self::expert_id(evicted)?;
                                evictions.push(ExpertEvictRequest {
                                    expert: evicted,
                                    target: ExpertStorageTier::LocalStorage,
                                });
                            }
                            let grant = self.install_prepared_handles(
                                residency,
                                prepared,
                                handles,
                                source_catalog.count(),
                            )?;
                            let lease = grant.lease().ok_or_else(|| {
                                Error::Internal(
                                    "selected publication did not return an execution lease".into(),
                                )
                            })?;
                            leases.push(lease);
                            loads.push(ExpertLoadRequest {
                                expert,
                                load_source: source.clone(),
                                reason: ExpertLoadReason::Selected,
                            });
                            grant
                        }
                        ExpertInstallPrepareOutcome::CapacityAllLeased => {
                            return Err(Error::Internal(
                                "selected expert install unexpectedly reported CapacityAllLeased"
                                    .into(),
                            ));
                        }
                    }
                };
                if !self.physical_binding_matches(layer, grant.binding())? {
                    return Err(self.poison_expert_layer(
                        layer,
                        "newly published selected binding is not physically installed",
                    ));
                }
            }
            Ok(())
        })();
        if let Err(error) = materialize_result {
            let release = Self::release_expert_leases(residency, leases);
            return Err(match release {
                Ok(()) => error,
                Err(release_error) => Error::Internal(format!(
                    "selected expert materialization failed ({error}); releasing acquired leases also failed ({release_error})"
                )),
            });
        }

        self.expert_selected_load_requests = self
            .expert_selected_load_requests
            .saturating_add(loads.len() as u64);
        let prefetched = predicted_experts
            .iter()
            .copied()
            .take(prefetch_capacity)
            .map(|expert| ExpertId::new(layer, expert))
            .collect::<Vec<_>>();
        Ok(CudaMoeMaterialization {
            streaming: Self::selected_streaming_step(
                layer,
                selected_ids,
                prefetched,
                loads,
                evictions,
            ),
            leases,
        })
    }

    pub(crate) fn prefetch_predicted_experts(
        &mut self,
        layer: usize,
        predicted_experts: &[usize],
        residency: &mut dyn ExpertResidencyControl,
        source_catalog: &ExpertSourceCatalog,
        prefetch_capacity: usize,
        reader: &ExpertStreamingReader,
    ) -> Result<usize> {
        if predicted_experts.is_empty() || prefetch_capacity == 0 {
            return Ok(0);
        }
        self.ensure_expert_layer_healthy(layer)?;
        let prefetch_start = profile_start(self.profile);
        self.expert_lookahead_prefetch_calls =
            self.expert_lookahead_prefetch_calls.saturating_add(1);
        self.expert_lookahead_prefetch_experts = self
            .expert_lookahead_prefetch_experts
            .saturating_add(predicted_experts.len() as u64);
        self.progress_prefetch_installs(residency, source_catalog.count(), reader)?;

        let model_instance = residency.requirements().model_instance;
        let mut candidates = predicted_experts.to_vec();
        candidates.sort_unstable();
        candidates.dedup();
        candidates.truncate(prefetch_capacity);
        let mut enqueued = 0usize;
        for expert_index in candidates {
            let expert = ExpertId::new(layer, expert_index);
            if self.uploading_experts.contains_key(&expert) {
                self.moe_prefetch_skipped_cached_or_inflight = self
                    .moe_prefetch_skipped_cached_or_inflight
                    .saturating_add(1);
                continue;
            }
            let key = Self::expert_key(model_instance, expert)?;
            let prepared = match residency.prepare_install(ExpertInstallIntent::prefetch(key))? {
                ExpertInstallPrepareOutcome::Resident(_) => {
                    self.moe_prefetch_resident = self.moe_prefetch_resident.saturating_add(1);
                    continue;
                }
                ExpertInstallPrepareOutcome::CapacityAllLeased => {
                    self.moe_prefetch_skipped_cached_or_inflight = self
                        .moe_prefetch_skipped_cached_or_inflight
                        .saturating_add(1);
                    continue;
                }
                ExpertInstallPrepareOutcome::Prepared(prepared) => prepared,
            };
            let source = match source_catalog.source(expert) {
                Some(source) => source.clone(),
                None => {
                    let error = Error::Model(format!(
                        "expert source catalog missing layer {layer} expert {expert_index}"
                    ));
                    return Err(Self::cancel_prepared_after_failure(
                        residency, prepared, error,
                    ));
                }
            };
            let mut pending = CudaPendingExpertInstall {
                prepared,
                load_source: source.clone(),
                ticket: None,
            };
            if self.upload_prefetches_in_flight() < self.max_upload_prefetch_in_flight() {
                let bundle = self
                    .unretained_host_experts
                    .remove(&expert)
                    .or_else(|| self.host_staged_cache.get(expert));
                if let Some(bundle) = bundle {
                    pending.ticket = Some(match self.prefetch_upload_ticket(&bundle) {
                        Ok(ticket) => ticket,
                        Err(error) => {
                            return Err(Self::cancel_prepared_after_failure(
                                residency, prepared, error,
                            ));
                        }
                    });
                }
            }
            if pending.ticket.is_none() {
                // Host staging is bounded and best-effort. A full queue leaves the
                // prepared install pending; `progress_prefetch_installs` retries it
                // after completed workers are drained.
                let _ = self
                    .async_host_stager
                    .enqueue(expert, source.clone(), reader);
            }
            self.uploading_experts.insert(expert, pending);
            self.moe_prefetch_loads = self.moe_prefetch_loads.saturating_add(1);
            self.moe_prefetch_enqueued = self.moe_prefetch_enqueued.saturating_add(1);
            enqueued = enqueued.saturating_add(1);
        }
        self.expert_lookahead_prefetch_enqueued = self
            .expert_lookahead_prefetch_enqueued
            .saturating_add(enqueued as u64);
        record_profile_duration(&mut self.expert_lookahead_prefetch_us, prefetch_start);
        Ok(enqueued)
    }

    pub(crate) fn runtime_counters(&self) -> DeepSeekV4OperatorRuntimeCounters {
        let cuda = self.ops.counters();
        let async_prefetch = self.async_host_stager.stats();
        let host_cache = self.host_staged_cache.stats();
        let pinned_cache = self.pinned_host_expert_cache.stats();
        let (expert_cuda_resident_entries, expert_cuda_resident_bytes) =
            self.resident_expert_totals();
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
            moe_tc_calls: cuda.moe_tc_calls,
            moe_scalar_calls: cuda.moe_scalar_calls,
            moe_reduce_calls: cuda.moe_reduce_calls,
            moe_total_us: cuda.moe_total_us,
            moe_input_prepare_us: cuda.moe_input_prepare_us,
            moe_gate_up_us: cuda.moe_gate_up_us,
            moe_swiglu_us: cuda.moe_swiglu_us,
            moe_hidden_pack_us: cuda.moe_hidden_pack_us,
            moe_down_us: cuda.moe_down_us,
            output_head_calls: self.output_head_calls,
            output_head_chunks: self.output_head_chunks,
            output_head_rows: self.output_head_rows,
            output_head_cache_hits: self.output_head_cache_hits,
            output_head_cache_misses: self.output_head_cache_misses,
            output_head_hidden_uploads: self.output_head_hidden_uploads,
            output_head_hidden_upload_us: self.output_head_hidden_upload_us,
            output_head_read_us: self.output_head_read_us,
            output_head_upload_us: self.output_head_upload_us,
            output_head_topk_us: self.output_head_topk_us,
            output_head_merge_us: self.output_head_merge_us,
            moe_router_us: self.moe_router_us,
            moe_routing_us: self.moe_routing_us,
            moe_plan_us: self.moe_plan_us,
            moe_predicted_experts: self.moe_predicted_experts,
            moe_prefetch_loads: self.moe_prefetch_loads,
            moe_prefetch_enqueued: self.moe_prefetch_enqueued,
            moe_prefetch_skipped_cached_or_inflight: self.moe_prefetch_skipped_cached_or_inflight,
            moe_prefetch_resident: self.moe_prefetch_resident,
            moe_prefetch_materializing: self.moe_prefetch_materializing,
            moe_prefetch_host_staged: self.moe_prefetch_host_staged,
            moe_prefetch_in_flight: self.moe_prefetch_in_flight,
            moe_prefetch_cold: self.moe_prefetch_cold,
            expert_selected_resident_hits: self.expert_selected_resident_hits,
            expert_selected_upload_hits: self.expert_selected_upload_hits,
            expert_selected_host_staged_hits: self.expert_selected_host_staged_hits,
            expert_selected_host_staging_waits: self.expert_selected_host_staging_waits,
            expert_selected_host_staging_hits: self.expert_selected_host_staging_hits,
            expert_selected_host_staging_wait_us: self.expert_selected_host_staging_wait_us,
            expert_selected_cold_misses: self.expert_selected_cold_misses,
            expert_upload_prefetch_submitted: self.expert_upload_prefetch_submitted,
            expert_upload_prefetch_completed: self.expert_upload_prefetch_completed,
            expert_upload_prefetch_in_flight: self.uploading_experts.len(),
            expert_selected_upload_waits: self.expert_selected_upload_waits,
            expert_selected_upload_wait_us: self.expert_selected_upload_wait_us,
            expert_async_upload_bytes: self.expert_async_upload_bytes,
            expert_lookahead_prefetch_calls: self.expert_lookahead_prefetch_calls,
            expert_lookahead_prefetch_experts: self.expert_lookahead_prefetch_experts,
            expert_lookahead_prefetch_enqueued: self.expert_lookahead_prefetch_enqueued,
            expert_lookahead_prefetch_us: self.expert_lookahead_prefetch_us,
            moe_cache_lookup_us: self.moe_cache_lookup_us,
            moe_expert_read_us: self.moe_expert_read_us,
            moe_expert_upload_us: self.moe_expert_upload_us,
            moe_shared_us: self.moe_shared_us,
            moe_workspace_us: self.moe_workspace_us,
            moe_compute_submit_us: self.moe_compute_submit_us,
            moe_commit_us: self.moe_commit_us,
            expert_selected: self.expert_selected,
            expert_selected_load_requests: self.expert_selected_load_requests,
            expert_loads: self.expert_loads,
            expert_load_bytes: self.expert_load_bytes,
            expert_evictions: self.expert_evictions,
            expert_host_cache: host_cache,
            expert_pinned_cache: pinned_cache,
            expert_cuda_resident_entries,
            expert_cuda_resident_bytes,
            expert_async_prefetch_submitted: async_prefetch.submitted,
            expert_async_prefetch_completed: async_prefetch.completed,
            expert_async_prefetch_failed: async_prefetch.failed,
            expert_async_prefetch_skipped: async_prefetch.skipped,
            expert_async_prefetch_in_flight: async_prefetch.in_flight,
            arena_hits: cuda.arena_hits,
            arena_misses: cuda.arena_misses,
            arena_grows: cuda.arena_grows,
            arena_reuses: cuda.arena_reuses,
            expert_residency_stats: Default::default(),
            expert_predictor_stats: Default::default(),
        }
    }

    pub(crate) fn linear_matvec_from_device_into(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &mut ferrule_cuda::context::CudaF32Buffer,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        if input.len() != linear.format.in_features() {
            return Err(Error::Model(format!(
                "artifact linear {:?} device input length mismatch: expected {}, got {}",
                linear.role,
                linear.format.in_features(),
                input.len()
            )));
        }
        let key = self.ensure_linear_uploaded(linear)?;
        let use_fp8_mma = {
            let handle = self.linears.get(&key).expect("inserted above");
            self.ops.artifact_linear_uses_fp8_mma(handle)
        };
        if !use_fp8_mma {
            if let Some(activation) = linear.execution.activation_quantization {
                match activation {
                    ArtifactActivationQuantization::Fp8E4M3WithE8M0Scale { block_size } => {
                        self.ops.fp8_activation_quantize_buffer_in_place(
                            input,
                            linear.format.in_features(),
                            block_size,
                        )?;
                    }
                }
            }
        }
        let handle = self.linears.get(&key).expect("inserted above");
        self.ops.artifact_linear_matvec_into(handle, input, output)
    }

    /// Upload a norm weight once for reuse with `rms_norm_from_device`.
    pub(crate) fn upload_norm_weight(
        &self,
        weight: &[f32],
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.ops.upload_norm_weight(weight)
    }

    /// Get or upload a named norm weight, cached on device.
    /// Returns a key that can be used to look up the weight in `norm_weights`.
    pub(crate) fn ensure_norm_uploaded(&mut self, name: &str, weight: &[f32]) -> Result<()> {
        if !self.norm_weights.contains_key(name) {
            let buf = self.upload_norm_weight(weight)?;
            self.norm_weights.insert(name.to_string(), buf);
        }
        Ok(())
    }

    pub(crate) fn rms_norm_device_cached_into(
        &mut self,
        name: &str,
        input: &ferrule_cuda::context::CudaF32Buffer,
        weight: &[f32],
        eps: f32,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.ensure_norm_uploaded(name, weight)?;
        let weight_buf = self.norm_weights.get(name).expect("inserted above");
        self.ops
            .rms_norm_from_device_into(input, weight_buf, eps, output)
    }

    pub(crate) fn rms_norm_rows_device_cached_into(
        &mut self,
        name: &str,
        input: &ferrule_cuda::context::CudaF32Buffer,
        rows: usize,
        weight: &[f32],
        eps: f32,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        if rows == 0 || input.len() != rows * weight.len() || output.len() != input.len() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 CUDA RMS device rows length mismatch: rows={rows} input={} output={} weight={}",
                input.len(),
                output.len(),
                weight.len()
            )));
        }
        self.ensure_norm_uploaded(name, weight)?;
        let weight_buf = self.norm_weights.get(name).expect("inserted above");
        self.ops
            .rms_norm_rows_from_device_into(input, rows, weight_buf, eps, output)
    }

    /// Ensure HC weights (function, scale, base) are uploaded once and cached.
    pub(crate) fn ensure_hc_weights_uploaded(
        &mut self,
        name: &str,
        weights: &HyperConnectionWeights,
    ) -> Result<()> {
        if !self.hc_weights.contains_key(name) {
            let function = self.ops.upload_f32_buffer(&weights.function)?;
            let scale = self.ops.upload_f32_buffer(&weights.scale)?;
            let base = self.ops.upload_f32_buffer(&weights.base)?;
            self.hc_weights.insert(
                name.to_string(),
                HcDeviceWeights {
                    function,
                    scale,
                    base,
                },
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hc_pre_from_device_into(
        &mut self,
        name: &str,
        state: &ferrule_cuda::context::CudaF32Buffer,
        weights: &HyperConnectionWeights,
        tokens: usize,
        config: HyperConnectionConfig,
        hidden: &mut ferrule_cuda::context::CudaF32Buffer,
        pre: &mut ferrule_cuda::context::CudaF32Buffer,
        post: &mut ferrule_cuda::context::CudaF32Buffer,
        comb: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.ensure_hc_weights_uploaded(name, weights)?;
        let hw = self.hc_weights.get(name).expect("inserted above");
        self.ops.hc_pre_from_device_into(
            state,
            &hw.function,
            &hw.scale,
            &hw.base,
            tokens,
            config.hc_mult,
            config.hidden_size,
            config.sinkhorn_iters,
            config.eps,
            config.norm_eps,
            hidden,
            pre,
            post,
            comb,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hc_post_from_device_into(
        &self,
        hidden: &ferrule_cuda::context::CudaF32Buffer,
        residual: &ferrule_cuda::context::CudaF32Buffer,
        split_post: &ferrule_cuda::context::CudaF32Buffer,
        split_comb: &ferrule_cuda::context::CudaF32Buffer,
        tokens: usize,
        config: HyperConnectionConfig,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
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

    /// Ensure HC head weights (function, scale, base) are uploaded once and cached.
    pub(crate) fn ensure_hc_head_weights_uploaded(
        &mut self,
        weights: &HyperConnectionHeadWeights,
    ) -> Result<()> {
        if self.hc_head_weights.is_none() {
            let function = self.ops.upload_f32_buffer(&weights.function)?;
            let scale = self.ops.upload_f32_buffer(&weights.scale)?;
            let base = self.ops.upload_f32_buffer(&weights.base)?;
            self.hc_head_weights = Some(HcDeviceWeights {
                function,
                scale,
                base,
            });
        }
        Ok(())
    }

    /// Device-resident hc_head: state on device, weights cached, output stays
    /// on device. This is the terminal HC projection applied after all layers.
    pub(crate) fn hc_head_from_device(
        &mut self,
        state: &ferrule_cuda::context::CudaF32Buffer,
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionHeadWeights,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.ensure_hc_head_weights_uploaded(weights)?;
        let hw = self.hc_head_weights.as_ref().expect("inserted above");
        self.ops.hc_head_from_device(
            state,
            &hw.function,
            &hw.scale,
            &hw.base,
            tokens,
            config.hc_mult,
            config.hidden_size,
            config.eps,
            config.norm_eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hc_head_from_device_into(
        &mut self,
        state: &ferrule_cuda::context::CudaF32Buffer,
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionHeadWeights,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.ensure_hc_head_weights_uploaded(weights)?;
        let hw = self.hc_head_weights.as_ref().expect("inserted above");
        self.ops.hc_head_from_device_into(
            state,
            &hw.function,
            &hw.scale,
            &hw.base,
            tokens,
            config.hc_mult,
            config.hidden_size,
            config.eps,
            config.norm_eps,
            output,
        )
    }

    /// Ensure the attention sink buffer for a named layer is uploaded once and
    /// cached on device. The sink is a small `[num_heads]` f32 buffer that
    /// never changes across decode steps.
    pub(crate) fn ensure_sink_buffer(
        &mut self,
        name: &str,
        sink: &[f32],
    ) -> Result<&ferrule_cuda::context::CudaF32Buffer> {
        if !self.sink_buffers.contains_key(name) {
            let buf = self.ops.upload_f32_buffer(sink)?;
            self.sink_buffers.insert(name.to_string(), buf);
        }
        Ok(self.sink_buffers.get(name).expect("inserted above"))
    }

    fn ensure_router_bias_buffer_key(
        &mut self,
        layer: usize,
        bias: Option<&[f32]>,
        experts: usize,
    ) -> Result<Option<String>> {
        let Some(bias) = bias else {
            return Ok(None);
        };
        if bias.len() != experts {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} router bias length mismatch: got {} expected {experts}",
                bias.len()
            )));
        }
        let key = format!("router_bias_L{layer}");
        if !self.router_bias_buffers.contains_key(&key) {
            let buffer = self.ops.upload_f32_buffer(bias)?;
            self.router_bias_buffers.insert(key.clone(), buffer);
        }
        Ok(Some(key))
    }

    fn ensure_router_hash_table(
        &mut self,
        layer: usize,
        router: &RouterArtifactPayload,
        experts: usize,
        top_k: usize,
    ) -> Result<()> {
        let table = router.hash_table.as_deref().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} hash router is missing its hash table"
            ))
        })?;
        if let Some(cached) = self.router_hash_tables.get(&layer) {
            return cached.identity.validate_request(
                layer,
                table,
                router.hash_rows,
                router.hash_cols,
                experts,
                top_k,
            );
        }
        let identity = CudaRouterHashTableIdentity::new(
            layer,
            table,
            router.hash_rows,
            router.hash_cols,
            experts,
            top_k,
        )?;
        let device = self.ops.upload_dsv4_router_hash_table(
            table,
            router.hash_rows,
            router.hash_cols,
            experts,
            top_k,
        )?;
        self.router_hash_tables
            .insert(layer, CudaRouterHashTableCache { identity, device });
        Ok(())
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
        logits: &ferrule_cuda::context::CudaF32Buffer,
        token_ids: &[u32],
        router: &RouterArtifactPayload,
        router_policy: &ExpertRouterPolicy,
        indices_dev: &mut ferrule_cuda::context::CudaI32Buffer,
        weights_dev: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<Vec<Vec<ExpertRoute>>> {
        if router_policy.score_function != RouterScoreFunction::SqrtSoftplus
            || !router_policy.normalize_non_softmax_weights
        {
            return Err(Error::Model(format!(
                "DeepSeek-V4 CUDA router does not support policy {:?}",
                router_policy
            )));
        }
        let tokens = token_ids.len();
        let experts = router.weight.format.out_features();
        let top_k = router_policy.top_k;
        if tokens == 0 || top_k == 0 || top_k > 64 || top_k > experts || experts > 512 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 CUDA router unsupported shape: tokens={tokens} experts={experts} top_k={top_k}"
            )));
        }
        if !router_policy.route_scale.is_finite() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 CUDA router route_scale must be finite, got {}",
                router_policy.route_scale
            )));
        }

        match router_policy.selection {
            RouterSelectionPolicy::ScoreTopK => {
                let bias_key =
                    self.ensure_router_bias_buffer_key(layer, router.bias.as_deref(), experts)?;
                let bias = bias_key
                    .as_ref()
                    .and_then(|key| self.router_bias_buffers.get(key));
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
                self.ensure_router_hash_table(layer, router, experts, top_k)?;
                self.ensure_router_token_ids(token_ids, router.hash_rows)?;
                let hash_table = &self
                    .router_hash_tables
                    .get(&layer)
                    .expect("hash table inserted above")
                    .device;
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

        // The host residency planner still consumes compact routes for miss
        // materialization and telemetry. Compute continues from these same
        // device ids/weights through stable slot resolution and grouping.
        let indices = self.ops.download_i32_buffer(indices_dev)?;
        let weights = self.ops.download_f32_buffer(weights_dev)?;
        let mut routes_by_token = Vec::with_capacity(tokens);
        for token in 0..tokens {
            let mut routes = Vec::with_capacity(top_k);
            for slot in 0..top_k {
                let idx = token * top_k + slot;
                routes.push(ExpertRoute {
                    expert: indices[idx] as usize,
                    weight: weights[idx],
                    score: 0.0,
                    selection_score: 0.0,
                });
            }
            routes_by_token.push(routes);
        }
        Ok(routes_by_token)
    }

    pub(crate) fn output_head_topk_chunks(
        &mut self,
        slice: &ArtifactTensorSlice,
        hidden: &[f32],
        top_k: usize,
        chunk_rows: usize,
        reader: &ArtifactTensorReader,
    ) -> Result<Vec<TokenLogit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let mut rows =
            self.output_head_topk_chunks_rows(slice, hidden, 1, top_k, chunk_rows, reader)?;
        Ok(rows.pop().expect("one output-head input row requested"))
    }

    pub(crate) fn output_head_topk_chunks_rows(
        &mut self,
        slice: &ArtifactTensorSlice,
        hidden: &[f32],
        batch_rows: usize,
        top_k: usize,
        chunk_rows: usize,
        reader: &ArtifactTensorReader,
    ) -> Result<Vec<Vec<TokenLogit>>> {
        validate_output_head_rows_request(
            &slice.shape,
            hidden.len(),
            batch_rows,
            top_k,
            chunk_rows,
        )?;
        if batch_rows == 0 {
            return Ok(Vec::new());
        }
        if top_k == 0 {
            return Ok((0..batch_rows).map(|_| Vec::new()).collect());
        }

        let upload_start = profile_start(self.profile);
        if self
            .decode_arena
            .hidden
            .as_ref()
            .is_none_or(|buffer| buffer.len() != hidden.len())
        {
            self.decode_arena.hidden = Some(self.ops.zero_f32_buffer(hidden.len())?);
        }
        let mut hidden_device = self
            .decode_arena
            .hidden
            .take()
            .expect("output-head hidden buffer initialized above");
        if let Err(error) = self.ops.overwrite_f32_buffer(hidden, &mut hidden_device) {
            self.decode_arena.hidden = Some(hidden_device);
            return Err(error);
        }
        self.output_head_hidden_uploads = self.output_head_hidden_uploads.saturating_add(1);
        record_profile_duration(&mut self.output_head_hidden_upload_us, upload_start);
        let result = self.output_head_topk_chunks_rows_with_device(
            slice,
            &hidden_device,
            batch_rows,
            top_k,
            chunk_rows,
            reader,
        );
        self.decode_arena.hidden = Some(hidden_device);
        result
    }

    pub(crate) fn output_head_topk_chunks_with_device(
        &mut self,
        slice: &ArtifactTensorSlice,
        hidden: &ferrule_cuda::context::CudaF32Buffer,
        top_k: usize,
        chunk_rows: usize,
        reader: &ArtifactTensorReader,
    ) -> Result<Vec<TokenLogit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let mut rows = self.output_head_topk_chunks_rows_with_device(
            slice, hidden, 1, top_k, chunk_rows, reader,
        )?;
        Ok(rows.pop().expect("one output-head device row requested"))
    }

    pub(crate) fn output_head_topk_chunks_rows_with_device(
        &mut self,
        slice: &ArtifactTensorSlice,
        hidden: &ferrule_cuda::context::CudaF32Buffer,
        batch_rows: usize,
        top_k: usize,
        chunk_rows: usize,
        reader: &ArtifactTensorReader,
    ) -> Result<Vec<Vec<TokenLogit>>> {
        let (vocab_rows, _) = validate_output_head_rows_request(
            &slice.shape,
            hidden.len(),
            batch_rows,
            top_k,
            chunk_rows,
        )?;
        if batch_rows == 0 {
            return Ok(Vec::new());
        }
        if top_k == 0 {
            return Ok((0..batch_rows).map(|_| Vec::new()).collect());
        }
        self.output_head_calls = self.output_head_calls.saturating_add(1);
        let mut top = (0..batch_rows)
            .map(|_| Vec::<TokenLogit>::new())
            .collect::<Vec<_>>();
        let mut start = 0usize;
        while start < vocab_rows {
            let rows = chunk_rows.min(vocab_rows - start);
            self.output_head_chunks = self.output_head_chunks.saturating_add(1);
            self.output_head_rows = self.output_head_rows.saturating_add(rows as u64);
            let key = artifact_linear_row_cache_key(slice, start, rows)?;
            if !self.linears.contains_key(&key) {
                self.output_head_cache_misses = self.output_head_cache_misses.saturating_add(1);
                let read_start = profile_start(self.profile);
                let payload = reader.read_2d_rows(slice, start, rows)?;
                let linear = ArtifactLinearPayload::from_weight_and_scale(
                    TensorRole::OutputHead,
                    payload,
                    None,
                )?;
                record_profile_duration(&mut self.output_head_read_us, read_start);
                let actual_key = artifact_linear_cache_key(&linear);
                if actual_key != key {
                    return Err(Error::Model(format!(
                        "DeepSeek-V4 output-head cache-key mismatch: predicted {key}, materialized {actual_key}"
                    )));
                }
                let upload_start = profile_start(self.profile);
                let handle = self.upload_linear(&linear)?;
                record_profile_duration(&mut self.output_head_upload_us, upload_start);
                self.linears.insert(key.clone(), handle);
            } else {
                self.output_head_cache_hits = self.output_head_cache_hits.saturating_add(1);
            }
            let chunk_k = top_k.min(rows);
            let logits_key = (batch_rows, rows);
            if !self.output_head_logits.contains_key(&logits_key) {
                let logits_len = batch_rows.checked_mul(rows).ok_or_else(|| {
                    Error::Model("DeepSeek-V4 output-head logits workspace overflow".into())
                })?;
                let logits = self.ops.zero_f32_buffer(logits_len)?;
                self.output_head_logits.insert(logits_key, logits);
            }
            let output_len = batch_rows.checked_mul(chunk_k).ok_or_else(|| {
                Error::Model("DeepSeek-V4 output-head top-k workspace overflow".into())
            })?;
            if !self.output_head_indices.contains_key(&output_len) {
                self.output_head_indices
                    .insert(output_len, self.ops.zero_f32_buffer(output_len)?);
                self.output_head_values
                    .insert(output_len, self.ops.zero_f32_buffer(output_len)?);
            }
            if !self
                .output_head_linear_workspaces
                .contains_key(&hidden.len())
            {
                let workspace = self
                    .ops
                    .artifact_linear_workspace(batch_rows, hidden.len() / batch_rows)?;
                self.output_head_linear_workspaces
                    .insert(hidden.len(), workspace);
            }
            let handle = self.linears.get(&key).expect("inserted above");
            let logits = self
                .output_head_logits
                .get_mut(&logits_key)
                .expect("output-head rows logits workspace initialized above");
            let indices = self
                .output_head_indices
                .get_mut(&output_len)
                .expect("output-head rows indices workspace initialized above");
            let values = self
                .output_head_values
                .get_mut(&output_len)
                .expect("output-head rows values workspace initialized above");
            let linear_workspace = self
                .output_head_linear_workspaces
                .get_mut(&hidden.len())
                .expect("output-head linear workspace initialized above");
            let topk_start = profile_start(self.profile);
            self.ops
                .artifact_linear_rows_from_device_into_with_scratch(
                    handle,
                    hidden,
                    batch_rows,
                    logits,
                    linear_workspace,
                )?;
            let (chunk_indices, chunk_values) = self.ops.topk_vocab_rows_from_device_into(
                logits, batch_rows, rows, chunk_k, indices, values,
            )?;
            record_profile_duration(&mut self.output_head_topk_us, topk_start);
            let merge_start = profile_start(self.profile);
            merge_output_head_chunk(
                &mut top,
                &chunk_indices,
                &chunk_values,
                chunk_k,
                start,
                top_k,
            )?;
            record_profile_duration(&mut self.output_head_merge_us, merge_start);
            start += rows;
        }
        Ok(top)
    }

    pub(crate) fn ensure_linear_uploaded(
        &mut self,
        linear: &ArtifactLinearPayload,
    ) -> Result<String> {
        let key = artifact_linear_cache_key(linear);
        if !self.linears.contains_key(&key) {
            let handle = self.upload_linear(linear)?;
            self.linears.insert(key.clone(), handle);
        }
        Ok(key)
    }

    pub(crate) fn upload_linear(
        &self,
        linear: &ArtifactLinearPayload,
    ) -> Result<ferrule_cuda::context::CudaArtifactLinearHandle> {
        match linear.format {
            ArtifactLinearFormat::F32 {
                out_features,
                in_features,
            } => self
                .ops
                .upload_f32_linear(&linear.weight.bytes, out_features, in_features),
            ArtifactLinearFormat::Bf16 {
                out_features,
                in_features,
            } => self
                .ops
                .upload_bf16_linear(&linear.weight.bytes, out_features, in_features),
            ArtifactLinearFormat::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m,
                block_k,
            } => {
                let scale = linear.scale.as_ref().ok_or_else(|| {
                    Error::Model(format!(
                        "artifact linear {:?} CUDA FP8 weight is missing E8M0 scale tensor",
                        linear.role
                    ))
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
            ArtifactLinearFormat::Fp4E2M1PackedWithE8M0Scale {
                out_features,
                in_features,
                block_size: 32,
            } => {
                let scale = linear.scale.as_ref().ok_or_else(|| {
                    Error::Model(format!(
                        "artifact linear {:?} CUDA FP4 weight is missing E8M0 scale tensor",
                        linear.role
                    ))
                })?;
                self.ops.upload_fp4_e2m1_e8m0_linear(
                    &linear.weight.bytes,
                    &scale.bytes,
                    out_features,
                    in_features,
                    self.managed_experts,
                )
            }
            ArtifactLinearFormat::Fp4E2M1PackedWithE8M0Scale { block_size, .. } => {
                Err(Error::Model(format!(
                    "artifact linear {:?} CUDA FP4 block_size {block_size} is unsupported (expected 32)",
                    linear.role
                )))
            }
        }
    }

    pub(crate) fn rms_norm_heads_from_device_into(
        &self,
        input: &ferrule_cuda::context::CudaF32Buffer,
        heads: usize,
        head_dim: usize,
        eps: f32,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.ops
            .rms_norm_heads_from_device_into(input, heads, head_dim, eps, output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn routed_moe_prefill_batch_from_device_into(
        &mut self,
        layer: usize,
        input_dev: &ferrule_cuda::context::CudaF32Buffer,
        token_ids: &[u32],
        row_to_sequence: Option<&[usize]>,
        router: &RouterArtifactPayload,
        predicted_experts: &[usize],
        router_policy: &ExpertRouterPolicy,
        residency: &mut dyn ExpertResidencyControl,
        source_catalog: &ExpertSourceCatalog,
        prefetch_capacity: usize,
        reader: &ExpertStreamingReader,
        shared_expert: &SwiGluFfnPayload,
        router_logits_dev: &mut ferrule_cuda::context::CudaF32Buffer,
        router_indices_dev: &mut ferrule_cuda::context::CudaI32Buffer,
        router_weights_dev: &mut ferrule_cuda::context::CudaF32Buffer,
        linear_workspace: &mut ferrule_cuda::context::CudaArtifactLinearWorkspace,
        shared_workspace: &mut ferrule_cuda::context::CudaSwiGLUWorkspace,
        segment_workspace: &mut Option<ferrule_cuda::context::CudaMoeSegmentWorkspace>,
        route_output_dev: &mut ferrule_cuda::context::CudaF32Buffer,
        output_dev: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        let tokens = token_ids.len();
        if tokens == 0 {
            return Err(Error::Model(
                "CUDA routed MoE prefill batch requires at least one token".into(),
            ));
        }
        if row_to_sequence.is_some_and(|sequences| sequences.len() != tokens) {
            return Err(Error::Model(
                "CUDA routed MoE packed row/sequence metadata mismatch".into(),
            ));
        }
        let hidden_size = router.weight.format.in_features();
        if input_dev.len() != tokens * hidden_size {
            return Err(Error::Model(format!(
                "CUDA routed MoE prefill device input length mismatch: input={} expected {} tokens x {} hidden",
                input_dev.len(),
                tokens,
                hidden_size
            )));
        }

        if output_dev.len() != input_dev.len() {
            return Err(Error::Model(format!(
                "CUDA routed MoE prefill output length mismatch: output={} expected {}",
                output_dev.len(),
                input_dev.len()
            )));
        }

        let stage_start = profile_start(self.profile);
        self.linear_rows_from_device_into(
            &router.weight,
            input_dev,
            tokens,
            router_logits_dev,
            linear_workspace,
        )?;
        let routes_by_token = self.dsv4_router_topk_routes_from_device_logits(
            layer,
            router_logits_dev,
            token_ids,
            router,
            router_policy,
            router_indices_dev,
            router_weights_dev,
        )?;
        for routes in &routes_by_token {
            self.expert_selected = self.expert_selected.saturating_add(routes.len() as u64);
        }
        record_profile_duration(&mut self.moe_router_us, stage_start);

        let stage_start = profile_start(self.profile);
        let gate_key = self.ensure_linear_uploaded(&shared_expert.gate)?;
        let up_key = self.ensure_linear_uploaded(&shared_expert.up)?;
        let down_key = self.ensure_linear_uploaded(&shared_expert.down)?;
        self.ops.artifact_swiglu_ffn_rows_from_device_into(
            self.linears.get(&gate_key).expect("inserted above"),
            self.linears.get(&up_key).expect("inserted above"),
            self.linears.get(&down_key).expect("inserted above"),
            input_dev,
            tokens,
            1.0,
            shared_expert.swiglu_limit,
            shared_workspace,
        )?;
        self.ops
            .copy_f32_into_slot(shared_workspace.output(), output_dev, 0)?;
        record_profile_duration(&mut self.moe_shared_us, stage_start);

        let streaming_steps = self.routed_moe_prefill_segments_from_device(
            layer,
            input_dev,
            &routes_by_token,
            router_indices_dev,
            router_weights_dev,
            predicted_experts,
            residency,
            source_catalog,
            prefetch_capacity,
            reader,
            shared_expert.swiglu_limit,
            segment_workspace,
            route_output_dev,
            output_dev,
        )?;
        self.moe_access_events
            .push(ExpertBatchAccessEvent::from_routes_by_token(
                layer,
                ExpertAccessPhase::Prefill,
                tokens,
                &routes_by_token,
                &streaming_steps,
            ));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn routed_moe_prefill_segments_from_device(
        &mut self,
        layer: usize,
        input_dev: &ferrule_cuda::context::CudaF32Buffer,
        routes_by_token: &[Vec<ExpertRoute>],
        router_indices: &ferrule_cuda::context::CudaI32Buffer,
        router_weights: &ferrule_cuda::context::CudaF32Buffer,
        predicted_experts: &[usize],
        residency: &mut dyn ExpertResidencyControl,
        source_catalog: &ExpertSourceCatalog,
        prefetch_capacity: usize,
        reader: &ExpertStreamingReader,
        swiglu_limit: f32,
        segment_workspace: &mut Option<ferrule_cuda::context::CudaMoeSegmentWorkspace>,
        route_output: &mut ferrule_cuda::context::CudaF32Buffer,
        output_dev: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<Vec<ExpertStreamingStep>> {
        let tokens = routes_by_token.len();
        if tokens == 0 || !input_dev.len().is_multiple_of(tokens) {
            return Err(Error::Internal(format!(
                "CUDA segmented MoE invalid input shape: tokens={tokens} input={}",
                input_dev.len()
            )));
        }
        let hidden_size = input_dev.len() / tokens;
        if output_dev.len() != input_dev.len() {
            return Err(Error::Internal(format!(
                "CUDA segmented MoE output mismatch: input={} output={}",
                input_dev.len(),
                output_dev.len()
            )));
        }
        let routes_per_token = routes_by_token
            .first()
            .map(Vec::len)
            .filter(|&routes| routes > 0)
            .ok_or_else(|| Error::Internal("CUDA segmented MoE has no routes".into()))?;
        let route_count = tokens
            .checked_mul(routes_per_token)
            .ok_or_else(|| Error::Internal("CUDA segmented MoE route count overflow".into()))?;
        if route_count > i32::MAX as usize || tokens > i32::MAX as usize {
            return Err(Error::Internal(format!(
                "CUDA segmented MoE exceeds i32 metadata ABI: tokens={tokens} routes={route_count}"
            )));
        }

        let mut unique_experts = BTreeSet::new();
        for (token, routes) in routes_by_token.iter().enumerate() {
            if routes.len() != routes_per_token {
                return Err(Error::Internal(format!(
                    "CUDA segmented MoE route count mismatch at token {token}: got {} expected {routes_per_token}",
                    routes.len()
                )));
            }
            unique_experts.extend(routes.iter().map(|route| route.expert));
        }
        let unique_experts = unique_experts.into_iter().collect::<Vec<_>>();
        if unique_experts.is_empty() {
            return Err(Error::Internal(
                "CUDA segmented MoE selected no experts".into(),
            ));
        }

        let expected_route_output = route_count
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal("CUDA segmented MoE route output overflow".into()))?;
        if route_output.len() != expected_route_output {
            return Err(Error::Internal(format!(
                "CUDA segmented MoE route output mismatch: got {} expected {expected_route_output}",
                route_output.len()
            )));
        }

        let resident_slots = residency
            .requirements()
            .layer_capacity(
                u32::try_from(layer)
                    .map_err(|_| Error::Model(format!("expert layer {layer} exceeds u32")))?,
            )
            .ok_or_else(|| {
                Error::Internal(format!(
                    "expert residency controller has no capacity for layer {layer}"
                ))
            })?
            .clamp(1, DSV4_EXPERT_TABLE_CAPACITY);
        let segment_capacity = fixed_eight_segment_capacity(route_count, resident_slots)?;
        let compute_start = profile_start(self.profile);
        let mut input_prepared = false;
        let mut expected_intermediate = None;
        let mut streaming_steps = Vec::new();

        for selected in unique_experts.chunks(resident_slots) {
            self.moe_predicted_experts = self
                .moe_predicted_experts
                .saturating_add(predicted_experts.len() as u64);
            let stage_start = profile_start(self.profile);
            let CudaMoeMaterialization { streaming, leases } = self.materialize_selected_experts(
                layer,
                selected,
                predicted_experts,
                residency,
                source_catalog,
                prefetch_capacity,
                reader,
            )?;
            record_profile_duration(&mut self.moe_plan_us, stage_start);

            let chunk_result = (|| -> Result<()> {
                let first_expert = selected
                    .first()
                    .copied()
                    .ok_or_else(|| Error::Internal("CUDA segmented MoE empty window".into()))?;
                let first_handles = self
                    .experts
                    .get(&ExpertId::new(layer, first_expert))
                    .ok_or_else(|| {
                        Error::Model(format!(
                            "CUDA segmented MoE missing layer {layer} expert {first_expert}"
                        ))
                    })?;
                let intermediate_size = first_handles.gate.shape().out_features();
                if first_handles.down.shape().out_features() != hidden_size {
                    return Err(Error::Model(format!(
                        "CUDA segmented MoE hidden mismatch: expert={} input={hidden_size}",
                        first_handles.down.shape().out_features()
                    )));
                }
                if let Some(expected) = expected_intermediate {
                    if intermediate_size != expected {
                        return Err(Error::Model(format!(
                            "CUDA segmented MoE intermediate mismatch: got {intermediate_size} expected {expected}"
                        )));
                    }
                } else {
                    expected_intermediate = Some(intermediate_size);
                }

                let workspace_needs_init = segment_workspace
                    .as_ref()
                    .map(|workspace| {
                        !workspace.matches(
                            DSV4_EXPERT_TABLE_CAPACITY,
                            segment_capacity,
                            tokens,
                            hidden_size,
                            intermediate_size,
                            hidden_size,
                        )
                    })
                    .unwrap_or(true);
                if workspace_needs_init {
                    *segment_workspace = Some(self.ops.moe_segment_workspace(
                        DSV4_EXPERT_TABLE_CAPACITY,
                        segment_capacity,
                        tokens,
                        hidden_size,
                        intermediate_size,
                        hidden_size,
                    )?);
                    input_prepared = false;
                }
                if !input_prepared {
                    let workspace = segment_workspace
                        .as_mut()
                        .expect("segmented MoE workspace initialized above");
                    self.ops.prepare_moe_segment_input_from_device(
                        input_dev,
                        tokens,
                        hidden_size,
                        workspace,
                    )?;
                    self.ops.begin_moe_segment_invocation(
                        routes_per_token,
                        workspace,
                        route_output,
                    )?;
                    input_prepared = true;
                }

                let table = self.expert_slot_tables.get(&layer).ok_or_else(|| {
                    Error::Internal(format!(
                        "CUDA stable expert slot table missing for routed layer {layer}"
                    ))
                })?;
                let workspace = segment_workspace
                    .as_mut()
                    .expect("segmented MoE workspace initialized above");
                self.ops.prepare_moe_segment_grouping_stable(
                    table,
                    router_indices,
                    router_weights,
                    route_count,
                    routes_per_token,
                    workspace,
                )?;
                self.ops.moe_expert_segments_stable_from_prepared(
                    table,
                    routes_per_token,
                    swiglu_limit,
                    workspace,
                    route_output,
                )?;
                Ok(())
            })();
            let release_result = Self::release_expert_leases(residency, leases);
            match (chunk_result, release_result) {
                (Err(error), Ok(())) => return Err(error),
                (Ok(()), Err(error)) => return Err(error),
                (Err(error), Err(release_error)) => {
                    return Err(Error::Internal(format!(
                        "CUDA segmented MoE dispatch failed ({error}); releasing expert leases also failed ({release_error})"
                    )));
                }
                (Ok(()), Ok(())) => {}
            }
            if !predicted_experts.is_empty() {
                self.prefetch_predicted_experts(
                    layer,
                    predicted_experts,
                    residency,
                    source_catalog,
                    prefetch_capacity,
                    reader,
                )?;
            }
            streaming_steps.push(streaming);
        }

        let workspace = segment_workspace
            .as_mut()
            .expect("segmented MoE workspace initialized above");
        self.ops.reduce_moe_segment_route_outputs_ranked(
            route_output,
            tokens,
            routes_per_token,
            hidden_size,
            workspace,
            output_dev,
        )?;
        record_profile_duration(&mut self.moe_compute_submit_us, compute_start);
        Ok(streaming_steps)
    }

    #[allow(clippy::too_many_arguments)]
    fn route_decode_moe_from_device_into(
        &mut self,
        layer: usize,
        input: &ferrule_cuda::context::CudaF32Buffer,
        token_id: u32,
        router: &RouterArtifactPayload,
        router_policy: &ExpertRouterPolicy,
        router_input: &mut ferrule_cuda::context::CudaF32Buffer,
        logits_dev: &mut ferrule_cuda::context::CudaF32Buffer,
        indices_dev: &mut ferrule_cuda::context::CudaI32Buffer,
        weights_dev: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<CudaMoeRoutes> {
        let stage_start = profile_start(self.profile);
        self.ops.copy_f32_into_slot(input, router_input, 0)?;
        self.linear_matvec_from_device_into(&router.weight, router_input, logits_dev)?;
        let mut routes_by_token = self.dsv4_router_topk_routes_from_device_logits(
            layer,
            logits_dev,
            std::slice::from_ref(&token_id),
            router,
            router_policy,
            indices_dev,
            weights_dev,
        )?;
        let routes = routes_by_token.pop().unwrap_or_default();
        record_profile_duration(&mut self.moe_router_us, stage_start);

        let stage_start = profile_start(self.profile);
        let selected = routes.iter().map(|route| route.expert).collect::<Vec<_>>();
        record_profile_duration(&mut self.moe_routing_us, stage_start);
        Ok(CudaMoeRoutes { routes, selected })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn routed_moe_step_device_output_into(
        &mut self,
        layer: usize,
        input: &ferrule_cuda::context::CudaF32Buffer,
        token_id: u32,
        router: &RouterArtifactPayload,
        predicted_experts: &[usize],
        router_policy: &ExpertRouterPolicy,
        residency: &mut dyn ExpertResidencyControl,
        source_catalog: &ExpertSourceCatalog,
        prefetch_capacity: usize,
        reader: &ExpertStreamingReader,
        shared_expert: &SwiGluFfnPayload,
        router_input: &mut ferrule_cuda::context::CudaF32Buffer,
        router_logits: &mut ferrule_cuda::context::CudaF32Buffer,
        router_indices: &mut ferrule_cuda::context::CudaI32Buffer,
        router_weights: &mut ferrule_cuda::context::CudaF32Buffer,
        shared_workspace: &mut ferrule_cuda::context::CudaSwiGLUWorkspace,
        accumulator: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<RoutedMoeStepOutput> {
        let CudaMoeRoutes { routes, selected } = self.route_decode_moe_from_device_into(
            layer,
            input,
            token_id,
            router,
            router_policy,
            router_input,
            router_logits,
            router_indices,
            router_weights,
        )?;
        self.moe_predicted_experts = self
            .moe_predicted_experts
            .saturating_add(predicted_experts.len() as u64);
        self.expert_selected = self.expert_selected.saturating_add(routes.len() as u64);
        let CudaMoeMaterialization { streaming, leases } = self.materialize_selected_experts(
            layer,
            &selected,
            predicted_experts,
            residency,
            source_catalog,
            prefetch_capacity,
            reader,
        )?;

        let execution_result = (|| -> Result<RoutedMoeStepOutput> {
            let stage_start = profile_start(self.profile);
            let swiglu_limit = shared_expert.swiglu_limit;
            let gate_key = self.ensure_linear_uploaded(&shared_expert.gate)?;
            let up_key = self.ensure_linear_uploaded(&shared_expert.up)?;
            let down_key = self.ensure_linear_uploaded(&shared_expert.down)?;
            self.ops.artifact_swiglu_ffn_from_device_into(
                self.linears.get(&gate_key).expect("inserted above"),
                self.linears.get(&up_key).expect("inserted above"),
                self.linears.get(&down_key).expect("inserted above"),
                input,
                1.0,
                swiglu_limit,
                shared_workspace,
            )?;
            self.ops
                .copy_f32_into_slot(shared_workspace.output(), accumulator, 0)?;
            record_profile_duration(&mut self.moe_shared_us, stage_start);

            if !routes.is_empty() {
                let stage_start = profile_start(self.profile);
                let num_experts = routes.len();
                let first_expert_id = ExpertId::new(layer, routes[0].expert);
                let first_expert = self.experts.get(&first_expert_id).ok_or_else(|| {
                    Error::Model(format!(
                        "CUDA expert handle missing for layer {} expert {}",
                        first_expert_id.layer, first_expert_id.expert
                    ))
                })?;
                let intermediate_size = first_expert.gate.shape().out_features();
                let hidden_size = first_expert.down.shape().out_features();
                let workspace_key = (6, input.len(), intermediate_size, hidden_size);
                if !self
                    .decode_arena
                    .moe_workspaces
                    .contains_key(&workspace_key)
                {
                    let workspace = self.ops.moe_batched_workspace(
                        6,
                        input.len(),
                        intermediate_size,
                        hidden_size,
                    )?;
                    self.decode_arena
                        .moe_workspaces
                        .insert(workspace_key, workspace);
                }
                if !self
                    .decode_arena
                    .moe_resolve_workspaces
                    .contains_key(&layer)
                {
                    let resolve = self.ops.expert_route_resolve_workspace(6, 6)?;
                    self.decode_arena
                        .moe_resolve_workspaces
                        .insert(layer, resolve);
                }
                let table = self.expert_slot_tables.get(&layer).ok_or_else(|| {
                    Error::Internal(format!(
                        "CUDA stable expert slot table missing for routed layer {layer}"
                    ))
                })?;
                let ops = &self.ops;
                let resolve = self
                    .decode_arena
                    .moe_resolve_workspaces
                    .get_mut(&layer)
                    .expect("MoE resolve workspace initialized above");
                let workspace = self
                    .decode_arena
                    .moe_workspaces
                    .get_mut(&workspace_key)
                    .expect("MoE workspace initialized above");
                record_profile_duration(&mut self.moe_workspace_us, stage_start);
                let stage_start = profile_start(self.profile);
                ops.prepare_moe_experts_batched_workspace_stable(
                    table,
                    &selected[..num_experts],
                    router_indices,
                    router_weights,
                    num_experts,
                    input.len(),
                    intermediate_size,
                    hidden_size,
                    resolve,
                    workspace,
                )?;
                ops.moe_experts_batched_add_into_from_device_prepared(
                    input,
                    swiglu_limit,
                    num_experts,
                    intermediate_size,
                    hidden_size,
                    workspace,
                    accumulator,
                )?;
                record_profile_duration(&mut self.moe_compute_submit_us, stage_start);
            }

            Ok(RoutedMoeStepOutput {
                routes,
                streaming,
                routed_output: Vec::new(),
                shared_output: None,
                output: Vec::new(),
            })
        })();
        let release_result = Self::release_expert_leases(residency, leases);
        match (execution_result, release_result) {
            (Ok(output), Ok(())) => Ok(output),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(release_error)) => Err(release_error),
            (Err(error), Err(release_error)) => Err(Error::Internal(format!(
                "CUDA decode MoE failed ({error}); releasing expert leases also failed ({release_error})"
            ))),
        }
    }

    fn selected_upload_ticket(
        &mut self,
        bundle: &ExpertComputeBundle,
    ) -> Result<CudaExpertUploadTicket> {
        let pinned = self.pinned_bundle_for_upload(bundle)?;
        let ticket = self.upload_pinned_expert_bundle_async(&pinned)?;
        self.expert_async_upload_bytes = self
            .expert_async_upload_bytes
            .saturating_add(ticket.bytes());
        Ok(ticket)
    }

    fn queue_selected_upload_wait(
        &mut self,
        ticket: CudaExpertUploadTicket,
    ) -> Result<CudaFp4ExpertHandles> {
        let wait_start = profile_start(self.profile);
        if !ticket.is_complete()? {
            self.expert_selected_upload_waits = self.expert_selected_upload_waits.saturating_add(1);
        }
        self.ops.wait_upload_event(ticket.event())?;
        record_profile_duration(&mut self.expert_selected_upload_wait_us, wait_start);
        Ok(ticket.into_handles())
    }

    fn upload_pinned_expert_bundle_async(
        &self,
        bundle: &CudaPinnedExpertBundle,
    ) -> Result<CudaExpertUploadTicket> {
        let gate = self.upload_pinned_expert_linear_async(&bundle.gate)?;
        let up = match self.upload_pinned_expert_linear_async(&bundle.up) {
            Ok(up) => up,
            Err(error) => {
                let _ = self.ops.sync_upload_stream();
                return Err(error);
            }
        };
        let down = match self.upload_pinned_expert_linear_async(&bundle.down) {
            Ok(down) => down,
            Err(error) => {
                let _ = self.ops.sync_upload_stream();
                return Err(error);
            }
        };
        let event = match self.ops.record_upload_event() {
            Ok(event) => event,
            Err(error) => {
                let _ = self.ops.sync_upload_stream();
                return Err(error);
            }
        };
        Ok(CudaExpertUploadTicket {
            gate: Some(gate),
            up: Some(up),
            down: Some(down),
            bytes: bundle.bytes,
            staging: Some(bundle.clone()),
            event: Some(event),
        })
    }

    fn upload_pinned_expert_linear_async(
        &self,
        linear: &CudaPinnedExpertLinear,
    ) -> Result<ferrule_cuda::context::CudaArtifactLinearAsyncUpload> {
        let ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
            out_features,
            in_features,
            block_size: 32,
        } = linear.format.clone()
        else {
            return Err(Error::Model(format!(
                "CUDA routed expert {:?} requires artifact FP4 block_size=32, got {:?}",
                linear.matrix, linear.format
            )));
        };
        self.ops.upload_fp4_e2m1_e8m0_linear_from_pinned_async(
            linear.weight.clone(),
            linear.scale.clone(),
            out_features,
            in_features,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sparse_attention_with_device_query_values_topk_into(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        values: &ferrule_cuda::context::CudaF32Buffer,
        topk: &ferrule_cuda::context::CudaI32Buffer,
        sink: &[f32],
        tokens: usize,
        kv_len: usize,
        spec: SparseAttentionSpec,
        layer: usize,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        if topk.len() < tokens * spec.topk {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} CUDA sparse attention device topk too small: got {} expected at least {}",
                topk.len(),
                tokens * spec.topk
            )));
        }
        let shape = ferrule_cuda::transformer::sparse_attention::CudaSparseAttentionShape {
            batch_size: 1,
            tokens_per_batch: tokens,
            kv_len,
            heads: spec.heads,
            head_dim: spec.head_dim,
            topk: spec.topk,
            softmax_scale: spec.softmax_scale,
        };
        let sink_name = format!("sink_L{layer}");
        self.ensure_sink_buffer(&sink_name, sink)?;
        let sink_buf = self.sink_buffers.get(&sink_name).expect("inserted above");
        self.ops.sparse_attention_sink_from_device_into(
            query,
            values,
            topk.as_device_buffer(),
            sink_buf,
            shape,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn paged_window_sparse_attention_rows_into(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        positions: &[usize],
        row_kv_lens: &ferrule_cuda::context::CudaI32Buffer,
        sink: &[f32],
        rows: usize,
        layer: usize,
        spec: SparseAttentionSpec,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        if positions.len() != rows || row_kv_lens.len() != rows {
            return Err(Error::Model(format!(
                "packed window attention row metadata mismatch: positions={} visible_lens={} rows={rows}",
                positions.len(),
                row_kv_lens.len()
            )));
        }
        let elements = rows
            .checked_mul(spec.topk)
            .ok_or_else(|| Error::Model("packed window top-k size overflow".into()))?;
        let mut logical = vec![-1i32; elements];
        for (row, position) in positions.iter().copied().enumerate() {
            let kv_len = position
                .checked_add(1)
                .ok_or_else(|| Error::Model("packed window position overflow".into()))?;
            let start = kv_len.saturating_sub(spec.topk);
            for (output, index) in logical[row * spec.topk..(row + 1) * spec.topk]
                .iter_mut()
                .zip(start..kv_len)
            {
                *output = i32::try_from(index)
                    .map_err(|_| Error::Model("packed window index exceeds i32 ABI".into()))?;
            }
        }
        self.ensure_topk_buffer(elements)?;
        self.ops.overwrite_i32_buffer(
            &logical,
            self.topk_buffers.get_mut(&elements).expect("ensured above"),
        )?;
        let sink_name = format!("sink_L{layer}");
        self.ensure_sink_buffer(&sink_name, sink)?;
        let active = self.active_paged_kv.as_ref().ok_or_else(|| {
            Error::Model("packed window attention requires an active paged binding".into())
        })?;
        if active.row_sequence_ids_device.len() != rows {
            return Err(Error::Model(format!(
                "packed window attention selector rows mismatch: got {} expected {rows}",
                active.row_sequence_ids_device.len()
            )));
        }
        let pool = self.kv_page_pool.as_ref().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
        })?;
        let plane = pool
            .plane_storage(0)
            .ok_or_else(|| Error::Model("window KV plane is missing".into()))?;
        let descriptor = &pool.planes()[0];
        let layout = ferrule_cuda::PagedSparseAttentionLayout {
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
        self.ops
            .paged_sparse_attention_selected_rows_from_device_into(
                query,
                plane,
                &active.block_slots_device,
                &active.block_offsets_device,
                &active.kv_len_device,
                &active.row_sequence_ids_device,
                row_kv_lens,
                self.topk_buffers.get(&elements).expect("filled above"),
                self.sink_buffers.get(&sink_name).expect("inserted above"),
                layout,
                output,
            )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dual_plane_paged_sparse_attention_rows_into(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        topk: &ferrule_cuda::context::CudaI32Buffer,
        selectors: &ferrule_cuda::context::CudaI32Buffer,
        row_kv_lens: &ferrule_cuda::context::CudaI32Buffer,
        sink: &[f32],
        rows: usize,
        layer: usize,
        spec: SparseAttentionSpec,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        let sink_name = format!("sink_L{layer}");
        self.ensure_sink_buffer(&sink_name, sink)?;
        let active = self.active_paged_kv.as_ref().ok_or_else(|| {
            Error::Model("packed sparse attention requires an active paged binding".into())
        })?;
        if active.row_sequence_ids_device.len() != rows || row_kv_lens.len() != rows {
            return Err(Error::Model(format!(
                "packed sparse attention row metadata mismatch: selectors={} visible_lens={} expected={rows}",
                active.row_sequence_ids_device.len(),
                row_kv_lens.len()
            )));
        }
        let pool = self.kv_page_pool.as_ref().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
        })?;
        let first = pool
            .plane_storage(0)
            .ok_or_else(|| Error::Model("window KV plane is missing".into()))?;
        let second = pool
            .plane_storage(1)
            .ok_or_else(|| Error::Model("compressed KV plane is missing".into()))?;
        let first_descriptor = &pool.planes()[0];
        let second_descriptor = &pool.planes()[1];
        let layout = ferrule_cuda::DualPlanePagedSparseAttentionLayout {
            base: ferrule_cuda::PagedSparseAttentionLayout {
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
        self.ops
            .dual_plane_paged_sparse_attention_selected_rows_from_device_into(
                query,
                first,
                second,
                &active.block_slots_device,
                &active.block_offsets_device,
                &active.kv_len_device,
                &active.row_sequence_ids_device,
                row_kv_lens,
                topk,
                selectors,
                self.sink_buffers.get(&sink_name).expect("inserted above"),
                layout,
                output,
            )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sparse_attention_with_paged_kv_topk_into(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        layer: usize,
        position: usize,
        window_size: usize,
        topk: &ferrule_cuda::context::CudaI32Buffer,
        sink: &[f32],
        tokens: usize,
        spec: SparseAttentionSpec,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        if topk.len() < tokens * spec.topk {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} CUDA combined sparse attention device topk too small: got {} expected at least {}",
                topk.len(),
                tokens * spec.topk
            )));
        }
        let (page_tokens, layer_count) = self
            .active_paged_kv
            .as_ref()
            .map(|active| (active.page_tokens, active.layer_count))
            .ok_or_else(|| {
                Error::Model("DeepSeek-V4 sparse attention requires an active paged binding".into())
            })?;
        {
            let elements = tokens
                .checked_mul(spec.topk)
                .ok_or_else(|| Error::Model("paged combined top-k size overflow".into()))?;
            if self
                .paged_topk_logical
                .as_ref()
                .is_none_or(|buffer| buffer.len() != elements)
            {
                self.paged_topk_logical = Some(self.ops.zero_i32_buffer(elements)?);
                self.paged_topk_selectors = Some(self.ops.zero_i32_buffer(elements)?);
            }
            self.ops.convert_combined_ring_topk_indices_into(
                topk,
                ferrule_cuda::CombinedRingWindowLens::PositionDerived,
                ferrule_cuda::CombinedRingTopkLayout {
                    rows: tokens,
                    topk: spec.topk,
                    start_position: position,
                    position_stride: 1,
                    window_size,
                },
                self.paged_topk_logical.as_mut().expect("ensured above"),
                self.paged_topk_selectors.as_mut().expect("ensured above"),
            )?;
            let sink_name = format!("sink_L{layer}");
            self.ensure_sink_buffer(&sink_name, sink)?;
            let active = self.active_paged_kv.as_ref().expect("validated above");
            let pool = self.kv_page_pool.as_ref().ok_or_else(|| {
                Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
            })?;
            let first = pool
                .plane_storage(0)
                .ok_or_else(|| Error::Model("window KV plane is missing".into()))?;
            let second = pool
                .plane_storage(1)
                .ok_or_else(|| Error::Model("compressed KV plane is missing".into()))?;
            let first_descriptor = &pool.planes()[0];
            let second_descriptor = &pool.planes()[1];
            let layout = ferrule_cuda::DualPlanePagedSparseAttentionLayout {
                base: ferrule_cuda::PagedSparseAttentionLayout {
                    batch_size: 1,
                    tokens_per_sequence: tokens,
                    heads: spec.heads,
                    head_dim: spec.head_dim,
                    topk: spec.topk,
                    page_tokens,
                    elements_per_token: first_descriptor.elements_per_token,
                    layer_index: layer,
                    layer_count,
                    softmax_scale: spec.softmax_scale,
                },
                second_elements_per_token: second_descriptor.elements_per_token,
            };
            return self
                .ops
                .dual_plane_paged_sparse_attention_sink_from_device_into(
                    query,
                    first,
                    second,
                    &active.block_slots_device,
                    &active.block_offsets_device,
                    &active.kv_len_device,
                    self.paged_topk_logical.as_ref().expect("ensured above"),
                    self.paged_topk_selectors.as_ref().expect("ensured above"),
                    self.sink_buffers.get(&sink_name).expect("inserted above"),
                    layout,
                    output,
                );
        }
    }

    /// Batched device-resident grouped output_a into caller-owned output.
    pub(crate) fn grouped_output_a_rows_from_device_into(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &ferrule_cuda::context::CudaF32Buffer,
        rows: usize,
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        check_linear(
            layer,
            "wo_a",
            output_a,
            cfg.output_latent_dim(),
            cfg.output_group_input_dim(),
        )?;
        if rows == 0 || context.len() != rows * cfg.q_full_dim() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} CUDA grouped wo_a device rows context length mismatch: rows={} expected {}, got {}",
                rows,
                rows * cfg.q_full_dim(),
                context.len()
            )));
        }
        if output.len() != rows * cfg.output_latent_dim() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} CUDA grouped wo_a output length mismatch: expected {}, got {}",
                rows * cfg.output_latent_dim(),
                output.len()
            )));
        }
        let group_in = cfg.output_group_input_dim();
        let linear_key = self.ensure_linear_uploaded(output_a)?;
        let handle = self
            .linears
            .get(&linear_key)
            .expect("grouped WO-A inserted above");
        if self.ops.grouped_output_a_bf16_mma_supported(
            handle,
            cfg.output_latent_dim(),
            group_in,
            cfg.o_lora_rank,
        ) {
            return self.ops.grouped_output_a_bf16_mma_from_device_into(
                context,
                rows,
                handle,
                cfg.output_latent_dim(),
                group_in,
                cfg.o_lora_rank,
                output,
            );
        }

        let key = format!("{}::grouped_wo_a_f32", output_a.weight.slice.name);
        if !self.grouped_wo_a_weights.contains_key(&key) {
            let weights = output_a.reference_weights_f32()?;
            let buf = self.ops.upload_f32_buffer(&weights)?;
            self.grouped_wo_a_weights.insert(key.clone(), buf);
        }
        let weight_buf = self.grouped_wo_a_weights.get(&key).expect("inserted above");
        self.ops.grouped_matvec_f32_rows_from_device_into(
            context,
            rows,
            weight_buf,
            cfg.output_latent_dim(),
            group_in,
            cfg.o_lora_rank,
            output,
        )
    }

    pub(crate) fn linear_rows_from_device_into(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &ferrule_cuda::context::CudaF32Buffer,
        rows: usize,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
        workspace: &mut ferrule_cuda::context::CudaArtifactLinearWorkspace,
    ) -> Result<()> {
        if rows == 0 || input.len() != rows * linear.format.in_features() {
            return Err(Error::Model(format!(
                "artifact linear {:?} device rows input length mismatch: rows={} expected {}, got {}",
                linear.role,
                rows,
                rows * linear.format.in_features(),
                input.len()
            )));
        }
        let key = self.ensure_linear_uploaded(linear)?;
        let handle = self.linears.get(&key).expect("inserted above");
        self.ops.artifact_linear_rows_from_device_into_with_scratch(
            handle, input, rows, output, workspace,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compressor_recurrent_seed_prefill(
        &mut self,
        state: &mut Option<ferrule_cuda::CudaCompressorRecurrentState>,
        needs_reset: &mut bool,
        name: &str,
        kv_rows: &ferrule_cuda::context::CudaF32Buffer,
        score_rows: &ferrule_cuda::context::CudaF32Buffer,
        ape: &[f32],
        tokens: usize,
        ratio: usize,
        head_dim: usize,
        out_dim: usize,
        overlap: bool,
    ) -> Result<usize> {
        if !self.compressor_ape_buffers.contains_key(name) {
            self.compressor_ape_buffers
                .insert(name.to_string(), self.ops.upload_f32_buffer(ape)?);
        }
        if state.is_none() {
            *state = Some(
                self.ops
                    .create_compressor_recurrent_state(ratio, head_dim, out_dim, overlap)?,
            );
        }
        let state = state.as_mut().expect("created above");
        if state.ratio() != ratio
            || state.head_dim() != head_dim
            || state.out_dim() != out_dim
            || state.overlap() != overlap
        {
            return Err(Error::Model(
                "DeepSeek-V4 compressor recurrent state shape mismatch".into(),
            ));
        }
        if *needs_reset {
            self.ops.reset_compressor_recurrent_state(state)?;
            *needs_reset = false;
        }
        let ape = self
            .compressor_ape_buffers
            .get(name)
            .expect("inserted above");
        self.ops
            .compressor_recurrent_seed_prefill(state, kv_rows, score_rows, ape, tokens)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compressor_recurrent_append_into(
        &mut self,
        state: &mut Option<ferrule_cuda::CudaCompressorRecurrentState>,
        needs_reset: &mut bool,
        name: &str,
        projected_kv: &ferrule_cuda::context::CudaF32Buffer,
        projected_score: &ferrule_cuda::context::CudaF32Buffer,
        ape: &[f32],
        position: usize,
        ratio: usize,
        head_dim: usize,
        out_dim: usize,
        overlap: bool,
        compressed: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<bool> {
        if !self.compressor_ape_buffers.contains_key(name) {
            self.compressor_ape_buffers
                .insert(name.to_string(), self.ops.upload_f32_buffer(ape)?);
        }
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
        let ape = self
            .compressor_ape_buffers
            .get(name)
            .expect("inserted above");
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
    pub(crate) fn compressor_prefill_softmax_from_device_into(
        &mut self,
        name: &str,
        kv_rows: &ferrule_cuda::context::CudaF32Buffer,
        score_rows: &ferrule_cuda::context::CudaF32Buffer,
        ape: &[f32],
        groups: usize,
        ratio: usize,
        head_dim: usize,
        out_dim: usize,
        overlap: bool,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        if !self.compressor_ape_buffers.contains_key(name) {
            let buffer = self.ops.upload_f32_buffer(ape)?;
            self.compressor_ape_buffers.insert(name.to_string(), buffer);
        }
        let ape = self
            .compressor_ape_buffers
            .get(name)
            .expect("compressor APE inserted above");
        self.ops.compressor_prefill_softmax_from_device_into(
            kv_rows, score_rows, ape, groups, ratio, head_dim, out_dim, overlap, output,
        )
    }

    pub(crate) fn concat_attention_values_device_buffers_into(
        &mut self,
        window_values: &ferrule_cuda::context::CudaF32Buffer,
        compressed_values: &ferrule_cuda::context::CudaF32Buffer,
        window_rows: usize,
        head_dim: usize,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.ops.concat_f32_buffers_into(
            window_values,
            compressed_values,
            window_rows,
            head_dim,
            output,
        )
    }

    pub(crate) fn write_topk_indices(
        &mut self,
        indices: &[isize],
        output: &mut ferrule_cuda::context::CudaI32Buffer,
    ) -> Result<()> {
        if output.len() < indices.len() {
            return Err(Error::Model(format!(
                "CUDA top-k output too small: need {}, got {}",
                indices.len(),
                output.len()
            )));
        }
        let indices = indices
            .iter()
            .copied()
            .map(|index| {
                i32::try_from(index)
                    .map_err(|_| Error::Model(format!("CUDA top-k index exceeds i32 ABI: {index}")))
            })
            .collect::<Result<Vec<_>>>()?;
        self.ops.overwrite_i32_prefix(&indices, output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_topk_indices_paged_indexer_into(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        weights: &ferrule_cuda::context::CudaF32Buffer,
        layer: usize,
        tokens: usize,
        window_size: usize,
        window_cols: usize,
        extra_cols: usize,
        value_offset: usize,
        compress_ratio: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        weight_scale: f32,
        output: &mut ferrule_cuda::context::CudaI32Buffer,
    ) -> Result<()> {
        let active = self.active_paged_kv.as_ref().ok_or_else(|| {
            Error::Model("DeepSeek-V4 prefill indexer requires an active paged binding".into())
        })?;
        let pool = self.kv_page_pool.as_ref().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
        })?;
        let indexer_plane = pool
            .plane_storage(2)
            .ok_or_else(|| Error::Model("indexer KV plane is missing".into()))?;
        self.ops
            .dsv4_prefill_topk_indices_paged_indexer_from_device_into(
                query,
                weights,
                indexer_plane,
                &active.block_slots_device,
                &active.block_offsets_device,
                tokens,
                window_size,
                window_cols,
                extra_cols,
                value_offset,
                compress_ratio,
                compressed_len,
                index_heads,
                index_head_dim,
                active.page_tokens,
                layer,
                active.layer_count,
                weight_scale,
                output,
            )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_topk_indices_fused_index_query_paged_indexer_into(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        weights: &ferrule_cuda::context::CudaF32Buffer,
        layer: usize,
        rope_name: &str,
        tokens: usize,
        window_size: usize,
        window_cols: usize,
        extra_cols: usize,
        value_offset: usize,
        compress_ratio: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        rope_dim: usize,
        start_position: usize,
        weight_scale: f32,
        output: &mut ferrule_cuda::context::CudaI32Buffer,
    ) -> Result<()> {
        let required_positions = start_position.checked_add(tokens).ok_or_else(|| {
            Error::Model("DeepSeek-V4 prefill indexer RoPE position overflow".into())
        })?;
        self.require_rope_tables(rope_name, rope_dim, required_positions)?;
        let active = self.active_paged_kv.as_ref().ok_or_else(|| {
            Error::Model("DeepSeek-V4 prefill indexer requires an active paged binding".into())
        })?;
        let pool = self.kv_page_pool.as_ref().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
        })?;
        let indexer_plane = pool
            .plane_storage(2)
            .ok_or_else(|| Error::Model("indexer KV plane is missing".into()))?;
        let cos = self.rope_cos_device(rope_name);
        let sin = self.rope_sin_device(rope_name);
        self.ops
            .dsv4_prefill_topk_indices_fused_index_query_paged_indexer_from_device_into(
                query,
                weights,
                indexer_plane,
                &active.block_slots_device,
                &active.block_offsets_device,
                cos,
                sin,
                tokens,
                window_size,
                window_cols,
                extra_cols,
                value_offset,
                compress_ratio,
                compressed_len,
                index_heads,
                index_head_dim,
                rope_dim,
                start_position,
                active.page_tokens,
                layer,
                active.layer_count,
                weight_scale,
                output,
            )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_topk_indices_paged_indexer_rows_into(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        weights: &ferrule_cuda::context::CudaF32Buffer,
        positions: &ferrule_cuda::context::CudaI32Buffer,
        window_lens: &ferrule_cuda::context::CudaI32Buffer,
        compressed_lens: &ferrule_cuda::context::CudaI32Buffer,
        layer: usize,
        window_size: usize,
        index_topk: usize,
        index_heads: usize,
        index_head_dim: usize,
        weight_scale: f32,
        logical_indices: &mut ferrule_cuda::context::CudaI32Buffer,
        plane_selectors: &mut ferrule_cuda::context::CudaI32Buffer,
    ) -> Result<()> {
        let rows = positions.len();
        if window_lens.len() != rows || compressed_lens.len() != rows {
            return Err(Error::Model(
                "packed decode top-k row metadata is inconsistent".into(),
            ));
        }
        let active = self.active_paged_kv.as_ref().ok_or_else(|| {
            Error::Model("packed decode top-k requires an active paged binding".into())
        })?;
        if active.row_sequence_ids_device.len() != rows {
            return Err(Error::Model(
                "packed decode top-k selector row count is inconsistent".into(),
            ));
        }
        let pool = self.kv_page_pool.as_ref().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
        })?;
        let indexer_plane = pool
            .plane_storage(2)
            .ok_or_else(|| Error::Model("indexer KV plane is missing".into()))?;
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_topk_indices_fused_index_query_paged_indexer_into(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        weights: &ferrule_cuda::context::CudaF32Buffer,
        layer: usize,
        rope_name: &str,
        position: usize,
        window_len: usize,
        window_size: usize,
        extra_cols: usize,
        value_offset: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        rope_dim: usize,
        weight_scale: f32,
        out: &mut ferrule_cuda::context::CudaI32Buffer,
    ) -> Result<()> {
        let required_positions = position.checked_add(1).ok_or_else(|| {
            Error::Model("DeepSeek-V4 decode indexer RoPE position overflow".into())
        })?;
        self.require_rope_tables(rope_name, rope_dim, required_positions)?;
        {
            let active = self.active_paged_kv.as_ref().ok_or_else(|| {
                Error::Model("DeepSeek-V4 decode indexer requires an active paged binding".into())
            })?;
            let pool = self.kv_page_pool.as_ref().ok_or_else(|| {
                Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
            })?;
            let indexer_plane = pool
                .plane_storage(2)
                .ok_or_else(|| Error::Model("indexer KV plane is missing".into()))?;
            let cos = self.rope_cos_device(rope_name);
            let sin = self.rope_sin_device(rope_name);
            return self
                .ops
                .dsv4_decode_topk_indices_fused_index_query_paged_indexer_from_device_into(
                    query,
                    weights,
                    indexer_plane,
                    &active.block_slots_device,
                    &active.block_offsets_device,
                    cos,
                    sin,
                    position,
                    window_len,
                    window_size,
                    extra_cols,
                    value_offset,
                    compressed_len,
                    index_heads,
                    index_head_dim,
                    rope_dim,
                    active.page_tokens,
                    layer,
                    active.layer_count,
                    weight_scale,
                    out,
                );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_topk_indices_from_paged_indexer_into(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        weights: &ferrule_cuda::context::CudaF32Buffer,
        layer: usize,
        position: usize,
        window_len: usize,
        window_size: usize,
        extra_cols: usize,
        value_offset: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        weight_scale: f32,
        out: &mut ferrule_cuda::context::CudaI32Buffer,
    ) -> Result<()> {
        {
            let active = self.active_paged_kv.as_ref().ok_or_else(|| {
                Error::Model("DeepSeek-V4 decode indexer requires an active paged binding".into())
            })?;
            let pool = self.kv_page_pool.as_ref().ok_or_else(|| {
                Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
            })?;
            let indexer_plane = pool
                .plane_storage(2)
                .ok_or_else(|| Error::Model("indexer KV plane is missing".into()))?;
            return self
                .ops
                .dsv4_decode_topk_indices_paged_indexer_from_device_into(
                    query,
                    weights,
                    indexer_plane,
                    &active.block_slots_device,
                    &active.block_offsets_device,
                    position,
                    window_len,
                    window_size,
                    extra_cols,
                    value_offset,
                    compressed_len,
                    index_heads,
                    index_head_dim,
                    active.page_tokens,
                    layer,
                    active.layer_count,
                    weight_scale,
                    out,
                );
        }
    }

    pub(crate) fn gather_f32_rows(
        &mut self,
        input: &ferrule_cuda::context::CudaF32Buffer,
        indices: &[i32],
        rows: usize,
        row_dim: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        if indices.len() != rows {
            return Err(Error::Model(format!(
                "CUDA gather rows index length mismatch: indices={} rows={rows}",
                indices.len()
            )));
        }
        let indices_dev = self.ops.upload_i32_buffer(indices)?;
        self.ops.gather_f32_rows(input, &indices_dev, rows, row_dim)
    }

    pub(crate) fn paged_scatter_rows_from_device(
        &mut self,
        plane: usize,
        layer: usize,
        values: &ferrule_cuda::context::CudaF32Buffer,
        positions: &ferrule_cuda::context::CudaI32Buffer,
        mask: Option<&ferrule_cuda::context::CudaI32Buffer>,
        row_dim: usize,
    ) -> Result<()> {
        let active = self.active_paged_kv.as_ref().ok_or_else(|| {
            Error::Model("packed paged KV scatter requires an active binding".into())
        })?;
        if active.row_sequence_ids_device.len() != positions.len()
            || values.len() != positions.len() * row_dim
        {
            return Err(Error::Model(
                "packed paged KV scatter row metadata is inconsistent".into(),
            ));
        }
        if mask.is_some_and(|mask| mask.len() != positions.len()) {
            return Err(Error::Model(
                "packed paged KV scatter mask length is inconsistent".into(),
            ));
        }
        let pool = self.kv_page_pool.as_mut().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
        })?;
        let descriptor = pool
            .planes()
            .get(plane)
            .ok_or_else(|| Error::Model(format!("paged KV plane {plane} is missing")))?;
        if descriptor.elements_per_token != row_dim || descriptor.layer_count != active.layer_count
        {
            return Err(Error::Model(format!(
                "paged KV plane {plane} layout mismatch: row_dim={row_dim}/{} layers={}/{}",
                descriptor.elements_per_token, active.layer_count, descriptor.layer_count
            )));
        }
        let layout = ferrule_cuda::PagedPlaneLayout {
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

    pub(crate) fn paged_write_rows(
        &mut self,
        plane: usize,
        layer: usize,
        values: &ferrule_cuda::context::CudaF32Buffer,
        start_position: usize,
        rows: usize,
        row_dim: usize,
    ) -> Result<()> {
        let active = self.active_paged_kv.as_ref().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA KV write requires an active paged binding".into())
        })?;
        if values.len() != rows.saturating_mul(row_dim) {
            return Err(Error::Model("paged KV source row shape mismatch".into()));
        }
        if layer >= active.layer_count || active.page_tokens == 0 {
            return Err(Error::Model(
                "paged KV layer/page metadata is invalid".into(),
            ));
        }
        if active.sequence_count != 1 {
            return Err(Error::Model(
                "contiguous-row paged write cannot target a packed multi-sequence binding".into(),
            ));
        }
        let slots = active.physical_block_slots.clone();
        let page_tokens = active.page_tokens;
        let pool = self.kv_page_pool.as_mut().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
        })?;
        let descriptor = pool
            .planes()
            .get(plane)
            .ok_or_else(|| Error::Model(format!("paged KV plane {plane} is missing")))?;
        if descriptor.elements_per_token != row_dim || descriptor.layer_count != active.layer_count
        {
            return Err(Error::Model(format!(
                "paged KV plane {plane} layout mismatch: row_dim={row_dim}/{} layers={}/{}",
                descriptor.elements_per_token, active.layer_count, descriptor.layer_count
            )));
        }
        let page_elements = pool
            .page_elements(plane)
            .ok_or_else(|| Error::Model(format!("paged KV plane {plane} has no storage")))?;
        let layer_stride = page_tokens
            .checked_mul(row_dim)
            .ok_or_else(|| Error::Model("paged KV layer stride overflow".into()))?;
        for row in 0..rows {
            let position = start_position
                .checked_add(row)
                .ok_or_else(|| Error::Model("paged KV position overflow".into()))?;
            let logical_page = position / page_tokens;
            let token_in_page = position % page_tokens;
            let physical_slot = *slots.get(logical_page).ok_or_else(|| {
                Error::Model(format!(
                    "paged KV position {position} has no physical block"
                ))
            })?;
            if physical_slot < 0 {
                return Err(Error::Model("paged KV block slot is negative".into()));
            }
            let destination = (physical_slot as usize)
                .checked_mul(page_elements)
                .and_then(|offset| offset.checked_add(layer.checked_mul(layer_stride)?))
                .and_then(|offset| offset.checked_add(token_in_page.checked_mul(row_dim)?))
                .ok_or_else(|| Error::Model("paged KV destination overflow".into()))?;
            self.ops.copy_f32_range(
                values,
                row * row_dim,
                pool.plane_storage_mut(plane)
                    .expect("validated paged KV plane"),
                destination,
                row_dim,
            )?;
        }
        Ok(())
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
            Error::Model(format!(
                "DeepSeek-V4 RoPE table '{name}' element count overflow: capacity={capacity} rope_dim={rope_dim}"
            ))
        })?;
        let mut cos = Vec::new();
        cos.try_reserve_exact(elements).map_err(|error| {
            Error::Model(format!(
                "DeepSeek-V4 RoPE cosine table '{name}' host allocation failed for {elements} elements: {error}"
            ))
        })?;
        cos.resize(elements, 0.0f32);
        let mut sin = Vec::new();
        sin.try_reserve_exact(elements).map_err(|error| {
            Error::Model(format!(
                "DeepSeek-V4 RoPE sine table '{name}' host allocation failed for {elements} elements: {error}"
            ))
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
            return Err(Error::Model(format!(
                "DeepSeek-V4 RoPE table '{name}' requires at least one position"
            )));
        }
        let table = self.rope_tables.get(name).ok_or_else(|| {
            Error::Model(format!("DeepSeek-V4 RoPE table '{name}' is not prepared"))
        })?;
        if table.rope_dim != rope_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 RoPE table '{name}' dimension mismatch: cached={} requested={rope_dim}",
                table.rope_dim
            )));
        }
        validate_rope_table_capacity(name, table, required_positions)
    }

    fn rope_cos_device(&self, name: &str) -> &ferrule_cuda::context::CudaF32Buffer {
        &self
            .rope_tables
            .get(name)
            .expect("rope tables must be required before reading")
            .cos
    }

    fn rope_sin_device(&self, name: &str) -> &ferrule_cuda::context::CudaF32Buffer {
        &self
            .rope_tables
            .get(name)
            .expect("rope tables must be required before reading")
            .sin
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope_tail_from_device(
        &mut self,
        name: &str,
        qk: &mut ferrule_cuda::context::CudaF32Buffer,
        position: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        if heads == 0 || rope_dim == 0 {
            return Ok(());
        }
        self.require_rope_tables(name, rope_dim as usize, position as usize + 1)?;
        let table = self
            .rope_tables
            .get(name)
            .expect("rope tables required immediately above");
        self.ops.rope_tail_from_device(
            qk, &table.cos, &table.sin, position, heads, head_dim, rope_dim, inverse,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope_tail_rows_from_device(
        &mut self,
        name: &str,
        qk: &mut ferrule_cuda::context::CudaF32Buffer,
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
        qk: &mut ferrule_cuda::context::CudaF32Buffer,
        positions: &ferrule_cuda::context::CudaI32Buffer,
        max_position: usize,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        if positions.is_empty() || heads == 0 || rope_dim == 0 {
            return Ok(());
        }
        let required_positions = max_position
            .checked_add(1)
            .ok_or_else(|| Error::Model("DeepSeek-V4 indexed RoPE position overflow".into()))?;
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
        qk: &mut ferrule_cuda::context::CudaF32Buffer,
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
            .ok_or_else(|| Error::Model("DeepSeek-V4 RoPE row-stride overflow".into()))?;
        let required_positions = (start_position as usize)
            .checked_add(last_offset)
            .and_then(|position| position.checked_add(1))
            .ok_or_else(|| Error::Model("DeepSeek-V4 RoPE position overflow".into()))?;
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

    // ── Device-resident top-k index buffer ───────────────────────────

    /// Ensure an exact-shape i32 device buffer is allocated for top-k indices.
    pub(crate) fn ensure_topk_buffer(&mut self, elements: usize) -> Result<()> {
        if !self.topk_buffers.contains_key(&elements) {
            self.topk_buffers
                .insert(elements, self.ops.zero_i32_buffer(elements)?);
        }
        Ok(())
    }

    fn fill_paged_window_topk(&mut self, position: usize, window_size: usize) -> Result<()> {
        self.ensure_topk_buffer(window_size)?;
        let kv_len = position
            .checked_add(1)
            .ok_or_else(|| Error::Model("paged attention position overflow".into()))?;
        let start = kv_len.saturating_sub(window_size);
        let mut indices = vec![-1i32; window_size];
        for (output, logical) in indices.iter_mut().zip(start..kv_len) {
            *output = i32::try_from(logical)
                .map_err(|_| Error::Model("paged attention index exceeds i32 ABI".into()))?;
        }
        self.ops.overwrite_i32_buffer(
            &indices,
            self.topk_buffers
                .get_mut(&window_size)
                .expect("ensured above"),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sparse_attention_topk_from_device_into(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        layer: usize,
        position: usize,
        window_size: usize,
        sink: &[f32],
        shape: ferrule_cuda::transformer::sparse_attention::CudaSparseAttentionShape,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        let (page_tokens, layer_count) = self
            .active_paged_kv
            .as_ref()
            .map(|active| (active.page_tokens, active.layer_count))
            .ok_or_else(|| {
                Error::Model("DeepSeek-V4 sparse attention requires an active paged binding".into())
            })?;
        {
            self.fill_paged_window_topk(position, window_size)?;
            let sink_name = format!("sink_L{layer}");
            self.ensure_sink_buffer(&sink_name, sink)?;
            let active = self.active_paged_kv.as_ref().expect("validated above");
            let pool = self.kv_page_pool.as_ref().ok_or_else(|| {
                Error::Model("DeepSeek-V4 CUDA physical KV pool is not configured".into())
            })?;
            let plane = pool
                .plane_storage(0)
                .ok_or_else(|| Error::Model("DeepSeek-V4 window KV plane is missing".into()))?;
            let descriptor = pool
                .planes()
                .first()
                .ok_or_else(|| Error::Model("DeepSeek-V4 window KV schema is missing".into()))?;
            let layout = ferrule_cuda::PagedSparseAttentionLayout {
                batch_size: 1,
                tokens_per_sequence: 1,
                heads: shape.heads,
                head_dim: shape.head_dim,
                topk: shape.topk,
                page_tokens,
                elements_per_token: descriptor.elements_per_token,
                layer_index: layer,
                layer_count,
                softmax_scale: shape.softmax_scale,
            };
            return self.ops.paged_sparse_attention_sink_from_device_into(
                query,
                plane,
                &active.block_slots_device,
                &active.block_offsets_device,
                &active.kv_len_device,
                self.topk_buffers.get(&window_size).expect("filled above"),
                self.sink_buffers.get(&sink_name).expect("inserted above"),
                layout,
                output,
            );
        }
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
    fn fixed_eight_segment_capacity_uses_resident_window_upper_bound() {
        assert_eq!(fixed_eight_segment_capacity(0, 0).unwrap(), 1);
        assert_eq!(fixed_eight_segment_capacity(8, 1).unwrap(), 1);
        assert_eq!(fixed_eight_segment_capacity(9, 1).unwrap(), 2);
        assert_eq!(fixed_eight_segment_capacity(9, 9).unwrap(), 9);
        assert_eq!(fixed_eight_segment_capacity(100, 4).unwrap(), 16);
    }

    #[test]
    fn fixed_eight_segment_capacity_rejects_overflow_and_u16_excess() {
        let error = fixed_eight_segment_capacity(usize::MAX, usize::MAX)
            .expect_err("arithmetic overflow must be rejected");
        assert!(error.to_string().contains("capacity overflow"));

        let error = fixed_eight_segment_capacity(8 * 65_536, 1)
            .expect_err("capacity above u16 must be rejected");
        assert!(error.to_string().contains("u16 limit 65535"));
    }

    #[test]
    fn output_head_rows_shape_validation_is_explicit() {
        assert_eq!(
            validate_output_head_rows_request(&[128, 4], 12, 3, 8, 32).unwrap(),
            (128, 4)
        );
        assert!(validate_output_head_rows_request(&[128], 12, 3, 8, 32).is_err());
        assert!(validate_output_head_rows_request(&[128, 4], 11, 3, 8, 32).is_err());
        assert!(validate_output_head_rows_request(&[128, 4], 12, 3, 8, 0).is_err());
        assert!(validate_output_head_rows_request(&[128, 4], 12, 3, 41, 32).is_err());
        assert_eq!(
            validate_output_head_rows_request(&[128, 4], 0, 0, 0, 32).unwrap(),
            (128, 4)
        );
    }

    #[test]
    fn output_head_rows_merge_is_stable_and_row_local() -> Result<()> {
        let mut top = vec![Vec::new(), Vec::new()];
        merge_output_head_chunk(
            &mut top,
            &[0.0, 1.0, 0.0, 1.0],
            &[5.0, 5.0, 1.0, 4.0],
            2,
            0,
            3,
        )?;
        merge_output_head_chunk(
            &mut top,
            &[0.0, 1.0, 0.0, 1.0],
            &[5.0, 4.0, 4.0, 6.0],
            2,
            2,
            3,
        )?;

        let row_zero = top[0]
            .iter()
            .map(|item| (item.token_id, item.logit))
            .collect::<Vec<_>>();
        let row_one = top[1]
            .iter()
            .map(|item| (item.token_id, item.logit))
            .collect::<Vec<_>>();
        assert_eq!(row_zero, vec![(0, 5.0), (1, 5.0), (2, 5.0)]);
        assert_eq!(row_one, vec![(3, 6.0), (1, 4.0), (2, 4.0)]);
        Ok(())
    }

    #[test]
    #[ignore = "requires cargo-oxide and CUDA"]
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
        operators.kv_page_pool = Some(ferrule_cuda::CudaKvPagePool::new(
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
            block_slots_device: operators.ops.upload_i32_buffer(&block_slots)?,
            block_offsets_device: operators.ops.upload_i32_buffer(&block_offsets)?,
            kv_len_device: operators.ops.upload_i32_buffer(&kv_lens)?,
            row_sequence_ids_device: operators.ops.upload_i32_buffer(&[0, 1])?,
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
    #[ignore = "requires cargo-oxide and CUDA"]
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
