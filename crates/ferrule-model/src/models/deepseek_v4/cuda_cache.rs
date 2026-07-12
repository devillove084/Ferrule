//! DeepSeek-V4 CUDA operator cache: device-resident weights, KV cache, MoE handles.

#![cfg(feature = "cuda")]

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::time::{Duration, Instant};

use ferrule_common::{Error, Result};

use crate::artifact::binding::RouterArtifactPayload;
use crate::artifact::linear::{artifact_linear_cache_key, artifact_linear_row_cache_key};
use crate::artifact::linear::{
    ArtifactActivationQuantization, ArtifactLinearFormat, ArtifactLinearPayload,
};
use crate::artifact::tensor::{ArtifactTensorReader, ArtifactTensorSlice};
use crate::attention_backend::SparseAttentionSpec;
use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::{
    HyperConnectionConfig, HyperConnectionHeadWeights, HyperConnectionWeights,
};
use crate::moe::handle::{
    CpuExpertHandleStore, ExpertHandleStore, ExpertResidentFormat, ResidentExpertHandle,
};
use crate::moe::prediction::{ExpertAccessPhase, ExpertBatchAccessEvent};
use crate::moe::routed::RoutedMoeStepOutput;
use crate::moe::routing::{
    ExpertRoute, ExpertRouterPolicy, RouterScoreFunction, RouterSelectionPolicy,
};
use crate::moe::streaming::{
    classify_expert_residency, read_experts_concurrent, AsyncHostStagedExpertLoader,
    ExpertComputeBundle, ExpertEvictRequest, ExpertId, ExpertLinearFormat, ExpertLinearPayload,
    ExpertLoadRequest, ExpertMatrixKind, ExpertResidencyPlan, ExpertResidencySelectedLoad,
    ExpertStorageTier, ExpertStreamingPlanner, ExpertStreamingReader, ExpertStreamingStep,
    HostStagedExpertCache,
};
use crate::runner::TokenLogit;
use crate::TensorRole;

use super::config::{DeepSeekV4AttentionConfig, DeepSeekV4RopeParams};
use super::helpers::{check_linear, rank_logits_desc, yarn_frequency};
use super::operators::DeepSeekV4OperatorRuntimeCounters;
use super::prepared::DeepSeekV4ExecutionPolicy;

const DSV4_ROPE_TABLE_MIN_CAPACITY: usize = 4096;

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

/// Device-resident KV ownership for one DeepSeek-V4 sequence/layer window.
///
/// This state deliberately lives with the host-semantic window cache rather than
/// the operator backend, so concurrent/forked sequences cannot address each
/// other's CUDA buffers through a backend-global layer key.
#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct DeepSeekV4CudaSequenceKvState {
    /// Device-resident window KV cache: `[window_size * head_dim]` f32.
    pub(crate) kv_cache: Option<ferrule_cuda::context::CudaF32Buffer>,
    /// Current device KV length, capped at `window_size`.
    pub(crate) kv_len: usize,
    /// Device-resident `[window KV | compressed KV]` values.
    pub(crate) combined_kv_cache: Option<ferrule_cuda::context::CudaF32Buffer>,
    /// Compressed slots allocated in `combined_kv_cache`.
    pub(crate) combined_kv_compressed_capacity: usize,
    /// Device-resident compressed indexer KV values.
    pub(crate) indexer_kv_cache: Option<ferrule_cuda::context::CudaF32Buffer>,
    /// Compressed slots allocated in `indexer_kv_cache`.
    pub(crate) indexer_kv_capacity: usize,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4CudaSequenceKvState {
    /// Reset sequence validity while retaining allocations for reuse.
    pub(crate) fn reset_for_reuse(&mut self) {
        self.kv_len = 0;
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeepSeekV4CudaSequenceKvState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepSeekV4CudaSequenceKvState")
            .field("has_kv_cache", &self.kv_cache.is_some())
            .field("kv_len", &self.kv_len)
            .field("has_combined_kv_cache", &self.combined_kv_cache.is_some())
            .field(
                "combined_kv_compressed_capacity",
                &self.combined_kv_compressed_capacity,
            )
            .field("has_indexer_kv_cache", &self.indexer_kv_cache.is_some())
            .field("indexer_kv_capacity", &self.indexer_kv_capacity)
            .finish()
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4CudaOperatorCache {
    pub(crate) ops: ferrule_cuda::context::CudaArtifactOperatorContext,
    managed_experts: bool,
    expert_upload_inflight: usize,
    device_router_topk: bool,
    moe_segment_batch: usize,
    linears: HashMap<String, ferrule_cuda::context::CudaArtifactLinearHandle>,
    experts: HashMap<ExpertId, CudaFp4ExpertHandles>,
    recycled_experts: Vec<CudaFp4ExpertHandles>,
    uploading_experts: HashMap<ExpertId, CudaExpertUploadTicket>,
    moe_access_events: Vec<ExpertBatchAccessEvent>,
    decode_arena: DeepSeekV4DecodeArena,
    host_staged_cache: HostStagedExpertCache,
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
    /// Cached dequantized f32 weights for grouped output_a, uploaded once.
    grouped_wo_a_weights: HashMap<String, ferrule_cuda::context::CudaF32Buffer>,

    /// Typed precomputed RoPE tables keyed by their stable layer/resource name.
    /// Each entry records its parameters and growable position capacity so a
    /// same-name shape/configuration mismatch cannot be silently reused.
    rope_tables: HashMap<String, CudaRopeTable>,
    /// Pre-allocated top-k index buffer `[window_size]` i32 for device-resident
    /// sparse attention.
    topk_buffer: Option<ferrule_cuda::context::CudaI32Buffer>,
    output_head_logits: HashMap<usize, ferrule_cuda::context::CudaF32Buffer>,
    output_head_indices: Option<ferrule_cuda::context::CudaF32Buffer>,
    output_head_values: Option<ferrule_cuda::context::CudaF32Buffer>,
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
    expert_planner_residency_syncs: u64,
    expert_planner_residency_synced: u64,
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
#[derive(Default)]
struct DeepSeekV4DecodeArena {
    hidden: Option<ferrule_cuda::context::CudaF32Buffer>,
    hc_input: Option<ferrule_cuda::context::CudaF32Buffer>,
    final_hidden: Option<ferrule_cuda::context::CudaF32Buffer>,
    final_norm: Option<ferrule_cuda::context::CudaF32Buffer>,
    /// Reusable routed MoE scratch/pointer buffers for the eager decode hot path.
    moe_workspace: Option<ferrule_cuda::context::CudaMoeBatchedWorkspace>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4DecodeBuffers {
    pub(crate) hc_input: ferrule_cuda::context::CudaF32Buffer,
    pub(crate) final_hidden: ferrule_cuda::context::CudaF32Buffer,
    pub(crate) final_norm: ferrule_cuda::context::CudaF32Buffer,
}

#[cfg(feature = "cuda")]
struct CudaFp4ExpertHandles {
    gate: ferrule_cuda::context::CudaArtifactLinearHandle,
    up: ferrule_cuda::context::CudaArtifactLinearHandle,
    down: ferrule_cuda::context::CudaArtifactLinearHandle,
    bytes: u64,
}

#[cfg(feature = "cuda")]
struct CudaExpertUploadTicket {
    gate: ferrule_cuda::context::CudaArtifactLinearAsyncUpload,
    up: ferrule_cuda::context::CudaArtifactLinearAsyncUpload,
    down: ferrule_cuda::context::CudaArtifactLinearAsyncUpload,
    bytes: u64,
    event: ferrule_cuda::context::CudaUploadEvent,
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
    entries: HashMap<ExpertId, CudaPinnedExpertBundle>,
    order: VecDeque<ExpertId>,
    capacity: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

#[cfg(feature = "cuda")]
struct CudaMoeRoutes {
    routes: Vec<ExpertRoute>,
    selected: Vec<usize>,
}

#[cfg(feature = "cuda")]
struct CudaMoeMaterialization {
    streaming: ExpertStreamingStep,
    loaded_experts: BTreeSet<ExpertId>,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct PrefetchQueueOutcome {
    enqueued: usize,
    ready: BTreeSet<ExpertId>,
}

#[cfg(feature = "cuda")]
impl CudaExpertUploadTicket {
    fn is_complete(&self) -> Result<bool> {
        self.event.is_complete()
    }

    fn synchronize(&self) -> Result<()> {
        self.event.synchronize()
    }

    fn bytes(&self) -> u64 {
        self.bytes
    }

    fn into_handles(self) -> CudaFp4ExpertHandles {
        CudaFp4ExpertHandles {
            gate: self.gate.into_handle(),
            up: self.up.into_handle(),
            down: self.down.into_handle(),
            bytes: self.bytes,
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
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            capacity,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    fn get(&mut self, expert: ExpertId) -> Option<CudaPinnedExpertBundle> {
        if self.entries.contains_key(&expert) {
            self.hits = self.hits.saturating_add(1);
            self.order.retain(|id| *id != expert);
            self.order.push_back(expert);
            self.entries.get(&expert).cloned()
        } else {
            self.misses = self.misses.saturating_add(1);
            None
        }
    }

    fn insert(&mut self, bundle: CudaPinnedExpertBundle) {
        if self.capacity == 0 {
            return;
        }
        let expert = bundle.expert;
        if self.entries.contains_key(&expert) {
            self.order.retain(|id| *id != expert);
        } else if self.entries.len() >= self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.entries.remove(&evicted);
                self.evictions = self.evictions.saturating_add(1);
            }
        }
        self.entries.insert(expert, bundle);
        self.order.push_back(expert);
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn total_bytes(&self) -> u64 {
        self.entries.values().map(|bundle| bundle.bytes).sum()
    }

    fn hits(&self) -> u64 {
        self.hits
    }

    fn misses(&self) -> u64 {
        self.misses
    }

    fn evictions(&self) -> u64 {
        self.evictions
    }
}

fn duration_us(d: Duration) -> u64 {
    d.as_micros().min(u128::from(u64::MAX)) as u64
}

#[cfg(feature = "cuda")]
impl DeepSeekV4CudaOperatorCache {
    pub(crate) fn new(policy: &DeepSeekV4ExecutionPolicy) -> Result<Self> {
        Ok(Self {
            ops: ferrule_cuda::context::CudaArtifactOperatorContext::new()?,
            managed_experts: policy.managed_experts(),
            expert_upload_inflight: policy.expert_upload_inflight(),
            device_router_topk: policy.device_router_topk(),
            moe_segment_batch: policy.moe_segment_batch(),
            linears: HashMap::new(),
            experts: HashMap::new(),
            recycled_experts: Vec::new(),
            uploading_experts: HashMap::new(),
            moe_access_events: Vec::new(),
            decode_arena: DeepSeekV4DecodeArena::default(),
            host_staged_cache: HostStagedExpertCache::new(256),
            pinned_host_expert_cache: CudaPinnedExpertCache::new(
                policy.pinned_expert_cache_capacity(),
            ),
            async_host_stager: AsyncHostStagedExpertLoader::default(),
            norm_weights: HashMap::new(),
            compressor_ape_buffers: HashMap::new(),
            hc_weights: HashMap::new(),
            hc_head_weights: None,
            sink_buffers: HashMap::new(),
            router_bias_buffers: HashMap::new(),
            grouped_wo_a_weights: HashMap::new(),
            rope_tables: HashMap::new(),
            topk_buffer: None,
            output_head_logits: HashMap::new(),
            output_head_indices: None,
            output_head_values: None,
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
            expert_planner_residency_syncs: 0,
            expert_planner_residency_synced: 0,
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

    pub(crate) fn take_decode_buffers(
        &mut self,
        hc_len: usize,
        hidden_len: usize,
    ) -> Result<DeepSeekV4DecodeBuffers> {
        if self
            .decode_arena
            .hc_input
            .as_ref()
            .is_none_or(|buffer| buffer.len() != hc_len)
        {
            self.decode_arena.hc_input = Some(self.ops.zero_f32_buffer(hc_len)?);
        }
        if self
            .decode_arena
            .final_hidden
            .as_ref()
            .is_none_or(|buffer| buffer.len() != hidden_len)
        {
            self.decode_arena.final_hidden = Some(self.ops.zero_f32_buffer(hidden_len)?);
        }
        if self
            .decode_arena
            .final_norm
            .as_ref()
            .is_none_or(|buffer| buffer.len() != hidden_len)
        {
            self.decode_arena.final_norm = Some(self.ops.zero_f32_buffer(hidden_len)?);
        }
        Ok(DeepSeekV4DecodeBuffers {
            hc_input: self
                .decode_arena
                .hc_input
                .take()
                .expect("decode HC input initialized above"),
            final_hidden: self
                .decode_arena
                .final_hidden
                .take()
                .expect("decode final hidden initialized above"),
            final_norm: self
                .decode_arena
                .final_norm
                .take()
                .expect("decode final norm initialized above"),
        })
    }

    pub(crate) fn restore_decode_buffers(&mut self, buffers: DeepSeekV4DecodeBuffers) {
        debug_assert!(self.decode_arena.hc_input.is_none());
        debug_assert!(self.decode_arena.final_hidden.is_none());
        debug_assert!(self.decode_arena.final_norm.is_none());
        self.decode_arena.hc_input = Some(buffers.hc_input);
        self.decode_arena.final_hidden = Some(buffers.final_hidden);
        self.decode_arena.final_norm = Some(buffers.final_norm);
    }

    pub(crate) fn clone_sequence_kv_state(
        &mut self,
        source: &DeepSeekV4CudaSequenceKvState,
    ) -> Result<DeepSeekV4CudaSequenceKvState> {
        fn clone_buffer(
            ops: &mut ferrule_cuda::context::CudaArtifactOperatorContext,
            source: &Option<ferrule_cuda::context::CudaF32Buffer>,
        ) -> Result<Option<ferrule_cuda::context::CudaF32Buffer>> {
            let Some(source) = source else {
                return Ok(None);
            };
            let mut destination = ops.zero_f32_buffer(source.len())?;
            ops.copy_f32_into_slot(source, &mut destination, 0)?;
            Ok(Some(destination))
        }

        Ok(DeepSeekV4CudaSequenceKvState {
            kv_cache: clone_buffer(&mut self.ops, &source.kv_cache)?,
            kv_len: source.kv_len,
            combined_kv_cache: clone_buffer(&mut self.ops, &source.combined_kv_cache)?,
            combined_kv_compressed_capacity: source.combined_kv_compressed_capacity,
            indexer_kv_cache: clone_buffer(&mut self.ops, &source.indexer_kv_cache)?,
            indexer_kv_capacity: source.indexer_kv_capacity,
        })
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

    pub(crate) fn clear_expert_residency(&mut self) -> Result<()> {
        for (_, ticket) in self.uploading_experts.drain() {
            ticket.synchronize()?;
        }
        self.experts.clear();
        self.recycled_experts.clear();
        Ok(())
    }

    pub(crate) fn shutdown(&mut self) -> Result<()> {
        self.clear_expert_residency()?;
        self.ops.sync_upload_stream()?;
        self.ops.sync_stream()?;
        self.decode_arena = DeepSeekV4DecodeArena::default();
        self.topk_buffer = None;
        self.linears.clear();
        self.norm_weights.clear();
        self.compressor_ape_buffers.clear();
        self.hc_weights.clear();
        self.hc_head_weights = None;
        self.sink_buffers.clear();
        self.router_bias_buffers.clear();
        self.grouped_wo_a_weights.clear();
        self.rope_tables.clear();
        Ok(())
    }

    fn sync_planner_residency_for_layer(
        &mut self,
        layer: usize,
        planner: &mut ExpertStreamingPlanner,
    ) -> Result<usize> {
        self.expert_planner_residency_syncs = self.expert_planner_residency_syncs.saturating_add(1);
        let synced = planner.sync_gpu_residents(layer, self.resident_experts_for_layer(layer))?;
        self.expert_planner_residency_synced = self
            .expert_planner_residency_synced
            .saturating_add(synced as u64);
        Ok(synced)
    }

    fn install_resident_handle(
        handles: &mut CpuExpertHandleStore,
        expert: ExpertId,
        bytes: u64,
    ) -> Result<()> {
        handles.insert_resident_handle(ResidentExpertHandle::new(
            expert,
            ExpertStorageTier::Gpu,
            ExpertResidentFormat::Opaque("cuda-fp4-artifact".into()),
            bytes,
        ))
    }

    fn drain_async_host_staging(&mut self) -> usize {
        self.async_host_stager
            .drain_into(&mut self.host_staged_cache)
    }

    fn max_upload_prefetch_in_flight(&self) -> usize {
        self.expert_upload_inflight
    }

    fn poll_completed_uploads(&mut self) -> Result<BTreeSet<ExpertId>> {
        let mut completed = Vec::new();
        for (expert, ticket) in &self.uploading_experts {
            if ticket.is_complete()? {
                completed.push(*expert);
            }
        }

        let mut ready = BTreeSet::new();
        for expert in completed {
            if let Some(ticket) = self.uploading_experts.remove(&expert) {
                self.install_uploaded_expert(expert, ticket)?;
                self.expert_upload_prefetch_completed =
                    self.expert_upload_prefetch_completed.saturating_add(1);
                ready.insert(expert);
            }
        }
        Ok(ready)
    }

    fn install_uploaded_expert(
        &mut self,
        expert: ExpertId,
        ticket: CudaExpertUploadTicket,
    ) -> Result<u64> {
        let bytes = ticket.bytes();
        if self.ops.failpoints().check_expert_upload() {
            return Err(Error::Internal(format!(
                "deterministic failpoint: expert upload install layer {} expert {}",
                expert.layer, expert.expert
            )));
        }
        self.experts.insert(expert, ticket.into_handles());
        self.expert_loads = self.expert_loads.saturating_add(1);
        self.expert_load_bytes = self.expert_load_bytes.saturating_add(bytes);
        Ok(bytes)
    }

    fn wait_upload_ticket(&mut self, expert: ExpertId) -> Result<CudaExpertUploadTicket> {
        let wait_start = Instant::now();
        let complete = self
            .uploading_experts
            .get(&expert)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "selected CUDA expert upload ticket missing: layer {} expert {}",
                    expert.layer, expert.expert
                ))
            })?
            .is_complete()?;
        if !complete {
            self.expert_selected_upload_waits = self.expert_selected_upload_waits.saturating_add(1);
            self.uploading_experts
                .get(&expert)
                .expect("checked above")
                .synchronize()?;
        }
        self.expert_selected_upload_wait_us = self
            .expert_selected_upload_wait_us
            .saturating_add(duration_us(wait_start.elapsed()));
        Ok(self
            .uploading_experts
            .remove(&expert)
            .expect("checked above"))
    }

    fn wait_selected_upload(&mut self, expert: ExpertId) -> Result<u64> {
        let ticket = self.wait_upload_ticket(expert)?;
        self.expert_upload_prefetch_completed =
            self.expert_upload_prefetch_completed.saturating_add(1);
        self.install_uploaded_expert(expert, ticket)
    }

    fn apply_cuda_expert_evictions(&mut self, evictions: &[ExpertEvictRequest]) -> usize {
        if evictions.is_empty() {
            return 0;
        }
        let mut removed = 0usize;
        for eviction in evictions {
            if let Some(handles) = self.experts.remove(&eviction.expert) {
                self.recycled_experts.push(handles);
                removed = removed.saturating_add(1);
            }
        }
        removed
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

    fn submit_prefetch_upload_from_bundle(
        &mut self,
        expert: ExpertId,
        bundle: &ExpertComputeBundle,
    ) -> Result<bool> {
        if self.experts.contains_key(&expert) || self.uploading_experts.contains_key(&expert) {
            return Ok(false);
        }
        if self.uploading_experts.len() >= self.max_upload_prefetch_in_flight() {
            return Ok(false);
        }
        let pinned = self.pinned_bundle_for_upload(bundle)?;
        let ticket = self.upload_pinned_expert_bundle_async(&pinned)?;
        self.expert_async_upload_bytes = self
            .expert_async_upload_bytes
            .saturating_add(ticket.bytes());
        self.uploading_experts.insert(expert, ticket);
        self.expert_upload_prefetch_submitted =
            self.expert_upload_prefetch_submitted.saturating_add(1);
        Ok(true)
    }

    fn wait_for_selected_host_staging(
        &mut self,
        selected: &ExpertResidencySelectedLoad,
    ) -> Result<Option<ExpertComputeBundle>> {
        self.expert_selected_host_staging_waits =
            self.expert_selected_host_staging_waits.saturating_add(1);
        let wait_start = Instant::now();
        let staged = self
            .async_host_stager
            .wait_for_into(selected.expert, &mut self.host_staged_cache)?;
        self.expert_selected_host_staging_wait_us = self
            .expert_selected_host_staging_wait_us
            .saturating_add(duration_us(wait_start.elapsed()));
        if staged {
            if let Some(bundle) = self.host_staged_cache.get(selected.expert) {
                self.expert_selected_host_staging_hits =
                    self.expert_selected_host_staging_hits.saturating_add(1);
                return Ok(Some(bundle));
            }
        }
        Ok(None)
    }

    fn record_prefetch_residency_plan(&mut self, plan: &ExpertResidencyPlan) {
        self.moe_prefetch_loads = self
            .moe_prefetch_loads
            .saturating_add(plan.prefetch_load_count() as u64);
        self.moe_prefetch_skipped_cached_or_inflight = self
            .moe_prefetch_skipped_cached_or_inflight
            .saturating_add(plan.prefetch_skipped_cached_or_inflight_count() as u64);
        self.moe_prefetch_resident = self
            .moe_prefetch_resident
            .saturating_add(plan.prefetch_resident.len() as u64);
        self.moe_prefetch_materializing = self
            .moe_prefetch_materializing
            .saturating_add(plan.prefetch_materializing.len() as u64);
        self.moe_prefetch_host_staged = self
            .moe_prefetch_host_staged
            .saturating_add(plan.prefetch_host_staged.len() as u64);
        self.moe_prefetch_in_flight = self
            .moe_prefetch_in_flight
            .saturating_add(plan.prefetch_in_flight.len() as u64);
        self.moe_prefetch_cold = self
            .moe_prefetch_cold
            .saturating_add(plan.prefetch_cold.len() as u64);
    }

    fn queue_prefetch_loads_only(
        &mut self,
        loads: &[ExpertLoadRequest],
        reader: &ExpertStreamingReader,
    ) -> Result<PrefetchQueueOutcome> {
        let mut outcome = PrefetchQueueOutcome::default();
        self.drain_async_host_staging();
        outcome.ready.extend(self.poll_completed_uploads()?);
        if loads.is_empty() {
            return Ok(outcome);
        }

        let plan = classify_expert_residency(
            loads,
            |expert| self.experts.contains_key(&expert),
            |expert| self.uploading_experts.contains_key(&expert),
            |expert| self.host_staged_cache.contains(expert),
            |expert| self.async_host_stager.is_in_flight(expert),
        );
        self.record_prefetch_residency_plan(&plan);
        outcome.ready.extend(plan.prefetch_resident.iter().copied());

        for expert_id in &plan.prefetch_host_staged {
            let Some(bundle) = self.host_staged_cache.get(*expert_id) else {
                self.moe_prefetch_skipped_cached_or_inflight = self
                    .moe_prefetch_skipped_cached_or_inflight
                    .saturating_add(1);
                continue;
            };
            if self.submit_prefetch_upload_from_bundle(*expert_id, &bundle)? {
                self.moe_prefetch_enqueued = self.moe_prefetch_enqueued.saturating_add(1);
                outcome.enqueued = outcome.enqueued.saturating_add(1);
            } else {
                self.moe_prefetch_skipped_cached_or_inflight = self
                    .moe_prefetch_skipped_cached_or_inflight
                    .saturating_add(1);
            }
        }

        for prefetch in &plan.prefetch_cold {
            if self
                .async_host_stager
                .enqueue(prefetch.expert, prefetch.load_source.clone(), reader)
            {
                self.moe_prefetch_enqueued = self.moe_prefetch_enqueued.saturating_add(1);
                outcome.enqueued = outcome.enqueued.saturating_add(1);
            } else {
                self.moe_prefetch_skipped_cached_or_inflight = self
                    .moe_prefetch_skipped_cached_or_inflight
                    .saturating_add(1);
            }
        }
        Ok(outcome)
    }

    fn materialize_selected_bundle_sync(
        &mut self,
        expert: ExpertId,
        bundle: &ExpertComputeBundle,
    ) -> Result<u64> {
        let handles = self.upload_expert_bundle(bundle)?;
        let bytes = handles.bytes;
        if self.ops.failpoints().check_expert_upload() {
            return Err(Error::Internal(format!(
                "deterministic failpoint: expert sync upload layer {} expert {}",
                expert.layer, expert.expert
            )));
        }
        self.experts.insert(expert, handles);
        self.expert_loads = self.expert_loads.saturating_add(1);
        self.expert_load_bytes = self.expert_load_bytes.saturating_add(bytes);
        Ok(bytes)
    }

    #[allow(clippy::too_many_arguments)]
    fn materialize_selected_and_queue_prefetch(
        &mut self,
        layer: usize,
        loads: &[ExpertLoadRequest],
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
    ) -> Result<BTreeSet<ExpertId>> {
        let mut loaded = BTreeSet::<ExpertId>::new();
        if loads.is_empty() {
            return Ok(loaded);
        }

        let stage_start = Instant::now();
        self.drain_async_host_staging();
        loaded.extend(self.poll_completed_uploads()?);
        let plan = classify_expert_residency(
            loads,
            |expert| self.experts.contains_key(&expert),
            |expert| self.uploading_experts.contains_key(&expert),
            |expert| self.host_staged_cache.contains(expert),
            |expert| self.async_host_stager.is_in_flight(expert),
        );

        self.expert_selected_load_requests = self.expert_selected_load_requests.saturating_add(
            (plan.selected_resident.len()
                + plan.selected_materializing.len()
                + plan.selected_host_staged.len()
                + plan.selected_in_flight.len()
                + plan.selected_cold.len()) as u64,
        );
        self.expert_selected_resident_hits = self
            .expert_selected_resident_hits
            .saturating_add(plan.selected_resident_count() as u64);
        self.expert_selected_upload_hits = self
            .expert_selected_upload_hits
            .saturating_add(plan.selected_materializing.len() as u64);
        self.expert_selected_host_staged_hits = self
            .expert_selected_host_staged_hits
            .saturating_add(plan.selected_host_staged.len() as u64);
        self.expert_selected_cold_misses = self
            .expert_selected_cold_misses
            .saturating_add(plan.selected_cold.len() as u64);
        self.record_prefetch_residency_plan(&plan);
        loaded.extend(plan.prefetch_resident.iter().copied());

        for expert_id in &plan.selected_resident {
            let bytes = self
                .experts
                .get(expert_id)
                .map(|expert| expert.bytes)
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "residency plan marked missing CUDA expert resident: layer {} expert {}",
                        expert_id.layer, expert_id.expert
                    ))
                })?;
            Self::install_resident_handle(handles, *expert_id, bytes)?;
            loaded.insert(*expert_id);
        }

        for expert_id in &plan.selected_materializing {
            let bytes = self.wait_selected_upload(*expert_id)?;
            Self::install_resident_handle(handles, *expert_id, bytes)?;
            loaded.insert(*expert_id);
        }

        let mut selected_bundles = Vec::<(ExpertId, Option<ExpertComputeBundle>)>::new();
        let mut selected_miss_loads = Vec::new();
        for selected in plan.selected_waiting_for_host_staging() {
            if let Some(bundle) = self.wait_for_selected_host_staging(selected)? {
                selected_bundles.push((selected.expert, Some(bundle)));
            } else {
                selected_bundles.push((selected.expert, None));
                selected_miss_loads.push((selected.expert, selected.load_source.clone()));
            }
        }
        for selected in plan.selected_to_materialize() {
            if selected.host_staged {
                let bundle = self.host_staged_cache.get(selected.expert).ok_or_else(|| {
                    Error::Internal(format!(
                        "residency plan marked missing host-staged expert: layer {} expert {}",
                        selected.expert.layer, selected.expert.expert
                    ))
                })?;
                selected_bundles.push((selected.expert, Some(bundle)));
            } else {
                selected_bundles.push((selected.expert, None));
                selected_miss_loads.push((selected.expert, selected.load_source.clone()));
            }
        }

        for expert_id in &plan.prefetch_host_staged {
            let Some(bundle) = self.host_staged_cache.get(*expert_id) else {
                self.moe_prefetch_skipped_cached_or_inflight = self
                    .moe_prefetch_skipped_cached_or_inflight
                    .saturating_add(1);
                continue;
            };
            if self.submit_prefetch_upload_from_bundle(*expert_id, &bundle)? {
                self.moe_prefetch_enqueued = self.moe_prefetch_enqueued.saturating_add(1);
            } else {
                self.moe_prefetch_skipped_cached_or_inflight = self
                    .moe_prefetch_skipped_cached_or_inflight
                    .saturating_add(1);
            }
        }

        for prefetch in &plan.prefetch_cold {
            if self
                .async_host_stager
                .enqueue(prefetch.expert, prefetch.load_source.clone(), reader)
            {
                self.moe_prefetch_enqueued = self.moe_prefetch_enqueued.saturating_add(1);
            } else {
                self.moe_prefetch_skipped_cached_or_inflight = self
                    .moe_prefetch_skipped_cached_or_inflight
                    .saturating_add(1);
            }
        }

        self.moe_cache_lookup_us = self
            .moe_cache_lookup_us
            .saturating_add(duration_us(stage_start.elapsed()));

        let stage_start = Instant::now();
        let read_payloads = if selected_miss_loads.is_empty() {
            Vec::new()
        } else {
            read_experts_concurrent(reader, &selected_miss_loads)?
        };
        self.moe_expert_read_us = self
            .moe_expert_read_us
            .saturating_add(duration_us(stage_start.elapsed()));

        let stage_start = Instant::now();
        let mut payload_iter = read_payloads.into_iter();
        for (expert_id, cached) in selected_bundles {
            let bundle = if let Some(bundle) = cached {
                bundle
            } else {
                let payload = payload_iter.next().ok_or_else(|| {
                    Error::Internal(format!(
                        "concurrent expert read returned fewer payloads than expected for layer {layer}"
                    ))
                })?;
                let bundle = ExpertComputeBundle::from_artifact_payload(payload)?;
                self.host_staged_cache.insert(bundle.clone());
                bundle
            };
            let bytes = self.materialize_selected_bundle_sync(expert_id, &bundle)?;
            Self::install_resident_handle(handles, expert_id, bytes)?;
            loaded.insert(expert_id);
        }
        self.moe_expert_upload_us = self
            .moe_expert_upload_us
            .saturating_add(duration_us(stage_start.elapsed()));
        Ok(loaded)
    }

    pub(crate) fn prefetch_predicted_experts(
        &mut self,
        layer: usize,
        predicted_experts: &[usize],
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
    ) -> Result<usize> {
        if predicted_experts.is_empty() {
            return Ok(0);
        }
        let prefetch_start = Instant::now();
        self.expert_lookahead_prefetch_calls =
            self.expert_lookahead_prefetch_calls.saturating_add(1);
        self.expert_lookahead_prefetch_experts = self
            .expert_lookahead_prefetch_experts
            .saturating_add(predicted_experts.len() as u64);

        self.sync_planner_residency_for_layer(layer, planner)?;
        let streaming = planner.plan_layer_step(layer, &[], predicted_experts)?;
        self.expert_evictions = self
            .expert_evictions
            .saturating_add(streaming.evictions.len() as u64);
        handles.apply_evictions(&streaming.evictions);
        self.apply_cuda_expert_evictions(&streaming.evictions);
        let outcome = self.queue_prefetch_loads_only(&streaming.loads, reader)?;
        planner.commit_step_loaded(&streaming, outcome.ready.iter().copied())?;

        self.expert_lookahead_prefetch_enqueued = self
            .expert_lookahead_prefetch_enqueued
            .saturating_add(outcome.enqueued as u64);
        self.expert_lookahead_prefetch_us = self
            .expert_lookahead_prefetch_us
            .saturating_add(duration_us(prefetch_start.elapsed()));
        Ok(outcome.enqueued)
    }

    pub(crate) fn prewarm_experts(
        &mut self,
        layer: usize,
        experts: &[usize],
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
    ) -> Result<usize> {
        if experts.is_empty() {
            return Ok(0);
        }
        let mut selected = experts.to_vec();
        selected.sort_unstable();
        selected.dedup();
        self.sync_planner_residency_for_layer(layer, planner)?;
        let streaming = planner.plan_layer_step(layer, &selected, &[])?;
        self.expert_evictions = self
            .expert_evictions
            .saturating_add(streaming.evictions.len() as u64);
        handles.apply_evictions(&streaming.evictions);
        self.apply_cuda_expert_evictions(&streaming.evictions);

        let mut warmed = 0usize;
        if !streaming.loads.is_empty() {
            let mut cached_bundles = Vec::with_capacity(streaming.loads.len());
            let mut miss_loads = Vec::new();
            for load in &streaming.loads {
                if let Some(expert) = self.experts.get(&load.expert) {
                    handles.insert_resident_handle(ResidentExpertHandle::new(
                        load.expert,
                        ExpertStorageTier::Gpu,
                        ExpertResidentFormat::Opaque("cuda-fp4-artifact".into()),
                        expert.bytes,
                    ))?;
                    continue;
                }
                if let Some(bundle) = self.host_staged_cache.get(load.expert) {
                    cached_bundles.push((load.expert, Some(bundle)));
                } else {
                    cached_bundles.push((load.expert, None));
                    miss_loads.push((load.expert, load.load_source.clone()));
                }
            }

            let read_payloads = if miss_loads.is_empty() {
                Vec::new()
            } else {
                read_experts_concurrent(reader, &miss_loads)?
            };
            let mut payload_iter = read_payloads.into_iter();
            for (expert_id, cached) in &cached_bundles {
                let bundle = if let Some(bundle) = cached {
                    bundle.clone()
                } else {
                    let payload = payload_iter.next().ok_or_else(|| {
                        Error::Internal(format!(
                            "expert prewarm read returned fewer payloads than expected for layer {layer}"
                        ))
                    })?;
                    let bundle = ExpertComputeBundle::from_artifact_payload(payload)?;
                    self.host_staged_cache.insert(bundle.clone());
                    bundle
                };
                let expert = self.upload_expert_bundle(&bundle)?;
                let bytes = expert.bytes;
                if self.ops.failpoints().check_resource_install() {
                    return Err(Error::Internal(format!(
                        "deterministic failpoint: expert prewarm install layer {} expert {}",
                        expert_id.layer, expert_id.expert
                    )));
                }
                self.experts.insert(*expert_id, expert);
                handles.insert_resident_handle(ResidentExpertHandle::new(
                    *expert_id,
                    ExpertStorageTier::Gpu,
                    ExpertResidentFormat::Opaque("cuda-fp4-artifact".into()),
                    bytes,
                ))?;
                warmed = warmed.saturating_add(1);
            }
        }
        planner.commit_step(&streaming)?;
        Ok(warmed)
    }

    pub(crate) fn runtime_counters(&self) -> DeepSeekV4OperatorRuntimeCounters {
        let cuda = self.ops.counters();
        let async_prefetch = self.async_host_stager.stats();
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
            moe_pointer_upload_us: cuda.moe_pointer_upload_us,
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
            expert_planner_residency_syncs: self.expert_planner_residency_syncs,
            expert_planner_residency_synced: self.expert_planner_residency_synced,
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
            expert_host_cache_hits: self.host_staged_cache.hits(),
            expert_host_cache_misses: self.host_staged_cache.misses(),
            expert_host_cache_evictions: self.host_staged_cache.evictions(),
            expert_host_cache_entries: self.host_staged_cache.len(),
            expert_host_cache_bytes: self.host_staged_cache.total_bytes(),
            expert_pinned_cache_hits: self.pinned_host_expert_cache.hits(),
            expert_pinned_cache_misses: self.pinned_host_expert_cache.misses(),
            expert_pinned_cache_evictions: self.pinned_host_expert_cache.evictions(),
            expert_pinned_cache_entries: self.pinned_host_expert_cache.len(),
            expert_pinned_cache_bytes: self.pinned_host_expert_cache.total_bytes(),
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

    /// Batched device-to-device RMS norm with cached affine weight.
    pub(crate) fn rms_norm_rows_device_cached(
        &mut self,
        name: &str,
        input: &ferrule_cuda::context::CudaF32Buffer,
        rows: usize,
        weight: &[f32],
        eps: f32,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        if rows == 0 || input.len() != rows * weight.len() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 CUDA RMS device rows length mismatch: rows={rows} input={} weight={}",
                input.len(),
                weight.len()
            )));
        }
        self.ensure_norm_uploaded(name, weight)?;
        let weight_buf = self.norm_weights.get(name).expect("inserted above");
        self.ops
            .rms_norm_rows_from_device(input, rows, weight_buf, eps)
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

    fn dsv4_router_topk_routes_from_device_logits(
        &mut self,
        layer: usize,
        logits: &ferrule_cuda::context::CudaF32Buffer,
        tokens: usize,
        router: &RouterArtifactPayload,
        router_policy: &ExpertRouterPolicy,
        indices_dev: &mut ferrule_cuda::context::CudaF32Buffer,
        weights_dev: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<Option<Vec<Vec<ExpertRoute>>>> {
        if !self.device_router_topk {
            return Ok(None);
        }
        if router_policy.selection != RouterSelectionPolicy::ScoreTopK
            || router_policy.score_function != RouterScoreFunction::SqrtSoftplus
            || !router_policy.normalize_non_softmax_weights
        {
            return Ok(None);
        }
        let experts = router.weight.format.out_features();
        let top_k = router_policy.top_k;
        if top_k == 0 || top_k > 64 || experts > 512 {
            return Ok(None);
        }
        let bias_key =
            self.ensure_router_bias_buffer_key(layer, router.bias.as_deref(), experts)?;
        let bias = bias_key
            .as_ref()
            .and_then(|key| self.router_bias_buffers.get(key));
        let (indices, weights) = self
            .ops
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
        Ok(Some(routes_by_token))
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
        if chunk_rows == 0 {
            return Err(Error::Model(
                "DeepSeek-V4 CUDA output-head chunk_rows must be > 0".into(),
            ));
        }
        if slice.shape.len() != 2 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 CUDA output head expects 2D slice, got {:?}",
                slice.shape
            )));
        }
        let hidden_cols = slice.shape[1];
        if hidden.len() != hidden_cols {
            return Err(Error::Model(format!(
                "DeepSeek-V4 CUDA output head input mismatch: expected {hidden_cols}, got {}",
                hidden.len()
            )));
        }

        let upload_start = Instant::now();
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
        self.output_head_hidden_upload_us = self
            .output_head_hidden_upload_us
            .saturating_add(duration_us(upload_start.elapsed()));
        let result = self.output_head_topk_chunks_with_device(
            slice,
            &hidden_device,
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
        let vocab_rows =
            slice.shape.first().copied().ok_or_else(|| {
                Error::Model("DeepSeek-V4 CUDA output head expects 2D slice".into())
            })?;
        self.output_head_calls = self.output_head_calls.saturating_add(1);
        let mut top = Vec::<TokenLogit>::new();
        let mut start = 0usize;
        while start < vocab_rows {
            let rows = chunk_rows.min(vocab_rows - start);
            self.output_head_chunks = self.output_head_chunks.saturating_add(1);
            self.output_head_rows = self.output_head_rows.saturating_add(rows as u64);
            let key = artifact_linear_row_cache_key(slice, start, rows)?;
            if !self.linears.contains_key(&key) {
                self.output_head_cache_misses = self.output_head_cache_misses.saturating_add(1);
                let read_start = Instant::now();
                let payload = reader.read_2d_rows(slice, start, rows)?;
                let linear = ArtifactLinearPayload::from_weight_and_scale(
                    TensorRole::OutputHead,
                    payload,
                    None,
                )?;
                self.output_head_read_us = self
                    .output_head_read_us
                    .saturating_add(duration_us(read_start.elapsed()));
                let actual_key = artifact_linear_cache_key(&linear);
                if actual_key != key {
                    return Err(Error::Model(format!(
                        "DeepSeek-V4 output-head cache-key mismatch: predicted {key}, materialized {actual_key}"
                    )));
                }
                let upload_start = Instant::now();
                let handle = self.upload_linear(&linear)?;
                self.output_head_upload_us = self
                    .output_head_upload_us
                    .saturating_add(duration_us(upload_start.elapsed()));
                self.linears.insert(key.clone(), handle);
            } else {
                self.output_head_cache_hits = self.output_head_cache_hits.saturating_add(1);
            }
            let chunk_k = top_k.min(rows);
            if !self.output_head_logits.contains_key(&rows) {
                let logits = self.ops.zero_f32_buffer(rows)?;
                self.output_head_logits.insert(rows, logits);
            }
            if self
                .output_head_indices
                .as_ref()
                .is_none_or(|buffer| buffer.len() < top_k)
            {
                self.output_head_indices = Some(self.ops.zero_f32_buffer(top_k)?);
                self.output_head_values = Some(self.ops.zero_f32_buffer(top_k)?);
            }
            let handle = self.linears.get(&key).expect("inserted above");
            let logits = self
                .output_head_logits
                .get_mut(&rows)
                .expect("output-head logits workspace initialized above");
            let indices = self
                .output_head_indices
                .as_mut()
                .expect("output-head indices workspace initialized above");
            let values = self
                .output_head_values
                .as_mut()
                .expect("output-head values workspace initialized above");
            let topk_start = Instant::now();
            let chunk_top = self.ops.artifact_linear_topk_from_device_into(
                handle, hidden, chunk_k, logits, indices, values,
            )?;
            self.output_head_topk_us = self
                .output_head_topk_us
                .saturating_add(duration_us(topk_start.elapsed()));
            let merge_start = Instant::now();
            top.extend(chunk_top.into_iter().map(|(token_id, logit)| TokenLogit {
                token_id: token_id + start as u32,
                logit,
            }));
            top.sort_by(rank_logits_desc);
            top.truncate(top_k);
            self.output_head_merge_us = self
                .output_head_merge_us
                .saturating_add(duration_us(merge_start.elapsed()));
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
        router: &RouterArtifactPayload,
        predicted_experts: &[usize],
        router_policy: &ExpertRouterPolicy,
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
        shared_expert: &SwiGluFfnPayload,
        router_logits_dev: &mut ferrule_cuda::context::CudaF32Buffer,
        router_indices_dev: &mut ferrule_cuda::context::CudaF32Buffer,
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
        let hidden_size = router.weight.format.in_features();
        if input_dev.len() != tokens * hidden_size {
            return Err(Error::Model(format!(
                "CUDA routed MoE prefill device input length mismatch: input={} expected {} tokens x {} hidden",
                input_dev.len(), tokens, hidden_size
            )));
        }

        if output_dev.len() != input_dev.len() {
            return Err(Error::Model(format!(
                "CUDA routed MoE prefill output length mismatch: output={} expected {}",
                output_dev.len(),
                input_dev.len()
            )));
        }

        let stage_start = Instant::now();
        self.linear_rows_from_device_into(
            &router.weight,
            input_dev,
            tokens,
            router_logits_dev,
            linear_workspace,
        )?;
        let routes_by_token = if let Some(routes) = self
            .dsv4_router_topk_routes_from_device_logits(
                layer,
                router_logits_dev,
                tokens,
                router,
                router_policy,
                router_indices_dev,
                router_weights_dev,
            )? {
            routes
        } else {
            let router_logits = self.ops.download_f32_buffer(router_logits_dev)?;
            let logits_width = router.weight.format.out_features();
            let mut routes_by_token = Vec::<Vec<ExpertRoute>>::with_capacity(tokens);
            for (token_idx, &token_id) in token_ids.iter().enumerate() {
                let logits =
                    &router_logits[token_idx * logits_width..(token_idx + 1) * logits_width];
                let hash_experts = router.hash_experts_for_token(token_id)?;
                let routes =
                    router_policy.route(logits, router.bias.as_deref(), hash_experts.as_deref())?;
                routes_by_token.push(routes);
            }
            routes_by_token
        };
        for routes in &routes_by_token {
            self.expert_selected = self.expert_selected.saturating_add(routes.len() as u64);
        }
        self.moe_router_us = self
            .moe_router_us
            .saturating_add(duration_us(stage_start.elapsed()));

        let stage_start = Instant::now();
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
        self.moe_shared_us = self
            .moe_shared_us
            .saturating_add(duration_us(stage_start.elapsed()));

        let streaming_steps = self.routed_moe_prefill_segments_from_device(
            layer,
            input_dev,
            &routes_by_token,
            predicted_experts,
            planner,
            reader,
            handles,
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
        predicted_experts: &[usize],
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
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

        let mut routes_by_expert = BTreeMap::<usize, Vec<(i32, i32, f32)>>::new();
        for (token, routes) in routes_by_token.iter().enumerate() {
            if routes.len() != routes_per_token {
                return Err(Error::Internal(format!(
                    "CUDA segmented MoE route count mismatch at token {token}: got {} expected {routes_per_token}",
                    routes.len()
                )));
            }
            for (rank, route) in routes.iter().enumerate() {
                let route_index = token
                    .checked_mul(routes_per_token)
                    .and_then(|base| base.checked_add(rank))
                    .ok_or_else(|| {
                        Error::Internal("CUDA segmented MoE route index overflow".into())
                    })?;
                routes_by_expert.entry(route.expert).or_default().push((
                    token as i32,
                    route_index as i32,
                    route.weight,
                ));
            }
        }
        let unique_experts = routes_by_expert.keys().copied().collect::<Vec<_>>();
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

        let resident_slots = planner.policy().gpu_slots_per_layer.clamp(1, 64);
        let segment_capacity = self.moe_segment_batch;
        let compute_start = Instant::now();
        let mut route_seen = vec![false; route_count];
        let mut input_prepared = false;
        let mut expected_intermediate = None;
        let mut streaming_steps = Vec::new();

        for selected in unique_experts.chunks(resident_slots) {
            self.moe_predicted_experts = self
                .moe_predicted_experts
                .saturating_add(predicted_experts.len() as u64);
            self.sync_planner_residency_for_layer(layer, planner)?;
            let stage_start = Instant::now();
            let streaming = planner.plan_layer_step(layer, selected, predicted_experts)?;
            self.moe_plan_us = self
                .moe_plan_us
                .saturating_add(duration_us(stage_start.elapsed()));
            self.expert_evictions = self
                .expert_evictions
                .saturating_add(streaming.evictions.len() as u64);

            let stage_start = Instant::now();
            handles.apply_evictions(&streaming.evictions);
            self.apply_cuda_expert_evictions(&streaming.evictions);
            self.moe_cache_lookup_us = self
                .moe_cache_lookup_us
                .saturating_add(duration_us(stage_start.elapsed()));
            let loaded_experts = self.materialize_selected_and_queue_prefetch(
                layer,
                &streaming.loads,
                reader,
                handles,
            )?;

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
                        resident_slots,
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
                    resident_slots,
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
                input_prepared = true;
            }

            let mut segment_expert_slots = Vec::new();
            let mut segment_token_indices = Vec::new();
            let mut segment_route_indices = Vec::new();
            let mut segment_route_weights = Vec::new();
            for (slot, expert) in selected.iter().copied().enumerate() {
                let records = routes_by_expert.get(&expert).ok_or_else(|| {
                    Error::Internal(format!(
                        "CUDA segmented MoE missing route records for expert {expert}"
                    ))
                })?;
                for records in records.chunks(8) {
                    segment_expert_slots.push(slot as i32);
                    for column in 0..8 {
                        if let Some(&(token, route, weight)) = records.get(column) {
                            let route_index = route as usize;
                            if std::mem::replace(&mut route_seen[route_index], true) {
                                return Err(Error::Internal(format!(
                                    "CUDA segmented MoE duplicate route index {route_index}"
                                )));
                            }
                            segment_token_indices.push(token);
                            segment_route_indices.push(route);
                            segment_route_weights.push(weight);
                        } else {
                            segment_token_indices.push(-1);
                            segment_route_indices.push(-1);
                            segment_route_weights.push(0.0);
                        }
                    }
                }
            }

            {
                let gate_handles = selected
                    .iter()
                    .map(|&expert| {
                        self.experts
                            .get(&ExpertId::new(layer, expert))
                            .map(|handles| &handles.gate)
                            .ok_or_else(|| {
                                Error::Model(format!(
                                    "CUDA segmented MoE gate missing for layer {layer} expert {expert}"
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let up_handles = selected
                    .iter()
                    .map(|&expert| {
                        self.experts
                            .get(&ExpertId::new(layer, expert))
                            .map(|handles| &handles.up)
                            .ok_or_else(|| {
                                Error::Model(format!(
                                    "CUDA segmented MoE up missing for layer {layer} expert {expert}"
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let down_handles = selected
                    .iter()
                    .map(|&expert| {
                        self.experts
                            .get(&ExpertId::new(layer, expert))
                            .map(|handles| &handles.down)
                            .ok_or_else(|| {
                                Error::Model(format!(
                                    "CUDA segmented MoE down missing for layer {layer} expert {expert}"
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let workspace = segment_workspace
                    .as_mut()
                    .expect("segmented MoE workspace initialized above");
                for segment_start in (0..segment_expert_slots.len()).step_by(segment_capacity) {
                    let segment_end =
                        (segment_start + segment_capacity).min(segment_expert_slots.len());
                    let metadata_start = segment_start * 8;
                    let metadata_end = segment_end * 8;
                    self.ops.moe_expert_segment_batch_from_prepared(
                        &gate_handles,
                        &up_handles,
                        &down_handles,
                        &segment_expert_slots[segment_start..segment_end],
                        &segment_token_indices[metadata_start..metadata_end],
                        &segment_route_indices[metadata_start..metadata_end],
                        &segment_route_weights[metadata_start..metadata_end],
                        routes_per_token,
                        swiglu_limit,
                        workspace,
                        route_output,
                    )?;
                }
            }

            let stage_start = Instant::now();
            planner.commit_step_loaded(&streaming, loaded_experts.iter().copied())?;
            self.moe_commit_us = self
                .moe_commit_us
                .saturating_add(duration_us(stage_start.elapsed()));
            streaming_steps.push(streaming);
        }

        if route_seen.iter().any(|seen| !seen) {
            let missing = route_seen.iter().filter(|seen| !**seen).count();
            return Err(Error::Internal(format!(
                "CUDA segmented MoE failed to materialize {missing} of {route_count} routes"
            )));
        }
        self.ops.reduce_moe_route_outputs_ranked(
            route_output,
            tokens,
            routes_per_token,
            hidden_size,
            output_dev,
        )?;
        self.moe_compute_submit_us = self
            .moe_compute_submit_us
            .saturating_add(duration_us(compute_start.elapsed()));
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
        indices_dev: &mut ferrule_cuda::context::CudaF32Buffer,
        weights_dev: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<CudaMoeRoutes> {
        let stage_start = Instant::now();
        self.ops.copy_f32_into_slot(input, router_input, 0)?;
        self.linear_matvec_from_device_into(&router.weight, router_input, logits_dev)?;
        let routes = if let Some(mut routes_by_token) = self
            .dsv4_router_topk_routes_from_device_logits(
                layer,
                logits_dev,
                1,
                router,
                router_policy,
                indices_dev,
                weights_dev,
            )? {
            routes_by_token.pop().unwrap_or_default()
        } else {
            let logits = self.ops.download_f32_buffer(logits_dev)?;
            let hash_experts = router.hash_experts_for_token(token_id)?;
            router_policy.route(&logits, router.bias.as_deref(), hash_experts.as_deref())?
        };
        self.moe_router_us = self
            .moe_router_us
            .saturating_add(duration_us(stage_start.elapsed()));

        let stage_start = Instant::now();
        let selected = routes.iter().map(|route| route.expert).collect::<Vec<_>>();
        self.moe_routing_us = self
            .moe_routing_us
            .saturating_add(duration_us(stage_start.elapsed()));
        Ok(CudaMoeRoutes { routes, selected })
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_and_materialize_decode_moe(
        &mut self,
        layer: usize,
        selected: &[usize],
        route_count: usize,
        predicted_experts: &[usize],
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
    ) -> Result<CudaMoeMaterialization> {
        self.moe_predicted_experts = self
            .moe_predicted_experts
            .saturating_add(predicted_experts.len() as u64);
        self.sync_planner_residency_for_layer(layer, planner)?;
        let stage_start = Instant::now();
        let streaming = planner.plan_layer_step(layer, selected, predicted_experts)?;
        self.moe_plan_us = self
            .moe_plan_us
            .saturating_add(duration_us(stage_start.elapsed()));
        self.expert_selected = self.expert_selected.saturating_add(route_count as u64);
        self.expert_evictions = self
            .expert_evictions
            .saturating_add(streaming.evictions.len() as u64);

        let stage_start = Instant::now();
        handles.apply_evictions(&streaming.evictions);
        self.apply_cuda_expert_evictions(&streaming.evictions);
        self.moe_cache_lookup_us = self
            .moe_cache_lookup_us
            .saturating_add(duration_us(stage_start.elapsed()));

        let loaded_experts =
            self.materialize_selected_and_queue_prefetch(layer, &streaming.loads, reader, handles)?;
        Ok(CudaMoeMaterialization {
            streaming,
            loaded_experts,
        })
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
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
        shared_expert: &SwiGluFfnPayload,
        router_input: &mut ferrule_cuda::context::CudaF32Buffer,
        router_logits: &mut ferrule_cuda::context::CudaF32Buffer,
        router_indices: &mut ferrule_cuda::context::CudaF32Buffer,
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
        let CudaMoeMaterialization {
            streaming,
            loaded_experts,
        } = self.plan_and_materialize_decode_moe(
            layer,
            &selected,
            routes.len(),
            predicted_experts,
            planner,
            reader,
            handles,
        )?;

        let stage_start = Instant::now();
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
        self.moe_shared_us = self
            .moe_shared_us
            .saturating_add(duration_us(stage_start.elapsed()));

        if !routes.is_empty() {
            let stage_start = Instant::now();
            let num_experts = routes.len();
            // Gather all expert handles and route weights.
            let mut gate_handles: Vec<&ferrule_cuda::context::CudaArtifactLinearHandle> =
                Vec::with_capacity(num_experts);
            let mut up_handles: Vec<&ferrule_cuda::context::CudaArtifactLinearHandle> =
                Vec::with_capacity(num_experts);
            let mut down_handles: Vec<&ferrule_cuda::context::CudaArtifactLinearHandle> =
                Vec::with_capacity(num_experts);
            let mut route_weights_arr = [0.0f32; 6];
            for (i, route) in routes.iter().enumerate() {
                let expert_id = ExpertId::new(layer, route.expert);
                let expert = self.experts.get(&expert_id).ok_or_else(|| {
                    Error::Model(format!(
                        "CUDA expert handle missing for layer {} expert {}",
                        expert_id.layer, expert_id.expert
                    ))
                })?;
                gate_handles.push(&expert.gate);
                up_handles.push(&expert.up);
                down_handles.push(&expert.down);
                if i < 6 {
                    route_weights_arr[i] = route.weight;
                }
            }

            // Get shapes from first expert. The workspace MoE path prepares the
            // activation internally (direct FP4 packing for TC, FP8-in-f32 for
            // scalar fallback), so there is no per-call prepared input buffer.
            let first_expert_id = ExpertId::new(layer, routes[0].expert);
            let first_expert = self.experts.get(&first_expert_id).ok_or_else(|| {
                Error::Model(format!(
                    "CUDA expert handle missing for layer {} expert {}",
                    first_expert_id.layer, first_expert_id.expert
                ))
            })?;

            let intermediate_size = first_expert.gate.shape().out_features();
            let hidden_size = first_expert.down.shape().out_features();

            // Pad handle arrays to fixed [6] for the batched kernel.
            let mut gate_arr: [&ferrule_cuda::context::CudaArtifactLinearHandle; 6] = [
                gate_handles[0],
                gate_handles[0],
                gate_handles[0],
                gate_handles[0],
                gate_handles[0],
                gate_handles[0],
            ];
            let mut up_arr: [&ferrule_cuda::context::CudaArtifactLinearHandle; 6] = [
                up_handles[0],
                up_handles[0],
                up_handles[0],
                up_handles[0],
                up_handles[0],
                up_handles[0],
            ];
            let mut down_arr: [&ferrule_cuda::context::CudaArtifactLinearHandle; 6] = [
                down_handles[0],
                down_handles[0],
                down_handles[0],
                down_handles[0],
                down_handles[0],
                down_handles[0],
            ];
            for i in 0..num_experts.min(6) {
                gate_arr[i] = gate_handles[i];
                up_arr[i] = up_handles[i];
                down_arr[i] = down_handles[i];
            }

            let workspace_needs_init = match &self.decode_arena.moe_workspace {
                Some(workspace) => {
                    !workspace.matches(6, input.len(), intermediate_size, hidden_size)
                }
                None => true,
            };
            if workspace_needs_init {
                self.decode_arena.moe_workspace = Some(self.ops.moe_batched_workspace(
                    6,
                    input.len(),
                    intermediate_size,
                    hidden_size,
                )?);
            }
            let ops = &self.ops;
            let workspace = self
                .decode_arena
                .moe_workspace
                .as_mut()
                .expect("MoE workspace initialized above");
            self.moe_workspace_us = self
                .moe_workspace_us
                .saturating_add(duration_us(stage_start.elapsed()));
            let stage_start = Instant::now();
            ops.moe_experts_batched_add_into_from_device(
                &gate_arr,
                &up_arr,
                &down_arr,
                &route_weights_arr,
                input,
                swiglu_limit,
                num_experts.min(6),
                intermediate_size,
                hidden_size,
                workspace,
                accumulator,
            )?;
            self.moe_compute_submit_us = self
                .moe_compute_submit_us
                .saturating_add(duration_us(stage_start.elapsed()));
        }

        let stage_start = Instant::now();
        planner.commit_step_loaded(&streaming, loaded_experts.iter().copied())?;
        self.moe_commit_us = self
            .moe_commit_us
            .saturating_add(duration_us(stage_start.elapsed()));

        Ok(RoutedMoeStepOutput {
            routes,
            streaming,
            routed_output: Vec::new(),
            shared_output: None,
            output: Vec::new(),
        })
    }

    fn upload_expert_bundle(
        &mut self,
        bundle: &ExpertComputeBundle,
    ) -> Result<CudaFp4ExpertHandles> {
        if let Some(mut handles) = self.recycled_experts.pop() {
            self.overwrite_expert_linear(&mut handles.gate, &bundle.gate)?;
            self.overwrite_expert_linear(&mut handles.up, &bundle.up)?;
            self.overwrite_expert_linear(&mut handles.down, &bundle.down)?;
            handles.bytes = bundle.total_bytes();
            return Ok(handles);
        }
        Ok(CudaFp4ExpertHandles {
            gate: self.upload_expert_linear(&bundle.gate)?,
            up: self.upload_expert_linear(&bundle.up)?,
            down: self.upload_expert_linear(&bundle.down)?,
            bytes: bundle.total_bytes(),
        })
    }

    fn overwrite_expert_linear(
        &self,
        handle: &mut ferrule_cuda::context::CudaArtifactLinearHandle,
        linear: &ExpertLinearPayload,
    ) -> Result<()> {
        let ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale { block_size: 32, .. } = linear.format
        else {
            return Err(Error::Model(format!(
                "CUDA routed expert {:?} requires artifact FP4 block_size=32, got {:?}",
                linear.matrix, linear.format
            )));
        };
        let scale = linear.scale.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "CUDA routed expert {:?} is missing E8M0 scale payload",
                linear.matrix
            ))
        })?;
        self.ops
            .overwrite_artifact_linear(handle, &linear.weight.bytes, &scale.bytes)
    }

    fn upload_pinned_expert_bundle_async(
        &self,
        bundle: &CudaPinnedExpertBundle,
    ) -> Result<CudaExpertUploadTicket> {
        let gate = self.upload_pinned_expert_linear_async(&bundle.gate)?;
        let up = self.upload_pinned_expert_linear_async(&bundle.up)?;
        let down = self.upload_pinned_expert_linear_async(&bundle.down)?;
        let event = self.ops.record_upload_event()?;
        Ok(CudaExpertUploadTicket {
            gate,
            up,
            down,
            bytes: bundle.bytes,
            event,
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

    pub(crate) fn upload_expert_linear(
        &self,
        linear: &ExpertLinearPayload,
    ) -> Result<ferrule_cuda::context::CudaArtifactLinearHandle> {
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
        let scale = linear.scale.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "CUDA routed expert {:?} is missing E8M0 scale payload",
                linear.matrix
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
    pub(crate) fn sparse_attention_with_combined_kv_topk_into(
        &mut self,
        state: &DeepSeekV4CudaSequenceKvState,
        query: &ferrule_cuda::context::CudaF32Buffer,
        layer: usize,
        topk: &ferrule_cuda::context::CudaI32Buffer,
        sink: &[f32],
        tokens: usize,
        kv_len: usize,
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
        let values = state.combined_kv_cache.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing sequence-owned combined KV device cache"
            ))
        })?;
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_topk_indices_from_device_into(
        &mut self,
        query: Option<&ferrule_cuda::context::CudaF32Buffer>,
        weights: Option<&ferrule_cuda::context::CudaF32Buffer>,
        indexer_kv: Option<&ferrule_cuda::context::CudaF32Buffer>,
        empty_query: &ferrule_cuda::context::CudaF32Buffer,
        empty_weights: &ferrule_cuda::context::CudaF32Buffer,
        empty_kv: &ferrule_cuda::context::CudaF32Buffer,
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
        self.ops.dsv4_prefill_topk_indices_from_device_into(
            query,
            weights,
            indexer_kv,
            empty_query,
            empty_weights,
            empty_kv,
            tokens,
            window_size,
            window_cols,
            extra_cols,
            value_offset,
            compress_ratio,
            compressed_len,
            index_heads,
            index_head_dim,
            weight_scale,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_topk_indices_fused_index_query_from_device_into(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        weights: &ferrule_cuda::context::CudaF32Buffer,
        indexer_kv: &ferrule_cuda::context::CudaF32Buffer,
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
        let cos = self.rope_cos_device(rope_name);
        let sin = self.rope_sin_device(rope_name);
        self.ops
            .dsv4_prefill_topk_indices_fused_index_query_from_device_into(
                query,
                weights,
                indexer_kv,
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
                weight_scale,
                output,
            )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_topk_indices_from_device_into(
        &mut self,
        query: Option<&ferrule_cuda::context::CudaF32Buffer>,
        weights: Option<&ferrule_cuda::context::CudaF32Buffer>,
        indexer_kv: Option<&ferrule_cuda::context::CudaF32Buffer>,
        empty_query: &ferrule_cuda::context::CudaF32Buffer,
        empty_weights: &ferrule_cuda::context::CudaF32Buffer,
        empty_kv: &ferrule_cuda::context::CudaF32Buffer,
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
        self.ops.dsv4_decode_topk_indices_from_device_into(
            query,
            weights,
            indexer_kv,
            empty_query,
            empty_weights,
            empty_kv,
            position,
            window_len,
            window_size,
            extra_cols,
            value_offset,
            compressed_len,
            index_heads,
            index_head_dim,
            weight_scale,
            out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_topk_indices_fused_index_query_from_indexer_cache_into(
        &mut self,
        state: &DeepSeekV4CudaSequenceKvState,
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
        let indexer_kv = state.indexer_kv_cache.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing sequence-owned indexer KV device cache"
            ))
        })?;
        let cos = self.rope_cos_device(rope_name);
        let sin = self.rope_sin_device(rope_name);
        self.ops
            .dsv4_decode_topk_indices_fused_index_query_from_device_into(
                query,
                weights,
                indexer_kv,
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
                weight_scale,
                out,
            )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_topk_indices_from_indexer_cache_into(
        &mut self,
        state: &DeepSeekV4CudaSequenceKvState,
        query: &ferrule_cuda::context::CudaF32Buffer,
        weights: &ferrule_cuda::context::CudaF32Buffer,
        empty_query: &ferrule_cuda::context::CudaF32Buffer,
        empty_weights: &ferrule_cuda::context::CudaF32Buffer,
        empty_kv: &ferrule_cuda::context::CudaF32Buffer,
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
        let indexer_kv = state.indexer_kv_cache.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing sequence-owned indexer KV device cache"
            ))
        })?;
        self.decode_topk_indices_from_device_into(
            Some(query),
            Some(weights),
            Some(indexer_kv),
            empty_query,
            empty_weights,
            empty_kv,
            position,
            window_len,
            window_size,
            extra_cols,
            value_offset,
            compressed_len,
            index_heads,
            index_head_dim,
            weight_scale,
            out,
        )
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

    // ── Device-resident window KV cache ───────────────────────────────

    /// Ensure a `[window_size * head_dim]` f32 device buffer exists for
    /// `layer`, zero-initialized. Idempotent.
    #[allow(dead_code)]
    pub(crate) fn ensure_kv_cache(
        &mut self,
        state: &mut DeepSeekV4CudaSequenceKvState,
        _layer: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        if state.kv_cache.is_none() {
            state.kv_cache = Some(self.ops.zero_f32_buffer(window_size * head_dim)?);
            state.kv_len = 0;
        }
        Ok(())
    }

    /// Append a single-token KV vector (already on device) into the slot for
    /// `position` using a device-to-device copy, then advance the cached
    /// length. `kv_buffer` must be `[head_dim]` f32.
    #[allow(dead_code)]
    pub(crate) fn kv_append_device(
        &mut self,
        state: &mut DeepSeekV4CudaSequenceKvState,
        layer: usize,
        kv_buffer: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        head_dim: usize,
        window_size: usize,
    ) -> Result<()> {
        self.ensure_kv_cache(state, layer, window_size, head_dim)?;
        if kv_buffer.len() != head_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} device KV append length mismatch: expected {head_dim}, got {}",
                kv_buffer.len()
            )));
        }
        let slot = position % window_size;
        let offset = slot * head_dim;
        let dst = state
            .kv_cache
            .as_mut()
            .expect("inserted by ensure_kv_cache");
        self.ops.copy_f32_into_slot(kv_buffer, dst, offset)?;
        state.kv_len = window_size.min(state.kv_len + 1);
        Ok(())
    }

    pub(crate) fn kv_write_window_rows_device(
        &mut self,
        state: &mut DeepSeekV4CudaSequenceKvState,
        layer: usize,
        values: &ferrule_cuda::context::CudaF32Buffer,
        start_position: usize,
        rows: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        self.ensure_kv_cache(state, layer, window_size, head_dim)?;
        if rows == 0 || values.len() != rows * head_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} device KV rows mismatch: rows={rows} expected {}, got {}",
                rows * head_dim,
                values.len()
            )));
        }

        {
            let dst = state
                .kv_cache
                .as_mut()
                .expect("inserted by ensure_kv_cache");
            if start_position + rows <= window_size {
                self.ops
                    .copy_f32_into_slot(values, dst, start_position * head_dim)?;
            } else {
                for row in 0..rows {
                    let row_indices = [row as i32];
                    let row_indices_dev = self.ops.upload_i32_buffer(&row_indices)?;
                    let row_dev =
                        self.ops
                            .gather_f32_rows(values, &row_indices_dev, 1, head_dim)?;
                    let slot = (start_position + row) % window_size;
                    self.ops
                        .copy_f32_into_slot(&row_dev, dst, slot * head_dim)?;
                }
            }
        }

        state.kv_len = window_size.min(state.kv_len.max(start_position.saturating_add(rows)));
        Ok(())
    }

    /// Borrow the device-resident KV buffer for `layer`.
    #[allow(dead_code)]
    pub(crate) fn kv_values_device<'a>(
        &self,
        state: &'a DeepSeekV4CudaSequenceKvState,
    ) -> &'a ferrule_cuda::context::CudaF32Buffer {
        state
            .kv_cache
            .as_ref()
            .expect("kv cache must be ensured before reading")
    }

    pub(crate) fn ensure_combined_kv_cache(
        &mut self,
        state: &mut DeepSeekV4CudaSequenceKvState,
        layer: usize,
        window_values: &[f32],
        head_dim: usize,
        compressed_values: &[f32],
        compressed_capacity: usize,
    ) -> Result<()> {
        if state.combined_kv_cache.is_some()
            && state.combined_kv_compressed_capacity >= compressed_capacity
        {
            return Ok(());
        }

        let compressed_len = compressed_values.len().checked_div(head_dim).unwrap_or(0);
        let requested_capacity = compressed_capacity.max(compressed_len).max(16);
        let capacity = requested_capacity
            .checked_next_power_of_two()
            .ok_or_else(|| {
                Error::Model(format!(
                    "DeepSeek-V4 layer {layer} combined KV capacity overflow: requested={requested_capacity}"
                ))
            })?;
        let compressed_elements = capacity.checked_mul(head_dim).ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} combined KV element count overflow: capacity={capacity} head_dim={head_dim}"
            ))
        })?;
        let buffer_len = window_values
            .len()
            .checked_add(compressed_elements)
            .ok_or_else(|| {
                Error::Model(format!(
                    "DeepSeek-V4 layer {layer} combined KV buffer length overflow"
                ))
            })?;
        if u64::try_from(buffer_len).unwrap_or(u64::MAX) > u64::from(u32::MAX) + 1 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} combined KV buffer exceeds CUDA u32 element indexing: {buffer_len}"
            )));
        }

        // Keep the installed device buffer and capacity metadata untouched until
        // replacement allocation and D2D preservation both succeed. The host
        // window is intentionally only a metadata shadow on CUDA paths, so it
        // cannot reconstruct the old device values after a failed growth.
        let buffer = if let Some(old) = state.combined_kv_cache.as_ref() {
            let mut replacement = self.ops.zero_f32_buffer(buffer_len)?;
            self.ops.copy_f32_into_slot(old, &mut replacement, 0)?;
            replacement
        } else {
            let mut values = Vec::new();
            values.try_reserve_exact(buffer_len).map_err(|error| {
                Error::Model(format!(
                    "DeepSeek-V4 layer {layer} combined KV host allocation failed for {buffer_len} elements: {error}"
                ))
            })?;
            values.resize(buffer_len, 0.0f32);
            values[..window_values.len()].copy_from_slice(window_values);
            let compressed_offset = window_values.len();
            values[compressed_offset..compressed_offset + compressed_values.len()]
                .copy_from_slice(compressed_values);
            self.ops.upload_f32_buffer(&values)?
        };
        state.combined_kv_cache = Some(buffer);
        state.combined_kv_compressed_capacity = capacity;
        Ok(())
    }

    pub(crate) fn combined_kv_append_window_device(
        &mut self,
        state: &mut DeepSeekV4CudaSequenceKvState,
        layer: usize,
        kv_dev: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        if kv_dev.len() != head_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} combined window KV device append mismatch: expected {head_dim}, got {}",
                kv_dev.len()
            )));
        }
        let dst = state.combined_kv_cache.as_mut().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing sequence-owned combined KV device cache"
            ))
        })?;
        let slot = position % window_size;
        self.ops.copy_f32_into_slot(kv_dev, dst, slot * head_dim)
    }

    pub(crate) fn combined_kv_write_window_rows_device(
        &mut self,
        state: &mut DeepSeekV4CudaSequenceKvState,
        layer: usize,
        values: &ferrule_cuda::context::CudaF32Buffer,
        start_position: usize,
        rows: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        if rows == 0 || values.len() != rows * head_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} combined window KV rows mismatch: rows={rows} expected {}, got {}",
                rows * head_dim,
                values.len()
            )));
        }
        let dst = state.combined_kv_cache.as_mut().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing sequence-owned combined KV device cache"
            ))
        })?;
        if start_position + rows <= window_size {
            return self
                .ops
                .copy_f32_into_slot(values, dst, start_position * head_dim);
        }
        for row in 0..rows {
            let row_indices = [row as i32];
            let row_indices_dev = self.ops.upload_i32_buffer(&row_indices)?;
            let row_dev = self
                .ops
                .gather_f32_rows(values, &row_indices_dev, 1, head_dim)?;
            let slot = (start_position + row) % window_size;
            self.ops
                .copy_f32_into_slot(&row_dev, dst, slot * head_dim)?;
        }
        Ok(())
    }

    pub(crate) fn combined_kv_write_compressed_rows_device(
        &mut self,
        state: &mut DeepSeekV4CudaSequenceKvState,
        layer: usize,
        values: &ferrule_cuda::context::CudaF32Buffer,
        compressed_start: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        if !values.len().is_multiple_of(head_dim) {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} combined compressed rows length {} is not divisible by head_dim {head_dim}",
                values.len()
            )));
        }
        let rows = values.len() / head_dim;
        if rows == 0 {
            return Ok(());
        }
        let capacity = state.combined_kv_compressed_capacity;
        if compressed_start + rows > capacity {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} compressed KV rows [{}..{}) exceed device capacity {capacity}",
                compressed_start,
                compressed_start + rows
            )));
        }
        let dst = state.combined_kv_cache.as_mut().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing sequence-owned combined KV device cache"
            ))
        })?;
        let offset = (window_size + compressed_start) * head_dim;
        self.ops.copy_f32_into_slot(values, dst, offset)
    }

    pub(crate) fn combined_kv_append_compressed_host(
        &mut self,
        state: &mut DeepSeekV4CudaSequenceKvState,
        layer: usize,
        value: &[f32],
        compressed_index: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        if value.len() != head_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} combined compressed KV append mismatch: expected {head_dim}, got {}",
                value.len()
            )));
        }
        let capacity = state.combined_kv_compressed_capacity;
        if compressed_index >= capacity {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} compressed KV index {compressed_index} exceeds device capacity {capacity}"
            )));
        }
        let src = self.ops.upload_f32_buffer(value)?;
        let dst = state.combined_kv_cache.as_mut().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing sequence-owned combined KV device cache"
            ))
        })?;
        let offset = (window_size + compressed_index) * head_dim;
        self.ops.copy_f32_into_slot(&src, dst, offset)
    }

    pub(crate) fn ensure_indexer_kv_cache(
        &mut self,
        state: &mut DeepSeekV4CudaSequenceKvState,
        layer: usize,
        compressed_capacity: usize,
        index_head_dim: usize,
    ) -> Result<()> {
        if index_head_dim == 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} indexer KV cache requires non-zero head dim"
            )));
        }
        if compressed_capacity == 0 {
            return Ok(());
        }
        let current_capacity = state.indexer_kv_capacity;
        if state.indexer_kv_cache.is_some() && current_capacity >= compressed_capacity {
            return Ok(());
        }

        let requested_capacity = compressed_capacity.max(16);
        let capacity = requested_capacity
            .checked_next_power_of_two()
            .ok_or_else(|| {
                Error::Model(format!(
                    "DeepSeek-V4 layer {layer} indexer KV capacity overflow: requested={requested_capacity}"
                ))
            })?;
        let buffer_len = capacity.checked_mul(index_head_dim).ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} indexer KV element count overflow: capacity={capacity} index_head_dim={index_head_dim}"
            ))
        })?;
        if u64::try_from(buffer_len).unwrap_or(u64::MAX) > u64::from(u32::MAX) + 1 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} indexer KV buffer exceeds CUDA u32 element indexing: {buffer_len}"
            )));
        }

        // Do not remove the authoritative buffer until replacement allocation and
        // preservation have both been accepted by the CUDA stream. Any validation,
        // allocation, or copy-launch error leaves the installed buffer/capacity intact.
        let mut replacement = self.ops.zero_f32_buffer(buffer_len)?;
        if let Some(old) = state.indexer_kv_cache.as_ref() {
            self.ops.copy_f32_into_slot(old, &mut replacement, 0)?;
        }
        state.indexer_kv_cache = Some(replacement);
        state.indexer_kv_capacity = capacity;
        Ok(())
    }

    pub(crate) fn indexer_kv_write_rows_device(
        &mut self,
        state: &mut DeepSeekV4CudaSequenceKvState,
        layer: usize,
        values: &ferrule_cuda::context::CudaF32Buffer,
        compressed_start: usize,
        index_head_dim: usize,
    ) -> Result<()> {
        if index_head_dim == 0 || !values.len().is_multiple_of(index_head_dim) {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} indexer KV rows length {} is not divisible by index_head_dim {index_head_dim}",
                values.len()
            )));
        }
        let rows = values.len() / index_head_dim;
        if rows == 0 {
            return Ok(());
        }
        let capacity = state.indexer_kv_capacity;
        if compressed_start + rows > capacity {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} indexer KV rows [{}..{}) exceed device capacity {capacity}",
                compressed_start,
                compressed_start + rows
            )));
        }
        let dst = state.indexer_kv_cache.as_mut().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing sequence-owned indexer KV device cache"
            ))
        })?;
        self.ops
            .copy_f32_into_slot(values, dst, compressed_start * index_head_dim)
    }

    pub(crate) fn indexer_kv_append_host(
        &mut self,
        state: &mut DeepSeekV4CudaSequenceKvState,
        layer: usize,
        value: &[f32],
        compressed_index: usize,
        index_head_dim: usize,
    ) -> Result<()> {
        if value.len() != index_head_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} indexer KV append mismatch: expected {index_head_dim}, got {}",
                value.len()
            )));
        }
        let capacity = state.indexer_kv_capacity;
        if compressed_index >= capacity {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} indexer KV index {compressed_index} exceeds device capacity {capacity}"
            )));
        }
        let src = self.ops.upload_f32_buffer(value)?;
        let dst = state.indexer_kv_cache.as_mut().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing sequence-owned indexer KV device cache"
            ))
        })?;
        self.ops
            .copy_f32_into_slot(&src, dst, compressed_index * index_head_dim)
    }

    #[allow(dead_code)]
    pub(crate) fn combined_kv_values_device<'a>(
        &self,
        state: &'a DeepSeekV4CudaSequenceKvState,
        layer: usize,
    ) -> Result<&'a ferrule_cuda::context::CudaF32Buffer> {
        state.combined_kv_cache.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing sequence-owned combined KV device cache"
            ))
        })
    }

    // ── Precomputed rope cos/sin tables ──────────────────────────────

    /// Ensure a typed `[capacity, rope_dim/2]` RoPE table can address every
    /// position below `required_positions`. Tables retain the historical 4096
    /// initial allocation, then grow geometrically beyond it. Rebuilding is
    /// failure-atomic: the old table remains installed until both replacement
    /// buffers have been generated and uploaded successfully.
    #[allow(dead_code)]
    pub(crate) fn ensure_rope_tables(
        &mut self,
        name: &str,
        rope_dim: usize,
        rope_theta: f32,
        required_positions: usize,
    ) -> Result<()> {
        self.ensure_rope_tables_with_params(
            name,
            rope_dim,
            DeepSeekV4RopeParams::plain(rope_theta),
            required_positions,
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

    /// Ensure a `[window_size]` i32 device buffer is allocated for top-k
    /// indices. Idempotent.
    #[allow(dead_code)]
    pub(crate) fn ensure_topk_buffer(&mut self, window_size: usize) -> Result<()> {
        if self.topk_buffer.is_none() {
            self.topk_buffer = Some(self.ops.zero_i32_buffer(window_size)?);
        }
        Ok(())
    }

    /// Fill the cached top-k buffer for `position` and the current KV length,
    /// mirroring `DeepSeekV4WindowKvCache::topk_indices`. Slots beyond the
    /// valid range are set to `-1`.
    #[allow(dead_code)]
    pub(crate) fn fill_topk_buffer(
        &mut self,
        state: &DeepSeekV4CudaSequenceKvState,
        position: usize,
        window_size: usize,
    ) -> Result<&ferrule_cuda::context::CudaI32Buffer> {
        self.ensure_topk_buffer(window_size)?;
        let kv_len = state.kv_len;
        let mut indices = vec![-1i32; window_size];
        if kv_len != 0 {
            if kv_len < window_size {
                let take = kv_len.min(window_size);
                for slot in 0..take {
                    indices[slot] = slot as i32;
                }
            } else {
                let slot = position % window_size;
                let mut write = 0usize;
                for idx in slot + 1..window_size {
                    if write < window_size {
                        indices[write] = idx as i32;
                        write += 1;
                    }
                }
                for idx in 0..=slot {
                    if write < window_size {
                        indices[write] = idx as i32;
                        write += 1;
                    }
                }
            }
        }
        let buf = self
            .topk_buffer
            .as_mut()
            .expect("inserted by ensure_topk_buffer");
        self.ops.copy_i32_into_buffer(&indices, buf)?;
        Ok(self.topk_buffer.as_ref().expect("filled above"))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sparse_attention_topk_from_device_into(
        &mut self,
        state: &DeepSeekV4CudaSequenceKvState,
        query: &ferrule_cuda::context::CudaF32Buffer,
        layer: usize,
        position: usize,
        window_size: usize,
        sink: &[f32],
        shape: ferrule_cuda::transformer::sparse_attention::CudaSparseAttentionShape,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.fill_topk_buffer(state, position, window_size)?;
        let sink_name = format!("sink_L{layer}");
        self.ensure_sink_buffer(&sink_name, sink)?;
        let topk = self.topk_buffer.as_ref().expect("filled above");
        let kv_values = state
            .kv_cache
            .as_ref()
            .ok_or_else(|| Error::Model("DeepSeek-V4 CUDA KV cache is not initialized".into()))?;
        let sink_buf = self.sink_buffers.get(&sink_name).expect("inserted above");
        self.ops.sparse_attention_sink_from_device_into(
            query,
            kv_values,
            topk.as_device_buffer(),
            sink_buf,
            shape,
            output,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::attention::DeepSeekV4AttentionCache;
    use super::super::helpers::apply_rotary_tail_scaled;
    use super::*;

    #[test]
    fn sequence_kv_reset_retains_capacity_metadata() {
        let mut state = DeepSeekV4CudaSequenceKvState {
            kv_len: 7,
            combined_kv_compressed_capacity: 32,
            indexer_kv_capacity: 64,
            ..DeepSeekV4CudaSequenceKvState::default()
        };

        state.reset_for_reuse();

        assert_eq!(state.kv_len, 0);
        assert_eq!(state.combined_kv_compressed_capacity, 32);
        assert_eq!(state.indexer_kv_capacity, 64);
    }

    #[test]
    #[ignore = "requires cargo-oxide and CUDA"]
    fn rope_table_grows_across_4096_boundary() -> Result<()> {
        let mut operators =
            DeepSeekV4CudaOperatorCache::new(&DeepSeekV4ExecutionPolicy::default())?;
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

    #[test]
    #[ignore = "requires cargo-oxide and CUDA"]
    fn combined_kv_growth_preserves_device_values() -> Result<()> {
        let cfg = DeepSeekV4AttentionConfig {
            hidden_size: 2,
            num_heads: 1,
            head_dim: 2,
            q_lora_rank: 2,
            rope_head_dim: 2,
            o_groups: 1,
            o_lora_rank: 2,
            window_size: 2,
            compress_ratio: 4,
            norm_eps: 1.0e-5,
            rope_theta: 10_000.0,
            compress_rope_theta: 10_000.0,
            original_seq_len: 2,
            rope_factor: 1.0,
            beta_fast: 1,
            beta_slow: 1,
            index_n_heads: 1,
            index_head_dim: 2,
            index_topk: 1,
        };
        let cache = DeepSeekV4AttentionCache::new(cfg);
        let mut state = DeepSeekV4CudaSequenceKvState::default();
        let mut operators =
            DeepSeekV4CudaOperatorCache::new(&DeepSeekV4ExecutionPolicy::default())?;

        operators.ensure_combined_kv_cache(
            &mut state,
            0,
            cache.window.values_full(),
            cache.window.head_dim,
            &cache.compressed,
            16,
        )?;
        let window = operators.ops.upload_f32_buffer(&[1.0, 2.0, 3.0, 4.0])?;
        operators.combined_kv_write_window_rows_device(&mut state, 0, &window, 0, 2, 2, 2)?;
        let last_compressed = operators.ops.upload_f32_buffer(&[9.0, 10.0])?;
        operators.combined_kv_write_compressed_rows_device(
            &mut state,
            0,
            &last_compressed,
            15,
            2,
            2,
        )?;

        operators.ensure_combined_kv_cache(
            &mut state,
            0,
            cache.window.values_full(),
            cache.window.head_dim,
            &cache.compressed,
            17,
        )?;
        let values = operators
            .ops
            .download_f32_buffer(operators.combined_kv_values_device(&state, 0)?)?;
        assert_eq!(&values[..4], &[1.0, 2.0, 3.0, 4.0]);
        let last_compressed_offset = (2 + 15) * 2;
        assert_eq!(
            &values[last_compressed_offset..last_compressed_offset + 2],
            &[9.0, 10.0]
        );

        Ok(())
    }

    #[test]
    #[ignore = "requires cargo-oxide and CUDA"]
    fn indexer_kv_growth_preserves_device_values() -> Result<()> {
        let mut state = DeepSeekV4CudaSequenceKvState::default();
        let mut operators =
            DeepSeekV4CudaOperatorCache::new(&DeepSeekV4ExecutionPolicy::default())?;
        operators.ensure_indexer_kv_cache(&mut state, 0, 16, 2)?;
        let values = operators.ops.upload_f32_buffer(&[1.0, 2.0, 9.0, 10.0])?;
        operators.indexer_kv_write_rows_device(&mut state, 0, &values, 0, 2)?;

        operators.ensure_indexer_kv_cache(&mut state, 0, 17, 2)?;
        assert_eq!(state.indexer_kv_capacity, 32);
        let actual = operators.ops.download_f32_buffer(
            state
                .indexer_kv_cache
                .as_ref()
                .expect("indexer cache grown above"),
        )?;
        assert_eq!(&actual[..4], &[1.0, 2.0, 9.0, 10.0]);
        Ok(())
    }

    #[test]
    #[ignore = "requires cargo-oxide and CUDA"]
    fn indexer_kv_growth_failure_keeps_installed_device_values() -> Result<()> {
        let mut state = DeepSeekV4CudaSequenceKvState::default();
        let mut operators =
            DeepSeekV4CudaOperatorCache::new(&DeepSeekV4ExecutionPolicy::default())?;
        operators.ensure_indexer_kv_cache(&mut state, 0, 16, 2)?;
        let values = operators.ops.upload_f32_buffer(&[1.0, 2.0, 9.0, 10.0])?;
        operators.indexer_kv_write_rows_device(&mut state, 0, &values, 0, 2)?;

        // Inject an incompatible growth request: the replacement has only 16 f32
        // slots while the installed buffer has 32. The D2D copy validation fails
        // before launch and must not remove the authoritative old buffer.
        state.indexer_kv_capacity = 8;
        let error = operators
            .ensure_indexer_kv_cache(&mut state, 0, 9, 1)
            .expect_err("injected undersized replacement must fail");
        assert!(error.to_string().contains("copy out of bounds"));
        assert_eq!(state.indexer_kv_capacity, 8);
        let actual = operators.ops.download_f32_buffer(
            state
                .indexer_kv_cache
                .as_ref()
                .expect("failed growth must retain installed indexer cache"),
        )?;
        assert_eq!(actual.len(), 32);
        assert_eq!(&actual[..4], &[1.0, 2.0, 9.0, 10.0]);
        Ok(())
    }
}
