//! DeepSeek-V4 CUDA operator cache: device-resident weights, KV cache, MoE handles.

#![cfg(feature = "cuda")]

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::time::{Duration, Instant};

use ferrule_common::{Error, Result};

use crate::artifact::binding::RouterArtifactPayload;
use crate::artifact::linear::{
    ArtifactActivationQuantization, ArtifactLinearFormat, ArtifactLinearPayload,
};
use crate::artifact::tensor::{ArtifactTensorReader, ArtifactTensorSlice};
use crate::attention_backend::SparseAttentionSpec;
use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::{
    HyperConnectionConfig, HyperConnectionHeadWeights, HyperConnectionPreOutput,
    HyperConnectionSplit, HyperConnectionWeights,
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
use crate::TensorRole;

use super::artifact::{artifact_linear_cache_key, artifact_linear_row_cache_key};
use super::attention::DeepSeekV4AttentionCache;
use super::config::{DeepSeekV4AttentionConfig, DeepSeekV4RopeParams};
use super::helpers::{check_linear, rank_logits_desc, sparse_topk_i32, yarn_frequency};
use super::operators::{
    CudaRoutedMoeStepOutput, DeepSeekV4Logit, DeepSeekV4OperatorRuntimeCounters,
};

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4CudaOperatorCache {
    pub(crate) ops: ferrule_cuda::context::CudaArtifactOperatorContext,
    linears: HashMap<String, ferrule_cuda::context::CudaArtifactLinearHandle>,
    experts: HashMap<ExpertId, CudaFp4ExpertHandles>,
    uploading_experts: HashMap<ExpertId, CudaExpertUploadTicket>,
    moe_access_events: Vec<ExpertBatchAccessEvent>,
    decode_arena: DeepSeekV4DecodeArena,
    host_staged_cache: HostStagedExpertCache,
    pinned_host_expert_cache: CudaPinnedExpertCache,
    async_host_stager: AsyncHostStagedExpertLoader,
    norm_weights: HashMap<String, ferrule_cuda::context::CudaF32Buffer>,
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
    /// Device-resident window KV cache per layer: `[window_size * head_dim]` f32.
    kv_cache: HashMap<usize, ferrule_cuda::context::CudaF32Buffer>,
    /// Current KV length per layer (capped at `window_size`).
    kv_len: HashMap<usize, usize>,
    /// Device-resident compressed attention values per layer:
    /// `[window_size * head_dim | compressed_capacity * head_dim]`.
    combined_kv_cache: HashMap<usize, ferrule_cuda::context::CudaF32Buffer>,
    /// Compressed slots allocated in `combined_kv_cache` per layer.
    combined_kv_compressed_capacity: HashMap<usize, usize>,
    /// Device-resident indexer compressed KV per layer:
    /// `[indexer_compressed_capacity * index_head_dim]`.
    indexer_kv_cache: HashMap<usize, ferrule_cuda::context::CudaF32Buffer>,
    /// Compressed slots allocated in `indexer_kv_cache` per layer.
    indexer_kv_capacity: HashMap<usize, usize>,
    /// Precomputed rope cos tables per layer tag: `[max_positions, rope_dim/2]`.
    rope_cos: HashMap<String, ferrule_cuda::context::CudaF32Buffer>,
    /// Precomputed rope sin tables per layer tag: `[max_positions, rope_dim/2]`.
    rope_sin: HashMap<String, ferrule_cuda::context::CudaF32Buffer>,
    /// Pre-allocated top-k index buffer `[window_size]` i32 for device-resident
    /// sparse attention.
    topk_buffer: Option<ferrule_cuda::context::CudaI32Buffer>,
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
    expert_selected_async_upload_submitted: u64,
    expert_selected_async_upload_completed: u64,
    expert_selected_async_upload_bytes: u64,
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
    /// Reusable routed MoE scratch/pointer buffers for the eager decode hot path.
    moe_workspace: Option<ferrule_cuda::context::CudaMoeBatchedWorkspace>,
    /// Per-layer routed MoE workspaces for CUDA graph capture/replay. Each layer
    /// needs stable expert pointer/route-weight device arrays; a single shared
    /// workspace would be overwritten by later layers before capture replay.
    moe_graph_workspaces: HashMap<usize, ferrule_cuda::context::CudaMoeBatchedWorkspace>,
    /// Reusable routed MoE scratch/pointer buffers for prefill grouped columns.
    moe_prefill_workspace: Option<ferrule_cuda::context::CudaMoeBatchedWorkspace>,
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
    pending_prefetch_uploads: Vec<ExpertId>,
    pending_selected_uploads: Vec<ExpertId>,
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

fn pinned_expert_cache_capacity() -> usize {
    std::env::var("FERRULE_DSV4_PINNED_EXPERT_CACHE_CAPACITY")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(64)
}

fn duration_us(d: Duration) -> u64 {
    d.as_micros().min(u128::from(u64::MAX)) as u64
}

#[cfg(feature = "cuda")]
impl DeepSeekV4CudaOperatorCache {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            ops: ferrule_cuda::context::CudaArtifactOperatorContext::new()?,
            linears: HashMap::new(),
            experts: HashMap::new(),
            uploading_experts: HashMap::new(),
            moe_access_events: Vec::new(),
            decode_arena: DeepSeekV4DecodeArena::default(),
            host_staged_cache: HostStagedExpertCache::new(256),
            pinned_host_expert_cache: CudaPinnedExpertCache::new(pinned_expert_cache_capacity()),
            async_host_stager: AsyncHostStagedExpertLoader::default(),
            norm_weights: HashMap::new(),
            hc_weights: HashMap::new(),
            hc_head_weights: None,
            sink_buffers: HashMap::new(),
            router_bias_buffers: HashMap::new(),
            grouped_wo_a_weights: HashMap::new(),
            kv_cache: HashMap::new(),
            kv_len: HashMap::new(),
            combined_kv_cache: HashMap::new(),
            combined_kv_compressed_capacity: HashMap::new(),
            indexer_kv_cache: HashMap::new(),
            indexer_kv_capacity: HashMap::new(),
            rope_cos: HashMap::new(),
            rope_sin: HashMap::new(),
            topk_buffer: None,
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
            expert_selected_async_upload_submitted: 0,
            expert_selected_async_upload_completed: 0,
            expert_selected_async_upload_bytes: 0,
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

    pub(crate) fn clear_sequence_workspaces(&mut self) {
        // Sequence state must not survive a session/request reset. Keep persistent
        // device-resident weights and expert handles, but drop all KV/indexer
        // device caches and scratch arenas that can contain prompt-specific data.
        self.kv_cache.clear();
        self.kv_len.clear();
        self.combined_kv_cache.clear();
        self.combined_kv_compressed_capacity.clear();
        self.indexer_kv_cache.clear();
        self.indexer_kv_capacity.clear();
        self.decode_arena.moe_workspace = None;
        self.decode_arena.moe_graph_workspaces.clear();
        self.decode_arena.moe_prefill_workspace = None;
    }

    pub(crate) fn clear_expert_residency(&mut self) -> Result<()> {
        for (_, ticket) in self.uploading_experts.drain() {
            ticket.synchronize()?;
        }
        self.experts.clear();
        self.clear_sequence_workspaces();
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
        std::env::var("FERRULE_DSV4_EXPERT_UPLOAD_INFLIGHT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(32)
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

    fn wait_selected_overlap_upload(&mut self, expert: ExpertId) -> Result<u64> {
        let ticket = self.wait_upload_ticket(expert)?;
        self.expert_selected_async_upload_completed = self
            .expert_selected_async_upload_completed
            .saturating_add(1);
        self.install_uploaded_expert(expert, ticket)
    }

    fn apply_cuda_expert_evictions(&mut self, evictions: &[ExpertEvictRequest]) -> usize {
        if evictions.is_empty() {
            return 0;
        }
        let mut removed = 0usize;
        for eviction in evictions {
            if self.experts.remove(&eviction.expert).is_some() {
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

    fn submit_selected_upload_from_bundle(
        &mut self,
        expert: ExpertId,
        bundle: &ExpertComputeBundle,
    ) -> Result<bool> {
        if self.experts.contains_key(&expert) || self.uploading_experts.contains_key(&expert) {
            return Ok(false);
        }
        let pinned = self.pinned_bundle_for_upload(bundle)?;
        let ticket = self.upload_pinned_expert_bundle_async(&pinned)?;
        let bytes = ticket.bytes();
        self.expert_async_upload_bytes = self.expert_async_upload_bytes.saturating_add(bytes);
        self.expert_selected_async_upload_bytes = self
            .expert_selected_async_upload_bytes
            .saturating_add(bytes);
        self.uploading_experts.insert(expert, ticket);
        self.expert_selected_async_upload_submitted = self
            .expert_selected_async_upload_submitted
            .saturating_add(1);
        Ok(true)
    }

    fn selected_upload_overlap_enabled(&self) -> bool {
        std::env::var("FERRULE_DSV4_SELECTED_UPLOAD_OVERLAP")
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                !(value.is_empty() || value == "0" || value == "false" || value == "off")
            })
            .unwrap_or(false)
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

    #[allow(clippy::too_many_arguments)]
    fn begin_selected_materialization_and_queue_prefetch_overlap(
        &mut self,
        layer: usize,
        loads: &[ExpertLoadRequest],
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
    ) -> Result<(BTreeSet<ExpertId>, Vec<ExpertId>, Vec<ExpertId>)> {
        let mut loaded = BTreeSet::<ExpertId>::new();
        let mut pending_prefetch_uploads = Vec::<ExpertId>::new();
        let mut pending_selected_uploads = Vec::<ExpertId>::new();
        if loads.is_empty() {
            return Ok((loaded, pending_prefetch_uploads, pending_selected_uploads));
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
            pending_prefetch_uploads.push(*expert_id);
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
        let mut pending_selected = BTreeSet::<ExpertId>::new();
        for (expert_id, cached) in selected_bundles {
            if let Some(expert) = self.experts.get(&expert_id) {
                Self::install_resident_handle(handles, expert_id, expert.bytes)?;
                loaded.insert(expert_id);
                continue;
            }
            if self.uploading_experts.contains_key(&expert_id) {
                pending_prefetch_uploads.push(expert_id);
                continue;
            }

            if let Some(bundle) = cached {
                if self.submit_selected_upload_from_bundle(expert_id, &bundle)? {
                    pending_selected.insert(expert_id);
                } else if self.uploading_experts.contains_key(&expert_id) {
                    pending_prefetch_uploads.push(expert_id);
                } else if let Some(expert) = self.experts.get(&expert_id) {
                    Self::install_resident_handle(handles, expert_id, expert.bytes)?;
                    loaded.insert(expert_id);
                } else {
                    return Err(Error::Internal(format!(
                        "selected host-staged CUDA expert upload was not submitted and no resident handle exists: layer {} expert {}",
                        expert_id.layer, expert_id.expert
                    )));
                }
                continue;
            }

            let payload = payload_iter.next().ok_or_else(|| {
                Error::Internal(format!(
                    "concurrent expert read returned fewer payloads than expected for layer {layer}"
                ))
            })?;
            let bundle = ExpertComputeBundle::from_artifact_payload(payload)?;
            self.host_staged_cache.insert(bundle.clone());
            let bytes = self.materialize_selected_bundle_sync(expert_id, &bundle)?;
            Self::install_resident_handle(handles, expert_id, bytes)?;
            loaded.insert(expert_id);
        }
        pending_selected_uploads.extend(pending_selected);
        self.moe_expert_upload_us = self
            .moe_expert_upload_us
            .saturating_add(duration_us(stage_start.elapsed()));

        Ok((loaded, pending_prefetch_uploads, pending_selected_uploads))
    }

    fn finish_pending_selected_uploads(
        &mut self,
        pending_prefetch_uploads: Vec<ExpertId>,
        pending_selected_uploads: Vec<ExpertId>,
        handles: &mut CpuExpertHandleStore,
        loaded: &mut BTreeSet<ExpertId>,
    ) -> Result<()> {
        for expert_id in pending_prefetch_uploads {
            let bytes = self.wait_selected_upload(expert_id)?;
            Self::install_resident_handle(handles, expert_id, bytes)?;
            loaded.insert(expert_id);
        }
        for expert_id in pending_selected_uploads {
            let bytes = self.wait_selected_overlap_upload(expert_id)?;
            Self::install_resident_handle(handles, expert_id, bytes)?;
            loaded.insert(expert_id);
        }
        Ok(())
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
            expert_selected_async_upload_submitted: self.expert_selected_async_upload_submitted,
            expert_selected_async_upload_completed: self.expert_selected_async_upload_completed,
            expert_selected_async_upload_bytes: self.expert_selected_async_upload_bytes,
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
            cuda_graph_capture_attempts: 0,
            cuda_graph_capture_successes: 0,
            cuda_graph_capture_failures: 0,
            cuda_graph_capture_us: 0,
            cuda_graph_capture_warmup_us: 0,
            cuda_graph_capture_prepare_us: 0,
            cuda_graph_capture_record_us: 0,
            cuda_graph_full_capture_failures: 0,
            cuda_graph_captured_segments: 0,
            cuda_graph_one_shot_retires: 0,
            cuda_graph_replays: 0,
            cuda_graph_replay_us: 0,
            cuda_graph_replay_fallbacks: 0,
            expert_predictor_stats: Default::default(),
        }
    }

    pub(crate) fn linear_matvec(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        if input.len() != linear.format.in_features() {
            return Err(Error::Model(format!(
                "artifact linear {:?} input length mismatch: expected {}, got {}",
                linear.role,
                linear.format.in_features(),
                input.len()
            )));
        }
        let input = linear.execution_input(input)?;
        let key = self.ensure_linear_uploaded(linear)?;
        let handle = self.linears.get(&key).expect("inserted above");
        self.ops.artifact_linear_matvec(handle, input.as_ref())
    }

    /// Device-resident linear matvec: input is already on device, output stays
    /// on device. Applies the same activation-quantization policy as the host
    /// `linear_matvec` path, in-place on the input buffer.
    pub(crate) fn linear_matvec_from_device(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let key = self.ensure_linear_uploaded(linear)?;
        let handle = self.linears.get(&key).expect("inserted above");
        let mut output = self.ops.zero_f32_buffer(handle.shape().out_features())?;
        self.linear_matvec_from_device_into(linear, input, &mut output)?;
        Ok(output)
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
        let key = self.ensure_linear_uploaded(linear)?;
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

    /// Device-resident RMS norm with cached weight.
    pub(crate) fn rms_norm_device_cached(
        &mut self,
        name: &str,
        input: &ferrule_cuda::context::CudaF32Buffer,
        weight: &[f32],
        eps: f32,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.ensure_norm_uploaded(name, weight)?;
        // Safe to unwrap: ensure_norm_uploaded just inserted it.
        let weight_buf = self.norm_weights.get(name).expect("inserted above");
        self.ops.rms_norm_from_device(input, weight_buf, eps)
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

    /// Batched host-to-host RMS norm using a device-resident rows kernel and a
    /// cached affine weight. This is the current prefill bridge until the whole
    /// layer prefill state stays on device.
    pub(crate) fn rms_norm_rows_cached(
        &mut self,
        name: &str,
        input: &[f32],
        rows: usize,
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>> {
        if rows == 0 || input.len() != rows * weight.len() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 CUDA RMS rows length mismatch: rows={rows} input={} weight={}",
                input.len(),
                weight.len()
            )));
        }
        self.ensure_norm_uploaded(name, weight)?;
        let input_dev = self.ops.upload_f32_buffer(input)?;
        let output_dev = self.rms_norm_rows_device_cached(name, &input_dev, rows, weight, eps)?;
        self.ops.download_f32_buffer(&output_dev)
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

    /// Device-resident hc_pre: state on device, weights cached, outputs stay on device.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hc_pre_from_device(
        &mut self,
        name: &str,
        state: &ferrule_cuda::context::CudaF32Buffer,
        weights: &HyperConnectionWeights,
        tokens: usize,
        config: HyperConnectionConfig,
    ) -> Result<(
        ferrule_cuda::context::CudaF32Buffer,
        ferrule_cuda::context::CudaF32Buffer,
        ferrule_cuda::context::CudaF32Buffer,
        ferrule_cuda::context::CudaF32Buffer,
    )> {
        self.ensure_hc_weights_uploaded(name, weights)?;
        let hw = self.hc_weights.get(name).expect("inserted above");
        self.ops.hc_pre_from_device(
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
        )
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

    /// Device-resident hc_post: all inputs on device, output stays on device.
    pub(crate) fn hc_post_from_device(
        &self,
        hidden: &ferrule_cuda::context::CudaF32Buffer,
        residual: &ferrule_cuda::context::CudaF32Buffer,
        split_post: &ferrule_cuda::context::CudaF32Buffer,
        split_comb: &ferrule_cuda::context::CudaF32Buffer,
        tokens: usize,
        config: HyperConnectionConfig,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.ops.hc_post_from_device(
            hidden,
            residual,
            split_post,
            split_comb,
            tokens,
            config.hc_mult,
            config.hidden_size,
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

    pub(crate) fn linear_topk(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &[f32],
        top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        if input.len() != linear.format.in_features() {
            return Err(Error::Model(format!(
                "artifact linear {:?} top-k input length mismatch: expected {}, got {}",
                linear.role,
                linear.format.in_features(),
                input.len()
            )));
        }
        let input = linear.execution_input(input)?;
        let key = self.ensure_linear_uploaded(linear)?;
        let handle = self.linears.get(&key).expect("inserted above");
        Ok(self
            .ops
            .artifact_linear_topk(handle, input.as_ref(), top_k)?
            .into_iter()
            .map(|(token_id, logit)| DeepSeekV4Logit { token_id, logit })
            .collect())
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
    ) -> Result<Option<Vec<Vec<ExpertRoute>>>> {
        let enabled = std::env::var("FERRULE_DSV4_DEVICE_ROUTER_TOPK")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !enabled {
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
        let (indices, weights) = self.ops.dsv4_router_topk_sqrt_softplus_rows_from_device(
            logits,
            bias,
            tokens,
            experts,
            top_k,
            router_policy.route_scale,
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
    ) -> Result<Vec<DeepSeekV4Logit>> {
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
        let vocab_rows = slice.shape[0];
        let hidden_cols = slice.shape[1];
        if hidden.len() != hidden_cols {
            return Err(Error::Model(format!(
                "DeepSeek-V4 CUDA output head input mismatch: expected {hidden_cols}, got {}",
                hidden.len()
            )));
        }

        let upload_start = Instant::now();
        self.decode_arena.hidden = Some(self.ops.upload_f32_buffer(hidden)?);
        self.output_head_hidden_uploads = self.output_head_hidden_uploads.saturating_add(1);
        self.output_head_hidden_upload_us = self
            .output_head_hidden_upload_us
            .saturating_add(duration_us(upload_start.elapsed()));
        let hidden_device = self
            .decode_arena
            .hidden
            .take()
            .expect("hidden uploaded immediately above");
        let result = self.output_head_topk_chunks_with_device(
            slice,
            &hidden_device,
            top_k,
            chunk_rows,
            reader,
            vocab_rows,
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
        vocab_rows: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        self.output_head_calls = self.output_head_calls.saturating_add(1);
        let mut top = Vec::<DeepSeekV4Logit>::new();
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
            let handle = self.linears.get(&key).expect("inserted above");
            let topk_start = Instant::now();
            let chunk_top =
                self.ops
                    .artifact_linear_topk_from_device(handle, hidden, top_k.min(rows))?;
            self.output_head_topk_us = self
                .output_head_topk_us
                .saturating_add(duration_us(topk_start.elapsed()));
            let merge_start = Instant::now();
            top.extend(
                chunk_top
                    .into_iter()
                    .map(|(token_id, logit)| DeepSeekV4Logit {
                        token_id: token_id + start as u32,
                        logit,
                    }),
            );
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

    pub(crate) fn rms_norm(&self, input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
        self.ops.rms_norm(input, weight, eps)
    }

    pub(crate) fn rms_norm_heads(
        &self,
        input: &[f32],
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        self.ops.rms_norm_heads(input, heads, head_dim, eps)
    }

    pub(crate) fn rms_norm_heads_from_device(
        &self,
        input: &ferrule_cuda::context::CudaF32Buffer,
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.ops
            .rms_norm_heads_from_device(input, heads, head_dim, eps)
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

    pub(crate) fn swiglu_ffn(
        &mut self,
        ffn: &SwiGluFfnPayload,
        input: &[f32],
        output_scale: f32,
    ) -> Result<Vec<f32>> {
        let gate_key = self.ensure_linear_uploaded(&ffn.gate)?;
        let up_key = self.ensure_linear_uploaded(&ffn.up)?;
        let down_key = self.ensure_linear_uploaded(&ffn.down)?;
        let gate = self.linears.get(&gate_key).expect("inserted above");
        let up = self.linears.get(&up_key).expect("inserted above");
        let down = self.linears.get(&down_key).expect("inserted above");
        self.ops
            .artifact_swiglu_ffn_matvec(gate, up, down, input, output_scale, ffn.swiglu_limit)
    }

    pub(crate) fn swiglu_ffn_from_device(
        &mut self,
        ffn: &SwiGluFfnPayload,
        input: &ferrule_cuda::context::CudaF32Buffer,
        output_scale: f32,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let gate_key = self.ensure_linear_uploaded(&ffn.gate)?;
        let up_key = self.ensure_linear_uploaded(&ffn.up)?;
        let down_key = self.ensure_linear_uploaded(&ffn.down)?;
        let gate = self.linears.get(&gate_key).expect("inserted above");
        let up = self.linears.get(&up_key).expect("inserted above");
        let down = self.linears.get(&down_key).expect("inserted above");
        self.ops.artifact_swiglu_ffn_from_device(
            gate,
            up,
            down,
            input,
            output_scale,
            ffn.swiglu_limit,
        )
    }

    pub(crate) fn hc_pre(
        &self,
        state: &[f32],
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionWeights,
    ) -> Result<HyperConnectionPreOutput> {
        weights.validate(config)?;
        let (hidden, pre, post, comb) = self.ops.hc_pre_f32(
            state,
            &weights.function,
            &weights.scale,
            &weights.base,
            tokens,
            config.hc_mult,
            config.hidden_size,
            config.sinkhorn_iters,
            config.eps,
            config.norm_eps,
        )?;
        Ok(HyperConnectionPreOutput {
            hidden,
            split: HyperConnectionSplit {
                tokens,
                hc_mult: config.hc_mult,
                pre,
                post,
                comb,
            },
        })
    }

    pub(crate) fn hc_post(
        &self,
        hidden: &[f32],
        residual: &[f32],
        config: HyperConnectionConfig,
        split: &HyperConnectionSplit,
    ) -> Result<Vec<f32>> {
        self.ops.hc_post_f32(
            hidden,
            residual,
            &split.post,
            &split.comb,
            split.tokens,
            config.hc_mult,
            config.hidden_size,
        )
    }

    pub(crate) fn hc_head(
        &self,
        state: &[f32],
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionHeadWeights,
    ) -> Result<Vec<f32>> {
        weights.validate(config)?;
        self.ops.hc_head_f32(
            state,
            &weights.function,
            &weights.scale,
            &weights.base,
            tokens,
            config.hc_mult,
            config.hidden_size,
            config.eps,
            config.norm_eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn routed_moe_step(
        &mut self,
        layer: usize,
        input: &[f32],
        token_id: u32,
        router: &RouterArtifactPayload,
        predicted_experts: &[usize],
        router_policy: &ExpertRouterPolicy,
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
        shared_expert: Option<&SwiGluFfnPayload>,
    ) -> Result<RoutedMoeStepOutput> {
        let stage_start = Instant::now();
        let logits = self.linear_matvec(&router.weight, input)?;
        self.moe_router_us = self
            .moe_router_us
            .saturating_add(duration_us(stage_start.elapsed()));

        let stage_start = Instant::now();
        let hash_experts = router.hash_experts_for_token(token_id)?;
        let routes =
            router_policy.route(&logits, router.bias.as_deref(), hash_experts.as_deref())?;
        let selected = routes.iter().map(|route| route.expert).collect::<Vec<_>>();
        self.moe_routing_us = self
            .moe_routing_us
            .saturating_add(duration_us(stage_start.elapsed()));

        self.moe_predicted_experts = self
            .moe_predicted_experts
            .saturating_add(predicted_experts.len() as u64);
        self.sync_planner_residency_for_layer(layer, planner)?;
        let stage_start = Instant::now();
        let streaming = planner.plan_layer_step(layer, &selected, predicted_experts)?;
        self.moe_plan_us = self
            .moe_plan_us
            .saturating_add(duration_us(stage_start.elapsed()));
        self.expert_selected = self.expert_selected.saturating_add(routes.len() as u64);
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

        // Upload the shared input once and reuse across all selected experts.
        // This eliminates 5/6 of the per-expert H2D copies (258 → 43 input uploads/token).
        // Device-side accumulation via saxpy eliminates 258 D2H copies/token.
        let input_len = input.len();
        if routes.is_empty() {
            let routed_output = vec![0.0f32; input_len];
            let stage_start = Instant::now();
            let shared_output = shared_expert
                .map(|shared| self.swiglu_ffn(shared, input, 1.0))
                .transpose()?;
            self.moe_shared_us = self
                .moe_shared_us
                .saturating_add(duration_us(stage_start.elapsed()));
            let mut output = routed_output.clone();
            if let Some(shared) = &shared_output {
                for (dst, value) in output.iter_mut().zip(shared.iter()) {
                    *dst += value;
                }
            }
            let stage_start = Instant::now();
            planner.commit_step_loaded(&streaming, loaded_experts.iter().copied())?;
            self.moe_commit_us = self
                .moe_commit_us
                .saturating_add(duration_us(stage_start.elapsed()));
            return Ok(RoutedMoeStepOutput {
                routes,
                streaming,
                routed_output,
                shared_output,
                output,
            });
        }

        // Use the first expert's handles to determine FP4 shapes for input prep.
        let first_expert_id = ExpertId::new(layer, routes[0].expert);
        let first_expert = self.experts.get(&first_expert_id).ok_or_else(|| {
            Error::Model(format!(
                "CUDA expert handle missing for layer {} expert {}",
                first_expert_id.layer, first_expert_id.expert
            ))
        })?;
        let stage_start = Instant::now();
        let shared_input = self.ops.prepare_fp4_expert_input(
            &first_expert.gate,
            &first_expert.up,
            &first_expert.down,
            input,
        )?;
        let mut accumulator = self.ops.zero_f32_buffer(input_len)?;
        self.moe_workspace_us = self
            .moe_workspace_us
            .saturating_add(duration_us(stage_start.elapsed()));
        let swiglu_limit = shared_expert.map(|ffn| ffn.swiglu_limit).unwrap_or(0.0);

        let stage_start = Instant::now();
        for route in &routes {
            let expert_id = ExpertId::new(layer, route.expert);
            let expert = self.experts.get(&expert_id).ok_or_else(|| {
                Error::Model(format!(
                    "CUDA expert handle missing for layer {} expert {}",
                    expert_id.layer, expert_id.expert
                ))
            })?;
            let expert_out = self.ops.fp4_swiglu_ffn_from_device(
                &expert.gate,
                &expert.up,
                &expert.down,
                &shared_input,
                route.weight,
                swiglu_limit,
            )?;
            // Accumulate on device: accumulator += 1.0 * expert_out
            self.ops.saxpy_into(1.0, &expert_out, &mut accumulator)?;
        }
        let routed_output = self.ops.download_f32_buffer(&accumulator)?;
        self.moe_compute_submit_us = self
            .moe_compute_submit_us
            .saturating_add(duration_us(stage_start.elapsed()));
        let stage_start = Instant::now();
        let shared_output = shared_expert
            .map(|shared| self.swiglu_ffn(shared, input, 1.0))
            .transpose()?;
        self.moe_shared_us = self
            .moe_shared_us
            .saturating_add(duration_us(stage_start.elapsed()));
        let mut output = routed_output.clone();
        if let Some(shared) = &shared_output {
            if shared.len() != output.len() {
                return Err(Error::Model(format!(
                    "shared expert output length mismatch: routed={}, shared={}",
                    output.len(),
                    shared.len()
                )));
            }
            for (dst, value) in output.iter_mut().zip(shared.iter()) {
                *dst += value;
            }
        }
        let stage_start = Instant::now();
        planner.commit_step_loaded(&streaming, loaded_experts.iter().copied())?;
        self.moe_commit_us = self
            .moe_commit_us
            .saturating_add(duration_us(stage_start.elapsed()));

        Ok(RoutedMoeStepOutput {
            routes,
            streaming,
            routed_output,
            shared_output,
            output,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn routed_moe_prefill_batch(
        &mut self,
        layer: usize,
        input: &[f32],
        token_ids: &[u32],
        router: &RouterArtifactPayload,
        predicted_experts: &[usize],
        router_policy: &ExpertRouterPolicy,
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
        shared_expert: Option<&SwiGluFfnPayload>,
    ) -> Result<Vec<f32>> {
        let input_dev = self.ops.upload_f32_buffer(input)?;
        let output_dev = self.routed_moe_prefill_batch_from_device(
            layer,
            &input_dev,
            token_ids,
            router,
            predicted_experts,
            router_policy,
            planner,
            reader,
            handles,
            shared_expert,
        )?;
        self.ops.download_f32_buffer(&output_dev)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn routed_moe_prefill_batch_from_device(
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
        shared_expert: Option<&SwiGluFfnPayload>,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
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

        let stage_start = Instant::now();
        let router_logits_dev = self.linear_rows_from_device(&router.weight, input_dev, tokens)?;
        let routes_by_token = if let Some(routes) = self
            .dsv4_router_topk_routes_from_device_logits(
                layer,
                &router_logits_dev,
                tokens,
                router,
                router_policy,
            )? {
            routes
        } else {
            let router_logits = self.ops.download_f32_buffer(&router_logits_dev)?;
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

        let mut output_dev = self.ops.zero_f32_buffer(input_dev.len())?;

        if let Some(shared) = shared_expert {
            let stage_start = Instant::now();
            let gate_key = self.ensure_linear_uploaded(&shared.gate)?;
            let up_key = self.ensure_linear_uploaded(&shared.up)?;
            let down_key = self.ensure_linear_uploaded(&shared.down)?;
            let gate = self.linears.get(&gate_key).expect("inserted above");
            let up = self.linears.get(&up_key).expect("inserted above");
            let down = self.linears.get(&down_key).expect("inserted above");
            let shared_dev = self.ops.artifact_swiglu_ffn_rows_from_device(
                gate,
                up,
                down,
                &input_dev,
                tokens,
                1.0,
                shared.swiglu_limit,
            )?;
            self.ops.saxpy_into(1.0, &shared_dev, &mut output_dev)?;
            self.moe_shared_us = self
                .moe_shared_us
                .saturating_add(duration_us(stage_start.elapsed()));
        }

        let top_k = router_policy.top_k.max(1);
        if top_k > 64 {
            return Err(Error::Model(format!(
                "CUDA routed MoE prefill supports at most 64 experts per batched tile, got top_k={top_k}"
            )));
        }
        let resident_slots = planner.policy().gpu_slots_per_layer.max(top_k);
        let default_prefill_experts = top_k.saturating_mul(2).min(16).max(top_k);
        let requested_prefill_experts = std::env::var("FERRULE_CUDA_MOE_PREFILL_MAX_EXPERTS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(default_prefill_experts);
        let prefill_expert_limit = requested_prefill_experts
            .max(top_k)
            .min(resident_slots)
            .min(64);
        let prefill_batch_cols = std::env::var("FERRULE_CUDA_MOE_PREFILL_BATCH_COLS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(8)
            .min(8);

        let mut streaming_steps = Vec::new();
        let mut tile_start = 0usize;
        while tile_start < tokens {
            let mut tile_end = tile_start;
            let mut selected_set = BTreeSet::<usize>::new();
            while tile_end < tokens && tile_end - tile_start < prefill_batch_cols {
                let mut candidate = selected_set.clone();
                for route in &routes_by_token[tile_end] {
                    candidate.insert(route.expert);
                }
                if candidate.len() > prefill_expert_limit && tile_end > tile_start {
                    break;
                }
                if candidate.len() > prefill_expert_limit {
                    return Err(Error::Model(format!(
                        "CUDA routed MoE prefill token selects {} experts but batched prefill limit is {} for layer {}",
                        candidate.len(), prefill_expert_limit, layer
                    )));
                }
                selected_set = candidate;
                tile_end += 1;
            }

            let selected = selected_set.iter().copied().collect::<Vec<_>>();
            self.moe_predicted_experts = self
                .moe_predicted_experts
                .saturating_add(predicted_experts.len() as u64);
            self.sync_planner_residency_for_layer(layer, planner)?;
            let stage_start = Instant::now();
            let streaming = planner.plan_layer_step(layer, &selected, predicted_experts)?;
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

            let compute_start = Instant::now();
            let batch_cols = tile_end - tile_start;
            let row_indices = (tile_start..tile_end)
                .map(|token_idx| token_idx as i32)
                .collect::<Vec<_>>();
            let row_indices_dev = self.ops.upload_i32_buffer(&row_indices)?;
            let grouped_input =
                self.ops
                    .gather_f32_rows(&input_dev, &row_indices_dev, batch_cols, hidden_size)?;
            let mut grouped_output = self.ops.zero_f32_buffer(batch_cols * hidden_size)?;

            let selected_slots = selected
                .iter()
                .enumerate()
                .map(|(slot, &expert)| (expert, slot))
                .collect::<BTreeMap<_, _>>();
            let mut route_weights = vec![0.0f32; selected.len() * batch_cols];
            for (col, token_idx) in (tile_start..tile_end).enumerate() {
                for route in &routes_by_token[token_idx] {
                    let slot = selected_slots.get(&route.expert).ok_or_else(|| {
                        Error::Internal(format!(
                            "CUDA prefill MoE route expert {} missing from selected union for layer {}",
                            route.expert, layer
                        ))
                    })?;
                    route_weights[*slot * batch_cols + col] += route.weight;
                }
            }

            let mut gate_handles = Vec::with_capacity(selected.len());
            let mut up_handles = Vec::with_capacity(selected.len());
            let mut down_handles = Vec::with_capacity(selected.len());
            let mut intermediate_size = None;
            for &expert in &selected {
                let expert_id = ExpertId::new(layer, expert);
                let expert_handles = self.experts.get(&expert_id).ok_or_else(|| {
                    Error::Model(format!(
                        "CUDA prefill MoE expert handle missing for layer {} expert {}",
                        layer, expert
                    ))
                })?;
                let expert_intermediate_size = expert_handles.gate.shape().out_features();
                let expert_hidden_size = expert_handles.down.shape().out_features();
                if expert_hidden_size != hidden_size {
                    return Err(Error::Model(format!(
                        "CUDA prefill MoE hidden mismatch: expert hidden {} input hidden {}",
                        expert_hidden_size, hidden_size
                    )));
                }
                if let Some(expected) = intermediate_size {
                    if expert_intermediate_size != expected {
                        return Err(Error::Model(format!(
                            "CUDA prefill MoE intermediate mismatch: expert {} has {} expected {}",
                            expert, expert_intermediate_size, expected
                        )));
                    }
                } else {
                    intermediate_size = Some(expert_intermediate_size);
                }
                gate_handles.push(&expert_handles.gate);
                up_handles.push(&expert_handles.up);
                down_handles.push(&expert_handles.down);
            }
            let intermediate_size = intermediate_size.ok_or_else(|| {
                Error::Internal(format!(
                    "CUDA prefill MoE selected no experts for non-empty tile in layer {}",
                    layer
                ))
            })?;

            let workspace_needs_init = match &self.decode_arena.moe_prefill_workspace {
                Some(workspace) => !workspace.matches_cols(
                    selected.len(),
                    batch_cols,
                    hidden_size,
                    intermediate_size,
                    hidden_size,
                ),
                None => true,
            };
            if workspace_needs_init {
                self.decode_arena.moe_prefill_workspace =
                    Some(self.ops.moe_batched_workspace_cols(
                        prefill_expert_limit,
                        prefill_batch_cols,
                        hidden_size,
                        intermediate_size,
                        hidden_size,
                    )?);
            }
            let ops = &self.ops;
            let workspace = self
                .decode_arena
                .moe_prefill_workspace
                .as_mut()
                .expect("prefill MoE workspace initialized above");
            ops.moe_experts_batched_cols_add_into_from_device(
                &gate_handles,
                &up_handles,
                &down_handles,
                &route_weights,
                &grouped_input,
                hidden_size,
                batch_cols,
                shared_expert.map(|ffn| ffn.swiglu_limit).unwrap_or(0.0),
                selected.len(),
                intermediate_size,
                hidden_size,
                workspace,
                &mut grouped_output,
            )?;
            self.ops.scatter_add_f32_rows(
                &grouped_output,
                &row_indices_dev,
                &mut output_dev,
                batch_cols,
                hidden_size,
            )?;
            self.moe_compute_submit_us = self
                .moe_compute_submit_us
                .saturating_add(duration_us(compute_start.elapsed()));

            let stage_start = Instant::now();
            planner.commit_step_loaded(&streaming, loaded_experts.iter().copied())?;
            self.moe_commit_us = self
                .moe_commit_us
                .saturating_add(duration_us(stage_start.elapsed()));
            streaming_steps.push(streaming);

            tile_start = tile_end;
        }

        self.moe_access_events
            .push(ExpertBatchAccessEvent::from_routes_by_token(
                layer,
                ExpertAccessPhase::Prefill,
                tokens,
                &routes_by_token,
                &streaming_steps,
            ));

        Ok(output_dev)
    }

    fn route_decode_moe_from_device(
        &mut self,
        layer: usize,
        input: &ferrule_cuda::context::CudaF32Buffer,
        token_id: u32,
        router: &RouterArtifactPayload,
        router_policy: &ExpertRouterPolicy,
    ) -> Result<CudaMoeRoutes> {
        // Clone the input buffer for the router linear (avoids consuming the
        // caller's buffer). Uses device-to-device copy — no host round-trip.
        let stage_start = Instant::now();
        let mut router_input = self.ops.clone_f32_buffer(input)?;
        let logits_dev = self.linear_matvec_from_device(&router.weight, &mut router_input)?;
        let routes = if let Some(mut routes_by_token) = self
            .dsv4_router_topk_routes_from_device_logits(
                layer,
                &logits_dev,
                1,
                router,
                router_policy,
            )? {
            routes_by_token.pop().unwrap_or_default()
        } else {
            let logits = self.ops.download_f32_buffer(&logits_dev)?;
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

        if self.selected_upload_overlap_enabled() {
            let (loaded_experts, pending_prefetch_uploads, pending_selected_uploads) = self
                .begin_selected_materialization_and_queue_prefetch_overlap(
                    layer,
                    &streaming.loads,
                    reader,
                    handles,
                )?;
            Ok(CudaMoeMaterialization {
                streaming,
                loaded_experts,
                pending_prefetch_uploads,
                pending_selected_uploads,
            })
        } else {
            let loaded_experts = self.materialize_selected_and_queue_prefetch(
                layer,
                &streaming.loads,
                reader,
                handles,
            )?;
            Ok(CudaMoeMaterialization {
                streaming,
                loaded_experts,
                pending_prefetch_uploads: Vec::new(),
                pending_selected_uploads: Vec::new(),
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn routed_moe_step_from_device(
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
        shared_expert: Option<&SwiGluFfnPayload>,
    ) -> Result<RoutedMoeStepOutput> {
        let mut device_output = self.routed_moe_step_device_output(
            layer,
            input,
            token_id,
            router,
            predicted_experts,
            router_policy,
            planner,
            reader,
            handles,
            shared_expert,
        )?;
        let output = self.ops.download_f32_buffer(&device_output.output_dev)?;
        device_output.moe.output = output;
        Ok(device_output.moe)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn routed_moe_step_device_output(
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
        shared_expert: Option<&SwiGluFfnPayload>,
    ) -> Result<CudaRoutedMoeStepOutput> {
        let CudaMoeRoutes { routes, selected } =
            self.route_decode_moe_from_device(layer, input, token_id, router, router_policy)?;
        let CudaMoeMaterialization {
            streaming,
            mut loaded_experts,
            pending_prefetch_uploads,
            pending_selected_uploads,
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
        let mut accumulator = self.ops.zero_f32_buffer(input.len())?;
        let swiglu_limit = shared_expert.map(|ffn| ffn.swiglu_limit).unwrap_or(0.0);
        let shared_dev = shared_expert
            .map(|shared| self.swiglu_ffn_from_device(shared, input, 1.0))
            .transpose()?;
        if let Some(shared) = &shared_dev {
            self.ops.saxpy_into(1.0, shared, &mut accumulator)?;
        }
        self.moe_shared_us = self
            .moe_shared_us
            .saturating_add(duration_us(stage_start.elapsed()));

        self.finish_pending_selected_uploads(
            pending_prefetch_uploads,
            pending_selected_uploads,
            handles,
            &mut loaded_experts,
        )?;

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
                &mut accumulator,
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

        Ok(CudaRoutedMoeStepOutput {
            moe: RoutedMoeStepOutput {
                routes,
                streaming,
                routed_output: Vec::new(),
                shared_output: None,
                output: Vec::new(),
            },
            output_dev: accumulator,
        })
    }

    /// Prepare fixed routed-MoE operands for CUDA graph capture.
    ///
    /// This must run before stream capture. It uploads expert pointer arrays and
    /// route weights into persistent device workspace buffers so replay never
    /// depends on stack-allocated host arrays captured by memcpy nodes.
    pub(crate) fn prepare_routed_moe_graph_safe(
        &mut self,
        input_len: usize,
        route_experts: &[usize],
        route_weights: &[f32],
        layer: usize,
    ) -> Result<()> {
        if route_experts.is_empty() {
            return Ok(());
        }
        if route_weights.len() < route_experts.len() {
            return Err(Error::Internal(format!(
                "graph-safe MoE route weights too short: weights={} experts={}",
                route_weights.len(),
                route_experts.len()
            )));
        }

        let num_experts = route_experts.len().min(6);
        let mut gate_handles: Vec<&ferrule_cuda::context::CudaArtifactLinearHandle> =
            Vec::with_capacity(num_experts);
        let mut up_handles: Vec<&ferrule_cuda::context::CudaArtifactLinearHandle> =
            Vec::with_capacity(num_experts);
        let mut down_handles: Vec<&ferrule_cuda::context::CudaArtifactLinearHandle> =
            Vec::with_capacity(num_experts);
        let mut route_weights_arr = [0.0f32; 6];
        for (i, &expert_id) in route_experts.iter().take(num_experts).enumerate() {
            let expert_key = ExpertId::new(layer, expert_id);
            let expert = self.experts.get(&expert_key).ok_or_else(|| {
                Error::Model(format!(
                    "graph-safe MoE prepare: expert {} not resident for layer {}",
                    expert_id, layer
                ))
            })?;
            gate_handles.push(&expert.gate);
            up_handles.push(&expert.up);
            down_handles.push(&expert.down);
            route_weights_arr[i] = route_weights[i];
        }

        let first_expert_id = ExpertId::new(layer, route_experts[0]);
        let first_expert = self.experts.get(&first_expert_id).expect("checked above");
        let intermediate_size = first_expert.gate.shape().out_features();
        let hidden_size = first_expert.down.shape().out_features();

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
        for i in 0..num_experts {
            gate_arr[i] = gate_handles[i];
            up_arr[i] = up_handles[i];
            down_arr[i] = down_handles[i];
        }

        let workspace_needs_init = self
            .decode_arena
            .moe_graph_workspaces
            .get(&layer)
            .map(|workspace| !workspace.matches(6, input_len, intermediate_size, hidden_size))
            .unwrap_or(true);
        if workspace_needs_init {
            self.decode_arena.moe_graph_workspaces.insert(
                layer,
                self.ops
                    .moe_batched_workspace(6, input_len, intermediate_size, hidden_size)?,
            );
        }
        let workspace = self
            .decode_arena
            .moe_graph_workspaces
            .get_mut(&layer)
            .expect("MoE graph workspace initialized above");
        self.ops.prepare_moe_experts_batched_workspace(
            &gate_arr,
            &up_arr,
            &down_arr,
            &route_weights_arr,
            input_len,
            num_experts,
            intermediate_size,
            hidden_size,
            workspace,
        )
    }

    /// Graph-safe MoE step: uses pre-determined routes (no router D2H),
    /// assumes all experts are already GPU-resident (no streaming), and
    /// only launches kernels. Safe for CUDA graph capture.
    ///
    /// `route_experts` and `route_weights` come from a warmup pass — they
    /// are fixed for the lifetime of the captured graph. The graph captures
    /// the exact kernel sequence for this expert set.
    pub(crate) fn routed_moe_step_graph_safe(
        &mut self,
        input: &ferrule_cuda::context::CudaF32Buffer,
        route_experts: &[usize],
        route_weights: &[f32],
        layer: usize,
        shared_expert: Option<&SwiGluFfnPayload>,
        accumulator: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        let swiglu_limit = shared_expert.map(|ffn| ffn.swiglu_limit).unwrap_or(0.0);
        // Zero the accumulator in-place (cuMemsetD32Async, graph-safe).
        self.ops.zero_f32_buffer_in_place(accumulator)?;

        let shared_dev = shared_expert
            .map(|shared| self.swiglu_ffn_from_device(shared, input, 1.0))
            .transpose()?;
        if let Some(shared) = &shared_dev {
            self.ops.saxpy_into(1.0, shared, accumulator)?;
        }

        if !route_experts.is_empty() {
            let num_experts = route_experts.len();
            let mut gate_handles: Vec<&ferrule_cuda::context::CudaArtifactLinearHandle> =
                Vec::with_capacity(num_experts);
            let mut up_handles: Vec<&ferrule_cuda::context::CudaArtifactLinearHandle> =
                Vec::with_capacity(num_experts);
            let mut down_handles: Vec<&ferrule_cuda::context::CudaArtifactLinearHandle> =
                Vec::with_capacity(num_experts);
            let mut route_weights_arr = [0.0f32; 6];
            for (i, &expert_id) in route_experts.iter().enumerate() {
                let expert_key = ExpertId::new(layer, expert_id);
                let expert = self.experts.get(&expert_key).ok_or_else(|| {
                    Error::Model(format!(
                        "graph-safe MoE: expert {} not resident for layer {}",
                        expert_id, layer
                    ))
                })?;
                gate_handles.push(&expert.gate);
                up_handles.push(&expert.up);
                down_handles.push(&expert.down);
                if i < 6 {
                    route_weights_arr[i] = route_weights[i];
                }
            }

            let first_expert_id = ExpertId::new(layer, route_experts[0]);
            let first_expert = self.experts.get(&first_expert_id).expect("checked above");
            let intermediate_size = first_expert.gate.shape().out_features();
            let hidden_size = first_expert.down.shape().out_features();

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

            let ops = &self.ops;
            let workspace = self
                .decode_arena
                .moe_graph_workspaces
                .get_mut(&layer)
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "graph-safe MoE workspace missing for layer {layer}; call prepare_routed_moe_graph_safe before capture"
                    ))
                })?;
            if !workspace.matches(6, input.len(), intermediate_size, hidden_size) {
                return Err(Error::Internal(format!(
                    "graph-safe MoE workspace shape mismatch for layer {layer}: input={} intermediate={} hidden={}",
                    input.len(), intermediate_size, hidden_size
                )));
            }
            // Pointer arrays and route weights must already be prepared outside
            // stream capture by `prepare_routed_moe_graph_safe`; this call only
            // records graph-safe kernels and device-to-device copies.
            let _ = (&gate_arr, &up_arr, &down_arr, &route_weights_arr);
            ops.moe_experts_batched_add_into_from_device_prepared(
                input,
                swiglu_limit,
                num_experts.min(6),
                intermediate_size,
                hidden_size,
                workspace,
                accumulator,
            )?;
        }

        Ok(())
    }

    fn upload_expert_bundle(&self, bundle: &ExpertComputeBundle) -> Result<CudaFp4ExpertHandles> {
        Ok(CudaFp4ExpertHandles {
            gate: self.upload_expert_linear(&bundle.gate)?,
            up: self.upload_expert_linear(&bundle.up)?,
            down: self.upload_expert_linear(&bundle.down)?,
            bytes: bundle.total_bytes(),
        })
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
        )
    }

    pub(crate) fn sparse_attention(
        &mut self,
        query: &[f32],
        values: &[f32],
        topk: &[isize],
        sink: &[f32],
        tokens: usize,
        kv_len: usize,
        spec: SparseAttentionSpec,
    ) -> Result<Vec<f32>> {
        let topk_i32 = sparse_topk_i32(topk)?;
        let shape = ferrule_cuda::transformer::sparse_attention::CudaSparseAttentionShape {
            batch_size: 1,
            tokens_per_batch: tokens,
            kv_len,
            heads: spec.heads,
            head_dim: spec.head_dim,
            topk: spec.topk,
            softmax_scale: spec.softmax_scale,
        };
        self.ops
            .sparse_attention_sink_f32(query, values, &topk_i32, sink, shape)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sparse_attention_with_device_query(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        values: &[f32],
        topk: &[isize],
        sink: &[f32],
        tokens: usize,
        kv_len: usize,
        spec: SparseAttentionSpec,
        layer: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let values_dev = self.ops.upload_f32_buffer(values)?;
        self.sparse_attention_with_device_query_and_values(
            query,
            &values_dev,
            topk,
            sink,
            tokens,
            kv_len,
            spec,
            layer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sparse_attention_with_device_query_and_values(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        values: &ferrule_cuda::context::CudaF32Buffer,
        topk: &[isize],
        sink: &[f32],
        tokens: usize,
        kv_len: usize,
        spec: SparseAttentionSpec,
        layer: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let topk_i32 = sparse_topk_i32(topk)?;
        let topk_dev = self.ops.upload_i32_buffer(&topk_i32)?;
        self.sparse_attention_with_device_query_values_topk(
            query, values, &topk_dev, sink, tokens, kv_len, spec, layer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sparse_attention_with_device_query_values_topk(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        values: &ferrule_cuda::context::CudaF32Buffer,
        topk: &ferrule_cuda::context::CudaI32Buffer,
        sink: &[f32],
        tokens: usize,
        kv_len: usize,
        spec: SparseAttentionSpec,
        layer: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        if topk.len() != tokens * spec.topk {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} CUDA sparse attention device topk mismatch: got {} expected {}",
                topk.len(),
                tokens * spec.topk
            )));
        }
        let mut output = self
            .ops
            .zero_f32_buffer(tokens * spec.heads * spec.head_dim)?;
        self.sparse_attention_with_device_query_values_topk_into(
            query,
            values,
            topk,
            sink,
            tokens,
            kv_len,
            spec,
            layer,
            &mut output,
        )?;
        Ok(output)
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
    pub(crate) fn sparse_attention_with_kv_cache_topk_into(
        &mut self,
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
                "DeepSeek-V4 layer {layer} CUDA KV sparse attention topk too small: got {} expected at least {}",
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
        let values = self.kv_values_device(layer);
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
    pub(crate) fn sparse_attention_with_combined_kv(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        layer: usize,
        topk: &[isize],
        sink: &[f32],
        tokens: usize,
        kv_len: usize,
        spec: SparseAttentionSpec,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let topk_i32 = sparse_topk_i32(topk)?;
        let topk_dev = self.ops.upload_i32_buffer(&topk_i32)?;
        self.sparse_attention_with_combined_kv_topk(
            query, layer, &topk_dev, sink, tokens, kv_len, spec,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sparse_attention_with_combined_kv_topk(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        layer: usize,
        topk: &ferrule_cuda::context::CudaI32Buffer,
        sink: &[f32],
        tokens: usize,
        kv_len: usize,
        spec: SparseAttentionSpec,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        if topk.len() != tokens * spec.topk {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} CUDA combined sparse attention device topk mismatch: got {} expected {}",
                topk.len(),
                tokens * spec.topk
            )));
        }
        let mut output = self
            .ops
            .zero_f32_buffer(tokens * spec.heads * spec.head_dim)?;
        self.sparse_attention_with_combined_kv_topk_into(
            query,
            layer,
            topk,
            sink,
            tokens,
            kv_len,
            spec,
            &mut output,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sparse_attention_with_combined_kv_topk_into(
        &mut self,
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
        let values = self.combined_kv_values_device(layer)?;
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

    pub(crate) fn grouped_output_a(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &[f32],
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
    ) -> Result<Vec<f32>> {
        let ArtifactLinearFormat::Fp8E4M3WithE8M0Scale {
            out_features,
            in_features,
            block_m,
            block_k,
        } = output_a.format
        else {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} CUDA grouped wo_a requires FP8 artifact format, got {:?}",
                output_a.format
            )));
        };
        if out_features != cfg.output_latent_dim() || in_features != cfg.output_group_input_dim() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} CUDA grouped wo_a shape mismatch: got [{out_features}, {in_features}], expected [{}, {}]",
                cfg.output_latent_dim(),
                cfg.output_group_input_dim()
            )));
        }
        if !cfg.o_lora_rank.is_multiple_of(block_m)
            || !cfg.output_group_input_dim().is_multiple_of(block_k)
        {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} CUDA grouped wo_a block shape unsupported: o_rank={} group_in={} block_m={block_m} block_k={block_k}",
                cfg.o_lora_rank,
                cfg.output_group_input_dim()
            )));
        }
        if context.len() != cfg.q_full_dim() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} context length mismatch: expected {}, got {}",
                cfg.q_full_dim(),
                context.len()
            )));
        }
        let scale = output_a.scale.as_ref().ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} grouped wo_a FP8 weight missing scale tensor"
            ))
        })?;
        let group_in = cfg.output_group_input_dim();
        let scale_cols = group_in.div_ceil(block_k);
        let scale_rows_per_group = cfg.o_lora_rank.div_ceil(block_m);
        let mut out = vec![0.0f32; cfg.output_latent_dim()];
        for group in 0..cfg.o_groups {
            let row_start = group * cfg.o_lora_rank;
            let weight_start = row_start * group_in;
            let weight_end = weight_start + cfg.o_lora_rank * group_in;
            let scale_start = (row_start / block_m) * scale_cols;
            let scale_end = scale_start + scale_rows_per_group * scale_cols;
            let key = format!("{}::grouped_wo_a::{group}", output_a.weight.slice.name);
            if !self.linears.contains_key(&key) {
                let handle = self.ops.upload_fp8_e4m3_e8m0_linear(
                    &output_a.weight.bytes[weight_start..weight_end],
                    &scale.bytes[scale_start..scale_end],
                    cfg.o_lora_rank,
                    group_in,
                    block_m,
                    block_k,
                )?;
                self.linears.insert(key.clone(), handle);
            }
            let handle = self.linears.get(&key).expect("inserted above");
            let context_start = group * group_in;
            let group_out = self.ops.artifact_linear_matvec(
                handle,
                &context[context_start..context_start + group_in],
            )?;
            out[row_start..row_start + cfg.o_lora_rank].copy_from_slice(&group_out);
        }
        Ok(out)
    }

    /// Device-resident grouped output_a matvec.
    ///
    /// `context` is the full `[q_full_dim]` device buffer. The dequantized f32
    /// output_a weights (`[output_latent_dim, group_in]`) are uploaded once and
    /// cached in `grouped_wo_a_weights`. Returns a `[output_latent_dim]` device
    /// buffer — no host round-trip.
    pub(crate) fn grouped_output_a_from_device(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &ferrule_cuda::context::CudaF32Buffer,
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.grouped_output_a_rows_from_device(output_a, context, 1, cfg, layer)
    }

    pub(crate) fn grouped_output_a_from_device_into(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &ferrule_cuda::context::CudaF32Buffer,
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.grouped_output_a_rows_from_device_into(output_a, context, 1, cfg, layer, output)
    }

    /// Batched device-resident grouped output_a for prefill rows.
    pub(crate) fn grouped_output_a_rows_from_host(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &[f32],
        rows: usize,
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        if context.len() != rows * cfg.q_full_dim() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} CUDA grouped wo_a rows context length mismatch: expected {}, got {}",
                rows * cfg.q_full_dim(),
                context.len()
            )));
        }
        let context_dev = self.ops.upload_f32_buffer(context)?;
        self.grouped_output_a_rows_from_device(output_a, &context_dev, rows, cfg, layer)
    }

    /// Batched device-resident grouped output_a for prefill rows.
    pub(crate) fn grouped_output_a_rows_from_device(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &ferrule_cuda::context::CudaF32Buffer,
        rows: usize,
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let mut output = self.ops.zero_f32_buffer(rows * cfg.output_latent_dim())?;
        self.grouped_output_a_rows_from_device_into(
            output_a,
            context,
            rows,
            cfg,
            layer,
            &mut output,
        )?;
        Ok(output)
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

    pub(crate) fn linear_rows_from_host_to_host(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        if rows == 0 || input.len() != rows * linear.format.in_features() {
            return Err(Error::Model(format!(
                "artifact linear {:?} rows input length mismatch: rows={} expected {}, got {}",
                linear.role,
                rows,
                rows * linear.format.in_features(),
                input.len()
            )));
        }
        let input_dev = self.ops.upload_f32_buffer(input)?;
        self.linear_rows_from_device_to_host(linear, &input_dev, rows)
    }

    pub(crate) fn linear_rows_from_device(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &ferrule_cuda::context::CudaF32Buffer,
        rows: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
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
        self.ops
            .artifact_linear_rows_from_device(handle, input, rows)
    }

    pub(crate) fn linear_rows_from_device_to_host(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &ferrule_cuda::context::CudaF32Buffer,
        rows: usize,
    ) -> Result<Vec<f32>> {
        let output = self.linear_rows_from_device(linear, input, rows)?;
        self.ops.download_f32_buffer(&output)
    }

    pub(crate) fn concat_attention_values_device(
        &mut self,
        window_values: &ferrule_cuda::context::CudaF32Buffer,
        compressed_values: &[f32],
        window_rows: usize,
        head_dim: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        if !compressed_values.len().is_multiple_of(head_dim) {
            return Err(Error::Model(format!(
                "CUDA attention value concat compressed length {} is not divisible by head_dim {head_dim}",
                compressed_values.len()
            )));
        }
        let compressed_dev = self.ops.upload_f32_buffer(compressed_values)?;
        self.concat_attention_values_device_buffers(
            window_values,
            &compressed_dev,
            window_rows,
            head_dim,
        )
    }

    pub(crate) fn concat_attention_values_device_buffers(
        &mut self,
        window_values: &ferrule_cuda::context::CudaF32Buffer,
        compressed_values: &ferrule_cuda::context::CudaF32Buffer,
        window_rows: usize,
        head_dim: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        if window_values.len() != window_rows * head_dim {
            return Err(Error::Model(format!(
                "CUDA attention value concat window mismatch: rows={window_rows} head_dim={head_dim} got {}",
                window_values.len()
            )));
        }
        if !compressed_values.len().is_multiple_of(head_dim) {
            return Err(Error::Model(format!(
                "CUDA attention value concat compressed length {} is not divisible by head_dim {head_dim}",
                compressed_values.len()
            )));
        }
        self.ops
            .concat_f32_buffers(window_values, compressed_values)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_topk_indices_from_device(
        &mut self,
        query: Option<&ferrule_cuda::context::CudaF32Buffer>,
        weights: Option<&ferrule_cuda::context::CudaF32Buffer>,
        indexer_kv: Option<&ferrule_cuda::context::CudaF32Buffer>,
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
    ) -> Result<ferrule_cuda::context::CudaI32Buffer> {
        self.ops.dsv4_prefill_topk_indices_from_device(
            query,
            weights,
            indexer_kv,
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
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_topk_indices_fused_index_query_from_device(
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
    ) -> Result<ferrule_cuda::context::CudaI32Buffer> {
        self.ensure_rope_tables(rope_name, rope_dim, 0.0, 0)?;
        let cos = self.rope_cos_device(rope_name);
        let sin = self.rope_sin_device(rope_name);
        self.ops
            .dsv4_prefill_topk_indices_fused_index_query_from_device(
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
            )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_topk_indices_from_device(
        &mut self,
        query: Option<&ferrule_cuda::context::CudaF32Buffer>,
        weights: Option<&ferrule_cuda::context::CudaF32Buffer>,
        indexer_kv: Option<&ferrule_cuda::context::CudaF32Buffer>,
        position: usize,
        window_len: usize,
        window_size: usize,
        extra_cols: usize,
        value_offset: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        weight_scale: f32,
    ) -> Result<ferrule_cuda::context::CudaI32Buffer> {
        self.ops.dsv4_decode_topk_indices_from_device(
            query,
            weights,
            indexer_kv,
            position,
            window_len,
            window_size,
            extra_cols,
            value_offset,
            compressed_len,
            index_heads,
            index_head_dim,
            weight_scale,
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
        self.ensure_rope_tables(rope_name, rope_dim, 0.0, 0)?;
        let indexer_kv = self.indexer_kv_cache.get(&layer).ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing indexer KV device cache"
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
    pub(crate) fn decode_topk_indices_fused_index_query_from_indexer_cache(
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
    ) -> Result<ferrule_cuda::context::CudaI32Buffer> {
        let total_cols = window_size.saturating_add(extra_cols);
        let mut out = self.ops.zero_i32_buffer(total_cols)?;
        self.decode_topk_indices_fused_index_query_from_indexer_cache_into(
            query,
            weights,
            layer,
            rope_name,
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
            &mut out,
        )?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_topk_indices_from_indexer_cache_into(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        weights: &ferrule_cuda::context::CudaF32Buffer,
        layer: usize,
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
        let indexer_kv = self.indexer_kv_cache.get(&layer).ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing indexer KV device cache"
            ))
        })?;
        self.ops.dsv4_decode_topk_indices_from_device_into(
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_topk_indices_from_indexer_cache(
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
    ) -> Result<ferrule_cuda::context::CudaI32Buffer> {
        let indexer_kv = self.indexer_kv_cache.get(&layer).ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing indexer KV device cache"
            ))
        })?;
        self.ops.dsv4_decode_topk_indices_from_device(
            Some(query),
            Some(weights),
            Some(indexer_kv),
            position,
            window_len,
            window_size,
            extra_cols,
            value_offset,
            compressed_len,
            index_heads,
            index_head_dim,
            weight_scale,
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
        layer: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        if !self.kv_cache.contains_key(&layer) {
            let buf = self.ops.zero_f32_buffer(window_size * head_dim)?;
            self.kv_cache.insert(layer, buf);
            self.kv_len.insert(layer, 0);
        }
        Ok(())
    }

    /// Append a single-token KV vector (already on device) into the slot for
    /// `position` using a device-to-device copy, then advance the cached
    /// length. `kv_buffer` must be `[head_dim]` f32.
    #[allow(dead_code)]
    pub(crate) fn kv_append_device(
        &mut self,
        layer: usize,
        kv_buffer: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        head_dim: usize,
        window_size: usize,
    ) -> Result<()> {
        self.ensure_kv_cache(layer, window_size, head_dim)?;
        if kv_buffer.len() != head_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} device KV append length mismatch: expected {head_dim}, got {}",
                kv_buffer.len()
            )));
        }
        let slot = position % window_size;
        let offset = slot * head_dim;
        let dst = self
            .kv_cache
            .get_mut(&layer)
            .expect("inserted by ensure_kv_cache");
        self.ops.copy_f32_into_slot(kv_buffer, dst, offset)?;
        let len = self.kv_len.get_mut(&layer).expect("inserted above");
        *len = window_size.min(*len + 1);
        Ok(())
    }

    pub(crate) fn kv_write_window_device(
        &mut self,
        layer: usize,
        kv_buffer: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        head_dim: usize,
        window_size: usize,
    ) -> Result<()> {
        self.ensure_kv_cache(layer, window_size, head_dim)?;
        if kv_buffer.len() != head_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} device KV write length mismatch: expected {head_dim}, got {}",
                kv_buffer.len()
            )));
        }
        let slot = position % window_size;
        let offset = slot * head_dim;
        let dst = self
            .kv_cache
            .get_mut(&layer)
            .expect("inserted by ensure_kv_cache");
        self.ops.copy_f32_into_slot(kv_buffer, dst, offset)
    }

    /// Borrow the device-resident KV buffer for `layer`.
    #[allow(dead_code)]
    pub(crate) fn kv_values_device(&self, layer: usize) -> &ferrule_cuda::context::CudaF32Buffer {
        self.kv_cache
            .get(&layer)
            .expect("kv cache must be ensured before reading")
    }

    pub(crate) fn ensure_combined_kv_cache(
        &mut self,
        layer: usize,
        cache: &DeepSeekV4AttentionCache,
        compressed_capacity: usize,
    ) -> Result<()> {
        let current_capacity = self
            .combined_kv_compressed_capacity
            .get(&layer)
            .copied()
            .unwrap_or(0);
        if self.combined_kv_cache.contains_key(&layer) && current_capacity >= compressed_capacity {
            return Ok(());
        }

        let window_values = cache.window.values_full();
        let head_dim = cache.window.head_dim;
        let capacity = compressed_capacity
            .max(cache.compressed_len())
            .max(16)
            .next_power_of_two();
        let mut values = vec![0.0f32; window_values.len() + capacity * head_dim];
        values[..window_values.len()].copy_from_slice(window_values);
        let compressed_offset = window_values.len();
        values[compressed_offset..compressed_offset + cache.compressed.len()]
            .copy_from_slice(&cache.compressed);

        let buffer = self.ops.upload_f32_buffer(&values)?;
        self.combined_kv_cache.insert(layer, buffer);
        self.combined_kv_compressed_capacity.insert(layer, capacity);
        Ok(())
    }

    pub(crate) fn combined_kv_append_window_device(
        &mut self,
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
        let dst = self.combined_kv_cache.get_mut(&layer).ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing combined KV device cache"
            ))
        })?;
        let slot = position % window_size;
        self.ops.copy_f32_into_slot(kv_dev, dst, slot * head_dim)
    }

    pub(crate) fn combined_kv_append_window_host(
        &mut self,
        layer: usize,
        kv: &[f32],
        position: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        if kv.len() != head_dim {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} combined window KV append mismatch: expected {head_dim}, got {}",
                kv.len()
            )));
        }
        let src = self.ops.upload_f32_buffer(kv)?;
        let dst = self.combined_kv_cache.get_mut(&layer).ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing combined KV device cache"
            ))
        })?;
        let slot = position % window_size;
        self.ops.copy_f32_into_slot(&src, dst, slot * head_dim)
    }

    pub(crate) fn combined_kv_write_window_rows_device(
        &mut self,
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
        let dst = self.combined_kv_cache.get_mut(&layer).ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing combined KV device cache"
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
        let capacity = self
            .combined_kv_compressed_capacity
            .get(&layer)
            .copied()
            .unwrap_or(0);
        if compressed_start + rows > capacity {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} compressed KV rows [{}..{}) exceed device capacity {capacity}",
                compressed_start,
                compressed_start + rows
            )));
        }
        let dst = self.combined_kv_cache.get_mut(&layer).ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing combined KV device cache"
            ))
        })?;
        let offset = (window_size + compressed_start) * head_dim;
        self.ops.copy_f32_into_slot(values, dst, offset)
    }

    pub(crate) fn combined_kv_append_compressed_host(
        &mut self,
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
        let capacity = self
            .combined_kv_compressed_capacity
            .get(&layer)
            .copied()
            .unwrap_or(0);
        if compressed_index >= capacity {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} compressed KV index {compressed_index} exceeds device capacity {capacity}"
            )));
        }
        let src = self.ops.upload_f32_buffer(value)?;
        let dst = self.combined_kv_cache.get_mut(&layer).ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing combined KV device cache"
            ))
        })?;
        let offset = (window_size + compressed_index) * head_dim;
        self.ops.copy_f32_into_slot(&src, dst, offset)
    }

    pub(crate) fn ensure_indexer_kv_cache(
        &mut self,
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
        let current_capacity = self.indexer_kv_capacity.get(&layer).copied().unwrap_or(0);
        if self.indexer_kv_cache.contains_key(&layer) && current_capacity >= compressed_capacity {
            return Ok(());
        }

        let capacity = compressed_capacity.max(16).next_power_of_two();
        let mut buffer = self.ops.zero_f32_buffer(capacity * index_head_dim)?;
        if let Some(old) = self.indexer_kv_cache.remove(&layer) {
            self.ops.copy_f32_into_slot(&old, &mut buffer, 0)?;
        }
        self.indexer_kv_cache.insert(layer, buffer);
        self.indexer_kv_capacity.insert(layer, capacity);
        Ok(())
    }

    pub(crate) fn indexer_kv_write_rows_device(
        &mut self,
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
        let capacity = self.indexer_kv_capacity.get(&layer).copied().unwrap_or(0);
        if compressed_start + rows > capacity {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} indexer KV rows [{}..{}) exceed device capacity {capacity}",
                compressed_start,
                compressed_start + rows
            )));
        }
        let dst = self.indexer_kv_cache.get_mut(&layer).ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing indexer KV device cache"
            ))
        })?;
        self.ops
            .copy_f32_into_slot(values, dst, compressed_start * index_head_dim)
    }

    pub(crate) fn indexer_kv_append_host(
        &mut self,
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
        let capacity = self.indexer_kv_capacity.get(&layer).copied().unwrap_or(0);
        if compressed_index >= capacity {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} indexer KV index {compressed_index} exceeds device capacity {capacity}"
            )));
        }
        let src = self.ops.upload_f32_buffer(value)?;
        let dst = self.indexer_kv_cache.get_mut(&layer).ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing indexer KV device cache"
            ))
        })?;
        self.ops
            .copy_f32_into_slot(&src, dst, compressed_index * index_head_dim)
    }

    pub(crate) fn combined_kv_values_device(
        &self,
        layer: usize,
    ) -> Result<&ferrule_cuda::context::CudaF32Buffer> {
        self.combined_kv_cache.get(&layer).ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 layer {layer} missing combined KV device cache"
            ))
        })
    }

    /// Current KV length for `layer` (capped at `window_size`).
    #[allow(dead_code)]
    pub(crate) fn kv_len_device(&self, layer: usize) -> usize {
        self.kv_len.get(&layer).copied().unwrap_or(0)
    }

    // ── Precomputed rope cos/sin tables ──────────────────────────────

    /// Ensure precomputed `[max_positions, rope_dim/2]` cos/sin tables are
    /// uploaded for `name`, using the YAARN frequency formula. For plain rope
    /// (`factor == 1.0`) this reduces to `freq = 1/theta^(2i/rope_dim)`.
    /// Idempotent.
    #[allow(dead_code)]
    pub(crate) fn ensure_rope_tables(
        &mut self,
        name: &str,
        rope_dim: usize,
        rope_theta: f32,
        max_positions: usize,
    ) -> Result<()> {
        self.ensure_rope_tables_with_params(
            name,
            rope_dim,
            DeepSeekV4RopeParams::plain(rope_theta),
            max_positions,
        )
    }

    pub(crate) fn ensure_rope_tables_with_params(
        &mut self,
        name: &str,
        rope_dim: usize,
        rope: DeepSeekV4RopeParams,
        max_positions: usize,
    ) -> Result<()> {
        if self.rope_cos.contains_key(name) {
            return Ok(());
        }
        let rd2 = rope_dim / 2;
        let mut cos = vec![0.0f32; max_positions * rd2];
        let mut sin = vec![0.0f32; max_positions * rd2];
        for position in 0..max_positions {
            for pair in 0..rd2 {
                let freq = yarn_frequency(pair, rope_dim, rope);
                let angle = position as f32 * freq;
                let (s, c) = angle.sin_cos();
                cos[position * rd2 + pair] = c;
                sin[position * rd2 + pair] = s;
            }
        }
        let cos_buf = self.ops.upload_f32_buffer(&cos)?;
        let sin_buf = self.ops.upload_f32_buffer(&sin)?;
        self.rope_cos.insert(name.to_string(), cos_buf);
        self.rope_sin.insert(name.to_string(), sin_buf);
        Ok(())
    }

    /// Borrow the cached cos table for `name`.
    #[allow(dead_code)]
    pub(crate) fn rope_cos_device(&self, name: &str) -> &ferrule_cuda::context::CudaF32Buffer {
        self.rope_cos
            .get(name)
            .expect("rope tables must be ensured before reading")
    }

    /// Borrow the cached sin table for `name`.
    #[allow(dead_code)]
    pub(crate) fn rope_sin_device(&self, name: &str) -> &ferrule_cuda::context::CudaF32Buffer {
        self.rope_sin
            .get(name)
            .expect("rope tables must be ensured before reading")
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
        position: usize,
        window_size: usize,
    ) -> Result<&ferrule_cuda::context::CudaI32Buffer> {
        self.ensure_topk_buffer(window_size)?;
        let kv_len = self.kv_len.values().copied().max().unwrap_or(0);
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

    /// Borrow the cached top-k index buffer.
    #[allow(dead_code)]
    pub(crate) fn topk_buffer_device(&self) -> &ferrule_cuda::context::CudaI32Buffer {
        self.topk_buffer
            .as_ref()
            .expect("topk buffer must be ensured before reading")
    }
}
