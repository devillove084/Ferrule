//! DeepSeek-V4 operator context: CPU/CUDA dispatch for linear, attention, MoE, HC ops.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use ferrule_common::{Error, Result};

use crate::artifact::binding::RouterArtifactPayload;
use crate::artifact::linear::ArtifactLinearPayload;
use crate::artifact::tensor::{ArtifactTensorReader, ArtifactTensorSlice};
use crate::attention_backend::{SparseAttentionSpec, sparse_attention_reference};
use crate::execution::ModelExecutionBackend;
use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::{
    HyperConnectionConfig, HyperConnectionHeadWeights, HyperConnectionPreOutput,
    HyperConnectionSplit, HyperConnectionWeights, hc_head_reference, hc_post_reference,
    hc_pre_reference,
};
use crate::moe::executor::ExpertExecutor;
use crate::moe::handle::CpuExpertHandleStore;
use crate::moe::prediction::{ExpertAccessPhase, ExpertBatchAccessEvent, ExpertPredictionStats};
use crate::moe::routed::{
    RoutedMoeStepOutput, execute_routed_moe_with_artifact_router_reference_with_handles,
};
use crate::moe::routing::ExpertRouterPolicy;
use crate::moe::streaming::{ExpertStreamingPlanner, ExpertStreamingReader};
use crate::runner::TokenLogit;

use super::config::DeepSeekV4AttentionConfig;
#[cfg(feature = "cuda")]
use super::cuda_cache::DeepSeekV4CudaOperatorCache;
use super::helpers::{grouped_output_a, rank_logits_desc, rms_norm, rms_norm_heads_in_place};
use super::prepared::DeepSeekV4ExecutionPolicy;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4LayerProfileStats {
    pub layer: usize,

    pub state_init_calls: u64,
    pub state_init_us: u64,
    pub decode_calls: u64,
    pub decode_total_us: u64,
    pub prefill_calls: u64,
    pub prefill_tokens: u64,
    pub prefill_total_us: u64,
    pub attn_hc_pre_us: u64,
    pub attn_norm_us: u64,
    pub attention_us: u64,
    pub attn_hc_post_us: u64,
    pub ffn_hc_pre_us: u64,
    pub ffn_norm_us: u64,
    pub moe_us: u64,
    pub ffn_hc_post_us: u64,
}

impl DeepSeekV4LayerProfileStats {
    fn new(layer: usize) -> Self {
        Self {
            layer,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4AttentionProfileStats {
    pub layer: usize,
    pub calls: u64,
    pub tokens: u64,
    pub q_a_us: u64,
    pub q_norm_us: u64,
    pub q_b_us: u64,
    pub q_head_norm_us: u64,
    pub q_rope_us: u64,
    pub kv_proj_us: u64,
    pub kv_norm_us: u64,
    pub kv_rope_quant_us: u64,
    pub kv_cache_append_us: u64,
    pub indexer_compress_us: u64,
    pub main_compress_us: u64,
    pub compressed_kv_upload_us: u64,
    pub topk_build_us: u64,
    pub sparse_attention_us: u64,
    pub context_rope_us: u64,
    pub output_a_us: u64,
    pub output_b_us: u64,
}

impl DeepSeekV4AttentionProfileStats {
    fn new(layer: usize) -> Self {
        Self {
            layer,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeepSeekV4LayerProfileStage {
    AttnHcPre,
    AttnNorm,
    Attention,
    AttnHcPost,
    FfnHcPre,
    FfnNorm,
    Moe,
    FfnHcPost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeepSeekV4AttentionProfileStage {
    Qa,
    QNorm,
    Qb,
    QHeadNorm,
    QRope,
    KvProj,
    KvNorm,
    KvRopeQuant,
    KvCacheAppend,
    IndexerCompress,
    MainCompress,
    #[cfg(feature = "cuda")]
    CompressedKvUpload,
    TopkBuild,
    SparseAttention,
    ContextRope,
    OutputA,
    OutputB,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4OperatorRuntimeCounters {
    pub kernel_launches: u64,
    pub host_to_device_copies: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_copies: u64,
    pub device_to_host_bytes: u64,
    pub artifact_uploads: u64,
    pub artifact_upload_bytes: u64,
    pub device_allocation_attempts: u64,
    pub device_allocations: u64,
    pub device_allocation_failures: u64,
    pub device_allocation_bytes: u64,
    pub stream_wide_syncs: u64,
    pub stream_wide_sync_failures: u64,
    pub moe_calls: u64,
    pub moe_tc_calls: u64,
    pub moe_scalar_calls: u64,
    pub moe_reduce_calls: u64,
    pub moe_total_us: u64,
    pub moe_pointer_upload_us: u64,
    pub moe_input_prepare_us: u64,
    pub moe_gate_up_us: u64,
    pub moe_swiglu_us: u64,
    pub moe_hidden_pack_us: u64,
    pub moe_down_us: u64,
    pub moe_router_us: u64,
    pub moe_routing_us: u64,
    pub moe_plan_us: u64,
    pub moe_predicted_experts: u64,
    pub moe_prefetch_loads: u64,
    pub moe_prefetch_enqueued: u64,
    pub moe_prefetch_skipped_cached_or_inflight: u64,
    pub moe_prefetch_resident: u64,
    pub moe_prefetch_materializing: u64,
    pub moe_prefetch_host_staged: u64,
    pub moe_prefetch_in_flight: u64,
    pub moe_prefetch_cold: u64,
    pub expert_selected_resident_hits: u64,
    pub expert_selected_upload_hits: u64,
    pub expert_selected_host_staged_hits: u64,
    pub expert_selected_host_staging_waits: u64,
    pub expert_selected_host_staging_hits: u64,
    pub expert_selected_host_staging_wait_us: u64,
    pub expert_selected_cold_misses: u64,
    pub expert_upload_prefetch_submitted: u64,
    pub expert_upload_prefetch_completed: u64,
    pub expert_upload_prefetch_in_flight: usize,
    pub expert_selected_upload_waits: u64,
    pub expert_selected_upload_wait_us: u64,
    pub expert_async_upload_bytes: u64,
    pub expert_lookahead_prefetch_calls: u64,
    pub expert_lookahead_prefetch_experts: u64,
    pub expert_lookahead_prefetch_enqueued: u64,
    pub expert_lookahead_prefetch_us: u64,
    pub expert_planner_residency_syncs: u64,
    pub expert_planner_residency_synced: u64,
    pub moe_cache_lookup_us: u64,
    pub moe_expert_read_us: u64,
    pub moe_expert_upload_us: u64,
    pub moe_shared_us: u64,
    pub moe_workspace_us: u64,
    pub moe_compute_submit_us: u64,
    pub moe_commit_us: u64,
    pub output_head_calls: u64,
    pub output_head_chunks: u64,
    pub output_head_rows: u64,
    pub output_head_cache_hits: u64,
    pub output_head_cache_misses: u64,
    pub output_head_hidden_uploads: u64,
    pub output_head_hidden_upload_us: u64,
    pub output_head_read_us: u64,
    pub output_head_upload_us: u64,
    pub output_head_topk_us: u64,
    pub output_head_merge_us: u64,
    pub expert_selected: u64,
    pub expert_selected_load_requests: u64,
    pub expert_loads: u64,
    pub expert_load_bytes: u64,
    pub expert_evictions: u64,
    pub expert_host_cache_hits: u64,
    pub expert_host_cache_misses: u64,
    pub expert_host_cache_evictions: u64,
    pub expert_host_cache_entries: usize,
    pub expert_host_cache_bytes: u64,
    pub expert_pinned_cache_hits: u64,
    pub expert_pinned_cache_misses: u64,
    pub expert_pinned_cache_evictions: u64,
    pub expert_pinned_cache_entries: usize,
    pub expert_pinned_cache_bytes: u64,
    pub expert_cuda_resident_entries: usize,
    pub expert_cuda_resident_bytes: u64,
    pub expert_async_prefetch_submitted: u64,
    pub expert_async_prefetch_completed: u64,
    pub expert_async_prefetch_failed: u64,
    pub expert_async_prefetch_skipped: u64,
    pub expert_async_prefetch_in_flight: usize,
    pub arena_hits: u64,
    pub arena_misses: u64,
    pub arena_grows: u64,
    pub arena_reuses: u64,
    pub expert_predictor_stats: ExpertPredictionStats,
}

fn duration_us(d: Duration) -> u64 {
    d.as_micros().min(u128::from(u64::MAX)) as u64
}

pub struct DeepSeekV4OperatorContext {
    pub(crate) backend: ModelExecutionBackend,
    layer_profiles: BTreeMap<usize, DeepSeekV4LayerProfileStats>,
    attention_profiles: BTreeMap<usize, DeepSeekV4AttentionProfileStats>,
    /// When enabled, profile stage timings synchronize the CUDA stream before
    /// sampling elapsed wall time. This is expensive but gives attribution that
    /// includes queued GPU work instead of only host enqueue time.
    profile_sync: bool,
    #[cfg(feature = "cuda")]
    fused_indexer_prefill_topk: bool,
    #[cfg(feature = "cuda")]
    fused_indexer_decode_topk: bool,
    #[cfg(feature = "cuda")]
    parity_checkpoint_selection: Option<(usize, String)>,
    parity_checkpoints: BTreeMap<String, Vec<f32>>,
    moe_access_events: Vec<ExpertBatchAccessEvent>,
    #[cfg(feature = "cuda")]
    pub(crate) cuda: Option<DeepSeekV4CudaOperatorCache>,
}

impl DeepSeekV4OperatorContext {
    pub fn new(backend: ModelExecutionBackend, policy: &DeepSeekV4ExecutionPolicy) -> Result<Self> {
        Ok(Self {
            backend,
            layer_profiles: BTreeMap::new(),
            attention_profiles: BTreeMap::new(),
            profile_sync: policy.profile_sync(),
            #[cfg(feature = "cuda")]
            fused_indexer_prefill_topk: policy.fused_indexer_prefill_topk(),
            #[cfg(feature = "cuda")]
            fused_indexer_decode_topk: policy.fused_indexer_decode_topk(),
            #[cfg(feature = "cuda")]
            parity_checkpoint_selection: policy.parity_checkpoint_selection(),
            parity_checkpoints: BTreeMap::new(),
            moe_access_events: Vec::new(),
            #[cfg(feature = "cuda")]
            cuda: match backend {
                ModelExecutionBackend::Cpu => None,
                ModelExecutionBackend::Cuda => Some(DeepSeekV4CudaOperatorCache::new(policy)?),
            },
        })
    }

    pub fn new_cpu() -> Result<Self> {
        Self::new(
            ModelExecutionBackend::Cpu,
            &DeepSeekV4ExecutionPolicy::default(),
        )
    }

    pub fn backend(&self) -> ModelExecutionBackend {
        self.backend
    }

    pub(crate) fn take_parity_checkpoints(&mut self) -> BTreeMap<String, Vec<f32>> {
        std::mem::take(&mut self.parity_checkpoints)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn capture_parity_checkpoint_host(
        &mut self,
        layer: usize,
        stage: &str,
        values: &[f32],
    ) {
        if self.parity_checkpoint_selected(layer, stage) {
            self.parity_checkpoints
                .insert(stage.to_string(), values.to_vec());
        }
    }

    #[cfg(feature = "cuda")]
    fn parity_checkpoint_selected(&self, layer: usize, stage: &str) -> bool {
        let Some((selected_layer, selected_stage)) = &self.parity_checkpoint_selection else {
            return false;
        };
        *selected_layer == layer
            && (selected_stage == "all" || selected_stage == "*" || selected_stage == stage)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn capture_parity_checkpoint_last_row(
        &mut self,
        layer: usize,
        stage: &str,
        values: &ferrule_cuda::context::CudaF32Buffer,
        row_width: usize,
    ) -> Result<()> {
        if !self.parity_checkpoint_selected(layer, stage) {
            return Ok(());
        }
        if row_width == 0 || values.len() == 0 || !values.len().is_multiple_of(row_width) {
            return Err(Error::Model(format!(
                "DeepSeek-V4 parity checkpoint {stage} has invalid row shape: len={} row_width={row_width}",
                values.len()
            )));
        }
        let host = self.cuda_mut()?.ops.download_f32_buffer(values)?;
        let start = host.len() - row_width;
        self.parity_checkpoints
            .insert(stage.to_string(), host[start..].to_vec());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn capture_parity_checkpoint_rows(
        &mut self,
        layer: usize,
        stage: &str,
        values: &ferrule_cuda::context::CudaF32Buffer,
        row_width: usize,
    ) -> Result<()> {
        if !self.parity_checkpoint_selected(layer, stage) {
            return Ok(());
        }
        if row_width == 0 || values.len() == 0 || !values.len().is_multiple_of(row_width) {
            return Err(Error::Model(format!(
                "DeepSeek-V4 parity checkpoint {stage} has invalid row shape: len={} row_width={row_width}",
                values.len()
            )));
        }
        let host = self.cuda_mut()?.ops.download_f32_buffer(values)?;
        self.parity_checkpoints
            .entry(stage.to_string())
            .or_default()
            .extend(host);
        Ok(())
    }

    pub fn runtime_counters(&self) -> DeepSeekV4OperatorRuntimeCounters {
        match self.backend {
            ModelExecutionBackend::Cpu => DeepSeekV4OperatorRuntimeCounters::default(),
            ModelExecutionBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    self.cuda
                        .as_ref()
                        .map(DeepSeekV4CudaOperatorCache::runtime_counters)
                        .unwrap_or_default()
                }
                #[cfg(not(feature = "cuda"))]
                {
                    DeepSeekV4OperatorRuntimeCounters::default()
                }
            }
        }
    }

    pub fn layer_profile_stats(&self) -> Vec<DeepSeekV4LayerProfileStats> {
        self.layer_profiles.values().copied().collect()
    }

    pub fn attention_profile_stats(&self) -> Vec<DeepSeekV4AttentionProfileStats> {
        self.attention_profiles.values().copied().collect()
    }

    pub fn profile_sync_enabled(&self) -> bool {
        self.profile_sync
    }

    #[cfg(feature = "cuda")]
    pub(crate) const fn fused_indexer_prefill_topk_enabled(&self) -> bool {
        self.fused_indexer_prefill_topk
    }

    #[cfg(feature = "cuda")]
    pub(crate) const fn fused_indexer_decode_topk_enabled(&self) -> bool {
        self.fused_indexer_decode_topk
    }

    pub(crate) fn record_moe_access_event(&mut self, event: ExpertBatchAccessEvent) {
        self.moe_access_events.push(event);
    }

    pub(crate) fn drain_moe_access_events(&mut self) -> Vec<ExpertBatchAccessEvent> {
        #[cfg(feature = "cuda")]
        {
            let mut events = std::mem::take(&mut self.moe_access_events);
            if let Some(cuda) = self.cuda.as_mut() {
                events.extend(cuda.drain_moe_access_events());
            }
            events
        }
        #[cfg(not(feature = "cuda"))]
        {
            std::mem::take(&mut self.moe_access_events)
        }
    }

    pub(crate) fn finish_profile_stage(&mut self, start: Instant) -> Result<u64> {
        self.sync_profile_stream()?;
        Ok(duration_us(start.elapsed()))
    }

    pub(crate) fn sync_profile_stream(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.profile_sync && self.backend == ModelExecutionBackend::Cuda {
            self.cuda_mut()?.ops.sync_stream()?;
        }
        Ok(())
    }

    pub(crate) fn record_layer_state_init(&mut self, layer: usize, elapsed_us: u64) {
        let stats = self.layer_profile_entry(layer);
        stats.state_init_calls = stats.state_init_calls.saturating_add(1);
        stats.state_init_us = stats.state_init_us.saturating_add(elapsed_us);
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn record_layer_decode(&mut self, layer: usize, elapsed_us: u64) {
        let stats = self.layer_profile_entry(layer);
        stats.decode_calls = stats.decode_calls.saturating_add(1);
        stats.decode_total_us = stats.decode_total_us.saturating_add(elapsed_us);
    }

    pub(crate) fn record_layer_prefill(&mut self, layer: usize, tokens: usize, elapsed_us: u64) {
        let stats = self.layer_profile_entry(layer);
        stats.prefill_calls = stats.prefill_calls.saturating_add(1);
        stats.prefill_tokens = stats.prefill_tokens.saturating_add(tokens as u64);
        stats.prefill_total_us = stats.prefill_total_us.saturating_add(elapsed_us);
    }

    pub(crate) fn record_layer_stage(
        &mut self,
        layer: usize,
        stage: DeepSeekV4LayerProfileStage,
        elapsed_us: u64,
    ) {
        let stats = self.layer_profile_entry(layer);
        match stage {
            DeepSeekV4LayerProfileStage::AttnHcPre => {
                stats.attn_hc_pre_us = stats.attn_hc_pre_us.saturating_add(elapsed_us)
            }
            DeepSeekV4LayerProfileStage::AttnNorm => {
                stats.attn_norm_us = stats.attn_norm_us.saturating_add(elapsed_us)
            }
            DeepSeekV4LayerProfileStage::Attention => {
                stats.attention_us = stats.attention_us.saturating_add(elapsed_us)
            }
            DeepSeekV4LayerProfileStage::AttnHcPost => {
                stats.attn_hc_post_us = stats.attn_hc_post_us.saturating_add(elapsed_us)
            }
            DeepSeekV4LayerProfileStage::FfnHcPre => {
                stats.ffn_hc_pre_us = stats.ffn_hc_pre_us.saturating_add(elapsed_us)
            }
            DeepSeekV4LayerProfileStage::FfnNorm => {
                stats.ffn_norm_us = stats.ffn_norm_us.saturating_add(elapsed_us)
            }
            DeepSeekV4LayerProfileStage::Moe => {
                stats.moe_us = stats.moe_us.saturating_add(elapsed_us)
            }
            DeepSeekV4LayerProfileStage::FfnHcPost => {
                stats.ffn_hc_post_us = stats.ffn_hc_post_us.saturating_add(elapsed_us)
            }
        }
    }

    pub(crate) fn record_attention_call(&mut self, layer: usize, tokens: usize) {
        let stats = self.attention_profile_entry(layer);
        stats.calls = stats.calls.saturating_add(1);
        stats.tokens = stats.tokens.saturating_add(tokens as u64);
    }

    pub(crate) fn record_attention_stage(
        &mut self,
        layer: usize,
        stage: DeepSeekV4AttentionProfileStage,
        elapsed_us: u64,
    ) {
        let stats = self.attention_profile_entry(layer);
        match stage {
            DeepSeekV4AttentionProfileStage::Qa => {
                stats.q_a_us = stats.q_a_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::QNorm => {
                stats.q_norm_us = stats.q_norm_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::Qb => {
                stats.q_b_us = stats.q_b_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::QHeadNorm => {
                stats.q_head_norm_us = stats.q_head_norm_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::QRope => {
                stats.q_rope_us = stats.q_rope_us.saturating_add(elapsed_us)
            }

            DeepSeekV4AttentionProfileStage::KvProj => {
                stats.kv_proj_us = stats.kv_proj_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::KvNorm => {
                stats.kv_norm_us = stats.kv_norm_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::KvRopeQuant => {
                stats.kv_rope_quant_us = stats.kv_rope_quant_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::KvCacheAppend => {
                stats.kv_cache_append_us = stats.kv_cache_append_us.saturating_add(elapsed_us)
            }

            DeepSeekV4AttentionProfileStage::IndexerCompress => {
                stats.indexer_compress_us = stats.indexer_compress_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::MainCompress => {
                stats.main_compress_us = stats.main_compress_us.saturating_add(elapsed_us)
            }
            #[cfg(feature = "cuda")]
            DeepSeekV4AttentionProfileStage::CompressedKvUpload => {
                stats.compressed_kv_upload_us =
                    stats.compressed_kv_upload_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::TopkBuild => {
                stats.topk_build_us = stats.topk_build_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::SparseAttention => {
                stats.sparse_attention_us = stats.sparse_attention_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::ContextRope => {
                stats.context_rope_us = stats.context_rope_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::OutputA => {
                stats.output_a_us = stats.output_a_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::OutputB => {
                stats.output_b_us = stats.output_b_us.saturating_add(elapsed_us)
            }
        }
    }

    fn layer_profile_entry(&mut self, layer: usize) -> &mut DeepSeekV4LayerProfileStats {
        self.layer_profiles
            .entry(layer)
            .or_insert_with(|| DeepSeekV4LayerProfileStats::new(layer))
    }

    fn attention_profile_entry(&mut self, layer: usize) -> &mut DeepSeekV4AttentionProfileStats {
        self.attention_profiles
            .entry(layer)
            .or_insert_with(|| DeepSeekV4AttentionProfileStats::new(layer))
    }

    pub(crate) fn linear_matvec(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        linear.reference_matvec(input)
    }

    pub(crate) fn linear_rows(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        let in_features = linear.format.in_features();
        if rows == 0 || input.len() != rows * in_features {
            return Err(Error::Model(format!(
                "artifact linear {:?} rows input length mismatch: rows={} expected {}, got {}",
                linear.role,
                rows,
                rows * in_features,
                input.len()
            )));
        }
        let mut output = Vec::with_capacity(rows * linear.format.out_features());
        for row in 0..rows {
            let start = row * in_features;
            output.extend_from_slice(&linear.reference_matvec(&input[start..start + in_features])?);
        }
        Ok(output)
    }

    pub(crate) fn linear_topk(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &[f32],
        top_k: usize,
    ) -> Result<Vec<TokenLogit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let logits = linear.reference_matvec(input)?;
        let mut top = logits
            .into_iter()
            .enumerate()
            .map(|(token_id, logit)| TokenLogit {
                token_id: token_id as u32,
                logit,
            })
            .collect::<Vec<_>>();
        top.sort_by(rank_logits_desc);
        top.truncate(top_k);
        Ok(top)
    }

    pub(crate) fn output_head_topk_chunks(
        &mut self,
        slice: &ArtifactTensorSlice,
        hidden: &[f32],
        top_k: usize,
        chunk_rows: usize,
        reader: &ArtifactTensorReader,
    ) -> Result<Vec<TokenLogit>> {
        #[cfg(not(feature = "cuda"))]
        let _ = (slice, hidden, top_k, chunk_rows, reader);
        match self.backend {
            ModelExecutionBackend::Cpu => Err(Error::Internal(
                "CPU output-head chunk top-k should use the reference row-read loop".into(),
            )),
            ModelExecutionBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    self.cuda_mut()?
                        .output_head_topk_chunks(slice, hidden, top_k, chunk_rows, reader)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(Error::Model(
                        "CUDA model execution requires the ferrule-model/cuda feature".into(),
                    ))
                }
            }
        }
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
        sparse_attention_reference(query, values, topk, Some(sink), tokens, kv_len, spec)
    }

    pub(crate) fn grouped_output_a(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &[f32],
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
    ) -> Result<Vec<f32>> {
        grouped_output_a(output_a, context, cfg, layer)
    }

    pub(crate) fn rms_norm(
        &mut self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        label: &str,
    ) -> Result<Vec<f32>> {
        rms_norm(input, weight, eps, label)
    }

    pub(crate) fn rms_norm_rows(
        &mut self,
        input: &[f32],
        rows: usize,
        weight: &[f32],
        eps: f32,
        label: &str,
    ) -> Result<Vec<f32>> {
        if rows == 0 || weight.is_empty() || input.len() != rows * weight.len() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 {label} batched RMS length mismatch: rows={rows} input={} weight={}",
                input.len(),
                weight.len()
            )));
        }
        let mut out = Vec::with_capacity(input.len());
        for row in 0..rows {
            let start = row * weight.len();
            let normalized = rms_norm(&input[start..start + weight.len()], weight, eps, label)?;
            out.extend_from_slice(&normalized);
        }
        Ok(out)
    }

    pub(crate) fn rms_norm_heads_in_place(
        &mut self,
        values: &mut [f32],
        heads: usize,
        head_dim: usize,
        eps: f32,
        layer: usize,
    ) -> Result<()> {
        rms_norm_heads_in_place(values, heads, head_dim, eps, layer)
    }

    pub(crate) fn hc_pre(
        &mut self,
        state: &[f32],
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionWeights,
    ) -> Result<HyperConnectionPreOutput> {
        hc_pre_reference(state, tokens, config, weights)
    }

    pub(crate) fn hc_post(
        &mut self,
        hidden: &[f32],
        residual: &[f32],
        config: HyperConnectionConfig,
        split: &HyperConnectionSplit,
    ) -> Result<Vec<f32>> {
        hc_post_reference(hidden, residual, config, split)
    }

    pub(crate) fn hc_head(
        &mut self,
        state: &[f32],
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionHeadWeights,
    ) -> Result<Vec<f32>> {
        hc_head_reference(state, tokens, config, weights)
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
        expert_executor: &impl ExpertExecutor,
        shared_expert: Option<&SwiGluFfnPayload>,
    ) -> Result<RoutedMoeStepOutput> {
        execute_routed_moe_with_artifact_router_reference_with_handles(
            layer,
            input,
            token_id,
            router,
            predicted_experts,
            router_policy,
            planner,
            reader,
            handles,
            expert_executor,
            shared_expert,
        )
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
        expert_executor: &impl ExpertExecutor,
        shared_expert: Option<&SwiGluFfnPayload>,
    ) -> Result<Vec<f32>> {
        let hidden = router.weight.format.in_features();
        if input.len() != token_ids.len() * hidden {
            return Err(Error::Model(format!(
                "DeepSeek-V4 CPU MoE prefill input length mismatch: input={} expected {}x{}",
                input.len(),
                token_ids.len(),
                hidden
            )));
        }
        let mut output = Vec::with_capacity(input.len());
        let mut routes_by_token = Vec::with_capacity(token_ids.len());
        let mut streaming_steps = Vec::with_capacity(token_ids.len());
        for (token_idx, &token_id) in token_ids.iter().enumerate() {
            let row = &input[token_idx * hidden..(token_idx + 1) * hidden];
            let moe = execute_routed_moe_with_artifact_router_reference_with_handles(
                layer,
                row,
                token_id,
                router,
                predicted_experts,
                router_policy,
                planner,
                reader,
                handles,
                expert_executor,
                shared_expert,
            )?;
            routes_by_token.push(moe.routes.clone());
            streaming_steps.push(moe.streaming.clone());
            output.extend_from_slice(&moe.output);
        }
        self.record_moe_access_event(ExpertBatchAccessEvent::from_routes_by_token(
            layer,
            ExpertAccessPhase::Prefill,
            token_ids.len(),
            &routes_by_token,
            &streaming_steps,
        ));
        Ok(output)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn routed_moe_prefill_batch_from_device_into(
        &mut self,
        layer: usize,
        input: &ferrule_cuda::context::CudaF32Buffer,
        token_ids: &[u32],
        row_to_sequence: Option<&[usize]>,
        router: &RouterArtifactPayload,
        predicted_experts: &[usize],
        router_policy: &ExpertRouterPolicy,
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
        shared_expert: &SwiGluFfnPayload,
        router_logits: &mut ferrule_cuda::context::CudaF32Buffer,
        router_indices: &mut ferrule_cuda::context::CudaF32Buffer,
        router_weights: &mut ferrule_cuda::context::CudaF32Buffer,
        linear_workspace: &mut ferrule_cuda::context::CudaArtifactLinearWorkspace,
        shared_workspace: &mut ferrule_cuda::context::CudaSwiGLUWorkspace,
        segment_workspace: &mut Option<ferrule_cuda::context::CudaMoeSegmentWorkspace>,
        route_output: &mut ferrule_cuda::context::CudaF32Buffer,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        if self.backend != ModelExecutionBackend::Cuda {
            return Err(Error::Model(
                "DeepSeek-V4 device-resident MoE prefill requires CUDA backend".into(),
            ));
        }
        self.cuda_mut()?.routed_moe_prefill_batch_from_device_into(
            layer,
            input,
            token_ids,
            row_to_sequence,
            router,
            predicted_experts,
            router_policy,
            planner,
            reader,
            handles,
            shared_expert,
            router_logits,
            router_indices,
            router_weights,
            linear_workspace,
            shared_workspace,
            segment_workspace,
            route_output,
            output,
        )?;
        if let Some(cuda) = self.cuda.as_mut() {
            self.moe_access_events
                .extend(cuda.drain_moe_access_events());
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn prefetch_predicted_experts(
        &mut self,
        layer: usize,
        predicted_experts: &[usize],
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
    ) -> Result<usize> {
        if self.backend != ModelExecutionBackend::Cuda {
            return Ok(0);
        }
        self.cuda_mut()?.prefetch_predicted_experts(
            layer,
            predicted_experts,
            planner,
            reader,
            handles,
        )
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn prefetch_predicted_experts(
        &mut self,
        _layer: usize,
        _predicted_experts: &[usize],
        _planner: &mut ExpertStreamingPlanner,
        _reader: &ExpertStreamingReader,
        _handles: &mut CpuExpertHandleStore,
    ) -> Result<usize> {
        Ok(0)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn prewarm_experts(
        &mut self,
        layer: usize,
        experts: &[usize],
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
    ) -> Result<usize> {
        if self.backend != ModelExecutionBackend::Cuda {
            return Ok(0);
        }
        self.cuda_mut()?
            .prewarm_experts(layer, experts, planner, reader, handles)
    }

    pub(crate) fn shutdown(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.backend == ModelExecutionBackend::Cuda {
            self.cuda_mut()?.shutdown()?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn fail_compressor_transition_if_armed(&self, indexer: bool) -> Result<()> {
        if self.backend != ModelExecutionBackend::Cuda {
            return Ok(());
        }
        let cuda = self.cuda.as_ref().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA operator cache is not initialized".into())
        })?;
        let failed = if indexer {
            cuda.ops.failpoints().check_indexer_compressor_transition()
        } else {
            cuda.ops.failpoints().check_main_compressor_transition()
        };
        if failed {
            let transition = if indexer { "indexer" } else { "main" };
            return Err(Error::Internal(format!(
                "deterministic failpoint: DeepSeek-V4 {transition} compressor transition"
            )));
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn check_cuda_arena_acquire(&self) -> Result<()> {
        if self.backend != ModelExecutionBackend::Cuda {
            return Ok(());
        }
        if self.cuda_failpoints()?.check_arena_acquire() {
            return Err(Error::Internal(
                "deterministic failpoint: DeepSeek-V4 arena acquire".into(),
            ));
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn record_cuda_arena_pool_hit(&self) -> Result<()> {
        let cuda = self.cuda.as_ref().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA operator cache is not initialized".into())
        })?;
        cuda.ops.add_arena_hit();
        cuda.ops.add_arena_reuse();
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn record_cuda_arena_pool_miss(&self, installed: bool) -> Result<()> {
        let cuda = self.cuda.as_ref().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA operator cache is not initialized".into())
        })?;
        cuda.ops.add_arena_miss();
        if installed {
            // The CUDA counter surface predates pool installs. A successful new
            // exact bucket grows the persistent arena footprint, so map install
            // to its existing grow counter.
            cuda.ops.add_arena_grow();
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_failpoints(&self) -> Result<&ferrule_cuda::CudaFailpoints> {
        self.cuda
            .as_ref()
            .map(|cuda| cuda.ops.failpoints())
            .ok_or_else(|| {
                Error::Model("DeepSeek-V4 CUDA operator cache is not initialized".into())
            })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_mut(&mut self) -> Result<&mut DeepSeekV4CudaOperatorCache> {
        self.cuda.as_mut().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA operator cache is not initialized".into())
        })
    }
}
