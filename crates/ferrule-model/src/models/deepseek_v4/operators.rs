//! DeepSeek-V4 operator context: CPU/CUDA dispatch for linear, attention, MoE, HC ops.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use ferrule_common::{Error, Result};

use crate::artifact::binding::RouterArtifactPayload;
use crate::artifact::linear::ArtifactLinearPayload;
use crate::artifact::tensor::{ArtifactTensorReader, ArtifactTensorSlice};
use crate::attention_backend::{sparse_attention_reference, SparseAttentionSpec};
use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::{
    hc_head_reference, hc_post_reference, hc_pre_reference, HyperConnectionConfig,
    HyperConnectionHeadWeights, HyperConnectionPreOutput, HyperConnectionSplit,
    HyperConnectionWeights,
};
use crate::moe::executor::ExpertExecutor;
use crate::moe::handle::CpuExpertHandleStore;
use crate::moe::prediction::{ExpertAccessPhase, ExpertBatchAccessEvent, ExpertPredictionStats};
use crate::moe::routed::{
    execute_routed_moe_with_artifact_router_reference_with_handles, RoutedMoeStepOutput,
};
use crate::moe::routing::ExpertRouterPolicy;
use crate::moe::streaming::{ExpertStreamingPlanner, ExpertStreamingReader};

#[cfg(feature = "cuda")]
use super::attention::DeepSeekV4AttentionCache;
use super::config::DeepSeekV4AttentionConfig;
#[cfg(feature = "cuda")]
use super::cuda_cache::DeepSeekV4CudaOperatorCache;
use super::helpers::{grouped_output_a, rank_logits_desc, rms_norm, rms_norm_heads_in_place};

#[cfg(feature = "cuda")]
pub(crate) struct CudaRoutedMoeStepOutput {
    pub(crate) moe: RoutedMoeStepOutput,
    pub(crate) output_dev: ferrule_cuda::context::CudaF32Buffer,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepSeekV4Logit {
    pub token_id: u32,
    pub logit: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4LayerProfileStats {
    pub layer: usize,
    pub bind_calls: u64,
    pub bind_us: u64,
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
    pub q_latent_download_us: u64,
    pub kv_proj_us: u64,
    pub kv_norm_us: u64,
    pub kv_rope_quant_us: u64,
    pub kv_cache_append_us: u64,
    pub hidden_download_us: u64,
    pub indexer_compress_us: u64,
    pub main_compress_us: u64,
    pub compressed_kv_upload_us: u64,
    pub topk_build_us: u64,
    pub sparse_attention_us: u64,
    pub context_rope_us: u64,
    pub output_a_us: u64,
    pub output_b_us: u64,
    pub output_download_us: u64,
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

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeepSeekV4AttentionProfileStage {
    Qa,
    QNorm,
    Qb,
    QHeadNorm,
    QRope,
    QLatentDownload,
    KvProj,
    KvNorm,
    KvRopeQuant,
    KvCacheAppend,
    HiddenDownload,
    IndexerCompress,
    MainCompress,
    CompressedKvUpload,
    TopkBuild,
    SparseAttention,
    ContextRope,
    OutputA,
    OutputB,
    OutputDownload,
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
    pub expert_selected_async_upload_submitted: u64,
    pub expert_selected_async_upload_completed: u64,
    pub expert_selected_async_upload_bytes: u64,
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
    pub cuda_graph_capture_attempts: u64,
    pub cuda_graph_capture_successes: u64,
    pub cuda_graph_capture_failures: u64,
    pub cuda_graph_capture_us: u64,
    pub cuda_graph_capture_warmup_us: u64,
    pub cuda_graph_capture_prepare_us: u64,
    pub cuda_graph_capture_record_us: u64,
    pub cuda_graph_full_capture_failures: u64,
    pub cuda_graph_captured_segments: u64,
    pub cuda_graph_one_shot_retires: u64,
    pub cuda_graph_replays: u64,
    pub cuda_graph_replay_us: u64,
    pub cuda_graph_replay_fallbacks: u64,
    pub expert_predictor_stats: ExpertPredictionStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DeepSeekV4OperatorBackend {
    /// CPU/reference execution for every operator. This is the correctness anchor.
    #[default]
    Cpu,
    /// CUDA execution for model operators. CPU remains only the explicit reference
    /// backend; unsupported CUDA formats fail instead of silently falling back.
    Cuda,
}

impl DeepSeekV4OperatorBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu-reference",
            Self::Cuda => "cuda",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "cpu" | "cpu-reference" | "reference" => Ok(Self::Cpu),
            // Keep the old spelling as a temporary CLI compatibility alias; it now
            // means the same strict CUDA backend, not a CPU-fallback hybrid path.
            "cuda" | "cuda-hybrid" | "gpu" => Ok(Self::Cuda),
            other => Err(Error::Model(format!(
                "unknown DeepSeek-V4 operator backend '{other}' (expected cpu or cuda)"
            ))),
        }
    }
}

fn dsv4_profile_sync_enabled() -> bool {
    std::env::var("FERRULE_DSV4_PROFILE_SYNC")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value.is_empty() || value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(false)
}

fn duration_us(d: Duration) -> u64 {
    d.as_micros().min(u128::from(u64::MAX)) as u64
}

pub struct DeepSeekV4OperatorContext {
    pub(crate) backend: DeepSeekV4OperatorBackend,
    layer_profiles: BTreeMap<usize, DeepSeekV4LayerProfileStats>,
    attention_profiles: BTreeMap<usize, DeepSeekV4AttentionProfileStats>,
    /// When enabled, profile stage timings synchronize the CUDA stream before
    /// sampling elapsed wall time. This is expensive but gives attribution that
    /// includes queued GPU work instead of only host enqueue time.
    profile_sync: bool,
    moe_access_events: Vec<ExpertBatchAccessEvent>,
    cuda_graph_capture_attempts: u64,
    cuda_graph_capture_successes: u64,
    cuda_graph_capture_failures: u64,
    cuda_graph_capture_us: u64,
    cuda_graph_capture_warmup_us: u64,
    cuda_graph_capture_prepare_us: u64,
    cuda_graph_capture_record_us: u64,
    cuda_graph_full_capture_failures: u64,
    cuda_graph_captured_segments: u64,
    cuda_graph_one_shot_retires: u64,
    cuda_graph_replays: u64,
    cuda_graph_replay_us: u64,
    cuda_graph_replay_fallbacks: u64,
    #[cfg(feature = "cuda")]
    pub(crate) cuda: Option<DeepSeekV4CudaOperatorCache>,
}

#[allow(dead_code)]
impl DeepSeekV4OperatorContext {
    pub fn new(backend: DeepSeekV4OperatorBackend) -> Result<Self> {
        Ok(Self {
            backend,
            layer_profiles: BTreeMap::new(),
            attention_profiles: BTreeMap::new(),
            profile_sync: dsv4_profile_sync_enabled(),
            moe_access_events: Vec::new(),
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
            #[cfg(feature = "cuda")]
            cuda: match backend {
                DeepSeekV4OperatorBackend::Cpu => None,
                DeepSeekV4OperatorBackend::Cuda => Some(DeepSeekV4CudaOperatorCache::new()?),
            },
        })
    }

    pub fn backend(&self) -> DeepSeekV4OperatorBackend {
        self.backend
    }

    pub fn runtime_counters(&self) -> DeepSeekV4OperatorRuntimeCounters {
        let mut counters = match self.backend {
            DeepSeekV4OperatorBackend::Cpu => DeepSeekV4OperatorRuntimeCounters::default(),
            DeepSeekV4OperatorBackend::Cuda => self.cuda_runtime_counters(),
        };
        counters.cuda_graph_capture_attempts = self.cuda_graph_capture_attempts;
        counters.cuda_graph_capture_successes = self.cuda_graph_capture_successes;
        counters.cuda_graph_capture_failures = self.cuda_graph_capture_failures;
        counters.cuda_graph_capture_us = self.cuda_graph_capture_us;
        counters.cuda_graph_capture_warmup_us = self.cuda_graph_capture_warmup_us;
        counters.cuda_graph_capture_prepare_us = self.cuda_graph_capture_prepare_us;
        counters.cuda_graph_capture_record_us = self.cuda_graph_capture_record_us;
        counters.cuda_graph_full_capture_failures = self.cuda_graph_full_capture_failures;
        counters.cuda_graph_captured_segments = self.cuda_graph_captured_segments;
        counters.cuda_graph_one_shot_retires = self.cuda_graph_one_shot_retires;
        counters.cuda_graph_replays = self.cuda_graph_replays;
        counters.cuda_graph_replay_us = self.cuda_graph_replay_us;
        counters.cuda_graph_replay_fallbacks = self.cuda_graph_replay_fallbacks;
        counters
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

    pub(crate) fn record_moe_access_event(&mut self, event: ExpertBatchAccessEvent) {
        self.moe_access_events.push(event);
    }

    pub(crate) fn record_cuda_graph_capture_attempt(&mut self) {
        self.cuda_graph_capture_attempts = self.cuda_graph_capture_attempts.saturating_add(1);
    }

    pub(crate) fn record_cuda_graph_capture_success(&mut self) {
        self.cuda_graph_capture_successes = self.cuda_graph_capture_successes.saturating_add(1);
    }

    pub(crate) fn record_cuda_graph_capture_failure(&mut self) {
        self.cuda_graph_capture_failures = self.cuda_graph_capture_failures.saturating_add(1);
    }

    pub(crate) fn record_cuda_graph_capture_elapsed(&mut self, elapsed: Duration) {
        self.cuda_graph_capture_us = self
            .cuda_graph_capture_us
            .saturating_add(duration_us(elapsed));
    }

    pub(crate) fn record_cuda_graph_capture_warmup(&mut self, elapsed: Duration) {
        self.cuda_graph_capture_warmup_us = self
            .cuda_graph_capture_warmup_us
            .saturating_add(duration_us(elapsed));
    }

    pub(crate) fn record_cuda_graph_capture_prepare(&mut self, elapsed: Duration) {
        self.cuda_graph_capture_prepare_us = self
            .cuda_graph_capture_prepare_us
            .saturating_add(duration_us(elapsed));
    }

    pub(crate) fn record_cuda_graph_capture_record(&mut self, elapsed: Duration) {
        self.cuda_graph_capture_record_us = self
            .cuda_graph_capture_record_us
            .saturating_add(duration_us(elapsed));
    }

    pub(crate) fn record_cuda_graph_full_capture_failure(&mut self) {
        self.cuda_graph_full_capture_failures =
            self.cuda_graph_full_capture_failures.saturating_add(1);
    }

    pub(crate) fn record_cuda_graph_captured_segments(&mut self, segments: usize) {
        self.cuda_graph_captured_segments = self
            .cuda_graph_captured_segments
            .saturating_add(segments as u64);
    }

    pub(crate) fn record_cuda_graph_one_shot_retire(&mut self) {
        self.cuda_graph_one_shot_retires = self.cuda_graph_one_shot_retires.saturating_add(1);
    }

    pub(crate) fn record_cuda_graph_replay(&mut self, elapsed: Duration) {
        self.cuda_graph_replays = self.cuda_graph_replays.saturating_add(1);
        self.cuda_graph_replay_us = self
            .cuda_graph_replay_us
            .saturating_add(duration_us(elapsed));
    }

    pub(crate) fn record_cuda_graph_replay_fallback(&mut self) {
        self.cuda_graph_replay_fallbacks = self.cuda_graph_replay_fallbacks.saturating_add(1);
    }

    pub(crate) fn drain_moe_access_events(&mut self) -> Vec<ExpertBatchAccessEvent> {
        let events = std::mem::take(&mut self.moe_access_events);
        self.extend_cuda_moe_access_events(events)
    }

    #[cfg(feature = "cuda")]
    fn extend_cuda_moe_access_events(
        &mut self,
        mut events: Vec<ExpertBatchAccessEvent>,
    ) -> Vec<ExpertBatchAccessEvent> {
        if let Some(cuda) = self.cuda.as_mut() {
            events.extend(cuda.drain_moe_access_events());
        }
        events
    }

    #[cfg(not(feature = "cuda"))]
    fn extend_cuda_moe_access_events(
        &mut self,
        events: Vec<ExpertBatchAccessEvent>,
    ) -> Vec<ExpertBatchAccessEvent> {
        events
    }

    pub(crate) fn finish_profile_stage(&mut self, start: Instant) -> Result<u64> {
        self.sync_profile_stream()?;
        Ok(duration_us(start.elapsed()))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn sync_profile_stream(&mut self) -> Result<()> {
        if self.profile_sync && self.backend == DeepSeekV4OperatorBackend::Cuda {
            self.cuda_mut()?.ops.sync_stream()?;
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn sync_profile_stream(&mut self) -> Result<()> {
        Ok(())
    }

    pub(crate) fn record_layer_bind(&mut self, layer: usize, elapsed_us: u64) {
        let stats = self.layer_profile_entry(layer);
        stats.bind_calls = stats.bind_calls.saturating_add(1);
        stats.bind_us = stats.bind_us.saturating_add(elapsed_us);
    }

    pub(crate) fn record_layer_state_init(&mut self, layer: usize, elapsed_us: u64) {
        let stats = self.layer_profile_entry(layer);
        stats.state_init_calls = stats.state_init_calls.saturating_add(1);
        stats.state_init_us = stats.state_init_us.saturating_add(elapsed_us);
    }

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
            DeepSeekV4AttentionProfileStage::QLatentDownload => {
                stats.q_latent_download_us = stats.q_latent_download_us.saturating_add(elapsed_us)
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
            DeepSeekV4AttentionProfileStage::HiddenDownload => {
                stats.hidden_download_us = stats.hidden_download_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::IndexerCompress => {
                stats.indexer_compress_us = stats.indexer_compress_us.saturating_add(elapsed_us)
            }
            DeepSeekV4AttentionProfileStage::MainCompress => {
                stats.main_compress_us = stats.main_compress_us.saturating_add(elapsed_us)
            }
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
            DeepSeekV4AttentionProfileStage::OutputDownload => {
                stats.output_download_us = stats.output_download_us.saturating_add(elapsed_us)
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
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => linear.reference_matvec(input),
            DeepSeekV4OperatorBackend::Cuda => self.cuda_matvec(linear, input),
        }
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
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => {
                let mut output = Vec::with_capacity(rows * linear.format.out_features());
                for row in 0..rows {
                    let start = row * in_features;
                    output.extend_from_slice(
                        &linear.reference_matvec(&input[start..start + in_features])?,
                    );
                }
                Ok(output)
            }
            DeepSeekV4OperatorBackend::Cuda => self.cuda_linear_rows(linear, input, rows),
        }
    }

    pub(crate) fn linear_topk(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &[f32],
        top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => {
                let logits = linear.reference_matvec(input)?;
                let mut top = logits
                    .into_iter()
                    .enumerate()
                    .map(|(token_id, logit)| DeepSeekV4Logit {
                        token_id: token_id as u32,
                        logit,
                    })
                    .collect::<Vec<_>>();
                top.sort_by(rank_logits_desc);
                top.truncate(top_k);
                Ok(top)
            }
            DeepSeekV4OperatorBackend::Cuda => self.cuda_linear_topk(linear, input, top_k),
        }
    }

    pub(crate) fn output_head_topk_chunks(
        &mut self,
        slice: &ArtifactTensorSlice,
        hidden: &[f32],
        top_k: usize,
        chunk_rows: usize,
        reader: &ArtifactTensorReader,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => Err(Error::Internal(
                "CPU output-head chunk top-k should use the reference row-read loop".into(),
            )),
            DeepSeekV4OperatorBackend::Cuda => {
                self.cuda_output_head_topk_chunks(slice, hidden, top_k, chunk_rows, reader)
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
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => {
                sparse_attention_reference(query, values, topk, Some(sink), tokens, kv_len, spec)
            }
            DeepSeekV4OperatorBackend::Cuda => {
                self.cuda_sparse_attention(query, values, topk, sink, tokens, kv_len, spec)
            }
        }
    }

    pub(crate) fn grouped_output_a(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &[f32],
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
    ) -> Result<Vec<f32>> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => grouped_output_a(output_a, context, cfg, layer),
            DeepSeekV4OperatorBackend::Cuda => {
                self.cuda_grouped_output_a(output_a, context, cfg, layer)
            }
        }
    }

    /// Device-resident grouped output_a matvec.
    ///
    /// CUDA only: `context` is a device buffer, returns a device buffer.
    /// Falls back to an error on CPU (the device-resident path is CUDA-only).
    #[cfg(feature = "cuda")]
    pub(crate) fn grouped_output_a_from_device(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &ferrule_cuda::context::CudaF32Buffer,
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_grouped_output_a_from_device(output_a, context, cfg, layer)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn grouped_output_a_from_device_into(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &ferrule_cuda::context::CudaF32Buffer,
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.cuda_mut()?
            .grouped_output_a_from_device_into(output_a, context, cfg, layer, output)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn grouped_output_a_rows_from_host(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &[f32],
        rows: usize,
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?
            .grouped_output_a_rows_from_host(output_a, context, rows, cfg, layer)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn grouped_output_a_rows_from_device(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &ferrule_cuda::context::CudaF32Buffer,
        rows: usize,
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?
            .grouped_output_a_rows_from_device(output_a, context, rows, cfg, layer)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn linear_rows_from_device(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &ferrule_cuda::context::CudaF32Buffer,
        rows: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?
            .linear_rows_from_device(linear, input, rows)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn linear_rows_from_device_to_host(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &ferrule_cuda::context::CudaF32Buffer,
        rows: usize,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?
            .linear_rows_from_device_to_host(linear, input, rows)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn gather_f32_rows(
        &mut self,
        input: &ferrule_cuda::context::CudaF32Buffer,
        indices: &[i32],
        rows: usize,
        row_dim: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?
            .gather_f32_rows(input, indices, rows, row_dim)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn concat_attention_values_device(
        &mut self,
        window_values: &ferrule_cuda::context::CudaF32Buffer,
        compressed_values: &[f32],
        window_rows: usize,
        head_dim: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?.concat_attention_values_device(
            window_values,
            compressed_values,
            window_rows,
            head_dim,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn concat_attention_values_device_buffers(
        &mut self,
        window_values: &ferrule_cuda::context::CudaF32Buffer,
        compressed_values: &ferrule_cuda::context::CudaF32Buffer,
        window_rows: usize,
        head_dim: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?.concat_attention_values_device_buffers(
            window_values,
            compressed_values,
            window_rows,
            head_dim,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_linear_rows(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?
            .linear_rows_from_host_to_host(linear, input, rows)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_linear_rows(
        &mut self,
        _linear: &ArtifactLinearPayload,
        _input: &[f32],
        _rows: usize,
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn grouped_output_a_from_device(
        &mut self,
        _output_a: &ArtifactLinearPayload,
        _context: &(),
        _cfg: DeepSeekV4AttentionConfig,
        _layer: usize,
    ) -> Result<()> {
        Err(Error::Model(
            "DeepSeek-V4 device-resident grouped_output_a requires CUDA backend".into(),
        ))
    }

    pub(crate) fn rms_norm(
        &mut self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
        label: &str,
    ) -> Result<Vec<f32>> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => rms_norm(input, weight, eps, label),
            DeepSeekV4OperatorBackend::Cuda => self.cuda_rms_norm(input, weight, eps),
        }
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
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => {
                let mut out = Vec::with_capacity(input.len());
                for row in 0..rows {
                    let start = row * weight.len();
                    let normalized =
                        rms_norm(&input[start..start + weight.len()], weight, eps, label)?;
                    out.extend_from_slice(&normalized);
                }
                Ok(out)
            }
            DeepSeekV4OperatorBackend::Cuda => {
                self.cuda_rms_norm_rows(input, rows, weight, eps, label)
            }
        }
    }

    pub(crate) fn rms_norm_heads_in_place(
        &mut self,
        values: &mut [f32],
        heads: usize,
        head_dim: usize,
        eps: f32,
        layer: usize,
    ) -> Result<()> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => {
                rms_norm_heads_in_place(values, heads, head_dim, eps, layer)
            }
            DeepSeekV4OperatorBackend::Cuda => {
                let normalized = self.cuda_rms_norm_heads(values, heads, head_dim, eps)?;
                values.copy_from_slice(&normalized);
                Ok(())
            }
        }
    }

    pub(crate) fn hc_pre(
        &mut self,
        state: &[f32],
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionWeights,
    ) -> Result<HyperConnectionPreOutput> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => hc_pre_reference(state, tokens, config, weights),
            DeepSeekV4OperatorBackend::Cuda => self.cuda_hc_pre(state, tokens, config, weights),
        }
    }

    pub(crate) fn hc_post(
        &mut self,
        hidden: &[f32],
        residual: &[f32],
        config: HyperConnectionConfig,
        split: &HyperConnectionSplit,
    ) -> Result<Vec<f32>> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => hc_post_reference(hidden, residual, config, split),
            DeepSeekV4OperatorBackend::Cuda => self.cuda_hc_post(hidden, residual, config, split),
        }
    }

    pub(crate) fn hc_head(
        &mut self,
        state: &[f32],
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionHeadWeights,
    ) -> Result<Vec<f32>> {
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => hc_head_reference(state, tokens, config, weights),
            DeepSeekV4OperatorBackend::Cuda => self.cuda_hc_head(state, tokens, config, weights),
        }
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
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => {
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
            DeepSeekV4OperatorBackend::Cuda => self.cuda_routed_moe_step(
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
            ),
        }
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
        match self.backend {
            DeepSeekV4OperatorBackend::Cpu => {
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
            DeepSeekV4OperatorBackend::Cuda => self.cuda_routed_moe_prefill_batch(
                layer,
                input,
                token_ids,
                router,
                predicted_experts,
                router_policy,
                planner,
                reader,
                handles,
                shared_expert,
            ),
        }
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn routed_moe_prefill_batch_from_device(
        &mut self,
        layer: usize,
        input: &ferrule_cuda::context::CudaF32Buffer,
        token_ids: &[u32],
        router: &RouterArtifactPayload,
        predicted_experts: &[usize],
        router_policy: &ExpertRouterPolicy,
        planner: &mut ExpertStreamingPlanner,
        reader: &ExpertStreamingReader,
        handles: &mut CpuExpertHandleStore,
        shared_expert: Option<&SwiGluFfnPayload>,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        if self.backend != DeepSeekV4OperatorBackend::Cuda {
            return Err(Error::Model(
                "DeepSeek-V4 device-resident MoE prefill requires CUDA backend".into(),
            ));
        }
        let output = self.cuda_mut()?.routed_moe_prefill_batch_from_device(
            layer,
            input,
            token_ids,
            router,
            predicted_experts,
            router_policy,
            planner,
            reader,
            handles,
            shared_expert,
        )?;
        if let Some(cuda) = self.cuda.as_mut() {
            self.moe_access_events
                .extend(cuda.drain_moe_access_events());
        }
        Ok(output)
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
        if self.backend != DeepSeekV4OperatorBackend::Cuda {
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
        if self.backend != DeepSeekV4OperatorBackend::Cuda {
            return Ok(0);
        }
        self.cuda_mut()?
            .prewarm_experts(layer, experts, planner, reader, handles)
    }

    pub(crate) fn clear_sequence_workspaces(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if self.backend == DeepSeekV4OperatorBackend::Cuda {
                self.cuda_mut()?.clear_sequence_workspaces();
            }
        }
        Ok(())
    }

    pub(crate) fn clear_expert_residency(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if self.backend == DeepSeekV4OperatorBackend::Cuda {
                self.cuda_mut()?.clear_expert_residency()?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_mut(&mut self) -> Result<&mut DeepSeekV4CudaOperatorCache> {
        self.cuda.as_mut().ok_or_else(|| {
            Error::Model("DeepSeek-V4 CUDA operator cache is not initialized".into())
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_runtime_counters(&self) -> DeepSeekV4OperatorRuntimeCounters {
        self.cuda
            .as_ref()
            .map(DeepSeekV4CudaOperatorCache::runtime_counters)
            .unwrap_or_default()
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_runtime_counters(&self) -> DeepSeekV4OperatorRuntimeCounters {
        DeepSeekV4OperatorRuntimeCounters::default()
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_matvec(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?.linear_matvec(linear, input)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_linear_topk(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &[f32],
        top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        self.cuda_mut()?.linear_topk(linear, input, top_k)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_matvec(
        &mut self,
        _linear: &ArtifactLinearPayload,
        _input: &[f32],
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_output_head_topk_chunks(
        &mut self,
        slice: &ArtifactTensorSlice,
        hidden: &[f32],
        top_k: usize,
        chunk_rows: usize,
        reader: &ArtifactTensorReader,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        self.cuda_mut()?
            .output_head_topk_chunks(slice, hidden, top_k, chunk_rows, reader)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_output_head_topk_chunks_with_device(
        &mut self,
        slice: &ArtifactTensorSlice,
        hidden: &ferrule_cuda::context::CudaF32Buffer,
        top_k: usize,
        chunk_rows: usize,
        reader: &ArtifactTensorReader,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        let vocab_rows =
            slice.shape.first().copied().ok_or_else(|| {
                Error::Model("DeepSeek-V4 CUDA output head expects 2D slice".into())
            })?;
        self.cuda_mut()?.output_head_topk_chunks_with_device(
            slice, hidden, top_k, chunk_rows, reader, vocab_rows,
        )
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_output_head_topk_chunks(
        &mut self,
        _slice: &ArtifactTensorSlice,
        _hidden: &[f32],
        _top_k: usize,
        _chunk_rows: usize,
        _reader: &ArtifactTensorReader,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_sparse_attention(
        &mut self,
        query: &[f32],
        values: &[f32],
        topk: &[isize],
        sink: &[f32],
        tokens: usize,
        kv_len: usize,
        spec: SparseAttentionSpec,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?
            .sparse_attention(query, values, topk, sink, tokens, kv_len, spec)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_linear_topk(
        &mut self,
        _linear: &ArtifactLinearPayload,
        _input: &[f32],
        _top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_sparse_attention(
        &mut self,
        _query: &[f32],
        _values: &[f32],
        _topk: &[isize],
        _sink: &[f32],
        _tokens: usize,
        _kv_len: usize,
        _spec: SparseAttentionSpec,
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_grouped_output_a(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &[f32],
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?
            .grouped_output_a(output_a, context, cfg, layer)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_grouped_output_a(
        &mut self,
        _output_a: &ArtifactLinearPayload,
        _context: &[f32],
        _cfg: DeepSeekV4AttentionConfig,
        _layer: usize,
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_grouped_output_a_from_device(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &ferrule_cuda::context::CudaF32Buffer,
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?
            .grouped_output_a_from_device(output_a, context, cfg, layer)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_grouped_output_a_from_device_into(
        &mut self,
        output_a: &ArtifactLinearPayload,
        context: &ferrule_cuda::context::CudaF32Buffer,
        cfg: DeepSeekV4AttentionConfig,
        layer: usize,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.cuda_mut()?
            .grouped_output_a_from_device_into(output_a, context, cfg, layer, output)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_grouped_output_a_from_device(
        &mut self,
        _output_a: &ArtifactLinearPayload,
        _context: &(),
        _cfg: DeepSeekV4AttentionConfig,
        _layer: usize,
    ) -> Result<()> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_rms_norm(
        &mut self,
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?.rms_norm(input, weight, eps)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_rms_norm_rows(
        &mut self,
        input: &[f32],
        rows: usize,
        weight: &[f32],
        eps: f32,
        label: &str,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?
            .rms_norm_rows_cached(label, input, rows, weight, eps)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_rms_norm(
        &mut self,
        _input: &[f32],
        _weight: &[f32],
        _eps: f32,
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_rms_norm_rows(
        &mut self,
        _input: &[f32],
        _rows: usize,
        _weight: &[f32],
        _eps: f32,
        _label: &str,
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_rms_norm_heads(
        &mut self,
        input: &[f32],
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?.rms_norm_heads(input, heads, head_dim, eps)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_rms_norm_heads(
        &mut self,
        _input: &[f32],
        _heads: usize,
        _head_dim: usize,
        _eps: f32,
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_rms_norm_heads_from_device(
        &mut self,
        input: &ferrule_cuda::context::CudaF32Buffer,
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?
            .rms_norm_heads_from_device(input, heads, head_dim, eps)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_rms_norm_heads_from_device_into(
        &mut self,
        input: &ferrule_cuda::context::CudaF32Buffer,
        heads: usize,
        head_dim: usize,
        eps: f32,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.cuda_mut()?
            .rms_norm_heads_from_device_into(input, heads, head_dim, eps, output)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_rms_norm_heads_from_device(
        &mut self,
        _input: &(),
        _heads: usize,
        _head_dim: usize,
        _eps: f32,
    ) -> Result<()> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_hc_pre(
        &mut self,
        state: &[f32],
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionWeights,
    ) -> Result<HyperConnectionPreOutput> {
        self.cuda_mut()?.hc_pre(state, tokens, config, weights)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_hc_pre(
        &mut self,
        _state: &[f32],
        _tokens: usize,
        _config: HyperConnectionConfig,
        _weights: &HyperConnectionWeights,
    ) -> Result<HyperConnectionPreOutput> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_hc_post(
        &mut self,
        hidden: &[f32],
        residual: &[f32],
        config: HyperConnectionConfig,
        split: &HyperConnectionSplit,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?.hc_post(hidden, residual, config, split)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_hc_post(
        &mut self,
        _hidden: &[f32],
        _residual: &[f32],
        _config: HyperConnectionConfig,
        _split: &HyperConnectionSplit,
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_hc_head(
        &mut self,
        state: &[f32],
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionHeadWeights,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?.hc_head(state, tokens, config, weights)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_hc_head(
        &mut self,
        _state: &[f32],
        _tokens: usize,
        _config: HyperConnectionConfig,
        _weights: &HyperConnectionHeadWeights,
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_routed_moe_step(
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
        self.cuda_mut()?.routed_moe_step(
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
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_routed_moe_step_from_device(
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
        self.cuda_mut()?.routed_moe_step_from_device(
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
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_routed_moe_step_device_output(
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
        self.cuda_mut()?.routed_moe_step_device_output(
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
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_routed_moe_prefill_batch(
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
        let output = self.cuda_mut()?.routed_moe_prefill_batch(
            layer,
            input,
            token_ids,
            router,
            predicted_experts,
            router_policy,
            planner,
            reader,
            handles,
            shared_expert,
        )?;
        if let Some(cuda) = self.cuda.as_mut() {
            self.moe_access_events
                .extend(cuda.drain_moe_access_events());
        }
        Ok(output)
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_routed_moe_prefill_batch(
        &mut self,
        _layer: usize,
        _input: &[f32],
        _token_ids: &[u32],
        _router: &RouterArtifactPayload,
        _predicted_experts: &[usize],
        _router_policy: &ExpertRouterPolicy,
        _planner: &mut ExpertStreamingPlanner,
        _reader: &ExpertStreamingReader,
        _handles: &mut CpuExpertHandleStore,
        _shared_expert: Option<&SwiGluFfnPayload>,
    ) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_routed_moe_step(
        &mut self,
        _layer: usize,
        _input: &[f32],
        _token_id: u32,
        _router: &RouterArtifactPayload,
        _predicted_experts: &[usize],
        _router_policy: &ExpertRouterPolicy,
        _planner: &mut ExpertStreamingPlanner,
        _reader: &ExpertStreamingReader,
        _handles: &mut CpuExpertHandleStore,
        _shared_expert: Option<&SwiGluFfnPayload>,
    ) -> Result<RoutedMoeStepOutput> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    // ── Device-resident op variants (CUDA only) ──
    //
    // These accept and return `CudaF32Buffer` instead of `&[f32]`/`Vec<f32>`,
    // allowing multiple ops to chain without host round-trips.

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_linear_matvec_from_device(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?.linear_matvec_from_device(linear, input)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_linear_matvec_from_device_into(
        &mut self,
        linear: &ArtifactLinearPayload,
        input: &mut ferrule_cuda::context::CudaF32Buffer,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.cuda_mut()?
            .linear_matvec_from_device_into(linear, input, output)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_linear_matvec_from_device(
        &mut self,
        _linear: &ArtifactLinearPayload,
        _input: &(),
    ) -> Result<()> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_rms_norm_device_cached(
        &mut self,
        name: &str,
        input: &ferrule_cuda::context::CudaF32Buffer,
        weight: &[f32],
        eps: f32,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?
            .rms_norm_device_cached(name, input, weight, eps)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_rms_norm_device_cached_into(
        &mut self,
        name: &str,
        input: &ferrule_cuda::context::CudaF32Buffer,
        weight: &[f32],
        eps: f32,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.cuda_mut()?
            .rms_norm_device_cached_into(name, input, weight, eps, output)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_rms_norm_rows_device_cached(
        &mut self,
        name: &str,
        input: &ferrule_cuda::context::CudaF32Buffer,
        rows: usize,
        weight: &[f32],
        eps: f32,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?
            .rms_norm_rows_device_cached(name, input, rows, weight, eps)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_rms_norm_device_cached(
        &mut self,
        _name: &str,
        _input: &(),
        _weight: &[f32],
        _eps: f32,
    ) -> Result<()> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_rms_norm_device_cached_into(
        &mut self,
        _name: &str,
        _input: &(),
        _weight: &[f32],
        _eps: f32,
        _output: &mut (),
    ) -> Result<()> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_rms_norm_rows_device_cached(
        &mut self,
        _name: &str,
        _input: &(),
        _rows: usize,
        _weight: &[f32],
        _eps: f32,
    ) -> Result<()> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_upload_f32(
        &mut self,
        values: &[f32],
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?.ops.upload_f32_buffer(values)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_upload_f32(&mut self, _values: &[f32]) -> Result<()> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_download_f32(
        &mut self,
        buffer: &ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<Vec<f32>> {
        self.cuda_mut()?.ops.download_f32_buffer(buffer)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_clone_f32(
        &mut self,
        src: &ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?.ops.clone_f32_buffer(src)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_zero_f32(
        &mut self,
        len: usize,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?.ops.zero_f32_buffer(len)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_zero_i32(
        &mut self,
        len: usize,
    ) -> Result<ferrule_cuda::context::CudaI32Buffer> {
        self.cuda_mut()?.ops.zero_i32_buffer(len)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_copy_f32_into_slot(
        &mut self,
        src: &ferrule_cuda::context::CudaF32Buffer,
        dst: &mut ferrule_cuda::context::CudaF32Buffer,
        dst_offset_elements: usize,
    ) -> Result<()> {
        self.cuda_mut()?
            .ops
            .copy_f32_into_slot(src, dst, dst_offset_elements)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_download_f32(&mut self, _buffer: &()) -> Result<Vec<f32>> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_clone_f32(&mut self, _src: &()) -> Result<()> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    /// Graph-safe MoE: uses pre-determined routes, no D2H, no streaming.
    /// Only launches kernels — safe for CUDA graph capture.
    /// The caller provides a pre-allocated accumulator buffer.
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_routed_moe_graph_safe(
        &mut self,
        input: &ferrule_cuda::context::CudaF32Buffer,
        route_experts: &[usize],
        route_weights: &[f32],
        layer: usize,
        shared_expert: Option<&SwiGluFfnPayload>,
        accumulator: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.cuda_mut()?.routed_moe_step_graph_safe(
            input,
            route_experts,
            route_weights,
            layer,
            shared_expert,
            accumulator,
        )
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_routed_moe_graph_safe(
        &mut self,
        _input: &(),
        _route_experts: &[usize],
        _route_weights: &[f32],
        _layer: usize,
        _shared_expert: Option<&SwiGluFfnPayload>,
        _accumulator: &mut (),
    ) -> Result<()> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_hc_pre_from_device(
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
        self.cuda_mut()?
            .hc_pre_from_device(name, state, weights, tokens, config)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_hc_pre_from_device_into(
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
        self.cuda_mut()?.hc_pre_from_device_into(
            name, state, weights, tokens, config, hidden, pre, post, comb,
        )
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_hc_pre_from_device(
        &mut self,
        _name: &str,
        _state: &(),
        _weights: &HyperConnectionWeights,
        _tokens: usize,
        _config: HyperConnectionConfig,
    ) -> Result<()> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_hc_post_from_device(
        &mut self,
        hidden: &ferrule_cuda::context::CudaF32Buffer,
        residual: &ferrule_cuda::context::CudaF32Buffer,
        split_post: &ferrule_cuda::context::CudaF32Buffer,
        split_comb: &ferrule_cuda::context::CudaF32Buffer,
        tokens: usize,
        config: HyperConnectionConfig,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?
            .hc_post_from_device(hidden, residual, split_post, split_comb, tokens, config)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_hc_post_from_device_into(
        &mut self,
        hidden: &ferrule_cuda::context::CudaF32Buffer,
        residual: &ferrule_cuda::context::CudaF32Buffer,
        split_post: &ferrule_cuda::context::CudaF32Buffer,
        split_comb: &ferrule_cuda::context::CudaF32Buffer,
        tokens: usize,
        config: HyperConnectionConfig,
        output: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        self.cuda_mut()?.hc_post_from_device_into(
            hidden, residual, split_post, split_comb, tokens, config, output,
        )
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_hc_post_from_device(
        &mut self,
        _hidden: &(),
        _residual: &(),
        _split_post: &(),
        _split_comb: &(),
        _tokens: usize,
        _config: HyperConnectionConfig,
    ) -> Result<()> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_hc_head_from_device(
        &mut self,
        state: &ferrule_cuda::context::CudaF32Buffer,
        tokens: usize,
        config: HyperConnectionConfig,
        weights: &HyperConnectionHeadWeights,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?
            .hc_head_from_device(state, tokens, config, weights)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn cuda_hc_head_from_device(
        &mut self,
        _state: &(),
        _tokens: usize,
        _config: HyperConnectionConfig,
        _weights: &HyperConnectionHeadWeights,
    ) -> Result<()> {
        Err(Error::Model(
            "DeepSeek-V4 CUDA backend requires ferrule-runtime/cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_stream_clone(&mut self) -> Result<std::sync::Arc<ferrule_cuda::CudaStream>> {
        self.cuda
            .as_ref()
            .map(|cache| cache.ops.stream_clone())
            .ok_or_else(|| {
                Error::Model("DeepSeek-V4 CUDA operator cache is not initialized".into())
            })
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_launch_graph(
        &mut self,
        graph: &ferrule_cuda::graph::CudaGraphHandle,
    ) -> Result<()> {
        self.cuda_mut()?.ops.launch_graph(graph)
    }

    // ── Fully device-resident attention wrappers ──

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_ensure_kv_cache(
        &mut self,
        layer: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        self.cuda_mut()?
            .ensure_kv_cache(layer, window_size, head_dim)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_kv_append_device(
        &mut self,
        layer: usize,
        kv_buffer: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        head_dim: usize,
        window_size: usize,
    ) -> Result<()> {
        self.cuda_mut()?
            .kv_append_device(layer, kv_buffer, position, head_dim, window_size)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_kv_write_window_device(
        &mut self,
        layer: usize,
        kv_buffer: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        head_dim: usize,
        window_size: usize,
    ) -> Result<()> {
        self.cuda_mut()?
            .kv_write_window_device(layer, kv_buffer, position, head_dim, window_size)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_kv_write_window_rows_device(
        &mut self,
        layer: usize,
        values: &ferrule_cuda::context::CudaF32Buffer,
        start_position: usize,
        rows: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        self.cuda_mut()?.kv_write_window_rows_device(
            layer,
            values,
            start_position,
            rows,
            window_size,
            head_dim,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_kv_values_device(
        &self,
        layer: usize,
    ) -> Result<&ferrule_cuda::context::CudaF32Buffer> {
        self.cuda
            .as_ref()
            .map(|cache| cache.kv_values_device(layer))
            .ok_or_else(|| {
                Error::Model("DeepSeek-V4 CUDA operator cache is not initialized".into())
            })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_ensure_rope_tables(
        &mut self,
        name: &str,
        rope_dim: usize,
        rope_theta: f32,
        max_positions: usize,
    ) -> Result<()> {
        self.cuda_mut()?
            .ensure_rope_tables(name, rope_dim, rope_theta, max_positions)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_ensure_rope_tables_with_params(
        &mut self,
        name: &str,
        rope_dim: usize,
        rope: super::config::DeepSeekV4RopeParams,
        max_positions: usize,
    ) -> Result<()> {
        self.cuda_mut()?
            .ensure_rope_tables_with_params(name, rope_dim, rope, max_positions)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_rope_tail_from_device(
        &mut self,
        name: &str,
        qk: &mut ferrule_cuda::context::CudaF32Buffer,
        position: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        let cache = self.cuda_mut()?;
        cache.ensure_rope_tables(name, rope_dim as usize, 0.0, 0)?;
        let cos = cache.rope_cos_device(name);
        let sin = cache.rope_sin_device(name);
        cache
            .ops
            .rope_tail_from_device(qk, cos, sin, position, heads, head_dim, rope_dim, inverse)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_rope_tail_rows_from_device(
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
        self.cuda_rope_tail_rows_strided_from_device(
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

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_rope_tail_rows_strided_from_device(
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
        let cache = self.cuda_mut()?;
        cache.ensure_rope_tables(name, rope_dim as usize, 0.0, 0)?;
        let cos = cache.rope_cos_device(name);
        let sin = cache.rope_sin_device(name);
        cache.ops.rope_tail_rows_strided_from_device(
            qk,
            cos,
            sin,
            start_position,
            position_stride,
            rows,
            heads,
            head_dim,
            rope_dim,
            inverse,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_fp8_activation_quantize_buffer_in_place(
        &mut self,
        values: &mut ferrule_cuda::context::CudaF32Buffer,
        row_width: usize,
        block_size: usize,
    ) -> Result<()> {
        self.cuda_mut()?
            .ops
            .fp8_activation_quantize_buffer_in_place(values, row_width, block_size)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_attention_kv_qat_quantize_buffer_in_place(
        &mut self,
        values: &mut ferrule_cuda::context::CudaF32Buffer,
        head_dim: usize,
        rope_dim: usize,
    ) -> Result<()> {
        self.cuda_mut()?
            .ops
            .fp8_attention_kv_qat_quantize_buffer_in_place(values, head_dim, rope_dim)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_indexer_qat_quantize_buffer_in_place(
        &mut self,
        values: &mut ferrule_cuda::context::CudaF32Buffer,
        row_width: usize,
    ) -> Result<()> {
        self.cuda_mut()?
            .ops
            .fp4_hadamard_qat_quantize_buffer_in_place(values, row_width)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_compressor_prefill_softmax_from_device(
        &mut self,
        kv_rows: &ferrule_cuda::context::CudaF32Buffer,
        score_rows: &ferrule_cuda::context::CudaF32Buffer,
        ape: &[f32],
        groups: usize,
        ratio: usize,
        head_dim: usize,
        out_dim: usize,
        overlap: bool,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?.ops.compressor_prefill_softmax_from_device(
            kv_rows, score_rows, ape, groups, ratio, head_dim, out_dim, overlap,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_prefill_topk_indices_from_device(
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
        self.cuda_mut()?.prefill_topk_indices_from_device(
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

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_prefill_topk_indices_fused_index_query_from_device(
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
        self.cuda_mut()?
            .prefill_topk_indices_fused_index_query_from_device(
                query,
                weights,
                indexer_kv,
                rope_name,
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

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_decode_topk_indices_from_device(
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
        self.cuda_mut()?.decode_topk_indices_from_device(
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

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_decode_topk_indices_from_device_into(
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
        self.cuda_mut()?.decode_topk_indices_from_device_into(
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

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_decode_topk_indices_fused_index_query_from_indexer_cache_into(
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
        self.cuda_mut()?
            .decode_topk_indices_fused_index_query_from_indexer_cache_into(
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
                out,
            )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_decode_topk_indices_fused_index_query_from_indexer_cache(
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
        self.cuda_mut()?
            .decode_topk_indices_fused_index_query_from_indexer_cache(
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
            )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_decode_topk_indices_from_indexer_cache_into(
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
        self.cuda_mut()?
            .decode_topk_indices_from_indexer_cache_into(
                query,
                weights,
                layer,
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

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_decode_topk_indices_from_indexer_cache(
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
        self.cuda_mut()?.decode_topk_indices_from_indexer_cache(
            query,
            weights,
            layer,
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

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_ensure_topk_buffer(&mut self, window_size: usize) -> Result<()> {
        self.cuda_mut()?.ensure_topk_buffer(window_size)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_fill_topk_buffer(
        &mut self,
        position: usize,
        window_size: usize,
    ) -> Result<()> {
        self.cuda_mut()?.fill_topk_buffer(position, window_size)?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_topk_buffer_device(&self) -> Result<&ferrule_cuda::context::CudaI32Buffer> {
        self.cuda
            .as_ref()
            .and_then(|cache| Some(cache.topk_buffer_device()))
            .ok_or_else(|| Error::Model("DeepSeek-V4 topk buffer not initialized".into()))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_sparse_attention_topk_from_device(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        layer: usize,
        position: usize,
        window_size: usize,
        sink: &[f32],
        shape: ferrule_cuda::transformer::sparse_attention::CudaSparseAttentionShape,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let cache = self.cuda_mut()?;
        cache.ensure_topk_buffer(window_size)?;
        cache.fill_topk_buffer(position, window_size)?;
        let sink_name = format!("sink_L{layer}");
        cache.ensure_sink_buffer(&sink_name, sink)?;
        let topk = cache.topk_buffer_device();
        let kv_values = cache.kv_values_device(layer);
        let sink_buf = cache.sink_buffers.get(&sink_name).expect("inserted above");
        cache.ops.sparse_attention_sink_from_device(
            query,
            kv_values,
            topk.as_device_buffer(),
            sink_buf,
            shape,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_sparse_attention_with_device_query(
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
        self.cuda_mut()?.sparse_attention_with_device_query(
            query, values, topk, sink, tokens, kv_len, spec, layer,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_sparse_attention_with_device_query_and_values(
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
        self.cuda_mut()?
            .sparse_attention_with_device_query_and_values(
                query, values, topk, sink, tokens, kv_len, spec, layer,
            )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_sparse_attention_with_device_query_values_topk(
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
        self.cuda_mut()?
            .sparse_attention_with_device_query_values_topk(
                query, values, topk, sink, tokens, kv_len, spec, layer,
            )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_ensure_combined_kv_cache(
        &mut self,
        layer: usize,
        cache: &DeepSeekV4AttentionCache,
        compressed_capacity: usize,
    ) -> Result<()> {
        self.cuda_mut()?
            .ensure_combined_kv_cache(layer, cache, compressed_capacity)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_combined_kv_append_window_host(
        &mut self,
        layer: usize,
        kv: &[f32],
        position: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        self.cuda_mut()?
            .combined_kv_append_window_host(layer, kv, position, window_size, head_dim)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_combined_kv_append_window_device(
        &mut self,
        layer: usize,
        kv: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        self.cuda_mut()?.combined_kv_append_window_device(
            layer,
            kv,
            position,
            window_size,
            head_dim,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_combined_kv_write_window_rows_device(
        &mut self,
        layer: usize,
        values: &ferrule_cuda::context::CudaF32Buffer,
        start_position: usize,
        rows: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        self.cuda_mut()?.combined_kv_write_window_rows_device(
            layer,
            values,
            start_position,
            rows,
            window_size,
            head_dim,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_combined_kv_write_compressed_rows_device(
        &mut self,
        layer: usize,
        values: &ferrule_cuda::context::CudaF32Buffer,
        compressed_start: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        self.cuda_mut()?.combined_kv_write_compressed_rows_device(
            layer,
            values,
            compressed_start,
            window_size,
            head_dim,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_combined_kv_append_compressed_host(
        &mut self,
        layer: usize,
        value: &[f32],
        compressed_index: usize,
        window_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        self.cuda_mut()?.combined_kv_append_compressed_host(
            layer,
            value,
            compressed_index,
            window_size,
            head_dim,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_ensure_indexer_kv_cache(
        &mut self,
        layer: usize,
        compressed_capacity: usize,
        index_head_dim: usize,
    ) -> Result<()> {
        self.cuda_mut()?
            .ensure_indexer_kv_cache(layer, compressed_capacity, index_head_dim)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_indexer_kv_write_rows_device(
        &mut self,
        layer: usize,
        values: &ferrule_cuda::context::CudaF32Buffer,
        compressed_start: usize,
        index_head_dim: usize,
    ) -> Result<()> {
        self.cuda_mut()?.indexer_kv_write_rows_device(
            layer,
            values,
            compressed_start,
            index_head_dim,
        )
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_indexer_kv_append_host(
        &mut self,
        layer: usize,
        value: &[f32],
        compressed_index: usize,
        index_head_dim: usize,
    ) -> Result<()> {
        self.cuda_mut()?
            .indexer_kv_append_host(layer, value, compressed_index, index_head_dim)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_sparse_attention_with_combined_kv(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        layer: usize,
        topk: &[isize],
        sink: &[f32],
        tokens: usize,
        kv_len: usize,
        spec: SparseAttentionSpec,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?
            .sparse_attention_with_combined_kv(query, layer, topk, sink, tokens, kv_len, spec)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_sparse_attention_with_kv_cache_topk_into(
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
        self.cuda_mut()?.sparse_attention_with_kv_cache_topk_into(
            query, layer, topk, sink, tokens, kv_len, spec, output,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_sparse_attention_with_combined_kv_topk(
        &mut self,
        query: &ferrule_cuda::context::CudaF32Buffer,
        layer: usize,
        topk: &ferrule_cuda::context::CudaI32Buffer,
        sink: &[f32],
        tokens: usize,
        kv_len: usize,
        spec: SparseAttentionSpec,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        self.cuda_mut()?
            .sparse_attention_with_combined_kv_topk(query, layer, topk, sink, tokens, kv_len, spec)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_sparse_attention_with_combined_kv_topk_into(
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
        self.cuda_mut()?
            .sparse_attention_with_combined_kv_topk_into(
                query, layer, topk, sink, tokens, kv_len, spec, output,
            )
    }
}
