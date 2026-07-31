//! DeepSeek-V4 operator context: CPU/CUDA dispatch for linear, attention, MoE, HC ops.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use ferrule_common::{CompletionHub, Error, ExpertResidencyStats, Result};

use crate::checkpoint::weight::LinearWeight;
use crate::moe::routing::RouterWeights;

use crate::attention_backend::{SparseAttentionSpec, sparse_attention_reference};
use crate::execution::ModelExecutionBackend;
use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::{
    HyperConnectionConfig, HyperConnectionPreOutput, HyperConnectionSplit, HyperConnectionWeights,
    hc_post_reference, hc_pre_reference,
};
use crate::moe::executor::ExpertExecutor;
use crate::moe::handle::CpuExpertHandleStore;
use crate::moe::prediction::{ExpertAccessPhase, ExpertBatchAccessEvent, ExpertPredictionStats};
use crate::moe::routed::{
    RoutedMoeStepOutput, execute_routed_moe_with_artifact_router_reference_with_handles,
};
use crate::moe::routing::ExpertRouterPolicy;
use crate::moe::streaming::{ExpertMemoryPolicy, ExpertStreamingPlanner, ExpertStreamingReader};

use super::config::DeepSeekV4AttentionConfig;
#[cfg(feature = "cuda")]
use super::cuda_cache::DeepSeekV4CudaOperatorCache;
#[cfg(feature = "cuda")]
use super::cuda_materialization::DeepSeekV4SharedExpertSubsystem;
use super::helpers::{grouped_output_a, rms_norm, rms_norm_heads_in_place};
use super::prepared::DeepSeekV4ExecutionPolicy;
#[cfg(feature = "cuda")]
use super::prepared::DeepSeekV4PreparedResources;

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
    pub moe_input_prepare_us: u64,
    pub moe_gate_up_us: u64,
    pub moe_swiglu_us: u64,
    pub moe_hidden_pack_us: u64,
    pub moe_down_us: u64,
    pub moe_router_us: u64,
    pub moe_routing_us: u64,
    pub moe_plan_us: u64,
    pub moe_shared_us: u64,
    pub moe_workspace_us: u64,
    pub moe_compute_submit_us: u64,
    pub moe_commit_us: u64,
    pub output_head_calls: u64,
    pub output_head_rows: u64,
    pub output_head_topk_us: u64,
    /// Sum of selected routes across MoE layer invocations.
    pub expert_selected: u64,
    /// Sum of unique experts within each MoE layer invocation.
    pub expert_unique_selected: u64,
    pub expert_selected_load_requests: u64,
    pub expert_io_submitted_extents: u64,
    pub expert_io_completed_extents: u64,
    pub expert_io_failed_extents: u64,
    pub expert_io_requested_bytes: u64,
    pub expert_io_aligned_bytes: u64,
    pub expert_io_coalesced_slices: u64,
    pub expert_io_fixed_file_registrations: u64,
    pub expert_io_slab_exhaustions: u64,
    pub expert_io_peak_queue_depth: usize,
    pub expert_io_read_us: u64,
    pub arena_hits: u64,
    pub arena_misses: u64,
    pub arena_grows: u64,
    pub arena_reuses: u64,
    pub expert_residency_stats: ExpertResidencyStats,
    pub expert_predictor_stats: ExpertPredictionStats,
}

fn duration_us(d: Duration) -> u64 {
    d.as_micros().min(u128::from(u64::MAX)) as u64
}

pub struct DeepSeekV4OperatorContext {
    pub(crate) backend: ModelExecutionBackend,
    layer_profiles: BTreeMap<usize, DeepSeekV4LayerProfileStats>,
    attention_profiles: BTreeMap<usize, DeepSeekV4AttentionProfileStats>,
    /// Frozen at preparation time. When false, hot paths do not sample the
    /// clock and profile maps remain empty.
    profile: bool,
    /// When enabled, profile stage timings synchronize the CUDA stream before
    /// sampling elapsed wall time. This is expensive but gives attribution that
    /// includes queued GPU work instead of only host enqueue time.
    profile_sync: bool,
    moe_access_events: Vec<ExpertBatchAccessEvent>,
    #[cfg(feature = "cuda")]
    pub(crate) cuda: Option<DeepSeekV4CudaOperatorCache>,
}

impl DeepSeekV4OperatorContext {
    pub fn new(
        backend: ModelExecutionBackend,
        policy: &DeepSeekV4ExecutionPolicy,
        expert_memory_policy: ExpertMemoryPolicy,
    ) -> Result<Self> {
        Self::new_with_completion_hub(backend, policy, expert_memory_policy, CompletionHub::new())
    }

    pub(crate) fn new_with_completion_hub(
        backend: ModelExecutionBackend,
        policy: &DeepSeekV4ExecutionPolicy,
        expert_memory_policy: ExpertMemoryPolicy,
        completion_hub: CompletionHub,
    ) -> Result<Self> {
        #[cfg(not(feature = "cuda"))]
        let _ = (expert_memory_policy, completion_hub);
        Ok(Self {
            backend,
            layer_profiles: BTreeMap::new(),
            attention_profiles: BTreeMap::new(),
            profile: policy.profile_enabled(),
            profile_sync: policy.profile_sync(),
            moe_access_events: Vec::new(),
            #[cfg(feature = "cuda")]
            cuda: match backend {
                ModelExecutionBackend::Cpu => None,
                ModelExecutionBackend::Cuda => {
                    Some(DeepSeekV4CudaOperatorCache::new_with_completion_hub(
                        policy,
                        expert_memory_policy,
                        completion_hub,
                    )?)
                }
            },
        })
    }

    pub fn new_cpu() -> Result<Self> {
        Self::new(
            ModelExecutionBackend::Cpu,
            &DeepSeekV4ExecutionPolicy::default(),
            ExpertMemoryPolicy::default(),
        )
    }

    pub fn backend(&self) -> ModelExecutionBackend {
        self.backend
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

    pub const fn profile_enabled(&self) -> bool {
        self.profile
    }

    pub fn profile_sync_enabled(&self) -> bool {
        self.profile && self.profile_sync
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

    #[inline]
    pub(crate) fn profile_start(&self) -> Option<Instant> {
        self.profile.then(Instant::now)
    }

    pub(crate) fn finish_profile_stage(&mut self, start: Option<Instant>) -> Result<Option<u64>> {
        let Some(start) = start else {
            return Ok(None);
        };
        self.sync_profile_stream()?;
        Ok(Some(duration_us(start.elapsed())))
    }

    pub(crate) fn sync_profile_stream(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.profile && self.profile_sync && self.backend == ModelExecutionBackend::Cuda {
            self.cuda_mut()?.ops.sync_stream()?;
        }
        Ok(())
    }

    pub(crate) fn record_layer_state_init(&mut self, layer: usize, elapsed_us: u64) {
        if !self.profile {
            return;
        }
        let stats = self.layer_profile_entry(layer);
        stats.state_init_calls = stats.state_init_calls.saturating_add(1);
        stats.state_init_us = stats.state_init_us.saturating_add(elapsed_us);
    }

    pub(crate) fn record_layer_prefill(&mut self, layer: usize, tokens: usize, elapsed_us: u64) {
        if !self.profile {
            return;
        }
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
        if !self.profile {
            return;
        }
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
        if !self.profile {
            return;
        }
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
        if !self.profile {
            return;
        }
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
        linear: &LinearWeight,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        linear.reference_matvec(input)
    }

    pub(crate) fn linear_rows(
        &mut self,
        linear: &LinearWeight,
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

    #[expect(
        clippy::too_many_arguments,
        reason = "the operator boundary keeps tensor views, dimensions, and kernel specification explicit"
    )]
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
        output_a: &LinearWeight,
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn routed_moe_step(
        &mut self,
        layer: usize,
        input: &[f32],
        token_id: u32,
        router: &RouterWeights,
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
        router: &RouterWeights,
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
    pub(crate) fn compile_cuda_execution_image(
        &mut self,
        generation: u64,
        resources: &DeepSeekV4PreparedResources,
    ) -> Result<()> {
        if self.backend != ModelExecutionBackend::Cuda {
            return Ok(());
        }
        self.cuda_mut()?
            .compile_execution_image(generation, resources)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn configure_expert_subsystem(
        &mut self,
        subsystem: DeepSeekV4SharedExpertSubsystem,
    ) -> Result<()> {
        if self.backend != ModelExecutionBackend::Cuda {
            return Err(Error::Execution(
                "DeepSeek-V4 shared expert subsystem requires the CUDA backend".into(),
            ));
        }
        self.cuda_mut()?.configure_expert_subsystem(subsystem)
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
