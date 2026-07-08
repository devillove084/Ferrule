//! DeepSeek-V4 CUDA operator cache: device-resident weights, KV cache, MoE handles.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
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
use crate::moe::routed::RoutedMoeStepOutput;
use crate::moe::routing::ExpertRouterPolicy;
use crate::moe::streaming::{
    read_experts_concurrent, ExpertComputeBundle, ExpertId, ExpertLinearFormat,
    ExpertLinearPayload, ExpertStorageTier, ExpertStreamingPlanner, ExpertStreamingReader,
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
    decode_arena: DeepSeekV4DecodeArena,
    host_staged_cache: HostStagedExpertCache,
    norm_weights: HashMap<String, ferrule_cuda::context::CudaF32Buffer>,
    /// Cached HC weights: function, scale, base — uploaded once per layer.
    hc_weights: HashMap<String, HcDeviceWeights>,
    /// Cached HC head weights: function, scale, base — uploaded once for the
    /// terminal HC head projection.
    hc_head_weights: Option<HcDeviceWeights>,
    /// Cached attention sink buffers, keyed by layer tag — uploaded once per
    /// layer and reused across decode steps.
    pub(crate) sink_buffers: HashMap<String, ferrule_cuda::context::CudaF32Buffer>,
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
    moe_cache_lookup_us: u64,
    moe_expert_read_us: u64,
    moe_expert_upload_us: u64,
    moe_shared_us: u64,
    moe_workspace_us: u64,
    moe_compute_submit_us: u64,
    moe_commit_us: u64,
    expert_selected: u64,
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
    /// Pre-allocated MoE accumulator, reused across graph replays.
    moe_accumulator: Option<ferrule_cuda::context::CudaF32Buffer>,
    /// Reusable routed MoE scratch/pointer buffers for the decode hot path.
    moe_workspace: Option<ferrule_cuda::context::CudaMoeBatchedWorkspace>,
}

#[cfg(feature = "cuda")]
struct CudaFp4ExpertHandles {
    gate: ferrule_cuda::context::CudaArtifactLinearHandle,
    up: ferrule_cuda::context::CudaArtifactLinearHandle,
    down: ferrule_cuda::context::CudaArtifactLinearHandle,
    bytes: u64,
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
            decode_arena: DeepSeekV4DecodeArena::default(),
            host_staged_cache: HostStagedExpertCache::new(256),
            norm_weights: HashMap::new(),
            hc_weights: HashMap::new(),
            hc_head_weights: None,
            sink_buffers: HashMap::new(),
            grouped_wo_a_weights: HashMap::new(),
            kv_cache: HashMap::new(),
            kv_len: HashMap::new(),
            combined_kv_cache: HashMap::new(),
            combined_kv_compressed_capacity: HashMap::new(),
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
            moe_cache_lookup_us: 0,
            moe_expert_read_us: 0,
            moe_expert_upload_us: 0,
            moe_shared_us: 0,
            moe_workspace_us: 0,
            moe_compute_submit_us: 0,
            moe_commit_us: 0,
            expert_selected: 0,
            expert_loads: 0,
            expert_load_bytes: 0,
            expert_evictions: 0,
        })
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

    pub(crate) fn reset_runtime_counters(&mut self) {
        self.ops.reset_counters();
        self.output_head_calls = 0;
        self.output_head_chunks = 0;
        self.output_head_rows = 0;
        self.output_head_cache_hits = 0;
        self.output_head_cache_misses = 0;
        self.output_head_hidden_uploads = 0;
        self.output_head_hidden_upload_us = 0;
        self.output_head_read_us = 0;
        self.output_head_upload_us = 0;
        self.output_head_topk_us = 0;
        self.output_head_merge_us = 0;
        self.moe_router_us = 0;
        self.moe_routing_us = 0;
        self.moe_plan_us = 0;
        self.moe_cache_lookup_us = 0;
        self.moe_expert_read_us = 0;
        self.moe_expert_upload_us = 0;
        self.moe_shared_us = 0;
        self.moe_workspace_us = 0;
        self.moe_compute_submit_us = 0;
        self.moe_commit_us = 0;
        self.expert_selected = 0;
        self.expert_loads = 0;
        self.expert_load_bytes = 0;
        self.expert_evictions = 0;
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
            moe_cache_lookup_us: self.moe_cache_lookup_us,
            moe_expert_read_us: self.moe_expert_read_us,
            moe_expert_upload_us: self.moe_expert_upload_us,
            moe_shared_us: self.moe_shared_us,
            moe_workspace_us: self.moe_workspace_us,
            moe_compute_submit_us: self.moe_compute_submit_us,
            moe_commit_us: self.moe_commit_us,
            expert_selected: self.expert_selected,
            expert_loads: self.expert_loads,
            expert_load_bytes: self.expert_load_bytes,
            expert_evictions: self.expert_evictions,
            expert_host_cache_hits: self.host_staged_cache.hits(),
            expert_host_cache_misses: self.host_staged_cache.misses(),
            expert_host_cache_evictions: self.host_staged_cache.evictions(),
            expert_host_cache_entries: self.host_staged_cache.len(),
            expert_host_cache_bytes: self.host_staged_cache.total_bytes(),
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
        let mut output = self.ops.zero_f32_buffer(handle.shape().out_features())?;
        self.ops
            .artifact_linear_matvec_into(handle, input, &mut output)?;
        Ok(output)
    }

    /// Graph-safe variant: writes into a pre-allocated output buffer
    /// instead of allocating a new one. Safe for CUDA graph capture.
    pub(crate) fn linear_matvec_into_from_device(
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
        for eviction in &streaming.evictions {
            self.experts.remove(&eviction.expert);
        }
        self.moe_cache_lookup_us = self
            .moe_cache_lookup_us
            .saturating_add(duration_us(stage_start.elapsed()));

        for load in &streaming.loads {
            // Host-staged cache: serve from host RAM on re-activation,
            // skipping the disk read. `expert_loads` intentionally counts
            // source/disk misses, not every GPU upload.
            let misses_before = self.host_staged_cache.misses();
            let stage_start = Instant::now();
            let bundle =
                self.host_staged_cache
                    .get_or_load(load.expert, &load.load_source, reader)?;
            self.moe_expert_read_us = self
                .moe_expert_read_us
                .saturating_add(duration_us(stage_start.elapsed()));
            let cache_miss = self.host_staged_cache.misses() != misses_before;
            let stage_start = Instant::now();
            let expert = self.upload_expert_bundle(&bundle)?;
            let bytes = expert.bytes;
            if cache_miss {
                self.expert_loads = self.expert_loads.saturating_add(1);
                self.expert_load_bytes = self.expert_load_bytes.saturating_add(bytes);
            }
            self.experts.insert(load.expert, expert);
            handles.insert_resident_handle(ResidentExpertHandle::new(
                load.expert,
                ExpertStorageTier::Gpu,
                ExpertResidentFormat::Opaque("cuda-fp4-artifact".into()),
                bytes,
            ))?;
            self.moe_expert_upload_us = self
                .moe_expert_upload_us
                .saturating_add(duration_us(stage_start.elapsed()));
        }

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
            planner.commit_step(&streaming)?;
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
        planner.commit_step(&streaming)?;
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
        // Clone the input buffer for the router linear (avoids consuming the
        // caller's buffer). Uses device-to-device copy — no host round-trip.
        let stage_start = Instant::now();
        let mut router_input = self.ops.clone_f32_buffer(input)?;
        let logits_dev = self.linear_matvec_from_device(&router.weight, &mut router_input)?;
        // Download router logits (small: ~37 floats for DSV4 Flash).
        // This is the only D2H sync in the MoE path — the routing logic
        // (SqrtSoftplus scoring, top-k, normalization) is too complex for
        // a simple kernel and runs on CPU.
        let logits = self.ops.download_f32_buffer(&logits_dev)?;
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
        for eviction in &streaming.evictions {
            self.experts.remove(&eviction.expert);
        }
        self.moe_cache_lookup_us = self
            .moe_cache_lookup_us
            .saturating_add(duration_us(stage_start.elapsed()));

        // Concurrent disk reads: read all missing experts in parallel,
        // then upload serially (GPU upload must be sequential on one stream).
        if !streaming.loads.is_empty() {
            // Phase 1: check cache, collect cache-miss experts for concurrent read.
            let stage_start = Instant::now();
            let mut cached_bundles = Vec::with_capacity(streaming.loads.len());
            let mut miss_loads = Vec::new();
            for load in &streaming.loads {
                if let Some(bundle) = self.host_staged_cache.get(load.expert) {
                    cached_bundles.push((load.expert, Some(bundle)));
                } else {
                    cached_bundles.push((load.expert, None));
                    miss_loads.push((load.expert, load.load_source.clone()));
                }
            }
            self.moe_cache_lookup_us = self
                .moe_cache_lookup_us
                .saturating_add(duration_us(stage_start.elapsed()));

            // Phase 2: concurrently read all cache-miss experts from disk.
            let stage_start = Instant::now();
            let read_payloads = if miss_loads.is_empty() {
                Vec::new()
            } else {
                read_experts_concurrent(reader, &miss_loads)?
            };
            self.moe_expert_read_us = self
                .moe_expert_read_us
                .saturating_add(duration_us(stage_start.elapsed()));

            // Phase 3: insert into cache, upload to GPU, install handles.
            let stage_start = Instant::now();
            let mut payload_iter = read_payloads.into_iter();
            for (expert_id, cached) in &cached_bundles {
                let bundle = if let Some(b) = cached {
                    b.clone()
                } else {
                    let payload = payload_iter
                        .next()
                        .ok_or_else(|| {
                            Error::Internal(format!(
                                "concurrent expert read returned fewer payloads than expected for layer {}",
                                layer
                            ))
                        })?;
                    let bundle = ExpertComputeBundle::from_artifact_payload(payload)?;
                    self.host_staged_cache.insert(bundle.clone());
                    bundle
                };
                let expert = self.upload_expert_bundle(&bundle)?;
                let bytes = expert.bytes;
                self.expert_loads = self.expert_loads.saturating_add(1);
                self.expert_load_bytes = self.expert_load_bytes.saturating_add(bytes);
                self.experts.insert(*expert_id, expert);
                handles.insert_resident_handle(ResidentExpertHandle::new(
                    *expert_id,
                    ExpertStorageTier::Gpu,
                    ExpertResidentFormat::Opaque("cuda-fp4-artifact".into()),
                    bytes,
                ))?;
            }
            self.moe_expert_upload_us = self
                .moe_expert_upload_us
                .saturating_add(duration_us(stage_start.elapsed()));
        }

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
        planner.commit_step(&streaming)?;
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

    pub(crate) fn upload_expert_linear(
        &self,
        linear: &ExpertLinearPayload,
    ) -> Result<ferrule_cuda::context::CudaArtifactLinearHandle> {
        let ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
            out_features,
            in_features,
            block_size: 32,
        } = linear.format
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
        let shape = ferrule_cuda::transformer::sparse_attention::CudaSparseAttentionShape {
            batch_size: 1,
            tokens_per_batch: tokens,
            kv_len,
            heads: spec.heads,
            head_dim: spec.head_dim,
            topk: spec.topk,
            softmax_scale: spec.softmax_scale,
        };
        let topk_dev = self.ops.upload_i32_buffer(&topk_i32)?;
        let sink_name = format!("sink_L{layer}");
        self.ensure_sink_buffer(&sink_name, sink)?;
        let sink_buf = self.sink_buffers.get(&sink_name).expect("inserted above");
        self.ops.sparse_attention_sink_from_device(
            query,
            values,
            topk_dev.as_device_buffer(),
            sink_buf,
            shape,
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
        let shape = ferrule_cuda::transformer::sparse_attention::CudaSparseAttentionShape {
            batch_size: 1,
            tokens_per_batch: tokens,
            kv_len,
            heads: spec.heads,
            head_dim: spec.head_dim,
            topk: spec.topk,
            softmax_scale: spec.softmax_scale,
        };
        let topk_dev = self.ops.upload_i32_buffer(&topk_i32)?;
        let sink_name = format!("sink_L{layer}");
        self.ensure_sink_buffer(&sink_name, sink)?;
        let sink_buf = self.sink_buffers.get(&sink_name).expect("inserted above");
        let values = self.combined_kv_values_device(layer)?;
        self.ops.sparse_attention_sink_from_device(
            query,
            values,
            topk_dev.as_device_buffer(),
            sink_buf,
            shape,
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
        check_linear(
            layer,
            "wo_a",
            output_a,
            cfg.output_latent_dim(),
            cfg.output_group_input_dim(),
        )?;
        if context.len() != cfg.q_full_dim() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} CUDA grouped wo_a device context length mismatch: expected {}, got {}",
                cfg.q_full_dim(),
                context.len()
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
        self.ops.grouped_matvec_f32_from_device(
            context,
            weight_buf,
            cfg.output_latent_dim(),
            group_in,
            cfg.o_lora_rank,
        )
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
        if self.rope_cos.contains_key(name) {
            return Ok(());
        }
        let rd2 = rope_dim / 2;
        let mut cos = vec![0.0f32; max_positions * rd2];
        let mut sin = vec![0.0f32; max_positions * rd2];
        let rope = DeepSeekV4RopeParams::plain(rope_theta);
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
