//! DeepSeek-V4 transformer layer: HC + attention + MoE + shared FFN.

use crate::artifact::binding::RouterArtifactPayload;
use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::{HyperConnectionConfig, HyperConnectionWeights};
use crate::moe::executor::ExpertExecutor;
use crate::moe::handle::CpuExpertHandleStore;
use crate::moe::routed::RoutedMoeStepOutput;
use crate::moe::routing::ExpertRouterPolicy;
use std::time::Instant;

use crate::moe::streaming::{ExpertStreamingPlanner, ExpertStreamingReader};
use ferrule_common::{Error, Result};

#[cfg(feature = "cuda")]
use super::attention::DeepSeekV4AttentionGraphArena;
use super::attention::{DeepSeekV4Attention, DeepSeekV4AttentionCache};
#[cfg(feature = "cuda")]
use super::config::DeepSeekV4AttentionConfig;
use super::helpers::rms_norm_rows_with_operators;
use super::operators::{
    DeepSeekV4LayerProfileStage, DeepSeekV4OperatorBackend, DeepSeekV4OperatorContext,
};

pub struct DeepSeekV4Layer {
    pub layer: usize,
    pub hc_config: HyperConnectionConfig,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub attention: DeepSeekV4Attention,
    pub hc_attention: HyperConnectionWeights,
    pub hc_feed_forward: HyperConnectionWeights,
    pub router: RouterArtifactPayload,
    pub shared_ffn: SwiGluFfnPayload,
    pub router_policy: ExpertRouterPolicy,
}

impl DeepSeekV4Layer {
    pub fn decode_step_reference(
        &self,
        state: &mut DeepSeekV4LayerState,
        hc_state: &[f32],
        token_id: u32,
        position: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        expert_executor: &impl ExpertExecutor,
    ) -> Result<DeepSeekV4LayerStepOutput> {
        self.decode_step_with_backend(
            state,
            hc_state,
            token_id,
            position,
            predicted_experts,
            expert_reader,
            expert_executor,
            DeepSeekV4OperatorBackend::Cpu,
        )
    }

    pub fn decode_step_with_backend(
        &self,
        state: &mut DeepSeekV4LayerState,
        hc_state: &[f32],
        token_id: u32,
        position: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        expert_executor: &impl ExpertExecutor,
        operator_backend: DeepSeekV4OperatorBackend,
    ) -> Result<DeepSeekV4LayerStepOutput> {
        let mut operators = DeepSeekV4OperatorContext::new(operator_backend)?;
        self.decode_step_with_operators(
            state,
            hc_state,
            token_id,
            position,
            predicted_experts,
            expert_reader,
            expert_executor,
            &mut operators,
        )
    }

    pub fn decode_step_with_operators(
        &self,
        state: &mut DeepSeekV4LayerState,
        hc_state: &[f32],
        token_id: u32,
        position: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        expert_executor: &impl ExpertExecutor,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<DeepSeekV4LayerStepOutput> {
        if hc_state.len() != self.hc_config.hc_hidden_size() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} HC input length mismatch: expected {}, got {}",
                self.layer,
                self.hc_config.hc_hidden_size(),
                hc_state.len()
            )));
        }

        // Device-resident path for CUDA: keeps norm weights on device and
        // avoids redundant H2D/D2H for rms_norm inputs/outputs.
        #[cfg(feature = "cuda")]
        if operators.backend == DeepSeekV4OperatorBackend::Cuda {
            return self.decode_step_device(
                state,
                hc_state,
                token_id,
                position,
                predicted_experts,
                expert_reader,
                expert_executor,
                operators,
            );
        }

        let attention_pre = operators.hc_pre(hc_state, 1, self.hc_config, &self.hc_attention)?;
        let attention_input = operators.rms_norm(
            &attention_pre.hidden,
            &self.attn_norm,
            self.hc_config.norm_eps,
            "attn_norm",
        )?;
        let attention_hidden = self.attention.decode_step_with_operators(
            &mut state.kv,
            &attention_input,
            position,
            operators,
        )?;
        let after_attention = operators.hc_post(
            &attention_hidden,
            hc_state,
            self.hc_config,
            &attention_pre.split,
        )?;

        let ffn_pre =
            operators.hc_pre(&after_attention, 1, self.hc_config, &self.hc_feed_forward)?;
        let ffn_input = operators.rms_norm(
            &ffn_pre.hidden,
            &self.ffn_norm,
            self.hc_config.norm_eps,
            "ffn_norm",
        )?;
        let moe = operators.routed_moe_step(
            self.layer,
            &ffn_input,
            token_id,
            &self.router,
            predicted_experts,
            &self.router_policy,
            &mut state.expert_planner,
            expert_reader,
            &mut state.expert_handles,
            expert_executor,
            Some(&self.shared_ffn),
        )?;
        let hc_state = operators.hc_post(
            &moe.output,
            &after_attention,
            self.hc_config,
            &ffn_pre.split,
        )?;

        Ok(DeepSeekV4LayerStepOutput {
            attention_hidden,
            feed_forward_hidden: moe.output.clone(),
            moe,
            hc_state,
        })
    }

    /// CUDA device-resident decode path.
    ///
    /// All HC pre/post ops run fully on device with cached weights.
    /// rms_norm also runs on device. Attention and MoE still need host
    /// boundaries (to be device-resident in follow-up work), but the
    /// HC boundary — the largest source of D2H sync points — is eliminated.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_step_device_hc_device(
        &self,
        state: &mut DeepSeekV4LayerState,
        hc_state_dev: &ferrule_cuda::context::CudaF32Buffer,
        token_id: u32,
        position: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<DeepSeekV4LayerDeviceStepOutput> {
        let decode_start = Instant::now();
        let norm_eps = self.hc_config.norm_eps;
        let layer_tag = format!("L{}", self.layer);

        let attn_hc_name = format!("hc_attn_{layer_tag}");
        let stage_start = Instant::now();
        let (attn_hidden_dev, _attn_pre_dev, attn_post_dev, attn_comb_dev) = operators
            .cuda_hc_pre_from_device(
                &attn_hc_name,
                hc_state_dev,
                &self.hc_attention,
                1,
                self.hc_config,
            )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnHcPre,
            stage_start,
        )?;

        let attn_norm_name = format!("attn_norm_{layer_tag}");
        let stage_start = Instant::now();
        let normed_dev = operators.cuda_rms_norm_device_cached(
            &attn_norm_name,
            &attn_hidden_dev,
            &self.attn_norm,
            norm_eps,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnNorm,
            stage_start,
        )?;

        // Device-resident attention: pass normed_dev directly, get device output.
        // Eliminates D2H (normed_dev → host) + H2D (host → hidden_device) +
        // D2H (output → host) + H2D (attention_hidden → attn_hidden_out_dev).
        let stage_start = Instant::now();
        let attn_hidden_out_dev = self.attention.decode_step_from_device(
            &mut state.kv,
            &normed_dev,
            position,
            operators,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::Attention,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let after_attn_dev = operators.cuda_hc_post_from_device(
            &attn_hidden_out_dev,
            hc_state_dev,
            &attn_post_dev,
            &attn_comb_dev,
            1,
            self.hc_config,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnHcPost,
            stage_start,
        )?;

        let ffn_hc_name = format!("hc_ffn_{layer_tag}");
        let stage_start = Instant::now();
        let (ffn_hidden_dev, _ffn_pre_dev, ffn_post_dev, ffn_comb_dev) = operators
            .cuda_hc_pre_from_device(
                &ffn_hc_name,
                &after_attn_dev,
                &self.hc_feed_forward,
                1,
                self.hc_config,
            )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnHcPre,
            stage_start,
        )?;

        let ffn_norm_name = format!("ffn_norm_{layer_tag}");
        let stage_start = Instant::now();
        let normed_dev = operators.cuda_rms_norm_device_cached(
            &ffn_norm_name,
            &ffn_hidden_dev,
            &self.ffn_norm,
            norm_eps,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnNorm,
            stage_start,
        )?;
        if !predicted_experts.is_empty() {
            operators.prefetch_predicted_experts(
                self.layer,
                predicted_experts,
                &mut state.expert_planner,
                expert_reader,
                &mut state.expert_handles,
            )?;
        }
        let stage_start = Instant::now();
        let moe_device = operators.cuda_routed_moe_step_device_output(
            self.layer,
            &normed_dev,
            token_id,
            &self.router,
            predicted_experts,
            &self.router_policy,
            &mut state.expert_planner,
            expert_reader,
            &mut state.expert_handles,
            Some(&self.shared_ffn),
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::Moe,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let out = operators.cuda_hc_post_from_device(
            &moe_device.output_dev,
            &after_attn_dev,
            &ffn_post_dev,
            &ffn_comb_dev,
            1,
            self.hc_config,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnHcPost,
            stage_start,
        )?;
        let decode_us = operators.finish_profile_stage(decode_start)?;
        operators.record_layer_decode(self.layer, decode_us);
        Ok(DeepSeekV4LayerDeviceStepOutput {
            hc_state: out,
            moe: moe_device.moe,
        })
    }

    /// Graph-safe decode step: only kernel launches, no D2H, no host
    /// computation. Uses pre-determined routes and pre-computed topk.
    /// Safe for CUDA graph capture.
    ///
    /// `route_experts` and `route_weights` come from the warmup pass.
    #[cfg(feature = "cuda")]
    pub(crate) fn decode_step_graph_safe(
        &self,
        state: &mut DeepSeekV4LayerState,
        hc_state_dev: &ferrule_cuda::context::CudaF32Buffer,
        position: usize,
        route_experts: &[usize],
        route_weights: &[f32],
        operators: &mut DeepSeekV4OperatorContext,
        moe_accumulator: &mut ferrule_cuda::context::CudaF32Buffer,
        output_hc_state: &mut ferrule_cuda::context::CudaF32Buffer,
    ) -> Result<()> {
        let norm_eps = self.hc_config.norm_eps;
        let layer_tag = format!("L{}", self.layer);

        state.ensure_graph_arena(self.hc_config, self.attention.config, operators)?;
        let kv = &state.kv;
        let arena = state.graph_arena.as_mut().expect("initialized above");

        // HC pre (kernel-only, idempotent weights) into stable graph arena buffers.
        let attn_hc_name = format!("hc_attn_{layer_tag}");
        operators.cuda_hc_pre_from_device_into(
            &attn_hc_name,
            hc_state_dev,
            &self.hc_attention,
            1,
            self.hc_config,
            &mut arena.attn_hidden,
            &mut arena.attn_pre,
            &mut arena.attn_post,
            &mut arena.attn_comb,
        )?;

        // Attention norm (kernel-only) into a stable graph arena buffer.
        let attn_norm_name = format!("attn_norm_{layer_tag}");
        operators.cuda_rms_norm_device_cached_into(
            &attn_norm_name,
            &arena.attn_hidden,
            &self.attn_norm,
            norm_eps,
            &mut arena.attn_norm,
        )?;

        // Attention graph path uses the warmup-prepared CUDA KV caches and writes into its arena.
        self.attention
            .decode_step_graph_safe_from_prepared_cache(
                kv,
                &arena.attn_norm,
                position,
                operators,
                &mut arena.attention,
            )
            .map_err(|err| {
                Error::Internal(format!(
                    "DSV4 layer {} graph-safe attention failed: {err}",
                    self.layer
                ))
            })?;

        // HC post (kernel-only) into stable graph arena buffer.
        operators.cuda_hc_post_from_device_into(
            arena.attention.output(),
            hc_state_dev,
            &arena.attn_post,
            &arena.attn_comb,
            1,
            self.hc_config,
            &mut arena.after_attn,
        )?;

        // FFN HC pre (kernel-only) into stable graph arena buffers.
        let ffn_hc_name = format!("hc_ffn_{layer_tag}");
        operators.cuda_hc_pre_from_device_into(
            &ffn_hc_name,
            &arena.after_attn,
            &self.hc_feed_forward,
            1,
            self.hc_config,
            &mut arena.ffn_hidden,
            &mut arena.ffn_pre,
            &mut arena.ffn_post,
            &mut arena.ffn_comb,
        )?;

        // FFN norm (kernel-only) into a stable graph arena buffer.
        let ffn_norm_name = format!("ffn_norm_{layer_tag}");
        operators.cuda_rms_norm_device_cached_into(
            &ffn_norm_name,
            &arena.ffn_hidden,
            &self.ffn_norm,
            norm_eps,
            &mut arena.ffn_norm,
        )?;

        // Graph-safe MoE (kernel-only, pre-determined routes) into caller-owned accumulator.
        operators.cuda_routed_moe_graph_safe(
            &arena.ffn_norm,
            route_experts,
            route_weights,
            self.layer,
            Some(&self.shared_ffn),
            moe_accumulator,
        )?;

        // Final HC post writes into a caller-owned ping-pong HC slot that is held by the graph.
        operators.cuda_hc_post_from_device_into(
            moe_accumulator,
            &arena.after_attn,
            &arena.ffn_post,
            &arena.ffn_comb,
            1,
            self.hc_config,
            output_hc_state,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn decode_step_device(
        &self,
        state: &mut DeepSeekV4LayerState,
        hc_state: &[f32],
        token_id: u32,
        position: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        _expert_executor: &impl ExpertExecutor,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<DeepSeekV4LayerStepOutput> {
        let norm_eps = self.hc_config.norm_eps;
        let layer_tag = format!("L{}", self.layer);

        // --- Upload hc_state to device once, reuse for hc_pre + hc_post ---
        let hc_state_dev = operators.cuda_upload_f32(hc_state)?;

        // --- Attention block: hc_pre on device ---
        let attn_hc_name = format!("hc_attn_{layer_tag}");
        let (attn_hidden_dev, _attn_pre_dev, attn_post_dev, attn_comb_dev) = operators
            .cuda_hc_pre_from_device(
                &attn_hc_name,
                &hc_state_dev,
                &self.hc_attention,
                1,
                self.hc_config,
            )?;

        // rms_norm on device, then device-resident attention (no D2H/H2D round-trip).
        let attn_norm_name = format!("attn_norm_{layer_tag}");
        let normed_dev = operators.cuda_rms_norm_device_cached(
            &attn_norm_name,
            &attn_hidden_dev,
            &self.attn_norm,
            norm_eps,
        )?;

        let attn_hidden_out_dev = self.attention.decode_step_from_device(
            &mut state.kv,
            &normed_dev,
            position,
            operators,
        )?;

        // hc_post on device: reuse hc_state_dev as residual
        let after_attn_dev = operators.cuda_hc_post_from_device(
            &attn_hidden_out_dev,
            &hc_state_dev,
            &attn_post_dev,
            &attn_comb_dev,
            1,
            self.hc_config,
        )?;

        // --- FFN/MoE block: hc_pre on device ---
        let ffn_hc_name = format!("hc_ffn_{layer_tag}");
        let (ffn_hidden_dev, _ffn_pre_dev, ffn_post_dev, ffn_comb_dev) = operators
            .cuda_hc_pre_from_device(
                &ffn_hc_name,
                &after_attn_dev,
                &self.hc_feed_forward,
                1,
                self.hc_config,
            )?;

        let ffn_norm_name = format!("ffn_norm_{layer_tag}");
        let normed_dev = operators.cuda_rms_norm_device_cached(
            &ffn_norm_name,
            &ffn_hidden_dev,
            &self.ffn_norm,
            norm_eps,
        )?;
        if !predicted_experts.is_empty() {
            operators.prefetch_predicted_experts(
                self.layer,
                predicted_experts,
                &mut state.expert_planner,
                expert_reader,
                &mut state.expert_handles,
            )?;
        }
        let moe_device = operators.cuda_routed_moe_step_device_output(
            self.layer,
            &normed_dev,
            token_id,
            &self.router,
            predicted_experts,
            &self.router_policy,
            &mut state.expert_planner,
            expert_reader,
            &mut state.expert_handles,
            Some(&self.shared_ffn),
        )?;

        // hc_post stays fully device-resident for MoE output.
        let new_hc_state_dev = operators.cuda_hc_post_from_device(
            &moe_device.output_dev,
            &after_attn_dev,
            &ffn_post_dev,
            &ffn_comb_dev,
            1,
            self.hc_config,
        )?;
        let hc_state = operators.cuda_download_f32(&new_hc_state_dev)?;

        Ok(DeepSeekV4LayerStepOutput {
            attention_hidden: Vec::new(),
            feed_forward_hidden: Vec::new(),
            moe: moe_device.moe,
            hc_state,
        })
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn prefill_start_cuda_device_bridge(
        &self,
        state: &mut DeepSeekV4LayerState,
        hc_state: &[f32],
        token_ids: &[u32],
        start_pos: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        _expert_executor: &impl ExpertExecutor,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        let hc_state_dev = operators.cuda_upload_f32(hc_state)?;
        let out_dev = self.prefill_start_cuda_device_chain(
            state,
            &hc_state_dev,
            token_ids,
            start_pos,
            predicted_experts,
            expert_reader,
            operators,
        )?;
        operators.cuda_download_f32(&out_dev)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_start_cuda_device_chain(
        &self,
        state: &mut DeepSeekV4LayerState,
        hc_state_dev: &ferrule_cuda::context::CudaF32Buffer,
        token_ids: &[u32],
        start_pos: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let tokens = token_ids.len();
        let prefill_start = Instant::now();
        let layer_tag = format!("L{}", self.layer);

        let stage_start = Instant::now();
        let attn_hc_name = format!("hc_attn_{layer_tag}");
        let (attn_hidden_dev, _attn_pre_dev, attn_post_dev, attn_comb_dev) = operators
            .cuda_hc_pre_from_device(
                &attn_hc_name,
                &hc_state_dev,
                &self.hc_attention,
                tokens,
                self.hc_config,
            )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnHcPre,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let attn_norm_name = format!("attn_norm_{layer_tag}");
        let attention_input_dev = operators.cuda_rms_norm_rows_device_cached(
            &attn_norm_name,
            &attn_hidden_dev,
            tokens,
            &self.attn_norm,
            self.hc_config.norm_eps,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnNorm,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let attention_hidden_dev = self.attention.prefill_start_from_device(
            &mut state.kv,
            &attention_input_dev,
            start_pos,
            operators,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::Attention,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let after_attention_dev = operators.cuda_hc_post_from_device(
            &attention_hidden_dev,
            &hc_state_dev,
            &attn_post_dev,
            &attn_comb_dev,
            tokens,
            self.hc_config,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnHcPost,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let ffn_hc_name = format!("hc_ffn_{layer_tag}");
        let (ffn_hidden_dev, _ffn_pre_dev, ffn_post_dev, ffn_comb_dev) = operators
            .cuda_hc_pre_from_device(
                &ffn_hc_name,
                &after_attention_dev,
                &self.hc_feed_forward,
                tokens,
                self.hc_config,
            )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnHcPre,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let ffn_norm_name = format!("ffn_norm_{layer_tag}");
        let ffn_input_dev = operators.cuda_rms_norm_rows_device_cached(
            &ffn_norm_name,
            &ffn_hidden_dev,
            tokens,
            &self.ffn_norm,
            self.hc_config.norm_eps,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnNorm,
            stage_start,
        )?;
        if !predicted_experts.is_empty() {
            operators.prefetch_predicted_experts(
                self.layer,
                predicted_experts,
                &mut state.expert_planner,
                expert_reader,
                &mut state.expert_handles,
            )?;
        }

        let stage_start = Instant::now();
        let feed_forward_hidden_dev = operators.routed_moe_prefill_batch_from_device(
            self.layer,
            &ffn_input_dev,
            token_ids,
            &self.router,
            predicted_experts,
            &self.router_policy,
            &mut state.expert_planner,
            expert_reader,
            &mut state.expert_handles,
            Some(&self.shared_ffn),
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::Moe,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let out_dev = operators.cuda_hc_post_from_device(
            &feed_forward_hidden_dev,
            &after_attention_dev,
            &ffn_post_dev,
            &ffn_comb_dev,
            tokens,
            self.hc_config,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnHcPost,
            stage_start,
        )?;

        let prefill_us = operators.finish_profile_stage(prefill_start)?;
        operators.record_layer_prefill(self.layer, tokens, prefill_us);
        Ok(out_dev)
    }

    pub fn prefill_start_with_operators(
        &self,
        state: &mut DeepSeekV4LayerState,
        hc_state: &[f32],
        token_ids: &[u32],
        start_pos: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        expert_executor: &impl ExpertExecutor,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Vec<f32>> {
        let tokens = token_ids.len();
        if tokens == 0 {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} prefill requires at least one token",
                self.layer
            )));
        }
        if hc_state.len() != tokens * self.hc_config.hc_hidden_size() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {} batched HC input length mismatch: expected {}, got {}",
                self.layer,
                tokens * self.hc_config.hc_hidden_size(),
                hc_state.len()
            )));
        }

        if matches!(operators.backend(), DeepSeekV4OperatorBackend::Cuda) {
            #[cfg(feature = "cuda")]
            {
                return self.prefill_start_cuda_device_bridge(
                    state,
                    hc_state,
                    token_ids,
                    start_pos,
                    predicted_experts,
                    expert_reader,
                    expert_executor,
                    operators,
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(Error::Model(
                    "DeepSeek-V4 CUDA prefill requires cuda feature".into(),
                ));
            }
        }

        let prefill_start = Instant::now();
        let stage_start = Instant::now();
        let attention_pre =
            operators.hc_pre(hc_state, tokens, self.hc_config, &self.hc_attention)?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnHcPre,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let attn_norm_name = format!("attn_norm_L{}", self.layer);
        let attention_input = rms_norm_rows_with_operators(
            operators,
            &attention_pre.hidden,
            tokens,
            &self.attn_norm,
            self.hc_config.norm_eps,
            &attn_norm_name,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let attention_hidden = if start_pos == 0 {
            self.attention.prefill_start_with_operators(
                &mut state.kv,
                &attention_input,
                start_pos,
                operators,
            )?
        } else {
            self.attention.prefill_segment_with_operators(
                &mut state.kv,
                &attention_input,
                start_pos,
                operators,
            )?
        };
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::Attention,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let after_attention = operators.hc_post(
            &attention_hidden,
            hc_state,
            self.hc_config,
            &attention_pre.split,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnHcPost,
            stage_start,
        )?;

        let stage_start = Instant::now();
        let ffn_pre = operators.hc_pre(
            &after_attention,
            tokens,
            self.hc_config,
            &self.hc_feed_forward,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnHcPre,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let ffn_norm_name = format!("ffn_norm_L{}", self.layer);
        let ffn_input = rms_norm_rows_with_operators(
            operators,
            &ffn_pre.hidden,
            tokens,
            &self.ffn_norm,
            self.hc_config.norm_eps,
            &ffn_norm_name,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let feed_forward_hidden = operators.routed_moe_prefill_batch(
            self.layer,
            &ffn_input,
            token_ids,
            &self.router,
            predicted_experts,
            &self.router_policy,
            &mut state.expert_planner,
            expert_reader,
            &mut state.expert_handles,
            expert_executor,
            Some(&self.shared_ffn),
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::Moe,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let out = operators.hc_post(
            &feed_forward_hidden,
            &after_attention,
            self.hc_config,
            &ffn_pre.split,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnHcPost,
            stage_start,
        )?;
        let prefill_us = operators.finish_profile_stage(prefill_start)?;
        operators.record_layer_prefill(self.layer, tokens, prefill_us);
        Ok(out)
    }
}

pub struct DeepSeekV4LayerState {
    pub kv: DeepSeekV4AttentionCache,
    pub expert_planner: ExpertStreamingPlanner,
    pub expert_handles: CpuExpertHandleStore,
    #[cfg(feature = "cuda")]
    pub(crate) graph_arena: Option<DeepSeekV4LayerGraphArena>,
}

impl Clone for DeepSeekV4LayerState {
    fn clone(&self) -> Self {
        Self {
            kv: self.kv.clone(),
            expert_planner: self.expert_planner.clone(),
            expert_handles: self.expert_handles.clone(),
            #[cfg(feature = "cuda")]
            graph_arena: None,
        }
    }
}

impl std::fmt::Debug for DeepSeekV4LayerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("DeepSeekV4LayerState");
        debug
            .field("kv", &self.kv)
            .field("expert_planner", &self.expert_planner)
            .field("expert_handles", &self.expert_handles);
        #[cfg(feature = "cuda")]
        debug.field("graph_arena", &self.graph_arena);
        debug.finish()
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4LayerGraphArena {
    attn_hidden: ferrule_cuda::context::CudaF32Buffer,
    attn_pre: ferrule_cuda::context::CudaF32Buffer,
    attn_post: ferrule_cuda::context::CudaF32Buffer,
    attn_comb: ferrule_cuda::context::CudaF32Buffer,
    attn_norm: ferrule_cuda::context::CudaF32Buffer,
    after_attn: ferrule_cuda::context::CudaF32Buffer,
    ffn_hidden: ferrule_cuda::context::CudaF32Buffer,
    ffn_pre: ferrule_cuda::context::CudaF32Buffer,
    ffn_post: ferrule_cuda::context::CudaF32Buffer,
    ffn_comb: ferrule_cuda::context::CudaF32Buffer,
    ffn_norm: ferrule_cuda::context::CudaF32Buffer,
    attention: DeepSeekV4AttentionGraphArena,
    hidden_size: usize,
    hc_mult: usize,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeepSeekV4LayerGraphArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepSeekV4LayerGraphArena")
            .field("hidden_size", &self.hidden_size)
            .field("hc_mult", &self.hc_mult)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda")]
impl DeepSeekV4LayerGraphArena {
    fn is_compatible(
        &self,
        config: HyperConnectionConfig,
        attention_config: DeepSeekV4AttentionConfig,
    ) -> bool {
        self.hidden_size == config.hidden_size
            && self.hc_mult == config.hc_mult
            && self.attention.is_compatible(attention_config)
    }

    fn new(
        config: HyperConnectionConfig,
        attention_config: DeepSeekV4AttentionConfig,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Self> {
        let hidden = config.hidden_size;
        let hc = config.hc_mult;
        let hc_dim = hc
            .checked_mul(hidden)
            .ok_or_else(|| Error::Internal("DSV4 graph arena HC dim overflow".into()))?;
        let comb = hc
            .checked_mul(hc)
            .ok_or_else(|| Error::Internal("DSV4 graph arena HC comb overflow".into()))?;
        Ok(Self {
            attn_hidden: operators.cuda_zero_f32(hidden)?,
            attn_pre: operators.cuda_zero_f32(hc)?,
            attn_post: operators.cuda_zero_f32(hc)?,
            attn_comb: operators.cuda_zero_f32(comb)?,
            attn_norm: operators.cuda_zero_f32(hidden)?,
            after_attn: operators.cuda_zero_f32(hc_dim)?,
            ffn_hidden: operators.cuda_zero_f32(hidden)?,
            ffn_pre: operators.cuda_zero_f32(hc)?,
            ffn_post: operators.cuda_zero_f32(hc)?,
            ffn_comb: operators.cuda_zero_f32(comb)?,
            ffn_norm: operators.cuda_zero_f32(hidden)?,
            attention: DeepSeekV4AttentionGraphArena::new(attention_config, operators)?,
            hidden_size: hidden,
            hc_mult: hc,
        })
    }
}

impl DeepSeekV4LayerState {
    #[cfg(feature = "cuda")]
    pub(crate) fn ensure_graph_arena(
        &mut self,
        config: HyperConnectionConfig,
        attention_config: DeepSeekV4AttentionConfig,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<&mut DeepSeekV4LayerGraphArena> {
        let needs_new = self
            .graph_arena
            .as_ref()
            .map(|arena| !arena.is_compatible(config, attention_config))
            .unwrap_or(true);
        if needs_new {
            self.graph_arena = Some(DeepSeekV4LayerGraphArena::new(
                config,
                attention_config,
                operators,
            )?);
        }
        Ok(self.graph_arena.as_mut().expect("initialized above"))
    }

    pub fn reset_sequence(&mut self) {
        self.reset_sequence_with_expert_residency(false);
    }

    pub fn reset_sequence_with_expert_residency(&mut self, preserve_expert_residency: bool) {
        self.kv.reset_sequence();
        self.expert_handles.clear();
        #[cfg(feature = "cuda")]
        {
            self.graph_arena = None;
        }
        if !preserve_expert_residency {
            self.expert_planner.clear_residency();
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4LayerStepOutput {
    pub attention_hidden: Vec<f32>,
    pub feed_forward_hidden: Vec<f32>,
    pub moe: RoutedMoeStepOutput,
    pub hc_state: Vec<f32>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4LayerDeviceStepOutput {
    pub hc_state: ferrule_cuda::context::CudaF32Buffer,
    pub moe: RoutedMoeStepOutput,
}

fn record_stage(
    operators: &mut DeepSeekV4OperatorContext,
    layer: usize,
    stage: DeepSeekV4LayerProfileStage,
    start: Instant,
) -> Result<()> {
    let elapsed_us = operators.finish_profile_stage(start)?;
    operators.record_layer_stage(layer, stage, elapsed_us);
    Ok(())
}
