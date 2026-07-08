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

use super::attention::{DeepSeekV4Attention, DeepSeekV4AttentionCache};
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
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
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
        Ok(out)
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
    ) -> Result<ferrule_cuda::context::CudaF32Buffer> {
        let norm_eps = self.hc_config.norm_eps;
        let layer_tag = format!("L{}", self.layer);

        // HC pre (kernel-only, idempotent weights).
        let attn_hc_name = format!("hc_attn_{layer_tag}");
        let (attn_hidden_dev, _attn_pre_dev, attn_post_dev, attn_comb_dev) = operators
            .cuda_hc_pre_from_device(
                &attn_hc_name,
                hc_state_dev,
                &self.hc_attention,
                1,
                self.hc_config,
            )?;

        // Attention norm (kernel-only).
        let attn_norm_name = format!("attn_norm_{layer_tag}");
        let normed_dev = operators.cuda_rms_norm_device_cached(
            &attn_norm_name,
            &attn_hidden_dev,
            &self.attn_norm,
            norm_eps,
        )?;

        // Attention: use the existing device-resident path (P0).
        // With RELAXED capture mode, allocations during capture are allowed.
        let attn_hidden_out_dev = self.attention.decode_step_from_device(
            &mut state.kv,
            &normed_dev,
            position,
            operators,
        )?;

        // HC post (kernel-only).
        let after_attn_dev = operators.cuda_hc_post_from_device(
            &attn_hidden_out_dev,
            hc_state_dev,
            &attn_post_dev,
            &attn_comb_dev,
            1,
            self.hc_config,
        )?;

        // FFN HC pre (kernel-only).
        let ffn_hc_name = format!("hc_ffn_{layer_tag}");
        let (ffn_hidden_dev, _ffn_pre_dev, ffn_post_dev, ffn_comb_dev) = operators
            .cuda_hc_pre_from_device(
                &ffn_hc_name,
                &after_attn_dev,
                &self.hc_feed_forward,
                1,
                self.hc_config,
            )?;

        // FFN norm (kernel-only).
        let ffn_norm_name = format!("ffn_norm_{layer_tag}");
        let normed_dev = operators.cuda_rms_norm_device_cached(
            &ffn_norm_name,
            &ffn_hidden_dev,
            &self.ffn_norm,
            norm_eps,
        )?;

        // Graph-safe MoE (kernel-only, pre-determined routes).
        operators.cuda_routed_moe_graph_safe(
            &normed_dev,
            route_experts,
            route_weights,
            self.layer,
            Some(&self.shared_ffn),
            moe_accumulator,
        )?;

        // HC post (kernel-only).
        operators.cuda_hc_post_from_device(
            moe_accumulator,
            &after_attn_dev,
            &ffn_post_dev,
            &ffn_comb_dev,
            1,
            self.hc_config,
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
        let attention_input = rms_norm_rows_with_operators(
            operators,
            &attention_pre.hidden,
            tokens,
            &self.attn_norm,
            self.hc_config.norm_eps,
            "attn_norm",
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
        let ffn_input = rms_norm_rows_with_operators(
            operators,
            &ffn_pre.hidden,
            tokens,
            &self.ffn_norm,
            self.hc_config.norm_eps,
            "ffn_norm",
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnNorm,
            stage_start,
        )?;
        let stage_start = Instant::now();
        let mut feed_forward_hidden = Vec::with_capacity(tokens * self.hc_config.hidden_size);
        for token in 0..tokens {
            let row = &ffn_input
                [token * self.hc_config.hidden_size..(token + 1) * self.hc_config.hidden_size];
            let moe = operators.routed_moe_step(
                self.layer,
                row,
                token_ids[token],
                &self.router,
                predicted_experts,
                &self.router_policy,
                &mut state.expert_planner,
                expert_reader,
                &mut state.expert_handles,
                expert_executor,
                Some(&self.shared_ffn),
            )?;
            feed_forward_hidden.extend_from_slice(&moe.output);
        }
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

#[derive(Debug, Clone)]
pub struct DeepSeekV4LayerState {
    pub kv: DeepSeekV4AttentionCache,
    pub expert_planner: ExpertStreamingPlanner,
    pub expert_handles: CpuExpertHandleStore,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4LayerStepOutput {
    pub attention_hidden: Vec<f32>,
    pub feed_forward_hidden: Vec<f32>,
    pub moe: RoutedMoeStepOutput,
    pub hc_state: Vec<f32>,
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
