//! DeepSeek-V4 transformer layer: HC + attention + MoE + shared FFN.

use crate::artifact::binding::RouterArtifactPayload;

use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::{HyperConnectionConfig, HyperConnectionWeights};
use crate::moe::executor::ExpertExecutor;
use crate::moe::handle::CpuExpertHandleStore;
use crate::moe::routed::RoutedMoeStepOutput;
use crate::moe::routing::ExpertRouterPolicy;
#[cfg(any(feature = "cuda", test))]
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::moe::streaming::{
    ExpertSourceCatalog, ExpertStreamingPlanner, ExpertStreamingPolicy, ExpertStreamingReader,
};
#[cfg(feature = "cuda")]
use ferrule_common::ExpertResidencyControl;
#[cfg(feature = "cuda")]
use ferrule_common::execution::ForwardMode;
use ferrule_common::{Error, Result};

use super::attention::{DeepSeekV4Attention, DeepSeekV4AttentionCache};
#[cfg(feature = "cuda")]
use super::attention::{DeepSeekV4AttentionDecodeArena, DeepSeekV4AttentionRowsTransitionArena};
use super::config::DeepSeekV4AttentionConfig;
use super::helpers::rms_norm_rows_with_operators;
use super::operators::{DeepSeekV4LayerProfileStage, DeepSeekV4OperatorContext};
#[cfg(feature = "cuda")]
use super::sequence::DeepSeekV4PagedKvBinding;

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

/// Exact per-row dimensions of one compressor's CUDA scratch buffers.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DeepSeekV4CompressorArenaShapeKey {
    compress_ratio: usize,
    head_dim: usize,
    overlap: bool,
    rotate_for_indexer: bool,
    ape_rows: usize,
    ape_cols: usize,
    norm_len: usize,
    kv_input_width: usize,
    score_input_width: usize,
    kv_width: usize,
    score_width: usize,
    compressed_width: usize,
    normalized_width: usize,
}

/// Exact per-row dimensions of every CUDA attention arena buffer.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DeepSeekV4AttentionArenaShapeKey {
    hidden_a_width: usize,
    hidden_b_width: usize,
    q_latent_width: usize,
    q_norm_width: usize,
    q_indexer_width: usize,
    query_raw_width: usize,
    query_width: usize,
    kv_raw_width: usize,
    kv_width: usize,
    index_query_width: usize,
    index_weights_width: usize,
    topk_width: usize,
    empty_query_width: usize,
    empty_weights_width: usize,
    empty_kv_width: usize,
    context_width: usize,
    latent_width: usize,
    output_width: usize,
    compact_value_width: usize,
    compact_compress_ratio: usize,
    linear_workspace_width: usize,
    main_compressor: Option<DeepSeekV4CompressorArenaShapeKey>,
    indexer_compressor: Option<DeepSeekV4CompressorArenaShapeKey>,
    indexer_query_b_input_width: Option<usize>,
    indexer_query_b_output_width: Option<usize>,
    indexer_weights_input_width: Option<usize>,
    indexer_weights_output_width: Option<usize>,
}

/// DSV4-specific key for all buffers owned by one layer scratch arena.
///
/// `rows` and execution phase remain in the outer `ExecutionShapeKey` bucket;
/// this key distinguishes the exact arena variants within that bucket.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DeepSeekV4LayerArenaShapeKey {
    attn_hidden_width: usize,
    attn_pre_width: usize,
    attn_post_width: usize,
    attn_comb_width: usize,
    attn_norm_width: usize,
    after_attn_width: usize,
    ffn_hidden_width: usize,
    ffn_pre_width: usize,
    ffn_post_width: usize,
    ffn_comb_width: usize,
    ffn_norm_width: usize,
    attention: DeepSeekV4AttentionArenaShapeKey,
    router_input_width: usize,
    router_logits_width: usize,
    router_indices_width: usize,
    router_weights_width: usize,
    moe_output_width: usize,
    shared_gate_input_width: usize,
    shared_gate_output_width: usize,
    shared_up_input_width: usize,
    shared_up_output_width: usize,
    shared_down_input_width: usize,
    shared_down_output_width: usize,
    moe_route_count: usize,
    moe_route_output_width: usize,
    layer_output_width: usize,
}

#[cfg(any(feature = "cuda", test))]
impl DeepSeekV4CompressorArenaShapeKey {
    fn from_payload(
        payload: &super::attention::DeepSeekV4CompressorPayload,
        hidden_size: usize,
    ) -> Self {
        Self {
            compress_ratio: payload.compress_ratio,
            head_dim: payload.head_dim,
            overlap: payload.overlap,
            rotate_for_indexer: payload.rotate_for_indexer,
            ape_rows: payload.ape_rows,
            ape_cols: payload.ape_cols,
            norm_len: payload.norm.len(),
            kv_input_width: hidden_size,
            score_input_width: hidden_size,
            kv_width: payload.wkv.format.out_features(),
            score_width: payload.wgate.format.out_features(),
            compressed_width: payload.head_dim,
            normalized_width: payload.head_dim,
        }
    }
}

#[cfg(any(feature = "cuda", test))]
impl DeepSeekV4LayerArenaShapeKey {
    fn for_layer(layer: &DeepSeekV4Layer) -> Self {
        let hc = layer.hc_config.hc_mult;
        let hidden = layer.hc_config.hidden_size;
        let hc_dim = hc * hidden;
        let hc_comb = hc * hc;
        let cfg = layer.attention.config;
        let q_full = cfg.q_full_dim();
        let output_latent = cfg.output_latent_dim();
        let index_query = cfg.index_n_heads * cfg.index_head_dim;
        let compressed = layer.attention.compressed.as_ref();
        let main_compressor = compressed.map(|payload| {
            DeepSeekV4CompressorArenaShapeKey::from_payload(&payload.compressor, cfg.hidden_size)
        });
        let indexer = compressed.and_then(|payload| payload.indexer.as_ref());
        let indexer_compressor = indexer.map(|payload| {
            DeepSeekV4CompressorArenaShapeKey::from_payload(&payload.compressor, cfg.hidden_size)
        });
        let linear_workspace_width = cfg
            .hidden_size
            .max(cfg.q_lora_rank)
            .max(q_full)
            .max(cfg.head_dim)
            .max(output_latent)
            .max(index_query);

        Self {
            attn_hidden_width: hidden,
            attn_pre_width: hc,
            attn_post_width: hc,
            attn_comb_width: hc_comb,
            attn_norm_width: hidden,
            after_attn_width: hc_dim,
            ffn_hidden_width: hidden,
            ffn_pre_width: hc,
            ffn_post_width: hc,
            ffn_comb_width: hc_comb,
            ffn_norm_width: hidden,
            attention: DeepSeekV4AttentionArenaShapeKey {
                hidden_a_width: cfg.hidden_size,
                hidden_b_width: cfg.hidden_size,
                q_latent_width: cfg.q_lora_rank,
                q_norm_width: cfg.q_lora_rank,
                q_indexer_width: cfg.q_lora_rank,
                query_raw_width: q_full,
                query_width: q_full,
                kv_raw_width: cfg.head_dim,
                kv_width: cfg.head_dim,
                index_query_width: index_query,
                index_weights_width: cfg.index_n_heads,
                topk_width: cfg.window_size + cfg.index_topk,
                empty_query_width: 1,
                empty_weights_width: 1,
                empty_kv_width: 1,
                context_width: q_full,
                latent_width: output_latent,
                output_width: cfg.hidden_size,
                compact_value_width: cfg.head_dim,
                compact_compress_ratio: cfg.compress_ratio,
                linear_workspace_width,
                main_compressor,
                indexer_compressor,
                indexer_query_b_input_width: indexer
                    .map(|payload| payload.wq_b.format.in_features()),
                indexer_query_b_output_width: indexer
                    .map(|payload| payload.wq_b.format.out_features()),
                indexer_weights_input_width: indexer
                    .map(|payload| payload.weights_proj.format.in_features()),
                indexer_weights_output_width: indexer
                    .map(|payload| payload.weights_proj.format.out_features()),
            },
            router_input_width: hidden,
            router_logits_width: layer.router.weight.format.out_features(),
            router_indices_width: layer.router_policy.top_k,
            router_weights_width: layer.router_policy.top_k,
            moe_output_width: hidden,
            shared_gate_input_width: layer.shared_ffn.gate.format.in_features(),
            shared_gate_output_width: layer.shared_ffn.gate.format.out_features(),
            shared_up_input_width: layer.shared_ffn.up.format.in_features(),
            shared_up_output_width: layer.shared_ffn.up.format.out_features(),
            shared_down_input_width: layer.shared_ffn.down.format.in_features(),
            shared_down_output_width: layer.shared_ffn.down.format.out_features(),
            moe_route_count: layer.router_policy.top_k,
            moe_route_output_width: hidden,
            layer_output_width: hc_dim,
        }
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn layer_arena_variant_layout(layers: &[DeepSeekV4Layer]) -> (Vec<usize>, Vec<usize>) {
    let mut variants = HashMap::new();
    let mut layer_to_variant = Vec::with_capacity(layers.len());
    let mut representative_layers = Vec::new();
    for (layer_idx, layer) in layers.iter().enumerate() {
        let key = DeepSeekV4LayerArenaShapeKey::for_layer(layer);
        let variant = match variants.get(&key).copied() {
            Some(variant) => variant,
            None => {
                let variant = representative_layers.len();
                variants.insert(key, variant);
                representative_layers.push(layer_idx);
                variant
            }
        };
        layer_to_variant.push(variant);
    }
    (layer_to_variant, representative_layers)
}

impl DeepSeekV4Layer {
    pub fn decode_step_reference(
        &self,
        state: &mut DeepSeekV4LayerState,
        expert_runtime: &mut DeepSeekV4LayerExpertRuntime,
        hc_state: &[f32],
        token_id: u32,
        position: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        expert_executor: &impl ExpertExecutor,
    ) -> Result<DeepSeekV4LayerStepOutput> {
        let mut operators = DeepSeekV4OperatorContext::new_cpu()?;
        self.decode_step_with_operators(
            state,
            expert_runtime,
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
        expert_runtime: &mut DeepSeekV4LayerExpertRuntime,
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

        // CPU reference path only. CUDA callers use decode_step_device_hc_device
        // directly, which keeps HC state on device and uses the persistent arena.
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
            &mut expert_runtime.expert_planner,
            expert_reader,
            &mut expert_runtime.expert_handles,
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
    /// HC boundary - the largest source of D2H sync points - is eliminated.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_step_device_hc_device(
        &self,
        state: &mut DeepSeekV4LayerState,
        residency: &mut dyn ExpertResidencyControl,
        source_catalog: &ExpertSourceCatalog,
        prefetch_capacity: usize,
        arena: &mut DeepSeekV4LayerArena,
        hc_state_dev: &mut ferrule_cuda::context::CudaF32Buffer,
        token_id: u32,
        position: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<DeepSeekV4LayerDeviceStepOutput> {
        let decode_start = operators.profile_start();
        let norm_eps = self.hc_config.norm_eps;
        let layer_tag = format!("L{}", self.layer);

        // ── Prefix: attention HC-pre + norm + attention + HC-post + FFN HC-pre + FFN norm ──
        let stage_start = operators.profile_start();
        let attn_hc_name = format!("hc_attn_{layer_tag}");
        operators.cuda_mut()?.hc_pre_from_device_into(
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
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attention_hc_hidden",
            &arena.attn_hidden,
            self.hc_config.hidden_size,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnHcPre,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        let attn_norm_name = format!("attn_norm_{layer_tag}");
        operators.cuda_mut()?.rms_norm_device_cached_into(
            &attn_norm_name,
            &arena.attn_hidden,
            &self.attn_norm,
            norm_eps,
            &mut arena.attn_norm,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attention_input",
            &arena.attn_norm,
            self.hc_config.hidden_size,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnNorm,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        self.attention.decode_step_from_device_into(
            &mut state.kv,
            &arena.attn_norm,
            position,
            operators,
            &mut arena.attention,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::Attention,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.hc_post_from_device_into(
            &arena.attention.output,
            hc_state_dev,
            &arena.attn_post,
            &arena.attn_comb,
            1,
            self.hc_config,
            &mut arena.after_attn,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "after_attention",
            &arena.after_attn,
            self.hc_config.hc_hidden_size(),
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnHcPost,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        let ffn_hc_name = format!("hc_ffn_{layer_tag}");
        operators.cuda_mut()?.hc_pre_from_device_into(
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
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnHcPre,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        let ffn_norm_name = format!("ffn_norm_{layer_tag}");
        operators.cuda_mut()?.rms_norm_device_cached_into(
            &ffn_norm_name,
            &arena.ffn_hidden,
            &self.ffn_norm,
            norm_eps,
            &mut arena.ffn_norm,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "ffn_input",
            &arena.ffn_norm,
            self.hc_config.hidden_size,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnNorm,
            stage_start,
        )?;

        // ── Dispatch: eager MoE with host-side routing ──
        if !predicted_experts.is_empty() {
            operators.prefetch_predicted_experts(
                self.layer,
                predicted_experts,
                residency,
                source_catalog,
                prefetch_capacity,
                expert_reader,
            )?;
        }
        let stage_start = operators.profile_start();
        let moe = operators.cuda_mut()?.routed_moe_step_device_output_into(
            self.layer,
            &arena.ffn_norm,
            token_id,
            &self.router,
            predicted_experts,
            &self.router_policy,
            residency,
            source_catalog,
            prefetch_capacity,
            expert_reader,
            &self.shared_ffn,
            &mut arena.router_input,
            &mut arena.router_logits,
            &mut arena.router_indices,
            &mut arena.router_weights,
            &mut arena.shared_workspace,
            &mut arena.moe_output,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "moe_output",
            &arena.moe_output,
            self.hc_config.hidden_size,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::Moe,
            stage_start,
        )?;

        // ── Suffix: FFN HC-post ──
        let stage_start = operators.profile_start();
        operators.cuda_mut()?.hc_post_from_device_into(
            &arena.moe_output,
            &arena.after_attn,
            &arena.ffn_post,
            &arena.ffn_comb,
            1,
            self.hc_config,
            &mut arena.layer_output,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "layer_output",
            &arena.layer_output,
            self.hc_config.hc_hidden_size(),
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnHcPost,
            stage_start,
        )?;
        if let Some(decode_us) = operators.finish_profile_stage(decode_start)? {
            operators.record_layer_decode(self.layer, decode_us);
        }
        std::mem::swap(hc_state_dev, &mut arena.layer_output);
        Ok(DeepSeekV4LayerDeviceStepOutput { moe })
    }

    /// Layer-major packed decode. HC, attention data-path kernels,
    /// normalization, and MoE execute over all rows together. Only the
    /// sequence-owned compressor recurrent transitions remain row-serial.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn packed_rows_device_hc_device(
        &self,
        states: &mut [&mut DeepSeekV4LayerState],
        row_to_sequence: &[usize],
        sequence_major_rows: &[usize],
        sequence_prefill: &[bool],
        paged_bindings: &[DeepSeekV4PagedKvBinding],
        residency: &mut dyn ExpertResidencyControl,
        source_catalog: &ExpertSourceCatalog,
        prefetch_capacity: usize,
        arena: &mut DeepSeekV4LayerArena,
        hc_state_dev: &mut ferrule_cuda::context::CudaF32Buffer,
        token_ids: &[u32],
        positions: &[usize],
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<()> {
        let rows = token_ids.len();
        if rows < 2
            || positions.len() != rows
            || row_to_sequence.len() != rows
            || sequence_major_rows.len() != rows
            || states.len() != paged_bindings.len()
            || sequence_prefill.len() != states.len()
            || row_to_sequence
                .iter()
                .any(|sequence| *sequence >= states.len())
        {
            return Err(Error::Model(
                "DeepSeek-V4 packed row/sequence metadata is inconsistent".into(),
            ));
        }
        let layer_tag = format!("L{}", self.layer);
        operators.cuda_mut()?.hc_pre_from_device_into(
            &format!("hc_attn_{layer_tag}"),
            hc_state_dev,
            &self.hc_attention,
            rows,
            self.hc_config,
            &mut arena.attn_hidden,
            &mut arena.attn_pre,
            &mut arena.attn_post,
            &mut arena.attn_comb,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attention_hc_hidden",
            &arena.attn_hidden,
            self.hc_config.hidden_size,
        )?;
        operators.cuda_mut()?.rms_norm_rows_device_cached_into(
            &format!("attn_norm_{layer_tag}"),
            &arena.attn_hidden,
            rows,
            &self.attn_norm,
            self.hc_config.norm_eps,
            &mut arena.attn_norm,
        )?;

        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attention_input",
            &arena.attn_norm,
            self.hc_config.hidden_size,
        )?;

        let transition = arena.attention_transition.as_mut().ok_or_else(|| {
            Error::Internal("packed decode arena is missing attention transition scratch".into())
        })?;

        let mut attention_caches = states
            .iter_mut()
            .map(|state| &mut state.kv)
            .collect::<Vec<_>>();
        self.attention.packed_rows_from_device_into(
            &mut attention_caches,
            &arena.attn_norm,
            positions,
            row_to_sequence,
            sequence_major_rows,
            sequence_prefill,
            paged_bindings,
            operators,
            &mut arena.attention,
            transition,
        )?;

        operators.cuda_mut()?.hc_post_from_device_into(
            &arena.attention.output,
            hc_state_dev,
            &arena.attn_post,
            &arena.attn_comb,
            rows,
            self.hc_config,
            &mut arena.after_attn,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "after_attention",
            &arena.after_attn,
            self.hc_config.hc_hidden_size(),
        )?;
        operators.cuda_mut()?.hc_pre_from_device_into(
            &format!("hc_ffn_{layer_tag}"),
            &arena.after_attn,
            &self.hc_feed_forward,
            rows,
            self.hc_config,
            &mut arena.ffn_hidden,
            &mut arena.ffn_pre,
            &mut arena.ffn_post,
            &mut arena.ffn_comb,
        )?;
        operators.cuda_mut()?.rms_norm_rows_device_cached_into(
            &format!("ffn_norm_{layer_tag}"),
            &arena.ffn_hidden,
            rows,
            &self.ffn_norm,
            self.hc_config.norm_eps,
            &mut arena.ffn_norm,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "ffn_input",
            &arena.ffn_norm,
            self.hc_config.hidden_size,
        )?;
        if !predicted_experts.is_empty() {
            operators.prefetch_predicted_experts(
                self.layer,
                predicted_experts,
                residency,
                source_catalog,
                prefetch_capacity,
                expert_reader,
            )?;
        }
        operators.routed_moe_prefill_batch_from_device_into(
            self.layer,
            &arena.ffn_norm,
            token_ids,
            Some(row_to_sequence),
            &self.router,
            predicted_experts,
            &self.router_policy,
            residency,
            source_catalog,
            prefetch_capacity,
            expert_reader,
            &self.shared_ffn,
            &mut arena.router_logits,
            &mut arena.router_indices,
            &mut arena.router_weights,
            &mut arena.attention.linear_workspace,
            &mut arena.shared_workspace,
            &mut arena.moe_segment_workspace,
            &mut arena.moe_route_output,
            &mut arena.moe_output,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "moe_output",
            &arena.moe_output,
            self.hc_config.hidden_size,
        )?;
        operators.cuda_mut()?.hc_post_from_device_into(
            &arena.moe_output,
            &arena.after_attn,
            &arena.ffn_post,
            &arena.ffn_comb,
            rows,
            self.hc_config,
            &mut arena.layer_output,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "layer_output",
            &arena.layer_output,
            self.hc_config.hc_hidden_size(),
        )?;
        std::mem::swap(hc_state_dev, &mut arena.layer_output);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_start_cuda_device_chain_into(
        &self,
        state: &mut DeepSeekV4LayerState,
        residency: &mut dyn ExpertResidencyControl,
        source_catalog: &ExpertSourceCatalog,
        prefetch_capacity: usize,
        arena: &mut DeepSeekV4LayerArena,
        hc_state_dev: &mut ferrule_cuda::context::CudaF32Buffer,
        token_ids: &[u32],
        start_pos: usize,
        predicted_experts: &[usize],
        expert_reader: &ExpertStreamingReader,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<()> {
        let tokens = token_ids.len();
        let prefill_start = operators.profile_start();
        let layer_tag = format!("L{}", self.layer);

        let stage_start = operators.profile_start();
        let attn_hc_name = format!("hc_attn_{layer_tag}");
        operators.cuda_mut()?.hc_pre_from_device_into(
            &attn_hc_name,
            hc_state_dev,
            &self.hc_attention,
            tokens,
            self.hc_config,
            &mut arena.attn_hidden,
            &mut arena.attn_pre,
            &mut arena.attn_post,
            &mut arena.attn_comb,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attention_hc_hidden",
            &arena.attn_hidden,
            self.hc_config.hidden_size,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnHcPre,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        let attn_norm_name = format!("attn_norm_{layer_tag}");
        operators.cuda_mut()?.rms_norm_rows_device_cached_into(
            &attn_norm_name,
            &arena.attn_hidden,
            tokens,
            &self.attn_norm,
            self.hc_config.norm_eps,
            &mut arena.attn_norm,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "attention_input",
            &arena.attn_norm,
            self.hc_config.hidden_size,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnNorm,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        self.attention.prefill_start_from_device_into(
            &mut state.kv,
            &arena.attn_norm,
            start_pos,
            &mut arena.attention,
            operators,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::Attention,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.hc_post_from_device_into(
            &arena.attention.output,
            hc_state_dev,
            &arena.attn_post,
            &arena.attn_comb,
            tokens,
            self.hc_config,
            &mut arena.after_attn,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "after_attention",
            &arena.after_attn,
            self.hc_config.hc_hidden_size(),
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnHcPost,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        let ffn_hc_name = format!("hc_ffn_{layer_tag}");
        operators.cuda_mut()?.hc_pre_from_device_into(
            &ffn_hc_name,
            &arena.after_attn,
            &self.hc_feed_forward,
            tokens,
            self.hc_config,
            &mut arena.ffn_hidden,
            &mut arena.ffn_pre,
            &mut arena.ffn_post,
            &mut arena.ffn_comb,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnHcPre,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        let ffn_norm_name = format!("ffn_norm_{layer_tag}");
        operators.cuda_mut()?.rms_norm_rows_device_cached_into(
            &ffn_norm_name,
            &arena.ffn_hidden,
            tokens,
            &self.ffn_norm,
            self.hc_config.norm_eps,
            &mut arena.ffn_norm,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "ffn_input",
            &arena.ffn_norm,
            self.hc_config.hidden_size,
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
                residency,
                source_catalog,
                prefetch_capacity,
                expert_reader,
            )?;
        }

        let stage_start = operators.profile_start();
        operators.routed_moe_prefill_batch_from_device_into(
            self.layer,
            &arena.ffn_norm,
            token_ids,
            None,
            &self.router,
            predicted_experts,
            &self.router_policy,
            residency,
            source_catalog,
            prefetch_capacity,
            expert_reader,
            &self.shared_ffn,
            &mut arena.router_logits,
            &mut arena.router_indices,
            &mut arena.router_weights,
            &mut arena.attention.linear_workspace,
            &mut arena.shared_workspace,
            &mut arena.moe_segment_workspace,
            &mut arena.moe_route_output,
            &mut arena.moe_output,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "moe_output",
            &arena.moe_output,
            self.hc_config.hidden_size,
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::Moe,
            stage_start,
        )?;

        let stage_start = operators.profile_start();
        operators.cuda_mut()?.hc_post_from_device_into(
            &arena.moe_output,
            &arena.after_attn,
            &arena.ffn_post,
            &arena.ffn_comb,
            tokens,
            self.hc_config,
            &mut arena.layer_output,
        )?;
        operators.capture_parity_checkpoint_last_row(
            self.layer,
            "layer_output",
            &arena.layer_output,
            self.hc_config.hc_hidden_size(),
        )?;
        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::FfnHcPost,
            stage_start,
        )?;

        if let Some(prefill_us) = operators.finish_profile_stage(prefill_start)? {
            operators.record_layer_prefill(self.layer, tokens, prefill_us);
        }
        std::mem::swap(hc_state_dev, &mut arena.layer_output);
        Ok(())
    }

    pub fn prefill_start_with_operators(
        &self,
        state: &mut DeepSeekV4LayerState,
        expert_runtime: &mut DeepSeekV4LayerExpertRuntime,
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

        let prefill_start = operators.profile_start();
        let stage_start = operators.profile_start();
        let attention_pre =
            operators.hc_pre(hc_state, tokens, self.hc_config, &self.hc_attention)?;

        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::AttnHcPre,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
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
        let stage_start = operators.profile_start();
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
        let stage_start = operators.profile_start();
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

        let stage_start = operators.profile_start();
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
        let stage_start = operators.profile_start();
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
        let stage_start = operators.profile_start();
        let feed_forward_hidden = operators.routed_moe_prefill_batch(
            self.layer,
            &ffn_input,
            token_ids,
            &self.router,
            predicted_experts,
            &self.router_policy,
            &mut expert_runtime.expert_planner,
            expert_reader,
            &mut expert_runtime.expert_handles,
            expert_executor,
            Some(&self.shared_ffn),
        )?;

        record_stage(
            operators,
            self.layer,
            DeepSeekV4LayerProfileStage::Moe,
            stage_start,
        )?;
        let stage_start = operators.profile_start();
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
        if let Some(prefill_us) = operators.finish_profile_stage(prefill_start)? {
            operators.record_layer_prefill(self.layer, tokens, prefill_us);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct DeepSeekV4LayerState {
    attention_config: DeepSeekV4AttentionConfig,
    pub kv: DeepSeekV4AttentionCache,
}

#[derive(Debug)]
pub struct DeepSeekV4LayerExpertRuntime {
    pub expert_planner: ExpertStreamingPlanner,
    pub expert_handles: CpuExpertHandleStore,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4LayerArena {
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
    attention: DeepSeekV4AttentionDecodeArena,
    attention_transition: Option<DeepSeekV4AttentionRowsTransitionArena>,
    router_input: ferrule_cuda::context::CudaF32Buffer,
    router_logits: ferrule_cuda::context::CudaF32Buffer,
    router_indices: ferrule_cuda::context::CudaI32Buffer,
    router_weights: ferrule_cuda::context::CudaF32Buffer,
    moe_output: ferrule_cuda::context::CudaF32Buffer,
    shared_workspace: ferrule_cuda::context::CudaSwiGLUWorkspace,
    moe_segment_workspace: Option<ferrule_cuda::context::CudaMoeSegmentWorkspace>,
    moe_route_output: ferrule_cuda::context::CudaF32Buffer,
    layer_output: ferrule_cuda::context::CudaF32Buffer,
    hidden_size: usize,
    hc_mult: usize,
}

/// One exact phase/rows bucket containing one arena per unique DSV4 scratch shape.
#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4LayerArenaVariants {
    arenas: Vec<DeepSeekV4LayerArena>,
    layer_to_variant: Box<[usize]>,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4LayerArenaVariants {
    pub(crate) fn try_build(
        layers: &[DeepSeekV4Layer],
        rows: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Self> {
        Self::try_build_for_mode(layers, ForwardMode::Decode, rows, operators)
    }

    pub(crate) fn try_build_for_mode(
        layers: &[DeepSeekV4Layer],
        mode: ForwardMode,
        rows: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Self> {
        Self::try_build_with_row_transitions(layers, rows, mode == ForwardMode::Decode, operators)
    }

    pub(crate) fn try_build_for_packed_mode(
        layers: &[DeepSeekV4Layer],
        rows: usize,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Self> {
        Self::try_build_with_row_transitions(layers, rows, true, operators)
    }

    fn try_build_with_row_transitions(
        layers: &[DeepSeekV4Layer],
        rows: usize,
        independent_rows: bool,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Self> {
        let (layer_to_variant, representative_layers) = layer_arena_variant_layout(layers);
        let mut arenas = Vec::with_capacity(representative_layers.len());
        for layer_idx in representative_layers {
            arenas.push(DeepSeekV4LayerArena::new(
                &layers[layer_idx],
                rows,
                independent_rows,
                operators,
            )?);
        }

        Ok(Self {
            arenas,
            layer_to_variant: layer_to_variant.into_boxed_slice(),
        })
    }

    pub(crate) fn get_for_layer_mut(&mut self, layer: usize) -> Option<&mut DeepSeekV4LayerArena> {
        let variant = *self.layer_to_variant.get(layer)?;
        self.arenas.get_mut(variant)
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeepSeekV4LayerArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepSeekV4LayerArena")
            .field("hidden_size", &self.hidden_size)
            .field("hc_mult", &self.hc_mult)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda")]
impl DeepSeekV4LayerArena {
    pub(crate) fn new(
        layer: &DeepSeekV4Layer,
        rows: usize,
        independent_rows: bool,
        operators: &mut DeepSeekV4OperatorContext,
    ) -> Result<Self> {
        let config = layer.hc_config;
        let hidden = config.hidden_size;
        let hc = config.hc_mult;
        let hc_dim = hc
            .checked_mul(hidden)
            .ok_or_else(|| Error::Internal("DSV4 arena HC dim overflow".into()))?;
        let comb = hc
            .checked_mul(hc)
            .ok_or_else(|| Error::Internal("DSV4 arena HC comb overflow".into()))?;
        Ok(Self {
            attn_hidden: operators.cuda_mut()?.ops.zero_f32_buffer(rows * hidden)?,
            attn_pre: operators.cuda_mut()?.ops.zero_f32_buffer(rows * hc)?,
            attn_post: operators.cuda_mut()?.ops.zero_f32_buffer(rows * hc)?,
            attn_comb: operators.cuda_mut()?.ops.zero_f32_buffer(rows * comb)?,
            attn_norm: operators.cuda_mut()?.ops.zero_f32_buffer(rows * hidden)?,
            after_attn: operators.cuda_mut()?.ops.zero_f32_buffer(rows * hc_dim)?,
            ffn_hidden: operators.cuda_mut()?.ops.zero_f32_buffer(rows * hidden)?,
            ffn_pre: operators.cuda_mut()?.ops.zero_f32_buffer(rows * hc)?,
            ffn_post: operators.cuda_mut()?.ops.zero_f32_buffer(rows * hc)?,
            ffn_comb: operators.cuda_mut()?.ops.zero_f32_buffer(rows * comb)?,
            ffn_norm: operators.cuda_mut()?.ops.zero_f32_buffer(rows * hidden)?,
            attention: DeepSeekV4AttentionDecodeArena::new(
                &layer.attention,
                rows,
                independent_rows,
                operators,
            )?,
            attention_transition: (rows > 1)
                .then(|| DeepSeekV4AttentionRowsTransitionArena::new(&layer.attention, operators))
                .transpose()?,
            router_input: operators.cuda_mut()?.ops.zero_f32_buffer(rows * hidden)?,
            router_logits: operators
                .cuda_mut()?
                .ops
                .zero_f32_buffer(rows * layer.router.weight.format.out_features())?,
            router_indices: operators
                .cuda_mut()?
                .ops
                .zero_i32_buffer(rows * layer.router_policy.top_k)?,
            router_weights: operators
                .cuda_mut()?
                .ops
                .zero_f32_buffer(rows * layer.router_policy.top_k)?,
            moe_output: operators.cuda_mut()?.ops.zero_f32_buffer(rows * hidden)?,
            shared_workspace: operators.cuda_mut()?.ops.swiglu_workspace_for_shape(
                rows,
                hidden,
                layer.shared_ffn.gate.format.out_features(),
                hidden,
            )?,
            moe_segment_workspace: None,
            moe_route_output: operators.cuda_mut()?.ops.allocate_moe_route_output(
                rows,
                layer.router_policy.top_k,
                hidden,
            )?,
            layer_output: operators.cuda_mut()?.ops.zero_f32_buffer(rows * hc_dim)?,
            hidden_size: hidden,
            hc_mult: hc,
        })
    }
}

impl DeepSeekV4LayerState {
    pub(crate) fn new(attention_config: DeepSeekV4AttentionConfig) -> Self {
        Self {
            attention_config,
            kv: DeepSeekV4AttentionCache::new(attention_config),
        }
    }

    pub fn reset_sequence(&mut self) {
        self.kv.reset_sequence();
    }

    pub(crate) fn fork_paged_prefix_metadata(&self) -> Self {
        Self {
            attention_config: self.attention_config,
            kv: self.kv.fork_paged_prefix_metadata(),
        }
    }

    pub fn release_sequence_capacity(&mut self) {
        self.kv = DeepSeekV4AttentionCache::new(self.attention_config);
    }
}

impl DeepSeekV4LayerExpertRuntime {
    pub(crate) fn new(expert_planner: ExpertStreamingPlanner) -> Self {
        Self {
            expert_planner,
            expert_handles: CpuExpertHandleStore::default(),
        }
    }

    pub(crate) fn from_catalog(
        source_catalog: Arc<ExpertSourceCatalog>,
        policy: ExpertStreamingPolicy,
    ) -> Self {
        Self::new(ExpertStreamingPlanner::from_catalog(policy, source_catalog))
    }
}

#[cfg(test)]
mod expert_runtime_tests {
    use crate::moe::streaming::{ExpertId, ExpertLoadSource};

    use super::*;

    #[test]
    fn cpu_expert_runtime_is_constructed_from_shared_catalog() {
        let catalog = Arc::new(ExpertSourceCatalog::from_sources([(
            ExpertId::new(4, 9),
            ExpertLoadSource::LocalShard {
                path: "experts.safetensors".into(),
                offset: 128,
                bytes: 64,
            },
        )]));
        let runtime = DeepSeekV4LayerExpertRuntime::from_catalog(
            Arc::clone(&catalog),
            ExpertStreamingPolicy::quality_first(1),
        );

        assert!(Arc::ptr_eq(
            runtime.expert_planner.source_catalog(),
            &catalog
        ));
        assert_eq!(runtime.expert_planner.source_catalog().count(), 1);
        assert!(runtime.expert_handles.is_empty());
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
    pub moe: RoutedMoeStepOutput,
}

fn record_stage(
    operators: &mut DeepSeekV4OperatorContext,
    layer: usize,
    stage: DeepSeekV4LayerProfileStage,
    start: Option<Instant>,
) -> Result<()> {
    if let Some(elapsed_us) = operators.finish_profile_stage(start)? {
        operators.record_layer_stage(layer, stage, elapsed_us);
    }
    Ok(())
}
