//! DeepSeek-V4 model configuration.

use std::path::Path;

use crate::TensorRole;
use crate::artifact::binding::MlaAttentionArtifactPayload;
use crate::artifact::linear::{
    ArtifactActivationQuantization, ArtifactLinearExecutionPolicy, ArtifactLinearFormat,
    ArtifactLinearPayload,
};
use crate::attention_backend::SparseAttentionSpec;
use crate::families::deepseek_v4;
use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::HyperConnectionConfig;
use ferrule_common::{Error, Result};

use super::helpers::{f32_key, usize_key};

const DSV4_LINEAR_ACTIVATION_QUANT_BLOCK_SIZE: usize = 128;

pub(crate) fn deepseek_v4_linear_activation_quantization() -> ArtifactActivationQuantization {
    ArtifactActivationQuantization::Fp8E4M3WithE8M0Scale {
        block_size: DSV4_LINEAR_ACTIVATION_QUANT_BLOCK_SIZE,
    }
}

fn deepseek_v4_quantized_linear_execution_policy() -> ArtifactLinearExecutionPolicy {
    ArtifactLinearExecutionPolicy::fp8_e4m3_e8m0_activation(DSV4_LINEAR_ACTIVATION_QUANT_BLOCK_SIZE)
}

pub(crate) fn with_deepseek_v4_linear_execution_policy(
    linear: ArtifactLinearPayload,
) -> ArtifactLinearPayload {
    if deepseek_v4_role_uses_official_linear_activation_quantization(&linear.role)
        && artifact_linear_format_has_quantized_weight(&linear.format)
    {
        linear.with_execution_policy(deepseek_v4_quantized_linear_execution_policy())
    } else {
        linear
    }
}

pub(crate) fn with_deepseek_v4_attention_execution_policies(
    mut payload: MlaAttentionArtifactPayload,
) -> MlaAttentionArtifactPayload {
    payload.query_a = with_deepseek_v4_linear_execution_policy(payload.query_a);
    payload.query_b = with_deepseek_v4_linear_execution_policy(payload.query_b);
    payload.key_value = with_deepseek_v4_linear_execution_policy(payload.key_value);
    // Official DSV4 does not call `linear()` for `wo_a`; it uses a grouped einsum
    // over the dequantized FP8 weight, so activation quantization must not be applied.
    payload.output_b = with_deepseek_v4_linear_execution_policy(payload.output_b);
    payload
}

pub(crate) fn with_deepseek_v4_swiglu_execution_policies(
    mut ffn: SwiGluFfnPayload,
) -> SwiGluFfnPayload {
    ffn.gate = with_deepseek_v4_linear_execution_policy(ffn.gate);
    ffn.up = with_deepseek_v4_linear_execution_policy(ffn.up);
    ffn.down = with_deepseek_v4_linear_execution_policy(ffn.down);
    ffn
}

fn deepseek_v4_role_uses_official_linear_activation_quantization(role: &TensorRole) -> bool {
    matches!(
        role,
        TensorRole::AttentionLatentQueryA
            | TensorRole::AttentionLatentQueryB
            | TensorRole::AttentionLatentKv
            | TensorRole::AttentionLatentOutputB
            | TensorRole::SharedExpertGate
            | TensorRole::SharedExpertUp
            | TensorRole::SharedExpertDown
            | TensorRole::AuxIndexer
    )
}

fn artifact_linear_format_has_quantized_weight(format: &ArtifactLinearFormat) -> bool {
    matches!(
        format,
        ArtifactLinearFormat::Fp8E4M3WithE8M0Scale { .. }
            | ArtifactLinearFormat::Fp4E2M1PackedWithE8M0Scale { .. }
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4Config {
    pub hidden_size: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub norm_eps: f32,
    pub num_layers: usize,
    pub num_hash_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub vocab_size: usize,
    pub num_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub swiglu_limit: f32,
    pub route_scale: f32,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub original_seq_len: usize,
    pub rope_factor: f32,
    pub beta_fast: usize,
    pub beta_slow: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub compress_ratios: Vec<usize>,
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: Option<u32>,
    pub dspark_target_layer_ids: Vec<usize>,
    pub dspark_markov_rank: Option<usize>,
}

impl DeepSeekV4Config {
    pub fn from_hf_config(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Model(format!("DeepSeek-V4 config '{}': {e}", path.display())))?;
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            Error::Model(format!("DeepSeek-V4 config json '{}': {e}", path.display()))
        })?;
        let rope_scaling = json.get("rope_scaling").unwrap_or(&serde_json::Value::Null);
        let compress_ratios = json
            .get("compress_ratios")
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .map(|item| item.as_u64().map(|value| value as usize).unwrap_or(0))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![0; deepseek_v4::NUM_LAYERS]);
        let dspark_target_layer_ids = json
            .get("dspark_target_layer_ids")
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_u64().map(|value| value as usize))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(Self {
            hidden_size: usize_key(&json, &["hidden_size"]).unwrap_or(deepseek_v4::HIDDEN_SIZE),
            hc_mult: usize_key(&json, &["hc_mult"]).unwrap_or(deepseek_v4::HC_MULT),
            hc_sinkhorn_iters: usize_key(&json, &["hc_sinkhorn_iters"])
                .unwrap_or(deepseek_v4::HC_SINKHORN_ITERS),
            hc_eps: f32_key(&json, &["hc_eps"]).unwrap_or(deepseek_v4::HC_EPS),
            norm_eps: f32_key(&json, &["rms_norm_eps", "norm_eps"])
                .unwrap_or(deepseek_v4::RMS_NORM_EPS),
            num_layers: usize_key(&json, &["num_hidden_layers", "n_layers"])
                .unwrap_or(deepseek_v4::NUM_LAYERS),
            num_hash_layers: usize_key(&json, &["num_hash_layers"])
                .unwrap_or(deepseek_v4::NUM_HASH_LAYERS),
            num_heads: usize_key(&json, &["num_attention_heads", "n_heads"])
                .unwrap_or(deepseek_v4::NUM_HEADS),
            head_dim: usize_key(&json, &["head_dim"]).unwrap_or(deepseek_v4::HEAD_DIM),
            q_lora_rank: usize_key(&json, &["q_lora_rank"]).unwrap_or(deepseek_v4::Q_LORA_RANK),
            qk_rope_head_dim: usize_key(&json, &["qk_rope_head_dim", "rope_head_dim"])
                .unwrap_or(deepseek_v4::QK_ROPE_HEAD_DIM),
            o_groups: usize_key(&json, &["o_groups"]).unwrap_or(deepseek_v4::O_GROUPS),
            o_lora_rank: usize_key(&json, &["o_lora_rank"]).unwrap_or(deepseek_v4::O_LORA_RANK),
            window_size: usize_key(&json, &["sliding_window", "window_size"])
                .unwrap_or(deepseek_v4::SLIDING_WINDOW),
            vocab_size: usize_key(&json, &["vocab_size"]).unwrap_or(deepseek_v4::VOCAB_SIZE),
            num_routed_experts: usize_key(&json, &["n_routed_experts", "num_experts"])
                .unwrap_or(deepseek_v4::N_ROUTED_EXPERTS),
            num_experts_per_tok: usize_key(&json, &["num_experts_per_tok"])
                .unwrap_or(deepseek_v4::NUM_EXPERTS_PER_TOK),
            moe_intermediate_size: usize_key(&json, &["moe_intermediate_size", "moe_inter_dim"])
                .unwrap_or(deepseek_v4::MOE_INTERMEDIATE_SIZE),
            swiglu_limit: f32_key(&json, &["swiglu_limit"]).unwrap_or(deepseek_v4::SWIGLU_LIMIT),
            route_scale: f32_key(&json, &["routed_scaling_factor", "route_scale"])
                .unwrap_or(deepseek_v4::ROUTED_SCALING_FACTOR),
            rope_theta: f32_key(&json, &["rope_theta"]).unwrap_or(deepseek_v4::ROPE_THETA),
            compress_rope_theta: f32_key(&json, &["compress_rope_theta"])
                .unwrap_or(deepseek_v4::COMPRESS_ROPE_THETA),
            original_seq_len: usize_key(rope_scaling, &["original_max_position_embeddings"])
                .or_else(|| usize_key(&json, &["original_seq_len"]))
                .unwrap_or(deepseek_v4::ORIGINAL_MAX_POSITION_EMBEDDINGS),
            rope_factor: f32_key(rope_scaling, &["factor"])
                .or_else(|| f32_key(&json, &["rope_factor"]))
                .unwrap_or(deepseek_v4::ROPE_FACTOR),
            beta_fast: usize_key(rope_scaling, &["beta_fast"])
                .or_else(|| usize_key(&json, &["beta_fast"]))
                .unwrap_or(deepseek_v4::ROPE_BETA_FAST),
            beta_slow: usize_key(rope_scaling, &["beta_slow"])
                .or_else(|| usize_key(&json, &["beta_slow"]))
                .unwrap_or(deepseek_v4::ROPE_BETA_SLOW),
            index_n_heads: usize_key(&json, &["index_n_heads"])
                .unwrap_or(deepseek_v4::INDEX_N_HEADS),
            index_head_dim: usize_key(&json, &["index_head_dim"])
                .unwrap_or(deepseek_v4::INDEX_HEAD_DIM),
            index_topk: usize_key(&json, &["index_topk"]).unwrap_or(deepseek_v4::INDEX_TOPK),
            compress_ratios,
            dspark_block_size: usize_key(&json, &["dspark_block_size"]).unwrap_or(1),
            dspark_noise_token_id: usize_key(&json, &["dspark_noise_token_id"])
                .map(|value| value as u32),
            dspark_target_layer_ids,
            dspark_markov_rank: usize_key(&json, &["dspark_markov_rank"]),
        })
    }

    pub fn hc_config(&self) -> HyperConnectionConfig {
        HyperConnectionConfig {
            hc_mult: self.hc_mult,
            hidden_size: self.hidden_size,
            sinkhorn_iters: self.hc_sinkhorn_iters,
            eps: self.hc_eps,
            norm_eps: self.norm_eps,
        }
    }

    pub fn attention_config_for_layer(&self, layer: usize) -> Result<DeepSeekV4AttentionConfig> {
        if layer >= self.num_layers {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} exceeds layer count {}",
                self.num_layers
            )));
        }
        Ok(DeepSeekV4AttentionConfig {
            hidden_size: self.hidden_size,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            q_lora_rank: self.q_lora_rank,
            rope_head_dim: self.qk_rope_head_dim,
            o_groups: self.o_groups,
            o_lora_rank: self.o_lora_rank,
            window_size: self.window_size,
            compress_ratio: self.compress_ratios.get(layer).copied().unwrap_or(0),
            norm_eps: self.norm_eps,
            rope_theta: self.rope_theta,
            compress_rope_theta: self.compress_rope_theta,
            original_seq_len: self.original_seq_len,
            rope_factor: self.rope_factor,
            beta_fast: self.beta_fast,
            beta_slow: self.beta_slow,
            index_n_heads: self.index_n_heads,
            index_head_dim: self.index_head_dim,
            index_topk: self.index_topk,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepSeekV4AttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub compress_ratio: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub original_seq_len: usize,
    pub rope_factor: f32,
    pub beta_fast: usize,
    pub beta_slow: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
}

impl DeepSeekV4AttentionConfig {
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.num_heads == 0
            || self.head_dim == 0
            || self.o_groups == 0
            || self.o_lora_rank == 0
            || self.window_size == 0
            || self.index_n_heads == 0
            || self.index_head_dim == 0
            || self.index_topk == 0
        {
            return Err(Error::Model(format!(
                "invalid DeepSeek-V4 attention shape: hidden={}, heads={}, head_dim={}, groups={}, o_rank={}, window={}, index_heads={}, index_dim={}, index_topk={}",
                self.hidden_size,
                self.num_heads,
                self.head_dim,
                self.o_groups,
                self.o_lora_rank,
                self.window_size,
                self.index_n_heads,
                self.index_head_dim,
                self.index_topk
            )));
        }
        if !self.num_heads.is_multiple_of(self.o_groups) {
            return Err(Error::Model(format!(
                "DeepSeek-V4 attention heads {} must be divisible by o_groups {}",
                self.num_heads, self.o_groups
            )));
        }
        if self.rope_head_dim > self.head_dim || !self.rope_head_dim.is_multiple_of(2) {
            return Err(Error::Model(format!(
                "DeepSeek-V4 rope_head_dim {} must be even and <= head_dim {}",
                self.rope_head_dim, self.head_dim
            )));
        }
        if self.norm_eps <= 0.0 || self.rope_theta <= 0.0 || self.compress_rope_theta <= 0.0 {
            return Err(Error::Model(
                "DeepSeek-V4 attention eps/theta values must be positive".into(),
            ));
        }
        Ok(())
    }

    pub fn q_full_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn output_group_input_dim(&self) -> usize {
        self.q_full_dim() / self.o_groups
    }

    pub fn output_latent_dim(&self) -> usize {
        self.o_groups * self.o_lora_rank
    }

    pub fn sparse_spec(&self) -> SparseAttentionSpec {
        self.sparse_spec_with_topk(self.window_size)
    }

    pub fn sparse_spec_with_topk(&self, topk: usize) -> SparseAttentionSpec {
        SparseAttentionSpec {
            heads: self.num_heads,
            head_dim: self.head_dim,
            topk,
            softmax_scale: (self.head_dim as f32).powf(-0.5),
            has_attention_sink: true,
        }
    }

    pub fn rope_params(&self) -> DeepSeekV4RopeParams {
        if self.compress_ratio == 0 {
            DeepSeekV4RopeParams::plain(self.rope_theta)
        } else {
            DeepSeekV4RopeParams {
                theta: self.compress_rope_theta,
                original_seq_len: self.original_seq_len,
                factor: self.rope_factor,
                beta_fast: self.beta_fast,
                beta_slow: self.beta_slow,
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepSeekV4RopeParams {
    pub theta: f32,
    pub original_seq_len: usize,
    pub factor: f32,
    pub beta_fast: usize,
    pub beta_slow: usize,
}

impl DeepSeekV4RopeParams {
    pub fn plain(theta: f32) -> Self {
        Self {
            theta,
            original_seq_len: 0,
            factor: 1.0,
            beta_fast: 32,
            beta_slow: 1,
        }
    }
}
