//! DeepSeek-V4 model configuration.

#[cfg(feature = "cuda")]
use crate::TensorRole;
#[cfg(feature = "cuda")]
use crate::transformer::attention::mla::{MlaConfig, MlaWeights};

#[cfg(feature = "cuda")]
use crate::checkpoint::weight::{LinearExecutionPolicy, LinearWeight, LinearWeightFormat};
use crate::families::deepseek_v4;
#[cfg(feature = "cuda")]
use crate::ffn::SwiGluFfnPayload;
use crate::transformer::connection::HyperConnectionConfig;
use ferrule_common::{Error, Result};

use crate::models::common::config_json::{f32_key, usize_key};

#[cfg(feature = "cuda")]
const DSV4_LINEAR_ACTIVATION_QUANT_BLOCK_SIZE: usize = 128;

#[cfg(feature = "cuda")]
fn deepseek_v4_quantized_linear_execution_policy() -> LinearExecutionPolicy {
    LinearExecutionPolicy::fp8_e4m3_e8m0_activation(DSV4_LINEAR_ACTIVATION_QUANT_BLOCK_SIZE)
}

#[cfg(feature = "cuda")]
pub(crate) fn with_deepseek_v4_linear_execution_policy(linear: LinearWeight) -> LinearWeight {
    if deepseek_v4_role_uses_official_linear_activation_quantization(&linear.role)
        && artifact_linear_format_has_quantized_weight(&linear.format)
    {
        linear.with_execution_policy(deepseek_v4_quantized_linear_execution_policy())
    } else {
        linear
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn with_deepseek_v4_attention_execution_policies(mut payload: MlaWeights) -> MlaWeights {
    payload.query_a = with_deepseek_v4_linear_execution_policy(payload.query_a);
    payload.query_b = with_deepseek_v4_linear_execution_policy(payload.query_b);
    payload.key_value = with_deepseek_v4_linear_execution_policy(payload.key_value);
    // Official DSV4 does not call `linear()` for `wo_a`; it uses a grouped einsum
    // over the dequantized FP8 weight, so activation quantization must not be applied.
    payload.output_b = with_deepseek_v4_linear_execution_policy(payload.output_b);
    payload
}

#[cfg(feature = "cuda")]
pub(crate) fn with_deepseek_v4_swiglu_execution_policies(
    mut ffn: SwiGluFfnPayload,
) -> SwiGluFfnPayload {
    ffn.gate = with_deepseek_v4_linear_execution_policy(ffn.gate);
    ffn.up = with_deepseek_v4_linear_execution_policy(ffn.up);
    ffn.down = with_deepseek_v4_linear_execution_policy(ffn.down);
    ffn
}

#[cfg(feature = "cuda")]
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
            | TensorRole::SpeculativeProjection
    )
}

#[cfg(feature = "cuda")]
fn artifact_linear_format_has_quantized_weight(format: &LinearWeightFormat) -> bool {
    matches!(
        format,
        LinearWeightFormat::Fp8E4M3WithE8M0Scale { .. }
            | LinearWeightFormat::Fp4E2M1PackedWithE8M0Scale { .. }
    )
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DeepSeekV4Config {
    pub architecture: String,
    pub max_position_embeddings: usize,
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
    pub proposal_block_size: usize,
    pub proposal_noise_token_id: Option<u32>,
    pub proposal_target_layer_ids: Vec<usize>,
    pub proposal_markov_rank: Option<usize>,
}

impl DeepSeekV4Config {
    pub(super) fn from_value(json: &serde_json::Value) -> Result<Self> {
        let object = json.as_object().ok_or_else(|| Error::Model {
            message: "DeepSeek-V4 config must be a JSON object".into(),
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
        let proposal_target_layer_ids = json
            .get("proposal_target_layer_ids")
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_u64().map(|value| value as usize))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let config = Self {
            architecture: object
                .get("architectures")
                .and_then(serde_json::Value::as_array)
                .and_then(|architectures| architectures.first())
                .and_then(serde_json::Value::as_str)
                .unwrap_or("DeepseekV4ForCausalLM")
                .to_string(),
            max_position_embeddings: usize_key(json, &["max_position_embeddings", "max_seq_len"])
                .unwrap_or(1_048_576),
            hidden_size: usize_key(json, &["hidden_size"]).unwrap_or(deepseek_v4::HIDDEN_SIZE),
            hc_mult: usize_key(json, &["hc_mult"]).unwrap_or(deepseek_v4::HC_MULT),
            hc_sinkhorn_iters: usize_key(json, &["hc_sinkhorn_iters"])
                .unwrap_or(deepseek_v4::HC_SINKHORN_ITERS),
            hc_eps: f32_key(json, &["hc_eps"]).unwrap_or(deepseek_v4::HC_EPS),
            norm_eps: f32_key(json, &["rms_norm_eps", "norm_eps"])
                .unwrap_or(deepseek_v4::RMS_NORM_EPS),
            num_layers: usize_key(json, &["num_hidden_layers", "n_layers"])
                .unwrap_or(deepseek_v4::NUM_LAYERS),
            num_hash_layers: usize_key(json, &["num_hash_layers"])
                .unwrap_or(deepseek_v4::NUM_HASH_LAYERS),
            num_heads: usize_key(json, &["num_attention_heads", "n_heads"])
                .unwrap_or(deepseek_v4::NUM_HEADS),
            head_dim: usize_key(json, &["head_dim"]).unwrap_or(deepseek_v4::HEAD_DIM),
            q_lora_rank: usize_key(json, &["q_lora_rank"]).unwrap_or(deepseek_v4::Q_LORA_RANK),
            qk_rope_head_dim: usize_key(json, &["qk_rope_head_dim", "rope_head_dim"])
                .unwrap_or(deepseek_v4::QK_ROPE_HEAD_DIM),
            o_groups: usize_key(json, &["o_groups"]).unwrap_or(deepseek_v4::O_GROUPS),
            o_lora_rank: usize_key(json, &["o_lora_rank"]).unwrap_or(deepseek_v4::O_LORA_RANK),
            window_size: usize_key(json, &["sliding_window", "window_size"])
                .unwrap_or(deepseek_v4::SLIDING_WINDOW),
            vocab_size: usize_key(json, &["vocab_size"]).unwrap_or(deepseek_v4::VOCAB_SIZE),
            num_routed_experts: usize_key(json, &["n_routed_experts", "num_experts"])
                .unwrap_or(deepseek_v4::N_ROUTED_EXPERTS),
            num_experts_per_tok: usize_key(json, &["num_experts_per_tok"])
                .unwrap_or(deepseek_v4::NUM_EXPERTS_PER_TOK),
            moe_intermediate_size: usize_key(json, &["moe_intermediate_size", "moe_inter_dim"])
                .unwrap_or(deepseek_v4::MOE_INTERMEDIATE_SIZE),
            swiglu_limit: f32_key(json, &["swiglu_limit"]).unwrap_or(deepseek_v4::SWIGLU_LIMIT),
            route_scale: f32_key(json, &["routed_scaling_factor", "route_scale"])
                .unwrap_or(deepseek_v4::ROUTED_SCALING_FACTOR),
            rope_theta: f32_key(json, &["rope_theta"]).unwrap_or(deepseek_v4::ROPE_THETA),
            compress_rope_theta: f32_key(json, &["compress_rope_theta"])
                .unwrap_or(deepseek_v4::COMPRESS_ROPE_THETA),
            original_seq_len: usize_key(rope_scaling, &["original_max_position_embeddings"])
                .or_else(|| usize_key(json, &["original_seq_len"]))
                .unwrap_or(deepseek_v4::ORIGINAL_MAX_POSITION_EMBEDDINGS),
            rope_factor: f32_key(rope_scaling, &["factor"])
                .or_else(|| f32_key(json, &["rope_factor"]))
                .unwrap_or(deepseek_v4::ROPE_FACTOR),
            beta_fast: usize_key(rope_scaling, &["beta_fast"])
                .or_else(|| usize_key(json, &["beta_fast"]))
                .unwrap_or(deepseek_v4::ROPE_BETA_FAST),
            beta_slow: usize_key(rope_scaling, &["beta_slow"])
                .or_else(|| usize_key(json, &["beta_slow"]))
                .unwrap_or(deepseek_v4::ROPE_BETA_SLOW),
            index_n_heads: usize_key(json, &["index_n_heads"])
                .unwrap_or(deepseek_v4::INDEX_N_HEADS),
            index_head_dim: usize_key(json, &["index_head_dim"])
                .unwrap_or(deepseek_v4::INDEX_HEAD_DIM),
            index_topk: usize_key(json, &["index_topk"]).unwrap_or(deepseek_v4::INDEX_TOPK),
            compress_ratios,
            proposal_block_size: json
                .get("proposal_block_size")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)
                .unwrap_or(deepseek_v4::PROPOSAL_BLOCK_SIZE),
            proposal_noise_token_id: json
                .get("proposal_noise_token_id")
                .and_then(|value| value.as_u64())
                .map(|value| value as u32),
            proposal_target_layer_ids,
            proposal_markov_rank: json
                .get("proposal_markov_rank")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize),
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("max_position_embeddings", self.max_position_embeddings),
            ("hidden_size", self.hidden_size),
            ("hc_mult", self.hc_mult),
            ("hc_sinkhorn_iters", self.hc_sinkhorn_iters),
            ("num_layers", self.num_layers),
            ("num_heads", self.num_heads),
            ("head_dim", self.head_dim),
            ("q_lora_rank", self.q_lora_rank),
            ("qk_rope_head_dim", self.qk_rope_head_dim),
            ("o_groups", self.o_groups),
            ("o_lora_rank", self.o_lora_rank),
            ("window_size", self.window_size),
            ("vocab_size", self.vocab_size),
            ("num_routed_experts", self.num_routed_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("index_n_heads", self.index_n_heads),
            ("index_head_dim", self.index_head_dim),
            ("index_topk", self.index_topk),
        ] {
            if value == 0 {
                return Err(config_error(format!("{name} must be non-zero")));
            }
        }
        for (name, value) in [
            ("hc_eps", self.hc_eps),
            ("norm_eps", self.norm_eps),
            ("swiglu_limit", self.swiglu_limit),
            ("route_scale", self.route_scale),
            ("rope_theta", self.rope_theta),
            ("compress_rope_theta", self.compress_rope_theta),
            ("rope_factor", self.rope_factor),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(config_error(format!("{name} must be finite and positive")));
            }
        }
        if self.architecture != "DeepseekV4ForCausalLM"
            || self.num_hash_layers > self.num_layers
            || self.num_experts_per_tok > self.num_routed_experts
            || self.qk_rope_head_dim > self.head_dim
            || !self.qk_rope_head_dim.is_multiple_of(2)
            || !self.num_heads.is_multiple_of(self.o_groups)
            || !self.hidden_size.is_multiple_of(32)
            || !self.moe_intermediate_size.is_multiple_of(32)
            || self.compress_ratios.len() < self.num_layers
            || self.compress_ratios[self.num_layers..]
                .iter()
                .any(|ratio| *ratio != 0)
            || self
                .proposal_target_layer_ids
                .iter()
                .any(|layer| *layer >= self.num_layers)
            || self
                .proposal_target_layer_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(config_error(
                "dimensions, routing, architecture, or proposal targets are inconsistent",
            ));
        }
        for (name, value) in [
            (
                "attention query width",
                self.num_heads.checked_mul(self.head_dim),
            ),
            (
                "attention output latent width",
                self.o_groups.checked_mul(self.o_lora_rank),
            ),
            (
                "index query width",
                self.index_n_heads.checked_mul(self.index_head_dim),
            ),
            (
                "hyper-connection width",
                self.hc_mult.checked_mul(self.hidden_size),
            ),
            (
                "proposal projection width",
                self.hidden_size
                    .checked_mul(self.proposal_target_layer_ids.len()),
            ),
        ] {
            if value.is_none() {
                return Err(config_error(format!("{name} overflows usize")));
            }
        }
        if self
            .proposal_markov_rank
            .is_some_and(|rank| self.hidden_size.checked_add(rank).is_none())
        {
            return Err(config_error("proposal confidence width overflows usize"));
        }
        #[cfg(feature = "cuda")]
        {
            let frequency = crate::models::common::rope::yarn_frequency(
                0,
                self.qk_rope_head_dim,
                crate::models::common::rope::RopeParams {
                    theta: self.compress_rope_theta,
                    original_seq_len: self.original_seq_len,
                    factor: self.rope_factor,
                    beta_fast: self.beta_fast,
                    beta_slow: self.beta_slow,
                },
            );
            if !frequency.is_finite() || frequency <= 0.0 {
                return Err(config_error("compressed RoPE frequency is invalid"));
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    /// Returns the model-independent MTP configuration.
    pub(super) fn mtp_config(&self) -> crate::transformer::proposal::MtpConfig {
        crate::transformer::proposal::MtpConfig {
            block_size: self.proposal_block_size,
            noise_token_id: self.proposal_noise_token_id,
            target_layer_ids: self.proposal_target_layer_ids.clone(),
            markov_rank: self.proposal_markov_rank,
        }
    }
    pub(super) fn hc_config(&self) -> HyperConnectionConfig {
        HyperConnectionConfig {
            hc_mult: self.hc_mult,
            hidden_size: self.hidden_size,
            sinkhorn_iters: self.hc_sinkhorn_iters,
            eps: self.hc_eps,
            norm_eps: self.norm_eps,
        }
    }

    /// Returns an attention configuration for one proposal transformer stage.
    ///
    #[cfg(feature = "cuda")]
    /// Proposal stages share the target MLA structure but do not use compressed
    /// attention.
    pub(super) fn attention_config_for_proposal_stage(
        &self,
        _proposal_stage: usize,
    ) -> Result<MlaConfig> {
        Ok(MlaConfig {
            hidden_size: self.hidden_size,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            q_lora_rank: self.q_lora_rank,
            rope_head_dim: self.qk_rope_head_dim,
            o_groups: self.o_groups,
            o_lora_rank: self.o_lora_rank,
            window_size: self.window_size,
            compress_ratio: 0,
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

    #[cfg(feature = "cuda")]
    pub(super) fn attention_config_for_layer(&self, layer: usize) -> Result<MlaConfig> {
        if layer >= self.num_layers {
            return Err(Error::Model {
                message: format!(
                    "DeepSeek-V4 layer {layer} exceeds layer count {}",
                    self.num_layers
                ),
            });
        }
        Ok(MlaConfig {
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

fn config_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("DeepSeek-V4 config: {}", message.into()),
    }
}
