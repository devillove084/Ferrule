use crate::TensorRole;
use crate::checkpoint::weight::{LinearExecutionPolicy, LinearWeight, LinearWeightFormat};
use crate::ffn::SwiGluFfnPayload;
use crate::models::common::rope::{RopeParams, yarn_frequency};
use crate::transformer::attention::mla::{MlaConfig, MlaWeights};
use ferrule_common::{Error, Result};

use super::DeepSeekV4Config;

const DSV4_LINEAR_ACTIVATION_QUANT_BLOCK_SIZE: usize = 128;

fn deepseek_v4_quantized_linear_execution_policy() -> LinearExecutionPolicy {
    LinearExecutionPolicy::fp8_e4m3_e8m0_activation(DSV4_LINEAR_ACTIVATION_QUANT_BLOCK_SIZE)
}

pub(crate) fn with_deepseek_v4_linear_execution_policy(linear: LinearWeight) -> LinearWeight {
    if deepseek_v4_role_uses_official_linear_activation_quantization(&linear.role)
        && artifact_linear_format_has_quantized_weight(&linear.format)
    {
        linear.with_execution_policy(deepseek_v4_quantized_linear_execution_policy())
    } else {
        linear
    }
}

pub(crate) fn with_deepseek_v4_attention_execution_policies(mut payload: MlaWeights) -> MlaWeights {
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
            | TensorRole::SpeculativeProjection
    )
}

fn artifact_linear_format_has_quantized_weight(format: &LinearWeightFormat) -> bool {
    matches!(
        format,
        LinearWeightFormat::Fp8E4M3WithE8M0Scale { .. }
            | LinearWeightFormat::Fp4E2M1PackedWithE8M0Scale { .. }
    )
}

impl DeepSeekV4Config {
    pub(super) fn validate_cuda_rope(&self) -> Result<()> {
        let frequency = yarn_frequency(
            0,
            self.qk_rope_head_dim,
            RopeParams {
                theta: self.compress_rope_theta,
                original_seq_len: self.original_seq_len,
                factor: self.rope_factor,
                beta_fast: self.beta_fast,
                beta_slow: self.beta_slow,
            },
        );
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(super::config_error("compressed RoPE frequency is invalid"));
        }
        Ok(())
    }

    /// Returns the model-independent MTP configuration.
    pub(in crate::models::deepseek_v4) fn mtp_config(
        &self,
    ) -> crate::transformer::proposal::MtpConfig {
        crate::transformer::proposal::MtpConfig {
            block_size: self.proposal_block_size,
            noise_token_id: self.proposal_noise_token_id,
            target_layer_ids: self.proposal_target_layer_ids.clone(),
            markov_rank: self.proposal_markov_rank,
        }
    }

    /// Returns an attention configuration for one proposal transformer stage.
    ///
    /// Proposal stages share the target MLA structure but do not use compressed
    /// attention.
    pub(in crate::models::deepseek_v4) fn attention_config_for_proposal_stage(
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

    pub(in crate::models::deepseek_v4) fn attention_config_for_layer(
        &self,
        layer: usize,
    ) -> Result<MlaConfig> {
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
