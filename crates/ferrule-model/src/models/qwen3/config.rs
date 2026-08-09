//! Strict Qwen3-MoE Hugging Face configuration.

use ferrule_common::{Error, Result};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen3MoeConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: Option<serde_json::Value>,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub output_router_logits: bool,
    pub router_aux_loss_coef: f32,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub mlp_only_layers: Vec<usize>,
    pub decoder_sparse_step: usize,
    pub hidden_act: String,
    pub torch_dtype: String,
    pub use_cache: bool,
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub initializer_range: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    pub transformers_version: Option<String>,
}

impl Qwen3MoeConfig {
    pub(super) fn from_value(value: &serde_json::Value) -> Result<Self> {
        reject_shared_experts(value)?;
        let config: Self =
            serde_json::from_value(value.clone()).map_err(|source| Error::ModelSource {
                source: Box::new(source),
            })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.model_type != "qwen3_moe"
            || self.architectures.as_slice() != ["Qwen3MoeForCausalLM"]
        {
            return config_error(format!(
                "requires model_type=qwen3_moe and architecture Qwen3MoeForCausalLM, got '{}' {:?}",
                self.model_type, self.architectures
            ));
        }
        for (field, value) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("max_position_embeddings", self.max_position_embeddings),
            ("vocab_size", self.vocab_size),
            ("num_experts", self.num_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
        ] {
            if value == 0 {
                return config_error(format!("{field} must be non-zero"));
            }
        }
        if !self.head_dim.is_multiple_of(2) {
            return config_error("head_dim must be even for split-half RoPE");
        }
        if self.num_key_value_heads > self.num_attention_heads
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return config_error("num_attention_heads must be divisible by num_key_value_heads");
        }
        if self.num_experts_per_tok > self.num_experts {
            return config_error("num_experts_per_tok exceeds num_experts");
        }
        self.num_attention_heads
            .checked_mul(self.head_dim)
            .and_then(|query| {
                self.num_key_value_heads
                    .checked_mul(self.head_dim)
                    .and_then(|key_value| query.checked_add(key_value))
            })
            .ok_or_else(|| model_error("attention projection size overflow"))?;
        self.expected_tensor_count()?;
        positive_finite("rms_norm_eps", self.rms_norm_eps)?;
        positive_finite("rope_theta", self.rope_theta)?;
        positive_finite("initializer_range", self.initializer_range)?;
        if !self.router_aux_loss_coef.is_finite() || self.router_aux_loss_coef < 0.0 {
            return config_error("router_aux_loss_coef must be finite and non-negative");
        }
        if self.attention_bias || self.attention_dropout != 0.0 {
            return config_error("only bias-free attention with zero dropout is supported");
        }
        if self.decoder_sparse_step != 1 || !self.mlp_only_layers.is_empty() {
            return config_error("only all-layer MoE is supported");
        }
        if self.hidden_act != "silu" {
            return config_error("hidden_act must be silu");
        }
        if !matches!(self.torch_dtype.as_str(), "bfloat16" | "bf16") {
            return config_error("torch_dtype must be bfloat16");
        }
        if !self.norm_topk_prob || self.output_router_logits {
            return config_error("router must use selected renormalization without output logits");
        }
        if !self.use_cache {
            return config_error("use_cache must be true");
        }
        if self.rope_scaling.is_some() {
            return config_error("rope_scaling must be null");
        }
        if self.use_sliding_window || self.sliding_window.is_some() {
            return config_error("sliding-window attention is unsupported");
        }
        if self.max_window_layers == 0 || self.max_window_layers > self.num_hidden_layers {
            return config_error("max_window_layers is outside the decoder layer range");
        }
        if self.bos_token_id as usize >= self.vocab_size
            || self.eos_token_id as usize >= self.vocab_size
        {
            return config_error("BOS/EOS token IDs must be below vocab_size");
        }
        Ok(())
    }

    fn expected_tensor_count(&self) -> Result<usize> {
        let per_layer = self
            .num_experts
            .checked_mul(3)
            .and_then(|experts| experts.checked_add(9))
            .ok_or_else(|| model_error("per-layer tensor count overflow"))?;
        self.num_hidden_layers
            .checked_mul(per_layer)
            .and_then(|layers| layers.checked_add(if self.tie_word_embeddings { 2 } else { 3 }))
            .ok_or_else(|| model_error("checkpoint tensor count overflow"))
    }
}

fn reject_shared_experts(value: &serde_json::Value) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| model_error("top-level config must be an object"))?;
    if object.iter().any(|(key, value)| {
        key.contains("shared_expert")
            && !matches!(
                value,
                serde_json::Value::Null | serde_json::Value::Bool(false)
            )
    }) {
        return config_error("shared experts are unsupported");
    }
    Ok(())
}

fn positive_finite(field: &str, value: f32) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        config_error(format!("{field} must be finite and positive"))
    }
}

fn config_error(message: impl Into<String>) -> Result<()> {
    Err(model_error(message))
}

fn model_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("Qwen3-MoE config: {}", message.into()),
    }
}
