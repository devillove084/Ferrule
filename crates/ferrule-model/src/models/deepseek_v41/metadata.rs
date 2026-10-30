//! Strict DeepSeek-V4.1 metadata-only schema and HF artifact validator.
//!
//! This module intentionally stops at config/index/safetensors-header metadata.
//! It does not construct a recipe, bind parameters, read tensor payloads, or
//! select a runtime/backend implementation.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use ferrule_common::{Error, Result};

use crate::checkpoint::{HfSafetensorsIndex, HfSafetensorsInventory, HfSafetensorsTensorInfo};
use crate::{
    AttentionKind, ModelDescriptor, ModelFamily, MoeSpec, QuantFormatCount, RouterKind,
    TransformerSemantics, TransformerSpec, WeightSource,
};

const MODEL_TYPE: &str = "deepseek_v41";
const ARCHITECTURE: &str = "DeepseekV41ForCausalLM";
const TEXT_MODEL_TYPE: &str = "deepseek_v41_text";
pub const DEEPSEEK_V41_REAL_SHARD_COUNT: usize = 48;
pub const DEEPSEEK_V41_REAL_TOTAL_SIZE: u64 = 510_286_023_000;
const MAX_SHARDS: usize = 1024;
const MAX_VISION_WH_RATIO: f32 = 1024.0;

fn model_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("DeepSeek-V4.1 metadata: {}", message.into()),
    }
}

fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    condition.then_some(()).ok_or_else(|| model_error(message))
}

fn object<'a>(
    value: &'a serde_json::Value,
    context: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>> {
    value
        .as_object()
        .ok_or_else(|| model_error(format!("{context} must be a JSON object")))
}

fn reject_unknown_fields(
    object: &serde_json::Map<String, serde_json::Value>,
    allowed: &[&str],
    context: &str,
) -> Result<()> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(model_error(format!("unsupported {context} field '{key}'")));
        }
    }
    Ok(())
}

fn string_field(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    context: &str,
) -> Result<String> {
    object
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| model_error(format!("{context}.{key} must be a string")))
}

fn usize_field(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    context: &str,
) -> Result<usize> {
    object
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| model_error(format!("{context}.{key} must be a non-negative integer")))
}

fn optional_usize_field(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    context: &str,
) -> Result<Option<usize>> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .map(Some)
        .ok_or_else(|| model_error(format!("{context}.{key} must be a non-negative integer")))
}

fn f32_field(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    context: &str,
) -> Result<f32> {
    let value = object
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| model_error(format!("{context}.{key} must be a number")))?
        as f32;
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| model_error(format!("{context}.{key} must be finite")))
}

fn optional_usize_array(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    context: &str,
) -> Result<Option<Vec<usize>>> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    let array = value
        .as_array()
        .ok_or_else(|| model_error(format!("{context}.{key} must be an array")))?;
    let mut output = Vec::with_capacity(array.len());
    for item in array {
        output.push(
            item.as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| model_error(format!("{context}.{key} contains a non-integer")))?,
        );
    }
    Ok(Some(output))
}

fn required_usize_array(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    context: &str,
) -> Result<Vec<usize>> {
    optional_usize_array(object, key, context)?
        .ok_or_else(|| model_error(format!("{context}.{key} is required")))
}

fn exact_usize(value: usize, expected: usize, field: &str) -> Result<()> {
    require(
        value == expected,
        format!("{field} must be {expected}, got {value}"),
    )
}

/// Storage dtypes accepted by the V4.1 safetensors contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeepSeekV41DType {
    F8E4M3,
    F8E8M0,
    BF16,
    F32,
    I8,
}

impl DeepSeekV41DType {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "F8_E4M3" => Ok(Self::F8E4M3),
            "F8_E8M0" => Ok(Self::F8E8M0),
            "BF16" => Ok(Self::BF16),
            "F32" => Ok(Self::F32),
            "I8" => Ok(Self::I8),
            other => Err(model_error(format!(
                "unsupported or unknown safetensors dtype '{other}'"
            ))),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E8M0 => "F8_E8M0",
            Self::BF16 => "BF16",
            Self::F32 => "F32",
            Self::I8 => "I8",
        }
    }

    const fn element_bytes(self) -> u64 {
        match self {
            Self::F8E4M3 | Self::F8E8M0 | Self::I8 => 1,
            Self::BF16 => 2,
            Self::F32 => 4,
        }
    }
}

/// The exact quantization contract recorded by V4.1 HF config metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV41QuantizationContract {
    pub quant_method: String,
    pub activation_scheme: String,
    pub weight_block_size: [usize; 2],
    pub scale_fmt: String,
    pub expert_dtype: String,
}

impl DeepSeekV41QuantizationContract {
    fn from_value(value: &serde_json::Value) -> Result<Self> {
        let object = object(value, "quantization_config")?;
        reject_unknown_fields(
            object,
            &[
                "quant_method",
                "activation_scheme",
                "weight_block_size",
                "scale_fmt",
                "expert_dtype",
            ],
            "quantization_config",
        )?;
        let quant_method = string_field(object, "quant_method", "quantization_config")?;
        let activation_scheme = string_field(object, "activation_scheme", "quantization_config")?;
        let block = required_usize_array(object, "weight_block_size", "quantization_config")?;
        let scale_fmt = string_field(object, "scale_fmt", "quantization_config")?;
        let expert_dtype = string_field(object, "expert_dtype", "quantization_config")?;
        require(
            quant_method == "fp8",
            "quantization_config.quant_method must be 'fp8'",
        )?;
        require(
            activation_scheme == "dynamic",
            "quantization_config.activation_scheme must be 'dynamic'",
        )?;
        require(
            block == [32, 32],
            format!("quantization_config.weight_block_size must be [32, 32], got {block:?}"),
        )?;
        require(
            scale_fmt == "ue8m0",
            "quantization_config.scale_fmt must be 'ue8m0'",
        )?;
        require(
            expert_dtype == "fp4",
            "quantization_config.expert_dtype must be 'fp4'",
        )?;
        Ok(Self {
            quant_method,
            activation_scheme,
            weight_block_size: [32, 32],
            scale_fmt,
            expert_dtype,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV41EngramConfig {
    pub layer_ids: Vec<usize>,
    pub num_embeddings: Vec<usize>,
    pub max_ngram_size: usize,
    pub vocab_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub pad_token_id: usize,
    pub compressed_vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV41VisionConfig {
    pub model_type: String,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub rope_theta: f32,
    pub downsample_ratio: usize,
    pub max_image_tokens: usize,
    pub min_pixels: usize,
    pub max_wh_ratio: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV41TextConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub hidden_act: String,
    pub swiglu_limit: f32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub rope_original_max_position_embeddings: usize,
    pub rope_beta_fast: usize,
    pub rope_beta_slow: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub scoring_func: String,
    pub topk_method: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub sliding_window: usize,
    pub compress_ratios: Vec<usize>,
    pub compress_rope_theta: f32,
    pub kv_source_layer_ids: Vec<usize>,
    pub index_source_layer_ids: Vec<usize>,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub num_nextn_predict_layers: usize,
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: usize,
    pub dspark_target_layer_ids: Vec<usize>,
    pub dspark_markov_rank: usize,
    pub dspark_n_routed_experts: usize,
    pub dspark_num_experts_per_tok: usize,
    pub engram: Option<DeepSeekV41EngramConfig>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV41Config {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub dtype: String,
    pub quantization: DeepSeekV41QuantizationContract,
    pub text: DeepSeekV41TextConfig,
    pub vision: Option<DeepSeekV41VisionConfig>,
    pub image_token_id: Option<usize>,
}

impl DeepSeekV41Config {
    pub fn from_value(value: &serde_json::Value) -> Result<Self> {
        let root = object(value, "DeepSeek-V4.1 config")?;
        reject_unknown_fields(
            root,
            &[
                "architectures",
                "model_type",
                "dtype",
                "transformers_version",
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "image_token_id",
                "quantization_config",
                "text_config",
                "vision_config",
            ],
            "config",
        )?;
        if root.get("transformers_version").is_some() {
            string_field(root, "transformers_version", "config")?;
        }
        for key in ["bos_token_id", "eos_token_id", "pad_token_id"] {
            optional_usize_field(root, key, "config")?;
        }
        let model_type = string_field(root, "model_type", "config")?;
        require(
            model_type == MODEL_TYPE,
            format!("model_type must be '{MODEL_TYPE}', got '{model_type}'"),
        )?;
        let architectures = root
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| model_error("architectures must be a one-item array"))?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(ToOwned::to_owned)
                    .ok_or_else(|| model_error("architectures must contain strings"))
            })
            .collect::<Result<Vec<_>>>()?;
        require(
            architectures == [ARCHITECTURE.to_owned()],
            format!("architectures must be ['{ARCHITECTURE}'], got {architectures:?}"),
        )?;
        let dtype = string_field(root, "dtype", "config")?;
        require(dtype == "bfloat16", "config.dtype must be 'bfloat16'")?;
        let quantization = DeepSeekV41QuantizationContract::from_value(
            root.get("quantization_config")
                .ok_or_else(|| model_error("quantization_config is required"))?,
        )?;
        let text = parse_text_config(
            root.get("text_config")
                .ok_or_else(|| model_error("text_config is required"))?,
        )?;
        let vision = root
            .get("vision_config")
            .map(parse_vision_config)
            .transpose()?;
        let config = Self {
            model_type,
            architectures,
            dtype,
            quantization,
            text,
            vision,
            image_token_id: optional_usize_field(root, "image_token_id", "config")?,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        require(
            self.model_type == MODEL_TYPE,
            format!("model_type must be '{MODEL_TYPE}'"),
        )?;
        require(
            self.architectures == [ARCHITECTURE.to_owned()],
            format!("architectures must be ['{ARCHITECTURE}']"),
        )?;
        require(self.dtype == "bfloat16", "config.dtype must be 'bfloat16'")?;
        require(
            self.quantization.quant_method == "fp8",
            "quantization_config.quant_method must be 'fp8'",
        )?;
        require(
            self.quantization.activation_scheme == "dynamic",
            "quantization_config.activation_scheme must be 'dynamic'",
        )?;
        require(
            self.quantization.weight_block_size == [32, 32],
            "quantization_config.weight_block_size must be [32, 32]",
        )?;
        require(
            self.quantization.scale_fmt == "ue8m0",
            "quantization_config.scale_fmt must be 'ue8m0'",
        )?;
        require(
            self.quantization.expert_dtype == "fp4",
            "quantization_config.expert_dtype must be 'fp4'",
        )?;
        validate_text_config(&self.text)?;
        if let Some(vision) = &self.vision {
            validate_vision_config(vision)?;
        }
        Ok(())
    }
}

fn parse_text_config(value: &serde_json::Value) -> Result<DeepSeekV41TextConfig> {
    let text_object = object(value, "text_config")?;
    reject_unknown_fields(
        text_object,
        &[
            "model_type",
            "vocab_size",
            "hidden_size",
            "moe_intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "qk_rope_head_dim",
            "q_lora_rank",
            "o_lora_rank",
            "o_groups",
            "hidden_act",
            "swiglu_limit",
            "rms_norm_eps",
            "attention_bias",
            "attention_dropout",
            "initializer_range",
            "use_cache",
            "tie_word_embeddings",
            "max_position_embeddings",
            "rope_theta",
            "rope_scaling",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "scoring_func",
            "topk_method",
            "norm_topk_prob",
            "routed_scaling_factor",
            "sliding_window",
            "compress_ratios",
            "compress_rope_theta",
            "kv_source_layer_ids",
            "index_source_layer_ids",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "candidate_source_layer_id",
            "candidate_topk_blocks",
            "candidate_block_size",
            "hc_mult",
            "hc_sinkhorn_iters",
            "hc_eps",
            "engram_layer_ids",
            "engram_num_embeddings",
            "engram_max_ngram_size",
            "engram_vocab_size",
            "engram_n_heads",
            "engram_head_dim",
            "engram_pad_token_id",
            "engram_compressed_vocab_size",
            "num_nextn_predict_layers",
            "dspark_block_size",
            "dspark_noise_token_id",
            "dspark_target_layer_ids",
            "dspark_markov_rank",
            "dspark_n_routed_experts",
            "dspark_num_experts_per_tok",
        ],
        "text_config",
    )?;
    let rope = object(
        text_object
            .get("rope_scaling")
            .ok_or_else(|| model_error("text_config.rope_scaling is required"))?,
        "text_config.rope_scaling",
    )?;
    reject_unknown_fields(
        rope,
        &[
            "rope_type",
            "factor",
            "beta_fast",
            "beta_slow",
            "original_max_position_embeddings",
        ],
        "text_config.rope_scaling",
    )?;
    require(
        text_object
            .get("attention_bias")
            .and_then(serde_json::Value::as_bool)
            == Some(false),
        "text_config.attention_bias must be false",
    )?;
    require(
        text_object
            .get("attention_dropout")
            .and_then(serde_json::Value::as_f64)
            .is_some_and(|value| value == 0.0),
        "text_config.attention_dropout must be 0",
    )?;
    require(
        text_object
            .get("initializer_range")
            .and_then(serde_json::Value::as_f64)
            .is_some_and(|value| (value - 0.02).abs() < f64::EPSILON),
        "text_config.initializer_range must be 0.02",
    )?;
    require(
        text_object
            .get("use_cache")
            .and_then(serde_json::Value::as_bool)
            == Some(true),
        "text_config.use_cache must be true",
    )?;
    require(
        text_object
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            == Some(false),
        "text_config.tie_word_embeddings must be false",
    )?;
    exact_usize(
        usize_field(text_object, "candidate_source_layer_id", "text_config")?,
        20,
        "text_config.candidate_source_layer_id",
    )?;
    exact_usize(
        usize_field(text_object, "candidate_topk_blocks", "text_config")?,
        2048,
        "text_config.candidate_topk_blocks",
    )?;
    exact_usize(
        usize_field(text_object, "candidate_block_size", "text_config")?,
        8,
        "text_config.candidate_block_size",
    )?;
    require(
        string_field(rope, "rope_type", "text_config.rope_scaling")? == "yarn",
        "text_config.rope_scaling.rope_type must be 'yarn'",
    )?;
    let engram_keys = [
        "engram_layer_ids",
        "engram_num_embeddings",
        "engram_max_ngram_size",
        "engram_vocab_size",
        "engram_n_heads",
        "engram_head_dim",
        "engram_pad_token_id",
        "engram_compressed_vocab_size",
    ];
    let engram_present = engram_keys
        .iter()
        .filter(|key| text_object.contains_key(**key))
        .count();
    let engram = if engram_present == 0 {
        None
    } else {
        require(
            engram_present == engram_keys.len(),
            "text_config Engram metadata must be complete when present",
        )?;
        Some(DeepSeekV41EngramConfig {
            layer_ids: required_usize_array(text_object, "engram_layer_ids", "text_config")?,
            num_embeddings: required_usize_array(
                text_object,
                "engram_num_embeddings",
                "text_config",
            )?,
            max_ngram_size: usize_field(text_object, "engram_max_ngram_size", "text_config")?,
            vocab_size: usize_field(text_object, "engram_vocab_size", "text_config")?,
            num_heads: usize_field(text_object, "engram_n_heads", "text_config")?,
            head_dim: usize_field(text_object, "engram_head_dim", "text_config")?,
            pad_token_id: usize_field(text_object, "engram_pad_token_id", "text_config")?,
            compressed_vocab_size: usize_field(
                text_object,
                "engram_compressed_vocab_size",
                "text_config",
            )?,
        })
    };
    Ok(DeepSeekV41TextConfig {
        model_type: string_field(text_object, "model_type", "text_config")?,
        vocab_size: usize_field(text_object, "vocab_size", "text_config")?,
        hidden_size: usize_field(text_object, "hidden_size", "text_config")?,
        moe_intermediate_size: usize_field(text_object, "moe_intermediate_size", "text_config")?,
        num_hidden_layers: usize_field(text_object, "num_hidden_layers", "text_config")?,
        num_attention_heads: usize_field(text_object, "num_attention_heads", "text_config")?,
        num_key_value_heads: usize_field(text_object, "num_key_value_heads", "text_config")?,
        head_dim: usize_field(text_object, "head_dim", "text_config")?,
        qk_rope_head_dim: usize_field(text_object, "qk_rope_head_dim", "text_config")?,
        q_lora_rank: usize_field(text_object, "q_lora_rank", "text_config")?,
        o_lora_rank: usize_field(text_object, "o_lora_rank", "text_config")?,
        o_groups: usize_field(text_object, "o_groups", "text_config")?,
        hidden_act: string_field(text_object, "hidden_act", "text_config")?,
        swiglu_limit: f32_field(text_object, "swiglu_limit", "text_config")?,
        rms_norm_eps: f32_field(text_object, "rms_norm_eps", "text_config")?,
        max_position_embeddings: usize_field(
            text_object,
            "max_position_embeddings",
            "text_config",
        )?,
        rope_theta: f32_field(text_object, "rope_theta", "text_config")?,
        rope_factor: f32_field(rope, "factor", "text_config.rope_scaling")?,
        rope_original_max_position_embeddings: usize_field(
            rope,
            "original_max_position_embeddings",
            "text_config.rope_scaling",
        )?,
        rope_beta_fast: usize_field(rope, "beta_fast", "text_config.rope_scaling")?,
        rope_beta_slow: usize_field(rope, "beta_slow", "text_config.rope_scaling")?,
        n_routed_experts: usize_field(text_object, "n_routed_experts", "text_config")?,
        n_shared_experts: usize_field(text_object, "n_shared_experts", "text_config")?,
        num_experts_per_tok: usize_field(text_object, "num_experts_per_tok", "text_config")?,
        scoring_func: string_field(text_object, "scoring_func", "text_config")?,
        topk_method: string_field(text_object, "topk_method", "text_config")?,
        norm_topk_prob: text_object
            .get("norm_topk_prob")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| model_error("text_config.norm_topk_prob must be boolean"))?,
        routed_scaling_factor: f32_field(text_object, "routed_scaling_factor", "text_config")?,
        sliding_window: usize_field(text_object, "sliding_window", "text_config")?,
        compress_ratios: required_usize_array(text_object, "compress_ratios", "text_config")?,
        compress_rope_theta: f32_field(text_object, "compress_rope_theta", "text_config")?,
        kv_source_layer_ids: required_usize_array(
            text_object,
            "kv_source_layer_ids",
            "text_config",
        )?,
        index_source_layer_ids: required_usize_array(
            text_object,
            "index_source_layer_ids",
            "text_config",
        )?,
        index_n_heads: usize_field(text_object, "index_n_heads", "text_config")?,
        index_head_dim: usize_field(text_object, "index_head_dim", "text_config")?,
        index_topk: usize_field(text_object, "index_topk", "text_config")?,
        hc_mult: usize_field(text_object, "hc_mult", "text_config")?,
        hc_sinkhorn_iters: usize_field(text_object, "hc_sinkhorn_iters", "text_config")?,
        hc_eps: f32_field(text_object, "hc_eps", "text_config")?,
        num_nextn_predict_layers: usize_field(
            text_object,
            "num_nextn_predict_layers",
            "text_config",
        )?,
        dspark_block_size: usize_field(text_object, "dspark_block_size", "text_config")?,
        dspark_noise_token_id: usize_field(text_object, "dspark_noise_token_id", "text_config")?,
        dspark_target_layer_ids: required_usize_array(
            text_object,
            "dspark_target_layer_ids",
            "text_config",
        )?,
        dspark_markov_rank: usize_field(text_object, "dspark_markov_rank", "text_config")?,
        dspark_n_routed_experts: usize_field(
            text_object,
            "dspark_n_routed_experts",
            "text_config",
        )?,
        dspark_num_experts_per_tok: usize_field(
            text_object,
            "dspark_num_experts_per_tok",
            "text_config",
        )?,
        engram,
    })
}

fn validate_text_config(config: &DeepSeekV41TextConfig) -> Result<()> {
    require(
        config.model_type == TEXT_MODEL_TYPE,
        format!("text_config.model_type must be '{TEXT_MODEL_TYPE}'"),
    )?;
    for (field, value, expected) in [
        ("hidden_size", config.hidden_size, 5120),
        ("vocab_size", config.vocab_size, 129280),
        ("moe_intermediate_size", config.moe_intermediate_size, 2304),
        ("num_hidden_layers", config.num_hidden_layers, 40),
        ("num_attention_heads", config.num_attention_heads, 64),
        ("num_key_value_heads", config.num_key_value_heads, 1),
        ("head_dim", config.head_dim, 512),
        ("qk_rope_head_dim", config.qk_rope_head_dim, 64),
        ("q_lora_rank", config.q_lora_rank, 1280),
        ("o_lora_rank", config.o_lora_rank, 1024),
        ("o_groups", config.o_groups, 8),
        (
            "max_position_embeddings",
            config.max_position_embeddings,
            1_048_576,
        ),
        ("n_routed_experts", config.n_routed_experts, 384),
        ("n_shared_experts", config.n_shared_experts, 1),
        ("num_experts_per_tok", config.num_experts_per_tok, 6),
        ("sliding_window", config.sliding_window, 128),
        ("index_n_heads", config.index_n_heads, 32),
        ("index_head_dim", config.index_head_dim, 128),
        ("index_topk", config.index_topk, 512),
        ("hc_mult", config.hc_mult, 4),
        ("hc_sinkhorn_iters", config.hc_sinkhorn_iters, 20),
        (
            "num_nextn_predict_layers",
            config.num_nextn_predict_layers,
            3,
        ),
        ("dspark_block_size", config.dspark_block_size, 5),
        ("dspark_markov_rank", config.dspark_markov_rank, 256),
        (
            "dspark_n_routed_experts",
            config.dspark_n_routed_experts,
            128,
        ),
        (
            "dspark_num_experts_per_tok",
            config.dspark_num_experts_per_tok,
            3,
        ),
    ] {
        exact_usize(value, expected, &format!("text_config.{field}"))?;
    }
    require(
        config.hidden_act == "silu",
        "text_config.hidden_act must be 'silu'",
    )?;
    require(
        config.scoring_func == "sqrtsoftplus",
        "text_config.scoring_func must be 'sqrtsoftplus'",
    )?;
    require(
        config.topk_method == "noaux_tc",
        "text_config.topk_method must be 'noaux_tc'",
    )?;
    require(
        config.norm_topk_prob,
        "text_config.norm_topk_prob must be true",
    )?;
    require(
        (config.swiglu_limit - 10.0).abs() < f32::EPSILON,
        "text_config.swiglu_limit must be 10",
    )?;
    require(
        (config.rms_norm_eps - 1e-20).abs() < 1e-25,
        "text_config.rms_norm_eps must be 1e-20",
    )?;
    require(
        (config.routed_scaling_factor - 1.5).abs() < f32::EPSILON,
        "text_config.routed_scaling_factor must be 1.5",
    )?;
    require(
        (config.rope_theta - 10_000.0).abs() < f32::EPSILON,
        "text_config.rope_theta must be 10000",
    )?;
    require(
        (config.rope_factor - 16.0).abs() < f32::EPSILON,
        "text_config.rope_scaling.factor must be 16",
    )?;
    exact_usize(
        config.rope_original_max_position_embeddings,
        65_536,
        "text_config.rope_scaling.original_max_position_embeddings",
    )?;
    exact_usize(
        config.rope_beta_fast,
        32,
        "text_config.rope_scaling.beta_fast",
    )?;
    exact_usize(
        config.rope_beta_slow,
        1,
        "text_config.rope_scaling.beta_slow",
    )?;
    require(
        (config.compress_rope_theta - 160_000.0).abs() < f32::EPSILON,
        "text_config.compress_rope_theta must be 160000",
    )?;
    require(
        config.compress_ratios
            == (0..43)
                .map(|layer| match layer {
                    0..=1 | 40..=42 => 0,
                    2..=19 => 2,
                    20..=39 => 1,
                    _ => unreachable!("bounded compress ratio index"),
                })
                .collect::<Vec<_>>(),
        "text_config.compress_ratios must equal the V4.1 40-layer plus MTP schedule",
    )?;
    require(
        config.kv_source_layer_ids == [2, 8, 14, 20],
        "text_config.kv_source_layer_ids is not the V4.1 schema",
    )?;
    require(
        config.index_source_layer_ids == [2, 8, 14, 20, 24, 28, 32, 36],
        "text_config.index_source_layer_ids is not the V4.1 schema",
    )?;
    require(
        config.dspark_target_layer_ids == [37, 38, 39],
        "text_config.dspark_target_layer_ids is not the V4.1 schema",
    )?;
    require(
        config.hc_eps == 1e-6_f32,
        "text_config.hc_eps must be exactly 1e-6",
    )?;
    exact_usize(
        config.dspark_noise_token_id,
        128_799,
        "text_config.dspark_noise_token_id",
    )?;
    if let Some(engram) = &config.engram {
        require(
            engram.layer_ids == [1, 14],
            "Engram layer_ids must be [1, 14]",
        )?;
        require(
            engram.num_embeddings == [384_006_168, 384_016_682],
            "Engram num_embeddings must equal the V4.1 contract",
        )?;
        exact_usize(engram.max_ngram_size, 4, "Engram max_ngram_size")?;
        exact_usize(engram.vocab_size, 16_000_000, "Engram vocab_size")?;
        exact_usize(engram.num_heads, 8, "Engram n_heads")?;
        exact_usize(engram.head_dim, 256, "Engram head_dim")?;
        exact_usize(engram.pad_token_id, 2, "Engram pad_token_id")?;
        exact_usize(
            engram.compressed_vocab_size,
            99_092,
            "Engram compressed_vocab_size",
        )?;
    }
    Ok(())
}

fn parse_vision_config(value: &serde_json::Value) -> Result<DeepSeekV41VisionConfig> {
    let object = object(value, "vision_config")?;
    reject_unknown_fields(
        object,
        &[
            "model_type",
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "intermediate_size",
            "patch_size",
            "rope_theta",
            "downsample_ratio",
            "max_image_tokens",
            "min_pixels",
            "max_wh_ratio",
        ],
        "vision_config",
    )?;
    let max_wh_ratio = match object.get("max_wh_ratio") {
        None | Some(serde_json::Value::Null) => None,
        Some(value) => {
            let value = value
                .as_f64()
                .ok_or_else(|| model_error("vision_config.max_wh_ratio must be null or a number"))?
                as f32;
            require(
                value.is_finite(),
                "vision_config.max_wh_ratio must be finite",
            )?;
            Some(value)
        }
    };
    Ok(DeepSeekV41VisionConfig {
        model_type: string_field(object, "model_type", "vision_config")?,
        num_hidden_layers: usize_field(object, "num_hidden_layers", "vision_config")?,
        hidden_size: usize_field(object, "hidden_size", "vision_config")?,
        num_attention_heads: usize_field(object, "num_attention_heads", "vision_config")?,
        intermediate_size: usize_field(object, "intermediate_size", "vision_config")?,
        patch_size: usize_field(object, "patch_size", "vision_config")?,
        rope_theta: f32_field(object, "rope_theta", "vision_config")?,
        downsample_ratio: usize_field(object, "downsample_ratio", "vision_config")?,
        max_image_tokens: usize_field(object, "max_image_tokens", "vision_config")?,
        min_pixels: usize_field(object, "min_pixels", "vision_config")?,
        max_wh_ratio,
    })
}

fn validate_vision_config(config: &DeepSeekV41VisionConfig) -> Result<()> {
    require(
        config.model_type == "deepseek_v41_vision",
        "vision_config.model_type must be 'deepseek_v41_vision'",
    )?;
    for (field, value, expected) in [
        ("num_hidden_layers", config.num_hidden_layers, 32),
        ("hidden_size", config.hidden_size, 1024),
        ("num_attention_heads", config.num_attention_heads, 16),
        ("intermediate_size", config.intermediate_size, 2816),
        ("patch_size", config.patch_size, 14),
        ("downsample_ratio", config.downsample_ratio, 3),
        ("max_image_tokens", config.max_image_tokens, 1024),
        ("min_pixels", config.min_pixels, 295936),
    ] {
        exact_usize(value, expected, &format!("vision_config.{field}"))?;
    }
    require(
        (config.rope_theta - 10_000.0).abs() < f32::EPSILON,
        "vision_config.rope_theta must be 10000",
    )?;
    if let Some(value) = config.max_wh_ratio {
        require(
            value.is_finite() && value > 0.0 && value <= MAX_VISION_WH_RATIO,
            format!(
                "vision_config.max_wh_ratio must be in (0, {MAX_VISION_WH_RATIO}], got {value}"
            ),
        )?;
    }
    Ok(())
}

/// Summary of the index-level contract, before optional safetensors headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV41IndexSummary {
    pub tensor_count: usize,
    pub shard_count: usize,
    pub total_size: u64,
    pub scale_pair_count: usize,
    pub headers_validated: bool,
}

/// Header-only, typed DeepSeek-V4.1 artifact metadata.
#[derive(Debug, Clone)]
pub struct DeepSeekV41Metadata {
    path: PathBuf,
    config: DeepSeekV41Config,
    index: HfSafetensorsIndex,
    summary: DeepSeekV41IndexSummary,
    inventory: Option<HfSafetensorsInventory>,
    descriptor: ModelDescriptor,
}

impl DeepSeekV41Metadata {
    pub fn open_hf(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_hf_with_validator(model_dir.as_ref(), validate_index)
    }

    pub fn open_hf_real(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_hf_with_validator(model_dir.as_ref(), validate_real_index)
    }

    fn open_hf_with_validator(
        model_dir: &Path,
        validator: fn(&DeepSeekV41Config, &HfSafetensorsIndex) -> Result<DeepSeekV41IndexSummary>,
    ) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).map_err(|error| {
                model_error(format!(
                    "config '{}' could not be read: {error}",
                    config_path.display()
                ))
            })?)
            .map_err(|error| model_error(format!("config '{}': {error}", config_path.display())))?;
        let index_path = model_dir.join("model.safetensors.index.json");
        let index = HfSafetensorsIndex::open(&index_path)
            .map_err(|error| model_error(format!("index '{}': {error}", index_path.display())))?;
        Self::from_value_with_validator(model_dir, &config_value, index, validator)
    }

    pub fn from_value(
        model_dir: &Path,
        value: &serde_json::Value,
        index: HfSafetensorsIndex,
    ) -> Result<Self> {
        Self::from_value_with_validator(model_dir, value, index, validate_index)
    }

    pub fn from_value_real(
        model_dir: &Path,
        value: &serde_json::Value,
        index: HfSafetensorsIndex,
    ) -> Result<Self> {
        Self::from_value_with_validator(model_dir, value, index, validate_real_index)
    }

    fn from_value_with_validator(
        model_dir: &Path,
        value: &serde_json::Value,
        index: HfSafetensorsIndex,
        validator: fn(&DeepSeekV41Config, &HfSafetensorsIndex) -> Result<DeepSeekV41IndexSummary>,
    ) -> Result<Self> {
        let config = DeepSeekV41Config::from_value(value)?;
        Self::from_config_and_index_with_validator(model_dir, config, index, validator)
    }

    pub fn from_config_and_index(
        model_dir: &Path,
        config: DeepSeekV41Config,
        index: HfSafetensorsIndex,
    ) -> Result<Self> {
        Self::from_config_and_index_with_validator(model_dir, config, index, validate_index)
    }

    pub fn from_config_and_index_real(
        model_dir: &Path,
        config: DeepSeekV41Config,
        index: HfSafetensorsIndex,
    ) -> Result<Self> {
        Self::from_config_and_index_with_validator(model_dir, config, index, validate_real_index)
    }

    fn from_config_and_index_with_validator(
        model_dir: &Path,
        config: DeepSeekV41Config,
        index: HfSafetensorsIndex,
        validator: fn(&DeepSeekV41Config, &HfSafetensorsIndex) -> Result<DeepSeekV41IndexSummary>,
    ) -> Result<Self> {
        config.validate()?;
        let mut summary = validator(&config, &index)?;
        let inventory = if index.missing_shards(model_dir).is_empty() {
            let inventory =
                HfSafetensorsInventory::from_index(model_dir, ModelFamily::DeepSeekV41, &index)?;
            summary.headers_validated =
                validate_header_inventory_inner(&config, &index, &inventory, model_dir)?;
            Some(inventory)
        } else {
            None
        };
        let descriptor = descriptor_from_metadata(
            model_dir,
            &config,
            &index,
            inventory.as_ref(),
            summary.headers_validated,
        );
        Ok(Self {
            path: model_dir.to_path_buf(),
            config,
            index,
            summary,
            inventory,
            descriptor,
        })
    }

    pub const fn config(&self) -> &DeepSeekV41Config {
        &self.config
    }

    pub const fn index(&self) -> &HfSafetensorsIndex {
        &self.index
    }

    pub const fn summary(&self) -> &DeepSeekV41IndexSummary {
        &self.summary
    }

    pub const fn inventory(&self) -> Option<&HfSafetensorsInventory> {
        self.inventory.as_ref()
    }

    pub const fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Validate the name-level V4.1 schema from an HF index without opening shards.
pub fn validate_index(
    config: &DeepSeekV41Config,
    index: &HfSafetensorsIndex,
) -> Result<DeepSeekV41IndexSummary> {
    config.validate()?;
    require(
        !index.weight_map.is_empty(),
        "safetensors index weight_map must not be empty",
    )?;
    let shards = validate_shard_identity(index)?;
    let names = index
        .weight_map
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for name in &names {
        require(
            known_tensor_name(name, config),
            format!("unknown or unsupported V4.1 tensor name '{name}'"),
        )?;
    }
    require_core_roles(config, &names)?;
    let mut scale_pair_count = 0;
    for name in &names {
        if let Some(base) = name.strip_suffix(".scale") {
            let weight = format!("{base}.weight");
            require(
                names.contains(weight.as_str()),
                format!("scale '{name}' has no paired weight '{weight}'"),
            )?;
            scale_pair_count += 1;
        }
        if requires_scale(name) {
            let scale = format!("{}.scale", name.trim_end_matches(".weight"));
            require(
                names.contains(scale.as_str()),
                format!("quantized weight '{name}' is missing scale '{scale}'"),
            )?;
        }
    }
    Ok(DeepSeekV41IndexSummary {
        tensor_count: names.len(),
        shard_count: shards.len(),
        total_size: index
            .total_size
            .ok_or_else(|| model_error("safetensors index metadata.total_size is required"))?,
        scale_pair_count,
        headers_validated: false,
    })
}

pub fn validate_real_index(
    config: &DeepSeekV41Config,
    index: &HfSafetensorsIndex,
) -> Result<DeepSeekV41IndexSummary> {
    let summary = validate_index(config, index)?;
    require(
        summary.shard_count == DEEPSEEK_V41_REAL_SHARD_COUNT,
        format!(
            "DeepSeek-V4.1 real contract requires {DEEPSEEK_V41_REAL_SHARD_COUNT} shards, got {}",
            summary.shard_count
        ),
    )?;
    require(
        summary.total_size == DEEPSEEK_V41_REAL_TOTAL_SIZE,
        format!(
            "DeepSeek-V4.1 real contract requires total_size={DEEPSEEK_V41_REAL_TOTAL_SIZE}, got {}",
            summary.total_size
        ),
    )?;
    Ok(summary)
}

fn validate_shard_identity(index: &HfSafetensorsIndex) -> Result<BTreeSet<String>> {
    let mut shards = BTreeSet::new();
    let mut expected_total = None;
    let mut ordinals = Vec::new();
    for shard in index.weight_map.values() {
        let components = Path::new(shard).components().collect::<Vec<_>>();
        require(
            components.len() == 1
                && matches!(components[0], Component::Normal(_))
                && shard.ends_with(".safetensors")
                && !shard.contains('\\'),
            format!("invalid safetensors shard filename '{shard}'"),
        )?;
        let stem = shard
            .strip_suffix(".safetensors")
            .ok_or_else(|| model_error(format!("invalid safetensors shard filename '{shard}'")))?;
        let fields = stem.split('-').collect::<Vec<_>>();
        require(
            fields.len() == 4 && fields[0] == "model" && fields[2] == "of",
            format!("shard '{shard}' must use model-N-of-M identity"),
        )?;
        let ordinal = fields[1]
            .parse::<u64>()
            .map_err(|_| model_error(format!("shard '{shard}' has invalid ordinal")))?;
        let total = fields[3]
            .parse::<u64>()
            .map_err(|_| model_error(format!("shard '{shard}' has invalid total")))?;
        let ordinal = usize::try_from(ordinal)
            .map_err(|_| model_error(format!("shard '{shard}' ordinal overflows usize")))?;
        let total = usize::try_from(total)
            .map_err(|_| model_error(format!("shard '{shard}' total overflows usize")))?;
        require(
            total > 0 && total <= MAX_SHARDS,
            format!("shard '{shard}' total {total} exceeds MAX_SHARDS={MAX_SHARDS}"),
        )?;
        require(
            ordinal > 0 && ordinal <= total,
            format!("shard '{shard}' has out-of-range ordinal {ordinal}/{total}"),
        )?;
        if let Some(previous) = expected_total {
            require(
                previous == total,
                format!("shard '{shard}' disagrees with shard total {previous}"),
            )?;
        } else {
            expected_total = Some(total);
        }
        if shards.insert(shard.clone()) {
            ordinals.push(ordinal);
        }
    }
    let total = expected_total.ok_or_else(|| model_error("safetensors index has no shards"))?;
    require(
        ordinals.len() == total,
        format!(
            "safetensors shard identity declares {total} shards but weight_map contains {} shard entries",
            ordinals.len()
        ),
    )?;
    ordinals.sort_unstable();
    for (position, ordinal) in ordinals.iter().enumerate() {
        let expected = position
            .checked_add(1)
            .ok_or_else(|| model_error("shard ordinal continuity overflows usize"))?;
        require(
            *ordinal == expected,
            format!(
                "safetensors shard identity is not contiguous at position {position}: expected {expected}, got {ordinal}"
            ),
        )?;
    }
    let total_size = index
        .total_size
        .ok_or_else(|| model_error("safetensors index metadata.total_size is required"))?;
    require(
        total_size > 0,
        "safetensors index metadata.total_size must be positive",
    )?;
    Ok(shards)
}

fn require_name(names: &BTreeSet<&str>, name: String) -> Result<()> {
    require(
        names.contains(name.as_str()),
        format!("missing required tensor '{name}'"),
    )
}

fn require_pair(names: &BTreeSet<&str>, base: String) -> Result<()> {
    require_name(names, format!("{base}.weight"))?;
    require_name(names, format!("{base}.scale"))
}

fn require_core_roles(config: &DeepSeekV41Config, names: &BTreeSet<&str>) -> Result<()> {
    for root in ["embed.weight", "head.weight", "norm.weight"] {
        require_name(names, root.to_owned())?;
    }
    for name in [
        "aligner.w1.weight",
        "aligner.w1.bias",
        "aligner.w2.weight",
        "aligner.w2.bias",
        "image_start",
        "image_end",
        "image_newline",
    ] {
        require_name(names, name.to_owned())?;
    }
    for layer in 0..config.text.num_hidden_layers {
        require_block_roles(
            names,
            &format!("layers.{layer}"),
            config.text.n_routed_experts,
        )?;
        let indexer_prefix = format!("layers.{layer}.attn.indexer.");
        let indexer_roles = match layer {
            2 | 8 | 14 | 20 => &[
                "k_norm.weight",
                "weights_proj.weight",
                "wk.weight",
                "wq_b.weight",
                "wq_b.scale",
            ][..],
            24 | 28 | 32 | 36 => &["weights_proj.weight", "wq_b.weight", "wq_b.scale"][..],
            _ => &[][..],
        };
        require_exact_role_set(names, &indexer_prefix, indexer_roles)?;
        if !indexer_roles.is_empty() {
            require_pair(names, format!("{indexer_prefix}wq_b"))?;
        }

        let compressor_prefix = format!("layers.{layer}.attn.compressor.");
        let compressor_roles = match layer {
            2 | 8 | 14 => &["norm.weight", "wgate.weight", "wkv.weight"][..],
            20 => &["norm.weight", "wkv.weight"][..],
            _ => &[][..],
        };
        require_exact_role_set(names, &compressor_prefix, compressor_roles)?;
        if config
            .text
            .engram
            .as_ref()
            .is_some_and(|engram| engram.layer_ids.contains(&layer))
        {
            for suffix in [
                "engram.embed.weight",
                "engram.embed.scale",
                "engram.k_weight",
                "engram.q_weight",
                "engram.wkv.weight",
                "engram.wkv.scale",
            ] {
                require_name(names, format!("layers.{layer}.{suffix}"))?;
            }
        }
    }
    for stage in 0..config.text.num_nextn_predict_layers {
        let prefix = format!("mtp.{stage}");
        require_block_roles(names, &prefix, config.text.dspark_n_routed_experts)?;
        if stage == 0 {
            require_pair(names, format!("{prefix}.main_proj"))?;
            require_name(names, format!("{prefix}.main_norm.weight"))?;
        }
        if stage + 1 == config.text.num_nextn_predict_layers {
            require_name(names, format!("{prefix}.confidence_head.proj.weight"))?;
            require_name(names, format!("{prefix}.markov_head.embed.weight"))?;
            require_name(names, format!("{prefix}.markov_head.head.weight"))?;
            require_name(names, format!("{prefix}.norm.weight"))?;
        }
    }
    if let Some(vision) = &config.vision {
        require_name(names, "vision.norm.weight".to_owned())?;
        require_name(names, "vision.patch_embed.proj.weight".to_owned())?;
        require_name(names, "vision.patch_embed.proj.bias".to_owned())?;
        for block in 0..vision.num_hidden_layers {
            for suffix in [
                "attn.wo.weight",
                "attn.wo.bias",
                "attn.wqkv.weight",
                "attn.wqkv.bias",
                "mlp.w1.weight",
                "mlp.w2.weight",
                "norm1.weight",
                "norm2.weight",
            ] {
                require_name(names, format!("vision.blocks.{block}.{suffix}"))?;
            }
        }
    }
    Ok(())
}

fn require_block_roles(names: &BTreeSet<&str>, prefix: &str, expert_count: usize) -> Result<()> {
    for suffix in [
        "attn.attn_sink",
        "attn.kv_norm.weight",
        "attn.q_norm.weight",
        "attn_norm.weight",
        "ffn_norm.weight",
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "ffn.gate.weight",
        "ffn.gate.bias",
        "ffn.gate.bias_vl",
    ] {
        require_name(names, format!("{prefix}.{suffix}"))?;
    }
    for matrix in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
        require_pair(names, format!("{prefix}.attn.{matrix}"))?;
    }
    for matrix in ["w1", "w2", "w3"] {
        require_pair(names, format!("{prefix}.ffn.shared_experts.{matrix}"))?;
    }
    for expert in 0..expert_count {
        for matrix in ["w1", "w2", "w3"] {
            require_pair(names, format!("{prefix}.ffn.experts.{expert}.{matrix}"))?;
        }
    }
    Ok(())
}

fn require_exact_role_set(names: &BTreeSet<&str>, prefix: &str, expected: &[&str]) -> Result<()> {
    let actual = names
        .iter()
        .filter_map(|name| name.strip_prefix(prefix))
        .collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    require(
        actual == expected,
        format!("tensor role set for '{prefix}' differs: expected {expected:?}, got {actual:?}"),
    )
}

fn known_tensor_name(name: &str, config: &DeepSeekV41Config) -> bool {
    match name {
        "embed.weight" | "head.weight" | "norm.weight" | "aligner.w1.weight"
        | "aligner.w1.bias" | "aligner.w2.weight" | "aligner.w2.bias" | "image_start"
        | "image_end" | "image_newline" => return true,
        _ => {}
    }
    let parts = name.split('.').collect::<Vec<_>>();
    match parts.as_slice() {
        ["layers", layer, tail @ ..] => {
            let Some(layer) = parse_index(layer) else {
                return false;
            };
            layer < config.text.num_hidden_layers
                && known_block_tail(
                    tail,
                    layer,
                    config.text.n_routed_experts,
                    config
                        .text
                        .engram
                        .as_ref()
                        .map(|engram| engram.layer_ids.as_slice()),
                )
        }
        ["mtp", stage, tail @ ..] => {
            let Some(stage) = parse_index(stage) else {
                return false;
            };
            stage < config.text.num_nextn_predict_layers
                && known_mtp_tail(tail, stage, config.text.dspark_n_routed_experts)
        }
        ["vision", "norm", "weight"]
        | ["vision", "patch_embed", "proj", "weight"]
        | ["vision", "patch_embed", "proj", "bias"] => config.vision.is_some(),
        ["vision", "blocks", block, tail @ ..] => {
            let Some(block) = parse_index(block) else {
                return false;
            };
            config
                .vision
                .as_ref()
                .is_some_and(|vision| block < vision.num_hidden_layers && known_vision_tail(tail))
        }
        _ => false,
    }
}

fn parse_index(value: &str) -> Option<usize> {
    (!value.is_empty() && (value == "0" || !value.starts_with('0')))
        .then(|| value.parse().ok())
        .flatten()
}

fn known_block_tail(
    tail: &[&str],
    layer: usize,
    expert_count: usize,
    engram_layers: Option<&[usize]>,
) -> bool {
    match tail {
        ["attn", rest @ ..] => known_attention_tail(rest),
        ["attn_norm", "weight"] | ["ffn_norm", "weight"] => true,
        [
            "hc_attn_base" | "hc_attn_fn" | "hc_attn_scale" | "hc_ffn_base" | "hc_ffn_fn"
            | "hc_ffn_scale",
        ] => true,
        ["ffn", rest @ ..] => known_ffn_tail(rest, expert_count),
        ["engram", rest @ ..] if engram_layers.is_some_and(|layers| layers.contains(&layer)) => {
            known_engram_tail(rest)
        }
        _ => false,
    }
}

fn known_mtp_tail(tail: &[&str], stage: usize, expert_count: usize) -> bool {
    match tail {
        ["main_norm", "weight"] if stage == 0 => true,
        ["main_proj", "weight" | "scale"] if stage == 0 => true,
        ["confidence_head", "proj", "weight"] if stage == 2 => true,
        ["markov_head", "embed", "weight"] | ["markov_head", "head", "weight"] if stage == 2 => {
            true
        }
        ["norm", "weight"] if stage == 2 => true,
        _ => known_block_tail(tail, usize::MAX, expert_count, None),
    }
}

fn known_attention_tail(tail: &[&str]) -> bool {
    matches!(
        tail,
        ["attn_sink"]
            | ["q_norm", "weight"]
            | ["kv_norm", "weight"]
            | ["wq_a", "weight" | "scale"]
            | ["wq_b", "weight" | "scale"]
            | ["wkv", "weight" | "scale"]
            | ["wo_a", "weight" | "scale"]
            | ["wo_b", "weight" | "scale"]
            | ["compressor", "norm", "weight"]
            | ["compressor", "wgate", "weight"]
            | ["compressor", "wkv", "weight"]
            | ["indexer", "k_norm", "weight"]
            | ["indexer", "weights_proj", "weight"]
            | ["indexer", "wk", "weight"]
            | ["indexer", "wq_b", "weight" | "scale"]
    )
}

fn known_ffn_tail(tail: &[&str], expert_count: usize) -> bool {
    match tail {
        ["gate", "weight" | "bias" | "bias_vl"] => true,
        ["shared_experts", matrix, "weight" | "scale"] if is_expert_matrix(matrix) => true,
        ["experts", expert, matrix, "weight" | "scale"]
            if parse_index(expert).is_some_and(|expert| expert < expert_count)
                && is_expert_matrix(matrix) =>
        {
            true
        }
        _ => false,
    }
}

fn is_expert_matrix(value: &str) -> bool {
    matches!(value, "w1" | "w2" | "w3")
}

fn known_engram_tail(tail: &[&str]) -> bool {
    matches!(
        tail,
        ["embed", "weight" | "scale"] | ["k_weight"] | ["q_weight"] | ["wkv", "weight" | "scale"]
    )
}

fn known_vision_tail(tail: &[&str]) -> bool {
    matches!(
        tail,
        ["attn", "wo", "weight" | "bias"]
            | ["attn", "wqkv", "weight" | "bias"]
            | ["mlp", "w1" | "w2", "weight"]
            | ["norm1" | "norm2", "weight"]
    )
}

fn requires_scale(name: &str) -> bool {
    let Some(base) = name.strip_suffix(".weight") else {
        return false;
    };
    base.ends_with(".attn.wq_a")
        || base.ends_with(".attn.wq_b")
        || base.ends_with(".attn.wkv")
        || base.ends_with(".attn.wo_a")
        || base.ends_with(".attn.wo_b")
        || base.ends_with(".indexer.wq_b")
        || base.contains(".ffn.experts.")
        || base.contains(".ffn.shared_experts.")
        || base.ends_with(".engram.embed")
        || base.ends_with(".engram.wkv")
        || base.ends_with(".main_proj")
}

fn is_fp4_expert_weight(name: &str) -> bool {
    match name.split('.').collect::<Vec<_>>().as_slice() {
        ["layers", layer, "ffn", "experts", expert, matrix, "weight"]
        | ["mtp", layer, "ffn", "experts", expert, matrix, "weight"] => {
            parse_index(layer).is_some()
                && parse_index(expert).is_some()
                && is_expert_matrix(matrix)
        }
        _ => false,
    }
}

fn engram_embed_contract(config: &DeepSeekV41Config, name: &str) -> Option<(usize, usize, usize)> {
    let parts = name.split('.').collect::<Vec<_>>();
    let ["layers", layer, "engram", "embed", "weight"] = parts.as_slice() else {
        return None;
    };
    let layer = parse_index(layer)?;
    let engram = config.text.engram.as_ref()?;
    let table = engram.layer_ids.iter().position(|id| *id == layer)?;
    Some((
        engram.num_embeddings[table],
        engram.head_dim,
        engram.num_heads,
    ))
}

fn checked_ceil_div(value: usize, divisor: usize, context: &str) -> Result<usize> {
    require(
        value > 0 && divisor > 0,
        format!("{context} dimensions must be positive"),
    )?;
    value
        .checked_add(divisor - 1)
        .map(|value| value / divisor)
        .ok_or_else(|| model_error(format!("{context} scale shape overflows")))
}

fn validate_header_scale_pair(
    config: &DeepSeekV41Config,
    weight: &HfSafetensorsTensorInfo,
    scale: &HfSafetensorsTensorInfo,
) -> Result<()> {
    let weight_dtype = DeepSeekV41DType::parse(&weight.dtype)?;
    let scale_dtype = DeepSeekV41DType::parse(&scale.dtype)?;
    require(
        scale_dtype == DeepSeekV41DType::F8E8M0,
        format!(
            "scale tensor '{}' must use F8_E8M0, got {}",
            scale.name, scale.dtype
        ),
    )?;
    let [rows, columns]: [usize; 2] = weight.shape.as_slice().try_into().map_err(|_| {
        model_error(format!(
            "scaled weight '{}' must have a 2D shape, got {:?}",
            weight.name, weight.shape
        ))
    })?;
    require(
        rows > 0 && columns > 0,
        format!(
            "scaled weight '{}' dimensions must be positive",
            weight.name
        ),
    )?;
    let block_size = config.quantization.weight_block_size;
    let expected_shape = if let Some((rows, columns, scale_columns)) =
        engram_embed_contract(config, &weight.name)
    {
        require(
            weight_dtype == DeepSeekV41DType::F8E4M3,
            format!(
                "Engram embed tensor '{}' must use F8_E4M3, got {}",
                weight.name, weight.dtype
            ),
        )?;
        require(
            weight.shape == vec![rows, columns],
            format!(
                "Engram embed tensor '{}' shape {:?} does not match [{rows}, {columns}]",
                weight.name, weight.shape
            ),
        )?;
        vec![rows, scale_columns]
    } else if is_fp4_expert_weight(&weight.name) {
        require(
            weight_dtype == DeepSeekV41DType::I8,
            format!(
                "FP4 expert tensor '{}' must be stored as I8, got {}",
                weight.name, weight.dtype
            ),
        )?;
        let logical_columns = columns.checked_mul(2).ok_or_else(|| {
            model_error(format!(
                "FP4 expert tensor '{}' packed input dimension overflows",
                weight.name
            ))
        })?;
        let block_columns = block_size[1];
        require(
            logical_columns.is_multiple_of(block_columns),
            format!(
                "FP4 expert tensor '{}' logical input width {} is not divisible by block width {}",
                weight.name, logical_columns, block_columns
            ),
        )?;
        vec![rows, logical_columns / block_columns]
    } else {
        require(
            weight_dtype == DeepSeekV41DType::F8E4M3,
            format!(
                "scaled base weight '{}' must use F8_E4M3, got {}",
                weight.name, weight.dtype
            ),
        )?;
        vec![
            checked_ceil_div(rows, block_size[0], &weight.name)?,
            checked_ceil_div(columns, block_size[1], &weight.name)?,
        ]
    };
    require(
        scale.shape == expected_shape,
        format!(
            "scale tensor '{}' shape {:?} does not match weight '{}' shape {:?}; expected {:?}",
            scale.name, scale.shape, weight.name, weight.shape, expected_shape
        ),
    )
}

fn validate_header_tensor_role(
    config: &DeepSeekV41Config,
    tensor: &HfSafetensorsTensorInfo,
    tensors: &BTreeMap<&str, &HfSafetensorsTensorInfo>,
) -> Result<bool> {
    let dtype = DeepSeekV41DType::parse(&tensor.dtype)?;
    if tensor.name.ends_with(".scale") {
        let base = tensor
            .name
            .strip_suffix(".scale")
            .ok_or_else(|| model_error(format!("invalid scale tensor name '{}'", tensor.name)))?;
        let weight_name = format!("{base}.weight");
        let weight = tensors.get(weight_name.as_str()).ok_or_else(|| {
            model_error(format!(
                "scale '{}' has no paired header weight '{}', despite index validation",
                tensor.name, weight_name
            ))
        })?;
        validate_header_scale_pair(config, weight, tensor)?;
        return Ok(true);
    }
    if requires_scale(&tensor.name) {
        let scale_name = format!("{}.scale", tensor.name.trim_end_matches(".weight"));
        let scale = tensors.get(scale_name.as_str()).ok_or_else(|| {
            model_error(format!(
                "quantized weight '{}' has no paired header scale '{}', despite index validation",
                tensor.name, scale_name
            ))
        })?;
        validate_header_scale_pair(config, tensor, scale)?;
        return Ok(true);
    }
    let covered = matches!(
        tensor.name.as_str(),
        "embed.weight" | "head.weight" | "norm.weight"
    ) || tensor.name.ends_with(".attn_norm.weight")
        || tensor.name.ends_with(".ffn_norm.weight");
    if covered {
        require(
            !matches!(dtype, DeepSeekV41DType::F8E8M0),
            format!(
                "unpaired scale dtype cannot be used for known tensor '{}'",
                tensor.name
            ),
        )?;
    }
    Ok(covered)
}

/// Validate safetensors header metadata without reading tensor payload bytes.
///
/// This entry point is intentionally metadata-only; callers provide an inventory
/// produced by the shared header reader and no runtime object is constructed.
pub fn validate_header_inventory(
    config: &DeepSeekV41Config,
    index: &HfSafetensorsIndex,
    inventory: &HfSafetensorsInventory,
    model_dir: &Path,
) -> Result<bool> {
    config.validate()?;
    validate_header_inventory_inner(config, index, inventory, model_dir)
}

fn validate_header_inventory_inner(
    config: &DeepSeekV41Config,
    index: &HfSafetensorsIndex,
    inventory: &HfSafetensorsInventory,
    model_dir: &Path,
) -> Result<bool> {
    require(
        !index.weight_map.is_empty() && !inventory.tensors.is_empty(),
        "safetensors header validation requires a non-empty index and header",
    )?;
    require(
        inventory.index_only_tensors.is_empty() && inventory.header_only_tensors.is_empty(),
        format!(
            "safetensors index/header identity differs: index_only={:?}, header_only={:?}",
            inventory.index_only_tensors, inventory.header_only_tensors
        ),
    )?;
    let mut actual_total = 0u64;
    let mut per_shard = BTreeMap::<&str, u64>::new();
    let mut data_extents = BTreeMap::<&str, Vec<(u64, u64, &str)>>::new();
    let mut data_prefixes = BTreeMap::<&str, u64>::new();
    let mut fully_validated = true;
    let tensors_by_name = inventory
        .tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    for tensor in &inventory.tensors {
        require(
            index.weight_map.get(&tensor.name) == Some(&tensor.shard),
            format!("index/header shard mismatch for '{}'", tensor.name),
        )?;
        let dtype = DeepSeekV41DType::parse(&tensor.dtype)?;
        if !known_tensor_name(&tensor.name, config) {
            fully_validated = false;
        }
        fully_validated &= validate_header_tensor_role(config, tensor, &tensors_by_name)?;
        let data_end = tensor
            .data_offset
            .checked_add(tensor.byte_size)
            .ok_or_else(|| {
                model_error(format!("tensor '{}' data extent overflows", tensor.name))
            })?;
        let data_prefix = tensor
            .file_offset
            .checked_sub(tensor.data_offset)
            .ok_or_else(|| {
                model_error(format!(
                    "tensor '{}' file offset precedes its data-section offset",
                    tensor.name
                ))
            })?;
        require(
            data_prefix >= 8 && data_prefix % 8 == 0,
            format!(
                "tensor '{}' has invalid safetensors data header prefix {data_prefix}",
                tensor.name
            ),
        )?;
        if let Some(previous) = data_prefixes.insert(tensor.shard.as_str(), data_prefix) {
            require(
                previous == data_prefix,
                format!("shard '{}' has inconsistent header prefixes", tensor.shard),
            )?;
        }
        data_extents
            .entry(tensor.shard.as_str())
            .or_default()
            .push((tensor.data_offset, data_end, tensor.name.as_str()));
        let elements = tensor.shape.iter().try_fold(1u64, |product, dimension| {
            let dimension = u64::try_from(*dimension).map_err(|_| {
                model_error(format!(
                    "tensor '{}' shape dimension overflows u64",
                    tensor.name
                ))
            })?;
            require(
                dimension > 0,
                format!(
                    "tensor '{}' has invalid shape {:?}",
                    tensor.name, tensor.shape
                ),
            )?;
            product.checked_mul(dimension).ok_or_else(|| {
                model_error(format!(
                    "tensor '{}' shape element count overflows",
                    tensor.name
                ))
            })
        })?;
        let expected_bytes = elements.checked_mul(dtype.element_bytes()).ok_or_else(|| {
            model_error(format!(
                "tensor '{}' shape byte size overflows",
                tensor.name
            ))
        })?;
        require(
            expected_bytes == tensor.byte_size,
            format!(
                "tensor '{}' shape {:?} dtype {} requires {} bytes, header declares {}",
                tensor.name,
                tensor.shape,
                dtype.as_str(),
                expected_bytes,
                tensor.byte_size
            ),
        )?;
        if matches!(tensor.name.as_str(), "embed.weight" | "head.weight") {
            require(
                tensor.shape == vec![config.text.vocab_size, config.text.hidden_size],
                format!(
                    "tensor '{}' has unexpected shape {:?}",
                    tensor.name, tensor.shape
                ),
            )?;
        }
        if tensor.name == "norm.weight"
            || tensor.name.ends_with(".attn_norm.weight")
            || tensor.name.ends_with(".ffn_norm.weight")
        {
            require(
                tensor.shape == vec![config.text.hidden_size],
                format!(
                    "tensor '{}' has unexpected shape {:?}",
                    tensor.name, tensor.shape
                ),
            )?;
        }
        actual_total = actual_total
            .checked_add(tensor.byte_size)
            .ok_or_else(|| model_error("header tensor byte total overflows"))?;
        let shard_total = per_shard.entry(tensor.shard.as_str()).or_default();
        *shard_total = shard_total
            .checked_add(tensor.byte_size)
            .ok_or_else(|| model_error("per-shard header byte total overflows"))?;
    }
    let expected_total = index
        .total_size
        .ok_or_else(|| model_error("safetensors index metadata.total_size is required"))?;
    require(
        actual_total == expected_total,
        format!("index total_size {expected_total} differs from header bytes {actual_total}"),
    )?;
    for (shard, extents) in &mut data_extents {
        extents.sort_unstable_by_key(|(start, _, _)| *start);
        for window in extents.windows(2) {
            let (_, previous_end, previous_name) = window[0];
            let (current_start, _, current_name) = window[1];
            require(
                current_start >= previous_end,
                format!(
                    "safetensors shard '{}' has overlapping tensor extents '{}' and '{}'",
                    shard, previous_name, current_name
                ),
            )?;
        }
    }
    for summary in &inventory.shard_summaries {
        require(
            per_shard.get(summary.shard.as_str()).copied().unwrap_or(0) == summary.bytes,
            format!("shard '{}' byte summary is inconsistent", summary.shard),
        )?;
        let file_length = std::fs::metadata(model_dir.join(&summary.shard))?.len();
        let max_end = inventory
            .tensors
            .iter()
            .filter(|tensor| tensor.shard == summary.shard)
            .map(|tensor| {
                tensor
                    .file_offset
                    .checked_add(tensor.byte_size)
                    .ok_or_else(|| {
                        model_error(format!("tensor '{}' file extent overflows", tensor.name))
                    })
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        require(
            max_end <= file_length,
            format!(
                "shard '{}' header extent exceeds file length",
                summary.shard
            ),
        )?;
    }
    Ok(fully_validated)
}

fn descriptor_from_metadata(
    model_dir: &Path,
    config: &DeepSeekV41Config,
    index: &HfSafetensorsIndex,
    inventory: Option<&HfSafetensorsInventory>,
    headers_validated: bool,
) -> ModelDescriptor {
    let text = &config.text;
    let quantization = inventory
        .map(|inventory| {
            inventory
                .dtype_counts
                .iter()
                .map(|item| QuantFormatCount {
                    format: item.dtype.clone(),
                    tensors: item.tensors,
                })
                .collect()
        })
        .unwrap_or_default();
    let mut notes = vec![
        "DeepSeek-V4.1 schema validated metadata-only; no runtime forward is registered".into(),
        format!(
            "HF safetensors index: {} tensors across {} shards, total_size={} bytes",
            index.tensor_count(),
            index.shard_count(),
            index.total_size.unwrap_or(0)
        ),
        "vision and Engram metadata are retained as optional artifact metadata; Engram tables are not active compute".into(),
    ];
    if inventory.is_none() {
        notes.push("safetensors shard payloads/headers are absent; dtype and shape validation is deferred until headers exist".into());
    } else if headers_validated {
        notes.push(
            "all covered safetensors shard headers validated without reading tensor payloads"
                .into(),
        );
    } else {
        notes.push("safetensors headers were inspected, but strict geometry coverage is incomplete; headers_validated remains false".into());
    }
    ModelDescriptor {
        path: model_dir.to_path_buf(),
        spec: TransformerSpec {
            family: ModelFamily::DeepSeekV41,
            architecture: Some(ARCHITECTURE.into()),
            weight_source: WeightSource::Safetensors,
            hidden_size: Some(text.hidden_size),
            num_layers: Some(text.num_hidden_layers),
            vocab_size: Some(text.vocab_size),
            num_heads: Some(text.num_attention_heads),
            num_kv_heads: Some(text.num_key_value_heads),
            head_dim: Some(text.head_dim),
            attention: AttentionKind::MultiLatentAttention,
            moe: MoeSpec {
                num_experts: Some(text.n_routed_experts),
                num_experts_per_tok: Some(text.num_experts_per_tok),
                has_shared_experts: text.n_shared_experts > 0,
                router: RouterKind::DenseTopK,
            },
            semantics: TransformerSemantics {
                norm_epsilon: Some(text.rms_norm_eps),
                hyper_connection_epsilon: Some(text.hc_eps),
                hyper_connection_sinkhorn_iters: Some(text.hc_sinkhorn_iters),
                rope_theta: Some(text.rope_theta),
                rope_head_dim: Some(text.qk_rope_head_dim),
                rope_factor: Some(text.rope_factor),
                rope_original_max_position_embeddings: Some(
                    text.rope_original_max_position_embeddings,
                ),
                rope_beta_fast: Some(text.rope_beta_fast),
                rope_beta_slow: Some(text.rope_beta_slow),
                compress_rope_theta: Some(text.compress_rope_theta),
                attention_window_size: Some(text.sliding_window),
                attention_index_topk: Some(text.index_topk),
                attention_index_num_heads: Some(text.index_n_heads),
                attention_index_head_dim: Some(text.index_head_dim),
                attention_compress_ratios: text.compress_ratios[..text.num_hidden_layers].to_vec(),
                output_projection_groups: Some(text.o_groups),
                output_projection_rank: Some(text.o_lora_rank),
                swiglu_limit: Some(text.swiglu_limit),
                route_scale: Some(text.routed_scaling_factor),
                num_hash_layers: None,
            },
            tensor_count: Some(index.tensor_count()),
            quantization,
            notes: std::mem::take(&mut notes),
        },
        tensor_classes: Vec::new(),
    }
}
