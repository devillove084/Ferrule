//! Validated metadata for the nested, dense Qwen3.5-0.8B BF16 artifact.

use ferrule_common::Result;
use serde::Deserialize;

use super::model_error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen35LayerType {
    LinearAttention,
    FullAttention,
}

/// Artifact profiles are deliberately narrower than architecture recognition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35Profile {
    Dense08Bbf16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35OutputGate {
    /// q_proj is [heads, 2 * head_dim, hidden], split within each head;
    /// sigmoid(gate) multiplies attention output before o_proj.
    PerHeadSigmoid,
}

/// Describes semantics, not an executable operator or a backend capability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen35TextSemantics {
    pub output_gate: Qwen35OutputGate,
    pub norm_weight_offset: f32,
    pub linear_gated_norm_weight_offset: f32,
    pub rotary_dimensions: usize,
    pub rotary_split_half: bool,
    pub rotary_prefix: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen35RopeConfig {
    pub rope_type: String,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],
}

/// Raw text fields are exposed read-only through a validated Qwen35Config.
/// Use Qwen35Config::layer_types for the resolved hybrid sequence.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen35TextConfig {
    pub model_type: String,
    #[serde(alias = "torch_dtype")]
    pub dtype: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<Qwen35LayerType>>,
    #[serde(default = "default_interval")]
    pub full_attention_interval: usize,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub attn_output_gate: bool,
    pub hidden_act: String,
    pub rms_norm_eps: f32,
    pub initializer_range: f32,
    pub max_position_embeddings: usize,
    pub rope_parameters: Qwen35RopeConfig,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub eos_token_id: u32,
    pub mamba_ssm_dtype: String,
    pub mlp_only_layers: Vec<usize>,
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
}

fn default_interval() -> usize {
    4
}

/// Vision metadata is validated solely to partition known attachment tensors.
/// It does not imply that vision forward is supported.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen35VisionConfig {
    pub model_type: String,
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub out_hidden_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub num_position_embeddings: usize,
    pub hidden_act: String,
    pub initializer_range: f32,
    pub deepstack_visual_indexes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    model_type: String,
    architectures: Vec<String>,
    text_config: Qwen35TextConfig,
    vision_config: Qwen35VisionConfig,
    tie_word_embeddings: bool,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    #[serde(default)]
    transformers_version: Option<String>,
}

/// Immutable validated profile. Construction never reads weights or selects a backend.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35Config {
    envelope: Envelope,
    layer_types: Vec<Qwen35LayerType>,
}

impl Qwen35Config {
    pub fn from_value(value: &serde_json::Value) -> Result<Self> {
        if value.get("model_type").and_then(|v| v.as_str()) != Some("qwen3_5") {
            return Err(super::Qwen35Unsupported::Profile(
                value
                    .get("model_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<missing>")
                    .into(),
            )
            .into());
        }
        if value.get("quantization_config").is_some()
            || value
                .get("text_config")
                .and_then(|v| v.get("quantization_config"))
                .is_some()
        {
            return Err(super::Qwen35Unsupported::Quantization.into());
        }
        let envelope: Envelope = serde_json::from_value(value.clone())
            .map_err(|error| model_error(format!("config: {error}")))?;
        require(
            envelope.architectures.as_slice() == ["Qwen3_5ForConditionalGeneration"],
            "architectures must be exactly [Qwen3_5ForConditionalGeneration]",
        )?;
        let t = &envelope.text_config;
        require(
            t.model_type == "qwen3_5_text",
            "text_config.model_type must be qwen3_5_text",
        )?;
        for (field, actual, expected) in [
            ("hidden_size", t.hidden_size, 1024),
            ("intermediate_size", t.intermediate_size, 3584),
            ("vocab_size", t.vocab_size, 248320),
            ("num_hidden_layers", t.num_hidden_layers, 24),
            ("num_attention_heads", t.num_attention_heads, 8),
            ("num_key_value_heads", t.num_key_value_heads, 2),
            ("head_dim", t.head_dim, 256),
            ("linear_conv_kernel_dim", t.linear_conv_kernel_dim, 4),
            ("linear_key_head_dim", t.linear_key_head_dim, 128),
            ("linear_value_head_dim", t.linear_value_head_dim, 128),
            ("linear_num_key_heads", t.linear_num_key_heads, 16),
            ("linear_num_value_heads", t.linear_num_value_heads, 16),
            ("full_attention_interval", t.full_attention_interval, 4),
            ("mtp_num_hidden_layers", t.mtp_num_hidden_layers, 1),
        ] {
            require(
                actual == expected,
                &format!(
                    "unsupported 0.8B profile: text_config.{field} must be {expected}, got {actual}"
                ),
            )?;
        }
        require(
            matches!(t.dtype.as_str(), "bfloat16" | "bf16"),
            "text dtype must be bfloat16",
        )?;
        require(
            t.mamba_ssm_dtype == "float32",
            "mamba_ssm_dtype must be float32",
        )?;
        require(
            !t.attention_bias && t.attention_dropout == 0.0,
            "requires bias-free attention and zero dropout",
        )?;
        require(
            t.attn_output_gate,
            "attn_output_gate must be true (per-head sigmoid)",
        )?;
        require(t.hidden_act == "silu", "hidden_act must be silu")?;
        require(
            t.tie_word_embeddings && envelope.tie_word_embeddings,
            "top-level and text tie_word_embeddings must both be true",
        )?;
        require(t.use_cache, "use_cache must be true")?;
        require(
            t.mlp_only_layers.is_empty(),
            "mlp_only_layers must be empty",
        )?;
        require(
            !t.mtp_use_dedicated_embeddings,
            "dedicated MTP embeddings are unsupported",
        )?;
        require(
            t.max_position_embeddings > 0 && t.max_position_embeddings <= 262144,
            "max_position_embeddings must be in 1..=262144",
        )?;
        positive("rms_norm_eps", t.rms_norm_eps)?;
        positive("initializer_range", t.initializer_range)?;
        for (name, id) in [
            ("text_config.eos_token_id", t.eos_token_id),
            ("image_token_id", envelope.image_token_id),
            ("video_token_id", envelope.video_token_id),
            ("vision_start_token_id", envelope.vision_start_token_id),
            ("vision_end_token_id", envelope.vision_end_token_id),
        ] {
            require(
                (id as usize) < t.vocab_size,
                &format!("{name} is outside vocab_size"),
            )?;
        }
        let r = &t.rope_parameters;
        require(
            r.rope_type == "default"
                && r.rope_theta == 10_000_000.0
                && r.partial_rotary_factor == 0.25
                && r.mrope_interleaved
                && r.mrope_section == [11, 11, 10],
            "unsupported rope_parameters for 0.8B (requires default partial split-half RoPE)",
        )?;
        let expected = (0..t.num_hidden_layers)
            .map(|i| {
                if (i + 1) % t.full_attention_interval == 0 {
                    Qwen35LayerType::FullAttention
                } else {
                    Qwen35LayerType::LinearAttention
                }
            })
            .collect::<Vec<_>>();
        let layer_types = t.layer_types.clone().unwrap_or_else(|| expected.clone());
        require(
            layer_types == expected,
            "layer_types must be [linear_attention, linear_attention, linear_attention, full_attention] repeated 6 times",
        )?;
        validate_vision(&envelope.vision_config)?;
        Ok(Self {
            envelope,
            layer_types,
        })
    }

    /// Test-only dimensions: never admitted by the public artifact parser.
    #[cfg(test)]
    pub(super) fn tiny_for_test() -> Self {
        let value =
            serde_json::from_str(include_str!("../../../tests/qwen35_08b_config.json")).unwrap();
        let mut config = Self::from_value(&value).unwrap();
        let t = &mut config.envelope.text_config;
        t.hidden_size = 8;
        t.intermediate_size = 12;
        t.vocab_size = 11;
        t.num_hidden_layers = 4;
        t.num_attention_heads = 2;
        t.num_key_value_heads = 1;
        t.head_dim = 8;
        t.linear_num_key_heads = 1;
        t.linear_num_value_heads = 2;
        t.linear_key_head_dim = 3;
        t.linear_value_head_dim = 2;
        t.linear_conv_kernel_dim = 3;
        t.max_position_embeddings = 32;
        t.eos_token_id = 10;
        t.rope_parameters.partial_rotary_factor = 0.5;
        t.rope_parameters.rope_theta = 10000.0;
        config.layer_types.truncate(4);
        t.layer_types = Some(config.layer_types.clone());
        config
    }

    pub const fn profile(&self) -> Qwen35Profile {
        Qwen35Profile::Dense08Bbf16
    }

    pub fn architecture(&self) -> &str {
        &self.envelope.architectures[0]
    }

    pub const fn text(&self) -> &Qwen35TextConfig {
        &self.envelope.text_config
    }

    pub const fn vision(&self) -> &Qwen35VisionConfig {
        &self.envelope.vision_config
    }

    pub fn layer_types(&self) -> &[Qwen35LayerType] {
        &self.layer_types
    }

    pub const fn semantics(&self) -> Qwen35TextSemantics {
        Qwen35TextSemantics {
            output_gate: Qwen35OutputGate::PerHeadSigmoid,
            norm_weight_offset: 1.0,
            linear_gated_norm_weight_offset: 0.0,
            rotary_dimensions: (self.envelope.text_config.head_dim as f32
                * self
                    .envelope
                    .text_config
                    .rope_parameters
                    .partial_rotary_factor) as usize,
            rotary_split_half: true,
            rotary_prefix: true,
        }
    }
}

fn validate_vision(v: &Qwen35VisionConfig) -> Result<()> {
    require(
        v.model_type == "qwen3_5",
        "vision_config.model_type must be qwen3_5",
    )?;
    for (field, actual, expected) in [
        ("depth", v.depth, 12),
        ("hidden_size", v.hidden_size, 768),
        ("intermediate_size", v.intermediate_size, 3072),
        ("out_hidden_size", v.out_hidden_size, 1024),
        ("num_heads", v.num_heads, 12),
        ("in_channels", v.in_channels, 3),
        ("patch_size", v.patch_size, 16),
        ("spatial_merge_size", v.spatial_merge_size, 2),
        ("temporal_patch_size", v.temporal_patch_size, 2),
        ("num_position_embeddings", v.num_position_embeddings, 2304),
    ] {
        require(
            actual == expected,
            &format!("unsupported 0.8B attachment: vision_config.{field} must be {expected}"),
        )?;
    }
    require(
        v.hidden_act == "gelu_pytorch_tanh" && v.deepstack_visual_indexes.is_empty(),
        "unsupported vision activation/deepstack attachments",
    )?;
    positive("vision initializer_range", v.initializer_range)
}

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(model_error(message))
    }
}

fn positive(field: &str, value: f32) -> Result<()> {
    require(
        value.is_finite() && value > 0.0,
        &format!("{field} must be finite and positive"),
    )
}
