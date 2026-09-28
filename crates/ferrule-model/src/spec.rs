use std::fmt;

use serde::{Deserialize, Serialize};

/// High-level model family understood by Ferrule's runtime boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFamily {
    DeepSeekV4,
    DeepSeekV3,
    DeepSeekV2,
    Qwen3,
    /// Qwen3.5 dense text-only hybrid CPU/CUDA (strict 0.8B profile).
    Qwen35,
    /// Strict Qwen3.5 MoE 35B-A3B numeric FP8 text profile on CUDA only.
    Qwen35Moe,
    QwenMoe,
    Mixtral,
    Llama,
    Unknown(String),
}

impl ModelFamily {
    pub fn from_architecture(name: &str) -> Self {
        // Qwen architecture identities must not use substring matching: qwen3_5
        // and qwen3_next are not ordinary Qwen3 decoders.
        match name {
            "qwen3" | "Qwen3ForCausalLM" => return Self::Qwen3,
            "qwen3_5"
            | "qwen3_5_text"
            | "Qwen3_5ForConditionalGeneration"
            | "Qwen3_5ForCausalLM" => return Self::Qwen35,
            "qwen3_5_moe"
            | "qwen3_5_moe_text"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3_5MoeForCausalLM" => return Self::Qwen35Moe,
            "qwen2_moe" | "Qwen2MoeForCausalLM" | "qwen3_moe" | "Qwen3MoeForCausalLM" => {
                return Self::QwenMoe;
            }
            _ => {}
        }
        let n = normalize_name(name);
        if n.contains("deepseek4") || n.contains("deepseekv4") {
            Self::DeepSeekV4
        } else if n.contains("deepseek3") || n.contains("deepseekv3") {
            Self::DeepSeekV3
        } else if n.contains("deepseek2") || n.contains("deepseekv2") {
            Self::DeepSeekV2
        } else if n.contains("mixtral") {
            Self::Mixtral
        } else if n.contains("llama") {
            Self::Llama
        } else {
            Self::Unknown(name.to_string())
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::DeepSeekV4 => "DeepSeek-V4",
            Self::DeepSeekV3 => "DeepSeek-V3",
            Self::DeepSeekV2 => "DeepSeek-V2",
            Self::Qwen3 => "Qwen3",
            Self::Qwen35 => "Qwen3.5",
            Self::Qwen35Moe => "Qwen3.5-MoE",
            Self::QwenMoe => "Qwen-MoE",
            Self::Mixtral => "Mixtral",
            Self::Llama => "Llama",
            Self::Unknown(name) => name.as_str(),
        }
    }

    /// At least one runtime profile is implemented for this family. This is not
    /// admission for every variant, backend, precision, or parallel topology.
    /// Use a strict descriptor and a backend-specific EnginePlan for those checks.
    pub fn is_supported_runtime_family(&self) -> bool {
        matches!(
            self,
            Self::DeepSeekV4 | Self::QwenMoe | Self::Qwen3 | Self::Qwen35 | Self::Qwen35Moe
        )
    }
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Attention layout exposed at the model-family boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionKind {
    DenseMha,
    GroupedQuery,
    MultiLatentAttention,
    Unknown(String),
}

impl AttentionKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::DenseMha => "MHA",
            Self::GroupedQuery => "GQA",
            Self::MultiLatentAttention => "MLA",
            Self::Unknown(name) => name.as_str(),
        }
    }
}

impl fmt::Display for AttentionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where the model weights come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightSource {
    Safetensors,
    Gguf,
    Unknown,
}

impl WeightSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Safetensors => "safetensors",
            Self::Gguf => "gguf",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for WeightSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouterKind {
    DenseTopK,
    HashAssistedTopK,
    None,
    Unknown(String),
}

impl RouterKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::DenseTopK => "dense top-k",
            Self::HashAssistedTopK => "hash-assisted top-k",
            Self::None => "none",
            Self::Unknown(name) => name.as_str(),
        }
    }
}

impl fmt::Display for RouterKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeSpec {
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub has_shared_experts: bool,
    pub router: RouterKind,
}

impl MoeSpec {
    pub fn none() -> Self {
        Self {
            num_experts: None,
            num_experts_per_tok: None,
            has_shared_experts: false,
            router: RouterKind::None,
        }
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts.unwrap_or(0) > 0 || !matches!(self.router, RouterKind::None)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantFormatCount {
    pub format: String,
    pub tensors: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TransformerSemantics {
    pub norm_epsilon: Option<f32>,
    pub hyper_connection_epsilon: Option<f32>,
    pub hyper_connection_sinkhorn_iters: Option<usize>,
    pub rope_theta: Option<f32>,
    pub rope_head_dim: Option<usize>,
    pub rope_factor: Option<f32>,
    pub rope_original_max_position_embeddings: Option<usize>,
    pub rope_beta_fast: Option<usize>,
    pub rope_beta_slow: Option<usize>,
    pub compress_rope_theta: Option<f32>,
    pub attention_window_size: Option<usize>,
    pub attention_index_topk: Option<usize>,
    pub attention_index_num_heads: Option<usize>,
    pub attention_index_head_dim: Option<usize>,
    pub attention_compress_ratios: Vec<usize>,
    pub output_projection_groups: Option<usize>,
    pub output_projection_rank: Option<usize>,
    pub swiglu_limit: Option<f32>,
    pub route_scale: Option<f32>,
    pub num_hash_layers: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformerSpec {
    pub family: ModelFamily,
    pub architecture: Option<String>,
    pub weight_source: WeightSource,
    pub hidden_size: Option<usize>,
    pub num_layers: Option<usize>,
    pub vocab_size: Option<usize>,
    pub num_heads: Option<usize>,
    pub num_kv_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub attention: AttentionKind,
    pub moe: MoeSpec,
    #[serde(default)]
    pub semantics: TransformerSemantics,
    pub tensor_count: Option<usize>,
    pub quantization: Vec<QuantFormatCount>,
    pub notes: Vec<String>,
}

impl TransformerSpec {
    /// Whether this descriptor has an implementation in this build on at least
    /// one backend. This does not select a backend: the default EnginePlan is CPU.
    pub fn supports_current_runtime(&self) -> bool {
        if self.family == ModelFamily::Qwen35Moe {
            return cfg!(feature = "cuda") && self.is_qwen35_moe_35b_a3b_fp8();
        }
        self.family.is_supported_runtime_family()
    }

    /// Exact summary emitted by the strict Qwen35Metadata boundary. Do not infer
    /// this profile from family identity or the presence of an FP8 dtype alone.
    /// Text storage is mandatory; visual and MTP attachments are independently
    /// absent or complete. Counts include weight AND numeric BF16 scale parts.
    ///
    /// This is a planning check, not a substitute for config/name/shape/pair and
    /// shard-extent validation by Qwen35Metadata before loading an artifact.
    pub(crate) fn is_qwen35_moe_35b_a3b_fp8(&self) -> bool {
        if self.family != ModelFamily::Qwen35Moe
            || self.architecture.as_deref() != Some("Qwen3_5MoeForConditionalGeneration")
            || self.weight_source != WeightSource::Safetensors
            || self.hidden_size != Some(2048)
            || self.num_layers != Some(40)
            || self.vocab_size != Some(248320)
            || self.num_heads != Some(16)
            || self.num_kv_heads != Some(2)
            || self.head_dim != Some(256)
            || self.attention != AttentionKind::Unknown("qwen35_hybrid_linear_full".into())
            || self.moe
                != (MoeSpec {
                    num_experts: Some(256),
                    num_experts_per_tok: Some(8),
                    has_shared_experts: true,
                    router: RouterKind::DenseTopK,
                })
            || self.semantics
                != (TransformerSemantics {
                    norm_epsilon: Some(1e-6),
                    rope_theta: Some(10_000_000.0),
                    rope_head_dim: Some(64),
                    ..TransformerSemantics::default()
                })
        {
            return false;
        }
        // A duplicate, missing, unknown, native-E8M0 or packed-BF16 format cannot
        // masquerade as the verified per-expert FP8 + numeric BF16 schema.
        if self.quantization.len() != 3 {
            return false;
        }
        [false, true].into_iter().any(|visual| {
            [false, true].into_iter().any(|mtp| {
                let fp8 = 30_970 + usize::from(mtp) * 775;
                let bf16 = 31_273 + usize::from(visual) * 333 + usize::from(mtp) * 785;
                let expected = [("F8_E4M3", fp8), ("BF16", bf16), ("F32", 60)];
                self.tensor_count == Some(fp8 + bf16 + 60)
                    && expected.iter().all(|(format, tensors)| {
                        self.quantization
                            .iter()
                            .filter(|q| q.format == *format && q.tensors == *tensors)
                            .count()
                            == 1
                    })
            })
        })
    }
}

fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_deepseek_v4_arch_names() {
        assert_eq!(
            ModelFamily::from_architecture("deepseek4"),
            ModelFamily::DeepSeekV4
        );
        assert_eq!(
            ModelFamily::from_architecture("DeepSeek-V4-Flash"),
            ModelFamily::DeepSeekV4
        );
    }

    #[test]
    fn runtime_family_support_is_exact() {
        assert!(ModelFamily::DeepSeekV4.is_supported_runtime_family());
        assert!(ModelFamily::QwenMoe.is_supported_runtime_family());
        assert!(ModelFamily::Qwen3.is_supported_runtime_family());
        assert!(ModelFamily::Qwen35Moe.is_supported_runtime_family());
    }
}
