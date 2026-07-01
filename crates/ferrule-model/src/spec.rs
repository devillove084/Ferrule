use std::fmt;

use crate::OlmoeConfig;

/// High-level model family understood by Ferrule's runtime boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelFamily {
    Olmoe,
    DeepSeekV4,
    DeepSeekV3,
    DeepSeekV2,
    QwenMoe,
    Mixtral,
    Llama,
    Unknown(String),
}

impl ModelFamily {
    pub fn from_architecture(name: &str) -> Self {
        let n = normalize_name(name);
        if n.contains("olmoe") {
            Self::Olmoe
        } else if n.contains("deepseek4") || n.contains("deepseekv4") {
            Self::DeepSeekV4
        } else if n.contains("deepseek3") || n.contains("deepseekv3") {
            Self::DeepSeekV3
        } else if n.contains("deepseek2") || n.contains("deepseekv2") {
            Self::DeepSeekV2
        } else if n.contains("qwen") && n.contains("moe") {
            Self::QwenMoe
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
            Self::Olmoe => "OLMoE",
            Self::DeepSeekV4 => "DeepSeek-V4",
            Self::DeepSeekV3 => "DeepSeek-V3",
            Self::DeepSeekV2 => "DeepSeek-V2",
            Self::QwenMoe => "Qwen-MoE",
            Self::Mixtral => "Mixtral",
            Self::Llama => "Llama",
            Self::Unknown(name) => name.as_str(),
        }
    }

    pub fn is_supported_runtime_family(&self) -> bool {
        matches!(self, Self::Olmoe)
    }
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Attention layout exposed at the model-family boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantFormatCount {
    pub format: String,
    pub tensors: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub tensor_count: Option<usize>,
    pub quantization: Vec<QuantFormatCount>,
    pub notes: Vec<String>,
}

impl TransformerSpec {
    pub fn from_olmoe_config(config: &OlmoeConfig, weight_source: WeightSource) -> Self {
        let attention = if config.num_kv_heads < config.num_heads {
            AttentionKind::GroupedQuery
        } else {
            AttentionKind::DenseMha
        };
        Self {
            family: ModelFamily::Olmoe,
            architecture: Some("olmoe".into()),
            weight_source,
            hidden_size: Some(config.hidden_size),
            num_layers: Some(config.num_layers),
            vocab_size: Some(config.vocab_size),
            num_heads: Some(config.num_heads),
            num_kv_heads: Some(config.num_kv_heads),
            head_dim: Some(config.head_dim),
            attention,
            moe: MoeSpec {
                num_experts: Some(config.num_experts),
                num_experts_per_tok: Some(config.num_experts_per_tok),
                has_shared_experts: false,
                router: RouterKind::DenseTopK,
            },
            tensor_count: None,
            quantization: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub fn supports_current_runtime(&self) -> bool {
        self.family.is_supported_runtime_family()
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
    fn olmoe_is_current_runtime_family() {
        assert!(ModelFamily::Olmoe.is_supported_runtime_family());
        assert!(!ModelFamily::DeepSeekV4.is_supported_runtime_family());
    }
}
