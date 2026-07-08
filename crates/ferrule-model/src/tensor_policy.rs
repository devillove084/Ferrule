use std::collections::BTreeMap;
use std::fmt;

use crate::gguf::TensorInfo;

use crate::families;
use crate::spec::ModelFamily;

/// Semantic tensor classes used to map checkpoint artifact names into Transformer
/// blocks and optional execution attachments.
///
/// Concrete artifact names live in `crate::families::*`. This enum should stay
/// semantic enough that multiple model families can share roles without forcing
/// DeepSeek/OLMoE/Llama naming into the generic runtime.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorClass {
    TokenEmbedding,
    OutputNorm,
    OutputHead,
    LayerNorm,
    AttentionNorm,
    FeedForwardNorm,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    AttentionSink,
    DenseMlpGate,
    DenseMlpUp,
    DenseMlpDown,
    MlaQueryA,
    MlaQueryB,
    MlaQueryNorm,
    MlaKv,
    MlaKvNorm,
    MlaOutputA,
    MlaOutputB,
    MlaCompressor,
    Router,
    RouterBias,
    RoutedExpertGate,
    RoutedExpertUp,
    RoutedExpertDown,
    SharedExpertGate,
    SharedExpertUp,
    SharedExpertDown,
    HashRouterTable,
    Indexer,
    HiddenCompressor,
    OutputHiddenCompressor,
    SpeculativeProjection,
    SpeculativeMarkovHead,
    SpeculativeConfidenceHead,
    Auxiliary,
    Unknown,
}

impl TensorClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TokenEmbedding => "token_embedding",
            Self::OutputNorm => "output_norm",
            Self::OutputHead => "output_head",
            Self::LayerNorm => "layer_norm",
            Self::AttentionNorm => "attention_norm",
            Self::FeedForwardNorm => "feed_forward_norm",
            Self::AttentionQuery => "attention_query",
            Self::AttentionKey => "attention_key",
            Self::AttentionValue => "attention_value",
            Self::AttentionOutput => "attention_output",
            Self::AttentionSink => "attention_sink",
            Self::DenseMlpGate => "dense_mlp_gate",
            Self::DenseMlpUp => "dense_mlp_up",
            Self::DenseMlpDown => "dense_mlp_down",
            Self::MlaQueryA => "mla_query_a",
            Self::MlaQueryB => "mla_query_b",
            Self::MlaQueryNorm => "mla_query_norm",
            Self::MlaKv => "mla_kv",
            Self::MlaKvNorm => "mla_kv_norm",
            Self::MlaOutputA => "mla_output_a",
            Self::MlaOutputB => "mla_output_b",
            Self::MlaCompressor => "mla_compressor",
            Self::Router => "router",
            Self::RouterBias => "router_bias",
            Self::RoutedExpertGate => "routed_expert_gate",
            Self::RoutedExpertUp => "routed_expert_up",
            Self::RoutedExpertDown => "routed_expert_down",
            Self::SharedExpertGate => "shared_expert_gate",
            Self::SharedExpertUp => "shared_expert_up",
            Self::SharedExpertDown => "shared_expert_down",
            Self::HashRouterTable => "hash_router_table",
            Self::Indexer => "indexer",
            Self::HiddenCompressor => "hidden_compressor",
            Self::OutputHiddenCompressor => "output_hidden_compressor",
            Self::SpeculativeProjection => "speculative_projection",
            Self::SpeculativeMarkovHead => "speculative_markov_head",
            Self::SpeculativeConfidenceHead => "speculative_confidence_head",
            Self::Auxiliary => "auxiliary",
            Self::Unknown => "unknown",
        }
    }

    pub fn is_mla_attention(&self) -> bool {
        matches!(
            self,
            Self::MlaQueryA
                | Self::MlaQueryB
                | Self::MlaQueryNorm
                | Self::MlaKv
                | Self::MlaKvNorm
                | Self::MlaOutputA
                | Self::MlaOutputB
                | Self::MlaCompressor
        )
    }

    pub fn is_attention_auxiliary(&self) -> bool {
        matches!(self, Self::AttentionSink)
    }

    pub fn is_routed_expert(&self) -> bool {
        matches!(
            self,
            Self::RoutedExpertGate | Self::RoutedExpertUp | Self::RoutedExpertDown
        )
    }

    pub fn is_auxiliary(&self) -> bool {
        matches!(
            self,
            Self::Indexer | Self::HiddenCompressor | Self::OutputHiddenCompressor | Self::Auxiliary
        )
    }

    pub fn is_speculative(&self) -> bool {
        matches!(
            self,
            Self::SpeculativeProjection
                | Self::SpeculativeMarkovHead
                | Self::SpeculativeConfidenceHead
        )
    }
}

impl fmt::Display for TensorClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorClassCount {
    pub class: TensorClass,
    pub tensors: usize,
}

/// Family-aware GGUF tensor classification policy.
#[derive(Debug, Clone)]
pub struct GgufTensorPolicy {
    family: ModelFamily,
}

impl GgufTensorPolicy {
    pub fn for_family(family: ModelFamily) -> Self {
        Self { family }
    }

    pub fn classify_tensor(&self, tensor: &TensorInfo) -> TensorClass {
        self.classify_name(&tensor.name)
    }

    pub fn classify_name(&self, name: &str) -> TensorClass {
        families::classify_gguf_tensor(&self.family, name)
    }

    pub fn summarize<'a>(
        &self,
        tensors: impl IntoIterator<Item = &'a TensorInfo>,
    ) -> Vec<TensorClassCount> {
        let mut counts = BTreeMap::<TensorClass, usize>::new();
        for tensor in tensors {
            *counts.entry(self.classify_tensor(tensor)).or_default() += 1;
        }
        counts
            .into_iter()
            .map(|(class, tensors)| TensorClassCount { class, tensors })
            .collect()
    }
}

/// Family-aware Hugging Face/safetensors tensor classification policy.
#[derive(Debug, Clone)]
pub struct HfTensorPolicy {
    family: ModelFamily,
}

impl HfTensorPolicy {
    pub fn for_family(family: ModelFamily) -> Self {
        Self { family }
    }

    pub fn classify_name(&self, name: &str) -> TensorClass {
        families::classify_hf_tensor(&self.family, name)
    }

    pub fn summarize<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> Vec<TensorClassCount> {
        let mut counts = BTreeMap::<TensorClass, usize>::new();
        for name in names {
            *counts.entry(self.classify_name(name)).or_default() += 1;
        }
        counts
            .into_iter()
            .map(|(class, tensors)| TensorClassCount { class, tensors })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_policy_classifies_dense_attention_names() {
        let policy = HfTensorPolicy::for_family(ModelFamily::Llama);
        assert_eq!(
            policy.classify_name("model.layers.0.self_attn.q_proj.weight"),
            TensorClass::AttentionQuery
        );
        assert_eq!(
            policy.classify_name("model.layers.0.self_attn.o_proj.weight"),
            TensorClass::AttentionOutput
        );
    }
}
