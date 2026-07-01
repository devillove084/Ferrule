use std::fmt;

/// Semantic tensor role consumed by the generic Transformer executor boundary.
///
/// Source tensor names such as `blk.0.attn_q_a.weight` should be translated into
/// these roles by model-family bindings. Generic runtime code should depend on
/// roles, not source names or one model family's naming scheme.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorRole {
    TokenEmbedding,
    OutputNorm,
    OutputHead,
    LayerNorm,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    AttentionSink,
    AttentionLatentQueryA,
    AttentionLatentQueryB,
    AttentionLatentKv,
    AttentionLatentOutputA,
    AttentionLatentOutputB,
    AttentionCompressor,
    DenseMlpGate,
    DenseMlpUp,
    DenseMlpDown,
    RouterLogits,
    RouterBias,
    HashRouterTable,
    RoutedExpertGate,
    RoutedExpertUp,
    RoutedExpertDown,
    SharedExpertGate,
    SharedExpertUp,
    SharedExpertDown,
    AuxIndexer,
    AuxHiddenCompressor,
    AuxOutputHiddenCompressor,
    SpeculativeProjection,
    SpeculativeMarkovHead,
    SpeculativeConfidenceHead,
    Auxiliary,
    Unknown,
}

impl TensorRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TokenEmbedding => "token_embedding",
            Self::OutputNorm => "output_norm",
            Self::OutputHead => "output_head",
            Self::LayerNorm => "layer_norm",
            Self::AttentionQuery => "attention_query",
            Self::AttentionKey => "attention_key",
            Self::AttentionValue => "attention_value",
            Self::AttentionOutput => "attention_output",
            Self::AttentionSink => "attention_sink",
            Self::AttentionLatentQueryA => "attention_latent_query_a",
            Self::AttentionLatentQueryB => "attention_latent_query_b",
            Self::AttentionLatentKv => "attention_latent_kv",
            Self::AttentionLatentOutputA => "attention_latent_output_a",
            Self::AttentionLatentOutputB => "attention_latent_output_b",
            Self::AttentionCompressor => "attention_compressor",
            Self::DenseMlpGate => "dense_mlp_gate",
            Self::DenseMlpUp => "dense_mlp_up",
            Self::DenseMlpDown => "dense_mlp_down",
            Self::RouterLogits => "router_logits",
            Self::RouterBias => "router_bias",
            Self::HashRouterTable => "hash_router_table",
            Self::RoutedExpertGate => "routed_expert_gate",
            Self::RoutedExpertUp => "routed_expert_up",
            Self::RoutedExpertDown => "routed_expert_down",
            Self::SharedExpertGate => "shared_expert_gate",
            Self::SharedExpertUp => "shared_expert_up",
            Self::SharedExpertDown => "shared_expert_down",
            Self::AuxIndexer => "aux_indexer",
            Self::AuxHiddenCompressor => "aux_hidden_compressor",
            Self::AuxOutputHiddenCompressor => "aux_output_hidden_compressor",
            Self::SpeculativeProjection => "speculative_projection",
            Self::SpeculativeMarkovHead => "speculative_markov_head",
            Self::SpeculativeConfidenceHead => "speculative_confidence_head",
            Self::Auxiliary => "auxiliary",
            Self::Unknown => "unknown",
        }
    }

    pub fn is_attention_latent(&self) -> bool {
        matches!(
            self,
            Self::AttentionLatentQueryA
                | Self::AttentionLatentQueryB
                | Self::AttentionLatentKv
                | Self::AttentionLatentOutputA
                | Self::AttentionLatentOutputB
                | Self::AttentionCompressor
        )
    }

    pub fn is_attention_auxiliary(&self) -> bool {
        matches!(self, Self::AttentionSink)
    }

    pub fn is_auxiliary(&self) -> bool {
        matches!(
            self,
            Self::AuxIndexer
                | Self::AuxHiddenCompressor
                | Self::AuxOutputHiddenCompressor
                | Self::Auxiliary
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

impl fmt::Display for TensorRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvCacheShape {
    None,
    FullKeysValues,
    GroupedKeysValues,
    LatentOrCompressed,
    Unknown,
}

impl KvCacheShape {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::FullKeysValues => "full_kv",
            Self::GroupedKeysValues => "grouped_kv",
            Self::LatentOrCompressed => "latent_or_compressed",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for KvCacheShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeedForwardKind {
    None,
    DenseMlp,
    RoutedExperts,
    RoutedAndSharedExperts,
    Unknown,
}

impl FeedForwardKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::DenseMlp => "dense_mlp",
            Self::RoutedExperts => "routed_experts",
            Self::RoutedAndSharedExperts => "routed_and_shared_experts",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for FeedForwardKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
