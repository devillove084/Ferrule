use crate::tensor_policy::{TensorClass, TensorClassCount};

use super::roles::TensorRole;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorBinding {
    pub tensor_class: TensorClass,
    pub role: TensorRole,
    pub tensors: usize,
}

impl TensorBinding {
    pub fn from_class_count(count: &TensorClassCount) -> Self {
        Self {
            tensor_class: count.class.clone(),
            role: tensor_role_for_class(&count.class),
            tensors: count.tensors,
        }
    }
}

pub fn tensor_role_for_class(class: &TensorClass) -> TensorRole {
    match class {
        TensorClass::TokenEmbedding => TensorRole::TokenEmbedding,
        TensorClass::OutputNorm => TensorRole::OutputNorm,
        TensorClass::OutputHead => TensorRole::OutputHead,
        TensorClass::LayerNorm => TensorRole::LayerNorm,
        TensorClass::AttentionNorm => TensorRole::AttentionNorm,
        TensorClass::FeedForwardNorm => TensorRole::FeedForwardNorm,
        TensorClass::AttentionQuery => TensorRole::AttentionQuery,
        TensorClass::AttentionKey => TensorRole::AttentionKey,
        TensorClass::AttentionValue => TensorRole::AttentionValue,
        TensorClass::AttentionOutput => TensorRole::AttentionOutput,
        TensorClass::AttentionSink => TensorRole::AttentionSink,
        TensorClass::DenseMlpGate => TensorRole::DenseMlpGate,
        TensorClass::DenseMlpUp => TensorRole::DenseMlpUp,
        TensorClass::DenseMlpDown => TensorRole::DenseMlpDown,
        TensorClass::MlaQueryA => TensorRole::AttentionLatentQueryA,
        TensorClass::MlaQueryB => TensorRole::AttentionLatentQueryB,
        TensorClass::MlaQueryNorm => TensorRole::AttentionQueryNorm,
        TensorClass::MlaKv => TensorRole::AttentionLatentKv,
        TensorClass::MlaKvNorm => TensorRole::AttentionKeyValueNorm,
        TensorClass::MlaOutputA => TensorRole::AttentionLatentOutputA,
        TensorClass::MlaOutputB => TensorRole::AttentionLatentOutputB,
        TensorClass::MlaCompressor => TensorRole::AttentionCompressor,
        TensorClass::Router => TensorRole::RouterLogits,
        TensorClass::RouterBias => TensorRole::RouterBias,
        TensorClass::RoutedExpertGate => TensorRole::RoutedExpertGate,
        TensorClass::RoutedExpertUp => TensorRole::RoutedExpertUp,
        TensorClass::RoutedExpertDown => TensorRole::RoutedExpertDown,
        TensorClass::SharedExpertGate => TensorRole::SharedExpertGate,
        TensorClass::SharedExpertUp => TensorRole::SharedExpertUp,
        TensorClass::SharedExpertDown => TensorRole::SharedExpertDown,
        TensorClass::HashRouterTable => TensorRole::HashRouterTable,
        TensorClass::Indexer => TensorRole::AuxIndexer,
        TensorClass::HiddenCompressor => TensorRole::AuxHiddenCompressor,
        TensorClass::OutputHiddenCompressor => TensorRole::AuxOutputHiddenCompressor,
        TensorClass::SpeculativeProjection => TensorRole::SpeculativeProjection,
        TensorClass::SpeculativeMarkovHead => TensorRole::SpeculativeMarkovHead,
        TensorClass::SpeculativeConfidenceHead => TensorRole::SpeculativeConfidenceHead,
        TensorClass::Auxiliary => TensorRole::Auxiliary,
        TensorClass::Unknown => TensorRole::Unknown,
    }
}
