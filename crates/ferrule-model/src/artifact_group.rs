//! Shared artifact group types used by both model and runtime layers.

use crate::artifact_tensor::ArtifactTensorSlice;

/// Semantic artifact group represented by a graph external.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ArtifactGroupKind {
    Attention,
    LayerNorm,
    HyperConnectionAttention,
    HyperConnectionFeedForward,
    HyperConnectionHead,
    Router,
    SharedExpert,
}

impl ArtifactGroupKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Attention => "attention",
            Self::LayerNorm => "layer_norm",
            Self::HyperConnectionAttention => "hyper_connection_attention",
            Self::HyperConnectionFeedForward => "hyper_connection_feed_forward",
            Self::HyperConnectionHead => "hyper_connection_head",
            Self::Router => "router",
            Self::SharedExpert => "shared_expert",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactObjectGroup {
    pub kind: ArtifactGroupKind,
    pub layer: Option<usize>,
    pub tensors: Vec<ArtifactTensorSlice>,
}
