//! Semantic tensor identifiers shared by family adapters, artifact inventory, and
//! runtime binding.
//!
//! Model-family modules should translate concrete checkpoint names into these
//! semantic references. Artifact/runtime layers should consume these references
//! instead of knowing DeepSeek/Qwen/Llama tensor-name strings.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RoutedExpertMatrix {
    Gate,
    Up,
    Down,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RoutedExpertTensorPart {
    Weight,
    Scale,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RoutedExpertTensorRef {
    pub layer: usize,
    pub expert: usize,
    pub matrix: RoutedExpertMatrix,
    pub part: RoutedExpertTensorPart,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SharedExpertTensorRef {
    pub layer: usize,
    pub matrix: RoutedExpertMatrix,
    pub part: RoutedExpertTensorPart,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RouterTensorKind {
    Weight,
    Bias,
    HashTable,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RouterTensorRef {
    pub layer: usize,
    pub kind: RouterTensorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ArtifactTensorPart {
    Weight,
    Scale,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AttentionTensorKind {
    QueryA,
    QueryB,
    KeyValue,
    OutputA,
    OutputB,
    QueryNorm,
    KeyValueNorm,
    AttentionSink,
    Compressor,
    Indexer,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AttentionTensorRef {
    pub layer: usize,
    pub kind: AttentionTensorKind,
    pub part: ArtifactTensorPart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DenseLayerTensorKind {
    InputNorm,
    PostAttentionNorm,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    DenseMlpGate,
    DenseMlpUp,
    DenseMlpDown,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DenseLayerTensorRef {
    pub layer: usize,
    pub kind: DenseLayerTensorKind,
    pub part: ArtifactTensorPart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HyperConnectionStage {
    Attention,
    FeedForward,
    Head,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HyperConnectionTensorKind {
    Function,
    Scale,
    Base,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HyperConnectionTensorRef {
    pub layer: Option<usize>,
    pub stage: HyperConnectionStage,
    pub kind: HyperConnectionTensorKind,
}
