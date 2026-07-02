//! Model-family policy registry.
//!
//! Generic runtime code should not know concrete tensor names from OLMoE,
//! DeepSeek, Llama, etc. Family modules translate source artifacts into Ferrule's
//! semantic model IR: `TransformerSpec`, `TensorClass`, support contracts, and
//! conversion plans.

pub mod common;
pub mod deepseek_v4;
pub mod qwen3;

use crate::spec::ModelFamily;
use crate::tensor_policy::{TensorClass, TensorClassCount};
use crate::TransformerSpec;

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
pub enum SourceTensorPart {
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
    pub part: SourceTensorPart,
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

pub fn classify_hf_tensor(family: &ModelFamily, name: &str) -> TensorClass {
    match family {
        ModelFamily::DeepSeekV4 => deepseek_v4::classify_hf_tensor(name),
        ModelFamily::Qwen3 => qwen3::classify_hf_tensor(name),
        _ => common::classify_hf_tensor(name),
    }
}

pub fn classify_gguf_tensor(family: &ModelFamily, name: &str) -> TensorClass {
    match family {
        ModelFamily::DeepSeekV4 => deepseek_v4::classify_gguf_tensor(name),
        ModelFamily::Qwen3 => qwen3::classify_gguf_tensor(name),
        _ => common::classify_gguf_tensor(name),
    }
}

pub fn has_mla_gguf_tensor_names<'a>(
    family: &ModelFamily,
    names: impl IntoIterator<Item = &'a str>,
) -> bool {
    match family {
        ModelFamily::DeepSeekV4 => deepseek_v4::has_mla_gguf_tensor_names(names),
        _ => false,
    }
}

pub fn refine_hf_spec(spec: &mut TransformerSpec, json: &serde_json::Value) {
    match spec.family {
        ModelFamily::DeepSeekV4 => deepseek_v4::refine_hf_spec(spec, json),
        _ => {}
    }
}

pub fn append_gguf_notes(
    family: &ModelFamily,
    tensor_classes: &[TensorClassCount],
    notes: &mut Vec<String>,
) {
    match family {
        ModelFamily::DeepSeekV4 => deepseek_v4::append_gguf_notes(tensor_classes, notes),
        _ => {}
    }
}

pub fn parse_hf_routed_expert_tensor(
    family: &ModelFamily,
    name: &str,
) -> Option<RoutedExpertTensorRef> {
    match family {
        ModelFamily::DeepSeekV4 => deepseek_v4::parse_hf_routed_expert_tensor(name),
        _ => None,
    }
}

pub fn parse_hf_shared_expert_tensor(
    family: &ModelFamily,
    name: &str,
) -> Option<SharedExpertTensorRef> {
    match family {
        ModelFamily::DeepSeekV4 => deepseek_v4::parse_hf_shared_expert_tensor(name),
        _ => None,
    }
}

pub fn parse_hf_router_tensor(family: &ModelFamily, name: &str) -> Option<RouterTensorRef> {
    match family {
        ModelFamily::DeepSeekV4 => deepseek_v4::parse_hf_router_tensor(name),
        _ => None,
    }
}

pub fn parse_hf_attention_tensor(family: &ModelFamily, name: &str) -> Option<AttentionTensorRef> {
    match family {
        ModelFamily::DeepSeekV4 => deepseek_v4::parse_hf_attention_tensor(name),
        _ => None,
    }
}

pub fn parse_hf_hyper_connection_tensor(
    family: &ModelFamily,
    name: &str,
) -> Option<HyperConnectionTensorRef> {
    match family {
        ModelFamily::DeepSeekV4 => deepseek_v4::parse_hf_hyper_connection_tensor(name),
        _ => None,
    }
}
