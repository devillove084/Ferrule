//! Model-family policy registry.
//!
//! Generic runtime code should not know concrete tensor names from OLMoE,
//! DeepSeek, Llama, etc. Family modules translate checkpoint artifacts into Ferrule's
//! semantic model IR: `TransformerSpec`, `TensorClass`, support contracts, and
//! conversion plans.

pub mod common;
pub mod deepseek_v4;
pub mod qwen3;

use crate::TransformerSpec;
use crate::semantic::{
    AttentionTensorRef, DenseLayerTensorRef, HyperConnectionTensorRef, RoutedExpertTensorRef,
    RouterTensorRef, SharedExpertTensorRef,
};
use crate::spec::ModelFamily;
use crate::tensor_policy::{TensorClass, TensorClassCount};

pub fn classify_hf_tensor(family: &ModelFamily, name: &str) -> TensorClass {
    match family {
        ModelFamily::DeepSeekV4 => deepseek_v4::classify_hf_tensor(name),
        ModelFamily::Qwen3 | ModelFamily::QwenMoe => qwen3::classify_hf_tensor(name),
        _ => common::classify_hf_tensor(name),
    }
}

pub fn classify_gguf_tensor(family: &ModelFamily, name: &str) -> TensorClass {
    match family {
        ModelFamily::DeepSeekV4 => deepseek_v4::classify_gguf_tensor(name),
        ModelFamily::Qwen3 | ModelFamily::QwenMoe => qwen3::classify_gguf_tensor(name),
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
    if spec.family == ModelFamily::DeepSeekV4 {
        deepseek_v4::refine_hf_spec(spec, json);
    }
}

pub fn append_gguf_notes(
    family: &ModelFamily,
    tensor_classes: &[TensorClassCount],
    notes: &mut Vec<String>,
) {
    if family == &ModelFamily::DeepSeekV4 {
        deepseek_v4::append_gguf_notes(tensor_classes, notes);
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

pub fn parse_hf_dense_layer_tensor(
    family: &ModelFamily,
    name: &str,
) -> Option<DenseLayerTensorRef> {
    match family {
        ModelFamily::Qwen3 | ModelFamily::QwenMoe => common::parse_hf_dense_layer_tensor(name),
        ModelFamily::DeepSeekV4 => None,
        _ => common::parse_hf_dense_layer_tensor(name),
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
