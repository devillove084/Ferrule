//! Qwen3 model-family tensor classification and HF semantic parsing.
//!
//! Qwen3-MoE uses standard decoder projections, per-head query/key RMS norms,
//! a dense top-k router, and individually stored SwiGLU expert matrices.

use crate::semantic::{
    DenseLayerTensorKind, DenseLayerTensorRef, RoutedExpertMatrix, RoutedExpertTensorPart,
    RoutedExpertTensorRef, RouterTensorKind, RouterTensorRef, SharedExpertTensorRef,
    TensorPayloadPart,
};
use crate::tensor_policy::TensorClass;

use super::common;

/// Classify a HuggingFace safetensors tensor name.
pub fn classify_hf_tensor(name: &str) -> TensorClass {
    match name {
        "model.embed_tokens.weight" => return TensorClass::TokenEmbedding,
        "model.norm.weight" => return TensorClass::OutputNorm,
        "lm_head.weight" => return TensorClass::OutputHead,
        _ => {}
    }

    if let Some(tensor) = parse_hf_routed_expert_tensor(name) {
        return match tensor.matrix {
            RoutedExpertMatrix::Gate => TensorClass::RoutedExpertGate,
            RoutedExpertMatrix::Up => TensorClass::RoutedExpertUp,
            RoutedExpertMatrix::Down => TensorClass::RoutedExpertDown,
        };
    }
    if let Some(tensor) = parse_hf_shared_expert_tensor(name) {
        return match tensor.matrix {
            RoutedExpertMatrix::Gate => TensorClass::SharedExpertGate,
            RoutedExpertMatrix::Up => TensorClass::SharedExpertUp,
            RoutedExpertMatrix::Down => TensorClass::SharedExpertDown,
        };
    }
    if let Some(tensor) = parse_hf_router_tensor(name) {
        return match tensor.kind {
            RouterTensorKind::Weight => TensorClass::Router,
            RouterTensorKind::Bias => TensorClass::RouterBias,
            RouterTensorKind::HashTable => TensorClass::HashRouterTable,
            RouterTensorKind::Other(_) => TensorClass::Unknown,
        };
    }
    if let Some(tensor) = parse_hf_dense_layer_tensor(name) {
        return match tensor.kind {
            DenseLayerTensorKind::InputNorm => TensorClass::AttentionNorm,
            DenseLayerTensorKind::PostAttentionNorm => TensorClass::FeedForwardNorm,
            DenseLayerTensorKind::AttentionQuery => TensorClass::AttentionQuery,
            DenseLayerTensorKind::AttentionKey => TensorClass::AttentionKey,
            DenseLayerTensorKind::AttentionValue => TensorClass::AttentionValue,
            DenseLayerTensorKind::AttentionOutput => TensorClass::AttentionOutput,
            DenseLayerTensorKind::AttentionQueryNorm => TensorClass::MlaQueryNorm,
            // TensorClass predates per-head GQA norm classes. Keep Qwen K-norm
            // out of the MLA KV-norm bucket; family-aware support binding promotes
            // this carrier to the key-only AttentionKeyNorm role.
            DenseLayerTensorKind::AttentionKeyNorm => TensorClass::Auxiliary,
            DenseLayerTensorKind::DenseMlpGate => TensorClass::DenseMlpGate,
            DenseLayerTensorKind::DenseMlpUp => TensorClass::DenseMlpUp,
            DenseLayerTensorKind::DenseMlpDown => TensorClass::DenseMlpDown,
        };
    }

    TensorClass::Unknown
}

/// Classify a GGUF tensor name (llama.cpp naming convention).
pub fn classify_gguf_tensor(name: &str) -> TensorClass {
    let lower = name.to_lowercase();
    if lower.contains("attn_q_norm") {
        TensorClass::MlaQueryNorm
    } else if lower.contains("attn_k_norm") {
        TensorClass::Auxiliary
    } else if lower.contains("attn_q") {
        TensorClass::AttentionQuery
    } else if lower.contains("attn_k") {
        TensorClass::AttentionKey
    } else if lower.contains("attn_v") {
        TensorClass::AttentionValue
    } else if lower.contains("attn_output") {
        TensorClass::AttentionOutput
    } else if lower.contains("ffn_gate_exps") {
        TensorClass::RoutedExpertGate
    } else if lower.contains("ffn_up_exps") {
        TensorClass::RoutedExpertUp
    } else if lower.contains("ffn_down_exps") {
        TensorClass::RoutedExpertDown
    } else if lower.contains("ffn_gate_shexp") {
        TensorClass::SharedExpertGate
    } else if lower.contains("ffn_up_shexp") {
        TensorClass::SharedExpertUp
    } else if lower.contains("ffn_down_shexp") {
        TensorClass::SharedExpertDown
    } else if lower.contains("ffn_gate_inp") {
        TensorClass::Router
    } else if lower.contains("ffn_gate") {
        TensorClass::DenseMlpGate
    } else if lower.contains("ffn_up") {
        TensorClass::DenseMlpUp
    } else if lower.contains("ffn_down") {
        TensorClass::DenseMlpDown
    } else if lower.contains("attention_norm") {
        TensorClass::AttentionNorm
    } else if lower.contains("ffn_norm") {
        TensorClass::FeedForwardNorm
    } else if lower.contains("output_norm") {
        TensorClass::OutputNorm
    } else if lower.contains("token_embd") {
        TensorClass::TokenEmbedding
    } else if lower.contains("output") {
        TensorClass::OutputHead
    } else {
        TensorClass::Unknown
    }
}

/// Parse a standard Qwen decoder tensor, including per-head Q/K norms.
pub fn parse_hf_dense_layer_tensor(name: &str) -> Option<DenseLayerTensorRef> {
    if let Some(tensor) = common::parse_hf_dense_layer_tensor(name) {
        return Some(tensor);
    }

    let (layer, rest) = strip_hf_layer_prefix(name)?;
    let field = rest.strip_prefix("self_attn.")?;
    let (field, part) = split_field_part(field)?;
    let kind = match field {
        "q_norm" => DenseLayerTensorKind::AttentionQueryNorm,
        "k_norm" => DenseLayerTensorKind::AttentionKeyNorm,
        _ => return None,
    };
    Some(DenseLayerTensorRef { layer, kind, part })
}

/// Parse `model.layers.{layer}.mlp.experts.{expert}.{projection}.{part}`.
pub fn parse_hf_routed_expert_tensor(name: &str) -> Option<RoutedExpertTensorRef> {
    let parts = name.split('.').collect::<Vec<_>>();
    let (layer, expert, projection, part) = match parts.as_slice() {
        [
            "model",
            "layers",
            layer,
            "mlp",
            "experts",
            expert,
            projection,
            part,
        ] => (*layer, *expert, *projection, *part),
        _ => return None,
    };
    Some(RoutedExpertTensorRef {
        layer: layer.parse().ok()?,
        expert: expert.parse().ok()?,
        matrix: projection_matrix(projection)?,
        part: expert_tensor_part(part),
    })
}

/// Parse a Qwen shared-expert name so unsupported artifacts can be rejected explicitly.
pub fn parse_hf_shared_expert_tensor(name: &str) -> Option<SharedExpertTensorRef> {
    let parts = name.split('.').collect::<Vec<_>>();
    let (layer, projection, part) = match parts.as_slice() {
        [
            "model",
            "layers",
            layer,
            "mlp",
            "shared_expert",
            projection,
            part,
        ]
        | [
            "model",
            "layers",
            layer,
            "mlp",
            "shared_experts",
            projection,
            part,
        ] => (*layer, *projection, *part),
        _ => return None,
    };
    Some(SharedExpertTensorRef {
        layer: layer.parse().ok()?,
        matrix: projection_matrix(projection)?,
        part: expert_tensor_part(part),
    })
}

/// Parse `model.layers.{layer}.mlp.gate.{field}` router tensors.
pub fn parse_hf_router_tensor(name: &str) -> Option<RouterTensorRef> {
    let parts = name.split('.').collect::<Vec<_>>();
    let (layer, field) = match parts.as_slice() {
        ["model", "layers", layer, "mlp", "gate", field] => (*layer, *field),
        _ => return None,
    };
    let kind = match field {
        "weight" => RouterTensorKind::Weight,
        "bias" => RouterTensorKind::Bias,
        "tid2eid" => RouterTensorKind::HashTable,
        other => RouterTensorKind::Other(other.to_owned()),
    };
    Some(RouterTensorRef {
        layer: layer.parse().ok()?,
        kind,
    })
}

fn strip_hf_layer_prefix(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    Some((layer.parse().ok()?, rest))
}

fn split_field_part(field: &str) -> Option<(&str, TensorPayloadPart)> {
    let (field, part) = field.rsplit_once('.')?;
    let part = match part {
        "weight" => TensorPayloadPart::Weight,
        "scale" => TensorPayloadPart::Scale,
        _ => TensorPayloadPart::Other,
    };
    Some((field, part))
}

fn projection_matrix(projection: &str) -> Option<RoutedExpertMatrix> {
    match projection {
        "gate_proj" => Some(RoutedExpertMatrix::Gate),
        "up_proj" => Some(RoutedExpertMatrix::Up),
        "down_proj" => Some(RoutedExpertMatrix::Down),
        _ => None,
    }
}

fn expert_tensor_part(part: &str) -> RoutedExpertTensorPart {
    match part {
        "weight" => RoutedExpertTensorPart::Weight,
        "scale" => RoutedExpertTensorPart::Scale,
        other => RoutedExpertTensorPart::Other(other.to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen_attention_head_norms() {
        assert_eq!(
            parse_hf_dense_layer_tensor("model.layers.7.self_attn.q_norm.weight"),
            Some(DenseLayerTensorRef {
                layer: 7,
                kind: DenseLayerTensorKind::AttentionQueryNorm,
                part: TensorPayloadPart::Weight,
            })
        );
        assert_eq!(
            parse_hf_dense_layer_tensor("model.layers.7.self_attn.k_norm.weight")
                .map(|tensor| tensor.kind),
            Some(DenseLayerTensorKind::AttentionKeyNorm)
        );
        assert_eq!(
            classify_hf_tensor("model.layers.7.self_attn.q_norm.weight"),
            TensorClass::MlaQueryNorm
        );
        let key_norm = classify_hf_tensor("model.layers.7.self_attn.k_norm.weight");
        assert_eq!(key_norm, TensorClass::Auxiliary);
        assert_ne!(key_norm, TensorClass::MlaKvNorm);
    }

    #[test]
    fn parses_qwen_router_and_routed_experts() {
        assert_eq!(
            parse_hf_router_tensor("model.layers.3.mlp.gate.weight"),
            Some(RouterTensorRef {
                layer: 3,
                kind: RouterTensorKind::Weight,
            })
        );
        assert_eq!(
            parse_hf_routed_expert_tensor("model.layers.12.mlp.experts.127.down_proj.weight"),
            Some(RoutedExpertTensorRef {
                layer: 12,
                expert: 127,
                matrix: RoutedExpertMatrix::Down,
                part: RoutedExpertTensorPart::Weight,
            })
        );
        assert_eq!(
            classify_hf_tensor("model.layers.1.mlp.experts.4.up_proj.weight"),
            TensorClass::RoutedExpertUp
        );
    }

    #[test]
    fn parses_shared_experts_for_explicit_rejection() {
        assert_eq!(
            parse_hf_shared_expert_tensor("model.layers.2.mlp.shared_expert.gate_proj.weight"),
            Some(SharedExpertTensorRef {
                layer: 2,
                matrix: RoutedExpertMatrix::Gate,
                part: RoutedExpertTensorPart::Weight,
            })
        );
    }
}
