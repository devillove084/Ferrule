use crate::semantic::{ArtifactTensorPart, DenseLayerTensorKind, DenseLayerTensorRef};
use crate::tensor_policy::TensorClass;

/// Common tensor-name policy for dense Transformer-style HF/GGUF names.
///
/// This module is deliberately conservative. Model-family modules should own
/// family-specific layouts, fused tensor names, auxiliary tensors, and draft /
/// speculation attachments.
pub fn classify_hf_tensor(name: &str) -> TensorClass {
    match name {
        "token_embd.weight"
        | "tok_embeddings.weight"
        | "model.embed_tokens.weight"
        | "embed.weight" => return TensorClass::TokenEmbedding,
        "output_norm.weight" | "model.norm.weight" | "norm.weight" => {
            return TensorClass::OutputNorm
        }
        "output.weight" | "lm_head.weight" | "head.weight" => return TensorClass::OutputHead,
        _ => {}
    }

    if name.contains("attn_q") || name.contains("q_proj") {
        TensorClass::AttentionQuery
    } else if name.contains("attn_k") || name.contains("k_proj") {
        TensorClass::AttentionKey
    } else if name.contains("attn_v") || name.contains("v_proj") {
        TensorClass::AttentionValue
    } else if name.contains("attn_output") || name.contains("o_proj") {
        TensorClass::AttentionOutput
    } else if name.contains("ffn_gate_exps")
        || name.contains("mlp.experts") && name.contains("gate_proj")
    {
        TensorClass::RoutedExpertGate
    } else if name.contains("ffn_up_exps")
        || name.contains("mlp.experts") && name.contains("up_proj")
    {
        TensorClass::RoutedExpertUp
    } else if name.contains("ffn_down_exps")
        || name.contains("mlp.experts") && name.contains("down_proj")
    {
        TensorClass::RoutedExpertDown
    } else if name.contains("ffn_gate_shexp")
        || name.contains("shared_expert") && name.contains("gate_proj")
    {
        TensorClass::SharedExpertGate
    } else if name.contains("ffn_up_shexp")
        || name.contains("shared_expert") && name.contains("up_proj")
    {
        TensorClass::SharedExpertUp
    } else if name.contains("ffn_down_shexp")
        || name.contains("shared_expert") && name.contains("down_proj")
    {
        TensorClass::SharedExpertDown
    } else if name.contains("mlp.gate.weight")
        || name.contains("router")
        || name.contains("ffn_gate_inp")
    {
        TensorClass::Router
    } else if name.contains("mlp.gate_proj") || name.contains("ffn_gate") {
        TensorClass::DenseMlpGate
    } else if name.contains("mlp.up_proj") || name.contains("ffn_up") {
        TensorClass::DenseMlpUp
    } else if name.contains("mlp.down_proj") || name.contains("ffn_down") {
        TensorClass::DenseMlpDown
    } else if name.contains("norm") || name.contains("layernorm") {
        TensorClass::LayerNorm
    } else {
        TensorClass::Unknown
    }
}

pub fn classify_gguf_tensor(name: &str) -> TensorClass {
    classify_hf_tensor(name)
}

pub fn parse_hf_dense_layer_tensor(name: &str) -> Option<DenseLayerTensorRef> {
    let (layer, rest) = strip_hf_layer_prefix(name)?;
    let (kind, part) = if let Some(field) = rest.strip_prefix("self_attn.") {
        let (field, part) = split_field_part(field)?;
        let kind = match field {
            "q_proj" => DenseLayerTensorKind::AttentionQuery,
            "k_proj" => DenseLayerTensorKind::AttentionKey,
            "v_proj" => DenseLayerTensorKind::AttentionValue,
            "o_proj" => DenseLayerTensorKind::AttentionOutput,
            _ => return None,
        };
        (kind, part)
    } else if let Some(field) = rest.strip_prefix("mlp.") {
        let (field, part) = split_field_part(field)?;
        let kind = match field {
            "gate_proj" => DenseLayerTensorKind::DenseMlpGate,
            "up_proj" => DenseLayerTensorKind::DenseMlpUp,
            "down_proj" => DenseLayerTensorKind::DenseMlpDown,
            _ => return None,
        };
        (kind, part)
    } else {
        let (field, part) = split_field_part(rest)?;
        let kind = match field {
            "input_layernorm" => DenseLayerTensorKind::InputNorm,
            "post_attention_layernorm" => DenseLayerTensorKind::PostAttentionNorm,
            _ => return None,
        };
        (kind, part)
    };

    Some(DenseLayerTensorRef { layer, kind, part })
}

fn strip_hf_layer_prefix(name: &str) -> Option<(usize, &str)> {
    let rest = name
        .strip_prefix("model.layers.")
        .or_else(|| name.strip_prefix("layers."))?;
    let (layer, rest) = rest.split_once('.')?;
    Some((layer.parse().ok()?, rest))
}

fn split_field_part(field: &str) -> Option<(&str, ArtifactTensorPart)> {
    let (field, part) = field.rsplit_once('.')?;
    Some((field, artifact_tensor_part(part)))
}

fn artifact_tensor_part(part: &str) -> ArtifactTensorPart {
    match part {
        "weight" => ArtifactTensorPart::Weight,
        "scale" => ArtifactTensorPart::Scale,
        _ => ArtifactTensorPart::Other,
    }
}
