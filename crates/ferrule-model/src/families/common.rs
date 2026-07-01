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
    } else if name.contains("norm") || name.contains("layernorm") {
        TensorClass::LayerNorm
    } else if name.contains("router")
        || name.contains("gate.weight")
        || name.contains("ffn_gate_inp")
    {
        TensorClass::Router
    } else {
        TensorClass::Unknown
    }
}

pub fn classify_gguf_tensor(name: &str) -> TensorClass {
    classify_hf_tensor(name)
}
