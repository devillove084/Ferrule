//! Qwen3 model-family tensor classification.
//!
//! Architecture: standard Llama-style decoder with GQA, SwiGLU FFN, RMSNorm, RoPE.
//! Covers both dense (0.6B–32B) and MoE (30B-A3B, 235B-A22B) variants.
//!
//! HF tensor naming: model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//!                  model.layers.{i}.mlp.{gate,up,down}_proj.weight

use crate::tensor_policy::TensorClass;

/// Classify a HuggingFace safetensors tensor name.
pub fn classify_hf_tensor(name: &str) -> TensorClass {
    if name.contains("self_attn.q_proj") {
        return TensorClass::AttentionQuery;
    }
    if name.contains("self_attn.k_proj") {
        return TensorClass::AttentionKey;
    }
    if name.contains("self_attn.v_proj") {
        return TensorClass::AttentionValue;
    }
    if name.contains("self_attn.o_proj") {
        return TensorClass::AttentionOutput;
    }
    // Router: gate_proj in MoE models acts as router
    if name.contains("mlp.gate.weight") || name.contains("mlp.gate_proj") {
        return TensorClass::Router;
    }
    // Shared experts (MoE variant)
    if name.contains("shared_expert.gate_proj") {
        return TensorClass::SharedExpertGate;
    }
    if name.contains("shared_expert.up_proj") {
        return TensorClass::SharedExpertUp;
    }
    if name.contains("shared_expert.down_proj") {
        return TensorClass::SharedExpertDown;
    }
    // Routed experts (MoE variant)
    if name.contains("mlp.experts") && name.contains("gate_proj") {
        return TensorClass::RoutedExpertGate;
    }
    if name.contains("mlp.experts") && name.contains("up_proj") {
        return TensorClass::RoutedExpertUp;
    }
    if name.contains("mlp.experts") && name.contains("down_proj") {
        return TensorClass::RoutedExpertDown;
    }
    // Embedding / output
    if name.contains("embed_tokens") {
        return TensorClass::TokenEmbedding;
    }
    if name.contains("lm_head") {
        return TensorClass::OutputHead;
    }
    // Norms
    if name.contains("model.norm") || name.contains("final_norm") {
        return TensorClass::OutputNorm;
    }
    if name.contains("input_layernorm") || name.contains("post_attention_layernorm") {
        return TensorClass::LayerNorm;
    }
    // Dense FFN projections (up/down) — no specific TensorClass variant yet
    if name.contains("mlp.up_proj") || name.contains("mlp.down_proj") {
        return TensorClass::Auxiliary;
    }
    TensorClass::Unknown
}

/// Classify a GGUF tensor name (llama.cpp naming convention).
pub fn classify_gguf_tensor(name: &str) -> TensorClass {
    let lower = name.to_lowercase();
    if lower.contains("attn_q") {
        TensorClass::AttentionQuery
    } else if lower.contains("attn_k") {
        TensorClass::AttentionKey
    } else if lower.contains("attn_v") {
        TensorClass::AttentionValue
    } else if lower.contains("attn_output") {
        TensorClass::AttentionOutput
    } else if lower.contains("ffn_gate") {
        TensorClass::Router
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
    } else if lower.contains("ffn_up") || lower.contains("ffn_down") {
        TensorClass::Auxiliary
    } else if lower.contains("attention_norm") || lower.contains("ffn_norm") {
        TensorClass::LayerNorm
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
