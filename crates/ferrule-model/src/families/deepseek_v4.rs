use crate::spec::TransformerSpec;
use crate::tensor_policy::{TensorClass, TensorClassCount};

/// Official DeepSeek-V4 / DeepSeek-V4-Flash artifact metadata defaults.
///
/// These constants belong to the model-family layer. Runtime code should consume
/// semantic layouts, policies, and payload shapes derived from descriptors rather
/// than hard-coding these values in generic execution paths.
pub const HIDDEN_SIZE: usize = 4096;
pub const HC_MULT: usize = 4;
pub const HC_SINKHORN_ITERS: usize = 20;
pub const HC_EPS: f32 = 1e-6;
pub const RMS_NORM_EPS: f32 = 1e-6;
pub const NUM_LAYERS: usize = 43;
pub const NUM_HASH_LAYERS: usize = 3;
pub const N_ROUTED_EXPERTS: usize = 256;
pub const NUM_EXPERTS_PER_TOK: usize = 6;
pub const MOE_INTERMEDIATE_SIZE: usize = 2048;
pub const HEAD_DIM: usize = 512;
pub const QK_ROPE_HEAD_DIM: usize = 64;
pub const NUM_HEADS: usize = 64;
pub const NUM_KV_HEADS: usize = 1;
pub const Q_LORA_RANK: usize = 1024;
pub const O_GROUPS: usize = 8;
pub const O_LORA_RANK: usize = 1024;
pub const SLIDING_WINDOW: usize = 128;
pub const VOCAB_SIZE: usize = 129_280;
pub const SWIGLU_LIMIT: f32 = 10.0;
pub const ROUTED_SCALING_FACTOR: f32 = 1.5;
pub const ROPE_THETA: f32 = 10_000.0;
pub const COMPRESS_ROPE_THETA: f32 = 160_000.0;
pub const ORIGINAL_MAX_POSITION_EMBEDDINGS: usize = 65_536;
pub const ROPE_FACTOR: f32 = 16.0;
pub const ROPE_BETA_FAST: usize = 32;
pub const ROPE_BETA_SLOW: usize = 1;
pub const INDEX_N_HEADS: usize = 64;
pub const INDEX_HEAD_DIM: usize = 128;
pub const INDEX_TOPK: usize = 512;
pub const DSPARK_BLOCK_SIZE: usize = 5;

use super::common;
use crate::semantic::{
    ArtifactTensorPart, AttentionTensorKind, AttentionTensorRef, HyperConnectionStage,
    HyperConnectionTensorKind, HyperConnectionTensorRef, RoutedExpertMatrix,
    RoutedExpertTensorPart, RoutedExpertTensorRef, RouterTensorKind, RouterTensorRef,
    SharedExpertTensorRef,
};

/// DeepSeek-V4 / DeepSeek-V4-Flash tensor-name policy.
///
/// Keep concrete artifact names here. The rest of the engine should consume
/// semantic `TensorClass` / `TensorRole` values instead of matching on strings
/// like `layers.0.attn.wq_a.weight` or `mtp.2.markov_head.markov_w1.weight`.
pub fn classify_hf_tensor(name: &str) -> TensorClass {
    classify_tensor_name(name)
}

pub fn classify_gguf_tensor(name: &str) -> TensorClass {
    classify_tensor_name(name)
}

pub fn has_mla_gguf_tensor_names<'a>(names: impl IntoIterator<Item = &'a str>) -> bool {
    names
        .into_iter()
        .any(|name| name.contains("attn_q_a") || name.contains("attn_compressor"))
}

pub fn refine_hf_spec(spec: &mut TransformerSpec, json: &serde_json::Value) {
    refine_semantics(spec, json);
    if has_dspark_metadata(json) {
        spec.notes.push(format!(
            "DSpark attachment metadata detected: block_size={}, target_layers={}, markov_rank={}",
            fmt_json_opt(json.get("dspark_block_size")),
            json_array_len(json.get("dspark_target_layer_ids"))
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unknown".into()),
            fmt_json_opt(json.get("dspark_markov_rank")),
        ));
    }
}

pub fn append_gguf_notes(tensor_classes: &[TensorClassCount], notes: &mut Vec<String>) {
    let mla = tensor_classes
        .iter()
        .filter(|item| item.class.is_mla_attention())
        .map(|item| item.tensors)
        .sum::<usize>();
    let routed = tensor_classes
        .iter()
        .filter(|item| item.class.is_routed_expert())
        .map(|item| item.tensors)
        .sum::<usize>();
    let aux = tensor_classes
        .iter()
        .filter(|item| item.class.is_auxiliary())
        .map(|item| item.tensors)
        .sum::<usize>();
    let speculative = tensor_classes
        .iter()
        .filter(|item| item.class.is_speculative())
        .map(|item| item.tensors)
        .sum::<usize>();

    notes.push(
        "DeepSeek-V4 GGUF is detected for metadata/inspection; execution needs MLA + GGUF K/IQ quant support"
            .into(),
    );
    notes.push(format!(
        "DeepSeek-V4 tensor policy classified {mla} MLA, {routed} routed-expert, {aux} auxiliary, and {speculative} speculative tensors"
    ));
}

pub fn parse_hf_routed_expert_tensor(name: &str) -> Option<RoutedExpertTensorRef> {
    let parts = name.split('.').collect::<Vec<_>>();
    let (layer, expert, matrix_name, part_name) = match parts.as_slice() {
        ["layers", layer, "ffn", "experts", expert, matrix, part] => {
            (*layer, *expert, *matrix, *part)
        }
        _ => return None,
    };
    let layer = layer.parse::<usize>().ok()?;
    let expert = expert.parse::<usize>().ok()?;
    let matrix = match matrix_name {
        "w1" => RoutedExpertMatrix::Gate,
        "w3" => RoutedExpertMatrix::Up,
        "w2" => RoutedExpertMatrix::Down,
        _ => return None,
    };
    let part = match part_name {
        "weight" => RoutedExpertTensorPart::Weight,
        "scale" => RoutedExpertTensorPart::Scale,
        other => RoutedExpertTensorPart::Other(other.to_string()),
    };
    Some(RoutedExpertTensorRef {
        layer,
        expert,
        matrix,
        part,
    })
}

pub fn parse_hf_shared_expert_tensor(name: &str) -> Option<SharedExpertTensorRef> {
    let parts = name.split('.').collect::<Vec<_>>();
    let (layer, matrix_name, part_name) = match parts.as_slice() {
        ["layers", layer, "ffn", "shared_experts", matrix, part] => (*layer, *matrix, *part),
        _ => return None,
    };
    let layer = layer.parse::<usize>().ok()?;
    let matrix = match matrix_name {
        "w1" => RoutedExpertMatrix::Gate,
        "w3" => RoutedExpertMatrix::Up,
        "w2" => RoutedExpertMatrix::Down,
        _ => return None,
    };
    let part = match part_name {
        "weight" => RoutedExpertTensorPart::Weight,
        "scale" => RoutedExpertTensorPart::Scale,
        other => RoutedExpertTensorPart::Other(other.to_string()),
    };
    Some(SharedExpertTensorRef {
        layer,
        matrix,
        part,
    })
}

pub fn parse_hf_router_tensor(name: &str) -> Option<RouterTensorRef> {
    let parts = name.split('.').collect::<Vec<_>>();
    let (layer, field) = match parts.as_slice() {
        ["layers", layer, "ffn", "gate", field] => (*layer, *field),
        _ => return None,
    };
    let layer = layer.parse::<usize>().ok()?;
    let kind = match field {
        "weight" => RouterTensorKind::Weight,
        "bias" => RouterTensorKind::Bias,
        "tid2eid" => RouterTensorKind::HashTable,
        other => RouterTensorKind::Other(other.to_string()),
    };
    Some(RouterTensorRef { layer, kind })
}

pub fn parse_hf_attention_tensor(name: &str) -> Option<AttentionTensorRef> {
    let parts = name.split('.').collect::<Vec<_>>();
    let (layer, field, part) = match parts.as_slice() {
        // Nested compressor/indexer tensors are intentionally represented as
        // attention auxiliaries. Runtime binding can keep them optional while the
        // core attention linears stay strongly typed.
        ["layers", layer, "attn", "compressor", ..] => {
            let layer = layer.parse::<usize>().ok()?;
            return Some(AttentionTensorRef {
                layer,
                kind: AttentionTensorKind::Compressor,
                part: ArtifactTensorPart::Other,
            });
        }
        ["layers", layer, "attn", "indexer", ..] => {
            let layer = layer.parse::<usize>().ok()?;
            return Some(AttentionTensorRef {
                layer,
                kind: AttentionTensorKind::Indexer,
                part: ArtifactTensorPart::Other,
            });
        }
        ["layers", layer, "attn", field, part] => (*layer, *field, Some(*part)),
        ["layers", layer, "attn", field] => (*layer, *field, None),
        _ => return None,
    };
    let layer = layer.parse::<usize>().ok()?;
    let kind = match field {
        "wq_a" => AttentionTensorKind::QueryA,
        "wq_b" => AttentionTensorKind::QueryB,
        "wkv" => AttentionTensorKind::KeyValue,
        "wo_a" => AttentionTensorKind::OutputA,
        "wo_b" => AttentionTensorKind::OutputB,
        "q_norm" => AttentionTensorKind::QueryNorm,
        "kv_norm" => AttentionTensorKind::KeyValueNorm,
        "attn_sink" => AttentionTensorKind::AttentionSink,
        _ => return None,
    };
    let part = match part {
        Some("weight") => ArtifactTensorPart::Weight,
        Some("scale") => ArtifactTensorPart::Scale,
        Some(_) | None => ArtifactTensorPart::Other,
    };
    Some(AttentionTensorRef { layer, kind, part })
}

pub fn parse_hf_hyper_connection_tensor(name: &str) -> Option<HyperConnectionTensorRef> {
    let parts = name.split('.').collect::<Vec<_>>();
    match parts.as_slice() {
        ["layers", layer, field] => {
            let layer = layer.parse::<usize>().ok()?;
            let (stage, kind) = parse_hc_field(field)?;
            Some(HyperConnectionTensorRef {
                layer: Some(layer),
                stage,
                kind,
            })
        }
        [field] => {
            let (stage, kind) = parse_hc_field(field)?;
            (stage == HyperConnectionStage::Head).then_some(HyperConnectionTensorRef {
                layer: None,
                stage,
                kind,
            })
        }
        _ => None,
    }
}

fn parse_hc_field(field: &str) -> Option<(HyperConnectionStage, HyperConnectionTensorKind)> {
    let (stage, suffix) = if let Some(suffix) = field.strip_prefix("hc_attn_") {
        (HyperConnectionStage::Attention, suffix)
    } else if let Some(suffix) = field.strip_prefix("hc_ffn_") {
        (HyperConnectionStage::FeedForward, suffix)
    } else if let Some(suffix) = field.strip_prefix("hc_head_") {
        (HyperConnectionStage::Head, suffix)
    } else {
        return None;
    };
    let kind = match suffix {
        "fn" => HyperConnectionTensorKind::Function,
        "scale" => HyperConnectionTensorKind::Scale,
        "base" => HyperConnectionTensorKind::Base,
        _ => return None,
    };
    Some((stage, kind))
}

fn classify_tensor_name(name: &str) -> TensorClass {
    // Top-level tensors.
    match name {
        "token_embd.weight"
        | "tok_embeddings.weight"
        | "model.embed_tokens.weight"
        | "embed.weight" => return TensorClass::TokenEmbedding,
        "output_norm.weight" | "model.norm.weight" | "norm.weight" => {
            return TensorClass::OutputNorm;
        }
        "output.weight" | "lm_head.weight" | "head.weight" => return TensorClass::OutputHead,
        _ => {}
    }

    // DeepSeek-V4 layer-stage norms.
    if name.contains(".attn_norm") || name.contains("attn_norm.weight") {
        return TensorClass::AttentionNorm;
    }
    if name.contains(".ffn_norm") || name.contains("ffn_norm.weight") {
        return TensorClass::FeedForwardNorm;
    }

    // DeepSeek-V4 MLA / CSA-HCA-compatible attention layouts.
    if name.contains("attn_sink") {
        return TensorClass::AttentionSink;
    }
    if name.contains("attn_q_a") || name.contains(".attn.wq_a") {
        return TensorClass::MlaQueryA;
    }
    if name.contains("attn_q_b") || name.contains(".attn.wq_b") {
        return TensorClass::MlaQueryB;
    }
    if name.contains("attn_q_norm") || name.contains(".attn.q_norm") {
        return TensorClass::MlaQueryNorm;
    }
    if name.contains("attn_kv_norm") || name.contains(".attn.kv_norm") {
        return TensorClass::MlaKvNorm;
    }
    if name.contains("attn_kv") || name.contains(".attn.wkv") {
        return TensorClass::MlaKv;
    }
    if name.contains("attn_output_a") || name.contains(".attn.wo_a") {
        return TensorClass::MlaOutputA;
    }
    if name.contains("attn_output_b") || name.contains(".attn.wo_b") {
        return TensorClass::MlaOutputB;
    }
    if name.contains("attn_compressor") || name.contains(".attn.compressor") {
        return TensorClass::MlaCompressor;
    }

    // DSpark / MTP speculative attachment tensors.
    if name.starts_with("mtp.") && name.contains(".main_proj.") {
        return TensorClass::SpeculativeProjection;
    }
    if name.starts_with("mtp.") && name.contains(".markov_head.") {
        return TensorClass::SpeculativeMarkovHead;
    }
    if name.starts_with("mtp.") && name.contains(".confidence_head.") {
        return TensorClass::SpeculativeConfidenceHead;
    }

    // Routing and routed/shared experts.
    if name.contains("ffn_gate_tid2eid") || name.contains(".ffn.gate.tid2eid") {
        return TensorClass::HashRouterTable;
    }
    if name.contains("exp_probs_b") || name.contains(".ffn.gate.bias") {
        return TensorClass::RouterBias;
    }
    if name.contains("ffn_gate_inp") || name.contains("router") || name.contains(".ffn.gate.") {
        return TensorClass::Router;
    }
    if name.contains("ffn_gate_exps") || (name.contains(".ffn.experts.") && name.contains(".w1.")) {
        return TensorClass::RoutedExpertGate;
    }
    if name.contains("ffn_up_exps") || (name.contains(".ffn.experts.") && name.contains(".w3.")) {
        return TensorClass::RoutedExpertUp;
    }
    if name.contains("ffn_down_exps") || (name.contains(".ffn.experts.") && name.contains(".w2.")) {
        return TensorClass::RoutedExpertDown;
    }
    if name.contains("ffn_gate_shexp")
        || name.contains("shared_expert_gate")
        || (name.contains(".ffn.shared_experts.") && name.contains(".w1."))
    {
        return TensorClass::SharedExpertGate;
    }
    if name.contains("ffn_up_shexp")
        || name.contains("shared_expert_up")
        || (name.contains(".ffn.shared_experts.") && name.contains(".w3."))
    {
        return TensorClass::SharedExpertUp;
    }
    if name.contains("ffn_down_shexp")
        || name.contains("shared_expert_down")
        || (name.contains(".ffn.shared_experts.") && name.contains(".w2."))
    {
        return TensorClass::SharedExpertDown;
    }

    // DeepSeek-V4 hyper-connection / compressor / indexer auxiliary blocks.
    if name.contains("indexer") {
        return TensorClass::Indexer;
    }
    if name.contains("output_hc") {
        return TensorClass::OutputHiddenCompressor;
    }
    if name.contains("hc_") || name.contains(".hc") {
        return TensorClass::HiddenCompressor;
    }

    common::classify_hf_tensor(name)
}

fn refine_semantics(spec: &mut TransformerSpec, json: &serde_json::Value) {
    let semantics = &mut spec.semantics;
    semantics.norm_epsilon.get_or_insert(RMS_NORM_EPS);
    semantics.hyper_connection_epsilon.get_or_insert(HC_EPS);
    semantics
        .hyper_connection_sinkhorn_iters
        .get_or_insert(HC_SINKHORN_ITERS);
    semantics.rope_theta.get_or_insert(ROPE_THETA);
    semantics.rope_head_dim.get_or_insert(QK_ROPE_HEAD_DIM);
    semantics
        .compress_rope_theta
        .get_or_insert(COMPRESS_ROPE_THETA);
    semantics
        .attention_window_size
        .get_or_insert(SLIDING_WINDOW);
    semantics.attention_index_topk.get_or_insert(INDEX_TOPK);
    semantics
        .attention_index_num_heads
        .get_or_insert(INDEX_N_HEADS);
    semantics
        .attention_index_head_dim
        .get_or_insert(INDEX_HEAD_DIM);
    semantics.output_projection_groups.get_or_insert(O_GROUPS);
    semantics.output_projection_rank.get_or_insert(O_LORA_RANK);
    semantics.swiglu_limit.get_or_insert(SWIGLU_LIMIT);
    semantics.route_scale.get_or_insert(ROUTED_SCALING_FACTOR);
    semantics.num_hash_layers.get_or_insert(NUM_HASH_LAYERS);
    let rope_scaling = json.get("rope_scaling").unwrap_or(&serde_json::Value::Null);
    semantics.rope_factor.get_or_insert_with(|| {
        f32_json_key(rope_scaling, &["factor"])
            .or_else(|| f32_json_key(json, &["rope_factor"]))
            .unwrap_or(ROPE_FACTOR)
    });
    semantics
        .rope_original_max_position_embeddings
        .get_or_insert_with(|| {
            usize_json_key(rope_scaling, &["original_max_position_embeddings"])
                .or_else(|| usize_json_key(json, &["original_seq_len"]))
                .unwrap_or(ORIGINAL_MAX_POSITION_EMBEDDINGS)
        });
    semantics.rope_beta_fast.get_or_insert_with(|| {
        usize_json_key(rope_scaling, &["beta_fast"])
            .or_else(|| usize_json_key(json, &["beta_fast"]))
            .unwrap_or(ROPE_BETA_FAST)
    });
    semantics.rope_beta_slow.get_or_insert_with(|| {
        usize_json_key(rope_scaling, &["beta_slow"])
            .or_else(|| usize_json_key(json, &["beta_slow"]))
            .unwrap_or(ROPE_BETA_SLOW)
    });
    if semantics.attention_compress_ratios.is_empty() {
        semantics.attention_compress_ratios = vec![0; spec.num_layers.unwrap_or(NUM_LAYERS)];
    }
}

fn usize_json_key(json: &serde_json::Value, keys: &[&str]) -> Option<usize> {
    keys.iter().find_map(|key| {
        json.get(*key)
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
    })
}

fn f32_json_key(json: &serde_json::Value, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|key| {
        json.get(*key)
            .and_then(|value| value.as_f64())
            .map(|value| value as f32)
    })
}

fn has_dspark_metadata(json: &serde_json::Value) -> bool {
    [
        "dspark_block_size",
        "dspark_target_layer_ids",
        "dspark_markov_rank",
    ]
    .iter()
    .any(|key| !json.get(*key).unwrap_or(&serde_json::Value::Null).is_null())
}

fn json_array_len(value: Option<&serde_json::Value>) -> Option<usize> {
    value.and_then(|value| value.as_array()).map(Vec::len)
}

fn fmt_json_opt(value: Option<&serde_json::Value>) -> String {
    value
        .map(|value| match value {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .unwrap_or_else(|| "unknown".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_mla_tensors() {
        assert_eq!(
            classify_gguf_tensor("blk.0.attn_q_a.weight"),
            TensorClass::MlaQueryA
        );
        assert_eq!(
            classify_gguf_tensor("blk.0.attn_q_norm.weight"),
            TensorClass::MlaQueryNorm
        );
        assert_eq!(
            classify_gguf_tensor("blk.0.attn_kv.weight"),
            TensorClass::MlaKv
        );
        assert_eq!(
            classify_gguf_tensor("blk.0.attn_kv_norm.weight"),
            TensorClass::MlaKvNorm
        );
        assert_eq!(
            classify_gguf_tensor("blk.0.attn_output_b.weight"),
            TensorClass::MlaOutputB
        );
    }

    #[test]
    fn classifies_experts_and_router() {
        assert_eq!(
            classify_gguf_tensor("blk.3.ffn_gate_exps.weight"),
            TensorClass::RoutedExpertGate
        );
        assert_eq!(
            classify_gguf_tensor("blk.3.ffn_down_exps.weight"),
            TensorClass::RoutedExpertDown
        );
        assert_eq!(
            classify_gguf_tensor("blk.0.ffn_gate_tid2eid"),
            TensorClass::HashRouterTable
        );
        assert_eq!(
            classify_hf_tensor("layers.0.ffn.gate.tid2eid"),
            TensorClass::HashRouterTable
        );
        assert_eq!(
            classify_gguf_tensor("blk.7.exp_probs_b"),
            TensorClass::RouterBias
        );
        assert_eq!(
            classify_hf_tensor("layers.7.ffn.gate.bias"),
            TensorClass::RouterBias
        );
    }

    #[test]
    fn classifies_hf_attention_tensors() {
        assert_eq!(
            classify_hf_tensor("layers.0.attn.wq_a.weight"),
            TensorClass::MlaQueryA
        );
        assert_eq!(
            classify_hf_tensor("layers.0.attn.q_norm.weight"),
            TensorClass::MlaQueryNorm
        );
        assert_eq!(
            classify_hf_tensor("layers.0.attn.wkv.scale"),
            TensorClass::MlaKv
        );
        assert_eq!(
            classify_hf_tensor("layers.0.attn.kv_norm.weight"),
            TensorClass::MlaKvNorm
        );
        assert_eq!(
            classify_hf_tensor("layers.0.attn.wo_b.weight"),
            TensorClass::MlaOutputB
        );
        assert_eq!(
            classify_hf_tensor("layers.0.attn.attn_sink"),
            TensorClass::AttentionSink
        );
    }

    #[test]
    fn parses_hf_routed_expert_tensor_refs() {
        assert_eq!(
            parse_hf_routed_expert_tensor("layers.12.ffn.experts.7.w1.weight"),
            Some(RoutedExpertTensorRef {
                layer: 12,
                expert: 7,
                matrix: RoutedExpertMatrix::Gate,
                part: RoutedExpertTensorPart::Weight,
            })
        );
        assert_eq!(
            parse_hf_routed_expert_tensor("layers.12.ffn.experts.7.w3.scale"),
            Some(RoutedExpertTensorRef {
                layer: 12,
                expert: 7,
                matrix: RoutedExpertMatrix::Up,
                part: RoutedExpertTensorPart::Scale,
            })
        );
        assert_eq!(
            parse_hf_routed_expert_tensor("layers.12.ffn.experts.7.w2.weight")
                .map(|tensor| tensor.matrix),
            Some(RoutedExpertMatrix::Down)
        );
        assert_eq!(
            parse_hf_routed_expert_tensor("layers.12.ffn.shared_experts.w2.weight"),
            None
        );
        assert_eq!(
            parse_hf_routed_expert_tensor("mtp.0.ffn.experts.7.w1.weight"),
            None
        );
    }

    #[test]
    fn parses_hf_attention_tensor_refs() {
        assert_eq!(
            parse_hf_attention_tensor("layers.2.attn.wq_a.weight"),
            Some(AttentionTensorRef {
                layer: 2,
                kind: AttentionTensorKind::QueryA,
                part: ArtifactTensorPart::Weight,
            })
        );
        assert_eq!(
            parse_hf_attention_tensor("layers.2.attn.wq_b.scale"),
            Some(AttentionTensorRef {
                layer: 2,
                kind: AttentionTensorKind::QueryB,
                part: ArtifactTensorPart::Scale,
            })
        );
        assert_eq!(
            parse_hf_attention_tensor("layers.2.attn.q_norm.weight"),
            Some(AttentionTensorRef {
                layer: 2,
                kind: AttentionTensorKind::QueryNorm,
                part: ArtifactTensorPart::Weight,
            })
        );
        assert_eq!(
            parse_hf_attention_tensor("layers.2.attn.attn_sink"),
            Some(AttentionTensorRef {
                layer: 2,
                kind: AttentionTensorKind::AttentionSink,
                part: ArtifactTensorPart::Other,
            })
        );
        assert_eq!(
            parse_hf_attention_tensor("layers.2.attn.compressor.wkv.weight")
                .map(|tensor| tensor.kind),
            Some(AttentionTensorKind::Compressor)
        );
        assert_eq!(
            parse_hf_attention_tensor("layers.2.attn.indexer.wq_b.weight")
                .map(|tensor| tensor.kind),
            Some(AttentionTensorKind::Indexer)
        );
    }

    #[test]
    fn parses_hf_hyper_connection_tensor_refs() {
        assert_eq!(
            parse_hf_hyper_connection_tensor("layers.0.hc_attn_fn"),
            Some(HyperConnectionTensorRef {
                layer: Some(0),
                stage: HyperConnectionStage::Attention,
                kind: HyperConnectionTensorKind::Function,
            })
        );
        assert_eq!(
            parse_hf_hyper_connection_tensor("layers.0.hc_ffn_scale"),
            Some(HyperConnectionTensorRef {
                layer: Some(0),
                stage: HyperConnectionStage::FeedForward,
                kind: HyperConnectionTensorKind::Scale,
            })
        );
        assert_eq!(
            parse_hf_hyper_connection_tensor("hc_head_base"),
            Some(HyperConnectionTensorRef {
                layer: None,
                stage: HyperConnectionStage::Head,
                kind: HyperConnectionTensorKind::Base,
            })
        );
    }

    #[test]
    fn parses_hf_shared_expert_and_router_tensor_refs() {
        assert_eq!(
            parse_hf_shared_expert_tensor("layers.12.ffn.shared_experts.w1.weight"),
            Some(SharedExpertTensorRef {
                layer: 12,
                matrix: RoutedExpertMatrix::Gate,
                part: RoutedExpertTensorPart::Weight,
            })
        );
        assert_eq!(
            parse_hf_shared_expert_tensor("layers.12.ffn.shared_experts.w3.scale"),
            Some(SharedExpertTensorRef {
                layer: 12,
                matrix: RoutedExpertMatrix::Up,
                part: RoutedExpertTensorPart::Scale,
            })
        );
        assert_eq!(
            parse_hf_router_tensor("layers.3.ffn.gate.weight"),
            Some(RouterTensorRef {
                layer: 3,
                kind: RouterTensorKind::Weight,
            })
        );
        assert_eq!(
            parse_hf_router_tensor("layers.3.ffn.gate.bias"),
            Some(RouterTensorRef {
                layer: 3,
                kind: RouterTensorKind::Bias,
            })
        );
        assert_eq!(
            parse_hf_router_tensor("layers.0.ffn.gate.tid2eid"),
            Some(RouterTensorRef {
                layer: 0,
                kind: RouterTensorKind::HashTable,
            })
        );
    }

    #[test]
    fn classifies_hf_expert_tensors() {
        assert_eq!(
            classify_hf_tensor("layers.0.ffn.experts.3.w1.weight"),
            TensorClass::RoutedExpertGate
        );
        assert_eq!(
            classify_hf_tensor("layers.0.ffn.experts.3.w3.scale"),
            TensorClass::RoutedExpertUp
        );
        assert_eq!(
            classify_hf_tensor("layers.0.ffn.experts.3.w2.weight"),
            TensorClass::RoutedExpertDown
        );
        assert_eq!(
            classify_hf_tensor("layers.0.ffn.shared_experts.w2.weight"),
            TensorClass::SharedExpertDown
        );
    }

    #[test]
    fn classifies_auxiliary_tensors() {
        assert_eq!(
            classify_gguf_tensor("blk.10.indexer.weight"),
            TensorClass::Indexer
        );
        assert_eq!(
            classify_gguf_tensor("blk.10.hc_gate.weight"),
            TensorClass::HiddenCompressor
        );
        assert_eq!(
            classify_gguf_tensor("output_hc.weight"),
            TensorClass::OutputHiddenCompressor
        );
    }

    #[test]
    fn classifies_dspark_mtp_tensors() {
        assert_eq!(
            classify_hf_tensor("mtp.0.main_proj.weight"),
            TensorClass::SpeculativeProjection
        );
        assert_eq!(
            classify_hf_tensor("mtp.2.markov_head.markov_w1.weight"),
            TensorClass::SpeculativeMarkovHead
        );
        assert_eq!(
            classify_hf_tensor("mtp.2.confidence_head.proj.weight"),
            TensorClass::SpeculativeConfidenceHead
        );
    }

    #[test]
    fn deepseek_hf_spec_records_dspark_metadata() {
        let mut spec = TransformerSpec {
            family: crate::ModelFamily::DeepSeekV4,
            architecture: Some("DeepseekV4ForCausalLM".into()),
            weight_source: crate::WeightSource::Safetensors,
            hidden_size: None,
            num_layers: None,
            vocab_size: None,
            num_heads: None,
            num_kv_heads: None,
            head_dim: None,
            attention: crate::AttentionKind::MultiLatentAttention,
            moe: crate::MoeSpec::none(),
            semantics: Default::default(),
            tensor_count: None,
            quantization: Vec::new(),
            notes: Vec::new(),
        };
        let json = serde_json::json!({
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [40, 41, 42],
            "dspark_markov_rank": 256
        });
        refine_hf_spec(&mut spec, &json);
        assert!(
            spec.notes
                .iter()
                .any(|note| note.contains("DSpark attachment metadata"))
        );
    }
}
