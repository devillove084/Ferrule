//! DeepSeek-V4 MTP (Multi-Token Prediction) model for DSpark speculative decoding.
//!
//! The DeepSeek-V4-Flash-DSpark checkpoint contains MTP layers used for
//! speculative decoding. Each MTP layer has the same structure as a main
//! transformer layer (MLA attention, MoE FFN, HyperConnection) plus a
//! `main_proj` / `main_norm` pair that projects the concatenated hidden state
//! and token embedding into the layer's input.
//!
//! The last MTP layer additionally carries prediction heads:
//! - `hc_head` – final HyperConnection reduction
//! - `norm` – final RMSNorm
//! - `markov_head` – low-rank Markov prediction head (markov_w1, markov_w2)
//! - `confidence_head` – confidence projection
//!
//! Tensor names use the `mtp.{i}.` prefix instead of `layers.{i}.`.

use std::path::Path;
use std::sync::Arc;

use crate::TensorRole;
use crate::artifact::binding::{
    bind_attention_from_hf, bind_hyper_connection_from_hf, bind_hyper_connection_head_from_hf,
    bind_router_from_hf, bind_shared_swiglu_ffn_from_hf,
};
use crate::artifact::inventory::{
    HfAttentionTensorInfo, HfHyperConnectionTensorInfo, HfRoutedExpertTensorInfo,
    HfRouterTensorInfo, HfSafetensorsTensorInfo, HfSharedExpertTensorInfo,
};
use crate::artifact::linear::ArtifactLinearPayload;
use crate::artifact::tensor::{ArtifactTensorReader, ArtifactTensorSlice};

use crate::hyper_connection::{HyperConnectionConfig, HyperConnectionHeadWeights};
use crate::moe::routing::ExpertRouterPolicy;
use crate::moe::streaming::ExpertSourceCatalog;
use crate::semantic::{
    ArtifactTensorPart, AttentionTensorKind, AttentionTensorRef, HyperConnectionStage,
    HyperConnectionTensorKind, HyperConnectionTensorRef, RoutedExpertMatrix,
    RoutedExpertTensorPart, RouterTensorKind, RouterTensorRef, SharedExpertTensorRef,
};
use ferrule_common::{Error, Result};

use super::attention::{DeepSeekV4Attention, DeepSeekV4CompressedAttentionPayload};
use super::config::{
    DSparkConfig, DeepSeekV4AttentionConfig, with_deepseek_v4_attention_execution_policies,
    with_deepseek_v4_swiglu_execution_policies,
};
use super::helpers::decode_vector_f32;
use super::layer::DeepSeekV4Layer;

/// One MTP layer's bound weights.
///
/// Structurally mirrors `DeepSeekV4Layer` but adds `main_proj` / `main_norm`
/// for the input projection step unique to MTP layers.
pub struct DeepSeekV4MtpLayer {
    /// Checkpoint stage index under `mtp.{stage}`.
    pub mtp_index: usize,
    /// Runtime layer identity. MTP stages follow the base transformer layers so
    /// CUDA prepared-resource keys never alias `layers.0/1/2`.
    pub execution_layer: usize,
    /// The normal HC + attention + MoE body reused from the target model.
    pub transformer: DeepSeekV4Layer,
    /// DSpark's target-hidden projection exists only on stage zero.
    pub main_proj: Option<ArtifactLinearPayload>,
    /// Normalization paired with `main_proj`; also stage-zero only.
    pub main_norm: Option<Vec<f32>>,
    /// Immutable routed-expert source catalog for this MTP stage.
    pub expert_source_catalog: Arc<ExpertSourceCatalog>,
}

/// The last MTP layer's prediction heads.
pub struct DeepSeekV4MtpPredictionHeads {
    pub hc_head: HyperConnectionHeadWeights,
    pub norm: Vec<f32>,
    pub markov_w1: ArtifactLinearPayload,
    pub markov_w2: ArtifactLinearPayload,
    pub confidence_proj: ArtifactLinearPayload,
}

/// Complete MTP model with all layers.
pub struct DeepSeekV4MtpModel {
    pub layers: Vec<DeepSeekV4MtpLayer>,
    pub prediction_heads: Option<DeepSeekV4MtpPredictionHeads>,
    pub config: DSparkConfig,
}

/// Output of a single MTP forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4MtpForwardOutput {
    pub token_ids: Vec<u32>,
    pub confidence_scores: Vec<f32>,
}

// ── MTP tensor name parsing ──────────────────────────────────────────────
//
// The family parse functions in `families/deepseek_v4.rs` only recognize the
// `layers.{i}.` prefix. MTP tensors use `mtp.{i}.` instead, so we provide
// dedicated parsers here that construct the same typed tensor-info structs
// expected by the existing `bind_*_from_hf` functions.

/// Parses an MTP attention tensor name into a typed `HfAttentionTensorInfo`.
///
/// Recognizes `mtp.{i}.attn.{field}.{part}` and `mtp.{i}.attn.{field}`.
fn parse_mtp_attention_tensor(info: &HfSafetensorsTensorInfo) -> Option<HfAttentionTensorInfo> {
    let parts = info.name.split('.').collect::<Vec<_>>();
    let (mtp_index, field, part) = match parts.as_slice() {
        ["mtp", idx, "attn", field, part] => (*idx, *field, Some(*part)),
        ["mtp", idx, "attn", field] => (*idx, *field, None),
        _ => return None,
    };
    let layer = mtp_index.parse::<usize>().ok()?;
    let kind = match field {
        "wq_a" => AttentionTensorKind::QueryA,
        "wq_b" => AttentionTensorKind::QueryB,
        "wkv" => AttentionTensorKind::KeyValue,
        "wo_a" => AttentionTensorKind::OutputA,
        "wo_b" => AttentionTensorKind::OutputB,
        "q_norm" => AttentionTensorKind::QueryNorm,
        "kv_norm" => AttentionTensorKind::KeyValueNorm,
        "attn_sink" => AttentionTensorKind::AttentionSink,
        // Compressor/Indexer auxiliary tensors use longer paths.
        _ if field.starts_with("compressor") => AttentionTensorKind::Compressor,
        _ if field.starts_with("indexer") => AttentionTensorKind::Indexer,
        _ => return None,
    };
    let part = match part {
        Some("weight") => ArtifactTensorPart::Weight,
        Some("scale") => ArtifactTensorPart::Scale,
        Some(_) | None => ArtifactTensorPart::Other,
    };
    Some(HfAttentionTensorInfo {
        descriptor: AttentionTensorRef { layer, kind, part },
        name: info.name.clone(),
        shard: info.shard.clone(),
        dtype: info.dtype.clone(),
        shape: info.shape.clone(),
        data_offset: info.data_offset,
        file_offset: info.file_offset,
        byte_size: info.byte_size,
    })
}

/// Parses an MTP router tensor name into a typed `HfRouterTensorInfo`.
///
/// Recognizes `mtp.{i}.ffn.gate.{field}`.
fn parse_mtp_router_tensor(info: &HfSafetensorsTensorInfo) -> Option<HfRouterTensorInfo> {
    let parts = info.name.split('.').collect::<Vec<_>>();
    let (mtp_index, field) = match parts.as_slice() {
        ["mtp", idx, "ffn", "gate", field] => (*idx, *field),
        _ => return None,
    };
    let layer = mtp_index.parse::<usize>().ok()?;
    let kind = match field {
        "weight" => RouterTensorKind::Weight,
        "bias" => RouterTensorKind::Bias,
        "tid2eid" => RouterTensorKind::HashTable,
        other => RouterTensorKind::Other(other.to_string()),
    };
    Some(HfRouterTensorInfo {
        descriptor: RouterTensorRef { layer, kind },
        name: info.name.clone(),
        shard: info.shard.clone(),
        dtype: info.dtype.clone(),
        shape: info.shape.clone(),
        data_offset: info.data_offset,
        file_offset: info.file_offset,
        byte_size: info.byte_size,
    })
}

/// Parses an MTP shared-expert tensor name into a typed `HfSharedExpertTensorInfo`.
///
/// Recognizes `mtp.{i}.ffn.shared_experts.{matrix}.{part}`.
fn parse_mtp_shared_expert_tensor(
    info: &HfSafetensorsTensorInfo,
) -> Option<HfSharedExpertTensorInfo> {
    let parts = info.name.split('.').collect::<Vec<_>>();
    let (mtp_index, matrix_name, part_name) = match parts.as_slice() {
        ["mtp", idx, "ffn", "shared_experts", matrix, part] => (*idx, *matrix, *part),
        _ => return None,
    };
    let layer = mtp_index.parse::<usize>().ok()?;
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
    Some(HfSharedExpertTensorInfo {
        descriptor: SharedExpertTensorRef {
            layer,
            matrix,
            part,
        },
        name: info.name.clone(),
        shard: info.shard.clone(),
        dtype: info.dtype.clone(),
        shape: info.shape.clone(),
        data_offset: info.data_offset,
        file_offset: info.file_offset,
        byte_size: info.byte_size,
    })
}

/// Parses one MTP routed-expert tensor while assigning its non-aliasing runtime layer.
fn parse_mtp_routed_expert_tensor(
    info: &HfSafetensorsTensorInfo,
    execution_layer: usize,
) -> Option<HfRoutedExpertTensorInfo> {
    let parts = info.name.split('.').collect::<Vec<_>>();
    let (expert, matrix_name, part_name) = match parts.as_slice() {
        ["mtp", _, "ffn", "experts", expert, matrix, part] => (*expert, *matrix, *part),
        _ => return None,
    };
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
    Some(HfRoutedExpertTensorInfo {
        descriptor: crate::semantic::RoutedExpertTensorRef {
            layer: execution_layer,
            expert,
            matrix,
            part,
        },
        name: info.name.clone(),
        shard: info.shard.clone(),
        dtype: info.dtype.clone(),
        shape: info.shape.clone(),
        data_offset: info.data_offset,
        file_offset: info.file_offset,
        byte_size: info.byte_size,
    })
}

/// Parses an MTP hyper-connection tensor name into a typed `HfHyperConnectionTensorInfo`.
///
/// Recognizes `mtp.{i}.hc_attn_{kind}`, `mtp.{i}.hc_ffn_{kind}`, and
/// `mtp.{i}.hc_head_{kind}`. For head-stage tensors, `layer` is set to `None`
/// to match `bind_hyper_connection_head_from_hf` expectations.
pub fn parse_mtp_hyper_connection_tensor(
    info: &HfSafetensorsTensorInfo,
) -> Option<HfHyperConnectionTensorInfo> {
    let parts = info.name.split('.').collect::<Vec<_>>();
    let (mtp_index, field) = match parts.as_slice() {
        ["mtp", idx, field] => (*idx, *field),
        _ => return None,
    };
    let mtp_layer = mtp_index.parse::<usize>().ok()?;
    let (stage, kind) = parse_hc_field(field)?;
    let layer = if stage == HyperConnectionStage::Head {
        None
    } else {
        Some(mtp_layer)
    };
    Some(HfHyperConnectionTensorInfo {
        descriptor: HyperConnectionTensorRef { layer, stage, kind },
        name: info.name.clone(),
        shard: info.shard.clone(),
        dtype: info.dtype.clone(),
        shape: info.shape.clone(),
        data_offset: info.data_offset,
        file_offset: info.file_offset,
        byte_size: info.byte_size,
    })
}

/// Mirrors `parse_hc_field` from `families/deepseek_v4.rs` for MTP tensor names.
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

/// Finds a named tensor in a slice of MTP tensors.
fn mtp_tensor_slice<'a>(
    tensors: &'a [HfSafetensorsTensorInfo],
    name: &str,
) -> Result<&'a HfSafetensorsTensorInfo> {
    tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::Model(format!("MTP missing tensor '{name}'")))
}

/// Reads a linear weight+scale pair from the MTP tensor list by exact name.
fn read_mtp_linear_payload(
    model_dir: &Path,
    tensors: &[HfSafetensorsTensorInfo],
    reader: &ArtifactTensorReader,
    weight_name: &str,
    scale_name: &str,
    role: TensorRole,
) -> Result<ArtifactLinearPayload> {
    let weight_info = mtp_tensor_slice(tensors, weight_name)?;
    let mut weight_slice = ArtifactTensorSlice::from_hf_inventory(model_dir, weight_info);
    weight_slice.role = role.clone();
    let weight_payload = reader.read_slice(&weight_slice)?;

    let scale_payload = tensors
        .iter()
        .find(|tensor| tensor.name == scale_name)
        .map(|scale_info| {
            let mut slice = ArtifactTensorSlice::from_hf_inventory(model_dir, scale_info);
            slice.role = role.clone();
            reader.read_slice(&slice)
        })
        .transpose()?;

    ArtifactLinearPayload::from_weight_and_scale(role, weight_payload, scale_payload)
}

/// Reads a named norm vector from the MTP tensor list.
fn read_mtp_norm_vector(
    model_dir: &Path,
    tensors: &[HfSafetensorsTensorInfo],
    reader: &ArtifactTensorReader,
    name: &str,
    role: TensorRole,
) -> Result<Vec<f32>> {
    let info = mtp_tensor_slice(tensors, name)?;
    let mut slice = ArtifactTensorSlice::from_hf_inventory(model_dir, info);
    slice.role = role;
    decode_vector_f32(&reader.read_slice(&slice)?)
}

/// Loads one MTP layer from bound weights.
///
/// This function is called by `DeepSeekV4ArtifactModel::load_mtp` and uses the
/// existing `bind_*_from_hf` functions by constructing typed tensor-info structs
/// from the raw MTP safetensors entries.
#[allow(clippy::too_many_arguments)]
pub fn load_mtp_layer(
    model_dir: &Path,
    mtp_index: usize,
    execution_layer: usize,
    layer_tensors: &[HfSafetensorsTensorInfo],
    reader: &ArtifactTensorReader,
    attention_config: DeepSeekV4AttentionConfig,
    hc_config: HyperConnectionConfig,
    swiglu_limit: f32,
    num_experts_per_tok: usize,
    route_scale: f32,
) -> Result<DeepSeekV4MtpLayer> {
    let prefix = format!("mtp.{mtp_index}.");

    // Transformer-body norms.
    let attn_norm = read_mtp_norm_vector(
        model_dir,
        layer_tensors,
        reader,
        &format!("{prefix}attn_norm.weight"),
        TensorRole::LayerNorm,
    )?;
    let ffn_norm = read_mtp_norm_vector(
        model_dir,
        layer_tensors,
        reader,
        &format!("{prefix}ffn_norm.weight"),
        TensorRole::LayerNorm,
    )?;

    // Only stage zero consumes the concatenated target-layer hidden states.
    let (main_proj, main_norm) = if mtp_index == 0 {
        (
            Some(read_mtp_linear_payload(
                model_dir,
                layer_tensors,
                reader,
                &format!("{prefix}main_proj.weight"),
                &format!("{prefix}main_proj.scale"),
                TensorRole::SpeculativeProjection,
            )?),
            Some(read_mtp_norm_vector(
                model_dir,
                layer_tensors,
                reader,
                &format!("{prefix}main_norm.weight"),
                TensorRole::LayerNorm,
            )?),
        )
    } else {
        (None, None)
    };

    // Attention tensors.
    let attention_tensors: Vec<HfAttentionTensorInfo> = layer_tensors
        .iter()
        .filter_map(parse_mtp_attention_tensor)
        .collect();
    let attention_payload = with_deepseek_v4_attention_execution_policies(bind_attention_from_hf(
        model_dir,
        mtp_index,
        &attention_tensors,
        reader,
    )?);
    let compressed = DeepSeekV4CompressedAttentionPayload::bind_optional(
        mtp_index,
        attention_config,
        &attention_payload.auxiliary,
        reader,
    )?;
    let attention = DeepSeekV4Attention::new_with_compressed(
        execution_layer,
        attention_config,
        attention_payload,
        compressed,
    )?;

    // HyperConnection attention and FFN.
    let hc_tensors: Vec<HfHyperConnectionTensorInfo> = layer_tensors
        .iter()
        .filter_map(parse_mtp_hyper_connection_tensor)
        .collect();
    let hc_attention = bind_hyper_connection_from_hf(
        model_dir,
        mtp_index,
        HyperConnectionStage::Attention,
        &hc_tensors,
        reader,
        hc_config,
    )?;
    let hc_feed_forward = bind_hyper_connection_from_hf(
        model_dir,
        mtp_index,
        HyperConnectionStage::FeedForward,
        &hc_tensors,
        reader,
        hc_config,
    )?;

    // Router.
    let router_tensors: Vec<HfRouterTensorInfo> = layer_tensors
        .iter()
        .filter_map(parse_mtp_router_tensor)
        .collect();
    let router = bind_router_from_hf(model_dir, mtp_index, &router_tensors, reader)?;

    // Shared FFN.
    let shared_tensors: Vec<HfSharedExpertTensorInfo> = layer_tensors
        .iter()
        .filter_map(parse_mtp_shared_expert_tensor)
        .collect();
    let shared_ffn = with_deepseek_v4_swiglu_execution_policies(bind_shared_swiglu_ffn_from_hf(
        model_dir,
        mtp_index,
        &shared_tensors,
        reader,
        swiglu_limit,
    )?);

    // MTP layers use score-top-k routing (no hash table).
    let router_policy =
        ExpertRouterPolicy::sqrt_softplus_score_topk(num_experts_per_tok, route_scale);
    let routed_tensors = layer_tensors
        .iter()
        .filter_map(|tensor| parse_mtp_routed_expert_tensor(tensor, execution_layer))
        .collect::<Vec<_>>();
    let expert_source_catalog = Arc::new(ExpertSourceCatalog::from_hf_routed_expert_tensor_sets(
        model_dir,
        routed_tensors,
    )?);

    Ok(DeepSeekV4MtpLayer {
        mtp_index,
        execution_layer,
        transformer: DeepSeekV4Layer {
            layer: execution_layer,
            hc_config,
            attn_norm,
            ffn_norm,
            attention,
            hc_attention,
            hc_feed_forward,
            router,
            shared_ffn,
            router_policy,
        },
        main_proj,
        main_norm,
        expert_source_catalog,
    })
}

/// Loads prediction heads for the last MTP layer.
pub fn load_mtp_prediction_heads(
    model_dir: &Path,
    mtp_index: usize,
    layer_tensors: &[HfSafetensorsTensorInfo],
    reader: &ArtifactTensorReader,
    hc_tensors: &[HfHyperConnectionTensorInfo],
    hc_config: HyperConnectionConfig,
) -> Result<DeepSeekV4MtpPredictionHeads> {
    let prefix = format!("mtp.{mtp_index}.");

    // HC head.
    let hc_head = bind_hyper_connection_head_from_hf(model_dir, hc_tensors, reader, hc_config)?;

    // Final norm.
    let norm = read_mtp_norm_vector(
        model_dir,
        layer_tensors,
        reader,
        &format!("{prefix}norm.weight"),
        TensorRole::OutputNorm,
    )?;

    // Markov prediction head (low-rank: w1 then w2).
    let markov_w1 = read_mtp_linear_payload(
        model_dir,
        layer_tensors,
        reader,
        &format!("{prefix}markov_head.markov_w1.weight"),
        &format!("{prefix}markov_head.markov_w1.scale"),
        TensorRole::SpeculativeMarkovHead,
    )?;
    let markov_w2 = read_mtp_linear_payload(
        model_dir,
        layer_tensors,
        reader,
        &format!("{prefix}markov_head.markov_w2.weight"),
        &format!("{prefix}markov_head.markov_w2.scale"),
        TensorRole::SpeculativeMarkovHead,
    )?;

    // Confidence head.
    let confidence_proj = read_mtp_linear_payload(
        model_dir,
        layer_tensors,
        reader,
        &format!("{prefix}confidence_head.proj.weight"),
        &format!("{prefix}confidence_head.proj.scale"),
        TensorRole::SpeculativeConfidenceHead,
    )?;

    Ok(DeepSeekV4MtpPredictionHeads {
        hc_head,
        norm,
        markov_w1,
        markov_w2,
        confidence_proj,
    })
}

impl DeepSeekV4MtpModel {
    /// CPU reference forward pass stub.
    ///
    /// Processes a single hidden state through all MTP layers and returns
    /// predicted token IDs and confidence scores. The full attention and MoE
    /// computation requires KV-cache and expert-runtime infrastructure that is
    /// not yet wired into the MTP path; this stub outlines the computation
    /// and will be completed when the CUDA layer-execution infrastructure is
    /// reused for MTP.
    pub fn forward(
        &self,
        hidden_state: &[f32],
        token_id: u32,
        position: usize,
    ) -> Result<DeepSeekV4MtpForwardOutput> {
        let _ = (hidden_state, token_id, position);
        // TODO: Implement the full MTP forward pass:
        //
        // For each MTP layer:
        //   1. Concatenate the input hidden state with the token embedding.
        //   2. Project through `main_proj` -> `main_norm` (RMSNorm).
        //   3. Apply `attn_norm` (RMSNorm) -> MLA attention -> HC post.
        //   4. Apply `ffn_norm` (RMSNorm) -> MoE FFN (router + experts + shared)
        //      -> HC post.
        //   5. The output becomes the input for the next MTP layer.
        //
        // For the last layer:
        //   6. Apply `hc_head` to reduce HC state.
        //   7. Apply `norm` (final RMSNorm).
        //   8. Apply `markov_w1` -> activation -> `markov_w2` to get logits.
        //   9. Apply `confidence_proj` to get confidence scores.
        //  10. Return argmax token IDs and confidence scores.
        //
        // The attention step requires a KV-cache and position embeddings, which
        // are managed by the sequence execution state. The MoE step requires an
        // expert runtime and reader. These will be provided when the MTP forward
        // is integrated with the existing layer execution infrastructure.
        Err(Error::Model(
            "MTP forward pass is not yet implemented; requires KV-cache and expert runtime".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_policy::TensorClass;

    fn fake_tensor(name: &str) -> HfSafetensorsTensorInfo {
        HfSafetensorsTensorInfo {
            name: name.to_string(),
            shard: "model-00001-of-00048.safetensors".to_string(),
            dtype: "F32".to_string(),
            shape: vec![1],
            data_offset: 0,
            file_offset: 0,
            byte_size: 4,
            class: TensorClass::Unknown,
            role: TensorRole::Unknown,
        }
    }

    #[test]
    fn parses_mtp_attention_tensor_names() {
        let info = fake_tensor("mtp.0.attn.wq_a.weight");
        let parsed = parse_mtp_attention_tensor(&info).unwrap();
        assert_eq!(parsed.descriptor.layer, 0);
        assert_eq!(parsed.descriptor.kind, AttentionTensorKind::QueryA);
        assert_eq!(parsed.descriptor.part, ArtifactTensorPart::Weight);

        let info = fake_tensor("mtp.2.attn.wo_b.scale");
        let parsed = parse_mtp_attention_tensor(&info).unwrap();
        assert_eq!(parsed.descriptor.layer, 2);
        assert_eq!(parsed.descriptor.kind, AttentionTensorKind::OutputB);
        assert_eq!(parsed.descriptor.part, ArtifactTensorPart::Scale);

        let info = fake_tensor("mtp.1.attn.q_norm.weight");
        let parsed = parse_mtp_attention_tensor(&info).unwrap();
        assert_eq!(parsed.descriptor.layer, 1);
        assert_eq!(parsed.descriptor.kind, AttentionTensorKind::QueryNorm);

        let info = fake_tensor("mtp.0.attn.attn_sink.weight");
        let parsed = parse_mtp_attention_tensor(&info).unwrap();
        assert_eq!(parsed.descriptor.kind, AttentionTensorKind::AttentionSink);

        // Non-attention MTP tensor returns None.
        let info = fake_tensor("mtp.0.ffn.gate.weight");
        assert!(parse_mtp_attention_tensor(&info).is_none());
    }

    #[test]
    fn parses_mtp_router_tensor_names() {
        let info = fake_tensor("mtp.0.ffn.gate.weight");
        let parsed = parse_mtp_router_tensor(&info).unwrap();
        assert_eq!(parsed.descriptor.layer, 0);
        assert_eq!(parsed.descriptor.kind, RouterTensorKind::Weight);

        let info = fake_tensor("mtp.1.ffn.gate.bias");
        let parsed = parse_mtp_router_tensor(&info).unwrap();
        assert_eq!(parsed.descriptor.kind, RouterTensorKind::Bias);

        let info = fake_tensor("mtp.2.ffn.gate.tid2eid");
        let parsed = parse_mtp_router_tensor(&info).unwrap();
        assert_eq!(parsed.descriptor.kind, RouterTensorKind::HashTable);
    }

    #[test]
    fn parses_mtp_shared_expert_tensor_names() {
        let info = fake_tensor("mtp.0.ffn.shared_experts.w1.weight");
        let parsed = parse_mtp_shared_expert_tensor(&info).unwrap();
        assert_eq!(parsed.descriptor.layer, 0);
        assert_eq!(parsed.descriptor.matrix, RoutedExpertMatrix::Gate);
        assert_eq!(parsed.descriptor.part, RoutedExpertTensorPart::Weight);

        let info = fake_tensor("mtp.1.ffn.shared_experts.w3.scale");
        let parsed = parse_mtp_shared_expert_tensor(&info).unwrap();
        assert_eq!(parsed.descriptor.matrix, RoutedExpertMatrix::Up);
        assert_eq!(parsed.descriptor.part, RoutedExpertTensorPart::Scale);

        let info = fake_tensor("mtp.2.ffn.shared_experts.w2.weight");
        let parsed = parse_mtp_shared_expert_tensor(&info).unwrap();
        assert_eq!(parsed.descriptor.matrix, RoutedExpertMatrix::Down);
    }

    #[test]
    fn parses_mtp_hyper_connection_tensor_names() {
        let info = fake_tensor("mtp.0.hc_attn_fn");
        let parsed = parse_mtp_hyper_connection_tensor(&info).unwrap();
        assert_eq!(parsed.descriptor.layer, Some(0));
        assert_eq!(parsed.descriptor.stage, HyperConnectionStage::Attention);
        assert_eq!(parsed.descriptor.kind, HyperConnectionTensorKind::Function);

        let info = fake_tensor("mtp.1.hc_ffn_scale");
        let parsed = parse_mtp_hyper_connection_tensor(&info).unwrap();
        assert_eq!(parsed.descriptor.layer, Some(1));
        assert_eq!(parsed.descriptor.stage, HyperConnectionStage::FeedForward);
        assert_eq!(parsed.descriptor.kind, HyperConnectionTensorKind::Scale);

        // Head stage has layer = None.
        let info = fake_tensor("mtp.2.hc_head_base");
        let parsed = parse_mtp_hyper_connection_tensor(&info).unwrap();
        assert_eq!(parsed.descriptor.layer, None);
        assert_eq!(parsed.descriptor.stage, HyperConnectionStage::Head);
        assert_eq!(parsed.descriptor.kind, HyperConnectionTensorKind::Base);
    }

    #[test]
    fn mtp_forward_stub_returns_not_implemented() {
        let model = DeepSeekV4MtpModel {
            layers: Vec::new(),
            prediction_heads: None,
            config: DSparkConfig {
                block_size: 1,
                noise_token_id: None,
                target_layer_ids: Vec::new(),
                markov_rank: None,
            },
        };
        let result = model.forward(&[1.0], 0, 0);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not yet implemented")
        );
    }
}
