//! DeepSeek-V4 DSpark attachment stored under the checkpoint's `mtp.*` namespace.
//!
//! This is not an ordinary autoregressive MTP stack. Stage zero projects and
//! normalizes the concatenated target hidden taps, while a separate five-row
//! `[anchor, noise × 4]` embedding block flows through three transformer stages.
//! Each stage attends to its committed target-context KV plus the complete
//! non-causal proposal block; proposal-block KV is ephemeral.
//!
//! The last checkpoint stage additionally carries prediction heads:
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
    with_deepseek_v4_linear_execution_policy, with_deepseek_v4_swiglu_execution_policies,
};
use super::helpers::decode_vector_f32;
use super::layer::DeepSeekV4Layer;

/// One MTP layer's bound weights.
///
/// Structurally mirrors `DeepSeekV4Layer`. Stage zero additionally owns the
/// projection/norm for concatenated target hidden taps; proposal token embeddings
/// remain a separate DSpark backbone input.
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

/// Complete DSpark attachment with all checkpoint `mtp.*` stages.
pub struct DeepSeekV4MtpModel {
    pub layers: Vec<DeepSeekV4MtpLayer>,
    pub prediction_heads: Option<DeepSeekV4MtpPredictionHeads>,
    pub config: DSparkConfig,
}

/// Frozen checkpoint-native DSpark row and commit contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4DsparkProtocol {
    /// Number of proposed draft tokens produced by one native block.
    pub gamma: usize,
    /// Backbone query rows: one anchor followed by `gamma - 1` noise rows.
    pub draft_backbone_rows: usize,
    /// Target rows: one carried anchor followed by `gamma` draft tokens.
    pub target_verify_rows: usize,
    /// Maximum externally committed tokens after full acceptance and bonus.
    pub max_external_commit_tokens: usize,
    pub noise_token_id: u32,
    pub target_layer_ids: Vec<usize>,
}

impl TryFrom<&DSparkConfig> for DeepSeekV4DsparkProtocol {
    type Error = Error;

    fn try_from(config: &DSparkConfig) -> Result<Self> {
        if config.block_size == 0 {
            return Err(Error::Model(
                "DeepSeek-V4 DSpark gamma must be greater than zero".into(),
            ));
        }
        let noise_token_id = config.noise_token_id.ok_or_else(|| {
            Error::Model("DeepSeek-V4 DSpark protocol requires a noise token id".into())
        })?;
        if config.target_layer_ids.is_empty() {
            return Err(Error::Model(
                "DeepSeek-V4 DSpark protocol requires target hidden-state layers".into(),
            ));
        }
        if config
            .target_layer_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::Model(
                "DeepSeek-V4 DSpark target layers must be strictly increasing".into(),
            ));
        }
        let target_verify_rows = config
            .block_size
            .checked_add(1)
            .ok_or_else(|| Error::Model("DeepSeek-V4 DSpark target width overflow".into()))?;
        Ok(Self {
            gamma: config.block_size,
            draft_backbone_rows: config.block_size,
            target_verify_rows,
            max_external_commit_tokens: target_verify_rows,
            noise_token_id,
            target_layer_ids: config.target_layer_ids.clone(),
        })
    }
}

impl DeepSeekV4DsparkProtocol {
    /// Builds the native semi-autoregressive backbone input. The first row is the
    /// carried target token; remaining rows use the checkpoint noise token.
    pub fn draft_input_ids(&self, anchor_token_id: u32) -> Vec<u32> {
        let mut input = vec![self.noise_token_id; self.draft_backbone_rows];
        input[0] = anchor_token_id;
        input
    }

    /// One carried anchor is verified in addition to every admitted draft token.
    pub fn target_rows_for_drafts(&self, proposed_draft_tokens: usize) -> Result<usize> {
        if proposed_draft_tokens > self.gamma {
            return Err(Error::Model(format!(
                "DeepSeek-V4 DSpark requested {proposed_draft_tokens} drafts above checkpoint gamma {}",
                self.gamma
            )));
        }
        proposed_draft_tokens
            .checked_add(1)
            .ok_or_else(|| Error::Model("DeepSeek-V4 DSpark target width overflow".into()))
    }
}

/// Output of a single DSpark proposal block.
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
        .map(with_deepseek_v4_linear_execution_policy)
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

fn round_to_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    if (bits & 0x7f80_0000) == 0x7f80_0000 {
        return value;
    }
    let rounding_bias = 0x7fffu32 + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding_bias) & 0xffff_0000)
}

impl DeepSeekV4MtpModel {
    pub fn protocol(&self) -> Result<DeepSeekV4DsparkProtocol> {
        DeepSeekV4DsparkProtocol::try_from(&self.config)
    }

    /// CPU oracle for the official stage-zero `main_norm(main_proj(taps))`
    /// boundary. Both the projection result and normalized output are rounded to
    /// BF16 exactly where the checkpoint Python returns BF16 tensors.
    pub fn stage_zero_main_reference(&self, target_taps: &[f32], rows: usize) -> Result<Vec<f32>> {
        let stage_zero = self
            .layers
            .first()
            .ok_or_else(|| Error::Model("DeepSeek-V4 DSpark stage zero is missing".into()))?;
        let projection = stage_zero.main_proj.as_ref().ok_or_else(|| {
            Error::Model("DeepSeek-V4 DSpark stage-zero main projection is missing".into())
        })?;
        let norm = stage_zero.main_norm.as_deref().ok_or_else(|| {
            Error::Model("DeepSeek-V4 DSpark stage-zero main norm is missing".into())
        })?;
        let input_size = projection.format.in_features();
        let output_size = projection.format.out_features();
        let expected = rows.checked_mul(input_size).ok_or_else(|| {
            Error::Model("DeepSeek-V4 DSpark stage-zero reference input size overflow".into())
        })?;
        if rows == 0 || target_taps.len() != expected || norm.len() != output_size {
            return Err(Error::Model(format!(
                "DeepSeek-V4 DSpark stage-zero reference shape mismatch: rows={rows} taps={}/{} norm={}/{}",
                target_taps.len(),
                expected,
                norm.len(),
                output_size
            )));
        }
        let mut output = Vec::with_capacity(rows * output_size);
        for row in 0..rows {
            let start = row * input_size;
            let mut projected =
                projection.reference_matvec(&target_taps[start..start + input_size])?;
            projected
                .iter_mut()
                .for_each(|value| *value = round_to_bf16(*value));
            let mean_square = projected
                .iter()
                .fold(0.0f32, |sum, value| value.mul_add(*value, sum))
                / output_size as f32;
            let inverse_rms = (mean_square + stage_zero.transformer.hc_config.norm_eps)
                .sqrt()
                .recip();
            output.extend(
                projected
                    .iter()
                    .zip(norm)
                    .map(|(value, weight)| round_to_bf16(value * inverse_rms * weight)),
            );
        }
        Ok(output)
    }

    /// Legacy placeholder retained until the CUDA DSpark block entry point is
    /// published. A single hidden row cannot represent the exact DSpark input:
    /// execution requires committed target taps, per-stage context KV, and the
    /// complete anchor/noise proposal block.
    pub fn forward(
        &self,
        hidden_state: &[f32],
        token_id: u32,
        position: usize,
    ) -> Result<DeepSeekV4MtpForwardOutput> {
        let _ = (hidden_state, token_id, position);
        Err(Error::Model(
            "generic single-row MTP forward is invalid for DSpark semi-autoregressive block execution"
                .into(),
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
    fn dspark_protocol_freezes_native_block_and_target_layout() {
        let model = DeepSeekV4MtpModel {
            layers: Vec::new(),
            prediction_heads: None,
            config: DSparkConfig {
                block_size: 5,
                noise_token_id: Some(128_799),
                target_layer_ids: vec![40, 41, 42],
                markov_rank: Some(256),
            },
        };

        let protocol = model.protocol().unwrap();
        assert_eq!(protocol.gamma, 5);
        assert_eq!(protocol.draft_backbone_rows, 5);
        assert_eq!(protocol.target_verify_rows, 6);
        assert_eq!(protocol.max_external_commit_tokens, 6);
        assert_eq!(
            protocol.draft_input_ids(17),
            vec![17, 128_799, 128_799, 128_799, 128_799]
        );
        assert_eq!(protocol.target_rows_for_drafts(0).unwrap(), 1);
        assert_eq!(protocol.target_rows_for_drafts(5).unwrap(), 6);
        assert!(protocol.target_rows_for_drafts(6).is_err());
        assert!(model.forward(&[1.0], 0, 0).is_err());
    }
}
