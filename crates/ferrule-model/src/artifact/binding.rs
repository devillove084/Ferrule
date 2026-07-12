//! Artifact binding helpers for semantic HF inventory tensors.
//!
//! Concrete model names are parsed in `ferrule-model::families`. This module
//! consumes the resulting semantic shared-expert/router descriptors and produces
//! runtime payloads: `SwiGluFfnPayload` and `RouterArtifactPayload`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::semantic::{
    ArtifactTensorPart, AttentionTensorKind, HyperConnectionStage, HyperConnectionTensorKind,
    RoutedExpertMatrix, RoutedExpertTensorPart, RouterTensorKind,
};
use crate::{
    HfAttentionTensorInfo, HfHyperConnectionTensorInfo, HfRouterTensorInfo,
    HfSharedExpertTensorInfo, TensorRole,
};
use ferrule_common::{Error, Result};

use crate::artifact::group::ArtifactGroupKind;
use crate::artifact::group::ArtifactObjectGroup;
use crate::artifact::linear::ArtifactLinearPayload;
use crate::artifact::tensor::{
    ArtifactDType, ArtifactTensorPayload, ArtifactTensorReader, ArtifactTensorSlice,
};
use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::{
    HyperConnectionConfig, HyperConnectionHeadWeights, HyperConnectionWeights,
};

#[derive(Debug, Clone, PartialEq)]
pub struct MlaAttentionArtifactPayload {
    pub layer: usize,
    pub query_a: ArtifactLinearPayload,
    pub query_b: ArtifactLinearPayload,
    pub key_value: ArtifactLinearPayload,
    pub output_a: ArtifactLinearPayload,
    pub output_b: ArtifactLinearPayload,
    pub query_norm: Vec<f32>,
    pub key_value_norm: Vec<f32>,
    pub attention_sink: Vec<f32>,
    /// Optional compressor/indexer tensors for compressed sparse attention. These
    /// remain as artifact slices until their execution path is wired; core MLA
    /// linears/norms are bound above.
    pub auxiliary: Vec<ArtifactTensorSlice>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerNormArtifactPayload {
    pub layer: usize,
    pub attention_norm: Option<Vec<f32>>,
    pub feed_forward_norm: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouterArtifactPayload {
    pub layer: usize,
    pub weight: ArtifactLinearPayload,
    pub bias: Option<Vec<f32>>,
    pub hash_table: Option<Vec<usize>>,
    pub hash_rows: usize,
    pub hash_cols: usize,
}

impl RouterArtifactPayload {
    pub fn logits(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.weight.reference_matvec(input)
    }

    pub fn hash_experts_for_token(&self, token_id: u32) -> Result<Option<Vec<usize>>> {
        let Some(table) = &self.hash_table else {
            return Ok(None);
        };
        let row = token_id as usize;
        if row >= self.hash_rows {
            return Err(Error::Model(format!(
                "router hash token id {row} exceeds table rows {} for layer {}",
                self.hash_rows, self.layer
            )));
        }
        let start = row * self.hash_cols;
        Ok(Some(table[start..start + self.hash_cols].to_vec()))
    }

    pub fn hash_expert_union_for_tokens(
        &self,
        token_ids: &[u32],
        per_token_limit: usize,
        max_experts: usize,
    ) -> Result<Option<Vec<usize>>> {
        let Some(table) = &self.hash_table else {
            return Ok(None);
        };
        if token_ids.is_empty() || per_token_limit == 0 || max_experts == 0 {
            return Ok(Some(Vec::new()));
        }

        let per_token_limit = per_token_limit.min(self.hash_cols);
        let mut seen = BTreeSet::new();
        let mut experts = Vec::with_capacity(max_experts.min(token_ids.len() * per_token_limit));
        for &token_id in token_ids {
            let row = token_id as usize;
            if row >= self.hash_rows {
                return Err(Error::Model(format!(
                    "router hash token id {row} exceeds table rows {} for layer {}",
                    self.hash_rows, self.layer
                )));
            }
            let start = row * self.hash_cols;
            for &expert in table[start..start + self.hash_cols]
                .iter()
                .take(per_token_limit)
            {
                if seen.insert(expert) {
                    experts.push(expert);
                    if experts.len() >= max_experts {
                        return Ok(Some(experts));
                    }
                }
            }
        }
        Ok(Some(experts))
    }
}

pub fn bind_shared_swiglu_ffn_from_artifact_group(
    group: &ArtifactObjectGroup,
    reader: &ArtifactTensorReader,
    swiglu_limit: f32,
) -> Result<SwiGluFfnPayload> {
    let layer = require_artifact_group_layer(group, ArtifactGroupKind::SharedExpert)?;
    let mut by_role = artifact_group_tensors_by_role(group);
    Ok(SwiGluFfnPayload {
        gate: bind_linear_from_artifact_slices(
            layer,
            "shared expert gate",
            TensorRole::SharedExpertGate,
            by_role.remove(&TensorRole::SharedExpertGate),
            reader,
        )?,
        up: bind_linear_from_artifact_slices(
            layer,
            "shared expert up",
            TensorRole::SharedExpertUp,
            by_role.remove(&TensorRole::SharedExpertUp),
            reader,
        )?,
        down: bind_linear_from_artifact_slices(
            layer,
            "shared expert down",
            TensorRole::SharedExpertDown,
            by_role.remove(&TensorRole::SharedExpertDown),
            reader,
        )?,
        swiglu_limit,
    })
}

pub fn bind_layer_norms_from_artifact_group(
    group: &ArtifactObjectGroup,
    reader: &ArtifactTensorReader,
) -> Result<LayerNormArtifactPayload> {
    let layer = require_artifact_group_layer(group, ArtifactGroupKind::LayerNorm)?;
    let mut by_role = artifact_group_tensors_by_role(group);
    Ok(LayerNormArtifactPayload {
        layer,
        attention_norm: bind_optional_vector_from_artifact_slices(
            layer,
            "attention norm",
            by_role.remove(&TensorRole::AttentionNorm),
            reader,
        )?,
        feed_forward_norm: bind_optional_vector_from_artifact_slices(
            layer,
            "feed-forward norm",
            by_role.remove(&TensorRole::FeedForwardNorm),
            reader,
        )?,
    })
}

pub fn bind_router_from_artifact_group(
    group: &ArtifactObjectGroup,
    reader: &ArtifactTensorReader,
) -> Result<RouterArtifactPayload> {
    let layer = require_artifact_group_layer(group, ArtifactGroupKind::Router)?;
    let mut by_role = artifact_group_tensors_by_role(group);
    let weight = bind_linear_from_artifact_slices(
        layer,
        "router weight",
        TensorRole::RouterLogits,
        by_role.remove(&TensorRole::RouterLogits),
        reader,
    )?;
    let bias = bind_optional_vector_from_artifact_slices(
        layer,
        "router bias",
        by_role.remove(&TensorRole::RouterBias),
        reader,
    )?;
    let (hash_table, hash_rows, hash_cols) = bind_optional_indices_from_artifact_slices(
        layer,
        "router hash table",
        by_role.remove(&TensorRole::HashRouterTable),
        reader,
    )?
    .map(|(values, rows, cols)| (Some(values), rows, cols))
    .unwrap_or((None, 0, 0));

    Ok(RouterArtifactPayload {
        layer,
        weight,
        bias,
        hash_table,
        hash_rows,
        hash_cols,
    })
}

pub fn bind_attention_from_artifact_group(
    group: &ArtifactObjectGroup,
    reader: &ArtifactTensorReader,
) -> Result<MlaAttentionArtifactPayload> {
    let layer = require_artifact_group_layer(group, ArtifactGroupKind::Attention)?;
    let mut by_role = artifact_group_tensors_by_role(group);
    let auxiliary = group
        .tensors
        .iter()
        .filter(|tensor| is_attention_auxiliary_role(&tensor.role))
        .cloned()
        .collect();

    Ok(MlaAttentionArtifactPayload {
        layer,
        query_a: bind_linear_from_artifact_slices(
            layer,
            "attention query A",
            TensorRole::AttentionLatentQueryA,
            by_role.remove(&TensorRole::AttentionLatentQueryA),
            reader,
        )?,
        query_b: bind_linear_from_artifact_slices(
            layer,
            "attention query B",
            TensorRole::AttentionLatentQueryB,
            by_role.remove(&TensorRole::AttentionLatentQueryB),
            reader,
        )?,
        key_value: bind_linear_from_artifact_slices(
            layer,
            "attention latent KV",
            TensorRole::AttentionLatentKv,
            by_role.remove(&TensorRole::AttentionLatentKv),
            reader,
        )?,
        output_a: bind_linear_from_artifact_slices(
            layer,
            "attention output A",
            TensorRole::AttentionLatentOutputA,
            by_role.remove(&TensorRole::AttentionLatentOutputA),
            reader,
        )?,
        output_b: bind_linear_from_artifact_slices(
            layer,
            "attention output B",
            TensorRole::AttentionLatentOutputB,
            by_role.remove(&TensorRole::AttentionLatentOutputB),
            reader,
        )?,
        query_norm: bind_vector_from_artifact_slices(
            layer,
            "attention query norm",
            by_role.remove(&TensorRole::AttentionQueryNorm),
            reader,
        )?,
        key_value_norm: bind_vector_from_artifact_slices(
            layer,
            "attention key/value norm",
            by_role.remove(&TensorRole::AttentionKeyValueNorm),
            reader,
        )?,
        attention_sink: bind_vector_from_artifact_slices(
            layer,
            "attention sink",
            by_role.remove(&TensorRole::AttentionSink),
            reader,
        )?,
        auxiliary,
    })
}

pub fn bind_hyper_connection_from_artifact_group(
    group: &ArtifactObjectGroup,
    reader: &ArtifactTensorReader,
    config: HyperConnectionConfig,
) -> Result<HyperConnectionWeights> {
    let layer = require_layer_hyper_connection_group(group)?;
    let weights = HyperConnectionWeights {
        function: read_hyper_connection_component_from_group(
            group,
            layer,
            "function",
            config.mix_hc() * config.hc_hidden_size(),
            Some(2),
            reader,
        )?,
        scale: read_hyper_connection_component_from_group(
            group,
            layer,
            "scale",
            3,
            Some(1),
            reader,
        )?,
        base: read_hyper_connection_component_from_group(
            group,
            layer,
            "base",
            config.mix_hc(),
            Some(1),
            reader,
        )?,
    };
    weights.validate(config)?;
    Ok(weights)
}

pub fn bind_hyper_connection_head_from_artifact_group(
    group: &ArtifactObjectGroup,
    reader: &ArtifactTensorReader,
    config: HyperConnectionConfig,
) -> Result<HyperConnectionHeadWeights> {
    if group.kind != ArtifactGroupKind::HyperConnectionHead {
        return Err(Error::Model(format!(
            "expected hyper-connection head artifact group, got {}",
            group.kind.as_str()
        )));
    }
    let weights = HyperConnectionHeadWeights {
        function: read_hyper_connection_component_from_group(
            group,
            0,
            "head function",
            config.hc_mult * config.hc_hidden_size(),
            Some(2),
            reader,
        )?,
        scale: read_hyper_connection_component_from_group(
            group,
            0,
            "head scale",
            1,
            Some(1),
            reader,
        )?,
        base: read_hyper_connection_component_from_group(
            group,
            0,
            "head base",
            config.hc_mult,
            Some(1),
            reader,
        )?,
    };
    weights.validate(config)?;
    Ok(weights)
}

pub fn bind_shared_swiglu_ffn_from_hf(
    model_dir: &Path,
    layer: usize,
    tensors: &[HfSharedExpertTensorInfo],
    reader: &ArtifactTensorReader,
    swiglu_limit: f32,
) -> Result<SwiGluFfnPayload> {
    let mut grouped = BTreeMap::<RoutedExpertMatrix, Vec<&HfSharedExpertTensorInfo>>::new();
    for tensor in tensors
        .iter()
        .filter(|tensor| tensor.descriptor.layer == layer)
    {
        grouped
            .entry(tensor.descriptor.matrix)
            .or_default()
            .push(tensor);
    }
    Ok(SwiGluFfnPayload {
        gate: bind_shared_linear(
            model_dir,
            layer,
            RoutedExpertMatrix::Gate,
            TensorRole::SharedExpertGate,
            grouped.remove(&RoutedExpertMatrix::Gate),
            reader,
        )?,
        up: bind_shared_linear(
            model_dir,
            layer,
            RoutedExpertMatrix::Up,
            TensorRole::SharedExpertUp,
            grouped.remove(&RoutedExpertMatrix::Up),
            reader,
        )?,
        down: bind_shared_linear(
            model_dir,
            layer,
            RoutedExpertMatrix::Down,
            TensorRole::SharedExpertDown,
            grouped.remove(&RoutedExpertMatrix::Down),
            reader,
        )?,
        swiglu_limit,
    })
}

pub fn bind_router_from_hf(
    model_dir: &Path,
    layer: usize,
    tensors: &[HfRouterTensorInfo],
    reader: &ArtifactTensorReader,
) -> Result<RouterArtifactPayload> {
    let mut weight = None;
    let mut bias = None;
    let mut hash = None;
    for tensor in tensors
        .iter()
        .filter(|tensor| tensor.descriptor.layer == layer)
    {
        match tensor.descriptor.kind {
            RouterTensorKind::Weight => set_once(&mut weight, tensor, layer, "router weight")?,
            RouterTensorKind::Bias => set_once(&mut bias, tensor, layer, "router bias")?,
            RouterTensorKind::HashTable => set_once(&mut hash, tensor, layer, "router hash table")?,
            RouterTensorKind::Other(_) => {}
        }
    }
    let weight = weight
        .ok_or_else(|| Error::Model(format!("missing router weight tensor for layer {layer}")))?;
    let weight_payload = reader.read_slice(&artifact_slice_from_router_info(
        model_dir,
        weight,
        TensorRole::RouterLogits,
    ))?;
    let weight = ArtifactLinearPayload::from_weight_and_scale(
        TensorRole::RouterLogits,
        weight_payload,
        None,
    )?;

    let bias = bias
        .map(|tensor| {
            let payload = reader.read_slice(&artifact_slice_from_router_info(
                model_dir,
                tensor,
                TensorRole::RouterBias,
            ))?;
            decode_vector_f32(&payload)
        })
        .transpose()?;
    let (hash_table, hash_rows, hash_cols) = hash
        .map(|tensor| {
            let payload = reader.read_slice(&artifact_slice_from_router_info(
                model_dir,
                tensor,
                TensorRole::HashRouterTable,
            ))?;
            let (rows, cols) = two_dim_shape(&payload.slice, "router hash table")?;
            let values = decode_indices_usize(&payload)?;
            Ok::<_, Error>((Some(values), rows, cols))
        })
        .transpose()?
        .unwrap_or((None, 0, 0));

    Ok(RouterArtifactPayload {
        layer,
        weight,
        bias,
        hash_table,
        hash_rows,
        hash_cols,
    })
}

pub fn bind_attention_from_hf(
    model_dir: &Path,
    layer: usize,
    tensors: &[HfAttentionTensorInfo],
    reader: &ArtifactTensorReader,
) -> Result<MlaAttentionArtifactPayload> {
    let mut grouped = BTreeMap::<AttentionTensorKind, Vec<&HfAttentionTensorInfo>>::new();
    let mut auxiliary = Vec::new();
    for tensor in tensors
        .iter()
        .filter(|tensor| tensor.descriptor.layer == layer)
    {
        match tensor.descriptor.kind {
            AttentionTensorKind::Compressor | AttentionTensorKind::Indexer => {
                auxiliary.push(artifact_slice_from_attention_info(
                    model_dir,
                    tensor,
                    attention_role_for_kind(tensor.descriptor.kind),
                ));
            }
            kind => {
                grouped.entry(kind).or_default().push(tensor);
            }
        }
    }

    Ok(MlaAttentionArtifactPayload {
        layer,
        query_a: bind_attention_linear(
            model_dir,
            layer,
            AttentionTensorKind::QueryA,
            TensorRole::AttentionLatentQueryA,
            grouped.remove(&AttentionTensorKind::QueryA),
            reader,
        )?,
        query_b: bind_attention_linear(
            model_dir,
            layer,
            AttentionTensorKind::QueryB,
            TensorRole::AttentionLatentQueryB,
            grouped.remove(&AttentionTensorKind::QueryB),
            reader,
        )?,
        key_value: bind_attention_linear(
            model_dir,
            layer,
            AttentionTensorKind::KeyValue,
            TensorRole::AttentionLatentKv,
            grouped.remove(&AttentionTensorKind::KeyValue),
            reader,
        )?,
        output_a: bind_attention_linear(
            model_dir,
            layer,
            AttentionTensorKind::OutputA,
            TensorRole::AttentionLatentOutputA,
            grouped.remove(&AttentionTensorKind::OutputA),
            reader,
        )?,
        output_b: bind_attention_linear(
            model_dir,
            layer,
            AttentionTensorKind::OutputB,
            TensorRole::AttentionLatentOutputB,
            grouped.remove(&AttentionTensorKind::OutputB),
            reader,
        )?,
        query_norm: bind_attention_vector(
            model_dir,
            layer,
            AttentionTensorKind::QueryNorm,
            TensorRole::AttentionQueryNorm,
            grouped.remove(&AttentionTensorKind::QueryNorm),
            reader,
        )?,
        key_value_norm: bind_attention_vector(
            model_dir,
            layer,
            AttentionTensorKind::KeyValueNorm,
            TensorRole::AttentionKeyValueNorm,
            grouped.remove(&AttentionTensorKind::KeyValueNorm),
            reader,
        )?,
        attention_sink: bind_attention_vector(
            model_dir,
            layer,
            AttentionTensorKind::AttentionSink,
            TensorRole::AttentionSink,
            grouped.remove(&AttentionTensorKind::AttentionSink),
            reader,
        )?,
        auxiliary,
    })
}

pub fn bind_hyper_connection_from_hf(
    model_dir: &Path,
    layer: usize,
    stage: HyperConnectionStage,
    tensors: &[HfHyperConnectionTensorInfo],
    reader: &ArtifactTensorReader,
    config: HyperConnectionConfig,
) -> Result<HyperConnectionWeights> {
    if stage == HyperConnectionStage::Head {
        return Err(Error::Model(
            "layer hyper-connection binding cannot use head stage; call bind_hyper_connection_head_from_hf"
                .into(),
        ));
    }

    let mut function = None;
    let mut scale = None;
    let mut base = None;
    for tensor in tensors
        .iter()
        .filter(|tensor| tensor.descriptor.layer == Some(layer) && tensor.descriptor.stage == stage)
    {
        match tensor.descriptor.kind {
            HyperConnectionTensorKind::Function => {
                set_once(&mut function, tensor, layer, "HC function")?
            }
            HyperConnectionTensorKind::Scale => set_once(&mut scale, tensor, layer, "HC scale")?,
            HyperConnectionTensorKind::Base => set_once(&mut base, tensor, layer, "HC base")?,
        }
    }

    let weights = HyperConnectionWeights {
        function: read_hyper_connection_tensor_f32(
            model_dir,
            layer,
            function,
            stage,
            HyperConnectionTensorKind::Function,
            reader,
        )?,
        scale: read_hyper_connection_tensor_f32(
            model_dir,
            layer,
            scale,
            stage,
            HyperConnectionTensorKind::Scale,
            reader,
        )?,
        base: read_hyper_connection_tensor_f32(
            model_dir,
            layer,
            base,
            stage,
            HyperConnectionTensorKind::Base,
            reader,
        )?,
    };
    weights.validate(config)?;
    Ok(weights)
}

pub fn bind_hyper_connection_head_from_hf(
    model_dir: &Path,
    tensors: &[HfHyperConnectionTensorInfo],
    reader: &ArtifactTensorReader,
    config: HyperConnectionConfig,
) -> Result<HyperConnectionHeadWeights> {
    let mut function = None;
    let mut scale = None;
    let mut base = None;
    for tensor in tensors.iter().filter(|tensor| {
        tensor.descriptor.layer.is_none() && tensor.descriptor.stage == HyperConnectionStage::Head
    }) {
        match tensor.descriptor.kind {
            HyperConnectionTensorKind::Function => {
                set_once(&mut function, tensor, 0, "HC head function")?
            }
            HyperConnectionTensorKind::Scale => set_once(&mut scale, tensor, 0, "HC head scale")?,
            HyperConnectionTensorKind::Base => set_once(&mut base, tensor, 0, "HC head base")?,
        }
    }

    let weights = HyperConnectionHeadWeights {
        function: read_hyper_connection_tensor_f32(
            model_dir,
            0,
            function,
            HyperConnectionStage::Head,
            HyperConnectionTensorKind::Function,
            reader,
        )?,
        scale: read_hyper_connection_tensor_f32(
            model_dir,
            0,
            scale,
            HyperConnectionStage::Head,
            HyperConnectionTensorKind::Scale,
            reader,
        )?,
        base: read_hyper_connection_tensor_f32(
            model_dir,
            0,
            base,
            HyperConnectionStage::Head,
            HyperConnectionTensorKind::Base,
            reader,
        )?,
    };
    weights.validate(config)?;
    Ok(weights)
}

fn artifact_group_tensors_by_role(
    group: &ArtifactObjectGroup,
) -> BTreeMap<TensorRole, Vec<&ArtifactTensorSlice>> {
    let mut by_role = BTreeMap::new();
    for tensor in &group.tensors {
        by_role
            .entry(tensor.role.clone())
            .or_insert_with(Vec::new)
            .push(tensor);
    }
    by_role
}

fn require_artifact_group_layer(
    group: &ArtifactObjectGroup,
    expected: ArtifactGroupKind,
) -> Result<usize> {
    if group.kind != expected {
        return Err(Error::Model(format!(
            "expected {} artifact group, got {}",
            expected.as_str(),
            group.kind.as_str()
        )));
    }
    group.layer.ok_or_else(|| {
        Error::Model(format!(
            "artifact group {} requires layer metadata",
            expected.as_str()
        ))
    })
}

fn require_layer_hyper_connection_group(group: &ArtifactObjectGroup) -> Result<usize> {
    if !matches!(
        group.kind,
        ArtifactGroupKind::HyperConnectionAttention | ArtifactGroupKind::HyperConnectionFeedForward
    ) {
        return Err(Error::Model(format!(
            "expected layer hyper-connection artifact group, got {}",
            group.kind.as_str()
        )));
    }
    group.layer.ok_or_else(|| {
        Error::Model(format!(
            "artifact group {} requires layer metadata",
            group.kind.as_str()
        ))
    })
}

fn bind_linear_from_artifact_slices(
    layer: usize,
    label: &str,
    role: TensorRole,
    tensors: Option<Vec<&ArtifactTensorSlice>>,
    reader: &ArtifactTensorReader,
) -> Result<ArtifactLinearPayload> {
    let tensors = tensors.ok_or_else(|| {
        Error::Model(format!(
            "missing {label} artifact tensors for layer {layer}"
        ))
    })?;
    let mut weight = None;
    let mut scale = None;
    for tensor in tensors {
        if is_artifact_linear_scale(tensor) {
            let scale_label = format!("{label} scale");
            set_once(&mut scale, tensor, layer, &scale_label)?;
        } else {
            let weight_label = format!("{label} weight");
            set_once(&mut weight, tensor, layer, &weight_label)?;
        }
    }
    let weight = weight.ok_or_else(|| {
        Error::Model(format!("missing {label} weight artifact for layer {layer}"))
    })?;
    let weight_payload = reader.read_slice(weight)?;
    let scale_payload = scale.map(|scale| reader.read_slice(scale)).transpose()?;
    ArtifactLinearPayload::from_weight_and_scale(role, weight_payload, scale_payload)
}

fn bind_vector_from_artifact_slices(
    layer: usize,
    label: &str,
    tensors: Option<Vec<&ArtifactTensorSlice>>,
    reader: &ArtifactTensorReader,
) -> Result<Vec<f32>> {
    let tensor = single_artifact_slice(layer, label, tensors)?;
    let payload = reader.read_slice(tensor)?;
    decode_vector_f32(&payload)
}

fn bind_optional_vector_from_artifact_slices(
    layer: usize,
    label: &str,
    tensors: Option<Vec<&ArtifactTensorSlice>>,
    reader: &ArtifactTensorReader,
) -> Result<Option<Vec<f32>>> {
    tensors
        .map(|tensors| bind_vector_from_artifact_slices(layer, label, Some(tensors), reader))
        .transpose()
}

fn bind_optional_indices_from_artifact_slices(
    layer: usize,
    label: &str,
    tensors: Option<Vec<&ArtifactTensorSlice>>,
    reader: &ArtifactTensorReader,
) -> Result<Option<(Vec<usize>, usize, usize)>> {
    let Some(tensors) = tensors else {
        return Ok(None);
    };
    let tensor = single_artifact_slice(layer, label, Some(tensors))?;
    let payload = reader.read_slice(tensor)?;
    let (rows, cols) = two_dim_shape(&payload.slice, label)?;
    let values = decode_indices_usize(&payload)?;
    Ok(Some((values, rows, cols)))
}

fn single_artifact_slice<'a>(
    layer: usize,
    label: &str,
    tensors: Option<Vec<&'a ArtifactTensorSlice>>,
) -> Result<&'a ArtifactTensorSlice> {
    let tensors = tensors.ok_or_else(|| {
        Error::Model(format!("missing {label} artifact tensor for layer {layer}"))
    })?;
    match tensors.as_slice() {
        [tensor] => Ok(*tensor),
        [] => Err(Error::Model(format!(
            "missing {label} artifact tensor for layer {layer}"
        ))),
        _ => Err(Error::Model(format!(
            "duplicate {label} artifact tensors for layer {layer}"
        ))),
    }
}

fn read_hyper_connection_component_from_group(
    group: &ArtifactObjectGroup,
    layer: usize,
    label: &str,
    expected_elements: usize,
    expected_rank: Option<usize>,
    reader: &ArtifactTensorReader,
) -> Result<Vec<f32>> {
    let mut found = None;
    for tensor in &group.tensors {
        if tensor.element_count()? == expected_elements
            && expected_rank.is_none_or(|rank| tensor.shape.len() == rank)
        {
            set_once(&mut found, tensor, layer, label)?;
        }
    }
    let tensor = found.ok_or_else(|| {
        Error::Model(format!(
            "missing hyper-connection {label} artifact for layer {layer} group {}",
            group.kind.as_str()
        ))
    })?;
    let payload = reader.read_slice(tensor)?;
    decode_tensor_f32(&payload)
}

fn is_artifact_linear_scale(tensor: &ArtifactTensorSlice) -> bool {
    matches!(tensor.dtype, ArtifactDType::F8E8M0)
}

fn is_attention_auxiliary_role(role: &TensorRole) -> bool {
    matches!(
        role,
        TensorRole::AttentionCompressor
            | TensorRole::AuxIndexer
            | TensorRole::AuxHiddenCompressor
            | TensorRole::AuxOutputHiddenCompressor
            | TensorRole::Auxiliary
    )
}

fn bind_attention_linear(
    model_dir: &Path,
    layer: usize,
    kind: AttentionTensorKind,
    role: TensorRole,
    tensors: Option<Vec<&HfAttentionTensorInfo>>,
    reader: &ArtifactTensorReader,
) -> Result<ArtifactLinearPayload> {
    let tensors = tensors.ok_or_else(|| {
        Error::Model(format!(
            "missing attention {:?} tensors for layer {layer}",
            kind
        ))
    })?;
    let mut weight = None;
    let mut scale = None;
    for tensor in tensors {
        match tensor.descriptor.part {
            ArtifactTensorPart::Weight => set_once(&mut weight, tensor, layer, "attention weight")?,
            ArtifactTensorPart::Scale => set_once(&mut scale, tensor, layer, "attention scale")?,
            ArtifactTensorPart::Other => {}
        }
    }
    let weight = weight.ok_or_else(|| {
        Error::Model(format!(
            "missing attention {:?} weight tensor for layer {layer}",
            kind
        ))
    })?;
    let weight_payload = reader.read_slice(&artifact_slice_from_attention_info(
        model_dir,
        weight,
        role.clone(),
    ))?;
    let scale_payload = scale
        .map(|scale| {
            reader.read_slice(&artifact_slice_from_attention_info(
                model_dir,
                scale,
                role.clone(),
            ))
        })
        .transpose()?;
    ArtifactLinearPayload::from_weight_and_scale(role, weight_payload, scale_payload)
}

fn bind_attention_vector(
    model_dir: &Path,
    layer: usize,
    kind: AttentionTensorKind,
    role: TensorRole,
    tensors: Option<Vec<&HfAttentionTensorInfo>>,
    reader: &ArtifactTensorReader,
) -> Result<Vec<f32>> {
    let tensors = tensors.ok_or_else(|| {
        Error::Model(format!(
            "missing attention {:?} tensor for layer {layer}",
            kind
        ))
    })?;
    let mut value = None;
    for tensor in tensors {
        match tensor.descriptor.part {
            ArtifactTensorPart::Weight | ArtifactTensorPart::Other => {
                set_once(&mut value, tensor, layer, "attention vector")?
            }
            ArtifactTensorPart::Scale => {}
        }
    }
    let value = value.ok_or_else(|| {
        Error::Model(format!(
            "missing attention {:?} vector payload for layer {layer}",
            kind
        ))
    })?;
    let payload = reader.read_slice(&artifact_slice_from_attention_info(model_dir, value, role))?;
    decode_vector_f32(&payload)
}

fn read_hyper_connection_tensor_f32(
    model_dir: &Path,
    layer: usize,
    tensor: Option<&HfHyperConnectionTensorInfo>,
    stage: HyperConnectionStage,
    kind: HyperConnectionTensorKind,
    reader: &ArtifactTensorReader,
) -> Result<Vec<f32>> {
    let tensor = tensor.ok_or_else(|| {
        let scope = if stage == HyperConnectionStage::Head {
            "head".to_string()
        } else {
            format!("layer {layer}")
        };
        Error::Model(format!(
            "missing hyper-connection {:?} {:?} tensor for {scope}",
            stage, kind
        ))
    })?;
    let payload = reader.read_slice(&artifact_slice_from_hyper_connection_info(
        model_dir,
        tensor,
        hyper_connection_role_for_stage(stage),
    ))?;
    decode_tensor_f32(&payload)
}

fn bind_shared_linear(
    model_dir: &Path,
    layer: usize,
    matrix: RoutedExpertMatrix,
    role: TensorRole,
    tensors: Option<Vec<&HfSharedExpertTensorInfo>>,
    reader: &ArtifactTensorReader,
) -> Result<ArtifactLinearPayload> {
    let tensors = tensors.ok_or_else(|| {
        Error::Model(format!(
            "missing shared expert {:?} tensors for layer {layer}",
            matrix
        ))
    })?;
    let mut weight = None;
    let mut scale = None;
    for tensor in tensors {
        match tensor.descriptor.part {
            RoutedExpertTensorPart::Weight => {
                set_once(&mut weight, tensor, layer, "shared weight")?
            }
            RoutedExpertTensorPart::Scale => set_once(&mut scale, tensor, layer, "shared scale")?,
            RoutedExpertTensorPart::Other(_) => {}
        }
    }
    let weight = weight.ok_or_else(|| {
        Error::Model(format!(
            "missing shared expert {:?} weight for layer {layer}",
            matrix
        ))
    })?;
    let weight_payload = reader.read_slice(&artifact_slice_from_shared_info(
        model_dir,
        weight,
        role.clone(),
    ))?;
    let scale_payload = scale
        .map(|scale| {
            reader.read_slice(&artifact_slice_from_shared_info(
                model_dir,
                scale,
                role.clone(),
            ))
        })
        .transpose()?;
    ArtifactLinearPayload::from_weight_and_scale(role, weight_payload, scale_payload)
}

fn set_once<'a, T>(
    slot: &mut Option<&'a T>,
    value: &'a T,
    layer: usize,
    label: &str,
) -> Result<()> {
    if slot.replace(value).is_some() {
        return Err(Error::Model(format!(
            "duplicate {label} tensor for layer {layer}"
        )));
    }
    Ok(())
}

fn artifact_slice_from_shared_info(
    model_dir: &Path,
    info: &HfSharedExpertTensorInfo,
    role: TensorRole,
) -> ArtifactTensorSlice {
    ArtifactTensorSlice {
        name: info.name.clone(),
        role,
        path: model_dir.join(&info.shard),
        offset: info.file_offset,
        bytes: info.byte_size,
        dtype: ArtifactDType::from_safetensors_dtype(&info.dtype),
        shape: info.shape.clone(),
    }
}

fn artifact_slice_from_router_info(
    model_dir: &Path,
    info: &HfRouterTensorInfo,
    role: TensorRole,
) -> ArtifactTensorSlice {
    ArtifactTensorSlice {
        name: info.name.clone(),
        role,
        path: model_dir.join(&info.shard),
        offset: info.file_offset,
        bytes: info.byte_size,
        dtype: ArtifactDType::from_safetensors_dtype(&info.dtype),
        shape: info.shape.clone(),
    }
}

fn artifact_slice_from_attention_info(
    model_dir: &Path,
    info: &HfAttentionTensorInfo,
    role: TensorRole,
) -> ArtifactTensorSlice {
    ArtifactTensorSlice {
        name: info.name.clone(),
        role,
        path: model_dir.join(&info.shard),
        offset: info.file_offset,
        bytes: info.byte_size,
        dtype: ArtifactDType::from_safetensors_dtype(&info.dtype),
        shape: info.shape.clone(),
    }
}

fn artifact_slice_from_hyper_connection_info(
    model_dir: &Path,
    info: &HfHyperConnectionTensorInfo,
    role: TensorRole,
) -> ArtifactTensorSlice {
    ArtifactTensorSlice {
        name: info.name.clone(),
        role,
        path: model_dir.join(&info.shard),
        offset: info.file_offset,
        bytes: info.byte_size,
        dtype: ArtifactDType::from_safetensors_dtype(&info.dtype),
        shape: info.shape.clone(),
    }
}

fn attention_role_for_kind(kind: AttentionTensorKind) -> TensorRole {
    match kind {
        AttentionTensorKind::QueryA => TensorRole::AttentionLatentQueryA,
        AttentionTensorKind::QueryB => TensorRole::AttentionLatentQueryB,
        AttentionTensorKind::KeyValue => TensorRole::AttentionLatentKv,
        AttentionTensorKind::OutputA => TensorRole::AttentionLatentOutputA,
        AttentionTensorKind::OutputB => TensorRole::AttentionLatentOutputB,
        AttentionTensorKind::QueryNorm => TensorRole::AttentionQueryNorm,
        AttentionTensorKind::KeyValueNorm => TensorRole::AttentionKeyValueNorm,
        AttentionTensorKind::AttentionSink => TensorRole::AttentionSink,
        AttentionTensorKind::Compressor => TensorRole::AttentionCompressor,
        AttentionTensorKind::Indexer => TensorRole::AuxIndexer,
    }
}

fn hyper_connection_role_for_stage(stage: HyperConnectionStage) -> TensorRole {
    match stage {
        HyperConnectionStage::Attention | HyperConnectionStage::FeedForward => {
            TensorRole::AuxHiddenCompressor
        }
        HyperConnectionStage::Head => TensorRole::AuxOutputHiddenCompressor,
    }
}

fn two_dim_shape(slice: &ArtifactTensorSlice, label: &str) -> Result<(usize, usize)> {
    match slice.shape.as_slice() {
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(Error::Model(format!(
            "{label} '{}' expects 2D shape, got {:?}",
            slice.name, slice.shape
        ))),
    }
}

fn decode_vector_f32(payload: &ArtifactTensorPayload) -> Result<Vec<f32>> {
    if payload.slice.shape.len() != 1 {
        return Err(Error::Model(format!(
            "artifact vector '{}' expects 1D shape, got {:?}",
            payload.slice.name, payload.slice.shape
        )));
    }
    decode_tensor_f32(payload)
}

fn decode_tensor_f32(payload: &ArtifactTensorPayload) -> Result<Vec<f32>> {
    let expected = payload.slice.element_count()?;
    match payload.slice.dtype {
        ArtifactDType::F32 => {
            if payload.bytes.len() != expected * 4 {
                return Err(Error::Model(format!(
                    "F32 tensor '{}' byte length mismatch",
                    payload.slice.name
                )));
            }
            Ok(payload
                .bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        ArtifactDType::Bf16 => {
            if payload.bytes.len() != expected * 2 {
                return Err(Error::Model(format!(
                    "BF16 tensor '{}' byte length mismatch",
                    payload.slice.name
                )));
            }
            Ok(payload
                .bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
                    f32::from_bits(bits << 16)
                })
                .collect())
        }
        _ => Err(Error::Model(format!(
            "artifact tensor '{}' has unsupported dtype {}",
            payload.slice.name,
            payload.slice.dtype.as_str()
        ))),
    }
}

fn decode_indices_usize(payload: &ArtifactTensorPayload) -> Result<Vec<usize>> {
    let expected = payload.slice.element_count()?;
    match payload.slice.dtype {
        ArtifactDType::I32 => {
            if payload.bytes.len() != expected * 4 {
                return Err(Error::Model(format!(
                    "I32 indices '{}' byte length mismatch",
                    payload.slice.name
                )));
            }
            payload
                .bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let value = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    usize::try_from(value).map_err(|_| {
                        Error::Model(format!(
                            "negative router index {value} in '{}'",
                            payload.slice.name
                        ))
                    })
                })
                .collect()
        }
        ArtifactDType::I64 => {
            if payload.bytes.len() != expected * 8 {
                return Err(Error::Model(format!(
                    "I64 indices '{}' byte length mismatch",
                    payload.slice.name
                )));
            }
            payload
                .bytes
                .chunks_exact(8)
                .map(|chunk| {
                    let value = i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]);
                    usize::try_from(value).map_err(|_| {
                        Error::Model(format!(
                            "negative router index {value} in '{}'",
                            payload.slice.name
                        ))
                    })
                })
                .collect()
        }
        _ => Err(Error::Model(format!(
            "router index tensor '{}' has unsupported dtype {}",
            payload.slice.name,
            payload.slice.dtype.as_str()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::semantic::{
        ArtifactTensorPart, AttentionTensorKind, AttentionTensorRef, HyperConnectionStage,
        HyperConnectionTensorKind, HyperConnectionTensorRef, RouterTensorRef,
        SharedExpertTensorRef,
    };

    use super::*;
    use crate::artifact::linear::ArtifactLinearFormat;

    #[test]
    fn router_artifact_payload_returns_hash_row() {
        let router = RouterArtifactPayload {
            layer: 0,
            weight: synthetic_router_weight(),
            bias: None,
            hash_table: Some(vec![3, 4, 5, 6]),
            hash_rows: 2,
            hash_cols: 2,
        };
        assert_eq!(router.hash_experts_for_token(1).unwrap(), Some(vec![5, 6]));
    }

    #[test]
    fn router_hash_expert_union_dedupes_in_prompt_order_and_caps() {
        let router = RouterArtifactPayload {
            layer: 0,
            weight: synthetic_router_weight(),
            bias: None,
            hash_table: Some(vec![3, 4, 5, 4, 7, 8, 9, 3, 10]),
            hash_rows: 3,
            hash_cols: 3,
        };

        assert_eq!(
            router
                .hash_expert_union_for_tokens(&[0, 1, 2], 2, 4)
                .unwrap(),
            Some(vec![3, 4, 7, 9])
        );
        assert_eq!(
            router.hash_expert_union_for_tokens(&[0, 1], 8, 8).unwrap(),
            Some(vec![3, 4, 5, 7, 8])
        );
    }

    #[test]
    fn binds_shared_ffn_from_synthetic_hf_infos() {
        let dir = unique_temp_dir("ferrule-shared-binding");
        std::fs::create_dir_all(&dir).unwrap();
        write_linear_pair(&dir, "gate", 1.0);
        write_linear_pair(&dir, "up", 2.0);
        write_linear_pair(&dir, "down", 3.0);
        let tensors = vec![
            shared_info(
                "gate.weight",
                "gate.bin",
                RoutedExpertMatrix::Gate,
                RoutedExpertTensorPart::Weight,
                "F32",
                vec![1, 2],
                0,
                8,
            ),
            shared_info(
                "up.weight",
                "up.bin",
                RoutedExpertMatrix::Up,
                RoutedExpertTensorPart::Weight,
                "F32",
                vec![1, 2],
                0,
                8,
            ),
            shared_info(
                "down.weight",
                "down.bin",
                RoutedExpertMatrix::Down,
                RoutedExpertTensorPart::Weight,
                "F32",
                vec![2, 1],
                0,
                8,
            ),
        ];
        let ffn = bind_shared_swiglu_ffn_from_hf(
            &dir,
            0,
            &tensors,
            &ArtifactTensorReader::new(1024),
            0.0,
        )
        .unwrap();
        assert_eq!(
            ffn.gate.format,
            ArtifactLinearFormat::F32 {
                out_features: 1,
                in_features: 2
            }
        );
        assert_eq!(
            ffn.down.format,
            ArtifactLinearFormat::F32 {
                out_features: 2,
                in_features: 1
            }
        );
        let group = ArtifactObjectGroup {
            kind: ArtifactGroupKind::SharedExpert,
            layer: Some(0),
            tensors: tensors
                .iter()
                .map(|tensor| {
                    artifact_slice_from_shared_info(
                        &dir,
                        tensor,
                        shared_role_for_matrix(tensor.descriptor.matrix),
                    )
                })
                .collect(),
        };
        let from_group = bind_shared_swiglu_ffn_from_artifact_group(
            &group,
            &ArtifactTensorReader::new(1024),
            0.0,
        )
        .unwrap();
        assert_eq!(from_group.gate.format, ffn.gate.format);
        assert_eq!(from_group.down.format, ffn.down.format);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn binds_router_weight_bias_and_hash_from_synthetic_hf_infos() {
        let dir = unique_temp_dir("ferrule-router-binding");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("weight.bin"), f32_bytes(&[1.0, 0.0, 0.0, 1.0])).unwrap();
        std::fs::write(dir.join("bias.bin"), f32_bytes(&[0.5, -0.5])).unwrap();
        std::fs::write(dir.join("hash.bin"), i64_bytes(&[1, 0, 0, 1])).unwrap();
        let tensors = vec![
            router_info(
                "router.weight",
                "weight.bin",
                RouterTensorKind::Weight,
                "F32",
                vec![2, 2],
                0,
                16,
            ),
            router_info(
                "router.bias",
                "bias.bin",
                RouterTensorKind::Bias,
                "F32",
                vec![2],
                0,
                8,
            ),
            router_info(
                "router.tid2eid",
                "hash.bin",
                RouterTensorKind::HashTable,
                "I64",
                vec![2, 2],
                0,
                32,
            ),
        ];
        let router =
            bind_router_from_hf(&dir, 0, &tensors, &ArtifactTensorReader::new(1024)).unwrap();
        assert_eq!(router.logits(&[2.0, 3.0]).unwrap(), vec![2.0, 3.0]);
        assert_eq!(router.bias.as_deref(), Some(&[0.5, -0.5][..]));
        assert_eq!(router.hash_experts_for_token(0).unwrap(), Some(vec![1, 0]));
        let group = ArtifactObjectGroup {
            kind: ArtifactGroupKind::Router,
            layer: Some(0),
            tensors: tensors
                .iter()
                .map(|tensor| {
                    artifact_slice_from_router_info(
                        &dir,
                        tensor,
                        router_role_for_kind(tensor.descriptor.kind.clone()),
                    )
                })
                .collect(),
        };
        let from_group =
            bind_router_from_artifact_group(&group, &ArtifactTensorReader::new(1024)).unwrap();
        assert_eq!(from_group.logits(&[2.0, 3.0]).unwrap(), vec![2.0, 3.0]);
        assert_eq!(from_group.bias.as_deref(), router.bias.as_deref());
        assert_eq!(
            from_group.hash_experts_for_token(0).unwrap(),
            Some(vec![1, 0])
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn binds_attention_from_synthetic_hf_infos() {
        let dir = unique_temp_dir("ferrule-attention-binding");
        std::fs::create_dir_all(&dir).unwrap();
        write_f32_file(&dir, "wq_a.bin", &[1.0, 0.0, 0.0, 1.0]);
        write_f32_file(&dir, "wq_b.bin", &[1.0, 0.0, 0.0, 1.0]);
        write_f32_file(&dir, "wkv.bin", &[1.0, 0.0, 0.0, 1.0]);
        write_f32_file(&dir, "wo_a.bin", &[1.0, 0.0, 0.0, 1.0]);
        write_f32_file(&dir, "wo_b.bin", &[1.0, 0.0, 0.0, 1.0]);
        write_f32_file(&dir, "q_norm.bin", &[1.0, 2.0]);
        write_f32_file(&dir, "kv_norm.bin", &[3.0, 4.0]);
        write_f32_file(&dir, "sink.bin", &[0.1, 0.2]);
        write_f32_file(&dir, "aux.bin", &[42.0]);
        let tensors = vec![
            attention_info(
                "wq_a.weight",
                "wq_a.bin",
                AttentionTensorKind::QueryA,
                ArtifactTensorPart::Weight,
                "F32",
                vec![2, 2],
                16,
            ),
            attention_info(
                "wq_b.weight",
                "wq_b.bin",
                AttentionTensorKind::QueryB,
                ArtifactTensorPart::Weight,
                "F32",
                vec![2, 2],
                16,
            ),
            attention_info(
                "wkv.weight",
                "wkv.bin",
                AttentionTensorKind::KeyValue,
                ArtifactTensorPart::Weight,
                "F32",
                vec![2, 2],
                16,
            ),
            attention_info(
                "wo_a.weight",
                "wo_a.bin",
                AttentionTensorKind::OutputA,
                ArtifactTensorPart::Weight,
                "F32",
                vec![2, 2],
                16,
            ),
            attention_info(
                "wo_b.weight",
                "wo_b.bin",
                AttentionTensorKind::OutputB,
                ArtifactTensorPart::Weight,
                "F32",
                vec![2, 2],
                16,
            ),
            attention_info(
                "q_norm.weight",
                "q_norm.bin",
                AttentionTensorKind::QueryNorm,
                ArtifactTensorPart::Weight,
                "F32",
                vec![2],
                8,
            ),
            attention_info(
                "kv_norm.weight",
                "kv_norm.bin",
                AttentionTensorKind::KeyValueNorm,
                ArtifactTensorPart::Weight,
                "F32",
                vec![2],
                8,
            ),
            attention_info(
                "attn_sink",
                "sink.bin",
                AttentionTensorKind::AttentionSink,
                ArtifactTensorPart::Other,
                "F32",
                vec![2],
                8,
            ),
            attention_info(
                "compressor.wkv.weight",
                "aux.bin",
                AttentionTensorKind::Compressor,
                ArtifactTensorPart::Other,
                "F32",
                vec![1],
                4,
            ),
        ];

        let attention =
            bind_attention_from_hf(&dir, 0, &tensors, &ArtifactTensorReader::new(1024)).unwrap();
        assert_eq!(attention.layer, 0);
        assert_eq!(attention.query_a.format.in_features(), 2);
        assert_eq!(attention.output_b.format.out_features(), 2);
        assert_eq!(attention.query_norm, vec![1.0, 2.0]);
        assert_eq!(attention.key_value_norm, vec![3.0, 4.0]);
        assert_eq!(attention.attention_sink, vec![0.1, 0.2]);
        assert_eq!(attention.auxiliary.len(), 1);
        assert_eq!(attention.auxiliary[0].role, TensorRole::AttentionCompressor);
        let group = ArtifactObjectGroup {
            kind: ArtifactGroupKind::Attention,
            layer: Some(0),
            tensors: tensors
                .iter()
                .map(|tensor| {
                    artifact_slice_from_attention_info(
                        &dir,
                        tensor,
                        attention_role_for_kind(tensor.descriptor.kind),
                    )
                })
                .collect(),
        };
        let from_group =
            bind_attention_from_artifact_group(&group, &ArtifactTensorReader::new(1024)).unwrap();
        assert_eq!(from_group, attention);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn binds_hyper_connection_from_synthetic_hf_infos() {
        let dir = unique_temp_dir("ferrule-hc-binding");
        std::fs::create_dir_all(&dir).unwrap();
        let config = HyperConnectionConfig {
            hc_mult: 2,
            hidden_size: 2,
            sinkhorn_iters: 4,
            eps: 1e-6,
            norm_eps: 1e-6,
        };
        let layer_fn = (0..32).map(|value| value as f32).collect::<Vec<_>>();
        let head_fn = (0..8).map(|value| value as f32).collect::<Vec<_>>();
        write_f32_file(&dir, "hc_attn_fn.bin", &layer_fn);
        write_f32_file(&dir, "hc_attn_scale.bin", &[1.0, 2.0, 3.0]);
        write_f32_file(&dir, "hc_attn_base.bin", &[0.0; 8]);
        write_f32_file(&dir, "hc_head_fn.bin", &head_fn);
        write_f32_file(&dir, "hc_head_scale.bin", &[1.0]);
        write_f32_file(&dir, "hc_head_base.bin", &[0.0, 1.0]);
        let tensors = vec![
            hyper_connection_info(
                "layers.0.hc_attn_fn",
                "hc_attn_fn.bin",
                Some(0),
                HyperConnectionStage::Attention,
                HyperConnectionTensorKind::Function,
                vec![8, 4],
                128,
            ),
            hyper_connection_info(
                "layers.0.hc_attn_scale",
                "hc_attn_scale.bin",
                Some(0),
                HyperConnectionStage::Attention,
                HyperConnectionTensorKind::Scale,
                vec![3],
                12,
            ),
            hyper_connection_info(
                "layers.0.hc_attn_base",
                "hc_attn_base.bin",
                Some(0),
                HyperConnectionStage::Attention,
                HyperConnectionTensorKind::Base,
                vec![8],
                32,
            ),
            hyper_connection_info(
                "hc_head_fn",
                "hc_head_fn.bin",
                None,
                HyperConnectionStage::Head,
                HyperConnectionTensorKind::Function,
                vec![2, 4],
                32,
            ),
            hyper_connection_info(
                "hc_head_scale",
                "hc_head_scale.bin",
                None,
                HyperConnectionStage::Head,
                HyperConnectionTensorKind::Scale,
                vec![1],
                4,
            ),
            hyper_connection_info(
                "hc_head_base",
                "hc_head_base.bin",
                None,
                HyperConnectionStage::Head,
                HyperConnectionTensorKind::Base,
                vec![2],
                8,
            ),
        ];

        let weights = bind_hyper_connection_from_hf(
            &dir,
            0,
            HyperConnectionStage::Attention,
            &tensors,
            &ArtifactTensorReader::new(1024),
            config,
        )
        .unwrap();
        assert_eq!(weights.function, layer_fn);
        assert_eq!(weights.scale, vec![1.0, 2.0, 3.0]);
        assert_eq!(weights.base.len(), 8);

        let head = bind_hyper_connection_head_from_hf(
            &dir,
            &tensors,
            &ArtifactTensorReader::new(1024),
            config,
        )
        .unwrap();
        assert_eq!(head.function, head_fn);
        assert_eq!(head.scale, vec![1.0]);
        assert_eq!(head.base, vec![0.0, 1.0]);

        let layer_group = ArtifactObjectGroup {
            kind: ArtifactGroupKind::HyperConnectionAttention,
            layer: Some(0),
            tensors: tensors
                .iter()
                .filter(|tensor| tensor.descriptor.layer == Some(0))
                .map(|tensor| {
                    artifact_slice_from_hyper_connection_info(
                        &dir,
                        tensor,
                        hyper_connection_role_for_stage(tensor.descriptor.stage),
                    )
                })
                .collect(),
        };
        let from_group = bind_hyper_connection_from_artifact_group(
            &layer_group,
            &ArtifactTensorReader::new(1024),
            config,
        )
        .unwrap();
        assert_eq!(from_group.function, weights.function);
        assert_eq!(from_group.scale, weights.scale);
        assert_eq!(from_group.base, weights.base);

        let head_group = ArtifactObjectGroup {
            kind: ArtifactGroupKind::HyperConnectionHead,
            layer: None,
            tensors: tensors
                .iter()
                .filter(|tensor| tensor.descriptor.layer.is_none())
                .map(|tensor| {
                    artifact_slice_from_hyper_connection_info(
                        &dir,
                        tensor,
                        hyper_connection_role_for_stage(tensor.descriptor.stage),
                    )
                })
                .collect(),
        };
        let head_from_group = bind_hyper_connection_head_from_artifact_group(
            &head_group,
            &ArtifactTensorReader::new(1024),
            config,
        )
        .unwrap();
        assert_eq!(head_from_group.function, head.function);
        assert_eq!(head_from_group.scale, head.scale);
        assert_eq!(head_from_group.base, head.base);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn shared_role_for_matrix(matrix: RoutedExpertMatrix) -> TensorRole {
        match matrix {
            RoutedExpertMatrix::Gate => TensorRole::SharedExpertGate,
            RoutedExpertMatrix::Up => TensorRole::SharedExpertUp,
            RoutedExpertMatrix::Down => TensorRole::SharedExpertDown,
        }
    }

    fn router_role_for_kind(kind: RouterTensorKind) -> TensorRole {
        match kind {
            RouterTensorKind::Weight => TensorRole::RouterLogits,
            RouterTensorKind::Bias => TensorRole::RouterBias,
            RouterTensorKind::HashTable => TensorRole::HashRouterTable,
            RouterTensorKind::Other(_) => TensorRole::Unknown,
        }
    }

    fn synthetic_router_weight() -> ArtifactLinearPayload {
        let payload = ArtifactTensorPayload {
            slice: ArtifactTensorSlice {
                name: "router.weight".into(),
                role: TensorRole::RouterLogits,
                path: PathBuf::from("synthetic"),
                offset: 0,
                bytes: 16,
                dtype: ArtifactDType::F32,
                shape: vec![2, 2],
            },
            bytes: f32_bytes(&[1.0, 0.0, 0.0, 1.0]),
        };
        ArtifactLinearPayload::from_weight_and_scale(TensorRole::RouterLogits, payload, None)
            .unwrap()
    }

    fn write_linear_pair(dir: &Path, name: &str, value: f32) {
        std::fs::write(dir.join(format!("{name}.bin")), f32_bytes(&[value, 0.0])).unwrap();
    }

    fn write_f32_file(dir: &Path, name: &str, values: &[f32]) {
        std::fs::write(dir.join(name), f32_bytes(values)).unwrap();
    }

    fn shared_info(
        name: &str,
        shard: &str,
        matrix: RoutedExpertMatrix,
        part: RoutedExpertTensorPart,
        dtype: &str,
        shape: Vec<usize>,
        file_offset: u64,
        byte_size: u64,
    ) -> HfSharedExpertTensorInfo {
        HfSharedExpertTensorInfo {
            descriptor: SharedExpertTensorRef {
                layer: 0,
                matrix,
                part,
            },
            name: name.into(),
            shard: shard.into(),
            dtype: dtype.into(),
            shape,
            data_offset: file_offset,
            file_offset,
            byte_size,
        }
    }

    fn router_info(
        name: &str,
        shard: &str,
        kind: RouterTensorKind,
        dtype: &str,
        shape: Vec<usize>,
        file_offset: u64,
        byte_size: u64,
    ) -> HfRouterTensorInfo {
        HfRouterTensorInfo {
            descriptor: RouterTensorRef { layer: 0, kind },
            name: name.into(),
            shard: shard.into(),
            dtype: dtype.into(),
            shape,
            data_offset: file_offset,
            file_offset,
            byte_size,
        }
    }

    fn attention_info(
        name: &str,
        shard: &str,
        kind: AttentionTensorKind,
        part: ArtifactTensorPart,
        dtype: &str,
        shape: Vec<usize>,
        byte_size: u64,
    ) -> HfAttentionTensorInfo {
        HfAttentionTensorInfo {
            descriptor: AttentionTensorRef {
                layer: 0,
                kind,
                part,
            },
            name: name.into(),
            shard: shard.into(),
            dtype: dtype.into(),
            shape,
            data_offset: 0,
            file_offset: 0,
            byte_size,
        }
    }

    fn hyper_connection_info(
        name: &str,
        shard: &str,
        layer: Option<usize>,
        stage: HyperConnectionStage,
        kind: HyperConnectionTensorKind,
        shape: Vec<usize>,
        byte_size: u64,
    ) -> HfHyperConnectionTensorInfo {
        HfHyperConnectionTensorInfo {
            descriptor: HyperConnectionTensorRef { layer, stage, kind },
            name: name.into(),
            shard: shard.into(),
            dtype: "F32".into(),
            shape,
            data_offset: 0,
            file_offset: 0,
            byte_size,
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn i64_bytes(values: &[i64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nonce}"))
    }
}
