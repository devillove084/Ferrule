//! Source binding helpers for semantic HF inventory tensors.
//!
//! Concrete model names are parsed in `ferrule-model::families`. This module
//! consumes the resulting semantic shared-expert/router descriptors and produces
//! runtime payloads: `SwiGluFfnPayload` and `RouterSourcePayload`.

use std::collections::BTreeMap;
use std::path::Path;

use ferrule_core::{Error, Result};
use ferrule_model::families::{
    AttentionTensorKind, HyperConnectionStage, HyperConnectionTensorKind, RoutedExpertMatrix,
    RoutedExpertTensorPart, RouterTensorKind, SourceTensorPart,
};
use ferrule_model::{
    HfAttentionTensorInfo, HfHyperConnectionTensorInfo, HfRouterTensorInfo,
    HfSharedExpertTensorInfo, TensorRole,
};

use crate::ffn::SwiGluFfnPayload;
use crate::hyper_connection::{
    HyperConnectionConfig, HyperConnectionHeadWeights, HyperConnectionWeights,
};
use crate::source_linear::SourceLinearPayload;
use crate::source_tensor::{
    SourceDType, SourceTensorPayload, SourceTensorReader, SourceTensorSlice,
};

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionSourcePayload {
    pub layer: usize,
    pub query_a: SourceLinearPayload,
    pub query_b: SourceLinearPayload,
    pub key_value: SourceLinearPayload,
    pub output_a: SourceLinearPayload,
    pub output_b: SourceLinearPayload,
    pub query_norm: Vec<f32>,
    pub key_value_norm: Vec<f32>,
    pub attention_sink: Vec<f32>,
    /// Optional compressor/indexer tensors for compressed sparse attention. These
    /// remain as source slices until their execution path is wired; core MLA
    /// linears/norms are bound above.
    pub auxiliary: Vec<SourceTensorSlice>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouterSourcePayload {
    pub layer: usize,
    pub weight: SourceLinearPayload,
    pub bias: Option<Vec<f32>>,
    pub hash_table: Option<Vec<usize>>,
    pub hash_rows: usize,
    pub hash_cols: usize,
}

impl RouterSourcePayload {
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
}

pub fn bind_shared_swiglu_ffn_from_hf(
    model_dir: &Path,
    layer: usize,
    tensors: &[HfSharedExpertTensorInfo],
    reader: &SourceTensorReader,
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
    reader: &SourceTensorReader,
) -> Result<RouterSourcePayload> {
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
    let weight_payload = reader.read_slice(&source_slice_from_router_info(
        model_dir,
        weight,
        TensorRole::RouterLogits,
    ))?;
    let weight =
        SourceLinearPayload::from_weight_and_scale(TensorRole::RouterLogits, weight_payload, None)?;

    let bias = bias
        .map(|tensor| {
            let payload = reader.read_slice(&source_slice_from_router_info(
                model_dir,
                tensor,
                TensorRole::RouterBias,
            ))?;
            decode_vector_f32(&payload)
        })
        .transpose()?;
    let (hash_table, hash_rows, hash_cols) = hash
        .map(|tensor| {
            let payload = reader.read_slice(&source_slice_from_router_info(
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

    Ok(RouterSourcePayload {
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
    reader: &SourceTensorReader,
) -> Result<AttentionSourcePayload> {
    let mut grouped = BTreeMap::<AttentionTensorKind, Vec<&HfAttentionTensorInfo>>::new();
    let mut auxiliary = Vec::new();
    for tensor in tensors
        .iter()
        .filter(|tensor| tensor.descriptor.layer == layer)
    {
        match tensor.descriptor.kind {
            AttentionTensorKind::Compressor | AttentionTensorKind::Indexer => {
                auxiliary.push(source_slice_from_attention_info(
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

    Ok(AttentionSourcePayload {
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
    reader: &SourceTensorReader,
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
    reader: &SourceTensorReader,
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

fn bind_attention_linear(
    model_dir: &Path,
    layer: usize,
    kind: AttentionTensorKind,
    role: TensorRole,
    tensors: Option<Vec<&HfAttentionTensorInfo>>,
    reader: &SourceTensorReader,
) -> Result<SourceLinearPayload> {
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
            SourceTensorPart::Weight => set_once(&mut weight, tensor, layer, "attention weight")?,
            SourceTensorPart::Scale => set_once(&mut scale, tensor, layer, "attention scale")?,
            SourceTensorPart::Other => {}
        }
    }
    let weight = weight.ok_or_else(|| {
        Error::Model(format!(
            "missing attention {:?} weight tensor for layer {layer}",
            kind
        ))
    })?;
    let weight_payload = reader.read_slice(&source_slice_from_attention_info(
        model_dir,
        weight,
        role.clone(),
    ))?;
    let scale_payload = scale
        .map(|scale| {
            reader.read_slice(&source_slice_from_attention_info(
                model_dir,
                scale,
                role.clone(),
            ))
        })
        .transpose()?;
    SourceLinearPayload::from_weight_and_scale(role, weight_payload, scale_payload)
}

fn bind_attention_vector(
    model_dir: &Path,
    layer: usize,
    kind: AttentionTensorKind,
    role: TensorRole,
    tensors: Option<Vec<&HfAttentionTensorInfo>>,
    reader: &SourceTensorReader,
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
            SourceTensorPart::Weight | SourceTensorPart::Other => {
                set_once(&mut value, tensor, layer, "attention vector")?
            }
            SourceTensorPart::Scale => {}
        }
    }
    let value = value.ok_or_else(|| {
        Error::Model(format!(
            "missing attention {:?} vector payload for layer {layer}",
            kind
        ))
    })?;
    let payload = reader.read_slice(&source_slice_from_attention_info(model_dir, value, role))?;
    decode_vector_f32(&payload)
}

fn read_hyper_connection_tensor_f32(
    model_dir: &Path,
    layer: usize,
    tensor: Option<&HfHyperConnectionTensorInfo>,
    stage: HyperConnectionStage,
    kind: HyperConnectionTensorKind,
    reader: &SourceTensorReader,
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
    let payload = reader.read_slice(&source_slice_from_hyper_connection_info(
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
    reader: &SourceTensorReader,
) -> Result<SourceLinearPayload> {
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
    let weight_payload = reader.read_slice(&source_slice_from_shared_info(
        model_dir,
        weight,
        role.clone(),
    ))?;
    let scale_payload = scale
        .map(|scale| {
            reader.read_slice(&source_slice_from_shared_info(
                model_dir,
                scale,
                role.clone(),
            ))
        })
        .transpose()?;
    SourceLinearPayload::from_weight_and_scale(role, weight_payload, scale_payload)
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

fn source_slice_from_shared_info(
    model_dir: &Path,
    info: &HfSharedExpertTensorInfo,
    role: TensorRole,
) -> SourceTensorSlice {
    SourceTensorSlice {
        name: info.name.clone(),
        role,
        path: model_dir.join(&info.shard),
        offset: info.file_offset,
        bytes: info.byte_size,
        dtype: SourceDType::from_safetensors_dtype(&info.dtype),
        shape: info.shape.clone(),
    }
}

fn source_slice_from_router_info(
    model_dir: &Path,
    info: &HfRouterTensorInfo,
    role: TensorRole,
) -> SourceTensorSlice {
    SourceTensorSlice {
        name: info.name.clone(),
        role,
        path: model_dir.join(&info.shard),
        offset: info.file_offset,
        bytes: info.byte_size,
        dtype: SourceDType::from_safetensors_dtype(&info.dtype),
        shape: info.shape.clone(),
    }
}

fn source_slice_from_attention_info(
    model_dir: &Path,
    info: &HfAttentionTensorInfo,
    role: TensorRole,
) -> SourceTensorSlice {
    SourceTensorSlice {
        name: info.name.clone(),
        role,
        path: model_dir.join(&info.shard),
        offset: info.file_offset,
        bytes: info.byte_size,
        dtype: SourceDType::from_safetensors_dtype(&info.dtype),
        shape: info.shape.clone(),
    }
}

fn source_slice_from_hyper_connection_info(
    model_dir: &Path,
    info: &HfHyperConnectionTensorInfo,
    role: TensorRole,
) -> SourceTensorSlice {
    SourceTensorSlice {
        name: info.name.clone(),
        role,
        path: model_dir.join(&info.shard),
        offset: info.file_offset,
        bytes: info.byte_size,
        dtype: SourceDType::from_safetensors_dtype(&info.dtype),
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

fn two_dim_shape(slice: &SourceTensorSlice, label: &str) -> Result<(usize, usize)> {
    match slice.shape.as_slice() {
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(Error::Model(format!(
            "{label} '{}' expects 2D shape, got {:?}",
            slice.name, slice.shape
        ))),
    }
}

fn decode_vector_f32(payload: &SourceTensorPayload) -> Result<Vec<f32>> {
    if payload.slice.shape.len() != 1 {
        return Err(Error::Model(format!(
            "source vector '{}' expects 1D shape, got {:?}",
            payload.slice.name, payload.slice.shape
        )));
    }
    decode_tensor_f32(payload)
}

fn decode_tensor_f32(payload: &SourceTensorPayload) -> Result<Vec<f32>> {
    let expected = payload.slice.element_count()?;
    match payload.slice.dtype {
        SourceDType::F32 => {
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
        SourceDType::Bf16 => {
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
            "source tensor '{}' has unsupported dtype {}",
            payload.slice.name,
            payload.slice.dtype.as_str()
        ))),
    }
}

fn decode_indices_usize(payload: &SourceTensorPayload) -> Result<Vec<usize>> {
    let expected = payload.slice.element_count()?;
    match payload.slice.dtype {
        SourceDType::I32 => {
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
        SourceDType::I64 => {
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

    use ferrule_model::families::{
        AttentionTensorKind, AttentionTensorRef, HyperConnectionStage, HyperConnectionTensorKind,
        HyperConnectionTensorRef, RouterTensorRef, SharedExpertTensorRef, SourceTensorPart,
    };

    use super::*;
    use crate::source_linear::SourceLinearFormat;

    #[test]
    fn router_source_payload_returns_hash_row() {
        let router = RouterSourcePayload {
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
        let ffn =
            bind_shared_swiglu_ffn_from_hf(&dir, 0, &tensors, &SourceTensorReader::new(1024), 0.0)
                .unwrap();
        assert_eq!(
            ffn.gate.format,
            SourceLinearFormat::F32 {
                out_features: 1,
                in_features: 2
            }
        );
        assert_eq!(
            ffn.down.format,
            SourceLinearFormat::F32 {
                out_features: 2,
                in_features: 1
            }
        );
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
            bind_router_from_hf(&dir, 0, &tensors, &SourceTensorReader::new(1024)).unwrap();
        assert_eq!(router.logits(&[2.0, 3.0]).unwrap(), vec![2.0, 3.0]);
        assert_eq!(router.bias.as_deref(), Some(&[0.5, -0.5][..]));
        assert_eq!(router.hash_experts_for_token(0).unwrap(), Some(vec![1, 0]));
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
                SourceTensorPart::Weight,
                "F32",
                vec![2, 2],
                16,
            ),
            attention_info(
                "wq_b.weight",
                "wq_b.bin",
                AttentionTensorKind::QueryB,
                SourceTensorPart::Weight,
                "F32",
                vec![2, 2],
                16,
            ),
            attention_info(
                "wkv.weight",
                "wkv.bin",
                AttentionTensorKind::KeyValue,
                SourceTensorPart::Weight,
                "F32",
                vec![2, 2],
                16,
            ),
            attention_info(
                "wo_a.weight",
                "wo_a.bin",
                AttentionTensorKind::OutputA,
                SourceTensorPart::Weight,
                "F32",
                vec![2, 2],
                16,
            ),
            attention_info(
                "wo_b.weight",
                "wo_b.bin",
                AttentionTensorKind::OutputB,
                SourceTensorPart::Weight,
                "F32",
                vec![2, 2],
                16,
            ),
            attention_info(
                "q_norm.weight",
                "q_norm.bin",
                AttentionTensorKind::QueryNorm,
                SourceTensorPart::Weight,
                "F32",
                vec![2],
                8,
            ),
            attention_info(
                "kv_norm.weight",
                "kv_norm.bin",
                AttentionTensorKind::KeyValueNorm,
                SourceTensorPart::Weight,
                "F32",
                vec![2],
                8,
            ),
            attention_info(
                "attn_sink",
                "sink.bin",
                AttentionTensorKind::AttentionSink,
                SourceTensorPart::Other,
                "F32",
                vec![2],
                8,
            ),
            attention_info(
                "compressor.wkv.weight",
                "aux.bin",
                AttentionTensorKind::Compressor,
                SourceTensorPart::Other,
                "F32",
                vec![1],
                4,
            ),
        ];

        let attention =
            bind_attention_from_hf(&dir, 0, &tensors, &SourceTensorReader::new(1024)).unwrap();
        assert_eq!(attention.layer, 0);
        assert_eq!(attention.query_a.format.in_features(), 2);
        assert_eq!(attention.output_b.format.out_features(), 2);
        assert_eq!(attention.query_norm, vec![1.0, 2.0]);
        assert_eq!(attention.key_value_norm, vec![3.0, 4.0]);
        assert_eq!(attention.attention_sink, vec![0.1, 0.2]);
        assert_eq!(attention.auxiliary.len(), 1);
        assert_eq!(attention.auxiliary[0].role, TensorRole::AttentionCompressor);
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
            &SourceTensorReader::new(1024),
            config,
        )
        .unwrap();
        assert_eq!(weights.function, layer_fn);
        assert_eq!(weights.scale, vec![1.0, 2.0, 3.0]);
        assert_eq!(weights.base.len(), 8);

        let head = bind_hyper_connection_head_from_hf(
            &dir,
            &tensors,
            &SourceTensorReader::new(1024),
            config,
        )
        .unwrap();
        assert_eq!(head.function, head_fn);
        assert_eq!(head.scale, vec![1.0]);
        assert_eq!(head.base, vec![0.0, 1.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn synthetic_router_weight() -> SourceLinearPayload {
        let payload = SourceTensorPayload {
            slice: SourceTensorSlice {
                name: "router.weight".into(),
                role: TensorRole::RouterLogits,
                path: PathBuf::from("synthetic"),
                offset: 0,
                bytes: 16,
                dtype: SourceDType::F32,
                shape: vec![2, 2],
            },
            bytes: f32_bytes(&[1.0, 0.0, 0.0, 1.0]),
        };
        SourceLinearPayload::from_weight_and_scale(TensorRole::RouterLogits, payload, None).unwrap()
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
        part: SourceTensorPart,
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
