//! Family-neutral helpers for loading floating-point checkpoint tensors.
//!
//! Model families supply their own error context and tensor roles while this
//! module owns inventory lookup, bounded reads, and F32/BF16 decoding.

use std::path::Path;

use crate::checkpoint::tensor::{
    CheckpointDType, CheckpointTensorPayload, CheckpointTensorReader, CheckpointTensorSlice,
};
use crate::{HfSafetensorsInventory, HfSafetensorsTensorInfo, TensorRole};
use ferrule_common::{Error, Result};

pub(crate) fn unique_top_level_slice(
    model_dir: &Path,
    inventory: &HfSafetensorsInventory,
    role: TensorRole,
    error_context: &str,
) -> Result<CheckpointTensorSlice> {
    let tensors = inventory
        .tensors
        .iter()
        .filter(|tensor| tensor.role == role)
        .collect::<Vec<_>>();
    match tensors.as_slice() {
        [tensor] => Ok(CheckpointTensorSlice::from_hf_inventory(model_dir, tensor)),
        [] => Err(Error::Model(format!(
            "{error_context} missing top-level tensor role {role}"
        ))),
        _ => Err(Error::Model(format!(
            "{error_context} expected exactly one top-level tensor role {role}, got {}",
            tensors.len()
        ))),
    }
}

pub(crate) fn read_named_vector_f32(
    model_dir: &Path,
    inventory: &HfSafetensorsInventory,
    reader: &CheckpointTensorReader,
    name: &str,
    role: TensorRole,
    error_context: &str,
) -> Result<Vec<f32>> {
    let tensor = inventory_tensor(inventory, name, error_context)?;
    let mut slice = CheckpointTensorSlice::from_hf_inventory(model_dir, tensor);
    slice.role = role;
    decode_vector_f32(&reader.read_slice(&slice)?, error_context)
}

pub(crate) fn inventory_tensor<'a>(
    inventory: &'a HfSafetensorsInventory,
    name: &str,
    error_context: &str,
) -> Result<&'a HfSafetensorsTensorInfo> {
    inventory
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::Model(format!("{error_context} missing tensor '{name}'")))
}

pub(crate) fn decode_vector_f32(
    payload: &CheckpointTensorPayload,
    error_context: &str,
) -> Result<Vec<f32>> {
    if payload.slice.shape.len() != 1 {
        return Err(Error::Model(format!(
            "{error_context} checkpoint vector '{}' expects 1D shape, got {:?}",
            payload.slice.name, payload.slice.shape
        )));
    }
    decode_tensor_f32(payload, error_context)
}

pub(crate) fn decode_tensor_f32(
    payload: &CheckpointTensorPayload,
    error_context: &str,
) -> Result<Vec<f32>> {
    let expected = payload.slice.element_count()?;
    match payload.slice.dtype {
        CheckpointDType::F32 => {
            if payload.bytes.len() != expected * 4 {
                return Err(Error::Model(format!(
                    "{error_context} F32 tensor '{}' byte length mismatch",
                    payload.slice.name
                )));
            }
            Ok(payload
                .bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        CheckpointDType::Bf16 => {
            if payload.bytes.len() != expected * 2 {
                return Err(Error::Model(format!(
                    "{error_context} BF16 tensor '{}' byte length mismatch",
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
            "{error_context} checkpoint tensor '{}' has unsupported vector dtype {}",
            payload.slice.name,
            payload.slice.dtype.as_str()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::checkpoint::inventory::HfSafetensorsInventory;
    use crate::{ModelFamily, TensorClass};

    #[test]
    fn decodes_f32_and_bf16_payloads() {
        let f32_payload = payload(
            CheckpointDType::F32,
            vec![2],
            [1.5f32, -2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        );
        assert_eq!(
            decode_vector_f32(&f32_payload, "test model").unwrap(),
            vec![1.5, -2.0]
        );

        let bf16_payload = payload(
            CheckpointDType::Bf16,
            vec![2],
            [0x3fc0u16, 0xc000]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        );
        assert_eq!(
            decode_tensor_f32(&bf16_payload, "test model").unwrap(),
            vec![1.5, -2.0]
        );
    }

    #[test]
    fn decode_errors_retain_caller_context() {
        let matrix = payload(
            CheckpointDType::F32,
            vec![1, 1],
            1.0f32.to_le_bytes().to_vec(),
        );
        let error = decode_vector_f32(&matrix, "DenseDecoder").unwrap_err();
        assert_eq!(
            error.to_string(),
            "Model: DenseDecoder checkpoint vector 'test.weight' expects 1D shape, got [1, 1]"
        );

        let unsupported = payload(CheckpointDType::I8, vec![1], vec![0]);
        let error = decode_tensor_f32(&unsupported, "DenseDecoder").unwrap_err();
        assert_eq!(
            error.to_string(),
            "Model: DenseDecoder checkpoint tensor 'test.weight' has unsupported vector dtype I8"
        );
    }

    #[test]
    fn reads_named_vector_and_selects_unique_top_level_slice() {
        let dir = unique_temp_dir("ferrule-common-checkpoint");
        std::fs::create_dir_all(&dir).unwrap();
        let shard = "weights.bin";
        let values = [0.25f32, -0.5];
        std::fs::write(
            dir.join(shard),
            values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let vector = tensor_info(
            "layers.0.input_norm.weight",
            shard,
            TensorRole::LayerNorm,
            vec![2],
            8,
        );
        let output = tensor_info(
            "model.norm.weight",
            shard,
            TensorRole::OutputNorm,
            vec![2],
            8,
        );
        let inventory = inventory(vec![vector, output]);
        let reader = CheckpointTensorReader::new(8);

        assert_eq!(
            read_named_vector_f32(
                &dir,
                &inventory,
                &reader,
                "layers.0.input_norm.weight",
                TensorRole::LayerNorm,
                "DenseDecoder",
            )
            .unwrap(),
            vec![0.25, -0.5]
        );
        let slice =
            unique_top_level_slice(&dir, &inventory, TensorRole::OutputNorm, "DenseDecoder")
                .unwrap();
        assert_eq!(slice.name, "model.norm.weight");
        assert_eq!(slice.path, dir.join(shard));

        let error = inventory_tensor(&inventory, "missing.weight", "DenseDecoder").unwrap_err();
        assert_eq!(
            error.to_string(),
            "Model: DenseDecoder missing tensor 'missing.weight'"
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unique_top_level_slice_rejects_duplicates() {
        let tensors = vec![
            tensor_info(
                "model.norm.weight",
                "a.bin",
                TensorRole::OutputNorm,
                vec![1],
                4,
            ),
            tensor_info(
                "model.final_norm.weight",
                "b.bin",
                TensorRole::OutputNorm,
                vec![1],
                4,
            ),
        ];
        let error = unique_top_level_slice(
            Path::new("model"),
            &inventory(tensors),
            TensorRole::OutputNorm,
            "DenseDecoder",
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Model: DenseDecoder expected exactly one top-level tensor role output_norm, got 2"
        );
    }

    fn payload(
        dtype: CheckpointDType,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    ) -> CheckpointTensorPayload {
        CheckpointTensorPayload {
            slice: CheckpointTensorSlice {
                name: "test.weight".into(),
                role: TensorRole::Unknown,
                path: PathBuf::from("test.bin"),
                offset: 0,
                bytes: bytes.len() as u64,
                dtype,
                shape,
            },
            bytes,
        }
    }

    fn tensor_info(
        name: &str,
        shard: &str,
        role: TensorRole,
        shape: Vec<usize>,
        byte_size: u64,
    ) -> HfSafetensorsTensorInfo {
        HfSafetensorsTensorInfo {
            name: name.into(),
            shard: shard.into(),
            dtype: "F32".into(),
            shape,
            data_offset: 0,
            file_offset: 0,
            byte_size,
            class: TensorClass::Unknown,
            role,
        }
    }

    fn inventory(tensors: Vec<HfSafetensorsTensorInfo>) -> HfSafetensorsInventory {
        HfSafetensorsInventory {
            family: ModelFamily::DeepSeekV4,
            total_size: None,
            shard_count: 0,
            tensor_count: tensors.len(),
            tensors,
            dtype_counts: Vec::new(),
            class_counts: Vec::new(),
            role_counts: Vec::new(),
            shard_summaries: Vec::new(),
            index_only_tensors: Vec::new(),
            header_only_tensors: Vec::new(),
        }
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()))
    }
}
