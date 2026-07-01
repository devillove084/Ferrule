//! Generic source tensor payloads and bounded local reads.
//!
//! Model-family adapters classify tensor names into semantic roles in
//! `ferrule-model`. Runtime code should consume these source tensor descriptors
//! without matching on model-specific names. This module is the generic bridge
//! from HF safetensors inventory byte ranges to small reference payloads and,
//! later, GPU/streaming tensor handles.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use ferrule_core::{Error, Result};
use ferrule_model::{HfSafetensorsTensorInfo, TensorRole};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceDType {
    F32,
    Bf16,
    F8E4M3,
    F8E8M0,
    I8,
    I32,
    I64,
    Unknown(String),
}

impl SourceDType {
    pub fn from_safetensors_dtype(dtype: &str) -> Self {
        match dtype {
            "F32" => Self::F32,
            "BF16" => Self::Bf16,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E8M0" => Self::F8E8M0,
            "I8" => Self::I8,
            "I32" => Self::I32,
            "I64" => Self::I64,
            other => Self::Unknown(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::F32 => "F32",
            Self::Bf16 => "BF16",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E8M0 => "F8_E8M0",
            Self::I8 => "I8",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Unknown(value) => value.as_str(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceTensorSlice {
    pub name: String,
    pub role: TensorRole,
    pub path: PathBuf,
    pub offset: u64,
    pub bytes: u64,
    pub dtype: SourceDType,
    pub shape: Vec<usize>,
}

impl SourceTensorSlice {
    pub fn from_hf_inventory(model_dir: &Path, info: &HfSafetensorsTensorInfo) -> Self {
        Self {
            name: info.name.clone(),
            role: info.role.clone(),
            path: model_dir.join(&info.shard),
            offset: info.file_offset,
            bytes: info.byte_size,
            dtype: SourceDType::from_safetensors_dtype(&info.dtype),
            shape: info.shape.clone(),
        }
    }

    pub fn element_count(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or_else(|| {
                Error::Model(format!(
                    "source tensor '{}' element count overflow for shape {:?}",
                    self.name, self.shape
                ))
            })
        })
    }

    pub fn end_offset(&self) -> u64 {
        self.offset.saturating_add(self.bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceTensorPayload {
    pub slice: SourceTensorSlice,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct SourceTensorReader {
    max_tensor_bytes: u64,
}

impl SourceTensorReader {
    pub fn new(max_tensor_bytes: u64) -> Self {
        Self { max_tensor_bytes }
    }

    pub fn read_slice(&self, slice: &SourceTensorSlice) -> Result<SourceTensorPayload> {
        if slice.bytes > self.max_tensor_bytes {
            return Err(Error::Model(format!(
                "source tensor '{}' exceeds bounded read size: {} > {} bytes",
                slice.name, slice.bytes, self.max_tensor_bytes
            )));
        }
        let mut file = std::fs::File::open(&slice.path).map_err(|e| {
            Error::Model(format!(
                "source tensor open '{}': {e}",
                slice.path.display()
            ))
        })?;
        file.seek(SeekFrom::Start(slice.offset)).map_err(|e| {
            Error::Model(format!(
                "source tensor seek '{}': {e}",
                slice.path.display()
            ))
        })?;
        let mut bytes = vec![0u8; slice.bytes as usize];
        file.read_exact(&mut bytes).map_err(|e| {
            Error::Model(format!(
                "source tensor read '{}': {e}",
                slice.path.display()
            ))
        })?;
        Ok(SourceTensorPayload {
            slice: slice.clone(),
            bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_dtype_maps_known_safetensors_names() {
        assert_eq!(SourceDType::from_safetensors_dtype("F32"), SourceDType::F32);
        assert_eq!(
            SourceDType::from_safetensors_dtype("BF16"),
            SourceDType::Bf16
        );
        assert_eq!(
            SourceDType::from_safetensors_dtype("F8_E4M3"),
            SourceDType::F8E4M3
        );
        assert_eq!(
            SourceDType::from_safetensors_dtype("F8_E8M0"),
            SourceDType::F8E8M0
        );
        assert_eq!(SourceDType::from_safetensors_dtype("I8"), SourceDType::I8);
        assert_eq!(SourceDType::from_safetensors_dtype("I32"), SourceDType::I32);
        assert_eq!(SourceDType::from_safetensors_dtype("I64"), SourceDType::I64);
    }

    #[test]
    fn bounded_reader_reads_exact_byte_range() {
        let dir = unique_temp_dir("ferrule-source-tensor-reader");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("source.bin");
        std::fs::write(&path, (0u8..32).collect::<Vec<_>>()).unwrap();
        let slice = SourceTensorSlice {
            name: "test.weight".into(),
            role: TensorRole::AttentionQuery,
            path,
            offset: 8,
            bytes: 6,
            dtype: SourceDType::F32,
            shape: vec![6],
        };
        let payload = SourceTensorReader::new(8).read_slice(&slice).unwrap();
        assert_eq!(payload.bytes, vec![8, 9, 10, 11, 12, 13]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bounded_reader_rejects_large_slice() {
        let slice = SourceTensorSlice {
            name: "large.weight".into(),
            role: TensorRole::AttentionQuery,
            path: PathBuf::from("missing.bin"),
            offset: 0,
            bytes: 9,
            dtype: SourceDType::F32,
            shape: vec![9],
        };
        let err = SourceTensorReader::new(8).read_slice(&slice).unwrap_err();
        assert!(err.to_string().contains("bounded read size"));
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nonce}"))
    }
}
