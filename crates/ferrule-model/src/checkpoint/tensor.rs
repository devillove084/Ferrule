//! Generic checkpoint tensor payloads and bounded local reads.
//!
//! Model-family adapters classify tensor names into semantic roles in
//! `ferrule-model`. Runtime code should consume these checkpoint tensor descriptors
//! without matching on model-specific names. This module is the generic bridge
//! from HF safetensors inventory byte ranges to small reference payloads and,
//! later, GPU/streaming tensor handles.

use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::{CheckpointReadExtent, CheckpointReadPlan, CheckpointSourceFileIdentity};

use crate::{HfSafetensorsTensorInfo, TensorRole};
use ferrule_common::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointDType {
    F32,
    Bf16,
    F8E4M3,
    F8E8M0,
    I8,
    I32,
    I64,
    Unknown(String),
}

impl CheckpointDType {
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

    pub fn element_size_bytes(&self) -> Option<usize> {
        match self {
            Self::F32 | Self::I32 => Some(4),
            Self::Bf16 => Some(2),
            Self::F8E4M3 | Self::F8E8M0 | Self::I8 => Some(1),
            Self::I64 => Some(8),
            Self::Unknown(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointTensorSlice {
    pub name: String,
    pub role: TensorRole,
    pub path: PathBuf,
    pub offset: u64,
    pub bytes: u64,
    pub dtype: CheckpointDType,
    pub shape: Vec<usize>,
}

impl CheckpointTensorSlice {
    pub fn from_hf_inventory(model_dir: &Path, info: &HfSafetensorsTensorInfo) -> Self {
        Self {
            name: info.name.clone(),
            role: info.role.clone(),
            path: model_dir.join(&info.shard),
            offset: info.file_offset,
            bytes: info.byte_size,
            dtype: CheckpointDType::from_safetensors_dtype(&info.dtype),
            shape: info.shape.clone(),
        }
    }

    pub fn element_count(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or_else(|| Error::Model {
                message: format!(
                    "checkpoint tensor '{}' element count overflow for shape {:?}",
                    self.name, self.shape
                ),
            })
        })
    }

    pub fn end_offset(&self) -> u64 {
        self.offset.saturating_add(self.bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointMatrixSlice {
    pub slice: CheckpointTensorSlice,
    pub rows: usize,
    pub cols: usize,
}

impl CheckpointMatrixSlice {
    pub fn from_slice(slice: CheckpointTensorSlice, label: &str) -> Result<Self> {
        let [rows, cols]: [usize; 2] =
            slice
                .shape
                .clone()
                .try_into()
                .map_err(|shape: Vec<usize>| Error::Model {
                    message: format!(
                        "checkpoint matrix {label} '{}' expects 2D shape, got {:?}",
                        slice.name, shape
                    ),
                })?;
        Ok(Self { slice, rows, cols })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointTensorPayload {
    pub slice: CheckpointTensorSlice,
    pub bytes: Vec<u8>,
}

/// Checked rectangular view of an immutable dense checkpoint matrix.
///
/// The original tensor remains intact: packed column bytes must never masquerade
/// as a contiguous on-disk `CheckpointTensorSlice`. Extents retain physical
/// provenance, while `local_shape` describes the packed row-major result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointMatrixRead {
    tensor: CheckpointTensorSlice,
    rows: Range<usize>,
    columns: Range<usize>,
    plan: CheckpointReadPlan,
}

impl CheckpointMatrixRead {
    pub fn tensor(&self) -> &CheckpointTensorSlice {
        &self.tensor
    }
    pub fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }
    pub fn columns(&self) -> Range<usize> {
        self.columns.clone()
    }
    pub fn local_shape(&self) -> [usize; 2] {
        [self.rows.len(), self.columns.len()]
    }
    pub fn read_plan(&self) -> &CheckpointReadPlan {
        &self.plan
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct CheckpointMatrixPayload {
    provenance: CheckpointMatrixRead,
    bytes: Vec<u8>,
}

impl CheckpointMatrixPayload {
    pub fn provenance(&self) -> &CheckpointMatrixRead {
        &self.provenance
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn into_parts(self) -> (CheckpointMatrixRead, Vec<u8>) {
        (self.provenance, self.bytes)
    }
}

#[derive(Debug, Clone)]
pub struct CheckpointTensorReader {
    max_tensor_bytes: u64,
}

impl CheckpointTensorReader {
    pub fn new(max_tensor_bytes: u64) -> Self {
        Self { max_tensor_bytes }
    }

    pub fn max_tensor_bytes(&self) -> u64 {
        self.max_tensor_bytes
    }

    pub fn read_slice(&self, slice: &CheckpointTensorSlice) -> Result<CheckpointTensorPayload> {
        self.read_physical_range(slice, slice.offset, slice.bytes, slice.shape.clone())
    }

    pub fn read_2d_rows(
        &self,
        slice: &CheckpointTensorSlice,
        start_row: usize,
        row_count: usize,
    ) -> Result<CheckpointTensorPayload> {
        let end = start_row
            .checked_add(row_count)
            .ok_or_else(|| matrix_error("row range overflow"))?;
        let columns = slice.shape.get(1).copied().unwrap_or(0);
        let source = CheckpointSourceFileIdentity::capture(&slice.path)?;
        let read = self.plan_2d_range(slice, start_row..end, 0..columns, &source)?;
        let (read, bytes) = self.read_matrix(&read)?.into_parts();
        let mut local = slice.clone();
        local.offset = read.plan.extents()[0].offset();
        local.bytes = read.plan.storage_bytes();
        local.shape = read.local_shape().to_vec();
        Ok(CheckpointTensorPayload {
            slice: local,
            bytes,
        })
    }

    /// Convenience read with a snapshot captured now. Bound checkpoint consumers
    /// should instead supply their catalog-time snapshot to `plan_2d_range`.
    pub fn read_2d_columns(
        &self,
        slice: &CheckpointTensorSlice,
        start_column: usize,
        column_count: usize,
    ) -> Result<CheckpointMatrixPayload> {
        let end = start_column
            .checked_add(column_count)
            .ok_or_else(|| matrix_error("column range overflow"))?;
        let rows = slice.shape.first().copied().unwrap_or(0);
        let source = CheckpointSourceFileIdentity::capture(&slice.path)?;
        let read = self.plan_2d_range(slice, 0..rows, start_column..end, &source)?;
        self.read_matrix(&read)
    }

    /// Plan only the requested BF16/F32 rectangle, preserving native encoding.
    /// Packed/quantized formats need encoding-aware scale and block geometry;
    /// interpreting their physical shapes as dense matrices is forbidden here.
    pub fn plan_2d_range(
        &self,
        slice: &CheckpointTensorSlice,
        rows: Range<usize>,
        columns: Range<usize>,
        source: &CheckpointSourceFileIdentity,
    ) -> Result<CheckpointMatrixRead> {
        let element_bytes = match slice.dtype {
            CheckpointDType::F32 => 4usize,
            CheckpointDType::Bf16 => 2usize,
            _ => {
                return Err(matrix_error(format!(
                    "matrix slicing supports only dense BF16/F32, got {}",
                    slice.dtype.as_str()
                )));
            }
        };
        let [full_rows, full_columns] = slice.shape.as_slice() else {
            return Err(matrix_error("matrix slicing requires a 2D shape"));
        };
        let full_bytes = full_rows
            .checked_mul(*full_columns)
            .and_then(|n| n.checked_mul(element_bytes))
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or_else(|| matrix_error("matrix byte size overflow"))?;
        if *full_rows == 0 || *full_columns == 0 || slice.bytes != full_bytes as u64 {
            return Err(matrix_error("matrix shape/dtype byte size mismatch"));
        }
        let full_end = slice
            .offset
            .checked_add(slice.bytes)
            .ok_or_else(|| matrix_error("matrix file extent overflow"))?;
        if source.catalog_path() != slice.path.as_path() || full_end > source.length() {
            return Err(matrix_error(
                "matrix extent exceeds or does not match its source snapshot",
            ));
        }
        if rows.start >= rows.end
            || rows.end > *full_rows
            || columns.start >= columns.end
            || columns.end > *full_columns
        {
            return Err(matrix_error(
                "matrix row/column range is empty or out of bounds",
            ));
        }
        let bytes = rows
            .len()
            .checked_mul(columns.len())
            .and_then(|n| n.checked_mul(element_bytes))
            .ok_or_else(|| matrix_error("local matrix byte size overflow"))?;
        self.check_matrix_read_limit(bytes as u64)?;
        let row_stride = *full_columns * element_bytes;
        let row_bytes = columns.len() * element_bytes;
        let mut extents = Vec::new();
        let extent_count = if columns.len() == *full_columns {
            1
        } else {
            rows.len()
        };
        extents
            .try_reserve_exact(extent_count)
            .map_err(|e| matrix_error(format!("allocate matrix extents: {e}")))?;
        if columns.len() == *full_columns {
            extents.push(CheckpointReadExtent::new(
                slice.path.clone(),
                slice.offset + (rows.start * row_stride) as u64,
                bytes as u64,
            )?);
        } else {
            for row in rows.clone() {
                // Full matrix and selected bounds were checked above, so every
                // subextent is contained in the original physical tensor.
                extents.push(CheckpointReadExtent::new(
                    slice.path.clone(),
                    slice.offset + (row * row_stride + columns.start * element_bytes) as u64,
                    row_bytes as u64,
                )?);
            }
        }
        let sources: Arc<[CheckpointSourceFileIdentity]> = Arc::from([source.clone()]);
        let plan = CheckpointReadPlan::new(extents, sources)?;
        plan.validate_source_identity()
            .map_err(|e| matrix_error(format!("stale matrix source: {e:?}")))?;
        Ok(CheckpointMatrixRead {
            tensor: slice.clone(),
            rows,
            columns,
            plan,
        })
    }

    /// Read directly into one local packed allocation, using one file handle even
    /// for noncontiguous columns. Never allocates or decodes the full tensor.
    pub fn read_matrix(&self, read: &CheckpointMatrixRead) -> Result<CheckpointMatrixPayload> {
        let length = self.check_matrix_read_limit(read.plan.storage_bytes())?;
        read.plan
            .validate_source_identity()
            .map_err(|e| matrix_error(format!("stale matrix source: {e:?}")))?;
        let mut file = std::fs::File::open(&read.tensor.path).map_err(|error| {
            matrix_error(format!(
                "open '{}' for tensor '{}': {error}",
                read.tensor.path.display(),
                read.tensor.name
            ))
        })?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|e| matrix_error(format!("allocate local matrix: {e}")))?;
        bytes.resize(length, 0);
        let mut cursor = 0;
        for extent in read.plan.extents() {
            let end = cursor + extent.bytes() as usize;
            file.seek(SeekFrom::Start(extent.offset()))
                .and_then(|_| file.read_exact(&mut bytes[cursor..end]))
                .map_err(|error| {
                    matrix_error(format!(
                        "read '{}' tensor '{}' extent {}..{}: {error}",
                        extent.path().display(),
                        read.tensor.name,
                        extent.offset(),
                        extent.end()
                    ))
                })?;
            cursor = end;
        }
        read.plan
            .validate_source_identity()
            .map_err(|e| matrix_error(format!("stale matrix source: {e:?}")))?;
        Ok(CheckpointMatrixPayload {
            provenance: read.clone(),
            bytes,
        })
    }

    fn check_matrix_read_limit(&self, bytes: u64) -> Result<usize> {
        if bytes > self.max_tensor_bytes {
            return Err(matrix_error(format!(
                "matrix exceeds bounded read size: {bytes} > {} bytes",
                self.max_tensor_bytes
            )));
        }
        usize::try_from(bytes)
            .ok()
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or_else(|| matrix_error("local matrix exceeds address space"))
    }

    fn read_physical_range(
        &self,
        slice: &CheckpointTensorSlice,
        offset: u64,
        bytes: u64,
        shape: Vec<usize>,
    ) -> Result<CheckpointTensorPayload> {
        if bytes > self.max_tensor_bytes {
            return Err(Error::Model {
                message: format!(
                    "checkpoint tensor '{}' exceeds bounded read size: {} > {} bytes",
                    slice.name, bytes, self.max_tensor_bytes
                ),
            });
        }
        let mut file = std::fs::File::open(&slice.path).map_err(|e| Error::Model {
            message: format!("checkpoint tensor open '{}': {e}", slice.path.display()),
        })?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| Error::Model {
                message: format!("checkpoint tensor seek '{}': {e}", slice.path.display()),
            })?;
        let mut payload_bytes = vec![0u8; bytes as usize];
        file.read_exact(&mut payload_bytes)
            .map_err(|e| Error::Model {
                message: format!("checkpoint tensor read '{}': {e}", slice.path.display()),
            })?;
        let mut range_slice = slice.clone();
        range_slice.offset = offset;
        range_slice.bytes = bytes;
        range_slice.shape = shape;
        Ok(CheckpointTensorPayload {
            slice: range_slice,
            bytes: payload_bytes,
        })
    }
}

fn matrix_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("checkpoint matrix: {}", message.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_dtype_maps_known_safetensors_names() {
        assert_eq!(
            CheckpointDType::from_safetensors_dtype("F32"),
            CheckpointDType::F32
        );
        assert_eq!(
            CheckpointDType::from_safetensors_dtype("BF16"),
            CheckpointDType::Bf16
        );
        assert_eq!(
            CheckpointDType::from_safetensors_dtype("F8_E4M3"),
            CheckpointDType::F8E4M3
        );
        assert_eq!(
            CheckpointDType::from_safetensors_dtype("F8_E8M0"),
            CheckpointDType::F8E8M0
        );
        assert_eq!(
            CheckpointDType::from_safetensors_dtype("I8"),
            CheckpointDType::I8
        );
        assert_eq!(
            CheckpointDType::from_safetensors_dtype("I32"),
            CheckpointDType::I32
        );
        assert_eq!(
            CheckpointDType::from_safetensors_dtype("I64"),
            CheckpointDType::I64
        );
    }

    #[test]
    fn matrix_slice_tracks_dimensions() {
        let slice = CheckpointTensorSlice {
            name: "matrix.weight".into(),
            role: TensorRole::OutputHead,
            path: PathBuf::from("model.safetensors"),
            offset: 16,
            bytes: 24,
            dtype: CheckpointDType::Bf16,
            shape: vec![3, 4],
        };
        let matrix = CheckpointMatrixSlice::from_slice(slice.clone(), "output head").unwrap();
        assert_eq!(matrix.slice, slice);
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 4);
    }

    #[test]
    fn matrix_slice_reports_model_neutral_shape_error() {
        let slice = CheckpointTensorSlice {
            name: "vector.weight".into(),
            role: TensorRole::OutputHead,
            path: PathBuf::from("model.safetensors"),
            offset: 0,
            bytes: 8,
            dtype: CheckpointDType::Bf16,
            shape: vec![4],
        };
        let err = CheckpointMatrixSlice::from_slice(slice, "output head").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("checkpoint matrix output head 'vector.weight' expects 2D shape"));
        assert!(!message.contains("DeepSeek"));
    }

    #[test]
    fn bounded_reader_reads_exact_byte_range() {
        let dir = unique_temp_dir("ferrule-checkpoint-tensor-reader");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("checkpoint.bin");
        std::fs::write(&path, (0u8..32).collect::<Vec<_>>()).unwrap();
        let slice = CheckpointTensorSlice {
            name: "test.weight".into(),
            role: TensorRole::AttentionQuery,
            path,
            offset: 8,
            bytes: 6,
            dtype: CheckpointDType::F32,
            shape: vec![6],
        };
        let payload = CheckpointTensorReader::new(8).read_slice(&slice).unwrap();
        assert_eq!(payload.bytes, vec![8, 9, 10, 11, 12, 13]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bounded_reader_rejects_large_slice() {
        let slice = CheckpointTensorSlice {
            name: "large.weight".into(),
            role: TensorRole::AttentionQuery,
            path: PathBuf::from("missing.bin"),
            offset: 0,
            bytes: 9,
            dtype: CheckpointDType::F32,
            shape: vec![9],
        };
        let err = CheckpointTensorReader::new(8)
            .read_slice(&slice)
            .unwrap_err();
        assert!(err.to_string().contains("bounded read size"));
    }

    #[test]
    fn bounded_reader_reads_2d_row_range() {
        let dir = unique_temp_dir("ferrule-checkpoint-tensor-row-reader");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("matrix.bin");
        let values = (0u16..12)
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(&path, values).unwrap();
        let slice = CheckpointTensorSlice {
            name: "matrix.weight".into(),
            role: TensorRole::OutputHead,
            path,
            offset: 0,
            bytes: 24,
            dtype: CheckpointDType::Bf16,
            shape: vec![3, 4],
        };
        let payload = CheckpointTensorReader::new(32)
            .read_2d_rows(&slice, 1, 2)
            .unwrap();
        assert_eq!(payload.slice.offset, 8);
        assert_eq!(payload.slice.bytes, 16);
        assert_eq!(payload.slice.shape, vec![2, 4]);
        assert_eq!(payload.bytes.len(), 16);
        assert_eq!(u16::from_le_bytes([payload.bytes[0], payload.bytes[1]]), 4);
        assert_eq!(
            u16::from_le_bytes([payload.bytes[14], payload.bytes[15]]),
            11
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nonce}"))
    }
}
