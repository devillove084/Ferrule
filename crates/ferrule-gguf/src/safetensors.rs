//! Safetensors format reader — mmap-based, zero-copy tensor access.
//!
//! Format:
//!   8 bytes: header size (u64 LE)
//!   N bytes: JSON header dict
//!   ... tensor data (aligned)
//!
//! Each tensor in header:
//!   { "dtype": "BF16", "shape": [2816, 262144], "data_offsets": [start, end] }

use ferrule_core::{Error, QuantType, Result};
use memmap2::Mmap;
use memmap2::MmapOptions;
use std::collections::HashMap;
use std::path::Path;

/// Descriptor for a tensor in a safetensors file.
#[derive(Debug, Clone)]
pub struct SafeTensorInfo {
    pub name: String,
    pub dtype: SafeDtype,
    pub shape: Vec<usize>,
    /// Byte offset range [start, end) into the data section.
    pub data_range: (usize, usize),
}

/// Dtypes we handle from safetensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeDtype {
    F32,
    F16,
    Bf16,
    I32,
    I64,
    F8E4M3,
    F8E5M2,
    Bool,
}

impl SafeDtype {
    pub fn element_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Bool => 1,
        }
    }

    pub fn to_quant_type(self) -> QuantType {
        match self {
            Self::F32 => QuantType::F32,
            Self::F16 => QuantType::F16,
            Self::Bf16 => QuantType::Bf16,
            _ => QuantType::F32, // fallback for quantized types
        }
    }
}

/// A memory-mapped safetensors file (single shard).
/// Uses MAP_POPULATE for eager page-fault to avoid lazy I/O during tensor access.
pub struct SafeTensorsFile {
    mmap: Mmap,
    /// Byte offset where tensor data begins (after header + alignment).
    data_start: usize,
    pub tensors: Vec<SafeTensorInfo>,
    /// Name → index lookup.
    index: HashMap<String, usize>,
}

impl SafeTensorsFile {
    /// Open a safetensors file. Uses regular mmap (no MAP_POPULATE) —
    /// deferred page faults let the OS pipeline I/O with subsequent BF16→F32
    /// conversion, avoiding the 30s blocking pre-fault for 15 GB+ models.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let data: &[u8] = &mmap;

        if data.len() < 8 {
            return Err(Error::Gguf("safetensors file too small".into()));
        }

        let header_size = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
        if 8 + header_size > data.len() {
            return Err(Error::Gguf("safetensors header exceeds file size".into()));
        }

        let header_json = std::str::from_utf8(&data[8..8 + header_size])
            .map_err(|e| Error::Gguf(format!("invalid UTF-8 header: {e}")))?;

        let header: serde_json::Value = serde_json::from_str(header_json)
            .map_err(|e| Error::Gguf(format!("invalid JSON header: {e}")))?;

        // Data starts after header, aligned to 256 bytes (typical) or 8.
        let data_start = (8 + header_size + 7) & !7; // 8-byte align

        let mut tensors = Vec::new();
        let mut index = HashMap::new();

        if let Some(obj) = header.as_object() {
            for (name, info) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dtype_str = info["dtype"].as_str().unwrap_or("F32");
                let shape: Vec<usize> = info["shape"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_u64().map(|x| x as usize))
                            .collect()
                    })
                    .unwrap_or_default();
                let offsets = info["data_offsets"].as_array();
                let (start, end) = if let Some(arr) = offsets {
                    (
                        arr[0].as_u64().unwrap_or(0) as usize,
                        arr[1].as_u64().unwrap_or(0) as usize,
                    )
                } else {
                    (0, 0)
                };

                let dtype = match dtype_str {
                    "F32" => SafeDtype::F32,
                    "F16" => SafeDtype::F16,
                    "BF16" => SafeDtype::Bf16,
                    "I32" => SafeDtype::I32,
                    "I64" => SafeDtype::I64,
                    "F8_E4M3" => SafeDtype::F8E4M3,
                    "F8_E5M2" => SafeDtype::F8E5M2,
                    "BOOL" => SafeDtype::Bool,
                    _ => {
                        tracing::warn!("unknown safetensor dtype: {dtype_str}, using F32");
                        SafeDtype::F32
                    }
                };

                let idx = tensors.len();
                index.insert(name.clone(), idx);
                tensors.push(SafeTensorInfo {
                    name: name.clone(),
                    dtype,
                    shape,
                    data_range: (start, end),
                });
            }
        }

        Ok(Self {
            mmap,
            data_start,
            tensors,
            index,
        })
    }

    /// Find a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&SafeTensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// Get raw bytes + dtype for a named tensor.
    pub fn tensor_raw(&self, name: &str) -> Option<(&[u8], SafeDtype, usize)> {
        let info = self.tensor(name)?;
        let raw = self.tensor_data(info);
        let n: usize = info.shape.iter().product();
        Some((raw, info.dtype, n))
    }

    /// Get the raw bytes for a tensor.
    pub fn tensor_data(&self, info: &SafeTensorInfo) -> &[u8] {
        let start = self.data_start + info.data_range.0;
        let end = self.data_start + info.data_range.1;
        &self.mmap[start..end.min(self.mmap.len())]
    }

    /// Dequantize a tensor to f32. Handles BF16 → f32 conversion.
    pub fn tensor_f32(&self, info: &SafeTensorInfo) -> Result<Vec<f32>> {
        let raw = self.tensor_data(info);
        Ok(Self::decode_to_f32(
            raw,
            info.dtype,
            info.shape.iter().product(),
        ))
    }

    /// Decode raw bytes to f32 vector (trait-free, usable from any crate).
    pub fn decode_to_f32(raw: &[u8], dtype: SafeDtype, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        match dtype {
            SafeDtype::F32 => {
                let src = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, n) };
                out.copy_from_slice(src);
            }
            SafeDtype::F16 => {
                let src =
                    unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const half::f16, n) };
                for (o, s) in out.iter_mut().zip(src.iter()) {
                    *o = s.to_f32();
                }
            }
            SafeDtype::Bf16 => {
                let src =
                    unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const half::bf16, n) };
                for (o, s) in out.iter_mut().zip(src.iter()) {
                    *o = s.to_f32();
                }
            }
            SafeDtype::F8E4M3 => {
                for (o, &b) in out.iter_mut().zip(raw.iter()) {
                    *o = fp8_e4m3_to_f32(b);
                }
            }
            _ => {
                out.fill(0.0);
            }
        }
        out
    }
}

/// Safetensors index file — maps tensor names to shard files.
pub struct SafeTensorsIndex {
    pub weight_map: HashMap<String, String>, // tensor_name → shard_filename
}

impl SafeTensorsIndex {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path.as_ref())?;
        let parsed: serde_json::Value =
            serde_json::from_str(&json).map_err(|e| Error::Gguf(format!("index JSON: {e}")))?;
        let map = parsed["weight_map"]
            .as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self { weight_map: map })
    }
}

// ---------------------------------------------------------------------------
// FP8 E4M3 decode (same as DeepSeek V3, NVIDIA/OCP MX format)
// ---------------------------------------------------------------------------

fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = (b >> 2) & 0x0F;
    let mant = b & 0x03;
    let sign_val = if sign == 1 { -1.0f32 } else { 1.0f32 };
    if exp == 0 {
        sign_val * (mant as f32) * 2.0f32.powi(-6) / 4.0
    } else {
        sign_val * (1.0 + mant as f32 / 4.0) * 2.0f32.powi(exp as i32 - 7)
    }
}
