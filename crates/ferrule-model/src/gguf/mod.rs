//! GGUF (GPT-Generated Unified Format) reader.
#![allow(clippy::needless_range_loop)]
//!
//! Full spec: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
//!
//! ## Key design:
//! - Mmap-based: the entire file is memory-mapped, zero-copy where possible
//! - Lazy tensor loading: metadata parsed eagerly, tensor data on demand
//! - Full quantization type support: IQ1_S through Q8_K, plus BF16/F16/F32
//!
//! ## File layout:
//! ```text
//! [magic: 4B "GGUF"] [version: 4B] [n_tensors: 8B] [n_kv: 8B]
//! [kv pairs: variable]
//! [tensor infos: n_tensors * (name + n_dims + dims + type + offset)]
//! [padding to GGUF_DEFAULT_ALIGNMENT]
//! [tensor data: variable]
//! ```
//!
//! # Safety
//! Memory mapping is encapsulated by the file readers, which expose only
//! bounds-checked immutable views of model artifacts.

use ferrule_common::{Error, QuantType, Result};
pub mod safetensors;
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::Path;

/// GGUF magic bytes.
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

/// Value types stored in GGUF key-value metadata.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

/// Descriptor for one tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    /// Shape: [ne0, ne1, ne2, ne3] in ggml convention.
    /// ne0 = fastest-varying dimension (inner).
    pub dims: [u64; 4],
    pub quant_type: QuantType,
    /// Byte offset of tensor data relative to the GGUF tensor data section.
    pub offset: u64,
}

/// A memory-mapped GGUF file.
pub struct GgufFile {
    mmap: Mmap,
    /// Key-value metadata from the header.
    pub metadata: HashMap<String, GgufValue>,
    /// All tensors in the file.
    pub tensors: Vec<TensorInfo>,
    /// Absolute byte offset where the tensor data section starts.
    data_start: usize,
}

// ---------------------------------------------------------------------------
// Reading helpers
// ---------------------------------------------------------------------------

fn read_u32(data: &[u8], offset: &mut usize) -> u32 {
    let v = u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    v
}

fn read_u64(data: &[u8], offset: &mut usize) -> u64 {
    let v = u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    v
}

fn read_i32(data: &[u8], offset: &mut usize) -> i32 {
    let v = i32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    v
}

fn read_i64(data: &[u8], offset: &mut usize) -> i64 {
    let v = i64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    v
}

fn read_f32(data: &[u8], offset: &mut usize) -> f32 {
    let v = f32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    v
}

fn read_f64(data: &[u8], offset: &mut usize) -> f64 {
    let v = f64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    v
}

fn read_u8(data: &[u8], offset: &mut usize) -> u8 {
    let v = data[*offset];
    *offset += 1;
    v
}

fn read_i8(data: &[u8], offset: &mut usize) -> i8 {
    read_u8(data, offset) as i8
}

fn read_u16(data: &[u8], offset: &mut usize) -> u16 {
    let v = u16::from_le_bytes(data[*offset..*offset + 2].try_into().unwrap());
    *offset += 2;
    v
}

fn read_i16(data: &[u8], offset: &mut usize) -> i16 {
    let v = i16::from_le_bytes(data[*offset..*offset + 2].try_into().unwrap());
    *offset += 2;
    v
}

fn read_bool(data: &[u8], offset: &mut usize) -> bool {
    read_u8(data, offset) != 0
}

fn read_string(data: &[u8], offset: &mut usize) -> Result<String> {
    let len = read_u64(data, offset) as usize;
    if *offset + len > data.len() {
        return Err(Error::Gguf("string length exceeds data".into()));
    }
    let s = String::from_utf8(data[*offset..*offset + len].to_vec())
        .map_err(|e| Error::Gguf(format!("invalid UTF-8: {e}")))?;
    *offset += len;
    Ok(s)
}

fn read_value(data: &[u8], offset: &mut usize, vtype: u32) -> Result<GgufValue> {
    Ok(match vtype {
        0 => GgufValue::U8(read_u8(data, offset)),
        1 => GgufValue::I8(read_i8(data, offset)),
        2 => GgufValue::U16(read_u16(data, offset)),
        3 => GgufValue::I16(read_i16(data, offset)),
        4 => GgufValue::U32(read_u32(data, offset)),
        5 => GgufValue::I32(read_i32(data, offset)),
        6 => GgufValue::F32(read_f32(data, offset)),
        7 => GgufValue::Bool(read_bool(data, offset)),
        8 => GgufValue::String(read_string(data, offset)?),
        9 => {
            let len = read_u64(data, offset) as usize;
            let elem_type = read_u32(data, offset);
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_value(data, offset, elem_type)?);
            }
            GgufValue::Array(arr)
        }
        10 => GgufValue::U64(read_u64(data, offset)),
        11 => GgufValue::I64(read_i64(data, offset)),
        12 => GgufValue::F64(read_f64(data, offset)),
        _ => return Err(Error::Gguf(format!("unknown value type: {vtype}"))),
    })
}

fn metadata_alignment(metadata: &HashMap<String, GgufValue>) -> Option<u32> {
    match metadata.get("general.alignment")? {
        GgufValue::U32(v) => Some(*v),
        GgufValue::U64(v) => Some(*v as u32),
        GgufValue::I32(v) if *v > 0 => Some(*v as u32),
        GgufValue::I64(v) if *v > 0 => Some(*v as u32),
        _ => None,
    }
}

fn align_up(offset: usize, alignment: usize) -> Result<usize> {
    let add = alignment
        .checked_sub(1)
        .ok_or_else(|| Error::Gguf("invalid GGUF alignment".into()))?;
    let sum = offset
        .checked_add(add)
        .ok_or_else(|| Error::Gguf("GGUF alignment overflow".into()))?;
    Ok(sum / alignment * alignment)
}

fn quant_block_bytes(qt: QuantType) -> usize {
    match qt {
        QuantType::F32 => 4,
        QuantType::F16 | QuantType::Bf16 => 2,
        QuantType::Q4_0 => 18,
        QuantType::Q4_1 => 20,
        QuantType::Q5_0 => 22,
        QuantType::Q5_1 => 24,
        QuantType::Q8_0 => 34,
        QuantType::Q8_1 => 36,
        QuantType::Q2_K => 84,
        QuantType::Q3_K => 110,
        QuantType::Q4_K => 144,
        QuantType::Q5_K => 176,
        QuantType::Q6_K => 210,
        QuantType::Q8_K => 292,
        // IQ sizes are derived from GGUF's average type size metadata in core.
        other => (other.type_size() * other.block_size() as f64).ceil() as usize,
    }
}

// ---------------------------------------------------------------------------
// GgufFile
// ---------------------------------------------------------------------------

impl GgufFile {
    /// Open and parse a GGUF file. The underlying file is memory-mapped.
    #[allow(unsafe_code)]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        // SAFETY: model artifacts are treated as immutable while loaded. The
        // mapping is read-only, and `Mmap` keeps it alive after `file` is dropped.
        let mmap = unsafe { Mmap::map(&file)? };

        let data: &[u8] = &mmap;
        let mut offset = 0usize;

        // Magic
        let magic = read_u32(data, &mut offset);
        if magic != GGUF_MAGIC {
            return Err(Error::Gguf(format!(
                "bad magic: 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X}"
            )));
        }

        // Version
        let version = read_u32(data, &mut offset);
        if !(2..=3).contains(&version) {
            return Err(Error::Gguf(format!("unsupported version: {version}")));
        }

        // Tensor + KV counts
        let n_tensors = read_u64(data, &mut offset);
        let n_kv = read_u64(data, &mut offset);

        // Parse KV metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = read_string(data, &mut offset)?;
            let vtype = read_u32(data, &mut offset);
            let value = read_value(data, &mut offset, vtype)?;
            metadata.insert(key, value);
        }

        // Parse tensor infos
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_string(data, &mut offset)?;
            let n_dims = read_u32(data, &mut offset);

            let mut dims = [0u64; 4];
            for i in 0..n_dims as usize {
                dims[i] = read_u64(data, &mut offset);
            }

            let type_id = read_u32(data, &mut offset);
            let tensor_offset = read_u64(data, &mut offset);

            // GGUF v3 supports rank 0 tensors (scalars)
            let quant_type = type_id_to_quant(type_id)?;

            tensors.push(TensorInfo {
                name,
                n_dims,
                dims,
                quant_type,
                offset: tensor_offset,
            });
        }

        let alignment = metadata_alignment(&metadata).unwrap_or(32).max(1) as usize;
        let data_start = align_up(offset, alignment)?;
        if data_start > data.len() {
            return Err(Error::Gguf(format!(
                "tensor data section starts past file: {data_start} > {}",
                data.len()
            )));
        }

        Ok(Self {
            mmap,
            metadata,
            tensors,
            data_start,
        })
    }

    /// Get a metadata value by key.
    pub fn meta(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get a string metadata value.
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// GGUF architecture name, e.g. `llama`, `qwen2`, `deepseek4`.
    pub fn architecture(&self) -> Option<&str> {
        self.meta_str("general.architecture")
    }

    /// Get a u32 metadata value.
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            GgufValue::U32(v) => Some(*v),
            GgufValue::U64(v) => Some(*v as u32),
            GgufValue::I32(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Get a f32 metadata value.
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            GgufValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    /// Find a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Total element count for a tensor.
    pub fn nelements(t: &TensorInfo) -> usize {
        t.dims[..t.n_dims as usize].iter().product::<u64>() as usize
    }

    /// Serialized byte size for a tensor, including quantization block overhead.
    pub fn tensor_nbytes(t: &TensorInfo) -> Result<usize> {
        let ne = Self::nelements(t);
        let block = t.quant_type.block_size().max(1);
        let blocks = ne.div_ceil(block);
        let block_bytes = quant_block_bytes(t.quant_type);
        blocks
            .checked_mul(block_bytes)
            .ok_or_else(|| Error::Gguf(format!("tensor '{}' byte size overflow", t.name)))
    }

    /// Get a raw byte slice for a tensor's data (zero-copy from mmap).
    pub fn tensor_data(&self, tensor: &TensorInfo) -> Result<&[u8]> {
        let relative = tensor.offset as usize;
        let start = self
            .data_start
            .checked_add(relative)
            .ok_or_else(|| Error::Gguf(format!("tensor '{}' offset overflow", tensor.name)))?;
        let size = Self::tensor_nbytes(tensor)?;
        let end = start
            .checked_add(size)
            .ok_or_else(|| Error::Gguf(format!("tensor '{}' size overflow", tensor.name)))?;
        if end > self.mmap.len() {
            return Err(Error::Gguf(format!(
                "tensor '{}' data extends past file: {end} > {}",
                tensor.name,
                self.mmap.len()
            )));
        }
        Ok(&self.mmap[start..end])
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> Option<u32> {
        // Try common GGUF keys for vocab size
        for key in &[
            "tokenizer.ggml.tokens.len",
            "llama.vocab_size",
            "gemma.vocab_size",
            "qwen2.vocab_size",
            "qwen3.vocab_size",
            "deepseek2.vocab_size",
            "deepseek4.vocab_size",
        ] {
            if let Some(v) = self.meta_u32(key) {
                return Some(v);
            }
        }
        // Fall back to array length of token list
        if let Some(GgufValue::Array(arr)) = self.metadata.get("tokenizer.ggml.tokens") {
            return Some(arr.len() as u32);
        }
        None
    }

    /// Number of layers in the model.
    pub fn num_layers(&self) -> Option<u32> {
        self.meta_u32("llama.block_count")
            .or_else(|| self.meta_u32("gemma.block_count"))
            .or_else(|| self.meta_u32("qwen2.block_count"))
            .or_else(|| self.meta_u32("qwen3.block_count"))
            .or_else(|| self.meta_u32("deepseek2.block_count"))
            .or_else(|| self.meta_u32("deepseek4.block_count"))
    }

    /// Hidden size (model dimension).
    pub fn hidden_size(&self) -> Option<u32> {
        self.meta_u32("llama.embedding_length")
            .or_else(|| self.meta_u32("gemma.embedding_length"))
            .or_else(|| self.meta_u32("qwen2.embedding_length"))
            .or_else(|| self.meta_u32("qwen3.embedding_length"))
            .or_else(|| self.meta_u32("deepseek2.embedding_length"))
            .or_else(|| self.meta_u32("deepseek4.embedding_length"))
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> Option<u32> {
        self.meta_u32("llama.attention.head_count")
            .or_else(|| self.meta_u32("gemma.attention.head_count"))
            .or_else(|| self.meta_u32("qwen2.attention.head_count"))
            .or_else(|| self.meta_u32("qwen3.attention.head_count"))
            .or_else(|| self.meta_u32("deepseek2.attention.head_count"))
            .or_else(|| self.meta_u32("deepseek4.attention.head_count"))
    }

    /// Number of key/value heads, when separate from attention heads.
    pub fn num_kv_heads(&self) -> Option<u32> {
        self.meta_u32("llama.attention.head_count_kv")
            .or_else(|| self.meta_u32("gemma.attention.head_count_kv"))
            .or_else(|| self.meta_u32("qwen2.attention.head_count_kv"))
            .or_else(|| self.meta_u32("qwen3.attention.head_count_kv"))
            .or_else(|| self.meta_u32("deepseek2.attention.head_count_kv"))
            .or_else(|| self.meta_u32("deepseek4.attention.head_count_kv"))
    }

    /// Number of routed experts, when encoded in GGUF metadata.
    pub fn expert_count(&self) -> Option<u32> {
        self.meta_u32("llama.expert_count")
            .or_else(|| self.meta_u32("qwen2.expert_count"))
            .or_else(|| self.meta_u32("qwen3.expert_count"))
            .or_else(|| self.meta_u32("deepseek2.expert_count"))
            .or_else(|| self.meta_u32("deepseek4.expert_count"))
            .or_else(|| self.meta_u32("deepseek2.feed_forward.expert_count"))
            .or_else(|| self.meta_u32("deepseek4.feed_forward.expert_count"))
    }

    /// Number of selected experts per token, when encoded in GGUF metadata.
    pub fn expert_used_count(&self) -> Option<u32> {
        self.meta_u32("llama.expert_used_count")
            .or_else(|| self.meta_u32("qwen2.expert_used_count"))
            .or_else(|| self.meta_u32("qwen3.expert_used_count"))
            .or_else(|| self.meta_u32("deepseek2.expert_used_count"))
            .or_else(|| self.meta_u32("deepseek4.expert_used_count"))
            .or_else(|| self.meta_u32("deepseek2.feed_forward.expert_used_count"))
            .or_else(|| self.meta_u32("deepseek4.feed_forward.expert_used_count"))
    }
}

// ---------------------------------------------------------------------------
// Type mapping
// ---------------------------------------------------------------------------

fn type_id_to_quant(type_id: u32) -> Result<QuantType> {
    Ok(match type_id {
        0 => QuantType::F32,
        1 => QuantType::F16,
        2 => QuantType::Q4_0,
        3 => QuantType::Q4_1,
        6 => QuantType::Q5_0,
        7 => QuantType::Q5_1,
        8 => QuantType::Q8_0,
        9 => QuantType::Q8_1,
        10 => QuantType::Q2_K,
        11 => QuantType::Q3_K,
        12 => QuantType::Q4_K,
        13 => QuantType::Q5_K,
        14 => QuantType::Q6_K,
        15 => QuantType::Q8_K,
        16 => QuantType::Iq2Xxs,
        17 => QuantType::Iq2Xs,
        18 => QuantType::Iq3Xxs,
        19 => QuantType::Iq1S,
        20 => QuantType::Iq4Nl,
        21 => QuantType::Iq3S,
        22 => QuantType::Iq2S,
        23 => QuantType::Iq4Xs,
        30 => QuantType::Bf16,
        41 => QuantType::Q1_0,
        _ => return Err(Error::Gguf(format!("unknown GGUF type id: {type_id}"))),
    })
}
