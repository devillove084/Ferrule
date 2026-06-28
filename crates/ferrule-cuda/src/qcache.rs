//! Quantized weight cache — serialise/deserialise Q4_0 weights to disk.
//! Format: magic "FERRULEQ" + layer count + per-layer size-prefixed blobs.
//!
//! Writing uses QCacheData (owned Vecs). Reading uses QCacheReader (mmap, zero-copy).

use ferrule_core::{Error, Result};
use ferrule_quant::f16_to_f32;
use std::path::Path;

const CACHE_MAGIC: &[u8; 8] = b"FERRULEQ";

pub fn cache_path(model_dir: &Path, qt_suffix: &str) -> std::path::PathBuf {
    model_dir.join(format!("model.{}.qcache", qt_suffix))
}

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
pub fn read_u64(data: &[u8], pos: &mut usize) -> u64 {
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    v
}

fn write_buf(buf: &mut Vec<u8>, data: &[u8]) {
    write_u64(buf, data.len() as u64);
    buf.extend_from_slice(data);
}
pub fn read_buf_slice<'a>(data: &'a [u8], pos: &mut usize) -> &'a [u8] {
    let len = read_u64(data, pos) as usize;
    let slice = &data[*pos..*pos + len];
    *pos += len;
    slice
}

pub fn bytes_to_f32_slice(data: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) }
}

fn f16scales_to_bytes(raw: &[u16]) -> Vec<u8> {
    let f32s: Vec<f32> = raw.iter().map(|&b| f16_to_f32(b)).collect();
    unsafe { std::slice::from_raw_parts(f32s.as_ptr() as *const u8, f32s.len() * 4).to_vec() }
}

// ── Owned quantized layer data (for writing) ──────────────────────────

pub struct QCacheData {
    pub qp_packed: Vec<u8>,
    pub qp_scales: Vec<u8>,
    pub kp_packed: Vec<u8>,
    pub kp_scales: Vec<u8>,
    pub vp_packed: Vec<u8>,
    pub vp_scales: Vec<u8>,
    pub op_packed: Vec<u8>,
    pub op_scales: Vec<u8>,
    pub gate_q_packed: Vec<u8>,
    pub gate_q_scales: Vec<u8>,
    pub up_q_packed: Vec<u8>,
    pub up_q_scales: Vec<u8>,
    pub down_q_packed: Vec<u8>,
    pub down_q_scales: Vec<u8>,
}

impl QCacheData {
    pub fn from_qmatrix(
        qp: &ferrule_quant::QMatrix,
        kp: &ferrule_quant::QMatrix,
        vp: &ferrule_quant::QMatrix,
        op: &ferrule_quant::QMatrix,
        gate_q_packed: &[u8],
        gate_q_scales: &[u16],
        up_q_packed: &[u8],
        up_q_scales: &[u16],
        down_q_packed: &[u8],
        down_q_scales: &[u16],
    ) -> Self {
        Self {
            qp_packed: qp.packed.clone(),
            qp_scales: f16scales_to_bytes(&qp.scales),
            kp_packed: kp.packed.clone(),
            kp_scales: f16scales_to_bytes(&kp.scales),
            vp_packed: vp.packed.clone(),
            vp_scales: f16scales_to_bytes(&vp.scales),
            op_packed: op.packed.clone(),
            op_scales: f16scales_to_bytes(&op.scales),
            gate_q_packed: gate_q_packed.to_vec(),
            gate_q_scales: f16scales_to_bytes(gate_q_scales),
            up_q_packed: up_q_packed.to_vec(),
            up_q_scales: f16scales_to_bytes(up_q_scales),
            down_q_packed: down_q_packed.to_vec(),
            down_q_scales: f16scales_to_bytes(down_q_scales),
        }
    }

    pub fn serialize_layer(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        write_buf(&mut buf, &self.qp_packed);
        write_buf(&mut buf, &self.qp_scales);
        write_buf(&mut buf, &self.kp_packed);
        write_buf(&mut buf, &self.kp_scales);
        write_buf(&mut buf, &self.vp_packed);
        write_buf(&mut buf, &self.vp_scales);
        write_buf(&mut buf, &self.op_packed);
        write_buf(&mut buf, &self.op_scales);
        write_buf(&mut buf, &self.gate_q_packed);
        write_buf(&mut buf, &self.gate_q_scales);
        write_buf(&mut buf, &self.up_q_packed);
        write_buf(&mut buf, &self.up_q_scales);
        write_buf(&mut buf, &self.down_q_packed);
        write_buf(&mut buf, &self.down_q_scales);
        buf
    }
}

// ── Zero-copy reader (mmap → slices → GPU, no intermediate allocs) ────

/// Holds an mmap'd cache file and provides zero-copy layer access.
pub struct QCacheReader {
    _mmap: memmap2::Mmap,
    num_layers: usize,
    /// Byte positions of each layer's blob within the mmap.
    layer_offsets: Vec<(usize, usize)>, // (start, end) in mmap bytes
}

impl QCacheReader {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            std::fs::File::open(path).map_err(|e| Error::Internal(format!("cache open: {e}")))?;
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| Error::Internal(format!("cache mmap: {e}")))?
        };
        let data: &[u8] = &mmap;
        if data.len() < 8 || &data[..8] != CACHE_MAGIC {
            return Err(Error::Internal("invalid cache magic".into()));
        }
        let mut pos = 8usize;
        let num_layers = read_u64(data, &mut pos) as usize;
        let mut layer_offsets = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let blob_len = read_u64(data, &mut pos) as usize;
            layer_offsets.push((pos, pos + blob_len));
            pos += blob_len;
        }
        Ok(Self {
            _mmap: mmap,
            num_layers,
            layer_offsets,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get zero-copy slices for a layer. All slices point into the mmap.
    pub fn layer(&self, li: usize) -> QCacheSlice<'_> {
        let (start, _end) = self.layer_offsets[li];
        let data = &self._mmap[start..];
        let mut pos = 0usize;
        QCacheSlice {
            qp_packed: read_buf_slice(data, &mut pos),
            qp_scales: read_buf_slice(data, &mut pos),
            kp_packed: read_buf_slice(data, &mut pos),
            kp_scales: read_buf_slice(data, &mut pos),
            vp_packed: read_buf_slice(data, &mut pos),
            vp_scales: read_buf_slice(data, &mut pos),
            op_packed: read_buf_slice(data, &mut pos),
            op_scales: read_buf_slice(data, &mut pos),
            gate_q_packed: read_buf_slice(data, &mut pos),
            gate_q_scales: read_buf_slice(data, &mut pos),
            up_q_packed: read_buf_slice(data, &mut pos),
            up_q_scales: read_buf_slice(data, &mut pos),
            down_q_packed: read_buf_slice(data, &mut pos),
            down_q_scales: read_buf_slice(data, &mut pos),
        }
    }
}

/// Zero-copy view of a single layer's quantized weights.
/// All slices borrow from the mmap'd cache file.
pub struct QCacheSlice<'a> {
    pub qp_packed: &'a [u8],
    pub qp_scales: &'a [u8],
    pub kp_packed: &'a [u8],
    pub kp_scales: &'a [u8],
    pub vp_packed: &'a [u8],
    pub vp_scales: &'a [u8],
    pub op_packed: &'a [u8],
    pub op_scales: &'a [u8],
    pub gate_q_packed: &'a [u8],
    pub gate_q_scales: &'a [u8],
    pub up_q_packed: &'a [u8],
    pub up_q_scales: &'a [u8],
    pub down_q_packed: &'a [u8],
    pub down_q_scales: &'a [u8],
}

// ── Cache file I/O ─────────────────────────────────────────────────────

/// Write quantized layers to cache file.
pub fn write_cache(path: &Path, layers: &[QCacheData]) -> Result<()> {
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(path).map_err(|e| Error::Internal(format!("cache create: {e}")))?,
    );
    use std::io::Write;
    f.write_all(CACHE_MAGIC)
        .map_err(|e| Error::Internal(format!("cache write: {e}")))?;
    f.write_all(&(layers.len() as u64).to_le_bytes())
        .map_err(|e| Error::Internal(format!("cache write: {e}")))?;
    for q in layers {
        let blob = q.serialize_layer();
        f.write_all(&(blob.len() as u64).to_le_bytes())
            .map_err(|e| Error::Internal(format!("cache write: {e}")))?;
        f.write_all(&blob)
            .map_err(|e| Error::Internal(format!("cache write: {e}")))?;
    }
    Ok(())
}
