//! Quantized weight cache — serialise/deserialise Q4_0 weights to disk.
//! Format: magic "FERRULEQ" + layer count + per-layer size-prefixed blobs.

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
pub fn read_buf_bytes<'a>(data: &'a [u8], pos: &mut usize) -> &'a [u8] {
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

/// Quantized layer data suitable for caching.
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

    pub fn deserialize_layer(data: &[u8]) -> Self {
        let mut pos = 0usize;
        Self {
            qp_packed: read_buf_bytes(data, &mut pos).to_vec(),
            qp_scales: read_buf_bytes(data, &mut pos).to_vec(),
            kp_packed: read_buf_bytes(data, &mut pos).to_vec(),
            kp_scales: read_buf_bytes(data, &mut pos).to_vec(),
            vp_packed: read_buf_bytes(data, &mut pos).to_vec(),
            vp_scales: read_buf_bytes(data, &mut pos).to_vec(),
            op_packed: read_buf_bytes(data, &mut pos).to_vec(),
            op_scales: read_buf_bytes(data, &mut pos).to_vec(),
            gate_q_packed: read_buf_bytes(data, &mut pos).to_vec(),
            gate_q_scales: read_buf_bytes(data, &mut pos).to_vec(),
            up_q_packed: read_buf_bytes(data, &mut pos).to_vec(),
            up_q_scales: read_buf_bytes(data, &mut pos).to_vec(),
            down_q_packed: read_buf_bytes(data, &mut pos).to_vec(),
            down_q_scales: read_buf_bytes(data, &mut pos).to_vec(),
        }
    }
}

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

/// Read quantized layers from cache file.
pub fn read_cache(path: &Path) -> Result<Vec<QCacheData>> {
    let data = std::fs::read(path).map_err(|e| Error::Internal(format!("cache read: {e}")))?;
    if data.len() < 8 || &data[..8] != CACHE_MAGIC {
        return Err(Error::Internal("invalid cache magic".into()));
    }
    let mut pos = 8usize;
    let num = read_u64(&data, &mut pos) as usize;
    let mut layers = Vec::with_capacity(num);
    for _ in 0..num {
        let blob_len = read_u64(&data, &mut pos) as usize;
        let blob = &data[pos..pos + blob_len];
        pos += blob_len;
        layers.push(QCacheData::deserialize_layer(blob));
    }
    Ok(layers)
}
