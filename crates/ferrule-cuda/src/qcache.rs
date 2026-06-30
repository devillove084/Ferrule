//! Quantized weight cache with versioned manifest.
//!
//! Format v2:
//!   magic "FERRULEQ" 8B | manifest_len u64 LE | manifest JSON | num_layers u64 LE | layers...
//! Format v1 (backward compat): manifest_len == 0

use ferrule_core::{Error, Result};
use ferrule_quant::{f16_to_f32, QuantType};
use serde::{Deserialize, Serialize};
use std::path::Path;

const CACHE_MAGIC: &[u8; 8] = b"FERRULEQ";
const CURRENT_FORMAT_VERSION: u32 = 2;

// ── Manifest ────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QCacheManifest {
    pub format_version: u32,
    pub quant_type: String,
    pub layout_version: String,
    pub num_layers: usize,
    pub model_config_hash: String,
    pub created_at: String,
    #[serde(default)]
    pub tensor_shapes: Vec<QCacheTensorInfo>,
    pub quant_suffix: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct QCacheTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
}

impl QCacheManifest {
    pub fn new(
        quant_type: &str,
        layout_version: &str,
        num_layers: usize,
        model_config_hash: &str,
        quant_suffix: &str,
    ) -> Self {
        Self {
            format_version: CURRENT_FORMAT_VERSION,
            quant_type: quant_type.to_string(),
            layout_version: layout_version.to_string(),
            num_layers,
            model_config_hash: model_config_hash.to_string(),
            created_at: now_iso(),
            tensor_shapes: Vec::new(),
            quant_suffix: quant_suffix.to_string(),
        }
    }
}

fn now_iso() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let days = (secs / 86400) as i64;
    let (y, m, d) = ymd_from_days(days);
    let sod = secs % 86400;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

fn ymd_from_days(mut days: i64) -> (i64, u32, i64) {
    let mut y = 1970i64;
    loop {
        let diy: i64 = if is_leap(y) { 366 } else { 365 };
        if days < diy {
            break;
        }
        days -= diy;
        y += 1;
    }
    let md = if is_leap(y) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut m = 0u32;
    while m < 12 && days >= md[m as usize] as i64 {
        days -= md[m as usize] as i64;
        m += 1;
    }
    (y, m + 1, days + 1)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

// ── Path helper ──────────────────────────────────────────────────────────

pub fn quant_suffix(qt: QuantType) -> &'static str {
    match qt {
        QuantType::Q4_0 => "q4_0_llama",
        QuantType::Q8_0 => "q8_0",
        QuantType::Q2S => "q2s",
        QuantType::T1S => "t1s",
    }
}

pub fn cache_path(model_dir: &Path, qt_suffix: &str) -> std::path::PathBuf {
    model_dir.join(format!("model.{}.qcache", qt_suffix))
}

/// Stable, cheap fingerprint for cache compatibility checks.
///
/// This intentionally hashes `config.json` contents plus safetensors file names
/// and sizes. It catches the common stale-cache cases without forcing a full
/// model-weight read before the qcache-only path.
pub fn model_config_hash(model_dir: &Path) -> Result<String> {
    let mut h = Fnv1a64::default();
    hash_named_file(&mut h, model_dir, "config.json")?;

    let mut shards: Vec<_> = std::fs::read_dir(model_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    shards.sort();

    for shard in shards {
        let file_name = shard
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| Error::Internal("invalid safetensors filename".into()))?;
        h.update(file_name.as_bytes());
        let len = std::fs::metadata(&shard)?.len();
        h.update(&len.to_le_bytes());
    }

    Ok(format!("{:016x}", h.finish()))
}

fn hash_named_file(h: &mut Fnv1a64, model_dir: &Path, name: &str) -> Result<()> {
    h.update(name.as_bytes());
    h.update(&std::fs::read(model_dir.join(name))?);
    Ok(())
}

#[derive(Default)]
struct Fnv1a64(u64);

impl Fnv1a64 {
    fn update(&mut self, bytes: &[u8]) {
        if self.0 == 0 {
            self.0 = 0xcbf29ce484222325;
        }
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(self) -> u64 {
        self.0
    }
}

// ── Primitive I/O ────────────────────────────────────────────────────────

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
pub fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64> {
    let end = pos
        .checked_add(8)
        .ok_or_else(|| Error::Internal("cache offset overflow".into()))?;
    let bytes = data
        .get(*pos..end)
        .ok_or_else(|| Error::Internal("cache u64 truncated".into()))?;
    let v = u64::from_le_bytes(bytes.try_into().unwrap());
    *pos = end;
    Ok(v)
}
fn write_buf(buf: &mut Vec<u8>, data: &[u8]) {
    write_u64(buf, data.len() as u64);
    buf.extend_from_slice(data);
}
pub fn read_buf_slice<'a>(data: &'a [u8], pos: &mut usize) -> Result<&'a [u8]> {
    let len = read_u64(data, pos)? as usize;
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::Internal("cache buffer offset overflow".into()))?;
    let slice = data
        .get(*pos..end)
        .ok_or_else(|| Error::Internal("cache buffer truncated".into()))?;
    *pos = end;
    Ok(slice)
}
pub fn bytes_to_f32_vec(data: &[u8]) -> Result<Vec<f32>> {
    let mut chunks = data.chunks_exact(4);
    let out: Vec<f32> = chunks
        .by_ref()
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    if !chunks.remainder().is_empty() {
        return Err(Error::Internal(
            "cache f32 buffer has non-multiple-of-4 length".into(),
        ));
    }
    Ok(out)
}
fn f16scales_to_bytes(raw: &[u16]) -> Vec<u8> {
    let f32s: Vec<f32> = raw.iter().map(|&b| f16_to_f32(b)).collect();
    unsafe { std::slice::from_raw_parts(f32s.as_ptr() as *const u8, f32s.len() * 4).to_vec() }
}

// ── Owned layer data ─────────────────────────────────────────────────────

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

// ── Zero-copy reader ────────────────────────────────────────────────────

pub struct QCacheReader {
    _mmap: memmap2::Mmap,
    manifest: Option<QCacheManifest>,
    num_layers: usize,
    layer_offsets: Vec<(usize, usize)>,
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
        if mmap.len() < 8 || &mmap[..8] != CACHE_MAGIC {
            return Err(Error::Internal("invalid cache magic".into()));
        }
        let mut pos = 8usize;
        let (manifest, num_layers, layer_offsets) = Self::parse_header(&mmap, &mut pos)?;
        Ok(Self {
            _mmap: mmap,
            manifest,
            num_layers,
            layer_offsets,
        })
    }

    fn parse_header(
        data: &[u8],
        pos: &mut usize,
    ) -> Result<(Option<QCacheManifest>, usize, Vec<(usize, usize)>)> {
        let manifest_len = read_u64(data, pos)? as usize;
        let manifest = if manifest_len > 0 {
            let end = pos
                .checked_add(manifest_len)
                .ok_or_else(|| Error::Internal("cache manifest offset overflow".into()))?;
            if end > data.len() {
                return Err(Error::Internal("cache manifest truncated".into()));
            }
            let json = &data[*pos..end];
            *pos = end;
            let m: QCacheManifest = serde_json::from_slice(json)
                .map_err(|e| Error::Internal(format!("cache manifest json: {e}")))?;
            if m.format_version > CURRENT_FORMAT_VERSION {
                return Err(Error::Internal(format!(
                    "cache format v{} > supported v{}",
                    m.format_version, CURRENT_FORMAT_VERSION
                )));
            }
            Some(m)
        } else {
            None
        };
        let num_layers = read_u64(data, pos)? as usize;
        let mut layer_offsets = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let blob_len = read_u64(data, pos)? as usize;
            let end = pos
                .checked_add(blob_len)
                .ok_or_else(|| Error::Internal("cache layer offset overflow".into()))?;
            if end > data.len() {
                return Err(Error::Internal("cache layer truncated".into()));
            }
            layer_offsets.push((*pos, end));
            *pos = end;
        }
        Ok((manifest, num_layers, layer_offsets))
    }

    pub fn manifest(&self) -> Option<&QCacheManifest> {
        self.manifest.as_ref()
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn layer(&self, li: usize) -> Result<QCacheSlice<'_>> {
        let (start, end) = *self
            .layer_offsets
            .get(li)
            .ok_or_else(|| Error::Internal(format!("cache layer {li} out of range")))?;
        let data = &self._mmap[start..end];
        let mut pos = 0usize;
        Ok(QCacheSlice {
            qp_packed: read_buf_slice(data, &mut pos)?,
            qp_scales: read_buf_slice(data, &mut pos)?,
            kp_packed: read_buf_slice(data, &mut pos)?,
            kp_scales: read_buf_slice(data, &mut pos)?,
            vp_packed: read_buf_slice(data, &mut pos)?,
            vp_scales: read_buf_slice(data, &mut pos)?,
            op_packed: read_buf_slice(data, &mut pos)?,
            op_scales: read_buf_slice(data, &mut pos)?,
            gate_q_packed: read_buf_slice(data, &mut pos)?,
            gate_q_scales: read_buf_slice(data, &mut pos)?,
            up_q_packed: read_buf_slice(data, &mut pos)?,
            up_q_scales: read_buf_slice(data, &mut pos)?,
            down_q_packed: read_buf_slice(data, &mut pos)?,
            down_q_scales: read_buf_slice(data, &mut pos)?,
        })
    }
}

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

// ── Cache write ──────────────────────────────────────────────────────────

pub fn write_cache(
    path: &Path,
    layers: &[QCacheData],
    manifest: Option<&QCacheManifest>,
) -> Result<()> {
    use std::io::Write;
    let tmp = path.with_extension("qcache.tmp");
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(&tmp).map_err(|e| Error::Internal(format!("cache create: {e}")))?,
    );
    f.write_all(CACHE_MAGIC)
        .map_err(|e| Error::Internal(format!("cache write: {e}")))?;

    if let Some(m) = manifest {
        let json = serde_json::to_vec(m)
            .map_err(|e| Error::Internal(format!("manifest serialize: {e}")))?;
        f.write_all(&(json.len() as u64).to_le_bytes())?;
        f.write_all(&json)?;
    } else {
        f.write_all(&0u64.to_le_bytes())?;
    }

    f.write_all(&(layers.len() as u64).to_le_bytes())?;
    for q in layers {
        let blob = q.serialize_layer();
        f.write_all(&(blob.len() as u64).to_le_bytes())?;
        f.write_all(&blob)?;
    }
    f.into_inner()
        .map_err(|e| Error::Internal(format!("cache flush: {e}")))?;
    std::fs::rename(&tmp, path).map_err(|e| Error::Internal(format!("cache rename: {e}")))?;
    Ok(())
}

// ── Streaming cache writer ────────────────────────────────────────────────

/// Write cache incrementally — appends one layer at a time.
///
/// This keeps peak RAM low during first-time quantization by letting the
/// caller drop FP32 weight tensors for each layer after writing it.
pub struct StreamingCacheWriter {
    file: std::fs::File,
    /// Byte offset where the layer count placeholder was written.
    layer_count_offset: u64,
    layer_count: u64,
}

impl StreamingCacheWriter {
    /// Create a new streaming cache file and write the header.
    ///
    /// The layer count is written as a placeholder (0) and will be
    /// updated when [`finish`](Self::finish) is called.
    pub fn create(path: &Path, manifest: Option<&QCacheManifest>) -> Result<Self> {
        use std::io::{Seek, Write};
        let mut f = std::fs::File::create(path)
            .map_err(|e| Error::Internal(format!("cache create: {e}")))?;
        f.write_all(CACHE_MAGIC)
            .map_err(|e| Error::Internal(format!("cache write: {e}")))?;

        if let Some(m) = manifest {
            let json = serde_json::to_vec(m)
                .map_err(|e| Error::Internal(format!("manifest serialize: {e}")))?;
            f.write_all(&(json.len() as u64).to_le_bytes())?;
            f.write_all(&json)?;
        } else {
            f.write_all(&0u64.to_le_bytes())?;
        }

        // Record the file position for the layer count field.
        let layer_count_offset = f
            .stream_position()
            .map_err(|e| Error::Internal(format!("seek: {e}")))?;
        // Write placeholder layer count.
        f.write_all(&0u64.to_le_bytes())?;

        Ok(Self {
            file: f,
            layer_count_offset,
            layer_count: 0,
        })
    }

    /// Append a single quantized layer to the cache.
    pub fn append_layer(&mut self, data: &QCacheData) -> Result<()> {
        use std::io::Write;
        let blob = data.serialize_layer();
        self.file
            .write_all(&(blob.len() as u64).to_le_bytes())
            .map_err(|e| Error::Internal(format!("cache write: {e}")))?;
        self.file
            .write_all(&blob)
            .map_err(|e| Error::Internal(format!("cache write: {e}")))?;
        self.layer_count += 1;
        Ok(())
    }

    /// Finalize the cache: write the actual layer count and flush.
    ///
    /// After calling this, the file is a valid qcache that can be read by
    /// [`QCacheReader`].
    pub fn finish(mut self) -> Result<()> {
        use std::io::{Seek, SeekFrom, Write};
        let end = self
            .file
            .stream_position()
            .map_err(|e| Error::Internal(format!("seek: {e}")))?;
        // Seek back to the layer count placeholder and overwrite it.
        self.file
            .seek(SeekFrom::Start(self.layer_count_offset))
            .map_err(|e| Error::Internal(format!("seek: {e}")))?;
        self.file
            .write_all(&self.layer_count.to_le_bytes())
            .map_err(|e| Error::Internal(format!("cache write: {e}")))?;
        // Flush and sync.
        self.file
            .flush()
            .map_err(|e| Error::Internal(format!("cache flush: {e}")))?;
        // Truncate any trailing junk (shouldn't be any, but safe).
        self.file
            .set_len(end)
            .map_err(|e| Error::Internal(format!("cache truncate: {e}")))?;
        Ok(())
    }

    /// Return the number of layers written so far.
    pub fn len(&self) -> u64 {
        self.layer_count
    }

    /// Return true if no layers have been written yet.
    pub fn is_empty(&self) -> bool {
        self.layer_count == 0
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_layer() -> QCacheData {
        QCacheData {
            qp_packed: vec![1, 2, 3],
            qp_scales: vec![10, 20],
            kp_packed: vec![],
            kp_scales: vec![],
            vp_packed: vec![],
            vp_scales: vec![],
            op_packed: vec![],
            op_scales: vec![],
            gate_q_packed: vec![],
            gate_q_scales: vec![],
            up_q_packed: vec![],
            up_q_scales: vec![],
            down_q_packed: vec![],
            down_q_scales: vec![],
        }
    }

    #[test]
    fn manifest_serialize_roundtrip() {
        let m = QCacheManifest::new("Q4_0", "q4_0_llama", 16, "abc123", "q4_0_llama");
        let json = serde_json::to_string(&m).unwrap();
        let parsed: QCacheManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.format_version, 2);
        assert_eq!(parsed.quant_type, "Q4_0");
        assert_eq!(parsed.layout_version, "q4_0_llama");
        assert_eq!(parsed.num_layers, 16);
        assert_eq!(parsed.model_config_hash, "abc123");
        assert!(!parsed.created_at.is_empty());
    }

    #[test]
    fn write_read_v2_with_manifest() {
        let tmp = std::env::temp_dir().join("ferrule_test_v2.qcache");
        let m = QCacheManifest::new("Q4_0", "q4_0_llama", 1, "testhash", "q4_0_llama");
        write_cache(&tmp, &[empty_layer()], Some(&m)).unwrap();
        let reader = QCacheReader::open(&tmp).unwrap();
        assert_eq!(reader.num_layers(), 1);
        let got = reader.manifest().unwrap();
        assert_eq!(got.format_version, 2);
        assert_eq!(got.quant_type, "Q4_0");
        assert_eq!(got.layout_version, "q4_0_llama");
        assert_eq!(got.num_layers, 1);
        assert_eq!(got.model_config_hash, "testhash");
        let sl = reader.layer(0).unwrap();
        assert_eq!(sl.qp_packed, &[1, 2, 3]);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn read_v1_no_manifest() {
        let tmp = std::env::temp_dir().join("ferrule_test_v1.qcache");
        let mut buf = Vec::new();
        buf.extend_from_slice(b"FERRULEQ");
        buf.extend_from_slice(&0u64.to_le_bytes()); // manifest_len = 0
        buf.extend_from_slice(&1u64.to_le_bytes()); // num_layers = 1
        let mut blob = Vec::new();
        write_buf(&mut blob, &[10u8, 20, 30]); // qp_packed
        for _ in 1..14 {
            write_buf(&mut blob, &[] as &[u8]);
        }
        buf.extend_from_slice(&(blob.len() as u64).to_le_bytes());
        buf.extend_from_slice(&blob);
        std::fs::write(&tmp, &buf).unwrap();
        let reader = QCacheReader::open(&tmp).unwrap();
        assert!(reader.manifest().is_none());
        assert_eq!(reader.num_layers(), 1);
        assert_eq!(reader.layer(0).unwrap().qp_packed, &[10, 20, 30]);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn reject_future_format() {
        let tmp = std::env::temp_dir().join("ferrule_test_future.qcache");
        let mut buf = Vec::new();
        buf.extend_from_slice(b"FERRULEQ");
        let m = serde_json::json!({"format_version":99,"quant_type":"","layout_version":"","num_layers":0,"model_config_hash":"","created_at":"","quant_suffix":""});
        let json = serde_json::to_vec(&m).unwrap();
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(&json);
        buf.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&tmp, &buf).unwrap();
        assert!(QCacheReader::open(&tmp).is_err());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn cache_path_naming() {
        let p = cache_path(Path::new("/tmp/model"), "q4_0_llama");
        assert!(p.to_string_lossy().contains("model.q4_0_llama.qcache"));
    }

    // ── StreamingCacheWriter tests ─────────────────────────────────────

    #[test]
    fn streaming_writer_empty_file_readable() {
        let tmp = std::env::temp_dir().join("ferrule_test_streaming_empty.qcache");
        let writer = StreamingCacheWriter::create(&tmp, None).unwrap();
        assert!(writer.is_empty());
        assert_eq!(writer.len(), 0);
        writer.finish().unwrap();
        // Should produce a valid (but empty) cache file.
        let reader = QCacheReader::open(&tmp).unwrap();
        assert_eq!(reader.num_layers(), 0);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn streaming_writer_one_layer_with_manifest() {
        let tmp = std::env::temp_dir().join("ferrule_test_streaming_one.qcache");
        let m = QCacheManifest::new("Q4_0", "q4_0_llama", 1, "hash1", "q4_0_llama");
        let mut writer = StreamingCacheWriter::create(&tmp, Some(&m)).unwrap();
        writer.append_layer(&empty_layer()).unwrap();
        assert_eq!(writer.len(), 1);
        assert!(!writer.is_empty());
        writer.finish().unwrap();

        let reader = QCacheReader::open(&tmp).unwrap();
        assert_eq!(reader.num_layers(), 1);
        let got = reader.manifest().unwrap();
        assert_eq!(got.quant_type, "Q4_0");
        assert_eq!(got.model_config_hash, "hash1");
        let sl = reader.layer(0).unwrap();
        assert_eq!(sl.qp_packed, &[1, 2, 3]);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn streaming_writer_multiple_layers() {
        let tmp = std::env::temp_dir().join("ferrule_test_streaming_multi.qcache");
        let mut writer = StreamingCacheWriter::create(&tmp, None).unwrap();
        for _ in 0..3 {
            writer.append_layer(&empty_layer()).unwrap();
        }
        assert_eq!(writer.len(), 3);
        writer.finish().unwrap();

        let reader = QCacheReader::open(&tmp).unwrap();
        assert_eq!(reader.num_layers(), 3);
        assert!(reader.manifest().is_none());
        for li in 0..3 {
            let sl = reader.layer(li).unwrap();
            assert_eq!(sl.qp_packed, &[1, 2, 3]);
        }
        let _ = std::fs::remove_file(&tmp);
    }
}
