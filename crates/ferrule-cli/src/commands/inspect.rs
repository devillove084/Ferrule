use std::path::Path;
use std::time::Instant;

use ferrule_model::{HfSafetensorsInventory, ModelDescriptor};
use ferrule_runtime::{
    ExpertComputeBundle, ExpertId, ExpertSource, ExpertStreamingPlanner, ExpertStreamingPolicy,
    ExpertStreamingReader,
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn read_u64_mmap(data: &[u8], pos: &mut usize, label: &str) -> anyhow::Result<u64> {
    let end = pos
        .checked_add(8)
        .ok_or_else(|| anyhow::anyhow!("{label} offset overflow"))?;
    let bytes = data
        .get(*pos..end)
        .ok_or_else(|| anyhow::anyhow!("{label} truncated"))?;
    let v = u64::from_le_bytes(bytes.try_into().unwrap());
    *pos = end;
    Ok(v)
}

fn read_slice<'a>(
    data: &'a [u8],
    pos: &mut usize,
    len: usize,
    label: &str,
) -> anyhow::Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| anyhow::anyhow!("{label} offset overflow"))?;
    let slice = data
        .get(*pos..end)
        .ok_or_else(|| anyhow::anyhow!("{label} truncated"))?;
    *pos = end;
    Ok(slice)
}

fn read_u64_slice(data: &[u8], pos: &mut usize) -> u64 {
    if *pos + 8 > data.len() {
        return 0;
    }
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    v
}

// ── expert-stream-smoke ──────────────────────────────────────────────────────

pub fn cmd_expert_stream_smoke(
    model_dir: &str,
    layer: usize,
    expert: usize,
    max_slice_mb: u64,
) -> anyhow::Result<()> {
    let model_path = Path::new(model_dir);
    let descriptor = ModelDescriptor::load(model_path)?;
    let family = descriptor.spec.family.clone();
    let top_k = descriptor.spec.moe.num_experts_per_tok.unwrap_or(1).max(1);
    let inventory = HfSafetensorsInventory::open(model_path, family.clone())?;
    let routed = inventory.routed_expert_tensors(&family);
    if routed.is_empty() {
        anyhow::bail!("no routed expert tensor slices found for family {family}");
    }

    let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first(top_k));
    let expert_sets = planner.register_hf_routed_expert_tensor_sets(model_path, routed)?;
    let step = planner.plan_layer_step(layer, &[expert], &[])?;
    let load = step
        .loads
        .iter()
        .find(|load| load.expert == ExpertId::new(layer, expert))
        .ok_or_else(|| anyhow::anyhow!("selected expert was already resident or not planned"))?;
    let ExpertSource::LocalTensorSet { tensors } = &load.source else {
        anyhow::bail!("selected expert source is not a local tensor set");
    };

    println!("=== Expert Streaming Smoke ===");
    println!("model:      {model_dir}");
    println!("family:     {family}");
    println!("expert sets registered: {expert_sets}");
    println!("selected:   layer={layer} expert={expert}");
    println!("slices:     {}", tensors.len());
    println!(
        "bytes:      {} ({:.3} MiB)",
        load.source.bytes(),
        load.source.bytes() as f64 / 1_048_576.0
    );
    for tensor in tensors {
        println!(
            "  {:?}/{:?}: {} bytes dtype={} shape={:?} path={} offset={}",
            tensor.key.matrix,
            tensor.component,
            tensor.bytes,
            tensor.dtype,
            tensor.shape,
            tensor.path.display(),
            tensor.offset
        );
    }

    let reader = ExpertStreamingReader::new(max_slice_mb.saturating_mul(1024 * 1024));
    let start = Instant::now();
    let payload = reader.read_source(load.expert, &load.source)?;
    let elapsed = start.elapsed();
    let read_bytes = payload
        .tensors
        .iter()
        .map(|tensor| tensor.bytes.len() as u64)
        .sum::<u64>();
    let bundle = ExpertComputeBundle::from_source_payload(payload)?;
    println!(
        "read:       {} bytes ({:.3} MiB) in {:.3} ms",
        read_bytes,
        read_bytes as f64 / 1_048_576.0,
        elapsed.as_secs_f64() * 1000.0
    );
    println!("bundle:     gate={:?}", bundle.gate.format);
    println!("            up={:?}", bundle.up.format);
    println!("            down={:?}", bundle.down.format);
    Ok(())
}

// ── inspect-weightpack ───────────────────────────────────────────────────────

#[allow(unsafe_code)]
pub fn cmd_inspect_weightpack(path: &str) -> anyhow::Result<()> {
    let p = Path::new(path);
    if !p.exists() {
        anyhow::bail!("weight pack file not found: {path}");
    }

    let file = std::fs::File::open(p)?;
    let data = unsafe {
        memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| anyhow::anyhow!("mmap: {e}"))?
    };

    if data.len() < 8 || &data[..8] != b"FERRULEW" {
        anyhow::bail!("not a valid Ferrule WeightPack (bad magic)");
    }

    let mut pos = 8usize;

    // Read optional manifest (v2+ format).
    let manifest_len = read_u64_mmap(&data, &mut pos, "manifest length")? as usize;
    if manifest_len > 0 {
        let manifest = read_slice(&data, &mut pos, manifest_len, "manifest")?;
        let m = serde_json::from_slice::<serde_json::Value>(manifest)
            .map_err(|e| anyhow::anyhow!("manifest json: {e}"))?;
        println!("=== WeightPack: {path} ===");
        println!("manifest:");
        if let Some(v) = m.get("format_version") {
            println!("  format_version: {v}");
        }
        if let Some(v) = m.get("quant_type") {
            println!("  quant_type: {v}");
        }
        if let Some(v) = m.get("layout_version") {
            println!("  layout_version: {v}");
        }
        if let Some(v) = m.get("model_config_hash") {
            println!("  model_config_hash: {v}");
        }
        if let Some(v) = m.get("created_at") {
            println!("  created_at: {v}");
        }
        if let Some(v) = m.get("quant_suffix") {
            println!("  quant_suffix: {v}");
        }
        if let Some(arr) = m.get("tensor_shapes").and_then(|v| v.as_array()) {
            println!("  tensor_shapes: {} entries", arr.len());
        }
    } else {
        println!("=== WeightPack: {path} ===");
        println!("(no manifest — v1 format)");
    }

    let num_layers = read_u64_mmap(&data, &mut pos, "layer count")? as usize;
    println!("layers:     {num_layers}");

    let mut total_bytes: u64 = 0;
    for li in 0..num_layers {
        let blob_len = read_u64_mmap(&data, &mut pos, "layer blob length")? as usize;
        total_bytes += blob_len as u64;
        let blob = read_slice(&data, &mut pos, blob_len, "layer blob")?;
        let mut bp = 0usize;
        let n_fields = 14;
        let mut field_sizes = Vec::with_capacity(n_fields);
        for _ in 0..n_fields {
            if bp + 8 > blob.len() {
                break;
            }
            let flen = read_u64_slice(blob, &mut bp) as usize;
            field_sizes.push(flen);
            bp = bp.saturating_add(flen).min(blob.len());
        }
        println!("  layer {li:>3}: {blob_len:>10} bytes  (fields: {field_sizes:?})",);
    }
    println!(
        "total:      {} bytes ({:.1} MB)",
        total_bytes,
        total_bytes as f64 / 1_048_576.0
    );
    println!("file size:  {} bytes", data.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_u64_slice_valid() {
        let data = [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF];
        let mut pos = 0;
        let v = read_u64_slice(&data, &mut pos);
        assert_eq!(v, 1);
        assert_eq!(pos, 8);
    }

    #[test]
    fn test_read_u64_slice_short() {
        let data = [0x01, 0x02, 0x03];
        let mut pos = 0;
        let v = read_u64_slice(&data, &mut pos);
        assert_eq!(v, 0);
        assert_eq!(pos, 0);
    }

    #[test]
    fn test_read_u64_mmap_truncated() {
        let data = [0x01, 0x02, 0x03];
        let mut pos = 0;
        assert!(read_u64_mmap(&data, &mut pos, "test").is_err());
        assert_eq!(pos, 0);
    }
}
