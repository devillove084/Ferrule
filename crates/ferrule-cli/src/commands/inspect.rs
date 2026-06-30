use std::path::Path;

// ── helpers ──────────────────────────────────────────────────────────────────

fn read_u64_mmap(data: &[u8], pos: &mut usize) -> u64 {
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    v
}

fn read_u64_slice(data: &[u8], pos: &mut usize) -> u64 {
    if *pos + 8 > data.len() {
        return 0;
    }
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    v
}

// ── inspect-cache ────────────────────────────────────────────────────────────

#[allow(unsafe_code)]
pub fn cmd_inspect_cache(path: &str) -> anyhow::Result<()> {
    let p = Path::new(path);
    if !p.exists() {
        anyhow::bail!("cache file not found: {path}");
    }

    let file = std::fs::File::open(p)?;
    let data = unsafe {
        memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| anyhow::anyhow!("mmap: {e}"))?
    };

    if data.len() < 8 || &data[..8] != b"FERRULEQ" {
        anyhow::bail!("not a valid Ferrule qcache (bad magic)");
    }

    let mut pos = 8usize;

    // Read optional manifest (v2 format)
    let manifest_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
    pos += 8;
    if manifest_len > 0 {
        if let Ok(m) = serde_json::from_slice::<serde_json::Value>(&data[pos..pos + manifest_len]) {
            println!("=== qcache: {path} ===");
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
        }
        pos += manifest_len;
    } else {
        println!("=== qcache: {path} ===");
        println!("(no manifest — v1 format)");
    }

    let num_layers = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
    pos += 8;

    println!("layers:     {num_layers}");

    let mut total_bytes: u64 = 0;
    for li in 0..num_layers {
        let blob_len = read_u64_mmap(&data, &mut pos) as usize;
        total_bytes += blob_len as u64;
        let blob = &data[pos..pos + blob_len];
        let mut bp = 0usize;
        let n_fields = 14;
        let mut field_sizes = Vec::with_capacity(n_fields);
        for _ in 0..n_fields {
            if bp + 8 > blob.len() {
                break;
            }
            let flen = read_u64_slice(blob, &mut bp) as usize;
            field_sizes.push(flen);
            bp += flen;
        }
        println!("  layer {li:>3}: {blob_len:>10} bytes  (fields: {field_sizes:?})",);
        pos += blob_len;
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
}
