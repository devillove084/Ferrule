use std::path::Path;
use std::time::Instant;

use ferrule_model::{
    ExpertComputeBundle, ExpertId, ExpertLoadSource, ExpertStreamingPlanner, ExpertStreamingPolicy,
    ExpertStreamingReader, HfSafetensorsInventory, ModelDescriptor,
};

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
    let routed = inventory.routed_expert_tensors();
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
    let ExpertLoadSource::LocalTensorSet { tensors } = &load.load_source else {
        anyhow::bail!("selected expert load source is not a local tensor set");
    };

    println!("=== Expert Streaming Smoke ===");
    println!("model:      {model_dir}");
    println!("family:     {family}");
    println!("expert sets registered: {expert_sets}");
    println!("selected:   layer={layer} expert={expert}");
    println!("slices:     {}", tensors.len());
    println!(
        "bytes:      {} ({:.3} MiB)",
        load.load_source.bytes(),
        load.load_source.bytes() as f64 / 1_048_576.0
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

    let reader = ExpertStreamingReader::from_env(max_slice_mb.saturating_mul(1024 * 1024))?;
    println!("io backend:  {}", reader.backend_name());
    let start = Instant::now();
    let payload = reader.read_load_source(load.expert, &load.load_source)?;
    let elapsed = start.elapsed();
    let read_bytes = payload
        .tensors
        .iter()
        .map(|tensor| tensor.bytes.len() as u64)
        .sum::<u64>();
    let payload_checksum = payload
        .tensors
        .iter()
        .flat_map(|tensor| tensor.bytes.iter().copied())
        .fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        });
    let repeat_start = Instant::now();
    let repeat = reader.read_load_source(load.expert, &load.load_source)?;
    let repeat_elapsed = repeat_start.elapsed();
    let repeat_checksum = repeat
        .tensors
        .iter()
        .flat_map(|tensor| tensor.bytes.iter().copied())
        .fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        });
    if repeat_checksum != payload_checksum {
        anyhow::bail!(
            "repeat expert payload checksum mismatch: {repeat_checksum:016x} != {payload_checksum:016x}"
        );
    }
    let bundle = ExpertComputeBundle::from_artifact_payload(payload)?;
    println!(
        "read:       {} bytes ({:.3} MiB) in {:.3} ms",
        read_bytes,
        read_bytes as f64 / 1_048_576.0,
        elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "repeat:     {:.3} ms (checksum verified)",
        repeat_elapsed.as_secs_f64() * 1000.0
    );
    println!("checksum:   fnv1a64={payload_checksum:016x}");
    println!("bundle:     gate={:?}", bundle.gate.format);
    println!("            up={:?}", bundle.up.format);
    println!("            down={:?}", bundle.down.format);
    Ok(())
}
