use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use ferrule_bench::{RuntimeBenchSummary, RuntimeCounters};
use ferrule_model::{HfSafetensorsInventory, ModelDescriptor};
use ferrule_runtime::{
    models::deepseek_v4::{
        DeepSeekV4OperatorBackend, DeepSeekV4ReferenceOptions, DeepSeekV4ReferenceRunner,
    },
    ChatTemplate, ExpertComputeBundle, ExpertId, ExpertLoadSource, ExpertStreamingPlanner,
    ExpertStreamingPolicy, ExpertStreamingReader,
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

    let reader = ExpertStreamingReader::new(max_slice_mb.saturating_mul(1024 * 1024));
    let start = Instant::now();
    let payload = reader.read_load_source(load.expert, &load.load_source)?;
    let elapsed = start.elapsed();
    let read_bytes = payload
        .tensors
        .iter()
        .map(|tensor| tensor.bytes.len() as u64)
        .sum::<u64>();
    let bundle = ExpertComputeBundle::from_artifact_payload(payload)?;
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

// ── deepseek-v4-probe / generate ─────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn cmd_deepseek_v4_generate(
    model_dir: &str,
    prompt: &str,
    max_new_tokens: usize,
    max_layers: usize,
    output_head_chunk_rows: usize,
    max_tensor_mb: u64,
    expert_reader_max_slice_mb: u64,
    backend: &str,
    stop_at_eos: bool,
    verbose_tokens: bool,
    chat_prompt: bool,
    json: bool,
) -> anyhow::Result<()> {
    let model_path = Path::new(model_dir);
    let options = DeepSeekV4ReferenceOptions {
        max_layers,
        output_head_chunk_rows,
        expert_reader_max_tensor_bytes: expert_reader_max_slice_mb.saturating_mul(1024 * 1024),
    };
    let operator_backend = DeepSeekV4OperatorBackend::parse(backend)?;
    let load_start = Instant::now();
    let mut runner = DeepSeekV4ReferenceRunner::load_hf_with_options_and_backend(
        model_path,
        max_tensor_mb.saturating_mul(1024 * 1024),
        options,
        operator_backend,
    )?;
    let load_elapsed = load_start.elapsed();

    let encoded_prompt = if chat_prompt {
        ChatTemplate::DeepSeekV4.format_turn(prompt, true)
    } else {
        prompt.to_string()
    };
    let prompt_tokens = runner.model.tokenizer.encode(&encoded_prompt)?;
    if prompt_tokens.is_empty() {
        anyhow::bail!("prompt encoded to zero tokens");
    }

    if !json {
        println!("=== DeepSeek-V4 Generate ===");
        println!("model:      {model_dir}");
        println!("backend:    {}", runner.operator_backend().as_str());
        println!("prompt:     {prompt:?}");
        if chat_prompt {
            println!("chat_prompt: {:?}", encoded_prompt);
        }
        println!("tokens:     {:?}", prompt_tokens);
        println!("max_new:   {max_new_tokens}");
        println!("max_layers: {max_layers}");
        println!("load:       {:.3} ms", load_elapsed.as_secs_f64() * 1000.0);
        println!("--- output ---");
    }

    let mut generated = Vec::new();
    let mut prefill_elapsed = Duration::ZERO;
    let mut decode_elapsed = Duration::ZERO;

    if max_new_tokens > 0 {
        let prefill_start = Instant::now();
        let mut top = runner.prefill_tokens_topk_batched(&prompt_tokens, 1)?;
        prefill_elapsed = prefill_start.elapsed();
        if top.is_empty() {
            anyhow::bail!("DSV4 generation produced no candidate after prefill");
        }

        let eos = runner.model.tokenizer.eos_token_id();
        let decode_start = Instant::now();
        for step in 0..max_new_tokens {
            let next = top[0];
            generated.push(next.token_id);
            if verbose_tokens {
                eprintln!("[{}] token={} logit={:.6}", step, next.token_id, next.logit);
            }
            if !json {
                let piece = runner
                    .model
                    .tokenizer
                    .decode(&[next.token_id])
                    .unwrap_or_else(|_| String::new());
                print!("{piece}");
                std::io::stdout().flush()?;
            }

            if stop_at_eos && Some(next.token_id) == eos {
                break;
            }
            if step + 1 == max_new_tokens {
                break;
            }
            top = runner.decode_token_topk(next.token_id, 1)?;
            if top.is_empty() {
                anyhow::bail!(
                    "DSV4 generation produced no candidate at decode step {}",
                    step + 1
                );
            }
        }
        decode_elapsed = decode_start.elapsed();
    }

    let elapsed = prefill_elapsed + decode_elapsed;
    if json {
        let layer_stats = runner.layer_runtime_stats();
        let resident_experts = layer_stats.iter().map(|stat| stat.resident_experts).sum();
        let resident_bytes = layer_stats
            .iter()
            .map(|stat| stat.resident_expert_bytes)
            .sum();
        let layers = layer_stats
            .iter()
            .map(|stat| {
                serde_json::json!({
                    "layer": stat.layer,
                    "window_kv_len": stat.window_kv_len,
                    "compressed_kv_len": stat.compressed_kv_len,
                    "indexer_compressed_kv_len": stat.indexer_compressed_kv_len,
                    "resident_experts": stat.resident_experts,
                    "resident_expert_bytes": stat.resident_expert_bytes,
                })
            })
            .collect::<Vec<_>>();
        let op_counters = runner.operator_runtime_counters();
        let mut counters = RuntimeCounters::default();
        counters.record_load(load_elapsed);
        counters.record_prefill(prefill_elapsed);
        counters.record_decode(decode_elapsed);
        counters.record_kernel_launches(op_counters.kernel_launches);
        counters.transfers.host_to_device_copies = op_counters.host_to_device_copies;
        counters.transfers.host_to_device_bytes = op_counters.host_to_device_bytes;
        counters.transfers.device_to_host_copies = op_counters.device_to_host_copies;
        counters.transfers.device_to_host_bytes = op_counters.device_to_host_bytes;
        counters.record_artifact_uploads(
            op_counters.artifact_uploads,
            op_counters.artifact_upload_bytes,
        );
        counters.record_selected_experts(op_counters.expert_selected);
        counters.record_expert_loads(op_counters.expert_loads, op_counters.expert_load_bytes);
        counters.record_expert_evictions(op_counters.expert_evictions);
        counters.set_expert_residency(resident_experts, resident_bytes);
        let summary =
            RuntimeBenchSummary::new(None, None, counters, prompt_tokens.len(), generated.len());
        let out = serde_json::json!({
            "model": model_dir,
            "backend": runner.operator_backend().as_str(),
            "prompt": prompt,
            "prompt_tokens": prompt_tokens.len(),
            "prompt_token_ids": prompt_tokens,
            "generated_tokens": generated.len(),
            "generated_token_ids": generated,
            "max_layers": max_layers,
            "bound_layers": runner.bound_layer_count(),
            "position": runner.position(),
            "layers": layers,
            "load_seconds": load_elapsed.as_secs_f64(),
            "prefill_seconds": prefill_elapsed.as_secs_f64(),
            "decode_seconds": decode_elapsed.as_secs_f64(),
            "total_seconds": elapsed.as_secs_f64(),
            "summary": summary,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!();
        println!("--- stats ---");
        println!("generated_tokens: {:?}", generated);
        println!("position:   {}", runner.position());
        println!("bound layers: {}", runner.bound_layer_count());
        print_deepseek_v4_runtime_stats(&runner);
        println!("run:        {:.3} ms", elapsed.as_secs_f64() * 1000.0);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn cmd_deepseek_v4_probe(
    model_dir: &str,
    prompt: &str,
    max_layers: usize,
    start_row: usize,
    row_count: usize,
    top_k: usize,
    full_vocab_topk: bool,
    output_head_chunk_rows: usize,
    max_tensor_mb: u64,
    expert_reader_max_slice_mb: u64,
    backend: &str,
    reference_json: Option<&str>,
    reference_atol: f32,
) -> anyhow::Result<()> {
    let model_path = Path::new(model_dir);
    let options = DeepSeekV4ReferenceOptions {
        max_layers,
        output_head_chunk_rows,
        expert_reader_max_tensor_bytes: expert_reader_max_slice_mb.saturating_mul(1024 * 1024),
    };

    let operator_backend = DeepSeekV4OperatorBackend::parse(backend)?;
    let load_start = Instant::now();
    let mut runner = DeepSeekV4ReferenceRunner::load_hf_with_options_and_backend(
        model_path,
        max_tensor_mb.saturating_mul(1024 * 1024),
        options,
        operator_backend,
    )?;
    let load_elapsed = load_start.elapsed();

    let token_ids = runner.model.tokenizer.encode(prompt)?;
    if token_ids.is_empty() {
        anyhow::bail!("prompt encoded to zero tokens");
    }

    println!("=== DeepSeek-V4 Probe ===");
    println!("model:      {model_dir}");
    println!("prompt:     {prompt:?}");
    println!("tokens:     {:?}", token_ids);
    println!(
        "config:     hidden={} layers={} vocab={} heads={} head_dim={} window={} compress_layers={}",
        runner.model.config.hidden_size,
        runner.model.config.num_layers,
        runner.model.config.vocab_size,
        runner.model.config.num_heads,
        runner.model.config.head_dim,
        runner.model.config.window_size,
        runner
            .model
            .config
            .compress_ratios
            .iter()
            .filter(|&&ratio| ratio != 0)
            .count()
    );
    println!(
        "mode:       max_layers={}{}",
        max_layers,
        if max_layers < runner.model.config.num_layers {
            " (partial/reference)"
        } else {
            " (full artifact path; sequential diagnostic prefill)"
        }
    );
    println!("backend:    {}", runner.operator_backend().as_str());
    if max_layers > 0 {
        if runner.operator_backend() == DeepSeekV4OperatorBackend::Cuda {
            println!(
                "note:       cuda reuses one cuda-oxide context/module and cached artifact-linear/expert handles; CPU is only the explicit reference backend"
            );
        } else {
            println!(
                "note:       CPU layer execution dequantizes artifact tensors and streams selected experts; use small prompts/layers for diagnostics"
            );
        }
    }
    println!("load:       {:.3} ms", load_elapsed.as_secs_f64() * 1000.0);

    let run_start = Instant::now();
    if full_vocab_topk {
        let top = runner.prefill_tokens_topk_batched(&token_ids, top_k)?;
        if let Some(path) = reference_json {
            compare_deepseek_v4_probe_reference(
                path,
                prompt,
                &token_ids,
                max_layers,
                None,
                Some(&top),
                reference_atol,
            )?;
        }
        let elapsed = run_start.elapsed();
        println!("position:   {}", runner.position());
        println!("bound layers: {}", runner.bound_layer_count());
        print_deepseek_v4_runtime_stats(&runner);
        println!(
            "run:        {:.3} ms (batched prefill + full-vocab top-{top_k})",
            elapsed.as_secs_f64() * 1000.0
        );
        println!("top logits:");
        for item in top {
            let piece = runner
                .model
                .tokenizer
                .decode(&[item.token_id])
                .unwrap_or_else(|_| String::new());
            println!("  {:>8}: {:>12.6}  {:?}", item.token_id, item.logit, piece);
        }
    } else {
        let logits =
            runner.prefill_tokens_logits_row_range_batched(&token_ids, start_row, row_count)?;
        let row_logits = logits
            .iter()
            .copied()
            .enumerate()
            .map(
                |(offset, logit)| ferrule_runtime::models::deepseek_v4::DeepSeekV4Logit {
                    token_id: (start_row + offset) as u32,
                    logit,
                },
            )
            .collect::<Vec<_>>();
        if let Some(path) = reference_json {
            compare_deepseek_v4_probe_reference(
                path,
                prompt,
                &token_ids,
                max_layers,
                Some(&row_logits),
                None,
                reference_atol,
            )?;
        }
        let elapsed = run_start.elapsed();
        println!("position:   {}", runner.position());
        println!("bound layers: {}", runner.bound_layer_count());
        print_deepseek_v4_runtime_stats(&runner);
        println!(
            "run:        {:.3} ms (batched prefill + lm_head rows [{}, {}))",
            elapsed.as_secs_f64() * 1000.0,
            start_row,
            start_row + row_count
        );
        println!("row logits:");
        for (offset, logit) in logits.iter().enumerate() {
            let token_id = (start_row + offset) as u32;
            let piece = runner
                .model
                .tokenizer
                .decode(&[token_id])
                .unwrap_or_else(|_| String::new());
            println!("  {:>8}: {:>12.6}  {:?}", token_id, logit, piece);
        }
        if top_k > 0 {
            let mut ranked = logits
                .iter()
                .copied()
                .enumerate()
                .map(|(offset, logit)| ((start_row + offset) as u32, logit))
                .collect::<Vec<_>>();
            ranked.sort_by(|(left_id, left), (right_id, right)| {
                right.total_cmp(left).then_with(|| left_id.cmp(right_id))
            });
            ranked.truncate(top_k.min(ranked.len()));
            println!("top logits in row range:");
            for (token_id, logit) in ranked {
                let piece = runner
                    .model
                    .tokenizer
                    .decode(&[token_id])
                    .unwrap_or_else(|_| String::new());
                println!("  {:>8}: {:>12.6}  {:?}", token_id, logit, piece);
            }
        }
    }

    Ok(())
}

fn compare_deepseek_v4_probe_reference(
    path: &str,
    prompt: &str,
    token_ids: &[u32],
    max_layers: usize,
    row_logits: Option<&[ferrule_runtime::models::deepseek_v4::DeepSeekV4Logit]>,
    top_logits: Option<&[ferrule_runtime::models::deepseek_v4::DeepSeekV4Logit]>,
    atol: f32,
) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&text)?;
    if let Some(expected_prompt) = json.get("prompt").and_then(|value| value.as_str()) {
        if expected_prompt != prompt {
            anyhow::bail!(
                "reference prompt mismatch: expected {:?}, got {:?}",
                expected_prompt,
                prompt
            );
        }
    }
    if let Some(expected_layers) = json.get("max_layers").and_then(|value| value.as_u64()) {
        if expected_layers as usize != max_layers {
            anyhow::bail!(
                "reference max_layers mismatch: expected {}, got {}",
                expected_layers,
                max_layers
            );
        }
    }
    if let Some(expected_tokens) = json.get("tokens").and_then(|value| value.as_array()) {
        let expected = expected_tokens
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .map(|value| value as u32)
                    .ok_or_else(|| anyhow::anyhow!("reference token ids must be u32 integers"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        if expected != token_ids {
            anyhow::bail!(
                "reference token ids mismatch: expected {:?}, got {:?}",
                expected,
                token_ids
            );
        }
    }
    if let Some(actual) = row_logits {
        if let Some(expected) = json.get("row_logits") {
            compare_logit_array(expected, actual, atol, "row_logits")?;
            println!("reference: row_logits matched within atol={atol}");
        }
    }
    if let Some(actual) = top_logits {
        if let Some(expected) = json.get("top_logits") {
            compare_logit_array(expected, actual, atol, "top_logits")?;
            println!("reference: top_logits matched within atol={atol}");
        }
    }
    Ok(())
}

fn compare_logit_array(
    expected: &serde_json::Value,
    actual: &[ferrule_runtime::models::deepseek_v4::DeepSeekV4Logit],
    atol: f32,
    label: &str,
) -> anyhow::Result<()> {
    let expected = expected
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("reference {label} must be an array"))?;
    if expected.len() != actual.len() {
        anyhow::bail!(
            "reference {label} length mismatch: expected {}, got {}",
            expected.len(),
            actual.len()
        );
    }
    for (idx, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
        let token_id = expected
            .get("token_id")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| anyhow::anyhow!("reference {label}[{idx}].token_id missing/u64"))?
            as u32;
        let logit = expected
            .get("logit")
            .and_then(|value| value.as_f64())
            .ok_or_else(|| anyhow::anyhow!("reference {label}[{idx}].logit missing/f64"))?
            as f32;
        if token_id != actual.token_id {
            anyhow::bail!(
                "reference {label}[{idx}] token mismatch: expected {}, got {}",
                token_id,
                actual.token_id
            );
        }
        let diff = (logit - actual.logit).abs();
        if diff > atol {
            anyhow::bail!(
                "reference {label}[{idx}] logit mismatch for token {}: expected {:.8}, got {:.8}, diff {:.8} > atol {:.8}",
                token_id,
                logit,
                actual.logit,
                diff,
                atol
            );
        }
    }
    Ok(())
}

fn print_deepseek_v4_runtime_stats(runner: &DeepSeekV4ReferenceRunner) {
    let stats = runner.layer_runtime_stats();
    if stats.is_empty() {
        return;
    }
    println!("layer stats:");
    for stat in stats {
        println!(
            "  L{:>2}: window_kv={} compressed_kv={} indexer_kv={} resident_experts={} resident_bytes={}",
            stat.layer,
            stat.window_kv_len,
            stat.compressed_kv_len,
            stat.indexer_compressed_kv_len,
            stat.resident_experts,
            stat.resident_expert_bytes,
        );
    }
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
