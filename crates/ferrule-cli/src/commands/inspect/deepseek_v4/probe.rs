use std::path::Path;
use std::time::Instant;

use ferrule_common::execution::ForwardPhase;
use ferrule_model::{
    ModelExecutionBackend,
    models::deepseek_v4::{DeepSeekV4PrepareOptions, DeepSeekV4Runner},
};

use crate::commands::resident::build_page_managed_diagnostic_harness;

use super::stats::print_deepseek_v4_runtime_stats;

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
    let options = DeepSeekV4PrepareOptions {
        max_layers,
        output_head_chunk_rows,
        expert_reader_max_tensor_bytes: expert_reader_max_slice_mb.saturating_mul(1024 * 1024),
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };

    let operator_backend = ModelExecutionBackend::parse(backend)?;
    let load_start = Instant::now();
    let runner = DeepSeekV4Runner::load_hf_with_options_and_backend(
        model_path,
        max_tensor_mb.saturating_mul(1024 * 1024),
        options,
        operator_backend,
    )?;
    let load_elapsed = load_start.elapsed();

    let token_ids = runner.model().tokenizer.encode(prompt)?;
    if token_ids.is_empty() {
        anyhow::bail!("prompt encoded to zero tokens");
    }

    println!("=== DeepSeek-V4 Probe ===");
    println!("model:      {model_dir}");
    println!("prompt:     {prompt:?}");
    println!("tokens:     {:?}", token_ids);
    println!(
        "config:     hidden={} layers={} vocab={} heads={} head_dim={} window={} compress_layers={}",
        runner.model().config.hidden_size,
        runner.model().config.num_layers,
        runner.model().config.vocab_size,
        runner.model().config.num_heads,
        runner.model().config.head_dim,
        runner.model().config.window_size,
        runner
            .model()
            .config
            .compress_ratios
            .iter()
            .filter(|&&ratio| ratio != 0)
            .count()
    );
    println!(
        "mode:       max_layers={}{}",
        max_layers,
        if max_layers < runner.model().config.num_layers {
            " (partial/reference)"
        } else {
            " (full artifact path; sequential diagnostic prefill)"
        }
    );
    println!("backend:    {}", runner.operator_backend().as_str());
    if max_layers > 0 {
        if runner.operator_backend() == ModelExecutionBackend::Cuda {
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

    let schema = runner.kv_layout_schema().clone();
    let mut diagnostic =
        build_page_managed_diagnostic_harness(runner, Box::new(schema), token_ids.len(), 1)?;
    let sequence = diagnostic.create_sequence(0)?;

    let run_start = Instant::now();
    if full_vocab_topk {
        let top = diagnostic.execute_sequence_step(
            sequence,
            ForwardPhase::Prefill,
            &token_ids,
            |runner| runner.prefill_tokens_topk_batched(&token_ids, top_k),
        )?;
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
        println!("position:   {}", token_ids.len());
        println!("bound layers: {}", diagnostic.runner().bound_layer_count());
        print_deepseek_v4_runtime_stats(diagnostic.runner());
        println!(
            "run:        {:.3} ms (batched prefill + full-vocab top-{top_k})",
            elapsed.as_secs_f64() * 1000.0
        );
        println!("top logits:");
        for item in top {
            let piece = diagnostic
                .runner()
                .model()
                .tokenizer
                .decode(&[item.token_id])
                .unwrap_or_else(|_| String::new());
            println!("  {:>8}: {:>12.6}  {:?}", item.token_id, item.logit, piece);
        }
    } else {
        let logits = diagnostic.execute_sequence_step(
            sequence,
            ForwardPhase::Prefill,
            &token_ids,
            |runner| {
                runner.prefill_tokens_logits_row_range_batched(&token_ids, start_row, row_count)
            },
        )?;
        let row_logits = logits
            .iter()
            .copied()
            .enumerate()
            .map(|(offset, logit)| ferrule_model::TokenLogit {
                token_id: (start_row + offset) as u32,
                logit,
            })
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
        println!("position:   {}", token_ids.len());
        println!("bound layers: {}", diagnostic.runner().bound_layer_count());
        print_deepseek_v4_runtime_stats(diagnostic.runner());
        println!(
            "run:        {:.3} ms (batched prefill + lm_head rows [{}, {}))",
            elapsed.as_secs_f64() * 1000.0,
            start_row,
            start_row + row_count
        );
        println!("row logits:");
        for (offset, logit) in logits.iter().enumerate() {
            let token_id = (start_row + offset) as u32;
            let piece = diagnostic
                .runner()
                .model()
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
                let piece = diagnostic
                    .runner()
                    .model()
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
    row_logits: Option<&[ferrule_model::TokenLogit]>,
    top_logits: Option<&[ferrule_model::TokenLogit]>,
    atol: f32,
) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&text)?;
    if let Some(expected_prompt) = json.get("prompt").and_then(|value| value.as_str())
        && expected_prompt != prompt
    {
        anyhow::bail!(
            "reference prompt mismatch: expected {:?}, got {:?}",
            expected_prompt,
            prompt
        );
    }
    if let Some(expected_layers) = json.get("max_layers").and_then(|value| value.as_u64())
        && expected_layers as usize != max_layers
    {
        anyhow::bail!(
            "reference max_layers mismatch: expected {}, got {}",
            expected_layers,
            max_layers
        );
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
    if let Some(actual) = row_logits
        && let Some(expected) = json.get("row_logits")
    {
        compare_logit_array(expected, actual, atol, "row_logits")?;
        println!("reference: row_logits matched within atol={atol}");
    }
    if let Some(actual) = top_logits
        && let Some(expected) = json.get("top_logits")
    {
        compare_logit_array(expected, actual, atol, "top_logits")?;
        println!("reference: top_logits matched within atol={atol}");
    }
    Ok(())
}

fn compare_logit_array(
    expected: &serde_json::Value,
    actual: &[ferrule_model::TokenLogit],
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
