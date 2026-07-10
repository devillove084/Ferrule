//! `deepseek-v4-prefill-parity` command.
//!
//! Compares the batched/device prefill path against the token-loop append
//! path, reporting the first diverging layer and the top-1 token at several
//! layer-depth cut points (1L, 5L, 23L, 43L by default).

use std::path::Path;
use std::time::Instant;

use ferrule_model::{
    models::deepseek_v4::{
        DeepSeekV4OperatorBackend, DeepSeekV4ReferenceOptions, DeepSeekV4ReferenceRunner,
    },
    ChatTemplate,
};

/// Default layer-depth cut points reported by the parity harness.
const DEFAULT_CUTS: &[usize] = &[1, 5, 23, 43];

#[allow(clippy::too_many_arguments)]
pub fn cmd_deepseek_v4_prefill_parity(
    model_dir: &str,
    prompt: &str,
    max_layers: usize,
    max_tensor_mb: u64,
    expert_reader_max_slice_mb: u64,
    backend: &str,
    chat_prompt: bool,
    atol: f32,
    cuts: &[usize],
    json: bool,
) -> anyhow::Result<()> {
    let model_path = Path::new(model_dir);
    let options = DeepSeekV4ReferenceOptions {
        max_layers,
        output_head_chunk_rows: 4096,
        expert_reader_max_tensor_bytes: expert_reader_max_slice_mb.saturating_mul(1024 * 1024),
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
    };
    let operator_backend = DeepSeekV4OperatorBackend::parse(backend)?;

    let encoded_prompt = if chat_prompt {
        ChatTemplate::DeepSeekV4.format_turn(prompt, true)
    } else {
        prompt.to_string()
    };

    let load_start = Instant::now();
    let mut runner = DeepSeekV4ReferenceRunner::load_hf_with_options_and_backend(
        model_path,
        max_tensor_mb.saturating_mul(1024 * 1024),
        options,
        operator_backend,
    )?;
    let load_elapsed = load_start.elapsed();

    let token_ids = runner.model.tokenizer.encode(&encoded_prompt)?;
    if token_ids.is_empty() {
        anyhow::bail!("prompt encoded to zero tokens");
    }

    let cuts = if cuts.is_empty() { DEFAULT_CUTS } else { cuts };

    if !json {
        println!("=== DeepSeek-V4 Prefill Parity ===");
        println!("model:      {model_dir}");
        println!("backend:    {}", runner.operator_backend().as_str());
        println!("prompt:     {prompt:?}");
        if chat_prompt {
            println!("chat_prompt: {:?}", encoded_prompt);
        }
        println!("tokens:     {:?}", token_ids);
        println!("max_layers: {max_layers}");
        println!("atol:       {atol:e}");
        println!("cuts:       {:?}", cuts);
        println!("load:       {:.3} ms", load_elapsed.as_secs_f64() * 1000.0);
        println!("--- batched prefill ---");
    }

    // ── Batched trace ───────────────────────────────────────────────────
    let batched_start = Instant::now();
    let batched_trace = runner.prefill_batched_layer_hc_trace(&token_ids)?;
    let batched_elapsed = batched_start.elapsed();

    if !json {
        println!(
            "batched:    {:.3} ms, {} layers captured",
            batched_elapsed.as_secs_f64() * 1000.0,
            batched_trace.len()
        );
    }

    // ── Reset and run token-loop trace ──────────────────────────────────
    runner.reset()?;

    if !json {
        println!("--- token-loop prefill ---");
    }
    let token_loop_start = Instant::now();
    let token_loop_trace = runner.prefill_token_loop_layer_hc_trace(&token_ids)?;
    let token_loop_elapsed = token_loop_start.elapsed();

    if !json {
        println!(
            "token_loop: {:.3} ms, {} layers captured",
            token_loop_elapsed.as_secs_f64() * 1000.0,
            token_loop_trace.len()
        );
        println!("--- per-layer comparison ---");
    }

    // ── Compare layer by layer ──────────────────────────────────────────
    let n_layers = batched_trace.len().min(token_loop_trace.len());
    let mut first_diverge: Option<(usize, f32)> = None;
    let mut layer_reports: Vec<LayerReport> = Vec::with_capacity(n_layers);

    for layer in 0..n_layers {
        let b = &batched_trace[layer];
        let t = &token_loop_trace[layer];
        let max_abs_diff = max_abs_difference(b, t);
        let diverged = b.len() != t.len() || !max_abs_diff.is_finite() || max_abs_diff > atol;

        if diverged && first_diverge.is_none() {
            first_diverge = Some((layer, max_abs_diff));
        }

        layer_reports.push(LayerReport {
            layer,
            batched_len: b.len(),
            token_loop_len: t.len(),
            max_abs_diff,
            diverged,
        });
    }

    if !json {
        for report in &layer_reports {
            let flag = if report.diverged { " *** DIVERGE" } else { "" };
            println!(
                "  L{:>2}  batched_len={:<6} token_loop_len={:<6} max_abs_diff={:.6e}{flag}",
                report.layer, report.batched_len, report.token_loop_len, report.max_abs_diff
            );
        }
    }

    // ── Top-1 at each cut ───────────────────────────────────────────────
    if !json {
        println!("--- top-1 at cut points ---");
    }

    let mut cut_results: Vec<CutResult> = Vec::new();
    for &cut in cuts {
        if cut == 0 || cut > n_layers {
            continue;
        }
        let hc_idx = cut - 1;

        runner.reset()?;
        let batched_top1 = top1_from_hc(&mut runner, &batched_trace[hc_idx])?;
        runner.reset()?;
        let token_loop_top1 = top1_from_hc(&mut runner, &token_loop_trace[hc_idx])?;
        runner.reset()?;

        let match_str = if batched_top1 == token_loop_top1 {
            "MATCH"
        } else {
            "DIFFER"
        };

        if !json {
            println!(
                "  {cut:>2}L  batched=[{batched_top1}]  token_loop=[{token_loop_top1}]  {match_str}"
            );
        }

        cut_results.push(CutResult {
            cut,
            batched_top1,
            token_loop_top1,
            match_str: match_str.to_string(),
        });
    }

    // ── Summary ─────────────────────────────────────────────────────────
    if !json {
        println!("--- summary ---");
        match first_diverge {
            Some((layer, diff)) => {
                println!(
                    "first diverging layer: L{layer}  max_abs_diff={diff:.6e}  (atol={atol:e})"
                );
            }
            None => {
                println!("no divergence found across {n_layers} layers (atol={atol:e})");
            }
        }
    }

    if json {
        let summary = serde_json::json!({
            "model": model_dir,
            "backend": runner.operator_backend().as_str(),
            "prompt": prompt,
            "chat_prompt": chat_prompt,
            "prompt_tokens": token_ids,
            "max_layers": max_layers,
            "atol": atol,
            "cuts": cuts,
            "batched_ms": batched_elapsed.as_secs_f64() * 1000.0,
            "token_loop_ms": token_loop_elapsed.as_secs_f64() * 1000.0,
            "n_layers_compared": n_layers,
            "first_diverging_layer": first_diverge.map(|(l, d)| {
                serde_json::json!({
                    "layer": l,
                    "max_abs_diff": d,
                })
            }),
            "layers": layer_reports.iter().map(|r| {
                serde_json::json!({
                    "layer": r.layer,
                    "max_abs_diff": r.max_abs_diff,
                    "diverged": r.diverged,
                })
            }).collect::<Vec<_>>(),
            "cut_results": cut_results.iter().map(|c| {
                serde_json::json!({
                    "cut": c.cut,
                    "batched_top1": c.batched_top1,
                    "token_loop_top1": c.token_loop_top1,
                    "match": c.match_str,
                })
            }).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&summary)?);
    }

    Ok(())
}

struct LayerReport {
    layer: usize,
    batched_len: usize,
    token_loop_len: usize,
    max_abs_diff: f32,
    diverged: bool,
}

struct CutResult {
    cut: usize,
    batched_top1: u32,
    token_loop_top1: u32,
    match_str: String,
}

fn max_abs_difference(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn top1_from_hc(runner: &mut DeepSeekV4ReferenceRunner, hc_state: &[f32]) -> anyhow::Result<u32> {
    let topk = runner.topk_from_hc_trace(hc_state)?;
    Ok(topk.first().map(|logit| logit.token_id).unwrap_or(u32::MAX))
}
