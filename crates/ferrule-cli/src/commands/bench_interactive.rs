//! Interactive multi-turn benchmark for the DSV4 chat path.
//!
//! Feeds a fixed set of user turns through the interactive prefill/decode
//! pipeline (the same code path as `ferrule chat`) and reports:
//!
//! - time-to-REPL / artifact load latency
//! - per-turn prefill and decode wall time
//! - generated tokens per turn and aggregate decode tok/s
//! - resident expert and host-cache counters
//! - optional golden-trace comparison

#[cfg(feature = "cuda")]
use std::path::Path;
#[cfg(feature = "cuda")]
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use crate::bench::{compare_interactive_trace, GoldenTurn, InteractiveTrace};
#[cfg(feature = "cuda")]
use ferrule_runtime::{
    models::deepseek_v4::{
        DeepSeekV4ArtifactModel, DeepSeekV4OperatorBackend, DeepSeekV4ReferenceOptions,
        DeepSeekV4ReferenceRunner,
    },
    ChatTemplate, GenerationConfig, ModelRunner,
};

/// A single turn measurement captured by the interactive benchmark.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default)]
struct InteractiveTurnMeasurement {
    prompt_text: String,
    prompt_tokens: Vec<u32>,
    prefill_us: u64,
    decode_us: u64,
    generated_tokens: Vec<u32>,
    stopped_by_eos: bool,
    stopped_by_string: Option<String>,
}

/// Full interactive benchmark report.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default)]
struct InteractiveBenchReport {
    model_dir: String,
    chat_template: String,
    max_new_tokens: usize,
    /// Wall time from load start to runner ready.
    artifact_load_us: u64,
    /// Wall time from REPL start to first prompt response ready (first-turn prefill).
    time_to_first_token_us: u64,
    turns: Vec<InteractiveTurnMeasurement>,
    /// Aggregate decode tokens per second across all turns.
    aggregate_decode_tok_per_s: f64,
    /// Total generated tokens across all turns.
    total_generated: usize,
    /// Resident experts at end of run (sum across layers).
    resident_experts: usize,
    /// Resident expert bytes at end of run.
    resident_expert_bytes: u64,
    /// Expert loads during the timed run.
    expert_loads: u64,
    /// Expert load bytes.
    expert_load_bytes: u64,
    /// Expert evictions.
    expert_evictions: u64,
    /// Host cache entries at end of run.
    host_cache_entries: usize,
    /// Host cache bytes at end of run.
    host_cache_bytes: u64,
}

#[cfg(feature = "cuda")]
pub fn cmd_bench_interactive(
    model_dir: &str,
    prompts: &[String],
    max_new_tokens: usize,
    chat_template_override: Option<&str>,
    golden_trace_path: Option<&str>,
    json: bool,
) -> anyhow::Result<()> {
    let model_path = Path::new(model_dir);
    let chat_template = if let Some(name) = chat_template_override {
        ChatTemplate::from_name(name).unwrap_or(ChatTemplate::DeepSeekV4)
    } else {
        ChatTemplate::DeepSeekV4
    };

    let gen_cfg = GenerationConfig {
        max_new_tokens,
        stop: Vec::new(),
        logprobs_k: 0,
        ctx_size: 4096,
        append_eos_to_session: true,
        ..GenerationConfig::default()
    };

    let options = DeepSeekV4ReferenceOptions {
        max_layers: 43,
        output_head_chunk_rows: 4096,
        moe_prefetch_experts: 8,
        moe_hotset_experts: 0,
        ..DeepSeekV4ReferenceOptions::default()
    };

    // ── Phase 1: load ────────────────────────────────────────────────────
    let load_start = Instant::now();
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(model_path, 128 * 1024 * 1024)?;
    let mut runner = DeepSeekV4ReferenceRunner::new_with_operator_backend(
        model,
        options,
        DeepSeekV4OperatorBackend::Cuda,
    )?;
    let artifact_load_us = duration_us(load_start.elapsed());

    // Capture baseline counters after load (before any prefill/decode).
    let counters_baseline = runner.operator_runtime_counters();
    let eos = runner.eos_token_id();

    // ── Phase 2: run turns ───────────────────────────────────────────────
    let mut report = InteractiveBenchReport {
        model_dir: model_dir.to_string(),
        chat_template: chat_template.name().to_string(),
        max_new_tokens,
        artifact_load_us,
        ..Default::default()
    };

    let mut first_token_measured = false;
    let mut total_decode_us: u64 = 0;
    let mut total_generated = 0usize;

    for (turn_idx, prompt_text) in prompts.iter().enumerate() {
        let first_turn = turn_idx == 0;
        let full_prompt = chat_template.format_turn(prompt_text, first_turn);
        let prompt_tokens = runner.encode(&full_prompt)?;

        if prompt_tokens.is_empty() {
            if !json {
                eprintln!(
                    "[bench] turn {} prompt encoded to zero tokens, skipping",
                    turn_idx
                );
            }
            continue;
        }

        // Prefill: interactive append for all but last token + top-k on last.
        let prefill_start = Instant::now();
        let mut top = runner.prefill_tokens_topk_interactive(&prompt_tokens, 1)?;
        let prefill_us = duration_us(prefill_start.elapsed());

        if !first_token_measured {
            report.time_to_first_token_us = prefill_us;
            first_token_measured = true;
        }

        if top.is_empty() {
            if !json {
                eprintln!(
                    "[bench] turn {} prefill produced no top-k candidates",
                    turn_idx
                );
            }
            continue;
        }

        // Decode loop.
        let decode_start = Instant::now();
        let mut generated: Vec<u32> = Vec::new();
        let mut stopped_by_eos = false;
        let mut stopped_by_string: Option<String> = None;

        for step in 0..max_new_tokens {
            if runner.position() >= gen_cfg.ctx_size {
                break;
            }
            let Some(&next) = top.first() else {
                break;
            };

            if eos == Some(next.token_id) {
                if gen_cfg.append_eos_to_session {
                    runner.feed_token(next.token_id)?;
                }
                stopped_by_eos = true;
                break;
            }

            generated.push(next.token_id);

            // Check stop strings.
            let full_text: String = generated
                .iter()
                .map(|&tid| runner.decode(&[tid]).unwrap_or_default())
                .collect();
            if !gen_cfg.stop.is_empty() {
                for stop_str in &gen_cfg.stop {
                    if !stop_str.is_empty() && full_text.ends_with(stop_str.as_str()) {
                        runner.feed_token(next.token_id)?;
                        stopped_by_string = Some(stop_str.clone());
                        break;
                    }
                }
            }
            if stopped_by_string.is_some() {
                break;
            }

            if step + 1 == max_new_tokens {
                runner.feed_token(next.token_id)?;
                break;
            }

            top = runner.decode_token_topk(next.token_id, 1)?;
            if top.is_empty() {
                break;
            }
        }
        let decode_us = duration_us(decode_start.elapsed());

        total_decode_us = total_decode_us.saturating_add(decode_us);
        total_generated = total_generated.saturating_add(generated.len());

        report.turns.push(InteractiveTurnMeasurement {
            prompt_text: prompt_text.clone(),
            prompt_tokens,
            prefill_us,
            decode_us,
            generated_tokens: generated,
            stopped_by_eos,
            stopped_by_string,
        });
    }

    // ── Phase 3: collect counters ────────────────────────────────────────
    let counters_now = runner.operator_runtime_counters();
    let layer_stats = runner.layer_runtime_stats();

    report.aggregate_decode_tok_per_s = if total_decode_us > 0 {
        total_generated as f64 / (total_decode_us as f64 / 1_000_000.0)
    } else {
        0.0
    };
    report.total_generated = total_generated;
    report.resident_experts = layer_stats.iter().map(|s| s.resident_experts).sum();
    report.resident_expert_bytes = layer_stats.iter().map(|s| s.resident_expert_bytes).sum();
    report.expert_loads = counters_now
        .expert_loads
        .saturating_sub(counters_baseline.expert_loads);
    report.expert_load_bytes = counters_now
        .expert_load_bytes
        .saturating_sub(counters_baseline.expert_load_bytes);
    report.expert_evictions = counters_now
        .expert_evictions
        .saturating_sub(counters_baseline.expert_evictions);
    report.host_cache_entries = counters_now.expert_host_cache_entries;
    report.host_cache_bytes = counters_now.expert_host_cache_bytes;

    // ── Phase 4: output ──────────────────────────────────────────────────
    if json {
        let mut out = serde_json::json!({
            "model": report.model_dir,
            "chat_template": report.chat_template,
            "max_new_tokens": report.max_new_tokens,
            "artifact_load_s": report.artifact_load_us as f64 / 1_000_000.0,
            "time_to_first_token_s": report.time_to_first_token_us as f64 / 1_000_000.0,
            "total_turns": report.turns.len(),
            "total_generated": report.total_generated,
            "aggregate_decode_tok_per_s": report.aggregate_decode_tok_per_s,
            "resident_experts": report.resident_experts,
            "resident_expert_bytes": report.resident_expert_bytes,
            "expert_loads": report.expert_loads,
            "expert_load_bytes": report.expert_load_bytes,
            "expert_evictions": report.expert_evictions,
            "host_cache_entries": report.host_cache_entries,
            "host_cache_bytes": report.host_cache_bytes,
            "turns": report.turns.iter().map(|turn| {
                serde_json::json!({
                    "prompt": turn.prompt_text,
                    "prompt_tokens": turn.prompt_tokens.len(),
                    "prompt_token_ids": turn.prompt_tokens,
                    "prefill_s": turn.prefill_us as f64 / 1_000_000.0,
                    "decode_s": turn.decode_us as f64 / 1_000_000.0,
                    "prefill_tok_per_s": turn.prompt_tokens.len() as f64 / (turn.prefill_us as f64 / 1_000_000.0).max(1e-6),
                    "generated_tokens": turn.generated_tokens.len(),
                    "generated_token_ids": turn.generated_tokens,
                    "stopped_by_eos": turn.stopped_by_eos,
                    "stopped_by_string": turn.stopped_by_string,
                })
            }).collect::<Vec<_>>(),
        });

        // ── Golden trace comparison ─────────────────────────────────────
        if let Some(golden_path) = golden_trace_path {
            let golden_json = std::fs::read_to_string(golden_path)?;
            let golden: InteractiveTrace = serde_json::from_str(&golden_json)?;

            let observed_turns: Vec<GoldenTurn> = report
                .turns
                .iter()
                .map(|turn| GoldenTurn {
                    prompt_text: turn.prompt_text.clone(),
                    prompt_tokens: turn.prompt_tokens.clone(),
                    generated_tokens: turn.generated_tokens.clone(),
                    stopped_by_eos: turn.stopped_by_eos,
                    stopped_by_string: turn.stopped_by_string.clone(),
                })
                .collect();

            let comparison = compare_interactive_trace(&golden, &observed_turns);
            out["golden"] = serde_json::json!({
                "label": comparison.label,
                "turns_compared": comparison.turns_compared,
                "turns_ok": comparison.turns_ok,
                "all_ok": comparison.all_ok(),
                "mismatches": comparison.mismatches.iter().map(|m| {
                    serde_json::json!({
                        "turn": m.turn_index,
                        "prompt": m.prompt_text,
                        "message": m.message,
                        "expected_tokens": m.expected_tokens,
                        "observed_tokens": m.observed_tokens,
                    })
                }).collect::<Vec<_>>(),
            });
        }

        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!("=== Interactive Benchmark ===");
        println!("model:             {}", report.model_dir);
        println!("chat_template:     {}", report.chat_template);
        println!("max_new_tokens:    {}", report.max_new_tokens);
        println!(
            "artifact_load:     {:.3}s",
            report.artifact_load_us as f64 / 1_000_000.0
        );
        println!(
            "time_to_first_token: {:.3}s",
            report.time_to_first_token_us as f64 / 1_000_000.0
        );
        println!();

        for (i, turn) in report.turns.iter().enumerate() {
            println!(
                "Turn {}: {:?} ({} prompt tokens)",
                i + 1,
                turn.prompt_text,
                turn.prompt_tokens.len()
            );
            println!(
                "  prefill: {:.3}s  decode: {:.3}s",
                turn.prefill_us as f64 / 1_000_000.0,
                turn.decode_us as f64 / 1_000_000.0
            );
            println!(
                "  generated: {:?}  eos: {}  stop_str: {:?}",
                turn.generated_tokens, turn.stopped_by_eos, turn.stopped_by_string
            );
        }

        println!();
        println!(
            "aggregate_decode_tok_per_s: {:.3}",
            report.aggregate_decode_tok_per_s
        );
        println!("total_generated:           {}", report.total_generated);
        println!("resident_experts:          {}", report.resident_experts);
        println!(
            "resident_expert_bytes:     {}",
            report.resident_expert_bytes
        );
        println!("expert_loads:              {}", report.expert_loads);
        println!("expert_evictions:          {}", report.expert_evictions);
        println!("host_cache_entries:        {}", report.host_cache_entries);
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn cmd_bench_interactive(
    _model_dir: &str,
    _prompts: &[String],
    _max_new_tokens: usize,
    _chat_template_override: Option<&str>,
    _golden_trace_path: Option<&str>,
    _json: bool,
) -> anyhow::Result<()> {
    anyhow::bail!("bench-interactive requires --features cuda")
}

#[cfg(feature = "cuda")]
fn duration_us(d: Duration) -> u64 {
    d.as_micros().min(u128::from(u64::MAX)) as u64
}
