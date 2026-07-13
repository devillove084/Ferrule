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
use std::collections::BTreeMap;
#[cfg(feature = "cuda")]
use std::path::Path;
#[cfg(feature = "cuda")]
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use crate::bench::{GoldenTurn, InteractiveTrace, compare_interactive_trace};
#[cfg(feature = "cuda")]
use ferrule_common::MemoryPoolStats;
#[cfg(feature = "cuda")]
use ferrule_model::{
    ChatTemplate, ModelExecutionBackend, ModelRunner,
    models::deepseek_v4::{
        DeepSeekV4ArtifactModel, DeepSeekV4AttentionProfileStats, DeepSeekV4LayerProfileStats,
        DeepSeekV4OperatorRuntimeCounters, DeepSeekV4OutputProfileStats,
        DeepSeekV4PrefillRuntimeStats, DeepSeekV4PrepareOptions, DeepSeekV4Runner,
    },
    moe::ExpertPredictionStats,
};
#[cfg(feature = "cuda")]
use ferrule_runtime::{
    GenerateRequest, GenerationConfig, RequestId, ResidentActionKind, ResidentDriverStep,
    ResidentSchedulerConfig, ResidentTopKDriverConfig, ResidentTopKDriverStats, SamplingConfig,
    SessionId,
};

#[cfg(feature = "cuda")]
use super::resident::build_resident_topk_driver;

/// A single turn measurement captured by the interactive benchmark.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default)]
struct RuntimeStepMeasurement {
    action_kind: String,
    rows: usize,
    staged: usize,
    finished: usize,
    elapsed_us: u64,
    runner_position: usize,
    dsv4_operator_counters: DeepSeekV4OperatorRuntimeCounters,
    dsv4_layer_profile_stats: Vec<DeepSeekV4LayerProfileStats>,
    dsv4_attention_profile_stats: Vec<DeepSeekV4AttentionProfileStats>,
    dsv4_output_profile_stats: DeepSeekV4OutputProfileStats,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default)]
struct InteractiveTurnMeasurement {
    prompt_text: String,
    prompt_tokens: Vec<u32>,
    first_token_us: u64,
    prefill_us: u64,
    decode_us: u64,
    generated_tokens: Vec<u32>,
    final_position: usize,
    finish_reason: String,
    stopped_by_eos: bool,
    stopped_by_string: Option<String>,
    runtime_driver_stats: ResidentTopKDriverStats,
    dsv4_operator_counters: DeepSeekV4OperatorRuntimeCounters,
    dsv4_prefill_stats: DeepSeekV4PrefillRuntimeStats,
    dsv4_layer_profile_stats: Vec<DeepSeekV4LayerProfileStats>,
    dsv4_attention_profile_stats: Vec<DeepSeekV4AttentionProfileStats>,
    dsv4_output_profile_stats: DeepSeekV4OutputProfileStats,
    runtime_steps: Vec<RuntimeStepMeasurement>,
}

/// Full interactive benchmark report.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default)]
struct InteractiveBenchReport {
    model_dir: String,
    chat_template: String,
    max_new_tokens: usize,
    max_layers: usize,
    prefill_chunk_size: usize,
    runtime_path: String,
    dsv4_profile_sync: bool,
    /// Wall time from load start to runner ready.
    artifact_load_us: u64,
    /// Warmup decode budget requested before measured turns.
    warmup_tokens: usize,
    /// Wall time spent in the optional warmup turn.
    warmup_us: u64,
    /// Warmup tokens actually generated.
    warmup_generated_tokens: usize,
    /// Whether routing-hotset expert prewarm is enabled after warmup reset.
    expert_prewarm_enabled: bool,
    /// Wall time spent prewarming predicted/hot experts after warmup reset.
    expert_prewarm_us: u64,
    /// Experts actually uploaded by the prewarm stage.
    expert_prewarm_experts: usize,
    /// Expert load counter delta attributed to prewarm.
    expert_prewarm_loads: u64,
    /// Expert load bytes attributed to prewarm.
    expert_prewarm_load_bytes: u64,
    /// Wall time from measured prompt submission to first emitted token.
    time_to_first_token_us: u64,
    turns: Vec<InteractiveTurnMeasurement>,
    /// Aggregate prompt/prefill tokens per second across all measured turns.
    aggregate_prefill_tok_per_s: f64,
    /// Aggregate decode tokens per second across all measured turns.
    aggregate_decode_tok_per_s: f64,
    /// Total prompt tokens across all measured turns.
    total_prompt_tokens: usize,
    /// Total prefill wall time across all measured turns.
    total_prefill_us: u64,
    /// Total generated tokens across all measured turns.
    total_generated: usize,
    /// Final logical runner/session position at the end of measured turns.
    final_position: usize,
    /// Runtime-driver scheduler/executor counters for measured turns only.
    runtime_driver_stats: ResidentTopKDriverStats,
    /// DSV4 CUDA/operator counters for measured turns only.
    dsv4_operator_counters: DeepSeekV4OperatorRuntimeCounters,
    /// DSV4 prefill-path counters for measured turns only.
    dsv4_prefill_stats: DeepSeekV4PrefillRuntimeStats,
    /// DSV4 per-layer profile counters for measured turns only.
    dsv4_layer_profile_stats: Vec<DeepSeekV4LayerProfileStats>,
    /// DSV4 per-layer attention-internal profile counters for measured turns only.
    dsv4_attention_profile_stats: Vec<DeepSeekV4AttentionProfileStats>,
    /// DSV4 final hidden/output-head profile counters for measured turns only.
    dsv4_output_profile_stats: DeepSeekV4OutputProfileStats,
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
    warmup_tokens: usize,
    max_layers: usize,
    prefill_chunk_size: usize,
    golden_trace_path: Option<&str>,
    json: bool,
) -> anyhow::Result<()> {
    let model_path = Path::new(model_dir);
    let max_layers = max_layers.max(1);
    let prefill_chunk_size = prefill_chunk_size.max(1);
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

    let options = DeepSeekV4PrepareOptions {
        max_layers,
        output_head_chunk_rows: 4096,
        moe_prefetch_experts: 32,
        moe_hotset_experts: 48,
        ..DeepSeekV4PrepareOptions::default()
    };

    // ── Phase 1: load ────────────────────────────────────────────────────
    let load_start = Instant::now();
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(model_path, 128 * 1024 * 1024)?;
    let runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;
    let artifact_load_us = duration_us(load_start.elapsed());

    let mut report = InteractiveBenchReport {
        model_dir: model_dir.to_string(),
        chat_template: chat_template.name().to_string(),
        max_new_tokens,
        max_layers,
        prefill_chunk_size,
        runtime_path: "resident_topk_driver".into(),
        dsv4_profile_sync: runner.execution_policy().profile_sync(),
        artifact_load_us,
        warmup_tokens,
        expert_prewarm_enabled: dsv4_expert_prewarm_enabled(),
        ..Default::default()
    };

    run_with_resident_driver(
        runner,
        &chat_template,
        &gen_cfg,
        prompts,
        warmup_tokens,
        json,
        &mut report,
    )?;

    // ── Output ────────────────────────────────────────────────────────────
    if json {
        let mut out = serde_json::json!({
            "model": report.model_dir,
            "chat_template": report.chat_template,
            "runtime_path": report.runtime_path,
            "dsv4_profile_sync": report.dsv4_profile_sync,
            "max_new_tokens": report.max_new_tokens,
            "max_layers": report.max_layers,
            "prefill_chunk_size": report.prefill_chunk_size,
            "artifact_load_s": report.artifact_load_us as f64 / 1_000_000.0,
            "warmup_tokens": report.warmup_tokens,
            "warmup_s": report.warmup_us as f64 / 1_000_000.0,
            "warmup_generated_tokens": report.warmup_generated_tokens,
            "expert_prewarm_enabled": report.expert_prewarm_enabled,
            "expert_prewarm_s": report.expert_prewarm_us as f64 / 1_000_000.0,
            "expert_prewarm_experts": report.expert_prewarm_experts,
            "expert_prewarm_loads": report.expert_prewarm_loads,
            "expert_prewarm_load_bytes": report.expert_prewarm_load_bytes,
            "time_to_first_token_s": report.time_to_first_token_us as f64 / 1_000_000.0,
            "total_turns": report.turns.len(),
            "total_prompt_tokens": report.total_prompt_tokens,
            "total_prefill_s": report.total_prefill_us as f64 / 1_000_000.0,
            "total_generated": report.total_generated,
            "final_position": report.final_position,
            "aggregate_prefill_tok_per_s": report.aggregate_prefill_tok_per_s,
            "aggregate_decode_tok_per_s": report.aggregate_decode_tok_per_s,
            "runtime_driver_stats": resident_driver_stats_json(&report.runtime_driver_stats),
            "dsv4_operator_counters": dsv4_operator_counters_json(&report.dsv4_operator_counters),
            "dsv4_prefill_stats": dsv4_prefill_stats_json(&report.dsv4_prefill_stats),
            "dsv4_layer_profile_summary": dsv4_layer_profile_summary_json(&report.dsv4_layer_profile_stats),
            "dsv4_layer_profile": dsv4_layer_profile_stats_json(&report.dsv4_layer_profile_stats),
            "dsv4_attention_profile_summary": dsv4_attention_profile_summary_json(&report.dsv4_attention_profile_stats),
            "dsv4_attention_profile": dsv4_attention_profile_stats_json(&report.dsv4_attention_profile_stats),
            "dsv4_output_profile": dsv4_output_profile_stats_json(&report.dsv4_output_profile_stats),
            "resident_experts": report.resident_experts,
            "resident_expert_bytes": report.resident_expert_bytes,
            "expert_loads": report.expert_loads,
            "expert_load_bytes": report.expert_load_bytes,
            "expert_evictions": report.expert_evictions,
            "host_cache_entries": report.host_cache_entries,
            "host_cache_bytes": report.host_cache_bytes,
            "turns": report.turns.iter().map(interactive_turn_json).collect::<Vec<_>>(),
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
        println!("runtime_path:      {}", report.runtime_path);
        println!("dsv4_profile_sync: {}", report.dsv4_profile_sync);
        println!("max_new_tokens:    {}", report.max_new_tokens);
        println!("max_layers:        {}", report.max_layers);
        println!("prefill_chunk:     {}", report.prefill_chunk_size);
        println!(
            "artifact_load:     {:.3}s",
            report.artifact_load_us as f64 / 1_000_000.0
        );
        println!(
            "warmup:           {} requested / {} generated in {:.3}s",
            report.warmup_tokens,
            report.warmup_generated_tokens,
            report.warmup_us as f64 / 1_000_000.0
        );
        println!(
            "expert_prewarm:   enabled={} experts={} loads={} bytes={} in {:.3}s",
            report.expert_prewarm_enabled,
            report.expert_prewarm_experts,
            report.expert_prewarm_loads,
            report.expert_prewarm_load_bytes,
            report.expert_prewarm_us as f64 / 1_000_000.0
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
                "  ttft: {:.3}s  prefill: {:.3}s ({:.2} tok/s)  decode: {:.3}s ({:.2} tok/s)  pos: {}",
                turn.first_token_us as f64 / 1_000_000.0,
                turn.prefill_us as f64 / 1_000_000.0,
                turn.prompt_tokens.len() as f64 / (turn.prefill_us as f64 / 1_000_000.0).max(1e-6),
                turn.decode_us as f64 / 1_000_000.0,
                turn.generated_tokens.len() as f64
                    / (turn.decode_us as f64 / 1_000_000.0).max(1e-6),
                turn.final_position
            );
            println!(
                "  generated: {:?}  finish: {}  eos: {}  stop_str: {:?}",
                turn.generated_tokens,
                turn.finish_reason,
                turn.stopped_by_eos,
                turn.stopped_by_string
            );
            if turn.runtime_driver_stats.actions > 0 {
                println!(
                    "  runtime: actions={} prefill_chunks={} prefill_tokens={} decode_steps={}",
                    turn.runtime_driver_stats.actions,
                    turn.runtime_driver_stats.prefill_chunks,
                    turn.runtime_driver_stats.prefill_tokens,
                    turn.runtime_driver_stats.decode_steps
                );
                if let Some(slowest) = turn.runtime_steps.iter().max_by_key(|step| step.elapsed_us)
                {
                    println!(
                        "  slowest_runtime_step: kind={} rows={} elapsed={:.3}s pos={}",
                        slowest.action_kind,
                        slowest.rows,
                        slowest.elapsed_us as f64 / 1_000_000.0,
                        slowest.runner_position
                    );
                }
            }
            let prefill = turn.dsv4_prefill_stats;
            if prefill.logits_calls > 0 || prefill.no_logits_calls > 0 {
                println!(
                    "  dsv4_prefill: logits={}/{} no_logits={}/{} start_seg={}/{} append_seg={}/{}",
                    prefill.logits_calls,
                    prefill.logits_tokens,
                    prefill.no_logits_calls,
                    prefill.no_logits_tokens,
                    prefill.start_segment_calls,
                    prefill.start_segment_tokens,
                    prefill.append_segment_calls,
                    prefill.append_segment_tokens
                );
            }
            if !turn.dsv4_layer_profile_stats.is_empty() {
                let summary = sum_layer_profile_stats(&turn.dsv4_layer_profile_stats);
                let attention = sum_attention_profile_stats(&turn.dsv4_attention_profile_stats);
                println!(
                    "  dsv4_profile: layer_total={:.3}s attention={:.3}s moe={:.3}s output_topk={:.3}s attn_sparse={:.3}s attn_main_comp={:.3}s",
                    summary
                        .prefill_total_us
                        .saturating_add(summary.decode_total_us) as f64
                        / 1_000_000.0,
                    summary.attention_us as f64 / 1_000_000.0,
                    summary.moe_us as f64 / 1_000_000.0,
                    turn.dsv4_output_profile_stats.lm_head_topk_us as f64 / 1_000_000.0,
                    attention.sparse_attention_us as f64 / 1_000_000.0,
                    attention.main_compress_us as f64 / 1_000_000.0,
                );
            }
        }

        println!();
        println!(
            "aggregate_prefill_tok_per_s: {:.3}",
            report.aggregate_prefill_tok_per_s
        );
        println!(
            "aggregate_decode_tok_per_s:  {:.3}",
            report.aggregate_decode_tok_per_s
        );
        println!("total_prompt_tokens:       {}", report.total_prompt_tokens);
        println!("total_generated:           {}", report.total_generated);
        println!("final_position:            {}", report.final_position);
        println!(
            "runtime_driver:            actions={} prefill_chunks={} prefill_tokens={} decode_steps={} emitted={}",
            report.runtime_driver_stats.actions,
            report.runtime_driver_stats.prefill_chunks,
            report.runtime_driver_stats.prefill_tokens,
            report.runtime_driver_stats.decode_steps,
            report.runtime_driver_stats.emitted_tokens
        );
        println!(
            "dsv4_prefill:              logits={}/{} no_logits={}/{} start_seg={}/{} append_seg={}/{}",
            report.dsv4_prefill_stats.logits_calls,
            report.dsv4_prefill_stats.logits_tokens,
            report.dsv4_prefill_stats.no_logits_calls,
            report.dsv4_prefill_stats.no_logits_tokens,
            report.dsv4_prefill_stats.start_segment_calls,
            report.dsv4_prefill_stats.start_segment_tokens,
            report.dsv4_prefill_stats.append_segment_calls,
            report.dsv4_prefill_stats.append_segment_tokens
        );
        if !report.dsv4_layer_profile_stats.is_empty() {
            let summary = sum_layer_profile_stats(&report.dsv4_layer_profile_stats);
            let attention = sum_attention_profile_stats(&report.dsv4_attention_profile_stats);
            println!(
                "dsv4_profile:              layer_total={:.3}s attention={:.3}s moe={:.3}s state_init={:.3}s output_topk={:.3}s attn_sparse={:.3}s attn_main_comp={:.3}s",
                summary
                    .prefill_total_us
                    .saturating_add(summary.decode_total_us) as f64
                    / 1_000_000.0,
                summary.attention_us as f64 / 1_000_000.0,
                summary.moe_us as f64 / 1_000_000.0,
                summary.state_init_us as f64 / 1_000_000.0,
                report.dsv4_output_profile_stats.lm_head_topk_us as f64 / 1_000_000.0,
                attention.sparse_attention_us as f64 / 1_000_000.0,
                attention.main_compress_us as f64 / 1_000_000.0,
            );
        }
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

#[cfg(feature = "cuda")]
fn maybe_prewarm_runner(
    runner: &mut DeepSeekV4Runner,
    report: &mut InteractiveBenchReport,
) -> anyhow::Result<()> {
    report.expert_prewarm_enabled = dsv4_expert_prewarm_enabled();
    if !report.expert_prewarm_enabled {
        return Ok(());
    }
    let counters_before = runner.operator_runtime_counters();
    let start = Instant::now();
    let warmed = runner.prewarm_predicted_experts()?;
    let elapsed_us = duration_us(start.elapsed());
    let counters_after = runner.operator_runtime_counters();
    report.expert_prewarm_us = report.expert_prewarm_us.saturating_add(elapsed_us);
    report.expert_prewarm_experts = report.expert_prewarm_experts.saturating_add(warmed);
    report.expert_prewarm_loads = report.expert_prewarm_loads.saturating_add(
        counters_after
            .expert_loads
            .saturating_sub(counters_before.expert_loads),
    );
    report.expert_prewarm_load_bytes = report.expert_prewarm_load_bytes.saturating_add(
        counters_after
            .expert_load_bytes
            .saturating_sub(counters_before.expert_load_bytes),
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_with_resident_driver(
    runner: DeepSeekV4Runner,
    chat_template: &ChatTemplate,
    gen_cfg: &GenerationConfig,
    prompts: &[String],
    warmup_tokens: usize,
    json: bool,
    report: &mut InteractiveBenchReport,
) -> anyhow::Result<()> {
    let scheduler_config = ResidentSchedulerConfig {
        prefill_chunk_size: report.prefill_chunk_size.max(1),
        max_active_sequences: 1,
        max_decode_batch: 1,
        ..Default::default()
    };
    let driver_config = ResidentTopKDriverConfig {
        ctx_size: gen_cfg.ctx_size,
        stop_at_eos: gen_cfg.stop_at_eos,
        append_eos_to_session: gen_cfg.append_eos_to_session,
        max_steps_per_run: gen_cfg
            .ctx_size
            .saturating_add(gen_cfg.max_new_tokens)
            .saturating_add(warmup_tokens)
            .saturating_add(64),
    };
    let build_driver = |runner: DeepSeekV4Runner| {
        let schema = runner.kv_layout_schema().clone();
        build_resident_topk_driver(runner, Box::new(schema), scheduler_config, driver_config)
    };
    let mut driver = build_driver(runner)?;

    if warmup_tokens > 0 {
        let warmup_prompt = chat_template.format_turn("warmup", true);
        let warmup_prompt_tokens = driver.executor().runner().encode(&warmup_prompt)?;
        let warmup_request = driver_request(
            0,
            SessionId(u64::MAX),
            warmup_prompt_tokens,
            warmup_tokens,
            &gen_cfg.stop,
        );
        let warmup_start = Instant::now();
        driver.submit(warmup_request);
        let warmup_stats = driver.run_until_blocked(|_| Ok(()))?;
        report.warmup_us = duration_us(warmup_start.elapsed());
        report.warmup_generated_tokens = warmup_stats.emitted_tokens;
        let _ = driver.drain_finished();
        let mut runner = driver.into_runner()?;
        runner.reset_session()?;
        maybe_prewarm_runner(&mut runner, report)?;
        driver = build_driver(runner)?;
    }

    let counters_baseline = driver.executor().runner().operator_runtime_counters();
    let driver_stats_baseline = driver.stats().clone();
    let prefill_stats_baseline = driver.executor().runner().prefill_runtime_stats();
    let layer_profile_baseline = driver.executor().runner().layer_profile_stats();
    let attention_profile_baseline = driver.executor().runner().attention_profile_stats();
    let output_profile_baseline = driver.executor().runner().output_profile_stats();
    let mut first_token_measured = false;
    let mut total_prefill_us: u64 = 0;
    let mut total_decode_us: u64 = 0;
    let mut total_prompt_tokens = 0usize;
    let mut total_generated = 0usize;

    for (turn_idx, prompt_text) in prompts.iter().enumerate() {
        let first_turn = turn_idx == 0;
        let full_prompt = chat_template.format_turn(prompt_text, first_turn);
        let prompt_tokens = driver.executor().runner().encode(&full_prompt)?;

        if prompt_tokens.is_empty() {
            if !json {
                eprintln!(
                    "[bench] turn {} prompt encoded to zero tokens, skipping",
                    turn_idx
                );
            }
            continue;
        }

        let request = driver_request(
            turn_idx as u64 + 1,
            SessionId(0),
            prompt_tokens.clone(),
            gen_cfg.max_new_tokens,
            &gen_cfg.stop,
        );
        let turn_driver_stats_before = driver.stats().clone();
        let turn_operator_counters_before = driver.executor().runner().operator_runtime_counters();
        let turn_prefill_stats_before = driver.executor().runner().prefill_runtime_stats();
        let turn_layer_profile_before = driver.executor().runner().layer_profile_stats();
        let turn_attention_profile_before = driver.executor().runner().attention_profile_stats();
        let turn_output_profile_before = driver.executor().runner().output_profile_stats();
        driver.submit(request);

        let turn_start = Instant::now();
        let mut first_token_us = None;
        let mut prefill_us = 0u64;
        let mut decode_us = 0u64;
        let mut generated_tokens = Vec::new();
        let mut runtime_steps = Vec::new();

        loop {
            let step_operator_counters_before =
                driver.executor().runner().operator_runtime_counters();
            let step_layer_profile_before = driver.executor().runner().layer_profile_stats();
            let step_attention_profile_before =
                driver.executor().runner().attention_profile_stats();
            let step_output_profile_before = driver.executor().runner().output_profile_stats();
            let step_start = Instant::now();
            let step = driver.step(&mut |event| {
                first_token_us.get_or_insert_with(|| duration_us(turn_start.elapsed()));
                generated_tokens.push(event.token);
                Ok(())
            })?;
            let step_us = duration_us(step_start.elapsed());
            let step_operator_counters = dsv4_operator_counters_delta(
                step_operator_counters_before,
                driver.executor().runner().operator_runtime_counters(),
            );
            let step_layer_profile_stats = dsv4_layer_profile_stats_delta(
                &step_layer_profile_before,
                &driver.executor().runner().layer_profile_stats(),
            );
            let step_attention_profile_stats = dsv4_attention_profile_stats_delta(
                &step_attention_profile_before,
                &driver.executor().runner().attention_profile_stats(),
            );
            let step_output_profile_stats = dsv4_output_profile_stats_delta(
                step_output_profile_before,
                driver.executor().runner().output_profile_stats(),
            );
            match step {
                ResidentDriverStep::Executed {
                    action_kind,
                    rows,
                    staged,
                    finished,
                } => {
                    match action_kind {
                        ResidentActionKind::Prefill => {
                            prefill_us = prefill_us.saturating_add(step_us);
                        }
                        ResidentActionKind::Decode => {
                            decode_us = decode_us.saturating_add(step_us);
                        }
                        ResidentActionKind::Mixed => {
                            prefill_us = prefill_us.saturating_add(step_us);
                        }
                        ResidentActionKind::Finish | ResidentActionKind::Cancel => {}
                    }
                    runtime_steps.push(RuntimeStepMeasurement {
                        action_kind: resident_action_kind_name(action_kind).to_string(),
                        rows,
                        staged,
                        finished,
                        elapsed_us: step_us,
                        runner_position: driver.executor().runner().position(),
                        dsv4_operator_counters: step_operator_counters,
                        dsv4_layer_profile_stats: step_layer_profile_stats,
                        dsv4_attention_profile_stats: step_attention_profile_stats,
                        dsv4_output_profile_stats: step_output_profile_stats,
                    });
                }
                ResidentDriverStep::Idle => break,
                ResidentDriverStep::Blocked => {
                    anyhow::bail!("resident runtime driver blocked while running measured turn")
                }
            }
        }

        let first_token_us = first_token_us.unwrap_or_else(|| duration_us(turn_start.elapsed()));
        if !first_token_measured {
            report.time_to_first_token_us = first_token_us;
            first_token_measured = true;
        }

        let finished = driver.drain_finished();
        let sequence = finished.last().ok_or_else(|| {
            anyhow::anyhow!("resident runtime driver produced no finished sequence")
        })?;
        let finish_reason = sequence.finish_reason;
        let stopped_by_string = if matches!(
            finish_reason,
            Some(ferrule_runtime::SequenceFinishReason::StopString)
        ) {
            matched_stop_string(&sequence.generated_text, &gen_cfg.stop)
        } else {
            None
        };
        let stopped_by_eos = matches!(
            finish_reason,
            Some(ferrule_runtime::SequenceFinishReason::Eos)
        );

        total_prefill_us = total_prefill_us.saturating_add(prefill_us);
        total_decode_us = total_decode_us.saturating_add(decode_us);
        total_prompt_tokens = total_prompt_tokens.saturating_add(prompt_tokens.len());
        total_generated = total_generated.saturating_add(generated_tokens.len());
        report.final_position = sequence.position;

        let turn_driver_stats =
            resident_driver_stats_delta(turn_driver_stats_before, driver.stats().clone());
        let turn_operator_counters = dsv4_operator_counters_delta(
            turn_operator_counters_before,
            driver.executor().runner().operator_runtime_counters(),
        );
        let turn_prefill_stats = dsv4_prefill_stats_delta(
            turn_prefill_stats_before,
            driver.executor().runner().prefill_runtime_stats(),
        );
        let turn_layer_profile_stats = dsv4_layer_profile_stats_delta(
            &turn_layer_profile_before,
            &driver.executor().runner().layer_profile_stats(),
        );
        let turn_attention_profile_stats = dsv4_attention_profile_stats_delta(
            &turn_attention_profile_before,
            &driver.executor().runner().attention_profile_stats(),
        );
        let turn_output_profile_stats = dsv4_output_profile_stats_delta(
            turn_output_profile_before,
            driver.executor().runner().output_profile_stats(),
        );

        report.turns.push(InteractiveTurnMeasurement {
            prompt_text: prompt_text.clone(),
            prompt_tokens,
            first_token_us,
            prefill_us,
            decode_us,
            generated_tokens,
            final_position: sequence.position,
            finish_reason: finish_reason
                .map(|reason| reason.as_str().to_string())
                .unwrap_or_else(|| "unknown".into()),
            stopped_by_eos,
            stopped_by_string,
            runtime_driver_stats: turn_driver_stats,
            dsv4_operator_counters: turn_operator_counters,
            dsv4_prefill_stats: turn_prefill_stats,
            dsv4_layer_profile_stats: turn_layer_profile_stats,
            dsv4_attention_profile_stats: turn_attention_profile_stats,
            dsv4_output_profile_stats: turn_output_profile_stats,
            runtime_steps,
        });
    }

    let counters_now = driver.executor().runner().operator_runtime_counters();
    report.runtime_driver_stats =
        resident_driver_stats_delta(driver_stats_baseline, driver.stats().clone());
    report.dsv4_operator_counters = dsv4_operator_counters_delta(counters_baseline, counters_now);
    report.dsv4_prefill_stats = dsv4_prefill_stats_delta(
        prefill_stats_baseline,
        driver.executor().runner().prefill_runtime_stats(),
    );
    report.dsv4_layer_profile_stats = dsv4_layer_profile_stats_delta(
        &layer_profile_baseline,
        &driver.executor().runner().layer_profile_stats(),
    );
    report.dsv4_attention_profile_stats = dsv4_attention_profile_stats_delta(
        &attention_profile_baseline,
        &driver.executor().runner().attention_profile_stats(),
    );
    report.dsv4_output_profile_stats = dsv4_output_profile_stats_delta(
        output_profile_baseline,
        driver.executor().runner().output_profile_stats(),
    );
    let layer_stats = driver.executor().runner().layer_runtime_stats();
    finish_report_counters(
        report,
        total_prefill_us,
        total_decode_us,
        total_prompt_tokens,
        total_generated,
        counters_baseline.expert_loads,
        counters_baseline.expert_load_bytes,
        counters_baseline.expert_evictions,
        counters_now.expert_loads,
        counters_now.expert_load_bytes,
        counters_now.expert_evictions,
        counters_now.expert_host_cache.entries_used,
        counters_now.expert_host_cache.bytes_used,
        layer_stats.iter().map(|s| s.resident_experts).sum(),
        layer_stats.iter().map(|s| s.resident_expert_bytes).sum(),
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn driver_request(
    id: u64,
    session_id: SessionId,
    prompt_tokens: Vec<u32>,
    max_new_tokens: usize,
    stop: &[String],
) -> GenerateRequest {
    GenerateRequest {
        id: RequestId(id),
        session_id: Some(session_id),
        prompt_tokens,
        sampling: SamplingConfig::greedy(),
        max_new_tokens,
        stop: stop.to_vec(),
        ignore_eos: false,
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn finish_report_counters(
    report: &mut InteractiveBenchReport,
    total_prefill_us: u64,
    total_decode_us: u64,
    total_prompt_tokens: usize,
    total_generated: usize,
    baseline_expert_loads: u64,
    baseline_expert_load_bytes: u64,
    baseline_expert_evictions: u64,
    expert_loads: u64,
    expert_load_bytes: u64,
    expert_evictions: u64,
    host_cache_entries: usize,
    host_cache_bytes: u64,
    resident_experts: usize,
    resident_expert_bytes: u64,
) {
    report.aggregate_prefill_tok_per_s = if total_prefill_us > 0 {
        total_prompt_tokens as f64 / (total_prefill_us as f64 / 1_000_000.0)
    } else {
        0.0
    };
    report.aggregate_decode_tok_per_s = if total_decode_us > 0 {
        total_generated as f64 / (total_decode_us as f64 / 1_000_000.0)
    } else {
        0.0
    };
    report.total_prompt_tokens = total_prompt_tokens;
    report.total_prefill_us = total_prefill_us;
    report.total_generated = total_generated;
    report.resident_experts = resident_experts;
    report.resident_expert_bytes = resident_expert_bytes;
    report.expert_loads = expert_loads.saturating_sub(baseline_expert_loads);
    report.expert_load_bytes = expert_load_bytes.saturating_sub(baseline_expert_load_bytes);
    report.expert_evictions = expert_evictions.saturating_sub(baseline_expert_evictions);
    report.host_cache_entries = host_cache_entries;
    report.host_cache_bytes = host_cache_bytes;
}

#[cfg(feature = "cuda")]
fn matched_stop_string(text: &str, stop: &[String]) -> Option<String> {
    stop.iter()
        .find(|candidate| !candidate.is_empty() && text.ends_with(candidate.as_str()))
        .cloned()
}

#[cfg(feature = "cuda")]
fn resident_action_kind_name(kind: ResidentActionKind) -> &'static str {
    match kind {
        ResidentActionKind::Prefill => "prefill",
        ResidentActionKind::Decode => "decode",
        ResidentActionKind::Mixed => "mixed",
        ResidentActionKind::Finish => "finish",
        ResidentActionKind::Cancel => "cancel",
    }
}

#[cfg(feature = "cuda")]
fn resident_driver_stats_delta(
    before: ResidentTopKDriverStats,
    after: ResidentTopKDriverStats,
) -> ResidentTopKDriverStats {
    ResidentTopKDriverStats {
        actions: after.actions.saturating_sub(before.actions),
        prefill_chunks: after.prefill_chunks.saturating_sub(before.prefill_chunks),
        prefill_tokens: after.prefill_tokens.saturating_sub(before.prefill_tokens),
        decode_steps: after.decode_steps.saturating_sub(before.decode_steps),
        emitted_tokens: after.emitted_tokens.saturating_sub(before.emitted_tokens),
        staged_tokens: after.staged_tokens.saturating_sub(before.staged_tokens),
        finished_sequences: after
            .finished_sequences
            .saturating_sub(before.finished_sequences),
    }
}

#[cfg(feature = "cuda")]
fn memory_pool_stats_delta(before: MemoryPoolStats, after: MemoryPoolStats) -> MemoryPoolStats {
    MemoryPoolStats {
        limits: after.limits,
        entries_used: after.entries_used,
        bytes_used: after.bytes_used,
        peak_bytes_used: after.peak_bytes_used,
        hits: after.hits.saturating_sub(before.hits),
        misses: after.misses.saturating_sub(before.misses),
        admissions: after.admissions.saturating_sub(before.admissions),
        evictions: after.evictions.saturating_sub(before.evictions),
        rejections: after.rejections.saturating_sub(before.rejections),
    }
}

#[cfg(feature = "cuda")]
fn dsv4_operator_counters_delta(
    before: DeepSeekV4OperatorRuntimeCounters,
    after: DeepSeekV4OperatorRuntimeCounters,
) -> DeepSeekV4OperatorRuntimeCounters {
    DeepSeekV4OperatorRuntimeCounters {
        kernel_launches: after.kernel_launches.saturating_sub(before.kernel_launches),
        host_to_device_copies: after
            .host_to_device_copies
            .saturating_sub(before.host_to_device_copies),
        host_to_device_bytes: after
            .host_to_device_bytes
            .saturating_sub(before.host_to_device_bytes),
        device_to_host_copies: after
            .device_to_host_copies
            .saturating_sub(before.device_to_host_copies),
        device_to_host_bytes: after
            .device_to_host_bytes
            .saturating_sub(before.device_to_host_bytes),
        artifact_uploads: after
            .artifact_uploads
            .saturating_sub(before.artifact_uploads),
        artifact_upload_bytes: after
            .artifact_upload_bytes
            .saturating_sub(before.artifact_upload_bytes),
        device_allocation_attempts: after
            .device_allocation_attempts
            .saturating_sub(before.device_allocation_attempts),
        device_allocations: after
            .device_allocations
            .saturating_sub(before.device_allocations),
        device_allocation_failures: after
            .device_allocation_failures
            .saturating_sub(before.device_allocation_failures),
        device_allocation_bytes: after
            .device_allocation_bytes
            .saturating_sub(before.device_allocation_bytes),
        stream_wide_syncs: after
            .stream_wide_syncs
            .saturating_sub(before.stream_wide_syncs),
        stream_wide_sync_failures: after
            .stream_wide_sync_failures
            .saturating_sub(before.stream_wide_sync_failures),
        moe_calls: after.moe_calls.saturating_sub(before.moe_calls),
        moe_tc_calls: after.moe_tc_calls.saturating_sub(before.moe_tc_calls),
        moe_scalar_calls: after
            .moe_scalar_calls
            .saturating_sub(before.moe_scalar_calls),
        moe_reduce_calls: after
            .moe_reduce_calls
            .saturating_sub(before.moe_reduce_calls),
        moe_total_us: after.moe_total_us.saturating_sub(before.moe_total_us),
        moe_input_prepare_us: after
            .moe_input_prepare_us
            .saturating_sub(before.moe_input_prepare_us),
        moe_gate_up_us: after.moe_gate_up_us.saturating_sub(before.moe_gate_up_us),
        moe_swiglu_us: after.moe_swiglu_us.saturating_sub(before.moe_swiglu_us),
        moe_hidden_pack_us: after
            .moe_hidden_pack_us
            .saturating_sub(before.moe_hidden_pack_us),
        moe_down_us: after.moe_down_us.saturating_sub(before.moe_down_us),
        moe_router_us: after.moe_router_us.saturating_sub(before.moe_router_us),
        moe_routing_us: after.moe_routing_us.saturating_sub(before.moe_routing_us),
        moe_plan_us: after.moe_plan_us.saturating_sub(before.moe_plan_us),
        moe_predicted_experts: after
            .moe_predicted_experts
            .saturating_sub(before.moe_predicted_experts),
        moe_prefetch_loads: after
            .moe_prefetch_loads
            .saturating_sub(before.moe_prefetch_loads),
        moe_prefetch_enqueued: after
            .moe_prefetch_enqueued
            .saturating_sub(before.moe_prefetch_enqueued),
        moe_prefetch_skipped_cached_or_inflight: after
            .moe_prefetch_skipped_cached_or_inflight
            .saturating_sub(before.moe_prefetch_skipped_cached_or_inflight),
        moe_prefetch_resident: after
            .moe_prefetch_resident
            .saturating_sub(before.moe_prefetch_resident),
        moe_prefetch_materializing: after
            .moe_prefetch_materializing
            .saturating_sub(before.moe_prefetch_materializing),
        moe_prefetch_host_staged: after
            .moe_prefetch_host_staged
            .saturating_sub(before.moe_prefetch_host_staged),
        moe_prefetch_in_flight: after
            .moe_prefetch_in_flight
            .saturating_sub(before.moe_prefetch_in_flight),
        moe_prefetch_cold: after
            .moe_prefetch_cold
            .saturating_sub(before.moe_prefetch_cold),
        expert_selected_resident_hits: after
            .expert_selected_resident_hits
            .saturating_sub(before.expert_selected_resident_hits),
        expert_selected_upload_hits: after
            .expert_selected_upload_hits
            .saturating_sub(before.expert_selected_upload_hits),
        expert_selected_host_staged_hits: after
            .expert_selected_host_staged_hits
            .saturating_sub(before.expert_selected_host_staged_hits),
        expert_selected_host_staging_waits: after
            .expert_selected_host_staging_waits
            .saturating_sub(before.expert_selected_host_staging_waits),
        expert_selected_host_staging_hits: after
            .expert_selected_host_staging_hits
            .saturating_sub(before.expert_selected_host_staging_hits),
        expert_selected_host_staging_wait_us: after
            .expert_selected_host_staging_wait_us
            .saturating_sub(before.expert_selected_host_staging_wait_us),
        expert_selected_cold_misses: after
            .expert_selected_cold_misses
            .saturating_sub(before.expert_selected_cold_misses),
        expert_upload_prefetch_submitted: after
            .expert_upload_prefetch_submitted
            .saturating_sub(before.expert_upload_prefetch_submitted),
        expert_upload_prefetch_completed: after
            .expert_upload_prefetch_completed
            .saturating_sub(before.expert_upload_prefetch_completed),
        expert_upload_prefetch_in_flight: after.expert_upload_prefetch_in_flight,
        expert_selected_upload_waits: after
            .expert_selected_upload_waits
            .saturating_sub(before.expert_selected_upload_waits),
        expert_selected_upload_wait_us: after
            .expert_selected_upload_wait_us
            .saturating_sub(before.expert_selected_upload_wait_us),
        expert_async_upload_bytes: after
            .expert_async_upload_bytes
            .saturating_sub(before.expert_async_upload_bytes),
        expert_lookahead_prefetch_calls: after
            .expert_lookahead_prefetch_calls
            .saturating_sub(before.expert_lookahead_prefetch_calls),
        expert_lookahead_prefetch_experts: after
            .expert_lookahead_prefetch_experts
            .saturating_sub(before.expert_lookahead_prefetch_experts),
        expert_lookahead_prefetch_enqueued: after
            .expert_lookahead_prefetch_enqueued
            .saturating_sub(before.expert_lookahead_prefetch_enqueued),
        expert_lookahead_prefetch_us: after
            .expert_lookahead_prefetch_us
            .saturating_sub(before.expert_lookahead_prefetch_us),

        moe_cache_lookup_us: after
            .moe_cache_lookup_us
            .saturating_sub(before.moe_cache_lookup_us),
        moe_expert_read_us: after
            .moe_expert_read_us
            .saturating_sub(before.moe_expert_read_us),
        moe_expert_upload_us: after
            .moe_expert_upload_us
            .saturating_sub(before.moe_expert_upload_us),
        moe_shared_us: after.moe_shared_us.saturating_sub(before.moe_shared_us),
        moe_workspace_us: after
            .moe_workspace_us
            .saturating_sub(before.moe_workspace_us),
        moe_compute_submit_us: after
            .moe_compute_submit_us
            .saturating_sub(before.moe_compute_submit_us),
        moe_commit_us: after.moe_commit_us.saturating_sub(before.moe_commit_us),
        output_head_calls: after
            .output_head_calls
            .saturating_sub(before.output_head_calls),
        output_head_chunks: after
            .output_head_chunks
            .saturating_sub(before.output_head_chunks),
        output_head_rows: after
            .output_head_rows
            .saturating_sub(before.output_head_rows),
        output_head_cache_hits: after
            .output_head_cache_hits
            .saturating_sub(before.output_head_cache_hits),
        output_head_cache_misses: after
            .output_head_cache_misses
            .saturating_sub(before.output_head_cache_misses),
        output_head_hidden_uploads: after
            .output_head_hidden_uploads
            .saturating_sub(before.output_head_hidden_uploads),
        output_head_hidden_upload_us: after
            .output_head_hidden_upload_us
            .saturating_sub(before.output_head_hidden_upload_us),
        output_head_read_us: after
            .output_head_read_us
            .saturating_sub(before.output_head_read_us),
        output_head_upload_us: after
            .output_head_upload_us
            .saturating_sub(before.output_head_upload_us),
        output_head_topk_us: after
            .output_head_topk_us
            .saturating_sub(before.output_head_topk_us),
        output_head_merge_us: after
            .output_head_merge_us
            .saturating_sub(before.output_head_merge_us),
        expert_selected: after.expert_selected.saturating_sub(before.expert_selected),
        expert_selected_load_requests: after
            .expert_selected_load_requests
            .saturating_sub(before.expert_selected_load_requests),
        expert_loads: after.expert_loads.saturating_sub(before.expert_loads),
        expert_load_bytes: after
            .expert_load_bytes
            .saturating_sub(before.expert_load_bytes),
        expert_evictions: after
            .expert_evictions
            .saturating_sub(before.expert_evictions),
        expert_host_cache: memory_pool_stats_delta(
            before.expert_host_cache,
            after.expert_host_cache,
        ),
        expert_pinned_cache: memory_pool_stats_delta(
            before.expert_pinned_cache,
            after.expert_pinned_cache,
        ),
        expert_cuda_resident_entries: after.expert_cuda_resident_entries,
        expert_cuda_resident_bytes: after.expert_cuda_resident_bytes,
        expert_async_prefetch_submitted: after
            .expert_async_prefetch_submitted
            .saturating_sub(before.expert_async_prefetch_submitted),
        expert_async_prefetch_completed: after
            .expert_async_prefetch_completed
            .saturating_sub(before.expert_async_prefetch_completed),
        expert_async_prefetch_failed: after
            .expert_async_prefetch_failed
            .saturating_sub(before.expert_async_prefetch_failed),
        expert_async_prefetch_skipped: after
            .expert_async_prefetch_skipped
            .saturating_sub(before.expert_async_prefetch_skipped),
        expert_async_prefetch_in_flight: after.expert_async_prefetch_in_flight,
        arena_hits: after.arena_hits.saturating_sub(before.arena_hits),
        arena_misses: after.arena_misses.saturating_sub(before.arena_misses),
        arena_grows: after.arena_grows.saturating_sub(before.arena_grows),
        arena_reuses: after.arena_reuses.saturating_sub(before.arena_reuses),
        expert_residency_stats: ferrule_common::ExpertResidencyStats {
            resident: after.expert_residency_stats.resident,
            active_leases: after.expert_residency_stats.active_leases,
            installs: after
                .expert_residency_stats
                .installs
                .saturating_sub(before.expert_residency_stats.installs),
            evictions: after
                .expert_residency_stats
                .evictions
                .saturating_sub(before.expert_residency_stats.evictions),
            resident_hits: after
                .expert_residency_stats
                .resident_hits
                .saturating_sub(before.expert_residency_stats.resident_hits),
            stale_releases: after
                .expert_residency_stats
                .stale_releases
                .saturating_sub(before.expert_residency_stats.stale_releases),
            prepare_cancellations: after
                .expert_residency_stats
                .prepare_cancellations
                .saturating_sub(before.expert_residency_stats.prepare_cancellations),
            prefetch_capacity_misses: after
                .expert_residency_stats
                .prefetch_capacity_misses
                .saturating_sub(before.expert_residency_stats.prefetch_capacity_misses),
        },
        expert_predictor_stats: expert_prediction_stats_delta(
            before.expert_predictor_stats,
            after.expert_predictor_stats,
        ),
    }
}

#[cfg(feature = "cuda")]
fn expert_prediction_stats_delta(
    before: ExpertPredictionStats,
    after: ExpertPredictionStats,
) -> ExpertPredictionStats {
    ExpertPredictionStats {
        predict_calls: after.predict_calls.saturating_sub(before.predict_calls),
        predicted_experts: after
            .predicted_experts
            .saturating_sub(before.predicted_experts),
        observe_calls: after.observe_calls.saturating_sub(before.observe_calls),
        observed_experts: after
            .observed_experts
            .saturating_sub(before.observed_experts),
        cold_miss_observations: after
            .cold_miss_observations
            .saturating_sub(before.cold_miss_observations),
        transition_observations: after
            .transition_observations
            .saturating_sub(before.transition_observations),
    }
}

#[cfg(feature = "cuda")]
fn dsv4_layer_profile_stats_delta(
    before: &[DeepSeekV4LayerProfileStats],
    after: &[DeepSeekV4LayerProfileStats],
) -> Vec<DeepSeekV4LayerProfileStats> {
    let before_by_layer: BTreeMap<usize, DeepSeekV4LayerProfileStats> =
        before.iter().map(|stats| (stats.layer, *stats)).collect();
    after
        .iter()
        .map(|stats| {
            let before =
                before_by_layer
                    .get(&stats.layer)
                    .copied()
                    .unwrap_or(DeepSeekV4LayerProfileStats {
                        layer: stats.layer,
                        ..DeepSeekV4LayerProfileStats::default()
                    });
            DeepSeekV4LayerProfileStats {
                layer: stats.layer,

                state_init_calls: stats
                    .state_init_calls
                    .saturating_sub(before.state_init_calls),
                state_init_us: stats.state_init_us.saturating_sub(before.state_init_us),
                decode_calls: stats.decode_calls.saturating_sub(before.decode_calls),
                decode_total_us: stats.decode_total_us.saturating_sub(before.decode_total_us),
                prefill_calls: stats.prefill_calls.saturating_sub(before.prefill_calls),
                prefill_tokens: stats.prefill_tokens.saturating_sub(before.prefill_tokens),
                prefill_total_us: stats
                    .prefill_total_us
                    .saturating_sub(before.prefill_total_us),
                attn_hc_pre_us: stats.attn_hc_pre_us.saturating_sub(before.attn_hc_pre_us),
                attn_norm_us: stats.attn_norm_us.saturating_sub(before.attn_norm_us),
                attention_us: stats.attention_us.saturating_sub(before.attention_us),
                attn_hc_post_us: stats.attn_hc_post_us.saturating_sub(before.attn_hc_post_us),
                ffn_hc_pre_us: stats.ffn_hc_pre_us.saturating_sub(before.ffn_hc_pre_us),
                ffn_norm_us: stats.ffn_norm_us.saturating_sub(before.ffn_norm_us),
                moe_us: stats.moe_us.saturating_sub(before.moe_us),
                ffn_hc_post_us: stats.ffn_hc_post_us.saturating_sub(before.ffn_hc_post_us),
            }
        })
        .filter(|stats| {
            stats.state_init_calls > 0 || stats.decode_calls > 0 || stats.prefill_calls > 0
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn dsv4_attention_profile_stats_delta(
    before: &[DeepSeekV4AttentionProfileStats],
    after: &[DeepSeekV4AttentionProfileStats],
) -> Vec<DeepSeekV4AttentionProfileStats> {
    let before_by_layer: BTreeMap<usize, DeepSeekV4AttentionProfileStats> =
        before.iter().map(|stats| (stats.layer, *stats)).collect();
    after
        .iter()
        .map(|stats| {
            let before = before_by_layer.get(&stats.layer).copied().unwrap_or(
                DeepSeekV4AttentionProfileStats {
                    layer: stats.layer,
                    ..DeepSeekV4AttentionProfileStats::default()
                },
            );
            DeepSeekV4AttentionProfileStats {
                layer: stats.layer,
                calls: stats.calls.saturating_sub(before.calls),
                tokens: stats.tokens.saturating_sub(before.tokens),
                q_a_us: stats.q_a_us.saturating_sub(before.q_a_us),
                q_norm_us: stats.q_norm_us.saturating_sub(before.q_norm_us),
                q_b_us: stats.q_b_us.saturating_sub(before.q_b_us),
                q_head_norm_us: stats.q_head_norm_us.saturating_sub(before.q_head_norm_us),
                q_rope_us: stats.q_rope_us.saturating_sub(before.q_rope_us),
                kv_proj_us: stats.kv_proj_us.saturating_sub(before.kv_proj_us),
                kv_norm_us: stats.kv_norm_us.saturating_sub(before.kv_norm_us),
                kv_rope_quant_us: stats
                    .kv_rope_quant_us
                    .saturating_sub(before.kv_rope_quant_us),
                kv_cache_append_us: stats
                    .kv_cache_append_us
                    .saturating_sub(before.kv_cache_append_us),
                indexer_compress_us: stats
                    .indexer_compress_us
                    .saturating_sub(before.indexer_compress_us),
                main_compress_us: stats
                    .main_compress_us
                    .saturating_sub(before.main_compress_us),
                compressed_kv_upload_us: stats
                    .compressed_kv_upload_us
                    .saturating_sub(before.compressed_kv_upload_us),
                topk_build_us: stats.topk_build_us.saturating_sub(before.topk_build_us),
                sparse_attention_us: stats
                    .sparse_attention_us
                    .saturating_sub(before.sparse_attention_us),
                context_rope_us: stats.context_rope_us.saturating_sub(before.context_rope_us),
                output_a_us: stats.output_a_us.saturating_sub(before.output_a_us),
                output_b_us: stats.output_b_us.saturating_sub(before.output_b_us),
            }
        })
        .filter(|stats| stats.calls > 0)
        .collect()
}

#[cfg(feature = "cuda")]
fn dsv4_output_profile_stats_delta(
    before: DeepSeekV4OutputProfileStats,
    after: DeepSeekV4OutputProfileStats,
) -> DeepSeekV4OutputProfileStats {
    DeepSeekV4OutputProfileStats {
        packed_prefill_batches: after
            .packed_prefill_batches
            .saturating_sub(before.packed_prefill_batches),
        packed_prefill_rows: after
            .packed_prefill_rows
            .saturating_sub(before.packed_prefill_rows),
        packed_decode_batches: after
            .packed_decode_batches
            .saturating_sub(before.packed_decode_batches),
        packed_decode_rows: after
            .packed_decode_rows
            .saturating_sub(before.packed_decode_rows),
        packed_mixed_batches: after
            .packed_mixed_batches
            .saturating_sub(before.packed_mixed_batches),
        packed_mixed_rows: after
            .packed_mixed_rows
            .saturating_sub(before.packed_mixed_rows),
        final_hc_head_calls: after
            .final_hc_head_calls
            .saturating_sub(before.final_hc_head_calls),
        final_hc_head_us: after
            .final_hc_head_us
            .saturating_sub(before.final_hc_head_us),
        final_norm_calls: after
            .final_norm_calls
            .saturating_sub(before.final_norm_calls),
        final_norm_us: after.final_norm_us.saturating_sub(before.final_norm_us),
        lm_head_topk_calls: after
            .lm_head_topk_calls
            .saturating_sub(before.lm_head_topk_calls),
        lm_head_topk_us: after.lm_head_topk_us.saturating_sub(before.lm_head_topk_us),
    }
}

#[cfg(feature = "cuda")]
fn dsv4_prefill_stats_delta(
    before: DeepSeekV4PrefillRuntimeStats,
    after: DeepSeekV4PrefillRuntimeStats,
) -> DeepSeekV4PrefillRuntimeStats {
    DeepSeekV4PrefillRuntimeStats {
        logits_calls: after.logits_calls.saturating_sub(before.logits_calls),
        logits_tokens: after.logits_tokens.saturating_sub(before.logits_tokens),
        no_logits_calls: after.no_logits_calls.saturating_sub(before.no_logits_calls),
        no_logits_tokens: after
            .no_logits_tokens
            .saturating_sub(before.no_logits_tokens),
        interactive_calls: after
            .interactive_calls
            .saturating_sub(before.interactive_calls),
        interactive_tokens: after
            .interactive_tokens
            .saturating_sub(before.interactive_tokens),
        batched_calls: after.batched_calls.saturating_sub(before.batched_calls),
        batched_tokens: after.batched_tokens.saturating_sub(before.batched_tokens),
        start_segment_calls: after
            .start_segment_calls
            .saturating_sub(before.start_segment_calls),
        start_segment_tokens: after
            .start_segment_tokens
            .saturating_sub(before.start_segment_tokens),
        append_segment_calls: after
            .append_segment_calls
            .saturating_sub(before.append_segment_calls),
        append_segment_tokens: after
            .append_segment_tokens
            .saturating_sub(before.append_segment_tokens),
    }
}

#[cfg(feature = "cuda")]
fn sum_layer_profile_stats(stats: &[DeepSeekV4LayerProfileStats]) -> DeepSeekV4LayerProfileStats {
    let mut out = DeepSeekV4LayerProfileStats::default();
    for item in stats {
        out.state_init_calls = out.state_init_calls.saturating_add(item.state_init_calls);
        out.state_init_us = out.state_init_us.saturating_add(item.state_init_us);
        out.decode_calls = out.decode_calls.saturating_add(item.decode_calls);
        out.decode_total_us = out.decode_total_us.saturating_add(item.decode_total_us);
        out.prefill_calls = out.prefill_calls.saturating_add(item.prefill_calls);
        out.prefill_tokens = out.prefill_tokens.saturating_add(item.prefill_tokens);
        out.prefill_total_us = out.prefill_total_us.saturating_add(item.prefill_total_us);
        out.attn_hc_pre_us = out.attn_hc_pre_us.saturating_add(item.attn_hc_pre_us);
        out.attn_norm_us = out.attn_norm_us.saturating_add(item.attn_norm_us);
        out.attention_us = out.attention_us.saturating_add(item.attention_us);
        out.attn_hc_post_us = out.attn_hc_post_us.saturating_add(item.attn_hc_post_us);
        out.ffn_hc_pre_us = out.ffn_hc_pre_us.saturating_add(item.ffn_hc_pre_us);
        out.ffn_norm_us = out.ffn_norm_us.saturating_add(item.ffn_norm_us);
        out.moe_us = out.moe_us.saturating_add(item.moe_us);
        out.ffn_hc_post_us = out.ffn_hc_post_us.saturating_add(item.ffn_hc_post_us);
    }
    out
}

#[cfg(feature = "cuda")]
fn sum_attention_profile_stats(
    stats: &[DeepSeekV4AttentionProfileStats],
) -> DeepSeekV4AttentionProfileStats {
    let mut out = DeepSeekV4AttentionProfileStats::default();
    for item in stats {
        out.calls = out.calls.saturating_add(item.calls);
        out.tokens = out.tokens.saturating_add(item.tokens);
        out.q_a_us = out.q_a_us.saturating_add(item.q_a_us);
        out.q_norm_us = out.q_norm_us.saturating_add(item.q_norm_us);
        out.q_b_us = out.q_b_us.saturating_add(item.q_b_us);
        out.q_head_norm_us = out.q_head_norm_us.saturating_add(item.q_head_norm_us);
        out.q_rope_us = out.q_rope_us.saturating_add(item.q_rope_us);
        out.kv_proj_us = out.kv_proj_us.saturating_add(item.kv_proj_us);
        out.kv_norm_us = out.kv_norm_us.saturating_add(item.kv_norm_us);
        out.kv_rope_quant_us = out.kv_rope_quant_us.saturating_add(item.kv_rope_quant_us);
        out.kv_cache_append_us = out
            .kv_cache_append_us
            .saturating_add(item.kv_cache_append_us);
        out.indexer_compress_us = out
            .indexer_compress_us
            .saturating_add(item.indexer_compress_us);
        out.main_compress_us = out.main_compress_us.saturating_add(item.main_compress_us);
        out.compressed_kv_upload_us = out
            .compressed_kv_upload_us
            .saturating_add(item.compressed_kv_upload_us);
        out.topk_build_us = out.topk_build_us.saturating_add(item.topk_build_us);
        out.sparse_attention_us = out
            .sparse_attention_us
            .saturating_add(item.sparse_attention_us);
        out.context_rope_us = out.context_rope_us.saturating_add(item.context_rope_us);
        out.output_a_us = out.output_a_us.saturating_add(item.output_a_us);
        out.output_b_us = out.output_b_us.saturating_add(item.output_b_us);
    }
    out
}

#[cfg(feature = "cuda")]
fn interactive_turn_json(turn: &InteractiveTurnMeasurement) -> serde_json::Value {
    serde_json::json!({
        "prompt": turn.prompt_text.as_str(),
        "prompt_tokens": turn.prompt_tokens.len(),
        "prompt_token_ids": &turn.prompt_tokens,
        "first_token_s": turn.first_token_us as f64 / 1_000_000.0,
        "prefill_s": turn.prefill_us as f64 / 1_000_000.0,
        "decode_s": turn.decode_us as f64 / 1_000_000.0,
        "prefill_tok_per_s": turn.prompt_tokens.len() as f64 / (turn.prefill_us as f64 / 1_000_000.0).max(1e-6),
        "decode_tok_per_s": turn.generated_tokens.len() as f64 / (turn.decode_us as f64 / 1_000_000.0).max(1e-6),
        "generated_tokens": turn.generated_tokens.len(),
        "generated_token_ids": &turn.generated_tokens,
        "final_position": turn.final_position,
        "finish_reason": turn.finish_reason.as_str(),
        "stopped_by_eos": turn.stopped_by_eos,
        "stopped_by_string": &turn.stopped_by_string,
        "runtime_driver_stats": resident_driver_stats_json(&turn.runtime_driver_stats),
        "dsv4_operator_counters": dsv4_operator_counters_json(&turn.dsv4_operator_counters),
        "dsv4_prefill_stats": dsv4_prefill_stats_json(&turn.dsv4_prefill_stats),
        "dsv4_layer_profile_summary": dsv4_layer_profile_summary_json(&turn.dsv4_layer_profile_stats),
        "dsv4_layer_profile": dsv4_layer_profile_stats_json(&turn.dsv4_layer_profile_stats),
        "dsv4_attention_profile_summary": dsv4_attention_profile_summary_json(&turn.dsv4_attention_profile_stats),
        "dsv4_attention_profile": dsv4_attention_profile_stats_json(&turn.dsv4_attention_profile_stats),
        "dsv4_output_profile": dsv4_output_profile_stats_json(&turn.dsv4_output_profile_stats),
        "runtime_steps": turn.runtime_steps.iter().map(runtime_step_json).collect::<Vec<_>>(),
    })
}

#[cfg(feature = "cuda")]
fn runtime_step_json(step: &RuntimeStepMeasurement) -> serde_json::Value {
    serde_json::json!({
        "action_kind": step.action_kind.as_str(),
        "rows": step.rows,
        "staged": step.staged,
        "finished": step.finished,
        "elapsed_s": step.elapsed_us as f64 / 1_000_000.0,
        "runner_position": step.runner_position,
        "dsv4_operator_counters": dsv4_operator_counters_json(&step.dsv4_operator_counters),
        "dsv4_layer_profile_summary": dsv4_layer_profile_summary_json(&step.dsv4_layer_profile_stats),
        "dsv4_attention_profile_summary": dsv4_attention_profile_summary_json(&step.dsv4_attention_profile_stats),
        "dsv4_output_profile": dsv4_output_profile_stats_json(&step.dsv4_output_profile_stats),
    })
}

#[cfg(feature = "cuda")]
fn dsv4_layer_profile_summary_json(stats: &[DeepSeekV4LayerProfileStats]) -> serde_json::Value {
    dsv4_layer_profile_stats_json_one(&sum_layer_profile_stats(stats))
}

#[cfg(feature = "cuda")]
fn dsv4_layer_profile_stats_json(stats: &[DeepSeekV4LayerProfileStats]) -> serde_json::Value {
    serde_json::Value::Array(
        stats
            .iter()
            .map(dsv4_layer_profile_stats_json_one)
            .collect::<Vec<_>>(),
    )
}

#[cfg(feature = "cuda")]
fn dsv4_layer_profile_stats_json_one(stats: &DeepSeekV4LayerProfileStats) -> serde_json::Value {
    serde_json::json!({
        "layer": stats.layer,
        "state_init_calls": stats.state_init_calls,
        "state_init_s": stats.state_init_us as f64 / 1_000_000.0,
        "decode_calls": stats.decode_calls,
        "decode_total_s": stats.decode_total_us as f64 / 1_000_000.0,
        "prefill_calls": stats.prefill_calls,
        "prefill_tokens": stats.prefill_tokens,
        "prefill_total_s": stats.prefill_total_us as f64 / 1_000_000.0,
        "attn_hc_pre_s": stats.attn_hc_pre_us as f64 / 1_000_000.0,
        "attn_norm_s": stats.attn_norm_us as f64 / 1_000_000.0,
        "attention_s": stats.attention_us as f64 / 1_000_000.0,
        "attn_hc_post_s": stats.attn_hc_post_us as f64 / 1_000_000.0,
        "ffn_hc_pre_s": stats.ffn_hc_pre_us as f64 / 1_000_000.0,
        "ffn_norm_s": stats.ffn_norm_us as f64 / 1_000_000.0,
        "moe_s": stats.moe_us as f64 / 1_000_000.0,
        "ffn_hc_post_s": stats.ffn_hc_post_us as f64 / 1_000_000.0,
    })
}

#[cfg(feature = "cuda")]
fn dsv4_attention_profile_summary_json(
    stats: &[DeepSeekV4AttentionProfileStats],
) -> serde_json::Value {
    dsv4_attention_profile_stats_json_one(&sum_attention_profile_stats(stats))
}

#[cfg(feature = "cuda")]
fn dsv4_attention_profile_stats_json(
    stats: &[DeepSeekV4AttentionProfileStats],
) -> serde_json::Value {
    serde_json::Value::Array(
        stats
            .iter()
            .map(dsv4_attention_profile_stats_json_one)
            .collect::<Vec<_>>(),
    )
}

#[cfg(feature = "cuda")]
fn dsv4_attention_profile_stats_json_one(
    stats: &DeepSeekV4AttentionProfileStats,
) -> serde_json::Value {
    serde_json::json!({
        "layer": stats.layer,
        "calls": stats.calls,
        "tokens": stats.tokens,
        "q_a_s": stats.q_a_us as f64 / 1_000_000.0,
        "q_norm_s": stats.q_norm_us as f64 / 1_000_000.0,
        "q_b_s": stats.q_b_us as f64 / 1_000_000.0,
        "q_head_norm_s": stats.q_head_norm_us as f64 / 1_000_000.0,
        "q_rope_s": stats.q_rope_us as f64 / 1_000_000.0,
        "kv_proj_s": stats.kv_proj_us as f64 / 1_000_000.0,
        "kv_norm_s": stats.kv_norm_us as f64 / 1_000_000.0,
        "kv_rope_quant_s": stats.kv_rope_quant_us as f64 / 1_000_000.0,
        "kv_cache_append_s": stats.kv_cache_append_us as f64 / 1_000_000.0,
        "indexer_compress_s": stats.indexer_compress_us as f64 / 1_000_000.0,
        "main_compress_s": stats.main_compress_us as f64 / 1_000_000.0,
        "compressed_kv_upload_s": stats.compressed_kv_upload_us as f64 / 1_000_000.0,
        "topk_build_s": stats.topk_build_us as f64 / 1_000_000.0,
        "sparse_attention_s": stats.sparse_attention_us as f64 / 1_000_000.0,
        "context_rope_s": stats.context_rope_us as f64 / 1_000_000.0,
        "output_a_s": stats.output_a_us as f64 / 1_000_000.0,
        "output_b_s": stats.output_b_us as f64 / 1_000_000.0,
    })
}

#[cfg(feature = "cuda")]
fn dsv4_output_profile_stats_json(stats: &DeepSeekV4OutputProfileStats) -> serde_json::Value {
    serde_json::json!({
        "final_hc_head_calls": stats.final_hc_head_calls,
        "final_hc_head_s": stats.final_hc_head_us as f64 / 1_000_000.0,
        "final_norm_calls": stats.final_norm_calls,
        "final_norm_s": stats.final_norm_us as f64 / 1_000_000.0,
        "lm_head_topk_calls": stats.lm_head_topk_calls,
        "lm_head_topk_s": stats.lm_head_topk_us as f64 / 1_000_000.0,
    })
}

#[cfg(feature = "cuda")]
fn resident_driver_stats_json(stats: &ResidentTopKDriverStats) -> serde_json::Value {
    serde_json::json!({
        "actions": stats.actions,
        "prefill_chunks": stats.prefill_chunks,
        "prefill_tokens": stats.prefill_tokens,
        "decode_steps": stats.decode_steps,
        "emitted_tokens": stats.emitted_tokens,
        "staged_tokens": stats.staged_tokens,
        "finished_sequences": stats.finished_sequences,
    })
}

#[cfg(feature = "cuda")]
fn dsv4_operator_counters_json(stats: &DeepSeekV4OperatorRuntimeCounters) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    {
        let mut u64_field = |key: &str, value: u64| {
            out.insert(key.into(), serde_json::Value::from(value));
        };
        u64_field("kernel_launches", stats.kernel_launches);
        u64_field("host_to_device_copies", stats.host_to_device_copies);
        u64_field("host_to_device_bytes", stats.host_to_device_bytes);
        u64_field("device_to_host_copies", stats.device_to_host_copies);
        u64_field("device_to_host_bytes", stats.device_to_host_bytes);
        u64_field("artifact_uploads", stats.artifact_uploads);
        u64_field("artifact_upload_bytes", stats.artifact_upload_bytes);
        u64_field(
            "device_allocation_attempts",
            stats.device_allocation_attempts,
        );
        u64_field("device_allocations", stats.device_allocations);
        u64_field(
            "device_allocation_failures",
            stats.device_allocation_failures,
        );
        u64_field("device_allocation_bytes", stats.device_allocation_bytes);
        u64_field("stream_wide_syncs", stats.stream_wide_syncs);
        u64_field("stream_wide_sync_failures", stats.stream_wide_sync_failures);
        u64_field("moe_calls", stats.moe_calls);
        u64_field("moe_tc_calls", stats.moe_tc_calls);
        u64_field("moe_scalar_calls", stats.moe_scalar_calls);
        u64_field("moe_reduce_calls", stats.moe_reduce_calls);
        u64_field("moe_predicted_experts", stats.moe_predicted_experts);
        u64_field("moe_prefetch_loads", stats.moe_prefetch_loads);
        u64_field("moe_prefetch_enqueued", stats.moe_prefetch_enqueued);
        u64_field(
            "moe_prefetch_skipped_cached_or_inflight",
            stats.moe_prefetch_skipped_cached_or_inflight,
        );
        u64_field("moe_prefetch_resident", stats.moe_prefetch_resident);
        u64_field(
            "moe_prefetch_materializing",
            stats.moe_prefetch_materializing,
        );
        u64_field("moe_prefetch_host_staged", stats.moe_prefetch_host_staged);
        u64_field("moe_prefetch_in_flight", stats.moe_prefetch_in_flight);
        u64_field("moe_prefetch_cold", stats.moe_prefetch_cold);
        u64_field(
            "expert_selected_resident_hits",
            stats.expert_selected_resident_hits,
        );
        u64_field(
            "expert_selected_upload_hits",
            stats.expert_selected_upload_hits,
        );
        u64_field(
            "expert_selected_host_staged_hits",
            stats.expert_selected_host_staged_hits,
        );
        u64_field(
            "expert_selected_host_staging_waits",
            stats.expert_selected_host_staging_waits,
        );
        u64_field(
            "expert_selected_host_staging_hits",
            stats.expert_selected_host_staging_hits,
        );
        u64_field(
            "expert_selected_host_staging_wait_us",
            stats.expert_selected_host_staging_wait_us,
        );
        u64_field(
            "expert_selected_cold_misses",
            stats.expert_selected_cold_misses,
        );
        u64_field(
            "expert_upload_prefetch_submitted",
            stats.expert_upload_prefetch_submitted,
        );
        u64_field(
            "expert_upload_prefetch_completed",
            stats.expert_upload_prefetch_completed,
        );

        u64_field(
            "expert_selected_upload_waits",
            stats.expert_selected_upload_waits,
        );
        u64_field("expert_async_upload_bytes", stats.expert_async_upload_bytes);
        u64_field(
            "expert_lookahead_prefetch_calls",
            stats.expert_lookahead_prefetch_calls,
        );
        u64_field(
            "expert_lookahead_prefetch_experts",
            stats.expert_lookahead_prefetch_experts,
        );
        u64_field(
            "expert_lookahead_prefetch_enqueued",
            stats.expert_lookahead_prefetch_enqueued,
        );

        u64_field("output_head_calls", stats.output_head_calls);
        u64_field("output_head_chunks", stats.output_head_chunks);
        u64_field("output_head_rows", stats.output_head_rows);
        u64_field("output_head_cache_hits", stats.output_head_cache_hits);
        u64_field("output_head_cache_misses", stats.output_head_cache_misses);
        u64_field(
            "output_head_hidden_uploads",
            stats.output_head_hidden_uploads,
        );
        u64_field("expert_selected", stats.expert_selected);
        u64_field(
            "expert_selected_load_requests",
            stats.expert_selected_load_requests,
        );
        u64_field("expert_loads", stats.expert_loads);
        u64_field("expert_load_bytes", stats.expert_load_bytes);
        u64_field("expert_evictions", stats.expert_evictions);
        u64_field("expert_host_cache_hits", stats.expert_host_cache.hits);
        u64_field("expert_host_cache_misses", stats.expert_host_cache.misses);
        u64_field(
            "expert_host_cache_evictions",
            stats.expert_host_cache.evictions,
        );
        u64_field(
            "expert_host_cache_rejections",
            stats.expert_host_cache.rejections,
        );
        u64_field(
            "expert_host_cache_bytes",
            stats.expert_host_cache.bytes_used,
        );
        u64_field(
            "expert_host_cache_lifetime_peak_bytes",
            stats.expert_host_cache.peak_bytes_used,
        );
        u64_field("expert_pinned_cache_hits", stats.expert_pinned_cache.hits);
        u64_field(
            "expert_pinned_cache_misses",
            stats.expert_pinned_cache.misses,
        );
        u64_field(
            "expert_pinned_cache_evictions",
            stats.expert_pinned_cache.evictions,
        );
        u64_field(
            "expert_pinned_cache_rejections",
            stats.expert_pinned_cache.rejections,
        );
        u64_field(
            "expert_pinned_cache_bytes",
            stats.expert_pinned_cache.bytes_used,
        );
        u64_field(
            "expert_pinned_cache_lifetime_peak_bytes",
            stats.expert_pinned_cache.peak_bytes_used,
        );
        u64_field(
            "expert_cuda_resident_bytes",
            stats.expert_cuda_resident_bytes,
        );
        u64_field(
            "expert_async_prefetch_submitted",
            stats.expert_async_prefetch_submitted,
        );
        u64_field(
            "expert_async_prefetch_completed",
            stats.expert_async_prefetch_completed,
        );
        u64_field(
            "expert_async_prefetch_failed",
            stats.expert_async_prefetch_failed,
        );
        u64_field(
            "expert_async_prefetch_skipped",
            stats.expert_async_prefetch_skipped,
        );
        u64_field("arena_hits", stats.arena_hits);
        u64_field("arena_misses", stats.arena_misses);
        u64_field("arena_grows", stats.arena_grows);
        u64_field("arena_reuses", stats.arena_reuses);
        u64_field(
            "expert_residency_installs",
            stats.expert_residency_stats.installs,
        );
        u64_field(
            "expert_residency_evictions",
            stats.expert_residency_stats.evictions,
        );
        u64_field(
            "expert_residency_resident_hits",
            stats.expert_residency_stats.resident_hits,
        );
        u64_field(
            "expert_residency_stale_releases",
            stats.expert_residency_stats.stale_releases,
        );
        u64_field(
            "expert_residency_prepare_cancellations",
            stats.expert_residency_stats.prepare_cancellations,
        );
        u64_field(
            "expert_residency_prefetch_capacity_misses",
            stats.expert_residency_stats.prefetch_capacity_misses,
        );
        u64_field(
            "expert_predictor_predict_calls",
            stats.expert_predictor_stats.predict_calls,
        );
        u64_field(
            "expert_predictor_predicted_experts",
            stats.expert_predictor_stats.predicted_experts,
        );
        u64_field(
            "expert_predictor_observe_calls",
            stats.expert_predictor_stats.observe_calls,
        );
        u64_field(
            "expert_predictor_observed_experts",
            stats.expert_predictor_stats.observed_experts,
        );
        u64_field(
            "expert_predictor_cold_miss_observations",
            stats.expert_predictor_stats.cold_miss_observations,
        );
        u64_field(
            "expert_predictor_transition_observations",
            stats.expert_predictor_stats.transition_observations,
        );
    }
    out.insert(
        "expert_residency_resident".into(),
        serde_json::Value::from(stats.expert_residency_stats.resident),
    );
    out.insert(
        "expert_residency_active_leases".into(),
        serde_json::Value::from(stats.expert_residency_stats.active_leases),
    );
    out.insert(
        "expert_host_cache_entries".into(),
        serde_json::Value::from(stats.expert_host_cache.entries_used),
    );
    out.insert(
        "expert_pinned_cache_entries".into(),
        serde_json::Value::from(stats.expert_pinned_cache.entries_used),
    );
    out.insert(
        "expert_cuda_resident_entries".into(),
        serde_json::Value::from(stats.expert_cuda_resident_entries),
    );
    out.insert(
        "expert_async_prefetch_in_flight".into(),
        serde_json::Value::from(stats.expert_async_prefetch_in_flight),
    );
    out.insert(
        "expert_upload_prefetch_in_flight".into(),
        serde_json::Value::from(stats.expert_upload_prefetch_in_flight),
    );

    {
        let mut seconds_field = |key: &str, value: u64| {
            out.insert(
                key.into(),
                serde_json::Value::from(value as f64 / 1_000_000.0),
            );
        };
        seconds_field("moe_total_s", stats.moe_total_us);
        seconds_field("moe_input_prepare_s", stats.moe_input_prepare_us);
        seconds_field("moe_gate_up_s", stats.moe_gate_up_us);
        seconds_field("moe_swiglu_s", stats.moe_swiglu_us);
        seconds_field("moe_hidden_pack_s", stats.moe_hidden_pack_us);
        seconds_field("moe_down_s", stats.moe_down_us);
        seconds_field("moe_router_s", stats.moe_router_us);
        seconds_field("moe_routing_s", stats.moe_routing_us);
        seconds_field("moe_plan_s", stats.moe_plan_us);
        seconds_field(
            "expert_lookahead_prefetch_s",
            stats.expert_lookahead_prefetch_us,
        );
        seconds_field("moe_cache_lookup_s", stats.moe_cache_lookup_us);
        seconds_field(
            "expert_selected_upload_wait_s",
            stats.expert_selected_upload_wait_us,
        );
        seconds_field(
            "expert_selected_host_staging_wait_s",
            stats.expert_selected_host_staging_wait_us,
        );
        seconds_field("moe_expert_read_s", stats.moe_expert_read_us);
        seconds_field("moe_expert_upload_s", stats.moe_expert_upload_us);
        seconds_field("moe_shared_s", stats.moe_shared_us);
        seconds_field("moe_workspace_s", stats.moe_workspace_us);
        seconds_field("moe_compute_submit_s", stats.moe_compute_submit_us);
        seconds_field("moe_commit_s", stats.moe_commit_us);
        seconds_field(
            "output_head_hidden_upload_s",
            stats.output_head_hidden_upload_us,
        );
        seconds_field("output_head_read_s", stats.output_head_read_us);
        seconds_field("output_head_upload_s", stats.output_head_upload_us);
        seconds_field("output_head_topk_s", stats.output_head_topk_us);
        seconds_field("output_head_merge_s", stats.output_head_merge_us);
    }

    serde_json::Value::Object(out)
}

#[cfg(feature = "cuda")]
fn dsv4_prefill_stats_json(stats: &DeepSeekV4PrefillRuntimeStats) -> serde_json::Value {
    serde_json::json!({
        "logits_calls": stats.logits_calls,
        "logits_tokens": stats.logits_tokens,
        "no_logits_calls": stats.no_logits_calls,
        "no_logits_tokens": stats.no_logits_tokens,
        "interactive_calls": stats.interactive_calls,
        "interactive_tokens": stats.interactive_tokens,
        "batched_calls": stats.batched_calls,
        "batched_tokens": stats.batched_tokens,
        "start_segment_calls": stats.start_segment_calls,
        "start_segment_tokens": stats.start_segment_tokens,
        "append_segment_calls": stats.append_segment_calls,
        "append_segment_tokens": stats.append_segment_tokens,
    })
}

#[cfg(not(feature = "cuda"))]
pub fn cmd_bench_interactive(
    _model_dir: &str,
    _prompts: &[String],
    _max_new_tokens: usize,
    _chat_template_override: Option<&str>,
    _warmup_tokens: usize,
    _max_layers: usize,
    _prefill_chunk_size: usize,
    _golden_trace_path: Option<&str>,
    _json: bool,
) -> anyhow::Result<()> {
    anyhow::bail!("bench-interactive requires --features cuda")
}

#[cfg(feature = "cuda")]
fn dsv4_expert_prewarm_enabled() -> bool {
    std::env::var("FERRULE_DSV4_EXPERT_PREWARM")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            value == "1" || value == "true" || value == "on"
        })
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn duration_us(d: Duration) -> u64 {
    d.as_micros().min(u128::from(u64::MAX)) as u64
}
