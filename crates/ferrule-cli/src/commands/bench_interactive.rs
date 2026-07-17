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
use std::num::NonZeroU32;
#[cfg(feature = "cuda")]
use std::path::Path;
#[cfg(feature = "cuda")]
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use crate::GenerationConfig;
#[cfg(feature = "cuda")]
use crate::bench::{GoldenTurn, InteractiveTrace, compare_interactive_trace};
#[cfg(feature = "cuda")]
use ferrule_common::{
    MemoryPoolStats,
    execution::{ExecutionOutput, ForwardPhase, LogitsOutput, LogitsRequest, StateSlot},
};
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
    GenerateRequest, PageManagedDiagnosticHarness, RequestId, ResidentActionKind,
    ResidentDriverStep, ResidentSchedulerConfig, ResidentTopKDriverConfig, ResidentTopKDriverStats,
    SessionId,
};

#[cfg(feature = "cuda")]
use super::resident::{build_page_managed_diagnostic_harness, build_resident_topk_driver};

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
    output_head_chunk_rows: usize,
    moe_prefetch_experts: usize,
    moe_hotset_experts: usize,
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
    output_head_chunk_rows: usize,
    moe_prefetch_experts: usize,
    moe_hotset_experts: usize,
    golden_trace_path: Option<&str>,
    json: bool,
    resident_replay: bool,
    verify_width_sweep: bool,
    verify_iterations: usize,
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
        ctx_size: 4096,
        append_eos_to_session: true,
        ..GenerationConfig::default()
    };

    let options = DeepSeekV4PrepareOptions {
        max_layers,
        output_head_chunk_rows,
        moe_prefetch_experts,
        moe_hotset_experts,
        ..DeepSeekV4PrepareOptions::default()
    };

    // ── Phase 1: load ────────────────────────────────────────────────────
    let output_head_chunk_bytes = u64::try_from(output_head_chunk_rows)
        .ok()
        .and_then(|rows| rows.checked_mul(4096))
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("output-head chunk byte size overflow"))?;
    let max_tensor_bytes = output_head_chunk_bytes.max(128 * 1024 * 1024);
    let load_start = Instant::now();
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(model_path, max_tensor_bytes)?;
    let checkpoint_dspark_block_size = model.config.dspark_block_size;
    let runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;
    let artifact_load_us = duration_us(load_start.elapsed());

    if verify_width_sweep {
        return run_resident_verify_width_sweep(
            runner,
            model_dir,
            &chat_template,
            prompts,
            artifact_load_us,
            max_layers,
            output_head_chunk_rows,
            moe_prefetch_experts,
            moe_hotset_experts,
            checkpoint_dspark_block_size,
            verify_iterations,
            json,
        );
    }
    if resident_replay {
        return run_resident_replay(
            runner,
            model_dir,
            &chat_template,
            prompts,
            artifact_load_us,
            max_layers,
            output_head_chunk_rows,
            moe_prefetch_experts,
            moe_hotset_experts,
            json,
        );
    }

    let mut report = InteractiveBenchReport {
        model_dir: model_dir.to_string(),
        chat_template: chat_template.name().to_string(),
        max_new_tokens,
        max_layers,
        prefill_chunk_size,
        output_head_chunk_rows,
        moe_prefetch_experts,
        moe_hotset_experts,
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
        out["output_head_chunk_rows"] = serde_json::json!(report.output_head_chunk_rows);
        out["moe_prefetch_experts"] = serde_json::json!(report.moe_prefetch_experts);
        out["moe_hotset_experts"] = serde_json::json!(report.moe_hotset_experts);

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
        dspark: Default::default(),
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
        expert_prefetch_outstanding: after.expert_prefetch_outstanding,
        expert_prefetch_slot_reservations: after.expert_prefetch_slot_reservations,
        expert_prefetch_host_queued: after.expert_prefetch_host_queued,
        expert_abandoned_uploads: after.expert_abandoned_uploads,
        expert_frame_reuses: after
            .expert_frame_reuses
            .saturating_sub(before.expert_frame_reuses),
        expert_frame_waits: after
            .expert_frame_waits
            .saturating_sub(before.expert_frame_waits),
        expert_free_frames: after.expert_free_frames,
        expert_io_submitted_extents: after
            .expert_io_submitted_extents
            .saturating_sub(before.expert_io_submitted_extents),
        expert_io_completed_extents: after
            .expert_io_completed_extents
            .saturating_sub(before.expert_io_completed_extents),
        expert_io_failed_extents: after
            .expert_io_failed_extents
            .saturating_sub(before.expert_io_failed_extents),
        expert_io_requested_bytes: after
            .expert_io_requested_bytes
            .saturating_sub(before.expert_io_requested_bytes),
        expert_io_aligned_bytes: after
            .expert_io_aligned_bytes
            .saturating_sub(before.expert_io_aligned_bytes),
        expert_io_coalesced_slices: after
            .expert_io_coalesced_slices
            .saturating_sub(before.expert_io_coalesced_slices),
        expert_io_fixed_file_registrations: after
            .expert_io_fixed_file_registrations
            .saturating_sub(before.expert_io_fixed_file_registrations),
        expert_io_fallback_count: after
            .expert_io_fallback_count
            .saturating_sub(before.expert_io_fallback_count),
        expert_io_slab_exhaustions: after
            .expert_io_slab_exhaustions
            .saturating_sub(before.expert_io_slab_exhaustions),
        expert_io_peak_queue_depth: after.expert_io_peak_queue_depth,
        expert_io_read_us: after
            .expert_io_read_us
            .saturating_sub(before.expert_io_read_us),
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
        transition_predictions: after
            .transition_predictions
            .saturating_sub(before.transition_predictions),
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
        u64_field("expert_frame_reuses", stats.expert_frame_reuses);
        u64_field("expert_frame_waits", stats.expert_frame_waits);
        u64_field(
            "expert_io_submitted_extents",
            stats.expert_io_submitted_extents,
        );
        u64_field(
            "expert_io_completed_extents",
            stats.expert_io_completed_extents,
        );
        u64_field("expert_io_failed_extents", stats.expert_io_failed_extents);
        u64_field("expert_io_requested_bytes", stats.expert_io_requested_bytes);
        u64_field("expert_io_aligned_bytes", stats.expert_io_aligned_bytes);
        u64_field(
            "expert_io_coalesced_slices",
            stats.expert_io_coalesced_slices,
        );
        u64_field(
            "expert_io_fixed_file_registrations",
            stats.expert_io_fixed_file_registrations,
        );
        u64_field("expert_io_fallback_count", stats.expert_io_fallback_count);
        u64_field(
            "expert_io_slab_exhaustions",
            stats.expert_io_slab_exhaustions,
        );
        u64_field("expert_io_read_us", stats.expert_io_read_us);

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
        u64_field(
            "expert_predictor_transition_predictions",
            stats.expert_predictor_stats.transition_predictions,
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
    out.insert(
        "expert_prefetch_outstanding".into(),
        serde_json::Value::from(stats.expert_prefetch_outstanding),
    );
    out.insert(
        "expert_prefetch_slot_reservations".into(),
        serde_json::Value::from(stats.expert_prefetch_slot_reservations),
    );
    out.insert(
        "expert_prefetch_host_queued".into(),
        serde_json::Value::from(stats.expert_prefetch_host_queued),
    );
    out.insert(
        "expert_abandoned_uploads".into(),
        serde_json::Value::from(stats.expert_abandoned_uploads),
    );
    out.insert(
        "expert_free_frames".into(),
        serde_json::Value::from(stats.expert_free_frames),
    );
    out.insert(
        "expert_io_peak_queue_depth".into(),
        serde_json::Value::from(stats.expert_io_peak_queue_depth),
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

#[cfg(feature = "cuda")]
struct ResidentReplayMeasurement {
    label: &'static str,
    elapsed_us: u64,
    counters: DeepSeekV4OperatorRuntimeCounters,
    layer_profile: Vec<DeepSeekV4LayerProfileStats>,
    attention_profile: Vec<DeepSeekV4AttentionProfileStats>,
    output_profile: DeepSeekV4OutputProfileStats,
    top1: Option<(u32, f32)>,
}

#[cfg(feature = "cuda")]
fn measure_resident_replay_topk(
    diagnostic: &mut PageManagedDiagnosticHarness<DeepSeekV4Runner>,
    slot: StateSlot,
    token_id: u32,
    label: &'static str,
) -> anyhow::Result<ResidentReplayMeasurement> {
    let counters_before = diagnostic.runner().operator_runtime_counters();
    let layer_before = diagnostic.runner().layer_profile_stats();
    let attention_before = diagnostic.runner().attention_profile_stats();
    let output_before = diagnostic.runner().output_profile_stats();
    let started = Instant::now();
    let top = diagnostic.execute_sequence_step(
        slot,
        ForwardPhase::Decode,
        std::slice::from_ref(&token_id),
        |runner| runner.decode_token_topk(token_id, 1),
    )?;
    let elapsed_us = duration_us(started.elapsed());
    let top1 = top
        .first()
        .map(|item| (item.token_id, item.logit))
        .ok_or_else(|| anyhow::anyhow!("resident replay top-k pass produced no token"))?;
    Ok(ResidentReplayMeasurement {
        label,
        elapsed_us,
        counters: dsv4_operator_counters_delta(
            counters_before,
            diagnostic.runner().operator_runtime_counters(),
        ),
        layer_profile: dsv4_layer_profile_stats_delta(
            &layer_before,
            &diagnostic.runner().layer_profile_stats(),
        ),
        attention_profile: dsv4_attention_profile_stats_delta(
            &attention_before,
            &diagnostic.runner().attention_profile_stats(),
        ),
        output_profile: dsv4_output_profile_stats_delta(
            output_before,
            diagnostic.runner().output_profile_stats(),
        ),
        top1: Some(top1),
    })
}

#[cfg(feature = "cuda")]
fn measure_resident_replay_body(
    diagnostic: &mut PageManagedDiagnosticHarness<DeepSeekV4Runner>,
    slot: StateSlot,
    token_id: u32,
) -> anyhow::Result<ResidentReplayMeasurement> {
    let counters_before = diagnostic.runner().operator_runtime_counters();
    let layer_before = diagnostic.runner().layer_profile_stats();
    let attention_before = diagnostic.runner().attention_profile_stats();
    let output_before = diagnostic.runner().output_profile_stats();
    let started = Instant::now();
    diagnostic.execute_sequence_step(
        slot,
        ForwardPhase::Decode,
        std::slice::from_ref(&token_id),
        |runner| runner.feed_token(token_id),
    )?;
    let elapsed_us = duration_us(started.elapsed());
    Ok(ResidentReplayMeasurement {
        label: "resident_body_without_output_head",
        elapsed_us,
        counters: dsv4_operator_counters_delta(
            counters_before,
            diagnostic.runner().operator_runtime_counters(),
        ),
        layer_profile: dsv4_layer_profile_stats_delta(
            &layer_before,
            &diagnostic.runner().layer_profile_stats(),
        ),
        attention_profile: dsv4_attention_profile_stats_delta(
            &attention_before,
            &diagnostic.runner().attention_profile_stats(),
        ),
        output_profile: dsv4_output_profile_stats_delta(
            output_before,
            diagnostic.runner().output_profile_stats(),
        ),
        top1: None,
    })
}

#[cfg(feature = "cuda")]
fn resident_replay_measurement_json(measurement: &ResidentReplayMeasurement) -> serde_json::Value {
    serde_json::json!({
        "label": measurement.label,
        "elapsed_s": measurement.elapsed_us as f64 / 1_000_000.0,
        "target_passes_per_s": if measurement.elapsed_us == 0 {
            0.0
        } else {
            1_000_000.0 / measurement.elapsed_us as f64
        },
        "top1_token_id": measurement.top1.map(|top1| top1.0),
        "top1_logit": measurement.top1.map(|top1| top1.1),
        "dsv4_operator_counters": dsv4_operator_counters_json(&measurement.counters),
        "dsv4_layer_profile_summary": dsv4_layer_profile_summary_json(&measurement.layer_profile),
        "dsv4_attention_profile_summary": dsv4_attention_profile_summary_json(&measurement.attention_profile),
        "dsv4_output_profile": dsv4_output_profile_stats_json(&measurement.output_profile),
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_resident_replay(
    runner: DeepSeekV4Runner,
    model_dir: &str,
    chat_template: &ChatTemplate,
    prompts: &[String],
    artifact_load_us: u64,
    max_layers: usize,
    output_head_chunk_rows: usize,
    moe_prefetch_experts: usize,
    moe_hotset_experts: usize,
    json: bool,
) -> anyhow::Result<()> {
    if prompts.len() != 1 {
        anyhow::bail!(
            "--resident-replay requires exactly one --prompt, got {}",
            prompts.len()
        );
    }
    if moe_prefetch_experts != 0 {
        anyhow::bail!(
            "--resident-replay requires --moe-prefetch-experts 0 so capture installs only exact selected experts"
        );
    }
    let full_prompt = chat_template.format_turn(&prompts[0], true);
    let token_ids = runner.encode(&full_prompt)?;
    if token_ids.len() < 2 {
        anyhow::bail!(
            "--resident-replay requires a prompt encoding to at least two tokens, got {}",
            token_ids.len()
        );
    }
    let decode_token_id = *token_ids.last().expect("token length checked above");
    let prefix = &token_ids[..token_ids.len() - 1];
    let schema = runner.kv_layout_schema().clone();
    let mut diagnostic =
        build_page_managed_diagnostic_harness(runner, Box::new(schema), token_ids.len(), 3)?;
    let capture_slot = diagnostic.create_sequence(0)?;
    let resident_head_slot = diagnostic.create_sequence(0)?;
    let resident_body_slot = diagnostic.create_sequence(0)?;

    for slot in [capture_slot, resident_head_slot, resident_body_slot] {
        diagnostic.execute_sequence_step(slot, ForwardPhase::Prefill, prefix, |runner| {
            runner.prefill_tokens_topk_batched(prefix, 1).map(|_| ())
        })?;
    }

    let capture = measure_resident_replay_topk(
        &mut diagnostic,
        capture_slot,
        decode_token_id,
        "capture_with_output_head",
    )?;
    let resident_head = measure_resident_replay_topk(
        &mut diagnostic,
        resident_head_slot,
        decode_token_id,
        "resident_with_output_head",
    )?;
    let resident_body =
        measure_resident_replay_body(&mut diagnostic, resident_body_slot, decode_token_id)?;

    let capture_top1 = capture.top1.expect("top-k capture records top-1");
    let resident_top1 = resident_head.top1.expect("top-k replay records top-1");
    let token_equal = capture_top1.0 == resident_top1.0;
    let logit_bits_equal = capture_top1.1.to_bits() == resident_top1.1.to_bits();
    let resident_no_io = resident_head.counters.expert_load_bytes == 0
        && resident_head.counters.expert_selected_cold_misses == 0
        && resident_body.counters.expert_load_bytes == 0
        && resident_body.counters.expert_selected_cold_misses == 0;
    if !token_equal || !logit_bits_equal {
        anyhow::bail!(
            "resident replay changed top-1: capture=({}, {:?}) resident=({}, {:?})",
            capture_top1.0,
            capture_top1.1,
            resident_top1.0,
            resident_top1.1
        );
    }
    if !resident_no_io {
        anyhow::bail!(
            "resident replay still performed selected expert I/O: with_head bytes={} cold={} body bytes={} cold={}",
            resident_head.counters.expert_load_bytes,
            resident_head.counters.expert_selected_cold_misses,
            resident_body.counters.expert_load_bytes,
            resident_body.counters.expert_selected_cold_misses
        );
    }

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "mode": "resident_no_io_replay",
                "model": model_dir,
                "prompt": prompts[0],
                "chat_template": chat_template.name(),
                "prompt_token_ids": token_ids,
                "prefix_tokens": prefix.len(),
                "decode_token_id": decode_token_id,
                "max_layers": max_layers,
                "output_head_chunk_rows": output_head_chunk_rows,
                "moe_prefetch_experts": moe_prefetch_experts,
                "moe_hotset_experts": moe_hotset_experts,
                "artifact_load_s": artifact_load_us as f64 / 1_000_000.0,
                "parity": {
                    "token_equal": token_equal,
                    "logit_bits_equal": logit_bits_equal,
                    "capture_top1_token_id": capture_top1.0,
                    "resident_top1_token_id": resident_top1.0,
                },
                "resident_no_io": resident_no_io,
                "capture": resident_replay_measurement_json(&capture),
                "resident_with_head": resident_replay_measurement_json(&resident_head),
                "resident_body_without_output_head": resident_replay_measurement_json(&resident_body),
            }))?
        );
    } else {
        println!("=== DSV4 Resident/No-I/O Replay ===");
        println!("prompt:             {:?}", prompts[0]);
        println!("prefix/decode:      {}/{}", prefix.len(), decode_token_id);
        println!("top-1:              {}", capture_top1.0);
        println!("resident_no_io:     {resident_no_io}");
        for measurement in [&capture, &resident_head, &resident_body] {
            println!(
                "{:<28} {:>8.3} ms  loads={:<4} bytes={:<12} cold={}",
                measurement.label,
                measurement.elapsed_us as f64 / 1_000.0,
                measurement.counters.expert_loads,
                measurement.counters.expert_load_bytes,
                measurement.counters.expert_selected_cold_misses,
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
struct ResidentVerifySample {
    elapsed_us: u64,
    counters: DeepSeekV4OperatorRuntimeCounters,
    layer_profile: Vec<DeepSeekV4LayerProfileStats>,
    attention_profile: Vec<DeepSeekV4AttentionProfileStats>,
    output_profile: DeepSeekV4OutputProfileStats,
    top1: Vec<(u32, f32)>,
}

#[cfg(feature = "cuda")]
struct ResidentVerifyWidthMeasurement {
    width: usize,
    capture_top1: Vec<(u32, f32)>,
    samples: Vec<ResidentVerifySample>,
    parity: bool,
    resident_no_io: bool,
    allocation_free: bool,
}

#[cfg(feature = "cuda")]
fn packed_output_top1(output: &ExecutionOutput, rows: usize) -> anyhow::Result<Vec<(u32, f32)>> {
    let mut top1 = Vec::with_capacity(rows);
    for input_row in 0..rows {
        let row = output
            .logits
            .iter()
            .find(|row| row.input_row as usize == input_row)
            .ok_or_else(|| {
                anyhow::anyhow!("packed verification output is missing row {input_row}")
            })?;
        let LogitsOutput::TopK(logits) = &row.logits else {
            anyhow::bail!("packed verification output row {input_row} is not top-k");
        };
        let token = logits.first().ok_or_else(|| {
            anyhow::anyhow!("packed verification output row {input_row} has no top-1 token")
        })?;
        top1.push((token.token_id, token.logit));
    }
    Ok(top1)
}

#[cfg(feature = "cuda")]
fn same_top1_bits(left: &[(u32, f32)], right: &[(u32, f32)]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.0 == right.0 && left.1.to_bits() == right.1.to_bits())
}

#[cfg(feature = "cuda")]
fn measure_resident_verify_rows(
    diagnostic: &mut PageManagedDiagnosticHarness<DeepSeekV4Runner>,
    slot: StateSlot,
    token_ids: &[u32],
) -> anyhow::Result<ResidentVerifySample> {
    let top_one = NonZeroU32::new(1).expect("one is non-zero");
    let logits = vec![LogitsRequest::TopK(top_one); token_ids.len()];
    let counters_before = diagnostic.runner().operator_runtime_counters();
    let layer_before = diagnostic.runner().layer_profile_stats();
    let attention_before = diagnostic.runner().attention_profile_stats();
    let output_before = diagnostic.runner().output_profile_stats();
    let started = Instant::now();
    let output =
        diagnostic.execute_sequence_batch(slot, ForwardPhase::Prefill, token_ids, &logits)?;
    Ok(ResidentVerifySample {
        elapsed_us: duration_us(started.elapsed()),
        counters: dsv4_operator_counters_delta(
            counters_before,
            diagnostic.runner().operator_runtime_counters(),
        ),
        layer_profile: dsv4_layer_profile_stats_delta(
            &layer_before,
            &diagnostic.runner().layer_profile_stats(),
        ),
        attention_profile: dsv4_attention_profile_stats_delta(
            &attention_before,
            &diagnostic.runner().attention_profile_stats(),
        ),
        output_profile: dsv4_output_profile_stats_delta(
            output_before,
            diagnostic.runner().output_profile_stats(),
        ),
        top1: packed_output_top1(&output, token_ids.len())?,
    })
}

#[cfg(feature = "cuda")]
fn verify_percentile_us(measurement: &ResidentVerifyWidthMeasurement, percentile: usize) -> u64 {
    let mut values = measurement
        .samples
        .iter()
        .map(|sample| sample.elapsed_us)
        .collect::<Vec<_>>();
    values.sort_unstable();
    let rank = values
        .len()
        .saturating_sub(1)
        .saturating_mul(percentile.min(100))
        .div_ceil(100);
    values[rank.min(values.len().saturating_sub(1))]
}

#[cfg(feature = "cuda")]
fn resident_verify_sample_json(sample: &ResidentVerifySample) -> serde_json::Value {
    serde_json::json!({
        "elapsed_s": sample.elapsed_us as f64 / 1_000_000.0,
        "top1": sample.top1.iter().map(|(token_id, logit)| serde_json::json!({
            "token_id": token_id,
            "logit": logit,
            "logit_bits": logit.to_bits(),
        })).collect::<Vec<_>>(),
        "dsv4_operator_counters": dsv4_operator_counters_json(&sample.counters),
        "dsv4_layer_profile_summary": dsv4_layer_profile_summary_json(&sample.layer_profile),
        "dsv4_attention_profile_summary": dsv4_attention_profile_summary_json(&sample.attention_profile),
        "dsv4_output_profile": dsv4_output_profile_stats_json(&sample.output_profile),
    })
}

#[cfg(feature = "cuda")]
fn resident_verify_width_json(
    measurement: &ResidentVerifyWidthMeasurement,
    checkpoint_reference_verify_rows: usize,
) -> serde_json::Value {
    let p50_us = verify_percentile_us(measurement, 50);
    let p95_us = verify_percentile_us(measurement, 95);
    serde_json::json!({
        "v": measurement.width,
        "t_verify_p50_s": p50_us as f64 / 1_000_000.0,
        "t_verify_p95_s": p95_us as f64 / 1_000_000.0,
        "rows_per_s_p50": if p50_us == 0 { 0.0 } else {
            measurement.width as f64 * 1_000_000.0 / p50_us as f64
        },
        "target_cycles_per_s_p50": if p50_us == 0 { 0.0 } else {
            1_000_000.0 / p50_us as f64
        },
        "checkpoint_reference_width": measurement.width == checkpoint_reference_verify_rows,
        "experimental_above_checkpoint_width": measurement.width > checkpoint_reference_verify_rows,
        "resident_no_io": measurement.resident_no_io,
        "allocation_free": measurement.allocation_free,
        "parity": measurement.parity,
        "capture_top1": measurement.capture_top1.iter().map(|(token_id, logit)| serde_json::json!({
            "token_id": token_id,
            "logit": logit,
            "logit_bits": logit.to_bits(),
        })).collect::<Vec<_>>(),
        "samples": measurement.samples.iter().map(resident_verify_sample_json).collect::<Vec<_>>(),
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_resident_verify_width_sweep(
    runner: DeepSeekV4Runner,
    model_dir: &str,
    chat_template: &ChatTemplate,
    prompts: &[String],
    artifact_load_us: u64,
    max_layers: usize,
    output_head_chunk_rows: usize,
    moe_prefetch_experts: usize,
    moe_hotset_experts: usize,
    checkpoint_dspark_block_size: usize,
    iterations: usize,
    json: bool,
) -> anyhow::Result<()> {
    let checkpoint_reference_verify_rows = checkpoint_dspark_block_size
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("checkpoint DSpark verification width overflow"))?;
    let mut widths = vec![2, 4, checkpoint_reference_verify_rows, 8];
    widths.sort_unstable();
    widths.dedup();
    if prompts.len() != 1 {
        anyhow::bail!(
            "--verify-width-sweep requires exactly one --prompt, got {}",
            prompts.len()
        );
    }
    if iterations == 0 {
        anyhow::bail!("--verify-iterations must be greater than zero");
    }
    if moe_prefetch_experts != 0 {
        anyhow::bail!(
            "--verify-width-sweep requires --moe-prefetch-experts 0 so measured I/O is exact"
        );
    }
    if moe_hotset_experts < 48 {
        anyhow::bail!(
            "--verify-width-sweep requires --moe-hotset-experts >= 48 for the maximum V=8 route set"
        );
    }

    let full_prompt = chat_template.format_turn(&prompts[0], true);
    let token_ids = runner.encode(&full_prompt)?;
    let max_width = *widths.last().expect("verification widths are non-empty");
    if token_ids.len() <= max_width {
        anyhow::bail!(
            "--verify-width-sweep prompt encoded to {} tokens; need at least {}",
            token_ids.len(),
            max_width + 1
        );
    }
    let prefix_len = token_ids.len() - max_width;
    let prefix = &token_ids[..prefix_len];
    let candidates = &token_ids[prefix_len..];
    let schema = runner.kv_layout_schema().clone();
    let mut diagnostic =
        build_page_managed_diagnostic_harness(runner, Box::new(schema), token_ids.len(), 2)?;
    let capture_slot = diagnostic.create_sequence(0)?;
    let replay_slot = diagnostic.create_sequence(0)?;

    let mut measurements = Vec::with_capacity(widths.len());
    for (width_index, width) in widths.into_iter().enumerate() {
        if width_index > 0 {
            diagnostic.reset_sequence(capture_slot)?;
            diagnostic.reset_sequence(replay_slot)?;
        }
        for slot in [capture_slot, replay_slot] {
            diagnostic
                .execute_sequence_step(slot, ForwardPhase::Prefill, prefix, |runner| {
                    runner.prefill_tokens_no_logits_batched(prefix)
                })
                .map_err(|error| {
                    anyhow::anyhow!(
                        "Gate F1 V={width} initial prefill failed for slot {slot:?}: {error}"
                    )
                })?;
        }

        let candidate_rows = &candidates[..width];
        let mut capture_top1: Option<Vec<(u32, f32)>> = None;
        let mut samples = Vec::with_capacity(iterations);
        let mut parity = true;
        let mut resident_no_io = true;
        let mut allocation_free = true;
        for iteration in 0..iterations {
            if iteration > 0 {
                for slot in [capture_slot, replay_slot] {
                    diagnostic.reset_sequence(slot)?;
                    diagnostic
                        .execute_sequence_step(
                            slot,
                            ForwardPhase::Prefill,
                            prefix,
                            |runner| runner.prefill_tokens_no_logits_batched(prefix),
                        )
                        .map_err(|error| {
                            anyhow::anyhow!(
                                "Gate F1 V={width} iteration={iteration} reset prefill failed for slot {slot:?}: {error}"
                            )
                        })?;
                }
            }
            // Capture is deliberately outside T_verify. It installs exactly the
            // route union used by the immediately following replay sample.
            let capture =
                measure_resident_verify_rows(&mut diagnostic, capture_slot, candidate_rows)
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "Gate F1 V={width} iteration={iteration} capture failed: {error}"
                        )
                    })?;
            if let Some(baseline) = &capture_top1 {
                parity &= same_top1_bits(baseline, &capture.top1);
            } else {
                capture_top1 = Some(capture.top1.clone());
            }
            let sample = measure_resident_verify_rows(&mut diagnostic, replay_slot, candidate_rows)
                .map_err(|error| {
                    anyhow::anyhow!(
                        "Gate F1 V={width} iteration={iteration} replay failed: {error}"
                    )
                })?;
            parity &= same_top1_bits(
                capture_top1
                    .as_deref()
                    .expect("capture baseline is initialized"),
                &sample.top1,
            );
            resident_no_io &= sample.counters.expert_load_bytes == 0
                && sample.counters.expert_selected_cold_misses == 0;
            allocation_free &= sample.counters.device_allocations == 0;
            samples.push(sample);
        }
        measurements.push(ResidentVerifyWidthMeasurement {
            width,
            capture_top1: capture_top1.expect("at least one verification iteration"),
            samples,
            parity,
            resident_no_io,
            allocation_free,
        });
    }

    let v4 = measurements
        .iter()
        .find(|measurement| measurement.width == 4)
        .ok_or_else(|| anyhow::anyhow!("V=4 verification measurement is missing"))?;
    let v4_p50_us = verify_percentile_us(v4, 50);
    let checkpoint_reference = measurements
        .iter()
        .find(|measurement| measurement.width == checkpoint_reference_verify_rows)
        .ok_or_else(|| {
            anyhow::anyhow!("checkpoint-reference verification measurement is missing")
        })?;
    let checkpoint_reference_p50_us = verify_percentile_us(checkpoint_reference, 50);
    let checkpoint_reference_rows_per_s_p50 = if checkpoint_reference_p50_us == 0 {
        0.0
    } else {
        checkpoint_reference_verify_rows as f64 * 1_000_000.0 / checkpoint_reference_p50_us as f64
    };
    let checkpoint_native_max_external_commit_tokens = checkpoint_reference_verify_rows;
    let target_only_non_target_budget_s_at_16_p50 =
        (checkpoint_native_max_external_commit_tokens as f64 / 16.0
            - checkpoint_reference_p50_us as f64 / 1_000_000.0)
            .max(0.0);
    let experimental_above_checkpoint_reaches_16 = measurements.iter().any(|measurement| {
        if measurement.width <= checkpoint_reference_verify_rows {
            return false;
        }
        let p50_us = verify_percentile_us(measurement, 50);
        p50_us != 0 && measurement.width as f64 * 1_000_000.0 / p50_us as f64 >= 16.0
    });
    let parity = measurements.iter().all(|measurement| measurement.parity);
    let resident_no_io = measurements
        .iter()
        .all(|measurement| measurement.resident_no_io);
    let allocation_free = measurements
        .iter()
        .all(|measurement| measurement.allocation_free);

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "mode": "resident_verify_width_sweep",
                "model": model_dir,
                "prompt": prompts[0],
                "chat_template": chat_template.name(),
                "prompt_token_ids": token_ids,
                "prefix_tokens": prefix_len,
                "verification_token_ids": candidates,
                "checkpoint_dspark_block_size": checkpoint_dspark_block_size,
                "checkpoint_reference_verify_rows": checkpoint_reference_verify_rows,
                "iterations": iterations,
                "max_layers": max_layers,
                "output_head_chunk_rows": output_head_chunk_rows,
                "moe_prefetch_experts": moe_prefetch_experts,
                "moe_hotset_experts": moe_hotset_experts,
                "artifact_load_s": artifact_load_us as f64 / 1_000_000.0,
                "widths": measurements.iter().map(|measurement| {
                    resident_verify_width_json(measurement, checkpoint_reference_verify_rows)
                }).collect::<Vec<_>>(),
                "gate_f1": {
                    "roofline_probe_only": true,
                    "checkpoint_native_width_contract_applied": true,
                    "checkpoint_native_max_external_commit_tokens": checkpoint_native_max_external_commit_tokens,
                    "target_only_non_target_budget_s_at_16_p50": target_only_non_target_budget_s_at_16_p50,
                    "complete_cycle_viability_measured": false,
                    "parity": parity,
                    "resident_no_io": resident_no_io,
                    "allocation_free": allocation_free,
                    "v4_over_250_ms": v4_p50_us > 250_000,
                    "v4_over_200_ms": v4_p50_us > 200_000,
                    "checkpoint_reference_rows_per_s_p50": checkpoint_reference_rows_per_s_p50,
                    "checkpoint_reference_target_only_reaches_16_rows_s": checkpoint_reference_rows_per_s_p50 >= 16.0,
                    "experimental_above_checkpoint_reaches_16_rows_s": experimental_above_checkpoint_reaches_16,
                },
            }))?
        );
    } else {
        println!("=== DSV4 Gate F1 Resident Verification Roofline Sweep ===");
        println!("prefix tokens:      {prefix_len}");
        println!("checkpoint gamma:   {checkpoint_dspark_block_size}");
        println!("reference rows:     {checkpoint_reference_verify_rows}");
        println!("iterations:         {iterations}");
        println!("resident no-I/O:    {resident_no_io}");
        println!("allocation-free:    {allocation_free}");
        println!("parity:             {parity}");
        for measurement in &measurements {
            let p50_us = verify_percentile_us(measurement, 50);
            let p95_us = verify_percentile_us(measurement, 95);
            let rows_per_s = if p50_us == 0 {
                0.0
            } else {
                measurement.width as f64 * 1_000_000.0 / p50_us as f64
            };
            println!(
                "V={:<2} p50={:>9.3} ms p95={:>9.3} ms rows/s={:>8.3} no_io={} alloc_free={} parity={}",
                measurement.width,
                p50_us as f64 / 1_000.0,
                p95_us as f64 / 1_000.0,
                rows_per_s,
                measurement.resident_no_io,
                measurement.allocation_free,
                measurement.parity,
            );
        }
        println!("V=4 > 250 ms:       {}", v4_p50_us > 250_000);
        println!("V=4 > 200 ms:       {}", v4_p50_us > 200_000);
        println!("reference rows/s:     {checkpoint_reference_rows_per_s_p50:.3} (target-only)");
        println!(
            "non-target budget:     {:.3} ms at 16 tok/s and 100% acceptance",
            target_only_non_target_budget_s_at_16_p50 * 1_000.0
        );
        println!(
            "experimental >=16:    {experimental_above_checkpoint_reaches_16} (not a release gate)"
        );
        println!("complete cycle:      not measured");
    }

    if !parity {
        anyhow::bail!("Gate F1 packed verification parity failed");
    }
    if !resident_no_io {
        anyhow::bail!("Gate F1 measured verification performed selected-expert I/O");
    }
    Ok(())
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
    _output_head_chunk_rows: usize,
    _moe_prefetch_experts: usize,
    _moe_hotset_experts: usize,
    _golden_trace_path: Option<&str>,
    _json: bool,
    _resident_replay: bool,
    _verify_width_sweep: bool,
    _verify_iterations: usize,
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
