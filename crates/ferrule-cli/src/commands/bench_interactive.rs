//! Interactive multi-turn benchmark for the DSV4 chat path.
//!
//! Feeds a fixed set of user turns through the interactive prefill/decode
//! pipeline (the same code path as `ferrule chat`) and reports:
//!
//! - time-to-REPL / artifact load latency
//! - per-turn prefill and decode wall time
//! - generated tokens per turn and aggregate decode tok/s
//! - runtime-owned physical materialization and critical-path counters
//! - optional golden-trace comparison

#[cfg(feature = "cuda")]
use std::collections::{BTreeMap, BTreeSet};

#[cfg(feature = "cuda")]
use std::path::Path;
#[cfg(feature = "cuda")]
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use crate::GenerationConfig;
#[cfg(feature = "cuda")]
use crate::bench::{GoldenTurn, InteractiveTrace, compare_interactive_trace};
#[cfg(feature = "cuda")]
use ferrule_common::io_protocol::{LoadStage, OperationId, RetirementReason};
#[cfg(feature = "cuda")]
use ferrule_model::{
    ChatTemplate, ModelExecutionBackend,
    models::deepseek_v4::{
        DeepSeekV4AttentionProfileStats, DeepSeekV4Checkpoint, DeepSeekV4LayerProfileStats,
        DeepSeekV4ObservabilitySnapshot, DeepSeekV4OperatorRuntimeCounters,
        DeepSeekV4OutputProfileStats, DeepSeekV4PrepareOptions, DeepSeekV4Runner,
    },
    moe::ExpertPredictionStats,
};
#[cfg(feature = "cuda")]
use ferrule_runtime::io::{OutputTokenId, RuntimeMaterializationResolverStats};
#[cfg(feature = "cuda")]
use ferrule_runtime::{ExecutionPhase, ResourceDemand, ResourceKind};
#[cfg(feature = "cuda")]
use ferrule_runtime::{
    FixedSequenceSlotPool, GenerateRequest, LocalResidentInferenceEngine, RequestId,
    ResidentActionKind, ResidentDriverStep, ResidentTopKDriverStats, SessionId, SpeculativeMetrics,
};

#[cfg(feature = "cuda")]
use super::resident::{
    block_on_local_inference, build_resident_topk_driver, require_finished_request,
    resident_driver_config, single_sequence_scheduler_config,
};

#[cfg(feature = "cuda")]
pub(crate) const CLI_RUNTIME_SCHEMA_VERSION: u32 = 4;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RuntimeDependencyStats {
    selected: u64,
    unique_selected: u64,
    resident: usize,
    waiting: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RuntimeMaterializationStageStats {
    read_stages: u64,
    read_bytes: u64,
    upload_stages: u64,
    upload_bytes: u64,
    install_stages: u64,
    install_bytes: u64,
    publish_stages: u64,
    publish_bytes: u64,
    failures: u64,
    stale: u64,
    cancellations: u64,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RuntimeLoadRegistryStats {
    operations_created: u64,
    active_operations: usize,
    retired_operations: u64,
    single_flight_joins: u64,
    physical_completions: u64,
    rejected_completions: u64,
    publications: u64,
    cancellations_requested: u64,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RuntimeExternalTokenSnapshot {
    external_token_id: u64,
    externally_committed_tokens: usize,
    captured_at_ns: u64,
    read_ns: u64,
    upload_ns: u64,
    publish_ns: u64,
    wait_ns: u64,
    covered_wait_ns: u64,
    uncovered_wait_ns: u64,
    resume_ns: u64,
    commit_ns: u64,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RuntimeCriticalPathStats {
    external_tokens: usize,
    snapshot_groups: usize,
    read_ns: u64,
    upload_ns: u64,
    publish_ns: u64,
    wait_ns: u64,
    covered_wait_ns: u64,
    uncovered_wait_ns: u64,
    resume_ns: u64,
    commit_ns: u64,
    per_external_token: Vec<RuntimeExternalTokenSnapshot>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ExternalSnapshotReconciliation {
    expected_external_tokens: usize,
    snapshot_external_tokens: usize,
    reconciled_external_tokens: usize,
    snapshot_groups: usize,
    consistent: bool,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RuntimeMaterializationStats {
    dependencies: RuntimeDependencyStats,
    resolver: RuntimeMaterializationResolverStats,
    stages: RuntimeMaterializationStageStats,
    registry: RuntimeLoadRegistryStats,
    critical_path: RuntimeCriticalPathStats,
    external_snapshot_reconciliation: ExternalSnapshotReconciliation,
}

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
    runtime_materialization: RuntimeMaterializationStats,
    dsv4_operator_counters: DeepSeekV4OperatorRuntimeCounters,
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
    /// Runtime-owned physical materialization and critical-path counters.
    runtime_materialization: RuntimeMaterializationStats,
    /// DSV4 CUDA/operator counters for measured turns only.
    dsv4_operator_counters: DeepSeekV4OperatorRuntimeCounters,
    /// DSV4 per-layer profile counters for measured turns only.
    dsv4_layer_profile_stats: Vec<DeepSeekV4LayerProfileStats>,
    /// DSV4 per-layer attention-internal profile counters for measured turns only.
    dsv4_attention_profile_stats: Vec<DeepSeekV4AttentionProfileStats>,
    /// DSV4 final hidden/output-head profile counters for measured turns only.
    dsv4_output_profile_stats: DeepSeekV4OutputProfileStats,
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
    moe_hotset_experts: usize,
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
        ctx_size: 4096,
        ..GenerationConfig::default()
    };

    let options = DeepSeekV4PrepareOptions {
        max_layers,
        output_head_chunk_rows,
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
    let model = DeepSeekV4Checkpoint::load_hf_with_limit(model_path, max_tensor_bytes)?;
    let runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;
    let artifact_load_us = duration_us(load_start.elapsed());

    let mut report = InteractiveBenchReport {
        model_dir: model_dir.to_string(),
        chat_template: chat_template.name().to_string(),
        max_new_tokens,
        max_layers,
        prefill_chunk_size,
        output_head_chunk_rows,
        runtime_path: "resident_topk_driver_speculative".into(),
        dsv4_profile_sync: runner.execution_policy().profile_sync(),
        artifact_load_us,
        warmup_tokens,
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
        let mut out = interactive_bench_report_json(&report);

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
                print_runtime_materialization_summary(&turn.runtime_materialization);
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
        print_runtime_materialization_summary(&report.runtime_materialization);
        print_hard_resource_high_water(&report.runtime_driver_stats.hard_resource_high_water);
    }

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
    block_on_local_inference(run_with_resident_driver_async(
        runner,
        chat_template,
        gen_cfg,
        prompts,
        warmup_tokens,
        json,
        report,
    ))
}

#[cfg(feature = "cuda")]
async fn run_with_resident_driver_async(
    runner: DeepSeekV4Runner,
    chat_template: &ChatTemplate,
    gen_cfg: &GenerationConfig,
    prompts: &[String],
    warmup_tokens: usize,
    json: bool,
    report: &mut InteractiveBenchReport,
) -> anyhow::Result<()> {
    let scheduler_config = single_sequence_scheduler_config(report.prefill_chunk_size);
    let driver_config = resident_driver_config(gen_cfg.ctx_size, gen_cfg.stop_at_eos);
    let build_driver = |runner: DeepSeekV4Runner| {
        let schema = runner.kv_layout_schema().clone();
        build_resident_topk_driver(runner, Box::new(schema), scheduler_config, driver_config)
    };
    let mut driver = LocalResidentInferenceEngine::new(build_driver(runner)?);
    driver.initialize().await?;
    driver.wait_for_model_warmup().await?;

    if warmup_tokens > 0 {
        let warmup_prompt = chat_template.format_turn("warmup", true);
        let warmup_prompt_tokens = driver.encode(&warmup_prompt)?;
        let warmup_request = driver_request(
            0,
            SessionId(u64::MAX),
            warmup_prompt_tokens,
            warmup_tokens,
            &gen_cfg.stop,
        );
        let warmup_start = Instant::now();
        driver.submit(warmup_request);
        loop {
            let step = driver.step(&mut |_| Ok(())).await?;
            if let Some(terminal) = driver.take_request_terminal(RequestId(0)) {
                let _ = require_finished_request(terminal, "interactive benchmark warmup")?;
                break;
            }
            match step {
                ResidentDriverStep::Idle => anyhow::bail!(
                    "interactive benchmark warmup became idle before request terminalization"
                ),
                ResidentDriverStep::Blocked => {
                    anyhow::bail!("speculative warmup blocked while running resident benchmark")
                }
                ResidentDriverStep::WaitingForModelProgress(_) => {}
                ResidentDriverStep::Executed { .. } => {}
            }
        }
        report.warmup_us = duration_us(warmup_start.elapsed());
        report.warmup_generated_tokens = driver.stats().emitted_tokens;
    }

    let observability_baseline: DeepSeekV4ObservabilitySnapshot =
        driver.model_observability_snapshot();
    let counters_baseline = observability_baseline.operator;
    let driver_stats_baseline = resident_driver_stats_snapshot(&driver);
    let runtime_materialization_baseline =
        runtime_materialization_snapshot(&driver, counters_baseline)?;
    let layer_profile_baseline = observability_baseline.layers;
    let attention_profile_baseline = observability_baseline.attention;
    let output_profile_baseline = observability_baseline.output;
    let mut first_token_measured = false;
    let mut total_prefill_us: u64 = 0;
    let mut total_decode_us: u64 = 0;
    let mut total_prompt_tokens = 0usize;
    let mut total_generated = 0usize;

    for (turn_idx, prompt_text) in prompts.iter().enumerate() {
        let first_turn = turn_idx == 0;
        let full_prompt = chat_template.format_turn(prompt_text, first_turn);
        let prompt_tokens = driver.encode(&full_prompt)?;

        if prompt_tokens.is_empty() {
            if !json {
                eprintln!(
                    "[bench] turn {} prompt encoded to zero tokens, skipping",
                    turn_idx
                );
            }
            continue;
        }

        let request_id = RequestId(turn_idx as u64 + 1);
        let request = driver_request(
            request_id.0,
            SessionId(0),
            prompt_tokens.clone(),
            gen_cfg.max_new_tokens,
            &gen_cfg.stop,
        );
        let turn_driver_stats_before = resident_driver_stats_snapshot(&driver);
        let turn_observability_before = driver.model_observability_snapshot();
        let turn_operator_counters_before = turn_observability_before.operator;
        let turn_runtime_materialization_before =
            runtime_materialization_snapshot(&driver, turn_operator_counters_before)?;
        let turn_layer_profile_before = turn_observability_before.layers;
        let turn_attention_profile_before = turn_observability_before.attention;
        let turn_output_profile_before = turn_observability_before.output;
        driver.submit(request);

        let turn_start = Instant::now();
        let mut first_token_us = None;
        let mut prefill_us = 0u64;
        let mut decode_us = 0u64;
        let mut generated_tokens = Vec::new();
        let mut runtime_steps = Vec::new();

        let sequence = loop {
            let step_observability_before = driver.model_observability_snapshot();
            let step_operator_counters_before = step_observability_before.operator;
            let step_layer_profile_before = step_observability_before.layers;
            let step_attention_profile_before = step_observability_before.attention;
            let step_output_profile_before = step_observability_before.output;
            let step_start = Instant::now();
            let step = driver
                .step(&mut |event| {
                    first_token_us.get_or_insert_with(|| duration_us(turn_start.elapsed()));
                    generated_tokens.push(event.token);
                    Ok(())
                })
                .await?;
            let step_us = duration_us(step_start.elapsed());
            let step_is_idle = matches!(&step, ResidentDriverStep::Idle);
            let step_observability_after = driver.model_observability_snapshot();
            let step_operator_counters = dsv4_operator_counters_delta(
                step_operator_counters_before,
                step_observability_after.operator,
            );
            let step_layer_profile_stats = dsv4_layer_profile_stats_delta(
                &step_layer_profile_before,
                &step_observability_after.layers,
            );
            let step_attention_profile_stats = dsv4_attention_profile_stats_delta(
                &step_attention_profile_before,
                &step_observability_after.attention,
            );
            let step_output_profile_stats = dsv4_output_profile_stats_delta(
                step_output_profile_before,
                step_observability_after.output,
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
                        runner_position: step_observability_after.position,
                        dsv4_operator_counters: step_operator_counters,
                        dsv4_layer_profile_stats: step_layer_profile_stats,
                        dsv4_attention_profile_stats: step_attention_profile_stats,
                        dsv4_output_profile_stats: step_output_profile_stats,
                    });
                }
                ResidentDriverStep::WaitingForModelProgress(_) => {
                    decode_us = decode_us.saturating_add(step_us);
                    runtime_steps.push(RuntimeStepMeasurement {
                        action_kind: "expert_wait".to_string(),
                        rows: 0,
                        staged: 0,
                        finished: 0,
                        elapsed_us: step_us,
                        runner_position: step_observability_after.position,
                        dsv4_operator_counters: step_operator_counters,
                        dsv4_layer_profile_stats: step_layer_profile_stats,
                        dsv4_attention_profile_stats: step_attention_profile_stats,
                        dsv4_output_profile_stats: step_output_profile_stats,
                    });
                }
                ResidentDriverStep::Idle => {}
                ResidentDriverStep::Blocked => {
                    anyhow::bail!("resident runtime driver blocked while running measured turn")
                }
            }
            if let Some(terminal) = driver.take_request_terminal(request_id) {
                break require_finished_request(terminal, "interactive benchmark turn")?;
            }
            if step_is_idle {
                anyhow::bail!("interactive benchmark became idle before request terminalization");
            }
        };

        let first_token_us = first_token_us.unwrap_or_else(|| duration_us(turn_start.elapsed()));
        if !first_token_measured {
            report.time_to_first_token_us = first_token_us;
            first_token_measured = true;
        }

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

        let turn_driver_stats = resident_driver_stats_delta(
            turn_driver_stats_before,
            resident_driver_stats_snapshot(&driver),
        );
        let turn_observability_after = driver.model_observability_snapshot();
        let turn_operator_counters = dsv4_operator_counters_delta(
            turn_operator_counters_before,
            turn_observability_after.operator,
        );
        let turn_runtime_materialization_after =
            runtime_materialization_snapshot(&driver, turn_observability_after.operator)?;
        let turn_runtime_materialization = runtime_materialization_stats_delta(
            turn_runtime_materialization_before,
            turn_runtime_materialization_after,
            turn_driver_stats.emitted_tokens,
        );

        let turn_layer_profile_stats = dsv4_layer_profile_stats_delta(
            &turn_layer_profile_before,
            &turn_observability_after.layers,
        );
        let turn_attention_profile_stats = dsv4_attention_profile_stats_delta(
            &turn_attention_profile_before,
            &turn_observability_after.attention,
        );
        let turn_output_profile_stats = dsv4_output_profile_stats_delta(
            turn_output_profile_before,
            turn_observability_after.output,
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
            runtime_materialization: turn_runtime_materialization,
            dsv4_operator_counters: turn_operator_counters,
            dsv4_layer_profile_stats: turn_layer_profile_stats,
            dsv4_attention_profile_stats: turn_attention_profile_stats,
            dsv4_output_profile_stats: turn_output_profile_stats,
            runtime_steps,
        });
    }

    let observability_now = driver.model_observability_snapshot();
    let counters_now = observability_now.operator;
    report.runtime_driver_stats = resident_driver_stats_delta(
        driver_stats_baseline,
        resident_driver_stats_snapshot(&driver),
    );
    report.dsv4_operator_counters = dsv4_operator_counters_delta(counters_baseline, counters_now);
    let runtime_materialization_now = runtime_materialization_snapshot(&driver, counters_now)?;
    report.runtime_materialization = runtime_materialization_stats_delta(
        runtime_materialization_baseline,
        runtime_materialization_now,
        report.runtime_driver_stats.emitted_tokens,
    );
    report.dsv4_layer_profile_stats =
        dsv4_layer_profile_stats_delta(&layer_profile_baseline, &observability_now.layers);
    report.dsv4_attention_profile_stats = dsv4_attention_profile_stats_delta(
        &attention_profile_baseline,
        &observability_now.attention,
    );
    report.dsv4_output_profile_stats =
        dsv4_output_profile_stats_delta(output_profile_baseline, observability_now.output);
    finish_report_counters(
        report,
        total_prefill_us,
        total_decode_us,
        total_prompt_tokens,
        total_generated,
    );
    driver.shutdown().await?;
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn runtime_materialization_snapshot(
    driver: &LocalResidentInferenceEngine<DeepSeekV4Runner, FixedSequenceSlotPool>,
    operator: DeepSeekV4OperatorRuntimeCounters,
) -> anyhow::Result<RuntimeMaterializationStats> {
    let registry = driver.driver().load_registry();
    let registry_stats = registry.stats();
    let waiting = registry
        .waiters()
        .active_waiters()
        .try_fold(0usize, |total, waiter| {
            let dependencies = registry
                .waiters()
                .loads_for(waiter)
                .ok_or_else(|| anyhow::anyhow!("active runtime waiter has no dependency set"))?;
            total
                .checked_add(dependencies.len())
                .ok_or_else(|| anyhow::anyhow!("runtime waiting dependency count overflow"))
        })?;

    let mut stages = RuntimeMaterializationStageStats::default();
    for raw_operation in 1..=registry_stats.operations_created {
        let operation = OperationId::new(raw_operation);
        let history = registry.stage_history(operation).ok_or_else(|| {
            anyhow::anyhow!(
                "load registry omitted stage history for created physical operation {}",
                raw_operation
            )
        })?;
        let plan = if let Some(active) = registry.operation(operation) {
            active.plan()
        } else {
            let key = registry.key_for_operation(operation).ok_or_else(|| {
                anyhow::anyhow!(
                    "load registry omitted key for physical operation {}",
                    raw_operation
                )
            })?;
            registry
                .prepare_execution_request(
                    key,
                    ResourceDemand::required(ExecutionPhase::Prefill),
                    ferrule_model::ResourceRetention::ThroughStage,
                )?
                .plan
        };

        for stage in history {
            match stage {
                LoadStage::ReadSubmitted => {
                    stages.read_stages = stages.read_stages.saturating_add(1);
                    stages.read_bytes = stages
                        .read_bytes
                        .saturating_add(plan.requirements.storage_read_bytes);
                }
                LoadStage::UploadSubmitted => {
                    stages.upload_stages = stages.upload_stages.saturating_add(1);
                    stages.upload_bytes = stages
                        .upload_bytes
                        .saturating_add(plan.requirements.h2d_bytes);
                }
                LoadStage::Installing => {
                    stages.install_stages = stages.install_stages.saturating_add(1);
                    stages.install_bytes = stages
                        .install_bytes
                        .saturating_add(plan.requirements.device_install_bytes);
                }
                LoadStage::Resident => {
                    stages.publish_stages = stages.publish_stages.saturating_add(1);
                    stages.publish_bytes = stages.publish_bytes.saturating_add(plan.resident_bytes);
                }
                LoadStage::Failed => stages.failures = stages.failures.saturating_add(1),
                LoadStage::Stale => stages.stale = stages.stale.saturating_add(1),
                LoadStage::Reserved
                | LoadStage::HostReady
                | LoadStage::Draining
                | LoadStage::Retired => {}
            }
        }
        if registry
            .retirement(operation)
            .is_some_and(|record| matches!(record.reason, RetirementReason::Cancelled(_)))
        {
            stages.cancellations = stages.cancellations.saturating_add(1);
        }
    }

    let mut external_token_snapshots = Vec::new();
    let mut external_token_id = 1u64;
    while let Some(snapshot) = registry
        .ledger()
        .output(OutputTokenId::new(external_token_id))
    {
        external_token_snapshots.push(RuntimeExternalTokenSnapshot {
            external_token_id,
            externally_committed_tokens: snapshot.externally_committed_tokens,
            captured_at_ns: snapshot.captured_at_ns,
            read_ns: snapshot.read_ns,
            upload_ns: snapshot.upload_ns,
            publish_ns: snapshot.publish_ns,
            wait_ns: snapshot.wait_ns,
            covered_wait_ns: snapshot.covered_wait_ns,
            uncovered_wait_ns: snapshot.uncovered_wait_ns,
            resume_ns: snapshot.resume_ns,
            commit_ns: snapshot.commit_ns,
        });
        external_token_id = external_token_id
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("external output-token identity space exhausted"))?;
    }
    let (critical_path, external_snapshot_reconciliation) =
        critical_path_stats(external_token_snapshots, driver.stats().emitted_tokens);

    Ok(RuntimeMaterializationStats {
        dependencies: RuntimeDependencyStats {
            selected: operator.expert_selected,
            unique_selected: operator.expert_unique_selected,
            resident: operator.expert_residency_stats.resident,
            waiting,
        },
        resolver: driver.driver().materialization_resolver_stats(),
        stages,
        registry: RuntimeLoadRegistryStats {
            operations_created: registry_stats.operations_created,
            active_operations: registry.active_operations(),
            retired_operations: registry_stats.retirements,
            single_flight_joins: registry_stats.single_flight_joins,
            physical_completions: registry_stats.physical_completions,
            rejected_completions: registry_stats.rejected_completions,
            publications: registry_stats.publications,
            cancellations_requested: registry_stats.cancellations_requested,
        },
        critical_path,
        external_snapshot_reconciliation,
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn runtime_materialization_stats_delta(
    before: RuntimeMaterializationStats,
    after: RuntimeMaterializationStats,
    expected_external_tokens: usize,
) -> RuntimeMaterializationStats {
    let baseline_tokens: BTreeSet<_> = before
        .critical_path
        .per_external_token
        .iter()
        .map(|snapshot| snapshot.external_token_id)
        .collect();
    let per_external_token = after
        .critical_path
        .per_external_token
        .into_iter()
        .filter(|snapshot| !baseline_tokens.contains(&snapshot.external_token_id))
        .collect();
    let (critical_path, external_snapshot_reconciliation) =
        critical_path_stats(per_external_token, expected_external_tokens);

    RuntimeMaterializationStats {
        dependencies: RuntimeDependencyStats {
            selected: after
                .dependencies
                .selected
                .saturating_sub(before.dependencies.selected),
            unique_selected: after
                .dependencies
                .unique_selected
                .saturating_sub(before.dependencies.unique_selected),
            resident: after.dependencies.resident,
            waiting: after.dependencies.waiting,
        },
        resolver: RuntimeMaterializationResolverStats {
            resolves: after
                .resolver
                .resolves
                .saturating_sub(before.resolver.resolves),
        },
        stages: RuntimeMaterializationStageStats {
            read_stages: after
                .stages
                .read_stages
                .saturating_sub(before.stages.read_stages),
            read_bytes: after
                .stages
                .read_bytes
                .saturating_sub(before.stages.read_bytes),
            upload_stages: after
                .stages
                .upload_stages
                .saturating_sub(before.stages.upload_stages),
            upload_bytes: after
                .stages
                .upload_bytes
                .saturating_sub(before.stages.upload_bytes),
            install_stages: after
                .stages
                .install_stages
                .saturating_sub(before.stages.install_stages),
            install_bytes: after
                .stages
                .install_bytes
                .saturating_sub(before.stages.install_bytes),
            publish_stages: after
                .stages
                .publish_stages
                .saturating_sub(before.stages.publish_stages),
            publish_bytes: after
                .stages
                .publish_bytes
                .saturating_sub(before.stages.publish_bytes),
            failures: after.stages.failures.saturating_sub(before.stages.failures),
            stale: after.stages.stale.saturating_sub(before.stages.stale),
            cancellations: after
                .stages
                .cancellations
                .saturating_sub(before.stages.cancellations),
        },
        registry: RuntimeLoadRegistryStats {
            operations_created: after
                .registry
                .operations_created
                .saturating_sub(before.registry.operations_created),
            active_operations: after.registry.active_operations,
            retired_operations: after
                .registry
                .retired_operations
                .saturating_sub(before.registry.retired_operations),
            single_flight_joins: after
                .registry
                .single_flight_joins
                .saturating_sub(before.registry.single_flight_joins),
            physical_completions: after
                .registry
                .physical_completions
                .saturating_sub(before.registry.physical_completions),
            rejected_completions: after
                .registry
                .rejected_completions
                .saturating_sub(before.registry.rejected_completions),
            publications: after
                .registry
                .publications
                .saturating_sub(before.registry.publications),
            cancellations_requested: after
                .registry
                .cancellations_requested
                .saturating_sub(before.registry.cancellations_requested),
        },
        critical_path,
        external_snapshot_reconciliation,
    }
}

#[cfg(feature = "cuda")]
fn critical_path_stats(
    per_external_token: Vec<RuntimeExternalTokenSnapshot>,
    expected_external_tokens: usize,
) -> (RuntimeCriticalPathStats, ExternalSnapshotReconciliation) {
    let mut summary = RuntimeCriticalPathStats {
        external_tokens: per_external_token.len(),
        per_external_token,
        ..RuntimeCriticalPathStats::default()
    };
    let mut reconciled_external_tokens = 0usize;
    let mut consistent = summary.external_tokens == expected_external_tokens;
    let mut index = 0usize;
    while index < summary.per_external_token.len() {
        let representative = &summary.per_external_token[index];
        let declared = representative.externally_committed_tokens;
        let group_len = declared.max(1);
        let Some(end) = index.checked_add(group_len) else {
            consistent = false;
            break;
        };
        if declared == 0 || end > summary.per_external_token.len() {
            consistent = false;
        }
        let bounded_end = end.min(summary.per_external_token.len());
        let group_is_exact = declared > 0
            && bounded_end - index == declared
            && summary.per_external_token[index..bounded_end]
                .iter()
                .all(|snapshot| same_external_snapshot_group(representative, snapshot));
        if group_is_exact {
            reconciled_external_tokens = reconciled_external_tokens.saturating_add(declared);
        } else {
            consistent = false;
        }

        summary.snapshot_groups = summary.snapshot_groups.saturating_add(1);
        summary.read_ns = summary.read_ns.saturating_add(representative.read_ns);
        summary.upload_ns = summary.upload_ns.saturating_add(representative.upload_ns);
        summary.publish_ns = summary.publish_ns.saturating_add(representative.publish_ns);
        summary.wait_ns = summary.wait_ns.saturating_add(representative.wait_ns);
        summary.covered_wait_ns = summary
            .covered_wait_ns
            .saturating_add(representative.covered_wait_ns);
        summary.uncovered_wait_ns = summary
            .uncovered_wait_ns
            .saturating_add(representative.uncovered_wait_ns);
        summary.resume_ns = summary.resume_ns.saturating_add(representative.resume_ns);
        summary.commit_ns = summary.commit_ns.saturating_add(representative.commit_ns);
        index = bounded_end.max(index + 1);
    }
    consistent &= reconciled_external_tokens == summary.external_tokens;

    let reconciliation = ExternalSnapshotReconciliation {
        expected_external_tokens,
        snapshot_external_tokens: summary.external_tokens,
        reconciled_external_tokens,
        snapshot_groups: summary.snapshot_groups,
        consistent,
    };
    (summary, reconciliation)
}

#[cfg(feature = "cuda")]
fn same_external_snapshot_group(
    left: &RuntimeExternalTokenSnapshot,
    right: &RuntimeExternalTokenSnapshot,
) -> bool {
    left.externally_committed_tokens == right.externally_committed_tokens
        && left.captured_at_ns == right.captured_at_ns
        && left.read_ns == right.read_ns
        && left.upload_ns == right.upload_ns
        && left.publish_ns == right.publish_ns
        && left.wait_ns == right.wait_ns
        && left.covered_wait_ns == right.covered_wait_ns
        && left.uncovered_wait_ns == right.uncovered_wait_ns
        && left.resume_ns == right.resume_ns
        && left.commit_ns == right.commit_ns
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
fn finish_report_counters(
    report: &mut InteractiveBenchReport,
    total_prefill_us: u64,
    total_decode_us: u64,
    total_prompt_tokens: usize,
    total_generated: usize,
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
pub(crate) fn resident_driver_stats_snapshot(
    driver: &LocalResidentInferenceEngine<DeepSeekV4Runner, FixedSequenceSlotPool>,
) -> ResidentTopKDriverStats {
    let mut stats = driver.stats().clone();
    stats.hard_resource_high_water = driver
        .driver()
        .load_registry()
        .resources()
        .snapshots()
        .map(|snapshot| (snapshot.kind, snapshot.high_water))
        .collect();
    stats
}

#[cfg(feature = "cuda")]
pub(crate) fn resident_driver_stats_delta(
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
        hard_resource_high_water: after.hard_resource_high_water,
        speculative: speculative_metrics_delta(&before.speculative, &after.speculative),
    }
}

#[cfg(feature = "cuda")]
fn speculative_metrics_delta(
    before: &SpeculativeMetrics,
    after: &SpeculativeMetrics,
) -> SpeculativeMetrics {
    let histogram_len = after
        .accepted_prefix_histogram
        .len()
        .max(before.accepted_prefix_histogram.len());
    let accepted_prefix_histogram = (0..histogram_len)
        .map(|index| {
            after
                .accepted_prefix_histogram
                .get(index)
                .copied()
                .unwrap_or_default()
                .saturating_sub(
                    before
                        .accepted_prefix_histogram
                        .get(index)
                        .copied()
                        .unwrap_or_default(),
                )
        })
        .collect();
    SpeculativeMetrics {
        cycles: after.cycles.saturating_sub(before.cycles),
        proposed_tokens: after.proposed_tokens.saturating_sub(before.proposed_tokens),
        verified_rows: after.verified_rows.saturating_sub(before.verified_rows),
        accepted_draft_tokens: after
            .accepted_draft_tokens
            .saturating_sub(before.accepted_draft_tokens),
        correction_tokens: after
            .correction_tokens
            .saturating_sub(before.correction_tokens),
        externally_committed_tokens: after
            .externally_committed_tokens
            .saturating_sub(before.externally_committed_tokens),
        runtime_emitted_tokens: after
            .runtime_emitted_tokens
            .saturating_sub(before.runtime_emitted_tokens),
        rolled_back_rows: after
            .rolled_back_rows
            .saturating_sub(before.rolled_back_rows),
        rejected_tokens: after.rejected_tokens.saturating_sub(before.rejected_tokens),
        accepted_prefix_histogram,
        total_proposal_time_us: after
            .total_proposal_time_us
            .saturating_sub(before.total_proposal_time_us),
        total_transaction_time_us: after
            .total_transaction_time_us
            .saturating_sub(before.total_transaction_time_us),
        total_verify_time_us: after
            .total_verify_time_us
            .saturating_sub(before.total_verify_time_us),
        total_cycle_time_us: after
            .total_cycle_time_us
            .saturating_sub(before.total_cycle_time_us),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn dsv4_operator_counters_delta(
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
        moe_total_us: after.moe_total_us.saturating_sub(before.moe_total_us),
        moe_input_prepare_us: after
            .moe_input_prepare_us
            .saturating_sub(before.moe_input_prepare_us),
        moe_gate_up_us: after.moe_gate_up_us.saturating_sub(before.moe_gate_up_us),
        moe_swiglu_us: after.moe_swiglu_us.saturating_sub(before.moe_swiglu_us),
        moe_down_us: after.moe_down_us.saturating_sub(before.moe_down_us),
        moe_router_us: after.moe_router_us.saturating_sub(before.moe_router_us),
        moe_routing_us: after.moe_routing_us.saturating_sub(before.moe_routing_us),
        moe_plan_us: after.moe_plan_us.saturating_sub(before.moe_plan_us),
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
        output_head_rows: after
            .output_head_rows
            .saturating_sub(before.output_head_rows),
        output_head_topk_us: after
            .output_head_topk_us
            .saturating_sub(before.output_head_topk_us),
        expert_selected: after.expert_selected.saturating_sub(before.expert_selected),
        expert_unique_selected: after
            .expert_unique_selected
            .saturating_sub(before.expert_unique_selected),
        expert_selected_load_requests: after
            .expert_selected_load_requests
            .saturating_sub(before.expert_selected_load_requests),
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
        expert_io_slab_exhaustions: after
            .expert_io_slab_exhaustions
            .saturating_sub(before.expert_io_slab_exhaustions),
        expert_io_peak_queue_depth: after.expert_io_peak_queue_depth,
        expert_io_read_us: after
            .expert_io_read_us
            .saturating_sub(before.expert_io_read_us),
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
fn interactive_bench_report_json(report: &InteractiveBenchReport) -> serde_json::Value {
    serde_json::json!({
        "schema_version": CLI_RUNTIME_SCHEMA_VERSION,
        "model": report.model_dir,
        "chat_template": report.chat_template,
        "runtime_path": report.runtime_path,
        "dsv4_profile_sync": report.dsv4_profile_sync,
        "max_new_tokens": report.max_new_tokens,
        "max_layers": report.max_layers,
        "prefill_chunk_size": report.prefill_chunk_size,
        "output_head_chunk_rows": report.output_head_chunk_rows,
        "artifact_load_s": report.artifact_load_us as f64 / 1_000_000.0,
        "warmup_tokens": report.warmup_tokens,
        "warmup_s": report.warmup_us as f64 / 1_000_000.0,
        "warmup_generated_tokens": report.warmup_generated_tokens,
        "time_to_first_token_s": report.time_to_first_token_us as f64 / 1_000_000.0,
        "total_turns": report.turns.len(),
        "total_prompt_tokens": report.total_prompt_tokens,
        "total_prefill_s": report.total_prefill_us as f64 / 1_000_000.0,
        "total_generated": report.total_generated,
        "final_position": report.final_position,
        "aggregate_prefill_tok_per_s": report.aggregate_prefill_tok_per_s,
        "aggregate_decode_tok_per_s": report.aggregate_decode_tok_per_s,
        "runtime_driver_stats": resident_driver_stats_json(&report.runtime_driver_stats),
        "runtime_materialization": runtime_materialization_stats_json(&report.runtime_materialization),
        "dsv4_operator_counters": dsv4_operator_counters_json(&report.dsv4_operator_counters),
        "dsv4_layer_profile_summary": dsv4_layer_profile_summary_json(&report.dsv4_layer_profile_stats),
        "dsv4_layer_profile": dsv4_layer_profile_stats_json(&report.dsv4_layer_profile_stats),
        "dsv4_attention_profile_summary": dsv4_attention_profile_summary_json(&report.dsv4_attention_profile_stats),
        "dsv4_attention_profile": dsv4_attention_profile_stats_json(&report.dsv4_attention_profile_stats),
        "dsv4_output_profile": dsv4_output_profile_stats_json(&report.dsv4_output_profile_stats),
        "turns": report.turns.iter().map(interactive_turn_json).collect::<Vec<_>>(),
    })
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
        "runtime_materialization": runtime_materialization_stats_json(&turn.runtime_materialization),
        "dsv4_operator_counters": dsv4_operator_counters_json(&turn.dsv4_operator_counters),
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
pub(crate) fn resident_driver_stats_json(stats: &ResidentTopKDriverStats) -> serde_json::Value {
    serde_json::json!({
        "actions": stats.actions,
        "prefill_chunks": stats.prefill_chunks,
        "prefill_tokens": stats.prefill_tokens,
        "decode_steps": stats.decode_steps,
        "emitted_tokens": stats.emitted_tokens,
        "staged_tokens": stats.staged_tokens,
        "finished_sequences": stats.finished_sequences,
        "hard_resource_high_water": hard_resource_high_water_json(&stats.hard_resource_high_water),
        "speculative": speculative_metrics_json(&stats.speculative),
    })
}

#[cfg(feature = "cuda")]
fn hard_resource_high_water_json(high_water: &[(ResourceKind, u64)]) -> serde_json::Value {
    serde_json::Value::Object(
        ResourceKind::ALL
            .into_iter()
            .map(|kind| {
                let value = high_water
                    .iter()
                    .find_map(|(candidate, value)| (*candidate == kind).then_some(*value))
                    .expect("resident driver publishes every hard ResourceKind high-water mark");
                (
                    resource_kind_name(kind).into(),
                    serde_json::Value::from(value),
                )
            })
            .collect(),
    )
}

#[cfg(feature = "cuda")]
fn resource_kind_name(kind: ResourceKind) -> &'static str {
    match kind {
        ResourceKind::ReadSlot => "read_slot",
        ResourceKind::PinnedHostBytes => "pinned_host_bytes",
        ResourceKind::StorageReadBytes => "storage_read_bytes",
        ResourceKind::UploadSlot => "upload_slot",
        ResourceKind::UploadBytes => "upload_bytes",
        ResourceKind::InstallSlot => "install_slot",
        ResourceKind::DeviceInstallBytes => "device_install_bytes",
        ResourceKind::ResidentBytes => "resident_bytes",
        ResourceKind::ResidencyLease => "residency_lease",
        ResourceKind::Arena => "arena",
        ResourceKind::KvPage => "kv_page",
        ResourceKind::Continuation => "continuation",
        ResourceKind::Waiter => "waiter",
        ResourceKind::LoadOperation => "load_operation",
        ResourceKind::ReadyCohort => "ready_cohort",
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn runtime_materialization_stats_json(
    stats: &RuntimeMaterializationStats,
) -> serde_json::Value {
    serde_json::json!({
        "dependencies": {
            "selected": stats.dependencies.selected,
            "unique_selected": stats.dependencies.unique_selected,
            "resident": stats.dependencies.resident,
            "waiting": stats.dependencies.waiting,
        },
        "resolver": {
            "resolves": stats.resolver.resolves,
        },
        "stages": {
            "read": { "count": stats.stages.read_stages, "bytes": stats.stages.read_bytes },
            "upload": { "count": stats.stages.upload_stages, "bytes": stats.stages.upload_bytes },
            "install": { "count": stats.stages.install_stages, "bytes": stats.stages.install_bytes },
            "publish": { "count": stats.stages.publish_stages, "bytes": stats.stages.publish_bytes },
            "failures": stats.stages.failures,
            "stale": stats.stages.stale,
            "cancellations": stats.stages.cancellations,
        },
        "load_registry": {
            "operations_created": stats.registry.operations_created,
            "active_operations": stats.registry.active_operations,
            "retired_operations": stats.registry.retired_operations,
            "single_flight_joins": stats.registry.single_flight_joins,
            "physical_completions": stats.registry.physical_completions,
            "rejected_completions": stats.registry.rejected_completions,
            "publications": stats.registry.publications,
            "cancellations_requested": stats.registry.cancellations_requested,
        },
        "critical_path": {
            "external_tokens": stats.critical_path.external_tokens,
            "snapshot_groups": stats.critical_path.snapshot_groups,
            "read_ns": stats.critical_path.read_ns,
            "upload_ns": stats.critical_path.upload_ns,
            "publish_ns": stats.critical_path.publish_ns,
            "wait_ns": stats.critical_path.wait_ns,
            "covered_wait_ns": stats.critical_path.covered_wait_ns,
            "uncovered_wait_ns": stats.critical_path.uncovered_wait_ns,
            "resume_ns": stats.critical_path.resume_ns,
            "commit_ns": stats.critical_path.commit_ns,
            "per_external_token": stats.critical_path.per_external_token.iter().map(|snapshot| serde_json::json!({
                "external_token_id": snapshot.external_token_id,
                "externally_committed_tokens": snapshot.externally_committed_tokens,
                "captured_at_ns": snapshot.captured_at_ns,
                "read_ns": snapshot.read_ns,
                "upload_ns": snapshot.upload_ns,
                "publish_ns": snapshot.publish_ns,
                "wait_ns": snapshot.wait_ns,
                "covered_wait_ns": snapshot.covered_wait_ns,
                "uncovered_wait_ns": snapshot.uncovered_wait_ns,
                "resume_ns": snapshot.resume_ns,
                "commit_ns": snapshot.commit_ns,
            })).collect::<Vec<_>>(),
        },
        "external_snapshot_reconciliation": {
            "expected_external_tokens": stats.external_snapshot_reconciliation.expected_external_tokens,
            "snapshot_external_tokens": stats.external_snapshot_reconciliation.snapshot_external_tokens,
            "reconciled_external_tokens": stats.external_snapshot_reconciliation.reconciled_external_tokens,
            "snapshot_groups": stats.external_snapshot_reconciliation.snapshot_groups,
            "consistent": stats.external_snapshot_reconciliation.consistent,
        },
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn print_runtime_materialization_summary(stats: &RuntimeMaterializationStats) {
    println!(
        "runtime_materialization: selected={} resident={} waiting={} resolves={}",
        stats.dependencies.selected,
        stats.dependencies.resident,
        stats.dependencies.waiting,
        stats.resolver.resolves,
    );
    println!(
        "materialization_stages:  read={}/{}B upload={}/{}B install={}/{}B publish={}/{}B failures={} stale={} cancelled={}",
        stats.stages.read_stages,
        stats.stages.read_bytes,
        stats.stages.upload_stages,
        stats.stages.upload_bytes,
        stats.stages.install_stages,
        stats.stages.install_bytes,
        stats.stages.publish_stages,
        stats.stages.publish_bytes,
        stats.stages.failures,
        stats.stages.stale,
        stats.stages.cancellations,
    );
    println!(
        "load_registry:          active={} retired={} joins={} created={} completions={}",
        stats.registry.active_operations,
        stats.registry.retired_operations,
        stats.registry.single_flight_joins,
        stats.registry.operations_created,
        stats.registry.physical_completions,
    );
    println!(
        "critical_path:          read={}ns upload={}ns publish={}ns wait={}ns resume={}ns uncovered={}ns external_tokens={}",
        stats.critical_path.read_ns,
        stats.critical_path.upload_ns,
        stats.critical_path.publish_ns,
        stats.critical_path.wait_ns,
        stats.critical_path.resume_ns,
        stats.critical_path.uncovered_wait_ns,
        stats.critical_path.external_tokens,
    );
    println!(
        "external_reconciliation: expected={} snapshots={} reconciled={} groups={} consistent={}",
        stats
            .external_snapshot_reconciliation
            .expected_external_tokens,
        stats
            .external_snapshot_reconciliation
            .snapshot_external_tokens,
        stats
            .external_snapshot_reconciliation
            .reconciled_external_tokens,
        stats.external_snapshot_reconciliation.snapshot_groups,
        stats.external_snapshot_reconciliation.consistent,
    );
}

#[cfg(feature = "cuda")]
pub(crate) fn print_hard_resource_high_water(high_water: &[(ResourceKind, u64)]) {
    let values = ResourceKind::ALL
        .into_iter()
        .map(|kind| {
            let value = high_water
                .iter()
                .find_map(|(candidate, value)| (*candidate == kind).then_some(*value))
                .expect("resident driver publishes every hard ResourceKind high-water mark");
            format!("{}={value}", resource_kind_name(kind))
        })
        .collect::<Vec<_>>()
        .join(" ");
    println!("hard_resource_high_water:  {values}");
}

#[cfg(feature = "cuda")]
fn speculative_metrics_json(metrics: &SpeculativeMetrics) -> serde_json::Value {
    serde_json::json!({
        "cycles": metrics.cycles,
        "proposed_tokens": metrics.proposed_tokens,
        "verified_rows": metrics.verified_rows,
        "accepted_draft_tokens": metrics.accepted_draft_tokens,
        "acceptance_rate": metrics.acceptance_rate(),
        "correction_tokens": metrics.correction_tokens,
        "externally_committed_tokens": metrics.externally_committed_tokens,
        "runtime_emitted_tokens": metrics.runtime_emitted_tokens,
        "rolled_back_rows": metrics.rolled_back_rows,
        "rejected_tokens": metrics.rejected_tokens,
        "accepted_prefix_histogram": &metrics.accepted_prefix_histogram,
        "total_proposal_s": metrics.total_proposal_time_us as f64 / 1_000_000.0,
        "total_verify_s": metrics.total_verify_time_us as f64 / 1_000_000.0,
        "total_transaction_s": metrics.total_transaction_time_us as f64 / 1_000_000.0,
        "total_cycle_s": metrics.total_cycle_time_us as f64 / 1_000_000.0,
        "mean_cycle_s": metrics.mean_cycle_time_us() as f64 / 1_000_000.0,
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn dsv4_operator_counters_json(
    stats: &DeepSeekV4OperatorRuntimeCounters,
) -> serde_json::Value {
    serde_json::json!({
        "cuda": {
            "kernel_launches": stats.kernel_launches,
            "host_to_device_copies": stats.host_to_device_copies,
            "host_to_device_bytes": stats.host_to_device_bytes,
            "device_to_host_copies": stats.device_to_host_copies,
            "device_to_host_bytes": stats.device_to_host_bytes,
            "artifact_uploads": stats.artifact_uploads,
            "artifact_upload_bytes": stats.artifact_upload_bytes,
            "device_allocation_attempts": stats.device_allocation_attempts,
            "device_allocations": stats.device_allocations,
            "device_allocation_failures": stats.device_allocation_failures,
            "device_allocation_bytes": stats.device_allocation_bytes,
            "stream_wide_syncs": stats.stream_wide_syncs,
            "stream_wide_sync_failures": stats.stream_wide_sync_failures,
        },
        "moe_compute": {
            "calls": stats.moe_calls,
            "total_us": stats.moe_total_us,
            "input_prepare_us": stats.moe_input_prepare_us,
            "gate_up_us": stats.moe_gate_up_us,
            "swiglu_us": stats.moe_swiglu_us,
            "down_us": stats.moe_down_us,
            "router_us": stats.moe_router_us,
            "routing_us": stats.moe_routing_us,
            "plan_us": stats.moe_plan_us,
            "shared_us": stats.moe_shared_us,
            "workspace_us": stats.moe_workspace_us,
            "compute_submit_us": stats.moe_compute_submit_us,
            "commit_us": stats.moe_commit_us,
        },
        "output_head": {
            "calls": stats.output_head_calls,
            "rows": stats.output_head_rows,
            "topk_us": stats.output_head_topk_us,
        },
        "expert_dependencies": {
            "selected": stats.expert_selected,
            "unique_selected": stats.expert_unique_selected,
            "selected_load_requests": stats.expert_selected_load_requests,
        },
        "expert_io": {
            "submitted_extents": stats.expert_io_submitted_extents,
            "completed_extents": stats.expert_io_completed_extents,
            "failed_extents": stats.expert_io_failed_extents,
            "requested_bytes": stats.expert_io_requested_bytes,
            "aligned_bytes": stats.expert_io_aligned_bytes,
            "coalesced_slices": stats.expert_io_coalesced_slices,
            "fixed_file_registrations": stats.expert_io_fixed_file_registrations,
            "slab_exhaustions": stats.expert_io_slab_exhaustions,
            "peak_queue_depth": stats.expert_io_peak_queue_depth,
            "read_us": stats.expert_io_read_us,
        },
        "arena": {
            "hits": stats.arena_hits,
            "misses": stats.arena_misses,
            "grows": stats.arena_grows,
            "reuses": stats.arena_reuses,
        },
        "expert_residency": {
            "resident": stats.expert_residency_stats.resident,
            "active_leases": stats.expert_residency_stats.active_leases,
            "installs": stats.expert_residency_stats.installs,
            "evictions": stats.expert_residency_stats.evictions,
            "resident_hits": stats.expert_residency_stats.resident_hits,
            "stale_releases": stats.expert_residency_stats.stale_releases,
            "prepare_cancellations": stats.expert_residency_stats.prepare_cancellations,
            "prefetch_capacity_misses": stats.expert_residency_stats.prefetch_capacity_misses,
        },
        "expert_prediction": {
            "predict_calls": stats.expert_predictor_stats.predict_calls,
            "predicted_experts": stats.expert_predictor_stats.predicted_experts,
            "observe_calls": stats.expert_predictor_stats.observe_calls,
            "observed_experts": stats.expert_predictor_stats.observed_experts,
            "cold_miss_observations": stats.expert_predictor_stats.cold_miss_observations,
            "transition_observations": stats.expert_predictor_stats.transition_observations,
            "transition_predictions": stats.expert_predictor_stats.transition_predictions,
        },
    })
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    fn external_snapshot(
        external_token_id: u64,
        externally_committed_tokens: usize,
        captured_at_ns: u64,
        read_ns: u64,
    ) -> RuntimeExternalTokenSnapshot {
        RuntimeExternalTokenSnapshot {
            external_token_id,
            externally_committed_tokens,
            captured_at_ns,
            read_ns,
            upload_ns: 11,
            publish_ns: 13,
            wait_ns: 17,
            covered_wait_ns: 5,
            uncovered_wait_ns: 12,
            resume_ns: 19,
            commit_ns: 23,
        }
    }

    #[test]
    fn runtime_delta_saturates_monotonic_counters_and_keeps_after_gauges() {
        let mut before = RuntimeMaterializationStats::default();
        before.dependencies.selected = 10;
        before.dependencies.unique_selected = 8;
        before.dependencies.resident = 6;
        before.dependencies.waiting = 4;
        before.resolver.resolves = 9;
        before.stages.read_stages = 7;
        before.stages.read_bytes = 700;
        before.registry.active_operations = 5;
        before.registry.retired_operations = 12;
        before.critical_path.per_external_token = vec![external_snapshot(1, 1, 100, 3)];

        let mut after = RuntimeMaterializationStats::default();
        after.dependencies.selected = 6;
        after.dependencies.unique_selected = 11;
        after.dependencies.resident = 2;
        after.dependencies.waiting = 3;
        after.resolver.resolves = 14;
        after.stages.read_stages = 5;
        after.stages.read_bytes = 900;
        after.registry.active_operations = 1;
        after.registry.retired_operations = 15;
        after.critical_path.per_external_token = vec![
            external_snapshot(1, 1, 100, 3),
            external_snapshot(2, 2, 200, 7),
            external_snapshot(3, 2, 200, 7),
        ];

        let delta = runtime_materialization_stats_delta(before, after, 2);
        assert_eq!(delta.dependencies.selected, 0);
        assert_eq!(delta.dependencies.unique_selected, 3);
        assert_eq!(delta.dependencies.resident, 2);
        assert_eq!(delta.dependencies.waiting, 3);
        assert_eq!(delta.resolver.resolves, 5);
        assert_eq!(delta.stages.read_stages, 0);
        assert_eq!(delta.stages.read_bytes, 200);
        assert_eq!(delta.registry.active_operations, 1);
        assert_eq!(delta.registry.retired_operations, 3);
        assert_eq!(
            delta
                .critical_path
                .per_external_token
                .iter()
                .map(|snapshot| snapshot.external_token_id)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(delta.critical_path.read_ns, 7);
        assert_eq!(delta.critical_path.snapshot_groups, 1);
        assert_eq!(
            delta
                .external_snapshot_reconciliation
                .reconciled_external_tokens,
            2
        );
        assert!(delta.external_snapshot_reconciliation.consistent);
    }

    #[test]
    fn operator_and_driver_delta_use_after_for_gauges_and_lists() {
        let before_operator = DeepSeekV4OperatorRuntimeCounters {
            kernel_launches: 10,
            expert_io_peak_queue_depth: 9,
            expert_residency_stats: ferrule_common::ExpertResidencyStats {
                resident: 8,
                active_leases: 7,
                installs: 5,
                ..Default::default()
            },
            ..Default::default()
        };
        let after_operator = DeepSeekV4OperatorRuntimeCounters {
            kernel_launches: 4,
            expert_io_peak_queue_depth: 3,
            expert_residency_stats: ferrule_common::ExpertResidencyStats {
                resident: 2,
                active_leases: 1,
                installs: 9,
                ..Default::default()
            },
            ..Default::default()
        };
        let operator_delta = dsv4_operator_counters_delta(before_operator, after_operator);
        assert_eq!(operator_delta.kernel_launches, 0);
        assert_eq!(operator_delta.expert_io_peak_queue_depth, 3);
        assert_eq!(operator_delta.expert_residency_stats.resident, 2);
        assert_eq!(operator_delta.expert_residency_stats.active_leases, 1);
        assert_eq!(operator_delta.expert_residency_stats.installs, 4);

        let before_driver = ResidentTopKDriverStats {
            hard_resource_high_water: ResourceKind::ALL
                .into_iter()
                .map(|kind| (kind, 99))
                .collect(),
            ..Default::default()
        };
        let after_high_water = ResourceKind::ALL
            .into_iter()
            .enumerate()
            .map(|(index, kind)| (kind, index as u64 + 1))
            .collect::<Vec<_>>();
        let after_driver = ResidentTopKDriverStats {
            hard_resource_high_water: after_high_water.clone(),
            ..Default::default()
        };
        let driver_delta = resident_driver_stats_delta(before_driver, after_driver);
        assert_eq!(driver_delta.hard_resource_high_water, after_high_water);
        assert_eq!(
            hard_resource_high_water_json(&driver_delta.hard_resource_high_water)
                .as_object()
                .unwrap()
                .len(),
            ResourceKind::ALL.len()
        );
    }

    #[test]
    fn v4_schema_keeps_registry_as_the_physical_authority() {
        let mut report = InteractiveBenchReport::default();
        report.runtime_driver_stats.hard_resource_high_water = ResourceKind::ALL
            .into_iter()
            .map(|kind| (kind, 1))
            .collect();
        let json = interactive_bench_report_json(&report);
        assert_eq!(json["schema_version"], CLI_RUNTIME_SCHEMA_VERSION);
        assert!(json["runtime_materialization"]["resolver"]["resolves"].is_number());
        assert!(
            json["runtime_materialization"]["resolver"]
                .get("physical_submissions")
                .is_none()
        );
        assert!(json["runtime_materialization"]["load_registry"]["operations_created"].is_number());
        assert!(json["runtime_materialization"]["critical_path"]["per_external_token"].is_array());
        let high_water = json["runtime_driver_stats"]["hard_resource_high_water"]
            .as_object()
            .unwrap();
        assert_eq!(high_water.len(), ResourceKind::ALL.len());
        assert_eq!(high_water["resident_bytes"], 1);
        assert_eq!(high_water["residency_lease"], 1);
        assert!(!high_water.contains_key("expert_frame"));
        assert!(!high_water.contains_key("lease"));

        let encoded = serde_json::to_string(&json).unwrap();
        for legacy in [
            concat!("expert_", "loads"),
            concat!("expert_", "load_bytes"),
            concat!("expert_", "host_cache"),
            concat!("expert_", "pinned_cache"),
            concat!("expert_cuda_", "resident_entries"),
            concat!("expert_async_", "prefetch"),
            concat!("expert_upload_", "prefetch"),
            concat!("moe_", "prefetch_"),
            concat!("moe_expert_", "read"),
            concat!("moe_expert_", "upload"),
            concat!("expert_selected_", "upload_wait"),
            concat!("operation_", "id"),
        ] {
            assert!(
                !encoded.contains(legacy),
                "legacy key fragment remained in v3 schema: {legacy}"
            );
        }
    }
}

#[cfg(not(feature = "cuda"))]
#[expect(
    clippy::too_many_arguments,
    reason = "the non-CUDA stub mirrors the CUDA command interface"
)]
pub fn cmd_bench_interactive(
    _model_dir: &str,
    _prompts: &[String],
    _max_new_tokens: usize,
    _chat_template_override: Option<&str>,
    _warmup_tokens: usize,
    _max_layers: usize,
    _prefill_chunk_size: usize,
    _output_head_chunk_rows: usize,
    _moe_hotset_experts: usize,
    _golden_trace_path: Option<&str>,
    _json: bool,
) -> anyhow::Result<()> {
    anyhow::bail!("bench-interactive requires --features cuda")
}

#[cfg(feature = "cuda")]
fn duration_us(d: Duration) -> u64 {
    d.as_micros().min(u128::from(u64::MAX)) as u64
}
