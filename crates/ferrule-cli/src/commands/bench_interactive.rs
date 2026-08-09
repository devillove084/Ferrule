//! Model-neutral interactive benchmark built through the resident model factory.

use std::time::{Duration, Instant};

use ferrule_common::io_protocol::LoadStage;
use ferrule_model::{AutoConfig, ChatTemplate};
use ferrule_runtime::engine::LocalSessionInferenceEngine;
use ferrule_runtime::engine::model_factory::ExpertCacheOptions;
use ferrule_runtime::{
    BackendSelection, GenerateRequest, ModelFactoryOptions, RequestId, ResidentActionKind,
    ResidentDriverStep, ResidentEngineObservability, ResidentExternalTokenObservability,
    ResidentKvCacheObservability, ResidentMaterializationObservability,
    ResidentModelBuildObservability, ResidentModelBuildPlan, ResidentModelPlanner,
    ResidentTopKDriverStats, ResourceKind, SessionId, SpeculativeMetrics,
};

use crate::GenerationConfig;
use crate::bench::{GoldenTurn, InteractiveTrace, compare_interactive_trace};

use super::resident::{
    block_on_local_inference, require_finished_request, resident_driver_config,
    single_sequence_scheduler_config,
};

pub(crate) const CLI_RUNTIME_SCHEMA_VERSION: u32 = 5;

#[derive(Debug, Clone, Default)]
struct RuntimeStepMeasurement {
    action_kind: String,
    rows: usize,
    staged: usize,
    finished: usize,
    elapsed_us: u64,
    session_position: usize,
}

#[derive(Debug, Clone)]
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
    observability: ResidentEngineObservability,
    runtime_steps: Vec<RuntimeStepMeasurement>,
}

#[derive(Debug, Clone)]
struct InteractiveBenchReport {
    model_dir: String,
    build: ResidentModelBuildObservability,
    output_head_chunk_rows: usize,
    runtime_path: &'static str,
    engine_build_us: u64,
    warmup_tokens: usize,
    warmup_us: u64,
    warmup_generated_tokens: usize,
    time_to_first_token_us: u64,
    turns: Vec<InteractiveTurnMeasurement>,
    aggregate_prefill_tok_per_s: f64,
    aggregate_decode_tok_per_s: f64,
    total_prompt_tokens: usize,
    total_prefill_us: u64,
    total_generated: usize,
    final_position: usize,
    observability: ResidentEngineObservability,
}

#[expect(
    clippy::too_many_arguments,
    reason = "the command entry point mirrors its CLI arguments"
)]
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
    if max_layers == 0 {
        anyhow::bail!("max-layers must be greater than zero");
    }
    if prefill_chunk_size == 0 {
        anyhow::bail!("prefill-chunk-size must be greater than zero");
    }
    if output_head_chunk_rows == 0 {
        anyhow::bail!("output-head-chunk-rows must be greater than zero");
    }

    let generation = GenerationConfig {
        max_new_tokens,
        stop: Vec::new(),
        ctx_size: 4096,
        ..GenerationConfig::default()
    };
    let config = AutoConfig::from_pretrained(model_dir)?;
    let prepared = ResidentModelPlanner::new().prepare(
        &config,
        BackendSelection::Auto,
        chat_template_override,
        ModelFactoryOptions {
            max_layers: Some(max_layers),
            max_tensor_mebibytes: 128,
            output_head_chunk_rows,
            expert_reader_max_tensor_mebibytes: 64,
            expert_cache: ExpertCacheOptions::default(),
            moe_hotset_experts,
            kv_cache_mebibytes: None,
            scheduler_config: single_sequence_scheduler_config(prefill_chunk_size),
            driver_config: resident_driver_config(generation.ctx_size, generation.stop_at_eos),
        },
    )?;
    let build = prepared.observability();
    let chat_template = prepared.chat_template();

    let report = run_with_resident_engine(
        prepared,
        model_dir,
        build,
        output_head_chunk_rows,
        chat_template,
        &generation,
        prompts,
        warmup_tokens,
        json,
    )?;

    if json {
        let mut output = interactive_bench_report_json(&report);
        if let Some(golden_path) = golden_trace_path {
            let golden_json = std::fs::read_to_string(golden_path)?;
            let golden: InteractiveTrace = serde_json::from_str(&golden_json)?;
            let observed_turns = report
                .turns
                .iter()
                .map(|turn| GoldenTurn {
                    prompt_text: turn.prompt_text.clone(),
                    prompt_tokens: turn.prompt_tokens.clone(),
                    generated_tokens: turn.generated_tokens.clone(),
                    stopped_by_eos: turn.stopped_by_eos,
                    stopped_by_string: turn.stopped_by_string.clone(),
                })
                .collect::<Vec<_>>();
            let comparison = compare_interactive_trace(&golden, &observed_turns);
            output["golden"] = serde_json::json!({
                "label": comparison.label,
                "turns_compared": comparison.turns_compared,
                "turns_ok": comparison.turns_ok,
                "all_ok": comparison.all_ok(),
                "mismatches": comparison.mismatches.iter().map(|mismatch| {
                    serde_json::json!({
                        "turn": mismatch.turn_index,
                        "prompt": mismatch.prompt_text,
                        "message": mismatch.message,
                        "expected_tokens": mismatch.expected_tokens,
                        "observed_tokens": mismatch.observed_tokens,
                    })
                }).collect::<Vec<_>>(),
            });
        }
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        print_report(&report);
    }

    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "benchmark construction keeps the prepared plan and report metadata explicit"
)]
fn run_with_resident_engine(
    prepared: ResidentModelBuildPlan,
    model_dir: &str,
    build: ResidentModelBuildObservability,
    output_head_chunk_rows: usize,
    chat_template: ChatTemplate,
    generation: &GenerationConfig,
    prompts: &[String],
    warmup_tokens: usize,
    json: bool,
) -> anyhow::Result<InteractiveBenchReport> {
    block_on_local_inference(run_with_resident_engine_async(
        prepared,
        model_dir,
        build,
        output_head_chunk_rows,
        chat_template,
        generation,
        prompts,
        warmup_tokens,
        json,
    ))
}

#[expect(
    clippy::too_many_arguments,
    reason = "benchmark construction keeps the prepared plan and report metadata explicit"
)]
async fn run_with_resident_engine_async(
    prepared: ResidentModelBuildPlan,
    model_dir: &str,
    build: ResidentModelBuildObservability,
    output_head_chunk_rows: usize,
    chat_template: ChatTemplate,
    generation: &GenerationConfig,
    prompts: &[String],
    warmup_tokens: usize,
    json: bool,
) -> anyhow::Result<InteractiveBenchReport> {
    let build_started = Instant::now();
    let mut engine = LocalSessionInferenceEngine::new(prepared.build()?);
    let engine_build_us = duration_us(build_started.elapsed());
    engine.initialize().await?;
    engine.wait_for_model_warmup().await?;

    let mut report = InteractiveBenchReport {
        model_dir: model_dir.to_owned(),
        build,
        output_head_chunk_rows,
        runtime_path: "resident_model_planner",
        engine_build_us,
        warmup_tokens,
        warmup_us: 0,
        warmup_generated_tokens: 0,
        time_to_first_token_us: 0,
        turns: Vec::new(),
        aggregate_prefill_tok_per_s: 0.0,
        aggregate_decode_tok_per_s: 0.0,
        total_prompt_tokens: 0,
        total_prefill_us: 0,
        total_generated: 0,
        final_position: 0,
        observability: engine.observability_snapshot(),
    };

    if warmup_tokens > 0 {
        let baseline = engine.observability_snapshot();
        let warmup_prompt = chat_template.format_turn("warmup", true);
        let warmup_prompt_tokens = engine.encode(&warmup_prompt)?;
        let warmup_started = Instant::now();
        engine.submit(driver_request(
            0,
            SessionId(u64::MAX),
            warmup_prompt_tokens,
            warmup_tokens,
            &generation.stop,
        ));
        drive_request_to_completion(
            &mut engine,
            RequestId(0),
            "interactive benchmark warmup",
            &mut |_| Ok(()),
        )
        .await?;
        report.warmup_us = duration_us(warmup_started.elapsed());
        report.warmup_generated_tokens = engine
            .observability_snapshot()
            .delta_since(&baseline)
            .driver
            .emitted_tokens;
    }

    let measured_baseline = engine.observability_snapshot();
    let mut first_token_measured = false;
    let mut total_decode_us = 0u64;

    for (turn_index, prompt_text) in prompts.iter().enumerate() {
        let full_prompt = chat_template.format_turn(prompt_text, turn_index == 0);
        let prompt_tokens = engine.encode(&full_prompt)?;
        if prompt_tokens.is_empty() {
            if !json {
                eprintln!(
                    "[bench] turn {} prompt encoded to zero tokens, skipping",
                    turn_index + 1
                );
            }
            continue;
        }

        let request_id = RequestId(turn_index as u64 + 1);
        let turn_baseline = engine.observability_snapshot();
        engine.submit(driver_request(
            request_id.0,
            SessionId(0),
            prompt_tokens.clone(),
            generation.max_new_tokens,
            &generation.stop,
        ));

        let turn_started = Instant::now();
        let mut first_token_us = None;
        let mut prefill_us = 0u64;
        let mut decode_us = 0u64;
        let mut generated_tokens = Vec::new();
        let mut runtime_steps = Vec::new();
        let sequence = loop {
            let step_started = Instant::now();
            let step = engine
                .step(&mut |event| {
                    first_token_us.get_or_insert_with(|| duration_us(turn_started.elapsed()));
                    generated_tokens.push(event.token);
                    Ok(())
                })
                .await?;
            let step_us = duration_us(step_started.elapsed());
            let step_is_idle = matches!(step, ResidentDriverStep::Idle);
            let session_position = engine
                .retained_session_position(SessionId(0))
                .unwrap_or_default();

            match step {
                ResidentDriverStep::Executed {
                    action_kind,
                    rows,
                    staged,
                    finished,
                } => {
                    match action_kind {
                        ResidentActionKind::Prefill | ResidentActionKind::Mixed => {
                            prefill_us = prefill_us.saturating_add(step_us);
                        }
                        ResidentActionKind::Decode => {
                            decode_us = decode_us.saturating_add(step_us);
                        }
                        ResidentActionKind::Finish | ResidentActionKind::Cancel => {}
                    }
                    runtime_steps.push(RuntimeStepMeasurement {
                        action_kind: resident_action_kind_name(action_kind).to_owned(),
                        rows,
                        staged,
                        finished,
                        elapsed_us: step_us,
                        session_position,
                    });
                }
                ResidentDriverStep::WaitingForModelProgress(_) => {
                    decode_us = decode_us.saturating_add(step_us);
                    runtime_steps.push(RuntimeStepMeasurement {
                        action_kind: "model_wait".to_owned(),
                        elapsed_us: step_us,
                        session_position,
                        ..Default::default()
                    });
                }
                ResidentDriverStep::Idle => {}
                ResidentDriverStep::Blocked => {
                    anyhow::bail!("resident runtime blocked while running measured turn")
                }
            }

            if let Some(terminal) = engine.take_request_terminal(request_id) {
                break require_finished_request(terminal, "interactive benchmark turn")?;
            }
            if step_is_idle {
                anyhow::bail!("interactive benchmark became idle before request terminalization");
            }
        };

        let first_token_us = first_token_us.unwrap_or_else(|| duration_us(turn_started.elapsed()));
        if !first_token_measured {
            report.time_to_first_token_us = first_token_us;
            first_token_measured = true;
        }
        let finish_reason = sequence.finish_reason;
        let stopped_by_string = finish_reason
            .is_some_and(|reason| reason == ferrule_runtime::SequenceFinishReason::StopString)
            .then(|| matched_stop_string(&sequence.generated_text, &generation.stop))
            .flatten();
        let stopped_by_eos = finish_reason
            .is_some_and(|reason| reason == ferrule_runtime::SequenceFinishReason::Eos);

        report.total_prefill_us = report.total_prefill_us.saturating_add(prefill_us);
        total_decode_us = total_decode_us.saturating_add(decode_us);
        report.total_prompt_tokens = report
            .total_prompt_tokens
            .saturating_add(prompt_tokens.len());
        report.total_generated = report
            .total_generated
            .saturating_add(generated_tokens.len());
        report.final_position = sequence.position;
        report.turns.push(InteractiveTurnMeasurement {
            prompt_text: prompt_text.clone(),
            prompt_tokens,
            first_token_us,
            prefill_us,
            decode_us,
            generated_tokens,
            final_position: sequence.position,
            finish_reason: finish_reason
                .map(|reason| reason.as_str().to_owned())
                .unwrap_or_else(|| "unknown".to_owned()),
            stopped_by_eos,
            stopped_by_string,
            observability: engine.observability_snapshot().delta_since(&turn_baseline),
            runtime_steps,
        });
    }

    report.observability = engine
        .observability_snapshot()
        .delta_since(&measured_baseline);
    report.aggregate_prefill_tok_per_s = report.total_prompt_tokens as f64
        / (report.total_prefill_us as f64 / 1_000_000.0).max(1e-6);
    report.aggregate_decode_tok_per_s =
        report.total_generated as f64 / (total_decode_us as f64 / 1_000_000.0).max(1e-6);
    engine.shutdown().await?;
    Ok(report)
}

async fn drive_request_to_completion(
    engine: &mut LocalSessionInferenceEngine,
    request_id: RequestId,
    context: &str,
    on_token: &mut dyn FnMut(&ferrule_runtime::ResidentTokenEvent) -> ferrule_runtime::Result<()>,
) -> anyhow::Result<()> {
    loop {
        let step = engine.step(on_token).await?;
        if let Some(terminal) = engine.take_request_terminal(request_id) {
            let _ = require_finished_request(terminal, context)?;
            return Ok(());
        }
        match step {
            ResidentDriverStep::Idle => {
                anyhow::bail!("{context} became idle before request terminalization")
            }
            ResidentDriverStep::Blocked => anyhow::bail!("{context} blocked"),
            ResidentDriverStep::WaitingForModelProgress(_)
            | ResidentDriverStep::Executed { .. } => {}
        }
    }
}

fn driver_request(
    request_id: u64,
    session_id: SessionId,
    prompt_tokens: Vec<u32>,
    max_new_tokens: usize,
    stop: &[String],
) -> GenerateRequest {
    GenerateRequest {
        id: RequestId(request_id),
        session_id: Some(session_id),
        prompt_tokens,
        max_new_tokens,
        stop: stop.to_vec(),
        ignore_eos: false,
    }
}

fn matched_stop_string(text: &str, stop: &[String]) -> Option<String> {
    stop.iter()
        .find(|value| text.ends_with(value.as_str()))
        .cloned()
}

fn resident_action_kind_name(kind: ResidentActionKind) -> &'static str {
    match kind {
        ResidentActionKind::Prefill => "prefill",
        ResidentActionKind::Decode => "decode",
        ResidentActionKind::Mixed => "mixed",
        ResidentActionKind::Finish => "finish",
        ResidentActionKind::Cancel => "cancel",
    }
}

fn interactive_bench_report_json(report: &InteractiveBenchReport) -> serde_json::Value {
    serde_json::json!({
        "schema_version": CLI_RUNTIME_SCHEMA_VERSION,
        "model": report.model_dir,
        "build": build_observability_json(report.build),
        "runtime_path": report.runtime_path,
        "output_head_chunk_rows": report.output_head_chunk_rows,
        "engine_build_s": seconds(report.engine_build_us),
        "warmup_tokens": report.warmup_tokens,
        "warmup_s": seconds(report.warmup_us),
        "warmup_generated_tokens": report.warmup_generated_tokens,
        "time_to_first_token_s": seconds(report.time_to_first_token_us),
        "total_turns": report.turns.len(),
        "total_prompt_tokens": report.total_prompt_tokens,
        "total_prefill_s": seconds(report.total_prefill_us),
        "total_generated": report.total_generated,
        "final_position": report.final_position,
        "aggregate_prefill_tok_per_s": report.aggregate_prefill_tok_per_s,
        "aggregate_decode_tok_per_s": report.aggregate_decode_tok_per_s,
        "runtime": engine_observability_json(&report.observability),
        "turns": report.turns.iter().map(interactive_turn_json).collect::<Vec<_>>(),
    })
}

fn interactive_turn_json(turn: &InteractiveTurnMeasurement) -> serde_json::Value {
    serde_json::json!({
        "prompt": turn.prompt_text,
        "prompt_tokens": turn.prompt_tokens.len(),
        "prompt_token_ids": turn.prompt_tokens,
        "first_token_s": seconds(turn.first_token_us),
        "prefill_s": seconds(turn.prefill_us),
        "decode_s": seconds(turn.decode_us),
        "prefill_tok_per_s": turn.prompt_tokens.len() as f64 / seconds(turn.prefill_us).max(1e-6),
        "decode_tok_per_s": turn.generated_tokens.len() as f64 / seconds(turn.decode_us).max(1e-6),
        "generated_tokens": turn.generated_tokens.len(),
        "generated_token_ids": turn.generated_tokens,
        "final_position": turn.final_position,
        "finish_reason": turn.finish_reason,
        "stopped_by_eos": turn.stopped_by_eos,
        "stopped_by_string": turn.stopped_by_string,
        "runtime": engine_observability_json(&turn.observability),
        "runtime_steps": turn.runtime_steps.iter().map(runtime_step_json).collect::<Vec<_>>(),
    })
}

fn runtime_step_json(step: &RuntimeStepMeasurement) -> serde_json::Value {
    serde_json::json!({
        "action_kind": step.action_kind,
        "rows": step.rows,
        "staged": step.staged,
        "finished": step.finished,
        "elapsed_s": seconds(step.elapsed_us),
        "session_position": step.session_position,
    })
}

fn build_observability_json(build: ResidentModelBuildObservability) -> serde_json::Value {
    serde_json::json!({
        "model_name": build.model_name,
        "backend": build.backend.as_str(),
        "backend_profile": build.backend_profile,
        "chat_template": build.chat_template.name(),
        "max_layers": build.max_layers,
        "ctx_size": build.ctx_size,
        "max_active_sequences": build.max_active_sequences,
        "kv_cache_budget_bytes": build.kv_cache_budget_bytes,
    })
}

fn engine_observability_json(snapshot: &ResidentEngineObservability) -> serde_json::Value {
    serde_json::json!({
        "model": {
            "family": snapshot.model.family.to_string(),
            "architecture": snapshot.model.architecture,
            "attention": format!("{:?}", snapshot.model.attention),
            "weight_source": format!("{:?}", snapshot.model.weight_source),
            "hidden_size": snapshot.model.hidden_size,
            "num_layers": snapshot.model.num_layers,
            "num_experts": snapshot.model.num_experts,
            "num_experts_per_tok": snapshot.model.num_experts_per_tok,
            "vocab_size": snapshot.model.vocab_size,
            "backend": snapshot.model.backend,
        },
        "driver": resident_driver_stats_json(&snapshot.driver),
        "prefix_cache": {
            "hits": snapshot.prefix_cache.hits,
            "misses": snapshot.prefix_cache.misses,
        },
        "kv_cache": snapshot.kv_cache.map(kv_cache_observability_json),
        "materialization": materialization_observability_json(&snapshot.materialization),
    })
}

fn resident_driver_stats_json(stats: &ResidentTopKDriverStats) -> serde_json::Value {
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

fn kv_cache_observability_json(kv: ResidentKvCacheObservability) -> serde_json::Value {
    serde_json::json!({
        "page_size_tokens": kv.page_size_tokens,
        "full_capacity_pages": kv.full_capacity_pages,
        "configured_pages": kv.configured_pages,
        "page_bytes": kv.page_bytes,
        "configured_bytes": kv.configured_bytes,
        "allocated_pages": kv.stats.allocated_pages,
        "free_pages": kv.stats.free_pages,
        "retiring_pages": kv.stats.retiring_pages,
        "shared_pages": kv.stats.shared_pages,
        "committed_tokens": kv.stats.committed_tokens,
        "capacity_tokens": kv.stats.capacity_tokens,
        "utilization": kv.stats.utilization,
        "fragmentation": kv.stats.fragmentation,
    })
}

fn materialization_observability_json(
    stats: &ResidentMaterializationObservability,
) -> serde_json::Value {
    serde_json::json!({
        "resolver": { "resolves": stats.resolver.resolves },
        "load_registry": {
            "operations_created": stats.registry.operations_created,
            "active_operations": stats.active_operations,
            "active_prefetches": stats.active_prefetches,
            "retired_operations": stats.registry.retirements,
            "single_flight_joins": stats.registry.single_flight_joins,
            "physical_completions": stats.registry.physical_completions,
            "rejected_completions": stats.registry.rejected_completions,
            "publications": stats.registry.publications,
            "cancellations_requested": stats.registry.cancellations_requested,
            "waiting_dependencies": stats.waiting_dependencies,
            "runnable_actions": stats.runnable_actions,
            "pending_completions": stats.pending_completions,
            "pending_physical_operations": stats.pending_physical_operations,
            "resident_entries": stats.resident_entries,
            "resident_bytes": stats.resident_bytes,
        },
        "active_stages": stats.stages.iter().map(|stage| serde_json::json!({
            "stage": load_stage_name(stage.stage),
            "active_operations": stage.active_operations,
        })).collect::<Vec<_>>(),
        "critical_path": critical_path_json(&stats.external_tokens),
    })
}

fn critical_path_json(samples: &[ResidentExternalTokenObservability]) -> serde_json::Value {
    let totals = samples.iter().fold(
        ResidentExternalTokenObservability::default(),
        |mut total, sample| {
            total.externally_committed_tokens = total
                .externally_committed_tokens
                .saturating_add(sample.externally_committed_tokens);
            total.read_ns = total.read_ns.saturating_add(sample.read_ns);
            total.upload_ns = total.upload_ns.saturating_add(sample.upload_ns);
            total.publish_ns = total.publish_ns.saturating_add(sample.publish_ns);
            total.wait_ns = total.wait_ns.saturating_add(sample.wait_ns);
            total.covered_wait_ns = total.covered_wait_ns.saturating_add(sample.covered_wait_ns);
            total.uncovered_wait_ns = total
                .uncovered_wait_ns
                .saturating_add(sample.uncovered_wait_ns);
            total.resume_ns = total.resume_ns.saturating_add(sample.resume_ns);
            total.commit_ns = total.commit_ns.saturating_add(sample.commit_ns);
            total
        },
    );
    serde_json::json!({
        "external_tokens": samples.len(),
        "externally_committed_tokens": totals.externally_committed_tokens,
        "read_ns": totals.read_ns,
        "upload_ns": totals.upload_ns,
        "publish_ns": totals.publish_ns,
        "wait_ns": totals.wait_ns,
        "covered_wait_ns": totals.covered_wait_ns,
        "uncovered_wait_ns": totals.uncovered_wait_ns,
        "resume_ns": totals.resume_ns,
        "commit_ns": totals.commit_ns,
        "per_external_token": samples.iter().map(|sample| serde_json::json!({
            "external_token_id": sample.external_token_id,
            "externally_committed_tokens": sample.externally_committed_tokens,
            "captured_at_ns": sample.captured_at_ns,
            "read_ns": sample.read_ns,
            "upload_ns": sample.upload_ns,
            "publish_ns": sample.publish_ns,
            "wait_ns": sample.wait_ns,
            "covered_wait_ns": sample.covered_wait_ns,
            "uncovered_wait_ns": sample.uncovered_wait_ns,
            "resume_ns": sample.resume_ns,
            "commit_ns": sample.commit_ns,
        })).collect::<Vec<_>>(),
    })
}

fn hard_resource_high_water_json(high_water: &[(ResourceKind, u64)]) -> serde_json::Value {
    serde_json::Value::Object(
        ResourceKind::ALL
            .into_iter()
            .map(|kind| {
                let value = high_water
                    .iter()
                    .find_map(|(candidate, value)| (*candidate == kind).then_some(*value))
                    .unwrap_or_default();
                (
                    resource_kind_name(kind).to_owned(),
                    serde_json::Value::from(value),
                )
            })
            .collect(),
    )
}

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
        "accepted_prefix_histogram": metrics.accepted_prefix_histogram,
        "total_proposal_s": seconds(metrics.total_proposal_time_us),
        "total_verify_s": seconds(metrics.total_verify_time_us),
        "total_transaction_s": seconds(metrics.total_transaction_time_us),
        "total_cycle_s": seconds(metrics.total_cycle_time_us),
        "mean_cycle_s": seconds(metrics.mean_cycle_time_us()),
    })
}

fn print_report(report: &InteractiveBenchReport) {
    println!("=== Interactive Benchmark ===");
    println!("model:             {}", report.model_dir);
    println!("model_adapter:     {}", report.build.model_name);
    println!("backend:           {}", report.build.backend.as_str());
    println!("backend_profile:   {}", report.build.backend_profile);
    println!("chat_template:     {}", report.build.chat_template.name());
    println!("runtime_path:      {}", report.runtime_path);
    println!("max_layers:        {}", report.build.max_layers);
    println!("ctx_size:          {}", report.build.ctx_size);
    println!("engine_build:      {:.3}s", seconds(report.engine_build_us));
    println!(
        "warmup:           {} requested / {} generated in {:.3}s",
        report.warmup_tokens,
        report.warmup_generated_tokens,
        seconds(report.warmup_us)
    );
    println!(
        "time_to_first_token: {:.3}s",
        seconds(report.time_to_first_token_us)
    );
    println!();

    for (index, turn) in report.turns.iter().enumerate() {
        println!(
            "Turn {}: {:?} ({} prompt tokens)",
            index + 1,
            turn.prompt_text,
            turn.prompt_tokens.len()
        );
        println!(
            "  ttft: {:.3}s  prefill: {:.3}s ({:.2} tok/s)  decode: {:.3}s ({:.2} tok/s)  pos: {}",
            seconds(turn.first_token_us),
            seconds(turn.prefill_us),
            turn.prompt_tokens.len() as f64 / seconds(turn.prefill_us).max(1e-6),
            seconds(turn.decode_us),
            turn.generated_tokens.len() as f64 / seconds(turn.decode_us).max(1e-6),
            turn.final_position
        );
        println!(
            "  generated: {:?}  finish: {}  eos: {}  stop_str: {:?}",
            turn.generated_tokens, turn.finish_reason, turn.stopped_by_eos, turn.stopped_by_string
        );
        println!(
            "  runtime: actions={} prefill_chunks={} prefill_tokens={} decode_steps={} emitted={}",
            turn.observability.driver.actions,
            turn.observability.driver.prefill_chunks,
            turn.observability.driver.prefill_tokens,
            turn.observability.driver.decode_steps,
            turn.observability.driver.emitted_tokens
        );
        if let Some(slowest) = turn.runtime_steps.iter().max_by_key(|step| step.elapsed_us) {
            println!(
                "  slowest_runtime_step: kind={} rows={} elapsed={:.3}s pos={}",
                slowest.action_kind,
                slowest.rows,
                seconds(slowest.elapsed_us),
                slowest.session_position
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
        report.observability.driver.actions,
        report.observability.driver.prefill_chunks,
        report.observability.driver.prefill_tokens,
        report.observability.driver.decode_steps,
        report.observability.driver.emitted_tokens
    );
    print_kv_summary(report.observability.kv_cache);
    print_materialization_summary(&report.observability.materialization);
    print_hard_resource_high_water(&report.observability.driver.hard_resource_high_water);
}

fn print_kv_summary(kv: Option<ResidentKvCacheObservability>) {
    let Some(kv) = kv else {
        println!("kv_cache:                  unavailable");
        return;
    };
    println!(
        "kv_cache:                  pages={}/{} full_capacity={} page_tokens={} page_bytes={:?} configured_bytes={:?} retiring={} shared={} committed_tokens={} utilization={:.3}",
        kv.stats.allocated_pages,
        kv.configured_pages,
        kv.full_capacity_pages,
        kv.page_size_tokens,
        kv.page_bytes,
        kv.configured_bytes,
        kv.stats.retiring_pages,
        kv.stats.shared_pages,
        kv.stats.committed_tokens,
        kv.stats.utilization,
    );
}

fn print_materialization_summary(stats: &ResidentMaterializationObservability) {
    println!(
        "runtime_materialization:   resolves={} created={} active={} retired={} joins={} completions={} resident={}/{}B waiting={}",
        stats.resolver.resolves,
        stats.registry.operations_created,
        stats.active_operations,
        stats.registry.retirements,
        stats.registry.single_flight_joins,
        stats.registry.physical_completions,
        stats.resident_entries,
        stats.resident_bytes,
        stats.waiting_dependencies,
    );
    let totals = stats.external_tokens.iter().fold(
        (0u64, 0u64, 0u64, 0u64),
        |(read, upload, wait, uncovered), sample| {
            (
                read.saturating_add(sample.read_ns),
                upload.saturating_add(sample.upload_ns),
                wait.saturating_add(sample.wait_ns),
                uncovered.saturating_add(sample.uncovered_wait_ns),
            )
        },
    );
    println!(
        "critical_path:             read={}ns upload={}ns wait={}ns uncovered={}ns external_tokens={}",
        totals.0,
        totals.1,
        totals.2,
        totals.3,
        stats.external_tokens.len(),
    );
}

fn print_hard_resource_high_water(high_water: &[(ResourceKind, u64)]) {
    let values = ResourceKind::ALL
        .into_iter()
        .map(|kind| {
            let value = high_water
                .iter()
                .find_map(|(candidate, value)| (*candidate == kind).then_some(*value))
                .unwrap_or_default();
            format!("{}={value}", resource_kind_name(kind))
        })
        .collect::<Vec<_>>()
        .join(" ");
    println!("hard_resource_high_water:  {values}");
}

fn load_stage_name(stage: LoadStage) -> &'static str {
    match stage {
        LoadStage::Reserved => "reserved",
        LoadStage::ReadSubmitted => "read_submitted",
        LoadStage::HostReady => "host_ready",
        LoadStage::UploadSubmitted => "upload_submitted",
        LoadStage::Installing => "installing",
        LoadStage::Resident => "resident",
        LoadStage::Draining => "draining",
        LoadStage::Retired => "retired",
        LoadStage::Failed => "failed",
        LoadStage::Stale => "stale",
    }
}

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

fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

fn seconds(microseconds: u64) -> f64 {
    microseconds as f64 / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use ferrule_model::{
        AttentionKind, ModelExecutionBackend, ModelFamily, ModelInfo, WeightSource,
    };
    use ferrule_runtime::{ResidentMaterializationStageObservability, ResidentPrefixCacheStats};

    use super::*;

    fn model_info() -> ModelInfo {
        ModelInfo {
            family: ModelFamily::QwenMoe,
            architecture: Some("qwen3_moe".to_owned()),
            attention: AttentionKind::GroupedQuery,
            weight_source: WeightSource::Safetensors,
            hidden_size: 64,
            num_layers: 2,
            num_experts: 8,
            num_experts_per_tok: 2,
            vocab_size: 128,
            backend: "cpu",
        }
    }

    fn engine_observability() -> ResidentEngineObservability {
        ResidentEngineObservability {
            model: model_info(),
            driver: ResidentTopKDriverStats {
                hard_resource_high_water: ResourceKind::ALL
                    .into_iter()
                    .map(|kind| (kind, 1))
                    .collect(),
                ..Default::default()
            },
            prefix_cache: ResidentPrefixCacheStats::default(),
            kv_cache: None,
            materialization: ResidentMaterializationObservability {
                stages: vec![ResidentMaterializationStageObservability {
                    stage: LoadStage::Resident,
                    active_operations: 2,
                }],
                external_tokens: vec![ResidentExternalTokenObservability {
                    external_token_id: 1,
                    read_ns: 3,
                    ..Default::default()
                }],
                ..Default::default()
            },
        }
    }

    #[test]
    fn schema_is_model_neutral_and_uses_runtime_observability() {
        let report = InteractiveBenchReport {
            model_dir: "model".to_owned(),
            build: ResidentModelBuildObservability {
                model_name: "qwen3-moe",
                backend: ModelExecutionBackend::Cpu,
                backend_profile: "cpu-standard-decoder",
                chat_template: ChatTemplate::Qwen3,
                max_layers: 2,
                ctx_size: 64,
                max_active_sequences: 1,
                kv_cache_budget_bytes: None,
            },
            output_head_chunk_rows: 4096,
            runtime_path: "resident_model_planner",
            engine_build_us: 0,
            warmup_tokens: 0,
            warmup_us: 0,
            warmup_generated_tokens: 0,
            time_to_first_token_us: 0,
            turns: Vec::new(),
            aggregate_prefill_tok_per_s: 0.0,
            aggregate_decode_tok_per_s: 0.0,
            total_prompt_tokens: 0,
            total_prefill_us: 0,
            total_generated: 0,
            final_position: 0,
            observability: engine_observability(),
        };

        let json = interactive_bench_report_json(&report);
        assert_eq!(json["schema_version"], CLI_RUNTIME_SCHEMA_VERSION);
        assert_eq!(json["build"]["backend"], "cpu-reference");
        assert_eq!(json["runtime"]["model"]["family"], "Qwen-MoE");
        assert_eq!(
            json["runtime"]["materialization"]["critical_path"]["read_ns"],
            3
        );
        assert_eq!(
            json["runtime"]["driver"]["hard_resource_high_water"]
                .as_object()
                .unwrap()
                .len(),
            ResourceKind::ALL.len()
        );

        assert!(json.get("runtime").is_some());
        assert!(json.get("build").is_some());
    }

    #[test]
    fn critical_path_aggregates_runtime_samples() {
        let samples = [
            ResidentExternalTokenObservability {
                external_token_id: 1,
                read_ns: 3,
                wait_ns: 5,
                ..Default::default()
            },
            ResidentExternalTokenObservability {
                external_token_id: 2,
                read_ns: 7,
                wait_ns: 11,
                ..Default::default()
            },
        ];

        let json = critical_path_json(&samples);
        assert_eq!(json["external_tokens"], 2);
        assert_eq!(json["read_ns"], 10);
        assert_eq!(json["wait_ns"], 16);
        assert_eq!(json["per_external_token"].as_array().unwrap().len(), 2);
    }
}
