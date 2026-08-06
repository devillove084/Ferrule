#[cfg(feature = "cuda")]
use std::{
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

#[cfg(feature = "cuda")]
use crate::commands::bench_interactive::{
    CLI_RUNTIME_SCHEMA_VERSION, RuntimeMaterializationStats, dsv4_operator_counters_delta,
    dsv4_operator_counters_json, print_hard_resource_high_water,
    print_runtime_materialization_summary, resident_driver_stats_delta, resident_driver_stats_json,
    resident_driver_stats_snapshot, runtime_materialization_snapshot,
    runtime_materialization_stats_delta, runtime_materialization_stats_json,
};
#[cfg(feature = "cuda")]
use ferrule_model::{
    ChatTemplate, ModelExecutionBackend,
    models::deepseek_v4::{DeepSeekV4PrepareOptions, DeepSeekV4Runner},
};
#[cfg(feature = "cuda")]
use ferrule_runtime::{
    GenerateRequest, LocalResidentInferenceEngine, RequestId, ResidentActionKind,
    ResidentDriverStep, SessionId,
};

#[cfg(feature = "cuda")]
use crate::commands::resident::{
    block_on_local_inference, build_resident_topk_driver, require_finished_request,
    resident_driver_config, single_sequence_scheduler_config,
};

#[cfg(feature = "cuda")]
use super::stats::print_deepseek_v4_runtime_stats;

// ── deepseek-v4-probe / generate ─────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn cmd_deepseek_v4_generate(
    model_dir: &str,
    prompt: &str,
    max_new_tokens: usize,
    max_layers: usize,
    output_head_chunk_rows: usize,
    max_tensor_mb: u64,
    expert_reader_max_slice_mb: u64,
    stop_at_eos: bool,
    enable_native_proposals: bool,
    verbose_tokens: bool,
    chat_prompt: bool,
    json: bool,
    warmup_tokens: usize,
    moe_hotset_experts: usize,
) -> anyhow::Result<()> {
    block_on_local_inference(cmd_deepseek_v4_generate_async(
        model_dir,
        prompt,
        max_new_tokens,
        max_layers,
        output_head_chunk_rows,
        max_tensor_mb,
        expert_reader_max_slice_mb,
        stop_at_eos,
        enable_native_proposals,
        verbose_tokens,
        chat_prompt,
        json,
        warmup_tokens,
        moe_hotset_experts,
    ))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
async fn cmd_deepseek_v4_generate_async(
    model_dir: &str,
    prompt: &str,
    max_new_tokens: usize,
    max_layers: usize,
    output_head_chunk_rows: usize,
    max_tensor_mb: u64,
    expert_reader_max_slice_mb: u64,
    stop_at_eos: bool,
    enable_native_proposals: bool,
    verbose_tokens: bool,
    chat_prompt: bool,
    json: bool,
    warmup_tokens: usize,
    moe_hotset_experts: usize,
) -> anyhow::Result<()> {
    let model_path = Path::new(model_dir);
    let options = DeepSeekV4PrepareOptions {
        max_layers,
        output_head_chunk_rows,
        expert_reader_max_tensor_bytes: expert_reader_max_slice_mb.saturating_mul(1024 * 1024),
        moe_hotset_experts,
        ..DeepSeekV4PrepareOptions::default()
    };
    let load_start = Instant::now();
    let runner = DeepSeekV4Runner::load_hf_with_options_and_backend(
        model_path,
        max_tensor_mb.saturating_mul(1024 * 1024),
        options,
        ModelExecutionBackend::Cuda,
    )?;
    let load_elapsed = load_start.elapsed();

    let encoded_prompt = if chat_prompt {
        ChatTemplate::DeepSeekV4.format_turn(prompt, true)
    } else {
        prompt.to_string()
    };
    let prompt_tokens = runner.model().tokenizer.encode(&encoded_prompt)?;
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
        println!("warmup:    {warmup_tokens}");

        println!("load:       {:.3} ms", load_elapsed.as_secs_f64() * 1000.0);
        println!("--- output ---");
    }

    let scheduler_config = single_sequence_scheduler_config(prompt_tokens.len());
    let build_driver = |runner: DeepSeekV4Runner, ctx_size: usize| {
        let schema = runner.kv_layout_schema().clone();
        build_resident_topk_driver(
            runner,
            Box::new(schema),
            scheduler_config,
            // Preserve this command's historical EOS behavior while allowing
            // target-only decode as an explicit correctness oracle.
            ferrule_runtime::ResidentTopKDriverConfig {
                enable_native_proposals,
                ..resident_driver_config(ctx_size, stop_at_eos)
            },
        )
    };

    // Warmup and measurement share one completion owner because model completion
    // reactors are transferred exactly once. The dedicated warmup session is
    // drained before measurement, and counter baselines exclude its work.
    let runtime_ctx = prompt_tokens
        .len()
        .saturating_add(max_new_tokens.max(warmup_tokens))
        .max(1);
    let mut driver = LocalResidentInferenceEngine::new(build_driver(runner, runtime_ctx)?);
    driver.initialize().await?;
    driver.wait_for_model_warmup().await?;
    if max_new_tokens > 0 && warmup_tokens > 0 {
        driver.submit(GenerateRequest {
            id: RequestId(0),
            session_id: Some(SessionId(u64::MAX)),
            prompt_tokens: prompt_tokens.clone(),
            max_new_tokens: warmup_tokens,
            stop: Vec::new(),
            ignore_eos: !stop_at_eos,
        });
        let warmup_start = Instant::now();
        loop {
            let step = driver.step(&mut |_| Ok(())).await?;
            if let Some(terminal) = driver.take_request_terminal(RequestId(0)) {
                let _ = require_finished_request(terminal, "resident warmup")?;
                break;
            }
            match step {
                ResidentDriverStep::Idle => {
                    anyhow::bail!("resident warmup became idle before request terminalization")
                }
                ResidentDriverStep::Blocked => {
                    anyhow::bail!("resident warmup remained blocked after a completion wake")
                }
                ResidentDriverStep::WaitingForModelProgress(_)
                | ResidentDriverStep::Executed { .. } => {}
            }
        }
        if !json {
            eprintln!(
                "[warmup] {warmup_tokens} tokens in {:.3}s",
                warmup_start.elapsed().as_secs_f64()
            );
        }
    }

    let measured_observability_baseline = driver.model_observability_snapshot();
    let measured_driver_stats_baseline = resident_driver_stats_snapshot(&driver);
    let measured_materialization_baseline =
        runtime_materialization_snapshot(&driver, measured_observability_baseline.operator)?;
    let mut generated = Vec::new();
    let mut final_position = measured_observability_baseline.position;
    let mut prefill_elapsed = Duration::ZERO;
    let mut decode_elapsed = Duration::ZERO;

    if max_new_tokens > 0 {
        driver.submit(GenerateRequest {
            id: RequestId(1),
            session_id: Some(SessionId(1)),
            prompt_tokens: prompt_tokens.clone(),
            max_new_tokens,
            stop: Vec::new(),
            ignore_eos: !stop_at_eos,
        });

        let sequence = loop {
            let step_start = Instant::now();
            let step = driver
                .step(&mut |event| {
                    if verbose_tokens {
                        eprintln!(
                            "[{}] token={} logit={:.6}",
                            event.index,
                            event.token,
                            event.logit.unwrap_or(f32::NAN)
                        );
                    }
                    if !json {
                        print!("{}", event.text);
                        std::io::stdout()
                            .flush()
                            .map_err(ferrule_common::Error::from)?;
                    }
                    generated.push(event.token);
                    Ok(())
                })
                .await?;
            let step_elapsed = step_start.elapsed();
            if let Some(terminal) = driver.take_request_terminal(RequestId(1)) {
                break require_finished_request(terminal, "DeepSeek-V4 generation")?;
            }
            match step {
                ResidentDriverStep::Executed { action_kind, .. } => match action_kind {
                    ResidentActionKind::Prefill | ResidentActionKind::Mixed => {
                        prefill_elapsed += step_elapsed;
                    }
                    ResidentActionKind::Decode => decode_elapsed += step_elapsed,
                    ResidentActionKind::Finish | ResidentActionKind::Cancel => {}
                },
                ResidentDriverStep::WaitingForModelProgress(_) => decode_elapsed += step_elapsed,
                ResidentDriverStep::Idle => {
                    anyhow::bail!("resident runtime became idle before request terminalization")
                }
                ResidentDriverStep::Blocked => {
                    anyhow::bail!("resident runtime driver blocked during DSV4 generation")
                }
            }
        };

        final_position = sequence.position;
    }

    let elapsed = prefill_elapsed + decode_elapsed;
    let model_info = driver.model_info();
    let bound_layer_count = driver.bound_layer_count().unwrap_or_default();
    let observability = driver.model_observability_snapshot();
    let operator_counters = dsv4_operator_counters_delta(
        measured_observability_baseline.operator,
        observability.operator,
    );
    let runtime_driver_stats = resident_driver_stats_delta(
        measured_driver_stats_baseline,
        resident_driver_stats_snapshot(&driver),
    );
    let runtime_materialization_after =
        runtime_materialization_snapshot(&driver, observability.operator)?;
    let runtime_materialization = runtime_materialization_stats_delta(
        measured_materialization_baseline,
        runtime_materialization_after,
        runtime_driver_stats.emitted_tokens,
    );
    if json {
        let layer_stats = observability.layer_runtime;
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
        let mut out = generate_runtime_schema_json(
            &runtime_driver_stats,
            &runtime_materialization,
            &operator_counters,
        );
        out["model"] = serde_json::json!(model_dir);
        out["backend"] = serde_json::json!(model_info.backend);
        out["prompt"] = serde_json::json!(prompt);
        out["prompt_tokens"] = serde_json::json!(prompt_tokens.len());
        out["prompt_token_ids"] = serde_json::json!(prompt_tokens);
        out["generated_tokens"] = serde_json::json!(generated.len());
        out["generated_token_ids"] = serde_json::json!(generated);
        out["max_layers"] = serde_json::json!(max_layers);
        out["warmup_tokens"] = serde_json::json!(warmup_tokens);
        out["bound_layers"] = serde_json::json!(bound_layer_count);
        out["position"] = serde_json::json!(final_position);
        out["layers"] = serde_json::json!(layers);
        let load_profile = observability.load;
        out["timing"] = serde_json::json!({
            "load_seconds": load_elapsed.as_secs_f64(),
            "load_profile": {
                "checkpoint_seconds": load_profile.checkpoint_us as f64 / 1_000_000.0,
                "prepare": {
                    "validation_seconds": load_profile.prepare.validation_us as f64 / 1_000_000.0,
                    "attachment_bind_seconds": load_profile.prepare.attachment_bind_us as f64 / 1_000_000.0,
                    "target_bind_seconds": load_profile.prepare.target_bind_us as f64 / 1_000_000.0,
                    "execution_plan_seconds": load_profile.prepare.execution_plan_us as f64 / 1_000_000.0,
                    "manifest_seconds": load_profile.prepare.manifest_us as f64 / 1_000_000.0,
                    "total_seconds": load_profile.prepare.total_us as f64 / 1_000_000.0,
                },
                "cuda_context_seconds": load_profile.cuda_context_us as f64 / 1_000_000.0,
                "sequence_state_seconds": load_profile.sequence_state_us as f64 / 1_000_000.0,
                "reader_seconds": load_profile.reader_us as f64 / 1_000_000.0,
                "static_image": {
                    "globals_seconds": load_profile.static_image_globals_us as f64 / 1_000_000.0,
                    "embedding_seconds": load_profile.static_image_embedding_us as f64 / 1_000_000.0,
                    "output_head_seconds": load_profile.static_image_output_head_us as f64 / 1_000_000.0,
                    "target_layers_seconds": load_profile.static_image_target_layers_us as f64 / 1_000_000.0,
                    "attachment_seconds": load_profile.static_image_attachment_us as f64 / 1_000_000.0,
                    "total_seconds": load_profile.static_image_total_us as f64 / 1_000_000.0,
                },
                "residency_seconds": load_profile.residency_us as f64 / 1_000_000.0,
                "total_seconds": load_profile.total_us as f64 / 1_000_000.0,
            },
            "prefill_seconds": prefill_elapsed.as_secs_f64(),
            "decode_seconds": decode_elapsed.as_secs_f64(),
            "total_seconds": elapsed.as_secs_f64(),
            "prefill_tokens_per_second": duration_rate(prompt_tokens.len(), prefill_elapsed),
            "decode_tokens_per_second": duration_rate(generated.len(), decode_elapsed),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!();
        println!("--- stats ---");
        println!("generated_tokens: {:?}", generated);
        println!("position:   {final_position}");
        println!("bound layers: {bound_layer_count}");
        print_deepseek_v4_runtime_stats(&observability.layer_runtime);
        print_runtime_materialization_summary(&runtime_materialization);
        print_hard_resource_high_water(&runtime_driver_stats.hard_resource_high_water);
        println!("run:        {:.3} ms", elapsed.as_secs_f64() * 1000.0);
    }
    driver.shutdown().await?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn generate_runtime_schema_json(
    runtime_driver_stats: &ferrule_runtime::ResidentTopKDriverStats,
    runtime_materialization: &RuntimeMaterializationStats,
    operator_counters: &ferrule_model::models::deepseek_v4::DeepSeekV4OperatorRuntimeCounters,
) -> serde_json::Value {
    serde_json::json!({
        "schema_version": CLI_RUNTIME_SCHEMA_VERSION,
        "runtime_driver_stats": resident_driver_stats_json(runtime_driver_stats),
        "runtime_materialization": runtime_materialization_stats_json(runtime_materialization),
        "dsv4_operator_counters": dsv4_operator_counters_json(operator_counters),
    })
}

#[cfg(feature = "cuda")]
fn duration_rate(tokens: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        0.0
    } else {
        tokens as f64 / elapsed.as_secs_f64()
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use ferrule_runtime::ResourceKind;

    #[test]
    fn generate_v4_runtime_schema_matches_interactive_report_and_has_no_legacy_keys() {
        let runtime_driver_stats = ferrule_runtime::ResidentTopKDriverStats {
            hard_resource_high_water: ResourceKind::ALL
                .into_iter()
                .map(|kind| (kind, 1))
                .collect(),
            ..Default::default()
        };
        let json = generate_runtime_schema_json(
            &runtime_driver_stats,
            &RuntimeMaterializationStats::default(),
            &ferrule_model::models::deepseek_v4::DeepSeekV4OperatorRuntimeCounters::default(),
        );

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
                "legacy key fragment remained in generate v3 schema: {legacy}"
            );
        }
    }

    #[test]
    fn duration_rate_handles_empty_and_measured_intervals() {
        assert_eq!(duration_rate(4, Duration::ZERO), 0.0);
        assert_eq!(duration_rate(4, Duration::from_secs(2)), 2.0);
    }
}

#[cfg(not(feature = "cuda"))]
#[expect(
    clippy::too_many_arguments,
    reason = "the non-CUDA stub mirrors the CUDA command interface"
)]
pub fn cmd_deepseek_v4_generate(
    _model_dir: &str,
    _prompt: &str,
    _max_new_tokens: usize,
    _max_layers: usize,
    _output_head_chunk_rows: usize,
    _max_tensor_mb: u64,
    _expert_reader_max_slice_mb: u64,
    _stop_at_eos: bool,
    _enable_native_proposals: bool,
    _verbose_tokens: bool,
    _chat_prompt: bool,
    _json: bool,
    _warmup_tokens: usize,
    _moe_hotset_experts: usize,
) -> anyhow::Result<()> {
    anyhow::bail!("deepseek-v4-generate requires --features cuda")
}
