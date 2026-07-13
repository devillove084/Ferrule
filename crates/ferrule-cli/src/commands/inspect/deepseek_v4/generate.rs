use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::bench::{RuntimeBenchSummary, RuntimeCounters};
use ferrule_model::{
    ChatTemplate, ModelExecutionBackend, ModelRunner,
    models::deepseek_v4::{DeepSeekV4PrepareOptions, DeepSeekV4Runner},
};
use ferrule_runtime::{
    GenerateRequest, RequestId, ResidentActionKind, ResidentDriverStep, ResidentSchedulerConfig,
    ResidentTopKDriverConfig, SamplingConfig, SessionId,
};

use crate::commands::resident::build_resident_topk_driver;

use super::stats::print_deepseek_v4_runtime_stats;

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
    warmup_tokens: usize,
    moe_prefetch_experts: usize,
    moe_hotset_experts: usize,
) -> anyhow::Result<()> {
    let model_path = Path::new(model_dir);
    let options = DeepSeekV4PrepareOptions {
        max_layers,
        output_head_chunk_rows,
        expert_reader_max_tensor_bytes: expert_reader_max_slice_mb.saturating_mul(1024 * 1024),
        moe_prefetch_experts,
        moe_hotset_experts,
        ..DeepSeekV4PrepareOptions::default()
    };
    let operator_backend = ModelExecutionBackend::parse(backend)?;
    let load_start = Instant::now();
    let mut runner = DeepSeekV4Runner::load_hf_with_options_and_backend(
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
        println!("prefetch:  {moe_prefetch_experts} hot experts/layer");
        println!("hotset:    {moe_hotset_experts} resident experts/layer (0 = managed default)");
        println!("load:       {:.3} ms", load_elapsed.as_secs_f64() * 1000.0);
        println!("--- output ---");
    }

    let scheduler_config = ResidentSchedulerConfig {
        prefill_chunk_size: prompt_tokens.len().max(1),
        max_active_sequences: 1,
        max_decode_batch: 1,
        ..Default::default()
    };
    let build_driver = |runner: DeepSeekV4Runner, ctx_size: usize, token_budget: usize| {
        let schema = runner.kv_layout_schema().clone();
        build_resident_topk_driver(
            runner,
            Box::new(schema),
            scheduler_config,
            ResidentTopKDriverConfig {
                ctx_size,
                stop_at_eos,
                // Preserve this command's historical EOS behavior.
                append_eos_to_session: false,
                max_steps_per_run: ctx_size.saturating_add(token_budget).saturating_add(64),
            },
        )
    };

    // Warmup uses its own explicit session and driver. Rebuilding the driver keeps
    // warmed expert residency while preventing warmup KV/session state and driver
    // counters from entering the measured run.
    if max_new_tokens > 0 && warmup_tokens > 0 {
        let warmup_ctx = prompt_tokens.len().saturating_add(warmup_tokens).max(1);
        let mut warmup_driver = build_driver(runner, warmup_ctx, warmup_tokens)?;
        warmup_driver.submit(GenerateRequest {
            id: RequestId(0),
            session_id: Some(SessionId(u64::MAX)),
            prompt_tokens: prompt_tokens.clone(),
            sampling: SamplingConfig::greedy(),
            max_new_tokens: warmup_tokens,
            stop: Vec::new(),
            ignore_eos: !stop_at_eos,
        });
        let warmup_start = Instant::now();
        warmup_driver.run_until_blocked(|_| Ok(()))?;
        let _ = warmup_driver.drain_finished();
        if !json {
            eprintln!(
                "[warmup] {warmup_tokens} tokens in {:.3}s",
                warmup_start.elapsed().as_secs_f64()
            );
        }
        runner = warmup_driver.into_runner()?;
        runner.reset_session()?;
    }

    let measured_ctx = prompt_tokens.len().saturating_add(max_new_tokens).max(1);
    let mut driver = build_driver(runner, measured_ctx, max_new_tokens)?;
    let mut generated = Vec::new();
    let mut final_position = driver.executor().runner().position();
    let mut prefill_elapsed = Duration::ZERO;
    let mut decode_elapsed = Duration::ZERO;
    let mut decode_counters_baseline: Option<
        ferrule_model::models::deepseek_v4::DeepSeekV4OperatorRuntimeCounters,
    > = None;

    if max_new_tokens > 0 {
        driver.submit(GenerateRequest {
            id: RequestId(1),
            session_id: Some(SessionId(1)),
            prompt_tokens: prompt_tokens.clone(),
            sampling: SamplingConfig::greedy(),
            max_new_tokens,
            stop: Vec::new(),
            ignore_eos: !stop_at_eos,
        });

        loop {
            let step_start = Instant::now();
            let step = driver.step(&mut |event| {
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
                    std::io::stdout().flush()?;
                }
                generated.push(event.token);
                Ok(())
            })?;
            let step_elapsed = step_start.elapsed();
            match step {
                ResidentDriverStep::Executed { action_kind, .. } => match action_kind {
                    ResidentActionKind::Prefill | ResidentActionKind::Mixed => {
                        prefill_elapsed += step_elapsed;
                        decode_counters_baseline =
                            Some(driver.executor().runner().operator_runtime_counters());
                    }
                    ResidentActionKind::Decode => decode_elapsed += step_elapsed,
                    ResidentActionKind::Finish | ResidentActionKind::Cancel => {}
                },
                ResidentDriverStep::Idle => break,
                ResidentDriverStep::Blocked => {
                    anyhow::bail!("resident runtime driver blocked during DSV4 generation")
                }
            }
        }

        let finished = driver.drain_finished();
        let sequence = finished.last().ok_or_else(|| {
            anyhow::anyhow!("resident runtime driver produced no finished sequence")
        })?;
        final_position = sequence.position;
    }

    let elapsed = prefill_elapsed + decode_elapsed;
    let runner = driver.executor().runner();
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
        // If warmup ran, subtract warmup baseline so counters reflect
        // only the timed decode phase.
        let (expert_loads, expert_load_bytes, expert_evictions, expert_selected) =
            match &decode_counters_baseline {
                Some(base) => (
                    op_counters.expert_loads.saturating_sub(base.expert_loads),
                    op_counters
                        .expert_load_bytes
                        .saturating_sub(base.expert_load_bytes),
                    op_counters
                        .expert_evictions
                        .saturating_sub(base.expert_evictions),
                    op_counters
                        .expert_selected
                        .saturating_sub(base.expert_selected),
                ),
                None => (
                    op_counters.expert_loads,
                    op_counters.expert_load_bytes,
                    op_counters.expert_evictions,
                    op_counters.expert_selected,
                ),
            };
        let (kernel_launches, h2d_copies, h2d_bytes, d2h_copies, d2h_bytes, uploads, upload_bytes) =
            match &decode_counters_baseline {
                Some(base) => (
                    op_counters
                        .kernel_launches
                        .saturating_sub(base.kernel_launches),
                    op_counters
                        .host_to_device_copies
                        .saturating_sub(base.host_to_device_copies),
                    op_counters
                        .host_to_device_bytes
                        .saturating_sub(base.host_to_device_bytes),
                    op_counters
                        .device_to_host_copies
                        .saturating_sub(base.device_to_host_copies),
                    op_counters
                        .device_to_host_bytes
                        .saturating_sub(base.device_to_host_bytes),
                    op_counters
                        .artifact_uploads
                        .saturating_sub(base.artifact_uploads),
                    op_counters
                        .artifact_upload_bytes
                        .saturating_sub(base.artifact_upload_bytes),
                ),
                None => (
                    op_counters.kernel_launches,
                    op_counters.host_to_device_copies,
                    op_counters.host_to_device_bytes,
                    op_counters.device_to_host_copies,
                    op_counters.device_to_host_bytes,
                    op_counters.artifact_uploads,
                    op_counters.artifact_upload_bytes,
                ),
            };
        let (
            moe_calls,
            moe_tc_calls,
            moe_scalar_calls,
            moe_reduce_calls,
            moe_total_us,
            moe_input_prepare_us,
            moe_gate_up_us,
            moe_swiglu_us,
            moe_hidden_pack_us,
            moe_down_us,
        ) = match &decode_counters_baseline {
            Some(base) => (
                op_counters.moe_calls.saturating_sub(base.moe_calls),
                op_counters.moe_tc_calls.saturating_sub(base.moe_tc_calls),
                op_counters
                    .moe_scalar_calls
                    .saturating_sub(base.moe_scalar_calls),
                op_counters
                    .moe_reduce_calls
                    .saturating_sub(base.moe_reduce_calls),
                op_counters.moe_total_us.saturating_sub(base.moe_total_us),
                op_counters
                    .moe_input_prepare_us
                    .saturating_sub(base.moe_input_prepare_us),
                op_counters
                    .moe_gate_up_us
                    .saturating_sub(base.moe_gate_up_us),
                op_counters.moe_swiglu_us.saturating_sub(base.moe_swiglu_us),
                op_counters
                    .moe_hidden_pack_us
                    .saturating_sub(base.moe_hidden_pack_us),
                op_counters.moe_down_us.saturating_sub(base.moe_down_us),
            ),
            None => (
                op_counters.moe_calls,
                op_counters.moe_tc_calls,
                op_counters.moe_scalar_calls,
                op_counters.moe_reduce_calls,
                op_counters.moe_total_us,
                op_counters.moe_input_prepare_us,
                op_counters.moe_gate_up_us,
                op_counters.moe_swiglu_us,
                op_counters.moe_hidden_pack_us,
                op_counters.moe_down_us,
            ),
        };
        let mut counters = RuntimeCounters::default();
        counters.record_load(load_elapsed);
        counters.record_prefill(prefill_elapsed);
        counters.record_decode(decode_elapsed);
        counters.timing.moe_calls = moe_calls;
        counters.timing.moe_tc_calls = moe_tc_calls;
        counters.timing.moe_scalar_calls = moe_scalar_calls;
        counters.timing.moe_reduce_calls = moe_reduce_calls;
        counters.timing.moe_total_us = moe_total_us;
        counters.timing.moe_input_prepare_us = moe_input_prepare_us;
        counters.timing.moe_gate_up_us = moe_gate_up_us;
        counters.timing.moe_swiglu_us = moe_swiglu_us;
        counters.timing.moe_hidden_pack_us = moe_hidden_pack_us;
        counters.timing.moe_down_us = moe_down_us;
        counters.record_kernel_launches(kernel_launches);
        counters.transfers.host_to_device_copies = h2d_copies;
        counters.transfers.host_to_device_bytes = h2d_bytes;
        counters.transfers.device_to_host_copies = d2h_copies;
        counters.transfers.device_to_host_bytes = d2h_bytes;
        counters.record_artifact_uploads(uploads, upload_bytes);
        counters.record_selected_experts(expert_selected);
        counters.record_expert_loads(expert_loads, expert_load_bytes);
        counters.record_expert_evictions(expert_evictions);
        let (host_cache_hits, host_cache_misses, host_cache_evictions) =
            match &decode_counters_baseline {
                Some(base) => (
                    op_counters
                        .expert_host_cache_hits
                        .saturating_sub(base.expert_host_cache_hits),
                    op_counters
                        .expert_host_cache_misses
                        .saturating_sub(base.expert_host_cache_misses),
                    op_counters
                        .expert_host_cache_evictions
                        .saturating_sub(base.expert_host_cache_evictions),
                ),
                None => (
                    op_counters.expert_host_cache_hits,
                    op_counters.expert_host_cache_misses,
                    op_counters.expert_host_cache_evictions,
                ),
            };
        counters.set_expert_host_cache(
            host_cache_hits,
            host_cache_misses,
            host_cache_evictions,
            op_counters.expert_host_cache_entries,
            op_counters.expert_host_cache_bytes,
        );
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
            "warmup_tokens": warmup_tokens,
            "moe_prefetch_experts": moe_prefetch_experts,
            "moe_hotset_experts": moe_hotset_experts,
            "bound_layers": runner.bound_layer_count(),
            "position": final_position,
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
        println!("position:   {final_position}");
        println!("bound layers: {}", runner.bound_layer_count());
        print_deepseek_v4_runtime_stats(&runner);
        println!("run:        {:.3} ms", elapsed.as_secs_f64() * 1000.0);
    }
    Ok(())
}
