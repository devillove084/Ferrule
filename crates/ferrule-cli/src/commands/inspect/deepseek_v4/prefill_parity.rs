//! `deepseek-v4-prefill-parity` command.
//!
//! Compares the batched/device prefill path against the token-loop append
//! path, reporting the first diverging layer and the top-1 token at several
//! layer-depth cut points (1L, 5L, 23L, 43L by default).

use std::path::Path;
use std::time::Instant;

use ferrule_common::execution::ForwardPhase;
use ferrule_common::{Error, Result as FerruleResult};
#[cfg(all(feature = "cuda", feature = "cutlass"))]
use ferrule_model::models::deepseek_v4::DeepSeekV4MtpModel;
use ferrule_model::{
    ChatTemplate, ModelExecutionBackend,
    models::deepseek_v4::{
        DeepSeekV4DsparkBackboneDebugSnapshot, DeepSeekV4PrepareOptions, DeepSeekV4Runner,
    },
};

use crate::commands::resident::build_page_managed_diagnostic_harness;

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
    let options = DeepSeekV4PrepareOptions {
        max_layers,
        output_head_chunk_rows: 4096,
        expert_reader_max_tensor_bytes: expert_reader_max_slice_mb.saturating_mul(1024 * 1024),
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };
    let operator_backend = ModelExecutionBackend::parse(backend)?;

    let encoded_prompt = if chat_prompt {
        ChatTemplate::DeepSeekV4.format_turn(prompt, true)
    } else {
        prompt.to_string()
    };

    let load_start = Instant::now();
    let runner = DeepSeekV4Runner::load_hf_with_options_and_backend(
        model_path,
        max_tensor_mb.saturating_mul(1024 * 1024),
        options,
        operator_backend,
    )?;
    let load_elapsed = load_start.elapsed();

    let token_ids = runner.model().tokenizer.encode(&encoded_prompt)?;
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

    let schema = runner.kv_layout_schema().clone();
    let mut diagnostic =
        build_page_managed_diagnostic_harness(runner, Box::new(schema), token_ids.len(), 2)?;
    let batched_sequence = diagnostic.create_sequence(0)?;
    let token_loop_sequence = diagnostic.create_sequence(0)?;

    // ── Batched trace ───────────────────────────────────────────────────
    let batched_start = Instant::now();
    let (batched_trace, dspark_context_kv_lengths) = diagnostic.execute_sequence_step(
        batched_sequence,
        ForwardPhase::Prefill,
        &token_ids,
        |runner| {
            let trace = runner.prefill_batched_layer_hc_trace(&token_ids)?;
            Ok((trace, runner.dspark_context_kv_lengths()))
        },
    )?;
    let batched_checkpoints = diagnostic.runner_mut().take_parity_checkpoints();
    let batched_elapsed = batched_start.elapsed();
    let dspark_main_report = {
        let runner = diagnostic.runner_mut();
        match runner.dspark_main_debug_snapshot(token_ids.len())? {
            Some((target_taps, device_main_x)) => {
                let input_size = target_taps.len() / token_ids.len();
                let output_size = device_main_x.len() / token_ids.len();
                let last_taps = &target_taps[(token_ids.len() - 1) * input_size..];
                let reference = runner
                    .mtp()
                    .ok_or_else(|| anyhow::anyhow!("DSpark snapshot exists without MTP resources"))?
                    .stage_zero_main_reference(last_taps, 1)?;
                let last_device = &device_main_x[(token_ids.len() - 1) * output_size..];
                Some(DsparkMainReport {
                    target_taps_len: target_taps.len(),
                    device_main_x_len: device_main_x.len(),
                    context_kv_lengths: dspark_context_kv_lengths.clone(),
                    compared_rows: 1,
                    max_abs_diff: max_abs_difference(last_device, &reference),
                })
            }
            None => None,
        }
    };

    if !json {
        println!(
            "batched:    {:.3} ms, {} layers captured",
            batched_elapsed.as_secs_f64() * 1000.0,
            batched_trace.len()
        );
        if let Some(report) = dspark_main_report.as_ref() {
            println!(
                "dspark main: taps={} main_x={} context_kv={:?} compared_rows={} max_abs_diff={:.6e}",
                report.target_taps_len,
                report.device_main_x_len,
                report.context_kv_lengths,
                report.compared_rows,
                report.max_abs_diff
            );
        }
    }

    // ── Independent page-managed token-loop oracle ──────────────────────
    if !json {
        println!("--- token-loop prefill ---");
    }
    let token_loop_start = Instant::now();
    for &token_id in &token_ids[..token_ids.len() - 1] {
        diagnostic.execute_sequence_step(
            token_loop_sequence,
            ForwardPhase::Decode,
            std::slice::from_ref(&token_id),
            |runner| runner.feed_token(token_id),
        )?;
    }
    let last_token = token_ids[token_ids.len() - 1];
    let (token_loop_trace, token_loop_context_kv_before_trace) = diagnostic.execute_sequence_step(
        token_loop_sequence,
        ForwardPhase::Decode,
        std::slice::from_ref(&last_token),
        |runner| {
            let context_kv = runner.dspark_context_kv_lengths();
            let trace = runner.decode_token_layer_hc_trace(last_token)?;
            Ok((trace, context_kv))
        },
    )?;
    let token_loop_checkpoints = diagnostic.runner_mut().take_parity_checkpoints();
    let token_loop_elapsed = token_loop_start.elapsed();

    if !json {
        println!(
            "token_loop: {:.3} ms, {} layers captured, context_kv_before_trace={:?}",
            token_loop_elapsed.as_secs_f64() * 1000.0,
            token_loop_trace.len(),
            token_loop_context_kv_before_trace
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

    let checkpoint_diffs = batched_checkpoints
        .iter()
        .filter_map(|(stage, batched)| {
            token_loop_checkpoints
                .get(stage)
                .map(|token_loop| (stage.clone(), max_abs_difference(batched, token_loop)))
        })
        .collect::<Vec<_>>();
    if !json && !checkpoint_diffs.is_empty() {
        println!("--- selected layer checkpoints ---");
        for (stage, diff) in &checkpoint_diffs {
            println!("  {stage:<20} max_abs_diff={diff:.6e}");
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

        let batched_top1 = top1_from_hc(diagnostic.runner_mut(), &batched_trace[hc_idx])?;
        let token_loop_top1 = top1_from_hc(diagnostic.runner_mut(), &token_loop_trace[hc_idx])?;

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

    // ── Production prefill + complete DSpark proposal ────────────────────
    // Trace-only paths above intentionally remain independent. Reuse the same
    // two page-managed sequences after reset and execute the production paths
    // that populate committed DSpark context-KV before proposing.
    let dspark_available = diagnostic.runner().operator_backend() == ModelExecutionBackend::Cuda
        && diagnostic.runner().mtp().is_some();
    let (
        batched_first_dspark_proposal,
        batched_dspark_proposal,
        token_loop_first_dspark_proposal,
        token_loop_dspark_proposal,
    ) = if dspark_available {
        if !json {
            println!("--- production DSpark proposal parity ---");
        }
        diagnostic.reset_sequence(batched_sequence)?;
        let (batched_first, batched_repeat) = diagnostic.execute_sequence_step(
            batched_sequence,
            ForwardPhase::Prefill,
            &token_ids,
            |runner| {
                let target = runner.prefill_tokens_topk_batched(&token_ids, 1)?;
                let anchor = target
                    .first()
                    .ok_or_else(|| Error::Execution("target prefill returned no frontier".into()))?
                    .token_id;
                let first = run_dspark_proposal_if_ready(runner, anchor, token_ids.len(), true)?
                    .ok_or_else(|| {
                        Error::Execution(
                            "production DSpark proposal was unavailable after batched prefill"
                                .into(),
                        )
                    })?;
                let repeat = run_dspark_proposal_if_ready(runner, anchor, token_ids.len(), false)?
                    .ok_or_else(|| {
                        Error::Execution(
                            "production DSpark proposal repeat was unavailable after batched prefill"
                                .into(),
                        )
                    })?;
                Ok((first, repeat))
            },
        )?;

        diagnostic.reset_sequence(token_loop_sequence)?;
        for &token_id in &token_ids[..token_ids.len() - 1] {
            diagnostic.execute_sequence_step(
                token_loop_sequence,
                ForwardPhase::Decode,
                std::slice::from_ref(&token_id),
                |runner| runner.feed_token(token_id),
            )?;
        }
        let (token_loop_first, token_loop_repeat) = diagnostic.execute_sequence_step(
            token_loop_sequence,
            ForwardPhase::Decode,
            std::slice::from_ref(&last_token),
            |runner| {
                let target = runner.decode_token_topk(last_token, 1)?;
                let anchor = target
                    .first()
                    .ok_or_else(|| Error::Execution("target decode returned no frontier".into()))?
                    .token_id;
                let first =
                    run_dspark_proposal_if_ready(runner, anchor, 1, true)?.ok_or_else(|| {
                        Error::Execution(
                            "production DSpark proposal was unavailable after token-loop prefill"
                                .into(),
                        )
                    })?;
                let repeat =
                    run_dspark_proposal_if_ready(runner, anchor, 1, false)?.ok_or_else(|| {
                        Error::Execution(
                        "production DSpark proposal repeat was unavailable after token-loop prefill"
                            .into(),
                    )
                    })?;
                Ok((first, repeat))
            },
        )?;
        if !json {
            for (label, report) in [
                ("batched first", &batched_first),
                ("batched repeat", &batched_repeat),
                ("token-loop first", &token_loop_first),
                ("token-loop repeat", &token_loop_repeat),
            ] {
                println!(
                    "{label} proposal: anchor={} tokens={:?} confidence={:?} elapsed={:.3} ms context_kv={:?}",
                    report.anchor_token_id,
                    report.token_ids,
                    report.confidence_scores,
                    report.elapsed_us as f64 / 1000.0,
                    report.context_kv_after,
                );
                println!(
                    "  counters: kernels={} h2d={}/{}B d2h={}/{}B allocations={} syncs={} moe_calls={} moe_total={:.3}ms router={:.3}ms routing={:.3}ms plan={:.3}ms shared={:.3}ms compute_submit={:.3}ms expert_loads={}/{}B",
                    report.counters.kernel_launches,
                    report.counters.host_to_device_copies,
                    report.counters.host_to_device_bytes,
                    report.counters.device_to_host_copies,
                    report.counters.device_to_host_bytes,
                    report.counters.device_allocations,
                    report.counters.stream_wide_syncs,
                    report.counters.moe_calls,
                    report.counters.moe_total_us as f64 / 1000.0,
                    report.counters.moe_router_us as f64 / 1000.0,
                    report.counters.moe_routing_us as f64 / 1000.0,
                    report.counters.moe_plan_us as f64 / 1000.0,
                    report.counters.moe_shared_us as f64 / 1000.0,
                    report.counters.moe_compute_submit_us as f64 / 1000.0,
                    report.counters.expert_loads,
                    report.counters.expert_load_bytes,
                );
                if let Some(debug) = report.backbone_debug.as_ref() {
                    println!(
                        "  backbone: initial_rms={:.6e} initial_max_abs={:.6e} stages={}",
                        vector_rms(&debug.initial_hc_state),
                        vector_max_abs(&debug.initial_hc_state),
                        debug.stage_hc_states.len(),
                    );
                    for (stage, state) in debug.stage_hc_states.iter().enumerate() {
                        let boundaries = &debug.stage_boundaries[stage];
                        println!(
                            "    stage={stage} hc_rms={:.6e} hc_max_abs={:.6e} attn_hidden_rms={:.6e} attn_norm_rms={:.6e} attn_output_rms={:.6e} after_attn_rms={:.6e} ffn_hidden_rms={:.6e} ffn_norm_rms={:.6e} moe_output_rms={:.6e}",
                            vector_rms(state),
                            vector_max_abs(state),
                            vector_rms(&boundaries.attention_hidden),
                            vector_rms(&boundaries.attention_normalized),
                            vector_rms(&boundaries.attention_output),
                            vector_rms(&boundaries.after_attention),
                            vector_rms(&boundaries.ffn_hidden),
                            vector_rms(&boundaries.ffn_normalized),
                            vector_rms(&boundaries.moe_output),
                        );
                    }
                }
                if let Some(debug) = report.head_debug.as_ref() {
                    println!(
                        "  head: hidden_rms={:.6e} hidden_max_abs={:.6e} normalized_rms={:.6e} normalized_max_abs={:.6e} cpu_confidence_max_abs_diff={:.6e}",
                        vector_rms(&debug.hidden),
                        vector_max_abs(&debug.hidden),
                        vector_rms(&debug.normalized),
                        vector_max_abs(&debug.normalized),
                        debug.cpu_confidence_max_abs_diff,
                    );
                    for row in &debug.rows {
                        println!(
                            "    row={} previous={} emitted={} cpu={} base_topk={:?} markov_topk={:?} combined_topk={:?} emitted_parts=({:.6e}, {:.6e}, {:.6e}) confidence=({:.6e}, {:.6e})",
                            row.row,
                            row.previous_token_id,
                            row.emitted_token_id,
                            row.cpu_token_id,
                            row.base_topk,
                            row.markov_bias_topk,
                            row.combined_topk,
                            row.emitted_base_logit,
                            row.emitted_markov_bias,
                            row.emitted_combined_logit,
                            row.gpu_confidence,
                            row.cpu_confidence,
                        );
                    }
                }
            }
        }
        (
            Some(batched_first),
            Some(batched_repeat),
            Some(token_loop_first),
            Some(token_loop_repeat),
        )
    } else {
        (None, None, None, None)
    };

    let dspark_proposal_parity = match (
        batched_dspark_proposal.as_ref(),
        token_loop_dspark_proposal.as_ref(),
    ) {
        (Some(batched), Some(token_loop)) => {
            let batched_first = batched_first_dspark_proposal
                .as_ref()
                .expect("DSpark first/repeat reports are paired");
            let token_loop_first = token_loop_first_dspark_proposal
                .as_ref()
                .expect("DSpark first/repeat reports are paired");
            Some(DsparkProposalParity {
                token_ids_match: batched.token_ids == token_loop.token_ids,
                confidence_max_abs_diff: max_abs_difference(
                    &batched.confidence_scores,
                    &token_loop.confidence_scores,
                ),
                batched_repeat_tokens_match: batched_first.token_ids == batched.token_ids,
                batched_repeat_confidence_max_abs_diff: max_abs_difference(
                    &batched_first.confidence_scores,
                    &batched.confidence_scores,
                ),
                token_loop_repeat_tokens_match: token_loop_first.token_ids == token_loop.token_ids,
                token_loop_repeat_confidence_max_abs_diff: max_abs_difference(
                    &token_loop_first.confidence_scores,
                    &token_loop.confidence_scores,
                ),
                target_taps_max_abs_diff: max_abs_difference(
                    &batched.target_taps_last_row,
                    &token_loop.target_taps_last_row,
                ),
                main_x_max_abs_diff: max_abs_difference(
                    &batched.main_x_last_row,
                    &token_loop.main_x_last_row,
                ),
            })
        }
        (None, None) => None,
        _ => anyhow::bail!(
            "DSpark proposal readiness diverged between batched and token-loop prefill"
        ),
    };

    // ── Summary ─────────────────────────────────────────────────────────
    if !json {
        println!("--- summary ---");
        if let Some(parity) = dspark_proposal_parity.as_ref() {
            println!(
                "dspark proposal parity: cross_tokens_match={} cross_confidence_max_abs_diff={:.6e} batched_repeat_tokens_match={} batched_repeat_confidence_max_abs_diff={:.6e} token_loop_repeat_tokens_match={} token_loop_repeat_confidence_max_abs_diff={:.6e} target_taps_max_abs_diff={:.6e} main_x_max_abs_diff={:.6e}",
                parity.token_ids_match,
                parity.confidence_max_abs_diff,
                parity.batched_repeat_tokens_match,
                parity.batched_repeat_confidence_max_abs_diff,
                parity.token_loop_repeat_tokens_match,
                parity.token_loop_repeat_confidence_max_abs_diff,
                parity.target_taps_max_abs_diff,
                parity.main_x_max_abs_diff,
            );
        }
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
            "backend": diagnostic.runner().operator_backend().as_str(),
            "prompt": prompt,
            "chat_prompt": chat_prompt,
            "prompt_tokens": token_ids,
            "max_layers": max_layers,
            "atol": atol,
            "cuts": cuts,
            "batched_ms": batched_elapsed.as_secs_f64() * 1000.0,
            "token_loop_ms": token_loop_elapsed.as_secs_f64() * 1000.0,
            "token_loop_context_kv_before_trace": token_loop_context_kv_before_trace,
            "n_layers_compared": n_layers,
            "dspark_main": dspark_main_report.as_ref().map(|report| {
                serde_json::json!({
                    "target_taps_len": report.target_taps_len,
                    "device_main_x_len": report.device_main_x_len,
                    "context_kv_lengths": report.context_kv_lengths,
                    "compared_rows": report.compared_rows,
                    "max_abs_diff": report.max_abs_diff,
                    "within_atol": report.max_abs_diff <= atol,
                })
            }),
            "dspark_proposal": {
                "batched_first": batched_first_dspark_proposal.as_ref().map(dspark_proposal_json),
                "batched_repeat": batched_dspark_proposal.as_ref().map(dspark_proposal_json),
                "token_loop_first": token_loop_first_dspark_proposal.as_ref().map(dspark_proposal_json),
                "token_loop_repeat": token_loop_dspark_proposal.as_ref().map(dspark_proposal_json),
                "parity": dspark_proposal_parity.as_ref().map(|parity| {
                    serde_json::json!({
                        "token_ids_match": parity.token_ids_match,
                        "confidence_max_abs_diff": parity.confidence_max_abs_diff,
                        "batched_repeat_tokens_match": parity.batched_repeat_tokens_match,
                        "batched_repeat_confidence_max_abs_diff": parity.batched_repeat_confidence_max_abs_diff,
                        "token_loop_repeat_tokens_match": parity.token_loop_repeat_tokens_match,
                        "token_loop_repeat_confidence_max_abs_diff": parity.token_loop_repeat_confidence_max_abs_diff,
                        "target_taps_max_abs_diff": parity.target_taps_max_abs_diff,
                        "main_x_max_abs_diff": parity.main_x_max_abs_diff,
                    })
                }),
            },
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
            "checkpoint_diffs": checkpoint_diffs.iter().map(|(stage, diff)| {
                serde_json::json!({
                    "stage": stage,
                    "max_abs_diff": diff,
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

#[derive(Debug)]
struct DsparkProposalReport {
    anchor_token_id: u32,
    token_ids: Vec<u32>,
    confidence_scores: Vec<f32>,
    elapsed_us: u64,
    context_kv_before: Vec<usize>,
    context_kv_after: Vec<usize>,
    target_taps_last_row: Vec<f32>,
    main_x_last_row: Vec<f32>,
    main_x_full: Option<Vec<f32>>,
    backbone_debug: Option<DeepSeekV4DsparkBackboneDebugSnapshot>,
    head_debug: Option<DsparkProposalHeadDebugReport>,
    counters: DsparkProposalCounterDelta,
}

#[derive(Debug)]
struct DsparkProposalHeadDebugReport {
    hidden: Vec<f32>,
    normalized: Vec<f32>,
    rows: Vec<DsparkProposalHeadDebugRow>,
    cpu_confidence_max_abs_diff: f32,
}

#[derive(Debug)]
struct DsparkProposalHeadDebugRow {
    row: usize,
    previous_token_id: u32,
    emitted_token_id: u32,
    cpu_token_id: u32,
    base_topk: Vec<DsparkRankedLogit>,
    markov_bias_topk: Vec<DsparkRankedLogit>,
    combined_topk: Vec<DsparkRankedLogit>,
    emitted_base_logit: f32,
    emitted_markov_bias: f32,
    emitted_combined_logit: f32,
    gpu_confidence: f32,
    cpu_confidence: f32,
}

#[derive(Debug, Clone, Copy)]
struct DsparkRankedLogit {
    token_id: u32,
    logit: f32,
}

#[derive(Debug, Clone, Copy)]
struct DsparkProposalCounterDelta {
    kernel_launches: u64,
    host_to_device_copies: u64,
    host_to_device_bytes: u64,
    device_to_host_copies: u64,
    device_to_host_bytes: u64,
    device_allocations: u64,
    stream_wide_syncs: u64,
    moe_calls: u64,
    moe_total_us: u64,
    moe_router_us: u64,
    moe_routing_us: u64,
    moe_plan_us: u64,
    moe_shared_us: u64,
    moe_compute_submit_us: u64,
    expert_loads: u64,
    expert_load_bytes: u64,
}

#[derive(Debug)]
struct DsparkProposalParity {
    token_ids_match: bool,
    confidence_max_abs_diff: f32,
    batched_repeat_tokens_match: bool,
    batched_repeat_confidence_max_abs_diff: f32,
    token_loop_repeat_tokens_match: bool,
    token_loop_repeat_confidence_max_abs_diff: f32,
    target_taps_max_abs_diff: f32,
    main_x_max_abs_diff: f32,
}

fn run_dspark_proposal_if_ready(
    runner: &mut DeepSeekV4Runner,
    anchor_token_id: u32,
    _main_snapshot_rows: usize,
    _capture_head_debug: bool,
) -> FerruleResult<Option<DsparkProposalReport>> {
    if runner.operator_backend() != ModelExecutionBackend::Cuda || runner.mtp().is_none() {
        return Ok(None);
    }
    let context_kv_before = runner.dspark_context_kv_lengths();
    if context_kv_before.is_empty()
        || context_kv_before.contains(&0)
        || context_kv_before.windows(2).any(|pair| pair[0] != pair[1])
    {
        return Err(Error::Execution(format!(
            "production DSpark proposal requires equal non-zero committed context-KV lengths, got {context_kv_before:?}"
        )));
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    {
        let (target_taps, main_x) = runner
            .dspark_main_debug_snapshot(_main_snapshot_rows)?
            .ok_or_else(|| {
                Error::Execution(format!(
                    "production DSpark main snapshot for {_main_snapshot_rows} rows is unavailable"
                ))
            })?;
        let tap_width = target_taps.len() / _main_snapshot_rows;
        let main_width = main_x.len() / _main_snapshot_rows;
        let target_taps_last_row = target_taps
            [(_main_snapshot_rows - 1) * tap_width.._main_snapshot_rows * tap_width]
            .to_vec();
        let main_x_last_row = main_x
            [(_main_snapshot_rows - 1) * main_width.._main_snapshot_rows * main_width]
            .to_vec();
        let counters_before = runner.operator_runtime_counters();
        let started = Instant::now();
        let proposal = runner.dspark_proposal_cuda(anchor_token_id)?;
        let elapsed_us = started.elapsed().as_micros() as u64;
        let counters_after = runner.operator_runtime_counters();
        let counters = DsparkProposalCounterDelta {
            kernel_launches: counters_after
                .kernel_launches
                .saturating_sub(counters_before.kernel_launches),
            host_to_device_copies: counters_after
                .host_to_device_copies
                .saturating_sub(counters_before.host_to_device_copies),
            host_to_device_bytes: counters_after
                .host_to_device_bytes
                .saturating_sub(counters_before.host_to_device_bytes),
            device_to_host_copies: counters_after
                .device_to_host_copies
                .saturating_sub(counters_before.device_to_host_copies),
            device_to_host_bytes: counters_after
                .device_to_host_bytes
                .saturating_sub(counters_before.device_to_host_bytes),
            device_allocations: counters_after
                .device_allocations
                .saturating_sub(counters_before.device_allocations),
            stream_wide_syncs: counters_after
                .stream_wide_syncs
                .saturating_sub(counters_before.stream_wide_syncs),
            moe_calls: counters_after
                .moe_calls
                .saturating_sub(counters_before.moe_calls),
            moe_total_us: counters_after
                .moe_total_us
                .saturating_sub(counters_before.moe_total_us),
            moe_router_us: counters_after
                .moe_router_us
                .saturating_sub(counters_before.moe_router_us),
            moe_routing_us: counters_after
                .moe_routing_us
                .saturating_sub(counters_before.moe_routing_us),
            moe_plan_us: counters_after
                .moe_plan_us
                .saturating_sub(counters_before.moe_plan_us),
            moe_shared_us: counters_after
                .moe_shared_us
                .saturating_sub(counters_before.moe_shared_us),
            moe_compute_submit_us: counters_after
                .moe_compute_submit_us
                .saturating_sub(counters_before.moe_compute_submit_us),
            expert_loads: counters_after
                .expert_loads
                .saturating_sub(counters_before.expert_loads),
            expert_load_bytes: counters_after
                .expert_load_bytes
                .saturating_sub(counters_before.expert_load_bytes),
        };
        let context_kv_after = runner.dspark_context_kv_lengths();
        if context_kv_after != context_kv_before {
            return Err(Error::Execution(format!(
                "DSpark proposal mutated committed context-KV lengths: before={context_kv_before:?} after={context_kv_after:?}"
            )));
        }
        let backbone_debug = if _capture_head_debug {
            let (debug_proposal, snapshot) = runner.dspark_proposal_cuda_debug(anchor_token_id)?;
            if debug_proposal.token_ids != proposal.token_ids
                || max_abs_difference(
                    &debug_proposal.confidence_scores,
                    &proposal.confidence_scores,
                ) > 1e-5
            {
                return Err(Error::Execution(format!(
                    "DSpark diagnostic rerun changed proposal output: production={:?}/{:?} debug={:?}/{:?}",
                    proposal.token_ids,
                    proposal.confidence_scores,
                    debug_proposal.token_ids,
                    debug_proposal.confidence_scores,
                )));
            }
            Some(snapshot)
        } else {
            None
        };
        let head_debug = if _capture_head_debug {
            let (hidden, normalized, base_logits) = runner
                .dspark_proposal_head_debug_snapshot()?
                .ok_or_else(|| {
                Error::Execution(
                    "production DSpark proposal-head debug snapshot is unavailable".into(),
                )
            })?;
            let mtp = runner.mtp().ok_or_else(|| {
                Error::Execution("DSpark proposal exists without MTP resources".into())
            })?;
            Some(build_dspark_proposal_head_debug_report(
                mtp,
                anchor_token_id,
                &proposal.token_ids,
                &proposal.confidence_scores,
                hidden,
                normalized,
                base_logits,
            )?)
        } else {
            None
        };
        return Ok(Some(DsparkProposalReport {
            anchor_token_id,
            token_ids: proposal.token_ids,
            confidence_scores: proposal.confidence_scores,
            elapsed_us,
            context_kv_before,
            context_kv_after,
            target_taps_last_row,
            main_x_last_row,
            main_x_full: _capture_head_debug.then_some(main_x),
            backbone_debug,
            head_debug,
            counters,
        }));
    }

    #[cfg(not(all(feature = "cuda", feature = "cutlass")))]
    {
        let _ = anchor_token_id;
        Ok(None)
    }
}

#[cfg(all(feature = "cuda", feature = "cutlass"))]
const DSPARK_HEAD_TOP_K: usize = 8;

#[cfg(all(feature = "cuda", feature = "cutlass"))]
#[allow(clippy::too_many_arguments)]
fn build_dspark_proposal_head_debug_report(
    mtp: &DeepSeekV4MtpModel,
    anchor_token_id: u32,
    proposal_token_ids: &[u32],
    gpu_confidence: &[f32],
    hidden: Vec<f32>,
    normalized: Vec<f32>,
    base_logits: Vec<f32>,
) -> FerruleResult<DsparkProposalHeadDebugReport> {
    let heads = mtp
        .prediction_heads
        .as_ref()
        .ok_or_else(|| Error::Model("DeepSeek-V4 DSpark prediction heads are missing".into()))?;
    let rows = proposal_token_ids.len();
    let hidden_size = heads.norm.len();
    let markov_rank = heads.markov_w1.format.in_features();
    let vocab = heads.markov_w2.format.out_features();
    if rows == 0
        || gpu_confidence.len() != rows
        || hidden.len() != rows.saturating_mul(hidden_size)
        || normalized.len() != hidden.len()
        || base_logits.len() != rows.saturating_mul(vocab)
        || heads.markov_w1.format.out_features() != vocab
        || heads.markov_w2.format.in_features() != markov_rank
        || heads.confidence_proj.format.in_features() != hidden_size + markov_rank
        || heads.confidence_proj.format.out_features() != 1
    {
        return Err(Error::Model(format!(
            "DeepSeek-V4 DSpark debug shape mismatch: rows={rows} hidden={}/{} normalized={} base_logits={}/{} confidence={}/{} w1={:?} w2={:?} confidence_head={:?}",
            hidden.len(),
            rows.saturating_mul(hidden_size),
            normalized.len(),
            base_logits.len(),
            rows.saturating_mul(vocab),
            gpu_confidence.len(),
            rows,
            heads.markov_w1.format,
            heads.markov_w2.format,
            heads.confidence_proj.format,
        )));
    }

    let previous_token_ids = std::iter::once(anchor_token_id)
        .chain(
            proposal_token_ids
                .iter()
                .copied()
                .take(rows.saturating_sub(1)),
        )
        .collect::<Vec<_>>();
    if previous_token_ids
        .iter()
        .any(|token_id| *token_id as usize >= vocab)
    {
        return Err(Error::Model(format!(
            "DeepSeek-V4 DSpark debug previous token is outside vocab {vocab}: {previous_token_ids:?}"
        )));
    }

    // Decode W1 once, retain only the five selected embedding rows, then release
    // it before decoding W2 so this diagnostic never holds both full matrices.
    let markov_w1 = heads.markov_w1.reference_weights_f32()?;
    let markov_embeds = previous_token_ids
        .iter()
        .map(|token_id| {
            let start = *token_id as usize * markov_rank;
            markov_w1[start..start + markov_rank].to_vec()
        })
        .collect::<Vec<_>>();
    drop(markov_w1);
    let markov_w2 = heads.markov_w2.reference_weights_f32()?;

    let mut debug_rows = Vec::with_capacity(rows);
    for row in 0..rows {
        let base_row = &base_logits[row * vocab..(row + 1) * vocab];
        let markov_embed = &markov_embeds[row];
        let mut base_topk = Vec::with_capacity(DSPARK_HEAD_TOP_K);
        let mut markov_bias_topk = Vec::with_capacity(DSPARK_HEAD_TOP_K);
        let mut combined_topk = Vec::with_capacity(DSPARK_HEAD_TOP_K);
        let emitted_token_id = proposal_token_ids[row];
        let emitted_index = emitted_token_id as usize;
        if emitted_index >= vocab {
            return Err(Error::Model(format!(
                "DeepSeek-V4 DSpark debug emitted token {emitted_token_id} is outside vocab {vocab}"
            )));
        }
        let mut emitted_markov_bias = 0.0f32;
        for token in 0..vocab {
            let weight_row = &markov_w2[token * markov_rank..(token + 1) * markov_rank];
            let markov_bias = weight_row
                .iter()
                .zip(markov_embed)
                .fold(0.0f32, |sum, (weight, value)| weight.mul_add(*value, sum));
            let combined = base_row[token] + markov_bias;
            offer_ranked_logit(
                &mut base_topk,
                DsparkRankedLogit {
                    token_id: token as u32,
                    logit: base_row[token],
                },
                DSPARK_HEAD_TOP_K,
            );
            offer_ranked_logit(
                &mut markov_bias_topk,
                DsparkRankedLogit {
                    token_id: token as u32,
                    logit: markov_bias,
                },
                DSPARK_HEAD_TOP_K,
            );
            offer_ranked_logit(
                &mut combined_topk,
                DsparkRankedLogit {
                    token_id: token as u32,
                    logit: combined,
                },
                DSPARK_HEAD_TOP_K,
            );
            if token == emitted_index {
                emitted_markov_bias = markov_bias;
            }
        }
        let cpu_token_id = combined_topk
            .first()
            .map(|entry| entry.token_id)
            .unwrap_or(u32::MAX);
        let hidden_row = &hidden[row * hidden_size..(row + 1) * hidden_size];
        let mut confidence_input = Vec::with_capacity(hidden_size + markov_rank);
        confidence_input.extend_from_slice(hidden_row);
        confidence_input.extend_from_slice(markov_embed);
        let cpu_confidence = heads.confidence_proj.reference_matvec(&confidence_input)?[0];
        debug_rows.push(DsparkProposalHeadDebugRow {
            row,
            previous_token_id: previous_token_ids[row],
            emitted_token_id,
            cpu_token_id,
            base_topk,
            markov_bias_topk,
            combined_topk,
            emitted_base_logit: base_row[emitted_index],
            emitted_markov_bias,
            emitted_combined_logit: base_row[emitted_index] + emitted_markov_bias,
            gpu_confidence: gpu_confidence[row],
            cpu_confidence,
        });
    }

    let cpu_confidence_max_abs_diff = debug_rows.iter().fold(0.0f32, |maximum, row| {
        maximum.max((row.gpu_confidence - row.cpu_confidence).abs())
    });
    Ok(DsparkProposalHeadDebugReport {
        hidden,
        normalized,
        rows: debug_rows,
        cpu_confidence_max_abs_diff,
    })
}

#[cfg(all(feature = "cuda", feature = "cutlass"))]
fn offer_ranked_logit(
    top: &mut Vec<DsparkRankedLogit>,
    candidate: DsparkRankedLogit,
    capacity: usize,
) {
    let insert_at = top
        .iter()
        .position(|current| {
            candidate.logit > current.logit
                || (candidate.logit == current.logit && candidate.token_id < current.token_id)
        })
        .unwrap_or(top.len());
    if insert_at < capacity {
        top.insert(insert_at, candidate);
        top.truncate(capacity);
    } else if top.len() < capacity {
        top.push(candidate);
    }
}

fn vector_rms(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().fold(0.0f64, |sum, value| {
        (*value as f64).mul_add(*value as f64, sum)
    }) / values.len() as f64)
        .sqrt() as f32
}

fn vector_max_abs(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()))
}

fn ranked_logits_json(values: &[DsparkRankedLogit]) -> Vec<serde_json::Value> {
    values
        .iter()
        .map(|entry| {
            serde_json::json!({
                "token_id": entry.token_id,
                "logit": entry.logit,
            })
        })
        .collect()
}

fn dspark_head_debug_json(report: &DsparkProposalHeadDebugReport) -> serde_json::Value {
    serde_json::json!({
        "hidden": {
            "values": report.hidden,
            "rms": vector_rms(&report.hidden),
            "max_abs": vector_max_abs(&report.hidden),
        },
        "normalized": {
            "values": report.normalized,
            "rms": vector_rms(&report.normalized),
            "max_abs": vector_max_abs(&report.normalized),
        },
        "cpu_confidence_max_abs_diff": report.cpu_confidence_max_abs_diff,
        "rows": report.rows.iter().map(|row| {
            serde_json::json!({
                "row": row.row,
                "previous_token_id": row.previous_token_id,
                "emitted_token_id": row.emitted_token_id,
                "cpu_token_id": row.cpu_token_id,
                "cpu_token_matches": row.cpu_token_id == row.emitted_token_id,
                "base_topk": ranked_logits_json(&row.base_topk),
                "markov_bias_topk": ranked_logits_json(&row.markov_bias_topk),
                "combined_topk": ranked_logits_json(&row.combined_topk),
                "emitted_base_logit": row.emitted_base_logit,
                "emitted_markov_bias": row.emitted_markov_bias,
                "emitted_combined_logit": row.emitted_combined_logit,
                "gpu_confidence": row.gpu_confidence,
                "cpu_confidence": row.cpu_confidence,
                "confidence_abs_diff": (row.gpu_confidence - row.cpu_confidence).abs(),
            })
        }).collect::<Vec<_>>(),
    })
}

fn dspark_backbone_debug_json(report: &DeepSeekV4DsparkBackboneDebugSnapshot) -> serde_json::Value {
    serde_json::json!({
        "initial_hc_state": {
            "values": report.initial_hc_state,
            "rms": vector_rms(&report.initial_hc_state),
            "max_abs": vector_max_abs(&report.initial_hc_state),
        },
        "stage_hc_states": report.stage_hc_states.iter().enumerate().map(|(stage, values)| {
            serde_json::json!({
                "stage": stage,
                "values": values,
                "rms": vector_rms(values),
                "max_abs": vector_max_abs(values),
            })
        }).collect::<Vec<_>>(),
        "stage_boundaries": report.stage_boundaries.iter().enumerate().map(|(stage, values)| {
            serde_json::json!({
                "stage": stage,
                "attention": {
                    "query_latent": values.attention.query_latent,
                    "query_normalized": values.attention.query_normalized,
                    "query_projected": values.attention.query_projected,
                    "query_rope": values.attention.query_rope,
                    "kv_raw": values.attention.kv_raw,
                    "kv_rope_qat": values.attention.kv_rope_qat,
                    "context_inverse_rope": values.attention.context_inverse_rope,
                    "output_a": values.attention.output_a,
                    "output_b": values.attention.output_b,
                },
                "attention_hidden": values.attention_hidden,
                "attention_normalized": values.attention_normalized,
                "attention_output": values.attention_output,
                "after_attention": values.after_attention,
                "ffn_hidden": values.ffn_hidden,
                "ffn_normalized": values.ffn_normalized,
                "moe_output": values.moe_output,
            })
        }).collect::<Vec<_>>(),
    })
}

fn dspark_proposal_json(report: &DsparkProposalReport) -> serde_json::Value {
    serde_json::json!({
        "anchor_token_id": report.anchor_token_id,
        "token_ids": report.token_ids,
        "confidence_scores": report.confidence_scores,
        "elapsed_us": report.elapsed_us,
        "context_kv_before": report.context_kv_before,
        "context_kv_after": report.context_kv_after,
        "committed_context_unchanged": report.context_kv_before == report.context_kv_after,
        "target_taps_last_row_values": report.target_taps_last_row.len(),
        "main_x_last_row_values": report.main_x_last_row.len(),
        "main_x_full": report.main_x_full,
        "backbone_debug": report.backbone_debug.as_ref().map(dspark_backbone_debug_json),
        "head_debug": report.head_debug.as_ref().map(dspark_head_debug_json),
        "counters": {
            "kernel_launches": report.counters.kernel_launches,
            "host_to_device_copies": report.counters.host_to_device_copies,
            "host_to_device_bytes": report.counters.host_to_device_bytes,
            "device_to_host_copies": report.counters.device_to_host_copies,
            "device_to_host_bytes": report.counters.device_to_host_bytes,
            "device_allocations": report.counters.device_allocations,
            "stream_wide_syncs": report.counters.stream_wide_syncs,
            "moe_calls": report.counters.moe_calls,
            "moe_total_us": report.counters.moe_total_us,
            "moe_router_us": report.counters.moe_router_us,
            "moe_routing_us": report.counters.moe_routing_us,
            "moe_plan_us": report.counters.moe_plan_us,
            "moe_shared_us": report.counters.moe_shared_us,
            "moe_compute_submit_us": report.counters.moe_compute_submit_us,
            "expert_loads": report.counters.expert_loads,
            "expert_load_bytes": report.counters.expert_load_bytes,
        },
    })
}

struct DsparkMainReport {
    target_taps_len: usize,
    device_main_x_len: usize,
    context_kv_lengths: Vec<usize>,
    compared_rows: usize,
    max_abs_diff: f32,
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
    if a.len() != b.len()
        || a.iter()
            .zip(b.iter())
            .any(|(x, y)| !x.is_finite() || !y.is_finite())
    {
        return f32::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn top1_from_hc(runner: &mut DeepSeekV4Runner, hc_state: &[f32]) -> anyhow::Result<u32> {
    let topk = runner.topk_from_hc_trace(hc_state)?;
    Ok(topk.first().map(|logit| logit.token_id).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::max_abs_difference;

    #[test]
    fn max_abs_difference_rejects_non_finite_values() {
        assert!(max_abs_difference(&[f32::NAN], &[f32::NAN]).is_infinite());
        assert!(max_abs_difference(&[f32::INFINITY], &[f32::INFINITY]).is_infinite());
    }

    #[test]
    fn max_abs_difference_reports_finite_difference() {
        assert_eq!(max_abs_difference(&[1.0, -2.0], &[1.25, -2.5]), 0.5);
    }
}
