#![cfg(all(feature = "cuda", feature = "local-dsv4-tests"))]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use ferrule_model::{
    models::deepseek_v4::{DeepSeekV4ArtifactModel, DeepSeekV4PrepareOptions, DeepSeekV4Runner},
    ChatTemplate, ModelExecutionBackend, ModelRunner, PrefillMode, TopKModelRunner,
};
use ferrule_runtime::{
    GenerateRequest, PagedSequenceKvCache, RequestId, ResidentSchedulerConfig, ResidentTopKDriver,
    ResidentTopKDriverConfig, SamplingConfig, SequenceFinishReason, SessionId,
    TopKCompatibilityExecutorConfig,
};

#[test]
#[ignore = "requires cargo-oxide, CUDA, and a local DeepSeek-V4-Flash-DSpark artifact"]
fn deepseek_v4_runs_latest_resident_runtime_driver_local() -> Result<()> {
    let model_dir = local_dsv4_model_dir();
    if !model_dir.is_dir() {
        eprintln!(
            "skipping local DSV4 runtime-driver test; model dir not found: {}",
            model_dir.display()
        );
        return Ok(());
    }

    let options = DeepSeekV4PrepareOptions {
        max_layers: 43,
        output_head_chunk_rows: 4096,
        moe_prefetch_experts: 8,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 128 * 1024 * 1024)
        .with_context(|| format!("load DSV4 artifact from {}", model_dir.display()))?;
    let runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;

    let prompt = ChatTemplate::DeepSeekV4.format_turn("Hello", true);
    let prompt_tokens = runner.encode(&prompt)?;
    assert!(
        !prompt_tokens.is_empty(),
        "DSV4 tokenizer returned empty prompt"
    );

    let mut driver = ResidentTopKDriver::with_configs(
        runner,
        // Metadata-only KV lifecycle for the current single-runner DSV4 backend.
        // The concrete DSV4 runner still owns physical CUDA KV/session state.
        PagedSequenceKvCache::new(1, 1, 1, 1),
        ResidentSchedulerConfig {
            prefill_chunk_size: 4096,
            max_active_sequences: 1,
            max_decode_batch: 1,
        },
        TopKCompatibilityExecutorConfig {
            top_k: 1,
            prefill_mode: PrefillMode::Interactive,
        },
        ResidentTopKDriverConfig {
            ctx_size: 4096,
            stop_at_eos: true,
            append_eos_to_session: true,
            max_steps_per_run: 4096 + 16,
        },
    );

    driver.submit_at_current_position(GenerateRequest {
        id: RequestId(1),
        session_id: Some(SessionId(0)),
        prompt_tokens: prompt_tokens.clone(),
        sampling: SamplingConfig::greedy(),
        max_new_tokens: 1,
        stop: Vec::new(),
    });

    let mut emitted = Vec::new();
    let stats = driver.run_until_blocked(|event| {
        emitted.push(event.token);
        Ok(())
    })?;

    assert_eq!(emitted.len(), 1, "expected exactly one emitted token");
    assert_eq!(stats.emitted_tokens, 1);
    assert_eq!(stats.finished_sequences, 1);
    assert_eq!(driver.kv_cache().active_count(), 0);

    let finished = driver.drain_finished();
    assert_eq!(finished.len(), 1, "expected one finished sequence");
    let sequence = &finished[0];
    assert_eq!(
        sequence.finish_reason,
        Some(SequenceFinishReason::MaxTokens)
    );
    assert_eq!(sequence.generated, 1);
    assert_eq!(sequence.position, prompt_tokens.len() + emitted.len());
    assert_eq!(
        driver.compatibility_executor().position(),
        sequence.position
    );

    Ok(())
}

#[test]
#[ignore = "requires cargo-oxide, CUDA, and a local DeepSeek-V4-Flash-DSpark artifact"]
fn deepseek_v4_cuda_eager_decode_correctness_local() -> Result<()> {
    let model_dir = local_dsv4_model_dir();
    if !model_dir.is_dir() {
        eprintln!(
            "skipping local DSV4 eager decode correctness test; model dir not found: {}",
            model_dir.display()
        );
        return Ok(());
    }

    let options = DeepSeekV4PrepareOptions {
        max_layers: 5,
        output_head_chunk_rows: 4096,
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 128 * 1024 * 1024)
        .with_context(|| format!("load DSV4 artifact from {}", model_dir.display()))?;
    let mut runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;
    let prompt = ChatTemplate::DeepSeekV4.format_turn("Hello", true);
    let prompt_tokens = runner.encode(&prompt)?;
    assert!(
        !prompt_tokens.is_empty() && prompt_tokens.len() < runner.model().config.window_size,
        "eager correctness prompt must be non-empty and shorter than attention window: prompt={} window={}",
        prompt_tokens.len(),
        runner.model().config.window_size
    );

    let prefill = runner.prefill_topk(&prompt_tokens, 1, PrefillMode::Batched)?;
    let prefill_top1 = *prefill
        .first()
        .context("DSV4 eager correctness prefill returned no top-1 candidate")?;
    assert!(prefill_top1.logit.is_finite());
    let checkpoint = runner.checkpoint_sequence_state()?;
    let decode = runner.decode_topk(prefill_top1.token_id, 1)?;
    let decode_top1 = *decode
        .first()
        .context("DSV4 eager correctness decode returned no top-1 candidate")?;
    assert!(decode_top1.logit.is_finite());

    runner.restore_sequence_state(&checkpoint)?;
    let warm = runner.operator_runtime_counters();
    let second_decode = runner.decode_topk(prefill_top1.token_id, 1)?;
    assert_eq!(second_decode, decode);
    let counters = runner.operator_runtime_counters();
    eprintln!(
        "DSV4 warm decode delta: allocations={} attempts={} bytes={} uploads={} h2d={}B d2h={}B expert_loads={} expert_evictions={} arena_hits={} arena_misses={} arena_grows={}",
        counters.device_allocations.saturating_sub(warm.device_allocations),
        counters
            .device_allocation_attempts
            .saturating_sub(warm.device_allocation_attempts),
        counters
            .device_allocation_bytes
            .saturating_sub(warm.device_allocation_bytes),
        counters.artifact_uploads.saturating_sub(warm.artifact_uploads),
        counters
            .host_to_device_bytes
            .saturating_sub(warm.host_to_device_bytes),
        counters
            .device_to_host_bytes
            .saturating_sub(warm.device_to_host_bytes),
        counters.expert_loads.saturating_sub(warm.expert_loads),
        counters.expert_evictions.saturating_sub(warm.expert_evictions),
        counters.arena_hits.saturating_sub(warm.arena_hits),
        counters.arena_misses.saturating_sub(warm.arena_misses),
        counters.arena_grows.saturating_sub(warm.arena_grows),
    );
    assert_eq!(
        counters
            .device_allocation_attempts
            .saturating_sub(warm.device_allocation_attempts),
        0,
        "second warm decode must make no device allocation attempts"
    );
    assert_eq!(
        counters
            .device_allocations
            .saturating_sub(warm.device_allocations),
        0,
        "second warm decode must make no device allocations"
    );
    assert_eq!(
        counters
            .artifact_uploads
            .saturating_sub(warm.artifact_uploads),
        0,
        "second warm decode must upload no artifacts"
    );
    assert_eq!(
        counters.arena_misses.saturating_sub(warm.arena_misses),
        0,
        "second warm decode must have no arena misses"
    );
    assert_eq!(
        counters.arena_grows.saturating_sub(warm.arena_grows),
        0,
        "second warm decode must have no arena grows"
    );
    assert_eq!(runner.position(), prompt_tokens.len() + 1);

    assert!(counters.device_allocations > 0);
    assert_eq!(
        counters.device_allocation_attempts,
        counters.device_allocations + counters.device_allocation_failures
    );
    assert_eq!(counters.device_allocation_failures, 0);
    assert!(counters.device_allocation_bytes > 0);
    assert_eq!(counters.stream_wide_sync_failures, 0);
    Ok(())
}

#[test]
#[ignore = "requires cargo-oxide, CUDA, and a local DeepSeek-V4-Flash-DSpark artifact"]
fn deepseek_v4_cuda_sequence_checkpoint_restore_local() -> Result<()> {
    let model_dir = local_dsv4_model_dir();
    if !model_dir.is_dir() {
        eprintln!(
            "skipping local DSV4 sequence checkpoint test; model dir not found: {}",
            model_dir.display()
        );
        return Ok(());
    }

    let options = DeepSeekV4PrepareOptions {
        max_layers: 5,
        output_head_chunk_rows: 4096,
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 128 * 1024 * 1024)
        .with_context(|| format!("load DSV4 artifact from {}", model_dir.display()))?;
    let mut runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;
    let prompt = ChatTemplate::DeepSeekV4.format_turn("Hello", true);
    let prompt_tokens = runner.encode(&prompt)?;
    let prefill = runner.prefill_topk(&prompt_tokens, 1, PrefillMode::Batched)?;
    let next = prefill
        .first()
        .context("DSV4 checkpoint prefill returned no top-1 candidate")?
        .token_id;
    let checkpoint_position = runner.position();
    let checkpoint = runner.checkpoint_sequence_state()?;

    let first = *runner
        .decode_topk(next, 1)?
        .first()
        .context("DSV4 checkpoint first decode returned no top-1 candidate")?;
    assert_eq!(runner.position(), checkpoint_position + 1);

    runner.restore_sequence_state(&checkpoint)?;
    assert_eq!(runner.position(), checkpoint_position);
    let restored = *runner
        .decode_topk(next, 1)?
        .first()
        .context("DSV4 checkpoint restored decode returned no top-1 candidate")?;
    assert_eq!(runner.position(), checkpoint_position + 1);
    assert_eq!(restored.token_id, first.token_id);
    let logit_abs_diff = (restored.logit - first.logit).abs();
    assert!(
        logit_abs_diff.is_finite() && logit_abs_diff <= 1.0e-3,
        "DSV4 checkpoint restore logit mismatch: first={} restored={} abs_diff={logit_abs_diff:e}",
        first.logit,
        restored.logit
    );
    Ok(())
}

#[test]
#[ignore = "requires cargo-oxide, CUDA, and a local DeepSeek-V4-Flash-DSpark artifact"]
fn deepseek_v4_cuda_interleaved_sequence_isolation_local() -> Result<()> {
    let model_dir = local_dsv4_model_dir();
    if !model_dir.is_dir() {
        eprintln!(
            "skipping local DSV4 sequence isolation test; model dir not found: {}",
            model_dir.display()
        );
        return Ok(());
    }

    let options = DeepSeekV4PrepareOptions {
        max_layers: 5,
        output_head_chunk_rows: 4096,
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 128 * 1024 * 1024)
        .with_context(|| format!("load DSV4 artifact from {}", model_dir.display()))?;
    let mut runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;
    let prompt = ChatTemplate::DeepSeekV4.format_turn("Hello", true);
    let prompt_tokens = runner.encode(&prompt)?;
    let mut a = runner.fork_sequence_state()?;
    let mut b = runner.fork_sequence_state()?;

    let a_prefill = runner.with_sequence_state(&mut a, |runner| {
        runner.prefill_topk(&prompt_tokens, 1, PrefillMode::Batched)
    })?;
    let b_prefill = runner.with_sequence_state(&mut b, |runner| {
        runner.prefill_topk(&prompt_tokens, 1, PrefillMode::Batched)
    })?;
    assert_eq!(a.position(), prompt_tokens.len());
    assert_eq!(b.position(), prompt_tokens.len());
    assert_eq!(a_prefill, b_prefill);

    let next = a_prefill
        .first()
        .context("DSV4 interleaved prefill returned no top-1 candidate")?
        .token_id;
    let a_decode = runner.with_sequence_state(&mut a, |runner| runner.decode_topk(next, 1))?;
    let b_decode = runner.with_sequence_state(&mut b, |runner| runner.decode_topk(next, 1))?;
    assert_eq!(a_decode, b_decode);
    assert_eq!(a.position(), prompt_tokens.len() + 1);
    assert_eq!(b.position(), prompt_tokens.len() + 1);

    let b_position = b.position();
    runner.release_sequence_state(a)?;
    assert_eq!(b.position(), b_position);

    runner.shutdown()?;
    runner.shutdown()?;
    let shutdown_error = runner
        .with_sequence_state(&mut b, |runner| runner.decode_topk(next, 1))
        .expect_err("shut down runner must reject execution");
    assert!(shutdown_error.to_string().contains("shut down"));
    Ok(())
}

#[test]
#[ignore = "requires cargo-oxide, CUDA, and a local DeepSeek-V4-Flash-DSpark artifact"]
fn deepseek_v4_cuda_transition_failure_poison_isolates_sequence_local() -> Result<()> {
    let model_dir = local_dsv4_model_dir();
    if !model_dir.is_dir() {
        eprintln!(
            "skipping local DSV4 transition failure test; model dir not found: {}",
            model_dir.display()
        );
        return Ok(());
    }

    let options = DeepSeekV4PrepareOptions {
        max_layers: 5,
        output_head_chunk_rows: 4096,
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 128 * 1024 * 1024)
        .with_context(|| format!("load DSV4 artifact from {}", model_dir.display()))?;
    let mut runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;
    let prompt = ChatTemplate::DeepSeekV4.format_turn("Hello", true);
    let prompt_tokens = runner.encode(&prompt)?;
    let prefill = runner.prefill_topk(&prompt_tokens, 1, PrefillMode::Batched)?;
    let next = prefill
        .first()
        .context("DSV4 transition prefill returned no top-1 candidate")?
        .token_id;
    let committed_position = runner.position();
    let mut baseline = runner.fork_sequence_state()?;
    let mut b = runner.fork_sequence_state()?;
    let mut indexer_failed = runner.fork_sequence_state()?;
    let mut main_failed = runner.fork_sequence_state()?;

    let expected =
        runner.with_sequence_state(&mut baseline, |runner| runner.decode_topk(next, 1))?;

    runner
        .cuda_failpoints()?
        .arm_indexer_compressor_transition();
    let indexer_error = runner
        .with_sequence_state(&mut indexer_failed, |runner| runner.decode_topk(next, 1))
        .expect_err("indexer transition failpoint must fail");
    assert!(indexer_error
        .to_string()
        .contains("indexer compressor transition"));
    assert_eq!(indexer_failed.position(), committed_position);
    assert!(indexer_failed.is_poisoned());
    let retry =
        runner.with_sequence_state(&mut indexer_failed, |runner| runner.decode_topk(next, 1));
    assert!(retry
        .expect_err("poisoned sequence retry must fail")
        .to_string()
        .contains("poisoned"));

    let actual_b = runner.with_sequence_state(&mut b, |runner| runner.decode_topk(next, 1))?;
    assert_eq!(actual_b, expected);
    assert_eq!(b.position(), committed_position + 1);
    assert!(!b.is_poisoned());

    runner.cuda_failpoints()?.arm_main_compressor_transition();
    let main_error = runner
        .with_sequence_state(&mut main_failed, |runner| runner.decode_topk(next, 1))
        .expect_err("main transition failpoint must fail");
    assert!(main_error
        .to_string()
        .contains("main compressor transition"));
    assert_eq!(main_failed.position(), committed_position);
    assert!(main_failed.is_poisoned());
    Ok(())
}

#[test]
#[ignore = "expensive: requires cargo-oxide, GB10, and the local 43-layer DSV4 artifact"]
fn deepseek_v4_cuda_continuation_crosses_4096_local() -> Result<()> {
    let model_dir = local_dsv4_model_dir();
    anyhow::ensure!(
        model_dir.is_dir(),
        "local DSV4 long-context gate requires model dir: {}",
        model_dir.display()
    );

    let max_layers = std::env::var("FERRULE_DSV4_LONG_CONTEXT_LAYERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(43);
    let prompt_len = std::env::var("FERRULE_DSV4_LONG_CONTEXT_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(4096);
    let options = DeepSeekV4PrepareOptions {
        max_layers,
        output_head_chunk_rows: 4096,
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 128 * 1024 * 1024)
        .with_context(|| format!("load DSV4 artifact from {}", model_dir.display()))?;
    let mut runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;

    let seed_prompt = ChatTemplate::DeepSeekV4.format_turn("Hello", true);
    let seed_tokens = runner.encode(&seed_prompt)?;
    assert!(!seed_tokens.is_empty());
    let prompt_tokens: Vec<u32> = seed_tokens
        .iter()
        .copied()
        .cycle()
        .take(prompt_len)
        .collect();
    if prompt_len >= 4096 {
        assert!(prompt_tokens.len() > runner.model().config.window_size);
    }

    eprintln!("DSV4 long-context prefill start: layers={max_layers} tokens={prompt_len}");
    let prefill_start = Instant::now();
    let prefill = runner.prefill_topk(&prompt_tokens, 1, PrefillMode::Batched)?;
    let prefill_elapsed = prefill_start.elapsed();
    let first = *prefill
        .first()
        .context("DSV4 4096-token prefill returned no top-1 candidate")?;
    assert_eq!(runner.position(), prompt_len);
    assert!(first.logit.is_finite());

    eprintln!(
        "DSV4 long-context prefill complete: layers={max_layers} tokens={prompt_len} elapsed={prefill_elapsed:?}"
    );
    let counters = runner.operator_runtime_counters();
    eprintln!(
        "DSV4 long-context counters: kernels={} allocations={} allocation_bytes={} h2d_copies={} h2d_bytes={} d2h_copies={} d2h_bytes={} expert_loads={} expert_load_bytes={} selected_experts={}",
        counters.kernel_launches,
        counters.device_allocations,
        counters.device_allocation_bytes,
        counters.host_to_device_copies,
        counters.host_to_device_bytes,
        counters.device_to_host_copies,
        counters.device_to_host_bytes,
        counters.expert_loads,
        counters.expert_load_bytes,
        counters.expert_selected,
    );
    eprintln!(
        "DSV4 long-context MoE counters: calls={} tc_calls={} total_us={} pointer_upload_us={} input_prepare_us={} gate_up_us={} swiglu_us={} down_us={} router_us={} routing_us={} plan_us={} cache_lookup_us={} expert_read_us={} expert_upload_us={} shared_us={} workspace_us={} compute_submit_us={} commit_us={} planner_syncs={}",
        counters.moe_calls,
        counters.moe_tc_calls,
        counters.moe_total_us,
        counters.moe_pointer_upload_us,
        counters.moe_input_prepare_us,
        counters.moe_gate_up_us,
        counters.moe_swiglu_us,
        counters.moe_down_us,
        counters.moe_router_us,
        counters.moe_routing_us,
        counters.moe_plan_us,
        counters.moe_cache_lookup_us,
        counters.moe_expert_read_us,
        counters.moe_expert_upload_us,
        counters.moe_shared_us,
        counters.moe_workspace_us,
        counters.moe_compute_submit_us,
        counters.moe_commit_us,
        counters.expert_planner_residency_syncs,
    );
    eprintln!(
        "DSV4 long-context layer profiles: {:?}",
        runner.layer_profile_stats()
    );
    eprintln!(
        "DSV4 long-context attention profiles: {:?}",
        runner.attention_profile_stats()
    );

    let decode = runner.decode_topk(first.token_id, 1)?;
    let second = *decode
        .first()
        .context("DSV4 position-4096 decode returned no top-1 candidate")?;
    assert_eq!(runner.position(), prompt_len + 1);
    assert!(second.logit.is_finite());
    eprintln!(
        "DSV4 >4096 continuation candidate: [{}, {}], logits=[{}, {}]",
        first.token_id, second.token_id, first.logit, second.logit
    );
    Ok(())
}

#[test]
#[ignore = "requires cargo-oxide, CUDA, and a local DeepSeek-V4-Flash-DSpark artifact"]
fn deepseek_v4_cuda_l0_prefill_parity_local() -> Result<()> {
    let model_dir = local_dsv4_model_dir();
    if !model_dir.is_dir() {
        eprintln!(
            "skipping local DSV4 L0 parity test; model dir not found: {}",
            model_dir.display()
        );
        return Ok(());
    }

    let options = DeepSeekV4PrepareOptions {
        max_layers: 1,
        ..DeepSeekV4PrepareOptions::default()
    };
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 128 * 1024 * 1024)
        .with_context(|| format!("load DSV4 artifact from {}", model_dir.display()))?;
    let mut runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;

    let prompt = ChatTemplate::DeepSeekV4.format_turn("Hello", true);
    let prompt_tokens = runner.encode(&prompt)?;
    assert!(
        prompt_tokens.len() > 1,
        "L0 parity regression requires the true batched prefill path"
    );

    let batched = runner.prefill_batched_layer_hc_trace(&prompt_tokens)?;
    let batched_checkpoints = runner.take_parity_checkpoints();
    runner.reset()?;
    let token_loop = runner.prefill_token_loop_layer_hc_trace(&prompt_tokens)?;
    let token_checkpoints = runner.take_parity_checkpoints();
    eprintln!(
        "DSV4 L0 checkpoint keys: batched={:?} token={:?}",
        batched_checkpoints.keys().collect::<Vec<_>>(),
        token_checkpoints.keys().collect::<Vec<_>>()
    );
    for (stage, batched_values) in &batched_checkpoints {
        if let Some(token_values) = token_checkpoints.get(stage) {
            let max_abs_diff = batched_values
                .iter()
                .zip(token_values)
                .map(|(batched, token)| (batched - token).abs())
                .fold(0.0f32, f32::max);
            eprintln!("DSV4 L0 checkpoint {stage}: max_abs_diff={max_abs_diff:e}");
        }
    }
    assert_eq!(batched.len(), 1);
    assert_eq!(token_loop.len(), 1);
    assert_eq!(batched[0].len(), token_loop[0].len());

    let mut max_abs_diff = 0.0f32;
    for (index, (batched, token_loop)) in batched[0].iter().zip(&token_loop[0]).enumerate() {
        assert!(
            batched.is_finite(),
            "L0 batched parity value at element {index} is non-finite: {batched}"
        );
        assert!(
            token_loop.is_finite(),
            "L0 token-loop parity value at element {index} is non-finite: {token_loop}"
        );
        let abs_diff = (batched - token_loop).abs();
        assert!(
            abs_diff.is_finite(),
            "L0 parity difference at element {index} is non-finite: batched={batched} token_loop={token_loop}"
        );
        max_abs_diff = max_abs_diff.max(abs_diff);
    }
    assert_eq!(
        max_abs_diff.to_bits(),
        0.0f32.to_bits(),
        "L0 batched/token-loop max_abs_diff={max_abs_diff:e} is not bit-exact"
    );

    runner.reset()?;
    runner.prefill_topk(&prompt_tokens, 1, PrefillMode::Batched)?;
    runner.reset()?;
    let warm = runner.operator_runtime_counters();
    runner.prefill_topk(&prompt_tokens, 1, PrefillMode::Batched)?;
    let counters = runner.operator_runtime_counters();
    eprintln!(
        "DSV4 warm prefill delta: allocations={} attempts={} bytes={} uploads={} h2d={}B d2h={}B expert_loads={} expert_load_bytes={} expert_evictions={} arena_hits={} arena_misses={} arena_grows={}",
        counters.device_allocations.saturating_sub(warm.device_allocations),
        counters
            .device_allocation_attempts
            .saturating_sub(warm.device_allocation_attempts),
        counters
            .device_allocation_bytes
            .saturating_sub(warm.device_allocation_bytes),
        counters.artifact_uploads.saturating_sub(warm.artifact_uploads),
        counters
            .host_to_device_bytes
            .saturating_sub(warm.host_to_device_bytes),
        counters
            .device_to_host_bytes
            .saturating_sub(warm.device_to_host_bytes),
        counters.expert_loads.saturating_sub(warm.expert_loads),
        counters
            .expert_load_bytes
            .saturating_sub(warm.expert_load_bytes),
        counters.expert_evictions.saturating_sub(warm.expert_evictions),
        counters.arena_hits.saturating_sub(warm.arena_hits),
        counters.arena_misses.saturating_sub(warm.arena_misses),
        counters.arena_grows.saturating_sub(warm.arena_grows),
    );
    assert_eq!(
        counters
            .device_allocation_attempts
            .saturating_sub(warm.device_allocation_attempts),
        0,
        "second warm prefill must make no device allocation attempts"
    );
    assert_eq!(
        counters
            .device_allocations
            .saturating_sub(warm.device_allocations),
        0,
        "second warm prefill must make no device allocations"
    );
    assert_eq!(
        counters
            .device_allocation_bytes
            .saturating_sub(warm.device_allocation_bytes),
        0,
        "second warm prefill must allocate no device bytes"
    );
    assert_eq!(
        counters
            .artifact_uploads
            .saturating_sub(warm.artifact_uploads),
        0,
        "second warm prefill must upload no artifacts"
    );
    assert_eq!(
        counters.expert_loads.saturating_sub(warm.expert_loads),
        0,
        "second same-prompt prefill must load no dynamic experts"
    );
    assert_eq!(
        counters
            .expert_load_bytes
            .saturating_sub(warm.expert_load_bytes),
        0,
        "second same-prompt prefill must load no dynamic expert bytes"
    );
    assert_eq!(
        counters.arena_misses.saturating_sub(warm.arena_misses),
        0,
        "second warm prefill must have no arena misses"
    );
    assert_eq!(
        counters.arena_grows.saturating_sub(warm.arena_grows),
        0,
        "second warm prefill must have no arena grows"
    );

    Ok(())
}

fn local_dsv4_model_dir() -> PathBuf {
    std::env::var_os("FERRULE_DSV4_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("models/DeepSeek-V4-Flash-DSpark")
        })
}
