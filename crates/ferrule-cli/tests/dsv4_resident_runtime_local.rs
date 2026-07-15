#![cfg(all(feature = "cuda", feature = "local-dsv4-tests"))]

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use ferrule_common::execution::ForwardPhase;
use ferrule_model::{
    ChatTemplate, ModelExecutionBackend, ModelRunner, PrefillMode, TopKModelRunner,
    models::deepseek_v4::{DeepSeekV4ArtifactModel, DeepSeekV4PrepareOptions, DeepSeekV4Runner},
};
use ferrule_runtime::cache::KvPageManager;
use ferrule_runtime::{
    FixedSequenceSlotPool, GenerateRequest, PageManagedDiagnosticHarness, RequestId,
    ResidentSchedulerConfig, ResidentTopKDriver, ResidentTopKDriverConfig, SequenceFinishReason,
    SessionId,
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

    let schema = runner.kv_layout_schema().clone();
    let max_pages = 4096usize.div_ceil(schema.page_size());
    let mut driver = ResidentTopKDriver::with_configs(
        runner,
        FixedSequenceSlotPool::new(1),
        ResidentSchedulerConfig {
            prefill_chunk_size: 4096,
            max_active_sequences: 1,
            max_decode_batch: 1,
            ..Default::default()
        },
        std::num::NonZeroU32::new(1).unwrap(),
        ResidentTopKDriverConfig {
            ctx_size: 4096,
            stop_at_eos: true,
            append_eos_to_session: true,
            max_steps_per_run: 4096 + 16,
        },
    )
    .try_with_page_manager(KvPageManager::new(Box::new(schema), max_pages))?;

    driver.submit(GenerateRequest {
        id: RequestId(1),
        session_id: Some(SessionId(0)),
        prompt_tokens: prompt_tokens.clone(),
        max_new_tokens: 1,
        stop: Vec::new(),
        ignore_eos: false,
    });

    let mut emitted = Vec::new();
    let stats = driver.run_until_blocked(|event| {
        emitted.push(event.token);
        Ok(())
    })?;

    assert_eq!(emitted.len(), 1, "expected exactly one emitted token");
    assert_eq!(stats.emitted_tokens, 1);
    assert_eq!(stats.finished_sequences, 1);
    assert_eq!(driver.slot_pool().active_count(), 0);

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
        driver.executor().runner().position(),
        0,
        "native driver execution must not mutate the runner's default sequence"
    );

    Ok(())
}

#[test]
#[ignore = "requires cargo-oxide, CUDA, and a local DeepSeek-V4-Flash-DSpark artifact"]
fn deepseek_v4_repeated_sequence_reuses_runtime_expert_residency_local() -> Result<()> {
    let model_dir = local_dsv4_model_dir();
    if !model_dir.is_dir() {
        return Ok(());
    }
    let options = DeepSeekV4PrepareOptions {
        max_layers: 1,
        output_head_chunk_rows: 4096,
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 128 * 1024 * 1024)?;
    let runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;
    let prompt = runner.encode(&ChatTemplate::DeepSeekV4.format_turn("Hello", true))?;
    let prompt = &prompt[..prompt.len().min(5)];
    let schema = runner.kv_layout_schema().clone();
    let mut harness = PageManagedDiagnosticHarness::new(runner, Box::new(schema), 4096, 2)?;

    let first = harness.create_sequence(0)?;
    harness.execute_sequence_step(first, ForwardPhase::Prefill, prompt, |runner| {
        runner.prefill_topk(prompt, 1, PrefillMode::Batched)
    })?;
    let after_first = harness.runner().operator_runtime_counters();
    assert!(after_first.expert_residency_stats.resident > 0);
    assert!(after_first.expert_load_bytes > 0);
    assert_eq!(after_first.expert_residency_stats.active_leases, 0);

    let second = harness.create_sequence(1)?;
    harness.execute_sequence_step(second, ForwardPhase::Prefill, prompt, |runner| {
        runner.prefill_topk(prompt, 1, PrefillMode::Batched)
    })?;
    let after_second = harness.runner().operator_runtime_counters();
    assert!(
        after_second.expert_residency_stats.resident_hits
            > after_first.expert_residency_stats.resident_hits
    );
    assert_eq!(
        after_second.expert_load_bytes,
        after_first.expert_load_bytes
    );
    assert_eq!(after_second.expert_loads, after_first.expert_loads);
    assert_eq!(after_second.expert_residency_stats.active_leases, 0);
    Ok(())
}

#[test]
#[ignore = "requires cargo-oxide, CUDA, and a local DeepSeek-V4-Flash-DSpark artifact"]
fn deepseek_v4_native_packed_decode_batch2_and_batch4_are_exact_local() -> Result<()> {
    let model_dir = local_dsv4_model_dir();
    if !model_dir.is_dir() {
        return Ok(());
    }
    let options = DeepSeekV4PrepareOptions {
        max_layers: 1,
        output_head_chunk_rows: 4096,
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 128 * 1024 * 1024)?;
    let runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;
    let prompt = runner.encode(&ChatTemplate::DeepSeekV4.format_turn("Hello", true))?;
    let schema = runner.kv_layout_schema().clone();
    let mut serial_harness =
        PageManagedDiagnosticHarness::new(runner, Box::new(schema.clone()), 4096, 1)?;
    let serial_slot = serial_harness.create_sequence(0)?;
    let serial_first = serial_harness.execute_sequence_step(
        serial_slot,
        ForwardPhase::Prefill,
        &prompt,
        |runner| runner.prefill_topk(&prompt, 1, PrefillMode::Batched),
    )?[0];
    let serial_second = serial_harness.execute_sequence_step(
        serial_slot,
        ForwardPhase::Decode,
        &[serial_first.token_id],
        |runner| runner.decode_topk(serial_first.token_id, 1),
    )?[0];
    let serial = [serial_first, serial_second];
    let runner = serial_harness.into_runner()?;
    let pages_per_sequence = 4096usize.div_ceil(schema.page_size());
    let mut driver = ResidentTopKDriver::with_configs(
        runner,
        FixedSequenceSlotPool::new(4),
        ResidentSchedulerConfig {
            prefill_chunk_size: 4096,
            max_active_sequences: 4,
            max_decode_batch: 4,
            max_batch_tokens: 4096,
            allow_mixed_batches: true,
        },
        NonZeroU32::new(1).unwrap(),
        ResidentTopKDriverConfig {
            ctx_size: 4096,
            stop_at_eos: false,
            append_eos_to_session: false,
            max_steps_per_run: 4096,
        },
    )
    .try_with_page_manager(KvPageManager::new(Box::new(schema), pages_per_sequence * 4))?;
    let mut next_id = 1u64;
    let mut paged_serial_launches = None;
    let mut paged_serial_h2d = None;
    let mut paged_serial_elapsed = None;
    for (wave, batch_size) in [1usize, 2, 4, 4].into_iter().enumerate() {
        let session_ids = (next_id..next_id + batch_size as u64)
            .map(SessionId)
            .collect::<Vec<_>>();
        for session_id in &session_ids {
            driver.submit(GenerateRequest {
                id: RequestId(session_id.0),
                session_id: Some(*session_id),
                prompt_tokens: prompt.clone(),
                max_new_tokens: 2,
                stop: Vec::new(),
                ignore_eos: false,
            });
        }
        let mut events = Vec::new();
        driver.step(&mut |event| {
            events.push((event.session_id, event.index, event.token, event.logit));
            Ok(())
        })?;

        let packed_before = driver.executor().runner().operator_runtime_counters();
        let decode_start = Instant::now();
        driver.step(&mut |event| {
            events.push((event.session_id, event.index, event.token, event.logit));
            Ok(())
        })?;

        let decode_elapsed = decode_start.elapsed();
        let packed_after = driver.executor().runner().operator_runtime_counters();
        let packed_launches = packed_after
            .kernel_launches
            .saturating_sub(packed_before.kernel_launches);
        let packed_h2d = packed_after
            .host_to_device_copies
            .saturating_sub(packed_before.host_to_device_copies);
        if batch_size == 1 {
            paged_serial_launches = Some(packed_launches);
            paged_serial_h2d = Some(packed_h2d);
            paged_serial_elapsed = Some(decode_elapsed);
        } else {
            let serial_launches = paged_serial_launches.expect("batch1 launch baseline exists");
            let serial_h2d = paged_serial_h2d.expect("batch1 H2D baseline exists");
            assert!(
                packed_launches < serial_launches.saturating_mul(batch_size as u64),
                "packed batch {batch_size} launches={packed_launches} did not beat paged serial launches={serial_launches} per row"
            );
            assert!(
                packed_h2d < serial_h2d.saturating_mul(batch_size as u64),
                "packed batch {batch_size} H2D copies={packed_h2d} did not beat paged serial copies={serial_h2d} per row"
            );
        }
        if wave == 3 {
            assert_eq!(
                packed_after
                    .device_allocations
                    .saturating_sub(packed_before.device_allocations),
                0,
                "warm batch-4 decode allocated device memory"
            );
            let serial_elapsed = paged_serial_elapsed.expect("batch1 timing baseline exists");
            assert!(
                decode_elapsed < serial_elapsed.saturating_mul(batch_size as u32),
                "warm packed batch-4 elapsed {decode_elapsed:?} did not exceed paged serial aggregate throughput ({serial_elapsed:?} per row)"
            );
        }
        driver.run_until_blocked(|event| {
            events.push((event.session_id, event.index, event.token, event.logit));
            Ok(())
        })?;
        for session_id in &session_ids {
            let session_events = events
                .iter()
                .filter(|event| event.0 == *session_id)
                .collect::<Vec<_>>();
            assert_eq!(session_events.len(), serial.len());
            for (index, (actual, expected)) in session_events.iter().zip(serial).enumerate() {
                assert_eq!(actual.1, index);
                assert_eq!(actual.2, expected.token_id);
                assert_eq!(actual.3.map(f32::to_bits), Some(expected.logit.to_bits()));
            }
        }
        let mut finished = driver.drain_finished();
        finished.sort_by_key(|sequence| sequence.session_id);
        assert_eq!(finished.len(), batch_size);
        for sequence in &finished[1..] {
            assert_eq!(sequence.generated_text, finished[0].generated_text);
            assert_eq!(sequence.tokens, finished[0].tokens);
        }
        next_id += batch_size as u64;
    }
    let packed = driver.executor().runner().output_profile_stats();
    assert!(packed.packed_decode_batches >= 3);
    assert!(packed.packed_decode_rows >= 10);
    assert_eq!(driver.executor().runner().position(), 0);
    Ok(())
}

#[test]
#[ignore = "requires cargo-oxide, CUDA, and a local DeepSeek-V4-Flash-DSpark artifact"]
fn deepseek_v4_native_ragged_prefill_and_mixed_are_exact_local() -> Result<()> {
    let model_dir = local_dsv4_model_dir();
    if !model_dir.is_dir() {
        return Ok(());
    }
    let options = DeepSeekV4PrepareOptions {
        max_layers: 1,
        output_head_chunk_rows: 4096,
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 128 * 1024 * 1024)?;
    let runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;
    let prompt = runner.encode(&ChatTemplate::DeepSeekV4.format_turn("Hello", true))?;
    if prompt.len() < 5 {
        return Ok(());
    }
    let prompts = [prompt[..2].to_vec(), prompt[..5].to_vec()];
    let schema = runner.kv_layout_schema().clone();
    let mut serial_harness =
        PageManagedDiagnosticHarness::new(runner, Box::new(schema.clone()), 4096, 2)?;
    let mut expected = Vec::new();
    for (generation, prompt) in prompts.iter().enumerate() {
        let slot = serial_harness.create_sequence(generation as u64)?;
        let first = serial_harness.execute_sequence_step(
            slot,
            ForwardPhase::Prefill,
            prompt,
            |runner| runner.prefill_topk(prompt, 1, PrefillMode::Batched),
        )?[0];
        let second = serial_harness.execute_sequence_step(
            slot,
            ForwardPhase::Decode,
            &[first.token_id],
            |runner| runner.decode_topk(first.token_id, 1),
        )?[0];
        expected.push([first, second]);
    }
    let runner = serial_harness.into_runner()?;

    let pages_per_sequence = 4096usize.div_ceil(schema.page_size());
    let mut driver = ResidentTopKDriver::with_configs(
        runner,
        FixedSequenceSlotPool::new(2),
        ResidentSchedulerConfig {
            prefill_chunk_size: 2,
            max_active_sequences: 2,
            max_decode_batch: 2,
            max_batch_tokens: 4,
            allow_mixed_batches: true,
        },
        NonZeroU32::new(1).unwrap(),
        ResidentTopKDriverConfig {
            ctx_size: 4096,
            stop_at_eos: false,
            append_eos_to_session: false,
            max_steps_per_run: 32,
        },
    )
    .try_with_page_manager(KvPageManager::new(Box::new(schema), pages_per_sequence * 2))?;
    for (index, prompt) in prompts.iter().enumerate() {
        driver.submit(GenerateRequest {
            id: RequestId((index + 1) as u64),
            session_id: Some(SessionId((index + 1) as u64)),
            prompt_tokens: prompt.clone(),
            max_new_tokens: 2,
            stop: Vec::new(),
            ignore_eos: false,
        });
    }
    let mut events = Vec::new();
    driver.run_until_blocked(|event| {
        events.push((event.session_id, event.index, event.token, event.logit));
        Ok(())
    })?;
    for (sequence, expected_rows) in expected.iter().enumerate() {
        let session = SessionId((sequence + 1) as u64);
        let actual = events
            .iter()
            .filter(|event| event.0 == session)
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected_rows.len());
        for (index, (actual, expected)) in actual.iter().zip(expected_rows).enumerate() {
            assert_eq!(actual.1, index);
            assert_eq!(actual.2, expected.token_id);
            assert_eq!(actual.3.map(f32::to_bits), Some(expected.logit.to_bits()));
        }
    }
    let profile = driver.executor().runner().output_profile_stats();
    assert!(profile.packed_prefill_batches > 0);
    assert!(profile.packed_prefill_rows >= 4);
    assert!(profile.packed_mixed_batches > 0);
    assert!(profile.packed_mixed_rows >= 2);
    Ok(())
}

#[test]
#[ignore = "requires cargo-oxide, CUDA, and a local DeepSeek-V4-Flash-DSpark artifact"]
fn deepseek_v4_exact_prefix_fork_cow_matches_full_serial_local() -> Result<()> {
    let model_dir = local_dsv4_model_dir();
    if !model_dir.is_dir() {
        return Ok(());
    }
    let options = DeepSeekV4PrepareOptions {
        max_layers: 1,
        output_head_chunk_rows: 4096,
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 128 * 1024 * 1024)?;
    let runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;
    let prompt = runner.encode(&ChatTemplate::DeepSeekV4.format_turn("Hello", true))?;
    if prompt.len() < 5 {
        return Ok(());
    }
    let prefix = prompt[..3].to_vec();
    let suffix = prompt[3..5].to_vec();
    let schema = runner.kv_layout_schema().clone();
    let mut serial_harness =
        PageManagedDiagnosticHarness::new(runner, Box::new(schema.clone()), 4096, 1)?;
    let baseline_slot = serial_harness.create_sequence(0)?;
    let first = serial_harness.execute_sequence_step(
        baseline_slot,
        ForwardPhase::Prefill,
        &prompt[..5],
        |runner| runner.prefill_topk(&prompt[..5], 1, PrefillMode::Batched),
    )?[0];
    let second = serial_harness.execute_sequence_step(
        baseline_slot,
        ForwardPhase::Decode,
        &[first.token_id],
        |runner| runner.decode_topk(first.token_id, 1),
    )?[0];
    let runner = serial_harness.into_runner()?;

    let pages_per_sequence = 4096usize.div_ceil(schema.page_size());
    let mut driver = ResidentTopKDriver::with_configs(
        runner,
        FixedSequenceSlotPool::new(2),
        ResidentSchedulerConfig {
            prefill_chunk_size: 4096,
            max_active_sequences: 2,
            max_decode_batch: 2,
            max_batch_tokens: 4096,
            allow_mixed_batches: true,
        },
        NonZeroU32::new(1).unwrap(),
        ResidentTopKDriverConfig {
            ctx_size: 4096,
            stop_at_eos: false,
            append_eos_to_session: false,
            max_steps_per_run: 32,
        },
    )
    .try_with_page_manager(KvPageManager::new(Box::new(schema), pages_per_sequence * 2))?;
    let source = SessionId(1);
    driver.submit(GenerateRequest {
        id: RequestId(1),
        session_id: Some(source),
        prompt_tokens: prefix.clone(),
        max_new_tokens: 2,
        stop: Vec::new(),
        ignore_eos: false,
    });
    driver.step(&mut |_| Ok(()))?;
    let pages_before_fork = driver.page_manager().unwrap().allocated_pages();
    let target = driver.fork_session_exact(
        source,
        GenerateRequest {
            id: RequestId(2),
            session_id: Some(SessionId(2)),
            prompt_tokens: suffix,
            max_new_tokens: 2,
            stop: Vec::new(),
            ignore_eos: false,
        },
        prefix.len(),
    )?;
    assert_eq!(target, SessionId(2));
    let fork_stats = driver.page_manager().unwrap().stats();
    assert_eq!(fork_stats.allocated_pages, pages_before_fork);
    assert!(fork_stats.shared_pages > 0);

    let mut target_events = Vec::new();
    driver.step(&mut |event| {
        if event.session_id == target {
            target_events.push((event.index, event.token, event.logit));
        }
        Ok(())
    })?;
    assert!(
        driver.page_manager().unwrap().allocated_pages() > pages_before_fork,
        "live partial-tail suffix append must allocate a COW replacement page"
    );
    driver.run_until_blocked(|event| {
        if event.session_id == target {
            target_events.push((event.index, event.token, event.logit));
        }
        Ok(())
    })?;
    assert_eq!(target_events.len(), 2);
    for (actual, expected) in target_events.iter().zip([first, second]) {
        assert_eq!(actual.1, expected.token_id);
        assert_eq!(actual.2.map(f32::to_bits), Some(expected.logit.to_bits()));
    }
    assert_eq!(
        driver.page_manager().unwrap().allocated_pages(),
        pages_before_fork,
        "finished target must release its private COW page"
    );
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
    let runner =
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
    let schema = runner.kv_layout_schema().clone();
    let max_tokens = prompt_len
        .checked_add(1)
        .context("DSV4 long-context diagnostic token capacity overflow")?;
    let mut diagnostic =
        PageManagedDiagnosticHarness::new(runner, Box::new(schema), max_tokens, 1)?;
    let sequence = diagnostic.create_sequence(0)?;

    eprintln!("DSV4 long-context prefill start: layers={max_layers} tokens={prompt_len}");
    let prefill_start = Instant::now();
    let prefill = diagnostic.execute_sequence_step(
        sequence,
        ForwardPhase::Prefill,
        &prompt_tokens,
        |runner| runner.prefill_topk(&prompt_tokens, 1, PrefillMode::Batched),
    )?;
    let prefill_elapsed = prefill_start.elapsed();
    let first = *prefill
        .first()
        .context("DSV4 4096-token prefill returned no top-1 candidate")?;
    assert_eq!(diagnostic.sequence_state(sequence)?.position(), prompt_len);
    assert!(first.logit.is_finite());

    eprintln!(
        "DSV4 long-context prefill complete: layers={max_layers} tokens={prompt_len} elapsed={prefill_elapsed:?}"
    );
    let counters = diagnostic.runner().operator_runtime_counters();
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
        "DSV4 long-context MoE counters: calls={} tc_calls={} total_us={} input_prepare_us={} gate_up_us={} swiglu_us={} down_us={} router_us={} routing_us={} plan_us={} cache_lookup_us={} expert_read_us={} expert_upload_us={} shared_us={} workspace_us={} compute_submit_us={} commit_us={}",
        counters.moe_calls,
        counters.moe_tc_calls,
        counters.moe_total_us,
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
    );
    eprintln!(
        "DSV4 long-context layer profiles: {:?}",
        diagnostic.runner().layer_profile_stats()
    );
    eprintln!(
        "DSV4 long-context attention profiles: {:?}",
        diagnostic.runner().attention_profile_stats()
    );

    let decode = diagnostic.execute_sequence_step(
        sequence,
        ForwardPhase::Decode,
        &[first.token_id],
        |runner| runner.decode_topk(first.token_id, 1),
    )?;
    let second = *decode
        .first()
        .context("DSV4 position-4096 decode returned no top-1 candidate")?;
    assert_eq!(
        diagnostic.sequence_state(sequence)?.position(),
        prompt_len + 1
    );
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
    let runner =
        DeepSeekV4Runner::new_with_operator_backend(model, options, ModelExecutionBackend::Cuda)?;

    let prompt = ChatTemplate::DeepSeekV4.format_turn("Hello", true);
    let prompt_tokens = runner.encode(&prompt)?;
    assert!(
        prompt_tokens.len() > 1,
        "L0 parity regression requires the true batched prefill path"
    );

    let schema = runner.kv_layout_schema().clone();
    let mut diagnostic =
        PageManagedDiagnosticHarness::new(runner, Box::new(schema), prompt_tokens.len(), 4)?;
    let batched_sequence = diagnostic.create_sequence(0)?;
    let token_loop_sequence = diagnostic.create_sequence(0)?;
    let batched = diagnostic.execute_sequence_step(
        batched_sequence,
        ForwardPhase::Prefill,
        &prompt_tokens,
        |runner| runner.prefill_batched_layer_hc_trace(&prompt_tokens),
    )?;
    let batched_checkpoints = diagnostic.runner_mut().take_parity_checkpoints();
    for &token_id in &prompt_tokens[..prompt_tokens.len() - 1] {
        diagnostic.execute_sequence_step(
            token_loop_sequence,
            ForwardPhase::Decode,
            std::slice::from_ref(&token_id),
            |runner| runner.feed_token(token_id),
        )?;
    }
    let last_token = prompt_tokens[prompt_tokens.len() - 1];
    let token_loop = diagnostic.execute_sequence_step(
        token_loop_sequence,
        ForwardPhase::Decode,
        std::slice::from_ref(&last_token),
        |runner| runner.decode_token_layer_hc_trace(last_token),
    )?;
    let token_checkpoints = diagnostic.runner_mut().take_parity_checkpoints();
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

    let first_warm_sequence = diagnostic.create_sequence(0)?;
    diagnostic.execute_sequence_step(
        first_warm_sequence,
        ForwardPhase::Prefill,
        &prompt_tokens,
        |runner| runner.prefill_topk(&prompt_tokens, 1, PrefillMode::Batched),
    )?;
    let warm = diagnostic.runner().operator_runtime_counters();
    let second_warm_sequence = diagnostic.create_sequence(0)?;
    diagnostic.execute_sequence_step(
        second_warm_sequence,
        ForwardPhase::Prefill,
        &prompt_tokens,
        |runner| runner.prefill_topk(&prompt_tokens, 1, PrefillMode::Batched),
    )?;
    let counters = diagnostic.runner().operator_runtime_counters();
    eprintln!(
        "DSV4 warm prefill delta: allocations={} attempts={} bytes={} uploads={} h2d={}B d2h={}B expert_loads={} expert_load_bytes={} expert_evictions={} arena_hits={} arena_misses={} arena_grows={}",
        counters
            .device_allocations
            .saturating_sub(warm.device_allocations),
        counters
            .device_allocation_attempts
            .saturating_sub(warm.device_allocation_attempts),
        counters
            .device_allocation_bytes
            .saturating_sub(warm.device_allocation_bytes),
        counters
            .artifact_uploads
            .saturating_sub(warm.artifact_uploads),
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
        counters
            .expert_evictions
            .saturating_sub(warm.expert_evictions),
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
