#![cfg(all(feature = "cuda", feature = "local-dsv4-tests"))]

use std::path::PathBuf;

use anyhow::{Context, Result};
use ferrule_model::{
    models::deepseek_v4::{
        DeepSeekV4ArtifactModel, DeepSeekV4OperatorBackend, DeepSeekV4ReferenceOptions,
        DeepSeekV4ReferenceRunner,
    },
    ChatTemplate, ModelRunner, PrefillMode,
};
use ferrule_runtime::{
    GenerateRequest, PagedSequenceKvCache, RequestId, ResidentActionExecutorConfig,
    ResidentSchedulerConfig, ResidentTopKDriver, ResidentTopKDriverConfig, SamplingConfig,
    SequenceFinishReason, SessionId,
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

    let options = DeepSeekV4ReferenceOptions {
        max_layers: 43,
        output_head_chunk_rows: 4096,
        moe_prefetch_experts: 8,
        moe_hotset_experts: 0,
        ..DeepSeekV4ReferenceOptions::default()
    };
    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 128 * 1024 * 1024)
        .with_context(|| format!("load DSV4 artifact from {}", model_dir.display()))?;
    let runner = DeepSeekV4ReferenceRunner::new_with_operator_backend(
        model,
        options,
        DeepSeekV4OperatorBackend::Cuda,
    )?;

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
        ResidentActionExecutorConfig {
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
    assert_eq!(driver.executor().position(), sequence.position);

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
