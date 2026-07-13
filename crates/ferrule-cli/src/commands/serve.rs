use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use ferrule_model::{
    ChatTemplate, ModelDescriptor, ModelExecutionBackend, ModelFamily,
    models::deepseek_v4::{DeepSeekV4PrepareOptions, DeepSeekV4Runner},
};
use ferrule_runtime::{ResidentSchedulerConfig, ResidentTopKDriverConfig};
use ferrule_server::{
    ModelRegistration, ServerState, WorkerConfig, serve_with_shutdown, spawn_model_worker_with,
};

use crate::args::ServeArgs;

use super::resident::build_resident_topk_driver;

pub fn cmd_serve(args: ServeArgs) -> anyhow::Result<()> {
    validate_args(&args)?;
    let model_path = Path::new(&args.model);
    let descriptor = ModelDescriptor::load(model_path)?;
    if !matches!(descriptor.spec.family, ModelFamily::DeepSeekV4) {
        anyhow::bail!(
            "serve bootstrap currently supports DeepSeek-V4; the HTTP and worker layers are model-neutral"
        );
    }
    let chat_template = match args.chat_template.as_deref() {
        Some(name) => ChatTemplate::from_name(name)
            .ok_or_else(|| anyhow::anyhow!("unknown chat template '{name}'"))?,
        None => ChatTemplate::DeepSeekV4,
    };
    let backend = ModelExecutionBackend::parse(&args.backend)?;
    if matches!(backend, ModelExecutionBackend::Cuda) && !cfg!(feature = "cuda") {
        anyhow::bail!("CUDA serving requires building ferrule-cli with --features cuda");
    }

    let model_path = PathBuf::from(&args.model);
    let prepare_options = DeepSeekV4PrepareOptions {
        max_layers: args.max_layers,
        output_head_chunk_rows: args.output_head_chunk_rows,
        expert_reader_max_tensor_bytes: args.expert_reader_max_slice_mb.saturating_mul(1024 * 1024),
        moe_prefetch_experts: args.moe_prefetch_experts,
        moe_hotset_experts: args.moe_hotset_experts,
    };
    let max_tensor_bytes = args.max_tensor_mb.saturating_mul(1024 * 1024);
    let scheduler_config = ResidentSchedulerConfig {
        prefill_chunk_size: args.prefill_chunk_size,
        max_active_sequences: args.max_active_sequences,
        max_decode_batch: args.max_active_sequences,
        max_batch_tokens: args.max_batch_tokens,
        allow_mixed_batches: true,
    };
    let driver_config = ResidentTopKDriverConfig {
        ctx_size: args.ctx_size,
        stop_at_eos: true,
        append_eos_to_session: false,
        // The serving worker executes exactly one step at a time; this remains a
        // safety bound only for callers that explicitly use run_until_blocked.
        max_steps_per_run: args.ctx_size.saturating_mul(2).max(1024),
    };
    let worker_config = WorkerConfig {
        command_queue_capacity: args.request_queue_capacity,
        event_queue_capacity: args.event_queue_capacity,
        admission_timeout: Duration::from_secs(args.admission_timeout_secs),
        ..WorkerConfig::default()
    };

    eprintln!(
        "loading {} on {} in the dedicated model worker...",
        args.served_model_name,
        backend.as_str()
    );
    let worker = spawn_model_worker_with(
        move || {
            let runner = DeepSeekV4Runner::load_hf_with_options_and_backend(
                &model_path,
                max_tensor_bytes,
                prepare_options,
                backend,
            )
            .map_err(|error| error.to_string())?;
            let schema = runner.kv_layout_schema().clone();
            build_resident_topk_driver(runner, Box::new(schema), scheduler_config, driver_config)
                .map_err(|error| error.to_string())
        },
        worker_config,
    )
    .map_err(anyhow::Error::msg)?;

    let address = SocketAddr::new(args.host, args.port);
    let state = ServerState::new(
        ModelRegistration::new(args.served_model_name.clone(), chat_template),
        worker.handle(),
    );
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("ferrule-http")
        .build()?;
    runtime.block_on(async move {
        eprintln!("Ferrule OpenAI API listening on http://{address}");
        eprintln!("  GET  /health");
        eprintln!("  GET  /v1/models");
        eprintln!("  POST /v1/chat/completions");
        eprintln!("  POST /v1/completions");
        let server_result = serve_with_shutdown(address, state, async {
            if let Err(error) = tokio::signal::ctrl_c().await {
                tracing::error!(%error, "failed to install Ctrl-C handler");
            }
        })
        .await;
        let shutdown_result = worker.shutdown().await;
        server_result?;
        shutdown_result.map_err(anyhow::Error::msg)
    })
}

fn validate_args(args: &ServeArgs) -> anyhow::Result<()> {
    if args.served_model_name.trim().is_empty() {
        anyhow::bail!("served model name must not be empty");
    }
    if args.ctx_size == 0 {
        anyhow::bail!("ctx-size must be greater than zero");
    }
    if args.max_active_sequences == 0 {
        anyhow::bail!("max-active-sequences must be greater than zero");
    }
    if args.prefill_chunk_size == 0 {
        anyhow::bail!("prefill-chunk-size must be greater than zero");
    }
    if args.max_batch_tokens == 0 {
        anyhow::bail!("max-batch-tokens must be greater than zero");
    }
    if args.request_queue_capacity == 0 {
        anyhow::bail!("request-queue-capacity must be greater than zero");
    }
    if args.event_queue_capacity < 2 {
        anyhow::bail!("event-queue-capacity must be at least two");
    }
    if args.admission_timeout_secs == 0 {
        anyhow::bail!("admission-timeout-secs must be greater than zero");
    }
    Ok(())
}
