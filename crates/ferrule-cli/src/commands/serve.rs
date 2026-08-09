use std::net::SocketAddr;

use std::time::Duration;

use anyhow::Context as _;
use ferrule_model::AutoConfig;
use ferrule_runtime::engine::model_factory::ExpertCacheOptions;
use ferrule_runtime::{
    BackendSelection, ModelFactoryOptions, ResidentModelPlanner, ResidentSchedulerConfig,
};
use ferrule_server::{
    ModelRegistration, ServerState, WorkerConfig, serve_with_shutdown, spawn_model_worker_with,
};

use crate::args::ServeArgs;

use super::resident::resident_driver_config;

pub fn cmd_serve(args: ServeArgs) -> anyhow::Result<()> {
    validate_args(&args)?;
    let config = AutoConfig::from_pretrained(&args.model)?;

    let scheduler_config = ResidentSchedulerConfig {
        prefill_chunk_size: args.prefill_chunk_size,
        max_active_sequences: args.max_active_sequences,
        max_decode_batch: args.max_active_sequences,
        decode_cohort_target: args.decode_cohort_target,
        decode_cohort_max_deferrals: args.decode_cohort_max_deferrals,
        max_batch_tokens: args.max_batch_tokens,
        ..ResidentSchedulerConfig::default()
    };
    let driver_config = resident_driver_config(args.ctx_size, true);
    let backend = args
        .backend
        .as_deref()
        .map(BackendSelection::parse)
        .transpose()?
        .unwrap_or_default();
    let prepared = ResidentModelPlanner::new().prepare(
        &config,
        backend,
        args.chat_template.as_deref(),
        ModelFactoryOptions {
            max_layers: args.max_layers,
            max_tensor_mebibytes: args.max_tensor_mb,
            output_head_chunk_rows: args.output_head_chunk_rows,
            expert_reader_max_tensor_mebibytes: args.expert_reader_max_slice_mb,
            expert_cache: ExpertCacheOptions {
                host_entries: args.expert_host_cache_entries,
                host_mebibytes: args.expert_host_cache_mb,
                pinned_entries: args.expert_pinned_cache_entries,
                pinned_mebibytes: args.expert_pinned_cache_mb,
            },
            moe_hotset_experts: args.moe_hotset_experts,
            kv_cache_mebibytes: Some(args.kv_cache_mb),
            scheduler_config,
            driver_config,
        },
    )?;
    let adapter_name = prepared.model_name();
    let backend_name = prepared.backend().as_str();
    let backend_profile = prepared.backend_profile();
    let chat_template = prepared.chat_template();
    let served_model_name = args
        .served_model_name
        .clone()
        .unwrap_or_else(|| adapter_name.to_owned());
    let worker_config = WorkerConfig {
        command_queue_capacity: args.request_queue_capacity,
        event_queue_capacity: args.event_queue_capacity,
        admission_timeout: Duration::from_secs(args.admission_timeout_secs),
        ..WorkerConfig::default()
    };

    eprintln!(
        "loading {served_model_name} with {adapter_name} on {backend_profile} ({backend_name} backend) in the dedicated model worker...",
    );
    let worker = spawn_model_worker_with(move || prepared.build(), worker_config)
        .context("failed to start model worker")?;

    let address = SocketAddr::new(args.host, args.port);
    let state = ServerState::new(
        ModelRegistration::new(served_model_name, chat_template),
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
        shutdown_result.context("failed to shut down model worker")
    })
}

fn validate_args(args: &ServeArgs) -> anyhow::Result<()> {
    if args
        .served_model_name
        .as_deref()
        .is_some_and(|name| name.trim().is_empty())
    {
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
    if args.kv_cache_mb == 0 {
        anyhow::bail!("kv-cache-mb must be greater than zero");
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
    if args.max_layers == Some(0) {
        anyhow::bail!("max-layers must be greater than zero");
    }
    Ok(())
}
