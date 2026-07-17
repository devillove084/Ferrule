use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use ferrule_common::MemoryPoolLimits;
use ferrule_common::execution::KvLayoutSchema;
use ferrule_model::{
    ChatTemplate, ExpertMemoryPolicy, ModelDescriptor, ModelExecutionBackend, ModelFamily,
    models::deepseek_v4::{DeepSeekV4PrepareOptions, DeepSeekV4Runner},
};
use ferrule_runtime::{ExpertIoBudget, ResidentSchedulerConfig, ResidentTopKDriverConfig};
use ferrule_server::{
    DeepSeekV4ResidentEngine, ModelRegistration, ServerState, WorkerConfig, serve_with_shutdown,
    spawn_model_worker_with,
};

use crate::args::ServeArgs;

use super::resident::build_resident_topk_driver_with_page_limit;

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
    let expert_memory_policy = ExpertMemoryPolicy::new(
        MemoryPoolLimits::new(
            args.expert_host_cache_entries,
            cache_byte_limit(args.expert_host_cache_mb, "expert-host-cache-mb")?,
        ),
        MemoryPoolLimits::new(
            args.expert_pinned_cache_entries,
            cache_byte_limit(args.expert_pinned_cache_mb, "expert-pinned-cache-mb")?,
        ),
    );
    let prepare_options = DeepSeekV4PrepareOptions {
        max_layers: args.max_layers,
        output_head_chunk_rows: args.output_head_chunk_rows,
        expert_reader_max_tensor_bytes: args.expert_reader_max_slice_mb.saturating_mul(1024 * 1024),
        moe_prefetch_experts: args.moe_prefetch_experts,
        expert_memory_policy,
        moe_hotset_experts: args.moe_hotset_experts,
    };
    let max_tensor_bytes = args.max_tensor_mb.saturating_mul(1024 * 1024);
    let kv_cache_bytes = required_mebibytes_to_bytes(args.kv_cache_mb, "kv-cache-mb")?;
    let expert_io_budget = ExpertIoBudget {
        max_incremental_expert_bytes: cache_byte_limit(
            args.expert_io_batch_mb,
            "expert-io-batch-mb",
        )?,
        ..ExpertIoBudget::unbounded()
    };
    let scheduler_config = ResidentSchedulerConfig {
        prefill_chunk_size: args.prefill_chunk_size,
        max_active_sequences: args.max_active_sequences,
        max_decode_batch: args.max_active_sequences,
        max_batch_tokens: args.max_batch_tokens,
        // DSpark decode owns a Q=1..6 transaction per ready sequence. Keep
        // prefill dispatch separate until cross-sequence Q packing is implemented.
        allow_mixed_batches: false,
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
            let page_bytes = schema
                .cuda_f32_data_page_bytes()
                .map_err(|error| error.to_string())?;
            let page_limit = page_limit_for_budget(kv_cache_bytes, page_bytes)
                .map_err(|error| error.to_string())?;
            let full_capacity_pages = schema
                .pages_for_tokens(driver_config.ctx_size)
                .checked_mul(scheduler_config.max_active_sequences)
                .ok_or_else(|| "serving KV page capacity overflow".to_owned())?;
            let configured_pages = full_capacity_pages.min(page_limit);
            let configured_bytes = u64::try_from(configured_pages)
                .ok()
                .and_then(|pages| pages.checked_mul(page_bytes))
                .ok_or_else(|| "serving KV byte estimate overflow".to_owned())?;
            eprintln!(
                "configuring CUDA KV pool: pages={configured_pages}/{full_capacity_pages}, page_bytes={page_bytes}, physical_budget={} MiB, allocated={} MiB",
                kv_cache_bytes / (1024 * 1024),
                configured_bytes / (1024 * 1024),
            );
            let driver = build_resident_topk_driver_with_page_limit(
                runner,
                Box::new(schema),
                scheduler_config,
                driver_config,
                Some(page_limit),
            )
            .map_err(|error| error.to_string())?;
            Ok(DeepSeekV4ResidentEngine::new(driver, expert_io_budget))
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

fn cache_byte_limit(mebibytes: u64, option: &str) -> anyhow::Result<u64> {
    if mebibytes == 0 {
        return Ok(u64::MAX);
    }
    mebibytes
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("{option} exceeds the supported byte range"))
}

fn required_mebibytes_to_bytes(mebibytes: u64, option: &str) -> anyhow::Result<u64> {
    if mebibytes == 0 {
        anyhow::bail!("{option} must be greater than zero");
    }
    mebibytes
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("{option} exceeds the supported byte range"))
}

fn page_limit_for_budget(budget_bytes: u64, page_bytes: u64) -> anyhow::Result<usize> {
    if page_bytes == 0 {
        anyhow::bail!("physical KV page size must be greater than zero");
    }
    let pages = budget_bytes / page_bytes;
    if pages == 0 {
        anyhow::bail!(
            "kv-cache-mb budget ({budget_bytes} bytes) is smaller than one physical KV page ({page_bytes} bytes)"
        );
    }
    usize::try_from(pages).map_err(|_| anyhow::anyhow!("KV page budget exceeds usize"))
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{cache_byte_limit, page_limit_for_budget, required_mebibytes_to_bytes};

    #[test]
    fn zero_cache_mebibytes_means_entry_limited_only() {
        assert_eq!(cache_byte_limit(0, "cache").unwrap(), u64::MAX);
    }

    #[test]
    fn cache_mebibytes_conversion_is_checked() {
        assert_eq!(cache_byte_limit(2, "cache").unwrap(), 2 * 1024 * 1024);
        assert!(cache_byte_limit(u64::MAX, "cache").is_err());
        assert!(required_mebibytes_to_bytes(0, "kv-cache-mb").is_err());
    }

    #[test]
    fn kv_page_limit_is_a_hard_byte_budget() {
        assert_eq!(page_limit_for_budget(10_000, 3_000).unwrap(), 3);
        assert!(page_limit_for_budget(2_999, 3_000).is_err());
        assert!(page_limit_for_budget(10_000, 0).is_err());
    }
}
