use std::future::Future;
use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Context as _;
use ferrule_composition::model_factory::{
    ExpertCacheOptions, PipelineBuildOptions, PipelineRankBackend, SelectedImplementationKind,
    ServingEngineSelection, ServingModelOptions, ServingOptionProvenance,
};
use ferrule_composition::{ModelFactoryOptions, ResidentModelBuildPlan, ResidentModelPlanner};
use ferrule_model::AutoConfig;
use ferrule_runtime::ResidentSchedulerConfig;
use ferrule_server::{
    ModelRegistration, ServerState, WorkerConfig, serve_with_shutdown, spawn_model_worker_with,
};

use crate::args::{RankBackend, ServeArgs, ServeEngine};

use super::resident::resident_driver_config;

pub fn cmd_serve(args: ServeArgs) -> anyhow::Result<()> {
    let worker_config = worker_config(&args)?;
    let prepared = prepare_model(&args)?;
    eprintln!("[model plan] {}", prepared.resolution_report());
    if let Some(budget) = prepared.dense_bf16_memory_plan()? {
        eprintln!(
            "[dense BF16 device budget] {budget:?}; not a VRAM reservation; native BF16 LM-head output expands to F32 host logits; no Qwen3.5/GDN/MoE support"
        );
    }
    if let Some(budget) = prepared.numeric_bf16_memory_plan()? {
        eprintln!(
            "[experimental Mixed BF16] profile={} activation/KV/conv=BF16 SSM/accumulation=F32; numeric E4M3FN/BF16 scales; one CUDA owner (visible device 0), lazy experts, resident KV; required_device_bytes={} budget={:?}; not a VRAM reservation; frozen two-prompt PASS is not universal quality acceptance; fresh quality remains blocking",
            budget.backend_profile(),
            budget.total_device_bytes(),
            budget.model_budget(),
        );
    }
    if let Some(report) = serving_resource_report(&prepared)? {
        eprintln!("{report}");
    }
    for line in execution_report(&prepared, args.rank_timeout_ms) {
        eprintln!("{line}");
    }
    let adapter_name = prepared.model_name();
    let backend_name = prepared.backend().as_str();
    let backend_profile = prepared.backend_profile();
    let chat_template = prepared.chat_template();
    let served_model_name = args
        .served_model_name
        .clone()
        .unwrap_or_else(|| adapter_name.to_owned());
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("ferrule-http")
        .build()?;
    // Register before loading the model so a signal during startup is not lost.
    let signal = runtime
        .block_on(async { shutdown_signal() })
        .context("failed to install shutdown signal handlers")?;
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
    runtime.block_on(async move {
        eprintln!("Ferrule OpenAI API listening on http://{address}");
        eprintln!("  GET  /health");
        eprintln!("  GET  /readyz");
        eprintln!("  GET  /v1/models");
        eprintln!("  POST /v1/chat/completions");
        eprintln!("  POST /v1/completions");
        serve_until_signal(address, state, worker, signal).await
    })
}

fn serving_resource_report(prepared: &ResidentModelBuildPlan) -> anyhow::Result<Option<String>> {
    Ok(prepared.serving_resource_requirements()?.map(|required| {
        format!(
            "[serving device budget] complete_owner_ledger={required:?}; source-bound claims and live admission remain required; not a VRAM reservation"
        )
    }))
}

fn execution_report(prepared: &ResidentModelBuildPlan, rank_timeout_ms: u64) -> Vec<String> {
    let topology = prepared.topology_placement();
    let mut report = vec![format!(
        "expert transfers: requested={:?} effective={}; unchanged host staging and device/cache budgets",
        prepared.requested_expert_transfer_policy(),
        prepared.expert_transfer_policy().as_str(),
    )];
    if let Some(read) = prepared.expert_read_pipeline_options() {
        report.push(format!(
            "expert read host staging: schedule={:?} host_cap_bytes={} metadata_bytes={} worker_stack_bytes={}; metadata and requested stack included in host cap; separate from CUDA cache/device budgets; OS/TLS/allocator headroom not included; no GPU-overlap claim",
            read.schedule, read.max_host_bytes, read.max_metadata_bytes, read.worker_stack_bytes,
        ));
    }
    match prepared.selected_implementation().kind {
        SelectedImplementationKind::Resident => {
            if let (Some(rank_backend), Some(root), Some(experts)) = (
                topology.rank_backend,
                topology.physical.root_device,
                &topology.physical.expert_devices,
            ) {
                report.push(format!(
                    "resident root + {rank_backend:?} EP{}: root CUDA {root}, expert CUDA {experts:?}; precision={:?}; full owned expert warm/residency; physical root+expert0 admission required; no restart/replay",
                    topology.logical.expert_parallel,
                    prepared.support().arithmetic,
                ));
                if let Some(profile) = prepared.numeric_fp8_profile() {
                    report.push(format!(
                        "numeric expert arithmetic: {}",
                        profile.arithmetic_description(),
                    ));
                    if rank_backend == PipelineRankBackend::Process {
                        report.push(format!(
                            "numeric expert wire v2; root remains resident; children warm owned partitions; startup/command/shutdown deadline={rank_timeout_ms}ms",
                        ));
                    }
                }
            }
        }
        SelectedImplementationKind::Pipeline => {
            if let Some(rank_backend) = topology.rank_backend {
                report.push(format!(
                    "pipeline PP={} TP={} EP={} {} {rank_backend:?} owners, precision={:?}; serial greedy decode, no mixed batches/cohort deferral, cancellation at chunk/token boundaries; no restart/replay",
                    topology.logical.pipeline_parallel,
                    topology.logical.tensor_parallel,
                    topology.logical.expert_parallel,
                    prepared.backend().as_str(),
                    prepared.support().arithmetic,
                ));
                if let Some(profile) = prepared.numeric_fp8_profile() {
                    report.push(format!(
                        "numeric hybrid stage-scoped EP: {}; full shared host prewarm and all-owned experts; physical root+expert colocation admission; no process fallback",
                        profile.arithmetic_description()
                    ));
                    if profile == ferrule_composition::model_factory::NumericFp8Profile::W8A16Bf16 {
                        report.push(
                            "W8A16 opt-in: quality-unaccepted; GPU quality acceptance is separate from factory admission".to_owned(),
                        );
                    }
                }
                if let Some(devices) = &topology.physical.devices {
                    let order = if topology.logical.tensor_parallel > 1 {
                        "PP-major TP roots, then stage EP children"
                    } else {
                        "PP, then stage EP owners"
                    };
                    report.push(format!("CUDA ordinals ({order}): {devices:?}"));
                }
                #[cfg(unix)]
                if rank_backend == PipelineRankBackend::Process {
                    let frame_ms = u128::from(rank_timeout_ms).max(
                        ferrule_runtime::parallel::process::ProcessChildConfig::default()
                            .io_timeout
                            .as_millis(),
                    );
                    report.push(format!(
                        "process startup/command/shutdown absolute deadline={rank_timeout_ms}ms; child frame I/O absolute deadline={frame_ms}ms once readable; healthy idle has no first-byte deadline (no restart/replay)",
                    ));
                }
            }
        }
    }
    report
}

fn prepare_model(args: &ServeArgs) -> anyhow::Result<ResidentModelBuildPlan> {
    let (backend, options, serving) = validate_args(args)?;
    let config = AutoConfig::from_pretrained(&args.model)?;
    Ok(ResidentModelPlanner::new().prepare_serving(
        &config,
        backend,
        args.chat_template.as_deref(),
        options,
        serving,
    )?)
}

fn serving_options(args: &ServeArgs) -> anyhow::Result<ServingModelOptions> {
    Ok(ServingModelOptions {
        expert_transfer_policy: args.expert_transfer_policy.map(Into::into),
        expert_read_pipeline: args.expert_read.options()?,
        model_profile: args.model_profile.map(Into::into),
        model_precision: args.model_precision.map(Into::into),
        kv_execution_policy: args.kv_execution_policy.into(),
        engine: match args.engine {
            ServeEngine::Auto => ServingEngineSelection::Auto,
            ServeEngine::Resident => ServingEngineSelection::Resident,
            ServeEngine::Pipeline => ServingEngineSelection::Pipeline,
        },
        parallel: if parallel_requested(args) || args.engine == ServeEngine::Pipeline {
            Some(pipeline_options(args)?)
        } else {
            None
        },
        max_tensor_mebibytes: args.max_tensor_mb,
        expert_host_cache_entries: args.expert_host_cache_entries,
        expert_host_cache_mebibytes: args.expert_host_cache_mb,
        expert_pinned_cache_entries: args.expert_pinned_cache_entries,
        expert_pinned_cache_mebibytes: args.expert_pinned_cache_mb,
        moe_hotset_experts: args.moe_hotset_experts,
    })
}

fn model_factory_options(args: &ServeArgs) -> anyhow::Result<ModelFactoryOptions> {
    let expert_cache = ExpertCacheOptions {
        host_entries: args.expert_host_cache_entries.unwrap_or(64),
        host_mebibytes: args.expert_host_cache_mb.unwrap_or(1024),
        pinned_entries: args.expert_pinned_cache_entries.unwrap_or(16),
        pinned_mebibytes: args.expert_pinned_cache_mb.unwrap_or(256),
    };
    Ok(ModelFactoryOptions {
        max_layers: args.max_layers,
        max_tensor_mebibytes: args.max_tensor_mb.unwrap_or(128),
        output_head_chunk_rows: args.output_head_chunk_rows,
        expert_reader_max_tensor_mebibytes: args.expert_reader_max_slice_mb,
        expert_cache,
        qwen35_moe_capacity: mixed_capacity_limits(args)?,
        numeric_fp8_profile: args.cuda_expert_device.numeric_profile(),
        qwen35_host_cache: if args.model_profile.is_some() {
            // TP1 is lazy; TP2 has no host cache and keeps local shards resident.
            None
        } else {
            args.expert_prewarm.options(None, None)?
        },
        moe_hotset_experts: args.moe_hotset_experts.unwrap_or(0),
        kv_cache_mebibytes: Some(args.kv_cache_mb),
        scheduler_config: ResidentSchedulerConfig {
            prefill_chunk_size: args.prefill_chunk_size,
            max_active_sequences: args.max_active_sequences,
            max_decode_batch: args.max_active_sequences,
            decode_cohort_target: args.decode_cohort_target,
            decode_cohort_max_deferrals: args.decode_cohort_max_deferrals,
            max_batch_tokens: args.max_batch_tokens,
            ..ResidentSchedulerConfig::default()
        },
        driver_config: resident_driver_config(args.ctx_size, true),
    })
}

fn mixed_capacity_limits(
    args: &ServeArgs,
) -> anyhow::Result<Option<ferrule_composition::model_factory::Qwen35MoeCapacityLimits>> {
    let current = args.cuda_expert_device.capacity_limits();
    if args.model_profile.is_none() {
        return Ok(current);
    }
    let mut limits = current.unwrap_or_default();
    if args.tensor_parallel == 2 {
        if args.cuda_expert_device.entries.is_some()
            || args.cuda_expert_device.bytes.is_some()
            || args.cuda_expert_scratch_mb.is_some()
        {
            anyhow::bail!(
                "Mixed TP2 uses resident local expert shards; TP1 expert cache/scratch flags are unsupported"
            );
        }
        limits.device_bytes = 22 << 30;
        limits.workspace_bytes =
            ferrule_composition::model_factory::MIXED_TP2_DEFAULT_WORKSPACE_BYTES;
    }
    for (flag, value, destination) in [
        (
            "cuda-device-budget-mb",
            args.cuda_device_budget_mb,
            &mut limits.device_bytes,
        ),
        (
            "cuda-workspace-mb",
            args.cuda_workspace_mb,
            &mut limits.workspace_bytes,
        ),
        (
            "cuda-expert-scratch-mb",
            args.cuda_expert_scratch_mb,
            &mut limits.scratch_bytes,
        ),
    ] {
        if let Some(value) = value {
            *destination = value
                .checked_mul(1 << 20)
                .ok_or_else(|| anyhow::anyhow!("{flag} overflows bytes"))?;
        }
    }
    Ok(Some(limits))
}

fn serving_provenance(args: &ServeArgs) -> ServingOptionProvenance {
    use crate::args::ExpertPrewarmArg;
    use ferrule_model::transformer::host_experts::ExpertPrewarmMode;
    ServingOptionProvenance {
        expert_prewarm: args.expert_prewarm.mode.map(|mode| match mode {
            ExpertPrewarmArg::Full => ExpertPrewarmMode::Full,
            ExpertPrewarmArg::Lazy => ExpertPrewarmMode::Lazy,
        }),
        expert_prewarm_workers: args.expert_prewarm.workers,
    }
}

fn shutdown_signal() -> std::io::Result<impl Future<Output = std::io::Result<()>>> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut interrupt = signal(SignalKind::interrupt())?;
        let mut terminate = signal(SignalKind::terminate())?;
        Ok(async move {
            tokio::select! {
                received = interrupt.recv() => received,
                received = terminate.recv() => received,
            }
            .ok_or_else(|| std::io::Error::other("shutdown signal stream closed"))
        })
    }
    #[cfg(not(unix))]
    {
        Ok(tokio::signal::ctrl_c())
    }
}

async fn serve_until_signal(
    address: SocketAddr,
    state: ServerState,
    worker: ferrule_server::ModelWorker,
    signal: impl Future<Output = std::io::Result<()>>,
) -> anyhow::Result<()> {
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let server = serve_with_shutdown(address, state, async {
        let _ = stopped.await;
    });
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => {
            worker.handle().begin_shutdown();
            combine_shutdown_results(
                result.context("HTTP server failed"),
                worker.shutdown().await.context("failed to shut down model worker"),
            )
        }
        result = signal => {
            // Close admission before asking Axum to drain. Cancel/close must run
            // concurrently with HTTP drain so a live stream cannot form a cycle.
            let deadline = worker.handle().begin_shutdown();
            let _ = stop.send(());
            let (http, close) = tokio::join!(
                tokio::time::timeout_at(deadline, &mut server),
                worker.shutdown(),
            );
            let http = match http {
                Ok(result) => result.context("HTTP server failed"),
                Err(error) => Err(anyhow::Error::new(error).context("HTTP graceful drain exceeded cooperative shutdown deadline")),
            };
            combine_shutdown_results(
                combine_shutdown_results(result.context("shutdown signal handler failed"), http),
                close.context("failed to shut down model worker"),
            )
        }
    }
}

fn combine_shutdown_results(
    serving: anyhow::Result<()>,
    shutdown: anyhow::Result<()>,
) -> anyhow::Result<()> {
    match (serving, shutdown) {
        (Ok(()), result) | (result, Ok(())) => result,
        (Err(serving), Err(shutdown)) => {
            Err(shutdown.context(format!("serving also failed: {serving:#}")))
        }
    }
}

fn pipeline_options(args: &ServeArgs) -> anyhow::Result<PipelineBuildOptions> {
    #[cfg(unix)]
    let process_launch = if args.rank_backend == RankBackend::Process {
        use ferrule_runtime::parallel::process::{
            ProcessChildConfig, ProcessFrameLimits, ProcessLaunch,
        };
        // This is a trusted ceiling, not a reservation. Runtime admission charges
        // the Boot's negotiated frames (dense hybrid full logits may exceed 8 MiB).
        Some(
            ProcessLaunch::new(std::env::current_exe().context("resolve rank-worker executable")?)
                .arg("__rank-worker")
                .arg("--max-frame-bytes")
                .arg(ProcessFrameLimits::MAX_FRAME_BYTES.to_string())
                .arg("--io-timeout-ms")
                .arg(
                    u128::from(args.rank_timeout_ms)
                        .max(ProcessChildConfig::default().io_timeout.as_millis())
                        .to_string(),
                ),
        )
    } else {
        None
    };
    Ok(PipelineBuildOptions {
        parallelism: ferrule_common::ParallelismPlan {
            pipeline_parallel: args.pipeline_parallel,
            tensor_parallel: args.tensor_parallel,
            expert_parallel: args.expert_parallel,
            ..Default::default()
        },
        rank_backend: match args.rank_backend {
            RankBackend::Thread => PipelineRankBackend::Thread,
            RankBackend::Process => PipelineRankBackend::Process,
        },
        #[cfg(unix)]
        process_launch,
        devices: args.devices.clone(),
        rank_timeout: Duration::from_millis(args.rank_timeout_ms),
        rank_restarts: args.rank_restarts,
    })
}

fn parallel_requested(args: &ServeArgs) -> bool {
    args.pipeline_parallel != 1
        || args.tensor_parallel != 1
        || args.expert_parallel != 1
        || args.rank_backend != RankBackend::Thread
        || args.devices.is_some()
        || args.rank_restarts != 0
        || args.rank_timeout_ms != 30000
}

fn validate_args(
    args: &ServeArgs,
) -> anyhow::Result<(
    ferrule_composition::BackendSelection,
    ModelFactoryOptions,
    ServingModelOptions,
)> {
    validate_common_args(args)?;
    let backend = super::backend_policy::select(args.backend.as_deref())?;
    let options = model_factory_options(args)?;
    let serving = serving_options(args)?;
    // Explicit policy and startup environment precede model-path I/O. Family,
    // metadata and device admission remain the composition planner's job.
    serving.validate_before_model_io(backend, &options, &serving_provenance(args))?;
    Ok((backend, options, serving))
}

fn worker_config(args: &ServeArgs) -> anyhow::Result<WorkerConfig> {
    let runtime_admission_options = if args.runtime_max_waiting_requests.is_some()
        || args.runtime_max_request_identities.is_some()
        || args.runtime_max_session_identities.is_some()
    {
        let defaults = ferrule_runtime::RuntimeAdmissionOptions::default();
        Some(ferrule_runtime::RuntimeAdmissionOptions {
            max_waiting_requests: args
                .runtime_max_waiting_requests
                .unwrap_or(defaults.max_waiting_requests),
            max_request_identities: args
                .runtime_max_request_identities
                .unwrap_or(defaults.max_request_identities),
            max_session_identities: args
                .runtime_max_session_identities
                .unwrap_or(defaults.max_session_identities),
        })
    } else {
        None
    };
    let config = WorkerConfig {
        // ResidentModelBuildPlan::build validates/installs the requested policy.
        // Leave the worker request at its preserving default: a second HostKV
        // configure would reset the same owner's policy during startup.
        command_queue_capacity: args.request_queue_capacity,
        event_queue_capacity: args.event_queue_capacity,
        admission_timeout: Duration::from_secs(args.admission_timeout_secs),
        max_inflight_requests: args.max_inflight_requests,
        max_prompt_bytes: args.max_prompt_bytes,
        max_body_bytes: args.max_body_bytes,
        runtime_admission_options,
        ..WorkerConfig::default()
    };
    config.validate_admission()?;
    Ok(config)
}

fn validate_common_args(args: &ServeArgs) -> anyhow::Result<()> {
    worker_config(args)?;
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

#[cfg(test)]
#[path = "../../tests/unit"]
mod tests {
    #[path = "expert_read_config.rs"]
    mod expert_read_config;
    #[path = "expert_transfer/mod.rs"]
    mod expert_transfer_tests;
    #[path = "host_kv_policy.rs"]
    mod host_kv_policy_tests;
    #[path = "model_precision/mod.rs"]
    mod model_precision_tests;
    #[cfg(feature = "cpu-reference")]
    #[path = "serve_reference.rs"]
    mod reference_tests;
    #[path = "serve_resolution.rs"]
    mod resolution_tests;
    #[path = "serve_parallel_plan.rs"]
    mod serve_parallel_plan;
    use super::*;
    use crate::args::{Cli, Command};
    use clap::Parser;
    #[cfg(feature = "cpu-reference")]
    use ferrule_composition::BackendSelection;

    fn serve(arguments: &[&str]) -> ServeArgs {
        let cli = Cli::try_parse_from(arguments).unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve")
        };
        *args
    }

    #[test]
    fn admission_cli_defaults_overrides_and_boundaries() {
        let args = serve(&["ferrule", "serve", "missing-model"]);
        let config = worker_config(&args).unwrap();
        let defaults = WorkerConfig::default();
        assert_eq!(config.max_inflight_requests, defaults.max_inflight_requests);
        assert_eq!(config.max_prompt_bytes, defaults.max_prompt_bytes);
        assert_eq!(config.max_body_bytes, defaults.max_body_bytes);
        assert_eq!(config.runtime_admission_options, None);
        let args = serve(&[
            "ferrule",
            "serve",
            "missing-model",
            "--max-inflight-requests",
            "7",
            "--max-prompt-bytes",
            "1024",
            "--max-body-bytes",
            "2048",
            "--runtime-max-waiting-requests",
            "2",
            "--runtime-max-request-identities",
            "3",
            "--runtime-max-session-identities",
            "1",
            "--request-queue-capacity",
            "5",
            "--event-queue-capacity",
            "2",
        ]);
        let config = worker_config(&args).unwrap();
        assert_eq!(
            (
                config.max_inflight_requests,
                config.max_prompt_bytes,
                config.max_body_bytes
            ),
            (7, 1024, 2048)
        );
        assert_eq!(
            (config.command_queue_capacity, config.event_queue_capacity),
            (5, 2)
        );
        assert_eq!(
            config.runtime_admission_options,
            Some(ferrule_runtime::RuntimeAdmissionOptions {
                max_waiting_requests: 2,
                max_request_identities: 3,
                max_session_identities: 1,
            })
        );
        let args = serve(&[
            "ferrule",
            "serve",
            "missing-model",
            "--runtime-max-session-identities",
            "1",
        ]);
        let options = worker_config(&args)
            .unwrap()
            .runtime_admission_options
            .unwrap();
        assert_eq!(options.max_waiting_requests, 1024);
        assert_eq!(options.max_request_identities, 4096);
        for flags in [
            vec!["--runtime-max-request-identities", "1"],
            vec![
                "--runtime-max-waiting-requests",
                "3",
                "--runtime-max-request-identities",
                "2",
            ],
            vec!["--request-queue-capacity", "18446744073709551615"],
            vec!["--event-queue-capacity", "18446744073709551615"],
            vec!["--admission-timeout-secs", "18446744073709551615"],
        ] {
            let mut argv = vec!["ferrule", "serve", "missing-model"];
            argv.extend(flags);
            if let Ok(cli) = Cli::try_parse_from(argv) {
                let Command::Serve(args) = cli.command else {
                    unreachable!()
                };
                assert!(validate_common_args(&args).is_err());
            }
        }
        for flag in [
            "--max-inflight-requests",
            "--max-prompt-bytes",
            "--max-body-bytes",
            "--runtime-max-waiting-requests",
            "--runtime-max-request-identities",
            "--runtime-max-session-identities",
        ] {
            for value in ["0", "-1", "18446744073709551616", "18446744073709551615"] {
                assert!(
                    Cli::try_parse_from(["ferrule", "serve", "missing-model", flag, value])
                        .is_err(),
                    "{flag}={value}"
                );
            }
            assert!(Cli::try_parse_from(["ferrule", "serve", "missing-model", flag, "1"]).is_ok());
        }
    }

    #[test]
    fn mixed_tp2_workspace_default_is_cache_aware_without_changing_tp1_or_legacy() {
        let tp2 = serve(&[
            "ferrule",
            "serve",
            "missing-model",
            "--model-profile",
            "qwen35-mixed-bf16",
            "--tensor-parallel",
            "2",
        ]);
        let limits = mixed_capacity_limits(&tp2).unwrap().unwrap();
        assert_eq!(limits.workspace_bytes, 4 << 30);
        assert_eq!(limits.device_bytes, 22 << 30);
        let explicit = serve(&[
            "ferrule",
            "serve",
            "missing-model",
            "--model-profile",
            "qwen35-mixed-bf16",
            "--tensor-parallel",
            "2",
            "--cuda-workspace-mb",
            "1024",
        ]);
        assert_eq!(
            mixed_capacity_limits(&explicit)
                .unwrap()
                .unwrap()
                .workspace_bytes,
            1 << 30
        );
        let tp1 = serve(&[
            "ferrule",
            "serve",
            "missing-model",
            "--model-profile",
            "qwen35-mixed-bf16",
        ]);
        assert_eq!(
            mixed_capacity_limits(&tp1)
                .unwrap()
                .unwrap()
                .workspace_bytes,
            1 << 30
        );
        let legacy = serve(&["ferrule", "serve", "missing-model"]);
        assert!(mixed_capacity_limits(&legacy).unwrap().is_none());
    }

    struct PlanFixture(std::path::PathBuf);

    impl PlanFixture {
        fn new(moe: bool) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "ferrule-cli-serve-plan-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            let mut config = serde_json::json!({
                "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 4, "intermediate_size": 8, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
                "vocab_size": 8, "max_position_embeddings": 64
            });
            if moe {
                config["model_type"] = serde_json::json!("qwen3_moe");
                config["architectures"] = serde_json::json!(["Qwen3MoeForCausalLM"]);
                config["num_experts"] = serde_json::json!(2);
                config["num_experts_per_tok"] = serde_json::json!(2);
            }
            std::fs::write(path.join("config.json"), config.to_string()).unwrap();
            Self(path)
        }

        #[cfg(feature = "cuda")]
        fn tensor() -> Self {
            let fixture = Self::new(false);
            std::fs::write(
                fixture.0.join("config.json"),
                serde_json::json!({
                    "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3",
                    "hidden_size":4, "intermediate_size":8, "num_hidden_layers":2,
                    "num_attention_heads":4, "num_key_value_heads":4, "head_dim":2,
                    "rms_norm_eps":0.000001, "rope_theta":10000.0, "rope_scaling":null,
                    "max_position_embeddings":64, "vocab_size":8, "tie_word_embeddings":true,
                    "attention_bias":false, "attention_dropout":0.0, "hidden_act":"silu",
                    "torch_dtype":"bfloat16", "use_cache":true, "use_sliding_window":false,
                    "sliding_window":null, "max_window_layers":2, "initializer_range":0.02,
                    "bos_token_id":2, "eos_token_id":7
                })
                .to_string(),
            )
            .unwrap();
            fixture
        }

        #[cfg(feature = "cpu-reference")]
        fn reference_args(&self, flags: &[&str]) -> ServeArgs {
            let mut args = self.args(flags);
            if args.backend.is_none() {
                args.backend = Some("cpu".into());
            }
            args
        }

        fn args(&self, flags: &[&str]) -> ServeArgs {
            let mut arguments = vec!["ferrule", "serve", self.0.to_str().unwrap()];
            arguments.extend_from_slice(flags);
            serve(&arguments)
        }
    }

    impl Drop for PlanFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn qwen35_moe_fixture() -> PlanFixture {
        use ferrule_model::models::qwen35::{
            Qwen35Config, Qwen35HfNameMapper, Qwen35TensorPartitionKind,
        };
        use std::io::Write;
        let fixture = PlanFixture::new(false);
        let value: serde_json::Value = serde_json::from_str(include_str!(
            "../../../ferrule-model/tests/qwen35_35b_fp8_config.json"
        ))
        .unwrap();
        std::fs::write(fixture.0.join("config.json"), value.to_string()).unwrap();
        let mapper = Qwen35HfNameMapper::new(&Qwen35Config::from_value(&value).unwrap());
        let mut header = serde_json::Map::new();
        let mut bytes = 0;
        for tensor in mapper
            .tensors()
            .filter(|t| t.partition == Qwen35TensorPartitionKind::Text)
        {
            header.insert(tensor.external_name.clone(), serde_json::json!({"dtype":tensor.dtype.as_str(), "shape":tensor.shape, "data_offsets":[bytes,bytes+tensor.bytes()]}));
            bytes += tensor.bytes();
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = std::fs::File::create(fixture.0.join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len(8 + header.len() as u64 + bytes).unwrap();
        fixture
    }

    #[test]
    fn qwen35_35b_default_cli_uses_proven_f32_and_rejects_unused_overrides() {
        let fixture = qwen35_moe_fixture();
        for (engine, rank_backend) in ["auto", "resident", "pipeline"]
            .into_iter()
            .flat_map(|engine| ["thread", "process"].map(|rank| (engine, rank)))
        {
            if rank_backend == "process" && !cfg!(unix) {
                continue;
            }
            for (degree, devices) in [("2", "3,1"), ("4", "3,2,1,0"), ("8", "0,1,2,3,4,5,6,7")] {
                let ep = prepare_model(&fixture.args(&[
                    "--engine",
                    engine,
                    "--rank-backend",
                    rank_backend,
                    "--expert-parallel",
                    degree,
                    "--devices",
                    devices,
                    "--expert-prewarm",
                    "full",
                    "--expert-host-cache-mb",
                    "32768",
                ]));
                if cfg!(feature = "cuda") {
                    let ep = ep.unwrap();
                    assert!(
                        ep.pipeline_options().is_none(),
                        "EP must not select generic Pipeline"
                    );
                    assert!(
                        ep.qwen35_moe_capacity_limits().unwrap().is_none(),
                        "EP must not charge the single root's 4 GiB routed cache"
                    );
                    assert_eq!(
                        ep.selected_implementation().kind,
                        SelectedImplementationKind::Resident
                    );
                    let topology = ep.topology_placement();
                    assert_eq!(topology.logical.pipeline_parallel, 1);
                    assert_eq!(topology.logical.tensor_parallel, 1);
                    assert_eq!(
                        topology.logical.expert_parallel,
                        degree.parse::<usize>().unwrap()
                    );
                    let expected_devices = devices
                        .split(',')
                        .map(|v| v.parse::<usize>().unwrap())
                        .collect::<Vec<_>>();
                    assert_eq!(topology.physical.devices, Some(expected_devices.clone()));
                    assert_eq!(topology.physical.root_device, Some(expected_devices[0]));
                    assert_eq!(
                        topology.physical.expert_devices,
                        Some(expected_devices.clone())
                    );
                    assert_eq!(
                        topology.physical.source_identity,
                        Some(ferrule_common::ParallelRankId::new(0))
                    );
                    assert_eq!(
                        topology.rank_backend,
                        Some(if rank_backend == "process" {
                            PipelineRankBackend::Process
                        } else {
                            PipelineRankBackend::Thread
                        })
                    );
                    let report = execution_report(&ep, 1234).join("\n");
                    assert!(
                        report.contains(&format!(
                            "EP{degree}: root CUDA {}, expert CUDA {expected_devices:?}",
                            expected_devices[0]
                        )),
                        "{report}"
                    );
                    assert!(report.contains("full owned expert warm/residency; physical root+expert0 admission required; no restart/replay"), "{report}");
                    assert!(
                        report.contains(ep.numeric_fp8_profile().unwrap().arithmetic_description()),
                        "{report}"
                    );
                    assert!(!report.contains("pipeline PP="), "{report}");
                    assert_eq!(report.contains("numeric expert wire v2; root remains resident; children warm owned partitions; startup/command/shutdown deadline=1234ms"), rank_backend == "process");
                    let placement = ep.qwen35_expert_placement().unwrap();
                    assert_eq!(
                        placement.rank_backend(),
                        if rank_backend == "process" {
                            PipelineRankBackend::Process
                        } else {
                            PipelineRankBackend::Thread
                        }
                    );
                    assert!(ep.backend_profile().contains(rank_backend));
                    assert_eq!(
                        placement.expert_parallel(),
                        degree.parse::<usize>().unwrap()
                    );
                    assert_eq!(
                        placement.expert_devices(),
                        devices
                            .split(',')
                            .map(|v| v.parse::<usize>().unwrap())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(placement.root_device(), placement.expert_devices()[0]);
                    assert_eq!(
                        ep.qwen35_host_cache_options().unwrap().max_bytes,
                        32768u64 << 20
                    );
                } else {
                    let error = ep.err().expect("CUDA feature is required");
                    let runtime = error.downcast_ref::<ferrule_runtime::Error>().unwrap();
                    assert!(
                        matches!(
                            runtime,
                            ferrule_runtime::Error::Backend {
                                source: ferrule_common::Error::ModelSource { .. },
                            }
                        ),
                        "{error:#}"
                    );
                }
            }
        }
        for invalid in [
            vec!["--backend", "cpu"],
            vec!["--tensor-parallel", "2"],
            vec!["--pipeline-parallel", "2"],
            vec!["--rank-backend", "process", "--rank-timeout-ms", "0"],
            vec!["--rank-restarts", "1"],
            vec!["--rank-timeout-ms", "1"],
            vec!["--expert-prewarm", "lazy"],
            vec!["--cuda-expert-device-entries", "1024"],
        ] {
            let mut flags = vec!["--expert-parallel", "2", "--devices", "0,1"];
            flags.extend(invalid);
            let error = prepare_model(&fixture.args(&flags))
                .err()
                .expect("unsupported EP option");
            if flags.contains(&"cpu") && !cfg!(feature = "cpu-reference") {
                assert!(error.to_string().contains("CPU execution is test-only"));
                continue;
            }
            let ferrule_runtime::Error::Backend {
                source: ferrule_common::Error::ModelSource { source },
            } = error.downcast_ref::<ferrule_runtime::Error>().unwrap()
            else {
                panic!("{error:#}")
            };
            assert_eq!(
                source
                    .downcast_ref::<ferrule_model::transformer::UnsupportedOperator>()
                    .unwrap()
                    .operator,
                if !cfg!(feature = "cuda")
                    && (flags.contains(&"--expert-prewarm")
                        || flags.contains(&"--cuda-expert-device-entries"))
                {
                    // Catalog availability precedes external-owner option validation.
                    "qwen35_moe_cuda"
                } else {
                    "qwen35_moe_expert_parallel"
                }
            );
        }
        let args = fixture.args(&[]);
        let result = prepare_model(&args);
        if cfg!(feature = "cuda") {
            let plan = result.unwrap();
            assert_eq!(plan.model_name(), "qwen3.5-35b-a3b-fp8");
            assert!(
                plan.qwen35_expert_placement().is_none(),
                "default stays single"
            );
            assert!(plan.backend_profile().contains("fp8-f32-tf32x3"));
            assert_eq!(plan.chat_template(), ferrule_model::ChatTemplate::Qwen35);
            assert_eq!(plan.observability().max_layers, 40);
            let limits = plan.qwen35_moe_capacity_limits().unwrap().unwrap();
            assert_eq!(
                (limits.max_experts, limits.max_bytes, limits.scratch_bytes),
                (1024, 4usize << 30, 64 << 20)
            );
            use ferrule_model::transformer::host_experts::{
                ExpertPrewarmMode, HostExpertCacheOptions,
            };
            assert_eq!(
                plan.qwen35_host_cache_options(),
                Some(HostExpertCacheOptions::default())
            );
            let explicit = prepare_model(&fixture.args(&[
                "--expert-host-cache-entries",
                "10240",
                "--expert-host-cache-mb",
                "40960",
                "--expert-prewarm-workers",
                "8",
            ]))
            .unwrap();
            assert_eq!(explicit.qwen35_host_cache_options().unwrap().workers, 8);
            let lazy = prepare_model(&fixture.args(&[
                "--expert-prewarm",
                "lazy",
                "--expert-host-cache-mb",
                "0",
            ]))
            .unwrap();
            assert_eq!(
                lazy.qwen35_host_cache_options().unwrap().mode,
                ExpertPrewarmMode::Lazy
            );
            #[cfg(feature = "cuda")]
            {
                use ferrule_composition::model_factory::Qwen35MoeMemoryPlan;
                for (entries, bytes) in [(1024, 4usize << 30), (320, 2usize << 30)] {
                    let custom = prepare_model(&fixture.args(&[
                        "--cuda-expert-device-entries",
                        &entries.to_string(),
                        "--cuda-expert-device-bytes",
                        &bytes.to_string(),
                    ]))
                    .unwrap();
                    let limits = custom.qwen35_moe_capacity_limits().unwrap().unwrap();
                    let admission =
                        Qwen35MoeMemoryPlan::open_hf(&fixture.0, limits, 1 << 30).unwrap();
                    let (cache, scratch) = admission
                        .runner_options()
                        .hybrid_cuda_numeric_fp8()
                        .unwrap();
                    assert_eq!(
                        (cache.max_experts, cache.max_bytes, scratch),
                        (entries, bytes, 64 << 20)
                    );
                    assert!(admission.check_available_device_bytes(0).is_err());
                }
                for (flag, value) in [
                    ("--cuda-expert-device-entries", "0"),
                    ("--cuda-expert-device-bytes", "0"),
                ] {
                    assert!(prepare_model(&fixture.args(&[flag, value])).is_err());
                }
                // A huge cache must fail metadata admission, before loading or CUDA allocation.
                let oversized =
                    prepare_model(&fixture.args(&["--cuda-expert-device-bytes", "25769803776"]))
                        .unwrap();
                assert!(
                    Qwen35MoeMemoryPlan::open_hf(
                        &fixture.0,
                        oversized.qwen35_moe_capacity_limits().unwrap().unwrap(),
                        1 << 30
                    )
                    .is_err()
                );
            }
        } else {
            assert!(result.err().unwrap().to_string().contains("cuda"));
        }
        assert!(prepare_model(&fixture.args(&["--backend", "cpu"])).is_err());
        for (flag, values) in [
            ("--expert-pinned-cache-entries", ["16", "0"]),
            ("--expert-pinned-cache-mb", ["256", "0"]),
            ("--moe-hotset-experts", ["1", "0"]),
        ] {
            for value in values {
                assert!(
                    prepare_model(&fixture.args(&[flag, value]))
                        .err()
                        .unwrap()
                        .to_string()
                        .contains("explicit MoE")
                );
            }
        }
        for args in [
            ["--max-layers", "1"],
            ["--rank-backend", "process"],
            ["--expert-parallel", "2"],
            ["--expert-max-slice-mb", "32"],
            ["--output-head-chunk-rows", "32"],
            ["--tensor-parallel", "2"],
            ["--pipeline-parallel", "2"],
        ] {
            assert!(prepare_model(&fixture.args(&args)).is_err());
        }
        if cfg!(feature = "cuda") {
            assert!(prepare_model(&fixture.args(&["--backend", "cuda"])).is_ok());
        }
    }

    #[test]
    fn parsed_tensor_options_select_pipeline_and_reject_unsupported_backends() {
        let defaults = serve(&["ferrule", "serve", "model"]);
        assert_eq!(defaults.tensor_parallel, 1);
        let fixture = PlanFixture::new(false);
        for (extra, expected) in [
            (
                vec!["--backend", "cpu"],
                if cfg!(feature = "cpu-reference") {
                    "CPU TP is unsupported"
                } else {
                    "CPU execution is test-only"
                },
            ),
            (
                vec!["--backend", "cuda", "--rank-backend", "process"],
                "process TP serving is unsupported",
            ),
            (
                vec!["--backend", "cuda", "--expert-parallel", "2"],
                "EP x TP serving is unsupported",
            ),
            (
                vec!["--engine", "resident"],
                if cfg!(feature = "cuda") {
                    "parallel/rank options require"
                } else {
                    "pipeline CUDA serving requires the 'cuda' feature"
                },
            ),
            (
                vec!["--backend", "cuda", "--tensor-parallel", "3"],
                "supported TP degrees",
            ),
        ] {
            let mut flags = vec!["--tensor-parallel", "2"];
            // clap deliberately rejects duplicate flags; replace the TP value.
            if extra.contains(&"--tensor-parallel") {
                flags.clear();
            }
            flags.extend(extra);
            let args = fixture.args(&flags);
            let error = prepare_model(&args)
                .err()
                .expect("unsupported TP must fail before build");
            assert!(error.to_string().contains(expected), "{error:#}");
        }
        // Omitted backend has exactly the explicit CUDA policy, never CPU Auto.
        let default_plan = prepare_model(&fixture.args(&["--tensor-parallel", "2"]));
        let cuda_plan =
            prepare_model(&fixture.args(&["--tensor-parallel", "2", "--backend", "cuda"]));
        assert_eq!(
            default_plan.err().map(|e| e.to_string()),
            cuda_plan.err().map(|e| e.to_string())
        );
        let args = fixture.args(&["--tensor-parallel", "2"]);
        assert!(serving_options(&args).unwrap().parallel.is_some());
        assert_eq!(
            pipeline_options(&args).unwrap().parallelism.tensor_parallel,
            2
        );
        #[cfg(not(feature = "cuda"))]
        {
            let error =
                prepare_model(&fixture.args(&["--tensor-parallel", "2", "--backend", "cuda"]))
                    .err()
                    .unwrap();
            assert!(error.to_string().contains("'cuda' feature"), "{error:#}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn parsed_tensor_build_plans_are_f32_and_preserve_device_order_without_loading() {
        let fixture = PlanFixture::tensor();
        // No tokenizer or weights exist: successful preflight must be metadata-only.
        for (pp, tp, devices) in [(1, 2, "1,0"), (1, 4, "3,2,1,0"), (2, 2, "3,1,2,0")] {
            for engine in ["auto", "pipeline"] {
                let args = fixture.args(&[
                    "--engine",
                    engine,
                    "--backend",
                    "cuda",
                    "--pipeline-parallel",
                    &pp.to_string(),
                    "--tensor-parallel",
                    &tp.to_string(),
                    "--devices",
                    devices,
                    "--ctx-size",
                    "64",
                ]);
                let plan = prepare_model(&args).unwrap();
                let options = plan.pipeline_options().unwrap();
                assert_eq!(options.parallelism.pipeline_parallel, pp);
                assert_eq!(options.parallelism.tensor_parallel, tp);
                assert_eq!(options.parallelism.expert_parallel, 1);
                assert_eq!(options.devices, args.devices);
                assert_eq!(options.rank_backend, PipelineRankBackend::Thread);
                assert_eq!(
                    options.precision(plan.backend()),
                    ferrule_model::execution::ExecutionPrecisionPolicy::f32()
                );
                assert_eq!(
                    prepare_model(&args)
                        .unwrap()
                        .requested_options()
                        .expert_cache,
                    ExpertCacheOptions::default()
                );
            }
        }
        // Default ordinals remain implicit until physical owners are constructed.
        let plan = prepare_model(&fixture.args(&[
            "--backend",
            "cuda",
            "--tensor-parallel",
            "2",
            "--ctx-size",
            "64",
        ]))
        .unwrap();
        assert_eq!(plan.pipeline_options().unwrap().devices, None);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn parsed_tensor_kv_budget_is_global_f32_not_replicated_per_rank() {
        let fixture = PlanFixture::tensor();
        // Logical KV: 2 layers * 2 K/V * 4 heads * 2 dim * 4 bytes * 64 tokens
        // = 8192 bytes/session. PR07 physical P=2L includes transaction shadows:
        // 16384 bytes/session, so 64 sessions fit in 1 MiB, independently of PP/TP.
        for (pp, tp) in [(1, 2), (1, 4), (2, 2)] {
            for (sessions, fits) in [(64, true), (65, false), (128, false)] {
                let args = fixture.args(&[
                    "--backend",
                    "cuda",
                    "--pipeline-parallel",
                    &pp.to_string(),
                    "--tensor-parallel",
                    &tp.to_string(),
                    "--ctx-size",
                    "64",
                    "--max-active-sequences",
                    &sessions.to_string(),
                    "--kv-cache-mb",
                    "1",
                ]);
                let result = prepare_model(&args);
                if fits {
                    result.unwrap();
                } else {
                    let error = result.err().unwrap();
                    assert!(error.to_string().contains("KV budget"), "{error:#}");
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn parsed_tensor_build_plans_strictly_prevalidate_devices_geometry_and_capacity() {
        let fixture = PlanFixture::tensor();
        for (extra, expected) in [
            (vec!["--devices", "0"], "requires 2 CUDA ordinals"),
            (vec!["--devices", "0,0"], "distinct CUDA device"),
            (
                vec!["--pipeline-parallel", "2", "--devices", "0,1"],
                "requires 4 CUDA ordinals",
            ),
            (
                vec!["--pipeline-parallel", "2", "--devices", "0,1,1,2"],
                "distinct CUDA device",
            ),
            (vec!["--devices", "0,2147483648"], "within i32"),
            (
                vec!["--pipeline-parallel", "3"],
                "exceeds the model layer count",
            ),
            (vec!["--max-layers", "1"], "requires all checkpoint layers"),
            (
                vec!["--kv-cache-mb", "1", "--max-active-sequences", "1024"],
                "KV budget",
            ),
            (
                vec!["--expert-host-cache-mb", "0"],
                "expert_cache overrides",
            ),
        ] {
            let mut flags = vec![
                "--backend",
                "cuda",
                "--tensor-parallel",
                "2",
                "--ctx-size",
                "64",
            ];
            flags.extend(extra);
            let error = prepare_model(&fixture.args(&flags))
                .err()
                .expect("invalid TP plan");
            assert!(error.to_string().contains(expected), "{error:#}");
        }
        let error = prepare_model(&fixture.args(&[
            "--backend",
            "cuda",
            "--tensor-parallel",
            "2",
            "--ctx-size",
            "65",
        ]))
        .err()
        .unwrap();
        assert!(error.to_string().contains("position range"), "{error:#}");
        let moe = PlanFixture::new(true);
        let error = prepare_model(&moe.args(&["--backend", "cuda", "--tensor-parallel", "2"]))
            .err()
            .unwrap();
        assert!(
            error.to_string().contains(
                "TP serving requires dense Qwen3 or the dedicated Qwen35Moe hybrid factory"
            ),
            "{error:#}"
        );
        let path = fixture.0.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["num_key_value_heads"] = serde_json::json!(1);
        std::fs::write(path, config.to_string()).unwrap();
        let error = prepare_model(&fixture.args(&[
            "--backend",
            "cuda",
            "--tensor-parallel",
            "2",
            "--ctx-size",
            "64",
        ]))
        .err()
        .unwrap();
        assert!(
            error.to_string().contains("heads must be divisible by TP"),
            "{error:#}"
        );
    }

    #[test]
    fn shutdown_errors_keep_their_source_even_when_serving_also_fails() {
        for serving_fails in [false, true] {
            for shutdown_fails in [false, true] {
                let serving = if serving_fails {
                    Err(anyhow::anyhow!("HTTP bind failure"))
                } else {
                    Ok(())
                };
                let shutdown = if shutdown_fails {
                    Err(anyhow::Error::new(std::io::Error::other(
                        "worker drain failure",
                    )))
                } else {
                    Ok(())
                };
                let result = combine_shutdown_results(serving, shutdown);
                assert_eq!(result.is_ok(), !serving_fails && !shutdown_fails);
                if let Err(error) = result {
                    let message = format!("{error:#}");
                    assert_eq!(message.contains("HTTP bind failure"), serving_fails);
                    assert_eq!(message.contains("worker drain failure"), shutdown_fails);
                    if shutdown_fails {
                        assert!(error.downcast_ref::<std::io::Error>().is_some());
                    }
                }
            }
        }
    }

    #[test]
    fn serve_pipeline_selection_and_reserved_flags_are_explicit() {
        let defaults = serve(&["ferrule", "serve", "model"]);
        assert!(!serving_options(&defaults).unwrap().parallel.is_some());
        let pipeline = serve(&[
            "ferrule",
            "serve",
            "model",
            "--engine",
            "pipeline",
            "--pipeline-parallel",
            "2",
            "--backend",
            "cpu",
        ]);
        assert!(serving_options(&pipeline).unwrap().parallel.is_some());
        assert_eq!(pipeline.pipeline_parallel, 2);
        assert_eq!(pipeline.rank_backend, RankBackend::Thread);
        let process = serve(&[
            "ferrule",
            "serve",
            "model",
            "--rank-backend",
            "process",
            "--devices",
            "2,3",
            "--rank-timeout-ms",
            "10",
            "--rank-restarts",
            "1",
        ]);
        assert!(serving_options(&process).unwrap().parallel.is_some());
        assert_eq!(process.devices, Some(vec![2, 3]));
        assert_eq!(process.rank_restarts, 1);
        assert_eq!(process.rank_timeout_ms, 10);
    }

    #[test]
    fn serve_maps_cuda_and_cpu_ep_options_without_claiming_sampling_or_precision_flags() {
        let args = serve(&[
            "ferrule",
            "serve",
            "model",
            "--backend",
            "cuda",
            "--pipeline-parallel",
            "2",
            "--expert-parallel",
            "2",
            "--devices",
            "5,4,3,2,1,0",
        ]);
        validate_args(&args).unwrap();
        assert!(serving_options(&args).unwrap().parallel.is_some());
        let options = pipeline_options(&args).unwrap();
        assert_eq!(options.parallelism.pipeline_parallel, 2);
        assert_eq!(options.parallelism.expert_parallel, 2);
        assert_eq!(options.devices, Some(vec![5, 4, 3, 2, 1, 0]));
        assert_eq!(options.rank_backend, PipelineRankBackend::Thread);
        assert_eq!(options.rank_timeout, Duration::from_secs(30));
        assert_eq!(options.rank_restarts, 0);
        let cpu = serve(&[
            "ferrule",
            "serve",
            "model",
            "--backend",
            "cpu",
            "--expert-parallel",
            "2",
        ]);
        assert!(serving_options(&cpu).unwrap().parallel.is_some());
        assert_eq!(pipeline_options(&cpu).unwrap().devices, None);
        for flag in ["--precision", "--temperature"] {
            assert!(Cli::try_parse_from(["ferrule", "serve", "model", flag, "1"]).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn serve_process_launch_uses_current_executable_and_finite_limits() {
        let args = serve(&[
            "ferrule",
            "serve",
            "model",
            "--rank-backend",
            "process",
            "--rank-timeout-ms",
            "45000",
        ]);
        let options = pipeline_options(&args).unwrap();
        assert_eq!(options.rank_backend, PipelineRankBackend::Process);
        assert_eq!(options.rank_timeout, Duration::from_secs(45));
        assert_eq!(options.rank_restarts, 0);
        let launch = options.process_launch.unwrap();
        assert_eq!(launch.executable, std::env::current_exe().unwrap());
        assert_eq!(
            launch.args,
            [
                "__rank-worker",
                "--max-frame-bytes",
                "134217728",
                "--io-timeout-ms",
                "300000"
            ]
            .map(std::ffi::OsString::from)
        );
        assert!(launch.environment.is_empty());
        let thread = serve(&["ferrule", "serve", "model"]);
        assert!(pipeline_options(&thread).unwrap().process_launch.is_none());
    }

    #[test]
    fn serve_rejects_zero_degrees_and_resident_parallel_options() {
        let fixture = PlanFixture::new(false);
        for flags in [
            vec!["--pipeline-parallel", "0"],
            vec!["--expert-parallel", "0"],
            vec!["--tensor-parallel", "0"],
            vec!["--engine", "resident", "--pipeline-parallel", "2"],
            vec!["--engine", "resident", "--rank-backend", "process"],
        ] {
            assert!(prepare_model(&fixture.args(&flags)).is_err());
        }
    }
}
