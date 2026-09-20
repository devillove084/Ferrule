use std::net::SocketAddr;

use std::time::Duration;

use anyhow::Context as _;
use ferrule_model::AutoConfig;
use ferrule_runtime::engine::model_factory::{
    ExpertCacheOptions, PipelineBuildOptions, PipelineRankBackend,
};
use ferrule_runtime::{
    BackendSelection, ModelFactoryOptions, ResidentModelPlanner, ResidentSchedulerConfig,
};
use ferrule_server::{
    ModelRegistration, ServerState, WorkerConfig, serve_with_shutdown, spawn_model_worker_with,
};

use crate::args::{RankBackend, ServeArgs, ServeEngine};

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
    let factory = ModelFactoryOptions {
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
    };
    let planner = ResidentModelPlanner::new();
    let prepared = if uses_pipeline(&args)? {
        planner.prepare_pipeline(
            &config,
            backend,
            args.chat_template.as_deref(),
            factory,
            pipeline_options(&args)?,
        )?
    } else {
        planner.prepare(&config, backend, args.chat_template.as_deref(), factory)?
    };
    if let Some(options) = prepared.pipeline_options() {
        eprintln!(
            "pipeline PP={} EP={} {} {:?} owners, precision={:?}; serial greedy decode, no mixed batches/cohort deferral, cancellation at chunk/token boundaries; no restart/replay",
            options.parallelism.pipeline_parallel,
            options.parallelism.expert_parallel,
            prepared.backend().as_str(),
            options.rank_backend,
            options.precision(prepared.backend()),
        );
        if let Some(devices) = &options.devices {
            eprintln!("CUDA ordinals (PP, then stage EP owners): {devices:?}");
        }
        #[cfg(unix)]
        if options.rank_backend == PipelineRankBackend::Process {
            let frame_ms = u128::from(args.rank_timeout_ms).max(
                ferrule_runtime::parallel::process::ProcessChildConfig::default()
                    .io_timeout
                    .as_millis(),
            );
            eprintln!(
                "process startup/command/shutdown absolute deadline={}ms; child frame I/O absolute deadline={frame_ms}ms once readable; healthy idle has no first-byte deadline (no restart/replay)",
                args.rank_timeout_ms
            );
        }
    }
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

fn pipeline_options(args: &ServeArgs) -> anyhow::Result<PipelineBuildOptions> {
    #[cfg(unix)]
    let process_launch = if args.rank_backend == RankBackend::Process {
        use ferrule_runtime::parallel::process::{
            ProcessChildConfig, ProcessFrameLimits, ProcessLaunch,
        };
        Some(
            ProcessLaunch::new(std::env::current_exe().context("resolve rank-worker executable")?)
                .arg("__rank-worker")
                .arg("--max-frame-bytes")
                .arg(ProcessFrameLimits::default().max_frame_bytes.to_string())
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

fn uses_pipeline(args: &ServeArgs) -> anyhow::Result<bool> {
    let requested = args.pipeline_parallel != 1
        || args.expert_parallel != 1
        || args.rank_backend != RankBackend::Thread
        || args.devices.is_some()
        || args.rank_restarts != 0
        || args.rank_timeout_ms != 30000;
    if args.engine == ServeEngine::Resident && requested {
        anyhow::bail!("parallel/rank options require --engine pipeline (or auto)");
    }
    Ok(args.engine == ServeEngine::Pipeline || requested)
}

fn validate_args(args: &ServeArgs) -> anyhow::Result<()> {
    uses_pipeline(args)?;
    if args.pipeline_parallel == 0 || args.expert_parallel == 0 {
        anyhow::bail!("parallel degrees must be greater than zero");
    }
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
mod tests {
    use super::*;
    use crate::args::{Cli, Command};
    use clap::Parser;

    fn serve(arguments: &[&str]) -> ServeArgs {
        let cli = Cli::try_parse_from(arguments).unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve")
        };
        args
    }

    #[test]
    fn serve_pipeline_selection_and_reserved_flags_are_explicit() {
        let defaults = serve(&["ferrule", "serve", "model"]);
        assert!(!uses_pipeline(&defaults).unwrap());
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
        assert!(uses_pipeline(&pipeline).unwrap());
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
        assert!(uses_pipeline(&process).unwrap());
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
        assert!(uses_pipeline(&args).unwrap());
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
        assert!(uses_pipeline(&cpu).unwrap());
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
                "8388608",
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
        for arguments in [
            vec!["ferrule", "serve", "model", "--pipeline-parallel", "0"],
            vec!["ferrule", "serve", "model", "--expert-parallel", "0"],
            vec![
                "ferrule",
                "serve",
                "model",
                "--engine",
                "resident",
                "--pipeline-parallel",
                "2",
            ],
            vec![
                "ferrule",
                "serve",
                "model",
                "--engine",
                "resident",
                "--rank-backend",
                "process",
            ],
        ] {
            assert!(validate_args(&serve(&arguments)).is_err());
        }
    }
}
