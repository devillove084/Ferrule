use std::future::Future;
use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Context as _;
use ferrule_model::AutoConfig;
use ferrule_runtime::engine::model_factory::{
    ExpertCacheOptions, PipelineBuildOptions, PipelineRankBackend,
};
use ferrule_runtime::{
    BackendSelection, ModelFactoryOptions, ResidentModelBuildPlan, ResidentModelPlanner,
    ResidentSchedulerConfig,
};
use ferrule_server::{
    ModelRegistration, ServerState, WorkerConfig, serve_with_shutdown, spawn_model_worker_with,
};

use crate::args::{RankBackend, ServeArgs, ServeEngine};

use super::resident::resident_driver_config;

pub fn cmd_serve(args: ServeArgs) -> anyhow::Result<()> {
    let prepared = prepare_model(&args)?;
    if let Some(options) = prepared.pipeline_options() {
        eprintln!(
            "pipeline PP={} TP={} EP={} {} {:?} owners, precision={:?}; serial greedy decode, no mixed batches/cohort deferral, cancellation at chunk/token boundaries; no restart/replay",
            options.parallelism.pipeline_parallel,
            options.parallelism.tensor_parallel,
            options.parallelism.expert_parallel,
            prepared.backend().as_str(),
            options.rank_backend,
            options.precision(prepared.backend()),
        );
        if let Some(devices) = &options.devices {
            let order = if options.parallelism.tensor_parallel > 1 {
                "PP-stage-major, then TP rank"
            } else {
                "PP, then stage EP owners"
            };
            eprintln!("CUDA ordinals ({order}): {devices:?}");
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
        eprintln!("  GET  /v1/models");
        eprintln!("  POST /v1/chat/completions");
        eprintln!("  POST /v1/completions");
        let server_result = serve_until_signal(address, state, signal).await;
        let shutdown_result = worker
            .shutdown()
            .await
            .context("failed to shut down model worker");
        combine_shutdown_results(server_result, shutdown_result)
    })
}

fn prepare_model(args: &ServeArgs) -> anyhow::Result<ResidentModelBuildPlan> {
    validate_args(args)?;
    let mut factory = model_factory_options(args)?;
    let config = AutoConfig::from_pretrained(&args.model)?;
    if config.descriptor().spec.family == ferrule_model::ModelFamily::Qwen35 {
        if args.expert_host_cache_entries.is_some()
            || args.expert_host_cache_mb.is_some()
            || args.expert_pinned_cache_entries.is_some()
            || args.expert_pinned_cache_mb.is_some()
            || args.moe_hotset_experts.is_some()
        {
            anyhow::bail!(
                "dense Qwen3.5 does not implement explicit MoE expert cache/hotset options"
            );
        }
        factory.driver_config.enable_native_proposals = false;
        factory.expert_cache = ExpertCacheOptions::default();
        factory.max_tensor_mebibytes = args.max_tensor_mb.unwrap_or(1024);
    }
    let backend = args
        .backend
        .as_deref()
        .map(BackendSelection::parse)
        .transpose()?
        .unwrap_or_default();
    let planner = ResidentModelPlanner::new();
    Ok(if uses_pipeline(args)? {
        planner.prepare_pipeline(
            &config,
            backend,
            args.chat_template.as_deref(),
            factory,
            pipeline_options(args)?,
        )?
    } else {
        planner.prepare(&config, backend, args.chat_template.as_deref(), factory)?
    })
}

fn model_factory_options(args: &ServeArgs) -> anyhow::Result<ModelFactoryOptions> {
    let expert_cache = if uses_pipeline(args)? {
        // Even an explicit value equal to a runtime default is a user policy
        // request. Pipeline does not implement these limits, so never erase it.
        for (flag, present) in [
            (
                "--expert-host-cache-entries",
                args.expert_host_cache_entries.is_some(),
            ),
            (
                "--expert-host-cache-mb",
                args.expert_host_cache_mb.is_some(),
            ),
            (
                "--expert-pinned-cache-entries",
                args.expert_pinned_cache_entries.is_some(),
            ),
            (
                "--expert-pinned-cache-mb",
                args.expert_pinned_cache_mb.is_some(),
            ),
        ] {
            if present {
                anyhow::bail!(
                    "pipeline serving does not implement expert_cache overrides ({flag})"
                );
            }
        }
        ExpertCacheOptions::default()
    } else {
        ExpertCacheOptions {
            host_entries: args.expert_host_cache_entries.unwrap_or(64),
            host_mebibytes: args.expert_host_cache_mb.unwrap_or(1024),
            pinned_entries: args.expert_pinned_cache_entries.unwrap_or(16),
            pinned_mebibytes: args.expert_pinned_cache_mb.unwrap_or(256),
        }
    };
    Ok(ModelFactoryOptions {
        max_layers: args.max_layers,
        max_tensor_mebibytes: args.max_tensor_mb.unwrap_or(128),
        output_head_chunk_rows: args.output_head_chunk_rows,
        expert_reader_max_tensor_mebibytes: args.expert_reader_max_slice_mb,
        expert_cache,
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
    signal: impl Future<Output = std::io::Result<()>>,
) -> anyhow::Result<()> {
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let server = serve_with_shutdown(address, state, async {
        let _ = stopped.await;
    });
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result.context("HTTP server failed"),
        result = signal => {
            let _ = stop.send(());
            combine_shutdown_results(
                result.context("shutdown signal handler failed"),
                server.await.context("HTTP server failed"),
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

fn uses_pipeline(args: &ServeArgs) -> anyhow::Result<bool> {
    let requested = args.pipeline_parallel != 1
        || args.tensor_parallel != 1
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
    if args.pipeline_parallel == 0 || args.tensor_parallel == 0 || args.expert_parallel == 0 {
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

    #[test]
    fn qwen35_default_cli_is_usable_and_explicit_moe_options_are_rejected() {
        use ferrule_model::models::qwen35::{
            Qwen35Config, Qwen35HfNameMapper, Qwen35TensorPartitionKind,
        };
        use std::io::Write;
        let fixture = PlanFixture::new(false);
        let value: serde_json::Value = serde_json::from_str(include_str!(
            "../../../ferrule-model/tests/qwen35_08b_config.json"
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
        let args = fixture.args(&[]);
        let plan = prepare_model(&args).unwrap();
        assert_eq!(plan.model_name(), "qwen3.5-0.8b");
        assert_eq!(plan.chat_template(), ferrule_model::ChatTemplate::Qwen35);
        for (flag, values) in [
            ("--expert-host-cache-entries", ["64", "0"]),
            ("--expert-host-cache-mb", ["1024", "0"]),
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
    fn parsed_omitted_cache_options_prepare_pipeline_plans() {
        for moe in [false, true] {
            let fixture = PlanFixture::new(moe);
            for backend in ["cpu", "cuda"] {
                if backend == "cuda" && !cfg!(feature = "cuda") {
                    continue;
                }
                for rank in ["thread", "process"] {
                    if rank == "process" && !cfg!(unix) {
                        continue;
                    }
                    for engine in ["auto", "pipeline"] {
                        let mut flags = vec![
                            "--engine",
                            engine,
                            "--backend",
                            backend,
                            "--pipeline-parallel",
                            "2",
                            "--rank-backend",
                            rank,
                        ];
                        if moe {
                            flags.extend(["--expert-parallel", "2"]);
                        }
                        let args = fixture.args(&flags);
                        assert_eq!(args.expert_host_cache_entries, None);
                        assert_eq!(args.expert_host_cache_mb, None);
                        assert_eq!(args.expert_pinned_cache_entries, None);
                        assert_eq!(args.expert_pinned_cache_mb, None);
                        assert_eq!(
                            model_factory_options(&args).unwrap().expert_cache,
                            ExpertCacheOptions::default()
                        );
                        let plan = prepare_model(&args).unwrap();
                        let expected_backend = if backend == "cpu" {
                            ferrule_model::ModelExecutionBackend::Cpu
                        } else {
                            ferrule_model::ModelExecutionBackend::Cuda
                        };
                        assert_eq!(plan.backend(), expected_backend);
                        assert_eq!(plan.observability().max_layers, 2);
                        assert_eq!(
                            plan.pipeline_options()
                                .unwrap()
                                .parallelism
                                .pipeline_parallel,
                            2
                        );
                    }
                }
            }
            // Explicit pipeline at PP1 must also avoid resident CLI defaults.
            assert!(
                prepare_model(&fixture.args(&["--engine", "pipeline"]))
                    .unwrap()
                    .pipeline_options()
                    .is_some()
            );
        }
    }

    #[test]
    fn parsed_explicit_pipeline_cache_constraints_are_never_discarded() {
        let fixture = PlanFixture::new(false);
        for engine in ["auto", "pipeline"] {
            for (flag, values) in [
                ("--expert-host-cache-entries", ["64", "256", "0"]),
                ("--expert-host-cache-mb", ["1024", "0", "1"]),
                ("--expert-pinned-cache-entries", ["16", "64", "0"]),
                ("--expert-pinned-cache-mb", ["256", "0", "1"]),
            ] {
                for value in values {
                    let args = fixture.args(&[
                        "--engine",
                        engine,
                        "--pipeline-parallel",
                        "2",
                        flag,
                        value,
                    ]);
                    let error = prepare_model(&args)
                        .err()
                        .expect("explicit policy must be rejected");
                    assert!(
                        error.to_string().contains("expert_cache overrides"),
                        "{error:#}"
                    );
                    assert!(error.to_string().contains(flag), "{error:#}");
                }
            }
        }
    }

    #[test]
    fn parsed_resident_cache_defaults_and_explicit_limits_are_preserved() {
        let fixture = PlanFixture::new(false);
        for engine in ["auto", "resident"] {
            let args = fixture.args(&["--engine", engine]);
            assert!(prepare_model(&args).unwrap().pipeline_options().is_none());
            assert_eq!(
                model_factory_options(&args).unwrap().expert_cache,
                ExpertCacheOptions {
                    host_entries: 64,
                    host_mebibytes: 1024,
                    pinned_entries: 16,
                    pinned_mebibytes: 256,
                }
            );
            let args = fixture.args(&["--engine", engine, "--expert-host-cache-mb", "7"]);
            assert!(prepare_model(&args).unwrap().pipeline_options().is_none());
            assert_eq!(
                model_factory_options(&args).unwrap().expert_cache,
                ExpertCacheOptions {
                    host_entries: 64,
                    host_mebibytes: 7,
                    pinned_entries: 16,
                    pinned_mebibytes: 256,
                }
            );
            let args = fixture.args(&[
                "--engine",
                engine,
                "--expert-host-cache-entries",
                "0",
                "--expert-host-cache-mb",
                "0",
                "--expert-pinned-cache-entries",
                "3",
                "--expert-pinned-cache-mb",
                "2",
            ]);
            assert!(prepare_model(&args).unwrap().pipeline_options().is_none());
            assert_eq!(
                model_factory_options(&args).unwrap().expert_cache,
                ExpertCacheOptions {
                    host_entries: 0,
                    host_mebibytes: 0,
                    pinned_entries: 3,
                    pinned_mebibytes: 2,
                }
            );
        }
    }

    #[test]
    fn parsed_pipeline_plan_still_rejects_unsupported_runtime_options() {
        let fixture = PlanFixture::new(false);
        for (flag, value, message) in [
            ("--max-layers", "1", "requires all checkpoint layers"),
            (
                "--moe-hotset-experts",
                "1",
                "does not implement moe_hotset_experts",
            ),
            (
                "--rank-restarts",
                "1",
                "does not support rank restart/replay",
            ),
        ] {
            let args = fixture.args(&["--engine", "pipeline", flag, value]);
            let error = prepare_model(&args)
                .err()
                .expect("unsupported option must be rejected");
            assert!(error.to_string().contains(message), "{error:#}");
        }
    }

    #[test]
    fn parsed_tensor_options_select_pipeline_and_reject_unsupported_backends() {
        let defaults = serve(&["ferrule", "serve", "model"]);
        assert_eq!(defaults.tensor_parallel, 1);
        let fixture = PlanFixture::new(false);
        for (extra, expected) in [
            (vec!["--backend", "cpu"], "CPU TP is unsupported"),
            (vec![], "CPU TP is unsupported"),
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
                "parallel/rank options require",
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
        let args = fixture.args(&["--tensor-parallel", "2"]);
        assert!(uses_pipeline(&args).unwrap());
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
                    model_factory_options(&args).unwrap().expert_cache,
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
        // 2 layers * 2 K/V * 4 heads * 2 dim * 4 bytes * 64 tokens
        // = 8192 bytes/session, so exactly 128 sessions fit in 1 MiB.
        for (pp, tp) in [(1, 2), (1, 4), (2, 2)] {
            for (sessions, fits) in [(128, true), (129, false)] {
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
            error.to_string().contains("MoE TP is unsupported"),
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
            vec!["ferrule", "serve", "model", "--tensor-parallel", "0"],
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
