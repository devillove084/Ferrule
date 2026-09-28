//! Ordinary resident engines and ModelWorker; no replacement forward/page manager.
use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use ferrule_common::CompletionHub;
use ferrule_model::{ChatTemplate, TokenizerHandle};
use ferrule_runtime::engine::BoxedSessionInferenceEngine;
use ferrule_runtime::{
    GenerateRequest, InferenceCancelProgress, InferenceCompletionReactor, InferenceEngine,
    InferenceShutdownProgress, RequestId, ResidentDriverStep, ResidentTokenEvent,
    Result as RuntimeResult, SequenceState, SessionId,
};
use ferrule_server::{
    ModelRegistration, ServerState, WorkerConfig, router, spawn_model_worker_with,
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};
use tower::ServiceExt;

#[path = "../../ferrule-runtime/tests/support/hybrid_fixture.rs"]
mod fixture;
const DEADLINE: Duration = Duration::from_secs(90);

#[derive(Default)]
struct Probe {
    cancelled: AtomicUsize,
    shutdown: AtomicUsize,
    tokens: AtomicUsize,
}
struct Observed {
    inner: BoxedSessionInferenceEngine,
    probe: Arc<Probe>,
}
impl InferenceEngine for Observed {
    fn completion_hub(&self) -> CompletionHub {
        self.inner.completion_hub()
    }
    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        self.inner.take_completion_reactors()
    }
    fn has_pending_async_work(&self) -> bool {
        self.inner.has_pending_async_work()
    }
    fn encode(&self, prompt: &str) -> RuntimeResult<Vec<u32>> {
        self.inner.encode(prompt)
    }
    fn submit(&mut self, request: GenerateRequest) {
        self.inner.submit(request);
    }
    fn request_cleanup(&self, request: RequestId) -> ferrule_runtime::InferenceRequestCleanup {
        self.inner.request_cleanup(request)
    }
    fn try_submit(&mut self, request: GenerateRequest) -> RuntimeResult<()> {
        self.inner.try_submit(request)
    }
    fn step(
        &mut self,
        emit: &mut dyn FnMut(&ResidentTokenEvent) -> RuntimeResult<()>,
    ) -> RuntimeResult<ResidentDriverStep> {
        self.inner.step(&mut |event| {
            self.probe.tokens.fetch_add(1, Ordering::Release);
            emit(event)
        })
    }
    fn cancel_request(&mut self, id: RequestId) -> RuntimeResult<InferenceCancelProgress> {
        self.inner.cancel_request(id)
    }
    fn drain_finished(&mut self) -> Vec<SequenceState> {
        self.inner.drain_finished()
    }
    fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        let states = self.inner.drain_cancelled();
        self.probe
            .cancelled
            .fetch_add(states.len(), Ordering::Release);
        states
    }
    fn drain_failed(&mut self) -> Vec<SequenceState> {
        let states = self.inner.drain_failed();
        assert!(states.is_empty(), "hybrid request failed: {states:?}");
        states
    }
    fn shutdown(&mut self) -> RuntimeResult<InferenceShutdownProgress> {
        let result = self.inner.shutdown()?;
        if result == InferenceShutdownProgress::Complete {
            let kv = self.inner.observability_snapshot().kv_cache.unwrap();
            assert_eq!(kv.stats.allocated_pages, 0);
            assert_eq!(kv.stats.retiring_pages, 0);
            self.probe.shutdown.fetch_add(1, Ordering::Release);
        }
        Ok(result)
    }
}
fn reset_roundtrip(engine: &mut BoxedSessionInferenceEngine, prompt: &str) {
    let session = SessionId(900);
    engine.retain_session(session).unwrap();
    let mut reference = None;
    for id in 900..902 {
        let prompt_tokens = engine.encode(prompt).unwrap();
        engine
            .try_submit(GenerateRequest {
                id: RequestId(id),
                session_id: Some(session),
                prompt_tokens,
                max_new_tokens: 1,
                stop: vec![],
                ignore_eos: true,
            })
            .unwrap();
        let deadline = Instant::now() + DEADLINE;
        let mut tokens = vec![];
        loop {
            engine
                .step(&mut |e| {
                    tokens.push(e.token);
                    Ok(())
                })
                .unwrap();
            assert!(engine.drain_failed().is_empty());
            if !engine.drain_finished().is_empty() {
                break;
            }
            assert!(Instant::now() < deadline, "reset generation deadline");
            std::thread::yield_now();
        }
        assert_eq!(tokens.len(), 1);
        if let Some(reference) = &reference {
            assert_eq!(&tokens, reference);
        }
        reference = Some(tokens);
        engine.reset_session(session).unwrap();
        assert_eq!(engine.retained_session_position(session), Some(0));
        assert_eq!(
            engine
                .observability_snapshot()
                .kv_cache
                .unwrap()
                .stats
                .allocated_pages,
            0
        );
    }
}
fn post(uri: &str, value: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(value.to_string()))
        .unwrap()
}
async fn text(response: axum::response::Response) -> String {
    assert_eq!(response.status(), StatusCode::OK);
    String::from_utf8(
        tokio::time::timeout(DEADLINE, response.into_body().collect())
            .await
            .unwrap()
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap()
}
async fn http_lifecycle(
    factory: impl FnOnce() -> RuntimeResult<BoxedSessionInferenceEngine> + Send + 'static,
    template: ChatTemplate,
    prompt: &'static str,
    expected: &'static str,
    chat_expected: &'static str,
) {
    let probe = Arc::new(Probe::default());
    let owner_probe = probe.clone();
    let worker = spawn_model_worker_with(
        move || {
            let mut inner = factory()?;
            reset_roundtrip(&mut inner, prompt);
            Ok::<_, ferrule_runtime::Error>(Observed {
                inner,
                probe: owner_probe,
            })
        },
        WorkerConfig {
            event_queue_capacity: 2,
            admission_timeout: DEADLINE,
            ..Default::default()
        },
    )
    .unwrap();
    let app = router(ServerState::new(
        ModelRegistration::new("hybrid", template),
        worker.handle(),
    ));
    let tokenized = text(
        app.clone()
            .oneshot(post(
                "/v1/tokenize",
                json!({"model":"hybrid", "prompt":prompt}),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert!(
        serde_json::from_str::<Value>(&tokenized).unwrap()["data"][0]["count"]
            .as_u64()
            .unwrap()
            > 0
    );
    for (uri, body, expected) in [
        (
            "/v1/completions",
            json!({"model":"hybrid","prompt":prompt,"max_tokens":1,"stream":true,"stream_options":{"include_usage":true}}),
            expected,
        ),
        (
            "/v1/chat/completions",
            json!({"model":"hybrid","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":1,"stream":true,"stream_options":{"include_usage":true}}),
            chat_expected,
        ),
    ] {
        let response = app.clone().oneshot(post(uri, body)).await.unwrap();
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let body = text(response).await;
        assert!(body.contains(expected), "{body}");
        assert!(body.contains("\"finish_reason\":\"length\""), "{body}");
        assert!(body.contains("\"completion_tokens\":1"), "{body}");
        assert_eq!(body.matches("data: [DONE]").count(), 1);
    }
    let stopped = text(app.clone().oneshot(post("/v1/completions", json!({"model":"hybrid","prompt":prompt,"max_tokens":8,"stop":expected,"stream":true}))).await.unwrap()).await;
    assert!(stopped.contains("\"finish_reason\":\"stop\""), "{stopped}");
    assert!(stopped.contains("[DONE]"));
    let before = probe.tokens.load(Ordering::Acquire);
    let response = app.clone().oneshot(post("/v1/completions", json!({"model":"hybrid","prompt":prompt,"max_tokens":24,"ignore_eos":true,"stream":true}))).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    // Do not consume the body. Bounded backpressure must cancel the actual owner
    // after a token, not merely remove a request that never started.
    tokio::time::timeout(DEADLINE, async {
        while probe.tokens.load(Ordering::Acquire) == before {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    drop(response);
    tokio::time::timeout(DEADLINE, async {
        while probe.cancelled.load(Ordering::Acquire) == 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    let again = text(
        app.oneshot(post(
            "/v1/completions",
            json!({"model":"hybrid","prompt":prompt,"max_tokens":1}),
        ))
        .await
        .unwrap(),
    )
    .await;
    let value: Value = serde_json::from_str(&again).unwrap();
    assert_eq!(value["choices"][0]["text"], expected);
    let handle = worker.handle();
    tokio::time::timeout(Duration::from_secs(5), async {
        while handle.admission_snapshot().held_requests != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("real engine cleanup receipts must return HTTP admission before shutdown");
    tokio::time::timeout(DEADLINE, worker.shutdown())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(probe.shutdown.load(Ordering::Acquire), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hermetic_hybrid_cpu_http_lifecycle() {
    let fixture = fixture::Fixture::new();
    http_lifecycle(
        move || cpu_engine(fixture),
        ChatTemplate::Plain,
        "t1 t3 t2",
        "t0",
        "t7",
    )
    .await;
}
fn cpu_engine(fixture: fixture::Fixture) -> RuntimeResult<BoxedSessionInferenceEngine> {
    use ferrule_model::decoder::{
        GenericDecoderOptions, GenericDecoderRunner, HybridCpuDecoder, HybridStateSchema,
    };
    let resources = fixture.resources();
    let options = GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ferrule_model::ModelFamily::Qwen35,
        ferrule_model::WeightSource::Safetensors,
        2,
        32,
        4,
        1,
        1 << 20,
        ferrule_model::execution::ExecutionPrecisionPolicy::f32(),
    )?;
    let schema = HybridStateSchema::from_spec(resources.spec(), 4)?.kv_planes(2, 32)?;
    let runner = GenericDecoderRunner::<HybridCpuDecoder>::hybrid_cpu(
        resources,
        TokenizerHandle::load(&fixture.dir)?,
        options,
    )?;
    ferrule_runtime::engine::build_resident_engine(
        runner,
        Box::new(schema),
        ferrule_runtime::engine::ResidentKvPageAccounting::ContextCapacity,
        ferrule_runtime::ResidentSchedulerConfig {
            prefill_chunk_size: 2,
            max_batch_tokens: 4,
            ..Default::default()
        },
        ferrule_runtime::ResidentTopKDriverConfig {
            ctx_size: 32,
            enable_native_proposals: false,
            ..Default::default()
        },
    )
}

#[tokio::test]
async fn hybrid_cpu_http_nested_eos_and_ignore_eos() {
    let fixture = fixture::Fixture::new();
    std::fs::write(
        fixture.dir.join("config.json"),
        json!({"text_config":{"eos_token_id":[0,10]}}).to_string(),
    )
    .unwrap();
    let worker =
        spawn_model_worker_with(move || cpu_engine(fixture), WorkerConfig::default()).unwrap();
    let app = router(ServerState::new(
        ModelRegistration::new("hybrid", ChatTemplate::Plain),
        worker.handle(),
    ));
    for ignore in [false, true] {
        let output = text(app.clone().oneshot(post("/v1/completions", json!({"model":"hybrid","prompt":"t1 t3 t2","max_tokens":1,"ignore_eos":ignore}))).await.unwrap()).await;
        let value: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(value["choices"][0]["text"], if ignore { "t0" } else { "" });
        assert_eq!(
            value["choices"][0]["finish_reason"],
            if ignore { "length" } else { "stop" }
        );
        assert_eq!(
            value["usage"]["completion_tokens"],
            if ignore { 1 } else { 0 }
        );
    }
    worker.shutdown().await.unwrap();
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires Qwen3.5-0.8B NAS artifact and one CUDA GPU; full-depth F32, about 4 minutes"]
async fn qwen35_cuda_http_token_stop_abort_reset_shutdown() {
    use ferrule_runtime::{BackendSelection, ModelFactoryOptions, ResidentModelPlanner};
    let path = std::env::var_os("FERRULE_QWEN35_08B_DIR")
        .unwrap_or_else(|| "/mnt/nas1/hf/Qwen3.5-0.8B".into());
    let config = ferrule_model::AutoConfig::from_pretrained(path).unwrap();
    let plan = ResidentModelPlanner
        .prepare(
            &config,
            BackendSelection::Cuda,
            None,
            ModelFactoryOptions {
                max_layers: None,
                max_tensor_mebibytes: 1024,
                output_head_chunk_rows: 4096,
                expert_reader_max_tensor_mebibytes: 64,
                expert_cache: Default::default(),
                qwen35_moe_capacity: None,
                qwen35_host_cache: None,
                moe_hotset_experts: 0,
                kv_cache_mebibytes: Some(32),
                scheduler_config: ferrule_runtime::ResidentSchedulerConfig {
                    max_batch_tokens: 32,
                    prefill_chunk_size: 32,
                    ..Default::default()
                },
                driver_config: ferrule_runtime::ResidentTopKDriverConfig {
                    ctx_size: 64,
                    enable_native_proposals: false,
                    ..Default::default()
                },
            },
        )
        .unwrap();
    http_lifecycle(
        move || plan.build(),
        ChatTemplate::Qwen35,
        "The capital of France is",
        " Paris",
        "Hello",
    )
    .await;
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires Qwen3.5-35B-A3B-FP8 and one 24 GiB GPU; two raw prompts, max_tokens=1; not full logits"]
async fn qwen35_35b_fp8_f32_http_hello_capital_and_shutdown() {
    qwen35_35b_http_smoke(false).await;
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires NAS 35B FP8 and eight free CUDA GPUs; eager EP8, Hello one-token SSE; run with external 900s deadline"]
async fn qwen35_35b_fp8_f32_ep8_http_hello_and_shutdown() {
    ferrule_common::observability::init_tracing();
    qwen35_35b_http_smoke(true).await;
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires eight free GPUs and NAS 35B FP8; concurrent SSE/cancel, 1k context; run with external 1200s deadline"]
async fn qwen35_35b_fp8_ep8_http_concurrent_cancel_and_shutdown() {
    use ferrule_runtime::{BackendSelection, ModelFactoryOptions, ResidentModelPlanner};

    ferrule_common::observability::init_tracing();
    let started = Instant::now();
    let path = std::env::var_os("FERRULE_NUMERIC_FP8_MODEL_DIR")
        .unwrap_or_else(|| "/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8".into());
    let config = ferrule_model::AutoConfig::from_pretrained(path).unwrap();
    let options = ModelFactoryOptions {
        max_layers: None,
        max_tensor_mebibytes: 1024,
        output_head_chunk_rows: 4096,
        expert_reader_max_tensor_mebibytes: 64,
        expert_cache: Default::default(),
        qwen35_moe_capacity: None,
        qwen35_host_cache: Some(
            ferrule_model::transformer::host_experts::HostExpertCacheOptions {
                max_bytes: 32u64 << 30,
                ..Default::default()
            },
        ),
        moe_hotset_experts: 0,
        kv_cache_mebibytes: Some(1024),
        scheduler_config: ferrule_runtime::ResidentSchedulerConfig {
            max_active_sequences: 2,
            max_decode_batch: 2,
            decode_cohort_target: 2,
            max_batch_tokens: 16,
            ..Default::default()
        },
        driver_config: ferrule_runtime::ResidentTopKDriverConfig {
            ctx_size: 1024,
            enable_native_proposals: false,
            ..Default::default()
        },
    };
    let plan = ResidentModelPlanner
        .prepare_qwen35_expert_parallel(
            &config,
            BackendSelection::Auto,
            None,
            options,
            ferrule_runtime::engine::model_factory::PipelineBuildOptions {
                parallelism: ferrule_common::ParallelismPlan {
                    expert_parallel: 8,
                    ..Default::default()
                },
                devices: Some((0..8).collect()),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(plan.qwen35_expert_placement().is_some());
    assert!(plan.pipeline_options().is_none());
    assert!(plan.backend_profile().contains("fp8-f32-tf32x3"));

    let worker = spawn_model_worker_with(
        move || plan.build(),
        WorkerConfig {
            admission_timeout: Duration::from_secs(180),
            shutdown_timeout: Duration::from_secs(120),
            ..Default::default()
        },
    )
    .unwrap();
    eprintln!(
        "EP8 startup ready phase={:?} elapsed={:.3}s",
        worker.handle().snapshot().phase,
        started.elapsed().as_secs_f64()
    );
    let app = router(ServerState::new(
        ModelRegistration::new("hybrid", ChatTemplate::Qwen35),
        worker.handle(),
    ));

    let handle = worker.handle();
    let smoke = tokio::time::timeout(Duration::from_secs(900), async {
        let request = |prompt: &'static str, max_tokens: usize| {
            post(
                "/v1/completions",
                json!({
                    "model":"hybrid", "prompt":prompt, "max_tokens":max_tokens,
                    "ignore_eos":true, "stream":true,
                    "stream_options":{"include_usage":true}
                }),
            )
        };
        let (left, right) = tokio::join!(
            app.clone().oneshot(request("Hello", 2)),
            app.clone().oneshot(request("The capital of France is", 2)),
        );
        let left = left.map_err(|error| format!("left concurrent request failed: {error:?}"))?;
        let right = right.map_err(|error| format!("right concurrent request failed: {error:?}"))?;
        let (left, right) = tokio::join!(collect_ep8_sse(left), collect_ep8_sse(right));
        let left = left?;
        let right = right?;
        require_ep8_usage("concurrent-left", &left, 2)?;
        require_ep8_usage("concurrent-right", &right, 2)?;
        eprintln!(
            "EP8 concurrent complete elapsed={:.3}s tokens=(2,2)",
            started.elapsed().as_secs_f64()
        );

        let response = app
            .clone()
            .oneshot(request("Hello", 64))
            .await
            .map_err(|error| format!("cancel request failed: {error:?}"))?;
        if response.status() != StatusCode::OK {
            return Err(format!("cancel request status: {}", response.status()));
        }
        let mut body = response.into_body();
        // SSE keep-alive comments are not generated tokens. Wait for an actual
        // nonterminal completion chunk before dropping the real subscription.
        let first = loop {
            let frame = body.frame().await
                .ok_or_else(|| "cancel stream ended before a token".to_string())?
                .map_err(|error| format!("cancel SSE frame failed: {error:?}"))?;
            let Some(data) = frame.data_ref() else { continue };
            let frame_text = std::str::from_utf8(data).map_err(|error| error.to_string())?;
            for line in frame_text.lines().filter_map(|line| line.strip_prefix("data: ")) {
                if line == "[DONE]" {
                    return Err("cancel stream completed before disconnect".to_string());
                }
                let event: Value = serde_json::from_str(line).map_err(|error| error.to_string())?;
                if event.get("error").is_some() {
                    return Err(format!("cancel SSE error: {event}"));
                }
                if event["choices"][0]["text"].as_str().is_some_and(|text| !text.is_empty())
                    && event["choices"][0]["finish_reason"].is_null()
                {
                    break;
                }
                return Err(format!("unexpected cancel SSE event: {event}"));
            }
            if frame_text.lines().any(|line| line.starts_with("data: ")) {
                break frame;
            }
        };
        eprintln!(
            "EP8 cancel first SSE frame received elapsed={:.3}s frame={first:?}",
            started.elapsed().as_secs_f64()
        );
        drop(body);
        eprintln!(
            "EP8 cancel client body dropped elapsed={:.3}s; submitting independent request",
            started.elapsed().as_secs_f64()
        );

        let independent = app
            .oneshot(request("Hello", 2))
            .await
            .map_err(|error| format!("independent request failed: {error:?}"))?;
        let independent = collect_ep8_sse(independent).await?;
        require_ep8_usage("independent-after-cancel", &independent, 2)?;
        eprintln!(
            "EP8 independent request complete elapsed={:.3}s tokens=2 cancellation=client_disconnect",
            started.elapsed().as_secs_f64()
        );
        while handle.admission_snapshot().held_requests != 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        eprintln!("EP8 cancellation cleanup: {:?}", handle.admission_snapshot());
        let metrics = ferrule_common::observability::METRICS.snapshot();
        eprintln!(
            "EP8 metrics generated_tokens={} avg_ttft_ms={:.3} avg_tpot_ms={:.3}",
            metrics.generated_tokens, metrics.avg_ttft_ms, metrics.avg_tpot_ms
        );
        Ok::<(), String>(())
    })
    .await;

    let shutdown = tokio::time::timeout(Duration::from_secs(125), worker.shutdown()).await;
    eprintln!(
        "EP8 shutdown result={shutdown:?} elapsed={:.3}s",
        started.elapsed().as_secs_f64()
    );
    eprintln!("EP8 final worker snapshot={:?}", handle.snapshot());
    smoke.expect("EP8 HTTP lifecycle exceeded 900s").unwrap();
    shutdown
        .map_err(|_| "EP8 shutdown timed out".to_string())
        .and_then(|result| result.map_err(|error| format!("EP8 shutdown failed: {error:?}")))
        .unwrap();
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires eight exclusive GPUs and NAS 35B FP8; real 1k-token prefill, ctx=2048; run with external 1200s deadline"]
async fn qwen35_35b_fp8_ep8_http_context_1k_and_shutdown() {
    use ferrule_runtime::{BackendSelection, ModelFactoryOptions, ResidentModelPlanner};

    ferrule_common::observability::init_tracing();
    let started = Instant::now();
    let memory = |stage: &str| {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=index,memory.used,memory.total", "--format=csv"])
            .output();
        match output {
            Ok(output) => eprintln!(
                "EP8 context-1k stage={stage} elapsed={:.3}s memory status={}\n{}{}",
                started.elapsed().as_secs_f64(),
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            ),
            Err(error) => eprintln!("EP8 context-1k stage={stage} memory unavailable: {error}"),
        }
    };
    let processes = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader",
        ])
        .output()
        .expect("exclusive EP8 smoke requires nvidia-smi process preflight");
    assert!(processes.status.success(), "{processes:?}");
    assert!(
        processes.stdout.iter().all(u8::is_ascii_whitespace),
        "EP8 requires idle GPUs; refusing to interfere with existing processes: {}",
        String::from_utf8_lossy(&processes.stdout)
    );
    memory("before-build");
    let path = std::path::PathBuf::from(
        std::env::var_os("FERRULE_NUMERIC_FP8_MODEL_DIR")
            .unwrap_or_else(|| "/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8".into()),
    );
    let tokenizer = TokenizerHandle::load(&path).unwrap();
    let mut prompt = String::from("Read the following notes and continue the final sentence.\n");
    let long_prefill_tokens = loop {
        let count = tokenizer.encode(&prompt).unwrap().len();
        if count >= 1024 {
            break count;
        }
        prompt.push_str(
            "The river flows through the valley and the village stands beside the bridge.\n",
        );
    };
    assert!((1024..=1100).contains(&long_prefill_tokens));
    assert!(long_prefill_tokens + 2 <= 2048);
    let short_prefill_tokens = tokenizer.encode("Hello").unwrap().len();
    eprintln!(
        "EP8 context-1k long_prefill_tokens={long_prefill_tokens} prompt_bytes={} max_new_tokens=2 ctx_size=2048 max_active=2; serving smoke, NOT strict logits",
        prompt.len()
    );
    let config = ferrule_model::AutoConfig::from_pretrained(&path).unwrap();
    // Keep the concurrent smoke's policy caps; the factory's existing formula
    // must admit ctx=2048/max_active=2 without a test-only budget increase.
    let options = ModelFactoryOptions {
        max_layers: None,
        max_tensor_mebibytes: 1024,
        output_head_chunk_rows: 4096,
        expert_reader_max_tensor_mebibytes: 64,
        expert_cache: Default::default(),
        qwen35_moe_capacity: None,
        qwen35_host_cache: Some(
            ferrule_model::transformer::host_experts::HostExpertCacheOptions {
                max_bytes: 32u64 << 30,
                ..Default::default()
            },
        ),
        moe_hotset_experts: 0,
        kv_cache_mebibytes: Some(1024),
        scheduler_config: ferrule_runtime::ResidentSchedulerConfig {
            max_active_sequences: 2,
            max_decode_batch: 2,
            decode_cohort_target: 2,
            max_batch_tokens: 16,
            ..Default::default()
        },
        driver_config: ferrule_runtime::ResidentTopKDriverConfig {
            ctx_size: 2048,
            enable_native_proposals: false,
            ..Default::default()
        },
    };
    let plan = ResidentModelPlanner
        .prepare_qwen35_expert_parallel(
            &config,
            BackendSelection::Auto,
            None,
            options,
            ferrule_runtime::engine::model_factory::PipelineBuildOptions {
                parallelism: ferrule_common::ParallelismPlan {
                    expert_parallel: 8,
                    ..Default::default()
                },
                devices: Some((0..8).collect()),
                ..Default::default()
            },
        )
        .expect("EP8 context-1k planning failed; do not enlarge default budgets");
    assert!(plan.qwen35_expert_placement().is_some());
    assert!(plan.pipeline_options().is_none());
    assert!(plan.backend_profile().contains("fp8-f32-tf32x3"));
    let worker = spawn_model_worker_with(
        move || plan.build(),
        WorkerConfig {
            admission_timeout: Duration::from_secs(180),
            shutdown_timeout: Duration::from_secs(120),
            ..Default::default()
        },
    )
    .expect("EP8 context-1k build/admission failed with unchanged budgets");
    let handle = worker.handle();
    let stage = |label: &str| {
        eprintln!(
            "EP8 context-1k stage={label} elapsed={:.3}s worker={:?} admission={:?}",
            started.elapsed().as_secs_f64(),
            handle.snapshot(),
            handle.admission_snapshot()
        );
        memory(label);
    };
    stage("ready");
    let app = router(ServerState::new(
        ModelRegistration::new("hybrid", ChatTemplate::Qwen35),
        handle.clone(),
    ));
    let smoke = tokio::time::timeout(Duration::from_secs(900), async {
        let request = |prompt: &str| post(
            "/v1/completions",
            json!({
                "model":"hybrid", "prompt":prompt, "max_tokens":2,
                "ignore_eos":true, "stream":true,
                "stream_options":{"include_usage":true}
            }),
        );
        let verify = |label: &str, body: &str, prompt_tokens: usize| -> Result<(), String> {
            require_ep8_usage(label, body, 2)?;
            let mut usage_seen = false;
            let mut completion = String::new();
            for line in body.lines().filter_map(|line| line.strip_prefix("data: ")) {
                if line == "[DONE]" { continue; }
                let event: Value = serde_json::from_str(line).map_err(|error| error.to_string())?;
                if let Some(text) = event["choices"][0]["text"].as_str() {
                    completion.push_str(text);
                }
                if let Some(usage) = event.get("usage").filter(|usage| !usage.is_null()) {
                    if usage["prompt_tokens"] != json!(prompt_tokens)
                        || usage["completion_tokens"] != json!(2)
                        || usage["total_tokens"] != json!(prompt_tokens + 2)
                    {
                        return Err(format!("{label} incorrect exact token usage: {usage}"));
                    }
                    usage_seen = true;
                }
            }
            if !usage_seen || completion.is_empty() {
                return Err(format!("{label} missing exact usage or actual completion: {body}"));
            }
            Ok(())
        };
        let cleanup = || async {
            tokio::time::timeout(Duration::from_secs(30), async {
                while handle.admission_snapshot().held_requests != 0 {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }).await.map_err(|_| format!("cleanup permits not returned: {:?}", handle.admission_snapshot()))
        };
        let prefill_started = Instant::now();
        let (long, short) = tokio::join!(
            app.clone().oneshot(request(&prompt)),
            app.clone().oneshot(request("Hello")),
        );
        let long = long.map_err(|error| format!("long request failed: {error:?}"))?;
        let short = short.map_err(|error| format!("short request failed: {error:?}"))?;
        stage("long-short-submitted");
        let (long, short) = tokio::join!(collect_ep8_sse(long), collect_ep8_sse(short));
        verify("context-1k-long", &long?, long_prefill_tokens)?;
        verify("context-1k-short", &short?, short_prefill_tokens)?;
        eprintln!(
            "EP8 context-1k concurrent long_prefill_tokens={long_prefill_tokens} max_new_tokens=2 elapsed={:.3}s (no latency threshold)",
            prefill_started.elapsed().as_secs_f64()
        );
        cleanup().await?;
        stage("concurrent-cleanup-permit0");
        let future = app.oneshot(request("Hello")).await
            .map_err(|error| format!("future short request failed: {error:?}"))?;
        verify("context-1k-future-short", &collect_ep8_sse(future).await?, short_prefill_tokens)?;
        cleanup().await?;
        eprintln!("EP8 context-1k runtime admission={:?}", handle.runtime_admission_snapshot().await);
        stage("before-shutdown-cleanup-permit0");
        Ok::<(), String>(())
    }).await;
    stage("smoke-finished-before-shutdown");
    let before_shutdown = handle.admission_snapshot();
    let shutdown = tokio::time::timeout(Duration::from_secs(125), worker.shutdown()).await;
    eprintln!("EP8 context-1k smoke={smoke:?} shutdown={shutdown:?}");
    stage("shutdown-complete");
    smoke.expect("EP8 context-1k smoke exceeded 900s").unwrap();
    assert_eq!(
        before_shutdown.held_requests, 0,
        "cleanup must precede shutdown"
    );
    assert_eq!(before_shutdown.held_prompt_bytes, 0);
    shutdown
        .expect("EP8 context-1k shutdown timed out")
        .unwrap();
    assert!(handle.snapshot().first_fatal.is_none());
    assert!(!handle.snapshot().shutdown_report.quarantine);
}

#[cfg(feature = "cuda")]
async fn collect_ep8_sse(response: axum::response::Response) -> Result<String, String> {
    if response.status() != StatusCode::OK {
        return Err(format!("SSE status: {}", response.status()));
    }
    let body = tokio::time::timeout(Duration::from_secs(750), response.into_body().collect())
        .await
        .map_err(|_| "SSE body timed out".to_string())?
        .map_err(|error| format!("SSE body failed: {error:?}"))?;
    String::from_utf8(body.to_bytes().to_vec())
        .map_err(|error| format!("SSE was not UTF-8: {error}"))
}

#[cfg(feature = "cuda")]
fn require_ep8_usage(label: &str, body: &str, expected_tokens: usize) -> Result<(), String> {
    let needle = format!("\"completion_tokens\":{expected_tokens}");
    if !body.contains(&needle)
        || !body.contains("\"finish_reason\":\"length\"")
        || body.contains("\"error\":")
        || body.matches("data: [DONE]").count() != 1
    {
        return Err(format!("{label} missing usage/DONE: {body}"));
    }
    eprintln!(
        "EP8 request={label} completion_tokens={expected_tokens} done=true sse_bytes={} SSE:\n{body}",
        body.len()
    );
    Ok(())
}

#[cfg(feature = "cuda")]
async fn qwen35_35b_http_smoke(ep: bool) {
    use ferrule_runtime::{BackendSelection, ModelFactoryOptions, ResidentModelPlanner};
    let path = std::env::var_os("FERRULE_NUMERIC_FP8_MODEL_DIR")
        .unwrap_or_else(|| "/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8".into());
    let config = ferrule_model::AutoConfig::from_pretrained(path).unwrap();
    let options = ModelFactoryOptions {
        max_layers: None,
        max_tensor_mebibytes: 1024,
        output_head_chunk_rows: 4096,
        expert_reader_max_tensor_mebibytes: 64,
        expert_cache: Default::default(),
        qwen35_moe_capacity: None,
        qwen35_host_cache: ep.then(|| {
            ferrule_model::transformer::host_experts::HostExpertCacheOptions {
                max_bytes: 32u64 << 30,
                ..Default::default()
            }
        }),
        moe_hotset_experts: 0,
        kv_cache_mebibytes: Some(1024),
        scheduler_config: ferrule_runtime::ResidentSchedulerConfig::default(),
        driver_config: ferrule_runtime::ResidentTopKDriverConfig {
            ctx_size: 1024,
            enable_native_proposals: false,
            ..Default::default()
        },
    };
    let plan = if ep {
        ResidentModelPlanner
            .prepare_qwen35_expert_parallel(
                &config,
                BackendSelection::Auto,
                None,
                options,
                ferrule_runtime::engine::model_factory::PipelineBuildOptions {
                    parallelism: ferrule_common::ParallelismPlan {
                        expert_parallel: 8,
                        ..Default::default()
                    },
                    devices: Some((0..8).collect()),
                    ..Default::default()
                },
            )
            .unwrap()
    } else {
        ResidentModelPlanner
            .prepare(&config, BackendSelection::Auto, None, options)
            .unwrap()
    };
    assert_eq!(plan.qwen35_expert_placement().is_some(), ep);
    assert!(plan.pipeline_options().is_none());
    assert!(plan.backend_profile().contains("fp8-f32-tf32x3"));
    let started = std::time::Instant::now();
    let worker = spawn_model_worker_with(move || plan.build(), WorkerConfig::default()).unwrap();
    eprintln!(
        "EP={ep} all owners/root Ready in {:.3}s",
        started.elapsed().as_secs_f64()
    );
    #[cfg(target_os = "linux")]
    let io_before = std::fs::read_to_string("/proc/self/io").unwrap();
    let app = router(ServerState::new(
        ModelRegistration::new("hybrid", ChatTemplate::Qwen35),
        worker.handle(),
    ));
    let prompts: &[(&str, &str)] = if ep {
        &[("Hello", ",")]
    } else {
        &[("Hello", ","), ("The capital of France is", " Paris")]
    };
    for &(prompt, expected) in prompts {
        let response = app
            .clone()
            .oneshot(post(
                "/v1/completions",
                json!({
                    "model":"hybrid", "prompt":prompt, "max_tokens":1,
                    "stream":true, "stream_options":{"include_usage":true}
                }),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = tokio::time::timeout(Duration::from_secs(750), response.into_body().collect())
            .await
            .unwrap()
            .unwrap()
            .to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        eprintln!(
            "EP={ep} prompt={prompt:?} elapsed={:.3}s SSE:\n{text}",
            started.elapsed().as_secs_f64()
        );
        assert!(text.contains(&format!("\"text\":\"{expected}\"")), "{text}");
        assert!(text.contains("\"finish_reason\":\"length\""), "{text}");
        assert!(text.contains("\"completion_tokens\":1"), "{text}");
        assert!(text.contains("data: [DONE]"), "{text}");
    }
    #[cfg(target_os = "linux")]
    eprintln!(
        "process I/O at Ready:\n{io_before}process I/O after SSE:\n{}",
        std::fs::read_to_string("/proc/self/io").unwrap()
    );
    worker.shutdown().await.unwrap();
    eprintln!(
        "EP={ep} graceful shutdown complete elapsed={:.3}s",
        started.elapsed().as_secs_f64()
    );
}

#[tokio::test]
async fn real_cpu_resident_admission_capacity_tracks_live_owners() {
    let fixture = fixture::Fixture::new();
    let worker = spawn_model_worker_with(
        move || {
            let mut engine = cpu_engine(fixture)?;
            let initial = engine.capacity_snapshot();
            let scheduler = initial.scheduler.unwrap();
            assert_eq!(
                (scheduler.active_sequences, scheduler.waiting_requests),
                (0, 0)
            );
            assert_eq!(
                (scheduler.max_active_sequences, scheduler.max_decode_batch),
                (1, 1)
            );
            assert_eq!(scheduler.max_batch_tokens, Some(4));
            assert_eq!(scheduler.prefill_chunk_size, 2);
            let kv = initial.kv.unwrap();
            assert_eq!(kv.logical_capacity_pages, Some(16));
            assert_eq!(kv.logical_free_pages, Some(16));
            assert_eq!(kv.logical_allocated_pages, 0);
            assert_eq!(kv.recycled_pages, 0);
            assert!(kv.physical.is_none());
            for id in 7001..=7002 {
                engine.try_submit(GenerateRequest {
                    id: RequestId(id),
                    session_id: Some(SessionId(id)),
                    prompt_tokens: engine.encode("t1 t3 t2")?,
                    max_new_tokens: 2,
                    stop: vec![],
                    ignore_eos: true,
                })?;
            }
            let waiting = engine.capacity_snapshot();
            assert_eq!(waiting.scheduler.unwrap().waiting_requests, 2);
            assert_eq!(waiting.scheduler.unwrap().active_sequences, 0);
            engine.step(&mut |_| Ok(()))?;
            let active = engine.capacity_snapshot();
            assert_eq!(active.scheduler.unwrap().active_sequences, 1);
            assert_eq!(active.scheduler.unwrap().waiting_requests, 1);
            assert_eq!(engine.admission_snapshot().unwrap().waiting_requests, 1);
            let kv = active.kv.unwrap();
            assert!(kv.logical_allocated_pages > 0);
            assert_eq!(
                kv.logical_allocated_pages + kv.logical_free_pages.unwrap(),
                16
            );
            assert_eq!(active, engine.capacity_snapshot());
            for id in 7001..=7002 {
                engine.cancel_request(RequestId(id))?;
            }
            engine.drain_cancelled();
            assert!(engine.drain_failed().is_empty());
            let cleaned = engine.capacity_snapshot();
            assert_eq!(cleaned.scheduler, initial.scheduler);
            assert_eq!(cleaned.kv.unwrap().logical_allocated_pages, 0);
            assert_eq!(cleaned.kv.unwrap().logical_free_pages, Some(16));

            // Retained KV proves that an idle scheduler is not an empty KV owner.
            engine.retain_session(SessionId(8001))?;
            engine.try_submit(GenerateRequest {
                id: RequestId(8001),
                session_id: Some(SessionId(8001)),
                prompt_tokens: engine.encode("t1 t3 t2")?,
                max_new_tokens: 1,
                stop: vec![],
                ignore_eos: true,
            })?;
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                engine.step(&mut |_| Ok(()))?;
                assert!(engine.drain_failed().is_empty());
                if !engine.drain_finished().is_empty() {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "resident snapshot setup deadline"
                );
            }
            assert!(
                engine
                    .capacity_snapshot()
                    .kv
                    .unwrap()
                    .logical_allocated_pages
                    > 0
            );
            Ok::<_, ferrule_runtime::Error>(engine)
        },
        WorkerConfig::default(),
    )
    .unwrap();
    let handle = worker.handle();
    let app = router(ServerState::new(
        ModelRegistration::new("hybrid", ChatTemplate::Plain),
        handle.clone(),
    ));
    let mut previous = None;
    for _ in 0..3 {
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            app.clone().oneshot(
                Request::builder()
                    .uri("/admission")
                    .body(Body::empty())
                    .unwrap(),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        let value: Value = serde_json::from_str(&text(response).await).unwrap();
        eprintln!("admission semantic snapshot: {value}");
        assert_eq!(value["runtime"]["status"], "available");
        assert_eq!(
            value["runtime"]["waiting_requests"]["held"],
            value["scheduler"]["waiting_requests"]
        );
        assert_eq!(value["scheduler"]["active_sequences"], 0);
        assert_eq!(value["scheduler"]["waiting_requests"], 0);
        assert_eq!(value["scheduler"]["max_active_sequences"], 1);
        assert_eq!(value["scheduler"]["max_batch_tokens"], 4);
        assert_eq!(value["scheduler"]["prefill_chunk_size"], 2);
        let kv = &value["kv"];
        assert_eq!(kv["logical_capacity_pages"], 16);
        assert!(kv["logical_allocated_pages"].as_u64().unwrap() > 0);
        assert_eq!(
            kv["logical_allocated_pages"].as_u64().unwrap()
                + kv["logical_free_pages"].as_u64().unwrap(),
            16
        );
        assert!(kv["physical"].is_null());
        assert_eq!(handle.admission_snapshot().held_requests, 0);
        if let Some(previous) = previous {
            assert_eq!(value, previous);
        }
        previous = Some(value);
    }
    tokio::time::timeout(Duration::from_secs(5), worker.shutdown())
        .await
        .unwrap()
        .unwrap();
}

// Reproduce a decorator that omits request_cleanup, while generation and cleanup
// still run through the real resident engine. The captured receipts are read-only.
struct MissingCleanupForwarding {
    inner: Observed,
    receipts: Arc<std::sync::Mutex<Vec<ferrule_runtime::RequestCleanupReceipt>>>,
}
impl InferenceEngine for MissingCleanupForwarding {
    fn completion_hub(&self) -> CompletionHub {
        self.inner.completion_hub()
    }
    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        self.inner.take_completion_reactors()
    }
    fn has_pending_async_work(&self) -> bool {
        self.inner.has_pending_async_work()
    }
    fn encode(&self, prompt: &str) -> RuntimeResult<Vec<u32>> {
        self.inner.encode(prompt)
    }
    fn submit(&mut self, request: GenerateRequest) {
        self.inner.submit(request);
    }
    fn try_submit(&mut self, request: GenerateRequest) -> RuntimeResult<()> {
        let id = request.id;
        self.inner.try_submit(request)?;
        let ferrule_runtime::InferenceRequestCleanup::Tracked(receipt) =
            self.inner.request_cleanup(id)
        else {
            panic!("real resident owner must issue an exact receipt")
        };
        assert!(!receipt.is_released());
        self.receipts.lock().unwrap().push(receipt);
        assert!(matches!(
            self.request_cleanup(id),
            ferrule_runtime::InferenceRequestCleanup::Unavailable
        ));
        Ok(())
    }
    fn step(
        &mut self,
        emit: &mut dyn FnMut(&ResidentTokenEvent) -> RuntimeResult<()>,
    ) -> RuntimeResult<ResidentDriverStep> {
        self.inner.step(emit)
    }
    fn cancel_request(&mut self, id: RequestId) -> RuntimeResult<InferenceCancelProgress> {
        self.inner.cancel_request(id)
    }
    fn drain_finished(&mut self) -> Vec<SequenceState> {
        self.inner.drain_finished()
    }
    fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        self.inner.drain_cancelled()
    }
    fn drain_failed(&mut self) -> Vec<SequenceState> {
        self.inner.drain_failed()
    }
    fn shutdown(&mut self) -> RuntimeResult<InferenceShutdownProgress> {
        self.inner.shutdown()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_cpu_missing_cleanup_forwarding_holds_permits_until_shutdown() {
    let fixture = fixture::Fixture::new();
    let probe = Arc::new(Probe::default());
    let owner_probe = probe.clone();
    let receipts = Arc::new(std::sync::Mutex::new(Vec::new()));
    let owner_receipts = receipts.clone();
    let worker = spawn_model_worker_with(
        move || {
            Ok::<_, ferrule_runtime::Error>(MissingCleanupForwarding {
                inner: Observed {
                    inner: cpu_engine(fixture)?,
                    probe: owner_probe,
                },
                receipts: owner_receipts,
            })
        },
        WorkerConfig {
            event_queue_capacity: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let handle = worker.handle();
    let app = router(ServerState::new(
        ModelRegistration::new("hybrid", ChatTemplate::Plain),
        handle.clone(),
    ));
    let request = |max_tokens| {
        post(
            "/v1/completions",
            json!({
                "model":"hybrid", "prompt":"t1 t3 t2", "max_tokens":max_tokens,
                "ignore_eos":true, "stream":true, "stream_options":{"include_usage":true}
            }),
        )
    };
    // BodyExt::collect consumes the whole stream, as in the EP8 test.
    for _ in 0..2 {
        let body = text(app.clone().oneshot(request(1)).await.unwrap()).await;
        assert_eq!(body.matches("data: [DONE]").count(), 1);
        assert!(body.contains("\"completion_tokens\":1"), "{body}");
    }
    let before = probe.tokens.load(Ordering::Acquire);
    let response = app.clone().oneshot(request(24)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    tokio::time::timeout(Duration::from_secs(5), async {
        while probe.tokens.load(Ordering::Acquire) == before {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    drop(response);
    tokio::time::timeout(Duration::from_secs(5), async {
        while probe.cancelled.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let independent = text(app.oneshot(request(1)).await.unwrap()).await;
    assert_eq!(independent.matches("data: [DONE]").count(), 1);
    assert!(independent.contains("\"completion_tokens\":1"));
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let released = {
                let receipts = receipts.lock().unwrap();
                receipts.len() == 4 && receipts.iter().all(|receipt| receipt.is_released())
            };
            if released {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the real runtime must release all four receipts");
    // Only plain Strings survive; repeated owner commands cannot repair the
    // capability lost at admission. Never infer release from runtime idle.
    for _ in 0..3 {
        assert!(handle.runtime_admission_snapshot().await.unwrap().is_none());
        assert_eq!(handle.admission_snapshot().held_requests, 4);
        assert_eq!(handle.admission_snapshot().held_prompt_bytes, 0);
        assert_eq!(handle.phase(), ferrule_server::WorkerPhase::Ready);
        assert!(handle.snapshot().first_fatal.is_none());
    }
    eprintln!(
        "missing forwarding: 4 real receipts released, HTTP bodies dropped, {:?}",
        handle.admission_snapshot()
    );
    tokio::time::timeout(Duration::from_secs(5), worker.shutdown())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(handle.admission_snapshot().held_requests, 0);
    assert_eq!(handle.phase(), ferrule_server::WorkerPhase::Stopped);
    assert!(!handle.snapshot().shutdown_report.quarantine);
    assert_eq!(probe.shutdown.load(Ordering::Acquire), 1);
}

#[test]
fn real_cpu_retained_session_releases_turn_receipts_without_reset() {
    let mut engine = cpu_engine(fixture::Fixture::new()).unwrap();
    let session = SessionId(42);
    engine.retain_session(session).unwrap();
    let mut previous: Option<ferrule_runtime::RequestCleanupReceipt> = None;
    for round in 0..2 {
        // Reuse the request identity with a new receipt, retaining session KV.
        let id = RequestId(7);
        engine
            .try_submit(GenerateRequest {
                id,
                session_id: Some(session),
                prompt_tokens: engine.encode("t1 t3 t2").unwrap(),
                max_new_tokens: 1,
                stop: vec![],
                ignore_eos: true,
            })
            .unwrap();
        let ferrule_runtime::InferenceRequestCleanup::Tracked(receipt) = engine.request_cleanup(id)
        else {
            panic!("missing actual resident receipt")
        };
        assert!(!receipt.is_released());
        if let Some(old) = &previous {
            assert!(old.is_released());
        }
        let mut tokens = 0;
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            engine
                .step(&mut |_| {
                    tokens += 1;
                    Ok(())
                })
                .unwrap();
            assert!(!receipt.is_released(), "terminal has not been consumed");
            if let Some(terminal) = engine.take_request_terminal(id) {
                assert!(matches!(
                    terminal,
                    ferrule_runtime::RequestTerminal::Finished(_)
                ));
                break;
            }
            assert!(Instant::now() < deadline);
        }
        assert_eq!(tokens, 1);
        assert!(
            receipt.is_released(),
            "retained KV is session-owned, not turn custody"
        );
        let snapshot = engine.admission_snapshot().unwrap();
        assert_eq!(snapshot.request_identities_held, 0);
        assert_eq!(snapshot.session_identities_held, 1);
        assert_eq!(
            engine.retained_session_position(session),
            Some((round + 1) * 4)
        );
        assert!(
            engine
                .capacity_snapshot()
                .kv
                .unwrap()
                .logical_allocated_pages
                > 0
        );
        previous = Some(receipt);
    }
    assert_eq!(
        engine.shutdown().unwrap(),
        InferenceShutdownProgress::Complete
    );
    assert_eq!(
        engine
            .capacity_snapshot()
            .kv
            .unwrap()
            .logical_allocated_pages,
        0
    );
}
