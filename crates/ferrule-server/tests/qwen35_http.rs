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
