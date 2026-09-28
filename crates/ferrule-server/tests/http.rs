use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use ferrule_common::CompletionHub;
use ferrule_model::ChatTemplate;
use ferrule_runtime::{
    CancelRequestResult, GenerateRequest, InferenceCancelProgress, InferenceCompletionReactor,
    InferenceEngine, RequestId, ResidentDriverStep, ResidentTokenEvent, Result as RuntimeResult,
    SequenceFinishReason, SequenceState,
};
use ferrule_server::{
    ModelRegistration, ModelWorker, ServerState, WorkerConfig, router, spawn_model_worker_with,
};
use http_body_util::BodyExt;
use tower::ServiceExt;

#[derive(Default)]
struct ImmediateEngine {
    completion_hub: CompletionHub,
    request: Option<GenerateRequest>,
    finished: Vec<SequenceState>,
    cancelled: Vec<SequenceState>,
    admission: Option<ferrule_runtime::RuntimeAdmissionOptions>,
    owner: Option<std::thread::ThreadId>,
}

impl InferenceEngine for ImmediateEngine {
    fn set_admission_options(
        &mut self,
        options: ferrule_runtime::RuntimeAdmissionOptions,
    ) -> RuntimeResult<()> {
        assert_eq!(self.owner, Some(std::thread::current().id()));
        self.admission = Some(options.validate()?);
        Ok(())
    }
    fn admission_snapshot(&self) -> Option<ferrule_runtime::RuntimeAdmissionSnapshot> {
        self.admission.map(|limits| {
            assert_eq!(self.owner, Some(std::thread::current().id()));
            ferrule_runtime::RuntimeAdmissionSnapshot {
                limits,
                waiting_requests: usize::from(self.request.is_some()),
                request_identities_held: usize::from(self.request.is_some())
                    + self.finished.len()
                    + self.cancelled.len(),
                session_identities_held: usize::from(self.request.is_some())
                    + self.finished.len()
                    + self.cancelled.len(),
                closed: false,
            }
        })
    }

    fn request_cleanup(&self, _: RequestId) -> ferrule_runtime::InferenceRequestCleanup {
        ferrule_runtime::InferenceRequestCleanup::TerminalQuiescent
    }
    fn close_admission(&mut self) -> RuntimeResult<()> {
        Ok(())
    }

    fn completion_hub(&self) -> CompletionHub {
        self.completion_hub.clone()
    }

    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        Vec::new()
    }

    fn has_pending_async_work(&self) -> bool {
        false
    }

    fn encode(&self, prompt: &str) -> RuntimeResult<Vec<u32>> {
        Ok(prompt.bytes().map(u32::from).collect())
    }

    fn submit(&mut self, request: GenerateRequest) {
        self.request = Some(request);
    }

    fn step(
        &mut self,
        on_token: &mut dyn FnMut(&ResidentTokenEvent) -> RuntimeResult<()>,
    ) -> RuntimeResult<ResidentDriverStep> {
        let Some(request) = self.request.take() else {
            return Ok(ResidentDriverStep::Idle);
        };
        let session_id = request.session_id.expect("worker assigns a session");
        on_token(&ResidentTokenEvent {
            session_id,
            request_id: Some(request.id),
            index: 0,
            token: 42,
            logit: Some(1.0),
            text: "ok".into(),
        })?;
        let mut state = SequenceState::from_request(&request, session_id);
        state.generated = 1;
        state.finish_reason = Some(SequenceFinishReason::MaxTokens);
        self.finished.push(state);
        Ok(ResidentDriverStep::Executed {
            action_kind: ferrule_runtime::ResidentActionKind::Decode,
            rows: 1,
            staged: 1,
            finished: 1,
        })
    }

    fn cancel_request(&mut self, request_id: RequestId) -> RuntimeResult<InferenceCancelProgress> {
        let Some(request) = self.request.take() else {
            return Ok(InferenceCancelProgress::Complete(
                CancelRequestResult::NotFound { request_id },
            ));
        };
        let session_id = request.session_id.expect("worker assigns a session");
        let mut state = SequenceState::from_request(&request, session_id);
        state.finish_reason = Some(SequenceFinishReason::Cancelled);
        self.cancelled.push(state);
        Ok(InferenceCancelProgress::Complete(
            CancelRequestResult::Waiting {
                request_id,
                session_id,
            },
        ))
    }

    fn drain_finished(&mut self) -> Vec<SequenceState> {
        std::mem::take(&mut self.finished)
    }

    fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        std::mem::take(&mut self.cancelled)
    }

    fn drain_failed(&mut self) -> Vec<SequenceState> {
        Vec::new()
    }
}

fn test_state() -> (ServerState, ModelWorker) {
    let worker = spawn_model_worker_with(
        || Ok::<ImmediateEngine, std::convert::Infallible>(ImmediateEngine::default()),
        WorkerConfig::default(),
    )
    .unwrap();
    let state = ServerState::new(
        ModelRegistration::new("test-model", ChatTemplate::Plain),
        worker.handle(),
    );
    (state, worker)
}

async fn response_text(response: axum::response::Response) -> String {
    let body = response.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(body.to_vec()).unwrap()
}

#[tokio::test]
async fn readiness_and_admission_follow_worker_phase() {
    let (state, worker) = test_state();
    let handle = worker.handle();
    let app = router(state);

    let ready = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ready.status(), StatusCode::OK);
    let live = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health/live")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(live.status(), StatusCode::OK);

    handle.begin_shutdown();
    let ready = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ready.status(), StatusCode::SERVICE_UNAVAILABLE);
    let tokenize = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model":"test-model","prompt":"hello"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(tokenize.status(), StatusCode::SERVICE_UNAVAILABLE);
    worker.shutdown().await.unwrap();
    assert_eq!(handle.phase(), ferrule_server::WorkerPhase::Stopped);
    let stopped = app
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(stopped.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn real_router_serves_models_and_openai_errors() {
    let (state, worker) = test_state();
    let app = router(state);

    let models = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(models.status(), StatusCode::OK);
    let models: serde_json::Value = serde_json::from_str(&response_text(models).await).unwrap();
    assert_eq!(models["data"][0]["id"], "test-model");

    let invalid = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "tools": []
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
    let invalid: serde_json::Value = serde_json::from_str(&response_text(invalid).await).unwrap();
    assert_eq!(invalid["error"]["type"], "invalid_request_error");
    assert_eq!(invalid["error"]["message"], "tool calling is not supported");

    worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn real_chat_and_completion_sse_emit_terminal_frames() {
    for (uri, body, object) in [
        (
            "/v1/chat/completions",
            serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 1,
                "stream": true,
                "stream_options": {"include_usage": true}
            }),
            "chat.completion.chunk",
        ),
        (
            "/v1/completions",
            serde_json::json!({
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 1,
                "stream": true,
                "stream_options": {"include_usage": true}
            }),
            "text_completion",
        ),
    ] {
        let (state, worker) = test_state();
        let response = router(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let body = response_text(response).await;
        assert!(body.contains(object));
        assert!(body.contains("\"finish_reason\":\"length\""));
        assert!(body.contains("\"choices\":[],\"usage\""));
        assert!(body.contains("data: [DONE]"));
        worker.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn tokenize_routes_share_the_typed_worker_boundary() {
    for uri in ["/v1/tokenize", "/tokenize"] {
        let (state, worker) = test_state();
        let response = router(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::json!({"model": "test-model", "prompt": "hello"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let value: serde_json::Value =
            serde_json::from_str(&response_text(response).await).unwrap();
        assert_eq!(value["data"][0]["count"], 5);
        assert_eq!(value["data"][0]["tokens"][0], u64::from(b'h'));
        worker.shutdown().await.unwrap();
    }
}

fn pr13_state(config: WorkerConfig) -> (axum::Router, ModelWorker) {
    let worker = spawn_model_worker_with(
        || Ok::<ImmediateEngine, std::convert::Infallible>(ImmediateEngine::default()),
        config,
    )
    .unwrap();
    let app = router(ServerState::new(
        ModelRegistration::new("test-model", ChatTemplate::Plain),
        worker.handle(),
    ));
    (app, worker)
}

fn pr13_request(path: &str, body: impl Into<Body>) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(body.into())
        .unwrap()
}

fn pr13_counts(handle: &ferrule_server::ModelWorkerHandle, requests: usize, bytes: usize) {
    let snapshot = handle.admission_snapshot();
    assert_eq!(snapshot.held_requests, requests, "{snapshot:?}");
    assert_eq!(snapshot.held_prompt_bytes, bytes, "{snapshot:?}");
    assert_eq!(
        snapshot.held_requests + snapshot.available_requests,
        snapshot.request_limit
    );
    assert_eq!(
        snapshot.held_prompt_bytes + snapshot.available_prompt_bytes,
        snapshot.prompt_bytes_limit
    );
    eprintln!("PR13 HTTP ledger={snapshot:?}");
}

struct Pr13PendingBody {
    polls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    first: Option<axum::body::Bytes>,
}
impl futures_core::Stream for Pr13PendingBody {
    type Item = Result<axum::body::Bytes, std::io::Error>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.polls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        match self.first.take() {
            Some(bytes) => std::task::Poll::Ready(Some(Ok(bytes))),
            None => std::task::Poll::Pending,
        }
    }
}

#[tokio::test]
async fn pr13_body_drop_and_capacity_reject_before_body_poll() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let (app, worker) = pr13_state(WorkerConfig {
        max_inflight_requests: 1,
        max_prompt_bytes: 64,
        ..Default::default()
    });
    let handle = worker.handle();
    let polls = Arc::new(AtomicUsize::new(0));
    let body = Body::from_stream(Pr13PendingBody {
        polls: polls.clone(),
        first: Some(axum::body::Bytes::from_static(b"{ ")),
    });
    let mut pending = Box::pin(app.clone().oneshot(pr13_request("/tokenize", body)));
    assert!(
        std::future::poll_fn(|cx| std::task::Poll::Ready(pending.as_mut().poll(cx)))
            .await
            .is_pending()
    );
    assert!(polls.load(Ordering::Relaxed) > 0);
    pr13_counts(&handle, 1, 64);
    let rejected_polls = Arc::new(AtomicUsize::new(0));
    let response = app
        .clone()
        .oneshot(pr13_request(
            "/tokenize",
            Body::from_stream(Pr13PendingBody {
                polls: rejected_polls.clone(),
                first: None,
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(rejected_polls.load(Ordering::Relaxed), 0);
    let body = response_text(response).await;
    assert!(body.contains("request_capacity"), "{body}");
    pr13_counts(&handle, 1, 64);
    drop(pending);
    pr13_counts(&handle, 0, 0);
    let snapshot = app
        .oneshot(
            Request::builder()
                .uri("/admission")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_str(&response_text(snapshot).await).unwrap();
    assert_eq!(value["requests"]["limit"], 1);
    assert_eq!(value["requests"]["held"], 0);
    assert_eq!(value["prompt_bytes"]["available"], 64);
    assert_eq!(value["runtime"]["status"], "unsupported");
    assert!(value["scheduler"].is_null());
    assert!(value["kv"].is_null());
    worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn pr13_prompt_body_json_rejection_matrix_returns_permits() {
    for (bytes_limit, body_limit, input, status, code) in [
        (
            8,
            128,
            r#"{"model":"test-model","prompt":"hello"}"#,
            StatusCode::TOO_MANY_REQUESTS,
            "prompt_bytes_capacity",
        ),
        (
            128,
            8,
            r#"{"model":"test-model","prompt":"hello"}"#,
            StatusCode::PAYLOAD_TOO_LARGE,
            "body_too_large",
        ),
        (
            128,
            128,
            r#"{"model":"secret/private-model","prompt":[]}"#,
            StatusCode::BAD_REQUEST,
            "invalid_json",
        ),
        (
            128,
            128,
            r#"{"model":"secret/private-model","prompt":"hi"}"#,
            StatusCode::BAD_REQUEST,
            "model_not_served",
        ),
    ] {
        let (app, worker) = pr13_state(WorkerConfig {
            max_prompt_bytes: bytes_limit,
            max_body_bytes: body_limit,
            ..Default::default()
        });
        let handle = worker.handle();
        let response = app.oneshot(pr13_request("/tokenize", input)).await.unwrap();
        assert_eq!(response.status(), status);
        let body = response_text(response).await;
        assert!(body.contains(code), "{body}");
        assert!(!body.contains("private-model"), "{body}");
        pr13_counts(&handle, 0, 0);
        worker.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn pr13_legacy_http_reuses_small_budget_for_ten_rounds() {
    let (app, worker) = pr13_state(WorkerConfig {
        max_inflight_requests: 2,
        ..Default::default()
    });
    let handle = worker.handle();
    for round in 0..10 {
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            app.clone().oneshot(pr13_request(
                "/v1/completions",
                r#"{"model":"test-model","prompt":"hello","max_tokens":1}"#,
            )),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "round {round}");
        let body = response_text(response).await;
        assert!(body.contains("choices"));
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while handle.admission_snapshot().held_requests != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        pr13_counts(&handle, 0, 0);
    }
    worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn pr13_slow_sse_holds_request_but_releases_prompt_and_reuses_after_consumption() {
    let (app, worker) = pr13_state(WorkerConfig {
        max_inflight_requests: 1,
        ..Default::default()
    });
    let handle = worker.handle();
    let request = r#"{"model":"test-model","prompt":"hello","max_tokens":1,"stream":true}"#;
    for _ in 0..10 {
        let response = app
            .clone()
            .oneshot(pr13_request("/v1/completions", request))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        // No SSE polling: the observer owns the same counted lease after runtime completes.
        pr13_counts(&handle, 1, 0);
        let rejected = app
            .clone()
            .oneshot(pr13_request(
                "/tokenize",
                r#"{"model":"test-model","prompt":"hi"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"));
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while handle.admission_snapshot().held_requests != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        pr13_counts(&handle, 0, 0);
    }
    worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn pr13_body_io_error_and_missing_content_type_are_sanitized() {
    struct Fault;
    impl futures_core::Stream for Fault {
        type Item = Result<axum::body::Bytes, std::io::Error>;
        fn poll_next(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            std::task::Poll::Ready(Some(Err(std::io::Error::other("/private/body-source"))))
        }
    }
    let (app, worker) = pr13_state(Default::default());
    let response = app
        .clone()
        .oneshot(pr13_request("/tokenize", Body::from_stream(Fault)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response_text(response).await;
    assert!(body.contains("invalid_body"));
    assert!(!body.contains("body-source"));
    pr13_counts(&worker.handle(), 0, 0);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
    assert!(
        response_text(response)
            .await
            .contains("invalid_content_type")
    );
    pr13_counts(&worker.handle(), 0, 0);
    worker.shutdown().await.unwrap();
}

#[derive(Default)]
struct Pr13FaultEngine {
    inner: ImmediateEngine,
    tokenization_failure: bool,
}
impl InferenceEngine for Pr13FaultEngine {
    fn completion_hub(&self) -> CompletionHub {
        self.inner.completion_hub()
    }
    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        vec![]
    }
    fn has_pending_async_work(&self) -> bool {
        false
    }
    fn encode(&self, prompt: &str) -> RuntimeResult<Vec<u32>> {
        if self.tokenization_failure {
            Err(ferrule_runtime::Error::Invariant {
                message: "/private/provider-secret/tokenizer".into(),
            })
        } else {
            self.inner.encode(prompt)
        }
    }
    fn submit(&mut self, request: GenerateRequest) {
        self.inner.submit(request);
    }
    fn step(
        &mut self,
        _: &mut dyn FnMut(&ResidentTokenEvent) -> RuntimeResult<()>,
    ) -> RuntimeResult<ResidentDriverStep> {
        Err(ferrule_runtime::Error::Invariant {
            message: "/private/provider-secret/execution".into(),
        })
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
        vec![]
    }
}

#[tokio::test]
async fn pr13_json_sse_execution_errors_share_contract_without_source_leakage() {
    let mut reference = None;
    for stream in [false, true] {
        let worker = spawn_model_worker_with(
            || Ok::<_, std::convert::Infallible>(Pr13FaultEngine::default()),
            Default::default(),
        )
        .unwrap();
        let app = router(ServerState::new(
            ModelRegistration::new("test-model", ChatTemplate::Plain),
            worker.handle(),
        ));
        let response = app.oneshot(pr13_request("/v1/completions", serde_json::json!({ "model": "test-model", "prompt": "hello", "max_tokens": 1, "stream": stream }).to_string())).await.unwrap();
        assert_eq!(
            response.status(),
            if stream {
                StatusCode::OK
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        );
        let body = response_text(response).await;
        assert!(!body.contains("provider-secret"), "{body}");
        let value: serde_json::Value = if stream {
            let line = body
                .lines()
                .find_map(|line| {
                    line.strip_prefix("data: ")
                        .filter(|data| data.contains("\"error\""))
                })
                .unwrap();
            assert_eq!(body.matches("data: [DONE]").count(), 1);
            serde_json::from_str(line).unwrap()
        } else {
            serde_json::from_str(&body).unwrap()
        };
        assert_eq!(value["error"]["code"], "execution_failed");
        assert_eq!(value["error"]["type"], "server_error");
        if let Some(reference) = &reference {
            assert_eq!(&value, reference);
        } else {
            reference = Some(value);
        }
        let failure = worker.shutdown().await.unwrap_err();
        assert!(failure.to_string().contains("provider-secret"));
    }
}

#[tokio::test]
async fn pr13_http_tokenization_failure_returns_permits_and_preserves_readiness() {
    for endpoint in ["/tokenize", "/v1/completions"] {
        let worker = spawn_model_worker_with(
            || {
                Ok::<_, std::convert::Infallible>(Pr13FaultEngine {
                    tokenization_failure: true,
                    ..Default::default()
                })
            },
            Default::default(),
        )
        .unwrap();
        let handle = worker.handle();
        let app = router(ServerState::new(
            ModelRegistration::new("test-model", ChatTemplate::Plain),
            handle.clone(),
        ));
        let response = app
            .clone()
            .oneshot(pr13_request(
                endpoint,
                r#"{"model":"test-model","prompt":"hi"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = response_text(response).await;
        assert!(body.contains("tokenization_failed"));
        assert!(!body.contains("provider-secret"));
        pr13_counts(&handle, 0, 0);
        let ready = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ready.status(), StatusCode::OK);
        worker.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn pr13_prompt_budget_exhaustion_rejects_before_poll_even_with_request_capacity() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let (app, worker) = pr13_state(WorkerConfig {
        max_inflight_requests: 2,
        max_prompt_bytes: 64,
        ..Default::default()
    });
    let mut first = Box::pin(app.clone().oneshot(pr13_request(
        "/tokenize",
        Body::from_stream(Pr13PendingBody {
            polls: Arc::new(AtomicUsize::new(0)),
            first: None,
        }),
    )));
    assert!(
        std::future::poll_fn(|cx| std::task::Poll::Ready(first.as_mut().poll(cx)))
            .await
            .is_pending()
    );
    pr13_counts(&worker.handle(), 1, 64);
    let polls = Arc::new(AtomicUsize::new(0));
    let response = app
        .oneshot(pr13_request(
            "/tokenize",
            Body::from_stream(Pr13PendingBody {
                polls: polls.clone(),
                first: None,
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(
        response_text(response)
            .await
            .contains("prompt_bytes_capacity")
    );
    assert_eq!(polls.load(Ordering::Relaxed), 0);
    pr13_counts(&worker.handle(), 1, 64);
    drop(first);
    pr13_counts(&worker.handle(), 0, 0);
    worker.shutdown().await.unwrap();
}

#[derive(Default)]
struct CancellationProbe {
    hub: CompletionHub,
    cancellations: std::sync::atomic::AtomicUsize,
    terminals: std::sync::atomic::AtomicUsize,
    release_first: std::sync::atomic::AtomicBool,
    release_second: std::sync::atomic::AtomicBool,
}

struct CleanupSseEngine {
    probe: std::sync::Arc<CancellationProbe>,
    slow: bool,
    requests: std::collections::HashMap<RequestId, GenerateRequest>,
    cleanup: std::collections::HashMap<RequestId, (bool, ferrule_runtime::RequestCleanupOwner)>,
    pending_cancelled: Vec<SequenceState>,
    cancelled: Vec<SequenceState>,
}

impl InferenceEngine for CleanupSseEngine {
    fn completion_hub(&self) -> CompletionHub {
        self.probe.hub.clone()
    }
    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        Vec::new()
    }
    fn has_pending_async_work(&self) -> bool {
        true
    }
    fn encode(&self, prompt: &str) -> RuntimeResult<Vec<u32>> {
        Ok(prompt.bytes().map(u32::from).collect())
    }
    fn submit(&mut self, request: GenerateRequest) {
        let first = request.prompt_tokens == vec![u32::from(b'a')];
        self.cleanup.insert(request.id, (first, Default::default()));
        self.requests.insert(request.id, request);
    }
    fn request_cleanup(&self, id: RequestId) -> ferrule_runtime::InferenceRequestCleanup {
        ferrule_runtime::InferenceRequestCleanup::Tracked(self.cleanup[&id].1.receipt())
    }
    fn step(
        &mut self,
        emit: &mut dyn FnMut(&ResidentTokenEvent) -> RuntimeResult<()>,
    ) -> RuntimeResult<ResidentDriverStep> {
        use std::sync::atomic::Ordering;
        let released: Vec<_> = self
            .cleanup
            .iter()
            .filter_map(|(id, (first, _))| {
                let released = if *first {
                    self.probe.release_first.load(Ordering::Acquire)
                } else {
                    self.probe.release_second.load(Ordering::Acquire)
                };
                (released && !self.requests.contains_key(id)).then_some(*id)
            })
            .collect();
        let made_progress = !released.is_empty() || !self.pending_cancelled.is_empty();
        for id in released {
            self.cleanup.remove(&id).unwrap().1.release();
        }
        if made_progress {
            self.cancelled.append(&mut self.pending_cancelled);
            return Ok(ResidentDriverStep::Executed {
                action_kind: ferrule_runtime::ResidentActionKind::Cancel,
                rows: 0,
                staged: 0,
                finished: 0,
            });
        }
        if self.slow {
            for request in self.requests.values() {
                if !self.cleanup[&request.id].0 {
                    continue;
                }
                for index in 0..8 {
                    emit(&ResidentTokenEvent {
                        session_id: request.session_id.unwrap(),
                        request_id: Some(request.id),
                        index,
                        token: 1,
                        logit: None,
                        text: "x".into(),
                    })?;
                }
            }
        }
        Ok(ResidentDriverStep::Blocked)
    }
    fn cancel_request(&mut self, id: RequestId) -> RuntimeResult<InferenceCancelProgress> {
        use std::sync::atomic::Ordering;
        self.probe.cancellations.fetch_add(1, Ordering::AcqRel);
        let request = self
            .requests
            .remove(&id)
            .expect("cancel must be submitted once");
        let mut sequence = SequenceState::from_request(&request, request.session_id.unwrap());
        sequence.finish_reason = Some(SequenceFinishReason::Cancelled);
        self.pending_cancelled.push(sequence);
        self.probe.hub.notify();
        Ok(InferenceCancelProgress::Pending)
    }
    fn drain_finished(&mut self) -> Vec<SequenceState> {
        Vec::new()
    }
    fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        use std::sync::atomic::Ordering;
        let cancelled = std::mem::take(&mut self.cancelled);
        self.probe
            .terminals
            .fetch_add(cancelled.len(), Ordering::AcqRel);
        cancelled
    }
    fn drain_failed(&mut self) -> Vec<SequenceState> {
        Vec::new()
    }
    fn shutdown(&mut self) -> RuntimeResult<ferrule_runtime::InferenceShutdownProgress> {
        assert!(self.requests.is_empty());
        assert!(self.cleanup.is_empty());
        Ok(ferrule_runtime::InferenceShutdownProgress::Complete)
    }
}

async fn wait_for_lifecycle(mut ready: impl FnMut() -> bool) {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !ready() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("worker lifecycle did not progress");
}

#[tokio::test]
async fn slow_sse_and_disconnect_keep_cleanup_independent_of_an_unrelated_live_request() {
    use std::sync::atomic::Ordering;
    for slow in [false, true] {
        let probe = std::sync::Arc::new(CancellationProbe::default());
        let owner_probe = probe.clone();
        let worker = spawn_model_worker_with(
            move || {
                Ok::<_, std::convert::Infallible>(CleanupSseEngine {
                    probe: owner_probe,
                    slow,
                    requests: Default::default(),
                    cleanup: Default::default(),
                    pending_cancelled: Vec::new(),
                    cancelled: Vec::new(),
                })
            },
            WorkerConfig {
                event_queue_capacity: 3,
                max_inflight_requests: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let handle = worker.handle();
        let app = router(ServerState::new(
            ModelRegistration::new("test-model", ChatTemplate::Plain),
            handle.clone(),
        ));
        let first = app
            .clone()
            .oneshot(pr13_request(
                "/v1/completions",
                r#"{"model":"test-model","prompt":"a","max_tokens":8,"stream":true}"#,
            ))
            .await
            .unwrap();
        let second = app
            .clone()
            .oneshot(pr13_request(
                "/v1/completions",
                r#"{"model":"test-model","prompt":"b","max_tokens":8,"stream":true}"#,
            ))
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(second.status(), StatusCode::OK);
        if slow {
            // Do not poll SSE until the bounded queue cancels its slow observer.
            wait_for_lifecycle(|| probe.terminals.load(Ordering::Acquire) == 1).await;
            let body =
                tokio::time::timeout(std::time::Duration::from_secs(2), response_text(first))
                    .await
                    .unwrap();
            assert_eq!(body.matches("data: [DONE]").count(), 1);
            assert_eq!(body.matches("\"error\"").count(), 1);
            assert!(body.contains("request_cancelled"));
        } else {
            drop(first);
            wait_for_lifecycle(|| probe.terminals.load(Ordering::Acquire) == 1).await;
        }
        assert_eq!(probe.cancellations.load(Ordering::Acquire), 1);
        pr13_counts(&handle, 2, 0);
        assert_eq!(handle.phase(), ferrule_server::WorkerPhase::Ready);
        probe.release_first.store(true, Ordering::Release);
        probe.hub.notify();
        wait_for_lifecycle(|| handle.admission_snapshot().held_requests == 1).await;
        // The second HTTP observer is still live and has not been cancelled.
        assert_eq!(probe.cancellations.load(Ordering::Acquire), 1);
        drop(second);
        wait_for_lifecycle(|| probe.terminals.load(Ordering::Acquire) == 2).await;
        pr13_counts(&handle, 1, 0);
        probe.release_second.store(true, Ordering::Release);
        probe.hub.notify();
        wait_for_lifecycle(|| handle.admission_snapshot().held_requests == 0).await;
        assert_eq!(probe.cancellations.load(Ordering::Acquire), 2);
        tokio::time::timeout(std::time::Duration::from_secs(2), worker.shutdown())
            .await
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn admission_snapshot_reads_owner_limits_without_server_permits() {
    let worker = spawn_model_worker_with(
        || {
            Ok::<_, std::convert::Infallible>(ImmediateEngine {
                owner: Some(std::thread::current().id()),
                ..Default::default()
            })
        },
        WorkerConfig {
            max_inflight_requests: 3,
            max_prompt_bytes: 99,
            max_body_bytes: 17,
            runtime_admission_options: Some(ferrule_runtime::RuntimeAdmissionOptions {
                max_waiting_requests: 2,
                max_request_identities: 4,
                max_session_identities: 1,
            }),
            ..Default::default()
        },
    )
    .unwrap();
    let handle = worker.handle();
    let app = router(ServerState::new(
        ModelRegistration::new("test-model", ChatTemplate::Plain),
        handle.clone(),
    ));
    for _ in 0..3 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/admission")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let value: serde_json::Value =
            serde_json::from_str(&response_text(response).await).unwrap();
        assert_eq!(value["requests"]["limit"], 3);
        assert_eq!(value["prompt_bytes"]["limit"], 99);
        assert_eq!(value["max_body_bytes"], 17);
        assert_eq!(value["runtime"]["status"], "available");
        for (field, limit) in [
            ("waiting_requests", 2),
            ("request_identities", 4),
            ("session_identities", 1),
        ] {
            assert_eq!(value["runtime"][field]["limit"], limit);
            assert_eq!(value["runtime"][field]["held"], 0);
        }
        assert!(value["scheduler"].is_null());
        assert!(value["kv"].is_null());
        pr13_counts(&handle, 0, 0);
    }
    handle.begin_shutdown();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/admission")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let value: serde_json::Value = serde_json::from_str(&response_text(response).await).unwrap();
    assert_eq!(value["closed"], true);
    assert_eq!(value["runtime"]["status"], "unavailable");
    assert_eq!(value["runtime"]["reason"], "worker_unavailable");
    assert!(value["scheduler"].is_null());
    assert!(value["kv"].is_null());
    worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn unknown_engine_never_reports_fabricated_scheduler_or_kv_gauges() {
    let (state, worker) = test_state();
    let app = router(state);
    for generated in [false, true] {
        if generated {
            let response = app
                .clone()
                .oneshot(pr13_request(
                    "/v1/completions",
                    r#"{"model":"test-model","prompt":"hello","max_tokens":1}"#,
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let _ = response_text(response).await;
        }
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(2),
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
        assert_eq!(response.status(), StatusCode::OK);
        let value: serde_json::Value =
            serde_json::from_str(&response_text(response).await).unwrap();
        assert_eq!(value["runtime"]["status"], "unsupported");
        assert!(value["scheduler"].is_null());
        assert!(value["kv"].is_null());
    }
    worker.shutdown().await.unwrap();
}
