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
}

impl InferenceEngine for ImmediateEngine {
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
