use std::collections::VecDeque;
use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_core::Stream;
use serde_json::json;
use tokio::net::TcpListener;

use crate::config::ModelRegistration;
use crate::openai::{
    AssistantMessage, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    ChunkChoice, ChunkDelta, CompletionChunk, CompletionRequest, CompletionResponse, ErrorEnvelope,
    ErrorObject, ModelList, ModelObject, ResponseChoice, TextCompletionChoice,
    openai_finish_reason,
};
use crate::worker::{
    EventSubscription, ModelWorkerHandle, SubmitError, SubmitErrorKind, WorkerEvent, WorkerRequest,
};

#[derive(Clone)]
pub struct ServerState {
    registration: Arc<ModelRegistration>,
    worker: ModelWorkerHandle,
}

impl ServerState {
    pub fn new(registration: ModelRegistration, worker: ModelWorkerHandle) -> Self {
        Self {
            registration: Arc::new(registration),
            worker,
        }
    }
}

pub fn router(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .with_state(state)
}

pub async fn serve_with_shutdown<F>(
    address: SocketAddr,
    state: ServerState,
    shutdown: F,
) -> std::io::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let listener = TcpListener::bind(address).await?;
    tracing::info!(%address, "Ferrule OpenAI server listening");
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

async fn models(State(state): State<ServerState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelObject {
            id: state.registration.id.clone(),
            object: "model",
            created: state.registration.created,
            owned_by: state.registration.owned_by.clone(),
        }],
    })
}

async fn chat_completions(
    State(state): State<ServerState>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(rejection) => return json_rejection(rejection),
    };
    let validated = match request.validate(&state.registration.id, state.registration.chat_template)
    {
        Ok(validated) => validated,
        Err(message) => {
            return api_error(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
        }
    };
    let stream = validated.stream;
    let include_usage = validated.include_usage;
    let subscription = match state
        .worker
        .submit(WorkerRequest {
            prompt: validated.prompt,
            max_tokens: validated.max_tokens,
            stop: validated.stop,
            ignore_eos: validated.ignore_eos,
        })
        .await
    {
        Ok(subscription) => subscription,
        Err(error) => return submit_error(error),
    };

    let completion_id = format!("chatcmpl-ferrule-{}", subscription.request_id.0);
    let created = unix_timestamp();
    if stream {
        let event_stream = ChatEventStream::new(
            subscription,
            completion_id,
            state.registration.id.clone(),
            created,
            include_usage,
        );
        let mut response = Sse::new(event_stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_secs(15))
                    .text("keep-alive"),
            )
            .into_response();
        response.headers_mut().insert(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache, no-transform"),
        );
        response
            .headers_mut()
            .insert("x-accel-buffering", HeaderValue::from_static("no"));
        response
    } else {
        chat_non_streaming_response(
            subscription,
            completion_id,
            state.registration.id.clone(),
            created,
        )
        .await
    }
}

async fn chat_non_streaming_response(
    mut subscription: EventSubscription,
    completion_id: String,
    model: String,
    created: u64,
) -> Response {
    let mut content = String::new();
    while let Some(event) = subscription.recv().await {
        match event {
            WorkerEvent::Token { text } => content.push_str(&text),
            WorkerEvent::Finished { reason, usage } => {
                return Json(ChatCompletionResponse {
                    id: &completion_id,
                    object: "chat.completion",
                    created,
                    model: &model,
                    choices: vec![ResponseChoice {
                        index: 0,
                        message: AssistantMessage {
                            role: "assistant",
                            content: &content,
                        },
                        finish_reason: openai_finish_reason(reason),
                    }],
                    usage,
                })
                .into_response();
            }
            WorkerEvent::Cancelled => {
                return api_error(
                    StatusCode::REQUEST_TIMEOUT,
                    "generation request was cancelled",
                    "request_cancelled",
                );
            }
            WorkerEvent::Failed { message } => {
                return api_error(StatusCode::INTERNAL_SERVER_ERROR, &message, "server_error");
            }
        }
    }
    api_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "model worker closed the request without a terminal event",
        "server_error",
    )
}

async fn completions(
    State(state): State<ServerState>,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(rejection) => return json_rejection(rejection),
    };
    let validated = match request.validate(&state.registration.id) {
        Ok(validated) => validated,
        Err(message) => {
            return api_error(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
        }
    };
    let stream = validated.stream;
    let include_usage = validated.include_usage;
    let subscription = match state
        .worker
        .submit(WorkerRequest {
            prompt: validated.prompt,
            max_tokens: validated.max_tokens,
            stop: validated.stop,
            ignore_eos: validated.ignore_eos,
        })
        .await
    {
        Ok(subscription) => subscription,
        Err(error) => return submit_error(error),
    };

    let completion_id = format!("cmpl-ferrule-{}", subscription.request_id.0);
    let created = unix_timestamp();
    if stream {
        let event_stream = CompletionEventStream::new(
            subscription,
            completion_id,
            state.registration.id.clone(),
            created,
            include_usage,
        );
        let mut response = Sse::new(event_stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_secs(15))
                    .text("keep-alive"),
            )
            .into_response();
        response.headers_mut().insert(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache, no-transform"),
        );
        response
            .headers_mut()
            .insert("x-accel-buffering", HeaderValue::from_static("no"));
        response
    } else {
        completion_non_streaming_response(
            subscription,
            completion_id,
            state.registration.id.clone(),
            created,
        )
        .await
    }
}

async fn completion_non_streaming_response(
    mut subscription: EventSubscription,
    completion_id: String,
    model: String,
    created: u64,
) -> Response {
    let mut content = String::new();
    while let Some(event) = subscription.recv().await {
        match event {
            WorkerEvent::Token { text } => content.push_str(&text),
            WorkerEvent::Finished { reason, usage } => {
                return Json(CompletionResponse {
                    id: &completion_id,
                    object: "text_completion",
                    created,
                    model: &model,
                    choices: vec![TextCompletionChoice {
                        text: &content,
                        index: 0,
                        logprobs: None,
                        finish_reason: Some(openai_finish_reason(reason)),
                    }],
                    usage,
                })
                .into_response();
            }
            WorkerEvent::Cancelled => {
                return api_error(
                    StatusCode::REQUEST_TIMEOUT,
                    "generation request was cancelled",
                    "request_cancelled",
                );
            }
            WorkerEvent::Failed { message } => {
                return api_error(StatusCode::INTERNAL_SERVER_ERROR, &message, "server_error");
            }
        }
    }
    api_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "model worker closed the request without a terminal event",
        "server_error",
    )
}

struct ChatEventStream {
    subscription: EventSubscription,
    completion_id: String,
    model: String,
    created: u64,
    include_usage: bool,
    pending: VecDeque<Result<Event, Infallible>>,
    done: bool,
}

impl ChatEventStream {
    fn new(
        subscription: EventSubscription,
        completion_id: String,
        model: String,
        created: u64,
        include_usage: bool,
    ) -> Self {
        Self {
            subscription,
            completion_id,
            model,
            created,
            include_usage,
            pending: VecDeque::new(),
            done: false,
        }
    }

    fn push_json<T: serde::Serialize>(&mut self, value: &T) {
        self.pending.push_back(Ok(json_event(value)));
    }

    fn push_done(&mut self) {
        self.pending.push_back(Ok(Event::default().data("[DONE]")));
        self.done = true;
    }

    fn push_error(&mut self, message: &str, kind: &'static str) {
        self.push_json(&ErrorEnvelope {
            error: ErrorObject {
                message,
                kind,
                param: None,
                code: None,
            },
        });
        self.push_done();
    }

    fn queue_event(&mut self, event: WorkerEvent) {
        match event {
            WorkerEvent::Token { text } => {
                let chunk = ChatCompletionChunk {
                    id: &self.completion_id,
                    object: "chat.completion.chunk",
                    created: self.created,
                    model: &self.model,
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            content: Some(&text),
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                };
                let event = json_event(&chunk);
                self.pending.push_back(Ok(event));
            }
            WorkerEvent::Finished { reason, usage } => {
                let finish_reason = openai_finish_reason(reason);
                let chunk = ChatCompletionChunk {
                    id: &self.completion_id,
                    object: "chat.completion.chunk",
                    created: self.created,
                    model: &self.model,
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta::default(),
                        finish_reason: Some(finish_reason),
                    }],
                    usage: None,
                };
                let event = json_event(&chunk);
                self.pending.push_back(Ok(event));
                if self.include_usage {
                    let usage_chunk = ChatCompletionChunk {
                        id: &self.completion_id,
                        object: "chat.completion.chunk",
                        created: self.created,
                        model: &self.model,
                        choices: Vec::new(),
                        usage: Some(usage),
                    };
                    let event = json_event(&usage_chunk);
                    self.pending.push_back(Ok(event));
                }
                self.push_done();
            }
            WorkerEvent::Cancelled => {
                self.push_error("generation request was cancelled", "request_cancelled");
            }
            WorkerEvent::Failed { message } => self.push_error(&message, "server_error"),
        }
    }
}

impl Stream for ChatEventStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(event) = self.pending.pop_front() {
            return Poll::Ready(Some(event));
        }
        if self.done {
            return Poll::Ready(None);
        }
        match self.subscription.poll_recv(context) {
            Poll::Ready(Some(event)) => {
                self.queue_event(event);
                Poll::Ready(self.pending.pop_front())
            }
            Poll::Ready(None) => {
                self.push_error(
                    "model worker closed the stream without a terminal event",
                    "server_error",
                );
                Poll::Ready(self.pending.pop_front())
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

struct CompletionEventStream {
    subscription: EventSubscription,
    completion_id: String,
    model: String,
    created: u64,
    include_usage: bool,
    pending: VecDeque<Result<Event, Infallible>>,
    done: bool,
}

impl CompletionEventStream {
    fn new(
        subscription: EventSubscription,
        completion_id: String,
        model: String,
        created: u64,
        include_usage: bool,
    ) -> Self {
        Self {
            subscription,
            completion_id,
            model,
            created,
            include_usage,
            pending: VecDeque::new(),
            done: false,
        }
    }

    fn push_json<T: serde::Serialize>(&mut self, value: &T) {
        self.pending.push_back(Ok(json_event(value)));
    }

    fn push_done(&mut self) {
        self.pending.push_back(Ok(Event::default().data("[DONE]")));
        self.done = true;
    }

    fn push_error(&mut self, message: &str, kind: &'static str) {
        self.push_json(&ErrorEnvelope {
            error: ErrorObject {
                message,
                kind,
                param: None,
                code: None,
            },
        });
        self.push_done();
    }

    fn queue_event(&mut self, event: WorkerEvent) {
        match event {
            WorkerEvent::Token { text } => {
                let chunk = CompletionChunk {
                    id: &self.completion_id,
                    object: "text_completion",
                    created: self.created,
                    model: &self.model,
                    choices: vec![TextCompletionChoice {
                        text: &text,
                        index: 0,
                        logprobs: None,
                        finish_reason: None,
                    }],
                    usage: None,
                };
                self.pending.push_back(Ok(json_event(&chunk)));
            }
            WorkerEvent::Finished { reason, usage } => {
                let chunk = CompletionChunk {
                    id: &self.completion_id,
                    object: "text_completion",
                    created: self.created,
                    model: &self.model,
                    choices: vec![TextCompletionChoice {
                        text: "",
                        index: 0,
                        logprobs: None,
                        finish_reason: Some(openai_finish_reason(reason)),
                    }],
                    usage: None,
                };
                self.pending.push_back(Ok(json_event(&chunk)));
                if self.include_usage {
                    let usage_chunk = CompletionChunk {
                        id: &self.completion_id,
                        object: "text_completion",
                        created: self.created,
                        model: &self.model,
                        choices: Vec::new(),
                        usage: Some(usage),
                    };
                    self.pending.push_back(Ok(json_event(&usage_chunk)));
                }
                self.push_done();
            }
            WorkerEvent::Cancelled => {
                self.push_error("generation request was cancelled", "request_cancelled");
            }
            WorkerEvent::Failed { message } => self.push_error(&message, "server_error"),
        }
    }
}

impl Stream for CompletionEventStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(event) = self.pending.pop_front() {
            return Poll::Ready(Some(event));
        }
        if self.done {
            return Poll::Ready(None);
        }
        match self.subscription.poll_recv(context) {
            Poll::Ready(Some(event)) => {
                self.queue_event(event);
                Poll::Ready(self.pending.pop_front())
            }
            Poll::Ready(None) => {
                self.push_error(
                    "model worker closed the stream without a terminal event",
                    "server_error",
                );
                Poll::Ready(self.pending.pop_front())
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

fn json_event<T: serde::Serialize>(value: &T) -> Event {
    let data = serde_json::to_string(value).expect("OpenAI SSE response must serialize");
    Event::default().data(data)
}

fn submit_error(error: SubmitError) -> Response {
    let status = match error.kind {
        SubmitErrorKind::Overloaded => StatusCode::TOO_MANY_REQUESTS,
        SubmitErrorKind::Unavailable | SubmitErrorKind::AdmissionTimeout => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        SubmitErrorKind::Rejected => StatusCode::INTERNAL_SERVER_ERROR,
    };
    api_error(status, &error.to_string(), "server_error")
}

fn json_rejection(rejection: JsonRejection) -> Response {
    api_error(
        rejection.status(),
        &rejection.body_text(),
        "invalid_request_error",
    )
}

fn api_error(status: StatusCode, message: &str, kind: &'static str) -> Response {
    (
        status,
        Json(ErrorEnvelope {
            error: ErrorObject {
                message,
                kind,
                param: None,
                code: None,
            },
        }),
    )
        .into_response()
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WorkerConfig;
    use crate::worker::{ModelEngine, spawn_model_worker};
    use ferrule_runtime::{
        CancelRequestResult, GenerateRequest, RequestId, ResidentDriverStep, SequenceFinishReason,
        SequenceState,
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[derive(Default)]
    struct ImmediateEngine {
        request: Option<GenerateRequest>,
        finished: Vec<SequenceState>,
        cancelled: Vec<SequenceState>,
    }

    impl ModelEngine for ImmediateEngine {
        fn encode(&self, prompt: &str) -> Result<Vec<u32>, String> {
            Ok(prompt.bytes().map(u32::from).collect())
        }

        fn submit(&mut self, request: GenerateRequest) {
            self.request = Some(request);
        }

        fn step(
            &mut self,
            on_token: &mut dyn FnMut(&ferrule_runtime::ResidentTokenEvent),
        ) -> Result<ResidentDriverStep, String> {
            let Some(request) = self.request.take() else {
                return Ok(ResidentDriverStep::Idle);
            };
            let session_id = request.session_id.unwrap();
            on_token(&ferrule_runtime::ResidentTokenEvent {
                session_id,
                request_id: Some(request.id),
                index: 0,
                token: 42,
                logit: Some(1.0),
                text: "ok".into(),
                generated_text: "ok".into(),
            });
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

        fn cancel_request(&mut self, request_id: RequestId) -> Result<CancelRequestResult, String> {
            let Some(request) = self.request.take() else {
                return Ok(CancelRequestResult::NotFound { request_id });
            };
            let session_id = request.session_id.unwrap();
            let mut state = SequenceState::from_request(&request, session_id);
            state.finish_reason = Some(SequenceFinishReason::Cancelled);
            self.cancelled.push(state);
            Ok(CancelRequestResult::Waiting {
                request_id,
                session_id,
            })
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

    fn test_state() -> (ServerState, crate::worker::ModelWorker) {
        let worker =
            spawn_model_worker(ImmediateEngine::default(), WorkerConfig::default()).unwrap();
        let state = ServerState::new(
            ModelRegistration::new("test-model", ferrule_model::ChatTemplate::Plain),
            worker.handle(),
        );
        (state, worker)
    }

    #[tokio::test]
    async fn models_is_openai_compatible() {
        let (state, worker) = test_state();
        let response = router(state)
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/models")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["data"][0]["id"], "test-model");
        worker.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn streaming_chat_has_content_finish_usage_and_done_without_role_chunk() {
        let (state, worker) = test_state();
        let response = router(state)
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        serde_json::json!({
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hello"}],
                            "max_completion_tokens": 1,
                            "temperature": 0,
                            "stream": true,
                            "stream_options": {"include_usage": true}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("\"content\":\"ok\""));
        assert!(text.contains("\"finish_reason\":\"length\""));
        assert!(text.contains("\"choices\":[],\"usage\""));
        assert!(text.contains("data: [DONE]"));
        assert!(!text.contains("\"role\":\"assistant\""));
        worker.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn streaming_completion_has_content_finish_usage_and_done() {
        let (state, worker) = test_state();
        let response = router(state)
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        serde_json::json!({
                            "model": "test-model",
                            "prompt": "hello",
                            "max_tokens": 1,
                            "temperature": 0,
                            "top_p": 1,
                            "top_k": 1,
                            "min_p": 0,
                            "repetition_penalty": 1,
                            "stop": "END",
                            "ignore_eos": true,
                            "seed": 7,
                            "stream": true,
                            "stream_options": {"include_usage": true}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("\"object\":\"text_completion\""));
        assert!(
            text.contains("\"text\":\"ok\",\"index\":0,\"logprobs\":null,\"finish_reason\":null")
        );
        assert!(
            text.contains(
                "\"text\":\"\",\"index\":0,\"logprobs\":null,\"finish_reason\":\"length\""
            )
        );
        assert!(text.contains("\"choices\":[],\"usage\""));
        assert!(text.contains("data: [DONE]"));
        worker.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn non_streaming_completion_is_openai_compatible() {
        let (state, worker) = test_state();
        let response = router(state)
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        serde_json::json!({
                            "model": "test-model",
                            "prompt": "hello",
                            "max_tokens": 1,
                            "n": 1,
                            "best_of": 1,
                            "stop": ["A", "B"]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(value["id"].as_str().unwrap().starts_with("cmpl-ferrule-"));
        assert_eq!(value["object"], "text_completion");
        assert_eq!(value["model"], "test-model");
        assert_eq!(value["choices"][0]["text"], "ok");
        assert_eq!(value["choices"][0]["index"], 0);
        assert!(value["choices"][0]["logprobs"].is_null());
        assert_eq!(value["choices"][0]["finish_reason"], "length");
        assert_eq!(value["usage"]["completion_tokens"], 1);
        worker.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn json_rejections_use_openai_error_envelope_for_both_completion_posts() {
        for uri in ["/v1/chat/completions", "/v1/completions"] {
            let (state, worker) = test_state();
            let response = router(state)
                .oneshot(
                    axum::http::Request::builder()
                        .method("POST")
                        .uri(uri)
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(axum::body::Body::from("{"))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = response.into_body().collect().await.unwrap().to_bytes();
            let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(value["error"]["type"], "invalid_request_error");
            assert!(value["error"]["message"].is_string());
            worker.shutdown().await.unwrap();
        }
    }

    #[tokio::test]
    async fn rejects_non_greedy_request_before_admission() {
        let (state, worker) = test_state();
        let response = router(state)
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        serde_json::json!({
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hello"}],
                            "temperature": 0.7
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        worker.shutdown().await.unwrap();
    }
}
