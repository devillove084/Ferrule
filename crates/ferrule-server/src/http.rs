use std::collections::VecDeque;

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use ferrule_common::{SseSerializationError, WorkerRequestError};
use futures_core::Stream;
use serde_json::json;
use tokio::net::TcpListener;

use crate::config::ModelRegistration;
use crate::openai::{
    AssistantMessage, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    ChunkChoice, ChunkDelta, CompletionChunk, CompletionRequest, CompletionResponse, ErrorEnvelope,
    ErrorObject, ModelList, ModelObject, ResponseChoice, TextCompletionChoice, TokenizeData,
    TokenizeRequest, TokenizeResponse, openai_finish_reason,
};
use crate::worker::{EventSubscription, ModelWorkerHandle, WorkerEvent, WorkerRequest};

type SubmitError = WorkerRequestError<ferrule_runtime::Error>;
type SseError = SseSerializationError<serde_json::Error>;

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
        .route("/v1/tokenize", post(tokenize))
        .route("/tokenize", post(tokenize))
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
    let request_started_at = Instant::now();
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(rejection) => return json_rejection(rejection),
    };
    let validated = match request.validate(&state.registration.id, state.registration.chat_template)
    {
        Ok(validated) => validated,
        Err(error) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                &error.to_string(),
                "invalid_request_error",
            );
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

    trace_http_request_admitted(
        subscription.request_id.0,
        "chat.completions",
        stream,
        request_started_at,
    );
    let completion_id = format!("chatcmpl-ferrule-{}", subscription.request_id.0);
    let created = unix_timestamp();
    if stream {
        let event_stream = OpenAiEventStream::new(
            SseEndpoint::Chat,
            subscription,
            completion_id,
            state.registration.id.clone(),
            created,
            include_usage,
            request_started_at,
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
            request_started_at,
        )
        .await
    }
}

async fn tokenize(
    State(state): State<ServerState>,
    payload: Result<Json<TokenizeRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(rejection) => return json_rejection(rejection),
    };
    let prompt = match request.validate(&state.registration.id) {
        Ok(prompt) => prompt,
        Err(error) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                &error.to_string(),
                "invalid_request_error",
            );
        }
    };
    let (request_id, tokens) = match state.worker.tokenize(prompt).await {
        Ok(result) => result,
        Err(error) => return submit_error(error),
    };
    let id = format!("tok-ferrule-{request_id}");
    let count = tokens.len();
    Json(TokenizeResponse {
        id: &id,
        object: "list",
        created: unix_timestamp(),
        model: &state.registration.id,
        data: vec![TokenizeData {
            object: "tokens",
            tokens,
            count,
        }],
    })
    .into_response()
}

async fn chat_non_streaming_response(
    mut subscription: EventSubscription,
    completion_id: String,
    model: String,
    created: u64,
    request_started_at: Instant,
) -> Response {
    let request_id = subscription.request_id.0;
    let mut token_events = 0usize;
    let mut disconnect_guard = NonStreamingDisconnectGuard {
        request_id,
        endpoint: "chat.completions",
        request_started_at,
        token_events: 0,
        terminal_seen: false,
    };
    let mut content = String::new();
    while let Some(event) = subscription.recv().await {
        match event {
            WorkerEvent::Token { text } => {
                token_events = token_events.saturating_add(1);
                disconnect_guard.token_events = token_events;
                content.push_str(&text);
            }
            WorkerEvent::Finished { reason, usage } => {
                disconnect_guard.terminal_seen = true;
                trace_http_request_terminal(
                    request_id,
                    "chat.completions",
                    false,
                    "finished",
                    reason.as_str(),
                    token_events,
                    request_started_at,
                );
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
                            reasoning_content: None,
                        },
                        finish_reason: openai_finish_reason(reason),
                    }],
                    usage,
                })
                .into_response();
            }
            WorkerEvent::Cancelled => {
                disconnect_guard.terminal_seen = true;
                trace_http_request_terminal(
                    request_id,
                    "chat.completions",
                    false,
                    "cancelled",
                    "cancelled",
                    token_events,
                    request_started_at,
                );
                return api_error(
                    StatusCode::REQUEST_TIMEOUT,
                    "generation request was cancelled",
                    "request_cancelled",
                );
            }
            WorkerEvent::Failed { error } => {
                disconnect_guard.terminal_seen = true;
                trace_http_request_terminal(
                    request_id,
                    "chat.completions",
                    false,
                    "failed",
                    "model_execution_failed",
                    token_events,
                    request_started_at,
                );
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &error.to_string(),
                    "server_error",
                );
            }
        }
    }
    disconnect_guard.terminal_seen = true;
    trace_http_request_terminal(
        request_id,
        "chat.completions",
        false,
        "failed",
        "worker_channel_closed",
        token_events,
        request_started_at,
    );
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
    let request_started_at = Instant::now();
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(rejection) => return json_rejection(rejection),
    };
    let validated = match request.validate(&state.registration.id) {
        Ok(validated) => validated,
        Err(error) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                &error.to_string(),
                "invalid_request_error",
            );
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

    trace_http_request_admitted(
        subscription.request_id.0,
        "completions",
        stream,
        request_started_at,
    );
    let completion_id = format!("cmpl-ferrule-{}", subscription.request_id.0);
    let created = unix_timestamp();
    if stream {
        let event_stream = OpenAiEventStream::new(
            SseEndpoint::Completion,
            subscription,
            completion_id,
            state.registration.id.clone(),
            created,
            include_usage,
            request_started_at,
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
            request_started_at,
        )
        .await
    }
}

async fn completion_non_streaming_response(
    mut subscription: EventSubscription,
    completion_id: String,
    model: String,
    created: u64,
    request_started_at: Instant,
) -> Response {
    let request_id = subscription.request_id.0;
    let mut token_events = 0usize;
    let mut disconnect_guard = NonStreamingDisconnectGuard {
        request_id,
        endpoint: "completions",
        request_started_at,
        token_events: 0,
        terminal_seen: false,
    };
    let mut content = String::new();
    while let Some(event) = subscription.recv().await {
        match event {
            WorkerEvent::Token { text } => {
                token_events = token_events.saturating_add(1);
                disconnect_guard.token_events = token_events;
                content.push_str(&text);
            }
            WorkerEvent::Finished { reason, usage } => {
                disconnect_guard.terminal_seen = true;
                trace_http_request_terminal(
                    request_id,
                    "completions",
                    false,
                    "finished",
                    reason.as_str(),
                    token_events,
                    request_started_at,
                );
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
                disconnect_guard.terminal_seen = true;
                trace_http_request_terminal(
                    request_id,
                    "completions",
                    false,
                    "cancelled",
                    "cancelled",
                    token_events,
                    request_started_at,
                );
                return api_error(
                    StatusCode::REQUEST_TIMEOUT,
                    "generation request was cancelled",
                    "request_cancelled",
                );
            }
            WorkerEvent::Failed { error } => {
                disconnect_guard.terminal_seen = true;
                trace_http_request_terminal(
                    request_id,
                    "completions",
                    false,
                    "failed",
                    "model_execution_failed",
                    token_events,
                    request_started_at,
                );
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &error.to_string(),
                    "server_error",
                );
            }
        }
    }
    disconnect_guard.terminal_seen = true;
    trace_http_request_terminal(
        request_id,
        "completions",
        false,
        "failed",
        "worker_channel_closed",
        token_events,
        request_started_at,
    );
    api_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "model worker closed the request without a terminal event",
        "server_error",
    )
}

fn trace_http_request_admitted(
    request_id: u64,
    endpoint: &'static str,
    stream: bool,
    request_started_at: Instant,
) {
    tracing::debug!(
        target: "ferrule_request",
        event = "request_http_admitted",
        request_id,
        session_id = request_id,
        endpoint,
        stream,
        http_admission_us = request_started_at.elapsed().as_micros() as u64,
        "production HTTP request admitted"
    );
}

fn trace_http_request_terminal(
    request_id: u64,
    endpoint: &'static str,
    stream: bool,
    status: &'static str,
    finish_reason: &'static str,
    token_events: usize,
    request_started_at: Instant,
) {
    tracing::debug!(
        target: "ferrule_request",
        event = "request_http_terminal",
        request_id,
        session_id = request_id,
        endpoint,
        stream,
        status,
        finish_reason,
        token_events,
        http_response_enqueue_us = request_started_at.elapsed().as_micros() as u64,
        "production HTTP request reached response terminal state"
    );
}

struct NonStreamingDisconnectGuard {
    request_id: u64,
    endpoint: &'static str,
    request_started_at: Instant,
    token_events: usize,
    terminal_seen: bool,
}

impl Drop for NonStreamingDisconnectGuard {
    fn drop(&mut self) {
        if !self.terminal_seen {
            trace_http_request_terminal(
                self.request_id,
                self.endpoint,
                false,
                "cancelled",
                "client_disconnect",
                self.token_events,
                self.request_started_at,
            );
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum SseEndpoint {
    Chat,
    Completion,
}

impl SseEndpoint {
    const fn name(self) -> &'static str {
        match self {
            Self::Chat => "chat.completions",
            Self::Completion => "completions",
        }
    }
}

struct OpenAiEventStream {
    endpoint: SseEndpoint,
    subscription: EventSubscription,
    completion_id: String,
    model: String,
    created: u64,
    include_usage: bool,
    request_started_at: Instant,
    token_events: usize,
    pending: VecDeque<Result<Event, SseError>>,
    done: bool,
}

impl OpenAiEventStream {
    fn new(
        endpoint: SseEndpoint,
        subscription: EventSubscription,
        completion_id: String,
        model: String,
        created: u64,
        include_usage: bool,
        request_started_at: Instant,
    ) -> Self {
        Self {
            endpoint,
            subscription,
            completion_id,
            model,
            created,
            include_usage,
            request_started_at,
            token_events: 0,
            pending: VecDeque::new(),
            done: false,
        }
    }

    fn push_json<T: serde::Serialize>(&mut self, value: &T) {
        self.pending.push_back(json_event(value));
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

    fn trace_terminal(&self, status: &'static str, finish_reason: &'static str) {
        trace_http_request_terminal(
            self.subscription.request_id.0,
            self.endpoint.name(),
            true,
            status,
            finish_reason,
            self.token_events,
            self.request_started_at,
        );
    }

    fn queue_token(&mut self, text: &str) {
        let event = match self.endpoint {
            SseEndpoint::Chat => json_event(&ChatCompletionChunk {
                id: &self.completion_id,
                object: "chat.completion.chunk",
                created: self.created,
                model: &self.model,
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        content: Some(text),
                        reasoning_content: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
            }),
            SseEndpoint::Completion => json_event(&CompletionChunk {
                id: &self.completion_id,
                object: "text_completion",
                created: self.created,
                model: &self.model,
                choices: vec![TextCompletionChoice {
                    text,
                    index: 0,
                    logprobs: None,
                    finish_reason: None,
                }],
                usage: None,
            }),
        };
        self.pending.push_back(event);
    }

    fn queue_finished(
        &mut self,
        reason: ferrule_runtime::SequenceFinishReason,
        usage: crate::openai::Usage,
    ) {
        let finish_reason = openai_finish_reason(reason);
        let terminal = match self.endpoint {
            SseEndpoint::Chat => json_event(&ChatCompletionChunk {
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
            }),
            SseEndpoint::Completion => json_event(&CompletionChunk {
                id: &self.completion_id,
                object: "text_completion",
                created: self.created,
                model: &self.model,
                choices: vec![TextCompletionChoice {
                    text: "",
                    index: 0,
                    logprobs: None,
                    finish_reason: Some(finish_reason),
                }],
                usage: None,
            }),
        };
        self.pending.push_back(terminal);

        if self.include_usage {
            let usage_event = match self.endpoint {
                SseEndpoint::Chat => json_event(&ChatCompletionChunk {
                    id: &self.completion_id,
                    object: "chat.completion.chunk",
                    created: self.created,
                    model: &self.model,
                    choices: Vec::new(),
                    usage: Some(usage),
                }),
                SseEndpoint::Completion => json_event(&CompletionChunk {
                    id: &self.completion_id,
                    object: "text_completion",
                    created: self.created,
                    model: &self.model,
                    choices: Vec::new(),
                    usage: Some(usage),
                }),
            };
            self.pending.push_back(usage_event);
        }
        self.push_done();
    }

    fn queue_event(&mut self, event: WorkerEvent) {
        match event {
            WorkerEvent::Token { text } => {
                self.token_events = self.token_events.saturating_add(1);
                self.queue_token(&text);
            }
            WorkerEvent::Finished { reason, usage } => {
                self.trace_terminal("finished", reason.as_str());
                self.queue_finished(reason, usage);
            }
            WorkerEvent::Cancelled => {
                self.trace_terminal("cancelled", "cancelled");
                self.push_error("generation request was cancelled", "request_cancelled");
            }
            WorkerEvent::Failed { error } => {
                self.trace_terminal("failed", "model_execution_failed");
                self.push_error(&error.to_string(), "server_error");
            }
        }
    }
}

impl Drop for OpenAiEventStream {
    fn drop(&mut self) {
        if !self.done {
            self.trace_terminal("cancelled", "client_disconnect");
        }
    }
}

impl Stream for OpenAiEventStream {
    type Item = Result<Event, SseError>;

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
                self.trace_terminal("failed", "worker_channel_closed");
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

fn json_event<T: serde::Serialize>(value: &T) -> Result<Event, SseError> {
    serde_json::to_string(value)
        .map(|data| Event::default().data(data))
        .map_err(|source| SseSerializationError { source })
}

fn submit_error(error: SubmitError) -> Response {
    let status = if error.is_overloaded() {
        StatusCode::TOO_MANY_REQUESTS
    } else if error.is_unavailable() {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
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
