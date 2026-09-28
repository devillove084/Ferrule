use std::collections::VecDeque;

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::rejection::JsonRejection;
use axum::extract::{Request, State};
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
use crate::worker::{
    EventSubscription, ModelWorkerHandle, ServerAdmissionError, ServerAdmissionPermit,
    ServerAdmissionResource, SuspendedCancellation, WorkerEvent, WorkerRequest,
};

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
        .route("/health/live", get(health))
        .route("/readyz", get(ready))
        .route("/admission", get(admission))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/tokenize", post(tokenize))
        .route("/tokenize", post(tokenize))
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

async fn ready(State(state): State<ServerState>) -> Response {
    let snapshot = state.worker.snapshot();
    let ready = snapshot.phase == crate::worker::WorkerPhase::Ready;
    (
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(json!({"ready": ready, "phase": format!("{:?}", snapshot.phase)})),
    )
        .into_response()
}

async fn admission(State(state): State<ServerState>) -> Json<serde_json::Value> {
    // These are independent owners, not an atomic combined ledger. All runtime
    // counts come from one owner-thread read, never a copied admission ledger.
    let observed = state.worker.runtime_serving_snapshot().await;
    let capacity = observed.as_ref().ok().map(|snapshot| &snapshot.capacity);
    let runtime = match observed.as_ref().map(|snapshot| snapshot.admission) {
        Ok(Some(snapshot)) => json!({
            "status": "available",
            "closed": snapshot.closed,
            "waiting_requests": {
                "limit": snapshot.limits.max_waiting_requests,
                "held": snapshot.waiting_requests,
            },
            "request_identities": {
                "limit": snapshot.limits.max_request_identities,
                "held": snapshot.request_identities_held,
            },
            "session_identities": {
                "limit": snapshot.limits.max_session_identities,
                "held": snapshot.session_identities_held,
            },
        }),
        Ok(None) => json!({"status": "unsupported"}),
        Err(error) => json!({
            "status": "unavailable",
            "reason": public_request_error(error).code,
        }),
    };
    let snapshot = state.worker.admission_snapshot();
    Json(json!({
        "runtime": runtime,
        "scheduler": capacity.and_then(|snapshot| snapshot.get("scheduler")),
        "kv": capacity.and_then(|snapshot| snapshot.get("kv")),
        "max_body_bytes": state.worker.max_body_bytes(),
        "closed": snapshot.closed,
        "requests": {
            "limit": snapshot.request_limit,
            "held": snapshot.held_requests,
            "available": snapshot.available_requests,
        },
        "prompt_bytes": {
            "limit": snapshot.prompt_bytes_limit,
            "held": snapshot.held_prompt_bytes,
            "available": snapshot.available_prompt_bytes,
        },
    }))
}

/// Reserve request and ingress-byte capacity before polling the body. The byte reservation covers ingress, then moves to the
/// formatted UTF-8 prompt; it does not claim to budget tokenizer output or GPU memory.
async fn admitted_json<T: serde::de::DeserializeOwned>(
    worker: &ModelWorkerHandle,
    request: Request,
) -> Result<(T, ServerAdmissionPermit), Response> {
    let permit = worker
        .try_acquire_admission(0)
        .map_err(|error| public_admission_error(&error).response())?;
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim);
    if !content_type.is_some_and(|mime| {
        mime.eq_ignore_ascii_case("application/json")
            || (mime.starts_with("application/") && mime.ends_with("+json"))
    }) {
        return Err(PublicError::new(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "JSON content type is required",
            "invalid_request_error",
            "invalid_content_type",
            Some("body"),
        )
        .response());
    }
    let limit = worker.max_body_bytes();
    let content_length = request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    if content_length.is_some_and(|length| length > limit as u64) {
        return Err(body_too_large().response());
    }
    // Chunked bodies do not announce their size: reserve their bounded ingress
    // window before polling, rather than reading uncharged data first.
    let reserved = content_length
        .map(|length| length as usize)
        .unwrap_or_else(|| limit.min(worker.admission_snapshot().available_prompt_bytes));
    if reserved == 0 && content_length != Some(0) {
        let snapshot = worker.admission_snapshot();
        return Err(public_admission_error(&ServerAdmissionError::Capacity {
            resource: ServerAdmissionResource::PromptBytes,
            limit: snapshot.prompt_bytes_limit,
            held: snapshot.held_prompt_bytes,
        })
        .response());
    }
    permit
        .reconcile_prompt_bytes(reserved)
        .map_err(|error| public_admission_error(&error).response())?;
    let mut stream = request.into_body().into_data_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = std::future::poll_fn(|cx| Pin::new(&mut stream).poll_next(cx)).await {
        let chunk = chunk.map_err(|source| {
            tracing::debug!(%source, "HTTP request body failed");
            PublicError::new(
                StatusCode::BAD_REQUEST,
                "request body could not be read",
                "invalid_request_error",
                "invalid_body",
                Some("body"),
            )
            .response()
        })?;
        let length = bytes
            .len()
            .checked_add(chunk.len())
            .filter(|length| *length <= limit)
            .ok_or_else(|| body_too_large().response())?;
        permit
            .reconcile_prompt_bytes(reserved.max(length))
            .map_err(|error| public_admission_error(&error).response())?;
        bytes.extend_from_slice(&chunk);
    }
    let wire_bytes = bytes.len();
    // Parsing owns decoded strings while the wire buffer is still live. Charge
    // both byte representations before making that copy, not after allocation.
    let parsing_bytes = wire_bytes
        .checked_mul(2)
        .ok_or_else(|| body_too_large().response())?;
    permit
        .reconcile_prompt_bytes(parsing_bytes)
        .map_err(|error| public_admission_error(&error).response())?;
    let Json(value) = Json::<T>::from_bytes(&bytes).map_err(json_rejection)?;
    drop(bytes);
    permit
        .reconcile_prompt_bytes(wire_bytes)
        .expect("shrinking cannot fail");
    // The parsed payload retains the conservative wire-byte reservation until
    // validation consumes it and the exact formatted prompt length is known.
    Ok((value, permit))
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

async fn chat_completions(State(state): State<ServerState>, request: Request) -> Response {
    if state.worker.phase() != crate::worker::WorkerPhase::Ready {
        return submit_error(WorkerRequestError::Unavailable {
            operation: ferrule_common::WorkerOperation::Admission,
        });
    }

    let request_started_at = Instant::now();
    let (request, permit) =
        match admitted_json::<ChatCompletionRequest>(&state.worker, request).await {
            Ok(admitted) => admitted,
            Err(response) => return response,
        };
    let validated = match request.validate(&state.registration.id, state.registration.chat_template)
    {
        Ok(validated) => validated,
        Err(error) => {
            tracing::debug!(%error, "HTTP request validation rejected");
            return public_validation_error(&error).response();
        }
    };
    if let Err(error) = permit.reconcile_prompt_bytes(validated.prompt.len()) {
        return public_admission_error(&error).response();
    }
    let stream = validated.stream;
    let include_usage = validated.include_usage;
    let subscription = match state
        .worker
        .submit_with_permit(
            WorkerRequest {
                prompt: validated.prompt,
                max_tokens: validated.max_tokens,
                stop: validated.stop,
                ignore_eos: validated.ignore_eos,
            },
            permit,
        )
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

async fn tokenize(State(state): State<ServerState>, request: Request) -> Response {
    if state.worker.phase() != crate::worker::WorkerPhase::Ready {
        return submit_error(WorkerRequestError::Unavailable {
            operation: ferrule_common::WorkerOperation::Tokenization,
        });
    }

    let (request, permit) = match admitted_json::<TokenizeRequest>(&state.worker, request).await {
        Ok(admitted) => admitted,
        Err(response) => return response,
    };
    let prompt = match request.validate(&state.registration.id) {
        Ok(prompt) => prompt,
        Err(error) => {
            tracing::debug!(%error, "HTTP request validation rejected");
            return public_validation_error(&error).response();
        }
    };
    if let Err(error) = permit.reconcile_prompt_bytes(prompt.len()) {
        return public_admission_error(&error).response();
    }
    let (request_id, tokens) = match state.worker.tokenize_with_permit(prompt, permit).await {
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
                tracing::error!(%error, "generation failed");
                return public_execution_error(&error).response();
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

async fn completions(State(state): State<ServerState>, request: Request) -> Response {
    if state.worker.phase() != crate::worker::WorkerPhase::Ready {
        return submit_error(WorkerRequestError::Unavailable {
            operation: ferrule_common::WorkerOperation::Admission,
        });
    }

    let request_started_at = Instant::now();
    let (request, permit) = match admitted_json::<CompletionRequest>(&state.worker, request).await {
        Ok(admitted) => admitted,
        Err(response) => return response,
    };
    let validated = match request.validate(&state.registration.id) {
        Ok(validated) => validated,
        Err(error) => {
            tracing::debug!(%error, "HTTP request validation rejected");
            return public_validation_error(&error).response();
        }
    };
    if let Err(error) = permit.reconcile_prompt_bytes(validated.prompt.len()) {
        return public_admission_error(&error).response();
    }
    let stream = validated.stream;
    let include_usage = validated.include_usage;
    let subscription = match state
        .worker
        .submit_with_permit(
            WorkerRequest {
                prompt: validated.prompt,
                max_tokens: validated.max_tokens,
                stop: validated.stop,
                ignore_eos: validated.ignore_eos,
            },
            permit,
        )
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
                tracing::error!(%error, "generation failed");
                return public_execution_error(&error).response();
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

    fn push_error(&mut self, message: &'static str, kind: &'static str) {
        self.push_public_error(PublicError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            message,
            kind,
            if kind == "request_cancelled" {
                "request_cancelled"
            } else {
                "worker_unavailable"
            },
            None,
        ));
    }

    fn push_public_error(&mut self, error: PublicError) {
        self.push_json(&error.envelope());
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
                tracing::error!(%error, "streaming generation failed");
                self.push_public_error(public_execution_error(&error));
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

#[derive(Clone, Copy)]
struct PublicError {
    status: StatusCode,
    message: &'static str,
    kind: &'static str,
    code: &'static str,
    param: Option<&'static str>,
}

impl PublicError {
    const fn new(
        status: StatusCode,
        message: &'static str,
        kind: &'static str,
        code: &'static str,
        param: Option<&'static str>,
    ) -> Self {
        Self {
            status,
            message,
            kind,
            code,
            param,
        }
    }

    fn envelope(self) -> ErrorEnvelope<'static> {
        ErrorEnvelope {
            error: ErrorObject {
                message: self.message,
                kind: self.kind,
                code: Some(self.code),
                param: self.param,
            },
        }
    }

    fn response(self) -> Response {
        (self.status, Json(self.envelope())).into_response()
    }
}

fn internal_error() -> PublicError {
    PublicError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "model execution failed",
        "server_error",
        "execution_failed",
        None,
    )
}

fn invalid_request(param: Option<&'static str>) -> PublicError {
    PublicError::new(
        StatusCode::BAD_REQUEST,
        "request parameters are invalid",
        "invalid_request_error",
        "invalid_request",
        param,
    )
}

fn unavailable() -> PublicError {
    PublicError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "model worker is unavailable",
        "server_error",
        "worker_unavailable",
        None,
    )
}

fn body_too_large() -> PublicError {
    PublicError::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        "request body exceeds the configured limit",
        "invalid_request_error",
        "body_too_large",
        Some("body"),
    )
}

fn public_admission_error(error: &ServerAdmissionError) -> PublicError {
    match error {
        ServerAdmissionError::Closed => unavailable(),
        ServerAdmissionError::Capacity { resource, .. } => PublicError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "server admission capacity is exhausted",
            "server_error",
            match resource {
                ServerAdmissionResource::Requests => "request_capacity",
                ServerAdmissionResource::PromptBytes => "prompt_bytes_capacity",
            },
            match resource {
                ServerAdmissionResource::Requests => None,
                ServerAdmissionResource::PromptBytes => Some("prompt"),
            },
        ),
    }
}

fn public_runtime_error(error: &ferrule_runtime::Error) -> PublicError {
    use ferrule_runtime::{Error, RuntimeAdmissionError as Admission};
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(cause) = source {
        if let Some(error) = cause.downcast_ref::<ServerAdmissionError>() {
            return public_admission_error(error);
        }
        if cause.is::<SuspendedCancellation>() {
            return PublicError::new(
                StatusCode::CONFLICT,
                "request requires session restore or worker shutdown",
                "invalid_request_error",
                "requires_restore_or_shutdown",
                Some("session_id"),
            );
        }
        source = cause.source();
    }
    match error {
        Error::Admission { source } => match source {
            Admission::Capacity { .. } => PublicError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "runtime admission capacity is exhausted",
                "server_error",
                "runtime_capacity",
                None,
            ),
            Admission::Closed | Admission::SessionIdentityExhausted => unavailable(),
            Admission::DuplicateRequest { .. } => PublicError::new(
                StatusCode::CONFLICT,
                "request identity is still in use",
                "invalid_request_error",
                "duplicate_request",
                Some("request_id"),
            ),
            Admission::SessionBusy { .. } => PublicError::new(
                StatusCode::CONFLICT,
                "session is still in use",
                "invalid_request_error",
                "session_busy",
                Some("session_id"),
            ),
            Admission::InvalidPosition { .. } | Admission::RetainedPositionMismatch { .. } => {
                invalid_request(Some("position"))
            }
            Admission::InvalidOptions | Admission::OptionsInUse => invalid_request(None),
        },
        Error::InvalidRequest { .. } => invalid_request(None),
        Error::RequestCapacity { .. } => PublicError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "runtime admission capacity is exhausted",
            "server_error",
            "runtime_capacity",
            None,
        ),
        Error::EngineUnavailable => unavailable(),
        _ => internal_error(),
    }
}

fn tokenization_error() -> PublicError {
    PublicError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "prompt tokenization failed",
        "server_error",
        "tokenization_failed",
        Some("prompt"),
    )
}

fn public_execution_error(
    error: &ferrule_common::WorkerExecutionError<ferrule_runtime::Error>,
) -> PublicError {
    use ferrule_common::WorkerExecutionError as Execution;
    match error {
        Execution::Runtime { source }
        | Execution::Cancellation { source }
        | Execution::ShutdownCancellation { source } => public_runtime_error(source),
        Execution::AdmissionEmptyPromptTokens => invalid_request(Some("prompt")),
        Execution::AdmissionTokenization { .. } => tokenization_error(),
        _ => internal_error(),
    }
}

fn submit_error(error: SubmitError) -> Response {
    tracing::debug!(%error, "worker request rejected");
    public_request_error(&error).response()
}

fn public_request_error(error: &SubmitError) -> PublicError {
    match error {
        WorkerRequestError::Rejected { source } => public_execution_error(source),
        WorkerRequestError::QueueFull => PublicError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "model request queue is full",
            "server_error",
            "command_queue_full",
            None,
        ),
        WorkerRequestError::Unavailable { .. } => unavailable(),
        WorkerRequestError::AdmissionTimeout => PublicError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "timed out waiting for model admission",
            "server_error",
            "admission_timeout",
            None,
        ),
        WorkerRequestError::EmptyPromptTokens => invalid_request(Some("prompt")),
        WorkerRequestError::Tokenization { .. } => tokenization_error(),
    }
}

fn public_validation_error(
    error: &ferrule_common::ServingRequestError<ferrule_model::ChatFormatError>,
) -> PublicError {
    use ferrule_common::ServingRequestError as Validation;
    match error {
        Validation::ToolCallingUnsupported => PublicError::new(
            StatusCode::BAD_REQUEST,
            "tool calling is not supported",
            "invalid_request_error",
            "unsupported_parameter",
            Some("tools"),
        ),
        Validation::ModelNotServed { .. } => PublicError::new(
            StatusCode::BAD_REQUEST,
            "requested model is not served",
            "invalid_request_error",
            "model_not_served",
            Some("model"),
        ),
        Validation::ChatFormat { .. } => invalid_request(Some("messages")),
        _ => invalid_request(None),
    }
}

fn json_rejection(rejection: JsonRejection) -> Response {
    tracing::debug!(%rejection, "invalid JSON request");
    PublicError::new(
        StatusCode::BAD_REQUEST,
        "request body is not valid JSON for this endpoint",
        "invalid_request_error",
        "invalid_json",
        Some("body"),
    )
    .response()
}

fn api_error(status: StatusCode, message: &'static str, kind: &'static str) -> Response {
    PublicError::new(
        status,
        message,
        kind,
        if kind == "request_cancelled" {
            "request_cancelled"
        } else {
            "worker_unavailable"
        },
        None,
    )
    .response()
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod pr13_tests {
    use super::*;
    use ferrule_runtime::{Error, RuntimeAdmissionError, RuntimeAdmissionResource};

    #[test]
    fn pr13_runtime_admission_status_baseline() {
        for (source, expected) in [
            (
                RuntimeAdmissionError::Capacity {
                    resource: RuntimeAdmissionResource::WaitingRequests,
                    limit: 1,
                    held: 1,
                },
                StatusCode::TOO_MANY_REQUESTS,
            ),
            (
                RuntimeAdmissionError::Closed,
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                RuntimeAdmissionError::DuplicateRequest {
                    request_id: ferrule_runtime::RequestId(1),
                },
                StatusCode::CONFLICT,
            ),
        ] {
            let response = submit_error(WorkerRequestError::Rejected {
                source: Arc::new(ferrule_common::WorkerExecutionError::Runtime {
                    source: Error::Admission { source },
                }),
            });
            assert_eq!(response.status(), expected);
        }
    }

    #[tokio::test]
    async fn pr13_public_error_redacts_source_baseline() {
        let response = submit_error(WorkerRequestError::Rejected {
            source: Arc::new(ferrule_common::WorkerExecutionError::Runtime {
                source: Error::InvalidRequest {
                    message: "/private/model-secret/provider-detail".into(),
                },
            }),
        });
        let body = axum::body::to_bytes(response.into_body(), 8192)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(!body.contains("model-secret"), "{body}");
        assert!(body.contains("\"code\":\"invalid_request\""), "{body}");
    }
    #[tokio::test]
    async fn pr13_all_public_error_variants_and_params() {
        use ferrule_common::{WorkerExecutionError as Execution, WorkerOperation};
        use ferrule_runtime::{RequestId, SessionId};
        let matrix = [
            (
                RuntimeAdmissionError::Capacity {
                    resource: RuntimeAdmissionResource::RequestIdentities,
                    limit: 2,
                    held: 2,
                },
                429,
                "runtime_capacity",
                None,
            ),
            (
                RuntimeAdmissionError::Closed,
                503,
                "worker_unavailable",
                None,
            ),
            (
                RuntimeAdmissionError::DuplicateRequest {
                    request_id: RequestId(1),
                },
                409,
                "duplicate_request",
                Some("request_id"),
            ),
            (
                RuntimeAdmissionError::SessionBusy {
                    session_id: SessionId(1),
                },
                409,
                "session_busy",
                Some("session_id"),
            ),
            (
                RuntimeAdmissionError::InvalidPosition {
                    position: 4,
                    prompt_tokens: 1,
                    context: 2,
                },
                400,
                "invalid_request",
                Some("position"),
            ),
            (
                RuntimeAdmissionError::RetainedPositionMismatch {
                    supplied: 0,
                    expected: 4,
                },
                400,
                "invalid_request",
                Some("position"),
            ),
            (
                RuntimeAdmissionError::InvalidOptions,
                400,
                "invalid_request",
                None,
            ),
            (
                RuntimeAdmissionError::OptionsInUse,
                400,
                "invalid_request",
                None,
            ),
            (
                RuntimeAdmissionError::SessionIdentityExhausted,
                503,
                "worker_unavailable",
                None,
            ),
        ];
        for (source, status, code, param) in matrix {
            let classified = public_runtime_error(&Error::Admission { source });
            assert_eq!(classified.status.as_u16(), status);
            let response = classified.response();
            let bytes = axum::body::to_bytes(response.into_body(), 8192)
                .await
                .unwrap();
            let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(json["error"]["code"], code);
            assert_eq!(json["error"]["param"], serde_json::json!(param));
            assert_eq!(
                json["error"]["type"],
                if status == 400 || status == 409 {
                    "invalid_request_error"
                } else {
                    "server_error"
                }
            );
        }
        for (error, status, code) in [
            (WorkerRequestError::QueueFull, 429, "command_queue_full"),
            (
                WorkerRequestError::Unavailable {
                    operation: WorkerOperation::Admission,
                },
                503,
                "worker_unavailable",
            ),
            (
                WorkerRequestError::AdmissionTimeout,
                503,
                "admission_timeout",
            ),
            (
                WorkerRequestError::EmptyPromptTokens,
                400,
                "invalid_request",
            ),
            (
                WorkerRequestError::Tokenization {
                    source: Error::InvalidRequest {
                        message: "/private/provider-secret".into(),
                    },
                },
                500,
                "tokenization_failed",
            ),
            (
                WorkerRequestError::Rejected {
                    source: Arc::new(Execution::AdmissionTokenization {
                        source: Error::InvalidRequest {
                            message: "/private/provider-secret".into(),
                        },
                    }),
                },
                500,
                "tokenization_failed",
            ),
            (
                WorkerRequestError::Rejected {
                    source: Arc::new(Execution::ModelExecution),
                },
                500,
                "execution_failed",
            ),
        ] {
            let response = submit_error(error);
            assert_eq!(response.status().as_u16(), status);
            let bytes = axum::body::to_bytes(response.into_body(), 8192)
                .await
                .unwrap();
            let text = String::from_utf8(bytes.to_vec()).unwrap();
            assert!(text.contains(code), "{text}");
            assert!(!text.contains("provider-secret"));
        }
        let source = ferrule_common::Error::Backend {
            source: Box::new(SuspendedCancellation {
                request_id: RequestId(1),
                session_id: SessionId(1),
            }),
        }
        .into();
        let error = public_execution_error(&Execution::Cancellation { source });
        assert_eq!(error.status, StatusCode::CONFLICT);
        assert_eq!(error.code, "requires_restore_or_shutdown");
        for resource in [
            ServerAdmissionResource::Requests,
            ServerAdmissionResource::PromptBytes,
        ] {
            let source = ferrule_common::Error::Backend {
                source: Box::new(ServerAdmissionError::Capacity {
                    resource,
                    held: 1,
                    limit: 1,
                }),
            }
            .into();
            assert_eq!(
                public_runtime_error(&source).status,
                StatusCode::TOO_MANY_REQUESTS
            );
        }
    }
}
