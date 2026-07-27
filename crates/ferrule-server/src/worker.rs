use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Instant;

use ferrule_runtime::{
    GenerateRequest, InferenceCancelProgress, InferenceCompletionOwner, InferenceEngine, RequestId,
    ResidentDriverStep, ResidentTokenEvent, SequenceFinishReason, SessionId,
};
use tokio::sync::{mpsc, oneshot};

use crate::config::WorkerConfig;
use crate::openai::Usage;

#[derive(Debug)]
pub(crate) struct WorkerRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub ignore_eos: bool,
}

#[derive(Debug)]
pub(crate) enum WorkerEvent {
    Token {
        text: String,
    },
    Finished {
        reason: SequenceFinishReason,
        usage: Usage,
    },
    Cancelled,
    Failed {
        message: String,
    },
}

struct SubmitCommand {
    request_id: RequestId,
    enqueued_at: Instant,
    request: WorkerRequest,
    events: mpsc::Sender<WorkerEvent>,
    cancellation: Arc<AtomicBool>,
    accepted: oneshot::Sender<Result<(), String>>,
}

struct TokenizeCommand {
    prompt: String,
    response: oneshot::Sender<Result<Vec<u32>, String>>,
}

enum WorkerCommand {
    Submit(SubmitCommand),
    Tokenize(TokenizeCommand),
    Shutdown,
}

struct ActiveRequest {
    events: mpsc::Sender<WorkerEvent>,
    cancellation: Arc<AtomicBool>,
    session_id: SessionId,
    submitted_at: Instant,
    emitted_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitErrorKind {
    Overloaded,
    Unavailable,
    AdmissionTimeout,
    Rejected,
}

#[derive(Debug)]
pub struct SubmitError {
    pub kind: SubmitErrorKind,
    message: String,
}

impl SubmitError {
    fn new(kind: SubmitErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for SubmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for SubmitError {}

#[derive(Clone)]
pub struct ModelWorkerHandle {
    commands: mpsc::Sender<WorkerCommand>,
    next_request_id: Arc<AtomicU64>,
    config: WorkerConfig,
}

impl ModelWorkerHandle {
    pub(crate) async fn submit(
        &self,
        request: WorkerRequest,
    ) -> Result<EventSubscription, SubmitError> {
        let request_id = RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed));
        let enqueued_at = Instant::now();
        let (events, receiver) = mpsc::channel(self.config.event_queue_capacity);
        let (accepted, acceptance) = oneshot::channel();
        let cancellation = Arc::new(AtomicBool::new(false));
        let command = WorkerCommand::Submit(SubmitCommand {
            request_id,
            enqueued_at,
            request,
            events,
            cancellation: Arc::clone(&cancellation),
            accepted,
        });

        match self.commands.try_send(command) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                return Err(SubmitError::new(
                    SubmitErrorKind::Overloaded,
                    "model request queue is full",
                ));
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(SubmitError::new(
                    SubmitErrorKind::Unavailable,
                    "model worker is unavailable",
                ));
            }
        }

        let admitted = tokio::time::timeout(self.config.admission_timeout, acceptance).await;
        match admitted {
            Ok(Ok(Ok(()))) => Ok(EventSubscription {
                request_id,
                receiver,
                cancellation,
                terminal_seen: false,
            }),
            Ok(Ok(Err(message))) => Err(SubmitError::new(SubmitErrorKind::Rejected, message)),
            Ok(Err(_)) => Err(SubmitError::new(
                SubmitErrorKind::Unavailable,
                "model worker stopped during admission",
            )),
            Err(_) => {
                cancellation.store(true, Ordering::Release);
                Err(SubmitError::new(
                    SubmitErrorKind::AdmissionTimeout,
                    "timed out waiting for model admission",
                ))
            }
        }
    }

    /// Tokenize a prompt on the model worker thread.
    ///
    /// Returns the allocated request id alongside the token ids so the caller
    /// can build a unique response identifier consistent with [`submit`].
    pub(crate) async fn tokenize(&self, prompt: String) -> Result<(u64, Vec<u32>), SubmitError> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (response, receiver) = oneshot::channel();
        let command = WorkerCommand::Tokenize(TokenizeCommand { prompt, response });

        match self.commands.try_send(command) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                return Err(SubmitError::new(
                    SubmitErrorKind::Overloaded,
                    "model request queue is full",
                ));
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(SubmitError::new(
                    SubmitErrorKind::Unavailable,
                    "model worker is unavailable",
                ));
            }
        }

        match receiver.await {
            Ok(Ok(tokens)) => Ok((request_id, tokens)),
            Ok(Err(message)) => Err(SubmitError::new(SubmitErrorKind::Rejected, message)),
            Err(_) => Err(SubmitError::new(
                SubmitErrorKind::Unavailable,
                "model worker stopped during tokenization",
            )),
        }
    }
}

pub(crate) struct EventSubscription {
    pub request_id: RequestId,
    receiver: mpsc::Receiver<WorkerEvent>,
    cancellation: Arc<AtomicBool>,
    terminal_seen: bool,
}

impl EventSubscription {
    pub(crate) async fn recv(&mut self) -> Option<WorkerEvent> {
        let event = self.receiver.recv().await;
        if matches!(
            event,
            Some(
                WorkerEvent::Finished { .. } | WorkerEvent::Cancelled | WorkerEvent::Failed { .. }
            )
        ) {
            self.terminal_seen = true;
        }
        event
    }

    pub(crate) fn poll_recv(
        &mut self,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<WorkerEvent>> {
        let event = self.receiver.poll_recv(context);
        if matches!(
            event,
            std::task::Poll::Ready(Some(
                WorkerEvent::Finished { .. } | WorkerEvent::Cancelled | WorkerEvent::Failed { .. }
            ))
        ) {
            self.terminal_seen = true;
        }
        event
    }
}

impl Drop for EventSubscription {
    fn drop(&mut self) {
        if !self.terminal_seen {
            self.cancellation.store(true, Ordering::Release);
        }
    }
}

pub struct ModelWorker {
    handle: ModelWorkerHandle,
    thread: Option<JoinHandle<()>>,
}

impl ModelWorker {
    pub fn handle(&self) -> ModelWorkerHandle {
        self.handle.clone()
    }

    pub async fn shutdown(mut self) -> Result<(), String> {
        let _ = self.handle.commands.send(WorkerCommand::Shutdown).await;
        let Some(thread) = self.thread.take() else {
            return Ok(());
        };
        tokio::task::spawn_blocking(move || thread.join())
            .await
            .map_err(|error| format!("failed to join model worker task: {error}"))?
            .map_err(|_| "model worker thread panicked".to_string())
    }
}

impl Drop for ModelWorker {
    fn drop(&mut self) {
        if self.thread.is_some() {
            let _ = self.handle.commands.try_send(WorkerCommand::Shutdown);
        }
    }
}

pub fn spawn_model_worker<E>(engine: E, config: WorkerConfig) -> Result<ModelWorker, String>
where
    E: InferenceEngine,
{
    spawn_model_worker_with(move || Ok(engine), config)
}

/// Construct and run the model engine on the same dedicated owner thread.
///
/// Production CUDA bootstraps should prefer this entry point so context creation,
/// prepared resources, the resident driver, and every execution step remain on
/// one OS thread for the worker's entire lifetime.
pub fn spawn_model_worker_with<F, E>(
    factory: F,
    config: WorkerConfig,
) -> Result<ModelWorker, String>
where
    F: FnOnce() -> Result<E, String> + Send + 'static,
    E: InferenceEngine,
{
    config.validate().map_err(str::to_string)?;
    let (commands, receiver) = mpsc::channel(config.command_queue_capacity);
    let (ready_sender, ready_receiver) = std::sync::mpsc::sync_channel(1);
    let thread_config = config.clone();
    let thread = std::thread::Builder::new()
        .name("ferrule-model-worker".into())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = ready_sender.send(Err(format!(
                        "failed to build model-owner async runtime: {error}"
                    )));
                    return;
                }
            };
            let local = tokio::task::LocalSet::new();
            local.block_on(&runtime, async move {
                let mut engine = match factory() {
                    Ok(engine) => engine,
                    Err(error) => {
                        let _ = ready_sender.send(Err(error));
                        return;
                    }
                };
                let completion_owner = InferenceCompletionOwner::attach(&mut engine);
                let _ = ready_sender.send(Ok(()));
                run_worker(engine, completion_owner, receiver, thread_config).await;
            });
        })
        .map_err(|error| format!("failed to spawn model worker: {error}"))?;

    match ready_receiver.recv() {
        Ok(Ok(())) => Ok(ModelWorker {
            handle: ModelWorkerHandle {
                commands,
                next_request_id: Arc::new(AtomicU64::new(1)),
                config,
            },
            thread: Some(thread),
        }),
        Ok(Err(error)) => {
            let _ = thread.join();
            Err(error)
        }
        Err(error) => {
            let _ = thread.join();
            Err(format!(
                "model worker stopped during initialization: {error}"
            ))
        }
    }
}

async fn run_worker<E>(
    mut engine: E,
    mut completion_owner: InferenceCompletionOwner,
    mut commands: mpsc::Receiver<WorkerCommand>,
    config: WorkerConfig,
) where
    E: InferenceEngine,
{
    let mut active = HashMap::<RequestId, ActiveRequest>::new();
    let mut cancellation_scratch = Vec::<RequestId>::new();
    let mut fatal_error: Option<String> = None;

    loop {
        if active.is_empty() {
            tokio::select! {
                error = completion_owner.reactor_failure() => {
                    tracing::error!(error = %error, "model completion reactor failed");
                    fatal_error = Some(error);
                }
                command = commands.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    if handle_command(command, &mut engine, &mut active, fatal_error.as_deref()) {
                        cancel_all(&mut engine, &mut completion_owner, &mut active).await;
                        break;
                    }
                }
            }
        }

        for _ in 0..config.max_commands_per_tick {
            match commands.try_recv() {
                Ok(command) => {
                    if handle_command(command, &mut engine, &mut active, fatal_error.as_deref()) {
                        cancel_all(&mut engine, &mut completion_owner, &mut active).await;
                        return;
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    cancel_all(&mut engine, &mut completion_owner, &mut active).await;
                    return;
                }
            }
        }

        cancel_disconnected(&mut engine, &mut active, &mut cancellation_scratch);
        drain_terminal(&mut engine, &mut active);
        if active.is_empty() || fatal_error.is_some() {
            continue;
        }

        // Arm before inspecting model state. A producer racing with `step` either
        // changes the epoch observed by this listener or wakes it after registration.
        let completion = completion_owner.listen();
        let step_result = {
            let mut emit = |event: &ResidentTokenEvent| -> Result<(), String> {
                let Some(request_id) = event.request_id else {
                    return Ok(());
                };
                let Some(request) = active.get_mut(&request_id) else {
                    return Ok(());
                };
                if request
                    .events
                    .try_send(WorkerEvent::Token {
                        text: event.text.clone(),
                    })
                    .is_err()
                {
                    request.cancellation.store(true, Ordering::Release);
                } else {
                    request.emitted_tokens = request.emitted_tokens.saturating_add(1);
                }
                Ok(())
            };
            engine.step(&mut emit)
        };

        match step_result {
            Ok(ResidentDriverStep::WaitingForModelProgress(_) | ResidentDriverStep::Blocked) => {
                if !engine.has_pending_async_work() {
                    let error = "runtime reported blocked work without an owned async continuation"
                        .to_owned();
                    tracing::error!(error = %error, "model worker entered a fatal scheduling state");
                    fatal_error = Some(error.clone());
                    fail_all(&mut engine, &mut active, error);
                    continue;
                }
                tokio::select! {
                    wake = completion => {
                        if matches!(wake, ferrule_common::CompletionWake::Closed) {
                            let error = "model completion source closed with live async work".to_owned();
                            tracing::error!(error = %error, "model completion source closed");
                            fatal_error = Some(error.clone());
                            fail_all(&mut engine, &mut active, error);
                        }
                    }
                    error = completion_owner.reactor_failure() => {
                        tracing::error!(error = %error, "model completion reactor failed");
                        fatal_error = Some(error.clone());
                        fail_all(&mut engine, &mut active, error);
                    }
                    command = commands.recv() => {
                        let Some(command) = command else {
                            cancel_all(&mut engine, &mut completion_owner, &mut active).await;
                            return;
                        };
                        if handle_command(command, &mut engine, &mut active, fatal_error.as_deref()) {
                            cancel_all(&mut engine, &mut completion_owner, &mut active).await;
                            return;
                        }
                    }
                }
            }
            Ok(ResidentDriverStep::Idle | ResidentDriverStep::Executed { .. }) => {}
            Err(error) => {
                tracing::error!(error = %error, "model worker entered a fatal execution state");
                fatal_error = Some(error.clone());
                fail_all(&mut engine, &mut active, error);
            }
        }
        cancel_disconnected(&mut engine, &mut active, &mut cancellation_scratch);
        drain_terminal(&mut engine, &mut active);
    }
}

fn handle_command<E>(
    command: WorkerCommand,
    engine: &mut E,
    active: &mut HashMap<RequestId, ActiveRequest>,
    fatal_error: Option<&str>,
) -> bool
where
    E: InferenceEngine,
{
    match command {
        WorkerCommand::Shutdown => true,
        WorkerCommand::Tokenize(command) => {
            let result = engine.encode(&command.prompt);
            let _ = command.response.send(result);
            false
        }
        WorkerCommand::Submit(command) => {
            let worker_started_at = Instant::now();
            let worker_queue_us = command.enqueued_at.elapsed().as_micros() as u64;
            if let Some(error) = fatal_error {
                let _ = command
                    .accepted
                    .send(Err(format!("model worker is unavailable: {error}")));
                return false;
            }
            let tokenize_started_at = Instant::now();
            let prompt_tokens = match engine.encode(&command.request.prompt) {
                Ok(tokens) if !tokens.is_empty() => tokens,
                Ok(_) => {
                    let _ = command
                        .accepted
                        .send(Err("formatted prompt produced no tokens".into()));
                    return false;
                }
                Err(error) => {
                    let _ = command
                        .accepted
                        .send(Err(format!("prompt tokenization failed: {error}")));
                    return false;
                }
            };
            let tokenization_us = tokenize_started_at.elapsed().as_micros() as u64;
            let request_id = command.request_id;
            let session_id = SessionId(request_id.0);
            let prompt_token_count = prompt_tokens.len();
            let max_new_tokens = command.request.max_tokens;
            let request = GenerateRequest {
                id: request_id,
                session_id: Some(session_id),
                prompt_tokens,
                max_new_tokens: command.request.max_tokens,
                stop: command.request.stop,
                ignore_eos: command.request.ignore_eos,
            };
            engine.submit(request);
            active.insert(
                request_id,
                ActiveRequest {
                    events: command.events,
                    cancellation: Arc::clone(&command.cancellation),
                    session_id,
                    submitted_at: command.enqueued_at,
                    emitted_tokens: 0,
                },
            );
            tracing::debug!(
                target: "ferrule_request",
                event = "request_admitted",
                request_id = request_id.0,
                session_id = session_id.0,
                worker_queue_us,
                tokenization_us,
                worker_admission_us = worker_started_at.elapsed().as_micros() as u64,
                prompt_tokens = prompt_token_count,
                max_new_tokens,
                "production request admitted"
            );
            if command.accepted.send(Ok(())).is_err() {
                command.cancellation.store(true, Ordering::Release);
            }
            false
        }
    }
}

fn cancel_disconnected<E>(
    engine: &mut E,
    active: &mut HashMap<RequestId, ActiveRequest>,
    scratch: &mut Vec<RequestId>,
) where
    E: InferenceEngine,
{
    scratch.clear();
    scratch.extend(active.iter().filter_map(|(request_id, request)| {
        request
            .cancellation
            .load(Ordering::Acquire)
            .then_some(*request_id)
    }));
    for request_id in scratch.iter().copied() {
        match engine.cancel_request(request_id) {
            Ok(
                InferenceCancelProgress::Complete(_)
                | InferenceCancelProgress::WaitingForModelProgress,
            ) => {}
            Err(error) => {
                if let Some(request) = active.remove(&request_id) {
                    trace_worker_request_terminal(
                        request_id,
                        request.session_id,
                        "failed",
                        "cancellation_failed",
                        &request,
                        None,
                    );
                    let _ = request.events.try_send(WorkerEvent::Failed {
                        message: format!("request cancellation failed: {error}"),
                    });
                }
            }
        }
    }
}

fn drain_terminal<E>(engine: &mut E, active: &mut HashMap<RequestId, ActiveRequest>)
where
    E: InferenceEngine,
{
    for sequence in engine.drain_finished() {
        let Some(request_id) = sequence.request_id else {
            continue;
        };
        let Some(request) = active.remove(&request_id) else {
            continue;
        };
        let reason = sequence
            .finish_reason
            .unwrap_or(SequenceFinishReason::NoCandidate);
        trace_worker_request_terminal(
            request_id,
            sequence.session_id,
            "finished",
            reason.as_str(),
            &request,
            Some(sequence.generated),
        );
        let _ = request.events.try_send(WorkerEvent::Finished {
            reason,
            usage: Usage::new(sequence.prompt_len, sequence.generated),
        });
    }
    for sequence in engine.drain_cancelled() {
        let Some(request_id) = sequence.request_id else {
            continue;
        };
        if let Some(request) = active.remove(&request_id) {
            trace_worker_request_terminal(
                request_id,
                sequence.session_id,
                "cancelled",
                SequenceFinishReason::Cancelled.as_str(),
                &request,
                Some(sequence.generated),
            );
            let _ = request.events.try_send(WorkerEvent::Cancelled);
        }
    }
    for sequence in engine.drain_failed() {
        let Some(request_id) = sequence.request_id else {
            continue;
        };
        if let Some(request) = active.remove(&request_id) {
            trace_worker_request_terminal(
                request_id,
                sequence.session_id,
                "failed",
                "model_execution_failed",
                &request,
                Some(sequence.generated),
            );
            let _ = request.events.try_send(WorkerEvent::Failed {
                message: "model execution failed".into(),
            });
        }
    }
}

fn trace_worker_request_terminal(
    request_id: RequestId,
    session_id: SessionId,
    status: &'static str,
    finish_reason: &'static str,
    request: &ActiveRequest,
    generated_tokens: Option<usize>,
) {
    tracing::debug!(
        target: "ferrule_request",
        event = "request_worker_terminal",
        request_id = request_id.0,
        session_id = session_id.0,
        status,
        finish_reason,
        worker_request_us = request.submitted_at.elapsed().as_micros() as u64,
        generated_tokens = ?generated_tokens,
        emitted_token_events = request.emitted_tokens,
        tokens_reconcile = ?generated_tokens.map(|generated| generated == request.emitted_tokens),
        "production request reached worker terminal state"
    );
}

fn fail_all<E>(engine: &mut E, active: &mut HashMap<RequestId, ActiveRequest>, message: String)
where
    E: InferenceEngine,
{
    let request_ids = active.keys().copied().collect::<Vec<_>>();
    for request_id in request_ids {
        let _ = engine.cancel_request(request_id);
        if let Some(request) = active.remove(&request_id) {
            trace_worker_request_terminal(
                request_id,
                request.session_id,
                "failed",
                "fatal_engine_error",
                &request,
                None,
            );
            let _ = request.events.try_send(WorkerEvent::Failed {
                message: message.clone(),
            });
        }
    }
    let _ = engine.drain_cancelled();
    let _ = engine.drain_failed();
}

async fn cancel_all<E>(
    engine: &mut E,
    completion_owner: &mut InferenceCompletionOwner,
    active: &mut HashMap<RequestId, ActiveRequest>,
) where
    E: InferenceEngine,
{
    loop {
        let completion = completion_owner.listen();
        let request_ids = active.keys().copied().collect::<Vec<_>>();
        let mut waiting = false;
        for request_id in request_ids {
            match engine.cancel_request(request_id) {
                Ok(InferenceCancelProgress::Complete(_)) => {}
                Ok(InferenceCancelProgress::WaitingForModelProgress) => waiting = true,
                Err(error) => {
                    if let Some(request) = active.remove(&request_id) {
                        trace_worker_request_terminal(
                            request_id,
                            request.session_id,
                            "failed",
                            "shutdown_cancellation_failed",
                            &request,
                            None,
                        );
                        let _ = request.events.try_send(WorkerEvent::Failed {
                            message: format!(
                                "request cancellation failed during shutdown: {error}"
                            ),
                        });
                    }
                }
            }
        }
        drain_terminal(engine, active);
        if active.is_empty() {
            return;
        }
        if !waiting {
            break;
        }
        if !engine.has_pending_async_work() {
            tracing::error!(
                "engine reported pending cancellation without an owned async continuation"
            );
            break;
        }
        if let Err(error) = completion_owner.wait(completion).await {
            tracing::error!(%error, "failed while waiting for model cancellation during shutdown");
            break;
        }
    }

    for (request_id, request) in active.drain() {
        trace_worker_request_terminal(
            request_id,
            request.session_id,
            "cancelled",
            "worker_shutdown",
            &request,
            None,
        );
        let _ = request.events.try_send(WorkerEvent::Cancelled);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_common::CompletionHub;
    use ferrule_common::execution::ExecutionTransactionId;
    use ferrule_model::{BatchContinuationId, PendingExpertLoad, PendingModelProgress};
    use ferrule_runtime::{CancelRequestResult, InferenceCompletionReactor, SequenceState};
    use std::sync::atomic::AtomicUsize;

    struct DisconnectEngine {
        completion_hub: CompletionHub,
        request: Option<GenerateRequest>,
        token_index: usize,
        cancellation_count: Arc<AtomicUsize>,
        cancellation_waits_remaining: usize,
        cancelled: Vec<SequenceState>,
    }

    impl InferenceEngine for DisconnectEngine {
        fn completion_hub(&self) -> CompletionHub {
            self.completion_hub.clone()
        }

        fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
            Vec::new()
        }

        fn has_pending_async_work(&self) -> bool {
            self.request.is_some() && self.cancellation_count.load(Ordering::Acquire) > 0
        }

        fn encode(&self, prompt: &str) -> Result<Vec<u32>, String> {
            Ok(prompt.bytes().map(u32::from).collect())
        }

        fn submit(&mut self, request: GenerateRequest) {
            self.request = Some(request);
            self.token_index = 0;
        }

        fn step(
            &mut self,
            on_token: &mut dyn FnMut(&ResidentTokenEvent) -> Result<(), String>,
        ) -> Result<ResidentDriverStep, String> {
            std::thread::sleep(std::time::Duration::from_millis(2));
            let Some(request) = self.request.as_ref() else {
                return Ok(ResidentDriverStep::Idle);
            };
            on_token(&ResidentTokenEvent {
                session_id: request.session_id.unwrap(),
                request_id: Some(request.id),
                index: self.token_index,
                token: 1,
                logit: Some(1.0),
                text: "x".into(),
            })?;
            self.token_index += 1;
            Ok(ResidentDriverStep::Executed {
                action_kind: ferrule_runtime::ResidentActionKind::Decode,
                rows: 1,
                staged: 1,
                finished: 0,
            })
        }

        fn cancel_request(
            &mut self,
            request_id: RequestId,
        ) -> Result<InferenceCancelProgress, String> {
            self.cancellation_count.fetch_add(1, Ordering::AcqRel);
            if self.cancellation_waits_remaining > 0 {
                self.cancellation_waits_remaining -= 1;
                self.completion_hub.notify();
                return Ok(InferenceCancelProgress::WaitingForModelProgress);
            }
            let Some(request) = self.request.take() else {
                return Ok(InferenceCancelProgress::Complete(
                    CancelRequestResult::NotFound { request_id },
                ));
            };
            let session_id = request.session_id.unwrap();
            let mut sequence = SequenceState::from_request(&request, session_id);
            sequence.finish_reason = Some(SequenceFinishReason::Cancelled);
            self.cancelled.push(sequence);
            Ok(InferenceCancelProgress::Complete(
                CancelRequestResult::Active {
                    request_id,
                    session_id,
                },
            ))
        }

        fn drain_finished(&mut self) -> Vec<SequenceState> {
            Vec::new()
        }

        fn drain_cancelled(&mut self) -> Vec<SequenceState> {
            std::mem::take(&mut self.cancelled)
        }

        fn drain_failed(&mut self) -> Vec<SequenceState> {
            Vec::new()
        }
    }

    struct CompletionDrivenEngine {
        completion_hub: CompletionHub,
        ready: Arc<AtomicBool>,
        step_calls: Arc<AtomicUsize>,
        request: Option<GenerateRequest>,
        finished: Vec<SequenceState>,
    }

    impl InferenceEngine for CompletionDrivenEngine {
        fn completion_hub(&self) -> CompletionHub {
            self.completion_hub.clone()
        }

        fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
            Vec::new()
        }

        fn has_pending_async_work(&self) -> bool {
            self.request.is_some() && !self.ready.load(Ordering::Acquire)
        }

        fn encode(&self, prompt: &str) -> Result<Vec<u32>, String> {
            Ok(prompt.bytes().map(u32::from).collect())
        }

        fn submit(&mut self, request: GenerateRequest) {
            self.request = Some(request);
        }

        fn step(
            &mut self,
            on_token: &mut dyn FnMut(&ResidentTokenEvent) -> Result<(), String>,
        ) -> Result<ResidentDriverStep, String> {
            self.step_calls.fetch_add(1, Ordering::AcqRel);
            if self.request.is_none() {
                return Ok(ResidentDriverStep::Idle);
            }
            if !self.ready.load(Ordering::Acquire) {
                let continuation =
                    BatchContinuationId::new(1).map_err(|error| error.to_string())?;
                let load = PendingExpertLoad::new(1, 0, 0).map_err(|error| error.to_string())?;
                let transaction =
                    ExecutionTransactionId::new(1).map_err(|error| error.to_string())?;
                let pending = PendingModelProgress::new(transaction, continuation, vec![load])
                    .map_err(|error| error.to_string())?;
                return Ok(ResidentDriverStep::WaitingForModelProgress(vec![pending]));
            }

            let request = self.request.take().expect("checked request above");
            let session_id = request.session_id.expect("worker assigns a session");
            on_token(&ResidentTokenEvent {
                session_id,
                request_id: Some(request.id),
                index: 0,
                token: 1,
                logit: Some(1.0),
                text: "ready".into(),
            })?;
            let mut sequence = SequenceState::from_request(&request, session_id);
            sequence.generated = 1;
            sequence.finish_reason = Some(SequenceFinishReason::MaxTokens);
            self.finished.push(sequence);
            Ok(ResidentDriverStep::Executed {
                action_kind: ferrule_runtime::ResidentActionKind::Decode,
                rows: 1,
                staged: 0,
                finished: 1,
            })
        }

        fn cancel_request(
            &mut self,
            request_id: RequestId,
        ) -> Result<InferenceCancelProgress, String> {
            let Some(request) = self.request.take() else {
                return Ok(InferenceCancelProgress::Complete(
                    CancelRequestResult::NotFound { request_id },
                ));
            };
            Ok(InferenceCancelProgress::Complete(
                CancelRequestResult::Active {
                    request_id,
                    session_id: request.session_id.expect("worker assigns a session"),
                },
            ))
        }

        fn drain_finished(&mut self) -> Vec<SequenceState> {
            std::mem::take(&mut self.finished)
        }

        fn drain_cancelled(&mut self) -> Vec<SequenceState> {
            Vec::new()
        }

        fn drain_failed(&mut self) -> Vec<SequenceState> {
            Vec::new()
        }
    }

    fn test_request(id: u64) -> GenerateRequest {
        GenerateRequest {
            id: RequestId(id),
            session_id: Some(SessionId(id)),
            prompt_tokens: vec![1],
            max_new_tokens: 8,
            stop: Vec::new(),
            ignore_eos: false,
        }
    }

    #[test]
    fn disconnected_cancellation_reuses_scratch_without_stale_requests() {
        let cancellation_count = Arc::new(AtomicUsize::new(0));
        let mut engine = DisconnectEngine {
            completion_hub: CompletionHub::new(),
            request: Some(test_request(1)),
            token_index: 0,
            cancellation_count: Arc::clone(&cancellation_count),
            cancellation_waits_remaining: 0,
            cancelled: Vec::new(),
        };
        let mut active = HashMap::new();
        let (events, _events_receiver) = mpsc::channel(1);
        active.insert(
            RequestId(1),
            ActiveRequest {
                events,
                cancellation: Arc::new(AtomicBool::new(true)),
                session_id: SessionId(1),
                submitted_at: Instant::now(),
                emitted_tokens: 0,
            },
        );
        let mut scratch = Vec::with_capacity(1);
        let scratch_pointer = scratch.as_ptr();

        cancel_disconnected(&mut engine, &mut active, &mut scratch);

        assert_eq!(scratch, vec![RequestId(1)]);
        assert_eq!(scratch.as_ptr(), scratch_pointer);
        assert_eq!(cancellation_count.load(Ordering::Acquire), 1);
        drain_terminal(&mut engine, &mut active);
        assert!(active.is_empty());

        engine.submit(test_request(2));
        let (events, _events_receiver) = mpsc::channel(1);
        active.insert(
            RequestId(2),
            ActiveRequest {
                events,
                cancellation: Arc::new(AtomicBool::new(false)),
                session_id: SessionId(2),
                submitted_at: Instant::now(),
                emitted_tokens: 0,
            },
        );
        cancel_disconnected(&mut engine, &mut active, &mut scratch);

        assert!(scratch.is_empty());
        assert_eq!(scratch.as_ptr(), scratch_pointer);
        assert_eq!(cancellation_count.load(Ordering::Acquire), 1);
        assert_eq!(
            engine.request.as_ref().map(|request| request.id),
            Some(RequestId(2))
        );
    }

    #[tokio::test]
    async fn waiting_worker_resumes_only_after_completion_wake() {
        let completion_hub = CompletionHub::new();
        let ready = Arc::new(AtomicBool::new(false));
        let step_calls = Arc::new(AtomicUsize::new(0));
        let worker = spawn_model_worker(
            CompletionDrivenEngine {
                completion_hub: completion_hub.clone(),
                ready: Arc::clone(&ready),
                step_calls: Arc::clone(&step_calls),
                request: None,
                finished: Vec::new(),
            },
            WorkerConfig::default(),
        )
        .unwrap();
        let mut subscription = worker
            .handle()
            .submit(WorkerRequest {
                prompt: "wake".into(),
                max_tokens: 1,
                stop: Vec::new(),
                ignore_eos: false,
            })
            .await
            .unwrap();

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while step_calls.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("worker never entered the waiting model step");
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert_eq!(step_calls.load(Ordering::Acquire), 1);

        ready.store(true, Ordering::Release);
        completion_hub.notify();
        assert!(matches!(
            tokio::time::timeout(std::time::Duration::from_secs(1), subscription.recv())
                .await
                .unwrap(),
            Some(WorkerEvent::Token { text }) if text == "ready"
        ));
        assert!(matches!(
            subscription.recv().await,
            Some(WorkerEvent::Finished { .. })
        ));
        assert_eq!(step_calls.load(Ordering::Acquire), 2);
        worker.shutdown().await.unwrap();
    }

    #[test]
    fn pending_cancellation_retains_request_ownership_until_model_quiesces() {
        let cancellation_count = Arc::new(AtomicUsize::new(0));
        let mut engine = DisconnectEngine {
            completion_hub: CompletionHub::new(),
            request: Some(test_request(3)),
            token_index: 0,
            cancellation_count: Arc::clone(&cancellation_count),
            cancellation_waits_remaining: 1,
            cancelled: Vec::new(),
        };
        let mut active = HashMap::new();
        let (events, _events_receiver) = mpsc::channel(1);
        active.insert(
            RequestId(3),
            ActiveRequest {
                events,
                cancellation: Arc::new(AtomicBool::new(true)),
                session_id: SessionId(3),
                submitted_at: Instant::now(),
                emitted_tokens: 0,
            },
        );
        let mut scratch = Vec::new();

        cancel_disconnected(&mut engine, &mut active, &mut scratch);

        assert!(active.contains_key(&RequestId(3)));
        assert!(engine.request.is_some());
        assert!(engine.cancelled.is_empty());
        assert_eq!(cancellation_count.load(Ordering::Acquire), 1);

        cancel_disconnected(&mut engine, &mut active, &mut scratch);

        assert!(active.contains_key(&RequestId(3)));
        assert!(engine.request.is_none());
        assert_eq!(cancellation_count.load(Ordering::Acquire), 2);
        drain_terminal(&mut engine, &mut active);
        assert!(active.is_empty());
    }

    #[tokio::test]
    async fn shutdown_waits_for_pending_model_cancellation_before_dropping_ownership() {
        let cancellation_count = Arc::new(AtomicUsize::new(0));
        let mut engine = DisconnectEngine {
            completion_hub: CompletionHub::new(),
            request: Some(test_request(4)),
            token_index: 0,
            cancellation_count: Arc::clone(&cancellation_count),
            cancellation_waits_remaining: 1,
            cancelled: Vec::new(),
        };
        let mut completion_owner = InferenceCompletionOwner::attach(&mut engine);
        let mut active = HashMap::new();
        let (events, mut events_receiver) = mpsc::channel(1);
        active.insert(
            RequestId(4),
            ActiveRequest {
                events,
                cancellation: Arc::new(AtomicBool::new(false)),
                session_id: SessionId(4),
                submitted_at: Instant::now(),
                emitted_tokens: 0,
            },
        );

        cancel_all(&mut engine, &mut completion_owner, &mut active).await;

        assert!(active.is_empty());
        assert_eq!(cancellation_count.load(Ordering::Acquire), 2);
        assert!(matches!(
            events_receiver.recv().await,
            Some(WorkerEvent::Cancelled)
        ));
    }

    #[tokio::test]
    async fn dropping_event_subscription_cancels_without_poisoning_worker() {
        let cancellation_count = Arc::new(AtomicUsize::new(0));
        let worker = spawn_model_worker(
            DisconnectEngine {
                completion_hub: CompletionHub::new(),
                request: None,
                token_index: 0,
                cancellation_count: Arc::clone(&cancellation_count),
                cancellation_waits_remaining: 0,
                cancelled: Vec::new(),
            },
            WorkerConfig {
                event_queue_capacity: 32,
                ..WorkerConfig::default()
            },
        )
        .unwrap();
        let handle = worker.handle();
        let mut first = handle
            .submit(WorkerRequest {
                prompt: "first".into(),
                max_tokens: 128,
                stop: Vec::new(),
                ignore_eos: false,
            })
            .await
            .unwrap();
        assert!(matches!(
            first.recv().await,
            Some(WorkerEvent::Token { .. })
        ));
        drop(first);

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while cancellation_count.load(Ordering::Acquire) == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("worker did not observe the dropped response");

        let mut second = handle
            .submit(WorkerRequest {
                prompt: "second".into(),
                max_tokens: 128,
                stop: Vec::new(),
                ignore_eos: false,
            })
            .await
            .unwrap();
        assert!(matches!(
            second.recv().await,
            Some(WorkerEvent::Token { .. })
        ));
        drop(second);
        worker.shutdown().await.unwrap();
    }
}
