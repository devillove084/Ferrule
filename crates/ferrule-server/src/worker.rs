use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Instant;

use ferrule_model::{ModelRunner, MultiSessionRunner, models::deepseek_v4::DeepSeekV4Runner};
use ferrule_runtime::{
    CancelRequestResult, ExpertIoBudget, FixedSequenceSlotPool, GenerateRequest, RequestId,
    ResidentActionKind, ResidentDriverStep, ResidentTokenEvent, ResidentTopKDriver,
    SequenceFinishReason, SequenceSlotPool, SequenceState, SessionId,
};
use tokio::sync::{mpsc, oneshot};

use crate::config::WorkerConfig;
use crate::openai::Usage;

/// Synchronous model execution boundary owned by one dedicated worker thread.
///
/// The HTTP layer only sees [`ModelWorkerHandle`]. This trait exists so the
/// ownership loop can be tested without a concrete model and so protocol code
/// never depends on a model family.
pub trait ModelEngine: Send + 'static {
    fn encode(&self, prompt: &str) -> Result<Vec<u32>, String>;
    fn submit(&mut self, request: GenerateRequest);
    fn step(
        &mut self,
        on_token: &mut dyn FnMut(&ResidentTokenEvent),
    ) -> Result<ResidentDriverStep, String>;
    fn cancel_request(&mut self, request_id: RequestId) -> Result<CancelRequestResult, String>;
    fn drain_finished(&mut self) -> Vec<SequenceState>;
    fn drain_cancelled(&mut self) -> Vec<SequenceState>;
    fn drain_failed(&mut self) -> Vec<SequenceState>;
}

pub struct DeepSeekV4ResidentEngine<C>
where
    C: SequenceSlotPool,
{
    driver: ResidentTopKDriver<DeepSeekV4Runner, C>,
    expert_budget: ExpertIoBudget,
    next_execution_step_id: u64,
}

impl<C> DeepSeekV4ResidentEngine<C>
where
    C: SequenceSlotPool,
{
    pub fn new(
        driver: ResidentTopKDriver<DeepSeekV4Runner, C>,
        expert_budget: ExpertIoBudget,
    ) -> Self {
        Self {
            driver,
            expert_budget,
            next_execution_step_id: 1,
        }
    }
}

impl<C> ModelEngine for DeepSeekV4ResidentEngine<C>
where
    C: SequenceSlotPool + Send + 'static,
{
    fn encode(&self, prompt: &str) -> Result<Vec<u32>, String> {
        self.driver
            .executor()
            .runner()
            .encode(prompt)
            .map_err(|error| error.to_string())
    }

    fn submit(&mut self, request: GenerateRequest) {
        self.driver.submit(request);
    }

    fn step(
        &mut self,
        on_token: &mut dyn FnMut(&ResidentTokenEvent),
    ) -> Result<ResidentDriverStep, String> {
        let execution_step_id = self.next_execution_step_id;
        self.next_execution_step_id = self
            .next_execution_step_id
            .checked_add(1)
            .ok_or_else(|| "production execution-step identity overflow".to_string())?;
        let mut adapter = |event: &ResidentTokenEvent| {
            on_token(event);
            Ok(())
        };
        let counters_before = self.driver.executor().runner().operator_runtime_counters();
        let dspark_before = self.driver.stats().dspark.clone();
        let result = self
            .driver
            .step_with_dspark_model_expert_io(&mut adapter, self.expert_budget);
        let counters_after = self.driver.executor().runner().operator_runtime_counters();
        let traces = self.driver.drain_dspark_cycle_traces().collect::<Vec<_>>();
        let cycle_identities = traces
            .iter()
            .map(|trace| (trace.request_id.0, trace.session_id.0, trace.cycle_attempt))
            .collect::<Vec<_>>();
        let physical_counters_cycle_exact = matches!(
            &result,
            Ok(ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            })
        ) && traces.len() == 1;
        for trace in traces {
            tracing::debug!(
                target: "ferrule_dspark_cycle",
                event = "dspark_cycle",
                request_id = trace.request_id.0,
                session_id = trace.session_id.0,
                cycle_attempt = trace.cycle_attempt,
                execution_step_id,
                position = trace.position,
                anchor_token = trace.anchor_token,
                proposal_source = trace.proposal_source.implementation,
                proposal_prepared_plan_id = trace.proposal_source.prepared_plan_id,
                native_proposal_width = trace.proposal_source.native_width,
                proposal_executed = trace.proposal_executed,
                native_proposal_token_ids = ?trace.native_proposed_tokens,
                native_confidence_logits = ?trace.native_confidence_logits,
                proposal_token_ids = ?trace.proposed_tokens,
                confidence_logits = ?trace.confidence_logits,
                capacity_truncated_tokens = trace.capacity_truncated_tokens,
                output_boundary_truncated_tokens = trace.output_boundary_truncated_tokens,
                scheduler_expert_io = ?trace.scheduler_expert_io,
                accepted_token_ids = ?trace.accepted_tokens,
                rejected_token = ?trace.rejected_token,
                target_correction_token = ?trace.target_correction_token,
                target_next_token = ?trace.target_next_token,
                target_row_top1 = ?trace.target_row_top1,
                proposed_tokens = trace.accounting.proposed_tokens,
                verified_rows = trace.accounting.verified_rows,
                accepted_draft_tokens = trace.accounting.accepted_draft_tokens,
                correction_tokens = trace.accounting.correction_tokens,
                externally_committed_tokens = trace.accounting.externally_committed_tokens,
                runtime_emitted_tokens = trace.runtime_emitted_tokens,
                rolled_back_rows = trace.accounting.rolled_back_rows,
                runtime_tokens_reconcile = trace.runtime_tokens_reconcile(),
                proposal_us = trace.proposal_time_us,
                verify_us = trace.verify_time_us,
                transaction_us = trace.transaction_time_us,
                cycle_us = trace.complete_cycle_time_us,
                finish_reason = trace.finish_reason.map(|reason| reason.as_str()).unwrap_or("none"),
                "production DSpark cycle trace"
            );
        }
        if !cycle_identities.is_empty() || result.is_err() {
            tracing::debug!(
                target: "ferrule_runtime_step",
                event = "dspark_physical_counters",
                execution_step_id,
                cycle_identities = ?cycle_identities,
                step_succeeded = result.is_ok(),
                physical_counters_cycle_exact,
                dropped_cycle_traces = self.driver.stats().dropped_dspark_cycle_traces,
                kernel_launches = counters_after.kernel_launches.saturating_sub(counters_before.kernel_launches),
                h2d_copies = counters_after.host_to_device_copies.saturating_sub(counters_before.host_to_device_copies),
                h2d_bytes = counters_after.host_to_device_bytes.saturating_sub(counters_before.host_to_device_bytes),
                d2h_copies = counters_after.device_to_host_copies.saturating_sub(counters_before.device_to_host_copies),
                d2h_bytes = counters_after.device_to_host_bytes.saturating_sub(counters_before.device_to_host_bytes),
                artifact_uploads = counters_after.artifact_uploads.saturating_sub(counters_before.artifact_uploads),
                artifact_upload_bytes = counters_after.artifact_upload_bytes.saturating_sub(counters_before.artifact_upload_bytes),
                device_allocation_attempts = counters_after.device_allocation_attempts.saturating_sub(counters_before.device_allocation_attempts),
                device_allocations = counters_after.device_allocations.saturating_sub(counters_before.device_allocations),
                device_allocation_failures = counters_after.device_allocation_failures.saturating_sub(counters_before.device_allocation_failures),
                device_allocation_bytes = counters_after.device_allocation_bytes.saturating_sub(counters_before.device_allocation_bytes),
                stream_wide_syncs = counters_after.stream_wide_syncs.saturating_sub(counters_before.stream_wide_syncs),
                selected_experts = counters_after.expert_selected.saturating_sub(counters_before.expert_selected),
                resident_hits = counters_after.expert_selected_resident_hits.saturating_sub(counters_before.expert_selected_resident_hits),
                upload_hits = counters_after.expert_selected_upload_hits.saturating_sub(counters_before.expert_selected_upload_hits),
                host_staged_hits = counters_after.expert_selected_host_staged_hits.saturating_sub(counters_before.expert_selected_host_staged_hits),
                cold_misses = counters_after.expert_selected_cold_misses.saturating_sub(counters_before.expert_selected_cold_misses),
                expert_loads = counters_after.expert_loads.saturating_sub(counters_before.expert_loads),
                expert_load_bytes = counters_after.expert_load_bytes.saturating_sub(counters_before.expert_load_bytes),
                expert_io_submitted_extents = counters_after.expert_io_submitted_extents.saturating_sub(counters_before.expert_io_submitted_extents),
                expert_io_completed_extents = counters_after.expert_io_completed_extents.saturating_sub(counters_before.expert_io_completed_extents),
                expert_io_failed_extents = counters_after.expert_io_failed_extents.saturating_sub(counters_before.expert_io_failed_extents),
                expert_io_requested_bytes = counters_after.expert_io_requested_bytes.saturating_sub(counters_before.expert_io_requested_bytes),
                expert_io_aligned_bytes = counters_after.expert_io_aligned_bytes.saturating_sub(counters_before.expert_io_aligned_bytes),
                expert_io_read_us = counters_after.expert_io_read_us.saturating_sub(counters_before.expert_io_read_us),
                expert_async_upload_bytes = counters_after.expert_async_upload_bytes.saturating_sub(counters_before.expert_async_upload_bytes),
                expert_frame_reuses = counters_after.expert_frame_reuses.saturating_sub(counters_before.expert_frame_reuses),
                expert_frame_waits = counters_after.expert_frame_waits.saturating_sub(counters_before.expert_frame_waits),
                abandoned_uploads = counters_after.expert_abandoned_uploads.saturating_sub(counters_before.expert_abandoned_uploads),
                "production DSpark physical counter delta"
            );
        }
        if matches!(
            &result,
            Ok(ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Decode,
                ..
            })
        ) {
            let dspark_after = &self.driver.stats().dspark;
            let proposed = dspark_after
                .proposed_tokens
                .saturating_sub(dspark_before.proposed_tokens);
            let accepted = dspark_after
                .accepted_draft_tokens
                .saturating_sub(dspark_before.accepted_draft_tokens);
            tracing::info!(
                cycles = dspark_after.cycles.saturating_sub(dspark_before.cycles),
                proposed_tokens = proposed,
                verified_rows = dspark_after
                    .verified_rows
                    .saturating_sub(dspark_before.verified_rows),
                accepted_draft_tokens = accepted,
                runtime_emitted_tokens = dspark_after
                    .runtime_emitted_tokens
                    .saturating_sub(dspark_before.runtime_emitted_tokens),
                acceptance = if proposed == 0 {
                    0.0
                } else {
                    accepted as f64 / proposed as f64
                },
                kernel_launches = counters_after
                    .kernel_launches
                    .saturating_sub(counters_before.kernel_launches),
                h2d_copies = counters_after
                    .host_to_device_copies
                    .saturating_sub(counters_before.host_to_device_copies),
                h2d_bytes = counters_after
                    .host_to_device_bytes
                    .saturating_sub(counters_before.host_to_device_bytes),
                d2h_copies = counters_after
                    .device_to_host_copies
                    .saturating_sub(counters_before.device_to_host_copies),
                d2h_bytes = counters_after
                    .device_to_host_bytes
                    .saturating_sub(counters_before.device_to_host_bytes),
                device_allocations = counters_after
                    .device_allocations
                    .saturating_sub(counters_before.device_allocations),
                stream_wide_syncs = counters_after
                    .stream_wide_syncs
                    .saturating_sub(counters_before.stream_wide_syncs),
                selected_experts = counters_after
                    .expert_selected
                    .saturating_sub(counters_before.expert_selected),
                resident_hits = counters_after
                    .expert_selected_resident_hits
                    .saturating_sub(counters_before.expert_selected_resident_hits),
                cold_misses = counters_after
                    .expert_selected_cold_misses
                    .saturating_sub(counters_before.expert_selected_cold_misses),
                expert_loads = counters_after
                    .expert_loads
                    .saturating_sub(counters_before.expert_loads),
                expert_load_bytes = counters_after
                    .expert_load_bytes
                    .saturating_sub(counters_before.expert_load_bytes),
                expert_io_requested_bytes = counters_after
                    .expert_io_requested_bytes
                    .saturating_sub(counters_before.expert_io_requested_bytes),
                moe_total_us = counters_after
                    .moe_total_us
                    .saturating_sub(counters_before.moe_total_us),
                "production DSpark decode counters"
            );
        }
        result.map_err(|error| error.to_string())
    }

    fn cancel_request(&mut self, request_id: RequestId) -> Result<CancelRequestResult, String> {
        self.driver
            .cancel_request(request_id)
            .map_err(|error| error.to_string())
    }

    fn drain_finished(&mut self) -> Vec<SequenceState> {
        self.driver.drain_finished()
    }

    fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        self.driver.drain_cancelled()
    }

    fn drain_failed(&mut self) -> Vec<SequenceState> {
        self.driver.drain_failed()
    }
}

impl<R, C> ModelEngine for ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner + Send + 'static,
    R::SequenceState: Send + 'static,
    C: SequenceSlotPool + Send + 'static,
{
    fn encode(&self, prompt: &str) -> Result<Vec<u32>, String> {
        self.executor()
            .runner()
            .encode(prompt)
            .map_err(|error| error.to_string())
    }

    fn submit(&mut self, request: GenerateRequest) {
        ResidentTopKDriver::submit(self, request);
    }

    fn step(
        &mut self,
        on_token: &mut dyn FnMut(&ResidentTokenEvent),
    ) -> Result<ResidentDriverStep, String> {
        let mut adapter = |event: &ResidentTokenEvent| {
            on_token(event);
            Ok(())
        };
        ResidentTopKDriver::step(self, &mut adapter).map_err(|error| error.to_string())
    }

    fn cancel_request(&mut self, request_id: RequestId) -> Result<CancelRequestResult, String> {
        ResidentTopKDriver::cancel_request(self, request_id).map_err(|error| error.to_string())
    }

    fn drain_finished(&mut self) -> Vec<SequenceState> {
        ResidentTopKDriver::drain_finished(self)
    }

    fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        ResidentTopKDriver::drain_cancelled(self)
    }

    fn drain_failed(&mut self) -> Vec<SequenceState> {
        ResidentTopKDriver::drain_failed(self)
    }
}

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
    E: ModelEngine,
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
    E: ModelEngine,
{
    config.validate().map_err(str::to_string)?;
    let (commands, receiver) = mpsc::channel(config.command_queue_capacity);
    let (ready_sender, ready_receiver) = std::sync::mpsc::sync_channel(1);
    let thread_config = config.clone();
    let thread = std::thread::Builder::new()
        .name("ferrule-model-worker".into())
        .spawn(move || match factory() {
            Ok(engine) => {
                let _ = ready_sender.send(Ok(()));
                run_worker(engine, receiver, thread_config);
            }
            Err(error) => {
                let _ = ready_sender.send(Err(error));
            }
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

fn run_worker<E>(mut engine: E, mut commands: mpsc::Receiver<WorkerCommand>, config: WorkerConfig)
where
    E: ModelEngine,
{
    let mut active = HashMap::<RequestId, ActiveRequest>::new();
    let mut cancellation_scratch = Vec::<RequestId>::new();
    let mut fatal_error: Option<String> = None;

    loop {
        if active.is_empty() {
            let Some(command) = commands.blocking_recv() else {
                break;
            };
            if handle_command(command, &mut engine, &mut active, fatal_error.as_deref()) {
                cancel_all(&mut engine, &mut active);
                break;
            }
        }

        for _ in 0..config.max_commands_per_tick {
            match commands.try_recv() {
                Ok(command) => {
                    if handle_command(command, &mut engine, &mut active, fatal_error.as_deref()) {
                        cancel_all(&mut engine, &mut active);
                        return;
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    cancel_all(&mut engine, &mut active);
                    return;
                }
            }
        }

        cancel_disconnected(&mut engine, &mut active, &mut cancellation_scratch);
        drain_terminal(&mut engine, &mut active);
        if active.is_empty() || fatal_error.is_some() {
            continue;
        }

        let step_result = {
            let mut emit = |event: &ResidentTokenEvent| {
                let Some(request_id) = event.request_id else {
                    return;
                };
                let Some(request) = active.get_mut(&request_id) else {
                    return;
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
            };
            engine.step(&mut emit)
        };

        match step_result {
            Ok(ResidentDriverStep::Blocked) => std::thread::sleep(config.blocked_backoff),
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
    E: ModelEngine,
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
    E: ModelEngine,
{
    scratch.clear();
    scratch.extend(active.iter().filter_map(|(request_id, request)| {
        request
            .cancellation
            .load(Ordering::Acquire)
            .then_some(*request_id)
    }));
    for request_id in scratch.iter().copied() {
        if let Err(error) = engine.cancel_request(request_id)
            && let Some(request) = active.remove(&request_id)
        {
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

fn drain_terminal<E>(engine: &mut E, active: &mut HashMap<RequestId, ActiveRequest>)
where
    E: ModelEngine,
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
    E: ModelEngine,
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

fn cancel_all<E>(engine: &mut E, active: &mut HashMap<RequestId, ActiveRequest>)
where
    E: ModelEngine,
{
    let request_ids = active.keys().copied().collect::<Vec<_>>();
    for request_id in request_ids {
        let _ = engine.cancel_request(request_id);
    }
    drain_terminal(engine, active);
    for (request_id, request) in active.drain() {
        trace_worker_request_terminal(
            request_id,
            request.session_id,
            "cancelled",
            "worker_shutdown",
            &request,
            None,
        );
    }
}

// Keep this concrete alias visible in rustdoc as the intended production engine shape.
#[allow(dead_code)]
type DefaultResidentEngine<R> = ResidentTopKDriver<R, FixedSequenceSlotPool>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct DisconnectEngine {
        request: Option<GenerateRequest>,
        token_index: usize,
        cancellation_count: Arc<AtomicUsize>,
        cancelled: Vec<SequenceState>,
    }

    impl ModelEngine for DisconnectEngine {
        fn encode(&self, prompt: &str) -> Result<Vec<u32>, String> {
            Ok(prompt.bytes().map(u32::from).collect())
        }

        fn submit(&mut self, request: GenerateRequest) {
            self.request = Some(request);
            self.token_index = 0;
        }

        fn step(
            &mut self,
            on_token: &mut dyn FnMut(&ResidentTokenEvent),
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
            });
            self.token_index += 1;
            Ok(ResidentDriverStep::Executed {
                action_kind: ferrule_runtime::ResidentActionKind::Decode,
                rows: 1,
                staged: 1,
                finished: 0,
            })
        }

        fn cancel_request(&mut self, request_id: RequestId) -> Result<CancelRequestResult, String> {
            let Some(request) = self.request.take() else {
                return Ok(CancelRequestResult::NotFound { request_id });
            };
            self.cancellation_count.fetch_add(1, Ordering::AcqRel);
            let session_id = request.session_id.unwrap();
            let mut sequence = SequenceState::from_request(&request, session_id);
            sequence.finish_reason = Some(SequenceFinishReason::Cancelled);
            self.cancelled.push(sequence);
            Ok(CancelRequestResult::Active {
                request_id,
                session_id,
            })
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
            request: Some(test_request(1)),
            token_index: 0,
            cancellation_count: Arc::clone(&cancellation_count),
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
    async fn dropping_event_subscription_cancels_without_poisoning_worker() {
        let cancellation_count = Arc::new(AtomicUsize::new(0));
        let worker = spawn_model_worker(
            DisconnectEngine {
                request: None,
                token_index: 0,
                cancellation_count: Arc::clone(&cancellation_count),
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
