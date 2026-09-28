use std::collections::HashMap;
use std::error::Error as StdError;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use ferrule_common::observability::{METRICS, Metrics, RequestMetricsGuard};
use ferrule_common::{
    WorkerExecutionError, WorkerOperation, WorkerRequestError, WorkerShutdownError,
    WorkerStartError,
};
use ferrule_runtime::{
    GenerateRequest, InferenceCancelProgress, InferenceCompletionOwner, InferenceEngine,
    InferenceRequestCleanup, InferenceShutdownProgress, RequestId, ResidentDriverStep,
    ResidentTokenEvent, Result as RuntimeResult, SequenceFinishReason, SequenceState, SessionId,
};
use tokio::sync::{Notify, mpsc, oneshot};
use tokio::time::Instant as ShutdownDeadline;

use crate::config::WorkerConfig;
use crate::openai::Usage;

type RuntimeWorkerRequestError = WorkerRequestError<ferrule_runtime::Error>;
type RuntimeWorkerExecutionError = WorkerExecutionError<ferrule_runtime::Error>;
fn server_source_error(source: impl StdError + Send + Sync + 'static) -> ferrule_runtime::Error {
    ferrule_common::Error::Backend {
        source: Box::new(source),
    }
    .into()
}

#[cfg(test)]
fn admission_request_error(error: ServerAdmissionError) -> RuntimeWorkerRequestError {
    match error {
        ServerAdmissionError::Closed => WorkerRequestError::Unavailable {
            operation: WorkerOperation::Admission,
        },
        error => WorkerRequestError::Rejected {
            source: Arc::new(WorkerExecutionError::Runtime {
                source: server_source_error(error),
            }),
        },
    }
}

#[derive(Debug)]
pub(crate) struct SuspendedCancellation {
    pub request_id: RequestId,
    pub session_id: SessionId,
}
impl std::fmt::Display for SuspendedCancellation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "request {:?} session {:?} requires restore or shutdown",
            self.request_id, self.session_id
        )
    }
}
impl StdError for SuspendedCancellation {}

fn execution_error(error: Arc<RuntimeWorkerExecutionError>) -> ferrule_runtime::Error {
    ferrule_common::Error::ModelSource {
        source: Box::new(error),
    }
    .into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WorkerPhase {
    Starting = 0,
    Ready = 1,
    Draining = 2,
    Fatal = 3,
    Stopped = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerAdmissionResource {
    Requests,
    PromptBytes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerAdmissionError {
    Closed,
    Capacity {
        resource: ServerAdmissionResource,
        limit: usize,
        held: usize,
    },
}

impl std::fmt::Display for ServerAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => f.write_str("server admission is closed"),
            Self::Capacity {
                resource,
                limit,
                held,
            } => {
                write!(
                    f,
                    "server {resource:?} capacity {limit} exhausted (held {held})"
                )
            }
        }
    }
}
impl StdError for ServerAdmissionError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerAdmissionSnapshot {
    pub request_limit: usize,
    pub held_requests: usize,
    pub available_requests: usize,
    pub prompt_bytes_limit: usize,
    pub held_prompt_bytes: usize,
    pub available_prompt_bytes: usize,
    pub closed: bool,
}

struct ServerAdmissionCounts {
    held_requests: usize,
    held_prompt_bytes: usize,
    closed: bool,
}

#[derive(Clone)]
struct ServerAdmission {
    request_limit: usize,
    prompt_bytes_limit: usize,
    counts: Arc<Mutex<ServerAdmissionCounts>>,
}

impl ServerAdmission {
    pub fn new(request_limit: usize, prompt_bytes_limit: usize) -> Self {
        Self {
            request_limit,
            prompt_bytes_limit,
            counts: Arc::new(Mutex::new(ServerAdmissionCounts {
                held_requests: 0,
                held_prompt_bytes: 0,
                closed: false,
            })),
        }
    }

    pub fn snapshot(&self) -> ServerAdmissionSnapshot {
        let counts = self.counts.lock().unwrap();
        ServerAdmissionSnapshot {
            request_limit: self.request_limit,
            held_requests: counts.held_requests,
            available_requests: self.request_limit - counts.held_requests,
            prompt_bytes_limit: self.prompt_bytes_limit,
            held_prompt_bytes: counts.held_prompt_bytes,
            available_prompt_bytes: self.prompt_bytes_limit - counts.held_prompt_bytes,
            closed: counts.closed,
        }
    }

    pub fn close(&self) {
        self.counts.lock().unwrap().closed = true;
    }

    pub fn try_acquire(
        &self,
        prompt_bytes: usize,
    ) -> Result<ServerAdmissionPermit, ServerAdmissionError> {
        let mut counts = self.counts.lock().unwrap();
        if counts.closed {
            return Err(ServerAdmissionError::Closed);
        }
        if counts.held_requests >= self.request_limit {
            return Err(ServerAdmissionError::Capacity {
                resource: ServerAdmissionResource::Requests,
                limit: self.request_limit,
                held: counts.held_requests,
            });
        }
        if prompt_bytes
            > self
                .prompt_bytes_limit
                .saturating_sub(counts.held_prompt_bytes)
        {
            return Err(ServerAdmissionError::Capacity {
                resource: ServerAdmissionResource::PromptBytes,
                limit: self.prompt_bytes_limit,
                held: counts.held_prompt_bytes,
            });
        }
        counts.held_requests += 1;
        counts.held_prompt_bytes += prompt_bytes;
        drop(counts);
        Ok(ServerAdmissionPermit {
            lease: Arc::new(ServerAdmissionLease {
                admission: self.clone(),
                prompt_bytes: std::sync::atomic::AtomicUsize::new(prompt_bytes),
                cleanup: Mutex::new(AdmissionCleanup::default()),
            }),
        })
    }
}

#[derive(Default)]
struct AdmissionCleanup {
    request_id: Option<RequestId>,
    proof: InferenceRequestCleanup,
    terminal_consumed: bool,
}

impl AdmissionCleanup {
    fn released(&self) -> bool {
        match &self.proof {
            InferenceRequestCleanup::Tracked(receipt) => receipt.is_released(),
            InferenceRequestCleanup::TerminalQuiescent => self.terminal_consumed,
            InferenceRequestCleanup::Unavailable => false,
        }
    }
}

struct ServerAdmissionLease {
    admission: ServerAdmission,
    prompt_bytes: std::sync::atomic::AtomicUsize,
    cleanup: Mutex<AdmissionCleanup>,
}

pub(crate) struct ServerAdmissionPermit {
    lease: Arc<ServerAdmissionLease>,
}

impl ServerAdmissionPermit {
    // The observer and command keep the same counted request alive; cloning
    // this reference never acquires a second permit.
    fn observer(&self) -> Self {
        Self {
            lease: Arc::clone(&self.lease),
        }
    }

    pub fn reconcile_prompt_bytes(&self, desired: usize) -> Result<(), ServerAdmissionError> {
        let mut counts = self.lease.admission.counts.lock().unwrap();
        let current = self.lease.prompt_bytes.load(Ordering::Acquire);
        if desired == current {
            return Ok(());
        }
        if desired > current {
            let additional = desired - current;
            if additional
                > self
                    .lease
                    .admission
                    .prompt_bytes_limit
                    .saturating_sub(counts.held_prompt_bytes)
            {
                return Err(ServerAdmissionError::Capacity {
                    resource: ServerAdmissionResource::PromptBytes,
                    limit: self.lease.admission.prompt_bytes_limit,
                    held: counts.held_prompt_bytes,
                });
            }
            counts.held_prompt_bytes += additional;
        } else {
            counts.held_prompt_bytes -= current - desired;
        }
        self.lease.prompt_bytes.store(desired, Ordering::Release);
        Ok(())
    }
}

impl Drop for ServerAdmissionLease {
    fn drop(&mut self) {
        let prompt_bytes = self.prompt_bytes.load(Ordering::Acquire);
        let mut counts = self.admission.counts.lock().unwrap();
        counts.held_requests -= 1;
        counts.held_prompt_bytes -= prompt_bytes;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownStage {
    WaitingForOwner,
    Cancel,
    Drain,
    PhysicalClose,
    HostJoin,
}

/// Best-known ownership, not a CUDA fence or proof of host termination.
#[derive(Debug, Clone)]
pub struct WorkerShutdownReport {
    pub stage: ShutdownStage,
    /// Logical request IDs last observed by the owner, not an exhaustive native ledger.
    pub remaining_requests: Vec<RequestId>,
    /// None: the engine boundary cannot enumerate physical operations/custody.
    pub remaining_operations: Option<usize>,
    pub custody_unknown: bool,
    /// Custody must be retained. This does not prove the owner observed control.
    pub quarantine: bool,
    pub deadline_exceeded: bool,
    pub first_fatal: Option<Arc<RuntimeWorkerExecutionError>>,
    pub errors: Vec<String>,
}

impl std::fmt::Display for WorkerShutdownReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "worker shutdown incomplete at {:?}: deadline_exceeded={}, remaining_requests={:?}, remaining_operations={:?}, custody_unknown={}, quarantine={}",
            self.stage,
            self.deadline_exceeded,
            self.remaining_requests,
            self.remaining_operations,
            self.custody_unknown,
            self.quarantine
        )
    }
}

impl StdError for WorkerShutdownReport {}

#[derive(Debug, Clone)]
pub struct WorkerSnapshot {
    pub phase: WorkerPhase,
    pub first_fatal: Option<Arc<RuntimeWorkerExecutionError>>,
    pub shutdown_report: WorkerShutdownReport,
}

struct ControlState {
    phase: AtomicU8,
    shutdown: AtomicBool,
    notify: Notify,
    report: Mutex<WorkerShutdownReport>,
    deadline: Mutex<Option<ShutdownDeadline>>,
    host: Mutex<Option<JoinHandle<()>>>,
    unresolved_requests: Mutex<Vec<RequestId>>,
    admission: ServerAdmission,
    terminal_admissions: Mutex<Vec<ServerAdmissionPermit>>,
    shutdown_timeout: std::time::Duration,
}

impl ControlState {
    #[cfg(test)]
    fn new() -> Self {
        Self::with_config(&WorkerConfig::default())
    }

    fn with_config(config: &WorkerConfig) -> Self {
        Self {
            phase: AtomicU8::new(WorkerPhase::Starting as u8),
            shutdown: AtomicBool::new(false),
            notify: Notify::new(),
            report: Mutex::new(WorkerShutdownReport {
                stage: ShutdownStage::WaitingForOwner,
                remaining_requests: Vec::new(),
                remaining_operations: None,
                custody_unknown: true,
                quarantine: false,
                deadline_exceeded: false,
                first_fatal: None,
                errors: Vec::new(),
            }),
            deadline: Mutex::new(None),
            host: Mutex::new(None),
            unresolved_requests: Mutex::new(Vec::new()),
            admission: ServerAdmission::new(config.max_inflight_requests, config.max_prompt_bytes),
            terminal_admissions: Mutex::new(Vec::new()),
            shutdown_timeout: config.shutdown_timeout,
        }
    }

    fn phase(&self) -> WorkerPhase {
        // Read-only status also notices a detached host exit. Neither an outcome
        // message nor a thread exit is treated as proof of physical close.
        let finished = self
            .host
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(JoinHandle::is_finished);
        if finished {
            let mut report = self.report.lock().unwrap();
            if !report.custody_unknown && !report.deadline_exceeded {
                self.phase
                    .store(WorkerPhase::Stopped as u8, Ordering::Release);
            } else if report.custody_unknown {
                report.first_fatal.get_or_insert_with(|| {
                    Arc::new(WorkerExecutionError::Runtime {
                        source: join_error(WorkerShutdownError::ThreadPanicked),
                    })
                });
                report.quarantine = true;
                self.phase
                    .store(WorkerPhase::Fatal as u8, Ordering::Release);
            }
        }
        match self.phase.load(Ordering::Acquire) {
            1 => WorkerPhase::Ready,
            2 => WorkerPhase::Draining,
            3 => WorkerPhase::Fatal,
            4 => WorkerPhase::Stopped,
            _ => WorkerPhase::Starting,
        }
    }

    fn snapshot(&self) -> WorkerSnapshot {
        let phase = self.phase();
        let report = self.report.lock().unwrap();
        WorkerSnapshot {
            phase,
            first_fatal: report.first_fatal.clone(),
            shutdown_report: report.clone(),
        }
    }

    // Exactly one owner control waiter. Other observers only read snapshots.
    fn signal(&self) {
        self.notify.notify_one();
    }

    fn mark_ready(&self) {
        let _ = self.phase.compare_exchange(
            WorkerPhase::Starting as u8,
            WorkerPhase::Ready as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    fn mark_fatal(&self, error: Arc<RuntimeWorkerExecutionError>) {
        self.admission.close();
        let mut report = self.report.lock().unwrap();
        report.first_fatal.get_or_insert(error);
        self.phase
            .store(WorkerPhase::Fatal as u8, Ordering::Release);
    }

    fn begin_shutdown(&self, deadline: ShutdownDeadline) -> ShutdownDeadline {
        self.admission.close();
        let mut stored = self.deadline.lock().unwrap();
        let deadline = *stored.get_or_insert(deadline);
        let _ = self
            .phase
            .try_update(Ordering::AcqRel, Ordering::Acquire, |phase| {
                (phase == WorkerPhase::Ready as u8 || phase == WorkerPhase::Starting as u8)
                    .then_some(WorkerPhase::Draining as u8)
            });
        if !self.shutdown.swap(true, Ordering::AcqRel) {
            self.signal();
        }
        deadline
    }

    fn deadline(&self) -> Option<ShutdownDeadline> {
        *self.deadline.lock().unwrap()
    }

    fn track(&self, stage: ShutdownStage, active: &HashMap<RequestId, ActiveRequest<'_>>) {
        let unresolved = self.unresolved_requests.lock().unwrap();
        let mut report = self.report.lock().unwrap();
        report.stage = stage;
        report.remaining_requests = active
            .keys()
            .copied()
            .chain(unresolved.iter().copied())
            .collect();
        report.remaining_requests.sort_by_key(|id| id.0);
        report.remaining_requests.dedup();
    }

    fn retain_admission(&self, permit: ServerAdmissionPermit) {
        self.terminal_admissions.lock().unwrap().push(permit);
    }

    fn has_admission_cleanup<E: InferenceEngine>(&self, _engine: &E) -> bool {
        let unresolved = self.unresolved_requests.lock().unwrap();
        self.terminal_admissions
            .lock()
            .unwrap()
            .iter()
            .any(|permit| {
                let cleanup = permit.lease.cleanup.lock().unwrap();
                matches!(cleanup.proof, InferenceRequestCleanup::Tracked(_))
                    && !cleanup.released()
                    && !cleanup
                        .request_id
                        .is_some_and(|id| unresolved.contains(&id))
            })
    }

    fn consume_admission_terminal(&self, request_id: RequestId) {
        for permit in self.terminal_admissions.lock().unwrap().iter() {
            let mut cleanup = permit.lease.cleanup.lock().unwrap();
            if cleanup.request_id == Some(request_id) {
                cleanup.terminal_consumed = true;
            }
        }
    }

    fn observe_admission_cleanup<E: InferenceEngine>(&self, _engine: &E) {
        // Each receipt belongs to one accepted generation. No aggregate idle
        // inference, and unrelated active requests cannot block its return.
        self.terminal_admissions
            .lock()
            .unwrap()
            .retain(|permit| !permit.lease.cleanup.lock().unwrap().released());
    }

    fn retain_unresolved(&self, request_id: RequestId) {
        let mut requests = self.unresolved_requests.lock().unwrap();
        if !requests.contains(&request_id) {
            requests.push(request_id);
        }
    }

    fn expired(&self) -> bool {
        self.deadline()
            .is_some_and(|deadline| ShutdownDeadline::now() >= deadline)
    }

    fn incomplete(&self) -> ferrule_runtime::Error {
        let mut report = self.report.lock().unwrap();
        report.deadline_exceeded = true;
        report.quarantine = report.custody_unknown;
        report_error(report.clone())
    }
}

fn report_error(report: WorkerShutdownReport) -> ferrule_runtime::Error {
    ferrule_common::Error::Backend {
        source: Box::new(report),
    }
    .into()
}

fn join_error(error: WorkerShutdownError<tokio::task::JoinError>) -> ferrule_runtime::Error {
    ferrule_common::Error::Backend {
        source: Box::new(error),
    }
    .into()
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
        error: Arc<RuntimeWorkerExecutionError>,
    },
}

struct SubmitCommand {
    request_id: RequestId,
    enqueued_at: Instant,
    request: WorkerRequest,
    permit: ServerAdmissionPermit,
    events: mpsc::Sender<WorkerEvent>,
    cancellation: Arc<AtomicBool>,
    accepted: oneshot::Sender<Result<(), Arc<RuntimeWorkerExecutionError>>>,
}

struct TokenizeCommand {
    prompt: String,
    permit: ServerAdmissionPermit,
    response: oneshot::Sender<Result<Vec<u32>, ferrule_runtime::Error>>,
}

pub(crate) struct RuntimeServingSnapshot {
    pub admission: Option<ferrule_runtime::RuntimeAdmissionSnapshot>,
    // A fixed-schema serialized projection, not a duplicate of runtime state.
    pub capacity: serde_json::Value,
}

enum WorkerCommand {
    Submit(SubmitCommand),
    Tokenize(TokenizeCommand),
    AdmissionSnapshot(oneshot::Sender<RuntimeServingSnapshot>),
}

struct ActiveRequest<'metrics> {
    // Observation follows the admitted generation owner, not the HTTP observer
    // or the admission permit that can outlive terminal delivery for cleanup.
    _metrics: RequestMetricsGuard<'metrics>,
    events: mpsc::Sender<WorkerEvent>,
    permit: ServerAdmissionPermit,
    cancellation: Arc<AtomicBool>,
    cancellation_submitted: bool,
    cancellation_failed: bool,
    terminal_sent: bool,
    requires_restore_or_shutdown: bool,
    session_id: SessionId,
    submitted_at: Instant,
    emitted_tokens: usize,
}

impl ActiveRequest<'_> {
    fn needs_cancellation(&self) -> bool {
        !self.cancellation_submitted
            && !self.cancellation_failed
            && !self.requires_restore_or_shutdown
    }

    fn send_terminal(&mut self, event: WorkerEvent) {
        if !self.terminal_sent {
            self.terminal_sent = true;
            let _ = self.events.try_send(event);
        }
    }
}

#[derive(Clone)]
pub struct ModelWorkerHandle {
    commands: mpsc::Sender<WorkerCommand>,
    control: Arc<ControlState>,
    next_request_id: Arc<AtomicU64>,
    config: WorkerConfig,
}

impl ModelWorkerHandle {
    fn allocate_request_id(&self) -> Result<u64, RuntimeWorkerRequestError> {
        self.next_request_id
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| WorkerRequestError::Unavailable {
                operation: WorkerOperation::Admission,
            })
    }

    pub fn snapshot(&self) -> WorkerSnapshot {
        self.control.snapshot()
    }

    pub fn phase(&self) -> WorkerPhase {
        self.control.phase()
    }

    pub fn admission_snapshot(&self) -> ServerAdmissionSnapshot {
        let mut snapshot = self.control.admission.snapshot();
        snapshot.closed |= self.phase() != WorkerPhase::Ready;
        snapshot
    }

    /// Read runtime-owned semantic counts on its sole owner, without copying a
    /// ledger or treating server leases as runtime identities. No request permit
    /// or generation metrics are acquired. Congestion uses the existing queue
    /// and timeout errors; unsupported engines return `None`, not zero gauges.
    pub async fn runtime_admission_snapshot(
        &self,
    ) -> Result<Option<ferrule_runtime::RuntimeAdmissionSnapshot>, RuntimeWorkerRequestError> {
        Ok(self.runtime_serving_snapshot().await?.admission)
    }

    /// Admission and capacity are sampled within a single owner command; no
    /// stepping, awaiting or mutable engine access occurs between these reads.
    pub(crate) async fn runtime_serving_snapshot(
        &self,
    ) -> Result<RuntimeServingSnapshot, RuntimeWorkerRequestError> {
        if self.phase() != WorkerPhase::Ready {
            return Err(WorkerRequestError::Unavailable {
                operation: WorkerOperation::Admission,
            });
        }
        let (response, receiver) = oneshot::channel();
        self.commands
            .try_send(WorkerCommand::AdmissionSnapshot(response))
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => WorkerRequestError::QueueFull,
                mpsc::error::TrySendError::Closed(_) => WorkerRequestError::Unavailable {
                    operation: WorkerOperation::Admission,
                },
            })?;
        match tokio::time::timeout(self.config.admission_timeout, receiver).await {
            Ok(Ok(snapshot)) => Ok(snapshot),
            Ok(Err(_)) => Err(WorkerRequestError::Unavailable {
                operation: WorkerOperation::Admission,
            }),
            Err(_) => Err(WorkerRequestError::AdmissionTimeout),
        }
    }

    pub(crate) fn max_body_bytes(&self) -> usize {
        self.config.max_body_bytes
    }

    pub(crate) fn try_acquire_admission(
        &self,
        prompt_bytes: usize,
    ) -> Result<ServerAdmissionPermit, ServerAdmissionError> {
        if self.phase() != WorkerPhase::Ready {
            return Err(ServerAdmissionError::Closed);
        }
        self.control.admission.try_acquire(prompt_bytes)
    }

    /// Close admission immediately and publish non-congestible shutdown intent.
    /// Returns the first absolute deadline; repeated calls never extend it.
    pub fn begin_shutdown(&self) -> ShutdownDeadline {
        self.control
            .begin_shutdown(ShutdownDeadline::now() + self.config.shutdown_timeout)
    }

    #[cfg(test)]
    pub(crate) async fn submit(
        &self,
        request: WorkerRequest,
    ) -> Result<EventSubscription, RuntimeWorkerRequestError> {
        let permit = self
            .try_acquire_admission(request.prompt.len())
            .map_err(admission_request_error)?;
        self.submit_with_permit(request, permit).await
    }

    pub(crate) async fn submit_with_permit(
        &self,
        request: WorkerRequest,
        permit: ServerAdmissionPermit,
    ) -> Result<EventSubscription, RuntimeWorkerRequestError> {
        if self.control.phase() != WorkerPhase::Ready {
            return Err(WorkerRequestError::Unavailable {
                operation: WorkerOperation::Admission,
            });
        }
        let request_id = RequestId(self.allocate_request_id()?);
        let enqueued_at = Instant::now();
        let (events, receiver) = mpsc::channel(self.config.event_queue_capacity);
        let (accepted, acceptance) = oneshot::channel();
        let cancellation = Arc::new(AtomicBool::new(false));
        let command = WorkerCommand::Submit(SubmitCommand {
            request_id,
            enqueued_at,
            request,
            permit: permit.observer(),
            events,
            cancellation: Arc::clone(&cancellation),
            accepted,
        });

        match self.commands.try_send(command) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                return Err(WorkerRequestError::QueueFull);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(WorkerRequestError::Unavailable {
                    operation: WorkerOperation::Admission,
                });
            }
        }

        // This guard also cancels if the admission future itself is dropped.
        let subscription = EventSubscription {
            request_id,
            receiver,
            cancellation,
            control: Arc::clone(&self.control),
            terminal_seen: false,
            _permit: permit,
        };
        match tokio::time::timeout(self.config.admission_timeout, acceptance).await {
            Ok(Ok(Ok(()))) => Ok(subscription),
            Ok(Ok(Err(source))) => Err(WorkerRequestError::Rejected { source }),
            Ok(Err(_)) => Err(WorkerRequestError::Unavailable {
                operation: WorkerOperation::Admission,
            }),
            Err(_) => Err(WorkerRequestError::AdmissionTimeout),
        }
    }

    /// Tokenize a prompt on the model worker thread.
    ///
    /// Returns the allocated request id alongside the token ids so the caller
    /// can build a unique response identifier consistent with [`submit`].
    #[cfg(test)]
    pub(crate) async fn tokenize(
        &self,
        prompt: String,
    ) -> Result<(u64, Vec<u32>), RuntimeWorkerRequestError> {
        let permit = self
            .try_acquire_admission(prompt.len())
            .map_err(admission_request_error)?;
        self.tokenize_with_permit(prompt, permit).await
    }

    pub(crate) async fn tokenize_with_permit(
        &self,
        prompt: String,
        permit: ServerAdmissionPermit,
    ) -> Result<(u64, Vec<u32>), RuntimeWorkerRequestError> {
        if self.control.phase() != WorkerPhase::Ready {
            return Err(WorkerRequestError::Unavailable {
                operation: WorkerOperation::Tokenization,
            });
        }
        let request_id = self.allocate_request_id()?;
        let (response, receiver) = oneshot::channel();
        let command = WorkerCommand::Tokenize(TokenizeCommand {
            prompt,
            permit,
            response,
        });

        match self.commands.try_send(command) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                return Err(WorkerRequestError::QueueFull);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(WorkerRequestError::Unavailable {
                    operation: WorkerOperation::Tokenization,
                });
            }
        }

        match tokio::time::timeout(self.config.admission_timeout, receiver).await {
            Ok(Ok(Ok(tokens))) => Ok((request_id, tokens)),
            Ok(Ok(Err(source))) => Err(WorkerRequestError::Tokenization { source }),
            Err(_) => Err(WorkerRequestError::AdmissionTimeout),
            Ok(Err(_)) => Err(WorkerRequestError::Unavailable {
                operation: WorkerOperation::Tokenization,
            }),
        }
    }
}

pub(crate) struct EventSubscription {
    pub request_id: RequestId,
    _permit: ServerAdmissionPermit,
    receiver: mpsc::Receiver<WorkerEvent>,
    cancellation: Arc<AtomicBool>,
    control: Arc<ControlState>,
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
            self.control.signal();
        }
    }
}

pub struct ModelWorker {
    handle: ModelWorkerHandle,
    outcome: oneshot::Receiver<RuntimeResult<()>>,
}

impl ModelWorker {
    pub fn handle(&self) -> ModelWorkerHandle {
        self.handle.clone()
    }

    /// One cooperative deadline; no blocking join task is left in Tokio's pool.
    /// Synchronous engine/native calls cannot be killed by this API.
    pub async fn shutdown(mut self) -> RuntimeResult<()> {
        let deadline = self.handle.begin_shutdown();
        if self.handle.control.host.lock().unwrap().is_none() {
            return Ok(());
        }
        let mut outcome = None;
        loop {
            if outcome.is_none() {
                if let Ok(result) = self.outcome.try_recv() {
                    outcome = Some(result);
                }
            }
            if self.handle.snapshot().shutdown_report.quarantine {
                if let Some(result) = outcome {
                    return result;
                }
            }
            let finished = self
                .handle
                .control
                .host
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(JoinHandle::is_finished);
            if finished {
                let thread = self.handle.control.host.lock().unwrap().take().unwrap();
                if thread.join().is_err() {
                    let error = Arc::new(WorkerExecutionError::Runtime {
                        source: join_error(WorkerShutdownError::ThreadPanicked),
                    });
                    self.handle.control.mark_fatal(Arc::clone(&error));
                    return Err(execution_error(error));
                }
                // The outcome is published before host exit, but neither alone
                // proves device quiescence. Stopped is only published after join.
                let result = outcome.unwrap_or_else(|| {
                    self.outcome
                        .try_recv()
                        .unwrap_or_else(|_| Err(join_error(WorkerShutdownError::ThreadPanicked)))
                });
                if !self.handle.snapshot().shutdown_report.deadline_exceeded {
                    self.handle
                        .control
                        .phase
                        .store(WorkerPhase::Stopped as u8, Ordering::Release);
                }
                return result;
            }
            if ShutdownDeadline::now() >= deadline {
                // The control snapshot retains the host handle; no unbounded join
                // task is spawned and no host/native owner is killed.
                return Err(self.handle.control.incomplete());
            }
            tokio::time::sleep_until(
                deadline.min(ShutdownDeadline::now() + std::time::Duration::from_millis(1)),
            )
            .await;
        }
    }
}

impl Drop for ModelWorker {
    fn drop(&mut self) {
        if self.handle.control.host.lock().unwrap().is_some() {
            self.handle.begin_shutdown();
        }
    }
}

/// Construct and run the model engine on the same dedicated owner thread.
///
/// Only the `Send` factory crosses the thread boundary. The engine it returns may
/// be `!Send`; it is constructed, initialized, shut down, and dropped inside the
/// owner thread.
///
/// A prebuilt `!Send` engine cannot be captured and smuggled through the factory:
///
/// ```compile_fail
/// use std::convert::Infallible;
/// use std::rc::Rc;
/// use ferrule_runtime::InferenceEngine;
/// use ferrule_server::{WorkerConfig, spawn_model_worker_with};
///
/// let prebuilt = Rc::new(());
/// let _ = spawn_model_worker_with(
///     move || -> Result<Box<dyn InferenceEngine>, Infallible> {
///         drop(prebuilt);
///         unreachable!()
///     },
///     WorkerConfig::default(),
/// );
/// ```
pub fn spawn_model_worker_with<F, E, FactorySource>(
    factory: F,
    config: WorkerConfig,
) -> Result<ModelWorker, WorkerStartError<FactorySource, ferrule_runtime::Error>>
where
    F: FnOnce() -> Result<E, FactorySource> + Send + 'static,
    E: InferenceEngine,
    FactorySource: StdError + Send + 'static,
{
    config
        .validate()
        .map_err(|source| WorkerStartError::InvalidConfig { source })?;
    config
        .validate_admission()
        .map_err(|source| WorkerStartError::Startup { source })?;
    let (commands, receiver) = mpsc::channel(config.command_queue_capacity);
    let control = Arc::new(ControlState::with_config(&config));
    let thread_control = Arc::clone(&control);
    let (ready_sender, ready_receiver) = std::sync::mpsc::sync_channel(1);
    let (outcome_sender, outcome) = oneshot::channel();
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
                    let _ =
                        ready_sender.send(Err(WorkerStartError::RuntimeBuild { source: error }));
                    return;
                }
            };
            let local = tokio::task::LocalSet::new();
            let owner_control = Arc::clone(&thread_control);
            let result = local.block_on(&runtime, async move {
                let mut engine = match factory() {
                    Ok(engine) => engine,
                    Err(error) => {
                        let _ = ready_sender
                            .send(Err(WorkerStartError::EngineFactory { source: error }));
                        return Ok(());
                    }
                };
                // Configure the authoritative runtime before background work or Ready.
                // Never silently discard explicit overrides on unsupported engines.
                if let Some(options) = thread_config.runtime_admission_options {
                    if let Err(source) = engine.set_admission_options(options) {
                        let _ = ready_sender.send(Err(WorkerStartError::Startup { source }));
                        return Ok(());
                    }
                }
                let mut completion_owner = InferenceCompletionOwner::attach(&mut engine);
                if let Err(error) = completion_owner.initialize(&mut engine).await {
                    let _ = ready_sender.send(Err(WorkerStartError::Startup { source: error }));
                    return Ok(());
                }
                thread_control.mark_ready();
                let _ = ready_sender.send(Ok(()));
                let result = run_worker(
                    engine,
                    completion_owner,
                    receiver,
                    thread_config,
                    thread_control.clone(),
                )
                .await;
                result
            });
            let quarantine = owner_control.snapshot().shutdown_report.quarantine;
            let _ = outcome_sender.send(result);
            if quarantine {
                loop {
                    std::thread::park();
                }
            }
        })
        .map_err(|source| WorkerStartError::ThreadSpawn { source })?;

    match ready_receiver.recv() {
        Ok(Ok(())) => {
            *control.host.lock().unwrap() = Some(thread);
            Ok(ModelWorker {
                handle: ModelWorkerHandle {
                    commands,
                    control,
                    next_request_id: Arc::new(AtomicU64::new(1)),
                    config,
                },
                outcome,
            })
        }
        Ok(Err(error)) => {
            let _ = thread.join();
            Err(error)
        }
        Err(source) => {
            let _ = thread.join();
            Err(WorkerStartError::InitializationChannel { source })
        }
    }
}

async fn run_worker<E>(
    engine: E,
    completion_owner: InferenceCompletionOwner,
    commands: mpsc::Receiver<WorkerCommand>,
    config: WorkerConfig,
    control: Arc<ControlState>,
) -> RuntimeResult<()>
where
    E: InferenceEngine,
{
    run_worker_with_metrics(
        engine,
        completion_owner,
        commands,
        config,
        control,
        &METRICS,
    )
    .await
}

async fn run_worker_with_metrics<E>(
    mut engine: E,
    mut completion_owner: InferenceCompletionOwner,
    mut commands: mpsc::Receiver<WorkerCommand>,
    config: WorkerConfig,
    control: Arc<ControlState>,
    metrics: &Metrics,
) -> RuntimeResult<()>
where
    E: InferenceEngine,
{
    let mut active = HashMap::<RequestId, ActiveRequest<'_>>::new();
    let mut cancellation_scratch = Vec::<RequestId>::new();
    let mut fatal_error: Option<Arc<RuntimeWorkerExecutionError>> = None;
    let mut failures = Vec::new();

    'worker: loop {
        // Registration, not just future creation, precedes every state check.
        let control_wake = control.notify.notified();
        tokio::pin!(control_wake);
        control_wake.as_mut().enable();
        if control.shutdown.load(Ordering::Acquire) {
            break;
        }
        cancel_disconnected(
            &mut engine,
            &mut active,
            &mut cancellation_scratch,
            &mut failures,
            &control,
        );
        drain_terminal(&mut engine, &mut active, &control);
        control.track(ShutdownStage::WaitingForOwner, &active);
        if control.shutdown.load(Ordering::Acquire) {
            break;
        }
        let should_step = fatal_error.is_none()
            && (!active.is_empty()
                || engine.has_background_work()
                || control.has_admission_cleanup(&engine));
        if !should_step {
            tokio::select! {
                biased;
                _ = &mut control_wake => continue,
                error = completion_owner.reactor_failure(), if fatal_error.is_none() => {
                    let error = Arc::new(WorkerExecutionError::Runtime { source: error });
                    control.mark_fatal(Arc::clone(&error));
                    fatal_error = Some(error);
                    continue;
                }
                command = commands.recv() => {
                    let Some(command) = command else { break; };
                    handle_command_with_metrics(command, &mut engine, &mut active, &control, metrics);
                }
            }
        }
        for _ in 0..config.max_commands_per_tick {
            if control.shutdown.load(Ordering::Acquire) {
                break 'worker;
            }
            match commands.try_recv() {
                Ok(command) => {
                    handle_command_with_metrics(
                        command,
                        &mut engine,
                        &mut active,
                        &control,
                        metrics,
                    );
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => break 'worker,
            }
        }
        cancel_disconnected(
            &mut engine,
            &mut active,
            &mut cancellation_scratch,
            &mut failures,
            &control,
        );
        drain_terminal(&mut engine, &mut active, &control);
        control.track(ShutdownStage::WaitingForOwner, &active);
        if control.shutdown.load(Ordering::Acquire) {
            break;
        }
        if fatal_error.is_some()
            || (active.is_empty()
                && !engine.has_background_work()
                && !control.has_admission_cleanup(&engine))
        {
            continue;
        }

        let completion = completion_owner.listen();
        let step_result = {
            let mut emit = |event: &ResidentTokenEvent| -> RuntimeResult<()> {
                let Some(request_id) = event.request_id else {
                    return Ok(());
                };
                let Some(request) = active.get_mut(&request_id) else {
                    return Ok(());
                };
                // Reserve one channel slot for the unique terminal event.
                if request.cancellation.load(Ordering::Acquire)
                    || request.events.capacity() <= 1
                    || request
                        .events
                        .try_send(WorkerEvent::Token {
                            text: event.text.clone(),
                        })
                        .is_err()
                {
                    request.cancellation.store(true, Ordering::Release);
                    control.signal();
                } else {
                    request.emitted_tokens = request.emitted_tokens.saturating_add(1);
                }
                Ok(())
            };
            engine.step(&mut emit)
        };
        let error = match step_result {
            Ok(ResidentDriverStep::WaitingForModelProgress(_) | ResidentDriverStep::Blocked) => {
                if !engine.has_pending_async_work() {
                    Some(Arc::new(WorkerExecutionError::BlockedWithoutContinuation))
                } else {
                    // A cancellation during step is either visible here or has
                    // notified the already enabled waiter; it needs no command.
                    if control.shutdown.load(Ordering::Acquire) {
                        break;
                    }
                    if active
                        .values()
                        .any(|r| r.cancellation.load(Ordering::Acquire) && r.needs_cancellation())
                    {
                        continue;
                    }
                    tokio::select! {
                        biased;
                        _ = &mut control_wake => continue,
                        wake = completion_owner.wait(completion) => {
                            wake.err().map(|source| Arc::new(WorkerExecutionError::Runtime { source }))
                        }
                        command = commands.recv() => {
                            let Some(command) = command else { break; };
                            handle_command_with_metrics(command, &mut engine, &mut active, &control, metrics);
                            None
                        }
                    }
                }
            }
            Ok(ResidentDriverStep::Idle)
                if active.is_empty() && control.has_admission_cleanup(&engine) =>
            {
                Some(Arc::new(WorkerExecutionError::Runtime {
                    source: ferrule_runtime::Error::ShutdownIncomplete {
                        message: "runtime reported idle with cleanup-owned admission identities"
                            .into(),
                    },
                }))
            }
            Ok(ResidentDriverStep::Idle) if active.is_empty() && engine.has_background_work() => {
                Some(Arc::new(
                    WorkerExecutionError::RunnableBackgroundReportedIdle,
                ))
            }
            Ok(_) => None,
            Err(source) => Some(Arc::new(WorkerExecutionError::Runtime { source })),
        };
        if let Some(error) = error {
            tracing::error!(%error, "model worker entered fatal state");
            control.mark_fatal(Arc::clone(&error));
            fatal_error = Some(Arc::clone(&error));
            fail_all(&mut engine, &mut active, error, &mut failures, &control);
        }
        // Let local completion reactors progress even during synchronous ticks.
        tokio::task::yield_now().await;
    }

    control.begin_shutdown(ShutdownDeadline::now() + config.shutdown_timeout);
    if engine.admission_snapshot().is_some() {
        if let Err(error) = engine.close_admission() {
            failures.push(error);
        }
    }
    commands.close();
    // Dropping queued responses is a stable Unavailable result, without encode.
    while commands.try_recv().is_ok() {}
    if let Some(error) = fatal_error {
        failures.insert(0, execution_error(error));
    }
    failures.extend(cancel_all(&mut engine, &mut completion_owner, &mut active, &control).await);
    if !control.expired() {
        if let Err(error) =
            shutdown_engine(&mut engine, &mut completion_owner, &active, &control).await
        {
            failures.push(error);
        }
    }
    if control.expired() {
        failures.push(control.incomplete());
    }
    let unknown = control.report.lock().unwrap().custody_unknown;
    if unknown {
        for request_id in active.keys().copied().collect::<Vec<_>>() {
            finish_request(
                &mut active,
                request_id,
                RequestOutcome::Quarantined,
                &control,
            );
        }
        control.report.lock().unwrap().quarantine = true;
        if control.snapshot().first_fatal.is_none() {
            if let Some(error) = failures.pop() {
                let error = Arc::new(WorkerExecutionError::Runtime { source: error });
                control.mark_fatal(Arc::clone(&error));
                failures.push(execution_error(error));
            }
        }
        // No Drop/join of unknown native custody. The enclosing owner parks
        // with its LocalSet/runtime retained; a hard SLA needs a process.
        std::mem::forget(engine);
        std::mem::forget(completion_owner);
    } else {
        control.report.lock().unwrap().stage = ShutdownStage::HostJoin;
    }
    control.report.lock().unwrap().errors = failures.iter().map(ToString::to_string).collect();
    match failures.len() {
        0 => Ok(()),
        1 => Err(failures.pop().unwrap()),
        _ => Err(ferrule_runtime::Error::FailureBatch {
            operation: "model worker shutdown",
            failures,
        }),
    }
}

async fn shutdown_engine<E>(
    engine: &mut E,
    completion_owner: &mut InferenceCompletionOwner,
    active: &HashMap<RequestId, ActiveRequest<'_>>,
    control: &ControlState,
) -> RuntimeResult<()>
where
    E: InferenceEngine,
{
    loop {
        let wake = control.notify.notified();
        tokio::pin!(wake);
        wake.as_mut().enable();
        control.track(ShutdownStage::PhysicalClose, active);
        if control.expired() {
            return Err(control.incomplete());
        }
        let completion = completion_owner.listen();
        match engine.shutdown()? {
            InferenceShutdownProgress::Complete => {
                let mut report = control.report.lock().unwrap();
                report.custody_unknown = false;
                report.remaining_operations = Some(0);
                report.remaining_requests.clear();
                drop(report);
                control.unresolved_requests.lock().unwrap().clear();
                control.terminal_admissions.lock().unwrap().clear();
                return Ok(());
            }
            InferenceShutdownProgress::Pending => {
                tokio::select! {
                    biased;
                    _ = tokio::time::sleep_until(control.deadline().unwrap()) => return Err(control.incomplete()),
                    _ = &mut wake => {},
                    result = completion_owner.wait(completion) => result?,
                }
            }
        }
    }
}

#[cfg(test)]
fn handle_command<E>(
    command: WorkerCommand,
    engine: &mut E,
    active: &mut HashMap<RequestId, ActiveRequest<'static>>,
    control: &ControlState,
) -> bool
where
    E: InferenceEngine,
{
    handle_command_with_metrics(command, engine, active, control, &METRICS)
}

fn handle_command_with_metrics<'metrics, E>(
    command: WorkerCommand,
    engine: &mut E,
    active: &mut HashMap<RequestId, ActiveRequest<'metrics>>,
    control: &ControlState,
    metrics: &'metrics Metrics,
) -> bool
where
    E: InferenceEngine,
{
    if control.phase() != WorkerPhase::Ready {
        return false;
    }
    match command {
        WorkerCommand::AdmissionSnapshot(response) => {
            if !response.is_closed() {
                let admission = engine.admission_snapshot();
                let capacity = serde_json::to_value(engine.capacity_snapshot())
                    .expect("capacity snapshot contains only finite integer gauges");
                let _ = response.send(RuntimeServingSnapshot {
                    admission,
                    capacity,
                });
            }
            false
        }
        WorkerCommand::Tokenize(command) => {
            if command.response.is_closed() {
                return false;
            }
            let result = engine.encode(&command.prompt);
            drop(command.prompt);
            command
                .permit
                .reconcile_prompt_bytes(0)
                .expect("shrinking cannot fail");
            let _ = command.response.send(result);
            false
        }
        WorkerCommand::Submit(command) => {
            if command.cancellation.load(Ordering::Acquire) || command.accepted.is_closed() {
                return false;
            }
            let worker_started_at = Instant::now();
            let worker_queue_us = command.enqueued_at.elapsed().as_micros() as u64;
            if active.contains_key(&command.request_id) {
                let _ = command
                    .accepted
                    .send(Err(Arc::new(WorkerExecutionError::Runtime {
                        source: ferrule_runtime::RuntimeAdmissionError::DuplicateRequest {
                            request_id: command.request_id,
                        }
                        .into(),
                    })));
                return false;
            }
            let tokenize_started_at = Instant::now();
            let prompt_tokens = match engine.encode(&command.request.prompt) {
                Ok(tokens) if !tokens.is_empty() => tokens,
                Ok(_) => {
                    let _ = command.accepted.send(Err(Arc::new(
                        WorkerExecutionError::AdmissionEmptyPromptTokens,
                    )));
                    return false;
                }
                Err(error) => {
                    let _ = command.accepted.send(Err(Arc::new(
                        WorkerExecutionError::AdmissionTokenization { source: error },
                    )));
                    return false;
                }
            };
            drop(command.request.prompt);
            command
                .permit
                .reconcile_prompt_bytes(0)
                .expect("shrinking cannot fail");
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
            if control.phase() != WorkerPhase::Ready
                || command.cancellation.load(Ordering::Acquire)
                || command.accepted.is_closed()
            {
                return false;
            }
            if let Err(source) = engine.try_submit(request) {
                let _ = command
                    .accepted
                    .send(Err(Arc::new(WorkerExecutionError::Runtime { source })));
                return false;
            }
            *command.permit.lease.cleanup.lock().unwrap() = AdmissionCleanup {
                request_id: Some(request_id),
                proof: engine.request_cleanup(request_id),
                terminal_consumed: false,
            };
            control
                .report
                .lock()
                .unwrap()
                .remaining_requests
                .push(request_id);
            active.insert(
                request_id,
                ActiveRequest {
                    _metrics: metrics.start_request(),
                    events: command.events,
                    permit: command.permit,
                    cancellation: Arc::clone(&command.cancellation),
                    cancellation_submitted: false,
                    cancellation_failed: false,
                    terminal_sent: false,
                    requires_restore_or_shutdown: false,
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

#[derive(Clone, Copy)]
enum CancellationCause {
    Disconnected,
    Fatal,
    Shutdown,
}

enum RuntimeTerminal {
    Finished,
    Cancelled,
    Failed,
}

enum RequestOutcome {
    Runtime {
        sequence: SequenceState,
        terminal: RuntimeTerminal,
    },
    Failed {
        error: Arc<RuntimeWorkerExecutionError>,
        reason: &'static str,
    },
    // End observation without manufacturing a terminal or cleanup proof.
    Quarantined,
}

fn finish_request(
    active: &mut HashMap<RequestId, ActiveRequest<'_>>,
    request_id: RequestId,
    outcome: RequestOutcome,
    control: &ControlState,
) {
    if matches!(&outcome, RequestOutcome::Runtime { .. }) {
        // A late runtime terminal can satisfy a retained legacy capability even
        // after a server failure removed the active observation owner.
        control.consume_admission_terminal(request_id);
    }
    let Some(mut request) = active.remove(&request_id) else {
        return;
    };
    match outcome {
        RequestOutcome::Runtime { sequence, terminal } => {
            request
                .permit
                .lease
                .cleanup
                .lock()
                .unwrap()
                .terminal_consumed = true;
            let (status, reason, event) = match terminal {
                RuntimeTerminal::Finished => {
                    // An atomic disconnect cannot rewrite runtime arbitration.
                    let reason = sequence
                        .finish_reason
                        .unwrap_or(SequenceFinishReason::NoCandidate);
                    (
                        "finished",
                        reason.as_str(),
                        WorkerEvent::Finished {
                            reason,
                            usage: Usage::new(sequence.prompt_len, sequence.generated),
                        },
                    )
                }
                RuntimeTerminal::Cancelled => (
                    "cancelled",
                    SequenceFinishReason::Cancelled.as_str(),
                    WorkerEvent::Cancelled,
                ),
                RuntimeTerminal::Failed => (
                    "failed",
                    "model_execution_failed",
                    WorkerEvent::Failed {
                        error: Arc::new(WorkerExecutionError::ModelExecution),
                    },
                ),
            };
            trace_worker_request_terminal(
                request_id,
                sequence.session_id,
                status,
                reason,
                &request,
                Some(sequence.generated),
            );
            request.send_terminal(event);
        }
        RequestOutcome::Failed { error, reason } => {
            trace_worker_request_terminal(
                request_id,
                request.session_id,
                "failed",
                reason,
                &request,
                None,
            );
            request.send_terminal(WorkerEvent::Failed { error });
        }
        RequestOutcome::Quarantined => {}
    }
    // Dropping the active owner's metrics guard ends observation exactly once.
    // Neither notification nor owner removal releases the cleanup-owned lease.
    control.retain_admission(request.permit);
}

fn request_cancellation<E: InferenceEngine>(
    engine: &mut E,
    active: &mut HashMap<RequestId, ActiveRequest<'_>>,
    request_id: RequestId,
    cause: CancellationCause,
    failures: &mut Vec<ferrule_runtime::Error>,
    control: &ControlState,
) {
    let Some(request) = active.get_mut(&request_id) else {
        return;
    };
    if !request.needs_cancellation() || control.expired() {
        return;
    }
    if matches!(cause, CancellationCause::Shutdown) {
        request.cancellation.store(true, Ordering::Release);
    }
    match engine.cancel_request(request_id) {
        Ok(InferenceCancelProgress::Complete(_) | InferenceCancelProgress::Pending) => {
            request.cancellation_submitted = true;
        }
        Ok(InferenceCancelProgress::RequiresRestoreOrShutdown { session_id, .. }) => {
            control.retain_unresolved(request_id);
            request.requires_restore_or_shutdown = true;
            // Fatal retains its original error; suspended cancellation otherwise
            // notifies once without ending active observation or accepting cancel.
            if !matches!(cause, CancellationCause::Fatal) {
                request.send_terminal(WorkerEvent::Failed {
                    error: Arc::new(WorkerExecutionError::Cancellation {
                        source: server_source_error(SuspendedCancellation {
                            request_id,
                            session_id,
                        }),
                    }),
                });
            }
            control.begin_shutdown(ShutdownDeadline::now() + control.shutdown_timeout);
        }
        Err(source) => {
            control.retain_unresolved(request_id);
            let (error, reason) = match cause {
                CancellationCause::Disconnected => (
                    WorkerExecutionError::Cancellation { source },
                    "cancellation_failed",
                ),
                CancellationCause::Shutdown => (
                    WorkerExecutionError::ShutdownCancellation { source },
                    "shutdown_cancellation_failed",
                ),
                CancellationCause::Fatal => {
                    // The fatal caller reports its primary error, preserving the
                    // raw cancellation failure as a separate shutdown failure.
                    failures.push(source);
                    return;
                }
            };
            let error = Arc::new(error);
            failures.push(execution_error(Arc::clone(&error)));
            if matches!(cause, CancellationCause::Disconnected) {
                // Failure is not cancellation acceptance or cleanup proof. Keep
                // observation alive while shutdown steps the retained runtime
                // owner, without submitting the failed cancellation again.
                request.cancellation_failed = true;
                request.send_terminal(WorkerEvent::Failed { error });
                control.begin_shutdown(ShutdownDeadline::now() + control.shutdown_timeout);
            } else {
                finish_request(
                    active,
                    request_id,
                    RequestOutcome::Failed { error, reason },
                    control,
                );
            }
        }
    }
}

fn cancel_disconnected<E>(
    engine: &mut E,
    active: &mut HashMap<RequestId, ActiveRequest<'_>>,
    scratch: &mut Vec<RequestId>,
    failures: &mut Vec<ferrule_runtime::Error>,
    control: &ControlState,
) where
    E: InferenceEngine,
{
    scratch.clear();
    scratch.extend(active.iter().filter_map(|(request_id, request)| {
        (request.cancellation.load(Ordering::Acquire) && request.needs_cancellation())
            .then_some(*request_id)
    }));
    for request_id in scratch.iter().copied() {
        if control.expired() {
            break;
        }
        request_cancellation(
            engine,
            active,
            request_id,
            CancellationCause::Disconnected,
            failures,
            control,
        );
    }
}

fn drain_terminal<E>(
    engine: &mut E,
    active: &mut HashMap<RequestId, ActiveRequest<'_>>,
    control: &ControlState,
) where
    E: InferenceEngine,
{
    for sequence in engine.drain_finished() {
        let Some(request_id) = sequence.request_id else {
            continue;
        };
        finish_request(
            active,
            request_id,
            RequestOutcome::Runtime {
                sequence,
                terminal: RuntimeTerminal::Finished,
            },
            control,
        );
    }
    for sequence in engine.drain_cancelled() {
        let Some(request_id) = sequence.request_id else {
            continue;
        };
        finish_request(
            active,
            request_id,
            RequestOutcome::Runtime {
                sequence,
                terminal: RuntimeTerminal::Cancelled,
            },
            control,
        );
    }
    for sequence in engine.drain_failed() {
        let Some(request_id) = sequence.request_id else {
            continue;
        };
        finish_request(
            active,
            request_id,
            RequestOutcome::Runtime {
                sequence,
                terminal: RuntimeTerminal::Failed,
            },
            control,
        );
    }
    control.observe_admission_cleanup(engine);
}

fn trace_worker_request_terminal(
    request_id: RequestId,
    session_id: SessionId,
    status: &'static str,
    finish_reason: &'static str,
    request: &ActiveRequest<'_>,
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

fn fail_all<E>(
    engine: &mut E,
    active: &mut HashMap<RequestId, ActiveRequest<'_>>,
    error: Arc<RuntimeWorkerExecutionError>,
    failures: &mut Vec<ferrule_runtime::Error>,
    control: &ControlState,
) where
    E: InferenceEngine,
{
    let request_ids = active.keys().copied().collect::<Vec<_>>();
    for request_id in request_ids {
        control.retain_unresolved(request_id);
        if control.expired() {
            break;
        }
        request_cancellation(
            engine,
            active,
            request_id,
            CancellationCause::Fatal,
            failures,
            control,
        );
        finish_request(
            active,
            request_id,
            RequestOutcome::Failed {
                error: Arc::clone(&error),
                reason: "fatal_engine_error",
            },
            control,
        );
    }
    drain_terminal(engine, active, control);
}

async fn cancel_all<E>(
    engine: &mut E,
    completion_owner: &mut InferenceCompletionOwner,
    active: &mut HashMap<RequestId, ActiveRequest<'_>>,
    control: &ControlState,
) -> Vec<ferrule_runtime::Error>
where
    E: InferenceEngine,
{
    let mut failures = Vec::new();
    let request_ids = active
        .iter()
        .filter_map(|(request_id, request)| request.needs_cancellation().then_some(*request_id))
        .collect::<Vec<_>>();
    control.track(ShutdownStage::Cancel, active);
    for request_id in request_ids {
        if control.expired() {
            return failures;
        }
        request_cancellation(
            engine,
            active,
            request_id,
            CancellationCause::Shutdown,
            &mut failures,
            control,
        );
    }
    drain_terminal(engine, active, control);

    while active
        .values()
        .any(|request| !request.requires_restore_or_shutdown)
    {
        let wake = control.notify.notified();
        tokio::pin!(wake);
        wake.as_mut().enable();
        control.track(ShutdownStage::Drain, active);
        if control.expired() {
            return failures;
        }
        let completion = completion_owner.listen();
        let mut discard_token = |_event: &ResidentTokenEvent| Ok(());
        let step = engine.step(&mut discard_token);
        drain_terminal(engine, active, control);
        if active.is_empty() {
            if let Err(error) = step {
                failures.push(error);
            }
            break;
        }
        match step {
            Ok(ResidentDriverStep::Executed { .. }) => continue,
            Ok(ResidentDriverStep::WaitingForModelProgress(_) | ResidentDriverStep::Blocked) => {
                if !engine.has_pending_async_work() {
                    failures.push(execution_error(Arc::new(
                        WorkerExecutionError::BlockedWithoutContinuation,
                    )));
                    break;
                }
                let result = tokio::select! {
                    biased;
                    _ = tokio::time::sleep_until(control.deadline().unwrap()) => return failures,
                    _ = &mut wake => Ok(()),
                    result = completion_owner.wait(completion) => result,
                };
                if let Err(error) = result {
                    failures.push(error);
                    break;
                }
            }
            Ok(ResidentDriverStep::Idle) => {
                failures.push(ferrule_runtime::Error::ShutdownIncomplete {
                    message: "engine became idle while cancellation retained requests".into(),
                });
                break;
            }
            Err(error) => {
                failures.push(error);
                break;
            }
        }
    }

    let terminal_error = (!failures.is_empty()).then(|| {
        Arc::new(WorkerExecutionError::Runtime {
            source: ferrule_runtime::Error::FailureBatch {
                operation: "shutdown cancellation",
                failures: std::mem::take(&mut failures),
            },
        })
    });
    for request_id in active.keys().copied().collect::<Vec<_>>() {
        control.retain_unresolved(request_id);
        let error = terminal_error.clone().unwrap_or_else(|| {
            Arc::new(WorkerExecutionError::Runtime {
                source: ferrule_runtime::Error::EngineUnavailable,
            })
        });
        finish_request(
            active,
            request_id,
            RequestOutcome::Failed {
                error,
                reason: "worker_shutdown",
            },
            control,
        );
    }
    if let Some(error) = terminal_error {
        failures.push(execution_error(error));
    }
    failures
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_common::CompletionHub;
    use ferrule_runtime::{
        CancelRequestResult, InferenceCompletionReactor, Result as RuntimeResult, SequenceState,
    };
    use std::sync::atomic::AtomicUsize;

    fn test_permit() -> ServerAdmissionPermit {
        ServerAdmission::new(1024, 16 * 1024 * 1024)
            .try_acquire(0)
            .unwrap()
    }

    fn ready_control() -> ControlState {
        let control = ControlState::new();
        control.mark_ready();
        control
    }

    #[derive(Default)]
    struct WakeProbe(AtomicBool);

    impl std::task::Wake for WakeProbe {
        fn wake(self: Arc<Self>) {
            self.0.store(true, Ordering::Release);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.store(true, Ordering::Release);
        }
    }

    #[tokio::test]
    async fn drop_only_cancellation_wakes_blocked_owner() {
        let mut engine = DisconnectEngine {
            fail_cancel: true,
            ..Default::default()
        };
        let cancellations = Arc::clone(&engine.cancellation_count);
        let completion_owner = InferenceCompletionOwner::attach(&mut engine);
        let (commands, receiver) = mpsc::channel(1);
        let (events, event_receiver) = mpsc::channel(4);
        let (accepted, _acceptance) = oneshot::channel();
        let cancellation = Arc::new(AtomicBool::new(false));
        commands
            .try_send(WorkerCommand::Submit(SubmitCommand {
                permit: test_permit(),
                request_id: RequestId(1),
                enqueued_at: Instant::now(),
                request: WorkerRequest {
                    prompt: "hello".into(),
                    max_tokens: 8,
                    stop: vec![],
                    ignore_eos: false,
                },
                events,
                cancellation: Arc::clone(&cancellation),
                accepted,
            }))
            .unwrap();
        let control = Arc::new(ControlState::new());
        control.mark_ready();
        let subscription = EventSubscription {
            _permit: test_permit(),
            request_id: RequestId(1),
            receiver: event_receiver,
            cancellation,
            control: Arc::clone(&control),
            terminal_seen: false,
        };
        let run = run_worker(
            engine,
            completion_owner,
            receiver,
            WorkerConfig::default(),
            control,
        );
        tokio::pin!(run);
        let probe = Arc::new(WakeProbe::default());
        let waker = std::task::Waker::from(Arc::clone(&probe));
        let mut cx = std::task::Context::from_waker(&waker);
        assert!(std::future::Future::poll(run.as_mut(), &mut cx).is_pending());
        probe.0.store(false, Ordering::Release);
        drop(subscription);
        assert!(
            probe.0.load(Ordering::Acquire),
            "drop-only cancellation did not wake the registered worker"
        );
        assert!(std::future::Future::poll(run.as_mut(), &mut cx).is_pending());
        assert_eq!(cancellations.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn full_command_queue_does_not_block_shutdown_signal() {
        let (commands, _receiver) = mpsc::channel(1);
        let (response, _received) = oneshot::channel();
        commands
            .try_send(WorkerCommand::Tokenize(TokenizeCommand {
                permit: test_permit(),
                prompt: "queued".into(),
                response,
            }))
            .unwrap();
        let worker = ModelWorker {
            handle: ModelWorkerHandle {
                commands,
                control: Arc::new(ControlState::new()),
                next_request_id: Arc::new(AtomicU64::new(1)),
                config: WorkerConfig::default(),
            },
            outcome: oneshot::channel().1,
        };
        let shutdown = worker.shutdown();
        tokio::pin!(shutdown);
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        assert!(
            std::future::Future::poll(shutdown.as_mut(), &mut cx).is_ready(),
            "shutdown blocked behind the full data queue before checking owner completion"
        );
    }

    fn poll_once<F: std::future::Future>(
        future: std::pin::Pin<&mut F>,
    ) -> std::task::Poll<F::Output> {
        future.poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
    }

    fn request() -> WorkerRequest {
        WorkerRequest {
            prompt: "hello".into(),
            max_tokens: 8,
            stop: Vec::new(),
            ignore_eos: false,
        }
    }

    #[derive(Default)]
    struct LifecycleProbe {
        encodes: AtomicUsize,
        steps: AtomicUsize,
        closes: AtomicUsize,
        drops: AtomicUsize,
        fatal: AtomicBool,
        retired: AtomicBool,
        closed: AtomicBool,
    }

    struct LifecycleEngine {
        inner: DisconnectEngine,
        probe: Arc<LifecycleProbe>,
    }

    impl Drop for LifecycleEngine {
        fn drop(&mut self) {
            self.probe.drops.fetch_add(1, Ordering::AcqRel);
        }
    }

    impl InferenceEngine for LifecycleEngine {
        fn completion_hub(&self) -> CompletionHub {
            self.inner.completion_hub()
        }
        fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
            Vec::new()
        }
        fn has_pending_async_work(&self) -> bool {
            true
        }
        fn encode(&self, prompt: &str) -> RuntimeResult<Vec<u32>> {
            self.probe.encodes.fetch_add(1, Ordering::AcqRel);
            self.inner.encode(prompt)
        }
        fn submit(&mut self, request: GenerateRequest) {
            self.inner.submit(request);
        }
        fn step(
            &mut self,
            emit: &mut dyn FnMut(&ResidentTokenEvent) -> RuntimeResult<()>,
        ) -> RuntimeResult<ResidentDriverStep> {
            self.probe.steps.fetch_add(1, Ordering::AcqRel);
            if self.probe.fatal.load(Ordering::Acquire) {
                return Err(ferrule_runtime::Error::EngineUnavailable);
            }
            if self.inner.cancellation_pending && self.probe.retired.load(Ordering::Acquire) {
                self.inner.step(emit)
            } else {
                Ok(ResidentDriverStep::Blocked)
            }
        }
        fn cancel_request(&mut self, _id: RequestId) -> RuntimeResult<InferenceCancelProgress> {
            self.inner.cancellation_count.fetch_add(1, Ordering::AcqRel);
            self.inner.cancellation_pending = true;
            Ok(InferenceCancelProgress::Pending)
        }
        fn shutdown(&mut self) -> RuntimeResult<InferenceShutdownProgress> {
            self.probe.closes.fetch_add(1, Ordering::AcqRel);
            Ok(if self.probe.closed.load(Ordering::Acquire) {
                InferenceShutdownProgress::Complete
            } else {
                InferenceShutdownProgress::Pending
            })
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
    }

    fn local_fixture() -> (
        ModelWorkerHandle,
        mpsc::Receiver<WorkerCommand>,
        LifecycleEngine,
    ) {
        let (commands, receiver) = mpsc::channel(1);
        let handle = ModelWorkerHandle {
            commands,
            control: Arc::new(ready_control()),
            next_request_id: Arc::new(AtomicU64::new(1)),
            config: WorkerConfig::default(),
        };
        (
            handle,
            receiver,
            LifecycleEngine {
                inner: DisconnectEngine::default(),
                probe: Arc::new(LifecycleProbe::default()),
            },
        )
    }

    fn admit_local<F: std::future::Future<Output = RuntimeResult<()>>>(
        handle: &ModelWorkerHandle,
        mut run: std::pin::Pin<&mut F>,
    ) -> EventSubscription {
        let submit = handle.submit(request());
        tokio::pin!(submit);
        assert!(poll_once(submit.as_mut()).is_pending());
        assert!(poll_once(run.as_mut()).is_pending());
        match poll_once(submit.as_mut()) {
            std::task::Poll::Ready(Ok(events)) => events,
            _ => panic!("local admission must have been acknowledged"),
        }
    }

    #[tokio::test]
    async fn control_notify_before_enable_after_check_and_select_competition() {
        for placement in 0..3 {
            let control = ready_control();
            let flag = AtomicBool::new(false);
            if placement == 0 {
                flag.store(true, Ordering::Release);
                control.signal();
            }
            let wake = control.notify.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            if placement == 1 {
                flag.store(true, Ordering::Release);
                control.signal();
            }
            let observed = flag.load(Ordering::Acquire);
            if placement == 2 {
                flag.store(true, Ordering::Release);
                control.signal();
            }
            assert!(observed || poll_once(wake.as_mut()).is_ready());
            assert!(flag.load(Ordering::Acquire));
        }
        let control = ready_control();
        let wake = control.notify.notified();
        tokio::pin!(wake);
        wake.as_mut().enable();
        control.begin_shutdown(ShutdownDeadline::now() + std::time::Duration::from_secs(30));
        tokio::select! {
            biased;
            _ = std::future::ready(()) => {},
            _ = &mut wake => panic!("competing branch must win"),
        }
        // A consumed/dropped wake is not authority; next registration then flag
        // check still sees shutdown, even if the competing branch won.
        let next = control.notify.notified();
        tokio::pin!(next);
        next.as_mut().enable();
        assert!(control.shutdown.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn full_queue_drop_and_shutdown_cancel_once_without_encoding_queued_command() {
        let (handle, receiver, mut engine) = local_fixture();
        let cancellations = Arc::clone(&engine.inner.cancellation_count);
        let probe = Arc::clone(&engine.probe);
        probe.retired.store(true, Ordering::Release);
        probe.closed.store(true, Ordering::Release);
        let owner = InferenceCompletionOwner::attach(&mut engine);
        let run = run_worker(
            engine,
            owner,
            receiver,
            handle.config.clone(),
            Arc::clone(&handle.control),
        );
        tokio::pin!(run);
        let events = admit_local(&handle, run.as_mut());
        let queued = handle.tokenize("must not encode".into());
        tokio::pin!(queued);
        assert!(poll_once(queued.as_mut()).is_pending());
        assert_eq!(handle.commands.capacity(), 0);
        drop(events);
        handle.begin_shutdown();
        assert!(matches!(
            poll_once(run.as_mut()),
            std::task::Poll::Ready(Ok(()))
        ));
        assert_eq!(cancellations.load(Ordering::Acquire), 1);
        assert_eq!(probe.encodes.load(Ordering::Acquire), 1);
        assert!(matches!(
            poll_once(queued.as_mut()),
            std::task::Poll::Ready(Err(WorkerRequestError::Unavailable { .. }))
        ));
    }

    #[tokio::test]
    async fn command_only_wait_and_shutdown_before_first_poll_are_controlled() {
        for before_poll in [false, true] {
            let (handle, receiver, mut engine) = local_fixture();
            let probe = Arc::clone(&engine.probe);
            probe.closed.store(true, Ordering::Release);
            let owner = InferenceCompletionOwner::attach(&mut engine);
            let run = run_worker(
                engine,
                owner,
                receiver,
                handle.config.clone(),
                Arc::clone(&handle.control),
            );
            tokio::pin!(run);
            if !before_poll {
                assert!(poll_once(run.as_mut()).is_pending());
            }
            handle.begin_shutdown();
            assert!(matches!(
                poll_once(run.as_mut()),
                std::task::Poll::Ready(Ok(()))
            ));
            assert_eq!(probe.closes.load(Ordering::Acquire), 1);
        }
    }

    #[tokio::test]
    async fn fatal_rejects_submit_and_tokenize_including_already_queued_commands() {
        let (handle, receiver, mut engine) = local_fixture();
        let probe = Arc::clone(&engine.probe);
        probe.fatal.store(true, Ordering::Release);
        probe.closed.store(true, Ordering::Release);
        let owner = InferenceCompletionOwner::attach(&mut engine);
        let run = run_worker(
            engine,
            owner,
            receiver,
            handle.config.clone(),
            Arc::clone(&handle.control),
        );
        tokio::pin!(run);
        let mut events = admit_local(&handle, run.as_mut());
        assert_eq!(handle.phase(), WorkerPhase::Fatal);
        let first = handle.snapshot().first_fatal.unwrap();
        handle
            .control
            .mark_fatal(Arc::new(WorkerExecutionError::ModelExecution));
        assert!(Arc::ptr_eq(&first, &handle.snapshot().first_fatal.unwrap()));
        assert!(matches!(
            handle.submit(request()).await,
            Err(WorkerRequestError::Unavailable { .. })
        ));
        assert!(matches!(
            handle.tokenize("rejected".into()).await,
            Err(WorkerRequestError::Unavailable { .. })
        ));
        let (response, mut received) = oneshot::channel();
        handle
            .commands
            .try_send(WorkerCommand::Tokenize(TokenizeCommand {
                permit: test_permit(),
                prompt: "raced".into(),
                response,
            }))
            .unwrap();
        assert!(poll_once(run.as_mut()).is_pending());
        assert!(matches!(
            received.try_recv(),
            Err(oneshot::error::TryRecvError::Closed)
        ));
        assert_eq!(probe.encodes.load(Ordering::Acquire), 1);
        assert!(matches!(
            events.recv().await,
            Some(WorkerEvent::Failed { .. })
        ));
        assert!(events.recv().await.is_none());
        handle.begin_shutdown();
        assert!(matches!(
            poll_once(run.as_mut()),
            std::task::Poll::Ready(Err(_))
        ));
        assert!(Arc::ptr_eq(&first, &handle.snapshot().first_fatal.unwrap()));
    }

    #[tokio::test(start_paused = true)]
    async fn absolute_deadline_covers_drain_then_physical_close_without_reset() {
        let (handle, receiver, mut engine) = local_fixture();
        let probe = Arc::clone(&engine.probe);
        let hub = engine.completion_hub();
        let cancellations = Arc::clone(&engine.inner.cancellation_count);
        let owner = InferenceCompletionOwner::attach(&mut engine);
        let run = run_worker(
            engine,
            owner,
            receiver,
            handle.config.clone(),
            Arc::clone(&handle.control),
        );
        tokio::pin!(run);
        let mut events = admit_local(&handle, run.as_mut());
        let deadline = handle.begin_shutdown();
        assert!(poll_once(run.as_mut()).is_pending());
        assert_eq!(
            handle.snapshot().shutdown_report.stage,
            ShutdownStage::Drain
        );
        assert_eq!(cancellations.load(Ordering::Acquire), 1);
        let steps = probe.steps.load(Ordering::Acquire);
        assert!(poll_once(run.as_mut()).is_pending());
        assert_eq!(
            probe.steps.load(Ordering::Acquire),
            steps,
            "observed shutdown intent must not busy-spin"
        );
        tokio::time::advance(std::time::Duration::from_secs(20)).await;
        assert_eq!(handle.begin_shutdown(), deadline);
        probe.retired.store(true, Ordering::Release);
        hub.notify();
        assert!(poll_once(run.as_mut()).is_pending());
        assert_eq!(
            handle.snapshot().shutdown_report.stage,
            ShutdownStage::PhysicalClose
        );
        assert!(matches!(events.recv().await, Some(WorkerEvent::Cancelled)));
        assert!(events.recv().await.is_none());
        let closes = probe.closes.load(Ordering::Acquire);
        assert!(poll_once(run.as_mut()).is_pending());
        assert_eq!(probe.closes.load(Ordering::Acquire), closes);
        tokio::time::advance(std::time::Duration::from_secs(10)).await;
        assert!(matches!(
            poll_once(run.as_mut()),
            std::task::Poll::Ready(Err(_))
        ));
        let report = handle.snapshot().shutdown_report;
        assert_eq!(report.stage, ShutdownStage::PhysicalClose);
        assert!(report.deadline_exceeded && report.custody_unknown && report.quarantine);
        assert_eq!(report.remaining_operations, None);
        assert_eq!(
            probe.drops.load(Ordering::Acquire),
            0,
            "unknown native ownership must not be dropped"
        );
        assert_ne!(handle.phase(), WorkerPhase::Stopped);
    }

    #[tokio::test(start_paused = true)]
    async fn missing_completion_deadline_reports_remaining_request_and_no_false_terminal() {
        let (handle, receiver, mut engine) = local_fixture();
        let probe = Arc::clone(&engine.probe);
        let owner = InferenceCompletionOwner::attach(&mut engine);
        let run = run_worker(
            engine,
            owner,
            receiver,
            handle.config.clone(),
            Arc::clone(&handle.control),
        );
        tokio::pin!(run);
        let mut events = admit_local(&handle, run.as_mut());
        handle.begin_shutdown();
        assert!(poll_once(run.as_mut()).is_pending());
        tokio::time::advance(std::time::Duration::from_secs(30)).await;
        assert!(matches!(
            poll_once(run.as_mut()),
            std::task::Poll::Ready(Err(_))
        ));
        let report = handle.snapshot().shutdown_report;
        assert_eq!(report.stage, ShutdownStage::Drain);
        assert_eq!(report.remaining_requests, vec![events.request_id]);
        assert!(report.quarantine && report.deadline_exceeded);
        assert_eq!(probe.closes.load(Ordering::Acquire), 0);
        assert_eq!(probe.drops.load(Ordering::Acquire), 0);
        assert!(
            events.recv().await.is_none(),
            "deadline is not a successful cancellation terminal"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn zero_budget_skips_engine_calls_and_reports_cancel_stage() {
        let (handle, receiver, mut engine) = local_fixture();
        let probe = Arc::clone(&engine.probe);
        let owner = InferenceCompletionOwner::attach(&mut engine);
        handle.control.begin_shutdown(ShutdownDeadline::now());
        let result = run_worker(
            engine,
            owner,
            receiver,
            handle.config.clone(),
            Arc::clone(&handle.control),
        )
        .await;
        assert!(result.is_err());
        let snapshot = handle.snapshot();
        assert_eq!(snapshot.shutdown_report.stage, ShutdownStage::Cancel);
        assert!(snapshot.shutdown_report.quarantine && snapshot.shutdown_report.deadline_exceeded);
        assert_eq!(probe.closes.load(Ordering::Acquire), 0);
        assert_eq!(probe.drops.load(Ordering::Acquire), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn host_join_deadline_is_typed_and_does_not_kill_a_blocked_host() {
        let (handle, _receiver, _engine) = local_fixture();
        let (release, blocked) = std::sync::mpsc::channel();
        let (outcome_sender, outcome) = oneshot::channel();
        let (exited, mut exit) = oneshot::channel();
        let thread = std::thread::spawn(move || {
            blocked.recv().unwrap();
            let _ = outcome_sender.send(Ok(()));
            let _ = exited.send(());
        });
        handle.control.report.lock().unwrap().stage = ShutdownStage::HostJoin;
        // Native close has completed; only host ownership remains. Host deadline
        // expiry must not re-invent unknown GPU operations or claim a kill.
        handle.control.report.lock().unwrap().custody_unknown = false;
        handle.control.report.lock().unwrap().remaining_operations = Some(0);
        handle.control.begin_shutdown(ShutdownDeadline::now());
        *handle.control.host.lock().unwrap() = Some(thread);
        let worker = ModelWorker {
            handle: handle.clone(),
            outcome,
        };
        let error = worker.shutdown().await.unwrap_err();
        let ferrule_runtime::Error::Backend {
            source: ferrule_common::Error::Backend { source },
        } = error
        else {
            panic!("deadline report lost typed source")
        };
        let report = source.downcast_ref::<WorkerShutdownReport>().unwrap();
        assert_eq!(report.stage, ShutdownStage::HostJoin);
        assert!(report.deadline_exceeded);
        assert!(!report.custody_unknown && !report.quarantine);
        assert_eq!(handle.phase(), WorkerPhase::Draining);
        assert!(matches!(
            exit.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        release.send(()).unwrap();
        exit.await.unwrap();
    }

    #[tokio::test]
    async fn terminal_drop_does_not_cancel_or_wake_again() {
        let control = Arc::new(ready_control());
        let cancellation = Arc::new(AtomicBool::new(false));
        let (events, receiver) = mpsc::channel(2);
        let mut subscription = EventSubscription {
            _permit: test_permit(),
            request_id: RequestId(1),
            receiver,
            cancellation: Arc::clone(&cancellation),
            control: Arc::clone(&control),
            terminal_seen: false,
        };
        let wake = control.notify.notified();
        tokio::pin!(wake);
        wake.as_mut().enable();
        events
            .try_send(WorkerEvent::Finished {
                reason: SequenceFinishReason::MaxTokens,
                usage: Usage::new(1, 1),
            })
            .unwrap();
        assert!(matches!(
            subscription.recv().await,
            Some(WorkerEvent::Finished { .. })
        ));
        drop(subscription);
        assert!(!cancellation.load(Ordering::Acquire));
        assert!(poll_once(wake.as_mut()).is_pending());
    }

    #[tokio::test]
    async fn all_http_phases_keep_liveness_but_gate_readiness_and_business_posts() {
        use axum::{
            body::Body,
            http::{Request, StatusCode},
        };
        use http_body_util::BodyExt;
        use tower::ServiceExt;
        let (handle, _receiver, _engine) = local_fixture();
        let app = crate::router(crate::ServerState::new(
            crate::ModelRegistration::new("test-model", ferrule_model::ChatTemplate::Plain),
            handle.clone(),
        ));
        for phase in [
            WorkerPhase::Starting,
            WorkerPhase::Ready,
            WorkerPhase::Draining,
            WorkerPhase::Fatal,
            WorkerPhase::Stopped,
        ] {
            handle.control.phase.store(phase as u8, Ordering::Release);
            for uri in ["/health", "/health/live", "/readyz"] {
                let response = app
                    .clone()
                    .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    if uri == "/readyz" && phase != WorkerPhase::Ready {
                        StatusCode::SERVICE_UNAVAILABLE
                    } else {
                        StatusCode::OK
                    }
                );
                if uri != "/readyz" {
                    assert_eq!(
                        response.into_body().collect().await.unwrap().to_bytes(),
                        r#"{"status":"ok"}"#
                    );
                }
            }
            if phase != WorkerPhase::Ready {
                for uri in [
                    "/tokenize",
                    "/v1/tokenize",
                    "/v1/completions",
                    "/v1/chat/completions",
                ] {
                    let response = app
                        .clone()
                        .oneshot(
                            Request::builder()
                                .method("POST")
                                .uri(uri)
                                .header("content-type", "application/json")
                                .body(Body::from("{}"))
                                .unwrap(),
                        )
                        .await
                        .unwrap();
                    assert_eq!(
                        response.status(),
                        StatusCode::SERVICE_UNAVAILABLE,
                        "{uri}: {phase:?}"
                    );
                }
                assert!(matches!(
                    handle.submit(request()).await,
                    Err(WorkerRequestError::Unavailable { .. })
                ));
                assert!(matches!(
                    handle.tokenize("hello".into()).await,
                    Err(WorkerRequestError::Unavailable { .. })
                ));
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn abandoned_admission_sets_cancel_and_wakes_without_an_extra_command() {
        let (handle, mut receiver, _engine) = local_fixture();
        let mut admission = Box::pin(handle.submit(request()));
        assert!(poll_once(admission.as_mut()).is_pending());
        let WorkerCommand::Submit(command) = receiver.try_recv().unwrap() else {
            panic!("submit")
        };
        let wake = handle.control.notify.notified();
        tokio::pin!(wake);
        wake.as_mut().enable();
        drop(admission);
        assert!(command.cancellation.load(Ordering::Acquire));
        assert!(poll_once(wake.as_mut()).is_ready());
        assert!(command.accepted.is_closed());
    }

    #[tokio::test]
    async fn full_queue_drop_has_its_own_wake_without_shutdown_or_tokenize() {
        let (handle, receiver, mut engine) = local_fixture();
        let probe = Arc::clone(&engine.probe);
        let cancellations = Arc::clone(&engine.inner.cancellation_count);
        let owner = InferenceCompletionOwner::attach(&mut engine);
        let run = run_worker(
            engine,
            owner,
            receiver,
            handle.config.clone(),
            Arc::clone(&handle.control),
        );
        tokio::pin!(run);
        let events = admit_local(&handle, run.as_mut());
        let wake_probe = Arc::new(WakeProbe::default());
        let waker = std::task::Waker::from(Arc::clone(&wake_probe));
        let mut cx = std::task::Context::from_waker(&waker);
        assert!(std::future::Future::poll(run.as_mut(), &mut cx).is_pending());
        let (accepted, acceptance) = oneshot::channel();
        drop(acceptance);
        handle
            .commands
            .try_send(WorkerCommand::Submit(SubmitCommand {
                permit: test_permit(),
                request_id: RequestId(99),
                enqueued_at: Instant::now(),
                request: request(),
                events: mpsc::channel(2).0,
                cancellation: Arc::new(AtomicBool::new(false)),
                accepted,
            }))
            .unwrap();
        assert_eq!(handle.commands.capacity(), 0);
        // Discard the queue's wake evidence. Only Drop can set this probe now.
        wake_probe.0.store(false, Ordering::Release);
        drop(events);
        assert!(wake_probe.0.load(Ordering::Acquire));
        assert!(poll_once(run.as_mut()).is_pending());
        assert_eq!(cancellations.load(Ordering::Acquire), 1);
        assert_eq!(probe.encodes.load(Ordering::Acquire), 1);
        probe.retired.store(true, Ordering::Release);
        probe.closed.store(true, Ordering::Release);
        handle.begin_shutdown();
        assert!(matches!(
            poll_once(run.as_mut()),
            std::task::Poll::Ready(Ok(()))
        ));
    }

    #[tokio::test]
    async fn fatal_after_pending_cancellation_does_not_submit_it_twice() {
        let (handle, receiver, mut engine) = local_fixture();
        let probe = Arc::clone(&engine.probe);
        let cancellations = Arc::clone(&engine.inner.cancellation_count);
        let hub = engine.completion_hub();
        let owner = InferenceCompletionOwner::attach(&mut engine);
        let run = run_worker(
            engine,
            owner,
            receiver,
            handle.config.clone(),
            Arc::clone(&handle.control),
        );
        tokio::pin!(run);
        drop(admit_local(&handle, run.as_mut()));
        assert!(poll_once(run.as_mut()).is_pending());
        assert_eq!(cancellations.load(Ordering::Acquire), 1);
        probe.fatal.store(true, Ordering::Release);
        hub.notify();
        // First return is the completion branch's cooperative yield.
        assert!(poll_once(run.as_mut()).is_pending());
        assert!(poll_once(run.as_mut()).is_pending());
        assert_eq!(handle.phase(), WorkerPhase::Fatal);
        assert_eq!(cancellations.load(Ordering::Acquire), 1);
        probe.closed.store(true, Ordering::Release);
        handle.begin_shutdown();
        assert!(matches!(
            poll_once(run.as_mut()),
            std::task::Poll::Ready(Err(_))
        ));
    }

    #[derive(Default)]
    struct DisconnectEngine {
        completion_hub: CompletionHub,
        request: Option<GenerateRequest>,
        token_index: usize,
        cancellation_count: Arc<AtomicUsize>,
        cancellation_waits_remaining: usize,
        cancellation_pending: bool,
        cancelled: Vec<SequenceState>,
        fail_cancel: bool,
        fail_retirement: bool,
        fail_shutdown: bool,
        fail_physical_shutdown: bool,
        shutdown_calls: Arc<AtomicUsize>,
    }

    impl InferenceEngine for DisconnectEngine {
        fn completion_hub(&self) -> CompletionHub {
            self.completion_hub.clone()
        }

        fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
            Vec::new()
        }

        fn has_pending_async_work(&self) -> bool {
            self.cancellation_pending || self.fail_cancel || self.fail_retirement
        }

        fn shutdown(&mut self) -> RuntimeResult<InferenceShutdownProgress> {
            self.shutdown_calls.fetch_add(1, Ordering::Release);
            if self.fail_physical_shutdown {
                return Err(ferrule_common::Error::Execution {
                    message: "injected final physical shutdown failure with no logical ownership"
                        .into(),
                }
                .into());
            }
            if self.fail_shutdown {
                return Err(ferrule_runtime::Error::EngineUnavailable);
            }
            Ok(InferenceShutdownProgress::Complete)
        }

        fn encode(&self, prompt: &str) -> RuntimeResult<Vec<u32>> {
            Ok(prompt.bytes().map(u32::from).collect())
        }

        fn submit(&mut self, request: GenerateRequest) {
            self.request = Some(request);
            self.token_index = 0;
        }

        fn step(
            &mut self,
            on_token: &mut dyn FnMut(&ResidentTokenEvent) -> RuntimeResult<()>,
        ) -> RuntimeResult<ResidentDriverStep> {
            if self.fail_retirement && self.cancellation_pending {
                return Err(ferrule_runtime::Error::ShutdownIncomplete {
                    message: "injected retirement failure".into(),
                });
            }
            if self.fail_cancel || self.fail_retirement {
                return Ok(ResidentDriverStep::Blocked);
            }
            if self.cancellation_pending {
                self.cancellation_pending = false;
                let request = self
                    .request
                    .take()
                    .expect("pending cancellation retains its request");
                let mut sequence = SequenceState::from_request(
                    &request,
                    request.session_id.expect("worker assigns a session"),
                );
                sequence.finish_reason = Some(SequenceFinishReason::Cancelled);
                self.cancelled.push(sequence);
                return Ok(ResidentDriverStep::Executed {
                    action_kind: ferrule_runtime::ResidentActionKind::Cancel,
                    rows: 0,
                    staged: 0,
                    finished: 0,
                });
            }
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
        ) -> RuntimeResult<InferenceCancelProgress> {
            self.cancellation_count.fetch_add(1, Ordering::AcqRel);
            if self.fail_cancel {
                return Err(ferrule_runtime::Error::ShutdownIncomplete {
                    message: "injected cancellation failure".into(),
                });
            }
            if self.cancellation_waits_remaining > 0 {
                self.cancellation_waits_remaining -= 1;
                self.cancellation_pending = true;
                self.completion_hub.notify();
                return Ok(InferenceCancelProgress::Pending);
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

    fn has_runtime_failure(error: &ferrule_runtime::Error, expected: &str) -> bool {
        match error {
            ferrule_runtime::Error::ShutdownIncomplete { message } => message == expected,
            ferrule_runtime::Error::EngineUnavailable => expected == "quarantine",
            ferrule_runtime::Error::FailureBatch { failures, .. } => failures
                .iter()
                .any(|error| has_runtime_failure(error, expected)),
            ferrule_runtime::Error::Backend {
                source: ferrule_common::Error::ModelSource { source },
            } => {
                let error = source
                    .downcast_ref::<Arc<RuntimeWorkerExecutionError>>()
                    .unwrap();
                match error.as_ref() {
                    WorkerExecutionError::Runtime { source }
                    | WorkerExecutionError::Cancellation { source }
                    | WorkerExecutionError::ShutdownCancellation { source } => {
                        has_runtime_failure(source, expected)
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    #[tokio::test]
    async fn shutdown_preserves_cancellation_retirement_and_quarantine_failures() {
        for retirement in [false, true] {
            for quarantine in [false, true] {
                let shutdown_calls = Arc::new(AtomicUsize::new(0));
                let calls = Arc::clone(&shutdown_calls);
                let worker = spawn_model_worker_with(
                    move || {
                        Ok::<_, std::convert::Infallible>(DisconnectEngine {
                            fail_cancel: !retirement,
                            fail_retirement: retirement,
                            fail_shutdown: quarantine,
                            cancellation_waits_remaining: usize::from(retirement),
                            shutdown_calls: calls,
                            ..Default::default()
                        })
                    },
                    WorkerConfig::default(),
                )
                .unwrap();
                let mut events = worker
                    .handle()
                    .submit(WorkerRequest {
                        prompt: "hello".into(),
                        max_tokens: 8,
                        stop: Vec::new(),
                        ignore_eos: false,
                    })
                    .await
                    .unwrap();
                let error =
                    tokio::time::timeout(std::time::Duration::from_secs(2), worker.shutdown())
                        .await
                        .unwrap()
                        .unwrap_err();
                let expected = if retirement {
                    "injected retirement failure"
                } else {
                    "injected cancellation failure"
                };
                assert!(has_runtime_failure(&error, expected), "{error:?}");
                assert_eq!(has_runtime_failure(&error, "quarantine"), quarantine);
                assert_eq!(shutdown_calls.load(Ordering::Acquire), 1);
                assert!(matches!(
                    events.recv().await,
                    Some(WorkerEvent::Failed { .. })
                ));
                assert!(events.recv().await.is_none());
            }
        }
    }

    #[tokio::test]
    async fn physical_shutdown_failure_without_requests_preserves_source_and_quarantines() {
        let calls = Arc::new(AtomicUsize::new(0));
        let owner_calls = Arc::clone(&calls);
        let worker = spawn_model_worker_with(
            move || {
                Ok::<_, std::convert::Infallible>(DisconnectEngine {
                    fail_physical_shutdown: true,
                    shutdown_calls: owner_calls,
                    ..Default::default()
                })
            },
            WorkerConfig::default(),
        )
        .unwrap();
        let error = tokio::time::timeout(std::time::Duration::from_secs(2), worker.shutdown())
            .await
            .unwrap()
            .unwrap_err();
        let ferrule_runtime::Error::Backend {
            source: ferrule_common::Error::ModelSource { source },
        } = &error
        else {
            panic!("missing typed worker failure: {error:?}")
        };
        let WorkerExecutionError::Runtime {
            source:
                ferrule_runtime::Error::Backend {
                    source: ferrule_common::Error::Execution { message },
                },
        } = source
            .downcast_ref::<Arc<RuntimeWorkerExecutionError>>()
            .unwrap()
            .as_ref()
        else {
            panic!("physical error lost its typed source: {error:?}")
        };
        assert!(message.contains("final physical shutdown failure"));
        assert_eq!(calls.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn disconnected_cancellation_failure_is_not_lost_at_shutdown() {
        let engine = RetryCleanupEngine::new(false, false);
        let cancellations = Arc::clone(&engine.cancel_calls);
        engine.retry_ready.store(true, Ordering::Release);
        let worker = spawn_model_worker_with(
            move || Ok::<_, std::convert::Infallible>(engine),
            WorkerConfig::default(),
        )
        .unwrap();
        let events = worker
            .handle()
            .submit(WorkerRequest {
                prompt: "hello".into(),
                max_tokens: 8,
                stop: Vec::new(),
                ignore_eos: false,
            })
            .await
            .unwrap();
        drop(events);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while cancellations.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let error = worker.shutdown().await.unwrap_err();
        assert!(has_runtime_failure(&error, "first cleanup failed"));
        assert_eq!(cancellations.load(Ordering::Acquire), 1);
    }

    #[test]
    fn duplicate_worker_admission_preserves_original_subscription_and_terminal() {
        let mut engine = DisconnectEngine::default();
        let mut active = HashMap::new();
        let mut receivers = Vec::new();
        for duplicate in [false, true] {
            let (events, receiver) = mpsc::channel(8);
            let (accepted, mut acceptance) = oneshot::channel();
            assert!(!handle_command(
                WorkerCommand::Submit(SubmitCommand {
                    permit: test_permit(),
                    request_id: RequestId(1),
                    enqueued_at: Instant::now(),
                    request: WorkerRequest {
                        prompt: "hello".into(),
                        max_tokens: 8,
                        stop: Vec::new(),
                        ignore_eos: false
                    },
                    events,
                    cancellation: Arc::new(AtomicBool::new(false)),
                    accepted,
                }),
                &mut engine,
                &mut active,
                &ready_control()
            ));
            assert_eq!(acceptance.try_recv().unwrap().is_err(), duplicate);
            receivers.push(receiver);
        }
        assert_eq!(active.len(), 1);
        assert!(receivers[1].try_recv().is_err());
        engine.cancel_request(RequestId(1)).unwrap();
        drain_terminal(&mut engine, &mut active, &ready_control());
        assert!(matches!(
            receivers[0].try_recv(),
            Ok(WorkerEvent::Cancelled)
        ));
        drain_terminal(&mut engine, &mut active, &ready_control());
        assert!(receivers[0].try_recv().is_err());
        assert!(active.is_empty());
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
            cancellation_pending: false,
            cancelled: Vec::new(),
            ..Default::default()
        };
        let mut active = HashMap::new();
        let (events, _events_receiver) = mpsc::channel(1);
        active.insert(
            RequestId(1),
            ActiveRequest {
                _metrics: METRICS.start_request(),
                permit: test_permit(),
                requires_restore_or_shutdown: false,
                events,
                cancellation: Arc::new(AtomicBool::new(true)),
                cancellation_submitted: false,
                cancellation_failed: false,
                terminal_sent: false,
                session_id: SessionId(1),
                submitted_at: Instant::now(),
                emitted_tokens: 0,
            },
        );
        let mut scratch = Vec::with_capacity(1);
        let scratch_pointer = scratch.as_ptr();

        cancel_disconnected(
            &mut engine,
            &mut active,
            &mut scratch,
            &mut Vec::new(),
            &ready_control(),
        );

        assert_eq!(scratch, vec![RequestId(1)]);
        assert_eq!(scratch.as_ptr(), scratch_pointer);
        assert_eq!(cancellation_count.load(Ordering::Acquire), 1);
        drain_terminal(&mut engine, &mut active, &ready_control());
        assert!(active.is_empty());

        engine.submit(test_request(2));
        let (events, _events_receiver) = mpsc::channel(1);
        active.insert(
            RequestId(2),
            ActiveRequest {
                _metrics: METRICS.start_request(),
                permit: test_permit(),
                requires_restore_or_shutdown: false,
                events,
                cancellation: Arc::new(AtomicBool::new(false)),
                cancellation_submitted: false,
                cancellation_failed: false,
                terminal_sent: false,
                session_id: SessionId(2),
                submitted_at: Instant::now(),
                emitted_tokens: 0,
            },
        );
        cancel_disconnected(
            &mut engine,
            &mut active,
            &mut scratch,
            &mut Vec::new(),
            &ready_control(),
        );

        assert!(scratch.is_empty());
        assert_eq!(scratch.as_ptr(), scratch_pointer);
        assert_eq!(cancellation_count.load(Ordering::Acquire), 1);
        assert_eq!(
            engine.request.as_ref().map(|request| request.id),
            Some(RequestId(2))
        );
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
            cancellation_pending: false,
            cancelled: Vec::new(),
            ..Default::default()
        };
        let mut active = HashMap::new();
        let (events, _events_receiver) = mpsc::channel(1);
        active.insert(
            RequestId(3),
            ActiveRequest {
                _metrics: METRICS.start_request(),
                permit: test_permit(),
                requires_restore_or_shutdown: false,
                events,
                cancellation: Arc::new(AtomicBool::new(true)),
                cancellation_submitted: false,
                cancellation_failed: false,
                terminal_sent: false,
                session_id: SessionId(3),
                submitted_at: Instant::now(),
                emitted_tokens: 0,
            },
        );
        let mut scratch = Vec::new();

        cancel_disconnected(
            &mut engine,
            &mut active,
            &mut scratch,
            &mut Vec::new(),
            &ready_control(),
        );

        assert!(active.contains_key(&RequestId(3)));
        assert!(engine.request.is_some());
        assert!(engine.cancelled.is_empty());
        assert_eq!(cancellation_count.load(Ordering::Acquire), 1);

        cancel_disconnected(
            &mut engine,
            &mut active,
            &mut scratch,
            &mut Vec::new(),
            &ready_control(),
        );

        assert!(active.contains_key(&RequestId(3)));
        assert!(engine.request.is_some());
        assert_eq!(cancellation_count.load(Ordering::Acquire), 1);
        assert!(matches!(
            engine.step(&mut |_| Ok(())).unwrap(),
            ResidentDriverStep::Executed {
                action_kind: ferrule_runtime::ResidentActionKind::Cancel,
                ..
            }
        ));
        drain_terminal(&mut engine, &mut active, &ready_control());
        assert!(active.is_empty());
    }
    fn pr13_assert_counts(admission: &ServerAdmission, requests: usize, bytes: usize) {
        let snapshot = admission.snapshot();
        assert_eq!(snapshot.held_requests, requests);
        assert_eq!(snapshot.held_prompt_bytes, bytes);
        assert_eq!(
            snapshot.available_requests + snapshot.held_requests,
            snapshot.request_limit
        );
        assert_eq!(
            snapshot.available_prompt_bytes + snapshot.held_prompt_bytes,
            snapshot.prompt_bytes_limit
        );
        eprintln!("PR13 ledger={snapshot:?}");
    }

    #[test]
    fn pr13_permit_conservation_transfer_and_failure_atomicity() {
        let admission = ServerAdmission::new(2, 8);
        let permit = admission.try_acquire(5).unwrap();
        let observer = permit.observer();
        pr13_assert_counts(&admission, 1, 5);
        assert!(matches!(
            admission.try_acquire(4),
            Err(ServerAdmissionError::Capacity {
                resource: ServerAdmissionResource::PromptBytes,
                ..
            })
        ));
        assert!(permit.reconcile_prompt_bytes(9).is_err());
        pr13_assert_counts(&admission, 1, 5);
        let second = admission.try_acquire(3).unwrap();
        assert!(matches!(
            admission.try_acquire(0),
            Err(ServerAdmissionError::Capacity {
                resource: ServerAdmissionResource::Requests,
                ..
            })
        ));
        drop(permit);
        pr13_assert_counts(&admission, 2, 8);
        observer.reconcile_prompt_bytes(0).unwrap();
        drop(observer);
        pr13_assert_counts(&admission, 1, 3);
        admission.close();
        assert!(matches!(
            admission.try_acquire(0),
            Err(ServerAdmissionError::Closed)
        ));
        drop(second);
        pr13_assert_counts(&admission, 0, 0);
    }

    #[tokio::test]
    async fn pr13_queued_timeout_drop_and_full_queue_keep_exact_owner() {
        let (mut handle, mut receiver, mut engine) = local_fixture();
        handle.config.admission_timeout = std::time::Duration::from_millis(1);
        // Expiration abandons the observer, not the queued prompt owner.
        assert!(matches!(
            handle.submit(request()).await,
            Err(WorkerRequestError::AdmissionTimeout)
        ));
        pr13_assert_counts(&handle.control.admission, 1, 5);
        assert!(matches!(
            handle.submit(request()).await,
            Err(WorkerRequestError::QueueFull)
        ));
        pr13_assert_counts(&handle.control.admission, 1, 5);
        let command = receiver.try_recv().unwrap();
        let mut active = HashMap::new();
        handle_command(command, &mut engine, &mut active, &handle.control);
        assert!(active.is_empty());
        pr13_assert_counts(&handle.control.admission, 0, 0);
        let mut future = Box::pin(handle.tokenize("hello".into()));
        assert!(poll_once(future.as_mut()).is_pending());
        drop(future);
        pr13_assert_counts(&handle.control.admission, 1, 5);
        handle_command(
            receiver.try_recv().unwrap(),
            &mut engine,
            &mut active,
            &handle.control,
        );
        pr13_assert_counts(&handle.control.admission, 0, 0);
    }

    struct Pr13Engine {
        cleanup: HashMap<RequestId, ferrule_runtime::RequestCleanupOwner>,
        hub: CompletionHub,
        requests: HashMap<RequestId, GenerateRequest>,
        finished: Vec<SequenceState>,
        cancelled: Vec<SequenceState>,
        failed: Vec<SequenceState>,
        cancel_calls: usize,
        fail_cancel: bool,
        terminal_quiescent: bool,
        held: usize,
        encode_error: bool,
        reject: bool,
        suspended: bool,
        submits: usize,
    }

    impl Default for Pr13Engine {
        fn default() -> Self {
            Self {
                cleanup: HashMap::new(),
                hub: CompletionHub::new(),
                requests: HashMap::new(),
                finished: Vec::new(),
                cancelled: Vec::new(),
                failed: Vec::new(),
                cancel_calls: 0,
                fail_cancel: false,
                terminal_quiescent: false,
                held: 0,
                encode_error: false,
                reject: false,
                suspended: false,
                submits: 0,
            }
        }
    }
    impl InferenceEngine for Pr13Engine {
        fn request_cleanup(&self, id: RequestId) -> InferenceRequestCleanup {
            if self.terminal_quiescent {
                return InferenceRequestCleanup::TerminalQuiescent;
            }
            self.cleanup
                .get(&id)
                .map_or(InferenceRequestCleanup::Unavailable, |owner| {
                    InferenceRequestCleanup::Tracked(owner.receipt())
                })
        }
        fn has_pending_async_work(&self) -> bool {
            false
        }
        fn completion_hub(&self) -> CompletionHub {
            self.hub.clone()
        }
        fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
            vec![]
        }
        fn encode(&self, _: &str) -> RuntimeResult<Vec<u32>> {
            if self.encode_error {
                Err(ferrule_runtime::Error::InvalidRequest {
                    message: "/private/tokenizer".into(),
                })
            } else {
                Ok(vec![1])
            }
        }
        fn submit(&mut self, request: GenerateRequest) {
            self.try_submit(request).unwrap();
        }
        fn try_submit(&mut self, request: GenerateRequest) -> RuntimeResult<()> {
            if self.reject {
                return Err(ferrule_runtime::RuntimeAdmissionError::Capacity {
                    resource: ferrule_runtime::RuntimeAdmissionResource::WaitingRequests,
                    limit: 1,
                    held: 1,
                }
                .into());
            }
            if self.requests.contains_key(&request.id) {
                return Err(ferrule_runtime::RuntimeAdmissionError::DuplicateRequest {
                    request_id: request.id,
                }
                .into());
            }
            self.submits += 1;
            self.cleanup.insert(request.id, Default::default());
            self.held += 1;
            self.requests.insert(request.id, request);
            Ok(())
        }
        fn step(
            &mut self,
            _: &mut dyn FnMut(&ResidentTokenEvent) -> RuntimeResult<()>,
        ) -> RuntimeResult<ResidentDriverStep> {
            Ok(ResidentDriverStep::Idle)
        }
        fn cancel_request(
            &mut self,
            request_id: RequestId,
        ) -> RuntimeResult<InferenceCancelProgress> {
            self.cancel_calls += 1;
            if self.fail_cancel {
                return Err(ferrule_runtime::Error::EngineUnavailable);
            }
            if self.suspended {
                Ok(InferenceCancelProgress::RequiresRestoreOrShutdown {
                    request_id,
                    session_id: SessionId(request_id.0),
                })
            } else {
                Ok(InferenceCancelProgress::Pending)
            }
        }
        fn drain_finished(&mut self) -> Vec<SequenceState> {
            std::mem::take(&mut self.finished)
        }
        fn drain_cancelled(&mut self) -> Vec<SequenceState> {
            std::mem::take(&mut self.cancelled)
        }
        fn drain_failed(&mut self) -> Vec<SequenceState> {
            std::mem::take(&mut self.failed)
        }
        fn admission_snapshot(&self) -> Option<ferrule_runtime::RuntimeAdmissionSnapshot> {
            Some(ferrule_runtime::RuntimeAdmissionSnapshot {
                limits: Default::default(),
                waiting_requests: self.requests.len(),
                request_identities_held: self.held,
                session_identities_held: self.held,
                closed: false,
            })
        }
    }

    #[tokio::test]
    async fn pr13_tokenization_runtime_rejection_and_duplicate_return_exactly_once() {
        let (handle, mut receiver, _) = local_fixture();
        let mut active = HashMap::new();
        let mut engine = Pr13Engine::default();
        for mode in 0..3 {
            engine.encode_error = mode == 0;
            engine.reject = mode == 1;
            let mut future = Box::pin(handle.submit(request()));
            assert!(poll_once(future.as_mut()).is_pending());
            handle_command(
                receiver.try_recv().unwrap(),
                &mut engine,
                &mut active,
                &handle.control,
            );
            if mode < 2 {
                assert!(matches!(
                    poll_once(future.as_mut()),
                    std::task::Poll::Ready(Err(WorkerRequestError::Rejected { .. }))
                ));
                drop(future);
                pr13_assert_counts(&handle.control.admission, 0, 0);
            } else {
                let events = match poll_once(future.as_mut()) {
                    std::task::Poll::Ready(Ok(events)) => events,
                    _ => panic!("admission"),
                };
                drop(future);
                assert_eq!(engine.submits, 1);
                pr13_assert_counts(&handle.control.admission, 1, 0);
                handle
                    .next_request_id
                    .store(events.request_id.0, Ordering::Relaxed);
                let mut duplicate = Box::pin(handle.submit(request()));
                assert!(poll_once(duplicate.as_mut()).is_pending());
                handle_command(
                    receiver.try_recv().unwrap(),
                    &mut engine,
                    &mut active,
                    &handle.control,
                );
                assert!(matches!(
                    poll_once(duplicate.as_mut()),
                    std::task::Poll::Ready(Err(WorkerRequestError::Rejected { .. }))
                ));
                drop(duplicate);
                assert_eq!(engine.submits, 1);
                assert_eq!(active.len(), 1);
                pr13_assert_counts(&handle.control.admission, 1, 0);
                drop(events);
                pr13_assert_counts(&handle.control.admission, 1, 0);
            }
        }
    }

    #[tokio::test]
    async fn pr13_terminal_cleanup_and_slow_observer_are_separate_barriers() {
        let (handle, mut receiver, _) = local_fixture();
        let mut engine = Pr13Engine::default();
        let mut active = HashMap::new();
        let mut future = Box::pin(handle.submit(request()));
        assert!(poll_once(future.as_mut()).is_pending());
        handle_command(
            receiver.try_recv().unwrap(),
            &mut engine,
            &mut active,
            &handle.control,
        );
        let mut events = match poll_once(future.as_mut()) {
            std::task::Poll::Ready(Ok(events)) => events,
            _ => panic!("admission"),
        };
        drop(future);
        let request = engine.requests.remove(&events.request_id).unwrap();
        let mut state = SequenceState::from_request(&request, request.session_id.unwrap());
        state.finish_reason = Some(SequenceFinishReason::MaxTokens);
        engine.finished.push(state);
        // Flag alone cannot rewrite owner Finished to Cancelled.
        events.cancellation.store(true, Ordering::Release);
        drain_terminal(&mut engine, &mut active, &handle.control);
        assert!(active.is_empty());
        pr13_assert_counts(&handle.control.admission, 1, 0);
        assert!(matches!(
            events.recv().await,
            Some(WorkerEvent::Finished { .. })
        ));
        drop(events);
        // Runtime still reports cleanup-owned identity, despite HTTP completion.
        pr13_assert_counts(&handle.control.admission, 1, 0);
        engine.held = 0;
        engine
            .cleanup
            .drain()
            .for_each(|(_, owner)| owner.release());
        drain_terminal(&mut engine, &mut active, &handle.control);
        pr13_assert_counts(&handle.control.admission, 0, 0);
        let permit = handle.try_acquire_admission(5).unwrap();
        pr13_assert_counts(&handle.control.admission, 1, 5);
        drop(permit);
        pr13_assert_counts(&handle.control.admission, 0, 0);
    }

    #[test]
    fn pr13_per_request_proof_ignores_unrelated_active_and_unresolved_requests() {
        let control = ControlState::with_config(&WorkerConfig {
            max_inflight_requests: 2,
            ..Default::default()
        });
        let engine = Pr13Engine::default();
        let a = control.admission.try_acquire(5).unwrap();
        let observer = a.observer();
        let owner = ferrule_runtime::RequestCleanupOwner::default();
        *a.lease.cleanup.lock().unwrap() = AdmissionCleanup {
            request_id: Some(RequestId(1)),
            proof: InferenceRequestCleanup::Tracked(owner.receipt()),
            terminal_consumed: true,
        };
        a.reconcile_prompt_bytes(0).unwrap();
        let b = control.admission.try_acquire(0).unwrap();
        control.retain_unresolved(RequestId(2));
        control.retain_admission(a);
        drop(observer);
        for _ in 0..3 {
            control.observe_admission_cleanup(&engine);
            pr13_assert_counts(&control.admission, 2, 0);
        }
        owner.release();
        for _ in 0..3 {
            control.observe_admission_cleanup(&engine);
            pr13_assert_counts(&control.admission, 1, 0);
        }
        assert!(control.terminal_admissions.lock().unwrap().is_empty());
        let reused = control.admission.try_acquire(0).unwrap();
        control.observe_admission_cleanup(&engine);
        pr13_assert_counts(&control.admission, 2, 0);
        drop(reused);
        drop(b);
        pr13_assert_counts(&control.admission, 0, 0);
    }

    #[test]
    fn pr13_unknown_receipt_drop_is_not_cleanup_or_idle_proof() {
        let control = ready_control();
        let engine = Pr13Engine::default();
        let permit = control.admission.try_acquire(0).unwrap();
        let owner = ferrule_runtime::RequestCleanupOwner::default();
        *permit.lease.cleanup.lock().unwrap() = AdmissionCleanup {
            request_id: Some(RequestId(1)),
            proof: InferenceRequestCleanup::Tracked(owner.receipt()),
            terminal_consumed: true,
        };
        control.retain_admission(permit);
        drop(owner);
        for _ in 0..3 {
            control.observe_admission_cleanup(&engine);
            pr13_assert_counts(&control.admission, 1, 0);
        }
    }

    #[tokio::test]
    async fn pr13_direct_worker_reuses_two_credits_for_ten_requests() {
        let (mut handle, mut receiver, _) = local_fixture();
        handle.control = Arc::new(ControlState::with_config(&WorkerConfig {
            max_inflight_requests: 2,
            ..Default::default()
        }));
        handle.control.mark_ready();
        let mut engine = Pr13Engine::default();
        let mut active = HashMap::new();
        for _ in 0..10 {
            let mut future = Box::pin(handle.submit(request()));
            assert!(poll_once(future.as_mut()).is_pending());
            handle_command(
                receiver.try_recv().unwrap(),
                &mut engine,
                &mut active,
                &handle.control,
            );
            let mut events = match poll_once(future.as_mut()) {
                std::task::Poll::Ready(Ok(events)) => events,
                _ => panic!("admission"),
            };
            drop(future);
            let id = events.request_id;
            let request = engine.requests.remove(&id).unwrap();
            let mut state = SequenceState::from_request(&request, request.session_id.unwrap());
            state.finish_reason = Some(SequenceFinishReason::MaxTokens);
            engine.finished.push(state);
            drain_terminal(&mut engine, &mut active, &handle.control);
            pr13_assert_counts(&handle.control.admission, 1, 0);
            assert!(matches!(
                events.recv().await,
                Some(WorkerEvent::Finished { .. })
            ));
            drop(events);
            pr13_assert_counts(&handle.control.admission, 1, 0);
            engine.cleanup.remove(&id).unwrap().release();
            engine.held -= 1;
            for _ in 0..3 {
                drain_terminal(&mut engine, &mut active, &handle.control);
                pr13_assert_counts(&handle.control.admission, 0, 0);
            }
            assert!(engine.cleanup.is_empty());
            assert!(
                handle
                    .control
                    .terminal_admissions
                    .lock()
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[tokio::test]
    async fn pr13_suspended_cancel_retains_owner_and_is_not_accepted() {
        let (handle, mut receiver, _) = local_fixture();
        let mut engine = Pr13Engine {
            suspended: true,
            ..Default::default()
        };
        let mut active = HashMap::new();
        let mut future = Box::pin(handle.submit(request()));
        assert!(poll_once(future.as_mut()).is_pending());
        handle_command(
            receiver.try_recv().unwrap(),
            &mut engine,
            &mut active,
            &handle.control,
        );
        let mut events = match poll_once(future.as_mut()) {
            std::task::Poll::Ready(Ok(events)) => events,
            _ => panic!("admission"),
        };
        drop(future);
        events.cancellation.store(true, Ordering::Release);
        cancel_disconnected(
            &mut engine,
            &mut active,
            &mut Vec::new(),
            &mut Vec::new(),
            &handle.control,
        );
        let record = &active[&events.request_id];
        assert!(!record.cancellation_submitted);
        assert!(record.requires_restore_or_shutdown);
        assert_eq!(handle.phase(), WorkerPhase::Draining);
        assert_eq!(engine.requests.len(), 1);
        assert_eq!(
            *handle.control.unresolved_requests.lock().unwrap(),
            vec![events.request_id]
        );
        assert!(matches!(
            events.recv().await,
            Some(WorkerEvent::Failed { .. })
        ));
        drop(events);
        pr13_assert_counts(&handle.control.admission, 1, 0);
    }

    #[test]
    fn pr13_config_and_id_exhaustion_are_failure_atomic() {
        let config = WorkerConfig {
            max_prompt_bytes: 0,
            ..Default::default()
        };
        assert!(config.validate_admission().is_err());
        let (handle, _, _) = local_fixture();
        handle.next_request_id.store(u64::MAX, Ordering::Relaxed);
        assert!(handle.allocate_request_id().is_err());
        assert_eq!(handle.next_request_id.load(Ordering::Relaxed), u64::MAX);
        pr13_assert_counts(&handle.control.admission, 0, 0);
    }

    #[tokio::test]
    async fn pr13_missing_cleanup_proof_holds_until_shutdown_and_closed_queue_releases() {
        let (handle, receiver, mut engine) = local_fixture();
        let permit = handle.try_acquire_admission(0).unwrap();
        handle.control.retain_admission(permit);
        // Legacy engine None is not a proof of cleanup or quiescence.
        handle.control.observe_admission_cleanup(&engine);
        pr13_assert_counts(&handle.control.admission, 1, 0);
        drop(receiver);
        assert!(matches!(
            handle.submit(request()).await,
            Err(WorkerRequestError::Unavailable { .. })
        ));
        pr13_assert_counts(&handle.control.admission, 1, 0);
        engine.probe.closed.store(true, Ordering::Release);
        let mut owner = InferenceCompletionOwner::attach(&mut engine);
        handle.begin_shutdown();
        shutdown_engine(&mut engine, &mut owner, &HashMap::new(), &handle.control)
            .await
            .unwrap();
        pr13_assert_counts(&handle.control.admission, 0, 0);
    }

    #[tokio::test]
    async fn pr13_cleanup_failure_keeps_server_request_after_observer_drop() {
        let (handle, _receiver, mut engine) = local_fixture();
        let permit = handle.try_acquire_admission(0).unwrap();
        handle.control.retain_admission(permit);
        engine.probe.closed.store(false, Ordering::Release);
        engine.probe.retired.store(false, Ordering::Release);
        let mut owner = InferenceCompletionOwner::attach(&mut engine);
        handle.control.begin_shutdown(ShutdownDeadline::now());
        let result =
            shutdown_engine(&mut engine, &mut owner, &HashMap::new(), &handle.control).await;
        assert!(result.is_err());
        pr13_assert_counts(&handle.control.admission, 1, 0);
        assert!(handle.snapshot().shutdown_report.custody_unknown);
    }

    struct RetryCleanupEngine {
        inner: Pr13Engine,
        retry_ready: Arc<AtomicBool>,
        cancel_calls: Arc<AtomicUsize>,
        cleanup_attempts: Arc<AtomicUsize>,
        shutdown_calls: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
        cleanup_pending: bool,
        fail_retry: bool,
    }

    impl Drop for RetryCleanupEngine {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::AcqRel);
        }
    }

    impl InferenceEngine for RetryCleanupEngine {
        fn completion_hub(&self) -> CompletionHub {
            self.inner.completion_hub()
        }
        fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
            vec![]
        }
        fn has_pending_async_work(&self) -> bool {
            true
        }
        fn request_cleanup(&self, id: RequestId) -> InferenceRequestCleanup {
            self.inner.request_cleanup(id)
        }
        fn encode(&self, prompt: &str) -> RuntimeResult<Vec<u32>> {
            self.inner.encode(prompt)
        }
        fn submit(&mut self, request: GenerateRequest) {
            self.inner.submit(request);
        }
        fn cancel_request(&mut self, id: RequestId) -> RuntimeResult<InferenceCancelProgress> {
            assert!(self.inner.requests.contains_key(&id));
            assert!(!self.cleanup_pending, "cancel must not be resubmitted");
            self.cancel_calls.fetch_add(1, Ordering::AcqRel);
            self.cleanup_attempts.fetch_add(1, Ordering::AcqRel);
            self.cleanup_pending = true;
            Err(ferrule_runtime::Error::ShutdownIncomplete {
                message: "first cleanup failed".into(),
            })
        }
        fn step(
            &mut self,
            _: &mut dyn FnMut(&ResidentTokenEvent) -> RuntimeResult<()>,
        ) -> RuntimeResult<ResidentDriverStep> {
            if !self.cleanup_pending || !self.retry_ready.load(Ordering::Acquire) {
                return Ok(ResidentDriverStep::Blocked);
            }
            self.cleanup_attempts.fetch_add(1, Ordering::AcqRel);
            if self.fail_retry {
                return Err(ferrule_runtime::Error::ShutdownIncomplete {
                    message: "cleanup retry failed".into(),
                });
            }
            self.cleanup_pending = false;
            for (id, request) in self.inner.requests.drain() {
                self.inner.cleanup.remove(&id).unwrap().release();
                self.inner.cancelled.push(SequenceState::from_request(
                    &request,
                    request.session_id.unwrap(),
                ));
            }
            Ok(ResidentDriverStep::Executed {
                action_kind: ferrule_runtime::ResidentActionKind::Cancel,
                rows: 0,
                staged: 0,
                finished: 0,
            })
        }
        fn shutdown(&mut self) -> RuntimeResult<InferenceShutdownProgress> {
            self.shutdown_calls.fetch_add(1, Ordering::AcqRel);
            if !self.inner.requests.is_empty() {
                return Err(ferrule_runtime::Error::ShutdownIncomplete {
                    message: "physical cleanup still unknown".into(),
                });
            }
            Ok(InferenceShutdownProgress::Complete)
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
    }

    impl RetryCleanupEngine {
        fn new(legacy: bool, fail_retry: bool) -> Self {
            Self {
                inner: Pr13Engine {
                    terminal_quiescent: legacy,
                    ..Default::default()
                },
                retry_ready: Arc::new(AtomicBool::new(false)),
                cancel_calls: Arc::new(AtomicUsize::new(0)),
                cleanup_attempts: Arc::new(AtomicUsize::new(0)),
                shutdown_calls: Arc::new(AtomicUsize::new(0)),
                drops: Arc::new(AtomicUsize::new(0)),
                cleanup_pending: false,
                fail_retry,
            }
        }
    }

    mod request_metrics {
        use super::*;

        fn assert_counts(metrics: &Metrics, total: u64, active: u64, finished: u64) {
            let snapshot = metrics.snapshot();
            assert_eq!(snapshot.total_requests, total);
            assert_eq!(snapshot.active_requests, active);
            assert_eq!(snapshot.finished_requests, finished);
            assert_eq!(total, active + finished);
        }

        fn admit<'metrics, E: InferenceEngine>(
            handle: &ModelWorkerHandle,
            receiver: &mut mpsc::Receiver<WorkerCommand>,
            engine: &mut E,
            active: &mut HashMap<RequestId, ActiveRequest<'metrics>>,
            metrics: &'metrics Metrics,
        ) -> EventSubscription {
            let mut submit = Box::pin(handle.submit(request()));
            assert!(poll_once(submit.as_mut()).is_pending());
            handle_command_with_metrics(
                receiver.try_recv().unwrap(),
                engine,
                active,
                &handle.control,
                metrics,
            );
            match poll_once(submit.as_mut()) {
                std::task::Poll::Ready(Ok(events)) => events,
                _ => panic!("worker must acknowledge admission"),
            }
        }

        async fn disconnected_cleanup_run_loop(legacy: bool, fail_retry: bool, expire: bool) {
            let metrics = Metrics::new();
            let (handle, receiver, _) = local_fixture();
            let mut engine = RetryCleanupEngine::new(legacy, fail_retry);
            let retry_ready = Arc::clone(&engine.retry_ready);
            let cancel_calls = Arc::clone(&engine.cancel_calls);
            let cleanup_attempts = Arc::clone(&engine.cleanup_attempts);
            let shutdown_calls = Arc::clone(&engine.shutdown_calls);
            let drops = Arc::clone(&engine.drops);
            let hub = engine.completion_hub();
            assert!(!engine.has_background_work());
            let owner = InferenceCompletionOwner::attach(&mut engine);
            let run = run_worker_with_metrics(
                engine,
                owner,
                receiver,
                handle.config.clone(),
                Arc::clone(&handle.control),
                &metrics,
            );
            tokio::pin!(run);
            let mut events = admit_local(&handle, run.as_mut());
            let id = events.request_id;
            // Use the same drop-only cancellation signal, retaining the receiver
            // solely to assert notification and cleanup are independent barriers.
            events.cancellation.store(true, Ordering::Release);
            handle.control.signal();
            assert!(poll_once(run.as_mut()).is_pending());
            assert_eq!(handle.phase(), WorkerPhase::Draining);
            assert_eq!(
                handle.snapshot().shutdown_report.stage,
                ShutdownStage::Drain
            );
            assert_counts(&metrics, 1, 1, 0);
            assert_eq!(cancel_calls.load(Ordering::Acquire), 1);
            assert_eq!(cleanup_attempts.load(Ordering::Acquire), 1);
            assert_eq!(shutdown_calls.load(Ordering::Acquire), 0);
            assert_eq!(drops.load(Ordering::Acquire), 0);
            assert!(
                handle
                    .control
                    .unresolved_requests
                    .lock()
                    .unwrap()
                    .contains(&id)
            );
            assert!(
                handle
                    .control
                    .terminal_admissions
                    .lock()
                    .unwrap()
                    .is_empty()
            );
            {
                let cleanup = events._permit.lease.cleanup.lock().unwrap();
                assert!(!cleanup.terminal_consumed);
                assert!(!cleanup.released());
            }
            let WorkerEvent::Failed { error } = events.receiver.try_recv().unwrap() else {
                panic!("missing cancellation failure")
            };
            assert!(matches!(
                error.as_ref(),
                WorkerExecutionError::Cancellation { .. }
            ));
            assert!(matches!(
                events.receiver.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
            pr13_assert_counts(&handle.control.admission, 1, 0);

            // No new command or explicit shutdown: only completion or deadline.
            if expire {
                tokio::time::advance(handle.config.shutdown_timeout).await;
            } else {
                retry_ready.store(true, Ordering::Release);
                hub.notify();
            }
            let error = tokio::time::timeout(std::time::Duration::from_secs(1), run.as_mut())
                .await
                .expect("worker must automatically drain")
                .unwrap_err();
            assert!(has_runtime_failure(&error, "first cleanup failed"));
            assert_eq!(cancel_calls.load(Ordering::Acquire), 1);
            assert_eq!(
                cleanup_attempts.load(Ordering::Acquire),
                if expire { 1 } else { 2 }
            );
            assert_eq!(shutdown_calls.load(Ordering::Acquire), usize::from(!expire));
            let unknown = fail_retry || expire;
            assert_eq!(drops.load(Ordering::Acquire), usize::from(!unknown));
            assert_counts(&metrics, 1, 0, 1);
            assert!(matches!(
                events.receiver.try_recv(),
                Err(mpsc::error::TryRecvError::Disconnected)
            ));
            {
                let cleanup = events._permit.lease.cleanup.lock().unwrap();
                assert_eq!(cleanup.terminal_consumed, !unknown);
                assert_eq!(cleanup.released(), !unknown);
            }
            let report = handle.snapshot().shutdown_report;
            assert_eq!(report.custody_unknown, unknown);
            assert_eq!(report.deadline_exceeded, expire);
            assert_eq!(report.quarantine, unknown);
            if fail_retry {
                assert!(has_runtime_failure(&error, "cleanup retry failed"));
                assert!(has_runtime_failure(
                    &error,
                    "physical cleanup still unknown"
                ));
            }
            if unknown {
                assert!(report.remaining_requests.contains(&id));
            } else {
                assert!(report.remaining_requests.is_empty());
            }
            pr13_assert_counts(&handle.control.admission, 1, 0);
            drop(events);
            pr13_assert_counts(&handle.control.admission, usize::from(unknown), 0);
            assert_eq!(
                handle.control.terminal_admissions.lock().unwrap().len(),
                usize::from(unknown)
            );
            assert_counts(&metrics, 1, 0, 1);
        }

        #[tokio::test]
        async fn disconnected_cleanup_error_automatically_drains_without_commands() {
            for legacy in [false, true] {
                disconnected_cleanup_run_loop(legacy, false, false).await;
            }
        }

        #[tokio::test]
        async fn disconnected_cleanup_retry_unknown_never_releases_permit() {
            for legacy in [false, true] {
                disconnected_cleanup_run_loop(legacy, true, false).await;
            }
        }

        #[tokio::test(start_paused = true)]
        async fn disconnected_cleanup_deadline_keeps_unknown_owner_and_permit() {
            for legacy in [false, true] {
                disconnected_cleanup_run_loop(legacy, false, true).await;
            }
        }

        #[tokio::test]
        async fn terminal_matrix_duplicates_and_unrelated_cleanup_are_independent() {
            for kind in 0..3 {
                let metrics = Metrics::new();
                let (handle, mut receiver, _) = local_fixture();
                let mut engine = Pr13Engine::default();
                let mut active = HashMap::new();
                let mut first = admit(&handle, &mut receiver, &mut engine, &mut active, &metrics);
                let second = admit(&handle, &mut receiver, &mut engine, &mut active, &metrics);
                let request = engine.requests.remove(&first.request_id).unwrap();
                first.cancellation.store(true, Ordering::Release);
                for _ in 0..3 {
                    let mut sequence =
                        SequenceState::from_request(&request, request.session_id.unwrap());
                    sequence.generated = 3;
                    sequence.finish_reason = Some(SequenceFinishReason::MaxTokens);
                    match kind {
                        0 => engine.finished.push(sequence),
                        1 => engine.cancelled.push(sequence),
                        _ => engine.failed.push(sequence),
                    }
                    drain_terminal(&mut engine, &mut active, &handle.control);
                    assert_counts(&metrics, 2, 1, 1);
                    assert!(active.contains_key(&second.request_id));
                    assert_eq!(handle.control.terminal_admissions.lock().unwrap().len(), 1);
                }
                match first.recv().await.unwrap() {
                    WorkerEvent::Finished { reason, usage } if kind == 0 => {
                        assert_eq!(reason, SequenceFinishReason::MaxTokens);
                        assert_eq!(usage.prompt_tokens, request.prompt_tokens.len());
                        assert_eq!(usage.completion_tokens, 3);
                    }
                    WorkerEvent::Cancelled if kind == 1 => {}
                    WorkerEvent::Failed { error } if kind == 2 => {
                        assert!(matches!(
                            error.as_ref(),
                            WorkerExecutionError::ModelExecution
                        ));
                    }
                    event => panic!("wrong terminal: {event:?}"),
                }
                assert!(first.recv().await.is_none());
                drop(first);
                pr13_assert_counts(&handle.control.admission, 2, 0);
                handle.control.retain_unresolved(second.request_id);
                engine.cleanup.remove(&request.id).unwrap().release();
                for _ in 0..3 {
                    drain_terminal(&mut engine, &mut active, &handle.control);
                    assert_counts(&metrics, 2, 1, 1);
                    pr13_assert_counts(&handle.control.admission, 1, 0);
                }
                // Losing another cleanup owner must never release its permit.
                let second_id = second.request_id;
                engine.cleanup.remove(&second_id).unwrap();
                drop(second);
                finish_request(
                    &mut active,
                    second_id,
                    RequestOutcome::Quarantined,
                    &handle.control,
                );
                control_cleanup_repeated(&handle, &engine);
                assert_counts(&metrics, 2, 0, 2);
                pr13_assert_counts(&handle.control.admission, 1, 0);
            }
        }

        fn control_cleanup_repeated(handle: &ModelWorkerHandle, engine: &Pr13Engine) {
            for _ in 0..3 {
                handle.control.observe_admission_cleanup(engine);
            }
        }

        #[tokio::test]
        async fn suspended_notification_then_terminal_or_fatal_never_sends_twice() {
            for fatal in [false, true] {
                let metrics = Metrics::new();
                let (handle, mut receiver, _) = local_fixture();
                let mut engine = Pr13Engine {
                    suspended: true,
                    ..Default::default()
                };
                let mut active = HashMap::new();
                let mut events = admit(&handle, &mut receiver, &mut engine, &mut active, &metrics);
                let id = events.request_id;
                events.cancellation.store(true, Ordering::Release);
                let mut failures = Vec::new();
                cancel_disconnected(
                    &mut engine,
                    &mut active,
                    &mut Vec::new(),
                    &mut failures,
                    &handle.control,
                );
                let deadline = handle.begin_shutdown();
                let Some(WorkerEvent::Failed { error }) = events.recv().await else {
                    panic!("missing suspended failure")
                };
                assert!(matches!(
                    error.as_ref(),
                    WorkerExecutionError::Cancellation { .. }
                ));
                assert_counts(&metrics, 1, 1, 0);
                assert!(!active[&id].cancellation_submitted);
                if fatal {
                    fail_all(
                        &mut engine,
                        &mut active,
                        Arc::new(WorkerExecutionError::ModelExecution),
                        &mut failures,
                        &handle.control,
                    );
                }
                let request = engine.requests.remove(&id).unwrap();
                for _ in 0..3 {
                    engine.cancelled.push(SequenceState::from_request(
                        &request,
                        request.session_id.unwrap(),
                    ));
                    drain_terminal(&mut engine, &mut active, &handle.control);
                    assert_counts(&metrics, 1, 0, 1);
                }
                assert!(events.recv().await.is_none());
                assert!(failures.is_empty());
                assert_eq!(engine.cancel_calls, 1);
                assert_eq!(handle.begin_shutdown(), deadline);
                drop(events);
                pr13_assert_counts(&handle.control.admission, 1, 0);
                engine.cleanup.remove(&id).unwrap().release();
                control_cleanup_repeated(&handle, &engine);
                pr13_assert_counts(&handle.control.admission, 0, 0);
            }
        }

        #[tokio::test]
        async fn cancellation_causes_keep_typed_errors_and_late_cleanup_proof() {
            for cause in [
                CancellationCause::Disconnected,
                CancellationCause::Shutdown,
                CancellationCause::Fatal,
            ] {
                for legacy in [false, true] {
                    let metrics = Metrics::new();
                    let (handle, mut receiver, _) = local_fixture();
                    let mut engine = Pr13Engine {
                        fail_cancel: true,
                        terminal_quiescent: legacy,
                        ..Default::default()
                    };
                    let mut active = HashMap::new();
                    let mut events =
                        admit(&handle, &mut receiver, &mut engine, &mut active, &metrics);
                    let id = events.request_id;
                    let mut failures = Vec::new();
                    let primary = Arc::new(WorkerExecutionError::ModelExecution);
                    for _ in 0..3 {
                        match cause {
                            CancellationCause::Fatal => fail_all(
                                &mut engine,
                                &mut active,
                                Arc::clone(&primary),
                                &mut failures,
                                &handle.control,
                            ),
                            _ => request_cancellation(
                                &mut engine,
                                &mut active,
                                id,
                                cause,
                                &mut failures,
                                &handle.control,
                            ),
                        }
                        if matches!(cause, CancellationCause::Disconnected) {
                            assert_counts(&metrics, 1, 1, 0);
                            assert!(active[&id].cancellation_failed);
                            assert!(!active[&id].cancellation_submitted);
                            assert_eq!(handle.phase(), WorkerPhase::Draining);
                        } else {
                            assert_counts(&metrics, 1, 0, 1);
                        }
                    }
                    assert_eq!(engine.cancel_calls, 1);
                    assert_eq!(failures.len(), 1);
                    let Some(WorkerEvent::Failed { error }) = events.recv().await else {
                        panic!("missing cancellation failure")
                    };
                    match cause {
                        CancellationCause::Disconnected => assert!(matches!(
                            error.as_ref(),
                            WorkerExecutionError::Cancellation { .. }
                        )),
                        CancellationCause::Shutdown => assert!(matches!(
                            error.as_ref(),
                            WorkerExecutionError::ShutdownCancellation { .. }
                        )),
                        CancellationCause::Fatal => {
                            assert!(Arc::ptr_eq(&error, &primary));
                            assert!(matches!(
                                failures[0],
                                ferrule_runtime::Error::EngineUnavailable
                            ));
                        }
                    }
                    if matches!(cause, CancellationCause::Disconnected) {
                        assert!(matches!(
                            events.receiver.try_recv(),
                            Err(mpsc::error::TryRecvError::Empty)
                        ));
                    } else {
                        assert!(events.recv().await.is_none());
                    }
                    drop(events);
                    control_cleanup_repeated(&handle, &engine);
                    pr13_assert_counts(&handle.control.admission, 1, 0);
                    let request = engine.requests.remove(&id).unwrap();
                    engine.failed.push(SequenceState::from_request(
                        &request,
                        request.session_id.unwrap(),
                    ));
                    drain_terminal(&mut engine, &mut active, &handle.control);
                    assert_counts(&metrics, 1, 0, 1);
                    pr13_assert_counts(&handle.control.admission, usize::from(!legacy), 0);
                    engine.cleanup.remove(&id).unwrap().release();
                    control_cleanup_repeated(&handle, &engine);
                    pr13_assert_counts(&handle.control.admission, 0, 0);
                }
            }
        }

        #[tokio::test(start_paused = true)]
        async fn deadline_keeps_active_metrics_until_unknown_custody_handoff() {
            let metrics = Metrics::new();
            let (handle, mut receiver, mut engine) = local_fixture();
            let mut active = HashMap::new();
            let mut events = admit(&handle, &mut receiver, &mut engine, &mut active, &metrics);
            let id = events.request_id;
            let mut owner = InferenceCompletionOwner::attach(&mut engine);
            let deadline = handle.begin_shutdown();
            {
                let cancel = cancel_all(&mut engine, &mut owner, &mut active, &handle.control);
                tokio::pin!(cancel);
                assert!(poll_once(cancel.as_mut()).is_pending());
                assert_counts(&metrics, 1, 1, 0);
                tokio::time::advance(handle.config.shutdown_timeout).await;
                assert!(cancel.await.is_empty());
            }
            assert_counts(&metrics, 1, 1, 0);
            // Deadline is neither a successful cancellation nor cleanup proof.
            assert!(events.receiver.try_recv().is_err());
            assert_eq!(handle.begin_shutdown(), deadline);
            assert!(handle.control.expired());
            assert!(handle.snapshot().shutdown_report.custody_unknown);
            for _ in 0..3 {
                finish_request(
                    &mut active,
                    id,
                    RequestOutcome::Quarantined,
                    &handle.control,
                );
                assert_counts(&metrics, 1, 0, 1);
            }
            assert!(events.recv().await.is_none());
            drop(events);
            handle.control.observe_admission_cleanup(&engine);
            pr13_assert_counts(&handle.control.admission, 1, 0);
        }

        #[tokio::test]
        async fn shutdown_drain_failure_finishes_metrics_but_retains_cleanup() {
            let metrics = Metrics::new();
            let (handle, mut receiver, _) = local_fixture();
            let mut engine = DisconnectEngine {
                fail_retirement: true,
                cancellation_waits_remaining: 1,
                ..Default::default()
            };
            let mut active = HashMap::new();
            let mut events = admit(&handle, &mut receiver, &mut engine, &mut active, &metrics);
            let mut owner = InferenceCompletionOwner::attach(&mut engine);
            handle.begin_shutdown();
            let failures = cancel_all(&mut engine, &mut owner, &mut active, &handle.control).await;
            assert_eq!(failures.len(), 1);
            assert!(has_runtime_failure(
                &failures[0],
                "injected retirement failure"
            ));
            assert_counts(&metrics, 1, 0, 1);
            assert!(matches!(
                events.recv().await,
                Some(WorkerEvent::Failed { .. })
            ));
            assert!(events.recv().await.is_none());
            for _ in 0..3 {
                assert!(
                    cancel_all(&mut engine, &mut owner, &mut active, &handle.control)
                        .await
                        .is_empty()
                );
                assert_counts(&metrics, 1, 0, 1);
            }
            assert_eq!(engine.cancellation_count.load(Ordering::Acquire), 1);
            drop(events);
            handle.control.observe_admission_cleanup(&engine);
            pr13_assert_counts(&handle.control.admission, 1, 0);
        }

        #[tokio::test]
        async fn success_starts_once_and_owner_drop_finishes_without_observer_drop() {
            let metrics = Metrics::new();
            let (handle, mut receiver, mut engine) = local_fixture();
            let mut active = HashMap::new();
            assert_counts(&metrics, 0, 0, 0);
            let events = admit(&handle, &mut receiver, &mut engine, &mut active, &metrics);
            assert_eq!(active.len(), 1);
            assert_counts(&metrics, 1, 1, 0);
            drop(active);
            assert_counts(&metrics, 1, 0, 1);
            drop(events);
            assert_counts(&metrics, 1, 0, 1);
        }

        #[tokio::test]
        async fn rejection_tokenization_and_duplicate_do_not_start_generations() {
            let metrics = Metrics::new();
            let (handle, mut receiver, _) = local_fixture();
            let mut engine = Pr13Engine::default();
            let mut active = HashMap::new();
            for tokenization_error in [true, false] {
                engine.encode_error = tokenization_error;
                engine.reject = !tokenization_error;
                let mut submit = Box::pin(handle.submit(request()));
                assert!(poll_once(submit.as_mut()).is_pending());
                handle_command_with_metrics(
                    receiver.try_recv().unwrap(),
                    &mut engine,
                    &mut active,
                    &handle.control,
                    &metrics,
                );
                assert!(matches!(
                    poll_once(submit.as_mut()),
                    std::task::Poll::Ready(Err(WorkerRequestError::Rejected { .. }))
                ));
                assert!(active.is_empty());
                assert_eq!(engine.submits, 0);
                assert_counts(&metrics, 0, 0, 0);
            }
            engine.reject = false;
            let mut tokenize = Box::pin(handle.tokenize("hello".into()));
            assert!(poll_once(tokenize.as_mut()).is_pending());
            handle_command_with_metrics(
                receiver.try_recv().unwrap(),
                &mut engine,
                &mut active,
                &handle.control,
                &metrics,
            );
            assert!(matches!(
                poll_once(tokenize.as_mut()),
                std::task::Poll::Ready(Ok((_, tokens))) if tokens == vec![1]
            ));
            assert_counts(&metrics, 0, 0, 0);

            let events = admit(&handle, &mut receiver, &mut engine, &mut active, &metrics);
            assert_counts(&metrics, 1, 1, 0);
            handle
                .next_request_id
                .store(events.request_id.0, Ordering::Relaxed);
            let mut duplicate = Box::pin(handle.submit(request()));
            assert!(poll_once(duplicate.as_mut()).is_pending());
            handle_command_with_metrics(
                receiver.try_recv().unwrap(),
                &mut engine,
                &mut active,
                &handle.control,
                &metrics,
            );
            assert!(matches!(
                poll_once(duplicate.as_mut()),
                std::task::Poll::Ready(Err(WorkerRequestError::Rejected { .. }))
            ));
            assert_eq!(engine.submits, 1);
            assert_eq!(active.len(), 1);
            assert_counts(&metrics, 1, 1, 0);
            drop(events);
            assert_counts(&metrics, 1, 1, 0);
            drop(active);
            assert_counts(&metrics, 1, 0, 1);
        }

        #[tokio::test(start_paused = true)]
        async fn queued_timeout_queue_full_and_abandoned_submit_do_not_count() {
            let metrics = Metrics::new();
            let (mut handle, mut receiver, _) = local_fixture();
            handle.config.admission_timeout = std::time::Duration::from_millis(1);
            let mut engine = Pr13Engine::default();
            let mut active = HashMap::new();
            assert!(matches!(
                handle.submit(request()).await,
                Err(WorkerRequestError::AdmissionTimeout)
            ));
            assert!(matches!(
                handle.submit(request()).await,
                Err(WorkerRequestError::QueueFull)
            ));
            assert_counts(&metrics, 0, 0, 0);
            handle_command_with_metrics(
                receiver.try_recv().unwrap(),
                &mut engine,
                &mut active,
                &handle.control,
                &metrics,
            );
            let mut submit = Box::pin(handle.submit(request()));
            assert!(poll_once(submit.as_mut()).is_pending());
            drop(submit);
            handle_command_with_metrics(
                receiver.try_recv().unwrap(),
                &mut engine,
                &mut active,
                &handle.control,
                &metrics,
            );
            assert!(active.is_empty());
            assert_eq!(engine.submits, 0);
            assert_counts(&metrics, 0, 0, 0);
        }

        #[tokio::test]
        async fn normal_terminal_and_repeated_observation_finish_only_the_matching_owner() {
            let metrics = Metrics::new();
            let (handle, mut receiver, _) = local_fixture();
            let mut engine = Pr13Engine::default();
            let mut active = HashMap::new();
            let mut first = admit(&handle, &mut receiver, &mut engine, &mut active, &metrics);
            let second = admit(&handle, &mut receiver, &mut engine, &mut active, &metrics);
            assert_counts(&metrics, 2, 2, 0);
            assert_eq!(metrics.snapshot().max_running, 2);
            let request = engine.requests.remove(&first.request_id).unwrap();
            for _ in 0..3 {
                let mut terminal =
                    SequenceState::from_request(&request, request.session_id.unwrap());
                terminal.finish_reason = Some(SequenceFinishReason::MaxTokens);
                engine.finished.push(terminal);
                drain_terminal(&mut engine, &mut active, &handle.control);
                assert!(active.contains_key(&second.request_id));
                assert_counts(&metrics, 2, 1, 1);
            }
            // Terminal observation ends accounting even while the client and
            // runtime cleanup still retain the admission lease.
            pr13_assert_counts(&handle.control.admission, 2, 0);
            assert!(matches!(
                first.recv().await,
                Some(WorkerEvent::Finished { .. })
            ));
            assert!(first.recv().await.is_none());
            drop(first);
            assert_counts(&metrics, 2, 1, 1);
            engine.cleanup.remove(&request.id).unwrap().release();
            engine.held -= 1;
            for _ in 0..3 {
                drain_terminal(&mut engine, &mut active, &handle.control);
                assert_counts(&metrics, 2, 1, 1);
            }
            pr13_assert_counts(&handle.control.admission, 1, 0);
            drop(second);
            assert_counts(&metrics, 2, 1, 1);
            drop(active);
            assert_counts(&metrics, 2, 0, 2);
        }

        #[tokio::test]
        async fn client_drop_cancellation_counts_until_the_worker_consumes_terminal() {
            for pending in [false, true] {
                let metrics = Metrics::new();
                let (handle, mut receiver, _) = local_fixture();
                let mut engine = DisconnectEngine {
                    cancellation_waits_remaining: usize::from(pending),
                    ..Default::default()
                };
                let mut active = HashMap::new();
                let events = admit(&handle, &mut receiver, &mut engine, &mut active, &metrics);
                let id = events.request_id;
                drop(events);
                assert!(active[&id].cancellation.load(Ordering::Acquire));
                assert_counts(&metrics, 1, 1, 0);
                for _ in 0..3 {
                    cancel_disconnected(
                        &mut engine,
                        &mut active,
                        &mut Vec::new(),
                        &mut Vec::new(),
                        &handle.control,
                    );
                    assert!(active[&id].cancellation_submitted);
                    assert_counts(&metrics, 1, 1, 0);
                }
                assert_eq!(engine.cancellation_count.load(Ordering::Acquire), 1);
                if pending {
                    drain_terminal(&mut engine, &mut active, &handle.control);
                    assert_counts(&metrics, 1, 1, 0);
                    engine.step(&mut |_| Ok(())).unwrap();
                }
                for _ in 0..3 {
                    drain_terminal(&mut engine, &mut active, &handle.control);
                    assert!(active.is_empty());
                    assert_counts(&metrics, 1, 0, 1);
                }
            }
        }

        #[tokio::test]
        async fn requires_restore_or_shutdown_retains_metrics_after_failure_delivery() {
            let metrics = Metrics::new();
            let (handle, mut receiver, _) = local_fixture();
            let mut engine = Pr13Engine {
                suspended: true,
                ..Default::default()
            };
            let mut active = HashMap::new();
            let mut events = admit(&handle, &mut receiver, &mut engine, &mut active, &metrics);
            let id = events.request_id;
            events.cancellation.store(true, Ordering::Release);
            cancel_disconnected(
                &mut engine,
                &mut active,
                &mut Vec::new(),
                &mut Vec::new(),
                &handle.control,
            );
            assert!(active[&id].requires_restore_or_shutdown);
            assert!(!active[&id].cancellation_submitted);
            assert_eq!(handle.phase(), WorkerPhase::Draining);
            assert!(matches!(
                events.recv().await,
                Some(WorkerEvent::Failed { .. })
            ));
            assert_counts(&metrics, 1, 1, 0);
            drop(events);
            for _ in 0..3 {
                cancel_disconnected(
                    &mut engine,
                    &mut active,
                    &mut Vec::new(),
                    &mut Vec::new(),
                    &handle.control,
                );
                drain_terminal(&mut engine, &mut active, &handle.control);
                assert_counts(&metrics, 1, 1, 0);
            }
            let mut owner = InferenceCompletionOwner::attach(&mut engine);
            assert!(
                cancel_all(&mut engine, &mut owner, &mut active, &handle.control)
                    .await
                    .is_empty()
            );
            assert!(active.is_empty());
            assert_counts(&metrics, 1, 0, 1);
            // Ending the worker observation does not prove physical cleanup.
            assert!(engine.requests.contains_key(&id));
            pr13_assert_counts(&handle.control.admission, 1, 0);
            assert!(
                cancel_all(&mut engine, &mut owner, &mut active, &handle.control)
                    .await
                    .is_empty()
            );
            assert_counts(&metrics, 1, 0, 1);
        }

        #[tokio::test]
        async fn fatal_drain_finishes_once_even_if_cancellation_fails() {
            for fail_cancel in [false, true] {
                let metrics = Metrics::new();
                let (handle, mut receiver, _) = local_fixture();
                let mut engine = DisconnectEngine {
                    fail_cancel,
                    ..Default::default()
                };
                let mut active = HashMap::new();
                let mut events = admit(&handle, &mut receiver, &mut engine, &mut active, &metrics);
                assert_counts(&metrics, 1, 1, 0);
                let error = Arc::new(WorkerExecutionError::ModelExecution);
                handle.control.mark_fatal(Arc::clone(&error));
                let mut failures = Vec::new();
                for _ in 0..3 {
                    fail_all(
                        &mut engine,
                        &mut active,
                        Arc::clone(&error),
                        &mut failures,
                        &handle.control,
                    );
                    assert!(active.is_empty());
                    assert_counts(&metrics, 1, 0, 1);
                }
                assert_eq!(failures.len(), usize::from(fail_cancel));
                assert!(matches!(
                    events.recv().await,
                    Some(WorkerEvent::Failed { .. })
                ));
                assert!(events.recv().await.is_none());
                drop(events);
                assert_counts(&metrics, 1, 0, 1);
            }
        }

        #[tokio::test]
        async fn cancellation_failure_and_shutdown_drain_finish_once() {
            for shutdown in [false, true] {
                let metrics = Metrics::new();
                let (handle, mut receiver, _) = local_fixture();
                let mut engine = DisconnectEngine {
                    fail_cancel: true,
                    ..Default::default()
                };
                let mut active = HashMap::new();
                let events = admit(&handle, &mut receiver, &mut engine, &mut active, &metrics);
                drop(events);
                assert_counts(&metrics, 1, 1, 0);
                let mut failures = Vec::new();
                if shutdown {
                    handle.begin_shutdown();
                    let mut owner = InferenceCompletionOwner::attach(&mut engine);
                    failures.extend(
                        cancel_all(&mut engine, &mut owner, &mut active, &handle.control).await,
                    );
                } else {
                    cancel_disconnected(
                        &mut engine,
                        &mut active,
                        &mut Vec::new(),
                        &mut failures,
                        &handle.control,
                    );
                }
                assert_eq!(failures.len(), 1);
                if !shutdown {
                    assert_counts(&metrics, 1, 1, 0);
                    assert_eq!(handle.phase(), WorkerPhase::Draining);
                    let request = engine.request.take().unwrap();
                    assert!(active[&request.id].cancellation_failed);
                    engine.cancelled.push(SequenceState::from_request(
                        &request,
                        request.session_id.unwrap(),
                    ));
                    drain_terminal(&mut engine, &mut active, &handle.control);
                }
                assert_eq!(engine.cancellation_count.load(Ordering::Acquire), 1);
                assert!(active.is_empty());
                assert_counts(&metrics, 1, 0, 1);
                drain_terminal(&mut engine, &mut active, &handle.control);
                assert_counts(&metrics, 1, 0, 1);
                pr13_assert_counts(&handle.control.admission, 1, 0);
            }
        }
    }

    #[tokio::test]
    async fn runtime_snapshot_is_read_only_and_does_not_acquire_a_permit() {
        let (mut handle, mut receiver, _) = local_fixture();
        handle.control = Arc::new(ControlState::with_config(&WorkerConfig {
            max_inflight_requests: 1,
            max_prompt_bytes: 7,
            ..Default::default()
        }));
        handle.control.mark_ready();
        let permit = handle.try_acquire_admission(7).unwrap();
        assert_eq!(handle.admission_snapshot().available_requests, 0);
        let before = handle.admission_snapshot();
        let metrics = Metrics::new();
        let mut engine = Pr13Engine {
            held: 2,
            ..Default::default()
        };
        let mut active = HashMap::new();
        let mut future = Box::pin(handle.runtime_admission_snapshot());
        assert!(poll_once(future.as_mut()).is_pending());
        assert!(matches!(
            handle.runtime_admission_snapshot().await,
            Err(WorkerRequestError::QueueFull)
        ));
        assert!(!handle_command_with_metrics(
            receiver.try_recv().unwrap(),
            &mut engine,
            &mut active,
            &handle.control,
            &metrics
        ));
        let snapshot = future.await.unwrap().unwrap();
        assert_eq!(snapshot.request_identities_held, 2);
        assert_eq!(snapshot.session_identities_held, 2);
        assert_eq!(snapshot.waiting_requests, 0);
        assert_eq!(engine.held, 2);
        assert_eq!(engine.submits, 0);
        assert_eq!(engine.cancel_calls, 0);
        assert!(active.is_empty());
        let observed = metrics.snapshot();
        assert_eq!(observed.total_requests, 0);
        assert_eq!(observed.active_requests, 0);
        assert_eq!(observed.finished_requests, 0);
        assert_eq!(handle.admission_snapshot(), before);
        drop(permit);
        pr13_assert_counts(&handle.control.admission, 0, 0);
    }

    #[tokio::test]
    async fn runtime_snapshot_http_queue_timeout_and_closed_reasons_are_compatible() {
        use axum::{body::Body, http::Request};
        use http_body_util::BodyExt;
        use tower::ServiceExt;
        for reason in [
            "command_queue_full",
            "admission_timeout",
            "worker_unavailable",
        ] {
            let (mut handle, mut receiver, _) = local_fixture();
            handle.config.admission_timeout = std::time::Duration::from_millis(1);
            let (response, _reply) = oneshot::channel();
            if reason == "command_queue_full" {
                handle
                    .commands
                    .try_send(WorkerCommand::AdmissionSnapshot(response))
                    .unwrap();
            } else if reason == "worker_unavailable" {
                receiver.close();
            }
            let app = crate::router(crate::ServerState::new(
                crate::ModelRegistration::new("test", ferrule_model::ChatTemplate::Plain),
                handle.clone(),
            ));
            let response = app
                .oneshot(
                    Request::builder()
                        .uri("/admission")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), axum::http::StatusCode::OK);
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["runtime"]["status"], "unavailable");
            assert_eq!(value["runtime"]["reason"], reason);
            pr13_assert_counts(&handle.control.admission, 0, 0);
            if reason == "admission_timeout" {
                let WorkerCommand::AdmissionSnapshot(response) = receiver.try_recv().unwrap()
                else {
                    panic!("snapshot command")
                };
                assert!(response.is_closed());
            }
        }
    }

    #[test]
    fn admission_config_overflow_is_rejected_before_channel_or_factory() {
        for field in 0..4 {
            let mut config = WorkerConfig::default();
            match field {
                0 => config.command_queue_capacity = usize::MAX,
                1 => config.event_queue_capacity = usize::MAX,
                2 => config.max_body_bytes = usize::MAX,
                3 => {
                    config.runtime_admission_options =
                        Some(ferrule_runtime::RuntimeAdmissionOptions {
                            max_waiting_requests: 2,
                            max_request_identities: 1,
                            max_session_identities: 1,
                        })
                }
                _ => unreachable!(),
            }
            let result = spawn_model_worker_with(
                || -> Result<Pr13Engine, std::convert::Infallible> {
                    panic!("invalid config reached factory")
                },
                config,
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn pr13_invalid_config_never_calls_factory() {
        let called = Arc::new(AtomicBool::new(false));
        let probe = called.clone();
        let result = spawn_model_worker_with(
            move || {
                probe.store(true, Ordering::Release);
                Ok::<_, std::convert::Infallible>(Pr13Engine::default())
            },
            WorkerConfig {
                max_inflight_requests: 0,
                ..Default::default()
            },
        );
        assert!(result.is_err());
        assert!(!called.load(Ordering::Acquire));
    }
}
