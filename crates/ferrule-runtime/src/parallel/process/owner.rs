use std::fmt;
use std::io;
use std::marker::PhantomData;
use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use ferrule_common::execution::ExecutionTransactionId;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::ipc::{self, Message};
use super::reaper::{self, Slot};
use super::{
    ProcessCommandId, ProcessCommandIdentity, ProcessError, ProcessIdentity, ProcessLaunch,
    ProcessOwnerConfig, ProcessOwnerState, ProcessQuiescence, ProcessTerminationReport,
    ReapOutcome, UnknownCause, deadline_after, protocol,
};

impl ProcessError {
    /// Classify errors returned by `ProcessRankOwner::execute[_observed]` only.
    /// These refusals preserve READY: nothing was sent, or the child explicitly
    /// fenced its failure. Post-send decode/limit/deadline errors are wrapped in
    /// QuiescenceUnknown and MUST NOT be classified by their nested source.
    pub fn is_command_rejection(&self) -> bool {
        match self {
            Self::InvalidConfig { .. }
            | Self::Json { .. }
            | Self::FrameTooLarge { .. }
            | Self::Deadline
            | Self::IdentityExhausted
            | Self::Cancelled => true,
            Self::RemoteFailure { failure } => failure.quiescence == ProcessQuiescence::Fenced,
            _ => false,
        }
    }
}

/// An unreaped startup child remains in this error. Dropping the error hands it
/// to the bounded reaper, rather than losing the Child or waiting indefinitely.
pub struct ProcessStartError<C, I, O> {
    pub source: ProcessError,
    pub owner: Option<Box<ProcessRankOwner<C, I, O>>>,
}
impl<C, I, O> fmt::Debug for ProcessStartError<C, I, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProcessStartError")
            .field("source", &self.source)
            .field("owner", &self.owner)
            .finish()
    }
}
impl<C, I, O> fmt::Display for ProcessStartError<C, I, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.source.fmt(f)
    }
}
impl<C, I, O> std::error::Error for ProcessStartError<C, I, O> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}
impl<C, I, O> From<ProcessError> for ProcessStartError<C, I, O> {
    fn from(source: ProcessError) -> Self {
        Self {
            source,
            owner: None,
        }
    }
}

/// One synchronous process rank. Only host config/input/output are serialized.
/// Loss invalidates the owner permanently; there is no automatic replay.
pub struct ProcessRankOwner<C, I, O> {
    identity: ProcessIdentity,
    options: ProcessOwnerConfig,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout: Option<ChildStdout>,
    slot: Option<Slot>,
    state: ProcessOwnerState,
    next_command: u64,
    exit_status: Option<ExitStatus>,
    reap_error: Option<io::Error>,
    shutdown_complete: bool,
    own_group: bool,
    _types: PhantomData<fn(C, I) -> O>,
}
impl<C, I, O> fmt::Debug for ProcessRankOwner<C, I, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProcessRankOwner")
            .field("identity", &self.identity)
            .field("state", &self.state)
            .field("pid", &self.pid())
            .finish_non_exhaustive()
    }
}

impl<C: Serialize, I: Serialize, O: DeserializeOwned> ProcessRankOwner<C, I, O> {
    pub fn spawn(
        launch: ProcessLaunch,
        identity: ProcessIdentity,
        config: &C,
        options: ProcessOwnerConfig,
    ) -> Result<Self, ProcessStartError<C, I, O>> {
        Self::spawn_group(launch, identity, config, options, true)
    }

    // EP children stay in the PP leader's isolation group. Their direct owner
    // signals only the EP process; the root can terminate the entire PP tree.
    pub(super) fn spawn_inherited(
        launch: ProcessLaunch,
        identity: ProcessIdentity,
        config: &C,
        options: ProcessOwnerConfig,
    ) -> Result<Self, ProcessStartError<C, I, O>> {
        Self::spawn_group(launch, identity, config, options, false)
    }

    fn spawn_group(
        launch: ProcessLaunch,
        identity: ProcessIdentity,
        config: &C,
        options: ProcessOwnerConfig,
        own_group: bool,
    ) -> Result<Self, ProcessStartError<C, I, O>> {
        options.validate()?;
        let deadline = deadline_after(options.startup_timeout)?;
        let config = ipc::payload(config, options.frame_limits.max_config_bytes)?;
        let boot = ipc::encode(
            Message::Boot {
                identity,
                limits: options.frame_limits,
                config,
            },
            options.frame_limits.max_frame_bytes,
        )?;
        let slot = Slot::reserve()?;
        ipc::check_deadline(deadline)?;
        let mut command = Command::new(launch.executable);
        command
            .args(launch.args)
            .envs(launch.environment)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if own_group {
            command.process_group(0);
        }
        let mut child = command.spawn().map_err(|source| ProcessError::Io {
            operation: "spawn process rank",
            source,
        })?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        // Child custody is established before any fallible descriptor setup.
        let mut owner = Self {
            identity,
            options,
            child: Some(child),
            stdin,
            stdout,
            slot: Some(slot),
            state: ProcessOwnerState::Starting,
            next_command: 1,
            exit_status: None,
            reap_error: None,
            shutdown_complete: false,
            own_group,
            _types: PhantomData,
        };
        let startup = (|| {
            ipc::nonblocking(
                owner
                    .stdin
                    .as_ref()
                    .ok_or(ProcessError::OwnerUnavailable)?
                    .as_raw_fd(),
            )?;
            ipc::nonblocking(
                owner
                    .stdout
                    .as_ref()
                    .ok_or(ProcessError::OwnerUnavailable)?
                    .as_raw_fd(),
            )?;
            owner.send_frame(&boot, deadline)?;
            match owner.receive(deadline)? {
                Message::Ready {
                    identity: actual,
                    limits,
                } if actual == identity && limits == options.frame_limits => Ok(()),
                Message::Failure {
                    owner: actual,
                    command: None,
                    session: None,
                    failure,
                } if actual == identity
                    && failure.message.len() <= options.frame_limits.max_error_bytes =>
                {
                    Err(ProcessError::RemoteFailure { failure })
                }
                _ => Err(protocol("invalid startup reply identity/type/limits")),
            }
        })();
        match startup {
            Ok(()) => {
                owner.state = ProcessOwnerState::Ready;
                Ok(owner)
            }
            Err(source) => {
                let source = owner.fail(source, UnknownCause::Startup);
                let owner = owner.child.is_some().then(|| Box::new(owner));
                Err(ProcessStartError { source, owner })
            }
        }
    }

    pub fn execute(
        &mut self,
        transaction: ExecutionTransactionId,
        session: u64,
        input: &I,
    ) -> Result<O, ProcessError> {
        self.execute_observed(transaction, session, input, &mut || {})
    }

    /// Observe host cancellation/liveness during bounded pipe waits. Observation
    /// does not interrupt admitted child work or turn process exit into a fence.
    /// A panicking observer invalidates the owner before unwinding.
    pub fn execute_observed(
        &mut self,
        transaction: ExecutionTransactionId,
        session: u64,
        input: &I,
        observe: &mut dyn FnMut(),
    ) -> Result<O, ProcessError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.execute_inner(transaction, session, input, observe)
        })) {
            Ok(result) => result,
            Err(payload) => {
                let _ = self.fail(
                    protocol("panic during observed process command"),
                    UnknownCause::TransportFailure,
                );
                std::panic::resume_unwind(payload)
            }
        }
    }

    fn execute_inner(
        &mut self,
        transaction: ExecutionTransactionId,
        session: u64,
        input: &I,
        observe: &mut dyn FnMut(),
    ) -> Result<O, ProcessError> {
        if self.state != ProcessOwnerState::Ready {
            return Err(ProcessError::OwnerUnavailable);
        }

        let deadline = deadline_after(self.options.command_timeout)?;
        let next = self
            .next_command
            .checked_add(1)
            .ok_or(ProcessError::IdentityExhausted)?;
        let key = ProcessCommandIdentity {
            owner: self.identity,
            command: ProcessCommandId::new(self.next_command)?,
            transaction: transaction.into(),
        };
        let payload = ipc::payload(input, self.options.frame_limits.max_command_bytes)?;
        let frame = ipc::encode(
            Message::Execute {
                identity: key,
                session,
                payload,
            },
            self.options.frame_limits.max_frame_bytes,
        )?;
        ipc::check_deadline(deadline)?;
        let mut sent = false;
        let result = (|| {
            ipc::send_observed(
                self.stdin.as_mut().ok_or(ProcessError::OwnerUnavailable)?,
                &frame,
                deadline,
                observe,
                &mut sent,
            )?;
            match ipc::receive_observed(
                self.stdout.as_mut().ok_or(ProcessError::OwnerUnavailable)?,
                self.options.frame_limits.max_frame_bytes,
                deadline,
                observe,
            )? {
                Message::Complete {
                    identity,
                    session: actual,
                    payload,
                } if identity == key && actual == session => {
                    ipc::serialize(&payload, self.options.frame_limits.max_output_bytes)?;
                    let output =
                        serde_json::from_value(payload).map_err(|source| ProcessError::Json {
                            operation: "decode process output",
                            source,
                        })?;
                    ipc::check_deadline(deadline)?;
                    Ok(output)
                }
                Message::Failure {
                    owner,
                    command: Some(command),
                    session: Some(actual),
                    failure,
                } if owner == self.identity
                    && command == key
                    && actual == session
                    && failure.message.len() <= self.options.frame_limits.max_error_bytes =>
                {
                    Err(ProcessError::RemoteFailure { failure })
                }
                _ => Err(protocol("invalid completion identity/type/size")),
            }
        })();
        if sent {
            self.next_command = next;
        }
        match result {
            Ok(output) => Ok(output),
            // Local refusal before the first prefix byte cannot admit work.
            // Transport failures (e.g. EPIPE) still invalidate a lost peer.
            Err(error) if !sent && error.is_command_rejection() => Err(error),
            Err(ProcessError::RemoteFailure { failure })
                if failure.quiescence == ProcessQuiescence::Fenced =>
            {
                Err(ProcessError::RemoteFailure { failure })
            }
            Err(error) => Err(self.fail(error, UnknownCause::TransportFailure)),
        }
    }
}

impl<C, I, O> ProcessRankOwner<C, I, O> {
    pub const fn identity(&self) -> ProcessIdentity {
        self.identity
    }
    pub const fn state(&self) -> ProcessOwnerState {
        self.state
    }
    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(Child::id)
    }
    pub fn exit_status(&self) -> Option<ExitStatus> {
        self.exit_status
    }
    pub fn is_quarantined(&self) -> bool {
        self.state == ProcessOwnerState::Quarantined
    }

    /// Only a validated shutdown acknowledgement AND reap count as graceful
    /// shutdown. Forced termination never converts a lost fence into success.
    pub fn shutdown(&mut self) -> Result<ProcessTerminationReport, ProcessError> {
        if self.shutdown_complete {
            return Ok(reaped_report());
        }
        if self.state != ProcessOwnerState::Ready {
            return Err(self.fail(ProcessError::OwnerUnavailable, UnknownCause::Shutdown));
        }
        let result = (|| {
            let deadline = deadline_after(self.options.command_timeout)?;
            let frame = ipc::encode(
                Message::Shutdown {
                    owner: self.identity,
                },
                self.options.frame_limits.max_frame_bytes,
            )?;
            self.send_frame(&frame, deadline)?;
            match self.receive(deadline)? {
                Message::ShutdownAck { owner } if owner == self.identity => {}
                Message::Failure {
                    owner,
                    command: None,
                    session: None,
                    failure,
                } if owner == self.identity
                    && failure.message.len() <= self.options.frame_limits.max_error_bytes =>
                {
                    // Preserve the child's fence/custody evidence, but never
                    // treat a shutdown failure (even Fenced) as a ShutdownAck.
                    return Err(ProcessError::RemoteFailure { failure });
                }
                _ => return Err(protocol("missing valid shutdown acknowledgement")),
            }
            if !self.reap_until(deadline) {
                return Err(ProcessError::Deadline);
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.shutdown_complete = true;
                Ok(reaped_report())
            }
            Err(source) => Err(self.fail(source, UnknownCause::Shutdown)),
        }
    }

    /// Stop admission, close pipes, TERM then KILL, and poll reap with finite
    /// grace periods. The report concerns OS state only, never device fences.
    pub fn terminate(&mut self) -> ProcessTerminationReport {
        if self.state == ProcessOwnerState::Reaped {
            return reaped_report();
        }
        self.state = ProcessOwnerState::Quarantined;
        self.stdin.take();
        self.stdout.take();
        let mut signal_errors = Vec::new();
        if let Some(child) = &self.child {
            if let Err(error) = reaper::signal_owned(child, libc::SIGTERM, self.own_group) {
                signal_errors.push(error);
            }
            // Keep the leader unreaped until the final group signal. Otherwise
            // an early leader exit could leave peers alive, or permit PID/PGID
            // reuse before our later SIGKILL. No blocking wait is involved.
            thread::sleep(self.options.terminate_grace);
            if let Err(error) = reaper::signal_owned(child, libc::SIGKILL, self.own_group) {
                signal_errors.push(error);
            }
        }
        let kill_end = Instant::now()
            .checked_add(self.options.kill_grace)
            .unwrap_or_else(Instant::now);
        self.reap_until(kill_end);
        let reap = match self.pid() {
            Some(pid) => ReapOutcome::NotReaped { pid },
            None => ReapOutcome::Reaped,
        };
        ProcessTerminationReport {
            reap,
            signal_errors,
            reap_error: self.reap_error.take(),
        }
    }

    /// Nonblocking follow-up after termination, not a quiescence claim. Do not
    /// reap a live leader before its final group signal (PID reuse protection).
    pub fn poll_reap(&mut self) -> Result<ReapOutcome, ProcessError> {
        if matches!(
            self.state,
            ProcessOwnerState::Starting | ProcessOwnerState::Ready
        ) {
            return Err(protocol("terminate owner before polling reap"));
        }
        if let Some(reap) = self.try_reap() {
            return Ok(reap);
        }
        if let Some(source) = self.reap_error.take() {
            return Err(ProcessError::Io {
                operation: "try_wait process rank",
                source,
            });
        }
        Ok(ReapOutcome::NotReaped {
            pid: self.pid().expect("unreaped child retained"),
        })
    }

    fn send_frame(&mut self, frame: &[u8], deadline: Instant) -> Result<(), ProcessError> {
        ipc::send(
            self.stdin.as_mut().ok_or(ProcessError::OwnerUnavailable)?,
            frame,
            deadline,
        )
    }
    fn receive(&mut self, deadline: Instant) -> Result<Message, ProcessError> {
        ipc::receive(
            self.stdout.as_mut().ok_or(ProcessError::OwnerUnavailable)?,
            self.options.frame_limits.max_frame_bytes,
            deadline,
        )
    }
    fn try_reap(&mut self) -> Option<ReapOutcome> {
        let Some(child) = self.child.as_mut() else {
            return Some(ReapOutcome::Reaped);
        };
        match child.try_wait() {
            Ok(Some(status)) => {
                self.exit_status = Some(status);
                self.child.take();
                self.stdin.take();
                self.stdout.take();
                self.slot.take();
                self.state = ProcessOwnerState::Reaped;
                Some(ReapOutcome::Reaped)
            }
            Ok(None) => None,
            Err(error) => {
                self.reap_error = Some(error);
                None
            }
        }
    }
    fn reap_until(&mut self, deadline: Instant) -> bool {
        loop {
            if self.try_reap().is_some() {
                return true;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return false;
            }
            thread::sleep(remaining.min(Duration::from_millis(2)));
        }
    }
    fn fail(&mut self, source: ProcessError, fallback: UnknownCause) -> ProcessError {
        let cause = match &source {
            ProcessError::Deadline => UnknownCause::Deadline,
            ProcessError::PipeClosed => UnknownCause::PeerExited,
            ProcessError::TruncatedFrame => UnknownCause::TruncatedFrame,
            ProcessError::Protocol { .. }
            | ProcessError::Json { .. }
            | ProcessError::FrameTooLarge { .. } => UnknownCause::ProtocolViolation,
            ProcessError::RemoteFailure { .. } => UnknownCause::Handler,
            _ => fallback,
        };
        let termination = self.terminate();
        ProcessError::QuiescenceUnknown {
            cause,
            source: Box::new(source),
            termination,
        }
    }
}

impl<C, I, O> Drop for ProcessRankOwner<C, I, O> {
    fn drop(&mut self) {
        if let Some(child) = self.child.take() {
            // No grace wait in Drop. The pre-reserved reaper slot takes custody.
            let _ = reaper::signal_owned(&child, libc::SIGKILL, self.own_group);
            self.stdin.take();
            self.stdout.take();
            self.slot
                .take()
                .expect("live child has reserved reaper slot")
                .adopt(child);
        }
    }
}
fn reaped_report() -> ProcessTerminationReport {
    ProcessTerminationReport {
        reap: ReapOutcome::Reaped,
        signal_errors: Vec::new(),
        reap_error: None,
    }
}
