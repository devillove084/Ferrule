//! Bounded Unix process isolation for rank-local workers.
//!
//! Children are re-executed from an explicit executable/argument vector, never
//! a shell. Only serialized host values cross the boundary. The caller owns
//! transaction decisions and publication; this module has no transaction state
//! machine and never replays a command. Process death/reaping is NOT a GPU fence.
//!
//! Parent IPC deadlines include partial reads/writes and do not slide on
//! progress. Timeout/invalid replies invalidate the owner, escalate TERM/KILL,
//! and use only `try_wait`. Unreaped children remain owned; Drop transfers them
//! to a bounded process-wide reaper with capacity reserved before spawn.
//!
//! This is a synchronous API. OS `spawn/exec`, arbitrary user serializers, and
//! handler code cannot themselves be preempted by Rust deadlines. Run this on a
//! host control owner, never inside the CUDA owner being supervised. Children
//! must not daemonize, escape their process group, or share DMA memory with the
//! parent. D-state/device recovery requires external isolation/escalation.

mod child;
pub mod decoder;
mod decoder_endpoint;
mod decoder_expert;
mod decoder_stage;
pub mod decoder_wire;
mod ipc;
mod owner;
mod reaper;
mod supervisor;

use std::ffi::OsString;
use std::fmt;
use std::io;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ferrule_common::ParallelRankId;
use ferrule_common::execution::ExecutionTransactionId;
use serde::{Deserialize, Serialize};

pub use child::{ProcessBoot, ProcessChildHandler, ProcessRequest, child_serve, child_serve_with};
pub use owner::{ProcessRankOwner, ProcessStartError};
pub use supervisor::ProcessRankSupervisor;
/// Short name for the recovery-domain supervisor.
pub type ProcessSupervisor<C, I, O> = ProcessRankSupervisor<C, I, O>;

pub const PROCESS_PROTOCOL_VERSION: u16 = 1;
/// Maximum simultaneously owned children, including unreaped children handed
/// off by Drop. Capacity exhaustion fails before spawning another process.
pub const PROCESS_REAPER_CAPACITY: usize = 128;

macro_rules! process_id {
    ($name:ident) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(NonZeroU64);
        impl $name {
            pub fn new(value: u64) -> Result<Self, ProcessError> {
                NonZeroU64::new(value)
                    .map(Self)
                    .ok_or_else(|| ProcessError::InvalidConfig {
                        message: concat!(stringify!($name), " must be non-zero").into(),
                    })
            }
            pub const fn get(self) -> u64 {
                self.0.get()
            }
        }
    };
}
process_id!(ProcessGroupEpoch);
process_id!(ProcessOwnerInstanceId);
process_id!(ProcessCommandId);
process_id!(ProcessTransactionId);

impl From<ExecutionTransactionId> for ProcessTransactionId {
    fn from(value: ExecutionTransactionId) -> Self {
        Self(NonZeroU64::new(value.get()).expect("validated transaction ID"))
    }
}
impl From<ProcessTransactionId> for ExecutionTransactionId {
    fn from(value: ProcessTransactionId) -> Self {
        Self::new(value.get()).expect("nonzero wire transaction ID")
    }
}

/// Rank is logical, not a device ordinal. A caller creating independent
/// supervisors must allocate distinct epochs/owner identities for their domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessIdentity {
    pub epoch: ProcessGroupEpoch,
    pub rank: ParallelRankId,
    pub owner_instance: ProcessOwnerInstanceId,
}
impl ProcessIdentity {
    pub const fn new(
        epoch: ProcessGroupEpoch,
        rank: ParallelRankId,
        owner_instance: ProcessOwnerInstanceId,
    ) -> Self {
        Self {
            epoch,
            rank,
            owner_instance,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessCommandIdentity {
    pub owner: ProcessIdentity,
    pub command: ProcessCommandId,
    pub transaction: ProcessTransactionId,
}

/// Byte limits on encoded JSON, not element counts. These also bound decoding
/// inputs; serde's normal JSON recursion limit is not disabled. Envelope overhead
/// counts against max_frame_bytes in addition to the individual payload limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessFrameLimits {
    pub max_frame_bytes: usize,
    pub max_config_bytes: usize,
    pub max_command_bytes: usize,
    pub max_output_bytes: usize,
    pub max_error_bytes: usize,
}
impl Default for ProcessFrameLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 8 * 1024 * 1024,
            max_config_bytes: 4 * 1024 * 1024,
            max_command_bytes: 4 * 1024 * 1024,
            max_output_bytes: 4 * 1024 * 1024,
            max_error_bytes: 4096,
        }
    }
}
impl ProcessFrameLimits {
    pub fn validate(self) -> Result<(), ProcessError> {
        if !(1024..=ipc::HARD_MAX_FRAME_BYTES).contains(&self.max_frame_bytes) {
            return Err(invalid("max_frame_bytes must be between 1024 and 128 MiB"));
        }
        for size in [
            self.max_config_bytes,
            self.max_command_bytes,
            self.max_output_bytes,
            self.max_error_bytes,
        ] {
            if size == 0 || size > self.max_frame_bytes {
                return Err(invalid(
                    "payload/error limits must be nonzero and no greater than max_frame_bytes",
                ));
            }
        }
        Ok(())
    }
}

/// Executed directly, with inherited environment plus the explicit overrides.
/// stdin/stdout are private protocol pipes; stderr is inherited, never captured
/// in an undrained pipe. The executable must keep stdout exclusively for frames.
#[derive(Debug, Clone)]
pub struct ProcessLaunch {
    pub executable: PathBuf,
    pub args: Vec<OsString>,
    pub environment: Vec<(OsString, OsString)>,
}
impl ProcessLaunch {
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            args: Vec::new(),
            environment: Vec::new(),
        }
    }
    pub fn arg(mut self, argument: impl Into<OsString>) -> Self {
        self.args.push(argument.into());
        self
    }
    pub fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.environment.push((key.into(), value.into()));
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProcessOwnerConfig {
    pub frame_limits: ProcessFrameLimits,
    pub startup_timeout: Duration,
    pub command_timeout: Duration,
    pub terminate_grace: Duration,
    /// Maximum time to reap after SIGKILL, not a promise that SIGKILL succeeds.
    pub kill_grace: Duration,
}
impl Default for ProcessOwnerConfig {
    fn default() -> Self {
        Self {
            frame_limits: ProcessFrameLimits::default(),
            startup_timeout: Duration::from_secs(30),
            command_timeout: Duration::from_secs(30),
            terminate_grace: Duration::from_millis(100),
            kill_grace: Duration::from_millis(500),
        }
    }
}
impl ProcessOwnerConfig {
    pub fn validate(self) -> Result<(), ProcessError> {
        self.frame_limits.validate()?;
        for duration in [
            self.startup_timeout,
            self.command_timeout,
            self.terminate_grace,
            self.kill_grace,
        ] {
            deadline_after(duration)?;
        }
        Ok(())
    }
}

/// Child-local ceilings are trusted application configuration, not limits
/// supplied by a peer. Boot, frame, and reply waits are bounded. A healthy
/// command pipe may remain idle without an idle timeout; handler/initializer
/// execution is controlled by an independent parent deadline.
#[derive(Debug, Clone, Copy)]
pub struct ProcessChildConfig {
    pub frame_limits: ProcessFrameLimits,
    pub io_timeout: Duration,
}
impl Default for ProcessChildConfig {
    fn default() -> Self {
        Self {
            frame_limits: ProcessFrameLimits::default(),
            io_timeout: Duration::from_secs(300),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProcessSupervisorConfig {
    pub owner: ProcessOwnerConfig,
    pub max_ranks: usize,
    pub max_restarts: usize,
    /// Reserve enough slots for every live rank to become unreapable at once.
    /// Must be >= max_ranks; this is not permission to abandon old handles.
    pub max_quarantined: usize,
}
impl Default for ProcessSupervisorConfig {
    fn default() -> Self {
        Self {
            owner: ProcessOwnerConfig::default(),
            max_ranks: 8,
            max_restarts: 3,
            max_quarantined: 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessOwnerState {
    Starting,
    Ready,
    Reaped,
    Quarantined,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownCause {
    Startup,
    Deadline,
    PeerExited,
    TruncatedFrame,
    ProtocolViolation,
    TransportFailure,
    Handler,
    Shutdown,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReapOutcome {
    Reaped,
    NotReaped { pid: u32 },
}

/// This report records OS evidence ONLY. Even Reaped says nothing about CUDA
/// completion or whether the device may be reused.
#[derive(Debug)]
pub struct ProcessTerminationReport {
    pub reap: ReapOutcome,
    pub signal_errors: Vec<io::Error>,
    pub reap_error: Option<io::Error>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessFailureKind {
    Startup,
    CommandDecode,
    Handler,
    OutputEncode,
    Shutdown,
    Panic,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessQuiescence {
    Fenced,
    Unknown,
}

/// Portable failure evidence, not a serialized Rust error object. Fenced is an
/// explicit handler promise that ALL asynchronous accesses have stopped. Use
/// Unknown for failed fences; the child then retains the handler without Drop.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessHandlerError {
    pub kind: ProcessFailureKind,
    pub quiescence: ProcessQuiescence,
    pub message: String,
}
impl ProcessHandlerError {
    pub fn fenced(kind: ProcessFailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            quiescence: ProcessQuiescence::Fenced,
            message: message.into(),
        }
    }
    pub fn unknown(kind: ProcessFailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            quiescence: ProcessQuiescence::Unknown,
            message: message.into(),
        }
    }
}
impl fmt::Display for ProcessHandlerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} ({:?}): {}",
            self.kind, self.quiescence, self.message
        )
    }
}
impl std::error::Error for ProcessHandlerError {}

#[derive(Debug)]
pub enum ProcessError {
    InvalidConfig {
        message: String,
    },
    Io {
        operation: &'static str,
        source: io::Error,
    },
    Json {
        operation: &'static str,
        source: serde_json::Error,
    },
    FrameTooLarge {
        limit: usize,
    },
    PipeClosed,
    TruncatedFrame,
    Deadline,
    Protocol {
        message: String,
    },
    RemoteFailure {
        failure: ProcessHandlerError,
    },
    QuiescenceUnknown {
        cause: UnknownCause,
        source: Box<ProcessError>,
        termination: ProcessTerminationReport,
    },
    OwnerUnavailable,
    Cancelled,
    GroupInvalidated,
    RankAlreadySpawned {
        rank: ParallelRankId,
    },
    RankNotFound {
        rank: ParallelRankId,
    },
    RestartBudgetExhausted,
    RestartBlocked {
        unreaped: usize,
    },
    Capacity {
        resource: &'static str,
        limit: usize,
    },
    IdentityExhausted,
}
impl ProcessError {
    pub fn is_quiescence_unknown(&self) -> bool {
        matches!(self, Self::QuiescenceUnknown { .. })
    }
}
impl fmt::Display for ProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { message } => write!(f, "invalid process config: {message}"),
            Self::Io { operation, source } => write!(f, "{operation}: {source}"),
            Self::Json { operation, source } => write!(f, "{operation}: {source}"),
            Self::FrameTooLarge { limit } => {
                write!(f, "process frame/payload exceeds {limit} bytes")
            }
            Self::PipeClosed => f.write_str("process pipe closed"),
            Self::TruncatedFrame => f.write_str("truncated process frame"),
            Self::Deadline => f.write_str("process deadline expired"),
            Self::Protocol { message } => write!(f, "process protocol violation: {message}"),
            Self::RemoteFailure { failure } => write!(f, "child: {failure}"),
            Self::QuiescenceUnknown {
                cause,
                source,
                termination,
            } => write!(
                f,
                "process quiescence unknown ({cause:?}, {:?}): {source}",
                termination.reap
            ),
            Self::OwnerUnavailable => f.write_str("process owner unavailable"),
            Self::Cancelled => f.write_str("command cancelled before dispatch"),
            Self::GroupInvalidated => f.write_str("process group epoch invalidated"),
            Self::RankAlreadySpawned { rank } => {
                write!(f, "rank {rank:?} already attempted in this epoch")
            }
            Self::RankNotFound { rank } => write!(f, "rank {rank:?} not found"),
            Self::RestartBudgetExhausted => f.write_str("process restart budget exhausted"),
            Self::RestartBlocked { unreaped } => {
                write!(f, "restart blocked: {unreaped} unreaped children")
            }
            Self::Capacity { resource, limit } => write!(f, "{resource} capacity reached: {limit}"),
            Self::IdentityExhausted => f.write_str("process identity exhausted"),
        }
    }
}
impl std::error::Error for ProcessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::RemoteFailure { failure } => Some(failure),
            Self::QuiescenceUnknown { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

fn invalid(message: &str) -> ProcessError {
    ProcessError::InvalidConfig {
        message: message.into(),
    }
}
fn protocol(message: &str) -> ProcessError {
    ProcessError::Protocol {
        message: message.into(),
    }
}
fn deadline_after(duration: Duration) -> Result<Instant, ProcessError> {
    if duration.is_zero() {
        return Err(invalid("timeouts must be nonzero"));
    }
    Instant::now()
        .checked_add(duration)
        .ok_or_else(|| invalid("timeout overflows Instant"))
}
