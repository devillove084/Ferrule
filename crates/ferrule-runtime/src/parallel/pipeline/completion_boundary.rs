//! Narrow pipeline completion evidence and payload-free boundary diagnostics.
//! IPC byte counts and encode/pipe timings remain owned by process::ipc.

use std::error::Error as StdError;
use std::time::Instant;

use ferrule_common::error::ErrorChain;
use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Error, Result};
use ferrule_model::decoder::{CompletionUnknown, KvPrepareQuiescenceUnknown};

use super::PipelineRank;
use crate::parallel::data::CompletionOutcome;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompletionClass {
    Failed,
    Cancelled,
    Panicked,
    Unavailable,
    Unknown,
}

impl CompletionClass {
    fn label(self) -> &'static str {
        match self {
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Panicked => "panicked",
            Self::Unavailable => "unavailable",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompletionContext {
    pub rank: PipelineRank,
    /// Owner operation (prepare/execute/rollback/startup/shutdown), not payload.
    pub stage: &'static str,
    pub transaction: Option<ExecutionTransactionId>,
}

#[derive(Debug)]
pub(crate) struct CompletionFailure {
    pub context: CompletionContext,
    pub class: CompletionClass,
    pub quarantine: bool,
    source: Option<Box<dyn StdError + Send + Sync>>,
}

impl std::fmt::Display for CompletionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "pipeline {} rank {} transaction {:?}: {} (quarantine={})",
            self.context.stage,
            self.context.rank.global.get(),
            self.context.transaction.map(ExecutionTransactionId::get),
            self.class.label(),
            self.quarantine,
        )?;
        if let Some(source) = &self.source {
            write!(f, ": {source}")?;
        }
        Ok(())
    }
}
impl StdError for CompletionFailure {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_deref().map(|source| source as &dyn StdError)
    }
}

impl CompletionContext {
    pub(crate) fn failure(
        self,
        class: CompletionClass,
        quarantine: bool,
        source: Option<Box<dyn StdError + Send + Sync>>,
    ) -> Error {
        // A wrapper may strengthen evidence, never hide Unknown in a source.
        let class = if source.as_deref().is_some_and(|e| is_unknown(e)) {
            CompletionClass::Unknown
        } else {
            class
        };
        Error::ModelSource {
            source: Box::new(CompletionFailure {
                context: self,
                class,
                quarantine: quarantine || class == CompletionClass::Unknown,
                source,
            }),
        }
    }

    pub(crate) fn completion<T>(self, outcome: CompletionOutcome<T, Error>) -> Result<T> {
        let (class, quarantine, source) = match outcome {
            CompletionOutcome::Success(value) => return Ok(value),
            CompletionOutcome::Failed(source) => (classify(&source), false, Some(source)),
            CompletionOutcome::Cancelled => (CompletionClass::Cancelled, true, None),
            CompletionOutcome::Panicked => (CompletionClass::Panicked, true, None),
            CompletionOutcome::ReplicaUnavailable => (CompletionClass::Unavailable, true, None),
            CompletionOutcome::QuiescenceUnknown => (CompletionClass::Unknown, true, None),
        };
        Err(self.failure(class, quarantine, source.map(|e| Box::new(e) as _)))
    }
}

// A subsequent call has no original command identity; the original failure is
// returned intact to its caller. This marker cannot be mistaken for a fence.
#[derive(Debug)]
struct Quarantined;
impl std::fmt::Display for Quarantined {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("pipeline quarantined; quiescence unknown, custody retained")
    }
}
impl StdError for Quarantined {}
pub(crate) fn quarantined() -> Error {
    Error::ModelSource {
        source: Box::new(Quarantined),
    }
}

pub(crate) fn is_unknown(source: &(dyn StdError + 'static)) -> bool {
    ErrorChain::new(source).any(|node| {
        if node.is::<CompletionUnknown>()
            || node.is::<Quarantined>()
            || node.is::<KvPrepareQuiescenceUnknown>()
            || node
                .downcast_ref::<CompletionFailure>()
                .is_some_and(|e| e.class == CompletionClass::Unknown)
        {
            return true;
        }
        #[cfg(unix)]
        {
            use crate::parallel::process::{ProcessError, ProcessHandlerError, ProcessQuiescence};
            if node
                .downcast_ref::<ProcessError>()
                .is_some_and(ProcessError::is_quiescence_unknown)
                || node
                    .downcast_ref::<ProcessHandlerError>()
                    .is_some_and(|e| e.quiescence == ProcessQuiescence::Unknown)
            {
                return true;
            }
        }
        false
    })
}

pub(crate) fn classify(source: &(dyn StdError + 'static)) -> CompletionClass {
    if is_unknown(source) {
        CompletionClass::Unknown
    } else {
        ErrorChain::new(source)
            .downcast_ref::<CompletionFailure>()
            .map_or(CompletionClass::Failed, |e| e.class)
    }
}

pub(crate) fn needs_quarantine(source: &Error) -> bool {
    is_unknown(source)
        || ErrorChain::new(source).any(|node| {
            node.downcast_ref::<CompletionFailure>()
                .is_some_and(|e| e.quarantine)
        })
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn checked_cuda_completion<T>(quarantined: bool, result: Result<T>) -> Result<T> {
    if quarantined || result.as_ref().err().is_some_and(|e| is_unknown(e)) {
        std::mem::forget(result);
        std::panic::panic_any(crate::parallel::data::PanicQuiescence::Unknown);
    }
    result
}

pub(crate) struct Boundary {
    context: CompletionContext,
    operation: &'static str,
    start: Option<Instant>,
}
impl Boundary {
    pub(crate) fn new(operation: &'static str, context: CompletionContext) -> Self {
        Self {
            context,
            operation,
            start: tracing::enabled!(target: "ferrule_runtime::parallel::boundary", tracing::Level::DEBUG)
                .then(Instant::now),
        }
    }

    pub(crate) fn finish<T>(self, result: &Result<T>) {
        let Some(start) = self.start else { return };
        let outcome = result
            .as_ref()
            .err()
            .map_or("success", |e| classify(e).label());
        // No error Display/Debug, command, reply, session, token or path fields.
        // duration is wall-clock nanoseconds; it is not an additive IPC metric.
        tracing::debug!(
            target: "ferrule_runtime::parallel::boundary",
            operation = self.operation,
            rank = self.context.rank.global.get(),
            phase = self.context.stage,
            duration = start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            outcome,
        );
    }
}
