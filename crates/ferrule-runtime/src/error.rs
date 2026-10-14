use snafu::Snafu;
use std::sync::Arc;

use ferrule_common::io_protocol::{CancellationReason, FailureReason, StaleReason};
use ferrule_common::{IoProtocolError, MaterializationResolveError, MaterializationResourceError};

use crate::io::{FairQueueError, RegistryError};
use crate::scheduling::PhysicalResourceError;

/// Runtime orchestration error.
///
/// Subsystem failures remain typed sources. Runtime-owned control failures are
/// deliberately limited to rejected requests, violated invariants, and a typed
/// primary/cleanup pair; transaction phases are represented by ownership state,
/// not by a parallel hierarchy of error variants.
#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(transparent)]
    Admission {
        source: crate::engine::RuntimeAdmissionError,
    },

    #[snafu(transparent)]
    Backend { source: ferrule_common::Error },

    #[snafu(transparent)]
    Protocol { source: IoProtocolError },

    #[snafu(transparent)]
    Materialization { source: MaterializationResolveError },

    #[snafu(transparent)]
    MaterializationResources {
        source: MaterializationResourceError,
    },

    #[snafu(transparent)]
    PhysicalResources { source: PhysicalResourceError },

    #[snafu(transparent)]
    Registry {
        #[snafu(source(from(RegistryError, Box::new)))]
        source: Box<RegistryError>,
    },

    #[snafu(transparent)]
    Fairness { source: FairQueueError },

    #[snafu(display("background warmup materialization failed: {source}"))]
    WarmupMaterialization { source: FailureReason },

    #[snafu(display("background warmup materialization was cancelled: {reason:?}"))]
    WarmupMaterializationCancelled { reason: CancellationReason },

    #[snafu(display("background warmup materialization became stale: {reason:?}"))]
    WarmupMaterializationStale { reason: StaleReason },

    #[snafu(display("background warmup retired without publishing residency"))]
    WarmupMaterializationNotPublished,

    #[snafu(display("invalid runtime request: {message}"))]
    InvalidRequest { message: String },

    #[snafu(display("resident request capacity ({limit}) is exhausted"))]
    RequestCapacity { limit: usize },

    #[snafu(display("inference engine is closed or quarantined"))]
    EngineUnavailable,

    #[snafu(display("runtime invariant violated: {message}"))]
    Invariant { message: String },

    #[snafu(display("runtime shutdown is incomplete: {message}"))]
    ShutdownIncomplete { message: String },

    /// An immutable failure shared by a retained cleanup owner and its caller.
    /// This carries diagnostics only, never resource custody or cleanup authority.
    #[snafu(display("{source}"))]
    Shared { source: Arc<Error> },

    #[snafu(display("{operation} failed: {source}; cleanup also failed: {cleanup}"))]
    Cleanup {
        operation: &'static str,
        source: Box<Error>,
        cleanup: Box<Error>,
    },

    #[snafu(display(
        "{operation} failed: {source}; {} cleanup steps also failed",
        cleanup.len()
    ))]
    CleanupBatch {
        operation: &'static str,
        source: Box<Error>,
        cleanup: Vec<CleanupStep>,
    },

    #[snafu(display(
        "{operation} encountered {} independent failures",
        failures.len()
    ))]
    FailureBatch {
        operation: &'static str,
        failures: Vec<Error>,
    },

    #[snafu(display(
        "{operation} completed, but {} cleanup steps failed",
        cleanup.len()
    ))]
    CleanupFailures {
        operation: &'static str,
        cleanup: Vec<CleanupStep>,
    },

    #[snafu(display("completion source closed while asynchronous work is live"))]
    CompletionSourceClosed,

    #[snafu(display("completion reactor stopped unexpectedly"))]
    CompletionReactorStopped,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub struct CleanupStep {
    pub operation: String,
    pub source: Error,
}

impl Error {
    /// The original runtime failure, excluding runtime cleanup branches.
    /// Common/backend sources remain typed and can be inspected with `ErrorChain`.
    /// Independent failure batches deliberately have no elected primary.
    pub fn primary(&self) -> &Self {
        match self {
            Self::Shared { source } => source.primary(),
            Self::Cleanup { source, .. } | Self::CleanupBatch { source, .. } => source.primary(),
            _ => self,
        }
    }

    /// Runtime cleanup branches, in occurrence order. These are diagnostics,
    /// not HTTP classification inputs or proof that cleanup succeeded.
    /// Each branch retains its typed source and any nested common error chain.
    pub fn cleanup_errors(&self) -> Vec<&Self> {
        let mut errors = Vec::new();
        self.collect_cleanup_errors(&mut errors);
        errors
    }

    fn collect_cleanup_errors<'a>(&'a self, errors: &mut Vec<&'a Self>) {
        match self {
            Self::Shared { source } => source.collect_cleanup_errors(errors),
            Self::Cleanup {
                source, cleanup, ..
            } => {
                source.collect_cleanup_errors(errors);
                errors.push(cleanup);
            }
            Self::CleanupBatch {
                source, cleanup, ..
            } => {
                source.collect_cleanup_errors(errors);
                errors.extend(cleanup.iter().map(|step| &step.source));
            }
            Self::CleanupFailures { cleanup, .. } => {
                errors.extend(cleanup.iter().map(|step| &step.source));
            }
            _ => {}
        }
    }

    /// Internal diagnostics across runtime and common primary/cleanup branches.
    /// Unlike `primary()`, this must never be used to choose HTTP policy.
    pub fn diagnostic_chain(
        &self,
    ) -> Box<dyn Iterator<Item = &(dyn std::error::Error + 'static)> + '_> {
        use ferrule_common::ErrorChain;
        let branches: Box<dyn Iterator<Item = &(dyn std::error::Error + 'static)> + '_> = match self
        {
            Self::Shared { source } => source.diagnostic_chain(),
            Self::Backend { source } => Box::new(ErrorChain::new(source)),
            Self::Cleanup {
                source, cleanup, ..
            } => Box::new(source.diagnostic_chain().chain(cleanup.diagnostic_chain())),
            Self::CleanupBatch {
                source, cleanup, ..
            } => Box::new(
                source.diagnostic_chain().chain(
                    cleanup
                        .iter()
                        .flat_map(|step| step.source.diagnostic_chain()),
                ),
            ),
            Self::CleanupFailures { cleanup, .. } => Box::new(
                cleanup
                    .iter()
                    .flat_map(|step| step.source.diagnostic_chain()),
            ),
            Self::FailureBatch { failures, .. } => {
                Box::new(failures.iter().flat_map(Self::diagnostic_chain))
            }
            _ => return Box::new(ErrorChain::new(self)),
        };
        Box::new(std::iter::once(self as &(dyn std::error::Error + 'static)).chain(branches))
    }

    /// Keep the complete failure at the existing retry owner while reporting the
    /// same snapshot now. Only the error path allocates or increments an Arc.
    pub(crate) fn retain_cleanup(&mut self, operation: &'static str, cleanup: Error) -> Self {
        let primary = std::mem::replace(self, Self::EngineUnavailable);
        let source = Arc::new(Self::cleanup(operation, primary, cleanup));
        *self = Self::Shared {
            source: Arc::clone(&source),
        };
        Self::Shared { source }
    }

    pub(crate) fn cleanup(
        operation: &'static str,
        source: impl Into<Error>,
        cleanup: impl Into<Error>,
    ) -> Self {
        Self::Cleanup {
            operation,
            source: Box::new(source.into()),
            cleanup: Box::new(cleanup.into()),
        }
    }

    pub(crate) fn with_cleanup(
        operation: &'static str,
        source: impl Into<Error>,
        cleanup: Result<()>,
    ) -> Self {
        let source = source.into();
        match cleanup {
            Ok(()) => source,
            Err(cleanup) => Self::cleanup(operation, source, cleanup),
        }
    }

    pub(crate) fn with_cleanup_batch(
        operation: &'static str,
        source: impl Into<Error>,
        cleanup: Vec<CleanupStep>,
    ) -> Self {
        let source = source.into();
        if cleanup.is_empty() {
            source
        } else {
            Self::CleanupBatch {
                operation,
                source: Box::new(source),
                cleanup,
            }
        }
    }

    pub(crate) fn combine(operation: &'static str, first: Error, second: Error) -> Self {
        let mut failures = match first {
            Self::FailureBatch {
                operation: existing,
                failures,
            } if existing == operation => failures,
            first => vec![first],
        };
        failures.push(second);
        Self::FailureBatch {
            operation,
            failures,
        }
    }
}

impl CleanupStep {
    pub(crate) fn new(operation: impl Into<String>, source: impl Into<Error>) -> Self {
        Self {
            operation: operation.into(),
            source: source.into(),
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/error.rs"]
mod tests;
