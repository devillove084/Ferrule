use snafu::Snafu;

use ferrule_common::io_protocol::{IoProtocolError, MaterializationResolveError};
use ferrule_common::materialization_io::MaterializationResourceError;

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

    #[snafu(display("invalid runtime request: {message}"))]
    InvalidRequest { message: String },

    #[snafu(display("runtime invariant violated: {message}"))]
    Invariant { message: String },

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

    pub(crate) fn cleanup_failures(
        operation: &'static str,
        cleanup: Vec<CleanupStep>,
    ) -> Result<()> {
        if cleanup.is_empty() {
            Ok(())
        } else {
            Err(Self::CleanupFailures { operation, cleanup })
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
