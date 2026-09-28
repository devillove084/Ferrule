//! Cross-crate compatibility error boundary and typed error domains.
//!
//! Domain types live in focused modules and remain re-exported here and at the
//! crate root. Existing message variants are legacy leaf APIs; new subsystem
//! failures should retain a typed source. Display is for internal diagnostics,
//! never a public HTTP message.

use snafu::Snafu;
use std::error::Error as StdError;

pub mod domain;
pub use domain::{ClassifiedError, ErrorChain, ErrorClass};

pub mod quantization;
pub use quantization::{QuantizationError, QuantizationResult};

pub mod serving;
pub use serving::{
    ServingConfigError, ServingRequestError, SseSerializationError, WorkerExecutionError,
    WorkerOperation, WorkerRequestError, WorkerShutdownError, WorkerStartError,
};

pub mod state_dict;
pub use state_dict::{
    NameMappingError, NameMappingFailureKind, StateDictBindingError, StateDictBindingIssue,
    StateDictIssues, StateDictMetadataError, StateDictSchemaError, StateDictTransformError,
};

pub mod materialization;
pub use materialization::{
    IoProtocolError, IoProtocolResult, MaterializationResolveError, MaterializationResolveResult,
    MaterializationResourceError, MaterializationResourceResult,
};

/// Cross-crate error boundary for model, execution, and backend APIs.
#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(transparent)]
    Io { source: std::io::Error },

    #[snafu(transparent)]
    IoProtocol { source: IoProtocolError },

    #[snafu(transparent)]
    Materialization { source: MaterializationResolveError },

    #[snafu(transparent)]
    MaterializationResources {
        source: MaterializationResourceError,
    },

    #[snafu(display("GGUF: {message}"))]
    Gguf { message: String },

    #[snafu(display("graph: {message}"))]
    Graph { message: String },

    #[snafu(display("kernel: {message}"))]
    Kernel { message: String },

    #[snafu(display("backend: {source}"))]
    Backend {
        source: Box<dyn StdError + Send + Sync>,
    },

    #[snafu(display("model: {message}"))]
    Model { message: String },

    #[snafu(display("model: {source}"))]
    ModelSource {
        source: Box<dyn StdError + Send + Sync>,
    },

    #[snafu(display("execution: {message}"))]
    Execution { message: String },

    #[snafu(display("tokenization: {message}"))]
    Tokenization { message: String },

    #[snafu(display("internal invariant: {message}"))]
    Internal { message: String },

    #[snafu(display("{operation}: {source}"))]
    Context {
        operation: String,
        source: Box<Error>,
    },

    #[snafu(display("{operation} failed: {source}; cleanup also failed: {cleanup}"))]
    Cleanup {
        operation: String,
        source: Box<Error>,
        cleanup: Box<Error>,
    },

    #[snafu(display(
        "{operation} encountered {} independent failures",
        failures.len()
    ))]
    FailureBatch {
        operation: String,
        failures: Vec<Error>,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn context(operation: impl Into<String>, source: Error) -> Self {
        Self::Context {
            operation: operation.into(),
            source: Box::new(source),
        }
    }

    pub fn with_cleanup(operation: impl Into<String>, source: Error, cleanup: Result<()>) -> Self {
        match cleanup {
            Ok(()) => source,
            Err(cleanup) => Self::Cleanup {
                operation: operation.into(),
                source: Box::new(source),
                cleanup: Box::new(cleanup),
            },
        }
    }

    pub fn failures(operation: impl Into<String>, failures: Vec<Error>) -> Result<()> {
        if failures.is_empty() {
            Ok(())
        } else {
            Err(Self::FailureBatch {
                operation: operation.into(),
                failures,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_errors_preserve_typed_sources() {
        let error = WorkerExecutionError::AdmissionTokenization {
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid tokens"),
        };

        assert_eq!(
            StdError::source(&error)
                .and_then(|source| source.downcast_ref::<std::io::Error>())
                .map(std::io::Error::kind),
            Some(std::io::ErrorKind::InvalidData)
        );
    }

    #[test]
    fn config_errors_keep_structured_fields() {
        let error = ServingConfigError::EventQueueCapacityTooSmall { actual: 1 };
        assert_eq!(
            error.to_string(),
            "event_queue_capacity must be at least two, got 1"
        );
    }

    #[test]
    fn name_mapping_error_preserves_typed_canonical_path_source() {
        let error = NameMappingError::invalid_canonical_path(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid module path",
        ));

        assert!(
            StdError::source(&error)
                .and_then(|source| source.downcast_ref::<std::io::Error>())
                .is_some()
        );
    }

    #[test]
    fn binding_error_preserves_first_typed_issue_as_source() {
        let error = StateDictBindingError::new(vec![StateDictMetadataError::ByteSizeMismatch {
            expected: 8,
            actual: 4,
        }]);

        assert!(matches!(
            StdError::source(&error)
                .and_then(|source| source.downcast_ref::<StateDictMetadataError>()),
            Some(StateDictMetadataError::ByteSizeMismatch {
                expected: 8,
                actual: 4
            })
        ));
        assert_eq!(error.issues().len(), 1);
        assert_eq!(
            error.to_string(),
            "state-dict binding failed with 1 issue(s)"
        );
    }
}
