//! Ferrule Common — shared types, errors, and observability infrastructure.

pub mod async_wake;
pub mod execution;
pub mod expert_residency;
pub mod io_protocol;

pub mod materialization_io;
pub mod memory;
pub mod observability;

pub use async_wake::{CompletionHub, CompletionListener, CompletionWake};
pub use expert_residency::{
    ExpertInstallActivationOutcome, ExpertInstallIntent, ExpertInstallPrepareOutcome,
    ExpertInstallReason, ExpertKey, ExpertLease, ExpertResidencyControl,
    ExpertResidencyCoordinator, ExpertResidencyCoordinatorStats, ExpertResidencyGrant,
    ExpertResidencyRequirements, ExpertResidencyStats, ExpertSlotBinding, ExpertSlotGeneration,
    ExpertSlotId, PreparedExpertInstall,
};
pub use io_protocol::*;
pub use memory::{
    MemoryPoolKind, MemoryPoolLimits, MemoryPoolStats, MemoryTopology, OwnerMemoryLru,
};

use snafu::Snafu;

/// Cross-crate error boundary for model, execution, and backend APIs.
///
/// Subsystems keep their own typed errors and preserve them as sources at their
/// outer boundary. Message variants remain only for legacy leaf APIs that have
/// not yet acquired a subsystem-specific error type.
#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(transparent)]
    Io { source: std::io::Error },

    #[snafu(transparent)]
    IoProtocol {
        source: io_protocol::IoProtocolError,
    },

    #[snafu(transparent)]
    Materialization {
        source: io_protocol::MaterializationResolveError,
    },

    #[snafu(transparent)]
    MaterializationResources {
        source: materialization_io::MaterializationResourceError,
    },

    #[snafu(display("GGUF: {message}"))]
    Gguf { message: String },

    #[snafu(display("graph: {message}"))]
    Graph { message: String },

    #[snafu(display("kernel: {message}"))]
    Kernel { message: String },

    #[snafu(display("backend: {source}"))]
    Backend {
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[snafu(display("model: {message}"))]
    Model { message: String },

    #[snafu(display("model: {source}"))]
    ModelSource {
        source: Box<dyn std::error::Error + Send + Sync>,
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

/// Quantization format identifier — mirrors GGUF's type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    Bf16 = 30,
    Q1_0 = 41,
}

impl QuantType {
    /// Bytes per block (the unit of quantization granularity).
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K => 256,
            Self::Q3_K => 256,
            Self::Q4_K => 256,
            Self::Q5_K => 256,
            Self::Q6_K => 256,
            Self::Q8_K => 256,
            Self::Iq2Xxs => 256,
            Self::Iq2Xs => 256,
            Self::Iq3Xxs => 256,
            Self::Iq1S => 256,
            Self::Iq4Nl => 32,
            Self::Iq3S => 256,
            Self::Iq2S => 256,
            Self::Iq4Xs => 256,
            Self::Q1_0 => 32,
        }
    }

    /// Bytes per element on average.
    pub fn type_size(self) -> f64 {
        match self {
            Self::F32 => 4.0,
            Self::F16 | Self::Bf16 => 2.0,
            Self::Q4_0 => 0.5,
            Self::Q4_1 => 0.5,
            Self::Q8_0 => 1.0,
            Self::Q8_1 => 1.0,
            Self::Q5_0 => 0.625,
            Self::Q5_1 => 0.625,
            Self::Q2_K => 0.3125,
            Self::Q3_K => 0.4375,
            Self::Q4_K => 0.5625,
            Self::Q5_K => 0.6875,
            Self::Q6_K => 0.8125,
            Self::Q8_K => 1.0625,
            Self::Iq2Xxs => 0.28125,
            Self::Iq2Xs => 0.3125,
            Self::Iq3Xxs => 0.375,
            Self::Iq1S => 0.1953125,
            Self::Iq4Nl => 0.5625,
            Self::Iq3S => 0.4375,
            Self::Iq2S => 0.3125,
            Self::Iq4Xs => 0.5625,
            Self::Q1_0 => 0.125,
        }
    }

    pub fn is_quantized(self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::Bf16)
    }
}
