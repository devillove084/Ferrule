//! Ferrule Common — shared types, errors, and observability infrastructure.

pub mod execution;
pub mod expert_io;
pub mod expert_residency;
pub mod kernel_plan;
pub mod memory;
pub mod observability;

pub use expert_io::{ExpertIoEstimate, ExpertIoPhase};
pub use expert_residency::{
    ExpertInstallIntent, ExpertInstallPrepareOutcome, ExpertInstallReason, ExpertKey, ExpertLease,
    ExpertResidencyControl, ExpertResidencyCoordinator, ExpertResidencyCoordinatorStats,
    ExpertResidencyGrant, ExpertResidencyRequirements, ExpertResidencyStats, ExpertSlotBinding,
    ExpertSlotGeneration, ExpertSlotId, PreparedExpertInstall,
};
pub use memory::{
    MemoryPoolKind, MemoryPoolLimits, MemoryPoolStats, MemoryTopology, OwnerMemoryLru,
};

use thiserror::Error;

/// The one error type for the entire system.
#[derive(Error, Debug)]
pub enum Error {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    #[error("GGUF: {0}")]
    Gguf(String),

    #[error("Graph: {0}")]
    Graph(String),

    #[error("Kernel: {0}")]
    Kernel(String),

    #[error("Model: {0}")]
    Model(String),

    #[error("Execution: {0}")]
    Execution(String),

    #[error("Tokenization: {0}")]
    Tokenization(String),

    #[error("Internal: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, Error>;

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
