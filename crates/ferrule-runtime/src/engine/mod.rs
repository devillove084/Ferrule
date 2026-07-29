//! Native resident multi-session execution.
//!
//! The engine owns request/session lifecycle, scheduling integration, explicit
//! per-sequence model state, and authoritative paged-KV transactions without
//! depending on a concrete model family.

mod driver;
mod inference;
mod native_executor;
mod observability;

pub use driver::{
    ResidentActionKind, ResidentDriverShutdownReport, ResidentDriverStep,
    ResidentRuntimeResourceLimits, ResidentTokenEvent, ResidentTopKDriver,
    ResidentTopKDriverConfig,
};
pub use inference::{
    InferenceCancelProgress, InferenceCompletionOwner, InferenceCompletionReactor, InferenceEngine,
    LocalResidentInferenceEngine, ResidentInferenceEngine,
};
pub use observability::ResidentTopKDriverStats;

pub use native_executor::{NativeBatchExecutionProgress, NativeMultiSessionExecutor};
