//! Native resident multi-session execution.
//!
//! The engine owns request/session lifecycle, scheduling integration, explicit
//! per-sequence model state, and authoritative paged-KV transactions without
//! depending on a concrete model family.

mod composition;
mod driver;
mod inference;
pub mod model_factory;
mod native_executor;
mod observability;
mod pipeline;

pub use composition::{
    ResidentKvPageAccounting, ResidentKvPagePlan, build_resident_engine, plan_resident_kv_pages,
};
pub use driver::{
    ResidentActionKind, ResidentCancelProgress, ResidentDriverShutdownReport, ResidentDriverStep,
    ResidentRuntimeResourceLimits, ResidentShutdownProgress, ResidentTokenEvent,
    ResidentTopKDriver, ResidentTopKDriverConfig, RuntimeAdmissionError, RuntimeAdmissionOptions,
    RuntimeAdmissionResource, RuntimeAdmissionSnapshot,
};
pub use inference::{
    BoxedSessionInferenceEngine, InferenceCancelProgress, InferenceCompletionOwner,
    InferenceCompletionReactor, InferenceEngine, InferenceRequestCleanup,
    InferenceShutdownProgress, LocalResidentInferenceEngine, LocalSessionInferenceEngine,
    RequestCleanupOwner, RequestCleanupReceipt, ResidentInferenceEngine, SessionInferenceEngine,
};
pub use model_factory::{
    BackendSelection, BuiltinModelResolver, ModelFactoryOptions, ResidentModelBuildObservability,
    ResidentModelBuildPlan, ResidentModelPlanner, ResolvedModelBackend,
};
pub use observability::{
    ResidentEngineObservability, ResidentExternalTokenObservability, ResidentKvCacheObservability,
    ResidentMaterializationObservability, ResidentMaterializationStageObservability,
    ResidentPrefixCacheStats, ResidentTopKDriverStats,
};

pub use native_executor::NativeMultiSessionExecutor;
pub use pipeline::PipelineInferenceEngine;
