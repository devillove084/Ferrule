//! Native resident multi-session execution.
//!
//! The engine owns request/session lifecycle, scheduling integration, explicit
//! per-sequence model state, and authoritative paged-KV transactions without
//! depending on a concrete model family.

mod resident_composition;

mod driver;
mod inference;
mod kv_policy;
pub mod kv_trace;

mod observability;
mod pipeline;
mod resource_summary;
mod runner_executor;

pub use driver::{
    ResidentActionKind, ResidentCancelProgress, ResidentDriverShutdownReport, ResidentDriverStep,
    ResidentRuntimeResourceLimits, ResidentShutdownProgress, ResidentTokenEvent,
    ResidentTopKDriver, ResidentTopKDriverConfig, RuntimeAdmissionError, RuntimeAdmissionOptions,
    RuntimeAdmissionResource, RuntimeAdmissionSnapshot,
};
pub use inference::{
    BoxedSessionInferenceEngine, InferenceCancelProgress, InferenceCapacitySnapshot,
    InferenceCompletionOwner, InferenceCompletionReactor, InferenceEngine, InferenceRequestCleanup,
    InferenceShutdownProgress, LocalResidentInferenceEngine, LocalSessionInferenceEngine,
    RequestCleanupOwner, RequestCleanupReceipt, ResidentInferenceEngine, SessionInferenceEngine,
};
pub use kv_policy::{
    HostKvDecodeConfig, IdleKvMemoryPlan, IdleKvPhysicalStats, IdleKvReclaimConfig,
    KvExecutionMode, KvExecutionPolicy, KvExecutionStats, validate_host_kv_spec,
};
pub use kv_trace::{
    RuntimeKvProjectionLimits, RuntimeKvTraceConfig, RuntimeKvTraceError, RuntimeKvTraceEvent,
    RuntimeKvTraceHeader, RuntimeKvTraceKind, RuntimeKvTraceSnapshot, RuntimeKvTraceStatus,
    RuntimeKvTraceUsage, RuntimePrefixReplay, project_runtime_kv_trace,
    project_runtime_kv_trace_with_limits, replay_runtime_prefix_lru,
};
pub use resident_composition::{
    ContextCapacityProfile, ResidentKvPageAccounting, ResidentKvPagePlan, build_resident_engine,
    compose_resident_engine, plan_pipeline_kv_pages, plan_resident_kv_pages,
};

pub use observability::{
    ResidentEngineObservability, ResidentExternalTokenObservability, ResidentKvCacheObservability,
    ResidentMaterializationObservability, ResidentMaterializationStageObservability,
    ResidentPrefixCacheStats, ResidentResourceObservability, ResidentTopKDriverStats,
};

pub use resource_summary::{
    BrokerDomainObservation, ResidentResourceShutdownSummary, ResourceShutdownOutcome,
    ResourceSummaryCoverage,
};

pub use pipeline::PipelineInferenceEngine;
pub use runner_executor::{NativeMultiSessionExecutor, RunnerMultiSessionExecutor};
