//! Mixture-of-Experts execution infrastructure.
//!
//! This module groups all expert-related machinery:
//!
//! - **Streaming** (`streaming`): artifact reader, residency planner, and the
//!   `HostStagedExpertCache` LRU host-side cache.
//! - **Handle** (`handle`): backend-agnostic expert handle stores for
//!   CPU reference and (future) CUDA resident expert management.
//! - **Executor** (`executor`): single-expert SwiGLLU execution for one
//!   activation vector.
//! - **Routing** (`routing`): router score functions, selection policies,
//!   and route normalization.
//! - **Routed MoE** (`routed`): orchestrates router → planner → executor →
//!   shared FFN into a single MoE step.
//! - **Residency** (`residency`): device-budget planning for routed experts.
//! - **Telemetry** (`telemetry`): expert activation counters.

#[cfg(feature = "cuda")]
pub(crate) mod cuda_materialization;
pub mod executor;
pub mod handle;
#[cfg(target_os = "linux")]
pub(crate) mod io_uring_reader;
pub mod prediction;
#[cfg(any(feature = "cuda", test))]
pub(crate) mod residency;
pub mod routed;
pub mod routing;
pub mod streaming;
pub mod telemetry;

pub use executor::{CpuReferenceExpertExecutor, ExpertExecutor, reference_linear};
pub use handle::{
    CpuExpertHandleStore, ExpertComputeHandle, ExpertHandleStore, ResidentExpertHandle,
    ResourceResidentFormat,
};
pub use prediction::{
    ExpertAccessEvent, ExpertAccessPhase, ExpertBatchAccessEvent, ExpertBatchExpertEvent,
    ExpertCacheAction, ExpertHotsetPredictor, ExpertPredictContext, ExpertPrediction,
    ExpertPredictionReason, ExpertPredictionStats, ExpertResidency, ExpertResidencyOutcome,
    ScoreBasedExpertPredictor, ScoreBasedExpertPredictorConfig,
};
pub use routed::{
    RoutedMoeStepOutput, execute_routed_moe_reference, execute_routed_moe_reference_with_handles,
    execute_routed_moe_with_artifact_router_reference,
    execute_routed_moe_with_artifact_router_reference_with_handles,
};
pub use routing::{ExpertRoute, ExpertRouterPolicy, RouterScoreFunction, RouterSelectionPolicy};
pub use streaming::{
    AsyncHostStagedExpertLoader, AsyncHostStagedExpertStats, ExpertArtifactPayload,
    ExpertComputeBundle, ExpertEvictRequest, ExpertId, ExpertIoTransport, ExpertIoTransportError,
    ExpertLinearFormat, ExpertLinearPayload, ExpertLoadReason, ExpertLoadRequest, ExpertLoadSource,
    ExpertMatrixKind, ExpertMemoryPolicy, ExpertStorageTier, ExpertStreamingPlanner,
    ExpertStreamingPolicy, ExpertStreamingReader, ExpertStreamingStep, ExpertTensorComponent,
    ExpertTensorKey, ExpertTensorPayload, ExpertTensorSlice, HostStagedExpertCache,
    read_experts_concurrent,
};
pub use telemetry::ExpertTelemetry;
