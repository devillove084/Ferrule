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

pub mod catalog;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_materialization;
pub mod executor;
pub mod handle;
#[cfg(target_os = "linux")]
pub(crate) mod io_uring_reader;
pub mod prediction;
#[cfg(any(feature = "cuda", test))]
pub(crate) mod residency;
#[cfg(feature = "cuda")]
mod resource_manager;
pub mod routed;
pub mod routing;
pub mod streaming;
pub mod telemetry;

pub use executor::ExpertExecutor;
#[cfg(any(test, feature = "cpu-reference"))]
pub use executor::{CpuReferenceExpertExecutor, reference_linear};
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
#[cfg(feature = "cuda")]
pub use resource_manager::{
    RoutedMoePrefetchLayer, RoutedMoeResidencyConfig, RoutedMoeResourceManager,
};
#[cfg(feature = "cuda")]
pub use routed::{
    PreparedRoutedMoe, RoutedMoeAttribution, RoutedMoeCancelProgress, RoutedMoeContinuation,
    RoutedMoeExecution, RoutedMoePendingExpert, RoutedMoeProgress, RoutedMoeQuiescence,
    RoutedMoeRouteState, RoutedMoeScratch, RoutedMoeSequenceEvent,
};
pub use routed::{
    RoutedMoePayload, RoutedMoeStepOutput, execute_routed_moe_reference,
    execute_routed_moe_reference_with_handles, execute_routed_moe_with_artifact_router_reference,
    execute_routed_moe_with_artifact_router_reference_with_handles,
};
pub use routing::{
    ExpertRoute, ExpertRouterPolicy, RouterScoreFunction, RouterSelectionPolicy,
    RouterWeightNormalization,
};
pub use streaming::{
    AsyncHostStagedExpertLoader, AsyncHostStagedExpertStats, ExpertArtifactPayload,
    ExpertComputeBundle, ExpertEvictRequest, ExpertId, ExpertIoTransport, ExpertIoTransportError,
    ExpertLayerSources, ExpertLinearFormat, ExpertLinearPayload, ExpertLoadReason,
    ExpertLoadRequest, ExpertLoadSource, ExpertMatrixKind, ExpertMemoryPolicy, ExpertStorageTier,
    ExpertStreamingPlanner, ExpertStreamingPolicy, ExpertStreamingReader, ExpertStreamingStep,
    ExpertTensorComponent, ExpertTensorKey, ExpertTensorPayload, ExpertTensorSlice,
    HostStagedExpertCache, read_experts_concurrent,
};
pub use telemetry::ExpertTelemetry;
