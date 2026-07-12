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
//! - **Telemetry** (`telemetry`): expert activation counters.

pub mod executor;
pub mod handle;
pub mod prediction;
pub mod routed;
pub mod routing;
pub mod streaming;
pub mod telemetry;

pub use executor::{CpuReferenceExpertExecutor, ExpertExecutor, reference_linear};
pub use handle::{
    CpuExpertHandleStore, ExpertComputeHandle, ExpertHandleStore, ExpertResidentFormat,
    ResidentExpertHandle,
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
    ExpertComputeBundle, ExpertEvictRequest, ExpertId, ExpertLinearFormat, ExpertLinearPayload,
    ExpertLoadReason, ExpertLoadRequest, ExpertLoadSource, ExpertMatrixKind, ExpertStorageTier,
    ExpertStreamingPlanner, ExpertStreamingPolicy, ExpertStreamingReader, ExpertStreamingStep,
    ExpertTensorComponent, ExpertTensorKey, ExpertTensorPayload, ExpertTensorSlice,
    HostStagedExpertCache, read_experts_concurrent,
};
pub use telemetry::ExpertTelemetry;
