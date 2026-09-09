#![allow(
    clippy::unnecessary_sort_by,
    clippy::needless_range_loop,
    clippy::result_large_err,
    clippy::too_many_arguments
)]
//! Ferrule runtime for resident workloads, shared scheduling and I/O, resource
//! residency, transactional state, and exact speculative execution.

mod error;

// ── Sub-directory modules ─────────────────────────────────────────────────
pub mod cache;
pub mod distributed;
pub mod io;
pub mod scheduling;

// ── Top-level modules ─────────────────────────────────────────────────────
pub mod engine;
pub mod expert_residency;
pub mod speculation;

// ── Convenience re-exports ────────────────────────────────────────────────
pub use distributed::{
    Decision, DistributedTransaction, DistributedTransactionError, DistributedTransactionRecord,
    FinalizeOutcome, TransactionReport, TransactionState,
};
pub use error::{CleanupStep, Error, Result};
pub use ferrule_common::ParallelRankId;
pub use ferrule_common::execution::ExecutionTransactionId;

pub use cache::{
    KvPageManager, KvPageManagerStats, KvReservation, KvReservationBindings, KvReservationCommit,
    KvReservationId, PageBlockTable, PreemptedKvState, PreparedKvSequenceFork,
};
pub use engine::{
    BackendSelection, BuiltinModelResolver, InferenceCancelProgress, InferenceCompletionOwner,
    InferenceCompletionReactor, InferenceEngine, InferenceShutdownProgress,
    LocalResidentInferenceEngine, ModelFactoryOptions, NativeMultiSessionExecutor,
    ResidentActionKind, ResidentCancelProgress, ResidentDriverShutdownReport, ResidentDriverStep,
    ResidentEngineObservability, ResidentExternalTokenObservability, ResidentInferenceEngine,
    ResidentKvCacheObservability, ResidentMaterializationObservability,
    ResidentMaterializationStageObservability, ResidentModelBuildObservability,
    ResidentModelBuildPlan, ResidentModelPlanner, ResidentPrefixCacheStats,
    ResidentShutdownProgress, ResidentTokenEvent, ResidentTopKDriver, ResidentTopKDriverConfig,
    ResidentTopKDriverStats, ResolvedModelBackend,
};
pub use expert_residency::{
    ExpertInstallIntent, ExpertInstallPrepareOutcome, ExpertInstallReason, ExpertKey, ExpertLease,
    ExpertResidencyControl, ExpertResidencyController, ExpertResidencyCoordinator,
    ExpertResidencyCoordinatorStats, ExpertResidencyGrant, ExpertResidencyRequirements,
    ExpertResidencyStats, ExpertSlotBinding, ExpertSlotGeneration, ExpertSlotId,
    PreparedExpertInstall,
};

pub use scheduling::{
    CancelRequestResult, DecodeAction, ExecutionPhase, ExecutionPhaseSet, FixedSequenceSlotPool,
    KvHandle, LogitsSelection, PhysicalResourceBroker, PhysicalResourceClaim,
    PhysicalResourceError, PhysicalResourceGrant, PhysicalResourceGrantId, PhysicalResourceLimit,
    PhysicalResourceSnapshot, PrefillChunkAction, RequestTerminal, ResidentScheduler,
    ResidentSchedulerConfig, ResourceDemand, ResourceKind, ResourceUnit, SchedulerAction,
    SequenceSlotPool, plan_prefill_chunk,
};
pub use scheduling::{
    GenerateRequest, RequestId, SequenceFinishReason, SequenceState, SequenceStatus, SessionId,
};

pub use speculation::{
    SpeculativeCycleAccounting, SpeculativeCycleResult, SpeculativeMetrics, TargetFrontier,
};
