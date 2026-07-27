#![allow(
    clippy::unnecessary_sort_by,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]
//! Ferrule runtime for resident serving, scheduling, KV transactions, expert
//! residency, and exact speculative verification.

// ── Sub-directory modules ─────────────────────────────────────────────────
pub mod cache;
pub mod scheduling;

// ── Top-level modules ─────────────────────────────────────────────────────
pub mod attention_kernel;
pub mod engine;
pub mod expert_residency;
pub mod profiler;
pub mod speculation;

// ── Convenience re-exports ────────────────────────────────────────────────
pub use attention_kernel::AttentionKernel;
pub use cache::{
    KvPageManager, KvPageManagerStats, KvReservation, KvReservationBindings, KvReservationCommit,
    KvReservationId, PageBlockTable, PreemptedKvState, PreparedKvSequenceFork,
};
pub use engine::{
    InferenceCancelProgress, InferenceCompletionOwner, InferenceCompletionReactor, InferenceEngine,
    LocalResidentInferenceEngine, NativeMultiSessionExecutor, ResidentActionKind,
    ResidentDriverStep, ResidentInferenceEngine, ResidentTokenEvent, ResidentTopKDriver,
    ResidentTopKDriverConfig, ResidentTopKDriverStats,
};
pub use expert_residency::{
    ExpertInstallIntent, ExpertInstallPrepareOutcome, ExpertInstallReason, ExpertKey, ExpertLease,
    ExpertResidencyControl, ExpertResidencyController, ExpertResidencyCoordinator,
    ExpertResidencyCoordinatorStats, ExpertResidencyGrant, ExpertResidencyRequirements,
    ExpertResidencyStats, ExpertSlotBinding, ExpertSlotGeneration, ExpertSlotId,
    PreparedExpertInstall,
};
pub use profiler::{KernelProfiler, Profiler, TimedRegion};

pub use scheduling::{
    CancelRequestResult, DecodeAction, ExpertIoAdvisor, ExpertIoBudget, ExpertIoCandidate,
    ExpertIoDecisionTrace, ExpertIoEstimate, ExpertIoPhase, ExpertIoQueueClass, ExpertIoRejection,
    FixedSequenceSlotPool, KvHandle, LogitsSelection, PrefillChunkAction, ResidentScheduler,
    ResidentSchedulerConfig, ResourceBroker, ResourceBrokerBuilder, ResourceBrokerStats,
    ResourceClaim, ResourceClass, ResourceGrantId, ResourceId, ResourceRejection, ResourceRequest,
    ResourceSnapshot, ResourceUnit, SchedulerAction, SequenceSlotPool, ZeroExpertIoAdvisor,
    plan_prefill_chunk,
};
pub use scheduling::{
    GenerateRequest, RequestId, SequenceFinishReason, SequenceState, SequenceStatus, SessionId,
};

pub use speculation::{
    SpeculativeCycleAccounting, SpeculativeCycleResult, SpeculativeMetrics, TargetFrontier,
};
