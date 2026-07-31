#![allow(
    clippy::unnecessary_sort_by,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]
//! Ferrule runtime for resident workloads, shared scheduling and I/O, resource
//! residency, transactional state, and exact speculative execution.

// ── Sub-directory modules ─────────────────────────────────────────────────
pub mod cache;
pub mod io;
pub mod scheduling;

// ── Top-level modules ─────────────────────────────────────────────────────
pub mod engine;
pub mod expert_residency;
pub mod speculation;

// ── Convenience re-exports ────────────────────────────────────────────────
pub use cache::{
    KvPageManager, KvPageManagerStats, KvReservation, KvReservationBindings, KvReservationCommit,
    KvReservationId, PageBlockTable, PreemptedKvState, PreparedKvSequenceFork,
};
pub use engine::{
    InferenceCancelProgress, InferenceCompletionOwner, InferenceCompletionReactor, InferenceEngine,
    LocalResidentInferenceEngine, NativeMultiSessionExecutor, ResidentActionKind,
    ResidentDriverShutdownReport, ResidentDriverStep, ResidentInferenceEngine, ResidentTokenEvent,
    ResidentTopKDriver, ResidentTopKDriverConfig, ResidentTopKDriverStats,
};
pub use expert_residency::{
    ExpertInstallIntent, ExpertInstallPrepareOutcome, ExpertInstallReason, ExpertKey, ExpertLease,
    ExpertResidencyControl, ExpertResidencyController, ExpertResidencyCoordinator,
    ExpertResidencyCoordinatorStats, ExpertResidencyGrant, ExpertResidencyRequirements,
    ExpertResidencyStats, ExpertSlotBinding, ExpertSlotGeneration, ExpertSlotId,
    PreparedExpertInstall,
};

pub use scheduling::{
    CancelRequestResult, DecodeAction, FixedSequenceSlotPool, KvHandle, LogitsSelection,
    PhysicalResourceBroker, PhysicalResourceClaim, PhysicalResourceError, PhysicalResourceGrant,
    PhysicalResourceGrantId, PhysicalResourceLimit, PhysicalResourceSnapshot, PrefillChunkAction,
    ResidentScheduler, ResidentSchedulerConfig, ResourceClass, ResourceKind, ResourceUnit,
    SchedulerAction, SequenceSlotPool, plan_prefill_chunk,
};
pub use scheduling::{
    GenerateRequest, RequestId, SequenceFinishReason, SequenceState, SequenceStatus, SessionId,
};

pub use speculation::{
    SpeculativeCycleAccounting, SpeculativeCycleResult, SpeculativeMetrics, TargetFrontier,
};
