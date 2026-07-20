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
    KvPageManager, KvPageManagerStats, KvReservationBindings, PageBlockTable, PreemptedKvState,
    PreparedKvSequenceFork,
};
pub use engine::{
    DSparkCycleTrace, NativeMultiSessionExecutor, PageManagedDiagnosticHarness, ResidentActionKind,
    ResidentDriverStep, ResidentTokenEvent, ResidentTopKDriver, ResidentTopKDriverConfig,
    ResidentTopKDriverStats,
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
    ResidentSchedulerConfig, SchedulerAction, SequenceSlotPool, ZeroExpertIoAdvisor,
    plan_prefill_chunk,
};
pub use scheduling::{
    GenerateRequest, RequestId, SequenceFinishReason, SequenceState, SequenceStatus, SessionId,
};

pub use speculation::{
    DSparkCycleResult, DSparkMetrics, SpeculativeCycleAccounting, TargetFrontier,
    VerificationWidth, run_dspark_verification,
};
