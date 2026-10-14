#![cfg_attr(not(feature = "cpu-reference"), doc = include_str!("../tests/compile/cpu_reference_api.md"))]
#![allow(
    clippy::unnecessary_sort_by,
    clippy::needless_range_loop,
    clippy::result_large_err,
    clippy::too_many_arguments
)]
//! Ferrule runtime for resident workloads, shared scheduling and I/O, resource
//! residency, transactional state, and exact speculative execution.

#[cfg(test)]
extern crate self as ferrule_runtime;

mod error;

#[cfg(all(test, unix))]
#[path = "../tests/support/build_process_child.rs"]
mod build_process_child;

#[cfg(test)]
#[path = "../tests/support/hybrid_fixture.rs"]
mod hybrid_fixture;

// ── Sub-directory modules ─────────────────────────────────────────────────
pub mod cache;
pub mod distributed;
pub mod io;
pub mod parallel;
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
pub use engine::NativeMultiSessionExecutor;
pub use engine::{
    HostKvDecodeConfig, IdleKvMemoryPlan, IdleKvPhysicalStats, IdleKvReclaimConfig,
    InferenceCancelProgress, InferenceCompletionOwner, InferenceCompletionReactor, InferenceEngine,
    InferenceRequestCleanup, InferenceShutdownProgress, KvExecutionMode, KvExecutionPolicy,
    KvExecutionStats, LocalResidentInferenceEngine, PipelineInferenceEngine, RequestCleanupOwner,
    RequestCleanupReceipt, ResidentActionKind, ResidentCancelProgress,
    ResidentDriverShutdownReport, ResidentDriverStep, ResidentEngineObservability,
    ResidentExternalTokenObservability, ResidentInferenceEngine, ResidentKvCacheObservability,
    ResidentMaterializationObservability, ResidentMaterializationStageObservability,
    ResidentPrefixCacheStats, ResidentShutdownProgress, ResidentTokenEvent, ResidentTopKDriver,
    ResidentTopKDriverConfig, ResidentTopKDriverStats, RunnerMultiSessionExecutor,
    RuntimeAdmissionError, RuntimeAdmissionOptions, RuntimeAdmissionResource,
    RuntimeAdmissionSnapshot, RuntimeKvTraceConfig, RuntimeKvTraceKind, RuntimeKvTraceSnapshot,
    RuntimePrefixReplay,
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
