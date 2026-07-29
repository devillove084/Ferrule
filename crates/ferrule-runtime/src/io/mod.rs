//! Runtime-owned materialization, completion, waiter, fairness, and critical-path core.

mod adapter;
mod backend;
mod fairness;
mod ledger;
mod registry;
mod waiters;

pub use adapter::{
    RuntimeMaterializationAdapter, RuntimeMaterializationAdapterStats,
    RuntimeMaterializationControl,
};
#[cfg(test)]
pub use backend::{FakeBackend, FakeCommand, FakeCompletionSpec};
pub use backend::{
    MaterializationBackend, MaterializationReservation, RunnerMaterializationBackend,
    UnavailableBackend,
};
pub use fairness::{FairQueue, FairQueueConfig, FairQueueError};
pub use ledger::{
    CohortId, CriticalPathLedger, CriticalPhase, LedgerError, OutputTokenId, OutputTokenSnapshot,
    PhaseDurations, TimeSpan,
};
pub use registry::{
    AttachReport, CompletionDisposition, CompletionRejection, CompletionRejectionReason,
    ContinuationFailure, FailedContinuation, LoadOp, LoadRegistry, LoadRequest, RegistryError,
    RegistryStats, ResumeLease, ShutdownReport,
};
pub use waiters::{
    ContinuationDetach, OperationResolution, WaiterDetach, WaiterIndex, WaiterIndexError,
};

#[cfg(test)]
pub(crate) mod physical_tests;
#[cfg(test)]
mod tests;
