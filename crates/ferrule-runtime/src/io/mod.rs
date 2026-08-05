//! Runtime-owned materialization, completion, waiter, fairness, and critical-path core.

mod fairness;
mod ledger;
mod provider;
mod registry;
mod resolver;
mod waiters;

pub use fairness::{FairQueue, FairQueueConfig, FairQueueError};
pub use ledger::{
    CohortId, CriticalPathLedger, CriticalPhase, LedgerError, OutputTokenId, OutputTokenSnapshot,
    PhaseDurations, TimeSpan,
};
pub use provider::{
    ExecutionPromotion, MaterializationOperationReservation, RuntimeMaterializationProvider,
    SharedMaterializationProvider, UnavailableMaterializationProvider,
};
#[cfg(test)]
pub use provider::{FakeCompletionSpec, FakeMaterializationCommand, FakeMaterializationProvider};

pub use registry::{
    AttachReport, CompletionDisposition, CompletionRejection, CompletionRejectionReason,
    ContinuationFailure, FailedContinuation, LoadOp, LoadRegistry, LoadRequest, PrefetchOwner,
    PrefetchReport, RegistryError, RegistryStats, ResumeDisposition, ResumeLease, ShutdownReport,
    TransactionCustodyOutcome,
};
pub use resolver::{RuntimeMaterializationResolver, RuntimeMaterializationResolverStats};
pub use waiters::{
    ContinuationDetach, OperationResolution, WaiterDetach, WaiterIndex, WaiterIndexError,
};

#[cfg(test)]
pub(crate) mod physical_tests;
#[cfg(test)]
mod tests;
