//! Runtime-owned materialization, completion, waiter, fairness, and critical-path core.

mod fairness;
mod ledger;
mod provider;
mod registry;
mod resolver;
pub mod testing;
mod waiters;

pub use fairness::{
    FairQueue, FairQueueBand, FairQueueConfig, FairQueueEntryState, FairQueueError,
};
pub use ledger::{
    CohortId, CriticalPathLedger, CriticalPhase, LedgerError, OutputTokenId, OutputTokenSnapshot,
    PhaseDurations, TimeSpan,
};
pub use provider::{
    ExecutionPromotion, MaterializationOperationReservation, RuntimeMaterializationProvider,
    SharedMaterializationProvider, UnavailableMaterializationProvider,
};
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
