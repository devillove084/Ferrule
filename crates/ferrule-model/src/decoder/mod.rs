//! Model-independent packed decoder execution core.
//!
//! The module owns sequence copy semantics, unified KV lowering, forward/backend
//! traits, output projection, exactly-once transaction custody, and a generic
//! runner bridge that model adapters can adopt incrementally.
mod batch;
mod kv;
mod logits;
mod runner;
mod runtime;
mod state;
mod traits;
mod transaction;
pub use batch::{
    DecoderKvPageStatus, DecoderKvPageView, LogitsPlan, LogitsPlanRow, PackedDecoderBatch,
    PackedDecoderSequence,
};
pub use kv::{
    CpuKvPlaneStorage, CpuKvView, CpuPagedKvBackend, CpuPagedKvPool, CpuPagedKvTransaction,
    DecoderKvPlaneStrategy, KvCommitOwner, KvPrepareQuiescenceUnknown, MlaPlaneStrategy,
    PagedKvBackend, PagedKvOwnership, PagedKvTransaction, PagedKvTransactionHandle,
    PhysicalKvCommitReady, PhysicalKvPool, PhysicalKvPreparedPool, PrepareKvCommitError,
    PreparedKvCommit, PreparedKvRetirement, StandardGqaPlanes,
};
#[cfg(feature = "cuda")]
pub use kv::{
    CudaGqaPlanesMut, CudaKvView, CudaPagedKvBackend, CudaPagedKvPool, CudaPagedKvTransaction,
};
pub use logits::{DecoderLogits, DecoderTopKRow, DenseLogits};
pub use runner::{
    GenericDecoderModelView, GenericDecoderObservabilitySnapshot, GenericDecoderOptions,
    GenericDecoderRunner, GenericDecoderSequenceState,
};
pub use runtime::{
    ComposedDecoderObserver, ComposedDecoderSnapshot, DecoderComponents, DecoderComposition,
    DecoderContinuationAllocator, DecoderControlPlane, DecoderForwardExecutor,
    DecoderForwardProgress, DecoderForwardResumeProgress, DecoderObserver, DecoderProposalExecutor,
    DecoderProposalProgress, DecoderProposalResumeProgress, DecoderProvisionalRetainContext,
    DecoderResolverHandle, DecoderResourceManager, DecoderResourceSnapshot, DecoderSequence,
    DecoderSequenceCheckout, DecoderSequenceLifecycle, DecoderTerminalGuard,
    DecoderTransactionContext, NoProposal, NoResourceManager, NoTerminalGuard,
    StandardDecoderObserver,
};
pub use state::{
    DecoderSequenceAttachment, DecoderSequenceReleaseError, DecoderSequenceState,
    StandardSequenceLifecycle,
};
pub use traits::{
    DecoderCancelProgress, DecoderContinuationWait, DecoderKvBackend, DecoderKvCapacity,
    DecoderKvCommitBackend, DecoderKvPageSnapshot, DecoderKvPrepare, DecoderKvSequenceCustody,
    DecoderWait, KvCommitBinding, KvEndProgress, KvRankAck,
};
#[cfg(test)]
pub(crate) use transaction::DecoderTransactionPhase;
pub(crate) use transaction::{DecoderTransactionProgress, PackedTransactionRegistry};
#[cfg(test)]
mod custom_runtime_tests;
#[cfg(test)]
mod tests;
