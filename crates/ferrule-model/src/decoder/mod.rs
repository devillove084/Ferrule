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
    DecoderKvPageStatus, DecoderKvPageView, KvCommitProjection, LogicalExecutionIdentity,
    LogitsPlan, LogitsPlanRow, PackedDecoderBatch, PackedDecoderSequence,
};
pub use kv::{
    CpuKvPlaneStorage, CpuKvView, CpuPagedKvBackend, CpuPagedKvPool, CpuPagedKvTransaction,
    DecoderKvPlaneStrategy, KvCommitOwner, KvPrepareQuiescenceUnknown, MlaPlaneStrategy,
    PagedKvBackend, PagedKvOwnership, PagedKvTransaction, PagedKvTransactionHandle,
    PhysicalKvCommitReady, PhysicalKvPool, PhysicalKvPreparedPool, PrepareKvCommitError,
    PreparedKvCommit, PreparedKvRetirement, StandardGqaPlanes, TypedCpuPagedKvPool,
};
#[cfg(feature = "cuda")]
pub use kv::{
    CudaGqaPlanesMut, CudaKvView, CudaPagedKvBackend, CudaPagedKvPool, CudaPagedKvTransaction,
    TypedCudaPagedKvPool,
};
pub use logits::{DecoderLogits, DecoderTopKRow, DenseLogits};
pub use runner::{
    GenericDecoderModelView, GenericDecoderObservabilitySnapshot, GenericDecoderOptions,
    GenericDecoderRunner, GenericDecoderSequenceState, HybridCpuDecoder, HybridCpuKvBackend,
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
    DecoderWait, KvCapacityInspector, KvCommitBinding, KvCommitParticipant, KvEndProgress,
    KvRankAck,
};
#[cfg(test)]
pub(crate) use transaction::DecoderTransactionPhase;
pub(crate) use transaction::{DecoderTransactionProgress, PackedTransactionRegistry};
#[cfg(test)]
mod custom_runtime_tests;
#[cfg(test)]
mod tests;

mod hybrid;
pub use hybrid::{
    GatedDeltaNetState, GatedDeltaStateRef, HybridDecoderSequenceState, HybridLayerSchema,
    HybridLayerState, HybridLayerStates, HybridSequenceLifecycle, HybridStateSchema,
    StandardSequenceState,
};

#[cfg(feature = "cuda")]
mod hybrid_cuda;
#[cfg(feature = "cuda")]
pub use hybrid_cuda::{
    CudaGatedDeltaState, CudaHybridLayerStates, CudaHybridSequenceLifecycle,
    CudaHybridSequenceState, HybridCudaCompletionUnknown, HybridCudaDecoder, HybridCudaDevice,
    HybridCudaExpertProgress, HybridCudaKvBackend, HybridCudaMemoryBudget,
    HybridCudaMemoryEstimate, HybridCudaRoutedExecutor, HybridCudaRoutedExperts,
};
