//! Model-independent packed decoder execution core.
//!
//! The module owns sequence copy semantics, unified KV lowering, forward/backend
//! traits, output projection, exactly-once transaction custody, and a generic
//! runner bridge that model adapters can adopt incrementally.
//!
//! Sequence/lifecycle contracts live in `sequence`; concrete state implementations
//! remain in `state`. `hybrid_memory_plan` owns feature-independent metadata only;
//! live CUDA admission and BF16 state remain under the CUDA-gated `hybrid_cuda`.
mod batch;
mod external_work;
pub use external_work::{CompletionUnknown, WorkProgress};
#[cfg(feature = "cuda")]
mod cuda_host_kv_invocation;
mod invocation;
pub use invocation::{DecoderInvocation, DecoderInvocationScope};
mod kv;
mod kv_reclaim;
pub use kv_reclaim::{IdleKvMemoryPlan, IdleKvPhysicalStats, IdleKvReclaimConfig};
mod logits;
mod runner;
mod runtime;
mod sequence;
#[cfg(feature = "cuda")]
pub use cuda_host_kv_invocation::CudaHostKvAppendContext;
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
    DecoderResolverHandle, DecoderResourceManager, DecoderResourceSnapshot, DecoderTerminalGuard,
    DecoderTransactionContext, NoProposal, NoResourceManager, NoTerminalGuard,
    StandardDecoderObserver,
};
pub use sequence::{DecoderSequence, DecoderSequenceCheckout, DecoderSequenceLifecycle};
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
#[path = "../../tests/unit/decoder/custom_runtime_tests.rs"]
mod custom_runtime_tests;
#[cfg(test)]
#[path = "../../tests/unit/decoder/tests.rs"]
mod tests;

mod hybrid;
pub(crate) mod hybrid_memory_plan;
pub(crate) mod mixed_attention;
// Existing owner/factory APIs consume the same reservation requirements.
pub use hybrid::{
    GatedDeltaNetState, GatedDeltaStateRef, HybridDecoderSequenceState, HybridLayerEntry,
    HybridLayerSchema, HybridLayerState, HybridLayerStates, HybridSchemaError,
    HybridSequenceLifecycle, HybridStateSchema, StandardSequenceState,
};
pub use hybrid_memory_plan::HybridReservationRequirements as HybridCudaMemoryBudget;
pub use hybrid_memory_plan::{HybridMemoryPlan, HybridReservationRequirements};

#[cfg(feature = "cuda")]
mod hybrid_cuda;
#[cfg(feature = "cuda")]
pub(crate) use hybrid_cuda::MixedSourceRetirement;
#[cfg(feature = "cuda")]
pub use hybrid_cuda::{
    CudaBf16GatedDeltaState, CudaGatedDeltaState, CudaHybridLayerStates,
    CudaHybridSequenceLifecycle, CudaHybridSequenceState, HybridCudaDecoder, HybridCudaDevice,
    HybridCudaKvBackend, HybridCudaMemoryEstimate, HybridCudaPreparationClaim,
    HybridCudaPreparationContext, HybridCudaRoutedExecutor, HybridCudaRoutedExperts,
};
