//! NVIDIA CUDA device runtime and kernel providers.
//!
//! CUDA Driver API handles, streams, events, memory and implementation-specific
//! kernels remain inside this module. Model preparation consumes provider-neutral
//! plans from [`crate::plan`].

pub mod architecture;
pub mod benchmark;
pub mod context;
pub mod counters;
pub mod cutlass;
pub mod graph;
pub mod kernels;
pub mod kv_page_pool;
pub mod provider;
pub mod runtime;
pub mod transformer;

pub use runtime::{CudaContext, CudaStream};

pub use architecture::{
    COMPILED_TARGET, CudaArchitectureFamily, CudaKernelCapabilities, CudaTarget,
    compiled_capabilities,
};
pub use benchmark::{CudaSmokeBenchmark, run_gemv_rms_smoke_benchmark, run_smoke_benchmark};
pub use context::{
    CombinedRingWindowLens, CudaArtifactOperatorContext, CudaBf16Buffer,
    CudaCompressorRecurrentCheckpointSlab, CudaCompressorRecurrentState, CudaExpertSlotPointers,
    CudaF32Buffer, CudaFailpoints, CudaHybridMlaAttentionWorkspace,
    CudaHybridMlaExplicitSelectionWorkspace, CudaI32Buffer, CudaPreparedRoutedExpert,
    CudaProposalHeadWorkspace, CudaRoutedExpertArena, CudaRoutedExpertMaterialization,
    CudaRoutedExpertShape, cuda_probe,
};
pub use counters::CudaOpCounters;
pub use kv_page_pool::{
    CudaKvPagePool, KvHostSnapshot, KvPagePoolStats, KvPoolReservation, PagedPlaneLayout,
};
pub use provider::{CudaProviderCatalog, compile_cuda_model_plan};
pub use transformer::combined_ring::CombinedRingTopkLayout;
pub use transformer::compressor_recurrent::CompressorRecurrentShape;
pub use transformer::sparse_attention::{
    DualPlanePagedSparseAttentionLayout, PagedSparseAttentionLayout,
};
