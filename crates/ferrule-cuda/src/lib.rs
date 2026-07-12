#![allow(
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unnecessary_sort_by,
    unsafe_code
)]
//! CUDA backend for Ferrule — kernels, forward pass, memory pool.
//!
//! Note: `graph` in this crate refers to CUDA driver graph capture/replay.
//! Ferrule's device-independent compute graph IR lives in `ferrule-runtime::graph`.

pub mod benchmark;
pub mod context;
pub mod counters;
pub mod graph;
pub mod kernels;
pub mod kv_page_pool;
pub mod transformer;

/// Re-export `CudaContext` for the same reason.
pub use cuda_core::CudaContext;
/// Re-export `CudaStream` so downstream crates (e.g. `ferrule-runtime`) can
/// use `ferrule_cuda::CudaStream` without directly depending on `cuda-core`.
pub use cuda_core::stream::CudaStream;

// Re-export common CUDA API surface for downstream crates.
pub use benchmark::{CudaSmokeBenchmark, run_gemv_rms_smoke_benchmark, run_smoke_benchmark};
pub use context::CudaFailpoints;
pub use context::CudaSwiGLUWorkspace;
pub use context::cuda_probe;
pub use context::{
    CombinedRingWindowLens, CudaArtifactOperatorContext, CudaCompressorRecurrentState,
    CudaF32Buffer, CudaI32Buffer,
};
pub use counters::CudaOpCounters;
pub use kv_page_pool::{
    CudaKvPagePool, KvHostSnapshot, KvPagePoolStats, KvPoolReservation, PagedPlaneLayout,
};
pub use transformer::combined_ring::CombinedRingTopkLayout;
pub use transformer::compressor_recurrent::CompressorRecurrentShape;
pub use transformer::sparse_attention::{
    DualPlanePagedSparseAttentionLayout, PagedSparseAttentionLayout,
};
