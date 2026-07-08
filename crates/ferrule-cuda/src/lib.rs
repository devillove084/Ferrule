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
//! Ferrule's device-independent compute graph IR lives in `ferrule-graph`.

pub mod context;
pub mod graph;
pub mod kernels;
pub mod transformer;

/// Re-export `CudaStream` so downstream crates (e.g. `ferrule-runtime`) can
/// use `ferrule_cuda::CudaStream` without directly depending on `cuda-core`.
pub use cuda_core::stream::CudaStream;
/// Re-export `CudaContext` for the same reason.
pub use cuda_core::CudaContext;

// Re-export from context for backward compatibility.
pub use context::cuda_probe;
