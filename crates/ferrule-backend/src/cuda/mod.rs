//! NVIDIA CUDA device runtime and kernel providers.
//!
//! CUDA Driver API handles, streams, events, memory and implementation-specific
//! kernels remain inside this module. Model preparation consumes provider-neutral
//! plans from [`crate::plan`].

mod allocator;
mod architecture;
mod benchmark;
#[doc(hidden)]
pub mod operator_owner;
#[doc(hidden)]
pub use operator_owner as context;
mod counters;
pub mod diagnostics;
mod ffi;
mod graph;

pub mod operators;

pub mod providers;
mod runtime;
pub use operator_owner::standard;
mod transport;

use ferrule_common::Result;

use crate::plan::{LayerKernelRequirements, ModelKernelPlan};

pub use allocator::CudaAllocatorMetrics;
pub use benchmark::{CudaSmokeBenchmark, run_gemv_rms_smoke_benchmark, run_smoke_benchmark};
pub use graph::{
    CachedDecodeGraph, CudaGraphHandle, capture_decode_graph, cuda_graph_enabled,
    flash_attn_enabled,
};
pub use runtime::MemoryTier;
pub use transport::{
    CudaAsyncTransport, CudaAsyncTransportStats, CudaTransferCompletion, CudaTransferConfig,
    CudaTransferDirection, CudaTransferDrainError, CudaTransferError, CudaTransferId,
    CudaTransferTicket,
};

/// Compile provider-neutral model requirements for the CUDA backend.
pub fn compile_model_plan(requirements: &[LayerKernelRequirements]) -> Result<ModelKernelPlan> {
    providers::catalog::compile_model_plan(requirements)
}

pub use operator_owner::streaming_attention;
pub use streaming_attention::{
    CudaStreamingF32GqaRequest, CudaStreamingF32GqaWorkspace, StreamingF32GqaResult,
    StreamingF32GqaStats,
};
