//! NVIDIA CUDA device runtime and kernel providers.
//!
//! CUDA Driver API handles, streams, events, memory and implementation-specific
//! kernels remain inside this module. Model preparation consumes provider-neutral
//! plans from [`crate::plan`].

mod allocator;
mod architecture;
mod benchmark;
#[doc(hidden)]
pub mod context;
mod counters;
pub mod diagnostics;
mod ffi;
mod graph;

pub mod operators {
    pub mod attention;
    mod contracts;
    pub mod kv;
    pub mod linear;
    pub mod moe;
    pub mod norm;
    pub mod rope;

    pub use contracts::*;
}

pub mod providers;
mod runtime;

use ferrule_common::Result;

use crate::plan::{LayerKernelRequirements, ModelKernelPlan};

pub use allocator::CudaAllocatorMetrics;
pub use benchmark::{CudaSmokeBenchmark, run_gemv_rms_smoke_benchmark, run_smoke_benchmark};
pub use graph::{
    CachedDecodeGraph, CudaGraphHandle, capture_decode_graph, cuda_graph_enabled,
    flash_attn_enabled,
};
pub use runtime::MemoryTier;

/// Compile provider-neutral model requirements for the CUDA backend.
pub fn compile_model_plan(requirements: &[LayerKernelRequirements]) -> Result<ModelKernelPlan> {
    providers::catalog::compile_model_plan(requirements)
}
