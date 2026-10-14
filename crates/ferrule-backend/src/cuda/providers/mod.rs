//! CUDA operator provider implementations and catalog discovery.

pub(crate) mod catalog;
pub mod core;
mod nccl_sha256;
pub(crate) mod tp_reduce;
pub mod cutlass;

pub use crate::cuda::architecture::{
    COMPILED_TARGET, CudaArchitectureFamily, CudaKernelCapabilities, CudaTarget,
    compiled_capabilities,
};
pub use crate::cuda::runtime::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig};

pub(crate) mod bf16_gemmex;

pub(crate) mod admission;
