//! Real libcuda qualification, one exact test per supervised process.
//! Interposition forwards every call to libcuda; faults alter only return status.
#![cfg(feature = "cuda")]
#![allow(unsafe_code, dead_code)]
#[path = "../../src/cuda/allocator.rs"]
mod allocator;
#[path = "../../src/cuda/allocation_layout.rs"]
mod cuda_allocation_layout;
#[path = "../../src/cuda/runtime.rs"]
mod runtime;
mod cuda {
    pub(crate) use crate::allocator;
}
mod driver;
mod tests;
