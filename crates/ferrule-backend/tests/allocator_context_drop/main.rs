//! CPU driver interposition: compile the real runtime + allocator, never execute CUDA.
#![cfg(feature = "cuda")]
#![allow(unsafe_code)] // This executable owns the FFI mock and real runtime boundary.

#[allow(dead_code)]
#[path = "../../src/cuda/allocator.rs"]
pub(crate) mod allocator;
#[path = "../../src/cuda/allocation_layout.rs"]
mod cuda_allocation_layout;
#[allow(dead_code)]
#[path = "../../src/cuda/runtime.rs"]
mod runtime;
// Existing runtime unit tests refer to this production module path.
mod cuda {
    pub(crate) use crate::allocator;
}
mod driver;
mod tests;
