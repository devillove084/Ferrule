//! Driver-free tests compile the production allocator against a mock runtime.
#[allow(dead_code)]
#[path = "../../src/cuda/allocator.rs"]
mod allocator;
#[path = "../../src/cuda/allocation_layout.rs"]
mod cuda_allocation_layout;
mod runtime;
mod tests;
