//! Public factory admission and CPU budgets; CUDA numerical coverage is opt-in.
#[path = "dense_bf16/admission.rs"]
mod admission;
#[cfg(feature = "cuda")]
#[path = "dense_bf16/cuda.rs"]
mod cuda;
#[path = "dense_bf16/fixture.rs"]
mod fixture;
