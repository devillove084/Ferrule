//! Native HF BF16 hybrid admission, boundary contracts and opt-in CUDA acceptance.
#[path = "hybrid_bf16/admission.rs"]
mod admission;
#[path = "hybrid_bf16/contracts.rs"]
mod contracts;
#[cfg(feature = "cuda")]
#[path = "hybrid_bf16/cuda.rs"]
mod cuda;
#[path = "hybrid_bf16/fixture.rs"]
mod fixture;

#[path = "hybrid_bf16/tiled.rs"]
mod tiled;

#[path = "hybrid_bf16/mixed.rs"]
mod mixed;

#[path = "hybrid_bf16/chunk/budget.rs"]
mod chunk;
