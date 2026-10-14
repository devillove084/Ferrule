//! Explicit typed tiled-GEMM candidate; no change to model/W8 default dispatch.
#[cfg(feature = "cuda")]
#[path = "numeric_bf16_tiled/gpu.rs"]
mod gpu;
#[path = "numeric_bf16_tiled/metadata.rs"]
mod metadata;
