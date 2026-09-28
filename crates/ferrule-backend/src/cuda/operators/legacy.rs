//! Explicit compatibility surface for pre-facade CUDA launch entry points.
//!
//! These functions retain raw provider-shaped buffers and native submission
//! contracts for backend migration. New production code should use the owning
//! family APIs instead. This module is intentionally the only aggregate legacy
//! export for the remaining projection/compressor/HC/shared-FFN functions.

pub use crate::cuda::providers::cutlass::{
    bf16_compressor, fp8_projection, fp8_query_a_kv, hc_producer, main_project_norm, mla_output,
    shared_ffn,
};

// Preserve the grouped preparation namespace from the earlier facade slice.
pub use crate::cuda::operators::moe::grouped_fp4::legacy::{
    grouped_fp4_moe_workspace_size, mxfp4_sfb_storage_bytes, prepare_mxfp4_sfb,
};
