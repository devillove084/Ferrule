//! Transformer execution steps for the CUDA backend.
//!
//! These modules keep the hot path monomorphized and concrete while separating
//! state ownership (KV), attention, MoE, and logits projection. New model
//! families should plug in by changing weight-layout / attention / router
//! policies, not by duplicating an entire model runner.

pub(crate) mod attention;
pub(crate) mod executor;
pub(crate) mod kv;
pub(crate) mod logits;
pub(crate) mod moe;
pub mod source_expert;
pub mod sparse_attention;

pub(crate) use executor::CudaTransformerExecutor;
pub(crate) use kv::CudaContiguousKvCache;
