//! Backend-neutral CPU execution layer.
//!
//! This module owns semantic operator contracts, reference fallback, typed paged
//! KV storage, and native CPU provider dispatch. Model crates adapt checkpoint
//! payloads and transaction metadata into these model-agnostic APIs.

mod attention;
mod kv;
mod moe;
mod operators;
mod provider;

pub use attention::{KvHistory, PagedCausalGqa, PagedKvHistory, paged_causal_gqa};
pub use kv::{
    CpuKvPlaneStorage, CpuKvView, CpuPagedKvPool, CpuPagedKvTransaction, KvCapacity, KvEndProgress,
    KvPageStatus, PagedKvBatch, PagedKvPrepare, PagedKvSequence,
};
pub use moe::{
    ExpertLinearRef, ExpertSwiGluRef, RouterRoutes, SwiGluRef, execute_reference_expert,
    execute_reference_expert_with_hidden_transform, expert_linear, softmax_topk_routes,
    swiglu_rows, weighted_reduce,
};
pub use operators::{
    CpuExecutionPrecision, HostRows, LinearRef, LinearWeight, RopeRef, RotaryFrequencyParams,
    RotaryPairing, RotaryRegion, RowsArenaId, RowsDType, RowsShape,
    apply_rotary_split_half_indexed, apply_rotary_tail_scaled, bf16_rne, bf16_rne_word,
    bf16_word_value, dot, embedding_rows, linear_rows, reference_linear_rows_into, residual_rows,
    rms_norm, rms_norm_heads_in_place, rms_norm_rows, rope_rows, rotary_correction_dimension,
    rotary_correction_range, rotary_frequency, rotary_linear_ramp,
};
pub use provider::{CpuCapabilities, CpuOperatorProvider, NativeCpuProvider, ReferenceCpuProvider};
