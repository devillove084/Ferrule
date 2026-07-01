#![allow(clippy::unnecessary_sort_by, clippy::needless_range_loop)]
//! Ferrule Runtime — state-aware generation loops over model backends.
//!
//! This crate keeps tokenization, prefill/decode state, sampling, and chat
//! formatting out of the CLI so CPU and GPU backends share the same behavior.

pub mod attention_backend;
pub mod chat;
pub mod config;
pub mod constraint;
pub mod cpu_kv;
pub mod expert_executor;
pub mod expert_handle;
pub mod expert_routing;
pub mod expert_streaming;
pub mod ffn;
pub mod generation;
pub mod hyper_connection;
pub mod kv;
pub mod paged_kv;
pub mod perplexity;
pub mod precision;
pub mod prefix_cache;
pub mod profiler;
pub mod program;
pub mod radix_cache;
pub mod residency;
pub mod routed_moe;
pub mod runner;
pub mod sampler;
pub mod scheduler;
pub mod session;
pub mod source_binding;
pub mod source_format;
pub mod source_linear;
pub mod source_tensor;
pub mod stats;
pub mod structured;
pub mod tokenizer;
pub mod transformer_plan;

pub use attention_backend::{
    sliding_window_topk_indices, sparse_attention_reference, AttentionBackendKind,
    AttentionBackendPlan, AttentionMaskKind, SparseAttentionSpec,
};
pub use chat::{detect_chat_template, ChatTemplate};
pub use config::ModelGenerationDefaults;
pub use cpu_kv::CpuContiguousKvState;
pub use expert_executor::{reference_linear, CpuReferenceExpertExecutor, ExpertExecutor};
pub use expert_handle::{
    CpuExpertHandleStore, ExpertComputeHandle, ExpertHandleStore, ExpertResidentFormat,
    ResidentExpertHandle,
};
pub use expert_routing::{
    ExpertRoute, ExpertRouterPolicy, RouterScoreFunction, RouterSelectionPolicy,
};
pub use expert_streaming::{
    ExpertComputeBundle, ExpertEvictRequest, ExpertId, ExpertLinearFormat, ExpertLinearPayload,
    ExpertLoadReason, ExpertLoadRequest, ExpertMatrixKind, ExpertSource, ExpertSourcePayload,
    ExpertStorageTier, ExpertStreamingPlanner, ExpertStreamingPolicy, ExpertStreamingReader,
    ExpertStreamingStep, ExpertTensorComponent, ExpertTensorKey, ExpertTensorPayload,
    ExpertTensorSlice,
};
pub use ffn::SwiGluFfnPayload;
pub use generation::{GenerationConfig, GenerationResult, InferenceEngine, TokenEvent};
pub use hyper_connection::{
    hc_head_reference, hc_post_reference, hc_pre_reference, hc_split_sinkhorn_reference,
    HyperConnectionConfig, HyperConnectionHeadWeights, HyperConnectionPreOutput,
    HyperConnectionSplit, HyperConnectionWeights,
};
pub use kv::{
    ContiguousKvCache, KvCache, KvCacheLayout, KvHandle, KvLayerView, MultiSessionKvCache,
    SequenceKvCache,
};
pub use profiler::Profiler;
pub use program::GenerationProgram;
pub use routed_moe::{
    execute_routed_moe_reference, execute_routed_moe_reference_with_handles,
    execute_routed_moe_with_source_router_reference,
    execute_routed_moe_with_source_router_reference_with_handles, RoutedMoeStepOutput,
};
pub use runner::{CpuModelRunner, CpuOlmoeRunner, ModelInfo, ModelRunner};
pub use sampler::{Logprobs, Sampler, SamplingConfig};

/// Argmax: return the index of the maximum logit value.
/// Used by golden-token regression tests.
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold(
            (0usize, logits[0]),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
        )
        .0 as u32
}
pub use scheduler::{BatchedScheduler, PreemptionPolicy, Scheduler};
pub use session::{GenerateRequest, RequestId, SequenceState, SequenceStatus, SessionId};
pub use source_binding::{
    bind_attention_from_hf, bind_hyper_connection_from_hf, bind_hyper_connection_head_from_hf,
    bind_router_from_hf, bind_shared_swiglu_ffn_from_hf, AttentionSourcePayload,
    RouterSourcePayload,
};
pub use source_linear::{SourceLinearFormat, SourceLinearPayload};
pub use source_tensor::{SourceDType, SourceTensorPayload, SourceTensorReader, SourceTensorSlice};
pub use stats::GenerateStats;
pub use tokenizer::TokenizerHandle;
pub use transformer_plan::{
    AttentionStepPlan, ExpertResidencyMode, FeedForwardStepPlan, RuntimeAttachment,
    RuntimeEpilogue, RuntimePrologue, TransformerLayerPlan, TransformerRuntimePlan,
};

#[cfg(feature = "cuda")]
pub use runner::{GpuModelRunner, GpuOlmoeRunner};
