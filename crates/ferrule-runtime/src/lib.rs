#![allow(
    clippy::unnecessary_sort_by,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]
//! Ferrule Runtime — state-aware generation loops over model backends.
//!
//! This crate keeps tokenization, prefill/decode state, sampling, and chat
//! formatting out of the CLI so CPU and GPU backends share the same behavior.

pub use ferrule_graph as graph;

pub mod attention_backend;
pub mod attention_kernel;
pub mod backend_object_store;
pub mod chat;
pub mod config;
pub mod constraint;
pub mod dialects;
pub mod graph_builder;
pub mod graph_layer_binding;
pub mod graph_program;
pub mod graph_runtime;
pub mod graph_translate;
pub mod graph_validation;

pub mod execution {
    pub use crate::graph_runtime::{
        ExecutionBatch, ExecutionRow, ExecutionSegment, LogitsSelection,
    };
}

pub mod external_binding {
    pub use crate::graph_runtime::{
        ArtifactGroupKind, ExternalBinding, ExternalBindingKind, ExternalBindingPlan,
        ExternalResidency,
    };
}

pub mod artifact_binding;
pub mod artifact_format;
pub mod artifact_linear;
pub mod artifact_tensor;
pub mod expert_executor;
pub mod expert_handle;
pub mod expert_routing;
pub mod expert_streaming;
pub mod expert_telemetry;
pub mod ffn;
pub mod generation;
pub mod hyper_connection;
pub mod kv;
pub mod layer_binding;
pub mod models;
pub mod paged_kv;
pub mod precision;
pub mod prefix_cache;
pub mod profiler;
pub mod program;
pub mod radix_cache;
pub mod reference_graph_backend;
pub mod routed_moe;
pub mod runner;
pub mod sampler;
pub mod scheduler;
pub mod session;
pub mod shape_registry;
pub mod speculation;
pub mod stats;
pub mod structured;
pub mod token_mask;
pub mod tokenizer;
pub mod transformer_plan;

pub use attention_backend::{
    sliding_window_topk_indices, sparse_attention_reference, AttentionBackendKind,
    AttentionBackendPlan, AttentionMaskKind, SparseAttentionSpec,
};
pub use attention_kernel::AttentionKernel;
pub use backend_object_store::{
    materialize_dense_hf_externals, materialize_graph_hf_externals, materialize_hf_externals,
    materialize_hf_externals_for_family, ArtifactObjectGroup, BackendObject, BackendObjectStore,
    ExpertRegistryObject,
};
pub use chat::{detect_chat_template, ChatTemplate};
pub use config::ModelGenerationDefaults;
pub use expert_executor::{reference_linear, CpuReferenceExpertExecutor, ExpertExecutor};
pub use expert_handle::{
    CpuExpertHandleStore, ExpertComputeHandle, ExpertHandleStore, ExpertResidentFormat,
    ResidentExpertHandle,
};
pub use expert_routing::{
    ExpertRoute, ExpertRouterPolicy, RouterScoreFunction, RouterSelectionPolicy,
};
pub use expert_streaming::{
    ExpertArtifactPayload, ExpertComputeBundle, ExpertEvictRequest, ExpertId, ExpertLinearFormat,
    ExpertLinearPayload, ExpertLoadReason, ExpertLoadRequest, ExpertLoadSource, ExpertMatrixKind,
    ExpertStorageTier, ExpertStreamingPlanner, ExpertStreamingPolicy, ExpertStreamingReader,
    ExpertStreamingStep, ExpertTensorComponent, ExpertTensorKey, ExpertTensorPayload,
    ExpertTensorSlice,
};
pub use expert_telemetry::ExpertTelemetry;
pub use ffn::SwiGluFfnPayload;

pub use generation::{GenerationConfig, GenerationResult, InferenceEngine, TokenEvent};
pub use graph_builder::{
    build_graph_program_from_descriptor, build_graph_program_from_descriptor_with_options,
    build_graph_program_from_runtime_plan, build_graph_program_from_runtime_plan_with_options,
    GraphProgramBuildOptions,
};
pub use graph_layer_binding::{GraphLayerObjects, GraphObjectRef};
pub use graph_program::{GraphProgram, GraphProgramProfile};
pub use graph_runtime::{
    ArtifactGroupKind, ExecutionBatch, ExecutionRow, ExecutionSegment, ExternalBinding,
    ExternalBindingKind, ExternalBindingPlan, ExternalResidency, LogitsSelection,
};
pub use graph_translate::{
    build_dense_decoder_graph_program, build_dense_decoder_graph_program_with_options,
    build_semantic_transformer_graph_program,
    build_semantic_transformer_graph_program_with_options, uses_semantic_artifact_groups,
    DenseGraphTranslationOptions, SemanticGraphTranslationOptions,
};
pub use graph_validation::{
    validate_graph_program, validate_graph_program_with_registry, GraphValidationReport,
};
pub use hyper_connection::{
    hc_head_reference, hc_post_reference, hc_pre_reference, hc_split_sinkhorn_reference,
    HyperConnectionConfig, HyperConnectionHeadWeights, HyperConnectionPreOutput,
    HyperConnectionSplit, HyperConnectionWeights,
};
pub use kv::{
    ContiguousKvCache, KvCache, KvCacheLayout, KvHandle, KvLayerView, MultiSessionKvCache,
    SequenceKvCache,
};
pub use layer_binding::{
    bind_layer_artifact_from_graph_objects, bind_layer_artifact_from_hf,
    new_layer_execution_state_from_graph_objects, GraphLayerBindingOptions, LayerArtifactBinding,
    LayerExecutionState, LayerKvState, LayerStepOutput, ReferenceLayerExecutor,
};

pub use profiler::{KernelProfiler, Profiler, TimedRegion};
pub use program::GenerationProgram;

pub use reference_graph_backend::{ReferenceGraphBackend, ReferenceGraphExecutor};

pub use routed_moe::{
    execute_routed_moe_reference, execute_routed_moe_reference_with_handles,
    execute_routed_moe_with_artifact_router_reference,
    execute_routed_moe_with_artifact_router_reference_with_handles, RoutedMoeStepOutput,
};
pub use runner::{unsupported_runtime_message, ModelInfo, ModelRunner, RuntimeRunner};
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
pub use artifact_binding::{
    bind_attention_from_artifact_group, bind_attention_from_hf,
    bind_hyper_connection_from_artifact_group, bind_hyper_connection_from_hf,
    bind_hyper_connection_head_from_artifact_group, bind_hyper_connection_head_from_hf,
    bind_layer_norms_from_artifact_group, bind_router_from_artifact_group, bind_router_from_hf,
    bind_shared_swiglu_ffn_from_artifact_group, bind_shared_swiglu_ffn_from_hf,
    AttentionArtifactPayload, LayerNormArtifactPayload, RouterArtifactPayload,
};
pub use artifact_linear::{ArtifactLinearFormat, ArtifactLinearPayload};
pub use artifact_tensor::{
    ArtifactDType, ArtifactTensorPayload, ArtifactTensorReader, ArtifactTensorSlice,
};
pub use scheduler::{BatchedScheduler, PreemptionPolicy, Scheduler};
pub use session::{GenerateRequest, RequestId, SequenceState, SequenceStatus, SessionId};
pub use shape_registry::TransformerShapeRegistry;
pub use speculation::{
    run_speculative_step, DraftModel, SpeculationMetrics, SpeculativeDecodingPolicy,
    SpeculativeStepOutput, TargetModel,
};
pub use stats::GenerateStats;
pub use token_mask::{JsonConstraint, MaxLengthConstraint, SamplerMask, TokenConstraint};
pub use tokenizer::TokenizerHandle;
pub use transformer_plan::{
    AttentionStepPlan, ExpertResidencyMode, FeedForwardStepPlan, RuntimeAttachment,
    RuntimeEpilogue, RuntimePrologue, TransformerLayerPlan, TransformerRuntimePlan,
};

#[cfg(feature = "cuda")]
pub use runner::GpuOlmoeRunner;
