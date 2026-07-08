#![allow(
    clippy::unnecessary_sort_by,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]
//! Ferrule Runtime — state-aware generation loops over model backends.
//!
//! This crate owns scheduling, sampling, session lifecycle, KV cache management,
//! graph execution, and generation algorithms over model-runner capabilities.
//! Runner traits and concrete model implementations live in `ferrule-model`.
//!
//! ## Crate layout
//!
//! | Module | Responsibility |
//! |---|---|
//! | `graph` | Compute graph IR, builders, translators, validation, dialects |
//! | `cache` | KV cache: contiguous, paged, prefix, radix |
//! | `sampling` | Sampler, token masks, constraints, structured output |
//! | `scheduling` | Batched scheduler, session/sequence lifecycle |
//! | `storage` | Storage residency: catalog, placement, transfer, policy |
//! | `generation` | `InferenceEngine` and top-k prefill/decode loops over runner capabilities |
//! | `engine` | Resident `EngineWorker` / lazy worker lifecycle over `TopKModelRunner` |
//! | `program` | `GenerationProgram` — chained constrained generation |
//! | `speculation` | Draft/target model speculation framework |
//! | `profiler` | Kernel profiling and timing regions |
//! | `stats` | Generation statistics |
//! | `attention_kernel` | Tiled attention kernel (CPU reference) |
//! | `backend_object_store` | Materialize HF artifacts into backend objects |
//! | `reference_graph_backend` | CPU reference graph executor |
//! | `layer_binding` | Bind artifact weights to layer execution state |
//! | `expert_residency` | Expert residency backend trait + storage adapters |

// ── Sub-directory modules ─────────────────────────────────────────────────
pub mod cache;
pub mod graph;
pub mod sampling;
pub mod scheduling;
pub mod storage;

// ── Top-level modules ─────────────────────────────────────────────────────
pub mod attention_kernel;
pub mod backend_object_store;
pub mod engine;
pub mod expert_residency;
pub mod generation;
pub mod layer_binding;
pub mod profiler;
pub mod program;
pub mod reference_graph_backend;
pub mod speculation;
pub mod stats;

// ── Convenience re-exports ────────────────────────────────────────────────
pub use attention_kernel::AttentionKernel;
pub use backend_object_store::{
    materialize_dense_hf_externals, materialize_graph_hf_externals, materialize_hf_externals,
    materialize_hf_externals_for_family, BackendObject, BackendObjectStore, ExpertRegistryObject,
};
pub use cache::{
    BlockId, BlockTable, ContiguousKvCache, KvCache, KvCacheDtype, KvCacheLayout, KvHandle,
    KvLayerView, MultiSessionKvCache, PagedKvCache, PagedSequenceKvCache, SequenceKvCache,
    BLOCK_SIZE,
};
pub use engine::{
    EngineWorker, EngineWorkerStats, LazyEngineLoadStats, LazyEngineWorker, ResidentActionExecutor,
    ResidentActionExecutorConfig, ResidentActionKind, ResidentDriverStep, ResidentTokenEvent,
    ResidentTopKDriver, ResidentTopKDriverConfig, ResidentTopKDriverStats, TopKDecodeState,
    TopKDecodeStep,
};
pub use generation::{
    generate_topk_from_candidates, generate_topk_turn, GenerationConfig, GenerationResult,
    InferenceEngine, TokenEvent, TopKFinishReason, TopKTokenEvent, TopKTurnResult,
};
pub use graph::builder::{
    build_graph_program_from_descriptor, build_graph_program_from_descriptor_with_options,
    build_graph_program_from_runtime_plan, build_graph_program_from_runtime_plan_with_options,
    GraphProgramBuildOptions,
};
pub use graph::layer_binding::{GraphLayerObjects, GraphObjectRef};
pub use graph::program::{GraphProgram, GraphProgramProfile};
pub use graph::runtime::{
    ExecutionBatch, ExecutionOutput, ExecutionRow, ExecutionRowOutput, ExecutionSegment,
    ExternalBinding, ExternalBindingKind, ExternalBindingPlan, ExternalResidency, LogitsSelection,
    RowLogits,
};
pub use graph::shape_registry::TransformerShapeRegistry;
pub use graph::translate::{
    build_dense_decoder_graph_program, build_dense_decoder_graph_program_with_options,
    build_semantic_transformer_graph_program,
    build_semantic_transformer_graph_program_with_options, uses_semantic_artifact_groups,
    DenseGraphTranslationOptions, SemanticGraphTranslationOptions,
};
pub use graph::validation::{
    validate_graph_program, validate_graph_program_with_registry, GraphValidationReport,
};

pub use layer_binding::{
    bind_layer_artifact_from_graph_objects, bind_layer_artifact_from_hf,
    new_layer_execution_state_from_graph_objects, GraphLayerBindingOptions, LayerArtifactBinding,
    LayerExecutionState, LayerKvState, LayerStepOutput, ReferenceLayerExecutor,
};

pub use profiler::{KernelProfiler, Profiler, TimedRegion};
pub use program::GenerationProgram;
pub use reference_graph_backend::{ReferenceGraphBackend, ReferenceGraphExecutor};

pub use sampling::{JsonConstraint, MaxLengthConstraint, SamplerMask, TokenConstraint};
pub use sampling::{Logprobs, Sampler, SamplingConfig};

pub use scheduling::{
    plan_prefill_chunk, BatchedScheduler, DecodeAction, PreemptionPolicy, PrefillChunkAction,
    ResidentScheduler, ResidentSchedulerConfig, Scheduler, SchedulerAction,
};
pub use scheduling::{
    GenerateRequest, RequestId, SequenceFinishReason, SequenceState, SequenceStatus, SessionId,
};

pub use speculation::{
    run_speculative_step, DraftModel, SpeculationMetrics, SpeculativeDecodingPolicy,
    SpeculativeStepOutput, TargetModel,
};
pub use stats::GenerateStats;

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
