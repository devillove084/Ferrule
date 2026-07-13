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
//! | `graph` | Compute graph IR, semantic-plan builders, translators, validation, dialects |
//! | `cache` | KV cache: contiguous, paged, prefix, radix |
//! | `sampling` | Sampler, token masks, constraints, structured output |
//! | `scheduling` | Resident request scheduling, action planning, and session lifecycle |
//! | `storage` | Storage residency: catalog, placement, transfer, policy |
//! | `generation` | `InferenceEngine` and top-k prefill/decode loops over runner capabilities |
//! | `engine` | Native resident multi-session driver and executor lifecycle |
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
    BackendObject, BackendObjectStore, ExpertRegistryObject, materialize_graph_hf_externals,
    materialize_hf_externals, materialize_hf_externals_for_family,
};
pub use cache::{
    BLOCK_SIZE, BlockId, BlockTable, ContiguousKvCache, FixedSequenceSlotPool, KvCache,
    KvCacheDtype, KvCacheLayout, KvHandle, KvLayerView, MultiSessionKvCache, PagedKvCache,
    PagedSequenceKvCache, SequenceKvCache, SequenceSlotPool,
};
pub use engine::{
    NativeMultiSessionExecutor, PageManagedDiagnosticHarness, ResidentActionKind,
    ResidentDriverStep, ResidentTokenEvent, ResidentTopKDriver, ResidentTopKDriverConfig,
    ResidentTopKDriverStats,
};
pub use expert_residency::{
    ExpertInstallIntent, ExpertInstallPrepareOutcome, ExpertInstallReason, ExpertKey, ExpertLease,
    ExpertResidencyControl, ExpertResidencyController, ExpertResidencyCoordinator,
    ExpertResidencyCoordinatorStats, ExpertResidencyGrant, ExpertResidencyRequirements,
    ExpertResidencyStats, ExpertSlotBinding, ExpertSlotGeneration, ExpertSlotId,
    PreparedExpertInstall,
};
pub use generation::{GenerationConfig, GenerationResult, InferenceEngine, TokenEvent};
pub use graph::builder::{
    GraphProgramBuildOptions, build_graph_program_from_descriptor,
    build_graph_program_from_descriptor_with_options, build_graph_program_from_semantic_plan,
    build_graph_program_from_semantic_plan_with_options,
};
pub use graph::external_bindings::{
    ExternalBinding, ExternalBindingKind, ExternalBindingPlan, ExternalResidency,
};
pub use graph::layer_binding::{GraphLayerObjects, GraphObjectRef};
pub use graph::program::{GraphProgram, GraphProgramProfile};
pub use graph::shape_registry::TransformerShapeRegistry;
pub use graph::translate::{
    DenseGraphTranslationOptions, SemanticGraphTranslationOptions,
    build_dense_decoder_graph_program, build_dense_decoder_graph_program_with_options,
    build_semantic_transformer_graph_program,
    build_semantic_transformer_graph_program_with_options, uses_semantic_artifact_groups,
};
pub use graph::validation::{
    GraphValidationReport, validate_graph_program, validate_graph_program_with_registry,
};

pub use layer_binding::{
    GraphLayerBindingOptions, LayerArtifactBinding, LayerExecutionState, LayerExpertRuntime,
    LayerKvState, LayerStepOutput, ReferenceLayerExecutor, bind_layer_artifact_from_graph_objects,
    bind_layer_artifact_from_hf, new_layer_expert_runtime_from_graph_objects,
};

pub use profiler::{KernelProfiler, Profiler, TimedRegion};
pub use program::GenerationProgram;
pub use reference_graph_backend::{
    ReferenceGraphBackend, ReferenceGraphExecutor, ReferenceGraphSequenceState,
};

pub use sampling::{JsonConstraint, MaxLengthConstraint, SamplerMask, TokenConstraint};
pub use sampling::{Logprobs, Sampler, SamplingConfig};

pub use scheduling::{
    CancelRequestResult, DecodeAction, LogitsSelection, PrefillChunkAction, ResidentScheduler,
    ResidentSchedulerConfig, SchedulerAction, plan_prefill_chunk,
};
pub use scheduling::{
    GenerateRequest, RequestId, SequenceFinishReason, SequenceState, SequenceStatus, SessionId,
};

pub use speculation::{
    DraftModel, SpeculationMetrics, SpeculativeDecodingPolicy, SpeculativeStepOutput, TargetModel,
    run_speculative_step,
};
pub use stats::GenerateStats;
