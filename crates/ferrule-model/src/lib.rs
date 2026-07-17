#![allow(clippy::needless_range_loop)]
//! Model metadata, artifact formats, quantization, and model-family execution.
//!
//! ## Crate layout
//!
//! | Module | Responsibility |
//! |---|---|
//! | `spec` | Model family, attention kind, transformer spec enums |
//! | `descriptor` | `ModelDescriptor` — parsed config.json + tokenizer + inventory |
//! | `support` | Policy types, layout contracts, engine plans |
//! | `families` | Per-family tensor name classification (HF/GGUF → semantic roles) |
//! | `artifact` | Artifact inventory, index, binding, format decoding, tensor I/O |
//! | `moe` | Expert streaming, handle stores, routing, MoE step orchestration |
//! | `runner` | `ModelRunner` trait — the execution boundary |
//! | `attention_backend` | Sparse attention reference + backend contracts |
//! | `transformer_plan` | Semantic transformer layer/step runtime plan |
//! | `hyper_connection` | Hyper-connection math (pre/post/head reference) |
//! | `ffn` | SwiGLU FFN payload |
//! | `models` | Concrete model implementations (e.g. DeepSeek-V4) |

// ── Top-level modules ─────────────────────────────────────────────────────
pub mod attention_backend;
pub mod chat;
pub mod descriptor;
pub mod execution;
pub mod ffn;
pub mod hyper_connection;
pub mod precision;
pub mod runner;
pub mod semantic;
pub mod semantic_plan;
pub mod spec;
pub mod tensor_policy;
pub mod tokenizer;

// ── Sub-directory modules ─────────────────────────────────────────────────
pub mod artifact;
pub mod families;
pub mod gguf;
pub mod models;
pub mod moe;
pub mod quant;
pub mod support;

// ── Re-exports: execution ─────────────────────────────────────────────────
pub use execution::{
    ArenaLease, ExecutionShapeKey, ModelExecutionBackend, PersistentArenaPool,
    PersistentArenaPoolStats, PreparedModel, SequenceStateCore, SequenceStepBinding,
};

// ── Re-exports: spec ──────────────────────────────────────────────────────
pub use spec::{
    AttentionKind, ModelFamily, MoeSpec, QuantFormatCount, RouterKind, TransformerSemantics,
    TransformerSpec, WeightSource,
};

// ── Re-exports: descriptor ────────────────────────────────────────────────
pub use descriptor::ModelDescriptor;

// ── Re-exports: support ───────────────────────────────────────────────────
pub use support::{
    AttentionLayout, AttentionPolicy, BoundRoleCount, EnginePlan, EnginePlanStatus, ExpertPolicy,
    FeedForwardKind, FeedForwardLayout, KvCacheShape, KvPolicy, LayerLayout,
    LayoutValidationReport, MissingPolicy, MissingRequiredRole, ModelLayout, ModelSupportContract,
    OptionalRoleStatus, ParallelismPlan, PolicyArea, PolicySet, QuantPolicy, ResidencyPolicy,
    RoleScope, RouterPolicy, SpeculationMode, SpeculationPolicy, TensorBinding, TensorRole,
    TokenizerPolicy, ValidationPolicy, validate_model_layout_bindings,
};

// ── Re-exports: semantic ──────────────────────────────────────────────────
pub use semantic::{
    ArtifactTensorPart, AttentionTensorKind, AttentionTensorRef, DenseLayerTensorKind,
    DenseLayerTensorRef, HyperConnectionStage, HyperConnectionTensorKind, HyperConnectionTensorRef,
    RoutedExpertMatrix, RoutedExpertTensorPart, RoutedExpertTensorRef, RouterTensorKind,
    RouterTensorRef, SharedExpertTensorRef,
};

// ── Re-exports: tensor_policy ─────────────────────────────────────────────
pub use tensor_policy::{GgufTensorPolicy, HfTensorPolicy, TensorClass, TensorClassCount};

// ── Re-exports: quant ─────────────────────────────────────────────────────
pub use quant::{QMatrix, f16_to_f32, f32_to_f16};

// ── Re-exports: artifact ──────────────────────────────────────────────────
pub use artifact::{
    ArtifactActivationQuantization, ArtifactLinearExecutionPolicy, ArtifactLinearFormat,
    ArtifactLinearPayload,
};
pub use artifact::{
    ArtifactDType, ArtifactFormat, ArtifactGroupKind, ArtifactIdentity, ArtifactObjectGroup,
    ArtifactTensorPayload, ArtifactTensorReader, ArtifactTensorSlice, DtypeCount,
    HfAttentionTensorInfo, HfDenseLayerTensorInfo, HfFilePurpose, HfHyperConnectionTensorInfo,
    HfRepoFile, HfRoutedExpertTensorInfo, HfRouterTensorInfo, HfSafetensorsArtifact,
    HfSafetensorsIndex, HfSafetensorsInventory, HfSafetensorsShardSummary, HfSafetensorsTensorInfo,
    HfSharedExpertTensorInfo, InputArtifact, TensorRoleCount,
};
pub use artifact::{
    LayerNormArtifactPayload, MlaAttentionArtifactPayload, RouterArtifactPayload,
    bind_attention_from_artifact_group, bind_attention_from_hf,
    bind_hyper_connection_from_artifact_group, bind_hyper_connection_from_hf,
    bind_hyper_connection_head_from_artifact_group, bind_hyper_connection_head_from_hf,
    bind_layer_norms_from_artifact_group, bind_router_from_artifact_group, bind_router_from_hf,
    bind_shared_swiglu_ffn_from_artifact_group, bind_shared_swiglu_ffn_from_hf,
};
pub use artifact::{
    decode_e8m0_scale, decode_fp4_e2m1_nibble, decode_fp4_e2m1_packed_low_first,
    decode_fp8_e4m3fn_byte, dequantize_fp4_e2m1_with_e8m0_scales,
    dequantize_fp8_e4m3fn_with_e8m0_scales, normalized_hadamard_transform_rows_in_place,
    simulate_fp4_e2m1_e8m0_activation_quant_in_place,
    simulate_fp8_e4m3fn_e8m0_activation_quant_in_place,
};

// ── Re-exports: moe ───────────────────────────────────────────────────────
pub use moe::{
    CpuExpertHandleStore, CpuReferenceExpertExecutor, ExpertArtifactPayload, ExpertComputeBundle,
    ExpertComputeHandle, ExpertEvictRequest, ExpertExecutor, ExpertHandleStore, ExpertId,
    ExpertLinearFormat, ExpertLinearPayload, ExpertLoadReason, ExpertLoadRequest, ExpertLoadSource,
    ExpertMatrixKind, ExpertMemoryPolicy, ExpertResidentFormat, ExpertRoute, ExpertRouterPolicy,
    ExpertStorageTier, ExpertStreamingPlanner, ExpertStreamingPolicy, ExpertStreamingReader,
    ExpertStreamingStep, ExpertTelemetry, ExpertTensorComponent, ExpertTensorKey,
    ExpertTensorPayload, ExpertTensorSlice, HostStagedExpertCache, ResidentExpertHandle,
    RoutedMoeStepOutput, RouterScoreFunction, RouterSelectionPolicy, execute_routed_moe_reference,
    execute_routed_moe_reference_with_handles, execute_routed_moe_with_artifact_router_reference,
    execute_routed_moe_with_artifact_router_reference_with_handles, read_experts_concurrent,
    reference_linear,
};

// ── Re-exports: runner ────────────────────────────────────────────────────
pub use runner::{
    DsparkProposal, DsparkProposalRunner, ExpertIoModelRunner, ModelInfo, ModelRunner,
    MultiSessionRunner, PrefillMode, TokenLogit, TopKModelRunner, unsupported_runtime_message,
};

// ── Re-exports: attention_backend ─────────────────────────────────────────
pub use attention_backend::{
    SparseAttentionSpec, sliding_window_topk_indices, sparse_attention_reference,
};

// ── Re-exports: semantic plan ─────────────────────────────────────────────
pub use semantic_plan::{
    AttentionSemantic, ExpertResidency, FeedForwardSemantic, SemanticAttachment, SemanticEpilogue,
    SemanticPrologue, TransformerLayerSemantic, TransformerSemanticPlan,
};

// ── Re-exports: hyper_connection ──────────────────────────────────────────
pub use hyper_connection::{
    HyperConnectionConfig, HyperConnectionHeadWeights, HyperConnectionPreOutput,
    HyperConnectionSplit, HyperConnectionWeights, hc_head_reference, hc_post_reference,
    hc_pre_reference, hc_split_sinkhorn_reference,
};

// ── Re-exports: ffn ───────────────────────────────────────────────────────
pub use ffn::SwiGluFfnPayload;

// ── Re-exports: chat ──────────────────────────────────────────────────────
pub use chat::{ChatFormatError, ChatMessage, ChatRole, ChatTemplate, detect_chat_template};

// ── Re-exports: precision ─────────────────────────────────────────────────
pub use precision::{PrecisionPolicy, QuantPreset, TensorDtypeOverride};

// ── Re-exports: tokenizer ─────────────────────────────────────────────────
pub use tokenizer::{IncrementalDecodeState, TokenizerHandle};
