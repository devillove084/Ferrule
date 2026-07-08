#![allow(clippy::needless_range_loop)]
//! Model metadata, artifact formats, quantization, and model-family execution.
pub mod artifact;
pub mod artifact_binding;
pub mod artifact_format;
pub mod artifact_group;
pub mod artifact_linear;
pub mod artifact_tensor;
pub mod chat;
pub mod descriptor;
pub mod families;
pub mod ffn;
pub mod gguf;
pub mod hyper_connection;
pub mod precision;
pub mod quant;
pub mod spec;
pub mod support;
pub mod tensor_policy;
pub mod tokenizer;

// Re-exports
pub use artifact::{
    ArtifactFormat, ArtifactIdentity, DtypeCount, HfAttentionTensorInfo, HfDenseLayerTensorInfo,
    HfFilePurpose, HfHyperConnectionTensorInfo, HfRepoFile, HfRoutedExpertTensorInfo,
    HfRouterTensorInfo, HfSafetensorsArtifact, HfSafetensorsIndex, HfSafetensorsInventory,
    HfSafetensorsShardSummary, HfSafetensorsTensorInfo, HfSharedExpertTensorInfo, InputArtifact,
    TensorRoleCount,
};
pub use descriptor::ModelDescriptor;
pub use spec::{
    AttentionKind, ModelFamily, MoeSpec, QuantFormatCount, RouterKind, TransformerSemantics,
    TransformerSpec, WeightSource,
};
pub use support::{
    validate_model_layout_bindings, AttentionLayout, AttentionPolicy, BoundRoleCount, EnginePlan,
    EnginePlanStatus, ExpertPolicy, FeedForwardKind, FeedForwardLayout, KvCacheShape, KvPolicy,
    LayerLayout, LayoutValidationReport, MissingPolicy, MissingRequiredRole, ModelLayout,
    ModelSupportContract, OptionalRoleStatus, ParallelismPlan, PolicyArea, PolicySet, QuantPolicy,
    ResidencyPolicy, RoleScope, RouterPolicy, SpeculationMode, SpeculationPolicy, TensorBinding,
    TensorRole, TokenizerPolicy, ValidationPolicy,
};
pub use tensor_policy::{GgufTensorPolicy, HfTensorPolicy, TensorClass, TensorClassCount};

// Re-exports from quant module
pub use quant::{f16_to_f32, f32_to_f16, QMatrix};

// Re-exports from moved runtime modules
pub use artifact_binding::{
    bind_attention_from_artifact_group, bind_attention_from_hf,
    bind_hyper_connection_from_artifact_group, bind_hyper_connection_from_hf,
    bind_hyper_connection_head_from_artifact_group, bind_hyper_connection_head_from_hf,
    bind_layer_norms_from_artifact_group, bind_router_from_artifact_group, bind_router_from_hf,
    bind_shared_swiglu_ffn_from_artifact_group, bind_shared_swiglu_ffn_from_hf,
    AttentionArtifactPayload, LayerNormArtifactPayload, RouterArtifactPayload,
};
pub use artifact_format::{
    decode_e8m0_scale, decode_fp4_e2m1_nibble, decode_fp4_e2m1_packed_low_first,
    decode_fp8_e4m3fn_byte, dequantize_fp4_e2m1_with_e8m0_scales,
    dequantize_fp8_e4m3fn_with_e8m0_scales, normalized_hadamard_transform_rows_in_place,
    simulate_fp4_e2m1_e8m0_activation_quant_in_place,
    simulate_fp8_e4m3fn_e8m0_activation_quant_in_place,
};
pub use artifact_group::{ArtifactGroupKind, ArtifactObjectGroup};
pub use artifact_linear::{
    ArtifactActivationQuantization, ArtifactLinearExecutionPolicy, ArtifactLinearFormat,
    ArtifactLinearPayload,
};
pub use artifact_tensor::{
    ArtifactDType, ArtifactTensorPayload, ArtifactTensorReader, ArtifactTensorSlice,
};
pub use chat::{detect_chat_template, ChatTemplate};
pub use ffn::SwiGluFfnPayload;
pub use hyper_connection::{
    hc_head_reference, hc_post_reference, hc_pre_reference, hc_split_sinkhorn_reference,
    HyperConnectionConfig, HyperConnectionHeadWeights, HyperConnectionPreOutput,
    HyperConnectionSplit, HyperConnectionWeights,
};
pub use precision::{PrecisionPolicy, QuantPreset, TensorDtypeOverride};
pub use tokenizer::TokenizerHandle;
