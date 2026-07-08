//! Artifact descriptions and processing.
//!
//! This module covers the full artifact lifecycle:
//!
//! - **Inventory** (`inventory`): scanning HF safetensors shards, building
//!   tensor metadata without loading payloads.
//! - **Index** (`index`): parsing `model.safetensors.index.json`.
//! - **Identity** (`identity`): format detection and artifact identity.
//! - **Input** (`input`): the `InputArtifact` enum (local path, HF repo, etc.).
//! - **HF** (`hf`): HuggingFace file purposes and repo descriptions.
//!
//! Artifact processing sub-modules:
//!
//! - **Binding** (`binding`): map artifact tensor names → semantic weight
//!   roles (attention, router, expert, hyper-connection, norms, FFN).
//! - **Format** (`format`): decode quantized formats (FP4 e2m1, FP8 e4m3,
//!   E8M0 scales, Hadamard transforms).
//! - **Group** (`group`): semantic artifact group kinds for graph externals.
//! - **Linear** (`linear`): artifact linear payload + execution policy
//!   (quantized weight + activation quantization metadata).
//! - **Tensor** (`tensor`): typed tensor reader, slice, and payload for
//!   zero-copy mmap access to safetensors shards.

pub mod binding;
pub mod format;
pub mod group;
pub mod hf;
pub mod identity;
pub mod index;
pub mod input;
pub mod inventory;
pub mod linear;
pub mod tensor;

// Re-exports — inventory / index / identity / input / hf
pub use hf::{HfFilePurpose, HfRepoFile, HfSafetensorsArtifact};
pub use identity::{ArtifactFormat, ArtifactIdentity};
pub use index::HfSafetensorsIndex;
pub use input::InputArtifact;
pub use inventory::{
    DtypeCount, HfAttentionTensorInfo, HfDenseLayerTensorInfo, HfHyperConnectionTensorInfo,
    HfRoutedExpertTensorInfo, HfRouterTensorInfo, HfSafetensorsInventory,
    HfSafetensorsShardSummary, HfSafetensorsTensorInfo, HfSharedExpertTensorInfo, TensorRoleCount,
};

// Re-exports — binding
#[allow(deprecated)]
pub use binding::{
    bind_attention_from_artifact_group, bind_attention_from_hf,
    bind_hyper_connection_from_artifact_group, bind_hyper_connection_from_hf,
    bind_hyper_connection_head_from_artifact_group, bind_hyper_connection_head_from_hf,
    bind_layer_norms_from_artifact_group, bind_router_from_artifact_group, bind_router_from_hf,
    bind_shared_swiglu_ffn_from_artifact_group, bind_shared_swiglu_ffn_from_hf,
    AttentionArtifactPayload, LayerNormArtifactPayload, MlaAttentionArtifactPayload,
    RouterArtifactPayload,
};

// Re-exports — format
pub use format::{
    decode_e8m0_scale, decode_fp4_e2m1_nibble, decode_fp4_e2m1_packed_low_first,
    decode_fp8_e4m3fn_byte, dequantize_fp4_e2m1_with_e8m0_scales,
    dequantize_fp8_e4m3fn_with_e8m0_scales, normalized_hadamard_transform_rows_in_place,
    simulate_fp4_e2m1_e8m0_activation_quant_in_place,
    simulate_fp8_e4m3fn_e8m0_activation_quant_in_place,
};

// Re-exports — group
pub use group::{ArtifactGroupKind, ArtifactObjectGroup};

// Re-exports — linear
pub use linear::{
    ArtifactActivationQuantization, ArtifactLinearExecutionPolicy, ArtifactLinearFormat,
    ArtifactLinearPayload,
};

// Re-exports — tensor
pub use tensor::{ArtifactDType, ArtifactTensorPayload, ArtifactTensorReader, ArtifactTensorSlice};

#[cfg(test)]
mod tests;
