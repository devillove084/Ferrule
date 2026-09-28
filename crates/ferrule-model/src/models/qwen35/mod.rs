//! Exact Qwen3.5 profiles over the shared generic hybrid decoder descriptors.
//!
//! Dense 0.8B BF16 and MoE 35B-A3B numeric FP8 metadata are accepted.
//! Only the dense profile supports execution here (F32 `HybridCpuDecoder`).
//! MoE FP8, vision and MTP execution remain unsupported. Visual/MTP tensors are
//! validated as complete known partitions and excluded from text binding.
//!
//! # Standard backend integration
//!
//! `Qwen35Recipe::spec` describes all layers, including `Moe::shared_expert_gate`:
//! a bias-free hidden -> 1 linear whose sigmoid multiplies the shared SwiGLU
//! output before it is added to the weighted routed sum. Its semantic role is
//! `TensorRole::SharedExpertOutputGate` (not `SharedExpertGate`, the SwiGLU gate).
//! The shared standard forward calls `StandardDecoderOperators::shared_expert_gate`
//! with [rows, hidden] output and [rows, 1] logits. Backends default to unsupported.
//!
//! `Qwen35Adapter::bind_hf_metadata` returns the strict metadata and generic bound
//! resources without reading weight payloads or a tokenizer. Numeric FP8 weight
//! and scale parts share a logical parameter/role/residency; physical storage is
//! `Dense`, never native E8M0. Pass `config.numeric_fp8_encoding().unwrap()` to
//! `BoundParameter::numeric_fp8_source` or `StateDictMaterializer::numeric_fp8_tile`.
//! These storage interfaces do not imply an executable backend. The 35B adapter
//! rejects execution until standard numeric FP8 and shared-gate GPU support land.
//! BF16 packed experts require a separate recipe/transform and remain unsupported.
//! The legacy checkpoint stage classifier also needs to classify the new
//! `SharedExpertOutputGate` role as FeedForward before GPU stage planning;
//! that checkpoint-owned integration is deliberately outside this profile.

mod adapter;
mod config;
mod metadata;
mod name_mapper;
mod recipe;

pub use adapter::{Qwen35Adapter, Qwen35CpuRunner, Qwen35PrepareOptions};
pub use config::{
    Qwen35Config, Qwen35Fp8Config, Qwen35LayerType, Qwen35OutputGate, Qwen35Profile,
    Qwen35RopeConfig, Qwen35TextConfig, Qwen35TextSemantics, Qwen35VisionConfig,
};
pub use metadata::Qwen35Metadata;
pub use name_mapper::{
    Qwen35HfNameMapper, Qwen35TensorPartition, Qwen35TensorPartitionKind, Qwen35TensorSpec,
};
pub use recipe::Qwen35Recipe;

/// Capability rejection, retained as the source of ferrule_common::Error::ModelSource.
/// Invalid fields within the supported profile remain ordinary config errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Qwen35Unsupported {
    #[error(
        "Qwen3.5 unsupported profile '{0}': requires exact nested dense 0.8B BF16 or MoE 35B-A3B FP8; flat text exports are not implemented"
    )]
    Profile(String),
    #[error("Qwen3.5 unsupported quantization for this profile (dense 0.8B requires BF16)")]
    Quantization,
    #[error(
        "Qwen3.5 BF16 packed experts are unsupported: require a separate packed-expert SplitTransform recipe"
    )]
    PackedBf16Experts,
    #[error(
        "Qwen3.5 MoE 35B FP8 supports metadata/binding only; standard numeric FP8 execution is not implemented"
    )]
    Execution,
    #[error("Qwen3.5 unsupported backend {0:?}: only dense 0.8B F32 CPU execution is available")]
    Backend(crate::ModelExecutionBackend),
}

impl From<Qwen35Unsupported> for ferrule_common::Error {
    fn from(source: Qwen35Unsupported) -> Self {
        Self::ModelSource {
            source: Box::new(source),
        }
    }
}

fn model_error(message: impl Into<String>) -> ferrule_common::Error {
    ferrule_common::Error::Model {
        message: format!("Qwen3.5 metadata: {}", message.into()),
    }
}
