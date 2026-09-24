//! Strict Qwen3.5-0.8B profile over the shared generic hybrid CPU decoder.
//!
//! Only the dense nested-text BF16 storage profile is accepted. Execution is
//! F32 on CPU through `HybridCpuDecoder`; MoE/FP8, CUDA, vision and MTP
//! execution remain unsupported. Visual/MTP tensors are validated as complete
//! known partitions and excluded from the text StateDictBinder input.

mod adapter;
mod config;
mod metadata;
mod name_mapper;
mod recipe;

pub use adapter::{Qwen35Adapter, Qwen35CpuRunner, Qwen35PrepareOptions};
pub use config::{
    Qwen35Config, Qwen35LayerType, Qwen35OutputGate, Qwen35Profile, Qwen35RopeConfig,
    Qwen35TextConfig, Qwen35TextSemantics, Qwen35VisionConfig,
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
        "Qwen3.5 unsupported profile '{0}': requires nested dense 0.8B qwen3_5; MoE and flat text exports are not implemented"
    )]
    Profile(String),
    #[error("Qwen3.5 unsupported quantized profile (including FP8)")]
    Quantization,
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
