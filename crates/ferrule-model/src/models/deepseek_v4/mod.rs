//! DeepSeek-V4 concrete model implementation.
//!
//! This module provides a full forward-path runner for DeepSeek-V4 / Flash / Proposal.
//! It is the first hard target for Ferrule's model bring-up contract.
//!
//! ## Module layout
//!
//! | Module | Responsibility |
//! |---|---|
//! | `config` | `DeepSeekV4Config`, `DeepSeekV4AttentionConfig`, `DeepSeekV4RopeParams` |
//! | `checkpoint` | `DeepSeekV4Checkpoint` — HF weight loading and tensor binding |
//! | `operators` | `DeepSeekV4OperatorContext` — CPU/CUDA operator dispatch |
//! | `cuda_cache` | `DeepSeekV4CudaOperatorCache` — device-resident weight/KV cache |
//! | `attention` | `DeepSeekV4Attention`, compressor, window KV, attention cache |
//! | `layer` | `DeepSeekV4Layer` - one transformer block (HC + attention + MoE) |
//! | `proposal_attachment` | Proposal protocol and stages stored under the checkpoint `mtp.*` namespace |
//! | `runner` | `DeepSeekV4Runner` - `ModelRunner` implementation |
//! | `helpers` | Free functions: RMSNorm, RoPE, YaRN, top-k, cache keys |

pub mod attention;
pub mod checkpoint;
mod checkpoint_binding;
pub mod config;
#[cfg(feature = "cuda")]
pub mod cuda_cache;

#[cfg(feature = "cuda")]
pub(crate) mod cuda_materialization {
    pub(crate) use crate::moe::cuda_materialization::CudaSharedExpertSubsystem as DeepSeekV4SharedExpertSubsystem;
}
pub mod helpers;
pub mod layer;
pub mod operators;
pub mod prepared;
pub mod proposal_attachment;

pub mod runner;
pub mod sequence;

#[cfg(test)]
mod local_checkpoint_tests;
#[cfg(test)]
mod tests;

// Re-exports
pub use attention::{
    DeepSeekV4Attention, DeepSeekV4AttentionCache, DeepSeekV4CompressedAttentionPayload,
    DeepSeekV4CompressorPayload, DeepSeekV4CompressorState, DeepSeekV4IndexerPayload,
    DeepSeekV4WindowKvCache,
};
pub use checkpoint::DeepSeekV4Checkpoint;
pub use config::{
    DeepSeekV4AttentionConfig, DeepSeekV4Config, DeepSeekV4RopeParams, ProposalAttachmentConfig,
};

pub use layer::{
    DeepSeekV4Layer, DeepSeekV4LayerExpertRuntime, DeepSeekV4LayerState, DeepSeekV4LayerStepOutput,
};
pub use operators::{
    DeepSeekV4AttentionProfileStats, DeepSeekV4LayerProfileStats, DeepSeekV4OperatorContext,
    DeepSeekV4OperatorRuntimeCounters,
};
pub use prepared::{
    DeepSeekV4ExecutionPolicy, DeepSeekV4KvLayoutSchema, DeepSeekV4PrepareProfile,
    DeepSeekV4PreparedModelPlan, DeepSeekV4PreparedResources, prepare,
};
pub use proposal_attachment::{
    DeepSeekV4ProposalAttachment, DeepSeekV4ProposalHeads, DeepSeekV4ProposalProtocol,
    DeepSeekV4ProposalStage,
};
pub use runner::{
    DeepSeekV4LayerRuntimeStats, DeepSeekV4LoadProfile, DeepSeekV4ObservabilitySnapshot,
    DeepSeekV4OutputProfileStats, DeepSeekV4PrepareOptions, DeepSeekV4Runner,
};
pub use sequence::DeepSeekV4SequenceExecutionState;
