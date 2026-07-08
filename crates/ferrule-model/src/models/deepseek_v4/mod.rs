//! DeepSeek-V4 concrete model implementation.
//!
//! This module provides a full forward-path runner for DeepSeek-V4 / Flash / DSpark.
//! It is the first hard target for Ferrule's model bring-up contract.
//!
//! ## Module layout
//!
//! | Module | Responsibility |
//! |---|---|
//! | `config` | `DeepSeekV4Config`, `DeepSeekV4AttentionConfig`, `DeepSeekV4RopeParams` |
//! | `artifact` | `DeepSeekV4ArtifactModel` — HF weight loading and tensor binding |
//! | `operators` | `DeepSeekV4OperatorContext` — CPU/CUDA operator dispatch |
//! | `cuda_cache` | `DeepSeekV4CudaOperatorCache` — device-resident weight/KV cache |
//! | `attention` | `DeepSeekV4Attention`, compressor, window KV, attention cache |
//! | `layer` | `DeepSeekV4Layer` — one transformer block (HC + attention + MoE) |
//! | `runner` | `DeepSeekV4ReferenceRunner` — `ModelRunner` implementation |
//! | `helpers` | Free functions: RMSNorm, RoPE, YaRN, top-k, cache keys |

pub mod artifact;
pub mod attention;
pub mod config;
#[cfg(feature = "cuda")]
pub mod cuda_cache;
pub mod helpers;
pub mod layer;
pub mod operators;
pub mod runner;

#[cfg(test)]
mod tests;

// Re-exports
pub use artifact::{ArtifactTensor2D, DeepSeekV4ArtifactModel};
pub use attention::{
    DeepSeekV4Attention, DeepSeekV4AttentionCache, DeepSeekV4CompressedAttentionPayload,
    DeepSeekV4CompressorPayload, DeepSeekV4CompressorState, DeepSeekV4IndexerPayload,
    DeepSeekV4WindowKvCache,
};
pub use config::{DeepSeekV4AttentionConfig, DeepSeekV4Config, DeepSeekV4RopeParams};
pub use layer::{DeepSeekV4Layer, DeepSeekV4LayerState, DeepSeekV4LayerStepOutput};
pub use operators::{
    DeepSeekV4AttentionProfileStats, DeepSeekV4LayerProfileStats, DeepSeekV4Logit,
    DeepSeekV4OperatorBackend, DeepSeekV4OperatorContext, DeepSeekV4OperatorRuntimeCounters,
};
pub use runner::{
    DeepSeekV4LayerRuntimeStats, DeepSeekV4OutputProfileStats, DeepSeekV4PrefillRuntimeStats,
    DeepSeekV4ReferenceOptions, DeepSeekV4ReferenceRunner,
};
