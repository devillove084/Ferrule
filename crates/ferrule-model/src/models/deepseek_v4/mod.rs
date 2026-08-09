//! DeepSeek-V4 recipe-bound decoder runtime.

#[cfg(feature = "cuda")]
mod adapter;
#[cfg(feature = "cuda")]
mod checkpoint;
mod config;
mod name_mapper;
mod recipe;

#[cfg(feature = "cuda")]
pub use adapter::DeepSeekDecoderComposition;
#[cfg(feature = "cuda")]
pub use adapter::{
    DeepSeekV4Adapter, DeepSeekV4AttentionProfileStats, DeepSeekV4LayerProfileStats,
    DeepSeekV4LayerRuntimeStats, DeepSeekV4LoadProfile, DeepSeekV4ObservabilitySnapshot,
    DeepSeekV4OperatorRuntimeCounters,
};
#[cfg(feature = "cuda")]
pub use checkpoint::DeepSeekV4Checkpoint;
pub(super) use config::DeepSeekV4Config;
pub(super) use name_mapper::DeepSeekV4NameMapper;
pub use recipe::DeepSeekV4Recipe;
#[cfg(feature = "cuda")]
pub use recipe::prepare;
#[cfg(feature = "cuda")]
pub use recipe::{
    DeepSeekV4OutputProfileStats, DeepSeekV4PrepareOptions, DeepSeekV4PrepareProfile,
};

#[cfg(test)]
mod tests;
