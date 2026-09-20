//! Dense/MoE Qwen3 recipes, artifact binding, and generic decoder adapters.

mod adapter;
mod checkpoint;
mod config;
mod name_mapper;
mod recipe;

pub use adapter::{
    Qwen3DenseAdapter, Qwen3DensePrepareOptions, Qwen3MoeAdapter, Qwen3MoePrepareOptions,
};
pub use checkpoint::Qwen3MoeCheckpoint;
pub use config::{Qwen3DenseConfig, Qwen3MoeConfig};
pub(super) use name_mapper::Qwen3HfNameMapper;
pub use recipe::{Qwen3DenseRecipe, Qwen3MoeRecipe};

#[cfg(test)]
mod tests;
