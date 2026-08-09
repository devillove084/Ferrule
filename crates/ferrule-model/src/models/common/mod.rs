//! Family-neutral transformer building blocks shared by model implementations.
//!
//! Everything in this module must stay independent of any concrete model
//! family (DeepSeek, Qwen, ...): shape validation, pure tensor math,
//! RoPE/YaRN frequency geometry, and HF `config.json` parsing helpers.

pub mod config_json;

#[cfg(feature = "cuda")]
pub mod rope;
#[cfg(feature = "cuda")]
pub mod shape;
