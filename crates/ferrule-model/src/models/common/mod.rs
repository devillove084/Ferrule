//! Family-neutral transformer building blocks shared by model implementations.
//!
//! Everything in this module must stay independent of any concrete model
//! family (DeepSeek, Qwen, ...): artifact tensor loading, shape validation,
//! pure tensor math, RoPE/YaRN frequency geometry, and HF `config.json`
//! parsing helpers.

pub mod checkpoint;
pub mod config_json;
pub mod math;
pub mod rope;
pub mod shape;
