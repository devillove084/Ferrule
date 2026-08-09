//! Concrete model family implementations.
//!
//! Family modules provide checkpoint bindings and, once runtime support is ready,
//! full forward-path implementations. The runtime crate (`ferrule-runtime`)
//! defines execution traits and scheduling infrastructure; only families with a
//! concrete runner are registered for execution.

pub mod common;
pub mod deepseek_v4;
pub mod qwen3;
