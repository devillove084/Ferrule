//! Concrete model family implementations.
//!
//! Each module provides a full forward-path implementation for a specific model
//! architecture. The runtime crate (`ferrule-runtime`) defines execution traits
//! and scheduling infrastructure; this crate provides the concrete runners that
//! implement those traits.

pub mod deepseek_v4;
