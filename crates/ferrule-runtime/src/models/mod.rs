//! Model-specific runtime implementations.
//!
//! Ferrule keeps scheduling, sampling, artifact tensor IO, and CUDA operator surfaces
//! generic. Architecture-specific forward semantics live under this module instead
//! of a universal Transformer interpreter.

pub mod deepseek_v4;
