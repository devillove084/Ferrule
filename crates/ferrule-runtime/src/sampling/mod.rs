//! Sampling and output constraints.
//!
//! - `sampler`: temperature, top-k, top-p, min-p, repeat penalty sampling.
//! - `token_mask`: token-level constraints (JSON, max-length).
//! - `constraint`: composable constraint framework for structured generation.
//! - `structured`: structured-output helpers (JSON schema, regex).

pub mod constraint;
pub mod sampler;
pub mod structured;
pub mod token_mask;

pub use sampler::{Logprobs, Sampler, SamplingConfig};
pub use token_mask::{JsonConstraint, MaxLengthConstraint, SamplerMask, TokenConstraint};
