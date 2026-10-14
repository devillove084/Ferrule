//! CPU-reference composition tests; no GPU execution is claimed.
#![cfg(feature = "cpu-reference")]
#[path = "hybrid_parallel/tests.rs"]
mod tests;
