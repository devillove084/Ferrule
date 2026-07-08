//! Ferrule benchmark utilities.
//!
//! These modules are intentionally kept separate from the runtime; runtime execution
//! does not depend on benchmarks, manifests, or comparisons.

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub mod interactive_trace;
pub mod summary;

#[cfg(feature = "cuda")]
pub use interactive_trace::{
    compare_interactive_trace, GoldenTurn, InteractiveTrace, InteractiveTraceComparison,
    InteractiveTurnMismatch,
};
pub use summary::{RuntimeBenchSummary, RuntimeCounters};
