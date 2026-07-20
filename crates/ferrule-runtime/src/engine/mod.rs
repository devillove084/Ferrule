//! Native resident multi-session execution.
//!
//! The engine owns request/session lifecycle, scheduling integration, explicit
//! per-sequence model state, and authoritative paged-KV transactions without
//! depending on a concrete model family.

mod diagnostic;
mod driver;
mod native_executor;

pub use diagnostic::PageManagedDiagnosticHarness;
pub use driver::{
    DSparkCycleTrace, ResidentActionKind, ResidentDriverStep, ResidentTokenEvent,
    ResidentTopKDriver, ResidentTopKDriverConfig, ResidentTopKDriverStats,
};

pub use native_executor::NativeMultiSessionExecutor;
