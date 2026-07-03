//! Ferrule benchmark, evaluation, and reporting utilities.
//!
//! This crate intentionally sits outside `ferrule-runtime`: it may depend on the
//! runtime to observe graph programs and model runners, but runtime execution must
//! not depend on benchmark manifests, reference comparisons, JSON summaries, or
//! smoke-test harnesses.

pub mod first_token_smoke;
pub mod perplexity;
pub mod pk_manifest;
pub mod reference_compare;
pub mod reference_manifest;
pub mod summary;
pub mod token_debug;

pub use first_token_smoke::{
    run_first_token_smoke, FirstTokenModel, FirstTokenSmokeReport, FirstTokenSmokeStatus,
    FirstTokenUnsupportedReason,
};
pub use perplexity::{compute_perplexity, PerplexityResult};
pub use pk_manifest::{
    render_pk_markdown_summary, CompetitivePkManifest, HardwareSpec, PkCommand, PkManifestId,
    PkMetricKind, PkMetricValue, PkModelId, PkPromptSetId, PkQuantizationId, PkResultRecord,
    PkRunSpec, PkRuntimeKind, PkSpeculationConfig,
};
pub use reference_compare::{
    compare_reference_observation, ReferenceComparisonReport, ReferenceMismatch,
    ReferenceObservation,
};
pub use reference_manifest::{
    GoldenPrompt, PromptId, ReferenceArtifact, ReferenceCommand, ReferenceCommandManifest,
    ReferenceEngineKind, ReferenceManifestId, ReferenceTopKLogit,
};
pub use summary::{
    BackendObjectSummary, ExpertRuntimeCounters, GraphProgramSummary, KernelCounters,
    MemoryCounters, RuntimeBenchSummary, RuntimeCounters, RuntimeTimingCounters, TransferCounters,
};
pub use token_debug::{TokenDebug, TokenDebugEntry};
