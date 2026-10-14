//! Test-only shared strict acceptance support. No public production API.
pub(super) mod oracle;
#[cfg(feature = "cuda")]
pub(super) mod pipeline;
