//! Actual observed-capacity replay uses the shared validator's no-expand sink.
use super::*;

pub fn replay_runtime_prefix_lru(
    batches: &[RuntimeKvTraceSnapshot],
) -> TraceResult<RuntimePrefixReplay> {
    validation::validate_semantics(
        batches,
        RuntimeKvProjectionLimits::default(),
        validation::no_expand,
    )
    .map(|validated| validated.report)
}
