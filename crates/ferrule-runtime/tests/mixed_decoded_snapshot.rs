//! Focused CPU-only tests of the actual production observation leaf.
#![cfg(feature = "cuda")]
#[path = "../src/parallel/pipeline/mixed_cache_observation.rs"]
mod observation;

#[test]
fn absent_cache_snapshot_requires_no_cuda_owner() {
    observation::emit_cache_snapshot(0, 0, "ready", None);
}
