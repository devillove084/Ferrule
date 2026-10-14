//! Bounded ready/shutdown observations only; no CUDA or per-call logging.
use ferrule_model::transformer::MixedDecodedCacheSnapshot;

// v1 described segmented allocations; v2 partitions one retained root reservation.
const CACHE_SNAPSHOT_SCHEMA_VERSION: u64 = 2;

fn cache_snapshot_json(snapshot: &MixedDecodedCacheSnapshot) -> String {
    // Scalar-only Serialize; None activity remains JSON null, never fake zeros.
    serde_json::to_string(snapshot).expect("decoded-cache snapshot contains serializable scalars")
}

pub(super) fn emit_cache_snapshot(
    owner: u32,
    device: usize,
    stage: &str,
    snapshot: Option<MixedDecodedCacheSnapshot>,
) {
    if let Some(snapshot) = snapshot
        && tracing::enabled!(target: "ferrule_cache", tracing::Level::INFO)
    {
        tracing::info!(target: "ferrule_cache", event = "mixed_decoded_cache_snapshot", schema_version = CACHE_SNAPSHOT_SCHEMA_VERSION,
            owner, device, stage, snapshot_json = %cache_snapshot_json(&snapshot),
            "Mixed decoded cache host ledger; not a CUDA completion receipt or device-global memory measurement");
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/parallel/hybrid_tensor/decoded_snapshot.rs"]
mod decoded_snapshot_tests;
