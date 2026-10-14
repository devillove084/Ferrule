//! CPU-only serialization at runtime observation boundaries, never CUDA work.
use super::*;
use ferrule_model::transformer::{MixedDecodedCacheActivity, MixedDecodedCachePhase};

fn snapshot() -> MixedDecodedCacheSnapshot {
    MixedDecodedCacheSnapshot {
        image_generation: 7,
        phase: MixedDecodedCachePhase::Idle,
        published: true,
        quarantined: false,
        capacity_grants: 64 << 20,
        arena_backing_grants: 64 << 20,
        pin_target_grants: 8 << 20,
        pinned_grants: 8 << 20,
        unallocated_pin_grants: 0,
        active_routed_grants: 16 << 20,
        pooled_grants: 4 << 20,
        free_slot_grants: 32 << 20,
        unused_tail_bytes: 4 << 20,
        slot_padding_bytes: 1024,
        pending_reserved_grants: 0,
        pending_allocation_count: 0,
        pending_lease_count: 0,
        borrowed_key_count: 0,
        segment_slack_bytes: 0,
        pending_segment_slack_bytes: 0,
        physical_current_bytes: 64 << 20,
        physical_peak_bytes: 64 << 20,
        driver_allocation_growth_bytes: 64 << 20,
        unknown_reserved_bytes: 0,
        backing_reuses: 3,
        native_fallbacks: 2,
        evictions: 4,
        rank_performance_enabled: false,
        rank_performance_active: false,
        profiled_activity: None,
    }
}

#[test]
fn decoded_snapshot_runtime_json_preserves_physical_fields_and_unobserved_activity() {
    let s = snapshot();
    let value: serde_json::Value = serde_json::from_str(&cache_snapshot_json(&s)).unwrap();
    assert_eq!(CACHE_SNAPSHOT_SCHEMA_VERSION, 2);
    assert_eq!(value["arena_backing_grants"], 64 << 20);
    assert_eq!(value["free_slot_grants"], 32 << 20);
    assert_eq!(value["unused_tail_bytes"], 4 << 20);
    assert_eq!(value["slot_padding_bytes"], 1024);
    assert_eq!(value["pinned_grants"], 8 << 20);
    assert_eq!(value["active_routed_grants"], 16 << 20);
    assert_eq!(value["pooled_grants"], 4 << 20);
    assert_eq!(value["pending_reserved_grants"], 0);
    assert_eq!(value["segment_slack_bytes"], 0);
    assert_eq!(value["pending_segment_slack_bytes"], 0);
    assert_eq!(value["physical_current_bytes"], 64 << 20);
    assert_eq!(value["physical_peak_bytes"], 64 << 20);
    assert_eq!(value["native_fallbacks"], 2);
    assert!(value["profiled_activity"].is_null());
    assert_eq!(value["phase"], "idle");
    assert_eq!(cache_snapshot_json(&s), cache_snapshot_json(&s));
}

#[test]
fn decoded_snapshot_runtime_json_preserves_profile_coverage_and_quarantine() {
    let mut s = snapshot();
    s.phase = MixedDecodedCachePhase::Quarantined;
    s.quarantined = true;
    s.unknown_reserved_bytes = s.physical_current_bytes;
    s.profiled_activity = Some(MixedDecodedCacheActivity {
        profiled_invocations: 2,
        hits: 9,
        misses: 4,
        decoded_bytes: 512,
        evictions: 1,
    });
    let value: serde_json::Value = serde_json::from_str(&cache_snapshot_json(&s)).unwrap();
    assert_eq!(value["phase"], "quarantined");
    assert_eq!(value["unknown_reserved_bytes"], 64 << 20);
    assert_eq!(value["profiled_activity"]["profiled_invocations"], 2);
    assert_eq!(value["profiled_activity"]["decoded_bytes"], 512);
    assert_eq!(value["profiled_activity"]["hits"], 9);
    assert_eq!(value["rank_performance_enabled"], false);
    assert_eq!(value["rank_performance_active"], false);
}

#[test]
fn decoded_snapshot_runtime_json_preserves_preallocation_and_failed_root_reservation() {
    let mut s = snapshot();
    s.phase = MixedDecodedCachePhase::Preparing;
    s.published = false;
    s.arena_backing_grants = 0;
    s.pinned_grants = 0;
    s.unallocated_pin_grants = s.pin_target_grants;
    s.active_routed_grants = 0;
    s.pooled_grants = 0;
    s.free_slot_grants = 0;
    s.unused_tail_bytes = 0;
    s.pending_reserved_grants = s.capacity_grants;
    s.pending_allocation_count = 1;
    s.driver_allocation_growth_bytes = 0;
    let value: serde_json::Value = serde_json::from_str(&cache_snapshot_json(&s)).unwrap();
    assert_eq!(value["phase"], "preparing");
    assert_eq!(value["arena_backing_grants"], 0);
    assert_eq!(value["pending_reserved_grants"], 64 << 20);
    assert_eq!(value["physical_current_bytes"], 64 << 20);
    assert_eq!(value["pending_allocation_count"], 1);
    assert!(value["profiled_activity"].is_null());

    s.phase = MixedDecodedCachePhase::Quarantined;
    s.quarantined = true;
    s.pending_allocation_count = 0;
    s.unknown_reserved_bytes = s.capacity_grants;
    let value: serde_json::Value = serde_json::from_str(&cache_snapshot_json(&s)).unwrap();
    assert_eq!(value["phase"], "quarantined");
    assert_eq!(value["arena_backing_grants"], 0);
    assert_eq!(value["unknown_reserved_bytes"], 64 << 20);
    assert_eq!(value["pending_reserved_grants"], 64 << 20);
    assert_eq!(value["physical_current_bytes"], 64 << 20);
}
