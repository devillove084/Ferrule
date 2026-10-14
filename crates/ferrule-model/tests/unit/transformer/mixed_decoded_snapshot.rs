//! Snapshot invariants use the actual CPU ledger; never initialize CUDA.
use super::tests::{MIB, descriptor, pending, touch};
use super::*;
fn descriptors() -> Vec<DecodedDescriptor> {
    vec![
        descriptor(0, 60 * MIB, None),
        descriptor(1, MIB, Some(0)),
        descriptor(2, MIB, Some(1)),
    ]
}
fn custody() -> DecodedCustody<usize> {
    pending(64 * MIB, &descriptors())
}
fn check(s: MixedDecodedCacheSnapshot) {
    assert_eq!(
        s.physical_current_bytes,
        s.pinned_grants
            + s.active_routed_grants
            + s.pooled_grants
            + s.free_slot_grants
            + s.unused_tail_bytes
            + s.pending_reserved_grants
    );
    assert_eq!(
        s.pin_target_grants,
        s.pinned_grants + s.unallocated_pin_grants
    );
    assert!(s.unallocated_pin_grants <= s.pending_reserved_grants);
    assert!(s.arena_backing_grants <= s.physical_current_bytes);
    assert_eq!(s.segment_slack_bytes, 0);
    assert_eq!(s.pending_segment_slack_bytes, 0);
    assert_eq!(s.physical_current_bytes, s.physical_peak_bytes);
    assert_eq!(s.physical_peak_bytes, s.capacity_grants);
}
#[test]
fn decoded_snapshot_prepare_execute_aftercomplete_are_disjoint_and_read_only() {
    let mut c = custody();
    let ds = descriptors();
    let first = c.snapshot(false, true, false, false, None);
    check(first);
    assert_eq!(first.phase, MixedDecodedCachePhase::Preparing);
    assert_eq!(first.image_generation, 73);
    assert_eq!(first.pin_target_grants, 60 * MIB);
    assert_eq!(first.pinned_grants, 0);
    assert_eq!(first.pending_reserved_grants, 64 * MIB);
    assert_eq!(first.arena_backing_grants, 0);
    assert_eq!(first.pending_allocation_count, 1);
    assert_eq!(first.profiled_activity, None);
    c.install_arena(0, 64 * MIB, |_, o, _| Ok(o)).unwrap();
    let installed = c.snapshot(false, false, false, false, None);
    check(installed);
    assert_eq!(installed.arena_backing_grants, 64 * MIB);
    assert_eq!(installed.pinned_grants, 60 * MIB);
    assert_eq!(installed.free_slot_grants, 2 * MIB);
    assert_eq!(installed.unused_tail_bytes, 2 * MIB);
    assert_eq!(installed.pending_allocation_count, 0);
    assert_eq!(installed.pending_reserved_grants, 0);
    touch(&mut c, &ds[0]);
    c.complete(true).unwrap();
    let ready = c.snapshot(true, false, false, false, None);
    check(ready);
    assert_eq!(ready.phase, MixedDecodedCachePhase::Idle);
    let Lookup::Decode(view) = c.acquire(&ds[1]).unwrap() else {
        panic!()
    };
    c.mark_ready(&ds[1]).unwrap();
    let executing = c.snapshot(true, true, false, false, None);
    check(executing);
    assert_eq!(executing.phase, MixedDecodedCachePhase::Executing);
    assert_eq!(executing.active_routed_grants, MIB);
    assert_eq!(executing.free_slot_grants, MIB);
    assert_eq!(executing.pending_lease_count, 1);
    let logical = c.stats();
    let physical = c.physical_stats();
    for _ in 0..10 {
        assert_eq!(c.snapshot(true, true, false, false, None), executing);
    }
    assert_eq!(c.stats(), logical);
    assert_eq!(c.physical_stats(), physical);
    drop(view);
    c.complete(true).unwrap();
    let completed = c.snapshot(true, false, false, false, None);
    check(completed);
    assert_eq!(completed.pending_lease_count, 0);
    assert_eq!(completed.borrowed_key_count, 0);
    assert_eq!(
        completed.physical_current_bytes,
        executing.physical_current_bytes
    );
}
#[test]
fn decoded_snapshot_pool_failed_reservations_and_unknown_remain_visible() {
    let mut c = custody();
    let ds = descriptors();
    c.install_arena(0, 0, |_, o, _| Ok(o)).unwrap();
    touch(&mut c, &ds[1]);
    c.complete(false).unwrap();
    let pooled = c.snapshot(true, false, false, false, None);
    check(pooled);
    assert_eq!(pooled.pooled_grants, MIB);
    assert_eq!(pooled.free_slot_grants, MIB);
    assert_eq!(pooled.driver_allocation_growth_bytes, 0);
    let external = c.snapshot(true, false, true, false, None);
    check(external);
    assert_eq!(external.unknown_reserved_bytes, 64 * MIB);
    assert!(!c.stats().quarantined, "snapshot is observation only");
    c.quarantine();
    let unknown = c.snapshot(true, false, false, false, None);
    check(unknown);
    assert_eq!(unknown.phase, MixedDecodedCachePhase::Quarantined);
    assert_eq!(unknown.unknown_reserved_bytes, 64 * MIB);
    assert_eq!(unknown.arena_backing_grants, 64 * MIB);
    let mut failed = custody();
    failed.arena_allocation_failed(64 * MIB).unwrap();
    let failed = failed.snapshot(false, false, false, false, None);
    check(failed);
    assert_eq!(failed.arena_backing_grants, 0);
    assert_eq!(failed.pending_reserved_grants, 64 * MIB);
    assert_eq!(failed.pending_allocation_count, 0);
    assert_eq!(failed.unknown_reserved_bytes, 64 * MIB);
}
#[test]
fn decoded_snapshot_partial_view_failure_partitions_installed_root_once() {
    let mut c = custody();
    let mut calls = 0;
    assert!(
        c.install_arena(0, 64 * MIB, |_, o, _| {
            calls += 1;
            if calls == 3 {
                Err(invalid("slice failed"))
            } else {
                Ok(o)
            }
        })
        .is_err()
    );
    let s = c.snapshot(false, false, false, false, None);
    check(s);
    assert_eq!(s.arena_backing_grants, 64 * MIB);
    assert_eq!(s.pinned_grants, 60 * MIB);
    assert_eq!(s.free_slot_grants, MIB);
    assert_eq!(s.pending_reserved_grants, MIB);
    assert_eq!(s.unused_tail_bytes, 2 * MIB);
    assert_eq!(s.unknown_reserved_bytes, 64 * MIB);
}
#[test]
fn decoded_snapshot_profile_flags_never_turn_unobserved_work_into_zero() {
    let mut c = custody();
    let off = c.snapshot(true, false, false, false, None);
    assert!(!off.rank_performance_enabled);
    assert!(!off.rank_performance_active);
    assert_eq!(off.profiled_activity, None);
    // Requested profiling, but tracing gate has never admitted an invocation.
    let gated_off = c.snapshot(true, false, false, true, None);
    assert!(gated_off.rank_performance_enabled);
    assert_eq!(gated_off.profiled_activity, None);
    let empty = c.snapshot(true, false, false, true, Some([0; 4]));
    assert!(empty.rank_performance_active);
    assert_eq!(empty.profiled_activity.unwrap().profiled_invocations, 1);
    let active = c.snapshot(true, true, false, true, Some([7, 2, 4096, 1]));
    assert_eq!(active.profiled_activity.unwrap().hits, 7);
    assert_eq!(
        c.snapshot(true, false, false, false, None)
            .profiled_activity,
        None,
        "snapshot must not accumulate an active scope"
    );
    c.record_profiled_invocation([7, 2, 4096, 1]);
    let ended = c.snapshot(true, false, false, true, None);
    assert!(!ended.rank_performance_active);
    assert_eq!(ended.profiled_activity, active.profiled_activity);
    let second = c
        .snapshot(true, true, false, true, Some([3, 1, 512, 0]))
        .profiled_activity
        .unwrap();
    assert_eq!(
        second,
        MixedDecodedCacheActivity {
            profiled_invocations: 2,
            hits: 10,
            misses: 3,
            decoded_bytes: 4608,
            evictions: 1
        }
    );
    c.record_profiled_invocation([3, 1, 512, 0]);
    let disabled = c.snapshot(true, false, false, false, None);
    assert!(!disabled.rank_performance_enabled);
    assert_eq!(disabled.profiled_activity, Some(second));
    assert!(serde_json::to_value(off).unwrap()["profiled_activity"].is_null());
    assert_eq!(
        serde_json::to_value(disabled).unwrap()["profiled_activity"]["hits"],
        10
    );
}
#[test]
fn decoded_snapshot_profile_aggregation_saturates() {
    let mut c = custody();
    c.record_profiled_invocation([u64::MAX; 4]);
    c.record_profiled_invocation([1; 4]);
    let a = c
        .snapshot(true, false, false, false, None)
        .profiled_activity
        .unwrap();
    assert_eq!(a.hits, u64::MAX);
    assert_eq!(a.misses, u64::MAX);
    assert_eq!(a.decoded_bytes, u64::MAX);
    assert_eq!(a.evictions, u64::MAX);
    assert_eq!(a.profiled_invocations, 2);
}
#[cfg(feature = "cuda")]
#[test]
fn decoded_snapshot_public_api_is_a_shared_read_only_borrow() {
    let _: fn(
        &crate::transformer::CudaStandardDecoderOperators,
    ) -> Option<crate::transformer::MixedDecodedCacheSnapshot> =
        crate::transformer::CudaStandardDecoderOperators::mixed_decoded_cache_snapshot;
}
