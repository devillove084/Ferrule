mod forwarding;

use std::num::NonZeroU64;

use ferrule_common::io_protocol::QuiescenceEvidence;

use crate::resource_observability::{
    ResourceClosure, ResourceEvidence, ResourceObservabilityLedger, ResourceObservationScope,
    ResourceObservationSnapshot,
};

fn scope(owner: u64, generation: u64) -> ResourceObservationScope {
    ResourceObservationScope {
        owner: NonZeroU64::new(owner).unwrap(),
        generation: NonZeroU64::new(generation).unwrap(),
    }
}

#[test]
fn unobserved_is_not_a_fabricated_zero_or_closure() {
    let ledger = ResourceObservabilityLedger::new(scope(1, 1));
    let snapshot = ledger.snapshot();
    assert_eq!(
        snapshot,
        ResourceObservationSnapshot::unobserved(scope(1, 1))
    );
    assert_eq!(snapshot.resident_bytes, ResourceEvidence::Unobserved);
    assert_eq!(snapshot.pending_operations, ResourceEvidence::Unobserved);
    assert_eq!(snapshot.scratch_bytes, ResourceEvidence::Unobserved);
    assert_eq!(snapshot.quarantine, ResourceEvidence::Unobserved);
    assert_eq!(snapshot.active_credit_grants, ResourceEvidence::Unobserved);
    assert_eq!(snapshot.pending_retirements, ResourceEvidence::Unobserved);
    assert_eq!(snapshot.retired_operations, ResourceEvidence::Unobserved);
    assert_eq!(snapshot.fence, ResourceEvidence::Unobserved);
    assert_eq!(snapshot.closure, ResourceEvidence::Unobserved);
    assert_eq!(ledger.snapshot(), snapshot);
}

#[test]
fn successful_zero_observation_neither_fences_nor_closes() {
    let mut ledger = ResourceObservabilityLedger::new(scope(1, 1));
    let mut sample = ledger.snapshot();
    sample.resident_bytes = ResourceEvidence::Observed(0);
    sample.pending_operations = ResourceEvidence::Observed(0);
    sample.scratch_bytes = ResourceEvidence::Observed(0);
    sample.active_credit_grants = ResourceEvidence::Observed(0);
    sample.pending_retirements = ResourceEvidence::Observed(0);
    sample.retired_operations = ResourceEvidence::Observed(8);
    sample.quarantine = ResourceEvidence::Observed(false);
    ledger.observe(&sample).unwrap();
    assert_eq!(ledger.snapshot(), sample);
    assert_eq!(ledger.snapshot().fence, ResourceEvidence::Unobserved);
    assert_eq!(ledger.snapshot().closure, ResourceEvidence::Unobserved);
}

#[test]
fn quiescence_status_is_not_a_verified_fence_or_owner_closure() {
    let mut ledger = ResourceObservabilityLedger::new(scope(1, 1));
    ledger.record_quiescence(QuiescenceEvidence::Pending);
    assert_eq!(
        ledger.snapshot().fence,
        ResourceEvidence::Observed(QuiescenceEvidence::Pending)
    );
    ledger.record_quiescence(QuiescenceEvidence::Quiescent);
    assert_eq!(
        ledger.snapshot().fence,
        ResourceEvidence::Observed(QuiescenceEvidence::Quiescent)
    );
    assert_eq!(ledger.snapshot().closure, ResourceEvidence::Unobserved);
    let mut authenticated = ledger.snapshot();
    authenticated.fence = ResourceEvidence::Verified(QuiescenceEvidence::Quiescent);
    ledger.observe(&authenticated).unwrap();
    assert_eq!(ledger.snapshot().fence, authenticated.fence);
    assert_eq!(ledger.snapshot().closure, ResourceEvidence::Unobserved);
    ledger.record_closed();
    assert_eq!(
        ledger.snapshot().closure,
        ResourceEvidence::Observed(ResourceClosure::Closed)
    );
}

#[test]
fn unknown_is_sticky_after_successful_observation_fence_and_close() {
    let mut ledger = ResourceObservabilityLedger::new(scope(1, 1));
    ledger.record_quiescence(QuiescenceEvidence::Unknown);
    let mut sample = ResourceObservationSnapshot::unobserved(scope(1, 1));
    sample.resident_bytes = ResourceEvidence::Observed(64);
    sample.pending_operations = ResourceEvidence::Observed(0);
    sample.quarantine = ResourceEvidence::Observed(false);
    sample.fence = ResourceEvidence::Verified(QuiescenceEvidence::Quiescent);
    sample.closure = ResourceEvidence::Verified(ResourceClosure::Closed);
    ledger.observe(&sample).unwrap();
    ledger.record_quiescence(QuiescenceEvidence::Quiescent);
    ledger.record_closed();
    let snapshot = ledger.snapshot();
    assert_eq!(snapshot.resident_bytes, ResourceEvidence::Observed(64));
    assert_eq!(snapshot.pending_operations, ResourceEvidence::Observed(0));
    assert_eq!(snapshot.quarantine, ResourceEvidence::Unknown);
    assert_eq!(snapshot.fence, ResourceEvidence::Unknown);
    assert_eq!(snapshot.closure, ResourceEvidence::Unknown);
    assert_eq!(ledger.snapshot(), snapshot);
}

#[test]
fn unknown_measurement_is_not_replaced_by_a_later_number() {
    let mut ledger = ResourceObservabilityLedger::new(scope(1, 1));
    let mut sample = ledger.snapshot();
    sample.scratch_bytes = ResourceEvidence::Unknown;
    ledger.observe(&sample).unwrap();
    sample.scratch_bytes = ResourceEvidence::Verified(0);
    ledger.observe(&sample).unwrap();
    assert_eq!(ledger.snapshot().scratch_bytes, ResourceEvidence::Unknown);
}

#[test]
fn protocol_unknown_cannot_hide_inside_an_observed_or_verified_fence() {
    for fence in [
        ResourceEvidence::Observed(QuiescenceEvidence::Unknown),
        ResourceEvidence::Verified(QuiescenceEvidence::Unknown),
        ResourceEvidence::Unknown,
    ] {
        let mut ledger = ResourceObservabilityLedger::new(scope(1, 1));
        let mut sample = ledger.snapshot();
        sample.fence = fence;
        sample.closure = ResourceEvidence::Verified(ResourceClosure::Closed);
        ledger.observe(&sample).unwrap();
        assert_eq!(ledger.snapshot().fence, ResourceEvidence::Unknown);
        assert_eq!(ledger.snapshot().closure, ResourceEvidence::Unknown);
        assert_eq!(ledger.snapshot().quarantine, ResourceEvidence::Unknown);
    }
}

#[test]
fn other_owner_or_generation_cannot_overwrite_or_clear_unknown() {
    let mut ledger = ResourceObservabilityLedger::new(scope(1, 1));
    ledger.record_quiescence(QuiescenceEvidence::Unknown);
    let before = ledger.snapshot();
    for received in [scope(2, 1), scope(1, 2)] {
        let mut sample = ResourceObservationSnapshot::unobserved(received);
        sample.closure = ResourceEvidence::Verified(ResourceClosure::Closed);
        let error = ledger.observe(&sample).unwrap_err();
        assert_eq!(error.expected, scope(1, 1));
        assert_eq!(error.received, received);
        assert_eq!(ledger.snapshot(), before);
    }
    let replacement = ResourceObservabilityLedger::new(scope(1, 2));
    assert_eq!(replacement.snapshot().closure, ResourceEvidence::Unobserved);
    assert_eq!(ledger.snapshot(), before);
}

#[test]
fn live_projection_preserves_unknown_without_mutating_the_ledger() {
    let mut ledger = ResourceObservabilityLedger::new(scope(1, 1));
    ledger.record_quiescence(QuiescenceEvidence::Unknown);
    let before = ledger.snapshot();
    let mut live = ResourceObservationSnapshot::unobserved(scope(1, 1));
    live.resident_bytes = ResourceEvidence::Observed(128);
    live.fence = ResourceEvidence::Verified(QuiescenceEvidence::Quiescent);
    live.closure = ResourceEvidence::Verified(ResourceClosure::Closed);
    let projected = ledger.snapshot_with_live(&live).unwrap();
    assert_eq!(projected.resident_bytes, ResourceEvidence::Observed(128));
    assert_eq!(projected.fence, ResourceEvidence::Unknown);
    assert_eq!(projected.closure, ResourceEvidence::Unknown);
    assert_eq!(projected.quarantine, ResourceEvidence::Unknown);
    assert_eq!(ledger.snapshot(), before);
    live.scope = scope(1, 2);
    assert!(ledger.snapshot_with_live(&live).is_err());
    assert_eq!(ledger.snapshot(), before);
}

#[test]
fn live_unknown_is_normalized_without_manufacturing_quarantine_or_persisting_a_read() {
    let ledger = ResourceObservabilityLedger::new(scope(1, 1));
    let before = ledger.snapshot();
    let mut live = before.clone();
    live.fence = ResourceEvidence::Observed(QuiescenceEvidence::Unknown);
    live.closure = ResourceEvidence::Verified(ResourceClosure::Closed);
    let projected = ledger.snapshot_with_live(&live).unwrap();
    assert_eq!(projected.fence, ResourceEvidence::Unknown);
    assert_eq!(projected.closure, ResourceEvidence::Unknown);
    assert_eq!(projected.quarantine, ResourceEvidence::Unknown);
    assert_eq!(ledger.snapshot(), before);
    live.quarantine = ResourceEvidence::Observed(true);
    assert_eq!(
        ledger.snapshot_with_live(&live).unwrap().quarantine,
        ResourceEvidence::Observed(true),
    );
    assert_eq!(ledger.snapshot(), before);
}

#[test]
fn omitted_measurements_do_not_erase_recorded_evidence() {
    let mut ledger = ResourceObservabilityLedger::new(scope(1, 1));
    let mut sample = ledger.snapshot();
    sample.resident_bytes = ResourceEvidence::Observed(4096);
    ledger.observe(&sample).unwrap();
    ledger
        .observe(&ResourceObservationSnapshot::unobserved(scope(1, 1)))
        .unwrap();
    assert_eq!(
        ledger.snapshot().resident_bytes,
        ResourceEvidence::Observed(4096)
    );
    assert_eq!(ledger.snapshot().closure, ResourceEvidence::Unobserved);
}
