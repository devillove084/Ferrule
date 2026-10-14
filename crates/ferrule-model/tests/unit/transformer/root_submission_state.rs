use super::*;
use std::cell::Cell;
use std::rc::Rc;

#[derive(Debug)]
struct Root(Rc<Cell<usize>>);
impl Drop for Root {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}
fn pin(segment: &mut SubmissionSegment<Root>, key: u64, bytes: usize, drops: &Rc<Cell<usize>>) {
    segment.pin(key, bytes, Root(Rc::clone(drops))).unwrap();
}

#[test]
fn same_stream_proofs_defer_only_inside_a_pinned_operation() {
    let drops = Rc::new(Cell::new(0));
    let mut segment = SubmissionSegment::new(128).unwrap();
    assert!(!segment.can_defer_ordered_proof());
    for op in 0..4 {
        assert!(segment.can_fit(32, 1));
        assert!(segment.begin_operation());
        pin(&mut segment, op, 32, &drops);
        for _ in 0..3 {
            assert!(segment.can_defer_ordered_proof());
        }
        assert!(!segment.retire_after_proof(BarrierReason::WorkspaceGrowth));
        assert_eq!(
            drops.get(),
            0,
            "mid-operation fence cannot release future consumers' inputs"
        );
        assert!(segment.finish_operation());
        assert!(!segment.can_defer_ordered_proof());
    }
    assert_eq!(segment.bytes(), 128);
    assert_eq!(segment.root_count(), 4);
    assert!(segment.retire_after_proof(BarrierReason::SegmentExit));
    assert_eq!(drops.get(), 4);
    assert_eq!(segment.bytes(), 0);
    assert!(!segment.has_pending());
}

#[test]
fn byte_root_and_operation_limits_are_independent_and_finite() {
    let drops = Rc::new(Cell::new(0));
    assert!(SubmissionSegment::<Root>::new(0).is_none());
    let mut segment = SubmissionSegment::new(31).unwrap();
    assert!(!segment.can_fit(32, 1));
    assert!(!segment.can_fit(usize::MAX, 1));
    assert!(!segment.can_fit(1, usize::MAX));
    assert!(segment.begin_operation());
    pin(&mut segment, 0, 31, &drops);
    assert!(segment.pin(1, 1, Root(Rc::clone(&drops))).is_err());
    assert!(segment.finish_operation());
    assert_eq!(segment.bytes(), segment.limit());
    assert!(!segment.can_fit(1, 1));
    assert!(segment.retire_after_proof(BarrierReason::PinBudget));
    assert!(segment.can_fit(31, 1));
    let mut segment = SubmissionSegment::new(usize::MAX).unwrap();
    for key in 0..MAX_ROOTS {
        assert!(segment.begin_operation());
        pin(&mut segment, key as u64, 1, &drops);
        assert!(segment.finish_operation());
    }
    assert_eq!(segment.root_count(), MAX_ROOTS);
    assert!(!segment.can_fit(1, 1));
    assert!(segment.retire_after_proof(BarrierReason::PinBudget));
    for _ in 0..MAX_OPERATIONS {
        assert!(segment.begin_operation());
        pin(&mut segment, 0, 1, &drops);
        assert!(segment.finish_operation());
    }
    assert_eq!(
        segment.root_count(),
        1,
        "dedup is not an unlimited operation allowance"
    );
    assert!(!segment.begin_operation());
}

#[test]
fn unknown_is_sticky_and_cannot_release_roots_or_credit() {
    let drops = Rc::new(Cell::new(0));
    let mut segment = SubmissionSegment::new(64).unwrap();
    assert!(segment.begin_operation());
    pin(&mut segment, 1, 64, &drops);
    segment.quarantine();
    for _ in 0..3 {
        assert!(!segment.retire_after_proof(BarrierReason::SegmentExit));
        assert!(!segment.can_defer_ordered_proof());
        assert!(!segment.finish_operation());
        assert!(!segment.begin_operation());
    }
    assert!(segment.is_unknown());
    assert_eq!(segment.bytes(), 64);
    assert_eq!(segment.root_count(), 1);
    assert_eq!(drops.get(), 0);
}

#[test]
fn host_ownership_and_storage_barriers_need_actual_proof() {
    for reason in [
        BarrierReason::HostRead,
        BarrierReason::CpuRouter,
        BarrierReason::ExternalExperts,
        BarrierReason::TensorCollective,
        BarrierReason::KvPublish,
        BarrierReason::RecurrentState,
        BarrierReason::WorkspaceGrowth,
        BarrierReason::CacheEvictionOrRebind,
        BarrierReason::Diagnostic,
        BarrierReason::PinBudget,
        BarrierReason::SegmentExit,
        BarrierReason::OpaqueOperator,
    ] {
        let drops = Rc::new(Cell::new(0));
        let mut segment = SubmissionSegment::new(32).unwrap();
        assert!(segment.begin_operation());
        pin(&mut segment, 0, 32, &drops);
        assert!(segment.finish_operation());
        assert!(segment.has_pending());
        assert_eq!(drops.get(), 0);
        assert!(segment.retire_after_proof(reason));
        assert_eq!(segment.last_barrier, Some(reason));
        assert_eq!(drops.get(), 1);
    }
}

#[test]
fn allocation_or_upload_evidence_is_not_consumer_completion() {
    let drops = Rc::new(Cell::new(0));
    let mut segment = SubmissionSegment::new(64).unwrap();
    assert!(segment.begin_operation());
    pin(&mut segment, 0, 64, &drops);
    // Intentionally no allocation/upload-complete transition in this ledger.
    assert!(!segment.begin_operation());
    assert_eq!(segment.bytes(), 64);
    assert_eq!(drops.get(), 0);
    segment.quarantine();
    assert!(!segment.retire_after_proof(BarrierReason::WorkspaceGrowth));
    assert_eq!(drops.get(), 0);
}

#[test]
fn external_error_requires_a_clean_producer_boundary_not_remote_completion() {
    let drops = Rc::new(Cell::new(0));
    let mut segment = SubmissionSegment::new(64).unwrap();
    assert!(!segment.can_return_external_error());
    assert!(segment.begin_operation());
    pin(&mut segment, 1, 64, &drops);
    assert!(!segment.note_idle_barrier(BarrierReason::ExternalExperts));
    assert!(!segment.can_return_external_error());
    assert!(segment.finish_operation());
    assert!(!segment.note_idle_barrier(BarrierReason::ExternalExperts));
    assert!(!segment.can_return_external_error());
    assert_eq!(drops.get(), 0);
    assert!(segment.retire_after_proof(BarrierReason::ExternalExperts));
    assert!(segment.can_return_external_error());
    assert_eq!(drops.get(), 1);
    assert!(segment.note_idle_barrier(BarrierReason::OpaqueOperator));
    assert!(!segment.can_return_external_error());
    assert_eq!(
        segment.last_barrier,
        Some(BarrierReason::ExternalExperts),
        "idle boundaries must not rewrite the last actual proof"
    );
    // A shared gate/router may have fenced everything before the external hook.
    assert!(segment.note_idle_barrier(BarrierReason::ExternalExperts));
    assert!(segment.can_return_external_error());
    assert!(segment.begin_operation());
    assert!(!segment.can_return_external_error());
    pin(&mut segment, 2, 64, &drops);
    assert!(segment.finish_operation());
    assert!(!segment.can_return_external_error());
    segment.quarantine();
    assert!(!segment.note_idle_barrier(BarrierReason::ExternalExperts));
    assert!(!segment.retire_after_proof(BarrierReason::ExternalExperts));
    assert!(!segment.can_return_external_error());
    assert_eq!(segment.root_count(), 1);
    assert_eq!(drops.get(), 1);
}

#[test]
fn external_error_permission_does_not_survive_another_idle_boundary_or_unknown() {
    for reason in [
        BarrierReason::HostRead,
        BarrierReason::CpuRouter,
        BarrierReason::OpaqueOperator,
        BarrierReason::SegmentExit,
    ] {
        let mut segment = SubmissionSegment::<Root>::new(64).unwrap();
        assert!(segment.note_idle_barrier(BarrierReason::ExternalExperts));
        assert!(segment.can_return_external_error());
        assert!(segment.note_idle_barrier(reason));
        assert!(!segment.can_return_external_error());
        assert!(segment.note_idle_barrier(BarrierReason::ExternalExperts));
        segment.quarantine();
        assert!(!segment.can_return_external_error());
        assert!(!segment.note_idle_barrier(BarrierReason::ExternalExperts));
    }
}
