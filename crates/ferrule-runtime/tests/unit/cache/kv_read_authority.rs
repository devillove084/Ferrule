#![expect(
    unsafe_code,
    reason = "Driver-free no-work physical attestations and adversarial routing tests; no device references exist"
)]
// SAFETY of the attestations in this private driver-free fixture: the modeled
// physical stages are synchronous/no-work; there are no device references or
// callbacks. Mismatched and replayed observations test rejection before mutation.
// Included as a private unit-test module; not a standalone integration target.
use super::*;
#[path = "kv_read_authority/append_contract.rs"]
mod append_contract;
use ferrule_common::execution::{KvElementType, KvLayoutSchemaIdentity, KvPlaneDescriptor};
use ferrule_common::host_kv::placement::{HostKvPlacementOwner, PlacementBudget, PlacementCredits};
use ferrule_common::host_kv::{HostKvCompletion, HostKvFence, HostKvStage};
#[derive(Debug)]
struct Schema;
impl KvLayoutSchema for Schema {
    fn planes(&self) -> &[KvPlaneDescriptor] {
        static PLANES: [KvPlaneDescriptor; 1] =
            [KvPlaneDescriptor::new("kv", 2, 1, KvElementType::F32)];
        &PLANES
    }
    fn page_size(&self) -> usize {
        4
    }
    fn max_sequence_len(&self) -> usize {
        128
    }
}
fn slot(n: u32) -> StateSlot {
    StateSlot::new(n)
}
fn append(m: &mut KvPageManager, s: u32, n: usize) {
    let r = m.reserve(slot(s), 7, n).unwrap();
    let p = m
        .prepare_commit(vec![KvReservationCommit::new(r, n)])
        .unwrap();
    let retired = m.publish_commit(p);
    m.confirm_page_retirement(retired).unwrap();
}
fn manager() -> KvPageManager {
    let mut m = KvPageManager::new(Box::new(Schema), 16);
    m.alloc_sequence(slot(0), 7).unwrap();
    append(&mut m, 0, 3);
    m
}
fn source(m: &KvPageManager) -> CommittedKvSource {
    m.committed_kv_source(slot(0), 7, 0..3).unwrap()
}
fn credits() -> PlacementCredits {
    PlacementCredits {
        pageable_host_bytes: 32,
        pinned_host_bytes: 32,
        staging_bytes: 32,
        staging_slots: 1,
        transfers: 1,
        consumers: 1,
    }
}
#[test]
fn reader_blocks_writes_sharing_preemption_and_retirement() {
    let mut m = manager();
    let s = source(&m);
    let prepared = m
        .prepare_fork_sequence_exact(slot(0), slot(1), 7, 3)
        .unwrap();
    let l = m.acquire_committed_read(&s).unwrap();
    assert!(m.required_physical_pages(slot(0), 7, 1).is_err());
    assert!(m.reserve(slot(0), 7, 1).is_err());
    assert!(m.capture_prefix_snapshot(slot(0), 3).is_err());
    assert!(
        m.prepare_fork_sequence_exact(slot(0), slot(1), 7, 3)
            .is_err()
    );
    assert!(m.publish_fork_sequence_exact(prepared).is_err());
    assert!(m.preempt_sequence(slot(0)).is_err());
    assert!(m.free_sequence_pages(slot(0)).is_err());
    m.validate_committed_read(&l).unwrap();
    m.release_committed_read(&l).unwrap();
    assert_eq!(m.release_committed_read(&l), Err(HostKvError::UnknownLease));
    append(&mut m, 0, 1);
}
#[test]
fn multiple_readers_release_independently_and_drop_is_not_ack() {
    let mut m = manager();
    let s = source(&m);
    let a = m.acquire_committed_read(&s).unwrap();
    let b = m.acquire_committed_read(&s).unwrap();
    m.release_committed_read(&a).unwrap();
    assert!(m.reserve(slot(0), 7, 1).is_err());
    drop(b);
    assert!(m.free_sequence_pages(slot(0)).is_err());
    assert!(m.ensure_no_slot_readers(slot(0)).is_err());
}
#[test]
fn source_identity_checks_are_exact_and_failure_atomic() {
    let mut m = manager();
    let original = source(&m);
    for field in 0..7 {
        let mut s = original.clone();
        match field {
            0 => s.owner += 1,
            1 => s.generation += 1,
            2 => s.incarnation += 1,
            3 => s.pages[0] = KvPageId(9),
            4 => s.positions = 0..4,
            5 => s.committed_tokens = 4,
            _ => {
                s.schema = KvLayoutSchemaIdentity::new(
                    vec![KvPlaneDescriptor::new("kv", 2, 1, KvElementType::Bf16)],
                    4,
                    128,
                )
            }
        }
        assert!(m.acquire_committed_read(&s).is_err());
        m.ensure_no_slot_readers(slot(0)).unwrap();
    }
    assert!(m.committed_kv_source(slot(0), 7, 0..0).is_err());
    let l = m.acquire_committed_read(&original).unwrap();
    let mut foreign = manager();
    assert_eq!(
        foreign.release_committed_read(&l),
        Err(HostKvError::ForeignOwner)
    );
    m.release_committed_read(&l).unwrap();
}
#[test]
fn active_transaction_shared_prefix_and_cow_reject_reads() {
    let mut m = manager();
    let s = source(&m);
    let r = m.reserve(slot(0), 7, 1).unwrap();
    assert_eq!(
        m.acquire_committed_read(&s).unwrap_err(),
        HostKvError::TransactionActive
    );
    let p = m
        .prepare_commit(vec![KvReservationCommit::new(r, 0)])
        .unwrap();
    assert_eq!(
        m.acquire_committed_read(&s).unwrap_err(),
        HostKvError::TransactionActive
    );
    let retired = m.abort_prepared_commit(p);
    m.confirm_page_retirement(retired).unwrap();
    let snapshot = m.capture_prefix_snapshot(slot(0), 3).unwrap();
    assert_eq!(
        m.acquire_committed_read(&s).unwrap_err(),
        HostKvError::SharedSource
    );
    let r = m.reserve(slot(0), 7, 1).unwrap();
    assert!(r.cow_replacement.is_some());
    assert_eq!(
        m.acquire_committed_read(&s).unwrap_err(),
        HostKvError::TransactionActive
    );
    let retired = m.abort_reservations(vec![r]).unwrap();
    m.confirm_page_retirement(retired).unwrap();
    let retired = m.release_prefix_snapshot(snapshot).unwrap();
    m.confirm_page_retirement(retired).unwrap();
    let l = m.acquire_committed_read(&s).unwrap();
    m.release_committed_read(&l).unwrap();
}
#[test]
fn cloned_suspension_cannot_restore_two_alias_frontiers() {
    let mut m = manager();
    let old = source(&m);
    let state = m.preempt_sequence(slot(0)).unwrap();
    let clone = state.clone();
    m.restore_sequence(slot(0), state).unwrap();
    assert!(m.restore_sequence(slot(1), clone.clone()).is_err());
    assert!(m.release_preempted_pages(clone).is_err());
    assert_eq!(m.page_refcount(old.pages[0]), 1);
    assert_eq!(
        m.acquire_committed_read(&old).unwrap_err(),
        HostKvError::StaleSource
    );
    append(&mut m, 0, 1);
    let s = m.committed_kv_source(slot(0), 7, 3..4).unwrap();
    let l = m.acquire_committed_read(&s).unwrap();
    assert!(m.reserve(slot(1), 7, 1).is_err());
    assert!(m.reserve(slot(0), 7, 1).is_err());
    m.release_committed_read(&l).unwrap();
}
#[test]
fn foreign_suspension_and_resuspension_aba_are_rejected() {
    let mut a = manager();
    let mut b = manager();
    let state = a.preempt_sequence(slot(0)).unwrap();
    assert_eq!(state.pages(), b.block_table(slot(0)).unwrap().pages());
    assert!(b.restore_sequence(slot(1), state.clone()).is_err());
    let (_, returned) = b
        .release_preempted_pages(state.clone())
        .unwrap_err()
        .into_parts();
    a.restore_sequence(slot(0), returned).unwrap();
    let new = a.preempt_sequence(slot(0)).unwrap();
    assert!(a.restore_sequence(slot(0), state).is_err());
    a.restore_sequence(slot(0), new).unwrap();
}
#[test]
fn released_suspension_cannot_touch_quarantine_or_recycled_page() {
    let mut m = manager();
    let old_source = source(&m);
    let state = m.preempt_sequence(slot(0)).unwrap();
    let retired = m.release_preempted_pages(state.clone()).unwrap();
    assert_eq!(m.free_pages(), 0);
    assert!(m.restore_sequence(slot(1), state.clone()).is_err());
    assert!(m.release_preempted_pages(state.clone()).is_err());
    m.confirm_page_retirement(retired).unwrap();
    m.alloc_sequence(slot(0), 7).unwrap();
    append(&mut m, 0, 3);
    assert_eq!(old_source.pages, source(&m).pages);
    assert!(m.acquire_committed_read(&old_source).is_err());
    assert!(m.restore_sequence(slot(1), state.clone()).is_err());
    assert!(m.release_preempted_pages(state).is_err());
    assert_eq!(m.page_refcount(old_source.pages[0]), 1);
}
#[test]
fn restore_target_and_reader_failure_leave_handle_retryable() {
    let mut m = manager();
    m.alloc_sequence(slot(1), 7).unwrap();
    append(&mut m, 1, 3);
    let state = m.preempt_sequence(slot(1)).unwrap();
    let l = m.acquire_committed_read(&source(&m)).unwrap();
    assert!(m.restore_sequence(slot(0), state.clone()).is_err());
    m.release_committed_read(&l).unwrap();
    assert!(m.restore_sequence(slot(0), state.clone()).is_err());
    m.restore_sequence(slot(1), state.clone()).unwrap();
    assert!(m.restore_sequence(slot(2), state).is_err());
}
#[test]
fn failed_release_preflight_does_not_partially_decrement_and_returns_handle() {
    let mut m = manager();
    append(&mut m, 0, 2);
    let state = m.preempt_sequence(slot(0)).unwrap();
    let pages = state.pages().to_vec();
    m.page_refcounts.remove(&pages[1]);
    let (_, state) = m.release_preempted_pages(state).unwrap_err().into_parts();
    assert_eq!(m.page_refcount(pages[0]), 1);
    assert_eq!(m.suspended.len(), 1);
    m.page_refcounts.insert(pages[1], 1);
    let retired = m.release_preempted_pages(state).unwrap();
    assert_eq!(retired.pages().len(), 2);
    assert_eq!(m.free_pages(), 0);
    m.confirm_page_retirement(retired).unwrap();
    assert_eq!(m.free_pages(), 2);
}
#[test]
fn reader_guard_checks_physical_aliases_and_execution_bindings() {
    let mut m = manager();
    let s = source(&m);
    let l = m.acquire_committed_read(&s).unwrap();
    // Fault injection: an alias with an older frontier and an incorrect refcount.
    m.sequences.insert(
        1,
        SequencePageState {
            incarnation: 999,
            generation: 7,
            block_table: BlockTable {
                pages: s.pages.clone(),
                committed_tokens: 2,
            },
        },
    );
    assert!(m.reserve(slot(1), 7, 1).is_err());
    assert!(m.free_sequence_pages(slot(1)).is_err());
    m.sequences.remove(&1);
    m.alloc_sequence(slot(1), 7).unwrap();
    let mut r = m.reserve(slot(1), 7, 1).unwrap();
    assert!(m.bind_reservation_execution(&mut r, slot(0), 7).is_err());
    m.release_committed_read(&l).unwrap();
    m.bind_reservation_execution(&mut r, slot(0), 7).unwrap();
    assert_eq!(
        m.acquire_committed_read(&s).unwrap_err(),
        HostKvError::TransactionActive
    );
    let retired = m.abort_reservations(vec![r]).unwrap();
    m.confirm_page_retirement(retired).unwrap();
}
#[test]
fn placement_release_preserves_logical_pages_and_unknown_drop_keeps_pin() {
    let mut m = manager();
    let s = source(&m);
    let before = m.stats();
    let budget = PlacementBudget::new(credits());
    let mut o = HostKvPlacementOwner::new(budget.clone()).unwrap();
    let h = o.reserve(&mut m, &s, credits()).unwrap();
    for stage in [
        HostKvStage::HostCopy,
        HostKvStage::Upload,
        HostKvStage::Consumer,
    ] {
        let fence = HostKvFence {
            owner: h.owner(),
            operation: h.operation(),
            generation: 1,
            stage,
            stream: 0,
            event: 1,
        };
        o.submit(&m, h, fence).unwrap();
        assert!(o.release(&mut m, h).is_err());
        assert_eq!(budget.in_use(), credits());
        unsafe { o.observe_complete(h, HostKvCompletion { fence }) }.unwrap();
    }
    o.release(&mut m, h).unwrap();
    assert_eq!(m.block_table(slot(0)).unwrap().pages(), &s.pages);
    assert_eq!(m.page_refcount(s.pages[0]), 1);
    assert_eq!(m.stats().free_pages, before.free_pages);
    assert_eq!(m.stats().retiring_pages, before.retiring_pages);
    assert_eq!(budget.in_use(), PlacementCredits::default());
    let h = o.reserve(&mut m, &s, credits()).unwrap();
    o.submit(
        &m,
        h,
        HostKvFence {
            owner: h.owner(),
            operation: h.operation(),
            generation: 2,
            stage: HostKvStage::HostCopy,
            stream: 0,
            event: 2,
        },
    )
    .unwrap();
    o.mark_unknown(h).unwrap();
    assert_eq!(
        unsafe {
            o.observe_complete(
                h,
                HostKvCompletion {
                    fence: HostKvFence {
                        owner: h.owner(),
                        operation: h.operation(),
                        generation: 2,
                        stage: HostKvStage::HostCopy,
                        stream: 0,
                        event: 2,
                    },
                },
            )
        },
        Err(HostKvError::Unknown)
    );
    assert!(o.release(&mut m, h).is_err());
    drop(o);
    assert_eq!(budget.in_use(), credits());
    assert!(m.preempt_sequence(slot(0)).is_err());
}
#[test]
fn incarnation_exhaustion_fails_without_consuming_custody() {
    let mut m = manager();
    let s = source(&m);
    let state = m.preempt_sequence(slot(0)).unwrap();
    m.next_incarnation = u64::MAX;
    assert!(m.restore_sequence(slot(0), state.clone()).is_err());
    assert_eq!(m.suspended.len(), 1);
    assert_eq!(m.page_refcount(s.pages[0]), 1);
    let retired = m.release_preempted_pages(state).unwrap();
    m.confirm_page_retirement(retired).unwrap();
}
#[test]
fn prepared_fork_rejects_foreign_owner_and_recycled_source() {
    let mut m = manager();
    let mut foreign = manager();
    let p = m
        .prepare_fork_sequence_exact(slot(0), slot(1), 7, 3)
        .unwrap();
    assert!(foreign.publish_fork_sequence_exact(p).is_err());
    let p = m
        .prepare_fork_sequence_exact(slot(0), slot(1), 7, 3)
        .unwrap();
    let state = m.preempt_sequence(slot(0)).unwrap();
    m.restore_sequence(slot(0), state).unwrap();
    assert!(m.publish_fork_sequence_exact(p).is_err());
    assert_eq!(m.page_refcount(source(&m).pages[0]), 1);
}

#[test]
fn source_authority_failure_does_not_consume_reader_registration() {
    let mut m = manager();
    let s = source(&m);
    let lease = m.acquire_committed_read(&s).unwrap();
    // The index alone still matches. Only the manager can reject a changed source.
    m.sequences.get_mut(&0).unwrap().generation += 1;
    assert_eq!(
        m.validate_committed_read(&lease),
        Err(HostKvError::StaleSource)
    );
    assert_eq!(
        m.release_committed_read(&lease),
        Err(HostKvError::StaleSource)
    );
    assert!(m.ensure_no_page_readers(&s.pages).is_err());
    m.sequences.get_mut(&0).unwrap().generation = s.generation;
    m.validate_committed_read(&lease).unwrap();
    m.release_committed_read(&lease).unwrap();
    m.ensure_no_slot_readers(slot(0)).unwrap();
}

#[test]
fn released_lease_cannot_ack_recycled_slot_and_page_with_same_generation() {
    let mut m = manager();
    let old_source = source(&m);
    let old = m.acquire_committed_read(&old_source).unwrap();
    m.release_committed_read(&old).unwrap();
    let retired = m.free_sequence_pages(slot(0)).unwrap();
    m.confirm_page_retirement(retired).unwrap();
    m.alloc_sequence(slot(0), 7).unwrap();
    append(&mut m, 0, 3);
    let current_source = source(&m);
    assert_eq!(old_source.pages, current_source.pages);
    assert_eq!(old_source.state_slot, current_source.state_slot);
    assert_eq!(old_source.generation, current_source.generation);
    assert_ne!(old_source.incarnation, current_source.incarnation);
    assert_eq!(
        m.acquire_committed_read(&old_source).unwrap_err(),
        HostKvError::StaleSource
    );
    let current = m.acquire_committed_read(&current_source).unwrap();
    for _ in 0..2 {
        assert_eq!(
            m.release_committed_read(&old),
            Err(HostKvError::UnknownLease)
        );
        m.validate_committed_read(&current).unwrap();
        assert!(m.free_sequence_pages(slot(0)).is_err());
        assert!(m.reserve(slot(0), 7, 1).is_err());
    }
    m.release_committed_read(&current).unwrap();
    append(&mut m, 0, 1);
}
