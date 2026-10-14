//! The production manager cannot turn an isolated read into append custody.
use super::*;
use ferrule_common::execution::ExecutionTransactionId;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[test]
fn either_admission_order_rejects_overlap_without_changing_frontier_or_budget() {
    for read_first in [false, true] {
        let mut m = manager();
        // Cover both partial-tail reuse and a page-aligned append. Restricting
        // DMA to an earlier subrange does not exempt the append transaction.
        for committed in [3, 4] {
            if committed == 4 {
                append(&mut m, 0, 1);
            }
            let s = m.committed_kv_source(slot(0), 7, 0..1).unwrap();
            let budget = PlacementBudget::new(credits());
            let mut owner = HostKvPlacementOwner::new(budget.clone()).unwrap();
            if read_first {
                let h = owner.reserve(&mut m, &s, credits()).unwrap();
                assert!(m.reserve(slot(0), 7, 1).is_err());
                owner.validate(&m, h).unwrap();
                assert_eq!(budget.in_use(), credits());
                owner.cancel(&mut m, h).unwrap();
            } else {
                let append = m.reserve(slot(0), 7, 1).unwrap();
                assert_eq!(
                    owner.reserve(&mut m, &s, credits()),
                    Err(HostKvError::TransactionActive)
                );
                assert_eq!(budget.in_use(), PlacementCredits::default());
                m.ensure_no_slot_readers(slot(0)).unwrap();
                let retired = m.abort_reservations(vec![append]).unwrap();
                m.confirm_page_retirement(retired).unwrap();
            }
            assert_eq!(
                m.block_table(slot(0)).unwrap().committed_tokens(),
                committed
            );
            assert_eq!(budget.in_use(), PlacementCredits::default());
            m.validate_committed_source(&s).unwrap();
        }
    }
}

#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<BTreeMap<String, String>>>>);
impl tracing::Subscriber for Logs {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target() == "ferrule::host_kv"
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Fields(BTreeMap<String, String>);
        impl tracing::field::Visit for Fields {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.0.insert(field.name().into(), format!("{value:?}"));
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                self.0.insert(field.name().into(), value.into());
            }
        }
        let mut fields = Fields(Default::default());
        event.record(&mut fields);
        self.0.lock().unwrap().push(fields.0);
    }
}

#[test]
fn append_capability_is_borrowed_from_reservation_and_cow_projects_new_tail() {
    let mut m = manager();
    let snapshot = m.capture_prefix_snapshot(slot(0), 3).unwrap();
    let reservation = m.reserve(slot(0), 7, 1).unwrap();
    assert!(reservation.cow_replacement.is_some());
    let capability = m
        .derive_append_prefix(&reservation, ExecutionTransactionId::new(19).unwrap(), 0)
        .unwrap();
    capability.validate().unwrap();
    assert_eq!(capability.witness().positions(), 0..4);
    assert_eq!(capability.witness().pages().len(), 1);
    assert_eq!(
        capability.witness().pages()[0],
        reservation.cow_replacement.unwrap().replacement
    );
    drop(capability);
    let retirement = m.abort_reservations(vec![reservation]).unwrap();
    m.confirm_page_retirement(retirement).unwrap();
    let retirement = m.release_prefix_snapshot(snapshot).unwrap();
    m.confirm_page_retirement(retirement).unwrap();
    m.validate_committed_source(&source(&m)).unwrap();
}

#[test]
fn reservation_identity_is_manager_scoped_and_abort_cannot_consume_foreign_token() {
    let mut left = manager();
    let mut right = manager();
    let left_reservation = left.reserve(slot(0), 7, 1).unwrap();
    let right_reservation = right.reserve(slot(0), 7, 1).unwrap();
    assert_eq!(left_reservation.id(), right_reservation.id());
    assert_eq!(left_reservation.view(), right_reservation.view());
    assert!(right.reservation_view(&left_reservation).is_err());
    assert!(
        right
            .derive_append_prefix(
                &left_reservation,
                ExecutionTransactionId::new(19).unwrap(),
                0
            )
            .is_err()
    );
    assert!(left.reservation_view(&right_reservation).is_err());
    let (_, returned) = right
        .abort_reservations(vec![left_reservation])
        .unwrap_err()
        .into_parts();
    assert_eq!(returned.len(), 1);
    let retired = left.abort_reservations(returned).unwrap();
    left.confirm_page_retirement(retired).unwrap();
    let retired = right.abort_reservations(vec![right_reservation]).unwrap();
    right.confirm_page_retirement(retired).unwrap();
}

#[test]
fn stale_reservation_cannot_reacquire_prefix_after_abort_and_aba() {
    let mut m = manager();
    let r = m.reserve(slot(0), 7, 1).unwrap();
    // Private test-only replay token; public reservations are not Clone.
    let stale = KvReservation {
        manager_id: m.manager_id,
        id: r.id,
        view: r.view.clone(),
    };
    let cap = m
        .derive_append_prefix(&r, ExecutionTransactionId::new(19).unwrap(), 0)
        .unwrap();
    cap.validate().unwrap();
    m.ensure_no_slot_readers(slot(0)).unwrap(); // no second logical pin authority
    drop(cap);
    let retired = m.abort_reservations(vec![r]).unwrap();
    m.confirm_page_retirement(retired).unwrap();
    let r = m.reserve(slot(0), 7, 1).unwrap();
    assert_eq!(r.view(), stale.view());
    assert_ne!(r.id(), stale.id());
    assert!(
        m.derive_append_prefix(&stale, ExecutionTransactionId::new(19).unwrap(), 0)
            .is_err()
    );
    assert!(
        m.derive_append_prefix(&r, ExecutionTransactionId::new(19).unwrap(), 1)
            .is_err()
    );
    let cap = m
        .derive_append_prefix(&r, ExecutionTransactionId::new(19).unwrap(), 0)
        .unwrap();
    cap.validate().unwrap();
    drop(cap);
    let retired = m.abort_reservations(vec![r]).unwrap();
    m.confirm_page_retirement(retired).unwrap();
}

#[test]
fn admission_rejection_is_observable_without_payload_or_poll_logging() {
    let mut m = manager();
    let s = source(&m);
    let logs = Logs::default();
    tracing::subscriber::with_default(logs.clone(), || {
        let append = m.reserve(slot(0), 7, 1).unwrap();
        assert_eq!(
            m.acquire_committed_read(&s).unwrap_err(),
            HostKvError::TransactionActive
        );
        let retired = m.abort_reservations(vec![append]).unwrap();
        m.confirm_page_retirement(retired).unwrap();
        let lease = m.acquire_committed_read(&s).unwrap();
        for _ in 0..10 {
            m.validate_committed_read(&lease).unwrap();
        }
        m.release_committed_read(&lease).unwrap();
    });
    let events = logs.0.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["event"], "committed_read_rejected");
    assert_eq!(events[0]["reason"], "TransactionActive");
    for field in events[0].keys() {
        assert!(matches!(
            field.as_str(),
            "message" | "event" | "owner" | "reason"
        ));
    }
}

#[test]
fn layer_subcapabilities_preserve_one_live_reservation_without_mutating_authority() {
    #[derive(Debug)]
    struct Layers;
    impl KvLayoutSchema for Layers {
        fn planes(&self) -> &[KvPlaneDescriptor] {
            static P: [KvPlaneDescriptor; 2] = [
                KvPlaneDescriptor::new("standard_gqa.key", 2, 3, KvElementType::F32),
                KvPlaneDescriptor::new("standard_gqa.value", 2, 3, KvElementType::F32),
            ];
            &P
        }
        fn page_size(&self) -> usize {
            4
        }
        fn max_sequence_len(&self) -> usize {
            128
        }
    }
    let mut m = KvPageManager::new(Box::new(Layers), 16);
    m.alloc_sequence(slot(0), 7).unwrap();
    append(&mut m, 0, 3);
    let snapshot = m.capture_prefix_snapshot(slot(0), 3).unwrap();
    let reservation = m.reserve(slot(0), 7, 1).unwrap();
    let prefix = m
        .derive_append_prefix(&reservation, ExecutionTransactionId::new(19).unwrap(), 0)
        .unwrap();
    let frozen = prefix.witness().clone();
    for layer in [1, 2, 0] {
        let child = prefix.for_layer(layer).unwrap();
        child.validate().unwrap();
        assert_eq!(child.witness(), &frozen);
        assert_eq!(child.layer(), layer);
        assert_eq!(prefix.layer(), 0);
        assert_eq!(prefix.witness(), &frozen);
        m.ensure_no_slot_readers(slot(0)).unwrap(); // same reservation, no extra lease
    }
    assert!(prefix.for_layer(3).is_err());
    prefix.validate().unwrap();
    drop(prefix);
    let retirement = m.abort_reservations(vec![reservation]).unwrap();
    m.confirm_page_retirement(retirement).unwrap();
    let retirement = m.release_prefix_snapshot(snapshot).unwrap();
    m.confirm_page_retirement(retirement).unwrap();
    assert_eq!(m.block_table(slot(0)).unwrap().committed_tokens(), 3);
}
