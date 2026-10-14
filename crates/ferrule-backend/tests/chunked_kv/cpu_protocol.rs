use super::plan::Plan;
use super::protocol::{StickyCompletion, admit, compact_slots};
use ferrule_backend::cpu::{PagedKvBatch, PagedKvPrepare, PagedKvSequence};
use ferrule_common::Error;
use ferrule_common::execution::{ExecutionTransactionId, KvCowReplacement, KvPageId};

#[test]
fn bounded_staging_admits_before_any_physical_reservation() {
    assert!(admit(2, 2, 0).is_ok());
    assert!(admit(3, 2, 0).is_err());
    assert!(admit(1, 2, 1).is_err());
    assert!(admit(0, 0, 0).is_err());
    let table = compact_slots([7, 2, 99], 3).unwrap();
    assert_eq!([table[&7], table[&2], table[&99]], [0, 1, 2]);
    assert!(compact_slots([2, 2], 2).is_err());
    assert!(compact_slots([2, 3], 1).is_err());
}

#[test]
fn pending_is_not_ack_and_unknown_cannot_be_laundered_by_late_success() {
    let c = StickyCompletion::default();
    assert!(!c.probe(|| Ok(false)).unwrap());
    assert!(!c.is_unknown());
    assert!(
        c.probe(|| Err::<bool, _>(Error::Execution {
            message: "event failed".into()
        }))
        .is_err()
    );
    assert!(c.is_unknown());
    assert!(c.probe(|| Ok(true)).is_err());
    let c = StickyCompletion::default();
    c.cancel_unknown();
    assert!(c.probe(|| Ok(true)).is_err());
}

#[test]
fn probe_unwind_retains_unknown_instead_of_refunding_custody() {
    let c = StickyCompletion::default();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: ferrule_common::Result<()> = c.probe(|| panic!("injected submission unwind"));
        }))
        .is_err()
    );
    assert!(c.is_unknown());
    assert!(c.probe(|| Ok(())).is_err());
}

#[test]
fn compact_coordinates_preserve_cow_originals_and_reject_stale_mapping() {
    let source = KvPageId(9);
    let replacement = KvPageId(10);
    let cow = [KvCowReplacement {
        logical_page: 0,
        source,
        replacement,
    }];
    let resident = |page| if page == source { Some(7) } else { None };
    let mut plan = Plan::new(
        PagedKvPrepare {
            transaction: ExecutionTransactionId::new(1).unwrap(),
            new_pages: &[],
            writable_pages: &[],
            cow_replacements: &cow,
            protected_pages: &[],
        },
        resident,
        |_| false,
    )
    .unwrap();
    plan.bind_slots(vec![2]);
    plan.validate(8, resident, |_| false).unwrap();
    assert_eq!(plan.original[&source], Some(7));
    assert_eq!(plan.slot(replacement), Some(2));
    let batch = PagedKvBatch {
        row_sequence_ids: vec![0].into(),
        row_positions: vec![0].into(),
        sequences: vec![PagedKvSequence {
            sequence_len: 1,
            block_table: vec![replacement].into(),
        }]
        .into(),
    };
    let (slots, offsets) = plan.packed_slots(&batch, 4).unwrap();
    let compact = compact_slots(plan.original.keys().map(|p| plan.slot(*p).unwrap()), 2).unwrap();
    assert_eq!(
        slots
            .iter()
            .map(|s| compact[&(*s as u32)])
            .collect::<Vec<_>>(),
        [1]
    );
    assert_eq!(offsets, [0, 1]);
    assert!(plan.validate(8, |_| Some(3), |_| false).is_err());
    // Slot-epoch/nonce ABA authentication is tested against actual Entry below.
}

#[test]
fn staging_bound_counts_all_declared_pages_not_only_packed_writes() {
    let writable = KvPageId(7);
    let source = KvPageId(9);
    let replacement = KvPageId(10);
    let new_page = KvPageId(13);
    let resident = |page: KvPageId| match page.0 {
        7 => Some(0),
        9 => Some(1),
        11 => Some(2),
        12 => Some(3),
        _ => None,
    };
    let mut plan = Plan::new(
        PagedKvPrepare {
            transaction: ExecutionTransactionId::new(2).unwrap(),
            new_pages: &[new_page],
            writable_pages: &[writable],
            cow_replacements: &[KvCowReplacement {
                logical_page: 0,
                source,
                replacement,
            }],
            protected_pages: &[KvPageId(11), KvPageId(12)],
        },
        resident,
        |_| false,
    )
    .unwrap();
    assert_eq!(plan.original.len(), 6);
    assert!(admit(plan.original.len(), 5, 0).is_err());
    admit(plan.original.len(), 6, 0).unwrap();
    assert!(admit(plan.original.len(), 6, 1).is_err());
    plan.bind_slots(vec![4, 5, 6]);
    plan.validate(7, resident, |_| false).unwrap();
    let batch = PagedKvBatch {
        row_sequence_ids: vec![0].into(),
        row_positions: vec![0].into(),
        sequences: vec![PagedKvSequence {
            sequence_len: 1,
            block_table: vec![writable].into(),
        }]
        .into(),
    };
    let (packed, _) = plan.packed_slots(&batch, 4).unwrap();
    assert_eq!(packed, [4]);
    let all_slots = || plan.original.keys().map(|page| plan.slot(*page).unwrap());
    assert!(compact_slots(all_slots(), 5).is_err());
    let staging = compact_slots(all_slots(), 6).unwrap();
    assert_eq!(
        staging.len(),
        6,
        "gather includes declared pages omitted from the packed batch"
    );
    for slot in [1, 2, 3, 4, 5, 6] {
        assert!(staging.contains_key(&slot));
    }
}
