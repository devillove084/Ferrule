//! Exercise the production transaction planner without linking CUDA.
use cpu::{PagedKvBatch, PagedKvPrepare, PagedKvSequence};
use ferrule_backend::cpu;
#[path = "../src/cuda/operators/kv/transaction/plan.rs"]
mod plan;
use ferrule_common::execution::{
    ExecutionTransactionId, KvCowReplacement, KvElementType, KvPageId, KvPlaneDescriptor,
};
use plan::{Plan, validate_schema};

fn request<'a>(
    new: &'a [KvPageId],
    writes: &'a [KvPageId],
    cows: &'a [KvCowReplacement],
    reads: &'a [KvPageId],
) -> PagedKvPrepare<'a> {
    PagedKvPrepare {
        transaction: ExecutionTransactionId::new(1).unwrap(),
        new_pages: new,
        writable_pages: writes,
        cow_replacements: cows,
        protected_pages: reads,
    }
}
fn resident(p: KvPageId) -> Option<u32> {
    match p.0 {
        10 => Some(0),
        20 => Some(1),
        _ => None,
    }
}
fn batch(pages: &[KvPageId], position: usize) -> PagedKvBatch {
    PagedKvBatch {
        row_sequence_ids: vec![0].into(),
        row_positions: vec![position].into(),
        sequences: vec![PagedKvSequence {
            sequence_len: position + 1,
            block_table: pages.into(),
        }]
        .into(),
    }
}
fn planes(dtype: KvElementType) -> [KvPlaneDescriptor; 2] {
    [
        KvPlaneDescriptor::new("standard_gqa.key", 8, 3, dtype),
        KvPlaneDescriptor::new("standard_gqa.value", 8, 3, dtype),
    ]
}
#[test]
fn schema_accepts_matching_typed_planes_and_rejects_overflow() {
    validate_schema(&planes(KvElementType::F32), 4, 8).unwrap();
    validate_schema(&planes(KvElementType::Bf16), 4, 8).unwrap();
    for (page_size, slots) in [(0, 8), (4, 0), (usize::MAX, 4), (4, usize::MAX)] {
        assert!(validate_schema(&planes(KvElementType::F32), page_size, slots).is_err());
    }
    let mut mismatch = planes(KvElementType::F32);
    mismatch[1].layer_count = 2;
    assert!(validate_schema(&mismatch, 4, 8).is_err());
}
#[test]
fn shadow_translation_precedes_resident_and_never_reads_undeclared_pages() {
    let mut p = Plan::new(
        request(&[KvPageId(30)], &[KvPageId(10)], &[], &[]),
        resident,
        |_| false,
    )
    .unwrap();
    p.bind_slots(vec![2, 3]);
    p.validate(4, resident, |_| false).unwrap();
    assert_eq!(p.slot(KvPageId(10)), Some(2));
    assert_eq!(p.slot(KvPageId(20)), None);
    assert_eq!(p.copies[&KvPageId(10)], Some(0));
    assert_eq!(
        p.packed_slots(&batch(&[KvPageId(10), KvPageId(30)], 4), 4)
            .unwrap(),
        (vec![2, 3], vec![0, 2])
    );
    assert!(p.packed_slots(&batch(&[KvPageId(20)], 0), 4).is_err());
    assert!(p.validate(3, resident, |_| false).is_err());
    assert!(p.validate(4, |_| None, |_| false).is_err());
    // Failed readiness does not mutate the staged plan.
    assert_eq!(p.slot(KvPageId(30)), Some(3));
}
#[test]
fn cow_sources_are_implicitly_protected_and_all_mutation_sets_conflict() {
    let cows = [KvCowReplacement {
        logical_page: 0,
        source: KvPageId(10),
        replacement: KvPageId(30),
    }];
    let mut cow = Plan::new(request(&[], &[], &cows, &[]), resident, |_| false).unwrap();
    cow.bind_slots(vec![2]);
    assert_eq!(cow.copies[&KvPageId(30)], Some(0));
    assert_eq!(cow.slot(KvPageId(10)), Some(0));
    let writer = Plan::new(request(&[], &[KvPageId(10)], &[], &[]), resident, |_| false).unwrap();
    assert!(cow.conflicts(&writer, false));
    assert!(writer.conflicts(&cow, false));
    let reader = Plan::new(request(&[], &[], &[], &[KvPageId(10)]), resident, |_| false).unwrap();
    assert!(!reader.conflicts(&cow, false));
    assert!(reader.conflicts(&cow, true));
    assert!(cow.packed_slots(&batch(&[KvPageId(10)], 0), 4).is_err());
}
#[test]
fn duplicate_occupied_and_preempted_pages_reject_before_slot_binding() {
    assert!(
        Plan::new(
            request(&[KvPageId(30), KvPageId(30)], &[], &[], &[]),
            resident,
            |_| false
        )
        .is_err()
    );
    assert!(Plan::new(request(&[KvPageId(10)], &[], &[], &[]), resident, |_| false).is_err());
    assert!(Plan::new(request(&[KvPageId(30)], &[], &[], &[]), resident, |_| true).is_err());
    assert!(Plan::new(request(&[], &[], &[], &[KvPageId(99)]), resident, |_| false).is_err());
}
#[test]
fn packed_sequences_keep_owner_local_slots_and_offsets() {
    let mut a = Plan::new(
        request(&[KvPageId(30), KvPageId(40)], &[], &[], &[]),
        resident,
        |_| false,
    )
    .unwrap();
    let mut b = Plan::new(
        request(&[KvPageId(30), KvPageId(40)], &[], &[], &[]),
        resident,
        |_| false,
    )
    .unwrap();
    a.bind_slots(vec![2, 3]);
    b.bind_slots(vec![7, 5]);
    let batch = PagedKvBatch {
        row_sequence_ids: vec![1, 0].into(),
        row_positions: vec![0, 0].into(),
        sequences: vec![
            PagedKvSequence {
                sequence_len: 1,
                block_table: vec![KvPageId(40)].into(),
            },
            PagedKvSequence {
                sequence_len: 1,
                block_table: vec![KvPageId(30)].into(),
            },
        ]
        .into(),
    };
    assert_eq!(
        a.packed_slots(&batch, 4).unwrap(),
        (vec![3, 2], vec![0, 1, 2])
    );
    assert_eq!(
        b.packed_slots(&batch, 4).unwrap(),
        (vec![5, 7], vec![0, 1, 2])
    );
}

#[test]
fn typed_schema_modes_and_physical_bytes_are_exact() {
    let bf16 = planes(KvElementType::Bf16);
    let f32 = planes(KvElementType::F32);
    assert_eq!(bf16[0].checked_page_bytes(4), Some(192));
    assert_eq!(f32[0].checked_page_bytes(4), Some(384));
    plan::validate_plane_type(&bf16, KvElementType::Bf16).unwrap();
    assert!(plan::validate_plane_type(&f32, KvElementType::Bf16).is_err());
    assert!(plan::validate_plane_type(&bf16, KvElementType::F32).is_err());
    let mut mixed = bf16;
    mixed[1].element_type = KvElementType::F32;
    assert!(validate_schema(&mixed, 4, 8).is_err());
    plan::validate_storage_mode(&bf16, false).unwrap();
    plan::validate_storage_mode(&f32, true).unwrap();
    assert!(plan::validate_storage_mode(&bf16, true).is_err());
    let large = [
        KvPlaneDescriptor::new(
            "standard_gqa.key",
            u32::MAX as usize,
            1,
            KvElementType::Bf16,
        ),
        KvPlaneDescriptor::new(
            "standard_gqa.value",
            u32::MAX as usize,
            1,
            KvElementType::Bf16,
        ),
    ];
    assert!(validate_schema(&large, u32::MAX as usize, 1).is_err());
    // Each plane fits usize on 64-bit, but their physical sum does not.
    assert!(validate_schema(&large, 2, i32::MAX as usize).is_err());
}

#[test]
fn bf16_rows_reject_aliases_and_i32_overflow_without_device_work() {
    let mut b = batch(&[KvPageId(30)], 0);
    assert_eq!(
        plan::bf16_row_metadata(&b, 4, &[2], &[0, 1]).unwrap(),
        (vec![0], vec![0], vec![1])
    );
    b.row_sequence_ids = vec![0, 0].into();
    b.row_positions = vec![0, 0].into();
    assert!(plan::bf16_row_metadata(&b, 4, &[2], &[0, 1]).is_err());
    b.row_sequence_ids = vec![0, 1].into();
    b.sequences = vec![b.sequences[0].clone(), b.sequences[0].clone()].into();
    assert!(plan::bf16_row_metadata(&b, 4, &[2, 2], &[0, 1, 2]).is_err());
    let mut b = batch(&[KvPageId(30)], 0);
    b.sequences[0].sequence_len = i32::MAX as usize + 1;
    assert!(plan::bf16_row_metadata(&b, 4, &[2], &[0, 1]).is_err());
    b.sequences[0].sequence_len = 1;
    for (slots, offsets) in [
        (vec![-1], vec![0, 1]),
        (vec![2], vec![0, 2]),
        (vec![2], vec![-1, 1]),
        (vec![2], vec![0]),
    ] {
        assert!(plan::bf16_row_metadata(&b, 4, &slots, &offsets).is_err());
    }
}
