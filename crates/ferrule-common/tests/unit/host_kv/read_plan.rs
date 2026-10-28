use super::*;
use crate::execution::{
    ExecutionTransactionId, KvElementType, KvLayoutSchemaIdentity, KvPageId, KvPlaneDescriptor,
    KvReservationView, StateSlot,
};

fn witness_with_transaction(transaction: u64) -> HostKvAppendReadWitness {
    let source = CommittedKvSource {
        owner: 41,
        state_slot: StateSlot::new(3),
        generation: 7,
        incarnation: 11,
        schema: KvLayoutSchemaIdentity::new(
            vec![KvPlaneDescriptor::new("kv", 2, 1, KvElementType::F32)],
            4,
            16,
        ),
        committed_tokens: 6,
        positions: 0..6,
        pages: vec![KvPageId(10), KvPageId(11)],
    };
    HostKvAppendReadWitness::new(
        ExecutionTransactionId::new(transaction).unwrap(),
        19,
        source.clone(),
        KvReservationView {
            state_slot: source.state_slot,
            execution_state_slot: StateSlot::new(8),
            positions: 6..7,
            newly_allocated: vec![],
            generation: source.generation,
            execution_generation: 23,
            cow_replacement: None,
        },
    )
    .unwrap()
}

fn witness() -> HostKvAppendReadWitness {
    witness_with_transaction(9)
}

#[test]
fn full_read_is_page_bounded_and_destination_packed() {
    let witness = witness();
    let plan = witness.read_plan().unwrap();

    assert_eq!(plan.logical_tokens, 0..7);
    assert_eq!(plan.destination_length, 56);
    assert_eq!(plan.segments().len(), 2);

    let first = &plan.segments()[0];
    assert_eq!(first.source_page, KvPageId(10));
    assert_eq!(first.logical_tokens, 0..4);
    assert_eq!(first.page_tokens, 0..4);
    assert_eq!(first.destination_offset, 0);
    assert_eq!(first.destination_length, 32);

    let second = &plan.segments()[1];
    assert_eq!(second.source_page, KvPageId(11));
    assert_eq!(second.logical_tokens, 4..7);
    assert_eq!(second.page_tokens, 0..3);
    assert_eq!(second.destination_offset, 32);
    assert_eq!(second.destination_length, 24);
    assert_eq!(first.capability(), plan.capability());
    assert_eq!(first.capability().transaction(), witness.transaction());
    assert_eq!(
        first.capability().reservation_id(),
        witness.reservation_id()
    );
    assert_eq!(first.capability().generation(), witness.source().generation);
    witness.validate_read_plan(&plan).unwrap();
}

#[test]
fn ragged_partial_and_page_boundary_ranges_lower_without_gaps() {
    let witness = witness();
    let ragged = witness.read_plan_for(1..6).unwrap();
    assert_eq!(ragged.segments().len(), 2);
    assert_eq!(ragged.segments()[0].logical_tokens, 1..4);
    assert_eq!(ragged.segments()[0].page_tokens, 1..4);
    assert_eq!(ragged.segments()[0].destination_offset, 0);
    assert_eq!(ragged.segments()[0].destination_length, 24);
    assert_eq!(ragged.segments()[1].logical_tokens, 4..6);
    assert_eq!(ragged.segments()[1].page_tokens, 0..2);
    assert_eq!(ragged.segments()[1].destination_offset, 24);
    assert_eq!(ragged.segments()[1].destination_length, 16);
    assert_eq!(ragged.destination_length, 40);

    let boundary = witness.read_plan_for(4..7).unwrap();
    assert_eq!(boundary.segments().len(), 1);
    assert_eq!(boundary.segments()[0].source_page, KvPageId(11));
    assert_eq!(boundary.segments()[0].page_tokens, 0..3);
    assert_eq!(boundary.segments()[0].destination_offset, 0);
    assert_eq!(boundary.destination_length, 24);

    assert_eq!(witness.read_plan_for(0..0), Err(HostKvError::InvalidRange));
    assert_eq!(witness.read_plan_for(7..8), Err(HostKvError::InvalidRange));
}

#[test]
fn stale_generation_and_transaction_capabilities_are_rejected() {
    let witness = witness();
    let mut stale = witness.read_plan().unwrap();
    std::sync::Arc::make_mut(&mut stale.capability).generation += 1;
    assert_eq!(
        witness.validate_read_plan(&stale),
        Err(HostKvError::StaleSource)
    );

    let other = witness_with_transaction(10);
    let other_plan = other.read_plan().unwrap();
    assert_eq!(
        witness.validate_read_plan(&other_plan),
        Err(HostKvError::StaleSource)
    );

    let mut stale_segment = witness.read_plan().unwrap();
    std::sync::Arc::make_mut(&mut stale_segment.segments[0].capability).transaction =
        ExecutionTransactionId::new(99).unwrap();
    assert_eq!(
        witness.validate_read_plan(&stale_segment),
        Err(HostKvError::StaleSource)
    );
}

#[test]
fn source_range_destination_overlap_and_page_aliases_are_rejected() {
    let witness = witness();

    let mut out_of_range = witness.read_plan().unwrap();
    out_of_range.logical_tokens = 0..8;
    assert_eq!(
        witness.validate_read_plan(&out_of_range),
        Err(HostKvError::InvalidRange)
    );

    let mut overlapping = witness.read_plan().unwrap();
    overlapping.segments[1].destination_offset = overlapping.segments[0].destination_offset;
    assert_eq!(
        witness.validate_read_plan(&overlapping),
        Err(HostKvError::InvalidRange)
    );

    let mut logical_gap = witness.read_plan().unwrap();
    logical_gap.segments[1].logical_tokens = 5..7;
    assert_eq!(
        witness.validate_read_plan(&logical_gap),
        Err(HostKvError::InvalidRange)
    );

    let mut wrong_length = witness.read_plan().unwrap();
    wrong_length.segments[0].destination_length -= 1;
    assert_eq!(
        witness.validate_read_plan(&wrong_length),
        Err(HostKvError::InvalidGeometry)
    );

    let mut forged_page = witness.read_plan().unwrap();
    forged_page.segments[0].source_page = KvPageId(999);
    assert_eq!(
        witness.validate_read_plan(&forged_page),
        Err(HostKvError::StaleSource)
    );
}

#[test]
fn checked_schema_byte_accounting_rejects_overflow_before_plan_creation() {
    let elements = usize::MAX / 4;
    let source = CommittedKvSource {
        owner: 41,
        state_slot: StateSlot::new(3),
        generation: 7,
        incarnation: 11,
        schema: KvLayoutSchemaIdentity::new(
            vec![
                KvPlaneDescriptor::new("left", elements, 1, KvElementType::F32),
                KvPlaneDescriptor::new("right", elements, 1, KvElementType::F32),
            ],
            1,
            2,
        ),
        committed_tokens: 1,
        positions: 0..1,
        pages: vec![KvPageId(10)],
    };
    let reservation = KvReservationView {
        state_slot: source.state_slot,
        execution_state_slot: StateSlot::new(8),
        positions: 1..2,
        newly_allocated: vec![KvPageId(11)],
        generation: source.generation,
        execution_generation: 23,
        cow_replacement: None,
    };
    assert_eq!(
        HostKvAppendReadWitness::new(
            ExecutionTransactionId::new(9).unwrap(),
            19,
            source,
            reservation,
        ),
        Err(HostKvError::Overflow)
    );
}
