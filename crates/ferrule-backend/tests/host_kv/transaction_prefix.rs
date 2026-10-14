//! Production Entry sub-record and planner, without CUDA or a second owner.
use super::{
    plan::Plan,
    prefix_read::{PrefixReadBinding, PrefixReadCustody},
};
use ferrule_backend::cpu::{PagedKvBatch, PagedKvPrepare, PagedKvSequence};
use ferrule_common::Error;
use ferrule_common::execution::*;
use ferrule_common::host_kv::{CommittedKvSource, HostKvAppendReadWitness};

fn binding(
    witness: &HostKvAppendReadWitness,
    layer: usize,
    nonce: u64,
    view_epoch: u64,
) -> PrefixReadBinding<'_> {
    PrefixReadBinding {
        witness,
        layer,
        nonce,
        view_epoch,
    }
}

fn fixture(cow: bool) -> (HostKvAppendReadWitness, Plan, PagedKvBatch) {
    let source = CommittedKvSource {
        owner: 71,
        state_slot: StateSlot::new(0),
        generation: 7,
        incarnation: 9,
        schema: KvLayoutSchemaIdentity::new(
            vec![
                KvPlaneDescriptor::new("standard_gqa.key", 2, 2, KvElementType::F32),
                KvPlaneDescriptor::new("standard_gqa.value", 2, 2, KvElementType::F32),
            ],
            4,
            16,
        ),
        committed_tokens: 3,
        positions: 0..3,
        pages: vec![KvPageId(10)],
    };
    let cow = cow.then_some(KvCowReplacement {
        logical_page: 0,
        source: KvPageId(10),
        replacement: KvPageId(20),
    });
    let view = KvReservationView {
        state_slot: source.state_slot,
        execution_state_slot: source.state_slot,
        generation: 7,
        execution_generation: 7,
        positions: 3..4,
        newly_allocated: vec![],
        cow_replacement: cow,
    };
    let witness =
        HostKvAppendReadWitness::new(ExecutionTransactionId::new(1).unwrap(), 2, source, view)
            .unwrap();
    let cows = cow.into_iter().collect::<Vec<_>>();
    let writes = if cow.is_none() {
        vec![KvPageId(10)]
    } else {
        vec![]
    };
    let mut plan = Plan::new(
        PagedKvPrepare {
            transaction: witness.transaction(),
            new_pages: &[],
            writable_pages: &writes,
            protected_pages: &[],
            cow_replacements: &cows,
        },
        |p| (p == KvPageId(10)).then_some(0),
        |_| false,
    )
    .unwrap();
    plan.bind_slots(vec![2]);
    plan.validate(4, |p| (p == KvPageId(10)).then_some(0), |_| false)
        .unwrap();
    let batch = PagedKvBatch {
        row_sequence_ids: vec![0].into(),
        row_positions: vec![3].into(),
        sequences: vec![PagedKvSequence {
            sequence_len: 4,
            block_table: witness.pages().into(),
        }]
        .into(),
    };
    (witness, plan, batch)
}
#[test]
fn page_aligned_prefix_uses_protected_original_and_new_tail_slot() {
    let (w, _, _) = fixture(false);
    let mut source = w.source().clone();
    source.committed_tokens = 4;
    source.positions = 0..4;
    let mut reservation = w.reservation().clone();
    reservation.positions = 4..5;
    reservation.newly_allocated = vec![KvPageId(20)];
    let w = HostKvAppendReadWitness::new(w.transaction(), w.reservation_id(), source, reservation)
        .unwrap();
    let mut plan = Plan::new(
        PagedKvPrepare {
            transaction: w.transaction(),
            new_pages: &[KvPageId(20)],
            writable_pages: &[],
            protected_pages: &[KvPageId(10)],
            cow_replacements: &[],
        },
        |p| (p == KvPageId(10)).then_some(0),
        |_| false,
    )
    .unwrap();
    plan.bind_slots(vec![2]);
    let batch = PagedKvBatch {
        row_sequence_ids: vec![0].into(),
        row_positions: vec![4].into(),
        sequences: vec![PagedKvSequence {
            sequence_len: 5,
            block_table: w.pages().into(),
        }]
        .into(),
    };
    let mut reads = PrefixReadCustody::<bool>::default();
    let handle = reads
        .derive(binding(&w, 1, 4, 2), &batch, &plan, |_| Some(7), true)
        .unwrap();
    reads
        .validate(&handle, binding(&w, 1, 4, 2), &plan, |_| Some(7))
        .unwrap();
    w.validate_partial_partition(4, 0..4, 4..5).unwrap();
    assert!(reads.finish(&handle, |f| Ok(*f)).unwrap());
}

#[test]
fn unsupported_or_forged_logical_geometry_is_rejected() {
    let (w, _, _) = fixture(true);
    for field in 0..6 {
        let mut source = w.source().clone();
        let mut view = w.reservation().clone();
        match field {
            0 => source.positions.start = 1,
            1 => {
                source.committed_tokens = 0;
                source.positions = 0..0;
                source.pages.clear();
            }
            2 => view.positions.end = 5,
            3 => view.generation += 1,
            4 => view.cow_replacement.as_mut().unwrap().source = KvPageId(99),
            _ => view.cow_replacement.as_mut().unwrap().replacement = KvPageId(10),
        }
        assert!(
            HostKvAppendReadWitness::new(w.transaction(), w.reservation_id(), source, view)
                .is_err()
        );
    }
}

fn fault() -> Error {
    Error::Execution {
        message: "injected event failure".into(),
    }
}

#[test]
fn missing_readset_frontier_and_incarnation_are_rejected() {
    let (w, mut p, mut b) = fixture(true);
    let mut reads = PrefixReadCustody::<bool>::default();
    b.row_positions[0] = 2;
    assert!(
        reads
            .derive(binding(&w, 1, 4, 2), &b, &p, |_| Some(7), true)
            .is_err()
    );
    b.row_positions[0] = 3;
    let h = reads
        .derive(binding(&w, 1, 4, 2), &b, &p, |_| Some(7), true)
        .unwrap();
    let mut source = w.source().clone();
    source.incarnation += 1;
    let stale = HostKvAppendReadWitness::new(
        w.transaction(),
        w.reservation_id(),
        source,
        w.reservation().clone(),
    )
    .unwrap();
    assert!(
        reads
            .validate(&h, binding(&stale, 1, 4, 2), &p, |_| Some(7))
            .is_err()
    );
    assert!(reads.finish(&h, |f| Ok(*f)).unwrap());
    assert!(
        reads
            .derive(binding(&stale, 1, 4, 2), &b, &p, |_| Some(7), true)
            .is_err()
    );
    p.original.remove(&KvPageId(10));
    assert!(
        reads
            .derive(binding(&w, 1, 4, 2), &b, &p, |_| Some(7), true)
            .is_err()
    );
    reads.ensure_idle().unwrap();
}

#[test]
fn abort_guard_waits_for_both_exact_domains_and_never_replays_handles() {
    let (w, p, b) = fixture(false);
    let mut reads = PrefixReadCustody::<u32>::default();
    let h = reads
        .derive(binding(&w, 1, 4, 2), &b, &p, |_| Some(7), 10)
        .unwrap();
    reads.begin_submission(&h).unwrap();
    reads.record_fences(&h, 20, 30).unwrap();
    assert!(reads.begin_submission(&h).is_err());
    assert!(reads.record_fences(&h, 40, 50).is_err());
    for pending in [10, 20, 30] {
        assert!(!reads.finish(&h, |f| Ok(*f != pending)).unwrap());
        assert!(reads.ensure_idle().is_err());
    }
    assert!(reads.finish(&h, |_| Ok(true)).unwrap());
    reads.ensure_idle().unwrap();
    let current = reads
        .derive(binding(&w, 1, 4, 2), &b, &p, |_| Some(7), 10)
        .unwrap();
    assert!(reads.finish(&h, |_| Ok(true)).is_err());
    let mut foreign = PrefixReadCustody::<u32>::default();
    let foreign_handle = foreign
        .derive(binding(&w, 1, 4, 2), &b, &p, |_| Some(7), 10)
        .unwrap();
    assert!(foreign.finish(&current, |_| Ok(true)).is_err());
    drop(current);
    assert!(reads.ensure_idle().is_err());
    assert!(foreign.finish(&foreign_handle, |_| Ok(true)).unwrap());
}

#[test]
fn unrecorded_submission_and_query_failure_leave_unknown_sticky() {
    for failure in [0, 10, 20, 30] {
        let (w, p, b) = fixture(false);
        let mut reads = PrefixReadCustody::<u32>::default();
        let h = reads
            .derive(binding(&w, 1, 4, 2), &b, &p, |_| Some(7), 10)
            .unwrap();
        reads.begin_submission(&h).unwrap();
        if failure != 0 {
            reads.record_fences(&h, 20, 30).unwrap();
        }
        assert!(
            reads
                .finish(&h, |f| if *f == failure {
                    Err(fault())
                } else {
                    Ok(true)
                })
                .is_err()
        );
        assert!(reads.is_unknown());
        assert!(reads.finish(&h, |_| Ok(true)).is_err());
        assert!(reads.record_fences(&h, 20, 30).is_err());
        assert!(
            reads
                .validate(&h, binding(&w, 1, 4, 2), &p, |_| Some(7))
                .is_err()
        );
        assert!(reads.ensure_idle().is_err());
        reads.mark_unknown(&h).unwrap();
    }
}

#[test]
fn prefix_tail_contract_rejects_overlap_gaps_and_wrong_query() {
    let (w, _, _) = fixture(false);
    use super::prefix_read::merge_append_partials;
    use ferrule_backend::cpu::{
        StreamingF32GqaConfig, StreamingF32GqaRequest, streaming_f32_gqa_partial,
    };
    let config = StreamingF32GqaConfig {
        rows: 1,
        q_heads: 1,
        kv_heads: 1,
        head_dim: 2,
        block_tokens: 1,
        softmax_scale: 1.0,
        sliding_window: None,
    };
    let request = StreamingF32GqaRequest {
        config,
        query: &[1.0, 0.0],
        key: &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        value: &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 20.0],
        query_start: 3,
        kv_start: 0,
    };
    let prefix = streaming_f32_gqa_partial(request.token_range(0, 3).unwrap()).unwrap();
    let tail = streaming_f32_gqa_partial(request.token_range(3, 4).unwrap()).unwrap();
    let merged = merge_append_partials(&w, config, 0..3, &prefix, 3..4, &tail).unwrap();
    let expected = streaming_f32_gqa_partial(request).unwrap().output();
    for (actual, expected) in merged.output().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
    let wrong_query = streaming_f32_gqa_partial(StreamingF32GqaRequest {
        query_start: 2,
        ..request
    })
    .unwrap();
    assert!(merge_append_partials(&w, config, 0..3, &wrong_query, 3..4, &tail).is_err());
    let wrong_mask = streaming_f32_gqa_partial(StreamingF32GqaRequest {
        config: StreamingF32GqaConfig {
            sliding_window: Some(1),
            ..config
        },
        ..request
    })
    .unwrap();
    assert!(merge_append_partials(&w, config, 0..3, &wrong_mask, 3..4, &tail).is_err());
    w.validate_partial_partition(3, 0..3, 3..4).unwrap();
    for (query, prefix, tail) in [
        (2, 0..3, 3..4),
        (3, 0..2, 3..4),
        (3, 0..4, 3..4),
        (3, 1..3, 3..4),
        (3, 0..3, 4..5),
    ] {
        assert!(w.validate_partial_partition(query, prefix, tail).is_err());
    }
}

#[test]
fn original_prefix_is_disjoint_from_shadow_and_cow_writes() {
    for cow in [false, true] {
        let (w, mut p, b) = fixture(cow);
        let mut reads = PrefixReadCustody::<bool>::default();
        let h = reads
            .derive(binding(&w, 1, 4, 2), &b, &p, |_| Some(7), true)
            .unwrap();
        reads
            .validate(&h, binding(&w, 1, 4, 2), &p, |_| Some(7))
            .unwrap();
        for (layer, nonce, epoch) in [(0, 4, 2), (1, 5, 2), (1, 4, 3)] {
            assert!(
                reads
                    .validate(&h, binding(&w, layer, nonce, epoch), &p, |_| Some(7))
                    .is_err()
            );
        }
        for changed_slot in [0, 2] {
            assert!(
                reads
                    .validate(&h, binding(&w, 1, 4, 2), &p, |s| Some(
                        if s == changed_slot { 8 } else { 7 }
                    ))
                    .is_err()
            );
        }
        assert!(reads.finish(&h, |f| Ok(*f)).unwrap());
        let page = w.pages()[0];
        p.staged.insert(page, 0);
        assert!(
            reads
                .derive(binding(&w, 1, 4, 2), &b, &p, |_| Some(7), true)
                .is_err()
        );
        reads.ensure_idle().unwrap();
        p.staged.insert(page, 2);
        p.copies.insert(page, Some(1));
        assert!(
            reads
                .derive(binding(&w, 1, 4, 2), &b, &p, |_| Some(7), true)
                .is_err()
        );
    }
}

#[test]
fn two_layers_share_entry_only_after_exact_consumer_retirement() {
    let (w, p, b) = fixture(false);
    let mut reads = PrefixReadCustody::<u32>::default();
    let first = reads
        .derive(binding(&w, 0, 4, 2), &b, &p, |_| Some(7), 10)
        .unwrap();
    reads.begin_submission(&first).unwrap();
    reads.record_fences(&first, 20, 30).unwrap();
    assert!(!reads.finish(&first, |f| Ok(*f != 30)).unwrap());
    assert!(
        reads
            .derive(binding(&w, 1, 4, 2), &b, &p, |_| Some(7), 40)
            .is_err()
    );
    assert!(reads.finish(&first, |_| Ok(true)).unwrap());
    let second = reads
        .derive(binding(&w, 1, 4, 2), &b, &p, |_| Some(7), 40)
        .unwrap();
    assert!(reads.finish(&first, |_| Ok(true)).is_err());
    reads.begin_submission(&second).unwrap();
    reads.record_fences(&second, 50, 60).unwrap();
    assert!(reads.finish(&second, |_| Ok(true)).unwrap());
    reads.ensure_idle().unwrap();
}

#[test]
fn unknown_first_layer_blocks_next_layer_even_after_all_late_proofs() {
    let (w, p, b) = fixture(true);
    let mut reads = PrefixReadCustody::<u32>::default();
    let first = reads
        .derive(binding(&w, 0, 4, 2), &b, &p, |_| Some(7), 10)
        .unwrap();
    reads.begin_submission(&first).unwrap();
    reads.record_fences(&first, 20, 30).unwrap();
    reads.mark_unknown(&first).unwrap();
    assert!(reads.finish(&first, |_| Ok(true)).is_err());
    assert!(
        reads
            .derive(binding(&w, 1, 4, 2), &b, &p, |_| Some(7), 40)
            .is_err()
    );
    assert!(reads.ensure_idle().is_err());
    assert!(reads.is_unknown());
}
