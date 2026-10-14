use super::*;
use crate::cpu::PagedKvSequence;
use crate::cuda::operators::kv::PagedPlaneLayout;

fn fixture() -> (
    TypedPagedPlaneLayout,
    PagedKvBatch,
    PagedBf16CausalGqaLayout,
) {
    let plane = PagedPlaneLayout {
        page_tokens: 2,
        elements_per_token: 6,
        layer_index: 1,
        layer_count: 3,
    }
    .typed(KvElementType::Bf16);
    let batch = PagedKvBatch {
        row_sequence_ids: vec![0, 1, 0].into(),
        row_positions: vec![0, 0, 1].into(),
        sequences: vec![
            PagedKvSequence {
                sequence_len: 2,
                block_table: vec![KvPageId(10)].into(),
            },
            PagedKvSequence {
                sequence_len: 1,
                block_table: vec![KvPageId(20)].into(),
            },
        ]
        .into(),
    };
    let layout = PagedBf16CausalGqaLayout::packed(3, 2, 4, 2, 3, 2, 1, 3, 5, 0.5).unwrap();
    (plane, batch, layout)
}

#[test]
fn bf16_layout_uses_exact_physical_bytes_and_nonzero_layer() {
    let (plane, batch, layout) = fixture();
    validate_layout(plane, 5, &batch, layout).unwrap();
    assert_eq!(plane.checked_elements_per_page(), Some(36));
    assert_eq!(plane.checked_bytes_per_page(), Some(72));
    assert_eq!(layout.key_cache.capacity_bytes, 360);
    assert_eq!(layout.key_cache.slot_stride_bytes, 72);
    assert_eq!(layout.key_cache.layer_stride_bytes, 24);
    assert_eq!(layout.key_cache.token_stride_bytes, 12);
    assert_eq!(layout.key_cache.head_stride_bytes, 6);
    assert_eq!(
        layout.resolve_cache_byte_offset(false, 0, 1, 1, 2, &[3, 1], &[0, 1, 2]),
        Some(262)
    );
    assert_eq!(
        layout.resolve_cache_byte_offset(false, 0, 2, 1, 2, &[3, 1], &[0, 1, 2]),
        None
    );
    // Row padding is allowed; changing the persistent cache shape is not.
    let mut padded = layout;
    padded.query.row_stride_bytes += 16;
    padded.append_key.row_stride_bytes += 4;
    padded.append_value.row_stride_bytes += 8;
    padded.output.row_stride_bytes += 16;
    validate_layout(plane, 5, &batch, padded).unwrap();
}

#[test]
fn bf16_layout_rejects_dtype_shape_alignment_and_capacity_forgery() {
    let (plane, batch, layout) = fixture();
    assert!(validate_layout(plane.layout.typed(KvElementType::F32), 5, &batch, layout).is_err());
    for capacity in [0, 4, 6, usize::MAX] {
        assert!(validate_layout(plane, capacity, &batch, layout).is_err());
    }
    let mut mutations = Vec::new();
    let mut bad = layout;
    bad.layer_index = 0;
    mutations.push(bad);
    let mut bad = layout;
    bad.page_tokens = 1;
    mutations.push(bad);
    let mut bad = layout;
    bad.layer_count = 2;
    mutations.push(bad);
    let mut bad = layout;
    bad.rows = 2;
    mutations.push(bad);
    let mut bad = layout;
    bad.sequences = 1;
    mutations.push(bad);
    let mut bad = layout;
    bad.kv_heads = 1;
    mutations.push(bad);
    let mut bad = layout;
    bad.head_dim = 2;
    mutations.push(bad);
    let mut bad = layout;
    bad.key_cache.capacity_bytes += 72;
    mutations.push(bad);
    let mut bad = layout;
    bad.value_cache.capacity_bytes -= 72;
    mutations.push(bad);
    let mut bad = layout;
    bad.key_cache.token_stride_bytes += 1;
    mutations.push(bad);
    let mut bad = layout;
    bad.value_cache.slot_stride_bytes += 2;
    mutations.push(bad);
    let mut bad = layout;
    bad.append_key.head_stride_bytes += 1;
    mutations.push(bad);
    let mut bad = layout;
    bad.query.row_stride_bytes = usize::MAX;
    mutations.push(bad);
    for bad in mutations {
        assert!(validate_layout(plane, 5, &batch, bad).is_err());
    }
}

#[test]
fn bf16_operand_ranges_check_alignment_overflow_and_overlap() {
    assert_eq!(byte_range(4096, 8, 2).unwrap(), (4096, 4112));
    assert!(byte_range(4097, 8, 2).is_err());
    assert!(byte_range(4098, 8, 4).is_err());
    assert!(byte_range(0, usize::MAX, 2).is_err());
    assert!(byte_range(u64::MAX - 1, 1, 2).is_err());
    assert!(disjoint((4096, 4112), (4110, 4114)).is_err());
    assert!(disjoint((4096, 4112), (4096, 4112)).is_err());
    disjoint((4096, 4112), (4112, 4114)).unwrap();
    disjoint((4096, 4112), (4100, 4100)).unwrap();
}

#[test]
#[ignore = "requires CUDA GPU; BF16 failed submission retains shared Entry/pins until exact cleanup"]
fn bf16_prepare_failure_retains_typed_custody() {
    use super::super::{CudaPagedKvTransaction, ExecutionTransactionId};
    use crate::cpu::{KvEndProgress, PagedKvPrepare};
    use ferrule_common::execution::KvPlaneDescriptor;
    use std::rc::Rc;

    let context = Rc::new(CudaOperators::new().unwrap());
    let mut pool = CudaPagedKvPool::new(
        context.clone(),
        &[
            KvPlaneDescriptor::new("standard_gqa.key", 2, 2, KvElementType::Bf16),
            KvPlaneDescriptor::new("standard_gqa.value", 2, 2, KvElementType::Bf16),
        ],
        2,
        4,
    )
    .unwrap();
    let id = ExecutionTransactionId::new(1).unwrap();
    let page = KvPageId(9);
    let failed = pool.prepare_with_submission(
        PagedKvPrepare {
            transaction: id,
            new_pages: &[page],
            writable_pages: &[],
            cow_replacements: &[],
            protected_pages: &[],
        },
        |inner, copies| {
            assert!(inner.entries.contains_key(&id));
            let context = inner.context.clone();
            for &(slot, _) in copies {
                clear_transaction_slot(inner, &context, slot)?;
            }
            Err(error("injected BF16 fence recording failure"))
        },
    );
    assert!(failed.unwrap_err().to_string().contains("injected BF16"));
    assert!(pool.has_unresolved_custody());
    assert_eq!(pool.capacity().active_transactions, 1);
    assert_eq!(pool.capacity().free_pages, 0);
    assert_eq!(pool.physical_slot(page), None);
    let mut clone = pool.clone();
    assert!(clone.release(&[page]).is_err());
    assert!(clone.preempt(&[page]).is_err());
    assert!(clone.restore(&[page]).is_err());
    assert!(clone.shutdown().is_err());
    assert_eq!(clone.physical_stats().device_payload_bytes, 128);
    // Test-only teardown follows the existing fault-injection tests: a real
    // positive fence, never a production API that could clear unknown custody.
    let fence = context.record_compute_event().unwrap();
    fence.synchronize().unwrap();
    let handle = {
        let mut inner = pool.inner.borrow_mut();
        let entry = inner.entries.get_mut(&id).unwrap();
        entry.fence = Some(fence);
        let nonce = entry.nonce;
        inner.poisoned = false;
        CudaPagedKvTransaction {
            id,
            identity: inner.identity.clone(),
            nonce,
        }
    };
    assert_eq!(
        pool.abort(&mut Some(handle)).unwrap(),
        KvEndProgress::Complete
    );
    assert_eq!(pool.capacity().free_pages, 4);
    pool.shutdown().unwrap();
}

#[test]
fn prepared_custody_view_is_composed_from_authenticated_batch() {
    use crate::cuda_plan::bf16_attention::Bf16AttentionShape;
    let p = Bf16AttentionPlan::from_shapes(
        &[
            Bf16AttentionShape { rows: 2, tokens: 4 },
            Bf16AttentionShape { rows: 1, tokens: 7 },
        ],
        3,
        2,
        4,
        2,
        3,
        64 << 20,
        16 << 20,
    )
    .unwrap();
    let mut b = PagedKvBatch {
        row_sequence_ids: vec![0, 0, 1].into(),
        row_positions: vec![2, 3, 6].into(),
        sequences: vec![
            PagedKvSequence {
                sequence_len: 4,
                block_table: vec![KvPageId(10), KvPageId(11)].into(),
            },
            PagedKvSequence {
                sequence_len: 7,
                block_table: vec![KvPageId(20), KvPageId(21), KvPageId(22), KvPageId(23)].into(),
            },
        ]
        .into(),
    };
    let v = prepared_view_for_batch(&b, &p).unwrap();
    assert_eq!(v.rows(), 3);
    assert_eq!(
        v.groups()[1],
        Bf16AttentionGroup {
            row_start: 2,
            rows: 1,
            sequence: 1,
            tokens: 7
        }
    );
    b.row_sequence_ids = vec![0, 1, 0].into();
    assert!(prepared_view_for_batch(&b, &p).is_err());
    b.row_sequence_ids = vec![0, 0, 1].into();
    b.row_positions = vec![1, 2, 6].into();
    assert!(prepared_view_for_batch(&b, &p).is_err());
}
