use ferrule_backend::cpu::{
    KvEndProgress, KvPageStatus, PagedKvBatch, PagedKvPrepare, PagedKvSequence,
    StreamingF32GqaConfig, StreamingF32GqaRequest, streaming_f32_gqa_partial,
};
use ferrule_backend::cuda::operator_owner::CudaOperators;
use ferrule_backend::cuda::operators::kv::{CudaPagedKvPool, CudaPagedKvTransaction};
use ferrule_backend::cuda::standard::{PagedF32GqaBuffers, PagedF32GqaLayout, PagedF32GqaMetadata};
use ferrule_common::execution::{
    ExecutionTransactionId, KvCowReplacement, KvElementType, KvPageId, KvPlaneDescriptor,
};
use std::rc::Rc;

fn schema() -> [KvPlaneDescriptor; 2] {
    [
        KvPlaneDescriptor::new("standard_gqa.key", 2, 2, KvElementType::F32),
        KvPlaneDescriptor::new("standard_gqa.value", 2, 2, KvElementType::F32),
    ]
}
fn request<'a>(
    id: u64,
    new: &'a [KvPageId],
    writes: &'a [KvPageId],
    reads: &'a [KvPageId],
) -> PagedKvPrepare<'a> {
    PagedKvPrepare {
        transaction: ExecutionTransactionId::new(id).unwrap(),
        new_pages: new,
        writable_pages: writes,
        protected_pages: reads,
        cow_replacements: &[],
    }
}
fn batch(page: KvPageId) -> PagedKvBatch {
    PagedKvBatch {
        row_sequence_ids: vec![0].into(),
        row_positions: vec![3].into(),
        sequences: vec![PagedKvSequence {
            sequence_len: 4,
            block_table: vec![page].into(),
        }]
        .into(),
    }
}
fn fence(ops: &CudaOperators) {
    ops.record_compute_event().unwrap().synchronize().unwrap();
}
fn payload(plane: usize) -> Vec<f32> {
    let mut values: Vec<_> = (0..16).map(|i| i as f32 * 0.125 + plane as f32).collect();
    // Unused layer payload verifies bitwise preservation of a whole page, not
    // just the layer/range consumed by attention.
    values[8] = f32::from_bits(0x7fc0_1234);
    values[9] = -0.0;
    values
}
fn seed(ops: &CudaOperators, pool: &mut CudaPagedKvPool, page: KvPageId) {
    let mut tx = pool.prepare(request(1, &[page], &[], &[])).unwrap();
    pool.enter(&mut tx, batch(page)).unwrap();
    pool.active_view(&mut tx)
        .unwrap()
        .with_f32_gqa_planes(ops, 0, &batch(page), |p| {
            let offset = p.block_slots[0] as usize * 16;
            ops.overwrite_f32_range(&payload(0), p.key, offset)?;
            ops.overwrite_f32_range(&payload(1), p.value, offset)
        })
        .unwrap();
    pool.leave(&mut tx).unwrap();
    fence(ops);
    assert_eq!(pool.commit(&mut Some(tx)).unwrap(), KvEndProgress::Complete);
}
fn attention(ops: &CudaOperators, pool: &mut CudaPagedKvPool, page: KvPageId) -> Vec<f32> {
    let mut tx = pool.prepare(request(2, &[], &[page], &[])).unwrap();
    pool.enter(&mut tx, batch(page)).unwrap();
    let query = ops.upload_f32_buffer(&[0.3, -0.2]).unwrap();
    let key = ops.upload_f32_buffer(&payload(0)[6..8]).unwrap();
    let value = ops.upload_f32_buffer(&payload(1)[6..8]).unwrap();
    let mut out = ops.zero_f32_buffer(2).unwrap();
    pool.active_view(&mut tx)
        .unwrap()
        .with_f32_gqa_planes(ops, 0, &batch(page), |p| {
            let layout = PagedF32GqaLayout {
                rows: 1,
                sequences: 1,
                q_heads: 1,
                kv_heads: 1,
                head_dim: 2,
                page_tokens: 4,
                layer_index: 0,
                layer_count: 2,
                physical_slots: p.kernel_slot_capacity(),
                softmax_scale: 0.7,
            };
            ops.append_and_attend_paged_f32(
                PagedF32GqaBuffers {
                    query: &query,
                    append_key: &key,
                    append_value: &value,
                    key_cache: p.key,
                    value_cache: p.value,
                    output: &mut out,
                },
                PagedF32GqaMetadata {
                    block_slots: p.block_slots,
                    block_offsets: p.block_offsets,
                    row_sequence_ids: &[0],
                    row_positions: &[3],
                    committed_lengths: &[3],
                },
                layout,
            )
        })
        .unwrap();
    pool.leave(&mut tx).unwrap();
    fence(ops);
    assert_eq!(pool.abort(&mut Some(tx)).unwrap(), KvEndProgress::Complete);
    ops.download_f32_buffer(&out).unwrap()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < 2e-6, "{a} != {b}");
    }
}
fn chunked(ops: &Rc<CudaOperators>) -> CudaPagedKvPool {
    CudaPagedKvPool::new_chunked(ops.clone(), &schema(), 4, 8, 3).unwrap()
}

#[test]
#[ignore = "requires explicitly selected CUDA GPU; compile only during implementation"]
fn all_plane_dma_release_cpu_attention_reload_matches_fixed_pool() {
    let ops = Rc::new(CudaOperators::new().unwrap());
    let mut pool = chunked(&ops);
    let page = KvPageId(7);
    assert_eq!(pool.physical_stats().device_payload_bytes, 0);
    seed(&ops, &mut pool, page);
    let expected = attention(&ops, &mut pool, page);
    let mut fixed = CudaPagedKvPool::new(ops.clone(), &schema(), 4, 8).unwrap();
    seed(&ops, &mut fixed, page);
    close(&expected, &attention(&ops, &mut fixed, page));
    // Clear retired scratch before measuring this page's release. A retained
    // weight allocation stays resident and is not migrated to CPU.
    let weight = ops.upload_f32_buffer(&[7.0; 256]).unwrap();
    fence(&ops);
    pool.trim_physical_allocations().unwrap();
    let before = pool.physical_stats();
    pool.preempt(&[page]).unwrap();
    assert_eq!(pool.page_status(page), KvPageStatus::Preempted);
    let after = pool.physical_stats();
    assert_eq!(after.device_payload_bytes, 0);
    assert_eq!(after.entry_staging_payload_bytes, 0);
    assert_eq!(after.pending_pinned_bytes, 0);
    assert_eq!(after.host_snapshot_bytes, before.device_payload_bytes);
    let snapshot = pool.host_snapshot(page).unwrap();
    for plane in 0..2 {
        assert_eq!(
            snapshot.planes[plane]
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            payload(plane)
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
    }
    // Only real completed allocator retirement + successful driver frees count.
    fence(&ops);
    let mut freed = 0;
    for _ in 0..100 {
        let reclaim = pool.trim_physical_allocations().unwrap();
        assert_eq!(
            reclaim.allocator_trimmed_bytes as u64,
            reclaim.driver_freed_bytes
        );
        freed += reclaim.driver_freed_bytes;
        if freed > 0 {
            break;
        }
        // Retirement records every registered stream; a compute fence alone
        // need not make their subsequent event queries immediately ready.
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(freed > 0);
    assert!(pool.physical_stats().allocator.reserved_bytes < before.allocator.reserved_bytes);
    assert_eq!(ops.download_f32_buffer(&weight).unwrap(), vec![7.0; 256]);
    let cpu = streaming_f32_gqa_partial(StreamingF32GqaRequest {
        config: StreamingF32GqaConfig {
            rows: 1,
            q_heads: 1,
            kv_heads: 1,
            head_dim: 2,
            block_tokens: 4,
            softmax_scale: 0.7,
            sliding_window: None,
        },
        query: &[0.3, -0.2],
        key: &snapshot.planes[0][..8],
        value: &snapshot.planes[1][..8],
        query_start: 3,
        kv_start: 0,
    })
    .unwrap()
    .output();
    close(&cpu, &expected);
    pool.restore(&[page]).unwrap();
    assert_eq!(pool.page_status(page), KvPageStatus::Resident);
    close(&attention(&ops, &mut pool, page), &expected);
    pool.preempt(&[page]).unwrap();
    assert_eq!(pool.host_snapshot(page).unwrap(), snapshot);
    pool.release(&[page]).unwrap();
    assert_eq!(pool.page_status(page), KvPageStatus::Vacant);
}

#[test]
#[ignore = "requires explicitly selected CUDA GPU; compile only during implementation"]
fn cow_pins_aba_reentry_and_escaped_staging_alias_do_not_release_authority() {
    let ops = Rc::new(CudaOperators::new().unwrap());
    let mut pool = chunked(&ops);
    let page = KvPageId(1);
    seed(&ops, &mut pool, page);
    let cow_page = KvPageId(2);
    let cows = [KvCowReplacement {
        logical_page: 0,
        source: page,
        replacement: cow_page,
    }];
    let mut req = request(4, &[], &[], &[]);
    req.cow_replacements = &cows;
    let mut tx = pool.prepare(req).unwrap();
    assert!(pool.preempt(&[page]).is_err());
    assert!(pool.release(&[page]).is_err());
    pool.enter(&mut tx, batch(cow_page)).unwrap();
    let mut stale = pool.active_view(&mut tx).unwrap();
    let mut escaped = stale
        .with_f32_gqa_planes(&ops, 0, &batch(cow_page), |p| p.key.prefix(p.key.len()))
        .unwrap();
    pool.leave(&mut tx).unwrap();
    pool.enter(&mut tx, batch(cow_page)).unwrap();
    assert!(
        stale
            .with_f32_gqa_planes(&ops, 0, &batch(cow_page), |_| Ok(()))
            .is_err()
    );
    pool.leave(&mut tx).unwrap();
    fence(&ops);
    assert_eq!(pool.commit(&mut Some(tx)).unwrap(), KvEndProgress::Complete);
    // Alias is staging, not the authoritative original or installed COW chunk.
    ops.zero_f32_buffer_in_place(&mut escaped).unwrap();
    fence(&ops);
    pool.preempt(&[page, cow_page]).unwrap();
    let a = pool.host_snapshot(page).unwrap();
    let b = pool.host_snapshot(cow_page).unwrap();
    assert_eq!(
        a.planes
            .iter()
            .flatten()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        b.planes
            .iter()
            .flatten()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    pool.restore(&[cow_page]).unwrap();
    // Reuse the same transaction ID after a new allocation/slot incarnation.
    let mut next = pool.prepare(request(4, &[], &[cow_page], &[])).unwrap();
    pool.enter(&mut next, batch(cow_page)).unwrap();
    assert!(stale.validate_batch(&batch(cow_page)).is_err());
    pool.leave(&mut next).unwrap();
    fence(&ops);
    assert_eq!(
        pool.abort(&mut Some(next)).unwrap(),
        KvEndProgress::Complete
    );
}

#[test]
#[ignore = "requires explicitly selected CUDA GPU; compile only during implementation"]
fn canceled_unknown_and_callback_error_remain_charged_after_late_fence() {
    for callback_error in [false, true] {
        let ops = Rc::new(CudaOperators::new().unwrap());
        let mut pool = chunked(&ops);
        let page = KvPageId(1);
        let mut tx = pool.prepare(request(1, &[page], &[], &[])).unwrap();
        pool.enter(&mut tx, batch(page)).unwrap();
        if callback_error {
            let result: ferrule_common::Result<()> = pool
                .active_view(&mut tx)
                .unwrap()
                .with_f32_gqa_planes(&ops, 0, &batch(page), |p| {
                    ops.zero_f32_buffer_in_place(p.key)?;
                    Err(ferrule_common::Error::Execution {
                        message: "injected consumer failure".into(),
                    })
                });
            assert!(result.is_err());
        } else {
            pool.mark_transaction_unknown(&tx).unwrap();
        }
        let bytes = pool.physical_stats();
        fence(&ops);
        assert!(pool.has_unresolved_custody());
        assert_eq!(pool.capacity().free_pages, 0);
        assert!(pool.leave(&mut tx).is_err());
        let mut tx: Option<CudaPagedKvTransaction> = Some(tx);
        assert!(pool.abort(&mut tx).is_err());
        assert!(pool.commit(&mut tx).is_err());
        assert!(pool.trim_physical_allocations().is_err());
        assert!(pool.preempt(&[page]).is_err());
        assert_eq!(
            pool.physical_stats().device_payload_bytes,
            bytes.device_payload_bytes
        );
        assert_eq!(
            pool.physical_stats().entry_staging_payload_bytes,
            bytes.entry_staging_payload_bytes
        );
        assert!(tx.is_some());
        // Intentionally quarantined allocations are not silently recycled on Drop.
    }
}

#[test]
#[ignore = "requires explicitly selected CUDA GPU; compile only during implementation"]
fn prepared_install_and_abort_release_physical_grants_only_at_finish() {
    for abort in [false, true] {
        let ops = Rc::new(CudaOperators::new().unwrap());
        let mut pool = chunked(&ops);
        let page = KvPageId(3);
        seed(&ops, &mut pool, page);
        let original_bytes = pool.physical_stats().device_payload_bytes;
        let mut tx = pool.prepare(request(3, &[], &[page], &[])).unwrap();
        pool.enter(&mut tx, batch(page)).unwrap();
        pool.leave(&mut tx).unwrap();
        fence(&ops);
        if abort {
            assert_eq!(
                pool.abort_prepared(&mut tx, 17).unwrap(),
                KvEndProgress::Complete
            );
        } else {
            assert_eq!(
                pool.commit_prepared(&mut tx, 17).unwrap(),
                KvEndProgress::Complete
            );
            assert_eq!(
                pool.poll_install_ack(&tx, 17).unwrap(),
                KvEndProgress::Complete
            );
            assert!(pool.poll_install_ack(&tx, 18).is_err());
            pool.publish_prepared(&tx).unwrap();
            assert_eq!(
                pool.retire_prepared(&mut tx, 17, &[]).unwrap(),
                KvEndProgress::Complete
            );
        }
        assert_eq!(
            pool.physical_stats().device_payload_bytes,
            2 * original_bytes
        );
        assert!(pool.physical_stats().entry_staging_payload_bytes > 0);
        assert!(pool.preempt(&[page]).is_err());
        assert!(pool.finish_prepared(&mut Some(tx)).unwrap().is_empty());
        assert_eq!(pool.physical_stats().device_payload_bytes, original_bytes);
        assert_eq!(pool.physical_stats().entry_staging_payload_bytes, 0);
        pool.preempt(&[page]).unwrap();
        assert_eq!(
            pool.host_snapshot(page).unwrap().planes[0][0].to_bits(),
            payload(0)[0].to_bits()
        );
    }
}
