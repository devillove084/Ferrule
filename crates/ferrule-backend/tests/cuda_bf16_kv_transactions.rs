#![cfg(feature = "cuda")]
//! Opt-in GPU evidence for the real BF16 transactional cache, not a model profile.
use ferrule_backend::cpu::{
    KvEndProgress, KvPageStatus, PagedKvBatch, PagedKvPrepare, PagedKvSequence,
};
use ferrule_backend::cuda::operator_owner::CudaOperators;
use ferrule_backend::cuda::operators::kv::{
    CudaBf16GqaRows, CudaKvView, CudaPagedKvPool, CudaPagedKvTransaction, TypedKvHostSnapshot,
};
use ferrule_common::execution::{
    ExecutionTransactionId, KvCowReplacement, KvElementType, KvPageId, KvPlaneDescriptor,
};
use std::rc::Rc;

fn schema(dtype: KvElementType) -> [KvPlaneDescriptor; 2] {
    [
        KvPlaneDescriptor::new("standard_gqa.key", 2, 2, dtype),
        KvPlaneDescriptor::new("standard_gqa.value", 2, 2, dtype),
    ]
}
fn request<'a>(
    n: u64,
    new: &'a [KvPageId],
    writes: &'a [KvPageId],
    cows: &'a [KvCowReplacement],
    reads: &'a [KvPageId],
) -> PagedKvPrepare<'a> {
    PagedKvPrepare {
        transaction: ExecutionTransactionId::new(n).unwrap(),
        new_pages: new,
        writable_pages: writes,
        cow_replacements: cows,
        protected_pages: reads,
    }
}
fn batch(pages: &[KvPageId], positions: &[usize], len: usize) -> PagedKvBatch {
    PagedKvBatch {
        row_sequence_ids: vec![0; positions.len()].into(),
        row_positions: positions.into(),
        sequences: vec![PagedKvSequence {
            sequence_len: len,
            block_table: pages.into(),
        }]
        .into(),
    }
}
fn fence(ops: &CudaOperators) {
    ops.record_compute_event().unwrap().synchronize().unwrap();
}
fn finish(
    ops: &CudaOperators,
    pool: &mut CudaPagedKvPool,
    mut tx: CudaPagedKvTransaction,
    commit: bool,
) {
    pool.leave(&mut tx).unwrap();
    fence(ops);
    let mut tx = Some(tx);
    let progress = if commit {
        pool.commit(&mut tx)
    } else {
        pool.abort(&mut tx)
    }
    .unwrap();
    assert_eq!(progress, KvEndProgress::Complete);
    assert!(tx.is_none());
}
fn append(
    ops: &CudaOperators,
    view: &mut CudaKvView,
    batch: &PagedKvBatch,
    layer: usize,
    key: &[u16],
    value: &[u16],
) -> Vec<f32> {
    let q = ops.zero_f32_buffer(batch.row_positions.len() * 4).unwrap();
    let k = ops.upload_bf16_words(key).unwrap();
    let v = ops.upload_bf16_words(value).unwrap();
    let mut out = ops.zero_f32_buffer(q.len()).unwrap();
    view.with_bf16_gqa_planes(ops, layer, batch, |mut planes| {
        assert_eq!(planes.layout().element_type, KvElementType::Bf16);
        assert_eq!(planes.layout().checked_bytes_per_page(), Some(16));
        let layout = planes.attention_layout(2, 1, 2, 0.5)?;
        planes.append_and_attend(
            CudaBf16GqaRows {
                query: &q,
                append_key: &k,
                append_value: &v,
                output: &mut out,
            },
            layout,
        )
    })
    .unwrap();
    ops.download_f32_buffer(&out).unwrap()
}
fn words(values: &[f32]) -> Vec<u16> {
    values.iter().map(|v| (v.to_bits() >> 16) as u16).collect()
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 2e-5, "{a} != {e}");
    }
}
fn snapshot(pool: &mut CudaPagedKvPool, page: KvPageId) -> TypedKvHostSnapshot {
    pool.preempt(&[page]).unwrap();
    let snap = pool.typed_host_snapshot(page).unwrap();
    assert_eq!(pool.page_status(page), KvPageStatus::Preempted);
    assert!(
        pool.host_snapshot(page).is_err(),
        "BF16 must not masquerade as F32"
    );
    assert_eq!(snap.planes.len(), 2);
    for plane in &snap.planes {
        assert_eq!(plane.element_type(), KvElementType::Bf16);
        assert_eq!(plane.byte_len(), 16);
        assert!(plane.f32_bits().is_none());
    }
    pool.restore(&[page]).unwrap();
    assert_eq!(pool.page_status(page), KvPageStatus::Resident);
    assert!(pool.typed_host_snapshot(page).is_err());
    snap
}

#[test]
#[ignore = "requires CUDA GPU; BF16 ragged prefill/decode COW abort/prepared commit"]
fn bf16_ragged_prefill_decode_cow_abort_commit_and_restore() {
    let ops = Rc::new(CudaOperators::new().unwrap());
    let mut pool = CudaPagedKvPool::new(ops.clone(), &schema(KvElementType::Bf16), 2, 10).unwrap();
    assert_eq!(pool.physical_stats().device_payload_bytes, 320);
    assert_eq!(pool.physical_stats().entry_staging_payload_bytes, 0);
    let (a, tail, b, cow_page) = (KvPageId(10), KvPageId(11), KvPageId(20), KvPageId(31));
    let prefill = PagedKvBatch {
        row_sequence_ids: vec![0, 1, 0, 0].into(),
        row_positions: vec![0, 0, 1, 2].into(),
        sequences: vec![
            PagedKvSequence {
                sequence_len: 3,
                block_table: vec![a, tail].into(),
            },
            PagedKvSequence {
                sequence_len: 1,
                block_table: vec![b].into(),
            },
        ]
        .into(),
    };
    let mut tx = pool
        .prepare(request(1, &[a, tail, b], &[], &[], &[]))
        .unwrap();
    pool.enter(&mut tx, prefill.clone()).unwrap();
    let mut view = pool.active_view(&mut tx).unwrap();
    let got = append(
        &ops,
        &mut view,
        &prefill,
        1,
        &words(&[1., 2., 3., 4., 5., 6., 7., 8.]),
        &words(&[2., 4., 10., 20., 6., 8., 12., 16.]),
    );
    close(
        &got,
        &[
            2.,
            4.,
            2.,
            4.,
            10.,
            20.,
            10.,
            20.,
            4.,
            6.,
            4.,
            6.,
            20. / 3.,
            28. / 3.,
            20. / 3.,
            28. / 3.,
        ],
    );
    finish(&ops, &mut pool, tx, true);
    let original = snapshot(&mut pool, tail);
    assert_eq!(
        original.planes[0].bf16_words().unwrap(),
        &[0, 0, 0, 0, 0x40e0, 0x4100, 0, 0]
    );
    let source_slot = pool.physical_slot(tail);
    let cow = KvCowReplacement {
        logical_page: 1,
        source: tail,
        replacement: cow_page,
    };
    let decode = batch(&[a, cow_page], &[3], 4);
    let free = pool.capacity().free_pages;
    let mut tx = pool.prepare(request(2, &[], &[], &[cow], &[a])).unwrap();
    assert!(pool.clone().release(&[tail]).is_err());
    assert!(pool.clone().preempt(&[tail]).is_err());
    assert!(pool.prepare(request(9, &[], &[tail], &[], &[])).is_err());
    pool.enter(&mut tx, decode.clone()).unwrap();
    let mut old = pool.active_view(&mut tx).unwrap();
    assert_ne!(old.physical_slot(cow_page), source_slot);
    let got = append(
        &ops,
        &mut old,
        &decode,
        1,
        &words(&[9., 10.]),
        &words(&[20., 24.]),
    );
    close(&got, &[10., 13., 10., 13.]);
    assert_eq!(pool.physical_slot(tail), source_slot);
    assert_eq!(pool.physical_slot(cow_page), None);
    pool.leave(&mut tx).unwrap();
    assert!(
        old.with_bf16_gqa_planes(&ops, 1, &decode, |_| Ok(()))
            .is_err()
    );
    pool.enter(&mut tx, decode.clone()).unwrap();
    assert!(
        old.with_bf16_gqa_planes(&ops, 1, &decode, |_| Ok(()))
            .is_err()
    );
    finish(&ops, &mut pool, tx, false);
    assert_eq!(pool.capacity().free_pages, free);
    assert_eq!(snapshot(&mut pool, tail), original);
    assert_eq!(pool.page_status(cow_page), KvPageStatus::Vacant);

    let mut tx = pool.prepare(request(3, &[], &[], &[cow], &[a])).unwrap();
    pool.enter(&mut tx, decode.clone()).unwrap();
    let mut view = pool.active_view(&mut tx).unwrap();
    close(
        &append(
            &ops,
            &mut view,
            &decode,
            1,
            &words(&[9., 10.]),
            &words(&[20., 24.]),
        ),
        &[10., 13., 10., 13.],
    );
    pool.leave(&mut tx).unwrap();
    fence(&ops);
    pool.preflight_prepared(&tx).unwrap();
    pool.commit_prepared(&mut tx, 7).unwrap();
    assert!(pool.poll_install_ack(&tx, 8).is_err());
    assert_eq!(
        pool.poll_install_ack(&tx, 7).unwrap(),
        KvEndProgress::Complete
    );
    assert!(pool.clone().release(&[tail, cow_page]).is_err());
    pool.publish_prepared(&tx).unwrap();
    pool.retire_prepared(&mut tx, 7, &[]).unwrap();
    pool.finish_prepared(&mut Some(tx)).unwrap();
    assert_eq!(snapshot(&mut pool, tail), original);
    let result = snapshot(&mut pool, cow_page);
    assert_eq!(
        result.planes[0].bf16_words().unwrap(),
        &[0, 0, 0, 0, 0x40e0, 0x4100, 0x4110, 0x4120]
    );
    assert_eq!(
        result.planes[1].bf16_words().unwrap(),
        &[0, 0, 0, 0, 0x4140, 0x4180, 0x41a0, 0x41c0]
    );
    assert_eq!(snapshot(&mut pool, cow_page), result);
    let boundary = KvPageId(32);
    let next = batch(&[a, cow_page, boundary], &[4], 5);
    let mut tx = pool
        .prepare(request(4, &[boundary], &[], &[], &[a, cow_page]))
        .unwrap();
    pool.enter(&mut tx, next.clone()).unwrap();
    close(
        &append(
            &ops,
            &mut pool.active_view(&mut tx).unwrap(),
            &next,
            1,
            &words(&[11., 12.]),
            &words(&[30., 32.]),
        ),
        &[14., 16.8, 14., 16.8],
    );
    finish(&ops, &mut pool, tx, true);
    assert_eq!(snapshot(&mut pool, cow_page), result);
    pool.release(&[a, tail, b, cow_page, boundary]).unwrap();
    pool.shutdown().unwrap();
}

#[test]
#[ignore = "requires CUDA GPU; BF16 NaN/signed-zero raw bits, COW, rollback, clearing and slot epochs"]
fn bf16_raw_bits_all_layers_survive_cow_abort_and_snapshots() {
    let ops = Rc::new(CudaOperators::new().unwrap());
    let mut pool = CudaPagedKvPool::new(ops.clone(), &schema(KvElementType::Bf16), 2, 4).unwrap();
    let page = KvPageId(1);
    let replacement = KvPageId(2);
    let packed = batch(&[page], &[0, 1], 2);
    let k = [0x8000, 0x7fc1, 0x7f81, 0x0001];
    let v = [0x7f80, 0xff80, 0x0000, 0xffc5];
    let mut tx = pool.prepare(request(1, &[page], &[], &[], &[])).unwrap();
    pool.enter(&mut tx, packed.clone()).unwrap();
    let mut view = pool.active_view(&mut tx).unwrap();
    append(&ops, &mut view, &packed, 0, &k, &v);
    append(&ops, &mut view, &packed, 1, &v, &k);
    finish(&ops, &mut pool, tx, true);
    let original = snapshot(&mut pool, page);
    assert_eq!(original.planes[0].bf16_words().unwrap(), [k, v].concat());
    assert_eq!(original.planes[1].bf16_words().unwrap(), [v, k].concat());
    let cow = KvCowReplacement {
        logical_page: 0,
        source: page,
        replacement,
    };
    let mut tx = pool.prepare(request(2, &[], &[], &[cow], &[])).unwrap();
    pool.enter(&mut tx, batch(&[replacement], &[1], 2)).unwrap();
    // No append: committing the staged copy must preserve every raw BF16 word.
    finish(&ops, &mut pool, tx, true);
    let copied = snapshot(&mut pool, replacement);
    assert_eq!(copied.planes, original.planes);
    let mut tx = pool
        .prepare(request(3, &[], &[replacement], &[], &[]))
        .unwrap();
    let packed = batch(&[replacement], &[0, 1], 2);
    pool.enter(&mut tx, packed.clone()).unwrap();
    let mut stale = pool.active_view(&mut tx).unwrap();
    append(&ops, &mut stale, &packed, 0, &[0; 4], &[0; 4]);
    finish(&ops, &mut pool, tx, false);
    assert_eq!(snapshot(&mut pool, replacement), copied);
    pool.release(&[replacement]).unwrap();
    let mut reused = pool
        .prepare(request(3, &[replacement], &[], &[], &[]))
        .unwrap();
    pool.enter(&mut reused, packed.clone()).unwrap();
    assert!(
        stale
            .with_bf16_gqa_planes(&ops, 0, &packed, |_| Ok(()))
            .is_err()
    );
    finish(&ops, &mut pool, reused, true);
    let cleared = snapshot(&mut pool, replacement);
    for plane in cleared.planes {
        assert_eq!(plane.bf16_words().unwrap(), &[0; 8]);
    }
    assert_eq!(snapshot(&mut pool, page), original);
    pool.release(&[page, replacement]).unwrap();
    pool.shutdown().unwrap();
}

#[test]
#[ignore = "requires CUDA GPU; BF16 typed guard rejects F32, foreign buffers, aliases and unsupported staging"]
fn bf16_guard_rejects_wrong_owner_dtype_layout_and_aliases() {
    let ops = Rc::new(CudaOperators::new().unwrap());
    let foreign = Rc::new(CudaOperators::new().unwrap());
    let mut pool = CudaPagedKvPool::new(ops.clone(), &schema(KvElementType::Bf16), 2, 4).unwrap();
    let mut f32 = CudaPagedKvPool::new(ops.clone(), &schema(KvElementType::F32), 2, 4).unwrap();
    let page = KvPageId(1);
    let packed = batch(&[page], &[0], 1);
    let mut tx = pool.prepare(request(1, &[page], &[], &[], &[])).unwrap();
    let mut ft = f32.prepare(request(1, &[page], &[], &[], &[])).unwrap();
    pool.enter(&mut tx, packed.clone()).unwrap();
    f32.enter(&mut ft, packed.clone()).unwrap();
    let mut view = pool.active_view(&mut tx).unwrap();
    assert!(
        view.with_f32_gqa_planes(&ops, 0, &packed, |_| Ok(()))
            .is_err()
    );
    assert!(
        f32.active_view(&mut ft)
            .unwrap()
            .with_bf16_gqa_planes(&ops, 0, &packed, |_| Ok(()))
            .is_err()
    );
    assert!(
        view.with_bf16_gqa_planes(&foreign, 0, &packed, |_| Ok(()))
            .is_err()
    );
    assert!(
        view.with_bf16_gqa_planes(&ops, 2, &packed, |_| Ok(()))
            .is_err()
    );
    assert!(
        view.with_bf16_gqa_planes(&ops, 0, &batch(&[page], &[1], 2), |_| Ok(()))
            .is_err()
    );
    let q = ops.zero_f32_buffer(4).unwrap();
    let k = ops.zero_bf16_buffer(2).unwrap();
    let v = ops.zero_bf16_buffer(2).unwrap();
    let foreign_k = foreign.zero_bf16_buffer(2).unwrap();
    let mut output = ops.zero_f32_buffer(4).unwrap();
    view.with_bf16_gqa_planes(&ops, 0, &packed, |mut planes| {
        let layout = planes.attention_layout(2, 1, 2, 0.5)?;
        assert!(
            planes
                .append_and_attend(
                    CudaBf16GqaRows {
                        query: &q,
                        append_key: &foreign_k,
                        append_value: &v,
                        output: &mut output
                    },
                    layout
                )
                .is_err()
        );
        let mut forged = layout;
        forged.key_cache.capacity_bytes += 16;
        assert!(
            planes
                .append_and_attend(
                    CudaBf16GqaRows {
                        query: &q,
                        append_key: &k,
                        append_value: &v,
                        output: &mut output
                    },
                    forged
                )
                .is_err()
        );
        let mut alias = q.prefix(4)?;
        assert!(
            planes
                .append_and_attend(
                    CudaBf16GqaRows {
                        query: &q,
                        append_key: &k,
                        append_value: &v,
                        output: &mut alias
                    },
                    layout
                )
                .is_err()
        );
        Ok(())
    })
    .unwrap();
    // An error after actual writes does not detach transaction custody.
    assert!(
        view.with_bf16_gqa_planes(&ops, 0, &packed, |mut planes| {
            let layout = planes.attention_layout(2, 1, 2, 0.5)?;
            planes.append_and_attend(
                CudaBf16GqaRows {
                    query: &q,
                    append_key: &k,
                    append_value: &v,
                    output: &mut output,
                },
                layout,
            )?;
            Err::<(), _>(ferrule_common::Error::Execution {
                message: "injected after append".into(),
            })
        })
        .is_err()
    );
    assert!(pool.preflight_prepared(&tx).is_err());
    finish(&ops, &mut pool, tx, false);
    finish(&ops, &mut f32, ft, false);
    assert_eq!(pool.capacity().free_pages, 4);
    match CudaPagedKvPool::new_chunked(ops.clone(), &schema(KvElementType::Bf16), 2, 4, 2) {
        Err(_) => {}
        Ok(mut chunked) => {
            assert!(chunked.prepare(request(1, &[page], &[], &[], &[])).is_err());
            assert_eq!(chunked.capacity().free_pages, 4);
            assert!(!chunked.has_unresolved_custody());
            chunked.shutdown().unwrap();
        }
    }
    pool.shutdown().unwrap();
    f32.shutdown().unwrap();
}
