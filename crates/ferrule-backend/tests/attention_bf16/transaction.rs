use super::*;
use ferrule_backend::cpu::{
    KvEndProgress, KvPageStatus, PagedKvBatch, PagedKvPrepare, PagedKvSequence,
};
use ferrule_backend::cuda::operators::attention::prepared_bf16::{
    Bf16AttentionShape, CudaPreparedBf16Attention,
};
use ferrule_backend::cuda::operators::kv::{
    CudaBf16GqaRows, CudaKvView, CudaPagedKvPool, CudaPagedKvTransaction,
};
use ferrule_common::execution::{
    ExecutionTransactionId, KvCowReplacement, KvElementType, KvPageId, KvPlaneDescriptor,
};
use std::rc::Rc;
fn req<'a>(id: u64, new: &'a [KvPageId], cows: &'a [KvCowReplacement]) -> PagedKvPrepare<'a> {
    PagedKvPrepare {
        transaction: ExecutionTransactionId::new(id).unwrap(),
        new_pages: new,
        writable_pages: &[],
        cow_replacements: cows,
        protected_pages: &[],
    }
}
fn end(
    op: &CudaOperators,
    pool: &mut CudaPagedKvPool,
    mut tx: CudaPagedKvTransaction,
    commit: bool,
) {
    pool.leave(&mut tx).unwrap();
    op.record_compute_event().unwrap().synchronize().unwrap();
    let mut tx = Some(tx);
    assert_eq!(
        if commit {
            pool.commit(&mut tx)
        } else {
            pool.abort(&mut tx)
        }
        .unwrap(),
        KvEndProgress::Complete
    );
    assert!(tx.is_none());
}
fn snapshot(
    pool: &mut CudaPagedKvPool,
    page: KvPageId,
) -> ferrule_backend::cuda::operators::kv::TypedKvHostSnapshot {
    pool.preempt(&[page]).unwrap();
    let s = pool.typed_host_snapshot(page).unwrap();
    pool.restore(&[page]).unwrap();
    s
}
fn run_layers(
    op: &CudaOperators,
    view: &mut CudaKvView,
    batch: &PagedKvBatch,
    provider: &mut CudaPreparedBf16Attention,
    status: &mut CudaI32Buffer,
    values: &[u16],
    bad_first: bool,
) -> Vec<f32> {
    let r = batch.row_positions.len();
    let q = op.zero_f32_buffer(r * 4).unwrap();
    let invalid = op.upload_f32_buffer(&vec![1.0000001; r * 4]).unwrap();
    let k = op.upload_bf16_words(&vec![0x3f80; r * 2]).unwrap();
    let v = op.upload_bf16_words(values).unwrap();
    let mut output = op.zero_f32_buffer(r * 4).unwrap();
    provider.begin_invocation(op, status).unwrap();
    for layer in 0..2 {
        view.with_bf16_gqa_planes(op, layer, batch, |mut planes| {
            let layout = planes.attention_layout(2, 1, 2, 0.5)?;
            let composed = planes.prepared_attention_view(provider.plan())?;
            assert_eq!(composed.rows(), r);
            planes.append_and_attend_prepared(
                CudaBf16GqaRows {
                    query: if bad_first && layer == 0 {
                        &invalid
                    } else {
                        &q
                    },
                    append_key: &k,
                    append_value: &v,
                    output: &mut output,
                },
                layout,
                provider,
                status,
            )
        })
        .unwrap();
    }
    if bad_first {
        assert!(provider.finish_invocation(op, status).is_err());
        assert!(!provider.is_poisoned());
    } else {
        provider.finish_invocation(op, status).unwrap();
    }
    op.download_f32_buffer(&output).unwrap()
}
#[test]
#[ignore = "owned GPU; prepared adapter ragged real transaction COW abort/commit and sticky cross-layer status"]
fn prepared_planes_adapter_transaction_cow_and_sticky_status() {
    let op = Rc::new(CudaOperators::new().unwrap());
    let schema = [
        KvPlaneDescriptor::new("standard_gqa.key", 2, 2, KvElementType::Bf16),
        KvPlaneDescriptor::new("standard_gqa.value", 2, 2, KvElementType::Bf16),
    ];
    let mut pool = CudaPagedKvPool::new(op.clone(), &schema, 2, 10).unwrap();
    let (a, b, c) = (KvPageId(10), KvPageId(20), KvPageId(21));
    let shapes = [
        Bf16AttentionShape { rows: 2, tokens: 2 },
        Bf16AttentionShape { rows: 1, tokens: 1 },
        Bf16AttentionShape { rows: 1, tokens: 2 },
    ];
    let plan = Bf16AttentionPlan::from_shapes(&shapes, 3, 2, 2, 1, 2, 64 << 20, 16 << 20).unwrap();
    let mut provider = op.prepare_bf16_attention(plan).unwrap();
    let mut status = op.zero_i32_buffer(1).unwrap();
    let prefill = PagedKvBatch {
        row_sequence_ids: vec![0, 0, 1].into(),
        row_positions: vec![0, 1, 0].into(),
        sequences: vec![
            PagedKvSequence {
                sequence_len: 2,
                block_table: vec![a].into(),
            },
            PagedKvSequence {
                sequence_len: 1,
                block_table: vec![b].into(),
            },
        ]
        .into(),
    };
    let mut tx = pool.prepare(req(1, &[a, b], &[])).unwrap();
    pool.enter(&mut tx, prefill.clone()).unwrap();
    let mut view = pool.active_view(&mut tx).unwrap();
    let got = run_layers(
        &op,
        &mut view,
        &prefill,
        &mut provider,
        &mut status,
        &[0x4000, 0x4080, 0x40c0, 0x4100, 0x4120, 0x4140],
        false,
    );
    assert_eq!(got, [2., 4., 2., 4., 4., 6., 4., 6., 10., 12., 10., 12.]);
    end(&op, &mut pool, tx, true);
    let original = snapshot(&mut pool, b);
    let decode = PagedKvBatch {
        row_sequence_ids: vec![0].into(),
        row_positions: vec![1].into(),
        sequences: vec![PagedKvSequence {
            sequence_len: 2,
            block_table: vec![c].into(),
        }]
        .into(),
    };
    let cow = KvCowReplacement {
        logical_page: 0,
        source: b,
        replacement: c,
    };
    for (id, bad, commit) in [(2, true, false), (3, false, false), (4, false, true)] {
        let mut tx = pool.prepare(req(id, &[], &[cow])).unwrap();
        pool.enter(&mut tx, decode.clone()).unwrap();
        let mut view = pool.active_view(&mut tx).unwrap();
        let got = run_layers(
            &op,
            &mut view,
            &decode,
            &mut provider,
            &mut status,
            &[0x4160, 0x4180],
            bad,
        );
        assert_eq!(
            got,
            if bad {
                vec![0.; 4]
            } else {
                vec![12., 14., 12., 14.]
            }
        );
        // Writable aliases reject through custody adapter before its metadata upload.
        let q = op.zero_f32_buffer(4).unwrap();
        let mut alias = q.prefix(4).unwrap();
        let k = op.upload_bf16_words(&[0x3f80; 2]).unwrap();
        let v = op.upload_bf16_words(&[0x4000; 2]).unwrap();
        provider.begin_invocation(&op, &mut status).unwrap();
        op.reset_counters();
        assert!(
            view.with_bf16_gqa_planes(&op, 0, &decode, |mut planes| {
                let l = planes.attention_layout(2, 1, 2, 0.5)?;
                planes.append_and_attend_prepared(
                    CudaBf16GqaRows {
                        query: &q,
                        append_key: &k,
                        append_value: &v,
                        output: &mut alias,
                    },
                    l,
                    &mut provider,
                    &mut status,
                )
            })
            .is_err()
        );
        assert_eq!(op.counters(), Default::default());
        provider.finish_invocation(&op, &status).unwrap();
        end(&op, &mut pool, tx, commit);
        assert!(
            view.with_bf16_gqa_planes(&op, 0, &decode, |_| Ok(()))
                .is_err()
        );
        assert_eq!(snapshot(&mut pool, b), original);
        assert_eq!(
            pool.page_status(c),
            if commit {
                KvPageStatus::Resident
            } else {
                KvPageStatus::Vacant
            }
        );
    }
    let final_page = snapshot(&mut pool, c);
    assert_eq!(
        final_page.planes[1].bf16_words().unwrap(),
        &[
            0x4120, 0x4140, 0x4160, 0x4180, 0x4120, 0x4140, 0x4160, 0x4180
        ]
    );
    println!(
        "prepared plane custody: ragged prefill, two layers, shared sticky status, actual COW abort/commit, stale views and aliases verified"
    );
}
