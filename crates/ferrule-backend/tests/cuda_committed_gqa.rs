#![cfg(feature = "cuda")]
//! Real payload test, never selected by the driver-free protocol lane.
use ferrule_backend::cpu::{
    KvEndProgress, PagedKvBatch, PagedKvPrepare, PagedKvSequence, StreamingF32GqaConfig,
};
use ferrule_backend::cuda::operator_owner::CudaOperators;
use ferrule_backend::cuda::operators::kv::committed_gqa::{
    CommittedGqaProgress, CudaHostPlacement,
};
use ferrule_backend::cuda::operators::kv::{CudaPagedKvPool, committed_gqa_credits};
use ferrule_common::execution::{
    ExecutionTransactionId, KvElementType, KvLayoutSchemaIdentity, KvPageId, KvPlaneDescriptor,
    StateSlot,
};
use ferrule_common::host_kv::{
    CommittedKvSource,
    placement::{PlacementBudget, PlacementCredits},
};
use std::rc::Rc;
use std::time::{Duration, Instant};

fn config() -> StreamingF32GqaConfig {
    StreamingF32GqaConfig {
        rows: 1,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 3,
        block_tokens: 3,
        softmax_scale: 0.5,
        sliding_window: Some(3),
    }
}
fn fixture() -> (
    Rc<CudaOperators>,
    CudaPagedKvPool,
    CommittedKvSource,
    Vec<f32>,
    Vec<f32>,
) {
    let ops = Rc::new(CudaOperators::new().unwrap());
    let planes = vec![
        KvPlaneDescriptor::new("standard_gqa.key", 6, 2, KvElementType::F32),
        KvPlaneDescriptor::new("standard_gqa.value", 6, 2, KvElementType::F32),
    ];
    let mut pool = CudaPagedKvPool::new(ops.clone(), &planes, 4, 4).unwrap();
    let pages = [KvPageId(9), KvPageId(3)]; // logical and physical order differ
    let mut tx = pool
        .prepare(PagedKvPrepare {
            transaction: ExecutionTransactionId::new(1).unwrap(),
            new_pages: &pages,
            writable_pages: &[],
            cow_replacements: &[],
            protected_pages: &[],
        })
        .unwrap();
    let batch = PagedKvBatch {
        row_sequence_ids: vec![0; 7].into(),
        row_positions: (0..7).collect(),
        sequences: vec![PagedKvSequence {
            sequence_len: 7,
            block_table: pages.into(),
        }]
        .into(),
    };
    pool.enter(&mut tx, batch.clone()).unwrap();
    let key = (0..42)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) / 7.0)
        .collect::<Vec<_>>();
    let value = (0..42)
        .map(|i| ((i * 3 % 19) as f32 - 9.0) / 5.0)
        .collect::<Vec<_>>();
    pool.active_view(&mut tx)
        .unwrap()
        .with_f32_gqa_planes(&ops, 1, &batch, |p| {
            for row in 0..7 {
                let offset = p
                    .layout
                    .resolve_row_offset(0, row, p.key.len(), p.block_slots, p.block_offsets)
                    .unwrap();
                ops.overwrite_f32_range(&key[row * 6..row * 6 + 6], p.key, offset)?;
                ops.overwrite_f32_range(&value[row * 6..row * 6 + 6], p.value, offset)?;
            }
            Ok(())
        })
        .unwrap();
    pool.leave(&mut tx).unwrap();
    ops.record_compute_event().unwrap().synchronize().unwrap();
    assert_eq!(pool.commit(&mut Some(tx)).unwrap(), KvEndProgress::Complete);
    pool.bind_committed_logical_owner(71).unwrap();
    let source = CommittedKvSource {
        owner: 71,
        state_slot: StateSlot::new(0),
        generation: 7,
        incarnation: 1,
        schema: KvLayoutSchemaIdentity::new(planes, 4, 32),
        committed_tokens: 7,
        positions: 0..7,
        pages: pages.into(),
    };
    (ops, pool, source, key, value)
}
fn oracle(c: StreamingF32GqaConfig, q: &[f32], k: &[f32], v: &[f32], position: usize) -> Vec<f32> {
    let mut out = vec![0.0; c.query_elements().unwrap()];
    for head in 0..c.q_heads {
        let kh = head / (c.q_heads / c.kv_heads);
        let visible = (0..7)
            .filter(|&t| {
                t <= position && c.sliding_window.is_none_or(|w| (position - t) < w as usize)
            })
            .collect::<Vec<_>>();
        let logits = visible
            .iter()
            .map(|&t| {
                (0..c.head_dim)
                    .map(|d| {
                        q[head * c.head_dim + d] as f64
                            * k[(t * c.kv_heads + kh) * c.head_dim + d] as f64
                    })
                    .sum::<f64>()
                    * c.softmax_scale as f64
            })
            .collect::<Vec<_>>();
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = logits.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
        let den: f64 = weights.iter().sum();
        for d in 0..c.head_dim {
            out[head * c.head_dim + d] = visible
                .iter()
                .zip(&weights)
                .map(|(&t, &w)| w * v[(t * c.kv_heads + kh) * c.head_dim + d] as f64)
                .sum::<f64>() as f32
                / den as f32;
        }
    }
    out
}
#[test]
#[ignore = "requires an explicitly selected idle CUDA device; compile only by default"]
fn tiny_committed_payload_matches_f64_for_cpu_gpu_and_mixed_splits() {
    let (ops, mut pool, source, key, value) = fixture();
    let q = (0..12).map(|i| (i as f32 - 5.0) / 4.0).collect::<Vec<_>>();
    for window in [None, Some(3)] {
        let c = StreamingF32GqaConfig {
            sliding_window: window,
            ..config()
        };
        for split in [0, 2, 7] {
            let credits = committed_gqa_credits(c, source.pages.len()).unwrap();
            let budget = PlacementBudget::new(credits);
            let admission = CudaHostPlacement::admit(&source, c, &budget).unwrap();
            let read = pool.acquire_committed_read(&source, 1).unwrap();
            let mut clone = pool.clone();
            assert!(clone.release(&source.pages).is_err());
            assert!(clone.preempt(&source.pages).is_err());
            assert!(clone.restore(&source.pages).is_err());
            assert!(clone.shutdown().is_err());
            assert!(
                clone
                    .prepare(PagedKvPrepare {
                        transaction: ExecutionTransactionId::new(2).unwrap(),
                        new_pages: &[],
                        writable_pages: &source.pages,
                        cow_replacements: &[],
                        protected_pages: &[]
                    })
                    .is_err()
            );
            let mut job = CudaHostPlacement::new(
                read,
                ops.upload_f32_buffer(&q).unwrap(),
                c,
                5,
                split,
                admission,
            )
            .unwrap();
            assert!(job.finish().is_err());
            let deadline = Instant::now() + Duration::from_secs(10);
            while job.poll().unwrap() != CommittedGqaProgress::Ready {
                assert!(
                    Instant::now() < deadline,
                    "committed DMA/consumer timed out"
                );
                std::thread::yield_now();
            }
            let mut output = job.finish().unwrap();
            assert!(!output.receipt().canceled());
            assert_eq!(output.receipt().source(), &source);
            assert_eq!(output.receipt().cpu_chunks() > 0, split > 0);
            assert_eq!(output.receipt().gpu_chunks() > 0, split < 7);
            assert!(job.finish().is_err());
            let actual = ops
                .download_f32_range(output.buffer().unwrap(), 0, q.len())
                .unwrap();
            let expected = oracle(c, &q, &key, &value, 5);
            for (a, b) in actual.iter().zip(&expected) {
                assert!((a - b).abs() < 3e-5, "{a} != {b}");
            }
            // The returned device allocation remains charged until release.
            assert_eq!(budget.in_use(), credits);
            output.release().unwrap();
            assert_eq!(budget.in_use(), PlacementCredits::default());
            assert_eq!(pool.capacity().resident_pages, 2);
        }
    }
    pool.release(&source.pages).unwrap();
    pool.shutdown().unwrap();
}
#[test]
#[ignore = "requires an explicitly selected idle CUDA device; compile only by default"]
fn cancel_retains_pin_and_budget_until_two_domain_reaper() {
    let (ops, mut pool, source, _, _) = fixture();
    let c = config();
    let credits = committed_gqa_credits(c, 2).unwrap();
    let budget = PlacementBudget::new(credits);
    let admission = CudaHostPlacement::admit(&source, c, &budget).unwrap();
    let mut foreign = source.clone();
    foreign.owner += 1;
    assert!(pool.acquire_committed_read(&foreign, 1).is_err());
    assert!(pool.bind_committed_logical_owner(foreign.owner).is_err());
    let read = pool.acquire_committed_read(&source, 1).unwrap();
    let mut job = CudaHostPlacement::new(
        read,
        ops.upload_f32_buffer(&[1.0; 12]).unwrap(),
        c,
        6,
        0,
        admission,
    )
    .unwrap();
    job.poll().unwrap();
    job.cancel();
    assert!(job.finish().is_err());
    assert!(pool.release(&source.pages).is_err());
    assert_eq!(budget.in_use(), credits);
    job.reap().unwrap();
    let output = job.finish().unwrap();
    assert!(output.receipt().canceled());
    assert!(output.buffer().is_none());
    assert_eq!(budget.in_use(), PlacementCredits::default());
    pool.release(&source.pages).unwrap();
}
