use super::*;
use crate::cpu::PagedKvSequence;
use ferrule_common::execution::KvElementType;

fn fixture() -> (Rc<CudaOperators>, CudaPagedKvPool, CudaPagedKvTransaction) {
    let context = Rc::new(CudaOperators::new().unwrap());
    let mut pool = CudaPagedKvPool::new(
        context.clone(),
        &[
            KvPlaneDescriptor::new("standard_gqa.key", 1, 1, KvElementType::F32),
            KvPlaneDescriptor::new("standard_gqa.value", 1, 1, KvElementType::F32),
        ],
        4,
        2,
    )
    .unwrap();
    let mut tx = pool.prepare(request()).unwrap();
    pool.enter(
        &mut tx,
        PagedKvBatch {
            row_sequence_ids: vec![0].into(),
            row_positions: vec![0].into(),
            sequences: vec![PagedKvSequence {
                sequence_len: 1,
                block_table: vec![KvPageId(1)].into(),
            }]
            .into(),
        },
    )
    .unwrap();
    pool.leave(&mut tx).unwrap();
    context
        .record_compute_event()
        .unwrap()
        .synchronize()
        .unwrap();
    (context, pool, tx)
}
fn request() -> PagedKvPrepare<'static> {
    PagedKvPrepare {
        transaction: ExecutionTransactionId::new(1).unwrap(),
        new_pages: &[KvPageId(1)],
        writable_pages: &[],
        cow_replacements: &[],
        protected_pages: &[],
    }
}

#[test]
#[ignore = "requires CUDA device and native providers"]
fn unknown_exact_fence_retains_mapping_slots_and_all_clone_pins() {
    let (_context, mut pool, mut tx) = fixture();
    let mut clone = pool.clone();
    let capacity = pool.capacity();
    let fence = pool
        .inner
        .borrow_mut()
        .entries
        .get_mut(&tx.id)
        .unwrap()
        .fence
        .take();
    assert!(pool.preflight_prepared(&tx).is_err());
    assert!(pool.commit_prepared(&mut tx, 1).is_err());
    assert!(pool.abort_prepared(&mut tx, 1).is_err());
    assert_eq!(pool.capacity(), capacity);
    assert_eq!(pool.physical_slot(KvPageId(1)), None);
    assert!(clone.release(&[KvPageId(1)]).is_err());
    assert!(clone.preempt(&[KvPageId(1)]).is_err());
    assert!(clone.restore(&[KvPageId(1)]).is_err());
    assert!(clone.configure_capacity(2).is_err());
    assert!(clone.shutdown().is_err());
    let mut tx = Some(tx);
    assert!(pool.commit(&mut tx).is_err());
    assert!(pool.abort(&mut tx).is_err());
    assert!(pool.abort_prepared(tx.as_mut().unwrap(), 2).is_err());
    assert!(pool.finish_prepared(&mut tx).is_err());
    assert!(tx.is_some());
    // Only restoration of the exact test-removed event makes this safe to finish.
    pool.inner
        .borrow_mut()
        .entries
        .get_mut(&tx.as_ref().unwrap().id)
        .unwrap()
        .fence = fence;
    pool.abort_prepared(tx.as_mut().unwrap(), 1).unwrap();
    pool.finish_prepared(&mut tx).unwrap();
    assert_eq!(pool.capacity().free_pages, 2);
}

#[test]
#[ignore = "requires CUDA device and native providers"]
fn stale_nonce_and_abandoned_handles_cannot_release_reused_transaction_custody() {
    let (context, mut pool, tx) = fixture();
    let stale = CudaPagedKvTransaction {
        id: tx.id,
        identity: tx.identity.clone(),
        nonce: tx.nonce,
    };
    pool.abort(&mut Some(tx)).unwrap();
    let current = pool.prepare(request()).unwrap();
    let mut stale = Some(stale);
    assert!(pool.abort(&mut stale).is_err());
    assert!(pool.commit(&mut stale).is_err());
    assert!(pool.finish_prepared(&mut stale).is_err());
    assert!(stale.is_some());
    let exact = CudaPagedKvTransaction {
        id: current.id,
        identity: current.identity.clone(),
        nonce: current.nonce,
    };
    drop(current);
    assert_eq!(pool.capacity().active_transactions, 1);
    assert!(pool.prepare(request()).is_err());
    assert!(pool.clone().shutdown().is_err());
    context
        .record_compute_event()
        .unwrap()
        .synchronize()
        .unwrap();
    // This private test-only reconstruction is impossible through the public API.
    pool.abort(&mut Some(exact)).unwrap();
}
