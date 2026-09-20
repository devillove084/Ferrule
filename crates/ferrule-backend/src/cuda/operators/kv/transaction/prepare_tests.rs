//! Submission failures are injected locally; no driver/context fault is induced.
use super::*;
use ferrule_common::execution::{KvCowReplacement, KvElementType};

fn pool() -> (Rc<CudaOperators>, CudaPagedKvPool) {
    let context = Rc::new(CudaOperators::new().unwrap());
    let pool = CudaPagedKvPool::new(
        context.clone(),
        &[
            KvPlaneDescriptor::new("standard_gqa.key", 1, 1, KvElementType::F32),
            KvPlaneDescriptor::new("standard_gqa.value", 1, 1, KvElementType::F32),
        ],
        4,
        4,
    )
    .unwrap();
    (context, pool)
}
fn request(id: u64, pages: &[KvPageId]) -> PagedKvPrepare<'_> {
    PagedKvPrepare {
        transaction: ExecutionTransactionId::new(id).unwrap(),
        new_pages: pages,
        writable_pages: &[],
        cow_replacements: &[],
        protected_pages: &[],
    }
}

#[test]
#[ignore = "requires CUDA device and native providers"]
fn failed_clear_copy_or_event_submission_reports_retained_custody() {
    for point in ["clear", "copy", "event"] {
        let (context, mut pool) = pool();
        let mut seed = Some(pool.prepare(request(1, &[KvPageId(1)])).unwrap());
        context
            .record_compute_event()
            .unwrap()
            .synchronize()
            .unwrap();
        pool.commit(&mut seed).unwrap();
        let cows = [KvCowReplacement {
            logical_page: 0,
            source: KvPageId(1),
            replacement: KvPageId(2),
        }];
        let new = [KvPageId(2)];
        let mut prepare = request(2, if point == "copy" { &[] } else { &new });
        if point == "copy" {
            prepare.cow_replacements = &cows;
        }
        let id = prepare.transaction;
        let err = pool
            .prepare_with_submission(prepare, |inner, copies| {
                assert!(inner.entries.contains_key(&id));
                let context = inner.context.clone();
                // Enqueue valid work, then simulate the corresponding submission or
                // final event error. The driver itself remains healthy throughout.
                for &(slot, source) in copies {
                    match source {
                        Some(source) => inner.storage_mut().copy_slot(&context, source, slot)?,
                        None => inner.storage_mut().clear_transaction_slot(&context, slot)?,
                    }
                    if point != "event" {
                        return Err(error(format!("injected {point} submission failure")));
                    }
                }
                Err(error("injected event recording failure"))
            })
            .unwrap_err();
        assert!(err.to_string().contains(point));
        assert!(pool.has_unresolved_custody());
        assert_eq!(pool.capacity().active_transactions, 1);
        assert_eq!(pool.capacity().free_pages, 0);
        assert_eq!(pool.physical_slot(KvPageId(2)), None);
        let mut clone = pool.clone();
        assert!(clone.has_unresolved_custody());
        assert!(clone.release(&[KvPageId(1), KvPageId(2)]).is_err());
        assert!(clone.preempt(&[KvPageId(1)]).is_err());
        assert!(clone.restore(&[KvPageId(2)]).is_err());
        assert!(clone.configure_capacity(4).is_err());
        assert!(clone.shutdown().is_err());
        assert!(
            clone
                .prepare_with_submission(request(3, &[KvPageId(3)]), |_, _| panic!(
                    "poisoned prepare redispatched"
                ))
                .is_err()
        );
        drop(err);
        assert_eq!(pool.capacity().active_transactions, 1);
        // Test-only teardown: positively fence the simulated work before
        // restoring its private authority. Production has no such escape hatch.
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
        pool.release(&[KvPageId(1)]).unwrap();
        pool.shutdown().unwrap();
    }
}

#[test]
#[ignore = "requires CUDA device and native providers"]
fn prepare_validation_failure_does_not_submit_or_retain_resources() {
    let (context, mut pool) = pool();
    let before = pool.capacity();
    assert!(
        pool.prepare_with_submission(request(1, &[KvPageId(1), KvPageId(1)]), |_, _| panic!(
            "invalid prepare submitted"
        ))
        .is_err()
    );
    assert!(!pool.has_unresolved_custody());
    assert_eq!(pool.capacity(), before);
    let mut handle = Some(pool.prepare(request(1, &[KvPageId(1)])).unwrap());
    context
        .record_compute_event()
        .unwrap()
        .synchronize()
        .unwrap();
    assert_eq!(pool.abort(&mut handle).unwrap(), KvEndProgress::Complete);
    assert_eq!(pool.capacity(), before);
    pool.shutdown().unwrap();
}
