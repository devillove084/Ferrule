//! Private failure injection after real DMA, not a fabricated completion ACK.
use super::*;
use ferrule_common::execution::KvElementType;
fn fixture() -> (Rc<CudaOperators>, CudaPagedKvPool) {
    let ops = Rc::new(CudaOperators::new().unwrap());
    let planes = [
        KvPlaneDescriptor::new("standard_gqa.key", 1, 2, KvElementType::F32),
        KvPlaneDescriptor::new("standard_gqa.value", 1, 2, KvElementType::F32),
    ];
    let pool = CudaPagedKvPool::new_chunked(ops.clone(), &planes, 4, 4, 2).unwrap();
    (ops, pool)
}
fn request() -> PagedKvPrepare<'static> {
    PagedKvPrepare {
        transaction: ExecutionTransactionId::new(1).unwrap(),
        new_pages: &[KvPageId(1)],
        writable_pages: &[],
        protected_pages: &[],
        cow_replacements: &[],
    }
}
fn seed(ops: &CudaOperators, pool: &mut CudaPagedKvPool) {
    let mut tx = Some(pool.prepare(request()).unwrap());
    ops.record_compute_event().unwrap().synchronize().unwrap();
    assert_eq!(pool.commit(&mut tx).unwrap(), KvEndProgress::Complete);
}

#[test]
#[ignore = "requires idle CUDA GPU; compile only during implementation"]
fn dma_final_event_failure_retains_pinned_source_destination_and_mapping() {
    for reload in [false, true] {
        let (ops, mut pool) = fixture();
        seed(&ops, &mut pool);
        if reload {
            pool.preempt(&[KvPageId(1)]).unwrap();
        }
        let before = pool.physical_stats();
        let result = {
            let mut inner = pool.inner.borrow_mut();
            let fail = |_: &CudaOperators| Err(error("injected final DMA fence failure"));
            if reload {
                inner
                    .storage_mut()
                    .restore_chunked_with_completion(&ops, &[KvPageId(1)], fail)
            } else {
                inner
                    .storage_mut()
                    .preempt_chunked_with_completion(&ops, &[KvPageId(1)], fail)
            }
        };
        assert!(result.is_err());
        ops.record_compute_event().unwrap().synchronize().unwrap();
        assert!(pool.has_unresolved_custody());
        assert!(pool.physical_stats().pending_pinned_bytes > 0);
        assert!(pool.physical_stats().device_payload_bytes > 0);
        assert_eq!(
            pool.physical_stats().host_snapshot_bytes,
            before.host_snapshot_bytes
        );
        assert_eq!(
            pool.page_status(KvPageId(1)),
            if reload {
                KvPageStatus::Preempted
            } else {
                KvPageStatus::Resident
            }
        );
        assert!(pool.restore(&[KvPageId(1)]).is_err());
        assert!(pool.preempt(&[KvPageId(1)]).is_err());
        assert!(pool.release(&[KvPageId(1)]).is_err());
        assert!(pool.trim_physical_allocations().is_err());
    }
}

#[test]
#[ignore = "requires idle CUDA GPU; compile only during implementation"]
fn submission_failure_keeps_entry_slots_and_partial_allocations_charged() {
    let (ops, mut pool) = fixture();
    let result = pool.prepare_with_submission(request(), |inner, copies| {
        let ops = inner.context.clone();
        inner
            .storage_mut()
            .clear_transaction_slot(&ops, copies[0].0)?;
        Err(error(
            "injected failure after allocating and initializing page",
        ))
    });
    assert!(result.is_err());
    ops.record_compute_event().unwrap().synchronize().unwrap();
    assert!(pool.has_unresolved_custody());
    assert_eq!(pool.capacity().active_transactions, 1);
    assert_eq!(pool.capacity().free_pages, 0);
    assert!(pool.physical_stats().device_payload_bytes > 0);
    assert!(pool.prepare(request()).is_err());
    assert!(pool.shutdown().is_err());
}

#[test]
#[ignore = "requires idle CUDA GPU; compile only during implementation"]
fn same_slot_reload_advances_physical_epoch_without_changing_logical_page() {
    let (ops, mut pool) = fixture();
    seed(&ops, &mut pool);
    let page = KvPageId(1);
    let slot = pool.physical_slot(page).unwrap();
    let old_epoch = pool.inner.borrow().storage().slot_epoch(slot).unwrap();
    pool.preempt(&[page]).unwrap();
    pool.restore(&[page]).unwrap();
    assert_eq!(pool.physical_slot(page), Some(slot));
    assert!(pool.inner.borrow().storage().slot_epoch(slot).unwrap() > old_epoch);
}
