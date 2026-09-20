#![cfg(feature = "cuda")]
use ferrule_backend::cpu::{KvEndProgress, PagedKvBatch, PagedKvPrepare, PagedKvSequence};
use ferrule_backend::cuda::context::CudaOperators;
use ferrule_backend::cuda::operators::kv::{CudaKvView, CudaPagedKvPool, CudaPagedKvTransaction};
use ferrule_common::execution::{
    ExecutionTransactionId, KvCowReplacement, KvElementType, KvPageId, KvPlaneDescriptor,
};
use std::rc::Rc;

fn id(n: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(n).unwrap()
}
fn pool(ops: &Rc<CudaOperators>) -> CudaPagedKvPool {
    CudaPagedKvPool::new(
        ops.clone(),
        &[
            KvPlaneDescriptor::new("standard_gqa.key", 1, 2, KvElementType::F32),
            KvPlaneDescriptor::new("standard_gqa.value", 1, 2, KvElementType::F32),
        ],
        4,
        6,
    )
    .unwrap()
}
fn request<'a>(
    n: u64,
    new: &'a [KvPageId],
    writes: &'a [KvPageId],
    cows: &'a [KvCowReplacement],
    reads: &'a [KvPageId],
) -> PagedKvPrepare<'a> {
    PagedKvPrepare {
        transaction: id(n),
        new_pages: new,
        writable_pages: writes,
        cow_replacements: cows,
        protected_pages: reads,
    }
}
fn batch(page: KvPageId) -> PagedKvBatch {
    PagedKvBatch {
        row_sequence_ids: vec![0].into(),
        row_positions: vec![0].into(),
        sequences: vec![PagedKvSequence {
            sequence_len: 1,
            block_table: vec![page].into(),
        }]
        .into(),
    }
}
fn fence(ops: &CudaOperators) {
    ops.record_compute_event().unwrap().synchronize().unwrap();
}
fn write(ops: &CudaOperators, view: &mut CudaKvView, page: KvPageId, value: f32) {
    view.with_f32_gqa_planes(ops, 1, &batch(page), |planes| {
        let offset = planes
            .layout
            .resolve_row_offset(
                0,
                0,
                planes.key.len(),
                planes.block_slots,
                planes.block_offsets,
            )
            .unwrap();
        ops.overwrite_f32_range(&[value], planes.key, offset)?;
        ops.overwrite_f32_range(&[value + 10.0], planes.value, offset)
    })
    .unwrap();
}
fn staged_value(
    ops: &CudaOperators,
    pool: &mut CudaPagedKvPool,
    tx: &mut CudaPagedKvTransaction,
    page: KvPageId,
) -> f32 {
    let mut view = pool.active_view(tx).unwrap();
    view.with_f32_gqa_planes(ops, 1, &batch(page), |planes| {
        let offset = planes
            .layout
            .resolve_row_offset(
                0,
                0,
                planes.key.len(),
                planes.block_slots,
                planes.block_offsets,
            )
            .unwrap();
        Ok(ops.download_f32_range(planes.key, offset, 1)?[0])
    })
    .unwrap()
}
fn seed(ops: &CudaOperators, pool: &mut CudaPagedKvPool, page: KvPageId, value: f32) {
    let mut tx = pool.prepare(request(1, &[page], &[], &[], &[])).unwrap();
    pool.enter(&mut tx, batch(page)).unwrap();
    write(ops, &mut pool.active_view(&mut tx).unwrap(), page, value);
    pool.leave(&mut tx).unwrap();
    fence(ops);
    assert_eq!(pool.commit(&mut Some(tx)).unwrap(), KvEndProgress::Complete);
}
fn committed_value(ops: &CudaOperators, pool: &mut CudaPagedKvPool, page: KvPageId) -> f32 {
    let mut tx = pool.prepare(request(99, &[], &[page], &[], &[])).unwrap();
    pool.enter(&mut tx, batch(page)).unwrap();
    let value = staged_value(ops, pool, &mut tx, page);
    pool.leave(&mut tx).unwrap();
    fence(ops);
    assert_eq!(pool.abort(&mut Some(tx)).unwrap(), KvEndProgress::Complete);
    value
}

#[test]
#[ignore = "requires CUDA device and native providers"]
fn shadow_abort_preserves_committed_payload_and_old_views_never_revive() {
    let ops = Rc::new(CudaOperators::new().unwrap());
    let mut pool = pool(&ops);
    let page = KvPageId(1);
    seed(&ops, &mut pool, page, 3.0);
    let resident = pool.physical_slot(page).unwrap();
    let free = pool.capacity().free_pages;
    let mut tx = pool.prepare(request(2, &[], &[page], &[], &[])).unwrap();
    pool.enter(&mut tx, batch(page)).unwrap();
    let mut old = pool.active_view(&mut tx).unwrap();
    assert_ne!(old.physical_slot(page), Some(resident));
    write(&ops, &mut old, page, 9.0);
    assert_eq!(pool.physical_slot(page), Some(resident));
    assert_eq!(pool.capacity().free_pages, free - 1);
    assert!(pool.preflight_prepared(&tx).is_err());
    pool.leave(&mut tx).unwrap();
    assert!(old.validate_batch(&batch(page)).is_err());
    pool.enter(&mut tx, batch(page)).unwrap();
    assert!(old.validate_batch(&batch(page)).is_err());
    assert!(
        old.with_f32_gqa_planes(&ops, 1, &batch(page), |_| Ok(()))
            .is_err()
    );
    pool.leave(&mut tx).unwrap();
    fence(&ops);
    pool.preflight_prepared(&tx).unwrap();
    assert_eq!(
        pool.abort_prepared(&mut tx, 7).unwrap(),
        KvEndProgress::Complete
    );
    assert!(pool.abort_prepared(&mut tx, 8).is_err());
    let mut clone = pool.clone();
    assert!(clone.release(&[page]).is_err());
    assert!(clone.preempt(&[page]).is_err());
    assert!(clone.restore(&[page]).is_err());
    assert!(clone.shutdown().is_err());
    assert_eq!(pool.capacity().free_pages, free - 1);
    let mut tx = Some(tx);
    assert!(clone.abort(&mut tx).is_err());
    assert!(tx.is_some());
    pool.finish_prepared(&mut tx).unwrap();
    assert!(tx.is_none());
    assert_eq!(pool.capacity().free_pages, free);
    assert_eq!(committed_value(&ops, &mut pool, page), 3.0);
    pool.release(&[page]).unwrap();
    pool.shutdown().unwrap();
}

#[test]
#[ignore = "requires CUDA device and native providers"]
fn cow_copy_protects_source_until_exact_retirement_finish() {
    let ops = Rc::new(CudaOperators::new().unwrap());
    let mut pool = pool(&ops);
    let source = KvPageId(1);
    let destination = KvPageId(2);
    seed(&ops, &mut pool, source, 3.0);
    let cow = KvCowReplacement {
        logical_page: 0,
        source,
        replacement: destination,
    };
    let mut tx = pool.prepare(request(2, &[], &[], &[cow], &[])).unwrap();
    assert!(pool.release(&[source]).is_err());
    assert!(pool.prepare(request(3, &[], &[source], &[], &[])).is_err());
    pool.enter(&mut tx, batch(destination)).unwrap();
    assert_eq!(staged_value(&ops, &mut pool, &mut tx, destination), 3.0);
    let mut view = pool.active_view(&mut tx).unwrap();
    assert_eq!(view.physical_slot(source), pool.physical_slot(source));
    write(&ops, &mut view, destination, 7.0);
    pool.leave(&mut tx).unwrap();
    fence(&ops);
    pool.preflight_prepared(&tx).unwrap();
    pool.commit_prepared(&mut tx, 41).unwrap();
    assert!(pool.commit_prepared(&mut tx, 41).is_err());
    assert!(pool.poll_install_ack(&tx, 42).is_err());
    assert_eq!(
        pool.poll_install_ack(&tx, 41).unwrap(),
        KvEndProgress::Complete
    );
    assert!(pool.clone().release(&[source, destination]).is_err());
    pool.publish_prepared(&tx).unwrap();
    pool.retire_prepared(&mut tx, 41, &[source]).unwrap();
    let free = pool.capacity().free_pages;
    pool.retire_prepared(&mut tx, 41, &[source]).unwrap();
    assert!(pool.retire_prepared(&mut tx, 41, &[]).is_err());
    assert!(pool.prepare(request(3, &[source], &[], &[], &[])).is_err());
    assert_eq!(pool.capacity().free_pages, free);
    assert_eq!(pool.finish_prepared(&mut Some(tx)).unwrap(), vec![source]);
    assert_eq!(pool.capacity().free_pages, free + 1);
    assert_eq!(committed_value(&ops, &mut pool, destination), 7.0);
    pool.preempt(&[destination]).unwrap();
    pool.restore(&[destination]).unwrap();
    assert_eq!(committed_value(&ops, &mut pool, destination), 7.0);
    pool.release(&[destination]).unwrap();
}

#[test]
#[ignore = "requires CUDA device and native providers"]
fn foreign_duplicate_handles_and_undeclared_resident_reads_are_rejected() {
    let ops = Rc::new(CudaOperators::new().unwrap());
    let other_ops = Rc::new(CudaOperators::new().unwrap());
    let mut a = pool(&ops);
    let mut b = pool(&other_ops);
    let page = KvPageId(9);
    seed(&ops, &mut a, KvPageId(1), 3.0);
    let mut ta = a.prepare(request(2, &[page], &[], &[], &[])).unwrap();
    let mut tb = b.prepare(request(2, &[page], &[], &[], &[])).unwrap();
    assert!(b.enter(&mut ta, batch(page)).is_err());
    assert!(b.preflight_prepared(&ta).is_err());
    assert!(
        a.prepare(request(2, &[KvPageId(7)], &[], &[], &[]))
            .is_err()
    );
    a.enter(&mut ta, batch(page)).unwrap();
    b.enter(&mut tb, batch(page)).unwrap();
    let mut view = a.active_view(&mut ta).unwrap();
    assert_eq!(view.physical_slot(KvPageId(1)), None);
    assert!(
        view.with_f32_gqa_planes(&other_ops, 1, &batch(page), |_| Ok(()))
            .is_err()
    );
    assert!(
        view.with_f32_gqa_planes(&ops, 2, &batch(page), |_| Ok(()))
            .is_err()
    );
    assert!(
        view.with_f32_gqa_planes(&ops, 1, &batch(KvPageId(1)), |_| Ok(()))
            .is_err()
    );
    a.leave(&mut ta).unwrap();
    b.leave(&mut tb).unwrap();
    fence(&ops);
    fence(&other_ops);
    let mut ta = Some(ta);
    assert!(b.abort(&mut ta).is_err());
    assert!(ta.is_some());
    a.abort(&mut ta).unwrap();
    b.abort(&mut Some(tb)).unwrap();
    let mut reused = a.prepare(request(2, &[page], &[], &[], &[])).unwrap();
    a.enter(&mut reused, batch(page)).unwrap();
    assert!(view.validate_batch(&batch(page)).is_err());
    a.leave(&mut reused).unwrap();
    fence(&ops);
    a.abort(&mut Some(reused)).unwrap();
    a.release(&[KvPageId(1)]).unwrap();
}

#[test]
#[ignore = "requires CUDA device and native providers"]
fn shadow_install_quarantines_the_old_slot_until_prepared_finish() {
    let ops = Rc::new(CudaOperators::new().unwrap());
    let mut pool = pool(&ops);
    let page = KvPageId(1);
    seed(&ops, &mut pool, page, 3.0);
    let old = pool.physical_slot(page);
    let mut tx = pool.prepare(request(2, &[], &[page], &[], &[])).unwrap();
    pool.enter(&mut tx, batch(page)).unwrap();
    write(&ops, &mut pool.active_view(&mut tx).unwrap(), page, 8.0);
    pool.leave(&mut tx).unwrap();
    fence(&ops);
    let free = pool.capacity().free_pages;
    pool.preflight_prepared(&tx).unwrap();
    pool.commit_prepared(&mut tx, 5).unwrap();
    assert_ne!(pool.physical_slot(page), old);
    assert_eq!(pool.capacity().free_pages, free);
    pool.publish_prepared(&tx).unwrap();
    pool.retire_prepared(&mut tx, 5, &[]).unwrap();
    assert!(pool.clone().release(&[page]).is_err());
    pool.finish_prepared(&mut Some(tx)).unwrap();
    assert_eq!(pool.capacity().free_pages, free + 1);
    assert_eq!(committed_value(&ops, &mut pool, page), 8.0);
    pool.release(&[page]).unwrap();
}
