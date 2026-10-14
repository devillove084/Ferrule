use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::Ordering::SeqCst;
use std::sync::{Arc, Barrier};

use crate::allocator::CudaDeviceBlock;
use crate::runtime::CudaContext;

fn blocks(context: &Arc<CudaContext>, count: usize, bytes: usize) -> Vec<Arc<CudaDeviceBlock>> {
    (0..count)
        .map(|_| context.allocator.allocate(bytes, 256).unwrap())
        .collect()
}
fn complete(context: &CudaContext) {
    context.driver.ready_through.store(usize::MAX, SeqCst);
}
fn no_sync(context: &CudaContext) {
    assert_eq!(context.driver.context_syncs.load(SeqCst), 0);
    assert_eq!(context.driver.event_syncs.load(SeqCst), 0);
}
fn barrier_hook() -> (Arc<Barrier>, Arc<Barrier>) {
    (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)))
}

#[test]
fn many_blocks_share_two_fences_and_cannot_reuse_before_completion() {
    let context = CudaContext::new(2);
    let live = blocks(&context, 8, 256);
    let old = live.iter().map(|block| block.ptr()).collect::<Vec<_>>();
    drop(live);
    let metrics = context.allocator.metrics();
    assert_eq!(metrics.retired_blocks, 8);
    assert_eq!(metrics.pending_retirement_blocks, 8);
    assert_eq!(metrics.pending_retirement_bytes, 2048);
    assert_eq!(metrics.pending_unfenced_bytes, 2048);
    assert_eq!(context.driver.records.load(SeqCst), 0);
    let other = context.allocator.allocate(256, 256).unwrap();
    assert!(!old.contains(&other.ptr()));
    assert_eq!(context.driver.records.load(SeqCst), 2);
    assert_eq!(context.driver.creates.load(SeqCst), 2);
    assert_eq!(context.allocator.metrics().retirement_created_fences, 2);
    assert_eq!(context.allocator.metrics().retirement_batches, 1);
    assert_eq!(context.allocator.metrics().pending_unfenced_bytes, 0);
    complete(&context);
    context.driver.pending_streams.store(2, SeqCst);
    assert_eq!(
        context.allocator.poll_retirements(),
        0,
        "both streams must complete"
    );
    context.driver.pending_streams.store(0, SeqCst);
    assert_eq!(context.allocator.poll_retirements(), 8);
    assert_eq!(context.driver.destroys.load(SeqCst), 2);
    assert_eq!(context.allocator.metrics().pending_retirement_bytes, 0);
    assert_eq!(context.allocator.allocate(256, 256).unwrap().ptr(), old[0]);
    no_sync(&context);
}

#[test]
fn partial_record_failure_retains_entire_batch_and_accounting() {
    let context = CudaContext::new(2);
    drop(blocks(&context, 3, 256));
    context.driver.fail_record_at.store(2, SeqCst);
    context.driver.fail_sync.store(true, SeqCst);
    assert_eq!(context.allocator.poll_retirements(), 0);
    let metrics = context.allocator.metrics();
    assert_eq!(metrics.pending_retirement_blocks, 3);
    assert_eq!(metrics.pending_retirement_bytes, 768);
    assert_eq!(metrics.pending_unfenced_bytes, 768);
    assert_eq!(
        metrics.retirement_created_fences, 0,
        "partial runtime creations are not returned"
    );
    assert_eq!(context.driver.creates.load(SeqCst), 2);
    assert_eq!(context.driver.destroys.load(SeqCst), 2);
    context.driver.fail_sync.store(false, SeqCst);
    complete(&context);
    assert_eq!(
        context.allocator.poll_retirements(),
        0,
        "not a new completion proof"
    );
    context.allocator.shutdown();
    assert_eq!(context.allocator.metrics().pending_retirement_bytes, 768);
    assert!(context.allocator.metrics().shutdown_retained);
}

#[test]
fn old_record_and_query_completion_fallbacks_are_preserved() {
    for record_failure in [false, true] {
        let context = CudaContext::new(2);
        drop(blocks(&context, 3, 256));
        if record_failure {
            context.driver.fail_record_at.store(2, SeqCst);
        } else {
            context.driver.fail_query.store(true, SeqCst);
        }
        assert_eq!(context.allocator.poll_retirements(), 3);
        assert_eq!(context.allocator.metrics().pending_retirement_bytes, 0);
        assert_eq!(
            context.driver.context_syncs.load(SeqCst),
            usize::from(record_failure)
        );
        assert_eq!(
            context.driver.event_syncs.load(SeqCst),
            if record_failure { 0 } else { 2 }
        );
    }
}

#[test]
fn query_and_sync_failure_is_terminal_unknown() {
    let context = CudaContext::new(2);
    drop(blocks(&context, 4, 256));
    context.driver.fail_query.store(true, SeqCst);
    context.driver.fail_sync.store(true, SeqCst);
    assert_eq!(context.allocator.poll_retirements(), 0);
    assert_eq!(context.allocator.metrics().pending_retirement_bytes, 1024);
    context.driver.fail_query.store(false, SeqCst);
    context.driver.fail_sync.store(false, SeqCst);
    complete(&context);
    assert_eq!(context.allocator.poll_retirements(), 0);
    context.capture.lock().unwrap().unknown = true;
    context.allocator.shutdown();
    assert_eq!(context.allocator.metrics().pending_retirement_blocks, 4);
    assert_eq!(context.driver.frees.load(SeqCst), 0);
}

#[test]
fn panic_in_record_query_or_sync_never_loses_the_batch() {
    for operation in 0..3 {
        let context = CudaContext::new(2);
        drop(blocks(&context, 5, 256));
        match operation {
            0 => context.driver.panic_record.store(true, SeqCst),
            1 => context.driver.panic_query.store(true, SeqCst),
            _ => {
                context.driver.fail_record_at.store(1, SeqCst);
                context.driver.panic_sync.store(true, SeqCst);
            }
        }
        assert!(catch_unwind(AssertUnwindSafe(|| context.allocator.poll_retirements())).is_err());
        assert_eq!(context.allocator.metrics().pending_retirement_blocks, 5);
        assert_eq!(context.allocator.metrics().pending_unfenced_bytes, 1280);
        context.driver.panic_record.store(false, SeqCst);
        context.driver.panic_query.store(false, SeqCst);
        context.driver.panic_sync.store(false, SeqCst);
        complete(&context);
        assert_eq!(context.allocator.poll_retirements(), 0);
        context.allocator.shutdown();
        assert_eq!(context.allocator.metrics().pending_retirement_bytes, 0);
    }
}

#[test]
fn concurrent_last_owner_drop_during_record_belongs_to_next_batch() {
    let context = CudaContext::new(2);
    let mut live = blocks(&context, 2, 256);
    drop(live.pop());
    let hook = barrier_hook();
    *context.driver.record_hook.lock().unwrap() = Some(hook.clone());
    let worker = context.clone();
    let poll = std::thread::spawn(move || worker.allocator.poll_retirements());
    hook.0.wait();
    drop(live);
    let metrics = context.allocator.metrics();
    assert_eq!(metrics.pending_retirement_blocks, 2);
    assert_eq!(metrics.pending_unfenced_bytes, 512);
    assert_eq!(metrics.retirement_batches, 1);
    hook.1.wait();
    assert_eq!(poll.join().unwrap(), 0);
    context.driver.ready_through.store(1, SeqCst);
    assert_eq!(context.allocator.poll_retirements(), 1);
    assert_eq!(context.allocator.metrics().pending_retirement_blocks, 1);
    assert_eq!(context.driver.records.load(SeqCst), 4);
    complete(&context);
    assert_eq!(context.allocator.poll_retirements(), 1);
    no_sync(&context);
}

#[test]
fn reentrant_poll_and_metrics_do_not_hold_allocator_lock_across_driver() {
    let context = CudaContext::new(2);
    drop(blocks(&context, 4, 256));
    *context.driver.reenter.lock().unwrap() = Some(Arc::downgrade(&context.allocator));
    assert_eq!(context.allocator.poll_retirements(), 0);
    assert_eq!(context.driver.records.load(SeqCst), 2);
    no_sync(&context);
}

#[test]
fn count_pressure_seals_without_new_synchronization() {
    let context = CudaContext::new(2);
    drop(blocks(&context, 65, 256));
    let metrics = context.allocator.metrics();
    assert_eq!(metrics.retirement_batches, 1);
    assert_eq!(metrics.pending_retirement_blocks, 65);
    assert_eq!(metrics.pending_unfenced_bytes, 256);
    assert_eq!(context.driver.records.load(SeqCst), 2);
    no_sync(&context);
    complete(&context);
    assert_eq!(context.allocator.poll_retirements(), 65);
    let metrics = context.allocator.metrics();
    assert_eq!(metrics.retirement_batches, 2);
    assert_eq!(metrics.pending_retirement_blocks, 0);
    assert_eq!(metrics.pending_retirement_bytes, 0);
    assert_eq!(metrics.pending_unfenced_bytes, 0);
    assert_eq!(context.allocator.trim().unwrap(), 64 << 20);
    no_sync(&context);
}

#[test]
fn byte_pressure_and_oversize_block_seal_bounded_batches() {
    let context = CudaContext::new(2);
    let huge = context.allocator.allocate(66 << 20, 256).unwrap();
    drop(blocks(&context, 4, 16 << 20));
    assert_eq!(context.allocator.metrics().retirement_batches, 1);
    let small = context.allocator.allocate(256, 256).unwrap();
    drop(small);
    drop(huge);
    let metrics = context.allocator.metrics();
    assert_eq!(
        metrics.retirement_batches, 3,
        "oversized allocation gets its own batch"
    );
    assert_eq!(metrics.pending_retirement_blocks, 6);
    assert_eq!(metrics.pending_retirement_bytes, (130 << 20) + 256);
    assert_eq!(context.driver.records.load(SeqCst), 6);
    no_sync(&context);
}

#[test]
fn pending_bytes_admission_stops_growth_but_never_discards_prior_live_owners() {
    let context = CudaContext::new(2);
    let late = context.allocator.allocate(256, 256).unwrap();
    drop(blocks(&context, 4, 64 << 20));
    let reused = context.allocator.allocate(256, 256).unwrap();
    assert_eq!(reused.requested_bytes(), 256);
    assert!(context.allocator.allocate_dedicated(256, 256).is_err());
    drop(late);
    assert_eq!(
        context.allocator.metrics().pending_retirement_bytes,
        (256 << 20) + 256
    );
    no_sync(&context);
    complete(&context);
    assert!(context.allocator.allocate(256, 256).is_ok());
}

#[test]
fn pending_batch_pressure_only_defers_driver_growth_without_sync() {
    let context = CudaContext::new(2);
    for _ in 0..256 {
        drop(context.allocator.allocate(256, 256).unwrap());
        context.allocator.poll_retirements();
    }
    let before = context.allocator.metrics();
    assert_eq!(before.pending_retirement_batches, 256);
    assert_eq!(before.pending_retirement_bytes, 65536);
    let reused = context.allocator.allocate(256, 256).unwrap();
    assert_eq!(reused.requested_bytes(), 256);
    assert!(context.allocator.allocate_dedicated(256, 256).is_err());
    assert_eq!(
        context.allocator.metrics().reserved_bytes,
        before.reserved_bytes
    );
    no_sync(&context);
}

#[test]
fn dedicated_owner_aliases_retire_only_on_final_drop() {
    let context = CudaContext::new(2);
    let root = context.allocator.allocate_dedicated(4096, 256).unwrap();
    let view = root.clone();
    let last_view = view.clone();
    drop(root);
    drop(view);
    assert_eq!(context.allocator.metrics().retired_blocks, 0);
    assert_eq!(context.allocator.metrics().live_granted_bytes, 4096);
    drop(last_view);
    assert_eq!(context.allocator.metrics().retired_blocks, 1);
    assert_eq!(context.allocator.metrics().pending_unfenced_bytes, 4096);
    assert_eq!(context.allocator.trim().unwrap(), 0);
    complete(&context);
    assert_eq!(context.allocator.trim().unwrap(), 4096);
    no_sync(&context);
}

#[test]
fn shutdown_sync_only_releases_its_preexisting_snapshot() {
    let context = CudaContext::new(2);
    let mut live = blocks(&context, 2, 256);
    drop(live.pop());
    let hook = barrier_hook();
    *context.driver.sync_hook.lock().unwrap() = Some(hook.clone());
    let worker = context.clone();
    let shutdown = std::thread::spawn(move || worker.allocator.shutdown());
    hook.0.wait();
    drop(live);
    hook.1.wait();
    shutdown.join().unwrap();
    let metrics = context.allocator.metrics();
    assert_eq!(metrics.pending_retirement_blocks, 1);
    assert_eq!(metrics.pending_unfenced_bytes, 256);
    assert_eq!(context.driver.frees.load(SeqCst), 0);
    context.allocator.shutdown();
    assert_eq!(context.allocator.metrics().pending_retirement_bytes, 0);
    assert_eq!(context.driver.frees.load(SeqCst), 1);
}

#[test]
fn context_drop_flushes_unfenced_without_per_batch_sync_or_early_free() {
    let context = CudaContext::new(2);
    let driver = context.driver.clone();
    drop(blocks(&context, 5, 256));
    assert_eq!(driver.frees.load(SeqCst), 0);
    assert_eq!(driver.context_syncs.load(SeqCst), 0);
    drop(context);
    assert_eq!(driver.context_syncs.load(SeqCst), 1);
    assert_eq!(driver.records.load(SeqCst), 0);
    assert_eq!(driver.frees.load(SeqCst), 1);
}

#[test]
fn failed_or_panicking_shutdown_keeps_all_blocks_and_reserved_bytes() {
    for panic in [false, true] {
        let context = CudaContext::new(2);
        drop(blocks(&context, 5, 256));
        context.driver.fail_sync.store(true, SeqCst);
        context.driver.panic_sync.store(panic, SeqCst);
        let result = catch_unwind(AssertUnwindSafe(|| context.allocator.shutdown()));
        assert_eq!(result.is_err(), panic);
        assert_eq!(context.allocator.metrics().pending_retirement_bytes, 1280);
        assert_eq!(context.allocator.metrics().reserved_bytes, 64 << 20);
        assert_eq!(context.driver.frees.load(SeqCst), 0);
        context.driver.panic_sync.store(false, SeqCst);
        drop(context);
    }
}

#[test]
fn capture_retirements_stay_separate_until_final_teardown() {
    let context = CudaContext::new(2);
    let mut live = blocks(&context, 2, 256);
    drop(live.pop());
    context.capture.lock().unwrap().active = true;
    drop(live);
    assert_eq!(context.allocator.poll_retirements(), 0);
    assert_eq!(context.driver.records.load(SeqCst), 0);
    assert_eq!(context.allocator.metrics().pending_retirement_bytes, 512);
    assert_eq!(context.allocator.metrics().capture_retirement_bytes, 256);
    context.capture.lock().unwrap().active = false;
    context.allocator.shutdown();
    assert_eq!(context.allocator.metrics().pending_retirement_bytes, 256);
    assert_eq!(context.allocator.metrics().capture_retirement_bytes, 256);
    let driver = context.driver.clone();
    drop(context);
    assert_eq!(driver.frees.load(SeqCst), 1);
}

#[test]
fn unknown_prevents_even_successful_recorded_fences_from_releasing_memory() {
    let context = CudaContext::new(2);
    drop(blocks(&context, 4, 256));
    context.allocator.poll_retirements();
    complete(&context);
    context.capture.lock().unwrap().unknown = true;
    assert_eq!(context.allocator.poll_retirements(), 0);
    context.allocator.shutdown();
    let metrics = context.allocator.metrics();
    assert_eq!(metrics.pending_retirement_blocks, 4);
    assert_eq!(metrics.pending_retirement_bytes, 1024);
    assert_eq!(context.driver.frees.load(SeqCst), 0);
}

#[test]
fn successful_stream_teardown_before_seal_keeps_old_empty_registry_contract() {
    let context = CudaContext::new(0);
    let stream = crate::runtime::MockStream::new(&context);
    drop(blocks(&context, 3, 256));
    drop(stream);
    assert_eq!(context.allocator.poll_retirements(), 3);
    assert_eq!(context.allocator.metrics().pending_retirement_bytes, 0);
    assert_eq!(context.driver.creates.load(SeqCst), 0);
    no_sync(&context);
}

#[test]
fn unknown_stream_teardown_before_seal_retains_unfenced_batch() {
    let context = CudaContext::new(0);
    let stream = crate::runtime::MockStream::new(&context);
    drop(blocks(&context, 3, 256));
    context.driver.fail_sync.store(true, SeqCst);
    drop(stream);
    assert_eq!(context.allocator.poll_retirements(), 0);
    let metrics = context.allocator.metrics();
    assert_eq!(metrics.pending_retirement_bytes, 768);
    assert_eq!(metrics.pending_unfenced_bytes, 768);
    assert_eq!(context.driver.frees.load(SeqCst), 0);
}

#[test]
fn zero_sized_owner_never_creates_a_retirement_or_fence() {
    let context = CudaContext::new(2);
    drop(context.allocator.allocate(0, 256).unwrap());
    assert_eq!(context.allocator.poll_retirements(), 0);
    assert_eq!(context.allocator.metrics().retired_blocks, 0);
    assert_eq!(context.allocator.metrics().retirement_batches, 0);
    assert_eq!(context.driver.creates.load(SeqCst), 0);
    no_sync(&context);
}

#[test]
fn unknown_during_record_cannot_be_overridden_by_local_event_success() {
    let context = CudaContext::new(2);
    drop(blocks(&context, 3, 256));
    complete(&context);
    let hook = barrier_hook();
    *context.driver.record_hook.lock().unwrap() = Some(hook.clone());
    let worker = context.clone();
    let poll = std::thread::spawn(move || worker.allocator.poll_retirements());
    hook.0.wait();
    context.capture.lock().unwrap().unknown = true;
    hook.1.wait();
    assert_eq!(poll.join().unwrap(), 0);
    assert_eq!(context.allocator.metrics().pending_retirement_bytes, 768);
    assert_eq!(context.allocator.metrics().pending_unfenced_bytes, 768);
    assert_eq!(context.driver.frees.load(SeqCst), 0);
}

#[test]
fn unknown_during_shutdown_sync_preserves_its_entire_snapshot() {
    let context = CudaContext::new(2);
    drop(blocks(&context, 3, 256));
    let hook = barrier_hook();
    *context.driver.sync_hook.lock().unwrap() = Some(hook.clone());
    let worker = context.clone();
    let shutdown = std::thread::spawn(move || worker.allocator.shutdown());
    hook.0.wait();
    context.capture.lock().unwrap().unknown = true;
    hook.1.wait();
    shutdown.join().unwrap();
    assert_eq!(context.allocator.metrics().pending_retirement_bytes, 768);
    assert_eq!(context.driver.frees.load(SeqCst), 0);
}
