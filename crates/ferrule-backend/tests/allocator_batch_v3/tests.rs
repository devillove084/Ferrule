use crate::driver::Driver;
use crate::runtime::{CudaContext, CudaStream, DeviceBuffer};
use std::sync::atomic::Ordering::SeqCst;
use std::sync::{Arc, Barrier};

fn buffer(stream: &CudaStream, bytes: usize) -> DeviceBuffer<u8> {
    unsafe { DeviceBuffer::uninitialized_async(stream, bytes) }.unwrap()
}
fn hook() -> (Arc<Barrier>, Arc<Barrier>) {
    (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)))
}
fn calls(driver: &Driver, name: &str) -> usize {
    driver
        .calls
        .lock()
        .unwrap()
        .iter()
        .filter(|call| **call == name)
        .count()
}

#[test]
fn v3_weak_gone_before_stream_sync_completion_defers_empty_fences() {
    let driver = Arc::new(Driver::default());
    let _guard = driver.enter();
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let weak = Arc::downgrade(&stream);
    drop(buffer(&stream, 256));
    let barrier = hook();
    *driver.stream_sync_hook.lock().unwrap() = Some(barrier.clone());
    let worker_driver = driver.clone();
    let teardown = std::thread::spawn(move || {
        let _guard = worker_driver.enter();
        drop(stream);
    });
    barrier.0.wait();
    assert!(
        weak.upgrade().is_none(),
        "real Arc destruction window, not a fake stream counter"
    );
    for _ in 0..3 {
        let metrics = context.allocator_metrics();
        assert_eq!(metrics.pending_retirement_bytes, 256);
        assert_eq!(metrics.pending_unfenced_bytes, 256);
        assert_eq!(metrics.free_bytes, (64 << 20) - 256);
    }
    assert_eq!(driver.creates.load(SeqCst), 0);
    assert_eq!(
        driver.syncs.load(SeqCst),
        0,
        "deferred teardown is not a fallback sync"
    );
    assert_eq!(driver.frees.load(SeqCst), 0);
    assert_eq!(calls(&driver, "stream_destroy"), 0);
    barrier.1.wait();
    teardown.join().unwrap();
    assert_eq!(calls(&driver, "stream_destroy"), 1);
    let metrics = context.allocator_metrics();
    assert_eq!(metrics.pending_retirement_bytes, 0);
    assert_eq!(metrics.free_bytes, 64 << 20);
    assert_eq!(
        driver.creates.load(SeqCst),
        0,
        "empty set valid only after ticket completion"
    );
    assert_eq!(driver.syncs.load(SeqCst), 0);
    drop(context);
    assert_eq!(
        driver.releases.load(SeqCst),
        1,
        "no strong custody cycle remains"
    );
}

#[test]
fn v3_registration_during_weak_gone_window_must_not_prune_ticket_or_reuse() {
    let driver = Arc::new(Driver::default());
    let _guard = driver.enter();
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let first = buffer(&stream, 256);
    let original = first.cu_deviceptr();
    drop(first);
    let barrier = hook();
    *driver.stream_sync_hook.lock().unwrap() = Some(barrier.clone());
    let worker_driver = driver.clone();
    let teardown = std::thread::spawn(move || {
        let _guard = worker_driver.enter();
        drop(stream);
    });
    barrier.0.wait();
    let other = context.default_stream();
    let live = buffer(&other, 256);
    assert_ne!(live.cu_deviceptr(), original);
    assert_eq!(context.allocator_metrics().pending_retirement_bytes, 256);
    assert_eq!(driver.creates.load(SeqCst), 0);
    assert_eq!(driver.syncs.load(SeqCst), 0);
    barrier.1.wait();
    teardown.join().unwrap();
    driver.ready.store(true, SeqCst);
    assert_eq!(context.allocator_metrics().pending_retirement_bytes, 0);
    assert_eq!(
        driver.creates.load(SeqCst),
        1,
        "one remaining live registration"
    );
    assert_eq!(buffer(&other, 256).cu_deviceptr(), original);
    assert_eq!(driver.syncs.load(SeqCst), 0);
}

#[test]
fn v3_failed_stream_teardown_ticket_is_terminal_not_empty_coverage() {
    let driver = Arc::new(Driver::default());
    let _guard = driver.enter();
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    drop(buffer(&stream, 256));
    let barrier = hook();
    *driver.stream_sync_hook.lock().unwrap() = Some(barrier.clone());
    driver.stream_sync_code.store(999, SeqCst);
    let worker_driver = driver.clone();
    let teardown = std::thread::spawn(move || {
        let _guard = worker_driver.enter();
        drop(stream);
    });
    barrier.0.wait();
    assert_eq!(context.allocator_metrics().pending_retirement_bytes, 256);
    barrier.1.wait();
    teardown.join().unwrap();
    assert!(context.graph_completion_unknown());
    driver.stream_sync_code.store(0, SeqCst);
    driver.ready.store(true, SeqCst);
    assert_eq!(context.allocator_metrics().pending_retirement_bytes, 256);
    drop(context);
    assert_eq!(driver.creates.load(SeqCst), 0);
    assert_eq!(driver.frees.load(SeqCst), 0);
    assert_eq!(driver.releases.load(SeqCst), 0);
}

fn capture_during_query(complete: bool) {
    let driver = Arc::new(Driver::default());
    let _guard = driver.enter();
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    drop(buffer(&stream, 256));
    driver.ready.store(complete, SeqCst);
    let barrier = hook();
    *driver.query_hook.lock().unwrap() = Some(barrier.clone());
    let worker_context = context.clone();
    let worker_driver = driver.clone();
    let poll = std::thread::spawn(move || {
        let _guard = worker_driver.enter();
        worker_context.allocator_metrics()
    });
    barrier.0.wait();
    stream.begin_capture().unwrap();
    barrier.1.wait();
    let metrics = poll.join().unwrap();
    assert_eq!(metrics.pending_retirement_bytes, 256);
    assert_eq!(
        metrics.pending_unfenced_bytes, 0,
        "capture must preserve valid proof"
    );
    assert_eq!(driver.creates.load(SeqCst), 1);
    assert_eq!(driver.destroys.load(SeqCst), usize::from(complete));
    assert_eq!(context.allocator_metrics().pending_retirement_bytes, 256);
    drop(stream.end_capture().unwrap());
    driver.ready.store(true, SeqCst);
    assert_eq!(context.allocator_metrics().pending_retirement_bytes, 0);
    assert_eq!(
        driver.creates.load(SeqCst),
        1,
        "do not re-record existing valid proof"
    );
    assert_eq!(calls(&driver, "event_query"), if complete { 1 } else { 2 });
    assert_eq!(driver.syncs.load(SeqCst), 0);
}

#[test]
fn v3_temporary_capture_preserves_recorded_fences() {
    capture_during_query(false);
}

#[test]
fn v3_temporary_capture_preserves_complete_proof_until_publication() {
    capture_during_query(true);
}

#[test]
fn v3_unknown_during_query_remains_terminal_unlike_capture() {
    let driver = Arc::new(Driver::default());
    let _guard = driver.enter();
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    drop(buffer(&stream, 256));
    driver.ready.store(true, SeqCst);
    let barrier = hook();
    *driver.query_hook.lock().unwrap() = Some(barrier.clone());
    let worker_context = context.clone();
    let worker_driver = driver.clone();
    let poll = std::thread::spawn(move || {
        let _guard = worker_driver.enter();
        worker_context.allocator_metrics()
    });
    barrier.0.wait();
    context.mark_graph_unknown();
    barrier.1.wait();
    let metrics = poll.join().unwrap();
    assert_eq!(metrics.pending_retirement_bytes, 256);
    assert_eq!(metrics.pending_unfenced_bytes, 256);
    assert_eq!(context.allocator_metrics().pending_retirement_bytes, 256);
    let error = unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, 256) }.unwrap_err();
    assert!(error.to_string().contains("completion Unknown"));
    assert!(error.to_string().contains("restart worker"));
    assert_eq!(driver.frees.load(SeqCst), 0);
}

#[test]
fn v3_count_pressure_reuses_safe_extents_including_admitted_22_gib() {
    let driver = Arc::new(Driver::default());
    let _guard = driver.enter();
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    // Only the mock address space grows: no host or GPU payload is allocated.
    let admitted = buffer(&stream, 22usize << 30);
    let original = admitted.cu_deviceptr();
    driver.ready.store(true, SeqCst);
    drop(admitted);
    driver.ready.store(false, SeqCst);
    for _ in 0..256 {
        drop(buffer(&stream, 256));
        context.allocator_metrics();
    }
    let before = context.allocator_metrics();
    assert_eq!(before.pending_retirement_batches, 256);
    assert_eq!(before.pending_retirement_bytes, 64 << 10);
    let small = buffer(&stream, 256);
    let admitted = buffer(&stream, 22usize << 30);
    assert_eq!(admitted.cu_deviceptr(), original);
    assert_eq!(
        context.allocator_metrics().driver_allocations,
        before.driver_allocations
    );
    assert_eq!(
        context.allocator_metrics().reserved_bytes,
        before.reserved_bytes
    );
    let error = DeviceBuffer::<u8>::uninitialized_dedicated(&stream, 256).unwrap_err();
    assert!(error.to_string().contains("driver growth deferred"));
    assert!(error.to_string().contains("ordinary retirement pressure"));
    assert!(error.to_string().contains("not a device memory budget"));
    assert_eq!(driver.syncs.load(SeqCst), 0);
    driver.ready.store(true, SeqCst);
    let grown = DeviceBuffer::<u8>::uninitialized_dedicated(&stream, 256).unwrap();
    assert_eq!(
        context.allocator_metrics().driver_allocations,
        before.driver_allocations + 1
    );
    drop((small, admitted, grown));
}

#[test]
fn v3_byte_pressure_defers_only_growth_not_standard_reuse() {
    let driver = Arc::new(Driver::default());
    let _guard = driver.enter();
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let live = buffer(&stream, 256);
    let large = (0..4)
        .map(|_| buffer(&stream, 64 << 20))
        .collect::<Vec<_>>();
    drop(large);
    let before = context.allocator_metrics();
    assert_eq!(before.pending_retirement_bytes, 256 << 20);
    let reused = buffer(&stream, 256);
    assert_ne!(reused.cu_deviceptr(), live.cu_deviceptr());
    assert_eq!(
        context.allocator_metrics().driver_allocations,
        before.driver_allocations
    );
    assert!(DeviceBuffer::<u8>::uninitialized_dedicated(&stream, 256).is_err());
    assert_eq!(driver.syncs.load(SeqCst), 0);
}

#[test]
fn v3_capture_quarantine_is_not_wait_progress_growth_pressure() {
    let driver = Arc::new(Driver::default());
    let _guard = driver.enter();
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let captured = (0..4)
        .map(|_| buffer(&stream, 64 << 20))
        .collect::<Vec<_>>();
    stream.begin_capture().unwrap();
    drop(captured);
    let error = DeviceBuffer::<u8>::uninitialized_dedicated(&stream, 256).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("capture active; retry after capture ends")
    );
    let graph = stream.end_capture().unwrap();
    let before = context.allocator_metrics();
    assert_eq!(before.capture_retirement_bytes, 256 << 20);
    assert_eq!(before.pending_retirement_batches, 0);
    let grown = DeviceBuffer::<u8>::uninitialized_dedicated(&stream, 256).unwrap();
    let after = context.allocator_metrics();
    assert_eq!(after.driver_allocations, before.driver_allocations + 1);
    assert_eq!(
        after.capture_retirement_bytes,
        256 << 20,
        "poll cannot clear graph quarantine"
    );
    assert_eq!(driver.frees.load(SeqCst), 0);
    assert_eq!(driver.syncs.load(SeqCst), 0);
    drop(graph);
    drop(grown);
    drop(stream);
    drop(context);
    assert_eq!(driver.releases.load(SeqCst), 1);
}

#[test]
fn v3_multiple_blocks_still_record_per_registration_not_per_block() {
    let driver = Arc::new(Driver::default());
    let _guard = driver.enter();
    let context = CudaContext::new(0).unwrap();
    let first = context.new_stream().unwrap();
    let second = context.new_stream().unwrap();
    assert_ne!(first.cu_stream(), second.cu_stream());
    let live = (0..8).map(|_| buffer(&first, 256)).collect::<Vec<_>>();
    drop(live);
    let metrics = context.allocator_metrics();
    assert_eq!(metrics.pending_retirement_blocks, 8);
    assert_eq!(driver.creates.load(SeqCst), 2);
    assert_eq!(calls(&driver, "event_record"), 2);
    assert_eq!(driver.syncs.load(SeqCst), 0);
    driver.ready.store(true, SeqCst);
    assert_eq!(context.allocator_metrics().pending_retirement_bytes, 0);
    drop(first);
    drop(second);
    let weak_context = Arc::downgrade(&context);
    drop(context);
    assert!(weak_context.upgrade().is_none());
    assert_eq!(driver.releases.load(SeqCst), 1);
}
