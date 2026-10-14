use std::sync::atomic::Ordering::SeqCst;
use std::sync::{Arc, Barrier, Mutex};

use crate::driver::{Driver, Guard};
use crate::runtime::{CudaContext, DeviceBuffer};
use tracing::{Event, Metadata, Subscriber, span};

#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<String>>>);
struct Fields(String);
impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write;
        write!(&mut self.0, "{}={value:?};", field.name()).unwrap();
    }
}
impl Subscriber for Logs {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
        span::Id::from_u64(1)
    }
    fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
    fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
    fn event(&self, event: &Event<'_>) {
        let mut fields = Fields(String::new());
        event.record(&mut fields);
        self.0.lock().unwrap().push(fields.0);
    }
    fn enter(&self, _: &span::Id) {}
    fn exit(&self, _: &span::Id) {}
}
struct Harness {
    driver: Arc<Driver>,
    logs: Logs,
    _driver: Guard,
    _tracing: tracing::subscriber::DefaultGuard,
}
impl Harness {
    fn new() -> Self {
        let driver = Arc::new(Driver::default());
        let guard = driver.enter();
        let logs = Logs::default();
        let tracing = tracing::subscriber::set_default(logs.clone());
        Self {
            driver,
            logs,
            _driver: guard,
            _tracing: tracing,
        }
    }
    fn retired_context(&self, record: bool) -> Arc<CudaContext> {
        let context = CudaContext::new(0).unwrap();
        let stream = context.default_stream();
        let buffer = unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, 256) }.unwrap();
        drop(buffer);
        if record {
            assert_eq!(context.allocator_metrics().pending_retirement_blocks, 1);
        }
        drop(stream);
        context
    }
    fn assert_retained(&self) -> String {
        assert_eq!(self.driver.retains.load(SeqCst), 1);
        assert_eq!(self.driver.releases.load(SeqCst), 0);
        assert_eq!(self.driver.frees.load(SeqCst), 0);
        let logs = self.logs.0.lock().unwrap();
        let drop_logs = logs
            .iter()
            .filter(|log| log.contains("CUDA context teardown quarantined"))
            .collect::<Vec<_>>();
        assert_eq!(
            drop_logs.len(),
            1,
            "one actionable Drop error, no silent retain"
        );
        let log = drop_logs[0].clone();
        for field in [
            "shutdown_retained=true",
            "reserved_bytes=67108864",
            "pending_blocks=1",
            "pending_bytes=256",
            "ordinal=0",
            "restart the worker",
        ] {
            assert!(log.contains(field), "missing {field}: {log}");
        }
        log
    }
}

#[test]
fn context_drop_quiescent_orders_sync_free_then_primary_release() {
    let h = Harness::new();
    let context = h.retired_context(false);
    drop(context);
    let calls = h.driver.calls.lock().unwrap();
    let index = |name| calls.iter().position(|call| *call == name).unwrap();
    assert!(index("context_sync") < index("free"));
    assert!(index("free") < index("release"));
    assert_eq!(h.driver.syncs.load(SeqCst), 1);
    assert_eq!(h.driver.frees.load(SeqCst), 1);
    assert_eq!(h.driver.releases.load(SeqCst), 1);
    assert!(h.logs.0.lock().unwrap().is_empty());
}

#[test]
fn context_drop_shutdown_sync_failure_retains_without_unwinding() {
    for code in [3, 700, 719, 999] {
        let h = Harness::new();
        let context = h.retired_context(false);
        h.driver.sync_code.store(code, SeqCst);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(context))).is_ok());
        assert_eq!(h.driver.syncs.load(SeqCst), 1);
        assert!(
            h.assert_retained()
                .contains("cuCtxSynchronizeAllocatorRetirement")
        );
    }
}

#[test]
fn context_drop_unknown_before_keeps_both_primary_retain_and_event_ledger() {
    let h = Harness::new();
    let context = h.retired_context(true);
    assert_eq!(h.driver.creates.load(SeqCst), 1);
    context.mark_graph_unknown();
    drop(context);
    h.assert_retained();
    assert_eq!(h.driver.syncs.load(SeqCst), 0);
    assert_eq!(
        h.driver.destroys.load(SeqCst),
        0,
        "allocator event ledger must remain owned"
    );
}

#[test]
fn context_drop_failed_shutdown_unknown_is_not_washed_by_later_success() {
    let h = Harness::new();
    let context = h.retired_context(false);
    h.driver.sync_code.store(999, SeqCst);
    context.shutdown_allocator();
    assert!(context.graph_completion_unknown());
    assert!(context.allocator_metrics().shutdown_retained);
    h.driver.sync_code.store(0, SeqCst);
    h.driver.ready.store(true, SeqCst);
    assert_eq!(context.allocator_metrics().pending_retirement_bytes, 256);
    drop(context);
    h.assert_retained();
    assert_eq!(
        h.driver.syncs.load(SeqCst),
        1,
        "Drop must not retry a terminal Unknown"
    );
}

#[test]
fn context_drop_query_failed_and_failed_fallback_is_terminal() {
    let h = Harness::new();
    let context = h.retired_context(true);
    h.driver.query_code.store(999, SeqCst);
    h.driver.event_sync_code.store(999, SeqCst);
    assert_eq!(context.allocator_metrics().pending_retirement_bytes, 256);
    assert!(context.graph_completion_unknown());
    h.driver.query_code.store(0, SeqCst);
    h.driver.event_sync_code.store(0, SeqCst);
    h.driver.ready.store(true, SeqCst);
    assert_eq!(context.allocator_metrics().pending_retirement_bytes, 256);
    drop(context);
    h.assert_retained();
    assert_eq!(h.driver.syncs.load(SeqCst), 0);
}

#[test]
fn context_drop_query_error_with_valid_existing_fallback_can_release() {
    let h = Harness::new();
    let context = h.retired_context(true);
    h.driver.query_code.store(999, SeqCst);
    assert_eq!(context.allocator_metrics().pending_retirement_bytes, 0);
    assert!(!context.graph_completion_unknown());
    drop(context);
    assert_eq!(h.driver.releases.load(SeqCst), 1);
    assert_eq!(h.driver.frees.load(SeqCst), 1);
    assert!(h.logs.0.lock().unwrap().is_empty());
}

#[test]
fn context_drop_unknown_arising_during_drain_defeats_successful_sync() {
    let h = Harness::new();
    let context = h.retired_context(false);
    let hook = (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)));
    *h.driver.sync_hook.lock().unwrap() = Some(hook.clone());
    let worker = context.clone();
    let driver = h.driver.clone();
    let shutdown = std::thread::spawn(move || {
        let _guard = driver.enter();
        worker.shutdown_allocator();
    });
    hook.0.wait();
    context.mark_graph_unknown();
    hook.1.wait();
    shutdown.join().unwrap();
    let metrics = context.allocator_metrics();
    assert!(metrics.shutdown_retained);
    assert!(!metrics.shutdown_quiescent);
    assert_eq!(metrics.pending_retirement_bytes, 256);
    drop(context);
    h.assert_retained();
}

#[test]
fn context_drop_concurrent_retirement_requires_a_new_complete_drain() {
    let h = Harness::new();
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let first = unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, 256) }.unwrap();
    let later = unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, 256) }.unwrap();
    drop(first);
    drop(stream);
    let hook = (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)));
    *h.driver.sync_hook.lock().unwrap() = Some(hook.clone());
    let worker = context.clone();
    let driver = h.driver.clone();
    let shutdown = std::thread::spawn(move || {
        let _guard = driver.enter();
        worker.shutdown_allocator();
    });
    hook.0.wait();
    drop(later);
    hook.1.wait();
    shutdown.join().unwrap();
    // Do not call allocator_metrics here: its poll would seal the late queue.
    // The stream registry is empty, so that poll would supply the old empty-set proof.
    assert_eq!(h.driver.frees.load(SeqCst), 0);
    assert_eq!(h.driver.releases.load(SeqCst), 0);
    assert!(
        !context.graph_completion_unknown(),
        "incomplete drain is not a driver failure"
    );
    drop(context);
    assert_eq!(h.driver.syncs.load(SeqCst), 2);
    assert_eq!(h.driver.frees.load(SeqCst), 1);
    assert_eq!(h.driver.releases.load(SeqCst), 1);
}

#[test]
fn context_drop_active_capture_preserves_capture_quarantine() {
    let h = Harness::new();
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let buffer = unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, 256) }.unwrap();
    stream.begin_capture().unwrap();
    drop(buffer);
    drop(stream);
    drop(context);
    assert!(h.assert_retained().contains("capture_bytes=256"));
    assert_eq!(h.driver.syncs.load(SeqCst), 0);
}

#[test]
fn context_drop_binding_failure_keeps_unfenced_resources() {
    let h = Harness::new();
    let context = h.retired_context(false);
    h.driver.fail_bind.store(true, SeqCst);
    drop(context);
    assert!(h.assert_retained().contains("cuCtxSetCurrent"));
    assert_eq!(h.driver.syncs.load(SeqCst), 0);
}

#[test]
fn context_drop_trim_failure_does_not_release_primary_context() {
    let h = Harness::new();
    let context = h.retired_context(false);
    h.driver.free_code.store(999, SeqCst);
    drop(context);
    assert_eq!(h.driver.releases.load(SeqCst), 0);
    assert_eq!(h.driver.frees.load(SeqCst), 1, "attempted but failed");
    let logs = h.logs.0.lock().unwrap();
    assert_eq!(logs.len(), 1);
    assert!(logs[0].contains("cuMemFreeAllocatorSegment"));
    assert!(logs[0].contains("reserved_bytes=67108864"));
}

#[test]
fn context_drop_primary_release_failure_is_reported_without_retry() {
    let h = Harness::new();
    let context = h.retired_context(false);
    h.driver.release_code.store(999, SeqCst);
    drop(context);
    assert_eq!(h.driver.releases.load(SeqCst), 1);
    assert_eq!(h.driver.frees.load(SeqCst), 1);
    let logs = h.logs.0.lock().unwrap();
    assert_eq!(logs.len(), 1);
    assert!(logs[0].contains("cuDevicePrimaryCtxRelease"));
    assert!(logs[0].contains("shutdown_retained=true"));
}

#[test]
fn context_drop_query_failure_is_logged_even_when_stream_owner_keeps_context_alive() {
    let h = Harness::new();
    let context = CudaContext::new(0).unwrap();
    let weak = Arc::downgrade(&context);
    let stream = context.default_stream();
    drop(unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, 256) }.unwrap());
    h.driver.query_code.store(999, SeqCst);
    h.driver.event_sync_code.store(999, SeqCst);
    assert_eq!(context.allocator_metrics().pending_retirement_bytes, 256);
    assert!(context.graph_completion_unknown());
    drop(stream);
    drop(context);
    assert!(
        weak.upgrade().is_some(),
        "preserve the existing Unknown stream owner-retain rule"
    );
    assert_eq!(h.driver.releases.load(SeqCst), 0);
    assert_eq!(h.driver.frees.load(SeqCst), 0);
    let logs = h.logs.0.lock().unwrap();
    assert_eq!(
        logs.len(),
        1,
        "fault must be observable even without a final Context Drop"
    );
    for field in [
        "cuEventQuery",
        "cuEventSynchronize",
        "retained_bytes=256",
        "restart worker",
    ] {
        assert!(logs[0].contains(field), "missing {field}: {}", logs[0]);
    }
}

#[test]
fn context_drop_record_fallback_failure_logs_before_stream_owner_quarantine() {
    let h = Harness::new();
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    drop(unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, 256) }.unwrap());
    h.driver.record_code.store(999, SeqCst);
    h.driver.sync_code.store(999, SeqCst);
    assert_eq!(context.allocator_metrics().pending_retirement_bytes, 256);
    assert!(context.graph_completion_unknown());
    drop(stream);
    drop(context);
    assert_eq!(h.driver.releases.load(SeqCst), 0);
    assert_eq!(h.driver.frees.load(SeqCst), 0);
    let logs = h.logs.0.lock().unwrap();
    assert_eq!(logs.len(), 1);
    for field in [
        "cuCtxSynchronizeAllocatorRetirement",
        "completion_unknown=true",
        "retained_bytes=256",
        "restart worker",
    ] {
        assert!(logs[0].contains(field), "missing {field}: {}", logs[0]);
    }
}
