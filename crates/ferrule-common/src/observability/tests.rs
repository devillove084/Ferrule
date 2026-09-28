use super::*;

#[test]
fn request_guard_finishes_once_on_explicit_finish_then_drop() {
    let metrics = Metrics::new();
    let mut guard = metrics.start_request();
    assert_eq!(metrics.snapshot().total_requests, 1);
    assert_eq!(metrics.snapshot().active_requests, 1);
    assert!(guard.finish());
    assert!(!guard.finish());
    drop(guard);
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.total_requests, 1);
    assert_eq!(snapshot.active_requests, 0);
    assert_eq!(snapshot.finished_requests, 1);
    assert_eq!(snapshot.max_running, 1);
}

#[test]
fn moved_guard_and_unwind_end_only_their_own_observation() {
    let metrics = Metrics::new();
    let first = metrics.start_request();
    let result = std::panic::catch_unwind(|| {
        let _second = metrics.start_request();
        panic!("injected request failure");
    });
    assert!(result.is_err());
    assert_eq!(metrics.snapshot().active_requests, 1);
    assert_eq!(metrics.snapshot().finished_requests, 1);
    let moved = first;
    drop(moved);
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.total_requests, 2);
    assert_eq!(snapshot.active_requests, 0);
    assert_eq!(snapshot.finished_requests, 2);
    assert_eq!(snapshot.max_running, 2);
}

#[test]
fn concurrent_guards_balance_at_quiescence() {
    let metrics = Metrics::default();
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                for _ in 0..100 {
                    let mut request = metrics.start_request();
                    request.finish();
                    request.finish();
                }
            });
        }
    });
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.total_requests, 800);
    assert_eq!(snapshot.finished_requests, 800);
    assert_eq!(snapshot.active_requests, 0);
}

#[test]
fn unmatched_legacy_finish_does_not_underflow_or_count_a_request() {
    let metrics = Metrics::default();
    metrics.request_finished();
    assert_eq!(metrics.snapshot().finished_requests, 0);
    assert_eq!(metrics.snapshot().active_requests, 0);
    metrics.request_started();
    metrics.request_finished();
    metrics.request_finished();
    assert_eq!(metrics.snapshot().finished_requests, 1);
    assert_eq!(metrics.snapshot().active_requests, 0);
}

#[test]
fn snapshot_and_dump_do_not_reset_or_mutate_accounting() {
    let metrics = Metrics::new();
    let request = metrics.start_request();
    metrics.prompt_tokens.fetch_add(10, Ordering::Relaxed);
    metrics.generated_tokens.fetch_add(3, Ordering::Relaxed);
    metrics.cache_hits.fetch_add(2, Ordering::Relaxed);
    metrics.cache_misses.fetch_add(1, Ordering::Relaxed);
    metrics.record_ttft(2_000);
    metrics.record_ttft(4_000);
    metrics.record_tpot(500);
    metrics.record_e2e_latency(8_000);
    metrics.record_queue_time(1_000);
    metrics.update_queue_depth(7);
    metrics.update_queue_depth(2);
    metrics.set_gpu_memory(1_048_576, 2_097_152);
    let before = metrics.snapshot();
    assert_eq!(before.avg_ttft_ms, 3.0);
    assert_eq!(before.avg_tpot_ms, 0.5);
    assert_eq!(before.avg_e2e_ms, 8.0);
    assert_eq!(before.avg_queue_ms, 1.0);
    assert_eq!(before.max_queue_depth, 7);
    assert_eq!(before.gpu_used_mb, 1.0);
    assert_eq!(before.gpu_total_mb, 2.0);
    assert_eq!(metrics.snapshot(), before);
    let old = Instant::now() - Duration::from_secs(2);
    *metrics.last_dump.lock().unwrap() = old;
    assert!(!metrics.maybe_dump_every(Duration::ZERO));
    assert_eq!(*metrics.last_dump.lock().unwrap(), old);
    assert!(metrics.maybe_dump_every(Duration::from_secs(1)));
    assert!(!metrics.maybe_dump_every(Duration::from_secs(60)));
    assert_eq!(metrics.snapshot(), before);
    shutdown();
    assert_eq!(metrics.snapshot(), before);
    drop(request);
    assert_eq!(metrics.snapshot().finished_requests, 1);
    assert_eq!(metrics.snapshot().active_requests, 0);
}

#[test]
fn optional_dump_skips_contended_or_poisoned_lock() {
    let metrics = Metrics::new();
    let before = metrics.snapshot();
    let lock = metrics.last_dump.lock().unwrap();
    assert!(!metrics.maybe_dump_every(Duration::from_nanos(1)));
    drop(lock);
    let _ = std::panic::catch_unwind(|| {
        let _lock = metrics.last_dump.lock().unwrap();
        panic!("poison only the optional dump throttle");
    });
    assert!(!metrics.maybe_dump_every(Duration::from_nanos(1)));
    assert_eq!(metrics.snapshot(), before);
    drop(metrics.start_request());
    assert_eq!(metrics.snapshot().finished_requests, 1);
}

#[test]
fn formatting_large_counters_cannot_overflow_the_optional_dump() {
    let snapshot = MetricsSnapshot {
        cache_hits: u64::MAX,
        cache_misses: u64::MAX,
        prefix_hits: u64::MAX,
        prefix_misses: u64::MAX,
        ..Default::default()
    };
    assert_eq!(snapshot.to_string().matches("50.0%").count(), 2);
}
