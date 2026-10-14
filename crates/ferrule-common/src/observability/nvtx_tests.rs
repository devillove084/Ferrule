use super::*;
use std::collections::BTreeSet;
use std::io;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::layer::SubscriberExt;

#[derive(Default)]
struct Calls {
    next: AtomicU64,
    starts: Mutex<Vec<(u64, String)>>,
    ends: Mutex<Vec<u64>>,
}

#[derive(Clone, Default)]
struct Callback(Arc<Calls>);

impl RangeBackend for Callback {
    fn start(&self, label: &CStr) -> u64 {
        let id = self.0.next.fetch_add(1, Ordering::Relaxed) + 1;
        self.0
            .starts
            .lock()
            .unwrap()
            .push((id, label.to_str().unwrap().into()));
        id
    }

    fn end(&self, range: u64) {
        self.0.ends.lock().unwrap().push(range);
    }
}

impl Callback {
    fn layer(&self, enabled: bool) -> NvtxLayer<Self> {
        NvtxLayer {
            backend: self.clone(),
            enabled,
        }
    }

    fn labels(&self) -> Vec<String> {
        self.0
            .starts
            .lock()
            .unwrap()
            .iter()
            .map(|(_, label)| label.clone())
            .collect()
    }

    fn assert_balanced(&self, count: usize) {
        let starts = self.0.starts.lock().unwrap();
        let ends = self.0.ends.lock().unwrap();
        assert_eq!(starts.len(), count);
        assert_eq!(ends.len(), count);
        let ids: BTreeSet<_> = starts.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids.len(), count);
        assert_eq!(ids, ends.iter().copied().collect());
    }
}

fn dispatch(callback: &Callback, enabled: bool) -> tracing::Dispatch {
    tracing::Dispatch::new(
        tracing_subscriber::registry().with(callback.layer(enabled).with_filter(filter_fn(allows))),
    )
}

#[test]
fn runtime_opt_in_requires_exactly_one() {
    assert!(enabled_value(Some("1")));
    for value in [
        None,
        Some(""),
        Some("0"),
        Some("true"),
        Some(" 1"),
        Some("1 "),
    ] {
        assert!(!enabled_value(value));
    }
}

#[test]
fn only_allowlisted_spans_on_the_exact_target_emit_callbacks() {
    let callback = Callback::default();
    tracing::dispatcher::with_default(&dispatch(&callback, true), || {
        let _a = tracing::debug_span!(target: "ferrule_profile", "read_subset").entered();
        let _b = tracing::debug_span!(target: "ferrule_profile", "layer.expert").entered();
        let _c = tracing::debug_span!(target: "ferrule_profile", "layer.expert.upload").entered();
        let _wrong_target =
            tracing::debug_span!(target: "ferrule_profile_extra", "read_subset").entered();
        let _wrong_name = tracing::debug_span!(target: "ferrule_profile", "prompt").entered();
        let _log = tracing::debug_span!(target: "ferrule_performance", "layer.expert").entered();
        tracing::debug!(target: "ferrule_profile", prompt = "private event");
    });
    assert_eq!(
        callback.labels(),
        ["read_subset", "layer.expert", "layer.expert.upload"]
    );
    callback.assert_balanced(3);
}

#[test]
fn numeric_fields_are_bounded_and_record_updates_apply_to_next_enter() {
    let callback = Callback::default();
    tracing::dispatcher::with_default(&dispatch(&callback, true), || {
        let span = tracing::debug_span!(target: "ferrule_profile", "read_subset",
            operation = u64::MAX, source_generation = 7u64, requested_bytes = tracing::field::Empty);
        span.in_scope(|| {});
        span.record("requested_bytes", 4096u64);
        span.in_scope(|| {});
        let _range = tracing::debug_span!(target: "ferrule_profile", "layer.expert",
            image_generation = u64::MAX, layer = u64::MAX, expert = u64::MAX, rows = u64::MAX,
            operation = u64::MAX, source_generation = u64::MAX, requested_bytes = u64::MAX)
        .entered();
    });
    let labels = callback.labels();
    assert_eq!(
        labels[0],
        "read_subset operation=18446744073709551615 source_generation=7"
    );
    assert_eq!(
        labels[1],
        "read_subset operation=18446744073709551615 source_generation=7 requested_bytes=4096"
    );
    assert!(labels.iter().all(|label| label.len() < 512));
    callback.assert_balanced(3);
}

#[test]
fn sensitive_and_non_numeric_values_are_never_formatted_or_exported() {
    struct MustNotFormat;
    impl std::fmt::Debug for MustNotFormat {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            panic!("profiling formatted an arbitrary value")
        }
    }
    let callback = Callback::default();
    tracing::dispatcher::with_default(&dispatch(&callback, true), || {
        let span = tracing::debug_span!(target: "ferrule_profile", "layer.expert",
            image_generation = 12u64, layer = 3i64, expert = -1i64, rows = 1.5f64,
            operation = "credential", source_generation = ?MustNotFormat,
            prompt = "private prompt", path = "/private/checkpoint", weight = ?MustNotFormat,
            token = "secret", requested_bytes = false, arbitrary_count = 123u64);
        span.record("image_generation", "sensitive replacement");
        span.in_scope(|| {});
    });
    assert_eq!(
        callback.labels(),
        ["layer.expert image_generation=12 layer=3"]
    );
    callback.assert_balanced(1);
}

#[test]
fn disabled_layer_emits_no_callbacks_including_record_updates() {
    let callback = Callback::default();
    tracing::dispatcher::with_default(&dispatch(&callback, false), || {
        let span = tracing::debug_span!(target: "ferrule_profile", "read_subset", operation = 1u64);
        span.record("operation", 2u64);
        span.in_scope(|| {});
    });
    callback.assert_balanced(0);
}

#[derive(Clone, Default)]
struct LogBuffer(Arc<Mutex<Vec<u8>>>);

impl io::Write for LogBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn fmt_local_filter_does_not_disable_debug_profile_ranges() {
    for filter in ["off", "info"] {
        let callback = Callback::default();
        let logs = LogBuffer::default();
        let writer = logs.clone();
        let subscriber = tracing_subscriber::registry()
            .with(callback.layer(true).with_filter(filter_fn(allows)))
            .with(
                tracing_subscriber::fmt::layer()
                    .with_ansi(false)
                    .with_writer(move || writer.clone())
                    .with_filter(EnvFilter::new(filter)),
            );
        tracing::subscriber::with_default(subscriber, || {
            let _range =
                tracing::debug_span!(target: "ferrule_profile", "read_subset", operation = 1u64)
                    .entered();
            tracing::debug!(target: "ferrule_profile", "not a profile range");
            tracing::debug!("debug message stays filtered");
            tracing::info!("ordinary info");
        });
        callback.assert_balanced(1);
        let text = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
        assert!(!text.contains("debug message"));
        assert!(!text.contains("not a profile range"));
        assert_eq!(text.contains("ordinary info"), filter == "info");
    }
}

#[test]
fn ranges_balance_reentrant_enter_errors_unwind_and_cross_thread_resumption() {
    let callback = Callback::default();
    let dispatch = dispatch(&callback, true);
    let span = tracing::dispatcher::with_default(
        &dispatch,
        || tracing::debug_span!(target: "ferrule_profile", "read_subset", operation = 9u64),
    );
    {
        let _outer = span.enter();
        let _inner = span.enter();
    }
    let error: Result<(), ()> = span.in_scope(|| Err(()));
    assert!(error.is_err());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            span.in_scope(|| panic!("injected unwind"));
        }))
        .is_err()
    );
    std::thread::spawn(move || span.in_scope(|| {}))
        .join()
        .unwrap();
    callback.assert_balanced(5);
}

#[test]
fn concurrent_enters_of_one_span_have_distinct_balanced_ranges() {
    let callback = Callback::default();
    let dispatch = dispatch(&callback, true);
    let span = tracing::dispatcher::with_default(
        &dispatch,
        || tracing::debug_span!(target: "ferrule_profile", "layer.expert", image_generation = 7u64),
    );
    let barrier = std::sync::Barrier::new(8);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let span = span.clone();
            let barrier = &barrier;
            scope.spawn(move || {
                let _range = span.enter();
                barrier.wait();
                span.in_scope(|| {});
            });
        }
    });
    callback.assert_balanced(16);
}

#[test]
fn nested_scoped_dispatchers_do_not_confuse_span_identity() {
    let first = Callback::default();
    let second = Callback::default();
    tracing::dispatcher::with_default(&dispatch(&first, true), || {
        let _first = tracing::debug_span!(target: "ferrule_profile", "read_subset").entered();
        tracing::dispatcher::with_default(&dispatch(&second, true), || {
            let _second = tracing::debug_span!(target: "ferrule_profile", "read_subset").entered();
        });
    });
    first.assert_balanced(1);
    second.assert_balanced(1);
}

#[test]
fn real_header_shim_links_and_runs_on_cpu_without_cuda_or_injection() {
    tracing::subscriber::with_default(
        tracing_subscriber::registry().with(NvtxLayer::new().with_filter(filter_fn(allows))),
        || {
            let span =
                tracing::debug_span!(target: "ferrule_profile", "read_subset", operation = 1u64);
            span.in_scope(|| span.in_scope(|| {}));
        },
    );
}

#[test]
fn async_pending_and_cross_thread_poll_do_not_keep_ranges_entered() {
    use std::future::Future;
    use std::task::{Context, Poll, Waker};
    use tracing::Instrument;

    let callback = Callback::default();
    let dispatch = dispatch(&callback, true);
    let span = tracing::dispatcher::with_default(
        &dispatch,
        || tracing::debug_span!(target: "ferrule_profile", "read_subset", operation = 11u64),
    );
    let mut ready = false;
    let future = std::future::poll_fn(move |_| {
        if ready {
            Poll::Ready(())
        } else {
            ready = true;
            Poll::Pending
        }
    })
    .instrument(span);
    let mut future = Box::pin(future);
    let mut context = Context::from_waker(Waker::noop());
    assert!(future.as_mut().poll(&mut context).is_pending());
    // A suspended future has already ended its first host range.
    callback.assert_balanced(1);
    std::thread::spawn(move || {
        let mut context = Context::from_waker(Waker::noop());
        assert!(future.as_mut().poll(&mut context).is_ready());
    })
    .join()
    .unwrap();
    // Instrument also scopes the future's drop; every emitted range must end.
    let count = callback.labels().len();
    assert!(count >= 2);
    callback.assert_balanced(count);
}

#[test]
fn ranges_measure_enters_not_span_lifetime_and_can_end_out_of_order() {
    let callback = Callback::default();
    let dispatch = dispatch(&callback, true);
    tracing::dispatcher::with_default(&dispatch, || {
        let first = tracing::debug_span!(target: "ferrule_profile", "read_subset");
        let second = tracing::debug_span!(target: "ferrule_profile", "layer.expert");
        callback.assert_balanced(0);
        let first = first.entered();
        let second = second.entered();
        drop(first);
        drop(second);
    });
    callback.assert_balanced(2);
}
