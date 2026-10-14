use super::*;
use std::collections::BTreeMap;
use std::time::Duration;

#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<BTreeMap<String, String>>>>);
impl tracing::Subscriber for Logs {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target() == "ferrule_request"
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Fields(BTreeMap<String, String>);
        impl tracing::field::Visit for Fields {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.0.insert(field.name().into(), format!("{value:?}"));
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                self.0.insert(field.name().into(), value.into());
            }
        }
        let mut fields = Fields(BTreeMap::new());
        event.record(&mut fields);
        self.0.lock().unwrap().push(fields.0);
    }
}

fn token(text: &str) -> ResidentTokenEvent {
    ResidentTokenEvent {
        request_id: Some(RequestId(7)),
        session_id: SessionId(7),
        index: 0,
        token: 42,
        logit: None,
        text: text.into(),
    }
}

fn request(timing: bool, capacity: usize) -> (ActiveRequest<'static>, mpsc::Receiver<WorkerEvent>) {
    let (events, receiver) = mpsc::channel(capacity);
    (
        ActiveRequest {
            _metrics: METRICS.start_request(),
            events,
            permit: test_permit(),
            cancellation: Arc::new(AtomicBool::new(false)),
            cancellation_submitted: false,
            cancellation_failed: false,
            terminal_sent: false,
            requires_restore_or_shutdown: false,
            session_id: SessionId(7),
            submitted_at: Instant::now(),
            emitted_tokens: 0,
            emission_timing: timing.then(WorkerEmissionTiming::default),
        },
        receiver,
    )
}

#[test]
fn worker_emission_zero_one_and_exact_interval_aggregates() {
    let mut timing = WorkerEmissionTiming::default();
    assert_eq!(timing.first, None);
    assert_eq!(timing.last, None);
    assert_eq!(timing.intervals, 0);
    assert_eq!(timing.interval_min, None);
    timing.record(Duration::from_nanos(1_100));
    assert_eq!(timing.first, timing.last);
    assert_eq!(timing.intervals, 0);
    assert_eq!(timing.interval_max, None);
    for ns in [2_900, 2_900, 7_100] {
        timing.record(Duration::from_nanos(ns));
    }
    assert_eq!(timing.intervals, 3);
    assert_eq!(timing.interval_sum, Duration::from_nanos(6_000));
    assert_eq!(timing.interval_min, Some(Duration::ZERO));
    assert_eq!(timing.interval_max, Some(Duration::from_nanos(4_200)));
    assert_eq!(emission_micros(timing.interval_sum), 6);
}

#[test]
fn worker_emission_counts_only_successful_sends_including_empty_text() {
    let (mut request, mut receiver) = request(true, 4);
    assert!(request.emit_token(&token("")));
    assert!(request.emit_token(&token("text")));
    assert_eq!(request.emitted_tokens, 2);
    let timing = request.emission_timing.as_ref().unwrap();
    assert_eq!(timing.intervals, 1);
    assert_eq!(
        timing.interval_sum,
        timing.last.unwrap() - timing.first.unwrap()
    );
    assert!(matches!(receiver.try_recv(), Ok(WorkerEvent::Token { text }) if text.is_empty()));
    request.cancellation.store(true, Ordering::Release);
    assert!(!request.emit_token(&token("after cancel")));
    assert_eq!(request.emitted_tokens, 2);
    assert_eq!(request.emission_timing.as_ref().unwrap().intervals, 1);
}

#[test]
fn worker_emission_backpressure_and_closed_channel_do_not_create_timing() {
    for capacity in [1, 4] {
        let (mut request, receiver) = request(true, capacity);
        if capacity == 4 {
            drop(receiver);
        }
        assert!(!request.emit_token(&token("not delivered")));
        assert!(request.cancellation.load(Ordering::Acquire));
        assert_eq!(request.emitted_tokens, 0);
        assert_eq!(request.emission_timing.as_ref().unwrap().first, None);
    }
    let (mut request, _receiver) = request(false, 4);
    assert!(request.emit_token(&token("unobserved")));
    assert!(request.emission_timing.is_none());
    assert_eq!(request.emitted_tokens, 1);
}

#[test]
fn worker_terminal_trace_has_numeric_units_and_cancel_state_without_token_logs() {
    let logs = Logs::default();
    tracing::subscriber::with_default(logs.clone(), || {
        let (mut request, _receiver) = request(true, 4);
        assert!(request.emit_token(&token("private token canary")));
        assert!(logs.0.lock().unwrap().is_empty(), "no per-token trace");
        // Deterministic timing payload independent of CPU scheduling.
        let timing = request.emission_timing.as_mut().unwrap();
        *timing = WorkerEmissionTiming::default();
        timing.record(Duration::from_micros(10));
        timing.record(Duration::from_micros(25));
        request.emitted_tokens = 2;
        request.cancellation.store(true, Ordering::Release);
        request.cancellation_submitted = true;
        trace_worker_request_terminal(
            RequestId(7),
            SessionId(7),
            "cancelled",
            "cancelled",
            &request,
            Some(Usage::new(1, 2)),
        );
    });
    let events = logs.0.lock().unwrap();
    assert_eq!(events.len(), 1);
    let fields = &events[0];
    for (name, value) in [
        ("event", "request_worker_terminal"),
        ("status", "cancelled"),
        ("emitted_token_events", "2"),
        ("worker_first_emission_elapsed_us", "10"),
        ("worker_last_emission_elapsed_us", "25"),
        ("worker_emission_interval_count", "1"),
        ("worker_emission_interval_sum_us", "15"),
        ("worker_emission_interval_min_us", "15"),
        ("worker_emission_interval_max_us", "15"),
        ("cancellation_requested", "true"),
        ("cancellation_submitted", "true"),
        ("emission_timing_enabled", "true"),
    ] {
        assert_eq!(fields[name], value, "{name}");
    }
    assert!(!format!("{fields:?}").contains("private token canary"));
}

#[test]
fn worker_terminal_zero_one_and_disabled_have_no_fabricated_intervals() {
    for (enabled, count) in [(true, 0), (true, 1), (false, 1)] {
        let logs = Logs::default();
        tracing::subscriber::with_default(logs.clone(), || {
            let (mut request, _receiver) = request(enabled, 4);
            for _ in 0..count {
                assert!(request.emit_token(&token("x")));
            }
            trace_worker_request_terminal(
                RequestId(7),
                SessionId(7),
                "finished",
                "length",
                &request,
                Some(Usage::new(1, count)),
            );
        });
        let events = logs.0.lock().unwrap();
        let fields = &events[0];
        assert!(!fields.contains_key("worker_emission_interval_min_us"));
        assert!(!fields.contains_key("worker_emission_interval_max_us"));
        assert_eq!(
            fields.contains_key("worker_first_emission_elapsed_us"),
            enabled && count > 0
        );
        assert_eq!(
            fields.contains_key("worker_last_emission_elapsed_us"),
            enabled && count > 0
        );
        if enabled {
            assert_eq!(fields["worker_emission_interval_count"], "0");
        } else {
            assert!(!fields.contains_key("worker_emission_interval_count"));
        }
    }
}

#[test]
fn worker_quarantine_emits_summary_without_claiming_cancellation_or_release() {
    let logs = Logs::default();
    tracing::subscriber::with_default(logs.clone(), || {
        let (mut request, _receiver) = request(true, 4);
        request.cancellation_failed = true;
        let mut active = HashMap::from([(RequestId(7), request)]);
        finish_request(
            &mut active,
            RequestId(7),
            RequestOutcome::Quarantined,
            &ready_control(),
        );
        assert!(active.is_empty());
    });
    let events = logs.0.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["status"], "quarantined");
    assert_eq!(events[0]["finish_reason"], "cleanup_unproven");
    assert_eq!(events[0]["cancellation_failed"], "true");
    assert_eq!(events[0]["cancellation_submitted"], "false");
}
