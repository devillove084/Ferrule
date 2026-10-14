use super::*;
use crate::engine::kv_trace::*;
fn config() -> RuntimeKvTraceConfig {
    RuntimeKvTraceConfig {
        run_scope: "driver-test".into(),
        ..Default::default()
    }
}
#[test]
fn kv_trace_disabled_and_enabled_capacity_zero_preserve_execution() {
    let mut plain = prefix_cache_driver(Vec::new(), 0, 8);
    let mut traced = prefix_cache_driver(Vec::new(), 0, 8);
    assert!(plain.drain_kv_trace().is_none());
    traced.enable_kv_trace(config()).unwrap();
    for d in [&mut plain, &mut traced] {
        d.try_submit(request(1, &[1, 2, 3], 0, Vec::new())).unwrap();
        d.drive_ready_test_work(|_| Ok(())).unwrap();
    }
    assert_eq!(
        plain.drain_finished()[0].tokens,
        traced.drain_finished()[0].tokens
    );
    assert_eq!(plain.stats().prefill_tokens, traced.stats().prefill_tokens);
    let b = traced.finalize_kv_trace().unwrap();
    assert!(b.complete, "{:?}", b.status);
    let p = project_runtime_kv_trace(std::slice::from_ref(&b)).unwrap();
    assert!(
        p.events
            .iter()
            .any(|e| matches!(e, ferrule_common::kvset::TraceEvent::Publish(_)))
    );
    let report = replay_runtime_prefix_lru(&[b]).unwrap();
    assert_eq!(report.charged_pages, 0);
}
#[test]
fn kv_trace_real_prefix_hit_and_shutdown() {
    let mut d = prefix_cache_driver(Vec::new(), 4, 8);
    d.enable_kv_trace(config()).unwrap();
    for (id, prompt) in [(1, vec![1, 2, 3]), (2, vec![1, 2, 3, 4])] {
        d.try_submit(request(id, &prompt, 0, Vec::new())).unwrap();
        d.drive_ready_test_work(|_| Ok(())).unwrap();
        d.drain_finished();
    }
    d.shutdown(&mut |_| Ok(()), 32).unwrap();
    let b = d.finalize_kv_trace().unwrap();
    assert!(b.complete, "{:?}", b.status);
    let r = replay_runtime_prefix_lru(std::slice::from_ref(&b)).unwrap();
    let input = project_runtime_kv_trace(std::slice::from_ref(&b)).unwrap();
    ferrule_common::kvset::analyze(&input, Default::default()).unwrap();
    assert_eq!(r.lookup_hits, 1);
    assert_eq!(r.actual_reused_tokens, 3);
    assert_eq!(r.charged_pages, 0);
    assert_eq!(r.confirmed_pages, r.backend_released_pages);
}
#[test]
fn kv_trace_cancel_waiting_has_demand_but_no_publish() {
    let mut d = prefix_cache_driver(Vec::new(), 0, 8);
    d.enable_kv_trace(config()).unwrap();
    d.try_submit(request(1, &[1, 2], 0, Vec::new())).unwrap();
    d.cancel_request(RequestId(1)).unwrap();
    d.drain_cancelled();
    let b = d.finalize_kv_trace().unwrap();
    assert!(b.complete);
    let p = project_runtime_kv_trace(&[b]).unwrap();
    assert_eq!(p.events.len(), 2);
}
#[test]
fn kv_trace_overflow_does_not_change_inference() {
    let mut d = prefix_cache_driver(Vec::new(), 0, 8);
    d.enable_kv_trace(RuntimeKvTraceConfig {
        max_events: 1,
        ..config()
    })
    .unwrap();
    d.try_submit(request(1, &[1, 2], 0, Vec::new())).unwrap();
    d.drive_ready_test_work(|_| Ok(())).unwrap();
    assert_eq!(d.drain_finished().len(), 1);
    let b = d.finalize_kv_trace().unwrap();
    assert!(b.status.overflow);
    assert!(!b.complete);
    assert!(project_runtime_kv_trace(&[b]).is_err());
}
#[test]
fn kv_trace_prepare_failure_does_not_publish() {
    let mut d = prefix_cache_driver(Vec::new(), 0, 8);
    d.enable_kv_trace(config()).unwrap();
    d.stage_faults.push_back("transaction prepare");
    d.try_submit(request(1, &[1, 2], 0, Vec::new())).unwrap();
    assert!(d.drive_ready_test_work(|_| Ok(())).is_err());
    d.drain_failed();
    let b = d.finalize_kv_trace().unwrap();
    assert!(
        !b.lifecycle_events
            .iter()
            .any(|e| matches!(e.kind, RuntimeKvTraceKind::LogicalKvPublished { .. }))
    );
    assert!(b.complete, "{:?}", b.status);
    replay_runtime_prefix_lru(&[b]).unwrap();
}
#[test]
fn kv_trace_boxed_engine_forwards_and_enable_is_fresh_only() {
    use crate::engine::{InferenceEngine, ResidentInferenceEngine};
    let d = prefix_cache_driver(Vec::new(), 0, 8);
    let mut e: Box<dyn InferenceEngine> = Box::new(ResidentInferenceEngine::new(d));
    e.enable_kv_trace(config()).unwrap();
    assert!(e.drain_kv_trace().is_some());
    assert!(e.finalize_kv_trace().unwrap().complete);
    let mut d = prefix_cache_driver(Vec::new(), 0, 8);
    d.try_submit(request(1, &[1], 0, Vec::new())).unwrap();
    assert!(d.enable_kv_trace(config()).is_err());
}

#[test]
fn kv_trace_real_speculative_partial_commit_only_publishes_accepted_rows() {
    let runner = MockTopKRunner::new(vec![top(10)]).with_speculative_cycle(
        NativeProposal {
            token_ids: vec![11, 12],
            confidence_logits: vec![1.0, 1.0],
        },
        vec![
            TokenLogit::new(11, 9.0),
            TokenLogit::new(99, 8.0),
            TokenLogit::new(98, 7.0),
        ],
    );
    let mut d = speculative_driver_from_runner(runner, 1);
    d.enable_kv_trace(config()).unwrap();
    let actions = ready_speculative_decode_actions(&mut d, &[1]);
    d.execute_speculative_decode_batch(actions, &mut |_| Ok(()))
        .unwrap();
    d.cancel_request(RequestId(1)).unwrap();
    d.drain_cancelled();
    let b = d.finalize_kv_trace().unwrap();
    assert!(b.complete, "{:?}", b.status);
    let p = project_runtime_kv_trace(std::slice::from_ref(&b)).unwrap();
    let published = p
        .events
        .iter()
        .filter_map(|e| match e {
            ferrule_common::kvset::TraceEvent::Publish(p) => Some(
                p.pages
                    .iter()
                    .flat_map(|p| p.tokens.iter().copied())
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(published.last().unwrap(), &vec![1, 10, 11]);
    replay_runtime_prefix_lru(&[b]).unwrap();
}
#[test]
fn kv_trace_lookup_touch_survives_failed_admission_and_unpin_retry() {
    let mut d = prefix_cache_driver(Vec::new(), 8, 16);
    d.enable_kv_trace(config()).unwrap();
    d.try_submit(request(1, &[1, 2], 0, Vec::new())).unwrap();
    d.drive_ready_test_work(|_| Ok(())).unwrap();
    d.drain_finished();
    d.stage_faults.extend(["prefix prepare", "prefix unpin"]);
    d.try_submit(request(2, &[1, 2, 3], 0, Vec::new())).unwrap();
    assert!(d.prepare_step().is_err());
    d.progress_pending_cleanups().unwrap();
    d.cancel_request(RequestId(2)).unwrap();
    d.drain_cancelled();
    let b = d.finalize_kv_trace().unwrap();
    assert!(b.complete, "{:?}", b.status);
    let r = replay_runtime_prefix_lru(&[b]).unwrap();
    assert_eq!(r.lookup_hits, 1);
    assert_eq!(r.actual_reused_tokens, 0);
    assert_eq!(r.published_admissions, 1);
}
#[test]
fn kv_trace_backend_release_retry_emits_only_successful_ack() {
    let mut d = prefix_cache_driver(Vec::new(), 0, 8);
    d.enable_kv_trace(config()).unwrap();
    d.executor.runner_mut().kv_release_failures_remaining = 1;
    d.try_submit(request(1, &[1, 2], 0, Vec::new())).unwrap();
    assert!(d.drive_ready_test_work(|_| Ok(())).is_err());
    d.progress_pending_cleanups().unwrap();
    d.progress_pending_cleanups().unwrap();
    d.drain_finished();
    d.drain_failed();
    let expected = d.executor.runner().released_kv_pages.len();
    let b = d.finalize_kv_trace().unwrap();
    let report = replay_runtime_prefix_lru(&[b]).unwrap();
    assert_eq!(report.backend_released_pages, expected);
    assert_eq!(report.confirmed_pages, expected);
    assert_eq!(expected, 1);
}
#[test]
fn kv_trace_positioned_ancestry_is_explicitly_incomplete() {
    let mut d = prefix_cache_driver(Vec::new(), 0, 8);
    d.enable_kv_trace(config()).unwrap();
    d.try_submit_at_position(request(1, &[1], 0, Vec::new()), 2)
        .unwrap();
    assert!(d.finalize_kv_trace().unwrap().status.unsupported);
}

#[test]
fn kv_trace_capacity_eviction_and_shared_charge_are_not_physical_release() {
    let mut d = prefix_cache_driver(Vec::new(), 2, 16);
    d.enable_kv_trace(config()).unwrap();
    for (id, prompt) in [
        (1, vec![1, 2, 3, 4]),
        (2, vec![1, 2, 3, 4, 5]),
        (3, vec![8, 9]),
        (4, vec![10, 11]),
    ] {
        d.try_submit(request(id, &prompt, 0, Vec::new())).unwrap();
        d.drive_ready_test_work(|_| Ok(())).unwrap();
        d.drain_finished();
    }
    let b = d.finalize_kv_trace().unwrap();
    assert!(b.complete);
    assert!(b.lifecycle_events.iter().any(|e| matches!(
        e.kind,
        RuntimeKvTraceKind::SnapshotReleased {
            retiring_pages: 0,
            ..
        }
    )));
    assert!(b.lifecycle_events.iter().any(|e| matches!(
        e.kind,
        RuntimeKvTraceKind::PrefixEntryRemoved {
            reason: RemovalReason::Capacity,
            ..
        }
    )));
    let r = replay_runtime_prefix_lru(&[b]).unwrap();
    assert_eq!(r.charged_pages, 2);
    assert!(r.removed_charge_pages > 0);
}
#[test]
fn kv_trace_page_pressure_and_partial_tail_cow_are_observed() {
    let mut d = prefix_cache_driver(Vec::new(), 8, 3);
    d.enable_kv_trace(config()).unwrap();
    for (id, prompt) in [
        (1, vec![1, 2, 3]),
        (2, vec![1, 2, 3, 4]),
        (3, vec![8, 9, 10, 11, 12]),
    ] {
        d.try_submit(request(id, &prompt, 0, Vec::new())).unwrap();
        d.drive_ready_test_work(|_| Ok(())).unwrap();
        d.drain_finished();
    }
    let b = d.finalize_kv_trace().unwrap();
    assert!(b.complete);
    assert!(b.lifecycle_events.iter().any(|e| matches!(
        e.kind,
        RuntimeKvTraceKind::ReserveSucceeded { cow: true, .. }
    )));
    assert!(b.lifecycle_events.iter().any(|e| matches!(
        e.kind,
        RuntimeKvTraceKind::PrefixEntryRemoved {
            reason: RemovalReason::PagePressure,
            ..
        }
    )));
    replay_runtime_prefix_lru(&[b]).unwrap();
}
#[test]
fn kv_trace_observer_preserves_tick_order_and_cleanup_when_disabled_or_overflowed() {
    let mut plain = prefix_cache_driver(Vec::new(), 4, 8);
    let mut traced = prefix_cache_driver(Vec::new(), 4, 8);
    let mut overflowed = prefix_cache_driver(Vec::new(), 4, 8);
    traced.enable_kv_trace(config()).unwrap();
    overflowed
        .enable_kv_trace(RuntimeKvTraceConfig {
            max_events: 0,
            ..config()
        })
        .unwrap();
    for driver in [&mut plain, &mut traced, &mut overflowed] {
        for (id, prompt) in [(1, vec![1, 2, 3]), (2, vec![1, 2, 3, 4])] {
            driver
                .try_submit(request(id, &prompt, 0, Vec::new()))
                .unwrap();
            driver.drive_ready_test_work(|_| Ok(())).unwrap();
            assert_eq!(driver.drain_finished().len(), 1);
        }
        driver.shutdown(&mut |_| Ok(()), 32).unwrap();
    }
    for driver in [&traced, &overflowed] {
        assert_eq!(driver.tick_trace, plain.tick_trace);
        assert_eq!(driver.prefix_hits(), plain.prefix_hits());
        assert_eq!(driver.prefix.pending_cleanup_count(), 0);
        assert_eq!(
            driver.executor.runner().released_kv_pages,
            plain.executor.runner().released_kv_pages
        );
        assert_eq!(
            driver.executor.runner().released_sequence_states,
            plain.executor.runner().released_sequence_states
        );
    }
    assert!(plain.finalize_kv_trace().is_none());
    assert!(traced.finalize_kv_trace().unwrap().complete);
    let overflow = overflowed.finalize_kv_trace().unwrap();
    assert!(overflow.status.overflow);
    assert!(!overflow.complete);
}

#[derive(Clone, Default)]
struct TraceLogs(std::sync::Arc<std::sync::Mutex<Vec<std::collections::BTreeMap<String, String>>>>);

impl tracing::Subscriber for TraceLogs {
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        // Evaluate the target filter against the current thread's subscriber.
        tracing::subscriber::Interest::sometimes()
    }

    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target().ends_with("engine::driver::trace")
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Fields(std::collections::BTreeMap<String, String>);
        impl tracing::field::Visit for Fields {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.0.insert(field.name().into(), format!("{value:?}"));
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                self.0.insert(field.name().into(), value.into());
            }
        }
        let mut fields = Fields(Default::default());
        event.record(&mut fields);
        self.0.lock().unwrap().push(fields.0);
    }
}

#[test]
fn kv_trace_boundary_logs_are_once_only_and_payload_free() {
    // tracing caches callsite interest process-wide, but this subscriber is
    // scoped to one thread. Other libtest workers can register these callsites
    // against their own default subscriber while this test runs. Keep the
    // real driver assertions in a fresh test process rather than mutating the
    // process-global tracing registry or installing a global subscriber.
    const CHILD: &str = "FERRULE_DRIVER_TRACE_BOUNDARY_CHILD";
    const VERIFIED: &str = "driver trace boundary capture and privacy assertions completed";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "engine::driver::tests::kv_trace_tests::kv_trace_boundary_logs_are_once_only_and_payload_free",
                "--test-threads=1",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        // A stale exact-test name must not silently pass by running zero tests.
        assert!(
            String::from_utf8_lossy(&output.stdout).contains(VERIFIED),
            "child did not finish the boundary assertions: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        return;
    }

    let logs = TraceLogs::default();
    let mut driver = prefix_cache_driver(Vec::new(), 0, 8);
    tracing::subscriber::with_default(logs.clone(), || {
        assert!(driver.drain_kv_trace().is_none());
        assert!(driver.finalize_kv_trace().is_none());
        driver
            .enable_kv_trace(RuntimeKvTraceConfig {
                run_scope: "private-run-scope-must-not-be-logged".into(),
                max_events: 1,
                ..config()
            })
            .unwrap();
        driver
            .try_submit(request(
                987654321,
                &[876543210, 765432109],
                0,
                vec!["private-stop-text-must-not-be-logged".into()],
            ))
            .unwrap();
        driver.drive_ready_test_work(|_| Ok(())).unwrap();
        assert_eq!(driver.drain_finished().len(), 1);
        // Overflow is reported at export, not on the token hot path.
        assert_eq!(logs.0.lock().unwrap().len(), 1);
        assert!(driver.drain_kv_trace().unwrap().status.overflow);
        assert!(driver.drain_kv_trace().unwrap().status.overflow);
        assert!(driver.finalize_kv_trace().unwrap().finalized);
        assert!(driver.finalize_kv_trace().is_none());
        assert!(driver.drain_kv_trace().is_none());
    });
    let events = logs.0.lock().unwrap();
    let states = events
        .iter()
        .map(|event| event["state"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(states, ["disabled", "overflow", "finalized"]);
    for event in events.iter() {
        assert_eq!(event["component"], "runtime_kv_trace");
        for (field, value) in event {
            assert!(matches!(
                field.as_str(),
                "message"
                    | "component"
                    | "state"
                    | "reason"
                    | "complete"
                    | "events"
                    | "observations"
                    | "token_bytes"
            ));
            for secret in [
                "private-run-scope",
                "private-stop-text",
                "987654321",
                "876543210",
                "765432109",
            ] {
                assert!(!value.contains(secret), "sensitive payload in {field}");
            }
        }
    }
    println!("{VERIFIED}");
}

#[tokio::test(flavor = "current_thread")]
async fn kv_trace_local_session_and_local_resident_forward() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let d = prefix_cache_driver(Vec::new(), 0, 8);
            let mut e = crate::engine::LocalSessionInferenceEngine::new(Box::new(
                crate::engine::ResidentInferenceEngine::new(d),
            ));
            e.enable_kv_trace(config()).unwrap();
            assert!(e.drain_kv_trace().is_some());
            assert!(e.finalize_kv_trace().unwrap().complete);
            let mut e = crate::engine::LocalResidentInferenceEngine::new(prefix_cache_driver(
                Vec::new(),
                0,
                8,
            ));
            e.enable_kv_trace(config()).unwrap();
            assert!(e.drain_kv_trace().is_some());
            assert!(e.finalize_kv_trace().unwrap().complete);
        })
        .await;
}
