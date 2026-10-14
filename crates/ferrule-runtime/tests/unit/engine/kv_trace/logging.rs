use super::*;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct ValidationLogs(Arc<Mutex<Vec<BTreeMap<String, String>>>>);

impl tracing::Subscriber for ValidationLogs {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target().ends_with("kv_trace::validation")
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        assert_eq!(*event.metadata().level(), tracing::Level::DEBUG);
        struct Fields(BTreeMap<String, String>);
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
fn validation_logs_only_rejection_or_final_summary_without_payload() {
    let logs = ValidationLogs::default();
    let mut r = RuntimeKvRecorder::new(
        RuntimeKvTraceConfig {
            run_scope: "private-run-scope".into(),
            ..config()
        },
        2,
        8,
        [1; 5],
    )
    .unwrap();
    tracing::subscriber::with_default(logs.clone(), || {
        r.demand(1, 1, &[987654321]);
        r.terminal(1, TerminalStatus::Cancelled);
        let batch = r.drain(true).unwrap();
        assert!(logs.0.lock().unwrap().is_empty());
        project_runtime_kv_trace(std::slice::from_ref(&batch)).unwrap();
        replay_runtime_prefix_lru(std::slice::from_ref(&batch)).unwrap();
        let mut forged = batch;
        forged.usage.token_bytes = u64::MAX;
        assert!(project_runtime_kv_trace(&[forged]).is_err());
        assert!(replay_runtime_prefix_lru(&[]).is_err());
    });
    let events = logs.0.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .map(|event| event["state"].as_str())
            .collect::<Vec<_>>(),
        ["final_summary", "final_summary", "rejected", "rejected"],
    );
    for event in events.iter() {
        assert_eq!(event["component"], "runtime_kv_trace_validation");
        for (field, value) in event {
            assert!(matches!(
                field.as_str(),
                "message"
                    | "component"
                    | "state"
                    | "reason"
                    | "batches"
                    | "resident_entries"
                    | "charged_pages"
            ));
            assert!(!value.contains("private-run-scope"));
            assert!(!value.contains("987654321"));
        }
    }
}
