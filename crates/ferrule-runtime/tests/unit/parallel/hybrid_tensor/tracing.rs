use super::*;

#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<BTreeMap<String, String>>>>);
impl ::tracing::Subscriber for Logs {
    fn register_callsite(
        &self,
        _: &'static ::tracing::Metadata<'static>,
    ) -> ::tracing::subscriber::Interest {
        ::tracing::subscriber::Interest::sometimes()
    }
    fn max_level_hint(&self) -> Option<::tracing::metadata::LevelFilter> {
        Some(::tracing::metadata::LevelFilter::DEBUG)
    }
    fn enabled(&self, metadata: &::tracing::Metadata<'_>) -> bool {
        metadata.target() == "ferrule_runtime::parallel::boundary"
    }
    fn new_span(&self, _: &::tracing::span::Attributes<'_>) -> ::tracing::span::Id {
        ::tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &::tracing::span::Id, _: &::tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &::tracing::span::Id, _: &::tracing::span::Id) {}
    fn enter(&self, _: &::tracing::span::Id) {}
    fn exit(&self, _: &::tracing::span::Id) {}
    fn event(&self, event: &::tracing::Event<'_>) {
        struct Fields(BTreeMap<String, String>);
        impl ::tracing::field::Visit for Fields {
            fn record_debug(
                &mut self,
                field: &::tracing::field::Field,
                value: &dyn std::fmt::Debug,
            ) {
                self.0.insert(field.name().into(), format!("{value:?}"));
            }
            fn record_str(&mut self, field: &::tracing::field::Field, value: &str) {
                self.0.insert(field.name().into(), value.into());
            }
        }
        let mut fields = Fields(BTreeMap::new());
        event.record(&mut fields);
        self.0.lock().unwrap().push(fields.0);
    }
}

#[test]
fn real_factory_traces_success_failure_unknown_without_payload_or_error_text() {
    // Isolate the tracing callsite registry from earlier parallel tests that ran
    // the same production callsites without a subscriber. This is still the
    // real factory/transport; no test emits replacement boundary events.
    const CHILD: &str = "FERRULE_HYBRID_BOUNDARY_TRACE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "parallel::pipeline::hybrid_tensor::tests::tracing::real_factory_traces_success_failure_unknown_without_payload_or_error_text", "--test-threads=1", "--nocapture"])
            .env(CHILD, "1")
            .output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let logs = Logs::default();
    ::tracing::subscriber::with_default(logs.clone(), || {
        for mode in [3, 1, 2] {
            let fixture = Arc::new(fixture::Fixture::new());
            let secret_path = fixture.dir.to_string_lossy().into_owned();
            let mut pp =
                retry::child_fault_pipeline(fixture, 2, mode, Arc::new(AtomicBool::new(false)));
            pp.forward(SessionId(987654321), &[1, 2], ForwardPhase::Prefill)
                .unwrap();
            let result = pp.forward(SessionId(987654321), &[9], ForwardPhase::Decode);
            match mode {
                3 => {
                    result.unwrap();
                    pp.shutdown().unwrap();
                }
                1 => {
                    let failure = result.unwrap_err();
                    assert!(!completion_boundary::is_unknown(&failure));
                    pp.shutdown().unwrap();
                }
                _ => {
                    let failure = result.unwrap_err();
                    assert!(completion_boundary::is_unknown(&failure));
                    assert!(pp.shutdown().is_err());
                }
            }
            assert!(
                logs.0
                    .lock()
                    .unwrap()
                    .iter()
                    .flat_map(|fields| fields.values())
                    .all(|v| !v.contains(&secret_path))
            );
        }
    });
    let records = logs.0.lock().unwrap();
    assert!(!records.is_empty());
    for outcome in ["success", "failed", "unknown"] {
        assert!(
            records
                .iter()
                .any(|f| f["outcome"] == outcome && f["phase"] == "execute"),
            "missing {outcome}: {records:?}"
        );
    }
    for fields in records.iter() {
        assert!(fields.keys().all(|key| matches!(
            key.as_str(),
            "operation" | "rank" | "phase" | "duration" | "outcome"
        )));
        assert!(fields["duration"].parse::<u64>().is_ok());
        assert!(fields["rank"].parse::<u32>().unwrap() < 4);
        assert!(
            fields
                .values()
                .all(|v| !v.contains("secret") && !v.contains("987654321"))
        );
    }
}
