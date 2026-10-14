//! Fault-injected admission boundaries only. No GPU, worker or custody substitute.
use super::*;
use ferrule_common::error::ErrorChain;
use ferrule_runtime::parallel::data::PanicQuiescence;
use std::{
    cell::Cell,
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[derive(Debug)]
struct InjectedFailure(&'static str);
impl std::fmt::Display for InjectedFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for InjectedFailure {}
fn failure(message: &'static str) -> ferrule_common::Error {
    ferrule_common::Error::ModelSource {
        source: Box::new(InjectedFailure(message)),
    }
}

#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<BTreeMap<String, String>>>>);
impl tracing::Subscriber for Logs {
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }
    fn max_level_hint(&self) -> Option<tracing::metadata::LevelFilter> {
        Some(tracing::metadata::LevelFilter::ERROR)
    }
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target() == "ferrule_composition::parallel::expert_admission"
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
            fn record_error(
                &mut self,
                field: &tracing::field::Field,
                value: &(dyn std::error::Error + 'static),
            ) {
                let source = ErrorChain::new(value)
                    .downcast_ref::<InjectedFailure>()
                    .expect("typed primary diagnostic");
                self.0.insert(field.name().into(), source.0.into());
            }
        }
        let mut fields = Fields(BTreeMap::new());
        event.record(&mut fields);
        self.0.lock().unwrap().push(fields.0);
    }
}

#[derive(Clone, Copy)]
enum Boundary {
    Pipeline,
    Resident,
}
impl Boundary {
    fn operation(self) -> &'static str {
        match self {
            Self::Pipeline => "expert Ready",
            Self::Resident => "Qwen3.5 EP owner admission",
        }
    }
    fn finish<T>(
        self,
        admission: ferrule_common::Result<T>,
        cleanup: impl FnOnce() -> ferrule_common::Result<()>,
    ) -> crate::Result<T> {
        match self {
            Self::Pipeline => finish_ready(admission, cleanup).map_err(crate::Error::from),
            Self::Resident => crate::model_factory::qwen35::finish_owner_admission(
                admission.map_err(crate::Error::from),
                || cleanup().map_err(crate::Error::from),
            ),
        }
    }
    fn assert_logged(self, logs: &Logs) {
        let records = logs.0.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["operation"], self.operation());
        assert_eq!(records[0]["error"], "Ready primary");
        assert_eq!(records[0]["phase"], "ready");
        assert_eq!(records[0]["cleanup"], "not_started");
    }
}

#[test]
fn ordinary_cleanup_retains_both_typed_errors_after_the_primary_event() {
    for boundary in [Boundary::Pipeline, Boundary::Resident] {
        let logs = Logs::default();
        let error = tracing::subscriber::with_default(logs.clone(), || {
            boundary
                .finish::<()>(Err(failure("Ready primary")), || {
                    boundary.assert_logged(&logs);
                    Err(failure("cleanup secondary"))
                })
                .unwrap_err()
        });
        let (primary, cleanup): (&dyn std::error::Error, &dyn std::error::Error) = match &error {
            crate::Error::Cleanup {
                source, cleanup, ..
            } => (source.as_ref(), cleanup.as_ref()),
            crate::Error::Backend {
                source:
                    ferrule_common::Error::Cleanup {
                        source, cleanup, ..
                    },
            } => (source.as_ref(), cleanup.as_ref()),
            _ => panic!("expected the original typed primary/cleanup pair: {error:?}"),
        };
        assert_eq!(
            ErrorChain::new(primary)
                .downcast_ref::<InjectedFailure>()
                .unwrap()
                .0,
            "Ready primary"
        );
        assert_eq!(
            ErrorChain::new(cleanup)
                .downcast_ref::<InjectedFailure>()
                .unwrap()
                .0,
            "cleanup secondary"
        );
    }
}

#[test]
fn unknown_cleanup_keeps_the_bare_runtime_payload_and_logs_primary_first() {
    for boundary in [Boundary::Pipeline, Boundary::Resident] {
        let logs = Logs::default();
        let calls = Cell::new(0);
        let continued = Cell::new(false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tracing::subscriber::with_default(logs.clone(), || {
                let _ = boundary.finish::<()>(Err(failure("Ready primary")), || {
                    boundary.assert_logged(&logs);
                    calls.set(calls.get() + 1);
                    std::panic::panic_any(PanicQuiescence::Unknown)
                });
                continued.set(true);
            });
        }));
        assert_eq!(
            result.err().unwrap().downcast_ref::<PanicQuiescence>(),
            Some(&PanicQuiescence::Unknown)
        );
        assert_eq!(calls.get(), 1);
        assert!(!continued.get(), "Unknown cannot become an ordinary return");
        boundary.assert_logged(&logs);
    }
}

#[test]
fn successful_admission_does_not_log_or_cleanup_and_clean_failure_keeps_primary() {
    for boundary in [Boundary::Pipeline, Boundary::Resident] {
        let logs = Logs::default();
        tracing::subscriber::with_default(logs.clone(), || {
            assert_eq!(
                boundary
                    .finish(Ok(7), || panic!("cleanup after success"))
                    .unwrap(),
                7
            );
            assert!(logs.0.lock().unwrap().is_empty());
            let error = boundary
                .finish::<()>(Err(failure("Ready primary")), || {
                    boundary.assert_logged(&logs);
                    Ok(())
                })
                .unwrap_err();
            assert_eq!(
                ErrorChain::new(&error)
                    .downcast_ref::<InjectedFailure>()
                    .unwrap()
                    .0,
                "Ready primary"
            );
            assert!(error.cleanup_errors().is_empty());
        });
    }
}
