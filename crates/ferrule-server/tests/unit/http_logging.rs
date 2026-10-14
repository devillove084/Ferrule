use super::*;
use http_body_util::BodyExt;
use std::collections::BTreeMap;
use std::sync::Mutex;

#[derive(Clone, Default)]
struct RejectionLogs(
    Arc<Mutex<Vec<BTreeMap<String, String>>>>,
    Option<tracing::Level>,
);

impl tracing::Subscriber for RejectionLogs {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        assert_eq!(
            *event.metadata().level(),
            self.1.unwrap_or(tracing::Level::DEBUG)
        );
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

#[test]
fn cleanup_policy_uses_primary_and_logs_every_secondary() {
    use ferrule_common::WorkerExecutionError;
    use ferrule_runtime::{CleanupStep, Error};
    for (primary, status, code) in [
        (
            Error::Invariant {
                message: "execution canary".into(),
            },
            StatusCode::INTERNAL_SERVER_ERROR,
            "execution_failed",
        ),
        (
            Error::InvalidRequest {
                message: "request canary".into(),
            },
            StatusCode::BAD_REQUEST,
            "invalid_request",
        ),
        (
            Error::RequestCapacity { limit: 1 },
            StatusCode::TOO_MANY_REQUESTS,
            "runtime_capacity",
        ),
    ] {
        let error = WorkerExecutionError::Runtime {
            source: Error::Shared {
                source: Arc::new(Error::CleanupBatch {
                    operation: "abort",
                    source: Box::new(primary),
                    cleanup: vec![
                        CleanupStep {
                            operation: "KV release".into(),
                            source: Error::Invariant {
                                message: "unknown paged KV page canary".into(),
                            },
                        },
                        CleanupStep {
                            operation: "rollback".into(),
                            source: Error::EngineUnavailable,
                        },
                    ],
                }),
            },
        };
        let public = public_execution_error(&error);
        assert_eq!(public.status, status);
        assert_eq!(public.code, code);
        let logs = RejectionLogs(Default::default(), Some(tracing::Level::ERROR));
        tracing::subscriber::with_default(logs.clone(), || {
            crate::worker::trace_execution_failure(&error, "generation failed");
        });
        let events = logs.0.lock().unwrap();
        assert_eq!(events.len(), 1);
        let chain = &events[0]["error_chain"];
        assert!(chain.contains("unknown paged KV page canary"), "{chain}");
        assert!(chain.contains("closed or quarantined"), "{chain}");
        assert_eq!(events[0]["message"], "generation failed");
    }
}

#[tokio::test]
async fn json_rejection_logs_only_stable_fields_and_preserves_public_projection() {
    const PRIVATE_VALUE: &str = "secret-value-canary /private/model/path token-canary-987654321";
    const PRIVATE_PROMPT: &str = "private-prompt-canary";
    const PRIVATE_MODEL: &str = "private-model-canary";

    let completion = serde_json::to_vec(&json!({
        "model": PRIVATE_MODEL,
        "prompt": PRIVATE_PROMPT,
        "max_tokens": PRIVATE_VALUE,
    }))
    .unwrap();
    let chat = serde_json::to_vec(&json!({
        "model": PRIVATE_MODEL,
        "messages": [{"role": "user", "content": PRIVATE_PROMPT}],
        "max_completion_tokens": PRIVATE_VALUE,
    }))
    .unwrap();
    let syntax =
        br#"{"model":"private-model-canary","prompt":"private-prompt-canary","max_tokens":"#;

    let rejections = [
        (
            Json::<CompletionRequest>::from_bytes(&completion).unwrap_err(),
            true,
        ),
        (
            Json::<ChatCompletionRequest>::from_bytes(&chat).unwrap_err(),
            true,
        ),
        (
            Json::<CompletionRequest>::from_bytes(syntax).unwrap_err(),
            false,
        ),
    ];
    for (rejection, contains_private_value) in rejections {
        if contains_private_value {
            // A real serde type error echoes the illegal integer-field value.
            assert!(rejection.to_string().contains(PRIVATE_VALUE));
        }
        let logs = RejectionLogs::default();
        let error = tracing::subscriber::with_default(logs.clone(), || json_rejection(rejection));
        let response = error.response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
            json!({
                "error": {
                    "message": "request body is not valid JSON for this endpoint",
                    "type": "invalid_request_error",
                    "code": "invalid_json",
                    "param": "body",
                }
            })
        );

        let events = logs.0.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            BTreeMap::from([
                ("message".into(), "invalid JSON request".into()),
                ("reason".into(), "invalid_json".into()),
                ("status".into(), "400".into()),
            ])
        );
        let recorded = format!("{events:?}");
        let public_body = std::str::from_utf8(&body).unwrap();
        for secret in [
            PRIVATE_VALUE,
            "secret-value-canary",
            "/private/model/path",
            "token-canary-987654321",
            PRIVATE_PROMPT,
            PRIVATE_MODEL,
        ] {
            assert!(!recorded.contains(secret), "payload leaked into tracing");
            assert!(
                !public_body.contains(secret),
                "payload leaked into response"
            );
        }
    }
}
