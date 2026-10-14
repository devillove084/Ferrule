use super::*;
use ferrule_runtime::{CleanupStep, Error};
use std::sync::Arc;

fn execution_and_cleanup() -> Error {
    Error::Shared {
        source: Arc::new(Error::CleanupBatch {
            operation: "resident transaction abort",
            source: Box::new(
                ferrule_common::Error::Execution {
                    message: "original execution failure /private/primary".into(),
                }
                .into(),
            ),
            cleanup: vec![
                CleanupStep {
                    operation: "KV release".into(),
                    source: ferrule_common::Error::Execution {
                        message: "cannot release an unknown paged KV page /private/secondary"
                            .into(),
                    }
                    .into(),
                },
                CleanupStep {
                    operation: "rollback".into(),
                    source: Error::EngineUnavailable,
                },
            ],
        }),
    }
}

#[tokio::test]
async fn cleanup_failure_keeps_json_and_sse_execution_semantics_and_fatal_readiness() {
    for stream in [false, true] {
        let worker = spawn_model_worker_with(
            || {
                Ok::<_, std::convert::Infallible>(Pr13FaultEngine {
                    execution_error: Some(execution_and_cleanup()),
                    cancellation_error: true,
                    ..Default::default()
                })
            },
            Default::default(),
        )
        .unwrap();
        let handle = worker.handle();
        let app = router(ServerState::new(
            ModelRegistration::new("test-model", ChatTemplate::Plain),
            handle.clone(),
        ));
        let response = app
            .clone()
            .oneshot(pr13_request(
                "/v1/completions",
                serde_json::json!({
                    "model": "test-model", "prompt": "hello", "max_tokens": 1, "stream": stream,
                })
                .to_string(),
            ))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            if stream {
                StatusCode::OK
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        );
        let body = response_text(response).await;
        assert!(body.contains("execution_failed"), "{body}");
        for private in [
            "/private",
            "unknown paged",
            "original execution",
            "worker_unavailable",
        ] {
            assert!(!body.contains(private), "{body}");
        }
        if stream {
            assert_eq!(body.matches("data: [DONE]").count(), 1);
            assert_eq!(body.matches("\"error\"").count(), 1);
            assert!(
                !body.contains("\"finish_reason\""),
                "failure must not emit a success terminal: {body}"
            );
        }
        let fatal = handle.snapshot().first_fatal.unwrap();
        let ferrule_common::WorkerExecutionError::Runtime { source } = fatal.as_ref() else {
            panic!("missing runtime primary");
        };
        assert!(
            source
                .primary()
                .to_string()
                .contains("original execution failure")
        );
        assert_eq!(source.cleanup_errors().len(), 2);
        assert!(
            source
                .diagnostic_chain()
                .any(|cause| cause.to_string().contains("unknown paged KV page"))
        );
        let readiness = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(readiness.status(), StatusCode::SERVICE_UNAVAILABLE);
        let next = app
            .oneshot(pr13_request(
                "/v1/completions",
                r#"{"model":"test-model","prompt":"hello","max_tokens":1}"#,
            ))
            .await
            .unwrap();
        assert_eq!(next.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(response_text(next).await.contains("worker_unavailable"));
        let shutdown = worker.shutdown().await.unwrap_err();
        let diagnostic = format!("{shutdown:?}");
        assert!(diagnostic.contains("original execution failure"));
        assert!(diagnostic.contains("unknown paged KV page"));
        let Error::CleanupBatch {
            source, cleanup, ..
        } = &shutdown
        else {
            panic!("shutdown must preserve the fatal primary: {shutdown:?}");
        };
        assert!(source.to_string().contains("original execution failure"));
        assert!(cleanup.iter().any(|step| {
            step.source
                .to_string()
                .contains("injected fatal cancellation cleanup")
        }));
    }
}
