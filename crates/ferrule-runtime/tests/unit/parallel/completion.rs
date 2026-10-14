//! CPU-only evidence/retention tests at the production pipeline boundaries.
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ferrule_common::Error;
use ferrule_common::error::ErrorChain;
use ferrule_model::decoder::{CompletionUnknown, KvPrepareQuiescenceUnknown, WorkProgress};

use super::completion_boundary::{
    Boundary, CompletionClass, CompletionContext, CompletionFailure, classify, is_unknown,
    needs_quarantine,
};
use super::*;
use crate::parallel::data::{CompletionOutcome, OwnerFailure, OwnerFailureKind};

fn context() -> CompletionContext {
    CompletionContext {
        rank: PipelineRank {
            local: ParallelRankId::new(0),
            global: ParallelRankId::new(7),
        },
        stage: "execute",
        transaction: Some(ExecutionTransactionId::new(41).unwrap()),
    }
}
fn unknown() -> Error {
    KvPrepareQuiescenceUnknown::wrap(context().transaction.unwrap(), error("secret fixture"))
}

#[test]
fn completion_variants_preserve_identity_and_quarantine() {
    let c = context();
    assert_eq!(
        c.completion(CompletionOutcome::<_, Error>::Success(3))
            .unwrap(),
        3
    );
    for (outcome, class, quarantine) in [
        (
            CompletionOutcome::Failed(error("ordinary")),
            CompletionClass::Failed,
            false,
        ),
        (
            CompletionOutcome::Cancelled,
            CompletionClass::Cancelled,
            true,
        ),
        (CompletionOutcome::Panicked, CompletionClass::Panicked, true),
        (
            CompletionOutcome::ReplicaUnavailable,
            CompletionClass::Unavailable,
            true,
        ),
        (
            CompletionOutcome::QuiescenceUnknown,
            CompletionClass::Unknown,
            true,
        ),
    ] {
        let failure = c.completion::<()>(outcome).unwrap_err();
        let evidence = ErrorChain::new(&failure)
            .downcast_ref::<CompletionFailure>()
            .unwrap();
        assert_eq!(evidence.context.rank, c.rank);
        assert_eq!(evidence.context.stage, "execute");
        assert_eq!(evidence.context.transaction, c.transaction);
        assert_eq!(classify(&failure), class);
        assert_eq!(needs_quarantine(&failure), quarantine);
    }
}

#[test]
fn unknown_in_secondary_branches_cannot_be_downgraded() {
    for source in [
        Error::with_cleanup("cleanup", error("primary"), Err(unknown())),
        Error::with_cleanup("cleanup", unknown(), Err(error("secondary"))),
        Error::FailureBatch {
            operation: "batch".into(),
            failures: vec![error("first"), unknown()],
        },
        Error::context("context", unknown()),
    ] {
        let failure = context().failure(CompletionClass::Failed, false, Some(Box::new(source)));
        assert_eq!(classify(&failure), CompletionClass::Unknown);
        assert!(needs_quarantine(&failure));
        assert_eq!(
            ErrorChain::new(&failure)
                .downcast_ref::<KvPrepareQuiescenceUnknown>()
                .unwrap()
                .transaction(),
            context().transaction.unwrap(),
        );
    }
    // Error text is not completion evidence.
    assert_eq!(
        classify(&error("QuiescenceUnknown")),
        CompletionClass::Failed
    );
}

#[test]
fn owner_failures_keep_transaction_and_secondary_unknown() {
    let c = context();
    let failure = pool_shutdown_error(crate::parallel::data::ShutdownError {
        failures: vec![
            OwnerFailure {
                rank: c.rank.local,
                transaction: None,
                kind: OwnerFailureKind::Shutdown(error("first")),
            },
            OwnerFailure {
                rank: c.rank.global,
                transaction: c.transaction,
                kind: OwnerFailureKind::QuiescenceUnknown,
            },
        ],
    });
    assert!(is_unknown(&failure));
    let evidence = ErrorChain::new(&failure)
        .filter_map(|e| e.downcast_ref::<CompletionFailure>())
        .find(|e| e.class == CompletionClass::Unknown)
        .unwrap();
    assert_eq!(evidence.context.rank.global, c.rank.global);
    assert_eq!(evidence.context.transaction, c.transaction);
}

#[test]
fn unknown_ack_is_terminal_even_if_a_later_poll_would_complete() {
    let mut calls = 0;
    let failure = poll_end(3, || {
        calls += 1;
        if calls == 1 {
            Err(unknown())
        } else {
            Ok(KvEndProgress::Complete)
        }
    })
    .unwrap_err();
    assert!(is_unknown(&failure));
    assert_eq!(calls, 1);
    let mut calls = 0;
    poll_end(3, || {
        calls += 1;
        if calls == 1 {
            Err(error("retryable"))
        } else {
            Ok(KvEndProgress::Complete)
        }
    })
    .unwrap();
    assert_eq!(calls, 2);
}

struct UnknownWorker {
    drops: Arc<AtomicUsize>,
    shutdowns: Arc<AtomicUsize>,
}
impl Drop for UnknownWorker {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
impl ReplicaWorker<Command> for UnknownWorker {
    type Output = Reply;
    type Error = Error;
    fn execute(&mut self, _: WorkRequest<Command>) -> Result<Reply> {
        // Use the production CUDA completion boundary with CPU-only evidence.
        // An ordinary execute failure must not hide unknown attention cleanup.
        super::completion_boundary::checked_cuda_completion(false, Err(cleanup_unknown()))
    }
    // Explicit Unknown must dominate even an incorrectly optimistic hook.
    fn panic_quiescence(&mut self) -> PanicQuiescence {
        PanicQuiescence::Quiescent
    }
    fn shutdown(&mut self) -> Result<()> {
        self.shutdowns.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn thread_unknown_never_releases_owner_or_becomes_shutdown_ack() {
    for batch in [false, true] {
        let drops = Arc::new(AtomicUsize::new(0));
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let d = Arc::clone(&drops);
        let s = Arc::clone(&shutdowns);
        let pool = DataParallelExecutor::new(
            DataParallelConfig {
                replicas: 1,
                max_outstanding_per_replica: 1,
                session_capacity: 1,
            },
            move |_| {
                Ok(UnknownWorker {
                    drops: d,
                    shutdowns: s,
                })
            },
        )
        .unwrap();
        let mut transport = ThreadTransport::new(pool);
        let rank = context().rank;
        let failure = if batch {
            transport
                .call_batch_observed(vec![(rank, Command::Describe)], &mut || {})
                .unwrap_err()
        } else {
            transport.call(rank, Command::Describe).unwrap_err()
        };
        assert!(is_unknown(&failure));
        let evidence = ErrorChain::new(&failure)
            .downcast_ref::<CompletionFailure>()
            .unwrap();
        assert_eq!(evidence.context.rank, rank);
        assert_eq!(evidence.context.stage, "describe");
        assert_eq!(evidence.context.transaction.unwrap().get(), 1);
        assert_eq!(transport.outstanding(), 0); // Host credit only, not custody.
        assert!(is_unknown(&transport.shutdown().unwrap_err()));
        assert!(is_unknown(
            &transport.call(rank, Command::Describe).unwrap_err()
        ));
        drop(transport);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(shutdowns.load(Ordering::SeqCst), 0);
    }
}

struct UnknownPrepare {
    description: Description,
    cleanup_calls: Rc<Cell<usize>>,
    quarantines: Rc<Cell<usize>>,
}
impl PipelineTransport for UnknownPrepare {
    fn call_observed(
        &mut self,
        _: PipelineRank,
        command: Command,
        _: &mut dyn FnMut(),
    ) -> Result<Reply> {
        match command {
            Command::Describe => Ok(Reply::Description(self.description.clone())),
            Command::Create { .. } => Ok(Reply::Generation(0)),
            Command::Prepare { key, .. } => Err(KvPrepareQuiescenceUnknown::wrap(
                key.binding().transaction(),
                error("prepare"),
            )),
            _ => {
                self.cleanup_calls.set(self.cleanup_calls.get() + 1);
                Ok(Reply::Ack(KvEndProgress::Complete))
            }
        }
    }
    fn outstanding(&self) -> usize {
        0
    }
    fn shutdown(&mut self) -> Result<()> {
        self.cleanup_calls.set(self.cleanup_calls.get() + 1);
        Ok(())
    }
    fn quarantine(&mut self) {
        self.quarantines.set(self.quarantines.get() + 1);
    }
}

#[test]
fn unknown_prepare_keeps_logical_pages_even_with_an_optimistic_transport() {
    let config = PipelineConfig {
        page_size: 2,
        max_pages: 4,
        max_positions: 8,
        max_batch_tokens: 2,
        session_capacity: 1,
        max_parameter_bytes: 1024,
        max_ack_polls: 2,
        precision: ExecutionPrecisionPolicy::f32(),
    };
    let plan = LayerSegmentPlan::new(1, 0..1, true, true).unwrap();
    let topology = ValidatedParallelTopology::new(
        ferrule_common::ParallelTopologyId::new(1),
        1,
        ParallelRankId::new(0),
        ferrule_common::ParallelismPlan::validated(1, 1, 1, 1, 1, 1).unwrap(),
    )
    .unwrap();
    let cleanup_calls = Rc::new(Cell::new(0));
    let quarantines = Rc::new(Cell::new(0));
    let mut pipeline = PipelineParallelExecutor::new_with_transport(
        topology,
        vec![plan.clone()],
        config,
        UnknownPrepare {
            description: Description {
                plan,
                config,
                physical_pages: 4,
                hidden: 2,
                vocabulary: 4,
                kv_heads: 1,
                head_dim: 2,
                expert_group: None,
            },
            cleanup_calls: Rc::clone(&cleanup_calls),
            quarantines: Rc::clone(&quarantines),
        },
    )
    .unwrap();
    let failure = pipeline
        .forward(SessionId(1), &[1], ForwardPhase::Prefill)
        .unwrap_err();
    assert!(is_unknown(&failure));
    assert!(pipeline.is_quarantined());
    assert_eq!(pipeline.page_manager().stats().allocated_pages, 1);
    assert_eq!(pipeline.page_manager().stats().committed_tokens, 0);
    assert_eq!(pipeline.coordinator().publication_count(), 0);
    assert_eq!(pipeline.coordinator().retained_transaction_count(), 1);
    assert!(pipeline.release_session(SessionId(1)).is_err());
    assert!(pipeline.shutdown().is_err());
    drop(pipeline);
    assert_eq!(cleanup_calls.get(), 0);
    assert!(quarantines.get() > 0);
}

#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<BTreeMap<String, String>>>>);
impl tracing::Subscriber for Logs {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target() == "ferrule_runtime::parallel::boundary"
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

#[test]
fn boundary_logs_have_only_allowlisted_fields_and_no_payload_or_error_text() {
    let logs = Logs::default();
    tracing::subscriber::with_default(logs.clone(), || {
        for operation in ["pipeline.thread", "pipeline.process"] {
            Boundary::new(operation, context()).finish(&Ok("secret payload"));
            Boundary::new(operation, context()).finish::<()>(&Err(error("secret checkpoint path")));
            Boundary::new(operation, context()).finish::<()>(&Err(unknown()));
        }
    });
    let records = logs.0.lock().unwrap();
    assert_eq!(records.len(), 6);
    for fields in records.iter() {
        assert!(fields.keys().all(|key| matches!(
            key.as_str(),
            "operation" | "rank" | "phase" | "bytes" | "duration" | "outcome"
        )));
        assert_eq!(fields["rank"], "7");
        assert_eq!(fields["phase"], "execute");
        assert!(fields["duration"].parse::<u64>().is_ok());
        assert!(fields.values().all(|value| !value.contains("secret")));
    }
    assert_eq!(records[0]["outcome"], "success");
    assert_eq!(records[1]["outcome"], "failed");
    assert_eq!(records[2]["outcome"], "unknown");
}

#[test]
fn external_work_unknown_startup_and_shutdown_preserve_sources_without_cuda() {
    for startup in [true, false] {
        let source = Error::with_cleanup(
            "external work",
            error("original failure"),
            Err(CompletionUnknown::wrap(
                std::io::Error::new(std::io::ErrorKind::TimedOut, "lost completion").into(),
            )),
        );
        let owner = OwnerFailure {
            rank: context().rank.global,
            transaction: None,
            kind: if startup {
                OwnerFailureKind::Initialization(source)
            } else {
                OwnerFailureKind::Shutdown(source)
            },
        };
        let failure = if startup {
            pool_build_error(crate::parallel::data::BuildError::Owners(vec![owner]))
        } else {
            pool_shutdown_error(crate::parallel::data::ShutdownError {
                failures: vec![owner],
            })
        };
        assert_eq!(classify(&failure), CompletionClass::Unknown);
        assert!(needs_quarantine(&failure));
        let evidence = ErrorChain::new(&failure)
            .downcast_ref::<CompletionFailure>()
            .unwrap();
        assert_eq!(
            evidence.context.stage,
            if startup { "startup" } else { "shutdown" }
        );
        assert_eq!(evidence.context.rank.global, context().rank.global);
        assert!(evidence.context.transaction.is_none());
        assert!(
            ErrorChain::new(&failure)
                .downcast_ref::<CompletionUnknown>()
                .is_some()
        );
        assert_eq!(
            ErrorChain::new(&failure)
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .kind(),
            std::io::ErrorKind::TimedOut,
        );
    }
}

#[test]
fn external_work_drain_unknown_cannot_be_laundered_by_a_late_kv_fence() {
    for terminal in [WorkProgress::Complete, WorkProgress::Unknown] {
        let mut progress = [WorkProgress::Pending, terminal, WorkProgress::Complete].into_iter();
        let mut calls = 0;
        let result = poll_end(3, || {
            calls += 1;
            match progress.next().unwrap() {
                WorkProgress::Pending => Ok(KvEndProgress::Pending),
                WorkProgress::Unknown => Err(CompletionUnknown::wrap(error("lost sub-work"))),
                // This is an independent, optimistic KV fence, not proof for unknown sub-work.
                WorkProgress::Complete => Ok(KvEndProgress::Complete),
            }
        });
        assert_eq!(calls, 2);
        if terminal == WorkProgress::Unknown {
            let failure = Error::with_cleanup("late fence", result.unwrap_err(), Ok(()));
            let failure =
                context().failure(CompletionClass::Failed, false, Some(Box::new(failure)));
            assert_eq!(classify(&failure), CompletionClass::Unknown);
            assert!(needs_quarantine(&failure));
        } else {
            result.unwrap();
        }
    }
}

fn cleanup_unknown() -> Error {
    CompletionUnknown::wrap(Error::with_cleanup(
        "Mixed attention cleanup",
        std::io::Error::new(std::io::ErrorKind::InvalidData, "primary execution failure").into(),
        Err(CompletionUnknown::wrap(
            std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "attention finish lost completion",
            )
            .into(),
        )),
    ))
}

#[test]
fn mixed_cleanup_unknown_preserves_primary_and_cleanup_at_completion_boundary() {
    let failure = context()
        .completion::<()>(CompletionOutcome::Failed(cleanup_unknown()))
        .unwrap_err();
    assert_eq!(classify(&failure), CompletionClass::Unknown);
    assert!(needs_quarantine(&failure));
    assert!(
        ErrorChain::new(&failure)
            .downcast_ref::<CompletionUnknown>()
            .is_some()
    );
    let causes = ErrorChain::new(&failure)
        .filter_map(|source| {
            source
                .downcast_ref::<std::io::Error>()
                .map(std::io::Error::kind)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        causes,
        vec![
            std::io::ErrorKind::InvalidData,
            std::io::ErrorKind::TimedOut
        ]
    );
    let evidence = ErrorChain::new(&failure)
        .downcast_ref::<CompletionFailure>()
        .unwrap();
    assert_eq!(evidence.context.transaction, context().transaction);

    let panic = std::panic::catch_unwind(|| {
        super::completion_boundary::checked_cuda_completion::<()>(false, Err(cleanup_unknown()))
    })
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<PanicQuiescence>(),
        Some(&PanicQuiescence::Unknown)
    );
}

#[test]
fn quarantined_completion_retains_output_and_never_launders_unknown_as_success() {
    struct RetainedOutput(Arc<AtomicUsize>);
    impl Drop for RetainedOutput {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let retained = Arc::clone(&drops);
    let panic = std::panic::catch_unwind(|| {
        super::completion_boundary::checked_cuda_completion(true, Ok(RetainedOutput(retained)))
    })
    .err()
    .expect("quarantined success must become typed Unknown");
    assert_eq!(
        panic.downcast_ref::<PanicQuiescence>(),
        Some(&PanicQuiescence::Unknown)
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    // Quiescent ordinary errors retain their original type instead of becoming Unknown.
    let source = super::completion_boundary::checked_cuda_completion::<()>(
        false,
        Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "ordinary").into()),
    )
    .unwrap_err();
    assert!(!is_unknown(&source));
    assert_eq!(
        ErrorChain::new(&source)
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .kind(),
        std::io::ErrorKind::InvalidInput
    );
}
