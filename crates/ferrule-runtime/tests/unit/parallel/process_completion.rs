//! Typed decoder evidence and real host IPC reply validation; no GPU or model load.
use super::*;
use ferrule_common::error::ErrorChain;
use ferrule_model::decoder::KvPrepareQuiescenceUnknown;
use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::parallel::process::decoder_wire::{InputFrame, KeyFrame};
use crate::parallel::process::ipc::{self, Message};

fn context() -> CompletionContext {
    CompletionContext {
        rank: PipelineRank {
            local: ParallelRankId::new(3),
            global: ParallelRankId::new(3),
        },
        stage: "execute",
        transaction: Some(ExecutionTransactionId::new(23).unwrap()),
    }
}
fn lost_process() -> ProcessError {
    ProcessError::QuiescenceUnknown {
        cause: UnknownCause::Deadline,
        source: Box::new(ProcessError::Deadline),
        termination: ProcessTerminationReport {
            reap: ReapOutcome::NotReaped { pid: 123 },
            signal_errors: vec![std::io::Error::from_raw_os_error(5)],
            reap_error: Some(std::io::Error::from_raw_os_error(4)),
        },
    }
}

#[test]
fn decoder_loss_preserves_process_unknown_and_termination_evidence() {
    let original = lost_process();
    assert!(!original.is_command_rejection());
    let mut transport = ProcessPipelineTransport {
        ranks: BTreeMap::new(),
        unavailable: false,
    };
    let failure = transport.lost(context(), original);
    assert!(transport.unavailable);
    assert_eq!(
        completion_boundary::classify(&failure),
        CompletionClass::Unknown
    );
    let source = ErrorChain::new(&failure)
        .downcast_ref::<ProcessError>()
        .unwrap();
    let ProcessError::QuiescenceUnknown {
        cause,
        source,
        termination,
    } = source
    else {
        panic!("lost typed process evidence")
    };
    assert_eq!(*cause, UnknownCause::Deadline);
    assert!(matches!(source.as_ref(), ProcessError::Deadline));
    // The nested deadline is a rejection in isolation, not at the Unknown boundary.
    assert!(source.is_command_rejection());
    assert_eq!(termination.reap, ReapOutcome::NotReaped { pid: 123 });
    assert_eq!(termination.signal_errors[0].raw_os_error(), Some(5));
    assert_eq!(
        termination.reap_error.as_ref().unwrap().raw_os_error(),
        Some(4)
    );
    let evidence = ErrorChain::new(&failure)
        .downcast_ref::<completion_boundary::CompletionFailure>()
        .unwrap();
    assert_eq!(evidence.context.rank, context().rank);
    assert_eq!(evidence.context.transaction, context().transaction);
    assert_eq!(evidence.context.stage, "execute");
    assert!(completion_boundary::is_unknown(
        &transport.shutdown().unwrap_err()
    ));
    assert!(completion_boundary::is_unknown(
        &transport.process_stats(3).unwrap_err()
    ));
    assert!(completion_boundary::is_unknown(
        &transport
            .call(context().rank, PipelineCommand::Describe)
            .unwrap_err()
    ));
}

#[derive(Clone, Default)]
struct BoundaryLogs(Arc<Mutex<Vec<BTreeMap<String, String>>>>);
impl tracing::Subscriber for BoundaryLogs {
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
impl BoundaryLogs {
    fn assert_once(&self, outcome: &str) {
        let records = self.0.lock().unwrap();
        assert_eq!(
            records.len(),
            1,
            "one final boundary event, never preliminary success"
        );
        let fields = &records[0];
        assert!(fields.keys().all(|key| matches!(
            key.as_str(),
            "operation" | "rank" | "phase" | "bytes" | "duration" | "outcome"
        )));
        assert_eq!(fields["operation"], "pipeline.process");
        assert_eq!(fields["rank"], "0");
        assert_eq!(fields["phase"], "execute");
        assert_eq!(fields["outcome"], outcome);
        assert!(fields["duration"].parse::<u64>().is_ok());
        assert!(fields.values().all(|value| !value.contains("secret")));
    }
}

fn execute_fixture() -> (DecoderBoot, PipelineStageDescription, DecoderCommand) {
    let config = PipelineConfig {
        page_size: 4,
        max_pages: 4,
        max_positions: 16,
        max_batch_tokens: 4,
        session_capacity: 1,
        max_parameter_bytes: 1024,
        max_ack_polls: 2,
        precision: ExecutionPrecisionPolicy::f32(),
    };
    let boot = DecoderBoot {
        version: DECODER_WIRE_VERSION,
        hybrid: None,
        rank: 0,
        checkpoint: "secret-metadata-only-checkpoint".into(),
        recipe: DecoderRecipeKind::Synthetic,
        segment: SegmentFrame {
            total_layers: 1,
            layers: 0..1,
            embedding: true,
            output: true,
        },
        precision: DecoderPrecision::F32,
        device: DecoderDevice::Cpu,
        kv: KvConfigFrame::encode(config),
        experts: None,
    };
    let description = boot.description(4, 8, 1, 4).unwrap();
    let command = DecoderCommand::Execute {
        key: KeyFrame {
            transaction: 23,
            topology: 9,
            world_size: 1,
            plan: ferrule_common::ParallelismPlan::validated(1, 1, 1, 1, 1, 1).unwrap(),
            participants: vec![0],
            generation: 1,
            rank: 0,
            session: 7,
        },
        input: InputFrame::Tokens,
        cancelled: false,
    };
    (boot, description, command)
}

// A scripted, bounded host peer feeds real encoded IPC frames into the actual
// ProcessRankOwner. It owns no model or KV. Preloading the transcript avoids a
// second wire codec or a test executable that would require invoking Cargo.
fn reply_transport(
    reply: DecoderReply,
    wrong_identity: bool,
) -> (ProcessPipelineTransport, PipelineCommand) {
    let (boot, description, wire) = execute_fixture();
    let key = wire.key().unwrap().clone();
    let command = wire.decode(&description).unwrap();
    let identity = ProcessIdentity::new(
        ProcessGroupEpoch::new(1).unwrap(),
        ParallelRankId::new(0),
        ProcessOwnerInstanceId::new(1).unwrap(),
    );
    let options = ProcessOwnerConfig {
        startup_timeout: Duration::from_secs(5),
        command_timeout: Duration::from_secs(5),
        terminate_grace: Duration::from_millis(20),
        kill_grace: Duration::from_secs(1),
        ..Default::default()
    };
    let messages = [
        Message::Ready {
            identity,
            limits: options.frame_limits,
        },
        Message::Complete {
            identity: ProcessCommandIdentity {
                owner: identity,
                command: ProcessCommandId::new(if wrong_identity { 2 } else { 1 }).unwrap(),
                transaction: ExecutionTransactionId::new(key.transaction).unwrap().into(),
            },
            session: key.session,
            payload: ipc::payload(&reply, options.frame_limits.max_output_bytes).unwrap(),
        },
    ];
    let mut transcript = tempfile::NamedTempFile::new().unwrap();
    for message in messages {
        let bytes = ipc::encode(message, options.frame_limits.max_frame_bytes).unwrap();
        transcript
            .write_all(&u32::try_from(bytes.len()).unwrap().to_be_bytes())
            .unwrap();
        transcript.write_all(&bytes).unwrap();
    }
    transcript.flush().unwrap();
    let launch = ProcessLaunch::new("/bin/sh")
        .arg("-c")
        .arg("cat -- \"$1\"; exec cat > /dev/null")
        .arg("decoder-reply-fixture")
        .arg(transcript.path());
    let owner = Owner::spawn(launch, identity, &boot, options).unwrap();
    // This is the preparation projection already checked by the parent. A
    // self-consistent one-row reply is still invalid for these two input rows.
    let entry = Rank {
        owner,
        boot,
        description,
        prepared: Some((key, 2)),
    };
    (
        ProcessPipelineTransport {
            ranks: BTreeMap::from([(0, entry)]),
            unavailable: false,
        },
        command,
    )
}

#[test]
fn semantic_bad_replies_log_unknown_once_after_real_validation() {
    for (reply, expected_message) in [
        // Valid IPC envelope, wrong reply variant for Execute: decode correlation.
        (
            DecoderReply::Generation { generation: 99 },
            "unexpected decoder reply",
        ),
        (
            DecoderReply::Logits {
                rows: 2,
                width: 7,
                values: vec![0.5; 14],
            },
            "invalid logits frame shape",
        ),
        // Passes DecoderReply::decode; fails the transport's prepared-row check.
        (
            DecoderReply::Logits {
                rows: 1,
                width: 8,
                values: vec![0.5; 8],
            },
            "invalid pipeline segment output endpoint or logits shape",
        ),
    ] {
        let (boot, description, wire) = execute_fixture();
        if matches!(&reply, DecoderReply::Logits { rows: 1, .. }) {
            assert!(reply.clone().decode(&wire, &description, &boot).is_ok());
        } else {
            assert!(reply.clone().decode(&wire, &description, &boot).is_err());
        }
        let (mut transport, command) = reply_transport(reply, false);
        let logs = BoundaryLogs::default();
        let failure = tracing::subscriber::with_default(logs.clone(), || {
            transport
                .call(
                    PipelineRank {
                        local: ParallelRankId::new(0),
                        global: ParallelRankId::new(0),
                    },
                    command,
                )
                .unwrap_err()
        });
        assert_eq!(
            completion_boundary::classify(&failure),
            CompletionClass::Unknown
        );
        assert!(failure.to_string().contains(expected_message), "{failure}");
        assert!(transport.unavailable);
        assert!(transport.ranks.is_empty());
        let evidence = ErrorChain::new(&failure)
            .downcast_ref::<completion_boundary::CompletionFailure>()
            .unwrap();
        assert_eq!(evidence.context.stage, "execute");
        assert_eq!(evidence.context.rank.global.get(), 0);
        assert_eq!(evidence.context.transaction.unwrap().get(), 23);
        assert!(evidence.quarantine);
        logs.assert_once("unknown");
    }
}

#[test]
fn ipc_bad_correlation_logs_unknown_once_and_keeps_process_evidence() {
    let (mut transport, command) = reply_transport(
        DecoderReply::Logits {
            rows: 2,
            width: 8,
            values: vec![0.5; 16],
        },
        true,
    );
    let logs = BoundaryLogs::default();
    let failure = tracing::subscriber::with_default(logs.clone(), || {
        transport
            .call(
                PipelineRank {
                    local: ParallelRankId::new(0),
                    global: ParallelRankId::new(0),
                },
                command,
            )
            .unwrap_err()
    });
    assert!(matches!(
        ErrorChain::new(&failure).downcast_ref::<ProcessError>(),
        Some(ProcessError::QuiescenceUnknown {
            cause: UnknownCause::ProtocolViolation,
            ..
        })
    ));
    assert!(transport.unavailable);
    logs.assert_once("unknown");
}

#[test]
fn valid_reply_and_post_drain_cancellation_each_log_the_final_result_once() {
    for cancel_after_drain in [false, true] {
        let (mut transport, command) = reply_transport(
            DecoderReply::Logits {
                rows: 2,
                width: 8,
                values: vec![0.5; 16],
            },
            false,
        );
        let cancellation = match &command {
            PipelineCommand::Execute { cancellation, .. } => Arc::clone(cancellation),
            _ => unreachable!(),
        };
        let logs = BoundaryLogs::default();
        let seen_wait = AtomicBool::new(false);
        let result = tracing::subscriber::with_default(logs.clone(), || {
            transport.call_observed(
                PipelineRank {
                    local: ParallelRankId::new(0),
                    global: ParallelRankId::new(0),
                },
                command,
                &mut || {
                    // The first poll precedes wire encoding; later observations
                    // cannot rewrite the accepted Execute cancellation field.
                    if seen_wait.swap(true, Ordering::AcqRel) && cancel_after_drain {
                        cancellation.store(true, Ordering::Release);
                    }
                },
            )
        });
        if cancel_after_drain {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("cancelled after draining")
            );
            logs.assert_once("failed");
        } else {
            assert!(matches!(result.unwrap(), PipelineReply::Executed { .. }));
            logs.assert_once("success");
        }
        assert!(!transport.unavailable);
        assert_eq!(transport.ranks[&0].prepared.as_ref().unwrap().1, 2);
        // Test peer has no physical custody and deliberately supplies no shutdown ACK.
        transport.quarantine();
    }
}

#[test]
fn child_secondary_unknown_never_becomes_a_fenced_failure() {
    for source in [
        Error::with_cleanup(
            "cleanup",
            error("primary"),
            Err(KvPrepareQuiescenceUnknown::wrap(
                context().transaction.unwrap(),
                error("retained"),
            )),
        ),
        Error::FailureBatch {
            operation: "batch".into(),
            failures: vec![
                error("first"),
                Error::ModelSource {
                    source: Box::new(lost_process()),
                },
            ],
        },
    ] {
        let failure = handler_failure(ProcessFailureKind::Handler, source);
        assert_eq!(failure.quiescence, ProcessQuiescence::Unknown);
        assert_eq!(failure.kind, ProcessFailureKind::Handler);
        assert!(!failure.message.contains("primary"));
    }
    assert_eq!(
        handler_failure(ProcessFailureKind::Handler, error("ordinary")).quiescence,
        ProcessQuiescence::Fenced
    );
}
