//! Real checkpoint execution through production Boot/dispatch and private pipes.
//! The matching process_rank_child is built once per test process.
//! Set FERRULE_DECODER_CHILD to a built ferrule CLI to exercise __rank-worker
//! with the same lifecycle tests instead of the fixture's production endpoint.
#![cfg(unix)]

#[path = "support/build_process_child.rs"]
mod build_process_child;

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[path = "process_decoder/cuda.rs"]
mod cuda;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use ferrule_common::execution::*;
use ferrule_common::{
    ParallelRankId, ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology,
};
use ferrule_model::decoder::{DenseLogits, KvCommitBinding, KvEndProgress};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::{
    DecoderRecipe, LayerSegmentPlan, SegmentInput, SyntheticDecoderRecipe,
};
use ferrule_runtime::parallel::pipeline::{
    PipelineCommand, PipelineCommandKey, PipelineConfig, PipelineParallelExecutor, PipelineRank,
    PipelineReply, PipelineTransport,
};
use ferrule_runtime::parallel::process::decoder::*;
use ferrule_runtime::parallel::process::decoder_wire::{
    BatchFrame, InputFrame, KeyFrame, ProgressFrame, ReservationFrame,
};
use ferrule_runtime::parallel::process::*;
use ferrule_runtime::{Decision, SessionId, TransactionState};
use serde_json::json;

type Owner = ProcessRankOwner<DecoderBoot, DecoderCommand, DecoderReply>;

fn launch() -> ProcessLaunch {
    launch_with_timeout(30000)
}
fn launch_with_timeout(timeout_ms: u64) -> ProcessLaunch {
    if let Some(executable) = std::env::var_os("FERRULE_DECODER_CHILD") {
        return ProcessLaunch::new(executable)
            .arg("__rank-worker")
            .arg("--max-frame-bytes")
            .arg(options().frame_limits.max_frame_bytes.to_string())
            .arg("--io-timeout-ms")
            .arg(timeout_ms.to_string());
    }
    static CHILD: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    let fixture = CHILD.get_or_init(|| {
        build_process_child::build("ferrule-runtime", "example", "process_rank_child")
    });
    ProcessLaunch::new(fixture)
        .arg("decoder")
        .arg(timeout_ms.to_string())
}
fn options() -> ProcessOwnerConfig {
    ProcessOwnerConfig {
        startup_timeout: Duration::from_secs(10),
        command_timeout: Duration::from_secs(3),
        terminate_grace: Duration::from_millis(20),
        kill_grace: Duration::from_secs(1),
        ..Default::default()
    }
}
fn identity(rank: u32) -> ProcessIdentity {
    ProcessIdentity::new(
        ProcessGroupEpoch::new(1).unwrap(),
        ParallelRankId::new(rank),
        ProcessOwnerInstanceId::new(u64::from(rank) + 1).unwrap(),
    )
}
fn tx(id: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(id).unwrap()
}
fn config() -> PipelineConfig {
    PipelineConfig {
        page_size: 2,
        max_pages: 16,
        max_positions: 16,
        max_batch_tokens: 16,
        session_capacity: 3,
        max_parameter_bytes: 4096,
        max_ack_polls: 8,
        precision: ExecutionPrecisionPolicy::f32(),
    }
}
fn topology(degree: u32) -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(71),
        degree,
        ParallelRankId::new(0),
        ParallelismPlan::validated(1, 1, 1, 1, 1, degree as usize).unwrap(),
    )
    .unwrap()
}

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ferrule-process-decoder-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        let config = json!({
            "vocab_size":8, "hidden_size":4, "num_attention_heads":2,
            "num_key_value_heads":1, "head_dim":2, "intermediate_size":4,
            "num_experts":2, "experts_per_token":2, "max_position_embeddings":16,
            "rms_norm_eps":0.00001, "rope_theta":10000.0, "tie_word_embeddings":false
        });
        std::fs::write(
            path.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let recipe = SyntheticDecoderRecipe::new().build(&config).unwrap();
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for parameter in recipe
            .schema()
            .parameters()
            .iter()
            .filter(|p| p.alias_of().is_none())
        {
            let seed = parameter
                .path()
                .as_str()
                .bytes()
                .fold(0usize, |sum, byte| (sum * 31 + byte as usize) % 251);
            let start = payload.len();
            for index in 0..parameter.shape().iter().product::<usize>() {
                let value = if parameter.shape().len() == 1 {
                    0.9 + index as f32 * 0.03
                } else {
                    ((seed + index * 17 + index * index * 3) % 41) as f32 * 0.012 - 0.24
                };
                payload.extend_from_slice(&value.to_le_bytes());
            }
            header.insert(SyntheticDecoderRecipe::external_name(parameter.path()), json!({"dtype":"F32", "shape":parameter.shape(), "data_offsets":[start,payload.len()]}));
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        std::fs::write(path.join("model.safetensors"), file).unwrap();
        Self(path)
    }
    fn boots(&self, degree: u32, ep: bool) -> Vec<(ProcessIdentity, DecoderBoot)> {
        (0..degree)
            .map(|rank| {
                let layers = if degree == 1 {
                    0..2
                } else {
                    rank as usize..rank as usize + 1
                };
                let experts = ep.then(|| {
                    let first = 10 + rank * 2;
                    ExpertPlacementFrame {
                        source_scope: ferrule_common::topology::ExpertSourceScope::ExternalStage,
                        source: rank,
                        members: vec![first, first + 1],
                        entries: layers
                            .clone()
                            .flat_map(|layer| [(layer, 0, first), (layer, 1, first + 1)])
                            .collect(),
                        max_tokens: 32,
                        max_bytes: 4096,
                        devices: vec![DecoderDevice::Cpu; 2],
                        timeout_ms: 3000,
                    }
                });
                (
                    identity(rank),
                    DecoderBoot {
                        version: DECODER_WIRE_VERSION,
                        rank,
                        checkpoint: self.0.clone(),
                        recipe: DecoderRecipeKind::Synthetic,
                        segment: SegmentFrame::encode(
                            &LayerSegmentPlan::new(2, layers, rank == 0, rank + 1 == degree)
                                .unwrap(),
                        ),
                        precision: DecoderPrecision::F32,
                        device: DecoderDevice::Cpu,
                        kv: KvConfigFrame::encode(config()),
                        experts,
                    },
                )
            })
            .collect()
    }
    fn pipeline(
        &self,
        degree: u32,
        ep: bool,
    ) -> (PipelineParallelExecutor, SharedProcessPipelineTransport) {
        let boots = self.boots(degree, ep);
        let plans = boots
            .iter()
            .map(|(_, boot)| boot.segment.decode().unwrap())
            .collect();
        let transport = ProcessPipelineTransport::spawn(launch(), boots, options())
            .unwrap()
            .shared();
        let pipeline = PipelineParallelExecutor::new_with_external_expert_transport(
            topology(degree),
            plans,
            config(),
            transport.clone(),
        )
        .unwrap();
        (pipeline, transport)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn same_logits(actual: &DenseLogits, expected: &DenseLogits) {
    assert_eq!(
        (actual.rows(), actual.width()),
        (expected.rows(), expected.width())
    );
    assert!(actual.values().iter().all(|value| value.is_finite()));
    assert!(actual.values().iter().any(|value| value.abs() > 1e-4));
    for (a, b) in actual.values().iter().zip(expected.values()) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
}
fn send(owner: &mut Owner, command: DecoderCommand) -> DecoderReply {
    owner
        .execute(
            tx(command.key().map_or(1, |key| key.transaction)),
            command.session(),
            &command,
        )
        .unwrap()
}
fn ack(owner: &mut Owner, command: DecoderCommand) {
    assert!(matches!(
        send(owner, command),
        DecoderReply::Ack {
            progress: ProgressFrame::Complete
        }
    ));
}
fn stats(owner: &mut Owner) -> DecoderProcessStats {
    let DecoderReply::Stats(stats) = send(owner, DecoderCommand::Stats) else {
        panic!("expected statistics")
    };
    assert_eq!(Some(stats.pid), owner.pid());
    assert_ne!(stats.pid, std::process::id());
    stats
}
fn preparation(generation: u64) -> (KeyFrame, BatchFrame, ReservationFrame) {
    let key = KeyFrame::encode(
        &PipelineCommandKey::new(
            KvCommitBinding::new(
                tx(7),
                topology(1).topology_id(),
                topology(1).participants(),
                1, // Decision generation is distinct from the fresh sequence's zero generation.
            )
            .unwrap(),
            PipelineRank {
                local: ParallelRankId::new(0),
                global: ParallelRankId::new(0),
            },
            SessionId(19),
        )
        .unwrap(),
    );
    (
        key,
        BatchFrame {
            decode: false,
            tokens: vec![1, 2],
            positions: vec![0, 1],
            writes: vec![Some(0), Some(1)],
            full_logits: vec![true; 2],
            state_slot: 0,
            context_len: 0,
            sequence_len: 2,
            blocks: vec![0],
        },
        ReservationFrame {
            state_slot: 0,
            execution_state_slot: 0,
            positions: 0..2,
            new_pages: vec![0],
            generation,
            execution_generation: generation,
            cow: None,
        },
    )
}

#[test]
fn child_owns_real_prepare_execute_commit_finalize_and_release() {
    let fixture = Fixture::new();
    let (identity, boot) = fixture.boots(1, false).remove(0);
    let mut owner = Owner::spawn(launch(), identity, &boot, options()).unwrap();
    let DecoderReply::Generation { generation } =
        send(&mut owner, DecoderCommand::Create { session: 19 })
    else {
        panic!("missing generation")
    };
    let (key, batch, reservation) = preparation(generation);
    assert!(matches!(
        send(
            &mut owner,
            DecoderCommand::Prepare {
                key: key.clone(),
                batch,
                reservation
            }
        ),
        DecoderReply::Prepared { .. }
    ));
    assert_eq!(stats(&mut owner).active_transactions, 1);
    let DecoderReply::Logits {
        rows,
        width,
        values,
    } = send(
        &mut owner,
        DecoderCommand::Execute {
            key: key.clone(),
            input: InputFrame::Tokens,
            cancelled: false,
        },
    )
    else {
        panic!("missing real logits")
    };
    assert_eq!((rows, width, values.len()), (2, 8, 16));
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.windows(2).any(|v| (v[0] - v[1]).abs() > 1e-4));
    assert_eq!(stats(&mut owner).executions, 1);
    ack(&mut owner, DecoderCommand::Ready { key: key.clone() });
    ack(
        &mut owner,
        DecoderCommand::Install {
            key: key.clone(),
            generation: 1,
        },
    );
    ack(
        &mut owner,
        DecoderCommand::Poll {
            key: key.clone(),
            generation: 1,
        },
    );
    assert_eq!(
        stats(&mut owner).active_transactions,
        1,
        "install is not finalization"
    );
    ack(&mut owner, DecoderCommand::Publish { key: key.clone() });
    ack(
        &mut owner,
        DecoderCommand::CheckRetirement {
            key: key.clone(),
            pages: vec![],
        },
    );
    ack(
        &mut owner,
        DecoderCommand::Retire {
            key: key.clone(),
            generation: 1,
            pages: vec![],
        },
    );
    ack(&mut owner, DecoderCommand::Finish { key });
    let committed = stats(&mut owner);
    assert_eq!(
        (
            committed.active_transactions,
            committed.sessions,
            committed.resident_pages
        ),
        (0, 1, 1)
    );
    ack(
        &mut owner,
        DecoderCommand::Release {
            session: 19,
            pages: vec![0],
        },
    );
    let released = stats(&mut owner);
    assert_eq!(
        (
            released.active_transactions,
            released.sessions,
            released.resident_pages
        ),
        (0, 0, 0)
    );
    assert_eq!(released.free_pages, released.physical_pages);
    assert_eq!(owner.shutdown().unwrap().reap, ReapOutcome::Reaped);
    assert!(owner.exit_status().unwrap().success());
}

#[test]
fn process_pp2_ep2_external_source_matches_unsplit_through_prefill_decode_fork_release() {
    for ep in [false, true] {
        let fixture = Fixture::new();
        let (mut pipeline, transport) = fixture.pipeline(2, ep);
        let (mut oracle, _) = fixture.pipeline(1, false);
        let pids: Vec<_> = (0..2)
            .map(|rank| transport.process_stats(rank).unwrap().pid)
            .collect();
        assert_ne!(pids[0], pids[1]);
        assert!(!pids.contains(&std::process::id()));
        let scopes = pipeline.execution_scopes();
        assert_eq!(
            scopes
                .kv_participants()
                .iter()
                .map(ParallelRankId::get)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        for rank in 0..2 {
            let group = scopes.expert_dispatch_members(rank, 0).unwrap();
            if ep {
                let group = group.unwrap();
                assert_eq!(
                    group.source_scope(),
                    ferrule_common::topology::ExpertSourceScope::ExternalStage
                );
                assert_eq!(group.source_rank(), ParallelRankId::new(rank));
                assert_eq!(group.len(), 2);
                assert!(!group.contains(group.source_rank()));
                assert!(
                    group
                        .iter()
                        .all(|worker| !scopes.kv_participants().contains(worker))
                );
            } else {
                assert!(group.is_none());
            }
        }
        pipeline.create_session(SessionId(19)).unwrap();
        oracle.create_session(SessionId(19)).unwrap();
        for (step, tokens) in [&[1, 2, 3][..], &[4]].into_iter().enumerate() {
            let phase = if step == 0 {
                ForwardPhase::Prefill
            } else {
                ForwardPhase::Decode
            };
            let expected = oracle.forward(SessionId(19), tokens, phase).unwrap();
            let mut committed = false;
            let mut published = false;
            let output = pipeline
                .forward_observed(
                    SessionId(19),
                    tokens,
                    phase,
                    &AtomicBool::new(false),
                    |progress| {
                        committed |= progress.state == TransactionState::Decided(Decision::Commit);
                        published |= progress.state == TransactionState::Published;
                    },
                )
                .unwrap();
            assert!(committed && published);
            same_logits(&output.logits, &expected.logits);
            assert_eq!(pipeline.page_manager().stats().committed_tokens, 3 + step);
            assert_eq!(pipeline.coordinator().publication_count(), step + 1);
            assert_eq!(pipeline.outstanding(), 0);
            for rank in 0..2 {
                let stats = transport.process_stats(rank).unwrap();
                assert_eq!(stats.pid, pids[rank as usize]);
                assert_eq!(
                    (stats.executions, stats.sessions, stats.active_transactions),
                    (step + 1, 1, 0)
                );
                assert_eq!(stats.resident_pages, 2);
                assert_eq!(stats.expert_outstanding, 0);
                if ep {
                    assert_eq!(stats.experts.len(), 2);
                    for expert in &stats.experts {
                        assert!(!pids.contains(&expert.pid));
                        assert_ne!(expert.pid, std::process::id());
                        assert_eq!(expert.owned_experts.len(), 1);
                        assert!(expert.calls > 0 && expert.tokens > 0);
                        assert_eq!(expert.last_context, Some((output.transaction.get(), rank)));
                    }
                }
            }
        }
        // Fork a partial tail and force COW in both children, then retire it.
        for execution in [&mut pipeline, &mut oracle] {
            execution
                .forward(SessionId(19), &[5], ForwardPhase::Decode)
                .unwrap();
            execution
                .fork_session(SessionId(19), SessionId(20))
                .unwrap();
        }
        for (session, token) in [(SessionId(19), 6), (SessionId(20), 7)] {
            same_logits(
                &pipeline
                    .forward(session, &[token], ForwardPhase::Decode)
                    .unwrap()
                    .logits,
                &oracle
                    .forward(session, &[token], ForwardPhase::Decode)
                    .unwrap()
                    .logits,
            );
        }
        for execution in [&mut pipeline, &mut oracle] {
            execution.release_session(SessionId(19)).unwrap();
            execution.release_session(SessionId(20)).unwrap();
            assert_eq!(execution.page_manager().allocated_pages(), 0);
            assert!(!execution.is_quarantined());
        }
        for rank in 0..2 {
            let stats = transport.process_stats(rank).unwrap();
            assert_eq!(
                (
                    stats.sessions,
                    stats.active_transactions,
                    stats.resident_pages
                ),
                (0, 0, 0)
            );
            assert_eq!(stats.free_pages, stats.physical_pages);
        }
        pipeline.shutdown().unwrap();
        oracle.shutdown().unwrap();
    }
}

#[test]
fn cancellation_drains_child_then_rolls_back_without_publication() {
    let fixture = Fixture::new();
    let mut transport =
        ProcessPipelineTransport::spawn(launch(), fixture.boots(1, false), options())
            .unwrap()
            .shared();
    let rank = PipelineRank {
        local: ParallelRankId::new(0),
        global: ParallelRankId::new(0),
    };
    let PipelineReply::Generation(generation) = transport
        .call(
            rank,
            PipelineCommand::Create {
                session: SessionId(19),
            },
        )
        .unwrap()
    else {
        panic!("generation")
    };
    let (key, batch, reservation) = preparation(generation);
    let description = fixture.boots(1, false)[0]
        .1
        .description(4, 8, 1, 2)
        .unwrap();
    transport
        .call(
            rank,
            DecoderCommand::Prepare {
                key: key.clone(),
                batch,
                reservation,
            }
            .decode(&description)
            .unwrap(),
        )
        .unwrap();
    let cancellation = Arc::new(AtomicBool::new(false));
    let mut observations = 0;
    let error = transport
        .call_observed(
            rank,
            PipelineCommand::Execute {
                key: key.decode().unwrap(),
                input: SegmentInput::Tokens,
                cancellation: Arc::clone(&cancellation),
            },
            &mut || {
                observations += 1;
                if observations > 1 {
                    cancellation.store(true, Ordering::Release);
                }
            },
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("cancelled after draining"),
        "{error}"
    );
    let stats = transport.process_stats(0).unwrap();
    assert_eq!((stats.executions, stats.active_transactions), (1, 1));
    assert!(matches!(
        transport
            .call(rank, PipelineCommand::Rollback(key.decode().unwrap()))
            .unwrap(),
        PipelineReply::Ack(KvEndProgress::Complete)
    ));
    assert_eq!(transport.process_stats(0).unwrap().active_transactions, 0);
    transport
        .call(
            rank,
            PipelineCommand::Release {
                session: SessionId(19),
                pages: vec![],
            },
        )
        .unwrap();
    assert!(!transport.is_quarantined());
    transport.shutdown().unwrap();
}

#[test]
fn child_shutdown_with_prepared_custody_is_unknown_not_a_fence() {
    let fixture = Fixture::new();
    let (identity, boot) = fixture.boots(1, false).remove(0);
    let mut owner = Owner::spawn(launch(), identity, &boot, options()).unwrap();
    let DecoderReply::Generation { generation } =
        send(&mut owner, DecoderCommand::Create { session: 19 })
    else {
        panic!("generation")
    };
    let (key, batch, reservation) = preparation(generation);
    send(
        &mut owner,
        DecoderCommand::Prepare {
            key,
            batch,
            reservation,
        },
    );
    let error = owner.shutdown().unwrap_err();
    let ProcessError::QuiescenceUnknown {
        cause: UnknownCause::Handler,
        source,
        ..
    } = error
    else {
        panic!("preserve the child's shutdown failure, not a protocol error")
    };
    assert!(matches!(
        source.as_ref(),
        ProcessError::RemoteFailure {
            failure: ProcessHandlerError {
                kind: ProcessFailureKind::Panic,
                quiescence: ProcessQuiescence::Unknown,
                ..
            }
        }
    ));
    assert!(owner.shutdown().unwrap_err().is_quiescence_unknown());
    assert!(matches!(
        owner.execute(tx(8), 0, &DecoderCommand::Stats),
        Err(ProcessError::OwnerUnavailable)
    ));
}

#[test]
fn stopped_prepared_child_keeps_observing_until_absolute_deadline_then_quarantines() {
    let fixture = Fixture::new();
    let (identity, boot) = fixture.boots(1, false).remove(0);
    let mut options = options();
    options.command_timeout = Duration::from_millis(150);
    let mut owner = Owner::spawn(launch(), identity, &boot, options).unwrap();
    let DecoderReply::Generation { generation } =
        send(&mut owner, DecoderCommand::Create { session: 19 })
    else {
        panic!("generation")
    };
    let (key, batch, reservation) = preparation(generation);
    send(
        &mut owner,
        DecoderCommand::Prepare {
            key: key.clone(),
            batch,
            reservation,
        },
    );
    assert_eq!(stats(&mut owner).active_transactions, 1);
    let pid = owner.pid().unwrap();
    #[expect(
        unsafe_code,
        reason = "suspend only the private test child with active KV custody"
    )]
    unsafe {
        assert_eq!(libc::kill(pid as i32, libc::SIGSTOP), 0);
    }
    let start = std::time::Instant::now();
    let mut observed_wait = false;
    let error = owner
        .execute_observed(
            tx(7),
            19,
            &DecoderCommand::Execute {
                key,
                input: InputFrame::Tokens,
                cancelled: false,
            },
            &mut || {
                observed_wait |= start.elapsed() >= Duration::from_millis(50);
            },
        )
        .unwrap_err();
    assert!(
        observed_wait,
        "observer must run during the pipe wait, not just at its edges"
    );
    assert!(matches!(
        error,
        ProcessError::QuiescenceUnknown {
            cause: UnknownCause::Deadline,
            termination: ProcessTerminationReport {
                reap: ReapOutcome::Reaped,
                ..
            },
            ..
        }
    ));
    assert!(start.elapsed() < Duration::from_secs(2));
    assert!(owner.shutdown().unwrap_err().is_quiescence_unknown());
    assert!(matches!(
        owner.execute(tx(8), 0, &DecoderCommand::Stats),
        Err(ProcessError::OwnerUnavailable)
    ));
}

#[test]
fn stale_decoder_envelopes_are_rejected_without_consuming_prepared_custody() {
    let fixture = Fixture::new();
    let (identity, boot) = fixture.boots(1, false).remove(0);
    let mut owner = Owner::spawn(launch(), identity, &boot, options()).unwrap();
    let DecoderReply::Generation { generation } =
        send(&mut owner, DecoderCommand::Create { session: 19 })
    else {
        panic!("generation")
    };
    let (key, batch, reservation) = preparation(generation);
    send(
        &mut owner,
        DecoderCommand::Prepare {
            key: key.clone(),
            batch,
            reservation,
        },
    );
    let command = DecoderCommand::Execute {
        key: key.clone(),
        input: InputFrame::Tokens,
        cancelled: false,
    };
    for (transaction, session) in [(8, 19), (7, 20)] {
        let error = owner
            .execute(tx(transaction), session, &command)
            .unwrap_err();
        assert!(matches!(
            error,
            ProcessError::RemoteFailure {
                failure: ProcessHandlerError {
                    kind: ProcessFailureKind::CommandDecode,
                    quiescence: ProcessQuiescence::Fenced,
                    ..
                }
            }
        ));
        assert_eq!(owner.state(), ProcessOwnerState::Ready);
        let stats = stats(&mut owner);
        assert_eq!((stats.executions, stats.active_transactions), (0, 1));
    }
    ack(&mut owner, DecoderCommand::Rollback { key });
    ack(
        &mut owner,
        DecoderCommand::Release {
            session: 19,
            pages: vec![],
        },
    );
    owner.shutdown().unwrap();
}

#[test]
fn lost_child_quarantines_transport_without_replay_or_retirement() {
    let fixture = Fixture::new();
    let (mut pipeline, transport) = fixture.pipeline(1, false);
    pipeline
        .forward(SessionId(19), &[1, 2], ForwardPhase::Prefill)
        .unwrap();
    let pid = transport.process_stats(0).unwrap().pid;
    #[expect(
        unsafe_code,
        reason = "fault injection into the private test child only"
    )]
    unsafe {
        assert_eq!(libc::kill(pid as i32, libc::SIGKILL), 0);
    }
    assert!(
        pipeline
            .forward(SessionId(19), &[3], ForwardPhase::Decode)
            .is_err()
    );
    assert!(transport.is_quarantined());
    assert!(pipeline.is_quarantined());
    assert_eq!(pipeline.page_manager().stats().committed_tokens, 2);
    assert_eq!(pipeline.coordinator().publication_count(), 1);
    assert!(pipeline.page_manager().allocated_pages() > 0);
    assert!(pipeline.release_session(SessionId(19)).is_err());
    assert!(transport.process_stats(0).is_err());
    assert!(pipeline.shutdown().is_err());
}

#[test]
fn idle_decoder_and_bucketless_experts_keep_the_same_children() {
    let fixture = Fixture::new();
    let (identity, mut boot) = fixture.boots(1, true).remove(0);
    boot.experts.as_mut().unwrap().timeout_ms = 150;
    let mut config = options();
    config.command_timeout = Duration::from_millis(150);
    let mut owner = Owner::spawn(launch_with_timeout(150), identity, &boot, config).unwrap();
    let initial = stats(&mut owner);
    assert_eq!(initial.experts.len(), 2);
    assert!(initial.experts.iter().all(|e| e.calls == 0));
    std::thread::sleep(Duration::from_millis(450));
    let after = stats(&mut owner);
    assert_eq!(initial.pid, after.pid);
    for (before, after) in initial.experts.iter().zip(&after.experts) {
        assert_eq!(before.pid, after.pid);
        assert_eq!(
            after.calls, 0,
            "no bucket or heartbeat execution was invented"
        );
    }
    let DecoderReply::Generation { generation } =
        send(&mut owner, DecoderCommand::Create { session: 19 })
    else {
        panic!("generation")
    };
    let (key, batch, reservation) = preparation(generation);
    assert!(matches!(
        send(
            &mut owner,
            DecoderCommand::Prepare {
                key: key.clone(),
                batch,
                reservation
            }
        ),
        DecoderReply::Prepared { .. }
    ));
    std::thread::sleep(Duration::from_millis(450));
    assert!(matches!(
        send(
            &mut owner,
            DecoderCommand::Execute {
                key: key.clone(),
                input: InputFrame::Tokens,
                cancelled: false
            }
        ),
        DecoderReply::Logits { .. }
    ));
    let executed = stats(&mut owner);
    assert_eq!(executed.executions, 1);
    assert!(executed.experts.iter().all(|e| e.calls > 0));
    ack(&mut owner, DecoderCommand::Rollback { key });
    ack(
        &mut owner,
        DecoderCommand::Release {
            session: 19,
            pages: vec![],
        },
    );
    std::thread::sleep(Duration::from_millis(450));
    assert_eq!(owner.shutdown().unwrap().reap, ReapOutcome::Reaped);
    assert!(owner.exit_status().unwrap().success());
}

#[test]
fn local_oversize_keeps_decoder_transport_and_prepared_custody_usable() {
    let fixture = Fixture::new();
    let mut config = options();
    config.frame_limits.max_command_bytes = 1024;
    let transport = ProcessPipelineTransport::spawn(launch(), fixture.boots(1, false), config)
        .unwrap()
        .shared();
    assert_local_oversize_keeps_prepared_custody(transport, &fixture.boots(1, false)[0].1);
}

fn assert_local_oversize_keeps_prepared_custody(
    mut transport: SharedProcessPipelineTransport,
    boot: &DecoderBoot,
) {
    let rank = PipelineRank {
        local: ParallelRankId::new(0),
        global: ParallelRankId::new(0),
    };
    let initial = transport.process_stats(0).unwrap();
    let PipelineReply::Generation(generation) = transport
        .call(
            rank,
            PipelineCommand::Create {
                session: SessionId(19),
            },
        )
        .unwrap()
    else {
        panic!("generation")
    };
    let (key, batch, reservation) = preparation(generation);
    let description = boot.description(4, 8, 1, 2).unwrap();
    transport
        .call(
            rank,
            DecoderCommand::Prepare {
                key: key.clone(),
                batch,
                reservation,
            }
            .decode(&description)
            .unwrap(),
        )
        .unwrap();
    let error = transport
        .call(
            rank,
            PipelineCommand::Release {
                session: SessionId(19),
                pages: vec![KvPageId(0); 2048],
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("exceeds 1024 bytes"), "{error}");
    assert!(!error.to_string().contains("quiescence unknown"), "{error}");
    assert!(!transport.is_quarantined());
    let after = transport.process_stats(0).unwrap();
    assert_eq!(after.pid, initial.pid);
    assert_eq!(
        (after.executions, after.active_transactions, after.sessions),
        (0, 1, 1)
    );
    assert!(matches!(
        transport
            .call(
                rank,
                PipelineCommand::Execute {
                    key: key.decode().unwrap(),
                    input: SegmentInput::Tokens,
                    cancellation: Arc::new(AtomicBool::new(false))
                }
            )
            .unwrap(),
        PipelineReply::Executed { .. }
    ));
    let executed = transport.process_stats(0).unwrap();
    assert_eq!(executed.executions, 1);
    assert!(executed.experts.iter().all(|expert| expert.calls > 0));
    transport
        .call(rank, PipelineCommand::Rollback(key.decode().unwrap()))
        .unwrap();
    transport
        .call(
            rank,
            PipelineCommand::Release {
                session: SessionId(19),
                pages: vec![],
            },
        )
        .unwrap();
    let after = transport.process_stats(0).unwrap();
    assert_eq!((after.active_transactions, after.sessions), (0, 0));
    transport.shutdown().unwrap();
}

#[test]
fn oversized_ep_bucket_is_fenced_and_experts_accept_the_next_command() {
    let fixture = Fixture::new();
    let (identity, boot) = fixture.boots(1, true).remove(0);
    let mut config = options();
    config.frame_limits.max_command_bytes = 1024;
    let mut owner = Owner::spawn(launch(), identity, &boot, config).unwrap();
    let initial = stats(&mut owner);
    let DecoderReply::Generation { generation } =
        send(&mut owner, DecoderCommand::Create { session: 19 })
    else {
        panic!("generation")
    };
    let (key, mut batch, mut reservation) = preparation(generation);
    batch.tokens = vec![1; 8];
    batch.positions = (0..8).collect();
    batch.writes = (0..8).map(Some).collect();
    batch.full_logits = vec![true; 8];
    batch.sequence_len = 8;
    batch.blocks = vec![0, 1, 2, 3];
    reservation.positions = 0..8;
    reservation.new_pages = vec![0, 1, 2, 3];
    assert!(matches!(
        send(
            &mut owner,
            DecoderCommand::Prepare {
                key: key.clone(),
                batch,
                reservation
            }
        ),
        DecoderReply::Prepared { .. }
    ));
    let error = owner
        .execute(
            tx(key.transaction),
            key.session,
            &DecoderCommand::Execute {
                key: key.clone(),
                input: InputFrame::Tokens,
                cancelled: false,
            },
        )
        .unwrap_err();
    assert!(
        matches!(&error, ProcessError::RemoteFailure { failure } if failure.quiescence == ProcessQuiescence::Fenced),
        "{error:?}"
    );
    assert!(error.to_string().contains("exceeds 1024 bytes"), "{error}");
    assert_eq!(owner.state(), ProcessOwnerState::Ready);
    let after = stats(&mut owner);
    assert_eq!(after.pid, initial.pid);
    assert_eq!(after.active_transactions, 1);
    for (before, after) in initial.experts.iter().zip(&after.experts) {
        assert_eq!(before.pid, after.pid);
        assert_eq!(
            (after.calls, after.tokens),
            (0, 0),
            "oversized bucket reached child"
        );
    }
    ack(&mut owner, DecoderCommand::Rollback { key });
    ack(
        &mut owner,
        DecoderCommand::Release {
            session: 19,
            pages: vec![],
        },
    );
    let DecoderReply::Generation { generation } =
        send(&mut owner, DecoderCommand::Create { session: 20 })
    else {
        panic!("generation")
    };
    let (mut key, batch, reservation) = preparation(generation);
    key.transaction = 8;
    key.session = 20;
    assert!(matches!(
        send(
            &mut owner,
            DecoderCommand::Prepare {
                key: key.clone(),
                batch,
                reservation
            }
        ),
        DecoderReply::Prepared { .. }
    ));
    assert!(matches!(
        send(
            &mut owner,
            DecoderCommand::Execute {
                key: key.clone(),
                input: InputFrame::Tokens,
                cancelled: false
            }
        ),
        DecoderReply::Logits { .. }
    ));
    assert!(
        stats(&mut owner)
            .experts
            .iter()
            .all(|expert| expert.calls == 2)
    );
    ack(&mut owner, DecoderCommand::Rollback { key });
    ack(
        &mut owner,
        DecoderCommand::Release {
            session: 20,
            pages: vec![],
        },
    );
    owner.shutdown().unwrap();
}

#[test]
fn capacity_wire_roundtrip_and_old_boot_rejection_precede_factory() {
    let fixture = Fixture::new();
    let (_, cpu) = fixture.boots(1, false).remove(0);
    for device in [DecoderDevice::Cpu, DecoderDevice::Cuda { ordinal: 3 }] {
        let mut boot = cpu.clone();
        boot.device = device;
        boot.kv = KvConfigFrame::encode_for_device(config(), device).unwrap();
        let encoded = serde_json::to_value(&boot).unwrap();
        let decoded: DecoderBoot = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, boot);
        let stage = decoded.stage_boot().unwrap();
        let expected = config().max_pages * if device == DecoderDevice::Cpu { 1 } else { 2 };
        assert_eq!(stage.config.max_pages, config().max_pages);
        assert_eq!(stage.physical_pages, expected);
        assert_eq!(
            decoded.description(4, 8, 1, 2).unwrap().physical_pages,
            expected
        );
        for pages in [
            0,
            expected + 1,
            if device == DecoderDevice::Cpu {
                expected * 2
            } else {
                expected / 2
            },
        ] {
            let mut invalid = boot.clone();
            invalid.kv.physical_pages = pages;
            assert!(invalid.stage_boot().is_err());
        }
        let mut missing = encoded;
        missing["kv"]
            .as_object_mut()
            .unwrap()
            .remove("physical_pages");
        assert!(serde_json::from_value::<DecoderBoot>(missing).is_err());
        boot.version = 1;
        assert!(boot.stage_boot().is_err());
        let called = std::cell::Cell::new(false);
        assert!(
            DecoderChild::initialize(
                ProcessBoot {
                    identity: identity(boot.rank),
                    limits: options().frame_limits,
                    config: serde_json::to_value(&boot).unwrap(),
                },
                |_| {
                    called.set(true);
                    unreachable!("old Boot must be rejected before loading/CUDA")
                }
            )
            .is_err()
        );
        assert!(!called.get());
    }
    assert_eq!(DECODER_WIRE_VERSION, 3);
    assert_eq!(PROCESS_PROTOCOL_VERSION, 1);
}

#[test]
fn process_cpu_single_logical_page_two_tokens_keeps_exact_capacity() {
    process_single_logical_page(DecoderDevice::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires one GPU and CUDA process_rank_child; --test-threads=1"]
fn process_cuda_single_logical_page_two_tokens_keeps_exact_capacity() {
    process_single_logical_page(DecoderDevice::Cuda { ordinal: 0 });
}

fn process_single_logical_page(device: DecoderDevice) {
    let fixture = Fixture::new();
    let cfg = PipelineConfig {
        max_pages: 1,
        max_positions: 2,
        max_batch_tokens: 2,
        session_capacity: 1,
        ..config()
    };
    let mut boots = fixture.boots(2, false);
    let physical = cfg
        .physical_pages(matches!(device, DecoderDevice::Cuda { .. }))
        .unwrap();
    for (_, boot) in &mut boots {
        boot.device = device;
        boot.kv = KvConfigFrame::encode_for_device(cfg, boot.device).unwrap();
    }
    let plans = boots
        .iter()
        .map(|(_, boot)| boot.segment.decode().unwrap())
        .collect();
    let transport = ProcessPipelineTransport::spawn(launch(), boots, options())
        .unwrap()
        .shared();
    let mut pipeline = PipelineParallelExecutor::new_with_external_expert_transport(
        topology(2),
        plans,
        cfg,
        transport.clone(),
    )
    .unwrap();
    pipeline
        .forward(SessionId(91), &[1], ForwardPhase::Prefill)
        .unwrap();
    pipeline
        .forward(SessionId(91), &[2], ForwardPhase::Decode)
        .unwrap();
    assert_eq!(pipeline.page_manager().max_pages(), 1);
    for rank in 0..2 {
        let stats = transport.process_stats(rank).unwrap();
        assert_eq!(
            (stats.physical_pages, stats.resident_pages, stats.free_pages),
            (physical, 1, physical - 1)
        );
    }
    assert!(
        pipeline
            .forward(SessionId(91), &[3], ForwardPhase::Decode)
            .is_err()
    );
    pipeline.release_session(SessionId(91)).unwrap();
    for rank in 0..2 {
        assert_eq!(transport.process_stats(rank).unwrap().free_pages, physical);
    }
    pipeline.shutdown().unwrap();
}

#[test]
fn external_source_boot_topology_roundtrip_and_invalid_identity_rejection() {
    use ferrule_common::topology::{ExpertDispatchMembers, ExpertSourceScope};
    let fixture = Fixture::new();
    let topology = topology(2);
    let mut scopes = topology.execution_scopes(0).unwrap();
    for (_, boot) in fixture.boots(2, true) {
        let json = serde_json::to_value(&boot).unwrap();
        assert_eq!(json["experts"]["source_scope"], "external_stage");
        let decoded: DecoderBoot = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(decoded, boot);
        let frame = decoded.experts.as_ref().unwrap();
        assert_eq!(frame.source, decoded.rank);
        assert!(!frame.members.contains(&frame.source));
        decoded.stage_boot().unwrap();
        let description = decoded.description(4, 8, 1, 2).unwrap();
        let group = description.expert_group.as_ref().unwrap();
        scopes
            .attach_expert_dispatch_members(
                ExpertDispatchMembers::new_with_scope(
                    &topology,
                    0,
                    decoded.rank,
                    frame.source_scope,
                    group.source_rank,
                    group.members.iter().copied(),
                )
                .unwrap(),
            )
            .unwrap();
        assert!(
            description
                .validate_expert_source(ExpertSourceScope::Member, group.source_rank)
                .is_err()
        );
        let mut missing = json;
        missing["experts"]
            .as_object_mut()
            .unwrap()
            .remove("source_scope");
        assert!(serde_json::from_value::<DecoderBoot>(missing).is_err());
        for axis in 0..8 {
            let mut invalid = boot.clone();
            let frame = invalid.experts.as_mut().unwrap();
            match axis {
                0 => frame.source = boot.rank ^ 1,
                1 => frame.source = frame.members[0],
                2 => frame.source = u32::MAX,
                3 => frame.members[0] = boot.rank,
                4 => frame.source_scope = ExpertSourceScope::Member,
                5 => {
                    frame.source_scope = ExpertSourceScope::Member;
                    frame.source = frame.members[0];
                }
                6 => invalid.version = 1,
                7 => invalid.version = 2,
                _ => unreachable!(),
            }
            assert!(invalid.stage_boot().is_err(), "axis {axis}");
            assert!(
                DecoderChild::initialize(
                    ProcessBoot {
                        identity: identity(boot.rank),
                        limits: options().frame_limits,
                        config: serde_json::to_value(&invalid).unwrap(),
                    },
                    |_| panic!("invalid source/version reached factory: {axis}"),
                )
                .is_err()
            );
        }
    }
    assert_eq!(
        scopes
            .kv_participants()
            .iter()
            .map(ParallelRankId::get)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    for scope in [ExpertSourceScope::Member, ExpertSourceScope::ExternalStage] {
        assert_eq!(
            serde_json::from_str::<ExpertSourceScope>(&serde_json::to_string(&scope).unwrap())
                .unwrap(),
            scope
        );
    }
}

#[test]
fn expert_external_source_spoof_quarantines_real_process_without_replay() {
    type ExpertOwner = ProcessRankOwner<ExpertBoot, ExpertCommand, ExpertReply>;
    let fixture = Fixture::new();
    // Exercise both PP caller ranks and spoof the envelope source, token source,
    // or both consistently. A worker ID is not a permitted caller identity.
    for stage in 0..2 {
        for axis in 0..4 {
            let boot = fixture.boots(2, true).remove(stage as usize).1;
            let worker = boot.experts.as_ref().unwrap().members[0];
            let mut owner = ExpertOwner::spawn(
                launch(),
                identity(worker),
                &ExpertBoot {
                    expert_boot: boot,
                    owner: worker,
                },
                options(),
            )
            .unwrap();
            let token = ExpertTokenFrame {
                transaction: 71,
                sequence: 19,
                source: stage,
                row: 0,
                route: 0,
                layer: stage as usize,
                expert: 0,
                weight: 0.5,
                values: vec![0.1; 4],
            };
            let valid = ExpertCommand::Compute {
                transaction: 71,
                source: stage,
                layer: stage as usize,
                tokens: vec![token],
            };
            let ExpertReply::Results {
                owner: actual,
                tokens,
            } = owner.execute(tx(71), 0, &valid).unwrap()
            else {
                panic!("real expert result")
            };
            assert_eq!(actual, worker);
            assert_eq!(
                (tokens[0].transaction, tokens[0].source, tokens[0].sequence),
                (71, stage, 19)
            );
            assert!(tokens[0].values.iter().all(|value| value.is_finite()));
            let mut spoof = valid.clone();
            let ExpertCommand::Compute { source, tokens, .. } = &mut spoof else {
                unreachable!()
            };
            match axis {
                0 => *source = stage ^ 1,
                1 => tokens[0].source = worker,
                2 => {
                    *source = worker;
                    tokens[0].source = worker;
                }
                3 => {
                    *source = u32::MAX;
                    tokens[0].source = u32::MAX;
                }
                _ => unreachable!(),
            }
            let error = owner.execute(tx(71), 0, &spoof).unwrap_err();
            assert!(error.is_quiescence_unknown(), "{error}");
            assert!(
                error
                    .to_string()
                    .contains("expert source identity mismatch"),
                "{error}"
            );
            assert!(matches!(
                owner.execute(tx(71), 0, &valid),
                Err(ProcessError::OwnerUnavailable)
            ));
            assert!(owner.shutdown().unwrap_err().is_quiescence_unknown());
        }
    }
}
