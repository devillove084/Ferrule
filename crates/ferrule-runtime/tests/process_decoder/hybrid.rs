//! Real tiny hybrid segments in the existing process child/dispatcher.
use ferrule_model::ModelExecutionBackend;
use ferrule_runtime::parallel::pipeline::PhysicalKvLayout;
#[path = "../support/hybrid_fixture.rs"]
mod fixture;
use super::*;
use ferrule_common::execution::KvLayoutSchema;
use ferrule_model::decoder::HybridStateSchema;
use fixture::Fixture as HybridFixture;

fn hybrid_launch() -> ProcessLaunch {
    static CHILD: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    ProcessLaunch::new(CHILD.get_or_init(|| {
        build_process_child::build("ferrule-runtime", "example", "process_rank_child")
    }))
    .arg("hybrid_decoder")
}
fn hybrid_config() -> PipelineConfig {
    PipelineConfig {
        max_positions: 32,
        max_parameter_bytes: 1 << 20,
        ..config()
    }
}
fn boots(f: &HybridFixture, degree: u32) -> Vec<(ProcessIdentity, DecoderBoot)> {
    let resources = f.resources();
    (0..degree)
        .map(|rank| {
            let plan = LayerSegmentPlan::new(
                4,
                rank as usize * 4 / degree as usize..(rank as usize + 1) * 4 / degree as usize,
                rank == 0,
                rank + 1 == degree,
            )
            .unwrap();
            let kv = KvConfigFrame::encode_for_hybrid(hybrid_config(), DecoderDevice::Cpu).unwrap();
            (
                identity(rank),
                DecoderBoot {
                    version: DECODER_WIRE_VERSION,
                    rank,
                    checkpoint: f.dir.clone(),
                    recipe: DecoderRecipeKind::Qwen35DenseHybrid,
                    segment: SegmentFrame::encode(&plan),
                    precision: DecoderPrecision::F32,
                    device: DecoderDevice::Cpu,
                    kv,
                    experts: None,
                    hybrid: Some(HybridBootFrame::for_resources(&resources, &plan, None).unwrap()),
                },
            )
        })
        .collect()
}
fn close(actual: &[f32], expected: &serde_json::Value) {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        let b = b.as_f64().unwrap() as f32;
        assert!((a - b).abs() <= 3e-5 + 3e-5 * b.abs(), "{a} != {b}");
    }
}

#[test]
fn process_hybrid_pp2_prefill_decode_fork_cow_and_release_match_oracle() {
    let f = HybridFixture::new();
    let resources = f.resources();
    let mut pipeline = PipelineParallelExecutor::new_hybrid_process(
        topology(2),
        hybrid_config(),
        &resources,
        hybrid_launch(),
        boots(&f, 2),
        options(),
    )
    .unwrap();
    let prefill = pipeline
        .forward(SessionId(1), &[1, 3, 2], ForwardPhase::Prefill)
        .unwrap();
    for row in 0..3 {
        close(prefill.logits.row(row).unwrap(), &f.oracle["logits"][row]);
    }
    pipeline.fork_session(SessionId(1), SessionId(2)).unwrap();
    for session in [SessionId(2), SessionId(1)] {
        for (token, row) in [(5, 3), (7, 4)] {
            let output = pipeline
                .forward(session, &[token], ForwardPhase::Decode)
                .unwrap();
            close(output.logits.row(0).unwrap(), &f.oracle["logits"][row]);
        }
    }
    assert_eq!(pipeline.coordinator().publication_count(), 5);
    assert_eq!(pipeline.execution_scopes().kv_participants().len(), 2);
    let schema = HybridStateSchema::from_spec(resources.spec(), 4).unwrap();
    let bytes = schema
        .kv_planes(2, 32)
        .unwrap()
        .checked_page_bytes()
        .unwrap() as u64;
    assert_eq!(
        pipeline.physical_kv_bytes().unwrap(),
        bytes
            * hybrid_config()
                .physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cpu))
                .unwrap() as u64
    );
    for session in [SessionId(2), SessionId(1)] {
        pipeline.release_session(session).unwrap();
    }
    assert_eq!(pipeline.page_manager().stats().committed_tokens, 0);
    assert_eq!(pipeline.outstanding(), 0);
    pipeline.shutdown().unwrap();
}

#[test]
fn process_hybrid_executed_working_state_rollback_keeps_source_pristine() {
    let f = HybridFixture::new();
    let (identity, boot) = boots(&f, 1).remove(0);
    let mut owner = Owner::spawn(hybrid_launch(), identity, &boot, options()).unwrap();
    let DecoderReply::Generation { generation } =
        send(&mut owner, DecoderCommand::Create { session: 19 })
    else {
        panic!("generation");
    };
    let mut previous = None;
    for transaction in [7, 8] {
        let (mut key, mut batch, reservation) = preparation(generation);
        key.transaction = transaction;
        batch.tokens = vec![1, 3];
        send(
            &mut owner,
            DecoderCommand::Prepare {
                key: key.clone(),
                batch,
                reservation,
            },
        );
        let DecoderReply::Logits {
            values,
            rows,
            width,
        } = send(
            &mut owner,
            DecoderCommand::Execute {
                key: key.clone(),
                input: InputFrame::Tokens,
                cancelled: false,
            },
        )
        else {
            panic!("logits");
        };
        assert_eq!((rows, width), (2, 11));
        close(&values[..11], &f.oracle["logits"][0]);
        close(&values[11..], &f.oracle["logits"][1]);
        if let Some(before) = previous {
            assert_eq!(values, before);
        }
        previous = Some(values);
        ack(&mut owner, DecoderCommand::Rollback { key });
        let s = stats(&mut owner);
        assert_eq!((s.active_transactions, s.resident_pages), (0, 0));
        assert_eq!(s.free_pages, s.physical_pages);
    }
    ack(
        &mut owner,
        DecoderCommand::Release {
            session: 19,
            pages: vec![],
        },
    );
    assert_eq!(owner.shutdown().unwrap().reap, ReapOutcome::Reaped);
}

#[test]
fn hybrid_boot_rejects_geometry_capacity_and_untransportable_logits_before_factory() {
    let f = HybridFixture::new();
    let (_, good) = boots(&f, 1).remove(0);
    let roundtrip: DecoderBoot =
        serde_json::from_value(serde_json::to_value(&good).unwrap()).unwrap();
    assert_eq!(roundtrip, good);
    good.validate_frame_limits(options().frame_limits).unwrap();
    for axis in 0..7 {
        let mut boot = good.clone();
        match axis {
            0 => boot.version = 3,
            1 => boot.kv.physical_pages /= 2,
            2 => boot
                .hybrid
                .as_mut()
                .unwrap()
                .layers
                .pop()
                .map(drop)
                .unwrap(),
            3 => boot.hybrid.as_mut().unwrap().vocabulary = 248320,
            4 => boot.kv.max_batch_tokens = usize::MAX,
            5 => boot.hybrid = None,
            6 => boot.precision = DecoderPrecision::Bf16Compatibility,
            _ => unreachable!(),
        }
        let called = std::cell::Cell::new(false);
        assert!(
            DecoderChild::initialize(
                ProcessBoot {
                    identity: identity(0),
                    limits: options().frame_limits,
                    config: serde_json::to_value(&boot).unwrap(),
                },
                |_| {
                    called.set(true);
                    panic!("invalid boot reached factory");
                }
            )
            .is_err(),
            "axis {axis}"
        );
        assert!(!called.get());
    }
    let mut bad = boots(&f, 2);
    bad[1].1.hybrid.as_mut().unwrap().vocabulary = 248320;
    let error = PipelineParallelExecutor::new_hybrid_process(
        topology(2),
        hybrid_config(),
        &f.resources(),
        ProcessLaunch::new("must-not-be-spawned"),
        bad,
        options(),
    )
    .err()
    .unwrap();
    assert!(!error.to_string().contains("No such file"));
}

#[test]
fn hybrid_rebound_schema_mismatch_is_not_accepted_by_child_factory() {
    let f = HybridFixture::new();
    let (_, mut boot) = boots(&f, 1).remove(0);
    let HybridLayerFrame::GatedDeltaNet { kernel, .. } =
        &mut boot.hybrid.as_mut().unwrap().layers[0]
    else {
        unreachable!()
    };
    *kernel += 1;
    assert!(DecoderChild::hybrid_factory(&boot, &f.resources()).is_err());
}

#[cfg(target_os = "linux")]
#[test]
fn lost_hybrid_child_retains_unknown_and_never_replays_rollback() {
    let f = HybridFixture::new();
    let (identity, boot) = boots(&f, 1).remove(0);
    let mut owner = Owner::spawn(hybrid_launch(), identity, &boot, options()).unwrap();
    let DecoderReply::Generation { generation } =
        send(&mut owner, DecoderCommand::Create { session: 19 })
    else {
        panic!("generation");
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
    send(
        &mut owner,
        DecoderCommand::Execute {
            key: key.clone(),
            input: InputFrame::Tokens,
            cancelled: false,
        },
    );
    let pid = owner.pid().unwrap();
    #[expect(
        unsafe_code,
        reason = "fault injection into private child with active hybrid custody"
    )]
    unsafe {
        assert_eq!(libc::kill(pid as i32, libc::SIGKILL), 0);
    }
    let error = owner
        .execute(
            tx(key.transaction),
            key.session,
            &DecoderCommand::Ready { key: key.clone() },
        )
        .unwrap_err();
    assert!(error.is_quiescence_unknown());
    assert!(matches!(
        owner.execute(
            tx(key.transaction),
            key.session,
            &DecoderCommand::Rollback { key }
        ),
        Err(ProcessError::OwnerUnavailable)
    ));
    assert!(owner.shutdown().unwrap_err().is_quiescence_unknown());
}

#[test]
fn generic_process_transport_does_not_admit_hybrid_without_compact_layout() {
    let f = HybridFixture::new();
    let error = ProcessPipelineTransport::spawn(
        ProcessLaunch::new("must-not-be-spawned"),
        boots(&f, 2),
        options(),
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("compact layout"));
}

#[cfg(target_os = "linux")]
#[test]
fn hybrid_stopped_child_observation_does_not_extend_original_deadline() {
    let f = HybridFixture::new();
    let (identity, boot) = boots(&f, 1).remove(0);
    let mut limits = options();
    limits.command_timeout = Duration::from_millis(150);
    let mut owner = Owner::spawn(hybrid_launch(), identity, &boot, limits).unwrap();
    let DecoderReply::Generation { generation } =
        send(&mut owner, DecoderCommand::Create { session: 19 })
    else {
        panic!("generation");
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
    let pid = owner.pid().unwrap();
    #[expect(
        unsafe_code,
        reason = "suspend only private child with hybrid working custody"
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
                key: key.clone(),
                input: InputFrame::Tokens,
                cancelled: false,
            },
            &mut || {
                observed_wait |= start.elapsed() >= Duration::from_millis(50);
            },
        )
        .unwrap_err();
    assert!(observed_wait);
    assert!(matches!(
        error,
        ProcessError::QuiescenceUnknown {
            cause: UnknownCause::Deadline,
            ..
        }
    ));
    assert!(start.elapsed() < Duration::from_secs(2));
    assert!(matches!(
        owner.execute(tx(7), 19, &DecoderCommand::Rollback { key }),
        Err(ProcessError::OwnerUnavailable)
    ));
    assert!(owner.shutdown().unwrap_err().is_quiescence_unknown());
}
