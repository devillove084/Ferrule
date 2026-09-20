//! Failures after real CUDA work, not fake GPU workers. SIGSTOP is a host
//! process stall; these tests do not claim to inject a wedged GPU/DMA engine.
use super::*;

#[expect(
    unsafe_code,
    reason = "fault injection only targets still-owned private test children"
)]
fn signal(pid: u32, signal: i32) {
    unsafe {
        assert_eq!(libc::kill(pid as i32, signal), 0);
    }
}
fn active_gpu_owner(fixture: &Fixture, probe: &NoParentCuda) -> (Owner, KeyFrame, u32) {
    let (identity, boot) = boots(
        &fixture.0,
        DecoderRecipeKind::Synthetic,
        2,
        1,
        config(),
        true,
        false,
    )
    .remove(0);
    let mut options = gpu_options();
    options.command_timeout = Duration::from_millis(500);
    let mut owner = Owner::spawn(gpu_launch(), identity, &boot, options).unwrap();
    let pid = owner.pid().unwrap();
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
    let DecoderReply::Logits { values, .. } = send(
        &mut owner,
        DecoderCommand::Execute {
            key: key.clone(),
            input: InputFrame::Tokens,
            cancelled: false,
        },
    ) else {
        panic!("real GPU logits")
    };
    assert!(values.iter().any(|v| v.abs() > 1e-4));
    let stats = stats(&mut owner);
    assert_eq!((stats.executions, stats.active_transactions), (1, 1));
    probe.check();
    (owner, key, pid)
}
fn unknown_reaped(
    owner: &mut Owner,
    pid: u32,
    error: ProcessError,
    cause: UnknownCause,
    probe: &NoParentCuda,
) {
    eprintln!("GPU child failure: {error:?}");
    let ProcessError::QuiescenceUnknown {
        cause: actual,
        termination,
        ..
    } = error
    else {
        panic!("failure must be Unknown, not a fence")
    };
    assert_eq!(actual, cause);
    assert_eq!(termination.reap, ReapOutcome::Reaped);
    assert!(termination.signal_errors.is_empty());
    assert!(termination.reap_error.is_none());
    assert!(owner.shutdown().unwrap_err().is_quiescence_unknown());
    assert!(matches!(
        owner.execute(tx(8), 0, &DecoderCommand::Stats),
        Err(ProcessError::OwnerUnavailable)
    ));
    wait_gone(pid);
    probe.check();
}

#[test]
#[ignore = "CUDA CLI and one GPU; real executed custody plus host SIGSTOP deadline"]
fn gpu_process_deadline_after_execute_is_unknown_and_reaped() {
    let probe = NoParentCuda::new();
    gpu_uuids(1);
    let fixture = Fixture::new();
    let (mut owner, key, pid) = active_gpu_owner(&fixture, &probe);
    signal(pid, libc::SIGSTOP);
    let start = Instant::now();
    let mut observed_wait = false;
    let error = owner
        .execute_observed(tx(7), 19, &DecoderCommand::Ready { key }, &mut || {
            observed_wait |= start.elapsed() > Duration::from_millis(100);
        })
        .unwrap_err();
    assert!(observed_wait);
    assert!(start.elapsed() < Duration::from_secs(5));
    unknown_reaped(&mut owner, pid, error, UnknownCause::Deadline, &probe);
}

#[test]
#[ignore = "CUDA CLI and one GPU; real executed physical custody must reject shutdown"]
fn gpu_process_active_custody_shutdown_is_unknown_not_retirement() {
    let probe = NoParentCuda::new();
    gpu_uuids(1);
    let fixture = Fixture::new();
    let (mut owner, _, pid) = active_gpu_owner(&fixture, &probe);
    let error = owner.shutdown().unwrap_err();
    let ProcessError::QuiescenceUnknown { source, .. } = &error else {
        panic!("Unknown shutdown")
    };
    assert!(
        matches!(
            source.as_ref(),
            ProcessError::RemoteFailure {
                failure: ProcessHandlerError {
                    kind: ProcessFailureKind::Panic,
                    quiescence: ProcessQuiescence::Unknown,
                    ..
                }
            }
        ),
        "must retain the child's actual custody failure: {error:?}"
    );
    unknown_reaped(&mut owner, pid, error, UnknownCause::Handler, &probe);
}

fn failed_pipeline(mut pipeline: ProcessPipeline, victim: u32, fault: i32, probe: &NoParentCuda) {
    pipeline.forward(SOURCE, &[1, 2, 3], ForwardPhase::Prefill, probe);
    let before = pipeline.executor.coordinator().publication_count();
    signal(victim, fault);
    let start = Instant::now();
    let error = pipeline
        .executor
        .forward(SOURCE, &[4], ForwardPhase::Decode)
        .unwrap_err();
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "nested loss/deadline must not hang the PP owner"
    );
    eprintln!("lost GPU child {victim}: {error}");
    assert!(
        error.to_string().to_lowercase().contains("unknown"),
        "{error}"
    );
    assert!(pipeline.executor.is_quarantined());
    assert!(pipeline.transport.is_quarantined());
    assert_eq!(pipeline.executor.coordinator().publication_count(), before);
    assert_eq!(pipeline.executor.page_manager().stats().committed_tokens, 3);
    assert!(pipeline.executor.page_manager().allocated_pages() > 0);
    assert!(pipeline.executor.release_session(SOURCE).is_err());
    assert!(pipeline.executor.shutdown().is_err());
    assert!(pipeline.transport.process_stats(0).is_err());
    let pids = pipeline.pids();
    drop(pipeline);
    for pid in pids {
        wait_gone(pid);
    }
    probe.check();
}

#[test]
#[ignore = "CUDA CLI and two GPUs; SIGKILL a PP child after published GPU prefill"]
fn gpu_process_pp2_child_loss_quarantines_without_replay_or_retirement() {
    let probe = NoParentCuda::new();
    gpu_uuids(2);
    let fixture = Fixture::new();
    let pipeline = ProcessPipeline::new(
        boots(
            &fixture.0,
            DecoderRecipeKind::Synthetic,
            2,
            2,
            config(),
            true,
            false,
        ),
        config(),
        &probe,
    );
    let victim = pipeline.initial[1].pid;
    failed_pipeline(pipeline, victim, libc::SIGKILL, &probe);
}

#[test]
#[ignore = "CUDA CLI and six GPUs; kill PP group leader with live GPU EP descendants"]
fn gpu_process_pp_leader_loss_terminates_all_ep_descendants() {
    let probe = NoParentCuda::new();
    gpu_uuids(6);
    let fixture = Fixture::new();
    let pipeline = ProcessPipeline::new(
        boots(
            &fixture.0,
            DecoderRecipeKind::Synthetic,
            2,
            2,
            config(),
            true,
            true,
        ),
        config(),
        &probe,
    );
    let victim = pipeline.initial[0].pid;
    failed_pipeline(pipeline, victim, libc::SIGKILL, &probe);
}

#[test]
#[ignore = "CUDA CLI and four visible GPUs; stalled EP child must timeout its PP owner"]
fn gpu_process_expert_deadline_propagates_unknown_without_deadlock() {
    let probe = NoParentCuda::new();
    gpu_uuids(4);
    let fixture = Fixture::new();
    let mut configs = boots(
        &fixture.0,
        DecoderRecipeKind::Synthetic,
        2,
        1,
        config(),
        true,
        true,
    );
    configs[0].1.experts.as_mut().unwrap().timeout_ms = 2000;
    let pipeline = ProcessPipeline::new(configs, config(), &probe);
    let victim = pipeline.initial[0].experts[0].pid;
    failed_pipeline(pipeline, victim, libc::SIGSTOP, &probe);
}

#[test]
#[ignore = "CUDA CLI and four visible GPUs; lose independent EP owner after real GPU routing"]
fn gpu_process_expert_child_loss_propagates_unknown_and_reaps_children() {
    let probe = NoParentCuda::new();
    gpu_uuids(4);
    let fixture = Fixture::new();
    let pipeline = ProcessPipeline::new(
        boots(
            &fixture.0,
            DecoderRecipeKind::Synthetic,
            2,
            1,
            config(),
            true,
            true,
        ),
        config(),
        &probe,
    );
    let victim = pipeline.initial[0].experts[0].pid;
    failed_pipeline(pipeline, victim, libc::SIGKILL, &probe);
}
