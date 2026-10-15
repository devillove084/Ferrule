use super::*;

#[test]
fn cancelled_running_read_must_finish_and_release_custody_before_reuse() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let gate = Gate::new(TestPhase::IoStarted);
    let mut pipeline = fixture.pipeline(&reader, Some(gate.action.clone()));
    let handoff = pipeline.join_handoff();
    let first = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    gate.entered();
    pipeline.cancel(&first).unwrap();
    assert_eq!(pipeline.budget().reserved_bytes, 8);
    assert_eq!(
        pipeline.budget().worker_stack.reserved_bytes,
        DEFAULT_READ_WORKER_STACK_BYTES
    );
    assert!(matches!(
        pipeline.poll(&first).unwrap(),
        ReadPoll::Pending(ReadStage::Reading)
    ));
    gate.release();
    ready(&pipeline, &first);
    let completion = pipeline.take(&first).unwrap();
    assert!(matches!(
        completion.outcome(),
        ReadOutcome::Cancelled(ReadCancellation::AfterStart)
    ));
    assert!(matches!(
        pipeline.ready_for_sequence(),
        Err(ReadPipelineError::NotReady)
    ));
    drop(completion);
    pipeline.ready_for_sequence().unwrap();
    let second = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    assert!(second.operation().get() > first.operation().get());
    assert!(matches!(
        pipeline.take(&first),
        Err(ReadPipelineError::TicketMismatch)
    ));
    ready(&pipeline, &second);
    let completion = pipeline.take(&second).unwrap();
    assert_eq!(completion.payload().unwrap(), [10, 11, 12, 13, 1, 2, 3, 4]);
    drop(completion);
    pipeline.ready_for_sequence().unwrap();
    assert_eq!(handoff, pipeline.join_handoff());
    assert!(handoff.observe_worker().alive);
    assert!(!handoff.observe_worker().joined);
    assert_eq!(reader.read_counters().source_checks, 4);
    drop(pipeline);
    assert!(handoff.observe_worker().joined);
    assert!(!handoff.observe_worker().alive);
    assert_eq!(
        handoff
            .observe_budget()
            .budget
            .requested_host_reservation_bytes(),
        0
    );
}

#[test]
fn empty_read_ledger_does_not_reuse_worker_before_publication_callbacks_return() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let gate = Gate::new(TestPhase::AfterPublish);
    let mut pipeline = fixture.pipeline(&reader, Some(gate.action.clone()));
    let handoff = pipeline.join_handoff();
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    gate.entered();
    drop(pipeline.take(&ticket).unwrap());
    assert_eq!(pipeline.budget().reserved_slots, 0);
    let (sent, receive) = mpsc::sync_channel(1);
    let supervisor = thread::spawn(move || {
        pipeline.ready_for_sequence().unwrap();
        sent.send(pipeline).unwrap();
    });
    assert!(matches!(
        receive.recv_timeout(Duration::from_millis(30)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    assert!(handoff.observe_worker().alive);
    assert_eq!(
        handoff.observe_budget().budget.worker_stack.reserved_bytes,
        DEFAULT_READ_WORKER_STACK_BYTES
    );
    gate.release();
    let pipeline = receive.recv_timeout(DEADLINE).unwrap();
    supervisor.join().unwrap();
    assert_eq!(pipeline.join_handoff(), handoff);
    assert!(!handoff.observe_worker().joined);
    drop(pipeline);
    assert!(handoff.observe_worker().joined);
}

#[test]
fn panic_after_terminal_publication_is_not_an_idle_reusable_worker() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let mut pipeline = fixture.pipeline(
        &reader,
        Some(hook(|phase, _| {
            if phase == TestPhase::AfterPublish {
                panic!("controlled panic after a proved completion");
            }
        })),
    );
    let handoff = pipeline.join_handoff();
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    ready(&pipeline, &ticket);
    drop(pipeline.take(&ticket).unwrap());
    assert!(pipeline.ready_for_sequence().is_err());
    let shutdown = pipeline.shutdown();
    assert!(shutdown.join_failed);
    assert!(!shutdown.join_pending);
    assert!(shutdown.unknown.is_empty());
    assert_eq!(
        shutdown.budget.worker_stack.unknown_bytes,
        DEFAULT_READ_WORKER_STACK_BYTES
    );
    assert_eq!(
        shutdown.budget.worker_stack.reserved_bytes,
        DEFAULT_READ_WORKER_STACK_BYTES
    );
    assert!(handoff.observe_worker().joined);
    assert!(handoff.observe_worker().join_failed);
    assert!(pipeline.ready_for_sequence().is_err());
}

#[test]
fn materializer_cannot_replace_a_worker_after_recovered_idle_ledger_poison() {
    use crate::transformer::ExpertReadSchedule;
    use crate::transformer::expert_reads::tests::{Fixture as ExpertFixture, IMAGE};
    let fixture = ExpertFixture::new(1);
    let materializer = fixture.materializer(ExpertReadSchedule::Blocking);
    let sequence = materializer
        .expert_read_sequence(IMAGE, &[&fixture.metadata[0]])
        .unwrap()
        .unwrap();
    let handoff = sequence.observation_test_handoff();
    drop(sequence);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _state = handoff.shared.state.lock().unwrap();
            panic!("controlled empty-ledger poison");
        }))
        .is_err()
    );
    let poisoned = materializer.expert_read_pipeline_observation();
    assert_eq!(poisoned.status, "unknown_ledger_poisoned");
    assert_eq!(poisoned.stats.workers_created, 1);
    for _ in 0..2 {
        assert!(
            materializer
                .expert_read_sequence(IMAGE, &[&fixture.metadata[0]])
                .is_err()
        );
    }
    let stopped = materializer.expert_read_pipeline_observation();
    assert_eq!(stopped.stats.workers_created, 1);
    assert_eq!(stopped.stats.worker_reuses, 0);
    assert_eq!(stopped.stats.workers_joined, 0);
    assert_eq!(stopped.status, "quarantined");
    assert!(stopped.worker_unknown);
    assert_eq!(
        stopped.current_charge.requested_stack_bytes,
        DEFAULT_READ_WORKER_STACK_BYTES as u64
    );
    drop(materializer);
    assert!(handoff.observe_worker().joined);
}
