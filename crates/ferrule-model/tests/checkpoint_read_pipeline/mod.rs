use super::*;
use crate::checkpoint::{
    CheckpointDType, CheckpointReadExtent, CheckpointSourceFileIdentity, CheckpointTensorSlice,
    NumericFp8Encoding, NumericFp8Source,
};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::AtomicUsize;
use std::task::{Context, Poll, Wake, Waker};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
const GENERATION: SourceGeneration = SourceGeneration::new(7);
const DEADLINE: Duration = Duration::from_secs(5);

struct Fixture {
    directory: PathBuf,
    path: PathBuf,
    plan: CheckpointReadPlan,
}

impl Fixture {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "ferrule-read-pipeline-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("shard.bin");
        std::fs::write(&path, (0..32u8).collect::<Vec<_>>()).unwrap();
        let source = CheckpointSourceFileIdentity::capture(&path).unwrap();
        let plan = CheckpointReadPlan::new(
            [
                CheckpointReadExtent::new(path.clone(), 10, 4).unwrap(),
                CheckpointReadExtent::new(path.clone(), 1, 4).unwrap(),
            ],
            Arc::<[CheckpointSourceFileIdentity]>::from([source]),
        )
        .unwrap();
        Self {
            directory,
            path,
            plan,
        }
    }

    fn pipeline(
        &self,
        reader: &CheckpointTensorReader,
        hook: Option<TestHook>,
    ) -> CheckpointReadPipeline {
        CheckpointReadPipeline::start(reader.clone(), limits(16, 2), hook).unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

fn limits(bytes: u64, slots: usize) -> ReadPipelineLimits {
    ReadPipelineLimits {
        max_host_bytes: bytes,
        max_read_slots: slots,
    }
}

fn hook(action: impl Fn(TestPhase, &mut ReadTicket) + Send + Sync + 'static) -> TestHook {
    Arc::new(move |phase, ticket| {
        action(phase, ticket);
        true
    })
}

/// A finite, deterministic test barrier around the production worker/read path.
struct Gate {
    entered: Mutex<mpsc::Receiver<()>>,
    release: Arc<(Mutex<bool>, Condvar)>,
    action: TestHook,
}

impl Gate {
    fn new(at: TestPhase) -> Self {
        let (sender, entered) = mpsc::sync_channel(1);
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let worker_release = Arc::clone(&release);
        let count = AtomicUsize::new(0);
        let action = hook(move |phase, _| {
            if phase != at || count.fetch_add(1, Ordering::Relaxed) != 0 {
                return;
            }
            sender.send(()).unwrap();
            let (lock, changed) = &*worker_release;
            let (guard, timeout) = changed
                .wait_timeout_while(lock.lock().unwrap(), DEADLINE, |released| !*released)
                .unwrap();
            assert!(
                *guard && !timeout.timed_out(),
                "worker gate watchdog expired"
            );
        });
        Self {
            entered: Mutex::new(entered),
            release,
            action,
        }
    }

    fn entered(&self) {
        self.entered.lock().unwrap().recv_timeout(DEADLINE).unwrap();
    }

    fn release(&self) {
        *self.release.0.lock().unwrap() = true;
        self.release.1.notify_all();
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        self.release();
    }
}

fn ready(pipeline: &CheckpointReadPipeline, ticket: &ReadTicket) -> ReadReceipt {
    // The tool invocation and every worker gate have a finite watchdog too.
    match pipeline.wait(ticket).unwrap() {
        ReadPoll::Ready(receipt) => receipt,
        other => panic!("expected a proved terminal result, got {other:?}"),
    }
}

#[test]
fn actual_positioned_read_progresses_during_owner_window_without_poll() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let io_gate = Gate::new(TestPhase::IoStarted);
    let post_gate = Gate::new(TestPhase::PayloadRead);
    let io_hook = io_gate.action.clone();
    let post_hook = post_gate.action.clone();
    let mut pipeline = fixture.pipeline(
        &reader,
        Some(hook(move |phase, ticket| {
            io_hook(phase, ticket);
            post_hook(phase, ticket);
        })),
    );
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    io_gate.entered();
    assert_eq!(reader.read_counters().bytes_read, 0);
    assert!(matches!(
        pipeline.poll(&ticket).unwrap(),
        ReadPoll::Pending(ReadStage::Reading)
    ));
    assert!(matches!(
        pipeline.take(&ticket),
        Err(ReadPipelineError::NotReady)
    ));
    let compute_start = Instant::now();
    io_gate.release();
    // The worker completes actual read_at calls without an owner poll/wait.
    let current_compute = (0..4096u64).fold(0, |sum, x| sum ^ std::hint::black_box(x));
    std::hint::black_box(current_compute);
    post_gate.entered();
    assert_eq!(reader.read_counters().bytes_read, 8);
    let compute_end = Instant::now();
    post_gate.release();
    let receipt = ready(&pipeline, &ticket);
    assert!(receipt.io_overlap(compute_start, compute_end).unwrap() > Duration::ZERO);
    assert_eq!(receipt.io_bytes, Some(8));
    assert_eq!(receipt.payload_bytes, 8);
    assert!(receipt.submitted_at <= receipt.worker_started_at.unwrap());
    assert!(receipt.worker_started_at.unwrap() <= receipt.io_started_at.unwrap());
    assert!(receipt.io_started_at.unwrap() <= receipt.io_finished_at.unwrap());
    assert!(receipt.io_finished_at.unwrap() <= receipt.source_verified_at.unwrap());
    assert!(receipt.source_verified_at.unwrap() <= receipt.completed_at.unwrap());
    let completion = pipeline.take(&ticket).unwrap();
    assert_eq!(completion.payload().unwrap(), [10, 11, 12, 13, 1, 2, 3, 4]);
    assert!(completion.receipt().taken_at.unwrap() >= receipt.completed_at.unwrap());
    assert_eq!(pipeline.budget().reserved_bytes, 8);
    drop(completion);
    assert_eq!(pipeline.budget().reserved_bytes, 0);
}

#[test]
fn blocking_baseline_uses_identical_worker_payload_and_budget() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let mut pipeline = fixture.pipeline(&reader, None);
    let first = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    ready(&pipeline, &first);
    let baseline = pipeline.take(&first).unwrap();
    let second = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    assert!(second.operation().get() > first.operation().get());
    ready(&pipeline, &second);
    let overlap = pipeline.take(&second).unwrap();
    assert_eq!(overlap.payload(), baseline.payload());
    assert_eq!(pipeline.budget().reserved_bytes, 16);
    assert_eq!(pipeline.budget().reserved_slots, 2);
    assert!(matches!(
        pipeline.submit(GENERATION, fixture.plan.clone()),
        Err(ReadPipelineError::BudgetExhausted { .. })
    ));
    assert!(matches!(
        pipeline.take(&first),
        Err(ReadPipelineError::AlreadyTaken)
    ));
    drop(baseline);
    drop(overlap);
    assert_eq!(pipeline.budget().reserved_slots, 0);
    assert_eq!(pipeline.budget().peak_slots, 2);
}

#[test]
fn queued_cancel_never_reads_and_keeps_charge_until_taken_result_drops() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let gate = Gate::new(TestPhase::SessionOpened);
    let mut pipeline = fixture.pipeline(&reader, Some(gate.action.clone()));
    let first = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    gate.entered();
    let second = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    assert_eq!(
        pipeline.cancel(&second).unwrap(),
        ReadCancelStatus::Requested
    );
    assert_eq!(pipeline.budget().reserved_bytes, 16);
    assert!(matches!(
        pipeline.poll(&second).unwrap(),
        ReadPoll::Pending(ReadStage::Queued)
    ));
    gate.release();
    ready(&pipeline, &first);
    let receipt = ready(&pipeline, &second);
    assert_eq!(receipt.io_started_at, None);
    assert_eq!(receipt.worker_started_at, None);
    assert_eq!(reader.read_counters().bytes_read, 8);
    let cancelled = pipeline.take(&second).unwrap();
    assert!(matches!(
        cancelled.outcome(),
        ReadOutcome::Cancelled(ReadCancellation::BeforeStart)
    ));
    assert_eq!(pipeline.budget().reserved_bytes, 16);
    drop(cancelled);
    drop(pipeline.take(&first).unwrap());
    assert_eq!(pipeline.budget().reserved_bytes, 0);
}

#[test]
fn cancellation_after_start_does_not_release_running_read_or_pins() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::with_max_shards(32, 1).unwrap();
    let gate = Gate::new(TestPhase::IoStarted);
    let mut pipeline = fixture.pipeline(&reader, Some(gate.action.clone()));
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    gate.entered();
    assert_eq!(reader.read_counters().live_handles, 1);
    assert_eq!(
        pipeline.cancel(&ticket).unwrap(),
        ReadCancelStatus::Requested
    );
    assert_eq!(pipeline.budget().reserved_bytes, 8);
    assert!(matches!(
        pipeline.take(&ticket),
        Err(ReadPipelineError::NotReady)
    ));
    gate.release();
    let receipt = ready(&pipeline, &ticket);
    assert_eq!(receipt.io_bytes, Some(8));
    assert_eq!(receipt.payload_bytes, 0);
    assert!(receipt.source_verified_at.is_some());
    let result = pipeline.take(&ticket).unwrap();
    assert!(matches!(
        result.outcome(),
        ReadOutcome::Cancelled(ReadCancellation::AfterStart)
    ));
    drop(result);
    assert_eq!(pipeline.budget().reserved_slots, 0);
}

#[test]
fn cancellation_of_unclaimed_success_suppresses_payload_without_refund() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let mut pipeline = fixture.pipeline(&reader, None);
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    ready(&pipeline, &ticket);
    assert_eq!(
        pipeline.cancel(&ticket).unwrap(),
        ReadCancelStatus::Suppressed
    );
    assert_eq!(pipeline.budget().reserved_bytes, 8);
    let completion = pipeline.take(&ticket).unwrap();
    assert!(matches!(
        completion.outcome(),
        ReadOutcome::Cancelled(ReadCancellation::AfterCompletion)
    ));
    assert!(completion.payload().is_none());
    assert_eq!(completion.receipt().payload_bytes, 0);
    drop(completion);
    assert_eq!(pipeline.budget().reserved_bytes, 0);
}

#[test]
fn stale_original_plan_fails_before_payload_io_without_recapture() {
    let fixture = Fixture::new();
    std::fs::write(&fixture.path, [0; 64]).unwrap();
    let reader = CheckpointTensorReader::new(32);
    let mut pipeline = fixture.pipeline(&reader, None);
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    let receipt = ready(&pipeline, &ticket);
    assert!(receipt.io_started_at.is_none());
    assert_eq!(reader.read_counters().bytes_read, 0);
    let completion = pipeline.take(&ticket).unwrap();
    match completion.outcome() {
        ReadOutcome::Failed(error) => assert!(
            error
                .to_string()
                .contains("stale checkpoint source identity")
        ),
        other => panic!("unexpected result {other:?}"),
    }
    assert_eq!(pipeline.budget().reserved_bytes, 8);
    drop(completion);
    assert_eq!(pipeline.budget().reserved_bytes, 0);
}

#[test]
fn source_replacement_after_actual_read_cannot_escape_mandatory_post_boundary() {
    let fixture = Fixture::new();
    let path = fixture.path.clone();
    let reader = CheckpointTensorReader::new(32);
    let mut pipeline = fixture.pipeline(
        &reader,
        Some(hook(move |phase, _| {
            if phase == TestPhase::PayloadRead {
                let replacement = path.with_extension("replacement");
                std::fs::write(&replacement, [0; 32]).unwrap();
                std::fs::rename(&replacement, &path).unwrap();
            }
        })),
    );
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    let receipt = ready(&pipeline, &ticket);
    assert_eq!(receipt.io_bytes, Some(8));
    assert_eq!(receipt.payload_bytes, 0);
    assert!(receipt.source_verified_at.is_none());
    let completion = pipeline.take(&ticket).unwrap();
    assert!(matches!(completion.outcome(), ReadOutcome::Failed(_)));
    assert!(completion.payload().is_none());
    assert_eq!(pipeline.budget().reserved_bytes, 8);
    drop(completion);
}

#[test]
fn truncation_between_open_and_read_is_ordinary_failure_not_unknown() {
    let fixture = Fixture::new();
    let path = fixture.path.clone();
    let reader = CheckpointTensorReader::new(32);
    let mut pipeline = fixture.pipeline(
        &reader,
        Some(hook(move |phase, _| {
            if phase == TestPhase::SessionOpened {
                std::fs::File::options()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_len(0)
                    .unwrap();
            }
        })),
    );
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    let receipt = ready(&pipeline, &ticket);
    assert!(receipt.io_started_at.is_some());
    assert!(receipt.io_finished_at.is_some());
    assert_eq!(receipt.io_bytes, None);
    assert_eq!(pipeline.budget().unknown_bytes, 0);
    assert!(matches!(
        pipeline.take(&ticket).unwrap().outcome(),
        ReadOutcome::Failed(_)
    ));
    assert_eq!(pipeline.budget().reserved_bytes, 0);
}

#[test]
fn completion_identity_mismatch_is_quarantined_for_generation_id_and_plan() {
    for field in 0..3 {
        let fixture = Fixture::new();
        let reader = CheckpointTensorReader::new(32);
        let other_plan = CheckpointReadPlan::new(
            [CheckpointReadExtent::new(fixture.path.clone(), 0, 8).unwrap()],
            Arc::<[CheckpointSourceFileIdentity]>::from(fixture.plan.source_files()),
        )
        .unwrap();
        let mut pipeline = fixture.pipeline(
            &reader,
            Some(hook(move |phase, ticket| {
                if phase == TestPhase::BeforePublish {
                    match field {
                        0 => ticket.source_generation = SourceGeneration::new(8),
                        1 => ticket.operation = OperationId::new(ticket.operation.get() + 1000),
                        _ => ticket.plan = other_plan.clone(),
                    }
                }
            })),
        );
        let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
        assert!(matches!(
            pipeline.wait(&ticket).unwrap(),
            ReadPoll::Unknown {
                reason: ReadUnknownReason::CompletionIdentityMismatch,
                ..
            }
        ));
        assert!(matches!(
            pipeline.take(&ticket),
            Err(ReadPipelineError::UnknownCustody(_))
        ));
        assert_eq!(
            pipeline.cancel(&ticket).unwrap(),
            ReadCancelStatus::RetainedUnknown
        );
        assert!(pipeline.drain().is_empty());
        let report = pipeline.shutdown();
        assert!(!report.join_failed);
        assert_eq!(report.budget.reserved_bytes, 8);
        assert_eq!(report.budget.unknown_bytes, 8);
        assert_eq!(report.unknown.len(), 1);
        assert!(matches!(
            pipeline.submit(GENERATION, fixture.plan.clone()),
            Err(ReadPipelineError::Closed)
        ));
    }
}

#[test]
fn foreign_or_modified_owner_ticket_never_consumes_result_or_charge() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let mut pipeline = fixture.pipeline(&reader, None);
    let mut foreign_pipeline = fixture.pipeline(&reader, None);
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    let foreign = foreign_pipeline
        .submit(GENERATION, fixture.plan.clone())
        .unwrap();
    ready(&pipeline, &ticket);
    let mut stale = ticket.clone();
    stale.source_generation = SourceGeneration::new(8);
    for bad in [&stale, &foreign] {
        assert!(matches!(
            pipeline.poll(bad),
            Err(ReadPipelineError::TicketMismatch)
        ));
        assert!(matches!(
            pipeline.take(bad),
            Err(ReadPipelineError::TicketMismatch)
        ));
        assert!(matches!(
            pipeline.cancel(bad),
            Err(ReadPipelineError::TicketMismatch)
        ));
    }
    assert_eq!(pipeline.budget().reserved_bytes, 8);
    drop(pipeline.take(&ticket).unwrap());
    assert_eq!(pipeline.budget().reserved_bytes, 0);
}

#[test]
fn worker_panic_and_join_error_keep_accepted_read_charged() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let gate = Gate::new(TestPhase::PayloadRead);
    let gated = gate.action.clone();
    let mut pipeline = fixture.pipeline(
        &reader,
        Some(hook(move |phase, ticket| {
            gated(phase, ticket);
            if phase == TestPhase::PayloadRead {
                panic!("injected panic after positioned read");
            }
        })),
    );
    let first = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    gate.entered();
    let second = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    gate.release();
    for ticket in [&first, &second] {
        assert!(matches!(
            pipeline.wait(ticket).unwrap(),
            ReadPoll::Unknown {
                reason: ReadUnknownReason::WorkerPanicked,
                ..
            }
        ));
    }
    assert_eq!(reader.read_counters().bytes_read, 8);
    assert!(pipeline.drain().is_empty());
    let report = pipeline.shutdown();
    assert!(report.join_failed);
    assert_eq!(report.budget.reserved_bytes, 16);
    assert_eq!(report.budget.unknown_bytes, 16);
    assert_eq!(report.budget.unknown_slots, 2);
    assert_eq!(report.unknown.len(), 2);
    assert_eq!(pipeline.shutdown(), report);
}

#[test]
fn channel_close_without_completion_does_not_refund_accepted_ticket() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let mut pipeline = fixture.pipeline(
        &reader,
        Some(Arc::new(|phase, _| phase != TestPhase::Dequeued)),
    );
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    assert!(matches!(
        pipeline.wait(&ticket).unwrap(),
        ReadPoll::Unknown {
            reason: ReadUnknownReason::WorkerStopped,
            ..
        }
    ));
    assert_eq!(reader.read_counters().read_calls, 0);
    let report = pipeline.shutdown();
    assert!(!report.join_failed);
    assert_eq!(report.budget.reserved_bytes, 8);
    assert_eq!(report.budget.unknown_slots, 1);
}

#[test]
fn queue_is_depth_one_and_rejected_requests_never_charge() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let gate = Gate::new(TestPhase::SessionOpened);
    let mut pipeline =
        CheckpointReadPipeline::start(reader, limits(32, 4), Some(gate.action.clone())).unwrap();
    let first = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    gate.entered();
    let second = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    assert!(matches!(
        pipeline.submit(GENERATION, fixture.plan.clone()),
        Err(ReadPipelineError::QueueFull)
    ));
    assert_eq!(pipeline.budget().reserved_slots, 2);
    assert_eq!(pipeline.budget().reserved_bytes, 16);
    assert_eq!(pipeline.budget().peak_bytes, 16);
    gate.release();
    ready(&pipeline, &first);
    ready(&pipeline, &second);
    drop(pipeline.drain());
    assert_eq!(pipeline.budget().reserved_slots, 0);
}

#[test]
fn limits_and_overflow_reject_before_open_or_allocation() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    for invalid in [limits(0, 1), limits(8, 0)] {
        assert!(matches!(
            CheckpointReadPipeline::new(reader.clone(), invalid),
            Err(ReadPipelineError::InvalidConfig(_))
        ));
    }
    let mut bounded = CheckpointReadPipeline::new(reader.clone(), limits(7, 1)).unwrap();
    assert!(matches!(
        bounded.submit(GENERATION, fixture.plan.clone()),
        Err(ReadPipelineError::BudgetExhausted { .. })
    ));
    assert!(matches!(
        bounded.submit(SourceGeneration::new(0), fixture.plan.clone()),
        Err(ReadPipelineError::InvalidConfig(_))
    ));
    assert_eq!(bounded.budget().reserved_slots, 0);
    let mut tiny_reader =
        CheckpointReadPipeline::new(CheckpointTensorReader::new(7), limits(100, 1)).unwrap();
    assert!(matches!(
        tiny_reader.submit(GENERATION, fixture.plan.clone()),
        Err(ReadPipelineError::SizeLimit { .. })
    ));
    let huge = CheckpointSourceFileIdentity::for_test(
        fixture.path.clone(),
        fixture.path.clone(),
        u64::MAX,
    );
    let extent = CheckpointReadExtent::new(fixture.path.clone(), 0, u64::MAX).unwrap();
    assert!(
        CheckpointReadPlan::new(
            [extent.clone(), extent.clone()],
            Arc::<[CheckpointSourceFileIdentity]>::from([huge.clone()])
        )
        .is_err()
    );
    let giant_plan = CheckpointReadPlan::new(
        [extent],
        Arc::<[CheckpointSourceFileIdentity]>::from([huge]),
    )
    .unwrap();
    let mut giant_budget = CheckpointReadPipeline::new(
        CheckpointTensorReader::new(u64::MAX),
        limits(u64::MAX, usize::MAX),
    )
    .unwrap();
    assert!(matches!(
        giant_budget.submit(GENERATION, giant_plan),
        Err(ReadPipelineError::SizeLimit { .. })
    ));
    assert_eq!(giant_budget.budget().reserved_bytes, 0);
    assert_eq!(reader.read_counters().open_calls, 0);
}

#[test]
fn shutdown_joins_cancels_and_drains_proved_results_without_leaking() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let gate = Gate::new(TestPhase::PayloadRead);
    let mut pipeline = fixture.pipeline(&reader, Some(gate.action.clone()));
    let first = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    gate.entered();
    let second = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    pipeline.cancel(&first).unwrap();
    gate.release();
    let report = pipeline.shutdown();
    assert!(!report.join_failed);
    assert_eq!(report.budget.reserved_bytes, 16);
    assert_eq!(report.budget.unknown_slots, 0);
    assert!(report.unknown.is_empty());
    for ticket in [&first, &second] {
        assert!(matches!(pipeline.poll(ticket).unwrap(), ReadPoll::Ready(_)));
    }
    let completions = pipeline.drain();
    assert_eq!(completions.len(), 2);
    assert_eq!(pipeline.budget().reserved_slots, 2);
    drop(completions);
    assert_eq!(pipeline.budget().reserved_slots, 0);
    assert_eq!(pipeline.budget().reserved_bytes, 0);
    assert_eq!(pipeline.shutdown().budget.reserved_slots, 0);
}

#[test]
fn dropping_pipeline_joins_worker_but_keeps_caller_held_payload_custody() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let mut pipeline = fixture.pipeline(&reader, None);
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    ready(&pipeline, &ticket);
    let held = pipeline.take(&ticket).unwrap();
    let shared = Arc::clone(&pipeline.shared);
    drop(pipeline);
    assert!(shared.wake.is_closed());
    assert_eq!(shared.lock().budget.reserved_slots, 1);
    assert_eq!(held.payload().unwrap(), [10, 11, 12, 13, 1, 2, 3, 4]);
    drop(held);
    assert_eq!(shared.lock().budget.reserved_slots, 0);
    drop(reader);
    // Sessions and the worker-owned reader no longer pin any shard handles.
    assert_eq!(Arc::strong_count(&shared), 1);
}

#[derive(Default)]
struct WakeCount(AtomicUsize);
impl Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn listener_wakes_on_completion_and_on_unproved_worker_exit() {
    for stop in [false, true] {
        let fixture = Fixture::new();
        let reader = CheckpointTensorReader::new(32);
        let gate = Gate::new(TestPhase::Dequeued);
        let gated = gate.action.clone();
        let mut pipeline = fixture.pipeline(
            &reader,
            Some(Arc::new(move |phase, ticket| {
                gated(phase, ticket);
                !(stop && phase == TestPhase::Dequeued)
            })),
        );
        let mut listener = pipeline.listen();
        let wake = Arc::new(WakeCount::default());
        let waker = Waker::from(Arc::clone(&wake));
        let mut cx = Context::from_waker(&waker);
        assert!(Pin::new(&mut listener).poll(&mut cx).is_pending());
        let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
        gate.entered();
        gate.release();
        pipeline.wait(&ticket).unwrap();
        pipeline.shutdown();
        assert!(matches!(
            Pin::new(&mut listener).poll(&mut cx),
            Poll::Ready(_)
        ));
        assert!(wake.0.load(Ordering::Relaxed) > 0);
    }
}

#[test]
fn paired_numeric_fp8_plan_charges_weight_and_scales_and_preserves_order() {
    let fixture = Fixture::new();
    let mut bytes = vec![0x38, 0x40, 0x42, 0x44];
    bytes.extend(0.5f32.to_le_bytes());
    std::fs::write(&fixture.path, bytes).unwrap();
    let source = CheckpointSourceFileIdentity::capture(&fixture.path).unwrap();
    let weight = CheckpointTensorSlice {
        name: "weight".into(),
        role: crate::TensorRole::RoutedExpertDown,
        path: fixture.path.clone(),
        offset: 0,
        bytes: 4,
        dtype: CheckpointDType::F8E4M3,
        shape: vec![2, 2],
    };
    let scale = CheckpointTensorSlice {
        name: "weight_scale_inv".into(),
        role: crate::TensorRole::RoutedExpertDown,
        path: fixture.path.clone(),
        offset: 4,
        bytes: 4,
        dtype: CheckpointDType::F32,
        shape: vec![1, 1],
    };
    let source = NumericFp8Source::new(
        weight,
        scale,
        NumericFp8Encoding::E4M3FnBlock128F32,
        source.clone(),
        source,
    )
    .unwrap();
    let reader = CheckpointTensorReader::new(8);
    let paired = source.plan_tile(&reader, None, 0..2, 0..2).unwrap();
    let mut underfunded = CheckpointReadPipeline::new(reader.clone(), limits(4, 1)).unwrap();
    assert!(matches!(
        underfunded.submit(GENERATION, paired.read_plan().clone()),
        Err(ReadPipelineError::BudgetExhausted {
            requested_bytes: 8,
            ..
        })
    ));
    let mut pipeline = CheckpointReadPipeline::new(reader, limits(8, 1)).unwrap();
    let ticket = pipeline
        .submit(GENERATION, paired.read_plan().clone())
        .unwrap();
    ready(&pipeline, &ticket);
    let completion = pipeline.take(&ticket).unwrap();
    assert_eq!(completion.ticket().read_plan(), paired.read_plan());
    assert_eq!(
        completion.payload().unwrap(),
        [0x38, 0x40, 0x42, 0x44, 0, 0, 0, 0x3f]
    );
    assert_eq!(pipeline.budget().reserved_bytes, 8);
    assert_eq!(completion.receipt().payload_bytes, 8);
    drop(completion);
    assert_eq!(pipeline.budget().reserved_bytes, 0);
}

#[test]
fn adding_large_virtual_reservations_cannot_wrap_host_budget() {
    let fixture = Fixture::new();
    let bytes = isize::MAX as u64;
    let source =
        CheckpointSourceFileIdentity::for_test(fixture.path.clone(), fixture.path.clone(), bytes);
    let plan = CheckpointReadPlan::new(
        [CheckpointReadExtent::new(fixture.path.clone(), 0, bytes).unwrap()],
        Arc::<[CheckpointSourceFileIdentity]>::from([source]),
    )
    .unwrap();
    let reader = CheckpointTensorReader::new(u64::MAX);
    let gate = Gate::new(TestPhase::Dequeued);
    let mut pipeline = CheckpointReadPipeline::start(
        reader.clone(),
        limits(u64::MAX, 3),
        Some(gate.action.clone()),
    )
    .unwrap();
    let first = pipeline.submit(GENERATION, plan.clone()).unwrap();
    gate.entered();
    let second = pipeline.submit(GENERATION, plan.clone()).unwrap();
    let reserved = bytes.checked_mul(2).unwrap();
    assert_eq!(pipeline.budget().reserved_bytes, reserved);
    assert!(
        matches!(pipeline.submit(GENERATION, plan), Err(ReadPipelineError::BudgetExhausted { available_bytes, .. }) if available_bytes == u64::MAX - reserved)
    );
    pipeline.cancel(&first).unwrap();
    pipeline.cancel(&second).unwrap();
    gate.release();
    ready(&pipeline, &first);
    ready(&pipeline, &second);
    drop(pipeline.drain());
    assert_eq!(pipeline.budget().reserved_bytes, 0);
    assert_eq!(reader.read_counters().open_calls, 0);
}

#[test]
fn operation_ids_stop_at_overflow_without_recycling_zero() {
    let next = AtomicU64::new(u64::MAX - 1);
    assert_eq!(allocate_operation(&next).unwrap().get(), u64::MAX - 1);
    assert!(matches!(
        allocate_operation(&next),
        Err(ReadPipelineError::IdExhausted)
    ));
    assert!(matches!(
        allocate_operation(&next),
        Err(ReadPipelineError::IdExhausted)
    ));
    assert_eq!(next.load(Ordering::Relaxed), u64::MAX);
}

#[test]
fn shutdown_cannot_finish_or_refund_before_running_worker_releases_buffer() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let gate = Gate::new(TestPhase::PayloadRead);
    let mut pipeline = fixture.pipeline(&reader, Some(gate.action.clone()));
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    gate.entered();
    let shared = Arc::clone(&pipeline.shared);
    let (sender, finished) = mpsc::sync_channel(1);
    let owner = thread::spawn(move || {
        let report = pipeline.shutdown();
        sender.send(report).unwrap();
        pipeline
    });
    let (state, timeout) = shared
        .changed
        .wait_timeout_while(shared.lock(), DEADLINE, |state| {
            !state.entries[&ticket.operation()]
                .receipt
                .cancellation_requested
        })
        .unwrap();
    assert!(!timeout.timed_out());
    assert_eq!(state.budget.reserved_bytes, 8);
    assert!(matches!(
        state.entries[&ticket.operation()].phase,
        Phase::Reading
    ));
    assert!(matches!(
        finished.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    drop(state);
    gate.release();
    let report = finished.recv_timeout(DEADLINE).unwrap();
    assert!(!report.join_failed);
    assert_eq!(report.budget.reserved_bytes, 8);
    let mut pipeline = owner.join().unwrap();
    let completion = pipeline.take(&ticket).unwrap();
    assert!(matches!(
        completion.outcome(),
        ReadOutcome::Cancelled(ReadCancellation::AfterStart)
    ));
    drop(completion);
    assert_eq!(pipeline.budget().reserved_bytes, 0);
}

#[test]
fn worker_panic_releases_session_pins_but_not_unknown_host_charge() {
    let fixture = Fixture::new();
    let other = Fixture::new();
    let reader = CheckpointTensorReader::with_max_shards(32, 1).unwrap();
    let mut pipeline = fixture.pipeline(
        &reader,
        Some(hook(|phase, _| {
            if phase == TestPhase::PayloadRead {
                panic!("injected panic retaining no trusted completion");
            }
        })),
    );
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    assert!(matches!(
        pipeline.wait(&ticket).unwrap(),
        ReadPoll::Unknown { .. }
    ));
    assert!(pipeline.shutdown().join_failed);
    // A remaining pin would prevent eviction under the existing strict FD cap.
    let bytes = reader
        .verified_read_session(&other.plan)
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(bytes, [10, 11, 12, 13, 1, 2, 3, 4]);
    assert_eq!(reader.read_counters().peak_handles, 1);
    assert_eq!(reader.read_counters().live_handles, 1);
    assert_eq!(pipeline.budget().unknown_bytes, 8);
}

#[test]
fn retired_operations_leave_no_payload_or_ticket_cache() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let mut pipeline = CheckpointReadPipeline::new(reader, limits(8, 1)).unwrap();
    let mut previous = 0;
    for _ in 0..64 {
        let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
        assert!(ticket.operation().get() > previous);
        previous = ticket.operation().get();
        ready(&pipeline, &ticket);
        drop(pipeline.take(&ticket).unwrap());
        assert_eq!(pipeline.budget().reserved_bytes, 0);
        assert!(pipeline.shared.lock().entries.is_empty());
        assert!(matches!(
            pipeline.poll(&ticket),
            Err(ReadPipelineError::TicketMismatch)
        ));
    }
    assert_eq!(pipeline.budget().peak_bytes, 8);
    assert_eq!(pipeline.budget().peak_slots, 1);
}

#[test]
fn configured_stack_is_explicit_capped_and_shared_by_both_schedules() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let options = ReadPipelineOptions {
        worker_stack_bytes: 256 * 1024,
        max_worker_stack_bytes: 512 * 1024,
    };
    for overlap in [false, true] {
        let mut pipeline =
            CheckpointReadPipeline::with_options(reader.clone(), limits(16, 2), options).unwrap();
        assert_eq!(pipeline.options(), options);
        let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
        if overlap {
            std::hint::black_box((0..256).sum::<u64>());
        }
        ready(&pipeline, &ticket);
        let completion = pipeline.take(&ticket).unwrap();
        let budget = pipeline.budget();
        assert_eq!(budget.worker_stack.configured_bytes, 256 * 1024);
        assert_eq!(budget.worker_stack.limit_bytes, 512 * 1024);
        assert_eq!(budget.worker_stack.reserved_bytes, 256 * 1024);
        assert_eq!(budget.worker_stack.peak_bytes, 256 * 1024);
        assert_eq!(budget.worker_stack.os_overhead_bytes, None);
        assert_eq!(budget.worker_stack.unknown_bytes, 0);
        assert_eq!(budget.reserved_bytes, 8);
        assert_eq!(pipeline.shutdown().budget.worker_stack.reserved_bytes, 0);
        assert_eq!(pipeline.budget().reserved_bytes, 8);
        drop(completion);
        assert_eq!(pipeline.budget().reserved_bytes, 0);
    }
    let mut default = CheckpointReadPipeline::new(reader, limits(16, 2)).unwrap();
    assert_eq!(default.options(), ReadPipelineOptions::default());
    assert_eq!(
        default.budget().worker_stack.configured_bytes,
        DEFAULT_READ_WORKER_STACK_BYTES
    );
    default.shutdown();
}

#[test]
fn stack_options_reject_zero_over_cap_and_host_overflow_before_spawn() {
    let reader = CheckpointTensorReader::new(32);
    for options in [
        ReadPipelineOptions {
            worker_stack_bytes: 0,
            max_worker_stack_bytes: 1,
        },
        ReadPipelineOptions {
            worker_stack_bytes: 1,
            max_worker_stack_bytes: 0,
        },
        ReadPipelineOptions {
            worker_stack_bytes: 1024,
            max_worker_stack_bytes: 512,
        },
        ReadPipelineOptions {
            worker_stack_bytes: isize::MAX as usize + 1,
            max_worker_stack_bytes: usize::MAX,
        },
    ] {
        assert!(matches!(
            CheckpointReadPipeline::with_options(reader.clone(), limits(16, 2), options),
            Err(ReadPipelineError::InvalidConfig(_))
        ));
    }
    assert_eq!(reader.read_counters().open_calls, 0);
}

#[test]
fn budget_snapshots_do_not_wait_for_or_refund_live_worker_stack() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let gate = Gate::new(TestPhase::PayloadRead);
    let mut pipeline = fixture.pipeline(&reader, Some(gate.action.clone()));
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    gate.entered();
    let before = pipeline.budget();
    for _ in 0..32 {
        assert_eq!(pipeline.budget(), before);
    }
    assert_eq!(
        before.worker_stack.reserved_bytes,
        DEFAULT_READ_WORKER_STACK_BYTES
    );
    assert_eq!(before.reserved_bytes, 8);
    pipeline.cancel(&ticket).unwrap();
    assert_eq!(pipeline.budget(), before);
    gate.release();
    ready(&pipeline, &ticket);
    drop(pipeline.take(&ticket).unwrap());
    assert_eq!(pipeline.budget().reserved_bytes, 0);
    assert_eq!(
        pipeline.budget().worker_stack.reserved_bytes,
        DEFAULT_READ_WORKER_STACK_BYTES
    );
    assert_eq!(pipeline.shutdown().budget.worker_stack.reserved_bytes, 0);
}

#[test]
fn failed_join_retains_requested_stack_and_unknown_payload_charge() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(32);
    let mut pipeline = fixture.pipeline(
        &reader,
        Some(hook(|phase, _| {
            if phase == TestPhase::PayloadRead {
                panic!("injected read panic with stack custody");
            }
        })),
    );
    let ticket = pipeline.submit(GENERATION, fixture.plan.clone()).unwrap();
    assert!(matches!(
        pipeline.wait(&ticket).unwrap(),
        ReadPoll::Unknown { .. }
    ));
    let report = pipeline.shutdown();
    assert!(report.join_failed);
    assert_eq!(
        report.budget.worker_stack.reserved_bytes,
        DEFAULT_READ_WORKER_STACK_BYTES
    );
    assert_eq!(
        report.budget.worker_stack.unknown_bytes,
        DEFAULT_READ_WORKER_STACK_BYTES
    );
    assert_eq!(report.budget.unknown_bytes, 8);
    assert_eq!(pipeline.shutdown(), report);
    assert_eq!(pipeline.budget(), report.budget);
}

#[test]
fn failed_worker_spawn_proves_no_started_worker_or_accepted_operation() {
    let shared = Shared {
        state: Mutex::new(State {
            entries: BTreeMap::new(),
            accepting: true,
            working: false,
            faulted: false,
            budget: ReadBudgetSnapshot {
                worker_stack: ReadWorkerStackSnapshot {
                    configured_bytes: 1024,
                    limit_bytes: 1024,
                    reserved_bytes: 1024,
                    peak_bytes: 1024,
                    unknown_bytes: 0,
                    os_overhead_bytes: None,
                },
                ..ReadBudgetSnapshot::default()
            },
        }),
        changed: Condvar::new(),
        wake: CompletionHub::new(),
        worker_id: OnceLock::new(),
        worker_alive: AtomicBool::new(false),
        join: Mutex::new(JoinCustody::default()),
    };
    let failed = Err(std::io::Error::other("injected worker spawn failure"));
    assert!(matches!(
        finish_worker_start(&shared, failed),
        Err(ReadPipelineError::Spawn(_))
    ));
    let state = shared.lock();
    assert!(!state.accepting);
    assert!(state.entries.is_empty());
    assert_eq!(state.budget.reserved_bytes, 0);
    assert_eq!(state.budget.worker_stack.reserved_bytes, 0);
    assert_eq!(state.budget.worker_stack.unknown_bytes, 0);
    assert!(shared.wake.is_closed());
}

#[test]
fn verified_numeric_conversion_rejects_wrong_ticket_generation_plan_and_payload_length() {
    let fixture = Fixture::new();
    let path = fixture.path.clone();
    let mut packed = vec![0x38, 0x40, 0x42, 0x44];
    packed.extend(0.5f32.to_le_bytes());
    std::fs::write(&path, packed).unwrap();
    let source = CheckpointSourceFileIdentity::capture(&path).unwrap();
    let weight = CheckpointTensorSlice {
        name: "weight".into(),
        role: crate::TensorRole::AttentionQuery,
        path: path.clone(),
        offset: 0,
        bytes: 4,
        dtype: CheckpointDType::F8E4M3,
        shape: vec![2, 2],
    };
    let scale = CheckpointTensorSlice {
        name: "scale".into(),
        role: crate::TensorRole::AttentionQuery,
        path: path.clone(),
        offset: 4,
        bytes: 4,
        dtype: CheckpointDType::F32,
        shape: vec![1, 1],
    };
    let paired = NumericFp8Source::new(
        weight,
        scale,
        NumericFp8Encoding::E4M3FnBlock128F32,
        source.clone(),
        source,
    )
    .unwrap();
    let reader = CheckpointTensorReader::new(32);
    let read = paired.plan_tile(&reader, None, 0..2, 0..2).unwrap();
    let mut pipeline = fixture.pipeline(&reader, None);
    let ticket = pipeline
        .submit(GENERATION, read.read_plan().clone())
        .unwrap();
    ready(&pipeline, &ticket);
    let mut completion = pipeline.take(&ticket).unwrap();
    for field in 0..3 {
        let mut wrong = ticket.clone();
        match field {
            0 => wrong.source_generation = SourceGeneration::new(GENERATION.get() + 1),
            1 => wrong.operation = OperationId::new(ticket.operation().get() + 1000),
            _ => {
                wrong.plan = CheckpointReadPlan::new(
                    ticket.read_plan().extents().iter().rev().cloned(),
                    Arc::<[CheckpointSourceFileIdentity]>::from(ticket.read_plan().source_files()),
                )
                .unwrap()
            }
        }
        assert!(read.materialize_verified_read(&completion, &wrong).is_err());
        assert_eq!(pipeline.budget().reserved_bytes, 8);
    }
    let ReadOutcome::SourceVerified(bytes) = &mut completion.outcome else {
        panic!("expected source-verified bytes");
    };
    bytes.pop();
    assert!(
        read.materialize_verified_read(&completion, &ticket)
            .unwrap_err()
            .to_string()
            .contains("payload length")
    );
    assert_eq!(pipeline.budget().reserved_bytes, 8);
    drop(completion);
    assert_eq!(pipeline.budget().reserved_bytes, 0);
}

#[path = "reentrant.rs"]
mod reentrant;

#[cfg(unix)]
#[path = "metadata.rs"]
mod metadata;

#[cfg(unix)]
use crate::checkpoint::metadata_test_support as metadata_support;

mod reuse;
