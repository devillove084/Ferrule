//! Device/transport seams exercised without CUDA or a second pipeline journal.
//! Numerical CPU/EP equivalence remains covered by pipeline_decoder.

use std::error::Error as StdError;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};

use ferrule_common::execution::{
    ExecutionBatch, ExecutionCapabilities, ExecutionSequence, ExecutionTransactionId, ForwardMode,
    ForwardPhase, KvBindingMode, KvBlockId, KvPageId, KvReservationView, KvWriteSlot,
    LogitsRequest, LogitsRowPolicy, StateSlot,
};
use ferrule_common::{
    Error, ParallelRankId, ParallelTopologyId, ParallelismPlan, Result, ValidatedParallelTopology,
};
use ferrule_model::decoder::{
    CpuKvView, CpuPagedKvBackend, CpuPagedKvPool, DecoderKvBackend, DecoderKvCapacity,
    DecoderKvCommitBackend, DecoderKvPageStatus, DecoderKvPrepare, DenseLogits,
    GenericDecoderSequenceState, KvCommitBinding, KvEndProgress, KvPrepareQuiescenceUnknown,
    PackedDecoderBatch, PagedKvBackend, PagedKvTransactionHandle, StandardGqaPlanes,
};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::{
    HostRows, LayerSegmentPlan, RowsDType, RowsShape, SegmentInput, SegmentOutput,
    UnsupportedOperator,
};
use ferrule_runtime::parallel::data::ReplicaWorker;
use ferrule_runtime::parallel::pipeline::{
    BoxedPipelineStageWorker, PipelineCommand, PipelineCommandKey, PipelineConfig,
    PipelineExecutionContext, PipelineParallelExecutor, PipelineRank, PipelineReply, PipelineStage,
    PipelineStageBoot, PipelineStageDescription, PipelineStageProgram, PipelineStageWorker,
    PipelineTransport,
};
use ferrule_runtime::{SessionId, TransactionState};

fn config() -> PipelineConfig {
    PipelineConfig {
        page_size: 2,
        max_pages: 8,
        max_positions: 8,
        max_batch_tokens: 8,
        session_capacity: 3,
        max_parameter_bytes: 1024,
        precision: ExecutionPrecisionPolicy::f32(),
        max_ack_polls: 2,
    }
}
fn rank(id: u32) -> PipelineRank {
    PipelineRank {
        local: ParallelRankId::new(id),
        global: ParallelRankId::new(id),
    }
}
fn topology() -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(501),
        2,
        rank(0).global,
        ParallelismPlan::validated(1, 1, 1, 1, 1, 2).unwrap(),
    )
    .unwrap()
}
fn plans() -> Vec<LayerSegmentPlan> {
    vec![
        LayerSegmentPlan::new(2, 0..1, true, false).unwrap(),
        LayerSegmentPlan::new(2, 1..2, false, true).unwrap(),
    ]
}
fn boots() -> Vec<PipelineStageBoot> {
    plans()
        .into_iter()
        .enumerate()
        .map(|(index, plan)| PipelineStageBoot {
            rank: rank(index as u32),
            plan,
            config: config(),
            program_spec: vec![index as u8, 7],
        })
        .collect()
}
fn capabilities() -> ExecutionCapabilities {
    ExecutionCapabilities {
        max_batch_tokens: 8,
        max_sequences: 1,
        max_prefill_query_tokens_per_sequence: 8,
        max_decode_query_tokens_per_sequence: 1,
        max_top_k: None,
        supports_prefill: true,
        supports_decode: true,
        supports_mixed: false,
        full_logits_width: NonZeroU32::new(4),
        kv_binding_mode: KvBindingMode::Paged,
        logits_row_policy: LogitsRowPolicy::Any,
    }
}
fn fault(message: &str) -> Error {
    Error::Execution {
        message: message.into(),
    }
}
fn unsupported() -> Error {
    Error::ModelSource {
        source: Box::new(UnsupportedOperator::new(
            "test program",
            "unsupported device capability",
        )),
    }
}
fn is_unsupported(error: &(dyn StdError + 'static)) -> bool {
    error.is::<UnsupportedOperator>() || error.source().is_some_and(is_unsupported)
}

#[derive(Clone, Copy, Default)]
enum PrepareFailure {
    #[default]
    None,
    Validation,
    Cleaned,
    Unknown,
    UntypedUnknown,
}

#[derive(Default)]
struct Probe {
    initialized: Mutex<Vec<ThreadId>>,
    program_shutdown: AtomicUsize,
    backend_shutdown: AtomicUsize,
    executed: AtomicUsize,
    unsupported: AtomicBool,
    prepare_failure: Mutex<PrepareFailure>,
    prepare_calls: AtomicUsize,
    hide_retained_count: AtomicBool,
    fail_program_shutdown: AtomicBool,
}

// A deliberately distinct, non-Send view. The program cannot accidentally
// satisfy a hardcoded CpuKvView bound; only this adapter unwraps the CPU oracle.
struct DeviceView {
    cpu: CpuKvView,
    owner: Rc<ThreadId>,
}
struct DeviceBackend {
    cpu: CpuPagedKvBackend,
    owner: Rc<ThreadId>,
    probe: Arc<Probe>,
    retained: Option<PagedKvTransactionHandle>,
}
impl DeviceBackend {
    fn here(&self) {
        assert_eq!(*self.owner, thread::current().id());
    }
}
impl DecoderKvBackend for DeviceBackend {
    type SequenceState = GenericDecoderSequenceState;
    type Transaction = PagedKvTransactionHandle;
    type KvView = DeviceView;
    fn configure_capacity(&mut self, n: usize) -> Result<()> {
        self.cpu.configure_capacity(n)
    }
    fn prepare(&mut self, request: DecoderKvPrepare<'_>) -> Result<Self::Transaction> {
        self.here();
        self.probe.prepare_calls.fetch_add(1, Ordering::Relaxed);
        let failure = std::mem::take(&mut *self.probe.prepare_failure.lock().unwrap());
        if matches!(failure, PrepareFailure::Validation) {
            return Err(fault("injected prepare validation failure"));
        }
        let id = request.transaction;
        let transaction = self.cpu.prepare(request)?;
        match failure {
            PrepareFailure::None => Ok(transaction),
            PrepareFailure::Cleaned => {
                let mut transaction = Some(transaction);
                assert_eq!(
                    self.cpu.rollback(&mut transaction)?,
                    KvEndProgress::Complete
                );
                assert!(transaction.is_none());
                Err(fault("injected prepare failure after proven cleanup"))
            }
            PrepareFailure::Unknown | PrepareFailure::UntypedUnknown => {
                self.retained = Some(transaction);
                let source = fault("injected prepare quiescence unknown");
                Err(if matches!(failure, PrepareFailure::Unknown) {
                    KvPrepareQuiescenceUnknown::wrap(id, source)
                } else {
                    source
                })
            }
            PrepareFailure::Validation => unreachable!(),
        }
    }
    fn enter(
        &mut self,
        tx: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
        states: &mut [Self::SequenceState],
    ) -> Result<()> {
        self.here();
        self.cpu.enter(tx, batch, states)
    }
    fn active_view(&mut self, tx: &mut Self::Transaction) -> Result<DeviceView> {
        self.here();
        Ok(DeviceView {
            cpu: self.cpu.active_view(tx)?,
            owner: Rc::clone(&self.owner),
        })
    }
    fn leave(&mut self, tx: &mut Self::Transaction) -> Result<()> {
        self.here();
        self.cpu.leave(tx)
    }
    fn commit(&mut self, _: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        panic!("pipeline must use cohort commit")
    }
    fn rollback(&mut self, tx: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        self.here();
        self.cpu.rollback(tx)
    }
    fn release(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.here();
        self.cpu.release(pages)
    }
    fn preempt(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.cpu.preempt(pages)
    }
    fn restore(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.cpu.restore(pages)
    }
    fn capacity(&self) -> DecoderKvCapacity {
        self.here();
        let mut capacity = self.cpu.capacity();
        if self.retained.is_some() && self.probe.hide_retained_count.load(Ordering::Acquire) {
            // Exercise the typed error independently of the capacity fallback.
            capacity.active_transactions = 0;
        }
        capacity
    }
    fn page_status(&self, page: KvPageId) -> DecoderKvPageStatus {
        self.cpu.page_status(page)
    }
    fn shutdown(&mut self) -> Result<()> {
        self.here();
        self.probe.backend_shutdown.fetch_add(1, Ordering::Relaxed);
        self.cpu.shutdown()
    }
}
impl DecoderKvCommitBackend for DeviceBackend {
    fn preflight_commit_ready(
        &self,
        tx: &Self::Transaction,
        binding: &KvCommitBinding,
        rank: ParallelRankId,
        states: &[Self::SequenceState],
    ) -> Result<()> {
        self.here();
        self.cpu.preflight_commit_ready(tx, binding, rank, states)
    }
    fn commit_batch(&self, tx: &Self::Transaction) -> Result<&PackedDecoderBatch> {
        self.cpu.commit_batch(tx)
    }
    fn install_commit(
        &mut self,
        tx: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        self.here();
        self.cpu.install_commit(tx, generation)
    }
    fn poll_install_ack(
        &mut self,
        tx: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        self.cpu.poll_install_ack(tx, generation)
    }
    fn abort_prepared(
        &mut self,
        tx: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        self.cpu.abort_prepared(tx, generation)
    }
    fn publish_committed(&mut self, tx: &Self::Transaction) {
        self.here();
        self.cpu.publish_committed(tx);
    }
    fn preflight_retirement(&self, tx: &Self::Transaction, pages: &[KvPageId]) -> Result<()> {
        self.cpu.preflight_retirement(tx, pages)
    }
    fn retire_prepared(
        &mut self,
        tx: &mut Self::Transaction,
        generation: u64,
        pages: &[KvPageId],
    ) -> Result<KvEndProgress> {
        self.cpu.retire_prepared(tx, generation, pages)
    }
    fn finish_prepared(&mut self, tx: Self::Transaction) {
        self.cpu.finish_prepared(tx);
    }
}

struct DeviceProgram {
    plan: LayerSegmentPlan,
    owner: Rc<ThreadId>,
    probe: Arc<Probe>,
}
impl PipelineStageProgram for DeviceProgram {
    type KvView = DeviceView;
    fn plan(&self) -> &LayerSegmentPlan {
        &self.plan
    }
    fn execute(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        view: &mut DeviceView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        assert_eq!(*self.owner, thread::current().id());
        assert!(Rc::ptr_eq(&self.owner, &view.owner));
        assert_eq!(context.transaction, view.cpu.transaction());
        assert_eq!(context.sequence_ids.len(), 1);
        (context.check_active)(context.transaction)?;
        view.cpu.validate_batch(batch)?;
        if self.probe.unsupported.load(Ordering::Acquire) {
            return Err(unsupported());
        }
        self.probe.executed.fetch_add(1, Ordering::Relaxed);
        // Deterministic host values exercise only the runtime seam, not model math.
        let values = match input {
            SegmentInput::Tokens => {
                assert!(self.plan.owns_embedding());
                batch
                    .token_ids()
                    .iter()
                    .flat_map(|&id| [id as f32, id as f32 + 1.0])
                    .collect::<Vec<_>>()
            }
            SegmentInput::Hidden { next_layer, rows } => {
                assert_eq!(next_layer, self.plan.layers().start);
                assert_eq!(rows.shape(), RowsShape::new(batch.len(), 2)?);
                rows.into_values()
            }
        };
        if self.plan.owns_output() {
            let logits = values
                .chunks_exact(2)
                .flat_map(|row| [row[0], row[1], row[0] + 2.0, row[1] + 2.0])
                .collect::<Vec<_>>();
            Ok(SegmentOutput::Logits(DenseLogits::new(
                batch.len(),
                4,
                logits,
            )?))
        } else {
            Ok(SegmentOutput::Hidden {
                next_layer: self.plan.layers().end,
                rows: HostRows::new(
                    RowsShape::new(batch.len(), 2)?,
                    RowsDType::F32,
                    None,
                    values,
                )?,
            })
        }
    }
    fn shutdown(&mut self) -> Result<()> {
        assert_eq!(*self.owner, thread::current().id());
        self.probe.program_shutdown.fetch_add(1, Ordering::Relaxed);
        if self.probe.fail_program_shutdown.load(Ordering::Acquire) {
            Err(fault("program shutdown"))
        } else {
            Ok(())
        }
    }
}
fn stage(
    boot: &PipelineStageBoot,
    probe: Arc<Probe>,
) -> Result<PipelineStage<DeviceBackend, DeviceProgram>> {
    let owner = Rc::new(thread::current().id());
    probe.initialized.lock().unwrap().push(*owner);
    let planes = StandardGqaPlanes::new(boot.plan.layer_count(), 1, 2, 2, 8, boot.config.dtype())?;
    let backend = DeviceBackend {
        cpu: PagedKvBackend::new(CpuPagedKvPool::from_strategy(&planes, 8)?),
        owner: Rc::clone(&owner),
        probe: Arc::clone(&probe),
        retained: None,
    };
    let program = DeviceProgram {
        plan: boot.plan.clone(),
        owner,
        probe,
    };
    let description = PipelineStageDescription {
        plan: boot.plan.clone(),
        config: boot.config,
        hidden: 2,
        vocabulary: 4,
        kv_heads: 1,
        head_dim: 2,
        expert_group: None,
    };
    PipelineStage::new(program, backend, description, capabilities())
}

#[test]
fn typed_program_and_distinct_non_send_view_use_the_existing_coordinator() {
    let probe = Arc::new(Probe::default());
    let make = Arc::clone(&probe);
    let parent = thread::current().id();
    let mut pipeline = PipelineParallelExecutor::new_with_program(
        topology(),
        plans(),
        config(),
        move |rank, plan| {
            stage(
                &PipelineStageBoot {
                    rank,
                    plan,
                    config: config(),
                    program_spec: vec![],
                },
                make,
            )
        },
    )
    .unwrap();
    assert!(
        probe
            .initialized
            .lock()
            .unwrap()
            .iter()
            .all(|&id| id != parent)
    );
    let output = pipeline
        .forward(SessionId(1), &[1, 2, 3], ForwardPhase::Prefill)
        .unwrap();
    assert_eq!(output.logits.row(0).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    pipeline.fork_session(SessionId(1), SessionId(2)).unwrap();
    pipeline
        .forward(SessionId(2), &[2], ForwardPhase::Decode)
        .unwrap();
    pipeline
        .forward(SessionId(1), &[1], ForwardPhase::Decode)
        .unwrap();
    assert_eq!(pipeline.coordinator().publication_count(), 3);
    assert_eq!(pipeline.coordinator().retained_transaction_count(), 0);
    assert_eq!(pipeline.outstanding(), 0);
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.page_manager().allocated_pages(), 0);
    assert_eq!(probe.program_shutdown.load(Ordering::Relaxed), 2);
    assert_eq!(probe.backend_shutdown.load(Ordering::Relaxed), 2);
}

#[test]
fn dynamic_factory_consumes_owned_boot_specs_inside_owners() {
    let probe = Arc::new(Probe::default());
    let make = Arc::clone(&probe);
    let mut pipeline =
        PipelineParallelExecutor::new_with_factory(topology(), boots(), move |boot| {
            assert_eq!(boot.program_spec, [boot.rank.local.get() as u8, 7]);
            let stage = stage(&boot, make)?.map_program(|program| {
                Box::new(program) as Box<dyn PipelineStageProgram<KvView = DeviceView>>
            });
            Ok(PipelineStageWorker::new(&boot, stage)?.boxed())
        })
        .unwrap();
    pipeline
        .forward(SessionId(3), &[1], ForwardPhase::Prefill)
        .unwrap();
    pipeline.shutdown().unwrap();
    assert_eq!(probe.executed.load(Ordering::Relaxed), 2);
    assert_eq!(probe.backend_shutdown.load(Ordering::Relaxed), 2);
}

#[test]
fn startup_failure_preserves_typed_source_and_shuts_down_previously_created_owners() {
    let probe = Arc::new(Probe::default());
    let make = Arc::clone(&probe);
    let result = PipelineParallelExecutor::new_with_factory(topology(), boots(), move |boot| {
        if boot.rank.local.get() == 1 {
            return Err(unsupported());
        }
        Ok(PipelineStageWorker::new(&boot, stage(&boot, make)?)?.boxed())
    });
    let error = result.err().expect("rank one must fail startup");
    assert!(is_unsupported(&error));
    assert_eq!(probe.program_shutdown.load(Ordering::Relaxed), 1);
    assert_eq!(probe.backend_shutdown.load(Ordering::Relaxed), 1);
}

#[test]
fn rejected_boot_shuts_down_both_returned_and_previously_started_owners() {
    let probe = Arc::new(Probe::default());
    let make = Arc::clone(&probe);
    let result = PipelineParallelExecutor::new_with_factory(topology(), boots(), move |boot| {
        let stage = stage(&boot, make)?;
        let mut mismatched = boot;
        if mismatched.rank.local.get() == 1 {
            mismatched.config.session_capacity += 1;
        }
        Ok(PipelineStageWorker::new(&mismatched, stage)?.boxed())
    });
    assert!(result.is_err());
    assert_eq!(probe.program_shutdown.load(Ordering::Relaxed), 2);
    assert_eq!(probe.backend_shutdown.load(Ordering::Relaxed), 2);
}

#[derive(Default)]
struct TransportProbe {
    calls: Mutex<Vec<&'static str>>,
    shutdown: AtomicUsize,
    quarantine: AtomicUsize,
    pending_install: AtomicBool,
    pending_retire: AtomicBool,
    pending_abort: AtomicBool,
    lose_install: AtomicBool,
    cancel_execute: AtomicBool,
    bad_description: AtomicBool,
    bad_projection: AtomicBool,
}
struct LocalTransport {
    workers: Vec<BoxedPipelineStageWorker>,
    probe: Arc<TransportProbe>,
}
impl LocalTransport {
    fn new(programs: Arc<Probe>, probe: Arc<TransportProbe>) -> Self {
        let workers = boots()
            .into_iter()
            .map(|boot| {
                PipelineStageWorker::new(&boot, stage(&boot, Arc::clone(&programs)).unwrap())
                    .unwrap()
                    .boxed()
            })
            .collect();
        Self { workers, probe }
    }
}
impl PipelineTransport for LocalTransport {
    fn call_observed(
        &mut self,
        rank: PipelineRank,
        command: PipelineCommand,
        poll: &mut dyn FnMut(),
    ) -> Result<PipelineReply> {
        let name = match &command {
            PipelineCommand::Describe => "describe",
            PipelineCommand::Stats => "stats",
            PipelineCommand::Create { .. } => "create",
            PipelineCommand::Fork { .. } => "fork",
            PipelineCommand::Release { .. } => "release",
            PipelineCommand::Prepare { .. } => "prepare",
            PipelineCommand::Execute { cancellation, .. } => {
                if self.probe.cancel_execute.load(Ordering::Relaxed) {
                    cancellation.store(true, Ordering::Release);
                }
                "execute"
            }
            PipelineCommand::Ready(_) => "ready",
            PipelineCommand::Install { .. } => "install",
            PipelineCommand::PollInstall { .. } => "poll",
            PipelineCommand::Abort { .. } => "abort",
            PipelineCommand::Rollback(_) => "rollback",
            PipelineCommand::Publish(_) => "publish",
            PipelineCommand::CheckRetirement { .. } => "check_retirement",
            PipelineCommand::Retire { .. } => "retire",
            PipelineCommand::Finish(_) => "finish",
        };
        self.probe.calls.lock().unwrap().push(name);
        poll();
        if name == "poll" && self.probe.lose_install.load(Ordering::Relaxed) {
            return Err(fault("bounded proxy deadline"));
        }
        let reply = self.workers[rank.local.get() as usize].dispatch(command)?;
        if name == "install" && self.probe.pending_install.load(Ordering::Relaxed) {
            return Ok(PipelineReply::Ack(KvEndProgress::Pending));
        }
        // Delay one completed ACK so the coordinator must retry the same command.
        if (name == "retire" && self.probe.pending_retire.swap(false, Ordering::Relaxed))
            || (name == "abort" && self.probe.pending_abort.swap(false, Ordering::Relaxed))
        {
            assert!(matches!(reply, PipelineReply::Ack(KvEndProgress::Complete)));
            return Ok(PipelineReply::Ack(KvEndProgress::Pending));
        }
        if let PipelineReply::Prepared { mut projection } = reply {
            if self.probe.bad_projection.load(Ordering::Relaxed) {
                projection.reservation.generation += 1;
            }
            return Ok(PipelineReply::Prepared { projection });
        }
        if let PipelineReply::Description(mut description) = reply {
            if rank.local.get() == 1 && self.probe.bad_description.load(Ordering::Relaxed) {
                description.hidden += 1;
            }
            return Ok(PipelineReply::Description(description));
        }
        Ok(reply)
    }
    fn outstanding(&self) -> usize {
        0
    }
    fn shutdown(&mut self) -> Result<()> {
        self.probe.shutdown.fetch_add(1, Ordering::Relaxed);
        let failures = self
            .workers
            .iter_mut()
            .filter_map(|worker| worker.shutdown().err())
            .collect();
        Error::failures("test transport shutdown", failures)
    }
    fn quarantine(&mut self) {
        self.probe.quarantine.fetch_add(1, Ordering::Relaxed);
        for worker in self.workers.drain(..) {
            std::mem::forget(worker);
        }
    }
}

#[test]
fn transport_injection_uses_same_prepare_install_poll_publish_retire_and_fork_journal() {
    let programs = Arc::new(Probe::default());
    let probe = Arc::new(TransportProbe::default());
    probe.pending_install.store(true, Ordering::Relaxed);
    probe.pending_retire.store(true, Ordering::Relaxed);
    let transport = LocalTransport::new(Arc::clone(&programs), Arc::clone(&probe));
    let mut pipeline =
        PipelineParallelExecutor::new_with_transport(topology(), plans(), config(), transport)
            .unwrap();
    pipeline.create_session(SessionId(1)).unwrap();
    pipeline
        .forward(SessionId(1), &[1], ForwardPhase::Prefill)
        .unwrap();
    pipeline.fork_session(SessionId(1), SessionId(2)).unwrap();
    pipeline
        .forward(SessionId(2), &[2], ForwardPhase::Decode)
        .unwrap();
    assert_eq!(pipeline.coordinator().publication_count(), 2);
    pipeline.shutdown().unwrap();
    let calls = probe.calls.lock().unwrap();
    for expected in [
        "create",
        "fork",
        "prepare",
        "execute",
        "ready",
        "install",
        "poll",
        "publish",
        "check_retirement",
        "retire",
        "finish",
        "release",
    ] {
        assert!(calls.contains(&expected), "missing {expected}");
    }
    assert_eq!(calls.iter().filter(|&&name| name == "install").count(), 4);
    assert_eq!(calls.iter().filter(|&&name| name == "retire").count(), 5);
    assert_eq!(probe.shutdown.load(Ordering::Relaxed), 1);
    assert_eq!(programs.backend_shutdown.load(Ordering::Relaxed), 2);
}

#[test]
fn transport_cancellation_and_typed_program_error_roll_back_without_publication() {
    let programs = Arc::new(Probe::default());
    let probe = Arc::new(TransportProbe::default());
    let mut pipeline = PipelineParallelExecutor::new_with_transport(
        topology(),
        plans(),
        config(),
        LocalTransport::new(Arc::clone(&programs), Arc::clone(&probe)),
    )
    .unwrap();
    programs.unsupported.store(true, Ordering::Release);
    let error = pipeline
        .forward(SessionId(1), &[1], ForwardPhase::Prefill)
        .unwrap_err();
    assert!(is_unsupported(&error));
    programs.unsupported.store(false, Ordering::Release);
    probe.cancel_execute.store(true, Ordering::Release);
    assert!(
        pipeline
            .forward(SessionId(1), &[1], ForwardPhase::Prefill)
            .is_err()
    );
    assert!(!pipeline.is_quarantined());
    assert_eq!(pipeline.page_manager().allocated_pages(), 0);
    assert_eq!(pipeline.coordinator().publication_count(), 0);
    assert_eq!(pipeline.coordinator().retained_transaction_count(), 0);
    probe.cancel_execute.store(false, Ordering::Release);
    pipeline
        .forward(SessionId(1), &[1], ForwardPhase::Prefill)
        .unwrap();
    pipeline.shutdown().unwrap();
}

#[test]
fn startup_description_failure_and_program_shutdown_error_still_shutdown_all() {
    let programs = Arc::new(Probe::default());
    programs
        .fail_program_shutdown
        .store(true, Ordering::Relaxed);
    let probe = Arc::new(TransportProbe::default());
    probe.bad_description.store(true, Ordering::Relaxed);
    let result = PipelineParallelExecutor::new_with_transport(
        topology(),
        plans(),
        config(),
        LocalTransport::new(Arc::clone(&programs), Arc::clone(&probe)),
    );
    assert!(result.is_err());
    assert_eq!(probe.shutdown.load(Ordering::Relaxed), 1);
    assert_eq!(programs.program_shutdown.load(Ordering::Relaxed), 2);
    assert_eq!(programs.backend_shutdown.load(Ordering::Relaxed), 2);
}

#[test]
fn unresolved_transport_ack_quarantines_without_shutdown_or_join_on_drop() {
    let programs = Arc::new(Probe::default());
    let probe = Arc::new(TransportProbe::default());
    probe.pending_install.store(true, Ordering::Relaxed);
    probe.lose_install.store(true, Ordering::Relaxed);
    let mut pipeline = PipelineParallelExecutor::new_with_transport(
        topology(),
        plans(),
        config(),
        LocalTransport::new(Arc::clone(&programs), Arc::clone(&probe)),
    )
    .unwrap();
    assert!(
        pipeline
            .forward(SessionId(1), &[1], ForwardPhase::Prefill)
            .is_err()
    );
    assert!(pipeline.is_quarantined());
    assert_eq!(pipeline.coordinator().publication_count(), 0);
    assert_eq!(pipeline.page_manager().allocated_pages(), 1);
    assert!(pipeline.shutdown().is_err());
    drop(pipeline);
    assert_eq!(probe.quarantine.load(Ordering::Relaxed), 1);
    assert_eq!(probe.shutdown.load(Ordering::Relaxed), 0);
    assert_eq!(programs.backend_shutdown.load(Ordering::Relaxed), 0);
}

fn key() -> PipelineCommandKey {
    let topology = topology();
    PipelineCommandKey::new(
        KvCommitBinding::new(
            ExecutionTransactionId::new(1).unwrap(),
            topology.topology_id(),
            topology.participants(),
            1,
        )
        .unwrap(),
        rank(0),
        SessionId(1),
    )
    .unwrap()
}
fn prepare(key: PipelineCommandKey) -> PipelineCommand {
    PipelineCommand::Prepare {
        key,
        batch: Box::new(ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1],
            vec![0],
            vec![Some(KvWriteSlot::new(0))],
            vec![LogitsRequest::Full],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Prefill,
                0..1,
                0,
                1,
                0..1,
            )],
            vec![KvBlockId::new(0)],
        )),
        reservation: KvReservationView {
            state_slot: StateSlot::new(0),
            execution_state_slot: StateSlot::new(0),
            positions: 0..1,
            newly_allocated: vec![KvPageId(0)],
            generation: 0,
            execution_generation: 0,
            cow_replacement: None,
        },
    }
}
#[test]
fn public_worker_rejects_stale_keys_duplicate_execute_and_install_without_ready() {
    let probe = Arc::new(Probe::default());
    let boot = boots().remove(0);
    let mut worker =
        PipelineStageWorker::new(&boot, stage(&boot, Arc::clone(&probe)).unwrap()).unwrap();
    worker
        .dispatch(PipelineCommand::Create {
            session: SessionId(1),
        })
        .unwrap();
    let key = key();
    let (binding, rank, session) = key.clone().into_parts();
    assert_eq!(
        PipelineCommandKey::new(binding.clone(), rank, session).unwrap(),
        key
    );
    assert!(PipelineCommandKey::new(binding, self::rank(9), session).is_err());
    worker.dispatch(prepare(key.clone())).unwrap();
    assert!(worker.has_active_custody());
    assert!(
        worker
            .dispatch(PipelineCommand::Ready(key.clone()))
            .is_err()
    );
    assert!(
        worker
            .dispatch(PipelineCommand::Install {
                key: key.clone(),
                generation: 1
            })
            .is_err()
    );
    let execute = || PipelineCommand::Execute {
        key: key.clone(),
        input: SegmentInput::Tokens,
        cancellation: Arc::new(AtomicBool::new(false)),
    };
    worker.dispatch(execute()).unwrap();
    assert!(worker.dispatch(execute()).is_err());
    assert_eq!(probe.executed.load(Ordering::Relaxed), 1);
    worker.dispatch(PipelineCommand::Rollback(key)).unwrap();
    assert!(!worker.has_active_custody());
    worker.shutdown().unwrap();
}

#[test]
fn public_worker_completed_abort_ack_requires_exact_generation() {
    let boot = boots().remove(0);
    let mut worker =
        PipelineStageWorker::new(&boot, stage(&boot, Arc::new(Probe::default())).unwrap()).unwrap();
    let key = key();
    worker
        .dispatch(PipelineCommand::Create {
            session: key.session(),
        })
        .unwrap();
    worker.dispatch(prepare(key.clone())).unwrap();
    let abort = |generation| PipelineCommand::Abort {
        key: key.clone(),
        generation,
    };
    for _ in 0..2 {
        assert!(matches!(
            worker.dispatch(abort(1)).unwrap(),
            PipelineReply::Ack(KvEndProgress::Complete)
        ));
    }
    for generation in [0, 2, u64::MAX] {
        assert!(worker.dispatch(abort(generation)).is_err());
        assert!(worker.has_active_custody());
        assert!(matches!(
            worker.dispatch(abort(1)).unwrap(),
            PipelineReply::Ack(KvEndProgress::Complete)
        ));
    }
    worker.dispatch(PipelineCommand::Finish(key)).unwrap();
    assert!(!worker.has_active_custody());
    worker.shutdown().unwrap();
}

#[test]
fn public_worker_completed_retire_ack_requires_exact_generation_and_pages() {
    let boot = boots().remove(0);
    let mut worker =
        PipelineStageWorker::new(&boot, stage(&boot, Arc::new(Probe::default())).unwrap()).unwrap();
    let key = key();
    worker
        .dispatch(PipelineCommand::Create {
            session: key.session(),
        })
        .unwrap();
    worker.dispatch(prepare(key.clone())).unwrap();
    worker
        .dispatch(PipelineCommand::Execute {
            key: key.clone(),
            input: SegmentInput::Tokens,
            cancellation: Arc::new(AtomicBool::new(false)),
        })
        .unwrap();
    for command in [
        PipelineCommand::Ready(key.clone()),
        PipelineCommand::Install {
            key: key.clone(),
            generation: 1,
        },
        PipelineCommand::Publish(key.clone()),
    ] {
        assert!(matches!(
            worker.dispatch(command).unwrap(),
            PipelineReply::Ack(KvEndProgress::Complete)
        ));
    }
    let retire = |generation, pages| PipelineCommand::Retire {
        key: key.clone(),
        generation,
        pages,
    };
    for _ in 0..2 {
        assert!(matches!(
            worker.dispatch(retire(1, vec![])).unwrap(),
            PipelineReply::Ack(KvEndProgress::Complete)
        ));
    }
    for (generation, pages) in [
        (0, vec![]),
        (2, vec![]),
        (1, vec![KvPageId(0)]),
        (1, vec![KvPageId(0), KvPageId(0)]),
        (1, vec![KvPageId(u32::MAX)]),
    ] {
        assert!(worker.dispatch(retire(generation, pages)).is_err());
        assert!(worker.has_active_custody());
        assert!(matches!(
            worker.dispatch(retire(1, vec![])).unwrap(),
            PipelineReply::Ack(KvEndProgress::Complete)
        ));
    }
    worker
        .dispatch(PipelineCommand::Finish(key.clone()))
        .unwrap();
    assert!(!worker.has_active_custody());
    worker
        .dispatch(PipelineCommand::Release {
            session: key.session(),
            pages: vec![KvPageId(0)],
        })
        .unwrap();
    worker.shutdown().unwrap();
}

#[test]
fn public_worker_unknown_prepare_retains_journal_and_rejects_cleanup_acks() {
    for (failure, hide_count) in [
        (PrepareFailure::Unknown, false),
        (PrepareFailure::UntypedUnknown, false),
        (PrepareFailure::Unknown, true),
    ] {
        let probe = Arc::new(Probe::default());
        *probe.prepare_failure.lock().unwrap() = failure;
        probe
            .hide_retained_count
            .store(hide_count, Ordering::Release);
        let boot = boots().remove(0);
        let mut worker =
            PipelineStageWorker::new(&boot, stage(&boot, Arc::clone(&probe)).unwrap()).unwrap();
        let key = key();
        worker
            .dispatch(PipelineCommand::Create {
                session: key.session(),
            })
            .unwrap();
        let source = worker.dispatch(prepare(key.clone())).unwrap_err();
        assert_eq!(
            KvPrepareQuiescenceUnknown::from_error(&source)
                .unwrap()
                .transaction(),
            key.binding().transaction()
        );
        drop(source);
        assert!(worker.has_active_custody());
        for command in [
            PipelineCommand::Execute {
                key: key.clone(),
                input: SegmentInput::Tokens,
                cancellation: Arc::new(AtomicBool::new(false)),
            },
            PipelineCommand::Rollback(key.clone()),
            PipelineCommand::Ready(key.clone()),
            PipelineCommand::Install {
                key: key.clone(),
                generation: 1,
            },
            PipelineCommand::PollInstall {
                key: key.clone(),
                generation: 1,
            },
            PipelineCommand::Abort {
                key: key.clone(),
                generation: 1,
            },
            PipelineCommand::Publish(key.clone()),
            PipelineCommand::CheckRetirement {
                key: key.clone(),
                pages: vec![],
            },
            PipelineCommand::Retire {
                key: key.clone(),
                generation: 1,
                pages: vec![],
            },
            PipelineCommand::Finish(key.clone()),
        ] {
            let source = worker.dispatch(command).unwrap_err();
            assert!(KvPrepareQuiescenceUnknown::from_error(&source).is_some());
            assert!(worker.has_active_custody());
        }
        assert!(worker.dispatch(prepare(key.clone())).is_err());
        assert!(
            worker
                .dispatch(PipelineCommand::Release {
                    session: key.session(),
                    pages: vec![KvPageId(0)]
                })
                .is_err()
        );
        let PipelineReply::Stats(stats) = worker.dispatch(PipelineCommand::Stats).unwrap() else {
            unreachable!()
        };
        assert_eq!(stats.kv.active_transactions, usize::from(!hide_count));
        assert_eq!(stats.kv.free_pages, 7);
        assert_eq!(stats.executions, 0);
        assert_eq!(stats.sessions, 1);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker.shutdown()))
            .unwrap_err();
        assert_eq!(
            panic.downcast_ref::<ferrule_runtime::parallel::data::PanicQuiescence>(),
            Some(&ferrule_runtime::parallel::data::PanicQuiescence::Unknown)
        );
        assert_eq!(probe.backend_shutdown.load(Ordering::Acquire), 0);
        assert_eq!(probe.program_shutdown.load(Ordering::Acquire), 0);
        assert_eq!(probe.prepare_calls.load(Ordering::Acquire), 1);
        // Unknown custody intentionally has no production cleanup escape hatch.
        std::mem::forget(worker);
    }
}

#[test]
fn public_worker_validation_and_cleaned_prepare_allow_rollback_ack_and_retry() {
    for failure in [PrepareFailure::Validation, PrepareFailure::Cleaned] {
        let probe = Arc::new(Probe::default());
        *probe.prepare_failure.lock().unwrap() = failure;
        let boot = boots().remove(0);
        let mut worker =
            PipelineStageWorker::new(&boot, stage(&boot, Arc::clone(&probe)).unwrap()).unwrap();
        let key = key();
        worker
            .dispatch(PipelineCommand::Create {
                session: key.session(),
            })
            .unwrap();
        let source = worker.dispatch(prepare(key.clone())).unwrap_err();
        assert!(KvPrepareQuiescenceUnknown::from_error(&source).is_none());
        assert!(!worker.has_active_custody());
        for _ in 0..2 {
            assert!(matches!(
                worker
                    .dispatch(PipelineCommand::Rollback(key.clone()))
                    .unwrap(),
                PipelineReply::Ack(KvEndProgress::Complete)
            ));
        }
        let PipelineReply::Stats(stats) = worker.dispatch(PipelineCommand::Stats).unwrap() else {
            unreachable!()
        };
        assert_eq!(stats.kv.active_transactions, 0);
        assert_eq!(stats.kv.free_pages, 8);
        worker.dispatch(prepare(key.clone())).unwrap();
        assert!(worker.has_active_custody());
        worker
            .dispatch(PipelineCommand::Rollback(key.clone()))
            .unwrap();
        assert!(!worker.has_active_custody());
        assert_eq!(probe.prepare_calls.load(Ordering::Acquire), 2);
        worker
            .dispatch(PipelineCommand::Release {
                session: key.session(),
                pages: vec![],
            })
            .unwrap();
        worker.shutdown().unwrap();
    }
}

#[test]
fn unknown_prepare_keeps_coordinator_and_logical_pages_quarantined() {
    let programs = Arc::new(Probe::default());
    let transport = Arc::new(TransportProbe::default());
    let mut pipeline = PipelineParallelExecutor::new_with_transport(
        topology(),
        plans(),
        config(),
        LocalTransport::new(Arc::clone(&programs), Arc::clone(&transport)),
    )
    .unwrap();
    pipeline
        .forward(SessionId(1), &[1], ForwardPhase::Prefill)
        .unwrap();
    pipeline.fork_session(SessionId(1), SessionId(2)).unwrap();
    transport.calls.lock().unwrap().clear();
    *programs.prepare_failure.lock().unwrap() = PrepareFailure::Unknown;
    let source = pipeline
        .forward(SessionId(2), &[2], ForwardPhase::Decode)
        .unwrap_err();
    let id = KvPrepareQuiescenceUnknown::from_error(&source)
        .unwrap()
        .transaction();
    drop(source);
    assert!(pipeline.is_quarantined());
    assert_eq!(
        pipeline.coordinator().state(id).unwrap(),
        TransactionState::Decided(ferrule_runtime::Decision::Abort)
    );
    assert_eq!(pipeline.coordinator().retained_transaction_count(), 1);
    assert_eq!(pipeline.coordinator().in_use_credits(), 2);
    assert_eq!(pipeline.coordinator().publication_count(), 1);
    assert_eq!(pipeline.page_manager().allocated_pages(), 2);
    assert_eq!(pipeline.page_manager().stats().committed_tokens, 2);
    assert!(
        pipeline
            .coordinator()
            .pending_ranks(id)
            .unwrap()
            .contains(&rank(0).global)
    );
    let stats = pipeline.owner_stats().unwrap();
    assert_eq!(stats[0].kv.active_transactions, 1);
    assert_eq!(stats[0].kv.free_pages, 6);
    assert!(pipeline.release_session(SessionId(1)).is_err());
    assert!(pipeline.release_session(SessionId(2)).is_err());
    assert!(
        pipeline
            .forward(SessionId(2), &[2], ForwardPhase::Decode)
            .is_err()
    );
    assert!(pipeline.shutdown().is_err());
    let calls = transport.calls.lock().unwrap().clone();
    assert_eq!(calls.iter().filter(|&&name| name == "prepare").count(), 1);
    for forbidden in [
        "execute", "install", "publish", "retire", "finish", "release",
    ] {
        assert!(!calls.contains(&forbidden), "unexpected {forbidden}");
    }
    drop(pipeline);
    assert_eq!(transport.quarantine.load(Ordering::Acquire), 1);
    assert_eq!(transport.shutdown.load(Ordering::Acquire), 0);
    assert_eq!(programs.backend_shutdown.load(Ordering::Acquire), 0);
}

#[test]
fn portable_projection_revalidates_request_snapshots_and_arbitrary_fork_generation() {
    let boot = boots().remove(0);
    let mut worker =
        PipelineStageWorker::new(&boot, stage(&boot, Arc::new(Probe::default())).unwrap()).unwrap();
    worker
        .dispatch(PipelineCommand::Create {
            session: SessionId(1),
        })
        .unwrap();
    let PipelineCommand::Prepare {
        batch, reservation, ..
    } = prepare(key())
    else {
        unreachable!()
    };
    let PipelineReply::Prepared { projection } = worker.dispatch(prepare(key())).unwrap() else {
        unreachable!()
    };
    let description = worker.boot_description();
    let decoded = projection
        .clone()
        .into_commit_batch(&batch, &reservation, description)
        .unwrap();
    assert_eq!(decoded.new_pages(), &[KvPageId(0)]);
    assert_eq!(
        decoded.sequences()[0].page_state_slot(),
        reservation.state_slot
    );
    assert_eq!(decoded.sequences()[0].page_generation(), 0);

    let mut stale = projection.clone();
    stale.reservation.generation += 1;
    assert!(
        stale
            .into_commit_batch(&batch, &reservation, description)
            .is_err()
    );
    let mut duplicate = projection.clone();
    duplicate.page_statuses.push(duplicate.page_statuses[0]);
    assert!(
        duplicate
            .into_commit_batch(&batch, &reservation, description)
            .is_err()
    );
    let mut missing = projection.clone();
    missing.page_statuses.clear();
    assert!(
        missing
            .into_commit_batch(&batch, &reservation, description)
            .is_err()
    );
    let mut wrong_status = projection.clone();
    wrong_status.page_statuses[0].status = DecoderKvPageStatus::Resident;
    assert!(
        wrong_status
            .into_commit_batch(&batch, &reservation, description)
            .is_err()
    );

    // Conversion is constant work in the generation, not a replay of forks to
    // manufacture an owner SequenceStateCore. This is only a logical projection.
    let mut deep = projection;
    deep.reservation.generation = u64::MAX - 1;
    deep.reservation.execution_generation = u64::MAX - 1;
    let expected = deep.reservation.clone();
    let decoded = deep
        .into_commit_batch(&batch, &expected, description)
        .unwrap();
    assert_eq!(decoded.sequences()[0].page_generation(), u64::MAX - 1);
    assert_eq!(decoded.sequences()[0].execution_generation(), 0);
    worker.dispatch(PipelineCommand::Rollback(key())).unwrap();
    worker.shutdown().unwrap();
}

#[test]
fn tampered_transport_projection_rolls_back_before_program_execution() {
    let programs = Arc::new(Probe::default());
    let probe = Arc::new(TransportProbe::default());
    probe.bad_projection.store(true, Ordering::Relaxed);
    let transport: Box<dyn PipelineTransport> = Box::new(LocalTransport::new(
        Arc::clone(&programs),
        Arc::clone(&probe),
    ));
    let mut pipeline =
        PipelineParallelExecutor::new_with_transport(topology(), plans(), config(), transport)
            .unwrap();
    assert!(
        pipeline
            .forward(SessionId(1), &[1], ForwardPhase::Prefill)
            .is_err()
    );
    assert!(!pipeline.is_quarantined());
    assert_eq!(programs.executed.load(Ordering::Relaxed), 0);
    assert_eq!(pipeline.page_manager().allocated_pages(), 0);
    assert_eq!(pipeline.coordinator().retained_transaction_count(), 0);
    pipeline.shutdown().unwrap();
}

#[test]
fn host_boundary_rejects_wrong_shape_precision_and_endpoint() {
    let boot = boots().remove(1);
    let worker =
        PipelineStageWorker::new(&boot, stage(&boot, Arc::new(Probe::default())).unwrap()).unwrap();
    let description = worker.boot_description();
    assert!(
        description
            .validate_input(&SegmentInput::Tokens, 1)
            .is_err()
    );
    let hidden = |width, dtype| SegmentInput::Hidden {
        next_layer: 1,
        rows: HostRows::new(
            RowsShape::new(1, width).unwrap(),
            dtype,
            None,
            vec![0.0; width],
        )
        .unwrap(),
    };
    assert!(
        description
            .validate_input(&hidden(3, RowsDType::F32), 1)
            .is_err()
    );
    assert!(
        description
            .validate_input(&hidden(2, RowsDType::Bf16), 1)
            .is_err()
    );
    description
        .validate_input(&hidden(2, RowsDType::F32), 1)
        .unwrap();
    assert!(
        description
            .validate_output(
                &SegmentOutput::Logits(DenseLogits::new(1, 3, vec![0.0; 3]).unwrap()),
                1
            )
            .is_err()
    );
}

#[test]
fn repeated_fork_into_reused_logical_slots_keeps_generation_and_cow_projection() {
    let mut pipeline = PipelineParallelExecutor::new_with_transport(
        topology(),
        plans(),
        config(),
        LocalTransport::new(
            Arc::new(Probe::default()),
            Arc::new(TransportProbe::default()),
        ),
    )
    .unwrap();
    let mut source = SessionId(1);
    let mut target = SessionId(2);
    pipeline
        .forward(source, &[1], ForwardPhase::Prefill)
        .unwrap();
    for generation in 1..=6 {
        pipeline.fork_session(source, target).unwrap();
        pipeline.release_session(source).unwrap();
        pipeline
            .forward(target, &[2], ForwardPhase::Decode)
            .unwrap();
        let slot = pipeline.session_slot(target).unwrap();
        assert_eq!(
            pipeline.page_manager().sequence_generation(slot).unwrap(),
            generation
        );
        std::mem::swap(&mut source, &mut target);
    }
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.page_manager().allocated_pages(), 0);
}

#[test]
fn injected_transport_observes_all_ready_abort_and_finish() {
    let probe = Arc::new(TransportProbe::default());
    probe.pending_abort.store(true, Ordering::Relaxed);
    let mut pipeline = PipelineParallelExecutor::new_with_transport(
        topology(),
        plans(),
        config(),
        LocalTransport::new(Arc::new(Probe::default()), Arc::clone(&probe)),
    )
    .unwrap();
    let cancel = AtomicBool::new(false);
    assert!(
        pipeline
            .forward_observed(
                SessionId(1),
                &[1],
                ForwardPhase::Prefill,
                &cancel,
                |progress| {
                    if progress.state == TransactionState::Preparing {
                        cancel.store(true, Ordering::Release);
                    }
                }
            )
            .is_err()
    );
    assert!(!pipeline.is_quarantined());
    assert_eq!(pipeline.page_manager().allocated_pages(), 0);
    let calls = probe.calls.lock().unwrap();
    assert_eq!(calls.iter().filter(|&&name| name == "abort").count(), 3);
    assert!(calls.contains(&"finish"));
    drop(calls);
    pipeline.shutdown().unwrap();
}
