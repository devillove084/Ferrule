//! Controlled non-Clone/non-Send state exercises the existing pipeline journal.
//! This fixture is not model math; real hybrid logits are tested separately.

use ferrule_model::ModelExecutionBackend;
use ferrule_runtime::parallel::pipeline::PhysicalKvLayout;
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
    CpuKvView, DecoderKvBackend, DecoderKvCapacity, DecoderKvCommitBackend, DecoderKvPageStatus,
    DecoderKvPrepare, DecoderSequence, DecoderSequenceCheckout, DecoderSequenceLifecycle,
    DenseLogits, GenericDecoderSequenceState, KvCommitBinding, KvEndProgress,
    KvPrepareQuiescenceUnknown, PackedDecoderBatch, PagedKvBackend, PagedKvTransactionHandle,
    StandardGqaPlanes, TypedCpuPagedKvPool,
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
    PipelineStageBoot, PipelineStageDescription, PipelineStageWorker, PipelineStateProgram,
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
            physical_pages: config().max_pages,
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
    #[cfg(feature = "cuda")]
    committed_traces: AtomicUsize,
    #[cfg(feature = "cuda")]
    trace_fault: AtomicUsize,
    expert_quiescence_pending: AtomicBool,
    expert_quiescence_unknown: AtomicBool,
    initialized: Mutex<Vec<ThreadId>>,
    program_shutdown: AtomicUsize,
    backend_shutdown: AtomicUsize,
    executed: AtomicUsize,
    unsupported: AtomicBool,
    prepare_failure: Mutex<PrepareFailure>,
    prepare_calls: AtomicUsize,
    hide_retained_count: AtomicBool,
    fail_program_shutdown: AtomicBool,
    unknown_capacity: Mutex<Option<DecoderKvCapacity>>,
    cleanup_calls: Mutex<Vec<&'static str>>,
    next_state: AtomicUsize,
    releases: Mutex<Vec<(usize, usize, usize)>>,
    drops: Mutex<Vec<usize>>,
    release_failures: AtomicUsize,
    rollbacks: AtomicUsize,
    physical_releases: AtomicUsize,
    aborts: AtomicUsize,
    retires: AtomicUsize,
    bad_frontier: AtomicBool,
    mutate_core: AtomicBool,
    fail_after_execute: AtomicBool,
    cancel_after_execute: Mutex<Option<Arc<AtomicBool>>>,
}

struct State {
    inner: GenericDecoderSequenceState,
    frontier: usize,
    id: usize,
    owner: Rc<ThreadId>,
    probe: Arc<Probe>,
}
impl DecoderSequence for State {
    fn topology_id(&self) -> ferrule_model::execution::SequenceTopologyId {
        self.inner.topology_id()
    }
    fn core(&self) -> &ferrule_model::execution::SequenceStateCore {
        self.inner.core()
    }
    fn core_mut(&mut self) -> &mut ferrule_model::execution::SequenceStateCore {
        self.inner.core_mut()
    }
}
impl Drop for State {
    fn drop(&mut self) {
        assert_eq!(*self.owner, thread::current().id());
        self.probe.drops.lock().unwrap().push(self.id);
    }
}
struct Lifecycle {
    owner: Rc<ThreadId>,
    probe: Arc<Probe>,
}
impl Lifecycle {
    fn wrap(&self, inner: GenericDecoderSequenceState, frontier: usize) -> State {
        State {
            inner,
            frontier,
            id: self.probe.next_state.fetch_add(1, Ordering::Relaxed),
            owner: self.owner.clone(),
            probe: self.probe.clone(),
        }
    }
}
impl DecoderSequenceLifecycle<State> for Lifecycle {
    fn create(&mut self) -> Result<State> {
        Ok(self.wrap(GenericDecoderSequenceState::new((), ()), 0))
    }
    fn checkout(&mut self, _: DecoderSequenceCheckout, source: &State) -> Result<State> {
        assert_eq!(source.frontier, source.core().position());
        Ok(self.wrap(source.inner.transaction_working_copy()?, source.frontier))
    }
    fn logical_fork(&mut self, source: &State, position: usize) -> Result<State> {
        assert_eq!(source.core().position(), position);
        assert_eq!(source.frontier, position);
        Ok(self.wrap(source.inner.logical_fork()?, position))
    }
    fn reset(&mut self, _: &mut State) -> Result<()> {
        panic!("pipeline must not reset state")
    }
    fn try_release(
        &mut self,
        state: State,
    ) -> std::result::Result<(), ferrule_model::runner::SequenceStateReleaseError<State>> {
        if self
            .probe
            .release_failures
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_ok()
        {
            return Err(ferrule_model::runner::SequenceStateReleaseError::new(
                fault("release pending"),
                state,
            ));
        }
        self.probe.releases.lock().unwrap().push((
            state.id,
            state.core().position(),
            state.frontier,
        ));
        drop(state);
        Ok(())
    }
}

// A deliberately distinct, non-Send view. The program cannot accidentally
// satisfy a hardcoded CpuKvView bound; only this adapter unwraps the CPU oracle.
struct DeviceView {
    cpu: CpuKvView,
    owner: Rc<ThreadId>,
}
struct DeviceBackend {
    cpu: PagedKvBackend<TypedCpuPagedKvPool<State>>,
    owner: Rc<ThreadId>,
    probe: Arc<Probe>,
    retained: Option<PagedKvTransactionHandle>,
}
impl DeviceBackend {
    fn here(&self) {
        assert_eq!(*self.owner, thread::current().id());
    }

    fn record_cleanup(&self, operation: &'static str) {
        self.here();
        self.probe.cleanup_calls.lock().unwrap().push(operation);
    }
}
impl Drop for DeviceBackend {
    fn drop(&mut self) {
        self.record_cleanup("drop");
    }
}
impl DecoderKvBackend for DeviceBackend {
    type SequenceState = State;
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
                // Observe on the owner thread before returning Unknown, not via
                // a transport Stats ACK after quiescence has become unknown.
                *self.probe.unknown_capacity.lock().unwrap() = Some(self.cpu.capacity());
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
        self.record_cleanup("rollback");
        self.here();
        self.probe.rollbacks.fetch_add(1, Ordering::Relaxed);
        self.cpu.rollback(tx)
    }
    fn release(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.record_cleanup("release");
        self.here();
        self.probe.physical_releases.fetch_add(1, Ordering::Relaxed);
        self.cpu.release(pages)
    }
    fn preempt(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.record_cleanup("preempt");
        self.cpu.preempt(pages)
    }
    fn restore(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.record_cleanup("restore");
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
        self.record_cleanup("shutdown");
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
        self.probe.aborts.fetch_add(1, Ordering::Relaxed);
        self.record_cleanup("abort");
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
        self.probe.retires.fetch_add(1, Ordering::Relaxed);
        self.record_cleanup("retire");
        self.cpu.retire_prepared(tx, generation, pages)
    }
    fn finish_prepared(&mut self, tx: Self::Transaction) {
        self.record_cleanup("finish");
        self.cpu.finish_prepared(tx);
    }
}

struct DeviceProgram {
    plan: LayerSegmentPlan,
    owner: Rc<ThreadId>,
    probe: Arc<Probe>,
}
impl PipelineStateProgram for DeviceProgram {
    #[cfg(feature = "cuda")]
    fn trace_committed(
        &mut self,
        backend: &mut dyn std::any::Any,
        transaction: ExecutionTransactionId,
        source: &ferrule_common::host_kv::CommittedKvSource,
    ) -> Result<()> {
        assert_eq!(*self.owner, thread::current().id());
        let backend = backend.downcast_mut::<DeviceBackend>().unwrap();
        assert_eq!(backend.capacity().active_transactions, 0);
        assert!(source.committed_tokens > 0);
        self.probe.committed_traces.fetch_add(1, Ordering::Relaxed);
        match self.probe.trace_fault.load(Ordering::Acquire) {
            1 => Err(fault("ordinary trace failure after proven completion")),
            2 => Err(KvPrepareQuiescenceUnknown::wrap(
                transaction,
                fault("trace readback unknown"),
            )),
            _ => Ok(()),
        }
    }
    fn poll_quiescence(&mut self) -> Result<bool> {
        if self.probe.expert_quiescence_unknown.load(Ordering::Acquire) {
            return Err(KvPrepareQuiescenceUnknown::wrap(
                ExecutionTransactionId::new(1).unwrap(),
                fault("expert completion unknown"),
            ));
        }
        Ok(!self.probe.expert_quiescence_pending.load(Ordering::Acquire))
    }
    type SequenceState = State;
    type Lifecycle = Lifecycle;
    fn new_lifecycle(&mut self) -> Result<Lifecycle> {
        Ok(Lifecycle {
            owner: self.owner.clone(),
            probe: self.probe.clone(),
        })
    }
    fn validate_frontier(&self, batch: &PackedDecoderBatch, states: &[State]) -> Result<()> {
        if states[0].frontier != batch.sequences()[0].sequence_len() {
            return Err(fault("frontier mismatch"));
        }
        Ok(())
    }
    type KvView = DeviceView;
    fn plan(&self) -> &LayerSegmentPlan {
        &self.plan
    }
    fn execute_state(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [State],
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
        assert_eq!(
            states[0].core().position(),
            batch.sequences()[0].context_len()
        );
        assert_eq!(states[0].frontier, states[0].core().position());
        states[0].frontier = batch.sequences()[0].sequence_len();
        if self.probe.bad_frontier.load(Ordering::Relaxed) {
            states[0].frontier += 1;
        }
        if self.probe.mutate_core.load(Ordering::Relaxed) {
            states[0].core_mut().reset();
        }
        if self.probe.fail_after_execute.load(Ordering::Relaxed) {
            return Err(fault("failure after state mutation"));
        }
        if let Some(cancel) = self.probe.cancel_after_execute.lock().unwrap().as_ref() {
            cancel.store(true, Ordering::Release);
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
                .as_chunks::<2>()
                .0
                .iter()
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
        cpu: PagedKvBackend::new(TypedCpuPagedKvPool::from_strategy(
            &planes,
            boot.physical_pages,
        )?),
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
        physical_pages: boot.physical_pages,
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
                    physical_pages: config().max_pages,
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
            let stage = stage(&boot, make)?;
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
            #[cfg(feature = "cuda")]
            PipelineCommand::TraceCommitted(_) => "trace_committed",
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
        .into_commit_projection(&batch, &reservation, description)
        .unwrap();
    let original = decoded;
    assert_eq!(original.protected_pages(), &[KvPageId(0)]);
    assert_eq!(
        original,
        projection
            .clone()
            .into_execution_identity(&batch, &reservation, description, &[1],)
            .unwrap()
            .commit_projection()
    );

    let mut stale = projection.clone();
    stale.reservation.generation += 1;
    assert!(
        stale
            .into_commit_projection(&batch, &reservation, description)
            .is_err()
    );
    let mut duplicate = projection.clone();
    duplicate.page_statuses.push(duplicate.page_statuses[0]);
    assert!(
        duplicate
            .into_commit_projection(&batch, &reservation, description)
            .is_err()
    );
    let mut missing = projection.clone();
    missing.page_statuses.clear();
    assert!(
        missing
            .into_commit_projection(&batch, &reservation, description)
            .is_err()
    );
    let mut wrong_status = projection.clone();
    wrong_status.page_statuses[0].status = DecoderKvPageStatus::Resident;
    assert!(
        wrong_status
            .into_commit_projection(&batch, &reservation, description)
            .is_err()
    );

    // Conversion is constant work in the generation, not a replay of forks to
    // manufacture an owner SequenceStateCore. This is only a logical projection.
    let mut deep = projection;
    deep.reservation.generation = u64::MAX - 1;
    deep.reservation.execution_generation = u64::MAX - 1;
    let expected = deep.reservation.clone();
    let decoded = deep
        .into_commit_projection(&batch, &expected, description)
        .unwrap();
    assert_ne!(decoded, original);
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

// CompletionFailure is crate-private: inspect its source diagnostic rather
// than exporting an unsafe post-quarantine statistics/classification bypass.
fn assert_unknown_stats_rejected(pipeline: &PipelineParallelExecutor) {
    let failure = pipeline.owner_stats().unwrap_err();
    let Error::ModelSource { source } = failure else {
        panic!("expected typed completion failure, got {failure:?}");
    };
    assert_eq!(
        source.to_string(),
        "pipeline stats rank 0 transaction None: unknown (quarantine=true)"
    );
    let evidence = format!("{source:?}");
    assert!(evidence.starts_with("CompletionFailure {"), "{evidence}");
    assert!(
        evidence.contains("class: Unknown, quarantine: true"),
        "{evidence}"
    );
}

#[test]
fn physical_shadow_capacity_never_widens_parent_or_releases_unknown_custody() {
    let probe = Arc::new(Probe::default());
    let mut specs = boots();
    for boot in &mut specs {
        boot.physical_pages = boot
            .config
            .physical_pages(PhysicalKvLayout::standard(ModelExecutionBackend::Cuda))
            .unwrap();
    }
    let make = Arc::clone(&probe);
    let mut pipeline = PipelineParallelExecutor::new_with_factory(topology(), specs, move |boot| {
        Ok(PipelineStageWorker::new(&boot, stage(&boot, make)?)?.boxed())
    })
    .unwrap();
    assert_eq!(pipeline.page_manager().max_pages(), 8);
    for owner in pipeline.owner_stats().unwrap() {
        assert_eq!((owner.kv.physical_pages, owner.kv.free_pages), (16, 16));
    }
    pipeline
        .forward(SessionId(1), &[1], ForwardPhase::Prefill)
        .unwrap();
    *probe.prepare_failure.lock().unwrap() = PrepareFailure::Unknown;
    assert!(
        pipeline
            .forward(SessionId(1), &[2], ForwardPhase::Decode)
            .is_err()
    );
    assert!(pipeline.is_quarantined());
    let before = probe.unknown_capacity.lock().unwrap().unwrap();
    assert_eq!(before.active_transactions, 1);
    assert_eq!(before.physical_pages, 16);
    assert!(before.free_pages < before.physical_pages);
    let cleanup = probe.cleanup_calls.lock().unwrap().clone();
    // Both owners completed the prefill cleanup; Unknown must not replay it.
    assert_eq!(cleanup.len(), 4);
    assert_eq!(cleanup.iter().filter(|&&call| call == "retire").count(), 2);
    assert_eq!(cleanup.iter().filter(|&&call| call == "finish").count(), 2);
    let prepares = probe.prepare_calls.load(Ordering::Acquire);
    let executions = probe.executed.load(Ordering::Acquire);
    let releases = probe.releases.lock().unwrap().clone();
    let drops = probe.drops.lock().unwrap().clone();
    let allocated = pipeline.page_manager().allocated_pages();
    let committed = pipeline.page_manager().stats().committed_tokens;
    let retained = pipeline.coordinator().retained_transaction_count();
    let credits = pipeline.coordinator().in_use_credits();
    assert_eq!(allocated, 1);
    assert_eq!(committed, 1);
    assert_eq!(retained, 1);
    assert_eq!(credits, 2);

    for _ in 0..2 {
        assert_unknown_stats_rejected(&pipeline);
        assert!(pipeline.release_session(SessionId(1)).is_err());
        assert!(pipeline.create_session(SessionId(2)).is_err());
        assert!(
            pipeline
                .forward(SessionId(1), &[2], ForwardPhase::Decode)
                .is_err()
        );
        assert!(pipeline.shutdown().is_err());
        assert_unknown_stats_rejected(&pipeline);
        assert!(pipeline.is_quarantined());
        assert_eq!(pipeline.page_manager().max_pages(), 8);
        assert_eq!(pipeline.page_manager().allocated_pages(), allocated);
        assert_eq!(pipeline.page_manager().stats().committed_tokens, committed);
        assert_eq!(
            pipeline.coordinator().retained_transaction_count(),
            retained
        );
        assert_eq!(pipeline.coordinator().in_use_credits(), credits);
        assert_eq!(pipeline.coordinator().publication_count(), 1);
    }
    drop(pipeline);
    // This is a saved owner-local observation, not a fresh quiescence proof.
    // No cleanup entry (including backend Drop) may free the observed slots or
    // consume the retained handle, even on repeated shutdown or executor Drop.
    assert_eq!(probe.unknown_capacity.lock().unwrap().unwrap(), before);
    assert_eq!(*probe.cleanup_calls.lock().unwrap(), cleanup);
    assert_eq!(probe.prepare_calls.load(Ordering::Acquire), prepares);
    assert_eq!(probe.executed.load(Ordering::Acquire), executions);
    assert_eq!(probe.backend_shutdown.load(Ordering::Acquire), 0);
    assert_eq!(probe.program_shutdown.load(Ordering::Acquire), 0);
    assert_eq!(*probe.releases.lock().unwrap(), releases);
    assert_eq!(*probe.drops.lock().unwrap(), drops);
}

#[test]
fn boot_stage_and_backend_require_exact_physical_capacity() {
    let probe = Arc::new(Probe::default());
    let mut boot = boots().remove(0);
    let original = stage(&boot, Arc::clone(&probe)).unwrap();
    boot.physical_pages *= 2;
    assert!(PipelineStageWorker::new(&boot, original).is_err());
    assert_eq!(probe.backend_shutdown.load(Ordering::Relaxed), 1);
    boot.physical_pages += 1;
    assert!(boot.validate().is_err());
    let mut config = config();
    config.page_size = 1;
    config.max_pages = i32::MAX as usize / 2 + 2;
    assert!(
        config
            .physical_pages(PhysicalKvLayout::standard(ModelExecutionBackend::Cuda))
            .is_err()
    );
    assert_eq!(
        config
            .physical_pages(PhysicalKvLayout::standard(ModelExecutionBackend::Cpu))
            .unwrap(),
        config.max_pages
    );
    config.max_pages = usize::MAX;
    assert!(
        config
            .physical_pages(PhysicalKvLayout::standard(ModelExecutionBackend::Cuda))
            .is_err()
    );
}

#[test]
fn parent_rejects_disagreeing_physical_stage_descriptions() {
    struct WrongCapacity(LocalTransport);
    impl PipelineTransport for WrongCapacity {
        fn call_observed(
            &mut self,
            rank: PipelineRank,
            command: PipelineCommand,
            poll: &mut dyn FnMut(),
        ) -> Result<PipelineReply> {
            let mut reply = self.0.call_observed(rank, command, poll)?;
            if rank.local.get() == 1
                && let PipelineReply::Description(description) = &mut reply
            {
                description.physical_pages *= 2;
            }
            Ok(reply)
        }
        fn outstanding(&self) -> usize {
            self.0.outstanding()
        }
        fn shutdown(&mut self) -> Result<()> {
            self.0.shutdown()
        }
        fn quarantine(&mut self) {
            self.0.quarantine();
        }
    }
    let probe = Arc::new(TransportProbe::default());
    let transport = WrongCapacity(LocalTransport::new(
        Arc::new(Probe::default()),
        Arc::clone(&probe),
    ));
    assert!(
        PipelineParallelExecutor::new_with_transport(topology(), plans(), config(), transport)
            .is_err()
    );
    assert_eq!(probe.shutdown.load(Ordering::Relaxed), 1);
}

fn controlled(probe: Arc<Probe>) -> PipelineStageWorker<DeviceBackend, DeviceProgram> {
    let boot = boots().remove(0);
    let mut worker = PipelineStageWorker::new(&boot, stage(&boot, probe).unwrap()).unwrap();
    worker
        .dispatch(PipelineCommand::Create {
            session: SessionId(1),
        })
        .unwrap();
    worker.dispatch(prepare(key())).unwrap();
    worker
}
fn execute_command(cancel: Arc<AtomicBool>) -> PipelineCommand {
    PipelineCommand::Execute {
        key: key(),
        input: SegmentInput::Tokens,
        cancellation: cancel,
    }
}
#[test]
fn working_mutations_never_touch_source_and_frontier_precedes_core() {
    for mode in 0..4 {
        let probe = Arc::new(Probe::default());
        let mut worker = controlled(probe.clone());
        let cancel = Arc::new(AtomicBool::new(false));
        match mode {
            0 => probe.bad_frontier.store(true, Ordering::Relaxed),
            1 => probe.mutate_core.store(true, Ordering::Relaxed),
            2 => probe.fail_after_execute.store(true, Ordering::Relaxed),
            _ => *probe.cancel_after_execute.lock().unwrap() = Some(cancel.clone()),
        }
        assert!(worker.dispatch(execute_command(cancel)).is_err());
        assert!(worker.dispatch(PipelineCommand::Ready(key())).is_err());
        worker.dispatch(PipelineCommand::Rollback(key())).unwrap();
        assert_eq!(probe.releases.lock().unwrap().len(), 1);
        worker
            .dispatch(PipelineCommand::Fork {
                source: SessionId(1),
                target: SessionId(2),
            })
            .unwrap();
        for session in [SessionId(1), SessionId(2)] {
            worker
                .dispatch(PipelineCommand::Release {
                    session,
                    pages: vec![],
                })
                .unwrap();
        }
        let released = probe.releases.lock().unwrap();
        assert_eq!((released[1].1, released[1].2), (0, 0));
        assert_eq!((released[2].1, released[2].2), (0, 0));
        drop(released);
        worker.shutdown().unwrap();
    }
}
#[test]
fn rollback_release_retry_keeps_state_and_never_resubmits_physical_rollback() {
    let probe = Arc::new(Probe::default());
    let mut worker = controlled(probe.clone());
    probe.release_failures.store(1, Ordering::Relaxed);
    assert!(worker.dispatch(PipelineCommand::Rollback(key())).is_err());
    assert!(worker.has_active_custody());
    assert!(probe.drops.lock().unwrap().is_empty());
    assert!(
        worker
            .dispatch(execute_command(Arc::new(AtomicBool::new(false))))
            .is_err()
    );
    assert!(
        worker
            .dispatch(PipelineCommand::Create {
                session: SessionId(2)
            })
            .is_err()
    );
    worker.dispatch(PipelineCommand::Rollback(key())).unwrap();
    assert_eq!(probe.rollbacks.load(Ordering::Relaxed), 1);
    assert_eq!(*probe.drops.lock().unwrap(), vec![1]);
    worker.shutdown().unwrap();
}
#[test]
fn clean_prepare_error_retains_failed_working_cleanup_for_rollback() {
    let probe = Arc::new(Probe::default());
    let boot = boots().remove(0);
    let mut worker = PipelineStageWorker::new(&boot, stage(&boot, probe.clone()).unwrap()).unwrap();
    worker
        .dispatch(PipelineCommand::Create {
            session: SessionId(1),
        })
        .unwrap();
    *probe.prepare_failure.lock().unwrap() = PrepareFailure::Validation;
    probe.release_failures.store(1, Ordering::Relaxed);
    assert!(worker.dispatch(prepare(key())).is_err());
    assert!(worker.has_active_custody());
    assert!(probe.drops.lock().unwrap().is_empty());
    worker.dispatch(PipelineCommand::Rollback(key())).unwrap();
    assert_eq!(probe.rollbacks.load(Ordering::Relaxed), 0);
    worker.shutdown().unwrap();
}
#[test]
fn abort_ack_waits_for_working_release_without_resubmission() {
    let probe = Arc::new(Probe::default());
    let mut worker = controlled(probe.clone());
    probe.release_failures.store(1, Ordering::Relaxed);
    let abort = || PipelineCommand::Abort {
        key: key(),
        generation: 1,
    };
    assert!(worker.dispatch(abort()).is_err());
    assert!(worker.has_active_custody());
    assert!(probe.drops.lock().unwrap().is_empty());
    worker.dispatch(abort()).unwrap();
    assert_eq!(probe.aborts.load(Ordering::Relaxed), 1);
    worker.dispatch(PipelineCommand::Finish(key())).unwrap();
    worker.shutdown().unwrap();
}
#[test]
fn publish_moves_working_and_retirement_ack_waits_for_displaced_release() {
    let probe = Arc::new(Probe::default());
    let mut worker = controlled(probe.clone());
    worker
        .dispatch(execute_command(Arc::new(AtomicBool::new(false))))
        .unwrap();
    for command in [
        PipelineCommand::Ready(key()),
        PipelineCommand::Install {
            key: key(),
            generation: 1,
        },
        PipelineCommand::Publish(key()),
    ] {
        worker.dispatch(command).unwrap();
    }
    assert!(probe.drops.lock().unwrap().is_empty());
    probe.release_failures.store(1, Ordering::Relaxed);
    let retire = || PipelineCommand::Retire {
        key: key(),
        generation: 1,
        pages: vec![],
    };
    assert!(worker.dispatch(retire()).is_err());
    assert!(worker.has_active_custody());
    assert!(probe.drops.lock().unwrap().is_empty());
    worker.dispatch(retire()).unwrap();
    assert_eq!(probe.retires.load(Ordering::Relaxed), 1);
    assert_eq!(*probe.drops.lock().unwrap(), vec![0]);
    worker.dispatch(PipelineCommand::Finish(key())).unwrap();
    worker
        .dispatch(PipelineCommand::Release {
            session: SessionId(1),
            pages: vec![KvPageId(0)],
        })
        .unwrap();
    assert_eq!(*probe.releases.lock().unwrap(), vec![(0, 0, 0), (1, 1, 1)]);
    worker.shutdown().unwrap();
}
#[test]
fn session_release_retry_is_exact_and_does_not_release_pages_twice() {
    let probe = Arc::new(Probe::default());
    let mut worker = controlled(probe.clone());
    worker.dispatch(PipelineCommand::Rollback(key())).unwrap();
    probe.release_failures.store(1, Ordering::Relaxed);
    let release = || PipelineCommand::Release {
        session: SessionId(1),
        pages: vec![],
    };
    assert!(worker.dispatch(release()).is_err());
    assert!(worker.has_active_custody());
    assert!(
        worker
            .dispatch(PipelineCommand::Release {
                session: SessionId(2),
                pages: vec![]
            })
            .is_err()
    );
    assert!(
        worker
            .dispatch(PipelineCommand::Release {
                session: SessionId(1),
                pages: vec![KvPageId(0)]
            })
            .is_err()
    );
    worker.dispatch(release()).unwrap();
    assert_eq!(probe.physical_releases.load(Ordering::Relaxed), 1);
    assert!(!worker.has_active_custody());
    worker.shutdown().unwrap();
}

#[test]
fn root_expert_quiescence_pending_and_unknown_keep_worker_rollback_custody() {
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
    worker.dispatch(prepare(key.clone())).unwrap();
    worker
        .dispatch(PipelineCommand::Execute {
            key: key.clone(),
            input: SegmentInput::Tokens,
            cancellation: Arc::new(AtomicBool::new(false)),
        })
        .unwrap();
    probe
        .expert_quiescence_pending
        .store(true, Ordering::Release);
    let released = probe.releases.lock().unwrap().len();
    for command in [
        PipelineCommand::Abort {
            key: key.clone(),
            generation: 1,
        },
        PipelineCommand::Rollback(key.clone()),
    ] {
        assert!(matches!(
            worker.dispatch(command).unwrap(),
            PipelineReply::Ack(KvEndProgress::Pending)
        ));
        assert!(worker.has_active_custody());
        assert_eq!(probe.rollbacks.load(Ordering::Relaxed), 0);
        assert_eq!(probe.aborts.load(Ordering::Relaxed), 0);
        assert_eq!(probe.releases.lock().unwrap().len(), released);
    }
    probe
        .expert_quiescence_unknown
        .store(true, Ordering::Release);
    for _ in 0..2 {
        let error = worker
            .dispatch(PipelineCommand::Rollback(key.clone()))
            .unwrap_err();
        assert!(KvPrepareQuiescenceUnknown::from_error(&error).is_some());
        assert!(worker.has_active_custody());
        assert_eq!(probe.rollbacks.load(Ordering::Relaxed), 0);
    }
    // The real Unknown transport is permanently quarantined; do not manufacture
    // a later Complete or let this synthetic worker drop physical custody.
    std::mem::forget(worker);
}

#[test]
fn root_expert_pending_can_complete_without_reexecuting_or_releasing_early() {
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
    worker.dispatch(prepare(key.clone())).unwrap();
    probe
        .expert_quiescence_pending
        .store(true, Ordering::Release);
    assert!(matches!(
        worker
            .dispatch(PipelineCommand::Rollback(key.clone()))
            .unwrap(),
        PipelineReply::Ack(KvEndProgress::Pending)
    ));
    assert!(worker.has_active_custody());
    probe
        .expert_quiescence_pending
        .store(false, Ordering::Release);
    assert!(matches!(
        worker.dispatch(PipelineCommand::Rollback(key)).unwrap(),
        PipelineReply::Ack(KvEndProgress::Complete)
    ));
    assert!(!worker.has_active_custody());
    assert_eq!(probe.rollbacks.load(Ordering::Relaxed), 1);
    assert_eq!(probe.executed.load(Ordering::Relaxed), 0);
    worker.shutdown().unwrap();
}

// Exercise the real post-publication command and retained logical reader on CPU;
// the CUDA feature only makes the optional diagnostic protocol available.
#[cfg(feature = "cuda")]
fn trace_pipeline(probe: &Arc<Probe>) -> PipelineParallelExecutor {
    let make = Arc::clone(probe);
    PipelineParallelExecutor::new_with_factory(topology(), boots(), move |boot| {
        Ok(PipelineStageWorker::new(&boot, stage(&boot, make)?)?.boxed())
    })
    .unwrap()
}

#[cfg(feature = "cuda")]
#[test]
fn committed_trace_requires_publication_exact_identity_and_releases_clean_reads() {
    let probe = Arc::new(Probe::default());
    let mut p = trace_pipeline(&probe);
    let session = SessionId(901);
    let absent = ExecutionTransactionId::new(99).unwrap();
    assert!(p.export_committed_trace(session, absent).is_err());
    assert_eq!(probe.committed_traces.load(Ordering::Relaxed), 0);
    let first = p.forward(session, &[1, 2], ForwardPhase::Prefill).unwrap();
    assert!(
        p.export_committed_trace(SessionId(902), first.transaction)
            .is_err()
    );
    assert!(p.export_committed_trace(session, absent).is_err());
    assert_eq!(probe.committed_traces.load(Ordering::Relaxed), 0);
    p.export_committed_trace(session, first.transaction)
        .unwrap();
    assert_eq!(probe.committed_traces.load(Ordering::Relaxed), 2);
    assert_eq!(p.coordinator().publication_count(), 1);
    let next = p.forward(session, &[3], ForwardPhase::Decode).unwrap();
    assert!(
        p.export_committed_trace(session, first.transaction)
            .is_err()
    );
    p.export_committed_trace(session, next.transaction).unwrap();
    p.release_session(session).unwrap();
    assert!(p.export_committed_trace(session, next.transaction).is_err());
    let recreated = p.forward(session, &[1], ForwardPhase::Prefill).unwrap();
    assert!(p.export_committed_trace(session, next.transaction).is_err());
    p.export_committed_trace(session, recreated.transaction)
        .unwrap();
    p.shutdown().unwrap();
    assert_eq!(p.page_manager().allocated_pages(), 0);
}

#[cfg(feature = "cuda")]
#[test]
fn committed_trace_ordinary_failure_releases_lease_unknown_retains_custody() {
    for fault_kind in [1, 2] {
        let probe = Arc::new(Probe::default());
        let mut p = trace_pipeline(&probe);
        let session = SessionId(903);
        let output = p.forward(session, &[1, 2], ForwardPhase::Prefill).unwrap();
        let pages = p.page_manager().allocated_pages();
        probe.trace_fault.store(fault_kind, Ordering::Release);
        let failure = p
            .export_committed_trace(session, output.transaction)
            .unwrap_err();
        assert!(failure.to_string().contains("trace"));
        assert_eq!(probe.committed_traces.load(Ordering::Relaxed), 1);
        assert_eq!(p.coordinator().publication_count(), 1);
        probe.trace_fault.store(0, Ordering::Release);
        if fault_kind == 1 {
            assert!(!p.is_quarantined());
            p.forward(session, &[3], ForwardPhase::Decode).unwrap();
            p.shutdown().unwrap();
            assert_eq!(p.page_manager().allocated_pages(), 0);
        } else {
            assert!(p.is_quarantined());
            assert!(
                p.export_committed_trace(session, output.transaction)
                    .is_err()
            );
            assert!(p.forward(session, &[3], ForwardPhase::Decode).is_err());
            assert!(p.release_session(session).is_err());
            assert!(p.shutdown().is_err());
            assert_eq!(p.page_manager().allocated_pages(), pages);
            assert_eq!(probe.backend_shutdown.load(Ordering::Relaxed), 0);
        }
    }
}
