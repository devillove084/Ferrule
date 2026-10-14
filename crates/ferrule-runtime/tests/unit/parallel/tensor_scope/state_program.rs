//! The decorator uses the real journal and backend, including pre-enter checks.
use super::*;
use crate::SessionId;
use crate::parallel::collective::HostCollectiveLimits;
use crate::parallel::data::ReplicaWorker;
use crate::parallel::pipeline::{
    PipelineCommand, PipelineCommandKey, PipelineConfig, PipelineRank, PipelineReply,
    PipelineStage, PipelineStageBoot, PipelineStageDescription, PipelineStageWorker,
    PipelineStateProgram,
};
use crate::parallel::tensor::decoder_collective::DecoderTensorCollective;
use ferrule_common::execution::*;
use ferrule_common::{
    ParallelGroupId, ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology,
};
use ferrule_model::decoder::{
    CpuKvView, DecoderKvBackend, DecoderKvCapacity, DecoderKvCommitBackend, DecoderKvPageStatus,
    DecoderKvPrepare, DecoderSequence, DecoderSequenceCheckout, DecoderSequenceLifecycle,
    DenseLogits, GenericDecoderSequenceState, KvEndProgress, PagedKvBackend,
    PagedKvTransactionHandle, StandardGqaPlanes, TypedCpuPagedKvPool,
};
use ferrule_model::execution::{ExecutionPrecisionPolicy, SequenceStateCore, SequenceTopologyId};
use ferrule_model::runner::SequenceStateReleaseError;
use ferrule_model::transformer::{StandardTensorCollective, parallel::TensorParallelCollective};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Duration;

#[derive(Default)]
struct Probe {
    enters: Cell<usize>,
    pending: Cell<bool>,
    unknown: Cell<bool>,
    validations: Cell<usize>,
    executions: Cell<usize>,
    frontiers: Cell<usize>,
    rollbacks: Cell<usize>,
    shutdowns: Cell<usize>,
    stats: Cell<usize>,
    checkouts: Cell<usize>,
    releases: RefCell<Vec<usize>>,
    reject_validation: Cell<bool>,
    skip_schedule: Cell<bool>,
    reject_frontier: Cell<bool>,
}
// Deliberately non-Clone and non-Send; working attachments are independent.
struct State {
    inner: GenericDecoderSequenceState,
    frontier: Rc<Cell<usize>>,
}
impl DecoderSequence for State {
    fn topology_id(&self) -> SequenceTopologyId {
        self.inner.topology_id()
    }
    fn core(&self) -> &SequenceStateCore {
        self.inner.core()
    }
    fn core_mut(&mut self) -> &mut SequenceStateCore {
        self.inner.core_mut()
    }
}
struct Lifecycle(Rc<Probe>);
impl DecoderSequenceLifecycle<State> for Lifecycle {
    fn create(&mut self) -> Result<State> {
        Ok(State {
            inner: GenericDecoderSequenceState::new((), ()),
            frontier: Rc::new(Cell::new(0)),
        })
    }
    fn checkout(&mut self, _: DecoderSequenceCheckout, source: &State) -> Result<State> {
        self.0.checkouts.set(self.0.checkouts.get() + 1);
        Ok(State {
            inner: source.inner.transaction_working_copy()?,
            frontier: Rc::new(Cell::new(source.frontier.get())),
        })
    }
    fn logical_fork(&mut self, source: &State, position: usize) -> Result<State> {
        assert_eq!(source.core().position(), position);
        Ok(State {
            inner: source.inner.logical_fork()?,
            frontier: Rc::new(Cell::new(position)),
        })
    }
    fn reset(&mut self, _: &mut State) -> Result<()> {
        panic!("journal must not reset state")
    }
    fn try_release(
        &mut self,
        state: State,
    ) -> std::result::Result<(), SequenceStateReleaseError<State>> {
        self.0.releases.borrow_mut().push(state.core().position());
        Ok(())
    }
}
struct View {
    cpu: CpuKvView,
    probe: Rc<Probe>,
}
struct Backend {
    cpu: PagedKvBackend<TypedCpuPagedKvPool<State>>,
    probe: Rc<Probe>,
}
impl DecoderKvBackend for Backend {
    type SequenceState = State;
    type Transaction = PagedKvTransactionHandle;
    type KvView = View;
    fn configure_capacity(&mut self, n: usize) -> Result<()> {
        self.cpu.configure_capacity(n)
    }
    fn prepare(&mut self, request: DecoderKvPrepare<'_>) -> Result<Self::Transaction> {
        self.cpu.prepare(request)
    }
    fn enter(
        &mut self,
        tx: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
        states: &mut [State],
    ) -> Result<()> {
        self.probe.enters.set(self.probe.enters.get() + 1);
        self.cpu.enter(tx, batch, states)
    }
    fn active_view(&mut self, tx: &mut Self::Transaction) -> Result<View> {
        Ok(View {
            cpu: self.cpu.active_view(tx)?,
            probe: self.probe.clone(),
        })
    }
    fn leave(&mut self, tx: &mut Self::Transaction) -> Result<()> {
        self.cpu.leave(tx)
    }
    fn commit(&mut self, _: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        panic!("use cohort journal")
    }
    fn rollback(&mut self, tx: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        self.probe.rollbacks.set(self.probe.rollbacks.get() + 1);
        self.cpu.rollback(tx)
    }
    fn release(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.cpu.release(pages)
    }
    fn preempt(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.cpu.preempt(pages)
    }
    fn restore(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.cpu.restore(pages)
    }
    fn capacity(&self) -> DecoderKvCapacity {
        self.cpu.capacity()
    }
    fn page_status(&self, page: KvPageId) -> DecoderKvPageStatus {
        self.cpu.page_status(page)
    }
    fn shutdown(&mut self) -> Result<()> {
        self.cpu.shutdown()
    }
}
impl DecoderKvCommitBackend for Backend {
    fn preflight_commit_ready(
        &self,
        tx: &Self::Transaction,
        binding: &KvCommitBinding,
        rank: ParallelRankId,
        states: &[State],
    ) -> Result<()> {
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
        self.cpu.publish_committed(tx)
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
        self.cpu.finish_prepared(tx)
    }
}
struct Program {
    group: Option<crate::parallel::expert::ExpertGroup>,
    plan: LayerSegmentPlan,
    endpoint: DecoderTensorCollective,
    probe: Rc<Probe>,
}
impl PipelineStateProgram for Program {
    type SequenceState = State;
    type Lifecycle = Lifecycle;
    type KvView = View;
    fn plan(&self) -> &LayerSegmentPlan {
        &self.plan
    }
    fn hybrid_expert_group(&self) -> Option<&crate::parallel::expert::ExpertGroup> {
        self.group.as_ref()
    }
    fn poll_quiescence(&mut self) -> Result<bool> {
        if self.probe.unknown.get() {
            return Err(ferrule_model::decoder::KvPrepareQuiescenceUnknown::wrap(
                ExecutionTransactionId::new(1).unwrap(),
                error("injected root/expert completion unknown"),
            ));
        }
        Ok(!self.probe.pending.get())
    }
    fn new_lifecycle(&mut self) -> Result<Lifecycle> {
        Ok(Lifecycle(self.probe.clone()))
    }
    fn validate_execution(
        &self,
        _: &KvCommitBinding,
        _: &PackedDecoderBatch,
        _: &[u64],
    ) -> Result<()> {
        self.probe.validations.set(self.probe.validations.get() + 1);
        if self.probe.reject_validation.get() {
            return Err(error("inner validation rejected"));
        }
        Ok(())
    }
    fn execute_state(
        &mut self,
        batch: &PackedDecoderBatch,
        _: SegmentInput,
        states: &mut [State],
        view: &mut View,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        assert!(Rc::ptr_eq(&self.probe, &view.probe));
        view.cpu.validate_batch(batch)?;
        self.probe.executions.set(self.probe.executions.get() + 1);
        states[0].frontier.set(batch.sequences()[0].sequence_len());
        if !self.probe.skip_schedule.get() {
            self.endpoint.exchange(
                context.transaction,
                8,
                TensorParallelCollective::Sum,
                vec![1.; batch.len() * 4],
            )?;
        }
        Ok(SegmentOutput::Logits(DenseLogits::new(
            batch.len(),
            4,
            vec![1.; batch.len() * 4],
        )?))
    }
    fn validate_frontier(&self, batch: &PackedDecoderBatch, states: &[State]) -> Result<()> {
        self.probe.frontiers.set(self.probe.frontiers.get() + 1);
        assert_eq!(
            states[0].core().position(),
            batch.sequences()[0].context_len()
        );
        if self.probe.reject_frontier.get() {
            return Err(error("inner frontier rejected"));
        }
        assert_eq!(
            states[0].frontier.get(),
            batch.sequences()[0].sequence_len()
        );
        Ok(())
    }
    fn expert_outstanding(&self) -> usize {
        7
    }
    fn owner_stats(&mut self) -> Result<Vec<crate::parallel::expert::ExpertOwnerStats>> {
        self.probe.stats.set(self.probe.stats.get() + 1);
        Ok(vec![])
    }
    fn shutdown(&mut self) -> Result<()> {
        self.probe.shutdowns.set(self.probe.shutdowns.get() + 1);
        Ok(())
    }
}
fn topology() -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(501),
        1,
        ParallelRankId::new(0),
        ParallelismPlan::validated(1, 1, 1, 1, 1, 1).unwrap(),
    )
    .unwrap()
}
fn key(id: u64) -> PipelineCommandKey {
    PipelineCommandKey::new(
        KvCommitBinding::new(
            ExecutionTransactionId::new(id).unwrap(),
            topology().topology_id(),
            topology().participants(),
            id,
        )
        .unwrap(),
        PipelineRank {
            local: ParallelRankId::new(0),
            global: ParallelRankId::new(0),
        },
        SessionId(11),
    )
    .unwrap()
}
fn batch(token: u32) -> ExecutionBatch {
    ExecutionBatch::new(
        ForwardMode::Prefill,
        vec![token],
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
    )
}
fn reservation() -> KvReservationView {
    KvReservationView {
        state_slot: StateSlot::new(0),
        execution_state_slot: StateSlot::new(0),
        positions: 0..1,
        newly_allocated: vec![KvPageId(0)],
        generation: 0,
        execution_generation: 0,
        cow_replacement: None,
    }
}
type Worker = PipelineStageWorker<Backend, TensorScopedStateProgram<Program>>;
fn worker(probe: Rc<Probe>, register: bool) -> (Worker, DecoderTensorCollectiveControl) {
    let endpoint = DecoderTensorCollective::new_group_from_members(
        &[ParallelRankId::new(0)],
        topology().topology_id(),
        ParallelGroupId::new(1),
        HostCollectiveLimits {
            max_ranks: 1,
            max_elements_per_rank: 16,
            max_host_bytes: 4096,
        },
        Duration::from_secs(5),
    )
    .unwrap()
    .pop()
    .unwrap();
    let control = endpoint.control();
    if register {
        control
            .register_sites(
                endpoint.owner(),
                vec![super::semantic_tests::site(
                    8,
                    Some(0),
                    TensorCollectiveSemantic::GatedDeltaOutput,
                    TensorCollectiveShape::PerRow(4),
                )],
            )
            .unwrap();
    }
    let config = PipelineConfig {
        page_size: 2,
        max_pages: 8,
        max_positions: 8,
        max_batch_tokens: 4,
        session_capacity: 2,
        max_parameter_bytes: 1024,
        precision: ExecutionPrecisionPolicy::f32(),
        max_ack_polls: 2,
    };
    let plan = LayerSegmentPlan::new(1, 0..1, true, true).unwrap();
    let description = PipelineStageDescription {
        plan: plan.clone(),
        config,
        physical_pages: 8,
        hidden: 4,
        vocabulary: 4,
        kv_heads: 1,
        head_dim: 2,
        expert_group: None,
    };
    let planes = StandardGqaPlanes::new(1, 1, 2, 2, 8, KvElementType::F32).unwrap();
    let backend = Backend {
        cpu: PagedKvBackend::new(TypedCpuPagedKvPool::from_strategy(&planes, 8).unwrap()),
        probe: probe.clone(),
    };
    let program = TensorScopedStateProgram::new(
        Program {
            group: None,
            plan: plan.clone(),
            endpoint,
            probe,
        },
        control.clone(),
        ParallelRankId::new(0),
    );
    let capabilities = description.execution_capabilities().unwrap();
    let stage = PipelineStage::new(program, backend, description, capabilities).unwrap();
    let boot = PipelineStageBoot {
        rank: key(1).rank(),
        plan,
        config,
        physical_pages: 8,
        program_spec: vec![],
    };
    let mut worker = PipelineStageWorker::new(&boot, stage).unwrap();
    assert!(matches!(
        worker
            .dispatch(PipelineCommand::Create {
                session: SessionId(11)
            })
            .unwrap(),
        PipelineReply::Generation(0)
    ));
    (worker, control)
}
fn sealed_identity(worker: &Worker, id: u64, token: u32, sessions: &[u64]) -> TensorBatchIdentity {
    let packed = PackedDecoderBatch::lower(
        &batch(token),
        &[reservation()],
        &[GenericDecoderSequenceState::new((), ())],
        &worker.boot_description().execution_capabilities().unwrap(),
        2,
        &|_| DecoderKvPageStatus::Vacant,
    )
    .unwrap();
    TensorBatchIdentity::from_packed(key(id).binding(), &packed, sessions).unwrap()
}
fn prepare(worker: &mut Worker, id: u64) {
    assert!(matches!(
        worker
            .dispatch(PipelineCommand::Prepare {
                key: key(id),
                batch: Box::new(batch(1)),
                reservation: reservation()
            })
            .unwrap(),
        PipelineReply::Prepared { .. }
    ));
}
fn execute(worker: &mut Worker, id: u64) -> Result<PipelineReply> {
    worker.dispatch(PipelineCommand::Execute {
        key: key(id),
        input: SegmentInput::Tokens,
        cancellation: Arc::new(AtomicBool::new(false)),
    })
}
fn ack(worker: &mut Worker, command: PipelineCommand) {
    assert!(matches!(
        worker.dispatch(command).unwrap(),
        PipelineReply::Ack(KvEndProgress::Complete)
    ));
}

#[test]
fn state_identity_and_inner_validation_reject_before_enter_and_retain_journal() {
    for case in 0..5 {
        let probe = Rc::new(Probe::default());
        let (mut worker, control) = worker(probe.clone(), case != 3);
        prepare(&mut worker, 1);
        if case == 3 {
            control
                .begin(
                    key(1).binding().transaction(),
                    Arc::new(AtomicBool::new(false)),
                )
                .unwrap();
        } else {
            let identity = sealed_identity(
                &worker,
                if case == 2 { 2 } else { 1 },
                if case == 0 { 2 } else { 1 },
                if case == 1 { &[12] } else { &[11] },
            );
            control
                .begin_sealed(identity, Arc::new(AtomicBool::new(false)))
                .unwrap();
        }
        probe.reject_validation.set(case == 4);
        assert!(execute(&mut worker, 1).is_err());
        assert_eq!(probe.enters.get(), 0);
        assert_eq!(probe.executions.get(), 0);
        assert_eq!(probe.validations.get(), usize::from(case == 4));
        assert!(control.is_failed());
        assert!(worker.has_active_custody());
        ack(&mut worker, PipelineCommand::Rollback(key(1)));
        ack(&mut worker, PipelineCommand::Rollback(key(1)));
        assert_eq!(probe.rollbacks.get(), 1);
        assert_eq!(*probe.releases.borrow(), vec![0]);
        assert!(!worker.has_active_custody());
        worker.shutdown().unwrap();
        assert_eq!(probe.shutdowns.get(), 1);
    }
}

#[test]
fn skipped_schedule_cannot_return_success_or_ready_and_frontier_is_forwarded() {
    for skip in [false, true] {
        let probe = Rc::new(Probe::default());
        let (mut worker, control) = worker(probe.clone(), true);
        prepare(&mut worker, 1);
        control
            .begin_sealed(
                sealed_identity(&worker, 1, 1, &[11]),
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
        probe.skip_schedule.set(skip);
        probe.reject_frontier.set(!skip);
        let failure = execute(&mut worker, 1).unwrap_err().to_string();
        assert!(
            failure.contains(if skip {
                "schedule completion"
            } else {
                "frontier rejected"
            }),
            "{failure}"
        );
        assert_eq!(probe.enters.get(), 1);
        assert_eq!(probe.executions.get(), 1);
        assert_eq!(probe.frontiers.get(), usize::from(!skip));
        assert!(worker.dispatch(PipelineCommand::Ready(key(1))).is_err());
        assert!(worker.has_active_custody());
        ack(&mut worker, PipelineCommand::Rollback(key(1)));
        assert_eq!(*probe.releases.borrow(), vec![0]);
        worker.shutdown().unwrap();
    }
}

#[test]
fn successful_state_program_keeps_checkout_publish_retire_and_shutdown_in_journal() {
    let probe = Rc::new(Probe::default());
    let (mut worker, control) = worker(probe.clone(), true);
    // A successfully computed but discarded step remains retryable with a newer ID.
    for id in [1, 2] {
        prepare(&mut worker, id);
        control
            .begin_sealed(
                sealed_identity(&worker, id, 1, &[11]),
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
        assert!(matches!(
            execute(&mut worker, id).unwrap(),
            PipelineReply::Executed { .. }
        ));
        control.finish(key(id).binding().transaction()).unwrap();
        assert!(worker.has_active_custody());
        if id == 1 {
            ack(&mut worker, PipelineCommand::Rollback(key(id)));
        } else {
            ack(&mut worker, PipelineCommand::Ready(key(id)));
            ack(
                &mut worker,
                PipelineCommand::Install {
                    key: key(id),
                    generation: id,
                },
            );
            ack(&mut worker, PipelineCommand::Publish(key(id)));
            ack(
                &mut worker,
                PipelineCommand::CheckRetirement {
                    key: key(id),
                    pages: vec![],
                },
            );
            ack(
                &mut worker,
                PipelineCommand::Retire {
                    key: key(id),
                    generation: id,
                    pages: vec![],
                },
            );
            ack(&mut worker, PipelineCommand::Finish(key(id)));
        }
        assert!(!worker.has_active_custody());
    }
    let PipelineReply::Stats(stats) = worker.dispatch(PipelineCommand::Stats).unwrap() else {
        panic!()
    };
    assert_eq!(stats.expert_outstanding, 7);
    assert_eq!(probe.stats.get(), 1);
    assert_eq!(probe.checkouts.get(), 2);
    assert_eq!(probe.frontiers.get(), 2);
    assert_eq!(*probe.releases.borrow(), vec![1, 0]);
    ack(
        &mut worker,
        PipelineCommand::Release {
            session: SessionId(11),
            pages: vec![KvPageId(0)],
        },
    );
    assert_eq!(*probe.releases.borrow(), vec![1, 0, 1]);
    worker.shutdown().unwrap();
    assert_eq!(probe.shutdowns.get(), 1);
    assert!(!control.is_failed());
}

#[test]
fn cancelled_seal_is_rejected_before_enter_even_with_a_different_command_flag() {
    let probe = Rc::new(Probe::default());
    let (mut worker, control) = worker(probe.clone(), true);
    prepare(&mut worker, 1);
    let cancellation = Arc::new(AtomicBool::new(false));
    control
        .begin_sealed(sealed_identity(&worker, 1, 1, &[11]), cancellation.clone())
        .unwrap();
    cancellation.store(true, std::sync::atomic::Ordering::Release);
    assert!(execute(&mut worker, 1).is_err());
    assert_eq!(probe.enters.get(), 0);
    assert_eq!(probe.validations.get(), 0);
    assert!(control.is_failed());
    assert!(worker.has_active_custody());
    ack(&mut worker, PipelineCommand::Rollback(key(1)));
    worker.shutdown().unwrap();
}

#[test]
fn tensor_state_decorator_retains_journal_on_pending_and_unknown_root_experts() {
    let probe = Rc::new(Probe::default());
    let (mut worker, control) = worker(probe.clone(), true);
    prepare(&mut worker, 1);
    control
        .begin_sealed(
            sealed_identity(&worker, 1, 1, &[11]),
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    execute(&mut worker, 1).unwrap();
    control.finish(key(1).binding().transaction()).unwrap();
    probe.pending.set(true);
    for command in [
        PipelineCommand::Ready(key(1)),
        PipelineCommand::Rollback(key(1)),
    ] {
        assert!(matches!(
            worker.dispatch(command).unwrap(),
            PipelineReply::Ack(KvEndProgress::Pending)
        ));
        assert!(worker.has_active_custody());
        assert_eq!(probe.rollbacks.get(), 0);
        assert!(probe.releases.borrow().is_empty());
    }
    probe.unknown.set(true);
    assert!(worker.dispatch(PipelineCommand::Rollback(key(1))).is_err());
    assert!(worker.has_active_custody());
    assert_eq!(probe.rollbacks.get(), 0);
    // This injected error has no physical unknown work; allow test cleanup only.
    probe.unknown.set(false);
    probe.pending.set(false);
    ack(&mut worker, PipelineCommand::Rollback(key(1)));
    assert!(!worker.has_active_custody());
    worker.shutdown().unwrap();
}

#[test]
fn tensor_state_decorator_forwards_production_source_attachment() {
    let group = crate::parallel::expert::ExpertGroup {
        source_rank: ParallelRankId::new(0),
        members: vec![ParallelRankId::new(2), ParallelRankId::new(3)],
        layers: 0..1,
        placement: ferrule_model::transformer::ExpertPlacement::new([
            (0, 0, ParallelRankId::new(2)),
            (0, 1, ParallelRankId::new(3)),
        ])
        .unwrap(),
        limits: ferrule_model::transformer::ExpertDispatchLimits {
            max_tokens: 4,
            max_bytes: 64,
        },
    };
    let mut peers = DecoderTensorCollective::new_group_from_members(
        &[ParallelRankId::new(0)],
        topology().topology_id(),
        ParallelGroupId::new(1),
        HostCollectiveLimits {
            max_ranks: 1,
            max_elements_per_rank: 16,
            max_host_bytes: 4096,
        },
        Duration::from_secs(5),
    )
    .unwrap();
    let endpoint = peers.pop().unwrap();
    let control = endpoint.control();
    let program = TensorScopedStateProgram::new(
        Program {
            group: Some(group.clone()),
            plan: LayerSegmentPlan::new(1, 0..1, true, true).unwrap(),
            endpoint,
            probe: Rc::new(Probe::default()),
        },
        control,
        ParallelRankId::new(0),
    );
    assert_eq!(program.hybrid_expert_group(), Some(&group));
}
