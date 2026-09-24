use super::*;
use crate::decoder::*;
use crate::runner::{
    ModelInfo, MultiSessionRunner, ResidentModelRunner, TransactionEndIntent,
    TransactionEndProgress,
};
use crate::tokenizer::TokenizerHandle;
use crate::{AttentionKind, ModelFamily, WeightSource};
use ferrule_common::CompletionHub;
use ferrule_common::execution::*;
use std::cell::Cell;
use std::convert::Infallible;
use std::num::NonZeroU32;

type State = DecoderSequenceState<(), u32>;
type Backend = PagedKvBackend<TypedCpuPagedKvPool<State>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Completion {
    Unknown,
    Pending,
    Complete,
}

#[derive(Debug)]
struct Control {
    fail: &'static str,
    pending: &'static str,
    wait: bool,
    completion: Completion,
    fail_begin: bool,
    fail_abort: bool,
    fail_cancel: bool,
    begin_calls: usize,
    abort_calls: usize,
    cancel_calls: usize,
    quiescence_polls: Vec<usize>,
    hidden_drops: Rc<Cell<usize>>,
    pending_drops: Rc<Cell<usize>>,
}
impl Control {
    fn new(fail: &'static str) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            fail,
            pending: "",
            wait: false,
            completion: Completion::Unknown,
            fail_begin: false,
            fail_abort: false,
            fail_cancel: false,
            begin_calls: 0,
            abort_calls: 0,
            cancel_calls: 0,
            quiescence_polls: Vec::new(),
            hidden_drops: Rc::new(Cell::new(0)),
            pending_drops: Rc::new(Cell::new(0)),
        }))
    }
}
#[derive(Debug)]
struct Held(Rc<Cell<usize>>);
impl Drop for Held {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}
// Non-Clone state models owned layer rows and a deferred closure.
struct Work {
    rows: Vec<f32>,
    closure: Box<dyn Fn()>,
    held: Held,
}
impl Debug for Work {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Work")
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}
struct FaultModule(Rc<RefCell<Control>>);
impl FaultModule {
    fn check(&self, at: &str) -> Result<()> {
        if self.0.borrow().fail == at {
            Err(transformer_error(format!("injected {at}")))
        } else {
            Ok(())
        }
    }
    fn pending(&self) -> Pending<Work> {
        let work = Work {
            rows: vec![13.0, 17.0],
            closure: Box::new(|| {}),
            held: Held(self.0.borrow().pending_drops.clone()),
        };
        if self.0.borrow().wait {
            Pending::waiting(
                work,
                DecoderWait::Dependencies(
                    ferrule_common::DependencySet::new([
                        ferrule_common::LogicalDependency::operation_retired(
                            ferrule_common::OperationId::new(1),
                        )
                        .unwrap(),
                    ])
                    .unwrap(),
                ),
            )
        } else {
            Pending::physical(work)
        }
    }
    fn poll_cancel(&self, pending: &mut Work) -> Result<bool> {
        assert_eq!(pending.rows, [13.0, 17.0]);
        assert_eq!(pending.held.0.get(), 0);
        (pending.closure)();
        self.completion()
    }
    fn completion(&self) -> Result<bool> {
        match self.0.borrow().completion {
            Completion::Unknown => Err(transformer_error("completion unknown")),
            Completion::Pending => Ok(false),
            Completion::Complete => Ok(true),
        }
    }
    fn cancel(&self, _pending: Work) -> Result<()> {
        let mut control = self.0.borrow_mut();
        control.cancel_calls += 1;
        if control.fail_cancel {
            Err(transformer_error("injected cancel"))
        } else {
            Ok(())
        }
    }
    fn logits(batch: &PackedDecoderBatch) -> DecoderLogits {
        DecoderLogits::Dense(DenseLogits::from_rows(vec![vec![1.0, 2.0]; batch.len()]).unwrap())
    }
}
impl TransformerModule for FaultModule {
    type State = State;
    type KvView = CpuKvView;
    type Hidden = Held;
    type ArenaKey = usize;
    type Arena = ();
    type EmbeddingPending = Work;
    type LayerPending = Work;
    type OutputPending = Work;
    type Quiescence = usize;
    type Event = Infallible;
    type TerminalGuard = NoTerminalGuard;

    fn layer_count(&self) -> usize {
        1
    }
    fn validate(
        &mut self,
        _: &DecoderTransactionContext,
        _: &PackedDecoderBatch,
        _: &mut [State],
        _: &mut CpuKvView,
    ) -> Result<()> {
        self.check("validate")
    }
    fn arena_key(&self, batch: &PackedDecoderBatch) -> Result<usize> {
        Ok(batch.len())
    }
    fn build_arena(&mut self, _: &usize) -> Result<()> {
        Ok(())
    }
    fn submit_embedding(
        &mut self,
        _: &DecoderTransactionContext,
        _: &PackedDecoderBatch,
        _: &mut [State],
        _: &mut CpuKvView,
        _: &mut (),
    ) -> Result<Step<Held, Work>> {
        self.check("embedding")?;
        if self.0.borrow().pending == "embedding" {
            Ok(Step::Pending(self.pending()))
        } else {
            Ok(Step::Complete(Held(self.0.borrow().hidden_drops.clone())))
        }
    }
    fn poll_embedding(
        &mut self,
        _: &DecoderTransactionContext,
        _: &PackedDecoderBatch,
        _: &mut [State],
        _: &mut CpuKvView,
        _: &mut (),
        _: &mut Work,
    ) -> Result<Poll<Held, Work>> {
        self.check("poll-embedding")?;
        Ok(Poll::Ready(Held(self.0.borrow().hidden_drops.clone())))
    }
    fn start_layer(
        &mut self,
        _: &DecoderTransactionContext,
        _: usize,
        request: LayerRequest<'_, State, CpuKvView>,
        _: &mut Held,
        _: &mut (),
    ) -> Result<Step<Vec<Infallible>, Work>> {
        let LayerRequest::Target { states, .. } = request else {
            unreachable!()
        };
        *states[0].kv_state_mut() = 99;
        self.check("layer")?;
        if self.0.borrow().pending == "layer" {
            Ok(Step::Pending(self.pending()))
        } else {
            Ok(Step::Complete(vec![]))
        }
    }
    fn poll_layer(
        &mut self,
        _: &DecoderTransactionContext,
        _: usize,
        _: LayerRequest<'_, State, CpuKvView>,
        _: &mut Held,
        _: &mut (),
        _: &mut Work,
    ) -> Result<Poll<Vec<Infallible>, Work>> {
        self.check("poll-layer")?;
        Ok(Poll::Ready(vec![]))
    }
    fn post_layer(&mut self, _: usize, _: &PackedDecoderBatch, _: &Held, _: &mut ()) -> Result<()> {
        self.check("post-layer")
    }
    fn submit_output(
        &mut self,
        _: &DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        _: &mut [State],
        _: &mut CpuKvView,
        _: &Held,
        _: &mut (),
    ) -> Result<Step<DecoderLogits, Work>> {
        self.check("output")?;
        if self.0.borrow().pending == "output" {
            Ok(Step::Pending(self.pending()))
        } else {
            Ok(Step::Complete(Self::logits(batch)))
        }
    }
    fn poll_output(
        &mut self,
        _: &DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        _: &mut [State],
        _: &mut CpuKvView,
        _: &Held,
        _: &mut (),
        _: &mut Work,
    ) -> Result<Poll<DecoderLogits, Work>> {
        self.check("poll-output")?;
        Ok(Poll::Ready(Self::logits(batch)))
    }
    fn poll_embedding_cancel(&mut self, pending: &mut Work) -> Result<bool> {
        self.poll_cancel(pending)
    }
    fn poll_layer_cancel(&mut self, _: usize, pending: &mut Work) -> Result<bool> {
        self.poll_cancel(pending)
    }
    fn poll_output_cancel(&mut self, pending: &mut Work) -> Result<bool> {
        self.poll_cancel(pending)
    }
    fn cancel_embedding(&mut self, pending: Work) -> Result<()> {
        self.cancel(pending)
    }
    fn cancel_layer(&mut self, _: usize, pending: Work) -> Result<()> {
        self.cancel(pending)
    }
    fn cancel_output(&mut self, pending: Work) -> Result<()> {
        self.cancel(pending)
    }
    fn begin_quiescence(&mut self) -> Result<usize> {
        let mut c = self.0.borrow_mut();
        c.begin_calls += 1;
        if c.fail_begin {
            Err(transformer_error("injected begin-quiescence"))
        } else {
            Ok(c.begin_calls)
        }
    }
    fn poll_quiescence(&mut self, token: &mut usize) -> Result<bool> {
        self.0.borrow_mut().quiescence_polls.push(*token);
        self.completion()
    }
    fn finish(
        &mut self,
        _: &PackedDecoderBatch,
        _: &mut [State],
        _: &mut CpuKvView,
        _: &mut (),
        _: Vec<Infallible>,
    ) -> Result<NoTerminalGuard> {
        self.check("finish")?;
        Ok(NoTerminalGuard)
    }
    fn abort(
        &mut self,
        _: &PackedDecoderBatch,
        _: &mut [State],
        _: &mut CpuKvView,
        _: &mut (),
    ) -> Result<()> {
        let mut c = self.0.borrow_mut();
        c.abort_calls += 1;
        if c.fail_abort {
            Err(transformer_error("injected abort"))
        } else {
            Ok(())
        }
    }
}
struct Composition;
impl DecoderComposition for Composition {
    type Resources = ();
    type SequenceState = State;
    type KvBackend = Backend;
    type ForwardExecutor = TransformerForwardExecutor<FaultModule>;
    type Proposal = NoProposal;
    type SequenceLifecycle = StandardSequenceLifecycle<(), u32>;
    type ResourceManager = NoResourceManager;
    type Observer = StandardDecoderObserver;
    type Snapshot = GenericDecoderObservabilitySnapshot;
    type Continuation = TransformerContinuation<FaultModule>;
    type ProposalContinuation = Infallible;
    type TerminalGuard = NoTerminalGuard;
}
fn capabilities() -> ExecutionCapabilities {
    ExecutionCapabilities {
        max_batch_tokens: 2,
        max_sequences: 1,
        max_prefill_query_tokens_per_sequence: 2,
        max_decode_query_tokens_per_sequence: 1,
        max_top_k: NonZeroU32::new(2),
        supports_prefill: true,
        supports_decode: true,
        supports_mixed: true,
        full_logits_width: NonZeroU32::new(2),
        kv_binding_mode: KvBindingMode::Paged,
        logits_row_policy: LogitsRowPolicy::Any,
    }
}
fn runner(control: Rc<RefCell<Control>>) -> GenericDecoderRunner<Composition> {
    let strategy = StandardGqaPlanes::new(1, 1, 1, 2, 4, KvElementType::F32).unwrap();
    GenericDecoderRunner::from_runtime(DecoderComponents {
        resources: (),
        model_info: ModelInfo {
            family: ModelFamily::Unknown("forward-fault".into()),
            architecture: None,
            attention: AttentionKind::DenseMha,
            weight_source: WeightSource::Safetensors,
            hidden_size: 2,
            num_layers: 1,
            num_experts: 0,
            num_experts_per_tok: 0,
            vocab_size: 2,
            backend: "cpu",
        },
        tokenizer: TokenizerHandle::from_parts(
            tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            None,
        ),
        capabilities: capabilities(),
        page_size: 2,
        prepared_plan_id: 1,
        forward_executor: TransformerForwardExecutor::new(FaultModule(control)),
        backend: Backend::new(TypedCpuPagedKvPool::from_strategy(&strategy, 4).unwrap()),
        default_state: State::new((), 0),
        sequence_lifecycle: StandardSequenceLifecycle::new(),
        proposal: NoProposal,
        resource_manager: NoResourceManager,
        observer: StandardDecoderObserver,
        completion_hub: CompletionHub::new(),
        completion_reactors: vec![],
    })
    .unwrap()
}
fn input<S: DecoderSequence>(state: &S) -> (ExecutionBatch, Vec<KvReservationView>) {
    (
        ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1],
            vec![0],
            vec![Some(KvWriteSlot::new(2))],
            vec![LogitsRequest::Full],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Prefill,
                0..1,
                0,
                1,
                0..1,
            )],
            vec![KvBlockId::new(1)],
        ),
        vec![KvReservationView {
            state_slot: StateSlot::new(0),
            execution_state_slot: StateSlot::new(0),
            positions: 0..1,
            newly_allocated: vec![KvPageId(1)],
            generation: state.core().generation(),
            execution_generation: state.core().generation(),
            cow_replacement: None,
        }],
    )
}
fn tx() -> ExecutionTransactionId {
    ExecutionTransactionId::new(1).unwrap()
}
fn assert_custody(
    runner: &mut GenericDecoderRunner<Composition>,
    states: &[State],
    original: &[State],
) {
    assert_eq!(states, original);
    assert_eq!(runner.observability_snapshot().active_transactions, 1);
    assert_eq!(runner.backend().capacity().active_transactions, 1);
    assert_eq!(runner.backend().pool().active_transaction_count(), 1);
    assert_eq!(runner.backend().pool().free_slot_count(), 3);
    assert!(runner.release_kv_pages(&[KvPageId(1)]).is_err());
    // Bypass the runner's registry to check the physical pool's pins as well.
    let mut probe = runner.backend().pool().clone();
    assert!(probe.release(&[KvPageId(1)]).is_err());
    assert!(probe.preempt(&[KvPageId(1)]).is_err());
    assert!(runner.shutdown().is_err());
}

#[test]
fn failed_active_runner_preserves_kv_custody_until_module_cancel_complete() {
    for failure in ["embedding", "layer", "post-layer", "output", "finish"] {
        let control = Control::new(failure);
        let mut runner = runner(control.clone());
        let mut states = vec![State::new((), 0)];
        let original = states.clone();
        let (batch, reservations) = input(&states[0]);
        runner
            .prepare_multi_session_batch(tx(), &mut states, &batch, &reservations)
            .unwrap();
        assert!(
            runner
                .execute_multi_session_batch_progress(tx(), &mut states, &batch)
                .unwrap_err()
                .to_string()
                .contains(failure)
        );
        for _ in 0..3 {
            assert!(
                runner
                    .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
                    .unwrap_err()
                    .to_string()
                    .contains("completion unknown")
            );
            assert_custody(&mut runner, &states, &original);
            assert_eq!(control.borrow().hidden_drops.get(), 0);
            assert_eq!(control.borrow().abort_calls, 0);
            assert!(runner.forward_executor().arenas.inner.borrow().is_empty());
        }
        assert!(
            runner
                .end_transaction(tx(), &mut states, TransactionEndIntent::Publish)
                .is_err()
        );
        control.borrow_mut().completion = Completion::Pending;
        for _ in 0..2 {
            assert_eq!(
                runner
                    .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
                    .unwrap(),
                TransactionEndProgress::Pending
            );
            assert_custody(&mut runner, &states, &original);
        }
        control.borrow_mut().completion = Completion::Complete;
        assert_eq!(
            runner
                .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
                .unwrap(),
            TransactionEndProgress::Complete
        );
        assert_eq!(states, original);
        assert_eq!(runner.observability_snapshot().active_transactions, 0);
        assert_eq!(runner.backend().capacity().active_transactions, 0);
        assert_eq!(runner.backend().pool().active_transaction_count(), 0);
        assert_eq!(runner.backend().pool().free_slot_count(), 4);
        let c = control.borrow();
        assert_eq!(c.hidden_drops.get(), usize::from(failure != "embedding"));
        assert_eq!(c.begin_calls, 1);
        assert_eq!(c.abort_calls, 1);
        assert_eq!(c.quiescence_polls, [1; 6]);
        assert!(!runner.forward_executor().arenas.inner.borrow().is_empty());
    }
}

#[test]
fn pending_poll_errors_keep_exact_rows_closure_and_cancel_obligations() {
    for (pending, failure) in [
        ("embedding", "poll-embedding"),
        ("layer", "poll-layer"),
        ("layer", "post-layer"),
        ("output", "poll-output"),
        ("output", "finish"),
    ] {
        let control = Control::new(failure);
        control.borrow_mut().pending = pending;
        let mut runner = runner(control.clone());
        let mut states = vec![State::new((), 0)];
        let original = states.clone();
        let (batch, reservations) = input(&states[0]);
        runner
            .prepare_multi_session_batch(tx(), &mut states, &batch, &reservations)
            .unwrap();
        assert!(
            runner
                .execute_multi_session_batch_progress(tx(), &mut states, &batch)
                .is_err()
        );
        for completion in [
            Completion::Unknown,
            Completion::Unknown,
            Completion::Pending,
        ] {
            control.borrow_mut().completion = completion;
            let result = runner.end_transaction(tx(), &mut states, TransactionEndIntent::Abort);
            if completion == Completion::Unknown {
                assert!(result.is_err());
            } else {
                assert_eq!(result.unwrap(), TransactionEndProgress::Pending);
            }
            assert_custody(&mut runner, &states, &original);
            assert_eq!(control.borrow().pending_drops.get(), 0);
            assert_eq!(control.borrow().cancel_calls, 0);
            assert_eq!(control.borrow().begin_calls, 0);
        }
        control.borrow_mut().completion = Completion::Complete;
        assert_eq!(
            runner
                .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
                .unwrap(),
            TransactionEndProgress::Complete
        );
        assert_eq!(control.borrow().pending_drops.get(), 1);
        assert_eq!(control.borrow().cancel_calls, 1);
        assert_eq!(control.borrow().abort_calls, 1);
    }
}

#[test]
fn cancellation_errors_retry_only_unfinished_obligations() {
    for failure in ["begin", "abort", "cancel"] {
        let control = Control::new("poll-layer");
        {
            let mut c = control.borrow_mut();
            c.pending = "layer";
            c.completion = Completion::Complete;
            c.fail_begin = failure == "begin";
            c.fail_abort = failure == "abort";
            c.fail_cancel = failure == "cancel";
        }
        let mut runner = runner(control.clone());
        let mut states = vec![State::new((), 0)];
        let original = states.clone();
        let (batch, reservations) = input(&states[0]);
        runner
            .prepare_multi_session_batch(tx(), &mut states, &batch, &reservations)
            .unwrap();
        assert!(
            runner
                .execute_multi_session_batch_progress(tx(), &mut states, &batch)
                .is_err()
        );
        assert!(
            runner
                .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
                .is_err()
        );
        assert_custody(&mut runner, &states, &original);
        {
            let mut c = control.borrow_mut();
            c.fail_begin = false;
            c.fail_abort = false;
            c.fail_cancel = false;
        }
        assert_eq!(
            runner
                .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
                .unwrap(),
            TransactionEndProgress::Complete
        );
        let c = control.borrow();
        assert_eq!(c.cancel_calls, 1);
        assert_eq!(c.begin_calls, if failure == "begin" { 2 } else { 1 });
        assert_eq!(c.quiescence_polls.len(), 1);
        assert_eq!(c.abort_calls, if failure == "abort" { 2 } else { 1 });
    }
}

#[test]
fn finish_error_retains_rows_and_finished_cancel_is_idempotent() {
    for fail in ["finish", ""] {
        let control = Control::new(fail);
        let mut executor = TransformerForwardExecutor::new(FaultModule(control.clone()));
        let mut states = vec![State::new((), 0)];
        let (input, reservations) = input(&states[0]);
        let batch =
            PackedDecoderBatch::lower(&input, &reservations, &states, &capabilities(), 2, &|_| {
                DecoderKvPageStatus::Vacant
            })
            .unwrap();
        let context = DecoderTransactionContext::new(
            tx(),
            DecoderControlPlane::new(CompletionHub::new()),
            vec![],
        );
        let mut state = executor.begin_state(&batch).unwrap();
        let strategy = StandardGqaPlanes::new(1, 1, 1, 2, 4, KvElementType::F32).unwrap();
        let mut backend = Backend::new(TypedCpuPagedKvPool::from_strategy(&strategy, 4).unwrap());
        let mut registry = PackedTransactionRegistry::<Composition>::new("forward-test");
        registry
            .prepare(tx(), batch.clone(), &states, &mut backend)
            .unwrap();
        let mut transaction = registry.take_kv_transaction(tx()).unwrap();
        backend
            .enter(&mut transaction, &batch, &mut states)
            .unwrap();
        let mut kv = backend.active_view(&mut transaction).unwrap();
        let result = executor.drive(&context, &batch, &mut states, &mut kv, &mut state, false);
        if fail == "finish" {
            assert!(result.is_err());
            assert!(matches!(state.phase, PhysicalPhase::Finishing(_)));
            assert!(state.logits.is_some());
            state.failed = true;
            assert!(
                executor
                    .poll_cancel(&batch, &mut states, &mut kv, &mut state)
                    .is_err()
            );
            assert!(state.logits.is_some());
            control.borrow_mut().completion = Completion::Complete;
            assert_eq!(
                executor
                    .poll_cancel(&batch, &mut states, &mut kv, &mut state)
                    .unwrap(),
                DecoderCancelProgress::Complete
            );
        } else {
            assert!(matches!(result.unwrap(), DriveProgress::Complete { .. }));
        }
        assert!(matches!(state.phase, PhysicalPhase::Finished));
        assert!(state.logits.is_none());
        let calls = control.borrow().abort_calls;
        assert_eq!(
            executor
                .poll_cancel(&batch, &mut states, &mut kv, &mut state)
                .unwrap(),
            DecoderCancelProgress::Complete
        );
        assert_eq!(control.borrow().abort_calls, calls);
        drop(kv);
        backend.leave(&mut transaction).unwrap();
        registry.restore_kv_transaction(tx(), transaction).unwrap();
        assert_eq!(
            registry.abort(tx(), &mut backend, &mut executor).unwrap(),
            KvEndProgress::Complete
        );
    }
}

#[cfg(feature = "cuda")]
#[path = "cuda_tests.rs"]
mod cuda;

#[test]
fn waiting_resume_errors_and_normal_cancel_retain_pending_custody() {
    for (pending, poll) in [
        ("embedding", "poll-embedding"),
        ("layer", "poll-layer"),
        ("output", "poll-output"),
    ] {
        for failure in [poll, "validate", "cancel", "success"] {
            let control = Control::new("");
            control.borrow_mut().pending = pending;
            control.borrow_mut().wait = true;
            let mut executor = TransformerForwardExecutor::new(FaultModule(control.clone()));
            let mut states = vec![State::new((), 0)];
            let (input, reservations) = input(&states[0]);
            let batch = PackedDecoderBatch::lower(
                &input,
                &reservations,
                &states,
                &capabilities(),
                2,
                &|_| DecoderKvPageStatus::Vacant,
            )
            .unwrap();
            let strategy = StandardGqaPlanes::new(1, 1, 1, 2, 4, KvElementType::F32).unwrap();
            let mut backend =
                Backend::new(TypedCpuPagedKvPool::from_strategy(&strategy, 4).unwrap());
            let mut registry = PackedTransactionRegistry::<Composition>::new("forward-resume");
            registry
                .prepare(tx(), batch.clone(), &states, &mut backend)
                .unwrap();
            let mut transaction = registry.take_kv_transaction(tx()).unwrap();
            backend
                .enter(&mut transaction, &batch, &mut states)
                .unwrap();
            let mut kv = backend.active_view(&mut transaction).unwrap();
            let mut context = DecoderTransactionContext::new(
                tx(),
                DecoderControlPlane::new(CompletionHub::new()),
                vec![],
            );
            let DecoderForwardProgress::Waiting {
                mut continuation, ..
            } = executor
                .start_forward(&mut context, &batch, &mut states, &mut kv)
                .unwrap()
            else {
                panic!("expected a real wait edge")
            };
            assert!(!continuation.state.failed);
            if failure != "cancel" {
                control.borrow_mut().fail = failure;
                let progress = executor
                    .resume_forward(
                        &mut context,
                        &batch,
                        &mut states,
                        &mut kv,
                        &mut continuation,
                    )
                    .unwrap();
                if failure == "success" {
                    assert!(matches!(
                        progress,
                        DecoderForwardResumeProgress::Complete { .. }
                    ));
                } else {
                    assert!(matches!(
                        progress,
                        DecoderForwardResumeProgress::FailedActive(_)
                    ));
                    assert!(continuation.state.failed);
                    control.borrow_mut().fail = "";
                    assert!(matches!(
                        executor
                            .resume_forward(
                                &mut context,
                                &batch,
                                &mut states,
                                &mut kv,
                                &mut continuation
                            )
                            .unwrap(),
                        DecoderForwardResumeProgress::FailedActive(_)
                    ));
                }
            }
            if failure != "success" {
                assert!(
                    executor
                        .cancel_forward(
                            &mut context,
                            &batch,
                            &mut states,
                            &mut kv,
                            &mut continuation
                        )
                        .is_err()
                );
                assert!(continuation.state.failed);
                assert_eq!(control.borrow().pending_drops.get(), 0);
                control.borrow_mut().completion = Completion::Pending;
                assert_eq!(
                    executor
                        .cancel_forward(
                            &mut context,
                            &batch,
                            &mut states,
                            &mut kv,
                            &mut continuation
                        )
                        .unwrap(),
                    DecoderCancelProgress::Waiting
                );
                assert_eq!(control.borrow().pending_drops.get(), 0);
                control.borrow_mut().completion = Completion::Complete;
            }
            assert_eq!(
                executor
                    .cancel_forward(
                        &mut context,
                        &batch,
                        &mut states,
                        &mut kv,
                        &mut continuation
                    )
                    .unwrap(),
                DecoderCancelProgress::Complete
            );
            assert_eq!(control.borrow().pending_drops.get(), 1);
            assert_eq!(
                control.borrow().cancel_calls,
                usize::from(failure != "success")
            );
            assert_eq!(
                control.borrow().abort_calls,
                usize::from(failure != "success")
            );
            drop(continuation);
            drop(kv);
            backend.leave(&mut transaction).unwrap();
            registry.restore_kv_transaction(tx(), transaction).unwrap();
            assert_eq!(
                registry.abort(tx(), &mut backend, &mut executor).unwrap(),
                KvEndProgress::Complete
            );
        }
    }
}
