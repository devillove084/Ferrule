//! Fake transport exercises real generic forward, cancellation and root GPU custody.
use super::*;
use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_common::{Error, ParallelRankId, Result};
use ferrule_model::transformer::expert_parallel::{RoutedSwiGluExecutor, RoutedSwiGluRequest};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

#[derive(Clone, Copy)]
enum Reply {
    Ready,
    Failure,
    Waiting,
}
struct Control {
    reply: Cell<Reply>,
    drain: Cell<HybridCudaExpertProgress>,
    drain_error: Cell<bool>,
    drain_calls: Cell<usize>,
    pending_at: Cell<usize>,
    shutdown: Cell<HybridCudaExpertProgress>,
    shutdown_calls: Cell<usize>,
    errors: RefCell<Vec<ExecutionTransactionId>>,
    calls: RefCell<Vec<(ExecutionTransactionId, usize, Vec<u64>)>>,
    dropped: Cell<bool>,
    cancelled: Cell<bool>,
}
impl Control {
    fn new(reply: Reply) -> Rc<Self> {
        Rc::new(Self {
            reply: Cell::new(reply),
            drain: Cell::new(HybridCudaExpertProgress::Complete),
            drain_error: Cell::new(false),
            drain_calls: Cell::new(0),
            pending_at: Cell::new(usize::MAX),
            shutdown: Cell::new(HybridCudaExpertProgress::Complete),
            shutdown_calls: Cell::new(0),
            errors: RefCell::new(Vec::new()),
            calls: RefCell::new(Vec::new()),
            dropped: Cell::new(false),
            cancelled: Cell::new(false),
        })
    }
}
fn failure(message: &str) -> Error {
    Error::Execution {
        message: message.into(),
    }
}
struct Fake {
    control: Rc<Control>,
    ops: Rc<CudaOperators>,
}
impl Drop for Fake {
    fn drop(&mut self) {
        self.control.dropped.set(true);
    }
}
impl RoutedSwiGluExecutor for Fake {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        check(request.context.transaction)?;
        assert!(
            check(ExecutionTransactionId::new(
                request.context.transaction.get() + 1000
            )?)
            .is_err()
        );
        assert_eq!(request.context.source_rank, ParallelRankId::new(7));
        request.input.cuda()?.validate_owner(&self.ops)?;
        self.control.calls.borrow_mut().push((
            request.context.transaction,
            request.context.layer,
            request.sequences.to_vec(),
        ));
        match self.control.reply.get() {
            Reply::Failure => Err(failure("fake external failure")),
            Reply::Waiting => Ok(OperatorProgress::Waiting(OperatorWaiting::Experts {
                layer: request.context.layer,
                experts: vec![0],
            })),
            Reply::Ready => Ok(OperatorProgress::Ready(Rows::Cuda(CudaRows::f32(
                request.input.shape(),
                request.arena,
                self.ops.zero_f32_buffer(request.input.shape().elements())?,
            )?))),
        }
    }
}
impl HybridCudaRoutedExecutor for Fake {
    fn drain(&mut self) -> Result<HybridCudaExpertProgress> {
        let calls = self.control.drain_calls.get() + 1;
        self.control.drain_calls.set(calls);
        if self.control.drain_error.get() {
            return Err(failure("fake drain failure"));
        }
        if calls == self.control.pending_at.get() {
            self.control.drain.set(HybridCudaExpertProgress::Pending);
        }
        Ok(self.control.drain.get())
    }
    fn shutdown(&mut self) -> Result<HybridCudaExpertProgress> {
        self.control
            .shutdown_calls
            .set(self.control.shutdown_calls.get() + 1);
        Ok(self.control.shutdown.get())
    }
    fn on_error(&mut self, transaction: ExecutionTransactionId) {
        self.control.errors.borrow_mut().push(transaction);
    }
}
fn external_options(r: &BoundDecoderResources) -> GenericDecoderOptions {
    numeric(
        GenericDecoderOptions::standard_cpu(
            r.spec(),
            ModelFamily::Unknown("external-test".into()),
            WeightSource::Safetensors,
            2,
            32,
            8,
            2,
            1 << 20,
            ExecutionPrecisionPolicy::f32(),
        )
        .unwrap(),
        1,
        16384,
        4096,
    )
}
fn runner(f: &fixture::Fixture, control: &Rc<Control>) -> (GpuRunner, HybridCudaDevice) {
    let r = resources(f);
    let opts = external_options(&r);
    let e = HybridCudaMemoryEstimate::for_resources_with_routed_experts(&r, &opts, 32).unwrap();
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 32,
            state_bytes: e.per_sequence_state_bytes * 8,
            weight_bytes: e.weight_bytes_upper_bound,
            workspace_bytes: e.workspace_bytes_upper_bound,
        },
    )
    .unwrap();
    let authority = control.clone();
    let attachment = HybridCudaRoutedExperts::new(
        ParallelRankId::new(7),
        Box::new(Fake {
            control: control.clone(),
            ops: device.operators().clone(),
        }),
    )
    .with_active_check(move |_| {
        if authority.cancelled.get() {
            Err(failure("external cancelled"))
        } else {
            Ok(())
        }
    });
    (
        GpuRunner::hybrid_cuda_with_routed_experts(
            r,
            tokenizer(f),
            opts,
            device.clone(),
            attachment,
        )
        .unwrap(),
        device,
    )
}
fn prepare(
    runner: &mut GpuRunner,
    states: &mut [CudaHybridSequenceState],
    id: u64,
) -> ExecutionBatch {
    // Packed sequence order intentionally differs from state-slot order.
    let mut tokens = Vec::new();
    let mut positions = Vec::new();
    let mut writes = Vec::new();
    let mut sequences = Vec::new();
    let mut blocks = Vec::new();
    let mut reservations = Vec::new();
    for (packed, state) in (0..states.len()).rev().enumerate() {
        let count = if packed == 0 { 2 } else { 1 };
        let start = tokens.len();
        let page = KvPageId(10 + state as u32);
        for row in 0..count {
            tokens.push(row as u32 + 1);
            positions.push(row as u32);
            writes.push(Some(KvWriteSlot::new(page.0 * 2 + row as u32)));
        }
        sequences.push(ExecutionSequence::new(
            StateSlot::new(state as u32),
            ForwardPhase::Prefill,
            start as u32..tokens.len() as u32,
            0,
            count as u32,
            packed as u32..packed as u32 + 1,
        ));
        blocks.push(KvBlockId::new(page.0));
        reservations.push(KvReservationView {
            state_slot: StateSlot::new(state as u32),
            execution_state_slot: StateSlot::new(state as u32),
            positions: 0..count,
            newly_allocated: vec![page],
            generation: states[state].core().generation(),
            execution_generation: states[state].core().generation(),
            cow_replacement: None,
        });
    }
    let logits = vec![LogitsRequest::Full; tokens.len()];
    let batch = ExecutionBatch::new(
        ForwardMode::Prefill,
        tokens,
        positions,
        writes,
        logits,
        sequences,
        blocks,
    );
    runner
        .prepare_multi_session_batch(
            ExecutionTransactionId::new(id).unwrap(),
            states,
            &batch,
            &reservations,
        )
        .unwrap();
    batch
}
fn tx() -> ExecutionTransactionId {
    ExecutionTransactionId::new(41).unwrap()
}
fn assert_pinned(pool: &mut TypedCudaPagedKvPool<CudaHybridSequenceState>) {
    assert_eq!(pool.active_transaction_count(), 1);
    assert!(PhysicalKvPool::release(pool, &[KvPageId(10)]).is_err());
    assert!(PhysicalKvPool::preempt(pool, &[KvPageId(10)]).is_err());
}
fn assert_no_local_experts(runner: &GpuRunner) {
    let stats = runner
        .forward_executor()
        .module()
        .expert_cache_stats()
        .unwrap();
    assert_eq!(stats.uploads, 0);
    assert_eq!(stats.resident_experts, 0);
}

#[test]
fn external_estimate_excludes_payload_cache_but_keeps_numeric_workspace() {
    let f = fixture::Fixture::new();
    let r = resources(&f);
    let opts = external_options(&r);
    let local = HybridCudaMemoryEstimate::for_resources(&r, &opts, 32).unwrap();
    let remote =
        HybridCudaMemoryEstimate::for_resources_with_routed_experts(&r, &opts, 32).unwrap();
    assert_eq!(
        local.weight_bytes_upper_bound - remote.weight_bytes_upper_bound,
        16384 - 4096
    );
    let larger = numeric(options(&r), 64, 128 << 20, 4096);
    let larger =
        HybridCudaMemoryEstimate::for_resources_with_routed_experts(&r, &larger, 32).unwrap();
    assert_eq!(
        remote.weight_bytes_upper_bound,
        larger.weight_bytes_upper_bound
    );
    assert_eq!(
        remote.workspace_bytes_upper_bound,
        larger.workspace_bytes_upper_bound
    );
    let static64 = numeric(options(&r), 64, 128 << 20, 64 << 20);
    let static64 =
        HybridCudaMemoryEstimate::for_resources_with_routed_experts(&r, &static64, 32).unwrap();
    assert_eq!(
        static64.workspace_bytes_upper_bound - remote.workspace_bytes_upper_bound,
        (64 << 20) - 4096
    );
    assert_eq!(
        static64.weight_bytes_upper_bound,
        remote.weight_bytes_upper_bound
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn external_packed_identity_same_forward_and_no_local_payload() {
    let f = fixture::Fixture::new();
    let control = Control::new(Reply::Ready);
    let (mut runner, device) = runner(&f, &control);
    let mut states = vec![
        runner.create_sequence_state().unwrap(),
        runner.create_sequence_state().unwrap(),
    ];
    let ids = [states[0].topology_id().get(), states[1].topology_id().get()];
    let batch = prepare(&mut runner, &mut states, 41);
    assert!(matches!(
        runner
            .execute_multi_session_batch_progress(tx(), &mut states, &batch)
            .unwrap(),
        MultiSessionBatchProgress::Complete(_)
    ));
    assert_eq!(
        *control.calls.borrow(),
        vec![
            (tx(), 0, vec![ids[1], ids[1], ids[0]]),
            (tx(), 1, vec![ids[1], ids[1], ids[0]])
        ]
    );
    assert!(states.iter().all(|s| s.core().position() == 0));
    assert_no_local_experts(&runner);
    assert_eq!(
        runner
            .end_transaction(tx(), &mut states, TransactionEndIntent::Publish)
            .unwrap(),
        TransactionEndProgress::Complete
    );
    assert_eq!(states[0].core().position(), 1);
    assert_eq!(states[1].core().position(), 2);
    for state in states {
        runner.try_release_sequence_state(state).unwrap();
    }
    runner.shutdown().unwrap();
    assert_eq!(control.shutdown_calls.get(), 1);
    drop(runner);
    assert_eq!(control.shutdown_calls.get(), 1);
    assert!(control.dropped.get());
    assert!(!device.needs_quarantine());
}

#[test]
#[ignore = "requires CUDA GPU"]
fn external_failure_and_waiting_retain_module_cancel_and_root_pins() {
    for reply in [Reply::Failure, Reply::Waiting] {
        let f = fixture::Fixture::new();
        let control = Control::new(reply);
        control.drain.set(HybridCudaExpertProgress::Pending);
        let (mut runner, device) = runner(&f, &control);
        let mut states = vec![runner.create_sequence_state().unwrap()];
        let bytes = device.live_state_bytes();
        let batch = prepare(&mut runner, &mut states, 41);
        let mut probe = runner.backend().pool().clone();
        assert!(
            runner
                .execute_multi_session_batch_progress(tx(), &mut states, &batch)
                .is_err()
        );
        assert_eq!(*control.errors.borrow(), vec![tx()]);
        assert_no_local_experts(&runner);
        assert!(
            runner
                .end_transaction(tx(), &mut states, TransactionEndIntent::Publish)
                .is_err()
        );
        for _ in 0..2 {
            assert_eq!(
                runner
                    .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
                    .unwrap(),
                TransactionEndProgress::Pending
            );
            assert_pinned(&mut probe);
            assert!(runner.shutdown().is_err());
        }
        control.drain_error.set(true);
        assert!(
            runner
                .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
                .is_err()
        );
        assert_pinned(&mut probe);
        control.drain_error.set(false);
        control.drain.set(HybridCudaExpertProgress::Complete);
        assert_eq!(
            runner
                .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
                .unwrap(),
            TransactionEndProgress::Complete
        );
        assert_eq!(device.live_state_bytes(), bytes);
        assert_eq!(states[0].core().position(), 0);
        assert_eq!(probe.active_transaction_count(), 0);
        assert_eq!(*control.errors.borrow(), vec![tx()]);
        runner
            .try_release_sequence_state(states.pop().unwrap())
            .unwrap();
        runner.shutdown().unwrap();
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn external_finish_pending_cannot_publish_and_shutdown_retries() {
    let f = fixture::Fixture::new();
    let control = Control::new(Reply::Ready);
    control.pending_at.set(3); // Two layer drains, then module finish.
    let (mut runner, _) = runner(&f, &control);
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let batch = prepare(&mut runner, &mut states, 41);
    let mut probe = runner.backend().pool().clone();
    assert!(
        runner
            .execute_multi_session_batch_progress(tx(), &mut states, &batch)
            .is_err()
    );
    assert_eq!(control.calls.borrow().len(), 2);
    assert_eq!(
        runner
            .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
            .unwrap(),
        TransactionEndProgress::Pending
    );
    assert_pinned(&mut probe);
    control.drain.set(HybridCudaExpertProgress::Complete);
    assert_eq!(
        runner
            .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
            .unwrap(),
        TransactionEndProgress::Complete
    );
    runner
        .try_release_sequence_state(states.pop().unwrap())
        .unwrap();
    control.shutdown.set(HybridCudaExpertProgress::Pending);
    assert!(runner.shutdown().is_err());
    assert!(!control.dropped.get());
    control.shutdown.set(HybridCudaExpertProgress::Complete);
    runner.shutdown().unwrap();
    assert_eq!(control.shutdown_calls.get(), 2);
}

#[test]
#[ignore = "requires CUDA GPU; intentionally quarantines retained resources"]
fn external_unknown_is_permanent_even_after_root_fence_and_drop() {
    let f = fixture::Fixture::new();
    let control = Control::new(Reply::Failure);
    let (mut runner, device) = runner(&f, &control);
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let batch = prepare(&mut runner, &mut states, 41);
    let mut probe = runner.backend().pool().clone();
    assert!(
        runner
            .execute_multi_session_batch_progress(tx(), &mut states, &batch)
            .is_err()
    );
    control.drain.set(HybridCudaExpertProgress::Unknown);
    assert!(
        runner
            .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
            .is_err()
    );
    let bytes = device.live_state_bytes();
    let drains = control.drain_calls.get();
    assert!(device.needs_quarantine());
    device.operators().sync_stream().unwrap();
    control.drain.set(HybridCudaExpertProgress::Complete);
    assert!(
        runner
            .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
            .is_err()
    );
    assert_eq!(control.drain_calls.get(), drains);
    assert_pinned(&mut probe);
    assert!(runner.shutdown().is_err());
    drop(states);
    drop(runner);
    assert!(!control.dropped.get());
    assert_eq!(device.live_state_bytes(), bytes);
    assert_pinned(&mut probe);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn external_cancel_hook_rejects_dispatch_without_local_fallback() {
    let f = fixture::Fixture::new();
    let control = Control::new(Reply::Ready);
    let (mut runner, _) = runner(&f, &control);
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let batch = prepare(&mut runner, &mut states, 41);
    control.cancelled.set(true);
    assert!(
        runner
            .execute_multi_session_batch_progress(tx(), &mut states, &batch)
            .is_err()
    );
    assert!(control.calls.borrow().is_empty());
    assert_eq!(*control.errors.borrow(), vec![tx()]);
    assert_no_local_experts(&runner);
    assert_eq!(
        runner
            .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
            .unwrap(),
        TransactionEndProgress::Complete
    );
    runner
        .try_release_sequence_state(states.pop().unwrap())
        .unwrap();
    runner.shutdown().unwrap();
}

#[test]
#[ignore = "requires CUDA GPU"]
fn external_source_preflight_precedes_dispatch_and_state_mutation() {
    let f = fixture::Fixture::new();
    let control = Control::new(Reply::Ready);
    let (mut runner, device) = runner(&f, &control);
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let batch = prepare(&mut runner, &mut states, 41);
    let before = states[0].kv_state().layers()[0]
        .as_ref()
        .unwrap()
        .diagnostic_snapshot()
        .unwrap();
    let source = f.dir.join("weights.bin");
    let bytes = std::fs::read(&source).unwrap();
    std::fs::remove_file(&source).unwrap();
    std::fs::write(source, bytes).unwrap(); // Same bytes/length, another image identity.
    assert!(
        runner
            .execute_multi_session_batch_progress(tx(), &mut states, &batch)
            .is_err()
    );
    assert!(control.calls.borrow().is_empty());
    assert!(control.errors.borrow().is_empty()); // FailedQuiescent, never active.
    assert_eq!(
        states[0].kv_state().layers()[0]
            .as_ref()
            .unwrap()
            .diagnostic_snapshot()
            .unwrap(),
        before
    );
    assert_eq!(
        runner
            .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
            .unwrap(),
        TransactionEndProgress::Complete
    );
    runner
        .try_release_sequence_state(states.pop().unwrap())
        .unwrap();
    runner.shutdown().unwrap();
    assert!(!device.needs_quarantine());
}

#[test]
#[ignore = "requires CUDA GPU; intentionally quarantines rejected attachment"]
fn external_rejected_preparation_unknown_quarantines_root_owner() {
    let f = fixture::Fixture::new();
    let r = resources(&f);
    let opts = external_options(&r);
    let e = HybridCudaMemoryEstimate::for_resources_with_routed_experts(&r, &opts, 32).unwrap();
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 32,
            state_bytes: e.per_sequence_state_bytes * 8,
            weight_bytes: 1,
            workspace_bytes: e.workspace_bytes_upper_bound,
        },
    )
    .unwrap();
    let control = Control::new(Reply::Ready);
    control.drain.set(HybridCudaExpertProgress::Unknown);
    let attachment = HybridCudaRoutedExperts::new(
        ParallelRankId::new(7),
        Box::new(Fake {
            control: control.clone(),
            ops: device.operators().clone(),
        }),
    );
    assert!(
        GpuRunner::hybrid_cuda_with_routed_experts(
            r,
            tokenizer(&f),
            opts,
            device.clone(),
            attachment
        )
        .is_err()
    );
    assert!(control.calls.borrow().is_empty());
    assert!(!control.dropped.get());
    assert!(device.needs_quarantine());
}

#[test]
#[ignore = "requires CUDA GPU; intentionally quarantines shutdown owner"]
fn external_physical_shutdown_unknown_is_not_forgiven_by_local_fence() {
    let f = fixture::Fixture::new();
    let control = Control::new(Reply::Ready);
    let (mut runner, device) = runner(&f, &control);
    control.shutdown.set(HybridCudaExpertProgress::Unknown);
    assert!(runner.shutdown().is_err());
    assert!(device.needs_quarantine());
    let bytes = device.live_state_bytes();
    device.operators().sync_stream().unwrap();
    control.shutdown.set(HybridCudaExpertProgress::Complete);
    assert!(runner.shutdown().is_err());
    assert_eq!(control.shutdown_calls.get(), 1);
    drop(runner);
    assert_eq!(device.live_state_bytes(), bytes);
    assert!(!control.dropped.get());
}
