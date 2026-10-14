//! CUDA routed-segment contracts that do not initialize a CUDA device.
use super::*;

fn routed_options(
    resources: &BoundDecoderResources,
    precision: NumericFp8Precision,
) -> GenericDecoderOptions {
    options(resources)
        .with_hybrid_cuda_numeric_fp8_precision(
            ExpertCacheLimits {
                max_experts: 8,
                max_bytes: 1 << 20,
            },
            4096,
            precision,
        )
        .unwrap()
}

#[test]
fn routed_segment_gate_accepts_partial_pp_ranges_for_explicit_profiles() {
    let fixture = fixture::Fixture::with_precision(NumericFp8Precision::F32Tf32x3);
    let resources = resources(&fixture);
    for precision in [
        NumericFp8Precision::F32Tf32x3,
        NumericFp8Precision::W8A16Bf16F32Accumulate,
    ] {
        let options = routed_options(&resources, precision);
        for range in [0..2, 1..2] {
            let plan = LayerSegmentPlan::new(2, range, false, false).unwrap();
            CudaHybridDecoderSegment::validate_routed_preparation(&resources, &plan, &options)
                .unwrap();
        }
    }
}

#[test]
fn routed_segment_gate_rejects_implicit_or_diagnostic_numeric_modes() {
    let fixture = fixture::Fixture::with_precision(NumericFp8Precision::F32Tf32x3);
    let resources = resources(&fixture);
    let plan = LayerSegmentPlan::new(2, 0..1, true, false).unwrap();

    assert!(
        CudaHybridDecoderSegment::validate_routed_preparation(
            &resources,
            &plan,
            &options(&resources)
        )
        .is_err()
    );
    assert!(
        CudaHybridDecoderSegment::validate_routed_preparation(
            &resources,
            &plan,
            &options(&resources)
                .with_hybrid_cuda_numeric_fp8(
                    ExpertCacheLimits {
                        max_experts: 8,
                        max_bytes: 1 << 20,
                    },
                    4096,
                )
                .unwrap()
        )
        .is_err()
    );
}

#[test]
fn routed_segment_gate_rejects_active_prefix_options_but_allows_global_stage_ranges() {
    let fixture = fixture::Fixture::with_precision(NumericFp8Precision::F32Tf32x3);
    let resources = resources(&fixture);
    let plan = LayerSegmentPlan::new(2, 0..1, true, false).unwrap();
    let options = routed_options(&resources, NumericFp8Precision::F32Tf32x3)
        .with_active_layers(1)
        .unwrap();

    assert!(
        CudaHybridDecoderSegment::validate_routed_preparation(&resources, &plan, &options).is_err()
    );
}

use ferrule_common::{ParallelRankId, Result};
use ferrule_model::transformer::expert_parallel::{RoutedSwiGluExecutor, RoutedSwiGluRequest};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

type RoutedCalls = Rc<RefCell<Vec<(usize, Vec<u64>)>>>;

// Fault transport only. These tests certify CUDA root/custody integration, NOT
// distributed EP arithmetic. Real expert children are runtime's integration gate.
struct FaultTransport {
    ops: Rc<ferrule_backend::cuda::operators::linear::CudaOperators>,
    progress: Rc<Cell<WorkProgress>>,
    calls: RoutedCalls,
    errors: Rc<Cell<usize>>,
}
impl RoutedSwiGluExecutor for FaultTransport {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        check_active(request.context.transaction)?;
        assert_eq!(request.context.source_rank, ParallelRankId::new(7));
        request.input.cuda()?.validate_owner(&self.ops)?;
        self.calls
            .borrow_mut()
            .push((request.context.layer, request.sequences.to_vec()));
        Ok(OperatorProgress::Ready(Rows::Cuda(CudaRows::f32(
            request.input.shape(),
            request.arena,
            self.ops.zero_f32_buffer(request.input.shape().elements())?,
        )?)))
    }
}
impl HybridCudaRoutedExecutor for FaultTransport {
    fn drain(&mut self) -> Result<WorkProgress> {
        Ok(self.progress.get())
    }
    fn shutdown(&mut self) -> Result<WorkProgress> {
        Ok(self.progress.get())
    }
    fn on_error(&mut self, _: ExecutionTransactionId) {
        self.errors.set(self.errors.get() + 1);
    }
}

#[test]
#[ignore = "requires CUDA GPU; fault transport, not numerical distributed EP acceptance"]
fn routed_segment_cuda_prepares_profiles_without_local_experts_and_polls_shutdown() {
    for precision in [
        NumericFp8Precision::F32Tf32x3,
        NumericFp8Precision::W8A16Bf16F32Accumulate,
    ] {
        let fixture = fixture::Fixture::with_precision(precision);
        let resources = resources(&fixture);
        let options = routed_options(&resources, precision);
        // Deliberately generous fixture budget; NOT stage production admission.
        let device = HybridCudaDevice::new_on_device(
            0,
            HybridCudaMemoryBudget {
                kv_pages: 16,
                state_bytes: 1 << 20,
                weight_bytes: 1 << 20,
                workspace_bytes: 1 << 20,
            },
        )
        .unwrap();
        let progress = Rc::new(Cell::new(WorkProgress::Complete));
        let attachment = HybridCudaRoutedExperts::new(
            ParallelRankId::new(7),
            Box::new(FaultTransport {
                ops: device.operators().clone(),
                progress: progress.clone(),
                calls: Rc::default(),
                errors: Rc::default(),
            }),
        );
        let mut segment = CudaHybridDecoderSegment::prepare_with_routed_experts(
            &resources,
            LayerSegmentPlan::new(2, 0..2, true, true).unwrap(),
            &options,
            device.clone(),
            attachment,
        )
        .unwrap();
        assert_eq!(segment.numeric_fp8_precision(), Some(precision));
        assert_eq!(segment.routed_source_rank(), Some(ParallelRankId::new(7)));
        assert_eq!(segment.operators().expert_cache_stats().unwrap().uploads, 0);
        let mut lifecycle = segment.new_lifecycle().unwrap();
        let state = lifecycle.create().unwrap();
        lifecycle.try_release(state).unwrap();
        progress.set(WorkProgress::Pending);
        assert!(!segment.poll_shutdown().unwrap());
        assert!(segment.retains_expert_custody());
        assert!(segment.new_lifecycle().is_err());
        progress.set(WorkProgress::Complete);
        assert!(segment.poll_shutdown().unwrap());
        assert!(!segment.retains_expert_custody());
        assert!(!device.needs_quarantine());
    }
}

#[test]
#[ignore = "requires CUDA GPU; exercises real root layer loop with fault-only external transport"]
fn routed_segment_cuda_execution_retains_pending_kv_then_allows_rollback() {
    let fixture = fixture::Fixture::with_precision(NumericFp8Precision::F32Tf32x3);
    let resources = resources(&fixture);
    let options = routed_options(&resources, NumericFp8Precision::F32Tf32x3);
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 16,
            state_bytes: 1 << 20,
            weight_bytes: 1 << 20,
            workspace_bytes: 1 << 20,
        },
    )
    .unwrap();
    let progress = Rc::new(Cell::new(WorkProgress::Complete));
    let calls = Rc::new(RefCell::new(Vec::new()));
    let errors = Rc::new(Cell::new(0));
    let attachment = HybridCudaRoutedExperts::new(
        ParallelRankId::new(7),
        Box::new(FaultTransport {
            ops: device.operators().clone(),
            progress: progress.clone(),
            calls: calls.clone(),
            errors: errors.clone(),
        }),
    );
    let mut segment = CudaHybridDecoderSegment::prepare_with_routed_experts(
        &resources,
        LayerSegmentPlan::new(2, 0..2, true, true).unwrap(),
        &options,
        device.clone(),
        attachment,
    )
    .unwrap();
    let mut lifecycle = segment.new_lifecycle().unwrap();
    let original = lifecycle.create().unwrap();
    let tx = ExecutionTransactionId::new(71).unwrap();
    let mut states = vec![
        lifecycle
            .checkout(DecoderSequenceCheckout::new(tx, 0, 0), &original)
            .unwrap(),
    ];
    let planes = segment.schema().kv_planes(2, 32).unwrap();
    let pool = TypedCudaPagedKvPool::<CudaHybridSequenceState>::from_strategy(
        device.operators().clone(),
        &planes,
        16,
    )
    .unwrap();
    let mut backend = PagedKvBackend::new(pool);
    let execution = ExecutionBatch::new(
        ForwardMode::Prefill,
        vec![1, 2],
        vec![0, 1],
        vec![Some(KvWriteSlot::new(2)), Some(KvWriteSlot::new(3))],
        vec![LogitsRequest::Full; 2],
        vec![ExecutionSequence::new(
            StateSlot::new(0),
            ForwardPhase::Prefill,
            0..2,
            0,
            2,
            0..1,
        )],
        vec![KvBlockId::new(1)],
    );
    let reservations = [KvReservationView {
        state_slot: StateSlot::new(0),
        execution_state_slot: StateSlot::new(0),
        positions: 0..2,
        newly_allocated: vec![KvPageId(1)],
        generation: original.core().generation(),
        execution_generation: original.core().generation(),
        cow_replacement: None,
    }];
    let batch = PackedDecoderBatch::lower(
        &execution,
        &reservations,
        &states,
        &options.capabilities(),
        2,
        &backend,
    )
    .unwrap();
    let sequence = &batch.sequences()[0];
    let custody = [DecoderKvSequenceCustody {
        source_index: 0,
        topology_id: sequence.topology_id(),
        page_state_slot: sequence.page_state_slot(),
        page_generation: sequence.page_generation(),
        execution_generation: sequence.execution_generation(),
        context_len: 0,
        query_len: 2,
    }];
    let snapshots = [DecoderKvPageSnapshot {
        page: KvPageId(1),
        status: DecoderKvPageView::page_status(&backend, KvPageId(1)),
    }];
    let mut handle = backend
        .prepare(DecoderKvPrepare {
            transaction: tx,
            sequences: &custody,
            new_pages: batch.new_pages(),
            writable_pages: batch.writable_pages(),
            cow_replacements: batch.cow_replacements(),
            protected_pages: batch.protected_pages(),
            capacity: backend.capacity(),
            page_statuses: &snapshots,
        })
        .unwrap();
    backend.enter(&mut handle, &batch, &mut states).unwrap();
    let mut kv = backend.active_view(&mut handle).unwrap();
    // Rejected bindings do not consume the transaction or mutate recurrent state.
    for (transaction, source, sessions) in [
        (
            ExecutionTransactionId::new(72).unwrap(),
            ParallelRankId::new(7),
            vec![900],
        ),
        (tx, ParallelRankId::new(8), vec![900]),
        (tx, ParallelRankId::new(7), vec![]),
    ] {
        assert!(
            segment
                .execute_cuda_bound_with_routed_experts(
                    &batch,
                    DeviceSegmentInput::Tokens,
                    &mut states,
                    &mut kv,
                    (transaction, source, &sessions),
                    &mut |_| Ok(())
                )
                .is_err()
        );
        assert!(calls.borrow().is_empty());
    }
    progress.set(WorkProgress::Pending);
    let error = segment
        .execute_cuda_bound_with_routed_experts(
            &batch,
            DeviceSegmentInput::Tokens,
            &mut states,
            &mut kv,
            (tx, ParallelRankId::new(7), &[900]),
            &mut |_| Ok(()),
        )
        .unwrap_err();
    assert!(error.to_string().contains("pending"));
    assert_eq!(
        *calls.borrow(),
        vec![(0, vec![900, 900]), (1, vec![900, 900])]
    );
    assert_eq!(errors.get(), 1);
    assert_eq!(segment.operators().expert_cache_stats().unwrap().uploads, 0);
    assert!(segment.retains_expert_custody());
    assert!(!segment.poll_quiescence().unwrap());
    assert_eq!(errors.get(), 1);
    assert_eq!(backend.capacity().active_transactions, 1);
    progress.set(WorkProgress::Complete);
    assert!(segment.poll_quiescence().unwrap());
    assert!(
        segment
            .execute_cuda_bound_with_routed_experts(
                &batch,
                DeviceSegmentInput::Tokens,
                &mut states,
                &mut kv,
                (tx, ParallelRankId::new(7), &[900]),
                &mut |_| Ok(())
            )
            .is_err()
    );
    drop(kv);
    backend.leave(&mut handle).unwrap();
    assert_eq!(
        backend.rollback(&mut Some(handle)).unwrap(),
        KvEndProgress::Complete
    );
    lifecycle.try_release(states.pop().unwrap()).unwrap();
    lifecycle.try_release(original).unwrap();
    segment.shutdown().unwrap();
    assert!(!device.needs_quarantine());
}

#[test]
fn routed_segment_gate_rejects_foreign_plan_and_stale_source_identity() {
    let fixture = fixture::Fixture::with_precision(NumericFp8Precision::F32Tf32x3);
    let resources = resources(&fixture);
    let options = routed_options(&resources, NumericFp8Precision::F32Tf32x3);
    let foreign = LayerSegmentPlan::new(3, 1..2, false, false).unwrap();
    assert!(
        CudaHybridDecoderSegment::validate_routed_preparation(&resources, &foreign, &options)
            .is_err()
    );
    let plan = LayerSegmentPlan::new(2, 1..2, false, true).unwrap();
    CudaHybridDecoderSegment::validate_routed_preparation(&resources, &plan, &options).unwrap();
    std::fs::remove_file(fixture.dir.join("weights.bin")).unwrap();
    assert!(
        CudaHybridDecoderSegment::validate_routed_preparation(&resources, &plan, &options).is_err()
    );
}

#[test]
fn routed_segment_gate_keeps_recurrent_only_stage_closed() {
    let fixture = fixture::Fixture::with_precision(NumericFp8Precision::F32Tf32x3);
    let resources = resources(&fixture);
    let options = routed_options(&resources, NumericFp8Precision::F32Tf32x3);
    let plan = LayerSegmentPlan::new(2, 0..1, true, false).unwrap();
    let error = CudaHybridDecoderSegment::validate_routed_preparation(&resources, &plan, &options)
        .unwrap_err();
    assert!(error.to_string().contains("recurrent-only"), "{error}");
}

#[test]
#[ignore = "requires CUDA GPU; lost child completion must permanently quarantine root"]
fn routed_segment_cuda_unknown_completion_is_sticky() {
    let fixture = fixture::Fixture::with_precision(NumericFp8Precision::F32Tf32x3);
    let resources = resources(&fixture);
    let options = routed_options(&resources, NumericFp8Precision::F32Tf32x3);
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 16,
            state_bytes: 1 << 20,
            weight_bytes: 1 << 20,
            workspace_bytes: 1 << 20,
        },
    )
    .unwrap();
    let progress = Rc::new(Cell::new(WorkProgress::Unknown));
    let attachment = HybridCudaRoutedExperts::new(
        ParallelRankId::new(7),
        Box::new(FaultTransport {
            ops: device.operators().clone(),
            progress: progress.clone(),
            calls: Rc::default(),
            errors: Rc::default(),
        }),
    );
    let mut segment = CudaHybridDecoderSegment::prepare_with_routed_experts(
        &resources,
        LayerSegmentPlan::new(2, 0..2, true, true).unwrap(),
        &options,
        device.clone(),
        attachment,
    )
    .unwrap();
    assert!(segment.poll_quiescence().is_err());
    assert!(segment.needs_quarantine());
    progress.set(WorkProgress::Complete);
    assert!(segment.poll_quiescence().is_err());
    assert!(segment.poll_shutdown().is_err());
    assert!(segment.retains_expert_custody());
    assert!(segment.new_lifecycle().is_err());
    // Drop intentionally retains the image/child/device because proof was lost.
}
