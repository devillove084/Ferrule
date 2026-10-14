//! Real GPU math through the shared segment and a separately prepared CUDA
//! expert operator. Host collective transport only; no reference model math.
use super::*;
use ferrule_common::{ParallelGroupId, Result};
use ferrule_model::transformer::expert_parallel::{RoutedSwiGluExecutor, RoutedSwiGluRequest};
use ferrule_runtime::parallel::{
    collective::HostCollectiveLimits, tensor::decoder_collective::DecoderTensorCollective,
};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    rc::Rc,
    sync::{Barrier, atomic::AtomicBool},
    time::Duration,
};

struct Provider(BTreeMap<(usize, usize), Arc<PreparedSwiGlu>>);
impl ExpertProvider for Provider {
    fn expert(&mut self, layer: usize, expert: usize) -> Result<ExpertAvailability> {
        Ok(ExpertAvailability::Ready(Arc::clone(
            self.0.get(&(layer, expert)).unwrap(),
        )))
    }
}
struct GpuExperts {
    operators: CudaStandardDecoderOperators,
    provider: Provider,
    calls: Rc<RefCell<Vec<usize>>>,
}
impl GpuExperts {
    fn prepare(
        r: &BoundDecoderResources,
        device: &HybridCudaDevice,
        precision: NumericFp8Precision,
        calls: Rc<RefCell<Vec<usize>>>,
    ) -> Self {
        let parameters = r
            .state_dict()
            .parameters()
            .iter()
            .filter(|p| matches!(p.residency(), ParameterResidency::Expert { .. }))
            .cloned()
            .collect::<Vec<_>>();
        let mut operators = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
            device.operators().clone(),
            &parameters,
            ExpertCachePolicy::KeepAll,
            1 << 20,
            precision,
        )
        .unwrap();
        let materializer = StateDictMaterializer::new(1 << 20).unwrap();
        let mut experts = BTreeMap::new();
        for layer in 1..4 {
            let FeedForward::Moe(m) = r.spec().layers()[layer].feed_forward() else {
                unreachable!()
            };
            for expert in 0..2 {
                let projection = |role: TensorRole, shape: [usize; 2]| {
                    let b = r
                        .experts()
                        .require_shape(layer, expert, role.clone(), &shape)
                        .unwrap();
                    materializer.prepared_linear(b, role).unwrap()
                };
                let value = Arc::new(
                    PreparedSwiGlu::new(
                        projection(
                            TensorRole::RoutedExpertGate,
                            m.expert().gate().weight_shape(),
                        ),
                        projection(TensorRole::RoutedExpertUp, m.expert().up().weight_shape()),
                        projection(
                            TensorRole::RoutedExpertDown,
                            m.expert().down().weight_shape(),
                        ),
                        None,
                    )
                    .unwrap(),
                );
                operators.prepare_expert(&value).unwrap();
                experts.insert((layer, expert), value);
            }
        }
        Self {
            operators,
            provider: Provider(experts),
            calls,
        }
    }
}
impl RoutedSwiGluExecutor for GpuExperts {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        active(request.context.transaction)?;
        self.calls.borrow_mut().push(request.context.layer);
        self.operators.routed_swiglu(
            request.context.layer,
            request.input,
            request.routes,
            &mut self.provider,
            request.arena,
        )
    }
}
impl HybridCudaRoutedExecutor for GpuExperts {
    fn drain(&mut self) -> Result<WorkProgress> {
        self.operators.quiesce()?;
        Ok(WorkProgress::Complete)
    }
    fn shutdown(&mut self) -> Result<WorkProgress> {
        self.drain()
    }
}

fn run(degree: usize, precision: NumericFp8Precision) -> Vec<f32> {
    let f = Arc::new(NumericFixture::new());
    let p = parallel(&f.resources, degree);
    let endpoints = DecoderTensorCollective::new_group_from_members(
        p.tensor_members(),
        ParallelTopologyId::new(907),
        ParallelGroupId::new(8),
        HostCollectiveLimits {
            max_ranks: degree,
            max_elements_per_rank: 8192,
            max_host_bytes: 1 << 20,
        },
        Duration::from_secs(30),
    )
    .unwrap();
    let tx = ExecutionTransactionId::new(71).unwrap();
    endpoints[0]
        .control()
        .begin(tx, Arc::new(AtomicBool::new(false)))
        .unwrap();
    let done = Arc::new(Barrier::new(degree));
    let threads = endpoints
        .into_iter()
        .enumerate()
        .map(|(lane, endpoint)| {
            let f = f.clone();
            let p = p.clone();
            let done = done.clone();
            std::thread::spawn(move || {
                let r = &f.resources;
                let o = options(r)
                    .with_hybrid_cuda_numeric_fp8_precision(
                        ExpertCacheLimits {
                            max_experts: 8,
                            max_bytes: 2 << 20,
                        },
                        1 << 20,
                        precision,
                    )
                    .unwrap();
                let layout = Arc::new(p.rank_layout(lane).unwrap());
                let m = CudaHybridRankMemory::for_rank(r, &segment(), &p, &layout, &o, 16).unwrap();
                let device = HybridCudaDevice::new_on_device(
                    0,
                    HybridCudaMemoryBudget {
                        kv_pages: 16,
                        state_bytes: 3 * m.per_sequence_state_grants,
                        weight_bytes: m.weight_grants,
                        workspace_bytes: m.numeric_workspace_grants + m.operation_workspace_grants,
                    },
                )
                .unwrap();
                let calls = Rc::new(RefCell::new(Vec::new()));
                let source = p.expert_attachment().unwrap().source_rank;
                let experts = (layout.owner() == source).then(|| {
                    HybridCudaRoutedExperts::new(
                        source,
                        Box::new(GpuExperts::prepare(r, &device, precision, calls.clone())),
                    )
                });
                let mut owner = CudaHybridRankSegment::prepare_rank(
                    r,
                    segment(),
                    p,
                    layout,
                    &o,
                    device.clone(),
                    Box::new(endpoint),
                    experts,
                )
                .unwrap();
                let mut life = owner.new_lifecycle().unwrap();
                let original = life.create().unwrap();
                let mut states = vec![
                    life.checkout(DecoderSequenceCheckout::new(tx, 0, 0), &original)
                        .unwrap(),
                ];
                assert_eq!(life.live_state_grants(), 2 * m.per_sequence_state_grants);
                let mut second = owner.new_lifecycle().unwrap();
                let fork = second.logical_fork(&original, 0).unwrap();
                assert!(second.create().is_err());
                second.try_release(fork).unwrap();
                let pool = TypedCudaPagedKvPool::<CudaHybridRankState>::from_strategy(
                    device.operators().clone(),
                    &owner.layout().kv_planes(2, 32).unwrap(),
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
                    &o.capabilities(),
                    2,
                    &backend,
                )
                .unwrap();
                let seq = &batch.sequences()[0];
                let custody = [DecoderKvSequenceCustody {
                    source_index: 0,
                    topology_id: seq.topology_id(),
                    page_state_slot: seq.page_state_slot(),
                    page_generation: seq.page_generation(),
                    execution_generation: seq.execution_generation(),
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
                let output = owner
                    .execute(
                        &batch,
                        SegmentInput::Tokens,
                        &mut states,
                        &mut kv,
                        &[900],
                        &mut |_| Ok(()),
                    )
                    .unwrap();
                owner.validate_frontier(&batch, &states).unwrap();
                assert_eq!(
                    *calls.borrow(),
                    if lane + 1 == degree {
                        vec![1, 2, 3]
                    } else {
                        vec![]
                    }
                );
                assert!(!owner.needs_quarantine());
                let SegmentOutput::Logits(logits) = output else {
                    panic!("missing head")
                };
                let values = logits.values().to_vec();
                assert!(values.iter().all(|x| x.is_finite()));
                assert!(owner.poll_quiescence().unwrap());
                drop(kv);
                backend.leave(&mut handle).unwrap();
                assert_eq!(
                    backend.rollback(&mut Some(handle)).unwrap(),
                    KvEndProgress::Complete
                );
                life.try_release(states.pop().unwrap()).unwrap();
                life.try_release(original).unwrap();
                assert_eq!(life.live_state_grants(), 0);
                done.wait();
                owner.shutdown().unwrap();
                values
            })
        })
        .collect::<Vec<_>>();
    let outputs = threads
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect::<Vec<_>>();
    for output in &outputs {
        assert_eq!(output, &outputs[0]);
    }
    outputs[0].clone()
}
#[test]
#[ignore = "requires real CUDA GPU; TP1/2/4 shared forward and single-source CUDA expert dispatch"]
fn tiny_numeric_hybrid_tp1_tp2_tp4_f32() {
    let expected = run(1, NumericFp8Precision::F32Tf32x3);
    for degree in [2, 4] {
        let actual = run(degree, NumericFp8Precision::F32Tf32x3);
        for (a, b) in actual.iter().zip(&expected) {
            assert!(
                (a - b).abs() <= 2e-4 + 2e-4 * b.abs(),
                "TP{degree}: {a} != {b}"
            );
        }
    }
}
#[test]
#[ignore = "requires real CUDA GPU; explicit W8A16 opt-in, no fallback"]
fn tiny_numeric_hybrid_tp1_tp2_tp4_w8a16() {
    for degree in [1, 2, 4] {
        run(degree, NumericFp8Precision::W8A16Bf16F32Accumulate);
    }
}
