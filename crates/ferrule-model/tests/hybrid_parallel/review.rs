//! Model contract regressions; physical custody remains in the existing KV backend.
use super::{fixture, support};
use ferrule_common::{ParallelRankId, ParallelTopologyId, execution::*};
use ferrule_model::decoder::*;
use ferrule_model::transformer::{hybrid_parallel::*, parallel::TensorParallelCollective, *};
use std::{
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

fn plan(
    resources: &BoundDecoderResources,
    range: Range<usize>,
    degree: usize,
) -> HybridParallelPlan {
    HybridParallelPlan::new(
        resources.spec(),
        range,
        HybridExpertScope {
            topology: ParallelTopologyId::new(907),
            replica: 0,
            stage: 0,
        },
        [7, 19, 31, 63][..degree]
            .iter()
            .map(|&r| ParallelRankId::new(r))
            .collect(),
        None,
    )
    .unwrap()
}

#[test]
fn public_partition_and_collective_shapes_fail_checked_without_allocating() {
    let partition = |rows, columns| HybridParameterPartition {
        rows,
        columns,
        collective: None,
    };
    for p in [
        partition(vec![], 0..1),
        partition(std::iter::once(2..2).collect(), 0..1),
        partition(std::iter::once(Range { start: 9, end: 2 }).collect(), 0..1),
        partition(std::iter::once(0..1).collect(), Range { start: 9, end: 2 }),
        partition(vec![0..usize::MAX, 0..1], 0..1),
    ] {
        assert!(p.local_shape().is_err());
        assert!(p.bytes_f32().is_err());
    }
    for p in [
        partition(std::iter::once(0..usize::MAX).collect(), 0..2),
        partition(
            std::iter::once(0..(isize::MAX as usize / 4 + 1)).collect(),
            0..1,
        ),
    ] {
        assert!(p.bytes_f32().is_err());
    }
    let p = partition(vec![2..5, 9..11], 4..8);
    assert_eq!(p.local_shape().unwrap(), [5, 4]);
    assert_eq!(p.bytes_f32().unwrap(), 80);
    for shape in [
        HybridCollectiveShape {
            fixed: usize::MAX,
            per_sequence: 1,
            per_row: 0,
        },
        HybridCollectiveShape {
            fixed: 0,
            per_sequence: usize::MAX,
            per_row: 0,
        },
        HybridCollectiveShape {
            fixed: 0,
            per_sequence: 0,
            per_row: usize::MAX,
        },
        HybridCollectiveShape {
            fixed: isize::MAX as usize / 4 + 1,
            per_sequence: 0,
            per_row: 0,
        },
    ] {
        assert!(shape.elements(2, 3).is_err());
    }
}

#[test]
fn ordered_schedule_owns_dynamic_shapes_padding_and_tp1_empty_wire() {
    use HybridCollectiveKind::*;
    use HybridCollectiveSemantic::*;
    let fixture = fixture::Fixture::for_degree(4, false);
    let resources = fixture.resources();
    // Existing mixed dense/MoE fixture includes a scalar shared gate on every MoE layer.
    let base = support::plan(&resources);
    let schedule = base
        .collective_schedule(&LayerSegmentPlan::new(4, 0..4, true, true).unwrap())
        .unwrap();
    let expected = [
        (None, Agreement, AllGather),
        (Some(0), AttentionOutput, ReduceSum),
        (Some(0), DenseMlpDown, ReduceSum),
        (Some(1), AttentionOutput, ReduceSum),
        (Some(1), SharedExpertDown, ReduceSum),
        (Some(1), RoutedExpertOutput, ReduceSum),
        (Some(2), AttentionOutput, ReduceSum),
        (Some(2), SharedExpertDown, ReduceSum),
        (Some(2), RoutedExpertOutput, ReduceSum),
        (Some(3), AttentionOutput, ReduceSum),
        (Some(3), SharedExpertDown, ReduceSum),
        (Some(3), RoutedExpertOutput, ReduceSum),
        (None, OutputHead, AllGather),
        (None, Completion, ReduceSum),
    ];
    assert_eq!(schedule.sites().len(), expected.len());
    for (i, (site, &(layer, semantic, kind))) in schedule.sites().iter().zip(&expected).enumerate()
    {
        assert_eq!(
            (site.ordinal, site.global_layer, site.semantic, site.kind),
            (i as u64, layer, semantic, kind)
        );
    }
    assert_eq!(schedule.agreement_version(), 2);
    assert_eq!(
        schedule.sites()[0].shape,
        HybridCollectiveShape {
            fixed: 124,
            per_sequence: 16,
            per_row: 12
        }
    );
    assert_eq!(schedule.sites()[0].shape.elements(2, 5).unwrap(), 216);
    assert_eq!(schedule.sites()[12].shape.per_row, 6); // ceil(11/2), not local ragged width
    assert_eq!(schedule.sites()[13].shape.elements(2, 5).unwrap(), 1);
    let encoded = schedule.encode();
    assert_eq!(&encoded[..3], &[2, 2, 14]);
    assert_eq!(&encoded[3..10], &[0, u64::MAX, 0, 0, 124, 16, 12]);
    assert_eq!(encoded.len(), 3 + 7 * 14);

    let dense = fixture::Fixture::for_degree(4, true);
    let resources = dense.resources();
    for degree in [1, 2, 4] {
        let p = plan(&resources, 2..4, degree);
        for output in [false, true] {
            let s = p
                .collective_schedule(&LayerSegmentPlan::new(4, 2..4, false, output).unwrap())
                .unwrap();
            assert_eq!(s.wire_sites().is_empty(), degree == 1);
            assert_eq!(
                s.sites()[0].shape,
                HybridCollectiveShape {
                    fixed: 4 * (12 + degree),
                    per_sequence: 16,
                    per_row: 20
                }
            );
            assert_eq!(s.sites()[1].global_layer, Some(2));
            assert_eq!(s.sites()[3].global_layer, Some(3));
            assert_eq!(s.sites().iter().any(|x| x.semantic == OutputHead), output);
            if output {
                assert_eq!(s.sites()[5].shape.per_row, 11usize.div_ceil(degree));
            }
            assert_eq!(s.sites().len(), if output { 7 } else { 6 });
        }
    }
}

struct Stage {
    module: HybridTensorCpuSegment,
    lifecycle: HybridTensorSequenceLifecycle,
    local: support::LocalTransaction<HybridTensorSequenceState>,
}
type Work = support::Work<HybridTensorSequenceState>;
impl Stage {
    fn new(
        resources: &BoundDecoderResources,
        range: Range<usize>,
        lane: usize,
        endpoint: Box<dyn StandardTensorCollective>,
    ) -> Self {
        let p = plan(resources, range.clone(), endpoint.members().len());
        let module = HybridTensorCpuSegment::prepare(
            resources,
            LayerSegmentPlan::new(4, range.clone(), range.start == 0, range.end == 4).unwrap(),
            p,
            lane,
            endpoint,
            32,
            1 << 20,
        )
        .unwrap();
        let mut lifecycle = module.lifecycle();
        let states = vec![lifecycle.create().unwrap(), lifecycle.create().unwrap()];
        let local = support::LocalTransaction::new(
            resources,
            module.layout().kv_planes(2, 32).unwrap(),
            states,
        );
        Self {
            module,
            lifecycle,
            local,
        }
    }
    fn enter(&mut self, id: u64) -> Work {
        // Two logical sequences, independent model/page state slots and generations.
        let source = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1, 2, 3],
            vec![0, 1, 0],
            vec![
                Some(KvWriteSlot::new(0)),
                Some(KvWriteSlot::new(1)),
                Some(KvWriteSlot::new(2)),
            ],
            vec![LogitsRequest::Full; 3],
            vec![
                ExecutionSequence::new(StateSlot::new(0), ForwardPhase::Prefill, 0..2, 0, 2, 0..1),
                ExecutionSequence::new(StateSlot::new(1), ForwardPhase::Prefill, 2..3, 0, 1, 1..2),
            ],
            vec![KvBlockId::new(0), KvBlockId::new(1)],
        );
        let reservations = (0..2)
            .map(|i| KvReservationView {
                state_slot: StateSlot::new(10 + i as u32),
                execution_state_slot: StateSlot::new(i as u32),
                positions: 0..if i == 0 { 2 } else { 1 },
                newly_allocated: vec![KvPageId(i as u32)],
                generation: 0,
                execution_generation: self.local.states[i].core().generation(),
                cow_replacement: None,
            })
            .collect::<Vec<_>>();
        self.local.enter_batch(
            &mut self.lifecycle,
            id,
            &source,
            &reservations,
            vec![KvPageId(0), KvPageId(1)],
        )
    }
    fn execute(
        &mut self,
        work: &mut Work,
        input: SegmentInput,
    ) -> ferrule_common::Result<SegmentOutput> {
        self.module.execute(
            &work.batch,
            input,
            &mut work.states,
            &mut work.view,
            &[73, 91],
            None,
            &mut |_| Ok(()),
        )
    }
    fn rollback(&mut self, work: Work) {
        self.local.rollback_local(&mut self.lifecycle, work);
    }
}

struct ObservedEndpoint {
    inner: Box<dyn StandardTensorCollective>,
    calls: Arc<AtomicUsize>,
    panic_site: Option<u64>,
    abort_panics: bool,
}
impl StandardTensorCollective for ObservedEndpoint {
    fn owner(&self) -> ParallelRankId {
        self.inner.owner()
    }
    fn members(&self) -> &[ParallelRankId] {
        self.inner.members()
    }
    fn exchange(
        &mut self,
        tx: ExecutionTransactionId,
        site: u64,
        kind: TensorParallelCollective,
        values: Vec<f32>,
    ) -> ferrule_common::Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.panic_site == Some(site) {
            panic!("original model execution panic");
        }
        self.inner.exchange(tx, site, kind, values)
    }
    fn abort(&mut self) {
        self.inner.abort();
        if self.abort_panics {
            panic!("abort hook panic");
        }
    }
}
fn observed(
    endpoint: impl StandardTensorCollective + 'static,
    calls: &Arc<AtomicUsize>,
    panic_site: Option<u64>,
    abort_panics: bool,
) -> Box<dyn StandardTensorCollective> {
    Box::new(ObservedEndpoint {
        inner: Box::new(endpoint),
        calls: Arc::clone(calls),
        panic_site,
        abort_panics,
    })
}

#[test]
fn panic_poisons_retained_module_wakes_peer_and_never_rolls_back_custody() {
    let fixture = Arc::new(fixture::Fixture::for_degree(4, true));
    let endpoints = support::endpoints_for(&[support::rank(7), support::rank(19)]);
    let control = endpoints[0].control();
    control
        .begin(support::tx(1), Arc::new(AtomicBool::new(false)))
        .unwrap();
    let threads = endpoints
        .into_iter()
        .enumerate()
        .map(|(lane, endpoint)| {
            let fixture = Arc::clone(&fixture);
            std::thread::spawn(move || {
                let calls = Arc::new(AtomicUsize::new(0));
                // Site 3 follows the first GDN layer's reduction and state writes.
                let endpoint = observed(endpoint, &calls, (lane == 0).then_some(3), lane == 0);
                let mut stage = Stage::new(&fixture.resources(), 0..4, lane, endpoint);
                let mut work = stage.enter(1);
                let result = catch_unwind(AssertUnwindSafe(|| {
                    stage.execute(&mut work, SegmentInput::Tokens)
                }));
                if lane == 0 {
                    assert_eq!(
                        *result.unwrap_err().downcast::<&'static str>().unwrap(),
                        "original model execution panic"
                    );
                } else {
                    let error = result.unwrap().unwrap_err().to_string();
                    assert!(
                        error.contains("aborted"),
                        "peer must wake on abort, not timeout: {error}"
                    );
                }
                assert!(stage.module.is_failed());
                assert_eq!(stage.local.backend.capacity().active_transactions, 1);
                assert_eq!(work.states[0].kv_state().recurrent()[&0].position(), 2);
                assert_eq!(
                    stage.local.states[0].kv_state().recurrent()[&0].position(),
                    0
                );
                // Only this caller closes custody. A proven rollback does not unpoison TP.
                stage.rollback(work);
                assert_eq!(stage.local.backend.capacity().active_transactions, 0);
                let before = calls.load(Ordering::SeqCst);
                let mut newer = stage.enter(2);
                let result = catch_unwind(AssertUnwindSafe(|| {
                    stage.execute(&mut newer, SegmentInput::Tokens)
                }));
                assert!(
                    result
                        .unwrap()
                        .unwrap_err()
                        .to_string()
                        .contains("failed/stale")
                );
                assert_eq!(calls.load(Ordering::SeqCst), before);
                assert!(stage.module.is_failed());
                stage.rollback(newer);
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        thread.join().unwrap();
    }
    assert!(control.is_failed());
}

#[test]
fn agreement_ignores_owner_generation_but_execution_validates_it_locally() {
    let fixture = fixture::Fixture::for_degree(4, true);
    let resources = fixture.resources();
    let mut endpoints = support::endpoints_for(&[support::rank(7)]);
    let calls = Arc::new(AtomicUsize::new(0));
    let mut stage = Stage::new(
        &resources,
        0..4,
        0,
        observed(endpoints.remove(0), &calls, None, false),
    );
    let first = stage.enter(1);
    let identity = first.batch.logical_execution_identity(&[73, 91]).unwrap();
    let packet = stage
        .module
        .collective_schedule()
        .encode_agreement(&first.batch, &[73, 91], &SegmentInput::Tokens)
        .unwrap();
    assert_eq!(
        packet.len(),
        stage.module.collective_schedule().sites()[0]
            .shape
            .elements(2, 3)
            .unwrap()
    );
    let words = packet
        .as_chunks::<4>()
        .0
        .iter()
        .map(|limbs| {
            limbs
                .iter()
                .enumerate()
                .fold(0u64, |word, (i, limb)| word | ((*limb as u64) << (i * 16)))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        words,
        vec![
            2, 907, 0, 0, 0, 4, 1, 7, 0, 0, // version/scope/range/TP/noEP/token
            3, 2, // row and sequence counts
            73, 0, 2, 0, 91, 0, 1, 0, // session/context/end/phase, no local generation
            1, 0, 0, 2, 1, 0, 3, 0, 1, // interleaved token/position/sequence
        ]
    );
    stage.rollback(first);
    for s in &mut stage.local.states {
        stage.lifecycle.reset(s).unwrap();
    }
    let mut second = stage.enter(2);
    assert_ne!(second.batch.sequences()[0].execution_generation(), 0);
    assert_eq!(
        second.batch.logical_execution_identity(&[73, 91]).unwrap(),
        identity
    );
    assert_eq!(
        stage
            .module
            .collective_schedule()
            .encode_agreement(&second.batch, &[73, 91], &SegmentInput::Tokens)
            .unwrap(),
        packet
    );
    stage.execute(&mut second, SegmentInput::Tokens).unwrap();
    stage.rollback(second);
    let mut stale = stage.enter(3);
    stale.states[0].core_mut().reset();
    assert!(
        stage
            .execute(&mut stale, SegmentInput::Tokens)
            .unwrap_err()
            .to_string()
            .contains("generation")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0); // TP1 consumes logical sites, never enters wire.
    stage.rollback(stale);
}

#[test]
fn malformed_hidden_is_rejected_before_any_agreement_wire_call() {
    let fixture = fixture::Fixture::for_degree(4, true);
    let resources = fixture.resources();
    for (next_layer, rows, width, dtype) in [
        (1, 3, 8, RowsDType::F32),
        (2, 2, 8, RowsDType::F32),
        (2, 3, 7, RowsDType::F32),
        (2, 3, 8, RowsDType::Bf16),
    ] {
        let mut endpoints = support::endpoints_for(&[support::rank(7), support::rank(19)]);
        let calls = Arc::new(AtomicUsize::new(0));
        let mut stage = Stage::new(
            &resources,
            2..4,
            0,
            observed(endpoints.remove(0), &calls, None, false),
        );
        let mut work = stage.enter(1);
        let valid = SegmentInput::Hidden {
            next_layer: 2,
            rows: HostRows::new(
                RowsShape::new(3, 8).unwrap(),
                RowsDType::F32,
                None,
                vec![0.25; 24],
            )
            .unwrap(),
        };
        let packet = stage
            .module
            .collective_schedule()
            .encode_agreement(&work.batch, &[73, 91], &valid)
            .unwrap();
        assert_eq!(packet.len(), 56 + 16 * 2 + 20 * 3);
        assert_eq!(&packet[packet.len() - 24..], &[0.25; 24]);
        let input = SegmentInput::Hidden {
            next_layer,
            rows: HostRows::new(
                RowsShape::new(rows, width).unwrap(),
                dtype,
                None,
                vec![0.; rows * width],
            )
            .unwrap(),
        };
        let error = stage.execute(&mut work, input).unwrap_err();
        assert!(
            error.to_string().contains("hidden boundary/shape/dtype"),
            "{error}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(work.states[0].kv_state().recurrent()[&2].position(), 0);
        stage.rollback(work);
    }
}

#[test]
fn tp2_accepts_two_sequences_with_different_owner_local_generations() {
    let fixture = Arc::new(fixture::Fixture::for_degree(4, true));
    let endpoints = support::endpoints_for(&[support::rank(7), support::rank(19)]);
    let control = endpoints[0].control();
    control
        .begin(support::tx(1), Arc::new(AtomicBool::new(false)))
        .unwrap();
    let threads = endpoints
        .into_iter()
        .enumerate()
        .map(|(lane, endpoint)| {
            let fixture = Arc::clone(&fixture);
            std::thread::spawn(move || {
                let mut stage = Stage::new(&fixture.resources(), 0..4, lane, Box::new(endpoint));
                if lane == 1 {
                    for state in &mut stage.local.states {
                        stage.lifecycle.reset(state).unwrap();
                    }
                }
                let mut work = stage.enter(1);
                assert_eq!(
                    work.batch.sequences()[0].execution_generation(),
                    lane as u64
                );
                let identity = work.batch.logical_execution_identity(&[73, 91]).unwrap();
                let SegmentOutput::Logits(rows) =
                    stage.execute(&mut work, SegmentInput::Tokens).unwrap()
                else {
                    panic!("logits")
                };
                assert!(!stage.module.is_failed());
                stage.rollback(work);
                (identity, rows.values().to_vec())
            })
        })
        .collect::<Vec<_>>();
    let results = threads
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect::<Vec<_>>();
    control.finish(support::tx(1)).unwrap();
    assert_eq!(results[0].0, results[1].0);
    support::close(&results[0].1, &results[1].1);
}
