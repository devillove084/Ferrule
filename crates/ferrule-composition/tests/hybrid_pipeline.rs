//! Real CPU hybrid segments plus a controlled fault wrapper; no copied forward.
#[path = "support/hybrid_fixture.rs"]
mod fixture;
use ferrule_common::execution::{ForwardPhase, KvLayoutSchema};
use ferrule_common::{
    Error, ParallelRankId, ParallelTopologyId, ParallelismPlan, Result, ValidatedParallelTopology,
};
use ferrule_composition::model_factory::{PipelineBuildOptions, PipelineRankBackend};
use ferrule_model::ModelExecutionBackend;
use ferrule_model::decoder::{
    CpuKvView, HybridDecoderSequenceState, HybridLayerState, HybridSequenceLifecycle,
    HybridStateSchema, PackedDecoderBatch,
};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::{LayerSegmentPlan, SegmentInput, SegmentOutput};
use ferrule_runtime::SessionId;
use ferrule_runtime::parallel::pipeline::*;
use fixture::Fixture;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

fn config() -> PipelineConfig {
    PipelineConfig {
        page_size: 2,
        max_pages: 48,
        max_positions: 32,
        max_batch_tokens: 8,
        session_capacity: 3,
        max_parameter_bytes: 1 << 20,
        precision: ExecutionPrecisionPolicy::f32(),
        max_ack_polls: 2,
    }
}
fn plans() -> Vec<LayerSegmentPlan> {
    vec![
        LayerSegmentPlan::new(4, 0..2, true, false).unwrap(),
        LayerSegmentPlan::new(4, 2..4, false, true).unwrap(),
    ]
}
fn topology() -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(809),
        2,
        ParallelRankId::new(0),
        ParallelismPlan::validated(1, 1, 1, 1, 1, 2).unwrap(),
    )
    .unwrap()
}
#[derive(Default)]
struct Probe {
    states: Mutex<Vec<(usize, HybridDecoderSequenceState)>>,
    fail: AtomicBool,
    cancel: AtomicBool,
    cancellation: Arc<AtomicBool>,
}
struct Observed {
    inner: HybridCpuPipelineStageProgram,
    probe: Arc<Probe>,
    rank: usize,
}
impl PipelineStateProgram for Observed {
    type SequenceState = HybridDecoderSequenceState;
    type Lifecycle = HybridSequenceLifecycle;
    type KvView = CpuKvView;
    fn plan(&self) -> &LayerSegmentPlan {
        self.inner.plan()
    }
    fn hybrid_schema(&self) -> Option<&HybridStateSchema> {
        self.inner.hybrid_schema()
    }
    fn new_lifecycle(&mut self) -> Result<Self::Lifecycle> {
        self.inner.new_lifecycle()
    }
    fn execute_state(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [Self::SequenceState],
        view: &mut CpuKvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        let output = self
            .inner
            .execute_state(batch, input, states, view, context)?;
        if self.rank == 1 && self.probe.fail.swap(false, Ordering::Relaxed) {
            return Err(Error::Execution {
                message: "injected post-model failure".into(),
            });
        }
        if self.rank == 1 && self.probe.cancel.swap(false, Ordering::Relaxed) {
            self.probe.cancellation.store(true, Ordering::Release);
        }
        Ok(output)
    }
    fn validate_frontier(
        &self,
        batch: &PackedDecoderBatch,
        states: &[Self::SequenceState],
    ) -> Result<()> {
        self.inner.validate_frontier(batch, states)?;
        // Test snapshot only. Production journal never clones state.
        self.probe
            .states
            .lock()
            .unwrap()
            .push((self.rank, states[0].clone()));
        Ok(())
    }
}
fn pipeline(fixture: Arc<Fixture>, probe: Arc<Probe>) -> PipelineParallelExecutor {
    let layout = HybridPipelineLayout::new(fixture.resources().spec(), &plans()).unwrap();
    PipelineParallelExecutor::new_hybrid_with_program(
        topology(),
        plans(),
        config(),
        layout,
        move |rank, plan| {
            PipelineStage::prepare_hybrid_cpu(&fixture.resources(), plan, config()).map(|stage| {
                stage.map_program(|inner| Observed {
                    inner,
                    probe,
                    rank: rank.local.get() as usize,
                })
            })
        },
    )
    .unwrap()
}
fn close(actual: &[f32], expected: &serde_json::Value) {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        let b = b.as_f64().unwrap() as f32;
        assert!((a - b).abs() <= 3e-5 + 3e-5 * b.abs(), "{a} != {b}");
    }
}
fn assert_state(probe: &Probe, expected: &serde_json::Value, position: usize) {
    let states = probe.states.lock().unwrap();
    for rank in 0..2 {
        let (_, state) = states.iter().rev().find(|(r, _)| *r == rank).unwrap();
        for (entry, layer) in state
            .kv_state()
            .schema()
            .entries()
            .iter()
            .zip(state.kv_state().layers())
        {
            if let HybridLayerState::GatedDeltaNet(s) = layer {
                assert_eq!(s.position(), position);
                let expected = &expected[entry.global_layer.to_string()];
                close(s.conv_history(), &expected["conv"]);
                close(s.recurrent(), &expected["recurrent"]);
            }
        }
    }
}
#[test]
fn real_pp2_prefill_decode_fork_cow_and_compact_budget_match_frozen_oracle() {
    let fixture = Arc::new(Fixture::new());
    let probe = Arc::new(Probe::default());
    let mut pp = pipeline(fixture.clone(), probe.clone());
    let prefill = pp
        .forward(SessionId(1), &[1, 3, 2], ForwardPhase::Prefill)
        .unwrap();
    for i in 0..3 {
        close(prefill.logits.row(i).unwrap(), &fixture.oracle["logits"][i]);
    }
    assert_state(&probe, &fixture.oracle["prefill_states"], 3);
    pp.fork_session(SessionId(1), SessionId(2)).unwrap();
    for session in [SessionId(2), SessionId(1)] {
        let output = pp.forward(session, &[5], ForwardPhase::Decode).unwrap();
        close(output.logits.row(0).unwrap(), &fixture.oracle["logits"][3]);
        let output = pp.forward(session, &[7], ForwardPhase::Decode).unwrap();
        close(output.logits.row(0).unwrap(), &fixture.oracle["logits"][4]);
        assert_state(&probe, &fixture.oracle["decoded_states"], 5);
    }
    assert_eq!(pp.coordinator().publication_count(), 5);
    assert_eq!(pp.execution_scopes().kv_participants().len(), 2);
    let schema = HybridStateSchema::from_spec(fixture.resources().spec(), 4).unwrap();
    let logical_page_bytes = schema
        .kv_planes(2, 32)
        .unwrap()
        .checked_page_bytes()
        .unwrap() as u64;
    assert_eq!(
        pp.physical_kv_bytes().unwrap(),
        logical_page_bytes * config().max_pages as u64 * 2
    );
    for stage in pp.stage_descriptions() {
        assert_eq!(stage.physical_pages, config().max_pages * 2);
    }
    pp.release_session(SessionId(2)).unwrap();
    pp.release_session(SessionId(1)).unwrap();
    assert_eq!(pp.page_manager().allocated_pages(), 0);
    pp.shutdown().unwrap();
}
#[test]
fn real_hybrid_failure_and_cancel_leave_source_replayable() {
    for cancel in [false, true] {
        let fixture = Arc::new(Fixture::new());
        let probe = Arc::new(Probe::default());
        let mut pp = pipeline(fixture.clone(), probe.clone());
        pp.forward(SessionId(1), &[1, 3, 2], ForwardPhase::Prefill)
            .unwrap();
        if cancel {
            probe.cancel.store(true, Ordering::Relaxed);
        } else {
            probe.fail.store(true, Ordering::Relaxed);
        }
        assert!(
            pp.forward_cancellable(
                SessionId(1),
                &[9],
                ForwardPhase::Decode,
                &probe.cancellation
            )
            .is_err()
        );
        assert_eq!(pp.coordinator().publication_count(), 1);
        assert!(!pp.is_quarantined());
        probe.cancellation.store(false, Ordering::Release);
        let output = pp
            .forward(SessionId(1), &[5], ForwardPhase::Decode)
            .unwrap();
        close(output.logits.row(0).unwrap(), &fixture.oracle["logits"][3]);
        let output = pp
            .forward(SessionId(1), &[7], ForwardPhase::Decode)
            .unwrap();
        close(output.logits.row(0).unwrap(), &fixture.oracle["logits"][4]);
        assert_state(&probe, &fixture.oracle["decoded_states"], 5);
        pp.shutdown().unwrap();
    }
}
#[test]
fn stage_state_and_2l_kv_budgets_aggregate_colocated_devices() {
    let fixture = Fixture::new();
    let resources = fixture.resources();
    let layout = HybridPipelineLayout::new(resources.spec(), &plans()).unwrap();
    assert_eq!(layout.kv_layers(), 2);
    let stages = (0..2)
        .map(|i| {
            layout
                .stage_budget(i, config(), 0, 100, 20, 10, ModelExecutionBackend::Cpu)
                .unwrap()
        })
        .collect::<Vec<_>>();
    for (i, stage) in stages.iter().enumerate() {
        assert_eq!(
            stage.state,
            layout.schemas()[i].per_sequence_state_bytes().unwrap() as u64 * 4
        );
        assert_eq!(
            stage.kv,
            layout.schemas()[i]
                .kv_planes(2, 32)
                .unwrap()
                .checked_page_bytes()
                .unwrap() as u64
                * 96
        );
    }
    let required = stages.iter().map(|s| s.total().unwrap()).sum::<u64>();
    assert_eq!(
        HybridDeviceBudget::admit(&stages, &[(0, required)]).unwrap()[0].required,
        required
    );
    assert!(HybridDeviceBudget::admit(&stages, &[(0, required - 1)]).is_err());
    assert!(HybridDeviceBudget::admit(&stages, &[(0, required), (0, required)]).is_err());
    assert!(HybridDeviceBudget::admit(&stages, &[]).is_err());
    assert!(HybridDeviceBudget::admit(&stages, &[(1, required)]).is_err());
    assert!(
        layout
            .stage_budget(0, config(), 0, u64::MAX, 1, 0, ModelExecutionBackend::Cpu)
            .is_err()
    );
}
#[test]
fn hybrid_does_not_enter_legacy_standard_or_process_factory() {
    let fixture = Arc::new(Fixture::new());
    let make = fixture.clone();
    assert!(
        PipelineParallelExecutor::new_with_program(
            topology(),
            plans(),
            config(),
            move |_, plan| PipelineStage::prepare_hybrid_cpu(&make.resources(), plan, config())
        )
        .is_err()
    );
    let boots = plans()
        .into_iter()
        .enumerate()
        .map(|(i, plan)| PipelineStageBoot {
            rank: PipelineRank {
                local: ParallelRankId::new(i as u32),
                global: ParallelRankId::new(i as u32),
            },
            plan,
            physical_pages: config().max_pages * 2,
            config: config(),
            program_spec: vec![],
        })
        .collect();
    assert!(
        PipelineParallelExecutor::new_with_factory(topology(), boots, move |boot| {
            let stage = PipelineStage::prepare_hybrid_cpu(
                &fixture.resources(),
                boot.plan.clone(),
                config(),
            )?;
            Ok(PipelineStageWorker::new(&boot, stage)?.boxed())
        })
        .is_err()
    );
    let mut options = PipelineBuildOptions {
        parallelism: topology().plan(),
        ..Default::default()
    };
    assert_eq!(
        options
            .hybrid_cpu_precision(&ferrule_model::ModelFamily::Qwen35)
            .unwrap(),
        ExecutionPrecisionPolicy::f32()
    );
    assert!(
        options
            .hybrid_cpu_precision(&ferrule_model::ModelFamily::Qwen35Moe)
            .is_err()
    );
    options.rank_backend = PipelineRankBackend::Process;
    assert!(
        options
            .hybrid_cpu_precision(&ferrule_model::ModelFamily::Qwen35)
            .is_err()
    );
}
#[test]
fn hybrid_schema_precision_and_topology_mismatch_fail_before_owner_creation() {
    let fixture = Arc::new(Fixture::new());
    let resources = fixture.resources();
    let recurrent_only = vec![
        LayerSegmentPlan::new(4, 0..1, true, false).unwrap(),
        LayerSegmentPlan::new(4, 1..4, false, true).unwrap(),
    ];
    assert!(HybridPipelineLayout::new(resources.spec(), &recurrent_only).is_err());
    let mut wrong = config();
    wrong.precision = ExecutionPrecisionPolicy::bf16_compatibility();
    assert!(PipelineStage::prepare_hybrid_cpu(&resources, plans().remove(0), wrong).is_err());
    let layout = HybridPipelineLayout::new(resources.spec(), &plans()).unwrap();
    assert!(
        PipelineParallelExecutor::new_hybrid_with_program(
            topology(),
            plans(),
            wrong,
            layout,
            move |_, plan| {
                let _ = (&fixture, wrong, plan);
                panic!("invalid precision must reject before factory");
                #[allow(unreachable_code)]
                PipelineStage::prepare_hybrid_cpu(&fixture.resources(), plan, wrong)
            }
        )
        .is_err()
    );
}
