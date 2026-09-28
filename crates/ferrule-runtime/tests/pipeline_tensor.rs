//! Public TP admission regressions. Sealed CPU PPxTP fault injection lives in
//! parallel::pipeline::tensor_tests, without exposing an unchecked TP constructor.
use ferrule_common::execution::ForwardPhase;
use ferrule_common::{
    ParallelRankId, ParallelTopologyId, ParallelismPlan, Result, ValidatedParallelTopology,
};
use ferrule_model::decoder::{
    CpuKvView, CpuPagedKvPool, DenseLogits, PackedDecoderBatch, PagedKvBackend, StandardGqaPlanes,
};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::{
    HostRows, LayerSegmentPlan, RowsDType, RowsShape, SegmentInput, SegmentOutput,
};
use ferrule_runtime::SessionId;
use ferrule_runtime::parallel::pipeline::{
    PipelineConfig, PipelineExecutionContext, PipelineParallelExecutor, PipelineStage,
    PipelineStageDescription, PipelineStageProgram,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn config() -> PipelineConfig {
    PipelineConfig {
        page_size: 2,
        max_pages: 8,
        max_positions: 16,
        max_batch_tokens: 8,
        session_capacity: 2,
        max_parameter_bytes: 1 << 30,
        precision: ExecutionPrecisionPolicy::f32(),
        max_ack_polls: 2,
    }
}
// Intentionally inherits validate_execution's default Ok implementation.
struct DefaultProgram(LayerSegmentPlan);
impl PipelineStageProgram for DefaultProgram {
    type KvView = CpuKvView;
    fn plan(&self) -> &LayerSegmentPlan {
        &self.0
    }
    fn execute(
        &mut self,
        batch: &PackedDecoderBatch,
        _: SegmentInput,
        _: &mut CpuKvView,
        _: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        let values = vec![1.0; batch.len() * 4];
        if self.0.owns_output() {
            Ok(SegmentOutput::Logits(DenseLogits::new(
                batch.len(),
                4,
                values,
            )?))
        } else {
            Ok(SegmentOutput::Hidden {
                next_layer: self.0.layers().end,
                rows: HostRows::new(
                    RowsShape::new(batch.len(), 4)?,
                    RowsDType::F32,
                    None,
                    values,
                )?,
            })
        }
    }
}
#[test]
fn generic_default_program_rejects_tp2_before_factory_but_retains_non_tp_pp() {
    for (pp, tp) in [(1, 2), (2, 2), (1, 1), (2, 1)] {
        let topology = ValidatedParallelTopology::new(
            ParallelTopologyId::new(501),
            (pp * tp) as u32,
            ParallelRankId::new(0),
            ParallelismPlan::validated(1, tp, 1, 1, 1, pp).unwrap(),
        )
        .unwrap();
        let plans = (0..pp)
            .map(|stage| {
                LayerSegmentPlan::new(
                    2,
                    stage * 2 / pp..(stage + 1) * 2 / pp,
                    stage == 0,
                    stage + 1 == pp,
                )
                .unwrap()
            })
            .collect();
        let invoked = Arc::new(AtomicBool::new(false));
        let factory_invoked = Arc::clone(&invoked);
        let result = PipelineParallelExecutor::new_thread_tensor_with_program(
            topology,
            plans,
            config(),
            move |_, _, plan| {
                factory_invoked.store(true, Ordering::Release);
                let planes = StandardGqaPlanes::new(
                    plan.layer_count(),
                    1,
                    2,
                    2,
                    16,
                    ferrule_common::execution::KvElementType::F32,
                )?;
                let description = PipelineStageDescription {
                    plan: plan.clone(),
                    config: config(),
                    physical_pages: config().max_pages,
                    hidden: 4,
                    vocabulary: 4,
                    kv_heads: 1,
                    head_dim: 2,
                    expert_group: None,
                };
                let capabilities = description.execution_capabilities()?;
                PipelineStage::new(
                    DefaultProgram(plan),
                    PagedKvBackend::new(CpuPagedKvPool::from_strategy(&planes, 8)?),
                    description,
                    capabilities,
                )
            },
        );
        if tp > 1 {
            let message = result
                .err()
                .expect("unsealed TP must be rejected")
                .to_string();
            assert!(
                message.contains("generic tensor program requires TP=1"),
                "{message}"
            );
            assert!(!invoked.load(Ordering::Acquire));
        } else {
            let mut pipeline = result.unwrap();
            assert!(invoked.load(Ordering::Acquire));
            pipeline
                .forward(SessionId(1), &[1, 2], ForwardPhase::Prefill)
                .unwrap();
            pipeline
                .forward(SessionId(1), &[3], ForwardPhase::Decode)
                .unwrap();
            assert!(!pipeline.is_quarantined());
            pipeline.shutdown().unwrap();
        }
    }
}

#[test]
fn process_transport_and_ep_tp_are_rejected_before_owner_dispatch() {
    use ferrule_runtime::parallel::pipeline::{
        PipelineCommand, PipelineRank, PipelineReply, PipelineTransport,
    };
    struct SerialTransport(Arc<AtomicBool>);
    impl PipelineTransport for SerialTransport {
        fn call_observed(
            &mut self,
            _: PipelineRank,
            _: PipelineCommand,
            _: &mut dyn FnMut(),
        ) -> Result<PipelineReply> {
            panic!("unsupported TP must not dispatch any owner command")
        }
        fn outstanding(&self) -> usize {
            0
        }
        fn shutdown(&mut self) -> Result<()> {
            self.0.store(true, Ordering::Release);
            Ok(())
        }
        fn quarantine(&mut self) {}
    }
    let topology = ValidatedParallelTopology::new(
        ParallelTopologyId::new(501),
        2,
        ParallelRankId::new(0),
        ParallelismPlan::validated(1, 2, 1, 1, 1, 1).unwrap(),
    )
    .unwrap();
    let plans = vec![LayerSegmentPlan::new(2, 0..2, true, true).unwrap()];
    let closed = Arc::new(AtomicBool::new(false));
    let result = PipelineParallelExecutor::new_with_transport(
        topology,
        plans.clone(),
        config(),
        SerialTransport(Arc::clone(&closed)),
    );
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("process/transport TP unsupported")
    );
    assert!(closed.load(Ordering::Acquire));
    let topology = ValidatedParallelTopology::new(
        ParallelTopologyId::new(501),
        2,
        ParallelRankId::new(0),
        ParallelismPlan::validated(1, 2, 2, 1, 1, 1).unwrap(),
    )
    .unwrap();
    let result = PipelineParallelExecutor::new_thread_tensor_with_program(
        topology,
        plans,
        config(),
        |_, _, _| -> Result<PipelineStage> {
            panic!("EP×TP must be rejected before owner creation")
        },
    );
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("EP×TP unsupported")
    );
    let rank = PipelineRank {
        local: ParallelRankId::new(0),
        global: ParallelRankId::new(0),
    };
    let mut transport = SerialTransport(closed);
    assert!(
        transport
            .call_batch_observed(
                vec![
                    (rank, PipelineCommand::Describe),
                    (rank, PipelineCommand::Describe)
                ],
                &mut || {}
            )
            .is_err()
    );
}
