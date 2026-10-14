use super::*;
use ferrule_common::Result;
use ferrule_model::decoder::{
    CpuKvView, CpuPagedKvPool, PackedDecoderBatch, PagedKvBackend, StandardGqaPlanes,
};
use ferrule_model::transformer::{LayerSegmentPlan, SegmentInput, SegmentOutput};
use ferrule_runtime::parallel::pipeline::{
    PipelineExecutionContext, PipelineStage, PipelineStageDescription, PipelineStageProgram,
};
use std::cell::Cell;
use std::rc::Rc;

struct Program {
    plan: LayerSegmentPlan,
    shutdown: Rc<Cell<bool>>,
}
impl PipelineStageProgram for Program {
    type KvView = CpuKvView;
    fn plan(&self) -> &LayerSegmentPlan {
        &self.plan
    }
    fn execute(
        &mut self,
        _: &PackedDecoderBatch,
        _: SegmentInput,
        _: &mut CpuKvView,
        _: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        panic!("capacity validation must not execute a model")
    }
    fn shutdown(&mut self) -> Result<()> {
        self.shutdown.set(true);
        Ok(())
    }
}

#[test]
fn actual_cpu_pool_must_match_declared_l_or_2l() {
    for declared in [7, 14] {
        for allocated in [7, 14] {
            let c = config(7);
            let plan = LayerSegmentPlan::new(1, 0..1, true, true).unwrap();
            let description = PipelineStageDescription {
                plan: plan.clone(),
                physical_pages: declared,
                config: c,
                hidden: 2,
                vocabulary: 4,
                kv_heads: 1,
                head_dim: 2,
                expert_group: None,
            };
            let capabilities = description.execution_capabilities().unwrap();
            let planes =
                StandardGqaPlanes::new(1, 1, 2, c.page_size, c.max_positions, c.dtype()).unwrap();
            let pool = CpuPagedKvPool::from_strategy(&planes, allocated).unwrap();
            let shutdown = Rc::new(Cell::new(false));
            let result = PipelineStage::new(
                Program {
                    plan,
                    shutdown: Rc::clone(&shutdown),
                },
                PagedKvBackend::new(pool),
                description,
                capabilities,
            );
            assert_eq!(result.is_ok(), declared == allocated);
            if declared != allocated {
                assert!(shutdown.get(), "rejected program must be shut down");
            }
        }
    }
}
