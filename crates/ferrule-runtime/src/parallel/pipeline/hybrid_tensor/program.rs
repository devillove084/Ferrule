//! Owner-local execution and child custody; the worker journal owns state.
use super::*;

/// A real owner-local model program. External experts are constructed only on
/// the explicitly attached source, never once per TP lane or KV participant.
pub struct HybridCpuTensorPipelineStageProgram {
    module: HybridTensorCpuSegment,
    experts: Option<ExpertParallelExecutor>,
    group: Option<ExpertGroup>,
}
impl HybridCpuTensorPipelineStageProgram {
    pub(super) fn new(
        module: HybridTensorCpuSegment,
        experts: Option<ExpertParallelExecutor>,
        group: Option<ExpertGroup>,
    ) -> Self {
        Self {
            module,
            experts,
            group,
        }
    }
}
impl PipelineStateProgram for HybridCpuTensorPipelineStageProgram {
    type SequenceState = HybridTensorSequenceState;
    type Lifecycle = HybridTensorSequenceLifecycle;
    type KvView = CpuKvView;
    fn plan(&self) -> &LayerSegmentPlan {
        self.module.plan()
    }
    fn hybrid_schema(&self) -> Option<&HybridStateSchema> {
        Some(self.module.schema())
    }
    fn hybrid_reference_expert_group(&self) -> Option<&ExpertGroup> {
        self.group.as_ref()
    }
    fn new_lifecycle(&mut self) -> Result<Self::Lifecycle> {
        Ok(self.module.lifecycle())
    }
    fn execute_state(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [Self::SequenceState],
        view: &mut CpuKvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        if context.experts.is_some() || batch.sequences().len() != 1 {
            return Err(error(
                "hybrid CPU owner requires its bound attachment and one sequence",
            ));
        }
        let parallel = self.module.parallel_plan();
        let binding = self
            .experts
            .as_mut()
            .map(|executor| {
                self.module.bind_expert_call(
                    parallel.scope(),
                    parallel.expert_attachment().expect("source attachment"),
                    executor,
                )
            })
            .transpose()?;
        let result = self.module.execute_in_transaction(
            context.transaction,
            batch,
            input,
            states,
            view,
            context.sequence_ids,
            binding,
            context.check_active,
        );
        if let Some(experts) = self.experts.as_mut() {
            experts.ensure_dispatch_quiescent();
        }
        result
    }
    fn validate_frontier(
        &self,
        batch: &PackedDecoderBatch,
        states: &[Self::SequenceState],
    ) -> Result<()> {
        self.module.validate_frontier(batch, states)
    }
    fn expert_outstanding(&self) -> usize {
        self.experts
            .as_ref()
            .map_or(0, ExpertParallelExecutor::outstanding)
    }
    fn owner_stats(&mut self) -> Result<Vec<ExpertOwnerStats>> {
        self.experts
            .as_mut()
            .map(ExpertParallelExecutor::owner_stats)
            .transpose()
            .map(Option::unwrap_or_default)
    }
    fn shutdown(&mut self) -> Result<()> {
        if let Some(experts) = self.experts.as_mut() {
            experts.ensure_dispatch_quiescent();
            experts.shutdown()?;
        }
        Ok(())
    }
}
