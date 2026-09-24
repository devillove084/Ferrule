//! Owner-local model execution. No device object crosses the transport boundary.

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Error, ParallelRankId, Result};
use ferrule_model::decoder::{CpuKvView, KvCommitBinding, PackedDecoderBatch};
use ferrule_model::transformer::expert_parallel::RoutedSwiGluExecutor;
use ferrule_model::transformer::{
    LayerSegmentPlan, SegmentError, SegmentInput, SegmentOutput, StandardDecoderSegment,
};

use super::super::expert::{ExpertOwnerStats, ExpertParallelExecutor};

/// Optional routed expert injection, using the existing pipeline transaction.
pub struct PipelineExpertContext<'a> {
    pub source_rank: ParallelRankId,
    pub executor: &'a mut dyn RoutedSwiGluExecutor,
}

/// Call-scoped identity and cooperative liveness check; not a decision authority.
pub struct PipelineExecutionContext<'a> {
    pub transaction: ExecutionTransactionId,
    /// Stable session IDs in packed sequence order, shared across stages.
    pub sequence_ids: &'a [u64],
    pub experts: Option<PipelineExpertContext<'a>>,
    pub check_active: &'a mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
}

/// A prepared model segment and its owner-local operators.
///
/// No `Send` or `Sync` bound: create the program and KV backend INSIDE the owner.
/// GPU implementations may upload/download at the `HostRows` boundary. Returning
/// (including an error) must make host input/output storage safe to drop. Device
/// work and physical KV fences remain under the backend's custody contract; a lost
/// fence must never be reported as successful rollback or retirement.
pub trait PipelineStageProgram: 'static {
    type KvView: 'static;

    fn plan(&self) -> &LayerSegmentPlan;

    /// Optional runtime admission check against the actual owner-local lowering.
    /// A rejection is not a KV rollback or a device completion acknowledgement.
    fn validate_execution(
        &self,
        _: &KvCommitBinding,
        _: &PackedDecoderBatch,
        _: &[u64],
    ) -> Result<()> {
        Ok(())
    }

    fn execute(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        view: &mut Self::KvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput>;

    fn expert_outstanding(&self) -> usize {
        0
    }

    fn owner_stats(&mut self) -> Result<Vec<ExpertOwnerStats>> {
        Ok(Vec::new())
    }

    /// Called on the owner, including rejected stage startup.
    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

impl<P: PipelineStageProgram + ?Sized> PipelineStageProgram for Box<P> {
    type KvView = P::KvView;

    fn plan(&self) -> &LayerSegmentPlan {
        (**self).plan()
    }

    fn validate_execution(
        &self,
        binding: &KvCommitBinding,
        batch: &PackedDecoderBatch,
        sessions: &[u64],
    ) -> Result<()> {
        (**self).validate_execution(binding, batch, sessions)
    }

    fn execute(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        view: &mut Self::KvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        (**self).execute(batch, input, view, context)
    }

    fn expert_outstanding(&self) -> usize {
        (**self).expert_outstanding()
    }

    fn owner_stats(&mut self) -> Result<Vec<ExpertOwnerStats>> {
        (**self).owner_stats()
    }

    fn shutdown(&mut self) -> Result<()> {
        (**self).shutdown()
    }
}

/// CPU adapter around the existing segment. It deliberately contains no copied
/// forward path; standard GQA and typed unsupported capability behavior remain in
/// `StandardDecoderSegment`.
pub struct CpuPipelineStageProgram {
    segment: StandardDecoderSegment,
    experts: Option<ExpertParallelExecutor>,
}

impl CpuPipelineStageProgram {
    pub(crate) fn new(segment: StandardDecoderSegment) -> Self {
        Self {
            segment,
            experts: None,
        }
    }

    pub(crate) fn with_experts(mut self, experts: ExpertParallelExecutor) -> Self {
        self.experts = Some(experts);
        self
    }
}

pub(super) fn segment_error(source: SegmentError) -> Error {
    Error::ModelSource {
        source: Box::new(source),
    }
}

impl PipelineStageProgram for CpuPipelineStageProgram {
    type KvView = CpuKvView;

    fn plan(&self) -> &LayerSegmentPlan {
        self.segment.plan()
    }

    fn execute(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        view: &mut Self::KvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        let PipelineExecutionContext {
            transaction,
            sequence_ids,
            experts: external_experts,
            check_active,
        } = context;
        check_active(transaction)?;
        let result = if let Some(experts) = self.experts.as_mut() {
            let source_rank = experts.group().source_rank;
            self.segment.execute_with_experts(
                batch,
                input,
                view,
                (transaction, source_rank, sequence_ids),
                experts,
                check_active,
            )
        } else if let Some(experts) = external_experts {
            self.segment.execute_with_experts(
                batch,
                input,
                view,
                (transaction, experts.source_rank, sequence_ids),
                experts.executor,
                check_active,
            )
        } else {
            self.segment.execute(batch, input, view)
        }
        .map_err(segment_error)?;
        check_active(transaction)?;
        Ok(result)
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
            .map(|stats| stats.unwrap_or_default())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.experts
            .as_mut()
            .map(ExpertParallelExecutor::shutdown)
            .transpose()
            .map(|_| ())
    }
}
