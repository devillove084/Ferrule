//! Owner-local model execution. No device object crosses the transport boundary.

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Error, ParallelRankId, Result};
use ferrule_model::decoder::{CpuKvView, KvCommitBinding, PackedDecoderBatch};
use ferrule_model::transformer::expert_parallel::RoutedSwiGluExecutor;
use ferrule_model::transformer::{
    LayerSegmentPlan, SegmentError, SegmentInput, SegmentOutput, StandardDecoderSegment,
};

use super::super::expert::ExpertGroup;
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
///
/// Commit metadata cannot execute a stage, even when a caller already owns a KV view:
/// ```compile_fail
/// use ferrule_model::decoder::KvCommitProjection;
/// use ferrule_model::transformer::SegmentInput;
/// use ferrule_runtime::parallel::pipeline::{PipelineStageProgram, PipelineExecutionContext};
/// fn execute<P: PipelineStageProgram>(program: &mut P, projection: &KvCommitProjection,
///     input: SegmentInput, view: &mut P::KvView, context: PipelineExecutionContext<'_>) {
///     program.execute(projection, input, view, context).unwrap();
/// }
/// ```
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

/// Owner-local state contract. State, lifecycle and view need not be Clone/Send.
/// Hybrid adapters implement this directly; standard programs use the blanket
/// adapter. Only the owner journal may advance core or publish sequence state.
/// Lifecycle checkout must return an independent working attachment while
/// retaining the source core/topology. Partial/failed copies remain lifecycle
/// custody until safely cleaned; a returned state is always journal custody.
/// Execute has the same host-storage/fence obligations as `PipelineStageProgram`.
///
#[doc = include_str!("../../../tests/compile/pipeline_state_program.md")]
pub trait PipelineStateProgram: 'static {
    #[cfg(feature = "cuda")]
    fn trace_committed(
        &mut self,
        _backend: &mut dyn std::any::Any,
        _transaction: ExecutionTransactionId,
        _source: &ferrule_common::host_kv::CommittedKvSource,
    ) -> Result<()> {
        Err(super::error(
            "program has no committed native trace capability",
        ))
    }

    type SequenceState: ferrule_model::decoder::DecoderSequence + 'static;
    type Lifecycle: ferrule_model::decoder::DecoderSequenceLifecycle<Self::SequenceState>;
    type KvView: 'static;
    fn plan(&self) -> &LayerSegmentPlan;
    fn new_lifecycle(&mut self) -> Result<Self::Lifecycle>;
    /// Authenticated CPU-reference attachment metadata, not a KV participant.
    #[cfg(feature = "cpu-reference")]
    fn hybrid_reference_expert_group(&self) -> Option<&ExpertGroup> {
        None
    }
    /// Owner-bound external attachment authenticated against the stage description.
    fn hybrid_expert_group(&self) -> Option<&ExpertGroup> {
        #[cfg(feature = "cpu-reference")]
        {
            self.hybrid_reference_expert_group()
        }
        #[cfg(not(feature = "cpu-reference"))]
        {
            None
        }
    }
    /// Root and child completion proof before physical rollback/abort.
    fn poll_quiescence(&mut self) -> Result<bool> {
        Ok(true)
    }
    /// Numeric storage contract for an explicitly bound hybrid program. Legacy
    /// owners default to F32; a BF16 configuration alone never changes the program.
    fn hybrid_precision(&self) -> ferrule_model::execution::ExecutionPrecisionPolicy {
        ferrule_model::execution::ExecutionPrecisionPolicy::f32()
    }
    /// Explicit compact layout; legacy/process constructors cannot admit this.
    fn hybrid_schema(&self) -> Option<&ferrule_model::decoder::HybridStateSchema> {
        None
    }
    fn validate_execution(
        &self,
        _: &KvCommitBinding,
        _: &PackedDecoderBatch,
        _: &[u64],
    ) -> Result<()> {
        Ok(())
    }
    fn execute_state(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [Self::SequenceState],
        view: &mut Self::KvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput>;
    /// Check every local model frontier against the packed sequence end, while
    /// the working core still describes the committed source.
    fn validate_frontier(
        &self,
        batch: &PackedDecoderBatch,
        states: &[Self::SequenceState],
    ) -> Result<()>;
    fn expert_outstanding(&self) -> usize {
        0
    }
    fn owner_stats(&mut self) -> Result<Vec<ExpertOwnerStats>> {
        Ok(Vec::new())
    }
    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}
impl<P: PipelineStageProgram> PipelineStateProgram for P {
    type SequenceState = ferrule_model::decoder::GenericDecoderSequenceState;
    type Lifecycle = ferrule_model::decoder::StandardSequenceLifecycle<(), ()>;
    type KvView = P::KvView;
    fn plan(&self) -> &LayerSegmentPlan {
        PipelineStageProgram::plan(self)
    }
    fn new_lifecycle(&mut self) -> Result<Self::Lifecycle> {
        Ok(Self::Lifecycle::new())
    }
    fn validate_execution(
        &self,
        binding: &KvCommitBinding,
        batch: &PackedDecoderBatch,
        ids: &[u64],
    ) -> Result<()> {
        PipelineStageProgram::validate_execution(self, binding, batch, ids)
    }
    fn execute_state(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        _: &mut [Self::SequenceState],
        view: &mut Self::KvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        PipelineStageProgram::execute(self, batch, input, view, context)
    }
    fn validate_frontier(&self, _: &PackedDecoderBatch, _: &[Self::SequenceState]) -> Result<()> {
        Ok(())
    }
    fn expert_outstanding(&self) -> usize {
        PipelineStageProgram::expert_outstanding(self)
    }
    fn owner_stats(&mut self) -> Result<Vec<ExpertOwnerStats>> {
        PipelineStageProgram::owner_stats(self)
    }
    fn shutdown(&mut self) -> Result<()> {
        PipelineStageProgram::shutdown(self)
    }
}
