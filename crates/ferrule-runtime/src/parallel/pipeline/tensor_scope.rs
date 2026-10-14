//! Logical packed-row identity and immutable operator schedules for TP. No physical KV slots, sequence
//! topology IDs, device pointers or rank-local execution indices participate.

use ferrule_common::{ParallelRankId, Result};
use ferrule_model::TensorRole;
use ferrule_model::decoder::{KvCommitBinding, PackedDecoderBatch};
use ferrule_model::transformer::parallel::{
    TensorParallelLinearPartition, TensorParallelLinearPlan,
};
use ferrule_model::transformer::{
    BoundDecoderResources, LayerSegmentPlan, SegmentInput, SegmentOutput,
};

use super::{PipelineExecutionContext, PipelineStageProgram, error};
use crate::parallel::collective::HostCollectiveKind;
use crate::parallel::tensor::decoder_collective::DecoderTensorCollectiveControl;

/// Binding plus the model's exact logical identity, never a weak commit projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorBatchIdentity {
    binding: KvCommitBinding,
    logical: ferrule_model::decoder::LogicalExecutionIdentity,
    /// Packed sequence cardinality, sealed alongside the logical identity.
    sequences: usize,
}
impl TensorBatchIdentity {
    pub fn from_packed(
        binding: &KvCommitBinding,
        batch: &PackedDecoderBatch,
        sessions: &[u64],
    ) -> Result<Self> {
        let sequences = batch.sequences().len();
        Ok(Self {
            binding: binding.clone(),
            logical: batch.logical_execution_identity(sessions)?,
            sequences,
        })
    }
    pub(crate) fn from_logical(
        binding: &KvCommitBinding,
        logical: ferrule_model::decoder::LogicalExecutionIdentity,
        sequences: usize,
    ) -> Self {
        Self {
            binding: binding.clone(),
            logical,
            sequences,
        }
    }
    pub fn binding(&self) -> &KvCommitBinding {
        &self.binding
    }
    pub fn rows(&self) -> usize {
        self.logical.rows()
    }
    pub fn sequences(&self) -> usize {
        self.sequences
    }
}

/// Per-member input element count, including independently checked sequence
/// and row coefficients, without manufacturing a dense model shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCollectiveShape {
    Fixed(usize),
    PerRow(usize),
    Rows {
        fixed: usize,
        per_row: usize,
    },
    PerSequence {
        fixed: usize,
        per_sequence: usize,
        per_row: usize,
    },
}
impl TensorCollectiveShape {
    /// Checked element count for the complete packed batch dimensions.
    pub fn elements(self, sequences: usize, rows: usize) -> Result<usize> {
        let (fixed, per_sequence, per_row) = match self {
            Self::Fixed(elements) => (elements, 0, 0),
            Self::PerRow(elements) => (0, 0, elements),
            Self::Rows { fixed, per_row } => (fixed, 0, per_row),
            Self::PerSequence {
                fixed,
                per_sequence,
                per_row,
            } => (fixed, per_sequence, per_row),
        };
        if fixed == 0 && per_sequence == 0 && per_row == 0 {
            return Err(error("empty tensor collective shape"));
        }
        per_sequence
            .checked_mul(sequences)
            .and_then(|n| n.checked_add(fixed))
            .and_then(|n| n.checked_add(per_row.checked_mul(rows)?))
            .filter(|elements| *elements != 0)
            .ok_or_else(|| error("tensor collective shape overflow or empty batch"))
    }
}

/// Semantic sites used by hybrid adapters. The runtime authenticates the
/// operation order and transport shape; model code owns the actual math.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCollectiveSemantic {
    InputAgreement,
    AttentionOutput,
    GqaOutput,
    /// BF16 K/V head replication within the owner's global layer.
    KvReplication,
    GatedDeltaOutput,
    DenseMlpOutput,
    SharedExpertOutput,
    RoutedExpertOutput,
    OutputHead,
    Frontier,
}

/// Dense linear sites retain their complete validated plan. Semantic sites do
/// not manufacture a dense projection merely to describe their wire shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorCollectiveOperator {
    Linear(TensorParallelLinearPlan),
    /// Exact model-exported protocol metadata; equality seals kind and version too.
    Protocol {
        version: u64,
        operation: TensorCollectiveSemantic,
        kind: HostCollectiveKind,
        shape: TensorCollectiveShape,
    },
}
impl From<TensorParallelLinearPlan> for TensorCollectiveOperator {
    fn from(plan: TensorParallelLinearPlan) -> Self {
        Self::Linear(plan)
    }
}

/// One collective-producing operator, in forward order. Registration compares
/// the complete semantic operator, shape and global layer on every owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCollectiveSite {
    pub site: u64,
    /// Global decoder layer; `None` denotes a stage-wide site or output head.
    pub layer: Option<usize>,
    pub operator: TensorCollectiveOperator,
}
impl TensorCollectiveSite {
    pub fn kind(&self) -> HostCollectiveKind {
        match &self.operator {
            TensorCollectiveOperator::Linear(plan) => match plan.partition() {
                TensorParallelLinearPartition::Row => HostCollectiveKind::AllReduceSumF32,
                TensorParallelLinearPartition::Column => HostCollectiveKind::AllGatherF32,
            },
            TensorCollectiveOperator::Protocol { kind, .. } => *kind,
        }
    }
    pub fn shape(&self) -> TensorCollectiveShape {
        match &self.operator {
            TensorCollectiveOperator::Linear(plan) => {
                TensorCollectiveShape::PerRow(match plan.partition() {
                    TensorParallelLinearPartition::Row => plan.out_features(),
                    TensorParallelLinearPartition::Column => {
                        plan.out_features().div_ceil(plan.ranks())
                    }
                })
            }
            TensorCollectiveOperator::Protocol { shape, .. } => *shape,
        }
    }
    pub fn elements(&self, sequences: usize, rows: usize) -> Result<usize> {
        self.shape().elements(sequences, rows)
    }
    pub(crate) fn validate(&self, members: usize) -> Result<()> {
        if let TensorCollectiveOperator::Linear(plan) = &self.operator
            && plan.ranks() != members
        {
            return Err(error(
                "tensor linear plan degree differs from collective roster",
            ));
        }
        if let TensorCollectiveOperator::Protocol {
            operation, shape, ..
        } = &self.operator
        {
            let stage_wide = matches!(
                operation,
                TensorCollectiveSemantic::InputAgreement
                    | TensorCollectiveSemantic::OutputHead
                    | TensorCollectiveSemantic::Frontier
            );
            if stage_wide != self.layer.is_none()
                || (matches!(operation, TensorCollectiveSemantic::Frontier)
                    && !matches!(
                        shape,
                        TensorCollectiveShape::Fixed(_)
                            | TensorCollectiveShape::PerSequence {
                                per_sequence: 0,
                                per_row: 0,
                                ..
                            }
                    ))
            {
                return Err(error("invalid tensor semantic layer or frontier shape"));
            }
        }
        self.elements(1, 1).map(|_| ())
    }
    pub fn standard(
        resources: &BoundDecoderResources,
        segment: &LayerSegmentPlan,
        tp: usize,
    ) -> Result<Vec<Self>> {
        if tp == 0 {
            return Err(error("zero tensor degree"));
        }
        let mut sites = Vec::new();
        for layer in segment.layers() {
            for role in [TensorRole::AttentionOutput, TensorRole::DenseMlpDown] {
                let binding = resources.require_layer(layer, role)?;
                let [out_features, in_features] = binding.weight().logical_shape() else {
                    return Err(error("tensor collective requires a matrix operator"));
                };
                sites.push(Self {
                    site: binding.id().get(),
                    layer: Some(layer),
                    operator: TensorParallelLinearPlan::new(
                        *out_features,
                        *in_features,
                        tp,
                        TensorParallelLinearPartition::Row,
                    )?
                    .into(),
                });
            }
        }
        if segment.owns_output() {
            sites.push(Self {
                site: resources.require_static(TensorRole::OutputHead)?.id().get(),
                layer: None,
                operator: TensorParallelLinearPlan::new(
                    resources.spec().vocab_size(),
                    resources.spec().hidden_size(),
                    tp,
                    TensorParallelLinearPartition::Column,
                )?
                .into(),
            });
        }
        Ok(sites)
    }
}

/// Translate the model-owned wire schedule without reconstructing its loop or
/// agreement packet. The exported ordinal/layer/semantic/kind/shape are sealed
/// verbatim in the runtime registration metadata.
pub fn translate_hybrid_schedule(
    plan: &ferrule_model::transformer::hybrid_parallel::HybridParallelPlan,
    segment: &LayerSegmentPlan,
) -> Result<Vec<TensorCollectiveSite>> {
    use TensorCollectiveSemantic as Op;
    use TensorCollectiveShape as Shape;
    use ferrule_model::transformer::hybrid_parallel::{
        HYBRID_AGREEMENT_VERSION, HybridCollectiveKind, HybridCollectiveSemantic,
    };
    // Legacy model sites have no per-site payload dtype. Do not silently lower
    // BF16 activations or partials to an F32 schedule while typed execution is gated.
    if plan.precision().is_mixed() {
        return Err(error(
            "Mixed rank schedule requires explicit validated per-site payload precision",
        ));
    }
    let schedule = plan.collective_schedule(segment)?;
    if schedule.agreement_version() != HYBRID_AGREEMENT_VERSION {
        return Err(error("unsupported hybrid agreement schedule version"));
    }
    schedule
        .wire_sites()
        .iter()
        .map(|site| {
            let operation = match site.semantic {
                HybridCollectiveSemantic::Agreement => Op::InputAgreement,
                HybridCollectiveSemantic::AttentionOutput => Op::AttentionOutput,
                HybridCollectiveSemantic::DenseMlpDown => Op::DenseMlpOutput,
                HybridCollectiveSemantic::SharedExpertDown => Op::SharedExpertOutput,
                HybridCollectiveSemantic::RoutedExpertOutput => Op::RoutedExpertOutput,
                HybridCollectiveSemantic::OutputHead => Op::OutputHead,
                HybridCollectiveSemantic::Completion => Op::Frontier,
            };
            let kind = match site.kind {
                HybridCollectiveKind::AllGather => HostCollectiveKind::AllGatherF32,
                HybridCollectiveKind::ReduceSum => HostCollectiveKind::AllReduceSumF32,
            };
            let shape = Shape::PerSequence {
                fixed: site.shape.fixed,
                per_sequence: site.shape.per_sequence,
                per_row: site.shape.per_row,
            };
            Ok(TensorCollectiveSite {
                site: site.ordinal,
                layer: site.global_layer,
                operator: TensorCollectiveOperator::Protocol {
                    version: schedule.agreement_version(),
                    operation,
                    kind,
                    shape,
                },
            })
        })
        .collect()
}

/// Runtime decorator: keeps model math/traits unchanged and checks the actual
/// owner packed batch before entering the physical KV backend or model program.
pub struct TensorScopedProgram<P> {
    inner: P,
    control: DecoderTensorCollectiveControl,
    owner: ParallelRankId,
}
impl<P> TensorScopedProgram<P> {
    pub fn new(inner: P, control: DecoderTensorCollectiveControl, owner: ParallelRankId) -> Self {
        Self {
            inner,
            control,
            owner,
        }
    }
}
impl<P: PipelineStageProgram> PipelineStageProgram for TensorScopedProgram<P> {
    type KvView = P::KvView;
    fn plan(&self) -> &LayerSegmentPlan {
        self.inner.plan()
    }
    fn validate_execution(
        &self,
        binding: &KvCommitBinding,
        batch: &PackedDecoderBatch,
        sessions: &[u64],
    ) -> Result<()> {
        let result = (|| {
            let identity = TensorBatchIdentity::from_packed(binding, batch, sessions)?;
            self.control
                .validate_owner_execution(self.owner, &identity)?;
            self.inner.validate_execution(binding, batch, sessions)
        })();
        result.inspect_err(|_| self.control.fail_execution(binding.transaction()))
    }
    fn execute(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        view: &mut Self::KvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        let transaction = context.transaction;
        let result = (|| {
            self.control.check_owner_execution(
                self.owner,
                transaction,
                batch,
                context.sequence_ids,
            )?;
            let output = self.inner.execute(batch, input, view, context)?;
            self.control
                .check_owner_completion(self.owner, transaction)?;
            Ok(output)
        })();
        result.inspect_err(|_| self.control.fail_execution(transaction))
    }
    fn expert_outstanding(&self) -> usize {
        self.inner.expert_outstanding()
    }
    fn owner_stats(&mut self) -> Result<Vec<crate::parallel::expert::ExpertOwnerStats>> {
        self.inner.owner_stats()
    }
    fn shutdown(&mut self) -> Result<()> {
        self.inner.shutdown()
    }
}

/// State-aware sealed TP decorator. The owner journal invokes
/// `validate_execution` before backend enter, so this wrapper authenticates the
/// complete packed identity without owning KV, publication or rollback.
pub struct TensorScopedStateProgram<P> {
    inner: P,
    control: DecoderTensorCollectiveControl,
    owner: ParallelRankId,
}
impl<P> TensorScopedStateProgram<P> {
    pub fn new(inner: P, control: DecoderTensorCollectiveControl, owner: ParallelRankId) -> Self {
        Self {
            inner,
            control,
            owner,
        }
    }
}
impl<P: super::PipelineStateProgram> super::PipelineStateProgram for TensorScopedStateProgram<P> {
    #[cfg(feature = "cpu-reference")]
    fn hybrid_reference_expert_group(&self) -> Option<&crate::parallel::expert::ExpertGroup> {
        self.inner.hybrid_reference_expert_group()
    }
    fn hybrid_expert_group(&self) -> Option<&crate::parallel::expert::ExpertGroup> {
        self.inner.hybrid_expert_group()
    }
    fn poll_quiescence(&mut self) -> Result<bool> {
        self.inner.poll_quiescence()
    }
    type SequenceState = P::SequenceState;
    type Lifecycle = P::Lifecycle;
    type KvView = P::KvView;

    fn plan(&self) -> &LayerSegmentPlan {
        self.inner.plan()
    }
    fn new_lifecycle(&mut self) -> Result<Self::Lifecycle> {
        self.inner.new_lifecycle()
    }
    fn hybrid_schema(&self) -> Option<&ferrule_model::decoder::HybridStateSchema> {
        self.inner.hybrid_schema()
    }
    fn hybrid_precision(&self) -> ferrule_model::execution::ExecutionPrecisionPolicy {
        self.inner.hybrid_precision()
    }
    #[cfg(feature = "cuda")]
    fn trace_committed(
        &mut self,
        backend: &mut dyn std::any::Any,
        transaction: ferrule_common::execution::ExecutionTransactionId,
        source: &ferrule_common::host_kv::CommittedKvSource,
    ) -> Result<()> {
        self.inner.trace_committed(backend, transaction, source)
    }

    fn validate_execution(
        &self,
        binding: &KvCommitBinding,
        batch: &PackedDecoderBatch,
        sessions: &[u64],
    ) -> Result<()> {
        let result = (|| {
            let identity = TensorBatchIdentity::from_packed(binding, batch, sessions)?;
            self.control
                .validate_owner_execution(self.owner, &identity)?;
            self.inner.validate_execution(binding, batch, sessions)
        })();
        result.inspect_err(|_| self.control.fail_execution(binding.transaction()))
    }
    fn execute_state(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [Self::SequenceState],
        view: &mut Self::KvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        let transaction = context.transaction;
        let result = (|| {
            self.control.check_owner_execution(
                self.owner,
                transaction,
                batch,
                context.sequence_ids,
            )?;
            let output = self
                .inner
                .execute_state(batch, input, states, view, context)?;
            self.control
                .check_owner_completion(self.owner, transaction)?;
            Ok(output)
        })();
        result.inspect_err(|_| self.control.fail_execution(transaction))
    }
    fn validate_frontier(
        &self,
        batch: &PackedDecoderBatch,
        states: &[Self::SequenceState],
    ) -> Result<()> {
        self.inner.validate_frontier(batch, states)
    }
    fn expert_outstanding(&self) -> usize {
        self.inner.expert_outstanding()
    }
    fn owner_stats(&mut self) -> Result<Vec<crate::parallel::expert::ExpertOwnerStats>> {
        self.inner.owner_stats()
    }
    fn shutdown(&mut self) -> Result<()> {
        self.inner.shutdown()
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/parallel/tensor_scope/sealed.rs"]
pub(crate) mod tests;

#[cfg(test)]
#[path = "../../../tests/unit/parallel/tensor_scope/semantic.rs"]
mod semantic_tests;

#[cfg(test)]
#[path = "../../../tests/unit/parallel/tensor_scope/state_program.rs"]
mod state_program_tests;

#[cfg(test)]
#[path = "../../../tests/unit/parallel/tensor_scope/per_sequence.rs"]
mod per_sequence_tests;
