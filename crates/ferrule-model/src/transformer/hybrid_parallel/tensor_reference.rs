//! CPU rank-local image executing the existing hybrid layer loop.
use super::tensor_state::{HybridTensorSequenceLifecycle, HybridTensorSequenceState};
use super::{HybridCollectiveKind, HybridCollectiveSchedule, HybridCollectiveSemantic};
use super::{
    HybridExpertAttachment, HybridExpertScope, HybridLocalAttention, HybridParallelPlan,
    HybridRankLayout, invalid,
};
use crate::decoder::*;
use crate::execution::ExecutionPrecisionPolicy;
use crate::transformer::expert_parallel::{RoutedSwiGluExecutor, RoutedSwiGluRequest};
use crate::transformer::parallel::{TensorParallelCollective, TensorParallelStagePlan};
use crate::transformer::standard::{LayerOperators, ready};
use crate::transformer::*;
use ferrule_common::{ParallelRankId, Result, execution::ExecutionTransactionId};
use std::{cell::RefCell, sync::Arc};

#[path = "tensor_reference/operators.rs"]
mod operators;

/// Caller-consistency binding for one routed call, NOT an authentication proof.
/// Runtime must create and validate the executor's actual owner roster. An
/// arbitrary RoutedSwiGluExecutor cannot be authenticated by supplied metadata.
pub struct HybridExpertCallBinding<'a> {
    scope: HybridExpertScope,
    attachment: HybridExpertAttachment,
    executor: &'a mut dyn RoutedSwiGluExecutor,
}

pub struct HybridTensorCpuSegment {
    segment: HybridDecoderSegment,
    lifecycle: HybridTensorSequenceLifecycle,
    parallel: HybridParallelPlan,
    schedule: HybridCollectiveSchedule,
    collective: Box<dyn StandardTensorCollective>,
    watermark: Option<ExecutionTransactionId>,
    failed: bool,
}
impl HybridTensorCpuSegment {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        resources: &BoundDecoderResources,
        segment: LayerSegmentPlan,
        plan: HybridParallelPlan,
        lane: usize,
        collective: Box<dyn StandardTensorCollective>,
        max_positions: usize,
        max_parameter_bytes: u64,
    ) -> Result<Self> {
        plan.validate_spec(resources.spec())?;
        if !matches!(plan.tensor_members().len(), 1 | 2 | 4) {
            return Err(invalid("hybrid CPU reference admits TP1/2/4"));
        }
        if lane >= plan.tensor_members().len()
            || collective.members() != plan.tensor_members()
            || collective.owner() != plan.tensor_members()[lane]
            || segment.layers() != plan.schema().layer_range()
        {
            return Err(invalid("TP segment/owner/collective membership mismatch"));
        }
        if resources.spec().output().has_bias() {
            return Err(invalid("biased vocabulary head unsupported"));
        }
        for layer in &resources.spec().layers()[segment.layers()] {
            let projections: Vec<&Linear> = match layer.attention() {
                Attention::Gqa(g) => vec![g.query(), g.key(), g.value(), g.output()],
                Attention::GatedDeltaNet(g) => vec![g.qkv(), g.z(), g.beta(), g.a(), g.output()],
                _ => return Err(invalid("unsupported attention")),
            };
            if projections.iter().any(|p| p.has_bias()) {
                return Err(invalid("biased local attention unsupported"));
            }
            if matches!(layer.feed_forward(), FeedForward::Moe(m) if m.router().has_bias()) {
                return Err(invalid("biased router unsupported"));
            }
            let mlp = match layer.feed_forward() {
                FeedForward::SwiGlu(m) => Some(m),
                FeedForward::Moe(m) => m.shared_expert(),
            };
            if let Some(m) = mlp
                && (m.activation_limit().is_some()
                    || [m.gate(), m.up(), m.down()].iter().any(|p| p.has_bias()))
            {
                return Err(invalid("clipped/biased TP SwiGLU unsupported"));
            }
        }
        let schedule = plan.collective_schedule(&segment)?;
        let layout = Arc::new(plan.rank_layout(lane)?);
        let budget = layout.weight_budget(resources, &segment)?;
        if budget.max_parameter_bytes as u64 > max_parameter_bytes {
            return Err(invalid("owner local parameter exceeds prepare read limit"));
        }
        let materializer =
            StateDictMaterializer::for_hybrid(max_parameter_bytes, Arc::clone(&layout))?;
        let segment = HybridDecoderSegment::prepare_with_materializer(
            resources,
            segment,
            ExecutionPrecisionPolicy::f32(),
            max_positions,
            materializer,
            Some(layout.state_schema().clone()),
        )
        .map_err(|source| ferrule_common::Error::ModelSource {
            source: Box::new(source),
        })?;
        Ok(Self {
            segment,
            lifecycle: HybridTensorSequenceLifecycle { layout },
            parallel: plan,
            schedule,
            collective,
            watermark: None,
            failed: false,
        })
    }
    pub fn layout(&self) -> &HybridRankLayout {
        &self.lifecycle.layout
    }
    pub fn parallel_plan(&self) -> &HybridParallelPlan {
        &self.parallel
    }
    /// Check caller-supplied scope/source/attachment consistency. This does not
    /// authenticate the executor; runtime must verify its actual owner roster.
    pub fn bind_expert_call<'a>(
        &self,
        scope: HybridExpertScope,
        attachment: &HybridExpertAttachment,
        executor: &'a mut dyn RoutedSwiGluExecutor,
    ) -> Result<HybridExpertCallBinding<'a>> {
        if scope != self.parallel.scope()
            || Some(attachment) != self.parallel.expert_attachment()
            || attachment.source_rank != self.layout().owner()
        {
            return Err(invalid("foreign EP binding scope/source/placement"));
        }
        Ok(HybridExpertCallBinding {
            scope,
            attachment: attachment.clone(),
            executor,
        })
    }
    pub fn collective_schedule(&self) -> &HybridCollectiveSchedule {
        &self.schedule
    }
    pub fn preparation_reads(
        &self,
    ) -> Result<Vec<crate::transformer::parallel::TensorParallelPreparationRead>> {
        self.segment.preparation_reads()
    }
    pub fn checkpoint_read_counters(&self) -> crate::checkpoint::CheckpointReadCounters {
        self.segment.checkpoint_read_counters()
    }
    pub fn lifecycle(&self) -> HybridTensorSequenceLifecycle {
        self.lifecycle.clone()
    }
    pub fn plan(&self) -> &LayerSegmentPlan {
        self.segment.plan()
    }
    pub fn is_failed(&self) -> bool {
        self.failed
    }
    /// Local physical shapes with preserved global layer/compact KV coordinates.
    pub fn schema(&self) -> &HybridStateSchema {
        self.segment.schema()
    }
    pub fn validate_frontier(
        &self,
        batch: &PackedDecoderBatch,
        states: &[HybridTensorSequenceState],
    ) -> Result<()> {
        for state in states {
            self.lifecycle.validate_owner(state)?;
        }
        self.segment
            .validate_frontier(batch, states)
            .map_err(|source| ferrule_common::Error::ModelSource {
                source: Box::new(source),
            })
    }
    pub fn validate_production_execution(&self) -> Result<()> {
        Err(invalid(
            "hybrid parallel CPU reference is not production execution",
        ))
    }

    /// Rank-local work only. The caller owns checkout/physical enter and all
    /// publication/abort. For TP>1 an error or panic aborts/wakes the collective for its
    /// lifetime. TP1 has no peer cohort to poison; the existing journal must
    /// still prove rollback/quiescence before a newer transaction can execute.
    /// The attached source alone receives an EP binding. Other lanes receive its output
    /// through Sum with exact-zero contributions, never dispatching a replica.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [HybridTensorSequenceState],
        kv: &mut CpuKvView,
        sequence_ids: &[u64],
        executor: Option<HybridExpertCallBinding<'_>>,
        active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<SegmentOutput> {
        self.execute_in_transaction(
            kv.transaction(),
            batch,
            input,
            states,
            kv,
            sequence_ids,
            executor,
            active,
        )
    }

    /// Explicit transaction admission for runtime owners. A mismatched KV
    /// transaction is rejected before collective entry or any state write.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_in_transaction(
        &mut self,
        transaction: ExecutionTransactionId,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [HybridTensorSequenceState],
        kv: &mut CpuKvView,
        sequence_ids: &[u64],
        executor: Option<HybridExpertCallBinding<'_>>,
        active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<SegmentOutput> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if transaction != kv.transaction() {
                return Err(invalid("expert transaction differs from entered KV"));
            }
            if self.failed || self.watermark.is_some_and(|last| last >= transaction) {
                return Err(invalid("failed/stale hybrid TP execution"));
            }
            self.watermark = Some(transaction);
            active(transaction)?;
            let plan = &self.parallel;
            let layout = &self.lifecycle.layout;
            let lane = layout.lane();
            let source = plan.expert_attachment().map(|ep| ep.source_rank);
            if let Some(binding) = &executor
                && (binding.scope != plan.scope()
                    || Some(&binding.attachment) != plan.expert_attachment())
            {
                return Err(invalid("foreign EP binding scope/placement"));
            }
            if executor.is_some() != (source == Some(layout.owner()))
                || sequence_ids.len() != batch.sequences().len()
            {
                return Err(invalid("TP EP executor/packed sequence identity mismatch"));
            }
            for state in states.iter() {
                self.lifecycle.validate_owner(state)?;
            }
            self.segment
                .validate_execution(batch, states)
                .map_err(|source| ferrule_common::Error::ModelSource {
                    source: Box::new(source),
                })?;
            kv.validate_batch(batch)?;
            // Validate hidden shape/ownership before allocating or entering agreement.
            // Owner-local execution generations remain checked above, never serialized.
            let identity = self
                .schedule
                .encode_agreement(batch, sequence_ids, &input)?;
            let channel = RefCell::new(Channel {
                collective: &mut *self.collective,
                transaction,
                schedule: &self.schedule,
                next: 0,
                rows: batch.len(),
                sequences: sequence_ids.len(),
            });
            let gathered = channel.borrow_mut().exchange(
                HybridCollectiveSemantic::Agreement,
                None,
                TensorParallelCollective::AllGather,
                identity.clone(),
            )?;
            if gathered
                .chunks_exact(identity.len())
                .any(|part| part != identity)
            {
                return Err(invalid("TP lanes disagree on packed input identity"));
            }
            let mut ops = operators::TensorOperators {
                cpu: CpuStandardDecoderOperators::default(),
                layout,
                channel: &channel,
            };
            let mut ep = TensorExperts {
                executor: executor.map(|b| b.executor),
                plan,
                lane,
                channel: &channel,
            };
            let output = self
                .segment
                .execute_bound_inner(
                    batch,
                    input.into(),
                    states,
                    kv,
                    LayerOperators::Standard(&mut ops),
                    active,
                    source.map(|source| {
                        (
                            (transaction, source, sequence_ids),
                            &mut ep as &mut dyn RoutedSwiGluExecutor,
                        )
                    }),
                )
                .map_err(|source| ferrule_common::Error::ModelSource {
                    source: Box::new(source),
                })?;
            let output = output.into_host(&mut ops).map_err(|source| {
                ferrule_common::Error::ModelSource {
                    source: Box::new(source),
                }
            })?;
            active(transaction)?;
            // Every peer has passed its recurrent frontier check before returning.
            channel.borrow_mut().exchange(
                HybridCollectiveSemantic::Completion,
                None,
                TensorParallelCollective::Sum,
                vec![1.],
            )?;
            channel.borrow().finish()?;
            Ok(output)
        }));
        match result {
            Ok(result) => {
                if result.is_err() && self.layout().members().len() > 1 {
                    self.failed = true;
                    // A broken transport abort must not replace the model error.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.collective.abort()
                    }));
                }
                result
            }
            Err(panic) => {
                // TP1 has no peer cohort to poison. For TP>1, fail closed before
                // restoring the original unwind; custody remains the caller's.
                if self.layout().members().len() > 1 {
                    self.failed = true;
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.collective.abort()
                    }));
                }
                std::panic::resume_unwind(panic)
            }
        }
    }
}
struct Channel<'a> {
    collective: &'a mut dyn StandardTensorCollective,
    transaction: ExecutionTransactionId,
    schedule: &'a HybridCollectiveSchedule,
    next: usize,
    rows: usize,
    sequences: usize,
}
impl Channel<'_> {
    fn exchange(
        &mut self,
        semantic: HybridCollectiveSemantic,
        layer: Option<usize>,
        kind: TensorParallelCollective,
        values: Vec<f32>,
    ) -> Result<Vec<f32>> {
        let site = self
            .schedule
            .sites()
            .get(self.next)
            .ok_or_else(|| invalid("collective outside model schedule"))?;
        let expected_kind = match site.kind {
            HybridCollectiveKind::AllGather => TensorParallelCollective::AllGather,
            HybridCollectiveKind::ReduceSum => TensorParallelCollective::Sum,
        };
        if site.semantic != semantic
            || site.global_layer != layer
            || expected_kind != kind
            || values.len() != site.shape.elements(self.sequences, self.rows)?
            || values.iter().any(|x| !x.is_finite())
        {
            return Err(invalid(
                "collective semantic/layer/kind/shape differs from model schedule",
            ));
        }
        self.next += 1;
        if self.schedule.wire_sites().is_empty() {
            return Ok(values);
        }
        let expected = values
            .len()
            .checked_mul(if kind == TensorParallelCollective::AllGather {
                self.schedule.degree()
            } else {
                1
            })
            .filter(|n| {
                n.checked_mul(4)
                    .is_some_and(|bytes| bytes <= isize::MAX as usize)
            })
            .ok_or_else(|| invalid("collective result size overflow"))?;
        let result = self
            .collective
            .exchange(self.transaction, site.ordinal, kind, values)?;
        if result.len() != expected || result.iter().any(|x| !x.is_finite()) {
            return Err(invalid("invalid TP collective output"));
        }
        Ok(result)
    }
    fn finish(&self) -> Result<()> {
        if self.next != self.schedule.sites().len() {
            return Err(invalid("incomplete model collective schedule"));
        }
        Ok(())
    }
}

fn rows(
    values: Vec<f32>,
    count: usize,
    width: usize,
    arena: Option<RowsArenaId>,
) -> Result<OperatorProgress<Rows>> {
    Ok(OperatorProgress::Ready(Rows::Host(HostRows::new(
        RowsShape::new(count, width)?,
        RowsDType::F32,
        arena,
        values,
    )?)))
}
struct TensorExperts<'a, 'b, 'e> {
    executor: Option<&'e mut dyn RoutedSwiGluExecutor>,
    plan: &'a HybridParallelPlan,
    lane: usize,
    channel: &'a RefCell<Channel<'b>>,
}
impl RoutedSwiGluExecutor for TensorExperts<'_, '_, '_> {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        active(request.context.transaction)?;
        if Some(request.context.source_rank)
            != self.plan.expert_attachment().map(|ep| ep.source_rank)
            || !self
                .plan
                .schema()
                .layer_range()
                .contains(&request.context.layer)
        {
            return Err(invalid("foreign TP routed source/layer"));
        }
        let layer = request.context.layer;
        let shape = request.input.shape();
        let arena = request.arena;
        let values = if self.plan.tensor_members()[self.lane] == request.context.source_rank {
            let output = ready(
                self.executor
                    .as_mut()
                    .ok_or_else(|| invalid("missing leader EP executor"))?
                    .routed_swiglu(request, active)?,
            )?
            .into_host()?;
            if output.shape() != shape || output.dtype() != RowsDType::F32 {
                return Err(invalid("EP output layout mismatch"));
            }
            output.into_values()
        } else {
            vec![0.; shape.rows() * shape.width()]
        };
        let values = self.channel.borrow_mut().exchange(
            HybridCollectiveSemantic::RoutedExpertOutput,
            Some(layer),
            TensorParallelCollective::Sum,
            values,
        )?;
        rows(values, shape.rows(), shape.width(), arena)
    }
}
