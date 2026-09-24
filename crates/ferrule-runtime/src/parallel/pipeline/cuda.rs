//! F32 CUDA programs over the existing PP owner journal and KV cohort.
//! Host staging is confined to PP/logits and explicit expert-result boundaries.

use std::rc::Rc;

use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Error, Result};
use ferrule_model::decoder::{
    CudaKvView, CudaPagedKvBackend, CudaPagedKvPool, PackedDecoderBatch, PagedKvBackend,
    StandardGqaPlanes,
};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::expert_parallel::{RoutedSwiGluExecutor, RoutedSwiGluRequest};
use ferrule_model::transformer::parallel::CudaShardError;
use ferrule_model::transformer::{
    Attention, BoundDecoderResources, CudaStandardDecoderOperators, CudaStandardDecoderSegment,
    LayerSegmentPlan, OperatorProgress, Rows, SegmentInput, SegmentOutput, StandardDecoderSegment,
};

use super::stage::segment_error;
use super::{
    PipelineConfig, PipelineExecutionContext, PipelineStage, PipelineStageDescription,
    PipelineStageProgram, error,
};
use crate::parallel::data::PanicQuiescence;
use crate::parallel::expert::{ExpertOwnerStats, ExpertParallelExecutor};

/// Construct on the stage owner. Neither the segment, operators nor KV view is
/// Send; only the existing pipeline's host commands/results leave that owner.
pub struct CudaPipelineStageProgram {
    segment: CudaStandardDecoderSegment,
    experts: Option<(ExpertParallelExecutor, CudaStandardDecoderOperators)>,
}

impl CudaPipelineStageProgram {
    pub fn new(segment: CudaStandardDecoderSegment) -> Self {
        Self {
            segment,
            experts: None,
        }
    }

    fn check_completion<T>(&self, result: Result<T>) -> Result<T> {
        if self.segment.needs_quarantine()
            || self
                .experts
                .as_ref()
                .is_some_and(|(_, ops)| ops.needs_quarantine())
        {
            std::mem::forget(result);
            std::panic::panic_any(PanicQuiescence::Unknown);
        }
        checked_completion(result)
    }
}

// Startup errors may be nested inside SegmentError, before there is a program
// for the owner to retain. Preserve the model's typed completion evidence.
fn checked_completion<T>(result: Result<T>) -> Result<T> {
    fn unknown(source: &(dyn std::error::Error + 'static)) -> bool {
        source
            .downcast_ref::<CudaShardError>()
            .is_some_and(CudaShardError::needs_quarantine)
            || source.source().is_some_and(unknown)
    }
    if result.as_ref().err().is_some_and(|e| unknown(e)) {
        std::mem::forget(result);
        std::panic::panic_any(PanicQuiescence::Unknown);
    }
    result
}

struct CudaRouted<'a> {
    experts: &'a mut ExpertParallelExecutor,
    operators: &'a mut CudaStandardDecoderOperators,
}
impl RoutedSwiGluExecutor for CudaRouted<'_> {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        self.experts
            .routed_swiglu_cuda(self.operators, request, check_active)
    }
}

impl PipelineStageProgram for CudaPipelineStageProgram {
    type KvView = CudaKvView;

    fn plan(&self) -> &LayerSegmentPlan {
        self.segment.plan()
    }

    fn execute(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        view: &mut CudaKvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        let PipelineExecutionContext {
            transaction,
            sequence_ids,
            experts,
            check_active,
        } = context;
        check_active(transaction)?;
        if transaction != view.transaction() || sequence_ids.len() != batch.sequences().len() {
            return Err(error("CUDA pipeline execution/KV identity mismatch"));
        }
        // A generic routed executor has no device guarantee. Do not silently
        // wrap it in the host-combine adapter or fall back to local CPU experts.
        if experts.is_some() {
            return Err(error(
                "CUDA pipeline requires owner-local CUDA expert injection",
            ));
        }
        let result = if let Some((experts, operators)) = self.experts.as_mut() {
            let source_rank = experts.group().source_rank;
            let output = self.segment.execute_cuda_bound_with_experts(
                batch,
                input.into(),
                view,
                (transaction, source_rank, sequence_ids),
                &mut CudaRouted { experts, operators },
                check_active,
            );
            output.and_then(|output| self.segment.download_output(output))
        } else {
            self.segment.execute(batch, input, view)
        };
        let output = self.check_completion(result.map_err(segment_error))?;
        check_active(transaction)?;
        Ok(output)
    }

    fn expert_outstanding(&self) -> usize {
        self.experts
            .as_ref()
            .map_or(0, |(experts, _)| experts.outstanding())
    }

    fn owner_stats(&mut self) -> Result<Vec<ExpertOwnerStats>> {
        self.experts
            .as_mut()
            .map(|(experts, _)| experts.owner_stats())
            .transpose()
            .map(Option::unwrap_or_default)
    }

    fn shutdown(&mut self) -> Result<()> {
        let mut failures = Vec::new();
        if let Some((experts, operators)) = self.experts.as_mut() {
            failures.extend(experts.shutdown().err());
            failures.extend(operators.quiesce().err());
        }
        failures.extend(self.segment.quiesce().err());
        self.check_completion(Error::failures("CUDA pipeline program shutdown", failures))
    }
}

impl PipelineStage<CudaPagedKvBackend, CudaPipelineStageProgram> {
    /// The caller creates operators INSIDE the existing stage owner factory.
    /// The exact same owner backs layer computation and physical F32 KV planes.
    pub fn prepare_cuda(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        config: PipelineConfig,
        ops: Rc<CudaOperators>,
    ) -> Result<Self> {
        config.validate()?;
        if config.precision != ExecutionPrecisionPolicy::f32() {
            return Err(error(
                "CUDA pipeline requires F32 activations/KV; BF16 compatibility and CPU fallback are disabled",
            ));
        }
        if plan.total_layers() != resources.spec().layers().len() {
            return Err(error("CUDA pipeline layer count mismatch"));
        }
        let mut geometry = None;
        for layer in &resources.spec().layers()[plan.layers()] {
            let Attention::Gqa(gqa) = layer.attention() else {
                return Err(error("CUDA pipeline requires standard GQA layers"));
            };
            let current = (gqa.num_kv_heads(), gqa.head_dim());
            if geometry.is_some_and(|previous| previous != current) {
                return Err(error("CUDA pipeline requires uniform GQA KV geometry"));
            }
            geometry = Some(current);
        }
        let (kv_heads, head_dim) = geometry.ok_or_else(|| error("empty CUDA stage"))?;
        let description = PipelineStageDescription {
            plan: plan.clone(),
            config,
            hidden: resources.spec().hidden_size(),
            vocabulary: resources.spec().vocab_size(),
            kv_heads,
            head_dim,
            expert_group: None,
        };
        let capabilities = description.execution_capabilities()?;
        let planes = StandardGqaPlanes::new(
            plan.layer_count(),
            kv_heads,
            head_dim,
            config.page_size,
            config.max_positions,
            config.dtype(),
        )?;
        let segment = checked_completion(
            StandardDecoderSegment::prepare_cuda(
                resources,
                plan,
                config.precision,
                config.max_positions,
                config.max_parameter_bytes,
                Rc::clone(&ops),
            )
            .map_err(segment_error),
        )?;
        let mut program = CudaPipelineStageProgram::new(segment);
        let backend = match CudaPagedKvPool::from_strategy(ops, &planes, config.max_pages) {
            Ok(pool) => PagedKvBackend::new(pool),
            Err(source) => {
                return checked_completion(Err(Error::with_cleanup(
                    "CUDA pipeline KV startup",
                    source,
                    program.shutdown(),
                )));
            }
        };
        Self::new(program, backend, description, capabilities)
    }

    /// Dense rank-local standard decoder using the same PP worker journal.
    /// `rank` is TP-local; the collective/plan carries the physical global owner.
    /// The description and backend both use local heads and segment-local layers,
    /// while the parent alone constructs the full logical PP×TP schema.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_cuda_tensor(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        tensor: ferrule_model::transformer::StandardTensorPlan,
        rank: ferrule_common::ParallelRankId,
        config: PipelineConfig,
        ops: Rc<CudaOperators>,
        collective: Box<dyn ferrule_model::transformer::StandardTensorCollective>,
    ) -> Result<Self> {
        config.validate()?;
        if config.precision != ExecutionPrecisionPolicy::f32() {
            return Err(error(
                "standard tensor pipeline requires F32 activations and KV",
            ));
        }
        tensor.validate_segment(resources.spec(), &plan)?;
        let description = PipelineStageDescription {
            plan: plan.clone(),
            config,
            hidden: resources.spec().hidden_size(),
            vocabulary: resources.spec().vocab_size(),
            kv_heads: tensor.local_kv_heads(),
            head_dim: tensor.head_dim(),
            expert_group: None,
        };
        let capabilities = description.execution_capabilities()?;
        let planes = tensor.kv_planes_for_segment(&plan, config.page_size, config.max_positions)?;
        let segment = checked_completion(
            StandardDecoderSegment::prepare_cuda_tensor(
                resources,
                plan,
                tensor,
                rank,
                config.max_positions,
                config.max_parameter_bytes,
                Rc::clone(&ops),
                collective,
            )
            .map_err(segment_error),
        )?;
        let mut program = CudaPipelineStageProgram::new(segment);
        let backend = match CudaPagedKvPool::from_strategy(ops, &planes, config.max_pages) {
            Ok(pool) => PagedKvBackend::new(pool),
            Err(source) => {
                return checked_completion(Err(Error::with_cleanup(
                    "tensor pipeline KV startup",
                    source,
                    program.shutdown(),
                )));
            }
        };
        Self::new(program, backend, description, capabilities)
    }

    pub fn prepare_cuda_with_experts(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        config: PipelineConfig,
        ops: Rc<CudaOperators>,
        mut experts: ExpertParallelExecutor,
    ) -> Result<Self> {
        let prepared = (|| {
            if !experts.is_cuda() {
                return Err(error(
                    "CUDA pipeline requires CUDA expert owners; CPU fallback is disabled",
                ));
            }
            if experts.group().layers != plan.layers() {
                return Err(error("expert group must cover exactly the segment layers"));
            }
            experts.group().validate_resources(resources)?;
            let staging =
                CudaStandardDecoderOperators::new(Rc::clone(&ops), config.precision, &[])?;
            let stage = Self::prepare_cuda(resources, plan, config, ops)?;
            Ok((stage, staging))
        })();
        match prepared {
            Ok((mut stage, staging)) => {
                stage.description.expert_group = Some(experts.group().clone());
                stage.program.experts = Some((experts, staging));
                Ok(stage)
            }
            Err(source) => checked_completion(Err(Error::with_cleanup(
                "CUDA pipeline expert startup",
                source,
                experts.shutdown(),
            ))),
        }
    }
}
