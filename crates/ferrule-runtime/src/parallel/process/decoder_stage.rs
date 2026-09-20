//! Child-local stage construction. Layer math and routed result combination
//! remain in model; physical KV remains in PipelineStageWorker's one journal.

use std::cell::RefCell;
use std::rc::Rc;

use ferrule_common::{Error, Result};
use ferrule_model::decoder::{
    CpuKvView, CpuPagedKvPool, PackedDecoderBatch, PagedKvBackend, StandardGqaPlanes,
};
use ferrule_model::transformer::expert_parallel::ExpertParallelRoutedExecutor;
use ferrule_model::transformer::{
    Attention, LayerSegmentPlan, SegmentInput, SegmentOutput, StandardDecoderSegment,
};

#[cfg(feature = "cuda")]
use super::decoder::Retained;
use super::decoder::{DecoderBoot, DecoderDevice, error};
use super::decoder_expert::ProcessExperts;
use crate::parallel::pipeline::{
    BoxedPipelineStageWorker, PipelineExecutionContext, PipelineStage, PipelineStageProgram,
    PipelineStageWorker,
};

pub(super) type Experts = Rc<RefCell<ProcessExperts>>;

pub(super) fn prepare(
    boot: &DecoderBoot,
    experts: Option<Experts>,
) -> Result<BoxedPipelineStageWorker> {
    let stage_boot = boot.stage_boot()?;
    let resources = boot.load_resources()?;
    if boot.device == DecoderDevice::Cpu && experts.is_none() {
        return Ok(PipelineStageWorker::new(
            &stage_boot,
            PipelineStage::prepare_cpu(&resources, stage_boot.plan.clone(), stage_boot.config)?,
        )?
        .boxed());
    }
    #[cfg(feature = "cuda")]
    if let DecoderDevice::Cuda { ordinal } = boot.device {
        let ops = Retained::new(Rc::new(
            ferrule_backend::cuda::operators::linear::CudaOperators::new_on_device(ordinal)?,
        ));
        if experts.is_none() {
            let stage = PipelineStage::prepare_cuda(
                &resources,
                stage_boot.plan.clone(),
                stage_boot.config,
                Rc::clone(&ops),
            )?;
            let worker = PipelineStageWorker::new(&stage_boot, stage)?.boxed();
            drop(ops.release());
            return Ok(worker);
        }
        let (description, planes) = description(boot, &resources)?;
        let capabilities = description.execution_capabilities()?;
        let segment = StandardDecoderSegment::prepare_cuda(
            &resources,
            stage_boot.plan.clone(),
            stage_boot.config.precision,
            stage_boot.config.max_positions,
            stage_boot.config.max_parameter_bytes,
            Rc::clone(&ops),
        )
        .map_err(segment_error)?;
        let mut program = Retained::new(CudaProgram {
            segment,
            staging: ferrule_model::transformer::CudaStandardDecoderOperators::new(
                Rc::clone(&ops),
                stage_boot.config.precision,
                &[],
            )?,
            experts: experts.expect("checked EP"),
        });
        let backend = match ferrule_model::decoder::CudaPagedKvPool::from_strategy(
            Rc::clone(&ops),
            &planes,
            stage_boot.config.max_pages,
        ) {
            Ok(pool) => PagedKvBackend::new(pool),
            Err(e) => {
                program.shutdown()?;
                return Err(e);
            }
        };
        let stage = PipelineStage::new(program.release(), backend, description, capabilities)?;
        let worker = PipelineStageWorker::new(&stage_boot, stage)?.boxed();
        drop(ops.release());
        return Ok(worker);
    }
    if boot.device != DecoderDevice::Cpu {
        return Err(error(
            "CUDA decoder requires --features cuda; no CPU fallback",
        ));
    }
    let (description, planes) = description(boot, &resources)?;
    let capabilities = description.execution_capabilities()?;
    let segment = StandardDecoderSegment::prepare(
        &resources,
        stage_boot.plan.clone(),
        stage_boot.config.precision,
        stage_boot.config.max_positions,
        stage_boot.config.max_parameter_bytes,
    )
    .map_err(segment_error)?;
    let backend = PagedKvBackend::new(CpuPagedKvPool::from_strategy(
        &planes,
        stage_boot.config.max_pages,
    )?);
    let stage = PipelineStage::new(
        CpuProgram {
            segment,
            experts: experts.expect("CPU EP"),
        },
        backend,
        description,
        capabilities,
    )?;
    Ok(PipelineStageWorker::new(&stage_boot, stage)?.boxed())
}
fn description(
    boot: &DecoderBoot,
    resources: &ferrule_model::transformer::BoundDecoderResources,
) -> Result<(
    crate::parallel::pipeline::PipelineStageDescription,
    StandardGqaPlanes,
)> {
    let plan = boot.segment.decode()?;
    if plan.total_layers() != resources.spec().layers().len() {
        return Err(error("checkpoint/segment layer count mismatch"));
    }
    let mut geometry = None;
    for layer in &resources.spec().layers()[plan.layers()] {
        let Attention::Gqa(gqa) = layer.attention() else {
            return Err(error("process stage requires GQA"));
        };
        let current = (gqa.num_kv_heads(), gqa.head_dim());
        if geometry.is_some_and(|previous| previous != current) {
            return Err(error("nonuniform stage KV geometry"));
        }
        geometry = Some(current);
    }
    let (heads, dim) = geometry.ok_or_else(|| error("empty stage"))?;
    let description = boot.description(
        resources.spec().hidden_size(),
        resources.spec().vocab_size(),
        heads,
        dim,
    )?;
    if let Some(group) = &description.expert_group {
        group.validate_resources(resources)?;
    }
    let planes = StandardGqaPlanes::new(
        plan.layer_count(),
        heads,
        dim,
        description.config.page_size,
        description.config.max_positions,
        description.config.dtype(),
    )?;
    Ok((description, planes))
}
fn segment_error(e: ferrule_model::transformer::SegmentError) -> Error {
    Error::ModelSource {
        source: Box::new(e),
    }
}

struct CpuProgram {
    segment: StandardDecoderSegment,
    experts: Experts,
}
impl PipelineStageProgram for CpuProgram {
    type KvView = CpuKvView;
    fn plan(&self) -> &LayerSegmentPlan {
        self.segment.plan()
    }
    fn execute(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        view: &mut CpuKvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        let mut experts = self.experts.borrow_mut();
        let group = experts.group.clone();
        self.segment
            .execute_with_experts(
                batch,
                input,
                view,
                (context.transaction, group.source_rank, context.sequence_ids),
                &mut ExpertParallelRoutedExecutor::new(
                    group.members,
                    &group.placement,
                    group.limits,
                    &mut *experts,
                ),
                context.check_active,
            )
            .map_err(segment_error)
    }
    fn shutdown(&mut self) -> Result<()> {
        self.experts.borrow_mut().shutdown()
    }
}

#[cfg(feature = "cuda")]
struct CudaProgram {
    segment: ferrule_model::transformer::CudaStandardDecoderSegment,
    staging: ferrule_model::transformer::CudaStandardDecoderOperators,
    experts: Experts,
}
#[cfg(feature = "cuda")]
impl CudaProgram {
    fn checked<T>(&self, result: Result<T>) -> Result<T> {
        if self.segment.needs_quarantine() || self.staging.needs_quarantine() {
            std::mem::forget(result);
            std::panic::panic_any(crate::parallel::data::PanicQuiescence::Unknown);
        }
        result
    }
}
#[cfg(feature = "cuda")]
impl PipelineStageProgram for CudaProgram {
    type KvView = ferrule_model::decoder::CudaKvView;
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
        let result = {
            let mut experts = self.experts.borrow_mut();
            let group = experts.group.clone();
            let mut routed = ferrule_model::transformer::CudaExpertParallelRoutedExecutor::new(
                &mut self.staging,
                group.members,
                &group.placement,
                group.limits,
                &mut *experts,
            );
            self.segment
                .execute_cuda_bound_with_experts(
                    batch,
                    input.into(),
                    view,
                    (context.transaction, group.source_rank, context.sequence_ids),
                    &mut routed,
                    context.check_active,
                )
                .and_then(|output| self.segment.download_output(output))
                .map_err(segment_error)
        };
        self.checked(result)
    }
    fn shutdown(&mut self) -> Result<()> {
        let experts = self.experts.borrow_mut().shutdown();
        let staging = self.staging.quiesce();
        let segment = self.segment.quiesce();
        self.checked(Error::failures(
            "process CUDA stage shutdown",
            experts
                .err()
                .into_iter()
                .chain(staging.err())
                .chain(segment.err())
                .collect(),
        ))
    }
}
