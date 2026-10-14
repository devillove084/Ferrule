//! Owner-local dense/routed hybrid CUDA adapter; custody remains in the PP journal.
use crate::parallel::pipeline::PhysicalKvLayout;
use std::rc::Rc;

use super::completion_boundary::checked_cuda_completion;
use crate::parallel::expert::ExpertGroup;
use ferrule_common::{Error, Result};
use ferrule_model::decoder::{
    CudaHybridSequenceLifecycle, CudaHybridSequenceState, CudaKvView, HybridCudaDevice,
    HybridLayerSchema, HybridStateSchema, PackedDecoderBatch, PagedKvBackend, TypedCudaPagedKvPool,
};
use ferrule_model::decoder::{GenericDecoderOptions, HybridCudaRoutedExperts};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::{
    BoundDecoderResources, CudaHybridDecoderSegment, LayerSegmentPlan, SegmentInput, SegmentOutput,
};

use super::{
    PipelineConfig, PipelineExecutionContext, PipelineStage, PipelineStageDescription,
    PipelineStateProgram, error, stage::segment_error,
};
use crate::parallel::data::PanicQuiescence;

pub struct HybridCudaPipelineStageProgram {
    segment: CudaHybridDecoderSegment,
    device: HybridCudaDevice,
    group: Option<ExpertGroup>,
}

fn checked<T>(device: &HybridCudaDevice, result: Result<T>) -> Result<T> {
    checked_cuda_completion(device.needs_quarantine(), result)
}
// Before a worker exists there is no panic hook to retain its model custody.
// Model/allocator unwind guards retain unsafe buffers; preserve that Unknown
// evidence at the runtime owner boundary instead of reporting plain Panicked.
fn cuda_call<T>(call: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)) {
        Ok(value) => value,
        Err(payload) => {
            std::mem::forget(payload);
            std::panic::panic_any(PanicQuiescence::Unknown);
        }
    }
}
impl HybridCudaPipelineStageProgram {
    fn completion<T>(&self, result: Result<T>) -> Result<T> {
        checked_cuda_completion(
            self.segment.needs_quarantine() || self.device.needs_quarantine(),
            result,
        )
    }
}
impl PipelineStateProgram for HybridCudaPipelineStageProgram {
    type SequenceState = CudaHybridSequenceState;
    type Lifecycle = CudaHybridSequenceLifecycle;
    type KvView = CudaKvView;

    fn plan(&self) -> &LayerSegmentPlan {
        self.segment.plan()
    }
    fn hybrid_schema(&self) -> Option<&HybridStateSchema> {
        Some(self.segment.schema())
    }
    fn hybrid_expert_group(&self) -> Option<&ExpertGroup> {
        self.group.as_ref()
    }
    fn poll_quiescence(&mut self) -> Result<bool> {
        let result = cuda_call(|| self.segment.poll_quiescence());
        self.completion(result)
    }
    fn new_lifecycle(&mut self) -> Result<Self::Lifecycle> {
        self.completion(CudaHybridSequenceLifecycle::new(
            self.segment.schema().clone(),
            self.device.clone(),
        ))
    }
    fn execute_state(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        states: &mut [Self::SequenceState],
        view: &mut CudaKvView,
        context: PipelineExecutionContext<'_>,
    ) -> Result<SegmentOutput> {
        if context.experts.is_some()
            || context.transaction != view.transaction()
            || context.sequence_ids.len() != batch.sequences().len()
        {
            return Err(error(
                "hybrid CUDA execution identity mismatch or unbound expert injection",
            ));
        }
        (context.check_active)(context.transaction)?;
        let result = cuda_call(|| {
            if let Some(group) = &self.group {
                self.segment.execute_with_routed_experts(
                    batch,
                    input,
                    states,
                    view,
                    (context.transaction, group.source_rank, context.sequence_ids),
                    context.check_active,
                )
            } else {
                self.segment
                    .execute_checked(batch, input, states, view, context.check_active)
            }
        });
        self.completion(result.map_err(segment_error))
    }
    fn validate_frontier(
        &self,
        batch: &PackedDecoderBatch,
        states: &[Self::SequenceState],
    ) -> Result<()> {
        self.completion(
            self.segment
                .validate_frontier(batch, states)
                .map_err(segment_error),
        )
    }
    fn shutdown(&mut self) -> Result<()> {
        // The thread worker has a synchronous shutdown boundary, not a second
        // completion journal. Poll model custody to proof within the existing
        // thread expert deadline; timeout is Unknown, never an ordinary ACK.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let result = cuda_call(|| self.segment.poll_shutdown());
            if self.completion(result)? {
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                return checked_cuda_completion(
                    true,
                    Err(error("hybrid root/expert shutdown completion unknown")),
                );
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }
}
impl
    PipelineStage<
        PagedKvBackend<TypedCudaPagedKvPool<CudaHybridSequenceState>>,
        HybridCudaPipelineStageProgram,
    >
{
    /// Construct on the persistent owner, sharing one context for model, state and KV.
    pub fn prepare_hybrid_cuda(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        config: PipelineConfig,
        device: HybridCudaDevice,
    ) -> Result<Self> {
        Self::prepare_inner(resources, plan, config, device, None)
    }

    /// Consume an already-ready, admitted attachment on its root owner.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_hybrid_cuda_with_routed_experts(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        config: PipelineConfig,
        device: HybridCudaDevice,
        options: &GenericDecoderOptions,
        group: ExpertGroup,
        experts: HybridCudaRoutedExperts,
    ) -> Result<Self> {
        Self::prepare_inner(
            resources,
            plan,
            config,
            device,
            Some((options, group, experts)),
        )
    }

    fn prepare_inner(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        config: PipelineConfig,
        device: HybridCudaDevice,
        routed: Option<(&GenericDecoderOptions, ExpertGroup, HybridCudaRoutedExperts)>,
    ) -> Result<Self> {
        config.validate()?;
        if config.precision != ExecutionPrecisionPolicy::f32() {
            return Err(error("hybrid CUDA requires F32"));
        }
        if let Some((options, group, _)) = &routed {
            group.validate_resources(resources)?;
            if group.layers != plan.layers()
                || options.page_size() != config.page_size
                || options.max_positions() != config.max_positions
                || options.max_parameter_bytes() != config.max_parameter_bytes
                || options.capabilities().max_batch_tokens != config.max_batch_tokens
                || options.capabilities().max_sequences != config.session_capacity
                || options.precision() != config.precision
            {
                return Err(error(
                    "hybrid CUDA attachment/options differ from stage admission",
                ));
            }
        }
        let schema = HybridStateSchema::for_segment(resources.spec(), plan.layers())?;
        let planes = schema.kv_planes(config.page_size, config.max_positions)?;
        let (kv_heads, head_dim) = schema
            .layers()
            .iter()
            .find_map(|layer| match layer {
                HybridLayerSchema::FullAttention {
                    kv_heads, head_dim, ..
                } => Some((*kv_heads, *head_dim)),
                _ => None,
            })
            .ok_or_else(|| error("hybrid stage has no full attention"))?;
        let physical_pages = config.physical_pages(PhysicalKvLayout::hybrid(
            ferrule_model::ModelExecutionBackend::Cuda,
        ))?;
        if device.budget().kv_pages != physical_pages {
            return Err(error("hybrid CUDA physical KV budget mismatch"));
        }
        let description = PipelineStageDescription {
            plan: plan.clone(),
            physical_pages,
            config,
            hidden: resources.spec().hidden_size(),
            vocabulary: resources.spec().vocab_size(),
            kv_heads,
            head_dim,
            expert_group: routed.as_ref().map(|(_, group, _)| group.clone()),
        };
        let capabilities = description.execution_capabilities()?;
        let mut segment = checked(
            &device,
            cuda_call(|| {
                if let Some((options, _, experts)) = routed {
                    CudaHybridDecoderSegment::prepare_with_routed_experts(
                        resources,
                        plan,
                        options,
                        device.clone(),
                        experts,
                    )
                } else {
                    CudaHybridDecoderSegment::prepare(
                        resources,
                        plan,
                        config.precision,
                        config.max_positions,
                        config.max_parameter_bytes,
                        device.clone(),
                    )
                }
            })
            .map_err(segment_error),
        )?;
        if segment.routed_source_rank() != description.expert_group.as_ref().map(|g| g.source_rank)
        {
            let cleanup = cuda_call(|| segment.shutdown());
            return checked(
                &device,
                Err(Error::with_cleanup(
                    "hybrid CUDA source binding",
                    error("attachment source differs from admitted root"),
                    cleanup,
                )),
            );
        }
        let mut program = HybridCudaPipelineStageProgram {
            segment,
            device: device.clone(),
            group: description.expert_group.clone(),
        };
        let pool = match cuda_call(|| {
            TypedCudaPagedKvPool::from_strategy(
                Rc::clone(device.operators()),
                &planes,
                physical_pages,
            )
        }) {
            Ok(pool) => pool,
            Err(primary) => {
                let cleanup = program.shutdown();
                return checked(
                    &device,
                    Err(Error::with_cleanup(
                        "hybrid CUDA KV startup",
                        primary,
                        cleanup,
                    )),
                );
            }
        };
        Self::new(
            program,
            PagedKvBackend::new(pool),
            description,
            capabilities,
        )
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/parallel/hybrid_cuda.rs"]
mod tests;
