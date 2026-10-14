//! Dense hybrid provider admission and stage/transport construction.
use super::*;
use ferrule_runtime::parallel::pipeline::PhysicalKvLayout;
#[cfg(unix)]
#[path = "hybrid_process.rs"]
mod process;
#[cfg(test)]
use ferrule_model::TensorRole;
use ferrule_model::decoder::{GenericDecoderOptions, HybridStateSchema};
use ferrule_model::transformer::HybridSegmentMemoryEstimate;
use ferrule_runtime::parallel::pipeline::{
    HybridDeviceBudget, HybridPipelineLayout, HybridStageBudget,
};

const PAGE_SIZE: usize = 16;
const MARGIN: u64 = 64 * 1024 * 1024;

fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("hybrid startup byte overflow"))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("hybrid startup byte overflow"))
}

/// All values are metadata-derived, before any selected payload is materialized.
struct Admission {
    topology: ValidatedParallelTopology,
    plans: Vec<LayerSegmentPlan>,
    layout: HybridPipelineLayout,
    config: PipelineConfig,
    kv: ferrule_runtime::engine::ResidentKvPagePlan,
    stages: Vec<HybridStageBudget>,
    host_required: u64,
    #[cfg(unix)]
    process: Option<process::Plan>,
}
impl Admission {
    fn new(request: &ResolvedModelRequest, resources: &BoundDecoderResources) -> Result<Self> {
        let options = request
            .pipeline
            .as_ref()
            .ok_or_else(|| invalid("missing hybrid pipeline options"))?;
        options.hybrid_precision(&request.family, request.backend)?;
        let spec = resources.spec();
        if request.max_layers != spec.layers().len()
            || spec
                .max_sequence_length()
                .is_some_and(|n| request.driver_config.ctx_size > n)
            || !spec.attachments().is_empty()
            || resources.host_experts().is_some()
        {
            return Err(invalid(
                "hybrid requires full-depth text-only resources within checkpoint context",
            ));
        }
        let geometry = super::super::parallel::plan::Geometry::new(request.max_layers, options)?;
        let topology = geometry.topology;
        let plans = geometry.segments;
        let layout = HybridPipelineLayout::new(spec, &plans)?;
        let schema = HybridStateSchema::from_spec(spec, request.max_layers)?;
        let planes = schema.kv_planes(PAGE_SIZE, request.driver_config.ctx_size)?;
        // Both current hybrid adapters keep shadow/COW slots. `true` here means
        // 2L custody, NOT the CPU standard pipeline's historical L allocation.
        let kv = ferrule_runtime::engine::plan_pipeline_kv_pages(
            &planes,
            physical_page_bytes(&planes)?,
            request.kv_cache_bytes,
            PhysicalKvLayout::hybrid(request.backend),
            request.scheduler_config,
            request.driver_config,
        )?;
        let config = PipelineConfig {
            page_size: PAGE_SIZE,
            max_pages: kv.full_capacity_pages,
            max_positions: request.driver_config.ctx_size,
            max_batch_tokens: request.scheduler_config.max_batch_tokens,
            session_capacity: request.scheduler_config.max_active_sequences,
            max_parameter_bytes: request.max_tensor_bytes,
            precision: ExecutionPrecisionPolicy::f32(),
            max_ack_polls: 8,
        };
        config.validate()?;
        if config.physical_pages(PhysicalKvLayout::hybrid(request.backend))? != kv.configured_pages
        {
            return Err(invalid("hybrid physical KV capacity mismatch"));
        }
        let execution = GenericDecoderOptions::standard_cpu(
            spec,
            ModelFamily::Qwen35,
            WeightSource::Safetensors,
            config.page_size,
            config.max_positions,
            config.max_batch_tokens,
            config.session_capacity,
            config.max_parameter_bytes,
            config.precision,
        )?;
        // One model ledger per owner, reused for both resident domains. In
        // particular, never derive host residency from canonical device weights.
        let estimates = plans
            .iter()
            .map(|plan| match request.backend {
                ModelExecutionBackend::Cpu => {
                    HybridSegmentMemoryEstimate::for_segment(resources, &execution, plan)
                }
                #[cfg(feature = "cuda")]
                ModelExecutionBackend::Cuda => {
                    HybridSegmentMemoryEstimate::for_cuda_segment(resources, &execution, plan)
                }
                #[cfg(not(feature = "cuda"))]
                ModelExecutionBackend::Cuda => Err(ferrule_common::Error::Execution {
                    message: "hybrid CUDA requires the cuda feature".into(),
                }),
            })
            .collect::<ferrule_common::Result<Vec<_>>>()?;
        // Conservative parent plus two simultaneous owner catalogs. Model owns
        // per-catalog geometry; runtime owns the number of live copies.
        let metadata = mul(
            HybridSegmentMemoryEstimate::binding_metadata_bytes(resources)? as u64,
            3,
        )?;
        let rows = estimates
            .iter()
            .map(|e| e.boundary_rows_bytes)
            .max()
            .unwrap_or(0) as u64;
        let custody_metadata = add(
            mul(kv.configured_pages as u64, 3 * 4096)?,
            mul(
                config.session_capacity as u64,
                add(8192, mul(request.max_layers as u64, 256)?)?,
            )?,
        )?;
        let mut host_required = add(
            add(add(metadata, custody_metadata)?, mul(rows, 8)?)?,
            MARGIN,
        )?;
        let mut stages = Vec::new();
        for (stage, estimate) in estimates.iter().enumerate() {
            if estimate.schema() != &layout.schemas()[stage] {
                return Err(invalid("hybrid model/runtime stage schema mismatch"));
            }
            let device = if request.backend == ModelExecutionBackend::Cpu {
                0
            } else {
                options.device(stage)
            };
            let budget = HybridStageBudget::from_segment_estimate(
                config,
                device,
                estimate,
                request.backend,
                MARGIN,
            )?;
            // The CUDA segment retains its prepared host bindings too. Include
            // simultaneous owner startup/read staging, not just device weights.
            let host_stage = if request.backend == ModelExecutionBackend::Cpu {
                budget.total()?
            } else {
                add(
                    add(
                        estimate.host.weight_bytes()? as u64,
                        estimate.host.workspace_bytes as u64,
                    )?,
                    MARGIN,
                )?
            };
            host_required = add(
                host_required,
                add(host_stage, estimate.host_startup_temporary_bytes as u64)?,
            )?;
            stages.push(budget);
        }
        let placed = stages.iter().try_fold(0, |sum, stage| add(sum, stage.kv))?;
        if Some(placed) != kv.configured_bytes {
            return Err(invalid("hybrid placed KV byte mismatch"));
        }
        #[cfg(unix)]
        let process = if options.rank_backend == PipelineRankBackend::Process {
            let process = process::Plan::new(request, resources, &plans, config, &stages)?;
            host_required = add(host_required, process.host_bytes)?;
            Some(process)
        } else {
            None
        };
        Ok(Self {
            topology,
            plans,
            layout,
            config,
            kv,
            stages,
            host_required,
            #[cfg(unix)]
            process,
        })
    }
    fn admit_host(&self, available: u64) -> Result<()> {
        if self.host_required > available {
            return Err(invalid(format!(
                "hybrid concurrent host startup requires {} bytes, available {available}",
                self.host_required
            )));
        }
        Ok(())
    }
    fn admit_live(&self, backend: ModelExecutionBackend) -> Result<()> {
        self.admit_host(super::super::host_memory::snapshot()?.available_bytes())?;
        if backend == ModelExecutionBackend::Cpu {
            HybridDeviceBudget::admit(&self.stages, &[(0, self.host_required)])?;
        }
        #[cfg(feature = "cuda")]
        if backend == ModelExecutionBackend::Cuda {
            super::super::parallel::budget::admit_devices(&self.stages)?;
        }
        Ok(())
    }
}

pub(in crate::model_factory) fn build_bound(
    request: ResolvedModelRequest,
    resources: BoundDecoderResources,
    tokenizer: TokenizerHandle,
) -> Result<PipelineInferenceEngine> {
    let admission = Admission::new(&request, &resources)?;
    super::super::parallel::assembly::build_bound(request, resources, tokenizer, admission)
}

impl super::super::parallel::assembly::PipelineAssembly for Admission {
    fn kv(&self) -> ferrule_runtime::engine::ResidentKvPagePlan {
        self.kv
    }
    fn admit(&self, request: &ResolvedModelRequest, _: &BoundDecoderResources) -> Result<()> {
        self.admit_live(request.backend)
    }
    fn start(
        self,
        request: &ResolvedModelRequest,
        resources: BoundDecoderResources,
    ) -> Result<PipelineParallelExecutor> {
        executor(request, resources, self)
    }
}

fn executor(
    request: &ResolvedModelRequest,
    resources: BoundDecoderResources,
    admission: Admission,
) -> Result<PipelineParallelExecutor> {
    let topology = admission.topology.clone();
    #[cfg(unix)]
    if let Some(process) = admission.process {
        tracing::info!(
            owners = 2,
            frame_bytes = process.options.frame_limits.max_frame_bytes,
            process_host_bytes = process.host_bytes,
            "starting admitted dense hybrid process pipeline"
        );
        return Ok(PipelineParallelExecutor::new_hybrid_process(
            topology,
            admission.config,
            &resources,
            process.launch,
            process.boots,
            process.options,
        )?);
    }
    let Admission {
        plans,
        layout,
        config,
        stages,
        ..
    } = admission;
    #[cfg(not(feature = "cuda"))]
    let _ = stages;
    match request.backend {
        ModelExecutionBackend::Cpu => Ok(PipelineParallelExecutor::new_hybrid_with_program(
            topology,
            plans,
            config,
            layout,
            move |_, plan| PipelineStage::prepare_hybrid_cpu(&resources, plan, config),
        )?),
        #[cfg(feature = "cuda")]
        ModelExecutionBackend::Cuda => Ok(PipelineParallelExecutor::new_hybrid_with_program(
            topology,
            plans,
            config,
            layout,
            move |rank, plan| {
                use ferrule_model::decoder::{HybridCudaDevice, HybridCudaMemoryBudget};
                let stage = &stages[rank.local.get() as usize];
                let size = |n| {
                    usize::try_from(n).map_err(|_| ferrule_common::Error::Execution {
                        message: "hybrid budget exceeds address space".into(),
                    })
                };
                let device = HybridCudaDevice::new_on_device(
                    stage.device,
                    HybridCudaMemoryBudget {
                        kv_pages: config.physical_pages(PhysicalKvLayout::hybrid(
                            ModelExecutionBackend::Cuda,
                        ))?,
                        state_bytes: size(stage.state)?,
                        weight_bytes: size(stage.weights)?,
                        workspace_bytes: size(stage.workspace)?,
                    },
                )?;
                let (free, total) = device.operators().memory_info()?;
                if stage.total()? > free.min(total) as u64 {
                    return Err(ferrule_common::Error::Execution {
                        message: "hybrid owner capacity changed after aggregate admission".into(),
                    });
                }
                PipelineStage::prepare_hybrid_cuda(&resources, plan, config, device)
            },
        )?),
        #[cfg(not(feature = "cuda"))]
        ModelExecutionBackend::Cuda => Err(invalid("hybrid CUDA requires the cuda feature")),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/hybrid_pipeline.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/hybrid_budget/mod.rs"]
mod budget_tests;
