//! Shared admission → host preparation → persistent owners → engine wrapping.
//! Provider and transport leaves construct existing runtime executors, never a
//! second publication, cleanup or quarantine authority.
use super::*;
use ferrule_runtime::engine::PipelineInferenceEngine;

pub(in crate::model_factory) trait PipelineAssembly: Sized {
    fn kv(&self) -> ResidentKvPagePlan;
    fn admit(
        &self,
        request: &ResolvedModelRequest,
        resources: &BoundDecoderResources,
    ) -> Result<()>;
    fn prepare_host(&self, resources: BoundDecoderResources) -> Result<BoundDecoderResources> {
        Ok(resources)
    }
    fn start(
        self,
        request: &ResolvedModelRequest,
        resources: BoundDecoderResources,
    ) -> Result<PipelineParallelExecutor>;
}

pub(in crate::model_factory) fn build_bound<A: PipelineAssembly>(
    request: ResolvedModelRequest,
    resources: BoundDecoderResources,
    tokenizer: TokenizerHandle,
    admission: A,
) -> Result<PipelineInferenceEngine> {
    admission.admit(&request, &resources)?;
    let kv = admission.kv();
    let pipeline = start_admitted(&request, resources, admission)?;
    PipelineInferenceEngine::new(
        pipeline,
        tokenizer,
        request.model_info,
        request.scheduler_config,
        request.driver_config.stop_at_eos,
        kv,
    )
}

/// Admission precedes this boundary. Host authority is prepared before any
/// persistent owner can upload; owner Ready and cleanup stay in runtime startup.
pub(in crate::model_factory) fn start_admitted<A: PipelineAssembly>(
    request: &ResolvedModelRequest,
    resources: BoundDecoderResources,
    admission: A,
) -> Result<PipelineParallelExecutor> {
    let resources = admission.prepare_host(resources)?;
    admission.start(request, resources)
}

#[cfg(feature = "cuda")]
impl PipelineAssembly for super::budget::Admission {
    fn kv(&self) -> ResidentKvPagePlan {
        self.plan.kv
    }
    fn admit(&self, _: &ResolvedModelRequest, resources: &BoundDecoderResources) -> Result<()> {
        self.admit_live(resources)
    }
    fn prepare_host(&self, resources: BoundDecoderResources) -> Result<BoundDecoderResources> {
        self.plan.validate_execution()?;
        if self.plan.owners.is_empty() {
            return Ok(resources);
        }
        Ok(resources.with_host_expert_prewarm(
            self.plan.host(),
            self.plan.config().max_parameter_bytes,
            |progress| tracing::info!(?progress, "parallel shared compressed host warm"),
        )?)
    }
    fn start(
        self,
        request: &ResolvedModelRequest,
        resources: BoundDecoderResources,
    ) -> Result<PipelineParallelExecutor> {
        start_numeric(request, resources, self)
    }
}

/// Both root providers consume the same plan, admission, warmed authority and
/// owner preparation. Only typed stage construction and collective wiring differ.
#[cfg(feature = "cuda")]
fn start_numeric(
    request: &ResolvedModelRequest,
    resources: BoundDecoderResources,
    admission: super::budget::Admission,
) -> Result<PipelineParallelExecutor> {
    use ferrule_model::decoder::HybridCudaRoutedExperts;
    use ferrule_runtime::parallel::pipeline::{
        PipelineStage, hybrid_cuda_tensor::HybridCudaTensorStage,
    };
    use std::sync::Arc;
    let placement = request.pipeline.as_ref().expect("admitted placement");
    let topology = admission.plan.topology.clone();
    let resources = Arc::new(resources);
    let config = admission.plan.config();
    let timeout = placement.rank_timeout;
    let admission = Arc::new(admission);
    if let Some(layout) = &admission.plan.segment_layout {
        return Ok(PipelineParallelExecutor::new_hybrid_with_program(
            topology,
            admission.plan.segments.clone(),
            config,
            layout.clone(),
            move |rank, segment| {
                let owner = super::provider::prepare_owner(
                    rank,
                    rank.local.get() as usize,
                    &resources,
                    &admission,
                    timeout,
                )?;
                let (group, routed) = owner.routed.expect("validated segment source");
                PipelineStage::prepare_hybrid_cuda_with_routed_experts(
                    &resources,
                    segment,
                    config,
                    owner.device,
                    &admission.options,
                    group,
                    HybridCudaRoutedExperts::new(rank.global, routed),
                )
            },
        )?);
    }
    let groups = admission
        .plan
        .owners
        .iter()
        .map(|o| o.group.clone())
        .collect();
    Ok(PipelineParallelExecutor::new_hybrid_cuda_tensor(
        topology,
        Arc::clone(&resources),
        admission.plan.segments.clone(),
        admission.plan.layout.clone(),
        groups,
        config,
        admission.plan.collective,
        timeout,
        move |rank, segment, parallel, lane, endpoint| {
            let owner = super::provider::prepare_owner(
                rank,
                parallel.scope().stage as usize,
                &resources,
                &admission,
                timeout,
            )?;
            HybridCudaTensorStage::prepare_hybrid_cuda_tensor(
                &resources,
                segment,
                parallel,
                lane,
                config,
                &admission.options,
                owner.device,
                endpoint,
                owner.routed,
            )
        },
    )?)
}

impl super::plan::ParallelExecutionPlan {
    /// Internal experimental execution, not catalog/profile quality enablement.
    /// Requires strict bound resources and the complete model-owned ledger before
    /// any persistent owner can create CUDA state or read a parameter payload.
    #[cfg(feature = "cuda")]
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn start_internal(
        &self,
        resources: BoundDecoderResources,
        config: PipelineConfig,
        margins: super::requirements::ParallelOwnerMargins,
        capacities: &[(usize, u64)],
        host_capacity: u64,
        limits: HostCollectiveLimits,
        timeout: std::time::Duration,
    ) -> Result<PipelineParallelExecutor> {
        self.start_internal_with_trace(
            resources,
            config,
            margins,
            capacities,
            host_capacity,
            limits,
            timeout,
            None,
        )
    }

    /// CPU-only preparation shared by internal tests and owner assembly.
    /// Device capacities use the placement's actual ordinals, never lane guesses.
    #[cfg(feature = "cuda")]
    #[doc(hidden)]
    pub fn internal_mixed_execution_inputs(
        &self,
        resources: &BoundDecoderResources,
        config: PipelineConfig,
        margins: super::requirements::ParallelOwnerMargins,
        capacities: &[(usize, u64)],
        limits: HostCollectiveLimits,
        requested_cache_grants: Option<usize>,
    ) -> Result<ferrule_runtime::parallel::pipeline::MixedTensorExecutionPlan> {
        use ferrule_runtime::parallel::pipeline::MixedTensorExecutionPlan;
        let inputs = MixedTensorExecutionPlan::new(
            resources.spec(),
            self.topology(),
            self.segments(),
            self.layout(),
            config,
            limits,
        )?;
        let Some(grants) = requested_cache_grants else {
            return Ok(inputs);
        };
        let capacity = super::requirements::ParallelRankCapacity {
            page_size: config.page_size,
            max_positions: config.max_positions,
            max_rows: config.max_batch_tokens,
            sessions: config.session_capacity,
            logical_pages: config.max_pages,
        };
        let programs = self.mixed_programs_with_decoded_weight_cache(
            resources, capacity, margins, capacities, grants,
        )?;
        Ok(inputs.with_programs(programs)?)
    }

    /// Opt-in diagnostic transport, not full-inventory acceptance or public Mixed support.
    #[cfg(feature = "cuda")]
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn start_internal_with_trace(
        &self,
        resources: BoundDecoderResources,
        config: PipelineConfig,
        margins: super::requirements::ParallelOwnerMargins,
        capacities: &[(usize, u64)],
        host_capacity: u64,
        limits: HostCollectiveLimits,
        timeout: std::time::Duration,
        trace: Option<ferrule_runtime::parallel::pipeline::MixedTraceConfig>,
    ) -> Result<PipelineParallelExecutor> {
        self.start_internal_with_decoded_cache(
            resources,
            config,
            margins,
            capacities,
            host_capacity,
            limits,
            timeout,
            trace,
            None,
        )
    }

    /// Internal experiment only. None preserves the native/public path; Some
    /// must pass the model's complete cache ledger before either owner opens.
    #[cfg(feature = "cuda")]
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn start_internal_with_decoded_cache(
        &self,
        resources: BoundDecoderResources,
        config: PipelineConfig,
        margins: super::requirements::ParallelOwnerMargins,
        capacities: &[(usize, u64)],
        host_capacity: u64,
        limits: HostCollectiveLimits,
        timeout: std::time::Duration,
        trace: Option<ferrule_runtime::parallel::pipeline::MixedTraceConfig>,
        requested_cache_grants: Option<usize>,
    ) -> Result<PipelineParallelExecutor> {
        let inputs = self.internal_mixed_execution_inputs(
            &resources,
            config,
            margins,
            capacities,
            limits,
            requested_cache_grants,
        )?;
        self.start_internal_with_inputs(
            resources,
            config,
            margins,
            capacities,
            host_capacity,
            limits,
            timeout,
            trace,
            inputs,
        )
    }

    /// Public catalog uses exactly the programs retained by metadata admission.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::model_factory) fn start_internal_with_programs(
        &self,
        resources: BoundDecoderResources,
        config: PipelineConfig,
        margins: super::requirements::ParallelOwnerMargins,
        capacities: &[(usize, u64)],
        host_capacity: u64,
        limits: HostCollectiveLimits,
        timeout: std::time::Duration,
        programs: Vec<ferrule_model::transformer::hybrid_parallel::HybridMixedRankProgramPlan>,
    ) -> Result<PipelineParallelExecutor> {
        let inputs = ferrule_runtime::parallel::pipeline::MixedTensorExecutionPlan::new(
            resources.spec(),
            self.topology(),
            self.segments(),
            self.layout(),
            config,
            limits,
        )?
        .with_programs(programs)?;
        self.start_internal_with_inputs(
            resources,
            config,
            margins,
            capacities,
            host_capacity,
            limits,
            timeout,
            None,
            inputs,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn start_internal_with_inputs(
        &self,
        resources: BoundDecoderResources,
        config: PipelineConfig,
        margins: super::requirements::ParallelOwnerMargins,
        capacities: &[(usize, u64)],
        host_capacity: u64,
        limits: HostCollectiveLimits,
        timeout: std::time::Duration,
        mut trace: Option<ferrule_runtime::parallel::pipeline::MixedTraceConfig>,
        inputs: ferrule_runtime::parallel::pipeline::MixedTensorExecutionPlan,
    ) -> Result<PipelineParallelExecutor> {
        use super::requirements::ParallelRankCapacity;
        use ferrule_runtime::parallel::pipeline::hybrid_cuda_tensor::{
            HybridCudaTensorStage, MixedCudaTensorProgram,
        };
        use std::sync::Arc;
        if std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref() != Ok(":0:0") {
            return Err(invalid(
                "Mixed startup requires launcher CUBLAS_WORKSPACE_CONFIG=:0:0 before CUDA initialization",
            ));
        }
        let capacity = ParallelRankCapacity {
            page_size: config.page_size,
            max_positions: config.max_positions,
            max_rows: config.max_batch_tokens,
            sessions: config.session_capacity,
            logical_pages: config.max_pages,
        };
        let report =
            self.resource_requirements_from_mixed_inputs(&resources, capacity, margins, &inputs)?;
        let trace_pages = capacity.physical_pages()?;
        let trace_host = trace
            .as_ref()
            .map(|t| t.host_reserve_with_pages(self.root_count(), trace_pages))
            .transpose()?
            .unwrap_or(0);
        let admitted_host = host_capacity
            .checked_sub(trace_host)
            .ok_or_else(|| invalid("trace host reserve exceeds capacity"))?;
        let mut budgets = report.prepare_internal(capacities, admitted_host, |owners| {
            owners
                .iter()
                .map(|o| o.stage_budget())
                .collect::<Result<Vec<_>>>()
        })?;
        if let Some(config) = trace.take() {
            for budget in &mut budgets {
                budget.workspace = super::add(budget.workspace, config.device_reserve()?)?;
            }
            ferrule_runtime::parallel::pipeline::HybridDeviceBudget::admit(&budgets, capacities)?;
            let pages = capacity.physical_pages()?;
            let host = config.host_reserve_with_pages(1, pages)?;
            trace = Some(config.admit_host(host, pages)?);
        }
        tracing::info!(
            trace_host,
            trace_device_per_owner = trace
                .as_ref()
                .map(|t| t.device_reserve())
                .transpose()?
                .unwrap_or(0),
            "Mixed optional diagnostic reserves"
        );
        tracing::info!(
            ?capacity,
            ?budgets,
            known_host_bytes = report.known_host_bytes()?,
            attention_views = inputs.attention_shapes().len(),
            "Mixed experimental source-bound admission"
        );
        for owner in report.owners() {
            tracing::info!(
                owner = owner.owner.get(),
                device = owner.device,
                providers = ?owner.operators.providers,
                boundary = ?owner.operators.boundary,
                "Mixed owner resource components"
            );
        }
        // Validate the exact model claim (including every warmed attention view)
        // before opening either CUDA owner, not after the first upload.
        for (lane, budget) in budgets.iter().enumerate() {
            let (estimate, context) = inputs.program(lane)?.cuda_claim_inputs(
                &resources,
                capacity.physical_pages()?,
                budget.device,
                config.max_parameter_bytes,
                inputs.attention_shapes(),
            )?;
            context.validate_device_budget(
                budget.device,
                ferrule_model::decoder::HybridCudaMemoryBudget {
                    kv_pages: capacity.physical_pages()?,
                    state_bytes: budget.state as usize,
                    weight_bytes: budget.weights as usize,
                    workspace_bytes: budget.workspace as usize,
                },
            )?;
            context.validate_estimate(estimate)?;
        }
        let resources = Arc::new(resources);
        let image = Arc::clone(&resources);
        let inputs = Arc::new(inputs);
        Ok(PipelineParallelExecutor::new_hybrid_cuda_tensor(
            self.topology().clone(),
            resources,
            self.segments().to_vec(),
            self.layout().clone(),
            vec![],
            config,
            limits,
            timeout,
            move |rank, segment, parallel, lane, endpoint| {
                let program = inputs.program(lane)?;
                if rank.global != program.layout().owner()
                    || segment != *program.segment()
                    || parallel.rank_layout(lane)? != *program.layout()
                {
                    return Err(ferrule_common::Error::Execution {
                        message: "Mixed owner differs from prepared parallel assembly".into(),
                    });
                }
                let budget = &budgets[rank.local.get() as usize];
                let device = super::provider::prepare_device(&image, config, budget)?;
                HybridCudaTensorStage::<MixedCudaTensorProgram>::prepare_mixed_with_trace(
                    &image,
                    &inputs,
                    lane,
                    device,
                    endpoint,
                    trace.clone(),
                )
            },
        )?)
    }
}
