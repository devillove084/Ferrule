//! Strict Qwen3.5 binding and catalog policy; execution belongs to shared providers.
use super::*;
use crate::model_factory::resident_f32_bound;
#[cfg(all(test, feature = "cuda"))]
use ferrule_model::decoder::GenericDecoderRunner;
use ferrule_model::decoder::{GenericDecoderOptions, HybridStateSchema};
#[cfg(test)]
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::models::qwen35::Qwen35Adapter;
use ferrule_model::{TokenizerHandle, transformer::BoundDecoderResources};
use ferrule_runtime::engine::ContextCapacityProfile;
mod options;
pub(super) use options::*;
mod capacity;
pub use capacity::{Qwen35MoeCapacityPlan, Qwen35MoeMemoryPlan};
pub(in crate::model_factory) mod binding;
pub(in crate::model_factory) mod ep;
#[cfg(all(test, feature = "cuda"))]
pub(super) use ep::strict::full_logits_step;
mod compatibility;
#[cfg(test)]
use crate::model_factory::parallel::budget::host_startup_options;
pub(super) use crate::model_factory::parallel::inputs::EXPERT_TOPOLOGY;
#[cfg(test)]
use crate::model_factory::parallel::reporting::{MoeShutdownReport, ReportedMoeEngine};
pub use compatibility::{
    Qwen35ExpertPhysicalPlan, Qwen35ExpertPlacement, Qwen35ExpertRootMemory,
    Qwen35MoeCapacityLimits, Qwen35PhysicalCardCapacity, Qwen35PhysicalCardMemory,
};
#[cfg(test)]
pub(super) fn finish_owner_admission<T>(
    admission: Result<T>,
    cleanup: impl FnOnce() -> Result<()>,
) -> Result<T> {
    crate::model_factory::parallel::experts::finish_owner_admission(
        "Qwen3.5 EP owner admission",
        admission,
        cleanup,
    )
}

impl ResidentModelPlanner {
    /// Resolve Qwen3.5 EP through the resident registry, never the generic PP/KV
    /// topology. This only prepares metadata; construction remains owner-local.
    pub fn prepare_qwen35_expert_parallel(
        &self,
        config: &AutoConfig,
        backend: BackendSelection,
        chat_template_override: Option<&str>,
        options: ModelFactoryOptions,
        parallel: PipelineBuildOptions,
    ) -> Result<ResidentModelBuildPlan> {
        if config.descriptor().spec.family != ModelFamily::Qwen35Moe {
            return Err(ep_unsupported("Qwen3.5 EP requires Qwen35Moe"));
        }
        let entry = implementation(&config.descriptor().spec.family)?;
        let strategy = qwen35_composition::select_strategy(
            entry,
            backend,
            &SelectionRequest::parallel(Some(parallel.clone())),
            &options,
        )?;
        if strategy.execution.construction.kind() != SelectedImplementationKind::Resident {
            return Err(ep_unsupported(
                "Qwen3.5 EP requires a resident root (PP1 TP1)",
            ));
        }
        self.prepare_strategy(
            config,
            backend,
            chat_template_override,
            options,
            Some(parallel),
            strategy,
        )
    }
}

pub(super) fn build(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    let request = request.resolved;
    if request.family == ModelFamily::Qwen35Moe {
        #[cfg(feature = "cuda")]
        return build_moe(request);
        #[cfg(not(feature = "cuda"))]
        return Err(moe_unavailable(request.backend.into()));
    }
    // Resources, tokenizer and CUDA state are created on the consuming owner.
    // The family adapter's load-with-backend helper is deliberately CPU-only.
    let (_, resources) = Qwen35Adapter::bind_hf_metadata(&request.model_path)?;
    let tokenizer = TokenizerHandle::load(&request.model_path)?;
    resident_f32_bound::build_resident_f32_bound(request, resources, tokenizer)
}

/// Full-depth single-GPU composition. Limits are explicit; never KeepAll.
#[cfg_attr(
    not(feature = "cuda"),
    allow(
        dead_code,
        reason = "host metadata remains compiled independently of CUDA construction"
    )
)]
fn numeric_profile(request: &ResolvedModelRequest) -> NumericFp8Profile {
    request
        .family_options
        .qwen35()
        .and_then(|options| options.moe())
        .expect("validated Qwen35Moe")
        .numeric_profile
}

pub(super) fn capacity_limits(request: &ResolvedModelRequest) -> Result<Qwen35MoeCapacityLimits> {
    let mut limits = request
        .family_options
        .qwen35()
        .and_then(|options| options.moe())
        .expect("validated Qwen35Moe")
        .device_capacity
        .unwrap_or_default();
    limits.max_positions = request.driver_config.ctx_size;
    limits.max_sequences = request.scheduler_config.max_active_sequences;
    limits.max_batch_tokens = request.scheduler_config.max_batch_tokens;
    if let Some(bytes) = request.kv_cache_bytes {
        limits.kv_bytes =
            usize::try_from(bytes).map_err(|_| invalid("35B KV budget exceeds usize"))?;
    }
    Ok(limits)
}

#[cfg(feature = "cuda")]
fn build_moe(request: ResolvedModelRequest) -> Result<BoxedSessionInferenceEngine> {
    if request.backend != ModelExecutionBackend::Cuda {
        return Err(moe_unavailable(BackendSelection::Cpu));
    }
    if request.pipeline.is_some() {
        return Err(ep_unsupported(
            "Qwen3.5 generic pipeline TP/PP/EP/process is unsupported; use the dedicated resident thread/process EP placement",
        ));
    }
    crate::model_factory::parallel::bindings::validate_cache_options(&request).map_err(|error| invalid(format!("35B does not consume legacy pinned expert cache, hotset, separate expert-reader or output-head chunk overrides; use explicit compressed host prewarm options for pageable host residency: {error}")))?;
    if request.driver_config.enable_native_proposals
        || request.scheduler_config.prefix_cache_capacity_pages != 0
        || request.scheduler_config.prefill_chunk_size == 0
        || request.scheduler_config.prefill_chunk_size > request.scheduler_config.max_batch_tokens
    {
        return Err(invalid(
            "35B requires bounded token-serial prefill with prefix cache and speculation disabled",
        ));
    }
    let (metadata, resources) = Qwen35Adapter::bind_hf_metadata(&request.model_path)?;
    if request.max_layers != metadata.config().text().num_hidden_layers {
        return Err(invalid("35B requires the full checkpoint layer count"));
    }
    let limits = capacity_limits(&request)?;
    let target = Qwen35MoeCapacityPlan::from_config_with_profile(
        metadata.config(),
        limits,
        numeric_profile(&request),
    )?;
    let admission =
        Qwen35MoeMemoryPlan::for_resources(target, &resources, request.max_tensor_bytes)?;
    let profile =
        admission.context_capacity_profile(request.scheduler_config.prefill_chunk_size)?;
    tracing::info!(?profile, "Qwen3.5-35B owner effective context capacity");
    let inputs = binding::local_inputs(&request)?;
    let bound = crate::model_factory::parallel::budget::BoundMemoryPlan {
        options: admission.options,
        estimate: admission.estimate,
        budget: admission.budget,
        total_device_bytes: admission.total_device_bytes,
    };
    crate::model_factory::parallel::resident::build_local(request, resources, inputs, bound)
}

pub(super) fn build_experts(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    #[cfg(feature = "cuda")]
    return ep::build(request.resolved);
    #[cfg(not(feature = "cuda"))]
    Err(moe_unavailable(request.resolved.backend.into()))
}

pub(super) fn attach_experts(
    mut plan: ResidentModelBuildPlan,
    parallel: PipelineBuildOptions,
    placement: Qwen35ExpertPlacement,
    execution: ResolvedExecution,
) -> Result<ResidentModelBuildPlan> {
    placement.validate_expert_shape(
        plan.request.model_info.num_layers,
        plan.request.model_info.num_experts,
    )?;
    if plan.requested_options.qwen35_moe_capacity.is_some() {
        return Err(ep_unsupported(
            "Qwen3.5 EP keeps all owned experts resident; single-device --cuda-expert-device-* cache overrides are unsupported",
        ));
    }
    if plan
        .requested_options
        .qwen35_host_cache
        .is_some_and(|host| {
            host.mode != ferrule_model::transformer::host_experts::ExpertPrewarmMode::Full
        })
    {
        return Err(ep_unsupported(
            "Qwen3.5 EP requires full prewarm (shared for threads, owned partitions for processes)",
        ));
    }
    plan.request.execution = execution;
    plan.backend_profile = plan
        .numeric_fp8_profile()
        .expect("validated numeric profile")
        .expert_backend_name(placement.rank_backend() == PipelineRankBackend::Process);
    plan.requested_parallel = Some(parallel);
    plan.request
        .family_options
        .qwen35_mut()
        .expect("validated Qwen35Moe")
        .set_expert_placement(placement);
    Ok(plan)
}

pub(super) fn attach_expert_options(
    plan: ResidentModelBuildPlan,
    parallel: PipelineBuildOptions,
    execution: ResolvedExecution,
) -> Result<ResidentModelBuildPlan> {
    let placement = Qwen35ExpertPlacement::from_options(&parallel, plan.backend.into())?;
    attach_experts(plan, parallel, placement, execution)
}

#[cfg(test)]
pub(super) use crate::hybrid_fixture as fixture;

#[cfg(test)]
#[path = "../../../../tests/unit/engine/model_factory/qwen35_adapter.rs"]
mod tests;
