//! Strict family binding occurs once, before shared parallel assembly.
use super::*;
#[cfg(any(feature = "cuda", test))]
use crate::model_factory::parallel::{self, inputs::NumericProviderInputs};
use ferrule_model::models::qwen35::{Qwen35Adapter, Qwen35Profile};
#[cfg(any(feature = "cuda", test))]
use ferrule_runtime::parallel::hybrid_topology::HybridExpertTopology;

pub(in crate::model_factory) fn hybrid(
    request: &ResolvedModelRequest,
) -> Result<BoundDecoderResources> {
    let placement = request
        .pipeline
        .as_ref()
        .ok_or_else(|| invalid("missing hybrid options"))?;
    if crate::model_factory::mixed_serving::tensor::is_selected(request) {
        crate::model_factory::mixed_serving::tensor::require_quality()?;
        crate::model_factory::mixed_serving::tensor::validate_placement(placement)?;
    } else {
        placement.hybrid_precision(&request.family, request.backend)?;
    }
    let (metadata, resources) = Qwen35Adapter::bind_hf_metadata(&request.model_path)?;
    let expected = match request.family {
        ModelFamily::Qwen35 => Qwen35Profile::Dense08Bbf16,
        ModelFamily::Qwen35Moe => Qwen35Profile::Moe35BA3Bfp8,
        _ => return Err(invalid("unsupported hybrid binding family")),
    };
    if metadata.config().profile() != expected {
        return Err(invalid(
            "hybrid composition requires its strict full checkpoint profile",
        ));
    }
    Ok(resources)
}

/// Static policy is independent of feature availability and allocation. This is
/// the same strict admission used by metadata tests and the production binder.
#[cfg(any(feature = "cuda", test))]
fn validate_numeric<'a>(
    request: &'a ResolvedModelRequest,
    resources: &BoundDecoderResources,
) -> Result<&'a crate::model_factory::qwen35::Qwen35MoeOptions> {
    let parallel = request
        .pipeline
        .as_ref()
        .ok_or_else(|| invalid("missing numeric hybrid composition options"))?;
    // Static policy is checked separately from feature availability so this
    // exact metadata admission remains executable in CPU-only tests.
    let p = parallel.parallelism;
    if request.family != ModelFamily::Qwen35Moe
        || request.backend != ModelExecutionBackend::Cuda
        || !((p.pipeline_parallel == 2
            && p.tensor_parallel == 1
            && matches!(p.expert_parallel, 2 | 4 | 8))
            || (matches!(p.pipeline_parallel, 1 | 2)
                && matches!(p.tensor_parallel, 2 | 4)
                && p.expert_parallel == 2))
    {
        return Err(invalid(
            "numeric hybrid requires CUDA thread PP2 TP1 EP2/4/8 or PP1/2 TP2/4 EP2 and explicit root-then-stage-EP devices; no process/replay/fallback",
        ));
    }
    let policy = parallel.hybrid_policy(&request.family);
    parallel.validate_policy(
        request.backend,
        policy.layout,
        policy.execution,
        policy.placement,
    )?;
    parallel::bindings::validate_bound(request, resources)?;
    let moe = request
        .family_options
        .qwen35()
        .and_then(|f| f.moe())
        .ok_or_else(|| invalid("missing numeric profile"))?;
    if moe.device_capacity.is_some()
        || moe.host_cache.mode != ferrule_model::transformer::host_experts::ExpertPrewarmMode::Full
    {
        return Err(invalid(
            "numeric hybrid composition requires full shared prewarm; local cache/reader/head/prefix/proposal overrides unsupported",
        ));
    }
    crate::model_factory::parallel::bindings::validate_options(request)?;
    Ok(moe)
}

#[cfg(any(feature = "cuda", test))]
pub(in crate::model_factory) fn validate_capability(
    request: &ResolvedModelRequest,
    layout: &HybridExpertTopology,
    numeric: NumericFp8Profile,
) -> Result<()> {
    use crate::model_factory::qwen35_composition::{
        HybridExecutionProfile, Qwen35CompositionAdapter,
    };
    let parallel = request.pipeline.as_ref().expect("validated placement");
    // The existing public capability query describes TP2/4 only. TP1 retains
    // its separate admitted support; no quality evidence is promoted here.
    if parallel.parallelism.tensor_parallel > 1 {
        Qwen35CompositionAdapter::new(layout).execution_capability(HybridExecutionProfile {
            model: Qwen35Profile::Moe35BA3Bfp8,
            backend: request.backend,
            rank_backend: parallel.rank_backend,
            parallelism: parallel.parallelism,
            numeric,
        })?;
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
pub(super) fn local_inputs(request: &ResolvedModelRequest) -> Result<NumericProviderInputs> {
    let family = request
        .family_options
        .qwen35()
        .ok_or_else(|| invalid("missing Qwen3.5 adapter inputs"))?;
    let moe = family
        .moe()
        .ok_or_else(|| invalid("missing numeric profile"))?;
    Ok(NumericProviderInputs {
        model_label: "Qwen3.5-35B",
        family: request.family.clone(),
        limits: capacity_limits(request)?.into(),
        numeric_profile: moe.numeric_profile,
        parallel_execution: crate::model_factory::ParallelExecutionContract::external_f32(),
        host_cache: moe.host_cache,
        placement: family.expert_placement().map(|p| p.0.clone()),
    })
}

#[cfg(feature = "cuda")]
pub(in crate::model_factory) fn resident_inputs(
    request: &ResolvedModelRequest,
    resources: &BoundDecoderResources,
) -> Result<NumericProviderInputs> {
    if request.family != ModelFamily::Qwen35Moe
        || request.backend != ModelExecutionBackend::Cuda
        || request.pipeline.is_some()
    {
        return Err(ep_unsupported(
            "Qwen3.5 EP requires a CUDA resident root, not CPU or Pipeline",
        ));
    }
    let moe = request
        .family_options
        .qwen35()
        .and_then(|a| a.moe())
        .ok_or_else(|| ep_unsupported("missing numeric adapter"))?;
    if moe.device_capacity.is_some()
        || moe.host_cache.mode != ferrule_model::transformer::host_experts::ExpertPrewarmMode::Full
    {
        return Err(ep_unsupported(
            "Qwen3.5 EP requires full prewarm and no local cache override",
        ));
    }
    parallel::bindings::validate_bound(request, resources)?;
    crate::model_factory::parallel::bindings::validate_options(request)?;
    local_inputs(request)
}

#[cfg(any(feature = "cuda", test))]
fn lower_pipeline(
    request: &ResolvedModelRequest,
    resources: &BoundDecoderResources,
) -> Result<parallel::plan::Plan> {
    validate_numeric(request, resources)?;
    let inputs = local_inputs(request)?;
    let plan = parallel::plan::Plan::new(request, resources, &inputs)?;
    validate_capability(request, &plan.layout, inputs.numeric_profile)?;
    Ok(plan)
}

#[cfg(test)]
pub(in crate::model_factory) fn pipeline_plan(
    request: &ResolvedModelRequest,
    resources: &BoundDecoderResources,
) -> Result<parallel::plan::Plan> {
    lower_pipeline(request, resources)
}

#[cfg(feature = "cuda")]
pub(in crate::model_factory) fn pipeline_admission(
    request: &ResolvedModelRequest,
    resources: &BoundDecoderResources,
) -> Result<parallel::budget::Admission> {
    let plan = lower_pipeline(request, resources)?;
    parallel::budget::Admission::new(resources, plan)
}

pub(in crate::model_factory) fn build_pipeline(
    build: ModelBuildRequest,
) -> Result<BoxedSessionInferenceEngine> {
    let request = &build.resolved;
    if build.prepared_source_plan.is_some()
        && !crate::model_factory::mixed_serving::tensor::is_selected(request)
    {
        return Err(invalid("prepared source profile changed"));
    }
    #[cfg(feature = "cuda")]
    if crate::model_factory::mixed_serving::tensor::is_selected(request) {
        let (resources, admission) =
            crate::model_factory::mixed_serving::tensor::consume_prepared(&build)?;
        let tokenizer = TokenizerHandle::load(&request.model_path)?;
        return Ok(Box::new(parallel::assembly::build_bound(
            build.resolved,
            resources,
            tokenizer,
            admission,
        )?));
    }
    let request = build.resolved;
    let resources = hybrid(&request)?;
    let tokenizer = TokenizerHandle::load(&request.model_path)?;
    if request.family == ModelFamily::Qwen35Moe {
        #[cfg(feature = "cuda")]
        {
            let admission = pipeline_admission(&request, &resources)?;
            return Ok(Box::new(parallel::assembly::build_bound(
                request, resources, tokenizer, admission,
            )?));
        }
        #[cfg(not(feature = "cuda"))]
        return Err(moe_unavailable(request.backend.into()));
    }
    Ok(Box::new(
        crate::model_factory::pipeline::hybrid::build_bound(request, resources, tokenizer)?,
    ))
}
