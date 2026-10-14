//! Explicit local dense-Qwen3 F32 route; the default catalog/TP factory is unchanged.
use super::*;
#[cfg(feature = "cuda")]
use crate::model_factory::resident_f32_bound;
use ferrule_runtime::engine::{HostKvDecodeConfig, KvExecutionPolicy};

pub(super) fn select(
    _: &ModelImplementation,
    backend: BackendSelection,
    request: &SelectionRequest,
    options: &ModelFactoryOptions,
) -> Result<ResolvedStrategy> {
    let serving = &request.serving;
    validate_request(backend, options, serving)?;
    let idle = matches!(
        serving.kv_execution_policy,
        KvExecutionPolicy::IdleReclaim(_)
    );
    let mut execution = ResolvedExecution::new(
        ModelExecutionBackend::Cuda,
        ResolvedConstruction::Resident(build),
        StorageEncoding::ProfileDefined,
        ArithmeticPrecision::F32,
        "local dense F32 KV policy",
    );
    execution.kv_execution_policy = serving.kv_execution_policy;
    execution.support.arithmetic = ArithmeticPrecision::F32;
    execution.support.static_constraints = "local dense Qwen3 CUDA F32 all-GQA; full depth, single sequence; resident prefill warming then scoped HostKV decode; no numeric/TP/EP; existing resident KV capacity unchanged";
    if idle {
        execution.support.static_constraints = "local dense Qwen3 CUDA F32 all-GQA; explicit suspended-session VRAM reclamation; bounded Entry staging; device attention after restore; no active HostKV prefix decode/numeric/TP/EP";
    }
    execution.conversion = PipelineConversion::Frozen(
        "with_pipeline cannot replace an explicit HostKV/IdleReclaim execution policy",
    );
    Ok(ResolvedStrategy {
        execution,
        backend: ModelExecutionBackend::Cuda,
        catalog_backend: BackendSelection::Cuda,
        resolve: if idle { resolve_idle } else { resolve_host },
        policy: &OPTIONS,
        parallel: None,
    })
}
fn resolve_host(_: &ModelDescriptor, _: BackendSelection) -> Result<ResolvedModelBackend> {
    Ok(ResolvedModelBackend {
        family: ModelFamily::Qwen3,
        model_name: "qwen3-dense-host-kv",
        backend: ModelExecutionBackend::Cuda,
        backend_profile: "cuda-f32-host-kv-decode",
        default_chat_template: ChatTemplate::Qwen3,
    })
}
fn resolve_idle(d: &ModelDescriptor, b: BackendSelection) -> Result<ResolvedModelBackend> {
    let mut entry = resolve_host(d, b)?;
    entry.model_name = "qwen3-dense-idle-kv";
    entry.backend_profile = "cuda-f32-idle-kv-reclaim";
    Ok(entry)
}
const OPTIONS: ModelOptionPolicy = ModelOptionPolicy {
    serving: common_serving_options,
    parallel: reject_parallel,
    finish: no_provider_admission,
    adapt: |descriptor, options, execution| {
        if descriptor.spec.family != ModelFamily::Qwen3
            || descriptor.spec.weight_source != WeightSource::Safetensors
            || options
                .max_layers
                .is_some_and(|n| Some(n) != descriptor.spec.num_layers)
        {
            return Err(invalid(
                "HostKV decode serving requires full-depth dense Qwen3 Safetensors",
            ));
        }
        generic_options(descriptor, options, execution)
    },
};

pub(super) fn validate_request(
    backend: BackendSelection,
    options: &ModelFactoryOptions,
    serving: &ServingModelOptions,
) -> Result<()> {
    let host = match serving.kv_execution_policy {
        KvExecutionPolicy::HostKvDecode(host) => host,
        KvExecutionPolicy::IdleReclaim(c)
            if c.staging_page_limit > 0
                && c.device_bytes > 0
                && c.host_bytes > 0
                && c.pinned_bytes > 0 =>
        {
            HostKvDecodeConfig::default()
        }
        _ => return Err(invalid("invalid dense CUDA KV policy/budgets")),
    };
    validate(backend, options, serving, host)
}
fn validate(
    backend: BackendSelection,
    options: &ModelFactoryOptions,
    serving: &ServingModelOptions,
    host: HostKvDecodeConfig,
) -> Result<()> {
    host.validate()?;
    if backend != BackendSelection::Cuda
        || serving.parallel.is_some()
        || serving.engine == ServingEngineSelection::Pipeline
        || options.numeric_fp8_profile.is_some()
        || options.qwen35_moe_capacity.is_some()
        || options.qwen35_host_cache.is_some()
        || (matches!(
            serving.kv_execution_policy,
            KvExecutionPolicy::HostKvDecode(_)
        ) && (options.scheduler_config.max_active_sequences != 1
            || options.scheduler_config.max_decode_batch != 1))
        || serving.expert_host_cache_entries.is_some()
        || serving.expert_host_cache_mebibytes.is_some()
        || serving.expert_pinned_cache_entries.is_some()
        || serving.expert_pinned_cache_mebibytes.is_some()
        || serving.moe_hotset_experts.is_some()
        || options.moe_hotset_experts != 0
    {
        return Err(invalid(
            "HostKV decode serving requires explicit CUDA, full-depth dense Qwen3, one active/decode sequence, local resident owner; no GDN/numeric/TP/EP/parallel or expert-policy overrides",
        ));
    }
    Ok(())
}
#[cfg(feature = "cuda")]
fn build(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    let request = request.resolved;
    let (_, checkpoint) = Qwen3DenseAdapter::bind_hf_metadata(&request.model_path)?;
    let resources = checkpoint.into_resources();
    ferrule_runtime::engine::validate_host_kv_spec(resources.spec(), false)?;
    resources.validate_parameter_limits(request.max_tensor_bytes, request.max_tensor_bytes)?;
    let tokenizer = ferrule_model::TokenizerHandle::load(&request.model_path)?;
    // Reuse the existing concrete F32 CUDA runner composition and sole KV owner.
    resident_f32_bound::build_resident_f32_bound(request, resources, tokenizer)
}
#[cfg(not(feature = "cuda"))]
fn build(_: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    Err(invalid(
        "HostKV decode requires a CUDA-enabled runtime build",
    ))
}

#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/dense_f32_selection.rs"]
mod selection_tests;
