//! Shared resident F32 runner assembly for already-bound decoder resources.

use super::{ResolvedModelRequest, invalid, kv_accounting, log_kv_plan, physical_page_bytes};
use crate::Result;
use ferrule_model::decoder::{
    GenericDecoderOptions, GenericDecoderRunner, HybridCpuDecoder, HybridStateSchema,
};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::{
    ModelExecutionBackend, ModelFamily, TokenizerHandle, WeightSource,
    transformer::BoundDecoderResources,
};
#[cfg(feature = "cuda")]
use ferrule_runtime::engine::plan_resident_kv_pages;
use ferrule_runtime::engine::{
    BoxedSessionInferenceEngine, ResidentKvPageAccounting, build_resident_engine,
};

pub(super) fn build_resident_f32_bound(
    request: ResolvedModelRequest,
    resources: BoundDecoderResources,
    tokenizer: TokenizerHandle,
) -> Result<BoxedSessionInferenceEngine> {
    let options = GenericDecoderOptions::standard_cpu(
        resources.spec(),
        request.family.clone(),
        WeightSource::Safetensors,
        16,
        request.driver_config.ctx_size,
        request.scheduler_config.max_batch_tokens,
        request.scheduler_config.max_active_sequences,
        request.max_tensor_bytes,
        ExecutionPrecisionPolicy::f32(),
    )?;
    let schema = HybridStateSchema::from_spec(resources.spec(), resources.spec().layers().len())?
        .kv_planes(options.page_size(), options.max_positions())?;
    let page_bytes = physical_page_bytes(&schema)?;
    let accounting = if request.backend == ModelExecutionBackend::Cuda {
        ResidentKvPageAccounting::TransactionCapacity {
            page_bytes,
            budget_bytes: request.kv_cache_bytes,
        }
    } else {
        kv_accounting(request.kv_cache_bytes, Some(page_bytes))
    };
    let model_label = if request.family == ModelFamily::Qwen3 {
        "dense Qwen3 F32 HostKV"
    } else {
        "Qwen3.5 hybrid"
    };
    log_kv_plan(model_label, &schema, accounting, &request)?;
    match request.backend {
        ModelExecutionBackend::Cpu => {
            let runner = GenericDecoderRunner::<HybridCpuDecoder>::hybrid_cpu(
                resources, tokenizer, options,
            )?;
            build_resident_engine(
                runner,
                Box::new(schema),
                accounting,
                request.scheduler_config,
                request.driver_config,
            )
        }
        #[cfg(feature = "cuda")]
        ModelExecutionBackend::Cuda => {
            use ferrule_model::decoder::{
                HybridCudaDecoder, HybridCudaDevice, HybridCudaMemoryBudget,
                HybridCudaMemoryEstimate,
            };
            let kv_plan = plan_resident_kv_pages(
                &schema,
                accounting,
                request.scheduler_config,
                request.driver_config,
            )?;
            let pages = kv_plan.configured_pages;
            let idle_reclaim = match request.execution.kv_execution_policy {
                ferrule_runtime::engine::KvExecutionPolicy::IdleReclaim(config) => Some(config),
                _ => None,
            };
            let estimate = match idle_reclaim {
                Some(config) => HybridCudaMemoryEstimate::for_resources_with_idle_kv_reclaim(
                    &resources, &options, pages, config,
                )?,
                None => HybridCudaMemoryEstimate::for_resources(&resources, &options, pages)?,
            };
            if idle_reclaim.is_some()
                && request
                    .kv_cache_bytes
                    .is_some_and(|limit| estimate.kv_bytes as u64 > limit)
            {
                return Err(invalid(
                    "idle KV allocator grants plus Entry staging exceed the KV device byte budget",
                ));
            }
            // Default + committed sequences + transaction working copies.
            // Forks consume this same hard cap; prefix cache is disabled.
            let state_bytes = request
                .scheduler_config
                .max_active_sequences
                .checked_mul(2)
                .and_then(|n| n.checked_add(1))
                .and_then(|n| n.checked_mul(estimate.per_sequence_state_bytes))
                // The CUDA owner requires a nonzero domain even for an all-GQA
                // model with no recurrent state. Charge the sentinel, not fake GDN.
                .map(|bytes| bytes.max(1))
                .ok_or_else(|| invalid("Qwen3.5 CUDA recurrent state budget overflow"))?;
            let budget = HybridCudaMemoryBudget {
                kv_pages: pages,
                state_bytes,
                weight_bytes: estimate.weight_bytes_upper_bound,
                workspace_bytes: estimate.workspace_bytes_upper_bound,
            };
            let root_required_bytes = [
                budget.weight_bytes,
                budget.state_bytes,
                budget.workspace_bytes,
                estimate.kv_bytes,
            ]
            .into_iter()
            .try_fold(0usize, |total, bytes| {
                total
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("Qwen3.5 root memory requirement overflow"))
            })?;
            let profile = kv_plan
                .context_capacity_profile(request.scheduler_config, request.driver_config)?
                .with_memory_requirements(
                    state_bytes,
                    budget.workspace_bytes,
                    root_required_bytes,
                )?;
            tracing::info!(
                ?estimate,
                ?budget,
                ?profile,
                model = model_label,
                "local F32 CUDA owner admission (text-only, no speculation/TP/PP)"
            );
            let device = HybridCudaDevice::new_on_device(0, budget)?;
            let runner = match idle_reclaim {
                Some(config) => {
                    GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda_with_idle_kv_reclaim(
                        resources, tokenizer, options, device, config,
                    )?
                }
                None => GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda(
                    resources, tokenizer, options, device,
                )?,
            };
            let mut engine = ferrule_runtime::engine::compose_resident_engine(
                runner,
                Box::new(schema),
                accounting,
                request.scheduler_config,
                request.driver_config,
            )?;
            engine.install_cuda_host_kv_dispatch();
            Ok(Box::new(engine))
        }
        #[cfg(not(feature = "cuda"))]
        ModelExecutionBackend::Cuda => Err(invalid("Qwen3.5 CUDA requires the 'cuda' feature")),
    }
}
