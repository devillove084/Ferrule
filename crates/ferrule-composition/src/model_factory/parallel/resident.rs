#![cfg(feature = "cuda")]
//! Resident root assembly with external experts; never a pipeline transaction owner.
use super::*;
use super::{
    budget::{ExpertPhysicalPlan, ExpertRootMemory, PhysicalCardCapacity, host_startup_options},
    experts::{OwnerPlan, finish_owner_admission},
    inputs::NumericCapacityLimits,
    reporting::ReportedMoeEngine,
};
use crate::model_factory::log_kv_plan;
#[cfg(feature = "cuda")]
use ferrule_backend::cuda::operators::linear::CudaOperators;
#[cfg(feature = "cuda")]
use ferrule_model::decoder::{
    HybridCudaDecoder, HybridCudaDevice, HybridCudaMemoryBudget, HybridCudaRoutedExperts,
};
use ferrule_model::{TokenizerHandle, decoder::GenericDecoderRunner};
use ferrule_runtime::engine::{
    BoxedSessionInferenceEngine, ResidentKvPageAccounting, plan_resident_kv_pages,
};
#[cfg(feature = "cuda")]
use ferrule_runtime::parallel::expert::{
    ImmutableOwnedAttachment, ImmutableOwnedAttachmentConfig, ImmutableOwnedBaseline,
    NumericExpertConfig,
};
#[cfg(feature = "cuda")]
use std::{rc::Rc, sync::Arc};
const OWNER_WORKSPACE: usize = super::EXPERT_WORKSPACE;
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| provider_unsupported("EP memory estimate overflow"))
}
#[cfg(feature = "cuda")]
fn check_physical(plan: &ExpertPhysicalPlan, device: &HybridCudaDevice) -> Result<()> {
    let mut capacities = Vec::with_capacity(plan.cards().len());
    for card in plan.cards() {
        let (free_bytes, total_bytes) = if card.ordinal == device.device_ordinal() {
            device.operators().memory_info()?
        } else {
            // A temporary probe has no model weights; drop it before uploads.
            CudaOperators::new_on_device(card.ordinal)?.memory_info()?
        };
        capacities.push(PhysicalCardCapacity {
            ordinal: card.ordinal,
            free_bytes,
            limit_bytes: total_bytes,
        });
    }
    plan.check_capacity(&capacities)
}

#[cfg(feature = "cuda")]
pub(in crate::model_factory) struct PreparedRoot {
    pub(in crate::model_factory) runner: GenericDecoderRunner<HybridCudaDecoder>,
    planes: ferrule_model::decoder::StandardGqaPlanes,
    accounting: ResidentKvPageAccounting,
    pub(in crate::model_factory) host:
        Option<Arc<ferrule_model::transformer::host_experts::HostExpertCache>>,
    limits: NumericCapacityLimits,
    #[cfg(test)]
    pub(in crate::model_factory) device: HybridCudaDevice,
}

#[cfg(feature = "cuda")]
pub(in crate::model_factory) fn build_bound(
    request: ResolvedModelRequest,
    resources: BoundDecoderResources,
    tokenizer: TokenizerHandle,
    inputs: NumericProviderInputs,
) -> Result<BoxedSessionInferenceEngine> {
    let prepared = prepare_bound(&request, resources, tokenizer, inputs.clone())?;
    let engine = ferrule_runtime::engine::compose_resident_engine(
        prepared.runner,
        Box::new(prepared.planes),
        prepared.accounting,
        request.scheduler_config,
        request.driver_config,
    )?;
    Ok(Box::new(ReportedMoeEngine {
        model_label: inputs.model_label,
        engine,
        host_experts: prepared.host,
        limits: prepared.limits,
        numeric_profile: inputs.numeric_profile,
        expert_placement: inputs.placement,
        shutdown_attempted: false,
        close_reported: false,
    }))
}

// Shared constructor boundary for the resident composition and opt-in strict
// full-logit tests. No second weight image or alternate forward is created.
#[cfg(feature = "cuda")]
pub(in crate::model_factory) fn prepare_bound(
    request: &ResolvedModelRequest,
    resources: BoundDecoderResources,
    tokenizer: TokenizerHandle,
    inputs: NumericProviderInputs,
) -> Result<PreparedRoot> {
    if request.backend != ModelExecutionBackend::Cuda || request.pipeline.is_some() {
        return Err(provider_unsupported(
            "external experts require a CUDA resident root",
        ));
    }
    let placement = inputs
        .placement
        .as_ref()
        .ok_or_else(|| provider_unsupported("missing explicit EP placement"))?;
    super::bindings::validate_bound(request, &resources)?;
    crate::model_factory::parallel::bindings::validate_options(request)?;
    let limits = inputs.limits;
    let options = super::budget::decoder_options(
        &resources,
        inputs.family.clone(),
        limits,
        request.max_tensor_bytes,
        inputs.numeric_profile,
    )?;
    let planes = HybridStateSchema::from_spec(resources.spec(), request.max_layers)?
        .kv_planes(options.page_size(), options.max_positions())?;
    let accounting = ResidentKvPageAccounting::TransactionCapacity {
        page_bytes: physical_page_bytes(&planes)?,
        budget_bytes: Some(limits.kv_bytes as u64),
    };
    let kv = plan_resident_kv_pages(
        &planes,
        accounting,
        request.scheduler_config,
        request.driver_config,
    )?;
    let root = ExpertRootMemory::for_resources(&resources, &options, kv.configured_pages)?;
    let owners = OwnerPlan::new(&resources, placement, limits.max_batch_tokens)?;
    let owner_workspace = add(OWNER_WORKSPACE, owners.activation_bytes)?;
    let physical = ExpertPhysicalPlan::new(
        placement,
        root,
        &owners.weights,
        owner_workspace,
        limits.allocator_margin_bytes,
    )?;
    let owner_limit = super::budget::expert_payload_envelope(
        *owners
            .weights
            .iter()
            .max()
            .ok_or_else(|| provider_unsupported("empty EP ownership"))?,
        owners.activation_bytes,
    )?;
    let host_options = if placement.rank_backend() == PipelineRankBackend::Thread {
        host_startup_options(
            &resources,
            inputs.host_cache,
            options.max_positions(),
            request.max_tensor_bytes,
        )?
    } else {
        // Process planning derives its own aggregate/root ledger; child Boot
        // carries only shard caps and the operator's explicit headroom.
        inputs.host_cache
    };
    if host_options.mode != ferrule_model::transformer::host_experts::ExpertPrewarmMode::Full {
        return Err(provider_unsupported("numeric EP requires full prewarm"));
    }
    // Finish all metadata, wire and aggregate host checks before any child or
    // root payload allocation. Process roots must never warm routed payloads.
    #[cfg(unix)]
    let process_plan = if placement.rank_backend() == PipelineRankBackend::Process {
        Some(process::Plan::new(
            request,
            &resources,
            placement,
            &owners,
            host_options,
            inputs.numeric_profile,
        )?)
    } else {
        None
    };
    #[cfg(not(unix))]
    if placement.rank_backend() == PipelineRankBackend::Process {
        return Err(provider_unsupported("numeric process EP requires Unix"));
    }
    #[cfg(unix)]
    if let Some(plan) = &process_plan {
        plan.admit_live()?;
    }
    let device = HybridCudaDevice::new_on_device(
        placement.root_device(),
        HybridCudaMemoryBudget {
            kv_pages: kv.configured_pages,
            state_bytes: root.state_bytes,
            weight_bytes: root.weights_bytes,
            workspace_bytes: root.workspace_bytes,
        },
    )?;
    check_physical(&physical, &device)?;
    let resources = if placement.rank_backend() == PipelineRankBackend::Thread {
        resources.with_host_expert_prewarm(host_options, request.max_tensor_bytes, |progress| {
            tracing::info!(?progress, "numeric EP shared host warm progress");
        })?
    } else {
        resources
    };
    let host = resources.host_experts().cloned();
    check_physical(&physical, &device)?;
    let source_rank = owners.group.source_rank;
    let mut executor = match placement.rank_backend() {
        PipelineRankBackend::Thread => crate::model_factory::parallel::experts::start_threads(
            &owners,
            Arc::new(resources.clone()),
            placement.expert_devices().to_vec(),
            NumericExpertConfig {
                max_parameter_bytes: request.max_tensor_bytes,
                max_device_bytes: owner_limit,
                workspace_bytes: OWNER_WORKSPACE,
                dispatch_timeout: placement.rank_timeout,
            },
            inputs.numeric_profile,
        )?,
        #[cfg(unix)]
        PipelineRankBackend::Process => process_plan.expect("process admission").start()?,
        #[cfg(not(unix))]
        PipelineRankBackend::Process => unreachable!(),
    };
    #[cfg(unix)]
    let process_baseline = if placement.rank_backend() == PipelineRankBackend::Process {
        let ready = (|| -> Result<_> {
            let stats = executor.process_stats()?;
            process::check_ready(&owners, placement, &stats)?;
            tracing::info!(?stats, "numeric EP actual process Ready barrier");
            ImmutableOwnedBaseline::processes(stats).map_err(Error::from)
        })();
        Some(finish_owner_admission(
            "resident expert owner admission",
            ready,
            || executor.shutdown().map_err(Error::from),
        )?)
    } else {
        None
    };
    let ready = (|| -> Result<_> {
        let attachment_config = ImmutableOwnedAttachmentConfig::new(
            source_rank,
            request.max_layers - 1,
            placement.rank_timeout,
        )?;
        #[cfg(unix)]
        let baseline = match process_baseline {
            Some(baseline) => baseline,
            None => crate::model_factory::parallel::experts::thread_baseline(
                &mut executor,
                &owners,
                placement.expert_devices().to_vec(),
                host.clone(),
            )?,
        };
        #[cfg(not(unix))]
        let baseline = crate::model_factory::parallel::experts::thread_baseline(
            &mut executor,
            &owners,
            placement.expert_devices().to_vec(),
            host.clone(),
        )?;
        Ok((attachment_config, baseline))
    })();
    let (attachment_config, baseline) =
        finish_owner_admission("resident expert owner admission", ready, || {
            executor.shutdown().map_err(Error::from)
        })?;
    let adapter = executor.into_hybrid_cuda(Rc::clone(device.operators()))?;
    #[cfg(test)]
    let test_device = device.clone();
    let runner = GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda_with_routed_experts(
        resources,
        tokenizer,
        options,
        device,
        HybridCudaRoutedExperts::new(
            source_rank,
            Box::new(ImmutableOwnedAttachment::new(
                adapter,
                attachment_config,
                baseline,
            )),
        ),
    )?;
    tracing::info!(resident_bytes=runner.forward_executor().module().resident_parameter_bytes(),
        root_budget=?root, physical_plan=?physical, host_generation=?host.as_ref().map(|host| host.generation()),
        "numeric EP root and all expert owners ready; root-only KV authority");
    Ok(PreparedRoot {
        runner,
        planes,
        accounting,
        host,
        limits,
        #[cfg(test)]
        device: test_device,
    })
}

#[cfg(feature = "cuda")]
pub(in crate::model_factory) fn build_local(
    request: ResolvedModelRequest,
    resources: BoundDecoderResources,
    inputs: NumericProviderInputs,
    admission: super::budget::BoundMemoryPlan,
) -> Result<BoxedSessionInferenceEngine> {
    let limits = inputs.limits;
    let planes = HybridStateSchema::from_spec(resources.spec(), request.max_layers)?.kv_planes(
        admission.options.page_size(),
        admission.options.max_positions(),
    )?;
    let accounting = ResidentKvPageAccounting::TransactionCapacity {
        page_bytes: physical_page_bytes(&planes)?,
        budget_bytes: Some(limits.kv_bytes as u64),
    };
    log_kv_plan(
        "numeric provider numeric FP8",
        &planes,
        accounting,
        &request,
    )?;
    let host_options = host_startup_options(
        &resources,
        inputs.host_cache,
        admission.options.max_positions(),
        request.max_tensor_bytes,
    )?;
    tracing::info!(
        ?host_options,
        "numeric provider host expert startup admission (readiness gated)"
    );
    let resources =
        resources.with_host_expert_prewarm(host_options, request.max_tensor_bytes, |progress| {
            tracing::info!(
                read_bytes = progress.read_bytes,
                total_bytes = progress.total_bytes,
                completed_parameters = progress.completed_parameters,
                total_parameters = progress.total_parameters,
                elapsed_seconds = progress.elapsed.as_secs_f64(),
                "host expert prewarm progress"
            );
        })?;
    let host_experts = resources.host_experts().cloned();
    if let Some(cache) = &host_experts {
        tracing::info!(generation = cache.generation(), stats = ?cache.stats(), "all routed host experts ready (compressed immutable proofs)");
    } else {
        tracing::warn!(
            "explicit lazy expert mode: routed weights are NOT prewarmed; new prompts may read NAS"
        );
    }
    let device = HybridCudaDevice::new_on_device(0, admission.budget)?;
    let (free_device_bytes, total_device_bytes) = device.operators().memory_info()?;
    admission.check_available_device_bytes(free_device_bytes)?;
    tracing::info!(
        free_device_bytes,
        total_device_bytes,
        required_device_bytes = admission.total_device_bytes,
        "numeric provider physical CUDA admission"
    );
    let tokenizer = TokenizerHandle::load(&request.model_path)?;
    let runner = GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda(
        resources,
        tokenizer,
        admission.options,
        device,
    )?;
    tracing::info!(
        resident_bytes = runner.forward_executor().module().resident_parameter_bytes(),
        cache = ?runner.forward_executor().module().expert_cache_stats(),
        workspace = ?runner.forward_executor().module().numeric_workspace_usage(),
        precision = ?runner.forward_executor().module().numeric_fp8_precision(),
        "numeric provider runner prepared: compressed FP8 storage, bounded cache"
    );
    let engine = ferrule_runtime::engine::compose_resident_engine(
        runner,
        Box::new(planes),
        accounting,
        request.scheduler_config,
        request.driver_config,
    )?;
    Ok(Box::new(ReportedMoeEngine {
        model_label: inputs.model_label,
        engine,
        host_experts,
        limits,
        numeric_profile: inputs.numeric_profile,
        expert_placement: None,
        shutdown_attempted: false,
        close_reported: false,
    }))
}
