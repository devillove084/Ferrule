use super::*;
/// Only MoE has compressed expert cache policy; dense Qwen3.5 cannot carry it.
#[derive(Debug, Clone)]
pub(in crate::model_factory) struct Qwen35MoeOptions {
    pub(in crate::model_factory) device_capacity: Option<Qwen35MoeCapacityLimits>,
    pub(in crate::model_factory) numeric_profile: NumericFp8Profile,
    pub(in crate::model_factory) host_cache:
        ferrule_model::transformer::host_experts::HostExpertCacheOptions,
}

/// Metadata policy only. Capacity and host startup ledgers are still derived by
/// the existing planners, and physical admission remains on the build owner.
#[derive(Debug, Clone)]
pub(in crate::model_factory) enum Qwen35ResolvedAdapter {
    Dense,
    Moe {
        options: Box<Qwen35MoeOptions>,
        expert_placement: Option<Qwen35ExpertPlacement>,
    },
}

impl Qwen35ResolvedAdapter {
    pub(in crate::model_factory) fn moe(&self) -> Option<&Qwen35MoeOptions> {
        match self {
            Self::Dense => None,
            Self::Moe { options, .. } => Some(options),
        }
    }

    pub(in crate::model_factory) fn expert_placement(&self) -> Option<&Qwen35ExpertPlacement> {
        match self {
            Self::Dense => None,
            Self::Moe {
                expert_placement, ..
            } => expert_placement.as_ref(),
        }
    }

    pub(in crate::model_factory) fn set_expert_placement(
        &mut self,
        placement: Qwen35ExpertPlacement,
    ) {
        let Self::Moe {
            expert_placement, ..
        } = self
        else {
            unreachable!("dedicated expert placement requires validated Qwen35Moe");
        };
        *expert_placement = Some(placement);
    }
}

/// Serving defaults belong to the family adapter; the CLI only preserves
/// explicit inputs. Direct legacy factory callers retain their old contract.
pub(in crate::model_factory) fn resolve_serving_options(
    family: &ModelFamily,
    serving: &ServingModelOptions,
    options: &mut ModelFactoryOptions,
) -> Result<()> {
    if !matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe) {
        return Ok(());
    }
    let moe = *family == ModelFamily::Qwen35Moe;
    if (!moe
        && (serving.expert_host_cache_entries.is_some()
            || serving.expert_host_cache_mebibytes.is_some()))
        || serving.expert_pinned_cache_entries.is_some()
        || serving.expert_pinned_cache_mebibytes.is_some()
        || serving.moe_hotset_experts.is_some()
    {
        return Err(invalid(
            "Qwen3.5 does not implement explicit MoE host/pinned expert cache/hotset options",
        ));
    }
    if moe
        && (serving.expert_host_cache_entries.is_some()
            || serving.expert_host_cache_mebibytes.is_some())
    {
        let mut host = options.qwen35_host_cache.unwrap_or_default();
        if let Some(n) = serving.expert_host_cache_entries {
            host.max_experts = n;
        }
        if let Some(n) = serving.expert_host_cache_mebibytes {
            host.max_bytes = n
                .checked_mul(1 << 20)
                .ok_or_else(|| invalid("host expert MiB budget overflows bytes"))?;
        }
        host.validate()?;
        options.qwen35_host_cache = Some(host);
    }
    options.driver_config.enable_native_proposals = false;
    options.expert_cache = ExpertCacheOptions::default();
    options.max_tensor_mebibytes = serving.max_tensor_mebibytes.unwrap_or(1024);
    Ok(())
}

/// Adapt the old flat DTO once, preserving validation order and error sources.
#[cfg(test)]
pub(in crate::model_factory) fn resolve_family_options(
    family: &ModelFamily,
    options: &ModelFactoryOptions,
) -> Result<ResolvedFamilyOptions> {
    resolve_family_options_with_model_bf16(family, options, false)
}

pub(in crate::model_factory) fn resolve_family_options_with_model_bf16(
    family: &ModelFamily,
    options: &ModelFactoryOptions,
    model_bf16: bool,
) -> Result<ResolvedFamilyOptions> {
    if options.numeric_fp8_profile.is_some() && *family != ModelFamily::Qwen35Moe {
        return Err(invalid(
            "--numeric-fp8-precision requires Qwen3.5 MoE FP8 CUDA; no fallback",
        ));
    }
    if options.qwen35_moe_capacity.is_some() && *family != ModelFamily::Qwen35Moe {
        return Err(invalid(
            "CUDA expert device cache options require Qwen3.5 MoE FP8",
        ));
    }
    if let Some(limits) = options.qwen35_moe_capacity {
        if model_bf16 {
            if limits.max_experts == 0 || limits.max_bytes <= limits.scratch_bytes {
                return Err(invalid(
                    "numeric BF16 requires a bounded cache beyond reserved scratch",
                ));
            }
        } else {
            limits.expert_cache_policy()?;
        }
    }
    if let Some(host) = options.qwen35_host_cache {
        if *family != ModelFamily::Qwen35Moe {
            return Err(invalid("expert prewarm options require Qwen3.5 MoE FP8"));
        }
        host.validate()?;
    }
    Ok(match family {
        ModelFamily::Qwen35 => ResolvedFamilyOptions::Qwen35(Qwen35ResolvedAdapter::Dense),
        ModelFamily::Qwen35Moe => ResolvedFamilyOptions::Qwen35(Qwen35ResolvedAdapter::Moe {
            options: Box::new(Qwen35MoeOptions {
                device_capacity: options.qwen35_moe_capacity,
                numeric_profile: options.numeric_fp8_profile.unwrap_or_default(),
                host_cache: if model_bf16 {
                    ferrule_model::transformer::host_experts::HostExpertCacheOptions {
                        mode: ferrule_model::transformer::host_experts::ExpertPrewarmMode::Lazy,
                        ..Default::default()
                    }
                } else {
                    options.qwen35_host_cache.unwrap_or_default()
                },
            }),
            expert_placement: None,
        }),
        _ => ResolvedFamilyOptions::Generic,
    })
}
pub(in crate::model_factory) const ADJUSTMENT_REASON: &str =
    "Qwen3.5 token-serial workspace: batch <= min(32, context); prefill <= effective batch";

pub(in crate::model_factory) fn adapt_options(
    descriptor: &ModelDescriptor,
    options: &mut ModelFactoryOptions,
    execution: ResolvedExecution,
) -> Result<ResolvedFamilyOptions> {
    let payload = resolve_family_options_with_model_bf16(
        &descriptor.spec.family,
        options,
        execution.numeric_bf16_precision.is_some(),
    )?;
    validate_options(descriptor, options)?;
    Ok(payload)
}

pub(in crate::model_factory) fn finish_options(plan: &mut ResidentModelBuildPlan) -> Result<()> {
    let Some(adapter) = plan.request.family_options.qwen35() else {
        return Ok(());
    };
    let placement = adapter.expert_placement().cloned();
    let Some(moe) = adapter.moe() else {
        return Ok(());
    };
    let host_cache = moe.host_cache;
    let numeric = moe.numeric_profile;
    // Preserve early unit/profile admission even when the selected owner topology
    // does not use the local routed cache. The existing formula is unchanged.
    let limits = capacity_limits(&plan.request)?;
    let local = placement.is_none() && plan.request.pipeline.is_none();
    plan.effective_options.qwen35_moe_capacity = local.then_some(limits);
    plan.effective_options.qwen35_host_cache = Some(host_cache);
    if local && plan.request.execution.numeric_bf16_precision.is_none() {
        plan.backend_profile = numeric.resident_backend_name();
    }
    let requested = &plan.requested_options;
    let effective = &plan.effective_options;
    if requested.qwen35_moe_capacity != effective.qwen35_moe_capacity {
        plan.provider_adjustments.push("qwen35_moe_capacity", format!("{:?}", requested.qwen35_moe_capacity), format!("{:?}", effective.qwen35_moe_capacity),
            "single-device policy defaults; context, concurrency, effective batch and KV budget override capacity hints; dedicated EP has no local routed cache");
    }
    if requested.qwen35_host_cache != effective.qwen35_host_cache {
        plan.provider_adjustments.push(
            "qwen35_host_cache",
            format!("{:?}", requested.qwen35_host_cache),
            format!("{:?}", effective.qwen35_host_cache),
            "default full host prewarm policy; source-dependent startup ledger remains owner-local",
        );
    }
    Ok(())
}

/// Capability failures retain a typed model source; never fall back to CPU.
pub(in crate::model_factory) fn moe_unavailable(selection: BackendSelection) -> Error {
    let (operator, reason) = if selection == BackendSelection::Cpu {
        (
            "qwen35_moe_cpu",
            "CPU MoE numeric FP8 execution is not implemented",
        )
    } else {
        (
            "qwen35_moe_cuda",
            "Qwen3.5-35B requires CUDA compiled with the cuda feature; CPU fallback is unsupported",
        )
    };
    ferrule_common::Error::ModelSource {
        source: Box::new(ferrule_model::transformer::UnsupportedOperator::new(
            operator, reason,
        )),
    }
    .into()
}

pub(in crate::model_factory) fn ep_unsupported(reason: impl Into<String>) -> Error {
    ferrule_common::Error::ModelSource {
        source: Box::new(ferrule_model::transformer::UnsupportedOperator::new(
            "qwen35_moe_expert_parallel",
            reason,
        )),
    }
    .into()
}
pub(in crate::model_factory) fn resolve(
    descriptor: &ModelDescriptor,
    selection: BackendSelection,
) -> Result<ResolvedModelBackend> {
    if descriptor.spec.weight_source != WeightSource::Safetensors {
        return Err(invalid("Qwen3.5 requires Hugging Face safetensors"));
    }
    if descriptor.spec.family == ModelFamily::Qwen35Moe {
        if selection == BackendSelection::Cpu || !cfg!(feature = "cuda") {
            return Err(moe_unavailable(selection));
        }
        return Ok(ResolvedModelBackend {
            family: ModelFamily::Qwen35Moe,
            model_name: "qwen3.5-35b-a3b-fp8",
            backend: ModelExecutionBackend::Cuda,
            backend_profile: Qwen35MoeCapacityPlan::PLANNED_BACKEND_PROFILE,
            default_chat_template: ChatTemplate::Qwen35,
        });
    }
    let backend =
        Option::<ModelExecutionBackend>::from(selection).unwrap_or(ModelExecutionBackend::Cpu);
    let profile = match backend {
        ModelExecutionBackend::Cpu => "cpu-hybrid-f32-qwen35-0.8b",
        ModelExecutionBackend::Cuda if cfg!(feature = "cuda") => "cuda-hybrid-f32-qwen35-0.8b",
        _ => return Err(invalid("Qwen3.5 CUDA requires the 'cuda' feature")),
    };
    Ok(ResolvedModelBackend {
        family: ModelFamily::Qwen35,
        model_name: "qwen3.5-0.8b",
        backend,
        backend_profile: profile,
        default_chat_template: ChatTemplate::Qwen35,
    })
}

pub(in crate::model_factory) fn validate_options(
    descriptor: &ModelDescriptor,
    options: &mut ModelFactoryOptions,
) -> Result<()> {
    if options
        .max_layers
        .is_some_and(|n| Some(n) != descriptor.spec.num_layers)
    {
        return Err(invalid(
            "Qwen3.5 requires all checkpoint layers; partial --max-layers is unsupported",
        ));
    }
    if options.expert_cache != ExpertCacheOptions::default() || options.moe_hotset_experts != 0 {
        return Err(invalid(
            "Qwen3.5 does not implement host/pinned expert_cache or moe_hotset_experts overrides",
        ));
    }
    if options.driver_config.enable_native_proposals {
        return Err(invalid(
            "Qwen3.5 speculative/native proposals are unsupported; disable enable_native_proposals",
        ));
    }
    if descriptor.spec.family == ModelFamily::Qwen35Moe {
        if options.expert_reader_max_tensor_mebibytes != 64
            || options.output_head_chunk_rows != 4096
        {
            return Err(invalid(
                "35B does not consume separate expert-reader or output-head chunk overrides",
            ));
        }
        if options.driver_config.ctx_size > 262144 {
            return Err(invalid(
                "35B context exceeds the strict profile limit 262144",
            ));
        }
        if options.max_tensor_mebibytes < 970 {
            return Err(invalid(
                "35B embedding/head need 970 MiB source reads; use --max-tensor-mb 1024",
            ));
        }
    }
    let scheduler = &mut options.scheduler_config;
    if scheduler.prefix_cache_capacity_pages != 0 {
        return Err(invalid(
            "Qwen3.5 prefix cache / partial retain is unsupported",
        ));
    }
    if scheduler.max_active_sequences == 0
        || scheduler.max_batch_tokens == 0
        || scheduler.prefill_chunk_size == 0
        || options.driver_config.ctx_size == 0
    {
        return Err(invalid(
            "Qwen3.5 requires nonzero bounded sequence, batch and context capacities",
        ));
    }
    // Token-serial execution: bound packed workspace instead of admitting 512
    // full-vocabulary rows by default (over 16 GiB for this vocabulary).
    scheduler.max_batch_tokens = scheduler
        .max_batch_tokens
        .min(32)
        .min(options.driver_config.ctx_size);
    scheduler.prefill_chunk_size = scheduler.prefill_chunk_size.min(scheduler.max_batch_tokens);
    Ok(())
}
