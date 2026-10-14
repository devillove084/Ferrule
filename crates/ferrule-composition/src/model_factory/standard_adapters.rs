//! Built-in checkpoint adapters. Shared planning only consumes their catalog hooks.
use super::*;

pub(super) fn resolve_qwen_backend(
    descriptor: &ModelDescriptor,
    selection: BackendSelection,
) -> Result<ResolvedModelBackend> {
    if descriptor.spec.weight_source != WeightSource::Safetensors {
        return Err(unsupported_model(
            descriptor,
            "Qwen3-MoE runtime loading requires Hugging Face safetensors",
        ));
    }
    let backend =
        Option::<ModelExecutionBackend>::from(selection).unwrap_or(ModelExecutionBackend::Cpu);
    if backend != ModelExecutionBackend::Cpu {
        return Err(unsupported_backend(descriptor, backend, "cpu"));
    }
    Ok(ResolvedModelBackend {
        family: ModelFamily::QwenMoe,
        model_name: "qwen3-moe",
        backend,
        backend_profile: "cpu-standard-decoder",
        default_chat_template: ChatTemplate::Qwen3,
    })
}

pub(super) fn resolve_dense_qwen3_backend(
    descriptor: &ModelDescriptor,
    selection: BackendSelection,
) -> Result<ResolvedModelBackend> {
    if descriptor.spec.weight_source != WeightSource::Safetensors {
        return Err(unsupported_model(
            descriptor,
            "dense Qwen3 runtime loading requires Hugging Face safetensors",
        ));
    }
    let backend =
        Option::<ModelExecutionBackend>::from(selection).unwrap_or(ModelExecutionBackend::Cpu);
    if backend != ModelExecutionBackend::Cpu {
        return Err(unsupported_backend(descriptor, backend, "cpu"));
    }
    Ok(ResolvedModelBackend {
        family: ModelFamily::Qwen3,
        model_name: "qwen3-dense",
        backend,
        backend_profile: "cpu-standard-decoder",
        default_chat_template: ChatTemplate::Qwen3,
    })
}

#[cfg(feature = "cuda")]
pub(super) fn resolve_deepseek_backend(
    descriptor: &ModelDescriptor,
    selection: BackendSelection,
) -> Result<ResolvedModelBackend> {
    if descriptor.spec.weight_source != WeightSource::Safetensors {
        return Err(unsupported_model(
            descriptor,
            "DeepSeek-V4 runtime loading requires Hugging Face safetensors",
        ));
    }
    let backend =
        Option::<ModelExecutionBackend>::from(selection).unwrap_or(ModelExecutionBackend::Cuda);
    if backend != ModelExecutionBackend::Cuda {
        return Err(unsupported_backend(descriptor, backend, "cuda"));
    }
    Ok(ResolvedModelBackend {
        family: ModelFamily::DeepSeekV4,
        model_name: "deepseek-v4",
        backend,
        backend_profile: "cuda",
        default_chat_template: ChatTemplate::DeepSeekV4,
    })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn resolve_deepseek_backend(
    descriptor: &ModelDescriptor,
    _selection: BackendSelection,
) -> Result<ResolvedModelBackend> {
    if descriptor.spec.weight_source != WeightSource::Safetensors {
        return Err(unsupported_model(
            descriptor,
            "DeepSeek-V4 runtime loading requires Hugging Face safetensors",
        ));
    }
    Err(unsupported_model(
        descriptor,
        "DeepSeek-V4 requires CUDA, but this build was compiled without the 'cuda' feature",
    ))
}

pub(super) fn qwen_prepare_options(request: &ResolvedModelRequest) -> Qwen3MoePrepareOptions {
    let defaults = Qwen3MoePrepareOptions::default();
    Qwen3MoePrepareOptions {
        max_dense_tensor_bytes: request
            .max_tensor_bytes
            .max(defaults.max_dense_tensor_bytes),
        max_expert_tensor_bytes: request.expert_reader_max_tensor_bytes,
        max_layers: Some(request.max_layers),
        ..defaults
    }
}

pub(super) fn build_qwen(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    let request = request.resolved;
    let prepare = qwen_prepare_options(&request);
    let adapter = Qwen3MoeAdapter::load_hf_with_options_and_backend(
        &request.model_path,
        prepare.max_dense_tensor_bytes,
        prepare,
        request.backend,
    )?;
    let schema = adapter.kv_schema().clone();
    let accounting = kv_accounting(
        request.kv_cache_bytes,
        request
            .kv_cache_bytes
            .map(|_| physical_page_bytes(&schema))
            .transpose()?,
    );
    log_kv_plan("CPU Qwen3-MoE", &schema, accounting, &request)?;
    let decoder = adapter.into_decoder(
        request.driver_config.ctx_size,
        request.scheduler_config.max_batch_tokens.max(1),
        request.scheduler_config.max_active_sequences.max(1),
    )?;
    build_resident_engine(
        decoder,
        Box::new(schema),
        accounting,
        request.scheduler_config,
        request.driver_config,
    )
}

pub(super) fn build_dense_qwen3(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    let request = request.resolved;
    let adapter = Qwen3DenseAdapter::load_hf_with_options_and_backend(
        &request.model_path,
        request.max_tensor_bytes,
        Qwen3DensePrepareOptions {
            max_dense_tensor_bytes: request.max_tensor_bytes,
            max_layers: Some(request.max_layers),
            ..Qwen3DensePrepareOptions::default()
        },
        request.backend,
    )?;
    let schema = adapter.kv_schema().clone();
    let accounting = kv_accounting(
        request.kv_cache_bytes,
        request
            .kv_cache_bytes
            .map(|_| physical_page_bytes(&schema))
            .transpose()?,
    );
    log_kv_plan("CPU Qwen3 dense", &schema, accounting, &request)?;
    let decoder = adapter.into_decoder(
        request.driver_config.ctx_size,
        request.scheduler_config.max_batch_tokens.max(1),
        request.scheduler_config.max_active_sequences.max(1),
    )?;
    build_resident_engine(
        decoder,
        Box::new(schema),
        accounting,
        request.scheduler_config,
        request.driver_config,
    )
}

#[cfg(feature = "cuda")]
pub(super) fn deepseek_prepare_options(
    request: &ResolvedModelRequest,
) -> Result<DeepSeekV4PrepareOptions> {
    let defaults = DeepSeekV4PrepareOptions::default();
    let reserved_device_bytes = match request.kv_cache_bytes {
        Some(bytes) => defaults
            .reserved_device_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::InvalidRequest {
                message: "device residency reserve overflow".into(),
            })?,
        None => defaults.reserved_device_bytes,
    };
    Ok(DeepSeekV4PrepareOptions {
        max_layers: request.max_layers,
        output_head_chunk_rows: request.output_head_chunk_rows,
        expert_reader_max_tensor_bytes: request.expert_reader_max_tensor_bytes,
        expert_memory_policy: request.expert_memory_policy,
        moe_hotset_experts: request.moe_hotset_experts,
        reserved_device_bytes,
    })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn build_deepseek(_request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    Err(Error::InvalidRequest {
        message: "DeepSeek-V4 requires a CUDA-enabled runtime build".into(),
    })
}

#[cfg(feature = "cuda")]
pub(super) fn build_deepseek(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    let request = request.resolved;
    let options = deepseek_prepare_options(&request)?;
    let decoder = DeepSeekV4Adapter::load_hf_with_options_and_backend(
        &request.model_path,
        request.max_tensor_bytes,
        options,
        request.backend,
    )?;
    let schema = decoder.resources().kv_layout().clone();
    let page_bytes = request
        .kv_cache_bytes
        .map(|_| physical_page_bytes(&schema))
        .transpose()?;
    let accounting = kv_accounting(request.kv_cache_bytes, page_bytes);
    log_kv_plan("CUDA DeepSeek-V4", &schema, accounting, &request)?;
    build_resident_engine(
        decoder,
        Box::new(schema),
        accounting,
        request.scheduler_config,
        request.driver_config,
    )
}

pub(super) fn generic_options(
    _: &ModelDescriptor,
    options: &mut ModelFactoryOptions,
    _: ResolvedExecution,
) -> Result<ResolvedFamilyOptions> {
    // Compatibility DTO validation belongs at the selected option-policy edge.
    if options.numeric_fp8_profile.is_some() {
        return Err(invalid(
            "--numeric-fp8-precision requires Qwen3.5 MoE FP8 CUDA; no fallback",
        ));
    }
    if options.qwen35_moe_capacity.is_some() {
        return Err(invalid(
            "CUDA expert device cache options require Qwen3.5 MoE FP8",
        ));
    }
    if options.qwen35_host_cache.is_some() {
        return Err(invalid("expert prewarm options require Qwen3.5 MoE FP8"));
    }
    Ok(ResolvedFamilyOptions::Generic)
}
