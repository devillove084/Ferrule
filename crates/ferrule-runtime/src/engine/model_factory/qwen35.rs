//! Text-only hybrid composition on the ordinary resident owner/transaction path.
use super::*;
use ferrule_model::decoder::{
    GenericDecoderOptions, GenericDecoderRunner, HybridCpuDecoder, HybridStateSchema,
};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::models::qwen35::Qwen35Adapter;
use ferrule_model::{TokenizerHandle, transformer::BoundDecoderResources};

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidRequest {
        message: message.into(),
    }
}

pub(super) fn resolve(
    descriptor: &ModelDescriptor,
    selection: BackendSelection,
) -> Result<ResolvedModelBackend> {
    if descriptor.spec.weight_source != WeightSource::Safetensors {
        return Err(invalid("Qwen3.5 requires Hugging Face safetensors"));
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

pub(super) fn validate_options(
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
            "dense Qwen3.5 does not implement expert_cache or moe_hotset_experts",
        ));
    }
    if options.driver_config.enable_native_proposals {
        return Err(invalid(
            "Qwen3.5 speculative/native proposals are unsupported; disable enable_native_proposals",
        ));
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

pub(super) fn build(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    if request.pipeline.is_some() {
        return Err(invalid("Qwen3.5 hybrid TP/PP is unsupported"));
    }
    // Resources, tokenizer and CUDA state are created on the consuming owner.
    // The family adapter's load-with-backend helper is deliberately CPU-only.
    let (_, resources) = Qwen35Adapter::bind_hf_metadata(&request.model_path)?;
    let tokenizer = TokenizerHandle::load(&request.model_path)?;
    build_bound(request, resources, tokenizer)
}

fn build_bound(
    request: ModelBuildRequest,
    resources: BoundDecoderResources,
    tokenizer: TokenizerHandle,
) -> Result<BoxedSessionInferenceEngine> {
    let options = GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Qwen35,
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
    log_kv_plan("Qwen3.5 hybrid", &schema, accounting, &request)?;
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
            let pages = plan_resident_kv_pages(
                &schema,
                accounting,
                request.scheduler_config,
                request.driver_config,
            )?
            .configured_pages;
            let estimate = HybridCudaMemoryEstimate::for_resources(&resources, &options, pages)?;
            // Default + committed sequences + transaction working copies.
            // Forks consume this same hard cap; prefix cache is disabled.
            let state_bytes = request
                .scheduler_config
                .max_active_sequences
                .checked_mul(2)
                .and_then(|n| n.checked_add(1))
                .and_then(|n| n.checked_mul(estimate.per_sequence_state_bytes))
                .ok_or_else(|| invalid("Qwen3.5 CUDA recurrent state budget overflow"))?;
            let budget = HybridCudaMemoryBudget {
                kv_pages: pages,
                state_bytes,
                weight_bytes: estimate.weight_bytes_upper_bound,
                workspace_bytes: estimate.workspace_bytes_upper_bound,
            };
            tracing::info!(
                ?estimate,
                ?budget,
                "Qwen3.5 CUDA owner admission (text-only, no prefix/speculation/TP/PP)"
            );
            let device = HybridCudaDevice::new_on_device(0, budget)?;
            let runner = GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda(
                resources, tokenizer, options, device,
            )?;
            build_resident_engine(
                runner,
                Box::new(schema),
                accounting,
                request.scheduler_config,
                request.driver_config,
            )
        }
        #[cfg(not(feature = "cuda"))]
        ModelExecutionBackend::Cuda => Err(invalid("Qwen3.5 CUDA requires the 'cuda' feature")),
    }
}

#[cfg(test)]
#[path = "../../../tests/support/hybrid_fixture.rs"]
mod fixture;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GenerateRequest, InferenceShutdownProgress, RequestId, ResidentDriverStep, SessionId,
    };

    fn run(engine: &mut BoxedSessionInferenceEngine, id: u64) -> Vec<u32> {
        engine
            .try_submit(GenerateRequest {
                id: RequestId(id),
                session_id: Some(SessionId(7)),
                prompt_tokens: vec![1, 3, 2],
                max_new_tokens: 2,
                stop: vec![],
                ignore_eos: false,
            })
            .unwrap();
        let mut tokens = vec![];
        for _ in 0..100 {
            let step = engine
                .step(&mut |event| {
                    tokens.push(event.token);
                    Ok(())
                })
                .unwrap();
            assert!(!matches!(step, ResidentDriverStep::Blocked));
            assert!(engine.drain_failed().is_empty());
            if !engine.drain_finished().is_empty() {
                return tokens;
            }
        }
        panic!("resident hybrid did not finish");
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires NAS Qwen3.5-0.8B headers; no payload or CUDA initialization"]
    fn qwen35_real_header_budget_uses_compact_kv_and_all_recurrent_layers() {
        use ferrule_model::decoder::HybridCudaMemoryEstimate;
        let path = std::env::var_os("FERRULE_QWEN35_08B_DIR")
            .unwrap_or_else(|| "/mnt/nas1/hf/Qwen3.5-0.8B".into());
        let (_, resources) = Qwen35Adapter::bind_hf_metadata(std::path::Path::new(&path)).unwrap();
        let schema = HybridStateSchema::from_spec(resources.spec(), 24)
            .unwrap()
            .kv_planes(16, 1024)
            .unwrap();
        assert_eq!(schema.planes()[0].layer_count, 6);
        assert_eq!(physical_page_bytes(&schema).unwrap(), 393216);
        let options = GenericDecoderOptions::standard_cpu(
            resources.spec(),
            ModelFamily::Qwen35,
            WeightSource::Safetensors,
            16,
            1024,
            32,
            4,
            1 << 30,
            ExecutionPrecisionPolicy::f32(),
        )
        .unwrap();
        let estimate = HybridCudaMemoryEstimate::for_resources(&resources, &options, 512).unwrap();
        assert_eq!(estimate.per_sequence_state_bytes, 20_643_840);
        assert_eq!(estimate.kv_bytes, 192 * 1024 * 1024);
        assert_eq!(estimate.weight_bytes_upper_bound, 8_054_954_496);
        assert_eq!(estimate.workspace_bytes_upper_bound, 1_018_167_296);
        assert_eq!(
            (2 * options.capabilities().max_sequences + 1) * estimate.per_sequence_state_bytes,
            185_794_560
        );
    }

    #[test]
    fn hermetic_cpu_factory_composition_generates_cancels_resets_and_shuts_down() {
        let fixture = fixture::Fixture::new();
        let descriptor = super::super::tests::descriptor(ModelFamily::Qwen35, "tiny-hybrid", 4);
        let mut options = super::super::tests::options();
        options.driver_config.ctx_size = 32;
        options.driver_config.enable_native_proposals = false;
        options.scheduler_config.max_batch_tokens = 4;
        options.scheduler_config.prefill_chunk_size = 2;
        let entry = resolve(&descriptor, BackendSelection::Cpu).unwrap();
        let request = configure_request(&descriptor, entry, options).unwrap();
        let mut engine = build_bound(
            request,
            fixture.resources(),
            TokenizerHandle::load(&fixture.dir).unwrap(),
        )
        .unwrap();
        engine.retain_session(SessionId(7)).unwrap();
        let first = run(&mut engine, 1);
        assert_eq!(first.len(), 2);
        assert_eq!(first[0], 0, "HF oracle greedy token after [1, 3, 2]");
        assert!(engine.retained_session_position(SessionId(7)).unwrap() > 3);
        engine.reset_session(SessionId(7)).unwrap();
        assert_eq!(engine.retained_session_position(SessionId(7)), Some(0));
        assert_eq!(run(&mut engine, 2), first);
        engine.reset_session(SessionId(7)).unwrap();
        engine
            .try_submit(GenerateRequest {
                id: RequestId(3),
                session_id: Some(SessionId(7)),
                prompt_tokens: vec![1, 3, 2],
                max_new_tokens: 16,
                stop: vec![],
                ignore_eos: true,
            })
            .unwrap();
        engine.step(&mut |_| Ok(())).unwrap();
        engine.cancel_request(RequestId(3)).unwrap();
        assert_eq!(engine.drain_cancelled().len(), 1);
        engine.reset_session(SessionId(7)).unwrap();
        assert_eq!(run(&mut engine, 4), first);
        assert_eq!(
            engine.shutdown().unwrap(),
            InferenceShutdownProgress::Complete
        );
    }
}
