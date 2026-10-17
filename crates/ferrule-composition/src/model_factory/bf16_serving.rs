//! Native BF16 dense Qwen3 through the existing single-stage owner and journal.
use super::*;
use ferrule_model::transformer::{ExpertReadPipelineOptions, NumericFp8Precision};
use ferrule_runtime::engine::KvExecutionPolicy;

/// Model boundary/storage precision, independent of numeric FP8 operand arithmetic.
/// Omission preserves the existing family/backend defaults (including CPU reference).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ModelPrecision {
    #[default]
    F32,
    Bf16,
}

fn invalid(message: &str) -> Error {
    Error::InvalidRequest {
        message: format!("dense BF16 serving: {message}"),
    }
}

pub(super) fn validate_dense_request(
    backend: BackendSelection,
    options: &ModelFactoryOptions,
    serving: &ServingModelOptions,
) -> Result<()> {
    if backend != BackendSelection::Cuda {
        return Err(invalid(
            "requires explicit CUDA; no CPU or F32 storage fallback",
        ));
    }
    if serving.kv_execution_policy != KvExecutionPolicy::Resident {
        return Err(invalid(
            "typed BF16 KV is mutually exclusive with F32 HostKV/IdleReclaim",
        ));
    }
    if serving.engine == ServingEngineSelection::Resident {
        return Err(invalid(
            "uses the existing single-stage standard pipeline; choose engine auto or pipeline",
        ));
    }
    if options.numeric_fp8_profile.is_some()
        || options.qwen35_moe_capacity.is_some()
        || options.qwen35_host_cache.is_some()
        || options.moe_hotset_experts != 0
        || serving.moe_hotset_experts.is_some()
        || serving.expert_host_cache_entries.is_some()
        || serving.expert_host_cache_mebibytes.is_some()
        || serving.expert_pinned_cache_entries.is_some()
        || serving.expert_pinned_cache_mebibytes.is_some()
    {
        return Err(invalid(
            "numeric FP8/W8 and expert policies are not model BF16 precision",
        ));
    }
    validate_topology(&serving.parallel.clone().unwrap_or_default())
}

pub(super) fn select_dense(
    entry: &ModelImplementation,
    backend: BackendSelection,
    request: &SelectionRequest,
    options: &ModelFactoryOptions,
) -> Result<ResolvedStrategy> {
    validate_dense_request(backend, options, &request.serving)?;
    let pipeline = request.serving.parallel.clone().unwrap_or_default();
    let mut execution = pipeline::standard_execution(ModelExecutionBackend::Cuda, &pipeline);
    execution.support.storage = StorageEncoding::NativeBf16;
    execution.support.arithmetic = ArithmeticPrecision::Bf16Compatibility;
    execution.support.static_constraints = "native BF16 dense Qwen3; full-depth CUDA TP1/PP1/EP1/thread; BF16 weights/activation/residual/KV/device logits; F32 GEMM accumulation and Q/attention-output scratch; BF16 LM-head logits expand to F32 host export; F32 RoPE frequencies; no GDN/MoE/HostKV/IdleReclaim/numeric FP8";
    execution.topology_policy = Some(LOCAL_TOPOLOGY);
    execution.conversion =
        PipelineConversion::Frozen("native BF16 topology/precision is frozen by prepare_serving");
    Ok(ResolvedStrategy {
        execution,
        backend: ModelExecutionBackend::Cuda,
        catalog_backend: BackendSelection::Cpu,
        resolve: entry.resolve,
        policy: &DENSE_OPTIONS,
        parallel: Some(pipeline),
    })
}

const DENSE_OPTIONS: ModelOptionPolicy = ModelOptionPolicy {
    serving: common_serving_options,
    parallel: pipeline::attach_dense,
    adapt: generic_options,
    finish: |plan| {
        plan.dense_bf16_memory_plan()?;
        Ok(())
    },
};

const LOCAL_TOPOLOGY: execution_policy::TopologyPolicy = execution_policy::TopologyPolicy {
    layout: execution_policy::OwnerLayout::SingleOwner,
    execution: execution_policy::ExecutionCapabilities {
        ranks: execution_policy::RankCapability::ThreadOnly,
        tensor: execution_policy::TensorCapability::Unsharded,
    },
    placement: execution_policy::PlacementPolicy {
        selection: execution_policy::DeviceSelection::AutomaticOrExplicit,
        sharing: execution_policy::DeviceSharing::Distinct,
    },
};

fn validate_topology(options: &PipelineBuildOptions) -> Result<()> {
    options.validate_policy(ModelExecutionBackend::Cuda, LOCAL_TOPOLOGY.layout, LOCAL_TOPOLOGY.execution, LOCAL_TOPOLOGY.placement)
        .map_err(|e| invalid(&format!("requires PP/TP/EP/DP/SP/CP=1, one device, thread ranks, no timeout override/restart: {e}")))?;
    Ok(())
}

impl ResidentModelBuildPlan {
    /// Rebind only metadata and quantify native weights, typed KV, Rows and F32
    /// work buffers. No payload reads, CUDA context, or VRAM reservation.
    pub fn dense_bf16_memory_plan(
        &self,
    ) -> Result<Option<ferrule_runtime::parallel::pipeline::DenseBf16MemoryPlan>> {
        if self.support().storage != StorageEncoding::NativeBf16 {
            return Ok(None);
        }
        pipeline::dense_bf16_memory_plan(&self.request).map(Some)
    }
}

/// Checkpoint storage is independent of model precision and the legacy W8/F32
/// operand profiles. No default selection is changed by this explicit policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericStoragePolicy {
    E4M3FnBf16Scales,
}

/// Strict 35B metadata admission for the BF16-model numeric-storage execution.
/// No payloads or CUDA context are created by `open_hf`.
#[derive(Debug, Clone)]
pub struct Qwen35NumericBf16MemoryPlan {
    resources: ferrule_model::transformer::BoundDecoderResources,
    options: ferrule_model::decoder::GenericDecoderOptions,
    estimate: ferrule_model::decoder::HybridMemoryPlan,
    budget: ferrule_model::decoder::HybridReservationRequirements,
    total_device_bytes: usize,
    expert_read_pipeline: Option<ExpertReadPipelineOptions>,
}
impl Qwen35NumericBf16MemoryPlan {
    pub fn open_hf(
        path: &std::path::Path,
        precision: ModelPrecision,
        storage: NumericStoragePolicy,
        limits: Qwen35MoeCapacityLimits,
        max_tensor_bytes: u64,
    ) -> Result<Self> {
        Self::open_hf_with_precision(
            path,
            precision,
            storage,
            limits,
            max_tensor_bytes,
            NumericFp8Precision::Bf16ModelF32Accumulate,
        )
    }

    /// Metadata-only admission for an explicit fused or tiled BF16-model image.
    pub fn open_hf_with_precision(
        path: &std::path::Path,
        precision: ModelPrecision,
        storage: NumericStoragePolicy,
        limits: Qwen35MoeCapacityLimits,
        max_tensor_bytes: u64,
        numeric_precision: NumericFp8Precision,
    ) -> Result<Self> {
        Self::open_hf_with_read_pipeline(
            path,
            precision,
            storage,
            limits,
            max_tensor_bytes,
            numeric_precision,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn open_hf_with_read_pipeline(
        path: &std::path::Path,
        precision: ModelPrecision,
        storage: NumericStoragePolicy,
        limits: Qwen35MoeCapacityLimits,
        max_tensor_bytes: u64,
        numeric_precision: NumericFp8Precision,
        read: Option<ExpertReadPipelineOptions>,
    ) -> Result<Self> {
        if !numeric_precision.has_bf16_model_boundaries() {
            return Err(invalid(
                "requires BF16 model numeric arithmetic, not legacy F32/W8",
            ));
        }
        use ferrule_model::{decoder::*, execution::ExecutionPrecisionPolicy, transformer::*};
        if precision != ModelPrecision::Bf16 || storage != NumericStoragePolicy::E4M3FnBf16Scales {
            return Err(invalid(
                "numeric storage requires explicit BF16 model precision",
            ));
        }
        let (metadata, resources) =
            ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(path)?;
        if metadata.config().profile() != ferrule_model::models::qwen35::Qwen35Profile::Moe35BA3Bfp8
        {
            return Err(invalid(
                "numeric BF16 factory requires the exact full-depth Qwen3.5-35B-A3B FP8 checkpoint",
            ));
        }
        if [
            limits.max_experts,
            limits.max_bytes,
            limits.workspace_bytes,
            limits.kv_bytes,
            limits.device_bytes,
            limits.max_positions,
            limits.max_sequences,
            limits.max_batch_tokens,
        ]
        .contains(&0)
            || limits.max_bytes <= limits.scratch_bytes
            || limits.max_batch_tokens > 32
            || limits.max_batch_tokens > limits.max_positions
            || limits.max_experts
                > resources.spec().layers().len() * metadata.config().text().num_experts.unwrap()
        {
            return Err(invalid(
                "numeric BF16 requires finite nonzero capacities, batch <= min(32, context), bounded cache beyond scratch",
            ));
        }
        NumericFp8Bf16HybridCudaProfile::validate(&resources)?;
        for p in resources.state_dict().parameters() {
            let bytes = p
                .weight()
                .slice()
                .bytes
                .checked_add(p.scale().map_or(0, |s| s.slice().bytes))
                .ok_or_else(|| invalid("numeric BF16 source size overflow"))?;
            if bytes > max_tensor_bytes {
                return Err(invalid(
                    "numeric BF16 tensor read cap is below checkpoint parameter size",
                ));
            }
        }
        let options = GenericDecoderOptions::standard_cpu(
            resources.spec(),
            ModelFamily::Qwen35Moe,
            WeightSource::Safetensors,
            16,
            limits.max_positions,
            limits.max_batch_tokens,
            limits.max_sequences,
            max_tensor_bytes,
            ExecutionPrecisionPolicy::bf16_compatibility(),
        )?
        .with_hybrid_cuda_numeric_fp8_bf16_precision(
            ExpertCacheLimits {
                max_experts: limits.max_experts,
                max_bytes: limits.max_bytes,
            },
            limits.scratch_bytes,
            numeric_precision,
        )?;
        let options = if let Some(read) = read {
            expert_read_policy::validate_sources(&resources, &options, read)?;
            options.with_expert_read_pipeline(read)?
        } else {
            options
        };
        let planes =
            HybridStateSchema::from_spec(resources.spec(), resources.spec().layers().len())?
                .kv_planes_with_dtype(
                    16,
                    limits.max_positions,
                    ferrule_common::execution::KvElementType::Bf16,
                )?;
        let pages = plan_resident_kv_pages(
            &planes,
            ResidentKvPageAccounting::TransactionCapacity {
                page_bytes: u64::try_from(
                    planes
                        .checked_page_bytes()
                        .ok_or_else(|| invalid("BF16 page bytes overflow"))?,
                )
                .map_err(|_| invalid("BF16 page bytes exceed u64"))?,
                budget_bytes: Some(limits.kv_bytes as u64),
            },
            ResidentSchedulerConfig {
                max_active_sequences: limits.max_sequences,
                ..Default::default()
            },
            ResidentTopKDriverConfig {
                ctx_size: limits.max_positions,
                ..Default::default()
            },
        )?;
        let estimate =
            HybridMemoryPlan::for_resources(&resources, &options, pages.configured_pages)?;
        let workspace = limits
            .workspace_bytes
            .checked_add(limits.scratch_bytes)
            .ok_or_else(|| invalid("BF16 workspace overflow"))?;
        if estimate.workspace_bytes_upper_bound > workspace {
            return Err(invalid(
                "numeric BF16 model workspace exceeds explicit workspace cap",
            ));
        }
        let copies = limits
            .max_sequences
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("BF16 state copies overflow"))?;
        let state_bytes = estimate
            .per_sequence_state_bytes
            .checked_mul(copies)
            .ok_or_else(|| invalid("BF16 state budget overflow"))?;
        let budget = HybridReservationRequirements {
            kv_pages: pages.configured_pages,
            state_bytes,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: workspace,
        };
        let total_device_bytes = [
            budget.weight_bytes,
            budget.workspace_bytes,
            state_bytes,
            estimate.kv_bytes,
            limits.allocator_margin_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |n, b| {
            n.checked_add(b)
                .ok_or_else(|| invalid("BF16 total device bytes overflow"))
        })?;
        if total_device_bytes > limits.device_bytes {
            return Err(invalid(
                "numeric BF16 image plus allocator margin exceeds device byte cap",
            ));
        }
        Ok(Self {
            resources,
            options,
            estimate,
            budget,
            total_device_bytes,
            expert_read_pipeline: read,
        })
    }
    pub fn resources(&self) -> &ferrule_model::transformer::BoundDecoderResources {
        &self.resources
    }
    pub fn expert_read_pipeline_options(&self) -> Option<ExpertReadPipelineOptions> {
        self.expert_read_pipeline
    }
    pub fn runner_options(&self) -> &ferrule_model::decoder::GenericDecoderOptions {
        &self.options
    }
    pub fn model_estimate(&self) -> ferrule_model::decoder::HybridMemoryPlan {
        self.estimate
    }
    pub fn model_budget(&self) -> ferrule_model::decoder::HybridReservationRequirements {
        self.budget
    }
    pub fn total_device_bytes(&self) -> usize {
        self.total_device_bytes
    }
    pub fn numeric_precision(&self) -> NumericFp8Precision {
        self.options
            .hybrid_cuda_numeric_fp8_precision()
            .expect("admitted BF16 numeric image")
    }
    pub fn backend_profile(&self) -> &'static str {
        self.numeric_precision().hybrid_backend_name()
    }
    /// Shared storage/dtype policy; `numeric_precision` identifies the arithmetic.
    pub fn profile(&self) -> ferrule_model::transformer::NumericFp8Bf16HybridCudaProfile {
        ferrule_model::transformer::NumericFp8Bf16HybridCudaProfile
    }
}

impl ResidentModelPlanner {
    /// Explicit public factory hook, separate from legacy F32/W8 serving flags.
    /// Full-depth TP1/PP1/EP1, device 0, resident typed KV, no HostKV/prefix cache
    /// or speculation. Build uses the existing resident owner, claim and journal.
    pub fn prepare_numeric_fp8_bf16_serving(
        &self,
        config: &AutoConfig,
        backend: BackendSelection,
        chat_template: Option<&str>,
        options: ModelFactoryOptions,
        serving: ServingModelOptions,
        storage: NumericStoragePolicy,
    ) -> Result<ResidentModelBuildPlan> {
        self.prepare_numeric_fp8_bf16_serving_with_precision(
            config,
            backend,
            chat_template,
            options,
            serving,
            storage,
            NumericFp8Precision::Bf16ModelF32Accumulate,
        )
    }

    /// Explicit model-BF16 numeric arithmetic; defaults and legacy W8 flags are unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_numeric_fp8_bf16_serving_with_precision(
        &self,
        config: &AutoConfig,
        backend: BackendSelection,
        chat_template: Option<&str>,
        options: ModelFactoryOptions,
        serving: ServingModelOptions,
        storage: NumericStoragePolicy,
        numeric_precision: NumericFp8Precision,
    ) -> Result<ResidentModelBuildPlan> {
        self.prepare_serving_request(
            config,
            backend,
            chat_template,
            options,
            SelectionRequest {
                serving,
                numeric_model: Some(NumericModelRequest {
                    storage,
                    precision: numeric_precision,
                }),
            },
        )
    }
}

pub(super) fn validate_numeric_request(
    backend: BackendSelection,
    options: &ModelFactoryOptions,
    serving: &ServingModelOptions,
    storage: NumericStoragePolicy,
    numeric_precision: NumericFp8Precision,
) -> Result<()> {
    if !numeric_precision.has_bf16_model_boundaries() {
        return Err(invalid(
            "requires BF16 model numeric arithmetic, not legacy F32/W8",
        ));
    }
    if serving.parallel.is_some()
        && numeric_precision == NumericFp8Precision::Bf16ModelMixedF32Accumulate
    {
        mixed_serving::tensor::validate_request(backend, options, serving, storage)?;
        return mixed_serving::tensor::require_quality();
    }
    if backend != BackendSelection::Cuda
        || serving.model_precision != Some(ModelPrecision::Bf16)
        || storage != NumericStoragePolicy::E4M3FnBf16Scales
        || serving.kv_execution_policy != KvExecutionPolicy::Resident
        || serving.parallel.is_some()
        || serving.engine == ServingEngineSelection::Pipeline
        || options.numeric_fp8_profile.is_some()
        || options.qwen35_host_cache.is_some()
        || serving.expert_host_cache_entries.is_some()
        || serving.expert_host_cache_mebibytes.is_some()
        || serving.expert_pinned_cache_entries.is_some()
        || serving.expert_pinned_cache_mebibytes.is_some()
        || serving.moe_hotset_experts.is_some()
        || options.moe_hotset_experts != 0
    {
        return Err(invalid(
            "numeric BF16 requires explicit CUDA/BF16/E4M3FN-BF16 storage, local resident engine, lazy bounded experts; no legacy numeric profile, parallel/HostKV/host-cache overrides",
        ));
    }
    if numeric_precision == NumericFp8Precision::Bf16ModelMixedF32Accumulate {
        ServingModelProfile::Qwen35MixedBf16
            .validate_startup_environment(std::env::var_os("CUBLAS_WORKSPACE_CONFIG").as_deref())?;
    }
    Ok(())
}

pub(super) fn select_numeric(
    backend: BackendSelection,
    request: &SelectionRequest,
    options: &ModelFactoryOptions,
) -> Result<ResolvedStrategy> {
    let numeric = request
        .numeric_model
        .expect("selected numeric model policy");
    validate_numeric_request(
        backend,
        options,
        &request.serving,
        numeric.storage,
        numeric.precision,
    )?;
    if request.serving.parallel.is_some() {
        return mixed_serving::tensor::select(request);
    }
    let mut execution = ResolvedExecution::new(
        ModelExecutionBackend::Cuda,
        ResolvedConstruction::Resident(build_numeric_bf16),
        StorageEncoding::NumericFp8E4m3fnBf16Scales,
        ArithmeticPrecision::Bf16Compatibility,
        "numeric model BF16 resident",
    );
    execution.support.static_constraints = "numeric E4M3FN/BF16 scales; native BF16 resident weights/Rows/residual/KV/conv/logits; router parameter weights/logits BF16, probabilities/route weights F32; GDN A_log/norm sources BF16 or F32, F32 sources cast once to resident/operator BF16; F32 accumulation/SSM; RoPE inv_freq/trig F32, resident cos/sin/output BF16; full-depth 35B TP1/PP1/EP1/device0; lazy bounded typed experts; batch<=32; no HostKV/prefix/speculation/host prewarm; declared device policy, metadata admission does not validate GPU storage; GPU quality acceptance pending";
    execution.supports_expert_transfer = true;
    execution.numeric_bf16_precision = Some(numeric.precision);
    execution.topology_policy = Some(LOCAL_TOPOLOGY);
    execution.conversion =
        PipelineConversion::Frozen("native BF16 topology/precision is frozen by prepare_serving");
    execution.adjustment_reason = qwen35::ADJUSTMENT_REASON;
    Ok(ResolvedStrategy {
        execution,
        backend: ModelExecutionBackend::Cuda,
        catalog_backend: BackendSelection::Cuda,
        resolve: resolve_numeric,
        policy: &NUMERIC_OPTIONS,
        parallel: None,
    })
}

pub(super) fn resolve_numeric(
    _: &ModelDescriptor,
    _: BackendSelection,
) -> Result<ResolvedModelBackend> {
    Ok(ResolvedModelBackend {
        family: ModelFamily::Qwen35Moe,
        model_name: "qwen3.5-35b-a3b-numeric-bf16",
        backend: ModelExecutionBackend::Cuda,
        backend_profile: "numeric model BF16 resident",
        default_chat_template: ChatTemplate::Qwen35,
    })
}
const NUMERIC_OPTIONS: ModelOptionPolicy = ModelOptionPolicy {
    serving: qwen35_composition::resident_options,
    parallel: reject_parallel,
    adapt: qwen35::adapt_options,
    finish: |plan| {
        qwen35::finish_options(plan)?;
        plan.backend_profile = plan
            .request
            .execution
            .numeric_bf16_precision
            .expect("numeric policy")
            .hybrid_backend_name();
        if plan.request.execution.expert_read_pipeline.is_some() {
            plan.prepare_numeric_read_admission()?;
        } else {
            plan.numeric_bf16_memory_plan()?;
        }
        Ok(())
    },
};

impl ResidentModelBuildPlan {
    pub fn numeric_bf16_memory_plan(&self) -> Result<Option<Qwen35NumericBf16MemoryPlan>> {
        if mixed_serving::tensor::is_selected(&self.request) {
            return Ok(None);
        }
        if !matches!(
            self.request.execution.support.storage,
            StorageEncoding::NumericFp8E4m3fnBf16Scales
        ) || !matches!(
            self.request.execution.support.arithmetic,
            ArithmeticPrecision::Bf16Compatibility
        ) {
            return Ok(None);
        }
        if self.request.execution.expert_read_pipeline.is_some() {
            let prepared = prepared_numeric_read(&self.prepared_source_plan)?;
            prepared.validate(&self.request)?;
            return Ok(Some(prepared.plan.clone()));
        }
        Qwen35NumericBf16MemoryPlan::open_hf_with_precision(
            &self.request.model_path,
            ModelPrecision::Bf16,
            NumericStoragePolicy::E4M3FnBf16Scales,
            qwen35::capacity_limits(&self.request)?,
            self.request.max_tensor_bytes,
            self.request
                .execution
                .numeric_bf16_precision
                .ok_or_else(|| invalid("missing immutable BF16 numeric arithmetic choice"))?,
        )
        .map(Some)
    }
}
#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/expert_read_reporting.rs"]
pub(super) mod expert_read_reporting;

#[cfg(all(test, feature = "cuda"))]
type NumericBf16RunnerObserver = Box<
    dyn FnOnce(
        &mut ferrule_model::decoder::GenericDecoderRunner<ferrule_model::decoder::HybridCudaDecoder>,
    ),
>;

#[cfg(all(test, feature = "cuda"))]
thread_local! {
    // Private unit-test mount on the actual factory runner, before type erasure.
    pub(super) static NUMERIC_BF16_TEST_RUNNER: std::cell::RefCell<Option<NumericBf16RunnerObserver>> = const { std::cell::RefCell::new(None) };
    // Observation of the real factory's owner only; never a replacement context.
    pub(super) static NUMERIC_BF16_TEST_DEVICE: std::cell::RefCell<Option<ferrule_model::decoder::HybridCudaDevice>> = const { std::cell::RefCell::new(None) };
}

pub(super) fn build_numeric_bf16(
    request: ModelBuildRequest,
) -> Result<BoxedSessionInferenceEngine> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = request;
        Err(invalid(
            "numeric BF16 requires the cuda feature; no CPU fallback",
        ))
    }
    #[cfg(feature = "cuda")]
    {
        use ferrule_model::{TokenizerHandle, decoder::*};
        let prepared = if request.resolved.execution.expert_read_pipeline.is_some() {
            let prepared = prepared_numeric_read(&request.prepared_source_plan)?;
            prepared.validate(&request.resolved)?;
            Some(prepared.plan.clone())
        } else {
            None
        };
        let request = request.resolved;
        let limits = qwen35::capacity_limits(&request)?;
        let plan = if let Some(plan) = prepared {
            plan
        } else {
            Qwen35NumericBf16MemoryPlan::open_hf_with_precision(
                &request.model_path,
                ModelPrecision::Bf16,
                NumericStoragePolicy::E4M3FnBf16Scales,
                limits,
                request.max_tensor_bytes,
                request
                    .execution
                    .numeric_bf16_precision
                    .ok_or_else(|| invalid("missing immutable BF16 numeric arithmetic choice"))?,
            )?
        };
        let planes = HybridStateSchema::from_spec(plan.resources.spec(), request.max_layers)?
            .kv_planes_with_dtype(
                plan.options.page_size(),
                plan.options.max_positions(),
                ferrule_common::execution::KvElementType::Bf16,
            )?;
        let accounting = ResidentKvPageAccounting::TransactionCapacity {
            page_bytes: u64::try_from(
                planes
                    .checked_page_bytes()
                    .ok_or_else(|| invalid("BF16 page bytes overflow"))?,
            )
            .map_err(|_| invalid("BF16 page bytes exceed u64"))?,
            budget_bytes: Some(limits.kv_bytes as u64),
        };
        if let Some(read) = plan.expert_read_pipeline {
            expert_read_policy::validate_available_host(read)?;
        }
        let device = HybridCudaDevice::new_on_device(0, plan.budget)?;
        if device.operators().memory_info()?.0 < plan.total_device_bytes {
            return Err(invalid(
                "numeric BF16 live free device memory is below admitted image plus margin",
            ));
        }
        #[cfg(test)]
        NUMERIC_BF16_TEST_DEVICE.with(|slot| *slot.borrow_mut() = Some(device.clone()));
        let tokenizer = TokenizerHandle::load(&request.model_path)?;
        let mut runner = GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda(
            plan.resources,
            tokenizer,
            plan.options,
            device,
        )?;
        runner.set_expert_transfer_policy(
            expert_read_policy::effective_transfer_policy(&request.execution).into(),
        )?;
        tracing::info!(
            requested_expert_transfer_policy = ?request.execution.expert_transfer_policy,
            expert_transfer_policy = ?runner.expert_transfer_policy(),
            "configured resident expert transfers before engine composition"
        );
        if request.execution.expert_read_pipeline.is_some() {
            use ferrule_model::runner::ResidentModelRunner;
            log_initial_expert_read_endpoint(
                || runner.physical_resource_snapshot(),
                request.execution.expert_transfer_policy,
                runner.expert_transfer_policy(),
            )?;
        }
        #[cfg(test)]
        NUMERIC_BF16_TEST_RUNNER.with(|slot| {
            if let Some(observe) = slot.borrow_mut().take() {
                observe(&mut runner);
            }
        });
        let engine = ferrule_runtime::engine::compose_resident_engine(
            runner,
            Box::new(planes),
            accounting,
            request.scheduler_config,
            request.driver_config,
        )?;
        Ok(Box::new(engine))
    }
}

// Metadata-only builds do not construct the numeric device owner. Keep this
// serialization seam feature-neutral so its evidence contract is CPU-testable.
#[allow(dead_code)]
fn log_initial_expert_read_endpoint(
    snapshot: impl FnOnce() -> ferrule_model::resource_observability::PhysicalResourceObservation,
    requested_policy: Option<ServingExpertTransferPolicy>,
    effective_policy: ferrule_model::transformer::ExpertTransferPolicy,
) -> Result<()> {
    if !tracing::enabled!(target: "ferrule_performance", tracing::Level::INFO) {
        return Ok(());
    }
    use ferrule_model::resource_observability::{ResourceEvidence, ResourceObservationScope};
    #[derive(serde::Serialize)]
    struct OwnerBinding {
        scope: ResourceEvidence<ResourceObservationScope>,
        prepared_plan_id: ResourceEvidence<u64>,
        image_generation: ResourceEvidence<u64>,
        device_ordinal: ResourceEvidence<usize>,
    }
    let snapshot = snapshot();
    let initial = serde_json::to_string(&snapshot.expert_read_pipeline).map_err(|source| {
        ferrule_common::Error::ModelSource {
            source: Box::new(source),
        }
    })?;
    let owner_binding = serde_json::to_string(&OwnerBinding {
        scope: snapshot.scope,
        prepared_plan_id: snapshot.prepared_plan_id,
        image_generation: snapshot.image_generation,
        device_ordinal: snapshot.device_ordinal,
    })
    .map_err(|source| ferrule_common::Error::ModelSource {
        source: Box::new(source),
    })?;
    tracing::info!(target: "ferrule_performance",
        event = "expert_read_initial_endpoint",
        producer_schema_version = 1u64,
        source_contract = "resident_model_runner.physical_resource_snapshot.expert_read_pipeline",
        endpoint = "after_owner_configuration_before_prediction",
        process_id = std::process::id(),
        owner_binding = owner_binding.as_str(),
        requested_expert_transfer_policy = ?requested_policy,
        expert_transfer_policy = ?effective_policy,
        expert_read_pipeline_initial = initial.as_str(),
        "Observed initial expert read endpoint before engine composition"
    );
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/expert_read_initial_endpoint.rs"]
mod initial_endpoint_tests;

/// Frozen metadata for the opt-in lazy read path. The original bound resources
/// and read budget cross admission and construction together, without rebinding.
#[derive(Debug)]
struct PreparedNumericReadAdmission {
    frozen: ResolvedModelRequest,
    plan: Qwen35NumericBf16MemoryPlan,
    sources: Vec<ferrule_model::checkpoint::CheckpointSourceFileIdentity>,
    indexed_source: bool,
}
impl PreparedSourcePlan for PreparedNumericReadAdmission {
    fn validate(&self, request: &ResolvedModelRequest) -> Result<()> {
        expert_read_policy::validate_resolved(request)?;
        if !selection::same_final_request(&self.frozen, request) {
            return Err(invalid(
                "expert read final request/budget changed after source-bound preparation",
            ));
        }
        if self.indexed_source
            != request
                .model_path
                .join("model.safetensors.index.json")
                .exists()
            || !self.sources.iter().all(|source| source.is_current())
        {
            return Err(invalid("expert read prepared source snapshot is stale"));
        }
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
fn prepared_numeric_read(
    source: &Option<std::sync::Arc<dyn PreparedSourcePlan>>,
) -> Result<&PreparedNumericReadAdmission> {
    source
        .as_ref()
        .and_then(|p| p.as_any().downcast_ref())
        .ok_or_else(|| invalid("expert read pipeline requires frozen source-bound admission"))
}
impl ResidentModelBuildPlan {
    fn prepare_numeric_read_admission(&mut self) -> Result<()> {
        if self.prepared_source_plan.is_some() {
            return prepared_numeric_read(&self.prepared_source_plan)?.validate(&self.request);
        }
        expert_read_policy::validate_resolved(&self.request)?;
        let indexed_source = self
            .request
            .model_path
            .join("model.safetensors.index.json")
            .exists();
        let metadata_sources = ["config.json", "model.safetensors.index.json"]
            .into_iter()
            .map(|name| self.request.model_path.join(name))
            .filter(|path| path.exists())
            .map(|path| ferrule_model::checkpoint::CheckpointSourceFileIdentity::capture(&path))
            .collect::<ferrule_common::Result<Vec<_>>>()?;
        let plan = Qwen35NumericBf16MemoryPlan::open_hf_with_read_pipeline(
            &self.request.model_path,
            ModelPrecision::Bf16,
            NumericStoragePolicy::E4M3FnBf16Scales,
            qwen35::capacity_limits(&self.request)?,
            self.request.max_tensor_bytes,
            self.request
                .execution
                .numeric_bf16_precision
                .ok_or_else(|| invalid("missing immutable BF16 numeric arithmetic choice"))?,
            self.request.execution.expert_read_pipeline,
        )?;
        let sources = plan
            .resources
            .state_dict()
            .parameters()
            .iter()
            .flat_map(|p| {
                std::iter::once(p.weight().source_identity())
                    .chain(p.scale().map(|scale| scale.source_identity()))
            })
            .cloned()
            .chain(metadata_sources)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let prepared = PreparedNumericReadAdmission {
            frozen: self.request.clone(),
            plan,
            sources,
            indexed_source,
        };
        prepared.validate(&self.request)?;
        self.prepared_source_plan = Some(std::sync::Arc::new(prepared));
        Ok(())
    }
}
