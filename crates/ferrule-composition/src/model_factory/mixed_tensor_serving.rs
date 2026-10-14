//! Mixed TP2 catalog policy and admission adapter. Execution and custody stay in
//! the existing parallel assembly, model program and runtime owner journal.
use super::*;
use ferrule_common::ParallelismPlan;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::hybrid_parallel::HybridRankPrecision;
use ferrule_model::transformer::{BoundDecoderResources, NumericFp8Precision};
use ferrule_runtime::engine::{KvExecutionPolicy, ResidentKvPagePlan};
use ferrule_runtime::parallel::collective::HostCollectiveLimits;
use ferrule_runtime::parallel::pipeline::{PhysicalKvLayout, PipelineConfig};
use std::sync::Arc;

// Authorized for the bounded TP2 profile after independent full40, multistep
// and lifecycle acceptance; widening this profile requires new acceptance.
// No environment override: metadata completeness is not numerical acceptance.
const PUBLIC_QUALITY_ACCEPTED: bool = true;
const DEVICE_LIMIT: u64 = 22 << 30;
const DECODED_CACHE_GRANTS: usize = 3 << 30;
/// Strict Mixed TP2 only; includes the decoded-cache ceiling, unlike TP1.
pub const MIXED_TP2_DEFAULT_WORKSPACE_BYTES: usize = 4 << 30;

/// Per-owner complete source-derived admission, not a VRAM reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct MixedTp2CacheRequirements {
    pub device: usize,
    pub cache_enabled: bool,
    pub cache_capacity_grants: usize,
    pub pinned_grants: usize,
    pub routed_working_set_grants: usize,
    pub required_workspace_bytes: u64,
    pub workspace_ceiling_bytes: usize,
    pub required_device_bytes: u64,
    pub device_ceiling_bytes: usize,
}
const HOST_LIMIT: u64 = 256 << 20;
const PROFILE: &str = "cuda-qwen35-mixed-bf16-tp2-local-shards-serial";

pub(in crate::model_factory) fn require_quality() -> Result<()> {
    if !PUBLIC_QUALITY_ACCEPTED {
        return Err(invalid(
            "Mixed TP2 public admission requires the authorized full40 multistep/lifecycle quality gate",
        ));
    }
    Ok(())
}

const TOPOLOGY: execution_policy::TopologyPolicy = execution_policy::TopologyPolicy {
    layout: execution_policy::OwnerLayout::StageRoots,
    execution: execution_policy::ExecutionCapabilities {
        ranks: execution_policy::RankCapability::ThreadOnly,
        tensor: execution_policy::TensorCapability::DeviceThreads { degrees: &[2] },
    },
    placement: execution_policy::PlacementPolicy {
        selection: execution_policy::DeviceSelection::Explicit,
        sharing: execution_policy::DeviceSharing::Distinct,
    },
};

pub(in crate::model_factory) fn validate_placement(placement: &PipelineBuildOptions) -> Result<()> {
    if placement.parallelism
        != (ParallelismPlan {
            tensor_parallel: 2,
            ..Default::default()
        })
    {
        return Err(invalid("Mixed BF16 TP2 requires PP1 TP2 EP1 DP/SP/CP1"));
    }
    placement.validate_policy(
        ModelExecutionBackend::Cuda,
        TOPOLOGY.layout,
        TOPOLOGY.execution,
        TOPOLOGY.placement,
    )?;
    Ok(())
}

fn capacity(options: Option<Qwen35MoeCapacityLimits>) -> Qwen35MoeCapacityLimits {
    options.unwrap_or(Qwen35MoeCapacityLimits {
        device_bytes: DEVICE_LIMIT as usize,
        workspace_bytes: MIXED_TP2_DEFAULT_WORKSPACE_BYTES,
        ..Default::default()
    })
}

fn validate_capacity(limits: Qwen35MoeCapacityLimits) -> Result<()> {
    let defaults = Qwen35MoeCapacityLimits::default();
    if limits.max_experts != defaults.max_experts
        || limits.max_bytes != defaults.max_bytes
        || limits.scratch_bytes != defaults.scratch_bytes
    {
        return Err(invalid(
            "Mixed TP2 keeps every local expert shard resident; TP1 expert cache/scratch overrides are unsupported",
        ));
    }
    if limits.workspace_bytes <= DECODED_CACHE_GRANTS {
        return Err(invalid(
            "Mixed TP2 cache-on requires workspace for the 3 GiB decoded cache PLUS native operators/providers; use --cuda-workspace-mb 4096 or omit it for the strict-profile default; the old 1024 MiB ceiling cannot admit cache-on",
        ));
    }
    if limits.device_bytes == 0
        || limits.device_bytes as u64 > DEVICE_LIMIT
        || limits.workspace_bytes == 0
        || limits.kv_bytes == 0
        || limits.allocator_margin_bytes == 0
    {
        return Err(invalid(
            "Mixed TP2 requires nonzero workspace/KV/margin and a per-device ceiling no greater than 22 GiB",
        ));
    }
    Ok(())
}

pub(in crate::model_factory) fn validate_request(
    backend: BackendSelection,
    options: &ModelFactoryOptions,
    serving: &ServingModelOptions,
    storage: NumericStoragePolicy,
) -> Result<()> {
    if backend != BackendSelection::Cuda
        || serving.model_precision != Some(ModelPrecision::Bf16)
        || storage != NumericStoragePolicy::E4M3FnBf16Scales
        || serving.kv_execution_policy != KvExecutionPolicy::Resident
        || serving.engine == ServingEngineSelection::Resident
        || options.numeric_fp8_profile.is_some()
        || options.qwen35_host_cache.is_some()
        || options.moe_hotset_experts != 0
        || serving.moe_hotset_experts.is_some()
        || serving.expert_host_cache_entries.is_some()
        || serving.expert_host_cache_mebibytes.is_some()
        || serving.expert_pinned_cache_entries.is_some()
        || serving.expert_pinned_cache_mebibytes.is_some()
    {
        return Err(invalid(
            "Mixed TP2 requires explicit CUDA/BF16 numeric storage, resident KV and auto/pipeline engine; no legacy arithmetic, host expert cache, hotset or offload",
        ));
    }
    validate_placement(
        serving
            .parallel
            .as_ref()
            .ok_or_else(|| invalid("Mixed TP2 requires explicit placement"))?,
    )?;
    validate_capacity(capacity(options.qwen35_moe_capacity))?;
    let scheduler = options.scheduler_config;
    if options.driver_config.ctx_size == 0
        || options.driver_config.ctx_size > 128
        || scheduler.max_batch_tokens == 0
        || scheduler.max_batch_tokens > 8
        || scheduler.max_batch_tokens > options.driver_config.ctx_size
        || scheduler.max_active_sequences == 0
        || scheduler.max_active_sequences > 2
        || scheduler.max_active_sequences > scheduler.max_batch_tokens
        || scheduler.prefill_chunk_size == 0
        || scheduler.prefill_chunk_size > scheduler.max_batch_tokens
        || scheduler.prefix_cache_capacity_pages != 0
    {
        return Err(invalid(
            "Mixed TP2 initial envelope requires context<=128, batch<=8, sessions<=2 and <=batch, prefill<=batch; nonzero capacities and no prefix cache",
        ));
    }
    ServingModelProfile::Qwen35MixedBf16
        .validate_startup_environment(std::env::var_os("CUBLAS_WORKSPACE_CONFIG").as_deref())
}

pub(in crate::model_factory) fn is_selected(request: &ResolvedModelRequest) -> bool {
    request.pipeline.is_some()
        && request.execution.numeric_bf16_precision
            == Some(NumericFp8Precision::Bf16ModelMixedF32Accumulate)
}

// Called only after public request validation; lowering remains metadata-only.
pub(in crate::model_factory) fn select(request: &SelectionRequest) -> Result<ResolvedStrategy> {
    let placement = request
        .serving
        .parallel
        .as_ref()
        .ok_or_else(|| invalid("Mixed TP2 placement absent"))?;
    validate_placement(placement)?;
    let mut execution = ResolvedExecution::new(
        ModelExecutionBackend::Cuda,
        ResolvedConstruction::Pipeline(build_hybrid_pipeline),
        StorageEncoding::NumericFp8E4m3fnBf16Scales,
        ArithmeticPrecision::Bf16Compatibility,
        "Mixed PP1 TP2 EP1 CUDA threads; two distinct devices; resident local expert shards; BF16 activation/KV/conv, F32 SSM/accumulation; serial greedy; context<=128 batch<=8 sessions<=2; decoded BF16 cache=3 GiB per owner (fixed pins + bounded routed backing and segment slack); workspace includes cache; <=22 GiB/card; no offload/process/restart; public quality gate required",
    );
    execution.numeric_bf16_precision = Some(NumericFp8Precision::Bf16ModelMixedF32Accumulate);
    execution.topology_policy = Some(TOPOLOGY);
    execution.conversion = PipelineConversion::Frozen(
        "Mixed TP2 topology and precision are frozen by catalog selection",
    );
    Ok(ResolvedStrategy {
        execution,
        backend: ModelExecutionBackend::Cuda,
        catalog_backend: BackendSelection::Cuda,
        resolve: bf16_serving::resolve_numeric,
        policy: &OPTIONS,
        parallel: Some(placement.clone()),
    })
}

const OPTIONS: ModelOptionPolicy = ModelOptionPolicy {
    serving: qwen35_composition::resident_options,
    parallel: |plan, placement, execution| {
        validate_placement(&placement)?;
        let mut plan = plan.with_resolved_pipeline(placement, execution)?;
        plan.backend_profile = PROFILE;
        Ok(plan)
    },
    adapt: |descriptor, options, execution| {
        if descriptor.spec.family != ModelFamily::Qwen35Moe {
            return Err(invalid(
                "Mixed TP2 requires the strict Qwen3.5-35B MoE profile",
            ));
        }
        options.qwen35_moe_capacity = Some(capacity(options.qwen35_moe_capacity));
        qwen35::adapt_options(descriptor, options, execution)
    },
    finish: |plan| {
        plan.backend_profile = PROFILE;
        plan.effective_options.qwen35_moe_capacity = Some(qwen35::capacity_limits(&plan.request)?);
        plan.effective_options.qwen35_host_cache = None;
        plan.prepare_mixed_tp2_admission()?;
        Ok(())
    },
};

impl ResidentModelBuildPlan {
    fn prepare_mixed_tp2_admission(&mut self) -> Result<()> {
        if !is_selected(&self.request) {
            return Err(invalid("prepared source profile changed"));
        }
        if let Some(existing) = &self.prepared_source_plan {
            let existing = existing
                .as_any()
                .downcast_ref::<PreparedMixedAdmission>()
                .ok_or_else(|| invalid("prepared source profile changed"))?;
            return existing.validate(&self.request);
        }
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
        let resources = Arc::new(qwen35::binding::hybrid(&self.request)?);
        let admission = Arc::new(Admission::new(&self.request, &resources)?);
        if !metadata_sources.iter().all(|source| source.is_current()) {
            return Err(invalid("Mixed TP2 metadata changed during preparation"));
        }
        self.prepared_source_plan = Some(Arc::new(PreparedMixedAdmission {
            frozen: self.request.clone(),
            resources,
            admission,
            metadata_sources,
            indexed_source,
        }));
        Ok(())
    }

    /// Metadata-derived complete per-card requirements, not reservations or free VRAM.
    /// TP1 retains its separate numeric_bf16_memory_plan and lazy expert policy.
    pub fn mixed_tp2_device_requirements(&self) -> Result<Option<Vec<(usize, u64)>>> {
        Ok(self.mixed_tp2_cache_requirements()?.map(|rows| {
            rows.into_iter()
                .map(|row| (row.device, row.required_device_bytes))
                .collect()
        }))
    }

    pub fn mixed_tp2_cache_requirements(&self) -> Result<Option<Vec<MixedTp2CacheRequirements>>> {
        if !is_selected(&self.request) {
            if self.prepared_source_plan.is_some() {
                return Err(invalid("prepared source profile changed"));
            }
            return Ok(None);
        }
        let prepared = prepared(&self.prepared_source_plan)?;
        prepared.validate(&self.request)?;
        Ok(Some(prepared.admission.cache_requirements.clone()))
    }
}

/// The detached frozen request cannot recursively contain this payload. Source
/// snapshots and paired numeric bindings remain those of the original catalog.
struct PreparedMixedAdmission {
    frozen: ResolvedModelRequest,
    resources: Arc<BoundDecoderResources>,
    admission: Arc<Admission>,
    metadata_sources: Vec<ferrule_model::checkpoint::CheckpointSourceFileIdentity>,
    indexed_source: bool,
}
impl std::fmt::Debug for PreparedMixedAdmission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedMixedAdmission")
            .finish_non_exhaustive()
    }
}
impl PreparedSourcePlan for PreparedMixedAdmission {
    fn validate(&self, request: &ResolvedModelRequest) -> Result<()> {
        PreparedMixedAdmission::validate(self, request)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
fn prepared(source: &Option<Arc<dyn PreparedSourcePlan>>) -> Result<&PreparedMixedAdmission> {
    source
        .as_ref()
        .and_then(|p| p.as_any().downcast_ref())
        .ok_or_else(|| invalid("Mixed TP2 requires its final frozen source-bound admission"))
}
#[cfg(any(feature = "cuda", test))]
pub(in crate::model_factory) fn consume_prepared(
    request: &ModelBuildRequest,
) -> Result<(BoundDecoderResources, Arc<Admission>)> {
    let prepared = prepared(&request.prepared_source_plan)?;
    prepared.validate(&request.resolved)?;
    Ok((
        prepared.resources.as_ref().clone(),
        Arc::clone(&prepared.admission),
    ))
}
impl PreparedMixedAdmission {
    fn validate(&self, request: &ResolvedModelRequest) -> Result<()> {
        if !selection::same_final_request(&self.frozen, request) {
            return Err(invalid(
                "Mixed TP2 final request changed after source-bound preparation",
            ));
        }
        require_quality()?;
        if self.indexed_source
            != request
                .model_path
                .join("model.safetensors.index.json")
                .exists()
        {
            return Err(invalid("Mixed TP2 prepared source catalog changed"));
        }
        if !self
            .metadata_sources
            .iter()
            .all(|source| source.is_current())
        {
            return Err(invalid(
                "Mixed TP2 prepared metadata source snapshot is stale",
            ));
        }
        parallel::bindings::validate_bound(request, &self.resources)?;
        parallel::bindings::validate_options(request)?;
        Ok(())
    }
}

#[cfg(test)]
thread_local! { static ADMISSION_CONSTRUCTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[cfg_attr(
    not(feature = "cuda"),
    allow(
        dead_code,
        reason = "immutable CPU admission also retains future owner inputs"
    )
)]
pub(in crate::model_factory) struct Admission {
    plan: ParallelExecutionPlan,
    config: PipelineConfig,
    kv: ResidentKvPagePlan,
    margins: ParallelOwnerMargins,
    capacities: Vec<(usize, u64)>,
    limits: HostCollectiveLimits,
    cache_requirements: Vec<MixedTp2CacheRequirements>,
    programs: Vec<ferrule_model::transformer::hybrid_parallel::HybridMixedRankProgramPlan>,
    host_required: u64,
}
impl Admission {
    pub(in crate::model_factory) fn new(
        request: &ResolvedModelRequest,
        resources: &BoundDecoderResources,
    ) -> Result<Self> {
        #[cfg(test)]
        ADMISSION_CONSTRUCTIONS.with(|n| n.set(n.get() + 1));
        if !is_selected(request)
            || request.backend != ModelExecutionBackend::Cuda
            || request.family != ModelFamily::Qwen35Moe
        {
            return Err(invalid(
                "Mixed TP2 admission requires the immutable catalog precision and family",
            ));
        }
        let placement = request.pipeline.as_ref().unwrap();
        validate_placement(placement)?;
        parallel::bindings::validate_bound(request, resources)?;
        parallel::bindings::validate_options(request)?;
        let budget = qwen35::capacity_limits(request)?;
        validate_capacity(budget)?;
        let plan = ParallelExecutionPlan::describe(
            resources.spec(),
            placement,
            ParallelExecutionContract::new(
                HybridRankPrecision::MixedBf16F32,
                ParallelExpertMode::LocalTensorShards,
            ),
            vec![],
        )?;
        // The pipeline owns logical pages and both BF16 rank-local COW pools.
        // Report their aggregate physical page size, not the unsharded F32 size.
        let page_size = 16;
        let mut page_bytes = 0u64;
        let mut planes = None;
        for lane in 0..2 {
            let local = plan.layout().stages()[0]
                .rank_layout(lane)?
                .kv_planes(page_size, request.driver_config.ctx_size)?;
            page_bytes = page_bytes
                .checked_add(physical_page_bytes(&local)?)
                .ok_or_else(|| invalid("Mixed KV byte overflow"))?;
            planes = Some(local);
        }
        let kv = ferrule_runtime::engine::plan_pipeline_kv_pages(
            planes.as_ref().unwrap(),
            page_bytes,
            request.kv_cache_bytes,
            PhysicalKvLayout::hybrid(ModelExecutionBackend::Cuda),
            request.scheduler_config,
            request.driver_config,
        )?;
        let config = PipelineConfig {
            page_size,
            max_pages: kv.full_capacity_pages,
            max_positions: request.driver_config.ctx_size,
            max_batch_tokens: request.scheduler_config.max_batch_tokens,
            session_capacity: request.scheduler_config.max_active_sequences,
            max_parameter_bytes: request.max_tensor_bytes,
            precision: ExecutionPrecisionPolicy::bf16_compatibility(),
            max_ack_polls: 8,
        };
        config.validate()?;
        let capacity = ParallelRankCapacity {
            page_size,
            max_positions: config.max_positions,
            max_rows: config.max_batch_tokens,
            sessions: config.session_capacity,
            logical_pages: config.max_pages,
        };
        let elements = config
            .max_batch_tokens
            .checked_mul(
                resources
                    .spec()
                    .vocab_size()
                    .div_ceil(2)
                    .max(resources.spec().hidden_size()),
            )
            .ok_or_else(|| invalid("Mixed collective size overflow"))?;
        let limits = HostCollectiveLimits {
            max_ranks: 2,
            max_elements_per_rank: elements,
            max_host_bytes: 64 << 20,
        };
        let margins = ParallelOwnerMargins {
            context_bytes: 64 << 20,
            allocator_bytes: budget.allocator_margin_bytes as u64,
        };
        let capacities = plan
            .devices()
            .iter()
            .map(|d| (*d, budget.device_bytes as u64))
            .collect::<Vec<_>>();
        let collectives = plan.mixed_collective_plans(capacity, limits)?;
        let programs = plan.mixed_programs_with_decoded_weight_cache(
            resources,
            capacity,
            margins,
            &capacities,
            DECODED_CACHE_GRANTS,
        )?;
        let report = plan.resource_requirements_from_programs(
            resources,
            capacity,
            margins,
            &collectives,
            &programs,
        )?;
        if !report.resource_complete() {
            return Err(invalid(
                "Mixed TP2 complete model/provider claims are required",
            ));
        }
        report.check_known_capacity(&capacities, HOST_LIMIT)?;
        let host_required = report.known_host_bytes()?;
        let mut cache_requirements = Vec::new();
        for owner in report.owners() {
            let total = owner.known_device_bytes()?;
            let workspace = total
                - owner.storage.weight_grants as u64
                - owner.storage.state.total_grants as u64
                - owner.storage.kv_grants as u64
                - margins.context_bytes
                - margins.allocator_bytes;
            if workspace > budget.workspace_bytes as u64 {
                return Err(invalid(
                    "Mixed TP2 model/provider workspace exceeds per-owner workspace ceiling",
                ));
            }
            let cache = programs
                .iter()
                .find(|p| p.layout().owner() == owner.owner)
                .and_then(|p| p.decoded_weight_cache_budget())
                .ok_or_else(|| {
                    invalid("strict Mixed TP2 lost its required decoded cache admission")
                })?;
            cache_requirements.push(MixedTp2CacheRequirements {
                device: owner.device,
                cache_enabled: true,
                cache_capacity_grants: cache.capacity_grants,
                pinned_grants: cache.pinned_grants,
                routed_working_set_grants: cache.routed_working_set_grants,
                required_workspace_bytes: workspace,
                workspace_ceiling_bytes: budget.workspace_bytes,
                required_device_bytes: total,
                device_ceiling_bytes: budget.device_bytes,
            });
        }
        Ok(Self {
            plan,
            config,
            kv,
            margins,
            capacities,
            limits,
            cache_requirements,
            programs,
            host_required,
        })
    }
}

#[cfg(feature = "cuda")]
impl parallel::assembly::PipelineAssembly for Admission {
    fn kv(&self) -> ResidentKvPagePlan {
        self.kv
    }
    fn admit(&self, _: &ResolvedModelRequest, _: &BoundDecoderResources) -> Result<()> {
        require_quality()?;
        if self.host_required > host_memory::snapshot()?.available_bytes() {
            return Err(invalid(
                "Mixed TP2 host claim exceeds live host/cgroup headroom",
            ));
        }
        Ok(())
    }
    fn start(
        self,
        request: &ResolvedModelRequest,
        resources: BoundDecoderResources,
    ) -> Result<ferrule_runtime::parallel::pipeline::PipelineParallelExecutor> {
        require_quality()?;
        // Revalidate complete source-bound claims, without rebuilding cache-off programs.
        self.plan.start_internal_with_programs(
            resources,
            self.config,
            self.margins,
            &self.capacities,
            HOST_LIMIT,
            self.limits,
            request.pipeline.as_ref().unwrap().rank_timeout,
            self.programs,
        )
    }
}

#[cfg(feature = "cuda")]
impl parallel::assembly::PipelineAssembly for Arc<Admission> {
    fn kv(&self) -> ResidentKvPagePlan {
        self.kv
    }
    fn admit(
        &self,
        request: &ResolvedModelRequest,
        resources: &BoundDecoderResources,
    ) -> Result<()> {
        self.as_ref().admit(request, resources)
    }
    fn start(
        self,
        request: &ResolvedModelRequest,
        resources: BoundDecoderResources,
    ) -> Result<ferrule_runtime::parallel::pipeline::PipelineParallelExecutor> {
        require_quality()?;
        parallel::bindings::validate_bound(request, &resources)?;
        parallel::bindings::validate_options(request)?;
        self.plan.start_internal_with_programs(
            resources,
            self.config,
            self.margins,
            &self.capacities,
            HOST_LIMIT,
            self.limits,
            request.pipeline.as_ref().unwrap().rank_timeout,
            self.programs.clone(),
        )
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/mixed_tensor_serving.rs"]
mod tests;
