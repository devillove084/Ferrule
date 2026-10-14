//! Thread/process pipeline construction through the existing model catalog.

#[path = "hybrid_pipeline.rs"]
pub(super) mod hybrid;

use ferrule_runtime::parallel::pipeline::PhysicalKvLayout;
use std::time::Duration;

use ferrule_common::execution::KvElementType;
use ferrule_common::{
    ParallelRankId, ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology,
};
use ferrule_model::decoder::StandardGqaPlanes;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::expert_parallel::{ExpertDispatchLimits, ExpertPlacement};
use ferrule_model::transformer::{
    Attention, BoundDecoderResources, DecoderModelSpec, DecoderRecipe, FeedForward,
    LayerSegmentPlan, StandardTensorPlacement, StandardTensorPlan,
};
use ferrule_model::{ModelExecutionBackend, ModelFamily, TokenizerHandle};

use ferrule_runtime::engine::PipelineInferenceEngine;
use ferrule_runtime::parallel::collective::HostCollectiveLimits;
use ferrule_runtime::parallel::expert::{ExpertGroup, ExpertParallelExecutor, ExpertRankWorker};
use ferrule_runtime::parallel::pipeline::{
    BoxedPipelineStageWorker, PipelineConfig, PipelineParallelExecutor, PipelineRank,
    PipelineStage, PipelineStageBoot, PipelineStageWorker,
};

#[cfg(unix)]
use ferrule_runtime::parallel::process::decoder::{
    DECODER_WIRE_VERSION, DecoderBoot, DecoderDevice, DecoderPrecision, DecoderRecipeKind,
    ExpertPlacementFrame, KvConfigFrame, ProcessPipelineTransport, SegmentFrame,
};
#[cfg(unix)]
use ferrule_runtime::parallel::process::{
    PROCESS_REAPER_CAPACITY, ProcessGroupEpoch, ProcessIdentity, ProcessLaunch, ProcessOwnerConfig,
    ProcessOwnerInstanceId,
};

use super::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PipelineRankBackend {
    #[default]
    Thread,
    Process,
}

/// Requested serving topology. EP uses explicit per-stage expert groups, not
/// the common PP/KV mesh. Neither transport supports automatic restart/replay.
#[derive(Debug, Clone)]
pub struct PipelineBuildOptions {
    pub parallelism: ParallelismPlan,
    pub rank_backend: PipelineRankBackend,
    /// Required for process ranks; executes a DecoderEndpoint-compatible child.
    /// stdin/stdout must be reserved for protocol frames, including nested EP.
    /// Standard serving uses default frames; dense hybrid grows negotiated frames
    /// for full-vocabulary logits up to ProcessFrameLimits::MAX_FRAME_BYTES.
    /// The executable must accept that trusted ceiling. Never used in thread mode.
    #[cfg(unix)]
    pub process_launch: Option<ProcessLaunch>,
    /// Roots: PP-stage-major then TP-rank, followed by stage EP children.
    /// Dense TP requires distinct devices; dedicated Qwen35 hybrid TP admits colocation.
    /// Otherwise CUDA ordinals: PP owners first, then each stage's EP owners in stage order
    /// (EP owners exist only when EP > 1). Defaults to 0..owner_count. Explicit
    /// repeated ordinals permit colocation; logical owner IDs remain distinct.
    pub devices: Option<Vec<usize>>,
    pub rank_timeout: Duration,
    pub rank_restarts: usize,
}

impl Default for PipelineBuildOptions {
    fn default() -> Self {
        Self {
            parallelism: ParallelismPlan::default(),
            rank_backend: PipelineRankBackend::Thread,
            #[cfg(unix)]
            process_launch: None,
            devices: None,
            rank_timeout: Duration::from_secs(30),
            rank_restarts: 0,
        }
    }
}

impl PipelineBuildOptions {
    pub(super) fn owner_layout(&self) -> execution_policy::OwnerLayout {
        if self.parallelism.expert_parallel > 1 {
            execution_policy::OwnerLayout::StageRootsWithExperts
        } else {
            execution_policy::OwnerLayout::StageRoots
        }
    }

    pub(super) fn owner_count(&self) -> Result<usize> {
        execution_policy::validate_topology(self.parallelism, self.owner_layout())
            .map(|topology| topology.logical_owners())
            .map_err(|e| invalid(e.to_string()))
    }

    pub(super) fn standard_policy(&self) -> execution_policy::TopologyPolicy {
        use execution_policy::*;
        TopologyPolicy {
            layout: self.owner_layout(),
            execution: ExecutionCapabilities {
                ranks: RankCapability::ThreadOrProcess,
                tensor: TensorCapability::DeviceThreads { degrees: &[2, 4] },
            },
            placement: PlacementPolicy {
                selection: DeviceSelection::AutomaticOrExplicit,
                sharing: if self.parallelism.tensor_parallel > 1 {
                    DeviceSharing::Distinct
                } else {
                    DeviceSharing::Colocated
                },
            },
        }
    }

    pub(super) fn validate(&self, backend: ModelExecutionBackend) -> Result<()> {
        if self.parallelism.tensor_parallel > 1 && self.parallelism.expert_parallel > 1 {
            return Err(invalid("EP x TP serving is unsupported"));
        }
        let policy = self.standard_policy();
        self.validate_policy(backend, policy.layout, policy.execution, policy.placement)?;
        self.require_compiled_backend(backend)
    }

    pub(super) fn require_compiled_backend(&self, backend: ModelExecutionBackend) -> Result<()> {
        if backend == ModelExecutionBackend::Cuda && !cfg!(feature = "cuda") {
            return Err(invalid("pipeline CUDA serving requires the 'cuda' feature"));
        }
        Ok(())
    }

    pub(super) fn validate_policy(
        &self,
        backend: ModelExecutionBackend,
        layout: execution_policy::OwnerLayout,
        capabilities: execution_policy::ExecutionCapabilities,
        placement: execution_policy::PlacementPolicy,
    ) -> Result<execution_policy::ValidatedTopology> {
        use execution_policy::*;
        let topology =
            validate_topology(self.parallelism, layout).map_err(|e| self.policy_error(e))?;
        let target = match backend {
            ModelExecutionBackend::Cpu => PlacementTarget::Host,
            ModelExecutionBackend::Cuda => PlacementTarget::Device,
        };
        validate_execution_kind(
            topology,
            match self.rank_backend {
                PipelineRankBackend::Thread => RankTransport::Thread,
                PipelineRankBackend::Process => RankTransport::Process,
            },
            target,
            capabilities,
        )
        .map_err(|e| {
            invalid(match e {
                PolicyError::TensorRequiresDeviceThreads
                    if backend == ModelExecutionBackend::Cpu =>
                {
                    "dense TP serving requires CUDA; CPU TP is unsupported".into()
                }
                PolicyError::TensorRequiresDeviceThreads => {
                    "process TP serving is unsupported; use thread ranks".into()
                }
                PolicyError::TensorUnsupported => "supported TP degrees are 1, 2 and 4".into(),
                _ => e.to_string(),
            })
        })?;
        let transport = self.validated_transport()?;
        validate_capabilities(topology, transport, target, capabilities)
            .map_err(|e| self.policy_error(e))?;
        validate_placement(topology, target, self.devices.as_deref(), placement)
            .map_err(|e| self.policy_error(e))?;
        #[cfg(unix)]
        if transport.requested().ranks == RankTransport::Process {
            ProcessOwnerConfig {
                startup_timeout: transport.timeout(),
                command_timeout: transport.timeout(),
                ..Default::default()
            }
            .validate()
            .map_err(|e| invalid(e.to_string()))?;
        }
        Ok(topology)
    }

    pub(super) fn validated_transport(&self) -> Result<execution_policy::ValidatedTransport<'_>> {
        use execution_policy::*;
        #[cfg(unix)]
        let (launch_executable, platform) = (
            self.process_launch
                .as_ref()
                .map(|p| p.executable.as_os_str()),
            ProcessSupport::Pipes {
                owner_capacity: PROCESS_REAPER_CAPACITY,
            },
        );
        #[cfg(not(unix))]
        let (launch_executable, platform) = (None, ProcessSupport::Unavailable);
        validate_transport(
            TransportRequest {
                ranks: match self.rank_backend {
                    PipelineRankBackend::Thread => RankTransport::Thread,
                    PipelineRankBackend::Process => RankTransport::Process,
                },
                launch_executable,
                timeout: Some(self.rank_timeout),
                restarts: Some(self.rank_restarts),
            },
            platform,
        )
        .map_err(|e| self.policy_error(e))
    }

    fn policy_error(&self, error: execution_policy::PolicyError) -> Error {
        use execution_policy::PolicyError;
        invalid(match error {
            PolicyError::RestartUnsupported => "pipeline serving does not support rank restart/replay".into(),
            PolicyError::ThreadDeadlineUnsupported => "rank timeout is a process capability; thread serving is cooperative".into(),
            PolicyError::ProcessUnavailable => "pipeline process serving requires Unix process pipes".into(),
            PolicyError::LaunchRequired => "process serving requires an explicit process_launch".into(),
            PolicyError::LaunchWithThreads => "process_launch requires process ranks".into(),
            PolicyError::ProcessCapacity => "pipeline PP+EP owner count exceeds process capacity".into(),
            PolicyError::DevicePlacementOnHost => "--devices requires the CUDA pipeline backend".into(),
            PolicyError::DuplicateDevice => "TP requires one distinct CUDA device per physical owner".into(),
            PolicyError::DeviceCount { expected, .. } => format!("--devices requires {expected} CUDA ordinals in PP-stage-major then TP-rank / stage EP order, each within i32"),
            PolicyError::DeviceOrdinal => "CUDA ordinals must be within i32; default CUDA placement exceeds the device ordinal ABI".into(),
            PolicyError::TensorUnsupported => "supported TP degrees are 1, 2 and 4".into(),
            _ => error.to_string(),
        })
    }

    /// Legacy topology default, not checkpoint dtype or a frozen serving selection.
    /// CUDA and injected EP default to F32; CPU non-EP keeps BF16 compatibility.
    /// Explicit native BF16 serving resolves precision separately; inspect the
    /// build plan's support/arithmetic rather than this default helper.
    pub fn precision(&self, backend: ModelExecutionBackend) -> ExecutionPrecisionPolicy {
        if backend == ModelExecutionBackend::Cuda || self.parallelism.expert_parallel > 1 {
            ExecutionPrecisionPolicy::f32()
        } else {
            ExecutionPrecisionPolicy::bf16_compatibility()
        }
    }

    #[cfg(unix)]
    fn process_timeout_ms(&self) -> Result<u64> {
        execution_policy::validate_process_timeout(self.rank_timeout)
            .map_err(|e| invalid(e.to_string()))
    }

    #[cfg(unix)]
    fn process_owner_config(&self) -> ProcessOwnerConfig {
        super::parallel::inputs::process_owner_config(self.rank_timeout)
    }

    fn device(&self, owner: usize) -> usize {
        self.devices
            .as_ref()
            .map_or(owner, |devices| devices[owner])
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidRequest {
        message: message.into(),
    }
}

const DENSE_OPTIONS: ModelOptionPolicy = ModelOptionPolicy {
    serving: standard_serving_options,
    parallel: attach_dense,
    adapt: generic_options,
    finish: no_provider_admission,
};
const MOE_OPTIONS: ModelOptionPolicy = ModelOptionPolicy {
    serving: standard_serving_options,
    parallel: attach_moe,
    adapt: generic_options,
    finish: no_provider_admission,
};

fn standard_serving_options(
    family: &ModelFamily,
    serving: &ServingModelOptions,
    options: ModelFactoryOptions,
    construction: ResolvedConstruction,
) -> Result<ModelFactoryOptions> {
    if options.qwen35_host_cache.is_some() {
        return Err(invalid(
            "pipeline serving does not implement expert prewarm",
        ));
    }
    common_serving_options(family, serving, options, construction)
}

pub(super) fn standard_execution(
    backend: ModelExecutionBackend,
    options: &PipelineBuildOptions,
) -> ResolvedExecution {
    let mut execution = ResolvedExecution::new(
        backend,
        ResolvedConstruction::Pipeline(build_pipeline),
        StorageEncoding::ProfileDefined,
        if options.precision(backend) == ExecutionPrecisionPolicy::bf16_compatibility() {
            ArithmeticPrecision::Bf16Compatibility
        } else {
            ArithmeticPrecision::F32
        },
        "serial pipeline; no mixed batches/cohort deferral/proposals/restart; CUDA thread-only dense TP2/4; no EP x TP",
    );
    execution.topology_policy = Some(options.standard_policy());
    execution
}

pub(super) fn select_dense_strategy(
    entry: &ModelImplementation,
    backend: BackendSelection,
    request: &SelectionRequest,
    options: &ModelFactoryOptions,
) -> Result<ResolvedStrategy> {
    if request.numeric_model.is_some() {
        request.reject_specialized()?;
    }
    if request.serving.model_precision == Some(ModelPrecision::Bf16) {
        return bf16_serving::select_dense(entry, backend, request, options);
    }
    if request.serving.kv_execution_policy != ferrule_runtime::engine::KvExecutionPolicy::Resident {
        return dense_f32_kv_policy::select(entry, backend, request, options);
    }
    request.reject_specialized()?;
    select_standard_strategy(
        entry,
        backend,
        request.serving.parallel.as_ref(),
        &DENSE_OPTIONS,
    )
}

pub(super) fn select_moe_strategy(
    entry: &ModelImplementation,
    backend: BackendSelection,
    request: &SelectionRequest,
    _: &ModelFactoryOptions,
) -> Result<ResolvedStrategy> {
    request.reject_specialized()?;
    let parallel = request.serving.parallel.as_ref();
    if parallel.is_some_and(|p| p.parallelism.tensor_parallel > 1) {
        return Err(invalid(
            "TP serving requires dense Qwen3 or the dedicated Qwen35Moe hybrid factory",
        ));
    }
    select_standard_strategy(entry, backend, parallel, &MOE_OPTIONS)
}

fn select_standard_strategy(
    entry: &ModelImplementation,
    backend: BackendSelection,
    parallel: Option<&PipelineBuildOptions>,
    policy: &'static ModelOptionPolicy,
) -> Result<ResolvedStrategy> {
    let backend =
        Option::<ModelExecutionBackend>::from(backend).unwrap_or(ModelExecutionBackend::Cpu);
    match parallel {
        Some(options) => {
            options.validate(backend)?;
            Ok(ResolvedStrategy {
                execution: standard_execution(backend, options),
                backend,
                catalog_backend: BackendSelection::Cpu,
                resolve: entry.resolve,
                policy,
                parallel: parallel.cloned(),
            })
        }
        None => Ok(ResolvedStrategy {
            execution: ResolvedExecution::new(
                backend,
                ResolvedConstruction::Resident(entry.build),
                StorageEncoding::ProfileDefined,
                ArithmeticPrecision::Bf16Compatibility,
                "resident catalog/configuration validated; checkpoint binding not yet validated",
            ),
            backend,
            catalog_backend: backend.into(),
            resolve: entry.resolve,
            policy: &RESIDENT_OPTIONS,
            parallel: None,
        }),
    }
}

pub(super) fn attach_dense(
    plan: ResidentModelBuildPlan,
    options: PipelineBuildOptions,
    execution: ResolvedExecution,
) -> Result<ResidentModelBuildPlan> {
    if options.parallelism.expert_parallel > 1 {
        return Err(invalid("EP > 1 requires a routed-expert model"));
    }
    let plan = plan.with_resolved_pipeline(options.clone(), execution)?;
    if options.parallelism.tensor_parallel > 1 {
        // Reuse strict model metadata validation without weight reads or CUDA.
        let path = plan.request.model_path.join("config.json");
        let bytes =
            std::fs::read(&path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
        let value = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        let spec = ferrule_model::models::qwen3::Qwen3DenseRecipe::new()
            .build_spec(&value)
            .map_err(|e| invalid(e.to_string()))?;
        let plans = segment_plans(
            plan.request.max_layers,
            options.parallelism.pipeline_parallel,
        )?;
        validate_tensor_spec(&spec, &plans, &options)?;
        let (config, _) = pipeline_capacity(
            &plan.request,
            &options,
            &spec,
            Qwen3DensePrepareOptions::default().page_size,
            &plans,
        )?;
        tensor_collective_limits(
            &spec,
            config.max_batch_tokens,
            options.parallelism.tensor_parallel,
        )?;
    }

    Ok(plan)
}

pub(super) fn attach_moe(
    plan: ResidentModelBuildPlan,
    options: PipelineBuildOptions,
    execution: ResolvedExecution,
) -> Result<ResidentModelBuildPlan> {
    if options.parallelism.tensor_parallel > 1 {
        return Err(invalid(
            "standard MoE pipeline does not implement tensor parallelism",
        ));
    }
    plan.with_resolved_pipeline(options, execution)
}

impl ResidentModelPlanner {
    /// Resolve pipeline execution without advertising CUDA support on the
    /// catalog's CPU resident adapters. Planning creates no device resources.
    /// Auto preserves the catalog's CPU default; CUDA must be explicit.
    pub fn prepare_pipeline(
        &self,
        config: &AutoConfig,
        backend: BackendSelection,
        chat_template_override: Option<&str>,
        options: ModelFactoryOptions,
        pipeline: PipelineBuildOptions,
    ) -> Result<ResidentModelBuildPlan> {
        let implementation = implementation(&config.descriptor().spec.family)?;
        let strategy = (implementation.select)(
            implementation,
            backend,
            &SelectionRequest::parallel(Some(pipeline.clone())),
            &options,
        )?;
        self.prepare_strategy(
            config,
            backend,
            chat_template_override,
            options,
            Some(pipeline),
            strategy,
        )
    }
}

impl ResidentModelBuildPlan {
    /// Select the real pipeline engine without replacing the model registry or
    /// changing the owner-thread factory boundary. No model/device is built here.
    pub fn with_pipeline(self, options: PipelineBuildOptions) -> Result<Self> {
        self.validate_pipeline_conversion()?;
        let implementation = implementation(&self.request.family)?;
        let strategy = (implementation.select)(
            implementation,
            self.backend.into(),
            &SelectionRequest::parallel(Some(options.clone())),
            &self.requested_options,
        )?;
        if strategy.execution.construction.kind() != SelectedImplementationKind::Pipeline {
            return Err(invalid(
                "with_pipeline requires a pipeline construction strategy",
            ));
        }
        let mut plan = strategy.attach(self, options)?;
        plan.refresh_facts()?;
        Ok(plan)
    }

    // Validate before selecting a replacement strategy or reading tensor metadata.
    pub(super) fn validate_pipeline_conversion(&self) -> Result<()> {
        match self.request.execution.conversion {
            PipelineConversion::Allowed => Ok(()),
            PipelineConversion::Frozen(reason) => Err(invalid(reason)),
        }
    }

    pub(super) fn with_resolved_pipeline(
        mut self,
        options: PipelineBuildOptions,
        execution: ResolvedExecution,
    ) -> Result<Self> {
        if self.request.moe_hotset_experts != 0 {
            return Err(invalid(
                "pipeline serving does not implement moe_hotset_experts (--moe-hotset-experts)",
            ));
        }
        // The shared resident defaults are not pipeline cache capabilities.
        // Reject overrides rather than silently promising an unused policy.
        if self.request.expert_memory_policy != ExpertMemoryPolicy::default() {
            return Err(invalid(
                "pipeline serving does not implement expert_cache overrides (host/pinned entries or budgets)",
            ));
        }
        if options.parallelism.expert_parallel > self.request.model_info.num_experts.max(1) {
            return Err(invalid("EP cannot exceed the model expert count"));
        }
        if self.request.max_layers != self.request.model_info.num_layers {
            return Err(invalid(
                "pipeline serving requires all checkpoint layers; partial --max-layers is unsupported",
            ));
        }
        if options.parallelism.pipeline_parallel > self.request.max_layers {
            return Err(invalid("pipeline degree exceeds the model layer count"));
        }
        let scheduler = &mut self.request.scheduler_config;
        if scheduler.max_active_sequences == 0
            || scheduler.max_batch_tokens == 0
            || scheduler.prefill_chunk_size == 0
            || scheduler.prefix_cache_capacity_pages != 0
            || self.request.driver_config.ctx_size == 0
        {
            return Err(invalid(
                "pipeline serving requires nonzero bounded capacities and no prefix cache",
            ));
        }
        // Resolved capabilities: no packed decode, mixed batches or cohort formation.
        scheduler.max_decode_batch = 1;
        scheduler.allow_mixed_batches = false;
        scheduler.decode_cohort_target = 1;
        scheduler.decode_cohort_max_deferrals = 0;
        scheduler.max_batch_tokens = scheduler
            .max_batch_tokens
            .min(self.request.driver_config.ctx_size);
        scheduler.prefill_chunk_size = scheduler.prefill_chunk_size.min(scheduler.max_batch_tokens);
        self.request.driver_config.enable_native_proposals = false;
        self.request.execution = execution;
        self.request.execution.adjustment_reason = "generic pipeline is serial; batch <= context and prefill <= batch; no mixed batches/cohort deferral/native proposals";

        self.backend_profile = match (
            options.rank_backend,
            self.backend,
            options.parallelism.expert_parallel > 1,
        ) {
            (PipelineRankBackend::Thread, ModelExecutionBackend::Cpu, false) => {
                "cpu-pipeline-serial"
            }
            (PipelineRankBackend::Thread, ModelExecutionBackend::Cpu, true) => {
                "cpu-pipeline-ep-f32-serial"
            }
            (PipelineRankBackend::Thread, ModelExecutionBackend::Cuda, _)
                if execution.support.storage == StorageEncoding::NativeBf16 =>
            {
                "cuda-dense-bf16-serial"
            }
            (PipelineRankBackend::Thread, ModelExecutionBackend::Cuda, _) => {
                "cuda-pipeline-f32-serial"
            }
            (PipelineRankBackend::Process, ModelExecutionBackend::Cpu, false) => {
                "cpu-pipeline-process-bf16-serial"
            }
            (PipelineRankBackend::Process, ModelExecutionBackend::Cpu, true) => {
                "cpu-pipeline-process-ep-f32-serial"
            }
            (PipelineRankBackend::Process, ModelExecutionBackend::Cuda, _) => {
                "cuda-pipeline-process-f32-serial"
            }
        };
        self.requested_parallel = Some(options.clone());
        self.request.pipeline = Some(options);
        Ok(self)
    }

    pub fn pipeline_options(&self) -> Option<&PipelineBuildOptions> {
        self.request.pipeline.as_ref()
    }
}

// These adapters only bind checkpoint metadata/limits here. Never call
// into_decoder: stage factories alone construct computation and physical KV.
enum Checkpoint {
    Dense(Qwen3DenseAdapter),
    DenseBf16 {
        resources: BoundDecoderResources,
        schema: StandardGqaPlanes,
    },
    DenseTensor {
        resources: BoundDecoderResources,
        schema: StandardGqaPlanes,
    },
    Moe(Qwen3MoeAdapter),
}
impl Checkpoint {
    fn load(request: &ResolvedModelRequest) -> ferrule_common::Result<Self> {
        match request.family {
            ModelFamily::Qwen3
                if request.execution.support.storage == StorageEncoding::NativeBf16 =>
            {
                let (config, checkpoint) =
                    Qwen3DenseAdapter::bind_hf_metadata(&request.model_path)?;
                let resources = checkpoint.into_resources();
                ferrule_model::transformer::Bf16DenseCudaProfile::validate(&resources)?;
                resources.validate_parameter_limits(
                    request.max_tensor_bytes,
                    request.max_tensor_bytes,
                )?;
                let schema = StandardGqaPlanes::new(
                    config.num_hidden_layers,
                    config.num_key_value_heads,
                    config.head_dim,
                    Qwen3DensePrepareOptions::default().page_size,
                    config.max_position_embeddings,
                    KvElementType::Bf16,
                )?;
                Ok(Self::DenseBf16 { resources, schema })
            }
            ModelFamily::Qwen3
                if request
                    .pipeline
                    .as_ref()
                    .is_some_and(|p| p.parallelism.tensor_parallel > 1) =>
            {
                let options = request.pipeline.as_ref().expect("TP options");
                let (config, checkpoint) =
                    Qwen3DenseAdapter::bind_hf_metadata(&request.model_path)?;
                let resources = checkpoint.into_resources();
                let tensor = StandardTensorPlan::new(
                    resources.spec(),
                    (0..options.parallelism.tensor_parallel)
                        .map(|rank| StandardTensorPlacement {
                            owner: ParallelRankId::new(rank as u32),
                            device: options.device(rank),
                        })
                        .collect(),
                )?;
                // Parent and every owner use exactly the materializer's dense
                // limit. Only bindings are retained; payloads remain owner-local.
                Qwen3DenseAdapter::validate_tensor_read_limits(
                    &resources,
                    &tensor,
                    request.max_tensor_bytes,
                )?;
                let schema = StandardGqaPlanes::new(
                    config.num_hidden_layers,
                    config.num_key_value_heads,
                    config.head_dim,
                    Qwen3DensePrepareOptions::default().page_size,
                    config.max_position_embeddings,
                    KvElementType::F32,
                )?;
                Ok(Self::DenseTensor { resources, schema })
            }
            ModelFamily::Qwen3 => Ok(Self::Dense(Qwen3DenseAdapter::load_hf_with_options(
                &request.model_path,
                request.max_tensor_bytes,
                Qwen3DensePrepareOptions {
                    max_dense_tensor_bytes: request.max_tensor_bytes,
                    max_layers: Some(request.max_layers),
                    ..Default::default()
                },
            )?)),
            ModelFamily::QwenMoe => Ok(Self::Moe(Qwen3MoeAdapter::load_hf_with_options(
                &request.model_path,
                request.max_tensor_bytes,
                qwen_prepare_options(request),
            )?)),
            _ => Err(ferrule_common::Error::Execution {
                message: "unsupported pipeline family".into(),
            }),
        }
    }
    fn resources(&self) -> &BoundDecoderResources {
        match self {
            Self::Dense(a) => a.resources(),
            Self::DenseTensor { resources, .. } | Self::DenseBf16 { resources, .. } => resources,
            Self::Moe(a) => a.resources(),
        }
    }
    fn schema(&self) -> &StandardGqaPlanes {
        match self {
            Self::Dense(a) => a.kv_schema(),
            Self::DenseTensor { schema, .. } | Self::DenseBf16 { schema, .. } => schema,
            Self::Moe(a) => a.kv_schema(),
        }
    }
}

fn expert_group(
    resources: &BoundDecoderResources,
    plan: &LayerSegmentPlan,
    stage: usize,
    options: &PipelineBuildOptions,
    max_batch_tokens: usize,
    geometry: &super::parallel::plan::Geometry,
) -> Result<Option<ExpertGroup>> {
    let ep = options.parallelism.expert_parallel;
    if ep == 1 {
        return Ok(None);
    }
    let members = geometry.expert_members(stage)?;
    let mut entries = Vec::new();
    let mut top_k = 0;
    for layer in plan.layers() {
        if let FeedForward::Moe(moe) = resources.spec().layers()[layer].feed_forward() {
            let router = moe.router_spec();
            if ep > router.num_experts() {
                return Err(invalid("EP degree exceeds a routed layer's expert count"));
            }
            top_k = top_k.max(router.experts_per_token());
            entries.extend(
                (0..router.num_experts()).map(|expert| (layer, expert, members[expert % ep])),
            );
        }
    }
    if top_k == 0 {
        return Err(invalid("EP stage has no routed layer"));
    }
    let max_tokens = max_batch_tokens
        .checked_mul(top_k)
        .ok_or_else(|| invalid("EP token bound overflow"))?;
    let max_bytes = max_tokens
        .checked_mul(resources.spec().hidden_size())
        .and_then(|size| size.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| invalid("EP activation byte bound overflow"))?;
    Ok(Some(ExpertGroup {
        source_rank: match options.rank_backend {
            PipelineRankBackend::Thread => members[0],
            PipelineRankBackend::Process => geometry.root_members(stage)?[0],
        },
        members,
        layers: plan.layers(),
        placement: ExpertPlacement::new(entries)?,
        limits: ExpertDispatchLimits {
            max_tokens,
            max_bytes,
        },
    }))
}

fn prepare_stage(
    request: ResolvedModelRequest,
    boot: PipelineStageBoot,
    group: Option<ExpertGroup>,
) -> ferrule_common::Result<BoxedPipelineStageWorker> {
    let checkpoint = Checkpoint::load(&request)?;
    let resources = checkpoint.resources();
    let config = boot.config;
    let options = request
        .pipeline
        .as_ref()
        .expect("validated pipeline options");
    let plans =
        segment_plans(request.max_layers, options.parallelism.pipeline_parallel).map_err(|e| {
            ferrule_common::Error::Execution {
                message: e.to_string(),
            }
        })?;
    let (live_config, live_plan) = pipeline_capacity(
        &request,
        options,
        resources.spec(),
        checkpoint.schema().page_size(),
        &plans,
    )
    .map_err(|e| ferrule_common::Error::Execution {
        message: e.to_string(),
    })?;
    if live_config != config || live_plan.configured_pages != boot.physical_pages {
        return Err(ferrule_common::Error::Execution {
            message: "pipeline owner capacity changed after planning".into(),
        });
    }
    match request.backend {
        ModelExecutionBackend::Cpu => {
            let stage = if let Some(group) = group {
                let experts = ExpertParallelExecutor::new(group, move |rank, group| {
                    let checkpoint = Checkpoint::load(&request)?;
                    ExpertRankWorker::prepare_cpu(
                        checkpoint.resources(),
                        rank,
                        group,
                        config.max_parameter_bytes,
                    )
                })?;
                PipelineStage::prepare_cpu_with_experts(
                    resources,
                    boot.plan.clone(),
                    config,
                    experts,
                )?
            } else {
                PipelineStage::prepare_cpu(resources, boot.plan.clone(), config)?
            };
            Ok(PipelineStageWorker::new(&boot, stage)?.boxed())
        }
        #[cfg(feature = "cuda")]
        ModelExecutionBackend::Cuda => {
            use ferrule_backend::cuda::operators::linear::CudaOperators;
            use std::rc::Rc;
            // Device/context creation is exclusively inside persistent owners.
            let options = request
                .pipeline
                .as_ref()
                .expect("validated pipeline options");
            let ops = Rc::new(CudaOperators::new_on_device(
                options.device(boot.rank.local.get() as usize),
            )?);
            let stage = if let Some(group) = group {
                let experts = ExpertParallelExecutor::new_cuda(group, move |rank, group| {
                    let checkpoint = Checkpoint::load(&request)?;
                    let options = request
                        .pipeline
                        .as_ref()
                        .expect("validated pipeline options");
                    let ops = Rc::new(CudaOperators::new_on_device(
                        options.device(rank.owner.get() as usize),
                    )?);
                    ExpertRankWorker::prepare_cuda(
                        checkpoint.resources(),
                        rank,
                        group,
                        config.max_parameter_bytes,
                        ops,
                    )
                })?;
                PipelineStage::prepare_cuda_with_experts(
                    resources,
                    boot.plan.clone(),
                    config,
                    ops,
                    experts,
                )?
            } else {
                PipelineStage::prepare_cuda(resources, boot.plan.clone(), config, ops)?
            };
            Ok(PipelineStageWorker::new(&boot, stage)?.boxed())
        }
        #[cfg(not(feature = "cuda"))]
        ModelExecutionBackend::Cuda => Err(ferrule_common::Error::Execution {
            message: "pipeline CUDA serving requires the 'cuda' feature".into(),
        }),
    }
}

pub(in crate::model_factory) fn segment_plans(
    layers: usize,
    degree: usize,
) -> Result<Vec<LayerSegmentPlan>> {
    (0..degree)
        .map(|stage| {
            let base = layers / degree;
            let remainder = layers % degree;
            let start = stage * base + stage.min(remainder);
            let end = start + base + usize::from(stage < remainder);
            LayerSegmentPlan::new(layers, start..end, stage == 0, stage + 1 == degree)
                .map_err(|e| invalid(e.to_string()))
        })
        .collect()
}

fn validate_tensor_spec(
    spec: &DecoderModelSpec,
    plans: &[LayerSegmentPlan],
    options: &PipelineBuildOptions,
) -> Result<Vec<StandardTensorPlan>> {
    let tp = options.parallelism.tensor_parallel;
    plans
        .iter()
        .enumerate()
        .map(|(stage, segment)| {
            let tensor = StandardTensorPlan::new(
                spec,
                (0..tp)
                    .map(|rank| {
                        let owner = stage * tp + rank;
                        StandardTensorPlacement {
                            owner: ParallelRankId::new(owner as u32),
                            device: options.device(owner),
                        }
                    })
                    .collect(),
            )?;
            tensor.validate_segment(spec, segment)?;
            Ok(tensor)
        })
        .collect()
}

fn tensor_collective_limits(
    spec: &DecoderModelSpec,
    rows: usize,
    tp: usize,
) -> Result<HostCollectiveLimits> {
    let elements = rows
        .checked_mul(spec.hidden_size().max(spec.vocab_size().div_ceil(tp)))
        .ok_or_else(|| invalid("TP collective element bound overflow"))?;
    // AllGather keeps all inputs plus a full gathered output for every peer.
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|n| n.checked_mul(tp))
        .and_then(|n| n.checked_mul(tp + 1))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| invalid("TP collective host byte bound overflow"))?;
    Ok(HostCollectiveLimits {
        max_ranks: tp,
        max_elements_per_rank: elements,
        max_host_bytes: bytes,
    })
}

fn pipeline_capacity(
    request: &ResolvedModelRequest,
    options: &PipelineBuildOptions,
    spec: &DecoderModelSpec,
    page_size: usize,
    plans: &[LayerSegmentPlan],
) -> Result<(PipelineConfig, ferrule_runtime::engine::ResidentKvPagePlan)> {
    if spec.layers().len() != request.max_layers {
        return Err(invalid(
            "pipeline checkpoint layer count changed after planning",
        ));
    }
    if spec
        .max_sequence_length()
        .is_some_and(|max| request.driver_config.ctx_size > max)
    {
        return Err(invalid(
            "pipeline context exceeds the checkpoint position range",
        ));
    }
    let precision = request.execution.pipeline_precision();
    let dtype = if precision == ExecutionPrecisionPolicy::f32() {
        KvElementType::F32
    } else {
        KvElementType::Bf16
    };
    let first = spec
        .layers()
        .first()
        .ok_or_else(|| invalid("pipeline model has no decoder layers"))?;
    let Attention::Gqa(attention) = first.attention() else {
        return Err(invalid("pipeline requires standard GQA attention"));
    };
    let tp = options.parallelism.tensor_parallel;
    if !attention.num_kv_heads().is_multiple_of(tp) {
        return Err(invalid(
            "KV heads must be divisible by TP; KV replication is unsupported",
        ));
    }
    let schema = StandardGqaPlanes::new(
        request.max_layers,
        attention.num_kv_heads(),
        attention.head_dim(),
        page_size,
        request.driver_config.ctx_size,
        dtype,
    )?;
    // One logical page spans every layer/global head. Sum segment-local,
    // local-head physical owners, not a full-model replica for each TP rank.
    let mut physical_bytes = 0u64;
    for plan in plans {
        let local = StandardGqaPlanes::new(
            plan.layer_count(),
            attention.num_kv_heads() / tp,
            attention.head_dim(),
            page_size,
            request.driver_config.ctx_size,
            dtype,
        )?;
        physical_bytes = physical_page_bytes(&local)?
            .checked_mul(tp as u64)
            .and_then(|n| physical_bytes.checked_add(n))
            .ok_or_else(|| invalid("pipeline physical KV byte bound overflow"))?;
    }
    if physical_bytes != physical_page_bytes(&schema)? {
        return Err(invalid(
            "physical owner KV does not match parent logical schema",
        ));
    }
    let kv_plan = ferrule_runtime::engine::plan_pipeline_kv_pages(
        &schema,
        physical_bytes,
        request.kv_cache_bytes,
        PhysicalKvLayout::standard(request.backend),
        request.scheduler_config,
        request.driver_config,
    )?;
    let config = PipelineConfig {
        page_size,
        max_pages: kv_plan.full_capacity_pages,
        max_positions: request.driver_config.ctx_size,
        max_batch_tokens: request.scheduler_config.max_batch_tokens,
        session_capacity: request.scheduler_config.max_active_sequences,
        max_parameter_bytes: if options.parallelism.tensor_parallel > 1
            || request.execution.support.storage == StorageEncoding::NativeBf16
        {
            request.max_tensor_bytes
        } else {
            request
                .max_tensor_bytes
                .max(request.expert_reader_max_tensor_bytes)
        },
        precision,
        max_ack_polls: 8,
    };
    config.validate()?;
    if config.physical_pages(PhysicalKvLayout::standard(request.backend))?
        != kv_plan.configured_pages
    {
        return Err(invalid("pipeline factory physical capacity mismatch"));
    }
    let per_device = pipeline_kv_bytes_by_device(spec, options, request.backend, config, plans)?;
    let total = per_device.values().try_fold(0u64, |sum, &bytes| {
        sum.checked_add(bytes)
            .ok_or_else(|| invalid("pipeline placed KV byte overflow"))
    })?;
    if Some(total) != kv_plan.configured_bytes {
        return Err(invalid("pipeline placed KV bytes differ from admission"));
    }
    Ok((config, kv_plan))
}

/// Sum KV owners sharing a card before comparing the global physical KV budget.
/// EP-only owners hold no KV; their separate expert accounting is unchanged.
fn pipeline_kv_bytes_by_device(
    spec: &DecoderModelSpec,
    options: &PipelineBuildOptions,
    backend: ModelExecutionBackend,
    config: PipelineConfig,
    plans: &[LayerSegmentPlan],
) -> Result<std::collections::BTreeMap<usize, u64>> {
    let tp = options.parallelism.tensor_parallel;
    let cuda = backend == ModelExecutionBackend::Cuda;
    let mut totals = std::collections::BTreeMap::<usize, u64>::new();
    for (stage, plan) in plans.iter().enumerate() {
        let Attention::Gqa(attention) = spec.layers()[plan.layers().start].attention() else {
            return Err(invalid("pipeline requires standard GQA attention"));
        };
        let description = ferrule_runtime::parallel::pipeline::PipelineStageDescription {
            plan: plan.clone(),
            config,
            physical_pages: config.physical_pages(PhysicalKvLayout::standard(backend))?,
            hidden: spec.hidden_size(),
            vocabulary: spec.vocab_size(),
            kv_heads: attention.num_kv_heads() / tp,
            head_dim: attention.head_dim(),
            expert_group: None,
        };
        description.validate()?;
        let bytes = description.physical_bytes()?;
        for rank in 0..tp {
            let device = if cuda {
                options.device(stage * tp + rank)
            } else {
                0
            };
            let total = totals.entry(device).or_default();
            *total = total
                .checked_add(bytes)
                .ok_or_else(|| invalid("pipeline same-card KV byte overflow"))?;
        }
    }
    Ok(totals)
}

pub(super) fn dense_bf16_memory_plan(
    request: &ResolvedModelRequest,
) -> Result<ferrule_runtime::parallel::pipeline::DenseBf16MemoryPlan> {
    let options = request
        .pipeline
        .as_ref()
        .ok_or_else(|| invalid("missing BF16 pipeline options"))?;
    let checkpoint = Checkpoint::load(request)?;
    let plans = segment_plans(request.max_layers, 1)?;
    let (config, _) = pipeline_capacity(
        request,
        options,
        checkpoint.resources().spec(),
        checkpoint.schema().page_size(),
        &plans,
    )?;
    Ok(
        ferrule_runtime::parallel::pipeline::DenseBf16MemoryPlan::new(
            checkpoint.resources(),
            &plans[0],
            config,
        )?,
    )
}

struct StandardAdmission {
    plans: Vec<LayerSegmentPlan>,
    config: PipelineConfig,
    kv: ferrule_runtime::engine::ResidentKvPagePlan,
    groups: Vec<Option<ExpertGroup>>,
    topology: ValidatedParallelTopology,
    spec: DecoderModelSpec,
    #[cfg(feature = "cuda")]
    devices: Vec<usize>,
}
impl StandardAdmission {
    fn new(request: &ResolvedModelRequest, checkpoint: &Checkpoint) -> Result<Self> {
        let options = request
            .pipeline
            .as_ref()
            .ok_or_else(|| invalid("missing pipeline build options"))?;
        options.validate(request.backend)?;
        let spec = checkpoint.resources().spec().clone();
        let geometry = super::parallel::plan::Geometry::new(request.max_layers, options)?;
        let plans = geometry.segments.clone();
        if options.parallelism.tensor_parallel > 1 {
            for tensor in validate_tensor_spec(&spec, &plans, options)? {
                tensor.validate_resources(checkpoint.resources())?;
            }
        }
        let (config, kv_plan) = pipeline_capacity(
            request,
            options,
            &spec,
            checkpoint.schema().page_size(),
            &plans,
        )?;
        let groups = plans
            .iter()
            .enumerate()
            .map(|(stage, plan)| {
                expert_group(
                    checkpoint.resources(),
                    plan,
                    stage,
                    options,
                    config.max_batch_tokens,
                    &geometry,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // EP owners are separate compute identities, never physical KV participants.
        let topology = ValidatedParallelTopology::new(
            ParallelTopologyId::new(1),
            geometry.roots as u32,
            ParallelRankId::new(0),
            ParallelismPlan {
                expert_parallel: 1,
                ..options.parallelism
            },
        )
        .map_err(|error| invalid(format!("invalid pipeline topology: {error:?}")))?;
        #[cfg(not(feature = "cuda"))]
        let _ = geometry.devices;
        Ok(Self {
            plans,
            config,
            kv: kv_plan,
            groups,
            topology,
            spec,
            #[cfg(feature = "cuda")]
            devices: geometry.devices,
        })
    }
}
impl super::parallel::assembly::PipelineAssembly for StandardAdmission {
    fn kv(&self) -> ferrule_runtime::engine::ResidentKvPagePlan {
        self.kv
    }
    fn admit(&self, _: &ResolvedModelRequest, _: &BoundDecoderResources) -> Result<()> {
        Ok(())
    }
    fn start(
        self,
        request: &ResolvedModelRequest,
        _: BoundDecoderResources,
    ) -> Result<PipelineParallelExecutor> {
        let Self {
            plans,
            config,
            kv: kv_plan,
            groups,
            topology,
            spec,
            #[cfg(feature = "cuda")]
            devices,
        } = self;
        #[cfg(not(feature = "cuda"))]
        let _ = spec;
        let options = request
            .pipeline
            .as_ref()
            .expect("admitted pipeline options");
        let pipeline = match options.rank_backend {
            #[cfg(feature = "cuda")]
            PipelineRankBackend::Thread if options.parallelism.tensor_parallel > 1 => {
                use ferrule_runtime::parallel::pipeline::StandardCudaTensorConfig;
                let tensor_config = StandardCudaTensorConfig {
                    devices,
                    collective_limits: tensor_collective_limits(
                        &spec,
                        config.max_batch_tokens,
                        options.parallelism.tensor_parallel,
                    )?,
                    collective_timeout: Duration::from_secs(30),
                };
                let owner_request = request.clone();
                PipelineParallelExecutor::new_standard_cuda_tensor(
                    topology,
                    plans,
                    config,
                    &spec,
                    tensor_config,
                    move |_| Ok(Checkpoint::load(&owner_request)?.resources().clone()),
                )?
            }
            PipelineRankBackend::Thread => {
                let boots = plans
                    .into_iter()
                    .enumerate()
                    .map(|(stage, plan)| {
                        let rank = ParallelRankId::new(stage as u32);
                        PipelineStageBoot {
                            rank: PipelineRank {
                                local: rank,
                                global: rank,
                            },
                            plan,
                            physical_pages: kv_plan.configured_pages,
                            config,
                            program_spec: Vec::new(),
                        }
                    })
                    .collect();
                let owner_request = request.clone();
                PipelineParallelExecutor::new_with_factory(topology, boots, move |boot| {
                    let group = groups[boot.rank.local.get() as usize].clone();
                    prepare_stage(owner_request, boot, group)
                })?
            }
            #[cfg(unix)]
            PipelineRankBackend::Process => {
                let boots = process_boots(request, &plans, config, &groups)?;
                let transport = ProcessPipelineTransport::spawn(
                    options
                        .process_launch
                        .clone()
                        .expect("validated process launch"),
                    boots,
                    options.process_owner_config(),
                )?;
                PipelineParallelExecutor::new_with_external_expert_transport(
                    topology, plans, config, transport,
                )?
            }
            #[cfg(not(unix))]
            PipelineRankBackend::Process => {
                return Err(invalid(
                    "pipeline process serving requires Unix process pipes",
                ));
            }
        };
        Ok(pipeline)
    }
}

pub(super) fn build(request: ResolvedModelRequest) -> Result<BoxedSessionInferenceEngine> {
    request
        .pipeline
        .as_ref()
        .ok_or_else(|| invalid("missing pipeline build options"))?
        .validate(request.backend)?;
    let checkpoint = Checkpoint::load(&request)?;
    let admission = StandardAdmission::new(&request, &checkpoint)?;
    let resources = checkpoint.resources().clone();
    drop(checkpoint);
    let tokenizer = TokenizerHandle::load(&request.model_path)?;
    Ok(Box::new(super::parallel::assembly::build_bound(
        request, resources, tokenizer, admission,
    )?))
}

/// Preflight bounded JSON before spawning owners. The wire returns full logits
/// for every row; silently accepting an oversized prefill could lose custody
/// when the child fails to encode its reply. These are conservative bounds, not
/// binary tensor byte counts. The transport still validates actual frames.
#[cfg(unix)]
fn validate_process_frames(
    request: &ResolvedModelRequest,
    config: PipelineConfig,
    groups: &[Option<ExpertGroup>],
) -> Result<()> {
    const OVERHEAD: usize = 16 * 1024;
    const FLOAT_BYTES: usize = 24; // Includes delimiter; exceeds any finite F32 JSON representation.
    let options = request
        .pipeline
        .as_ref()
        .expect("validated pipeline options");
    let limits = options.process_owner_config().frame_limits;
    let max_payload = limits.max_command_bytes.min(limits.max_output_bytes);
    let check = |rows: usize, width: usize, metadata: usize| -> Result<()> {
        let bytes = width
            .checked_mul(FLOAT_BYTES)
            .and_then(|row| row.checked_add(metadata))
            .and_then(|row| row.checked_mul(rows))
            .and_then(|bytes| bytes.checked_add(OVERHEAD))
            .ok_or_else(|| invalid("process IPC capacity overflow"))?;
        if bytes > max_payload {
            return Err(invalid(format!(
                "process IPC capacity needs up to {bytes} JSON bytes, exceeds {max_payload}; reduce --max-batch-tokens/--ctx-size/--max-active-sequences"
            )));
        }
        Ok(())
    };
    check(
        config.max_batch_tokens,
        request
            .model_info
            .vocab_size
            .max(request.model_info.hidden_size),
        512,
    )?;
    check(config.max_pages, 0, 32)?;
    for group in groups.iter().flatten() {
        check(group.limits.max_tokens, request.model_info.hidden_size, 512)?;
    }
    Ok(())
}

#[cfg(unix)]
fn process_boots(
    request: &ResolvedModelRequest,
    plans: &[LayerSegmentPlan],
    config: PipelineConfig,
    groups: &[Option<ExpertGroup>],
) -> Result<Vec<(ProcessIdentity, DecoderBoot)>> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);
    validate_process_frames(request, config, groups)?;
    let epoch = NEXT_EPOCH
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| invalid("process serving epoch exhausted"))?;
    let epoch = ProcessGroupEpoch::new(epoch).map_err(|e| invalid(e.to_string()))?;
    let options = request
        .pipeline
        .as_ref()
        .expect("validated pipeline options");
    let device = |owner: usize| match request.backend {
        ModelExecutionBackend::Cpu => DecoderDevice::Cpu,
        ModelExecutionBackend::Cuda => DecoderDevice::Cuda {
            ordinal: options.device(owner),
        },
    };
    let recipe = match request.family {
        ModelFamily::Qwen3 => DecoderRecipeKind::Qwen3Dense,
        ModelFamily::QwenMoe => DecoderRecipeKind::Qwen3Moe,
        _ => return Err(invalid("unsupported process pipeline family")),
    };
    plans
        .iter()
        .zip(groups)
        .enumerate()
        .map(|(stage, (plan, group))| {
            let experts = group
                .as_ref()
                .map(|group| -> Result<ExpertPlacementFrame> {
                    let mut entries = Vec::new();
                    for layer in group.layers.clone() {
                        for expert in 0..request.model_info.num_experts {
                            let owner = group
                                .placement
                                .owner(layer, expert)
                                .ok_or_else(|| invalid("missing process expert placement"))?;
                            entries.push((layer, expert, owner.get()));
                        }
                    }
                    Ok(ExpertPlacementFrame {
                        source_scope: ferrule_common::topology::ExpertSourceScope::ExternalStage,
                        source: group.source_rank.get(),
                        members: group.members.iter().map(|rank| rank.get()).collect(),
                        entries,
                        max_tokens: group.limits.max_tokens,
                        max_bytes: group.limits.max_bytes,
                        devices: group
                            .members
                            .iter()
                            .map(|rank| device(rank.get() as usize))
                            .collect(),
                        timeout_ms: options.process_timeout_ms()?,
                    })
                })
                .transpose()?;
            let boot = DecoderBoot {
                version: DECODER_WIRE_VERSION,
                hybrid: None,
                rank: stage as u32,
                checkpoint: request.model_path.clone(),
                recipe,
                segment: SegmentFrame::encode(plan),
                precision: if config.precision == ExecutionPrecisionPolicy::f32() {
                    DecoderPrecision::F32
                } else {
                    DecoderPrecision::Bf16Compatibility
                },
                device: device(stage),
                kv: KvConfigFrame::encode_for_device(config, device(stage))?,
                experts,
            };
            boot.stage_boot()?;
            // EP wraps this Boot in a small additional object. Reserve space for it.
            let bytes = serde_json::to_vec(&boot)
                .map_err(|e| invalid(e.to_string()))?
                .len();
            if bytes.checked_add(1024).is_none_or(|bytes| {
                bytes > options.process_owner_config().frame_limits.max_config_bytes
            }) {
                return Err(invalid(
                    "process decoder Boot exceeds the bounded config frame",
                ));
            }
            let identity = ProcessIdentity::new(
                epoch,
                ParallelRankId::new(stage as u32),
                ProcessOwnerInstanceId::new(stage as u64 + 1)
                    .map_err(|e| invalid(e.to_string()))?,
            );
            Ok((identity, boot))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(pp: usize, ep: usize) -> PipelineBuildOptions {
        PipelineBuildOptions {
            parallelism: ParallelismPlan {
                pipeline_parallel: pp,
                expert_parallel: ep,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn pipeline_capacity_separates_logical_physical_and_placed_bytes_without_cuda() {
        let fixture = TensorCheckpointFixture::new(4, false);
        let checkpoint = Checkpoint::load(&fixture.request(1, 2, 4096)).unwrap();
        let spec = checkpoint.resources().spec();
        for (backend, ep, tp) in [
            (ModelExecutionBackend::Cpu, 1, 1),
            (ModelExecutionBackend::Cpu, 2, 1),
            (ModelExecutionBackend::Cuda, 1, 1),
            (ModelExecutionBackend::Cuda, 1, 2),
            (ModelExecutionBackend::Cuda, 1, 4),
        ] {
            for pp in [1, 2] {
                let mut request = fixture.request(pp, tp, 4096);
                request.backend = backend;
                request.driver_config.ctx_size = 2;
                request.scheduler_config.max_batch_tokens = 2;
                request.scheduler_config.prefill_chunk_size = 2;
                request
                    .pipeline
                    .as_mut()
                    .unwrap()
                    .parallelism
                    .expert_parallel = ep;
                // Checkpoint metadata and pure budget planning need no CUDA feature/device.
                let plans = segment_plans(2, pp).unwrap();
                let options = request.pipeline.as_ref().unwrap().clone();
                request.execution = standard_execution(backend, &options);
                let (config, plan) =
                    pipeline_capacity(&request, &options, spec, 2, &plans).unwrap();
                let cuda = backend == ModelExecutionBackend::Cuda;
                assert_eq!((config.max_pages, plan.full_capacity_pages), (1, 1));
                assert_eq!(plan.configured_pages, if cuda { 2 } else { 1 });
                let bytes = if cuda {
                    512
                } else if ep > 1 {
                    256
                } else {
                    128
                };
                assert_eq!(plan.configured_bytes, Some(bytes));
                request.kv_cache_bytes = Some(bytes);
                assert_eq!(
                    pipeline_capacity(&request, &options, spec, 2, &plans)
                        .unwrap()
                        .1,
                    plan
                );
                request.kv_cache_bytes = Some(bytes - 1);
                assert!(pipeline_capacity(&request, &options, spec, 2, &plans).is_err());
                let mut placement = options.clone();
                placement.devices = Some(vec![3; pp * tp]);
                let totals =
                    pipeline_kv_bytes_by_device(spec, &placement, backend, config, &plans).unwrap();
                assert_eq!(totals.len(), 1);
                assert_eq!(totals[&if cuda { 3 } else { 0 }], bytes);
                #[cfg(unix)]
                if tp == 1 {
                    // Transport changes neither capacity nor decoder boot meaning.
                    request.kv_cache_bytes = Some(bytes);
                    for rank_backend in [PipelineRankBackend::Thread, PipelineRankBackend::Process]
                    {
                        request.pipeline.as_mut().unwrap().rank_backend = rank_backend;
                        let boots =
                            process_boots(&request, &plans, config, &vec![None; pp]).unwrap();
                        for (_, boot) in boots {
                            assert_eq!(boot.version, DECODER_WIRE_VERSION);
                            assert_eq!(boot.kv.max_pages, 1);
                            assert_eq!(boot.kv.physical_pages, plan.configured_pages);
                            assert_eq!(
                                boot.stage_boot().unwrap().physical_pages,
                                plan.configured_pages
                            );
                        }
                    }
                }
            }
        }
    }

    struct TensorCheckpointFixture(std::path::PathBuf);

    impl TensorCheckpointFixture {
        fn new(vocabulary: usize, tied: bool) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "ferrule-factory-tp-budget-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            std::fs::create_dir_all(&path).unwrap();
            let config = serde_json::json!({
                "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3",
                "hidden_size":8, "intermediate_size":21, "num_hidden_layers":2,
                "num_attention_heads":4, "num_key_value_heads":4, "head_dim":2,
                "rms_norm_eps":0.000001, "rope_theta":10000.0, "rope_scaling":null,
                "max_position_embeddings":64, "vocab_size":vocabulary, "tie_word_embeddings":tied,
                "attention_bias":false, "attention_dropout":0.0, "hidden_act":"silu",
                "torch_dtype":"bfloat16", "use_cache":true, "use_sliding_window":false,
                "sliding_window":null, "max_window_layers":2, "initializer_range":0.02,
                "bos_token_id":1, "eos_token_id":2
            });
            std::fs::write(path.join("config.json"), config.to_string()).unwrap();
            let mut tensors = vec![
                ("model.embed_tokens.weight".to_owned(), vec![vocabulary, 8]),
                ("model.norm.weight".to_owned(), vec![8]),
            ];
            if !tied {
                tensors.push(("lm_head.weight".to_owned(), vec![vocabulary, 8]));
            }
            for layer in 0..2 {
                for (name, shape) in [
                    ("input_layernorm.weight", vec![8]),
                    ("post_attention_layernorm.weight", vec![8]),
                    ("self_attn.q_proj.weight", vec![8, 8]),
                    ("self_attn.k_proj.weight", vec![8, 8]),
                    ("self_attn.v_proj.weight", vec![8, 8]),
                    ("self_attn.o_proj.weight", vec![8, 8]),
                    ("self_attn.q_norm.weight", vec![2]),
                    ("self_attn.k_norm.weight", vec![2]),
                    ("mlp.gate_proj.weight", vec![21, 8]),
                    ("mlp.up_proj.weight", vec![21, 8]),
                    ("mlp.down_proj.weight", vec![8, 21]),
                ] {
                    tensors.push((format!("model.layers.{layer}.{name}"), shape));
                }
            }
            let mut header = serde_json::Map::new();
            let mut payload = Vec::new();
            for (name, shape) in tensors {
                let start = payload.len();
                for _ in 0..shape.iter().product::<usize>() {
                    payload.extend_from_slice(&0x3f80u16.to_le_bytes());
                }
                header.insert(name, serde_json::json!({"dtype":"BF16", "shape":shape, "data_offsets":[start, payload.len()]}));
            }
            Self::write_checkpoint(&path, &serde_json::Value::Object(header), &payload);
            // TP metadata binding must not require a tokenizer.
            Self(path)
        }
        fn write_checkpoint(path: &std::path::Path, header: &serde_json::Value, payload: &[u8]) {
            let mut header = serde_json::to_vec(header).unwrap();
            while !header.len().is_multiple_of(8) {
                header.push(b' ');
            }
            let mut file = (header.len() as u64).to_le_bytes().to_vec();
            file.extend(header);
            file.extend_from_slice(payload);
            std::fs::write(path.join("model.safetensors"), file).unwrap();
        }
        fn request(&self, pp: usize, tp: usize, limit: u64) -> ResolvedModelRequest {
            let config = AutoConfig::from_pretrained(&self.0).unwrap();
            ResolvedModelRequest {
                execution: standard_execution(ModelExecutionBackend::Cuda, &options(pp, 1)),
                family: ModelFamily::Qwen3,
                backend: ModelExecutionBackend::Cuda,
                model_path: self.0.clone(),
                model_info: ferrule_model::ModelInfo::from_descriptor(config.descriptor(), "cuda"),
                pipeline: Some(PipelineBuildOptions {
                    parallelism: ParallelismPlan {
                        pipeline_parallel: pp,
                        tensor_parallel: tp,
                        ..Default::default()
                    },
                    ..Default::default()
                }),
                max_layers: 2,
                max_tensor_bytes: limit,
                output_head_chunk_rows: 4,
                // Must not widen the dense TP materializer's read limit.
                expert_reader_max_tensor_bytes: 1 << 20,
                expert_memory_policy: Default::default(),
                family_options: ResolvedFamilyOptions::Generic,
                moe_hotset_experts: 0,
                kv_cache_bytes: None,
                scheduler_config: ferrule_runtime::ResidentSchedulerConfig {
                    max_active_sequences: 1,
                    max_batch_tokens: 4,
                    prefill_chunk_size: 4,
                    ..Default::default()
                },
                driver_config: ferrule_runtime::ResidentTopKDriverConfig {
                    ctx_size: 32,
                    ..Default::default()
                },
            }
        }
    }
    impl Drop for TensorCheckpointFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn tensor_checkpoint_factory_accepts_local_reads_not_full_projection_limits() {
        use ferrule_model::{TensorRole, transformer::StateDictMaterializer};
        for tied in [false, true] {
            let fixture = TensorCheckpointFixture::new(4, tied);
            // Global MLP matrices: 336 bytes. Ragged TP2: 176/160 bytes;
            // TP4: 96/80/80/80 bytes, for Column gate/up AND Row down.
            for (pp, tp, limit) in [(1, 2, 176), (2, 2, 176), (1, 4, 96), (2, 4, 96)] {
                let request = fixture.request(pp, tp, limit);
                for _ in 0..=pp * tp {
                    // The same entrypoint runs in the parent and each owner.
                    assert!(matches!(
                        Checkpoint::load(&request).unwrap(),
                        Checkpoint::DenseTensor { .. }
                    ));
                }
                let checkpoint = Checkpoint::load(&request).unwrap();
                let resources = checkpoint.resources();
                let plans = segment_plans(2, pp).unwrap();
                let options = request.pipeline.as_ref().unwrap();
                let tensor_plans = validate_tensor_spec(resources.spec(), &plans, options).unwrap();
                let (config, _) =
                    pipeline_capacity(&request, options, resources.spec(), 16, &plans).unwrap();
                assert_eq!(config.max_parameter_bytes, limit);
                assert_eq!(config.precision, ExecutionPrecisionPolicy::f32());
                for tensor in tensor_plans {
                    for rank in 0..tp {
                        let materializer = StateDictMaterializer::for_tensor(
                            limit,
                            tensor.clone(),
                            ParallelRankId::new(rank as u32),
                        )
                        .unwrap();
                        for binding in resources.state_dict().parameters() {
                            if matches!(
                                binding.role(),
                                TensorRole::TokenEmbedding
                                    | TensorRole::OutputNorm
                                    | TensorRole::AttentionNorm
                                    | TensorRole::FeedForwardNorm
                                    | TensorRole::AttentionQueryNorm
                                    | TensorRole::AttentionKeyNorm
                            ) {
                                materializer.parameter(binding).unwrap();
                            } else {
                                let linear = materializer
                                    .prepared_linear(binding, binding.role().clone())
                                    .unwrap();
                                let shard = linear.tensor_shard().unwrap();
                                assert!(shard.bytes().len() as u64 <= limit);
                                assert_eq!(
                                    shard.provenance().unwrap().tensor(),
                                    binding.weight().slice()
                                );
                            }
                        }
                    }
                }
                assert!(!fixture.0.join("tokenizer.json").exists());
                let error = Checkpoint::load(&fixture.request(pp, tp, limit - 1))
                    .err()
                    .unwrap()
                    .to_string();
                assert!(
                    error.contains("rank 0") && error.contains("limit"),
                    "{error}"
                );
                // A global/TP average must not admit the larger ragged rank.
                assert!(Checkpoint::load(&fixture.request(pp, tp, 336 / tp as u64)).is_err());
            }
            let mut request = fixture.request(1, 1, 176);
            let error = Checkpoint::load(&request).err().unwrap().to_string();
            assert!(
                error.contains("336 bytes") && error.contains("load limit"),
                "{error}"
            );
            request.pipeline = None;
            request.backend = ModelExecutionBackend::Cpu;
            assert!(
                Checkpoint::load(&request)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("load limit")
            );
        }
    }

    #[test]
    fn tensor_checkpoint_factory_preserves_replicated_embedding_budget() {
        for tied in [false, true] {
            let fixture = TensorCheckpointFixture::new(32, tied);
            for tp in [2, 4] {
                let error = Checkpoint::load(&fixture.request(2, tp, 256))
                    .err()
                    .unwrap()
                    .to_string();
                assert!(
                    error.contains("replicated parameter 'token_embedding.weight'")
                        && error.contains("512 bytes"),
                    "{error}"
                );
            }
        }
    }

    #[test]
    fn tensor_checkpoint_factory_keeps_source_identity_and_reader_checks() {
        use ferrule_model::{TensorRole, transformer::StateDictMaterializer};
        let fixture = TensorCheckpointFixture::new(4, true);
        let request = fixture.request(2, 2, 176);
        let checkpoint = Checkpoint::load(&request).unwrap();
        let resources = checkpoint.resources();
        let embedding = resources
            .require_static(TensorRole::TokenEmbedding)
            .unwrap();
        let head = resources.require_static(TensorRole::OutputHead).unwrap();
        assert!(head.shares_storage_with(embedding));
        let tensor = validate_tensor_spec(
            resources.spec(),
            &segment_plans(2, 2).unwrap(),
            request.pipeline.as_ref().unwrap(),
        )
        .unwrap()
        .remove(0);
        let materializer =
            StateDictMaterializer::for_tensor(176, tensor.clone(), ParallelRankId::new(0)).unwrap();
        let replacement = fixture.0.join("replacement.safetensors");
        std::fs::copy(fixture.0.join("model.safetensors"), &replacement).unwrap();
        std::fs::rename(replacement, fixture.0.join("model.safetensors")).unwrap();
        // Same bytes/length, new inode: do not recapture an old binding's source.
        assert!(!resources.state_dict().validate_source_identities());
        let error =
            Qwen3DenseAdapter::validate_tensor_read_limits(resources, &tensor, 176).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("stale checkpoint source identity"),
            "{error}"
        );
        assert!(materializer.parameter(embedding).is_err());
        assert!(
            materializer
                .prepared_linear(head, TensorRole::OutputHead)
                .is_err()
        );
        assert!(
            materializer
                .prepared_linear(
                    resources
                        .require_layer(0, TensorRole::DenseMlpDown)
                        .unwrap(),
                    TensorRole::DenseMlpDown
                )
                .is_err()
        );
        assert!(materializer.tensor_reads().unwrap().is_empty());
    }

    #[test]
    fn tensor_checkpoint_factory_keeps_strict_index_header_and_binding_validation() {
        for corrupt in [
            "index",
            "missing-shard",
            "index-mismatch",
            "truncated",
            "dtype",
            "shape",
            "name",
        ] {
            let fixture = TensorCheckpointFixture::new(4, true);
            let request = fixture.request(1, 2, 176);
            let path = fixture.0.join("model.safetensors");
            let bytes = std::fs::read(&path).unwrap();
            let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            match corrupt {
                "index" => {
                    std::fs::write(fixture.0.join("model.safetensors.index.json"), "not-json")
                        .unwrap()
                }
                "missing-shard" | "index-mismatch" => {
                    let shard = if corrupt == "missing-shard" {
                        "missing.safetensors"
                    } else {
                        "model.safetensors"
                    };
                    std::fs::write(
                        fixture.0.join("model.safetensors.index.json"),
                        serde_json::json!({
                            "weight_map":{"model.embed_tokens.weight":shard}
                        })
                        .to_string(),
                    )
                    .unwrap();
                }
                "truncated" => std::fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_len((8 + header_len) as u64)
                    .unwrap(),
                _ => {
                    let mut header: serde_json::Value =
                        serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
                    let name = "model.layers.0.mlp.gate_proj.weight";
                    match corrupt {
                        "dtype" => header[name]["dtype"] = serde_json::json!("F16"),
                        "shape" => header[name]["shape"] = serde_json::json!([8, 21]),
                        "name" => {
                            let value = header.as_object_mut().unwrap().remove(name).unwrap();
                            header["model.layers.0.mlp.invalid.weight"] = value;
                        }
                        _ => unreachable!(),
                    }
                    TensorCheckpointFixture::write_checkpoint(
                        &fixture.0,
                        &header,
                        &bytes[8 + header_len..],
                    );
                }
            }
            let error = Checkpoint::load(&request).err().expect(corrupt).to_string();
            assert!(
                !error.contains("read limit"),
                "strict {corrupt} validation masked by budgeting: {error}"
            );
        }
    }

    #[test]
    fn pipeline_precision_and_owner_count_are_explicit() {
        let dense = options(2, 1);
        assert_eq!(dense.owner_count().unwrap(), 2);
        assert_eq!(
            dense.precision(ModelExecutionBackend::Cpu),
            ExecutionPrecisionPolicy::bf16_compatibility()
        );
        assert_eq!(
            dense.precision(ModelExecutionBackend::Cuda),
            ExecutionPrecisionPolicy::f32()
        );
        let ep = options(2, 3);
        assert_eq!(ep.owner_count().unwrap(), 8);
        assert_eq!(
            ep.precision(ModelExecutionBackend::Cpu),
            ExecutionPrecisionPolicy::f32()
        );
        ep.validate(ModelExecutionBackend::Cpu).unwrap();
        for (pp, ep) in [
            (0, 1),
            (1, 0),
            (usize::MAX, 1),
            (1, usize::MAX),
            (usize::MAX, 2),
        ] {
            assert!(
                options(pp, ep)
                    .validate(ModelExecutionBackend::Cpu)
                    .is_err()
            );
        }
    }

    #[test]
    fn tensor_owner_count_is_pp_times_tp_and_never_silently_falls_back() {
        for (pp, tp) in [(1, 2), (1, 4), (2, 2), (3, 4)] {
            let mut tensor = options(pp, 1);
            tensor.parallelism.tensor_parallel = tp;
            assert_eq!(tensor.owner_count().unwrap(), pp * tp);
            assert!(
                tensor
                    .validate(ModelExecutionBackend::Cpu)
                    .unwrap_err()
                    .to_string()
                    .contains("CPU TP")
            );
            #[cfg(feature = "cuda")]
            {
                tensor.validate(ModelExecutionBackend::Cuda).unwrap();
                tensor.devices = Some((0..pp * tp).rev().collect());
                tensor.validate(ModelExecutionBackend::Cuda).unwrap();
                assert_eq!(tensor.device(0), pp * tp - 1);
                tensor.devices = Some(vec![0; pp * tp]);
                assert!(
                    tensor
                        .validate(ModelExecutionBackend::Cuda)
                        .unwrap_err()
                        .to_string()
                        .contains("distinct")
                );
                tensor.devices = None;
            }
            tensor.parallelism.expert_parallel = 2;
            assert!(
                tensor
                    .validate(ModelExecutionBackend::Cuda)
                    .unwrap_err()
                    .to_string()
                    .contains("EP x TP")
            );
            tensor.parallelism.expert_parallel = 1;
            tensor.rank_backend = PipelineRankBackend::Process;
            assert!(
                tensor
                    .validate(ModelExecutionBackend::Cuda)
                    .unwrap_err()
                    .to_string()
                    .contains("process TP")
            );
        }
        let mut overflow = options(usize::MAX, 1);
        overflow.parallelism.tensor_parallel = 4;
        assert!(overflow.owner_count().is_err());
    }

    #[test]
    fn pipeline_rejects_unimplemented_thread_capabilities() {
        let mut cases = vec![
            PipelineBuildOptions {
                rank_backend: PipelineRankBackend::Process,
                ..Default::default()
            },
            PipelineBuildOptions {
                rank_restarts: 1,
                ..Default::default()
            },
            PipelineBuildOptions {
                rank_timeout: Duration::ZERO,
                ..Default::default()
            },
            PipelineBuildOptions {
                devices: Some(vec![0]),
                ..Default::default()
            },
        ];
        for parallelism in [
            ParallelismPlan {
                data_parallel: 2,
                ..Default::default()
            },
            ParallelismPlan {
                tensor_parallel: 2,
                ..Default::default()
            },
            ParallelismPlan {
                sequence_parallel: 2,
                ..Default::default()
            },
            ParallelismPlan {
                context_parallel: 2,
                ..Default::default()
            },
        ] {
            cases.push(PipelineBuildOptions {
                parallelism,
                ..Default::default()
            });
        }
        for case in cases {
            assert!(
                case.validate(ModelExecutionBackend::Cpu).is_err(),
                "{case:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn pipeline_process_boot_preserves_external_stage_identity_and_worker_only_members() {
        use ferrule_common::topology::ExpertSourceScope;
        let fixture = TensorCheckpointFixture::new(4, false);
        let mut request = fixture.request(2, 1, 4096);
        let (_, checkpoint) = Qwen3DenseAdapter::bind_hf_metadata(&fixture.0).unwrap();
        let resources = checkpoint.into_resources();
        request.backend = ModelExecutionBackend::Cpu;
        request.family = ModelFamily::QwenMoe;
        request.model_info.num_experts = 2;
        let mut process = options(2, 2);
        process.rank_backend = PipelineRankBackend::Process;
        process.rank_timeout = Duration::from_millis(7250);
        request.pipeline = Some(process.clone());
        request.execution = standard_execution(request.backend, &process);
        let plans = segment_plans(2, 2).unwrap();
        let (config, _) =
            pipeline_capacity(&request, &process, resources.spec(), 2, &plans).unwrap();
        // Boot projection uses only placement metadata, never expert weights.
        let mut groups = (0..2)
            .map(|stage| {
                let members = [3 + stage * 2, 2 + stage * 2].map(ParallelRankId::new);
                Some(ExpertGroup {
                    source_rank: ParallelRankId::new(stage),
                    members: members.to_vec(),
                    layers: stage as usize..stage as usize + 1,
                    placement: ExpertPlacement::new([
                        (stage as usize, 0, members[0]),
                        (stage as usize, 1, members[1]),
                    ])
                    .unwrap(),
                    limits: ExpertDispatchLimits {
                        max_tokens: 8,
                        max_bytes: 256,
                    },
                })
            })
            .collect::<Vec<_>>();
        let boots = process_boots(&request, &plans, config, &groups).unwrap();
        for (identity, boot) in boots {
            let frame = boot.experts.as_ref().unwrap();
            assert_eq!(frame.source_scope, ExpertSourceScope::ExternalStage);
            assert_eq!(frame.source, boot.rank);
            assert_eq!(frame.source, identity.rank.get());
            assert_eq!(frame.members, [3 + boot.rank * 2, 2 + boot.rank * 2]);
            assert!(!frame.members.contains(&frame.source));
            assert_eq!(frame.timeout_ms, 7250);
            assert_eq!(boot.version, DECODER_WIRE_VERSION);
            assert_eq!(
                serde_json::from_slice::<DecoderBoot>(&serde_json::to_vec(&boot).unwrap()).unwrap(),
                boot
            );
        }
        let first = groups[0].as_mut().unwrap();
        first.source_rank = first.members[0];
        assert!(process_boots(&request, &plans, config, &groups).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn pipeline_process_launch_timeout_and_capacity_are_strict() {
        let mut process = options(2, 2);
        process.rank_backend = PipelineRankBackend::Process;
        assert!(process.validate(ModelExecutionBackend::Cpu).is_err());
        process.process_launch = Some(ProcessLaunch::new(std::env::current_exe().unwrap()));
        process.rank_timeout = Duration::from_millis(7250);
        process.validate(ModelExecutionBackend::Cpu).unwrap();
        assert_eq!(process.process_timeout_ms().unwrap(), 7250);
        let owner = process.process_owner_config();
        assert_eq!(owner.startup_timeout, process.rank_timeout);
        assert_eq!(owner.command_timeout, process.rank_timeout);
        for timeout in [Duration::ZERO, Duration::from_nanos(1), Duration::MAX] {
            let mut invalid = process.clone();
            invalid.rank_timeout = timeout;
            assert!(invalid.validate(ModelExecutionBackend::Cpu).is_err());
        }
        let mut invalid = process.clone();
        invalid.process_launch = Some(ProcessLaunch::new(""));
        assert!(invalid.validate(ModelExecutionBackend::Cpu).is_err());
        invalid = process.clone();
        invalid.rank_restarts = 1;
        assert!(invalid.validate(ModelExecutionBackend::Cpu).is_err());
        invalid = process.clone();
        invalid.parallelism.pipeline_parallel = PROCESS_REAPER_CAPACITY;
        assert!(invalid.validate(ModelExecutionBackend::Cpu).is_err());
        invalid = process;
        invalid.rank_backend = PipelineRankBackend::Thread;
        invalid.rank_timeout = Duration::from_secs(30);
        assert!(invalid.validate(ModelExecutionBackend::Cpu).is_err());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn pipeline_cuda_requires_feature_without_cpu_fallback() {
        let error = options(2, 1)
            .validate(ModelExecutionBackend::Cuda)
            .unwrap_err();
        assert!(error.to_string().contains("'cuda' feature"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn pipeline_cuda_device_layout_has_disjoint_owner_slots_without_device_initialization() {
        let mut options = options(2, 2);
        options.validate(ModelExecutionBackend::Cuda).unwrap();
        assert_eq!(options.device(5), 5);
        for devices in [vec![], vec![0, 1], vec![0, 1, 2, 3, 4, usize::MAX]] {
            options.devices = Some(devices);
            assert!(options.validate(ModelExecutionBackend::Cuda).is_err());
        }
        options.devices = Some(vec![5, 4, 3, 2, 1, 0]);
        options.validate(ModelExecutionBackend::Cuda).unwrap();
        for (owner, device) in [5, 4, 3, 2, 1, 0].into_iter().enumerate() {
            assert_eq!(options.device(owner), device);
        }
        options.devices = Some(vec![0; 6]);
        options.validate(ModelExecutionBackend::Cuda).unwrap();
        #[cfg(unix)]
        {
            options.rank_backend = PipelineRankBackend::Process;
            options.process_launch = Some(ProcessLaunch::new(std::env::current_exe().unwrap()));
            options.validate(ModelExecutionBackend::Cuda).unwrap();
            assert_eq!(
                options.precision(ModelExecutionBackend::Cuda),
                ExecutionPrecisionPolicy::f32()
            );
            options.devices = Some(vec![0, 1]);
            assert!(options.validate(ModelExecutionBackend::Cuda).is_err());
        }
    }
}
