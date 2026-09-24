//! Thread/process pipeline construction through the existing model catalog.

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

use crate::engine::PipelineInferenceEngine;
use crate::parallel::collective::HostCollectiveLimits;
use crate::parallel::expert::{ExpertGroup, ExpertParallelExecutor, ExpertRankWorker};
use crate::parallel::pipeline::{
    BoxedPipelineStageWorker, PipelineConfig, PipelineParallelExecutor, PipelineRank,
    PipelineStage, PipelineStageBoot, PipelineStageWorker,
};

#[cfg(unix)]
use crate::parallel::process::decoder::{
    DECODER_WIRE_VERSION, DecoderBoot, DecoderDevice, DecoderPrecision, DecoderRecipeKind,
    ExpertPlacementFrame, KvConfigFrame, ProcessPipelineTransport, SegmentFrame,
};
#[cfg(unix)]
use crate::parallel::process::{
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
    /// Process serving uses ProcessFrameLimits::default(); the executable's
    /// trusted child ceilings must accept those limits. Never used in thread mode.
    #[cfg(unix)]
    pub process_launch: Option<ProcessLaunch>,
    /// TP: PP-stage-major then TP-rank order, with PP * TP distinct devices.
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
    fn owner_count(&self) -> Result<usize> {
        let p = self.parallelism;
        let experts = if p.expert_parallel > 1 {
            p.expert_parallel
        } else {
            0
        };
        experts
            .checked_add(p.tensor_parallel)
            .and_then(|owners| p.pipeline_parallel.checked_mul(owners))
            .filter(|&count| u32::try_from(count).is_ok())
            .ok_or_else(|| invalid("pipeline/tensor/expert owner count exceeds the rank ABI"))
    }

    fn validate(&self, backend: ModelExecutionBackend) -> Result<()> {
        self.parallelism
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        if self.rank_restarts != 0 {
            return Err(invalid(
                "pipeline serving does not support rank restart/replay",
            ));
        }
        if self.rank_backend == PipelineRankBackend::Thread
            && self.rank_timeout != Duration::from_secs(30)
        {
            return Err(invalid(
                "rank timeout is a process capability; thread serving is cooperative",
            ));
        }
        let p = self.parallelism;
        if p.data_parallel != 1 || p.sequence_parallel != 1 || p.context_parallel != 1 {
            return Err(invalid("pipeline serving requires DP/SP/CP = 1"));
        }
        if p.tensor_parallel > 1 {
            if backend != ModelExecutionBackend::Cuda {
                return Err(invalid(
                    "dense TP serving requires CUDA; CPU TP is unsupported",
                ));
            }
            if self.rank_backend != PipelineRankBackend::Thread {
                return Err(invalid(
                    "process TP serving is unsupported; use thread ranks",
                ));
            }
            if p.expert_parallel != 1 {
                return Err(invalid("EP x TP serving is unsupported"));
            }
            if !matches!(p.tensor_parallel, 2 | 4) {
                return Err(invalid("supported TP degrees are 1, 2 and 4"));
            }
        }
        let count = self.owner_count()?;
        if self.rank_backend == PipelineRankBackend::Process {
            #[cfg(not(unix))]
            return Err(invalid(
                "pipeline process serving requires Unix process pipes",
            ));
            #[cfg(unix)]
            {
                if self
                    .process_launch
                    .as_ref()
                    .is_none_or(|launch| launch.executable.as_os_str().is_empty())
                {
                    return Err(invalid(
                        "process serving requires an explicit process_launch",
                    ));
                }
                if count > PROCESS_REAPER_CAPACITY {
                    return Err(invalid(
                        "pipeline PP+EP owner count exceeds process capacity",
                    ));
                }
                self.process_timeout_ms()?;
                self.process_owner_config()
                    .validate()
                    .map_err(|e| invalid(e.to_string()))?;
            }
        }
        #[cfg(unix)]
        if self.rank_backend == PipelineRankBackend::Thread && self.process_launch.is_some() {
            return Err(invalid("process_launch requires process ranks"));
        }
        match backend {
            ModelExecutionBackend::Cpu if self.devices.is_some() => {
                return Err(invalid("--devices requires the CUDA pipeline backend"));
            }
            ModelExecutionBackend::Cuda => {
                if !cfg!(feature = "cuda") {
                    return Err(invalid("pipeline CUDA serving requires the 'cuda' feature"));
                }
                if let Some(devices) = &self.devices {
                    if devices.len() != count
                        || devices.iter().any(|&id| i32::try_from(id).is_err())
                    {
                        return Err(invalid(format!(
                            "--devices requires {count} CUDA ordinals in {} order, each within i32",
                            if p.tensor_parallel > 1 {
                                "PP-stage-major then TP-rank"
                            } else {
                                "PP-then-stage-EP"
                            }
                        )));
                    }
                    if p.tensor_parallel > 1
                        && devices
                            .iter()
                            .collect::<std::collections::BTreeSet<_>>()
                            .len()
                            != count
                    {
                        return Err(invalid(
                            "TP requires one distinct CUDA device per physical owner",
                        ));
                    }
                } else if count > i32::MAX as usize {
                    return Err(invalid(
                        "default CUDA placement exceeds the device ordinal ABI",
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Execution precision, not checkpoint storage dtype. CUDA and injected EP
    /// are F32; the original CPU non-EP path retains BF16 compatibility behavior.
    pub fn precision(&self, backend: ModelExecutionBackend) -> ExecutionPrecisionPolicy {
        if backend == ModelExecutionBackend::Cuda || self.parallelism.expert_parallel > 1 {
            ExecutionPrecisionPolicy::f32()
        } else {
            ExecutionPrecisionPolicy::bf16_compatibility()
        }
    }

    #[cfg(unix)]
    fn process_timeout_ms(&self) -> Result<u64> {
        let millis = u64::try_from(self.rank_timeout.as_millis())
            .map_err(|_| invalid("process timeout exceeds millisecond ABI"))?;
        if millis == 0 || Duration::from_millis(millis) != self.rank_timeout {
            return Err(invalid(
                "process timeout must be a nonzero whole number of milliseconds",
            ));
        }
        Ok(millis)
    }

    #[cfg(unix)]
    fn process_owner_config(&self) -> ProcessOwnerConfig {
        ProcessOwnerConfig {
            startup_timeout: self.rank_timeout,
            command_timeout: self.rank_timeout,
            ..Default::default()
        }
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
        let backend =
            Option::<ModelExecutionBackend>::from(backend).unwrap_or(ModelExecutionBackend::Cpu);
        pipeline.validate(backend)?;
        if !matches!(
            config.descriptor().spec.family,
            ModelFamily::Qwen3 | ModelFamily::QwenMoe
        ) {
            return Err(invalid(
                "this model family has no standard pipeline serving binding",
            ));
        }
        let mut plan = self.prepare(
            config,
            BackendSelection::Cpu,
            chat_template_override,
            options,
        )?;
        plan.backend = backend;
        plan.request.backend = backend;
        plan.request.model_info =
            ferrule_model::ModelInfo::from_descriptor(config.descriptor(), backend.as_str());
        plan.with_pipeline(pipeline)
    }
}

impl ResidentModelBuildPlan {
    /// Select the real pipeline engine without replacing the model registry or
    /// changing the owner-thread factory boundary. No model/device is built here.
    pub fn with_pipeline(mut self, options: PipelineBuildOptions) -> Result<Self> {
        options.validate(self.backend)?;
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
        if !matches!(
            self.request.family,
            ModelFamily::Qwen3 | ModelFamily::QwenMoe
        ) {
            return Err(invalid(
                "this model family has no standard pipeline serving binding",
            ));
        }
        if options.parallelism.tensor_parallel > 1 && self.request.family != ModelFamily::Qwen3 {
            return Err(invalid(
                "TP serving requires dense Qwen3; MoE TP is unsupported",
            ));
        }
        if options.parallelism.expert_parallel > 1
            && (self.request.family != ModelFamily::QwenMoe
                || options.parallelism.expert_parallel > self.request.model_info.num_experts)
        {
            return Err(invalid(
                "EP > 1 requires Qwen3-MoE and cannot exceed its expert count",
            ));
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
        if options.parallelism.tensor_parallel > 1 {
            // Reuse strict model metadata validation without weight reads or CUDA.
            let path = self.request.model_path.join("config.json");
            let bytes =
                std::fs::read(&path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
            let value = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
            let spec = ferrule_model::models::qwen3::Qwen3DenseRecipe::new()
                .build_spec(&value)
                .map_err(|e| invalid(e.to_string()))?;
            let plans = segment_plans(
                self.request.max_layers,
                options.parallelism.pipeline_parallel,
            )?;
            validate_tensor_spec(&spec, &plans, &options)?;
            let (config, _) = pipeline_capacity(
                &self.request,
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
    DenseTensor {
        resources: BoundDecoderResources,
        schema: StandardGqaPlanes,
    },
    Moe(Qwen3MoeAdapter),
}
impl Checkpoint {
    fn load(request: &ModelBuildRequest) -> ferrule_common::Result<Self> {
        match request.family {
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
            Self::DenseTensor { resources, .. } => resources,
            Self::Moe(a) => a.resources(),
        }
    }
    fn schema(&self) -> &StandardGqaPlanes {
        match self {
            Self::Dense(a) => a.kv_schema(),
            Self::DenseTensor { schema, .. } => schema,
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
) -> Result<Option<ExpertGroup>> {
    let ep = options.parallelism.expert_parallel;
    if ep == 1 {
        return Ok(None);
    }
    let base = options.parallelism.pipeline_parallel + stage * ep;
    let members = (0..ep)
        .map(|slot| ParallelRankId::new((base + slot) as u32))
        .collect::<Vec<_>>();
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
        source_rank: members[0],
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
    request: ModelBuildRequest,
    boot: PipelineStageBoot,
    group: Option<ExpertGroup>,
) -> ferrule_common::Result<BoxedPipelineStageWorker> {
    let checkpoint = Checkpoint::load(&request)?;
    let resources = checkpoint.resources();
    let config = boot.config;
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

fn segment_plans(layers: usize, degree: usize) -> Result<Vec<LayerSegmentPlan>> {
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
    request: &ModelBuildRequest,
    options: &PipelineBuildOptions,
    spec: &DecoderModelSpec,
    page_size: usize,
    plans: &[LayerSegmentPlan],
) -> Result<(PipelineConfig, crate::engine::ResidentKvPagePlan)> {
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
    let precision = options.precision(request.backend);
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
    let kv_plan = plan_resident_kv_pages(
        &schema,
        kv_accounting(request.kv_cache_bytes, Some(physical_bytes)),
        request.scheduler_config,
        request.driver_config,
    )?;
    if kv_plan.configured_pages < kv_plan.full_capacity_pages {
        return Err(invalid(
            "pipeline KV budget cannot cover max-active-sequences full contexts; increase --kv-cache-mb or reduce capacity",
        ));
    }
    let config = PipelineConfig {
        page_size,
        max_pages: kv_plan.configured_pages,
        max_positions: request.driver_config.ctx_size,
        max_batch_tokens: request.scheduler_config.max_batch_tokens,
        session_capacity: request.scheduler_config.max_active_sequences,
        max_parameter_bytes: if options.parallelism.tensor_parallel > 1 {
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
    Ok((config, kv_plan))
}

pub(super) fn build(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    let options = request
        .pipeline
        .as_ref()
        .ok_or_else(|| invalid("missing pipeline build options"))?;
    options.validate(request.backend)?;
    let degree = options.parallelism.pipeline_parallel;
    let checkpoint = Checkpoint::load(&request)?;
    let spec = checkpoint.resources().spec().clone();
    let plans = segment_plans(request.max_layers, degree)?;
    if options.parallelism.tensor_parallel > 1 {
        for tensor in validate_tensor_spec(&spec, &plans, options)? {
            tensor.validate_resources(checkpoint.resources())?;
        }
    }
    let (config, kv_plan) = pipeline_capacity(
        &request,
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
            )
        })
        .collect::<Result<Vec<_>>>()?;
    drop(checkpoint);
    // EP owners are separate compute identities, never physical KV participants.
    let topology = ValidatedParallelTopology::new(
        ParallelTopologyId::new(1),
        (degree * options.parallelism.tensor_parallel) as u32,
        ParallelRankId::new(0),
        ParallelismPlan {
            expert_parallel: 1,
            ..options.parallelism
        },
    )
    .map_err(|error| invalid(format!("invalid pipeline topology: {error:?}")))?;
    let tokenizer = TokenizerHandle::load(&request.model_path)?;
    let pipeline = match options.rank_backend {
        #[cfg(feature = "cuda")]
        PipelineRankBackend::Thread if options.parallelism.tensor_parallel > 1 => {
            use crate::parallel::pipeline::StandardCudaTensorConfig;
            let tensor_config = StandardCudaTensorConfig {
                devices: (0..options.owner_count()?)
                    .map(|owner| options.device(owner))
                    .collect(),
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
            let boots = process_boots(&request, &plans, config, &groups)?;
            let transport = ProcessPipelineTransport::spawn(
                options
                    .process_launch
                    .clone()
                    .expect("validated process launch"),
                boots,
                options.process_owner_config(),
            )?;
            PipelineParallelExecutor::new_with_transport(topology, plans, config, transport)?
        }
        #[cfg(not(unix))]
        PipelineRankBackend::Process => {
            return Err(invalid(
                "pipeline process serving requires Unix process pipes",
            ));
        }
    };
    Ok(Box::new(PipelineInferenceEngine::new(
        pipeline,
        tokenizer,
        request.model_info,
        request.scheduler_config,
        request.driver_config.stop_at_eos,
        kv_plan,
    )?))
}

/// Preflight bounded JSON before spawning owners. The wire returns full logits
/// for every row; silently accepting an oversized prefill could lose custody
/// when the child fails to encode its reply. These are conservative bounds, not
/// binary tensor byte counts. The transport still validates actual frames.
#[cfg(unix)]
fn validate_process_frames(
    request: &ModelBuildRequest,
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
    request: &ModelBuildRequest,
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
                kv: KvConfigFrame::encode(config),
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
        fn request(&self, pp: usize, tp: usize, limit: u64) -> ModelBuildRequest {
            let config = AutoConfig::from_pretrained(&self.0).unwrap();
            ModelBuildRequest {
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
                moe_hotset_experts: 0,
                kv_cache_bytes: None,
                scheduler_config: crate::ResidentSchedulerConfig {
                    max_active_sequences: 1,
                    max_batch_tokens: 4,
                    prefill_chunk_size: 4,
                    ..Default::default()
                },
                driver_config: crate::ResidentTopKDriverConfig {
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
