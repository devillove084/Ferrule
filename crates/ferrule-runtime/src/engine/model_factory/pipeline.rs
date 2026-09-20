//! Thread/process pipeline construction through the existing model catalog.

use std::time::Duration;

use ferrule_common::execution::KvElementType;
use ferrule_common::{
    ParallelRankId, ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology,
};
use ferrule_model::decoder::StandardGqaPlanes;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::expert_parallel::{ExpertDispatchLimits, ExpertPlacement};
use ferrule_model::transformer::{Attention, BoundDecoderResources, FeedForward, LayerSegmentPlan};
use ferrule_model::{ModelExecutionBackend, ModelFamily, TokenizerHandle};

use crate::engine::PipelineInferenceEngine;
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
    /// CUDA ordinals: PP owners first, then each stage's EP owners in stage order
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
            .checked_add(1)
            .and_then(|owners| p.pipeline_parallel.checked_mul(owners))
            .filter(|&count| u32::try_from(count).is_ok())
            .ok_or_else(|| invalid("pipeline/expert owner count exceeds the rank ABI"))
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
        if p.data_parallel != 1
            || p.tensor_parallel != 1
            || p.sequence_parallel != 1
            || p.context_parallel != 1
        {
            return Err(invalid(
                "pipeline serving supports PP/EP only; DP/TP/SP/CP must be 1",
            ));
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
                            "--devices requires {count} CUDA ordinals in PP-then-stage-EP order, each within i32"
                        )));
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

    #[cfg(any(feature = "cuda", unix))]
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
    Moe(Qwen3MoeAdapter),
}
impl Checkpoint {
    fn load(request: &ModelBuildRequest) -> ferrule_common::Result<Self> {
        match request.family {
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
            Self::Moe(a) => a.resources(),
        }
    }
    fn schema(&self) -> &StandardGqaPlanes {
        match self {
            Self::Dense(a) => a.kv_schema(),
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

pub(super) fn build(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    let options = request
        .pipeline
        .as_ref()
        .ok_or_else(|| invalid("missing pipeline build options"))?;
    options.validate(request.backend)?;
    let degree = options.parallelism.pipeline_parallel;
    let precision = options.precision(request.backend);
    let checkpoint = Checkpoint::load(&request)?;
    let template = checkpoint.schema();
    if request.driver_config.ctx_size > template.max_sequence_len() {
        return Err(invalid(
            "pipeline context exceeds the checkpoint position range",
        ));
    }
    // Budget every layer at the actual execution KV dtype, not the BF16 storage
    // dtype of the checkpoint. EP owners do not allocate additional KV pages.
    let first = checkpoint
        .resources()
        .spec()
        .layers()
        .first()
        .ok_or_else(|| invalid("pipeline model has no decoder layers"))?;
    let Attention::Gqa(attention) = first.attention() else {
        return Err(invalid("pipeline requires standard GQA attention"));
    };
    let schema = StandardGqaPlanes::new(
        request.max_layers,
        attention.num_kv_heads(),
        attention.head_dim(),
        template.page_size(),
        request.driver_config.ctx_size,
        if precision == ExecutionPrecisionPolicy::f32() {
            KvElementType::F32
        } else {
            KvElementType::Bf16
        },
    )?;
    let accounting = kv_accounting(request.kv_cache_bytes, Some(physical_page_bytes(&schema)?));
    let kv_plan = plan_resident_kv_pages(
        &schema,
        accounting,
        request.scheduler_config,
        request.driver_config,
    )?;
    if kv_plan.configured_pages < kv_plan.full_capacity_pages {
        return Err(invalid(
            "pipeline KV budget cannot cover max-active-sequences full contexts; increase --kv-cache-mb or reduce capacity",
        ));
    }
    let config = PipelineConfig {
        page_size: schema.page_size(),
        max_pages: kv_plan.configured_pages,
        max_positions: request.driver_config.ctx_size,
        max_batch_tokens: request.scheduler_config.max_batch_tokens,
        session_capacity: request.scheduler_config.max_active_sequences,
        max_parameter_bytes: request
            .max_tensor_bytes
            .max(request.expert_reader_max_tensor_bytes),
        precision,
        max_ack_polls: 8,
    };
    config.validate()?;
    let plans = (0..degree)
        .map(|stage| {
            let base = request.max_layers / degree;
            let remainder = request.max_layers % degree;
            let start = stage * base + stage.min(remainder);
            let end = start + base + usize::from(stage < remainder);
            LayerSegmentPlan::new(
                request.max_layers,
                start..end,
                stage == 0,
                stage + 1 == degree,
            )
            .map_err(|error| invalid(error.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
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
        degree as u32,
        ParallelRankId::new(0),
        ParallelismPlan {
            expert_parallel: 1,
            ..options.parallelism
        },
    )
    .map_err(|error| invalid(format!("invalid pipeline topology: {error:?}")))?;
    let tokenizer = TokenizerHandle::load(&request.model_path)?;
    let pipeline = match options.rank_backend {
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
