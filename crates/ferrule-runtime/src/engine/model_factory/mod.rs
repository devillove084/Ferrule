//! Model-aware catalog selection and resident engine construction.

mod pipeline;
mod qwen35;
pub use pipeline::{PipelineBuildOptions, PipelineRankBackend};
#[cfg(feature = "cuda")]
pub use qwen35::Qwen35MoeCudaAdmission;
pub use qwen35::{
    Qwen35ExpertPhysicalPlan, Qwen35ExpertPlacement, Qwen35ExpertRootMemory,
    Qwen35MoeCapacityLimits, Qwen35MoeCapacityPlan, Qwen35PhysicalCardCapacity,
    Qwen35PhysicalCardMemory,
};

use std::path::PathBuf;

use ferrule_common::{MemoryPoolLimits, ParallelRankId, execution::KvLayoutSchema};
use ferrule_model::{
    AutoConfig, ChatTemplate, ExpertMemoryPolicy, ModelDescriptor, ModelExecutionBackend,
    ModelFamily, WeightSource,
    models::qwen3::{
        Qwen3DenseAdapter, Qwen3DensePrepareOptions, Qwen3MoeAdapter, Qwen3MoePrepareOptions,
    },
};

#[cfg(feature = "cuda")]
use ferrule_model::models::deepseek_v4::{DeepSeekV4Adapter, DeepSeekV4PrepareOptions};

use crate::scheduling::ResidentSchedulerConfig;
use crate::{Error, Result};

use super::{
    BoxedSessionInferenceEngine, ResidentKvPageAccounting, ResidentTopKDriverConfig,
    build_resident_engine, plan_resident_kv_pages,
};

/// Requested model backend. `Auto` is resolved from the detected model family.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BackendSelection {
    #[default]
    Auto,
    Cpu,
    Cuda,
}

impl BackendSelection {
    pub fn parse(value: &str) -> ferrule_common::Result<Self> {
        if value.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        ModelExecutionBackend::parse(value).map(Self::from)
    }
}

impl From<ModelExecutionBackend> for BackendSelection {
    fn from(backend: ModelExecutionBackend) -> Self {
        match backend {
            ModelExecutionBackend::Cpu => Self::Cpu,
            ModelExecutionBackend::Cuda => Self::Cuda,
        }
    }
}

impl From<BackendSelection> for Option<ModelExecutionBackend> {
    fn from(selection: BackendSelection) -> Self {
        match selection {
            BackendSelection::Auto => None,
            BackendSelection::Cpu => Some(ModelExecutionBackend::Cpu),
            BackendSelection::Cuda => Some(ModelExecutionBackend::Cuda),
        }
    }
}

/// A resolved built-in model backend before heavyweight model loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModelBackend {
    family: ModelFamily,
    model_name: &'static str,
    backend: ModelExecutionBackend,
    backend_profile: &'static str,
    default_chat_template: ChatTemplate,
}

impl ResolvedModelBackend {
    pub const fn model_name(&self) -> &'static str {
        self.model_name
    }

    pub const fn backend(&self) -> ModelExecutionBackend {
        self.backend
    }

    pub const fn backend_profile(&self) -> &'static str {
        self.backend_profile
    }

    pub const fn default_chat_template(&self) -> ChatTemplate {
        self.default_chat_template
    }
}

/// One built-in resident model implementation. Adding a family means adding
/// one catalog entry; the resolver and planner scan the catalog generically.
pub struct ModelImplementation {
    /// Model family recognized by this implementation.
    pub family: ModelFamily,
    /// Resolves the runtime backend for one descriptor.
    pub resolve: fn(&ModelDescriptor, BackendSelection) -> Result<ResolvedModelBackend>,
    /// Builds the resident engine from one validated request.
    pub build: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine>,
}

/// Static catalog of built-in model implementations.
pub static MODEL_IMPLEMENTATIONS: &[ModelImplementation] = &[
    ModelImplementation {
        family: ModelFamily::Qwen35Moe,
        resolve: qwen35::resolve,
        build: qwen35::build,
    },
    ModelImplementation {
        family: ModelFamily::Qwen35,
        resolve: qwen35::resolve,
        build: qwen35::build,
    },
    ModelImplementation {
        family: ModelFamily::QwenMoe,
        resolve: resolve_qwen_backend,
        build: build_qwen,
    },
    ModelImplementation {
        family: ModelFamily::DeepSeekV4,
        resolve: resolve_deepseek_backend,
        build: build_deepseek,
    },
    ModelImplementation {
        family: ModelFamily::Qwen3,
        resolve: resolve_dense_qwen3_backend,
        build: build_dense_qwen3,
    },
];

/// Resolver for model families with built-in resident runtime implementations.
#[derive(Debug, Clone, Copy, Default)]
pub struct BuiltinModelResolver;

impl BuiltinModelResolver {
    pub const fn new() -> Self {
        Self
    }

    pub fn resolve(
        &self,
        descriptor: &ModelDescriptor,
        selection: BackendSelection,
    ) -> Result<ResolvedModelBackend> {
        let family = &descriptor.spec.family;
        if let Some(implementation) = MODEL_IMPLEMENTATIONS
            .iter()
            .find(|entry| &entry.family == family)
        {
            return (implementation.resolve)(descriptor, selection);
        }
        match family {
            ModelFamily::Unknown(_) => Err(Error::InvalidRequest {
                message: format!(
                    "no model implementation recognizes family '{}' (architecture {})",
                    descriptor.spec.family,
                    descriptor.spec.architecture.as_deref().unwrap_or("unknown")
                ),
            }),
            _ => Err(unsupported_model(
                descriptor,
                "no resident model implementation is available",
            )),
        }
    }
}

fn resolve_qwen_backend(
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

fn resolve_dense_qwen3_backend(
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
fn resolve_deepseek_backend(
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
fn resolve_deepseek_backend(
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

fn unsupported_model(descriptor: &ModelDescriptor, reason: &'static str) -> Error {
    Error::InvalidRequest {
        message: format!(
            "model family '{}' is not runnable: {reason}",
            descriptor.spec.family
        ),
    }
}

fn unsupported_backend(
    descriptor: &ModelDescriptor,
    backend: ModelExecutionBackend,
    supported: &'static str,
) -> Error {
    Error::InvalidRequest {
        message: format!(
            "model family '{}' does not support backend '{}' (supported: {supported})",
            descriptor.spec.family,
            backend.as_str()
        ),
    }
}

/// Entry and byte limits for model-owned expert caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertCacheOptions {
    pub host_entries: usize,
    pub host_mebibytes: u64,
    pub pinned_entries: usize,
    pub pinned_mebibytes: u64,
}

impl Default for ExpertCacheOptions {
    fn default() -> Self {
        Self {
            host_entries: 256,
            host_mebibytes: 0,
            pinned_entries: 64,
            pinned_mebibytes: 0,
        }
    }
}

/// Compatibility inputs used to load and compose a resident model.
/// Family-specific fields are adapted at the factory boundary.
#[derive(Debug, Clone)]
pub struct ModelFactoryOptions {
    pub max_layers: Option<usize>,
    pub max_tensor_mebibytes: u64,
    pub output_head_chunk_rows: usize,
    pub expert_reader_max_tensor_mebibytes: u64,
    pub expert_cache: ExpertCacheOptions,
    /// CUDA compressed-device expert policy and hard admission caps, Qwen3.5 FP8 only.
    /// None selects the model defaults, not an unbounded cache.
    pub qwen35_moe_capacity: Option<Qwen35MoeCapacityLimits>,
    /// Pageable compressed expert residency; None selects full 40 GiB prewarm for 35B.
    pub qwen35_host_cache: Option<ferrule_model::transformer::host_experts::HostExpertCacheOptions>,
    pub moe_hotset_experts: usize,
    pub kv_cache_mebibytes: Option<u64>,
    pub scheduler_config: ResidentSchedulerConfig,
    pub driver_config: ResidentTopKDriverConfig,
}

/// Family-specific options after the legacy public DTO has been resolved.
///
/// Keeping this behind the family boundary prevents Qwen3.5 capacity, host
/// residency and dedicated placement from becoming fields of every build
/// request. The public [`ModelFactoryOptions`] remains the compatibility DTO.
#[derive(Debug, Clone)]
enum ResolvedFamilyOptions {
    Qwen35(qwen35::Qwen35ResolvedAdapter),
    Generic,
}

impl ResolvedFamilyOptions {
    fn qwen35(&self) -> Option<&qwen35::Qwen35ResolvedAdapter> {
        match self {
            Self::Qwen35(options) => Some(options),
            Self::Generic => None,
        }
    }

    fn qwen35_mut(&mut self) -> Option<&mut qwen35::Qwen35ResolvedAdapter> {
        match self {
            Self::Qwen35(options) => Some(options),
            Self::Generic => None,
        }
    }
}

/// The reason a resolved option differs from the request supplied by a caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionAdjustment {
    pub field: &'static str,
    pub requested: String,
    pub effective: String,
    pub reason: &'static str,
}

/// Auditable changes made while resolving a model request. An empty report is
/// meaningful: the requested and effective factory options are identical.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AdjustmentReport {
    adjustments: Vec<OptionAdjustment>,
}

impl AdjustmentReport {
    pub fn adjustments(&self) -> &[OptionAdjustment] {
        &self.adjustments
    }

    pub fn is_empty(&self) -> bool {
        self.adjustments.is_empty()
    }

    fn push(
        &mut self,
        field: &'static str,
        requested: impl ToString,
        effective: impl ToString,
        reason: &'static str,
    ) {
        self.adjustments.push(OptionAdjustment {
            field,
            requested: requested.to_string(),
            effective: effective.to_string(),
            reason,
        });
    }

    pub fn render(&self) -> String {
        self.adjustments
            .iter()
            .map(|item| {
                format!(
                    "{}: {} -> {} ({})",
                    item.field, item.requested, item.effective, item.reason
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedImplementationKind {
    Resident,
    GenericPipeline,
    DedicatedQwen35ExpertParallel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectedImplementation {
    pub name: &'static str,
    pub kind: SelectedImplementationKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageEncoding {
    /// Encodings are checked when the owner binds the checkpoint.
    ProfileDefined,
    /// Required by the selected profile, not inferred from a dtype alone.
    NumericFp8E4m3fnBf16Scales,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticPrecision {
    ModelDefined,
    F32,
    F32Tf32x3,
    Bf16Compatibility,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildabilityStatus {
    CatalogValidated,
    FeatureUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionBoundary {
    OwnerLiveAdmissionRequired,
}

/// Static support facts are intentionally separate from live owner admission.
/// In particular, `CatalogValidated` does not reserve memory or prove free
/// VRAM, allocator headroom, source identity, or cleanup success.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSupport {
    pub required_feature: Option<&'static str>,
    pub cuda_feature_enabled: bool,
    pub static_constraints: &'static str,
    pub static_supported: bool,
    pub buildability: BuildabilityStatus,
    pub admission: AdmissionBoundary,
    pub storage: StorageEncoding,
    pub arithmetic: ArithmeticPrecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalTopology {
    pub pipeline_parallel: usize,
    pub tensor_parallel: usize,
    pub expert_parallel: usize,
    pub data_parallel: usize,
    pub sequence_parallel: usize,
    pub context_parallel: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalPlacement {
    pub devices: Option<Vec<usize>>,
    pub root_device: Option<usize>,
    pub expert_devices: Option<Vec<usize>>,
    pub source_identity: Option<ParallelRankId>,
}

/// Logical topology and physical placement are separate views. A logical EP
/// owner is not a physical CUDA slot, and metadata placement is not admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelTopologyPlacement {
    pub rank_backend: Option<PipelineRankBackend>,
    pub logical: LogicalTopology,
    pub physical: PhysicalPlacement,
}

fn logical_topology(options: &PipelineBuildOptions) -> LogicalTopology {
    let p = options.parallelism;
    LogicalTopology {
        pipeline_parallel: p.pipeline_parallel,
        tensor_parallel: p.tensor_parallel,
        expert_parallel: p.expert_parallel,
        data_parallel: p.data_parallel,
        sequence_parallel: p.sequence_parallel,
        context_parallel: p.context_parallel,
    }
}

fn resident_topology() -> ModelTopologyPlacement {
    ModelTopologyPlacement {
        rank_backend: None,
        logical: LogicalTopology {
            pipeline_parallel: 1,
            tensor_parallel: 1,
            expert_parallel: 1,
            data_parallel: 1,
            sequence_parallel: 1,
            context_parallel: 1,
        },
        physical: PhysicalPlacement {
            devices: None,
            root_device: None,
            expert_devices: None,
            source_identity: None,
        },
    }
}

/// Planner for preparing a model-aware resident engine build.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResidentModelPlanner;

impl ResidentModelPlanner {
    pub const fn new() -> Self {
        Self
    }

    pub fn prepare(
        &self,
        config: &AutoConfig,
        backend: BackendSelection,
        chat_template_override: Option<&str>,
        options: ModelFactoryOptions,
    ) -> Result<ResidentModelBuildPlan> {
        let descriptor = config.descriptor();
        let entry = BuiltinModelResolver.resolve(descriptor, backend)?;
        let chat_template = match chat_template_override {
            Some(name) => ChatTemplate::from_name(name).ok_or_else(|| Error::InvalidRequest {
                message: format!("unknown chat template '{name}'"),
            })?,
            None => entry.default_chat_template,
        };
        let requested_backend = backend;
        let requested_options = options.clone();
        let model_name = entry.model_name;
        let backend = entry.backend;
        let backend_profile = entry.backend_profile;
        let request = configure_request(descriptor, entry, options)?;
        if request.family == ModelFamily::Qwen35Moe {
            qwen35::capacity_limits(&request)?;
        }

        Ok(ResidentModelBuildPlan {
            model_name,
            backend,
            backend_profile,
            chat_template,
            requested_backend,
            requested_options,
            requested_parallel: None,
            request,
        })
    }
}

/// Sendable resident-model construction inputs.
///
/// The plan may cross into a dedicated owner thread, but [`Self::build`] creates
/// all model, backend, and engine state on the thread that consumes it.
pub struct ResidentModelBuildPlan {
    model_name: &'static str,
    backend: ModelExecutionBackend,
    backend_profile: &'static str,
    chat_template: ChatTemplate,
    requested_backend: BackendSelection,
    requested_options: ModelFactoryOptions,
    requested_parallel: Option<PipelineBuildOptions>,
    request: ResolvedModelRequest,
}

/// Model-neutral metadata for one prepared resident engine build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentModelBuildObservability {
    pub model_name: &'static str,
    pub backend: ModelExecutionBackend,
    pub backend_profile: &'static str,
    pub chat_template: ChatTemplate,
    pub max_layers: usize,
    pub ctx_size: usize,
    pub max_active_sequences: usize,
    pub kv_cache_budget_bytes: Option<u64>,
}

impl ResidentModelBuildPlan {
    pub const fn requested_backend(&self) -> BackendSelection {
        self.requested_backend
    }

    /// Inputs to the compatibility factory adapter, before family/topology
    /// normalization. CLI-only explicit-option validation precedes this boundary.
    pub fn requested_options(&self) -> &ModelFactoryOptions {
        &self.requested_options
    }

    /// Effective factory inputs, not a promise about allocations or loaded
    /// artifacts. Shape-dependent memory and KV L/P budgets remain owner-local.
    pub fn effective_options(&self) -> ModelFactoryOptions {
        let mut options = self.requested_options.clone();
        options.max_layers = Some(self.request.max_layers);
        options.scheduler_config = self.request.scheduler_config;
        options.driver_config = self.request.driver_config;
        if self.request.family == ModelFamily::Qwen35Moe {
            options.qwen35_host_cache = self.qwen35_host_cache_options();
            options.qwen35_moe_capacity = self
                .qwen35_moe_capacity_limits()
                .expect("capacity units validated during prepare");
        }
        options
    }

    pub fn requested_parallel_options(&self) -> Option<&PipelineBuildOptions> {
        self.requested_parallel.as_ref()
    }

    pub fn adjustment_report(&self) -> AdjustmentReport {
        let mut report = AdjustmentReport::default();
        let requested = &self.requested_options;
        let effective = self.effective_options();
        if requested.max_layers != effective.max_layers {
            report.push(
                "max_layers",
                "descriptor default",
                self.request.max_layers,
                "use the descriptor layer count",
            );
        }
        let reason = if self.request.pipeline.is_some() {
            "generic pipeline is serial; batch <= context and prefill <= batch; no mixed batches/cohort deferral/native proposals"
        } else {
            "Qwen3.5 token-serial workspace: batch <= min(32, context); prefill <= effective batch"
        };
        macro_rules! changed {
            ($section:ident, $field:ident) => {
                if requested.$section.$field != effective.$section.$field {
                    report.push(
                        concat!(stringify!($section), ".", stringify!($field)),
                        requested.$section.$field,
                        effective.$section.$field,
                        reason,
                    );
                }
            };
        }
        changed!(scheduler_config, max_batch_tokens);
        changed!(scheduler_config, prefill_chunk_size);
        changed!(scheduler_config, max_decode_batch);
        changed!(scheduler_config, allow_mixed_batches);
        changed!(scheduler_config, decode_cohort_target);
        changed!(scheduler_config, decode_cohort_max_deferrals);
        changed!(driver_config, enable_native_proposals);
        if requested.qwen35_moe_capacity != effective.qwen35_moe_capacity {
            report.push("qwen35_moe_capacity", format!("{:?}", requested.qwen35_moe_capacity),
                format!("{:?}", effective.qwen35_moe_capacity),
                "single-device policy defaults; context, concurrency, effective batch and KV budget override capacity hints; dedicated EP has no local routed cache");
        }
        if requested.qwen35_host_cache != effective.qwen35_host_cache {
            report.push("qwen35_host_cache", format!("{:?}", requested.qwen35_host_cache),
                format!("{:?}", effective.qwen35_host_cache),
                "default full host prewarm policy; source-dependent startup ledger remains owner-local");
        }
        report
    }

    pub fn selected_implementation(&self) -> SelectedImplementation {
        SelectedImplementation {
            name: self.model_name,
            kind: if self.qwen35_expert_placement().is_some() {
                SelectedImplementationKind::DedicatedQwen35ExpertParallel
            } else if self.request.pipeline.is_some() {
                SelectedImplementationKind::GenericPipeline
            } else {
                SelectedImplementationKind::Resident
            },
        }
    }

    /// Only catalog/configuration checks have run. Binding, source identity,
    /// budgets and physical admission must still be checked by the build owner.
    pub fn support(&self) -> ModelSupport {
        let cuda = self.backend == ModelExecutionBackend::Cuda;
        let fp8 = self.request.family == ModelFamily::Qwen35Moe;
        let arithmetic = if fp8 {
            ArithmeticPrecision::F32Tf32x3
        } else if let Some(pipeline) = &self.request.pipeline {
            if pipeline.precision(self.backend)
                == ferrule_model::execution::ExecutionPrecisionPolicy::bf16_compatibility()
            {
                ArithmeticPrecision::Bf16Compatibility
            } else {
                ArithmeticPrecision::F32
            }
        } else if matches!(
            self.request.family,
            ModelFamily::Qwen3 | ModelFamily::QwenMoe
        ) {
            ArithmeticPrecision::Bf16Compatibility
        } else if self.request.family == ModelFamily::Qwen35 {
            ArithmeticPrecision::F32
        } else {
            ArithmeticPrecision::ModelDefined
        };
        ModelSupport {
            required_feature: cuda.then_some("cuda"),
            cuda_feature_enabled: cfg!(feature = "cuda"),
            static_constraints: match self.selected_implementation().kind {
                SelectedImplementationKind::DedicatedQwen35ExpertParallel => {
                    "Qwen35Moe CUDA thread EP2/4/8; explicit distinct devices; resident root shares expert0 card; root-only KV; no TP/PP/process/restart"
                }
                SelectedImplementationKind::GenericPipeline => {
                    "serial pipeline; no mixed batches/cohort deferral/proposals/restart; CUDA thread-only dense TP2/4; no EP x TP"
                }
                SelectedImplementationKind::Resident
                    if matches!(
                        self.request.family,
                        ModelFamily::Qwen35 | ModelFamily::Qwen35Moe
                    ) =>
                {
                    "full-depth Qwen3.5; batch <= min(32, context); no prefix cache/proposals; profile binding and source identity not yet validated"
                }
                SelectedImplementationKind::Resident => {
                    "resident catalog/configuration validated; checkpoint binding not yet validated"
                }
            },
            static_supported: true,
            buildability: if cuda && !cfg!(feature = "cuda") {
                BuildabilityStatus::FeatureUnavailable
            } else {
                BuildabilityStatus::CatalogValidated
            },
            admission: AdmissionBoundary::OwnerLiveAdmissionRequired,
            storage: if fp8 {
                StorageEncoding::NumericFp8E4m3fnBf16Scales
            } else {
                StorageEncoding::ProfileDefined
            },
            arithmetic,
        }
    }

    pub fn topology_placement(&self) -> ModelTopologyPlacement {
        let mut topology = resident_topology();
        if let Some(placement) = self.qwen35_expert_placement() {
            topology.rank_backend = Some(PipelineRankBackend::Thread);
            topology.logical.expert_parallel = placement.expert_parallel();
            topology.physical.root_device = Some(placement.root_device());
            topology.physical.expert_devices = Some(placement.expert_devices().to_vec());
            topology.physical.devices = Some(placement.expert_devices().to_vec());
            topology.physical.source_identity = Some(ParallelRankId::new(0));
        } else if let Some(options) = &self.request.pipeline {
            topology.rank_backend = Some(options.rank_backend);
            topology.logical = logical_topology(options);
            if self.backend == ModelExecutionBackend::Cuda {
                topology.physical.devices = Some(options.devices.clone().unwrap_or_else(|| {
                    (0..options.owner_count().expect("validated pipeline owners")).collect()
                }));
            }
        } else if self.backend == ModelExecutionBackend::Cuda {
            topology.physical.root_device = Some(0);
            topology.physical.devices = Some(vec![0]);
        }
        topology
    }

    /// Human-readable pre-build view shared by CLI and build tracing.
    /// Use the typed accessors rather than parsing this presentation string.
    pub fn resolution_report(&self) -> String {
        format!(
            "{} {:?}: backend {:?} -> {:?}, profile={}; {:?}; topology={:?}; scheduler requested={:?}, effective={:?}; context={}; adjustments=[{}]; metadata only, owner live admission required (source identity, free memory, allocator/headroom and cleanup are not guaranteed); KV logical pages L and physical transaction slots P remain separately owner-planned",
            self.model_name,
            self.selected_implementation().kind,
            self.requested_backend,
            self.backend,
            self.backend_profile,
            self.support(),
            self.topology_placement(),
            self.requested_options.scheduler_config,
            self.request.scheduler_config,
            self.request.driver_config.ctx_size,
            self.adjustment_report().render(),
        )
    }

    pub const fn model_name(&self) -> &'static str {
        self.model_name
    }

    pub const fn backend(&self) -> ModelExecutionBackend {
        self.backend
    }

    pub const fn backend_profile(&self) -> &'static str {
        self.backend_profile
    }

    pub const fn chat_template(&self) -> ChatTemplate {
        self.chat_template
    }

    pub const fn observability(&self) -> ResidentModelBuildObservability {
        ResidentModelBuildObservability {
            model_name: self.model_name,
            backend: self.backend,
            backend_profile: self.backend_profile,
            chat_template: self.chat_template,
            max_layers: self.request.max_layers,
            ctx_size: self.request.driver_config.ctx_size,
            max_active_sequences: self.request.scheduler_config.max_active_sequences,
            kv_cache_budget_bytes: self.request.kv_cache_bytes,
        }
    }

    /// Effective FP8 device limits, including this plan's context/concurrency/KV caps.
    /// This does not initialize CUDA or reserve memory.
    pub fn qwen35_moe_capacity_limits(&self) -> Result<Option<Qwen35MoeCapacityLimits>> {
        if self.request.family == ModelFamily::Qwen35Moe && self.qwen35_expert_placement().is_none()
        {
            qwen35::capacity_limits(&self.request).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Explicit EP owners, independent of the single root's KV topology.
    pub fn qwen35_expert_placement(&self) -> Option<&Qwen35ExpertPlacement> {
        self.request
            .family_options
            .qwen35()
            .and_then(|options| options.expert_placement())
    }

    /// Effective host policy, independent of device-cache flags. Metadata only.
    pub fn qwen35_host_cache_options(
        &self,
    ) -> Option<ferrule_model::transformer::host_experts::HostExpertCacheOptions> {
        self.request
            .family_options
            .qwen35()
            .and_then(|options| options.moe())
            .map(|options| options.host_cache)
    }

    /// Load model state and build the resident engine on the calling thread.
    pub fn build(self) -> Result<BoxedSessionInferenceEngine> {
        tracing::info!(report = %self.resolution_report(), "resident model build entering owner-local admission");
        self.build_with(|request| {
            let family = &request.resolved.family;
            let implementation = MODEL_IMPLEMENTATIONS
                .iter()
                .find(|entry| &entry.family == family)
                .ok_or_else(|| Error::InvalidRequest {
                    message: format!(
                        "no resident model implementation can build family '{family}'"
                    ),
                })?;
            (implementation.build)(request)
        })
    }

    // A narrow injection seam for owner-admission tests, not a public alternate
    // factory. Production always dispatches through MODEL_IMPLEMENTATIONS.
    fn build_with<T>(self, build: impl FnOnce(ModelBuildRequest) -> Result<T>) -> Result<T> {
        build(ModelBuildRequest {
            resolved: self.request,
        })
    }
}

/// Compatibility DTO for the public catalog builder signature.
/// Its private resolved inputs are consumed by the existing build owner.
#[derive(Debug, Clone)]
pub struct ModelBuildRequest {
    resolved: ResolvedModelRequest,
}

/// Internal construction inputs; Qwen3.5 policies stay behind the family adapter.
#[derive(Debug, Clone)]
struct ResolvedModelRequest {
    family: ModelFamily,
    backend: ModelExecutionBackend,
    model_path: PathBuf,
    model_info: ferrule_model::ModelInfo,
    pipeline: Option<PipelineBuildOptions>,
    family_options: ResolvedFamilyOptions,
    max_layers: usize,
    max_tensor_bytes: u64,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    output_head_chunk_rows: usize,
    expert_reader_max_tensor_bytes: u64,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    expert_memory_policy: ExpertMemoryPolicy,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    moe_hotset_experts: usize,
    kv_cache_bytes: Option<u64>,
    scheduler_config: ResidentSchedulerConfig,
    driver_config: ResidentTopKDriverConfig,
}

fn configure_request(
    descriptor: &ModelDescriptor,
    entry: ResolvedModelBackend,
    mut options: ModelFactoryOptions,
) -> Result<ResolvedModelRequest> {
    let family_options = qwen35::resolve_family_options(&entry.family, &options)?;
    if matches!(entry.family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe) {
        qwen35::validate_options(descriptor, &mut options)?;
    }
    let max_layers = options
        .max_layers
        .or(descriptor.spec.num_layers)
        .ok_or_else(|| Error::InvalidRequest {
            message: format!(
                "{} descriptor does not declare a layer count",
                entry.model_name
            ),
        })?;
    if max_layers == 0 {
        return Err(Error::InvalidRequest {
            message: format!("{} max_layers must be greater than zero", entry.model_name),
        });
    }
    if descriptor
        .spec
        .num_layers
        .is_some_and(|model_layers| max_layers > model_layers)
    {
        return Err(Error::InvalidRequest {
            message: format!(
                "{} max_layers {max_layers} exceeds descriptor layer count {}",
                entry.model_name,
                descriptor.spec.num_layers.unwrap_or_default()
            ),
        });
    }
    let max_tensor_bytes = required_mebibytes_to_bytes(
        options.max_tensor_mebibytes,
        entry.model_name,
        "max tensor read limit",
    )?;
    let expert_reader_max_tensor_bytes = required_mebibytes_to_bytes(
        options.expert_reader_max_tensor_mebibytes,
        entry.model_name,
        "expert tensor read limit",
    )?;
    let kv_cache_bytes = options
        .kv_cache_mebibytes
        .map(|mebibytes| {
            required_mebibytes_to_bytes(mebibytes, entry.model_name, "KV cache budget")
        })
        .transpose()?;
    let expert_memory_policy = ExpertMemoryPolicy::new(
        MemoryPoolLimits::new(
            options.expert_cache.host_entries,
            optional_mebibytes_to_bytes(
                options.expert_cache.host_mebibytes,
                entry.model_name,
                "host expert cache budget",
            )?,
        ),
        MemoryPoolLimits::new(
            options.expert_cache.pinned_entries,
            optional_mebibytes_to_bytes(
                options.expert_cache.pinned_mebibytes,
                entry.model_name,
                "pinned expert cache budget",
            )?,
        ),
    );

    Ok(ResolvedModelRequest {
        family: entry.family,
        backend: entry.backend,
        model_path: descriptor.path.clone(),
        model_info: ferrule_model::ModelInfo::from_descriptor(descriptor, entry.backend.as_str()),
        pipeline: None,
        family_options,
        max_layers,
        max_tensor_bytes,
        output_head_chunk_rows: options.output_head_chunk_rows,
        expert_reader_max_tensor_bytes,
        expert_memory_policy,
        moe_hotset_experts: options.moe_hotset_experts,
        kv_cache_bytes,
        scheduler_config: options.scheduler_config,
        driver_config: options.driver_config,
    })
}

fn optional_mebibytes_to_bytes(
    mebibytes: u64,
    model_name: &'static str,
    resource: &'static str,
) -> Result<u64> {
    if mebibytes == 0 {
        return Ok(u64::MAX);
    }
    checked_mebibytes_to_bytes(mebibytes, model_name, resource)
}

fn required_mebibytes_to_bytes(
    mebibytes: u64,
    model_name: &'static str,
    resource: &'static str,
) -> Result<u64> {
    if mebibytes == 0 {
        return Err(Error::InvalidRequest {
            message: format!("{model_name} {resource} must be greater than zero"),
        });
    }
    checked_mebibytes_to_bytes(mebibytes, model_name, resource)
}

fn checked_mebibytes_to_bytes(
    mebibytes: u64,
    model_name: &'static str,
    resource: &'static str,
) -> Result<u64> {
    mebibytes
        .checked_mul(1024 * 1024)
        .ok_or_else(|| Error::InvalidRequest {
            message: format!("{model_name} {resource} exceeds the supported byte range"),
        })
}

fn qwen_prepare_options(request: &ResolvedModelRequest) -> Qwen3MoePrepareOptions {
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

fn physical_page_bytes(schema: &dyn KvLayoutSchema) -> Result<u64> {
    let bytes = schema
        .checked_page_bytes()
        .ok_or_else(|| Error::InvalidRequest {
            message: "physical KV page size overflow".into(),
        })?;
    u64::try_from(bytes).map_err(|_| Error::InvalidRequest {
        message: "physical KV page size exceeds u64".into(),
    })
}

fn build_qwen(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    let request = request.resolved;
    if request.pipeline.is_some() {
        return pipeline::build(request);
    }
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

fn build_dense_qwen3(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    let request = request.resolved;
    if request.pipeline.is_some() {
        return pipeline::build(request);
    }
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
fn deepseek_prepare_options(request: &ResolvedModelRequest) -> Result<DeepSeekV4PrepareOptions> {
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
fn build_deepseek(_request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    Err(Error::InvalidRequest {
        message: "DeepSeek-V4 requires a CUDA-enabled runtime build".into(),
    })
}

#[cfg(feature = "cuda")]
fn build_deepseek(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
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

fn kv_accounting(budget_bytes: Option<u64>, page_bytes: Option<u64>) -> ResidentKvPageAccounting {
    match (budget_bytes, page_bytes) {
        (Some(budget_bytes), Some(page_bytes)) => ResidentKvPageAccounting::PhysicalBytes {
            budget_bytes,
            page_bytes,
        },
        _ => ResidentKvPageAccounting::ContextCapacity,
    }
}

fn log_kv_plan(
    label: &str,
    schema: &dyn KvLayoutSchema,
    accounting: ResidentKvPageAccounting,
    request: &ResolvedModelRequest,
) -> Result<()> {
    let plan = plan_resident_kv_pages(
        schema,
        accounting,
        request.scheduler_config,
        request.driver_config,
    )?;
    if let (Some(budget_bytes), Some(page_bytes), Some(configured_bytes)) = (
        request.kv_cache_bytes,
        plan.page_bytes,
        plan.configured_bytes,
    ) {
        tracing::info!(
            model = label,
            configured_pages = plan.configured_pages,
            full_capacity_pages = plan.full_capacity_pages,
            page_bytes,
            physical_budget_mib = budget_bytes / (1024 * 1024),
            allocated_mib = configured_bytes / (1024 * 1024),
            "configured resident KV pool"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ferrule_model::{
        AttentionKind, MoeSpec, RouterKind, TransformerSemantics, TransformerSpec,
    };

    use super::*;

    pub(super) fn descriptor(
        family: ModelFamily,
        architecture: &str,
        layers: usize,
    ) -> ModelDescriptor {
        ModelDescriptor {
            path: PathBuf::from("model"),
            spec: TransformerSpec {
                family,
                architecture: Some(architecture.into()),
                weight_source: WeightSource::Safetensors,
                hidden_size: Some(64),
                num_layers: Some(layers),
                vocab_size: Some(128),
                num_heads: Some(4),
                num_kv_heads: Some(4),
                head_dim: Some(16),
                attention: AttentionKind::MultiLatentAttention,
                moe: MoeSpec::none(),
                semantics: TransformerSemantics::default(),
                tensor_count: None,
                quantization: Vec::new(),
                notes: Vec::new(),
            },
            tensor_classes: Vec::new(),
        }
    }

    fn qwen_descriptor() -> ModelDescriptor {
        let mut descriptor = descriptor(ModelFamily::QwenMoe, "qwen3_moe", 48);
        descriptor.spec.attention = AttentionKind::GroupedQuery;
        descriptor.spec.num_kv_heads = Some(1);
        descriptor.spec.moe = MoeSpec {
            num_experts: Some(128),
            num_experts_per_tok: Some(8),
            has_shared_experts: false,
            router: RouterKind::DenseTopK,
        };
        descriptor
    }

    pub(super) fn options() -> ModelFactoryOptions {
        ModelFactoryOptions {
            max_layers: None,
            max_tensor_mebibytes: 128,
            output_head_chunk_rows: 4096,
            expert_reader_max_tensor_mebibytes: 64,
            expert_cache: ExpertCacheOptions::default(),
            qwen35_moe_capacity: None,
            qwen35_host_cache: None,
            moe_hotset_experts: 0,
            kv_cache_mebibytes: None,
            scheduler_config: ResidentSchedulerConfig::default(),
            driver_config: ResidentTopKDriverConfig::default(),
        }
    }

    #[test]
    fn owner_build_consumes_effective_request_on_the_consuming_thread() {
        let config = AutoConfig::from_descriptor(descriptor(ModelFamily::Qwen35, "qwen35", 24));
        let mut input = options();
        input.driver_config.enable_native_proposals = false;
        input.driver_config.ctx_size = 8;
        input.scheduler_config.max_batch_tokens = 512;
        input.scheduler_config.prefill_chunk_size = 512;
        let plan = ResidentModelPlanner
            .prepare(&config, BackendSelection::Cpu, None, input)
            .unwrap();
        let effective = plan.effective_options();
        let report = plan.resolution_report();
        let caller = std::thread::current().id();
        std::thread::spawn(move || {
            plan.build_with(|request| {
                let request = request.resolved;
                assert_ne!(std::thread::current().id(), caller);
                assert_eq!(request.scheduler_config, effective.scheduler_config);
                assert_eq!(request.driver_config, effective.driver_config);
                assert_eq!(request.max_layers, effective.max_layers.unwrap());
                assert!(report.contains("512 -> 8"));
                Ok(())
            })
            .unwrap();
        })
        .join()
        .unwrap();
    }

    #[test]
    fn legacy_options_adapter_and_resolved_plan_keep_the_same_build_request() {
        for family in [
            ModelFamily::Qwen3,
            ModelFamily::QwenMoe,
            ModelFamily::Qwen35,
        ] {
            let config = AutoConfig::from_descriptor(descriptor(family, "compatibility", 24));
            for context in [8, 32, 1024] {
                for batch in [8, 32, 512] {
                    let mut input = options();
                    input.driver_config.enable_native_proposals = false;
                    input.driver_config.ctx_size = context;
                    input.scheduler_config.max_batch_tokens = batch;
                    let legacy = configure_request(
                        config.descriptor(),
                        BuiltinModelResolver
                            .resolve(config.descriptor(), BackendSelection::Auto)
                            .unwrap(),
                        input.clone(),
                    )
                    .unwrap();
                    let plan = ResidentModelPlanner
                        .prepare(&config, BackendSelection::Auto, None, input)
                        .unwrap();
                    assert_eq!(
                        plan.effective_options().scheduler_config,
                        legacy.scheduler_config
                    );
                    assert_eq!(plan.effective_options().driver_config, legacy.driver_config);
                    plan.build_with(|request| {
                        let request = request.resolved;
                        assert_eq!(request.scheduler_config, legacy.scheduler_config);
                        assert_eq!(request.driver_config, legacy.driver_config);
                        assert_eq!(request.max_layers, legacy.max_layers);
                        assert_eq!(request.max_tensor_bytes, legacy.max_tensor_bytes);
                        assert_eq!(request.expert_memory_policy, legacy.expert_memory_policy);
                        assert_eq!(request.kv_cache_bytes, legacy.kv_cache_bytes);
                        Ok(())
                    })
                    .unwrap();
                }
            }
        }
    }

    #[test]
    fn metadata_success_does_not_suppress_owner_build_failure() {
        let config = AutoConfig::from_descriptor(qwen_descriptor());
        let plan = ResidentModelPlanner
            .prepare(&config, BackendSelection::Auto, None, options())
            .unwrap();
        assert_eq!(
            plan.support().buildability,
            BuildabilityStatus::CatalogValidated
        );
        let error = plan
            .build_with(|_| -> Result<()> { Err(Error::RequestCapacity { limit: 0 }) })
            .unwrap_err();
        assert!(matches!(error, Error::RequestCapacity { limit: 0 }));
    }

    #[test]
    fn planner_prepares_sendable_qwen_plan_without_loading_payloads() {
        fn assert_send<T: Send>() {}
        assert_send::<ResidentModelBuildPlan>();

        let config = AutoConfig::from_descriptor(qwen_descriptor());
        let prepared = ResidentModelPlanner::new()
            .prepare(&config, BackendSelection::Auto, None, options())
            .unwrap();
        assert_eq!(prepared.model_name(), "qwen3-moe");
        assert_eq!(prepared.backend(), ModelExecutionBackend::Cpu);
        assert_eq!(prepared.request.max_layers, 48);
        assert_eq!(
            prepared.observability(),
            ResidentModelBuildObservability {
                model_name: "qwen3-moe",
                backend: ModelExecutionBackend::Cpu,
                backend_profile: "cpu-standard-decoder",
                chat_template: ChatTemplate::Qwen3,
                max_layers: 48,
                ctx_size: ResidentTopKDriverConfig::default().ctx_size,
                max_active_sequences: ResidentSchedulerConfig::default().max_active_sequences,
                kv_cache_budget_bytes: None,
            }
        );
    }

    #[test]
    fn qwen_rejects_cuda_and_unsupported_artifacts() {
        let resolver = BuiltinModelResolver::new();
        let error = resolver
            .resolve(&qwen_descriptor(), BackendSelection::Cuda)
            .unwrap_err();
        assert!(error.to_string().contains("supported: cpu"));

        let mut gguf = qwen_descriptor();
        gguf.spec.weight_source = WeightSource::Gguf;
        assert!(resolver.resolve(&gguf, BackendSelection::Auto).is_err());

        let dense = descriptor(ModelFamily::Qwen3, "qwen3", 28);
        for selection in [BackendSelection::Auto, BackendSelection::Cpu] {
            let resolved = resolver.resolve(&dense, selection).unwrap();
            assert_eq!(resolved.model_name(), "qwen3-dense");
            assert_eq!(resolved.backend(), ModelExecutionBackend::Cpu);
            assert_eq!(resolved.backend_profile(), "cpu-standard-decoder");
        }
        assert!(
            resolver
                .resolve(&dense, BackendSelection::Cuda)
                .unwrap_err()
                .to_string()
                .contains("supported: cpu")
        );
        let mut dense_gguf = dense;
        dense_gguf.spec.weight_source = WeightSource::Gguf;
        assert!(
            resolver
                .resolve(&dense_gguf, BackendSelection::Auto)
                .is_err()
        );
    }

    #[test]
    fn reports_unknown_models_and_chat_templates() {
        let unknown = descriptor(ModelFamily::Unknown("mystery".into()), "mystery", 1);
        assert!(
            BuiltinModelResolver::new()
                .resolve(&unknown, BackendSelection::Auto)
                .unwrap_err()
                .to_string()
                .contains("no model implementation")
        );
        let config = AutoConfig::from_descriptor(qwen_descriptor());
        assert!(
            ResidentModelPlanner::new()
                .prepare(
                    &config,
                    BackendSelection::Auto,
                    Some("not-a-template"),
                    options()
                )
                .is_err()
        );
    }

    #[test]
    fn backend_selection_parses_auto_and_explicit_backends() {
        assert_eq!(
            BackendSelection::parse("auto").unwrap(),
            BackendSelection::Auto
        );
        assert_eq!(
            BackendSelection::parse("cpu").unwrap(),
            BackendSelection::Cpu
        );
        assert_eq!(
            BackendSelection::parse("cuda").unwrap(),
            BackendSelection::Cuda
        );
        assert!(BackendSelection::parse("optimized-cuda").is_err());
    }

    #[test]
    fn qwen35_plans_owner_local_full_depth_and_rejects_unsupported_policies() {
        let config = AutoConfig::from_descriptor(descriptor(
            ModelFamily::Qwen35,
            "Qwen3_5ForConditionalGeneration",
            24,
        ));
        let options = || {
            let mut o = self::options();
            o.driver_config.enable_native_proposals = false;
            o
        };
        for backend in [
            BackendSelection::Auto,
            BackendSelection::Cpu,
            BackendSelection::Cuda,
        ] {
            let plan = ResidentModelPlanner.prepare(&config, backend, None, options());
            if backend == BackendSelection::Cuda && !cfg!(feature = "cuda") {
                assert!(plan.err().unwrap().to_string().contains("cuda"));
                continue;
            }
            let plan = plan.unwrap();
            assert_eq!(plan.chat_template(), ChatTemplate::Qwen35);
            assert_eq!(plan.request.max_layers, 24);
            assert!(!plan.request.driver_config.enable_native_proposals);
            assert_eq!(plan.request.scheduler_config.max_batch_tokens, 32);
        }
        for case in 0..6 {
            let mut o = options();
            match case {
                0 => o.max_layers = Some(1),
                1 => o.scheduler_config.prefix_cache_capacity_pages = 1,
                2 => o.expert_cache.host_entries = 1,
                3 => o.moe_hotset_experts = 1,
                4 => o.driver_config.enable_native_proposals = true,
                _ => o.scheduler_config.max_batch_tokens = 0,
            }
            assert!(
                ResidentModelPlanner
                    .prepare(&config, BackendSelection::Cpu, None, o)
                    .is_err()
            );
        }
        assert!(
            ResidentModelPlanner
                .prepare_pipeline(
                    &config,
                    BackendSelection::Cpu,
                    None,
                    options(),
                    PipelineBuildOptions::default()
                )
                .is_err()
        );
        let plan = ResidentModelPlanner
            .prepare(&config, BackendSelection::Cpu, None, options())
            .unwrap();
        assert!(plan.with_pipeline(PipelineBuildOptions::default()).is_err());
    }

    #[test]
    fn planner_checks_capacity_unit_conversion() {
        let config = AutoConfig::from_descriptor(qwen_descriptor());
        let mut invalid = options();
        invalid.kv_cache_mebibytes = Some(0);
        assert!(
            ResidentModelPlanner::new()
                .prepare(&config, BackendSelection::Auto, None, invalid)
                .is_err()
        );

        let mut overflowing = options();
        overflowing.max_tensor_mebibytes = u64::MAX;
        assert!(
            ResidentModelPlanner::new()
                .prepare(&config, BackendSelection::Auto, None, overflowing)
                .is_err()
        );
    }
}
