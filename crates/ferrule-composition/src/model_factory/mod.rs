//! Model-aware catalog selection and resident engine construction.

mod bf16_serving;
mod mixed_serving;
pub use mixed_serving::tensor::{MIXED_TP2_DEFAULT_WORKSPACE_BYTES, MixedTp2CacheRequirements};
pub use mixed_serving::{ServingModelProfile, ServingOptionProvenance};
mod dense_f32_kv_policy;
#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/full40_strict/mod.rs"]
mod full40_strict;
pub use bf16_serving::{ModelPrecision, NumericStoragePolicy, Qwen35NumericBf16MemoryPlan};
mod host_memory;
mod numeric_profile;
mod parallel;
mod pipeline;
mod standard_adapters;
use standard_adapters::*;
mod execution_policy;
mod expert_read_policy;
mod selection;
use selection::*;

#[path = "adapters/qwen35/mod.rs"]
mod qwen35;
#[path = "adapters/qwen35/composition.rs"]
pub mod qwen35_composition;
mod resident_f32_bound;
pub use numeric_profile::NumericFp8Profile;
pub use pipeline::{PipelineBuildOptions, PipelineRankBackend};
pub use qwen35::Qwen35MoeMemoryPlan;
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

use crate::{Error, Result};
use ferrule_runtime::scheduling::ResidentSchedulerConfig;

use ferrule_runtime::engine::{
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

/// One repository-owned model composition entry. Adding a family means adding
/// one catalog entry; the resolver and planner scan the catalog generically.
///
/// # Catalog access and migration
///
/// Obtain entries from [`MODEL_IMPLEMENTATIONS`]. The public `family`, `resolve`,
/// and `build` fields remain available for inspection and callback use; the
/// `build` callback retains its `fn(ModelBuildRequest)` signature.
///
/// External struct literals are no longer supported: the private selector must
/// pair construction with the matching metadata/option policy. Migrate callers
/// inspecting or using built-ins to a borrowed catalog entry. This is not a
/// plugin registration API: there is no public registration or prepare-from-entry
/// operation, and [`ModelBuildRequest`] and [`ResolvedModelBackend`] contain
/// private validated inputs. A three-callback constructor alone would therefore
/// create an entry that the planner cannot use, not a usable custom implementation.
/// Custom family composition belongs in this repository's single catalog.
///
/// ```compile_fail
/// use ferrule_composition::model_factory::ModelImplementation;
/// fn copy_entry(entry: &ModelImplementation) -> ModelImplementation {
///     ModelImplementation {
///         family: entry.family.clone(),
///         resolve: entry.resolve,
///         build: entry.build,
///     }
/// }
/// ```
pub struct ModelImplementation {
    /// Model family recognized by this implementation.
    pub family: ModelFamily,
    /// Resolves the runtime backend for one descriptor.
    pub resolve: fn(&ModelDescriptor, BackendSelection) -> Result<ResolvedModelBackend>,
    /// Builds the resident engine from one validated request.
    pub build: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine>,
    // One catalog-owned lowering selects both construction and option policy.
    select: StrategySelector,
}

/// Static catalog of built-in model implementations.
pub static MODEL_IMPLEMENTATIONS: &[ModelImplementation] = &[
    ModelImplementation {
        family: ModelFamily::Qwen35Moe,
        resolve: qwen35::resolve,
        build: qwen35::build,
        select: qwen35_composition::select_strategy,
    },
    ModelImplementation {
        family: ModelFamily::Qwen35,
        resolve: qwen35::resolve,
        build: qwen35::build,
        select: qwen35_composition::select_strategy,
    },
    ModelImplementation {
        family: ModelFamily::QwenMoe,
        resolve: resolve_qwen_backend,
        build: build_qwen,
        select: pipeline::select_moe_strategy,
    },
    ModelImplementation {
        family: ModelFamily::DeepSeekV4,
        resolve: resolve_deepseek_backend,
        build: build_deepseek,
        select: select_resident_strategy,
    },
    ModelImplementation {
        family: ModelFamily::Qwen3,
        resolve: resolve_dense_qwen3_backend,
        build: build_dense_qwen3,
        select: pipeline::select_dense_strategy,
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
    /// Explicit numeric arithmetic for Qwen3.5 MoE only; omitted preserves F32.
    pub numeric_fp8_profile: Option<NumericFp8Profile>,
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

/// Serving engine only; model lowering and expert ownership are separate facts.
/// Both standard and hybrid stage programs use `Pipeline`. A resident decoder
/// with external expert owners remains `Resident`; inspect topology for EP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedImplementationKind {
    Resident,
    Pipeline,
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
    /// Native unscaled BF16 checkpoint and device parameter storage.
    NativeBf16,
    /// Required by the selected profile, not inferred from a dtype alone.
    NumericFp8E4m3fnBf16Scales,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticPrecision {
    ModelDefined,
    F32,
    F32Tf32x3,
    W8A16Bf16F32Accumulate,
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

/// Serving intent, not a family-specific execution implementation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ServingEngineSelection {
    #[default]
    Auto,
    Resident,
    Pipeline,
}

/// Explicit serving inputs. `Some(default)` is still an explicit policy and
/// must not disappear when an implementation does not support that option.
/// The legacy `ModelFactoryOptions` APIs keep their existing default semantics.
#[derive(Debug, Clone, Default)]
pub struct ServingModelOptions {
    /// Explicit bounded host staging for lazy unsharded numeric experts.
    /// None preserves synchronous demand reads; not a device or host cache budget.
    pub expert_read_pipeline: Option<ferrule_model::transformer::ExpertReadPipelineOptions>,
    /// Explicit model/storage/arithmetic bundle; omission preserves legacy dispatch.
    pub model_profile: Option<ServingModelProfile>,
    /// Omitted preserves the existing family/backend defaults.
    pub model_precision: Option<ModelPrecision>,
    pub engine: ServingEngineSelection,
    pub kv_execution_policy: ferrule_runtime::engine::KvExecutionPolicy,
    /// None means no parallel/rank options were requested.
    pub parallel: Option<PipelineBuildOptions>,
    pub max_tensor_mebibytes: Option<u64>,
    pub expert_host_cache_entries: Option<usize>,
    pub expert_host_cache_mebibytes: Option<u64>,
    pub expert_pinned_cache_entries: Option<usize>,
    pub expert_pinned_cache_mebibytes: Option<u64>,
    pub moe_hotset_experts: Option<usize>,
}

impl ServingModelOptions {
    fn resolve_options(
        &self,
        construction: ResolvedConstruction,
        mut options: ModelFactoryOptions,
    ) -> Result<ModelFactoryOptions> {
        selection::validate_serving_engine(self, construction)?;
        for (flag, present) in [
            (
                "--expert-host-cache-entries",
                self.expert_host_cache_entries.is_some(),
            ),
            (
                "--expert-host-cache-mb",
                self.expert_host_cache_mebibytes.is_some(),
            ),
            (
                "--expert-pinned-cache-entries",
                self.expert_pinned_cache_entries.is_some(),
            ),
            (
                "--expert-pinned-cache-mb",
                self.expert_pinned_cache_mebibytes.is_some(),
            ),
        ] {
            let policy = if construction.kind() == SelectedImplementationKind::Pipeline {
                execution_policy::OverridePolicy::Unsupported
            } else {
                execution_policy::OverridePolicy::Consumed
            };
            execution_policy::validate_override(flag, present.then_some(&()), &(), policy)
                .map_err(|_| {
                    invalid(format!(
                        "pipeline serving does not implement expert_cache overrides ({flag})"
                    ))
                })?;
        }
        if construction.kind() == SelectedImplementationKind::Pipeline {
            options.expert_cache = ExpertCacheOptions::default();
        } else {
            if let Some(n) = self.expert_host_cache_entries {
                options.expert_cache.host_entries = n;
            }
            if let Some(n) = self.expert_host_cache_mebibytes {
                options.expert_cache.host_mebibytes = n;
            }
            if let Some(n) = self.expert_pinned_cache_entries {
                options.expert_cache.pinned_entries = n;
            }
            if let Some(n) = self.expert_pinned_cache_mebibytes {
                options.expert_cache.pinned_mebibytes = n;
            }
        }
        selection::apply_common_serving_options(self, &mut options);
        Ok(options)
    }
}

/// Frozen metadata selection, not a device admission or source-binding proof.
#[derive(Debug, Clone, Copy)]
struct ResolvedExecution {
    expert_read_pipeline: Option<ferrule_model::transformer::ExpertReadPipelineOptions>,
    // Immutable, model-BF16-only choice; not the legacy W8/F32 expert wire profile.
    numeric_bf16_precision: Option<ferrule_model::transformer::NumericFp8Precision>,
    numeric_profile: Option<NumericFp8Profile>,
    option_policy: &'static ModelOptionPolicy,
    topology_policy: Option<execution_policy::TopologyPolicy>,
    conversion: PipelineConversion,
    adjustment_reason: &'static str,
    kv_execution_policy: ferrule_runtime::engine::KvExecutionPolicy,
    construction: ResolvedConstruction,
    support: ModelSupport,
}

impl ResolvedExecution {
    fn new(
        backend: ModelExecutionBackend,
        construction: ResolvedConstruction,
        storage: StorageEncoding,
        arithmetic: ArithmeticPrecision,
        static_constraints: &'static str,
    ) -> Self {
        let cuda = backend == ModelExecutionBackend::Cuda;
        Self {
            expert_read_pipeline: None,
            numeric_bf16_precision: None,
            numeric_profile: None,
            option_policy: &RESIDENT_OPTIONS,
            topology_policy: None,
            conversion: PipelineConversion::Allowed,
            adjustment_reason: "selected execution policy",
            kv_execution_policy: ferrule_runtime::engine::KvExecutionPolicy::Resident,
            construction,
            support: ModelSupport {
                required_feature: cuda.then_some("cuda"),
                cuda_feature_enabled: cfg!(feature = "cuda"),
                static_constraints,
                static_supported: true,
                buildability: if cuda && !cfg!(feature = "cuda") {
                    BuildabilityStatus::FeatureUnavailable
                } else {
                    BuildabilityStatus::CatalogValidated
                },
                admission: AdmissionBoundary::OwnerLiveAdmissionRequired,
                storage,
                arithmetic,
            },
        }
    }

    fn pipeline_precision(self) -> ferrule_model::execution::ExecutionPrecisionPolicy {
        use ferrule_model::execution::ExecutionPrecisionPolicy;
        match self.support.arithmetic {
            ArithmeticPrecision::F32
            | ArithmeticPrecision::F32Tf32x3
            | ArithmeticPrecision::W8A16Bf16F32Accumulate => ExecutionPrecisionPolicy::f32(),
            ArithmeticPrecision::Bf16Compatibility => {
                ExecutionPrecisionPolicy::bf16_compatibility()
            }
            _ => unreachable!("validated pipeline arithmetic"),
        }
    }
}

fn build_pipeline(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    pipeline::build(request.resolved)
}

fn build_hybrid_pipeline(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    qwen35::binding::build_pipeline(request)
}

/// Planner for preparing a model-aware resident engine build.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResidentModelPlanner;

impl ResidentModelPlanner {
    pub const fn new() -> Self {
        Self
    }

    /// Resolve serving policy from metadata only. This never samples hardware,
    /// binds checkpoint payloads, or starts a rank/owner. Build still performs
    /// source validation and live admission on the consuming owner.
    pub fn prepare_serving(
        &self,
        config: &AutoConfig,
        backend: BackendSelection,
        chat_template_override: Option<&str>,
        options: ModelFactoryOptions,
        serving: ServingModelOptions,
    ) -> Result<ResidentModelBuildPlan> {
        self.prepare_serving_request(
            config,
            backend,
            chat_template_override,
            options,
            SelectionRequest {
                serving,
                numeric_model: None,
            },
        )
    }

    pub fn prepare(
        &self,
        config: &AutoConfig,
        backend: BackendSelection,
        chat_template_override: Option<&str>,
        options: ModelFactoryOptions,
    ) -> Result<ResidentModelBuildPlan> {
        self.prepare_selected(config, backend, chat_template_override, options, None)
    }

    fn prepare_selected(
        &self,
        config: &AutoConfig,
        backend: BackendSelection,
        chat_template_override: Option<&str>,
        options: ModelFactoryOptions,
        execution: Option<ResolvedExecution>,
    ) -> Result<ResidentModelBuildPlan> {
        let entry = BuiltinModelResolver.resolve(config.descriptor(), backend)?;
        let mut plan = self.prepare_with_entry(
            config,
            backend,
            chat_template_override,
            options,
            execution,
            entry,
        )?;
        plan.refresh_facts()?;
        Ok(plan)
    }

    fn prepare_with_entry(
        &self,
        config: &AutoConfig,
        backend: BackendSelection,
        chat_template_override: Option<&str>,
        options: ModelFactoryOptions,
        execution: Option<ResolvedExecution>,
        entry: ResolvedModelBackend,
    ) -> Result<ResidentModelBuildPlan> {
        let descriptor = config.descriptor();
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
        let request = match execution {
            Some(execution) => configure_selected_request(descriptor, entry, options, execution)?,
            None => configure_request(descriptor, entry, options)?,
        };
        Ok(ResidentModelBuildPlan {
            model_name,
            backend,
            backend_profile,
            chat_template,
            requested_backend,
            effective_options: requested_options.clone(),
            requested_options,
            requested_parallel: None,
            topology: resident_topology(),
            provider_adjustments: AdjustmentReport::default(),
            request,
            prepared_source_plan: None,
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
    prepared_source_plan: Option<std::sync::Arc<dyn PreparedSourcePlan>>,
    effective_options: ModelFactoryOptions,
    topology: ModelTopologyPlacement,
    provider_adjustments: AdjustmentReport,
}

/// Model-neutral metadata for one prepared resident engine build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentModelBuildObservability {
    /// Numeric operand arithmetic; model activation/KV/state boundaries remain F32.
    pub numeric_fp8_profile: Option<NumericFp8Profile>,
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
    pub fn expert_read_pipeline_options(
        &self,
    ) -> Option<ferrule_model::transformer::ExpertReadPipelineOptions> {
        self.request.execution.expert_read_pipeline
    }

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
        let mut options = self.effective_options.clone();
        options.max_layers = Some(self.request.max_layers);
        options.scheduler_config = self.request.scheduler_config;
        options.driver_config = self.request.driver_config;
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
        let reason = self.request.execution.adjustment_reason;
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
        report
            .adjustments
            .extend(self.provider_adjustments.adjustments.iter().cloned());
        report
    }

    pub fn selected_implementation(&self) -> SelectedImplementation {
        SelectedImplementation {
            name: self.model_name,
            kind: self.request.execution.construction.kind(),
        }
    }

    /// Catalog support only; source binding and physical admission remain owner-local.
    pub fn support(&self) -> ModelSupport {
        self.request.execution.support
    }

    pub fn topology_placement(&self) -> ModelTopologyPlacement {
        self.topology.clone()
    }

    /// Human-readable pre-build view shared by CLI and build tracing.
    /// Use the typed accessors rather than parsing this presentation string.
    pub fn resolution_report(&self) -> String {
        format!(
            "{} {:?}: backend {:?} -> {:?}, profile={}; numeric arithmetic={}; {:?}; topology={:?}; scheduler requested={:?}, effective={:?}; context={}; adjustments=[{}]; metadata only, owner live admission required (source identity, free memory, allocator/headroom and cleanup are not guaranteed); KV logical pages L and physical transaction slots P remain separately owner-planned",
            self.model_name,
            self.selected_implementation().kind,
            self.requested_backend,
            self.backend,
            self.backend_profile,
            self.numeric_fp8_profile()
                .map_or("not applicable", NumericFp8Profile::arithmetic_description),
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
            numeric_fp8_profile: self.numeric_fp8_profile(),
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
        Ok(self.effective_options.qwen35_moe_capacity)
    }

    /// Numeric operand arithmetic, distinct from model-boundary precision.
    pub const fn numeric_fp8_profile(&self) -> Option<NumericFp8Profile> {
        self.request.execution.numeric_profile
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
        if mixed_serving::tensor::is_selected(&self.request) {
            return None;
        }
        self.request
            .family_options
            .qwen35()
            .and_then(|options| options.moe())
            .map(|options| options.host_cache)
    }

    /// Load model state and build the resident engine on the calling thread.
    pub fn build(self) -> Result<BoxedSessionInferenceEngine> {
        tracing::info!(report = %self.resolution_report(), "resident model build entering owner-local admission");
        let build = self.request.execution.construction.builder();
        let policy = self.request.execution.kv_execution_policy;
        let mut engine = self.build_with(build)?;
        if policy != ferrule_runtime::engine::KvExecutionPolicy::Resident {
            engine.configure_kv_execution(policy)?;
        }
        Ok(engine)
    }

    // A narrow injection seam for owner-admission tests, not a public alternate
    // factory. Production consumes the builder frozen in ResolvedExecution.
    fn build_with<T>(self, build: impl FnOnce(ModelBuildRequest) -> Result<T>) -> Result<T> {
        if let Some(prepared) = &self.prepared_source_plan {
            prepared.validate(&self.request)?;
        }
        build(ModelBuildRequest {
            resolved: self.request,
            prepared_source_plan: self.prepared_source_plan,
        })
    }
}

/// Compatibility DTO for the public catalog builder signature.
/// Its private resolved inputs are consumed by the existing build owner.
#[derive(Debug, Clone)]
pub struct ModelBuildRequest {
    resolved: ResolvedModelRequest,
    prepared_source_plan: Option<std::sync::Arc<dyn PreparedSourcePlan>>,
}

/// Metadata payloads cross the owner boundary without exposing family recipes.
trait PreparedSourcePlan: std::fmt::Debug + Send + Sync {
    fn validate(&self, request: &ResolvedModelRequest) -> Result<()>;
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Internal construction inputs; Qwen3.5 policies stay behind the family adapter.
#[derive(Debug, Clone)]
struct ResolvedModelRequest {
    execution: ResolvedExecution,
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
    options: ModelFactoryOptions,
) -> Result<ResolvedModelRequest> {
    let implementation = implementation(&entry.family)?;
    let strategy = (implementation.select)(
        implementation,
        entry.backend.into(),
        &SelectionRequest::default(),
        &options,
    )?;
    let mut execution = strategy.execution;
    execution.option_policy = strategy.policy;
    configure_selected_request(descriptor, entry, options, execution)
}

fn configure_selected_request(
    descriptor: &ModelDescriptor,
    entry: ResolvedModelBackend,
    mut options: ModelFactoryOptions,
    execution: ResolvedExecution,
) -> Result<ResolvedModelRequest> {
    let family_options = (execution.option_policy.adapt)(descriptor, &mut options, execution)?;
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
        execution,
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

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidRequest {
        message: message.into(),
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
#[path = "../../tests/unit/engine/model_factory/bf16_serving.rs"]
mod dense_bf16_tests;

#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/resolution.rs"]
mod resolution_tests;

#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/expert_read_config.rs"]
mod expert_read_config_tests;

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
            numeric_fp8_profile: None,
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
                numeric_fp8_profile: None,
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

#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/bf16_tiled.rs"]
mod bf16_tiled_tests;

#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/bf16_mixed.rs"]
mod bf16_mixed_tests;

pub use parallel::plan::{
    ParallelCollectiveClaims, ParallelExecutionContract, ParallelExecutionPlan, ParallelExpertMode,
    ParallelRankCollectiveClaim, ParallelRankStorageClaim,
};

pub use parallel::requirements::{
    ParallelAuxiliaryRequirements, ParallelBoundaryRequirements, ParallelOperatorEstimate,
    ParallelOwnerMargins, ParallelProviderPools, ParallelRankCapacity, ParallelRankRequirements,
    ParallelRequirementGap, ParallelResourceRequirements,
};
