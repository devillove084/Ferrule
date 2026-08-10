//! Model-aware catalog selection and resident engine construction.

use std::path::PathBuf;

use ferrule_common::{MemoryPoolLimits, execution::KvLayoutSchema};
use ferrule_model::{
    AutoConfig, ChatTemplate, ExpertMemoryPolicy, ModelDescriptor, ModelExecutionBackend,
    ModelFamily, WeightSource,
    models::qwen3::{Qwen3MoeAdapter, Qwen3MoePrepareOptions},
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
        family: ModelFamily::QwenMoe,
        resolve: resolve_qwen_backend,
        build: build_qwen,
    },
    ModelImplementation {
        family: ModelFamily::DeepSeekV4,
        resolve: resolve_deepseek_backend,
        build: build_deepseek,
    },
    // Dense Qwen3 is recognized but deliberately rejected with a precise reason.
    ModelImplementation {
        family: ModelFamily::Qwen3,
        resolve: resolve_dense_qwen3_backend,
        build: unbuildable_model,
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
    _selection: BackendSelection,
) -> Result<ResolvedModelBackend> {
    Err(unsupported_model(
        descriptor,
        "dense Qwen3 is not supported; the resident implementation is Qwen3-MoE only",
    ))
}

fn unbuildable_model(_request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
    Err(Error::InvalidRequest {
        message: "model implementation is not buildable".into(),
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

/// Model-neutral inputs used to load and compose a resident model.
#[derive(Debug, Clone)]
pub struct ModelFactoryOptions {
    pub max_layers: Option<usize>,
    pub max_tensor_mebibytes: u64,
    pub output_head_chunk_rows: usize,
    pub expert_reader_max_tensor_mebibytes: u64,
    pub expert_cache: ExpertCacheOptions,
    pub moe_hotset_experts: usize,
    pub kv_cache_mebibytes: Option<u64>,
    pub scheduler_config: ResidentSchedulerConfig,
    pub driver_config: ResidentTopKDriverConfig,
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
        let model_name = entry.model_name;
        let backend = entry.backend;
        let backend_profile = entry.backend_profile;
        let request = configure_request(descriptor, entry, options)?;

        Ok(ResidentModelBuildPlan {
            model_name,
            backend,
            backend_profile,
            chat_template,
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
    request: ModelBuildRequest,
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

    /// Load model state and build the resident engine on the calling thread.
    pub fn build(self) -> Result<BoxedSessionInferenceEngine> {
        let family = &self.request.family;
        let implementation = MODEL_IMPLEMENTATIONS
            .iter()
            .find(|entry| &entry.family == family)
            .ok_or_else(|| Error::InvalidRequest {
                message: format!("no resident model implementation can build family '{family}'"),
            })?;
        (implementation.build)(self.request)
    }
}

/// Validated resident-model construction inputs for one catalog family.
#[derive(Debug, Clone)]
pub struct ModelBuildRequest {
    family: ModelFamily,
    backend: ModelExecutionBackend,
    model_path: PathBuf,
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
) -> Result<ModelBuildRequest> {
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

    Ok(ModelBuildRequest {
        family: entry.family,
        backend: entry.backend,
        model_path: descriptor.path.clone(),
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

fn qwen_prepare_options(request: &ModelBuildRequest) -> Qwen3MoePrepareOptions {
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

#[cfg(feature = "cuda")]
fn deepseek_prepare_options(request: &ModelBuildRequest) -> Result<DeepSeekV4PrepareOptions> {
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
    request: &ModelBuildRequest,
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

    fn descriptor(family: ModelFamily, architecture: &str, layers: usize) -> ModelDescriptor {
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

    fn options() -> ModelFactoryOptions {
        ModelFactoryOptions {
            max_layers: None,
            max_tensor_mebibytes: 128,
            output_head_chunk_rows: 4096,
            expert_reader_max_tensor_mebibytes: 64,
            expert_cache: ExpertCacheOptions::default(),
            moe_hotset_experts: 0,
            kv_cache_mebibytes: None,
            scheduler_config: ResidentSchedulerConfig::default(),
            driver_config: ResidentTopKDriverConfig::default(),
        }
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

        let dense = descriptor(ModelFamily::Qwen3, "qwen3", 40);
        assert!(
            resolver
                .resolve(&dense, BackendSelection::Auto)
                .unwrap_err()
                .to_string()
                .contains("dense Qwen3")
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
