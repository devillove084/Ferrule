//! Immutable DeepSeek-V4 preparation output and execution policy resolution.

use std::env as process_environment;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ferrule_common::execution::{KvLayoutSchema, KvPlaneDescriptor};
use ferrule_common::kernel_plan::ModelKernelPlan;
#[cfg(feature = "cuda")]
use ferrule_common::kernel_plan::{
    KernelOperation, LayerKernelRequirements, LinearBundleRequirement, WeightLayout,
};
use ferrule_common::{Error, Result};

#[cfg(feature = "cuda")]
use crate::checkpoint::weight::{LinearWeight, LinearWeightFormat};
use crate::checkpoint::{CheckpointSourceCatalog, CheckpointTensorSlice, HfSafetensorsInventory};
use crate::execution::{
    ExecutableStage, PreparedExecutable, PreparedModel, ResourceLayout, ResourceManifest,
    StageResourceUse, TransformerResourceSlot, TransformerStage, WorkspaceClaim,
};
use crate::materialization::{
    HF_SAFETENSORS_TENSOR_BUNDLE_V1, MaterializationSourceCatalog, MaterializationSourceEntry,
};
use crate::moe::streaming::{
    ExpertLoadSource, ExpertMemoryPolicy, ExpertSourceCatalog, ExpertStreamingPolicy,
};

use super::checkpoint::DeepSeekV4Checkpoint;
use super::layer::DeepSeekV4Layer;
use super::mtp::DeepSeekV4MtpModel;

static NEXT_PREPARED_GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4PrepareOptions {
    pub max_layers: usize,
    pub output_head_chunk_rows: usize,
    pub expert_reader_max_tensor_bytes: u64,
    pub moe_prefetch_experts: usize,
    /// Retention limits for pageable and pinned whole-expert host caches.
    pub expert_memory_policy: ExpertMemoryPolicy,
    /// Optional bounded per-layer resident hotset. `0` keeps the managed-memory
    /// default (no planner eviction); non-zero clamps residency to at least
    /// `num_experts_per_tok` and retains hottest routed experts first.
    pub moe_hotset_experts: usize,
}

impl Default for DeepSeekV4PrepareOptions {
    fn default() -> Self {
        Self {
            max_layers: crate::families::deepseek_v4::NUM_LAYERS,
            output_head_chunk_rows: 1024,
            expert_reader_max_tensor_bytes: 64 * 1024 * 1024,
            moe_prefetch_experts: 0,
            expert_memory_policy: ExpertMemoryPolicy::default(),
            moe_hotset_experts: 0,
        }
    }
}

/// Physical paged-KV layout compiled for DeepSeek-V4 resident execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4KvLayoutSchema {
    layer_count: usize,
    window_size: usize,
    head_dim: usize,
    compress_ratios: Box<[usize]>,
    planes: Box<[KvPlaneDescriptor]>,
    page_size: usize,
    max_sequence_len: usize,
}

impl DeepSeekV4KvLayoutSchema {
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub const fn window_size(&self) -> usize {
        self.window_size
    }

    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub const fn page_size(&self) -> usize {
        self.page_size
    }

    /// Bytes per physical CUDA page for the three f32 token-scaled data planes.
    pub fn cuda_f32_data_page_bytes(&self) -> Result<u64> {
        let data_planes = self.planes.get(..3).ok_or_else(|| {
            Error::Model("DeepSeek-V4 KV schema is missing CUDA data planes".into())
        })?;
        let elements_per_page = data_planes.iter().try_fold(0usize, |total, plane| {
            self.page_size
                .checked_mul(plane.elements_per_token)
                .and_then(|elements| elements.checked_mul(plane.layer_count))
                .and_then(|elements| total.checked_add(elements))
                .ok_or_else(|| Error::Model("DeepSeek-V4 CUDA KV page size overflow".into()))
        })?;
        let bytes = elements_per_page
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::Model("DeepSeek-V4 CUDA KV page byte size overflow".into()))?;
        u64::try_from(bytes)
            .map_err(|_| Error::Model("DeepSeek-V4 CUDA KV page bytes exceed u64".into()))
    }

    pub fn compress_ratios(&self) -> &[usize] {
        &self.compress_ratios
    }
}

impl KvLayoutSchema for DeepSeekV4KvLayoutSchema {
    fn planes(&self) -> &[KvPlaneDescriptor] {
        &self.planes
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn max_sequence_len(&self) -> usize {
        self.max_sequence_len
    }
}

/// Environment-derived controls frozen at plan preparation time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4ExecutionPolicy {
    prefill_progress: bool,
    lookahead_prefetch: bool,
    decode_lookahead_prefetch: bool,
    hash_decode_prefetch_window: bool,
    prefill_hash_lookahead: bool,
    managed_experts: bool,
    expert_upload_inflight: usize,
    profile: bool,
    profile_sync: bool,
}

impl Default for DeepSeekV4ExecutionPolicy {
    fn default() -> Self {
        Self {
            prefill_progress: false,
            lookahead_prefetch: true,
            decode_lookahead_prefetch: true,
            hash_decode_prefetch_window: false,
            prefill_hash_lookahead: false,
            managed_experts: true,
            expert_upload_inflight: 32,
            profile: false,
            profile_sync: false,
        }
    }
}

impl DeepSeekV4ExecutionPolicy {
    pub const fn prefill_progress(&self) -> bool {
        self.prefill_progress
    }

    pub const fn lookahead_prefetch(&self) -> bool {
        self.lookahead_prefetch
    }

    pub const fn decode_lookahead_prefetch(&self) -> bool {
        self.decode_lookahead_prefetch
    }

    pub const fn hash_decode_prefetch_window(&self) -> bool {
        self.hash_decode_prefetch_window
    }

    pub const fn prefill_hash_lookahead(&self) -> bool {
        self.prefill_hash_lookahead
    }

    pub const fn managed_experts(&self) -> bool {
        self.managed_experts
    }

    pub const fn expert_upload_inflight(&self) -> usize {
        self.expert_upload_inflight
    }

    pub const fn profile_enabled(&self) -> bool {
        self.profile
    }

    pub const fn profile_sync(&self) -> bool {
        self.profile_sync
    }

    fn resolve() -> Result<Self> {
        Self::resolve_with(|name| match process_environment::var_os(name) {
            None => Ok(None),
            Some(value) => value.into_string().map(Some).map_err(|_| {
                Error::Model(format!(
                    "DeepSeek-V4 execution policy environment variable {name} is not valid Unicode"
                ))
            }),
        })
    }

    fn resolve_with(mut lookup: impl FnMut(&str) -> Result<Option<String>>) -> Result<Self> {
        let prefill_progress = parse_env_bool(
            "FERRULE_DSV4_PREFILL_PROGRESS",
            lookup("FERRULE_DSV4_PREFILL_PROGRESS")?,
            false,
        )?;
        let lookahead_prefetch = parse_env_bool(
            "FERRULE_DSV4_LOOKAHEAD_PREFETCH",
            lookup("FERRULE_DSV4_LOOKAHEAD_PREFETCH")?,
            true,
        )?;
        let decode_lookahead_prefetch = parse_env_bool(
            "FERRULE_DSV4_DECODE_LOOKAHEAD_PREFETCH",
            lookup("FERRULE_DSV4_DECODE_LOOKAHEAD_PREFETCH")?,
            true,
        )?;
        let hash_decode_prefetch_window = parse_env_bool(
            "FERRULE_DSV4_HASH_PREFETCH_WINDOW",
            lookup("FERRULE_DSV4_HASH_PREFETCH_WINDOW")?,
            false,
        )?;
        let prefill_hash_lookahead = parse_env_bool(
            "FERRULE_DSV4_PREFILL_HASH_LOOKAHEAD",
            lookup("FERRULE_DSV4_PREFILL_HASH_LOOKAHEAD")?,
            false,
        )?;
        let managed_experts = parse_env_bool(
            "FERRULE_MANAGED_EXPERTS",
            lookup("FERRULE_MANAGED_EXPERTS")?,
            true,
        )?;

        let expert_upload_inflight = parse_env_usize(
            "FERRULE_DSV4_EXPERT_UPLOAD_INFLIGHT",
            lookup("FERRULE_DSV4_EXPERT_UPLOAD_INFLIGHT")?,
            32,
        )?;

        let profile_sync = parse_env_bool(
            "FERRULE_DSV4_PROFILE_SYNC",
            lookup("FERRULE_DSV4_PROFILE_SYNC")?,
            false,
        )?;
        // Existing diagnostic modes preserve their semantics by implying
        // profiling even when PROFILE is absent or explicitly off.
        let profile = profile_sync
            || prefill_progress
            || parse_env_bool(
                "FERRULE_DSV4_PROFILE",
                lookup("FERRULE_DSV4_PROFILE")?,
                false,
            )?;

        Ok(Self {
            prefill_progress,
            lookahead_prefetch,
            decode_lookahead_prefetch,
            hash_decode_prefetch_window,
            prefill_hash_lookahead,
            managed_experts,
            expert_upload_inflight,
            profile,
            profile_sync,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DeepSeekV4PreparedLayerExperts {
    source_catalog: Arc<ExpertSourceCatalog>,
    streaming_policy: ExpertStreamingPolicy,
    resident_capacity: usize,
    prefetch_capacity: usize,
}

impl DeepSeekV4PreparedLayerExperts {
    pub(crate) fn new(
        source_catalog: Arc<ExpertSourceCatalog>,
        streaming_policy: ExpertStreamingPolicy,
    ) -> Self {
        Self {
            resident_capacity: streaming_policy.gpu_slots_per_layer,
            prefetch_capacity: streaming_policy.prefetch_per_layer,
            source_catalog,
            streaming_policy,
        }
    }

    pub fn source_catalog(&self) -> &Arc<ExpertSourceCatalog> {
        &self.source_catalog
    }

    pub const fn streaming_policy(&self) -> &ExpertStreamingPolicy {
        &self.streaming_policy
    }

    pub const fn resident_capacity(&self) -> usize {
        self.resident_capacity
    }

    pub const fn prefetch_capacity(&self) -> usize {
        self.prefetch_capacity
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DeepSeekV4InstallDescriptor {
    StaticTensorBundle(Arc<ResourceManifest>),
    RoutedExpert(ExpertLoadSource),
}

pub struct DeepSeekV4PreparedResources {
    model: DeepSeekV4Checkpoint,
    options: DeepSeekV4PrepareOptions,
    layers: Box<[DeepSeekV4Layer]>,
    layer_experts: Box<[DeepSeekV4PreparedLayerExperts]>,
    /// Bound DSpark attachment owned by the same immutable prepared generation
    /// as the target model. CUDA image compilation is a subsequent R1 step.
    mtp: Option<DeepSeekV4MtpModel>,
    mtp_layer_experts: Box<[DeepSeekV4PreparedLayerExperts]>,
    materialization_sources: Arc<MaterializationSourceCatalog<DeepSeekV4InstallDescriptor>>,
    kv_layout: DeepSeekV4KvLayoutSchema,
    policy: DeepSeekV4ExecutionPolicy,
    /// Required per-layer semantic superkernel plan. Missing operations are
    /// prepare-time errors; row-count schedule selection is provider-owned.
    kernel_plan: ModelKernelPlan,
    /// Transformer-body and stage-zero DSpark projection plan for attachment
    /// execution layers. Prediction-head plans remain explicit follow-up work.
    mtp_transformer_kernel_plan: Option<ModelKernelPlan>,
}

impl DeepSeekV4PreparedResources {
    pub const fn model(&self) -> &DeepSeekV4Checkpoint {
        &self.model
    }

    pub const fn prepare_options(&self) -> &DeepSeekV4PrepareOptions {
        &self.options
    }

    pub fn layers(&self) -> &[DeepSeekV4Layer] {
        &self.layers
    }

    pub fn layer_experts(&self) -> &[DeepSeekV4PreparedLayerExperts] {
        &self.layer_experts
    }

    pub const fn mtp(&self) -> Option<&DeepSeekV4MtpModel> {
        self.mtp.as_ref()
    }

    pub fn mtp_layer_experts(&self) -> &[DeepSeekV4PreparedLayerExperts] {
        &self.mtp_layer_experts
    }

    pub fn materialization_source_count(&self) -> usize {
        self.materialization_sources.len()
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn materialization_sources(
        &self,
    ) -> &Arc<MaterializationSourceCatalog<DeepSeekV4InstallDescriptor>> {
        &self.materialization_sources
    }

    pub fn layer_expert_source_catalog(&self, layer: usize) -> Option<&Arc<ExpertSourceCatalog>> {
        self.layer_experts
            .get(layer)
            .map(DeepSeekV4PreparedLayerExperts::source_catalog)
    }

    pub fn layer_resident_expert_capacity(&self, layer: usize) -> Option<usize> {
        self.layer_experts
            .get(layer)
            .map(DeepSeekV4PreparedLayerExperts::resident_capacity)
    }

    pub fn layer_prefetch_expert_capacity(&self, layer: usize) -> Option<usize> {
        self.layer_experts
            .get(layer)
            .map(DeepSeekV4PreparedLayerExperts::prefetch_capacity)
    }

    pub const fn kv_layout(&self) -> &DeepSeekV4KvLayoutSchema {
        &self.kv_layout
    }

    pub const fn policy(&self) -> &DeepSeekV4ExecutionPolicy {
        &self.policy
    }

    /// Returns the per-layer kernel plan (executable plan, Section 3.2).
    pub fn kernel_plan(&self) -> &ModelKernelPlan {
        &self.kernel_plan
    }

    pub const fn mtp_transformer_kernel_plan(&self) -> Option<&ModelKernelPlan> {
        self.mtp_transformer_kernel_plan.as_ref()
    }
}

pub type DeepSeekV4PreparedModelPlan = PreparedModel<DeepSeekV4PreparedResources, TransformerStage>;

/// Validates and atomically prepares all immutable DSV4 model-global resources.
pub fn prepare(
    model: DeepSeekV4Checkpoint,
    options: DeepSeekV4PrepareOptions,
) -> Result<DeepSeekV4PreparedModelPlan> {
    validate_options(&model, options)?;
    let policy = DeepSeekV4ExecutionPolicy::resolve()?;
    let mtp = model.load_mtp()?;
    validate_mtp_attachment(&model, mtp.as_ref())?;

    let mut layers = Vec::new();
    layers
        .try_reserve_exact(options.max_layers)
        .map_err(|error| {
            Error::Model(format!(
                "DeepSeek-V4 prepared layer allocation failed for {} layers: {error}",
                options.max_layers
            ))
        })?;
    let mut layer_experts = Vec::new();
    layer_experts
        .try_reserve_exact(options.max_layers)
        .map_err(|error| {
            Error::Model(format!(
                "DeepSeek-V4 prepared expert catalog allocation failed for {} layers: {error}",
                options.max_layers
            ))
        })?;
    let expert_streaming_policy = model
        .resolved_expert_streaming_policy(options.moe_prefetch_experts, options.moe_hotset_experts);
    for layer in 0..options.max_layers {
        let source_catalog = Arc::clone(model.expert_source_catalog(layer)?);
        if source_catalog.count() != model.config.num_routed_experts {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer {layer} catalog has {} routed experts, expected {}",
                source_catalog.count(),
                model.config.num_routed_experts
            )));
        }
        layers.push(model.bind_layer(layer)?);
        layer_experts.push(DeepSeekV4PreparedLayerExperts::new(
            source_catalog,
            expert_streaming_policy.clone(),
        ));
    }
    let mtp_layer_experts = mtp
        .as_ref()
        .map(|mtp| {
            mtp.layers
                .iter()
                .map(|layer| {
                    DeepSeekV4PreparedLayerExperts::new(
                        Arc::clone(&layer.expert_source_catalog),
                        expert_streaming_policy.clone(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
        .into_boxed_slice();

    let max_compress_ratio = model
        .config
        .compress_ratios
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let main_metadata_width = 4usize
        .saturating_mul(max_compress_ratio)
        .saturating_mul(model.config.head_dim);
    let indexer_metadata_width = 4usize
        .saturating_mul(max_compress_ratio)
        .saturating_mul(model.config.index_head_dim);
    // DSpark stages own committed target-context KV at their non-aliasing
    // execution identities (43–45 for the release checkpoint). Keeping these
    // slots in the same paged transaction makes target and DSpark context
    // promotion/rollback atomic while proposal-block KV remains scratch-only.
    let kv_layer_count = if let Some(mtp) = mtp.as_ref() {
        let attachment_end = model
            .config
            .num_layers
            .checked_add(mtp.layers.len())
            .ok_or_else(|| Error::Model("DeepSeek-V4 DSpark KV layer count overflow".into()))?;
        options.max_layers.max(attachment_end)
    } else {
        options.max_layers
    };
    let kv_layout = DeepSeekV4KvLayoutSchema {
        layer_count: kv_layer_count,
        window_size: model.config.window_size,
        head_dim: model.config.head_dim,
        planes: vec![
            KvPlaneDescriptor {
                name: "window_latent_kv",
                elements_per_token: model.config.head_dim,
                layer_count: kv_layer_count,
            },
            KvPlaneDescriptor {
                name: "compressed_main_kv",
                elements_per_token: model.config.head_dim,
                layer_count: kv_layer_count,
            },
            KvPlaneDescriptor {
                name: "indexer_kv",
                elements_per_token: model.config.index_head_dim,
                layer_count: kv_layer_count,
            },
            KvPlaneDescriptor {
                name: "compressor_metadata",
                elements_per_token: main_metadata_width,
                layer_count: kv_layer_count,
            },
            KvPlaneDescriptor {
                name: "indexer_metadata",
                elements_per_token: indexer_metadata_width,
                layer_count: kv_layer_count,
            },
        ]
        .into_boxed_slice(),
        page_size: 16,
        max_sequence_len: u32::MAX as usize,
        compress_ratios: (0..kv_layer_count)
            .map(|layer| {
                model
                    .config
                    .compress_ratios
                    .get(layer)
                    .copied()
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    };
    #[cfg(feature = "cuda")]
    let kernel_plan = {
        let requirements = layers
            .iter()
            .map(deepseek_v4_layer_kernel_requirements)
            .collect::<Result<Vec<_>>>()?;
        ferrule_cuda::provider::compile_cuda_model_plan(&requirements)?
    };
    #[cfg(feature = "cuda")]
    let mtp_transformer_kernel_plan = mtp
        .as_ref()
        .map(|mtp| {
            let requirements = mtp
                .layers
                .iter()
                .enumerate()
                .map(|(stage, layer)| {
                    deepseek_v4_mtp_kernel_requirements(
                        stage,
                        layer,
                        mtp.config.target_layer_ids.len(),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            ferrule_cuda::provider::compile_cuda_model_plan(&requirements)
        })
        .transpose()?;
    #[cfg(not(feature = "cuda"))]
    let kernel_plan = ModelKernelPlan::new(options.max_layers);
    #[cfg(not(feature = "cuda"))]
    let mtp_transformer_kernel_plan = mtp
        .as_ref()
        .map(|mtp| ModelKernelPlan::new(mtp.layers.len()));
    let executable = prepare_executable(&model, options.max_layers)?;
    let materialization_sources = Arc::new(prepare_materialization_sources(
        &executable,
        &layer_experts,
        mtp.as_ref(),
        &mtp_layer_experts,
    )?);
    let resources = DeepSeekV4PreparedResources {
        model,
        options,
        layers: layers.into_boxed_slice(),
        layer_experts: layer_experts.into_boxed_slice(),
        mtp,
        mtp_layer_experts,
        materialization_sources,
        kv_layout,
        policy,
        kernel_plan,
        mtp_transformer_kernel_plan,
    };

    publish_prepared(Ok((resources, executable)))
}

fn prepare_materialization_sources(
    executable: &PreparedExecutable<TransformerStage>,
    layer_experts: &[DeepSeekV4PreparedLayerExperts],
    mtp: Option<&DeepSeekV4MtpModel>,
    mtp_layer_experts: &[DeepSeekV4PreparedLayerExperts],
) -> Result<MaterializationSourceCatalog<DeepSeekV4InstallDescriptor>> {
    let static_entries =
        MaterializationSourceCatalog::from_checkpoint_manifests(executable.resources())?
            .iter()
            .map(|entry| {
                MaterializationSourceEntry::new(
                    entry.resource(),
                    entry.source(),
                    entry.read_plan().clone(),
                    DeepSeekV4InstallDescriptor::StaticTensorBundle(Arc::new(
                        entry.descriptor().clone(),
                    )),
                )
            })
            .collect::<Result<Vec<_>>>()?;
    let mut entries = static_entries;
    for experts in layer_experts {
        entries.extend(
            experts
                .source_catalog()
                .materialization_sources()?
                .iter()
                .map(|entry| {
                    MaterializationSourceEntry::new(
                        entry.resource(),
                        entry.source(),
                        entry.read_plan().clone(),
                        DeepSeekV4InstallDescriptor::RoutedExpert(entry.descriptor().clone()),
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        );
    }
    if let Some(mtp) = mtp {
        if mtp.layers.len() != mtp_layer_experts.len() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 DSpark stage/expert source mismatch: stages={} sources={}",
                mtp.layers.len(),
                mtp_layer_experts.len()
            )));
        }
        for experts in mtp_layer_experts {
            entries.extend(
                experts
                    .source_catalog()
                    .materialization_sources()?
                    .iter()
                    .map(|entry| {
                        MaterializationSourceEntry::new(
                            entry.resource(),
                            entry.source(),
                            entry.read_plan().clone(),
                            DeepSeekV4InstallDescriptor::RoutedExpert(entry.descriptor().clone()),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
    }
    MaterializationSourceCatalog::new(entries)
}

fn prepare_executable(
    model: &DeepSeekV4Checkpoint,
    max_layers: usize,
) -> Result<PreparedExecutable<TransformerStage>> {
    prepare_executable_from_inventory(&model.descriptor.path, model.inventory(), max_layers)
}

fn prepare_executable_from_inventory(
    model_dir: &std::path::Path,
    inventory: &HfSafetensorsInventory,
    max_layers: usize,
) -> Result<PreparedExecutable<TransformerStage>> {
    use crate::semantic::HyperConnectionStage;

    struct StaticBundle {
        slot: TransformerResourceSlot,
        operation: TransformerStage,
        tensors: Vec<CheckpointTensorSlice>,
    }

    let family = &inventory.family;
    let mut bundles = Vec::new();

    let top_level = |role: crate::TensorRole| {
        inventory
            .tensors
            .iter()
            .filter(|tensor| tensor.role == role)
            .map(|tensor| CheckpointTensorSlice::from_hf_inventory(model_dir, tensor))
            .collect::<Vec<_>>()
    };
    bundles.push(StaticBundle {
        slot: TransformerResourceSlot::Embedding,
        operation: TransformerStage::Embed,
        tensors: top_level(crate::TensorRole::TokenEmbedding),
    });

    for layer in 0..max_layers {
        let layer_id = u32::try_from(layer)
            .map_err(|_| Error::Model("DeepSeek-V4 layer index exceeds u32".into()))?;
        let layer_prefix = format!("layers.{layer}.");
        let mut attention = Vec::new();
        let mut router = Vec::new();
        let mut feed_forward = Vec::new();
        for tensor in &inventory.tensors {
            if tensor.name.starts_with("mtp.") {
                continue;
            }
            let target = if crate::families::parse_hf_attention_tensor(family, &tensor.name)
                .is_some_and(|descriptor| descriptor.layer == layer)
                || crate::families::parse_hf_hyper_connection_tensor(family, &tensor.name)
                    .is_some_and(|descriptor| {
                        descriptor.layer == Some(layer)
                            && descriptor.stage == HyperConnectionStage::Attention
                    })
                || tensor.role == crate::TensorRole::AttentionNorm
                    && tensor.name.starts_with(&layer_prefix)
            {
                Some(&mut attention)
            } else if crate::families::parse_hf_router_tensor(family, &tensor.name)
                .is_some_and(|descriptor| descriptor.layer == layer)
            {
                Some(&mut router)
            } else if crate::families::parse_hf_shared_expert_tensor(family, &tensor.name)
                .is_some_and(|descriptor| descriptor.layer == layer)
                || crate::families::parse_hf_hyper_connection_tensor(family, &tensor.name)
                    .is_some_and(|descriptor| {
                        descriptor.layer == Some(layer)
                            && descriptor.stage == HyperConnectionStage::FeedForward
                    })
                || tensor.role == crate::TensorRole::FeedForwardNorm
                    && tensor.name.starts_with(&layer_prefix)
            {
                Some(&mut feed_forward)
            } else {
                None
            };
            if let Some(target) = target {
                target.push(CheckpointTensorSlice::from_hf_inventory(model_dir, tensor));
            }
        }
        bundles.extend([
            StaticBundle {
                slot: TransformerResourceSlot::Attention { layer: layer_id },
                operation: TransformerStage::Attention { layer: layer_id },
                tensors: attention,
            },
            StaticBundle {
                slot: TransformerResourceSlot::Router { layer: layer_id },
                operation: TransformerStage::Router { layer: layer_id },
                tensors: router,
            },
            StaticBundle {
                slot: TransformerResourceSlot::FeedForward { layer: layer_id },
                operation: TransformerStage::FeedForward { layer: layer_id },
                tensors: feed_forward,
            },
        ]);
    }

    let mut output = inventory
        .tensors
        .iter()
        .filter(|tensor| {
            !tensor.name.starts_with("mtp.")
                && (matches!(
                    tensor.role,
                    crate::TensorRole::OutputNorm | crate::TensorRole::OutputHead
                ) || crate::families::parse_hf_hyper_connection_tensor(family, &tensor.name)
                    .is_some_and(|descriptor| {
                        descriptor.layer.is_none() && descriptor.stage == HyperConnectionStage::Head
                    }))
        })
        .map(|tensor| CheckpointTensorSlice::from_hf_inventory(model_dir, tensor))
        .collect::<Vec<_>>();
    output.sort_by(|left, right| left.name.cmp(&right.name));
    bundles.push(StaticBundle {
        slot: TransformerResourceSlot::Output,
        operation: TransformerStage::Output,
        tensors: output,
    });

    for (&attachment, tensors) in &inventory.mtp_layer_tensors() {
        let attachment = u32::try_from(attachment)
            .map_err(|_| Error::Model("DeepSeek-V4 attachment index exceeds u32".into()))?;
        let tensors = tensors
            .iter()
            .filter(|tensor| !tensor.name.contains(".ffn.experts."))
            .map(|tensor| CheckpointTensorSlice::from_hf_inventory(model_dir, tensor))
            .collect::<Vec<_>>();
        bundles.push(StaticBundle {
            slot: TransformerResourceSlot::Attachment { index: attachment },
            operation: TransformerStage::Attachment { index: attachment },
            tensors,
        });
    }

    if let Some(empty) = bundles.iter().find(|bundle| bundle.tensors.is_empty()) {
        return Err(Error::Model(format!(
            "DeepSeek-V4 prepared stage {:?} has no checkpoint-backed tensors",
            empty.operation
        )));
    }
    for bundle in &mut bundles {
        bundle.tensors.sort_by(|left, right| {
            left.role
                .cmp(&right.role)
                .then_with(|| left.name.cmp(&right.name))
                .then_with(|| left.path.cmp(&right.path))
                .then_with(|| left.offset.cmp(&right.offset))
                .then_with(|| left.bytes.cmp(&right.bytes))
        });
    }
    let sources =
        CheckpointSourceCatalog::capture(bundles.iter().flat_map(|bundle| bundle.tensors.iter()))?;
    let mut manifests = Vec::with_capacity(bundles.len());
    let mut stages = Vec::with_capacity(bundles.len());
    for bundle in bundles {
        let resource = bundle.slot.parameter();
        let bundle_source = sources.bundle_source(
            b"deepseek-v4-static-stage-bundle-v1",
            HF_SAFETENSORS_TENSOR_BUNDLE_V1,
            &bundle.tensors,
        )?;
        manifests.push(ResourceManifest::checkpoint(
            resource,
            bundle_source,
            bundle.tensors,
            ResourceLayout::TensorBundle,
        )?);
        stages.push(ExecutableStage::new(
            bundle.operation,
            [StageResourceUse::read(resource)],
            WorkspaceClaim::NONE,
        ));
    }
    PreparedExecutable::new(manifests, stages).map_err(Into::into)
}

#[cfg(feature = "cuda")]
fn deepseek_v4_layer_kernel_requirements(
    layer: &DeepSeekV4Layer,
) -> Result<LayerKernelRequirements> {
    let mut requirements = LayerKernelRequirements::default();
    if layer.hc_config.hc_mult != 4
        || layer.hc_config.hidden_size != 4096
        || layer.hc_config.mix_hc() != 24
        || layer.attn_norm.len() != 4096
        || layer.ffn_norm.len() != 4096
    {
        return Err(Error::Model(format!(
            "SM121 HC producer requires hc=4 hidden=4096 mix=24 at layer {}, got hc={} hidden={} mix={} attn_norm={} ffn_norm={}",
            layer.layer,
            layer.hc_config.hc_mult,
            layer.hc_config.hidden_size,
            layer.hc_config.mix_hc(),
            layer.attn_norm.len(),
            layer.ffn_norm.len()
        )));
    }
    validate_shared_ffn_requirement(layer)?;
    validate_mla_output_requirement(layer)?;
    requirements.require_operation(KernelOperation::AttentionHcPre);
    requirements.require_operation(KernelOperation::FeedForwardHcPre);
    requirements.require_operation(KernelOperation::SharedFfn);
    requirements.require_operation(KernelOperation::RoutedFp4Moe);
    requirements.require_operation(KernelOperation::MlaOutput);
    requirements.add_linear_bundle(fp8_linear_bundle_requirement(
        KernelOperation::MlaQueryAKv,
        [
            &layer.attention.payload.query_a,
            &layer.attention.payload.key_value,
        ],
    )?);
    requirements.add_linear_bundle(fp8_single_linear_requirement(
        KernelOperation::MlaQueryB,
        &layer.attention.payload.query_b,
    )?);

    let Some(compressed) = layer.attention.compressed.as_ref() else {
        return Ok(requirements);
    };

    requirements.add_linear_bundle(bf16_linear_bundle_requirement(
        KernelOperation::MainCompressorProjection,
        [&compressed.compressor.wkv, &compressed.compressor.wgate],
    )?);
    if let Some(indexer) = compressed.indexer.as_ref() {
        requirements.add_linear_bundle(bf16_linear_bundle_requirement(
            KernelOperation::IndexerCompressorProjection,
            [&indexer.compressor.wkv, &indexer.compressor.wgate],
        )?);
    }
    Ok(requirements)
}

#[cfg(feature = "cuda")]
fn validate_shared_ffn_requirement(layer: &DeepSeekV4Layer) -> Result<()> {
    let formats = (
        &layer.shared_ffn.gate.format,
        &layer.shared_ffn.up.format,
        &layer.shared_ffn.down.format,
    );
    let (
        LinearWeightFormat::Fp8E4M3WithE8M0Scale {
            out_features: gate_out,
            in_features: gate_in,
            block_m: 128,
            block_k: 128,
        },
        LinearWeightFormat::Fp8E4M3WithE8M0Scale {
            out_features: up_out,
            in_features: up_in,
            block_m: 128,
            block_k: 128,
        },
        LinearWeightFormat::Fp8E4M3WithE8M0Scale {
            out_features: down_out,
            in_features: down_in,
            block_m: 128,
            block_k: 128,
        },
    ) = formats
    else {
        return Err(Error::Model(format!(
            "SM121 shared FFN requires FP8 K128 weights at layer {}: gate={:?} up={:?} down={:?}",
            layer.layer, formats.0, formats.1, formats.2
        )));
    };
    if gate_in != up_in
        || gate_out != up_out
        || down_in != gate_out
        || !gate_in.is_multiple_of(128)
        || !gate_out.is_multiple_of(128)
        || !down_out.is_multiple_of(16)
        || !layer.shared_ffn.swiglu_limit.is_finite()
    {
        return Err(Error::Model(format!(
            "SM121 shared FFN shape is unsupported at layer {}: gate=[{gate_out},{gate_in}] up=[{up_out},{up_in}] down=[{down_out},{down_in}] limit={}",
            layer.layer, layer.shared_ffn.swiglu_limit
        )));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn deepseek_v4_mtp_kernel_requirements(
    stage: usize,
    layer: &super::mtp::DeepSeekV4MtpLayer,
    target_layer_count: usize,
) -> Result<LayerKernelRequirements> {
    let mut requirements = deepseek_v4_layer_kernel_requirements(&layer.transformer)?;
    let attention = layer.transformer.attention.config;
    if attention.num_heads != ferrule_cuda::cutlass::DSPARK_ATTENTION_HEADS
        || attention.head_dim != ferrule_cuda::cutlass::DSPARK_ATTENTION_HEAD_DIM
        || attention.window_size != ferrule_cuda::cutlass::DSPARK_ATTENTION_WINDOW
        || attention.compress_ratio != 0
    {
        return Err(Error::Model(format!(
            "SM121 DSpark hybrid attention shape mismatch at stage {stage}: heads={} head_dim={} window={} compress_ratio={}",
            attention.num_heads,
            attention.head_dim,
            attention.window_size,
            attention.compress_ratio
        )));
    }
    requirements.require_operation(KernelOperation::DsparkHybridMlaAttention);
    if stage == 0 {
        requirements.require_operation(KernelOperation::DsparkProposalHead);
    }
    if stage != 0 {
        return Ok(requirements);
    }
    let main_proj = layer
        .main_proj
        .as_ref()
        .ok_or_else(|| Error::Model("DeepSeek-V4 DSpark stage zero is missing main_proj".into()))?;
    let main_norm = layer
        .main_norm
        .as_deref()
        .ok_or_else(|| Error::Model("DeepSeek-V4 DSpark stage zero is missing main_norm".into()))?;
    let LinearWeightFormat::Fp8E4M3WithE8M0Scale {
        out_features,
        in_features,
        block_m: 128,
        block_k: 128,
    } = &main_proj.format
    else {
        return Err(Error::Model(format!(
            "SM121 DSpark main projection requires FP8/E8M0 K128 weights, got {:?}",
            main_proj.format
        )));
    };
    if *out_features != layer.transformer.hc_config.hidden_size
        || *in_features != main_norm.len().saturating_mul(target_layer_count)
        || main_norm.len() != *out_features
        || !out_features.is_multiple_of(128)
        || !in_features.is_multiple_of(128)
    {
        return Err(Error::Model(format!(
            "SM121 DSpark main projection shape mismatch: weight=[{out_features},{in_features}] norm={} hidden={} target_layers={target_layer_count}",
            main_norm.len(),
            layer.transformer.hc_config.hidden_size
        )));
    }
    requirements.require_operation(KernelOperation::DsparkMainProjectNorm);
    Ok(requirements)
}

#[cfg(feature = "cuda")]
fn validate_mla_output_requirement(layer: &DeepSeekV4Layer) -> Result<()> {
    let cfg = layer.attention.config;
    let output_a = &layer.attention.payload.output_a.format;
    let output_b = &layer.attention.payload.output_b.format;
    let (
        LinearWeightFormat::Fp8E4M3WithE8M0Scale {
            out_features: output_a_out,
            in_features: output_a_in,
            block_m: 128,
            block_k: 128,
        },
        LinearWeightFormat::Fp8E4M3WithE8M0Scale {
            out_features: output_b_out,
            in_features: output_b_in,
            block_m: 128,
            block_k: 128,
        },
    ) = (output_a, output_b)
    else {
        return Err(Error::Model(format!(
            "SM121 MLA output requires FP8/E8M0 output-A and output-B at layer {}: output_a={output_a:?} output_b={output_b:?}",
            layer.layer
        )));
    };
    if *output_a_out != cfg.output_latent_dim()
        || *output_a_in != cfg.output_group_input_dim()
        || *output_b_in != cfg.output_latent_dim()
        || *output_b_out != cfg.hidden_size
        || !cfg.output_group_input_dim().is_multiple_of(128)
        || !cfg.o_lora_rank.is_multiple_of(16)
    {
        return Err(Error::Model(format!(
            "SM121 MLA output shape mismatch at layer {}: output_a=[{output_a_out},{output_a_in}] output_b=[{output_b_out},{output_b_in}] groups={} rank={} hidden={}",
            layer.layer, cfg.o_groups, cfg.o_lora_rank, cfg.hidden_size
        )));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn fp8_linear_bundle_requirement(
    operation: KernelOperation,
    linears: [&LinearWeight; 2],
) -> Result<LinearBundleRequirement> {
    let [first, second] = linears;
    let (
        LinearWeightFormat::Fp8E4M3WithE8M0Scale {
            out_features: first_out,
            in_features: first_in,
            block_m: first_block_m,
            block_k: first_block_k,
        },
        LinearWeightFormat::Fp8E4M3WithE8M0Scale {
            out_features: second_out,
            in_features: second_in,
            block_m: second_block_m,
            block_k: second_block_k,
        },
    ) = (&first.format, &second.format)
    else {
        return Err(Error::Model(format!(
            "DeepSeek-V4 {operation:?} requires two FP8 bindings"
        )));
    };
    if first_in != second_in
        || *first_block_m != 128
        || *first_block_k != 128
        || *second_block_m != 128
        || *second_block_k != 128
    {
        return Err(Error::Model(format!(
            "DeepSeek-V4 {operation:?} requires matching FP8 K128 layouts"
        )));
    }
    Ok(LinearBundleRequirement::new(
        operation,
        *first_in,
        [*first_out, *second_out],
        WeightLayout::Fp8E4m3BlockScaled,
    ))
}

#[cfg(feature = "cuda")]
fn fp8_single_linear_requirement(
    operation: KernelOperation,
    linear: &LinearWeight,
) -> Result<LinearBundleRequirement> {
    let LinearWeightFormat::Fp8E4M3WithE8M0Scale {
        out_features,
        in_features,
        block_m: 128,
        block_k: 128,
    } = &linear.format
    else {
        return Err(Error::Model(format!(
            "DeepSeek-V4 {operation:?} requires one FP8 K128 binding"
        )));
    };
    Ok(LinearBundleRequirement::new(
        operation,
        *in_features,
        [*out_features],
        WeightLayout::Fp8E4m3BlockScaled,
    ))
}

#[cfg(feature = "cuda")]
fn bf16_linear_bundle_requirement(
    operation: KernelOperation,
    linears: [&LinearWeight; 2],
) -> Result<LinearBundleRequirement> {
    let [first, second] = linears;
    let (
        LinearWeightFormat::Bf16 {
            out_features: first_out,
            in_features: first_in,
        },
        LinearWeightFormat::Bf16 {
            out_features: second_out,
            in_features: second_in,
        },
    ) = (&first.format, &second.format)
    else {
        return Err(Error::Model(format!(
            "DeepSeek-V4 {operation:?} requires two BF16 bindings"
        )));
    };
    if first_in != second_in {
        return Err(Error::Model(format!(
            "DeepSeek-V4 {operation:?} input mismatch: first={first_in} second={second_in}"
        )));
    }
    Ok(LinearBundleRequirement::new(
        operation,
        *first_in,
        [*first_out, *second_out],
        WeightLayout::Bf16RowMajor,
    ))
}

fn validate_mtp_attachment(
    model: &DeepSeekV4Checkpoint,
    mtp: Option<&DeepSeekV4MtpModel>,
) -> Result<()> {
    let declares_attachment = model.config.num_mtp_layers > 0
        || model.config.dspark_block_size > 1
        || !model.config.dspark_target_layer_ids.is_empty()
        || model.config.dspark_markov_rank.is_some();
    if !declares_attachment {
        return Ok(());
    }

    let mtp = mtp.ok_or_else(|| {
        Error::Model(
            "DeepSeek-V4 config declares a DSpark attachment but no MTP tensors were found".into(),
        )
    })?;
    let _protocol = mtp.protocol()?;
    if mtp.config.block_size == 0 {
        return Err(Error::Model(
            "DeepSeek-V4 DSpark block size must be greater than zero".into(),
        ));
    }
    let noise_token_id = mtp.config.noise_token_id.ok_or_else(|| {
        Error::Model("DeepSeek-V4 DSpark attachment is missing its noise token id".into())
    })?;
    if noise_token_id as usize >= model.config.vocab_size {
        return Err(Error::Model(format!(
            "DeepSeek-V4 DSpark noise token {noise_token_id} exceeds vocabulary {}",
            model.config.vocab_size
        )));
    }
    if mtp.config.target_layer_ids.is_empty() {
        return Err(Error::Model(
            "DeepSeek-V4 DSpark attachment requires target hidden-state layers".into(),
        ));
    }
    for &target_layer in &mtp.config.target_layer_ids {
        if target_layer >= model.config.num_layers {
            return Err(Error::Model(format!(
                "DeepSeek-V4 DSpark target layer {target_layer} exceeds target layer count {}",
                model.config.num_layers
            )));
        }
    }
    if mtp
        .config
        .target_layer_ids
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(Error::Model(
            "DeepSeek-V4 DSpark target layers must be strictly increasing".into(),
        ));
    }
    if mtp.layers.len() != model.config.num_mtp_layers {
        return Err(Error::Model(format!(
            "DeepSeek-V4 prepared MTP stage count {} does not match config {}",
            mtp.layers.len(),
            model.config.num_mtp_layers
        )));
    }
    if mtp.prediction_heads.is_none() {
        return Err(Error::Model(
            "DeepSeek-V4 DSpark attachment is missing prediction heads".into(),
        ));
    }
    for (stage, layer) in mtp.layers.iter().enumerate() {
        let expected_execution_layer = model
            .config
            .num_layers
            .checked_add(stage)
            .ok_or_else(|| Error::Model("DeepSeek-V4 MTP execution layer overflow".into()))?;
        if layer.mtp_index != stage || layer.execution_layer != expected_execution_layer {
            return Err(Error::Model(format!(
                "DeepSeek-V4 MTP stage {stage} has checkpoint index {} and execution layer {}, expected execution layer {expected_execution_layer}",
                layer.mtp_index, layer.execution_layer
            )));
        }
        let is_stage_zero = stage == 0;
        if layer.main_proj.is_some() != is_stage_zero || layer.main_norm.is_some() != is_stage_zero
        {
            return Err(Error::Model(format!(
                "DeepSeek-V4 MTP stage {stage} has an invalid stage-zero projection contract"
            )));
        }
    }
    Ok(())
}

fn validate_options(model: &DeepSeekV4Checkpoint, options: DeepSeekV4PrepareOptions) -> Result<()> {
    if options.max_layers > model.config.num_layers {
        return Err(Error::Model(format!(
            "DeepSeek-V4 prepared plan max_layers {} exceeds model layers {}",
            options.max_layers, model.config.num_layers
        )));
    }

    if options.output_head_chunk_rows == 0 {
        return Err(Error::Model(
            "DeepSeek-V4 prepared plan output_head_chunk_rows must be > 0".into(),
        ));
    }
    if options.expert_reader_max_tensor_bytes == 0 {
        return Err(Error::Model(
            "DeepSeek-V4 prepared plan expert_reader_max_tensor_bytes must be > 0".into(),
        ));
    }
    if model.config.num_routed_experts == 0 || model.config.num_experts_per_tok == 0 {
        return Err(Error::Model(
            "DeepSeek-V4 prepared plan requires a non-empty routed-expert catalog".into(),
        ));
    }
    if model.config.num_experts_per_tok > model.config.num_routed_experts {
        return Err(Error::Model(format!(
            "DeepSeek-V4 experts per token {} exceed routed experts {}",
            model.config.num_experts_per_tok, model.config.num_routed_experts
        )));
    }
    Ok(())
}

fn publish_prepared<R, O>(
    prepared: Result<(R, PreparedExecutable<O>)>,
) -> Result<PreparedModel<R, O>> {
    publish_prepared_with_generation(&NEXT_PREPARED_GENERATION, prepared)
}

fn publish_prepared_with_generation<R, O>(
    generations: &AtomicU64,
    prepared: Result<(R, PreparedExecutable<O>)>,
) -> Result<PreparedModel<R, O>> {
    let (resources, executable) = prepared?;
    let generation = generations.fetch_add(1, Ordering::Relaxed);
    Ok(PreparedModel::new(generation, resources, executable))
}

fn parse_env_bool(name: &str, value: Option<String>, default: bool) -> Result<bool> {
    let Some(value) = value else {
        return Ok(default);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Ok(true),
        "0" | "false" | "off" | "no" => Ok(false),
        _ => Err(Error::Model(format!(
            "DeepSeek-V4 execution policy {name} must be one of 1/0, true/false, on/off, or yes/no; got {value:?}"
        ))),
    }
}

fn parse_env_usize(name: &str, value: Option<String>, default: usize) -> Result<usize> {
    let Some(value) = value else {
        return Ok(default);
    };
    value.trim().parse::<usize>().map_err(|_| {
        Error::Model(format!(
            "DeepSeek-V4 execution policy {name} must be a non-negative integer; got {value:?}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use ferrule_common::MaterializedResourceKind;

    use super::*;
    use crate::checkpoint::{HfRoutedExpertTensorInfo, HfSafetensorsTensorInfo};
    use crate::execution::ModelExecutionBackend;
    use crate::models::deepseek_v4::operators::DeepSeekV4OperatorContext;
    use crate::semantic::{RoutedExpertMatrix, RoutedExpertTensorPart, RoutedExpertTensorRef};
    use crate::support::tensor_role_for_class;
    use crate::tensor_policy::HfTensorPolicy;

    #[test]
    fn static_executable_classifies_exact_dense_stage_bundles() {
        let dir = unique_temp_dir("ferrule-dsv4-static-executable");
        std::fs::create_dir_all(&dir).unwrap();
        let shard = "model.safetensors";
        let shard_path = dir.join(shard);
        let file = std::fs::File::create(&shard_path).unwrap();
        file.set_len(4096).unwrap();

        let names = [
            "embed.weight",
            "layers.0.attn.wq_a.weight",
            "layers.0.attn_norm.weight",
            "layers.0.hc_attn_fn",
            "layers.0.ffn.gate.weight",
            "layers.0.ffn.shared_experts.w1.weight",
            "layers.0.ffn_norm.weight",
            "layers.0.hc_ffn_scale",
            "layers.0.ffn.experts.3.w1.weight",
            "norm.weight",
            "lm_head.weight",
            "hc_head_base",
        ];
        let policy = HfTensorPolicy::for_family(crate::ModelFamily::DeepSeekV4);
        let tensors = names
            .into_iter()
            .enumerate()
            .map(|(index, name)| {
                let class = policy.classify_name(name);
                HfSafetensorsTensorInfo {
                    name: name.into(),
                    shard: shard.into(),
                    dtype: "BF16".into(),
                    shape: vec![2, 2],
                    data_offset: (index * 8) as u64,
                    file_offset: (index * 8) as u64,
                    byte_size: 8,
                    role: tensor_role_for_class(&class),
                    class,
                }
            })
            .collect::<Vec<_>>();
        let inventory = HfSafetensorsInventory {
            family: crate::ModelFamily::DeepSeekV4,
            total_size: Some(4096),
            shard_count: 1,
            tensor_count: tensors.len(),
            tensors,
            dtype_counts: Vec::new(),
            class_counts: Vec::new(),
            role_counts: Vec::new(),
            shard_summaries: Vec::new(),
            index_only_tensors: Vec::new(),
            header_only_tensors: Vec::new(),
        };

        let executable = prepare_executable_from_inventory(&dir, &inventory, 1).unwrap();
        assert_eq!(
            executable
                .stages()
                .iter()
                .map(|stage| *stage.operation())
                .collect::<Vec<_>>(),
            vec![
                TransformerStage::Embed,
                TransformerStage::Attention { layer: 0 },
                TransformerStage::Router { layer: 0 },
                TransformerStage::FeedForward { layer: 0 },
                TransformerStage::Output,
            ]
        );
        assert_eq!(executable.resources().len(), executable.stages().len());
        assert!(executable.resources().iter().all(|manifest| {
            manifest.resource().kind() == MaterializedResourceKind::Parameter
                && manifest.source().unwrap().generation().get() != 0
                && manifest
                    .checkpoint_bundle_source()
                    .is_some_and(|source| source.validate_source_identity())
        }));

        let stage_names = |stage_index: usize| {
            let resource = executable.stages()[stage_index].resources()[0].resource();
            executable
                .resource(resource)
                .unwrap()
                .checkpoint_tensors()
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>()
        };
        assert_eq!(stage_names(0), vec!["embed.weight"]);
        assert_eq!(
            stage_names(1),
            vec![
                "layers.0.attn_norm.weight",
                "layers.0.attn.wq_a.weight",
                "layers.0.hc_attn_fn",
            ]
        );
        assert_eq!(stage_names(2), vec!["layers.0.ffn.gate.weight"]);
        assert_eq!(
            stage_names(3),
            vec![
                "layers.0.ffn_norm.weight",
                "layers.0.ffn.shared_experts.w1.weight",
                "layers.0.hc_ffn_scale",
            ]
        );
        assert_eq!(
            stage_names(4),
            vec!["norm.weight", "lm_head.weight", "hc_head_base"]
        );
        assert!(executable.resources().iter().all(|manifest| {
            manifest
                .checkpoint_tensors()
                .iter()
                .all(|tensor| !tensor.name.contains(".ffn.experts."))
        }));
        assert!(executable.resources().iter().all(|manifest| {
            manifest.checkpoint_tensors().windows(2).all(|pair| {
                (pair[0].role.clone(), pair[0].name.as_str())
                    <= (pair[1].role.clone(), pair[1].name.as_str())
            })
        }));

        let expert = crate::moe::streaming::ExpertId::new(0, 3);
        let expert_catalog = Arc::new(
            ExpertSourceCatalog::from_hf_routed_expert_tensor_sets(
                &dir,
                [HfRoutedExpertTensorInfo {
                    descriptor: RoutedExpertTensorRef {
                        layer: expert.layer,
                        expert: expert.expert,
                        matrix: RoutedExpertMatrix::Gate,
                        part: RoutedExpertTensorPart::Weight,
                    },
                    name: "layers.0.ffn.experts.3.w1.weight".into(),
                    shard: shard.into(),
                    dtype: "BF16".into(),
                    shape: vec![2, 2],
                    data_offset: 64,
                    file_offset: 64,
                    byte_size: 8,
                }],
            )
            .unwrap(),
        );
        let expert_source = expert_catalog.require_resource_source(expert).unwrap();
        let prepared_experts = DeepSeekV4PreparedLayerExperts::new(
            Arc::clone(&expert_catalog),
            ExpertStreamingPolicy {
                gpu_slots_per_layer: 1,
                prefetch_per_layer: 0,
                preserve_source_encoding: true,
                allow_cpu_staging: false,
                allow_remote_sources: false,
            },
        );
        let sources = prepare_materialization_sources(
            &executable,
            std::slice::from_ref(&prepared_experts),
            None,
            &[],
        )
        .unwrap();
        assert_eq!(sources.len(), executable.resources().len() + 1);

        for manifest in executable.resources() {
            let entry = sources
                .get(manifest.resource(), manifest.source().unwrap())
                .unwrap();
            assert_eq!(entry.source(), manifest.source().unwrap());
            match entry.descriptor() {
                DeepSeekV4InstallDescriptor::StaticTensorBundle(descriptor) => {
                    assert_eq!(descriptor.as_ref(), manifest);
                }
                DeepSeekV4InstallDescriptor::RoutedExpert(_) => {
                    panic!("static resource resolved to routed-expert install semantics")
                }
            }
            assert_eq!(
                entry.read_plan().extents().len(),
                manifest.checkpoint_tensors().len()
            );
            for (extent, tensor) in entry
                .read_plan()
                .extents()
                .iter()
                .zip(manifest.checkpoint_tensors())
            {
                assert_eq!(extent.path(), tensor.path);
                assert_eq!(extent.offset(), tensor.offset);
                assert_eq!(extent.bytes(), tensor.bytes);
            }
        }

        let routed_resource = ferrule_common::MaterializedResourceId::routed_expert(
            ferrule_common::LayerId::new(0),
            ferrule_common::ExpertId::new(3),
        );
        let routed_entry = sources.get(routed_resource, expert_source).unwrap();
        assert_eq!(routed_entry.source(), expert_source);
        assert_eq!(
            routed_entry.descriptor(),
            &DeepSeekV4InstallDescriptor::RoutedExpert(
                expert_catalog.source(expert).unwrap().clone()
            )
        );
        assert!(matches!(
            prepare_materialization_sources(
                &executable,
                &[prepared_experts.clone(), prepared_experts],
                None,
                &[],
            ),
            Err(Error::Model(message)) if message.contains("duplicate exact materialization source")
        ));

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn prepared_layer_experts_retain_catalog_identity_and_resolved_capacities() {
        let catalog = Arc::new(ExpertSourceCatalog::from_sources([(
            crate::moe::streaming::ExpertId::new(2, 7),
            crate::moe::streaming::ExpertLoadSource::LocalShard {
                path: "experts.safetensors".into(),
                offset: 64,
                bytes: 32,
            },
        )]));
        let policy = ExpertStreamingPolicy {
            gpu_slots_per_layer: 6,
            prefetch_per_layer: 2,
            preserve_source_encoding: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        };
        let prepared = DeepSeekV4PreparedLayerExperts::new(Arc::clone(&catalog), policy.clone());

        assert!(Arc::ptr_eq(prepared.source_catalog(), &catalog));
        assert_eq!(prepared.streaming_policy(), &policy);
        assert_eq!(prepared.resident_capacity(), 6);
        assert_eq!(prepared.prefetch_capacity(), 2);
    }

    #[test]
    fn dsv4_kv_schema_publishes_all_physical_planes() {
        let schema = DeepSeekV4KvLayoutSchema {
            layer_count: 2,
            window_size: 128,
            head_dim: 64,
            compress_ratios: vec![4, 2].into_boxed_slice(),
            planes: vec![
                KvPlaneDescriptor {
                    name: "window_latent_kv",
                    elements_per_token: 64,
                    layer_count: 2,
                },
                KvPlaneDescriptor {
                    name: "compressed_main_kv",
                    elements_per_token: 64,
                    layer_count: 2,
                },
                KvPlaneDescriptor {
                    name: "indexer_kv",
                    elements_per_token: 32,
                    layer_count: 2,
                },
                KvPlaneDescriptor {
                    name: "compressor_metadata",
                    elements_per_token: 1024,
                    layer_count: 2,
                },
                KvPlaneDescriptor {
                    name: "indexer_metadata",
                    elements_per_token: 512,
                    layer_count: 2,
                },
            ]
            .into_boxed_slice(),
            page_size: 16,
            max_sequence_len: u32::MAX as usize,
        };
        assert_eq!(schema.planes().len(), 5);
        assert_eq!(schema.page_size(), 16);
        assert_eq!(schema.cuda_f32_data_page_bytes().unwrap(), 20_480);
        assert_eq!(schema.pages_for_tokens(4097), 257);
        assert_eq!(schema.planes()[2].name, "indexer_kv");
    }

    #[test]
    fn failed_preparation_stage_does_not_publish_a_generation() {
        let generations = AtomicU64::new(41);
        let failed = publish_prepared_with_generation::<(), TransformerStage>(
            &generations,
            Err(Error::Model("bind failed".into())),
        );
        assert!(failed.is_err());
        assert_eq!(generations.load(Ordering::Relaxed), 41);
    }

    #[test]
    fn execution_policy_parses_once_with_documented_defaults() {
        let values = BTreeMap::from([
            ("FERRULE_DSV4_PREFILL_PROGRESS", "yes"),
            ("FERRULE_DSV4_LOOKAHEAD_PREFETCH", "0"),
            ("FERRULE_DSV4_HASH_PREFETCH_WINDOW", "true"),
            ("FERRULE_MANAGED_EXPERTS", "false"),
            ("FERRULE_DSV4_EXPERT_UPLOAD_INFLIGHT", "7"),
            ("FERRULE_DSV4_PROFILE", "0"),
            ("FERRULE_DSV4_PROFILE_SYNC", "1"),
        ]);
        let policy = DeepSeekV4ExecutionPolicy::resolve_with(|name| {
            Ok(values.get(name).map(ToString::to_string))
        })
        .unwrap();

        assert!(policy.prefill_progress());
        assert!(!policy.lookahead_prefetch());
        assert!(policy.decode_lookahead_prefetch());
        assert!(policy.hash_decode_prefetch_window());
        assert!(!policy.prefill_hash_lookahead());
        assert!(!policy.managed_experts());
        assert_eq!(policy.expert_upload_inflight(), 7);
        assert!(policy.profile_enabled());
        assert!(policy.profile_sync());
    }

    #[test]
    fn execution_policy_defaults_to_profile_off() {
        let policy = DeepSeekV4ExecutionPolicy::resolve_with(|_| Ok(None)).unwrap();
        assert!(!policy.profile_enabled());
        assert!(!policy.profile_sync());

        let memory = DeepSeekV4PrepareOptions::default().expert_memory_policy;
        assert_eq!(memory.host_staged.max_entries, 256);
        assert_eq!(memory.host_staged.max_bytes, u64::MAX);
        assert_eq!(memory.pinned_host.max_entries, 64);
        assert_eq!(memory.pinned_host.max_bytes, u64::MAX);
    }

    #[test]
    fn execution_policy_enables_profile_without_sync() {
        let policy = DeepSeekV4ExecutionPolicy::resolve_with(|name| {
            Ok((name == "FERRULE_DSV4_PROFILE").then(|| "true".to_string()))
        })
        .unwrap();
        assert!(policy.profile_enabled());
        assert!(!policy.profile_sync());
    }

    #[test]
    fn profile_gate_keeps_stats_empty_off_and_records_on() {
        let mut off = DeepSeekV4OperatorContext::new(
            ModelExecutionBackend::Cpu,
            &DeepSeekV4ExecutionPolicy::default(),
            ExpertMemoryPolicy::default(),
        )
        .unwrap();
        assert!(off.profile_start().is_none());
        off.record_layer_prefill(3, 7, 11);
        off.record_attention_call(3, 7);
        assert!(off.layer_profile_stats().is_empty());
        assert!(off.attention_profile_stats().is_empty());

        let policy = DeepSeekV4ExecutionPolicy {
            profile: true,
            ..DeepSeekV4ExecutionPolicy::default()
        };
        let mut on = DeepSeekV4OperatorContext::new(
            ModelExecutionBackend::Cpu,
            &policy,
            ExpertMemoryPolicy::default(),
        )
        .unwrap();
        let start = on.profile_start();
        assert!(start.is_some());
        let elapsed_us = on.finish_profile_stage(start).unwrap().unwrap();
        on.record_layer_prefill(3, 7, elapsed_us);
        on.record_attention_call(3, 7);
        assert_eq!(on.layer_profile_stats()[0].prefill_calls, 1);
        assert_eq!(on.attention_profile_stats()[0].calls, 1);
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn execution_policy_rejects_invalid_values() {
        let error = DeepSeekV4ExecutionPolicy::resolve_with(|name| {
            Ok((name == "FERRULE_DSV4_LOOKAHEAD_PREFETCH").then(|| "sometimes".to_string()))
        })
        .unwrap_err();
        assert!(matches!(error, Error::Model(_)));

        let error = DeepSeekV4ExecutionPolicy::resolve_with(|name| {
            Ok((name == "FERRULE_DSV4_EXPERT_UPLOAD_INFLIGHT").then(|| "many".to_string()))
        })
        .unwrap_err();
        assert!(matches!(error, Error::Model(_)));
    }
}
