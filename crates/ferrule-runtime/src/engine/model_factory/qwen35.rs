//! Text-only hybrid composition on the ordinary resident owner/transaction path.
use super::super::composition::ContextCapacityProfile;
use super::*;
#[cfg(feature = "cuda")]
#[path = "qwen35_ep.rs"]
mod ep;
use ferrule_model::decoder::{
    GenericDecoderOptions, GenericDecoderRunner, HybridCpuDecoder, HybridStateSchema,
};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::models::qwen35::Qwen35Adapter;

use ferrule_model::{TokenizerHandle, transformer::BoundDecoderResources};

impl ResidentModelBuildPlan {
    /// Metadata-only effective context limits. Shape-dependent memory stays
    /// unknown until the owner resolves KV and model memory; legacy requested,
    /// effective and resolution reports are intentionally unchanged.
    pub fn context_capacity_profile(&self) -> Result<ContextCapacityProfile> {
        ContextCapacityProfile::from_configs(
            self.request.scheduler_config,
            self.request.driver_config,
        )
    }
}

/// Only MoE has compressed expert cache policy; dense Qwen3.5 cannot carry it.
#[derive(Debug, Clone)]
pub(super) struct Qwen35MoeOptions {
    device_capacity: Option<Qwen35MoeCapacityLimits>,
    pub(super) host_cache: ferrule_model::transformer::host_experts::HostExpertCacheOptions,
}

/// Metadata policy only. Capacity and host startup ledgers are still derived by
/// the existing planners, and physical admission remains on the build owner.
#[derive(Debug, Clone)]
pub(super) enum Qwen35ResolvedAdapter {
    Dense,
    Moe {
        options: Qwen35MoeOptions,
        expert_placement: Option<Qwen35ExpertPlacement>,
    },
}

impl Qwen35ResolvedAdapter {
    pub(super) fn moe(&self) -> Option<&Qwen35MoeOptions> {
        match self {
            Self::Dense => None,
            Self::Moe { options, .. } => Some(options),
        }
    }

    pub(super) fn expert_placement(&self) -> Option<&Qwen35ExpertPlacement> {
        match self {
            Self::Dense => None,
            Self::Moe {
                expert_placement, ..
            } => expert_placement.as_ref(),
        }
    }

    fn set_expert_placement(&mut self, placement: Qwen35ExpertPlacement) {
        let Self::Moe {
            expert_placement, ..
        } = self
        else {
            unreachable!("dedicated expert placement requires validated Qwen35Moe");
        };
        *expert_placement = Some(placement);
    }
}

/// Adapt the old flat DTO once, preserving validation order and error sources.
pub(super) fn resolve_family_options(
    family: &ModelFamily,
    options: &ModelFactoryOptions,
) -> Result<ResolvedFamilyOptions> {
    if options.qwen35_moe_capacity.is_some() && *family != ModelFamily::Qwen35Moe {
        return Err(invalid(
            "CUDA expert device cache options require Qwen3.5 MoE FP8",
        ));
    }
    if let Some(limits) = options.qwen35_moe_capacity {
        limits.expert_cache_policy()?;
    }
    if let Some(host) = options.qwen35_host_cache {
        if *family != ModelFamily::Qwen35Moe {
            return Err(invalid("expert prewarm options require Qwen3.5 MoE FP8"));
        }
        host.validate()?;
    }
    Ok(match family {
        ModelFamily::Qwen35 => ResolvedFamilyOptions::Qwen35(Qwen35ResolvedAdapter::Dense),
        ModelFamily::Qwen35Moe => ResolvedFamilyOptions::Qwen35(Qwen35ResolvedAdapter::Moe {
            options: Qwen35MoeOptions {
                device_capacity: options.qwen35_moe_capacity,
                host_cache: options.qwen35_host_cache.unwrap_or_default(),
            },
            expert_placement: None,
        }),
        _ => ResolvedFamilyOptions::Generic,
    })
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidRequest {
        message: message.into(),
    }
}

/// Capability failures retain a typed model source; never fall back to CPU.
pub(super) fn moe_unavailable(selection: BackendSelection) -> Error {
    let (operator, reason) = if selection == BackendSelection::Cpu {
        (
            "qwen35_moe_cpu",
            "CPU MoE numeric FP8 execution is not implemented",
        )
    } else {
        (
            "qwen35_moe_cuda",
            "Qwen3.5-35B requires CUDA compiled with the cuda feature; CPU fallback is unsupported",
        )
    };
    ferrule_common::Error::ModelSource {
        source: Box::new(ferrule_model::transformer::UnsupportedOperator::new(
            operator, reason,
        )),
    }
    .into()
}

fn ep_unsupported(reason: impl Into<String>) -> Error {
    ferrule_common::Error::ModelSource {
        source: Box::new(ferrule_model::transformer::UnsupportedOperator::new(
            "qwen35_moe_expert_parallel",
            reason,
        )),
    }
    .into()
}

/// Preserve admission and secondary shutdown failures as separate typed sources.
/// A cleanup error is not proof of physical retirement.
#[cfg(any(feature = "cuda", test))]
fn finish_owner_admission<T>(
    admission: Result<T>,
    cleanup: impl FnOnce() -> Result<()>,
) -> Result<T> {
    admission
        .map_err(|primary| Error::with_cleanup("Qwen3.5 EP owner admission", primary, cleanup()))
}

/// Explicit GPU-thread placement, separate from tensor and KV rank topology.
/// The root and expert owner zero use separate contexts on devices[0].
/// No placement is inferred from the number of visible GPUs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ExpertPlacement {
    devices: Vec<usize>,
}

impl Qwen35ExpertPlacement {
    pub fn new(expert_parallel: usize, devices: Vec<usize>) -> Result<Self> {
        if !matches!(expert_parallel, 2 | 4 | 8) {
            return Err(ep_unsupported(
                "Qwen3.5 GPU-thread EP requires degree 2, 4 or 8",
            ));
        }
        if devices.len() != expert_parallel
            || devices
                .iter()
                .any(|&ordinal| i32::try_from(ordinal).is_err())
            || devices
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != devices.len()
        {
            return Err(ep_unsupported(format!(
                "Qwen3.5 EP{expert_parallel} requires exactly {expert_parallel} distinct CUDA ordinals; root = devices[0], experts = the same list"
            )));
        }
        Ok(Self { devices })
    }

    pub fn root_device(&self) -> usize {
        self.devices[0]
    }

    pub fn expert_devices(&self) -> &[usize] {
        &self.devices
    }

    pub fn expert_parallel(&self) -> usize {
        self.devices.len()
    }

    pub fn validate_expert_shape(&self, layers: usize, experts_per_layer: usize) -> Result<usize> {
        if layers == 0
            || experts_per_layer == 0
            || !experts_per_layer.is_multiple_of(self.expert_parallel())
        {
            return Err(ep_unsupported(
                "Qwen3.5 EP requires a nonempty expert shape divisible by the owner count",
            ));
        }
        layers
            .checked_mul(experts_per_layer / self.expert_parallel())
            .ok_or_else(|| ep_unsupported("Qwen3.5 owned expert count overflow"))
    }

    fn from_options(options: &PipelineBuildOptions, backend: BackendSelection) -> Result<Self> {
        let p = options.parallelism;
        if backend == BackendSelection::Cpu
            || p.pipeline_parallel != 1
            || p.tensor_parallel != 1
            || p.data_parallel != 1
            || p.sequence_parallel != 1
            || p.context_parallel != 1
            || options.rank_backend != PipelineRankBackend::Thread
            || options.rank_restarts != 0
            || options.rank_timeout != std::time::Duration::from_secs(30)
        {
            return Err(ep_unsupported(
                "Qwen3.5 EP supports CUDA thread experts only; CPU, TP/PP/DP/SP/CP, process, custom rank deadlines and restart/replay are unsupported",
            ));
        }
        #[cfg(unix)]
        if options.process_launch.is_some() {
            return Err(ep_unsupported(
                "Qwen3.5 thread EP does not consume process_launch",
            ));
        }
        Self::new(
            p.expert_parallel,
            options.devices.clone().ok_or_else(|| {
                ep_unsupported(
                    "Qwen3.5 EP requires explicit --devices; no automatic multi-GPU allocation",
                )
            })?,
        )
    }
}

/// Actual external-routing root estimate supplied by the model estimator.
/// `weights_bytes` excludes routed experts and any local routed-cache ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35ExpertRootMemory {
    pub weights_bytes: usize,
    pub state_bytes: usize,
    pub kv_bytes: usize,
    pub workspace_bytes: usize,
}

#[cfg(feature = "cuda")]
impl Qwen35ExpertRootMemory {
    /// Authoritative model estimate for external routing, not the local cache
    /// estimator. Root retains router/shared/static weights and numeric scratch;
    /// all routed weights and owner workspaces are charged separately.
    pub fn for_resources(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        kv_pages: usize,
    ) -> Result<Self> {
        if options.hybrid_cuda_numeric_fp8_precision()
            != Some(ferrule_model::transformer::NumericFp8Precision::F32Tf32x3)
        {
            return Err(ep_unsupported(
                "Qwen3.5 EP root requires numeric FP8 storage with pure F32 TF32x3 execution",
            ));
        }
        let estimate =
            ferrule_model::decoder::HybridCudaMemoryEstimate::for_resources_with_routed_experts(
                resources, options, kv_pages,
            )?;
        let copies = options
            .capabilities()
            .max_sequences
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| ep_unsupported("Qwen3.5 root state copy count overflow"))?;
        let state_bytes = estimate
            .per_sequence_state_bytes
            .checked_mul(copies)
            .ok_or_else(|| ep_unsupported("Qwen3.5 root state budget overflow"))?;
        Ok(Self {
            weights_bytes: estimate.weight_bytes_upper_bound,
            state_bytes,
            kv_bytes: estimate.kv_bytes,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        })
    }
}

/// One admission ledger per physical card, not per logical CUDA context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35PhysicalCardMemory {
    pub ordinal: usize,
    pub root_weights_bytes: usize,
    pub expert_weights_bytes: usize,
    pub state_bytes: usize,
    pub kv_bytes: usize,
    pub workspace_bytes: usize,
    pub contexts: usize,
    pub allocator_margin_bytes: usize,
    pub required_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35PhysicalCardCapacity {
    pub ordinal: usize,
    pub free_bytes: usize,
    pub limit_bytes: usize,
}

/// Immutable pre-allocation proof. This is not a CUDA reservation: callers must
/// sample every selected physical device before starting any owner uploads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ExpertPhysicalPlan {
    cards: Vec<Qwen35PhysicalCardMemory>,
}

impl Qwen35ExpertPhysicalPlan {
    /// `owner_weights_bytes` must come from exact owned compressed bindings,
    /// including scales, not the single-device routed-cache cap or disk size.
    pub fn new(
        placement: &Qwen35ExpertPlacement,
        root: Qwen35ExpertRootMemory,
        owner_weights_bytes: &[usize],
        owner_workspace_bytes: usize,
        allocator_margin_per_context: usize,
    ) -> Result<Self> {
        if owner_weights_bytes.len() != placement.expert_parallel()
            || owner_weights_bytes.contains(&0)
            || root.weights_bytes == 0
            || root.workspace_bytes == 0
            || owner_workspace_bytes == 0
            || allocator_margin_per_context == 0
        {
            return Err(ep_unsupported(
                "Qwen3.5 EP needs complete nonzero actual root/owner estimates and per-context allocator margins",
            ));
        }
        let add = |a: usize, b: usize| {
            a.checked_add(b)
                .ok_or_else(|| ep_unsupported("Qwen3.5 physical-card budget overflow"))
        };
        let mut cards = Vec::with_capacity(placement.expert_parallel());
        for (owner, (&ordinal, &expert_weights_bytes)) in placement
            .devices
            .iter()
            .zip(owner_weights_bytes)
            .enumerate()
        {
            let is_root = owner == 0;
            let contexts = if is_root { 2 } else { 1 };
            let mut card = Qwen35PhysicalCardMemory {
                ordinal,
                root_weights_bytes: if is_root { root.weights_bytes } else { 0 },
                expert_weights_bytes,
                state_bytes: if is_root { root.state_bytes } else { 0 },
                kv_bytes: if is_root { root.kv_bytes } else { 0 },
                workspace_bytes: add(
                    owner_workspace_bytes,
                    if is_root { root.workspace_bytes } else { 0 },
                )?,
                contexts,
                allocator_margin_bytes: if is_root {
                    add(allocator_margin_per_context, allocator_margin_per_context)?
                } else {
                    allocator_margin_per_context
                },
                required_bytes: 0,
            };
            card.required_bytes = [
                card.root_weights_bytes,
                card.expert_weights_bytes,
                card.state_bytes,
                card.kv_bytes,
                card.workspace_bytes,
                card.allocator_margin_bytes,
            ]
            .into_iter()
            .try_fold(0, add)?;
            cards.push(card);
        }
        Ok(Self { cards })
    }

    pub fn cards(&self) -> &[Qwen35PhysicalCardMemory] {
        &self.cards
    }

    /// Reject incomplete, duplicate or foreign device samples. Log every card,
    /// including the root+expert0 sum, before returning any capacity failure.
    pub fn check_capacity(&self, capacities: &[Qwen35PhysicalCardCapacity]) -> Result<()> {
        if capacities.len() != self.cards.len()
            || capacities
                .iter()
                .map(|c| c.ordinal)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != capacities.len()
            || self
                .cards
                .iter()
                .any(|card| !capacities.iter().any(|c| c.ordinal == card.ordinal))
        {
            return Err(ep_unsupported(
                "Qwen3.5 EP requires exactly one capacity sample per selected physical card",
            ));
        }
        let mut failure = None;
        for card in &self.cards {
            let capacity = capacities
                .iter()
                .find(|c| c.ordinal == card.ordinal)
                .unwrap();
            tracing::info!(
                ordinal = card.ordinal,
                root_weights_bytes = card.root_weights_bytes,
                expert_weights_bytes = card.expert_weights_bytes,
                state_bytes = card.state_bytes,
                kv_bytes = card.kv_bytes,
                workspace_bytes = card.workspace_bytes,
                contexts = card.contexts,
                allocator_margin_bytes = card.allocator_margin_bytes,
                required_bytes = card.required_bytes,
                free_bytes = capacity.free_bytes,
                limit_bytes = capacity.limit_bytes,
                "Qwen3.5 EP physical-card admission (before uploads)"
            );
            if card.required_bytes > capacity.free_bytes.min(capacity.limit_bytes)
                && failure.is_none()
            {
                failure = Some(ep_unsupported(format!(
                    "Qwen3.5 EP CUDA ordinal {} requires {} bytes across {} contexts including margins; free {}, limit {}",
                    card.ordinal,
                    card.required_bytes,
                    card.contexts,
                    capacity.free_bytes,
                    capacity.limit_bytes
                )));
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

impl ResidentModelPlanner {
    /// Resolve Qwen3.5 EP through the resident registry, never the generic PP/KV
    /// topology. This only prepares metadata; construction remains owner-local.
    pub fn prepare_qwen35_expert_parallel(
        &self,
        config: &AutoConfig,
        backend: BackendSelection,
        chat_template_override: Option<&str>,
        options: ModelFactoryOptions,
        parallel: PipelineBuildOptions,
    ) -> Result<ResidentModelBuildPlan> {
        if config.descriptor().spec.family != ModelFamily::Qwen35Moe {
            return Err(ep_unsupported("Qwen3.5 GPU-thread EP requires Qwen35Moe"));
        }
        let placement = Qwen35ExpertPlacement::from_options(&parallel, backend)?;
        placement.validate_expert_shape(
            config.descriptor().spec.num_layers.unwrap_or(0),
            config.descriptor().spec.moe.num_experts.unwrap_or(0),
        )?;
        if options.qwen35_moe_capacity.is_some() {
            return Err(ep_unsupported(
                "Qwen3.5 EP keeps all owned experts resident; single-device --cuda-expert-device-* cache overrides are unsupported",
            ));
        }
        if options.qwen35_host_cache.is_some_and(|host| {
            host.mode != ferrule_model::transformer::host_experts::ExpertPrewarmMode::Full
        }) {
            return Err(ep_unsupported(
                "Qwen3.5 EP requires one complete shared full host prewarm before owner startup",
            ));
        }
        let mut plan = self.prepare(config, backend, chat_template_override, options)?;
        plan.backend_profile = "cuda-hybrid-numeric-fp8-f32-tf32x3-qwen35-thread-ep";
        plan.requested_parallel = Some(parallel);
        plan.request
            .family_options
            .qwen35_mut()
            .expect("validated Qwen35Moe")
            .set_expert_placement(placement);
        Ok(plan)
    }
}

/// Explicit capacity inputs for metadata admission and single-GPU construction.
/// Cache bytes include resident + pending compressed experts and the shared scratch.
/// Host residency is separately configured by HostExpertCacheOptions; no pinned hotset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35MoeCapacityLimits {
    pub max_experts: usize,
    pub max_bytes: usize,
    pub scratch_bytes: usize,
    /// F32 operator workspace excluding numeric scratch (already in max_bytes).
    pub workspace_bytes: usize,
    pub kv_bytes: usize,
    pub device_bytes: usize,
    /// Separate allowance for allocator segments and library/context overhead.
    pub allocator_margin_bytes: usize,
    pub max_positions: usize,
    pub max_sequences: usize,
    pub max_batch_tokens: usize,
}

impl Default for Qwen35MoeCapacityLimits {
    fn default() -> Self {
        Self {
            max_experts: 1024,
            max_bytes: 4usize << 30,
            scratch_bytes: 64 << 20,
            workspace_bytes: 1 << 30,
            kv_bytes: 1 << 30,
            device_bytes: 16usize << 30,
            allocator_margin_bytes: 512 << 20,
            max_positions: 1024,
            max_sequences: 4,
            max_batch_tokens: 32,
        }
    }
}

/// Full-depth text-only single-copy target, distinct from model admission.
/// This is neither a device reservation nor proof that an artifact can execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35MoeCapacityPlan {
    pub limits: Qwen35MoeCapacityLimits,
    pub resident_f32_bytes: usize,
    pub compressed_projection_bytes: usize,
    /// All routed experts on disk, excluded from the resident weight sum.
    pub routed_expert_storage_bytes: usize,
    pub one_expert_bytes: usize,
    pub rotary_bytes: usize,
    pub per_sequence_state_bytes: usize,
    pub state_bytes: usize,
    pub kv_pages: usize,
    pub kv_bytes: usize,
    pub minimum_workspace_bytes: usize,
    pub minimum_scratch_bytes: usize,
    /// Includes full cache/workspace caps and allocator margin, not all experts.
    pub total_device_bytes: usize,
}

impl Qwen35MoeCapacityPlan {
    pub const PLANNED_BACKEND_PROFILE: &'static str =
        "cuda-hybrid-numeric-fp8-f32-tf32x3-qwen35-35b-a3b";

    /// Uses the exact immutable profile schema; does not read payloads, initialize
    /// CUDA, or change Qwen35Config::supports_execution(). Strict artifact checks
    /// must still run via Qwen35Metadata before any future owner construction.
    pub fn from_config(
        config: &ferrule_model::models::qwen35::Qwen35Config,
        limits: Qwen35MoeCapacityLimits,
    ) -> Result<Self> {
        use ferrule_model::decoder::HybridLayerSchema;
        use ferrule_model::models::qwen35::{
            Qwen35HfNameMapper, Qwen35Profile, Qwen35Recipe, Qwen35TensorPartitionKind,
        };
        use ferrule_model::nn::{ParameterDType, ParameterPart};
        if config.profile() != Qwen35Profile::Moe35BA3Bfp8 {
            return Err(invalid(
                "35B capacity planning requires the exact MoE FP8 profile",
            ));
        }
        let t = config.text();
        if [
            limits.max_experts,
            limits.max_bytes,
            limits.scratch_bytes,
            limits.workspace_bytes,
            limits.kv_bytes,
            limits.device_bytes,
            limits.max_positions,
            limits.max_sequences,
            limits.max_batch_tokens,
        ]
        .contains(&0)
            || limits.max_positions > t.max_position_embeddings
            || limits.max_batch_tokens > 32
            || limits.max_batch_tokens > limits.max_positions
            || limits.max_experts < t.num_experts_per_tok.unwrap()
            || limits.max_experts > t.num_hidden_layers * t.num_experts.unwrap()
        {
            return Err(invalid(
                "35B requires finite nonzero budgets, full-route expert capacity, and token-serial batch <= 32 within context",
            ));
        }
        let add = |a: usize, b: usize| {
            a.checked_add(b)
                .ok_or_else(|| invalid("35B capacity overflow"))
        };
        let mul = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| invalid("35B capacity overflow"))
        };
        let mapper = Qwen35HfNameMapper::new(config);
        let mut resident_f32_bytes = 0;
        let mut compressed_projection_bytes = 0;
        let mut routed_expert_storage_bytes = 0;
        let mut one_expert_bytes = 0;
        let mut minimum_scratch_bytes = 0;
        let mut width = t.vocab_size.max(t.hidden_size);
        for tensor in mapper
            .tensors()
            .filter(|t| t.partition == Qwen35TensorPartitionKind::Text)
        {
            let bytes = usize::try_from(tensor.bytes())
                .map_err(|_| invalid("35B tensor bytes overflow"))?;
            // Names come from the strict profile mapper, not untrusted prefix matching.
            let routed = tensor.external_name.contains(".mlp.experts.");
            if routed {
                routed_expert_storage_bytes = add(routed_expert_storage_bytes, bytes)?;
                if tensor.external_name.contains(".layers.0.mlp.experts.0.") {
                    one_expert_bytes = add(one_expert_bytes, bytes)?;
                }
            } else if tensor.part == ParameterPart::Scale || tensor.dtype == ParameterDType::F8E4M3
            {
                compressed_projection_bytes = add(compressed_projection_bytes, bytes)?;
            } else {
                // Includes untied embedding AND head, router, shared output gate,
                // norms, convolution and every other nonexpert BF16/F32 tensor.
                let elements = tensor.shape.iter().try_fold(1, |n, d| mul(n, *d))?;
                resident_f32_bytes = add(resident_f32_bytes, mul(elements, 4)?)?;
            }
            if tensor.part == ParameterPart::Weight && tensor.shape.len() == 2 {
                width = width.max(tensor.shape[0]).max(tensor.shape[1]);
                // F32 numeric storage bridge needs at least one decoded weight
                // row. Activation/output buffers belong to operator workspace.
                let padded_k = mul(tensor.shape[1].div_ceil(8), 8)?;
                minimum_scratch_bytes = minimum_scratch_bytes.max(mul(padded_k, 4)?);
            }
        }
        let cache_required = add(
            mul(limits.max_experts, one_expert_bytes)?,
            limits.scratch_bytes,
        )?;
        if one_expert_bytes == 0
            || cache_required > limits.max_bytes
            || minimum_scratch_bytes > limits.scratch_bytes
        {
            return Err(invalid(format!(
                "35B bounded cache requires {cache_required} bytes (experts + shared scratch), scratch minimum {minimum_scratch_bytes}; limits={limits:?}"
            )));
        }
        let spec = Qwen35Recipe::spec(config).map_err(|e| invalid(e.to_string()))?;
        let schema = HybridStateSchema::from_spec(&spec, t.num_hidden_layers)?;
        let planes = schema.kv_planes(16, limits.max_positions)?;
        let pages = plan_resident_kv_pages(
            &planes,
            ResidentKvPageAccounting::TransactionCapacity {
                page_bytes: physical_page_bytes(&planes)?,
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
        let kv_bytes = usize::try_from(pages.configured_bytes.unwrap())
            .map_err(|_| invalid("35B KV bytes overflow"))?;
        let mut per_sequence_state_bytes = 0;
        let mut rotary_bytes = 0;
        for layer in schema.layers() {
            match layer {
                HybridLayerSchema::GatedDeltaNet(shape) => {
                    let (channels, conv, recurrent) = shape.sizes()?;
                    width = width.max(channels);
                    per_sequence_state_bytes =
                        add(per_sequence_state_bytes, mul(add(conv, recurrent)?, 4)?)?;
                }
                HybridLayerSchema::FullAttention { .. } => {
                    rotary_bytes = add(
                        rotary_bytes,
                        mul(
                            mul(limits.max_positions, config.semantics().rotary_dimensions)?,
                            4,
                        )?,
                    )?;
                }
            }
        }
        let state_bytes = mul(
            add(mul(limits.max_sequences, 2)?, 1)?,
            per_sequence_state_bytes,
        )?;
        // Conservative F32 operator/workspace envelope, separate from conversion
        // scratch. Model-side allocation accounting must validate this target.
        let minimum_workspace_bytes = mul(
            mul(
                add(
                    mul(width, 32)?,
                    mul(t.num_attention_heads, limits.max_positions)?,
                )?,
                limits.max_batch_tokens,
            )?,
            4,
        )?;
        if minimum_workspace_bytes > limits.workspace_bytes {
            return Err(invalid(format!(
                "35B workspace requires at least {minimum_workspace_bytes} bytes"
            )));
        }
        let total_device_bytes = [
            resident_f32_bytes,
            compressed_projection_bytes,
            rotary_bytes,
            state_bytes,
            kv_bytes,
            limits.workspace_bytes,
            limits.max_bytes,
            limits.allocator_margin_bytes,
        ]
        .into_iter()
        .try_fold(0, add)?;
        if total_device_bytes > limits.device_bytes {
            return Err(invalid(format!(
                "35B capacity requires {total_device_bytes} device bytes, budget {}",
                limits.device_bytes
            )));
        }
        Ok(Self {
            limits,
            resident_f32_bytes,
            compressed_projection_bytes,
            routed_expert_storage_bytes,
            one_expert_bytes,
            rotary_bytes,
            per_sequence_state_bytes,
            state_bytes,
            kv_pages: pages.configured_pages,
            kv_bytes,
            minimum_workspace_bytes,
            minimum_scratch_bytes,
            total_device_bytes,
        })
    }

    /// Schema target only, not the model owner's authoritative CUDA estimate.
    /// Prefill is supplied separately because the legacy capacity DTO only carries
    /// the packed-token bound. Cache and scratch stay charged exactly once.
    pub fn context_capacity_profile(
        &self,
        prefill_chunk_size: usize,
    ) -> Result<ContextCapacityProfile> {
        let logical_pages = self
            .limits
            .max_positions
            .div_ceil(16)
            .checked_mul(self.limits.max_sequences)
            .filter(|pages| *pages > 0)
            .ok_or_else(|| invalid("35B context profile page overflow"))?;
        if self.kv_pages == 0
            || logical_pages.checked_mul(2) != Some(self.kv_pages)
            || self.kv_bytes == 0
            || !self.kv_bytes.is_multiple_of(self.kv_pages)
        {
            return Err(invalid(
                "35B context profile differs from planned KV geometry",
            ));
        }
        let pages = super::super::ResidentKvPagePlan {
            full_capacity_pages: logical_pages,
            configured_pages: self.kv_pages,
            page_bytes: Some(
                u64::try_from(self.kv_bytes / self.kv_pages)
                    .map_err(|_| invalid("35B context page bytes overflow"))?,
            ),
            configured_bytes: Some(
                u64::try_from(self.kv_bytes)
                    .map_err(|_| invalid("35B context KV bytes overflow"))?,
            ),
        };
        pages
            .context_capacity_profile(
                ResidentSchedulerConfig {
                    max_active_sequences: self.limits.max_sequences,
                    max_batch_tokens: self.limits.max_batch_tokens,
                    prefill_chunk_size,
                    ..Default::default()
                },
                ResidentTopKDriverConfig {
                    ctx_size: self.limits.max_positions,
                    ..Default::default()
                },
            )?
            .with_memory_requirements(
                self.state_bytes,
                self.limits.workspace_bytes,
                self.total_device_bytes,
            )
    }

    /// A pure check; the future model owner must repeat it against actual free
    /// memory before allocation and enforce the individual caps during execution.
    pub fn check_available_device_bytes(&self, free_bytes: usize) -> Result<()> {
        if self.total_device_bytes > free_bytes {
            return Err(invalid(format!(
                "35B requires {} device bytes including margin, only {free_bytes} free",
                self.total_device_bytes
            )));
        }
        Ok(())
    }
}

/// Metadata-only bridge to the real model numeric options and memory estimator.
/// Payload loading remains on the consuming factory owner, not this admission.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct Qwen35MoeCudaAdmission {
    target: Qwen35MoeCapacityPlan,
    options: GenericDecoderOptions,
    estimate: ferrule_model::decoder::HybridCudaMemoryEstimate,
    budget: ferrule_model::decoder::HybridCudaMemoryBudget,
    total_device_bytes: usize,
}

#[cfg(feature = "cuda")]
impl Qwen35MoeCudaAdmission {
    /// Strict headers and source identity only: no payload, tokenizer, device or
    /// runner is created. The returned admission is not a serving capability.
    pub fn open_hf(
        model_path: &std::path::Path,
        limits: Qwen35MoeCapacityLimits,
        max_tensor_bytes: u64,
    ) -> Result<Self> {
        let (metadata, resources) = Qwen35Adapter::bind_hf_metadata(model_path)?;
        let target = Qwen35MoeCapacityPlan::from_config(metadata.config(), limits)?;
        Self::for_resources(target, &resources, max_tensor_bytes)
    }

    fn for_resources(
        target: Qwen35MoeCapacityPlan,
        resources: &BoundDecoderResources,
        max_tensor_bytes: u64,
    ) -> Result<Self> {
        use ferrule_model::decoder::{HybridCudaMemoryBudget, HybridCudaMemoryEstimate};
        use ferrule_model::nn::ParameterResidency;
        let limits = target.limits;
        // Both the untouched embedding and head must pass the source reader's
        // bound. This does not authorize any persistent decoded expert cache.
        let largest = resources
            .state_dict()
            .parameters()
            .iter()
            .try_fold(0u64, |n, p| {
                let bytes = p
                    .weight()
                    .slice()
                    .bytes
                    .checked_add(p.scale().map_or(0, |scale| scale.slice().bytes))
                    .ok_or_else(|| invalid("35B parameter read size overflow"))?;
                Ok::<_, Error>(n.max(bytes))
            })?;
        if max_tensor_bytes < largest {
            return Err(invalid(format!(
                "35B max tensor read budget {max_tensor_bytes} is smaller than required {largest} bytes"
            )));
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
            ExecutionPrecisionPolicy::f32(),
        )?;
        let ferrule_model::transformer::ExpertCachePolicy::Bounded(cache) =
            limits.expert_cache_policy()?
        else {
            unreachable!("35B never uses KeepAll")
        };
        let options = options.with_hybrid_cuda_numeric_fp8_precision(
            cache,
            limits.scratch_bytes,
            ferrule_model::transformer::NumericFp8Precision::F32Tf32x3,
        )?;
        let estimate =
            HybridCudaMemoryEstimate::for_resources(resources, &options, target.kv_pages)?;
        if estimate.per_sequence_state_bytes != target.per_sequence_state_bytes
            || estimate.kv_bytes != target.kv_bytes
        {
            return Err(invalid("35B model/runtime state or KV geometry mismatch"));
        }
        // The model includes numeric scratch in workspace, but its reservation
        // already belongs to max_bytes. Do not charge a second cache or scratch.
        let workspace_cap = limits
            .workspace_bytes
            .checked_add(limits.scratch_bytes)
            .ok_or_else(|| invalid("35B workspace plus numeric scratch overflow"))?;
        if estimate.workspace_bytes_upper_bound > workspace_cap {
            return Err(invalid(format!(
                "35B model workspace {} exceeds workspace plus scratch cap {workspace_cap}",
                estimate.workspace_bytes_upper_bound
            )));
        }
        // The actual model estimator is authoritative, including its current
        // binding/storage accounting. Never substitute the schema target.
        let budget = HybridCudaMemoryBudget {
            kv_pages: target.kv_pages,
            state_bytes: target.state_bytes,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: workspace_cap,
        };
        let total_device_bytes = [
            budget.weight_bytes,
            budget.state_bytes,
            budget.workspace_bytes,
            estimate.kv_bytes,
            limits.allocator_margin_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |n, bytes| {
            n.checked_add(bytes)
                .ok_or_else(|| invalid("35B model admission total overflow"))
        })?;
        if total_device_bytes > limits.device_bytes {
            return Err(invalid(format!(
                "35B model admission requires {total_device_bytes} bytes including margin, device budget {} (schema target {} is not authoritative)",
                limits.device_bytes, target.total_device_bytes
            )));
        }
        // Report the directory size separately; it is not a residency count.
        let routed_parameters = resources
            .state_dict()
            .parameters()
            .iter()
            .filter(|p| matches!(p.residency(), ParameterResidency::Expert { .. }))
            .count();
        tracing::info!(
            profile = Qwen35MoeCapacityPlan::PLANNED_BACKEND_PROFILE,
            ?cache,
            numeric_scratch_bytes = limits.scratch_bytes,
            ?estimate,
            ?budget,
            total_device_bytes,
            routed_parameters,
            schema_target_bytes = target.total_device_bytes,
            "Qwen3.5-35B compressed FP8 / F32 TF32x3 bounded model admission"
        );
        Ok(Self {
            target,
            options,
            estimate,
            budget,
            total_device_bytes,
        })
    }

    /// Owner-estimator requirements, not the lighter schema target or free VRAM.
    pub fn context_capacity_profile(
        &self,
        prefill_chunk_size: usize,
    ) -> Result<ContextCapacityProfile> {
        self.target
            .context_capacity_profile(prefill_chunk_size)?
            .with_memory_requirements(
                self.budget.state_bytes,
                self.budget.workspace_bytes,
                self.total_device_bytes,
            )
    }

    pub fn target(&self) -> &Qwen35MoeCapacityPlan {
        &self.target
    }
    pub fn runner_options(&self) -> &GenericDecoderOptions {
        &self.options
    }
    pub fn model_estimate(&self) -> ferrule_model::decoder::HybridCudaMemoryEstimate {
        self.estimate
    }
    pub fn model_budget(&self) -> ferrule_model::decoder::HybridCudaMemoryBudget {
        self.budget
    }
    pub fn total_device_bytes(&self) -> usize {
        self.total_device_bytes
    }

    /// Compare against memory_info().0 on the actual owner after creating its
    /// CUDA context and BEFORE allocating weights/KV/state. The runner must
    /// still perform its own admission; this check reserves no memory.
    pub fn check_available_device_bytes(&self, free_bytes: usize) -> Result<()> {
        if self.total_device_bytes > free_bytes {
            return Err(invalid(format!(
                "35B model admission requires {} bytes including margin, only {free_bytes} free",
                self.total_device_bytes
            )));
        }
        Ok(())
    }
}

impl Qwen35MoeCapacityLimits {
    /// The explicit GPU policy, not the unrelated pageable/pinned host options.
    /// The model options API accepts only these bounded limits, never KeepAll.
    pub fn expert_cache_policy(&self) -> Result<ferrule_model::transformer::ExpertCachePolicy> {
        use ferrule_model::transformer::{ExpertCacheLimits, ExpertCachePolicy};
        if self.max_experts == 0 || self.scratch_bytes == 0 || self.max_bytes <= self.scratch_bytes
        {
            return Err(invalid(
                "35B requires nonzero bounded expert cache with room beyond scratch",
            ));
        }
        Ok(ExpertCachePolicy::Bounded(ExpertCacheLimits {
            max_experts: self.max_experts,
            max_bytes: self.max_bytes,
        }))
    }
}

pub(super) fn resolve(
    descriptor: &ModelDescriptor,
    selection: BackendSelection,
) -> Result<ResolvedModelBackend> {
    if descriptor.spec.weight_source != WeightSource::Safetensors {
        return Err(invalid("Qwen3.5 requires Hugging Face safetensors"));
    }
    if descriptor.spec.family == ModelFamily::Qwen35Moe {
        if selection == BackendSelection::Cpu || !cfg!(feature = "cuda") {
            return Err(moe_unavailable(selection));
        }
        return Ok(ResolvedModelBackend {
            family: ModelFamily::Qwen35Moe,
            model_name: "qwen3.5-35b-a3b-fp8",
            backend: ModelExecutionBackend::Cuda,
            backend_profile: Qwen35MoeCapacityPlan::PLANNED_BACKEND_PROFILE,
            default_chat_template: ChatTemplate::Qwen35,
        });
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
            "Qwen3.5 does not implement host/pinned expert_cache or moe_hotset_experts overrides",
        ));
    }
    if options.driver_config.enable_native_proposals {
        return Err(invalid(
            "Qwen3.5 speculative/native proposals are unsupported; disable enable_native_proposals",
        ));
    }
    if descriptor.spec.family == ModelFamily::Qwen35Moe {
        if options.expert_reader_max_tensor_mebibytes != 64
            || options.output_head_chunk_rows != 4096
        {
            return Err(invalid(
                "35B does not consume separate expert-reader or output-head chunk overrides",
            ));
        }
        if options.driver_config.ctx_size > 262144 {
            return Err(invalid(
                "35B context exceeds the strict profile limit 262144",
            ));
        }
        if options.max_tensor_mebibytes < 970 {
            return Err(invalid(
                "35B embedding/head need 970 MiB source reads; use --max-tensor-mb 1024",
            ));
        }
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
    let request = request.resolved;
    if request
        .family_options
        .qwen35()
        .and_then(|options| options.expert_placement())
        .is_some()
    {
        #[cfg(feature = "cuda")]
        return ep::build(request);
        #[cfg(not(feature = "cuda"))]
        return Err(moe_unavailable(request.backend.into()));
    }
    if request.family == ModelFamily::Qwen35Moe {
        #[cfg(feature = "cuda")]
        return build_moe(request);
        #[cfg(not(feature = "cuda"))]
        return Err(moe_unavailable(request.backend.into()));
    }
    if request.pipeline.is_some() {
        return Err(invalid("Qwen3.5 hybrid TP/PP is unsupported"));
    }
    // Resources, tokenizer and CUDA state are created on the consuming owner.
    // The family adapter's load-with-backend helper is deliberately CPU-only.
    let (_, resources) = Qwen35Adapter::bind_hf_metadata(&request.model_path)?;
    let tokenizer = TokenizerHandle::load(&request.model_path)?;
    build_bound(request, resources, tokenizer)
}

/// Full-depth single-GPU composition. Limits are explicit; never KeepAll.
pub(super) fn capacity_limits(request: &ResolvedModelRequest) -> Result<Qwen35MoeCapacityLimits> {
    let mut limits = request
        .family_options
        .qwen35()
        .and_then(|options| options.moe())
        .expect("validated Qwen35Moe")
        .device_capacity
        .unwrap_or_default();
    limits.max_positions = request.driver_config.ctx_size;
    limits.max_sequences = request.scheduler_config.max_active_sequences;
    limits.max_batch_tokens = request.scheduler_config.max_batch_tokens;
    if let Some(bytes) = request.kv_cache_bytes {
        limits.kv_bytes =
            usize::try_from(bytes).map_err(|_| invalid("35B KV budget exceeds usize"))?;
    }
    Ok(limits)
}

/// Metadata-only host ledger for the actual bound image. Kept CPU-testable;
/// no CUDA device, tokenizer or weight payload has been created at this point.
#[cfg(any(feature = "cuda", test))]
fn host_startup_options(
    resources: &BoundDecoderResources,
    mut options: ferrule_model::transformer::host_experts::HostExpertCacheOptions,
    max_positions: usize,
    max_tensor_bytes: u64,
) -> Result<ferrule_model::transformer::host_experts::HostExpertCacheOptions> {
    use ferrule_model::transformer::host_experts::{ExpertPrewarmMode, StandardHostStartupMemory};
    if options.mode == ExpertPrewarmMode::Full {
        let base =
            StandardHostStartupMemory::for_resources(resources, max_positions, max_tensor_bytes)?;
        // Fresh construction: no base payload credit. Catalog/planning allocations
        // already live in RSS/current and must not be charged a second time.
        options.memory_budget = base.memory_budget()?;
        tracing::info!(
            ?base,
            "35B remaining host base ledger (after expert warm, before ready)"
        );
    }
    Ok(options)
}

#[cfg(feature = "cuda")]
fn build_moe(request: ResolvedModelRequest) -> Result<BoxedSessionInferenceEngine> {
    use ferrule_model::decoder::{HybridCudaDecoder, HybridCudaDevice};
    if request.backend != ModelExecutionBackend::Cuda {
        return Err(moe_unavailable(BackendSelection::Cpu));
    }
    if request.pipeline.is_some() {
        return Err(ep_unsupported(
            "Qwen3.5 generic pipeline TP/PP/EP/process is unsupported; use the dedicated resident GPU-thread EP placement",
        ));
    }
    if request.expert_memory_policy != ExpertMemoryPolicy::default()
        || request.moe_hotset_experts != 0
        || request.expert_reader_max_tensor_bytes != 64 << 20
        || request.output_head_chunk_rows != 4096
    {
        return Err(invalid(
            "35B does not consume legacy pinned expert cache, hotset, separate expert-reader or output-head chunk overrides; use explicit compressed host prewarm options for pageable host residency",
        ));
    }
    if request.driver_config.enable_native_proposals
        || request.scheduler_config.prefix_cache_capacity_pages != 0
        || request.scheduler_config.prefill_chunk_size == 0
        || request.scheduler_config.prefill_chunk_size > request.scheduler_config.max_batch_tokens
    {
        return Err(invalid(
            "35B requires bounded token-serial prefill with prefix cache and speculation disabled",
        ));
    }
    let (metadata, resources) = Qwen35Adapter::bind_hf_metadata(&request.model_path)?;
    if request.max_layers != metadata.config().text().num_hidden_layers {
        return Err(invalid("35B requires the full checkpoint layer count"));
    }
    let limits = capacity_limits(&request)?;
    let target = Qwen35MoeCapacityPlan::from_config(metadata.config(), limits)?;
    let admission =
        Qwen35MoeCudaAdmission::for_resources(target, &resources, request.max_tensor_bytes)?;
    let profile =
        admission.context_capacity_profile(request.scheduler_config.prefill_chunk_size)?;
    tracing::info!(?profile, "Qwen3.5-35B owner effective context capacity");
    let planes = HybridStateSchema::from_spec(resources.spec(), request.max_layers)?.kv_planes(
        admission.options.page_size(),
        admission.options.max_positions(),
    )?;
    let accounting = ResidentKvPageAccounting::TransactionCapacity {
        page_bytes: physical_page_bytes(&planes)?,
        budget_bytes: Some(limits.kv_bytes as u64),
    };
    log_kv_plan("Qwen3.5-35B numeric FP8", &planes, accounting, &request)?;
    let host_options = host_startup_options(
        &resources,
        request
            .family_options
            .qwen35()
            .and_then(|options| options.moe())
            .expect("validated Qwen35Moe")
            .host_cache,
        admission.options.max_positions(),
        request.max_tensor_bytes,
    )?;
    tracing::info!(
        ?host_options,
        "Qwen3.5-35B host expert startup admission (readiness gated)"
    );
    let resources =
        resources.with_host_expert_prewarm(host_options, request.max_tensor_bytes, |progress| {
            tracing::info!(
                read_bytes = progress.read_bytes,
                total_bytes = progress.total_bytes,
                completed_parameters = progress.completed_parameters,
                total_parameters = progress.total_parameters,
                elapsed_seconds = progress.elapsed.as_secs_f64(),
                "host expert prewarm progress"
            );
        })?;
    let host_experts = resources.host_experts().cloned();
    if let Some(cache) = &host_experts {
        tracing::info!(generation = cache.generation(), stats = ?cache.stats(), "all routed host experts ready (compressed immutable proofs)");
    } else {
        tracing::warn!(
            "explicit lazy expert mode: routed weights are NOT prewarmed; new prompts may read NAS"
        );
    }
    let device = HybridCudaDevice::new_on_device(0, admission.budget)?;
    let (free_device_bytes, total_device_bytes) = device.operators().memory_info()?;
    admission.check_available_device_bytes(free_device_bytes)?;
    tracing::info!(
        free_device_bytes,
        total_device_bytes,
        required_device_bytes = admission.total_device_bytes(),
        "Qwen3.5-35B physical CUDA admission"
    );
    let tokenizer = TokenizerHandle::load(&request.model_path)?;
    let runner = GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda(
        resources,
        tokenizer,
        admission.options,
        device,
    )?;
    tracing::info!(
        resident_bytes = runner.forward_executor().module().resident_parameter_bytes(),
        cache = ?runner.forward_executor().module().expert_cache_stats(),
        workspace = ?runner.forward_executor().module().numeric_workspace_usage(),
        precision = ?runner.forward_executor().module().numeric_fp8_precision(),
        "Qwen3.5-35B runner prepared: compressed FP8 storage, bounded cache"
    );
    let engine = super::super::composition::compose_resident_engine(
        runner,
        Box::new(planes),
        accounting,
        request.scheduler_config,
        request.driver_config,
    )?;
    Ok(Box::new(ReportedMoeEngine {
        engine,
        host_experts,
        limits,
        expert_placement: None,
        shutdown_attempted: false,
        close_reported: false,
    }))
}

#[cfg(feature = "cuda")]
type MoeEngine = crate::engine::ResidentInferenceEngine<
    GenericDecoderRunner<ferrule_model::decoder::HybridCudaDecoder>,
    crate::scheduling::FixedSequenceSlotPool,
>;

#[cfg(any(feature = "cuda", test))]
trait MoeShutdownReport: crate::engine::SessionInferenceEngine {
    fn report_shutdown(&self);
}

/// Only adds reports to the existing engine; does not create another runner,
/// scheduler, worker or KV pool. Read model cache stats after shutdown custody.
#[cfg(any(feature = "cuda", test))]
struct ReportedMoeEngine<E: MoeShutdownReport> {
    engine: E,
    host_experts: Option<std::sync::Arc<ferrule_model::transformer::host_experts::HostExpertCache>>,
    limits: Qwen35MoeCapacityLimits,
    expert_placement: Option<Qwen35ExpertPlacement>,
    shutdown_attempted: bool,
    close_reported: bool,
}

#[cfg(any(feature = "cuda", test))]
impl<E: MoeShutdownReport> ReportedMoeEngine<E> {
    fn inner(&self) -> &E {
        &self.engine
    }
    fn inner_mut(&mut self) -> &mut E {
        &mut self.engine
    }
}

#[cfg(any(feature = "cuda", test))]
impl<E: MoeShutdownReport> crate::engine::InferenceEngine for ReportedMoeEngine<E> {
    fn completion_hub(&self) -> ferrule_common::CompletionHub {
        self.inner().completion_hub()
    }
    fn take_completion_reactors(&mut self) -> Vec<crate::engine::InferenceCompletionReactor> {
        self.inner_mut().take_completion_reactors()
    }
    fn has_background_work(&self) -> bool {
        self.inner().has_background_work()
    }
    fn has_pending_async_work(&self) -> bool {
        self.inner().has_pending_async_work()
    }
    fn start_background_work(&mut self) -> Result<()> {
        self.inner_mut().start_background_work()
    }
    fn shutdown(&mut self) -> Result<crate::engine::InferenceShutdownProgress> {
        self.shutdown_attempted = true;
        let progress = self.inner_mut().shutdown()?;
        if progress == crate::engine::InferenceShutdownProgress::Complete && !self.close_reported {
            self.engine.report_shutdown();
            self.close_reported = true;
        }
        Ok(progress)
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner().encode(text)
    }
    fn submit(&mut self, request: crate::GenerateRequest) {
        self.inner_mut().submit(request);
    }
    fn try_submit(&mut self, request: crate::GenerateRequest) -> Result<()> {
        self.inner_mut().try_submit(request)
    }
    fn request_cleanup(&self, request_id: crate::RequestId) -> crate::InferenceRequestCleanup {
        self.inner().request_cleanup(request_id)
    }
    fn admission_snapshot(&self) -> Option<crate::RuntimeAdmissionSnapshot> {
        self.inner().admission_snapshot()
    }
    fn capacity_snapshot(&self) -> crate::engine::inference::InferenceCapacitySnapshot {
        self.inner().capacity_snapshot()
    }
    fn set_admission_options(&mut self, options: crate::RuntimeAdmissionOptions) -> Result<()> {
        self.inner_mut().set_admission_options(options)
    }
    fn close_admission(&mut self) -> Result<()> {
        self.inner_mut().close_admission()
    }
    fn step(
        &mut self,
        emit: &mut dyn FnMut(&crate::ResidentTokenEvent) -> Result<()>,
    ) -> Result<crate::ResidentDriverStep> {
        self.inner_mut().step(emit)
    }
    fn cancel_request(&mut self, id: crate::RequestId) -> Result<crate::InferenceCancelProgress> {
        self.inner_mut().cancel_request(id)
    }
    fn drain_finished(&mut self) -> Vec<crate::scheduling::SequenceState> {
        self.inner_mut().drain_finished()
    }
    fn drain_cancelled(&mut self) -> Vec<crate::scheduling::SequenceState> {
        self.inner_mut().drain_cancelled()
    }
    fn drain_failed(&mut self) -> Vec<crate::scheduling::SequenceState> {
        self.inner_mut().drain_failed()
    }
}

#[cfg(any(feature = "cuda", test))]
impl<E: MoeShutdownReport> crate::engine::SessionInferenceEngine for ReportedMoeEngine<E> {
    fn model_info(&self) -> ferrule_model::ModelInfo {
        self.inner().model_info()
    }
    fn observability_snapshot(&self) -> crate::ResidentEngineObservability {
        self.inner().observability_snapshot()
    }
    fn bound_layer_count(&self) -> Option<usize> {
        self.inner().bound_layer_count()
    }
    fn expert_report(&self) -> Option<String> {
        if let Some(placement) = &self.expert_placement {
            return Some(format!(
                "Qwen3.5-35B F32Tf32x3 CUDA thread EP{}; root={} experts={:?}; all owned experts eager resident, no local routed cache; root-only KV; TP/PP/process/restarts unsupported; host={}\n",
                placement.expert_parallel(),
                placement.root_device(),
                placement.expert_devices(),
                self.host_experts.as_ref().map_or_else(
                    || "missing".into(),
                    |host| format!(
                        "full ready generation={} stats={:?} hits={}",
                        host.generation(),
                        host.stats(),
                        host.hits()
                    )
                ),
            ));
        }
        Some(format!(
            "Qwen3.5-35B: compressed E4M3FN + numeric BF16 scales; F32Tf32x3; device cache entries={} bytes={} scratch={}; host={}\n",
            self.limits.max_experts,
            self.limits.max_bytes,
            self.limits.scratch_bytes,
            self.host_experts.as_ref().map_or_else(
                || "lazy (NOT prewarmed)".into(),
                |cache| format!(
                    "full ready generation={} stats={:?} hits={}",
                    cache.generation(),
                    cache.stats(),
                    cache.hits()
                )
            )
        ))
    }
    fn retain_session(&mut self, id: crate::SessionId) -> Result<()> {
        self.inner_mut().retain_session(id)
    }
    fn retained_session_position(&self, id: crate::SessionId) -> Option<usize> {
        self.inner().retained_session_position(id)
    }
    fn reset_session(&mut self, id: crate::SessionId) -> Result<()> {
        self.inner_mut().reset_session(id)
    }
    fn take_request_terminal(
        &mut self,
        id: crate::RequestId,
    ) -> Option<crate::scheduling::RequestTerminal> {
        self.inner_mut().take_request_terminal(id)
    }
}

#[cfg(feature = "cuda")]
impl MoeShutdownReport for MoeEngine {
    fn report_shutdown(&self) {
        use crate::engine::SessionInferenceEngine;
        let snapshot = self.observability_snapshot();
        let physically_closed = self.driver().physical_shutdown_complete();
        tracing::info!(kv = ?snapshot.kv_cache, driver = ?snapshot.driver, physically_closed,
            "Qwen3.5-35B resident owner final runtime stats");
        let module = self
            .driver()
            .runner_for_report()
            .forward_executor()
            .module();
        tracing::info!(resident_bytes = module.resident_parameter_bytes(),
            cache = ?module.expert_cache_stats(), workspace = ?module.numeric_workspace_usage(),
            precision = ?module.numeric_fp8_precision(), physically_closed,
            "Qwen3.5-35B owner final model stats");
    }
}

#[cfg(any(feature = "cuda", test))]
impl<E: MoeShutdownReport> Drop for ReportedMoeEngine<E> {
    fn drop(&mut self) {
        use crate::engine::InferenceEngine;
        // Never retry a reported failure behind the caller's back. The same
        // engine still owns the runner and any unknown-completion quarantine.
        if !self.shutdown_attempted {
            match self.shutdown() {
                Ok(crate::engine::InferenceShutdownProgress::Complete) => {}
                Ok(crate::engine::InferenceShutdownProgress::Pending) => {
                    tracing::warn!("Qwen3.5-35B dropped before shutdown completed");
                }
                Err(error) => tracing::error!(%error, "Qwen3.5-35B best-effort shutdown failed"),
            }
        }
        if !self.close_reported {
            self.engine.report_shutdown();
        }
    }
}

fn build_bound(
    request: ResolvedModelRequest,
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
            let kv_plan = plan_resident_kv_pages(
                &schema,
                accounting,
                request.scheduler_config,
                request.driver_config,
            )?;
            let pages = kv_plan.configured_pages;
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
            let root_required_bytes = [
                budget.weight_bytes,
                budget.state_bytes,
                budget.workspace_bytes,
                estimate.kv_bytes,
            ]
            .into_iter()
            .try_fold(0usize, |total, bytes| {
                total
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("Qwen3.5 root memory requirement overflow"))
            })?;
            let profile = kv_plan
                .context_capacity_profile(request.scheduler_config, request.driver_config)?
                .with_memory_requirements(
                    state_bytes,
                    budget.workspace_bytes,
                    root_required_bytes,
                )?;
            tracing::info!(
                ?estimate,
                ?budget,
                ?profile,
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

    // Only the CUDA-specific telemetry hook is substituted. The production
    // decorator and real resident inference/cleanup implementations run unchanged.
    impl MoeShutdownReport for BoxedSessionInferenceEngine {
        fn report_shutdown(&self) {
            assert_eq!(
                self.capacity_snapshot().kv.unwrap().logical_allocated_pages,
                0
            );
        }
    }

    fn reported_cpu_engine() -> ReportedMoeEngine<BoxedSessionInferenceEngine> {
        let fixture = fixture::Fixture::new();
        let descriptor = super::super::tests::descriptor(ModelFamily::Qwen35, "tiny-hybrid", 4);
        let mut options = super::super::tests::options();
        options.driver_config.ctx_size = 32;
        options.driver_config.enable_native_proposals = false;
        options.scheduler_config.max_batch_tokens = 4;
        options.scheduler_config.prefill_chunk_size = 2;
        let entry = resolve(&descriptor, BackendSelection::Cpu).unwrap();
        let request = configure_request(&descriptor, entry, options).unwrap();
        ReportedMoeEngine {
            engine: build_bound(
                request,
                fixture.resources(),
                TokenizerHandle::load(&fixture.dir).unwrap(),
            )
            .unwrap(),
            host_experts: None,
            limits: Qwen35MoeCapacityLimits::default(),
            expert_placement: None,
            shutdown_attempted: false,
            close_reported: false,
        }
    }

    #[test]
    fn reported_moe_cpu_forwards_admission_capacity_and_close_barrier() {
        use crate::engine::InferenceEngine;
        let mut engine = reported_cpu_engine();
        let limits = crate::RuntimeAdmissionOptions {
            max_waiting_requests: 1,
            max_request_identities: 2,
            max_session_identities: 2,
        };
        engine.set_admission_options(limits).unwrap();
        assert_eq!(engine.admission_snapshot().unwrap().limits, limits);
        assert_eq!(
            engine.admission_snapshot(),
            engine.inner().admission_snapshot()
        );
        let capacity = engine.capacity_snapshot();
        assert!(capacity.scheduler.is_some());
        assert!(capacity.kv.is_some());
        assert_eq!(capacity, engine.inner().capacity_snapshot());
        assert!(matches!(
            engine.set_admission_options(crate::RuntimeAdmissionOptions {
                max_waiting_requests: 0,
                ..limits
            }),
            Err(Error::Admission {
                source: crate::RuntimeAdmissionError::InvalidOptions
            })
        ));
        assert_eq!(engine.admission_snapshot().unwrap().limits, limits);
        let request = |id| GenerateRequest {
            id: RequestId(id),
            session_id: Some(SessionId(id)),
            prompt_tokens: vec![1, 3, 2],
            max_new_tokens: 2,
            stop: vec![],
            ignore_eos: true,
        };
        engine.try_submit(request(1)).unwrap();
        assert_eq!(engine.admission_snapshot().unwrap().waiting_requests, 1);
        assert_eq!(
            engine
                .capacity_snapshot()
                .scheduler
                .unwrap()
                .waiting_requests,
            1
        );
        assert!(matches!(
            engine.try_submit(request(1)),
            Err(Error::Admission {
                source: crate::RuntimeAdmissionError::DuplicateRequest { .. }
            })
        ));
        assert!(matches!(
            engine.try_submit(request(2)),
            Err(Error::Admission {
                source: crate::RuntimeAdmissionError::Capacity { .. }
            })
        ));
        let crate::InferenceRequestCleanup::Tracked(receipt) = engine.request_cleanup(RequestId(1))
        else {
            panic!("production wrapper lost the resident receipt")
        };
        engine.step(&mut |_| Ok(())).unwrap();
        assert_eq!(
            engine.capacity_snapshot(),
            engine.inner().capacity_snapshot()
        );
        assert!(
            engine
                .capacity_snapshot()
                .kv
                .unwrap()
                .logical_allocated_pages
                > 0
        );
        engine.close_admission().unwrap();
        assert!(engine.admission_snapshot().unwrap().closed);
        assert!(engine.inner().admission_snapshot().unwrap().closed);
        assert!(
            !receipt.is_released(),
            "close is not terminal or physical cleanup"
        );
        assert!(matches!(
            engine.try_submit(request(3)),
            Err(Error::Admission {
                source: crate::RuntimeAdmissionError::Closed
            })
        ));
        engine.cancel_request(RequestId(1)).unwrap();
        assert!(!receipt.is_released(), "terminal is still unconsumed");
        assert_eq!(engine.drain_cancelled().len(), 1);
        assert!(receipt.is_released());
        assert_eq!(
            engine.admission_snapshot().unwrap().request_identities_held,
            0
        );
        assert_eq!(
            engine.shutdown().unwrap(),
            InferenceShutdownProgress::Complete
        );
        assert!(engine.shutdown_attempted && engine.close_reported);
    }

    #[test]
    fn reported_moe_cpu_forwards_exact_receipts_across_retained_turns() {
        use crate::engine::{InferenceEngine, SessionInferenceEngine};
        let mut engine = reported_cpu_engine();
        let report = engine.expert_report().unwrap();
        assert!(report.contains("compressed E4M3FN + numeric BF16 scales; F32Tf32x3"));
        engine.expert_placement = Some(Qwen35ExpertPlacement::new(8, (0..8).collect()).unwrap());
        let ep_report = engine.expert_report().unwrap();
        assert!(ep_report.contains("CUDA thread EP8; root=0"));
        assert!(
            ep_report.contains("root-only KV; TP/PP/process/restarts unsupported; host=missing")
        );
        engine.retain_session(SessionId(7)).unwrap();
        let mut previous: Option<crate::RequestCleanupReceipt> = None;
        for round in 0..2 {
            engine
                .try_submit(GenerateRequest {
                    id: RequestId(1),
                    session_id: Some(SessionId(7)),
                    prompt_tokens: engine.encode("t1 t3 t2").unwrap(),
                    max_new_tokens: 1,
                    stop: vec![],
                    ignore_eos: true,
                })
                .unwrap();
            let crate::InferenceRequestCleanup::Tracked(receipt) =
                engine.request_cleanup(RequestId(1))
            else {
                panic!("production wrapper must forward the exact generation receipt")
            };
            let crate::InferenceRequestCleanup::Tracked(inner_receipt) =
                engine.inner().request_cleanup(RequestId(1))
            else {
                panic!("real resident must issue a receipt")
            };
            if let Some(old) = &previous {
                assert!(old.is_released());
            }
            let mut tokens = 0;
            for _ in 0..100 {
                engine
                    .step(&mut |_| {
                        tokens += 1;
                        Ok(())
                    })
                    .unwrap();
                assert!(!receipt.is_released());
                assert!(!inner_receipt.is_released());
                let consumed = if round == 0 {
                    !engine.drain_finished().is_empty()
                } else {
                    engine.take_request_terminal(RequestId(1)).is_some()
                };
                if consumed {
                    break;
                }
            }
            assert_eq!(tokens, 1);
            assert!(receipt.is_released());
            assert!(inner_receipt.is_released());
            assert!(matches!(
                engine.request_cleanup(RequestId(1)),
                crate::InferenceRequestCleanup::Unavailable
            ));
            assert_eq!(
                engine.admission_snapshot().unwrap().request_identities_held,
                0
            );
            assert_eq!(
                engine.admission_snapshot().unwrap().session_identities_held,
                1
            );
            assert_eq!(
                engine.retained_session_position(SessionId(7)),
                Some((round + 1) * 4)
            );
            assert!(
                engine
                    .capacity_snapshot()
                    .kv
                    .unwrap()
                    .logical_allocated_pages
                    > 0
            );
            previous = Some(receipt);
        }
        engine.reset_session(SessionId(7)).unwrap();
        assert_eq!(engine.retained_session_position(SessionId(7)), Some(0));
        assert_eq!(
            engine
                .capacity_snapshot()
                .kv
                .unwrap()
                .logical_allocated_pages,
            0
        );
        assert!(engine.drain_failed().is_empty());
        assert_eq!(engine.expert_report().unwrap(), ep_report);
        assert_eq!(
            engine.shutdown().unwrap(),
            InferenceShutdownProgress::Complete
        );
        assert!(engine.shutdown_attempted && engine.close_reported);
    }

    #[test]
    fn family_options_legacy_dto_matrix_preserves_defaults_and_rejections() {
        use ferrule_model::transformer::host_experts::{ExpertPrewarmMode, HostExpertCacheOptions};
        for family in [
            ModelFamily::Qwen3,
            ModelFamily::QwenMoe,
            ModelFamily::DeepSeekV4,
            ModelFamily::Qwen35,
            ModelFamily::Qwen35Moe,
        ] {
            for capacity in [None, Some(Qwen35MoeCapacityLimits::default())] {
                for host in [
                    None,
                    Some(HostExpertCacheOptions::default()),
                    Some(HostExpertCacheOptions {
                        mode: ExpertPrewarmMode::Lazy,
                        max_bytes: 0,
                        max_experts: 0,
                        ..Default::default()
                    }),
                ] {
                    let mut input = super::super::tests::options();
                    input.qwen35_moe_capacity = capacity;
                    input.qwen35_host_cache = host;
                    let result = resolve_family_options(&family, &input);
                    if family != ModelFamily::Qwen35Moe && (capacity.is_some() || host.is_some()) {
                        let Error::InvalidRequest { message } = result.unwrap_err() else {
                            panic!("legacy family rejection must remain InvalidRequest");
                        };
                        assert_eq!(
                            message,
                            if capacity.is_some() {
                                "CUDA expert device cache options require Qwen3.5 MoE FP8"
                            } else {
                                "expert prewarm options require Qwen3.5 MoE FP8"
                            }
                        );
                        continue;
                    }
                    match result.unwrap() {
                        ResolvedFamilyOptions::Qwen35(Qwen35ResolvedAdapter::Moe {
                            options,
                            expert_placement,
                        }) => {
                            assert_eq!(family, ModelFamily::Qwen35Moe);
                            assert_eq!(options.device_capacity, capacity);
                            assert_eq!(options.host_cache, host.unwrap_or_default());
                            assert!(expert_placement.is_none());
                        }
                        ResolvedFamilyOptions::Qwen35(Qwen35ResolvedAdapter::Dense) => {
                            assert_eq!(family, ModelFamily::Qwen35);
                        }
                        ResolvedFamilyOptions::Generic => {
                            assert!(!matches!(
                                family,
                                ModelFamily::Qwen35 | ModelFamily::Qwen35Moe
                            ));
                        }
                    }
                    assert_eq!(input.qwen35_moe_capacity, capacity);
                    assert_eq!(input.qwen35_host_cache, host);
                }
            }
        }
    }

    #[test]
    fn family_adapter_keeps_capacity_before_host_validation_and_typed_host_errors() {
        use ferrule_model::transformer::host_experts::{
            HostExpertCacheError, HostExpertCacheOptions,
        };
        let mut input = super::super::tests::options();
        input.qwen35_moe_capacity = Some(Qwen35MoeCapacityLimits {
            max_experts: 0,
            ..Default::default()
        });
        input.qwen35_host_cache = Some(HostExpertCacheOptions {
            workers: 0,
            ..Default::default()
        });
        let error = resolve_family_options(&ModelFamily::Qwen35Moe, &input).unwrap_err();
        assert!(matches!(error, Error::InvalidRequest { ref message }
            if message == "35B requires nonzero bounded expert cache with room beyond scratch"));
        input.qwen35_moe_capacity = None;
        let Error::Backend {
            source: ferrule_common::Error::ModelSource { source },
        } = resolve_family_options(&ModelFamily::Qwen35Moe, &input).unwrap_err()
        else {
            panic!("host validation must preserve the typed model source");
        };
        assert!(source.downcast_ref::<HostExpertCacheError>().is_some());
    }

    #[test]
    fn resolved_capacity_uses_current_common_inputs_without_a_second_budget() {
        let descriptor = super::super::tests::descriptor(ModelFamily::Qwen35Moe, "metadata", 40);
        let mut input = super::super::tests::options();
        input.max_tensor_mebibytes = 1024;
        input.driver_config.enable_native_proposals = false;
        let requested = Qwen35MoeCapacityLimits {
            max_positions: 99,
            max_sequences: 99,
            max_batch_tokens: 99,
            ..Default::default()
        };
        input.qwen35_moe_capacity = Some(requested);
        // Internal metadata seam also exercises the adapter in a CPU-only build.
        // This does not bypass the public resolver's CUDA feature check.
        let entry = ResolvedModelBackend {
            family: ModelFamily::Qwen35Moe,
            model_name: "qwen3.5-35b-a3b-fp8",
            backend: ModelExecutionBackend::Cuda,
            backend_profile: Qwen35MoeCapacityPlan::PLANNED_BACKEND_PROFILE,
            default_chat_template: ChatTemplate::Qwen35,
        };
        let mut request = configure_request(&descriptor, entry, input).unwrap();
        for (context, sequences, batch, kv) in [(8, 1, 4, None), (64, 3, 32, Some(2 << 20))] {
            request.driver_config.ctx_size = context;
            request.scheduler_config.max_active_sequences = sequences;
            request.scheduler_config.max_batch_tokens = batch;
            request.kv_cache_bytes = kv;
            assert_eq!(
                capacity_limits(&request).unwrap(),
                Qwen35MoeCapacityLimits {
                    max_positions: context,
                    max_sequences: sequences,
                    max_batch_tokens: batch,
                    kv_bytes: kv.map_or(requested.kv_bytes, |bytes| bytes as usize),
                    ..requested
                }
            );
        }
        let family = request.family_options.qwen35_mut().unwrap();
        family.set_expert_placement(Qwen35ExpertPlacement::new(2, vec![3, 1]).unwrap());
        let compatibility = ModelBuildRequest { resolved: request };
        let cloned = compatibility.clone();
        assert_eq!(
            capacity_limits(&compatibility.resolved).unwrap(),
            capacity_limits(&cloned.resolved).unwrap()
        );
        let family = cloned.resolved.family_options.qwen35().unwrap();
        assert_eq!(family.moe().unwrap().device_capacity, Some(requested));
        assert_eq!(family.expert_placement().unwrap().expert_devices(), &[3, 1]);
    }

    fn assert_ep_error(error: Error) -> String {
        let Error::Backend {
            source: ferrule_common::Error::ModelSource { source },
        } = error
        else {
            panic!("expected typed Qwen3.5 EP capability error")
        };
        let unsupported = source
            .downcast_ref::<ferrule_model::transformer::UnsupportedOperator>()
            .unwrap();
        assert_eq!(unsupported.operator, "qwen35_moe_expert_parallel");
        unsupported.to_string()
    }

    #[test]
    fn ep_placement_is_explicit_distinct_and_not_a_kv_mesh() {
        for degree in [2, 4, 8] {
            let devices = (0..degree).rev().collect::<Vec<_>>();
            let placement = Qwen35ExpertPlacement::new(degree, devices.clone()).unwrap();
            assert_eq!(placement.root_device(), degree - 1);
            assert_eq!(placement.expert_devices(), devices);
            assert_eq!(
                placement.validate_expert_shape(40, 256).unwrap(),
                10240 / degree
            );
            assert_eq!(placement.validate_expert_shape(2, 8).unwrap(), 16 / degree);
            for (layers, experts) in [(0, 256), (40, 0), (40, 3)] {
                assert_ep_error(
                    placement
                        .validate_expert_shape(layers, experts)
                        .unwrap_err(),
                );
            }
        }
        for (degree, devices) in [
            (0, vec![]),
            (1, vec![0]),
            (3, vec![0, 1, 2]),
            (2, vec![0]),
            (2, vec![0, 1, 2]),
            (2, vec![1, 1]),
            (2, vec![0, usize::MAX]),
        ] {
            assert_ep_error(Qwen35ExpertPlacement::new(degree, devices).unwrap_err());
        }
        assert!(
            Qwen35ExpertPlacement::new(2, vec![0, 1])
                .unwrap()
                .validate_expert_shape(usize::MAX, 4)
                .is_err()
        );
    }

    #[test]
    fn ep_rejects_unimplemented_topologies_without_cuda_or_payloads() {
        let valid = PipelineBuildOptions {
            parallelism: ferrule_common::ParallelismPlan {
                expert_parallel: 8,
                ..Default::default()
            },
            devices: Some((0..8).collect()),
            ..Default::default()
        };
        assert!(Qwen35ExpertPlacement::from_options(&valid, BackendSelection::Auto).is_ok());
        assert_ep_error(
            Qwen35ExpertPlacement::from_options(&valid, BackendSelection::Cpu).unwrap_err(),
        );
        for variant in 0..10 {
            let mut options = valid.clone();
            match variant {
                0 => options.parallelism.tensor_parallel = 2,
                1 => options.parallelism.pipeline_parallel = 2,
                2 => options.parallelism.data_parallel = 2,
                3 => options.parallelism.context_parallel = 2,
                4 => options.parallelism.sequence_parallel = 2,
                5 => options.rank_backend = PipelineRankBackend::Process,
                6 => options.rank_restarts = 1,
                7 => options.rank_timeout = std::time::Duration::from_secs(1),
                8 => options.devices = None,
                _ => options.parallelism.expert_parallel = 1,
            }
            assert_ep_error(
                Qwen35ExpertPlacement::from_options(&options, BackendSelection::Auto).unwrap_err(),
            );
        }
    }

    fn tiny_ep_plan() -> Qwen35ExpertPhysicalPlan {
        Qwen35ExpertPhysicalPlan::new(
            &Qwen35ExpertPlacement::new(2, vec![7, 3]).unwrap(),
            Qwen35ExpertRootMemory {
                weights_bytes: 60,
                state_bytes: 10,
                kv_bytes: 10,
                workspace_bytes: 5,
            },
            &[40, 41],
            5,
            10,
        )
        .unwrap()
    }

    #[test]
    fn ep_physical_budget_rejects_individually_fitting_colocated_contexts() {
        let plan = tiny_ep_plan();
        assert_eq!(
            plan.cards()[0],
            Qwen35PhysicalCardMemory {
                ordinal: 7,
                root_weights_bytes: 60,
                expert_weights_bytes: 40,
                state_bytes: 10,
                kv_bytes: 10,
                workspace_bytes: 10,
                contexts: 2,
                allocator_margin_bytes: 20,
                required_bytes: 150,
            }
        );
        assert_eq!(plan.cards()[1].required_bytes, 56);
        assert_eq!(plan.cards()[1].contexts, 1);
        assert_eq!(plan.cards()[1].state_bytes, 0);
        assert_eq!(plan.cards()[1].kv_bytes, 0);
        // Root = 95 and expert0 = 55 both fit 100; their combined 150 does not.
        let mut capacities = vec![
            Qwen35PhysicalCardCapacity {
                ordinal: 3,
                free_bytes: 56,
                limit_bytes: 56,
            },
            Qwen35PhysicalCardCapacity {
                ordinal: 7,
                free_bytes: 100,
                limit_bytes: 150,
            },
        ];
        assert!(
            assert_ep_error(plan.check_capacity(&capacities).unwrap_err()).contains("ordinal 7")
        );
        capacities[1].free_bytes = 150;
        plan.check_capacity(&capacities).unwrap();
        capacities[1].limit_bytes = 149;
        assert!(plan.check_capacity(&capacities).is_err());
        capacities[1].limit_bytes = 150;
        capacities[0].free_bytes = 55;
        assert!(
            assert_ep_error(plan.check_capacity(&capacities).unwrap_err()).contains("ordinal 3")
        );
    }

    #[test]
    fn ep_physical_budget_requires_complete_unique_samples_and_checked_sums() {
        let plan = tiny_ep_plan();
        let sample = Qwen35PhysicalCardCapacity {
            ordinal: 7,
            free_bytes: usize::MAX,
            limit_bytes: usize::MAX,
        };
        for samples in [
            vec![],
            vec![sample],
            vec![sample, sample],
            vec![
                sample,
                Qwen35PhysicalCardCapacity {
                    ordinal: 99,
                    ..sample
                },
            ],
        ] {
            assert_ep_error(plan.check_capacity(&samples).unwrap_err());
        }
        let placement = Qwen35ExpertPlacement::new(2, vec![0, 1]).unwrap();
        let root = Qwen35ExpertRootMemory {
            weights_bytes: 1,
            state_bytes: 1,
            kv_bytes: 1,
            workspace_bytes: 1,
        };
        for owners in [&[1usize][..], &[0, 1], &[usize::MAX, 1]] {
            assert!(Qwen35ExpertPhysicalPlan::new(&placement, root, owners, 1, 1).is_err());
        }
        assert!(Qwen35ExpertPhysicalPlan::new(&placement, root, &[1, 1], 1, usize::MAX).is_err());
    }

    #[test]
    fn mock_live_capacity_reject_preserves_typed_primary_and_secondary_cleanup() {
        let plan = tiny_ep_plan();
        for cleanup_fails in [false, true] {
            // Metadata budgets remain valid, but live free memory has changed.
            let samples = plan
                .cards()
                .iter()
                .map(|card| Qwen35PhysicalCardCapacity {
                    ordinal: card.ordinal,
                    free_bytes: card.required_bytes - 1,
                    limit_bytes: card.required_bytes,
                })
                .collect::<Vec<_>>();
            let mut cleanup_calls = 0;
            let held_owners = std::cell::Cell::new(1);
            let result = finish_owner_admission(plan.check_capacity(&samples), || {
                cleanup_calls += 1;
                if cleanup_fails {
                    Err(ep_unsupported("mock owner shutdown unknown"))
                } else {
                    held_owners.set(0);
                    Ok(())
                }
            });
            assert_eq!(cleanup_calls, 1);
            assert_eq!(held_owners.get(), usize::from(cleanup_fails));
            let error = result.unwrap_err();
            if cleanup_fails {
                let Error::Cleanup {
                    source, cleanup, ..
                } = error
                else {
                    panic!("typed pair lost")
                };
                assert!(assert_ep_error(*source).contains("requires"));
                assert!(assert_ep_error(*cleanup).contains("shutdown unknown"));
            } else {
                assert!(assert_ep_error(error).contains("requires"));
            }
        }
        // Successful admission transfers ownership; it must not run rollback.
        assert_eq!(
            finish_owner_admission(Ok(7), || panic!("cleanup after success")).unwrap(),
            7
        );
    }

    #[test]
    fn ep8_expert_estimates_charge_all_owned_weights_without_root_cache_ceiling() {
        let placement = Qwen35ExpertPlacement::new(8, (0..8).collect()).unwrap();
        let owned = placement.validate_expert_shape(40, 256).unwrap();
        assert_eq!(owned, 1280);
        let owner_weights = owned * 3_146_112;
        let root = Qwen35ExpertRootMemory {
            // Mock estimator, not an artifact admission or hardcoded runtime budget.
            weights_bytes: 5_578_000_000,
            state_bytes: 512 << 20,
            kv_bytes: 1 << 30,
            workspace_bytes: 1 << 30,
        };
        let plan = Qwen35ExpertPhysicalPlan::new(
            &placement,
            root,
            &[owner_weights; 8],
            64 << 20,
            512 << 20,
        )
        .unwrap();
        assert_eq!(plan.cards()[0].root_weights_bytes, root.weights_bytes);
        assert_eq!(
            plan.cards()[1].required_bytes,
            owner_weights + (64 << 20) + (512 << 20)
        );
        assert_eq!(
            plan.cards()[0].required_bytes - plan.cards()[1].required_bytes,
            root.weights_bytes
                + root.state_bytes
                + root.kv_bytes
                + root.workspace_bytes
                + (512 << 20)
        );
        let capacities = plan
            .cards()
            .iter()
            .map(|card| Qwen35PhysicalCardCapacity {
                ordinal: card.ordinal,
                free_bytes: card.required_bytes,
                limit_bytes: card.required_bytes,
            })
            .collect::<Vec<_>>();
        plan.check_capacity(&capacities).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn ep_root_admission_uses_published_external_api_without_a_local_cache_charge() {
        use ferrule_model::decoder::{
            HybridCudaDecoder, HybridCudaDevice, HybridCudaMemoryEstimate, HybridCudaRoutedExperts,
        };
        use ferrule_model::transformer::{ExpertCacheLimits, NumericFp8Precision};
        let fixture = fixture::Fixture::new();
        let resources = fixture.resources();
        let options = GenericDecoderOptions::standard_cpu(
            resources.spec(),
            ModelFamily::Qwen35Moe,
            WeightSource::Safetensors,
            16,
            32,
            2,
            2,
            1 << 20,
            ExecutionPrecisionPolicy::f32(),
        )
        .unwrap();
        assert!(Qwen35ExpertRootMemory::for_resources(&resources, &options, 8).is_err());
        let options = options
            .with_hybrid_cuda_numeric_fp8_precision(
                ExpertCacheLimits {
                    max_experts: 8,
                    max_bytes: 4usize << 30,
                },
                64 << 20,
                NumericFp8Precision::F32Tf32x3,
            )
            .unwrap();
        let root = Qwen35ExpertRootMemory::for_resources(&resources, &options, 8).unwrap();
        let local = HybridCudaMemoryEstimate::for_resources(&resources, &options, 8).unwrap();
        let external =
            HybridCudaMemoryEstimate::for_resources_with_routed_experts(&resources, &options, 8)
                .unwrap();
        assert_eq!(
            local.weight_bytes_upper_bound - root.weights_bytes,
            (4usize << 30) - (64 << 20)
        );
        assert_eq!(root.weights_bytes, external.weight_bytes_upper_bound);
        assert_eq!(root.workspace_bytes, external.workspace_bytes_upper_bound);
        assert_eq!(root.kv_bytes, external.kv_bytes);
        assert_eq!(root.state_bytes, external.per_sequence_state_bytes * 5);
        let _constructor: fn(
            BoundDecoderResources,
            TokenizerHandle,
            GenericDecoderOptions,
            HybridCudaDevice,
            HybridCudaRoutedExperts,
        )
            -> ferrule_common::Result<GenericDecoderRunner<HybridCudaDecoder>> =
            GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda_with_routed_experts;
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn ep_construct_rejects_cpu_before_any_payload_or_worker_creation() {
        let descriptor =
            super::super::tests::descriptor(ModelFamily::Qwen35Moe, "absent-ep-fixture", 40);
        let entry = ResolvedModelBackend {
            family: ModelFamily::Qwen35Moe,
            model_name: "qwen35-ep-fixture",
            backend: ModelExecutionBackend::Cuda,
            backend_profile: Qwen35MoeCapacityPlan::PLANNED_BACKEND_PROFILE,
            default_chat_template: ChatTemplate::Qwen35,
        };
        let mut options = super::super::tests::options();
        options.max_tensor_mebibytes = 1024;
        options.driver_config.enable_native_proposals = false;
        let mut request = configure_request(&descriptor, entry, options).unwrap();
        assert!(
            request
                .family_options
                .qwen35()
                .unwrap()
                .expert_placement()
                .is_none(),
            "default stays single"
        );
        request
            .family_options
            .qwen35_mut()
            .unwrap()
            .set_expert_placement(Qwen35ExpertPlacement::new(2, vec![1, 0]).unwrap());
        request.backend = ModelExecutionBackend::Cpu;
        let error = build(ModelBuildRequest { resolved: request })
            .err()
            .expect("EP must not fall back to CPU");
        assert!(assert_ep_error(error).contains("CUDA resident root"));
    }

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

    #[cfg(feature = "cuda")]
    #[test]
    fn staged_moe_factory_rejects_unconsumed_options_before_loading() {
        let descriptor = super::super::tests::descriptor(
            ModelFamily::Qwen35Moe,
            "Qwen3_5MoeForConditionalGeneration",
            40,
        );
        let entry = ResolvedModelBackend {
            family: ModelFamily::Qwen35Moe,
            model_name: "qwen3.5-35b-a3b",
            backend: ModelExecutionBackend::Cuda,
            backend_profile: Qwen35MoeCapacityPlan::PLANNED_BACKEND_PROFILE,
            default_chat_template: ChatTemplate::Qwen35,
        };
        let mut options = super::super::tests::options();
        options.driver_config.enable_native_proposals = false;
        options.max_tensor_mebibytes = 1024;
        let base = configure_request(&descriptor, entry, options).unwrap();
        for case in 0..9 {
            let mut request = base.clone();
            let expected = match case {
                0 => {
                    request.backend = ModelExecutionBackend::Cpu;
                    "qwen35_moe_cpu"
                }
                1 => {
                    request.pipeline = Some(PipelineBuildOptions::default());
                    "TP/PP/EP/process"
                }
                2 => {
                    request.moe_hotset_experts = 1;
                    "does not consume"
                }
                3 => {
                    request.expert_reader_max_tensor_bytes = 1;
                    "does not consume"
                }
                4 => {
                    request.output_head_chunk_rows = 1;
                    "does not consume"
                }
                5 => {
                    request.driver_config.enable_native_proposals = true;
                    "speculation disabled"
                }
                6 => {
                    request.scheduler_config.prefix_cache_capacity_pages = 1;
                    "prefix cache"
                }
                7 => {
                    request.scheduler_config.prefill_chunk_size = 0;
                    "bounded token-serial"
                }
                _ => {
                    request.expert_memory_policy = ExpertMemoryPolicy::new(
                        MemoryPoolLimits::new(1, 64 << 20),
                        MemoryPoolLimits::new(0, 0),
                    );
                    "does not consume"
                }
            };
            let error = build_moe(request).err().unwrap();
            assert!(error.to_string().contains(expected), "case {case}: {error}");
        }
    }

    #[test]
    fn host_startup_passes_remaining_base_peak_without_turning_cap_into_reservation() {
        use ferrule_model::transformer::host_experts::{
            ExpertPrewarmMode, HostExpertCacheOptions, StandardHostStartupMemory,
        };
        let fixture = fixture::Fixture::new();
        let resources = fixture.resources();
        let input = HostExpertCacheOptions::default();
        let options = host_startup_options(&resources, input, 32, 1 << 20).unwrap();
        let ledger = StandardHostStartupMemory::for_resources(&resources, 32, 1 << 20).unwrap();
        assert_eq!(options.max_bytes, input.max_bytes);
        assert_eq!(
            options.memory_budget.base_resident_bytes,
            ledger.resident_bytes().unwrap()
        );
        assert_eq!(
            options.memory_budget.base_temporary_bytes,
            ledger.temporary_bytes()
        );
        assert!(options.memory_budget.base_resident_bytes > 0);
        assert!(options.memory_budget.base_temporary_bytes > 0);
        assert_eq!(options.memory_budget.base_already_resident_bytes, 0);
        assert_eq!(options.memory_budget.concurrent_warm_bytes, 0);
        let lazy = HostExpertCacheOptions {
            mode: ExpertPrewarmMode::Lazy,
            ..input
        };
        assert_eq!(host_startup_options(&resources, lazy, 32, 1).unwrap(), lazy);
    }

    #[test]
    fn long_context_kv_pressure_preserves_retained_owner_and_blocked_identity() {
        let context = 1024;
        let fixture = fixture::Fixture::new();
        let resources = fixture.resources_with_context(context);
        let descriptor = super::super::tests::descriptor(ModelFamily::Qwen35, "tiny-hybrid", 4);
        let mut options = super::super::tests::options();
        options.driver_config.ctx_size = context;
        options.driver_config.enable_native_proposals = false;
        options.scheduler_config.max_batch_tokens = 16;
        options.scheduler_config.prefill_chunk_size = 16;
        let entry = resolve(&descriptor, BackendSelection::Cpu).unwrap();
        let mut request = configure_request(&descriptor, entry, options).unwrap();
        let schema = HybridStateSchema::from_spec(resources.spec(), 4)
            .unwrap()
            .kv_planes(16, context)
            .unwrap();
        request.kv_cache_bytes = Some(physical_page_bytes(&schema).unwrap());
        let mut engine = build_bound(
            request,
            resources,
            TokenizerHandle::load(&fixture.dir).unwrap(),
        )
        .unwrap();
        engine.retain_session(SessionId(7)).unwrap();
        assert_eq!(run(&mut engine, 1).len(), 2);
        let retained = engine.capacity_snapshot().kv.unwrap();
        assert_eq!(retained.logical_allocated_pages, 1);
        assert_eq!(retained.logical_free_pages, Some(0));
        engine
            .try_submit(GenerateRequest {
                id: RequestId(2),
                session_id: Some(SessionId(8)),
                prompt_tokens: vec![1],
                max_new_tokens: context - 1,
                stop: vec![],
                ignore_eos: true,
            })
            .unwrap();
        assert_eq!(
            engine.admission_snapshot().unwrap().request_identities_held,
            1
        );
        assert_eq!(
            engine
                .step(&mut |_| panic!("KV-blocked work emitted output"))
                .unwrap(),
            ResidentDriverStep::Blocked
        );
        assert_eq!(
            engine
                .capacity_snapshot()
                .kv
                .unwrap()
                .logical_allocated_pages,
            1
        );
        assert_eq!(
            engine.admission_snapshot().unwrap().request_identities_held,
            1
        );
        assert!(engine.drain_failed().is_empty());
        assert!(engine.retained_session_position(SessionId(7)).unwrap() > 3);
        engine.cancel_request(RequestId(2)).unwrap();
        engine.drain_cancelled();
        assert_eq!(
            engine
                .capacity_snapshot()
                .kv
                .unwrap()
                .logical_allocated_pages,
            1,
            "cancel of blocked request cannot release retained owner's KV"
        );
        assert_eq!(
            engine.shutdown().unwrap(),
            InferenceShutdownProgress::Complete
        );
    }

    #[test]
    fn long_context_hybrid_engine_rejects_envelope_without_ownership_changes() {
        for context in [1024, 2048, 4096] {
            let fixture = fixture::Fixture::new();
            let descriptor = super::super::tests::descriptor(ModelFamily::Qwen35, "tiny-hybrid", 4);
            let mut options = super::super::tests::options();
            options.driver_config.ctx_size = context;
            options.driver_config.enable_native_proposals = false;
            options.scheduler_config.max_batch_tokens = 32;
            options.scheduler_config.prefill_chunk_size = 32;
            let entry = resolve(&descriptor, BackendSelection::Cpu).unwrap();
            let request = configure_request(&descriptor, entry, options).unwrap();
            let mut engine = build_bound(
                request,
                fixture.resources_with_context(context),
                TokenizerHandle::load(&fixture.dir).unwrap(),
            )
            .unwrap();
            let make = |id, prompt, max_new_tokens| GenerateRequest {
                id: RequestId(id),
                session_id: Some(SessionId(7)),
                prompt_tokens: vec![1; prompt],
                max_new_tokens,
                stop: vec![],
                ignore_eos: true,
            };
            let snapshot = engine.capacity_snapshot();
            for (prompt, output) in [(context - 16, 17), (context + 1, 0), (1, usize::MAX)] {
                assert!(matches!(
                    engine.try_submit(make(1, prompt, output)),
                    Err(Error::Admission {
                        source: crate::RuntimeAdmissionError::InvalidPosition { .. }
                    })
                ));
                assert_eq!(engine.capacity_snapshot(), snapshot);
                assert_eq!(
                    engine.admission_snapshot().unwrap().request_identities_held,
                    0
                );
                assert!(engine.drain_failed().is_empty());
            }
            engine.try_submit(make(1, context - 16, 16)).unwrap();
            assert_eq!(engine.admission_snapshot().unwrap().waiting_requests, 1);
            assert_eq!(
                engine.capacity_snapshot().kv,
                snapshot.kv,
                "acceptance acquires no KV pages"
            );
            engine.step(&mut |_| Ok(())).unwrap();
            let active_kv = engine.capacity_snapshot().kv;
            assert!(active_kv.unwrap().logical_allocated_pages > 0);
            let mut waiting = make(4, context - 16, 16);
            waiting.session_id = Some(SessionId(8));
            engine.try_submit(waiting).unwrap();
            assert_eq!(engine.admission_snapshot().unwrap().waiting_requests, 1);
            assert_eq!(
                engine.admission_snapshot().unwrap().request_identities_held,
                2
            );
            assert_eq!(engine.capacity_snapshot().kv, active_kv);
            engine.cancel_request(RequestId(4)).unwrap();
            assert_eq!(engine.drain_cancelled().len(), 1);
            assert_eq!(
                engine.capacity_snapshot().kv,
                active_kv,
                "waiting cancellation cannot release active KV"
            );
            assert_eq!(
                engine.admission_snapshot().unwrap().request_identities_held,
                1
            );
            engine.cancel_request(RequestId(1)).unwrap();
            assert_eq!(engine.drain_cancelled().len(), 1);
            engine.retain_session(SessionId(7)).unwrap();
            engine.try_submit(make(2, 1, 1)).unwrap();
            for _ in 0..8 {
                engine.step(&mut |_| Ok(())).unwrap();
                if !engine.drain_finished().is_empty() {
                    break;
                }
            }
            let position = engine.retained_session_position(SessionId(7)).unwrap();
            assert_eq!(position, 2);
            let retained = engine.capacity_snapshot();
            assert!(matches!(
                engine.try_submit(make(3, context - position - 1, 2)),
                Err(Error::Admission {
                    source: crate::RuntimeAdmissionError::InvalidPosition { .. }
                })
            ));
            assert_eq!(
                engine.capacity_snapshot(),
                retained,
                "invalid turn must not release retained KV"
            );
            assert_eq!(
                engine.retained_session_position(SessionId(7)),
                Some(position)
            );
            engine
                .try_submit(make(3, context - position - 1, 1))
                .unwrap();
            engine.cancel_request(RequestId(3)).unwrap();
            engine.drain_cancelled();
            assert_eq!(
                engine.shutdown().unwrap(),
                InferenceShutdownProgress::Complete
            );
        }
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
