//! Immutable DeepSeek-V4 preparation output and execution policy resolution.

use std::env as process_environment;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};

use ferrule_common::execution::{ExecutionCapabilities, KvBindingMode, LogitsRowPolicy};
use ferrule_common::{Error, Result};

use crate::execution::PreparedModel;

use super::artifact::DeepSeekV4ArtifactModel;
use super::layer::DeepSeekV4Layer;

static NEXT_PREPARED_GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4PrepareOptions {
    pub max_layers: usize,
    pub output_head_chunk_rows: usize,
    pub expert_reader_max_tensor_bytes: u64,
    pub moe_prefetch_experts: usize,
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
            moe_hotset_experts: 0,
        }
    }
}

/// Prepared external KV contract for the current serial DSV4 runner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4KvLayoutSchema {
    binding_mode: KvBindingMode,
    max_sequences: usize,
    layer_count: usize,
    window_size: usize,
    head_dim: usize,
    compress_ratios: Box<[usize]>,
}

impl DeepSeekV4KvLayoutSchema {
    pub const fn binding_mode(&self) -> KvBindingMode {
        self.binding_mode
    }

    pub const fn max_sequences(&self) -> usize {
        self.max_sequences
    }

    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub const fn window_size(&self) -> usize {
        self.window_size
    }

    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn compress_ratios(&self) -> &[usize] {
        &self.compress_ratios
    }
}

/// Immutable description of the routed-expert namespace bound by a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4ExpertCatalog {
    layer_count: usize,
    experts_per_layer: usize,
    experts_per_token: usize,
    hash_layer_count: usize,
}

impl DeepSeekV4ExpertCatalog {
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub const fn experts_per_layer(&self) -> usize {
        self.experts_per_layer
    }

    pub const fn experts_per_token(&self) -> usize {
        self.experts_per_token
    }

    pub const fn hash_layer_count(&self) -> usize {
        self.hash_layer_count
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
    fused_indexer_topk: bool,
    fused_indexer_topk_prefill: bool,
    fused_indexer_topk_decode: bool,
    pinned_expert_cache_capacity: usize,
    expert_upload_inflight: usize,
    device_router_topk: bool,
    moe_segment_batch: usize,
    profile_sync: bool,
    parity_checkpoint_selection: Option<(usize, String)>,
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
            fused_indexer_topk: true,
            fused_indexer_topk_prefill: false,
            fused_indexer_topk_decode: false,
            pinned_expert_cache_capacity: 64,
            expert_upload_inflight: 32,
            device_router_topk: false,
            moe_segment_batch: 256,
            profile_sync: false,
            parity_checkpoint_selection: None,
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

    pub const fn fused_indexer_prefill_topk(&self) -> bool {
        self.fused_indexer_topk && self.fused_indexer_topk_prefill
    }

    pub const fn fused_indexer_decode_topk(&self) -> bool {
        self.fused_indexer_topk && self.fused_indexer_topk_decode
    }

    pub const fn pinned_expert_cache_capacity(&self) -> usize {
        self.pinned_expert_cache_capacity
    }

    pub const fn expert_upload_inflight(&self) -> usize {
        self.expert_upload_inflight
    }

    pub const fn device_router_topk(&self) -> bool {
        self.device_router_topk
    }

    pub const fn moe_segment_batch(&self) -> usize {
        self.moe_segment_batch
    }

    pub const fn profile_sync(&self) -> bool {
        self.profile_sync
    }

    pub fn parity_checkpoint_selection(&self) -> Option<(usize, String)> {
        self.parity_checkpoint_selection.clone()
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
        let fused_indexer_topk = parse_env_bool(
            "FERRULE_DSV4_FUSED_INDEXER_TOPK",
            lookup("FERRULE_DSV4_FUSED_INDEXER_TOPK")?,
            true,
        )?;
        let fused_indexer_topk_prefill = parse_env_bool(
            "FERRULE_DSV4_FUSED_INDEXER_TOPK_PREFILL",
            lookup("FERRULE_DSV4_FUSED_INDEXER_TOPK_PREFILL")?,
            false,
        )?;
        let fused_indexer_topk_decode = parse_env_bool(
            "FERRULE_DSV4_FUSED_INDEXER_TOPK_DECODE",
            lookup("FERRULE_DSV4_FUSED_INDEXER_TOPK_DECODE")?,
            false,
        )?;
        let pinned_expert_cache_capacity = parse_env_usize(
            "FERRULE_DSV4_PINNED_EXPERT_CACHE_CAPACITY",
            lookup("FERRULE_DSV4_PINNED_EXPERT_CACHE_CAPACITY")?,
            64,
        )?;
        let expert_upload_inflight = parse_env_usize(
            "FERRULE_DSV4_EXPERT_UPLOAD_INFLIGHT",
            lookup("FERRULE_DSV4_EXPERT_UPLOAD_INFLIGHT")?,
            32,
        )?;
        let device_router_topk = parse_env_bool(
            "FERRULE_DSV4_DEVICE_ROUTER_TOPK",
            lookup("FERRULE_DSV4_DEVICE_ROUTER_TOPK")?,
            false,
        )?;
        let moe_segment_batch = parse_env_usize(
            "FERRULE_CUDA_MOE_SEGMENT_BATCH",
            lookup("FERRULE_CUDA_MOE_SEGMENT_BATCH")?,
            256,
        )?;
        if moe_segment_batch == 0 {
            return Err(Error::Model(
                "DeepSeek-V4 execution policy FERRULE_CUDA_MOE_SEGMENT_BATCH must be > 0".into(),
            ));
        }
        let profile_sync = parse_env_bool(
            "FERRULE_DSV4_PROFILE_SYNC",
            lookup("FERRULE_DSV4_PROFILE_SYNC")?,
            false,
        )?;
        let parity_checkpoint_selection = parse_parity_checkpoint_selection(
            lookup("FERRULE_DSV4_PARITY_CHECKPOINT_LAYER")?,
            lookup("FERRULE_DSV4_PARITY_CHECKPOINT_STAGE")?,
        )?;

        Ok(Self {
            prefill_progress,
            lookahead_prefetch,
            decode_lookahead_prefetch,
            hash_decode_prefetch_window,
            prefill_hash_lookahead,
            managed_experts,
            fused_indexer_topk,
            fused_indexer_topk_prefill,
            fused_indexer_topk_decode,
            pinned_expert_cache_capacity,
            expert_upload_inflight,
            device_router_topk,
            moe_segment_batch: moe_segment_batch.min(65_535),
            profile_sync,
            parity_checkpoint_selection,
        })
    }
}

pub struct DeepSeekV4PreparedResources {
    model: DeepSeekV4ArtifactModel,
    options: DeepSeekV4PrepareOptions,
    layers: Box<[DeepSeekV4Layer]>,
    kv_layout: DeepSeekV4KvLayoutSchema,
    expert_catalog: DeepSeekV4ExpertCatalog,
    policy: DeepSeekV4ExecutionPolicy,
}

impl DeepSeekV4PreparedResources {
    pub const fn model(&self) -> &DeepSeekV4ArtifactModel {
        &self.model
    }

    pub const fn prepare_options(&self) -> &DeepSeekV4PrepareOptions {
        &self.options
    }

    pub fn layers(&self) -> &[DeepSeekV4Layer] {
        &self.layers
    }

    pub const fn kv_layout(&self) -> &DeepSeekV4KvLayoutSchema {
        &self.kv_layout
    }

    pub const fn expert_catalog(&self) -> &DeepSeekV4ExpertCatalog {
        &self.expert_catalog
    }

    pub const fn policy(&self) -> &DeepSeekV4ExecutionPolicy {
        &self.policy
    }
}

pub type DeepSeekV4PreparedModelPlan = PreparedModel<DeepSeekV4PreparedResources>;

/// Validates and atomically prepares all immutable DSV4 model-global resources.
pub fn prepare(
    model: DeepSeekV4ArtifactModel,
    options: DeepSeekV4PrepareOptions,
) -> Result<DeepSeekV4PreparedModelPlan> {
    validate_options(&model, options)?;
    let policy = DeepSeekV4ExecutionPolicy::resolve()?;
    let capabilities = execution_capabilities(model.config.vocab_size)?;

    let mut layers = Vec::new();
    layers
        .try_reserve_exact(options.max_layers)
        .map_err(|error| {
            Error::Model(format!(
                "DeepSeek-V4 prepared layer allocation failed for {} layers: {error}",
                options.max_layers
            ))
        })?;
    for layer in 0..options.max_layers {
        layers.push(model.bind_layer(layer)?);
    }

    let kv_layout = DeepSeekV4KvLayoutSchema {
        binding_mode: KvBindingMode::None,
        max_sequences: 1,
        layer_count: options.max_layers,
        window_size: model.config.window_size,
        head_dim: model.config.head_dim,
        compress_ratios: (0..options.max_layers)
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
    let expert_catalog = DeepSeekV4ExpertCatalog {
        layer_count: options.max_layers,
        experts_per_layer: model.config.num_routed_experts,
        experts_per_token: model.config.num_experts_per_tok,
        hash_layer_count: options.max_layers.min(model.config.num_hash_layers),
    };
    let resources = DeepSeekV4PreparedResources {
        model,
        options,
        layers: layers.into_boxed_slice(),
        kv_layout,
        expert_catalog,
        policy,
    };

    publish_prepared(Ok((capabilities, resources)))
}

fn validate_options(
    model: &DeepSeekV4ArtifactModel,
    options: DeepSeekV4PrepareOptions,
) -> Result<()> {
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

fn execution_capabilities(vocab_size: usize) -> Result<ExecutionCapabilities> {
    let vocab_size = u32::try_from(vocab_size)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| {
            Error::Model(format!(
                "DeepSeek-V4 vocabulary size {vocab_size} is not representable as a non-zero u32"
            ))
        })?;
    let max_packed_rows = usize::try_from(u32::MAX).unwrap_or(usize::MAX);
    Ok(ExecutionCapabilities {
        max_batch_tokens: max_packed_rows,
        max_sequences: 1,
        max_prefill_query_tokens_per_sequence: max_packed_rows,
        max_decode_query_tokens_per_sequence: 1,
        max_top_k: NonZeroU32::new(40),
        supports_prefill: true,
        supports_decode: true,
        supports_mixed: false,
        full_logits_width: Some(vocab_size),
        kv_binding_mode: KvBindingMode::None,
        logits_row_policy: LogitsRowPolicy::LastPerSequence,
    })
}

fn publish_prepared<R>(prepared: Result<(ExecutionCapabilities, R)>) -> Result<PreparedModel<R>> {
    publish_prepared_with_generation(&NEXT_PREPARED_GENERATION, prepared)
}

fn publish_prepared_with_generation<R>(
    generations: &AtomicU64,
    prepared: Result<(ExecutionCapabilities, R)>,
) -> Result<PreparedModel<R>> {
    let (capabilities, resources) = prepared?;
    let generation = generations.fetch_add(1, Ordering::Relaxed);
    Ok(PreparedModel::new(generation, capabilities, resources))
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

fn parse_parity_checkpoint_selection(
    layer: Option<String>,
    stage: Option<String>,
) -> Result<Option<(usize, String)>> {
    match (layer, stage) {
        (None, None) => Ok(None),
        (Some(layer), Some(stage)) => {
            let layer = parse_env_usize("FERRULE_DSV4_PARITY_CHECKPOINT_LAYER", Some(layer), 0)?;
            let stage = stage.trim();
            if stage.is_empty() {
                return Err(Error::Model(
                    "DeepSeek-V4 execution policy FERRULE_DSV4_PARITY_CHECKPOINT_STAGE must not be empty".into(),
                ));
            }
            Ok(Some((layer, stage.to_owned())))
        }
        _ => Err(Error::Model(
            "DeepSeek-V4 parity checkpoint selection requires both FERRULE_DSV4_PARITY_CHECKPOINT_LAYER and FERRULE_DSV4_PARITY_CHECKPOINT_STAGE".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ferrule_common::execution::{KvBindingMode, LogitsRowPolicy};

    use super::*;

    #[test]
    fn dsv4_capabilities_truthfully_describe_serial_runner() {
        let capabilities = execution_capabilities(129_280).unwrap();
        assert_eq!(capabilities.max_sequences, 1);
        assert_eq!(capabilities.max_decode_query_tokens_per_sequence, 1);
        assert_eq!(capabilities.max_top_k.unwrap().get(), 40);
        assert!(capabilities.supports_prefill);
        assert!(capabilities.supports_decode);
        assert!(!capabilities.supports_mixed);
        assert_eq!(capabilities.full_logits_width.unwrap().get(), 129_280);
        assert_eq!(capabilities.kv_binding_mode, KvBindingMode::None);
        assert_eq!(
            capabilities.logits_row_policy,
            LogitsRowPolicy::LastPerSequence
        );
    }

    #[test]
    fn failed_preparation_stage_does_not_publish_a_generation() {
        let generations = AtomicU64::new(41);
        let failed = publish_prepared_with_generation::<()>(
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
            ("FERRULE_DSV4_FUSED_INDEXER_TOPK_PREFILL", "true"),
            ("FERRULE_DSV4_PINNED_EXPERT_CACHE_CAPACITY", "12"),
            ("FERRULE_DSV4_EXPERT_UPLOAD_INFLIGHT", "7"),
            ("FERRULE_DSV4_DEVICE_ROUTER_TOPK", "on"),
            ("FERRULE_CUDA_MOE_SEGMENT_BATCH", "70000"),
            ("FERRULE_DSV4_PROFILE_SYNC", "1"),
            ("FERRULE_DSV4_PARITY_CHECKPOINT_LAYER", "4"),
            ("FERRULE_DSV4_PARITY_CHECKPOINT_STAGE", "attention"),
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
        assert!(policy.fused_indexer_prefill_topk());
        assert!(!policy.fused_indexer_decode_topk());
        assert_eq!(policy.pinned_expert_cache_capacity(), 12);
        assert_eq!(policy.expert_upload_inflight(), 7);
        assert!(policy.device_router_topk());
        assert_eq!(policy.moe_segment_batch(), 65_535);
        assert!(policy.profile_sync());
        assert_eq!(
            policy.parity_checkpoint_selection(),
            Some((4, "attention".to_owned()))
        );
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

        let error = DeepSeekV4ExecutionPolicy::resolve_with(|name| {
            Ok((name == "FERRULE_DSV4_PARITY_CHECKPOINT_LAYER").then(|| "2".to_string()))
        })
        .unwrap_err();
        assert!(matches!(error, Error::Model(_)));
    }
}
