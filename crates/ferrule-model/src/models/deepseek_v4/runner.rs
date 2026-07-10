//! DeepSeek-V4 reference runner: ModelRunner implementation.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::families::deepseek_v4;
use crate::moe::executor::CpuReferenceExpertExecutor;
use crate::moe::prediction::{
    ExpertAccessPhase, ExpertBatchAccessEvent, ExpertHotsetPredictor, ExpertPredictContext,
    ExpertPredictionStats, ScoreBasedExpertPredictor,
};
use crate::moe::routed::RoutedMoeStepOutput;
use crate::moe::streaming::ExpertStreamingReader;
use crate::runner::{ModelInfo, ModelRunner, PrefillMode, TokenLogit, TopKModelRunner};
use ferrule_common::{Error, Result};

use super::artifact::DeepSeekV4ArtifactModel;
use super::config::deepseek_v4_linear_activation_quantization;
use super::layer::{DeepSeekV4Layer, DeepSeekV4LayerState};
use super::operators::{
    DeepSeekV4AttentionProfileStats, DeepSeekV4LayerProfileStats, DeepSeekV4Logit,
    DeepSeekV4OperatorBackend, DeepSeekV4OperatorContext, DeepSeekV4OperatorRuntimeCounters,
};

fn dsv4_preserve_expert_residency_on_reset() -> bool {
    std::env::var("FERRULE_DSV4_PRESERVE_EXPERT_RESIDENCY_ON_RESET")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value.is_empty() || value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

#[cfg(feature = "cuda")]
fn dsv4_graph_after_preserved_reset_enabled() -> bool {
    std::env::var("FERRULE_DSV4_GRAPH_AFTER_PRESERVED_RESET")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value.is_empty() || value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

#[cfg(feature = "cuda")]
fn dsv4_graph_replay_parity_guard_enabled() -> bool {
    std::env::var("FERRULE_DSV4_GRAPH_REPLAY_PARITY_GUARD")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value.is_empty() || value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn dsv4_graph_replay_parity_abs_tol() -> f32 {
    std::env::var("FERRULE_DSV4_GRAPH_REPLAY_PARITY_ABS_TOL")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(1.0e-3)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4ReferenceOptions {
    pub max_layers: usize,
    pub output_head_chunk_rows: usize,
    pub expert_reader_max_tensor_bytes: u64,
    pub moe_prefetch_experts: usize,
    /// Optional bounded per-layer resident hotset. `0` keeps the managed-memory
    /// default (no planner eviction); non-zero clamps residency to at least
    /// `num_experts_per_tok` and retains hottest routed experts first.
    pub moe_hotset_experts: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4LayerRuntimeStats {
    pub layer: usize,
    pub window_kv_len: usize,
    pub compressed_kv_len: usize,
    pub indexer_compressed_kv_len: usize,
    pub resident_experts: usize,
    pub resident_expert_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4OutputProfileStats {
    pub final_hc_head_calls: u64,
    pub final_hc_head_us: u64,
    pub final_norm_calls: u64,
    pub final_norm_us: u64,
    pub lm_head_topk_calls: u64,
    pub lm_head_topk_us: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4PrefillRuntimeStats {
    /// Public prefill calls that requested logits/top-k/full logits.
    pub logits_calls: u64,
    pub logits_tokens: u64,
    /// Public prefill calls that only advanced prompt/session state.
    pub no_logits_calls: u64,
    pub no_logits_tokens: u64,
    /// Calls routed through the interactive prefill policy.
    pub interactive_calls: u64,
    pub interactive_tokens: u64,
    /// Calls routed through the batched/segment prefill policy.
    pub batched_calls: u64,
    pub batched_tokens: u64,
    /// Segment prefill from position 0, using the full causal prompt-start path.
    pub start_segment_calls: u64,
    pub start_segment_tokens: u64,
    /// Segment prefill appended to an existing resident session prefix.
    pub append_segment_calls: u64,
    pub append_segment_tokens: u64,
    /// Token-by-token fallback calls/tokens that still use decode/feed primitives.
    pub token_fallback_calls: u64,
    pub token_fallback_tokens: u64,
}

impl Default for DeepSeekV4ReferenceOptions {
    fn default() -> Self {
        Self {
            max_layers: deepseek_v4::NUM_LAYERS,
            output_head_chunk_rows: 1024,
            expert_reader_max_tensor_bytes: 64 * 1024 * 1024,
            moe_prefetch_experts: 0,
            moe_hotset_experts: 0,
        }
    }
}

pub struct DeepSeekV4ReferenceRunner {
    pub model: DeepSeekV4ArtifactModel,
    pub options: DeepSeekV4ReferenceOptions,
    operators: DeepSeekV4OperatorContext,
    layers: Vec<Option<DeepSeekV4Layer>>,
    states: Vec<Option<DeepSeekV4LayerState>>,
    position: usize,
    prefill_stats: DeepSeekV4PrefillRuntimeStats,
    output_profile: DeepSeekV4OutputProfileStats,
    expert_reader: ExpertStreamingReader,
    expert_executor: CpuReferenceExpertExecutor,
    expert_predictor: ScoreBasedExpertPredictor,
    #[cfg(feature = "cuda")]
    decode_graph: Option<DeepSeekV4DecodeGraph>,
    #[cfg(feature = "cuda")]
    decode_graph_disabled: bool,
    #[cfg(feature = "cuda")]
    decode_graph_segmented_only: bool,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepSeekV4DecodeGraph {
    /// Captured CUDA graph segments for one decode step.
    /// Usually one full-step graph; falls back to per-layer segments when the driver
    /// rejects a monolithic graph instantiation.
    graphs: Vec<ferrule_cuda::graph::CudaGraphHandle>,
    /// Initial HC state buffer captured by the graph. Kept alive for replay pointer stability.
    _initial_hc_state: ferrule_cuda::context::CudaF32Buffer,
    /// Ping-pong HC state buffers used between layers and kept alive for graph replay.
    hc_slots: Vec<ferrule_cuda::context::CudaF32Buffer>,
    /// Index into `hc_slots` containing the final layer output.
    final_hc_slot: usize,
    /// Per-layer MoE accumulators captured by graph nodes. Kept alive for graph replay pointer stability.
    _moe_accumulators: Vec<ferrule_cuda::context::CudaF32Buffer>,
    /// Final HC state produced by the eager capture warmup. This is only used by
    /// the opt-in debug parity guard; default replay uses the native graph output.
    warmup_final_hc_state: ferrule_cuda::context::CudaF32Buffer,
    /// Captured routes/KV position are still token-specific, so this graph is a
    /// safe one-shot until bucketed graph input patching lands.
    replays_remaining: usize,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4DecodeGraph {
    fn final_hc_state(&self) -> Result<&ferrule_cuda::context::CudaF32Buffer> {
        self.hc_slots.get(self.final_hc_slot).ok_or_else(|| {
            Error::Internal(format!(
                "DSV4 decode graph final HC slot {} missing from {} slots",
                self.final_hc_slot,
                self.hc_slots.len()
            ))
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum ExpertPredictionInput<'a> {
    Prefill { token_ids: &'a [u32] },
    Decode { token_id: u32 },
}

impl ExpertPredictionInput<'_> {
    fn phase(self) -> ExpertAccessPhase {
        match self {
            Self::Prefill { .. } => ExpertAccessPhase::Prefill,
            Self::Decode { .. } => ExpertAccessPhase::Decode,
        }
    }
}

impl DeepSeekV4ReferenceRunner {
    pub fn new(
        model: DeepSeekV4ArtifactModel,
        options: DeepSeekV4ReferenceOptions,
    ) -> Result<Self> {
        Self::new_with_operator_backend(model, options, DeepSeekV4OperatorBackend::Cpu)
    }

    pub fn new_with_operator_backend(
        model: DeepSeekV4ArtifactModel,
        options: DeepSeekV4ReferenceOptions,
        operator_backend: DeepSeekV4OperatorBackend,
    ) -> Result<Self> {
        if options.max_layers > model.config.num_layers {
            return Err(Error::Model(format!(
                "DeepSeek-V4 reference runner max_layers {} exceeds model layers {}",
                options.max_layers, model.config.num_layers
            )));
        }
        if options.output_head_chunk_rows == 0 {
            return Err(Error::Model(
                "DeepSeek-V4 reference runner output_head_chunk_rows must be > 0".into(),
            ));
        }
        let mut layers = Vec::new();
        layers.resize_with(options.max_layers, || None);
        let mut states = Vec::new();
        states.resize_with(options.max_layers, || None);
        let swiglu_limit = model.config.swiglu_limit;
        let expert_reader_max_tensor_bytes = options.expert_reader_max_tensor_bytes;
        let expert_predictor =
            ScoreBasedExpertPredictor::new(options.max_layers, model.config.num_routed_experts);
        Ok(Self {
            model,
            options,
            operators: DeepSeekV4OperatorContext::new(operator_backend)?,
            layers,
            states,
            position: 0,
            prefill_stats: DeepSeekV4PrefillRuntimeStats::default(),
            output_profile: DeepSeekV4OutputProfileStats::default(),
            expert_reader: ExpertStreamingReader::new(expert_reader_max_tensor_bytes),
            expert_executor: CpuReferenceExpertExecutor::new(swiglu_limit)
                .with_activation_quantization(deepseek_v4_linear_activation_quantization()),
            expert_predictor,
            #[cfg(feature = "cuda")]
            decode_graph: None,
            #[cfg(feature = "cuda")]
            decode_graph_disabled: false,
            #[cfg(feature = "cuda")]
            decode_graph_segmented_only: true,
        })
    }

    pub fn load_hf_with_options(
        model_dir: &Path,
        max_tensor_bytes: u64,
        options: DeepSeekV4ReferenceOptions,
    ) -> Result<Self> {
        Self::new(
            DeepSeekV4ArtifactModel::load_hf_with_limit(model_dir, max_tensor_bytes)?,
            options,
        )
    }

    pub fn load_hf_with_options_and_backend(
        model_dir: &Path,
        max_tensor_bytes: u64,
        options: DeepSeekV4ReferenceOptions,
        operator_backend: DeepSeekV4OperatorBackend,
    ) -> Result<Self> {
        Self::new_with_operator_backend(
            DeepSeekV4ArtifactModel::load_hf_with_limit(model_dir, max_tensor_bytes)?,
            options,
            operator_backend,
        )
    }

    pub fn operator_backend(&self) -> DeepSeekV4OperatorBackend {
        self.operators.backend()
    }

    pub fn operator_runtime_counters(&self) -> DeepSeekV4OperatorRuntimeCounters {
        let mut counters = self.operators.runtime_counters();
        counters.expert_predictor_stats = self.expert_predictor.stats();
        counters
    }

    pub fn prefill_runtime_stats(&self) -> DeepSeekV4PrefillRuntimeStats {
        self.prefill_stats
    }

    pub fn layer_profile_stats(&self) -> Vec<DeepSeekV4LayerProfileStats> {
        self.operators.layer_profile_stats()
    }

    pub fn attention_profile_stats(&self) -> Vec<DeepSeekV4AttentionProfileStats> {
        self.operators.attention_profile_stats()
    }

    pub fn output_profile_stats(&self) -> DeepSeekV4OutputProfileStats {
        self.output_profile
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn reset(&mut self) -> Result<()> {
        let preserve_expert_residency = self.operators.backend() == DeepSeekV4OperatorBackend::Cuda
            && dsv4_preserve_expert_residency_on_reset();
        #[cfg(feature = "cuda")]
        let cuda_graph_requested = ferrule_cuda::graph::cuda_graph_enabled();
        #[cfg(feature = "cuda")]
        let graph_after_preserved_reset = dsv4_graph_after_preserved_reset_enabled();
        #[cfg(feature = "cuda")]
        {
            if self.decode_graph.is_some() {
                self.operators.cuda_mut()?.ops.sync_stream()?;
            }
            self.decode_graph = None;
            self.decode_graph_disabled =
                preserve_expert_residency && cuda_graph_requested && !graph_after_preserved_reset;
        }
        for state in self.states.iter_mut().flatten() {
            state.reset_sequence_with_expert_residency(preserve_expert_residency);
        }
        if preserve_expert_residency {
            self.operators.clear_sequence_workspaces()?;
        } else {
            self.operators.clear_expert_residency()?;
        }
        self.position = 0;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn ensure_layer_ready(&mut self, layer_idx: usize) -> Result<()> {
        if layer_idx >= self.options.max_layers {
            return Err(Error::Model(format!(
                "DeepSeek-V4 layer index {layer_idx} exceeds configured max_layers {}",
                self.options.max_layers
            )));
        }
        if self.layers[layer_idx].is_none() {
            let bind_start = Instant::now();
            self.layers[layer_idx] = Some(self.model.bind_layer(layer_idx)?);
            self.operators
                .record_layer_bind(layer_idx, duration_us(bind_start.elapsed()));
        }
        if self.states[layer_idx].is_none() {
            let state_start = Instant::now();
            self.states[layer_idx] =
                Some(self.model.new_quality_first_layer_state_with_residency(
                    layer_idx,
                    self.options.moe_prefetch_experts,
                    self.options.moe_hotset_experts,
                )?);
            self.operators
                .record_layer_state_init(layer_idx, duration_us(state_start.elapsed()));
        }
        Ok(())
    }

    fn predicted_experts_for_layer(
        &mut self,
        layer: usize,
        input: ExpertPredictionInput<'_>,
    ) -> Result<Vec<usize>> {
        self.predicted_experts_for_layer_with_router_hash(layer, input, true)
    }

    fn predicted_experts_for_layer_with_router_hash(
        &mut self,
        layer: usize,
        input: ExpertPredictionInput<'_>,
        use_router_hash: bool,
    ) -> Result<Vec<usize>> {
        let count = self
            .options
            .moe_prefetch_experts
            .min(self.model.config.num_routed_experts);
        if count == 0 {
            return Ok(Vec::new());
        }
        let phase = input.phase();

        #[cfg(feature = "cuda")]
        let (resident, materializing, host_staged) = self
            .operators
            .cuda
            .as_ref()
            .map(|cache| {
                (
                    cache.resident_experts_for_layer(layer),
                    cache.materializing_experts_for_layer(layer),
                    cache.host_staged_experts_for_layer(layer),
                )
            })
            .unwrap_or_else(|| (Vec::new(), Vec::new(), Vec::new()));
        #[cfg(not(feature = "cuda"))]
        let (resident, materializing, host_staged) = (Vec::new(), Vec::new(), Vec::new());

        let mut predicted = Vec::with_capacity(count);
        let mut excluded = resident
            .iter()
            .chain(materializing.iter())
            .copied()
            .collect::<BTreeSet<_>>();

        if use_router_hash
            && self.operators.backend() == DeepSeekV4OperatorBackend::Cuda
            && layer < self.model.config.num_hash_layers
        {
            if let Some(layer_ref) = self.layers.get(layer).and_then(Option::as_ref) {
                match input {
                    ExpertPredictionInput::Decode { token_id } => {
                        if let Some(hash_experts) =
                            layer_ref.router.hash_experts_for_token(token_id)?
                        {
                            push_candidate_experts(
                                &mut predicted,
                                &mut excluded,
                                hash_experts
                                    .into_iter()
                                    .take(self.model.config.num_experts_per_tok),
                                count,
                                self.model.config.num_routed_experts,
                            );
                        }
                    }
                    ExpertPredictionInput::Prefill { token_ids }
                        if self.prefill_hash_lookahead_enabled() =>
                    {
                        if let Some(hash_experts) = layer_ref.router.hash_expert_union_for_tokens(
                            token_ids,
                            self.model.config.num_experts_per_tok,
                            count,
                        )? {
                            push_candidate_experts(
                                &mut predicted,
                                &mut excluded,
                                hash_experts,
                                count,
                                self.model.config.num_routed_experts,
                            );
                        }
                    }
                    ExpertPredictionInput::Prefill { .. } => {}
                }
            }
        }

        if predicted.len() < count {
            for prediction in self.expert_predictor.predict(ExpertPredictContext {
                layer,
                phase,
                budget: count,
                num_experts: self.model.config.num_routed_experts,
                resident: &resident,
                materializing: &materializing,
                host_staged: &host_staged,
            }) {
                if predicted.len() >= count {
                    break;
                }
                if excluded.insert(prediction.expert.expert) {
                    predicted.push(prediction.expert.expert);
                }
            }
        }

        if predicted.len() < count {
            if let Some(state) = self.states.get(layer).and_then(Option::as_ref) {
                push_candidate_experts(
                    &mut predicted,
                    &mut excluded,
                    state
                        .expert_planner
                        .hot_experts(layer, self.model.config.num_routed_experts),
                    count,
                    self.model.config.num_routed_experts,
                );
            }
        }
        Ok(predicted)
    }

    fn observe_moe_step(
        &mut self,
        layer: usize,
        phase: ExpertAccessPhase,
        moe: &RoutedMoeStepOutput,
    ) {
        self.expert_predictor
            .observe_batch(ExpertBatchAccessEvent::from_routes(
                layer,
                phase,
                1,
                &moe.routes,
                &moe.streaming,
            ));
    }

    fn observe_pending_moe_access_events(&mut self) {
        for event in self.operators.drain_moe_access_events() {
            self.expert_predictor.observe_batch(event);
        }
    }

    fn lookahead_prefetch_enabled(&self) -> bool {
        std::env::var("FERRULE_DSV4_LOOKAHEAD_PREFETCH")
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                !(value.is_empty() || value == "0" || value == "false" || value == "off")
            })
            .unwrap_or(true)
    }

    #[cfg(feature = "cuda")]
    fn hash_decode_prefetch_window_enabled(&self) -> bool {
        std::env::var("FERRULE_DSV4_HASH_PREFETCH_WINDOW")
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                value == "1" || value == "true" || value == "on"
            })
            .unwrap_or(false)
    }

    #[cfg(feature = "cuda")]
    fn decode_lookahead_prefetch_enabled(&self) -> bool {
        std::env::var("FERRULE_DSV4_DECODE_LOOKAHEAD_PREFETCH")
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                !(value.is_empty() || value == "0" || value == "false" || value == "off")
            })
            .unwrap_or(true)
    }

    fn prefill_hash_lookahead_enabled(&self) -> bool {
        std::env::var("FERRULE_DSV4_PREFILL_HASH_LOOKAHEAD")
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                value == "1" || value == "true" || value == "on"
            })
            .unwrap_or(false)
    }

    fn prepare_predicted_experts_for_layer(
        &mut self,
        layer: usize,
        predicted_experts: &[usize],
    ) -> Result<()> {
        if predicted_experts.is_empty()
            || self.options.moe_prefetch_experts == 0
            || !self.lookahead_prefetch_enabled()
        {
            return Ok(());
        }
        let Some(state) = self.states.get_mut(layer).and_then(Option::as_mut) else {
            return Ok(());
        };
        self.operators.prefetch_predicted_experts(
            layer,
            predicted_experts,
            &mut state.expert_planner,
            &self.expert_reader,
            &mut state.expert_handles,
        )?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn prepare_hash_decode_prefetch_window(&mut self, token_id: u32) -> Result<()> {
        if self.operators.backend() != DeepSeekV4OperatorBackend::Cuda
            || self.options.moe_prefetch_experts == 0
            || !self.lookahead_prefetch_enabled()
            || !self.hash_decode_prefetch_window_enabled()
        {
            return Ok(());
        }
        let layers = self
            .options
            .max_layers
            .min(self.model.config.num_hash_layers);
        for layer_idx in 0..layers {
            self.ensure_layer_ready(layer_idx)?;
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Decode { token_id },
            )?;
            self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn prepare_decode_lookahead_prefetch(
        &mut self,
        token_id: u32,
    ) -> Result<Option<Vec<Vec<usize>>>> {
        if self.operators.backend() != DeepSeekV4OperatorBackend::Cuda
            || self.options.moe_prefetch_experts == 0
            || !self.lookahead_prefetch_enabled()
            || !self.decode_lookahead_prefetch_enabled()
        {
            return Ok(None);
        }

        let mut predictions = Vec::with_capacity(self.options.max_layers);
        for layer_idx in 0..self.options.max_layers {
            self.ensure_layer_ready(layer_idx)?;
            // Use predictor/hotset only here. Router hash lookahead remains opt-in
            // because the older hash-window path can alter residency too
            // aggressively for short interactive correctness probes.
            let predicted_experts = self.predicted_experts_for_layer_with_router_hash(
                layer_idx,
                ExpertPredictionInput::Decode { token_id },
                false,
            )?;
            self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            predictions.push(predicted_experts);
        }
        Ok(Some(predictions))
    }

    pub fn expert_prediction_stats(&self) -> ExpertPredictionStats {
        self.expert_predictor.stats()
    }

    #[cfg(feature = "cuda")]
    pub fn prewarm_predicted_experts(&mut self) -> Result<usize> {
        if self.operators.backend() != DeepSeekV4OperatorBackend::Cuda {
            return Ok(0);
        }
        let count = self
            .options
            .moe_prefetch_experts
            .max(self.options.moe_hotset_experts)
            .min(self.model.config.num_routed_experts);
        if count == 0 {
            return Ok(0);
        }
        let mut warmed = 0usize;
        for layer_idx in 0..self.options.max_layers {
            if self.layers[layer_idx].is_none() {
                self.layers[layer_idx] = Some(self.model.bind_layer(layer_idx)?);
            }
            if self.states[layer_idx].is_none() {
                self.states[layer_idx] =
                    Some(self.model.new_quality_first_layer_state_with_residency(
                        layer_idx,
                        self.options.moe_prefetch_experts,
                        self.options.moe_hotset_experts,
                    )?);
            }
            let predicted = if count >= self.model.config.num_routed_experts {
                (0..self.model.config.num_routed_experts).collect::<Vec<_>>()
            } else {
                self.states
                    .get(layer_idx)
                    .and_then(Option::as_ref)
                    .map(|state| state.expert_planner.hot_experts(layer_idx, count))
                    .unwrap_or_default()
            };
            if predicted.is_empty() {
                continue;
            }
            let state = self.states[layer_idx].as_mut().expect("initialized above");
            warmed = warmed.saturating_add(self.operators.prewarm_experts(
                layer_idx,
                &predicted,
                &mut state.expert_planner,
                &self.expert_reader,
                &mut state.expert_handles,
            )?);
        }
        Ok(warmed)
    }

    pub fn bound_layer_count(&self) -> usize {
        self.layers.iter().filter(|layer| layer.is_some()).count()
    }

    pub fn layer_runtime_stats(&self) -> Vec<DeepSeekV4LayerRuntimeStats> {
        let mut stats = Vec::new();
        for layer_idx in 0..self.options.max_layers {
            let (Some(layer), Some(state)) = (&self.layers[layer_idx], &self.states[layer_idx])
            else {
                continue;
            };
            let index_head_dim = layer.attention.config.index_head_dim;
            let (resident_experts, resident_expert_bytes) = {
                #[cfg(feature = "cuda")]
                {
                    if let Some(cache) = self.operators.cuda.as_ref() {
                        cache.resident_expert_stats_for_layer(layer_idx)
                    } else {
                        (
                            state.expert_handles.len(),
                            state.expert_handles.total_bytes(),
                        )
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    (
                        state.expert_handles.len(),
                        state.expert_handles.total_bytes(),
                    )
                }
            };
            stats.push(DeepSeekV4LayerRuntimeStats {
                layer: layer_idx,
                window_kv_len: state.kv.len(),
                compressed_kv_len: state.kv.compressed_len(),
                indexer_compressed_kv_len: state.kv.indexer_compressed_len(index_head_dim),
                resident_experts,
                resident_expert_bytes,
            });
        }
        stats
    }

    pub fn decode_token_hidden(&mut self, token_id: u32) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        if self.operators.backend == DeepSeekV4OperatorBackend::Cuda {
            return self.decode_token_hidden_cuda(token_id);
        }
        self.decode_token_hidden_reference(token_id)
    }

    fn decode_token_hidden_reference(&mut self, token_id: u32) -> Result<Vec<f32>> {
        self.advance_token_hidden_reference(token_id, true)?
            .ok_or_else(|| {
                Error::Internal("DeepSeek-V4 reference decode did not materialize hidden".into())
            })
    }

    fn advance_token_hidden_reference(
        &mut self,
        token_id: u32,
        materialize_hidden: bool,
    ) -> Result<Option<Vec<f32>>> {
        let mut hc_state = self.model.initial_hc_state_for_token(token_id)?;
        for layer_idx in 0..self.options.max_layers {
            if self.layers[layer_idx].is_none() {
                let bind_start = Instant::now();
                self.layers[layer_idx] = Some(self.model.bind_layer(layer_idx)?);
                self.operators
                    .record_layer_bind(layer_idx, duration_us(bind_start.elapsed()));
            }
            if self.states[layer_idx].is_none() {
                let state_start = Instant::now();
                self.states[layer_idx] =
                    Some(self.model.new_quality_first_layer_state_with_residency(
                        layer_idx,
                        self.options.moe_prefetch_experts,
                        self.options.moe_hotset_experts,
                    )?);
                self.operators
                    .record_layer_state_init(layer_idx, duration_us(state_start.elapsed()));
            }
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Decode { token_id },
            )?;
            self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            let layer = self.layers[layer_idx].as_ref().expect("initialized above");
            let state = self.states[layer_idx].as_mut().expect("initialized above");
            let step = layer.decode_step_with_operators(
                state,
                &hc_state,
                token_id,
                self.position,
                &predicted_experts,
                &self.expert_reader,
                &self.expert_executor,
                &mut self.operators,
            )?;
            self.observe_moe_step(layer_idx, ExpertAccessPhase::Decode, &step.moe);
            hc_state = step.hc_state;
        }
        self.position += 1;
        if !materialize_hidden {
            return Ok(None);
        }
        Ok(Some(self.normalized_last_hidden_profiled(&hc_state, 1)?))
    }

    fn normalized_last_hidden_profiled(
        &mut self,
        hc_state: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>> {
        if tokens == 0 || hc_state.len() != tokens * self.model.config.hc_config().hc_hidden_size()
        {
            return Err(Error::Model(format!(
                "DeepSeek-V4 HC head input mismatch: tokens={tokens} len={} expected {}",
                hc_state.len(),
                tokens * self.model.config.hc_config().hc_hidden_size()
            )));
        }

        let hc_head_start = Instant::now();
        let hidden = {
            #[cfg(feature = "cuda")]
            if self.operators.backend() == DeepSeekV4OperatorBackend::Cuda {
                let state_buf = self.operators.cuda_upload_f32(hc_state)?;
                let hidden_buf = self.operators.cuda_hc_head_from_device(
                    &state_buf,
                    tokens,
                    self.model.config.hc_config(),
                    &self.model.hc_head,
                )?;
                self.operators.cuda_download_f32(&hidden_buf)?
            } else {
                self.operators.hc_head(
                    hc_state,
                    tokens,
                    self.model.config.hc_config(),
                    &self.model.hc_head,
                )?
            }
            #[cfg(not(feature = "cuda"))]
            {
                self.operators.hc_head(
                    hc_state,
                    tokens,
                    self.model.config.hc_config(),
                    &self.model.hc_head,
                )?
            }
        };
        self.output_profile.final_hc_head_calls =
            self.output_profile.final_hc_head_calls.saturating_add(1);
        self.output_profile.final_hc_head_us = self
            .output_profile
            .final_hc_head_us
            .saturating_add(self.operators.finish_profile_stage(hc_head_start)?);

        let start = (tokens - 1) * self.model.config.hidden_size;
        let norm_start = Instant::now();
        let normed = self.operators.rms_norm(
            &hidden[start..start + self.model.config.hidden_size],
            &self.model.output_norm,
            self.model.config.norm_eps,
            "output_norm",
        )?;
        self.output_profile.final_norm_calls =
            self.output_profile.final_norm_calls.saturating_add(1);
        self.output_profile.final_norm_us = self
            .output_profile
            .final_norm_us
            .saturating_add(self.operators.finish_profile_stage(norm_start)?);
        Ok(normed)
    }

    #[cfg(feature = "cuda")]
    fn decode_token_hidden_cuda(&mut self, token_id: u32) -> Result<Vec<f32>> {
        let normed_dev = self
            .advance_token_hidden_cuda_device(token_id, true)?
            .ok_or_else(|| {
                Error::Internal("DeepSeek-V4 CUDA decode did not materialize hidden".into())
            })?;
        self.operators.cuda_download_f32(&normed_dev)
    }

    #[cfg(feature = "cuda")]
    fn advance_token_hidden_cuda_device(
        &mut self,
        token_id: u32,
        materialize_hidden: bool,
    ) -> Result<Option<ferrule_cuda::context::CudaF32Buffer>> {
        // The current graph captures one concrete decode row: token id,
        // position, routes, and prepared KV/cache pointers are fixed. Drop it
        // before the next row so we never replay stale token-specific work.
        // A later bucketed graph path should patch input/KV/route buffers
        // instead of recapturing.
        if self
            .decode_graph
            .as_ref()
            .map(|graph| graph.replays_remaining == 0)
            .unwrap_or(false)
        {
            self.operators.cuda_mut()?.ops.sync_stream()?;
            self.operators.record_cuda_graph_one_shot_retire();
            self.decode_graph = None;
        }

        if ferrule_cuda::graph::cuda_graph_enabled()
            && self.decode_graph.is_none()
            && !self.decode_graph_disabled
        {
            self.operators.record_cuda_graph_capture_attempt();
            let capture_start = Instant::now();
            match self.try_capture_decode_graph(token_id) {
                Ok(()) => {
                    self.operators.record_cuda_graph_capture_success();
                }
                Err(e) => {
                    self.operators.record_cuda_graph_capture_failure();
                    self.decode_graph_disabled = true;
                    eprintln!(
                        "[ferrule] CUDA graph capture failed, falling back to eager decode: {e}"
                    );
                }
            }
            self.operators
                .record_cuda_graph_capture_elapsed(capture_start.elapsed());
        }

        if self.decode_graph.is_some() {
            // Replay the captured graph: all 43 layer decode steps execute
            // as a single graph launch with no per-kernel host overhead.
            let stream = self.operators.cuda_stream_clone()?;

            // Split borrow: graph state is mutable, operators/model/profile are disjoint.
            let DeepSeekV4ReferenceRunner {
                decode_graph,
                operators,
                model,
                position,
                output_profile,
                ..
            } = self;
            let graph = decode_graph.as_mut().expect("checked above");
            let replay_start = Instant::now();
            for graph_segment in &graph.graphs {
                graph_segment.launch(&stream)?;
            }
            operators.record_cuda_graph_replay(replay_start.elapsed());
            graph.replays_remaining = graph.replays_remaining.saturating_sub(1);

            let mut use_warmup_final_hc = false;
            if dsv4_graph_replay_parity_guard_enabled() {
                let replay_hc = operators.cuda_download_f32(graph.final_hc_state()?)?;
                let warmup_hc = operators.cuda_download_f32(&graph.warmup_final_hc_state)?;
                let max_abs_diff = replay_hc
                    .iter()
                    .zip(warmup_hc.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0f32, f32::max);
                let parity_failed = replay_hc.len() != warmup_hc.len()
                    || !max_abs_diff.is_finite()
                    || max_abs_diff > dsv4_graph_replay_parity_abs_tol();
                if parity_failed {
                    operators.record_cuda_graph_replay_fallback();
                    use_warmup_final_hc = true;
                    eprintln!(
                        "[ferrule] CUDA graph replay parity guard fell back to warmup output: final_hc_len replay={} warmup={} max_abs_diff={max_abs_diff:.6e}",
                        replay_hc.len(),
                        warmup_hc.len()
                    );
                }
            }

            *position += 1;
            if !materialize_hidden {
                return Ok(None);
            }
            // hc_head + output_norm run after replay (not inside graph). The parity
            // guard is opt-in debugging; by default native graph output is used.
            let final_hc_state = if use_warmup_final_hc {
                &graph.warmup_final_hc_state
            } else {
                graph.final_hc_state()?
            };
            let final_hc_start = Instant::now();
            let hidden_dev = operators.cuda_hc_head_from_device(
                final_hc_state,
                1,
                model.config.hc_config(),
                &model.hc_head,
            )?;
            output_profile.final_hc_head_calls =
                output_profile.final_hc_head_calls.saturating_add(1);
            output_profile.final_hc_head_us = output_profile
                .final_hc_head_us
                .saturating_add(operators.finish_profile_stage(final_hc_start)?);
            let final_norm_start = Instant::now();
            let normed_dev = operators.cuda_rms_norm_device_cached(
                "output_norm",
                &hidden_dev,
                &model.output_norm,
                model.config.norm_eps,
            )?;
            output_profile.final_norm_calls = output_profile.final_norm_calls.saturating_add(1);
            output_profile.final_norm_us = output_profile
                .final_norm_us
                .saturating_add(operators.finish_profile_stage(final_norm_start)?);
            return Ok(Some(normed_dev));
        }

        let mut hc_state_dev = {
            let hc_state = self.model.initial_hc_state_for_token(token_id)?;
            self.operators.cuda_upload_f32(&hc_state)?
        };
        let decode_lookahead_predictions = self.prepare_decode_lookahead_prefetch(token_id)?;
        self.prepare_hash_decode_prefetch_window(token_id)?;
        for layer_idx in 0..self.options.max_layers {
            self.ensure_layer_ready(layer_idx)?;
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Decode { token_id },
            )?;
            // The all-layer lookahead above is deliberately predictor/hotset-only.
            // Keep the original per-layer prepare so current-token hash experts are
            // still available to the actual MoE plan.
            if decode_lookahead_predictions.is_none() {
                self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            } else if !predicted_experts.is_empty() {
                self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            }
            let layer = self.layers[layer_idx].as_ref().expect("initialized above");
            let state = self.states[layer_idx].as_mut().expect("initialized above");
            let step = layer.decode_step_device_hc_device(
                state,
                &hc_state_dev,
                token_id,
                self.position,
                &predicted_experts,
                &self.expert_reader,
                &mut self.operators,
            )?;
            self.observe_moe_step(layer_idx, ExpertAccessPhase::Decode, &step.moe);
            hc_state_dev = step.hc_state;
        }
        self.position += 1;
        if !materialize_hidden {
            return Ok(None);
        }
        let final_hc_start = Instant::now();
        let hidden_dev = self.operators.cuda_hc_head_from_device(
            &hc_state_dev,
            1,
            self.model.config.hc_config(),
            &self.model.hc_head,
        )?;
        self.output_profile.final_hc_head_calls =
            self.output_profile.final_hc_head_calls.saturating_add(1);
        self.output_profile.final_hc_head_us = self
            .output_profile
            .final_hc_head_us
            .saturating_add(self.operators.finish_profile_stage(final_hc_start)?);
        let final_norm_start = Instant::now();
        let normed_dev = self.operators.cuda_rms_norm_device_cached(
            "output_norm",
            &hidden_dev,
            &self.model.output_norm,
            self.model.config.norm_eps,
        )?;
        self.output_profile.final_norm_calls =
            self.output_profile.final_norm_calls.saturating_add(1);
        self.output_profile.final_norm_us = self
            .output_profile
            .final_norm_us
            .saturating_add(self.operators.finish_profile_stage(final_norm_start)?);
        Ok(Some(normed_dev))
    }

    #[cfg(feature = "cuda")]
    fn try_capture_decode_graph(&mut self, token_id: u32) -> Result<()> {
        use ferrule_cuda::graph::capture_decode_graph;

        // Ensure all layers are bound before capture.
        for layer_idx in 0..self.options.max_layers {
            if self.layers[layer_idx].is_none() {
                self.layers[layer_idx] = Some(self.model.bind_layer(layer_idx)?);
            }
            if self.states[layer_idx].is_none() {
                self.states[layer_idx] =
                    Some(self.model.new_quality_first_layer_state_with_residency(
                        layer_idx,
                        self.options.moe_prefetch_experts,
                        self.options.moe_hotset_experts,
                    )?);
            }
        }

        let max_layers = self.options.max_layers;
        let position = self.position;
        let mut capture_states: Vec<DeepSeekV4LayerState> = self
            .states
            .iter()
            .take(max_layers)
            .map(|state| state.as_ref().expect("bound above").clone())
            .collect();

        // ── Phase 1: Warmup pass (eager, isolated state) ──
        // Run one full decode step to:
        // - Pre-allocate all device buffers (KV cache, combined KV, topk, sink)
        // - Upload all weights (HC, norm, attention linears, rope tables)
        // - Determine per-layer routing (expert selection + route weights)
        // - Populate KV cache and compressor state
        //
        // This pass intentionally runs against `capture_states`, not live
        // `self.states`. On graph-capture failure the measured session remains
        // untouched and falls back to eager decode. On success we commit
        // `capture_states` exactly once as the semantic state transition for
        // the decode token; replay only materializes the output hidden/logits.
        // After warmup, all idempotent resource checks will return early,
        // so the capture pass only records kernel launches.
        let warmup_start = Instant::now();
        let mut hc_state_dev = {
            let hc_state = self.model.initial_hc_state_for_token(token_id)?;
            self.operators.cuda_upload_f32(&hc_state)?
        };
        // Keep graph capture's residency preparation aligned with eager decode.
        // Without this, capture may see an empty predictor/hotset path and record
        // a different expert-residency shape from the eager runtime path.
        let _ = self.prepare_decode_lookahead_prefetch(token_id)?;
        self.prepare_hash_decode_prefetch_window(token_id)?;
        let mut layer_routes: Vec<(Vec<usize>, Vec<f32>)> = Vec::with_capacity(max_layers);
        let mut layer_moe_steps: Vec<RoutedMoeStepOutput> = Vec::with_capacity(max_layers);
        for layer_idx in 0..max_layers {
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Decode { token_id },
            )?;
            let layer = self.layers[layer_idx].as_ref().expect("bound above");
            let state = capture_states
                .get_mut(layer_idx)
                .expect("capture state initialized above");
            let step = layer.decode_step_device_hc_device(
                state,
                &hc_state_dev,
                token_id,
                position,
                &predicted_experts,
                &self.expert_reader,
                &mut self.operators,
            )?;
            hc_state_dev = step.hc_state;
            layer_routes.push((
                step.moe
                    .routes
                    .iter()
                    .map(|route| route.expert)
                    .collect::<Vec<_>>(),
                step.moe
                    .routes
                    .iter()
                    .map(|route| route.weight)
                    .collect::<Vec<_>>(),
            ));
            layer_moe_steps.push(step.moe);
        }
        self.operators
            .record_cuda_graph_capture_warmup(warmup_start.elapsed());
        let warmup_final_hc_state = hc_state_dev;

        let prepare_start = Instant::now();
        for layer_idx in 0..max_layers {
            let layer = self.layers[layer_idx].as_ref().expect("bound above");
            let (experts, weights) = &layer_routes[layer_idx];
            self.operators
                .cuda
                .as_mut()
                .expect("cuda initialized")
                .prepare_routed_moe_graph_safe(
                    self.model.config.hidden_size,
                    experts,
                    weights,
                    layer.layer,
                )?;
        }

        // ── Phase 2: Capture pass (graph-safe) ──
        // All buffers referenced by graph nodes must outlive capture and remain
        // stable across replay. Keep them in `DeepSeekV4DecodeGraph` instead of
        // returning/dropping transient per-layer buffers.
        if max_layers == 0 {
            return Err(Error::Internal(
                "DSV4 decode graph capture requires at least one layer".into(),
            ));
        }
        let hc_state_init = self.model.initial_hc_state_for_token(token_id)?;
        let initial_hc_state = self.operators.cuda_upload_f32(&hc_state_init)?;
        let hc_dim = self.model.config.hc_config().hc_hidden_size();
        let mut hc_slots: Vec<ferrule_cuda::context::CudaF32Buffer> = Vec::with_capacity(2);
        hc_slots.push(self.operators.cuda_zero_f32(hc_dim)?);
        hc_slots.push(self.operators.cuda_zero_f32(hc_dim)?);

        // Pre-allocate per-layer MoE accumulators before capture.
        let hidden_size = self.model.config.hidden_size;
        let mut moe_accumulators: Vec<ferrule_cuda::context::CudaF32Buffer> =
            Vec::with_capacity(max_layers);
        for _ in 0..max_layers {
            moe_accumulators.push(self.operators.cuda_zero_f32(hidden_size)?);
        }

        // Ensure all layer/attention graph arenas are allocated before stream
        // capture. The old monolithic-fail-then-segment fallback accidentally
        // did this during the failed full capture; make it explicit so direct
        // segmented capture is capture-safe and deterministic.
        for layer_idx in 0..max_layers {
            let layer = self.layers[layer_idx].as_ref().expect("bound above");
            let state = capture_states
                .get_mut(layer_idx)
                .expect("capture state initialized above");
            state.ensure_graph_arena(
                layer.hc_config,
                layer.attention.config,
                &mut self.operators,
            )?;
        }

        // Ensure graph-safe pointer/route-weight workspaces prepared above are
        // visible before stream capture records kernels that consume them.
        self.operators.cuda_mut()?.ops.sync_stream()?;

        // Clone the stream for capture.
        let stream = self.operators.cuda_stream_clone()?;
        self.operators
            .record_cuda_graph_capture_prepare(prepare_start.elapsed());

        macro_rules! capture_layer {
            ($layer_idx:expr, $next_hc_slot:expr) => {{
                let layer_idx = $layer_idx;
                tracing::debug!("graph capture: layer {}", layer_idx);
                let layer = self.layers[layer_idx].as_ref().expect("bound above");
                let state = capture_states
                    .get_mut(layer_idx)
                    .expect("capture state initialized above");
                let (experts, weights) = &layer_routes[layer_idx];
                let moe_accum = &mut moe_accumulators[layer_idx];
                if layer_idx == 0 {
                    let output_hc = &mut hc_slots[$next_hc_slot];
                    layer.decode_step_graph_safe(
                        state,
                        &initial_hc_state,
                        position,
                        experts,
                        weights,
                        &mut self.operators,
                        moe_accum,
                        output_hc,
                    )
                } else if $next_hc_slot == 0 {
                    let (slot0, slot1) = hc_slots.split_at_mut(1);
                    layer.decode_step_graph_safe(
                        state,
                        &slot1[0],
                        position,
                        experts,
                        weights,
                        &mut self.operators,
                        moe_accum,
                        &mut slot0[0],
                    )
                } else {
                    let (slot0, slot1) = hc_slots.split_at_mut(1);
                    layer.decode_step_graph_safe(
                        state,
                        &slot0[0],
                        position,
                        experts,
                        weights,
                        &mut self.operators,
                        moe_accum,
                        &mut slot1[0],
                    )
                }
                .map_err(|err| {
                    Error::Internal(format!(
                        "DSV4 decode graph capture layer {layer_idx} failed: {err}"
                    ))
                })
            }};
        }

        let capture_record_start = Instant::now();
        let mut next_hc_slot = 0usize;
        let (graphs, final_hc_slot) = if self.decode_graph_segmented_only {
            let mut graphs = Vec::with_capacity(max_layers);
            for layer_idx in 0..max_layers {
                let graph = capture_decode_graph(&stream, || {
                    capture_layer!(layer_idx, next_hc_slot)?;
                    Ok(())
                })?;
                graphs.push(graph);
                next_hc_slot = 1 - next_hc_slot;
            }
            self.operators
                .record_cuda_graph_captured_segments(graphs.len());
            (graphs, 1 - next_hc_slot)
        } else {
            let full_graph = capture_decode_graph(&stream, || {
                for layer_idx in 0..max_layers {
                    capture_layer!(layer_idx, next_hc_slot)?;
                    next_hc_slot = 1 - next_hc_slot;
                }
                Ok(())
            });

            match full_graph {
                Ok(graph) => {
                    self.operators.record_cuda_graph_captured_segments(1);
                    (vec![graph], 1 - next_hc_slot)
                }
                Err(full_err) => {
                    self.decode_graph_segmented_only = true;
                    self.operators.record_cuda_graph_full_capture_failure();
                    eprintln!(
                        "[ferrule] CUDA full decode graph capture failed, trying per-layer graph segments: {full_err}"
                    );
                    let mut graphs = Vec::with_capacity(max_layers);
                    next_hc_slot = 0;
                    for layer_idx in 0..max_layers {
                        let graph = capture_decode_graph(&stream, || {
                            capture_layer!(layer_idx, next_hc_slot)?;
                            Ok(())
                        })?;
                        graphs.push(graph);
                        next_hc_slot = 1 - next_hc_slot;
                    }
                    self.operators
                        .record_cuda_graph_captured_segments(graphs.len());
                    (graphs, 1 - next_hc_slot)
                }
            }
        };
        self.operators
            .record_cuda_graph_capture_record(capture_record_start.elapsed());

        for (layer_idx, state) in capture_states.into_iter().enumerate() {
            self.states[layer_idx] = Some(state);
        }
        for (layer_idx, moe) in layer_moe_steps.iter().enumerate() {
            self.observe_moe_step(layer_idx, ExpertAccessPhase::Decode, moe);
        }

        self.decode_graph = Some(DeepSeekV4DecodeGraph {
            graphs,
            _initial_hc_state: initial_hc_state,
            hc_slots,
            final_hc_slot,
            _moe_accumulators: moe_accumulators,
            warmup_final_hc_state,
            replays_remaining: 1,
        });
        Ok(())
    }

    pub fn decode_token_logits_row_range(
        &mut self,
        token_id: u32,
        start_row: usize,
        row_count: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.decode_token_hidden(token_id)?;
        self.model.logits_for_hidden_row_range_with_operators(
            &hidden,
            start_row,
            row_count,
            &mut self.operators,
        )
    }

    pub fn decode_token_logits(&mut self, token_id: u32) -> Result<Vec<f32>> {
        let hidden = self.decode_token_hidden(token_id)?;
        self.model.logits_for_hidden_chunked_with_operators(
            &hidden,
            self.options.output_head_chunk_rows,
            &mut self.operators,
        )
    }

    pub fn decode_token_topk(
        &mut self,
        token_id: u32,
        top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        #[cfg(feature = "cuda")]
        if self.operators.backend == DeepSeekV4OperatorBackend::Cuda {
            let hidden = self
                .advance_token_hidden_cuda_device(token_id, true)?
                .ok_or_else(|| {
                    Error::Internal("DeepSeek-V4 CUDA top-k did not materialize hidden".into())
                })?;
            let topk_start = Instant::now();
            let logits = self.model.topk_logits_for_hidden_device_with_operators(
                &hidden,
                top_k,
                self.options.output_head_chunk_rows,
                &mut self.operators,
            )?;
            self.output_profile.lm_head_topk_calls =
                self.output_profile.lm_head_topk_calls.saturating_add(1);
            self.output_profile.lm_head_topk_us = self
                .output_profile
                .lm_head_topk_us
                .saturating_add(self.operators.finish_profile_stage(topk_start)?);
            return Ok(logits);
        }

        let hidden = self.decode_token_hidden(token_id)?;
        let topk_start = Instant::now();
        let logits = self.model.topk_logits_for_hidden_with_operators(
            &hidden,
            top_k,
            self.options.output_head_chunk_rows,
            &mut self.operators,
        )?;
        self.output_profile.lm_head_topk_calls =
            self.output_profile.lm_head_topk_calls.saturating_add(1);
        self.output_profile.lm_head_topk_us = self
            .output_profile
            .lm_head_topk_us
            .saturating_add(self.operators.finish_profile_stage(topk_start)?);
        Ok(logits)
    }

    /// Advance the model session with a token without materializing lm_head logits
    /// or the final output hidden. This is the hot prompt/generated-token append
    /// path used by interactive chat.
    pub fn feed_token(&mut self, token_id: u32) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.operators.backend == DeepSeekV4OperatorBackend::Cuda {
            self.advance_token_hidden_cuda_device(token_id, false)?;
            return Ok(());
        }
        self.advance_token_hidden_reference(token_id, false)
            .map(|_| ())
    }

    /// Correctness-first prefill fallback: execute prompt tokens one-by-one through
    /// the decode path while preserving KV/compressor state. This is intentionally
    /// not the production batched prefill kernel, but it exercises real DSV4 weights
    /// and keeps semantics close to the official causal path.
    pub fn prefill_tokens_hidden(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(Error::Model(
                "DeepSeek-V4 prefill requires at least one token".into(),
            ));
        }
        self.prefill_stats.token_fallback_calls =
            self.prefill_stats.token_fallback_calls.saturating_add(1);
        self.prefill_stats.token_fallback_tokens = self
            .prefill_stats
            .token_fallback_tokens
            .saturating_add(token_ids.len() as u64);

        let mut hidden = Vec::new();
        for (idx, &token_id) in token_ids.iter().enumerate() {
            if idx + 1 == token_ids.len() {
                hidden = self.decode_token_hidden(token_id)?;
            } else {
                self.feed_token(token_id)?;
            }
        }
        Ok(hidden)
    }

    pub fn prefill_tokens_logits_row_range(
        &mut self,
        token_ids: &[u32],
        start_row: usize,
        row_count: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.prefill_tokens_hidden(token_ids)?;
        self.model.logits_for_hidden_row_range_with_operators(
            &hidden,
            start_row,
            row_count,
            &mut self.operators,
        )
    }

    pub fn prefill_tokens_logits(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.prefill_tokens_hidden(token_ids)?;
        self.model.logits_for_hidden_chunked_with_operators(
            &hidden,
            self.options.output_head_chunk_rows,
            &mut self.operators,
        )
    }

    pub fn prefill_tokens_topk(
        &mut self,
        token_ids: &[u32],
        top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        let hidden = self.prefill_tokens_hidden(token_ids)?;
        self.model.topk_logits_for_hidden_with_operators(
            &hidden,
            top_k,
            self.options.output_head_chunk_rows,
            &mut self.operators,
        )
    }

    /// Interactive prefill optimized for short chat turns on the CUDA backend.
    ///
    /// The existing batched prefill path is correctness-first and host-heavy. For
    /// terminal chat, short prompts are faster when appended through the
    /// device-resident decode chain: all prompt tokens except the last update KV
    /// and MoE residency without materializing output hidden/logits; the last
    /// prompt token materializes top-k for generation.
    pub fn prefill_tokens_topk_interactive(
        &mut self,
        token_ids: &[u32],
        top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        if token_ids.is_empty() {
            return Err(Error::Model(
                "DeepSeek-V4 interactive prefill requires at least one token".into(),
            ));
        }
        self.prefill_stats.interactive_calls =
            self.prefill_stats.interactive_calls.saturating_add(1);
        self.prefill_stats.interactive_tokens = self
            .prefill_stats
            .interactive_tokens
            .saturating_add(token_ids.len() as u64);
        self.prefill_tokens_topk_batched(token_ids, top_k)
    }

    /// Public prefill entry point for chat/run integration.
    ///
    /// Process a prompt segment through the existing session prefix.
    ///
    /// At session start this follows the official DSV4 prefill shape. For later
    /// chat turns it keeps existing KV/compressor/expert residency and processes
    /// the new segment layer-by-layer, matching the prefix-cache execution shape
    /// used by serving engines instead of re-running the whole model once per
    /// prompt token.
    fn prefill_tokens_hc_states_batched(&mut self, token_ids: &[u32]) -> Result<(Vec<f32>, usize)> {
        if token_ids.is_empty() {
            return Err(Error::Model(
                "DeepSeek-V4 prefill requires at least one token".into(),
            ));
        }
        let tokens = token_ids.len();
        let start_pos = self.position;
        if start_pos == 0 {
            self.prefill_stats.start_segment_calls =
                self.prefill_stats.start_segment_calls.saturating_add(1);
            self.prefill_stats.start_segment_tokens = self
                .prefill_stats
                .start_segment_tokens
                .saturating_add(tokens as u64);
        } else {
            self.prefill_stats.append_segment_calls =
                self.prefill_stats.append_segment_calls.saturating_add(1);
            self.prefill_stats.append_segment_tokens = self
                .prefill_stats
                .append_segment_tokens
                .saturating_add(tokens as u64);
        }

        let mut hc_state = self.model.initial_hc_state_for_tokens(token_ids)?;
        for layer_idx in 0..self.options.max_layers {
            if self.layers[layer_idx].is_none() {
                let bind_start = Instant::now();
                self.layers[layer_idx] = Some(self.model.bind_layer(layer_idx)?);
                self.operators
                    .record_layer_bind(layer_idx, duration_us(bind_start.elapsed()));
            }
            if self.states[layer_idx].is_none() {
                let state_start = Instant::now();
                self.states[layer_idx] =
                    Some(self.model.new_quality_first_layer_state_with_residency(
                        layer_idx,
                        self.options.moe_prefetch_experts,
                        self.options.moe_hotset_experts,
                    )?);
                self.operators
                    .record_layer_state_init(layer_idx, duration_us(state_start.elapsed()));
            }
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Prefill { token_ids },
            )?;
            self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            let layer = self.layers[layer_idx].as_ref().expect("initialized above");
            let state = self.states[layer_idx].as_mut().expect("initialized above");
            hc_state = layer.prefill_start_with_operators(
                state,
                &hc_state,
                token_ids,
                start_pos,
                &predicted_experts,
                &self.expert_reader,
                &self.expert_executor,
                &mut self.operators,
            )?;
            self.observe_pending_moe_access_events();
        }
        self.position += tokens;
        Ok((hc_state, tokens))
    }

    pub fn prefill_tokens_no_logits_batched(&mut self, token_ids: &[u32]) -> Result<()> {
        if token_ids.is_empty() {
            return Err(Error::Model(
                "DeepSeek-V4 no-logits prefill requires at least one token".into(),
            ));
        }
        self.prefill_stats.no_logits_calls = self.prefill_stats.no_logits_calls.saturating_add(1);
        self.prefill_stats.no_logits_tokens = self
            .prefill_stats
            .no_logits_tokens
            .saturating_add(token_ids.len() as u64);
        self.prefill_stats.batched_calls = self.prefill_stats.batched_calls.saturating_add(1);
        self.prefill_stats.batched_tokens = self
            .prefill_stats
            .batched_tokens
            .saturating_add(token_ids.len() as u64);

        if token_ids.len() == 1 {
            self.prefill_stats.token_fallback_calls =
                self.prefill_stats.token_fallback_calls.saturating_add(1);
            self.prefill_stats.token_fallback_tokens =
                self.prefill_stats.token_fallback_tokens.saturating_add(1);
            return self.feed_token(token_ids[0]);
        }
        let _ = self.prefill_tokens_hc_states_batched(token_ids)?;
        Ok(())
    }

    pub fn prefill_tokens_no_logits_interactive(&mut self, token_ids: &[u32]) -> Result<()> {
        if token_ids.is_empty() {
            return Err(Error::Model(
                "DeepSeek-V4 interactive no-logits prefill requires at least one token".into(),
            ));
        }
        self.prefill_stats.interactive_calls =
            self.prefill_stats.interactive_calls.saturating_add(1);
        self.prefill_stats.interactive_tokens = self
            .prefill_stats
            .interactive_tokens
            .saturating_add(token_ids.len() as u64);
        self.prefill_tokens_no_logits_batched(token_ids)
    }

    pub fn prefill_tokens_hidden_batched(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(Error::Model(
                "DeepSeek-V4 prefill requires at least one token".into(),
            ));
        }
        if token_ids.len() == 1 {
            return self.prefill_tokens_hidden(token_ids);
        }
        let (hc_state, tokens) = self.prefill_tokens_hc_states_batched(token_ids)?;
        self.normalized_last_hidden_profiled(&hc_state, tokens)
    }

    pub fn prefill_tokens_logits_batched(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.prefill_tokens_hidden_batched(token_ids)?;
        self.model.logits_for_hidden_chunked_with_operators(
            &hidden,
            self.options.output_head_chunk_rows,
            &mut self.operators,
        )
    }

    pub fn prefill_tokens_logits_row_range_batched(
        &mut self,
        token_ids: &[u32],
        start_row: usize,
        row_count: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.prefill_tokens_hidden_batched(token_ids)?;
        self.model.logits_for_hidden_row_range_with_operators(
            &hidden,
            start_row,
            row_count,
            &mut self.operators,
        )
    }

    pub fn prefill_tokens_topk_batched(
        &mut self,
        token_ids: &[u32],
        top_k: usize,
    ) -> Result<Vec<DeepSeekV4Logit>> {
        self.prefill_stats.logits_calls = self.prefill_stats.logits_calls.saturating_add(1);
        self.prefill_stats.logits_tokens = self
            .prefill_stats
            .logits_tokens
            .saturating_add(token_ids.len() as u64);
        self.prefill_stats.batched_calls = self.prefill_stats.batched_calls.saturating_add(1);
        self.prefill_stats.batched_tokens = self
            .prefill_stats
            .batched_tokens
            .saturating_add(token_ids.len() as u64);

        let hidden = self.prefill_tokens_hidden_batched(token_ids)?;
        let topk_start = Instant::now();
        let logits = self.model.topk_logits_for_hidden_with_operators(
            &hidden,
            top_k,
            self.options.output_head_chunk_rows,
            &mut self.operators,
        )?;
        self.output_profile.lm_head_topk_calls =
            self.output_profile.lm_head_topk_calls.saturating_add(1);
        self.output_profile.lm_head_topk_us = self
            .output_profile
            .lm_head_topk_us
            .saturating_add(self.operators.finish_profile_stage(topk_start)?);
        Ok(logits)
    }

    // ── Prefill parity harness ────────────────────────────────────────────
    //
    // These methods run the two prefill paths (batched vs token-loop) and
    // capture the last prompt token's HC state after every layer.  The CLI
    // `deepseek-v4-prefill-parity` command calls both, then compares layer
    // by layer to find the first divergence.

    /// Batched/device prefill: all tokens through each layer at once.
    ///
    /// Returns the HC state of the *last* prompt token after each layer
    /// (0-indexed).  Element `i` is the HC state after layer `i`.
    pub fn prefill_batched_layer_hc_trace(&mut self, token_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        if token_ids.is_empty() {
            return Err(Error::Model(
                "DeepSeek-V4 prefill parity requires at least one token".into(),
            ));
        }
        let tokens = token_ids.len();
        let start_pos = self.position;
        let hc_dim = self.model.config.hc_config().hc_hidden_size();

        // Single-token: the batched path falls back to the token-loop, so the
        // two traces are identical by construction.
        if tokens == 1 {
            return self.prefill_token_loop_layer_hc_trace(token_ids);
        }

        let mut hc_state = self.model.initial_hc_state_for_tokens(token_ids)?;
        let mut trace = Vec::with_capacity(self.options.max_layers);
        for layer_idx in 0..self.options.max_layers {
            self.ensure_layer_bound_and_state(layer_idx)?;
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Prefill { token_ids },
            )?;
            self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            let layer = self.layers[layer_idx].as_ref().expect("bound above");
            let state = self.states[layer_idx].as_mut().expect("state above");
            hc_state = layer.prefill_start_with_operators(
                state,
                &hc_state,
                token_ids,
                start_pos,
                &predicted_experts,
                &self.expert_reader,
                &self.expert_executor,
                &mut self.operators,
            )?;
            self.observe_pending_moe_access_events();
            // Extract last token's HC state.
            let last_start = (tokens - 1) * hc_dim;
            trace.push(hc_state[last_start..last_start + hc_dim].to_vec());
        }
        self.position += tokens;
        Ok(trace)
    }

    /// Token-loop prefill: tokens one-by-one through all layers.
    ///
    /// Returns the HC state of the *last* prompt token after each layer
    /// (0-indexed).  Element `i` is the HC state after layer `i`.
    pub fn prefill_token_loop_layer_hc_trace(
        &mut self,
        token_ids: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        if token_ids.is_empty() {
            return Err(Error::Model(
                "DeepSeek-V4 prefill parity requires at least one token".into(),
            ));
        }
        // Feed all tokens except the last without capturing.
        for &token_id in &token_ids[..token_ids.len() - 1] {
            self.feed_token(token_id)?;
        }
        // Last token: run layer-by-layer, capturing HC state after each layer.
        let last_token = token_ids[token_ids.len() - 1];
        self.decode_token_layer_hc_trace(last_token)
    }

    /// Run a single token through all layers, capturing HC state after each.
    ///
    /// This is the per-layer capture version of `decode_token_hidden` /
    /// `advance_token_hidden_*`.  Used by the prefill parity harness for the
    /// last prompt token.
    pub fn decode_token_layer_hc_trace(&mut self, token_id: u32) -> Result<Vec<Vec<f32>>> {
        #[cfg(feature = "cuda")]
        if self.operators.backend == DeepSeekV4OperatorBackend::Cuda {
            return self.decode_token_layer_hc_trace_cuda(token_id);
        }
        self.decode_token_layer_hc_trace_reference(token_id)
    }

    fn decode_token_layer_hc_trace_reference(&mut self, token_id: u32) -> Result<Vec<Vec<f32>>> {
        let mut hc_state = self.model.initial_hc_state_for_token(token_id)?;
        let mut trace = Vec::with_capacity(self.options.max_layers);
        for layer_idx in 0..self.options.max_layers {
            self.ensure_layer_bound_and_state(layer_idx)?;
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Decode { token_id },
            )?;
            self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            let layer = self.layers[layer_idx].as_ref().expect("bound above");
            let state = self.states[layer_idx].as_mut().expect("state above");
            let step = layer.decode_step_with_operators(
                state,
                &hc_state,
                token_id,
                self.position,
                &predicted_experts,
                &self.expert_reader,
                &self.expert_executor,
                &mut self.operators,
            )?;
            self.observe_moe_step(layer_idx, ExpertAccessPhase::Decode, &step.moe);
            hc_state = step.hc_state;
            trace.push(hc_state.clone());
        }
        self.position += 1;
        Ok(trace)
    }

    #[cfg(feature = "cuda")]
    fn decode_token_layer_hc_trace_cuda(&mut self, token_id: u32) -> Result<Vec<Vec<f32>>> {
        let mut hc_state_dev = {
            let hc_state = self.model.initial_hc_state_for_token(token_id)?;
            self.operators.cuda_upload_f32(&hc_state)?
        };
        let mut trace = Vec::with_capacity(self.options.max_layers);
        for layer_idx in 0..self.options.max_layers {
            self.ensure_layer_ready(layer_idx)?;
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Decode { token_id },
            )?;
            if !predicted_experts.is_empty() {
                self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            }
            let layer = self.layers[layer_idx].as_ref().expect("bound above");
            let state = self.states[layer_idx].as_mut().expect("state above");
            let step = layer.decode_step_device_hc_device(
                state,
                &hc_state_dev,
                token_id,
                self.position,
                &predicted_experts,
                &self.expert_reader,
                &mut self.operators,
            )?;
            self.observe_moe_step(layer_idx, ExpertAccessPhase::Decode, &step.moe);
            hc_state_dev = step.hc_state;
            trace.push(self.operators.cuda_download_f32(&hc_state_dev)?);
        }
        self.position += 1;
        Ok(trace)
    }

    /// Convenience: run the last-token HC trace through hc_head + output_norm
    /// + lm_head top-k, returning the top-1 token id.  Mirrors what
    /// `prefill_tokens_topk_batched` does with the final hidden.
    pub fn topk_from_hc_trace(&mut self, hc_state: &[f32]) -> Result<Vec<DeepSeekV4Logit>> {
        let hidden = self.normalized_last_hidden_profiled(hc_state, 1)?;
        self.model.topk_logits_for_hidden_with_operators(
            &hidden,
            1,
            self.options.output_head_chunk_rows,
            &mut self.operators,
        )
    }
}

/// Private helper: ensure a layer is bound and its state is initialised.
///
/// Both the batched and token-loop traces call this; it is a small extraction
/// of the common bind/state code that was previously duplicated in
/// `prefill_tokens_hc_states_batched` and `advance_token_hidden_reference`.
impl DeepSeekV4ReferenceRunner {
    fn ensure_layer_bound_and_state(&mut self, layer_idx: usize) -> Result<()> {
        if self.layers[layer_idx].is_none() {
            let bind_start = Instant::now();
            self.layers[layer_idx] = Some(self.model.bind_layer(layer_idx)?);
            self.operators
                .record_layer_bind(layer_idx, duration_us(bind_start.elapsed()));
        }
        if self.states[layer_idx].is_none() {
            let state_start = Instant::now();
            self.states[layer_idx] =
                Some(self.model.new_quality_first_layer_state_with_residency(
                    layer_idx,
                    self.options.moe_prefetch_experts,
                    self.options.moe_hotset_experts,
                )?);
            self.operators
                .record_layer_state_init(layer_idx, duration_us(state_start.elapsed()));
        }
        Ok(())
    }
}

fn push_candidate_experts(
    predicted: &mut Vec<usize>,
    excluded: &mut BTreeSet<usize>,
    candidates: impl IntoIterator<Item = usize>,
    budget: usize,
    num_experts: usize,
) {
    for expert in candidates {
        if predicted.len() >= budget {
            break;
        }
        if expert < num_experts && excluded.insert(expert) {
            predicted.push(expert);
        }
    }
}

fn duration_us(d: Duration) -> u64 {
    d.as_micros().min(u128::from(u64::MAX)) as u64
}

fn generic_token_logits(logits: Vec<DeepSeekV4Logit>) -> Vec<TokenLogit> {
    logits
        .into_iter()
        .map(|logit| TokenLogit {
            token_id: logit.token_id,
            logit: logit.logit,
        })
        .collect()
}

impl ModelRunner for DeepSeekV4ReferenceRunner {
    fn model_info(&self) -> ModelInfo {
        let mut info = self.model.model_info();
        info.backend = self.operator_backend().as_str();
        info
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.model.tokenizer.encode(text)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.model.tokenizer.decode(tokens)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.prefill_tokens_logits_batched(tokens)
    }

    fn decode_token(&mut self, token: u32) -> Result<Vec<f32>> {
        self.decode_token_logits(token)
    }

    fn reset_session(&mut self) -> Result<()> {
        self.reset()
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.model.tokenizer.eos_token_id()
    }

    fn bound_layer_count(&self) -> Option<usize> {
        Some(DeepSeekV4ReferenceRunner::bound_layer_count(self))
    }

    fn expert_report(&self) -> Option<String> {
        let stats = self.layer_runtime_stats();
        if stats.is_empty() {
            return Some("DeepSeek-V4 layers are not bound yet.\n".into());
        }
        let mut report = String::new();
        for stat in stats {
            report.push_str(&format!(
                "L{:>2}: window_kv={} compressed_kv={} indexer_kv={} resident_experts={} resident_bytes={}\n",
                stat.layer,
                stat.window_kv_len,
                stat.compressed_kv_len,
                stat.indexer_compressed_kv_len,
                stat.resident_experts,
                stat.resident_expert_bytes
            ));
        }
        Some(report)
    }
}

impl TopKModelRunner for DeepSeekV4ReferenceRunner {
    fn position(&self) -> usize {
        self.position()
    }

    fn feed_token(&mut self, token_id: u32) -> Result<()> {
        DeepSeekV4ReferenceRunner::feed_token(self, token_id)
    }

    fn prefill_tokens(&mut self, token_ids: &[u32], mode: PrefillMode) -> Result<()> {
        match mode {
            PrefillMode::Batched => self.prefill_tokens_no_logits_batched(token_ids),
            PrefillMode::Interactive => self.prefill_tokens_no_logits_interactive(token_ids),
        }
    }

    fn prefill_topk(
        &mut self,
        token_ids: &[u32],
        top_k: usize,
        mode: PrefillMode,
    ) -> Result<Vec<TokenLogit>> {
        let logits = match mode {
            PrefillMode::Batched => self.prefill_tokens_topk_batched(token_ids, top_k)?,
            PrefillMode::Interactive => self.prefill_tokens_topk_interactive(token_ids, top_k)?,
        };
        Ok(generic_token_logits(logits))
    }

    fn decode_topk(&mut self, token_id: u32, top_k: usize) -> Result<Vec<TokenLogit>> {
        self.decode_token_topk(token_id, top_k)
            .map(generic_token_logits)
    }
}
