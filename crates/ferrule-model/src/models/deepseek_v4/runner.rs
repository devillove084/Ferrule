//! DeepSeek-V4 runner: ModelRunner implementation.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::{Duration, Instant};

use crate::execution::ModelExecutionBackend;
#[cfg(feature = "cuda")]
use crate::execution::{ExecutionShapeKey, PersistentArenaPool};
use crate::moe::executor::CpuReferenceExpertExecutor;
use crate::moe::prediction::{
    ExpertAccessPhase, ExpertBatchAccessEvent, ExpertHotsetPredictor, ExpertPredictContext,
    ExpertPredictionStats,
};
use crate::moe::routed::RoutedMoeStepOutput;
use crate::moe::streaming::ExpertStreamingReader;
use crate::runner::{ModelInfo, ModelRunner, PrefillMode, TokenLogit, TopKModelRunner};
#[cfg(feature = "cuda")]
use ferrule_common::execution::ForwardMode;
use ferrule_common::{Error, Result};

use super::artifact::DeepSeekV4ArtifactModel;
use super::config::deepseek_v4_linear_activation_quantization;
#[cfg(feature = "cuda")]
use super::cuda_cache::DeepSeekV4DecodeBuffers;
#[cfg(feature = "cuda")]
use super::layer::DeepSeekV4LayerArenaVariants;
use super::layer::{DeepSeekV4LayerExpertRuntime, DeepSeekV4LayerState};
use super::operators::{
    DeepSeekV4AttentionProfileStats, DeepSeekV4LayerProfileStats, DeepSeekV4OperatorContext,
    DeepSeekV4OperatorRuntimeCounters,
};
pub use super::prepared::DeepSeekV4PrepareOptions;
use super::prepared::{prepare, DeepSeekV4ExecutionPolicy, DeepSeekV4PreparedModelPlan};
use super::sequence::{DeepSeekV4SequenceCheckpoint, DeepSeekV4SequenceExecutionState};

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

pub struct DeepSeekV4Runner {
    plan: DeepSeekV4PreparedModelPlan,
    operators: DeepSeekV4OperatorContext,
    /// Backend-global expert residency shared by all serially executed sequences.
    expert_runtimes: Vec<DeepSeekV4LayerExpertRuntime>,
    #[cfg(feature = "cuda")]
    layer_arena_pool: PersistentArenaPool<ExecutionShapeKey, DeepSeekV4LayerArenaVariants>,
    /// E3: per-sequence state. The runner wraps one default sequence.
    sequence: DeepSeekV4SequenceExecutionState,
    prefill_stats: DeepSeekV4PrefillRuntimeStats,
    output_profile: DeepSeekV4OutputProfileStats,
    expert_reader: ExpertStreamingReader,
    expert_executor: CpuReferenceExpertExecutor,
    shutdown: bool,
}

#[derive(Debug, Clone, Copy)]
enum ExpertPredictionInput<'a> {
    Prefill { token_ids: &'a [u32] },
    Decode { token_id: u32 },
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
enum CudaDecodeCompletion {
    Feed,
    DownloadHidden,
    TopK(usize),
}

#[cfg(feature = "cuda")]
enum CudaDecodeOutput {
    Feed,
    Hidden(Vec<f32>),
    TopK(Vec<TokenLogit>),
}

impl ExpertPredictionInput<'_> {
    fn phase(self) -> ExpertAccessPhase {
        match self {
            Self::Prefill { .. } => ExpertAccessPhase::Prefill,
            Self::Decode { .. } => ExpertAccessPhase::Decode,
        }
    }
}

impl DeepSeekV4Runner {
    pub fn new(model: DeepSeekV4ArtifactModel, options: DeepSeekV4PrepareOptions) -> Result<Self> {
        Self::new_with_operator_backend(model, options, ModelExecutionBackend::Cpu)
    }

    pub fn new_with_operator_backend(
        model: DeepSeekV4ArtifactModel,
        options: DeepSeekV4PrepareOptions,
        operator_backend: ModelExecutionBackend,
    ) -> Result<Self> {
        let plan = prepare(model, options)?;
        let options = *plan.resources().prepare_options();
        let policy = plan.resources().policy();
        let model = plan.resources().model();
        let mut operators = DeepSeekV4OperatorContext::new(operator_backend, policy)?;
        let mut layer_states = Vec::with_capacity(options.max_layers);
        let mut expert_runtimes = Vec::with_capacity(options.max_layers);
        for layer_idx in 0..options.max_layers {
            let state_start = Instant::now();
            layer_states.push(model.new_layer_sequence_state(layer_idx)?);
            expert_runtimes.push(model.new_quality_first_layer_expert_runtime_with_residency(
                layer_idx,
                options.moe_prefetch_experts,
                options.moe_hotset_experts,
            )?);
            operators.record_layer_state_init(layer_idx, duration_us(state_start.elapsed()));
        }

        let sequence =
            DeepSeekV4SequenceExecutionState::new(layer_states, model.config.num_routed_experts);
        let swiglu_limit = model.config.swiglu_limit;
        let expert_reader_max_tensor_bytes = options.expert_reader_max_tensor_bytes;
        Ok(Self {
            plan,
            operators,
            expert_runtimes,
            #[cfg(feature = "cuda")]
            layer_arena_pool: PersistentArenaPool::new(),
            sequence,
            prefill_stats: DeepSeekV4PrefillRuntimeStats::default(),
            output_profile: DeepSeekV4OutputProfileStats::default(),
            expert_reader: ExpertStreamingReader::new(expert_reader_max_tensor_bytes),
            expert_executor: CpuReferenceExpertExecutor::new(swiglu_limit)
                .with_activation_quantization(deepseek_v4_linear_activation_quantization()),
            shutdown: false,
        })
    }

    pub fn load_hf_with_options(
        model_dir: &Path,
        max_tensor_bytes: u64,
        options: DeepSeekV4PrepareOptions,
    ) -> Result<Self> {
        Self::new(
            DeepSeekV4ArtifactModel::load_hf_with_limit(model_dir, max_tensor_bytes)?,
            options,
        )
    }

    pub fn load_hf_with_options_and_backend(
        model_dir: &Path,
        max_tensor_bytes: u64,
        options: DeepSeekV4PrepareOptions,
        operator_backend: ModelExecutionBackend,
    ) -> Result<Self> {
        Self::new_with_operator_backend(
            DeepSeekV4ArtifactModel::load_hf_with_limit(model_dir, max_tensor_bytes)?,
            options,
            operator_backend,
        )
    }

    pub fn model(&self) -> &DeepSeekV4ArtifactModel {
        self.plan.resources().model()
    }

    pub fn prepare_options(&self) -> &DeepSeekV4PrepareOptions {
        self.plan.resources().prepare_options()
    }

    pub fn execution_policy(&self) -> &DeepSeekV4ExecutionPolicy {
        self.plan.resources().policy()
    }

    pub fn operator_backend(&self) -> ModelExecutionBackend {
        self.operators.backend()
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_failpoints(&self) -> Result<&ferrule_cuda::CudaFailpoints> {
        self.operators.cuda_failpoints()
    }

    pub fn operator_runtime_counters(&self) -> DeepSeekV4OperatorRuntimeCounters {
        let mut counters = self.operators.runtime_counters();
        counters.expert_predictor_stats = self.sequence.predictor.stats();
        counters
    }

    pub fn prefill_runtime_stats(&self) -> DeepSeekV4PrefillRuntimeStats {
        self.prefill_stats
    }

    pub fn take_parity_checkpoints(&mut self) -> BTreeMap<String, Vec<f32>> {
        self.operators.take_parity_checkpoints()
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
        self.sequence.core.position()
    }

    /// Create an independent checkpoint, including D2D copies of physical CUDA KV.
    pub fn checkpoint_sequence_state(&mut self) -> Result<DeepSeekV4SequenceCheckpoint> {
        self.sequence.begin_step()?;
        let layers = clone_sequence_layers(&mut self.operators, &self.sequence.layers)?;
        Ok(DeepSeekV4SequenceCheckpoint {
            core: self.sequence.core.clone(),
            layers,
            predictor: self.sequence.predictor.clone(),
        })
    }

    /// Restore a checkpoint without sharing its physical CUDA buffers.
    pub fn restore_sequence_state(
        &mut self,
        checkpoint: &DeepSeekV4SequenceCheckpoint,
    ) -> Result<()> {
        if checkpoint.layers.len() != self.sequence.max_layers() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 checkpoint layer count {} does not match runner layer count {}",
                checkpoint.layers.len(),
                self.sequence.max_layers()
            )));
        }
        let layers = clone_sequence_layers(&mut self.operators, &checkpoint.layers)?;
        self.sequence.layers = layers;
        self.sequence.predictor = checkpoint.predictor.clone();
        self.sequence.core.restore_from(&checkpoint.core);
        Ok(())
    }

    /// Fork the default runner sequence, including independent physical CUDA KV.
    pub fn fork_sequence_state(&mut self) -> Result<DeepSeekV4SequenceExecutionState> {
        self.sequence.begin_step()?;
        Ok(DeepSeekV4SequenceExecutionState {
            core: self.sequence.core.forked()?,
            layers: clone_sequence_layers(&mut self.operators, &self.sequence.layers)?,
            predictor: self.sequence.predictor.clone(),
        })
    }

    /// Execute serially with an explicit sequence while retaining the runner's
    /// prepared layers, weights, expert residency, and backend scratch resources.
    pub fn with_sequence_state<T>(
        &mut self,
        state: &mut DeepSeekV4SequenceExecutionState,
        execute: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        if self.shutdown {
            return Err(Error::Model("DeepSeek-V4 runner is shut down".into()));
        }
        if state.max_layers() != self.sequence.max_layers() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 sequence layer count {} does not match runner layer count {}",
                state.max_layers(),
                self.sequence.max_layers()
            )));
        }
        state.begin_step()?;
        std::mem::swap(&mut self.sequence, state);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(self)));
        std::mem::swap(&mut self.sequence, state);
        match result {
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    pub fn release_sequence_state(
        &mut self,
        mut state: DeepSeekV4SequenceExecutionState,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.operators.backend() == ModelExecutionBackend::Cuda {
            self.operators.cuda_mut()?.ops.sync_stream()?;
        }
        state.release_capacity();
        Ok(())
    }

    pub fn shutdown(&mut self) -> Result<()> {
        if self.shutdown {
            return Ok(());
        }
        self.sequence.release_capacity();
        self.expert_runtimes.clear();
        #[cfg(feature = "cuda")]
        self.layer_arena_pool.clear();
        self.operators.shutdown()?;
        self.shutdown = true;
        Ok(())
    }

    fn run_sequence_step<T>(
        &mut self,
        rows: usize,
        step: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        if self.shutdown {
            return Err(Error::Model("DeepSeek-V4 runner is shut down".into()));
        }
        let binding = self.sequence.begin_step()?;
        match step(self) {
            Ok(output) => {
                if let Err(error) = self.sequence.commit_step(binding, rows) {
                    self.sequence.poison_step(binding);
                    return Err(error);
                }
                Ok(output)
            }
            Err(error) => {
                self.sequence.poison_step(binding);
                Err(error)
            }
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        if self.shutdown {
            return Err(Error::Model("DeepSeek-V4 runner is shut down".into()));
        }
        self.sequence.reset_for_reuse();
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
            .plan
            .resources()
            .prepare_options()
            .moe_prefetch_experts
            .min(self.plan.resources().model().config.num_routed_experts);
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
            && self.operators.backend() == ModelExecutionBackend::Cuda
            && layer < self.plan.resources().model().config.num_hash_layers
        {
            let layer_ref = self.plan.resources().layers().get(layer).ok_or_else(|| {
                Error::Internal(format!("DeepSeek-V4 prepared layer {layer} is missing"))
            })?;
            match input {
                ExpertPredictionInput::Decode { token_id } => {
                    if let Some(hash_experts) = layer_ref.router.hash_experts_for_token(token_id)? {
                        push_candidate_experts(
                            &mut predicted,
                            &mut excluded,
                            hash_experts
                                .into_iter()
                                .take(self.plan.resources().model().config.num_experts_per_tok),
                            count,
                            self.plan.resources().model().config.num_routed_experts,
                        );
                    }
                }
                ExpertPredictionInput::Prefill { token_ids }
                    if self.prefill_hash_lookahead_enabled() =>
                {
                    if let Some(hash_experts) = layer_ref.router.hash_expert_union_for_tokens(
                        token_ids,
                        self.plan.resources().model().config.num_experts_per_tok,
                        count,
                    )? {
                        push_candidate_experts(
                            &mut predicted,
                            &mut excluded,
                            hash_experts,
                            count,
                            self.plan.resources().model().config.num_routed_experts,
                        );
                    }
                }
                ExpertPredictionInput::Prefill { .. } => {}
            }
        }

        if predicted.len() < count {
            for prediction in self.sequence.predictor.predict(ExpertPredictContext {
                layer,
                phase,
                budget: count,
                num_experts: self.plan.resources().model().config.num_routed_experts,
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
            let expert_runtime = self.expert_runtimes.get(layer).ok_or_else(|| {
                Error::Internal(format!(
                    "DeepSeek-V4 layer expert runtime {layer} is missing"
                ))
            })?;
            push_candidate_experts(
                &mut predicted,
                &mut excluded,
                expert_runtime.expert_planner.hot_experts(
                    layer,
                    self.plan.resources().model().config.num_routed_experts,
                ),
                count,
                self.plan.resources().model().config.num_routed_experts,
            );
        }
        Ok(predicted)
    }

    fn observe_moe_step(
        &mut self,
        layer: usize,
        phase: ExpertAccessPhase,
        moe: &RoutedMoeStepOutput,
    ) {
        self.sequence
            .predictor
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
            self.sequence.predictor.observe_batch(event);
        }
    }

    fn lookahead_prefetch_enabled(&self) -> bool {
        self.plan.resources().policy().lookahead_prefetch()
    }

    #[cfg(feature = "cuda")]
    fn hash_decode_prefetch_window_enabled(&self) -> bool {
        self.plan.resources().policy().hash_decode_prefetch_window()
    }

    #[cfg(feature = "cuda")]
    fn decode_lookahead_prefetch_enabled(&self) -> bool {
        self.plan.resources().policy().decode_lookahead_prefetch()
    }

    fn prefill_hash_lookahead_enabled(&self) -> bool {
        self.plan.resources().policy().prefill_hash_lookahead()
    }

    fn prepare_predicted_experts_for_layer(
        &mut self,
        layer: usize,
        predicted_experts: &[usize],
    ) -> Result<()> {
        if predicted_experts.is_empty()
            || self.plan.resources().prepare_options().moe_prefetch_experts == 0
            || !self.lookahead_prefetch_enabled()
        {
            return Ok(());
        }
        let expert_runtime = self.expert_runtimes.get_mut(layer).ok_or_else(|| {
            Error::Internal(format!(
                "DeepSeek-V4 layer expert runtime {layer} is missing"
            ))
        })?;
        self.operators.prefetch_predicted_experts(
            layer,
            predicted_experts,
            &mut expert_runtime.expert_planner,
            &self.expert_reader,
            &mut expert_runtime.expert_handles,
        )?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn prepare_hash_decode_prefetch_window(&mut self, token_id: u32) -> Result<()> {
        if self.operators.backend() != ModelExecutionBackend::Cuda
            || self.plan.resources().prepare_options().moe_prefetch_experts == 0
            || !self.lookahead_prefetch_enabled()
            || !self.hash_decode_prefetch_window_enabled()
        {
            return Ok(());
        }
        let layers = self
            .plan
            .resources()
            .prepare_options()
            .max_layers
            .min(self.plan.resources().model().config.num_hash_layers);
        for layer_idx in 0..layers {
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
        if self.operators.backend() != ModelExecutionBackend::Cuda
            || self.plan.resources().prepare_options().moe_prefetch_experts == 0
            || !self.lookahead_prefetch_enabled()
            || !self.decode_lookahead_prefetch_enabled()
        {
            return Ok(None);
        }

        let mut predictions =
            Vec::with_capacity(self.plan.resources().prepare_options().max_layers);
        for layer_idx in 0..self.plan.resources().prepare_options().max_layers {
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
        self.sequence.predictor.stats()
    }

    #[cfg(feature = "cuda")]
    pub fn prewarm_predicted_experts(&mut self) -> Result<usize> {
        if self.operators.backend() != ModelExecutionBackend::Cuda {
            return Ok(0);
        }
        let count = self
            .plan
            .resources()
            .prepare_options()
            .moe_prefetch_experts
            .max(self.plan.resources().prepare_options().moe_hotset_experts)
            .min(self.plan.resources().model().config.num_routed_experts);
        if count == 0 {
            return Ok(0);
        }
        let mut warmed = 0usize;
        for layer_idx in 0..self.plan.resources().prepare_options().max_layers {
            let predicted = if count >= self.plan.resources().model().config.num_routed_experts {
                (0..self.plan.resources().model().config.num_routed_experts).collect::<Vec<_>>()
            } else {
                self.expert_runtimes[layer_idx]
                    .expert_planner
                    .hot_experts(layer_idx, count)
            };
            if predicted.is_empty() {
                continue;
            }
            let expert_runtime = &mut self.expert_runtimes[layer_idx];
            warmed = warmed.saturating_add(self.operators.prewarm_experts(
                layer_idx,
                &predicted,
                &mut expert_runtime.expert_planner,
                &self.expert_reader,
                &mut expert_runtime.expert_handles,
            )?);
        }
        Ok(warmed)
    }

    pub fn bound_layer_count(&self) -> usize {
        self.plan.resources().layers().len()
    }

    pub fn layer_runtime_stats(&self) -> Vec<DeepSeekV4LayerRuntimeStats> {
        let mut stats = Vec::new();
        for layer_idx in 0..self.plan.resources().prepare_options().max_layers {
            let layer = &self.plan.resources().layers()[layer_idx];
            let state = &self.sequence.layers[layer_idx];
            let expert_runtime = &self.expert_runtimes[layer_idx];
            let index_head_dim = layer.attention.config.index_head_dim;
            let (resident_experts, resident_expert_bytes) = {
                #[cfg(feature = "cuda")]
                {
                    if let Some(cache) = self.operators.cuda.as_ref() {
                        cache.resident_expert_stats_for_layer(layer_idx)
                    } else {
                        (
                            expert_runtime.expert_handles.len(),
                            expert_runtime.expert_handles.total_bytes(),
                        )
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    (
                        expert_runtime.expert_handles.len(),
                        expert_runtime.expert_handles.total_bytes(),
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
        if self.operators.backend == ModelExecutionBackend::Cuda {
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
        self.run_sequence_step(1, |runner| {
            runner.advance_token_hidden_reference_uncommitted(token_id, materialize_hidden)
        })
    }

    fn advance_token_hidden_reference_uncommitted(
        &mut self,
        token_id: u32,
        materialize_hidden: bool,
    ) -> Result<Option<Vec<f32>>> {
        let mut hc_state = self
            .plan
            .resources()
            .model()
            .initial_hc_state_for_token(token_id)?;
        for layer_idx in 0..self.plan.resources().prepare_options().max_layers {
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Decode { token_id },
            )?;
            self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            let layer = &self.plan.resources().layers()[layer_idx];
            let state = &mut self.sequence.layers[layer_idx];
            let expert_runtime = &mut self.expert_runtimes[layer_idx];
            let step = layer.decode_step_with_operators(
                state,
                expert_runtime,
                &hc_state,
                token_id,
                self.sequence.core.position(),
                &predicted_experts,
                &self.expert_reader,
                &self.expert_executor,
                &mut self.operators,
            )?;
            self.observe_moe_step(layer_idx, ExpertAccessPhase::Decode, &step.moe);
            hc_state = step.hc_state;
        }
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
        if tokens == 0
            || hc_state.len()
                != tokens
                    * self
                        .plan
                        .resources()
                        .model()
                        .config
                        .hc_config()
                        .hc_hidden_size()
        {
            return Err(Error::Model(format!(
                "DeepSeek-V4 HC head input mismatch: tokens={tokens} len={} expected {}",
                hc_state.len(),
                tokens
                    * self
                        .plan
                        .resources()
                        .model()
                        .config
                        .hc_config()
                        .hc_hidden_size()
            )));
        }

        let hc_head_start = Instant::now();
        let hidden = {
            #[cfg(feature = "cuda")]
            if self.operators.backend() == ModelExecutionBackend::Cuda {
                let hidden_len = tokens * self.plan.resources().model().config.hidden_size;
                let mut buffers = self
                    .operators
                    .cuda_mut()?
                    .take_decode_buffers(hc_state.len(), hidden_len)?;
                let execution = (|| {
                    self.operators
                        .cuda_mut()?
                        .ops
                        .overwrite_f32_buffer(hc_state, &mut buffers.hc_input)?;
                    self.operators.cuda_mut()?.hc_head_from_device_into(
                        &buffers.hc_input,
                        tokens,
                        self.plan.resources().model().config.hc_config(),
                        &self.plan.resources().model().hc_head,
                        &mut buffers.final_hidden,
                    )?;
                    self.operators
                        .cuda_mut()?
                        .ops
                        .download_f32_buffer(&buffers.final_hidden)
                })();
                self.operators.cuda_mut()?.restore_decode_buffers(buffers);
                execution?
            } else {
                self.operators.hc_head(
                    hc_state,
                    tokens,
                    self.plan.resources().model().config.hc_config(),
                    &self.plan.resources().model().hc_head,
                )?
            }
            #[cfg(not(feature = "cuda"))]
            {
                self.operators.hc_head(
                    hc_state,
                    tokens,
                    self.plan.resources().model().config.hc_config(),
                    &self.plan.resources().model().hc_head,
                )?
            }
        };
        self.output_profile.final_hc_head_calls =
            self.output_profile.final_hc_head_calls.saturating_add(1);
        self.output_profile.final_hc_head_us = self
            .output_profile
            .final_hc_head_us
            .saturating_add(self.operators.finish_profile_stage(hc_head_start)?);

        let start = (tokens - 1) * self.plan.resources().model().config.hidden_size;
        let norm_start = Instant::now();
        let normed = self.operators.rms_norm(
            &hidden[start..start + self.plan.resources().model().config.hidden_size],
            &self.plan.resources().model().output_norm,
            self.plan.resources().model().config.norm_eps,
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
        match self.run_cuda_decode(token_id, CudaDecodeCompletion::DownloadHidden)? {
            CudaDecodeOutput::Hidden(hidden) => Ok(hidden),
            _ => Err(Error::Internal(
                "DeepSeek-V4 CUDA hidden decode returned the wrong completion".into(),
            )),
        }
    }

    #[cfg(feature = "cuda")]
    fn restore_cuda_decode_buffers(&mut self, buffers: DeepSeekV4DecodeBuffers) {
        self.operators
            .cuda
            .as_mut()
            .expect("CUDA cache exists for CUDA decode")
            .restore_decode_buffers(buffers);
    }

    #[cfg(feature = "cuda")]
    fn run_cuda_decode(
        &mut self,
        token_id: u32,
        completion: CudaDecodeCompletion,
    ) -> Result<CudaDecodeOutput> {
        self.run_sequence_step(1, |runner| {
            runner.run_cuda_decode_uncommitted(token_id, completion)
        })
    }

    #[cfg(feature = "cuda")]
    fn run_cuda_decode_uncommitted(
        &mut self,
        token_id: u32,
        completion: CudaDecodeCompletion,
    ) -> Result<CudaDecodeOutput> {
        let hc_state = self
            .plan
            .resources()
            .model()
            .initial_hc_state_for_token(token_id)?;
        let max_layers = self.plan.resources().prepare_options().max_layers;
        let decode_lookahead_predictions = self.prepare_decode_lookahead_prefetch(token_id)?;
        self.prepare_hash_decode_prefetch_window(token_id)?;
        let mut layer_predictions = Vec::with_capacity(max_layers);
        for layer_idx in 0..max_layers {
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Decode { token_id },
            )?;
            // The all-layer lookahead above is deliberately predictor/hotset-only.
            // Keep the original per-layer prepare so current-token hash experts are
            // still available to the actual MoE plan. All preparation happens before
            // the arena lease so one lease can cover the complete execution step.
            if decode_lookahead_predictions.is_none() || !predicted_experts.is_empty() {
                self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            }
            layer_predictions.push(predicted_experts);
        }

        self.operators.check_cuda_arena_acquire()?;
        let arena_key = ExecutionShapeKey::new(ForwardMode::Decode, 1, 1, 1);
        let mut arena_lease = match {
            let layers = self.plan.resources().layers();
            let operators = &mut self.operators;
            self.layer_arena_pool.acquire(arena_key, || {
                DeepSeekV4LayerArenaVariants::try_build(layers, 1, operators)
            })
        } {
            Ok(lease) => lease,
            Err(error) => {
                self.operators.record_cuda_arena_pool_miss(false)?;
                return Err(error);
            }
        };
        if arena_lease.reused() {
            self.operators.record_cuda_arena_pool_hit()?;
        } else {
            self.operators.record_cuda_arena_pool_miss(true)?;
        }

        let hidden_size = self.plan.resources().model().config.hidden_size;
        let mut decode_buffers = self
            .operators
            .cuda_mut()?
            .take_decode_buffers(hc_state.len(), hidden_size)?;
        let layer_result: Result<Vec<RoutedMoeStepOutput>> = {
            let layers = self.plan.resources().layers();
            let position = self.sequence.core.position();
            let states = &mut self.sequence.layers;
            let expert_runtimes = &mut self.expert_runtimes;
            let expert_reader = &self.expert_reader;
            let operators = &mut self.operators;
            (|| {
                operators
                    .cuda_mut()?
                    .ops
                    .overwrite_f32_buffer(&hc_state, &mut decode_buffers.hc_input)?;
                let mut layer_moe_steps = Vec::with_capacity(max_layers);
                for layer_idx in 0..max_layers {
                    let arena = arena_lease
                        .get_mut()
                        .get_for_layer_mut(layer_idx)
                        .expect("pooled layer arena variants match prepared layers");
                    let step = layers[layer_idx].decode_step_device_hc_device(
                        &mut states[layer_idx],
                        &mut expert_runtimes[layer_idx],
                        arena,
                        &mut decode_buffers.hc_input,
                        token_id,
                        position,
                        &layer_predictions[layer_idx],
                        expert_reader,
                        operators,
                    )?;
                    layer_moe_steps.push(step.moe);
                }
                Ok(layer_moe_steps)
            })()
        };
        drop(arena_lease);
        let layer_moe_steps = match layer_result {
            Ok(steps) => steps,
            Err(error) => {
                self.restore_cuda_decode_buffers(decode_buffers);
                return Err(error);
            }
        };
        for (layer_idx, moe) in layer_moe_steps.iter().enumerate() {
            self.observe_moe_step(layer_idx, ExpertAccessPhase::Decode, moe);
        }

        let result = (|| {
            if matches!(completion, CudaDecodeCompletion::Feed) {
                return Ok(CudaDecodeOutput::Feed);
            }

            let final_hc_start = Instant::now();
            self.operators.cuda_mut()?.hc_head_from_device_into(
                &decode_buffers.hc_input,
                1,
                self.plan.resources().model().config.hc_config(),
                &self.plan.resources().model().hc_head,
                &mut decode_buffers.final_hidden,
            )?;
            self.output_profile.final_hc_head_calls =
                self.output_profile.final_hc_head_calls.saturating_add(1);
            self.output_profile.final_hc_head_us = self
                .output_profile
                .final_hc_head_us
                .saturating_add(self.operators.finish_profile_stage(final_hc_start)?);

            let final_norm_start = Instant::now();
            self.operators.cuda_mut()?.rms_norm_device_cached_into(
                "output_norm",
                &decode_buffers.final_hidden,
                &self.plan.resources().model().output_norm,
                self.plan.resources().model().config.norm_eps,
                &mut decode_buffers.final_norm,
            )?;
            self.output_profile.final_norm_calls =
                self.output_profile.final_norm_calls.saturating_add(1);
            self.output_profile.final_norm_us = self
                .output_profile
                .final_norm_us
                .saturating_add(self.operators.finish_profile_stage(final_norm_start)?);

            match completion {
                CudaDecodeCompletion::Feed => unreachable!("handled before finalization"),
                CudaDecodeCompletion::DownloadHidden => self
                    .operators
                    .cuda_mut()?
                    .ops
                    .download_f32_buffer(&decode_buffers.final_norm)
                    .map(CudaDecodeOutput::Hidden),
                CudaDecodeCompletion::TopK(top_k) => {
                    let topk_start = Instant::now();
                    let logits = self
                        .plan
                        .resources()
                        .model()
                        .topk_logits_for_hidden_device_with_operators(
                            &decode_buffers.final_norm,
                            top_k,
                            self.plan
                                .resources()
                                .prepare_options()
                                .output_head_chunk_rows,
                            &mut self.operators,
                        )?;
                    self.output_profile.lm_head_topk_calls =
                        self.output_profile.lm_head_topk_calls.saturating_add(1);
                    self.output_profile.lm_head_topk_us = self
                        .output_profile
                        .lm_head_topk_us
                        .saturating_add(self.operators.finish_profile_stage(topk_start)?);
                    Ok(CudaDecodeOutput::TopK(logits))
                }
            }
        })();
        self.restore_cuda_decode_buffers(decode_buffers);
        result
    }

    pub fn decode_token_logits_row_range(
        &mut self,
        token_id: u32,
        start_row: usize,
        row_count: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.decode_token_hidden(token_id)?;
        self.plan
            .resources()
            .model()
            .logits_for_hidden_row_range_with_operators(
                &hidden,
                start_row,
                row_count,
                &mut self.operators,
            )
    }

    pub fn decode_token_logits(&mut self, token_id: u32) -> Result<Vec<f32>> {
        let hidden = self.decode_token_hidden(token_id)?;
        self.plan
            .resources()
            .model()
            .logits_for_hidden_chunked_with_operators(
                &hidden,
                self.plan
                    .resources()
                    .prepare_options()
                    .output_head_chunk_rows,
                &mut self.operators,
            )
    }

    pub fn decode_token_topk(&mut self, token_id: u32, top_k: usize) -> Result<Vec<TokenLogit>> {
        #[cfg(feature = "cuda")]
        if self.operators.backend == ModelExecutionBackend::Cuda {
            return match self.run_cuda_decode(token_id, CudaDecodeCompletion::TopK(top_k))? {
                CudaDecodeOutput::TopK(logits) => Ok(logits),
                _ => Err(Error::Internal(
                    "DeepSeek-V4 CUDA top-k decode returned the wrong completion".into(),
                )),
            };
        }

        let hidden = self.decode_token_hidden(token_id)?;
        let topk_start = Instant::now();
        let logits = self
            .plan
            .resources()
            .model()
            .topk_logits_for_hidden_with_operators(
                &hidden,
                top_k,
                self.plan
                    .resources()
                    .prepare_options()
                    .output_head_chunk_rows,
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
        if self.operators.backend == ModelExecutionBackend::Cuda {
            return match self.run_cuda_decode(token_id, CudaDecodeCompletion::Feed)? {
                CudaDecodeOutput::Feed => Ok(()),
                _ => Err(Error::Internal(
                    "DeepSeek-V4 CUDA feed returned the wrong completion".into(),
                )),
            };
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
        self.plan
            .resources()
            .model()
            .logits_for_hidden_row_range_with_operators(
                &hidden,
                start_row,
                row_count,
                &mut self.operators,
            )
    }

    pub fn prefill_tokens_logits(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.prefill_tokens_hidden(token_ids)?;
        self.plan
            .resources()
            .model()
            .logits_for_hidden_chunked_with_operators(
                &hidden,
                self.plan
                    .resources()
                    .prepare_options()
                    .output_head_chunk_rows,
                &mut self.operators,
            )
    }

    pub fn prefill_tokens_topk(
        &mut self,
        token_ids: &[u32],
        top_k: usize,
    ) -> Result<Vec<TokenLogit>> {
        let hidden = self.prefill_tokens_hidden(token_ids)?;
        self.plan
            .resources()
            .model()
            .topk_logits_for_hidden_with_operators(
                &hidden,
                top_k,
                self.plan
                    .resources()
                    .prepare_options()
                    .output_head_chunk_rows,
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
    ) -> Result<Vec<TokenLogit>> {
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
        self.run_sequence_step(token_ids.len(), |runner| {
            runner.prefill_tokens_hc_states_batched_uncommitted(token_ids)
        })
    }

    fn prefill_tokens_hc_states_batched_uncommitted(
        &mut self,
        token_ids: &[u32],
    ) -> Result<(Vec<f32>, usize)> {
        if token_ids.is_empty() {
            return Err(Error::Model(
                "DeepSeek-V4 prefill requires at least one token".into(),
            ));
        }
        let tokens = token_ids.len();
        let start_pos = self.sequence.core.position();
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

        // ── Slice D: cross-layer device-resident prefill ──
        // Keep HC/hidden rows on device across all layers. Only download the
        // final HC state when the caller needs it as Vec<f32>.
        #[cfg(feature = "cuda")]
        if self.operators.backend == ModelExecutionBackend::Cuda {
            let hc_state = self.prefill_tokens_hc_states_device(token_ids, start_pos, None)?;
            return Ok((hc_state, tokens));
        }

        let mut hc_state = self
            .plan
            .resources()
            .model()
            .initial_hc_state_for_tokens(token_ids)?;
        let report_progress = self.plan.resources().policy().prefill_progress();
        for layer_idx in 0..self.plan.resources().prepare_options().max_layers {
            let layer_start = report_progress.then(Instant::now);
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Prefill { token_ids },
            )?;
            self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            let layer = &self.plan.resources().layers()[layer_idx];
            let state = &mut self.sequence.layers[layer_idx];
            let expert_runtime = &mut self.expert_runtimes[layer_idx];
            hc_state = layer.prefill_start_with_operators(
                state,
                expert_runtime,
                &hc_state,
                token_ids,
                start_pos,
                &predicted_experts,
                &self.expert_reader,
                &self.expert_executor,
                &mut self.operators,
            )?;
            self.observe_pending_moe_access_events();
            if let Some(layer_start) = layer_start {
                let counters = self.operator_runtime_counters();
                eprintln!(
                    "[ferrule] DSV4 prefill layer {}/{} complete in {:?}: kernels={} allocations={} h2d={}B d2h={}B expert_loads={} expert_load_bytes={}",
                    layer_idx + 1,
                    self.plan.resources().prepare_options().max_layers,
                    layer_start.elapsed(),
                    counters.kernel_launches,
                    counters.device_allocations,
                    counters.host_to_device_bytes,
                    counters.device_to_host_bytes,
                    counters.expert_loads,
                    counters.expert_load_bytes,
                );
            }
        }
        Ok((hc_state, tokens))
    }

    /// Cross-layer device-resident prefill (Slice D).
    ///
    /// Executes one exact-shape prefill bucket through the shared persistent
    /// layer arena. Trace downloads are opt-in and never affect normal execution.
    #[cfg(feature = "cuda")]
    fn prefill_tokens_hc_states_device(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        mut layer_trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Vec<f32>> {
        let hc_state = self
            .plan
            .resources()
            .model()
            .initial_hc_state_for_tokens(token_ids)?;
        let tokens = token_ids.len();
        let max_layers = self.plan.resources().prepare_options().max_layers;
        let mut layer_predictions = Vec::with_capacity(max_layers);
        for layer_idx in 0..max_layers {
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Prefill { token_ids },
            )?;
            self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            layer_predictions.push(predicted_experts);
        }

        self.operators.check_cuda_arena_acquire()?;
        let arena_key = ExecutionShapeKey::new(ForwardMode::Prefill, tokens, 1, tokens);
        let mut arena_lease = match {
            let layers = self.plan.resources().layers();
            let operators = &mut self.operators;
            self.layer_arena_pool.acquire(arena_key, || {
                DeepSeekV4LayerArenaVariants::try_build(layers, tokens, operators)
            })
        } {
            Ok(lease) => lease,
            Err(error) => {
                self.operators.record_cuda_arena_pool_miss(false)?;
                return Err(error);
            }
        };
        if arena_lease.reused() {
            self.operators.record_cuda_arena_pool_hit()?;
        } else {
            self.operators.record_cuda_arena_pool_miss(true)?;
        }

        let hidden_size = self.plan.resources().model().config.hidden_size;
        let mut buffers = self
            .operators
            .cuda_mut()?
            .take_decode_buffers(hc_state.len(), tokens * hidden_size)?;
        let execution = (|| {
            self.operators
                .cuda_mut()?
                .ops
                .overwrite_f32_buffer(&hc_state, &mut buffers.hc_input)?;
            for layer_idx in 0..max_layers {
                let layer = &self.plan.resources().layers()[layer_idx];
                let arena = arena_lease
                    .get_mut()
                    .get_for_layer_mut(layer_idx)
                    .expect("pooled layer arena variants match prepared layers");
                layer.prefill_start_cuda_device_chain_into(
                    &mut self.sequence.layers[layer_idx],
                    &mut self.expert_runtimes[layer_idx],
                    arena,
                    &mut buffers.hc_input,
                    token_ids,
                    start_pos,
                    &layer_predictions[layer_idx],
                    &self.expert_reader,
                    &mut self.operators,
                )?;
                if let Some(trace) = layer_trace.as_deref_mut() {
                    let host = self
                        .operators
                        .cuda_mut()?
                        .ops
                        .download_f32_buffer(&buffers.hc_input)?;
                    let hc_dim = layer.hc_config.hc_hidden_size();
                    let last_start = (tokens - 1) * hc_dim;
                    trace.push(host[last_start..last_start + hc_dim].to_vec());
                }
            }
            self.operators
                .cuda_mut()?
                .ops
                .download_f32_buffer(&buffers.hc_input)
        })();
        drop(arena_lease);
        self.operators.cuda_mut()?.restore_decode_buffers(buffers);
        let hc_state = execution?;
        self.observe_pending_moe_access_events();
        Ok(hc_state)
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
        self.plan
            .resources()
            .model()
            .logits_for_hidden_chunked_with_operators(
                &hidden,
                self.plan
                    .resources()
                    .prepare_options()
                    .output_head_chunk_rows,
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
        self.plan
            .resources()
            .model()
            .logits_for_hidden_row_range_with_operators(
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
    ) -> Result<Vec<TokenLogit>> {
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
        let logits = self
            .plan
            .resources()
            .model()
            .topk_logits_for_hidden_with_operators(
                &hidden,
                top_k,
                self.plan
                    .resources()
                    .prepare_options()
                    .output_head_chunk_rows,
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
        // Single-token: the batched path falls back to the token-loop, so the
        // two traces are identical by construction.
        if tokens == 1 {
            return self.prefill_token_loop_layer_hc_trace(token_ids);
        }
        self.run_sequence_step(tokens, |runner| {
            runner.prefill_batched_layer_hc_trace_uncommitted(token_ids)
        })
    }

    fn prefill_batched_layer_hc_trace_uncommitted(
        &mut self,
        token_ids: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        let tokens = token_ids.len();
        let start_pos = self.sequence.core.position();
        #[cfg(feature = "cuda")]
        if self.operators.backend == ModelExecutionBackend::Cuda {
            let mut trace = Vec::with_capacity(self.plan.resources().prepare_options().max_layers);
            let _hc_state =
                self.prefill_tokens_hc_states_device(token_ids, start_pos, Some(&mut trace))?;
            return Ok(trace);
        }

        let hc_dim = self
            .plan
            .resources()
            .model()
            .config
            .hc_config()
            .hc_hidden_size();
        let mut hc_state = self
            .plan
            .resources()
            .model()
            .initial_hc_state_for_tokens(token_ids)?;
        let mut trace = Vec::with_capacity(self.plan.resources().prepare_options().max_layers);
        for layer_idx in 0..self.plan.resources().prepare_options().max_layers {
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Prefill { token_ids },
            )?;
            self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            let layer = &self.plan.resources().layers()[layer_idx];
            let state = &mut self.sequence.layers[layer_idx];
            let expert_runtime = &mut self.expert_runtimes[layer_idx];
            hc_state = layer.prefill_start_with_operators(
                state,
                expert_runtime,
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
        self.run_sequence_step(1, |runner| {
            #[cfg(feature = "cuda")]
            if runner.operators.backend == ModelExecutionBackend::Cuda {
                return runner.decode_token_layer_hc_trace_cuda(token_id);
            }
            runner.decode_token_layer_hc_trace_reference(token_id)
        })
    }

    fn decode_token_layer_hc_trace_reference(&mut self, token_id: u32) -> Result<Vec<Vec<f32>>> {
        let mut hc_state = self
            .plan
            .resources()
            .model()
            .initial_hc_state_for_token(token_id)?;
        let mut trace = Vec::with_capacity(self.plan.resources().prepare_options().max_layers);
        for layer_idx in 0..self.plan.resources().prepare_options().max_layers {
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Decode { token_id },
            )?;
            self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            let layer = &self.plan.resources().layers()[layer_idx];
            let state = &mut self.sequence.layers[layer_idx];
            let expert_runtime = &mut self.expert_runtimes[layer_idx];
            let step = layer.decode_step_with_operators(
                state,
                expert_runtime,
                &hc_state,
                token_id,
                self.sequence.core.position(),
                &predicted_experts,
                &self.expert_reader,
                &self.expert_executor,
                &mut self.operators,
            )?;
            self.observe_moe_step(layer_idx, ExpertAccessPhase::Decode, &step.moe);
            hc_state = step.hc_state;
            trace.push(hc_state.clone());
        }
        Ok(trace)
    }

    #[cfg(feature = "cuda")]
    fn decode_token_layer_hc_trace_cuda(&mut self, token_id: u32) -> Result<Vec<Vec<f32>>> {
        let mut hc_state_dev = {
            let hc_state = self
                .plan
                .resources()
                .model()
                .initial_hc_state_for_token(token_id)?;
            self.operators
                .cuda_mut()?
                .ops
                .upload_f32_buffer(&hc_state)?
        };
        let max_layers = self.plan.resources().prepare_options().max_layers;
        let mut layer_predictions = Vec::with_capacity(max_layers);
        for layer_idx in 0..max_layers {
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Decode { token_id },
            )?;
            if !predicted_experts.is_empty() {
                self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            }
            layer_predictions.push(predicted_experts);
        }

        self.operators.check_cuda_arena_acquire()?;
        let arena_key = ExecutionShapeKey::new(ForwardMode::Decode, 1, 1, 1);
        let mut arena_lease = match {
            let layers = self.plan.resources().layers();
            let operators = &mut self.operators;
            self.layer_arena_pool.acquire(arena_key, || {
                DeepSeekV4LayerArenaVariants::try_build(layers, 1, operators)
            })
        } {
            Ok(lease) => lease,
            Err(error) => {
                self.operators.record_cuda_arena_pool_miss(false)?;
                return Err(error);
            }
        };
        if arena_lease.reused() {
            self.operators.record_cuda_arena_pool_hit()?;
        } else {
            self.operators.record_cuda_arena_pool_miss(true)?;
        }

        let mut trace = Vec::with_capacity(max_layers);
        let mut layer_moe_steps = Vec::with_capacity(max_layers);
        for layer_idx in 0..max_layers {
            let layer = &self.plan.resources().layers()[layer_idx];
            let state = &mut self.sequence.layers[layer_idx];
            let expert_runtime = &mut self.expert_runtimes[layer_idx];
            let arena = arena_lease
                .get_mut()
                .get_for_layer_mut(layer_idx)
                .expect("pooled layer arena variants match prepared layers");
            let step = layer.decode_step_device_hc_device(
                state,
                expert_runtime,
                arena,
                &mut hc_state_dev,
                token_id,
                self.sequence.core.position(),
                &layer_predictions[layer_idx],
                &self.expert_reader,
                &mut self.operators,
            )?;
            layer_moe_steps.push(step.moe);
            trace.push(
                self.operators
                    .cuda_mut()?
                    .ops
                    .download_f32_buffer(&hc_state_dev)?,
            );
        }
        drop(arena_lease);
        for (layer_idx, moe) in layer_moe_steps.iter().enumerate() {
            self.observe_moe_step(layer_idx, ExpertAccessPhase::Decode, moe);
        }
        Ok(trace)
    }

    /// Convenience: run the last-token HC trace through hc_head + output_norm
    /// + lm_head top-k, returning the top-1 token id.  Mirrors what
    /// `prefill_tokens_topk_batched` does with the final hidden.
    pub fn topk_from_hc_trace(&mut self, hc_state: &[f32]) -> Result<Vec<TokenLogit>> {
        let hidden = self.normalized_last_hidden_profiled(hc_state, 1)?;
        self.plan
            .resources()
            .model()
            .topk_logits_for_hidden_with_operators(
                &hidden,
                1,
                self.plan
                    .resources()
                    .prepare_options()
                    .output_head_chunk_rows,
                &mut self.operators,
            )
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

fn clone_sequence_layers(
    operators: &mut DeepSeekV4OperatorContext,
    layers: &[DeepSeekV4LayerState],
) -> Result<Vec<DeepSeekV4LayerState>> {
    let mut cloned = Vec::new();
    cloned.try_reserve_exact(layers.len()).map_err(|error| {
        Error::Model(format!(
            "DeepSeek-V4 sequence layer clone allocation failed: {error}"
        ))
    })?;
    for layer in layers {
        let destination = layer.clone();
        #[cfg(feature = "cuda")]
        let destination = {
            let mut destination = destination;
            let physical = operators
                .cuda_mut()?
                .clone_sequence_kv_state(layer.kv.window.cuda_state())?;
            destination.kv.window.replace_cuda_state(physical);
            destination
        };
        #[cfg(not(feature = "cuda"))]
        let _ = operators;
        cloned.push(destination);
    }
    Ok(cloned)
}

fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

impl ModelRunner for DeepSeekV4Runner {
    fn model_info(&self) -> ModelInfo {
        let mut info = self.plan.resources().model().model_info();
        info.backend = self.operator_backend().as_str();
        info
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.plan.resources().model().tokenizer.encode(text)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.plan.resources().model().tokenizer.decode(tokens)
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
        self.plan.resources().model().tokenizer.eos_token_id()
    }

    fn bound_layer_count(&self) -> Option<usize> {
        Some(DeepSeekV4Runner::bound_layer_count(self))
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

impl TopKModelRunner for DeepSeekV4Runner {
    fn position(&self) -> usize {
        self.sequence.core.position()
    }

    fn feed_token(&mut self, token_id: u32) -> Result<()> {
        DeepSeekV4Runner::feed_token(self, token_id)
    }

    fn max_top_k(&self) -> usize {
        40
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
        match mode {
            PrefillMode::Batched => self.prefill_tokens_topk_batched(token_ids, top_k),
            PrefillMode::Interactive => self.prefill_tokens_topk_interactive(token_ids, top_k),
        }
    }

    fn decode_topk(&mut self, token_id: u32, top_k: usize) -> Result<Vec<TokenLogit>> {
        self.decode_token_topk(token_id, top_k)
    }
}
