//! DeepSeek-V4 runner: ModelRunner implementation.

use std::collections::{BTreeMap, BTreeSet};
#[cfg(any(feature = "cuda", test))]
use std::ops::Range;
use std::path::Path;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::execution::ModelExecutionBackend;
#[cfg(feature = "cuda")]
use crate::execution::{ExecutionShapeKey, PersistentArenaPool};
use crate::moe::executor::CpuReferenceExpertExecutor;
use crate::moe::prediction::{
    ExpertAccessPhase, ExpertBatchAccessEvent, ExpertHotsetPredictor, ExpertPredictContext,
    ScoreBasedExpertPredictor,
};
use crate::moe::routed::RoutedMoeStepOutput;
use crate::moe::streaming::ExpertStreamingReader;
use crate::runner::{
    ModelInfo, ModelRunner, MultiSessionRunner, PrefillMode, TokenLogit, TopKModelRunner,
};
#[cfg(any(feature = "cuda", test))]
use ferrule_common::execution::{ExecutionBatch, ForwardMode, ForwardPhase};
#[cfg(feature = "cuda")]
use ferrule_common::execution::{
    ExecutionOutput, LogitsOutput, LogitsRequest, LogitsRow, TokenLogit as ExecutionTokenLogit,
};
use ferrule_common::expert_residency::{ExpertResidencyControl, ExpertResidencyRequirements};
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
use super::prepared::{
    DeepSeekV4ExecutionPolicy, DeepSeekV4PreparedLayerExperts, DeepSeekV4PreparedModelPlan,
    DeepSeekV4PreparedResources, prepare,
};
use super::sequence::DeepSeekV4SequenceExecutionState;

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
    pub packed_prefill_batches: u64,
    pub packed_prefill_rows: u64,
    pub packed_decode_batches: u64,
    pub packed_decode_rows: u64,
    pub packed_mixed_batches: u64,
    pub packed_mixed_rows: u64,
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
}

#[cfg(feature = "cuda")]
static NEXT_DSV4_MODEL_INSTANCE: AtomicU64 = AtomicU64::new(1);

pub struct DeepSeekV4Runner {
    plan: DeepSeekV4PreparedModelPlan,
    operators: DeepSeekV4OperatorContext,
    /// CPU/reference planner and handle stores. CUDA consumes immutable prepared
    /// catalogs while runtime owns logical slots, generations, leases, and policy.
    cpu_expert_runtimes: Option<Box<[DeepSeekV4LayerExpertRuntime]>>,
    #[cfg(feature = "cuda")]
    model_instance: u64,
    #[cfg(feature = "cuda")]
    expert_residency: Option<Box<dyn ExpertResidencyControl>>,
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

fn build_cpu_expert_runtimes(
    backend: ModelExecutionBackend,
    layers: &[DeepSeekV4PreparedLayerExperts],
) -> Option<Box<[DeepSeekV4LayerExpertRuntime]>> {
    (backend == ModelExecutionBackend::Cpu).then(|| {
        layers
            .iter()
            .map(|layer| {
                DeepSeekV4LayerExpertRuntime::from_catalog(
                    std::sync::Arc::clone(layer.source_catalog()),
                    layer.streaming_policy().clone(),
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    })
}

fn require_cpu_expert_runtimes_mut(
    runtimes: &mut Option<Box<[DeepSeekV4LayerExpertRuntime]>>,
) -> Result<&mut [DeepSeekV4LayerExpertRuntime]> {
    runtimes.as_deref_mut().ok_or_else(|| {
        Error::Internal(
            "DeepSeek-V4 CPU execution requires per-layer expert runtimes, but none were allocated"
                .into(),
        )
    })
}

fn require_cpu_expert_runtime_mut(
    runtimes: &mut Option<Box<[DeepSeekV4LayerExpertRuntime]>>,
    layer: usize,
) -> Result<&mut DeepSeekV4LayerExpertRuntime> {
    require_cpu_expert_runtimes_mut(runtimes)?
        .get_mut(layer)
        .ok_or_else(|| {
            Error::Internal(format!("DeepSeek-V4 CPU expert runtime {layer} is missing"))
        })
}

/// Model-local lowering of model-neutral packed execution metadata.
///
/// Rows remain in packed query order. `sequences` supplies the mutable state and
/// query range for each sequence, while `row_to_sequence` makes row-owned CUDA
/// work independent of the aggregate forward mode.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PackedBatchMetadata {
    pub(super) mode: ForwardMode,
    pub(super) sequences: Vec<PackedSequenceMetadata>,
    pub(super) row_to_sequence: Vec<usize>,
    pub(super) sequence_major_rows: Vec<usize>,
    pub(super) max_query_tokens: usize,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PackedSequenceMetadata {
    pub(super) state_index: usize,
    pub(super) phase: ForwardPhase,
    pub(super) query: Range<usize>,
}

#[cfg(any(feature = "cuda", test))]
impl PackedBatchMetadata {
    pub(super) fn lower(batch: &ExecutionBatch, state_count: usize) -> Result<Self> {
        let mut sequences = Vec::with_capacity(batch.sequences().len());
        let mut row_to_sequence = vec![usize::MAX; batch.len()];
        let mut sequence_major_rows = Vec::with_capacity(batch.len());
        let mut expected_start = 0usize;
        let mut max_query_tokens = 0usize;
        let mut state_indices = BTreeSet::new();

        for (sequence_index, sequence) in batch.sequences().iter().enumerate() {
            let state_index = sequence
                .state_slot
                .try_as_usize()
                .map_err(|_| Error::Model("DeepSeek-V4 state slot exceeds usize".into()))?;
            if state_index >= state_count {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 state slot {state_index} is missing from {state_count} states"
                )));
            }
            if !state_indices.insert(state_index) {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 state slot {state_index} is referenced more than once"
                )));
            }
            let start = usize::try_from(sequence.query.start)
                .map_err(|_| Error::Model("DeepSeek-V4 query start exceeds usize".into()))?;
            let end = usize::try_from(sequence.query.end)
                .map_err(|_| Error::Model("DeepSeek-V4 query end exceeds usize".into()))?;
            if start != expected_start || start >= end || end > batch.len() {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 sequence {sequence_index} query range {start}..{end} does not densely cover packed rows from {expected_start}"
                )));
            }
            for row in start..end {
                row_to_sequence[row] = sequence_index;
                sequence_major_rows.push(row);
            }
            max_query_tokens = max_query_tokens.max(end - start);
            expected_start = end;
            sequences.push(PackedSequenceMetadata {
                state_index,
                phase: sequence.phase,
                query: start..end,
            });
        }
        if expected_start != batch.len() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 sequence queries cover {expected_start} of {} packed rows",
                batch.len()
            )));
        }

        Ok(Self {
            mode: batch.mode(),
            sequences,
            row_to_sequence,
            sequence_major_rows,
            max_query_tokens,
        })
    }

    /// Native CUDA keeps row-owned projection/HC/MoE work packed while mutable
    /// recurrent state is advanced once per sequence in query order.
    #[cfg(feature = "cuda")]
    pub(super) fn supports_native_cuda(&self) -> bool {
        self.row_to_sequence.len() >= 2
            && self.sequences.len() >= 2
            && self.sequence_major_rows.len() == self.row_to_sequence.len()
            && self
                .row_to_sequence
                .iter()
                .all(|sequence| *sequence < self.sequences.len())
    }
}

#[cfg(any(feature = "cuda", test))]
pub(super) fn begin_packed_sequence_steps(
    states: &[DeepSeekV4SequenceExecutionState],
    metadata: &PackedBatchMetadata,
) -> Result<Vec<crate::execution::SequenceStepBinding>> {
    metadata
        .sequences
        .iter()
        .map(|sequence| states[sequence.state_index].begin_step())
        .collect()
}

#[cfg(any(feature = "cuda", test))]
pub(super) fn poison_packed_sequence_steps(
    states: &mut [DeepSeekV4SequenceExecutionState],
    metadata: &PackedBatchMetadata,
    bindings: &[crate::execution::SequenceStepBinding],
) {
    for (sequence, binding) in metadata.sequences.iter().zip(bindings.iter().copied()) {
        states[sequence.state_index].poison_step(binding);
    }
}

#[cfg(any(feature = "cuda", test))]
pub(super) fn commit_packed_sequence_steps(
    states: &mut [DeepSeekV4SequenceExecutionState],
    metadata: &PackedBatchMetadata,
    bindings: Vec<crate::execution::SequenceStepBinding>,
) -> Result<()> {
    for (sequence, binding) in metadata.sequences.iter().zip(bindings) {
        states[sequence.state_index].commit_step(binding, sequence.query.len())?;
    }
    Ok(())
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
        let mut operators =
            DeepSeekV4OperatorContext::new(operator_backend, policy, options.expert_memory_policy)?;
        #[cfg(feature = "cuda")]
        if operator_backend == ModelExecutionBackend::Cuda {
            let layer_slot_capacities = plan
                .resources()
                .layer_experts()
                .iter()
                .map(DeepSeekV4PreparedLayerExperts::resident_capacity)
                .collect::<Vec<_>>();
            operators.configure_expert_frame_pool(
                model.config.num_routed_experts,
                &layer_slot_capacities,
                model.config.hidden_size,
                model.config.moe_intermediate_size,
            )?;
        }
        let mut layer_states = Vec::with_capacity(options.max_layers);
        for layer_idx in 0..options.max_layers {
            let state_start = operators.profile_start();
            layer_states.push(model.new_layer_sequence_state(layer_idx)?);
            if let Some(state_start) = state_start {
                operators.record_layer_state_init(layer_idx, duration_us(state_start.elapsed()));
            }
        }
        let cpu_expert_runtimes =
            build_cpu_expert_runtimes(operator_backend, plan.resources().layer_experts());

        let sequence =
            DeepSeekV4SequenceExecutionState::new(layer_states, model.config.num_routed_experts);
        let swiglu_limit = model.config.swiglu_limit;
        let expert_reader_max_tensor_bytes = options.expert_reader_max_tensor_bytes;
        #[cfg(feature = "cuda")]
        let expert_reader = if operator_backend == ModelExecutionBackend::Cuda {
            let allocator = operators
                .cuda
                .as_ref()
                .expect("CUDA backend initialized above")
                .pinned_host_allocator();
            ExpertStreamingReader::from_env_with_cuda_pinned(
                expert_reader_max_tensor_bytes,
                allocator,
            )?
        } else {
            ExpertStreamingReader::from_env(expert_reader_max_tensor_bytes)?
        };
        #[cfg(not(feature = "cuda"))]
        let expert_reader = ExpertStreamingReader::from_env(expert_reader_max_tensor_bytes)?;
        Ok(Self {
            plan,
            operators,
            cpu_expert_runtimes,
            #[cfg(feature = "cuda")]
            model_instance: NEXT_DSV4_MODEL_INSTANCE.fetch_add(1, Ordering::Relaxed),
            #[cfg(feature = "cuda")]
            expert_residency: None,
            #[cfg(feature = "cuda")]
            layer_arena_pool: PersistentArenaPool::new(),
            sequence,
            prefill_stats: DeepSeekV4PrefillRuntimeStats::default(),
            output_profile: DeepSeekV4OutputProfileStats::default(),
            expert_reader,
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

    pub fn kv_layout_schema(&self) -> &super::prepared::DeepSeekV4KvLayoutSchema {
        self.plan.resources().kv_layout()
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
        #[cfg(feature = "cuda")]
        if let Some(residency) = self.expert_residency.as_ref() {
            counters.expert_residency_stats = residency.stats();
        }
        let io = self.expert_reader.io_stats();
        counters.expert_io_submitted_extents = io.submitted_extents;
        counters.expert_io_completed_extents = io.completed_extents;
        counters.expert_io_failed_extents = io.failed_extents;
        counters.expert_io_requested_bytes = io.requested_bytes;
        counters.expert_io_aligned_bytes = io.aligned_bytes;
        counters.expert_io_coalesced_slices = io.coalesced_slices;
        counters.expert_io_fixed_file_registrations = io.fixed_file_registrations;
        counters.expert_io_fallback_count = io.fallback_count;
        counters.expert_io_slab_exhaustions = io.slab_exhaustions;
        counters.expert_io_peak_queue_depth = io.peak_queue_depth;
        counters.expert_io_read_us = io.read_us;
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

    fn record_output_profile_stage(
        &mut self,
        start: Option<Instant>,
        update: impl FnOnce(&mut DeepSeekV4OutputProfileStats, u64),
    ) -> Result<()> {
        if let Some(elapsed_us) = self.operators.finish_profile_stage(start)? {
            update(&mut self.output_profile, elapsed_us);
        }
        Ok(())
    }

    pub fn position(&self) -> usize {
        self.sequence.core.position()
    }

    /// Construct a fresh serving sequence without cloning default-session KV.
    pub fn create_sequence_state(&self) -> Result<DeepSeekV4SequenceExecutionState> {
        let resources = self.plan.resources();
        let mut layers = Vec::with_capacity(resources.prepare_options().max_layers);
        for layer in 0..resources.prepare_options().max_layers {
            layers.push(resources.model().new_layer_sequence_state(layer)?);
        }
        Ok(DeepSeekV4SequenceExecutionState::new(
            layers,
            resources.model().config.num_routed_experts,
        ))
    }

    /// Fork the default runner sequence as a runtime-shared paged prefix.
    pub fn fork_sequence_state(&mut self) -> Result<DeepSeekV4SequenceExecutionState> {
        let position = self.sequence.position();
        Self::fork_sequence_state_from_explicit(&self.sequence, position)
    }

    fn fork_sequence_state_from_explicit(
        source: &DeepSeekV4SequenceExecutionState,
        expected_position: usize,
    ) -> Result<DeepSeekV4SequenceExecutionState> {
        source.begin_step()?;
        if source.position() != expected_position {
            return Err(Error::Execution(format!(
                "DeepSeek-V4 exact fork expected committed position {expected_position}, source is at {}",
                source.position()
            )));
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(source.layers.len())
            .map_err(|error| {
                Error::Model(format!(
                    "DeepSeek-V4 paged fork layer metadata allocation failed: {error}"
                ))
            })?;
        layers.extend(
            source
                .layers
                .iter()
                .map(DeepSeekV4LayerState::fork_paged_prefix_metadata),
        );
        Ok(DeepSeekV4SequenceExecutionState {
            core: source.core.forked()?,
            layers,
            predictor: source.predictor.clone(),
            #[cfg(feature = "cuda")]
            paged_kv_binding: None,
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
        #[cfg(feature = "cuda")]
        if self.operators.backend() == ModelExecutionBackend::Cuda {
            if let Err(error) = self
                .operators
                .cuda_mut()?
                .activate_paged_binding(self.sequence.paged_kv_binding.as_ref())
            {
                std::mem::swap(&mut self.sequence, state);
                return Err(error);
            }
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(self)));
        #[cfg(feature = "cuda")]
        if self.operators.backend() == ModelExecutionBackend::Cuda {
            if let Ok(cuda) = self.operators.cuda_mut() {
                let _ = cuda.activate_paged_binding(None);
            }
        }
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

    /// Reset a sequence state for reuse with a new logical sequence.
    pub fn reset_sequence_state(
        &mut self,
        state: &mut DeepSeekV4SequenceExecutionState,
    ) -> Result<()> {
        state.reset_for_reuse();
        Ok(())
    }

    pub fn shutdown(&mut self) -> Result<()> {
        if self.shutdown {
            return Ok(());
        }
        self.sequence.release_capacity();
        self.cpu_expert_runtimes = None;
        #[cfg(feature = "cuda")]
        {
            self.layer_arena_pool.clear();
            self.operators
                .shutdown(self.expert_residency.as_deref_mut())?;
        }
        #[cfg(not(feature = "cuda"))]
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
        predict_experts_for_layer(
            self.plan.resources(),
            &mut self.operators,
            &mut self.sequence.predictor,
            self.cpu_expert_runtimes.as_deref(),
            layer,
            input,
        )
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
    fn decode_lookahead_prefetch_enabled(&self) -> bool {
        self.plan.resources().policy().decode_lookahead_prefetch()
    }

    fn prepare_predicted_experts_for_layer(
        &mut self,
        layer: usize,
        predicted_experts: &[usize],
    ) -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        let _ = layer;
        if predicted_experts.is_empty()
            || self.plan.resources().prepare_options().moe_prefetch_experts == 0
            || !self.lookahead_prefetch_enabled()
        {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        {
            if self.operators.backend() != ModelExecutionBackend::Cuda {
                return Ok(());
            }
            let prepared = self
                .plan
                .resources()
                .layer_experts()
                .get(layer)
                .ok_or_else(|| {
                    Error::Internal(format!("prepared expert layer {layer} is missing"))
                })?;
            let source_catalog = std::sync::Arc::clone(prepared.source_catalog());
            let prefetch_capacity = prepared.prefetch_capacity();
            let residency = self.expert_residency.as_deref_mut().ok_or_else(|| {
                Error::Execution("runtime expert residency controller is not installed".into())
            })?;
            self.operators.prefetch_predicted_experts(
                layer,
                predicted_experts,
                residency,
                source_catalog.as_ref(),
                prefetch_capacity,
                &self.expert_reader,
            )?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_decode_prefetch_enabled(&self) -> bool {
        self.operators.backend() == ModelExecutionBackend::Cuda
            && self.plan.resources().prepare_options().moe_prefetch_experts > 0
            && self.lookahead_prefetch_enabled()
            && self.decode_lookahead_prefetch_enabled()
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
        let residency = self.expert_residency.as_deref_mut().ok_or_else(|| {
            Error::Execution("runtime expert residency controller is not installed".into())
        })?;
        for (layer_idx, prepared) in self.plan.resources().layer_experts().iter().enumerate() {
            let predicted = prepared
                .source_catalog()
                .iter()
                .map(|(expert, _)| expert.expert)
                .take(count)
                .collect::<Vec<_>>();
            if predicted.is_empty() {
                continue;
            }
            warmed = warmed.saturating_add(self.operators.prewarm_experts(
                layer_idx,
                &predicted,
                residency,
                prepared.source_catalog().as_ref(),
                prepared.prefetch_capacity(),
                &self.expert_reader,
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
            let cpu_expert_runtime = self
                .cpu_expert_runtimes
                .as_deref()
                .and_then(|runtimes| runtimes.get(layer_idx));
            let index_head_dim = layer.attention.config.index_head_dim;
            let (resident_experts, resident_expert_bytes) = {
                #[cfg(feature = "cuda")]
                {
                    if let Some(cache) = self.operators.cuda.as_ref() {
                        cache.resident_expert_stats_for_layer(layer_idx)
                    } else {
                        cpu_expert_runtime
                            .map(|runtime| {
                                (
                                    runtime.expert_handles.len(),
                                    runtime.expert_handles.total_bytes(),
                                )
                            })
                            .unwrap_or((0, 0))
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    cpu_expert_runtime
                        .map(|runtime| {
                            (
                                runtime.expert_handles.len(),
                                runtime.expert_handles.total_bytes(),
                            )
                        })
                        .unwrap_or((0, 0))
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
            let expert_runtime =
                require_cpu_expert_runtime_mut(&mut self.cpu_expert_runtimes, layer_idx)?;
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

        let hc_head_start = self.operators.profile_start();
        let hidden = {
            #[cfg(feature = "cuda")]
            if self.operators.backend() == ModelExecutionBackend::Cuda {
                let hidden_len = tokens * self.plan.resources().model().config.hidden_size;
                let mut buffers = self.operators.cuda_mut()?.take_decode_buffers(
                    hc_state.len(),
                    hidden_len,
                    self.plan.resources().model().config.hidden_size,
                )?;
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
                    self.operators.capture_parity_checkpoint_last_row(
                        0,
                        "final_hidden",
                        &buffers.final_hidden,
                        self.plan.resources().model().config.hidden_size,
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
        self.record_output_profile_stage(hc_head_start, |profile, elapsed_us| {
            profile.final_hc_head_calls = profile.final_hc_head_calls.saturating_add(1);
            profile.final_hc_head_us = profile.final_hc_head_us.saturating_add(elapsed_us);
        })?;

        let start = (tokens - 1) * self.plan.resources().model().config.hidden_size;
        let norm_start = self.operators.profile_start();
        let normed = self.operators.rms_norm(
            &hidden[start..start + self.plan.resources().model().config.hidden_size],
            &self.plan.resources().model().output_norm,
            self.plan.resources().model().config.norm_eps,
            "output_norm",
        )?;
        #[cfg(feature = "cuda")]
        self.operators
            .capture_parity_checkpoint_host(0, "final_norm", &normed);
        self.record_output_profile_stage(norm_start, |profile, elapsed_us| {
            profile.final_norm_calls = profile.final_norm_calls.saturating_add(1);
            profile.final_norm_us = profile.final_norm_us.saturating_add(elapsed_us);
        })?;
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
    fn run_cuda_packed_batch_uncommitted(
        &mut self,
        states: &mut [DeepSeekV4SequenceExecutionState],
        batch: &ExecutionBatch,
        metadata: &PackedBatchMetadata,
    ) -> Result<ExecutionOutput> {
        let rows = batch.len();
        if !metadata.supports_native_cuda() {
            return Err(Error::Model(
                "DeepSeek-V4 CUDA packed shell does not support this batch shape yet".into(),
            ));
        }
        for (sequence_index, sequence) in metadata.sequences.iter().enumerate() {
            let state = &states[sequence.state_index];
            for (offset, row) in sequence.query.clone().enumerate() {
                let expected = state.position().checked_add(offset).ok_or_else(|| {
                    Error::Model("DeepSeek-V4 packed sequence position overflow".into())
                })?;
                if expected != batch.positions()[row] as usize {
                    return Err(Error::Model(format!(
                        "DeepSeek-V4 packed sequence {sequence_index} row {row} position mismatch: expected={expected} batch={}",
                        batch.positions()[row]
                    )));
                }
            }
            if state.paged_kv_binding.is_none() {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 packed sequence {sequence_index} has no prepared paged binding"
                )));
            }
        }

        let requested_logit_rows = batch
            .logits()
            .iter()
            .enumerate()
            .filter_map(|(row, request)| matches!(request, LogitsRequest::TopK(_)).then_some(row))
            .collect::<Vec<_>>();
        let max_top_k = batch
            .logits()
            .iter()
            .filter_map(|request| match request {
                LogitsRequest::TopK(k) => Some(k.get() as usize),
                LogitsRequest::None => None,
                LogitsRequest::Full => None,
            })
            .max()
            .unwrap_or(0);
        let bindings = begin_packed_sequence_steps(states, metadata)?;
        let paged_bindings = metadata
            .sequences
            .iter()
            .map(|sequence| {
                states[sequence.state_index]
                    .paged_kv_binding
                    .clone()
                    .expect("validated above")
            })
            .collect::<Vec<_>>();
        let sequence_prefill = metadata
            .sequences
            .iter()
            .map(|sequence| sequence.phase == ForwardPhase::Prefill)
            .collect::<Vec<_>>();
        let positions = batch
            .positions()
            .iter()
            .map(|position| *position as usize)
            .collect::<Vec<_>>();
        let mut hc_state = Vec::new();
        for token_id in batch.token_ids() {
            hc_state.extend(
                self.plan
                    .resources()
                    .model()
                    .initial_hc_state_for_token(*token_id)?,
            );
        }
        let max_layers = self.plan.resources().prepare_options().max_layers;
        let hidden_size = self.plan.resources().model().config.hidden_size;
        let hc_row_size = hc_state.len() / rows;

        self.operators.check_cuda_arena_acquire()?;

        let arena_key = ExecutionShapeKey::new(
            metadata.mode,
            rows,
            metadata.sequences.len(),
            metadata.max_query_tokens,
        );
        let mut arena_lease = {
            let layers = self.plan.resources().layers();
            let operators = &mut self.operators;
            self.layer_arena_pool.acquire(arena_key, || {
                DeepSeekV4LayerArenaVariants::try_build_for_packed_mode(layers, rows, operators)
            })?
        };

        let mut decode_buffers = self.operators.cuda_mut()?.take_decode_buffers(
            rows * hc_row_size,
            hidden_size,
            hidden_size,
        )?;

        let residency = self.expert_residency.as_deref_mut().ok_or_else(|| {
            Error::Execution("runtime expert residency controller is not installed".into())
        })?;
        let layer_experts = self.plan.resources().layer_experts();
        let execution = (|| -> Result<Vec<Vec<TokenLogit>>> {
            self.operators
                .cuda_mut()?
                .ops
                .overwrite_f32_buffer(&hc_state, &mut decode_buffers.hc_input)?;
            let binding_refs = paged_bindings.iter().collect::<Vec<_>>();
            self.operators
                .cuda_mut()?
                .activate_paged_bindings_for_rows(&binding_refs, &metadata.row_to_sequence)?;

            for layer_idx in 0..max_layers {
                let arena = arena_lease
                    .get_mut()
                    .get_for_layer_mut(layer_idx)
                    .expect("pooled layer arena variants match prepared layers");
                let requested_states = metadata
                    .sequences
                    .iter()
                    .map(|sequence| sequence.state_index)
                    .collect::<BTreeSet<_>>();
                let mut available_states = states
                    .iter_mut()
                    .enumerate()
                    .filter(|(state_index, _)| requested_states.contains(state_index))
                    .collect::<BTreeMap<_, _>>();
                let mut layer_states = metadata
                    .sequences
                    .iter()
                    .map(|sequence| {
                        available_states
                            .remove(&sequence.state_index)
                            .map(|state| &mut state.layers[layer_idx])
                            .ok_or_else(|| {
                                Error::Model(format!(
                                    "DeepSeek-V4 state slot {} is referenced more than once",
                                    sequence.state_index
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.plan.resources().layers()[layer_idx].packed_rows_device_hc_device(
                    &mut layer_states,
                    &metadata.row_to_sequence,
                    &metadata.sequence_major_rows,
                    &sequence_prefill,
                    &paged_bindings,
                    residency,
                    layer_experts[layer_idx].source_catalog().as_ref(),
                    layer_experts[layer_idx].prefetch_capacity(),
                    arena,
                    &mut decode_buffers.hc_input,
                    batch.token_ids(),
                    &positions,
                    &[],
                    &self.expert_reader,
                    &mut self.operators,
                )?;
            }
            if max_top_k == 0 {
                return Ok(vec![Vec::new(); rows]);
            }
            let mut logits = vec![Vec::new(); rows];
            for &row in &requested_logit_rows {
                if row != 0 {
                    self.operators.cuda_mut()?.ops.copy_f32_within(
                        &mut decode_buffers.hc_input,
                        row * hc_row_size,
                        0,
                        hc_row_size,
                    )?;
                }
                self.operators.cuda_mut()?.hc_head_from_device_into(
                    &decode_buffers.hc_input,
                    1,
                    self.plan.resources().model().config.hc_config(),
                    &self.plan.resources().model().hc_head,
                    &mut decode_buffers.final_hidden,
                )?;
                self.operators.capture_parity_checkpoint_last_row(
                    0,
                    "final_hidden",
                    &decode_buffers.final_hidden,
                    hidden_size,
                )?;
                let sequence = metadata.row_to_sequence[row];
                if sequence_prefill[sequence] {
                    let final_hidden = self
                        .operators
                        .cuda_mut()?
                        .ops
                        .download_f32_buffer(&decode_buffers.final_hidden)?;
                    let final_norm = self.operators.rms_norm(
                        &final_hidden,
                        &self.plan.resources().model().output_norm,
                        self.plan.resources().model().config.norm_eps,
                        "output_norm",
                    )?;
                    self.operators
                        .capture_parity_checkpoint_host(0, "final_norm", &final_norm);
                    self.operators
                        .cuda_mut()?
                        .ops
                        .overwrite_f32_buffer(&final_norm, &mut decode_buffers.topk_row)?;
                } else {
                    self.operators.cuda_mut()?.rms_norm_device_cached_into(
                        "output_norm",
                        &decode_buffers.final_hidden,
                        &self.plan.resources().model().output_norm,
                        self.plan.resources().model().config.norm_eps,
                        &mut decode_buffers.topk_row,
                    )?;
                }
                logits[row] = self
                    .plan
                    .resources()
                    .model()
                    .topk_logits_for_hidden_device_with_operators(
                        &decode_buffers.topk_row,
                        max_top_k,
                        self.plan
                            .resources()
                            .prepare_options()
                            .output_head_chunk_rows,
                        &mut self.operators,
                    )?;
            }
            Ok(logits)
        })();

        drop(arena_lease);
        let _ = self.operators.cuda_mut()?.activate_paged_binding(None);
        self.restore_cuda_decode_buffers(decode_buffers);
        let logits = match execution {
            Ok(logits) => logits,
            Err(error) => {
                poison_packed_sequence_steps(states, metadata, &bindings);
                return Err(error);
            }
        };

        let mut output_rows = Vec::new();
        for (row, request) in batch.logits().iter().copied().enumerate() {
            if let LogitsRequest::TopK(k) = request {
                output_rows.push(LogitsRow::new(
                    row as u32,
                    LogitsOutput::TopK(
                        logits[row]
                            .iter()
                            .take(k.get() as usize)
                            .map(|item| ExecutionTokenLogit {
                                token_id: item.token_id,
                                logit: item.logit,
                            })
                            .collect(),
                    ),
                ));
            }
        }
        let output = ExecutionOutput::new(output_rows);
        if let Err(error) =
            output.validate_with_capabilities(batch, &self.multi_session_capabilities())
        {
            poison_packed_sequence_steps(states, metadata, &bindings);
            return Err(error);
        }
        commit_packed_sequence_steps(states, metadata, bindings)?;
        match metadata.mode {
            ForwardMode::Prefill => {
                self.output_profile.packed_prefill_batches =
                    self.output_profile.packed_prefill_batches.saturating_add(1);
                self.output_profile.packed_prefill_rows = self
                    .output_profile
                    .packed_prefill_rows
                    .saturating_add(rows as u64);
            }
            ForwardMode::Decode => {
                self.output_profile.packed_decode_batches =
                    self.output_profile.packed_decode_batches.saturating_add(1);
                self.output_profile.packed_decode_rows = self
                    .output_profile
                    .packed_decode_rows
                    .saturating_add(rows as u64);
            }
            ForwardMode::Mixed => {
                self.output_profile.packed_mixed_batches =
                    self.output_profile.packed_mixed_batches.saturating_add(1);
                self.output_profile.packed_mixed_rows = self
                    .output_profile
                    .packed_mixed_rows
                    .saturating_add(rows as u64);
            }
        }
        Ok(output)
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
        let prefetch_enabled = self.cuda_decode_prefetch_enabled();
        let mut predicted_experts = if max_layers == 0 {
            Vec::new()
        } else {
            self.predicted_experts_for_layer(0, ExpertPredictionInput::Decode { token_id })?
        };
        if prefetch_enabled {
            self.prepare_predicted_experts_for_layer(0, &predicted_experts)?;
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
        let mut decode_buffers = self.operators.cuda_mut()?.take_decode_buffers(
            hc_state.len(),
            hidden_size,
            hidden_size,
        )?;
        let position = self.sequence.core.position();
        let layer_result: Result<()> = (|| {
            self.operators
                .cuda_mut()?
                .ops
                .overwrite_f32_buffer(&hc_state, &mut decode_buffers.hc_input)?;
            for layer_idx in 0..max_layers {
                let step = {
                    let layer = &self.plan.resources().layers()[layer_idx];
                    let prepared = &self.plan.resources().layer_experts()[layer_idx];
                    let state = &mut self.sequence.layers[layer_idx];
                    let residency = self.expert_residency.as_deref_mut().ok_or_else(|| {
                        Error::Execution(
                            "runtime expert residency controller is not installed".into(),
                        )
                    })?;
                    let arena = arena_lease
                        .get_mut()
                        .get_for_layer_mut(layer_idx)
                        .expect("pooled layer arena variants match prepared layers");
                    layer.decode_step_device_hc_device(
                        state,
                        residency,
                        prepared.source_catalog().as_ref(),
                        prepared.prefetch_capacity(),
                        arena,
                        &mut decode_buffers.hc_input,
                        token_id,
                        position,
                        &predicted_experts,
                        &self.expert_reader,
                        &mut self.operators,
                    )?
                };
                self.sequence
                    .predictor
                    .observe_batch(ExpertBatchAccessEvent::from_routes(
                        layer_idx,
                        ExpertAccessPhase::Decode,
                        1,
                        &step.moe.routes,
                        &step.moe.streaming,
                    ));

                let next_layer = layer_idx + 1;
                if next_layer < max_layers {
                    predicted_experts = predict_experts_for_layer(
                        self.plan.resources(),
                        &mut self.operators,
                        &mut self.sequence.predictor,
                        self.cpu_expert_runtimes.as_deref(),
                        next_layer,
                        ExpertPredictionInput::Decode { token_id },
                    )?;
                    if prefetch_enabled && !predicted_experts.is_empty() {
                        let prepared = &self.plan.resources().layer_experts()[next_layer];
                        let source_catalog = std::sync::Arc::clone(prepared.source_catalog());
                        let prefetch_capacity = prepared.prefetch_capacity();
                        let residency = self.expert_residency.as_deref_mut().ok_or_else(|| {
                            Error::Execution(
                                "runtime expert residency controller is not installed".into(),
                            )
                        })?;
                        self.operators.prefetch_predicted_experts(
                            next_layer,
                            &predicted_experts,
                            residency,
                            source_catalog.as_ref(),
                            prefetch_capacity,
                            &self.expert_reader,
                        )?;
                    }
                }
            }
            Ok(())
        })();
        drop(arena_lease);
        if let Err(error) = layer_result {
            self.restore_cuda_decode_buffers(decode_buffers);
            return Err(error);
        }

        let result = (|| {
            if matches!(completion, CudaDecodeCompletion::Feed) {
                return Ok(CudaDecodeOutput::Feed);
            }

            let final_hc_start = self.operators.profile_start();
            self.operators.cuda_mut()?.hc_head_from_device_into(
                &decode_buffers.hc_input,
                1,
                self.plan.resources().model().config.hc_config(),
                &self.plan.resources().model().hc_head,
                &mut decode_buffers.final_hidden,
            )?;
            self.record_output_profile_stage(final_hc_start, |profile, elapsed_us| {
                profile.final_hc_head_calls = profile.final_hc_head_calls.saturating_add(1);
                profile.final_hc_head_us = profile.final_hc_head_us.saturating_add(elapsed_us);
            })?;

            let final_norm_start = self.operators.profile_start();
            self.operators.cuda_mut()?.rms_norm_device_cached_into(
                "output_norm",
                &decode_buffers.final_hidden,
                &self.plan.resources().model().output_norm,
                self.plan.resources().model().config.norm_eps,
                &mut decode_buffers.final_norm,
            )?;
            self.record_output_profile_stage(final_norm_start, |profile, elapsed_us| {
                profile.final_norm_calls = profile.final_norm_calls.saturating_add(1);
                profile.final_norm_us = profile.final_norm_us.saturating_add(elapsed_us);
            })?;

            match completion {
                CudaDecodeCompletion::Feed => unreachable!("handled before finalization"),
                CudaDecodeCompletion::DownloadHidden => self
                    .operators
                    .cuda_mut()?
                    .ops
                    .download_f32_buffer(&decode_buffers.final_norm)
                    .map(CudaDecodeOutput::Hidden),
                CudaDecodeCompletion::TopK(top_k) => {
                    let topk_start = self.operators.profile_start();
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
                    self.record_output_profile_stage(topk_start, |profile, elapsed_us| {
                        profile.lm_head_topk_calls = profile.lm_head_topk_calls.saturating_add(1);
                        profile.lm_head_topk_us =
                            profile.lm_head_topk_us.saturating_add(elapsed_us);
                    })?;
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
        let topk_start = self.operators.profile_start();
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
        self.record_output_profile_stage(topk_start, |profile, elapsed_us| {
            profile.lm_head_topk_calls = profile.lm_head_topk_calls.saturating_add(1);
            profile.lm_head_topk_us = profile.lm_head_topk_us.saturating_add(elapsed_us);
        })?;
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

    pub fn prefill_tokens_logits_row_range(
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
            let layer_start = report_progress
                .then(|| self.operators.profile_start())
                .flatten();
            let predicted_experts = self.predicted_experts_for_layer(
                layer_idx,
                ExpertPredictionInput::Prefill { token_ids },
            )?;
            self.prepare_predicted_experts_for_layer(layer_idx, &predicted_experts)?;
            let layer = &self.plan.resources().layers()[layer_idx];
            let state = &mut self.sequence.layers[layer_idx];
            let expert_runtime =
                require_cpu_expert_runtime_mut(&mut self.cpu_expert_runtimes, layer_idx)?;
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
                DeepSeekV4LayerArenaVariants::try_build_for_mode(
                    layers,
                    ForwardMode::Prefill,
                    tokens,
                    operators,
                )
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
        let mut buffers = self.operators.cuda_mut()?.take_decode_buffers(
            hc_state.len(),
            tokens * hidden_size,
            hidden_size,
        )?;
        let residency = self.expert_residency.as_deref_mut().ok_or_else(|| {
            Error::Execution("runtime expert residency controller is not installed".into())
        })?;
        let layer_experts = self.plan.resources().layer_experts();
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
                    residency,
                    layer_experts[layer_idx].source_catalog().as_ref(),
                    layer_experts[layer_idx].prefetch_capacity(),
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
        let topk_start = self.operators.profile_start();
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
        self.record_output_profile_stage(topk_start, |profile, elapsed_us| {
            profile.lm_head_topk_calls = profile.lm_head_topk_calls.saturating_add(1);
            profile.lm_head_topk_us = profile.lm_head_topk_us.saturating_add(elapsed_us);
        })?;
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
            let expert_runtime =
                require_cpu_expert_runtime_mut(&mut self.cpu_expert_runtimes, layer_idx)?;
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
            let expert_runtime =
                require_cpu_expert_runtime_mut(&mut self.cpu_expert_runtimes, layer_idx)?;
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
        let residency = self.expert_residency.as_deref_mut().ok_or_else(|| {
            Error::Execution("runtime expert residency controller is not installed".into())
        })?;
        let layer_experts = self.plan.resources().layer_experts();
        for layer_idx in 0..max_layers {
            let layer = &self.plan.resources().layers()[layer_idx];
            let state = &mut self.sequence.layers[layer_idx];
            let arena = arena_lease
                .get_mut()
                .get_for_layer_mut(layer_idx)
                .expect("pooled layer arena variants match prepared layers");
            let step = layer.decode_step_device_hc_device(
                state,
                residency,
                layer_experts[layer_idx].source_catalog().as_ref(),
                layer_experts[layer_idx].prefetch_capacity(),
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

fn predict_experts_for_layer(
    resources: &DeepSeekV4PreparedResources,
    operators: &mut DeepSeekV4OperatorContext,
    predictor: &mut ScoreBasedExpertPredictor,
    cpu_expert_runtimes: Option<&[DeepSeekV4LayerExpertRuntime]>,
    layer: usize,
    input: ExpertPredictionInput<'_>,
) -> Result<Vec<usize>> {
    let count = resources
        .prepare_options()
        .moe_prefetch_experts
        .min(resources.model().config.num_routed_experts);
    if count == 0 {
        return Ok(Vec::new());
    }
    let phase = input.phase();

    #[cfg(feature = "cuda")]
    let (resident, materializing, host_staged) = operators
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

    if operators.backend() == ModelExecutionBackend::Cuda
        && layer < resources.model().config.num_hash_layers
    {
        let layer_ref = resources.layers().get(layer).ok_or_else(|| {
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
                            .take(resources.model().config.num_experts_per_tok),
                        count,
                        resources.model().config.num_routed_experts,
                    );
                }
            }
            ExpertPredictionInput::Prefill { token_ids }
                if resources.policy().prefill_hash_lookahead() =>
            {
                if let Some(hash_experts) = layer_ref.router.hash_expert_union_for_tokens(
                    token_ids,
                    resources.model().config.num_experts_per_tok,
                    count,
                )? {
                    push_candidate_experts(
                        &mut predicted,
                        &mut excluded,
                        hash_experts,
                        count,
                        resources.model().config.num_routed_experts,
                    );
                }
            }
            ExpertPredictionInput::Prefill { .. } => {}
        }
    }

    if predicted.len() < count {
        for prediction in predictor.predict(ExpertPredictContext {
            layer,
            phase,
            budget: count,
            num_experts: resources.model().config.num_routed_experts,
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
        let candidates = if let Some(expert_runtime) =
            cpu_expert_runtimes.and_then(|runtimes| runtimes.get(layer))
        {
            Some(
                expert_runtime
                    .expert_planner
                    .hot_experts(layer, resources.model().config.num_routed_experts),
            )
        } else if source_catalog_fallback_enabled(operators.backend(), phase) {
            Some(
                resources
                    .layer_expert_source_catalog(layer)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "DeepSeek-V4 prepared expert catalog {layer} is missing"
                        ))
                    })?
                    .iter()
                    .map(|(expert, _)| expert.expert)
                    .collect(),
            )
        } else {
            None
        };
        if let Some(candidates) = candidates {
            push_candidate_experts(
                &mut predicted,
                &mut excluded,
                candidates,
                count,
                resources.model().config.num_routed_experts,
            );
        }
    }
    Ok(predicted)
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

fn source_catalog_fallback_enabled(
    backend: ModelExecutionBackend,
    phase: ExpertAccessPhase,
) -> bool {
    backend != ModelExecutionBackend::Cuda || phase != ExpertAccessPhase::Decode
}

fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod expert_runtime_tests {
    use std::sync::Arc;

    use crate::moe::streaming::{
        ExpertId, ExpertLoadSource, ExpertSourceCatalog, ExpertStreamingPolicy,
    };

    use super::*;

    fn prepared_layer() -> DeepSeekV4PreparedLayerExperts {
        DeepSeekV4PreparedLayerExperts::new(
            Arc::new(ExpertSourceCatalog::from_sources([(
                ExpertId::new(0, 0),
                ExpertLoadSource::LocalShard {
                    path: "expert.safetensors".into(),
                    offset: 0,
                    bytes: 16,
                },
            )])),
            ExpertStreamingPolicy::quality_first(1),
        )
    }

    #[test]
    fn runner_allocates_layer_expert_runtime_only_for_cpu() {
        let layers = [prepared_layer()];
        let cpu = build_cpu_expert_runtimes(ModelExecutionBackend::Cpu, &layers);
        let cuda = build_cpu_expert_runtimes(ModelExecutionBackend::Cuda, &layers);

        assert_eq!(cpu.as_deref().map(<[_]>::len), Some(1));
        assert!(cuda.is_none());
        assert!(Arc::ptr_eq(
            cpu.as_ref().unwrap()[0].expert_planner.source_catalog(),
            layers[0].source_catalog(),
        ));
    }

    #[test]
    fn source_catalog_fallback_is_disabled_only_for_cuda_decode() {
        assert!(source_catalog_fallback_enabled(
            ModelExecutionBackend::Cpu,
            ExpertAccessPhase::Prefill,
        ));
        assert!(source_catalog_fallback_enabled(
            ModelExecutionBackend::Cpu,
            ExpertAccessPhase::Decode,
        ));
        assert!(source_catalog_fallback_enabled(
            ModelExecutionBackend::Cuda,
            ExpertAccessPhase::Prefill,
        ));
        assert!(!source_catalog_fallback_enabled(
            ModelExecutionBackend::Cuda,
            ExpertAccessPhase::Decode,
        ));
    }
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

impl MultiSessionRunner for DeepSeekV4Runner {
    type SequenceState = DeepSeekV4SequenceExecutionState;

    fn expert_residency_requirements(&self) -> Option<ExpertResidencyRequirements> {
        #[cfg(feature = "cuda")]
        if self.operators.backend() == ModelExecutionBackend::Cuda {
            return Some(ExpertResidencyRequirements::new(
                self.model_instance,
                self.plan
                    .resources()
                    .layer_experts()
                    .iter()
                    .map(DeepSeekV4PreparedLayerExperts::resident_capacity)
                    .collect(),
            ));
        }
        None
    }

    fn expert_residency_control_installed(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            return self.expert_residency.is_some();
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    fn install_expert_residency_control(
        &mut self,
        control: Box<dyn ExpertResidencyControl>,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let expected = self.expert_residency_requirements().ok_or_else(|| {
                Error::Execution(
                    "DeepSeek-V4 CPU runner does not accept an expert residency controller".into(),
                )
            })?;
            if control.requirements() != expected {
                return Err(Error::Execution(format!(
                    "DeepSeek-V4 expert residency requirements mismatch: expected {:?}, got {:?}",
                    expected,
                    control.requirements()
                )));
            }
            if self.expert_residency.is_some() {
                return Err(Error::Execution(
                    "DeepSeek-V4 expert residency controller is already installed".into(),
                ));
            }
            self.expert_residency = Some(control);
            return Ok(());
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = control;
            Err(Error::Execution(
                "DeepSeek-V4 was built without CUDA expert residency support".into(),
            ))
        }
    }

    fn configure_kv_page_capacity(&mut self, max_pages: usize) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.operators.backend() == ModelExecutionBackend::Cuda {
            let schema = self.plan.resources().kv_layout().clone();
            self.operators
                .cuda_mut()?
                .configure_kv_page_pool(&schema, max_pages)?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = max_pages;
        Ok(())
    }

    fn release_kv_pages(&mut self, pages: &[ferrule_common::execution::KvPageId]) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.operators.cuda.is_some() {
            self.operators.cuda_mut()?.release_kv_pages(pages)?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = pages;
        Ok(())
    }

    fn preempt_kv_pages(&mut self, pages: &[ferrule_common::execution::KvPageId]) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.operators.cuda.is_some() {
            self.operators.cuda_mut()?.preempt_kv_pages(pages)?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = pages;
        Ok(())
    }

    fn restore_kv_pages(&mut self, pages: &[ferrule_common::execution::KvPageId]) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.operators.cuda.is_some() {
            self.operators.cuda_mut()?.restore_kv_pages(pages)?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = pages;
        Ok(())
    }

    fn prepare_multi_session_batch(
        &mut self,
        states: &mut [Self::SequenceState],
        batch: &ferrule_common::execution::ExecutionBatch,
        kv_reservations: &[ferrule_common::execution::KvReservation],
    ) -> Result<bool> {
        #[cfg(feature = "cuda")]
        if self
            .operators
            .cuda
            .as_ref()
            .is_some_and(|cuda| cuda.has_kv_page_pool())
            && !batch.kv_block_ids().is_empty()
        {
            if kv_reservations.len() != batch.sequences().len() {
                return Err(Error::Model(format!(
                    "DeepSeek-V4 paged batch has {} sequences but {} KV reservations",
                    batch.sequences().len(),
                    kv_reservations.len()
                )));
            }
            let physical = kv_reservations
                .iter()
                .map(|reservation| {
                    (
                        reservation.newly_allocated.clone(),
                        reservation.cow_replacement,
                    )
                })
                .collect::<Vec<_>>();
            self.operators.cuda_mut()?.prepare_kv_pages(&physical)?;
            let lowered = (|| -> Result<Vec<_>> {
                batch
                    .sequences()
                    .iter()
                    .map(|sequence| {
                        let state_index = sequence.state_slot.try_as_usize().map_err(|_| {
                            Error::Model("DeepSeek-V4 state slot exceeds usize".into())
                        })?;
                        if state_index >= states.len() {
                            return Err(Error::Model(format!(
                                "DeepSeek-V4 state slot {state_index} is missing during paged prepare"
                            )));
                        }
                        let block_start = usize::try_from(sequence.block_table.start)
                            .map_err(|_| Error::Model("KV block range exceeds usize".into()))?;
                        let block_end = usize::try_from(sequence.block_table.end)
                            .map_err(|_| Error::Model("KV block range exceeds usize".into()))?;
                        let binding = self.operators.cuda_mut()?.lower_paged_binding(
                            &batch.kv_block_ids()[block_start..block_end],
                            sequence.sequence_len as usize,
                        )?;
                        Ok((state_index, binding))
                    })
                    .collect()
            })();
            let lowered = match lowered {
                Ok(lowered) => lowered,
                Err(error) => {
                    let _ = self.operators.cuda_mut()?.rollback_kv_pages();
                    return Err(error);
                }
            };
            for (state_index, binding) in lowered {
                states[state_index].paged_kv_binding = Some(binding);
            }
            return Ok(true);
        }
        let _ = (states, batch, kv_reservations);
        Ok(false)
    }

    fn commit_multi_session_batch(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self
            .operators
            .cuda
            .as_ref()
            .is_some_and(|cuda| cuda.has_kv_page_pool())
        {
            self.operators.cuda_mut()?.commit_kv_pages()?;
        }
        Ok(())
    }

    fn rollback_multi_session_batch(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.operators.cuda.is_some() {
            self.operators.cuda_mut()?.rollback_kv_pages()?;
        }
        Ok(())
    }

    fn execute_multi_session_batch(
        &mut self,
        states: &mut [Self::SequenceState],
        batch: &ferrule_common::execution::ExecutionBatch,
    ) -> Result<Option<ferrule_common::execution::ExecutionOutput>> {
        #[cfg(not(feature = "cuda"))]
        let _ = (states, batch);
        #[cfg(feature = "cuda")]
        if self.operators.backend() == ModelExecutionBackend::Cuda
            && self
                .operators
                .cuda
                .as_ref()
                .is_some_and(|cuda| cuda.has_kv_page_pool())
            && !batch
                .logits()
                .iter()
                .any(|request| matches!(request, LogitsRequest::Full))
        {
            let metadata = PackedBatchMetadata::lower(batch, states.len())?;
            if metadata.supports_native_cuda() {
                return self
                    .run_cuda_packed_batch_uncommitted(states, batch, &metadata)
                    .map(Some);
            }
        }
        Ok(None)
    }

    fn create_sequence_state(&mut self) -> Result<Self::SequenceState> {
        DeepSeekV4Runner::create_sequence_state(self)
    }

    fn with_sequence_state<T>(
        &mut self,
        state: &mut Self::SequenceState,
        execute: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        DeepSeekV4Runner::with_sequence_state(self, state, execute)
    }

    fn fork_sequence_state(&mut self) -> Result<Self::SequenceState> {
        DeepSeekV4Runner::fork_sequence_state(self)
    }

    fn fork_sequence_state_from(
        &mut self,
        source: &Self::SequenceState,
        expected_position: usize,
    ) -> Result<Self::SequenceState> {
        Self::fork_sequence_state_from_explicit(source, expected_position)
    }

    fn reset_sequence_state(&mut self, state: &mut Self::SequenceState) -> Result<()> {
        DeepSeekV4Runner::reset_sequence_state(self, state)
    }

    fn release_sequence_state(&mut self, state: Self::SequenceState) -> Result<()> {
        DeepSeekV4Runner::release_sequence_state(self, state)
    }

    fn multi_session_capabilities(&self) -> ferrule_common::execution::ExecutionCapabilities {
        let max_top_k = u32::try_from(self.max_top_k()).unwrap_or(u32::MAX);
        let vocab_size = self.model_info().vocab_size;
        let full_logits_width = u32::try_from(vocab_size)
            .ok()
            .and_then(std::num::NonZeroU32::new);
        let max_packed_rows = usize::try_from(u32::MAX).unwrap_or(usize::MAX);

        ferrule_common::execution::ExecutionCapabilities {
            max_batch_tokens: max_packed_rows,
            max_sequences: usize::MAX,
            max_prefill_query_tokens_per_sequence: max_packed_rows,
            max_decode_query_tokens_per_sequence: 1,
            max_top_k: std::num::NonZeroU32::new(max_top_k),
            supports_prefill: true,
            supports_decode: true,
            supports_mixed: true,
            full_logits_width,
            kv_binding_mode: {
                #[cfg(feature = "cuda")]
                {
                    if self
                        .operators
                        .cuda
                        .as_ref()
                        .is_some_and(|cuda| cuda.has_kv_page_pool())
                    {
                        ferrule_common::execution::KvBindingMode::Paged
                    } else {
                        ferrule_common::execution::KvBindingMode::None
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    ferrule_common::execution::KvBindingMode::None
                }
            },
            logits_row_policy: ferrule_common::execution::LogitsRowPolicy::LastPerSequence,
        }
    }
}
