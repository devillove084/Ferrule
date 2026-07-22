//! Per-sequence execution state for DeepSeek-V4.
//!
//! Model-neutral cursor, generation, and poison semantics live in
//! [`crate::execution::SequenceStateCore`]. This object composes that core with
//! DeepSeek-V4-specific layer and predictor state.

#[cfg(feature = "cuda")]
use ferrule_common::Error;
use ferrule_common::Result;

use crate::execution::{SequenceStateCore, SequenceStepBinding};
#[cfg(feature = "cuda")]
use crate::moe::prediction::ExpertBatchAccessEvent;
use crate::moe::prediction::ScoreBasedExpertPredictor;

use super::layer::DeepSeekV4LayerState;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default)]
pub(crate) struct DeepSeekV4PagedKvBinding {
    pub(crate) physical_block_slots: Vec<i32>,
    pub(crate) sequence_len: usize,
    pub(crate) page_tokens: usize,
    pub(crate) layer_count: usize,
}

#[cfg(feature = "cuda")]
impl DeepSeekV4PagedKvBinding {
    pub(crate) fn retain_sequence_len(&mut self, sequence_len: usize) -> Result<()> {
        if self.page_tokens == 0 {
            return Err(Error::Model(
                "DeepSeek-V4 paged binding has zero page size".into(),
            ));
        }
        let retained_blocks = sequence_len.div_ceil(self.page_tokens);
        if retained_blocks > self.physical_block_slots.len() {
            return Err(Error::Model(format!(
                "DeepSeek-V4 paged binding prefix needs {retained_blocks} blocks but only {} are available",
                self.physical_block_slots.len()
            )));
        }
        self.physical_block_slots.truncate(retained_blocks);
        self.sequence_len = sequence_len;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub(crate) struct DeepSeekV4SequenceMoeAccessEvent {
    pub(crate) sequence_index: usize,
    pub(crate) event: ExpertBatchAccessEvent,
}

/// Per-sequence execution state.
///
/// This object owns all mutable DeepSeek-V4 per-sequence resources while the
/// composed core owns committed cursor/version/error semantics. It does not own
/// immutable prepared weights, backend-global expert residency, arena scratch, or
/// diagnostics.
#[derive(Debug)]
pub struct DeepSeekV4SequenceExecutionState {
    pub(crate) core: SequenceStateCore,
    /// Fully initialized target-layer attention/compressor/indexer/physical KV state.
    pub(crate) layers: Vec<DeepSeekV4LayerState>,
    /// Dedicated DSpark stage context state. These caches hold only committed
    /// projected target context; proposal-block KV remains ephemeral scratch.
    pub(crate) dspark_stages: Vec<DeepSeekV4LayerState>,
    /// Sequence-specific expert prediction state.
    pub(crate) predictor: ScoreBasedExpertPredictor,
    #[cfg(feature = "cuda")]
    pub(crate) paged_kv_binding: Option<DeepSeekV4PagedKvBinding>,
}

impl DeepSeekV4SequenceExecutionState {
    pub fn new(
        layers: Vec<DeepSeekV4LayerState>,
        dspark_stages: Vec<DeepSeekV4LayerState>,
        num_routed_experts: usize,
    ) -> Self {
        let max_layers = layers.len();
        Self {
            core: SequenceStateCore::new(),
            layers,
            dspark_stages,
            predictor: ScoreBasedExpertPredictor::new(max_layers, num_routed_experts),
            #[cfg(feature = "cuda")]
            paged_kv_binding: None,
        }
    }

    pub fn max_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn dspark_stage_count(&self) -> usize {
        self.dspark_stages.len()
    }

    pub fn generation(&self) -> u64 {
        self.core.generation()
    }

    pub fn position(&self) -> usize {
        self.core.position()
    }

    pub fn is_poisoned(&self) -> bool {
        self.core.is_poisoned()
    }

    pub fn expert_predictor(&self) -> &ScoreBasedExpertPredictor {
        &self.predictor
    }

    /// Starts a mutation against the currently committed generation and cursor.
    pub fn begin_step(&self) -> Result<SequenceStepBinding> {
        self.core.begin_step()
    }

    /// Publishes a successfully staged mutation by advancing the committed cursor.
    pub fn commit_step(&mut self, binding: SequenceStepBinding, rows: usize) -> Result<()> {
        self.core.commit_step(binding, rows)
    }

    /// Marks staged state as non-reusable when rollback is unavailable.
    pub fn poison_step(&mut self, binding: SequenceStepBinding) {
        self.core.poison_step(binding);
    }

    /// Marks the sequence as poisoned after an unrecoverable error.
    pub fn poison(&mut self) {
        self.core.poison();
    }

    /// Resets semantic state while retaining reusable layer/KV capacity.
    pub fn reset_for_reuse(&mut self) {
        for state in &mut self.layers {
            state.reset_sequence();
        }
        for state in &mut self.dspark_stages {
            state.reset_sequence();
        }
        self.finish_reset();
    }

    /// Resets semantic state and releases all per-sequence layer/KV capacity.
    pub fn release_capacity(&mut self) {
        for state in &mut self.layers {
            state.release_sequence_capacity();
        }
        for state in &mut self.dspark_stages {
            state.release_sequence_capacity();
        }
        self.finish_reset();
    }

    fn finish_reset(&mut self) {
        self.predictor.clear();
        #[cfg(feature = "cuda")]
        {
            self.paged_kv_binding = None;
        }
        self.core.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_increments_generation_and_clears_state() {
        let mut seq = DeepSeekV4SequenceExecutionState::new(Vec::new(), Vec::new(), 256);
        let binding = seq.begin_step().unwrap();
        seq.commit_step(binding, 42).unwrap();
        let gen0 = seq.generation();
        seq.release_capacity();
        assert_eq!(seq.position(), 0);
        assert_eq!(seq.generation(), gen0 + 1);
        assert!(!seq.is_poisoned());
    }

    #[test]
    fn poison_prevents_use_until_reset() {
        let mut seq = DeepSeekV4SequenceExecutionState::new(Vec::new(), Vec::new(), 64);
        assert!(!seq.is_poisoned());
        seq.poison();
        assert!(seq.is_poisoned());
        seq.reset_for_reuse();
        assert!(!seq.is_poisoned());
    }

    #[test]
    fn reset_invalidates_existing_binding() {
        let mut seq = DeepSeekV4SequenceExecutionState::new(Vec::new(), Vec::new(), 64);
        let initial = seq.begin_step().unwrap();
        seq.commit_step(initial, 15).unwrap();
        let stale = seq.begin_step().unwrap();
        seq.reset_for_reuse();
        assert!(seq.commit_step(stale, 1).is_err());
    }

    #[test]
    fn failed_step_keeps_cursor_uncommitted_and_blocks_reuse() {
        let mut seq = DeepSeekV4SequenceExecutionState::new(Vec::new(), Vec::new(), 64);
        let binding = seq.begin_step().unwrap();
        seq.poison_step(binding);
        assert_eq!(seq.position(), 0);
        assert!(seq.begin_step().is_err());

        seq.reset_for_reuse();
        let retry = seq.begin_step().unwrap();
        seq.commit_step(retry, 3).unwrap();
        assert_eq!(seq.position(), 3);
    }

    #[test]
    fn interleaved_sequences_commit_independently() {
        let mut a = DeepSeekV4SequenceExecutionState::new(Vec::new(), Vec::new(), 64);
        let mut b = DeepSeekV4SequenceExecutionState::new(Vec::new(), Vec::new(), 64);
        let a0 = a.begin_step().unwrap();
        let b0 = b.begin_step().unwrap();
        a.commit_step(a0, 5).unwrap();
        b.commit_step(b0, 2).unwrap();
        let a1 = a.begin_step().unwrap();
        a.commit_step(a1, 1).unwrap();
        assert_eq!(a.position(), 6);
        assert_eq!(b.position(), 2);

        a.release_capacity();
        assert_eq!(a.position(), 0);
        assert_eq!(b.position(), 2);
    }
}
