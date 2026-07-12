//! Per-sequence execution state for DeepSeek-V4.
//!
//! Model-neutral cursor, generation, and poison semantics live in
//! [`crate::execution::SequenceStateCore`]. This object composes that core with
//! DeepSeek-V4-specific layer and predictor state.

use ferrule_common::Result;

use crate::execution::{SequenceStateCore, SequenceStepBinding};
use crate::moe::prediction::ScoreBasedExpertPredictor;

use super::layer::DeepSeekV4LayerState;

/// Per-sequence execution state.
///
/// This object owns all mutable DeepSeek-V4 per-sequence resources while the
/// composed core owns committed cursor/version/error semantics. It does not own
/// immutable prepared weights, backend-global expert residency, arena scratch, or
/// diagnostics.
#[derive(Debug)]
pub struct DeepSeekV4SequenceExecutionState {
    pub(crate) core: SequenceStateCore,
    /// Fully initialized per-layer attention/compressor/indexer/physical KV state.
    pub(crate) layers: Vec<DeepSeekV4LayerState>,
    /// Sequence-specific expert prediction state.
    pub(crate) predictor: ScoreBasedExpertPredictor,
}

impl DeepSeekV4SequenceExecutionState {
    pub fn new(layers: Vec<DeepSeekV4LayerState>, num_routed_experts: usize) -> Self {
        let max_layers = layers.len();
        Self {
            core: SequenceStateCore::new(),
            layers,
            predictor: ScoreBasedExpertPredictor::new(max_layers, num_routed_experts),
        }
    }

    pub fn max_layers(&self) -> usize {
        self.layers.len()
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
        self.finish_reset();
    }

    /// Resets semantic state and releases all per-sequence layer/KV capacity.
    pub fn release_capacity(&mut self) {
        for state in &mut self.layers {
            state.release_sequence_capacity();
        }
        self.finish_reset();
    }

    fn finish_reset(&mut self) {
        self.predictor.clear();
        self.core.reset();
    }
}

/// A snapshot of sequence state for checkpoint/restore.
///
/// Graph and diagnostics are not part of a checkpoint. Physical CUDA KV is
/// sequence-owned but still requires a separate operator-aware D2D checkpoint API.
#[derive(Debug)]
pub struct DeepSeekV4SequenceCheckpoint {
    pub(crate) core: SequenceStateCore,
    pub(crate) layers: Vec<DeepSeekV4LayerState>,
    pub(crate) predictor: ScoreBasedExpertPredictor,
}

impl DeepSeekV4SequenceCheckpoint {
    pub fn generation(&self) -> u64 {
        self.core.generation()
    }

    pub fn position(&self) -> usize {
        self.core.position()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_increments_generation_and_clears_state() {
        let mut seq = DeepSeekV4SequenceExecutionState::new(Vec::new(), 256);
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
        let mut seq = DeepSeekV4SequenceExecutionState::new(Vec::new(), 64);
        assert!(!seq.is_poisoned());
        seq.poison();
        assert!(seq.is_poisoned());
        seq.reset_for_reuse();
        assert!(!seq.is_poisoned());
    }

    #[test]
    fn reset_invalidates_existing_binding() {
        let mut seq = DeepSeekV4SequenceExecutionState::new(Vec::new(), 64);
        let initial = seq.begin_step().unwrap();
        seq.commit_step(initial, 15).unwrap();
        let stale = seq.begin_step().unwrap();
        seq.reset_for_reuse();
        assert!(seq.commit_step(stale, 1).is_err());
    }

    #[test]
    fn failed_step_keeps_cursor_uncommitted_and_blocks_reuse() {
        let mut seq = DeepSeekV4SequenceExecutionState::new(Vec::new(), 64);
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
        let mut a = DeepSeekV4SequenceExecutionState::new(Vec::new(), 64);
        let mut b = DeepSeekV4SequenceExecutionState::new(Vec::new(), 64);
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
