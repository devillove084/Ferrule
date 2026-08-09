use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use ferrule_common::{Error, Result};

static NEXT_SEQUENCE_TOPOLOGY_ID: AtomicU64 = AtomicU64::new(1);

/// Stable identity of one committed sequence topology.
///
/// The identity is independent of cursor generations. Transaction-local working
/// copies retain it, while an explicit logical fork receives a fresh identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceTopologyId(NonZeroU64);

impl SequenceTopologyId {
    /// Allocates a fresh process-local topology identity.
    pub fn take() -> Self {
        let value = NEXT_SEQUENCE_TOPOLOGY_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .expect("sequence topology ID space is exhausted");
        Self(NonZeroU64::new(value).expect("sequence topology IDs start at one"))
    }

    /// Returns the non-zero numeric identity.
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Versioned binding for one staged sequence-state mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceStepBinding {
    generation: u64,
    committed_position: usize,
}

impl SequenceStepBinding {
    /// Generation observed when the step began.
    pub const fn generation(self) -> u64 {
        self.generation
    }

    /// Committed cursor observed when the step began.
    pub const fn committed_position(self) -> usize {
        self.committed_position
    }
}

/// A sequence commit that has already passed stale-binding and overflow checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ValidatedSequenceStepCommit {
    generation: u64,
    committed_position: usize,
    next_position: usize,
}

/// Model-family-neutral committed sequence lifecycle state.
///
/// Model-specific state should compose this core alongside its KV, predictor, and
/// other mutable resources instead of duplicating cursor/version/error semantics.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SequenceStateCore {
    generation: u64,
    position: usize,
    poisoned: bool,
}

impl SequenceStateCore {
    /// Creates an unpoisoned sequence at position zero and generation zero.
    pub const fn new() -> Self {
        Self {
            generation: 0,
            position: 0,
            poisoned: false,
        }
    }

    /// Creates an unpoisoned sequence with an already committed initial cursor.
    pub const fn with_position(position: usize) -> Self {
        Self {
            generation: 0,
            position,
            poisoned: false,
        }
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn position(&self) -> usize {
        self.position
    }

    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Starts a mutation against the currently committed generation and cursor.
    pub fn begin_step(&self) -> Result<SequenceStepBinding> {
        if self.poisoned {
            return Err(execution_error(
                "sequence state is poisoned; reset or release it before reuse",
            ));
        }
        Ok(SequenceStepBinding {
            generation: self.generation,
            committed_position: self.position,
        })
    }

    /// Publishes a successfully staged mutation by advancing the committed cursor.
    pub fn commit_step(&mut self, binding: SequenceStepBinding, rows: usize) -> Result<()> {
        let commit = self.validate_step_commit(binding, rows)?;
        self.apply_validated_step_commit(commit);
        Ok(())
    }

    pub(crate) fn validate_step_commit(
        &self,
        binding: SequenceStepBinding,
        rows: usize,
    ) -> Result<ValidatedSequenceStepCommit> {
        if self.poisoned {
            return Err(execution_error("cannot commit a poisoned sequence state"));
        }
        if binding.generation != self.generation || binding.committed_position != self.position {
            return Err(execution_error(format!(
                "stale sequence binding: generation/position {}/{} no longer matches {}/{}",
                binding.generation, binding.committed_position, self.generation, self.position
            )));
        }
        let next_position = self
            .position
            .checked_add(rows)
            .ok_or_else(|| execution_error("committed sequence position overflow"))?;
        Ok(ValidatedSequenceStepCommit {
            generation: self.generation,
            committed_position: self.position,
            next_position,
        })
    }

    pub(crate) fn apply_validated_step_commit(&mut self, commit: ValidatedSequenceStepCommit) {
        debug_assert!(!self.poisoned);
        debug_assert_eq!(self.generation, commit.generation);
        debug_assert_eq!(self.position, commit.committed_position);
        self.position = commit.next_position;
    }

    /// Marks staged state as non-reusable when rollback is unavailable.
    pub fn poison_step(&mut self, binding: SequenceStepBinding) {
        if binding.generation == self.generation {
            self.poisoned = true;
        }
    }

    /// Marks the current generation as non-reusable.
    pub fn poison(&mut self) {
        self.poisoned = true;
    }

    /// Clears committed semantic state and invalidates all outstanding bindings.
    pub fn reset(&mut self) {
        self.position = 0;
        self.poisoned = false;
        self.generation = self.generation.wrapping_add(1);
    }

    /// Returns an independent clean core at the same cursor and a newer generation.
    pub fn forked(&self) -> Result<Self> {
        self.begin_step()?;
        Ok(Self {
            generation: self.generation.wrapping_add(1),
            position: self.position,
            poisoned: false,
        })
    }

    /// Restores committed cursor state while invalidating both current and snapshot bindings.
    pub fn restore_from(&mut self, snapshot: &Self) {
        self.position = snapshot.position;
        self.poisoned = false;
        self.generation = self.generation.max(snapshot.generation).wrapping_add(1);
    }
}

fn execution_error(message: impl Into<String>) -> Error {
    Error::Execution {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_ids_are_nonzero_and_unique() {
        let first = SequenceTopologyId::take();
        let second = SequenceTopologyId::take();

        assert_ne!(first, second);
        assert_ne!(first.get(), 0);
        assert_ne!(second.get(), 0);
    }

    #[test]
    fn commit_advances_only_matching_binding() {
        let mut core = SequenceStateCore::new();
        let binding = core.begin_step().unwrap();
        core.commit_step(binding, 3).unwrap();
        assert_eq!(core.position(), 3);
        assert!(matches!(
            core.commit_step(binding, 1),
            Err(Error::Execution { message: _ })
        ));
    }

    #[test]
    fn reset_invalidates_bindings_and_clears_poison() {
        let mut core = SequenceStateCore::with_position(9);
        let binding = core.begin_step().unwrap();
        core.poison_step(binding);
        core.reset();
        assert_eq!(core.position(), 0);
        assert_eq!(core.generation(), 1);
        assert!(!core.is_poisoned());
        assert!(matches!(
            core.commit_step(binding, 1),
            Err(Error::Execution { message: _ })
        ));
    }

    #[test]
    fn errors_are_neutral_execution_errors() {
        let mut core = SequenceStateCore::new();
        core.poison();
        let error = core.begin_step().unwrap_err();
        match error {
            Error::Execution { message } => {
                assert!(!message.contains("DeepSeek"));
                assert!(message.contains("sequence state"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}
