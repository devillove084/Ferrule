//! Model-neutral sequence identity, checkout context, and fallible lifecycle contracts.
//! Concrete state and the standard clone-based lifecycle live in `state`.

use crate::execution::{SequenceStateCore, SequenceTopologyId};
use crate::runner::SequenceStateReleaseError;
use ferrule_common::Result;
use ferrule_common::execution::ExecutionTransactionId;

/// Public identity/core projection required by the model-neutral transaction shell.
///
/// Model sequence attachments do not need to be `Clone`. Their fallible working-copy,
/// fork, reset, and release behavior is supplied separately by
/// [`DecoderSequenceLifecycle`].
pub trait DecoderSequence {
    fn topology_id(&self) -> SequenceTopologyId;
    fn core(&self) -> &SequenceStateCore;
    fn core_mut(&mut self) -> &mut SequenceStateCore;
}
/// Context supplied when a committed sequence is checked out into a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderSequenceCheckout {
    transaction: ExecutionTransactionId,
    sequence_index: usize,
    source_index: usize,
}
impl DecoderSequenceCheckout {
    pub const fn new(
        transaction: ExecutionTransactionId,
        sequence_index: usize,
        source_index: usize,
    ) -> Self {
        Self {
            transaction,
            sequence_index,
            source_index,
        }
    }
    pub const fn transaction(self) -> ExecutionTransactionId {
        self.transaction
    }
    pub const fn sequence_index(self) -> usize {
        self.sequence_index
    }
    pub const fn source_index(self) -> usize {
        self.source_index
    }
}
/// Complete fallible lifecycle for model-owned sequence state.
///
/// The transaction shell never clones sequence state. A model may deep-copy,
/// checkpoint, borrow from a pool, or otherwise construct a working topology in
/// `checkout`. Failed release returns the exact original state to its caller.
pub trait DecoderSequenceLifecycle<S> {
    fn create(&mut self) -> Result<S>;
    fn checkout(&mut self, request: DecoderSequenceCheckout, source: &S) -> Result<S>;
    fn logical_fork(&mut self, source: &S, expected_position: usize) -> Result<S>;
    fn reset(&mut self, state: &mut S) -> Result<()>;
    fn try_release(&mut self, state: S) -> std::result::Result<(), SequenceStateReleaseError<S>>;
}

#[cfg(test)]
#[path = "../../tests/unit/decoder/sequence.rs"]
mod tests;
