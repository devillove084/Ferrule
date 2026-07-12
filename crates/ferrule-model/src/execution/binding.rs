use ferrule_common::execution::{
    ExecutionBatch, ForwardMode, ForwardPhase, KvWriteSlot, LogitsRequest,
};
use ferrule_common::{Error, Result};

/// Exact aggregate execution shape used to select persistent resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionShapeKey {
    mode: ForwardMode,
    batch_tokens: usize,
    sequence_count: usize,
    max_query_tokens: usize,
}

impl ExecutionShapeKey {
    pub const fn new(
        mode: ForwardMode,
        batch_tokens: usize,
        sequence_count: usize,
        max_query_tokens: usize,
    ) -> Self {
        Self {
            mode,
            batch_tokens,
            sequence_count,
            max_query_tokens,
        }
    }

    /// Derives the exact shape represented by a packed execution batch.
    pub fn from_batch(batch: &ExecutionBatch) -> Result<Self> {
        validate_basic_batch_shape(batch)?;
        let max_query_tokens = batch
            .sequences()
            .iter()
            .map(|sequence| sequence.query.end - sequence.query.start)
            .max()
            .unwrap_or(0);
        let max_query_tokens = usize::try_from(max_query_tokens).map_err(|_| {
            execution_error("maximum query length cannot be represented as a host size")
        })?;
        Ok(Self::new(
            batch.mode(),
            batch.len(),
            batch.sequences().len(),
            max_query_tokens,
        ))
    }

    pub const fn mode(self) -> ForwardMode {
        self.mode
    }

    /// Alias for callers that describe aggregate mode as the invocation phase.
    pub const fn phase(self) -> ForwardMode {
        self.mode
    }

    pub const fn batch_tokens(self) -> usize {
        self.batch_tokens
    }

    pub const fn sequence_count(self) -> usize {
        self.sequence_count
    }

    pub const fn max_query_tokens(self) -> usize {
        self.max_query_tokens
    }
}

/// Borrowed dynamic binding for one prepared execution invocation.
///
/// Token IDs, positions, KV slots, and logits intent remain borrowed from the
/// caller's [`ExecutionBatch`]; construction never clones its packed vectors.
#[derive(Debug, Clone, Copy)]
pub struct PreparedStepBinding<'a> {
    plan_generation: u64,
    shape: ExecutionShapeKey,
    token_ids: &'a [u32],
    positions: &'a [u32],
    kv_write_slots: &'a [Option<KvWriteSlot>],
    route_count: usize,
    expert_generation: u64,
    logits: &'a [LogitsRequest],
}

impl<'a> PreparedStepBinding<'a> {
    /// Binds an explicitly selected exact shape to the borrowed batch metadata.
    pub fn new(
        plan_generation: u64,
        shape: ExecutionShapeKey,
        batch: &'a ExecutionBatch,
        route_count: usize,
        expert_generation: u64,
    ) -> Result<Self> {
        let actual_shape = ExecutionShapeKey::from_batch(batch)?;
        if shape != actual_shape {
            return Err(execution_error(format!(
                "prepared step shape {shape:?} does not match execution batch shape {actual_shape:?}"
            )));
        }
        Ok(Self {
            plan_generation,
            shape,
            token_ids: batch.token_ids(),
            positions: batch.positions(),
            kv_write_slots: batch.kv_write_slots(),
            route_count,
            expert_generation,
            logits: batch.logits(),
        })
    }

    /// Derives and freezes the exact shape directly from the borrowed batch.
    pub fn from_batch(
        plan_generation: u64,
        batch: &'a ExecutionBatch,
        route_count: usize,
        expert_generation: u64,
    ) -> Result<Self> {
        let shape = ExecutionShapeKey::from_batch(batch)?;
        Self::new(
            plan_generation,
            shape,
            batch,
            route_count,
            expert_generation,
        )
    }

    pub const fn plan_generation(&self) -> u64 {
        self.plan_generation
    }

    pub const fn phase(&self) -> ForwardMode {
        self.shape.mode
    }

    pub const fn shape(&self) -> ExecutionShapeKey {
        self.shape
    }

    pub const fn token_ids(&self) -> &'a [u32] {
        self.token_ids
    }

    pub const fn positions(&self) -> &'a [u32] {
        self.positions
    }

    pub const fn kv_write_slots(&self) -> &'a [Option<KvWriteSlot>] {
        self.kv_write_slots
    }

    pub const fn route_count(&self) -> usize {
        self.route_count
    }

    pub const fn expert_generation(&self) -> u64 {
        self.expert_generation
    }

    pub const fn logits(&self) -> &'a [LogitsRequest] {
        self.logits
    }
}

fn validate_basic_batch_shape(batch: &ExecutionBatch) -> Result<()> {
    let rows = batch.token_ids().len();
    if rows == 0 {
        return Err(execution_error(
            "prepared step requires at least one packed row",
        ));
    }
    for (name, actual) in [
        ("positions", batch.positions().len()),
        ("KV write slots", batch.kv_write_slots().len()),
        ("logits intent", batch.logits().len()),
    ] {
        if actual != rows {
            return Err(execution_error(format!(
                "prepared step {name} length {actual} does not match packed row count {rows}"
            )));
        }
    }
    if batch.sequences().is_empty() {
        return Err(execution_error(
            "prepared step requires at least one sequence",
        ));
    }

    let row_count = u32::try_from(rows)
        .map_err(|_| execution_error("packed row count cannot be represented by query ranges"))?;
    let mut expected_start = 0_u32;
    let mut saw_prefill = false;
    let mut saw_decode = false;
    for (index, sequence) in batch.sequences().iter().enumerate() {
        if sequence.query.start != expected_start
            || sequence.query.start >= sequence.query.end
            || sequence.query.end > row_count
        {
            return Err(execution_error(format!(
                "sequence {index} query range {:?} is not a contiguous non-empty packed-row span",
                sequence.query
            )));
        }
        let query_len = sequence.query.end - sequence.query.start;
        let expected_sequence_len = sequence
            .context_len
            .checked_add(query_len)
            .ok_or_else(|| execution_error(format!("sequence {index} logical length overflows")))?;
        if sequence.sequence_len != expected_sequence_len {
            return Err(execution_error(format!(
                "sequence {index} logical length {} does not match context {} plus query {query_len}",
                sequence.sequence_len, sequence.context_len
            )));
        }
        let start = usize::try_from(sequence.query.start)
            .map_err(|_| execution_error("query start cannot be represented as a host index"))?;
        let end = usize::try_from(sequence.query.end)
            .map_err(|_| execution_error("query end cannot be represented as a host index"))?;
        for (offset, position) in batch.positions()[start..end].iter().enumerate() {
            let offset = u32::try_from(offset)
                .map_err(|_| execution_error("position offset cannot be represented as u32"))?;
            let expected = sequence
                .context_len
                .checked_add(offset)
                .ok_or_else(|| execution_error(format!("sequence {index} position overflows")))?;
            if *position != expected {
                return Err(execution_error(format!(
                    "packed position {position} does not match expected position {expected}"
                )));
            }
        }
        match sequence.phase {
            ForwardPhase::Prefill => saw_prefill = true,
            ForwardPhase::Decode => saw_decode = true,
        }
        expected_start = sequence.query.end;
    }
    if expected_start != row_count {
        return Err(execution_error(format!(
            "sequence queries cover {expected_start} packed rows, expected {row_count}"
        )));
    }
    let mode_matches = match batch.mode() {
        ForwardMode::Prefill => saw_prefill && !saw_decode,
        ForwardMode::Decode => !saw_prefill && saw_decode,
        ForwardMode::Mixed => saw_prefill && saw_decode,
    };
    if !mode_matches {
        return Err(execution_error(format!(
            "sequence phases do not match aggregate mode {:?}",
            batch.mode()
        )));
    }
    Ok(())
}

fn execution_error(message: impl Into<String>) -> Error {
    Error::Execution(message.into())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use ferrule_common::execution::{ExecutionSequence, StateSlot};

    use super::*;

    fn batch() -> ExecutionBatch {
        ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![11, 12],
            vec![3, 4],
            vec![None, None],
            vec![
                LogitsRequest::None,
                LogitsRequest::TopK(NonZeroU32::new(2).unwrap()),
            ],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Prefill,
                0..2,
                3,
                5,
                0..0,
            )],
            Vec::new(),
        )
    }

    #[test]
    fn binding_borrows_packed_vectors_and_freezes_metadata() {
        let batch = batch();
        let binding = PreparedStepBinding::from_batch(7, &batch, 16, 9).unwrap();
        assert_eq!(binding.plan_generation(), 7);
        assert_eq!(binding.phase(), ForwardMode::Prefill);
        assert_eq!(binding.route_count(), 16);
        assert_eq!(binding.expert_generation(), 9);
        assert!(std::ptr::eq(
            binding.token_ids().as_ptr(),
            batch.token_ids().as_ptr()
        ));
        assert!(std::ptr::eq(
            binding.positions().as_ptr(),
            batch.positions().as_ptr()
        ));
    }

    #[test]
    fn binding_rejects_mismatched_shape() {
        let batch = batch();
        let wrong = ExecutionShapeKey::new(ForwardMode::Prefill, 1, 1, 2);
        assert!(matches!(
            PreparedStepBinding::new(0, wrong, &batch, 0, 0),
            Err(Error::Execution(_))
        ));
    }

    #[test]
    fn binding_rejects_invalid_parallel_lengths() {
        let invalid = ExecutionBatch::new(
            ForwardMode::Decode,
            vec![1],
            Vec::new(),
            vec![None],
            vec![LogitsRequest::None],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Decode,
                0..1,
                0,
                1,
                0..0,
            )],
            Vec::new(),
        );
        assert!(matches!(
            PreparedStepBinding::from_batch(0, &invalid, 0, 0),
            Err(Error::Execution(_))
        ));
    }
}
