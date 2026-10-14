//! Metadata preflight for the explicit Mixed attention provider. This does not
//! issue KV write authority or allocate/prepare a backend arithmetic owner.
use super::GenericDecoderOptions;
#[cfg(any(feature = "cuda", test))]
use super::PackedDecoderBatch;
use crate::transformer::{Attention, BoundDecoderResources, NumericFp8Precision};
use ferrule_common::{Error, Result};
#[cfg(any(feature = "cuda", test))]
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MixedAttentionCapacity {
    max_rows: usize,
    max_sequences: usize,
    max_positions: usize,
}

fn invalid(message: &str) -> Error {
    Error::Execution {
        message: format!("Mixed BF16 attention: {message}"),
    }
}

impl MixedAttentionCapacity {
    pub(crate) fn for_resources(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
    ) -> Result<Option<Self>> {
        Self::for_profile(
            options.hybrid_cuda_numeric_fp8_precision(),
            resources
                .spec()
                .layers()
                .iter()
                .any(|l| matches!(l.attention(), Attention::Gqa(_))),
            options.capabilities().max_batch_tokens,
            options.capabilities().max_sequences,
            options.max_positions(),
        )
    }

    fn for_profile(
        precision: Option<NumericFp8Precision>,
        has_gqa: bool,
        max_rows: usize,
        max_sequences: usize,
        max_positions: usize,
    ) -> Result<Option<Self>> {
        if precision != Some(NumericFp8Precision::Bf16ModelMixedF32Accumulate) || !has_gqa {
            return Ok(None);
        }
        if !(1..=1024).contains(&max_positions) {
            return Err(invalid(
                "initial image admission requires max_positions in 1..=1024; no late context fallback",
            ));
        }
        if max_rows == 0
            || max_rows > i32::MAX as usize
            || max_sequences == 0
            || max_sequences > i32::MAX as usize
        {
            return Err(invalid(
                "positive native-i32 row/sequence capacities required",
            ));
        }
        Ok(Some(Self {
            max_rows,
            max_sequences,
            max_positions,
        }))
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn validate_batch(self, batch: &PackedDecoderBatch) -> Result<()> {
        self.validate_cohorts(
            batch.positions(),
            batch.row_to_sequence(),
            batch
                .sequences()
                .iter()
                .map(|s| (s.query(), s.context_len(), s.sequence_len())),
        )
    }

    // Cohorts must be complete sequence matrices, not subsets/padded M or KV.
    // The packed batch remains the authority for positions, independent of phase.
    #[cfg(any(feature = "cuda", test))]
    fn validate_cohorts(
        self,
        positions: &[usize],
        sequence_ids: &[usize],
        cohorts: impl Iterator<Item = (Range<usize>, usize, usize)>,
    ) -> Result<()> {
        if positions.is_empty()
            || positions.len() > self.max_rows
            || sequence_ids.len() != positions.len()
        {
            return Err(invalid(
                "packed rows exceed admitted capacity or metadata extent",
            ));
        }
        let mut end = 0;
        let mut count = 0;
        for (sequence, (query, context, tokens)) in cohorts.enumerate() {
            if sequence >= self.max_sequences
                || query.start != end
                || query.end <= query.start
                || query.end > positions.len()
                || tokens > self.max_positions
                || context.checked_add(query.len()) != Some(tokens)
            {
                return Err(invalid(
                    "cohorts must partition rows with context + query = KV within admitted capacity",
                ));
            }
            if query.clone().enumerate().any(|(offset, row)| {
                sequence_ids[row] != sequence || context.checked_add(offset) != Some(positions[row])
            }) {
                return Err(invalid(
                    "cohort positions/sequence IDs differ from contiguous history",
                ));
            }
            end = query.end;
            count += 1;
        }
        if count == 0 || end != positions.len() {
            return Err(invalid("incomplete sequence cohort"));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/decoder/mixed_attention/plan.rs"]
mod tests;
