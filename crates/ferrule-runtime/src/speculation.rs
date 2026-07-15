//! DSpark speculative decoding transaction.
//!
//! Implements the exact DSpark transaction from Section 3.14 of the roadmap:
//!
//! ```text
//! draft proposal
//!  -> target branch KV/state
//!  -> packed exact verification
//!  -> device accepted-prefix result
//!  -> commit accepted prefix
//!  -> rollback rejected suffix
//!  -> update residency/acceptance telemetry
//! ```
//!
//! Unlike the former token-by-token `DraftModel`/`TargetModel` traits, this
//! module implements the correct DSpark pattern: the draft proposes V tokens
//! in one shot, the target verifies all V in a single packed forward pass,
//! and the longest accepted prefix is committed atomically.
//!
//! The draft source (MTP model) plugs in through [`DraftSource`].  The
//! verification uses the existing `MultiSessionRunner` packed batch path.

use std::num::NonZeroU32;
use std::time::Instant;

use ferrule_common::execution::{
    ExecutionBatch, ExecutionOutput, ForwardMode, ForwardPhase, LogitsOutput, LogitsRequest,
    StateSlot,
};
use ferrule_common::{Error, Result};
use ferrule_model::MultiSessionRunner;

// ── Draft source ──────────────────────────────────────────────────────

/// Source of draft token proposals for DSpark verification.
///
/// The MTP model implements this trait.  A simple greedy-decode fallback
/// (using the target model itself) can also implement it for testing.
pub trait DraftSource {
    /// Propose up to `max_tokens` candidate tokens given the current context.
    ///
    /// Returns the proposed token IDs.  The proposal may be shorter than
    /// `max_tokens` if the draft source decides to stop early (e.g., EOS).
    fn propose(&mut self, context: &[u32], max_tokens: usize) -> Result<Vec<u32>>;
}

// ── Transaction types ─────────────────────────────────────────────────

/// Verification width V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerificationWidth(pub usize);

/// Result of one DSpark verification cycle.
#[derive(Debug, Clone, PartialEq)]
pub struct DSparkCycleResult {
    /// Tokens accepted by the target (committed to the sequence).
    pub accepted: Vec<u32>,
    /// First rejected draft token, if any.
    pub rejected: Option<u32>,
    /// The target's own top-1 prediction at the rejection point.
    /// This token should be used as the next token instead of the rejected draft.
    pub target_correction: Option<u32>,
    /// Number of draft tokens proposed.
    pub proposed: usize,
    /// Wall-clock time for the complete cycle (draft + verify + commit/rollback).
    pub cycle_time_us: u64,
    /// Wall-clock time for the verification forward pass only.
    pub verify_time_us: u64,
}

impl DSparkCycleResult {
    /// Acceptance rate: accepted / proposed.
    pub fn acceptance_rate(&self) -> f32 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted.len() as f32 / self.proposed as f32
        }
    }

    /// Accepted tokens per second.
    pub fn accepted_tok_per_s(&self) -> f64 {
        if self.cycle_time_us == 0 {
            0.0
        } else {
            (self.accepted.len() as f64) * 1_000_000.0 / (self.cycle_time_us as f64)
        }
    }
}

/// Accumulated DSpark telemetry across multiple cycles.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DSparkMetrics {
    pub cycles: usize,
    pub proposed_tokens: usize,
    pub accepted_tokens: usize,
    pub rejected_tokens: usize,
    pub total_cycle_time_us: u64,
    pub total_verify_time_us: u64,
}

impl DSparkMetrics {
    pub fn acceptance_rate(&self) -> f32 {
        if self.proposed_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.proposed_tokens as f32
        }
    }

    pub fn accepted_tok_per_s(&self) -> f64 {
        if self.total_cycle_time_us == 0 {
            0.0
        } else {
            (self.accepted_tokens as f64) * 1_000_000.0 / (self.total_cycle_time_us as f64)
        }
    }

    pub fn mean_cycle_time_us(&self) -> u64 {
        if self.cycles == 0 {
            0
        } else {
            self.total_cycle_time_us / self.cycles as u64
        }
    }

    pub fn record(&mut self, result: &DSparkCycleResult) {
        self.cycles += 1;
        self.proposed_tokens += result.proposed;
        self.accepted_tokens += result.accepted.len();
        self.rejected_tokens += result.rejected.is_some() as usize;
        self.total_cycle_time_us += result.cycle_time_us;
        self.total_verify_time_us += result.verify_time_us;
    }
}

// ── DSpark transaction ────────────────────────────────────────────────

/// Runs one DSpark verification cycle.
///
/// Given a draft proposal of V tokens, this function:
///
/// 1. Creates a packed verification batch (1 sequence × V rows)
/// 2. Executes the target model forward pass
/// 3. Compares target top-1 with draft proposals
/// 4. Returns the accepted prefix and rejection info
///
/// The caller is responsible for committing the accepted prefix to the
/// sequence state (via `commit_multi_session_batch`) or rolling back
/// (via `rollback_multi_session_batch`).
///
/// # Arguments
///
/// * `runner` - The target model runner
/// * `state` - The sequence state to verify against (forked from the main sequence)
/// * `context` - Current committed context tokens
/// * `proposal` - Draft-proposed candidate tokens
/// * `top_k` - Top-k to request from the target (must be >= 1)
/// * `position` - Current position in the sequence
pub fn run_dspark_verification<R>(
    runner: &mut R,
    state: &mut R::SequenceState,
    _context: &[u32],
    proposal: &[u32],
    top_k: NonZeroU32,
    position: usize,
) -> Result<DSparkCycleResult>
where
    R: MultiSessionRunner,
{
    let cycle_start = Instant::now();
    let v = proposal.len();
    if v == 0 {
        return Ok(DSparkCycleResult {
            accepted: Vec::new(),
            rejected: None,
            target_correction: None,
            proposed: 0,
            cycle_time_us: 0,
            verify_time_us: 0,
        });
    }

    // Build a packed verification batch: 1 sequence × V rows.
    // Each row is a candidate token at position + offset.
    let token_ids: Vec<u32> = proposal.to_vec();
    let positions: Vec<u32> = (0..v).map(|i| (position + i) as u32).collect();
    let logits: Vec<LogitsRequest> = (0..v).map(|_| LogitsRequest::TopK(top_k)).collect();
    let kv_write_slots: Vec<Option<ferrule_common::execution::KvWriteSlot>> = vec![None; v];

    let batch = ExecutionBatch::new(
        ForwardMode::Prefill,
        token_ids,
        positions,
        kv_write_slots,
        logits,
        vec![ferrule_common::execution::ExecutionSequence::new(
            StateSlot::new(0),
            ForwardPhase::Prefill,
            0..v as u32,
            position as u32,
            (position + v) as u32,
            0..0,
        )],
        Vec::new(),
    );

    // Execute the packed verification pass.
    let verify_start = Instant::now();
    let output = runner.execute_multi_session_batch(std::slice::from_mut(state), &batch)?;
    let verify_time_us = verify_start.elapsed().as_micros() as u64;

    let output = output.ok_or_else(|| {
        Error::Execution("DSpark verification requires native packed execution support".into())
    })?;

    // Extract target top-1 for each row and compare with draft proposals.
    let accepted = compute_accepted_prefix(&output, proposal, v);
    let rejected = if accepted.len() < v {
        Some(proposal[accepted.len()])
    } else {
        None
    };

    // Get the target's correction token at the rejection point.
    let target_correction = if let Some(reject_idx) = (accepted.len() < v).then_some(accepted.len())
    {
        output
            .logits
            .iter()
            .find(|row| row.input_row as usize == reject_idx)
            .and_then(|row| {
                if let LogitsOutput::TopK(logits) = &row.logits {
                    logits.first().map(|item| item.token_id)
                } else {
                    None
                }
            })
    } else {
        None
    };

    let cycle_time_us = cycle_start.elapsed().as_micros() as u64;

    Ok(DSparkCycleResult {
        accepted,
        rejected,
        target_correction,
        proposed: v,
        cycle_time_us,
        verify_time_us,
    })
}

/// Compare target top-1 predictions with draft proposals and return the
/// longest accepted prefix.
fn compute_accepted_prefix(output: &ExecutionOutput, proposal: &[u32], v: usize) -> Vec<u32> {
    let mut accepted = Vec::new();
    for row in &output.logits {
        let row_idx = row.input_row as usize;
        if row_idx >= v {
            continue;
        }
        if let LogitsOutput::TopK(logits) = &row.logits {
            if let Some(target_top1) = logits.first() {
                if target_top1.token_id == proposal[row_idx] {
                    accepted.push(proposal[row_idx]);
                } else {
                    break;
                }
            } else {
                break;
            }
        } else {
            break;
        }
    }
    accepted
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_common::execution::LogitsRow;

    #[test]
    fn accepted_prefix_all_accepted() {
        let proposal = vec![10, 11, 12, 13];
        let output = ExecutionOutput::new(vec![
            LogitsRow::new(0, LogitsOutput::TopK(vec![logit(10, 1.0)])),
            LogitsRow::new(1, LogitsOutput::TopK(vec![logit(11, 1.0)])),
            LogitsRow::new(2, LogitsOutput::TopK(vec![logit(12, 1.0)])),
            LogitsRow::new(3, LogitsOutput::TopK(vec![logit(13, 1.0)])),
        ]);
        let accepted = compute_accepted_prefix(&output, &proposal, 4);
        assert_eq!(accepted, vec![10, 11, 12, 13]);
    }

    #[test]
    fn accepted_prefix_partial_rejection() {
        let proposal = vec![10, 11, 12, 13];
        let output = ExecutionOutput::new(vec![
            LogitsRow::new(0, LogitsOutput::TopK(vec![logit(10, 1.0)])),
            LogitsRow::new(1, LogitsOutput::TopK(vec![logit(11, 1.0)])),
            LogitsRow::new(2, LogitsOutput::TopK(vec![logit(99, 1.0)])), // mismatch
            LogitsRow::new(3, LogitsOutput::TopK(vec![logit(13, 1.0)])),
        ]);
        let accepted = compute_accepted_prefix(&output, &proposal, 4);
        assert_eq!(accepted, vec![10, 11]);
    }

    #[test]
    fn accepted_prefix_first_rejected() {
        let proposal = vec![10, 11, 12];
        let output = ExecutionOutput::new(vec![
            LogitsRow::new(0, LogitsOutput::TopK(vec![logit(99, 1.0)])), // mismatch
            LogitsRow::new(1, LogitsOutput::TopK(vec![logit(11, 1.0)])),
            LogitsRow::new(2, LogitsOutput::TopK(vec![logit(12, 1.0)])),
        ]);
        let accepted = compute_accepted_prefix(&output, &proposal, 3);
        assert!(accepted.is_empty());
    }

    #[test]
    fn dspark_metrics_record_and_rates() {
        let mut metrics = DSparkMetrics::default();
        metrics.record(&DSparkCycleResult {
            accepted: vec![10, 11],
            rejected: Some(12),
            target_correction: Some(99),
            proposed: 3,
            cycle_time_us: 100_000,
            verify_time_us: 80_000,
        });
        metrics.record(&DSparkCycleResult {
            accepted: vec![20, 21, 22, 23],
            rejected: None,
            target_correction: None,
            proposed: 4,
            cycle_time_us: 120_000,
            verify_time_us: 90_000,
        });
        assert_eq!(metrics.cycles, 2);
        assert_eq!(metrics.proposed_tokens, 7);
        assert_eq!(metrics.accepted_tokens, 6);
        assert_eq!(metrics.rejected_tokens, 1);
        assert_eq!(metrics.total_cycle_time_us, 220_000);
        assert_eq!(metrics.total_verify_time_us, 170_000);
        assert!((metrics.acceptance_rate() - 6.0 / 7.0).abs() < 1e-6);
        assert!((metrics.accepted_tok_per_s() - 6.0 * 1_000_000.0 / 220_000.0).abs() < 0.01);
        assert_eq!(metrics.mean_cycle_time_us(), 110_000);
    }

    #[test]
    fn cycle_result_accepted_tok_per_s() {
        let result = DSparkCycleResult {
            accepted: vec![10, 11, 12, 13],
            rejected: None,
            target_correction: None,
            proposed: 4,
            cycle_time_us: 250_000,
            verify_time_us: 200_000,
        };
        assert!((result.accepted_tok_per_s() - 4.0 * 1_000_000.0 / 250_000.0).abs() < 0.01);
        assert!((result.acceptance_rate() - 1.0).abs() < 1e-6);
    }

    fn logit(token_id: u32, logit: f32) -> ferrule_common::execution::TokenLogit {
        ferrule_common::execution::TokenLogit { token_id, logit }
    }
}
