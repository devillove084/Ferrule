//! DSpark speculative verification primitives.
//!
//! Defines the causal verification contract used by the exact transaction from
//! Section 3.14 of the roadmap:
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
//! Unlike token-by-token draft/target loops, the draft proposes V tokens and
//! the target verifies them in one packed pass. Transaction ownership remains
//! with the caller: verification must run on a disposable model/KV branch.
//!
//! The draft source (MTP model) plugs in through [`DraftSource`].  The
//! verification uses the existing `MultiSessionRunner` packed batch path.

use std::num::NonZeroU32;
use std::time::Instant;

use ferrule_common::execution::{
    ExecutionBatch, ExecutionIntent, ExecutionOutput, ForwardMode, ForwardPhase, LogitsOutput,
    LogitsRequest, StateSlot, TokenLogit,
};
use ferrule_common::{Error, Result};
use ferrule_model::MultiSessionRunner;

use crate::cache::{KvPageManager, KvReservationBindings};
use crate::engine::NativeMultiSessionExecutor;

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

/// Target prediction at the current committed frontier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetFrontier {
    pub position: usize,
    pub top1: TokenLogit,
}

/// Result of one DSpark verification cycle.
#[derive(Debug, Clone, PartialEq)]
pub struct DSparkCycleResult {
    /// Tokens accepted by the target (committed to the sequence).
    pub accepted: Vec<u32>,
    /// First rejected draft token, if any.
    pub rejected: Option<u32>,
    /// The target's own top-1 prediction at the rejection point.
    /// This token should be used instead of the rejected draft.
    pub target_correction: Option<u32>,
    /// Target prediction at the resulting accepted-prefix frontier. For partial
    /// acceptance this equals the correction; for full acceptance it comes from
    /// the final packed row and avoids an extra target pass.
    pub target_next: Option<TokenLogit>,
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
/// * `proposal` - Draft-proposed candidate tokens
/// * `top_k` - Top-k to request from the target (must be >= 1)
/// * `frontier` - Target top-1 produced at the current committed position
pub fn run_dspark_verification<R>(
    executor: &mut NativeMultiSessionExecutor<R>,
    page_manager: &mut KvPageManager,
    source_state: &mut R::SequenceState,
    state_slot: StateSlot,
    generation: u64,
    proposal: &[u32],
    top_k: NonZeroU32,
    frontier: TargetFrontier,
) -> Result<DSparkCycleResult>
where
    R: MultiSessionRunner,
{
    let cycle_start = Instant::now();
    let width = proposal.len();
    if width == 0 {
        return Ok(DSparkCycleResult {
            accepted: Vec::new(),
            rejected: None,
            target_correction: None,
            target_next: Some(frontier.top1),
            proposed: 0,
            cycle_time_us: 0,
            verify_time_us: 0,
        });
    }

    let reservation = page_manager.reserve(state_slot, generation, width)?;
    if reservation.positions.start != frontier.position {
        let actual = reservation.positions.start;
        let error = Error::Execution(format!(
            "DSpark frontier position {} does not match committed KV position {actual}",
            frontier.position
        ));
        return Err(rollback_reservation(page_manager, reservation, error));
    }
    let bindings = match page_manager.reservation_bindings(&reservation) {
        Ok(bindings) => bindings,
        Err(error) => {
            return Err(rollback_reservation(page_manager, reservation, error));
        }
    };
    let verification_batch = match build_dspark_batch(
        proposal,
        frontier.position,
        &bindings,
        Some(top_k),
        ExecutionIntent::ProvisionalVerification,
    ) {
        Ok(batch) => batch,
        Err(error) => {
            return Err(rollback_reservation(page_manager, reservation, error));
        }
    };
    let mut verification_branch =
        match executor.fork_sequence_state_from(source_state, frontier.position) {
            Ok(branch) => branch,
            Err(error) => {
                return Err(rollback_reservation(page_manager, reservation, error));
            }
        };

    let verify_start = Instant::now();
    let output = match executor.execute_batch_with_kv(
        std::slice::from_mut(&mut verification_branch),
        &verification_batch,
        std::slice::from_ref(&reservation),
    ) {
        Ok(output) => output,
        Err(error) => {
            return Err(discard_verification_branch(
                executor,
                page_manager,
                reservation,
                verification_branch,
                error,
            ));
        }
    };
    let verify_time_us = verify_start.elapsed().as_micros() as u64;
    let verification = match verify_causal_prefix(&output, proposal, frontier.top1) {
        Ok(verification) => verification,
        Err(error) => {
            return Err(discard_verification_branch(
                executor,
                page_manager,
                reservation,
                verification_branch,
                error,
            ));
        }
    };

    let accepted_rows = verification.accepted;
    if accepted_rows == width {
        let freed = match page_manager.commit_prefix_with_freed(reservation, accepted_rows) {
            Ok(freed) => freed,
            Err(error) => {
                let backend_cleanup = executor.rollback_prepared_batch().err();
                let state_cleanup = executor.release_sequence_state(verification_branch).err();
                return Err(with_cleanup_errors(
                    error,
                    backend_cleanup,
                    None,
                    state_cleanup,
                ));
            }
        };
        if let Err(error) = executor.commit_prepared_batch() {
            promote_branch(executor, source_state, verification_branch)?;
            return Err(error);
        }
        promote_branch(executor, source_state, verification_branch)?;
        executor.release_kv_pages(&freed)?;
    } else {
        if let Err(error) = executor.rollback_prepared_batch() {
            return Err(discard_verification_branch(
                executor,
                page_manager,
                reservation,
                verification_branch,
                error,
            ));
        }
        if let Err(error) = executor.release_sequence_state(verification_branch) {
            return Err(rollback_reservation(page_manager, reservation, error));
        }
        if accepted_rows == 0 {
            page_manager.commit_prefix_with_freed(reservation, 0)?;
        } else {
            let prefix_view =
                match page_manager.reservation_prefix_view(&reservation, accepted_rows) {
                    Ok(view) => view,
                    Err(error) => {
                        return Err(rollback_reservation(page_manager, reservation, error));
                    }
                };
            let prefix_bindings = match page_manager.reservation_bindings(&prefix_view) {
                Ok(bindings) => bindings,
                Err(error) => {
                    return Err(rollback_reservation(page_manager, reservation, error));
                }
            };
            let replay_batch = match build_dspark_batch(
                &proposal[..accepted_rows],
                frontier.position,
                &prefix_bindings,
                None,
                ExecutionIntent::Committed,
            ) {
                Ok(batch) => batch,
                Err(error) => {
                    return Err(rollback_reservation(page_manager, reservation, error));
                }
            };
            let mut accepted_branch =
                match executor.fork_sequence_state_from(source_state, frontier.position) {
                    Ok(branch) => branch,
                    Err(error) => {
                        return Err(rollback_reservation(page_manager, reservation, error));
                    }
                };
            if let Err(error) = executor.execute_batch_with_kv(
                std::slice::from_mut(&mut accepted_branch),
                &replay_batch,
                std::slice::from_ref(&prefix_view),
            ) {
                return Err(discard_verification_branch(
                    executor,
                    page_manager,
                    reservation,
                    accepted_branch,
                    error,
                ));
            }
            let freed = match page_manager.commit_prefix_with_freed(reservation, accepted_rows) {
                Ok(freed) => freed,
                Err(error) => {
                    let backend_cleanup = executor.rollback_prepared_batch().err();
                    let state_cleanup = executor.release_sequence_state(accepted_branch).err();
                    return Err(with_cleanup_errors(
                        error,
                        backend_cleanup,
                        None,
                        state_cleanup,
                    ));
                }
            };
            if let Err(error) = executor.commit_prepared_batch() {
                promote_branch(executor, source_state, accepted_branch)?;
                return Err(error);
            }
            promote_branch(executor, source_state, accepted_branch)?;
            executor.release_kv_pages(&freed)?;
        }
    }

    let accepted = proposal[..accepted_rows].to_vec();
    let rejected = proposal.get(accepted_rows).copied();
    Ok(DSparkCycleResult {
        accepted,
        rejected,
        target_correction: rejected.map(|_| verification.target_next.token_id),
        target_next: Some(verification.target_next),
        proposed: width,
        cycle_time_us: cycle_start.elapsed().as_micros() as u64,
        verify_time_us,
    })
}

fn build_dspark_batch(
    tokens: &[u32],
    position: usize,
    bindings: &KvReservationBindings,
    top_k: Option<NonZeroU32>,
    intent: ExecutionIntent,
) -> Result<ExecutionBatch> {
    if tokens.is_empty() || bindings.write_slots.len() != tokens.len() {
        return Err(Error::Execution(
            "DSpark batch requires one KV write slot per non-empty token row".into(),
        ));
    }
    let position = u32::try_from(position)
        .map_err(|_| Error::Execution("DSpark verification position exceeds u32".into()))?;
    let width = u32::try_from(tokens.len())
        .map_err(|_| Error::Execution("DSpark verification width exceeds u32".into()))?;
    let sequence_len = position
        .checked_add(width)
        .ok_or_else(|| Error::Execution("DSpark verification sequence length overflow".into()))?;
    let block_count = u32::try_from(bindings.block_ids.len())
        .map_err(|_| Error::Execution("DSpark block table exceeds u32".into()))?;
    Ok(ExecutionBatch::new(
        ForwardMode::Prefill,
        tokens.to_vec(),
        (position..sequence_len).collect(),
        bindings.write_slots.iter().copied().map(Some).collect(),
        vec![top_k.map_or(LogitsRequest::None, LogitsRequest::TopK); tokens.len()],
        vec![ferrule_common::execution::ExecutionSequence::new(
            StateSlot::new(0),
            ForwardPhase::Prefill,
            0..width,
            position,
            sequence_len,
            0..block_count,
        )],
        bindings.block_ids.clone(),
    )
    .with_intent(intent))
}

fn promote_branch<R: MultiSessionRunner>(
    executor: &mut NativeMultiSessionExecutor<R>,
    source: &mut R::SequenceState,
    branch: R::SequenceState,
) -> Result<()> {
    let previous = std::mem::replace(source, branch);
    executor.release_sequence_state(previous)
}

fn rollback_reservation(
    page_manager: &mut KvPageManager,
    reservation: ferrule_common::execution::KvReservation,
    cause: Error,
) -> Error {
    match page_manager.rollback(reservation) {
        Ok(()) => cause,
        Err(cleanup) => Error::Execution(format!(
            "{cause}; DSpark logical KV rollback also failed: {cleanup}"
        )),
    }
}

fn discard_verification_branch<R: MultiSessionRunner>(
    executor: &mut NativeMultiSessionExecutor<R>,
    page_manager: &mut KvPageManager,
    reservation: ferrule_common::execution::KvReservation,
    branch: R::SequenceState,
    cause: Error,
) -> Error {
    let backend_cleanup = executor.rollback_prepared_batch().err();
    let logical_cleanup = page_manager.rollback(reservation).err();
    let state_cleanup = executor.release_sequence_state(branch).err();
    with_cleanup_errors(cause, backend_cleanup, logical_cleanup, state_cleanup)
}

fn with_cleanup_errors(
    cause: Error,
    backend: Option<Error>,
    logical: Option<Error>,
    state: Option<Error>,
) -> Error {
    if backend.is_none() && logical.is_none() && state.is_none() {
        return cause;
    }
    Error::Execution(format!(
        "{cause}; DSpark cleanup failed: backend={backend:?} logical={logical:?} state={state:?}"
    ))
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CausalVerification {
    accepted: usize,
    target_next: TokenLogit,
}

/// Predictions are causally shifted: the committed frontier verifies proposal
/// row zero, while packed output row `i` verifies proposal row `i + 1`.
fn verify_causal_prefix(
    output: &ExecutionOutput,
    proposal: &[u32],
    frontier: TokenLogit,
) -> Result<CausalVerification> {
    let mut row_top1 = vec![None; proposal.len()];
    for row in &output.logits {
        let row_index = usize::try_from(row.input_row)
            .map_err(|_| Error::Execution("DSpark output row exceeds usize".into()))?;
        let slot = row_top1.get_mut(row_index).ok_or_else(|| {
            Error::Execution(format!(
                "DSpark output row {row_index} exceeds verification width {}",
                proposal.len()
            ))
        })?;
        if slot.is_some() {
            return Err(Error::Execution(format!(
                "DSpark output contains duplicate row {row_index}"
            )));
        }
        let LogitsOutput::TopK(logits) = &row.logits else {
            return Err(Error::Execution(format!(
                "DSpark output row {row_index} is not top-k"
            )));
        };
        *slot = Some(*logits.first().ok_or_else(|| {
            Error::Execution(format!("DSpark output row {row_index} has empty top-k"))
        })?);
    }
    let row_top1 = row_top1
        .into_iter()
        .enumerate()
        .map(|(row, top1)| {
            top1.ok_or_else(|| Error::Execution(format!("DSpark output is missing row {row}")))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut accepted = 0usize;
    let mut prediction = frontier;
    while accepted < proposal.len() && prediction.token_id == proposal[accepted] {
        prediction = row_top1[accepted];
        accepted += 1;
    }
    Ok(CausalVerification {
        accepted,
        target_next: prediction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_common::execution::LogitsRow;

    #[test]
    fn accepted_prefix_all_accepted() {
        let proposal = vec![10, 11, 12, 13];
        let output = ExecutionOutput::new(vec![
            LogitsRow::new(0, LogitsOutput::TopK(vec![logit(11, 1.0)])),
            LogitsRow::new(1, LogitsOutput::TopK(vec![logit(12, 1.0)])),
            LogitsRow::new(2, LogitsOutput::TopK(vec![logit(13, 1.0)])),
            LogitsRow::new(3, LogitsOutput::TopK(vec![logit(42, 1.0)])),
        ]);
        let verification = verify_causal_prefix(&output, &proposal, logit(10, 1.0)).unwrap();
        assert_eq!(verification.accepted, 4);
        assert_eq!(verification.target_next.token_id, 42);
    }

    #[test]
    fn accepted_prefix_partial_rejection() {
        let proposal = vec![10, 11, 12, 13];
        let output = ExecutionOutput::new(vec![
            LogitsRow::new(0, LogitsOutput::TopK(vec![logit(11, 1.0)])),
            LogitsRow::new(1, LogitsOutput::TopK(vec![logit(99, 1.0)])),
            LogitsRow::new(2, LogitsOutput::TopK(vec![logit(13, 1.0)])),
            LogitsRow::new(3, LogitsOutput::TopK(vec![logit(42, 1.0)])),
        ]);
        let verification = verify_causal_prefix(&output, &proposal, logit(10, 1.0)).unwrap();
        assert_eq!(verification.accepted, 2);
        assert_eq!(verification.target_next.token_id, 99);
    }

    #[test]
    fn accepted_prefix_first_rejected() {
        let proposal = vec![10, 11, 12];
        let output = ExecutionOutput::new(vec![
            LogitsRow::new(0, LogitsOutput::TopK(vec![logit(11, 1.0)])),
            LogitsRow::new(1, LogitsOutput::TopK(vec![logit(12, 1.0)])),
            LogitsRow::new(2, LogitsOutput::TopK(vec![logit(13, 1.0)])),
        ]);
        let verification = verify_causal_prefix(&output, &proposal, logit(99, 1.0)).unwrap();
        assert_eq!(verification.accepted, 0);
        assert_eq!(verification.target_next.token_id, 99);
    }

    #[test]
    fn verification_width_one_uses_frontier_and_preserves_next_prediction() {
        let proposal = vec![10];
        let output = ExecutionOutput::new(vec![LogitsRow::new(
            0,
            LogitsOutput::TopK(vec![logit(77, 1.0)]),
        )]);
        let verification = verify_causal_prefix(&output, &proposal, logit(10, 1.0)).unwrap();
        assert_eq!(verification.accepted, 1);
        assert_eq!(verification.target_next.token_id, 77);
    }

    #[test]
    fn verification_rejects_missing_output_rows() {
        let output = ExecutionOutput::new(vec![LogitsRow::new(
            0,
            LogitsOutput::TopK(vec![logit(11, 1.0)]),
        )]);
        assert!(verify_causal_prefix(&output, &[10, 11], logit(10, 1.0)).is_err());
    }

    #[test]
    fn dspark_metrics_record_and_rates() {
        let mut metrics = DSparkMetrics::default();
        metrics.record(&DSparkCycleResult {
            accepted: vec![10, 11],
            rejected: Some(12),
            target_correction: Some(99),
            target_next: Some(logit(99, 1.0)),
            proposed: 3,
            cycle_time_us: 100_000,
            verify_time_us: 80_000,
        });
        metrics.record(&DSparkCycleResult {
            accepted: vec![20, 21, 22, 23],
            rejected: None,
            target_correction: None,
            target_next: Some(logit(24, 1.0)),
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
            target_next: Some(logit(14, 1.0)),
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
