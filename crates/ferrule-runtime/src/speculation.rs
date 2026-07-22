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
//! Proposal execution is a checkpoint-native model capability. Verification
//! uses the existing `MultiSessionRunner` packed batch path.

use std::num::NonZeroU32;
use std::ops::Range;
use std::time::Instant;

use ferrule_common::execution::{
    ExecutionBatch, ExecutionIntent, ExecutionOutput, ExecutionSequence, ForwardMode, ForwardPhase,
    LogitsOutput, LogitsRequest, StateSlot, TokenLogit,
};
use ferrule_common::{Error, Result};
use ferrule_model::MultiSessionRunner;

use crate::cache::{KvPageManager, KvReservationBindings, KvReservationCommit};
use crate::engine::NativeMultiSessionExecutor;

// ── Transaction types ─────────────────────────────────────────────────

/// Number of target rows submitted by one packed verification transaction.
///
/// This is intentionally not called the draft width: DSpark checkpoints may
/// distinguish an anchor token, draft slots, target verification rows, and the
/// trailing target bonus. The production proposal source must define that
/// mapping explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerificationWidth(pub usize);

/// Counters for one speculative transaction.
///
/// These values are deliberately separate. Accepted draft tokens, target
/// correction/bonus tokens, and externally committed output tokens are not
/// interchangeable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeCycleAccounting {
    pub proposed_tokens: usize,
    pub verified_rows: usize,
    pub accepted_draft_tokens: usize,
    pub correction_tokens: usize,
    pub externally_committed_tokens: usize,
    pub rolled_back_rows: usize,
}

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
    /// Target top-1 for every packed verification row, in input-row order.
    pub target_row_top1: Vec<TokenLogit>,
    pub accounting: SpeculativeCycleAccounting,
    /// Wall-clock time for target verification plus this function's
    /// commit/rollback work. Proposal generation happens before this function
    /// and is therefore not included.
    pub transaction_time_us: u64,
    /// Wall-clock time for the verification forward pass only.
    pub verify_time_us: u64,
}

impl DSparkCycleResult {
    /// Draft acceptance rate. This is explanatory telemetry, not serving
    /// throughput.
    pub fn acceptance_rate(&self) -> f32 {
        if self.accounting.proposed_tokens == 0 {
            0.0
        } else {
            self.accounting.accepted_draft_tokens as f32 / self.accounting.proposed_tokens as f32
        }
    }
}

/// One sequence in a packed DSpark verification cohort.
///
/// `state_slot` and `generation` identify the sequence in the logical page
/// manager. Model state is supplied separately, in the same sequence-major
/// order, so this request remains model-neutral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DSparkVerificationItem<'a> {
    pub state_slot: StateSlot,
    pub generation: u64,
    pub proposal: &'a [u32],
    pub frontier: TargetFrontier,
}

/// Results from one true multi-sequence DSpark verification transaction.
///
/// The entries in `results` correspond one-for-one with the input items. Each
/// cycle result carries the same shared timings for compatibility with scalar
/// DSpark metrics; the top-level fields make their cohort-wide nature explicit.
#[derive(Debug, Clone, PartialEq)]
pub struct DSparkCohortResult {
    pub results: Vec<DSparkCycleResult>,
    pub transaction_time_us: u64,
    pub verify_time_us: u64,
}

/// Accumulated DSpark telemetry across multiple cycles.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DSparkMetrics {
    pub cycles: usize,
    pub proposed_tokens: usize,
    pub verified_rows: usize,
    pub accepted_draft_tokens: usize,
    pub correction_tokens: usize,
    pub externally_committed_tokens: usize,
    /// Runtime token callback invocations; serving delivery is tracked separately.
    pub runtime_emitted_tokens: usize,
    pub rolled_back_rows: usize,
    pub rejected_tokens: usize,
    /// Indexed by accepted draft-prefix length.
    pub accepted_prefix_histogram: Vec<usize>,
    pub total_proposal_time_us: u64,
    pub total_transaction_time_us: u64,
    pub total_verify_time_us: u64,
    pub total_cycle_time_us: u64,
}

impl DSparkMetrics {
    pub fn acceptance_rate(&self) -> f32 {
        if self.proposed_tokens == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f32 / self.proposed_tokens as f32
        }
    }

    pub fn mean_transaction_time_us(&self) -> u64 {
        if self.cycles == 0 {
            0
        } else {
            self.total_transaction_time_us / self.cycles as u64
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
        let accounting = result.accounting;
        self.cycles += 1;
        self.proposed_tokens += accounting.proposed_tokens;
        self.verified_rows += accounting.verified_rows;
        self.accepted_draft_tokens += accounting.accepted_draft_tokens;
        self.correction_tokens += accounting.correction_tokens;
        self.externally_committed_tokens += accounting.externally_committed_tokens;
        self.rolled_back_rows += accounting.rolled_back_rows;
        self.rejected_tokens += result.rejected.is_some() as usize;
        if self.accepted_prefix_histogram.len() <= accounting.accepted_draft_tokens {
            self.accepted_prefix_histogram
                .resize(accounting.accepted_draft_tokens + 1, 0);
        }
        self.accepted_prefix_histogram[accounting.accepted_draft_tokens] += 1;
        self.total_transaction_time_us += result.transaction_time_us;
        self.total_verify_time_us += result.verify_time_us;
        self.total_cycle_time_us += result.transaction_time_us;
    }

    pub fn record_complete_cycle(
        &mut self,
        result: &DSparkCycleResult,
        proposal_time_us: u64,
        complete_cycle_time_us: u64,
    ) {
        self.record(result);
        self.total_proposal_time_us += proposal_time_us;
        self.total_cycle_time_us = self
            .total_cycle_time_us
            .saturating_sub(result.transaction_time_us)
            .saturating_add(complete_cycle_time_us);
    }

    pub fn record_runtime_emitted_tokens(&mut self, token_count: usize) {
        self.runtime_emitted_tokens += token_count;
    }
}

// ── DSpark transaction ────────────────────────────────────────────────

/// Runs one true multi-sequence DSpark verification cohort.
///
/// The cohort is sequence-major and may be ragged: each item contributes one
/// anchor row followed by its local proposal. The target executes exactly one
/// provisional packed batch. Every accepted prefix must then be retained by the
/// backend as one atomic cohort operation; this path deliberately does not
/// replay prefixes when retention is unavailable.
pub fn run_dspark_verification_cohort<R>(
    executor: &mut NativeMultiSessionExecutor<R>,
    page_manager: &mut KvPageManager,
    source_states: &mut [R::SequenceState],
    items: &[DSparkVerificationItem<'_>],
    top_k: NonZeroU32,
) -> Result<DSparkCohortResult>
where
    R: MultiSessionRunner,
{
    let transaction_start = Instant::now();
    if items.is_empty() {
        return Err(Error::Execution(
            "DSpark verification cohort must contain at least one sequence".into(),
        ));
    }
    if source_states.len() != items.len() {
        return Err(Error::Execution(format!(
            "DSpark verification cohort state/item mismatch: states={} items={}",
            source_states.len(),
            items.len()
        )));
    }

    let executed_rows = items
        .iter()
        .map(|item| {
            item.proposal
                .len()
                .checked_add(1)
                .ok_or_else(|| Error::Execution("DSpark verification row count overflow".into()))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut reservations = Vec::with_capacity(items.len());
    for (sequence, (item, &rows)) in items.iter().zip(&executed_rows).enumerate() {
        let reservation = match page_manager.reserve(item.state_slot, item.generation, rows) {
            Ok(reservation) => reservation,
            Err(error) => {
                return Err(discard_dspark_cohort(
                    executor,
                    page_manager,
                    reservations,
                    Vec::new(),
                    false,
                    error,
                ));
            }
        };
        reservations.push(reservation);
        if reservations[sequence].positions.start != item.frontier.position {
            let actual = reservations[sequence].positions.start;
            let error = Error::Execution(format!(
                "DSpark cohort sequence {sequence} frontier position {} does not match committed KV position {actual}",
                item.frontier.position
            ));
            return Err(discard_dspark_cohort(
                executor,
                page_manager,
                reservations,
                Vec::new(),
                false,
                error,
            ));
        }
    }

    let mut bindings = Vec::with_capacity(reservations.len());
    for reservation in &reservations {
        match page_manager.reservation_bindings(reservation) {
            Ok(sequence_bindings) => bindings.push(sequence_bindings),
            Err(error) => {
                return Err(discard_dspark_cohort(
                    executor,
                    page_manager,
                    reservations,
                    Vec::new(),
                    false,
                    error,
                ));
            }
        }
    }
    let (verification_batch, query_ranges) =
        match build_dspark_cohort_batch(items, &bindings, top_k) {
            Ok(batch) => batch,
            Err(error) => {
                return Err(discard_dspark_cohort(
                    executor,
                    page_manager,
                    reservations,
                    Vec::new(),
                    false,
                    error,
                ));
            }
        };

    let mut verification_branches = Vec::with_capacity(items.len());
    for (source, item) in source_states.iter().zip(items) {
        match executor.fork_sequence_state_from(source, item.frontier.position) {
            Ok(branch) => verification_branches.push(branch),
            Err(error) => {
                return Err(discard_dspark_cohort(
                    executor,
                    page_manager,
                    reservations,
                    verification_branches,
                    false,
                    error,
                ));
            }
        }
    }

    let verify_start = Instant::now();
    let output = match executor.execute_batch_with_kv(
        &mut verification_branches,
        &verification_batch,
        &reservations,
    ) {
        Ok(output) => output,
        Err(error) => {
            return Err(discard_dspark_cohort(
                executor,
                page_manager,
                reservations,
                verification_branches,
                true,
                error,
            ));
        }
    };
    let verify_time_us = verify_start.elapsed().as_micros() as u64;

    let global_row_top1 = match collect_global_row_top1(&output, verification_batch.len()) {
        Ok(rows) => rows,
        Err(error) => {
            return Err(discard_dspark_cohort(
                executor,
                page_manager,
                reservations,
                verification_branches,
                true,
                error,
            ));
        }
    };
    let mut verifications = Vec::with_capacity(items.len());
    for (sequence, (item, query)) in items.iter().zip(&query_ranges).enumerate() {
        let local_rows = match global_row_top1.get(query.clone()) {
            Some(rows) => rows,
            None => {
                let error = Error::Execution(format!(
                    "DSpark cohort sequence {sequence} query range {query:?} exceeds {} global rows",
                    global_row_top1.len()
                ));
                return Err(discard_dspark_cohort(
                    executor,
                    page_manager,
                    reservations,
                    verification_branches,
                    true,
                    error,
                ));
            }
        };
        match verify_causal_slice(local_rows, item.proposal) {
            Ok(verification) => verifications.push(verification),
            Err(error) => {
                return Err(discard_dspark_cohort(
                    executor,
                    page_manager,
                    reservations,
                    verification_branches,
                    true,
                    error,
                ));
            }
        }
    }

    let retained_rows = verifications
        .iter()
        .map(|verification| verification.accepted + 1)
        .collect::<Vec<_>>();
    let retained = match executor.retain_prepared_prefixes(
        source_states,
        &mut verification_branches,
        &executed_rows,
        &retained_rows,
    ) {
        Ok(retained) => retained,
        Err(error) => {
            return Err(discard_dspark_cohort(
                executor,
                page_manager,
                reservations,
                verification_branches,
                true,
                error,
            ));
        }
    };
    if !retained {
        return Err(discard_dspark_cohort(
            executor,
            page_manager,
            reservations,
            verification_branches,
            true,
            Error::Execution(
                "DSpark cohort backend cannot atomically retain all accepted prefixes; replay is disabled"
                    .into(),
            ),
        ));
    }

    let commits = reservations
        .into_iter()
        .zip(&retained_rows)
        .map(|(reservation, &rows)| KvReservationCommit::new(reservation, rows))
        .collect();
    let freed = match page_manager.commit_prefix_batch_with_freed(commits) {
        Ok(freed) => freed,
        Err(error) => {
            return Err(discard_dspark_cohort(
                executor,
                page_manager,
                Vec::new(),
                verification_branches,
                true,
                error,
            ));
        }
    };

    let backend_commit_error = executor.commit_prepared_batch().err();
    let state_promotion_error =
        promote_cohort_branches(executor, source_states, verification_branches).err();
    let page_release_error = executor.release_kv_pages(&freed).err();
    finish_dspark_cohort_publication(
        backend_commit_error,
        state_promotion_error,
        page_release_error,
    )?;

    let transaction_time_us = transaction_start.elapsed().as_micros() as u64;
    let results = items
        .iter()
        .zip(verifications)
        .zip(executed_rows)
        .map(|((item, verification), verified_rows)| {
            let accepted_rows = verification.accepted;
            let committed_rows = accepted_rows + 1;
            let rejected = item.proposal.get(accepted_rows).copied();
            let target_next = verification.target_next;
            DSparkCycleResult {
                accepted: item.proposal[..accepted_rows].to_vec(),
                rejected,
                target_correction: rejected.map(|_| target_next.token_id),
                target_next: Some(target_next),
                target_row_top1: verification.target_row_top1,
                accounting: SpeculativeCycleAccounting {
                    proposed_tokens: item.proposal.len(),
                    verified_rows,
                    accepted_draft_tokens: accepted_rows,
                    correction_tokens: usize::from(accepted_rows < item.proposal.len()),
                    externally_committed_tokens: committed_rows,
                    rolled_back_rows: verified_rows - committed_rows,
                },
                transaction_time_us,
                verify_time_us,
            }
        })
        .collect();
    Ok(DSparkCohortResult {
        results,
        transaction_time_us,
        verify_time_us,
    })
}

/// Run a one-row exact probe before the full packed verification.
///
/// A first-draft miss commits the already-computed anchor branch directly, so
/// the rejected suffix is never executed or replayed. A first-draft hit rolls
/// the probe back completely and delegates to the original atomic packed
/// transaction. This preserves exactness while making target work proportional
/// to observed acceptance.
pub fn run_acceptance_aware_dspark_verification<R>(
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
    if proposal.is_empty() {
        return run_dspark_verification(
            executor,
            page_manager,
            source_state,
            state_slot,
            generation,
            proposal,
            top_k,
            frontier,
        );
    }

    let transaction_start = Instant::now();
    let reservation = page_manager.reserve(state_slot, generation, 1)?;
    if reservation.positions.start != frontier.position {
        let actual = reservation.positions.start;
        let error = Error::Execution(format!(
            "DSpark probe frontier position {} does not match committed KV position {actual}",
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
    let probe_batch = match build_dspark_batch(
        std::slice::from_ref(&frontier.top1.token_id),
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
    let mut probe_branch = match executor.fork_sequence_state_from(source_state, frontier.position)
    {
        Ok(branch) => branch,
        Err(error) => {
            return Err(rollback_reservation(page_manager, reservation, error));
        }
    };

    let verify_start = Instant::now();
    let output = match executor.execute_batch_with_kv(
        std::slice::from_mut(&mut probe_branch),
        &probe_batch,
        std::slice::from_ref(&reservation),
    ) {
        Ok(output) => output,
        Err(error) => {
            return Err(discard_verification_branch(
                executor,
                page_manager,
                reservation,
                probe_branch,
                error,
            ));
        }
    };
    let probe_verify_time_us = verify_start.elapsed().as_micros() as u64;
    let probe = match verify_causal_prefix(&output, &[]) {
        Ok(verification) => verification,
        Err(error) => {
            return Err(discard_verification_branch(
                executor,
                page_manager,
                reservation,
                probe_branch,
                error,
            ));
        }
    };

    if probe.target_next.token_id != proposal[0] {
        let freed = match page_manager.commit_prefix_with_freed(reservation, 1) {
            Ok(freed) => freed,
            Err(error) => {
                let backend_cleanup = executor.rollback_prepared_batch().err();
                let state_cleanup = executor.release_sequence_state(probe_branch).err();
                return Err(with_cleanup_errors(
                    error,
                    backend_cleanup,
                    None,
                    state_cleanup,
                ));
            }
        };
        if let Err(error) = executor.commit_prepared_batch() {
            promote_branch(executor, source_state, probe_branch)?;
            return Err(error);
        }
        promote_branch(executor, source_state, probe_branch)?;
        executor.release_kv_pages(&freed)?;
        return Ok(DSparkCycleResult {
            accepted: Vec::new(),
            rejected: Some(proposal[0]),
            target_correction: Some(probe.target_next.token_id),
            target_next: Some(probe.target_next),
            target_row_top1: probe.target_row_top1,
            accounting: SpeculativeCycleAccounting {
                proposed_tokens: proposal.len(),
                verified_rows: 1,
                accepted_draft_tokens: 0,
                correction_tokens: 1,
                externally_committed_tokens: 1,
                rolled_back_rows: 0,
            },
            transaction_time_us: transaction_start.elapsed().as_micros() as u64,
            verify_time_us: probe_verify_time_us,
        });
    }

    if let Err(error) = executor.rollback_prepared_batch() {
        return Err(discard_verification_branch(
            executor,
            page_manager,
            reservation,
            probe_branch,
            error,
        ));
    }
    if let Err(error) = executor.release_sequence_state(probe_branch) {
        return Err(rollback_reservation(page_manager, reservation, error));
    }
    page_manager.rollback(reservation)?;

    let mut result = run_dspark_verification(
        executor,
        page_manager,
        source_state,
        state_slot,
        generation,
        proposal,
        top_k,
        frontier,
    )?;
    result.accounting.verified_rows = result.accounting.verified_rows.saturating_add(1);
    result.accounting.rolled_back_rows = result.accounting.rolled_back_rows.saturating_add(1);
    result.verify_time_us = result.verify_time_us.saturating_add(probe_verify_time_us);
    result.transaction_time_us = transaction_start.elapsed().as_micros() as u64;
    Ok(result)
}

/// Runs one DSpark verification cycle.
///
/// Given a draft proposal, this function:
///
/// 1. Creates a packed provisional verification batch
/// 2. Executes the target model forward pass
/// 3. Compares target top-1 with draft proposals
/// 4. Promotes a full-accept branch, or rolls it back and replays an accepted prefix
/// 5. Returns the accepted draft prefix and an uncommitted target correction/bonus
///
/// Proposal generation is outside this function's timer. The caller remains
/// responsible for committing the returned target correction/bonus according
/// to the production DSpark protocol and for reconciling externally emitted
/// tokens with [`SpeculativeCycleAccounting`].
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
    let transaction_start = Instant::now();
    let width = proposal.len();
    let verified_rows = width
        .checked_add(1)
        .ok_or_else(|| Error::Execution("DSpark verification row count overflow".into()))?;
    let mut verification_tokens = Vec::with_capacity(verified_rows);
    verification_tokens.push(frontier.top1.token_id);
    verification_tokens.extend_from_slice(proposal);

    let reservation = page_manager.reserve(state_slot, generation, verified_rows)?;
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
        &verification_tokens,
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
    let verification = match verify_causal_prefix(&output, proposal) {
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
    let committed_rows = accepted_rows + 1;
    let mut prefix_promoted = false;
    if accepted_rows == width {
        let freed = match page_manager.commit_prefix_with_freed(reservation, verified_rows) {
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
        let retained = match executor.retain_prepared_prefix(
            source_state,
            &mut verification_branch,
            verified_rows,
            committed_rows,
        ) {
            Ok(retained) => retained,
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
        if retained {
            prefix_promoted = true;
            let freed = match page_manager.commit_prefix_with_freed(reservation, committed_rows) {
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
            let prefix_view =
                match page_manager.reservation_prefix_view(&reservation, committed_rows) {
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
                &verification_tokens[..committed_rows],
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
            let freed = match page_manager.commit_prefix_with_freed(reservation, committed_rows) {
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
    let target_next = verification.target_next;
    Ok(DSparkCycleResult {
        accepted,
        rejected,
        target_correction: rejected.map(|_| target_next.token_id),
        target_next: Some(target_next),
        target_row_top1: verification.target_row_top1,
        accounting: SpeculativeCycleAccounting {
            proposed_tokens: width,
            verified_rows,
            accepted_draft_tokens: accepted_rows,
            // The correction/bonus is the next exact frontier and is not part
            // of the committed input rows until the following cycle.
            correction_tokens: usize::from(accepted_rows < width),
            externally_committed_tokens: committed_rows,
            rolled_back_rows: if accepted_rows == width {
                0
            } else if prefix_promoted {
                verified_rows - committed_rows
            } else {
                verified_rows
            },
        },
        transaction_time_us: transaction_start.elapsed().as_micros() as u64,
        verify_time_us,
    })
}

fn build_dspark_cohort_batch(
    items: &[DSparkVerificationItem<'_>],
    bindings: &[KvReservationBindings],
    top_k: NonZeroU32,
) -> Result<(ExecutionBatch, Vec<Range<usize>>)> {
    if items.is_empty() || items.len() != bindings.len() {
        return Err(Error::Execution(format!(
            "DSpark cohort batch shape mismatch: items={} bindings={}",
            items.len(),
            bindings.len()
        )));
    }
    let total_rows = items.iter().try_fold(0usize, |total, item| {
        total
            .checked_add(item.proposal.len())
            .and_then(|rows| rows.checked_add(1))
            .ok_or_else(|| Error::Execution("DSpark cohort packed row count overflow".into()))
    })?;

    let mut token_ids = Vec::with_capacity(total_rows);
    let mut positions = Vec::with_capacity(total_rows);
    let mut kv_write_slots = Vec::with_capacity(total_rows);
    let mut logits = Vec::with_capacity(total_rows);
    let mut sequences = Vec::with_capacity(items.len());
    let mut kv_block_ids = Vec::new();
    let mut query_ranges = Vec::with_capacity(items.len());

    for (sequence, (item, bindings)) in items.iter().zip(bindings).enumerate() {
        let local_rows = item
            .proposal
            .len()
            .checked_add(1)
            .ok_or_else(|| Error::Execution("DSpark verification row count overflow".into()))?;
        if bindings.write_slots.len() != local_rows {
            return Err(Error::Execution(format!(
                "DSpark cohort sequence {sequence} has {} KV write slots for {local_rows} rows",
                bindings.write_slots.len()
            )));
        }

        let query_start = token_ids.len();
        let query_end = query_start
            .checked_add(local_rows)
            .ok_or_else(|| Error::Execution("DSpark cohort packed query range overflow".into()))?;
        let query_start_u32 = u32::try_from(query_start)
            .map_err(|_| Error::Execution("DSpark cohort query start exceeds u32".into()))?;
        let query_end_u32 = u32::try_from(query_end)
            .map_err(|_| Error::Execution("DSpark cohort query end exceeds u32".into()))?;
        let context_len = u32::try_from(item.frontier.position).map_err(|_| {
            Error::Execution(format!(
                "DSpark cohort sequence {sequence} frontier position exceeds u32"
            ))
        })?;
        let local_rows_u32 = u32::try_from(local_rows).map_err(|_| {
            Error::Execution(format!(
                "DSpark cohort sequence {sequence} verification width exceeds u32"
            ))
        })?;
        let sequence_len = context_len.checked_add(local_rows_u32).ok_or_else(|| {
            Error::Execution(format!(
                "DSpark cohort sequence {sequence} sequence length overflow"
            ))
        })?;
        let block_start = u32::try_from(kv_block_ids.len())
            .map_err(|_| Error::Execution("DSpark cohort block table exceeds u32".into()))?;
        kv_block_ids.extend_from_slice(&bindings.block_ids);
        let block_end = u32::try_from(kv_block_ids.len())
            .map_err(|_| Error::Execution("DSpark cohort block table exceeds u32".into()))?;
        let dense_state_slot = u32::try_from(sequence)
            .map(StateSlot::new)
            .map_err(|_| Error::Execution("DSpark cohort sequence count exceeds u32".into()))?;

        token_ids.push(item.frontier.top1.token_id);
        token_ids.extend_from_slice(item.proposal);
        positions.extend(context_len..sequence_len);
        kv_write_slots.extend(bindings.write_slots.iter().copied().map(Some));
        logits.extend(std::iter::repeat_n(LogitsRequest::TopK(top_k), local_rows));
        sequences.push(ExecutionSequence::new(
            dense_state_slot,
            ForwardPhase::Prefill,
            query_start_u32..query_end_u32,
            context_len,
            sequence_len,
            block_start..block_end,
        ));
        query_ranges.push(query_start..query_end);
    }

    Ok((
        ExecutionBatch::new(
            ForwardMode::Prefill,
            token_ids,
            positions,
            kv_write_slots,
            logits,
            sequences,
            kv_block_ids,
        )
        .with_intent(ExecutionIntent::ProvisionalVerification),
        query_ranges,
    ))
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

fn discard_dspark_cohort<R: MultiSessionRunner>(
    executor: &mut NativeMultiSessionExecutor<R>,
    page_manager: &mut KvPageManager,
    reservations: Vec<ferrule_common::execution::KvReservation>,
    branches: Vec<R::SequenceState>,
    rollback_backend: bool,
    cause: Error,
) -> Error {
    let mut cleanup_errors = Vec::new();
    if rollback_backend && let Err(error) = executor.rollback_prepared_batch() {
        cleanup_errors.push(format!("backend rollback: {error}"));
    }
    for (sequence, reservation) in reservations.into_iter().enumerate() {
        if let Err(error) = page_manager.rollback(reservation) {
            cleanup_errors.push(format!("logical rollback for sequence {sequence}: {error}"));
        }
    }
    for (sequence, branch) in branches.into_iter().enumerate() {
        if let Err(error) = executor.release_sequence_state(branch) {
            cleanup_errors.push(format!("branch release for sequence {sequence}: {error}"));
        }
    }
    if cleanup_errors.is_empty() {
        cause
    } else {
        Error::Execution(format!(
            "{cause}; DSpark cohort cleanup also failed: {}",
            cleanup_errors.join("; ")
        ))
    }
}

fn promote_cohort_branches<R: MultiSessionRunner>(
    executor: &mut NativeMultiSessionExecutor<R>,
    sources: &mut [R::SequenceState],
    branches: Vec<R::SequenceState>,
) -> Result<()> {
    let branch_count = branches.len();
    let mut branches = branches.into_iter();
    let mut errors = Vec::new();
    for (sequence, source) in sources.iter_mut().enumerate() {
        let Some(branch) = branches.next() else {
            errors.push(format!("missing branch for sequence {sequence}"));
            continue;
        };
        let previous = std::mem::replace(source, branch);
        if let Err(error) = executor.release_sequence_state(previous) {
            errors.push(format!("source release for sequence {sequence}: {error}"));
        }
    }
    for (offset, branch) in branches.enumerate() {
        if let Err(error) = executor.release_sequence_state(branch) {
            errors.push(format!(
                "extra branch release for sequence {}: {error}",
                sources.len() + offset
            ));
        }
    }
    if branch_count != sources.len() {
        errors.push(format!(
            "state/branch promotion mismatch: states={} branches={branch_count}",
            sources.len()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(Error::Execution(format!(
            "DSpark cohort state promotion failed: {}",
            errors.join("; ")
        )))
    }
}

fn finish_dspark_cohort_publication(
    backend: Option<Error>,
    state: Option<Error>,
    pages: Option<Error>,
) -> Result<()> {
    if backend.is_none() && state.is_none() && pages.is_none() {
        return Ok(());
    }
    Err(Error::Execution(format!(
        "DSpark cohort publication failed: backend={backend:?} state={state:?} pages={pages:?}"
    )))
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

#[derive(Debug, Clone, PartialEq)]
struct CausalVerification {
    accepted: usize,
    target_next: TokenLogit,
    target_row_top1: Vec<TokenLogit>,
}

fn collect_global_row_top1(
    output: &ExecutionOutput,
    expected_rows: usize,
) -> Result<Vec<TokenLogit>> {
    let mut row_top1 = vec![None; expected_rows];
    for row in &output.logits {
        let row_index = usize::try_from(row.input_row)
            .map_err(|_| Error::Execution("DSpark output row exceeds usize".into()))?;
        let slot = row_top1.get_mut(row_index).ok_or_else(|| {
            Error::Execution(format!(
                "DSpark output row {row_index} exceeds verification row count {expected_rows}"
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
    row_top1
        .into_iter()
        .enumerate()
        .map(|(row, top1)| {
            top1.ok_or_else(|| Error::Execution(format!("DSpark output is missing row {row}")))
        })
        .collect()
}

/// The local target input is `[anchor, draft × V]`. Local output row `i`
/// verifies draft `i`, while local row `V` is the exact correction/bonus
/// frontier.
fn verify_causal_slice(row_top1: &[TokenLogit], proposal: &[u32]) -> Result<CausalVerification> {
    let verified_rows = proposal
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::Execution("DSpark output row count overflow".into()))?;
    if row_top1.len() != verified_rows {
        return Err(Error::Execution(format!(
            "DSpark causal slice has {} rows, expected {verified_rows}",
            row_top1.len()
        )));
    }
    let mut accepted = 0usize;
    while accepted < proposal.len() && row_top1[accepted].token_id == proposal[accepted] {
        accepted += 1;
    }
    Ok(CausalVerification {
        accepted,
        target_next: row_top1[accepted],
        target_row_top1: row_top1.to_vec(),
    })
}

fn verify_causal_prefix(output: &ExecutionOutput, proposal: &[u32]) -> Result<CausalVerification> {
    let verified_rows = proposal
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::Execution("DSpark output row count overflow".into()))?;
    let row_top1 = collect_global_row_top1(output, verified_rows)?;
    verify_causal_slice(&row_top1, proposal)
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
            LogitsRow::new(4, LogitsOutput::TopK(vec![logit(42, 1.0)])),
        ]);
        let verification = verify_causal_prefix(&output, &proposal).unwrap();
        assert_eq!(verification.accepted, 4);
        assert_eq!(verification.target_next.token_id, 42);
    }

    #[test]
    fn accepted_prefix_partial_rejection() {
        let proposal = vec![10, 11, 12, 13];
        let output = ExecutionOutput::new(vec![
            LogitsRow::new(0, LogitsOutput::TopK(vec![logit(10, 1.0)])),
            LogitsRow::new(1, LogitsOutput::TopK(vec![logit(11, 1.0)])),
            LogitsRow::new(2, LogitsOutput::TopK(vec![logit(99, 1.0)])),
            LogitsRow::new(3, LogitsOutput::TopK(vec![logit(13, 1.0)])),
            LogitsRow::new(4, LogitsOutput::TopK(vec![logit(42, 1.0)])),
        ]);
        let verification = verify_causal_prefix(&output, &proposal).unwrap();
        assert_eq!(verification.accepted, 2);
        assert_eq!(verification.target_next.token_id, 99);
    }

    #[test]
    fn accepted_prefix_first_rejected() {
        let proposal = vec![10, 11, 12];
        let output = ExecutionOutput::new(vec![
            LogitsRow::new(0, LogitsOutput::TopK(vec![logit(99, 1.0)])),
            LogitsRow::new(1, LogitsOutput::TopK(vec![logit(11, 1.0)])),
            LogitsRow::new(2, LogitsOutput::TopK(vec![logit(12, 1.0)])),
            LogitsRow::new(3, LogitsOutput::TopK(vec![logit(13, 1.0)])),
        ]);
        let verification = verify_causal_prefix(&output, &proposal).unwrap();
        assert_eq!(verification.accepted, 0);
        assert_eq!(verification.target_next.token_id, 99);
    }

    #[test]
    fn verification_width_one_preserves_bonus_prediction() {
        let proposal = vec![10];
        let output = ExecutionOutput::new(vec![
            LogitsRow::new(0, LogitsOutput::TopK(vec![logit(10, 1.0)])),
            LogitsRow::new(1, LogitsOutput::TopK(vec![logit(77, 1.0)])),
        ]);
        let verification = verify_causal_prefix(&output, &proposal).unwrap();
        assert_eq!(verification.accepted, 1);
        assert_eq!(verification.target_next.token_id, 77);
    }

    #[test]
    fn verification_rejects_missing_output_rows() {
        let output = ExecutionOutput::new(vec![LogitsRow::new(
            0,
            LogitsOutput::TopK(vec![logit(11, 1.0)]),
        )]);
        assert!(verify_causal_prefix(&output, &[10, 11]).is_err());
    }

    #[test]
    fn ragged_global_spans_verify_distinct_local_acceptance() {
        let output = ExecutionOutput::new(vec![
            LogitsRow::new(6, LogitsOutput::TopK(vec![logit(42, 1.0)])),
            LogitsRow::new(1, LogitsOutput::TopK(vec![logit(99, 1.0)])),
            LogitsRow::new(4, LogitsOutput::TopK(vec![logit(21, 1.0)])),
            LogitsRow::new(0, LogitsOutput::TopK(vec![logit(10, 1.0)])),
            LogitsRow::new(5, LogitsOutput::TopK(vec![logit(22, 1.0)])),
            LogitsRow::new(2, LogitsOutput::TopK(vec![logit(77, 1.0)])),
            LogitsRow::new(3, LogitsOutput::TopK(vec![logit(20, 1.0)])),
        ]);
        let global = collect_global_row_top1(&output, 7).unwrap();
        let first = verify_causal_slice(&global[0..3], &[10, 11]).unwrap();
        let second = verify_causal_slice(&global[3..7], &[20, 21, 22]).unwrap();

        assert_eq!(first.accepted, 1);
        assert_eq!(first.target_next.token_id, 99);
        assert_eq!(first.target_row_top1.len(), 3);
        assert_eq!(second.accepted, 3);
        assert_eq!(second.target_next.token_id, 42);
        assert_eq!(second.target_row_top1.len(), 4);
    }

    #[test]
    fn global_row_collection_rejects_duplicate_missing_and_out_of_range_rows() {
        let duplicate = ExecutionOutput::new(vec![
            LogitsRow::new(0, LogitsOutput::TopK(vec![logit(10, 1.0)])),
            LogitsRow::new(0, LogitsOutput::TopK(vec![logit(11, 1.0)])),
        ]);
        let missing = ExecutionOutput::new(vec![LogitsRow::new(
            0,
            LogitsOutput::TopK(vec![logit(10, 1.0)]),
        )]);
        let out_of_range = ExecutionOutput::new(vec![
            LogitsRow::new(0, LogitsOutput::TopK(vec![logit(10, 1.0)])),
            LogitsRow::new(2, LogitsOutput::TopK(vec![logit(11, 1.0)])),
        ]);

        assert!(
            collect_global_row_top1(&duplicate, 2)
                .unwrap_err()
                .to_string()
                .contains("duplicate row 0")
        );
        assert!(
            collect_global_row_top1(&missing, 2)
                .unwrap_err()
                .to_string()
                .contains("missing row 1")
        );
        assert!(
            collect_global_row_top1(&out_of_range, 2)
                .unwrap_err()
                .to_string()
                .contains("row 2 exceeds")
        );
    }

    #[test]
    fn dspark_metrics_record_explicit_accounting() {
        let mut metrics = DSparkMetrics::default();
        metrics.record(&DSparkCycleResult {
            accepted: vec![10, 11],
            rejected: Some(12),
            target_correction: Some(99),
            target_next: Some(logit(99, 1.0)),
            target_row_top1: vec![logit(10, 1.0), logit(11, 1.0), logit(99, 1.0)],
            accounting: SpeculativeCycleAccounting {
                proposed_tokens: 3,
                verified_rows: 3,
                accepted_draft_tokens: 2,
                correction_tokens: 0,
                externally_committed_tokens: 2,
                rolled_back_rows: 3,
            },
            transaction_time_us: 100_000,
            verify_time_us: 80_000,
        });
        metrics.record(&DSparkCycleResult {
            accepted: vec![20, 21, 22, 23],
            rejected: None,
            target_correction: None,
            target_next: Some(logit(24, 1.0)),
            target_row_top1: vec![
                logit(20, 1.0),
                logit(21, 1.0),
                logit(22, 1.0),
                logit(23, 1.0),
                logit(24, 1.0),
            ],
            accounting: SpeculativeCycleAccounting {
                proposed_tokens: 4,
                verified_rows: 4,
                accepted_draft_tokens: 4,
                correction_tokens: 0,
                externally_committed_tokens: 4,
                rolled_back_rows: 0,
            },
            transaction_time_us: 120_000,
            verify_time_us: 90_000,
        });
        assert_eq!(metrics.cycles, 2);
        assert_eq!(metrics.proposed_tokens, 7);
        assert_eq!(metrics.verified_rows, 7);
        assert_eq!(metrics.accepted_draft_tokens, 6);
        assert_eq!(metrics.correction_tokens, 0);
        assert_eq!(metrics.externally_committed_tokens, 6);
        assert_eq!(metrics.rolled_back_rows, 3);
        assert_eq!(metrics.rejected_tokens, 1);
        assert_eq!(metrics.total_transaction_time_us, 220_000);
        assert_eq!(metrics.total_verify_time_us, 170_000);
        assert!((metrics.acceptance_rate() - 6.0 / 7.0).abs() < 1e-6);
        assert_eq!(metrics.mean_transaction_time_us(), 110_000);
    }

    #[test]
    fn cycle_result_acceptance_uses_draft_counters_only() {
        let result = DSparkCycleResult {
            accepted: vec![10, 11, 12, 13],
            rejected: None,
            target_correction: None,
            target_next: Some(logit(14, 1.0)),
            target_row_top1: vec![
                logit(10, 1.0),
                logit(11, 1.0),
                logit(12, 1.0),
                logit(13, 1.0),
                logit(14, 1.0),
            ],
            accounting: SpeculativeCycleAccounting {
                proposed_tokens: 4,
                verified_rows: 4,
                accepted_draft_tokens: 4,
                correction_tokens: 1,
                externally_committed_tokens: 5,
                rolled_back_rows: 0,
            },
            transaction_time_us: 250_000,
            verify_time_us: 200_000,
        };
        assert!((result.acceptance_rate() - 1.0).abs() < 1e-6);
        assert_eq!(result.accounting.externally_committed_tokens, 5);
    }

    fn logit(token_id: u32, logit: f32) -> ferrule_common::execution::TokenLogit {
        ferrule_common::execution::TokenLogit { token_id, logit }
    }
}
