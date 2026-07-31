//! Speculative verification primitives.
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
    ExecutionBatch, ExecutionIntent, ExecutionOutput, ExecutionSequence, ExecutionTransactionId,
    ForwardMode, ForwardPhase, LogitsOutput, LogitsRequest, StateSlot, TokenLogit,
};
use ferrule_common::{Error, Result};
use ferrule_model::{MultiSessionRunner, PendingModelProgress};

use crate::cache::{
    KvPageManager, KvReservation, KvReservationBindings, KvReservationCommit, KvRetirement,
    PreparedKvCommit,
};
use crate::engine::{NativeBatchExecutionProgress, NativeMultiSessionExecutor};

// ── Transaction types ─────────────────────────────────────────────────

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

/// Result of one speculative verification cycle.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeCycleResult {
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
    pub accounting: SpeculativeCycleAccounting,
    /// Wall-clock time for target verification plus this function's
    /// commit/rollback work. Proposal generation happens before this function
    /// and is therefore not included.
    pub transaction_time_us: u64,
    /// Wall-clock time for the verification forward pass only.
    pub verify_time_us: u64,
}

impl SpeculativeCycleResult {
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

/// One sequence in a packed Speculative verification cohort.
///
/// `state_slot` and `generation` identify the sequence in the logical page
/// manager. Model state is supplied separately, in the same sequence-major
/// order, so this request remains model-neutral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeculativeVerificationItem<'a> {
    pub state_slot: StateSlot,
    pub generation: u64,
    pub proposal: &'a [u32],
    pub frontier: TargetFrontier,
}

/// Results from one true multi-sequence Speculative verification transaction.
///
/// The entries in `results` correspond one-for-one with the input items. Each
/// cycle result carries the same shared timings as scalar speculative metrics;
/// the top-level fields make their cohort-wide nature explicit.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeCohortResult {
    pub results: Vec<SpeculativeCycleResult>,
    pub transaction_time_us: u64,
    pub verify_time_us: u64,
}

/// Accumulated speculative-decoding telemetry across multiple cycles.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpeculativeMetrics {
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

impl SpeculativeMetrics {
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

    pub fn record(&mut self, result: &SpeculativeCycleResult) {
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
        result: &SpeculativeCycleResult,
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

// ── Speculative transaction ────────────────────────────────────────────────

/// Runs one true multi-sequence Speculative verification cohort.
///
/// The cohort is sequence-major and may be ragged: each item contributes one
/// anchor row followed by its local proposal. The target executes exactly one
/// provisional packed batch. Every accepted prefix is then retained by the
/// backend as one atomic cohort operation.
/// Progress of one exact, possibly suspended Speculative verification cohort.
pub(crate) enum SpeculativeCohortProgress<S> {
    Complete(SpeculativeCohortResult),
    Waiting(Box<PendingSpeculativeVerificationCohort<S>>),
}

/// Error from a resumable cohort operation.
///
/// `pending` is retained only when the executor still owns a live backend
/// continuation. Callers must quarantine or explicitly cancel that transaction;
/// dropping it would lose model state and provisional-page ownership.
pub(crate) struct SpeculativeCohortExecutionError<S> {
    error: Error,
    pending: Option<Box<PendingSpeculativeVerificationCohort<S>>>,
}

impl<S> SpeculativeCohortExecutionError<S> {
    pub(crate) fn into_parts(
        self,
    ) -> (Error, Option<Box<PendingSpeculativeVerificationCohort<S>>>) {
        (self.error, self.pending)
    }
}

struct SpeculativeCohortTransaction<S> {
    id: ExecutionTransactionId,
    transaction_start: Instant,
    verify_start: Instant,
    proposals: Vec<Vec<u32>>,
    executed_rows: Vec<usize>,
    query_ranges: Vec<Range<usize>>,
    reservations: Vec<KvReservation>,
    prepared_commit: Option<PreparedKvCommit>,
    verification_batch: ExecutionBatch,
    verification_branches: Vec<S>,
}

/// Exact ownership retained while target verification waits on selected experts.
pub(crate) struct PendingSpeculativeVerificationCohort<S> {
    transaction: SpeculativeCohortTransaction<S>,
    pending_progress: PendingModelProgress,
}

impl<S> PendingSpeculativeVerificationCohort<S> {
    pub(crate) fn pending_progress(&self) -> &PendingModelProgress {
        &self.pending_progress
    }
}

/// Begin one exact packed target-verification transaction.
pub(crate) fn begin_resumable_speculative_verification_cohort<R>(
    executor: &mut NativeMultiSessionExecutor<R>,
    page_manager: &mut KvPageManager,
    transaction_id: ExecutionTransactionId,
    source_states: &mut [R::SequenceState],
    items: &[SpeculativeVerificationItem<'_>],
    reservations: Vec<KvReservation>,
    top_k: NonZeroU32,
    retirements: &mut Vec<KvRetirement>,
) -> std::result::Result<
    SpeculativeCohortProgress<R::SequenceState>,
    SpeculativeCohortExecutionError<R::SequenceState>,
>
where
    R: MultiSessionRunner,
{
    let mut transaction = prepare_speculative_cohort_transaction(
        executor,
        page_manager,
        transaction_id,
        source_states,
        items,
        reservations,
        top_k,
        retirements,
    )
    .map_err(|error| SpeculativeCohortExecutionError {
        error,
        pending: None,
    })?;
    let reservation_views = match page_manager.reservation_views(&transaction.reservations) {
        Ok(views) => views,
        Err(error) => {
            return Err(SpeculativeCohortExecutionError {
                error: discard_speculative_transaction(
                    executor,
                    page_manager,
                    transaction,
                    false,
                    error,
                    retirements,
                ),
                pending: None,
            });
        }
    };
    match executor.begin_resumable_batch_with_kv(
        transaction_id,
        &mut transaction.verification_branches,
        &transaction.verification_batch,
        &reservation_views,
    ) {
        Ok(NativeBatchExecutionProgress::Complete(output)) => {
            finish_speculative_cohort_transaction(
                executor,
                page_manager,
                source_states,
                transaction,
                output,
                retirements,
            )
            .map(SpeculativeCohortProgress::Complete)
            .map_err(|error| SpeculativeCohortExecutionError {
                error,
                pending: None,
            })
        }
        Ok(NativeBatchExecutionProgress::Waiting(pending_progress)) => Ok(
            SpeculativeCohortProgress::Waiting(Box::new(PendingSpeculativeVerificationCohort {
                transaction,
                pending_progress,
            })),
        ),
        Err(error) => Err(reconcile_speculative_execution_error(
            executor,
            page_manager,
            transaction,
            error,
            retirements,
        )),
    }
}

/// Resume the exact stored verification batch and branch set.
pub(crate) fn resume_resumable_speculative_verification_cohort<R>(
    executor: &mut NativeMultiSessionExecutor<R>,
    page_manager: &mut KvPageManager,
    source_states: &mut [R::SequenceState],
    mut pending: PendingSpeculativeVerificationCohort<R::SequenceState>,
    leases: ferrule_common::ResidencyLeaseSet,
    retirements: &mut Vec<KvRetirement>,
) -> std::result::Result<
    SpeculativeCohortProgress<R::SequenceState>,
    SpeculativeCohortExecutionError<R::SequenceState>,
>
where
    R: MultiSessionRunner,
{
    let transaction_id = pending.transaction.id;
    let continuation = pending.pending_progress.continuation();
    match executor.resume_resumable_batch(
        transaction_id,
        &mut pending.transaction.verification_branches,
        &pending.transaction.verification_batch,
        continuation,
        leases,
    ) {
        Ok(NativeBatchExecutionProgress::Complete(output)) => {
            finish_speculative_cohort_transaction(
                executor,
                page_manager,
                source_states,
                pending.transaction,
                output,
                retirements,
            )
            .map(SpeculativeCohortProgress::Complete)
            .map_err(|error| SpeculativeCohortExecutionError {
                error,
                pending: None,
            })
        }
        Ok(NativeBatchExecutionProgress::Waiting(pending_progress)) => {
            pending.pending_progress = pending_progress;
            Ok(SpeculativeCohortProgress::Waiting(Box::new(pending)))
        }
        Err(error) => {
            if let Some(progress) = executor.pending_model_progress(transaction_id).cloned() {
                pending.pending_progress = progress;
                Err(SpeculativeCohortExecutionError {
                    error,
                    pending: Some(Box::new(pending)),
                })
            } else {
                let error = discard_speculative_transaction(
                    executor,
                    page_manager,
                    pending.transaction,
                    true,
                    error,
                    retirements,
                );
                Err(SpeculativeCohortExecutionError {
                    error,
                    pending: None,
                })
            }
        }
    }
}

/// Cancel one suspended cohort. Logical reservations and branch states are
/// released only after model quiescence and backend rollback are confirmed.
pub(crate) fn cancel_resumable_speculative_verification_cohort<R>(
    executor: &mut NativeMultiSessionExecutor<R>,
    page_manager: &mut KvPageManager,
    mut pending: PendingSpeculativeVerificationCohort<R::SequenceState>,
    retirements: &mut Vec<KvRetirement>,
) -> std::result::Result<(), SpeculativeCohortExecutionError<R::SequenceState>>
where
    R: MultiSessionRunner,
{
    let transaction_id = pending.transaction.id;
    let continuation = pending.pending_progress.continuation();
    if let Err(error) = executor.cancel_resumable_batch(
        transaction_id,
        &mut pending.transaction.verification_branches,
        continuation,
    ) {
        return Err(SpeculativeCohortExecutionError {
            error,
            pending: Some(Box::new(pending)),
        });
    }
    let transaction = pending.transaction;
    match cleanup_speculative_cohort(
        executor,
        page_manager,
        Some(transaction.id),
        transaction.reservations,
        transaction.prepared_commit,
        transaction.verification_branches,
        false,
        retirements,
    ) {
        None => Ok(()),
        Some(error) => Err(SpeculativeCohortExecutionError {
            error,
            pending: None,
        }),
    }
}

fn reconcile_speculative_execution_error<R: MultiSessionRunner>(
    executor: &mut NativeMultiSessionExecutor<R>,
    page_manager: &mut KvPageManager,
    transaction: SpeculativeCohortTransaction<R::SequenceState>,
    error: Error,
    retirements: &mut Vec<KvRetirement>,
) -> SpeculativeCohortExecutionError<R::SequenceState> {
    match executor.pending_model_progress(transaction.id).cloned() {
        Some(pending_progress) => SpeculativeCohortExecutionError {
            error,
            pending: Some(Box::new(PendingSpeculativeVerificationCohort {
                transaction,
                pending_progress,
            })),
        },
        None => SpeculativeCohortExecutionError {
            error: discard_speculative_transaction(
                executor,
                page_manager,
                transaction,
                true,
                error,
                retirements,
            ),
            pending: None,
        },
    }
}

fn prepare_speculative_cohort_transaction<R: MultiSessionRunner>(
    executor: &mut NativeMultiSessionExecutor<R>,
    page_manager: &mut KvPageManager,
    transaction_id: ExecutionTransactionId,
    source_states: &[R::SequenceState],
    items: &[SpeculativeVerificationItem<'_>],
    mut reservations: Vec<KvReservation>,
    top_k: NonZeroU32,
    retirements: &mut Vec<KvRetirement>,
) -> Result<SpeculativeCohortTransaction<R::SequenceState>> {
    let transaction_start = Instant::now();
    if items.is_empty() {
        return Err(Error::Execution(
            "Speculative verification cohort must contain at least one sequence".into(),
        ));
    }
    if source_states.len() != items.len() {
        return Err(Error::Execution(format!(
            "Speculative verification cohort state/item mismatch: states={} items={}",
            source_states.len(),
            items.len()
        )));
    }

    let executed_rows = items
        .iter()
        .map(|item| {
            item.proposal.len().checked_add(1).ok_or_else(|| {
                Error::Execution("Speculative verification row count overflow".into())
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if reservations.len() != items.len() {
        let error = Error::Execution(format!(
            "Speculative verification cohort reservation/item mismatch: reservations={} items={}",
            reservations.len(),
            items.len()
        ));
        return Err(discard_speculative_cohort(
            executor,
            page_manager,
            reservations,
            Vec::new(),
            false,
            error,
            retirements,
        ));
    }
    for (sequence, ((item, &rows), reservation)) in items
        .iter()
        .zip(&executed_rows)
        .zip(&reservations)
        .enumerate()
    {
        if reservation.state_slot != item.state_slot
            || reservation.generation != item.generation
            || reservation.positions.len() != rows
        {
            let error = Error::Execution(format!(
                "Speculative cohort sequence {sequence} reservation does not match its exact state/generation/row request"
            ));
            return Err(discard_speculative_cohort(
                executor,
                page_manager,
                reservations,
                Vec::new(),
                false,
                error,
                retirements,
            ));
        }
        if reservation.positions.start != item.frontier.position {
            let actual = reservation.positions.start;
            let error = Error::Execution(format!(
                "Speculative cohort sequence {sequence} frontier position {} does not match committed KV position {actual}",
                item.frontier.position
            ));
            return Err(discard_speculative_cohort(
                executor,
                page_manager,
                reservations,
                Vec::new(),
                false,
                error,
                retirements,
            ));
        }
    }

    let mut bindings = Vec::with_capacity(reservations.len());
    for reservation in &reservations {
        match page_manager.reservation_bindings(reservation) {
            Ok(sequence_bindings) => bindings.push(sequence_bindings),
            Err(error) => {
                return Err(discard_speculative_cohort(
                    executor,
                    page_manager,
                    reservations,
                    Vec::new(),
                    false,
                    error,
                    retirements,
                ));
            }
        }
    }
    let (verification_batch, query_ranges) =
        match build_speculative_cohort_batch(items, &bindings, top_k) {
            Ok(batch) => batch,
            Err(error) => {
                return Err(discard_speculative_cohort(
                    executor,
                    page_manager,
                    reservations,
                    Vec::new(),
                    false,
                    error,
                    retirements,
                ));
            }
        };

    let mut verification_branches = Vec::with_capacity(items.len());
    for (source, item) in source_states.iter().zip(items) {
        match executor.fork_sequence_state_from(source, item.frontier.position) {
            Ok(branch) => verification_branches.push(branch),
            Err(error) => {
                return Err(discard_speculative_cohort(
                    executor,
                    page_manager,
                    reservations,
                    verification_branches,
                    false,
                    error,
                    retirements,
                ));
            }
        }
    }
    for index in 0..reservations.len() {
        let execution_generation = executor
            .runner()
            .sequence_generation(&verification_branches[index]);
        if let Err(error) = page_manager.bind_reservation_execution(
            &mut reservations[index],
            verification_batch.sequences()[index].state_slot,
            execution_generation,
        ) {
            return Err(discard_speculative_cohort(
                executor,
                page_manager,
                reservations,
                verification_branches,
                false,
                error,
                retirements,
            ));
        }
    }

    Ok(SpeculativeCohortTransaction {
        id: transaction_id,
        transaction_start,
        verify_start: Instant::now(),
        proposals: items.iter().map(|item| item.proposal.to_vec()).collect(),
        executed_rows,
        query_ranges,
        reservations,
        prepared_commit: None,
        verification_batch,
        verification_branches,
    })
}

fn finish_speculative_cohort_transaction<R: MultiSessionRunner>(
    executor: &mut NativeMultiSessionExecutor<R>,
    page_manager: &mut KvPageManager,
    source_states: &mut [R::SequenceState],
    mut transaction: SpeculativeCohortTransaction<R::SequenceState>,
    output: ExecutionOutput,
    retirements: &mut Vec<KvRetirement>,
) -> Result<SpeculativeCohortResult> {
    let verify_time_us = transaction.verify_start.elapsed().as_micros() as u64;
    let global_row_top1 =
        match collect_global_row_top1(&output, transaction.verification_batch.len()) {
            Ok(rows) => rows,
            Err(error) => {
                return Err(discard_speculative_transaction(
                    executor,
                    page_manager,
                    transaction,
                    true,
                    error,
                    retirements,
                ));
            }
        };

    let mut verifications = Vec::with_capacity(transaction.proposals.len());
    for (sequence, (proposal, query)) in transaction
        .proposals
        .iter()
        .zip(&transaction.query_ranges)
        .enumerate()
    {
        let local_rows = match global_row_top1.get(query.clone()) {
            Some(rows) => rows,
            None => {
                let error = Error::Execution(format!(
                    "Speculative cohort sequence {sequence} query range {query:?} exceeds {} global rows",
                    global_row_top1.len()
                ));
                return Err(discard_speculative_transaction(
                    executor,
                    page_manager,
                    transaction,
                    true,
                    error,
                    retirements,
                ));
            }
        };
        match verify_causal_slice(local_rows, proposal) {
            Ok(verification) => verifications.push(verification),
            Err(error) => {
                return Err(discard_speculative_transaction(
                    executor,
                    page_manager,
                    transaction,
                    true,
                    error,
                    retirements,
                ));
            }
        }
    }

    let retained_rows = verifications
        .iter()
        .map(|verification| verification.accepted + 1)
        .collect::<Vec<_>>();
    if let Err(error) = executor.retain_prepared_prefixes(
        transaction.id,
        source_states,
        &mut transaction.verification_branches,
        &transaction.executed_rows,
        &retained_rows,
    ) {
        return Err(discard_speculative_transaction(
            executor,
            page_manager,
            transaction,
            true,
            error,
            retirements,
        ));
    }

    let commits = std::mem::take(&mut transaction.reservations)
        .into_iter()
        .zip(&retained_rows)
        .map(|(reservation, &rows)| KvReservationCommit::new(reservation, rows))
        .collect();
    transaction.prepared_commit = match page_manager.prepare_commit(commits) {
        Ok(prepared) => Some(prepared),
        Err(error) => {
            let (error, commits) = error.into_parts();
            transaction.reservations = commits
                .into_iter()
                .map(|commit| commit.reservation)
                .collect();
            return Err(discard_speculative_transaction(
                executor,
                page_manager,
                transaction,
                true,
                error,
                retirements,
            ));
        }
    };

    if let Err(error) =
        executor.commit_prepared_batch(transaction.id, &mut transaction.verification_branches)
    {
        return Err(discard_speculative_transaction(
            executor,
            page_manager,
            transaction,
            true,
            error,
            retirements,
        ));
    }
    let retirement = page_manager.publish_commit(
        transaction
            .prepared_commit
            .take()
            .expect("Speculative logical commit was prepared before backend commit"),
    );
    let state_promotion_error = promote_cohort_branches(
        executor,
        source_states,
        std::mem::take(&mut transaction.verification_branches),
    )
    .err();
    retirements.push(retirement);
    finish_speculative_cohort_publication(None, state_promotion_error, None)?;

    let transaction_time_us = transaction.transaction_start.elapsed().as_micros() as u64;
    let results = transaction
        .proposals
        .into_iter()
        .zip(verifications)
        .zip(transaction.executed_rows)
        .map(|((proposal, verification), verified_rows)| {
            let accepted_rows = verification.accepted;
            let committed_rows = accepted_rows + 1;
            let rejected = proposal.get(accepted_rows).copied();
            let target_next = verification.target_next;
            SpeculativeCycleResult {
                accepted: proposal[..accepted_rows].to_vec(),
                rejected,
                target_correction: rejected.map(|_| target_next.token_id),
                target_next: Some(target_next),
                accounting: SpeculativeCycleAccounting {
                    proposed_tokens: proposal.len(),
                    verified_rows,
                    accepted_draft_tokens: accepted_rows,
                    correction_tokens: usize::from(accepted_rows < proposal.len()),
                    externally_committed_tokens: committed_rows,
                    rolled_back_rows: verified_rows - committed_rows,
                },
                transaction_time_us,
                verify_time_us,
            }
        })
        .collect();
    Ok(SpeculativeCohortResult {
        results,
        transaction_time_us,
        verify_time_us,
    })
}

fn discard_speculative_transaction<R: MultiSessionRunner>(
    executor: &mut NativeMultiSessionExecutor<R>,
    page_manager: &mut KvPageManager,
    transaction: SpeculativeCohortTransaction<R::SequenceState>,
    rollback_backend: bool,
    cause: Error,
    retirements: &mut Vec<KvRetirement>,
) -> Error {
    match cleanup_speculative_cohort(
        executor,
        page_manager,
        Some(transaction.id),
        transaction.reservations,
        transaction.prepared_commit,
        transaction.verification_branches,
        rollback_backend,
        retirements,
    ) {
        None => cause,
        Some(cleanup) => Error::Execution(format!(
            "{cause}; Speculative transaction cleanup also failed: {cleanup}"
        )),
    }
}

fn build_speculative_cohort_batch(
    items: &[SpeculativeVerificationItem<'_>],
    bindings: &[KvReservationBindings],
    top_k: NonZeroU32,
) -> Result<(ExecutionBatch, Vec<Range<usize>>)> {
    if items.is_empty() || items.len() != bindings.len() {
        return Err(Error::Execution(format!(
            "Speculative cohort batch shape mismatch: items={} bindings={}",
            items.len(),
            bindings.len()
        )));
    }
    let total_rows = items.iter().try_fold(0usize, |total, item| {
        total
            .checked_add(item.proposal.len())
            .and_then(|rows| rows.checked_add(1))
            .ok_or_else(|| Error::Execution("Speculative cohort packed row count overflow".into()))
    })?;

    let mut token_ids = Vec::with_capacity(total_rows);
    let mut positions = Vec::with_capacity(total_rows);
    let mut kv_write_slots = Vec::with_capacity(total_rows);
    let mut logits = Vec::with_capacity(total_rows);
    let mut sequences = Vec::with_capacity(items.len());
    let mut kv_block_ids = Vec::new();
    let mut query_ranges = Vec::with_capacity(items.len());

    for (sequence, (item, bindings)) in items.iter().zip(bindings).enumerate() {
        let local_rows = item.proposal.len().checked_add(1).ok_or_else(|| {
            Error::Execution("Speculative verification row count overflow".into())
        })?;
        if bindings.write_slots.len() != local_rows {
            return Err(Error::Execution(format!(
                "Speculative cohort sequence {sequence} has {} KV write slots for {local_rows} rows",
                bindings.write_slots.len()
            )));
        }

        let query_start = token_ids.len();
        let query_end = query_start.checked_add(local_rows).ok_or_else(|| {
            Error::Execution("Speculative cohort packed query range overflow".into())
        })?;
        let query_start_u32 = u32::try_from(query_start)
            .map_err(|_| Error::Execution("Speculative cohort query start exceeds u32".into()))?;
        let query_end_u32 = u32::try_from(query_end)
            .map_err(|_| Error::Execution("Speculative cohort query end exceeds u32".into()))?;
        let context_len = u32::try_from(item.frontier.position).map_err(|_| {
            Error::Execution(format!(
                "Speculative cohort sequence {sequence} frontier position exceeds u32"
            ))
        })?;
        let local_rows_u32 = u32::try_from(local_rows).map_err(|_| {
            Error::Execution(format!(
                "Speculative cohort sequence {sequence} verification width exceeds u32"
            ))
        })?;
        let sequence_len = context_len.checked_add(local_rows_u32).ok_or_else(|| {
            Error::Execution(format!(
                "Speculative cohort sequence {sequence} sequence length overflow"
            ))
        })?;
        let block_start = u32::try_from(kv_block_ids.len())
            .map_err(|_| Error::Execution("Speculative cohort block table exceeds u32".into()))?;
        kv_block_ids.extend_from_slice(&bindings.block_ids);
        let block_end = u32::try_from(kv_block_ids.len())
            .map_err(|_| Error::Execution("Speculative cohort block table exceeds u32".into()))?;
        let dense_state_slot = u32::try_from(sequence).map(StateSlot::new).map_err(|_| {
            Error::Execution("Speculative cohort sequence count exceeds u32".into())
        })?;

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

fn discard_speculative_cohort<R: MultiSessionRunner>(
    executor: &mut NativeMultiSessionExecutor<R>,
    page_manager: &mut KvPageManager,
    reservations: Vec<KvReservation>,
    branches: Vec<R::SequenceState>,
    rollback_backend: bool,
    cause: Error,
    retirements: &mut Vec<KvRetirement>,
) -> Error {
    match cleanup_speculative_cohort(
        executor,
        page_manager,
        None,
        reservations,
        None,
        branches,
        rollback_backend,
        retirements,
    ) {
        None => cause,
        Some(cleanup) => Error::Execution(format!(
            "{cause}; Speculative cohort cleanup also failed: {cleanup}"
        )),
    }
}

fn cleanup_speculative_cohort<R: MultiSessionRunner>(
    executor: &mut NativeMultiSessionExecutor<R>,
    page_manager: &mut KvPageManager,
    transaction_id: Option<ExecutionTransactionId>,
    reservations: Vec<KvReservation>,
    prepared_commit: Option<PreparedKvCommit>,
    mut branches: Vec<R::SequenceState>,
    rollback_backend: bool,
    retirements: &mut Vec<KvRetirement>,
) -> Option<Error> {
    let mut cleanup_errors = Vec::new();
    let backend_quiesced = if rollback_backend {
        match transaction_id {
            Some(transaction_id) if executor.has_transaction(transaction_id) => {
                match executor.rollback_prepared_batch(transaction_id, &mut branches) {
                    Ok(()) => true,
                    Err(error) => {
                        cleanup_errors.push(format!("backend rollback: {error}"));
                        false
                    }
                }
            }
            Some(_) => true,
            None => {
                cleanup_errors.push("backend rollback has no transaction identity".into());
                false
            }
        }
    } else {
        true
    };

    if backend_quiesced {
        let retirement = match prepared_commit {
            Some(prepared) => Ok(page_manager.abort_prepared_commit(prepared)),
            None => page_manager
                .abort_reservations(reservations)
                .map_err(|error| error.into_parts().0),
        };
        match retirement {
            Ok(retirement) => retirements.push(retirement),
            Err(error) => cleanup_errors.push(format!("logical KV abort: {error}")),
        }
        for (sequence, branch) in branches.into_iter().enumerate() {
            if let Err(error) = executor.release_sequence_state(branch) {
                cleanup_errors.push(format!("branch release for sequence {sequence}: {error}"));
            }
        }
    } else {
        cleanup_errors.push(
            "backend transaction remains active; logical and branch ownership was quarantined"
                .into(),
        );
    }
    (!cleanup_errors.is_empty()).then(|| Error::Execution(cleanup_errors.join("; ")))
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
            "Speculative cohort state promotion failed: {}",
            errors.join("; ")
        )))
    }
}

fn finish_speculative_cohort_publication(
    backend: Option<Error>,
    state: Option<Error>,
    pages: Option<Error>,
) -> Result<()> {
    if backend.is_none() && state.is_none() && pages.is_none() {
        return Ok(());
    }
    Err(Error::Execution(format!(
        "Speculative cohort publication failed: backend={backend:?} state={state:?} pages={pages:?}"
    )))
}

#[derive(Debug, Clone, PartialEq)]
struct CausalVerification {
    accepted: usize,
    target_next: TokenLogit,
}

fn collect_global_row_top1(
    output: &ExecutionOutput,
    expected_rows: usize,
) -> Result<Vec<TokenLogit>> {
    let mut row_top1 = vec![None; expected_rows];
    for row in &output.logits {
        let row_index = usize::try_from(row.input_row)
            .map_err(|_| Error::Execution("Speculative output row exceeds usize".into()))?;
        let slot = row_top1.get_mut(row_index).ok_or_else(|| {
            Error::Execution(format!(
                "Speculative output row {row_index} exceeds verification row count {expected_rows}"
            ))
        })?;
        if slot.is_some() {
            return Err(Error::Execution(format!(
                "Speculative output contains duplicate row {row_index}"
            )));
        }
        let LogitsOutput::TopK(logits) = &row.logits else {
            return Err(Error::Execution(format!(
                "Speculative output row {row_index} is not top-k"
            )));
        };
        *slot = Some(*logits.first().ok_or_else(|| {
            Error::Execution(format!(
                "Speculative output row {row_index} has empty top-k"
            ))
        })?);
    }
    row_top1
        .into_iter()
        .enumerate()
        .map(|(row, top1)| {
            top1.ok_or_else(|| Error::Execution(format!("Speculative output is missing row {row}")))
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
        .ok_or_else(|| Error::Execution("Speculative output row count overflow".into()))?;
    if row_top1.len() != verified_rows {
        return Err(Error::Execution(format!(
            "Speculative causal slice has {} rows, expected {verified_rows}",
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
    })
}

#[cfg(test)]
fn verify_causal_prefix(output: &ExecutionOutput, proposal: &[u32]) -> Result<CausalVerification> {
    let verified_rows = proposal
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::Execution("Speculative output row count overflow".into()))?;
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
        assert_eq!(second.accepted, 3);
        assert_eq!(second.target_next.token_id, 42);
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
    fn speculative_metrics_record_explicit_accounting() {
        let mut metrics = SpeculativeMetrics::default();
        metrics.record(&SpeculativeCycleResult {
            accepted: vec![10, 11],
            rejected: Some(12),
            target_correction: Some(99),
            target_next: Some(logit(99, 1.0)),
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
        metrics.record(&SpeculativeCycleResult {
            accepted: vec![20, 21, 22, 23],
            rejected: None,
            target_correction: None,
            target_next: Some(logit(24, 1.0)),
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
        let result = SpeculativeCycleResult {
            accepted: vec![10, 11, 12, 13],
            rejected: None,
            target_correction: None,
            target_next: Some(logit(14, 1.0)),
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
