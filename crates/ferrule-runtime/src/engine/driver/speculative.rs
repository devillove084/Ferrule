//! Speculative operations on the driver's existing authority.

use super::*;

pub(super) fn proposal_confidence_probability(logit: f32) -> f32 {
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exponential = logit.exp();
        exponential / (1.0 + exponential)
    }
}

pub(super) fn confident_proposal_prefix_length(logits: &[f32], threshold: f32) -> Result<usize> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(Error::InvalidRequest {
            message: format!(
                "proposal confidence threshold must be finite and within [0, 1], got {threshold}"
            ),
        });
    }
    if threshold == 0.0 {
        return Ok(logits.len());
    }
    Ok(logits
        .iter()
        .position(|logit| proposal_confidence_probability(*logit) < threshold)
        .unwrap_or(logits.len()))
}

pub(super) struct PendingSpeculativeBookkeeping<S> {
    pub(super) next: PendingSpeculativeDriverCohort<S>,
    pub(super) finish: Option<(ContinuationId, crate::io::ResumeLease, ResumeDisposition)>,
    pub(super) started_ns: u64,
    pub(super) finished_ns: u64,
    pub(super) errors: Vec<String>,
}

pub(super) enum PendingSpeculativeDriverCohort<S> {
    AdmissionRollback(Box<PendingSpeculativeAdmissionRollback<S>>),
    Bookkeeping(Box<PendingSpeculativeBookkeeping<S>>),
    Proposing(Box<PendingNativeProposalCohort<S>>),
    Verifying(Box<PendingSpeculativeVerificationDriverCohort<S>>),
    Ending(Box<PendingSpeculativeEndingDriverCohort<S>>),
}

impl<S> PendingSpeculativeDriverCohort<S> {
    pub(super) fn accept_cancellation(&mut self, request: RequestId) {
        match self {
            Self::AdmissionRollback(pending) => pending.cancellation_request = Some(request),
            Self::Bookkeeping(pending) => pending.next.accept_cancellation(request),
            Self::Proposing(pending) => pending.cancellation_request = Some(request),
            Self::Verifying(pending) => pending.cancellation_request = Some(request),
            Self::Ending(pending) => pending.request_id = Some(request),
        }
    }

    pub(super) fn actions(&self) -> &[DecodeAction] {
        match self {
            Self::AdmissionRollback(pending) => &pending.admission.actions,
            Self::Bookkeeping(pending) => pending.next.actions(),
            Self::Proposing(pending) => &pending.actions,
            Self::Verifying(pending) => &pending.actions,
            Self::Ending(pending) => &pending.actions,
        }
    }

    pub(super) fn extend_pending_progress(&self, output: &mut Vec<PendingModelProgress>) {
        match self {
            Self::AdmissionRollback(_) | Self::Bookkeeping(_) => {}
            Self::Proposing(pending) => {
                output.extend(pending.slots.iter().filter_map(|slot| match &slot.status {
                    NativeProposalSlotStatus::Waiting(progress) => Some(progress.clone()),
                    NativeProposalSlotStatus::NotStarted
                    | NativeProposalSlotStatus::Complete { .. } => None,
                }));
            }
            Self::Verifying(pending) => {
                output.push(pending.verification.pending_progress().clone());
            }
            Self::Ending(_) => {}
        }
    }
}

/// Only the gap between completed proposals and backend verification admission.
/// Waiting/active verification keeps its existing, distinct ownership phases.
struct SpeculativeAdmission<S> {
    transaction: ExecutionTransactionId,
    cohort_start: Instant,
    actions: Vec<DecodeAction>,
    prepared: Vec<PreparedSpeculativeAction>,
    sessions: AdmissionSessions<S>,
}

pub(super) struct PendingSpeculativeAdmissionRollback<S> {
    admission: SpeculativeAdmission<S>,
    registry_pending: bool,
    error: Error,
    stage: &'static str,
    pub(super) cancellation_request: Option<RequestId>,
}

#[cfg(test)]
impl<S> PendingSpeculativeAdmissionRollback<S> {
    pub(super) fn primary_error(&self) -> &Error {
        &self.error
    }
    pub(super) fn retained_sessions(&self) -> (usize, usize) {
        (
            self.admission.sessions.schedules.len(),
            self.admission.sessions.states.len(),
        )
    }
    pub(super) fn registry_pending(&self) -> bool {
        self.registry_pending
    }
}

impl<S> SpeculativeAdmission<S> {
    fn rollback(self, error: Error, stage: &'static str) -> PendingSpeculativeAdmissionRollback<S> {
        PendingSpeculativeAdmissionRollback {
            admission: self,
            registry_pending: true,
            error,
            stage,
            cancellation_request: None,
        }
    }
}

pub(super) struct PendingNativeProposalCohort<S> {
    pub(super) transaction: ExecutionTransactionId,
    pub(super) cohort_start: Instant,
    pub(super) actions: Vec<DecodeAction>,
    pub(super) proposal_source: NativeProposalSource,
    pub(super) source_states: Vec<S>,
    pub(super) schedules: Vec<SuspendedSequenceSchedule>,
    pub(super) slots: Vec<PendingNativeProposalSlot>,
    pub(super) cancellation_request: Option<RequestId>,
    pub(super) abort_cause: Option<Error>,
    phase: NativeProposalPhase,
}

/// Requeue and custody ACK exist only after backend Abort has completed. The
/// phase cannot be mistaken for a still-runnable proposal on a cleanup retry.
enum NativeProposalPhase {
    Executing,
    BackendAbortedPendingCleanup {
        custody: Option<TransactionCustodyOutcome>,
        decode_requeued: bool,
    },
}

impl<S> PendingNativeProposalCohort<S> {
    #[cfg(test)]
    pub(super) fn has_pending_custody(&self) -> bool {
        matches!(
            self.phase,
            NativeProposalPhase::BackendAbortedPendingCleanup {
                custody: Some(_),
                ..
            }
        )
    }

    pub(super) fn backend_aborted(&self) -> bool {
        matches!(
            self.phase,
            NativeProposalPhase::BackendAbortedPendingCleanup { .. }
        )
    }
}

pub(super) enum SpeculativeEnding<S> {
    Publish(PreparedSpeculativeCohort<S>),
    Abort {
        transaction: SpeculativeCohortTransaction<S>,
        failure: Option<(Error, &'static str)>,
    },
    BackendCommittedPendingPublish {
        progress: QuiescedSpeculativePublishProgress<S>,
        retirements: VecDeque<PendingKvRetirement>,
        custody: Option<TransactionCustodyOutcome>,
    },
    BackendAbortedPendingCleanup {
        progress: QuiescedSpeculativeAbortProgress<S>,
        retirements: VecDeque<PendingKvRetirement>,
        custody: Option<TransactionCustodyOutcome>,
        failure: Option<(Error, &'static str)>,
        decode_requeued: bool,
    },
}

impl<S> SpeculativeEnding<S> {
    pub(super) fn is_post_terminal(&self) -> bool {
        matches!(
            self,
            Self::BackendCommittedPendingPublish { .. } | Self::BackendAbortedPendingCleanup { .. }
        )
    }

    fn backend_submission(
        &mut self,
    ) -> Option<(ExecutionTransactionId, &mut [S], TransactionEndIntent)> {
        match self {
            Self::Publish(prepared) => {
                let transaction = prepared.transaction_mut();
                Some((
                    transaction.id(),
                    transaction.states_mut(),
                    TransactionEndIntent::Publish,
                ))
            }
            Self::Abort { transaction, .. } => Some((
                transaction.id(),
                transaction.states_mut(),
                TransactionEndIntent::Abort,
            )),
            Self::BackendCommittedPendingPublish { .. }
            | Self::BackendAbortedPendingCleanup { .. } => None,
        }
    }

    fn backend_completed(self, started_ns: u64, finished_ns: u64, cancelled: bool) -> Self {
        match self {
            Self::Publish(prepared) => Self::BackendCommittedPendingPublish {
                progress: QuiescedSpeculativePublishProgress::from_prepared(prepared),
                retirements: VecDeque::new(),
                custody: Some(TransactionCustodyOutcome::Committed {
                    started_ns,
                    finished_ns,
                }),
            },
            Self::Abort {
                transaction,
                failure,
            } => Self::BackendAbortedPendingCleanup {
                progress: QuiescedSpeculativeAbortProgress::from_transaction(transaction),
                retirements: VecDeque::new(),
                custody: Some(if cancelled {
                    TransactionCustodyOutcome::Cancelled
                } else {
                    TransactionCustodyOutcome::RolledBack
                }),
                failure,
                decode_requeued: false,
            },
            _ => unreachable!("only a submitted ending can acknowledge backend completion"),
        }
    }
}

pub(super) struct PendingSpeculativeEndingDriverCohort<S> {
    pub(super) transaction: ExecutionTransactionId,
    pub(super) cohort_start: Instant,
    pub(super) started_ns: u64,
    pub(super) actions: Vec<DecodeAction>,
    pub(super) prepared: Vec<PreparedSpeculativeAction>,
    pub(super) source_states: Vec<S>,
    pub(super) schedules: Vec<SuspendedSequenceSchedule>,
    pub(super) ending: SpeculativeEnding<S>,
    pub(super) request_id: Option<RequestId>,
    pub(super) continuation: Option<ContinuationId>,
}

pub(super) struct PendingSpeculativeVerificationDriverCohort<S> {
    pub(super) transaction: ExecutionTransactionId,
    pub(super) cohort_start: Instant,
    pub(super) actions: Vec<DecodeAction>,
    pub(super) prepared: Vec<PreparedSpeculativeAction>,
    pub(super) source_states: Vec<S>,
    pub(super) schedules: Vec<SuspendedSequenceSchedule>,
    pub(super) verification: PendingSpeculativeVerificationCohort<S>,
    pub(super) cancellation_request: Option<RequestId>,
}

pub(super) struct PendingNativeProposalSlot {
    pub(super) sequence: SequenceState,
    pub(super) page_slot: StateSlot,
    pub(super) max_drafts: usize,
    pub(super) proposal_start: Option<Instant>,
    pub(super) status: NativeProposalSlotStatus,
}

pub(super) enum NativeProposalSlotStatus {
    NotStarted,
    Waiting(PendingModelProgress),
    Complete {
        proposal: NativeProposal,
        proposal_time_us: u64,
        prepared: Box<Option<PreparedSpeculativeAction>>,
    },
}

pub(super) struct PreparedSpeculativeAction {
    pub(super) sequence: SequenceState,
    pub(super) page_slot: StateSlot,
    pub(super) proposal: Vec<u32>,
    pub(super) proposal_time_us: u64,
}

pub(super) fn record_speculative_sequence_metrics(
    metrics: &mut SpeculativeMetrics,
    result: &SpeculativeCycleResult,
    proposal_time_us: u64,
    runtime_emitted_tokens: usize,
) {
    let accounting = result.accounting;
    metrics.cycles = metrics.cycles.saturating_add(1);
    metrics.proposed_tokens = metrics
        .proposed_tokens
        .saturating_add(accounting.proposed_tokens);
    metrics.verified_rows = metrics
        .verified_rows
        .saturating_add(accounting.verified_rows);
    metrics.accepted_draft_tokens = metrics
        .accepted_draft_tokens
        .saturating_add(accounting.accepted_draft_tokens);
    metrics.correction_tokens = metrics
        .correction_tokens
        .saturating_add(accounting.correction_tokens);
    metrics.externally_committed_tokens = metrics
        .externally_committed_tokens
        .saturating_add(accounting.externally_committed_tokens);
    metrics.rolled_back_rows = metrics
        .rolled_back_rows
        .saturating_add(accounting.rolled_back_rows);
    metrics.rejected_tokens = metrics
        .rejected_tokens
        .saturating_add(result.rejected.is_some() as usize);
    if metrics.accepted_prefix_histogram.len() <= accounting.accepted_draft_tokens {
        metrics
            .accepted_prefix_histogram
            .resize(accounting.accepted_draft_tokens + 1, 0);
    }
    metrics.accepted_prefix_histogram[accounting.accepted_draft_tokens] =
        metrics.accepted_prefix_histogram[accounting.accepted_draft_tokens].saturating_add(1);
    metrics.total_proposal_time_us = metrics
        .total_proposal_time_us
        .saturating_add(proposal_time_us);
    metrics.record_runtime_emitted_tokens(runtime_emitted_tokens);
}

pub(super) fn record_speculative_cohort_metrics(
    metrics: &mut SpeculativeMetrics,
    transaction_time_us: u64,
    verify_time_us: u64,
    complete_cohort_time_us: u64,
) {
    metrics.total_transaction_time_us = metrics
        .total_transaction_time_us
        .saturating_add(transaction_time_us);
    metrics.total_verify_time_us = metrics.total_verify_time_us.saturating_add(verify_time_us);
    metrics.total_cycle_time_us = metrics
        .total_cycle_time_us
        .saturating_add(complete_cohort_time_us);
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner,
    C: SequenceSlotPool,
{
    pub(super) fn progress_speculative_admission_rollback(
        &mut self,
        mut pending: PendingSpeculativeAdmissionRollback<R::SequenceState>,
    ) -> Result<ResidentDriverStep>
    where
        R: ResidentModelRunner,
    {
        let transaction = pending.admission.transaction;
        let cleanup = (|| -> Result<()> {
            if pending.registry_pending {
                #[cfg(test)]
                self.fail_stage("spec admission custody")?;
                self.finish_speculative_transaction_custody(
                    transaction,
                    TransactionCustodyOutcome::RolledBack,
                )?;
                pending.registry_pending = false;
            }
            pending.admission.sessions.restore(self)
        })();
        if let Err(error) = cleanup {
            self.speculative_transactions.insert(
                transaction,
                PendingSpeculativeDriverCohort::AdmissionRollback(Box::new(pending)),
            );
            return Err(error);
        }
        let error = self.abort_speculative_decode_batch(
            &pending.admission.actions,
            pending.error,
            pending.stage,
        );
        let cleanup = self
            .finish_transaction_cancellations(transaction, pending.cancellation_request)
            .map(|_| ());
        Err(Error::with_cleanup(
            "speculative admission cancellation",
            error,
            cleanup,
        ))
    }

    pub(super) fn finish_speculative_transaction_custody(
        &mut self,
        transaction: ExecutionTransactionId,
        outcome: TransactionCustodyOutcome,
    ) -> Result<()> {
        let now_ns = self.runtime_now_ns();
        self.load_registry
            .finish_transaction_custody(transaction, outcome, now_ns)
            .map_err(Error::from)
    }

    pub(super) fn cancel_native_proposal_cohort(
        &mut self,
        mut pending: PendingNativeProposalCohort<R::SequenceState>,
        request_id: Option<RequestId>,
        failure: Option<Error>,
    ) -> Result<TransactionEndProgress>
    where
        R: ResidentModelRunner,
    {
        if let Some(request_id) = request_id {
            pending.cancellation_request = Some(request_id);
        }
        if let Some(failure) = failure {
            pending.abort_cause = Some(match pending.abort_cause.take() {
                Some(previous) => {
                    Error::combine("speculative proposal cancellation", previous, failure)
                }
                None => failure,
            });
        }

        let transaction = pending.transaction;
        if !pending.backend_aborted() {
            match self.executor.end_transaction(
                transaction,
                &mut pending.source_states,
                TransactionEndIntent::Abort,
            ) {
                Ok(TransactionEndProgress::Pending) => {
                    self.speculative_transactions.insert(
                        transaction,
                        PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
                    );
                    return Ok(TransactionEndProgress::Pending);
                }
                Ok(TransactionEndProgress::Complete) => {
                    pending.phase = NativeProposalPhase::BackendAbortedPendingCleanup {
                        custody: Some(if pending.cancellation_request.is_some() {
                            TransactionCustodyOutcome::Cancelled
                        } else {
                            TransactionCustodyOutcome::RolledBack
                        }),
                        decode_requeued: false,
                    };
                }
                Err(error) => {
                    self.speculative_transactions.insert(
                        transaction,
                        PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
                    );
                    return Err(error);
                }
            }
        }
        let continuations = pending
            .slots
            .iter()
            .filter_map(|slot| match &slot.status {
                NativeProposalSlotStatus::Waiting(progress) => Some(progress.continuation()),
                NativeProposalSlotStatus::NotStarted
                | NativeProposalSlotStatus::Complete { .. } => None,
            })
            .collect::<Vec<_>>();
        for continuation in continuations {
            if let Err(error) = self
                .detach_registered_continuation(continuation, CancellationReason::ExternalRequest)
            {
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
                );
                return Err(error);
            }
        }
        let NativeProposalPhase::BackendAbortedPendingCleanup {
            custody,
            decode_requeued,
        } = &mut pending.phase
        else {
            unreachable!("proposal cleanup requires backend quiescence")
        };
        if let Some(outcome) = custody.take()
            && let Err(error) = self.finish_speculative_transaction_custody(transaction, outcome)
        {
            *custody = Some(outcome);
            self.speculative_transactions.insert(
                transaction,
                PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
            );
            return Err(error);
        }
        if let Err(error) = self.progress_transaction_session_restore(
            &mut pending.schedules,
            &mut pending.source_states,
        ) {
            self.speculative_transactions.insert(
                transaction,
                PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
            );
            return Err(error);
        }
        if !*decode_requeued {
            if let Err(error) = self
                .scheduler
                .requeue_decode_actions_front(&pending.actions)
            {
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
                );
                return Err(error);
            }
            *decode_requeued = true;
        }
        match pending.abort_cause {
            Some(error) => Err(error),
            None => Ok(TransactionEndProgress::Complete),
        }
    }

    pub(super) fn cancel_speculative_verification_cohort(
        &mut self,
        pending: PendingSpeculativeVerificationDriverCohort<R::SequenceState>,
        request_id: RequestId,
    ) -> Result<TransactionEndProgress>
    where
        R: ResidentModelRunner,
    {
        let PendingSpeculativeVerificationDriverCohort {
            transaction,
            cohort_start,
            actions,
            prepared,
            source_states,
            schedules,
            verification,
            cancellation_request: _,
        } = pending;
        let continuation = verification.pending_progress().continuation();
        let ending = PendingSpeculativeEndingDriverCohort {
            transaction,
            cohort_start,
            started_ns: self.runtime_now_ns(),
            actions,
            prepared,
            source_states,
            schedules,
            ending: SpeculativeEnding::Abort {
                transaction: verification.into_transaction(),
                failure: None,
            },
            request_id: Some(request_id),
            continuation: Some(continuation),
        };
        self.drive_speculative_ending(ending, &mut |_| Ok(()))
            .map(|step| {
                if matches!(step, ResidentDriverStep::Blocked) {
                    TransactionEndProgress::Pending
                } else {
                    TransactionEndProgress::Complete
                }
            })
    }

    pub(super) fn drive_speculative_ending<F>(
        &mut self,
        mut pending: PendingSpeculativeEndingDriverCohort<R::SequenceState>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        R: ResidentModelRunner,
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transaction = pending.transaction;
        let terminal =
            pending
                .ending
                .backend_submission()
                .map(|(backend_transaction, states, intent)| {
                    debug_assert_eq!(transaction, backend_transaction);
                    self.executor
                        .end_transaction(backend_transaction, states, intent)
                });
        if let Some(terminal) = terminal {
            match terminal {
                Ok(TransactionEndProgress::Pending) => {
                    self.speculative_transactions.insert(
                        transaction,
                        PendingSpeculativeDriverCohort::Ending(Box::new(pending)),
                    );
                    return Ok(ResidentDriverStep::Blocked);
                }
                Err(error) => {
                    self.speculative_transactions.insert(
                        transaction,
                        PendingSpeculativeDriverCohort::Ending(Box::new(pending)),
                    );
                    return Err(error);
                }
                Ok(TransactionEndProgress::Complete) => {
                    pending.ending = pending.ending.backend_completed(
                        pending.started_ns,
                        self.runtime_now_ns(),
                        pending.request_id.is_some(),
                    );
                }
            }
        }

        match pending.ending {
            SpeculativeEnding::BackendCommittedPendingPublish { .. } => {
                self.publish_quiesced_speculative_ending(pending, on_token)
            }
            SpeculativeEnding::BackendAbortedPendingCleanup { .. } => {
                self.cleanup_quiesced_speculative_ending(pending)
            }
            SpeculativeEnding::Publish(_) | SpeculativeEnding::Abort { .. } => {
                unreachable!("completed speculative ending was converted to post-terminal progress")
            }
        }
    }

    pub(super) fn publish_quiesced_speculative_ending<F>(
        &mut self,
        mut pending: PendingSpeculativeEndingDriverCohort<R::SequenceState>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        R: ResidentModelRunner,
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transaction = pending.transaction;
        let mut newly_retired = Vec::new();
        let progress_result = {
            let SpeculativeEnding::BackendCommittedPendingPublish { progress, .. } =
                &mut pending.ending
            else {
                unreachable!("speculative publish requires post-terminal publish progress")
            };
            publish_quiesced_speculative_cohort(
                &mut self.executor,
                self.page_manager
                    .as_mut()
                    .expect("speculative transaction retains its page manager"),
                &mut pending.source_states,
                progress,
                &mut newly_retired,
            )
        };
        if let SpeculativeEnding::BackendCommittedPendingPublish { retirements, .. } =
            &mut pending.ending
        {
            retirements.extend(
                newly_retired
                    .into_iter()
                    .map(PendingKvRetirement::BackendRelease),
            );
        }
        if let Err(error) = progress_result {
            return self.retain_speculative_ending_error(pending, error);
        }
        if let Err(error) = self.progress_speculative_ending_retirements(&mut pending) {
            return self.retain_speculative_ending_error(pending, error);
        }

        let custody = match &mut pending.ending {
            SpeculativeEnding::BackendCommittedPendingPublish { custody, .. } => custody.take(),
            _ => unreachable!("speculative publish retained its post-terminal phase"),
        };
        if let Some(outcome) = custody
            && let Err(error) = self.finish_speculative_transaction_custody(transaction, outcome)
        {
            if let SpeculativeEnding::BackendCommittedPendingPublish { custody, .. } =
                &mut pending.ending
            {
                *custody = Some(outcome);
            }
            return self.retain_speculative_ending_error(pending, error);
        }
        if let Some(continuation) = pending.continuation
            && let Err(error) = self
                .detach_registered_continuation(continuation, CancellationReason::ExternalRequest)
        {
            return self.retain_speculative_ending_error(pending, error);
        }
        pending.continuation = None;
        if let Err(error) = self.progress_transaction_session_restore(
            &mut pending.schedules,
            &mut pending.source_states,
        ) {
            return self.retain_speculative_ending_error(pending, error);
        }
        let cohort = match &mut pending.ending {
            SpeculativeEnding::BackendCommittedPendingPublish { progress, .. } => progress
                .take_result()
                .expect("completed speculative publish retains its result"),
            _ => unreachable!("speculative publish retained its post-terminal phase"),
        };
        let cleanup_actions = pending.actions.clone();
        let cancellation_request = pending.request_id;
        let step = match self.publish_speculative_decode_cohort(
            transaction,
            pending.cohort_start,
            pending.actions,
            pending.prepared,
            cohort,
        ) {
            Ok(step) => step,
            Err(error) => {
                let error = self.abort_speculative_decode_batch(
                    &cleanup_actions,
                    error,
                    "committed speculative publication",
                );
                let cleanup = self
                    .finish_transaction_cancellations(transaction, cancellation_request)
                    .map(|_| ());
                return Err(Error::with_cleanup(
                    "committed speculative cancellation",
                    error,
                    cleanup,
                ));
            }
        };
        self.finish_transaction_cancellations(transaction, cancellation_request)?;
        self.flush_committed_token_outbox(on_token)?;
        Ok(step)
    }

    pub(super) fn cleanup_quiesced_speculative_ending(
        &mut self,
        mut pending: PendingSpeculativeEndingDriverCohort<R::SequenceState>,
    ) -> Result<ResidentDriverStep>
    where
        R: ResidentModelRunner,
    {
        let transaction = pending.transaction;
        let mut newly_retired = Vec::new();
        let progress_result = {
            let SpeculativeEnding::BackendAbortedPendingCleanup { progress, .. } =
                &mut pending.ending
            else {
                unreachable!("speculative abort requires post-terminal abort progress")
            };
            abort_quiesced_speculative_transaction(
                &mut self.executor,
                self.page_manager
                    .as_mut()
                    .expect("speculative transaction retains its page manager"),
                progress,
                &mut newly_retired,
            )
        };
        if let SpeculativeEnding::BackendAbortedPendingCleanup { retirements, .. } =
            &mut pending.ending
        {
            retirements.extend(
                newly_retired
                    .into_iter()
                    .map(PendingKvRetirement::BackendRelease),
            );
        }
        if let Err(error) = progress_result {
            return self.retain_speculative_ending_error(pending, error);
        }
        if let Err(error) = self.progress_speculative_ending_retirements(&mut pending) {
            return self.retain_speculative_ending_error(pending, error);
        }

        let custody = match &mut pending.ending {
            SpeculativeEnding::BackendAbortedPendingCleanup { custody, .. } => custody.take(),
            _ => unreachable!("speculative abort retained its post-terminal phase"),
        };
        if let Some(outcome) = custody
            && let Err(error) = self.finish_speculative_transaction_custody(transaction, outcome)
        {
            if let SpeculativeEnding::BackendAbortedPendingCleanup { custody, .. } =
                &mut pending.ending
            {
                *custody = Some(outcome);
            }
            return self.retain_speculative_ending_error(pending, error);
        }
        if let Some(continuation) = pending.continuation
            && let Err(error) = self
                .detach_registered_continuation(continuation, CancellationReason::ExternalRequest)
        {
            return self.retain_speculative_ending_error(pending, error);
        }
        pending.continuation = None;
        if let Err(error) = self.progress_transaction_session_restore(
            &mut pending.schedules,
            &mut pending.source_states,
        ) {
            return self.retain_speculative_ending_error(pending, error);
        }

        let (failure, mut decode_requeued) = match &mut pending.ending {
            SpeculativeEnding::BackendAbortedPendingCleanup {
                failure,
                decode_requeued,
                ..
            } => (failure.take(), *decode_requeued),
            _ => unreachable!("speculative abort retained its post-terminal phase"),
        };
        if let Some((error, stage)) = failure {
            let error = self.abort_speculative_decode_batch(&pending.actions, error, stage);
            let cleanup = self
                .finish_transaction_cancellations(transaction, pending.request_id)
                .map(|_| ());
            return Err(Error::with_cleanup(
                "speculative abort cancellation",
                error,
                cleanup,
            ));
        }
        if !decode_requeued {
            if let Err(error) = self
                .scheduler
                .requeue_decode_actions_front(&pending.actions)
            {
                return self.retain_speculative_ending_error(pending, error);
            }
            decode_requeued = true;
            if let SpeculativeEnding::BackendAbortedPendingCleanup {
                decode_requeued: retained,
                ..
            } = &mut pending.ending
            {
                *retained = true;
            }
        }
        let requested = pending.request_id;
        if let Err(error) = self.finish_transaction_cancellations(transaction, requested) {
            debug_assert!(decode_requeued);
            return self.retain_speculative_ending_error(pending, error);
        }
        Ok(ResidentDriverStep::Executed {
            action_kind: ResidentActionKind::Cancel,
            rows: 0,
            staged: 0,
            finished: 0,
        })
    }

    pub(super) fn retain_speculative_ending_error<T>(
        &mut self,
        pending: PendingSpeculativeEndingDriverCohort<R::SequenceState>,
        error: Error,
    ) -> Result<T> {
        self.speculative_transactions.insert(
            pending.transaction,
            PendingSpeculativeDriverCohort::Ending(Box::new(pending)),
        );
        Err(error)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn drive_quiesced_speculative_failure<F>(
        &mut self,
        transaction: ExecutionTransactionId,
        cohort_start: Instant,
        started_ns: u64,
        actions: Vec<DecodeAction>,
        prepared: Vec<PreparedSpeculativeAction>,
        source_states: Vec<R::SequenceState>,
        schedules: Vec<SuspendedSequenceSchedule>,
        cleanup: QuiescedSpeculativeAbortProgress<R::SequenceState>,
        retirements: Vec<KvRetirement>,
        error: Error,
        stage: &'static str,
        request_id: Option<RequestId>,
        continuation: Option<ContinuationId>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        R: ResidentModelRunner,
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        self.drive_speculative_ending(
            PendingSpeculativeEndingDriverCohort {
                transaction,
                cohort_start,
                started_ns,
                actions,
                prepared,
                source_states,
                schedules,
                ending: SpeculativeEnding::BackendAbortedPendingCleanup {
                    progress: cleanup,
                    retirements: retirements
                        .into_iter()
                        .map(PendingKvRetirement::BackendRelease)
                        .collect(),
                    custody: Some(if request_id.is_some() {
                        TransactionCustodyOutcome::Cancelled
                    } else {
                        TransactionCustodyOutcome::RolledBack
                    }),
                    failure: Some((error, stage)),
                    decode_requeued: false,
                },
                request_id,
                continuation,
            },
            on_token,
        )
    }
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: ResidentModelRunner,
    C: SequenceSlotPool,
{
    pub(super) fn try_execute_speculative_decode_batch<F>(
        &mut self,
        actions: &[DecodeAction],
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transaction = self.take_transaction_id()?;
        let cohort_start = Instant::now();
        let proposal_source = match self.executor.runner().native_proposal_source() {
            Ok(Some(proposal_source)) => proposal_source,
            Ok(None) => {
                return Err(self.abort_speculative_decode_batch(
                    actions,
                    Error::InvalidRequest {
                        message:
                            "resident decode entered proposal verification without model capability"
                                .into(),
                    },
                    "production speculative initialization",
                ));
            }
            Err(error) => {
                return Err(self.abort_speculative_decode_batch(
                    actions,
                    error.into(),
                    "production speculative initialization",
                ));
            }
        };
        if let Err(error) = proposal_source.validate() {
            return Err(self.abort_speculative_decode_batch(
                actions,
                error.into(),
                "production speculative initialization",
            ));
        }

        let slots = match self.prepare_native_proposal_slots(actions) {
            Ok(slots) => slots,
            Err(error) => {
                return Err(self.abort_speculative_decode_batch(
                    actions,
                    error,
                    "production speculative proposal preparation",
                ));
            }
        };
        let session_ids = actions
            .iter()
            .map(|action| action.session_id)
            .collect::<Vec<_>>();
        let sessions = match AdmissionSessions::claim(self, transaction, &session_ids) {
            Ok(ownership) => ownership,
            Err(error) => {
                return Err(self.abort_speculative_decode_batch(
                    actions,
                    error,
                    "production speculative state collection",
                ));
            }
        };
        let (schedules, source_states) = sessions.commit();
        self.advance_native_proposal_cohort(
            PendingNativeProposalCohort {
                transaction,
                cohort_start,
                actions: actions.to_vec(),
                proposal_source,
                source_states,
                schedules,
                slots,
                cancellation_request: None,
                abort_cause: None,
                phase: NativeProposalPhase::Executing,
            },
            on_token,
            None,
        )
    }

    pub(super) fn handoff_speculative_progress(
        &mut self,
        transaction: ExecutionTransactionId,
        next: PendingSpeculativeDriverCohort<R::SequenceState>,
        finish: Option<(ContinuationId, crate::io::ResumeLease, ResumeDisposition)>,
        started_ns: u64,
    ) -> Result<PendingSpeculativeDriverCohort<R::SequenceState>> {
        let finished_ns = self.runtime_now_ns();
        self.speculative_transactions.insert(
            transaction,
            PendingSpeculativeDriverCohort::Bookkeeping(Box::new(PendingSpeculativeBookkeeping {
                next,
                finish,
                started_ns,
                finished_ns,
                errors: Vec::new(),
            })),
        );
        self.finish_speculative_bookkeeping(transaction)
    }

    pub(super) fn finish_speculative_bookkeeping(
        &mut self,
        transaction: ExecutionTransactionId,
    ) -> Result<PendingSpeculativeDriverCohort<R::SequenceState>> {
        let result = (|| -> Result<()> {
            #[cfg(test)]
            if matches!(self.speculative_transactions.get(&transaction), Some(PendingSpeculativeDriverCohort::Bookkeeping(pending)) if pending.finish.is_some())
            {
                self.fail_stage("spec finish resume")?;
            }
            let finished_ns = self.runtime_now_ns();
            let Some(PendingSpeculativeDriverCohort::Bookkeeping(pending)) =
                self.speculative_transactions.get_mut(&transaction)
            else {
                unreachable!("bookkeeping owner")
            };
            if let Some((continuation, lease, disposition)) = pending.finish.as_mut() {
                self.load_registry.finish_resume(
                    lease,
                    *disposition,
                    pending.started_ns,
                    finished_ns,
                )?;
                let consumed =
                    (*disposition == ResumeDisposition::Consumed).then_some(*continuation);
                pending.finish = None;
                if let Some(continuation) = consumed {
                    self.unregister_continuation(continuation);
                }
            }
            #[cfg(test)]
            if matches!(self.speculative_transactions.get(&transaction), Some(PendingSpeculativeDriverCohort::Bookkeeping(pending)) if !matches!(pending.next, PendingSpeculativeDriverCohort::Proposing(_)))
            {
                self.fail_stage("verification observation")?;
            }
            #[cfg(test)]
            self.fail_stage("spec observation")?;
            let Some(PendingSpeculativeDriverCohort::Bookkeeping(pending)) =
                self.speculative_transactions.get(&transaction)
            else {
                unreachable!()
            };
            // Retry the original span, not time spent waiting for cleanup.
            self.load_registry
                .record_runnable_work(pending.started_ns, pending.finished_ns)?;
            Ok(())
        })();
        if let Err(error) = result {
            if let Some(PendingSpeculativeDriverCohort::Bookkeeping(pending)) =
                self.speculative_transactions.get_mut(&transaction)
            {
                pending.errors.push(error.to_string());
            }
            return Err(error);
        }
        let Some(PendingSpeculativeDriverCohort::Bookkeeping(pending)) =
            self.speculative_transactions.remove(&transaction)
        else {
            unreachable!()
        };
        Ok(pending.next)
    }

    pub(super) fn drive_speculative_bookkeeping<F>(
        &mut self,
        transaction: ExecutionTransactionId,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        #[cfg(test)]
        self.tick_trace.push(DriverTickEvent::Bookkeeping);
        let pending = self.finish_speculative_bookkeeping(transaction)?;
        self.accept_speculative_progress(pending, on_token)
    }

    pub(super) fn accept_speculative_progress<F>(
        &mut self,
        pending: PendingSpeculativeDriverCohort<R::SequenceState>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        match pending {
            PendingSpeculativeDriverCohort::AdmissionRollback(pending) => {
                self.progress_speculative_admission_rollback(*pending)
            }
            PendingSpeculativeDriverCohort::Proposing(mut pending) => {
                self.register_proposal_progress(&mut pending);
                self.advance_native_proposal_cohort(*pending, on_token, None)
            }
            PendingSpeculativeDriverCohort::Verifying(pending) => {
                let transaction = pending.transaction;
                if let Err(error) = self.register_pending_progress(
                    pending.verification.pending_progress(),
                    crate::scheduling::ResourceDemand::required(
                        ExecutionPhase::SpeculativeVerification,
                    ),
                ) {
                    return self.drive_speculative_ending(
                        PendingSpeculativeEndingDriverCohort {
                            transaction,
                            cohort_start: pending.cohort_start,
                            started_ns: self.runtime_now_ns(),
                            actions: pending.actions,
                            prepared: pending.prepared,
                            source_states: pending.source_states,
                            schedules: pending.schedules,
                            ending: SpeculativeEnding::Abort {
                                transaction: pending.verification.into_transaction(),
                                failure: Some((error, "speculative continuation registration")),
                            },
                            request_id: pending.cancellation_request,
                            continuation: None,
                        },
                        on_token,
                    );
                }
                if let Some(request_id) = pending.cancellation_request {
                    self.cancel_speculative_verification_cohort(*pending, request_id)?;
                    return Ok(ResidentDriverStep::Blocked);
                }
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::Verifying(pending),
                );
                Ok(ResidentDriverStep::WaitingForModelProgress(
                    self.pending_model_progresses(),
                ))
            }
            PendingSpeculativeDriverCohort::Ending(pending) => {
                self.drive_speculative_ending(*pending, on_token)
            }
            PendingSpeculativeDriverCohort::Bookkeeping(_) => unreachable!("nested bookkeeping"),
        }
    }

    pub(super) fn register_proposal_progress(
        &mut self,
        pending: &mut PendingNativeProposalCohort<R::SequenceState>,
    ) {
        for slot in &pending.slots {
            if let NativeProposalSlotStatus::Waiting(waiting) = &slot.status {
                if !self.continuations.contains(waiting.continuation()) {
                    if let Err(error) = self.register_pending_progress(
                        waiting,
                        crate::scheduling::ResourceDemand::required(
                            ExecutionPhase::SpeculativeProposal,
                        ),
                    ) {
                        pending.abort_cause.get_or_insert(error);
                    }
                }
            }
        }
    }

    pub(super) fn advance_native_proposal_cohort<F>(
        &mut self,
        mut pending: PendingNativeProposalCohort<R::SequenceState>,
        on_token: &mut F,
        mut resume: Option<(ContinuationId, crate::io::ResumeLease)>,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        debug_assert_eq!(pending.actions.len(), pending.source_states.len());
        debug_assert_eq!(pending.actions.len(), pending.slots.len());
        if pending.cancellation_request.is_some() || pending.abort_cause.is_some() {
            let transaction = pending.transaction;
            let request_id = pending.cancellation_request;
            return match self.cancel_native_proposal_cohort(pending, None, None)? {
                TransactionEndProgress::Pending => Ok(ResidentDriverStep::Blocked),
                TransactionEndProgress::Complete => {
                    self.finish_transaction_cancellations(transaction, request_id)?;
                    Ok(ResidentDriverStep::Executed {
                        action_kind: ResidentActionKind::Cancel,
                        rows: 0,
                        staged: 0,
                        finished: 0,
                    })
                }
            };
        }
        let mut first_error = None;
        for slot_index in 0..pending.slots.len() {
            if matches!(
                &pending.slots[slot_index].status,
                NativeProposalSlotStatus::NotStarted
            ) {
                pending.slots[slot_index].proposal_start = Some(Instant::now());
            }
            let mut bookkeeping = None;
            let mut span = None;
            let progress = match &pending.slots[slot_index].status {
                NativeProposalSlotStatus::NotStarted => {
                    let anchor_token = pending.actions[slot_index].token_id;
                    let proposal_started_ns = self.runtime_now_ns();
                    let progress = self.executor.begin_native_proposal_for(
                        &mut pending.source_states[slot_index],
                        pending.transaction,
                        anchor_token,
                    );
                    span = Some(proposal_started_ns);
                    Some(progress)
                }
                NativeProposalSlotStatus::Waiting(waiting) => {
                    let continuation = waiting.continuation();
                    if resume
                        .as_ref()
                        .is_none_or(|(ready, _)| *ready != continuation)
                    {
                        None
                    } else {
                        let (_, mut resume_lease) = resume
                            .take()
                            .expect("matching proposal resume lease exists");
                        let leases = resume_lease.take()?;
                        let resume_started = self.runtime_now_ns();
                        let progress = self.executor.resume_native_proposal_for(
                            &mut pending.source_states[slot_index],
                            pending.transaction,
                            continuation,
                            leases,
                        );
                        let disposition = if progress.is_err() {
                            ResumeDisposition::StillActive
                        } else {
                            ResumeDisposition::Consumed
                        };
                        bookkeeping = Some((continuation, resume_lease, disposition));
                        span = Some(resume_started);
                        Some(progress)
                    }
                }
                NativeProposalSlotStatus::Complete { .. } => None,
            };
            if let Some(progress) = progress {
                match progress {
                    Ok(NativeProposalProgress::Complete(proposal)) => {
                        pending.slots[slot_index].status = NativeProposalSlotStatus::Complete {
                            proposal,
                            proposal_time_us: pending.slots[slot_index]
                                .proposal_start
                                .as_ref()
                                .expect("a completed native proposal was started")
                                .elapsed()
                                .as_micros() as u64,
                            prepared: Box::new(None),
                        };
                    }
                    Ok(NativeProposalProgress::Waiting(waiting)) => {
                        pending.slots[slot_index].status =
                            NativeProposalSlotStatus::Waiting(waiting);
                    }
                    Err(error) => {
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                }
            }
            if let Some(started_ns) = span {
                pending.abort_cause = first_error.take();
                let transaction = pending.transaction;
                let next = self.handoff_speculative_progress(
                    transaction,
                    PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
                    bookkeeping,
                    started_ns,
                )?;
                let PendingSpeculativeDriverCohort::Proposing(restored) = next else {
                    unreachable!()
                };
                pending = *restored;
                self.register_proposal_progress(&mut pending);
                first_error = pending.abort_cause.take();
            }
            if let Err(error) = self.prepare_completed_native_proposal_slot(
                &mut pending.slots[slot_index],
                pending.proposal_source,
            ) && first_error.is_none()
            {
                first_error = Some(error);
            }
        }

        if let Some(error) = first_error {
            return match self.cancel_native_proposal_cohort(pending, None, Some(error)) {
                Ok(TransactionEndProgress::Pending) => Ok(ResidentDriverStep::Blocked),
                Ok(TransactionEndProgress::Complete) => Err(Error::Invariant {
                    message: "speculative proposal failure was lost during cancellation".into(),
                }),
                Err(error) => Err(error),
            };
        }

        let waiting = pending
            .slots
            .iter()
            .filter_map(|slot| match &slot.status {
                NativeProposalSlotStatus::Waiting(waiting) => Some(waiting.clone()),
                NativeProposalSlotStatus::NotStarted
                | NativeProposalSlotStatus::Complete { .. } => None,
            })
            .collect::<Vec<_>>();
        if !waiting.is_empty() {
            let transaction = pending.transaction;
            self.speculative_transactions.insert(
                transaction,
                PendingSpeculativeDriverCohort::Proposing(Box::new(pending)),
            );
            return Ok(ResidentDriverStep::WaitingForModelProgress(
                self.pending_model_progresses(),
            ));
        }

        let PendingNativeProposalCohort {
            transaction,
            cohort_start,
            actions,
            source_states,
            schedules,
            slots,
            ..
        } = pending;
        let mut admission = SpeculativeAdmission {
            transaction,
            cohort_start,
            actions,
            prepared: Vec::new(),
            sessions: AdmissionSessions {
                schedules,
                states: source_states,
            },
        };
        #[cfg(test)]
        if let Err(error) = self.fail_stage("spec admission proposals") {
            return self.progress_speculative_admission_rollback(
                admission.rollback(error, "production speculative proposal completion"),
            );
        }
        admission.prepared = match Self::take_prepared_speculative_actions(slots) {
            Ok(prepared) => prepared,
            Err(error) => {
                return self.progress_speculative_admission_rollback(
                    admission.rollback(error, "production speculative proposal completion"),
                );
            }
        };
        self.begin_speculative_verification_cohort(admission, on_token)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn handoff_verification_progress<F>(
        &mut self,
        transaction: ExecutionTransactionId,
        cohort_start: Instant,
        started_ns: u64,
        actions: Vec<DecodeAction>,
        prepared: Vec<PreparedSpeculativeAction>,
        source_states: Vec<R::SequenceState>,
        schedules: Vec<SuspendedSequenceSchedule>,
        progress: std::result::Result<
            SpeculativeCohortProgress<R::SequenceState>,
            SpeculativeCohortFailure<R::SequenceState>,
        >,
        request_id: Option<RequestId>,
        finish: Option<(ContinuationId, crate::io::ResumeLease, ResumeDisposition)>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let continuation = finish.as_ref().map(|(id, _, _)| *id);
        let ending = match progress {
            Ok(SpeculativeCohortProgress::Waiting(verification)) => {
                let next = PendingSpeculativeDriverCohort::Verifying(Box::new(
                    PendingSpeculativeVerificationDriverCohort {
                        transaction,
                        cohort_start,
                        actions,
                        prepared,
                        source_states,
                        schedules,
                        verification: *verification,
                        cancellation_request: request_id,
                    },
                ));
                let next =
                    self.handoff_speculative_progress(transaction, next, finish, started_ns)?;
                return self.accept_speculative_progress(next, on_token);
            }
            Ok(SpeculativeCohortProgress::Ready(prepared_cohort)) => {
                SpeculativeEnding::Publish(*prepared_cohort)
            }
            Err(SpeculativeCohortFailure::Active {
                error,
                transaction: backend,
            }) => SpeculativeEnding::Abort {
                transaction: *backend,
                failure: Some((error, "production speculative verification")),
            },
            Err(SpeculativeCohortFailure::Quiesced { error, cleanup }) => {
                SpeculativeEnding::BackendAbortedPendingCleanup {
                    progress: *cleanup,
                    retirements: VecDeque::new(),
                    custody: Some(if request_id.is_some() {
                        TransactionCustodyOutcome::Cancelled
                    } else {
                        TransactionCustodyOutcome::RolledBack
                    }),
                    failure: Some((error, "production speculative verification")),
                    decode_requeued: false,
                }
            }
        };
        let continuation = matches!(ending, SpeculativeEnding::Abort { .. })
            .then_some(continuation)
            .flatten();
        let next = PendingSpeculativeDriverCohort::Ending(Box::new(
            PendingSpeculativeEndingDriverCohort {
                transaction,
                cohort_start,
                started_ns,
                actions,
                prepared,
                source_states,
                schedules,
                ending,
                request_id,
                continuation,
            },
        ));
        let next = self.handoff_speculative_progress(transaction, next, finish, started_ns)?;
        self.accept_speculative_progress(next, on_token)
    }

    fn begin_speculative_verification_cohort<F>(
        &mut self,
        admission: SpeculativeAdmission<R::SequenceState>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transaction = admission.transaction;
        #[cfg(test)]
        if let Err(error) = self.fail_stage("spec admission generation") {
            return self.progress_speculative_admission_rollback(
                admission.rollback(error, "production speculative KV generation"),
            );
        }
        let verification_items = {
            let page_manager = self
                .page_manager
                .as_ref()
                .expect("speculative verification requires a page manager");
            admission
                .actions
                .iter()
                .zip(&admission.prepared)
                .map(|(action, prepared)| {
                    Ok(SpeculativeVerificationItem {
                        state_slot: prepared.page_slot,
                        generation: page_manager.sequence_generation(prepared.page_slot)?,
                        proposal: &prepared.proposal,
                        frontier: TargetFrontier {
                            position: action.position,
                            top1: ferrule_common::execution::TokenLogit::new(
                                action.token_id,
                                action.logit.unwrap_or(0.0),
                            ),
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()
        };
        let verification_items = match verification_items {
            Ok(items) => items,
            Err(error) => {
                return self.progress_speculative_admission_rollback(
                    admission.rollback(error, "production speculative KV generation"),
                );
            }
        };
        #[cfg(test)]
        if let Err(error) = self.fail_stage("spec admission reserve") {
            drop(verification_items);
            return self.progress_speculative_admission_rollback(
                admission.rollback(error, "production speculative KV reserve"),
            );
        }
        let reservations = self.reserve_speculative_pages(transaction, &verification_items);
        let reservations = match reservations {
            Ok(reservations) => reservations,
            Err(error) => {
                drop(verification_items);
                return self.progress_speculative_admission_rollback(
                    admission.rollback(error, "production speculative KV reserve"),
                );
            }
        };
        let verification_started_ns = self.runtime_now_ns();
        let verification = match self.page_manager.as_mut() {
            Some(page_manager) => prepare_speculative_verification_transaction(
                &mut self.executor,
                page_manager,
                transaction,
                &admission.sessions.states,
                &verification_items,
                reservations,
                self.top_k,
            ),
            None => unreachable!("speculative page reservations require a page manager"),
        };
        drop(verification_items);
        let SpeculativeAdmission {
            transaction,
            cohort_start,
            actions,
            prepared,
            sessions,
        } = admission;
        let (schedules, source_states) = sessions.commit();
        let verification = match verification {
            Ok(verification) => verification,
            Err(SpeculativeCohortFailure::Quiesced { error, cleanup }) => {
                return self.drive_quiesced_speculative_failure(
                    transaction,
                    cohort_start,
                    verification_started_ns,
                    actions,
                    prepared,
                    source_states,
                    schedules,
                    *cleanup,
                    Vec::new(),
                    error,
                    "production speculative verification preparation",
                    None,
                    None,
                    on_token,
                );
            }
            Err(SpeculativeCohortFailure::Active { .. }) => {
                unreachable!("verification preparation cannot activate backend ownership")
            }
        };
        let demand =
            crate::scheduling::ResourceDemand::required(ExecutionPhase::SpeculativeVerification);
        if let Err(error) =
            self.declare_transaction_prefetch(transaction, demand, verification.batch())
        {
            return self.drive_quiesced_speculative_failure(
                transaction,
                cohort_start,
                verification_started_ns,
                actions,
                prepared,
                source_states,
                schedules,
                QuiescedSpeculativeAbortProgress::from_transaction(verification),
                Vec::new(),
                error,
                "speculative verification prefetch",
                None,
                None,
                on_token,
            );
        }
        let progress = match self.page_manager.as_mut() {
            Some(page_manager) => begin_prepared_speculative_verification(
                &mut self.executor,
                page_manager,
                &source_states,
                verification,
            ),
            None => unreachable!("speculative page reservations require a page manager"),
        };
        self.handoff_verification_progress(
            transaction,
            cohort_start,
            verification_started_ns,
            actions,
            prepared,
            source_states,
            schedules,
            progress,
            None,
            None,
            on_token,
        )
    }

    pub(super) fn resume_speculative_transaction<F>(
        &mut self,
        transaction: ExecutionTransactionId,
        ready_continuation: ContinuationId,
        on_token: &mut F,
    ) -> Result<Option<ResidentDriverStep>>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let pending = self
            .speculative_transactions
            .remove(&transaction)
            .ok_or_else(|| Error::Invariant {
                message: format!("speculative transaction {transaction:?} disappeared"),
            })?;
        match pending {
            PendingSpeculativeDriverCohort::AdmissionRollback(pending) => {
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::AdmissionRollback(pending),
                );
                Ok(None)
            }
            PendingSpeculativeDriverCohort::Bookkeeping(pending) => {
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::Bookkeeping(pending),
                );
                self.drive_speculative_bookkeeping(transaction, on_token)
                    .map(Some)
            }
            PendingSpeculativeDriverCohort::Proposing(pending) => {
                let lease = match self.prepare_resume_lease(ready_continuation) {
                    Ok(lease) => lease,
                    Err(error) => {
                        self.speculative_transactions.insert(
                            transaction,
                            PendingSpeculativeDriverCohort::Proposing(pending),
                        );
                        return Err(error);
                    }
                };
                let step = self.advance_native_proposal_cohort(
                    *pending,
                    on_token,
                    Some((ready_continuation, lease)),
                )?;
                Ok(
                    (!matches!(step, ResidentDriverStep::WaitingForModelProgress(_)))
                        .then_some(step),
                )
            }
            PendingSpeculativeDriverCohort::Verifying(pending)
                if pending.cancellation_request.is_some() =>
            {
                let request_id = pending
                    .cancellation_request
                    .expect("guarded speculative cancellation request");
                match self.cancel_speculative_verification_cohort(*pending, request_id) {
                    Ok(TransactionEndProgress::Pending) => Ok(None),
                    Ok(TransactionEndProgress::Complete) => {
                        Ok(Some(ResidentDriverStep::Executed {
                            action_kind: ResidentActionKind::Cancel,
                            rows: 0,
                            staged: 0,
                            finished: 0,
                        }))
                    }
                    Err(error) => Err(error),
                }
            }
            PendingSpeculativeDriverCohort::Verifying(pending) => self
                .resume_speculative_verification_transaction(
                    *pending,
                    ready_continuation,
                    on_token,
                ),
            PendingSpeculativeDriverCohort::Ending(pending) => {
                self.speculative_transactions
                    .insert(transaction, PendingSpeculativeDriverCohort::Ending(pending));
                Ok(None)
            }
        }
    }

    pub(super) fn resume_speculative_verification_transaction<F>(
        &mut self,
        pending: PendingSpeculativeVerificationDriverCohort<R::SequenceState>,
        ready_continuation: ContinuationId,
        on_token: &mut F,
    ) -> Result<Option<ResidentDriverStep>>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let PendingSpeculativeVerificationDriverCohort {
            transaction,
            cohort_start,
            actions,
            prepared,
            source_states,
            schedules,
            verification,
            cancellation_request,
        } = pending;
        if self.page_manager.is_none() {
            self.speculative_transactions.insert(
                transaction,
                PendingSpeculativeDriverCohort::Verifying(Box::new(
                    PendingSpeculativeVerificationDriverCohort {
                        transaction,
                        cohort_start,
                        actions,
                        prepared,
                        source_states,
                        schedules,
                        verification,
                        cancellation_request,
                    },
                )),
            );
            self.enqueue_transaction(transaction);
            return Err(Error::Invariant { message:
                "authoritative KvPageManager disappeared while speculative verification was suspended"
                    .into(),
             });
        }
        if verification.pending_progress().continuation() != ready_continuation {
            self.speculative_transactions.insert(
                transaction,
                PendingSpeculativeDriverCohort::Verifying(Box::new(
                    PendingSpeculativeVerificationDriverCohort {
                        transaction,
                        cohort_start,
                        actions,
                        prepared,
                        source_states,
                        schedules,
                        verification,
                        cancellation_request,
                    },
                )),
            );
            return Err(Error::InvalidRequest {
                message: format!(
                    "speculative verification transaction {} owns continuation {}, not ready continuation {}",
                    transaction.get(),
                    self.speculative_transactions
                        .get(&transaction)
                        .and_then(|pending| match pending {
                            PendingSpeculativeDriverCohort::Verifying(pending) => {
                                Some(pending.verification.pending_progress().continuation().get())
                            }
                            PendingSpeculativeDriverCohort::AdmissionRollback(_)
                            | PendingSpeculativeDriverCohort::Bookkeeping(_)
                            | PendingSpeculativeDriverCohort::Proposing(_)
                            | PendingSpeculativeDriverCohort::Ending(_) => None,
                        })
                        .unwrap_or_default(),
                    ready_continuation.get()
                ),
            });
        }
        let mut resume_lease = match self.prepare_resume_lease(ready_continuation) {
            Ok(lease) => lease,
            Err(error) => {
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::Verifying(Box::new(
                        PendingSpeculativeVerificationDriverCohort {
                            transaction,
                            cohort_start,
                            actions,
                            prepared,
                            source_states,
                            schedules,
                            verification,
                            cancellation_request,
                        },
                    )),
                );
                return Err(error);
            }
        };
        let leases = match resume_lease.take() {
            Ok(leases) => leases,
            Err(error) => {
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::Verifying(Box::new(
                        PendingSpeculativeVerificationDriverCohort {
                            transaction,
                            cohort_start,
                            actions,
                            prepared,
                            source_states,
                            schedules,
                            verification,
                            cancellation_request,
                        },
                    )),
                );
                return Err(error.into());
            }
        };
        let resume_started = self.runtime_now_ns();
        let progress = resume_resumable_speculative_verification_cohort(
            &mut self.executor,
            self.page_manager
                .as_mut()
                .expect("page manager was checked above"),
            &source_states,
            verification,
            leases,
        );
        let disposition = if matches!(progress, Err(SpeculativeCohortFailure::Active { .. })) {
            ResumeDisposition::StillActive
        } else {
            ResumeDisposition::Consumed
        };
        self.handoff_verification_progress(
            transaction,
            cohort_start,
            resume_started,
            actions,
            prepared,
            source_states,
            schedules,
            progress,
            cancellation_request,
            Some((ready_continuation, resume_lease, disposition)),
            on_token,
        )
        .map(|step| {
            (!matches!(step, ResidentDriverStep::WaitingForModelProgress(_))).then_some(step)
        })
    }

    pub(super) fn prepare_native_proposal_slots(
        &self,
        actions: &[DecodeAction],
    ) -> Result<Vec<PendingNativeProposalSlot>> {
        let mut slots = Vec::with_capacity(actions.len());

        for action in actions {
            let sequence = self
                .scheduler
                .active_sequence(action.session_id)
                .cloned()
                .ok_or_else(|| Error::Invariant {
                    message: format!(
                        "cannot execute speculative for inactive session {:?}",
                        action.session_id
                    ),
                })?;
            if sequence.request_id != action.request_id
                || sequence.kv_handle != action.kv_handle
                || sequence.position != action.position
                || sequence.next_decode_token != Some(action.token_id)
            {
                return Err(Error::Invariant {
                    message: format!(
                        "speculative action no longer matches session {:?}: action(request={:?}, kv={:?}, position={}, token={}), sequence(request={:?}, kv={:?}, position={}, token={:?})",
                        action.session_id,
                        action.request_id,
                        action.kv_handle,
                        action.position,
                        action.token_id,
                        sequence.request_id,
                        sequence.kv_handle,
                        sequence.position,
                        sequence.next_decode_token,
                    ),
                });
            }

            let remaining_output = sequence.max_new_tokens.saturating_sub(sequence.generated);
            let remaining_context = self.config.ctx_size.saturating_sub(sequence.position);
            let commit_capacity = remaining_output.min(remaining_context);
            if commit_capacity == 0 {
                return Err(Error::Invariant {
                    message: format!(
                        "speculative decode action for session {:?} has no output/context capacity",
                        action.session_id
                    ),
                });
            }
            let max_drafts = commit_capacity.saturating_sub(1);
            let page_slot =
                *self
                    .sessions
                    .page_slot(&action.session_id)
                    .ok_or_else(|| Error::Invariant {
                        message: format!(
                            "speculative session {:?} has no authoritative page slot",
                            action.session_id
                        ),
                    })?;

            let status = if max_drafts == 0 {
                NativeProposalSlotStatus::Complete {
                    proposal: NativeProposal {
                        token_ids: Vec::new(),
                        confidence_logits: Vec::new(),
                    },
                    proposal_time_us: 0,
                    prepared: Box::new(None),
                }
            } else {
                NativeProposalSlotStatus::NotStarted
            };
            slots.push(PendingNativeProposalSlot {
                sequence,
                page_slot,
                max_drafts,
                proposal_start: None,
                status,
            });
        }

        Ok(slots)
    }

    pub(super) fn prepare_completed_native_proposal_slot(
        &self,
        slot: &mut PendingNativeProposalSlot,
        proposal_source: NativeProposalSource,
    ) -> Result<()> {
        let NativeProposalSlotStatus::Complete {
            proposal,
            proposal_time_us,
            prepared,
        } = &mut slot.status
        else {
            return Ok(());
        };
        let prepared = prepared.as_mut();
        if prepared.is_some() {
            return Ok(());
        }

        if slot.max_drafts != 0 {
            proposal.validate_for_source(proposal_source)?;
        } else {
            proposal.validate()?;
        }
        let anchor_token_id = slot
            .sequence
            .next_decode_token
            .ok_or_else(|| Error::Invariant {
                message: format!(
                    "speculative sequence {:?} lost its validated anchor token",
                    slot.sequence.session_id
                ),
            })?;
        let mut proposal_tokens = std::mem::take(&mut proposal.token_ids);
        let mut confidence_logits = std::mem::take(&mut proposal.confidence_logits);
        let capacity_width = proposal_tokens.len().min(slot.max_drafts);
        proposal_tokens.truncate(capacity_width);
        confidence_logits.truncate(capacity_width);
        proposal_tokens = self.truncate_native_proposal_at_output_boundary(
            &slot.sequence,
            anchor_token_id,
            proposal_tokens,
        )?;
        confidence_logits.truncate(proposal_tokens.len());
        let confidence_width = confident_proposal_prefix_length(
            &confidence_logits,
            self.config.proposal_confidence_threshold,
        )?;
        proposal_tokens.truncate(confidence_width);
        *prepared = Some(PreparedSpeculativeAction {
            sequence: slot.sequence.clone(),
            page_slot: slot.page_slot,
            proposal: proposal_tokens,
            proposal_time_us: *proposal_time_us,
        });
        Ok(())
    }

    pub(super) fn take_prepared_speculative_actions(
        slots: Vec<PendingNativeProposalSlot>,
    ) -> Result<Vec<PreparedSpeculativeAction>> {
        let mut prepared_actions = Vec::with_capacity(slots.len());
        for slot in slots {
            match slot.status {
                NativeProposalSlotStatus::Complete { prepared, .. } => match *prepared {
                    Some(prepared) => prepared_actions.push(prepared),
                    None => {
                        return Err(Error::Invariant {
                            message:
                                "speculative proposal cohort completed with an unfinished slot"
                                    .into(),
                        });
                    }
                },
                NativeProposalSlotStatus::NotStarted | NativeProposalSlotStatus::Waiting(_) => {
                    return Err(Error::Invariant {
                        message: "speculative proposal cohort completed with an unfinished slot"
                            .into(),
                    });
                }
            }
        }
        Ok(prepared_actions)
    }

    pub(super) fn abort_speculative_decode_batch(
        &mut self,
        actions: &[DecodeAction],
        error: Error,
        stage: &'static str,
    ) -> Error {
        let mut cleanup = Vec::new();
        let mut sessions = actions
            .iter()
            .map(|action| action.session_id)
            .collect::<Vec<_>>();
        sessions.sort_unstable_by_key(|session| session.0);
        sessions.dedup();
        for session_id in sessions {
            if self.scheduler.active_sequence(session_id).is_some()
                && let Err(source) = self
                    .scheduler
                    .fail_sequence(session_id, &mut self.slot_pool)
            {
                cleanup.push(CleanupStep::new(
                    format!("session {session_id:?} scheduler cleanup"),
                    source,
                ));
            }
            self.sessions.unretain(&session_id);
            if let Err(source) = self.release_sequence_state(session_id) {
                cleanup.push(CleanupStep::new(
                    format!("session {session_id:?} state cleanup"),
                    source,
                ));
            }
        }
        Error::with_cleanup_batch(stage, error, cleanup)
    }
}
