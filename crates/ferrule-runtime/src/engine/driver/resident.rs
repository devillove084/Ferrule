//! Resident operations on the driver's existing authority.

use super::*;

pub(super) struct PendingResidentBatch<S> {
    pub(super) transaction: ExecutionTransactionId,
    pub(super) action: SchedulerAction,
    pub(super) scheduled: ScheduledBatch,
    pub(super) kv: Option<PendingResidentKv>,
    pub(super) states: Vec<S>,
    pub(super) schedules: Vec<SuspendedSequenceSchedule>,
    pub(super) phase: ResidentTransactionPhase,
}

pub(super) enum ResidentTransactionPhase {
    Executing(Option<PendingModelProgress>),
    Publishing {
        output: ExecutionOutput,
        started_ns: u64,
        cancellation_request: Option<RequestId>,
    },
    Aborting {
        request: Option<RequestId>,
        custody: TransactionCustodyOutcome,
        continuation: Option<ContinuationId>,
        failure: Option<(Error, &'static str)>,
    },
    BackendCommittedPendingPublish {
        output: ExecutionOutput,
        custody: Option<TransactionCustodyOutcome>,
        cancellation_request: Option<RequestId>,
    },
    BackendAbortedPendingCleanup {
        request: Option<RequestId>,
        custody: Option<TransactionCustodyOutcome>,
        continuation: Option<ContinuationId>,
        failure: Option<(Error, &'static str)>,
        decode_requeued: bool,
    },
}

pub(super) enum ResidentCancellationRoute {
    Abort,
    Publish,
    Cleanup,
}

impl ResidentTransactionPhase {
    /// Cancellation cannot rewrite an already submitted Publish intent.
    pub(super) fn accept_cancellation(
        &mut self,
        request_id: RequestId,
    ) -> ResidentCancellationRoute {
        match self {
            Self::Publishing {
                cancellation_request,
                ..
            }
            | Self::BackendCommittedPendingPublish {
                cancellation_request,
                ..
            } => {
                cancellation_request.get_or_insert(request_id);
                ResidentCancellationRoute::Publish
            }
            Self::Aborting { request, .. } => {
                request.get_or_insert(request_id);
                ResidentCancellationRoute::Abort
            }
            Self::BackendAbortedPendingCleanup { request, .. } => {
                request.get_or_insert(request_id);
                ResidentCancellationRoute::Cleanup
            }
            Self::Executing(progress) => {
                let continuation = progress.as_ref().map(PendingModelProgress::continuation);
                *self = Self::Aborting {
                    request: Some(request_id),
                    custody: TransactionCustodyOutcome::Cancelled,
                    continuation,
                    failure: None,
                };
                ResidentCancellationRoute::Abort
            }
        }
    }

    pub(super) fn backend_completed(self, finished_ns: u64) -> Self {
        match self {
            Self::Publishing {
                output,
                started_ns,
                cancellation_request,
            } => Self::BackendCommittedPendingPublish {
                output,
                custody: Some(TransactionCustodyOutcome::Committed {
                    started_ns,
                    finished_ns,
                }),
                cancellation_request,
            },
            Self::Aborting {
                request,
                custody,
                continuation,
                failure,
            } => Self::BackendAbortedPendingCleanup {
                request,
                custody: Some(custody),
                continuation,
                failure,
                decode_requeued: false,
            },
            _ => unreachable!("only a submitted ending can acknowledge backend completion"),
        }
    }

    pub(super) fn is_post_terminal(&self) -> bool {
        matches!(
            self,
            Self::BackendCommittedPendingPublish { .. } | Self::BackendAbortedPendingCleanup { .. }
        )
    }

    pub(super) fn pending_progress(&self) -> Option<&PendingModelProgress> {
        match self {
            Self::Executing(pending) => pending.as_ref(),
            Self::Publishing { .. }
            | Self::Aborting { .. }
            | Self::BackendCommittedPendingPublish { .. }
            | Self::BackendAbortedPendingCleanup { .. } => None,
        }
    }

    pub(super) const fn is_ending(&self) -> bool {
        !matches!(self, Self::Executing(_))
    }
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner,
    C: SequenceSlotPool,
{
    pub(super) fn resume_resident_transaction<F>(
        &mut self,
        transaction: ExecutionTransactionId,
        ready_continuation: ContinuationId,
        on_token: &mut F,
    ) -> Result<Option<ResidentDriverStep>>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let mut pending = self
            .resident_transactions
            .remove(&transaction)
            .ok_or_else(|| Error::Invariant {
                message: format!("resident transaction {transaction:?} disappeared"),
            })?;
        let owned_continuation = pending
            .phase
            .pending_progress()
            .map(PendingModelProgress::continuation);
        let continuation = match owned_continuation {
            Some(continuation) if continuation == ready_continuation => continuation,
            Some(continuation) => {
                let error = Error::InvalidRequest {
                    message: format!(
                        "resident transaction {transaction:?} owns continuation {}, not ready continuation {}",
                        continuation.get(),
                        ready_continuation.get()
                    ),
                };
                self.resident_transactions.insert(transaction, pending);
                return Err(error);
            }
            None => {
                self.resident_transactions.insert(transaction, pending);
                return Err(Error::InvalidRequest {
                    message: format!("resident transaction {transaction:?} is quarantined"),
                });
            }
        };
        let mut resume_lease = match self.prepare_resume_lease(continuation) {
            Ok(lease) => lease,
            Err(error) => {
                self.resident_transactions.insert(transaction, pending);
                return Err(error);
            }
        };
        let leases = match resume_lease.take() {
            Ok(leases) => leases,
            Err(error) => {
                self.resident_transactions.insert(transaction, pending);
                return Err(error.into());
            }
        };
        let resume_started = self.runtime_now_ns();
        let progress = self.executor.resume_prepared_batch(
            transaction,
            &mut pending.states,
            pending.scheduled.execution(),
            continuation,
            leases,
        );
        let disposition = if progress.is_err() {
            ResumeDisposition::StillActive
        } else {
            ResumeDisposition::Consumed
        };
        if let Err(error) =
            self.finish_resume_lease(continuation, resume_lease, disposition, resume_started)
        {
            return self
                .abort_failed_resident(pending, error, "model resume lease completion")
                .map(Some);
        }
        if let Err(error) = self.record_runnable_work_span(resume_started) {
            return self
                .abort_failed_resident(pending, error, "model resume observation")
                .map(Some);
        }
        match progress {
            Ok(MultiSessionBatchProgress::Complete(output)) => {
                pending.phase = ResidentTransactionPhase::Executing(None);
                self.finish_resident_transaction(pending, output, on_token)
                    .map(Some)
            }
            Ok(MultiSessionBatchProgress::Waiting(progress)) => {
                pending.phase = ResidentTransactionPhase::Executing(None);
                if let Err(error) =
                    self.register_pending_progress(&progress, Self::action_demand(&pending.action))
                {
                    return self
                        .abort_failed_resident(pending, error, "model continuation registration")
                        .map(Some);
                }
                pending.phase = ResidentTransactionPhase::Executing(Some(progress));
                self.resident_transactions.insert(transaction, pending);
                Ok(None)
            }
            Err(error) => self
                .abort_failed_resident(pending, error, "resumable model execution")
                .map(Some),
        }
    }

    pub(super) fn finish_resident_transaction<F>(
        &mut self,
        mut pending: PendingResidentBatch<R::SequenceState>,
        output: ExecutionOutput,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transaction = pending.transaction;
        if let Err(error) = pending.scheduled.validate_output(&output) {
            return self.abort_failed_resident(pending, error, "model output contract");
        }

        let reserved = match pending
            .kv
            .take()
            .expect("executing resident transaction owns logical KV state")
        {
            PendingResidentKv::Reserved(reservations) => reservations,
            PendingResidentKv::Prepared(prepared) => {
                pending.kv = Some(PendingResidentKv::Prepared(prepared));
                self.resident_transactions.insert(transaction, pending);
                return Err(Error::Invariant {
                    message: format!(
                        "transaction {transaction:?} reached model completion with an existing prepared logical commit"
                    ),
                });
            }
            PendingResidentKv::Retiring(retirement) => {
                pending.kv = Some(PendingResidentKv::Retiring(retirement));
                self.resident_transactions.insert(transaction, pending);
                return Err(Error::Invariant {
                    message: format!(
                        "transaction {transaction:?} reached model completion while logical KV retirement was already active"
                    ),
                });
            }
        };
        let prepared = match self.page_manager.as_mut() {
            Some(manager) => {
                let commits = reserved
                    .into_iter()
                    .map(|reservation| {
                        let rows = reservation.view().positions.len();
                        KvReservationCommit::new(reservation, rows)
                    })
                    .collect();
                match manager.prepare_commit(commits) {
                    Ok(prepared) => Some(prepared),
                    Err(error) => {
                        let (error, commits) = error.into_parts();
                        pending.kv = Some(PendingResidentKv::Reserved(
                            commits
                                .into_iter()
                                .map(|commit| commit.reservation)
                                .collect(),
                        ));
                        return self.abort_failed_resident(pending, error, "logical KV prepare");
                    }
                }
            }
            None if reserved.is_empty() => None,
            None => {
                pending.kv = Some(PendingResidentKv::Reserved(reserved));
                return self.abort_failed_resident(
                    pending,
                    Error::Invariant {
                        message: "KV reservations exist without an authoritative page manager"
                            .into(),
                    },
                    "logical KV prepare",
                );
            }
        };
        if let Some(prepared) = prepared {
            pending.kv = Some(PendingResidentKv::Prepared(prepared));
        }

        pending.phase = ResidentTransactionPhase::Publishing {
            output,
            started_ns: self.runtime_now_ns(),
            cancellation_request: None,
        };
        self.drive_resident_ending(pending, on_token)
    }

    pub(super) fn publish_resident_transaction<F>(
        &mut self,
        mut pending: PendingResidentBatch<R::SequenceState>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transaction = pending.transaction;
        let ResidentTransactionPhase::BackendCommittedPendingPublish {
            output,
            mut custody,
            cancellation_request,
        } = std::mem::replace(
            &mut pending.phase,
            ResidentTransactionPhase::Executing(None),
        )
        else {
            unreachable!("resident publish progress requires a backend-committed phase")
        };

        if let Some(outcome) = custody
            && let Err(error) = self.load_registry.finish_transaction_custody(
                transaction,
                outcome,
                self.runtime_now_ns(),
            )
        {
            custody = Some(outcome);
            pending.phase = ResidentTransactionPhase::BackendCommittedPendingPublish {
                output,
                custody,
                cancellation_request,
            };
            return self.retain_resident_ending_error(pending, error.into());
        }

        if let Some(kv) = pending.kv.take()
            && let Err((error, kv)) = self.progress_committed_resident_kv(kv)
        {
            pending.kv = Some(kv);
            pending.phase = ResidentTransactionPhase::BackendCommittedPendingPublish {
                output,
                custody: None,
                cancellation_request,
            };
            return self.retain_resident_ending_error(pending, error);
        }

        if let Err(error) =
            self.progress_transaction_session_restore(&mut pending.schedules, &mut pending.states)
        {
            pending.phase = ResidentTransactionPhase::BackendCommittedPendingPublish {
                output,
                custody: None,
                cancellation_request,
            };
            return self.retain_resident_ending_error(pending, error);
        }

        let step = match self.publish_restored_resident_transaction(pending, output) {
            Ok(step) => step,
            Err(error) => {
                let cleanup = self
                    .finish_transaction_cancellations(transaction, cancellation_request)
                    .map(|_| ());
                return Err(Error::with_cleanup(
                    "committed resident cancellation",
                    error,
                    cleanup,
                ));
            }
        };
        self.finish_transaction_cancellations(transaction, cancellation_request)?;
        self.flush_committed_token_outbox(on_token)?;
        Ok(step)
    }

    pub(super) fn publish_restored_resident_transaction(
        &mut self,
        pending: PendingResidentBatch<R::SequenceState>,
        output: ExecutionOutput,
    ) -> Result<ResidentDriverStep> {
        let transaction = pending.transaction;
        debug_assert!(pending.kv.is_none());
        debug_assert!(pending.states.is_empty());
        debug_assert!(pending.schedules.is_empty());
        let action_kind = action_kind(&pending.action);
        let rows = action_rows(&pending.action);
        let publication = (|| -> Result<(usize, usize)> {
            self.scheduler.commit_action(&pending.action)?;
            self.capture_committed_prefill_prefixes(&pending.action)?;
            self.observability.stats.actions += 1;
            let queued_decode = match &pending.action {
                SchedulerAction::Execute { prefills, decodes } => {
                    self.observability.stats.prefill_chunks += prefills.len();
                    self.observability.stats.prefill_tokens += prefills
                        .iter()
                        .map(|action| action.token_range.len())
                        .sum::<usize>();
                    self.observability.stats.decode_steps += decodes.len();
                    self.enqueue_committed_decode_tokens(decodes)?
                }
                SchedulerAction::PrefillChunk(prefill) => {
                    self.observability.stats.prefill_chunks += 1;
                    self.observability.stats.prefill_tokens += prefill.token_range.len();
                    0
                }
                SchedulerAction::DecodeBatch(actions) => {
                    self.observability.stats.decode_steps += actions.len();
                    self.enqueue_committed_decode_tokens(actions)?
                }
                SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => 0,
            };

            let action_finish = self.finish_after_decode_action(&pending.action)?;
            let mut finished = action_finish.finished;
            let output_outcome = self.apply_execution_output(
                &pending.scheduled,
                &output,
                &action_finish.session_ids,
            )?;
            finished += output_outcome.finished;
            // Attribute delivery to the forward that selected the token, not the
            // subsequent KV append. The last append has no new output snapshot.
            self.snapshot_transaction_outputs(transaction, queued_decode + output_outcome.queued)?;
            Ok((output_outcome.staged, finished))
        })();
        let (staged, finished) = match publication {
            Ok(outcome) => outcome,
            Err(error) => {
                return Err(self.abort_action(
                    &pending.action,
                    error,
                    false,
                    "committed resident publication",
                ));
            }
        };
        Ok(ResidentDriverStep::Executed {
            action_kind,
            rows,
            staged,
            finished,
        })
    }

    pub(super) fn drive_resident_ending<F>(
        &mut self,
        mut pending: PendingResidentBatch<R::SequenceState>,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let transaction = pending.transaction;
        let intent = match &pending.phase {
            ResidentTransactionPhase::Publishing { .. } => Some(TransactionEndIntent::Publish),
            ResidentTransactionPhase::Aborting { .. } => Some(TransactionEndIntent::Abort),
            ResidentTransactionPhase::BackendCommittedPendingPublish { .. } => {
                return self.publish_resident_transaction(pending, on_token);
            }
            ResidentTransactionPhase::BackendAbortedPendingCleanup { .. } => {
                return self
                    .finish_resident_abort(pending, None)
                    .map(|(step, _)| step);
            }
            ResidentTransactionPhase::Executing(_) => {
                self.resident_transactions.insert(transaction, pending);
                return Err(Error::Invariant {
                    message: format!("transaction {transaction:?} is not ending"),
                });
            }
        }
        .expect("backend ending phase has an end intent");
        match self
            .executor
            .end_transaction(transaction, &mut pending.states, intent)
        {
            Ok(TransactionEndProgress::Pending) => {
                self.resident_transactions.insert(transaction, pending);
                Ok(ResidentDriverStep::Blocked)
            }
            Err(error) => {
                self.resident_transactions.insert(transaction, pending);
                Err(error)
            }
            Ok(TransactionEndProgress::Complete) => {
                let phase = std::mem::replace(
                    &mut pending.phase,
                    ResidentTransactionPhase::Executing(None),
                );
                pending.phase = phase.backend_completed(self.runtime_now_ns());
                // The acknowledged phase has no backend submission route.
                self.drive_resident_ending(pending, on_token)
            }
        }
    }

    pub(super) fn finish_resident_abort(
        &mut self,
        mut pending: PendingResidentBatch<R::SequenceState>,
        requested: Option<RequestId>,
    ) -> Result<(ResidentDriverStep, Option<CancelRequestResult>)> {
        let transaction = pending.transaction;
        let ResidentTransactionPhase::BackendAbortedPendingCleanup {
            request,
            mut custody,
            mut continuation,
            failure,
            mut decode_requeued,
        } = std::mem::replace(
            &mut pending.phase,
            ResidentTransactionPhase::Executing(None),
        )
        else {
            unreachable!("resident abort progress requires a backend-aborted phase")
        };

        if let Some(outcome) = custody
            && let Err(error) = self.load_registry.finish_transaction_custody(
                transaction,
                outcome,
                self.runtime_now_ns(),
            )
        {
            custody = Some(outcome);
            pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
                request,
                custody,
                continuation,
                failure,
                decode_requeued,
            };
            return self.retain_resident_ending_error(pending, error.into());
        }

        if let Some(continuation_id) = continuation {
            if let Err(error) = self.detach_registered_continuation(
                continuation_id,
                CancellationReason::ExternalRequest,
            ) {
                continuation = Some(continuation_id);
                pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
                    request,
                    custody: None,
                    continuation,
                    failure,
                    decode_requeued,
                };
                return self.retain_resident_ending_error(pending, error);
            }
            continuation = None;
        }

        if let Some(kv) = pending.kv.take()
            && let Err((error, kv)) = self.progress_aborted_resident_kv(kv)
        {
            pending.kv = Some(kv);
            pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
                request,
                custody: None,
                continuation,
                failure,
                decode_requeued,
            };
            return self.retain_resident_ending_error(pending, error);
        }

        if let Err(error) =
            self.progress_transaction_session_restore(&mut pending.schedules, &mut pending.states)
        {
            pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
                request,
                custody: None,
                continuation,
                failure,
                decode_requeued,
            };
            return self.retain_resident_ending_error(pending, error);
        }

        if let Some((error, stage)) = failure {
            let error = self.abort_action(&pending.action, error, false, stage);
            let cleanup = self
                .finish_transaction_cancellations(transaction, request)
                .map(|_| ());
            return Err(Error::with_cleanup(
                "resident abort cancellation",
                error,
                cleanup,
            ));
        }
        if !decode_requeued {
            if let Err(error) = self
                .scheduler
                .requeue_decode_actions_front(action_decode_actions(&pending.action))
            {
                pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
                    request,
                    custody: None,
                    continuation: None,
                    failure: None,
                    decode_requeued,
                };
                return self.retain_resident_ending_error(pending, error);
            }
            decode_requeued = true;
        }
        let cancellation =
            match self.finish_transaction_cancellations(transaction, requested.or(request)) {
                Ok(result) => result,
                Err(error) => {
                    pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
                        request,
                        custody: None,
                        continuation: None,
                        failure: None,
                        decode_requeued,
                    };
                    return self
                        .retain_resident_ending_error(pending, error)
                        .map(|step| (step, None));
                }
            };
        Ok((
            ResidentDriverStep::Executed {
                action_kind: ResidentActionKind::Cancel,
                rows: 0,
                staged: 0,
                finished: 0,
            },
            cancellation,
        ))
    }

    pub(super) fn retain_resident_ending_error<T>(
        &mut self,
        pending: PendingResidentBatch<R::SequenceState>,
        error: Error,
    ) -> Result<T> {
        self.resident_transactions
            .insert(pending.transaction, pending);
        Err(error)
    }

    pub(super) fn abort_failed_resident(
        &mut self,
        mut pending: PendingResidentBatch<R::SequenceState>,
        error: Error,
        stage: &'static str,
    ) -> Result<ResidentDriverStep> {
        if let ResidentTransactionPhase::Aborting { failure, .. } = &mut pending.phase {
            *failure = Some(match failure.take() {
                Some((previous, previous_stage)) => (
                    Error::combine("resident transaction abort", previous, error),
                    previous_stage,
                ),
                None => (error, stage),
            });
        } else {
            let continuation = pending
                .phase
                .pending_progress()
                .map(PendingModelProgress::continuation);
            pending.phase = ResidentTransactionPhase::Aborting {
                request: None,
                custody: TransactionCustodyOutcome::RolledBack,
                continuation,
                failure: Some((error, stage)),
            };
        }
        self.drive_resident_ending(pending, &mut |_| Ok(()))
    }

    pub(super) fn abort_action(
        &mut self,
        action: &SchedulerAction,
        error: Error,
        poison_executor: bool,
        stage: &'static str,
    ) -> Error {
        let _ = poison_executor;
        let session_ids = action_session_ids(action);
        let mut cleanup = Vec::new();
        if let Err(source) = self.scheduler.fail_action(action, &mut self.slot_pool) {
            cleanup.push(CleanupStep::new("scheduler action cleanup", source));
        }
        for session_id in &session_ids {
            self.sessions.unretain(session_id);
            if let Err(source) = self.release_sequence_state(*session_id) {
                cleanup.push(CleanupStep::new(
                    format!("session {session_id:?} state cleanup"),
                    source,
                ));
            }
        }
        Error::with_cleanup_batch(stage, error, cleanup)
    }
}
