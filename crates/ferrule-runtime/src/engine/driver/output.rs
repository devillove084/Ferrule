//! Output operations on the driver's existing authority.

use super::*;

/// Delivery custody, not execution custody: early events never replace staged
/// decodes. Only callback ACK consumes the FIFO; cancellation removes only the
/// still-uncommitted early event and preserves other committed output.
#[derive(Default)]
pub(super) struct OutputArbiter {
    cancellations: HashMap<RequestId, PendingRequestCancellation>,
    outbox: VecDeque<ResidentTokenEvent>,
    early: HashMap<SessionId, ResidentTokenEvent>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PendingRequestCancellation {
    pub(super) session_id: SessionId,
    pub(super) transaction: Option<ExecutionTransactionId>,
    pub(super) ready: bool,
    pub(super) result: Option<CancelRequestResult>,
}

/// The only terminal choice allowed after a committed forward. Accepted
/// cancellation is a driver-owned linearization point; it cannot be replaced
/// by a natural finish reason, while physical Publish remains irrevocable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TerminalDecision {
    Continue,
    DeferForCancellation,
    Finish(SequenceFinishReason),
}

pub(super) fn arbitrate_terminal(
    cancellation_accepted: bool,
    natural_reason: Option<SequenceFinishReason>,
) -> TerminalDecision {
    if cancellation_accepted {
        TerminalDecision::DeferForCancellation
    } else {
        natural_reason.map_or(TerminalDecision::Continue, TerminalDecision::Finish)
    }
}

impl OutputArbiter {
    pub(super) fn cancellations_empty(&self) -> bool {
        self.cancellations.is_empty()
    }
    pub(super) fn cancellation_count(&self) -> usize {
        self.cancellations.len()
    }
    pub(super) fn cancellation_accepted(&self, request: Option<RequestId>) -> bool {
        request.is_some_and(|id| self.cancellations.contains_key(&id))
    }
    pub(super) fn cancellation_owns_session(&self, session: SessionId) -> bool {
        self.cancellations
            .values()
            .any(|pending| pending.session_id == session)
    }
    pub(super) fn cancellation(&self, request: RequestId) -> Option<PendingRequestCancellation> {
        self.cancellations.get(&request).copied()
    }
    pub(super) fn take_cancellation(
        &mut self,
        request: RequestId,
    ) -> Option<PendingRequestCancellation> {
        self.cancellations.remove(&request)
    }
    pub(super) fn retry_cancellation(
        &mut self,
        request: RequestId,
        pending: PendingRequestCancellation,
    ) {
        let previous = self.cancellations.insert(request, pending);
        debug_assert!(
            previous.is_none(),
            "cancellation retry has exclusive custody"
        );
    }
    pub(super) fn cancellation_requests(&self) -> Vec<RequestId> {
        self.cancellations.keys().copied().collect()
    }
    pub(super) fn accept_scheduled_cancellation(
        &mut self,
        request: RequestId,
        session_id: SessionId,
    ) {
        self.cancellations
            .entry(request)
            .or_insert(PendingRequestCancellation {
                session_id,
                transaction: None,
                ready: true,
                result: None,
            });
    }
    pub(super) fn accept_transaction_cancellation(
        &mut self,
        transaction: ExecutionTransactionId,
        request: RequestId,
        session_id: SessionId,
    ) -> Result<()> {
        if let Some(existing) = self.cancellations.get(&request) {
            if existing.session_id != session_id || existing.transaction != Some(transaction) {
                return Err(Error::Invariant {
                    message: format!(
                        "request {request:?} cancellation owner changed from session {:?} transaction {:?} to session {session_id:?} transaction {transaction:?}",
                        existing.session_id, existing.transaction
                    ),
                });
            }
            return Ok(());
        }
        self.cancellations.insert(
            request,
            PendingRequestCancellation {
                session_id,
                transaction: Some(transaction),
                ready: false,
                result: None,
            },
        );
        Ok(())
    }
    /// Snapshot before mutation, as with the driver's ready-transaction pass.
    pub(super) fn release_transaction_cancellations(
        &mut self,
        transaction: ExecutionTransactionId,
    ) -> Vec<RequestId> {
        let requests = self
            .cancellations
            .iter()
            .filter_map(|(request, pending)| {
                (pending.transaction == Some(transaction)).then_some(*request)
            })
            .collect();
        for pending in self.cancellations.values_mut() {
            if pending.transaction == Some(transaction) {
                pending.transaction = None;
                pending.ready = true;
            }
        }
        requests
    }

    fn finish_success<C: SequenceSlotPool>(
        &self,
        scheduler: &mut ResidentScheduler,
        slots: &mut C,
        session: SessionId,
        reason: SequenceFinishReason,
    ) -> Result<Option<usize>> {
        let Some(sequence) = scheduler.active_sequence(session) else {
            return Ok(None);
        };
        if self.cancellation_accepted(sequence.request_id) {
            return Ok(None);
        }
        let position = sequence.position;
        scheduler.finish_sequence(session, reason, slots)?;
        Ok(Some(position))
    }

    pub(super) fn outbox_empty(&self) -> bool {
        self.outbox.is_empty()
    }
    pub(super) fn early_empty(&self) -> bool {
        self.early.is_empty()
    }
    pub(super) fn has_early(&self, session: &SessionId) -> bool {
        self.early.contains_key(session)
    }
    pub(super) fn owns_session(&self, session: SessionId) -> bool {
        self.early.contains_key(&session)
            || self.outbox.iter().any(|event| event.session_id == session)
    }
    #[cfg(test)]
    pub(super) fn queued_count(&self) -> usize {
        self.outbox.len()
    }
    #[cfg(test)]
    pub(super) fn early_count(&self) -> usize {
        self.early.len()
    }
    #[cfg(test)]
    pub(super) fn front(&self) -> Option<&ResidentTokenEvent> {
        self.outbox.front()
    }

    fn flush<F>(
        &mut self,
        on_token: &mut F,
        emitted_tokens: &mut usize,
        #[cfg(test)] trace: &mut Vec<DriverTickEvent>,
    ) -> Result<()>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        // Err means not accepted: keep the identical event at the FIFO head.
        while let Some(event) = self.outbox.front() {
            on_token(event)?;
            #[cfg(test)]
            trace.push(DriverTickEvent::TokenAcknowledged(
                event.session_id,
                event.token,
            ));
            self.outbox.pop_front();
            *emitted_tokens += 1;
        }
        Ok(())
    }

    fn enqueue_committed(&mut self, event: ResidentTokenEvent) {
        self.outbox.push_back(event);
    }

    fn reconcile_decode(&mut self, event: ResidentTokenEvent) -> Result<usize> {
        if let Some(early) = self.early.remove(&event.session_id) {
            if early != event {
                return Err(Error::Invariant {
                    message: "early token differs from committed decode token".into(),
                });
            }
            Ok(0)
        } else {
            self.outbox.push_back(event);
            Ok(1)
        }
    }

    fn enqueue_early(&mut self, event: ResidentTokenEvent) -> Result<()> {
        if self.early.insert(event.session_id, event.clone()).is_some() {
            return Err(Error::Invariant {
                message: "early token still pending at next selection".into(),
            });
        }
        self.outbox.push_back(event);
        Ok(())
    }

    pub(super) fn discard_uncommitted(&mut self, session: SessionId) {
        if let Some(event) = self.early.remove(&session) {
            self.outbox.retain(|queued| queued != &event);
        }
    }
}

pub(super) fn matched_stop(text: &str, stop: &[String]) -> bool {
    stop.iter()
        .any(|candidate| !candidate.is_empty() && text.ends_with(candidate))
}

#[derive(Default)]
pub(super) struct ActionFinishOutcome {
    pub(super) finished: usize,
    pub(super) session_ids: Vec<SessionId>,
}

#[derive(Default)]
pub(super) struct OutputOutcome {
    pub(super) staged: usize,
    pub(super) finished: usize,
    pub(super) queued: usize,
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner,
    C: SequenceSlotPool,
{
    pub fn take_request_terminal(
        &mut self,
        request_id: RequestId,
    ) -> Option<crate::scheduling::RequestTerminal> {
        let terminal = self.scheduler.take_request_terminal(request_id)?;
        if let Some(record) = self.request_identities.get_mut(&request_id) {
            record.terminal_consumed = true;
        }
        self.reap_request_identities();
        Some(terminal)
    }

    pub fn drain_finished(&mut self) -> Vec<SequenceState> {
        let sequences = self.scheduler.drain_finished();
        for sequence in &sequences {
            self.consume_terminal_identity(sequence);
        }
        self.reap_request_identities();
        sequences
    }

    pub fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        let sequences = self.scheduler.drain_cancelled();
        for sequence in &sequences {
            self.consume_terminal_identity(sequence);
        }
        self.reap_request_identities();
        sequences
    }

    pub fn drain_failed(&mut self) -> Vec<SequenceState> {
        let sequences = self.scheduler.drain_failed();
        for sequence in &sequences {
            self.consume_terminal_identity(sequence);
        }
        self.reap_request_identities();
        sequences
    }

    pub(super) fn flush_committed_token_outbox<F>(&mut self, on_token: &mut F) -> Result<()>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        #[cfg(test)]
        self.tick_trace.push(DriverTickEvent::Outbox);
        self.output.flush(
            on_token,
            &mut self.observability.stats.emitted_tokens,
            #[cfg(test)]
            &mut self.tick_trace,
        )
    }

    pub(super) fn enqueue_committed_decode_tokens(
        &mut self,
        actions: &[DecodeAction],
    ) -> Result<usize> {
        let mut queued = 0;
        for action in actions {
            let runner = self.executor.runner();
            let sequence = self
                .scheduler
                .active_sequence_mut(action.session_id)
                .ok_or_else(|| Error::Invariant {
                    message: format!(
                        "cannot emit token for inactive session {:?}",
                        action.session_id
                    ),
                })?;
            let text = runner
                .decode_incremental(action.token_id, &mut sequence.incremental_decode)?
                .unwrap_or_default();
            sequence.append_generated_text(&text);
            let index = sequence.generated.saturating_sub(1);
            let event = ResidentTokenEvent {
                session_id: sequence.session_id,
                request_id: sequence.request_id,
                index,
                token: action.token_id,
                logit: action.logit,
                text,
            };
            queued += self.output.reconcile_decode(event)?;
        }
        Ok(queued)
    }

    pub(super) fn finish_successful_sequence(
        &mut self,
        session: SessionId,
        reason: SequenceFinishReason,
    ) -> Result<bool> {
        let Some(position) = self.output.finish_success(
            &mut self.scheduler,
            &mut self.slot_pool,
            session,
            reason,
        )?
        else {
            return Ok(false);
        };
        self.finalize_sequence_state(session, position)?;
        self.observability.stats.finished_sequences += 1;
        Ok(true)
    }

    pub(super) fn cancellation_accepted(&self, request_id: Option<RequestId>) -> bool {
        // Only the runtime owner's accepted cancellation arbitrates a terminal;
        // a frontend atomic flag alone has not crossed this linearization point.
        self.output.cancellation_accepted(request_id)
    }

    pub(super) fn finish_after_decode_action(
        &mut self,
        action: &SchedulerAction,
    ) -> Result<ActionFinishOutcome> {
        let actions: &[DecodeAction] = match action {
            SchedulerAction::Execute { decodes, .. } => decodes,
            SchedulerAction::DecodeBatch(actions) => actions,
            _ => return Ok(ActionFinishOutcome::default()),
        };

        let mut outcome = ActionFinishOutcome::default();
        for action in actions {
            let Some(sequence) = self.scheduler.active_sequence(action.session_id) else {
                continue;
            };
            // Publish is irrevocable, but a previously accepted cancellation
            // still owns the request terminal. Leave it active for cancellation
            // cleanup after the committed action/output have been reconciled.
            let reason = if sequence.generated >= sequence.max_new_tokens {
                Some(SequenceFinishReason::MaxTokens)
            } else if matched_stop(&sequence.generated_text, &sequence.stop) {
                Some(SequenceFinishReason::StopString)
            } else {
                None
            };
            if let TerminalDecision::Finish(reason) =
                arbitrate_terminal(self.cancellation_accepted(sequence.request_id), reason)
                && self.finish_successful_sequence(action.session_id, reason)?
            {
                outcome.finished += 1;
                outcome.session_ids.push(action.session_id);
            }
        }
        Ok(outcome)
    }

    pub(super) fn apply_execution_output(
        &mut self,
        scheduled: &ScheduledBatch,
        output: &ExecutionOutput,
        action_finished_sessions: &[SessionId],
    ) -> Result<OutputOutcome> {
        let mut outcome = OutputOutcome::default();
        for row in &output.logits {
            let correlation = scheduled
                .sequence_for_input_row(row.input_row)
                .copied()
                .ok_or_else(|| Error::InvalidRequest {
                    message: format!(
                        "output input row {} has no scheduled sequence",
                        row.input_row
                    ),
                })?;
            let execution_sequence = scheduled
                .execution()
                .sequences()
                .iter()
                .find(|sequence| sequence.query.contains(&row.input_row))
                .ok_or_else(|| Error::InvalidRequest {
                    message: format!(
                        "output input row {} has no execution sequence span",
                        row.input_row
                    ),
                })?;
            let session_id = correlation.session_id;
            let Some(sequence) = self.scheduler.active_sequence(session_id) else {
                if action_finished_sessions.contains(&session_id) {
                    // The just-committed token ended the sequence (for example via
                    // a stop string), so its already-computed next-token logits are
                    // intentionally discarded after successful correlation.
                    continue;
                }
                return Err(Error::InvalidRequest {
                    message: format!(
                        "output for input row {} references inactive session {:?}",
                        row.input_row, session_id
                    ),
                });
            };
            if sequence.request_id != correlation.request_id {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "output correlation request mismatch for session {:?}: active {:?}, scheduled {:?}",
                        session_id, sequence.request_id, correlation.request_id
                    ),
                });
            }
            if sequence.kv_handle != correlation.kv_handle {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "output correlation KV mismatch for session {:?}: active {:?}, scheduled {:?}",
                        session_id, sequence.kv_handle, correlation.kv_handle
                    ),
                });
            }
            if sequence.position != execution_sequence.sequence_len as usize {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "output correlation position mismatch for session {:?}: active {}, executed {}",
                        session_id, sequence.position, execution_sequence.sequence_len
                    ),
                });
            }
            // Correlate the committed output above, but do not finish or stage
            // another token for a request whose cancellation was accepted while
            // Publish was pending. Physical commit and Cancelled are compatible.
            let natural_reason = if sequence.generated >= sequence.max_new_tokens {
                Some(SequenceFinishReason::MaxTokens)
            } else if sequence.position >= self.config.ctx_size {
                Some(SequenceFinishReason::Context)
            } else {
                None
            };
            match arbitrate_terminal(
                self.cancellation_accepted(sequence.request_id),
                natural_reason,
            ) {
                TerminalDecision::DeferForCancellation => continue,
                TerminalDecision::Finish(reason) => {
                    outcome.finished +=
                        usize::from(self.finish_successful_sequence(session_id, reason)?);
                    continue;
                }
                TerminalDecision::Continue => {}
            }

            let Some(candidate) = greedy_candidate(&row.logits) else {
                outcome.finished += usize::from(
                    self.finish_successful_sequence(session_id, SequenceFinishReason::NoCandidate)?,
                );
                continue;
            };

            if self.config.stop_at_eos
                && !sequence.ignore_eos
                && self.executor.runner().is_eos_token(candidate.token_id)
            {
                outcome.finished += usize::from(
                    self.finish_successful_sequence(session_id, SequenceFinishReason::Eos)?,
                );
                continue;
            }

            self.scheduler.stage_decode_candidate(
                session_id,
                candidate.token_id,
                Some(candidate.logit),
            )?;
            self.observability.stats.staged_tokens += 1;
            outcome.staged += 1;
            // Speculative anchors retain their existing publication protocol.
            // In target-only mode selection is irrevocable after this forward's
            // commit, but generated/text/position remain at the KV frontier until
            // the next decode commits, including the final requested token.
            if !self.config.enable_native_proposals {
                let sequence = self.scheduler.active_sequence(session_id).unwrap();
                let mut decode = sequence.incremental_decode.clone();
                let text = self
                    .executor
                    .runner()
                    .decode_incremental(candidate.token_id, &mut decode)?
                    .unwrap_or_default();
                let event = ResidentTokenEvent {
                    session_id,
                    request_id: sequence.request_id,
                    index: sequence.generated,
                    token: candidate.token_id,
                    logit: Some(candidate.logit),
                    text,
                };
                self.output.enqueue_early(event)?;
                outcome.queued += 1;
            }
        }
        Ok(outcome)
    }
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: ResidentModelRunner,
    C: SequenceSlotPool,
{
    pub(super) fn truncate_native_proposal_at_output_boundary(
        &self,
        sequence: &SequenceState,
        anchor_token_id: u32,
        proposal: Vec<u32>,
    ) -> Result<Vec<u32>> {
        if self.config.stop_at_eos
            && !sequence.ignore_eos
            && self.executor.runner().is_eos_token(anchor_token_id)
        {
            return Err(Error::Invariant {
                message: "an EOS token must not be staged as a speculative anchor".into(),
            });
        }

        let mut decode_state = sequence.incremental_decode.clone();
        let mut generated_text = sequence.generated_text.clone();
        let anchor_text = self
            .executor
            .runner()
            .decode_incremental(anchor_token_id, &mut decode_state)?
            .unwrap_or_default();
        generated_text.push_str(&anchor_text);
        if matched_stop(&generated_text, &sequence.stop) {
            return Ok(Vec::new());
        }

        let mut admitted = Vec::with_capacity(proposal.len());
        for token_id in proposal {
            if self.config.stop_at_eos
                && !sequence.ignore_eos
                && self.executor.runner().is_eos_token(token_id)
            {
                break;
            }
            let text = self
                .executor
                .runner()
                .decode_incremental(token_id, &mut decode_state)?
                .unwrap_or_default();
            generated_text.push_str(&text);
            admitted.push(token_id);
            if matched_stop(&generated_text, &sequence.stop) {
                break;
            }
        }
        Ok(admitted)
    }

    pub(super) fn enqueue_speculative_committed_tokens(
        &mut self,
        action: &DecodeAction,
        accepted: &[u32],
    ) -> Result<usize> {
        let mut tokens = Vec::with_capacity(accepted.len() + 1);
        tokens.push((action.token_id, action.logit));
        tokens.extend(accepted.iter().copied().map(|token| (token, None)));
        let emitted_tokens = tokens.len();
        let runner = self.executor.runner();
        let sequence = self
            .scheduler
            .active_sequence_mut(action.session_id)
            .ok_or_else(|| Error::Invariant {
                message: format!(
                    "cannot emit speculative block for inactive session {:?}",
                    action.session_id
                ),
            })?;
        let start_index = sequence
            .generated
            .checked_sub(tokens.len())
            .ok_or_else(|| Error::Invariant {
                message: "speculative emitted block exceeds committed generation count".into(),
            })?;
        for (offset, (token, logit)) in tokens.into_iter().enumerate() {
            let text = runner
                .decode_incremental(token, &mut sequence.incremental_decode)?
                .unwrap_or_default();
            sequence.append_generated_text(&text);
            let event = ResidentTokenEvent {
                session_id: sequence.session_id,
                request_id: sequence.request_id,
                index: start_index + offset,
                token,
                logit,
                text,
            };
            self.output.enqueue_committed(event);
        }
        Ok(emitted_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(session: u64, token: u32) -> ResidentTokenEvent {
        ResidentTokenEvent {
            session_id: SessionId(session),
            request_id: Some(RequestId(session)),
            index: 0,
            token,
            logit: None,
            text: token.to_string(),
        }
    }

    #[test]
    fn nack_then_cancel_discards_only_uncommitted_delivery() {
        let mut output = OutputArbiter::default();
        let early = event(1, 65);
        let committed = event(2, 66);
        output.enqueue_early(early.clone()).unwrap();
        output.enqueue_committed(committed.clone());
        let mut emitted = 0;
        let mut trace = Vec::new();
        assert!(
            output
                .flush(
                    &mut |_| Err(Error::InvalidRequest {
                        message: "not accepted".into()
                    }),
                    &mut emitted,
                    &mut trace
                )
                .is_err()
        );
        assert_eq!(output.front(), Some(&early));
        assert_eq!(emitted, 0);
        assert!(trace.is_empty());
        assert!(output.owns_session(SessionId(1)));
        assert!(output.owns_session(SessionId(2)));
        output.discard_uncommitted(SessionId(1));
        output.discard_uncommitted(SessionId(2));
        assert!(!output.owns_session(SessionId(1)));
        assert_eq!(output.front(), Some(&committed));
        let mut received = Vec::new();
        output
            .flush(
                &mut |event| {
                    received.push(event.clone());
                    Ok(())
                },
                &mut emitted,
                &mut trace,
            )
            .unwrap();
        assert_eq!(received, vec![committed]);
        assert_eq!(emitted, 1);
        assert!(!output.owns_session(SessionId(2)));
    }

    #[test]
    fn early_ack_retains_frontier_until_decode_reconciliation() {
        let mut output = OutputArbiter::default();
        let selected = event(1, 65);
        output.enqueue_early(selected.clone()).unwrap();
        let mut emitted = 0;
        let mut trace = Vec::new();
        let mut received = Vec::new();
        output
            .flush(
                &mut |event| {
                    received.push(event.clone());
                    Ok(())
                },
                &mut emitted,
                &mut trace,
            )
            .unwrap();
        assert!(output.outbox_empty());
        assert!(output.has_early(&SessionId(1)));
        assert!(output.owns_session(SessionId(1)));
        assert_eq!(output.reconcile_decode(selected.clone()).unwrap(), 0);
        assert!(!output.owns_session(SessionId(1)));
        output
            .flush(
                &mut |_| panic!("reconciled early event must not be emitted twice"),
                &mut emitted,
                &mut trace,
            )
            .unwrap();
        assert_eq!(received, vec![selected]);
        assert_eq!(emitted, 1);
    }
}
