//! Admission operations on the driver's existing authority.

use super::*;

/// One lifecycle permit, never a second copy of model or physical ownership.
pub(super) struct RuntimeRequestIdentity {
    pub(super) session_id: SessionId,
    pub(super) terminal_consumed: bool,
    pub(super) cleanup_receipt: super::RequestCleanupOwner,
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner,
    C: SequenceSlotPool,
{
    pub fn admission_options(&self) -> RuntimeAdmissionOptions {
        self.admission_options
    }

    /// Failure-atomic limit update. Existing identities, including quarantined
    /// cleanup, cannot be revoked by reducing a limit.
    pub fn set_admission_options(&mut self, options: RuntimeAdmissionOptions) -> Result<()> {
        let options = options.validate()?;
        let snapshot = self.admission_snapshot();
        if snapshot.waiting_requests > options.max_waiting_requests
            || snapshot.request_identities_held > options.max_request_identities
            || snapshot.session_identities_held > options.max_session_identities
        {
            return Err(RuntimeAdmissionError::OptionsInUse.into());
        }
        self.admission_options = options;
        Ok(())
    }

    /// Monotonic barrier for new requests; existing work may continue to step.
    /// This does not claim logical drain or physical shutdown completion.
    pub fn close_admission(&mut self) {
        self.admission_closed = true;
    }

    /// Available capacity in each domain is its limit minus its held gauge.
    /// Never add the waiting subset to the request identity count.
    pub fn admission_snapshot(&self) -> RuntimeAdmissionSnapshot {
        RuntimeAdmissionSnapshot {
            limits: self.admission_options,
            waiting_requests: self.scheduler.waiting_len(),
            request_identities_held: self
                .request_identities
                .iter()
                .filter(|(id, record)| !self.identity_releasable(**id, record))
                .count(),
            session_identities_held: self.owned_admission_sessions().len(),
            closed: self.admission_closed || self.shutting_down,
        }
    }

    pub(super) fn session_has_pending_ownership(&self, session: SessionId) -> bool {
        self.sessions.is_owned(&session)
            || self.sessions.is_suspended(&session)
            || self.sessions.has_cleanup(&session)
            || self.output.cancellation_owns_session(session)
            || self
                .resident_transactions
                .values()
                .any(|pending| action_session_ids(&pending.action).contains(&session))
            || self.speculative_transactions.values().any(|pending| {
                pending
                    .actions()
                    .iter()
                    .any(|action| action.session_id == session)
            })
            || self.output.owns_session(session)
    }

    pub(super) fn identity_releasable(
        &self,
        request: RequestId,
        record: &RuntimeRequestIdentity,
    ) -> bool {
        // Retained session state is intentionally session-owned, not turn-owned.
        // Unattributed cohort/prefix cleanup queues impose a conservative global
        // reuse barrier. Query those owners instead of copying physical grants.
        record.terminal_consumed
            && !self.scheduler.contains_request_identity(request)
            && !self.session_has_pending_ownership(record.session_id)
            && (self.sessions.is_retained(&record.session_id)
                || (!self.sessions.contains_sequence_state(&record.session_id)
                    && !self.sessions.has_page_slot(&record.session_id)))
            && !self.continuations.has_pending_detaches()
            && !self.continuations.has_pending_failures()
            && self.kv.pending_retirements_empty()
            && self.kv.pending_abort_empty()
            && self.prefix.pending_cleanups_empty()
    }

    /// Capture the current generation before terminal draining can reap it.
    pub fn request_cleanup(&self, request: RequestId) -> super::InferenceRequestCleanup {
        self.request_identities
            .get(&request)
            .map_or(super::InferenceRequestCleanup::Unavailable, |record| {
                super::InferenceRequestCleanup::Tracked(record.cleanup_receipt.receipt())
            })
    }

    pub(super) fn reap_request_identities(&mut self) {
        let released = self
            .request_identities
            .iter()
            .filter_map(|(id, record)| self.identity_releasable(*id, record).then_some(*id))
            .collect::<Vec<_>>();
        for id in released {
            if let Some(record) = self.request_identities.remove(&id) {
                record.cleanup_receipt.release();
            }
        }
    }

    pub(super) fn owned_admission_sessions(&self) -> HashSet<SessionId> {
        let mut sessions = self
            .scheduler
            .identity_sessions()
            .into_iter()
            .collect::<HashSet<_>>();
        sessions.extend(self.request_identities.iter().filter_map(|(id, record)| {
            (!self.identity_releasable(*id, record)).then_some(record.session_id)
        }));
        sessions.extend(self.sessions.retained_ids());
        sessions.extend(self.sessions.sequence_state_ids());
        sessions.extend(self.sessions.page_slot_ids());
        sessions.extend(self.sessions.suspended_ids());
        sessions.extend(self.sessions.owner_ids());
        sessions.extend(self.sessions.cleanup_ids());
        sessions
    }

    pub(super) fn check_admission(
        &self,
        request: &GenerateRequest,
        waiting: bool,
    ) -> Result<SessionId> {
        if self.shutting_down || self.admission_closed {
            return Err(RuntimeAdmissionError::Closed.into());
        }
        if self.scheduler.contains_request_identity(request.id)
            || self
                .request_identities
                .get(&request.id)
                .is_some_and(|record| !self.identity_releasable(request.id, record))
            || self.transaction_for_request(request.id).is_some()
            || self.output.cancellation_accepted(Some(request.id))
        {
            return Err(RuntimeAdmissionError::DuplicateRequest {
                request_id: request.id,
            }
            .into());
        }
        let owned_sessions = self.owned_admission_sessions();
        let session = if let Some(session) = request.session_id {
            let retained_only = self.sessions.is_retained(&session)
                && !self.scheduler.contains_session_identity(session)
                && !self.session_has_pending_ownership(session)
                && self.request_identities.iter().all(|(id, record)| {
                    record.session_id != session || self.identity_releasable(*id, record)
                });
            if !retained_only && owned_sessions.contains(&session) {
                return Err(RuntimeAdmissionError::SessionBusy {
                    session_id: session,
                }
                .into());
            }
            session
        } else {
            let mut candidate = self
                .next_admission_session
                .ok_or(RuntimeAdmissionError::SessionIdentityExhausted)?;
            while owned_sessions.contains(&SessionId(candidate)) {
                candidate = candidate
                    .checked_add(1)
                    .ok_or(RuntimeAdmissionError::SessionIdentityExhausted)?;
            }
            SessionId(candidate)
        };
        let held = self.admission_snapshot().request_identities_held;
        if held >= self.admission_options.max_request_identities {
            return Err(RuntimeAdmissionError::Capacity {
                resource: RuntimeAdmissionResource::RequestIdentities,
                held,
                limit: self.admission_options.max_request_identities,
            }
            .into());
        }
        if waiting && self.scheduler.waiting_len() >= self.admission_options.max_waiting_requests {
            return Err(RuntimeAdmissionError::Capacity {
                resource: RuntimeAdmissionResource::WaitingRequests,
                held: self.scheduler.waiting_len(),
                limit: self.admission_options.max_waiting_requests,
            }
            .into());
        }
        if !owned_sessions.contains(&session)
            && owned_sessions.len() >= self.admission_options.max_session_identities
        {
            return Err(RuntimeAdmissionError::Capacity {
                resource: RuntimeAdmissionResource::SessionIdentities,
                held: owned_sessions.len(),
                limit: self.admission_options.max_session_identities,
            }
            .into());
        }
        Ok(session)
    }

    pub(super) fn validate_admission_position(
        &self,
        request: &GenerateRequest,
        position: usize,
    ) -> Result<()> {
        super::super::composition::validate_context_envelope(
            self.config.ctx_size,
            position,
            request.prompt_tokens.len(),
            request.max_new_tokens,
        )
    }

    pub(super) fn submit_admitted(
        &mut self,
        mut request: GenerateRequest,
        supplied_position: Option<usize>,
    ) -> Result<()> {
        let session = self.check_admission(&request, true)?;
        self.validate_configuration()?;
        let retained_position = self.sessions.retained_position(&session);
        if let (Some(supplied), Some(expected)) = (supplied_position, retained_position) {
            if supplied != expected {
                return Err(
                    RuntimeAdmissionError::RetainedPositionMismatch { supplied, expected }.into(),
                );
            }
        }
        let position = supplied_position.or(retained_position).unwrap_or(0);
        self.validate_admission_position(&request, position)?;
        let automatic_session = request.session_id.is_none();
        self.reap_request_identities();
        request.session_id = Some(session);
        self.request_identities.insert(
            request.id,
            RuntimeRequestIdentity {
                session_id: session,
                terminal_consumed: false,
                cleanup_receipt: super::RequestCleanupOwner::default(),
            },
        );
        if automatic_session {
            self.next_admission_session = session.0.checked_add(1);
        }
        if supplied_position.or(retained_position).is_some() {
            self.scheduler.submit_at_position(request, position);
        } else {
            self.scheduler.submit(request);
        }
        Ok(())
    }

    /// Acquire one lifecycle identity and move it into waiting. Err enqueues
    /// nothing and publishes no terminal, even for a duplicate legacy submit.
    pub fn try_submit(&mut self, request: GenerateRequest) -> Result<()> {
        self.submit_admitted(request, None)
    }

    pub fn try_submit_at_position(
        &mut self,
        request: GenerateRequest,
        position_start: usize,
    ) -> Result<()> {
        self.submit_admitted(request, Some(position_start))
    }

    /// Fork an active session from exactly its currently committed paged prefix.
    ///
    /// `target_request.prompt_tokens` is the suffix for the target branch; the
    /// target starts at `expected_committed_position` and never re-executes the
    /// shared prefix. Scheduler, model, and page-table state are all prepared
    /// before any target becomes visible.
    pub fn fork_session_exact(
        &mut self,
        source_session_id: SessionId,
        target_request: GenerateRequest,
        expected_committed_position: usize,
    ) -> Result<SessionId> {
        let target_identity = self.check_admission(&target_request, false)?;
        self.validate_configuration()?;
        self.validate_admission_position(&target_request, expected_committed_position)?;
        self.ensure_no_suspended_execution("fork a session")?;
        if self.output.has_early(&source_session_id) {
            return Err(Error::InvalidRequest {
                message: "cannot fork before the delivered token's KV append commits".into(),
            });
        }

        let target_session_id = target_request
            .session_id
            .ok_or_else(|| Error::InvalidRequest {
                message: "exact fork target request requires an explicit session ID".into(),
            })?;
        if self.sessions.is_suspended(&source_session_id) {
            return Err(Error::InvalidRequest {
                message: "cannot fork from a suspended source session".into(),
            });
        }
        if self.sessions.has_cleanup(&target_session_id)
            || self.sessions.is_suspended(&target_session_id)
            || self.sessions.contains_sequence_state(&target_session_id)
            || self.sessions.has_page_slot(&target_session_id)
        {
            return Err(Error::InvalidRequest {
                message: format!("fork target session {target_session_id:?} already exists"),
            });
        }
        let source_page_slot =
            *self
                .sessions
                .page_slot(&source_session_id)
                .ok_or_else(|| Error::InvalidRequest {
                    message: format!(
                        "fork source session {source_session_id:?} has no authoritative page slot"
                    ),
                })?;
        let target_page_slot = StateSlot::new(self.next_page_slot);
        let next_page_slot =
            self.next_page_slot
                .checked_add(1)
                .ok_or_else(|| Error::InvalidRequest {
                    message: "driver page slot generation overflow during fork".into(),
                })?;
        let kv_handle = self.slot_pool.alloc_slot()?;
        self.reap_request_identities();
        self.request_identities.insert(
            target_request.id,
            RuntimeRequestIdentity {
                session_id: target_identity,
                terminal_consumed: false,
                cleanup_receipt: super::RequestCleanupOwner::default(),
            },
        );

        self.sessions.begin_cleanup(
            target_session_id,
            PendingSequenceCleanup::Admission {
                model_state: None,
                slot: Some(kv_handle),
                lease: prefix::PrefixAdmissionPin::default(),
                errors: Vec::new(),
            },
        );
        let preparation = (|| {
            let prepared_schedule = self.scheduler.prepare_fork_session_exact(
                source_session_id,
                target_session_id,
                &target_request,
                expected_committed_position,
                kv_handle,
            )?;
            debug_assert_eq!(prepared_schedule.target_session_id(), target_session_id);
            self.validate_admission_position(&target_request, expected_committed_position)?;
            let prepared_pages = self
                .page_manager
                .as_ref()
                .ok_or_else(|| Error::InvalidRequest {
                    message: "exact-prefix fork requires an authoritative KvPageManager".into(),
                })?
                .prepare_fork_sequence_exact(
                    source_page_slot,
                    target_page_slot,
                    0,
                    expected_committed_position,
                )?;
            let source = self
                .sessions
                .sequence_state(&source_session_id)
                .ok_or_else(|| Error::InvalidRequest {
                    message: format!(
                        "fork source session {source_session_id:?} has no model state"
                    ),
                })?;
            let state = self
                .executor
                .fork_sequence_state_from(source, expected_committed_position)?;
            let PendingSequenceCleanup::Admission { model_state, .. } = self
                .sessions
                .cleanup_mut(&target_session_id)
                .expect("prepared owner")
            else {
                unreachable!()
            };
            *model_state = Some(state);
            #[cfg(test)]
            self.fail_stage("fork publish")?;
            self.page_manager
                .as_mut()
                .expect("validated page manager")
                .publish_fork_sequence_exact(prepared_pages)?;
            Ok(prepared_schedule)
        })();
        let prepared_schedule = match preparation {
            Ok(prepared) => prepared,
            Err(error) => {
                // A rejected fork has no accepted request terminal to consume,
                // but its identity remains reserved until every undo completes.
                self.request_identities
                    .get_mut(&target_request.id)
                    .expect("fork identity")
                    .terminal_consumed = true;
                let error = self.abort_prepared_admission(target_session_id, error);
                self.reap_request_identities();
                return Err(error);
            }
        };
        let PendingSequenceCleanup::Admission {
            model_state: Some(prepared_model),
            ..
        } = self
            .sessions
            .take_cleanup(&target_session_id)
            .expect("prepared owner")
        else {
            unreachable!()
        };

        self.scheduler.publish_fork_session_exact(prepared_schedule);
        self.sessions
            .publish_admitted(target_session_id, prepared_model, Some(target_page_slot));
        self.next_page_slot = next_page_slot;
        Ok(target_session_id)
    }

    pub(super) fn consume_terminal_identity(&mut self, sequence: &SequenceState) {
        if let Some(request_id) = sequence.request_id {
            if let Some(record) = self.request_identities.get_mut(&request_id) {
                record.terminal_consumed = true;
            }
        }
    }

    pub(super) fn abort_prepared_admission(
        &mut self,
        session_id: SessionId,
        error: Error,
    ) -> Error {
        if let Some(PendingSequenceCleanup::Admission { errors, .. }) =
            self.sessions.cleanup_mut(&session_id)
        {
            errors.push(error.to_string());
        }
        let cleanup = self.progress_sequence_cleanup(session_id);
        Error::with_cleanup("prepared admission", error, cleanup)
    }

    pub(super) fn cleanup_prepared_admission(&mut self, session_id: SessionId) -> Result<()> {
        let mut failures = Vec::new();
        if let Err(error) = self.unpin_admission(session_id) {
            failures.push(CleanupStep::new("prefix unpin", error));
        }
        let Some(PendingSequenceCleanup::Admission {
            model_state,
            slot,
            errors,
            lease,
            ..
        }) = self.sessions.cleanup_mut(&session_id)
        else {
            return Ok(());
        };
        if let Some(state) = model_state.take() {
            if let Err(failure) = self.executor.try_release_sequence_state(state) {
                let (error, state) = failure.into_parts();
                *model_state = Some(state);
                failures.push(CleanupStep::new("admission model release", error));
            }
        }
        if let Some(handle) = *slot {
            match self.slot_pool.free_slot(handle) {
                Ok(()) => *slot = None,
                Err(error) => failures.push(CleanupStep::new("admission slot release", error)),
            }
        }
        for failure in &failures {
            errors.push(format!("{failure:?}"));
        }
        // Scheduler abort already owns a failed waiting-admission slot. Do not
        // free it here or release the identity before that owner has drained.
        if model_state.is_none()
            && slot.is_none()
            && lease.is_none()
            && self.scheduler.failed_slot_ownership() == 0
        {
            self.sessions.take_cleanup(&session_id);
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(Error::with_cleanup_batch(
                "prepared admission undo",
                Error::Invariant {
                    message: "prepared admission cleanup incomplete".into(),
                },
                failures,
            ))
        }
    }

    /// Admit waiting sequences only after their model and logical KV ownership is
    /// fully prepared. No fallible work remains once the scheduler publishes the
    /// sequence as active.
    pub(super) fn admit_new_sequences(&mut self) -> Result<()> {
        #[cfg(test)]
        self.tick_trace.push(DriverTickEvent::Admission);
        if self.shutting_down {
            return Ok(());
        }
        while let Some(mut prepared) = self
            .scheduler
            .prepare_waiting_admission(&mut self.slot_pool)?
        {
            let session_id = prepared.session_id();
            if self.sessions.has_cleanup(&session_id) || self.sessions.is_suspended(&session_id) {
                self.scheduler
                    .abort_waiting_admission(prepared, &mut self.slot_pool)?;
                break;
            }
            if self.sessions.contains_sequence_state(&session_id) {
                self.scheduler.publish_waiting_admission(prepared);
                #[cfg(test)]
                self.tick_trace.push(DriverTickEvent::Admitted(session_id));
                continue;
            }

            let cache_eligible = self.prefix_cache_namespace().is_some()
                && !self.has_live_transactions()
                && !self.sessions.is_retained(&session_id)
                && prepared.is_fresh_prompt()
                && !prepared.prompt_tokens().is_empty();
            let (page_slot, next_page_slot) = if self.page_manager.is_some() {
                let slot = StateSlot::new(self.next_page_slot);
                let next_page_slot = match self.next_page_slot.checked_add(1) {
                    Some(next_page_slot) => next_page_slot,
                    None => {
                        let error = Error::InvalidRequest {
                            message: "driver page slot generation overflow".into(),
                        };
                        let cleanup = self
                            .scheduler
                            .abort_waiting_admission(prepared, &mut self.slot_pool);
                        return Err(Error::with_cleanup(
                            "resident admission slot generation",
                            error,
                            cleanup,
                        ));
                    }
                };
                (Some(slot), Some(next_page_slot))
            } else {
                (None, None)
            };

            if cache_eligible {
                let target_slot = page_slot.expect("cache eligibility requires a page manager");
                match self.prepare_prefix_admission(&mut prepared, target_slot) {
                    Ok(Some(prefix)) => {
                        debug_assert!(prefix.matched_tokens > 0);
                        let publication = (|| {
                            #[cfg(test)]
                            self.fail_stage("prefix publish")?;
                            self.page_manager
                                .as_mut()
                                .expect("prefix page manager")
                                .publish_fork_prefix_snapshot(prefix.pages)
                        })();
                        if let Err(error) = publication {
                            let scheduler_cleanup = self
                                .scheduler
                                .abort_waiting_admission(prepared, &mut self.slot_pool);
                            let error = Error::with_cleanup(
                                "prefix scheduler admission cleanup",
                                error,
                                scheduler_cleanup,
                            );
                            return Err(self.abort_prepared_admission(session_id, error));
                        }
                        let PendingSequenceCleanup::Admission {
                            model_state: Some(state),
                            ..
                        } = self
                            .sessions
                            .take_cleanup(&session_id)
                            .expect("prefix owner")
                        else {
                            unreachable!()
                        };
                        self.sessions
                            .publish_admitted(session_id, state, Some(target_slot));
                        self.next_page_slot =
                            next_page_slot.expect("page-manager admission reserves the next slot");
                        self.prefix.mark_cached_session(session_id);
                        self.scheduler.publish_waiting_admission(prepared);
                        #[cfg(test)]
                        self.tick_trace.push(DriverTickEvent::Admitted(session_id));
                        self.observability.prefix_cache.hits =
                            self.observability.prefix_cache.hits.saturating_add(1);
                        continue;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        let cleanup = self
                            .scheduler
                            .abort_waiting_admission(prepared, &mut self.slot_pool);
                        let error = Error::with_cleanup("prefix cache admission", error, cleanup);
                        return Err(self.abort_prepared_admission(session_id, error));
                    }
                }
            }

            // A fresh sequence needs pages for its first executable chunk.
            // Query the existing owners; acceptance is not a KV reservation.
            // Prefix reuse/eviction keeps its ordinary custody and retry path.
            if let Some(manager) = &self.page_manager {
                let config = self.scheduler.config();
                let chunk = prepared
                    .prompt_tokens()
                    .len()
                    .min(config.prefill_chunk_size)
                    .min(if config.max_batch_tokens == 0 {
                        usize::MAX
                    } else {
                        config.max_batch_tokens
                    });
                let required = chunk.div_ceil(manager.page_size());
                if let Err(error) = self.evict_prefixes_for_kv_pages(required) {
                    let cleanup = self
                        .scheduler
                        .abort_waiting_admission(prepared, &mut self.slot_pool);
                    return Err(Error::with_cleanup("waiting KV capacity", error, cleanup));
                }
                let manager = self.page_manager.as_ref().expect("page manager installed");
                let logical_available = manager.max_pages() == 0
                    || manager
                        .max_pages()
                        .saturating_sub(manager.allocated_pages())
                        >= required;
                if !logical_available || self.available_kv_page_credits() < required {
                    self.scheduler
                        .abort_waiting_admission(prepared, &mut self.slot_pool)?;
                    break;
                }
            }

            if let Some(slot) = page_slot
                && let Err(error) = self
                    .page_manager
                    .as_mut()
                    .expect("page-manager presence was checked")
                    .alloc_sequence(slot, 0)
            {
                let cleanup = self
                    .scheduler
                    .abort_waiting_admission(prepared, &mut self.slot_pool);
                return Err(Error::with_cleanup(
                    "resident admission logical KV preparation",
                    error,
                    cleanup,
                ));
            }
            if let Some(next_page_slot) = next_page_slot {
                self.next_page_slot = next_page_slot;
            }

            let state = match self.executor.create_sequence_state() {
                Ok(state) => state,
                Err(error) => {
                    let mut cleanup = Vec::new();
                    if let Some(slot) = page_slot {
                        let previous = self.sessions.retain_page_slot(session_id, slot);
                        debug_assert!(
                            previous.is_none(),
                            "fresh admission page slot must remain absent"
                        );
                        if let Err(source) = self.release_sequence_state(session_id) {
                            cleanup.push(CleanupStep::new(
                                format!("session {session_id:?} logical KV cleanup"),
                                source,
                            ));
                        }
                    }
                    if let Err(source) = self
                        .scheduler
                        .abort_waiting_admission(prepared, &mut self.slot_pool)
                    {
                        cleanup.push(CleanupStep::new(
                            format!("session {session_id:?} scheduler admission cleanup"),
                            source,
                        ));
                    }
                    return Err(Error::with_cleanup_batch(
                        "resident admission model preparation",
                        error,
                        cleanup,
                    ));
                }
            };

            self.sessions.publish_admitted(session_id, state, page_slot);
            if cache_eligible {
                self.prefix.mark_cached_session(session_id);
            }
            self.scheduler.publish_waiting_admission(prepared);
            #[cfg(test)]
            self.tick_trace.push(DriverTickEvent::Admitted(session_id));
            if cache_eligible {
                self.observability.prefix_cache.misses =
                    self.observability.prefix_cache.misses.saturating_add(1);
            }
        }
        Ok(())
    }
}

/// One moved schedule/model pair, shared by resident and speculative admission.
/// Restoration is fallible and borrows the payload, so a failed compensation
/// leaves both sides available to the transaction's existing pending owner.
pub(super) struct AdmissionSessions<S> {
    pub(super) schedules: Vec<SuspendedSequenceSchedule>,
    pub(super) states: Vec<S>,
}

impl<S> AdmissionSessions<S> {
    pub(super) fn claim<R, C>(
        driver: &mut ResidentTopKDriver<R, C>,
        transaction: ExecutionTransactionId,
        sessions: &[SessionId],
    ) -> Result<Self>
    where
        R: MultiSessionRunner<SequenceState = S>,
        C: SequenceSlotPool,
    {
        let (schedules, states) = driver.claim_transaction_sessions(transaction, sessions)?;
        Ok(Self { schedules, states })
    }

    pub(super) fn restore<R, C>(&mut self, driver: &mut ResidentTopKDriver<R, C>) -> Result<()>
    where
        R: MultiSessionRunner<SequenceState = S>,
        C: SequenceSlotPool,
    {
        driver.progress_transaction_session_restore(&mut self.schedules, &mut self.states)
    }

    pub(super) fn commit(self) -> (Vec<SuspendedSequenceSchedule>, Vec<S>) {
        (self.schedules, self.states)
    }
}

/// Transient custody between session claim and successful backend preparation.
/// On failure the same payload moves into the existing resident cleanup phase;
/// there is no second registry and no backend end submission for a rejected
/// preparation. In particular a failed compensation cannot drop claimed states.
struct TransactionAdmission<S> {
    transaction: ExecutionTransactionId,
    action: SchedulerAction,
    scheduled: ScheduledBatch,
    kv: Option<PendingResidentKv>,
    sessions: AdmissionSessions<S>,
    prefetch_declared: bool,
}

impl<S> TransactionAdmission<S> {
    fn commit(self) -> PendingResidentBatch<S> {
        let (schedules, states) = self.sessions.commit();
        PendingResidentBatch {
            transaction: self.transaction,
            action: self.action,
            scheduled: self.scheduled,
            kv: self.kv,
            schedules,
            states,
            phase: ResidentTransactionPhase::Executing(None),
        }
    }

    fn rollback<R, C>(
        self,
        driver: &mut ResidentTopKDriver<R, C>,
        error: Error,
        stage: &'static str,
    ) -> Error
    where
        R: MultiSessionRunner<SequenceState = S>,
        C: SequenceSlotPool,
    {
        let custody = self
            .prefetch_declared
            .then_some(TransactionCustodyOutcome::RolledBack);
        let mut pending = self.commit();
        pending.phase = ResidentTransactionPhase::BackendAbortedPendingCleanup {
            request: None,
            custody,
            continuation: None,
            failure: Some((error, stage)),
            decode_requeued: false,
        };
        driver
            .finish_resident_abort(pending, None)
            .expect_err("failed admission retains its primary error until compensation completes")
    }
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner,
    C: SequenceSlotPool,
{
    pub(super) fn prepare_resident_transaction(
        &mut self,
        action: SchedulerAction,
        scheduled: ScheduledBatch,
    ) -> Result<(PendingResidentBatch<R::SequenceState>, u64)> {
        let transaction = self.take_transaction_id()?;
        let session_ids = scheduled
            .sequences
            .iter()
            .map(|sequence| sequence.session_id)
            .collect::<Vec<_>>();
        let sessions = AdmissionSessions::claim(self, transaction, &session_ids)
            .map_err(|error| self.abort_action(&action, error, false, "session claim"))?;
        let mut admission = TransactionAdmission {
            transaction,
            action,
            scheduled,
            kv: None,
            sessions,
            prefetch_declared: false,
        };
        let preparation = (|| -> std::result::Result<u64, (Error, &'static str)> {
            let resource_demand = Self::action_demand(&admission.action);
            let generations = admission
                .sessions
                .states
                .iter()
                .map(|state| self.executor.runner().sequence_generation(state))
                .collect::<Vec<_>>();
            let reservations = self
                .reserve_batch_pages(
                    transaction,
                    resource_demand,
                    &admission.scheduled,
                    &generations,
                )
                .map_err(|error| (error, "KV reserve"))?;
            admission.kv = Some(PendingResidentKv::Reserved(reservations));
            let Some(PendingResidentKv::Reserved(reservations)) = &admission.kv else {
                unreachable!("admission owns only uncommitted reservations")
            };
            self.bind_reserved_pages(&mut admission.scheduled, reservations)
                .map_err(|error| (error, "KV binding"))?;
            let views = match &self.page_manager {
                Some(manager) => manager.reservation_views(reservations),
                None if reservations.is_empty() => Ok(Vec::<KvReservationView>::new()),
                None => Err(Error::Invariant {
                    message: "KV reservations exist without an authoritative page manager".into(),
                }),
            }
            .map_err(|error| (error, "KV reservation view"))?;
            // declare may partially attach custody before returning an error.
            admission.prefetch_declared = true;
            self.declare_transaction_prefetch(
                transaction,
                resource_demand,
                admission.scheduled.execution(),
            )
            .map_err(|error| (error, "transaction prefetch"))?;
            let started_ns = self.runtime_now_ns();
            #[cfg(test)]
            self.fail_stage("transaction prepare")
                .map_err(|error| (error, "backend prepare"))?;
            self.executor
                .prepare_batch_with_kv(
                    transaction,
                    &mut admission.sessions.states,
                    admission.scheduled.execution(),
                    &views,
                )
                .map_err(|error| (error, "backend prepare"))?;
            Ok(started_ns)
        })();
        match preparation {
            Ok(started_ns) => Ok((admission.commit(), started_ns)),
            Err((error, stage)) => Err(admission.rollback(self, error, stage)),
        }
    }
}
