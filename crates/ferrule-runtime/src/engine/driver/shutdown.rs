//! Shutdown operations on the driver's existing authority.

use super::*;

impl<R, C> ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner,
    C: SequenceSlotPool,
{
    pub fn try_into_runner(mut self) -> std::result::Result<R, Box<(Error, Self)>>
    where
        R: ResidentModelRunner,
    {
        if let Err(error) = self
            .scheduler
            .retry_failed_slot_releases(&mut self.slot_pool)
        {
            return Err(Box::new((error, self)));
        }
        if self.has_pending_non_prefix_async_work()
            || self.warmup_pending()
            || self.sessions.owner_count() != 0
            || !self.output.outbox_empty()
            || !self.output.early_empty()
        {
            return Err(Box::new((
                Error::InvalidRequest { message:
                    "cannot extract resident runner with live execution transactions or undelivered committed events"
                        .into(),
                 },
                self,
            )));
        }
        if !self.scheduler.is_idle()
            || self.sessions.suspended_count() != 0
            || self.sessions.sequence_state_count() != 0
            || self.sessions.retained_count() != 0
            || !self.prefix.cached_sessions_empty()
        {
            return Err(Box::new((
                Error::InvalidRequest { message:
                    "cannot extract resident runner while session state is still retained or active"
                        .into(),
                 },
                self,
            )));
        }

        if let Err(error) = self.drain_prefix_cache() {
            return Err(Box::new((error, self)));
        }
        let retiring_pages = self
            .page_manager
            .as_ref()
            .map_or(0, |manager| manager.stats().retiring_pages);
        let active_page_sequences = self
            .page_manager
            .as_ref()
            .map_or(0, KvPageManager::active_sequences);
        if !self.prefix.pending_cleanups_empty()
            || !self.kv.pending_retirements_empty()
            || !self.kv.grants_empty()
            || retiring_pages != 0
            || active_page_sequences != 0
        {
            return Err(Box::new((
                Error::Invariant {
                    message: format!(
                        "cannot extract resident runner with retained KV ownership: prefix_cleanups={} retirements={} grants={} retiring_pages={retiring_pages} active_page_sequences={active_page_sequences}",
                        self.prefix.pending_cleanup_count(),
                        self.kv.pending_retirement_count(),
                        self.kv.grant_count(),
                    ),
                },
                self,
            )));
        }
        if let Err(error) = self.load_registry.shutdown(self.runtime_now_ns(), 0) {
            let error = Error::from(error);
            return Err(Box::new((error, self)));
        }

        Ok(self.executor.into_runner())
    }
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: ResidentModelRunner,
    C: SequenceSlotPool,
{
    /// Drain logical ownership, then close the model's physical owner in place.
    /// Failed close retains the runner and returns the original error, not Pending
    /// or Complete. In particular, zero logical KV pages is not a GPU fence.
    pub fn shutdown_and_close_progress<F>(
        &mut self,
        on_token: &mut F,
    ) -> Result<ResidentShutdownProgress>
    where
        R: ResidentModelRunner,
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        if let Some(report) = self.physical_shutdown_report {
            return Ok(ResidentShutdownProgress::Complete(report));
        }
        match (ShutdownCoordinator { driver: self }).progress(on_token)? {
            None => Ok(ResidentShutdownProgress::Pending),
            Some(proof) => proof
                .close_physical()
                .map(ResidentShutdownProgress::Complete),
        }
    }

    pub fn physical_shutdown_complete(&self) -> bool {
        self.physical_shutdown_report.is_some()
    }

    /// Read-only reporting never extracts the runner or discards failed custody.
    #[cfg(feature = "cuda")]
    pub(crate) fn runner_for_report(&self) -> &R {
        self.executor.runner()
    }

    /// Logical drain only; frontend shutdown must use shutdown_and_close_progress.
    pub fn shutdown_progress<F>(&mut self, on_token: &mut F) -> Result<ResidentShutdownProgress>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        Ok(
            match (ShutdownCoordinator { driver: self }).progress(on_token)? {
                None => ResidentShutdownProgress::Pending,
                Some(proof) => ResidentShutdownProgress::Complete(proof.report),
            },
        )
    }

    /// Stop admission, quiesce model transactions, release all session/KV
    /// ownership, and drain provider completions. An error retains any ownership
    /// that could not yet be proven quiescent so the caller may retry.
    pub fn shutdown<F>(
        &mut self,
        on_token: &mut F,
        maximum_completions: usize,
    ) -> Result<ResidentDriverShutdownReport>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        (ShutdownCoordinator { driver: self })
            .pass(on_token, maximum_completions)
            .map(|proof| proof.report)
    }

    fn quiesce_shutdown_transactions<F>(&mut self, on_token: &mut F) -> Result<()>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        self.shutting_down = true;
        let bookkeeping = self
            .speculative_transactions
            .iter()
            .filter_map(|(id, pending)| {
                matches!(pending, PendingSpeculativeDriverCohort::Bookkeeping(_)).then_some(*id)
            })
            .collect::<Vec<_>>();
        for transaction in bookkeeping {
            self.drive_speculative_bookkeeping(transaction, on_token)?;
        }
        self.flush_committed_token_outbox(on_token)?;
        self.progress_ending_transactions(on_token)?;

        let mut transactions = self
            .resident_transactions
            .keys()
            .chain(self.speculative_transactions.keys())
            .copied()
            .filter(|transaction| {
                !self
                    .resident_transactions
                    .get(transaction)
                    .is_some_and(|pending| pending.phase.is_ending())
                    && !matches!(
                        self.speculative_transactions.get(transaction),
                        Some(
                            PendingSpeculativeDriverCohort::Ending(_)
                                | PendingSpeculativeDriverCohort::AdmissionRollback(_)
                        )
                    )
            })
            .collect::<Vec<_>>();
        transactions.sort_unstable();
        transactions.dedup();
        for transaction in transactions {
            let request_id =
                self.request_for_transaction(transaction)
                    .ok_or_else(|| Error::InvalidRequest {
                        message: format!(
                            "cannot quiesce transaction {transaction:?} without a request owner"
                        ),
                    })?;
            self.request_transaction_abort(transaction, request_id)?;
        }
        // A cancellation request can retain post-terminal synchronous cleanup
        // while reporting Pending to the request API. Retry only that owner-local
        // work before classifying pre-terminal backend progress as wake-dependent.
        self.progress_post_terminal_transaction_cleanups(on_token)?;
        if !self.resident_transactions.is_empty()
            || !self.speculative_transactions.is_empty()
            || self.has_live_transactions()
        {
            return Err(Error::ShutdownIncomplete {
                message: "driver could not quiesce every execution transaction".into(),
            });
        }
        Ok(())
    }

    fn release_shutdown_sessions(&mut self) -> Result<()> {
        self.retry_pending_continuation_cleanups()?;
        self.cleanup_materialization_failures(false)?;
        self.progress_pending_cleanups()?;
        self.drain_prefix_cache()?;

        for request_id in self.scheduler.request_ids() {
            self.cancel_scheduled_request(request_id)?;
        }

        let suspended = self.sessions.suspended_ids().collect::<Vec<_>>();
        for session_id in suspended {
            self.advance_session_transition(session_id)?;
            if !self.sessions.is_suspended(&session_id) {
                continue;
            }
            let schedule = self.sessions.suspend_into_cleanup(session_id);
            self.prefix.unmark_cached_session(&session_id);
            self.scheduler
                .cancel_suspended(schedule, &mut self.slot_pool)?;
            self.progress_sequence_cleanup(session_id)?;
        }

        for session_id in self.scheduler.active_session_ids() {
            let cancellation = self
                .scheduler
                .cancel_sequence(session_id, &mut self.slot_pool);
            let cleanup = self.release_sequence_state(session_id);
            match (cancellation, cleanup) {
                (Ok(_), Ok(())) => {}
                (Err(error), Ok(())) | (Ok(_), Err(error)) => return Err(error),
                (Err(error), Err(cleanup)) => {
                    return Err(Error::cleanup("active session shutdown", error, cleanup));
                }
            }
        }
        let retained = self.sessions.sequence_state_ids().collect::<Vec<_>>();
        for session_id in retained {
            self.release_sequence_state(session_id)?;
        }
        self.sessions.clear_retained();
        self.prefix.clear_cached_sessions();
        self.progress_pending_cleanups()?;
        self.retry_pending_request_cancellations()?;

        Ok(())
    }

    fn confirm_shutdown_drain(
        &self,
        registry: crate::io::ShutdownReport,
    ) -> Result<ResidentDriverShutdownReport> {
        let executor_transactions =
            self.resident_transactions.len() + self.speculative_transactions.len();
        let report = ResidentDriverShutdownReport {
            registry,
            executor_transactions,
            kv_page_grants: self.kv.grant_count(),
            pending_kv_retirements: self.kv.pending_retirement_count(),
        };
        let retiring_pages = self
            .page_manager
            .as_ref()
            .map_or(0, |manager| manager.stats().retiring_pages);
        if report.executor_transactions != 0
            || report.kv_page_grants != 0
            || report.pending_kv_retirements != 0
            || !self.kv.pending_abort_empty()
            || retiring_pages != 0
            || !self.continuations.is_empty()
            || self.continuations.transaction_count() != 0
            || self.continuations.has_pending_failures()
            || self.continuations.has_pending_detaches()
            || self.sessions.owner_count() != 0
            || self.sessions.cleanup_count() != 0
            || !self.output.cancellations_empty()
            || !self.prefix.pending_cleanups_empty()
            || !self.prefix.is_empty()
            || !self.prefix.cached_sessions_empty()
            || self.sessions.sequence_state_count() != 0
            || self.sessions.suspended_count() != 0
            || self.scheduler.failed_slot_ownership() != 0
            || !self.scheduler.is_idle()
            || !self.output.outbox_empty()
            || !self.output.early_empty()
            || self
                .page_manager
                .as_ref()
                .is_some_and(|manager| manager.active_sequences() != 0)
        {
            return Err(Error::Invariant {
                message: format!(
                    "driver shutdown retained ownership: {report:?}, resident_kv_aborts={}, retiring_pages={retiring_pages}, continuations={}, transaction_continuations={}, session_owners={}, cleanups={}, request_cancellations={}, prefix_cleanups={}, cached_prefixes={}, prefix_sessions={}, sequence_states={}, suspended={}, failed_slots={}, scheduler_idle={}, undelivered={}, early_pending={}, active_page_sequences={}",
                    self.kv.pending_abort_count(),
                    self.continuations.len(),
                    self.continuations.transaction_count(),
                    self.sessions.owner_count(),
                    self.sessions.cleanup_count(),
                    self.output.cancellation_count(),
                    self.prefix.pending_cleanup_count(),
                    self.prefix.len(),
                    self.prefix.cached_session_count(),
                    self.sessions.sequence_state_count(),
                    self.sessions.suspended_count(),
                    self.scheduler.failed_slot_ownership(),
                    self.scheduler.is_idle(),
                    !self.output.outbox_empty(),
                    !self.output.early_empty(),
                    self.page_manager
                        .as_ref()
                        .map_or(0, KvPageManager::active_sequences),
                ),
            });
        }
        Ok(report)
    }
}

/// An exclusive borrow, not another lifecycle owner. Each pass samples the
/// actual owners; completion is not inferred from request terminals or counters.
struct ShutdownCoordinator<'a, R: MultiSessionRunner, C: SequenceSlotPool> {
    driver: &'a mut ResidentTopKDriver<R, C>,
}

/// The borrow prevents new work between drain validation and physical close.
/// This proof is never cached: only a successful physical close caches a report.
struct LogicalDrain<'a, R: MultiSessionRunner, C: SequenceSlotPool> {
    driver: &'a mut ResidentTopKDriver<R, C>,
    report: ResidentDriverShutdownReport,
}

impl<R: ResidentModelRunner, C: SequenceSlotPool> LogicalDrain<'_, R, C> {
    fn close_physical(self) -> Result<ResidentDriverShutdownReport> {
        self.driver.executor.shutdown_physical()?;
        self.driver.physical_shutdown_report = Some(self.report);
        Ok(self.report)
    }
}

/// Only queued completion evidence permits another immediate pass. Unknown
/// custody and pending backend endings are wake-dependent, not runnable work.
enum DrainRetry {
    QueuedCompletions,
    AwaitEvidence,
}

impl DrainRetry {
    fn classify(error: Error, queued_completions: usize) -> Result<Self> {
        match error {
            Error::Registry { source }
                if matches!(*source, crate::io::RegistryError::ShutdownIncomplete { .. })
                    && queued_completions != 0 =>
            {
                Ok(Self::QueuedCompletions)
            }
            Error::Registry { source }
                if matches!(
                    *source,
                    crate::io::RegistryError::ShutdownIncomplete { .. }
                        | crate::io::RegistryError::LostCompletion { .. }
                ) =>
            {
                Ok(Self::AwaitEvidence)
            }
            Error::ShutdownIncomplete { .. } => Ok(Self::AwaitEvidence),
            error => Err(error),
        }
    }
}

impl<'a, R: ResidentModelRunner, C: SequenceSlotPool> ShutdownCoordinator<'a, R, C> {
    fn drain_pass<F>(
        &mut self,
        on_token: &mut F,
        maximum_completions: usize,
    ) -> Result<ResidentDriverShutdownReport>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        self.driver.quiesce_shutdown_transactions(on_token)?;
        self.driver.release_shutdown_sessions()?;
        let registry = self
            .driver
            .load_registry
            .shutdown(self.driver.runtime_now_ns(), maximum_completions)?;
        self.driver.progress_pending_cleanups()?;
        self.driver.update_hard_resource_observability();
        self.driver.confirm_shutdown_drain(registry)
    }

    fn pass<F>(
        mut self,
        on_token: &mut F,
        maximum_completions: usize,
    ) -> Result<LogicalDrain<'a, R, C>>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        let report = self.drain_pass(on_token, maximum_completions)?;
        Ok(LogicalDrain {
            driver: self.driver,
            report,
        })
    }

    fn progress<F>(mut self, on_token: &mut F) -> Result<Option<LogicalDrain<'a, R, C>>>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        const COMPLETIONS_PER_TICK: usize = 512;
        loop {
            match self.drain_pass(on_token, COMPLETIONS_PER_TICK) {
                Ok(report) => {
                    return Ok(Some(LogicalDrain {
                        driver: self.driver,
                        report,
                    }));
                }
                Err(error) => match DrainRetry::classify(
                    error,
                    self.driver.load_registry.pending_completions(),
                )? {
                    DrainRetry::QueuedCompletions => continue,
                    DrainRetry::AwaitEvidence => return Ok(None),
                },
            }
        }
    }
}
