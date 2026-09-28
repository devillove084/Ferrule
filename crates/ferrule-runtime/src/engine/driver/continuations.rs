//! Continuation and materialization operations on the driver's existing authority.

use std::collections::{HashMap, HashSet, VecDeque};

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::io_protocol::{
    CancellationReason, ContinuationId, DependencySetEpoch, RequestGeneration, RetirementReason,
    WaiterId,
};
use ferrule_model::{MultiSessionRunner, PendingModelProgress, ResidentModelRunner};

use crate::io::{
    FailedContinuation, FairQueueConfig, LoadRegistry, ResumeDisposition,
    RuntimeMaterializationProvider, RuntimeMaterializationResolverStats,
};
use crate::scheduling::{ResourceKind, SequenceSlotPool};
use crate::{Error, Result};

use super::{
    PendingSpeculativeDriverCohort, PendingSpeculativeEndingDriverCohort,
    PendingSpeculativeVerificationDriverCohort, ResidentTopKDriver, SpeculativeEnding,
};

/// One ready FIFO and its membership index. Neither container can be mutated
/// independently, and a tick's budget is a value, not another queue or ledger.
#[derive(Default)]
struct ReadyTransactions {
    queue: VecDeque<ExecutionTransactionId>,
    queued: HashSet<ExecutionTransactionId>,
}

impl ReadyTransactions {
    fn enqueue(&mut self, transaction: ExecutionTransactionId) {
        if self.queued.insert(transaction) {
            self.queue.push_back(transaction);
        }
    }

    fn pop(&mut self) -> Option<ExecutionTransactionId> {
        let transaction = self.queue.pop_front()?;
        self.queued.remove(&transaction);
        Some(transaction)
    }

    fn remove(&mut self, transaction: ExecutionTransactionId) {
        self.queue.retain(|queued| *queued != transaction);
        self.queued.remove(&transaction);
    }

    fn snapshot_len(&self) -> usize {
        self.queue.len()
    }
}

struct RegisteredModelContinuation {
    transaction: ExecutionTransactionId,
    dependencies: ferrule_common::DependencySet,
}

#[derive(Debug)]
struct PendingMaterializationFailure {
    failed: FailedContinuation,
    transaction: Option<ExecutionTransactionId>,
}

/// The driver's logical continuation owner. These indexes and retry records
/// move together; materialization grants and execution custody stay in the
/// driver's LoadRegistry and transaction records respectively.
pub(super) struct ContinuationRegistry {
    registrations: HashMap<ContinuationId, RegisteredModelContinuation>,
    by_transaction: HashMap<ExecutionTransactionId, HashSet<ContinuationId>>,
    ready: HashSet<ContinuationId>,
    ready_transactions: ReadyTransactions,
    pending_detaches: HashMap<ContinuationId, CancellationReason>,
    failures: VecDeque<PendingMaterializationFailure>,
    next_dependency_epoch: u64,
}

impl Default for ContinuationRegistry {
    fn default() -> Self {
        Self {
            registrations: HashMap::new(),
            by_transaction: HashMap::new(),
            ready: HashSet::new(),
            ready_transactions: ReadyTransactions::default(),
            pending_detaches: HashMap::new(),
            failures: VecDeque::new(),
            next_dependency_epoch: 1,
        }
    }
}

impl ContinuationRegistry {
    fn duplicate(continuation: ContinuationId) -> Error {
        Error::InvalidRequest {
            message: format!(
                "model continuation {} is already registered",
                continuation.get()
            ),
        }
    }

    fn prepare_waiter(&mut self, progress: &PendingModelProgress) -> Result<WaiterId> {
        let continuation = progress.continuation();
        if self.contains(continuation) {
            return Err(Self::duplicate(continuation));
        }
        let epoch = DependencySetEpoch::new(self.next_dependency_epoch);
        self.next_dependency_epoch =
            self.next_dependency_epoch
                .checked_add(1)
                .ok_or_else(|| Error::InvalidRequest {
                    message: "dependency-set epoch space is exhausted".into(),
                })?;
        Ok(WaiterId::new(
            progress.transaction(),
            RequestGeneration::new(1),
            epoch,
            continuation,
        )?)
    }

    /// Preflight before attachment; publish both indexes only after it succeeds.
    /// No fallible bookkeeping follows a successful external attachment. A
    /// failed attachment's physical undo remains owned by LoadRegistry.
    fn register(
        &mut self,
        progress: &PendingModelProgress,
        attach: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        let continuation = progress.continuation();
        let std::collections::hash_map::Entry::Vacant(entry) =
            self.registrations.entry(continuation)
        else {
            return Err(Self::duplicate(continuation));
        };
        attach()?;
        entry.insert(RegisteredModelContinuation {
            transaction: progress.transaction(),
            dependencies: progress.dependencies().clone(),
        });
        self.by_transaction
            .entry(progress.transaction())
            .or_default()
            .insert(continuation);
        Ok(())
    }

    pub(super) fn contains(&self, continuation: ContinuationId) -> bool {
        self.registrations.contains_key(&continuation)
    }

    pub(super) fn transaction(
        &self,
        continuation: ContinuationId,
    ) -> Option<ExecutionTransactionId> {
        self.registrations
            .get(&continuation)
            .map(|record| record.transaction)
    }

    fn dependencies(&self, continuation: ContinuationId) -> Option<&ferrule_common::DependencySet> {
        self.registrations
            .get(&continuation)
            .map(|record| &record.dependencies)
    }

    pub(super) fn continuation_for(
        &self,
        transaction: ExecutionTransactionId,
    ) -> Option<ContinuationId> {
        self.by_transaction
            .get(&transaction)
            .and_then(|ids| ids.iter().copied().next())
    }

    fn ready_continuation_for(
        &self,
        transaction: ExecutionTransactionId,
    ) -> Option<ContinuationId> {
        self.by_transaction
            .get(&transaction)
            .into_iter()
            .flatten()
            .filter(|id| self.ready.contains(id))
            .copied()
            .min_by_key(|id| id.get())
    }

    fn mark_ready(&mut self, continuation: ContinuationId) -> Result<ExecutionTransactionId> {
        let transaction = self
            .transaction(continuation)
            .ok_or_else(|| Error::Invariant {
                message: format!(
                    "ready continuation {} has no transaction registration",
                    continuation.get()
                ),
            })?;
        self.ready.insert(continuation);
        Ok(transaction)
    }

    fn enqueue_transaction(&mut self, transaction: ExecutionTransactionId) {
        self.ready_transactions.enqueue(transaction);
    }

    fn pop_ready_transaction(&mut self) -> Option<ExecutionTransactionId> {
        self.ready_transactions.pop()
    }

    fn remove_transaction_from_queue(&mut self, transaction: ExecutionTransactionId) {
        self.ready_transactions.remove(transaction);
    }

    pub(super) fn ready_snapshot_len(&self) -> usize {
        self.ready_transactions.snapshot_len()
    }

    /// Retain both indexes and the original undo intent until the external
    /// registry confirms detach. Retrying cannot overwrite that intent.
    fn detach(
        &mut self,
        continuation: ContinuationId,
        reason: CancellationReason,
        detach: impl FnOnce(CancellationReason) -> Result<()>,
    ) -> Result<()> {
        if !self.contains(continuation) {
            return Ok(());
        }
        let reason = self
            .pending_detaches
            .entry(continuation)
            .or_insert(reason)
            .clone();
        detach(reason)?;
        self.confirm_release(continuation);
        Ok(())
    }

    fn pending_detaches(&self) -> Vec<(ContinuationId, CancellationReason)> {
        let mut pending = self
            .pending_detaches
            .iter()
            .map(|(id, reason)| (*id, reason.clone()))
            .collect::<Vec<_>>();
        pending.sort_unstable_by_key(|(id, _)| *id);
        pending
    }

    /// Called only after detach, consumed resume, or terminal lease cleanup has
    /// succeeded. Removing a registration must also remove its ready/retry marks
    /// and exactly its reverse edge, without touching sibling continuations.
    fn confirm_release(&mut self, continuation: ContinuationId) {
        self.ready.remove(&continuation);
        self.pending_detaches.remove(&continuation);
        let Some(record) = self.registrations.remove(&continuation) else {
            return;
        };
        if let Some(ids) = self.by_transaction.get_mut(&record.transaction) {
            ids.remove(&continuation);
            if ids.is_empty() {
                self.by_transaction.remove(&record.transaction);
            }
        }
    }

    fn collect_failure(&mut self, failed: FailedContinuation) {
        let transaction = self.transaction(failed.continuation);
        self.failures.push_back(PendingMaterializationFailure {
            failed,
            transaction,
        });
    }

    fn pop_failure(&mut self) -> Option<PendingMaterializationFailure> {
        self.failures.pop_front()
    }

    pub(super) fn has_pending_failures(&self) -> bool {
        !self.failures.is_empty()
    }
    pub(super) fn has_pending_detaches(&self) -> bool {
        !self.pending_detaches.is_empty()
    }
    pub(super) fn is_empty(&self) -> bool {
        self.registrations.is_empty()
    }
    pub(super) fn len(&self) -> usize {
        self.registrations.len()
    }
    pub(super) fn transaction_count(&self) -> usize {
        self.by_transaction.len()
    }

    #[cfg(test)]
    pub(super) fn exhaust_dependency_epochs(&mut self) {
        self.next_dependency_epoch = u64::MAX;
    }

    #[cfg(test)]
    pub(super) fn assert_consistent(&self) {
        for (id, record) in &self.registrations {
            assert!(self.by_transaction[&record.transaction].contains(id));
        }
        for (transaction, ids) in &self.by_transaction {
            assert!(!ids.is_empty());
            for id in ids {
                assert_eq!(self.transaction(*id), Some(*transaction));
            }
        }
        assert!(self.ready.iter().all(|id| self.contains(*id)));
        assert!(self.pending_detaches.keys().all(|id| self.contains(*id)));
        let queue = &self.ready_transactions;
        assert_eq!(queue.queue.len(), queue.queued.len());
        assert_eq!(
            queue.queue.iter().copied().collect::<HashSet<_>>(),
            queue.queued
        );
    }
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner,
    C: SequenceSlotPool,
{
    pub fn warmup_pending(&self) -> bool {
        !self.warmup_requests.is_empty()
            || self
                .load_registry
                .prefetch_active(crate::io::PrefetchOwner::ModelWarmup)
            || self
                .load_registry
                .prefetch_failed(crate::io::PrefetchOwner::ModelWarmup)
    }

    pub(crate) fn start_background_work(&mut self) -> Result<()> {
        const INITIAL_BACKGROUND_TRANSITIONS: usize = 64;

        self.begin_warmup()?;
        let now_ns = self.runtime_now_ns();
        self.load_registry
            .drive(now_ns, INITIAL_BACKGROUND_TRANSITIONS)?;
        self.check_warmup()?;
        self.update_hard_resource_observability();
        Ok(())
    }

    pub(super) fn begin_warmup(&mut self) -> Result<()> {
        #[cfg(test)]
        self.tick_trace.push(super::DriverTickEvent::Warmup);
        let owner = crate::io::PrefetchOwner::ModelWarmup;
        if self.load_registry.prefetch_active(owner) || self.warmup_requests.is_empty() {
            return Ok(());
        }
        let requests = std::mem::take(&mut self.warmup_requests);
        match self.prefetch(
            owner,
            crate::scheduling::ResourceDemand::ModelWarmup,
            requests.clone(),
        ) {
            Ok(_) => {
                self.warmup_started_ns = Some(self.runtime_now_ns());
                self.warmup_last_report_ns = 0;
                Ok(())
            }
            Err(error) => {
                self.warmup_requests = requests;
                Err(error)
            }
        }
    }

    pub(super) fn check_warmup(&mut self) -> Result<()> {
        #[cfg(test)]
        self.tick_trace.push(super::DriverTickEvent::WarmupCheck);
        let owner = crate::io::PrefetchOwner::ModelWarmup;
        let Some(reason) = self.load_registry.take_prefetch_failure(owner) else {
            if !self.load_registry.prefetch_active(owner) && self.warmup_requests.is_empty() {
                self.warmup_started_ns = None;
            }
            return Ok(());
        };
        match reason {
            RetirementReason::Failed(source) => Err(Error::WarmupMaterialization { source }),
            RetirementReason::Cancelled(reason) => {
                Err(Error::WarmupMaterializationCancelled { reason })
            }
            RetirementReason::Stale(reason) => Err(Error::WarmupMaterializationStale { reason }),
            RetirementReason::ResidentOwnershipTransferred => Ok(()),
            RetirementReason::Drained
            | RetirementReason::OrphanCompletion
            | RetirementReason::OwnerShutdown => Err(Error::WarmupMaterializationNotPublished),
        }
    }

    pub fn has_background_work(&self) -> bool {
        self.warmup_pending()
    }

    /// Replaces the materialization provider before any execution suspends.
    pub fn try_with_materialization_provider<B>(
        mut self,
        provider: B,
        resources: crate::scheduling::PhysicalResourceBroker,
        fairness: FairQueueConfig,
    ) -> Result<Self>
    where
        B: RuntimeMaterializationProvider + 'static,
    {
        self.ensure_no_suspended_execution("replace the materialization provider")?;
        if let Some(resolver) = self.uninstalled_materialization_resolver.take() {
            self.executor
                .runner_mut()
                .install_materialization_resolver(Box::new(resolver))?;
        }
        self.load_registry = LoadRegistry::new(
            Box::new(provider) as Box<dyn RuntimeMaterializationProvider>,
            resources,
            fairness,
        )?;
        Ok(self)
    }

    #[cfg(test)]
    pub fn with_materialization_provider<B>(self, provider: B) -> Result<Self>
    where
        B: RuntimeMaterializationProvider + 'static,
    {
        self.try_with_materialization_provider(
            provider,
            crate::scheduling::PhysicalResourceBroker::testing_default(),
            FairQueueConfig::default(),
        )
    }

    pub fn load_registry(&self) -> &LoadRegistry<Box<dyn RuntimeMaterializationProvider>> {
        &self.load_registry
    }

    pub fn prefetch(
        &mut self,
        owner: crate::io::PrefetchOwner,
        demand: crate::scheduling::ResourceDemand,
        requests: impl IntoIterator<Item = ferrule_model::MaterializationRequest>,
    ) -> Result<crate::io::PrefetchReport> {
        if !demand.is_prefetch() {
            return Err(Error::InvalidRequest {
                message: format!(
                    "prefetch requires a non-blocking resource demand, got {demand:?}"
                ),
            });
        }
        if self.shutting_down {
            return Err(Error::InvalidRequest {
                message: "cannot submit prefetch after driver shutdown begins".into(),
            });
        }
        let mut prepared = Vec::new();
        for request in requests {
            let key = match self.materialization_resolver.prepare_prefetch(request) {
                Ok(key) => key,
                Err(error) => {
                    let cleanup = self
                        .load_registry
                        .discard_preparations(prepared.iter().copied());
                    return Err(Error::with_cleanup(
                        "prefetch preparation",
                        error,
                        cleanup.map_err(Error::from),
                    ));
                }
            };
            prepared.push(key);
        }
        let loads = match prepared
            .iter()
            .copied()
            .map(|key| self.load_registry.prepare_prefetch_request(key, demand))
            .collect::<std::result::Result<Vec<_>, _>>()
        {
            Ok(loads) => loads,
            Err(error) => {
                let cleanup = self
                    .load_registry
                    .discard_preparations(prepared.iter().copied());
                return Err(Error::with_cleanup(
                    "prefetch request validation",
                    error,
                    cleanup.map_err(Error::from),
                ));
            }
        };
        match self
            .load_registry
            .prefetch(owner, loads, self.runtime_now_ns())
        {
            Ok(report) => {
                if !report.created.is_empty() {
                    self.completion_hub.notify();
                }
                Ok(report)
            }
            Err(error) => {
                let cleanup = self.load_registry.discard_preparations(prepared);
                Err(Error::with_cleanup(
                    "prefetch admission",
                    error,
                    cleanup.map_err(Error::from),
                ))
            }
        }
    }

    pub fn cancel_prefetch(&mut self, owner: crate::io::PrefetchOwner) -> Result<()> {
        self.load_registry
            .cancel_prefetch(owner, self.runtime_now_ns())
            .map_err(Error::from)
    }

    pub fn materialization_resolver_stats(&self) -> RuntimeMaterializationResolverStats {
        self.materialization_resolver.stats()
    }

    pub(super) fn enqueue_transaction(&mut self, transaction: ExecutionTransactionId) {
        self.continuations.enqueue_transaction(transaction);
    }

    pub(super) fn pop_ready_transaction(&mut self) -> Option<ExecutionTransactionId> {
        self.continuations.pop_ready_transaction()
    }

    pub(super) fn remove_transaction_from_queue(&mut self, transaction: ExecutionTransactionId) {
        self.continuations
            .remove_transaction_from_queue(transaction);
    }

    pub(super) fn ready_continuation_for(
        &self,
        transaction: ExecutionTransactionId,
    ) -> Option<ContinuationId> {
        self.continuations.ready_continuation_for(transaction)
    }

    pub(super) fn declare_transaction_prefetch(
        &mut self,
        transaction: ExecutionTransactionId,
        required: crate::scheduling::ResourceDemand,
        batch: &ferrule_common::execution::ExecutionBatch,
    ) -> Result<()> {
        let phases = required.phases().ok_or_else(|| Error::Invariant {
            message: "transaction execution demand has no execution phase".into(),
        })?;
        let demand = crate::scheduling::ResourceDemand::prefetch_phases(phases);
        let requests = self
            .executor
            .runner()
            .transaction_prefetch_requests(transaction, batch)?;
        if requests.is_empty() {
            return Ok(());
        }
        let owner = crate::io::PrefetchOwner::transaction(transaction, phases);
        let report = self.prefetch(owner, demand, requests)?;
        let operations = report.created.len().saturating_add(report.joined.len());
        if operations == 0 {
            return Ok(());
        }
        let transition_budget = operations.saturating_mul(2).max(1);
        let now_ns = self.runtime_now_ns();
        self.load_registry
            .drive_foreground(now_ns, transition_budget)?;
        Ok(())
    }

    pub(super) fn register_pending_progress(
        &mut self,
        progress: &PendingModelProgress,
        demand: crate::scheduling::ResourceDemand,
    ) -> Result<()> {
        progress.dependencies().validate()?;
        let waiter = self.continuations.prepare_waiter(progress)?;
        let keys = progress
            .dependencies()
            .iter()
            .filter_map(|dependency| dependency.materialization_key())
            .collect::<Vec<_>>();
        if keys.len() != progress.resources().len()
            || progress
                .resources()
                .iter()
                .zip(&keys)
                .any(|(resource, key)| resource.key() != *key)
        {
            return Err(Error::InvalidRequest { message:
                "pending model progress resource custody does not exactly match its dependencies"
                    .into(),
             });
        }
        let requests = match progress
            .resources()
            .iter()
            .map(|resource| {
                self.load_registry.prepare_execution_request(
                    resource.key(),
                    demand,
                    resource.retention(),
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()
        {
            Ok(requests) => requests,
            Err(error) => {
                let cleanup = self
                    .load_registry
                    .discard_preparations(keys.iter().copied());
                return Err(Error::with_cleanup(
                    "materialization request validation",
                    error,
                    cleanup.map_err(Error::from),
                ));
            }
        };
        let now_ns = self.runtime_now_ns();
        let registry = &mut self.load_registry;
        if let Err(error) = self.continuations.register(progress, || {
            registry
                .attach_waiter(waiter, demand, requests, now_ns)
                .map(|_| ())
                .map_err(Error::from)
        }) {
            let cleanup = self
                .load_registry
                .discard_preparations(keys.iter().copied());
            return Err(Error::with_cleanup(
                "materialization attachment",
                error,
                cleanup.map_err(Error::from),
            ));
        }

        if !keys.is_empty() {
            // The completion listener is armed before model execution. Newly
            // attached registry work is owner-local and has no provider event yet,
            // so explicitly schedule the next owner step that submits it.
            self.completion_hub.notify();
        }
        Ok(())
    }

    pub(super) fn retry_pending_continuation_cleanups(&mut self) -> Result<()> {
        let mut first_error = None;
        for (continuation, reason) in self.continuations.pending_detaches() {
            if let Err(error) = self.detach_registered_continuation(continuation, reason) {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn collect_materialization_failures(&mut self) {
        while let Some(failed) = self.load_registry.pop_failed() {
            self.continuations.collect_failure(failed);
        }
    }

    pub(super) fn cleanup_materialization_failures(
        &mut self,
        report_business_error: bool,
    ) -> Result<()>
    where
        R: ResidentModelRunner,
    {
        self.collect_materialization_failures();
        if !self.continuations.has_pending_failures() {
            return Ok(());
        }
        self.load_registry.finish_pending_lease_releases()?;

        let mut failures = Vec::<(Option<ExecutionTransactionId>, Error)>::new();
        while let Some(failure) = self.continuations.pop_failure() {
            let continuation = failure.failed.continuation;
            self.unregister_continuation(continuation);
            let error = Error::InvalidRequest {
                message: format!(
                    "materialization for continuation {} failed ({:?}); transaction={:?}",
                    continuation.get(),
                    failure.failed.failure,
                    failure.transaction
                ),
            };
            if let Some(index) = failures
                .iter()
                .position(|(transaction, _)| *transaction == failure.transaction)
            {
                let (transaction, previous) = failures.remove(index);
                failures.insert(
                    index,
                    (
                        transaction,
                        Error::combine("transaction materialization", previous, error),
                    ),
                );
            } else {
                failures.push((failure.transaction, error));
            }
        }

        let mut first_error = None;
        for (transaction, error) in failures {
            let terminal_error = match transaction {
                Some(transaction)
                    if self.resident_transactions.contains_key(&transaction)
                        || self.speculative_transactions.contains_key(&transaction) =>
                {
                    self.abort_materialization_failed_transaction(transaction, error)
                }
                _ => Some(error),
            };
            if first_error.is_none() {
                first_error = terminal_error;
            }
        }

        if report_business_error {
            first_error.map_or(Ok(()), Err)
        } else {
            Ok(())
        }
    }

    fn abort_materialization_failed_transaction(
        &mut self,
        transaction: ExecutionTransactionId,
        error: Error,
    ) -> Option<Error>
    where
        R: ResidentModelRunner,
    {
        self.remove_transaction_from_queue(transaction);
        if let Some(pending) = self.resident_transactions.remove(&transaction) {
            return self
                .abort_failed_resident(pending, error, "model materialization")
                .err();
        }

        let Some(pending) = self.speculative_transactions.remove(&transaction) else {
            return Some(Error::Invariant {
                message: format!(
                    "materialization failure lost execution transaction {transaction:?} ownership"
                ),
            });
        };
        match pending {
            PendingSpeculativeDriverCohort::AdmissionRollback(pending) => {
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::AdmissionRollback(pending),
                );
                Some(error)
            }
            PendingSpeculativeDriverCohort::Bookkeeping(pending) => {
                self.speculative_transactions.insert(
                    transaction,
                    PendingSpeculativeDriverCohort::Bookkeeping(pending),
                );
                Some(error)
            }
            PendingSpeculativeDriverCohort::Proposing(pending) => self
                .cancel_native_proposal_cohort(*pending, None, Some(error))
                .err(),
            PendingSpeculativeDriverCohort::Verifying(pending) => {
                let PendingSpeculativeVerificationDriverCohort {
                    transaction,
                    cohort_start,
                    actions,
                    prepared,
                    source_states,
                    schedules,
                    verification,
                    cancellation_request,
                } = *pending;
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
                        failure: Some((error, "model materialization")),
                    },
                    request_id: cancellation_request,
                    continuation: None,
                };
                self.drive_speculative_ending(ending, &mut |_| Ok(())).err()
            }
            PendingSpeculativeDriverCohort::Ending(pending) => {
                self.speculative_transactions
                    .insert(transaction, PendingSpeculativeDriverCohort::Ending(pending));
                Some(Error::Invariant {
                    message: format!(
                        "transaction {transaction:?} reported materialization failure while terminalizing"
                    ),
                })
            }
        }
    }

    pub(super) fn foreground_lifecycle_active(&self) -> bool {
        !self.scheduler.is_idle()
            || !self.resident_transactions.is_empty()
            || !self.speculative_transactions.is_empty()
            || !self.continuations.is_empty()
            || self.sessions.cleanup_count() != 0
            || self.continuations.has_pending_failures()
    }

    pub(super) fn materialization_transition_budget(&self) -> usize {
        const ACTIVE_TRANSITIONS_PER_SLICE: usize = 2;
        const IDLE_MINIMUM_TRANSITIONS: usize = 4_096;
        const IDLE_MAXIMUM_TRANSITIONS: usize = 32_768;
        const FOREGROUND_TRANSITIONS: usize = 512;

        if self.foreground_lifecycle_active() || !self.warmup_pending() {
            FOREGROUND_TRANSITIONS
        } else {
            self.load_registry
                .active_operations()
                .saturating_mul(ACTIVE_TRANSITIONS_PER_SLICE)
                .clamp(IDLE_MINIMUM_TRANSITIONS, IDLE_MAXIMUM_TRANSITIONS)
        }
    }

    pub(super) fn progress_materialization(&mut self) -> Result<()>
    where
        R: ResidentModelRunner,
    {
        #[cfg(test)]
        self.tick_trace
            .push(super::DriverTickEvent::Materialization);
        self.retry_pending_continuation_cleanups()?;
        self.cleanup_materialization_failures(true)?;

        let now_ns = self.runtime_now_ns();
        let transition_budget = self.materialization_transition_budget();
        let progressed = if self.foreground_lifecycle_active() {
            self.load_registry
                .drive_foreground(now_ns, transition_budget)?
        } else {
            self.load_registry.drive(now_ns, transition_budget)?
        };
        // The inference owner arms its listener before entering this method. If
        // owner-side work consumes the whole slice, publish a local wake so a
        // runnable transition left at the budget boundary cannot wait forever
        // for a provider completion that may never be needed.
        if progressed == transition_budget {
            self.completion_hub.notify();
        }
        const WARMUP_REPORT_INTERVAL_NS: u64 = 1_000_000_000;
        let trace_enabled = std::env::var_os("FERRULE_IO_TRACE").is_some();
        let report_warmup = self.warmup_pending()
            && now_ns.saturating_sub(self.warmup_last_report_ns) >= WARMUP_REPORT_INTERVAL_NS;
        if trace_enabled && report_warmup {
            self.warmup_last_report_ns = now_ns;
            let stages = self.load_registry.stage_counts().collect::<Vec<_>>();
            let resources = self
                .load_registry
                .resources()
                .snapshots()
                .filter(|snapshot| {
                    matches!(
                        snapshot.kind,
                        ResourceKind::ReadSlot
                            | ResourceKind::PinnedHostBytes
                            | ResourceKind::StorageReadBytes
                            | ResourceKind::UploadSlot
                            | ResourceKind::UploadBytes
                            | ResourceKind::InstallSlot
                            | ResourceKind::DeviceInstallBytes
                            | ResourceKind::ResidentBytes
                            | ResourceKind::ResidencyLease
                            | ResourceKind::LoadOperation
                    )
                })
                .map(|snapshot| (snapshot.kind, snapshot.in_use, snapshot.capacity))
                .collect::<Vec<_>>();
            let resident = self.load_registry.resident_entries();
            let resident_bytes = self.load_registry.resident_bytes();
            let elapsed_seconds = self
                .warmup_started_ns
                .map(|started| now_ns.saturating_sub(started) as f64 / 1_000_000_000.0)
                .unwrap_or_default();
            let gib_per_second = if elapsed_seconds > 0.0 {
                resident_bytes as f64 / elapsed_seconds / (1u64 << 30) as f64
            } else {
                0.0
            };
            let total = resident.saturating_add(
                self.load_registry
                    .prefetch_operations(crate::io::PrefetchOwner::ModelWarmup)
                    .count(),
            );
            let eta_seconds = if resident != 0 && elapsed_seconds > 0.0 {
                elapsed_seconds * total.saturating_sub(resident) as f64 / resident as f64
            } else {
                0.0
            };
            eprintln!(
                "[ferrule-io] tick={} progressed={} active={} physical={} runnable={} completions={} resident={}/{} resident_bytes={} elapsed_s={elapsed_seconds:.2} gib_s={gib_per_second:.2} eta_s={eta_seconds:.1} stages={stages:?} resources={resources:?}",
                self.runtime_tick,
                progressed,
                self.load_registry.active_operations(),
                self.load_registry.pending_physical_operations(),
                self.load_registry.runnable_actions(),
                self.load_registry.pending_completions(),
                resident,
                total,
                resident_bytes,
            );
        }
        self.cleanup_materialization_failures(true)?;
        let mut newly_ready = Vec::new();
        while let Some(continuation) = self.load_registry.pop_ready(now_ns)? {
            let transaction = self.continuations.mark_ready(continuation)?;
            #[cfg(test)]
            self.tick_trace
                .push(super::DriverTickEvent::ContinuationReady(continuation));
            newly_ready.push((continuation, transaction));
        }
        newly_ready.sort_unstable_by_key(|(continuation, transaction)| {
            (
                self.transaction_is_cancelling(*transaction),
                transaction.get(),
                continuation.get(),
            )
        });
        for (_, transaction) in newly_ready {
            self.enqueue_transaction(transaction);
        }
        Ok(())
    }

    pub(super) fn prepare_resume_lease(
        &mut self,
        continuation: ContinuationId,
    ) -> Result<crate::io::ResumeLease> {
        let dependencies = self
            .continuations
            .dependencies(continuation)
            .ok_or_else(|| Error::InvalidRequest {
                message: format!(
                    "continuation {} is not registered for resume",
                    continuation.get()
                ),
            })?
            .clone();
        self.load_registry
            .prepare_resume(continuation, &dependencies)
            .map_err(Error::from)
    }

    pub(super) fn finish_resume_lease(
        &mut self,
        continuation: ContinuationId,
        mut lease: crate::io::ResumeLease,
        disposition: ResumeDisposition,
        started_ns: u64,
    ) -> Result<()> {
        debug_assert_eq!(lease.continuation(), continuation);
        self.load_registry.finish_resume(
            &mut lease,
            disposition,
            started_ns,
            self.runtime_now_ns(),
        )?;
        if disposition == ResumeDisposition::Consumed {
            self.unregister_continuation(continuation);
        }
        Ok(())
    }

    pub(super) fn detach_registered_continuation(
        &mut self,
        continuation: ContinuationId,
        reason: CancellationReason,
    ) -> Result<()> {
        let now_ns = self.runtime_now_ns();
        let registry = &mut self.load_registry;
        self.continuations.detach(continuation, reason, |reason| {
            registry
                .detach_continuation(continuation, reason, now_ns)
                .map_err(Error::from)
        })
    }

    pub(super) fn unregister_continuation(&mut self, continuation: ContinuationId) {
        self.continuations.confirm_release(continuation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pending(transaction: u64, continuation: u64) -> PendingModelProgress {
        PendingModelProgress::new(
            ExecutionTransactionId::new(transaction).unwrap(),
            ContinuationId::new(continuation),
            ferrule_common::DependencySet::new([
                ferrule_common::LogicalDependency::operation_retired(
                    ferrule_common::OperationId::new(continuation),
                )
                .unwrap(),
            ])
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn registration_checks_before_attach_and_publishes_both_indexes_only_on_success() {
        let mut owner = ContinuationRegistry::default();
        let progress = pending(1, 10);
        let error = owner
            .register(&progress, || {
                Err(Error::Invariant {
                    message: "attachment failed".into(),
                })
            })
            .unwrap_err();
        assert!(error.to_string().contains("attachment failed"));
        assert!(owner.is_empty());
        assert_eq!(owner.transaction_count(), 0);
        owner.assert_consistent();
        owner.register(&progress, || Ok(())).unwrap();
        assert_eq!(
            owner.transaction(progress.continuation()),
            Some(progress.transaction())
        );
        assert_eq!(
            owner.continuation_for(progress.transaction()),
            Some(progress.continuation())
        );
        let replacement = pending(2, 10);
        assert!(
            owner
                .register(&replacement, || panic!(
                    "duplicate must not call physical attachment"
                ))
                .is_err()
        );
        assert_eq!(
            owner.transaction(progress.continuation()),
            Some(progress.transaction())
        );
        assert_eq!(owner.continuation_for(replacement.transaction()), None);
        owner.assert_consistent();
    }

    #[test]
    fn detach_errors_preserve_original_undo_and_ready_sibling_edges() {
        let mut owner = ContinuationRegistry::default();
        let first = pending(1, 10);
        let sibling = pending(1, 11);
        let other = pending(2, 12);
        for progress in [&first, &sibling, &other] {
            owner.register(progress, || Ok(())).unwrap();
            let transaction = owner.mark_ready(progress.continuation()).unwrap();
            owner.enqueue_transaction(transaction);
        }
        assert_eq!(owner.ready_snapshot_len(), 2);
        let id = first.continuation();
        for reason in [
            CancellationReason::ExternalRequest,
            CancellationReason::Superseded,
        ] {
            assert!(
                owner
                    .detach(id, reason, |reason| {
                        assert_eq!(reason, CancellationReason::ExternalRequest);
                        Err(Error::Invariant {
                            message: "detach not confirmed".into(),
                        })
                    })
                    .is_err()
            );
            assert_eq!(owner.transaction(id), Some(first.transaction()));
            assert_eq!(owner.ready_continuation_for(first.transaction()), Some(id));
            assert_eq!(
                owner.pending_detaches(),
                [(id, CancellationReason::ExternalRequest)]
            );
            owner.assert_consistent();
        }
        owner
            .detach(id, CancellationReason::Superseded, |reason| {
                assert_eq!(reason, CancellationReason::ExternalRequest);
                Ok(())
            })
            .unwrap();
        owner.assert_consistent();
        assert!(!owner.has_pending_detaches());
        assert_eq!(
            owner.ready_continuation_for(first.transaction()),
            Some(sibling.continuation())
        );
        assert_eq!(
            owner.ready_continuation_for(other.transaction()),
            Some(other.continuation())
        );
        owner
            .detach(id, CancellationReason::ExternalRequest, |_| {
                panic!("confirmed detach must not replay")
            })
            .unwrap();
        assert!(owner.mark_ready(id).is_err());
        assert_eq!(
            owner.ready_continuation_for(first.transaction()),
            Some(sibling.continuation())
        );
        assert_eq!(owner.pop_ready_transaction(), Some(first.transaction()));
        owner.enqueue_transaction(first.transaction());
        owner.remove_transaction_from_queue(other.transaction());
        assert_eq!(owner.pop_ready_transaction(), Some(first.transaction()));
        assert_eq!(owner.pop_ready_transaction(), None);
        owner.assert_consistent();
        for progress in [&sibling, &other] {
            owner
                .detach(
                    progress.continuation(),
                    CancellationReason::ExternalRequest,
                    |_| Ok(()),
                )
                .unwrap();
            owner.assert_consistent();
        }
        assert!(owner.is_empty());
        assert_eq!(owner.transaction_count(), 0);
    }
}
