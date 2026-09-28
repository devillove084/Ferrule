//! Prefix operations on the driver's existing authority.

use super::*;

pub(super) fn prefix_cache_error(operation: &str, error: impl std::fmt::Display) -> Error {
    Error::Invariant {
        message: format!("{operation}: {error}"),
    }
}

pub(super) struct ResidentPrefixPayload<S> {
    pub(super) model_state: S,
    pub(super) snapshot: KvPrefixSnapshot,
}

pub(super) struct PreparedPrefixAdmission {
    pub(super) pages: PreparedKvSnapshotFork,
    pub(super) matched_tokens: usize,
}

pub(super) struct PendingPrefixCleanup<S> {
    snapshot: Option<KvPrefixSnapshot>,
    retirement: Option<PendingKvRetirement>,
    model_state: Option<S>,
}

/// A lookup pin stays in the existing admission undo record until unpin ACK.
/// Only this module can consume or restore the underlying lease.
#[derive(Default)]
pub(super) struct PrefixAdmissionPin {
    lease: Option<crate::cache::PrefixCacheLease>,
}

impl PrefixAdmissionPin {
    pub(super) fn is_none(&self) -> bool {
        self.lease.is_none()
    }
    fn matched_tokens(&self) -> usize {
        self.lease.as_ref().expect("lookup pin").matched_tokens()
    }
}

/// Owns prefix payloads, session cache eligibility, and the FIFO cleanup
/// records that pair cache eviction with snapshot/model release.
pub(super) struct PrefixLifecycleState<S> {
    cache: RadixPrefixCache<ResidentPrefixPayload<S>>,
    cached_sessions: HashSet<SessionId>,
    pending_cleanups: VecDeque<PendingPrefixCleanup<S>>,
}

impl<S> PrefixLifecycleState<S> {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            cache: RadixPrefixCache::new(capacity),
            cached_sessions: HashSet::new(),
            pending_cleanups: VecDeque::new(),
        }
    }

    pub(super) fn capacity(&self) -> usize {
        self.cache.capacity()
    }
    pub(super) fn len(&self) -> usize {
        self.cache.len()
    }
    pub(super) fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
    #[cfg(test)]
    pub(super) fn used_capacity(&self) -> usize {
        self.cache.used_capacity()
    }
    #[cfg(test)]
    pub(super) fn contains_exact(&self, namespace: PrefixCacheNamespace, tokens: &[u32]) -> bool {
        self.cache.contains_exact(namespace, tokens)
    }
    #[cfg(test)]
    pub(super) fn front_cleanup(&self) -> Option<&PendingPrefixCleanup<S>> {
        self.pending_cleanups.front()
    }
    pub(super) fn clear_cached_sessions(&mut self) {
        self.cached_sessions.clear();
    }

    fn queue_removed(&mut self, removed: RemovedPrefixEntry<ResidentPrefixPayload<S>>) {
        self.queue_cleanup(PendingPrefixCleanup::from_payload(removed.into_payload()));
    }

    pub(super) fn insert_and_queue_cleanup(
        &mut self,
        namespace: PrefixCacheNamespace,
        tokens: &[u32],
        payload: ResidentPrefixPayload<S>,
        charge: usize,
    ) -> std::result::Result<(), crate::cache::PrefixCacheError> {
        match self.cache.insert(namespace, tokens, payload, charge) {
            Ok(outcome) => {
                let (_, replaced, evicted) = outcome.into_parts();
                if let Some(replaced) = replaced {
                    self.queue_removed(replaced);
                }
                for entry in evicted {
                    self.queue_removed(entry);
                }
                Ok(())
            }
            Err(error) => {
                let (kind, payload) = error.into_parts();
                self.queue_cleanup(PendingPrefixCleanup::from_payload(payload));
                Err(kind)
            }
        }
    }

    fn drain_into_cleanup(&mut self) -> Result<()> {
        let removed = self
            .cache
            .drain()
            .map_err(|error| prefix_cache_error("prefix cache drain", error))?;
        for entry in removed {
            self.queue_removed(entry);
        }
        Ok(())
    }

    fn evict_into_cleanup(&mut self) -> bool {
        let Some(removed) = self.cache.evict_lru() else {
            return false;
        };
        self.queue_removed(removed);
        true
    }

    fn lookup_pinned(
        &mut self,
        namespace: PrefixCacheNamespace,
        prompt: &[u32],
    ) -> Result<Option<PrefixAdmissionPin>> {
        self.cache
            .lookup_longest_prefix(namespace, prompt, PrefixLookupLimit::BeforeLastPromptToken)
            .map(|lease| lease.map(|lease| PrefixAdmissionPin { lease: Some(lease) }))
            .map_err(|error| prefix_cache_error("prefix lookup", error))
    }

    fn pinned_payload(&self, pin: &PrefixAdmissionPin) -> Result<&ResidentPrefixPayload<S>> {
        self.cache
            .payload(pin.lease.as_ref().expect("admission owns its lookup pin"))
            .map_err(|error| prefix_cache_error("pinned prefix payload", error))
    }

    fn unpin_admission(&mut self, pin: &mut PrefixAdmissionPin) -> Result<()> {
        let Some(lease) = pin.lease.take() else {
            return Ok(());
        };
        match self.cache.unpin(lease) {
            Ok(()) => Ok(()),
            Err(failure) => {
                let (error, lease) = failure.into_parts();
                pin.lease = Some(lease);
                Err(prefix_cache_error("prefix lookup unpin", error))
            }
        }
    }

    #[cfg(test)]
    pub(super) fn admission_pin_count(&self, pin: &PrefixAdmissionPin) -> Option<u32> {
        pin.lease
            .as_ref()
            .and_then(|lease| self.cache.entry_pin_count(lease.entry_id()))
    }

    pub(super) fn has_cached_session(&self, session: &SessionId) -> bool {
        self.cached_sessions.contains(session)
    }
    pub(super) fn mark_cached_session(&mut self, session: SessionId) {
        self.cached_sessions.insert(session);
    }
    pub(super) fn unmark_cached_session(&mut self, session: &SessionId) {
        self.cached_sessions.remove(session);
    }
    pub(super) fn cached_session_count(&self) -> usize {
        self.cached_sessions.len()
    }
    pub(super) fn cached_sessions_empty(&self) -> bool {
        self.cached_sessions.is_empty()
    }
    pub(super) fn queue_cleanup(&mut self, cleanup: PendingPrefixCleanup<S>) {
        self.pending_cleanups.push_back(cleanup);
    }
    pub(super) fn pop_cleanup(&mut self) -> Option<PendingPrefixCleanup<S>> {
        self.pending_cleanups.pop_front()
    }
    pub(super) fn retry_cleanup(&mut self, cleanup: PendingPrefixCleanup<S>) {
        self.pending_cleanups.push_front(cleanup);
    }
    pub(super) fn pending_cleanup_count(&self) -> usize {
        self.pending_cleanups.len()
    }
    pub(super) fn pending_cleanups_empty(&self) -> bool {
        self.pending_cleanups.is_empty()
    }
}

impl<S> PendingPrefixCleanup<S> {
    #[cfg(test)]
    pub(super) fn model_state(&self) -> Option<&S> {
        self.model_state.as_ref()
    }

    pub(super) fn from_payload(payload: ResidentPrefixPayload<S>) -> Self {
        Self {
            snapshot: Some(payload.snapshot),
            retirement: None,
            model_state: Some(payload.model_state),
        }
    }

    pub(super) fn model_only(model_state: S) -> Self {
        Self {
            snapshot: None,
            retirement: None,
            model_state: Some(model_state),
        }
    }
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner,
    C: SequenceSlotPool,
{
    pub(super) fn prefix_cache_namespace(&self) -> Option<PrefixCacheNamespace> {
        if self.prefix.capacity() == 0 {
            return None;
        }
        let manager = self.page_manager.as_ref()?;
        let placement = self.materialization_resolver.placement();
        let plan = self.executor.runner().prefix_cache_plan_identity();
        if plan == 0 {
            return None;
        }
        Some(PrefixCacheNamespace::for_placement(
            placement.model().get(),
            placement.backend().get(),
            placement.device().get(),
            plan,
            manager.owner_identity(),
        ))
    }

    pub(super) fn progress_prefix_cleanup(
        &mut self,
        mut cleanup: PendingPrefixCleanup<R::SequenceState>,
    ) -> std::result::Result<(), (Error, PendingPrefixCleanup<R::SequenceState>)> {
        if self.has_live_transactions() {
            return Err((
                Error::InvalidRequest {
                    message: "cannot release a cached prefix while packed transactions are live"
                        .into(),
                },
                cleanup,
            ));
        }
        if let Some(snapshot) = cleanup.snapshot {
            let retirement = match self.page_manager.as_mut() {
                Some(manager) => match manager.release_prefix_snapshot(snapshot) {
                    Ok(retirement) => retirement,
                    Err(error) => return Err((error, cleanup)),
                },
                None => {
                    return Err((
                        Error::Invariant {
                            message: "cached prefix snapshot has no authoritative page manager"
                                .into(),
                        },
                        cleanup,
                    ));
                }
            };
            cleanup.snapshot = None;
            cleanup.retirement = Some(PendingKvRetirement::BackendRelease(retirement));
        }
        if let Some(retirement) = cleanup.retirement.take()
            && let Err((error, retirement)) = self.progress_kv_retirement(retirement)
        {
            cleanup.retirement = Some(retirement);
            return Err((error, cleanup));
        }
        if let Some(model_state) = cleanup.model_state.take()
            && let Err(failure) = self.executor.try_release_sequence_state(model_state)
        {
            let (error, model_state) = failure.into_parts();
            cleanup.model_state = Some(model_state);
            return Err((error.into(), cleanup));
        }
        Ok(())
    }

    pub(super) fn progress_prefix_cleanups(&mut self) -> Result<()> {
        if self.has_live_transactions() {
            return Ok(());
        }
        while let Some(cleanup) = self.prefix.pop_cleanup() {
            if let Err((error, cleanup)) = self.progress_prefix_cleanup(cleanup) {
                self.prefix.retry_cleanup(cleanup);
                return Err(error);
            }
        }
        Ok(())
    }

    pub(super) fn drain_prefix_cache(&mut self) -> Result<()> {
        self.prefix.drain_into_cleanup()?;
        self.progress_prefix_cleanups()
    }

    pub(super) fn evict_prefixes_for_kv_pages(&mut self, required: usize) -> Result<()> {
        if required == 0 || self.available_kv_page_credits() >= required {
            return Ok(());
        }
        if self.has_live_transactions() {
            return Ok(());
        }
        while self.available_kv_page_credits() < required {
            if !self.prefix.evict_into_cleanup() {
                break;
            }
            self.progress_prefix_cleanups()?;
        }
        Ok(())
    }

    pub fn prefix_cache_stats(&self) -> &ResidentPrefixCacheStats {
        self.observability.prefix_cache_stats()
    }

    pub fn prefix_hits(&self) -> usize {
        self.observability.prefix_hits()
    }

    pub fn prefix_misses(&self) -> usize {
        self.observability.prefix_misses()
    }

    pub(super) fn prepare_prefix_admission(
        &mut self,
        prepared: &mut PreparedWaitingAdmission,
        target_slot: StateSlot,
    ) -> Result<Option<PreparedPrefixAdmission>> {
        let Some(namespace) = self.prefix_cache_namespace() else {
            return Ok(None);
        };
        if self.has_live_transactions()
            || !prepared.is_fresh_prompt()
            || prepared.prompt_tokens().is_empty()
        {
            return Ok(None);
        }
        let prompt = prepared.prompt_tokens().to_vec();
        let lease = self.prefix.lookup_pinned(namespace, &prompt)?;
        let Some(lease) = lease else {
            return Ok(None);
        };
        let matched_tokens = lease.matched_tokens();
        let session_id = prepared.session_id();
        self.sessions.begin_cleanup(
            session_id,
            PendingSequenceCleanup::Admission {
                model_state: None,
                slot: None,
                lease,
                errors: Vec::new(),
            },
        );
        let PendingSequenceCleanup::Admission { lease, .. } =
            self.sessions.cleanup(&session_id).unwrap()
        else {
            unreachable!()
        };
        let payload = self.prefix.pinned_payload(lease)?;
        let snapshot = payload.snapshot;
        let state = self
            .executor
            .fork_sequence_state_from(&payload.model_state, matched_tokens)?;
        let PendingSequenceCleanup::Admission { model_state, .. } = self
            .sessions
            .cleanup_mut(&session_id)
            .expect("prefix owner")
        else {
            unreachable!()
        };
        *model_state = Some(state);
        #[cfg(test)]
        self.fail_stage("prefix prepare")?;
        let pages = self
            .page_manager
            .as_ref()
            .ok_or_else(|| Error::Invariant {
                message: "prefix cache hit has no authoritative page manager".into(),
            })?
            .prepare_fork_prefix_snapshot(snapshot, target_slot, 0, matched_tokens)?;
        prepared.apply_committed_prefix(matched_tokens)?;
        self.unpin_admission(session_id)?;
        Ok(Some(PreparedPrefixAdmission {
            pages,
            matched_tokens,
        }))
    }

    pub(super) fn unpin_admission(&mut self, session_id: SessionId) -> Result<()> {
        #[cfg(test)]
        self.fail_stage("prefix unpin")?;
        match self.sessions.cleanup_mut(&session_id) {
            Some(PendingSequenceCleanup::Admission { lease, .. }) => {
                self.prefix.unpin_admission(lease)
            }
            _ => Ok(()),
        }
    }

    pub(super) fn capture_committed_prefill_prefixes(
        &mut self,
        action: &SchedulerAction,
    ) -> Result<()> {
        match action {
            SchedulerAction::Execute { prefills, .. } => {
                for prefill in prefills {
                    self.capture_committed_prefill_prefix(prefill)?;
                }
            }
            SchedulerAction::PrefillChunk(prefill) => {
                self.capture_committed_prefill_prefix(prefill)?;
            }
            SchedulerAction::DecodeBatch(_)
            | SchedulerAction::Finish { .. }
            | SchedulerAction::Cancel { .. } => {}
        }
        Ok(())
    }
}
