//! Sessions operations on the driver's existing authority.

use super::*;

/// The single owner of logical session custody.
///
/// The driver remains the authority for transaction scheduling and the page
/// manager remains the authority for physical KV. This component only keeps
/// the paired logical session relations together so callers cannot update one
/// side without the other.
pub(super) struct SessionCustody<S> {
    sequence_states: HashMap<SessionId, S>,
    page_slots: HashMap<SessionId, StateSlot>,
    retained_sessions: HashMap<SessionId, usize>,
    suspended_sequences: HashMap<SessionId, SuspendedDriverSequence<S>>,
    session_owner: HashMap<SessionId, ExecutionTransactionId>,
    pending_sequence_cleanups: HashMap<SessionId, PendingSequenceCleanup<S>>,
}

impl<S> Default for SessionCustody<S> {
    fn default() -> Self {
        Self {
            sequence_states: HashMap::new(),
            page_slots: HashMap::new(),
            retained_sessions: HashMap::new(),
            suspended_sequences: HashMap::new(),
            session_owner: HashMap::new(),
            pending_sequence_cleanups: HashMap::new(),
        }
    }
}

impl<S> SessionCustody<S> {
    pub(super) fn publish_admitted(
        &mut self,
        session: SessionId,
        state: S,
        slot: Option<StateSlot>,
    ) {
        debug_assert!(!self.is_owned(&session) && !self.is_suspended(&session));
        let previous = self.sequence_states.insert(session, state);
        debug_assert!(
            previous.is_none(),
            "prepared admission model target must remain absent"
        );
        if let Some(slot) = slot {
            let previous = self.page_slots.insert(session, slot);
            debug_assert!(
                previous.is_none(),
                "prepared admission page target must remain absent"
            );
        }
    }

    #[cfg(test)]
    pub(super) fn assert_consistent(&self) {
        for id in self.session_owner.keys() {
            assert!(!self.sequence_states.contains_key(id));
            assert!(!self.suspended_sequences.contains_key(id));
        }
        for (id, suspended) in &self.suspended_sequences {
            assert_eq!(*id, suspended.schedule.session_id());
            assert!(!self.sequence_states.contains_key(id));
            assert!(!self.page_slots.contains_key(id));
        }
    }

    pub(super) fn page_slot(&self, session: &SessionId) -> Option<&StateSlot> {
        self.page_slots.get(session)
    }
    pub(super) fn has_page_slot(&self, session: &SessionId) -> bool {
        self.page_slots.contains_key(session)
    }
    pub(super) fn page_slot_ids(&self) -> impl Iterator<Item = SessionId> + '_ {
        self.page_slots.keys().copied()
    }
    #[cfg(test)]
    pub(super) fn page_slot_count(&self) -> usize {
        self.page_slots.len()
    }
    pub(super) fn retain_page_slot(
        &mut self,
        session: SessionId,
        slot: StateSlot,
    ) -> Option<StateSlot> {
        self.page_slots.insert(session, slot)
    }

    fn begin_suspension(
        &mut self,
        session: SessionId,
        slot: StateSlot,
        schedule: SuspendedSequenceSchedule,
    ) {
        let model_state = self
            .sequence_states
            .remove(&session)
            .expect("validated model state");
        self.page_slots.remove(&session);
        self.suspended_sequences.insert(
            session,
            SuspendedDriverSequence {
                model_state,
                page_slot: slot,
                kv_state: None,
                schedule,
                phase: SuspendedDriverPhase::PreemptLogical,
                rolling_back: false,
                errors: Vec::new(),
            },
        );
    }

    pub(super) fn suspend_into_cleanup(&mut self, session: SessionId) -> SuspendedSequenceSchedule {
        let pending = self
            .suspended_sequences
            .remove(&session)
            .expect("suspended session identity was collected above");
        self.pending_sequence_cleanups.insert(
            session,
            PendingSequenceCleanup::Suspended {
                kv_state: pending.kv_state,
                retirement: None,
                model_state: Some(pending.model_state),
            },
        );
        pending.schedule
    }

    pub(super) fn claim_transaction_sessions(
        &mut self,
        scheduler: &mut ResidentScheduler,
        transaction: ExecutionTransactionId,
        session_ids: &[SessionId],
    ) -> Result<(Vec<SuspendedSequenceSchedule>, Vec<S>)> {
        let mut unique = HashSet::with_capacity(session_ids.len());
        for session_id in session_ids {
            if !unique.insert(*session_id) {
                return Err(Error::Invariant {
                    message: format!(
                        "transaction {transaction:?} contains duplicate session {session_id:?}"
                    ),
                });
            }
            if let Some(owner) = self.owner(session_id) {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "session {session_id:?} is already owned by transaction {owner:?}"
                    ),
                });
            }
        }

        let mut schedules = Vec::with_capacity(session_ids.len());
        let mut states = Vec::with_capacity(session_ids.len());
        for session_id in session_ids {
            let schedule = match scheduler.suspend_sequence(*session_id) {
                Ok(schedule) => schedule,
                Err(error) => {
                    self.restore_transaction_sessions(scheduler, schedules, states)?;
                    return Err(error);
                }
            };
            let Some(state) = self.take_sequence_state(session_id) else {
                scheduler.restore_suspended(schedule)?;
                self.restore_transaction_sessions(scheduler, schedules, states)?;
                return Err(Error::Invariant {
                    message: format!("session {session_id:?} has no model sequence state"),
                });
            };
            self.assign_owner(*session_id, transaction);
            schedules.push(schedule);
            states.push(state);
        }
        Ok((schedules, states))
    }

    pub(super) fn restore_transaction_sessions(
        &mut self,
        scheduler: &mut ResidentScheduler,
        mut schedules: Vec<SuspendedSequenceSchedule>,
        mut states: Vec<S>,
    ) -> Result<()> {
        self.progress_transaction_session_restore(scheduler, &mut schedules, &mut states)
    }

    pub(super) fn progress_transaction_session_restore(
        &mut self,
        scheduler: &mut ResidentScheduler,
        schedules: &mut Vec<SuspendedSequenceSchedule>,
        states: &mut Vec<S>,
    ) -> Result<()> {
        if schedules.len() != states.len() {
            return Err(Error::Invariant {
                message: format!(
                    "transaction schedule/state mismatch: schedules={} states={}",
                    schedules.len(),
                    states.len()
                ),
            });
        }
        let mut unique = HashSet::with_capacity(schedules.len());
        for schedule in schedules.iter() {
            let session_id = schedule.session_id();
            if !unique.insert(session_id) {
                return Err(Error::Invariant {
                    message: format!(
                        "transaction restore contains duplicate session {session_id:?}"
                    ),
                });
            }
            if self.contains_sequence_state(&session_id) {
                return Err(Error::Invariant {
                    message: format!("session {session_id:?} model state was already published"),
                });
            }
            if !self.is_owned(&session_id) {
                return Err(Error::Invariant {
                    message: format!(
                        "session {session_id:?} lost transaction ownership before restoration"
                    ),
                });
            }
            if scheduler.active_sequence(session_id).is_some() {
                return Err(Error::Invariant {
                    message: format!(
                        "cannot restore already-active resident session {session_id:?}"
                    ),
                });
            }
        }
        while let Some(schedule) = schedules.first() {
            let session_id = schedule.session_id();
            scheduler
                .restore_suspended(schedule.clone())
                .expect("transaction session restore was preflighted");
            let schedule = schedules.remove(0);
            debug_assert_eq!(schedule.session_id(), session_id);
            let state = states.remove(0);
            self.release_owner(&session_id);
            let previous = self.publish_sequence_state(session_id, state);
            debug_assert!(
                previous.is_none(),
                "transaction session restore was preflighted"
            );
        }
        Ok(())
    }

    pub(super) fn advance_session_transition<R: MultiSessionRunner<SequenceState = S>>(
        &mut self,
        scheduler: &mut ResidentScheduler,
        page_manager: &mut Option<KvPageManager>,
        executor: &mut NativeMultiSessionExecutor<R>,
        #[cfg(test)] stage_faults: &mut VecDeque<&'static str>,
        session_id: SessionId,
    ) -> Result<()> {
        use SuspendedDriverPhase::*;
        loop {
            let phase = self.suspended(&session_id).unwrap().phase;
            if phase == Suspended {
                return Ok(());
            }
            if phase == Quarantined {
                return Err(Error::Invariant {
                    message: format!(
                        "session {session_id:?} physical completion unknown; custody quarantined: {:?}",
                        self.suspended(&session_id).unwrap().errors
                    ),
                });
            }
            let result = (|| -> Result<()> {
                #[cfg(test)]
                Self::fail_stage(
                    stage_faults,
                    match phase {
                        PreemptLogical => "preempt logical",
                        PreemptPhysical => "preempt physical",
                        RestorePhysical => "restore physical",
                        RestoreLogical => "restore logical",
                        RestoreScheduler => "restore scheduler",
                        RollbackLogical => "rollback logical",
                        RollbackPhysical => "rollback physical",
                        _ => unreachable!(),
                    },
                )?;
                match phase {
                    PreemptLogical | RollbackLogical => {
                        let slot = self.suspended(&session_id).unwrap().page_slot;
                        let state = page_manager
                            .as_mut()
                            .ok_or_else(|| Error::Invariant {
                                message: "KvPageManager was removed while suspended".into(),
                            })?
                            .preempt_sequence(slot)?;
                        let pending = self.suspended_mut(&session_id).expect("transition owner");
                        pending.kv_state = Some(state);
                        pending.phase = if phase == PreemptLogical {
                            PreemptPhysical
                        } else {
                            RollbackPhysical
                        };
                    }
                    PreemptPhysical | RollbackPhysical | RestorePhysical => {
                        let pages = self
                            .suspended(&session_id)
                            .unwrap()
                            .kv_state
                            .as_ref()
                            .expect("preempted KV")
                            .evicted_pages()
                            .to_vec();
                        self.suspended_mut(&session_id)
                            .expect("transition owner")
                            .phase = Quarantined;
                        let physical = if phase == RestorePhysical {
                            executor.restore_kv_pages(&pages)
                        } else {
                            executor.preempt_kv_pages(&pages)
                        };
                        let pending = self.suspended_mut(&session_id).expect("transition owner");
                        pending.phase = Quarantined;
                        match physical {
                            Ok(()) => {
                                pending.phase = if phase == RestorePhysical {
                                    RestoreLogical
                                } else {
                                    Suspended
                                }
                            }
                            Err(error) => {
                                pending.errors.push(error.to_string());
                                return Err(error);
                            }
                        }
                    }
                    RestoreLogical => {
                        let (slot, state) = {
                            let pending = self.suspended(&session_id).unwrap();
                            (
                                pending.page_slot,
                                pending.kv_state.as_ref().expect("preempted KV").clone(),
                            )
                        };
                        page_manager
                            .as_mut()
                            .ok_or_else(|| Error::Invariant {
                                message: "KvPageManager was removed while suspended".into(),
                            })?
                            .restore_sequence(slot, state)?;
                        let pending = self.suspended_mut(&session_id).expect("transition owner");
                        pending.kv_state = None;
                        pending.phase = RestoreScheduler;
                    }
                    RestoreScheduler => {
                        let schedule = self.suspended(&session_id).unwrap().schedule.clone();
                        scheduler.restore_suspended(schedule)?;
                        let pending = self
                            .remove_suspended(&session_id)
                            .expect("transition owner");
                        self.page_slots.insert(session_id, pending.page_slot);
                        self.publish_sequence_state(session_id, pending.model_state);
                    }
                    Suspended | Quarantined => unreachable!(),
                }
                Ok(())
            })();
            if let Err(error) = result {
                let pending = self
                    .suspended_mut(&session_id)
                    .expect("failed transition retains owner");
                pending.errors.push(error.to_string());
                if pending.phase == Quarantined || pending.rolling_back {
                    return Err(error);
                }
                pending.rolling_back = true;
                pending.phase = match phase {
                    PreemptLogical => RestoreScheduler,
                    PreemptPhysical => RestoreLogical,
                    RestorePhysical => Suspended,
                    RestoreLogical => RollbackPhysical,
                    RestoreScheduler => RollbackLogical,
                    _ => phase,
                };
                let cleanup = self.advance_session_transition(
                    scheduler,
                    page_manager,
                    executor,
                    #[cfg(test)]
                    stage_faults,
                    session_id,
                );
                return Err(Error::with_cleanup(
                    "session transition rollback",
                    error,
                    cleanup,
                ));
            }
            if !self.is_suspended(&session_id) {
                return Ok(());
            }
        }
    }

    pub(super) fn sequence_state(&self, session_id: &SessionId) -> Option<&S> {
        self.sequence_states.get(session_id)
    }

    pub(super) fn contains_sequence_state(&self, session_id: &SessionId) -> bool {
        self.sequence_states.contains_key(session_id)
    }

    fn take_sequence_state(&mut self, session_id: &SessionId) -> Option<S> {
        self.sequence_states.remove(session_id)
    }

    fn publish_sequence_state(&mut self, session_id: SessionId, state: S) -> Option<S> {
        self.sequence_states.insert(session_id, state)
    }

    pub(super) fn sequence_state_count(&self) -> usize {
        self.sequence_states.len()
    }

    pub(super) fn sequence_state_ids(&self) -> impl Iterator<Item = SessionId> + '_ {
        self.sequence_states.keys().copied()
    }

    pub(super) fn retained_position(&self, session_id: &SessionId) -> Option<usize> {
        self.retained_sessions.get(session_id).copied()
    }

    pub(super) fn retained_position_mut(&mut self, session_id: &SessionId) -> Option<&mut usize> {
        self.retained_sessions.get_mut(session_id)
    }

    pub(super) fn is_retained(&self, session_id: &SessionId) -> bool {
        self.retained_sessions.contains_key(session_id)
    }

    pub(super) fn retain(&mut self, session_id: SessionId) {
        self.retained_sessions.entry(session_id).or_insert(0);
    }

    pub(super) fn set_retained_position(&mut self, session_id: SessionId, position: usize) {
        self.retained_sessions.insert(session_id, position);
    }

    pub(super) fn unretain(&mut self, session_id: &SessionId) -> Option<usize> {
        self.retained_sessions.remove(session_id)
    }

    pub(super) fn retained_ids(&self) -> impl Iterator<Item = SessionId> + '_ {
        self.retained_sessions.keys().copied()
    }

    pub(super) fn retained_count(&self) -> usize {
        self.retained_sessions.len()
    }

    pub(super) fn is_suspended(&self, session_id: &SessionId) -> bool {
        self.suspended_sequences.contains_key(session_id)
    }

    pub(super) fn suspended(&self, session_id: &SessionId) -> Option<&SuspendedDriverSequence<S>> {
        self.suspended_sequences.get(session_id)
    }

    fn suspended_mut(&mut self, session_id: &SessionId) -> Option<&mut SuspendedDriverSequence<S>> {
        self.suspended_sequences.get_mut(session_id)
    }

    pub(super) fn suspended_ids(&self) -> impl Iterator<Item = SessionId> + '_ {
        self.suspended_sequences.keys().copied()
    }

    pub(super) fn suspended_count(&self) -> usize {
        self.suspended_sequences.len()
    }

    fn remove_suspended(&mut self, session_id: &SessionId) -> Option<SuspendedDriverSequence<S>> {
        self.suspended_sequences.remove(session_id)
    }

    pub(super) fn owner(&self, session_id: &SessionId) -> Option<ExecutionTransactionId> {
        self.session_owner.get(session_id).copied()
    }

    pub(super) fn is_owned(&self, session_id: &SessionId) -> bool {
        self.session_owner.contains_key(session_id)
    }

    pub(super) fn owner_ids(&self) -> impl Iterator<Item = SessionId> + '_ {
        self.session_owner.keys().copied()
    }

    pub(super) fn owner_count(&self) -> usize {
        self.session_owner.len()
    }

    fn assign_owner(
        &mut self,
        session_id: SessionId,
        transaction: ExecutionTransactionId,
    ) -> Option<ExecutionTransactionId> {
        self.session_owner.insert(session_id, transaction)
    }

    fn release_owner(&mut self, session_id: &SessionId) -> Option<ExecutionTransactionId> {
        self.session_owner.remove(session_id)
    }

    pub(super) fn cleanup(&self, session_id: &SessionId) -> Option<&PendingSequenceCleanup<S>> {
        self.pending_sequence_cleanups.get(session_id)
    }

    pub(super) fn cleanup_mut(
        &mut self,
        session_id: &SessionId,
    ) -> Option<&mut PendingSequenceCleanup<S>> {
        self.pending_sequence_cleanups.get_mut(session_id)
    }

    pub(super) fn has_cleanup(&self, session_id: &SessionId) -> bool {
        self.pending_sequence_cleanups.contains_key(session_id)
    }

    pub(super) fn cleanup_ids(&self) -> impl Iterator<Item = SessionId> + '_ {
        self.pending_sequence_cleanups.keys().copied()
    }

    pub(super) fn cleanup_count(&self) -> usize {
        self.pending_sequence_cleanups.len()
    }

    pub(super) fn begin_cleanup(
        &mut self,
        session_id: SessionId,
        cleanup: PendingSequenceCleanup<S>,
    ) -> Option<PendingSequenceCleanup<S>> {
        self.pending_sequence_cleanups.insert(session_id, cleanup)
    }

    pub(super) fn take_cleanup(
        &mut self,
        session_id: &SessionId,
    ) -> Option<PendingSequenceCleanup<S>> {
        self.pending_sequence_cleanups.remove(session_id)
    }

    pub(super) fn clear_retained(&mut self) {
        self.retained_sessions.clear();
    }

    pub(super) fn defer_cleanup(&mut self, session_id: SessionId) {
        self.pending_sequence_cleanups
            .entry(session_id)
            .or_insert(PendingSequenceCleanup::Deferred);
    }

    pub(super) fn cleanups(
        &self,
    ) -> impl Iterator<Item = (&SessionId, &PendingSequenceCleanup<S>)> {
        self.pending_sequence_cleanups.iter()
    }

    pub(super) fn suspended_entries(
        &self,
    ) -> impl Iterator<Item = (&SessionId, &SuspendedDriverSequence<S>)> {
        self.suspended_sequences.iter()
    }

    #[cfg(test)]
    pub(super) fn sequence_state_mut(&mut self, session_id: &SessionId) -> Option<&mut S> {
        self.sequence_states.get_mut(session_id)
    }

    #[cfg(test)]
    fn fail_stage(faults: &mut VecDeque<&'static str>, stage: &'static str) -> Result<()> {
        if faults.front() == Some(&stage) {
            faults.pop_front();
            return Err(Error::Invariant {
                message: format!("injected {stage}"),
            });
        }
        Ok(())
    }
}

/// Synchronous resident driver over scheduler + KV + native multi-session executor.
///
/// This is the end-to-end resident workload loop in runtime. It remains concrete
/// and synchronous: no async frontend, no trait-object framework, and no concrete
/// model ownership. The driver connects request admission, scheduled execution,
/// output policy, event delivery, and resource/session lifecycle through typed
/// runtime values.
///
/// The driver requires `R: MultiSessionRunner`, so each sequence's state is
/// explicitly managed and swapped into the runner during execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SuspendedDriverPhase {
    PreemptLogical,
    PreemptPhysical,
    Suspended,
    RestorePhysical,
    RestoreLogical,
    RestoreScheduler,
    RollbackLogical,
    RollbackPhysical,
    Quarantined,
}

pub(super) struct SuspendedDriverSequence<S> {
    pub(super) model_state: S,
    pub(super) page_slot: StateSlot,
    pub(super) kv_state: Option<PreemptedKvState>,
    pub(super) schedule: SuspendedSequenceSchedule,
    pub(super) phase: SuspendedDriverPhase,
    // Compensation proceeds to its original endpoint before a new operation.
    pub(super) rolling_back: bool,
    pub(super) errors: Vec<String>,
}

pub(super) enum PendingSequenceCleanup<S> {
    Deferred,
    Admission {
        model_state: Option<S>,
        slot: Option<crate::scheduling::KvHandle>,
        lease: prefix::PrefixAdmissionPin,
        errors: Vec<String>,
    },
    Suspended {
        kv_state: Option<PreemptedKvState>,
        retirement: Option<PendingKvRetirement>,
        model_state: Option<S>,
    },
    Owned {
        retirement: Option<PendingKvRetirement>,
        model_state: Option<S>,
    },
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner,
    C: SequenceSlotPool,
{
    pub fn suspended_len(&self) -> usize {
        self.sessions.suspended_count()
    }

    /// Suspend one active session, retaining all custody before changing KV.
    pub fn preempt_session(&mut self, session_id: SessionId) -> Result<()> {
        self.ensure_no_suspended_execution("preempt a session")?;
        if self.sessions.is_suspended(&session_id) {
            return Err(Error::InvalidRequest {
                message: format!("session {session_id:?} is already suspended"),
            });
        }
        let slot = *self
            .sessions
            .page_slot(&session_id)
            .ok_or_else(|| Error::InvalidRequest {
                message: format!("session {session_id:?} has no authoritative page slot"),
            })?;
        if self.page_manager.is_none() || !self.sessions.contains_sequence_state(&session_id) {
            return Err(Error::InvalidRequest {
                message:
                    "session preemption requires model state and an authoritative KvPageManager"
                        .into(),
            });
        }
        #[cfg(test)]
        self.fail_stage("preempt scheduler")?;
        let schedule = self.scheduler.suspend_sequence(session_id)?;
        self.sessions.begin_suspension(session_id, slot, schedule);
        self.advance_session_transition(session_id)
    }

    /// Resume compensation first, without replaying completed physical stages.
    pub fn restore_session(&mut self, session_id: SessionId) -> Result<()> {
        self.ensure_no_suspended_execution("restore a session")?;
        let pending =
            self.sessions
                .suspended_mut(&session_id)
                .ok_or_else(|| Error::InvalidRequest {
                    message: format!("session {session_id:?} is not suspended"),
                })?;
        if pending.phase == SuspendedDriverPhase::Suspended {
            pending.phase = SuspendedDriverPhase::RestorePhysical;
            pending.rolling_back = false;
        }
        self.advance_session_transition(session_id)?;
        if let Some(pending) = self.sessions.suspended_mut(&session_id) {
            if pending.phase == SuspendedDriverPhase::Suspended {
                pending.phase = SuspendedDriverPhase::RestorePhysical;
                pending.rolling_back = false;
                return self.advance_session_transition(session_id);
            }
        }
        Ok(())
    }

    pub(super) fn advance_session_transition(&mut self, session_id: SessionId) -> Result<()> {
        self.sessions.advance_session_transition(
            &mut self.scheduler,
            &mut self.page_manager,
            &mut self.executor,
            #[cfg(test)]
            &mut self.stage_faults,
            session_id,
        )
    }

    /// Keep a session's model and KV state resident after each request finishes.
    /// Subsequent requests with this explicit session ID append at the last
    /// committed position instead of creating a fresh sequence.
    pub fn retain_session(&mut self, session_id: SessionId) -> Result<()> {
        if self.shutting_down || self.admission_closed {
            return Err(RuntimeAdmissionError::Closed.into());
        }
        if !self.scheduler.is_idle() || self.sessions.is_suspended(&session_id) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "cannot retain session {session_id:?} while scheduler work is active or suspended"
                ),
            });
        }
        if self.session_has_pending_ownership(session_id)
            || self.scheduler.contains_session_identity(session_id)
            || self
                .request_identities
                .values()
                .any(|record| record.session_id == session_id)
        {
            return Err(RuntimeAdmissionError::SessionBusy { session_id }.into());
        }
        let sessions = self.owned_admission_sessions();
        if !sessions.contains(&session_id)
            && sessions.len() >= self.admission_options.max_session_identities
        {
            return Err(RuntimeAdmissionError::Capacity {
                resource: RuntimeAdmissionResource::SessionIdentities,
                held: sessions.len(),
                limit: self.admission_options.max_session_identities,
            }
            .into());
        }
        self.sessions.retain(session_id);
        Ok(())
    }

    /// Return the committed position of an explicitly retained session.
    pub fn retained_session_position(&self, session_id: SessionId) -> Option<usize> {
        self.sessions.retained_position(&session_id)
    }

    /// Release an idle retained session and all model/KV state owned by it.
    pub fn release_session(&mut self, session_id: SessionId) -> Result<()> {
        self.ensure_no_suspended_execution("release a session")?;
        if !self.scheduler.is_idle() || self.sessions.is_suspended(&session_id) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "cannot release session {session_id:?} while scheduler work is active or suspended"
                ),
            });
        }
        self.release_sequence_state(session_id)?;
        self.sessions.unretain(&session_id);
        self.reap_request_identities();
        Ok(())
    }

    /// Reset an idle retained session to an empty position while preserving its
    /// retained lifecycle registration for future request turns.
    pub fn reset_session(&mut self, session_id: SessionId) -> Result<()> {
        if !self.sessions.is_retained(&session_id) {
            return Err(Error::InvalidRequest {
                message: format!("session {session_id:?} is not retained"),
            });
        }
        self.release_session(session_id)?;
        self.sessions.set_retained_position(session_id, 0);
        Ok(())
    }

    pub(super) fn claim_transaction_sessions(
        &mut self,
        transaction: ExecutionTransactionId,
        session_ids: &[SessionId],
    ) -> Result<(Vec<SuspendedSequenceSchedule>, Vec<R::SequenceState>)> {
        self.sessions
            .claim_transaction_sessions(&mut self.scheduler, transaction, session_ids)
    }

    pub(super) fn progress_transaction_session_restore(
        &mut self,
        schedules: &mut Vec<SuspendedSequenceSchedule>,
        states: &mut Vec<R::SequenceState>,
    ) -> Result<()> {
        #[cfg(test)]
        self.fail_stage("transaction session restore")?;
        self.sessions
            .progress_transaction_session_restore(&mut self.scheduler, schedules, states)
    }

    /// Preserve a retained session at its committed position, or release a
    /// normal one immediately after the request turn finishes.
    pub(super) fn finalize_sequence_state(
        &mut self,
        session_id: SessionId,
        position: usize,
    ) -> Result<()> {
        self.prefix.unmark_cached_session(&session_id);
        if let Some(retained_position) = self.sessions.retained_position_mut(&session_id) {
            *retained_position = position;
            Ok(())
        } else {
            self.release_sequence_state(session_id)
        }
    }

    /// Release sequence/KV ownership now, or retain it in a driver-owned cleanup
    /// record until every packed backend transaction is quiescent.
    pub(super) fn release_sequence_state(&mut self, session_id: SessionId) -> Result<()> {
        self.output.discard_uncommitted(session_id);
        self.prefix.unmark_cached_session(&session_id);
        self.sessions.defer_cleanup(session_id);
        if self.has_live_transactions() {
            return Ok(());
        }
        self.progress_sequence_cleanup(session_id)
    }

    pub(super) fn progress_sequence_cleanup(&mut self, session_id: SessionId) -> Result<()> {
        if self.has_live_transactions() {
            return Ok(());
        }
        if matches!(
            self.sessions.cleanup(&session_id),
            Some(PendingSequenceCleanup::Admission { .. })
        ) {
            return self.cleanup_prepared_admission(session_id);
        }
        let Some(cleanup) = self.sessions.take_cleanup(&session_id) else {
            return Ok(());
        };
        let cleanup = match cleanup {
            PendingSequenceCleanup::Admission { .. } => unreachable!("handled before removal"),
            PendingSequenceCleanup::Deferred => {
                let retirement = match self.sessions.page_slot(&session_id).copied() {
                    Some(slot) => {
                        let Some(manager) = self.page_manager.as_mut() else {
                            self.sessions
                                .begin_cleanup(session_id, PendingSequenceCleanup::Deferred);
                            return Err(Error::Invariant {
                                message: format!(
                                    "session {session_id:?} has a page slot without an authoritative page manager"
                                ),
                            });
                        };
                        let retirement = match manager.free_sequence_pages(slot) {
                            Ok(retirement) => retirement,
                            Err(error) => {
                                self.sessions
                                    .begin_cleanup(session_id, PendingSequenceCleanup::Deferred);
                                return Err(error);
                            }
                        };
                        self.sessions.page_slots.remove(&session_id);
                        Some(PendingKvRetirement::BackendRelease(retirement))
                    }
                    None => None,
                };
                PendingSequenceCleanup::Owned {
                    retirement,
                    model_state: self.sessions.take_sequence_state(&session_id),
                }
            }
            cleanup @ PendingSequenceCleanup::Suspended { .. }
            | cleanup @ PendingSequenceCleanup::Owned { .. } => cleanup,
        };
        let cleanup = if let PendingSequenceCleanup::Suspended {
            mut kv_state,
            mut retirement,
            model_state,
        } = cleanup
        {
            if let Some(state) = kv_state.take() {
                let Some(manager) = self.page_manager.as_mut() else {
                    self.sessions.begin_cleanup(
                        session_id,
                        PendingSequenceCleanup::Suspended {
                            kv_state: Some(state),
                            retirement,
                            model_state,
                        },
                    );
                    return Err(Error::Invariant {
                        message: "suspended KV state has no authoritative page manager".into(),
                    });
                };
                match manager.release_preempted_pages(state) {
                    Ok(released) => {
                        retirement = Some(PendingKvRetirement::BackendRelease(released));
                    }
                    Err(failure) => {
                        let (error, state) = failure.into_parts();
                        self.sessions.begin_cleanup(
                            session_id,
                            PendingSequenceCleanup::Suspended {
                                kv_state: Some(state),
                                retirement,
                                model_state,
                            },
                        );
                        return Err(error);
                    }
                }
            }
            PendingSequenceCleanup::Owned {
                retirement,
                model_state,
            }
        } else {
            cleanup
        };
        let PendingSequenceCleanup::Owned {
            retirement,
            model_state,
        } = cleanup
        else {
            unreachable!("sequence cleanup was converted to owned state")
        };
        if let Some(retirement) = retirement
            && let Err((error, retirement)) = self.progress_kv_retirement(retirement)
        {
            self.sessions.begin_cleanup(
                session_id,
                PendingSequenceCleanup::Owned {
                    retirement: Some(retirement),
                    model_state,
                },
            );
            return Err(error);
        }
        if let Some(state) = model_state
            && let Err(failure) = self.executor.try_release_sequence_state(state)
        {
            let (error, state) = failure.into_parts();
            self.sessions.begin_cleanup(
                session_id,
                PendingSequenceCleanup::Owned {
                    retirement: None,
                    model_state: Some(state),
                },
            );
            return Err(error.into());
        }
        Ok(())
    }
}
