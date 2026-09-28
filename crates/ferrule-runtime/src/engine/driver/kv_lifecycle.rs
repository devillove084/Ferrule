//! Kv lifecycle operations on the driver's existing authority.

use super::*;

pub(super) enum PendingResidentKv {
    Reserved(Vec<KvReservation>),
    Prepared(PreparedKvCommit),
    Retiring(PendingKvRetirement),
}

pub(super) enum PendingKvRetirement {
    BackendRelease(KvRetirement),
    LogicalConfirmation(KvRetirement),
}

/// Logical KV ownership bookkeeping. The page manager and resource registry
/// remain the physical authorities; this component only keeps each acquired
/// page's hard grant paired with its retirement work.
pub(super) struct KvLifecycleState {
    grants: HashMap<KvPageId, PhysicalResourceGrant>,
    pending_retirements: VecDeque<PendingKvRetirement>,
    pending_aborts: VecDeque<PendingResidentKv>,
}

impl Default for KvLifecycleState {
    fn default() -> Self {
        Self {
            grants: HashMap::new(),
            pending_retirements: VecDeque::new(),
            pending_aborts: VecDeque::new(),
        }
    }
}

impl KvLifecycleState {
    pub(super) fn pending_abort_count(&self) -> usize {
        self.pending_aborts.len()
    }

    pub(super) fn release_unbound_kv_page_grants(
        &mut self,
        registry: &mut LoadRegistry<Box<dyn RuntimeMaterializationProvider>>,
        grants: Vec<PhysicalResourceGrant>,
    ) {
        for mut grant in grants {
            registry
                .release_hard_resources(&mut grant)
                .expect("unbound KV page hard grant matches its registry broker");
        }
    }
    pub(super) fn track_kv_page_grants(
        &mut self,
        registry: &mut LoadRegistry<Box<dyn RuntimeMaterializationProvider>>,
        physical_pages: Vec<KvPageId>,
        grants: Vec<PhysicalResourceGrant>,
    ) -> Result<()> {
        if physical_pages.len() != grants.len() {
            let page_count = physical_pages.len();
            let grant_count = grants.len();
            self.release_unbound_kv_page_grants(registry, grants);
            return Err(Error::Invariant {
                message: format!(
                    "cannot track {grant_count} KV page hard grants for {page_count} physical pages"
                ),
            });
        }
        let mut unique = HashSet::with_capacity(physical_pages.len());
        if let Some(duplicate) = physical_pages
            .iter()
            .copied()
            .find(|page| !unique.insert(*page) || self.has_grant(page))
        {
            self.release_unbound_kv_page_grants(registry, grants);
            return Err(Error::Invariant {
                message: format!(
                    "KV page {} acquired hard credit more than once",
                    duplicate.0
                ),
            });
        }
        for (page, grant) in physical_pages.into_iter().zip(grants) {
            let previous = self.insert_grant(page, grant);
            debug_assert!(previous.is_none(), "KV grant preflight rejected duplicates");
        }
        Ok(())
    }
    fn confirm_retirement(
        &mut self,
        registry: &mut LoadRegistry<Box<dyn RuntimeMaterializationProvider>>,
        page_manager: &mut Option<KvPageManager>,
        retirement: KvRetirement,
    ) -> std::result::Result<(), (Error, PendingKvRetirement)> {
        let Some(manager) = page_manager.as_mut() else {
            return Err((
                Error::Invariant {
                    message: "retiring KV pages have no authoritative page manager".into(),
                },
                PendingKvRetirement::LogicalConfirmation(retirement),
            ));
        };
        let retired_pages = retirement.pages().to_vec();
        if let Some(untracked) = retired_pages.iter().find(|page| !self.has_grant(page)) {
            return Err((
                Error::Invariant {
                    message: format!(
                        "retiring KV page {} has no exact hard-credit grant",
                        untracked.0
                    ),
                },
                PendingKvRetirement::LogicalConfirmation(retirement),
            ));
        }
        match manager.confirm_page_retirement(retirement) {
            Ok(()) => {
                for page in retired_pages {
                    let mut grant = self
                        .remove_grant(&page)
                        .expect("retirement hard-credit ownership was preflighted");
                    registry
                        .release_hard_resources(&mut grant)
                        .expect("KV page hard grant matches its registry broker");
                }
                Ok(())
            }
            Err(error) => {
                let (error, retirement) = error.into_parts();
                Err((error, PendingKvRetirement::LogicalConfirmation(retirement)))
            }
        }
    }

    pub(super) fn has_grant(&self, page: &KvPageId) -> bool {
        self.grants.contains_key(page)
    }
    pub(super) fn grant_count(&self) -> usize {
        self.grants.len()
    }
    pub(super) fn grants_empty(&self) -> bool {
        self.grants.is_empty()
    }
    fn insert_grant(
        &mut self,
        page: KvPageId,
        grant: PhysicalResourceGrant,
    ) -> Option<PhysicalResourceGrant> {
        self.grants.insert(page, grant)
    }
    fn remove_grant(&mut self, page: &KvPageId) -> Option<PhysicalResourceGrant> {
        self.grants.remove(page)
    }
    pub(super) fn pending_retirement_count(&self) -> usize {
        self.pending_retirements.len()
    }
    pub(super) fn pending_retirements_empty(&self) -> bool {
        self.pending_retirements.is_empty()
    }
    pub(super) fn push_retirement(&mut self, retirement: PendingKvRetirement) {
        self.pending_retirements.push_back(retirement);
    }
    pub(super) fn pop_retirement(&mut self) -> Option<PendingKvRetirement> {
        self.pending_retirements.pop_front()
    }
    pub(super) fn retry_retirement(&mut self, retirement: PendingKvRetirement) {
        self.pending_retirements.push_front(retirement);
    }
    pub(super) fn pending_abort_empty(&self) -> bool {
        self.pending_aborts.is_empty()
    }
    pub(super) fn push_abort(&mut self, abort: PendingResidentKv) {
        self.pending_aborts.push_back(abort);
    }
    pub(super) fn pop_abort(&mut self) -> Option<PendingResidentKv> {
        self.pending_aborts.pop_front()
    }
    pub(super) fn retry_abort(&mut self, abort: PendingResidentKv) {
        self.pending_aborts.push_front(abort);
    }
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: MultiSessionRunner,
    C: SequenceSlotPool,
{
    /// Installs the page manager, panicking on an invalid test topology.
    pub fn with_page_manager(mut self, page_manager: KvPageManager) -> Self {
        let explicit_kv_limit = self
            .load_registry
            .resources()
            .snapshots()
            .find(|snapshot| snapshot.kind == ResourceKind::KvPage)
            .expect("hard resource catalog contains KV pages")
            .capacity;
        if explicit_kv_limit == 0 {
            return self
                .try_with_page_manager(page_manager)
                .expect("test page-manager topology must be valid");
        }
        self.executor
            .configure_kv_page_capacity(page_manager.max_pages())
            .expect("test backend accepts page-manager capacity");
        self.page_manager = Some(page_manager);
        self
    }

    /// Install the authoritative runtime page manager and configure a backend
    /// physical pool with the same bounded page capacity.
    pub fn try_with_page_manager(mut self, page_manager: KvPageManager) -> Result<Self> {
        if self.page_manager.is_some() {
            return Err(Error::InvalidRequest {
                message: "the authoritative KV page manager is already installed".into(),
            });
        }
        let max_pages = page_manager.max_pages();
        if max_pages == 0 {
            return Err(Error::InvalidRequest {
                message: "a physical KV backend requires a bounded non-zero page capacity".into(),
            });
        }
        self.ensure_no_suspended_execution("install a page manager")?;
        self.executor.configure_kv_page_capacity(max_pages)?;
        self.load_registry.resources_mut().reconfigure_limit(
            ResourceKind::KvPage,
            u64::try_from(max_pages).map_err(|_| Error::InvalidRequest {
                message: "KV page capacity exceeds runtime resource range".into(),
            })?,
            0,
        )?;
        self.page_manager = Some(page_manager);
        Ok(self)
    }

    pub fn page_manager(&self) -> Option<&KvPageManager> {
        self.page_manager.as_ref()
    }

    pub(super) fn available_kv_page_credits(&self) -> usize {
        self.load_registry
            .resources()
            .snapshots()
            .find(|snapshot| snapshot.kind == ResourceKind::KvPage)
            .map_or(0, |snapshot| {
                usize::try_from(snapshot.capacity.saturating_sub(snapshot.in_use))
                    .unwrap_or(usize::MAX)
            })
    }

    pub(super) fn progress_kv_retirement(
        &mut self,
        retirement: PendingKvRetirement,
    ) -> std::result::Result<(), (Error, PendingKvRetirement)> {
        if self.has_live_transactions() {
            return Err((
                Error::InvalidRequest {
                    message: "cannot progress KV retirement while packed transactions are live"
                        .into(),
                },
                retirement,
            ));
        }
        let retirement = match retirement {
            PendingKvRetirement::BackendRelease(retirement) => {
                if !retirement.is_empty()
                    && let Err(error) = self
                        .executor
                        .runner_mut()
                        .release_kv_pages(retirement.pages())
                {
                    return Err((
                        error.into(),
                        PendingKvRetirement::BackendRelease(retirement),
                    ));
                }
                retirement
            }
            PendingKvRetirement::LogicalConfirmation(retirement) => retirement,
        };
        self.kv
            .confirm_retirement(&mut self.load_registry, &mut self.page_manager, retirement)
    }

    pub(super) fn track_kv_page_grants(
        &mut self,
        physical_pages: Vec<KvPageId>,
        grants: Vec<PhysicalResourceGrant>,
    ) -> Result<()> {
        self.kv
            .track_kv_page_grants(&mut self.load_registry, physical_pages, grants)
    }

    pub(super) fn reserve_batch_pages(
        &mut self,
        transaction: ExecutionTransactionId,
        demand: crate::scheduling::ResourceDemand,
        batch: &ScheduledBatch,
        execution_generations: &[u64],
    ) -> Result<Vec<KvReservation>> {
        if self.page_manager.is_none() {
            return Ok(Vec::new());
        }
        if execution_generations.len() != batch.sequences.len() {
            return Err(Error::Invariant {
                message: format!(
                    "KV execution generation count {} does not match scheduled sequence count {}",
                    execution_generations.len(),
                    batch.sequences.len()
                ),
            });
        }
        let mut reservations = Vec::with_capacity(batch.sequences.len());
        for ((scheduled, execution), execution_generation) in batch
            .sequences
            .iter()
            .zip(batch.execution().sequences())
            .zip(execution_generations)
        {
            let prepared = (|| -> Result<(StateSlot, usize, u64, usize)> {
                let slot = *self
                    .sessions
                    .page_slot(&scheduled.session_id)
                    .ok_or_else(|| Error::Invariant {
                        message: format!(
                            "no page slot for active session {:?}",
                            scheduled.session_id
                        ),
                    })?;
                let token_count = usize::try_from(execution.query.end - execution.query.start)
                    .map_err(|_| Error::InvalidRequest {
                        message: "query length exceeds usize".into(),
                    })?;
                let page_generation = self
                    .page_manager
                    .as_ref()
                    .expect("page manager presence checked above")
                    .sequence_generation(slot)?;
                let required_before_eviction = self
                    .page_manager
                    .as_ref()
                    .expect("page manager presence checked above")
                    .required_physical_pages(slot, page_generation, token_count)?;
                self.evict_prefixes_for_kv_pages(required_before_eviction)?;
                let required = self
                    .page_manager
                    .as_ref()
                    .expect("page manager presence checked above")
                    .required_physical_pages(slot, page_generation, token_count)?;
                Ok((slot, token_count, page_generation, required))
            })();
            let (slot, token_count, page_generation, required) = match prepared {
                Ok(prepared) => prepared,
                Err(error) => {
                    let cleanup =
                        self.abort_quiesced_resident_kv(PendingResidentKv::Reserved(reservations));
                    return Err(Error::with_cleanup(
                        "resident KV demand preparation",
                        error,
                        cleanup,
                    ));
                }
            };
            let mut grants = Vec::with_capacity(required);
            for _ in 0..required {
                match self.load_registry.acquire_hard_resources(
                    transaction.get(),
                    demand,
                    [PhysicalResourceClaim::new(ResourceKind::KvPage, 1)],
                ) {
                    Ok(grant) => grants.push(grant),
                    Err(error) => {
                        for mut grant in grants {
                            self.load_registry
                                .release_hard_resources(&mut grant)
                                .expect("unsubmitted KV page grant is releasable");
                        }
                        let cleanup = self
                            .abort_quiesced_resident_kv(PendingResidentKv::Reserved(reservations));
                        return Err(Error::with_cleanup("KV hard admission", error, cleanup));
                    }
                }
            }

            let mut reservation = match self
                .page_manager
                .as_mut()
                .expect("page manager presence checked above")
                .reserve(slot, page_generation, token_count)
            {
                Ok(reservation) => reservation,
                Err(error) => {
                    for mut grant in grants {
                        self.load_registry
                            .release_hard_resources(&mut grant)
                            .expect("unsubmitted KV page grant is releasable");
                    }
                    let cleanup =
                        self.abort_quiesced_resident_kv(PendingResidentKv::Reserved(reservations));
                    return Err(Error::with_cleanup("KV reservation", error, cleanup));
                }
            };
            if let Err(error) = self
                .page_manager
                .as_mut()
                .expect("page manager presence checked above")
                .bind_reservation_execution(
                    &mut reservation,
                    execution.state_slot,
                    *execution_generation,
                )
            {
                for mut grant in grants {
                    self.load_registry
                        .release_hard_resources(&mut grant)
                        .expect("unsubmitted KV page grant is releasable");
                }
                let mut cleanup_reservations = reservations;
                cleanup_reservations.push(reservation);
                let cleanup = self
                    .abort_quiesced_resident_kv(PendingResidentKv::Reserved(cleanup_reservations));
                return Err(Error::with_cleanup(
                    "KV execution-generation binding",
                    error,
                    cleanup,
                ));
            }
            let mut physical_pages = reservation.view().newly_allocated.clone();
            if let Some(cow) = reservation.view().cow_replacement {
                physical_pages.push(cow.replacement);
            }
            if physical_pages.len() != grants.len() {
                for mut grant in grants {
                    self.load_registry
                        .release_hard_resources(&mut grant)
                        .expect("unsubmitted KV page grant is releasable");
                }
                let mut cleanup_reservations = reservations;
                cleanup_reservations.push(reservation);
                let error = Error::Invariant {
                    message: format!(
                        "KV page manager reserved {} physical pages after exact hard admission for {}",
                        physical_pages.len(),
                        required
                    ),
                };
                let cleanup = self
                    .abort_quiesced_resident_kv(PendingResidentKv::Reserved(cleanup_reservations));
                return Err(Error::with_cleanup(
                    "KV hard-credit correlation",
                    error,
                    cleanup,
                ));
            }
            if let Err(error) = self.track_kv_page_grants(physical_pages, grants) {
                let mut cleanup_reservations = reservations;
                cleanup_reservations.push(reservation);
                let cleanup = self
                    .abort_quiesced_resident_kv(PendingResidentKv::Reserved(cleanup_reservations));
                return Err(Error::with_cleanup(
                    "KV hard-credit tracking",
                    error,
                    cleanup,
                ));
            }
            reservations.push(reservation);
        }
        Ok(reservations)
    }

    pub(super) fn reserve_speculative_pages(
        &mut self,
        transaction: ExecutionTransactionId,
        items: &[SpeculativeVerificationItem<'_>],
    ) -> Result<Vec<KvReservation>> {
        let mut reservations = Vec::with_capacity(items.len());
        for item in items {
            let required =
                (|| {
                    let token_count = item.proposal.len().checked_add(1).ok_or_else(|| {
                        Error::InvalidRequest {
                            message: "speculative verification row count overflow".into(),
                        }
                    })?;
                    let required = self
                        .page_manager
                        .as_ref()
                        .ok_or_else(|| Error::InvalidRequest {
                            message:
                                "speculative KV reservation requires an authoritative page manager"
                                    .into(),
                        })?
                        .required_physical_pages(item.state_slot, item.generation, token_count)?;
                    Ok((token_count, required))
                })();
            let (token_count, required) = match required {
                Ok(required) => required,
                Err(error) => {
                    return Err(self.rollback_speculative_reservations(
                        reservations,
                        error,
                        "speculative KV demand preparation",
                    ));
                }
            };
            // Speculative proposal/verification owns model topology outside the
            // ordinary resident reserve path, so prefix eviction is deliberately
            // bypassed here until that ownership can be quiesced independently.
            let mut grants = Vec::with_capacity(required);
            for _ in 0..required {
                match self.load_registry.acquire_hard_resources(
                    transaction.get(),
                    crate::scheduling::ResourceDemand::required(
                        ExecutionPhase::SpeculativeVerification,
                    ),
                    [PhysicalResourceClaim::new(ResourceKind::KvPage, 1)],
                ) {
                    Ok(grant) => grants.push(grant),
                    Err(error) => {
                        for mut grant in grants {
                            self.load_registry
                                .release_hard_resources(&mut grant)
                                .expect("unsubmitted speculative KV page grant is releasable");
                        }
                        return Err(self.rollback_speculative_reservations(
                            reservations,
                            error.into(),
                            "speculative KV hard admission",
                        ));
                    }
                }
            }

            let reservation = match self
                .page_manager
                .as_mut()
                .expect("page manager presence checked above")
                .reserve(item.state_slot, item.generation, token_count)
            {
                Ok(reservation) => reservation,
                Err(error) => {
                    for mut grant in grants {
                        self.load_registry
                            .release_hard_resources(&mut grant)
                            .expect("unsubmitted speculative KV page grant is releasable");
                    }
                    return Err(self.rollback_speculative_reservations(
                        reservations,
                        error,
                        "speculative KV reserve",
                    ));
                }
            };
            let mut physical_pages = reservation.view().newly_allocated.clone();
            if let Some(cow) = reservation.view().cow_replacement {
                physical_pages.push(cow.replacement);
            }
            if physical_pages.len() != grants.len() {
                for mut grant in grants {
                    self.load_registry
                        .release_hard_resources(&mut grant)
                        .expect("unsubmitted speculative KV page grant is releasable");
                }
                reservations.push(reservation);
                let error = Error::Invariant {
                    message: format!(
                        "speculative KV manager reserved {} physical pages after exact hard admission for {required}",
                        physical_pages.len()
                    ),
                };
                return Err(self.rollback_speculative_reservations(
                    reservations,
                    error,
                    "speculative KV hard-credit correlation",
                ));
            }
            if let Err(error) = self.track_kv_page_grants(physical_pages, grants) {
                reservations.push(reservation);
                return Err(self.rollback_speculative_reservations(
                    reservations,
                    error,
                    "speculative KV hard-credit tracking",
                ));
            }
            reservations.push(reservation);
        }
        Ok(reservations)
    }

    fn rollback_speculative_reservations(
        &mut self,
        reservations: Vec<KvReservation>,
        error: Error,
        stage: &'static str,
    ) -> Error {
        let cleanup = self.abort_quiesced_resident_kv(PendingResidentKv::Reserved(reservations));
        Error::with_cleanup(stage, error, cleanup)
    }

    pub(super) fn bind_reserved_pages(
        &self,
        batch: &mut ScheduledBatch,
        reservations: &[KvReservation],
    ) -> Result<()> {
        if self.executor.capabilities().kv_binding_mode != KvBindingMode::Paged {
            return Ok(());
        }
        let manager = self
            .page_manager
            .as_ref()
            .ok_or_else(|| Error::InvalidRequest {
                message: "paged executor requires a runtime KvPageManager".into(),
            })?;
        let views = manager.reservation_views(reservations)?;
        let bindings = views
            .iter()
            .map(|reservation| manager.reservation_bindings(reservation))
            .collect::<Result<Vec<_>>>()?;
        batch.bind_paged_kv(&bindings)
    }

    pub(super) fn abort_quiesced_resident_kv(&mut self, kv: PendingResidentKv) -> Result<()> {
        let Some(manager) = &mut self.page_manager else {
            return match kv {
                PendingResidentKv::Reserved(reservations) if reservations.is_empty() => Ok(()),
                kv => {
                    self.kv.push_abort(kv);
                    Err(Error::Invariant {
                        message: "KV transaction exists without an authoritative page manager"
                            .into(),
                    })
                }
            };
        };
        let retirement = match kv {
            PendingResidentKv::Reserved(reservations) => {
                match manager.abort_reservations(reservations) {
                    Ok(retirement) => retirement,
                    Err(error) => {
                        let (error, reservations) = error.into_parts();
                        self.kv
                            .push_abort(PendingResidentKv::Reserved(reservations));
                        return Err(error);
                    }
                }
            }
            PendingResidentKv::Prepared(prepared) => manager.abort_prepared_commit(prepared),
            PendingResidentKv::Retiring(retirement) => {
                return match self.progress_kv_retirement(retirement) {
                    Ok(()) => Ok(()),
                    Err((error, retirement)) => {
                        self.kv.push_retirement(retirement);
                        Err(error)
                    }
                };
            }
        };
        self.release_and_confirm_retirement(retirement)
    }

    /// Retires every page owned by one sequence slot and confirms the retirement.
    pub fn retire_sequence_pages(
        &mut self,
        state_slot: ferrule_common::execution::StateSlot,
    ) -> Result<()> {
        let retirement = self
            .page_manager
            .as_mut()
            .ok_or_else(|| Error::InvalidRequest {
                message: "the authoritative KV page manager is not installed".into(),
            })?
            .free_sequence_pages(state_slot)?;
        self.release_and_confirm_retirement(retirement)
    }

    pub(super) fn release_and_confirm_retirement(
        &mut self,
        retirement: KvRetirement,
    ) -> Result<()> {
        let retirement = PendingKvRetirement::BackendRelease(retirement);
        if self.has_live_transactions() {
            self.kv.push_retirement(retirement);
            return Ok(());
        }
        match self.progress_kv_retirement(retirement) {
            Ok(()) => Ok(()),
            Err((error, retirement)) => {
                self.kv.push_retirement(retirement);
                Err(error)
            }
        }
    }

    pub(super) fn progress_committed_resident_kv(
        &mut self,
        kv: PendingResidentKv,
    ) -> std::result::Result<(), (Error, PendingResidentKv)> {
        let kv = match kv {
            PendingResidentKv::Prepared(prepared) => {
                let Some(manager) = self.page_manager.as_mut() else {
                    return Err((
                        Error::Invariant {
                            message: "prepared logical commit has no authoritative page manager"
                                .into(),
                        },
                        PendingResidentKv::Prepared(prepared),
                    ));
                };
                PendingResidentKv::Retiring(PendingKvRetirement::BackendRelease(
                    manager.publish_commit(prepared),
                ))
            }
            PendingResidentKv::Reserved(reservations) if reservations.is_empty() => return Ok(()),
            PendingResidentKv::Reserved(reservations) => {
                return Err((
                    Error::Invariant {
                        message: "backend committed while logical reservations remained unprepared"
                            .into(),
                    },
                    PendingResidentKv::Reserved(reservations),
                ));
            }
            kv @ PendingResidentKv::Retiring(_) => kv,
        };
        let PendingResidentKv::Retiring(retirement) = kv else {
            unreachable!("committed logical KV was converted to retirement")
        };
        self.progress_or_defer_resident_retirement(retirement)
            .map_err(|(error, retirement)| (error, PendingResidentKv::Retiring(retirement)))
    }

    pub(super) fn progress_or_defer_resident_retirement(
        &mut self,
        retirement: PendingKvRetirement,
    ) -> std::result::Result<(), (Error, PendingKvRetirement)> {
        if self.has_live_transactions() {
            self.kv.push_retirement(retirement);
            Ok(())
        } else {
            self.progress_kv_retirement(retirement)
        }
    }

    pub(super) fn progress_speculative_ending_retirements(
        &mut self,
        pending: &mut PendingSpeculativeEndingDriverCohort<R::SequenceState>,
    ) -> Result<()> {
        let retirements = match &mut pending.ending {
            SpeculativeEnding::BackendCommittedPendingPublish { retirements, .. }
            | SpeculativeEnding::BackendAbortedPendingCleanup { retirements, .. } => retirements,
            _ => unreachable!("speculative retirement progress requires a post-terminal phase"),
        };
        while let Some(retirement) = retirements.pop_front() {
            if let Err((error, retirement)) = self.progress_or_defer_resident_retirement(retirement)
            {
                retirements.push_front(retirement);
                return Err(error);
            }
        }
        Ok(())
    }

    pub(super) fn progress_aborted_resident_kv(
        &mut self,
        kv: PendingResidentKv,
    ) -> std::result::Result<(), (Error, PendingResidentKv)> {
        let kv = match kv {
            PendingResidentKv::Reserved(reservations) if reservations.is_empty() => return Ok(()),
            PendingResidentKv::Reserved(reservations) => {
                let Some(manager) = self.page_manager.as_mut() else {
                    return Err((
                        Error::Invariant {
                            message: "KV reservations have no authoritative page manager".into(),
                        },
                        PendingResidentKv::Reserved(reservations),
                    ));
                };
                let retirement = match manager.abort_reservations(reservations) {
                    Ok(retirement) => retirement,
                    Err(error) => {
                        let (error, reservations) = error.into_parts();
                        return Err((error, PendingResidentKv::Reserved(reservations)));
                    }
                };
                PendingResidentKv::Retiring(PendingKvRetirement::BackendRelease(retirement))
            }
            PendingResidentKv::Prepared(prepared) => {
                let Some(manager) = self.page_manager.as_mut() else {
                    return Err((
                        Error::Invariant {
                            message: "prepared logical commit has no authoritative page manager"
                                .into(),
                        },
                        PendingResidentKv::Prepared(prepared),
                    ));
                };
                PendingResidentKv::Retiring(PendingKvRetirement::BackendRelease(
                    manager.abort_prepared_commit(prepared),
                ))
            }
            kv @ PendingResidentKv::Retiring(_) => kv,
        };
        let PendingResidentKv::Retiring(retirement) = kv else {
            unreachable!("aborted logical KV was converted to retirement")
        };
        self.progress_or_defer_resident_retirement(retirement)
            .map_err(|(error, retirement)| (error, PendingResidentKv::Retiring(retirement)))
    }
}
