//! Model-side adapter state shared with the runtime-owned load registry.
//!
//! The runner receives only [`RuntimeMaterializationAdapter`]. It can resolve exact
//! logical dependencies and observe runtime-published state, but it never owns a
//! physical operation, hard-resource grant, waiter, completion, or retirement.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};

use ferrule_common::io_protocol::{
    CancellationReason, LoadKey, LoadStage, ResidencyBinding, ValidatedResidencyBinding,
};
use ferrule_common::{CompletionHub, Error, Result};
use ferrule_model::{
    ExpertDependencyResolution, ExpertMaterializationAdapter, ExpertMaterializationCancelOutcome,
    ExpertMaterializationPlacement, ExpertMaterializationProgress, ExpertMaterializationRequest,
    ExpertMaterializationSubmission, PhysicalExpertReservation,
};

use super::backend::RunnerMaterializationBackend;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeMaterializationAdapterStats {
    pub resolves: u64,
    pub physical_submissions: u64,
    pub single_flight_joins: u64,
    pub progress_polls: u64,
    pub publications: u64,
    pub logical_detaches: u64,
    pub cancellation_hooks: u64,
}

#[derive(Debug)]
struct AdapterEntry {
    request: ExpertMaterializationRequest,
    stage: LoadStage,
    resident: Option<ResidencyBinding>,
    pending_eviction: Option<LoadKey>,
    submitted: bool,
    logical_demands: u64,
}

#[derive(Debug)]
struct AdapterState {
    placement: ExpertMaterializationPlacement,
    physical: Option<RunnerMaterializationBackend>,
    completion_hub: CompletionHub,
    request_keys: BTreeMap<ExpertMaterializationRequest, LoadKey>,
    entries: BTreeMap<LoadKey, AdapterEntry>,
    evicted: Vec<LoadKey>,
    detached: BTreeSet<(ferrule_common::ContinuationId, LoadKey)>,
    stats: RuntimeMaterializationAdapterStats,
}

impl AdapterState {
    fn new(
        placement: ExpertMaterializationPlacement,
        physical: Option<RunnerMaterializationBackend>,
        completion_hub: CompletionHub,
    ) -> Self {
        Self {
            placement,
            physical,
            completion_hub,
            request_keys: BTreeMap::new(),
            entries: BTreeMap::new(),
            evicted: Vec::new(),
            detached: BTreeSet::new(),
            stats: RuntimeMaterializationAdapterStats::default(),
        }
    }
}

/// Runtime-side control plane. Cloning this handle does not clone physical
/// ownership; it only addresses the adapter state installed in the runner.
#[derive(Debug, Clone)]
pub struct RuntimeMaterializationControl {
    state: Arc<Mutex<AdapterState>>,
}

impl RuntimeMaterializationControl {
    pub fn new(
        placement: ExpertMaterializationPlacement,
        physical: Option<RunnerMaterializationBackend>,
        completion_hub: CompletionHub,
    ) -> (RuntimeMaterializationAdapter, Self) {
        let state = Arc::new(Mutex::new(AdapterState::new(
            placement,
            physical,
            completion_hub,
        )));
        (
            RuntimeMaterializationAdapter {
                state: Arc::clone(&state),
            },
            Self { state },
        )
    }

    pub fn placement(&self) -> ExpertMaterializationPlacement {
        self.lock().placement
    }

    pub fn stats(&self) -> RuntimeMaterializationAdapterStats {
        self.lock().stats
    }

    pub fn keys(&self) -> Vec<LoadKey> {
        self.lock().entries.keys().copied().collect()
    }

    /// Whether this key was fixed by the physical resolve/reserve authority.
    /// Driver continuation registration must reject arbitrary model-synthesized
    /// keys that never crossed that boundary.
    pub fn is_resolved(&self, key: LoadKey) -> bool {
        self.lock().entries.contains_key(&key)
    }

    pub fn take_evicted(&self) -> Vec<LoadKey> {
        std::mem::take(&mut self.lock().evicted)
    }

    pub fn record_stage(&self, key: LoadKey, stage: LoadStage) -> Result<()> {
        let mut state = self.lock();
        let entry = state.entries.get_mut(&key).ok_or_else(|| {
            Error::Execution(format!(
                "runtime materialization adapter has no resolved key {key:?}"
            ))
        })?;
        entry.stage = stage;
        Ok(())
    }

    pub fn record_resident(&self, key: LoadKey, binding: ResidencyBinding) -> Result<()> {
        let validated = ValidatedResidencyBinding::new(key, binding)?;
        let mut state = self.lock();
        let evicted = {
            let entry = state.entries.get_mut(&key).ok_or_else(|| {
                Error::Execution(format!(
                    "runtime materialization adapter has no resolved key {key:?}"
                ))
            })?;
            entry.stage = LoadStage::Resident;
            entry.resident = Some(validated.binding());
            entry.pending_eviction.take()
        };
        if let Some(evicted) = evicted {
            state.evicted.push(evicted);
        }
        Ok(())
    }

    pub fn forget(&self, key: LoadKey) {
        let mut state = self.lock();
        if let Some(entry) = state.entries.remove(&key) {
            state.request_keys.remove(&entry.request);
        }
    }

    fn lock(&self) -> MutexGuard<'_, AdapterState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Non-owning model-side view installed through `MultiSessionRunner`.
#[derive(Debug)]
pub struct RuntimeMaterializationAdapter {
    state: Arc<Mutex<AdapterState>>,
}

impl RuntimeMaterializationAdapter {
    fn lock(&self) -> MutexGuard<'_, AdapterState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl ExpertMaterializationAdapter for RuntimeMaterializationAdapter {
    fn placement(&self) -> ExpertMaterializationPlacement {
        self.lock().placement
    }

    fn resolve(
        &mut self,
        request: ExpertMaterializationRequest,
    ) -> Result<ExpertDependencyResolution> {
        let physical = {
            let mut state = self.lock();
            state.stats.resolves = state.stats.resolves.saturating_add(1);
            if request.model() != state.placement.model()
                || request.backend() != state.placement.backend()
                || request.device() != state.placement.device()
            {
                return Err(Error::Execution(
                    "expert materialization request does not match the runtime placement".into(),
                ));
            }
            if let Some(key) = state.request_keys.get(&request).copied() {
                let entry = state
                    .entries
                    .get(&key)
                    .expect("request and key adapter indices are updated together");
                if let Some(binding) = entry.resident {
                    if entry.logical_demands > 0 {
                        return ExpertDependencyResolution::resident(
                            request,
                            ValidatedResidencyBinding::new(key, binding)?,
                        );
                    }
                    // The prior cohort released its selected physical lease.
                    // Re-enter physical resolution to reacquire or replace the
                    // exact slot before exposing this binding to a new cohort.
                } else {
                    return ExpertDependencyResolution::waiting(request, key);
                }
            }
            state.physical.clone().ok_or_else(|| {
                Error::Execution(
                    "runner requested expert materialization without a physical backend".into(),
                )
            })?
        };

        let resolution = physical.resolve_or_reserve(request).map_err(|reason| {
            Error::Execution(format!("physical expert reservation failed: {reason:?}"))
        })?;
        let mut state = self.lock();
        match resolution {
            PhysicalExpertReservation::Resident(binding) => {
                let key = binding.key();
                request.validate_key(key)?;
                state.request_keys.insert(request, key);
                state.entries.insert(
                    key,
                    AdapterEntry {
                        request,
                        stage: LoadStage::Resident,
                        resident: Some(binding.binding()),
                        pending_eviction: None,
                        submitted: true,
                        logical_demands: 0,
                    },
                );
                ExpertDependencyResolution::resident(request, binding)
            }
            PhysicalExpertReservation::Reserved(reservation) => {
                let key = reservation.key();
                request.validate_key(key)?;
                if reservation.binding().generation != key.destination_generation() {
                    return Err(Error::Execution(
                        "physical expert reservation returned a mismatched slot generation".into(),
                    ));
                }
                let pending_eviction = reservation.evicted();
                state.request_keys.insert(request, key);
                state.entries.insert(
                    key,
                    AdapterEntry {
                        request,
                        stage: LoadStage::Reserved,
                        resident: None,
                        pending_eviction,
                        submitted: false,
                        logical_demands: 0,
                    },
                );
                ExpertDependencyResolution::waiting(request, key)
            }
        }
    }

    fn submit(&mut self, key: LoadKey) -> Result<ExpertMaterializationSubmission> {
        let mut state = self.lock();
        let entry = state.entries.get_mut(&key).ok_or_else(|| {
            Error::Execution(format!(
                "cannot submit unresolved expert materialization key {key:?}"
            ))
        })?;
        entry.logical_demands = entry
            .logical_demands
            .checked_add(1)
            .ok_or_else(|| Error::Execution("expert logical demand count overflow".into()))?;
        if let Some(binding) = entry.resident {
            return Ok(ExpertMaterializationSubmission::Resident(
                ValidatedResidencyBinding::new(key, binding)?,
            ));
        }
        if entry.submitted {
            state.stats.single_flight_joins = state.stats.single_flight_joins.saturating_add(1);
            Ok(ExpertMaterializationSubmission::Joined)
        } else {
            entry.submitted = true;
            state.stats.physical_submissions = state.stats.physical_submissions.saturating_add(1);
            Ok(ExpertMaterializationSubmission::Submitted)
        }
    }

    fn progress(&mut self, key: LoadKey) -> Result<ExpertMaterializationProgress> {
        let mut state = self.lock();
        state.stats.progress_polls = state.stats.progress_polls.saturating_add(1);
        let entry = state.entries.get(&key).ok_or_else(|| {
            Error::Execution(format!(
                "cannot progress unresolved expert materialization key {key:?}"
            ))
        })?;
        if let Some(binding) = entry.resident {
            return Ok(ExpertMaterializationProgress::Resident(
                ValidatedResidencyBinding::new(key, binding)?,
            ));
        }
        if matches!(entry.stage, LoadStage::Installing | LoadStage::Resident) {
            Ok(ExpertMaterializationProgress::ReadyToPublish)
        } else {
            Ok(ExpertMaterializationProgress::Materializing)
        }
    }

    fn publish(&mut self, key: LoadKey) -> Result<ValidatedResidencyBinding> {
        let mut state = self.lock();
        let binding = state
            .entries
            .get(&key)
            .and_then(|entry| entry.resident)
            .ok_or_else(|| {
                Error::Execution(format!(
                    "cannot publish expert key {key:?} before registry residency"
                ))
            })?;
        state.stats.publications = state.stats.publications.saturating_add(1);
        Ok(ValidatedResidencyBinding::new(key, binding)?)
    }

    fn detach(&mut self, continuation: ferrule_common::ContinuationId, key: LoadKey) -> Result<()> {
        let release_authority = {
            let mut state = self.lock();
            if !state.entries.contains_key(&key) {
                return Err(Error::Execution(format!(
                    "cannot detach unresolved expert materialization key {key:?}"
                )));
            }
            if state
                .entries
                .get(&key)
                .is_some_and(|entry| entry.logical_demands == 0)
            {
                return Err(Error::Internal(format!(
                    "expert key {key:?} detached without a matching logical demand"
                )));
            }
            if !state.detached.insert((continuation, key)) {
                return Err(Error::Execution(format!(
                    "continuation {} detached expert key {key:?} more than once",
                    continuation.get()
                )));
            }
            let entry = state
                .entries
                .get_mut(&key)
                .expect("entry presence was validated before detach");
            entry.logical_demands -= 1;
            (entry.logical_demands == 0).then(|| state.physical.clone())
        };

        let released_physical_lease = release_authority.is_some();
        if let Some(physical) = release_authority {
            let release = physical
                .ok_or_else(|| {
                    Error::Internal("expert demand has no physical lease authority".into())
                })
                .and_then(|physical| {
                    physical.release_selected(key).map_err(|reason| {
                        Error::Execution(format!(
                            "physical selected-expert lease release failed: {reason:?}"
                        ))
                    })
                });
            if let Err(error) = release {
                let mut state = self.lock();
                state.detached.remove(&(continuation, key));
                let entry = state
                    .entries
                    .get_mut(&key)
                    .expect("failed release retains the adapter entry");
                entry.logical_demands = entry.logical_demands.saturating_add(1);
                return Err(error);
            }
        }
        let completion_hub = {
            let mut state = self.lock();
            state.stats.logical_detaches = state.stats.logical_detaches.saturating_add(1);
            state.completion_hub.clone()
        };
        if released_physical_lease {
            completion_hub.notify();
        }
        Ok(())
    }

    fn cancel(
        &mut self,
        key: LoadKey,
        _reason: CancellationReason,
    ) -> Result<ExpertMaterializationCancelOutcome> {
        let mut state = self.lock();
        state.stats.cancellation_hooks = state.stats.cancellation_hooks.saturating_add(1);
        let Some(entry) = state.entries.get(&key) else {
            return Ok(ExpertMaterializationCancelOutcome::AlreadyTerminal);
        };
        if entry.resident.is_some() {
            Ok(ExpertMaterializationCancelOutcome::AlreadyTerminal)
        } else {
            Ok(ExpertMaterializationCancelOutcome::CancelRequested)
        }
    }
}
