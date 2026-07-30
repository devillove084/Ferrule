//! Model-side adapter state shared with the runtime-owned load registry.
//!
//! The runner receives only [`RuntimeMaterializationAdapter`]. It can resolve exact
//! logical dependencies and observe runtime-published state, but it never owns a
//! physical operation, hard-resource grant, waiter, completion, or retirement.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use ahash::RandomState;

use ferrule_common::io_protocol::{LoadKey, ResidencyBinding, ValidatedResidencyBinding};
use ferrule_common::{CompletionHub, Error, Result};
use ferrule_model::{
    ExpertDependencyResolution, ExpertMaterializationAdapter, ExpertMaterializationPlacement,
    ExpertMaterializationRequest, PhysicalExpertReservation,
};

use super::backend::RunnerMaterializationBackend;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeMaterializationAdapterStats {
    pub resolves: u64,
    pub logical_detaches: u64,
}

#[derive(Debug)]
struct AdapterEntry {
    request: ExpertMaterializationRequest,
    binding: ResidencyBinding,
    pending_eviction: Option<LoadKey>,
    resident: bool,
    active_attachments: HashSet<ferrule_common::ContinuationId, RandomState>,
    selected_lease_held: bool,
}

#[derive(Debug)]
struct AdapterState {
    placement: ExpertMaterializationPlacement,
    physical: Option<RunnerMaterializationBackend>,
    completion_hub: CompletionHub,
    request_keys: HashMap<ExpertMaterializationRequest, LoadKey, RandomState>,
    entries: HashMap<LoadKey, AdapterEntry, RandomState>,
    pending_evictions: HashSet<LoadKey, RandomState>,
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
            request_keys: HashMap::default(),
            entries: HashMap::default(),
            pending_evictions: HashSet::default(),
            stats: RuntimeMaterializationAdapterStats::default(),
        }
    }
}

/// Reconcile the adapter entry's binding with the physical authority's
/// canonical binding. Returns the canonical binding if the entry exists.
/// When the physical authority has changed the binding (e.g. after eviction
/// and re-installation), the adapter entry is updated to match rather than
/// rejected, since the physical backend is the binding source of truth.
fn reconcile_canonical_binding(state: &mut AdapterState, key: LoadKey) -> Result<ResidencyBinding> {
    let physical = state.physical.as_ref().ok_or_else(|| {
        Error::Internal("resolved expert key has no physical materialization authority".into())
    })?;
    let canonical = physical.canonical_binding(key).ok_or_else(|| {
        Error::Internal(format!(
            "physical materialization authority has no canonical binding for key {key:?}"
        ))
    })?;
    let entry = state.entries.get_mut(&key).ok_or_else(|| {
        Error::Execution(format!(
            "runtime materialization adapter has no resolved key {key:?}"
        ))
    })?;
    if entry.binding != canonical {
        entry.binding = canonical;
    }
    Ok(canonical)
}

fn release_selected(physical: Option<RunnerMaterializationBackend>, key: LoadKey) -> Result<()> {
    physical
        .ok_or_else(|| Error::Internal("expert demand has no physical lease authority".into()))?
        .release_selected(key)
        .map_err(|reason| {
            Error::Execution(format!(
                "physical selected-expert lease release failed: {reason:?}"
            ))
        })
}

fn detach_logical_demand(
    shared: &Arc<Mutex<AdapterState>>,
    continuation: ferrule_common::ContinuationId,
    key: LoadKey,
    required: bool,
) -> Result<bool> {
    let release_authority = {
        let mut state = shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(entry) = state.entries.get_mut(&key) else {
            return if required {
                Err(Error::Execution(format!(
                    "cannot detach unresolved expert materialization key {key:?}"
                )))
            } else {
                Ok(false)
            };
        };
        if !entry.active_attachments.remove(&continuation) {
            return if required {
                Err(Error::Execution(format!(
                    "continuation {} has no active attachment for expert key {key:?}",
                    continuation.get()
                )))
            } else {
                Ok(false)
            };
        }
        let release = entry.active_attachments.is_empty() && entry.selected_lease_held;
        if release {
            entry.selected_lease_held = false;
        }
        release.then(|| state.physical.clone())
    };

    let released_physical_lease = release_authority.is_some();
    if let Some(physical) = release_authority
        && let Err(error) = release_selected(physical, key)
    {
        let mut state = shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = state
            .entries
            .get_mut(&key)
            .expect("failed release retains the adapter entry");
        entry.active_attachments.insert(continuation);
        entry.selected_lease_held = true;
        return Err(error);
    }
    let completion_hub = {
        let mut state = shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.stats.logical_detaches = state.stats.logical_detaches.saturating_add(1);
        state.completion_hub.clone()
    };
    if released_physical_lease {
        completion_hub.notify();
    }
    Ok(true)
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

    pub fn pending_evictions(&self) -> Vec<LoadKey> {
        self.lock().pending_evictions.iter().copied().collect()
    }

    pub fn confirm_eviction(&self, key: LoadKey) {
        self.lock().pending_evictions.remove(&key);
    }

    pub fn resident_binding(&self, key: LoadKey) -> Option<ResidencyBinding> {
        self.lock()
            .entries
            .get(&key)
            .filter(|entry| entry.resident)
            .map(|entry| entry.binding)
    }

    pub fn record_resident(&self, key: LoadKey, binding: ResidencyBinding) -> Result<()> {
        let release_authority = {
            let mut state = self.lock();
            // The physical authority is the binding source of truth. If it changed
            // the binding (e.g. after eviction + re-installation between turns),
            // reconcile_canonical_binding updates the adapter entry to match.
            // We do NOT compare against the registry's binding here, because the
            // registry may still hold the stale pre-eviction binding.
            let _canonical = reconcile_canonical_binding(&mut state, key)?;
            let entry = state.entries.get_mut(&key).ok_or_else(|| {
                Error::Execution(format!(
                    "runtime materialization adapter has no resolved key {key:?}"
                ))
            })?;
            let _ = ValidatedResidencyBinding::new(key, binding);
            if !entry.resident {
                entry.selected_lease_held = true;
            }
            entry.resident = true;
            let release = entry.active_attachments.is_empty() && entry.selected_lease_held;
            if release {
                entry.selected_lease_held = false;
            }
            release.then(|| state.physical.clone())
        };

        let released_physical_lease = release_authority.is_some();
        if let Some(physical) = release_authority
            && let Err(error) = release_selected(physical, key)
        {
            let mut state = self.lock();
            let entry = state
                .entries
                .get_mut(&key)
                .expect("failed resident release retains the adapter entry");
            entry.selected_lease_held = true;
            return Err(error);
        }

        let completion_hub = {
            let mut state = self.lock();
            let evicted = state
                .entries
                .get_mut(&key)
                .expect("resident adapter entry remains present")
                .pending_eviction
                .take();
            if let Some(evicted) = evicted {
                state.pending_evictions.insert(evicted);
            }
            state.completion_hub.clone()
        };
        if released_physical_lease {
            completion_hub.notify();
        }
        Ok(())
    }

    pub fn attach(&self, continuation: ferrule_common::ContinuationId, key: LoadKey) -> Result<()> {
        let mut state = self.lock();
        let entry = state.entries.get_mut(&key).ok_or_else(|| {
            Error::Execution(format!(
                "cannot attach unresolved expert materialization key {key:?}"
            ))
        })?;
        if entry.active_attachments.contains(&continuation) {
            return Err(Error::Execution(format!(
                "continuation {} already has an active attachment for expert key {key:?}",
                continuation.get()
            )));
        }
        if entry.resident && !entry.selected_lease_held {
            return Err(Error::Internal(format!(
                "resident expert key {key:?} has no selected physical lease"
            )));
        }
        entry.active_attachments.insert(continuation);
        Ok(())
    }

    pub fn detach_if_attached(
        &self,
        continuation: ferrule_common::ContinuationId,
        key: LoadKey,
    ) -> Result<bool> {
        detach_logical_demand(&self.state, continuation, key, false)
    }

    pub fn forget(&self, key: LoadKey) -> Result<()> {
        self.forget_inner(key, false).map(|_| ())
    }

    pub fn forget_if_idle(&self, key: LoadKey) -> Result<bool> {
        self.forget_inner(key, true)
    }

    fn forget_inner(&self, key: LoadKey, require_idle: bool) -> Result<bool> {
        let (entry, physical, completion_hub) = {
            let mut state = self.lock();
            let Some(entry) = state.entries.get(&key) else {
                return Ok(false);
            };
            if require_idle && !entry.active_attachments.is_empty() {
                return Ok(false);
            }
            let entry = state
                .entries
                .remove(&key)
                .expect("adapter entry presence was checked before forget");
            if state.request_keys.get(&entry.request) == Some(&key) {
                state.request_keys.remove(&entry.request);
            }
            (entry, state.physical.clone(), state.completion_hub.clone())
        };
        let released_physical_lease = entry.selected_lease_held;
        if released_physical_lease && let Err(error) = release_selected(physical, key) {
            let mut state = self.lock();
            state.request_keys.insert(entry.request, key);
            state.entries.insert(key, entry);
            return Err(error);
        }
        if released_physical_lease {
            completion_hub.notify();
        }
        Ok(true)
    }

    pub fn forget_all(&self) -> Result<()> {
        for key in self.keys() {
            self.forget(key)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn active_attachment_count(&self, key: LoadKey) -> usize {
        self.lock()
            .entries
            .get(&key)
            .map_or(0, |entry| entry.active_attachments.len())
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
                if entry.resident {
                    if !entry.active_attachments.is_empty() || entry.selected_lease_held {
                        return ExpertDependencyResolution::resident(
                            request,
                            ValidatedResidencyBinding::new(key, entry.binding)?,
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
        let (key, binding, resident, pending_eviction, selected_lease_held) = match &resolution {
            PhysicalExpertReservation::Resident(binding) => {
                (binding.key(), binding.binding(), true, None, true)
            }
            PhysicalExpertReservation::Reserved(reservation) => (
                reservation.key(),
                reservation.binding(),
                false,
                reservation.evicted(),
                false,
            ),
        };
        request.validate_key(key)?;
        if binding.generation != key.destination_generation() {
            return Err(Error::Execution(
                "physical expert reservation returned a mismatched slot generation".into(),
            ));
        }
        let previous = state.request_keys.get(&request).copied();
        if let Some(previous) = previous
            && previous != key
        {
            let previous_entry = state.entries.get(&previous).ok_or_else(|| {
                Error::Internal("request and key adapter indices diverged".into())
            })?;
            if !previous_entry.active_attachments.is_empty() {
                return Err(Error::Execution(
                    "physical authority changed a key with active logical attachments".into(),
                ));
            }
        }
        if let Some(existing) = state.entries.get(&key)
            && (existing.request != request || existing.binding != binding)
        {
            return Err(Error::Execution(
                "physical authority returned a canonical key with conflicting identity".into(),
            ));
        }
        if let Some(previous) = previous
            && previous != key
        {
            state.entries.remove(&previous);
        }
        state.request_keys.insert(request, key);
        state.entries.insert(
            key,
            AdapterEntry {
                request,
                binding,
                pending_eviction,
                resident,
                active_attachments: HashSet::default(),
                selected_lease_held,
            },
        );
        match resolution {
            PhysicalExpertReservation::Resident(binding) => {
                ExpertDependencyResolution::resident(request, binding)
            }
            PhysicalExpertReservation::Reserved(_) => {
                ExpertDependencyResolution::waiting(request, key)
            }
        }
    }

    fn detach(&mut self, continuation: ferrule_common::ContinuationId, key: LoadKey) -> Result<()> {
        detach_logical_demand(&self.state, continuation, key, true).map(|_| ())
    }
}
