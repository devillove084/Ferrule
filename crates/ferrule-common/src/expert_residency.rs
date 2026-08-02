//! Model-neutral expert residency control types and stable-slot coordination.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Globally meaningful identity for an expert within one loaded model instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExpertKey {
    pub model_instance: u64,
    pub layer: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub const fn new(model_instance: u64, layer: u32, expert: u32) -> Self {
        Self {
            model_instance,
            layer,
            expert,
        }
    }
}

/// Stable index into a layer's backend expert table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExpertSlotId(u32);

impl ExpertSlotId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// Version of the payload currently published in an expert slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExpertSlotGeneration(u32);

impl ExpertSlotGeneration {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Exact expert-to-slot mapping consumed by an execution backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExpertSlotBinding<K = ExpertKey> {
    pub key: K,
    pub slot: ExpertSlotId,
    pub generation: ExpertSlotGeneration,
}

/// Why an expert installation is being requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpertInstallReason {
    /// The expert is required by the current execution and must be leased.
    Selected,
    /// The expert is speculative cache warming and must not displace leased work.
    Prefetch,
}

/// A model-qualified request to make one expert resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExpertInstallIntent {
    pub key: ExpertKey,
    pub reason: ExpertInstallReason,
}

impl ExpertInstallIntent {
    pub const fn selected(key: ExpertKey) -> Self {
        Self {
            key,
            reason: ExpertInstallReason::Selected,
        }
    }

    pub const fn prefetch(key: ExpertKey) -> Self {
        Self {
            key,
            reason: ExpertInstallReason::Prefetch,
        }
    }
}

/// Execution lease that prevents a binding from being evicted or reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertLease<K = ExpertKey> {
    binding: ExpertSlotBinding<K>,
}

impl<K: Copy> ExpertLease<K> {
    pub const fn binding(self) -> ExpertSlotBinding<K> {
        self.binding
    }
}

/// Residency result for an already-resident or newly published install.
///
/// Selected grants always contain a lease. Prefetch grants deliberately do not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertResidencyGrant {
    binding: ExpertSlotBinding,
    reason: ExpertInstallReason,
    lease: Option<ExpertLease>,
}

impl ExpertResidencyGrant {
    pub const fn binding(self) -> ExpertSlotBinding {
        self.binding
    }

    pub const fn reason(self) -> ExpertInstallReason {
        self.reason
    }

    pub const fn lease(self) -> Option<ExpertLease> {
        self.lease
    }
}

/// Reserved slot returned by the first phase of installation.
///
/// The mapping remains unpublished until this token is passed to
/// [`ExpertResidencyControl::publish_install`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedExpertInstall<K = ExpertKey> {
    transaction: u64,
    key: K,
    reason: ExpertInstallReason,
    slot: ExpertSlotId,
    previous_key: Option<K>,
    previous_generation: ExpertSlotGeneration,
    next_generation: ExpertSlotGeneration,
}

impl<K: Copy> PreparedExpertInstall<K> {
    pub const fn binding(self) -> ExpertSlotBinding<K> {
        ExpertSlotBinding {
            key: self.key,
            slot: self.slot,
            generation: self.next_generation,
        }
    }

    pub const fn evicted_key(self) -> Option<K> {
        self.previous_key
    }

    pub const fn evicted_binding(self) -> Option<ExpertSlotBinding<K>> {
        match self.previous_key {
            Some(key) => Some(ExpertSlotBinding {
                key,
                slot: self.slot,
                generation: self.previous_generation,
            }),
            None => None,
        }
    }

    pub const fn reason(self) -> ExpertInstallReason {
        self.reason
    }
}

/// Result of the prepare phase. Capacity pressure is nonfatal for prefetches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertInstallPrepareOutcome {
    /// The expert was already resident. Selected hits carry a lease.
    Resident(ExpertResidencyGrant),
    /// A slot is reserved for backend transfer and later publication.
    Prepared(PreparedExpertInstall),
    /// No unleased, unreserved slot was available for this prefetch.
    CapacityAllLeased,
}

/// Result of activating a prepared install immediately before physical mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertInstallActivationOutcome {
    /// The old mapping is hidden and the provider exclusively owns slot mutation.
    Activated,
    /// New selected work leased the old mapping while transfer was in flight.
    BlockedByLeases,
}

/// Model namespace and independent capacity of each layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertResidencyRequirements {
    pub model_instance: u64,
    pub layer_capacities: Vec<usize>,
}

impl ExpertResidencyRequirements {
    pub fn new(model_instance: u64, layer_capacities: Vec<usize>) -> Self {
        Self {
            model_instance,
            layer_capacities,
        }
    }

    pub fn layer_capacity(&self, layer: u32) -> Option<usize> {
        self.layer_capacities.get(layer as usize).copied()
    }
}

/// Aggregate controller counters across all layers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertResidencyStats {
    pub resident: usize,
    pub active_leases: usize,
    pub installs: u64,
    pub evictions: u64,
    pub resident_hits: u64,
    pub stale_releases: u64,
    pub prepare_cancellations: u64,
    pub prefetch_capacity_misses: u64,
}

/// Per-coordinator counters retained for compatibility and layer diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertResidencyCoordinatorStats {
    pub resident: usize,
    pub active_leases: usize,
    pub installs: u64,
    pub evictions: u64,
    pub resident_hits: u64,
    pub stale_releases: u64,
    pub prepare_cancellations: u64,
}

/// Object-safe ABI between model-independent scheduling and residency ownership.
pub trait ExpertResidencyControl: Send {
    fn requirements(&self) -> ExpertResidencyRequirements;

    fn binding(&self, key: ExpertKey) -> Result<Option<ExpertSlotBinding>>;

    /// Lease an already-resident selected expert. A miss returns `Ok(None)`.
    fn acquire_selected(&mut self, key: ExpertKey) -> Result<Option<ExpertResidencyGrant>>;

    fn release(&mut self, lease: ExpertLease) -> Result<()>;

    /// Reserve an exact slot/generation without publishing a mapping.
    fn prepare_install(
        &mut self,
        intent: ExpertInstallIntent,
    ) -> Result<ExpertInstallPrepareOutcome>;

    /// Promote an in-flight prefetch reservation to selected demand without
    /// changing its slot or generation. Promotion is one-way.
    fn promote_install(&mut self, prepared: PreparedExpertInstall)
    -> Result<PreparedExpertInstall>;

    /// Hide the old mapping and acquire exclusive slot mutation authority.
    /// Providers call this only after transfer completes and before touching the
    /// physical slot. A lease conflict is retryable and leaves the old mapping live.
    fn activate_install(
        &mut self,
        prepared: PreparedExpertInstall,
    ) -> Result<ExpertInstallActivationOutcome>;

    /// Publish the exact prepared slot/generation after backend installation succeeds.
    fn publish_install(&mut self, prepared: PreparedExpertInstall) -> Result<ExpertResidencyGrant>;

    /// Cancel a prepared install after transfer failure or request cancellation.
    fn cancel_install(&mut self, prepared: PreparedExpertInstall) -> Result<()>;

    fn stats(&self) -> ExpertResidencyStats;
}

#[derive(Debug)]
struct ExpertSlot<K> {
    key: Option<K>,
    generation: ExpertSlotGeneration,
    leases: u32,
    last_used: u64,
    pending_transaction: Option<u64>,
    install_activated: bool,
}

/// Stable slot/generation/lease primitive used by residency controllers.
#[derive(Debug)]
pub struct ExpertResidencyCoordinator<K = ExpertKey> {
    slots: Vec<ExpertSlot<K>>,
    by_key: HashMap<K, ExpertSlotId>,
    pending: HashMap<u64, PreparedExpertInstall<K>>,
    clock: u64,
    next_transaction: u64,
    stats: ExpertResidencyCoordinatorStats,
}

impl<K> ExpertResidencyCoordinator<K>
where
    K: Copy + Debug + Eq + Hash,
{
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::Execution {
                message: "expert residency coordinator capacity must be greater than zero".into(),
            });
        }
        if capacity > u32::MAX as usize {
            return Err(Error::Execution {
                message: "expert residency coordinator capacity exceeds u32".into(),
            });
        }
        Ok(Self {
            slots: (0..capacity)
                .map(|_| ExpertSlot {
                    key: None,
                    generation: ExpertSlotGeneration(0),
                    leases: 0,
                    last_used: 0,
                    pending_transaction: None,
                    install_activated: false,
                })
                .collect(),
            by_key: HashMap::with_capacity(capacity),
            pending: HashMap::new(),
            clock: 0,
            next_transaction: 1,
            stats: ExpertResidencyCoordinatorStats::default(),
        })
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn binding(&self, key: K) -> Option<ExpertSlotBinding<K>> {
        let slot = *self.by_key.get(&key)?;
        let entry = &self.slots[slot.index()];
        (entry.key == Some(key)).then_some(ExpertSlotBinding {
            key,
            slot,
            generation: entry.generation,
        })
    }

    pub fn acquire(&mut self, key: K) -> Result<Option<ExpertLease<K>>> {
        let Some(slot) = self.by_key.get(&key).copied() else {
            return Ok(None);
        };
        self.clock = self.clock.saturating_add(1);
        let entry = &mut self.slots[slot.index()];
        if entry.key != Some(key) {
            return Err(Error::Internal {
                message: "expert residency key map and slot table diverged".into(),
            });
        }
        entry.leases = entry
            .leases
            .checked_add(1)
            .ok_or_else(|| Error::Execution {
                message: "expert lease count overflow".into(),
            })?;
        entry.last_used = self.clock;
        self.stats.active_leases += 1;
        self.stats.resident_hits = self.stats.resident_hits.saturating_add(1);
        Ok(Some(ExpertLease {
            binding: ExpertSlotBinding {
                key,
                slot,
                generation: entry.generation,
            },
        }))
    }

    pub fn release(&mut self, lease: ExpertLease<K>) -> Result<()> {
        let binding = lease.binding;
        let Some(entry) = self.slots.get_mut(binding.slot.index()) else {
            self.stats.stale_releases = self.stats.stale_releases.saturating_add(1);
            return Err(Error::Execution {
                message: "expert lease references a missing slot".into(),
            });
        };
        if entry.key != Some(binding.key) || entry.generation != binding.generation {
            self.stats.stale_releases = self.stats.stale_releases.saturating_add(1);
            return Err(Error::Execution {
                message: "expert lease has a stale slot generation".into(),
            });
        }
        entry.leases = entry.leases.checked_sub(1).ok_or_else(|| Error::Internal {
            message: "expert lease count underflow".into(),
        })?;
        self.stats.active_leases =
            self.stats
                .active_leases
                .checked_sub(1)
                .ok_or_else(|| Error::Internal {
                    message: "active expert lease count underflow".into(),
                })?;
        Ok(())
    }

    /// Compatibility entry point: selected installation with a fatal capacity miss.
    pub fn prepare_install(&mut self, key: K) -> Result<PreparedExpertInstall<K>> {
        self.try_prepare_install(key, ExpertInstallReason::Selected)?
            .ok_or_else(|| Error::Execution {
                message: "all expert residency slots are leased or reserved".into(),
            })
    }

    /// Reserve an install candidate, returning `None` when every slot is unavailable.
    pub fn try_prepare_install(
        &mut self,
        key: K,
        reason: ExpertInstallReason,
    ) -> Result<Option<PreparedExpertInstall<K>>> {
        if self.by_key.contains_key(&key) {
            return Err(Error::Execution {
                message: format!("expert {key:?} is already resident"),
            });
        }
        let Some((index, entry)) = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.leases == 0 && entry.pending_transaction.is_none())
            .min_by_key(|(index, entry)| (entry.key.is_some(), entry.last_used, *index))
        else {
            return Ok(None);
        };
        let next = entry
            .generation
            .0
            .checked_add(1)
            .filter(|generation| *generation != 0)
            .ok_or_else(|| Error::Execution {
                message: "expert slot generation exhausted".into(),
            })?;
        let transaction = self.next_transaction;
        self.next_transaction = self
            .next_transaction
            .checked_add(1)
            .filter(|next| *next != 0)
            .ok_or_else(|| Error::Execution {
                message: "expert install transaction IDs exhausted".into(),
            })?;
        let slot = ExpertSlotId(index as u32);
        let previous_key = entry.key;
        let prepared = PreparedExpertInstall {
            transaction,
            key,
            reason,
            slot,
            previous_key,
            previous_generation: entry.generation,
            next_generation: ExpertSlotGeneration(next),
        };
        self.slots[index].pending_transaction = Some(transaction);
        self.slots[index].install_activated = false;
        self.pending.insert(transaction, prepared);
        Ok(Some(prepared))
    }

    pub fn promote_install(
        &mut self,
        mut prepared: PreparedExpertInstall<K>,
    ) -> Result<PreparedExpertInstall<K>> {
        let stored =
            self.pending
                .get_mut(&prepared.transaction)
                .ok_or_else(|| Error::Execution {
                    message: "prepared expert install is unknown, canceled, or already published"
                        .into(),
                })?;
        if *stored != prepared {
            return Err(Error::Execution {
                message: "prepared expert install does not match coordinator ownership".into(),
            });
        }
        if prepared.reason == ExpertInstallReason::Prefetch {
            prepared.reason = ExpertInstallReason::Selected;
            stored.reason = ExpertInstallReason::Selected;
        }
        Ok(prepared)
    }

    pub fn activate_install(
        &mut self,
        prepared: PreparedExpertInstall<K>,
    ) -> Result<ExpertInstallActivationOutcome> {
        if self.pending.get(&prepared.transaction) != Some(&prepared) {
            return Err(Error::Execution {
                message: "prepared expert install is unknown, canceled, or already published"
                    .into(),
            });
        }
        let entry = self
            .slots
            .get(prepared.slot.index())
            .ok_or_else(|| Error::Internal {
                message: "prepared expert slot is missing".into(),
            })?;
        if entry.key != prepared.previous_key
            || entry.generation != prepared.previous_generation
            || entry.pending_transaction != Some(prepared.transaction)
        {
            return Err(Error::Execution {
                message: "prepared expert install became stale before activation".into(),
            });
        }
        if entry.install_activated {
            return Ok(ExpertInstallActivationOutcome::Activated);
        }
        if entry.leases != 0 {
            return Ok(ExpertInstallActivationOutcome::BlockedByLeases);
        }
        if self.by_key.contains_key(&prepared.key) {
            return Err(Error::Execution {
                message: format!(
                    "expert {:?} became resident before install activation",
                    prepared.key
                ),
            });
        }
        if let Some(previous) = prepared.previous_key {
            if self.by_key.get(&previous) != Some(&prepared.slot) {
                return Err(Error::Execution {
                    message: "prepared eviction lost its published source mapping".into(),
                });
            }
            self.by_key.remove(&previous);
        }
        self.slots[prepared.slot.index()].install_activated = true;
        Ok(ExpertInstallActivationOutcome::Activated)
    }

    pub fn publish_install(
        &mut self,
        prepared: PreparedExpertInstall<K>,
    ) -> Result<ExpertSlotBinding<K>> {
        self.publish(prepared, false).map(|(binding, _)| binding)
    }

    /// Publish and acquire the initial execution lease as one state transition.
    pub fn publish_install_leased(
        &mut self,
        prepared: PreparedExpertInstall<K>,
    ) -> Result<(ExpertSlotBinding<K>, ExpertLease<K>)> {
        let (binding, lease) = self.publish(prepared, true)?;
        Ok((
            binding,
            lease.expect("leased publication always creates a lease"),
        ))
    }

    fn publish(
        &mut self,
        prepared: PreparedExpertInstall<K>,
        acquire_lease: bool,
    ) -> Result<(ExpertSlotBinding<K>, Option<ExpertLease<K>>)> {
        if self.pending.get(&prepared.transaction) != Some(&prepared) {
            return Err(Error::Execution {
                message: "prepared expert install is unknown, canceled, or already published"
                    .into(),
            });
        }
        if self.by_key.contains_key(&prepared.key) {
            return Err(Error::Execution {
                message: format!(
                    "expert {:?} became resident before install publication",
                    prepared.key
                ),
            });
        }
        let entry = self
            .slots
            .get(prepared.slot.index())
            .ok_or_else(|| Error::Internal {
                message: "prepared expert slot is missing".into(),
            })?;
        if entry.key != prepared.previous_key
            || entry.generation != prepared.previous_generation
            || entry.leases != 0
            || entry.pending_transaction != Some(prepared.transaction)
            || !entry.install_activated
        {
            return Err(Error::Execution {
                message: "prepared expert install became stale before publication".into(),
            });
        }
        if acquire_lease && entry.leases == u32::MAX {
            return Err(Error::Execution {
                message: "expert lease count overflow".into(),
            });
        }

        let entry = &mut self.slots[prepared.slot.index()];
        if entry.key.is_some() {
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
        self.clock = self.clock.saturating_add(1);
        entry.key = Some(prepared.key);
        entry.generation = prepared.next_generation;
        entry.last_used = self.clock;
        entry.pending_transaction = None;
        entry.install_activated = false;
        entry.leases = u32::from(acquire_lease);
        self.pending.remove(&prepared.transaction);
        self.by_key.insert(prepared.key, prepared.slot);
        self.stats.installs = self.stats.installs.saturating_add(1);
        self.stats.resident = self.slots.iter().filter(|slot| slot.key.is_some()).count();
        if acquire_lease {
            self.stats.active_leases += 1;
        }
        let binding = prepared.binding();
        let lease = acquire_lease.then_some(ExpertLease { binding });
        Ok((binding, lease))
    }

    pub fn cancel_install(&mut self, prepared: PreparedExpertInstall<K>) -> Result<()> {
        if self.pending.get(&prepared.transaction) != Some(&prepared) {
            return Err(Error::Execution {
                message: "prepared expert install is unknown, canceled, or already published"
                    .into(),
            });
        }
        let entry = self
            .slots
            .get(prepared.slot.index())
            .ok_or_else(|| Error::Internal {
                message: "prepared expert slot is missing".into(),
            })?;
        if entry.key != prepared.previous_key
            || entry.generation != prepared.previous_generation
            || entry.pending_transaction != Some(prepared.transaction)
        {
            return Err(Error::Execution {
                message: "prepared expert install reservation became stale".into(),
            });
        }
        if let Some(previous) = prepared.previous_key {
            match (entry.install_activated, self.by_key.get(&previous)) {
                (false, Some(slot)) if *slot == prepared.slot => {}
                (true, None) => {
                    self.by_key.insert(previous, prepared.slot);
                }
                _ => {
                    return Err(Error::Execution {
                        message: "prepared install cancellation found divergent source mapping"
                            .into(),
                    });
                }
            }
        }
        let entry = &mut self.slots[prepared.slot.index()];
        entry.pending_transaction = None;
        entry.install_activated = false;
        self.pending.remove(&prepared.transaction);
        self.stats.prepare_cancellations = self.stats.prepare_cancellations.saturating_add(1);
        Ok(())
    }

    pub fn evict(&mut self, key: K) -> Result<Option<ExpertSlotBinding<K>>> {
        let Some(slot) = self.by_key.get(&key).copied() else {
            return Ok(None);
        };
        let entry = &mut self.slots[slot.index()];
        if entry.leases != 0 {
            return Err(Error::Execution {
                message: format!("expert {key:?} cannot be evicted while leased"),
            });
        }
        if entry.pending_transaction.is_some() {
            return Err(Error::Execution {
                message: format!("expert {key:?} cannot be evicted while its slot is reserved"),
            });
        }
        let binding = ExpertSlotBinding {
            key,
            slot,
            generation: entry.generation,
        };
        entry.key = None;
        entry.last_used = 0;
        self.by_key.remove(&key);
        self.stats.evictions = self.stats.evictions.saturating_add(1);
        self.stats.resident = self.slots.iter().filter(|slot| slot.key.is_some()).count();
        Ok(Some(binding))
    }

    pub fn stats(&self) -> ExpertResidencyCoordinatorStats {
        self.stats
    }
}

impl ExpertResidencyGrant {
    /// Construct a grant from a coordinator-produced binding and lease.
    pub const fn new(
        binding: ExpertSlotBinding,
        reason: ExpertInstallReason,
        lease: Option<ExpertLease>,
    ) -> Self {
        Self {
            binding,
            reason,
            lease,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn install(
        coordinator: &mut ExpertResidencyCoordinator<u32>,
        key: u32,
    ) -> ExpertSlotBinding<u32> {
        let prepared = coordinator.prepare_install(key).unwrap();
        let expected = prepared.binding();
        assert_eq!(
            coordinator.activate_install(prepared).unwrap(),
            ExpertInstallActivationOutcome::Activated
        );
        assert_eq!(coordinator.publish_install(prepared).unwrap(), expected);
        expected
    }

    #[test]
    fn cancel_preserves_published_mapping_and_releases_reservation() {
        let mut coordinator = ExpertResidencyCoordinator::new(1).unwrap();
        let old = install(&mut coordinator, 1);
        let failed = coordinator.prepare_install(2).unwrap();

        assert_eq!(failed.evicted_key(), Some(1));
        assert_eq!(failed.evicted_binding(), Some(old));
        assert_eq!(coordinator.binding(1), Some(old));
        assert_eq!(coordinator.binding(2), None);
        assert_eq!(
            coordinator.activate_install(failed).unwrap(),
            ExpertInstallActivationOutcome::Activated
        );
        assert_eq!(coordinator.binding(1), None);
        coordinator.cancel_install(failed).unwrap();
        assert_eq!(coordinator.binding(1), Some(old));
        assert!(coordinator.publish_install(failed).is_err());

        let retry = coordinator.prepare_install(2).unwrap();
        let expected = retry.binding();
        assert_eq!(
            coordinator.activate_install(retry).unwrap(),
            ExpertInstallActivationOutcome::Activated
        );
        assert_eq!(coordinator.publish_install(retry).unwrap(), expected);
        assert_eq!(expected.slot, old.slot);
        assert!(expected.generation.get() > old.generation.get());
    }

    #[test]
    fn prepared_eviction_remains_leasable_until_activation_then_hides_atomically() {
        let mut coordinator = ExpertResidencyCoordinator::new(2).unwrap();
        let first = install(&mut coordinator, 1);
        let second = install(&mut coordinator, 2);
        let replacement = coordinator.prepare_install(3).unwrap();
        assert_eq!(replacement.evicted_key(), Some(1));
        assert_eq!(replacement.evicted_binding(), Some(first));
        assert_eq!(coordinator.binding(1), Some(first));

        let lease = coordinator.acquire(1).unwrap().unwrap();
        assert_eq!(
            coordinator.activate_install(replacement).unwrap(),
            ExpertInstallActivationOutcome::BlockedByLeases
        );
        assert_eq!(coordinator.binding(1), Some(first));

        coordinator.release(lease).unwrap();
        assert_eq!(
            coordinator.activate_install(replacement).unwrap(),
            ExpertInstallActivationOutcome::Activated
        );
        assert_eq!(coordinator.binding(1), None);
        assert!(coordinator.acquire(1).unwrap().is_none());

        coordinator.cancel_install(replacement).unwrap();
        assert_eq!(coordinator.binding(1), Some(first));
        assert_eq!(coordinator.binding(2), Some(second));
    }

    #[test]
    fn leases_block_slot_reuse_and_stale_release_is_rejected() {
        let mut coordinator = ExpertResidencyCoordinator::new(1).unwrap();
        let old = install(&mut coordinator, 1);
        let lease = coordinator.acquire(1).unwrap().unwrap();
        let stale = lease;

        assert!(coordinator.prepare_install(2).is_err());
        coordinator.release(lease).unwrap();
        let new = install(&mut coordinator, 2);
        assert_eq!(new.slot, old.slot);
        assert!(new.generation.get() > old.generation.get());
        assert!(coordinator.release(stale).is_err());
        assert_eq!(coordinator.stats().stale_releases, 1);
    }

    #[test]
    fn lru_choice_is_deterministic_among_unleased_slots() {
        let mut coordinator = ExpertResidencyCoordinator::new(2).unwrap();
        let first = install(&mut coordinator, 1);
        let second = install(&mut coordinator, 2);
        let touch = coordinator.acquire(1).unwrap().unwrap();
        coordinator.release(touch).unwrap();

        let third = install(&mut coordinator, 3);
        assert_eq!(third.slot, second.slot);
        assert_ne!(third.slot, first.slot);
        assert!(coordinator.binding(1).is_some());
        assert!(coordinator.binding(2).is_none());
    }
}
