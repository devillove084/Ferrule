//! One physical arena, immutable exact-class views, and the existing lease/LRU.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]
use super::decoded_snapshot::{
    MixedDecodedCacheActivity, MixedDecodedCachePhase, MixedDecodedCacheSnapshot,
};
use super::expert_cache::{Cache, ExpertCacheLimits, ExpertCacheStats, Lease};
use crate::transformer::hybrid_parallel::HybridMixedDecodedCacheBudget;
use crate::transformer::hybrid_parallel::mixed_decoded::{DecodedDescriptor, DecodedKey};
use crate::transformer::hybrid_parallel::mixed_decoded_arena::{
    DecodedArenaDescriptors, DecodedArenaPlan,
};
use ferrule_common::{Error, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    sync::Arc,
};

pub(super) fn invalid(message: &str) -> Error {
    Error::Model {
        message: format!("Mixed decoded custody: {message}"),
    }
}

pub(super) enum Lookup<T> {
    Hit(Rc<T>),
    Decode(Rc<T>),
    Native,
}
type Class = (usize, usize);
struct Backing<T> {
    view: Rc<T>,
    class: Class,
    used: bool,
}
type Pool<T> = BTreeMap<Class, Vec<Backing<T>>>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct PhysicalStats {
    /// Logical slots, not additional allocations on top of the root.
    pub active: usize,
    pub pooled: usize,
    pub free: usize,
    pub unused_tail: usize,
    /// Full root reservation, including its unused tail, retained on failure.
    pub reserved_grants: usize,
    pub arena_backing: usize,
    pub driver_growth: usize,
    pub peak: usize,
    pub unknown: usize,
    pub reuses: u64,
    pub fallbacks: u64,
}
impl PhysicalStats {
    pub fn charged(self) -> usize {
        self.reserved_grants
    }
}

pub(super) struct DecodedCustody<T> {
    ledger: Cache<DecodedKey, Class>,
    plan: Arc<DecodedArenaPlan>,
    descriptors: BTreeMap<DecodedKey, (Class, Option<usize>)>,
    fixed_keys: Vec<DecodedKey>,
    // Retained independently of all views, including partial slice creation.
    arena: Option<T>,
    initialized: bool,
    allocation_pending: bool,
    roots: BTreeMap<DecodedKey, Backing<T>>,
    pool: Pool<T>,
    pending: BTreeMap<DecodedKey, Lease<DecodedKey>>,
    ready: BTreeSet<DecodedKey>,
    borrowed: BTreeSet<DecodedKey>,
    generation: u64,
    physical: PhysicalStats,
    pinned_allocated: usize,
    profiled_activity: Option<MixedDecodedCacheActivity>,
}

fn pool_binding<T>(
    roots: &mut BTreeMap<DecodedKey, Backing<T>>,
    pool: &mut Pool<T>,
    ready: &mut BTreeSet<DecodedKey>,
    physical: &mut PhysicalStats,
    key: &DecodedKey,
) {
    if let Some(backing) = roots.remove(key) {
        physical.active -= backing.class.1;
        physical.pooled += backing.class.1;
        pool.entry(backing.class).or_default().push(backing);
    }
    ready.remove(key);
}

impl<T> DecodedCustody<T> {
    /// Pure CPU preflight, before any allocation or destruction of useful state.
    pub fn new(
        budget: HybridMixedDecodedCacheBudget,
        generation: u64,
        ds: &DecodedArenaDescriptors,
    ) -> Result<Self> {
        ds.validate(budget)?;
        let plan = ds.layout.clone();
        let mut ledger = Cache::with_groups(
            ExpertCacheLimits {
                max_experts: budget.routed_entries.max(1),
                max_bytes: budget.capacity_grants,
            },
            ds.iter()
                .filter(|d| d.routed_layer.is_some())
                .map(|d| (d.key(generation), (d.elements(), d.grants))),
        )?;
        ledger.scratch(budget.pinned_grants, |_| unreachable!())?;
        let fixed_keys = plan
            .fixed
            .iter()
            .map(|s| ds[s.descriptor].key(generation))
            .collect();
        let descriptors = ds
            .iter()
            .map(|d| {
                (
                    d.key(generation),
                    ((d.elements(), d.grants), d.routed_layer),
                )
            })
            .collect();
        Ok(Self {
            ledger,
            plan,
            descriptors,
            fixed_keys,
            arena: None,
            initialized: false,
            allocation_pending: true,
            roots: BTreeMap::new(),
            pool: BTreeMap::new(),
            pending: BTreeMap::new(),
            ready: BTreeSet::new(),
            borrowed: BTreeSet::new(),
            generation,
            pinned_allocated: 0,
            profiled_activity: None,
            physical: PhysicalStats {
                reserved_grants: budget.capacity_grants,
                peak: budget.capacity_grants,
                ..Default::default()
            },
        })
    }
    pub fn plan(&self) -> &DecodedArenaPlan {
        &self.plan
    }
    pub fn stats(&self) -> ExpertCacheStats {
        self.ledger.stats()
    }
    pub fn physical_stats(&self) -> PhysicalStats {
        self.physical
    }

    /// Transfer the ONE full-grant allocation before fallible view construction.
    /// `slice` must return a non-allocating BF16 view at (offset_words, elements),
    /// with shared storage lifetime and no externally retained mutable aliases.
    /// No CUDA submission is permitted in this closure. Views are never charged.
    pub fn install_arena(
        &mut self,
        root: T,
        growth: usize,
        mut slice: impl FnMut(&T, usize, usize) -> Result<T>,
    ) -> Result<()> {
        if self.ledger.stats().quarantined || !self.allocation_pending || self.arena.is_some() {
            std::mem::forget(root);
            self.quarantine();
            return Err(invalid("unreserved or duplicate arena"));
        }
        self.arena = Some(root);
        self.allocation_pending = false;
        self.physical.arena_backing = self.plan.arena_grants;
        self.physical.driver_growth = growth;
        self.physical.unused_tail = self.plan.unused_bytes;
        if growth > self.plan.arena_grants {
            self.quarantine();
            return Err(invalid("allocator exceeded full arena grant"));
        }
        let result = (|| {
            for (slot, key) in self.plan.fixed.iter().zip(&self.fixed_keys) {
                let view = slice(
                    self.arena.as_ref().unwrap(),
                    slot.offset_words,
                    slot.elements,
                )?;
                self.roots.insert(
                    key.clone(),
                    Backing {
                        view: Rc::new(view),
                        class: (slot.elements, slot.slot_bytes),
                        used: true,
                    },
                );
                self.pinned_allocated += slot.slot_bytes;
                self.physical.active += slot.slot_bytes;
            }
            for class in &self.plan.classes {
                let slots = self
                    .pool
                    .entry((class.elements, class.slot_bytes))
                    .or_default();
                for i in 0..class.count {
                    let offset = class.offset_words + i * (class.slot_bytes / 2);
                    let view = slice(self.arena.as_ref().unwrap(), offset, class.elements)?;
                    slots.push(Backing {
                        view: Rc::new(view),
                        class: (class.elements, class.slot_bytes),
                        used: false,
                    });
                    self.physical.free += class.slot_bytes;
                }
            }
            Ok(())
        })();
        if result.is_err() {
            self.quarantine();
        } else {
            self.initialized = true;
        }
        result
    }

    /// No retry/refund: a failed backend allocation may have queued retirement.
    pub fn arena_allocation_failed(&mut self, growth: usize) -> Result<()> {
        self.ledger.preflight(0, 0)?;
        if !self.allocation_pending || self.arena.is_some() {
            self.quarantine();
            return Err(invalid("missing arena allocation reservation"));
        }
        self.allocation_pending = false;
        self.physical.driver_growth = growth;
        self.quarantine();
        if growth > self.plan.arena_grants {
            return Err(invalid("failed allocator exceeded full arena grant"));
        }
        Ok(())
    }

    pub fn quarantine(&mut self) {
        self.ledger.quarantine();
        self.physical.unknown = self.physical.charged();
    }

    fn descriptor_key(&self, d: &DecodedDescriptor) -> Result<DecodedKey> {
        let key = d.key(self.generation);
        let elements = d.identity.geometry[0]
            .checked_mul(d.identity.geometry[1])
            .ok_or_else(|| invalid("descriptor extent overflow"))?;
        if self.descriptors.get(&key) != Some(&((elements, d.grants), d.routed_layer)) {
            return Err(invalid("descriptor is not in the admitted arena plan"));
        }
        Ok(key)
    }

    fn lease(&mut self, d: &DecodedDescriptor, key: &DecodedKey) -> Result<()> {
        if d.routed_layer.is_some() && !self.pending.contains_key(key) {
            // A selected empty slot proves both logical byte and entry capacity.
            // Thus byte admission must never choose a second, cross-class victim.
            let lease = self.ledger.acquire(key.clone(), d.grants, |_| {
                unreachable!("arena slot proves logical capacity")
            })?;
            self.pending.insert(key.clone(), lease);
        }
        self.borrowed.insert(key.clone());
        Ok(())
    }

    pub fn acquire(&mut self, d: &DecodedDescriptor) -> Result<Lookup<T>> {
        self.ledger.preflight(0, 0)?;
        let key = self.descriptor_key(d)?;
        if !self.initialized {
            return Err(invalid("arena not installed"));
        }
        if let Some(backing) = self.roots.get(&key) {
            let ready = self.ready.contains(&key);
            if !ready && self.borrowed.contains(&key) {
                return Err(invalid("incomplete decode cannot be consumed"));
            }
            let view = backing.view.clone();
            self.lease(d, &key)?;
            return Ok(if ready {
                Lookup::Hit(view)
            } else {
                Lookup::Decode(view)
            });
        }
        let class = (d.elements(), d.grants);
        if self.pool.get(&class).is_none_or(|slots| slots.is_empty()) {
            // Only a full class may evict, and only that class's completed LRU.
            let evicted = self.ledger.evict_one_completed_in_group(&class, |victim| {
                pool_binding(
                    &mut self.roots,
                    &mut self.pool,
                    &mut self.ready,
                    &mut self.physical,
                    victim,
                )
            })?;
            if !evicted {
                self.physical.fallbacks += 1;
                return Ok(Lookup::Native);
            }
        }
        self.lease(d, &key)?;
        let slots = self.pool.get_mut(&class).expect("planned class");
        let mut backing = slots.pop().expect("empty or evicted slot");
        if backing.used {
            self.physical.pooled -= d.grants;
            self.physical.reuses += 1;
        } else {
            self.physical.free -= d.grants;
            backing.used = true;
        }
        self.physical.active += d.grants;
        let view = backing.view.clone();
        self.roots.insert(key, backing);
        Ok(Lookup::Decode(view))
    }

    /// Successful submission permits same-stream reads, not Idle publication.
    pub fn mark_ready(&mut self, d: &DecodedDescriptor) -> Result<()> {
        self.ledger.preflight(0, 0)?;
        let key = self.descriptor_key(d)?;
        if !self.borrowed.contains(&key) || !self.roots.contains_key(&key) {
            return Err(invalid("decode without borrowed custody"));
        }
        self.ready.insert(key);
        Ok(())
    }

    /// Only the existing outermost healthy completion makes slots reusable.
    /// `false` is a proved, fenced failure; Unknown MUST call quarantine instead.
    pub fn complete(&mut self, succeeded: bool) -> Result<()> {
        self.ledger.preflight(0, 0)?;
        if !self.initialized
            || self
                .borrowed
                .iter()
                .filter_map(|k| self.roots.get(k))
                .any(|b| Rc::strong_count(&b.view) != 1 || Rc::weak_count(&b.view) != 0)
        {
            self.quarantine();
            return Err(invalid("borrow/allocation survives final consumer"));
        }
        for (key, lease) in &self.pending {
            let installed = succeeded && self.ready.contains(key);
            if !installed {
                pool_binding(
                    &mut self.roots,
                    &mut self.pool,
                    &mut self.ready,
                    &mut self.physical,
                    key,
                );
            }
            if let Err(error) = self.ledger.complete(lease, installed) {
                self.quarantine();
                return Err(error);
            }
        }
        if !succeeded {
            for key in &self.borrowed {
                self.ready.remove(key);
            }
        }
        self.pending.clear();
        self.borrowed.clear();
        Ok(())
    }

    pub fn record_profiled_invocation(&mut self, counters: [u64; 4]) {
        self.profiled_activity = Some(
            self.profiled_activity
                .unwrap_or_default()
                .with_invocation(counters),
        );
    }

    pub fn snapshot(
        &self,
        published: bool,
        executing: bool,
        owner_quarantined: bool,
        rank_performance_enabled: bool,
        active_profile: Option<[u64; 4]>,
    ) -> MixedDecodedCacheSnapshot {
        let logical = self.ledger.stats();
        let p = self.physical;
        let quarantined = owner_quarantined || logical.quarantined;
        let phase = if quarantined {
            MixedDecodedCachePhase::Quarantined
        } else if !published {
            MixedDecodedCachePhase::Preparing
        } else if executing {
            MixedDecodedCachePhase::Executing
        } else {
            MixedDecodedCachePhase::Idle
        };
        let profiled_activity = match active_profile {
            Some(counters) => Some(
                self.profiled_activity
                    .unwrap_or_default()
                    .with_invocation(counters),
            ),
            None => self.profiled_activity,
        };
        MixedDecodedCacheSnapshot {
            image_generation: self.generation,
            phase,
            published,
            quarantined,
            capacity_grants: self.plan.arena_grants,
            arena_backing_grants: p.arena_backing,
            pin_target_grants: logical.scratch_bytes,
            pinned_grants: self.pinned_allocated,
            unallocated_pin_grants: logical.scratch_bytes - self.pinned_allocated,
            active_routed_grants: p.active - self.pinned_allocated,
            pooled_grants: p.pooled,
            free_slot_grants: p.free,
            unused_tail_bytes: p.unused_tail,
            slot_padding_bytes: self.plan.slot_padding_bytes,
            pending_reserved_grants: p.reserved_grants
                - p.active
                - p.pooled
                - p.free
                - p.unused_tail,
            pending_allocation_count: usize::from(self.allocation_pending),
            pending_lease_count: self.pending.len(),
            borrowed_key_count: self.borrowed.len(),
            segment_slack_bytes: 0,
            pending_segment_slack_bytes: 0,
            physical_current_bytes: p.charged(),
            physical_peak_bytes: p.peak,
            driver_allocation_growth_bytes: p.driver_growth,
            unknown_reserved_bytes: if quarantined { p.charged() } else { p.unknown },
            backing_reuses: p.reuses,
            native_fallbacks: p.fallbacks,
            evictions: logical.evictions,
            rank_performance_enabled,
            rank_performance_active: active_profile.is_some(),
            profiled_activity,
        }
    }
}
impl<T> Drop for DecodedCustody<T> {
    fn drop(&mut self) {
        if self.ledger.stats().quarantined
            || (!self.initialized && self.arena.is_some())
            || !self.borrowed.is_empty()
            || !self.pending.is_empty()
        {
            std::mem::forget(self.arena.take());
            std::mem::forget(std::mem::take(&mut self.roots));
            std::mem::forget(std::mem::take(&mut self.pool));
        }
    }
}
#[cfg(test)]
#[path = "../../../tests/unit/transformer/mixed_decoded_snapshot.rs"]
mod snapshot_tests;
#[cfg(test)]
#[path = "../../../tests/unit/transformer/mixed_decoded_custody.rs"]
mod tests;
