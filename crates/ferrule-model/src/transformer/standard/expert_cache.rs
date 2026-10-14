//! CPU-testable accounting for the standard CUDA expert cache.
//!
//! A budget covers dense F32 payloads (including converted BF16) or compressed
//! numeric FP8 weights plus scales, pending uploads, numeric workspace reservation
//! and operation scratch. It excludes allocator segment slack, KV and non-expert
//! weights; the caller must leave room for those separately.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use crate::nn::ParameterId;
use crate::support::TensorRole;
use ferrule_common::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExpertCachePolicy {
    /// Historical policy: retain every visited expert for this image.
    #[default]
    KeepAll,
    /// LRU among completed experts only. Never evicts a pending consumer.
    Bounded(ExpertCacheLimits),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertCacheLimits {
    /// Resident and pending expert slots; independent of router top-k.
    pub max_experts: usize,
    /// Device payload budget: resident weights (compressed including scales for
    /// numeric FP8) + pending upload reservations + numeric/operation scratch.
    /// Both limits must be nonzero.
    pub max_bytes: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertCacheStats {
    pub resident_experts: usize,
    pub resident_bytes: usize,
    pub pending_upload_bytes: usize,
    pub scratch_bytes: usize,
    pub peak_bytes: usize,
    pub uploads: u64,
    pub hits: u64,
    pub evictions: u64,
    pub quarantined: bool,
    /// Subset of charged bytes deliberately retained after unknown completion;
    /// not an additional charge on top of resident/pending/scratch.
    pub unknown_quarantine_bytes: usize,
}

impl ExpertCacheStats {
    pub fn charged_bytes(self) -> usize {
        self.resident_bytes + self.pending_upload_bytes + self.scratch_bytes
    }
}

fn error(message: &str) -> Error {
    Error::Model {
        message: format!("bounded CUDA expert cache: {message}"),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct ExpertCacheKey {
    pub image_generation: u64,
    pub numeric_precision: Option<super::NumericFp8Precision>,
    pub layer: usize,
    pub expert: usize,
    pub parameters: [(ParameterId, TensorRole); 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Uploading,
    Consuming,
    Idle,
}

#[derive(Debug)]
struct Entry<G> {
    bytes: usize,
    epoch: u64,
    state: State,
    group: Option<G>,
}

/// Epochs bind completion to an exact use, so stale proofs cannot retire a reload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Lease<K> {
    pub key: K,
    epoch: u64,
}

// Registration is frozen before the first lease. Only completed entries appear
// in the ordered sets; (epoch, key) also preserves the linear LRU tie break.
struct CompletedGroups<K, G> {
    registration: BTreeMap<K, G>,
    completed: BTreeMap<G, BTreeSet<(u64, K)>>,
}

pub(super) struct Cache<K, G = ()> {
    limits: ExpertCacheLimits,
    entries: BTreeMap<K, Entry<G>>,
    groups: Option<Box<CompletedGroups<K, G>>>,
    clock: u64,
    stats: ExpertCacheStats,
}

impl<K: Ord + Clone> Cache<K> {
    pub fn new(limits: ExpertCacheLimits) -> Result<Self> {
        Self::empty(limits, None)
    }
}

impl<K: Ord + Clone, G: Ord + Clone> Cache<K, G> {
    /// Opt-in, immutable key-to-group registration. Unregistered keys and even
    /// identical duplicate registrations are rejected, never silently rebound.
    pub fn with_groups(
        limits: ExpertCacheLimits,
        registration: impl IntoIterator<Item = (K, G)>,
    ) -> Result<Self> {
        let mut groups = CompletedGroups {
            registration: BTreeMap::new(),
            completed: BTreeMap::new(),
        };
        for (key, group) in registration {
            if groups.registration.insert(key, group).is_some() {
                return Err(error("duplicate group registration"));
            }
        }
        Self::empty(limits, Some(Box::new(groups)))
    }

    fn empty(
        limits: ExpertCacheLimits,
        groups: Option<Box<CompletedGroups<K, G>>>,
    ) -> Result<Self> {
        if limits.max_experts == 0 || limits.max_bytes == 0 {
            return Err(error("limits must be nonzero"));
        }
        Ok(Self {
            limits,
            entries: BTreeMap::new(),
            groups,
            clock: 0,
            stats: ExpertCacheStats::default(),
        })
    }

    pub fn stats(&self) -> ExpertCacheStats {
        self.stats
    }
    pub fn quarantine(&mut self) {
        self.stats.quarantined = true;
        if let Some(groups) = &mut self.groups {
            groups.completed.clear();
        }
        self.stats.unknown_quarantine_bytes = self
            .stats
            .unknown_quarantine_bytes
            .max(self.stats.charged_bytes());
    }
    fn healthy(&self) -> Result<()> {
        if self.stats.quarantined {
            Err(error("completion unknown; owner quarantined"))
        } else {
            Ok(())
        }
    }
    fn tick(&mut self) -> Result<u64> {
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or_else(|| error("epoch exhausted"))?;
        Ok(self.clock)
    }
    fn peak(&mut self) {
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.charged_bytes());
    }

    /// The caller removes idle bindings before logical credit is released.
    fn make_room(&mut self, extra: usize, slot: bool, mut evict: impl FnMut(&K)) -> Result<()> {
        self.healthy()?;
        if extra > self.limits.max_bytes {
            return Err(error("one operation exceeds byte limit"));
        }
        while self.stats.charged_bytes() > self.limits.max_bytes - extra
            || (slot && self.entries.len() >= self.limits.max_experts)
        {
            if !self.evict_one_completed(&mut evict)? {
                return Err(error("budget exhausted; no completed expert to evict"));
            }
        }
        Ok(())
    }

    /// The same completed-only LRU victim selection used by byte admission.
    /// This releases a logical binding, not physical allocator retirement.
    /// A caller retaining a backing pool must charge that pool independently.
    pub fn evict_one_completed(&mut self, evict: impl FnMut(&K)) -> Result<bool> {
        self.evict_one_completed_if(|_| true, evict)
    }

    /// Restrict the existing completed-only LRU to compatible physical slots.
    pub fn evict_one_completed_if(
        &mut self,
        mut eligible: impl FnMut(&K) -> bool,
        evict: impl FnMut(&K),
    ) -> Result<bool> {
        self.healthy()?;
        let Some(key) = self
            .entries
            .iter()
            .filter(|(key, entry)| entry.state == State::Idle && eligible(key))
            .min_by_key(|(_, entry)| entry.epoch)
            .map(|(key, _)| key.clone())
        else {
            return Ok(false);
        };
        self.evict_completed_key(key, evict);
        Ok(true)
    }

    /// Select only the requested completed LRU, without visiting other entries
    /// or consulting registration/descriptors. Pending/consuming uses are absent.
    pub fn evict_one_completed_in_group(
        &mut self,
        group: &G,
        evict: impl FnMut(&K),
    ) -> Result<bool> {
        self.healthy()?;
        let groups = self
            .groups
            .as_ref()
            .ok_or_else(|| error("groups not registered"))?;
        let Some((_, key)) = groups.completed.get(group).and_then(BTreeSet::first) else {
            return Ok(false);
        };
        self.evict_completed_key(key.clone(), evict);
        Ok(true)
    }

    fn evict_completed_key(&mut self, key: K, mut evict: impl FnMut(&K)) {
        evict(&key);
        let entry = self.entries.remove(&key).expect("idle victim");
        Self::remove_completed(&mut self.groups, &key, &entry);
        self.stats.resident_bytes -= entry.bytes;
        self.stats.resident_experts -= 1;
        self.stats.evictions += 1;
    }

    fn remove_completed(
        groups: &mut Option<Box<CompletedGroups<K, G>>>,
        key: &K,
        entry: &Entry<G>,
    ) {
        if let (Some(groups), Some(group)) = (groups, &entry.group) {
            let completed = groups.completed.get_mut(group).expect("idle group");
            let removed = completed.remove(&(entry.epoch, key.clone()));
            debug_assert!(removed);
        }
    }

    pub fn scratch(&mut self, bytes: usize, evict: impl FnMut(&K)) -> Result<()> {
        self.healthy()?;
        if bytes > self.stats.scratch_bytes {
            self.make_room(bytes - self.stats.scratch_bytes, false, evict)?;
        }
        self.stats.scratch_bytes = bytes;
        self.peak();
        Ok(())
    }

    /// Pure intrinsic admission check: invalid plans must not evict useful state.
    pub fn preflight(&self, bytes: usize, scratch: usize) -> Result<()> {
        self.healthy()?;
        if bytes
            .checked_add(scratch)
            .is_none_or(|n| n > self.limits.max_bytes)
        {
            return Err(error("expert plus scratch exceeds byte limit"));
        }
        Ok(())
    }

    /// Reserve the entire expert before the first upload. Hits are also pinned.
    pub fn acquire(&mut self, key: K, bytes: usize, evict: impl FnMut(&K)) -> Result<Lease<K>> {
        self.healthy()?;
        let epoch = self.tick()?;
        if let Some(entry) = self.entries.get_mut(&key) {
            if entry.bytes != bytes || entry.state != State::Idle {
                return Err(error("changed binding or overlapping expert use"));
            }
            Self::remove_completed(&mut self.groups, &key, entry);
            entry.state = State::Consuming;
            entry.epoch = epoch;
            self.stats.hits += 1;
        } else {
            // Reject an intrinsically oversized operation without destroying useful entries.
            if bytes
                .checked_add(self.stats.scratch_bytes)
                .is_none_or(|n| n > self.limits.max_bytes)
            {
                return Err(error("expert plus scratch exceeds byte limit"));
            }
            let group = self
                .groups
                .as_ref()
                .map(|groups| {
                    groups
                        .registration
                        .get(&key)
                        .cloned()
                        .ok_or_else(|| error("unregistered group key"))
                })
                .transpose()?;
            self.make_room(bytes, true, evict)?;
            self.entries.insert(
                key.clone(),
                Entry {
                    bytes,
                    epoch,
                    state: State::Uploading,
                    group,
                },
            );
            self.stats.pending_upload_bytes += bytes;
            self.peak();
        }
        Ok(Lease { key, epoch })
    }

    /// Called only after upload synchronization and a consumer event succeed.
    /// On a failed operation, the physical partial install is removed first.
    pub fn complete(&mut self, lease: &Lease<K>, installed: bool) -> Result<()> {
        self.healthy()?;
        let entry = self
            .entries
            .get_mut(&lease.key)
            .ok_or_else(|| error("missing lease"))?;
        if entry.epoch != lease.epoch || entry.state == State::Idle {
            return Err(error("stale completion epoch"));
        }
        if entry.state == State::Uploading {
            self.stats.pending_upload_bytes -= entry.bytes;
            if installed {
                self.stats.resident_bytes += entry.bytes;
                self.stats.resident_experts += 1;
                self.stats.uploads += 1;
            }
        } else if !installed {
            self.stats.resident_bytes -= entry.bytes;
            self.stats.resident_experts -= 1;
        }
        if installed {
            entry.state = State::Idle;
            if let (Some(groups), Some(group)) = (&mut self.groups, &entry.group) {
                let inserted = groups
                    .completed
                    .entry(group.clone())
                    .or_default()
                    .insert((entry.epoch, lease.key.clone()));
                debug_assert!(inserted);
            }
        } else {
            self.entries.remove(&lease.key);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cache(entries: usize, bytes: usize) -> Cache<u64> {
        Cache::new(ExpertCacheLimits {
            max_experts: entries,
            max_bytes: bytes,
        })
        .unwrap()
    }
    #[test]
    fn forced_eviction_reload_and_counts() {
        let mut cache = cache(1, 20);
        cache.scratch(4, |_| panic!()).unwrap();
        for key in [1, 2, 1, 3, 2, 1] {
            let lease = cache.acquire(key, 16, |_| {}).unwrap();
            assert_eq!(cache.stats().pending_upload_bytes, 16);
            assert_eq!(cache.stats().charged_bytes(), 20);
            cache.complete(&lease, true).unwrap();
            assert_eq!(cache.stats().resident_experts, 1);
        }
        assert_eq!(cache.stats().uploads, 6);
        assert_eq!(cache.stats().evictions, 5);
        assert_eq!(cache.stats().peak_bytes, 20);
    }
    #[test]
    fn image_roles_and_reload_epochs_cannot_alias() {
        let original = ExpertCacheKey {
            image_generation: 1,
            numeric_precision: Some(super::super::NumericFp8Precision::Bf16RneF32Accumulate),
            layer: 0,
            expert: 0,
            parameters: [
                (ParameterId::new(1), TensorRole::RoutedExpertGate),
                (ParameterId::new(2), TensorRole::RoutedExpertUp),
                (ParameterId::new(3), TensorRole::RoutedExpertDown),
            ],
        };
        let mut other_precision = original.clone();
        other_precision.numeric_precision = Some(super::super::NumericFp8Precision::F32Tf32x3);
        assert_ne!(original, other_precision);
        let mut other_image = original.clone();
        other_image.image_generation += 1;
        let mut other_role = original.clone();
        other_role.parameters[0].1 = TensorRole::RoutedExpertUp;
        assert_ne!(original, other_role);
        assert_ne!(original, other_image);
        let mut cache = Cache::new(ExpertCacheLimits {
            max_experts: 1,
            max_bytes: 16,
        })
        .unwrap();
        let old = cache.acquire(original.clone(), 16, |_| panic!()).unwrap();
        cache.complete(&old, true).unwrap();
        let other = cache.acquire(other_image, 16, |_| {}).unwrap();
        cache.complete(&other, true).unwrap();
        let reloaded = cache.acquire(original, 16, |_| {}).unwrap();
        assert!(cache.complete(&old, true).is_err());
        cache.complete(&reloaded, true).unwrap();
        assert_eq!(cache.stats().evictions, 2);
    }

    #[test]
    fn pending_and_consuming_are_not_eviction_proofs() {
        let mut cache = cache(1, 32);
        let lease = cache.acquire(1, 16, |_| panic!()).unwrap();
        assert!(cache.acquire(2, 16, |_| panic!()).is_err());
        cache.complete(&lease, true).unwrap();
        let use_again = cache.acquire(1, 16, |_| panic!()).unwrap();
        assert!(cache.complete(&lease, true).is_err());
        assert!(cache.acquire(2, 16, |_| panic!()).is_err());
        cache.complete(&use_again, true).unwrap();
    }
    #[test]
    fn unknown_is_sticky_and_never_refunds_even_with_late_proof() {
        for uploaded in [false, true] {
            let mut cache = cache(1, 20);
            cache.scratch(4, |_| panic!()).unwrap();
            let mut lease = cache.acquire(1, 16, |_| panic!()).unwrap();
            if uploaded {
                cache.complete(&lease, true).unwrap();
                lease = cache.acquire(1, 16, |_| panic!()).unwrap();
            }
            cache.quarantine();
            let before = cache.stats();
            assert!(cache.complete(&lease, true).is_err());
            assert!(cache.complete(&lease, false).is_err());
            assert!(cache.scratch(0, |_| panic!()).is_err());
            assert!(cache.acquire(2, 1, |_| panic!()).is_err());
            assert_eq!(before, cache.stats());
            assert_eq!(before.charged_bytes(), 20);
            assert_eq!(before.unknown_quarantine_bytes, 20);
        }
    }
    #[test]
    fn byte_limit_lru_clean_failure_and_overflow() {
        assert!(
            Cache::<u64>::new(ExpertCacheLimits {
                max_experts: 0,
                max_bytes: 1
            })
            .is_err()
        );
        let mut cache = cache(8, 36);
        for key in [1, 2, 1] {
            let lease = cache.acquire(key, 16, |_| panic!()).unwrap();
            cache.complete(&lease, true).unwrap();
        }
        let mut victims = Vec::new();
        cache.scratch(8, |key| victims.push(*key)).unwrap();
        assert_eq!(victims, [2]);
        assert!(cache.acquire(3, usize::MAX, |_| panic!()).is_err());
        let lease = cache.acquire(3, 12, |_| panic!()).unwrap();
        cache.complete(&lease, false).unwrap();
        assert_eq!(cache.stats().pending_upload_bytes, 0);
        assert_eq!(cache.stats().resident_bytes, 16);
        cache.scratch(0, |_| panic!()).unwrap();
    }
}

#[cfg(test)]
mod w8a16_tests {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/unit/transformer/numeric_w8a16_cache.rs"
    ));
}

#[cfg(test)]
#[path = "../../../tests/unit/transformer/grouped_expert_cache.rs"]
mod grouped_tests;
