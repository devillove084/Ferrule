//! Model- and backend-neutral memory policy vocabulary.

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Physical relationship between host and accelerator memory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryTopology {
    /// Host and accelerator allocations occupy independent physical memory.
    #[default]
    Discrete,
    /// CPU and accelerator allocations share one coherent physical memory pool.
    CoherentUnified,
}

impl fmt::Display for MemoryTopology {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Discrete => "discrete",
            Self::CoherentUnified => "coherent-unified",
        })
    }
}

impl FromStr for MemoryTopology {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "discrete" => Ok(Self::Discrete),
            "coherent-unified" | "coherent_unified" | "unified" => Ok(Self::CoherentUnified),
            _ => Err("expected 'discrete' or 'coherent-unified'"),
        }
    }
}

/// Stable classification for independently budgeted memory pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPoolKind {
    ResidentDevice,
    HostStaged,
    PinnedHost,
    TransientUpload,
    FilePageCache,
}

/// Retention limits for one owner-managed memory pool.
///
/// `max_entries == 0` disables admission. `max_bytes == u64::MAX` means the
/// pool is entry-limited only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPoolLimits {
    pub max_entries: usize,
    pub max_bytes: u64,
}

impl MemoryPoolLimits {
    pub const fn new(max_entries: usize, max_bytes: u64) -> Self {
        Self {
            max_entries,
            max_bytes,
        }
    }

    pub const fn entries_only(max_entries: usize) -> Self {
        Self::new(max_entries, u64::MAX)
    }

    pub const fn disabled() -> Self {
        Self::new(0, 0)
    }

    pub const fn admission_enabled(self) -> bool {
        self.max_entries != 0 && self.max_bytes != 0
    }
}

/// Point-in-time accounting for one owner-managed memory pool.
///
/// Bytes describe retained payload bytes, not allocator overhead or process
/// RSS. Evicted shared allocations may remain alive while another owner holds
/// a reference.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPoolStats {
    pub limits: MemoryPoolLimits,
    pub entries_used: usize,
    pub bytes_used: u64,
    pub peak_bytes_used: u64,
    pub hits: u64,
    pub misses: u64,
    pub admissions: u64,
    pub evictions: u64,
    pub rejections: u64,
}

#[derive(Debug)]
struct OwnerMemoryLruEntry<K, V> {
    value: V,
    bytes: u64,
    previous: Option<K>,
    next: Option<K>,
}

/// Synchronization-free, owner-thread LRU with simultaneous entry and byte limits.
///
/// The owner supplies retained payload bytes on insertion. Lookups, admission,
/// accounting, and each individual eviction are O(1) on average.
#[derive(Debug)]
pub struct OwnerMemoryLru<K, V> {
    entries: HashMap<K, OwnerMemoryLruEntry<K, V>>,
    least_recently_used: Option<K>,
    most_recently_used: Option<K>,
    limits: MemoryPoolLimits,
    resident_bytes: u64,
    peak_resident_bytes: u64,
    hits: u64,
    misses: u64,
    admissions: u64,
    evictions: u64,
    rejections: u64,
}

impl<K, V> OwnerMemoryLru<K, V>
where
    K: Copy + Eq + Hash,
{
    pub fn new(limits: MemoryPoolLimits) -> Self {
        Self {
            entries: HashMap::new(),
            least_recently_used: None,
            most_recently_used: None,
            limits,
            resident_bytes: 0,
            peak_resident_bytes: 0,
            hits: 0,
            misses: 0,
            admissions: 0,
            evictions: 0,
            rejections: 0,
        }
    }

    pub fn get_cloned(&mut self, key: K) -> Option<V>
    where
        V: Clone,
    {
        let value = self.entries.get(&key).map(|entry| entry.value.clone());
        if value.is_some() {
            self.hits = self.hits.saturating_add(1);
            self.touch(key);
        } else {
            self.misses = self.misses.saturating_add(1);
        }
        value
    }

    pub fn contains(&self, key: K) -> bool {
        self.entries.contains_key(&key)
    }

    pub fn keys(&self) -> impl Iterator<Item = K> + '_ {
        self.entries.keys().copied()
    }

    /// Admit a value and return whether it was retained.
    ///
    /// An inadmissible replacement leaves the old value untouched. An
    /// admissible replacement is not counted as an eviction.
    pub fn insert(&mut self, key: K, value: V, bytes: u64) -> bool {
        if !self.limits.admission_enabled() || bytes > self.limits.max_bytes {
            self.rejections = self.rejections.saturating_add(1);
            return false;
        }

        self.remove(key, false);
        while self.entries.len() >= self.limits.max_entries
            || bytes > self.limits.max_bytes.saturating_sub(self.resident_bytes)
        {
            if !self.evict_least_recently_used() {
                self.rejections = self.rejections.saturating_add(1);
                return false;
            }
        }

        let previous = self.most_recently_used;
        self.entries.insert(
            key,
            OwnerMemoryLruEntry {
                value,
                bytes,
                previous,
                next: None,
            },
        );
        if let Some(previous) = previous {
            self.entries
                .get_mut(&previous)
                .expect("LRU tail must name a resident entry")
                .next = Some(key);
        } else {
            self.least_recently_used = Some(key);
        }
        self.most_recently_used = Some(key);
        self.resident_bytes = self.resident_bytes.saturating_add(bytes);
        self.peak_resident_bytes = self.peak_resident_bytes.max(self.resident_bytes);
        self.admissions = self.admissions.saturating_add(1);
        true
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub const fn limits(&self) -> MemoryPoolLimits {
        self.limits
    }

    pub fn stats(&self) -> MemoryPoolStats {
        MemoryPoolStats {
            limits: self.limits,
            entries_used: self.entries.len(),
            bytes_used: self.resident_bytes,
            peak_bytes_used: self.peak_resident_bytes,
            hits: self.hits,
            misses: self.misses,
            admissions: self.admissions,
            evictions: self.evictions,
            rejections: self.rejections,
        }
    }

    fn touch(&mut self, key: K) {
        if self.most_recently_used == Some(key) {
            return;
        }
        let (previous, next) = {
            let entry = self
                .entries
                .get(&key)
                .expect("touched entry must be resident");
            (entry.previous, entry.next)
        };
        if let Some(previous) = previous {
            self.entries
                .get_mut(&previous)
                .expect("LRU previous entry must be resident")
                .next = next;
        } else {
            self.least_recently_used = next;
        }
        if let Some(next) = next {
            self.entries
                .get_mut(&next)
                .expect("LRU next entry must be resident")
                .previous = previous;
        }

        let old_tail = self.most_recently_used;
        if let Some(old_tail) = old_tail {
            self.entries
                .get_mut(&old_tail)
                .expect("LRU tail must be resident")
                .next = Some(key);
        } else {
            self.least_recently_used = Some(key);
        }
        let entry = self
            .entries
            .get_mut(&key)
            .expect("touched entry must remain resident");
        entry.previous = old_tail;
        entry.next = None;
        self.most_recently_used = Some(key);
    }

    fn evict_least_recently_used(&mut self) -> bool {
        let Some(key) = self.least_recently_used else {
            return false;
        };
        self.remove(key, true);
        true
    }

    fn remove(&mut self, key: K, eviction: bool) -> Option<V> {
        let entry = self.entries.remove(&key)?;
        if let Some(previous) = entry.previous {
            self.entries
                .get_mut(&previous)
                .expect("LRU previous entry must be resident")
                .next = entry.next;
        } else {
            self.least_recently_used = entry.next;
        }
        if let Some(next) = entry.next {
            self.entries
                .get_mut(&next)
                .expect("LRU next entry must be resident")
                .previous = entry.previous;
        } else {
            self.most_recently_used = entry.previous;
        }
        self.resident_bytes = self.resident_bytes.saturating_sub(entry.bytes);
        if eviction {
            self.evictions = self.evictions.saturating_add(1);
        }
        Some(entry.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_parsing_is_stable() {
        assert_eq!("discrete".parse(), Ok(MemoryTopology::Discrete));
        assert_eq!(
            "coherent-unified".parse(),
            Ok(MemoryTopology::CoherentUnified)
        );
        assert_eq!("unified".parse(), Ok(MemoryTopology::CoherentUnified));
        assert!("other".parse::<MemoryTopology>().is_err());
    }

    #[test]
    fn pool_limit_constructors_have_explicit_disable_semantics() {
        assert_eq!(
            MemoryPoolLimits::entries_only(8),
            MemoryPoolLimits::new(8, u64::MAX)
        );
        assert!(MemoryPoolLimits::entries_only(8).admission_enabled());
        assert!(!MemoryPoolLimits::disabled().admission_enabled());
    }

    #[test]
    fn owner_lru_enforces_entry_and_byte_limits_and_promotes_hits() {
        let mut cache = OwnerMemoryLru::new(MemoryPoolLimits::new(2, 10));
        assert!(cache.insert(1, "one", 4));
        assert!(cache.insert(2, "two", 4));
        assert_eq!(cache.get_cloned(1), Some("one"));
        assert!(cache.insert(3, "three", 6));

        assert!(cache.contains(1));
        assert!(!cache.contains(2));
        assert!(cache.contains(3));
        assert_eq!(
            cache.stats(),
            MemoryPoolStats {
                limits: MemoryPoolLimits::new(2, 10),
                entries_used: 2,
                bytes_used: 10,
                peak_bytes_used: 10,
                hits: 1,
                admissions: 3,
                evictions: 1,
                ..MemoryPoolStats::default()
            }
        );
    }

    #[test]
    fn owner_lru_rejects_oversized_replacement_without_disturbing_resident() {
        let mut cache = OwnerMemoryLru::new(MemoryPoolLimits::new(2, 8));
        assert!(cache.insert(1, "resident", 6));
        assert!(!cache.insert(1, "oversized", 9));
        assert_eq!(cache.get_cloned(1), Some("resident"));
        assert_eq!(cache.stats().bytes_used, 6);
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(cache.stats().rejections, 1);
    }

    #[test]
    fn owner_lru_disabled_pool_rejects_without_retention() {
        let mut cache = OwnerMemoryLru::new(MemoryPoolLimits::disabled());
        assert!(!cache.insert(1, "value", 1));
        assert!(cache.is_empty());
        assert_eq!(cache.stats().rejections, 1);
    }
}
