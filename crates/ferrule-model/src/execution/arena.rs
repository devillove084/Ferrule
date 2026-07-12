use std::collections::HashMap;
use std::hash::Hash;
use std::ops::{Deref, DerefMut};

use ferrule_common::Result;

/// Cumulative lifecycle statistics for a [`PersistentArenaPool`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PersistentArenaPoolStats {
    /// Acquisitions that reused an existing exact-key bucket.
    pub hits: u64,
    /// Acquisitions for which the exact key was absent, including failed builds.
    pub misses: u64,
    /// Successfully built buckets. Failed builds are never installed or counted.
    pub installs: u64,
}

/// Reusable arenas indexed only by exact execution-shape buckets.
///
/// A lease mutably borrows the pool, making the single-active-lease rule explicit
/// in the type system. Dropping the lease returns its arena to the same exact key.
#[derive(Debug)]
pub struct PersistentArenaPool<K, A> {
    available: HashMap<K, A>,
    stats: PersistentArenaPoolStats,
}

impl<K, A> Default for PersistentArenaPool<K, A> {
    fn default() -> Self {
        Self {
            available: HashMap::new(),
            stats: PersistentArenaPoolStats::default(),
        }
    }
}

impl<K, A> PersistentArenaPool<K, A>
where
    K: Eq + Hash,
{
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of exact buckets currently available for leasing.
    pub fn len(&self) -> usize {
        self.available.len()
    }

    pub fn is_empty(&self) -> bool {
        self.available.is_empty()
    }

    pub fn contains(&self, key: &K) -> bool {
        self.available.contains_key(key)
    }

    pub fn stats(&self) -> PersistentArenaPoolStats {
        self.stats
    }

    /// Drops all buckets without resetting cumulative lifecycle statistics.
    pub fn clear(&mut self) {
        self.available.clear();
    }

    /// Acquires an exact bucket, building it only when that exact key is absent.
    ///
    /// A failed build is returned directly and never publishes a pool entry.
    pub fn acquire<F>(&mut self, key: K, build: F) -> Result<ArenaLease<'_, K, A>>
    where
        F: FnOnce() -> Result<A>,
    {
        let (key, arena, reused) = match self.available.remove_entry(&key) {
            Some((key, arena)) => {
                self.stats.hits = self.stats.hits.saturating_add(1);
                (key, arena, true)
            }
            None => {
                self.stats.misses = self.stats.misses.saturating_add(1);
                let arena = build()?;
                self.stats.installs = self.stats.installs.saturating_add(1);
                (key, arena, false)
            }
        };
        Ok(ArenaLease {
            available: &mut self.available,
            key: Some(key),
            arena: Some(arena),
            reused,
        })
    }

    /// Alias emphasizing that allocation/build may fail.
    pub fn try_lease<F>(&mut self, key: K, build: F) -> Result<ArenaLease<'_, K, A>>
    where
        F: FnOnce() -> Result<A>,
    {
        self.acquire(key, build)
    }
}

/// Exclusive lease of one persistent arena bucket.
#[derive(Debug)]
pub struct ArenaLease<'a, K, A>
where
    K: Eq + Hash,
{
    available: &'a mut HashMap<K, A>,
    key: Option<K>,
    arena: Option<A>,
    reused: bool,
}

impl<K, A> ArenaLease<'_, K, A>
where
    K: Eq + Hash,
{
    pub fn key(&self) -> &K {
        self.key.as_ref().expect("arena lease key is present")
    }

    /// Whether this lease reused an existing exact-key bucket.
    pub const fn reused(&self) -> bool {
        self.reused
    }

    pub fn get(&self) -> &A {
        self.arena.as_ref().expect("arena lease value is present")
    }

    pub fn get_mut(&mut self) -> &mut A {
        self.arena.as_mut().expect("arena lease value is present")
    }
}

impl<K, A> Deref for ArenaLease<'_, K, A>
where
    K: Eq + Hash,
{
    type Target = A;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<K, A> DerefMut for ArenaLease<'_, K, A>
where
    K: Eq + Hash,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

impl<K, A> Drop for ArenaLease<'_, K, A>
where
    K: Eq + Hash,
{
    fn drop(&mut self) {
        let key = self.key.take().expect("arena lease key is present on drop");
        let arena = self
            .arena
            .take()
            .expect("arena lease value is present on drop");
        let replaced = self.available.insert(key, arena);
        debug_assert!(
            replaced.is_none(),
            "leased arena bucket was already present"
        );
    }
}

#[cfg(test)]
mod tests {
    use ferrule_common::Error;

    use super::*;

    #[test]
    fn drop_returns_mutated_arena_to_exact_bucket() {
        let mut pool = PersistentArenaPool::<u32, Vec<u32>>::new();
        {
            let mut lease = pool.acquire(7, || Ok(vec![1])).unwrap();
            lease.push(2);
            assert_eq!(lease.key(), &7);
        }
        assert_eq!(pool.len(), 1);
        let lease = pool
            .acquire(7, || -> Result<Vec<u32>> {
                panic!("exact bucket should be reused")
            })
            .unwrap();
        assert_eq!(&*lease, &[1, 2]);
    }

    #[test]
    fn different_keys_build_different_exact_buckets() {
        let mut pool = PersistentArenaPool::<u32, u32>::new();
        drop(pool.acquire(1, || Ok(10)).unwrap());
        drop(pool.acquire(2, || Ok(20)).unwrap());
        assert_eq!(pool.len(), 2);
        assert!(pool.contains(&1));
        assert!(pool.contains(&2));
    }

    #[test]
    fn failed_build_does_not_publish_bucket() {
        let mut pool = PersistentArenaPool::<u32, u32>::new();
        {
            let result = pool.acquire(3, || Err(Error::Execution("injected build failure".into())));
            assert!(matches!(result, Err(Error::Execution(_))));
        }
        assert!(pool.is_empty());
        assert!(!pool.contains(&3));
        assert_eq!(
            pool.stats(),
            PersistentArenaPoolStats {
                hits: 0,
                misses: 1,
                installs: 0,
            }
        );
    }

    #[test]
    fn exact_bucket_a_b_a_reports_reuse_and_stats() {
        let mut pool = PersistentArenaPool::<char, u32>::new();
        let first_a = pool.acquire('A', || Ok(10)).unwrap();
        assert!(!first_a.reused());
        drop(first_a);
        let first_b = pool.acquire('B', || Ok(20)).unwrap();
        assert!(!first_b.reused());
        drop(first_b);
        let second_a = pool.acquire('A', || panic!("A must be reused")).unwrap();
        assert!(second_a.reused());
        assert_eq!(*second_a, 10);
        drop(second_a);
        assert_eq!(
            pool.stats(),
            PersistentArenaPoolStats {
                hits: 1,
                misses: 2,
                installs: 2,
            }
        );
    }

    #[test]
    fn failed_new_bucket_keeps_old_bucket_reusable() {
        let mut pool = PersistentArenaPool::<char, u32>::new();
        drop(pool.acquire('A', || Ok(10)).unwrap());
        {
            let failed_b = pool.acquire('B', || Err(Error::Execution("B failed".into())));
            assert!(failed_b.is_err());
        }
        assert!(pool.contains(&'A'));
        assert!(!pool.contains(&'B'));

        let second_a = pool
            .acquire('A', || panic!("old A bucket must survive"))
            .unwrap();
        assert!(second_a.reused());
        assert_eq!(*second_a, 10);
        drop(second_a);
        assert_eq!(
            pool.stats(),
            PersistentArenaPoolStats {
                hits: 1,
                misses: 2,
                installs: 1,
            }
        );
    }

    #[test]
    fn existing_bucket_never_calls_builder() {
        let mut pool = PersistentArenaPool::<u32, u32>::new();
        drop(pool.acquire(5, || Ok(41)).unwrap());
        let lease = pool
            .acquire(5, || -> Result<u32> {
                panic!("builder called for available bucket")
            })
            .unwrap();
        assert_eq!(*lease, 41);
    }
}
