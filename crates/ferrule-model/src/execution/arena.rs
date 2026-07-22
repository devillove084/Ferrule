use std::collections::{HashMap, hash_map::Entry};
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
/// An owned checkout instead permits independent pool access until explicitly checked in.
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

    fn take_or_build<F>(&mut self, key: K, build: F) -> Result<(K, A, bool)>
    where
        F: FnOnce() -> Result<A>,
    {
        match self.available.remove_entry(&key) {
            Some((key, arena)) => {
                self.stats.hits = self.stats.hits.saturating_add(1);
                Ok((key, arena, true))
            }
            None => {
                self.stats.misses = self.stats.misses.saturating_add(1);
                let arena = build()?;
                self.stats.installs = self.stats.installs.saturating_add(1);
                Ok((key, arena, false))
            }
        }
    }

    /// Acquires an exact bucket, building it only when that exact key is absent.
    ///
    /// A failed build is returned directly and never publishes a pool entry.
    pub fn acquire<F>(&mut self, key: K, build: F) -> Result<ArenaLease<'_, K, A>>
    where
        F: FnOnce() -> Result<A>,
    {
        let (key, arena, reused) = self.take_or_build(key, build)?;
        Ok(ArenaLease {
            available: &mut self.available,
            key: Some(key),
            arena: Some(arena),
            reused,
        })
    }

    /// Checks out an owned exact bucket without borrowing the pool after return.
    ///
    /// A failed build is returned directly and never publishes a pool entry.
    pub fn checkout<F>(&mut self, key: K, build: F) -> Result<OwnedArenaCheckout<K, A>>
    where
        F: FnOnce() -> Result<A>,
    {
        let (key, arena, reused) = self.take_or_build(key, build)?;
        Ok(OwnedArenaCheckout { key, arena, reused })
    }

    /// Returns an owned checkout to its exact-key bucket.
    ///
    /// If that key is already available, the available arena is preserved and an error
    /// is returned instead of silently replacing it.
    pub fn checkin(&mut self, checkout: OwnedArenaCheckout<K, A>) -> Result<()> {
        let OwnedArenaCheckout {
            key,
            arena,
            reused: _,
        } = checkout;
        match self.available.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(arena);
                Ok(())
            }
            Entry::Occupied(_) => Err(ferrule_common::Error::Execution(
                "arena checkout exact-key bucket is already available".into(),
            )),
        }
    }

    /// Alias emphasizing that allocation/build may fail.
    pub fn try_lease<F>(&mut self, key: K, build: F) -> Result<ArenaLease<'_, K, A>>
    where
        F: FnOnce() -> Result<A>,
    {
        self.acquire(key, build)
    }
}

/// Owned checkout of one persistent arena bucket.
///
/// Unlike [`ArenaLease`], this value does not borrow its originating pool. It must be
/// passed to [`PersistentArenaPool::checkin`] to make its arena available for reuse.
#[derive(Debug)]
pub struct OwnedArenaCheckout<K, A> {
    key: K,
    arena: A,
    reused: bool,
}

impl<K, A> OwnedArenaCheckout<K, A> {
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Whether this checkout reused an existing exact-key bucket.
    pub const fn reused(&self) -> bool {
        self.reused
    }

    pub fn get(&self) -> &A {
        &self.arena
    }

    pub fn get_mut(&mut self) -> &mut A {
        &mut self.arena
    }
}

impl<K, A> Deref for OwnedArenaCheckout<K, A> {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<K, A> DerefMut for OwnedArenaCheckout<K, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
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

    #[test]
    fn owned_checkout_does_not_borrow_pool() {
        let checkout = pool_checkout_for_independent_access();
        assert_eq!(checkout.key(), &'A');
        assert_eq!(*checkout, 10);
    }

    fn pool_checkout_for_independent_access() -> OwnedArenaCheckout<char, u32> {
        let mut pool = PersistentArenaPool::<char, u32>::new();
        let checkout_a = pool.checkout('A', || Ok(10)).unwrap();

        assert!(pool.is_empty());
        let checkout_b = pool.checkout('B', || Ok(20)).unwrap();
        pool.checkin(checkout_b).unwrap();
        assert!(pool.contains(&'B'));
        assert_eq!(
            pool.stats(),
            PersistentArenaPoolStats {
                hits: 0,
                misses: 2,
                installs: 2,
            }
        );

        checkout_a
    }

    #[test]
    fn owned_checkout_mutation_survives_checkin_and_recheckout() {
        let mut pool = PersistentArenaPool::<u32, Vec<u32>>::new();
        let mut checkout = pool.checkout(7, || Ok(vec![1])).unwrap();
        assert_eq!(checkout.key(), &7);
        assert!(!checkout.reused());
        assert_eq!(checkout.get(), &[1]);
        checkout.get_mut().push(2);
        checkout.push(3);
        pool.checkin(checkout).unwrap();

        let checkout = pool
            .checkout(7, || -> Result<Vec<u32>> {
                panic!("checked-in exact bucket should be reused")
            })
            .unwrap();
        assert!(checkout.reused());
        assert_eq!(&*checkout, &[1, 2, 3]);
        assert_eq!(
            pool.stats(),
            PersistentArenaPoolStats {
                hits: 1,
                misses: 1,
                installs: 1,
            }
        );
    }

    #[test]
    fn owned_checkouts_for_different_exact_keys_coexist() {
        let mut pool = PersistentArenaPool::<char, u32>::new();
        let checkout_a = pool.checkout('A', || Ok(10)).unwrap();
        let checkout_b = pool.checkout('B', || Ok(20)).unwrap();

        pool.checkin(checkout_a).unwrap();
        pool.checkin(checkout_b).unwrap();
        assert_eq!(pool.len(), 2);
        assert!(pool.contains(&'A'));
        assert!(pool.contains(&'B'));
    }

    #[test]
    fn duplicate_key_checkin_is_rejected_without_replacement() {
        let mut pool = PersistentArenaPool::<u32, u32>::new();
        let first = pool.checkout(5, || Ok(10)).unwrap();
        let second = pool.checkout(5, || Ok(20)).unwrap();
        pool.checkin(second).unwrap();

        let result = pool.checkin(first);
        assert!(matches!(result, Err(Error::Execution(_))));

        let checkout = pool
            .checkout(5, || -> Result<u32> {
                panic!("duplicate checkin must preserve the available bucket")
            })
            .unwrap();
        assert_eq!(*checkout, 20);
    }
}
