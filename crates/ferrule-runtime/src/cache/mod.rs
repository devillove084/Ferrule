//! Authoritative logical KV page transactions.

use std::sync::atomic::{AtomicU64, Ordering};

pub mod page_manager;
pub mod radix;

pub use page_manager::{
    AbortKvReservationsError, BlockTable as PageBlockTable, ConfirmKvRetirementError,
    KvPageManager, KvPageManagerStats, KvPrefixSnapshot, KvReservation, KvReservationBindings,
    KvReservationCommit, KvReservationId, KvRetirement, PreemptedKvState, PrepareKvCommitError,
    PreparedKvCommit, PreparedKvSequenceFork, PreparedKvSnapshotFork,
};
pub use radix::{
    PrefixCacheEntryId, PrefixCacheError, PrefixCacheEvictionCandidate, PrefixCacheInsertError,
    PrefixCacheInsertOutcome, PrefixCacheLease, PrefixCacheNamespace, PrefixCacheUnpinError,
    PrefixLookupLimit, PrefixToken, RadixPrefixCache, RemovedPrefixEntry,
};

/// Allocate each non-zero identity at most once, then remain permanently exhausted.
///
/// Zero is an absorbing sentinel. The terminal `u64::MAX` identity is published
/// exactly once by the successful compare-exchange that installs that sentinel.
#[track_caller]
fn take_permanent_identity(counter: &AtomicU64, exhausted_message: &'static str) -> u64 {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        if current == 0 {
            panic!("{exhausted_message}");
        }
        let next = current.checked_add(1).unwrap_or(0);
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return current,
            Err(observed) => current = observed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permanent_identity_allocator_never_wraps_or_recovers() {
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(
            take_permanent_identity(&counter, "test identity exhausted"),
            u64::MAX - 1
        );
        assert_eq!(
            take_permanent_identity(&counter, "test identity exhausted"),
            u64::MAX
        );
        assert_eq!(counter.load(Ordering::Relaxed), 0);

        for _ in 0..2 {
            assert!(
                std::panic::catch_unwind(|| {
                    take_permanent_identity(&counter, "test identity exhausted")
                })
                .is_err()
            );
            assert_eq!(counter.load(Ordering::Relaxed), 0);
        }
    }
}
