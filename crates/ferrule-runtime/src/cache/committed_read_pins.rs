//! Private reader-registration index, not a logical or physical KV authority.
//!
//! Source descriptors here are frozen registration metadata. Only KvPageManager
//! can validate their frontier, incarnation, refcounts, sharing and transactions.

use std::collections::HashMap;

use ferrule_common::execution::{KvPageId, StateSlot};
use ferrule_common::host_kv::{CommittedKvSource, HostKvError, HostKvResult};

/// Drop does not release the registered pin; physical quiescence requires an
/// explicit release. This token deliberately has no Clone implementation.
#[derive(Debug)]
#[must_use = "release the reader explicitly after physical quiescence"]
pub struct CommittedKvReadLease {
    owner: u64,
    id: u64,
    source: CommittedKvSource,
}
impl CommittedKvReadLease {
    pub fn source(&self) -> &CommittedKvSource {
        &self.source
    }
}

pub(super) struct CommittedReadPins {
    next_id: u64,
    readers: HashMap<u64, CommittedKvSource>,
}
impl CommittedReadPins {
    pub(super) fn new() -> Self {
        Self {
            next_id: 1,
            readers: HashMap::new(),
        }
    }

    /// Caller must validate the source against the page manager before registering.
    pub(super) fn register(
        &mut self,
        owner: u64,
        source: &CommittedKvSource,
    ) -> HostKvResult<CommittedKvReadLease> {
        let next = self.next_id.checked_add(1).ok_or(HostKvError::Overflow)?;
        let id = self.next_id;
        self.next_id = next;
        self.readers.insert(id, source.clone());
        Ok(CommittedKvReadLease {
            owner,
            id,
            source: source.clone(),
        })
    }

    /// Check registration identity only; this is not source validation.
    pub(super) fn validate(&self, owner: u64, lease: &CommittedKvReadLease) -> HostKvResult<()> {
        if lease.owner != owner {
            return Err(HostKvError::ForeignOwner);
        }
        if self.readers.get(&lease.id) != Some(&lease.source) {
            return Err(HostKvError::UnknownLease);
        }
        Ok(())
    }

    /// Caller completes source preflight first. Failure leaves the index intact.
    pub(super) fn release(&mut self, owner: u64, lease: &CommittedKvReadLease) -> HostKvResult<()> {
        self.validate(owner, lease)?;
        self.readers.remove(&lease.id);
        Ok(())
    }

    pub(super) fn check_pages(&self, pages: &[KvPageId]) -> HostKvResult<()> {
        if self
            .readers
            .values()
            .any(|s| s.pages.iter().any(|p| pages.contains(p)))
        {
            return Err(HostKvError::ReaderActive);
        }
        Ok(())
    }

    /// The manager also checks the slot's current pages to protect aliases whose
    /// slot differs from the registered reader. Do not replace that page check.
    pub(super) fn check_slot(&self, slot: StateSlot) -> HostKvResult<()> {
        if self.readers.values().any(|s| s.state_slot == slot) {
            return Err(HostKvError::ReaderActive);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cache/committed_read_pins.rs"]
mod tests;
