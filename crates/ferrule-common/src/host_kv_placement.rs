//! Owner-local metadata custody, not a physical allocator or a broker replacement.
//!
//! Dropping an unresolved owner intentionally retains leases and budget charges.
//! Recovery metadata is then lost: this is a conservative leak, not a reaper.

use std::cell::Cell;
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::host_kv::{
    CommittedKvReadAuthority, CommittedKvSource, HostKvCompletion, HostKvError, HostKvFence,
    HostKvPlacementState, HostKvResult, HostKvStage,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlacementCredits {
    pub pageable_host_bytes: u64,
    pub pinned_host_bytes: u64,
    pub staging_bytes: u64,
    pub staging_slots: u64,
    pub transfers: u64,
    pub consumers: u64,
}

impl PlacementCredits {
    fn domains(self) -> [u64; 6] {
        [
            self.pageable_host_bytes,
            self.pinned_host_bytes,
            self.staging_bytes,
            self.staging_slots,
            self.transfers,
            self.consumers,
        ]
    }
    fn from_domains(v: [u64; 6]) -> Self {
        Self {
            pageable_host_bytes: v[0],
            pinned_host_bytes: v[1],
            staging_bytes: v[2],
            staging_slots: v[3],
            transfers: v[4],
            consumers: v[5],
        }
    }
}

/// Clones share exactly one six-domain budget; callers cannot forge a release.
#[derive(Debug, Clone)]
pub struct PlacementBudget(Rc<BudgetState>);
#[derive(Debug)]
struct BudgetState {
    capacity: PlacementCredits,
    used: Cell<PlacementCredits>,
}
impl PlacementBudget {
    pub fn new(capacity: PlacementCredits) -> Self {
        Self(Rc::new(BudgetState {
            capacity,
            used: Cell::new(PlacementCredits::default()),
        }))
    }
    /// Reserves adapter resources without manufacturing metadata DMA fences.
    pub fn reserve_resources(
        &self,
        credits: PlacementCredits,
    ) -> HostKvResult<PlacementReservation> {
        self.acquire(credits)?;
        Ok(PlacementReservation {
            budget: self.clone(),
            credits,
        })
    }
    pub fn capacity(&self) -> PlacementCredits {
        self.0.capacity
    }
    pub fn in_use(&self) -> PlacementCredits {
        self.0.used.get()
    }
    fn acquire(&self, credits: PlacementCredits) -> HostKvResult<()> {
        let mut next = self.in_use().domains();
        for (i, amount) in credits.domains().into_iter().enumerate() {
            next[i] = next[i].checked_add(amount).ok_or(HostKvError::Overflow)?;
            if next[i] > self.capacity().domains()[i] {
                return Err(HostKvError::BudgetExceeded);
            }
        }
        self.0.used.set(PlacementCredits::from_domains(next));
        Ok(())
    }
    fn release(&self, credits: PlacementCredits) {
        let mut next = self.in_use().domains();
        for (i, amount) in credits.domains().into_iter().enumerate() {
            next[i] = next[i]
                .checked_sub(amount)
                .expect("owner held placement charge");
        }
        self.0.used.set(PlacementCredits::from_domains(next));
    }
}

/// Copying this handle does not duplicate the registered lease or credits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HostKvPlacementHandle {
    owner: u64,
    operation: u64,
}
impl HostKvPlacementHandle {
    pub fn owner(self) -> u64 {
        self.owner
    }
    pub fn operation(self) -> u64 {
        self.operation
    }
}
struct Entry<L> {
    lease: ManuallyDrop<L>,
    credits: PlacementCredits,
    state: HostKvPlacementState,
    pending: Option<HostKvFence>,
    generation: Option<u64>,
}

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

pub struct HostKvPlacementOwner<L> {
    owner: u64,
    next_operation: u64,
    budget: PlacementBudget,
    entries: HashMap<u64, Entry<L>>,
}
impl<L> std::fmt::Debug for HostKvPlacementOwner<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostKvPlacementOwner")
            .field("owner", &self.owner)
            .field("placements", &self.entries.len())
            .field("budget", &self.budget)
            .finish()
    }
}
impl<L> HostKvPlacementOwner<L> {
    pub fn new(budget: PlacementBudget) -> HostKvResult<Self> {
        let owner = NEXT_OWNER
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| {
                if id == 0 {
                    None
                } else {
                    Some(id.checked_add(1).unwrap_or(0))
                }
            })
            .map_err(|_| HostKvError::Overflow)?;
        Ok(Self {
            owner,
            next_operation: 1,
            budget,
            entries: HashMap::new(),
        })
    }
    pub fn owner_identity(&self) -> u64 {
        self.owner
    }
    pub fn budget(&self) -> &PlacementBudget {
        &self.budget
    }

    pub fn reserve<A: CommittedKvReadAuthority<Lease = L>>(
        &mut self,
        authority: &mut A,
        source: &CommittedKvSource,
        credits: PlacementCredits,
    ) -> HostKvResult<HostKvPlacementHandle> {
        // Every entry consumes bounded metadata credits, even with zero byte demand.
        if credits.staging_slots == 0 || credits.transfers == 0 || credits.consumers == 0 {
            return Err(HostKvError::InvalidGeometry);
        }
        if self
            .entries
            .values()
            .any(|e| e.state == HostKvPlacementState::Unknown)
        {
            return Err(HostKvError::Unknown);
        }
        source.validate_geometry()?;
        let next = self
            .next_operation
            .checked_add(1)
            .ok_or(HostKvError::Overflow)?;
        self.budget.acquire(credits)?;
        let lease = match authority.acquire(source) {
            Ok(lease) => lease,
            Err(error) => {
                self.budget.release(credits);
                return Err(error);
            }
        };
        let handle = HostKvPlacementHandle {
            owner: self.owner,
            operation: self.next_operation,
        };
        self.next_operation = next;
        self.entries.insert(
            handle.operation,
            Entry {
                lease: ManuallyDrop::new(lease),
                credits,
                state: HostKvPlacementState::Reserved,
                pending: None,
                generation: None,
            },
        );
        Ok(handle)
    }
    fn entry(&self, handle: HostKvPlacementHandle) -> HostKvResult<&Entry<L>> {
        if handle.owner != self.owner {
            return Err(HostKvError::ForeignOwner);
        }
        self.entries
            .get(&handle.operation)
            .ok_or(HostKvError::UnknownPlacement)
    }
    pub fn state(&self, handle: HostKvPlacementHandle) -> HostKvResult<HostKvPlacementState> {
        Ok(self.entry(handle)?.state)
    }
    pub fn validate<A: CommittedKvReadAuthority<Lease = L>>(
        &self,
        authority: &A,
        handle: HostKvPlacementHandle,
    ) -> HostKvResult<()> {
        authority.validate(&self.entry(handle)?.lease)
    }
    /// Register a fence before invoking the backend. A submitted fence cannot be
    /// overwritten; an ambiguous submission must become Unknown. After an
    /// audited rejection, retry requires a strictly greater generation.
    pub fn submit<A: CommittedKvReadAuthority<Lease = L>>(
        &mut self,
        authority: &A,
        handle: HostKvPlacementHandle,
        fence: HostKvFence,
    ) -> HostKvResult<()> {
        self.validate(authority, handle)?;
        if fence.owner != handle.owner
            || fence.operation != handle.operation
            || fence.generation == 0
        {
            return Err(HostKvError::FenceMismatch);
        }
        let entry = self.entry(handle)?;
        let state = entry.state;
        if entry.generation.is_some_and(|previous| {
            if state == HostKvPlacementState::Reserved {
                // A proven rejection allows retry, not reuse of its identity.
                fence.generation <= previous
            } else {
                fence.generation != previous
            }
        }) {
            return Err(HostKvError::FenceMismatch);
        }
        let next = match (state, fence.stage) {
            (HostKvPlacementState::Reserved, HostKvStage::HostCopy) => {
                HostKvPlacementState::HostCopyPending
            }
            (HostKvPlacementState::HostReady, HostKvStage::Upload) => {
                HostKvPlacementState::UploadPending
            }
            (HostKvPlacementState::StagingReady, HostKvStage::Consumer) => {
                HostKvPlacementState::ConsumerPending
            }
            _ => return Err(HostKvError::InvalidTransition),
        };
        let entry = self
            .entries
            .get_mut(&handle.operation)
            .expect("validated handle");
        entry.generation = Some(fence.generation);
        entry.pending = Some(fence);
        entry.state = next;
        trace_transition("submit", handle, entry);
        Ok(())
    }
    /// Apply a physically verified stage completion through the trusted adapter.
    /// Numeric matching below checks routing/replay, NOT physical completion.
    /// Unknown remains terminal even if later evidence appears complete.
    ///
    /// # Safety
    /// The adapter must have authenticated the exact pending owner, operation,
    /// generation, stage and source lease against its backend-owned operation.
    /// It must hold non-forgeable backend evidence that ALL work in this stage
    /// (including partial submissions and asynchronous consumers) has completed
    /// and no stage references can outlive this transition. Producer events,
    /// acceptance, cancellation, a matching `HostKvCompletion`, or a generic
    /// provider result alone are NOT that evidence. Retain resources/charges
    /// through all downstream stages. Never translate another operation's proof,
    /// replay consumed evidence, or use this API to recover Unknown.
    ///
    /// Safe metadata callers cannot acknowledge even an exactly matching tuple:
    /// ```compile_fail,E0133
    /// use ferrule_common::host_kv::{HostKvCompletion, placement::{HostKvPlacementOwner, HostKvPlacementHandle}};
    /// fn forge<L>(owner: &mut HostKvPlacementOwner<L>, handle: HostKvPlacementHandle, metadata: HostKvCompletion) {
    ///     owner.observe_complete(handle, metadata).unwrap();
    /// }
    /// ```
    #[expect(
        unsafe_code,
        reason = "Audited physical custody boundary; see the method Safety contract"
    )]
    pub unsafe fn observe_complete(
        &mut self,
        handle: HostKvPlacementHandle,
        completion: HostKvCompletion,
    ) -> HostKvResult<()> {
        if self.entry(handle)?.pending != Some(completion.fence) {
            return Err(HostKvError::FenceMismatch);
        }
        let entry = self
            .entries
            .get_mut(&handle.operation)
            .expect("validated handle");
        if entry.state == HostKvPlacementState::Unknown {
            // A matching metadata tuple cannot prove quiescence after an
            // ambiguous submission. This metadata owner has no physical
            // recovery API; the live adapter uses its separate opaque reaper.
            return Err(HostKvError::Unknown);
        }
        entry.state = match completion.fence.stage {
            HostKvStage::HostCopy => HostKvPlacementState::HostReady,
            HostKvStage::Upload => HostKvPlacementState::StagingReady,
            HostKvStage::Consumer => HostKvPlacementState::Complete,
        };
        entry.pending = None;
        Ok(())
    }
    /// Apply a verified initial no-retained-reference rejection. This is a
    /// physical assertion, not an interpretation of an arbitrary error code.
    ///
    /// # Safety
    /// An audited provider integration must prove that the EXACT pending initial
    /// submission never acquired physical references, or has synchronously and
    /// completely retired every partially submitted reference on every stream.
    /// No DMA, callback, consumer, deferred queue item or retained buffer may
    /// subsequently access this placement. Validate the backend operation/source
    /// identity, not just the fence tuple. A generic safe provider's `Err`,
    /// cancellation request, drain attempt, timeout or Unknown is insufficient.
    /// This attestation permits cancellation to release both lease and budget.
    ///
    /// ```compile_fail,E0133
    /// use ferrule_common::host_kv::{HostKvFence, placement::{HostKvPlacementOwner, HostKvPlacementHandle}};
    /// fn forge<L>(owner: &mut HostKvPlacementOwner<L>, handle: HostKvPlacementHandle, metadata: HostKvFence) {
    ///     owner.reject_submission(handle, metadata).unwrap();
    /// }
    /// ```
    #[expect(
        unsafe_code,
        reason = "Audited physical custody boundary; see the method Safety contract"
    )]
    pub unsafe fn reject_submission(
        &mut self,
        handle: HostKvPlacementHandle,
        fence: HostKvFence,
    ) -> HostKvResult<()> {
        let entry = self.entry(handle)?;
        if entry.state != HostKvPlacementState::HostCopyPending || entry.pending != Some(fence) {
            return Err(HostKvError::InvalidTransition);
        }
        let entry = self
            .entries
            .get_mut(&handle.operation)
            .expect("validated handle");
        entry.state = HostKvPlacementState::Reserved;
        entry.pending = None;
        // Keep the existing generation as a retry high-water mark. Delayed
        // observations from this rejected attempt cannot match its successor.
        trace_transition("reject", handle, entry);
        Ok(())
    }
    pub fn mark_unknown(&mut self, handle: HostKvPlacementHandle) -> HostKvResult<()> {
        if self.entry(handle)?.pending.is_none() {
            return Err(HostKvError::InvalidTransition);
        }
        let entry = self
            .entries
            .get_mut(&handle.operation)
            .expect("validated handle");
        if entry.state != HostKvPlacementState::Unknown {
            entry.state = HostKvPlacementState::Unknown;
            trace_transition("unknown", handle, entry);
        }
        Ok(())
    }
    pub fn cancel<A: CommittedKvReadAuthority<Lease = L>>(
        &mut self,
        authority: &mut A,
        handle: HostKvPlacementHandle,
    ) -> HostKvResult<()> {
        if self.entry(handle)?.state != HostKvPlacementState::Reserved {
            return Err(HostKvError::InvalidTransition);
        }
        self.finish(authority, handle)
    }
    pub fn release<A: CommittedKvReadAuthority<Lease = L>>(
        &mut self,
        authority: &mut A,
        handle: HostKvPlacementHandle,
    ) -> HostKvResult<()> {
        if self.entry(handle)?.state != HostKvPlacementState::Complete {
            return Err(HostKvError::InvalidTransition);
        }
        self.finish(authority, handle)
    }
    fn finish<A: CommittedKvReadAuthority<Lease = L>>(
        &mut self,
        authority: &mut A,
        handle: HostKvPlacementHandle,
    ) -> HostKvResult<()> {
        authority.release(&self.entry(handle)?.lease)?;
        let entry = self
            .entries
            .remove(&handle.operation)
            .expect("validated handle");
        self.budget.release(entry.credits);
        trace_transition("retire", handle, &entry);
        drop(ManuallyDrop::into_inner(entry.lease));
        Ok(())
    }
}

// Emit only successful metadata transitions, never per-page/reader validation or
// polling. Counts describe this reservation, not physical allocation or DMA ACKs.
fn trace_transition<L>(transition: &'static str, handle: HostKvPlacementHandle, entry: &Entry<L>) {
    tracing::debug!(
        target: "ferrule::host_kv",
        scope = "metadata",
        transition,
        owner = handle.owner,
        operation = handle.operation,
        stage = ?entry.state,
        pageable_host_bytes = entry.credits.pageable_host_bytes,
        pinned_host_bytes = entry.credits.pinned_host_bytes,
        staging_bytes = entry.credits.staging_bytes,
        staging_slots = entry.credits.staging_slots,
        transfers = entry.credits.transfers,
        consumers = entry.credits.consumers,
        "host KV metadata transition"
    );
}

#[cfg(test)]
#[path = "../tests/unit/host_kv/placement.rs"]
mod tests;

/// Non-cloneable budget custody. Drop intentionally retains the charge.
#[must_use = "release only after physical quiescence"]
pub struct PlacementReservation {
    budget: PlacementBudget,
    credits: PlacementCredits,
}
impl PlacementReservation {
    pub fn release(self) {
        self.budget.release(self.credits);
    }
}
