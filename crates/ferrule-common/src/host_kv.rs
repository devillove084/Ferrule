//! Metadata contracts for committed, exclusive host KV reads.
//!
//! These descriptors do not prove physical residency or DMA completion. A backend
//! adapter must retain its allocations and obtain real producer/consumer fences.

use std::ops::Range;

use crate::execution::{
    ExecutionTransactionId, KvLayoutSchemaIdentity, KvPageId, KvReservationView, StateSlot,
};

pub use crate::host_kv_placement as placement;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostKvError {
    ForeignOwner,
    StaleSource,
    InvalidRange,
    InvalidGeometry,
    SharedSource,
    TransactionActive,
    ReaderActive,
    UnknownLease,
    UnknownPlacement,
    InvalidTransition,
    FenceMismatch,
    BudgetExceeded,
    Overflow,
    ProviderUnsupported,
    Unknown,
}

impl std::fmt::Display for HostKvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "host KV: {self:?}")
    }
}
impl std::error::Error for HostKvError {}
pub type HostKvResult<T> = Result<T, HostKvError>;

/// Untrusted descriptor, revalidated against the original logical authority on
/// every acquisition. Pages describe the entire committed frontier; `positions`
/// selects a nonempty subrange. This is not an ownership token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedKvSource {
    pub owner: u64,
    pub state_slot: StateSlot,
    pub generation: u64,
    pub incarnation: u64,
    pub schema: KvLayoutSchemaIdentity,
    pub committed_tokens: usize,
    pub positions: Range<usize>,
    pub pages: Vec<KvPageId>,
}

impl CommittedKvSource {
    pub fn validate_geometry(&self) -> HostKvResult<()> {
        if self.positions.start >= self.positions.end
            || self.positions.end > self.committed_tokens
            || self.committed_tokens > self.schema.max_sequence_len()
        {
            return Err(HostKvError::InvalidRange);
        }
        let page_size = self.schema.page_size();
        if page_size == 0
            || self.schema.planes().is_empty()
            || self.pages.len() != self.committed_tokens.div_ceil(page_size)
            || self.schema.planes().iter().any(|p| {
                p.elements_per_token == 0
                    || p.layer_count == 0
                    || p.checked_page_bytes(page_size).is_none()
            })
        {
            return Err(HostKvError::InvalidGeometry);
        }
        Ok(())
    }
}

/// Logical witness for a read spanning a committed prefix and one existing
/// append reservation. This is metadata only: it owns neither reservation nor
/// physical storage. The runtime must retain the reservation token, while the
/// backend must derive its own capability from the matching transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostKvAppendReadWitness {
    transaction: ExecutionTransactionId,
    reservation_id: u64,
    source: CommittedKvSource,
    reservation: KvReservationView,
    pages: Vec<KvPageId>,
}
impl HostKvAppendReadWitness {
    pub fn new(
        transaction: ExecutionTransactionId,
        reservation_id: u64,
        source: CommittedKvSource,
        reservation: KvReservationView,
    ) -> HostKvResult<Self> {
        source.validate_geometry()?;
        if source.owner == 0
            || source.incarnation == 0
            || reservation_id == 0
            || source.positions.start != 0
            || reservation
                .positions
                .end
                .checked_sub(reservation.positions.start)
                != Some(1)
            || reservation.positions.end > source.schema.max_sequence_len()
            || reservation.positions.start != source.positions.end
            || reservation.positions.start != source.committed_tokens
            || reservation.positions.start >= reservation.positions.end
            || reservation.state_slot != source.state_slot
            || reservation.generation != source.generation
        {
            return Err(HostKvError::InvalidTransition);
        }
        let mut pages = source.pages.clone();
        if let Some(cow) = reservation.cow_replacement {
            if cow.logical_page.checked_add(1) != Some(pages.len())
                || source
                    .committed_tokens
                    .is_multiple_of(source.schema.page_size())
                || pages.get(cow.logical_page) != Some(&cow.source)
                || pages.contains(&cow.replacement)
            {
                return Err(HostKvError::StaleSource);
            }
            pages[cow.logical_page] = cow.replacement;
        }
        pages.extend(reservation.newly_allocated.iter().copied());
        let expected = reservation
            .positions
            .end
            .div_ceil(source.schema.page_size());
        if pages.len() != expected
            || pages
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != pages.len()
            || source
                .pages
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != source.pages.len()
            || reservation
                .newly_allocated
                .iter()
                .any(|p| source.pages.contains(p))
        {
            return Err(HostKvError::InvalidGeometry);
        }
        Ok(Self {
            transaction,
            reservation_id,
            source,
            reservation,
            pages,
        })
    }
    pub const fn transaction(&self) -> ExecutionTransactionId {
        self.transaction
    }
    pub const fn reservation_id(&self) -> u64 {
        self.reservation_id
    }
    pub fn source(&self) -> &CommittedKvSource {
        &self.source
    }
    pub fn reservation(&self) -> &KvReservationView {
        &self.reservation
    }
    pub fn pages(&self) -> &[KvPageId] {
        &self.pages
    }
    pub fn positions(&self) -> Range<usize> {
        self.source.positions.start..self.reservation.positions.end
    }
    /// Exact, disjoint coverage for partial attention of the one appended query.
    /// Both operands must be unnormalized (max, denominator, numerator) states
    /// for the same Q, head geometry and mask. This metadata check is not a DMA
    /// receipt or a proof of the numerical operands' origin.
    pub fn validate_partial_partition(
        &self,
        query_position: usize,
        prefix: Range<usize>,
        tail: Range<usize>,
    ) -> HostKvResult<()> {
        if query_position != self.source.committed_tokens
            || prefix != self.source.positions
            || tail != self.reservation.positions
        {
            return Err(HostKvError::InvalidRange);
        }
        Ok(())
    }
}

/// Release failures must leave the lease registered and retryable. Dropping a
/// lease is never an ACK. Implementations must reject writes, sharing and logical
/// retirement while a lease is live, including physical-page aliases.
pub trait CommittedKvReadAuthority {
    type Lease;
    fn acquire(&mut self, source: &CommittedKvSource) -> HostKvResult<Self::Lease>;
    fn validate(&self, lease: &Self::Lease) -> HostKvResult<()>;
    fn release(&mut self, lease: &Self::Lease) -> HostKvResult<()>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HostKvStage {
    HostCopy,
    Upload,
    Consumer,
}

/// Exact adapter-local fence identity. `generation` qualifies staging reuse,
/// independently of the logical source and operation incarnation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostKvFence {
    pub owner: u64,
    pub operation: u64,
    pub generation: u64,
    pub stage: HostKvStage,
    pub stream: u64,
    pub event: u64,
}

/// Untrusted completion **metadata**, NOT physical evidence or a receipt.
/// Fields intentionally remain inspectable/copyable for reporting and transport.
/// Matching these numbers cannot discharge custody. Only an audited adapter may
/// attest physical completion at the unsafe placement-owner boundary, after
/// obtaining the exact backend's opaque DMA/consumer proof. There is no safe
/// conversion from this descriptor into completion authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostKvCompletion {
    pub fence: HostKvFence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostKvPlacementState {
    Reserved,
    HostCopyPending,
    HostReady,
    UploadPending,
    StagingReady,
    ConsumerPending,
    Complete,
    Unknown,
}

/// Memory-source command; deliberately not a checkpoint read plan. Credits are
/// allocated bytes (including alignment/allocator overhead), not payload bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostKvPlan {
    pub source: CommittedKvSource,
    pub credits: placement::PlacementCredits,
    pub fence: HostKvFence,
}
