//! Metadata contracts for committed, exclusive host KV reads.
//!
//! These descriptors do not prove physical residency or DMA completion. A backend
//! adapter must retain its allocations and obtain real producer/consumer fences.

use std::ops::Range;
use std::sync::Arc;

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

/// A witness-issued identity for one logical host-KV read.
///
/// This capability contains only transaction/request metadata. It has no page
/// address, placement handle, fence, allocation, or completion authority. Its
/// fields are private so safe callers can obtain one only from a witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostKvReadCapability {
    transaction: ExecutionTransactionId,
    reservation_id: u64,
    owner: u64,
    state_slot: StateSlot,
    generation: u64,
    incarnation: u64,
    schema: KvLayoutSchemaIdentity,
    committed_tokens: usize,
    source_positions: Range<usize>,
    reservation_positions: Range<usize>,
    execution_state_slot: StateSlot,
    execution_generation: u64,
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
        let witness = Self {
            transaction,
            reservation_id,
            source,
            reservation,
            pages,
        };
        // Keep the same page/source/destination geometry invariant at the
        // existing witness admission boundary without allocating a plan.
        witness.validate_read_geometry(witness.positions())?;
        Ok(witness)
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
    /// Issue the packed logical read description for the complete witness range.
    /// This contains no physical placement or completion information.
    pub fn read_plan(&self) -> HostKvResult<KvReadPlan> {
        self.read_plan_for(self.positions())
    }
    /// Issue a packed logical read description for a nonempty subrange of the
    /// witness. Each emitted segment is page-bounded and destination-packed.
    pub fn read_plan_for(&self, logical_tokens: Range<usize>) -> HostKvResult<KvReadPlan> {
        KvReadPlan::from_witness(self, logical_tokens)
    }
    /// Validate a plan against this exact transaction, reservation and source
    /// witness. This is metadata validation, not a DMA or consumer receipt.
    pub fn validate_read_plan(&self, plan: &KvReadPlan) -> HostKvResult<()> {
        if !plan.capability.matches_witness(self) {
            return Err(HostKvError::StaleSource);
        }

        let mut segment_index = 0usize;
        let geometry =
            self.for_each_read_segment_geometry(plan.logical_tokens.clone(), |expected| {
                let segment = plan
                    .segments
                    .get(segment_index)
                    .ok_or(HostKvError::InvalidGeometry)?;
                if segment.capability.as_ref() != plan.capability.as_ref() {
                    return Err(HostKvError::StaleSource);
                }
                if segment.logical_tokens.start != expected.logical_start
                    || segment.logical_tokens.end != expected.logical_end
                    || segment.page_tokens.start != expected.page_tokens_start
                    || segment.page_tokens.end != expected.page_tokens_end
                {
                    return Err(HostKvError::InvalidRange);
                }
                if segment.source_page != expected.source_page {
                    return Err(HostKvError::StaleSource);
                }
                if segment.destination_offset != expected.destination_offset {
                    return Err(HostKvError::InvalidRange);
                }
                if segment.destination_length != expected.destination_length {
                    return Err(HostKvError::InvalidGeometry);
                }
                segment_index = segment_index.checked_add(1).ok_or(HostKvError::Overflow)?;
                Ok(())
            })?;
        if segment_index != plan.segments.len()
            || geometry.destination_length != plan.destination_length
        {
            return Err(HostKvError::InvalidGeometry);
        }
        Ok(())
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

impl HostKvReadCapability {
    fn from_witness(witness: &HostKvAppendReadWitness) -> Self {
        Self {
            transaction: witness.transaction,
            reservation_id: witness.reservation_id,
            owner: witness.source.owner,
            state_slot: witness.source.state_slot,
            generation: witness.source.generation,
            incarnation: witness.source.incarnation,
            schema: witness.source.schema.clone(),
            committed_tokens: witness.source.committed_tokens,
            source_positions: witness.source.positions.clone(),
            reservation_positions: witness.reservation.positions.clone(),
            execution_state_slot: witness.reservation.execution_state_slot,
            execution_generation: witness.reservation.execution_generation,
        }
    }

    fn matches_witness(&self, witness: &HostKvAppendReadWitness) -> bool {
        self.transaction == witness.transaction
            && self.reservation_id == witness.reservation_id
            && self.owner == witness.source.owner
            && self.state_slot == witness.source.state_slot
            && self.generation == witness.source.generation
            && self.incarnation == witness.source.incarnation
            && self.schema == witness.source.schema
            && self.committed_tokens == witness.source.committed_tokens
            && self.source_positions == witness.source.positions
            && self.reservation_positions == witness.reservation.positions
            && self.execution_state_slot == witness.reservation.execution_state_slot
            && self.execution_generation == witness.reservation.execution_generation
    }

    pub const fn transaction(&self) -> ExecutionTransactionId {
        self.transaction
    }
    pub const fn reservation_id(&self) -> u64 {
        self.reservation_id
    }
    pub const fn owner(&self) -> u64 {
        self.owner
    }
    pub const fn state_slot(&self) -> StateSlot {
        self.state_slot
    }
    pub const fn generation(&self) -> u64 {
        self.generation
    }
    pub const fn incarnation(&self) -> u64 {
        self.incarnation
    }
    pub fn schema(&self) -> &KvLayoutSchemaIdentity {
        &self.schema
    }
    pub const fn committed_tokens(&self) -> usize {
        self.committed_tokens
    }
    pub fn source_positions(&self) -> Range<usize> {
        self.source_positions.clone()
    }
    pub fn reservation_positions(&self) -> Range<usize> {
        self.reservation_positions.clone()
    }
    pub const fn execution_state_slot(&self) -> StateSlot {
        self.execution_state_slot
    }
    pub const fn execution_generation(&self) -> u64 {
        self.execution_generation
    }
}

#[derive(Debug, Clone, Copy)]
struct KvReadSegmentGeometry {
    source_page: KvPageId,
    logical_start: usize,
    logical_end: usize,
    page_tokens_start: usize,
    page_tokens_end: usize,
    destination_offset: usize,
    destination_length: usize,
}

#[derive(Debug, Clone, Copy)]
struct KvReadGeometry {
    destination_length: usize,
}

fn checked_bytes_per_token(schema: &KvLayoutSchemaIdentity) -> HostKvResult<usize> {
    if schema.page_size() == 0 || schema.planes().is_empty() {
        return Err(HostKvError::InvalidGeometry);
    }
    let mut total = 0usize;
    for plane in schema.planes() {
        if plane.elements_per_token == 0 || plane.layer_count == 0 {
            return Err(HostKvError::InvalidGeometry);
        }
        let elements = plane
            .elements_per_token
            .checked_mul(plane.layer_count)
            .ok_or(HostKvError::Overflow)?;
        let bytes = elements
            .checked_mul(plane.element_type.bytes_per_element())
            .ok_or(HostKvError::Overflow)?;
        total = total.checked_add(bytes).ok_or(HostKvError::Overflow)?;
    }
    Ok(total)
}

/// One page-bounded logical source range packed into a destination byte range.
///
/// The private capability makes a segment witness-issued. The public fields are
/// only logical metadata: `source_page` is never a physical address and the
/// destination values do not identify an allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvReadSegment {
    pub source_page: KvPageId,
    /// Absolute logical token positions in the witnessed sequence.
    pub logical_tokens: Range<usize>,
    /// Token positions relative to `source_page`.
    pub page_tokens: Range<usize>,
    pub destination_offset: usize,
    pub destination_length: usize,
    capability: Arc<HostKvReadCapability>,
}
impl KvReadSegment {
    pub fn capability(&self) -> &HostKvReadCapability {
        self.capability.as_ref()
    }
}

/// Provider-neutral, non-owning packed description of a witnessed KV read.
///
/// Current append-read semantics are exact contiguous coverage, so this plan
/// emits no implicit zero-fill or gap segments. Physical placement, DMA
/// submission, stream/event state, and completion remain outside this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvReadPlan {
    pub logical_tokens: Range<usize>,
    pub destination_length: usize,
    segments: Vec<KvReadSegment>,
    capability: Arc<HostKvReadCapability>,
}
impl KvReadPlan {
    /// Derive a plan only from an existing witness capability.
    pub fn from_witness(
        witness: &HostKvAppendReadWitness,
        logical_tokens: Range<usize>,
    ) -> HostKvResult<Self> {
        witness.build_read_plan(logical_tokens)
    }

    pub fn segments(&self) -> &[KvReadSegment] {
        &self.segments
    }
    pub fn capability(&self) -> &HostKvReadCapability {
        self.capability.as_ref()
    }
}

impl HostKvAppendReadWitness {
    fn validate_read_geometry(&self, logical_tokens: Range<usize>) -> HostKvResult<KvReadGeometry> {
        self.for_each_read_segment_geometry(logical_tokens, |_| Ok(()))
    }

    fn for_each_read_segment_geometry<F>(
        &self,
        logical_tokens: Range<usize>,
        mut visit: F,
    ) -> HostKvResult<KvReadGeometry>
    where
        F: FnMut(KvReadSegmentGeometry) -> HostKvResult<()>,
    {
        let witness_range = self.positions();
        if logical_tokens.start >= logical_tokens.end
            || logical_tokens.start < witness_range.start
            || logical_tokens.end > witness_range.end
        {
            return Err(HostKvError::InvalidRange);
        }
        let page_size = self.source.schema.page_size();
        let bytes_per_token = checked_bytes_per_token(&self.source.schema)?;
        if self.pages.is_empty() {
            return Err(HostKvError::InvalidGeometry);
        }

        let first_page = logical_tokens.start / page_size;
        let last_page = (logical_tokens.end - 1) / page_size;
        let mut destination_offset = 0usize;
        for page_index in first_page..=last_page {
            let page_origin = page_index
                .checked_mul(page_size)
                .ok_or(HostKvError::Overflow)?;
            let page_limit = page_origin
                .checked_add(page_size)
                .ok_or(HostKvError::Overflow)?;
            let logical_start = logical_tokens.start.max(page_origin);
            let logical_end = logical_tokens.end.min(page_limit);
            if logical_start >= logical_end {
                return Err(HostKvError::InvalidGeometry);
            }
            let token_count = logical_end
                .checked_sub(logical_start)
                .ok_or(HostKvError::Overflow)?;
            let destination_length = token_count
                .checked_mul(bytes_per_token)
                .ok_or(HostKvError::Overflow)?;
            let source_page = *self.pages.get(page_index).ok_or(HostKvError::StaleSource)?;
            visit(KvReadSegmentGeometry {
                source_page,
                logical_start,
                logical_end,
                page_tokens_start: logical_start - page_origin,
                page_tokens_end: logical_end - page_origin,
                destination_offset,
                destination_length,
            })?;
            destination_offset = destination_offset
                .checked_add(destination_length)
                .ok_or(HostKvError::Overflow)?;
        }
        Ok(KvReadGeometry {
            destination_length: destination_offset,
        })
    }

    fn build_read_plan(&self, logical_tokens: Range<usize>) -> HostKvResult<KvReadPlan> {
        let capability = Arc::new(HostKvReadCapability::from_witness(self));
        let mut segments = Vec::new();
        let geometry = self.for_each_read_segment_geometry(logical_tokens.clone(), |expected| {
            segments.push(KvReadSegment {
                source_page: expected.source_page,
                logical_tokens: expected.logical_start..expected.logical_end,
                page_tokens: expected.page_tokens_start..expected.page_tokens_end,
                destination_offset: expected.destination_offset,
                destination_length: expected.destination_length,
                capability: Arc::clone(&capability),
            });
            Ok(())
        })?;
        Ok(KvReadPlan {
            logical_tokens,
            destination_length: geometry.destination_length,
            segments,
            capability,
        })
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

#[cfg(test)]
#[path = "../tests/unit/host_kv/read_plan.rs"]
mod read_plan_tests;
