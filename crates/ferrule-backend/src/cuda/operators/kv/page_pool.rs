//! Model-independent, multi-plane physical KV page storage.
//!
//! Every plane owns one fixed-capacity device buffer. Runtime page IDs map to
//! fixed-size slot ranges inside those buffers. Reservations remain invisible
//! until commit, so allocation or copy failures preserve the previous view.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use ferrule_common::execution::{
    KvBlockId, KvCowReplacement, KvElementType, KvPageId, KvPlaneDescriptor,
};
use ferrule_common::{Error, Result};

use crate::cuda::context::{CudaBf16Buffer, CudaF32Buffer, CudaI32Buffer, CudaOperators};
use crate::cuda::runtime;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedPlaneLayout {
    pub page_tokens: usize,
    pub elements_per_token: usize,
    pub layer_index: usize,
    pub layer_count: usize,
}

impl PagedPlaneLayout {
    /// Interprets this legacy layout as F32 without changing its kernel ABI.
    pub const fn element_type(&self) -> KvElementType {
        KvElementType::F32
    }

    /// Adds an explicit physical element type for byte addressing and typed storage.
    pub const fn typed(self, element_type: KvElementType) -> TypedPagedPlaneLayout {
        TypedPagedPlaneLayout {
            layout: self,
            element_type,
        }
    }

    pub const fn checked_elements_per_page(&self) -> Option<usize> {
        match self.page_tokens.checked_mul(self.elements_per_token) {
            Some(elements) => elements.checked_mul(self.layer_count),
            None => None,
        }
    }

    pub const fn checked_bytes_per_page(&self) -> Option<usize> {
        match self.checked_elements_per_page() {
            Some(elements) => elements.checked_mul(std::mem::size_of::<f32>()),
            None => None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.page_tokens == 0
            || self.elements_per_token == 0
            || self.layer_count == 0
            || self.layer_index >= self.layer_count
        {
            return Err(pool_error("invalid paged plane layout"));
        }
        for value in [
            self.page_tokens,
            self.elements_per_token,
            self.layer_index,
            self.layer_count,
        ] {
            u32::try_from(value)
                .map_err(|_| pool_error("paged plane layout exceeds u32 kernel ABI"))?;
        }
        Ok(())
    }

    pub fn resolve_row_offset(
        &self,
        sequence: usize,
        logical_row: usize,
        storage_elements: usize,
        block_slots: &[i32],
        block_offsets: &[i32],
    ) -> Option<usize> {
        self.validate().ok()?;
        let start = usize::try_from(*block_offsets.get(sequence)?).ok()?;
        let end = usize::try_from(*block_offsets.get(sequence + 1)?).ok()?;
        let entry = start.checked_add(logical_row / self.page_tokens)?;
        if entry >= end {
            return None;
        }
        let slot = usize::try_from(*block_slots.get(entry)?).ok()?;
        let slot_stride = self
            .layer_count
            .checked_mul(self.page_tokens)?
            .checked_mul(self.elements_per_token)?;
        let layer_stride = self.page_tokens.checked_mul(self.elements_per_token)?;
        let offset = slot
            .checked_mul(slot_stride)?
            .checked_add(self.layer_index.checked_mul(layer_stride)?)?
            .checked_add((logical_row % self.page_tokens).checked_mul(self.elements_per_token)?)?;
        (offset.checked_add(self.elements_per_token)? <= storage_elements).then_some(offset)
    }

    /// Resolves a row-owned logical position through a flattened sequence table.
    /// `row_sequence_ids[row]` selects the sequence's block-offset range.
    pub fn resolve_selected_row_offset(
        &self,
        row: usize,
        logical_row: usize,
        storage_elements: usize,
        block_slots: &[i32],
        block_offsets: &[i32],
        row_sequence_ids: &[i32],
    ) -> Option<usize> {
        let sequence = usize::try_from(*row_sequence_ids.get(row)?).ok()?;
        self.resolve_row_offset(
            sequence,
            logical_row,
            storage_elements,
            block_slots,
            block_offsets,
        )
    }
}

/// Dtype-aware view of a paged plane layout.
///
/// Element offsets remain the kernel-facing coordinate system; byte helpers use
/// the explicit element type and never report BF16 storage as F32-equivalent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypedPagedPlaneLayout {
    pub layout: PagedPlaneLayout,
    pub element_type: KvElementType,
}

impl TypedPagedPlaneLayout {
    pub const fn new(layout: PagedPlaneLayout, element_type: KvElementType) -> Self {
        Self {
            layout,
            element_type,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.layout.validate()
    }

    pub const fn checked_elements_per_page(&self) -> Option<usize> {
        self.layout.checked_elements_per_page()
    }

    pub const fn checked_bytes_per_page(&self) -> Option<usize> {
        match self.checked_elements_per_page() {
            Some(elements) => elements.checked_mul(self.element_type.bytes_per_element()),
            None => None,
        }
    }

    pub fn resolve_row_offset(
        &self,
        sequence: usize,
        logical_row: usize,
        storage_elements: usize,
        block_slots: &[i32],
        block_offsets: &[i32],
    ) -> Option<usize> {
        self.layout.resolve_row_offset(
            sequence,
            logical_row,
            storage_elements,
            block_slots,
            block_offsets,
        )
    }

    pub fn resolve_row_byte_range(
        &self,
        sequence: usize,
        logical_row: usize,
        storage_bytes: usize,
        block_slots: &[i32],
        block_offsets: &[i32],
    ) -> Option<Range<usize>> {
        let element_bytes = self.element_type.bytes_per_element();
        if !storage_bytes.is_multiple_of(element_bytes) {
            return None;
        }
        let offset = self.resolve_row_offset(
            sequence,
            logical_row,
            storage_bytes / element_bytes,
            block_slots,
            block_offsets,
        )?;
        let start = offset.checked_mul(element_bytes)?;
        let len = self.layout.elements_per_token.checked_mul(element_bytes)?;
        Some(start..start.checked_add(len)?)
    }

    pub fn resolve_selected_row_offset(
        &self,
        row: usize,
        logical_row: usize,
        storage_elements: usize,
        block_slots: &[i32],
        block_offsets: &[i32],
        row_sequence_ids: &[i32],
    ) -> Option<usize> {
        self.layout.resolve_selected_row_offset(
            row,
            logical_row,
            storage_elements,
            block_slots,
            block_offsets,
            row_sequence_ids,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvPagePoolStats {
    pub allocated_slots: usize,
    pub resident_pages: usize,
    pub pending_pages: usize,
    pub free_slots: usize,
    pub allocated_bytes: usize,
    pub resident_bytes: usize,
    pub utilization: f64,
    /// Fraction of free slots not contained in the largest contiguous free run.
    pub external_fragmentation: f64,
}

#[derive(Debug, Clone)]
pub struct KvPoolReservation {
    token: u64,
    pages: Vec<KvPageId>,
    cow: Option<KvCowReplacement>,
}

impl KvPoolReservation {
    pub fn pages(&self) -> &[KvPageId] {
        &self.pages
    }

    pub fn cow_replacement(&self) -> Option<KvCowReplacement> {
        self.cow
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvHostPlaneSnapshot {
    F32(Vec<u32>),
    Bf16(Vec<u16>),
}

impl KvHostPlaneSnapshot {
    pub const fn element_type(&self) -> KvElementType {
        match self {
            Self::F32(_) => KvElementType::F32,
            Self::Bf16(_) => KvElementType::Bf16,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::Bf16(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn byte_len(&self) -> usize {
        self.len()
            .saturating_mul(self.element_type().bytes_per_element())
    }

    pub fn f32_bits(&self) -> Option<&[u32]> {
        match self {
            Self::F32(values) => Some(values),
            Self::Bf16(_) => None,
        }
    }

    pub fn bf16_words(&self) -> Option<&[u16]> {
        match self {
            Self::F32(_) => None,
            Self::Bf16(values) => Some(values),
        }
    }

    pub fn bytes(&self) -> Vec<u8> {
        match self {
            Self::F32(values) => values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
            Self::Bf16(values) => values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KvHostSnapshot {
    pub page_id: KvPageId,
    pub planes: Vec<Vec<f32>>,
}

impl PartialEq for KvHostSnapshot {
    fn eq(&self, other: &Self) -> bool {
        self.page_id == other.page_id
            && self.planes.len() == other.planes.len()
            && self.planes.iter().zip(&other.planes).all(|(left, right)| {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right)
                        .all(|(left, right)| left.to_bits() == right.to_bits())
            })
    }
}

impl Eq for KvHostSnapshot {}

/// Authoritative raw host snapshot for mixed typed planes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedKvHostSnapshot {
    pub page_id: KvPageId,
    pub planes: Vec<KvHostPlaneSnapshot>,
}

#[derive(Debug, Clone)]
enum StoredKvHostSnapshot {
    F32(KvHostSnapshot),
    Typed(TypedKvHostSnapshot),
}

impl StoredKvHostSnapshot {
    const fn page_id(&self) -> KvPageId {
        match self {
            Self::F32(snapshot) => snapshot.page_id,
            Self::Typed(snapshot) => snapshot.page_id,
        }
    }
}

struct PendingReservation {
    pages: Vec<(KvPageId, u32)>,
    cow: Option<(KvCowReplacement, u32)>,
}

#[derive(Default)]
struct MetadataState {
    mappings: HashMap<KvPageId, u32>,
    free_slots: Vec<u32>,
}

impl MetadataState {
    fn with_slots(max_slots: usize) -> Self {
        Self {
            mappings: HashMap::new(),
            // Reverse order makes the first pop return slot zero.
            free_slots: (0..max_slots as u32).rev().collect(),
        }
    }

    fn commit(&mut self, pages: &[(KvPageId, u32)]) {
        self.mappings.extend(pages.iter().copied());
    }

    fn release(&mut self, page: KvPageId) -> Option<u32> {
        let slot = self.mappings.remove(&page)?;
        self.free_slots.push(slot);
        Some(slot)
    }

    fn physical_slot_i32(&self, page: KvPageId, index: usize) -> Result<i32> {
        let slot = self.mappings.get(&page).copied().ok_or_else(|| {
            pool_error(format!(
                "logical page {} at block index {index} is not resident",
                page.0
            ))
        })?;
        i32::try_from(slot).map_err(|_| {
            pool_error(format!(
                "physical slot {slot} for logical page {} exceeds the i32 kernel ABI",
                page.0
            ))
        })
    }

    fn physical_slots(&self, pages: &[KvPageId]) -> Result<Vec<i32>> {
        pages
            .iter()
            .enumerate()
            .map(|(index, page)| self.physical_slot_i32(*page, index))
            .collect()
    }

    fn physical_slots_for_block_ids(&self, blocks: &[KvBlockId]) -> Result<Vec<i32>> {
        blocks
            .iter()
            .enumerate()
            .map(|(index, block)| self.physical_slot_i32(KvPageId(block.get()), index))
            .collect()
    }

    fn compact_table(&self) -> Vec<i32> {
        let mut mappings: Vec<_> = self.mappings.iter().collect();
        mappings.sort_unstable_by_key(|(page, _)| page.0);
        let mut table = Vec::with_capacity(mappings.len() * 2);
        for (page, slot) in mappings {
            table.push(i32::from_ne_bytes(page.0.to_ne_bytes()));
            table.push(*slot as i32);
        }
        table
    }
}

#[derive(Debug)]
struct PageLayout {
    page_elements: Box<[usize]>,
    page_bytes: Box<[usize]>,
    page_tokens: usize,
    bytes_per_slot: usize,
    max_slots: usize,
}

impl PageLayout {
    fn new(planes: &[KvPlaneDescriptor], page_tokens: usize, max_slots: usize) -> Result<Self> {
        if planes.is_empty() {
            return Err(pool_error("at least one KV plane is required"));
        }
        if page_tokens == 0 {
            return Err(pool_error("page_tokens must be positive"));
        }
        if max_slots == 0 {
            return Err(pool_error("max_slots must be positive"));
        }
        if max_slots > i32::MAX as usize + 1 {
            return Err(pool_error("max_slots exceeds i32 page-table slot range"));
        }
        let page_elements = planes
            .iter()
            .map(|plane| {
                plane
                    .checked_page_elements(page_tokens)
                    .ok_or_else(|| pool_error("plane page element count overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        if page_elements.contains(&0) {
            return Err(pool_error("plane pages must not be empty"));
        }
        let page_bytes = planes
            .iter()
            .map(|plane| {
                plane
                    .checked_page_bytes(page_tokens)
                    .ok_or_else(|| pool_error("plane page byte count overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        let bytes_per_slot = page_bytes
            .iter()
            .try_fold(0usize, |sum, bytes| sum.checked_add(*bytes))
            .ok_or_else(|| pool_error("physical slot byte count overflow"))?;
        for (elements, bytes) in page_elements.iter().zip(&page_bytes) {
            elements
                .checked_mul(max_slots)
                .ok_or_else(|| pool_error("plane pool element count overflow"))?;
            bytes
                .checked_mul(max_slots)
                .ok_or_else(|| pool_error("plane pool byte count overflow"))?;
        }
        Ok(Self {
            page_elements: page_elements.into_boxed_slice(),
            page_bytes: page_bytes.into_boxed_slice(),
            page_tokens,
            bytes_per_slot,
            max_slots,
        })
    }

    fn slot_range(&self, slot: u32, plane: usize) -> Option<Range<usize>> {
        let page_elements = *self.page_elements.get(plane)?;
        let start = (slot as usize).checked_mul(page_elements)?;
        Some(start..start.checked_add(page_elements)?)
    }

    fn slot_byte_range(&self, slot: u32, plane: usize) -> Option<Range<usize>> {
        let page_bytes = *self.page_bytes.get(plane)?;
        let start = (slot as usize).checked_mul(page_bytes)?;
        Some(start..start.checked_add(page_bytes)?)
    }
}

/// One real, typed CUDA allocation for a KV plane.
pub enum CudaKvPlaneStorage {
    F32(CudaF32Buffer),
    Bf16(CudaBf16Buffer),
}

impl CudaKvPlaneStorage {
    pub const fn element_type(&self) -> KvElementType {
        match self {
            Self::F32(_) => KvElementType::F32,
            Self::Bf16(_) => KvElementType::Bf16,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F32(buffer) => buffer.len(),
            Self::Bf16(buffer) => buffer.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn byte_len(&self) -> usize {
        self.len()
            .saturating_mul(self.element_type().bytes_per_element())
    }

    pub fn as_f32(&self) -> Option<&CudaF32Buffer> {
        match self {
            Self::F32(buffer) => Some(buffer),
            Self::Bf16(_) => None,
        }
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut CudaF32Buffer> {
        match self {
            Self::F32(buffer) => Some(buffer),
            Self::Bf16(_) => None,
        }
    }

    pub fn as_bf16(&self) -> Option<&CudaBf16Buffer> {
        match self {
            Self::F32(_) => None,
            Self::Bf16(buffer) => Some(buffer),
        }
    }

    pub fn as_bf16_mut(&mut self) -> Option<&mut CudaBf16Buffer> {
        match self {
            Self::F32(_) => None,
            Self::Bf16(buffer) => Some(buffer),
        }
    }
}

/// Fixed-capacity CUDA KV page pool with one contiguous typed buffer per plane.
pub struct CudaKvPagePool {
    planes: Box<[KvPlaneDescriptor]>,
    layout: PageLayout,
    plane_storage: Vec<CudaKvPlaneStorage>,
    metadata: MetadataState,
    pending: HashMap<u64, PendingReservation>,
    snapshots: HashMap<KvPageId, StoredKvHostSnapshot>,
    next_token: u64,
}

impl CudaKvPagePool {
    /// Allocates exactly one correctly typed fixed-capacity buffer per plane.
    pub fn new(
        context: &CudaOperators,
        planes: &[KvPlaneDescriptor],
        page_tokens: usize,
        max_slots: usize,
    ) -> Result<Self> {
        let layout = PageLayout::new(planes, page_tokens, max_slots)?;
        let plane_storage = planes
            .iter()
            .zip(layout.page_elements.iter())
            .map(|(plane, elements)| {
                let capacity = elements
                    .checked_mul(max_slots)
                    .ok_or_else(|| pool_error("plane pool element count overflow"))?;
                match plane.element_type {
                    KvElementType::F32 => context
                        .zero_f32_buffer(capacity)
                        .map(CudaKvPlaneStorage::F32),
                    KvElementType::Bf16 => context
                        .zero_bf16_buffer(capacity)
                        .map(CudaKvPlaneStorage::Bf16),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            planes: planes.to_vec().into_boxed_slice(),
            layout,
            plane_storage,
            metadata: MetadataState::with_slots(max_slots),
            pending: HashMap::new(),
            snapshots: HashMap::new(),
            next_token: 1,
        })
    }

    pub fn planes(&self) -> &[KvPlaneDescriptor] {
        &self.planes
    }

    pub fn page_elements(&self, plane: usize) -> Option<usize> {
        self.layout.page_elements.get(plane).copied()
    }

    pub fn page_bytes(&self, plane: usize) -> Option<usize> {
        self.layout.page_bytes.get(plane).copied()
    }

    pub fn bytes_per_slot(&self) -> usize {
        self.layout.bytes_per_slot
    }

    pub fn page_tokens(&self) -> usize {
        self.layout.page_tokens
    }

    /// Returns the complete contiguous storage with its real physical type.
    pub fn typed_plane_storage(&self, plane: usize) -> Option<&CudaKvPlaneStorage> {
        self.plane_storage.get(plane)
    }

    pub fn typed_plane_storage_mut(&mut self, plane: usize) -> Option<&mut CudaKvPlaneStorage> {
        self.plane_storage.get_mut(plane)
    }

    /// Compatibility accessor for an F32 plane. Returns `None` for BF16.
    pub fn plane_storage(&self, plane: usize) -> Option<&CudaF32Buffer> {
        self.typed_plane_storage(plane)?.as_f32()
    }

    /// Compatibility accessor for an F32 plane. Returns `None` for BF16.
    pub fn plane_storage_mut(&mut self, plane: usize) -> Option<&mut CudaF32Buffer> {
        self.typed_plane_storage_mut(plane)?.as_f32_mut()
    }

    pub fn plane_bf16_storage(&self, plane: usize) -> Option<&CudaBf16Buffer> {
        self.typed_plane_storage(plane)?.as_bf16()
    }

    pub fn plane_bf16_storage_mut(&mut self, plane: usize) -> Option<&mut CudaBf16Buffer> {
        self.typed_plane_storage_mut(plane)?.as_bf16_mut()
    }

    /// Mutably borrow two distinct BF16 plane allocations without aliasing.
    ///
    /// The returned tuple follows the caller-provided plane order even when the
    /// second index precedes the first in storage. Equal or out-of-range indices
    /// are rejected before the slice is split.
    pub fn bf16_plane_pair_mut(
        &mut self,
        first_plane: usize,
        second_plane: usize,
    ) -> Result<(&mut CudaBf16Buffer, &mut CudaBf16Buffer)> {
        let (first, second) =
            distinct_pair_mut(&mut self.plane_storage, first_plane, second_plane)?;
        let first = first
            .as_bf16_mut()
            .ok_or_else(|| pool_error(format!("plane {first_plane} is not BF16 storage")))?;
        let second = second
            .as_bf16_mut()
            .ok_or_else(|| pool_error(format!("plane {second_plane} is not BF16 storage")))?;
        Ok((first, second))
    }

    pub fn plane_storage_bytes(&self, plane: usize) -> Option<usize> {
        Some(self.typed_plane_storage(plane)?.byte_len())
    }

    pub fn physical_slot(&self, page: KvPageId) -> Option<u32> {
        self.metadata.mappings.get(&page).copied()
    }

    /// Lower logical page IDs to physical i32 slots in the same packed order.
    pub fn physical_slots_for_pages(&self, pages: &[KvPageId]) -> Result<Vec<i32>> {
        self.metadata.physical_slots(pages)
    }

    /// Lower common execution block IDs through this pool authoritative mapping.
    pub fn physical_slots_for_block_ids(&self, blocks: &[KvBlockId]) -> Result<Vec<i32>> {
        self.metadata.physical_slots_for_block_ids(blocks)
    }

    pub fn page_element_offset(&self, page: KvPageId, plane: usize) -> Option<usize> {
        Some(self.plane_slot_range(page, plane)?.start)
    }

    pub fn page_byte_offset(&self, page: KvPageId, plane: usize) -> Option<usize> {
        Some(self.plane_slot_byte_range(page, plane)?.start)
    }

    pub fn plane_slot_range(&self, page: KvPageId, plane: usize) -> Option<Range<usize>> {
        self.layout.slot_range(self.physical_slot(page)?, plane)
    }

    pub fn plane_slot_byte_range(&self, page: KvPageId, plane: usize) -> Option<Range<usize>> {
        self.layout
            .slot_byte_range(self.physical_slot(page)?, plane)
    }

    /// Returns a page's reserved slot before commit, for prepared kernel wiring.
    pub fn pending_slot(&self, reservation: &KvPoolReservation, page: KvPageId) -> Option<u32> {
        self.pending
            .get(&reservation.token)?
            .pages
            .iter()
            .find_map(|(pending_page, slot)| (*pending_page == page).then_some(*slot))
    }

    pub fn pending_plane_slot_range(
        &self,
        reservation: &KvPoolReservation,
        page: KvPageId,
        plane: usize,
    ) -> Option<Range<usize>> {
        self.layout
            .slot_range(self.pending_slot(reservation, page)?, plane)
    }

    pub fn pending_plane_slot_byte_range(
        &self,
        reservation: &KvPoolReservation,
        page: KvPageId,
        plane: usize,
    ) -> Option<Range<usize>> {
        self.layout
            .slot_byte_range(self.pending_slot(reservation, page)?, plane)
    }

    pub fn pending_replacement_slot(&self, reservation: &KvPoolReservation) -> Option<u32> {
        self.pending
            .get(&reservation.token)?
            .cow
            .map(|(_, slot)| slot)
    }

    pub fn ensure(&mut self, context: &CudaOperators, page: KvPageId) -> Result<u32> {
        if let Some(slot) = self.physical_slot(page) {
            return Ok(slot);
        }
        let reservation = self.reserve(context, &[page], None)?;
        self.commit(reservation)?;
        self.physical_slot(page)
            .ok_or_else(|| pool_error("ensure commit did not publish its page"))
    }

    /// Reserves free physical slots and clears only those slot ranges.
    pub fn reserve(
        &mut self,
        context: &CudaOperators,
        pages: &[KvPageId],
        cow: Option<KvCowReplacement>,
    ) -> Result<KvPoolReservation> {
        let mut requested = Vec::with_capacity(pages.len() + usize::from(cow.is_some()));
        let mut unique = HashSet::with_capacity(requested.capacity());
        for page in pages {
            if !unique.insert(*page) {
                return Err(pool_error("reservation contains a duplicate page ID"));
            }
            requested.push(*page);
        }
        if let Some(replacement) = cow {
            if self.physical_slot(replacement.source).is_none() {
                return Err(pool_error("COW source page is not resident"));
            }
            if !unique.insert(replacement.replacement) {
                return Err(pool_error("COW replacement duplicates a reserved page ID"));
            }
            requested.push(replacement.replacement);
        }
        for page in &requested {
            if self.metadata.mappings.contains_key(page)
                || self
                    .pending
                    .values()
                    .any(|pending| pending.pages.iter().any(|(id, _)| id == page))
            {
                return Err(pool_error(
                    "runtime page ID is already resident or reserved",
                ));
            }
        }

        let token = self.next_token;
        let next_token = token
            .checked_add(1)
            .ok_or_else(|| pool_error("reservation token overflow"))?;
        let mut allocated = Vec::with_capacity(requested.len());
        for page in &requested {
            match self.allocate_slot(context) {
                Ok(slot) => allocated.push((*page, slot)),
                Err(error) => {
                    self.recycle_allocated(&allocated);
                    return Err(error);
                }
            }
        }

        let cow_slot = cow.map(|replacement| {
            let slot = allocated
                .iter()
                .find(|(page, _)| *page == replacement.replacement)
                .expect("COW replacement was allocated")
                .1;
            (replacement, slot)
        });
        if let Some((replacement, destination)) = cow_slot {
            let source = self
                .physical_slot(replacement.source)
                .expect("validated COW source");
            if let Err(error) = self.copy_slot(context, source, destination) {
                self.recycle_allocated(&allocated);
                return Err(error);
            }
        }

        self.next_token = next_token;
        self.pending.insert(
            token,
            PendingReservation {
                pages: allocated,
                cow: cow_slot,
            },
        );
        Ok(KvPoolReservation {
            token,
            pages: pages.to_vec(),
            cow,
        })
    }

    /// Atomically publishes a reservation into the visible page-table view.
    pub fn commit(&mut self, reservation: KvPoolReservation) -> Result<()> {
        self.commit_many(vec![reservation])
    }

    /// Atomically publishes every reservation in one packed backend batch.
    ///
    /// All reservation tokens and COW sources are validated before any mapping is
    /// made visible. Validation failure recycles every still-pending reservation
    /// supplied by the caller, so no provisional physical slots are stranded.
    pub fn commit_many(&mut self, reservations: Vec<KvPoolReservation>) -> Result<()> {
        let mut tokens = HashSet::with_capacity(reservations.len());
        let validation = (|| -> Result<()> {
            for reservation in &reservations {
                if !tokens.insert(reservation.token) {
                    return Err(pool_error("packed commit contains a duplicate reservation"));
                }
                let pending = self
                    .pending
                    .get(&reservation.token)
                    .ok_or_else(|| pool_error("unknown or completed reservation"))?;
                if pending
                    .pages
                    .iter()
                    .any(|(page, _)| self.metadata.mappings.contains_key(page))
                {
                    return Err(pool_error("reservation became stale before commit"));
                }
                if pending
                    .cow
                    .is_some_and(|(cow, _)| self.physical_slot(cow.source).is_none())
                {
                    return Err(pool_error("COW source was released before commit"));
                }
            }
            Ok(())
        })();

        if let Err(error) = validation {
            for reservation in reservations {
                if let Some(pending) = self.pending.remove(&reservation.token) {
                    self.recycle_allocated(&pending.pages);
                }
            }
            return Err(error);
        }

        let mut pending_batch = Vec::with_capacity(reservations.len());
        for reservation in reservations {
            pending_batch.push(
                self.pending
                    .remove(&reservation.token)
                    .expect("packed reservation was validated"),
            );
        }
        for pending in pending_batch {
            self.metadata.commit(&pending.pages);
        }
        Ok(())
    }

    pub fn rollback(
        &mut self,
        _context: &CudaOperators,
        reservation: KvPoolReservation,
    ) -> Result<()> {
        let pending = self
            .pending
            .remove(&reservation.token)
            .ok_or_else(|| pool_error("unknown or completed reservation"))?;
        self.recycle_allocated(&pending.pages);
        Ok(())
    }

    pub fn release(&mut self, _context: &CudaOperators, page: KvPageId) -> Result<()> {
        if self.metadata.release(page).is_some() || self.snapshots.remove(&page).is_some() {
            return Ok(());
        }
        Err(pool_error(
            "cannot release an unknown physical or suspended page",
        ))
    }

    pub fn download_bf16_words_range(
        &self,
        context: &CudaOperators,
        plane: usize,
        offset: usize,
        len: usize,
    ) -> Result<Vec<u16>> {
        let storage = self
            .plane_bf16_storage(plane)
            .ok_or_else(|| pool_error("requested plane is not BF16 storage"))?;
        download_bf16_words_range(context, storage, offset, len)
    }

    pub fn overwrite_bf16_words_range(
        &mut self,
        context: &CudaOperators,
        plane: usize,
        offset: usize,
        values: &[u16],
    ) -> Result<()> {
        let storage = self
            .plane_bf16_storage(plane)
            .ok_or_else(|| pool_error("requested plane is not BF16 storage"))?;
        overwrite_bf16_words_range(context, storage, offset, values)
    }

    /// Scatters a typed BF16 device buffer into rows whose sequence index equals
    /// their row index.
    #[allow(clippy::too_many_arguments)]
    pub fn paged_plane_scatter_bf16_rows_from_device(
        &mut self,
        context: &CudaOperators,
        plane: usize,
        values: &CudaBf16Buffer,
        logical_rows: &[usize],
        block_slots: &[i32],
        block_offsets: &[i32],
        mask: Option<&[i32]>,
        layout: TypedPagedPlaneLayout,
    ) -> Result<()> {
        let row_sequence_ids = (0..logical_rows.len())
            .map(|row| {
                i32::try_from(row)
                    .map_err(|_| pool_error("BF16 scatter row index exceeds i32 sequence range"))
            })
            .collect::<Result<Vec<_>>>()?;
        self.paged_plane_scatter_bf16_selected_rows_from_device(
            context,
            plane,
            values,
            logical_rows,
            block_slots,
            block_offsets,
            &row_sequence_ids,
            mask,
            layout,
        )
    }

    /// Scatters a typed BF16 device buffer through an explicit row-to-sequence
    /// mapping.
    #[allow(clippy::too_many_arguments)]
    pub fn paged_plane_scatter_bf16_selected_rows_from_device(
        &mut self,
        context: &CudaOperators,
        plane: usize,
        values: &CudaBf16Buffer,
        logical_rows: &[usize],
        block_slots: &[i32],
        block_offsets: &[i32],
        row_sequence_ids: &[i32],
        mask: Option<&[i32]>,
        layout: TypedPagedPlaneLayout,
    ) -> Result<()> {
        layout.validate()?;
        if layout.element_type != KvElementType::Bf16 {
            return Err(pool_error("BF16 scatter requires an explicit BF16 layout"));
        }
        let physical_layout = layout.layout;
        if physical_layout.page_tokens != self.page_tokens() {
            return Err(pool_error("BF16 scatter page-token layout mismatch"));
        }
        let descriptor = self
            .planes
            .get(plane)
            .ok_or_else(|| pool_error("BF16 scatter plane index is out of range"))?;
        if descriptor.element_type != layout.element_type
            || descriptor.elements_per_token != physical_layout.elements_per_token
            || descriptor.layer_count != physical_layout.layer_count
        {
            return Err(pool_error("BF16 scatter typed plane layout mismatch"));
        }
        if logical_rows.len() != row_sequence_ids.len()
            || mask.is_some_and(|mask| mask.len() != logical_rows.len())
        {
            return Err(pool_error("BF16 scatter row metadata length mismatch"));
        }
        let expected_values = logical_rows
            .len()
            .checked_mul(physical_layout.elements_per_token)
            .ok_or_else(|| pool_error("BF16 scatter value count overflow"))?;
        if values.len() != expected_values {
            return Err(pool_error("BF16 scatter value count mismatch"));
        }
        let storage = self
            .plane_bf16_storage(plane)
            .ok_or_else(|| pool_error("BF16 scatter plane is not BF16 storage"))?;
        let stream = context.stream_clone();
        for row in 0..logical_rows.len() {
            if mask.is_some_and(|mask| mask[row] == 0) {
                continue;
            }
            let offset = layout
                .resolve_selected_row_offset(
                    row,
                    logical_rows[row],
                    storage.len(),
                    block_slots,
                    block_offsets,
                    row_sequence_ids,
                )
                .ok_or_else(|| pool_error("BF16 scatter row resolves outside storage"))?;
            let value_start = row
                .checked_mul(physical_layout.elements_per_token)
                .ok_or_else(|| pool_error("BF16 scatter source offset overflow"))?;
            let value_end = value_start
                .checked_add(physical_layout.elements_per_token)
                .ok_or_else(|| pool_error("BF16 scatter source range overflow"))?;
            let source = values
                .as_device_buffer()
                .slice(value_start, value_end - value_start)?;
            let destination = storage
                .as_device_buffer()
                .slice(offset, physical_layout.elements_per_token)?;
            runtime::copy_device_to_device(
                &stream,
                destination.cu_deviceptr(),
                source.cu_deviceptr(),
                source.num_bytes(),
            )?;
        }
        Ok(())
    }

    /// Downloads only F32 page ranges, then releases mappings after all copies succeed.
    ///
    /// Mixed pools must use [`Self::preempt_typed`] so BF16 payloads remain raw words.
    pub fn preempt(
        &mut self,
        context: &CudaOperators,
        pages: &[KvPageId],
    ) -> Result<Vec<KvHostSnapshot>> {
        if self
            .planes
            .iter()
            .any(|plane| plane.element_type != KvElementType::F32)
        {
            return Err(pool_error(
                "preempt requires all-F32 planes; use preempt_typed",
            ));
        }
        let mut unique = HashSet::with_capacity(pages.len());
        let mut snapshots = Vec::with_capacity(pages.len());
        for page in pages {
            if !unique.insert(*page) {
                return Err(pool_error("preempt contains a duplicate page ID"));
            }
            let slot = self
                .physical_slot(*page)
                .ok_or_else(|| pool_error("cannot preempt a non-resident page"))?;
            let mut planes = Vec::with_capacity(self.plane_storage.len());
            for (plane, storage) in self.plane_storage.iter().enumerate() {
                let range = self
                    .layout
                    .slot_range(slot, plane)
                    .expect("resident slot and plane are in range");
                let storage = storage
                    .as_f32()
                    .expect("legacy preempt validated all-F32 storage");
                planes.push(context.download_f32_range(storage, range.start, range.len())?);
            }
            snapshots.push(KvHostSnapshot {
                page_id: *page,
                planes,
            });
        }
        for snapshot in &snapshots {
            self.metadata.release(snapshot.page_id);
            self.snapshots.insert(
                snapshot.page_id,
                StoredKvHostSnapshot::F32(snapshot.clone()),
            );
        }
        Ok(snapshots)
    }

    pub fn preempt_typed(
        &mut self,
        context: &CudaOperators,
        pages: &[KvPageId],
    ) -> Result<Vec<TypedKvHostSnapshot>> {
        let mut unique = HashSet::with_capacity(pages.len());
        let mut snapshots = Vec::with_capacity(pages.len());
        for page in pages {
            if !unique.insert(*page) {
                return Err(pool_error("preempt contains a duplicate page ID"));
            }
            let slot = self
                .physical_slot(*page)
                .ok_or_else(|| pool_error("cannot preempt a non-resident page"))?;
            let mut planes = Vec::with_capacity(self.plane_storage.len());
            for (plane, storage) in self.plane_storage.iter().enumerate() {
                let range = self
                    .layout
                    .slot_range(slot, plane)
                    .expect("resident slot and plane are in range");
                planes.push(match storage {
                    CudaKvPlaneStorage::F32(storage) => KvHostPlaneSnapshot::F32(
                        context
                            .download_f32_range(storage, range.start, range.len())?
                            .into_iter()
                            .map(f32::to_bits)
                            .collect(),
                    ),
                    CudaKvPlaneStorage::Bf16(storage) => KvHostPlaneSnapshot::Bf16(
                        download_bf16_words_range(context, storage, range.start, range.len())?,
                    ),
                });
            }
            snapshots.push(TypedKvHostSnapshot {
                page_id: *page,
                planes,
            });
        }
        for snapshot in &snapshots {
            self.metadata.release(snapshot.page_id);
            self.snapshots.insert(
                snapshot.page_id,
                StoredKvHostSnapshot::Typed(snapshot.clone()),
            );
        }
        Ok(snapshots)
    }

    /// Uploads only reserved page ranges. Failed uploads remain invisible and roll back.
    pub fn restore(&mut self, context: &CudaOperators, pages: &[KvPageId]) -> Result<()> {
        let snapshots: Vec<_> = pages
            .iter()
            .map(|page| {
                self.snapshots
                    .get(page)
                    .cloned()
                    .ok_or_else(|| pool_error("no host snapshot for requested page"))
            })
            .collect::<Result<_>>()?;
        let reservation = self.reserve(context, pages, None)?;
        let upload = (|| -> Result<()> {
            for snapshot in &snapshots {
                let slot = self
                    .pending_slot(&reservation, snapshot.page_id())
                    .expect("restored page is pending");
                match snapshot {
                    StoredKvHostSnapshot::F32(snapshot) => {
                        if snapshot.planes.len() != self.plane_storage.len() {
                            return Err(pool_error("host snapshot plane count mismatch"));
                        }
                        for (plane, values) in snapshot.planes.iter().enumerate() {
                            let range = self
                                .layout
                                .slot_range(slot, plane)
                                .expect("pending slot and plane are in range");
                            if values.len() != range.len() {
                                return Err(pool_error("host snapshot plane length mismatch"));
                            }
                            let storage = self.plane_storage[plane]
                                .as_f32_mut()
                                .ok_or_else(|| pool_error("host snapshot plane dtype mismatch"))?;
                            context.overwrite_f32_range(values, storage, range.start)?;
                        }
                    }
                    StoredKvHostSnapshot::Typed(snapshot) => {
                        if snapshot.planes.len() != self.plane_storage.len() {
                            return Err(pool_error("host snapshot plane count mismatch"));
                        }
                        for (plane, values) in snapshot.planes.iter().enumerate() {
                            let range = self
                                .layout
                                .slot_range(slot, plane)
                                .expect("pending slot and plane are in range");
                            if values.len() != range.len() {
                                return Err(pool_error("host snapshot plane length mismatch"));
                            }
                            match (&mut self.plane_storage[plane], values) {
                                (
                                    CudaKvPlaneStorage::F32(storage),
                                    KvHostPlaneSnapshot::F32(bits),
                                ) => {
                                    let values = bits
                                        .iter()
                                        .copied()
                                        .map(f32::from_bits)
                                        .collect::<Vec<_>>();
                                    context.overwrite_f32_range(&values, storage, range.start)?;
                                }
                                (
                                    CudaKvPlaneStorage::Bf16(storage),
                                    KvHostPlaneSnapshot::Bf16(words),
                                ) => overwrite_bf16_words_range(
                                    context,
                                    storage,
                                    range.start,
                                    words,
                                )?,
                                _ => return Err(pool_error("host snapshot plane dtype mismatch")),
                            }
                        }
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = upload {
            self.rollback(context, reservation)?;
            return Err(error);
        }
        self.commit(reservation)?;
        for page in pages {
            self.snapshots.remove(page);
        }
        Ok(())
    }

    pub fn snapshot(&self, page: KvPageId) -> Option<&KvHostSnapshot> {
        match self.snapshots.get(&page)? {
            StoredKvHostSnapshot::F32(snapshot) => Some(snapshot),
            StoredKvHostSnapshot::Typed(_) => None,
        }
    }

    pub fn typed_snapshot(&self, page: KvPageId) -> Option<&TypedKvHostSnapshot> {
        match self.snapshots.get(&page)? {
            StoredKvHostSnapshot::F32(_) => None,
            StoredKvHostSnapshot::Typed(snapshot) => Some(snapshot),
        }
    }

    pub fn has_snapshot(&self, page: KvPageId) -> bool {
        self.snapshots.contains_key(&page)
    }

    /// Sorted packed pairs `[runtime_page_id_bits, physical_slot, ...]`.
    pub fn compact_page_table(&self) -> Vec<i32> {
        self.metadata.compact_table()
    }

    pub fn upload_page_table(&self, context: &CudaOperators) -> Result<CudaI32Buffer> {
        context.upload_i32_buffer(&self.metadata.compact_table())
    }

    pub fn stats(&self) -> KvPagePoolStats {
        let allocated_slots = self.layout.max_slots;
        let resident_pages = self.metadata.mappings.len();
        let pending_pages = self
            .pending
            .values()
            .map(|reservation| reservation.pages.len())
            .sum();
        let free_slots = self.metadata.free_slots.len();
        let largest_run = largest_contiguous_run(&self.metadata.free_slots);
        let external_fragmentation = if free_slots == 0 {
            0.0
        } else {
            1.0 - largest_run as f64 / free_slots as f64
        };
        KvPagePoolStats {
            allocated_slots,
            resident_pages,
            pending_pages,
            free_slots,
            allocated_bytes: allocated_slots.saturating_mul(self.layout.bytes_per_slot),
            resident_bytes: resident_pages.saturating_mul(self.layout.bytes_per_slot),
            utilization: resident_pages as f64 / allocated_slots as f64,
            external_fragmentation,
        }
    }

    fn allocate_slot(&mut self, context: &CudaOperators) -> Result<u32> {
        let slot = self
            .metadata
            .free_slots
            .pop()
            .ok_or_else(|| pool_error("physical KV page pool is exhausted"))?;
        for (plane, storage) in self.plane_storage.iter_mut().enumerate() {
            let range = self
                .layout
                .slot_range(slot, plane)
                .expect("free slot and plane are in range");
            let cleared = match storage {
                CudaKvPlaneStorage::F32(storage) => {
                    context.zero_f32_range(storage, range.start, range.len())
                }
                CudaKvPlaneStorage::Bf16(storage) => {
                    zero_bf16_range(context, storage, range.start, range.len())
                }
            };
            if let Err(error) = cleared {
                self.metadata.free_slots.push(slot);
                return Err(error);
            }
        }
        Ok(slot)
    }

    fn recycle_allocated(&mut self, allocated: &[(KvPageId, u32)]) {
        self.metadata
            .free_slots
            .extend(allocated.iter().map(|(_, slot)| *slot));
    }

    fn copy_slot(&mut self, context: &CudaOperators, source: u32, destination: u32) -> Result<()> {
        if source == destination {
            return Ok(());
        }
        for (plane, storage) in self.plane_storage.iter_mut().enumerate() {
            let source_range = self
                .layout
                .slot_range(source, plane)
                .expect("source slot and plane are in range");
            let destination_range = self
                .layout
                .slot_range(destination, plane)
                .expect("destination slot and plane are in range");
            match storage {
                CudaKvPlaneStorage::F32(storage) => context.copy_f32_within(
                    storage,
                    source_range.start,
                    destination_range.start,
                    source_range.len(),
                )?,
                CudaKvPlaneStorage::Bf16(storage) => copy_bf16_within(
                    context,
                    storage,
                    source_range.start,
                    destination_range.start,
                    source_range.len(),
                )?,
            }
        }
        Ok(())
    }
}

fn download_bf16_words_range(
    context: &CudaOperators,
    buffer: &CudaBf16Buffer,
    offset: usize,
    len: usize,
) -> Result<Vec<u16>> {
    let range = buffer.as_device_buffer().slice(offset, len)?;
    Ok(range.to_host_vec(&context.stream_clone())?)
}

fn overwrite_bf16_words_range(
    context: &CudaOperators,
    buffer: &CudaBf16Buffer,
    offset: usize,
    values: &[u16],
) -> Result<()> {
    let range = buffer.as_device_buffer().slice(offset, values.len())?;
    let stream = context.stream_clone();
    range.copy_from_host(&stream, values)?;
    stream.synchronize()?;
    Ok(())
}

fn zero_bf16_range(
    context: &CudaOperators,
    buffer: &CudaBf16Buffer,
    offset: usize,
    len: usize,
) -> Result<()> {
    overwrite_bf16_words_range(context, buffer, offset, &vec![0; len])
}

fn copy_bf16_within(
    context: &CudaOperators,
    buffer: &CudaBf16Buffer,
    source_offset: usize,
    destination_offset: usize,
    len: usize,
) -> Result<()> {
    let source_end = source_offset
        .checked_add(len)
        .ok_or_else(|| pool_error("BF16 copy source range overflow"))?;
    let destination_end = destination_offset
        .checked_add(len)
        .ok_or_else(|| pool_error("BF16 copy destination range overflow"))?;
    if source_end > buffer.len() || destination_end > buffer.len() {
        return Err(pool_error("BF16 copy range exceeds plane storage"));
    }
    if len != 0 && source_offset < destination_end && destination_offset < source_end {
        return Err(pool_error("BF16 copy ranges must not overlap"));
    }
    let source = buffer.as_device_buffer().slice(source_offset, len)?;
    let destination = buffer.as_device_buffer().slice(destination_offset, len)?;
    runtime::copy_device_to_device(
        &context.stream_clone(),
        destination.cu_deviceptr(),
        source.cu_deviceptr(),
        source.num_bytes(),
    )?;
    Ok(())
}

fn largest_contiguous_run(slots: &[u32]) -> usize {
    if slots.is_empty() {
        return 0;
    }
    let mut sorted = slots.to_vec();
    sorted.sort_unstable();
    let mut largest = 1;
    let mut current = 1;
    for pair in sorted.windows(2) {
        if pair[1] == pair[0] + 1 {
            current += 1;
            largest = largest.max(current);
        } else if pair[1] != pair[0] {
            current = 1;
        }
    }
    largest
}

fn distinct_pair_mut<T>(values: &mut [T], first: usize, second: usize) -> Result<(&mut T, &mut T)> {
    if first == second {
        return Err(pool_error(format!(
            "plane indices must be distinct, got {first} twice"
        )));
    }
    if first >= values.len() || second >= values.len() {
        return Err(pool_error(format!(
            "plane pair indices are out of range: first={first} second={second} planes={}",
            values.len()
        )));
    }
    if first < second {
        let (before_second, from_second) = values.split_at_mut(second);
        Ok((&mut before_second[first], &mut from_second[0]))
    } else {
        let (before_first, from_first) = values.split_at_mut(first);
        Ok((&mut from_first[0], &mut before_first[second]))
    }
}

fn pool_error(message: impl Into<String>) -> Error {
    Error::Execution {
        message: format!("CUDA KV page pool: {}", message.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paged_plane_row_mapping_crosses_page_at_nonzero_layer() {
        let layout = PagedPlaneLayout {
            page_tokens: 2,
            elements_per_token: 4,
            layer_index: 1,
            layer_count: 3,
        };
        let slots = [2, 5, -1];
        let offsets = [0, 2, 3];
        assert_eq!(
            layout.resolve_row_offset(0, 1, 144, &slots, &offsets),
            Some(60)
        );
        assert_eq!(
            layout.resolve_row_offset(0, 2, 144, &slots, &offsets),
            Some(128)
        );
        assert_eq!(layout.resolve_row_offset(1, 0, 144, &slots, &offsets), None);
        assert_eq!(layout.resolve_row_offset(0, 4, 144, &slots, &offsets), None);
    }

    #[test]
    fn paged_plane_scatter_addresses_cover_packed_sequences_and_bounds() {
        let layout = PagedPlaneLayout {
            page_tokens: 2,
            elements_per_token: 3,
            layer_index: 1,
            layer_count: 2,
        };
        let slots = [2, 0, 1, 0];
        let offsets = [0, 1, 3, 4];

        assert_eq!(
            layout.resolve_row_offset(0, 1, 36, &slots, &offsets),
            Some(33)
        );
        assert_eq!(
            layout.resolve_row_offset(1, 2, 36, &slots, &offsets),
            Some(18)
        );
        assert_eq!(
            layout.resolve_row_offset(2, 0, 36, &slots, &offsets),
            Some(6)
        );
        assert_eq!(layout.resolve_row_offset(0, 2, 36, &slots, &offsets), None);
        assert_eq!(layout.resolve_row_offset(3, 0, 36, &slots, &offsets), None);
        assert_eq!(layout.resolve_row_offset(0, 1, 35, &slots, &offsets), None);
        assert_eq!(layout.resolve_row_offset(0, 0, 36, &[-1], &[0, 1]), None);
        assert_eq!(layout.resolve_row_offset(0, 0, 36, &slots, &[-1, 1]), None);
    }

    #[test]
    fn selected_rows_can_share_and_reorder_sequence_block_ranges() {
        let layout = PagedPlaneLayout {
            page_tokens: 2,
            elements_per_token: 3,
            layer_index: 1,
            layer_count: 2,
        };
        let slots = [2, 0, 1, 0];
        let offsets = [0, 1, 3, 4];
        let row_sequence_ids = [1, 0, 1, 2];

        assert_eq!(
            layout.resolve_selected_row_offset(0, 2, 36, &slots, &offsets, &row_sequence_ids,),
            Some(18)
        );
        assert_eq!(
            layout.resolve_selected_row_offset(1, 1, 36, &slots, &offsets, &row_sequence_ids,),
            Some(33)
        );
        assert_eq!(
            layout.resolve_selected_row_offset(2, 0, 36, &slots, &offsets, &row_sequence_ids,),
            Some(6)
        );
        assert_eq!(
            layout.resolve_selected_row_offset(3, 0, 36, &slots, &offsets, &row_sequence_ids,),
            Some(6)
        );
        assert_eq!(
            layout.resolve_selected_row_offset(4, 0, 36, &slots, &offsets, &row_sequence_ids),
            None
        );
    }

    #[test]
    fn compressed_boundaries_resolve_in_token_scaled_pages() {
        let layout = PagedPlaneLayout {
            page_tokens: 16,
            elements_per_token: 1,
            layer_index: 0,
            layer_count: 1,
        };
        let slots = (0..16).rev().collect::<Vec<i32>>();
        let offsets = [0, 16];
        let storage_elements = 16 * 16;

        let ratio4_boundary = 5 * 4 - 1;
        let ratio128_boundary = 2 * 128 - 1;
        assert_eq!(ratio4_boundary / layout.page_tokens, 1);
        assert_eq!(ratio128_boundary / layout.page_tokens, 15);
        assert_eq!(
            layout.resolve_row_offset(0, ratio4_boundary, storage_elements, &slots, &offsets,),
            Some(14 * 16 + 3)
        );
        assert_eq!(
            layout.resolve_row_offset(0, ratio128_boundary, storage_elements, &slots, &offsets,),
            Some(15)
        );
        assert_ne!(
            layout.resolve_row_offset(0, 4, storage_elements, &slots, &offsets),
            layout.resolve_row_offset(0, ratio4_boundary, storage_elements, &slots, &offsets,)
        );
    }

    #[test]
    fn distinct_pair_mut_preserves_requested_order_without_aliasing() {
        let mut values = [10, 20, 30];
        {
            let (first, second) = distinct_pair_mut(&mut values, 2, 0).unwrap();
            *first = 31;
            *second = 11;
        }
        assert_eq!(values, [11, 20, 31]);
        assert!(distinct_pair_mut(&mut values, 1, 1).is_err());
        assert!(distinct_pair_mut(&mut values, 0, 3).is_err());
    }

    #[test]
    fn layout_and_metadata_are_cuda_independent() {
        let planes = [
            KvPlaneDescriptor::new("a", 8, 2, KvElementType::F32),
            KvPlaneDescriptor::new("b", 3, 1, KvElementType::F32),
        ];
        let layout = PageLayout::new(&planes, 16, 4).unwrap();
        assert_eq!(&*layout.page_elements, &[256, 48]);
        assert_eq!(&*layout.page_bytes, &[1024, 192]);
        assert_eq!(layout.bytes_per_slot, 1216);
        assert_eq!(layout.slot_range(2, 0), Some(512..768));
        assert_eq!(layout.slot_byte_range(2, 0), Some(2048..3072));
        assert!(PageLayout::new(&planes, 16, 0).is_err());

        let mut metadata = MetadataState::with_slots(4);
        assert_eq!(metadata.free_slots.pop(), Some(0));
        metadata.commit(&[(KvPageId(7), 0), (KvPageId(9), 3)]);
        assert_eq!(
            metadata
                .physical_slots(&[KvPageId(9), KvPageId(7), KvPageId(9)])
                .unwrap(),
            [3, 0, 3]
        );
        assert_eq!(
            metadata
                .physical_slots_for_block_ids(&[
                    KvBlockId::new(9),
                    KvBlockId::new(7),
                    KvBlockId::new(9),
                ])
                .unwrap(),
            [3, 0, 3]
        );
        assert!(metadata.physical_slots(&[KvPageId(8)]).is_err());
        assert!(
            metadata
                .physical_slots_for_block_ids(&[KvBlockId::new(8)])
                .is_err()
        );
        assert_eq!(metadata.release(KvPageId(7)), Some(0));
        assert_eq!(metadata.free_slots.pop(), Some(0));
    }

    #[test]
    fn typed_layout_uses_physical_plane_widths_and_byte_ranges() {
        let planes = [
            KvPlaneDescriptor::new("f32", 2, 2, KvElementType::F32),
            KvPlaneDescriptor::new("bf16", 3, 1, KvElementType::Bf16),
        ];
        let layout = PageLayout::new(&planes, 4, 3).unwrap();
        assert_eq!(&*layout.page_elements, &[16, 12]);
        assert_eq!(&*layout.page_bytes, &[64, 24]);
        assert_eq!(layout.bytes_per_slot, 88);
        assert_eq!(layout.slot_range(2, 1), Some(24..36));
        assert_eq!(layout.slot_byte_range(2, 1), Some(48..72));

        let legacy = PagedPlaneLayout {
            page_tokens: 4,
            elements_per_token: 3,
            layer_index: 0,
            layer_count: 1,
        };
        assert_eq!(legacy.element_type(), KvElementType::F32);
        assert_eq!(legacy.checked_elements_per_page(), Some(12));
        assert_eq!(legacy.checked_bytes_per_page(), Some(48));

        let typed = legacy.typed(KvElementType::Bf16);
        assert_eq!(typed.checked_elements_per_page(), Some(12));
        assert_eq!(typed.checked_bytes_per_page(), Some(24));
        assert_eq!(
            typed.resolve_row_byte_range(0, 3, 24, &[0], &[0, 1]),
            Some(18..24)
        );
        assert_eq!(typed.resolve_row_byte_range(0, 3, 23, &[0], &[0, 1]), None);
    }

    #[test]
    fn typed_layout_rejects_element_and_byte_overflow_without_cuda() {
        let overflow = [KvPlaneDescriptor::new(
            "overflow",
            usize::MAX,
            2,
            KvElementType::Bf16,
        )];
        assert!(PageLayout::new(&overflow, 2, 1).is_err());

        let layout = PagedPlaneLayout {
            page_tokens: usize::MAX,
            elements_per_token: 2,
            layer_index: 0,
            layer_count: 1,
        };
        assert_eq!(layout.checked_elements_per_page(), None);
        assert_eq!(
            layout.typed(KvElementType::Bf16).checked_bytes_per_page(),
            None
        );
    }

    #[test]
    fn typed_host_snapshot_exposes_raw_physical_payloads() {
        let f32 = KvHostPlaneSnapshot::F32(vec![0x8000_0000, 0x7fc1_2345]);
        let bf16 = KvHostPlaneSnapshot::Bf16(vec![0x8000, 0x7fc1]);
        assert_eq!(f32.element_type(), KvElementType::F32);
        assert_eq!(bf16.element_type(), KvElementType::Bf16);
        assert_eq!(f32.byte_len(), 8);
        assert_eq!(bf16.byte_len(), 4);
        assert_eq!(f32.f32_bits(), Some(&[0x8000_0000, 0x7fc1_2345][..]));
        assert_eq!(bf16.bf16_words(), Some(&[0x8000, 0x7fc1][..]));
        assert_eq!(f32.bytes().len(), 8);
        assert_eq!(bf16.bytes().len(), 4);
    }

    #[test]
    fn uncommitted_and_rolled_back_pages_do_not_change_old_mapping() {
        let old = KvPageId(7);
        let pending = KvPageId(11);
        let mut metadata = MetadataState::with_slots(2);
        let old_slot = metadata.free_slots.pop().unwrap();
        metadata.commit(&[(old, old_slot)]);
        let pending_slot = metadata.free_slots.pop().unwrap();

        assert_eq!(metadata.mappings.get(&old), Some(&old_slot));
        assert!(!metadata.mappings.contains_key(&pending));
        metadata.free_slots.push(pending_slot);
        assert_eq!(metadata.mappings.get(&old), Some(&old_slot));
    }

    #[test]
    fn compact_table_and_fragmentation_are_deterministic() {
        let mut metadata = MetadataState::default();
        metadata.commit(&[(KvPageId(u32::MAX), 2), (KvPageId(3), 9)]);
        assert_eq!(metadata.compact_table(), vec![3, 9, -1, 2]);
        assert_eq!(largest_contiguous_run(&[]), 0);
        assert_eq!(largest_contiguous_run(&[7, 2, 3, 4, 9]), 3);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_paged_plane_scatter_rows_matches_cpu_addresses_and_mask() {
        let context = CudaOperators::new().unwrap();
        let layout = PagedPlaneLayout {
            page_tokens: 2,
            elements_per_token: 3,
            layer_index: 1,
            layer_count: 2,
        };
        let values = context
            .upload_f32_buffer(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let positions = context.upload_i32_buffer(&[1, 2, 0]).unwrap();
        let block_slots = context.upload_i32_buffer(&[2, 0, 1, 0]).unwrap();
        let block_offsets = context.upload_i32_buffer(&[0, 1, 3, 4]).unwrap();
        let mask = context.upload_i32_buffer(&[1, 1, 0]).unwrap();
        let mut plane = context.zero_f32_buffer(36).unwrap();

        context
            .paged_plane_scatter_rows_from_device(
                &values,
                &positions,
                &block_slots,
                &block_offsets,
                Some(&mask),
                &mut plane,
                layout,
            )
            .unwrap();

        let actual = context.download_f32_buffer(&plane).unwrap();
        let mut expected = vec![0.0; 36];
        expected[33..36].copy_from_slice(&[1.0, 2.0, 3.0]);
        expected[18..21].copy_from_slice(&[4.0, 5.0, 6.0]);
        assert_eq!(actual, expected);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_paged_plane_scatter_selected_rows_matches_cpu_addresses() {
        let context = CudaOperators::new().unwrap();
        let layout = PagedPlaneLayout {
            page_tokens: 2,
            elements_per_token: 3,
            layer_index: 1,
            layer_count: 2,
        };
        let values = context
            .upload_f32_buffer(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let positions = context.upload_i32_buffer(&[2, 1, 0]).unwrap();
        let row_sequence_ids = context.upload_i32_buffer(&[1, 0, 1]).unwrap();
        let block_slots = context.upload_i32_buffer(&[2, 0, 1, 0]).unwrap();
        let block_offsets = context.upload_i32_buffer(&[0, 1, 3, 4]).unwrap();
        let mut plane = context.zero_f32_buffer(36).unwrap();

        context
            .paged_plane_scatter_selected_rows_from_device(
                &values,
                &positions,
                &block_slots,
                &block_offsets,
                &row_sequence_ids,
                None,
                &mut plane,
                layout,
            )
            .unwrap();

        let actual = context.download_f32_buffer(&plane).unwrap();
        let mut expected = vec![0.0; 36];
        expected[18..21].copy_from_slice(&[1.0, 2.0, 3.0]);
        expected[33..36].copy_from_slice(&[4.0, 5.0, 6.0]);
        expected[6..9].copy_from_slice(&[7.0, 8.0, 9.0]);
        assert_eq!(actual, expected);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_paged_indexer_decode_matches_cpu_reference_across_page_boundary() {
        let context = CudaOperators::new().unwrap();
        let descriptor = KvPlaneDescriptor::new("paged_indexer_test", 4, 2, KvElementType::F32);
        let mut pool = CudaKvPagePool::new(&context, &[descriptor], 2, 2).unwrap();
        let first = KvPageId(20);
        let second = KvPageId(21);
        pool.ensure(&context, first).unwrap();
        pool.ensure(&context, second).unwrap();
        let rows: Vec<f32> = (0..16).map(|index| index as f32 * 0.125 - 0.75).collect();
        for (page, values) in [(first, &rows[..8]), (second, &rows[8..])] {
            let range = pool.plane_slot_range(page, 0).unwrap();
            context
                .overwrite_f32_range(values, pool.plane_storage_mut(0).unwrap(), range.start + 8)
                .unwrap();
        }
        let query = context.upload_f32_buffer(&[0.25, -0.5, 0.75, 1.0]).unwrap();
        let weights = context.upload_f32_buffer(&[1.0]).unwrap();
        let query_values = [0.25, -0.5, 0.75, 1.0];
        let mut candidates = rows
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let score = row
                    .iter()
                    .zip(query_values)
                    .map(|(value, query)| value * query)
                    .sum::<f32>()
                    .max(0.0);
                (index, score)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|(left_index, left_score), (right_index, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_index.cmp(right_index))
        });
        let mut expected = vec![0, 1, 2, 3];
        expected.extend(
            candidates
                .into_iter()
                .take(2)
                .map(|(index, _)| 4 + index as i32),
        );
        let block_slots = context
            .upload_i32_buffer(&[
                pool.physical_slot(first).unwrap() as i32,
                pool.physical_slot(second).unwrap() as i32,
            ])
            .unwrap();
        let block_offsets = context.upload_i32_buffer(&[0, 2]).unwrap();
        let mut actual = context.zero_i32_buffer(6).unwrap();
        context
            .dsv4_decode_topk_indices_paged_indexer_from_device_into(
                &query,
                &weights,
                pool.plane_storage(0).unwrap(),
                &block_slots,
                &block_offsets,
                3,
                4,
                4,
                2,
                4,
                4,
                1,
                4,
                2,
                1,
                2,
                1.0,
                &mut actual,
            )
            .unwrap();
        assert_eq!(context.download_i32_buffer(&actual).unwrap(), expected);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_compressed_boundary_scatter_isolates_cow_token_pages() {
        let context = CudaOperators::new().unwrap();
        let descriptor =
            KvPlaneDescriptor::new("compressed_boundary_test", 1, 1, KvElementType::F32);
        let mut pool = CudaKvPagePool::new(&context, &[descriptor], 16, 5).unwrap();
        let filler = KvPageId(10);
        let ratio4_source = KvPageId(11);
        let ratio128_source = KvPageId(12);
        let ratio4_branch = KvPageId(13);
        let ratio128_branch = KvPageId(14);
        for page in [filler, ratio4_source, ratio128_source] {
            pool.ensure(&context, page).unwrap();
        }
        for (page, sentinel) in [(ratio4_source, 4.0f32), (ratio128_source, 128.0)] {
            let offset = pool.page_element_offset(page, 0).unwrap();
            context
                .overwrite_f32_range(&[sentinel; 16], pool.plane_storage_mut(0).unwrap(), offset)
                .unwrap();
        }
        for (source, replacement) in [
            (ratio4_source, ratio4_branch),
            (ratio128_source, ratio128_branch),
        ] {
            let reservation = pool
                .reserve(
                    &context,
                    &[],
                    Some(KvCowReplacement {
                        logical_page: 0,
                        source,
                        replacement,
                    }),
                )
                .unwrap();
            pool.commit(reservation).unwrap();
        }

        let filler_slot = pool.physical_slot(filler).unwrap() as i32;
        let mut block_slots = vec![
            filler_slot,
            pool.physical_slot(ratio4_branch).unwrap() as i32,
        ];
        block_slots.extend(vec![filler_slot; 15]);
        block_slots.push(pool.physical_slot(ratio128_branch).unwrap() as i32);
        let block_slots = context.upload_i32_buffer(&block_slots).unwrap();
        let block_offsets = context.upload_i32_buffer(&[0, 2, 18]).unwrap();
        let row_sequence_ids = context.upload_i32_buffer(&[0, 1]).unwrap();
        let positions = context.upload_i32_buffer(&[19, 255]).unwrap();
        let values = context.upload_f32_buffer(&[19.0, 255.0]).unwrap();
        context
            .paged_plane_scatter_selected_rows_from_device(
                &values,
                &positions,
                &block_slots,
                &block_offsets,
                &row_sequence_ids,
                None,
                pool.plane_storage_mut(0).unwrap(),
                PagedPlaneLayout {
                    page_tokens: 16,
                    elements_per_token: 1,
                    layer_index: 0,
                    layer_count: 1,
                },
            )
            .unwrap();

        let download = |pool: &CudaKvPagePool, page| {
            let range = pool.plane_slot_range(page, 0).unwrap();
            context
                .download_f32_range(pool.plane_storage(0).unwrap(), range.start, range.len())
                .unwrap()
        };
        assert_eq!(download(&pool, ratio4_source), vec![4.0; 16]);
        assert_eq!(download(&pool, ratio128_source), vec![128.0; 16]);
        let mut ratio4_expected = vec![4.0; 16];
        ratio4_expected[3] = 19.0;
        assert_eq!(download(&pool, ratio4_branch), ratio4_expected);
        let mut ratio128_expected = vec![128.0; 16];
        ratio128_expected[15] = 255.0;
        assert_eq!(download(&pool, ratio128_branch), ratio128_expected);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_mixed_f32_bf16_scatter_cow_rollback_preempt_restore() {
        fn download_f32_bits(
            context: &CudaOperators,
            pool: &CudaKvPagePool,
            page: KvPageId,
        ) -> Vec<u32> {
            let range = pool.plane_slot_range(page, 0).unwrap();
            context
                .download_f32_range(pool.plane_storage(0).unwrap(), range.start, range.len())
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        }

        fn download_bf16_words(
            context: &CudaOperators,
            pool: &CudaKvPagePool,
            page: KvPageId,
        ) -> Vec<u16> {
            let range = pool.plane_slot_range(page, 1).unwrap();
            pool.download_bf16_words_range(context, 1, range.start, range.len())
                .unwrap()
        }

        fn upload_bf16_words(context: &CudaOperators, words: &[u16]) -> CudaBf16Buffer {
            let buffer = context.zero_bf16_buffer(words.len()).unwrap();
            let stream = context.stream_clone();
            buffer
                .as_device_buffer()
                .copy_from_host(&stream, words)
                .unwrap();
            stream.synchronize().unwrap();
            buffer
        }

        let context = CudaOperators::new().unwrap();
        let planes = [
            KvPlaneDescriptor::new("f32", 1, 1, KvElementType::F32),
            KvPlaneDescriptor::new("bf16", 2, 1, KvElementType::Bf16),
        ];
        let mut pool = CudaKvPagePool::new(&context, &planes, 2, 4).unwrap();
        assert_eq!(pool.planes(), &planes);
        assert!(pool.plane_storage(0).is_some());
        assert!(pool.plane_storage(1).is_none());
        assert!(pool.plane_bf16_storage(0).is_none());
        assert!(pool.plane_bf16_storage(1).is_some());
        assert_eq!(pool.page_bytes(0), Some(8));
        assert_eq!(pool.page_bytes(1), Some(8));
        assert_eq!(pool.bytes_per_slot(), 16);
        assert_eq!(pool.plane_storage_bytes(0), Some(32));
        assert_eq!(pool.plane_storage_bytes(1), Some(32));

        let source = KvPageId(1);
        let replacement = KvPageId(2);
        let rolled_back = KvPageId(3);
        let fresh = KvPageId(4);
        pool.ensure(&context, source).unwrap();

        let f32_bits = [0x8000_0000, 0x7fc1_2345];
        let f32_values = f32_bits.map(f32::from_bits);
        let f32_range = pool.plane_slot_range(source, 0).unwrap();
        context
            .overwrite_f32_range(
                &f32_values,
                pool.plane_storage_mut(0).unwrap(),
                f32_range.start,
            )
            .unwrap();

        let source_slot = pool.physical_slot(source).unwrap() as i32;
        let source_words = [0x3f80, 0x8000, 0x7fc1, 0xdead];
        let source_words_device = upload_bf16_words(&context, &source_words);
        pool.paged_plane_scatter_bf16_selected_rows_from_device(
            &context,
            1,
            &source_words_device,
            &[0, 1],
            &[source_slot],
            &[0, 1],
            &[0, 0],
            None,
            PagedPlaneLayout {
                page_tokens: 2,
                elements_per_token: 2,
                layer_index: 0,
                layer_count: 1,
            }
            .typed(KvElementType::Bf16),
        )
        .unwrap();
        assert_eq!(download_bf16_words(&context, &pool, source), source_words);

        let cow = pool
            .reserve(
                &context,
                &[],
                Some(KvCowReplacement {
                    logical_page: 0,
                    source,
                    replacement,
                }),
            )
            .unwrap();
        pool.commit(cow).unwrap();
        assert_eq!(download_f32_bits(&context, &pool, replacement), f32_bits);
        assert_eq!(
            download_bf16_words(&context, &pool, replacement),
            source_words
        );

        let replacement_slot = pool.physical_slot(replacement).unwrap() as i32;
        let branch_tail = [0x4000, 0x4040];
        let branch_tail_device = upload_bf16_words(&context, &branch_tail);
        pool.paged_plane_scatter_bf16_rows_from_device(
            &context,
            1,
            &branch_tail_device,
            &[1],
            &[replacement_slot],
            &[0, 1],
            None,
            PagedPlaneLayout {
                page_tokens: 2,
                elements_per_token: 2,
                layer_index: 0,
                layer_count: 1,
            }
            .typed(KvElementType::Bf16),
        )
        .unwrap();
        assert_eq!(download_bf16_words(&context, &pool, source), source_words);
        let branch_words = [source_words[0], source_words[1], 0x4000, 0x4040];
        assert_eq!(
            download_bf16_words(&context, &pool, replacement),
            branch_words
        );
        assert_eq!(download_f32_bits(&context, &pool, replacement), f32_bits);

        let old_mapping = pool.compact_page_table();
        let rollback = pool
            .reserve(
                &context,
                &[],
                Some(KvCowReplacement {
                    logical_page: 0,
                    source: replacement,
                    replacement: rolled_back,
                }),
            )
            .unwrap();
        let pending = pool
            .pending_plane_slot_range(&rollback, rolled_back, 1)
            .unwrap();
        pool.overwrite_bf16_words_range(
            &context,
            1,
            pending.start,
            &[0xffff, 0xffff, 0xffff, 0xffff],
        )
        .unwrap();
        pool.rollback(&context, rollback).unwrap();
        assert_eq!(pool.compact_page_table(), old_mapping);
        assert_eq!(pool.physical_slot(rolled_back), None);
        assert_eq!(
            download_bf16_words(&context, &pool, replacement),
            branch_words
        );

        let stats = pool.stats();
        assert_eq!(stats.allocated_bytes, 64);
        assert_eq!(stats.resident_bytes, 32);
        let snapshots = pool.preempt_typed(&context, &[replacement]).unwrap();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].planes.len(), 2);
        assert_eq!(snapshots[0].planes[0].f32_bits(), Some(&f32_bits[..]));
        assert_eq!(snapshots[0].planes[1].bf16_words(), Some(&branch_words[..]));
        assert_eq!(pool.physical_slot(replacement), None);
        pool.restore(&context, &[replacement]).unwrap();
        assert_eq!(download_f32_bits(&context, &pool, replacement), f32_bits);
        assert_eq!(
            download_bf16_words(&context, &pool, replacement),
            branch_words
        );

        pool.release(&context, source).unwrap();
        pool.ensure(&context, fresh).unwrap();
        assert_eq!(download_f32_bits(&context, &pool, fresh), [0, 0]);
        assert_eq!(download_bf16_words(&context, &pool, fresh), [0, 0, 0, 0]);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_cow_touches_only_destination_and_rollback_preserves_old_mapping() {
        fn bits(values: &[f32]) -> Vec<u32> {
            values.iter().map(|value| value.to_bits()).collect()
        }

        fn download_page(
            context: &CudaOperators,
            pool: &CudaKvPagePool,
            page: KvPageId,
        ) -> Vec<f32> {
            let range = pool.plane_slot_range(page, 0).unwrap();
            context
                .download_f32_range(pool.plane_storage(0).unwrap(), range.start, range.len())
                .unwrap()
        }

        let context = CudaOperators::new().unwrap();
        let descriptor = KvPlaneDescriptor::new("test", 2, 1, KvElementType::F32);
        let mut pool = CudaKvPagePool::new(&context, &[descriptor], 2, 4).unwrap();
        assert_eq!(pool.plane_storage(0).unwrap().len(), 16);

        let source = KvPageId(1);
        let untouched = KvPageId(2);
        let replacement = KvPageId(3);
        let rolled_back = KvPageId(4);
        pool.ensure(&context, source).unwrap();
        pool.ensure(&context, untouched).unwrap();

        let source_pattern = vec![
            f32::from_bits(0x0000_0000),
            f32::from_bits(0x8000_0000),
            f32::from_bits(0x7fc1_2345),
            f32::from_bits(0xdead_beef),
        ];
        let untouched_pattern = vec![
            f32::from_bits(1),
            f32::from_bits(2),
            f32::from_bits(3),
            f32::from_bits(4),
        ];
        let source_offset = pool.page_element_offset(source, 0).unwrap();
        context
            .overwrite_f32_range(
                &source_pattern,
                pool.plane_storage_mut(0).unwrap(),
                source_offset,
            )
            .unwrap();
        let untouched_offset = pool.page_element_offset(untouched, 0).unwrap();
        context
            .overwrite_f32_range(
                &untouched_pattern,
                pool.plane_storage_mut(0).unwrap(),
                untouched_offset,
            )
            .unwrap();

        let reservation = pool
            .reserve(
                &context,
                &[],
                Some(KvCowReplacement {
                    logical_page: 0,
                    source,
                    replacement,
                }),
            )
            .unwrap();
        assert_eq!(pool.physical_slot(replacement), None);
        assert!(pool.pending_replacement_slot(&reservation).is_some());
        pool.commit(reservation).unwrap();
        assert_eq!(
            bits(&download_page(&context, &pool, replacement)),
            bits(&source_pattern)
        );
        assert_eq!(
            bits(&download_page(&context, &pool, untouched)),
            bits(&untouched_pattern)
        );

        let old_mapping = pool.compact_page_table();
        let rollback = pool
            .reserve(
                &context,
                &[],
                Some(KvCowReplacement {
                    logical_page: 0,
                    source: replacement,
                    replacement: rolled_back,
                }),
            )
            .unwrap();
        pool.rollback(&context, rollback).unwrap();
        assert_eq!(pool.compact_page_table(), old_mapping);
        assert_eq!(pool.physical_slot(rolled_back), None);
        assert_eq!(
            bits(&download_page(&context, &pool, replacement)),
            bits(&source_pattern)
        );

        pool.preempt(&context, &[replacement]).unwrap();
        pool.restore(&context, &[replacement]).unwrap();
        assert_eq!(
            bits(&download_page(&context, &pool, replacement)),
            bits(&source_pattern)
        );
        assert_eq!(
            bits(&download_page(&context, &pool, untouched)),
            bits(&untouched_pattern)
        );
    }
}
