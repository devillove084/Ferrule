//! Model-independent, multi-plane physical KV page storage.
//!
//! Every plane owns one fixed-capacity device buffer. Runtime page IDs map to
//! fixed-size slot ranges inside those buffers. Reservations remain invisible
//! until commit, so allocation or copy failures preserve the previous view.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use ferrule_common::execution::{KvCowReplacement, KvPageId, KvPlaneDescriptor};
use ferrule_common::{Error, Result};

use crate::context::{CudaArtifactOperatorContext, CudaF32Buffer, CudaI32Buffer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedPlaneLayout {
    pub page_tokens: usize,
    pub elements_per_token: usize,
    pub layer_index: usize,
    pub layer_count: usize,
}

impl PagedPlaneLayout {
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
    page_tokens: usize,
    elements_per_slot: usize,
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
        let page_elements: Vec<usize> = planes
            .iter()
            .map(|plane| {
                page_tokens
                    .checked_mul(plane.elements_per_token)
                    .and_then(|elements| elements.checked_mul(plane.layer_count))
                    .ok_or_else(|| pool_error("plane page element count overflow"))
            })
            .collect::<Result<_>>()?;
        if page_elements.contains(&0) {
            return Err(pool_error("plane pages must not be empty"));
        }
        let elements_per_slot = page_elements
            .iter()
            .try_fold(0usize, |sum, elements| sum.checked_add(*elements))
            .ok_or_else(|| pool_error("physical slot element count overflow"))?;
        for elements in &page_elements {
            elements
                .checked_mul(max_slots)
                .ok_or_else(|| pool_error("plane pool element count overflow"))?;
        }
        Ok(Self {
            page_elements: page_elements.into_boxed_slice(),
            page_tokens,
            elements_per_slot,
            max_slots,
        })
    }

    fn slot_range(&self, slot: u32, plane: usize) -> Option<Range<usize>> {
        let page_elements = *self.page_elements.get(plane)?;
        let start = (slot as usize).checked_mul(page_elements)?;
        Some(start..start.checked_add(page_elements)?)
    }
}

/// Fixed-capacity CUDA KV page pool with one contiguous buffer per plane.
pub struct CudaKvPagePool {
    planes: Box<[KvPlaneDescriptor]>,
    layout: PageLayout,
    plane_storage: Vec<CudaF32Buffer>,
    metadata: MetadataState,
    pending: HashMap<u64, PendingReservation>,
    snapshots: HashMap<KvPageId, KvHostSnapshot>,
    next_token: u64,
}

impl CudaKvPagePool {
    /// Allocates exactly one fixed-capacity device buffer for every plane.
    pub fn new(
        context: &CudaArtifactOperatorContext,
        planes: &[KvPlaneDescriptor],
        page_tokens: usize,
        max_slots: usize,
    ) -> Result<Self> {
        let layout = PageLayout::new(planes, page_tokens, max_slots)?;
        let plane_storage = layout
            .page_elements
            .iter()
            .map(|elements| {
                let capacity = elements
                    .checked_mul(max_slots)
                    .ok_or_else(|| pool_error("plane pool element count overflow"))?;
                context.zero_f32_buffer(capacity)
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

    pub fn page_tokens(&self) -> usize {
        self.layout.page_tokens
    }

    /// Returns the complete contiguous storage for one plane.
    pub fn plane_storage(&self, plane: usize) -> Option<&CudaF32Buffer> {
        self.plane_storage.get(plane)
    }

    pub fn plane_storage_mut(&mut self, plane: usize) -> Option<&mut CudaF32Buffer> {
        self.plane_storage.get_mut(plane)
    }

    pub fn physical_slot(&self, page: KvPageId) -> Option<u32> {
        self.metadata.mappings.get(&page).copied()
    }

    pub fn page_element_offset(&self, page: KvPageId, plane: usize) -> Option<usize> {
        Some(self.plane_slot_range(page, plane)?.start)
    }

    pub fn plane_slot_range(&self, page: KvPageId, plane: usize) -> Option<Range<usize>> {
        self.layout.slot_range(self.physical_slot(page)?, plane)
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

    pub fn pending_replacement_slot(&self, reservation: &KvPoolReservation) -> Option<u32> {
        self.pending
            .get(&reservation.token)?
            .cow
            .map(|(_, slot)| slot)
    }

    pub fn ensure(&mut self, context: &CudaArtifactOperatorContext, page: KvPageId) -> Result<u32> {
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
        context: &CudaArtifactOperatorContext,
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
        _context: &CudaArtifactOperatorContext,
        reservation: KvPoolReservation,
    ) -> Result<()> {
        let pending = self
            .pending
            .remove(&reservation.token)
            .ok_or_else(|| pool_error("unknown or completed reservation"))?;
        self.recycle_allocated(&pending.pages);
        Ok(())
    }

    pub fn release(
        &mut self,
        _context: &CudaArtifactOperatorContext,
        page: KvPageId,
    ) -> Result<()> {
        if self.metadata.release(page).is_some() || self.snapshots.remove(&page).is_some() {
            return Ok(());
        }
        Err(pool_error(
            "cannot release an unknown physical or suspended page",
        ))
    }

    /// Downloads only the page ranges, then releases mappings after all copies succeed.
    pub fn preempt(
        &mut self,
        context: &CudaArtifactOperatorContext,
        pages: &[KvPageId],
    ) -> Result<Vec<KvHostSnapshot>> {
        let mut unique = HashSet::with_capacity(pages.len());
        let mut snapshots = Vec::with_capacity(pages.len());
        for page in pages {
            if !unique.insert(*page) {
                return Err(pool_error("preempt contains a duplicate page ID"));
            }
            let slot = self
                .physical_slot(*page)
                .ok_or_else(|| pool_error("cannot preempt a non-resident page"))?;
            let mut plane_snapshots = Vec::with_capacity(self.plane_storage.len());
            for plane in 0..self.plane_storage.len() {
                let range = self
                    .layout
                    .slot_range(slot, plane)
                    .expect("resident slot and plane are in range");
                plane_snapshots.push(context.download_f32_range(
                    &self.plane_storage[plane],
                    range.start,
                    range.len(),
                )?);
            }
            snapshots.push(KvHostSnapshot {
                page_id: *page,
                planes: plane_snapshots,
            });
        }
        for snapshot in &snapshots {
            self.metadata.release(snapshot.page_id);
            self.snapshots.insert(snapshot.page_id, snapshot.clone());
        }
        Ok(snapshots)
    }

    /// Uploads only reserved page ranges. Failed uploads remain invisible and roll back.
    pub fn restore(
        &mut self,
        context: &CudaArtifactOperatorContext,
        pages: &[KvPageId],
    ) -> Result<()> {
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
        for snapshot in &snapshots {
            let slot = self
                .pending_slot(&reservation, snapshot.page_id)
                .expect("restored page is pending");
            for (plane, values) in snapshot.planes.iter().enumerate() {
                let range = self
                    .layout
                    .slot_range(slot, plane)
                    .expect("pending slot and plane are in range");
                if values.len() != range.len() {
                    self.rollback(context, reservation)?;
                    return Err(pool_error("host snapshot plane length mismatch"));
                }
                if let Err(error) =
                    context.overwrite_f32_range(values, &mut self.plane_storage[plane], range.start)
                {
                    self.rollback(context, reservation)?;
                    return Err(error);
                }
            }
        }
        self.commit(reservation)?;
        for page in pages {
            self.snapshots.remove(page);
        }
        Ok(())
    }

    pub fn snapshot(&self, page: KvPageId) -> Option<&KvHostSnapshot> {
        self.snapshots.get(&page)
    }

    pub fn has_snapshot(&self, page: KvPageId) -> bool {
        self.snapshots.contains_key(&page)
    }

    /// Sorted packed pairs `[runtime_page_id_bits, physical_slot, ...]`.
    pub fn compact_page_table(&self) -> Vec<i32> {
        self.metadata.compact_table()
    }

    pub fn upload_page_table(
        &self,
        context: &CudaArtifactOperatorContext,
    ) -> Result<CudaI32Buffer> {
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
            allocated_bytes: allocated_slots
                .saturating_mul(self.layout.elements_per_slot)
                .saturating_mul(std::mem::size_of::<f32>()),
            resident_bytes: resident_pages
                .saturating_mul(self.layout.elements_per_slot)
                .saturating_mul(std::mem::size_of::<f32>()),
            utilization: resident_pages as f64 / allocated_slots as f64,
            external_fragmentation,
        }
    }

    fn allocate_slot(&mut self, context: &CudaArtifactOperatorContext) -> Result<u32> {
        let slot = self
            .metadata
            .free_slots
            .pop()
            .ok_or_else(|| pool_error("physical KV page pool is exhausted"))?;
        for plane in 0..self.plane_storage.len() {
            let range = self
                .layout
                .slot_range(slot, plane)
                .expect("free slot and plane are in range");
            if let Err(error) =
                context.zero_f32_range(&mut self.plane_storage[plane], range.start, range.len())
            {
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

    fn copy_slot(
        &mut self,
        context: &CudaArtifactOperatorContext,
        source: u32,
        destination: u32,
    ) -> Result<()> {
        if source == destination {
            return Ok(());
        }
        for plane in 0..self.plane_storage.len() {
            let source_range = self
                .layout
                .slot_range(source, plane)
                .expect("source slot and plane are in range");
            let destination_range = self
                .layout
                .slot_range(destination, plane)
                .expect("destination slot and plane are in range");
            context.copy_f32_within(
                &mut self.plane_storage[plane],
                source_range.start,
                destination_range.start,
                source_range.len(),
            )?;
        }
        Ok(())
    }
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

fn pool_error(message: impl Into<String>) -> Error {
    Error::Execution(format!("CUDA KV page pool: {}", message.into()))
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
    fn layout_and_metadata_are_cuda_independent() {
        let planes = [
            KvPlaneDescriptor {
                name: "a",
                elements_per_token: 8,
                layer_count: 2,
            },
            KvPlaneDescriptor {
                name: "b",
                elements_per_token: 3,
                layer_count: 1,
            },
        ];
        let layout = PageLayout::new(&planes, 16, 4).unwrap();
        assert_eq!(&*layout.page_elements, &[256, 48]);
        assert_eq!(layout.slot_range(2, 0), Some(512..768));
        assert!(PageLayout::new(&planes, 16, 0).is_err());

        let mut metadata = MetadataState::with_slots(4);
        assert_eq!(metadata.free_slots.pop(), Some(0));
        metadata.commit(&[(KvPageId(7), 0)]);
        assert_eq!(metadata.release(KvPageId(7)), Some(0));
        assert_eq!(metadata.free_slots.pop(), Some(0));
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
        let context = CudaArtifactOperatorContext::new().unwrap();
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
        let context = CudaArtifactOperatorContext::new().unwrap();
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
        let context = CudaArtifactOperatorContext::new().unwrap();
        let descriptor = KvPlaneDescriptor {
            name: "paged_indexer_test",
            elements_per_token: 4,
            layer_count: 2,
        };
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
            .chunks_exact(4)
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
    fn cuda_cow_touches_only_destination_and_rollback_preserves_old_mapping() {
        fn bits(values: &[f32]) -> Vec<u32> {
            values.iter().map(|value| value.to_bits()).collect()
        }

        fn download_page(
            context: &CudaArtifactOperatorContext,
            pool: &CudaKvPagePool,
            page: KvPageId,
        ) -> Vec<f32> {
            let range = pool.plane_slot_range(page, 0).unwrap();
            context
                .download_f32_range(pool.plane_storage(0).unwrap(), range.start, range.len())
                .unwrap()
        }

        let context = CudaArtifactOperatorContext::new().unwrap();
        let descriptor = KvPlaneDescriptor {
            name: "test",
            elements_per_token: 2,
            layer_count: 1,
        };
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
