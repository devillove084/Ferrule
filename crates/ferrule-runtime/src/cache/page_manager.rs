//! Runtime KV page manager.
//!
//! The page manager is the single owner of logical page allocation, refcount,
//! block tables, and reservation semantics. It is model-agnostic: it works
//! with any [`KvLayoutSchema`] supplied by the model.
//!
//! ## Lifecycle
//!
//! ```text
//! reserve -> execute -> commit
//!                  ↘ rollback on failure
//! ```
//!
//! Pages are reserved before model execution and committed only after success.
//! A failure or cancellation rolls back all newly allocated pages, restoring
//! the previous committed KV view.
//!
//! The backend (CUDA or reference) owns the physical device buffers. The page
//! manager owns logical allocation and metadata only.

use std::collections::{BTreeMap, HashMap};

use ferrule_common::execution::{
    KvBlockId, KvCowReplacement, KvLayoutSchema, KvPageId, KvReservation, KvWriteSlot, StateSlot,
};
use ferrule_common::{Error, Result};

/// Per-sequence block table mapping logical page indices to physical page IDs.
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    /// Page IDs indexed by logical page position within the sequence.
    pages: Vec<KvPageId>,
    /// Committed token count (number of tokens with stable KV).
    committed_tokens: usize,
}

impl BlockTable {
    /// Returns the number of committed tokens.
    pub fn committed_tokens(&self) -> usize {
        self.committed_tokens
    }

    /// Returns the page IDs covering the committed token range.
    pub fn pages(&self) -> &[KvPageId] {
        &self.pages
    }

    /// Returns the number of pages.
    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }
}

/// Per-sequence KV state tracked by the page manager.
#[derive(Debug)]
struct SequencePageState {
    /// Generation of the sequence state when this KV state was last committed.
    generation: u64,
    /// Block table mapping logical pages to physical page IDs.
    block_table: BlockTable,
}

/// Runtime page manager for logical KV allocation.
///
/// Owns page allocation, free-list, refcounting, block tables, and reservation
/// semantics. The backend owns physical buffers; this manager owns metadata.
#[derive(Debug, Clone)]
pub struct PreemptedKvState {
    generation: u64,
    block_table: BlockTable,
    evicted_pages: Vec<KvPageId>,
}

impl PreemptedKvState {
    pub fn pages(&self) -> &[KvPageId] {
        self.block_table.pages()
    }

    /// Pages exclusively referenced by this sequence and therefore safe to
    /// remove from backend residency while the logical state is suspended.
    pub fn evicted_pages(&self) -> &[KvPageId] {
        &self.evicted_pages
    }

    pub fn committed_tokens(&self) -> usize {
        self.block_table.committed_tokens()
    }
}

/// Compact provisional bindings for one reservation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvReservationBindings {
    pub block_ids: Vec<KvBlockId>,
    pub write_slots: Vec<KvWriteSlot>,
}

/// Validated but unpublished exact-prefix page-table fork.
#[derive(Debug)]
pub struct PreparedKvSequenceFork {
    source: StateSlot,
    target: StateSlot,
    target_generation: u64,
    block_table: BlockTable,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvPageManagerStats {
    pub allocated_pages: usize,
    pub free_pages: usize,
    pub shared_pages: usize,
    pub committed_tokens: usize,
    pub capacity_tokens: usize,
    pub utilization: f64,
    pub fragmentation: f64,
}

pub struct KvPageManager {
    /// The KV layout schema describing page size and planes.
    schema: Box<dyn KvLayoutSchema>,
    /// Free list of available page IDs.
    free_pages: Vec<KvPageId>,
    /// Next page ID to allocate if the free list is empty.
    next_page_id: u32,
    /// Maximum number of pages (0 = unlimited).
    max_pages: usize,
    /// Per-sequence page state, keyed by state slot index.
    sequences: BTreeMap<u32, SequencePageState>,
    /// Global refcount on each page (for COW and prefix sharing).
    page_refcounts: HashMap<KvPageId, u32>,
}

impl std::fmt::Debug for KvPageManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvPageManager")
            .field("page_size", &self.schema.page_size())
            .field("free_pages", &self.free_pages.len())
            .field("next_page_id", &self.next_page_id)
            .field("max_pages", &self.max_pages)
            .field("active_sequences", &self.sequences.len())
            .finish_non_exhaustive()
    }
}

impl KvPageManager {
    /// Create a new page manager with the given schema and maximum page count.
    pub fn new(schema: Box<dyn KvLayoutSchema>, max_pages: usize) -> Self {
        Self {
            schema,
            free_pages: Vec::new(),
            next_page_id: 0,
            max_pages,
            sequences: BTreeMap::new(),
            page_refcounts: HashMap::new(),
        }
    }

    /// Returns the page size in tokens.
    pub fn page_size(&self) -> usize {
        self.schema.page_size()
    }

    /// Returns the configured physical page limit. Zero means logically unlimited.
    pub fn max_pages(&self) -> usize {
        self.max_pages
    }

    /// Returns the number of active sequences.
    pub fn active_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Returns the number of free pages available.
    pub fn free_pages(&self) -> usize {
        self.free_pages.len()
    }

    /// Returns the total number of allocated pages (not free).
    pub fn allocated_pages(&self) -> usize {
        let total = self.next_page_id as usize;
        total - self.free_pages.len()
    }

    pub fn stats(&self) -> KvPageManagerStats {
        let allocated_pages = self.allocated_pages();
        let capacity_tokens = self
            .sequences
            .values()
            .map(|state| {
                state
                    .block_table
                    .pages
                    .len()
                    .saturating_mul(self.schema.page_size())
            })
            .sum::<usize>();
        let committed_tokens = self
            .sequences
            .values()
            .map(|state| state.block_table.committed_tokens)
            .sum::<usize>();
        let utilization = if capacity_tokens == 0 {
            0.0
        } else {
            committed_tokens as f64 / capacity_tokens as f64
        };
        KvPageManagerStats {
            allocated_pages,
            free_pages: self.free_pages.len(),
            shared_pages: self
                .page_refcounts
                .values()
                .filter(|refcount| **refcount > 1)
                .count(),
            committed_tokens,
            capacity_tokens,
            utilization,
            fragmentation: 1.0 - utilization,
        }
    }

    /// Register a new sequence in the page manager.
    pub fn alloc_sequence(&mut self, state_slot: StateSlot, generation: u64) -> Result<()> {
        let slot = state_slot.get();
        if self.sequences.contains_key(&slot) {
            return Err(Error::Internal(format!(
                "page manager: state slot {slot} is already allocated"
            )));
        }
        self.sequences.insert(
            slot,
            SequencePageState {
                generation,
                block_table: BlockTable::default(),
            },
        );
        Ok(())
    }

    /// Reserve pages for appending `token_count` tokens to a sequence.
    ///
    /// Returns a reservation containing the newly allocated page IDs. The
    /// reservation must be committed or rolled back.
    pub fn reserve(
        &mut self,
        state_slot: StateSlot,
        generation: u64,
        token_count: usize,
    ) -> Result<KvReservation> {
        let slot = state_slot.get();
        let seq = self.sequences.get(&slot).ok_or_else(|| {
            Error::Internal(format!(
                "page manager: cannot reserve for unallocated state slot {slot}"
            ))
        })?;

        if seq.generation != generation {
            return Err(Error::Execution(format!(
                "page manager: stale generation {generation} for state slot {slot} (expected {})",
                seq.generation
            )));
        }

        let committed_tokens = seq.block_table.committed_tokens;
        let existing_pages = seq.block_table.pages.clone();
        let end = committed_tokens
            .checked_add(token_count)
            .ok_or_else(|| Error::Execution("page manager: sequence length overflow".into()))?;
        if end > self.schema.max_sequence_len() {
            return Err(Error::Execution(format!(
                "page manager: sequence length {end} exceeds schema maximum {}",
                self.schema.max_sequence_len()
            )));
        }
        let positions = committed_tokens..end;
        let pages_needed = self
            .schema
            .pages_for_tokens(end)
            .saturating_sub(existing_pages.len());

        let shared_tail = token_count > 0
            && !committed_tokens.is_multiple_of(self.schema.page_size())
            && existing_pages
                .last()
                .is_some_and(|page| self.page_refcounts.get(page).copied().unwrap_or(0) > 1);
        let mut reserved_pages = Vec::with_capacity(pages_needed + usize::from(shared_tail));
        for _ in 0..pages_needed + usize::from(shared_tail) {
            match self.alloc_page() {
                Ok(page) => reserved_pages.push(page),
                Err(error) => {
                    self.free_pages.extend(reserved_pages);
                    return Err(error);
                }
            }
        }
        let cow_replacement = if shared_tail {
            let replacement = reserved_pages.remove(0);
            let logical_page = existing_pages.len() - 1;
            Some(KvCowReplacement {
                logical_page,
                source: existing_pages[logical_page],
                replacement,
            })
        } else {
            None
        };

        Ok(KvReservation {
            state_slot,
            positions,
            newly_allocated: reserved_pages,
            generation,
            cow_replacement,
        })
    }

    /// Commit a reservation, making its pages part of the committed KV view.
    pub fn commit(&mut self, reservation: KvReservation) -> Result<()> {
        self.commit_batch_with_freed(vec![reservation]).map(|_| ())
    }

    /// Commit only the first `committed_rows` tokens of a provisional reservation.
    ///
    /// Pages used exclusively by the rejected suffix are recycled and returned so
    /// a backend that provisionally published the full batch can release the same
    /// physical page IDs. Bytes beyond the committed cursor in a retained tail
    /// page are intentionally ignored and overwritten by the next append.
    pub fn commit_prefix_with_freed(
        &mut self,
        mut reservation: KvReservation,
        committed_rows: usize,
    ) -> Result<Vec<KvPageId>> {
        let reserved_rows = reservation.positions.len();
        if committed_rows > reserved_rows {
            let _ = self.rollback(reservation);
            return Err(Error::Execution(format!(
                "page manager: committed prefix {committed_rows} exceeds reservation length {reserved_rows}"
            )));
        }
        if committed_rows == reserved_rows {
            return self.commit_batch_with_freed(vec![reservation]);
        }
        if committed_rows == 0 {
            let mut freed = Vec::with_capacity(
                reservation.newly_allocated.len()
                    + usize::from(reservation.cow_replacement.is_some()),
            );
            if let Some(cow) = reservation.cow_replacement {
                freed.push(cow.replacement);
            }
            freed.extend(reservation.newly_allocated.iter().copied());
            self.rollback(reservation)?;
            return Ok(freed);
        }

        let final_end = reservation
            .positions
            .start
            .checked_add(committed_rows)
            .ok_or_else(|| Error::Execution("page manager: prefix end overflow".into()))?;
        let pages_before = self.schema.pages_for_tokens(reservation.positions.start);
        let pages_after = self.schema.pages_for_tokens(final_end);
        let kept_new_pages = pages_after.checked_sub(pages_before).ok_or_else(|| {
            Error::Internal("page manager: prefix page count moved backwards".into())
        })?;
        let available_new_pages = reservation.newly_allocated.len();
        if kept_new_pages > available_new_pages {
            let _ = self.rollback(reservation);
            return Err(Error::Internal(format!(
                "page manager: prefix needs {kept_new_pages} new pages but reservation has {available_new_pages}"
            )));
        }
        let rejected_pages = reservation.newly_allocated.split_off(kept_new_pages);
        reservation.positions.end = final_end;

        let commit = self.commit_batch_with_freed(vec![reservation]);
        match commit {
            Ok(mut freed) => {
                self.free_pages.extend(rejected_pages.iter().copied());
                freed.extend(rejected_pages);
                Ok(freed)
            }
            Err(error) => {
                self.free_pages.extend(rejected_pages);
                Err(error)
            }
        }
    }

    /// Atomically publish all reservations in a packed batch.
    ///
    /// Every reservation is validated before any block table or refcount changes.
    /// A batch cannot contain multiple reservations for the same state slot because
    /// their ordering would otherwise be implicit rather than part of the ABI.
    pub fn commit_batch_with_freed(
        &mut self,
        reservations: Vec<KvReservation>,
    ) -> Result<Vec<KvPageId>> {
        let mut slots = std::collections::HashSet::with_capacity(reservations.len());
        let mut provisional_pages = std::collections::HashSet::new();
        let validation = (|| -> Result<()> {
            for reservation in &reservations {
                let slot = reservation.state_slot.get();
                if !slots.insert(slot) {
                    return Err(Error::Execution(format!(
                        "page manager: packed commit contains duplicate state slot {slot}"
                    )));
                }
                let seq = self.sequences.get(&slot).ok_or_else(|| {
                    Error::Internal(format!(
                        "page manager: cannot commit for unallocated state slot {slot}"
                    ))
                })?;
                if seq.generation != reservation.generation {
                    return Err(Error::Execution(format!(
                        "page manager: stale generation on commit for state slot {slot}"
                    )));
                }
                if seq.block_table.committed_tokens != reservation.positions.start {
                    return Err(Error::Execution(format!(
                        "page manager: reservation starts at {}, committed view is {}",
                        reservation.positions.start, seq.block_table.committed_tokens
                    )));
                }
                if let Some(cow) = reservation.cow_replacement {
                    if seq.block_table.pages.get(cow.logical_page) != Some(&cow.source) {
                        return Err(Error::Execution(
                            "page manager: stale COW tail mapping".into(),
                        ));
                    }
                    if self.page_refcount(cow.source) == 0 {
                        return Err(Error::Internal(
                            "page manager: COW source has no live refcount".into(),
                        ));
                    }
                    if !provisional_pages.insert(cow.replacement)
                        || self.page_refcounts.contains_key(&cow.replacement)
                    {
                        return Err(Error::Execution(
                            "page manager: COW replacement is already allocated".into(),
                        ));
                    }
                }
                for page in &reservation.newly_allocated {
                    if !provisional_pages.insert(*page) || self.page_refcounts.contains_key(page) {
                        return Err(Error::Execution(
                            "page manager: provisional page is already allocated".into(),
                        ));
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = validation {
            let mut recycled = std::collections::HashSet::new();
            for reservation in &reservations {
                if let Some(cow) = reservation.cow_replacement
                    && recycled.insert(cow.replacement)
                {
                    self.free_pages.push(cow.replacement);
                }
                for page in &reservation.newly_allocated {
                    if recycled.insert(*page) {
                        self.free_pages.push(*page);
                    }
                }
            }
            return Err(error);
        }

        let mut freed_pages = Vec::new();
        for reservation in reservations {
            let seq = self
                .sequences
                .get_mut(&reservation.state_slot.get())
                .expect("packed reservation was validated");
            if let Some(cow) = reservation.cow_replacement {
                seq.block_table.pages[cow.logical_page] = cow.replacement;
                self.page_refcounts.insert(cow.replacement, 1);
                if decrement_refcount(&mut self.page_refcounts, &mut self.free_pages, cow.source)? {
                    freed_pages.push(cow.source);
                }
            }
            for page_id in reservation.newly_allocated {
                seq.block_table.pages.push(page_id);
                self.page_refcounts.insert(page_id, 1);
            }
            seq.block_table.committed_tokens = reservation.positions.end;
        }
        Ok(freed_pages)
    }

    /// Rollback a reservation, freeing its newly allocated pages.
    pub fn rollback(&mut self, reservation: KvReservation) -> Result<()> {
        if let Some(cow) = reservation.cow_replacement {
            self.free_pages.push(cow.replacement);
        }
        self.free_pages.extend(reservation.newly_allocated);
        Ok(())
    }

    /// Free a sequence and return physical page IDs no longer referenced globally.
    pub fn free_sequence_pages(&mut self, state_slot: StateSlot) -> Result<Vec<KvPageId>> {
        let slot = state_slot.get();
        let seq = self.sequences.remove(&slot).ok_or_else(|| {
            Error::Internal(format!(
                "page manager: cannot free unallocated state slot {slot}"
            ))
        })?;

        let mut freed_pages = Vec::new();
        for page_id in seq.block_table.pages {
            if decrement_refcount(&mut self.page_refcounts, &mut self.free_pages, page_id)? {
                freed_pages.push(page_id);
            }
        }
        Ok(freed_pages)
    }

    /// Returns the block table for a sequence.
    pub fn block_table(&self, state_slot: StateSlot) -> Option<&BlockTable> {
        self.sequences
            .get(&state_slot.get())
            .map(|s| &s.block_table)
    }

    /// Validate an exact committed-prefix fork without changing either sequence.
    pub fn prepare_fork_sequence_exact(
        &self,
        source: StateSlot,
        target: StateSlot,
        target_generation: u64,
        expected_prefix_tokens: usize,
    ) -> Result<PreparedKvSequenceFork> {
        if self.sequences.contains_key(&target.get()) {
            return Err(Error::Internal(format!(
                "page manager: target state slot {} is already allocated",
                target.get()
            )));
        }
        let block_table = self
            .sequences
            .get(&source.get())
            .ok_or_else(|| Error::Internal("page manager: fork source is not allocated".into()))?
            .block_table
            .clone();
        if block_table.committed_tokens != expected_prefix_tokens {
            return Err(Error::Execution(format!(
                "page manager: fork prefix mismatch: expected {expected_prefix_tokens} committed tokens, source has {}",
                block_table.committed_tokens
            )));
        }
        for page in &block_table.pages {
            let refcount = self.page_refcounts.get(page).copied().ok_or_else(|| {
                Error::Internal(format!(
                    "page manager: fork source page {} has no refcount",
                    page.0
                ))
            })?;
            if refcount == 0 {
                return Err(Error::Internal(format!(
                    "page manager: fork source page {} has zero refcount",
                    page.0
                )));
            }
            refcount.checked_add(1).ok_or_else(|| {
                Error::Execution(format!(
                    "page manager: fork source page {} refcount overflow",
                    page.0
                ))
            })?;
        }
        Ok(PreparedKvSequenceFork {
            source,
            target,
            target_generation,
            block_table,
        })
    }

    /// Publish a previously validated page-table fork.
    pub fn publish_fork_sequence_exact(&mut self, prepared: PreparedKvSequenceFork) -> Result<()> {
        if self.sequences.contains_key(&prepared.target.get()) {
            return Err(Error::Internal(format!(
                "page manager: prepared fork target state slot {} became allocated",
                prepared.target.get()
            )));
        }
        let source = self.sequences.get(&prepared.source.get()).ok_or_else(|| {
            Error::Internal("page manager: prepared fork source disappeared".into())
        })?;
        if source.block_table.pages != prepared.block_table.pages
            || source.block_table.committed_tokens != prepared.block_table.committed_tokens
        {
            return Err(Error::Execution(
                "page manager: prepared fork source changed before publish".into(),
            ));
        }
        for page in &prepared.block_table.pages {
            let refcount = self.page_refcounts.get(page).copied().ok_or_else(|| {
                Error::Internal("page manager: prepared fork source refcount disappeared".into())
            })?;
            refcount.checked_add(1).ok_or_else(|| {
                Error::Execution("page manager: prepared fork refcount overflow".into())
            })?;
        }
        for page in &prepared.block_table.pages {
            *self
                .page_refcounts
                .get_mut(page)
                .expect("prepared fork refcounts were revalidated") += 1;
        }
        self.sequences.insert(
            prepared.target.get(),
            SequencePageState {
                generation: prepared.target_generation,
                block_table: prepared.block_table,
            },
        );
        Ok(())
    }

    /// Detach a sequence from scheduling while retaining its page references.
    pub fn preempt_sequence(&mut self, state_slot: StateSlot) -> Result<PreemptedKvState> {
        let state = self.sequences.remove(&state_slot.get()).ok_or_else(|| {
            Error::Internal("page manager: cannot preempt an unallocated sequence".into())
        })?;
        let evicted_pages = state
            .block_table
            .pages
            .iter()
            .copied()
            .filter(|page| self.page_refcount(*page) == 1)
            .collect();
        Ok(PreemptedKvState {
            generation: state.generation,
            block_table: state.block_table,
            evicted_pages,
        })
    }

    /// Restore a previously preempted sequence without changing page identity.
    pub fn restore_sequence(
        &mut self,
        state_slot: StateSlot,
        state: PreemptedKvState,
    ) -> Result<()> {
        if self.sequences.contains_key(&state_slot.get()) {
            return Err(Error::Internal(
                "page manager: restore target is allocated".into(),
            ));
        }
        self.sequences.insert(
            state_slot.get(),
            SequencePageState {
                generation: state.generation,
                block_table: state.block_table,
            },
        );
        Ok(())
    }

    /// Release a preempted state and return pages whose global refcount reached zero.
    pub fn release_preempted_pages(&mut self, state: PreemptedKvState) -> Result<Vec<KvPageId>> {
        let mut freed = Vec::new();
        for page in state.block_table.pages {
            if decrement_refcount(&mut self.page_refcounts, &mut self.free_pages, page)? {
                freed.push(page);
            }
        }
        Ok(freed)
    }

    pub fn page_refcount(&self, page: KvPageId) -> u32 {
        self.page_refcounts.get(&page).copied().unwrap_or(0)
    }

    /// Derive a read-only prefix view for backend replay without publishing or
    /// transferring ownership of the original reservation.
    pub fn reservation_prefix_view(
        &self,
        reservation: &KvReservation,
        prefix_rows: usize,
    ) -> Result<KvReservation> {
        if prefix_rows > reservation.positions.len() {
            return Err(Error::Execution(format!(
                "page manager: reservation prefix {prefix_rows} exceeds reserved rows {}",
                reservation.positions.len()
            )));
        }
        let prefix_end = reservation
            .positions
            .start
            .checked_add(prefix_rows)
            .ok_or_else(|| Error::Execution("page manager: prefix view end overflow".into()))?;
        let pages_before = self.schema.pages_for_tokens(reservation.positions.start);
        let pages_after = self.schema.pages_for_tokens(prefix_end);
        let prefix_new_pages = pages_after.checked_sub(pages_before).ok_or_else(|| {
            Error::Internal("page manager: prefix view page count moved backwards".into())
        })?;
        if prefix_new_pages > reservation.newly_allocated.len() {
            return Err(Error::Internal(format!(
                "page manager: prefix view needs {prefix_new_pages} new pages but reservation has {}",
                reservation.newly_allocated.len()
            )));
        }
        let mut view = reservation.clone();
        view.positions.end = prefix_end;
        view.newly_allocated.truncate(prefix_new_pages);
        if prefix_rows == 0 {
            view.cow_replacement = None;
        }
        Ok(view)
    }

    /// Build provisional block/write bindings without publishing the reservation.
    pub fn reservation_bindings(
        &self,
        reservation: &KvReservation,
    ) -> Result<KvReservationBindings> {
        let state = self
            .sequences
            .get(&reservation.state_slot.get())
            .ok_or_else(|| {
                Error::Internal("page manager: reservation sequence is not allocated".into())
            })?;
        if state.generation != reservation.generation {
            return Err(Error::Execution(
                "page manager: stale reservation binding".into(),
            ));
        }
        let mut pages = state.block_table.pages.clone();
        if let Some(cow) = reservation.cow_replacement {
            if pages.get(cow.logical_page) != Some(&cow.source) {
                return Err(Error::Execution("page manager: stale COW binding".into()));
            }
            pages[cow.logical_page] = cow.replacement;
        }
        pages.extend_from_slice(&reservation.newly_allocated);
        let block_ids = pages
            .iter()
            .map(|page| KvBlockId::new(page.0))
            .collect::<Vec<_>>();
        let mut write_slots = Vec::with_capacity(reservation.positions.len());
        for position in reservation.positions.clone() {
            let logical_page = position / self.schema.page_size();
            let offset = position % self.schema.page_size();
            let page = pages.get(logical_page).ok_or_else(|| {
                Error::Internal("page manager: reservation has no page for write".into())
            })?;
            let physical = usize::try_from(page.0)
                .ok()
                .and_then(|page| page.checked_mul(self.schema.page_size()))
                .and_then(|base| base.checked_add(offset))
                .ok_or_else(|| Error::Execution("page manager: write slot overflow".into()))?;
            write_slots.push(KvWriteSlot::try_from(physical).map_err(|_| {
                Error::Execution("page manager: write slot exceeds u32 ABI".into())
            })?);
        }
        Ok(KvReservationBindings {
            block_ids,
            write_slots,
        })
    }

    fn alloc_page(&mut self) -> Result<KvPageId> {
        if let Some(page_id) = self.free_pages.pop() {
            return Ok(page_id);
        }

        if self.max_pages > 0 && (self.next_page_id as usize) >= self.max_pages {
            return Err(Error::Internal(format!(
                "page manager: out of pages (max {})",
                self.max_pages
            )));
        }

        let page_id = KvPageId(self.next_page_id);
        self.next_page_id += 1;
        Ok(page_id)
    }
}

fn decrement_refcount(
    refcounts: &mut HashMap<KvPageId, u32>,
    free_pages: &mut Vec<KvPageId>,
    page: KvPageId,
) -> Result<bool> {
    let refcount = refcounts
        .get_mut(&page)
        .ok_or_else(|| Error::Internal(format!("page manager: page {} has no refcount", page.0)))?;
    if *refcount == 0 {
        return Err(Error::Internal(format!(
            "page manager: page {} has zero refcount",
            page.0
        )));
    }
    *refcount -= 1;
    if *refcount == 0 {
        refcounts.remove(&page);
        free_pages.push(page);
        return Ok(true);
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_common::execution::KvPlaneDescriptor;

    /// Simple test schema: 1 plane, page_size=4, max 1024 tokens.
    #[derive(Debug)]
    struct TestSchema {
        page_size: usize,
    }

    static TEST_PLANE: KvPlaneDescriptor = KvPlaneDescriptor {
        name: "test",
        elements_per_token: 1,
        layer_count: 1,
    };

    impl KvLayoutSchema for TestSchema {
        fn planes(&self) -> &[KvPlaneDescriptor] {
            std::slice::from_ref(&TEST_PLANE)
        }
        fn page_size(&self) -> usize {
            self.page_size
        }
        fn max_sequence_len(&self) -> usize {
            8192
        }
    }

    fn slot(n: u32) -> StateSlot {
        StateSlot::new(n)
    }

    fn publish_exact_fork(
        manager: &mut KvPageManager,
        source: StateSlot,
        target: StateSlot,
        generation: u64,
        expected_prefix_tokens: usize,
    ) -> Result<()> {
        let prepared = manager.prepare_fork_sequence_exact(
            source,
            target,
            generation,
            expected_prefix_tokens,
        )?;
        manager.publish_fork_sequence_exact(prepared)
    }

    #[test]
    fn alloc_and_free_sequence() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        assert_eq!(mgr.active_sequences(), 1);
        mgr.free_sequence_pages(slot(0)).unwrap();
        assert_eq!(mgr.active_sequences(), 0);
    }

    #[test]
    fn reserve_commit_extends_block_table() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();

        // Reserve 4 tokens = 1 page
        let res = mgr.reserve(slot(0), 0, 4).unwrap();
        assert_eq!(res.newly_allocated.len(), 1);
        mgr.commit(res).unwrap();

        let table = mgr.block_table(slot(0)).unwrap();
        assert_eq!(table.committed_tokens(), 4);
        assert_eq!(table.num_pages(), 1);
    }

    #[test]
    fn reservation_prefix_view_is_non_publishing_and_truncates_backend_bindings() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let initial = mgr.reserve(slot(0), 0, 3).unwrap();
        mgr.commit(initial).unwrap();

        let reservation = mgr.reserve(slot(0), 0, 10).unwrap();
        assert_eq!(reservation.newly_allocated.len(), 3);
        let view = mgr.reservation_prefix_view(&reservation, 5).unwrap();
        assert_eq!(view.positions, 3..8);
        assert_eq!(view.newly_allocated.len(), 1);
        let bindings = mgr.reservation_bindings(&view).unwrap();
        assert_eq!(bindings.write_slots.len(), 5);
        assert_eq!(bindings.block_ids.len(), 2);
        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 3);
        assert_eq!(reservation.positions, 3..13);
        assert_eq!(reservation.newly_allocated.len(), 3);

        mgr.rollback(reservation).unwrap();
        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 3);
    }

    #[test]
    fn commit_prefix_recycles_rejected_suffix_pages() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();

        let reservation = mgr.reserve(slot(0), 0, 10).unwrap();
        let rejected_page = reservation.newly_allocated[2];
        let freed = mgr.commit_prefix_with_freed(reservation, 5).unwrap();

        let table = mgr.block_table(slot(0)).unwrap();
        assert_eq!(table.committed_tokens(), 5);
        assert_eq!(table.num_pages(), 2);
        assert_eq!(freed, vec![rejected_page]);
        assert_eq!(mgr.allocated_pages(), 2);
        assert_eq!(mgr.free_pages(), 1);
    }

    #[test]
    fn zero_prefix_is_an_exact_reservation_rollback() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();

        let reservation = mgr.reserve(slot(0), 0, 9).unwrap();
        let expected = reservation.newly_allocated.clone();
        let freed = mgr.commit_prefix_with_freed(reservation, 0).unwrap();

        assert_eq!(freed, expected);
        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 0);
        assert_eq!(mgr.allocated_pages(), 0);
        assert_eq!(mgr.free_pages(), 3);
    }

    #[test]
    fn commit_prefix_rejects_lengths_beyond_the_reservation() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let reservation = mgr.reserve(slot(0), 0, 2).unwrap();

        assert!(mgr.commit_prefix_with_freed(reservation, 3).is_err());
        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 0);
        assert_eq!(mgr.allocated_pages(), 0);
    }

    #[test]
    fn reserve_rollback_frees_pages() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();

        let res = mgr.reserve(slot(0), 0, 4).unwrap();
        let _page_id = res.newly_allocated[0];
        mgr.rollback(res).unwrap();

        // Page should be back on the free list
        assert_eq!(mgr.free_pages(), 1);
        assert_eq!(mgr.allocated_pages(), 0);

        // Block table should not have been extended
        let table = mgr.block_table(slot(0)).unwrap();
        assert_eq!(table.committed_tokens(), 0);
        assert_eq!(table.num_pages(), 0);
    }

    #[test]
    fn stale_generation_rejected() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();

        let res = mgr.reserve(slot(0), 1, 4);
        assert!(res.is_err());
    }

    #[test]
    fn packed_commit_validation_failure_publishes_nothing_and_recycles_pages() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        mgr.alloc_sequence(slot(1), 0).unwrap();
        let first = mgr.reserve(slot(0), 0, 4).unwrap();
        let mut stale = mgr.reserve(slot(1), 0, 4).unwrap();
        stale.generation = 1;

        assert!(mgr.commit_batch_with_freed(vec![first, stale]).is_err());
        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 0);
        assert_eq!(mgr.block_table(slot(1)).unwrap().committed_tokens(), 0);
        assert_eq!(mgr.allocated_pages(), 0);
        assert_eq!(mgr.free_pages(), 2);
    }

    #[test]
    fn multiple_pages_for_long_sequence() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();

        // 10 tokens with page_size=4 = 3 pages
        let res = mgr.reserve(slot(0), 0, 10).unwrap();
        assert_eq!(res.newly_allocated.len(), 3);
        mgr.commit(res).unwrap();

        let table = mgr.block_table(slot(0)).unwrap();
        assert_eq!(table.committed_tokens(), 10);
        assert_eq!(table.num_pages(), 3);
        let stats = mgr.stats();
        assert_eq!(stats.capacity_tokens, 12);
        assert_eq!(stats.committed_tokens, 10);
        assert!((stats.utilization - 10.0 / 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn incremental_reserve_commits() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();

        // First batch: 4 tokens = 1 page
        let res = mgr.reserve(slot(0), 0, 4).unwrap();
        mgr.commit(res).unwrap();

        // Second batch: 4 more tokens = 1 more page
        let res = mgr.reserve(slot(0), 0, 4).unwrap();
        assert_eq!(res.newly_allocated.len(), 1);
        mgr.commit(res).unwrap();

        let table = mgr.block_table(slot(0)).unwrap();
        assert_eq!(table.committed_tokens(), 8);
        assert_eq!(table.num_pages(), 2);
    }

    #[test]
    fn out_of_pages_returns_error() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 2);
        mgr.alloc_sequence(slot(0), 0).unwrap();

        // 4 tokens = 1 page, OK
        let res = mgr.reserve(slot(0), 0, 4).unwrap();
        mgr.commit(res).unwrap();

        // 4 more tokens = 1 more page, OK (total 2)
        let res = mgr.reserve(slot(0), 0, 4).unwrap();
        mgr.commit(res).unwrap();

        // 4 more tokens = 1 more page, OOM
        let res = mgr.reserve(slot(0), 0, 4);
        assert!(res.is_err());
    }

    #[test]
    fn free_sequence_returns_pages_to_free_list() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();

        let res = mgr.reserve(slot(0), 0, 8).unwrap();
        mgr.commit(res).unwrap();
        assert_eq!(mgr.allocated_pages(), 2);

        mgr.free_sequence_pages(slot(0)).unwrap();
        assert_eq!(mgr.allocated_pages(), 0);
        assert_eq!(mgr.free_pages(), 2);
    }

    #[test]
    fn page_reuse_after_free() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let res = mgr.reserve(slot(0), 0, 4).unwrap();
        let first_page = res.newly_allocated[0];
        mgr.commit(res).unwrap();
        mgr.free_sequence_pages(slot(0)).unwrap();

        // Allocate a new sequence - should reuse the freed page
        mgr.alloc_sequence(slot(1), 0).unwrap();
        let res = mgr.reserve(slot(1), 0, 4).unwrap();
        assert_eq!(res.newly_allocated[0], first_page);
    }

    #[test]
    fn page_boundaries_and_position_over_4096_lower_exact_write_slots() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 2048);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let first = mgr.reserve(slot(0), 0, 4095).unwrap();
        mgr.commit(first).unwrap();
        let boundary = mgr.reserve(slot(0), 0, 3).unwrap();
        let bindings = mgr.reservation_bindings(&boundary).unwrap();
        assert_eq!(boundary.positions, 4095..4098);
        assert_eq!(bindings.write_slots.len(), 3);
        assert_eq!(bindings.block_ids.len(), 1025);
        mgr.commit(boundary).unwrap();
        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 4098);
    }

    #[test]
    fn fork_shares_pages_and_partial_tail_append_uses_cow() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let initial = mgr.reserve(slot(0), 0, 3).unwrap();
        mgr.commit(initial).unwrap();
        let shared_page = mgr.block_table(slot(0)).unwrap().pages()[0];
        publish_exact_fork(&mut mgr, slot(0), slot(1), 7, 3).unwrap();
        assert_eq!(mgr.page_refcount(shared_page), 2);

        let append = mgr.reserve(slot(1), 7, 1).unwrap();
        let cow = append.cow_replacement.expect("shared tail requires COW");
        assert_eq!(cow.source, shared_page);
        mgr.commit(append).unwrap();
        assert_ne!(mgr.block_table(slot(1)).unwrap().pages()[0], shared_page);
        assert_eq!(mgr.block_table(slot(0)).unwrap().pages()[0], shared_page);
        assert_eq!(mgr.page_refcount(shared_page), 1);
    }

    #[test]
    fn prepared_fork_is_invisible_until_publish() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let initial = mgr.reserve(slot(0), 0, 6).unwrap();
        mgr.commit(initial).unwrap();
        let pages = mgr.block_table(slot(0)).unwrap().pages().to_vec();

        let prepared = mgr
            .prepare_fork_sequence_exact(slot(0), slot(1), 7, 6)
            .unwrap();
        assert!(mgr.block_table(slot(1)).is_none());
        assert!(pages.iter().all(|page| mgr.page_refcount(*page) == 1));

        mgr.publish_fork_sequence_exact(prepared).unwrap();
        assert_eq!(mgr.block_table(slot(1)).unwrap().committed_tokens(), 6);
        assert!(pages.iter().all(|page| mgr.page_refcount(*page) == 2));
    }

    #[test]
    fn exact_fork_rejects_stale_prefix_without_changing_refcounts() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let initial = mgr.reserve(slot(0), 0, 6).unwrap();
        mgr.commit(initial).unwrap();
        let pages = mgr.block_table(slot(0)).unwrap().pages().to_vec();
        let before = pages
            .iter()
            .map(|page| mgr.page_refcount(*page))
            .collect::<Vec<_>>();

        let error = publish_exact_fork(&mut mgr, slot(0), slot(1), 1, 5).unwrap_err();
        assert!(error.to_string().contains("fork prefix mismatch"));
        assert!(mgr.block_table(slot(1)).is_none());
        assert_eq!(
            pages
                .iter()
                .map(|page| mgr.page_refcount(*page))
                .collect::<Vec<_>>(),
            before
        );
    }

    #[test]
    fn exact_fork_refcount_overflow_is_failure_atomic() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let initial = mgr.reserve(slot(0), 0, 8).unwrap();
        mgr.commit(initial).unwrap();
        let pages = mgr.block_table(slot(0)).unwrap().pages().to_vec();
        mgr.page_refcounts.insert(pages[1], u32::MAX);

        let error = publish_exact_fork(&mut mgr, slot(0), slot(1), 1, 8).unwrap_err();
        assert!(error.to_string().contains("refcount overflow"));
        assert!(mgr.block_table(slot(1)).is_none());
        assert_eq!(mgr.page_refcount(pages[0]), 1);
        assert_eq!(mgr.page_refcount(pages[1]), u32::MAX);
    }

    #[test]
    fn cow_rollback_preserves_shared_view() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let initial = mgr.reserve(slot(0), 0, 3).unwrap();
        mgr.commit(initial).unwrap();
        publish_exact_fork(&mut mgr, slot(0), slot(1), 1, 3).unwrap();
        let before = mgr.block_table(slot(1)).unwrap().clone();
        let append = mgr.reserve(slot(1), 1, 1).unwrap();
        mgr.rollback(append).unwrap();
        assert_eq!(mgr.block_table(slot(1)).unwrap().pages(), before.pages());
        assert_eq!(mgr.block_table(slot(1)).unwrap().committed_tokens(), 3);
    }

    #[test]
    fn preempt_restore_preserves_exact_block_table() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 9).unwrap();
        let reservation = mgr.reserve(slot(0), 9, 9).unwrap();
        mgr.commit(reservation).unwrap();
        let before = mgr.block_table(slot(0)).unwrap().clone();
        let preempted = mgr.preempt_sequence(slot(0)).unwrap();
        assert_eq!(mgr.active_sequences(), 0);
        mgr.restore_sequence(slot(0), preempted).unwrap();
        let after = mgr.block_table(slot(0)).unwrap();
        assert_eq!(after.pages(), before.pages());
        assert_eq!(after.committed_tokens(), before.committed_tokens());
    }

    #[test]
    fn failed_multi_page_reserve_returns_all_partial_allocations() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 2);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        assert!(mgr.reserve(slot(0), 0, 12).is_err());
        assert_eq!(mgr.allocated_pages(), 0);
        assert_eq!(mgr.free_pages(), 2);
    }
}
