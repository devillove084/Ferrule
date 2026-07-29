//! Runtime KV page manager.
//!
//! The page manager is the single owner of logical page allocation, refcount,
//! block tables, and reservation semantics. It is model-agnostic: it works
//! with any [`KvLayoutSchema`] supplied by the model.
//!
//! ## Lifecycle
//!
//! ```text
//! reserve -> execute -> prepare logical commit -> commit backend -> publish logical commit
//!                      ↘ backend rollback -> abort reservation
//! publish/abort/free -> quarantine -> release backend pages -> confirm retirement -> reuse
//! ```
//!
//! Reservation, prepared-commit, and retirement values are non-cloneable ownership
//! tokens. Logical publication cannot precede backend commit, and a page cannot be
//! reused until backend retirement is explicitly confirmed.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use ferrule_common::execution::{
    KvBlockId, KvCowReplacement, KvLayoutSchema, KvPageId, KvReservationView, KvWriteSlot,
    StateSlot,
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

/// Unique identity of one live logical KV reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvReservationId(NonZeroU64);

impl KvReservationId {
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Non-cloneable ownership token for one provisional logical KV reservation.
#[derive(Debug)]
pub struct KvReservation {
    id: KvReservationId,
    view: KvReservationView,
}

impl KvReservation {
    pub const fn id(&self) -> KvReservationId {
        self.id
    }

    pub const fn view(&self) -> &KvReservationView {
        &self.view
    }
}

impl std::ops::Deref for KvReservation {
    type Target = KvReservationView;

    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

/// Model-agnostic commit decision for one provisional KV reservation.
///
/// Packed transactions carry one decision per sequence. `committed_rows` may be
/// any prefix length from zero through the full reservation width; no model or
/// speculative decoding policy is encoded in the page manager.
#[derive(Debug)]
pub struct KvReservationCommit {
    pub reservation: KvReservation,
    pub committed_rows: usize,
}

impl KvReservationCommit {
    pub const fn new(reservation: KvReservation, committed_rows: usize) -> Self {
        Self {
            reservation,
            committed_rows,
        }
    }
}

#[derive(Debug)]
struct PreparedKvCommitEntry {
    reservation_id: KvReservationId,
    state_slot: StateSlot,
    after: BlockTable,
    retained_pages: Vec<KvPageId>,
    cow_source: Option<KvPageId>,
    rejected_pages: Vec<KvPageId>,
    abort_pages: Vec<KvPageId>,
}

/// Entire validated packed logical KV commit, still invisible to readers.
///
/// This token is deliberately non-cloneable and cannot be split per sequence.
/// The owning backend transaction must commit before this token is published.
#[must_use = "a prepared KV commit must be published or aborted"]
#[derive(Debug)]
pub struct PreparedKvCommit {
    manager_id: u64,
    entries: Vec<PreparedKvCommitEntry>,
    retirement_capacity: usize,
}

/// Failure to prepare a logical commit without consuming reservation ownership.
#[derive(Debug)]
pub struct PrepareKvCommitError {
    error: Error,
    commits: Vec<KvReservationCommit>,
}

impl PrepareKvCommitError {
    pub fn into_parts(self) -> (Error, Vec<KvReservationCommit>) {
        (self.error, self.commits)
    }
}

impl std::fmt::Display for PrepareKvCommitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for PrepareKvCommitError {}

/// Failure to abort reservations without consuming their ownership tokens.
#[derive(Debug)]
pub struct AbortKvReservationsError {
    error: Error,
    reservations: Vec<KvReservation>,
}

impl AbortKvReservationsError {
    pub fn into_parts(self) -> (Error, Vec<KvReservation>) {
        (self.error, self.reservations)
    }
}

impl std::fmt::Display for AbortKvReservationsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for AbortKvReservationsError {}

/// Pages quarantined until their backend storage is no longer reachable.
#[must_use = "retiring KV pages must be released and confirmed"]
#[derive(Debug)]
pub struct KvRetirement {
    manager_id: u64,
    pages: Vec<KvPageId>,
}

impl KvRetirement {
    pub fn pages(&self) -> &[KvPageId] {
        &self.pages
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

/// Failed retirement confirmation retaining the linear retirement token.
#[derive(Debug)]
pub struct ConfirmKvRetirementError {
    error: Error,
    retirement: KvRetirement,
}

impl ConfirmKvRetirementError {
    pub fn into_parts(self) -> (Error, KvRetirement) {
        (self.error, self.retirement)
    }
}

impl std::fmt::Display for ConfirmKvRetirementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for ConfirmKvRetirementError {}

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
    pub retiring_pages: usize,
    pub shared_pages: usize,
    pub committed_tokens: usize,
    pub capacity_tokens: usize,
    pub utilization: f64,
    pub fragmentation: f64,
}

static NEXT_PAGE_MANAGER_ID: AtomicU64 = AtomicU64::new(1);

pub struct KvPageManager {
    manager_id: u64,
    /// The KV layout schema describing page size and planes.
    schema: Box<dyn KvLayoutSchema>,
    /// Free list of available page IDs.
    free_pages: Vec<KvPageId>,
    /// Pages whose logical refcount reached zero but whose backend retirement has
    /// not yet been acknowledged. These pages are not allocatable.
    retiring_pages: HashSet<KvPageId>,
    /// Live reservation identities and their immutable views.
    pending_reservations: HashMap<KvReservationId, KvReservationView>,
    /// At most one live reservation may own a sequence state slot.
    reservation_owners: HashMap<u32, KvReservationId>,
    next_reservation_id: NonZeroU64,
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
            .field("retiring_pages", &self.retiring_pages.len())
            .field("pending_reservations", &self.pending_reservations.len())
            .field("active_sequences", &self.sequences.len())
            .finish_non_exhaustive()
    }
}

impl KvPageManager {
    /// Create a new page manager with the given schema and maximum page count.
    pub fn new(schema: Box<dyn KvLayoutSchema>, max_pages: usize) -> Self {
        let manager_id = NEXT_PAGE_MANAGER_ID.fetch_add(1, Ordering::Relaxed);
        assert_ne!(manager_id, 0, "KV page-manager identity space exhausted");
        Self {
            manager_id,
            schema,
            free_pages: Vec::new(),
            retiring_pages: HashSet::new(),
            pending_reservations: HashMap::new(),
            reservation_owners: HashMap::new(),
            next_reservation_id: NonZeroU64::new(1).expect("one is non-zero"),
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

    fn take_reservation_id(&mut self) -> Result<KvReservationId> {
        let id = KvReservationId(self.next_reservation_id);
        let next = self
            .next_reservation_id
            .get()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .ok_or_else(|| Error::Execution("KV reservation ID space exhausted".into()))?;
        self.next_reservation_id = next;
        Ok(id)
    }

    fn validate_live_reservations(&self, reservations: &[&KvReservation]) -> Result<()> {
        let mut ids = HashSet::with_capacity(reservations.len());
        let mut slots = HashSet::with_capacity(reservations.len());
        for reservation in reservations {
            if !ids.insert(reservation.id) {
                return Err(Error::Execution(format!(
                    "page manager: reservation {} appears more than once",
                    reservation.id.get()
                )));
            }
            let slot = reservation.state_slot.get();
            if !slots.insert(slot) {
                return Err(Error::Execution(format!(
                    "page manager: packed transaction contains duplicate state slot {slot}"
                )));
            }
            let stored = self
                .pending_reservations
                .get(&reservation.id)
                .ok_or_else(|| {
                    Error::Execution(format!(
                        "page manager: reservation {} is not live",
                        reservation.id.get()
                    ))
                })?;
            if stored != &reservation.view
                || self.reservation_owners.get(&slot) != Some(&reservation.id)
            {
                return Err(Error::Internal(format!(
                    "page manager: reservation {} ownership changed",
                    reservation.id.get()
                )));
            }
        }
        Ok(())
    }

    fn release_reservation_ownership(&mut self, id: KvReservationId, slot: StateSlot) {
        let stored = self
            .pending_reservations
            .remove(&id)
            .expect("prepared reservation must remain registered");
        assert_eq!(stored.state_slot, slot);
        let owner = self
            .reservation_owners
            .remove(&slot.get())
            .expect("prepared reservation owner must remain registered");
        assert_eq!(owner, id);
    }

    fn begin_retirement(&mut self, mut pages: Vec<KvPageId>) -> KvRetirement {
        pages.sort_unstable_by_key(|page| page.0);
        pages.dedup();
        self.retiring_pages.reserve(pages.len());
        for page in &pages {
            assert!(
                self.retiring_pages.insert(*page),
                "page {} entered retirement more than once",
                page.0
            );
        }
        KvRetirement {
            manager_id: self.manager_id,
            pages,
        }
    }

    /// Publish backend retirement completion and make pages allocatable again.
    pub fn confirm_page_retirement(
        &mut self,
        retirement: KvRetirement,
    ) -> std::result::Result<(), ConfirmKvRetirementError> {
        let invalid = if retirement.manager_id != self.manager_id {
            Some(Error::Execution(
                "page manager: retirement belongs to another manager".into(),
            ))
        } else {
            retirement.pages.iter().find_map(|page| {
                (!self.retiring_pages.contains(page)).then(|| {
                    Error::Execution(format!("page manager: page {} is not retiring", page.0))
                })
            })
        };
        if let Some(error) = invalid {
            return Err(ConfirmKvRetirementError { error, retirement });
        }
        for page in &retirement.pages {
            self.retiring_pages.remove(page);
        }
        self.free_pages.extend(retirement.pages);
        Ok(())
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
            retiring_pages: self.retiring_pages.len(),
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

    pub fn sequence_generation(&self, state_slot: StateSlot) -> Result<u64> {
        self.sequences
            .get(&state_slot.get())
            .map(|sequence| sequence.generation)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "page manager: state slot {} is not allocated",
                    state_slot.get()
                ))
            })
    }

    /// Return the exact number of new physical pages (including a COW tail)
    /// required by a reservation without mutating manager state.
    pub fn required_physical_pages(
        &self,
        state_slot: StateSlot,
        generation: u64,
        token_count: usize,
    ) -> Result<usize> {
        let slot = state_slot.get();
        if self.reservation_owners.contains_key(&slot) {
            return Err(Error::Execution(format!(
                "page manager: state slot {slot} already owns a live reservation"
            )));
        }
        let sequence = self.sequences.get(&slot).ok_or_else(|| {
            Error::Internal(format!(
                "page manager: cannot size reservation for unallocated state slot {slot}"
            ))
        })?;
        if sequence.generation != generation {
            return Err(Error::Execution(format!(
                "page manager: stale generation {generation} for state slot {slot} (expected {})",
                sequence.generation
            )));
        }
        let end = sequence
            .block_table
            .committed_tokens
            .checked_add(token_count)
            .ok_or_else(|| Error::Execution("page manager: sequence length overflow".into()))?;
        if end > self.schema.max_sequence_len() {
            return Err(Error::Execution(format!(
                "page manager: sequence length {end} exceeds schema maximum {}",
                self.schema.max_sequence_len()
            )));
        }
        let pages_needed = self
            .schema
            .pages_for_tokens(end)
            .saturating_sub(sequence.block_table.pages.len());
        let shared_tail = token_count > 0
            && !sequence
                .block_table
                .committed_tokens
                .is_multiple_of(self.schema.page_size())
            && sequence
                .block_table
                .pages
                .last()
                .is_some_and(|page| self.page_refcounts.get(page).copied().unwrap_or(0) > 1);
        Ok(pages_needed + usize::from(shared_tail))
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
        if let Some(owner) = self.reservation_owners.get(&slot) {
            return Err(Error::Execution(format!(
                "page manager: state slot {slot} is already owned by reservation {}",
                owner.get()
            )));
        }
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

        let view = KvReservationView {
            state_slot,
            execution_state_slot: state_slot,
            positions,
            newly_allocated: reserved_pages,
            generation,
            execution_generation: generation,
            cow_replacement,
        };
        let id = self.take_reservation_id()?;
        self.pending_reservations.insert(id, view.clone());
        self.reservation_owners.insert(slot, id);
        Ok(KvReservation { id, view })
    }

    /// Validate and seal an entire packed logical commit without publishing it.
    ///
    /// On failure every reservation token is returned unchanged and manager state
    /// remains untouched. On success the returned token owns the complete cohort.
    pub fn prepare_commit(
        &mut self,
        commits: Vec<KvReservationCommit>,
    ) -> std::result::Result<PreparedKvCommit, PrepareKvCommitError> {
        let reservations = commits
            .iter()
            .map(|commit| &commit.reservation)
            .collect::<Vec<_>>();
        let planned = (|| -> Result<(Vec<PreparedKvCommitEntry>, usize, usize)> {
            self.validate_live_reservations(&reservations)?;
            let mut provisional_pages = HashSet::new();
            let mut entries = Vec::with_capacity(commits.len());
            let mut retirement_capacity = 0usize;
            let mut retained_capacity = 0usize;

            for commit in &commits {
                let reservation = &commit.reservation;
                let slot = reservation.state_slot.get();
                let sequence = self.sequences.get(&slot).ok_or_else(|| {
                    Error::Internal(format!(
                        "page manager: cannot commit unallocated state slot {slot}"
                    ))
                })?;
                if sequence.generation != reservation.generation {
                    return Err(Error::Execution(format!(
                        "page manager: stale generation on commit for state slot {slot}"
                    )));
                }
                if sequence.block_table.committed_tokens != reservation.positions.start {
                    return Err(Error::Execution(format!(
                        "page manager: reservation starts at {}, committed view is {}",
                        reservation.positions.start, sequence.block_table.committed_tokens
                    )));
                }
                if commit.committed_rows > reservation.positions.len() {
                    return Err(Error::Execution(format!(
                        "page manager: committed prefix {} exceeds reservation length {}",
                        commit.committed_rows,
                        reservation.positions.len()
                    )));
                }

                let final_end = reservation
                    .positions
                    .start
                    .checked_add(commit.committed_rows)
                    .ok_or_else(|| Error::Execution("page manager: prefix end overflow".into()))?;
                let pages_before = self.schema.pages_for_tokens(reservation.positions.start);
                let pages_after = self.schema.pages_for_tokens(final_end);
                let kept_new_pages = pages_after.checked_sub(pages_before).ok_or_else(|| {
                    Error::Internal("page manager: committed page count moved backwards".into())
                })?;
                if kept_new_pages > reservation.newly_allocated.len() {
                    return Err(Error::Internal(format!(
                        "page manager: prefix ending at {final_end} needs {kept_new_pages} new pages but reservation has {}",
                        reservation.newly_allocated.len()
                    )));
                }

                let mut after = sequence.block_table.clone();
                let mut retained_pages = Vec::with_capacity(
                    kept_new_pages
                        + usize::from(
                            commit.committed_rows > 0 && reservation.cow_replacement.is_some(),
                        ),
                );
                let mut rejected_pages = Vec::with_capacity(
                    reservation.newly_allocated.len() - kept_new_pages
                        + usize::from(
                            commit.committed_rows == 0 && reservation.cow_replacement.is_some(),
                        ),
                );
                let mut abort_pages = Vec::with_capacity(
                    reservation.newly_allocated.len()
                        + usize::from(reservation.cow_replacement.is_some()),
                );
                let mut cow_source = None;

                if let Some(cow) = reservation.cow_replacement {
                    if sequence.block_table.pages.get(cow.logical_page) != Some(&cow.source) {
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
                    abort_pages.push(cow.replacement);
                    if commit.committed_rows == 0 {
                        rejected_pages.push(cow.replacement);
                    } else {
                        after.pages[cow.logical_page] = cow.replacement;
                        retained_pages.push(cow.replacement);
                        cow_source = Some(cow.source);
                    }
                }

                for page in &reservation.newly_allocated {
                    if !provisional_pages.insert(*page) || self.page_refcounts.contains_key(page) {
                        return Err(Error::Execution(
                            "page manager: provisional page is already allocated".into(),
                        ));
                    }
                }
                retained_pages.extend_from_slice(&reservation.newly_allocated[..kept_new_pages]);
                rejected_pages.extend_from_slice(&reservation.newly_allocated[kept_new_pages..]);
                abort_pages.extend_from_slice(&reservation.newly_allocated);
                after
                    .pages
                    .extend_from_slice(&reservation.newly_allocated[..kept_new_pages]);
                after.committed_tokens = final_end;

                retirement_capacity = retirement_capacity
                    .checked_add(rejected_pages.len() + usize::from(cow_source.is_some()))
                    .ok_or_else(|| Error::Execution("page retirement capacity overflow".into()))?;
                retained_capacity = retained_capacity
                    .checked_add(retained_pages.len())
                    .ok_or_else(|| Error::Execution("page commit capacity overflow".into()))?;
                entries.push(PreparedKvCommitEntry {
                    reservation_id: reservation.id,
                    state_slot: reservation.state_slot,
                    after,
                    retained_pages,
                    cow_source,
                    rejected_pages,
                    abort_pages,
                });
            }
            Ok((entries, retirement_capacity, retained_capacity))
        })();

        let (entries, retirement_capacity, retained_capacity) = match planned {
            Ok(planned) => planned,
            Err(error) => return Err(PrepareKvCommitError { error, commits }),
        };
        self.page_refcounts.reserve(retained_capacity);
        self.retiring_pages.reserve(retirement_capacity);
        Ok(PreparedKvCommit {
            manager_id: self.manager_id,
            entries,
            retirement_capacity,
        })
    }

    /// Publish one backend-committed packed transaction in a single infallible step.
    pub fn publish_commit(&mut self, prepared: PreparedKvCommit) -> KvRetirement {
        assert_eq!(
            prepared.manager_id, self.manager_id,
            "prepared KV commit belongs to another manager"
        );
        let mut retiring = Vec::with_capacity(prepared.retirement_capacity);
        for entry in prepared.entries {
            self.release_reservation_ownership(entry.reservation_id, entry.state_slot);
            let sequence = self
                .sequences
                .get_mut(&entry.state_slot.get())
                .expect("prepared KV sequence must remain allocated");
            sequence.block_table = entry.after;
            for page in entry.retained_pages {
                assert!(
                    self.page_refcounts.insert(page, 1).is_none(),
                    "prepared provisional page was already committed"
                );
            }
            if let Some(source) = entry.cow_source
                && decrement_refcount_infallible(&mut self.page_refcounts, source)
            {
                retiring.push(source);
            }
            retiring.extend(entry.rejected_pages);
        }
        self.begin_retirement(retiring)
    }

    /// Abort a prepared logical commit after backend rollback has quiesced it.
    pub fn abort_prepared_commit(&mut self, prepared: PreparedKvCommit) -> KvRetirement {
        assert_eq!(
            prepared.manager_id, self.manager_id,
            "prepared KV commit belongs to another manager"
        );
        let mut retiring = Vec::new();
        for entry in prepared.entries {
            self.release_reservation_ownership(entry.reservation_id, entry.state_slot);
            retiring.extend(entry.abort_pages);
        }
        self.begin_retirement(retiring)
    }

    /// Abort a packed set of live reservations without partial consumption.
    pub fn abort_reservations(
        &mut self,
        reservations: Vec<KvReservation>,
    ) -> std::result::Result<KvRetirement, AbortKvReservationsError> {
        let references = reservations.iter().collect::<Vec<_>>();
        if let Err(error) = self.validate_live_reservations(&references) {
            return Err(AbortKvReservationsError {
                error,
                reservations,
            });
        }
        let mut retiring = Vec::new();
        for reservation in reservations {
            self.release_reservation_ownership(reservation.id, reservation.state_slot);
            retiring.extend(reservation.view.newly_allocated);
            if let Some(cow) = reservation.view.cow_replacement {
                retiring.push(cow.replacement);
            }
        }
        Ok(self.begin_retirement(retiring))
    }

    /// Free a sequence and quarantine pages whose global refcount reaches zero.
    pub fn free_sequence_pages(&mut self, state_slot: StateSlot) -> Result<KvRetirement> {
        let slot = state_slot.get();
        if let Some(owner) = self.reservation_owners.get(&slot) {
            return Err(Error::Execution(format!(
                "page manager: cannot free state slot {slot} owned by reservation {}",
                owner.get()
            )));
        }
        let pages = self
            .sequences
            .get(&slot)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "page manager: cannot free unallocated state slot {slot}"
                ))
            })?
            .block_table
            .pages
            .clone();
        self.validate_refcount_decrements(&pages)?;
        self.sequences.remove(&slot);
        let retiring = pages
            .into_iter()
            .filter(|page| decrement_refcount_infallible(&mut self.page_refcounts, *page))
            .collect();
        Ok(self.begin_retirement(retiring))
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
        if let Some(owner) = self.reservation_owners.get(&state_slot.get()) {
            return Err(Error::Execution(format!(
                "page manager: cannot preempt state slot {} owned by reservation {}",
                state_slot.get(),
                owner.get()
            )));
        }
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

    /// Release a preempted state and quarantine pages whose refcount reaches zero.
    pub fn release_preempted_pages(&mut self, state: PreemptedKvState) -> Result<KvRetirement> {
        self.validate_refcount_decrements(&state.block_table.pages)?;
        let retiring = state
            .block_table
            .pages
            .into_iter()
            .filter(|page| decrement_refcount_infallible(&mut self.page_refcounts, *page))
            .collect();
        Ok(self.begin_retirement(retiring))
    }

    pub fn page_refcount(&self, page: KvPageId) -> u32 {
        self.page_refcounts.get(&page).copied().unwrap_or(0)
    }

    pub fn bind_reservation_execution(
        &mut self,
        reservation: &mut KvReservation,
        execution_state_slot: StateSlot,
        execution_generation: u64,
    ) -> Result<()> {
        let stored = self
            .pending_reservations
            .get_mut(&reservation.id)
            .ok_or_else(|| {
                Error::Execution(format!(
                    "page manager: reservation {} is not live",
                    reservation.id.get()
                ))
            })?;
        if stored != &reservation.view {
            return Err(Error::Internal(format!(
                "page manager: reservation {} view changed before execution binding",
                reservation.id.get()
            )));
        }
        stored.execution_state_slot = execution_state_slot;
        stored.execution_generation = execution_generation;
        reservation.view.execution_state_slot = execution_state_slot;
        reservation.view.execution_generation = execution_generation;
        Ok(())
    }

    pub fn reservation_view(&self, reservation: &KvReservation) -> Result<KvReservationView> {
        let stored = self
            .pending_reservations
            .get(&reservation.id)
            .ok_or_else(|| {
                Error::Execution(format!(
                    "page manager: reservation {} is not live",
                    reservation.id.get()
                ))
            })?;
        if stored != &reservation.view {
            return Err(Error::Internal(format!(
                "page manager: reservation {} view changed",
                reservation.id.get()
            )));
        }
        Ok(stored.clone())
    }

    pub fn reservation_views(
        &self,
        reservations: &[KvReservation],
    ) -> Result<Vec<KvReservationView>> {
        reservations
            .iter()
            .map(|reservation| self.reservation_view(reservation))
            .collect()
    }

    /// Derive a read-only prefix view for backend replay without publishing or
    /// transferring ownership of the original reservation.
    pub fn reservation_prefix_view(
        &self,
        reservation: &KvReservation,
        prefix_rows: usize,
    ) -> Result<KvReservationView> {
        let mut view = self.reservation_view(reservation)?;
        if prefix_rows > view.positions.len() {
            return Err(Error::Execution(format!(
                "page manager: reservation prefix {prefix_rows} exceeds reserved rows {}",
                view.positions.len()
            )));
        }
        let prefix_end = view
            .positions
            .start
            .checked_add(prefix_rows)
            .ok_or_else(|| Error::Execution("page manager: prefix view end overflow".into()))?;
        let pages_before = self.schema.pages_for_tokens(view.positions.start);
        let pages_after = self.schema.pages_for_tokens(prefix_end);
        let prefix_new_pages = pages_after.checked_sub(pages_before).ok_or_else(|| {
            Error::Internal("page manager: prefix view page count moved backwards".into())
        })?;
        if prefix_new_pages > view.newly_allocated.len() {
            return Err(Error::Internal(format!(
                "page manager: prefix view needs {prefix_new_pages} new pages but reservation has {}",
                view.newly_allocated.len()
            )));
        }
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
        reservation: &KvReservationView,
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

    fn validate_refcount_decrements(&self, pages: &[KvPageId]) -> Result<()> {
        let mut decrements = HashMap::<KvPageId, u32>::new();
        for page in pages {
            let count = decrements.entry(*page).or_default();
            *count = count.checked_add(1).ok_or_else(|| {
                Error::Execution("page manager: refcount decrement overflow".into())
            })?;
        }
        for (page, decrement) in decrements {
            let refcount = self.page_refcounts.get(&page).copied().ok_or_else(|| {
                Error::Internal(format!("page manager: page {} has no refcount", page.0))
            })?;
            if refcount < decrement {
                return Err(Error::Internal(format!(
                    "page manager: page {} refcount {refcount} is below decrement {decrement}",
                    page.0
                )));
            }
        }
        Ok(())
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

fn decrement_refcount_infallible(refcounts: &mut HashMap<KvPageId, u32>, page: KvPageId) -> bool {
    let refcount = refcounts
        .get_mut(&page)
        .expect("validated page must have a refcount");
    assert!(*refcount > 0, "validated page refcount must be non-zero");
    *refcount -= 1;
    if *refcount == 0 {
        refcounts.remove(&page);
        true
    } else {
        false
    }
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

    fn commit_full(manager: &mut KvPageManager, reservation: KvReservation) {
        let rows = reservation.positions.len();
        let prepared = manager
            .prepare_commit(vec![KvReservationCommit::new(reservation, rows)])
            .unwrap();
        let retirement = manager.publish_commit(prepared);
        manager.confirm_page_retirement(retirement).unwrap();
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
        let retirement = mgr.free_sequence_pages(slot(0)).unwrap();
        mgr.confirm_page_retirement(retirement).unwrap();
        assert_eq!(mgr.active_sequences(), 0);
    }

    #[test]
    fn reserve_commit_extends_block_table() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();

        // Reserve 4 tokens = 1 page
        let res = mgr.reserve(slot(0), 0, 4).unwrap();
        assert_eq!(res.newly_allocated.len(), 1);
        commit_full(&mut mgr, res);

        let table = mgr.block_table(slot(0)).unwrap();
        assert_eq!(table.committed_tokens(), 4);
        assert_eq!(table.num_pages(), 1);
    }

    #[test]
    fn reservation_prefix_view_is_non_publishing_and_truncates_backend_bindings() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let initial = mgr.reserve(slot(0), 0, 3).unwrap();
        commit_full(&mut mgr, initial);

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

        let retirement = mgr.abort_reservations(vec![reservation]).unwrap();
        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 3);
        assert_eq!(mgr.free_pages(), 0);
        assert_eq!(mgr.stats().retiring_pages, 3);
        mgr.confirm_page_retirement(retirement).unwrap();
        assert_eq!(mgr.free_pages(), 3);
    }

    #[test]
    fn commit_prefix_quarantines_rejected_suffix_until_confirmation() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();

        let reservation = mgr.reserve(slot(0), 0, 10).unwrap();
        let rejected_page = reservation.newly_allocated[2];
        let prepared = mgr
            .prepare_commit(vec![KvReservationCommit::new(reservation, 5)])
            .unwrap();
        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 0);
        let retirement = mgr.publish_commit(prepared);

        let table = mgr.block_table(slot(0)).unwrap();
        assert_eq!(table.committed_tokens(), 5);
        assert_eq!(table.num_pages(), 2);
        assert_eq!(retirement.pages(), &[rejected_page]);
        assert_eq!(mgr.allocated_pages(), 3);
        assert_eq!(mgr.free_pages(), 0);
        assert_eq!(mgr.stats().retiring_pages, 1);
        mgr.confirm_page_retirement(retirement).unwrap();
        assert_eq!(mgr.allocated_pages(), 2);
        assert_eq!(mgr.free_pages(), 1);
    }

    #[test]
    fn packed_prefix_commit_publishes_ragged_widths_atomically() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 32);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        mgr.alloc_sequence(slot(1), 0).unwrap();

        let first = mgr.reserve(slot(0), 0, 10).unwrap();
        let second = mgr.reserve(slot(1), 0, 7).unwrap();
        let prepared = mgr
            .prepare_commit(vec![
                KvReservationCommit::new(first, 5),
                KvReservationCommit::new(second, 3),
            ])
            .unwrap();
        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 0);
        assert_eq!(mgr.block_table(slot(1)).unwrap().committed_tokens(), 0);
        let retirement = mgr.publish_commit(prepared);

        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 5);
        assert_eq!(mgr.block_table(slot(1)).unwrap().committed_tokens(), 3);
        assert_eq!(mgr.block_table(slot(0)).unwrap().num_pages(), 2);
        assert_eq!(mgr.block_table(slot(1)).unwrap().num_pages(), 1);
        assert_eq!(retirement.pages().len(), 2);
        assert_eq!(mgr.allocated_pages(), 5);
        assert_eq!(mgr.free_pages(), 0);
        mgr.confirm_page_retirement(retirement).unwrap();
        assert_eq!(mgr.allocated_pages(), 3);
        assert_eq!(mgr.free_pages(), 2);
    }

    #[test]
    fn packed_prefix_validation_failure_returns_all_ownership_without_publication() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 32);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        mgr.alloc_sequence(slot(1), 0).unwrap();

        let first = mgr.reserve(slot(0), 0, 10).unwrap();
        let mut stale = mgr.reserve(slot(1), 0, 7).unwrap();
        stale.view.generation = 1;
        let error = mgr
            .prepare_commit(vec![
                KvReservationCommit::new(first, 5),
                KvReservationCommit::new(stale, 3),
            ])
            .unwrap_err();
        let (_, mut commits) = error.into_parts();

        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 0);
        assert_eq!(mgr.block_table(slot(1)).unwrap().committed_tokens(), 0);
        assert_eq!(mgr.allocated_pages(), 5);
        assert_eq!(mgr.free_pages(), 0);
        commits[1].reservation.view.generation = 0;
        let retirement = mgr
            .abort_reservations(
                commits
                    .into_iter()
                    .map(|commit| commit.reservation)
                    .collect(),
            )
            .unwrap();
        assert_eq!(retirement.pages().len(), 5);
        mgr.confirm_page_retirement(retirement).unwrap();
        assert_eq!(mgr.allocated_pages(), 0);
        assert_eq!(mgr.free_pages(), 5);
    }

    #[test]
    fn packed_prefix_supports_commit_and_rollback_in_one_transaction() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        mgr.alloc_sequence(slot(1), 0).unwrap();

        let committed = mgr.reserve(slot(0), 0, 5).unwrap();
        let cancelled = mgr.reserve(slot(1), 0, 5).unwrap();
        let prepared = mgr
            .prepare_commit(vec![
                KvReservationCommit::new(committed, 5),
                KvReservationCommit::new(cancelled, 0),
            ])
            .unwrap();
        let retirement = mgr.publish_commit(prepared);

        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 5);
        assert_eq!(mgr.block_table(slot(1)).unwrap().committed_tokens(), 0);
        assert_eq!(retirement.pages().len(), 2);
        assert_eq!(mgr.allocated_pages(), 4);
        mgr.confirm_page_retirement(retirement).unwrap();
        assert_eq!(mgr.allocated_pages(), 2);
    }

    #[test]
    fn zero_prefix_is_an_exact_quarantined_rollback() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();

        let reservation = mgr.reserve(slot(0), 0, 9).unwrap();
        let expected = reservation.newly_allocated.clone();
        let prepared = mgr
            .prepare_commit(vec![KvReservationCommit::new(reservation, 0)])
            .unwrap();
        let retirement = mgr.publish_commit(prepared);

        assert_eq!(retirement.pages(), expected);
        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 0);
        assert_eq!(mgr.allocated_pages(), 3);
        assert_eq!(mgr.free_pages(), 0);
        mgr.confirm_page_retirement(retirement).unwrap();
        assert_eq!(mgr.allocated_pages(), 0);
        assert_eq!(mgr.free_pages(), 3);
    }

    #[test]
    fn commit_prefix_rejects_lengths_beyond_the_reservation_without_consuming_it() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let reservation = mgr.reserve(slot(0), 0, 2).unwrap();

        let error = mgr
            .prepare_commit(vec![KvReservationCommit::new(reservation, 3)])
            .unwrap_err();
        let (_, commits) = error.into_parts();
        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 0);
        assert_eq!(mgr.allocated_pages(), 1);
        let retirement = mgr
            .abort_reservations(
                commits
                    .into_iter()
                    .map(|commit| commit.reservation)
                    .collect(),
            )
            .unwrap();
        mgr.confirm_page_retirement(retirement).unwrap();
        assert_eq!(mgr.allocated_pages(), 0);
    }

    #[test]
    fn reservation_abort_requires_retirement_confirmation_before_reuse() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 1);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        mgr.alloc_sequence(slot(1), 0).unwrap();

        let reservation = mgr.reserve(slot(0), 0, 4).unwrap();
        let page_id = reservation.newly_allocated[0];
        let retirement = mgr.abort_reservations(vec![reservation]).unwrap();
        assert_eq!(retirement.pages(), &[page_id]);
        assert_eq!(mgr.free_pages(), 0);
        assert_eq!(mgr.allocated_pages(), 1);
        assert!(mgr.reserve(slot(1), 0, 4).is_err());

        mgr.confirm_page_retirement(retirement).unwrap();
        let reused = mgr.reserve(slot(1), 0, 4).unwrap();
        assert_eq!(reused.newly_allocated, vec![page_id]);
    }

    #[test]
    fn one_state_slot_cannot_have_two_live_reservations() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 4);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let first = mgr.reserve(slot(0), 0, 1).unwrap();
        let error = mgr.reserve(slot(0), 0, 1).unwrap_err();
        assert!(error.to_string().contains("already owned by reservation"));
        assert_eq!(mgr.pending_reservations.len(), 1);
        let retirement = mgr.abort_reservations(vec![first]).unwrap();
        mgr.confirm_page_retirement(retirement).unwrap();
    }

    #[test]
    fn duplicate_retirement_confirmation_is_rejected_without_free_list_mutation() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 1);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let reservation = mgr.reserve(slot(0), 0, 4).unwrap();
        let retirement = mgr.abort_reservations(vec![reservation]).unwrap();
        let duplicate = KvRetirement {
            manager_id: retirement.manager_id,
            pages: retirement.pages.clone(),
        };
        mgr.confirm_page_retirement(retirement).unwrap();
        assert_eq!(mgr.free_pages(), 1);

        let error = mgr.confirm_page_retirement(duplicate).unwrap_err();
        let (_, duplicate) = error.into_parts();
        assert_eq!(duplicate.pages().len(), 1);
        assert_eq!(mgr.free_pages(), 1);
        assert_eq!(mgr.stats().retiring_pages, 0);
    }

    #[test]
    fn retirement_cannot_be_confirmed_by_another_manager() {
        let mut owner = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 1);
        let mut other = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 1);
        owner.alloc_sequence(slot(0), 0).unwrap();
        let reservation = owner.reserve(slot(0), 0, 4).unwrap();
        let retirement = owner.abort_reservations(vec![reservation]).unwrap();

        let error = other.confirm_page_retirement(retirement).unwrap_err();
        let (_, retirement) = error.into_parts();
        assert_eq!(other.free_pages(), 0);
        owner.confirm_page_retirement(retirement).unwrap();
        assert_eq!(owner.free_pages(), 1);
    }

    #[test]
    fn consumed_reservation_token_cannot_change_manager_state() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 1);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let reservation = mgr.reserve(slot(0), 0, 4).unwrap();
        let id = reservation.id;
        let view = reservation.view.clone();
        let retirement = mgr.abort_reservations(vec![reservation]).unwrap();
        mgr.confirm_page_retirement(retirement).unwrap();
        let free_before = mgr.free_pages();

        let consumed = KvReservation { id, view };
        let error = mgr.abort_reservations(vec![consumed]).unwrap_err();
        let (_, returned) = error.into_parts();
        assert_eq!(returned.len(), 1);
        assert_eq!(mgr.free_pages(), free_before);
        assert_eq!(mgr.stats().retiring_pages, 0);
        assert!(mgr.pending_reservations.is_empty());
    }

    #[test]
    fn stale_generation_rejected() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();

        let res = mgr.reserve(slot(0), 1, 4);
        assert!(res.is_err());
    }

    #[test]
    fn packed_full_commit_validation_failure_is_failure_atomic() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        mgr.alloc_sequence(slot(1), 0).unwrap();
        let first = mgr.reserve(slot(0), 0, 4).unwrap();
        let mut stale = mgr.reserve(slot(1), 0, 4).unwrap();
        stale.view.generation = 1;

        let error = mgr
            .prepare_commit(vec![
                KvReservationCommit::new(first, 4),
                KvReservationCommit::new(stale, 4),
            ])
            .unwrap_err();
        let (_, mut commits) = error.into_parts();
        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 0);
        assert_eq!(mgr.block_table(slot(1)).unwrap().committed_tokens(), 0);
        assert_eq!(mgr.allocated_pages(), 2);
        assert_eq!(mgr.free_pages(), 0);
        commits[1].reservation.view.generation = 0;
        let retirement = mgr
            .abort_reservations(
                commits
                    .into_iter()
                    .map(|commit| commit.reservation)
                    .collect(),
            )
            .unwrap();
        mgr.confirm_page_retirement(retirement).unwrap();
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
        commit_full(&mut mgr, res);

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
        commit_full(&mut mgr, res);

        // Second batch: 4 more tokens = 1 more page
        let res = mgr.reserve(slot(0), 0, 4).unwrap();
        assert_eq!(res.newly_allocated.len(), 1);
        commit_full(&mut mgr, res);

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
        commit_full(&mut mgr, res);

        // 4 more tokens = 1 more page, OK (total 2)
        let res = mgr.reserve(slot(0), 0, 4).unwrap();
        commit_full(&mut mgr, res);

        // 4 more tokens = 1 more page, OOM
        let res = mgr.reserve(slot(0), 0, 4);
        assert!(res.is_err());
    }

    #[test]
    fn free_sequence_returns_pages_to_free_list() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();

        let res = mgr.reserve(slot(0), 0, 8).unwrap();
        commit_full(&mut mgr, res);
        assert_eq!(mgr.allocated_pages(), 2);

        let retirement = mgr.free_sequence_pages(slot(0)).unwrap();
        assert_eq!(mgr.allocated_pages(), 2);
        assert_eq!(mgr.free_pages(), 0);
        assert_eq!(mgr.stats().retiring_pages, 2);
        mgr.confirm_page_retirement(retirement).unwrap();
        assert_eq!(mgr.allocated_pages(), 0);
        assert_eq!(mgr.free_pages(), 2);
    }

    #[test]
    fn page_reuse_after_free() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let res = mgr.reserve(slot(0), 0, 4).unwrap();
        let first_page = res.newly_allocated[0];
        commit_full(&mut mgr, res);
        let retirement = mgr.free_sequence_pages(slot(0)).unwrap();
        mgr.confirm_page_retirement(retirement).unwrap();

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
        commit_full(&mut mgr, first);
        let boundary = mgr.reserve(slot(0), 0, 3).unwrap();
        let bindings = mgr.reservation_bindings(&boundary).unwrap();
        assert_eq!(boundary.positions, 4095..4098);
        assert_eq!(bindings.write_slots.len(), 3);
        assert_eq!(bindings.block_ids.len(), 1025);
        commit_full(&mut mgr, boundary);
        assert_eq!(mgr.block_table(slot(0)).unwrap().committed_tokens(), 4098);
    }

    #[test]
    fn fork_shares_pages_and_partial_tail_append_uses_cow() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let initial = mgr.reserve(slot(0), 0, 3).unwrap();
        commit_full(&mut mgr, initial);
        let shared_page = mgr.block_table(slot(0)).unwrap().pages()[0];
        publish_exact_fork(&mut mgr, slot(0), slot(1), 7, 3).unwrap();
        assert_eq!(mgr.page_refcount(shared_page), 2);

        let append = mgr.reserve(slot(1), 7, 1).unwrap();
        let cow = append.cow_replacement.expect("shared tail requires COW");
        assert_eq!(cow.source, shared_page);
        commit_full(&mut mgr, append);
        assert_ne!(mgr.block_table(slot(1)).unwrap().pages()[0], shared_page);
        assert_eq!(mgr.block_table(slot(0)).unwrap().pages()[0], shared_page);
        assert_eq!(mgr.page_refcount(shared_page), 1);
    }

    #[test]
    fn prepared_fork_is_invisible_until_publish() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 0).unwrap();
        let initial = mgr.reserve(slot(0), 0, 6).unwrap();
        commit_full(&mut mgr, initial);
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
        commit_full(&mut mgr, initial);
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
        commit_full(&mut mgr, initial);
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
        commit_full(&mut mgr, initial);
        publish_exact_fork(&mut mgr, slot(0), slot(1), 1, 3).unwrap();
        let before = mgr.block_table(slot(1)).unwrap().clone();
        let append = mgr.reserve(slot(1), 1, 1).unwrap();
        let retirement = mgr.abort_reservations(vec![append]).unwrap();
        mgr.confirm_page_retirement(retirement).unwrap();
        assert_eq!(mgr.block_table(slot(1)).unwrap().pages(), before.pages());
        assert_eq!(mgr.block_table(slot(1)).unwrap().committed_tokens(), 3);
    }

    #[test]
    fn preempt_restore_preserves_exact_block_table() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 16);
        mgr.alloc_sequence(slot(0), 9).unwrap();
        let reservation = mgr.reserve(slot(0), 9, 9).unwrap();
        commit_full(&mut mgr, reservation);
        let before = mgr.block_table(slot(0)).unwrap().clone();
        let preempted = mgr.preempt_sequence(slot(0)).unwrap();
        assert_eq!(mgr.active_sequences(), 0);
        mgr.restore_sequence(slot(0), preempted).unwrap();
        let after = mgr.block_table(slot(0)).unwrap();
        assert_eq!(after.pages(), before.pages());
        assert_eq!(after.committed_tokens(), before.committed_tokens());
    }

    #[test]
    fn execution_binding_is_independent_from_page_owner_identity() {
        let mut mgr = KvPageManager::new(Box::new(TestSchema { page_size: 4 }), 4);
        mgr.alloc_sequence(slot(7), 3).unwrap();
        let mut reservation = mgr.reserve(slot(7), 3, 1).unwrap();

        mgr.bind_reservation_execution(&mut reservation, slot(0), 9)
            .unwrap();
        let view = mgr.reservation_view(&reservation).unwrap();
        assert_eq!(view.state_slot, slot(7));
        assert_eq!(view.generation, 3);
        assert_eq!(view.execution_state_slot, slot(0));
        assert_eq!(view.execution_generation, 9);
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
