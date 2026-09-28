use super::DecoderSequence;
use crate::execution::{ExecutionShapeKey, SequenceTopologyId};
use ferrule_common::execution::{
    ExecutionBatch, ExecutionCapabilities, ExecutionIntent, ForwardMode, ForwardPhase,
    KvCowReplacement, KvPageId, KvReservationView, KvWriteSlot, LogitsRequest, StateSlot,
};
use ferrule_common::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
/// Backend residency visible during packed KV lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderKvPageStatus {
    Vacant,
    Resident,
    Preempted,
}
/// Read-only page-state boundary used to validate reservation custody.
pub trait DecoderKvPageView {
    fn page_status(&self, page: KvPageId) -> DecoderKvPageStatus;
}
impl<F> DecoderKvPageView for F
where
    F: Fn(KvPageId) -> DecoderKvPageStatus,
{
    fn page_status(&self, page: KvPageId) -> DecoderKvPageStatus {
        self(page)
    }
}
/// One requested output row in packed input order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LogitsPlanRow {
    input_row: usize,
    request: LogitsRequest,
}
impl LogitsPlanRow {
    pub const fn input_row(self) -> usize {
        self.input_row
    }
    pub const fn request(self) -> LogitsRequest {
        self.request
    }
}
/// Exact requested-row projection for decoder output lowering.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LogitsPlan {
    packed_row_count: usize,
    full_logits_width: Option<usize>,
    rows: Box<[LogitsPlanRow]>,
}
impl LogitsPlan {
    pub fn from_batch(batch: &ExecutionBatch) -> Self {
        let rows = batch
            .logits()
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(input_row, request)| {
                (!matches!(request, LogitsRequest::None))
                    .then_some(LogitsPlanRow { input_row, request })
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            packed_row_count: batch.len(),
            full_logits_width: None,
            rows,
        }
    }
    fn from_validated_batch(
        batch: &ExecutionBatch,
        capabilities: &ExecutionCapabilities,
    ) -> Result<Self> {
        let mut plan = Self::from_batch(batch);
        plan.full_logits_width = capabilities
            .full_logits_width
            .map(|width| {
                usize::try_from(width.get()).map_err(|_| {
                    execution_error("full-logits width cannot be represented as usize")
                })
            })
            .transpose()?;
        Ok(plan)
    }
    pub const fn packed_row_count(&self) -> usize {
        self.packed_row_count
    }
    pub const fn full_logits_width(&self) -> Option<usize> {
        self.full_logits_width
    }
    pub fn rows(&self) -> &[LogitsPlanRow] {
        &self.rows
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}
/// One validated sequence projection in a [`PackedDecoderBatch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedDecoderSequence {
    state_index: usize,
    topology_id: SequenceTopologyId,
    page_state_slot: StateSlot,
    page_generation: u64,
    execution_generation: u64,
    phase: ForwardPhase,
    query: Range<usize>,
    context_len: usize,
    sequence_len: usize,
    block_table: Box<[KvPageId]>,
}
impl PackedDecoderSequence {
    pub const fn state_index(&self) -> usize {
        self.state_index
    }
    pub const fn topology_id(&self) -> SequenceTopologyId {
        self.topology_id
    }
    pub const fn page_state_slot(&self) -> StateSlot {
        self.page_state_slot
    }
    pub const fn page_generation(&self) -> u64 {
        self.page_generation
    }
    pub const fn execution_generation(&self) -> u64 {
        self.execution_generation
    }
    pub const fn phase(&self) -> ForwardPhase {
        self.phase
    }
    pub fn query(&self) -> Range<usize> {
        self.query.clone()
    }
    pub fn query_len(&self) -> usize {
        self.query.len()
    }
    pub const fn context_len(&self) -> usize {
        self.context_len
    }
    pub const fn sequence_len(&self) -> usize {
        self.sequence_len
    }
    pub fn block_table(&self) -> &[KvPageId] {
        &self.block_table
    }
}
/// Commit equality only; no tokens, intent, phase, logits or owner-local execution proof.
///
/// ```compile_fail
/// use ferrule_model::decoder::{DecoderKvBackend, KvCommitProjection};
/// fn enter<B: DecoderKvBackend>(b: &mut B, tx: &mut B::Transaction,
///     p: &KvCommitProjection, states: &mut [B::SequenceState]) {
///     b.enter(tx, p, states).unwrap();
/// }
/// ```
/// ```compile_fail
/// use ferrule_model::decoder::{KvCommitProjection, PackedDecoderBatch};
/// fn executable(p: KvCommitProjection) -> PackedDecoderBatch { p.into() }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCommitProjection {
    page_size: usize,
    new_pages: Box<[KvPageId]>,
    writable_pages: Box<[KvPageId]>,
    cow_replacements: Box<[KvCowReplacement]>,
    protected_pages: Box<[KvPageId]>,
    row_to_sequence: Box<[usize]>,
    positions: Box<[usize]>,
    sequences: Box<[CommitSequence]>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct CommitSequence {
    page_state_slot: StateSlot,
    page_generation: u64,
    context_len: usize,
    query_len: usize,
    block_table: Box<[KvPageId]>,
}
impl From<&LogicalSequence> for CommitSequence {
    fn from(s: &LogicalSequence) -> Self {
        Self {
            page_state_slot: s.page_state_slot,
            page_generation: s.page_generation,
            context_len: s.context_len,
            query_len: s.query.len(),
            block_table: s.block_table.clone(),
        }
    }
}
impl KvCommitProjection {
    /// Validate logical reservations and page custody without constructing executable state.
    pub fn validate(
        batch: &ExecutionBatch,
        reservations: &[KvReservationView],
        capabilities: &ExecutionCapabilities,
        page_size: usize,
        pages: &impl DecoderKvPageView,
    ) -> Result<Self> {
        Ok(
            ValidatedLayout::validate(batch, reservations, capabilities, page_size, pages)?
                .projection(),
        )
    }
    pub fn protected_pages(&self) -> &[KvPageId] {
        &self.protected_pages
    }
}
/// Exact logical execution identity including session order, tokens, intent, mode,
/// phase, write slots and logits. Excludes rank-local execution slots/generations,
/// topology IDs and physical pool slots. It cannot construct executable input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalExecutionIdentity {
    sessions: Box<[u64]>,
    layout: ValidatedLayout,
}
impl LogicalExecutionIdentity {
    pub fn validate(
        batch: &ExecutionBatch,
        reservations: &[KvReservationView],
        capabilities: &ExecutionCapabilities,
        page_size: usize,
        pages: &impl DecoderKvPageView,
        sessions: &[u64],
    ) -> Result<Self> {
        ValidatedLayout::validate(batch, reservations, capabilities, page_size, pages)?
            .identity(sessions)
    }
    pub fn rows(&self) -> usize {
        self.layout.token_ids.len()
    }
    pub fn commit_projection(&self) -> KvCommitProjection {
        self.layout.projection()
    }
}
// Private proof only: no public builder can skip a validation phase.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ValidatedLayout {
    intent: ExecutionIntent,
    mode: ForwardMode,
    page_size: usize,
    token_ids: Box<[u32]>,
    positions: Box<[usize]>,
    write_slots: Box<[KvWriteSlot]>,
    sequences: Box<[LogicalSequence]>,
    row_to_sequence: Box<[usize]>,
    sequence_major_rows: Box<[usize]>,
    new_pages: Box<[KvPageId]>,
    writable_pages: Box<[KvPageId]>,
    cow_replacements: Box<[KvCowReplacement]>,
    protected_pages: Box<[KvPageId]>,
    logits_plan: LogitsPlan,
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct LogicalSequence {
    page_state_slot: StateSlot,
    page_generation: u64,
    phase: ForwardPhase,
    query: Range<usize>,
    context_len: usize,
    sequence_len: usize,
    block_table: Box<[KvPageId]>,
}
impl ValidatedLayout {
    fn validate(
        batch: &ExecutionBatch,
        reservations: &[KvReservationView],
        capabilities: &ExecutionCapabilities,
        page_size: usize,
        pages: &impl DecoderKvPageView,
    ) -> Result<Self> {
        if page_size == 0 {
            return Err(execution_error("decoder KV page size must be non-zero"));
        }
        validate_logical_layout(batch, capabilities)?;
        Self::accumulate_custody(batch, reservations, capabilities, page_size, pages)
    }

    fn accumulate_custody(
        batch: &ExecutionBatch,
        reservations: &[KvReservationView],
        capabilities: &ExecutionCapabilities,
        page_size: usize,
        pages: &impl DecoderKvPageView,
    ) -> Result<Self> {
        if reservations.len() != batch.sequences().len() {
            return Err(execution_error(format!(
                "packed decoder batch has {} sequences but {} KV reservations",
                batch.sequences().len(),
                reservations.len()
            )));
        }
        let positions = batch
            .positions()
            .iter()
            .copied()
            .map(|position| {
                usize::try_from(position)
                    .map_err(|_| execution_error("packed position cannot be represented as usize"))
            })
            .collect::<Result<Vec<_>>>()?;
        let write_slots = batch
            .kv_write_slots()
            .iter()
            .enumerate()
            .map(|(row, slot)| {
                slot.ok_or_else(|| {
                    execution_error(format!(
                        "packed decoder row {row} has no paged KV write slot"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut lowered_sequences = Vec::with_capacity(batch.sequences().len());
        let mut row_to_sequence = vec![usize::MAX; batch.len()];
        let mut expected_query_start = 0usize;
        let mut expected_block_start = 0usize;
        let mut page_state_owners = BTreeMap::new();
        let mut new_pages = Vec::new();
        let mut new_page_set = BTreeSet::new();
        let mut cow_replacements = Vec::new();
        let mut cow_replacement_set = BTreeSet::new();
        let mut mutation_owners: BTreeMap<KvPageId, usize> = BTreeMap::new();
        let mut table_readers: BTreeMap<KvPageId, BTreeSet<usize>> = BTreeMap::new();
        let mut writable_owners: BTreeMap<KvPageId, usize> = BTreeMap::new();
        let mut protected_pages = BTreeSet::new();
        for (sequence_index, (sequence, reservation)) in
            batch.sequences().iter().zip(reservations).enumerate()
        {
            let query =
                checked_range(sequence.query.clone(), batch.len(), "query", sequence_index)?;
            if query.start != expected_query_start {
                return Err(execution_error(format!(
                    "decoder sequence {sequence_index} query starts at {}, expected {expected_query_start}",
                    query.start
                )));
            }
            expected_query_start = query.end;
            for row in query.clone() {
                row_to_sequence[row] = sequence_index;
            }
            let context_len = usize::try_from(sequence.context_len).map_err(|_| {
                execution_error(format!(
                    "decoder sequence {sequence_index} context length exceeds usize"
                ))
            })?;
            let sequence_len = usize::try_from(sequence.sequence_len).map_err(|_| {
                execution_error(format!(
                    "decoder sequence {sequence_index} length exceeds usize"
                ))
            })?;
            if reservation.execution_state_slot != sequence.state_slot {
                return Err(execution_error(format!(
                    "decoder reservation {sequence_index} is bound to execution state slot {}, expected {}",
                    reservation.execution_state_slot.get(),
                    sequence.state_slot.get()
                )));
            }
            if reservation.state_slot == reservation.execution_state_slot
                && reservation.generation != reservation.execution_generation
            {
                return Err(execution_error(format!(
                    "decoder reservation {sequence_index} uses one state slot with inconsistent page/model generations {}/{}",
                    reservation.generation, reservation.execution_generation
                )));
            }
            if let Some((previous_sequence, previous_generation)) = page_state_owners.insert(
                reservation.state_slot,
                (sequence_index, reservation.generation),
            ) {
                return Err(execution_error(format!(
                    "page-manager state slot {} is shared by decoder sequences {previous_sequence} and {sequence_index} (generations {previous_generation} and {})",
                    reservation.state_slot.get(),
                    reservation.generation
                )));
            }
            if reservation.positions != (context_len..sequence_len) {
                return Err(execution_error(format!(
                    "decoder reservation {sequence_index} covers {:?}, expected {context_len}..{sequence_len}",
                    reservation.positions
                )));
            }
            let block_range = checked_range(
                sequence.block_table.clone(),
                batch.kv_block_ids().len(),
                "block table",
                sequence_index,
            )?;
            if block_range.start != expected_block_start {
                return Err(execution_error(format!(
                    "decoder sequence {sequence_index} block table starts at {}, expected {expected_block_start}",
                    block_range.start
                )));
            }
            expected_block_start = block_range.end;
            let block_table = batch.kv_block_ids()[block_range]
                .iter()
                .map(|block| KvPageId(block.get()))
                .collect::<Vec<_>>();
            if block_table.iter().collect::<BTreeSet<_>>().len() != block_table.len() {
                return Err(execution_error(format!(
                    "decoder sequence {sequence_index} block table aliases one physical page at multiple logical indices"
                )));
            }
            let required_pages = sequence_len.div_ceil(page_size);
            if block_table.len() != required_pages {
                return Err(execution_error(format!(
                    "decoder sequence {sequence_index} block table has {} pages, expected {required_pages}",
                    block_table.len()
                )));
            }
            let pages_before = context_len.div_ceil(page_size);
            let expected_new_count = required_pages.checked_sub(pages_before).ok_or_else(|| {
                execution_error(format!(
                    "decoder reservation {sequence_index} page count moved backwards"
                ))
            })?;
            if reservation.newly_allocated.len() != expected_new_count
                || block_table[pages_before..] != reservation.newly_allocated
            {
                return Err(execution_error(format!(
                    "decoder reservation {sequence_index} new pages {:?} do not match block-table suffix {:?}",
                    reservation.newly_allocated,
                    &block_table[pages_before..]
                )));
            }
            for page in &reservation.newly_allocated {
                if cow_replacement_set.contains(page) || !new_page_set.insert(*page) {
                    return Err(execution_error(format!(
                        "new decoder KV page {} is duplicated or also a COW replacement",
                        page.0
                    )));
                }
                register_mutation_owner(&mut mutation_owners, *page, sequence_index, "new page")?;
                require_page_status(pages, *page, DecoderKvPageStatus::Vacant, "new")?;
                new_pages.push(*page);
            }
            if let Some(cow) = reservation.cow_replacement {
                if context_len == 0
                    || context_len.is_multiple_of(page_size)
                    || cow.logical_page != context_len / page_size
                    || block_table.get(cow.logical_page) != Some(&cow.replacement)
                    || cow.source == cow.replacement
                {
                    return Err(execution_error(format!(
                        "decoder reservation {sequence_index} has invalid COW replacement {cow:?}"
                    )));
                }
                if new_page_set.contains(&cow.replacement)
                    || !cow_replacement_set.insert(cow.replacement)
                {
                    return Err(execution_error(format!(
                        "decoder COW replacement page {} is duplicated or also newly allocated",
                        cow.replacement.0
                    )));
                }
                register_mutation_owner(
                    &mut mutation_owners,
                    cow.replacement,
                    sequence_index,
                    "COW replacement",
                )?;
                require_page_status(
                    pages,
                    cow.source,
                    DecoderKvPageStatus::Resident,
                    "COW source",
                )?;
                require_page_status(
                    pages,
                    cow.replacement,
                    DecoderKvPageStatus::Vacant,
                    "COW replacement",
                )?;
                table_readers
                    .entry(cow.source)
                    .or_default()
                    .insert(sequence_index);
                protected_pages.insert(cow.source);
                protected_pages.insert(cow.replacement);
                cow_replacements.push(cow);
            }
            let local_cow_replacement = reservation.cow_replacement.map(|cow| cow.replacement);
            for (logical_page, page) in block_table.iter().copied().enumerate() {
                table_readers
                    .entry(page)
                    .or_default()
                    .insert(sequence_index);
                protected_pages.insert(page);
                // Exact per-sequence suffix validated above; no repeated membership scan.
                let locally_created =
                    logical_page >= pages_before || local_cow_replacement == Some(page);
                if (new_page_set.contains(&page) || cow_replacement_set.contains(&page))
                    && !locally_created
                {
                    return Err(execution_error(format!(
                        "decoder sequence {sequence_index} borrows transaction-created page {} owned by another sequence",
                        page.0
                    )));
                }
                if !locally_created {
                    require_page_status(pages, page, DecoderKvPageStatus::Resident, "block-table")?;
                }
                if logical_page >= required_pages {
                    return Err(execution_error("decoder logical page index overflow"));
                }
            }
            for row in query.clone() {
                let position = positions[row];
                let logical_page = position / page_size;
                let token_in_page = position % page_size;
                let page = *block_table.get(logical_page).ok_or_else(|| {
                    execution_error(format!(
                        "packed decoder row {row} has no logical KV page {logical_page}"
                    ))
                })?;
                let expected_write = usize::try_from(page.0)
                    .ok()
                    .and_then(|page| page.checked_mul(page_size))
                    .and_then(|base| base.checked_add(token_in_page))
                    .ok_or_else(|| execution_error("decoder KV write slot overflow"))?;
                let supplied = write_slots[row].try_as_usize().map_err(|_| {
                    execution_error(format!(
                        "packed decoder row {row} KV write slot exceeds usize"
                    ))
                })?;
                if supplied != expected_write {
                    return Err(execution_error(format!(
                        "packed decoder row {row} write slot {supplied} does not match page {} offset {token_in_page} ({expected_write})",
                        page.0
                    )));
                }
                if !new_page_set.contains(&page) && !cow_replacement_set.contains(&page) {
                    if let Some(previous) = writable_owners.insert(page, sequence_index)
                        && previous != sequence_index
                    {
                        return Err(execution_error(format!(
                            "decoder KV page {} is writable by sequences {previous} and {sequence_index}",
                            page.0
                        )));
                    }
                    register_mutation_owner(
                        &mut mutation_owners,
                        page,
                        sequence_index,
                        "writable page",
                    )?;
                }
            }
            lowered_sequences.push(LogicalSequence {
                page_state_slot: reservation.state_slot,
                page_generation: reservation.generation,
                phase: sequence.phase,
                query,
                context_len,
                sequence_len,
                block_table: block_table.into_boxed_slice(),
            });
        }
        if expected_query_start != batch.len() {
            return Err(execution_error(format!(
                "decoder queries cover {expected_query_start} of {} packed rows",
                batch.len()
            )));
        }
        if expected_block_start != batch.kv_block_ids().len() {
            return Err(execution_error(format!(
                "decoder block tables cover {expected_block_start} of {} flattened KV blocks",
                batch.kv_block_ids().len()
            )));
        }
        finalize_conflicts(&mutation_owners, &table_readers)?;
        for page in &new_pages {
            protected_pages.insert(*page);
        }
        Ok(Self {
            intent: batch.intent(),
            mode: batch.mode(),
            page_size,
            token_ids: batch.token_ids().to_vec().into_boxed_slice(),
            positions: positions.into_boxed_slice(),
            write_slots: write_slots.into_boxed_slice(),
            sequences: lowered_sequences.into_boxed_slice(),
            row_to_sequence: row_to_sequence.into_boxed_slice(),
            sequence_major_rows: (0..batch.len()).collect::<Vec<_>>().into_boxed_slice(),
            new_pages: new_pages.into_boxed_slice(),
            writable_pages: writable_owners.keys().copied().collect(),
            cow_replacements: cow_replacements.into_boxed_slice(),
            protected_pages: protected_pages.into_iter().collect(),
            logits_plan: LogitsPlan::from_validated_batch(batch, capabilities)?,
        })
    }
    fn projection(&self) -> KvCommitProjection {
        KvCommitProjection {
            page_size: self.page_size,
            new_pages: self.new_pages.clone(),
            writable_pages: self.writable_pages.clone(),
            cow_replacements: self.cow_replacements.clone(),
            protected_pages: self.protected_pages.clone(),
            row_to_sequence: self.row_to_sequence.clone(),
            positions: self.positions.clone(),
            sequences: self.sequences.iter().map(CommitSequence::from).collect(),
        }
    }
    fn identity(self, sessions: &[u64]) -> Result<LogicalExecutionIdentity> {
        if self.token_ids.is_empty() || sessions.len() != self.sequences.len() {
            return Err(execution_error(
                "logical identity requires exact nonempty session order",
            ));
        }
        Ok(LogicalExecutionIdentity {
            sessions: sessions.into(),
            layout: self,
        })
    }
}
fn validate_logical_layout(
    batch: &ExecutionBatch,
    capabilities: &ExecutionCapabilities,
) -> Result<()> {
    // Slot uniqueness/ranges without fabricated sequence states or allocation by slot ID.
    let state_bound = batch
        .sequences()
        .iter()
        .try_fold(0usize, |bound, sequence| {
            let end = sequence
                .state_slot
                .try_as_usize()
                .ok()
                .and_then(|slot| slot.checked_add(1))
                .ok_or_else(|| execution_error("decoder execution state slot overflow"))?;
            Ok::<_, Error>(bound.max(end))
        })?;
    batch.validate(state_bound, capabilities)
}
fn finalize_conflicts(
    mutation_owners: &BTreeMap<KvPageId, usize>,
    table_readers: &BTreeMap<KvPageId, BTreeSet<usize>>,
) -> Result<()> {
    for (page, owner) in mutation_owners {
        if let Some(readers) = table_readers.get(page)
            && (readers.len() != 1 || !readers.contains(owner))
        {
            return Err(execution_error(format!(
                "mutated decoder KV page {} is shared with another packed sequence without independent custody",
                page.0
            )));
        }
    }
    Ok(())
}
fn bind_local_states<S: DecoderSequence>(
    batch: &ExecutionBatch,
    reservations: &[KvReservationView],
    logical: &[LogicalSequence],
    states: &[S],
) -> Result<Box<[PackedDecoderSequence]>> {
    let mut bound = Vec::with_capacity(logical.len());
    for (index, ((sequence, reservation), logical)) in batch
        .sequences()
        .iter()
        .zip(reservations)
        .zip(logical)
        .enumerate()
    {
        let state_index = sequence
            .state_slot
            .try_as_usize()
            .map_err(|_| execution_error("decoder state slot exceeds usize"))?;
        let state = states.get(state_index).ok_or_else(|| {
            execution_error(format!(
                "decoder sequence {index} state slot {state_index} is missing from {} states",
                states.len()
            ))
        })?;
        if state.core().position() != logical.context_len {
            return Err(execution_error(format!(
                "decoder state slot {state_index} is at position {}, batch context is {}",
                state.core().position(),
                logical.context_len
            )));
        }
        if reservation.execution_generation != state.core().generation() {
            return Err(execution_error(format!(
                "decoder reservation {index} execution generation {} is stale; state slot {state_index} is generation {}",
                reservation.execution_generation,
                state.core().generation()
            )));
        }
        bound.push(PackedDecoderSequence {
            state_index,
            topology_id: state.topology_id(),
            execution_generation: reservation.execution_generation,
            page_state_slot: logical.page_state_slot,
            page_generation: logical.page_generation,
            phase: logical.phase,
            query: logical.query.clone(),
            context_len: logical.context_len,
            sequence_len: logical.sequence_len,
            block_table: logical.block_table.clone(),
        });
    }
    Ok(bound.into_boxed_slice())
}

/// Fully validated model-neutral packed decoder input and KV custody plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedDecoderBatch {
    source_batch: ExecutionBatch,
    intent: ExecutionIntent,
    mode: ForwardMode,
    page_size: usize,
    token_ids: Box<[u32]>,
    positions: Box<[usize]>,
    write_slots: Box<[KvWriteSlot]>,
    sequences: Box<[PackedDecoderSequence]>,
    row_to_sequence: Box<[usize]>,
    sequence_major_rows: Box<[usize]>,
    new_pages: Box<[KvPageId]>,
    writable_pages: Box<[KvPageId]>,
    cow_replacements: Box<[KvCowReplacement]>,
    protected_pages: Box<[KvPageId]>,
    logits_plan: LogitsPlan,
}
impl PackedDecoderBatch {
    /// Unifies `ExecutionBatch` and runtime `KvReservationView` lowering.
    ///
    /// This validates logical state, physical page layout, write destinations,
    /// reservation generations, and transaction custody before a backend transaction is
    /// created. No partial backend mutation occurs on failure.
    pub fn lower<S>(
        batch: &ExecutionBatch,
        reservations: &[KvReservationView],
        states: &[S],
        capabilities: &ExecutionCapabilities,
        page_size: usize,
        pages: &impl DecoderKvPageView,
    ) -> Result<Self>
    where
        S: DecoderSequence,
    {
        let layout =
            ValidatedLayout::validate(batch, reservations, capabilities, page_size, pages)?;
        let sequences = bind_local_states(batch, reservations, &layout.sequences, states)?;
        Ok(Self {
            source_batch: batch.clone(),
            sequences,
            intent: layout.intent,
            mode: layout.mode,
            page_size: layout.page_size,
            token_ids: layout.token_ids,
            positions: layout.positions,
            write_slots: layout.write_slots,
            row_to_sequence: layout.row_to_sequence,
            sequence_major_rows: layout.sequence_major_rows,
            new_pages: layout.new_pages,
            writable_pages: layout.writable_pages,
            cow_replacements: layout.cow_replacements,
            protected_pages: layout.protected_pages,
            logits_plan: layout.logits_plan,
        })
    }
    /// One-way logical metadata; never permission to enter a physical backend.
    pub fn commit_projection(&self) -> KvCommitProjection {
        KvCommitProjection {
            page_size: self.page_size,
            new_pages: self.new_pages.clone(),
            writable_pages: self.writable_pages.clone(),
            cow_replacements: self.cow_replacements.clone(),
            protected_pages: self.protected_pages.clone(),
            row_to_sequence: self.row_to_sequence.clone(),
            positions: self.positions.clone(),
            sequences: self
                .sequences
                .iter()
                .map(|s| CommitSequence {
                    page_state_slot: s.page_state_slot,
                    page_generation: s.page_generation,
                    context_len: s.context_len,
                    query_len: s.query.len(),
                    block_table: s.block_table.clone(),
                })
                .collect(),
        }
    }
    pub fn logical_execution_identity(&self, sessions: &[u64]) -> Result<LogicalExecutionIdentity> {
        self.logical_layout().identity(sessions)
    }
    fn logical_layout(&self) -> ValidatedLayout {
        ValidatedLayout {
            intent: self.intent,
            mode: self.mode,
            page_size: self.page_size,
            token_ids: self.token_ids.clone(),
            positions: self.positions.clone(),
            write_slots: self.write_slots.clone(),
            row_to_sequence: self.row_to_sequence.clone(),
            sequence_major_rows: self.sequence_major_rows.clone(),
            new_pages: self.new_pages.clone(),
            writable_pages: self.writable_pages.clone(),
            cow_replacements: self.cow_replacements.clone(),
            protected_pages: self.protected_pages.clone(),
            logits_plan: self.logits_plan.clone(),
            sequences: self
                .sequences
                .iter()
                .map(|s| LogicalSequence {
                    page_state_slot: s.page_state_slot,
                    page_generation: s.page_generation,
                    phase: s.phase,
                    query: s.query.clone(),
                    context_len: s.context_len,
                    sequence_len: s.sequence_len,
                    block_table: s.block_table.clone(),
                })
                .collect(),
        }
    }
    pub const fn intent(&self) -> ExecutionIntent {
        self.intent
    }
    /// Returns the exact public execution batch from which this plan was lowered.
    pub const fn source_batch(&self) -> &ExecutionBatch {
        &self.source_batch
    }
    /// Rejects execution against a batch other than the one prepared.
    pub fn validate_source_batch(&self, batch: &ExecutionBatch) -> Result<()> {
        if self.source_batch != *batch {
            return Err(execution_error(
                "decoder execution batch differs from its prepared batch",
            ));
        }
        Ok(())
    }
    pub const fn mode(&self) -> ForwardMode {
        self.mode
    }
    pub const fn page_size(&self) -> usize {
        self.page_size
    }
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }
    pub fn write_slots(&self) -> &[KvWriteSlot] {
        &self.write_slots
    }
    pub fn sequences(&self) -> &[PackedDecoderSequence] {
        &self.sequences
    }
    pub fn row_to_sequence(&self) -> &[usize] {
        &self.row_to_sequence
    }
    pub fn sequence_major_rows(&self) -> &[usize] {
        &self.sequence_major_rows
    }
    pub fn max_top_k(&self) -> usize {
        self.logits_plan
            .rows()
            .iter()
            .filter_map(|row| match row.request() {
                LogitsRequest::TopK(k) => usize::try_from(k.get()).ok(),
                LogitsRequest::None | LogitsRequest::Full => None,
            })
            .max()
            .unwrap_or(0)
    }
    pub fn execution_shape_key(&self) -> Result<ExecutionShapeKey> {
        ExecutionShapeKey::from_batch(&self.source_batch)
    }
    pub fn new_pages(&self) -> &[KvPageId] {
        &self.new_pages
    }
    pub fn writable_pages(&self) -> &[KvPageId] {
        &self.writable_pages
    }
    pub fn cow_replacements(&self) -> &[KvCowReplacement] {
        &self.cow_replacements
    }
    pub fn protected_pages(&self) -> &[KvPageId] {
        &self.protected_pages
    }
    pub const fn logits_plan(&self) -> &LogitsPlan {
        &self.logits_plan
    }
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }
    pub fn validate_states<S>(&self, states: &[S]) -> Result<()>
    where
        S: DecoderSequence,
    {
        for (sequence_index, sequence) in self.sequences.iter().enumerate() {
            let state = states.get(sequence.state_index).ok_or_else(|| {
                execution_error(format!(
                    "decoder sequence {sequence_index} state slot {} is missing from {} states",
                    sequence.state_index,
                    states.len()
                ))
            })?;
            let binding = state.core().begin_step()?;
            if state.topology_id() != sequence.topology_id {
                return Err(execution_error(format!(
                    "decoder sequence {sequence_index} state slot {} changed topology identity from {} to {}",
                    sequence.state_index,
                    sequence.topology_id.get(),
                    state.topology_id().get()
                )));
            }
            if binding.generation() != sequence.execution_generation
                || binding.committed_position() != sequence.context_len
            {
                return Err(execution_error(format!(
                    "decoder sequence {sequence_index} state slot {} is stale: expected generation/position {}/{}, got {}/{}",
                    sequence.state_index,
                    sequence.execution_generation,
                    sequence.context_len,
                    binding.generation(),
                    binding.committed_position()
                )));
            }
        }
        Ok(())
    }
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
    pub(crate) fn remap_state_indices_in_sequence_order(&mut self) {
        for (state_index, sequence) in self.sequences.iter_mut().enumerate() {
            sequence.state_index = state_index;
        }
    }
}
fn checked_range(
    range: Range<u32>,
    upper_bound: usize,
    name: &str,
    sequence_index: usize,
) -> Result<Range<usize>> {
    let start = usize::try_from(range.start).map_err(|_| {
        execution_error(format!(
            "decoder sequence {sequence_index} {name} start exceeds usize"
        ))
    })?;
    let end = usize::try_from(range.end).map_err(|_| {
        execution_error(format!(
            "decoder sequence {sequence_index} {name} end exceeds usize"
        ))
    })?;
    if start > end || end > upper_bound {
        return Err(execution_error(format!(
            "decoder sequence {sequence_index} {name} range {start}..{end} is outside 0..{upper_bound}"
        )));
    }
    Ok(start..end)
}
fn register_mutation_owner(
    owners: &mut BTreeMap<KvPageId, usize>,
    page: KvPageId,
    sequence_index: usize,
    role: &str,
) -> Result<()> {
    if let Some(previous) = owners.insert(page, sequence_index)
        && previous != sequence_index
    {
        return Err(execution_error(format!(
            "decoder {role} {} is also mutated by sequence {previous}",
            page.0
        )));
    }
    Ok(())
}
fn require_page_status(
    pages: &impl DecoderKvPageView,
    page: KvPageId,
    expected: DecoderKvPageStatus,
    role: &str,
) -> Result<()> {
    let actual = pages.page_status(page);
    if actual != expected {
        return Err(execution_error(format!(
            "decoder {role} page {} is {actual:?}, expected {expected:?}",
            page.0
        )));
    }
    Ok(())
}
fn execution_error(message: impl Into<String>) -> Error {
    Error::Execution {
        message: message.into(),
    }
}

#[cfg(test)]
mod identity_tests {
    use super::super::GenericDecoderSequenceState;
    use super::*;
    use ferrule_common::execution::{ExecutionSequence, KvBindingMode, KvBlockId, LogitsRowPolicy};
    use std::num::NonZeroU32;

    fn packed() -> PackedDecoderBatch {
        let caps = ExecutionCapabilities {
            max_batch_tokens: 4,
            max_sequences: 2,
            max_prefill_query_tokens_per_sequence: 4,
            max_decode_query_tokens_per_sequence: 4,
            max_top_k: NonZeroU32::new(4),
            supports_prefill: true,
            supports_decode: true,
            supports_mixed: true,
            full_logits_width: NonZeroU32::new(4),
            kv_binding_mode: KvBindingMode::Paged,
            logits_row_policy: LogitsRowPolicy::Any,
        };
        let batch = ExecutionBatch::new(
            ForwardMode::Decode,
            vec![3],
            vec![1],
            vec![Some(KvWriteSlot::new(15))],
            vec![LogitsRequest::Full],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Decode,
                0..1,
                1,
                2,
                0..1,
            )],
            vec![KvBlockId::new(7)],
        );
        PackedDecoderBatch::lower(
            &batch,
            &[KvReservationView {
                state_slot: StateSlot::new(9),
                execution_state_slot: StateSlot::new(0),
                positions: 1..2,
                newly_allocated: vec![],
                generation: 5,
                execution_generation: 0,
                cow_replacement: None,
            }],
            &[GenericDecoderSequenceState::with_position(1, (), ())],
            &caps,
            2,
            &|_| DecoderKvPageStatus::Resident,
        )
        .unwrap()
    }
    #[test]
    fn every_strong_identity_axis_remains_significant_after_type_split() {
        let original = packed().logical_execution_identity(&[11]).unwrap();
        let mutations: &[fn(&mut LogicalExecutionIdentity)] = &[
            |x| x.sessions[0] += 1,
            |x| x.layout.token_ids[0] += 1,
            |x| x.layout.intent = ExecutionIntent::ProvisionalVerification,
            |x| x.layout.mode = ForwardMode::Prefill,
            |x| x.layout.page_size += 1,
            |x| x.layout.positions[0] += 1,
            |x| x.layout.write_slots[0] = KvWriteSlot::new(14),
            |x| x.layout.sequences[0].phase = ForwardPhase::Prefill,
            |x| x.layout.sequences[0].page_state_slot = StateSlot::new(10),
            |x| x.layout.sequences[0].page_generation += 1,
            |x| x.layout.sequences[0].query = 1..2,
            |x| x.layout.sequences[0].context_len = 0,
            |x| x.layout.sequences[0].sequence_len += 1,
            |x| x.layout.sequences[0].block_table[0] = KvPageId(8),
            |x| x.layout.row_to_sequence[0] = 1,
            |x| x.layout.sequence_major_rows[0] = 1,
            |x| x.layout.new_pages = vec![KvPageId(9)].into(),
            |x| x.layout.writable_pages = vec![KvPageId(9)].into(),
            |x| x.layout.protected_pages = vec![KvPageId(9)].into(),
            |x| {
                x.layout.cow_replacements = vec![KvCowReplacement {
                    logical_page: 0,
                    source: KvPageId(7),
                    replacement: KvPageId(8),
                }]
                .into()
            },
            |x| x.layout.logits_plan.rows[0].request = LogitsRequest::None,
            |x| x.layout.logits_plan.full_logits_width = Some(8),
        ];
        for (axis, mutate) in mutations.iter().enumerate() {
            let mut changed = original.clone();
            mutate(&mut changed);
            assert_ne!(changed, original, "identity axis {axis}");
        }
    }

    #[test]
    fn projection_equality_exactly_matches_legacy_comparator_for_every_axis() {
        fn legacy(a: &PackedDecoderBatch, b: &PackedDecoderBatch) -> bool {
            a.page_size() == b.page_size()
                && a.new_pages() == b.new_pages()
                && a.writable_pages() == b.writable_pages()
                && a.cow_replacements() == b.cow_replacements()
                && a.protected_pages() == b.protected_pages()
                && a.row_to_sequence() == b.row_to_sequence()
                && a.positions() == b.positions()
                && a.sequences().len() == b.sequences().len()
                && a.sequences().iter().zip(b.sequences()).all(|(x, y)| {
                    x.page_state_slot() == y.page_state_slot()
                        && x.page_generation() == y.page_generation()
                        && x.context_len() == y.context_len()
                        && x.query_len() == y.query_len()
                        && x.block_table() == y.block_table()
                })
        }
        let original = packed();
        let mutations: &[fn(&mut PackedDecoderBatch)] = &[
            |x| x.token_ids[0] += 1,
            |x| x.intent = ExecutionIntent::ProvisionalVerification,
            |x| x.mode = ForwardMode::Prefill,
            |x| x.sequences[0].phase = ForwardPhase::Prefill,
            |x| x.write_slots[0] = KvWriteSlot::new(14),
            |x| x.logits_plan.rows[0].request = LogitsRequest::None,
            |x| x.sequence_major_rows[0] = 1,
            |x| x.sequences[0].state_index = 1,
            |x| x.sequences[0].execution_generation += 1,
            |x| x.sequences[0].query = 1..2,
            |x| x.sequences[0].sequence_len += 1,
            |x| x.page_size += 1,
            |x| x.new_pages = vec![KvPageId(9)].into(),
            |x| x.writable_pages = vec![KvPageId(9)].into(),
            |x| {
                x.cow_replacements = vec![KvCowReplacement {
                    logical_page: 0,
                    source: KvPageId(7),
                    replacement: KvPageId(8),
                }]
                .into()
            },
            |x| x.protected_pages = vec![KvPageId(9)].into(),
            |x| x.row_to_sequence[0] = 1,
            |x| x.positions[0] += 1,
            |x| x.sequences = vec![x.sequences[0].clone(), x.sequences[0].clone()].into(),
            |x| x.sequences[0].page_state_slot = StateSlot::new(10),
            |x| x.sequences[0].page_generation += 1,
            |x| x.sequences[0].context_len = 0,
            |x| x.sequences[0].query = 0..2,
            |x| x.sequences[0].block_table[0] = KvPageId(8),
        ];
        let mut variants = vec![original.clone()];
        for mutate in mutations {
            let mut changed = original.clone();
            mutate(&mut changed);
            variants.push(changed);
        }
        for a in &variants {
            for b in &variants {
                assert_eq!(legacy(a, b), a.commit_projection() == b.commit_projection());
            }
        }
    }
}
