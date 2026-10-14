//! CUDA-independent transaction mapping validation (also tested without CUDA).

use crate::cpu::{PagedKvBatch, PagedKvPrepare};
use ferrule_common::execution::{KvElementType, KvPageId, KvPlaneDescriptor};
use ferrule_common::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn error(message: impl Into<String>) -> Error {
    Error::Execution {
        message: message.into(),
    }
}

pub(super) fn unique(pages: &[KvPageId]) -> Result<BTreeSet<KvPageId>> {
    let set: BTreeSet<_> = pages.iter().copied().collect();
    if set.len() != pages.len() {
        return Err(error("duplicate CUDA KV page"));
    }
    Ok(set)
}

pub(super) fn validate_schema(
    planes: &[KvPlaneDescriptor],
    page_size: usize,
    capacity: usize,
) -> Result<()> {
    if planes.len() != 2 || page_size == 0 || capacity == 0 || capacity > i32::MAX as usize {
        return Err(error(
            "CUDA GQA requires two planes, positive page size and i32-bounded slots",
        ));
    }
    if planes[0].name != "standard_gqa.key"
        || planes[1].name != "standard_gqa.value"
        || planes[0].elements_per_token != planes[1].elements_per_token
        || planes[0].layer_count != planes[1].layer_count
        || planes[0].element_type != planes[1].element_type
    {
        return Err(error(
            "CUDA GQA requires matching standard key/value plane shapes",
        ));
    }
    let mut total_bytes = 0usize;
    for plane in planes {
        if plane.elements_per_token == 0 || plane.layer_count == 0 {
            return Err(error("CUDA GQA plane dimensions must be positive"));
        }
        for dim in [page_size, plane.elements_per_token, plane.layer_count] {
            u32::try_from(dim).map_err(|_| error("CUDA GQA layout exceeds u32 ABI"))?;
        }
        let bytes = plane
            .checked_page_bytes(page_size)
            .and_then(|bytes| bytes.checked_mul(capacity))
            .ok_or_else(|| error("CUDA GQA plane capacity overflow"))?;
        total_bytes = total_bytes
            .checked_add(bytes)
            .ok_or_else(|| error("CUDA GQA total capacity overflow"))?;
    }
    Ok(())
}

/// Chunked staging and HostKV operators still have an explicitly F32 ABI.
pub(super) fn validate_plane_type(
    planes: &[KvPlaneDescriptor],
    dtype: KvElementType,
) -> Result<()> {
    if planes.len() != 2 || planes.iter().any(|plane| plane.element_type != dtype) {
        return Err(error(format!(
            "CUDA transactional GQA requires two {dtype:?} planes"
        )));
    }
    Ok(())
}

pub(super) fn validate_storage_mode(planes: &[KvPlaneDescriptor], chunked: bool) -> Result<()> {
    if chunked {
        validate_plane_type(planes, KvElementType::F32).map_err(|_| {
            error("BF16 transactional KV does not support F32 chunked/HostKV staging")
        })?;
    }
    Ok(())
}

/// Produce only transaction-authenticated row metadata. Duplicate physical
/// writers would race the append kernel even when their sequence IDs differ.
pub(super) fn bf16_row_metadata(
    batch: &PagedKvBatch,
    page_size: usize,
    slots: &[i32],
    offsets: &[i32],
) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
    batch.validate(page_size)?;
    if offsets.len() != batch.sequences.len() + 1 {
        return Err(error("BF16 packed sequence offsets mismatch"));
    }
    let mut writers = BTreeSet::new();
    let mut ids = Vec::with_capacity(batch.row_positions.len());
    let mut positions = Vec::with_capacity(batch.row_positions.len());
    let mut lengths = Vec::with_capacity(batch.row_positions.len());
    for (&sequence, &position) in batch.row_sequence_ids.iter().zip(&batch.row_positions) {
        let start =
            usize::try_from(offsets[sequence]).map_err(|_| error("negative BF16 offset"))?;
        let end =
            usize::try_from(offsets[sequence + 1]).map_err(|_| error("negative BF16 offset"))?;
        let index = start
            .checked_add(position / page_size)
            .filter(|&index| start <= end && index < end && end <= slots.len())
            .ok_or_else(|| error("BF16 row outside packed slots"))?;
        let slot = slots[index];
        if slot < 0 || !writers.insert((slot, position % page_size)) {
            return Err(error(
                "BF16 append rows alias or have an invalid physical slot",
            ));
        }
        let i32_dim = |n| i32::try_from(n).map_err(|_| error("BF16 row metadata exceeds i32 ABI"));
        ids.push(i32_dim(sequence)?);
        positions.push(i32_dim(position)?);
        lengths.push(i32_dim(batch.sequences[sequence].sequence_len)?);
    }
    Ok((ids, positions, lengths))
}

/// Every write has a distinct provisional slot, including an existing writable
/// page. Sources and declared reads are pinned even if omitted from the batch.
#[derive(Debug)]
pub(super) struct Plan {
    pub original: BTreeMap<KvPageId, Option<u32>>,
    pub copies: BTreeMap<KvPageId, Option<u32>>,
    pub staged: BTreeMap<KvPageId, u32>,
}

impl Plan {
    pub fn new(
        request: PagedKvPrepare<'_>,
        resident: impl Fn(KvPageId) -> Option<u32>,
        preempted: impl Fn(KvPageId) -> bool,
    ) -> Result<Self> {
        let new = unique(request.new_pages)?;
        let writable = unique(request.writable_pages)?;
        let mut protected = unique(request.protected_pages)?;
        let mut copies = BTreeMap::new();
        for &page in &new {
            copies.insert(page, None);
        }
        for &page in &writable {
            let slot = resident(page).ok_or_else(|| error("CUDA writable page is not resident"))?;
            if copies.insert(page, Some(slot)).is_some() {
                return Err(error("overlapping CUDA mutation sets"));
            }
        }
        for cow in request.cow_replacements {
            if cow.source == cow.replacement {
                return Err(error("CUDA COW source equals replacement"));
            }
            let slot =
                resident(cow.source).ok_or_else(|| error("CUDA COW source is not resident"))?;
            if copies.insert(cow.replacement, Some(slot)).is_some() {
                return Err(error("duplicate CUDA COW destination"));
            }
            protected.insert(cow.source);
        }
        protected.extend(copies.keys().copied());
        let mut original = BTreeMap::new();
        for page in protected {
            let slot = resident(page);
            if copies.contains_key(&page) && !writable.contains(&page) {
                if slot.is_some() || preempted(page) {
                    return Err(error("CUDA new/COW page is already occupied"));
                }
            } else if slot.is_none() {
                return Err(error("CUDA protected read has no resident mapping"));
            }
            original.insert(page, slot);
        }
        Ok(Self {
            original,
            copies,
            staged: BTreeMap::new(),
        })
    }

    pub fn conflicts(&self, other: &Self, other_sealed: bool) -> bool {
        self.original.keys().any(|page| {
            other.copies.contains_key(page) || (other_sealed && other.original.contains_key(page))
        }) || self
            .copies
            .keys()
            .any(|page| other.original.contains_key(page))
    }

    pub fn bind_slots(&mut self, slots: Vec<u32>) {
        assert_eq!(slots.len(), self.copies.len());
        self.staged = self.copies.keys().copied().zip(slots).collect();
    }

    pub fn slot(&self, page: KvPageId) -> Option<u32> {
        // Never fall back to resident for undeclared reads.
        let original = self.original.get(&page)?;
        self.staged.get(&page).copied().or(*original)
    }

    pub fn validate(
        &self,
        capacity: usize,
        resident: impl Fn(KvPageId) -> Option<u32>,
        preempted: impl Fn(KvPageId) -> bool,
    ) -> Result<()> {
        if self.staged.keys().ne(self.copies.keys()) {
            return Err(error("CUDA pending mapping is incomplete"));
        }
        let mut slots = BTreeSet::new();
        for (&page, &original) in &self.original {
            if resident(page) != original || (original.is_none() && preempted(page)) {
                return Err(error("CUDA resident/COW mapping changed before install"));
            }
            let slot = self
                .slot(page)
                .ok_or_else(|| error("CUDA declared page has no physical slot"))?;
            if slot as usize >= capacity || !slots.insert(slot) {
                return Err(error("CUDA staged slots alias or exceed capacity"));
            }
        }
        if self
            .staged
            .values()
            .any(|slot| self.original.values().any(|old| *old == Some(*slot)))
        {
            return Err(error("CUDA provisional write aliases committed storage"));
        }
        Ok(())
    }

    pub fn packed_slots(
        &self,
        batch: &PagedKvBatch,
        page_size: usize,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        batch.validate(page_size)?;
        let mut slots = Vec::new();
        let mut offsets = vec![0];
        for sequence in &batch.sequences {
            if sequence.sequence_len.div_ceil(page_size) > sequence.block_table.len() {
                return Err(error("CUDA packed sequence history is missing pages"));
            }
            for &page in &sequence.block_table {
                let slot = self
                    .slot(page)
                    .ok_or_else(|| error("CUDA packed page is outside transaction read set"))?;
                slots.push(i32::try_from(slot).map_err(|_| error("CUDA slot exceeds i32 ABI"))?);
            }
            offsets.push(
                i32::try_from(slots.len())
                    .map_err(|_| error("CUDA packed block offsets exceed i32 ABI"))?,
            );
        }
        for (&seq, &position) in batch.row_sequence_ids.iter().zip(&batch.row_positions) {
            let page = batch.sequences[seq].block_table[position / page_size];
            if !self.staged.contains_key(&page) {
                return Err(error("CUDA packed write is not a staged mutation"));
            }
        }
        Ok((slots, offsets))
    }
}
