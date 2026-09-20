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
    {
        return Err(error(
            "CUDA GQA requires matching standard key/value plane shapes",
        ));
    }
    for plane in planes {
        if plane.element_type != KvElementType::F32 {
            return Err(error(
                "unsupported CUDA transactional GQA KV dtype: only F32 is implemented; BF16 planes are not supported",
            ));
        }
        if plane.elements_per_token == 0 || plane.layer_count == 0 {
            return Err(error("CUDA GQA plane dimensions must be positive"));
        }
        for dim in [page_size, plane.elements_per_token, plane.layer_count] {
            u32::try_from(dim).map_err(|_| error("CUDA GQA layout exceeds u32 ABI"))?;
        }
        plane
            .checked_page_bytes(page_size)
            .and_then(|bytes| bytes.checked_mul(capacity))
            .ok_or_else(|| error("CUDA GQA plane capacity overflow"))?;
    }
    Ok(())
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
