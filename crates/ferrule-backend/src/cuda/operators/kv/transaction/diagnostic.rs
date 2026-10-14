//! Synchronous native-word export, isolated from provisional transaction views.
use super::*;
use ferrule_common::execution::KvElementType;
use ferrule_common::host_kv::CommittedKvSource;
use std::ops::Range;

/// One page-local portion of a full committed prefix. Runtime must keep the
/// logical authority's validated source lease alive throughout export. This is
/// physical completion evidence, not a replacement for that logical lease.
pub struct CudaCommittedBf16KvChunk<'a> {
    pub source: &'a CommittedKvSource,
    pub plane: usize,
    pub layer: usize,
    pub positions: Range<usize>,
    pub page: KvPageId,
    pub slot: u32,
    pub slot_epoch: u64,
    pub elements_per_token: usize,
    pub words: &'a [u16],
}

fn validate_export(
    source: &CommittedKvSource,
    planes: &[KvPlaneDescriptor],
    page_tokens: usize,
    layer: usize,
    max_chunk_words: usize,
    owner: Option<u64>,
    quiescent: bool,
) -> Result<usize> {
    source
        .validate_geometry()
        .map_err(|e| error(e.to_string()))?;
    if !quiescent
        || owner != Some(source.owner)
        || source.owner == 0
        || source.incarnation == 0
        || source.positions != (0..source.committed_tokens)
        || source.schema.planes() != planes
        || source.schema.page_size() != page_tokens
        || planes.len() != 2
        || planes
            .iter()
            .any(|p| p.element_type != KvElementType::Bf16 || layer >= p.layer_count)
        || planes[0].elements_per_token != planes[1].elements_per_token
    {
        return Err(error(
            "BF16 trace requires bound quiescent full committed source and matching schema",
        ));
    }
    if source
        .pages
        .iter()
        .enumerate()
        .any(|(i, page)| source.pages[..i].contains(page))
    {
        return Err(error("duplicate committed trace page"));
    }
    let width = planes[0].elements_per_token;
    let chunk_tokens = max_chunk_words / width;
    if chunk_tokens == 0
        || max_chunk_words
            .checked_mul(2)
            .is_none_or(|b| b > isize::MAX as usize)
    {
        return Err(error("BF16 trace chunk cap must hold at least one token"));
    }
    Ok(chunk_tokens.min(page_tokens))
}

/// Peak payload/metadata allocated by export, excluding the caller-owned source,
/// callback output storage, and existing resident KV. No device or pinned grant.
pub fn committed_bf16_kv_trace_host_bytes(
    source: &CommittedKvSource,
    max_chunk_words: usize,
) -> Result<usize> {
    source
        .validate_geometry()
        .map_err(|e| error(e.to_string()))?;
    source
        .pages
        .len()
        .checked_mul(std::mem::size_of::<(KvPageId, u32, u64)>())
        .and_then(|metadata| {
            max_chunk_words
                .checked_mul(2)
                .and_then(|payload| payload.checked_add(metadata))
        })
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| error("committed BF16 trace host bound overflow"))
}

impl CudaPagedKvPool {
    /// Full retained prefix + committed append, streamed in page-local chunks.
    /// No device/pinned allocation or BF16 expansion. At most max_chunk_words*2
    /// host bytes live in this method. Rejects active/prepared/Unknown custody,
    /// foreign logical owners, partial frontiers, and chunked/offloaded storage.
    ///
    /// Not reentrant: the shared physical owner is exclusively borrowed until
    /// callbacks return. Call only after journal publication, with a live source
    /// lease revalidated by the bound logical manager. A descriptor alone does
    /// not prove runtime publication, generation freshness, or a live lease.
    /// Generation zero is valid; equality to the current generation is checked
    /// by the logical authority, not inferred from a nonzero integer here.
    /// Readback Unknown poisons this owner. Callbacks must only consume host
    /// words; they must not submit GPU work or re-enter any clone of this pool.
    pub fn export_committed_bf16_kv(
        &mut self,
        source: &CommittedKvSource,
        layer: usize,
        max_chunk_words: usize,
        mut observe: impl FnMut(CudaCommittedBf16KvChunk<'_>) -> Result<()>,
    ) -> Result<()> {
        let mut inner = self
            .inner
            .try_borrow_mut()
            .map_err(|_| error("BF16 committed trace reentry"))?;
        inner.available()?;
        inner.no_committed_reader()?;
        let pool = inner.storage();
        let chunk_tokens = validate_export(
            source,
            pool.planes(),
            pool.page_tokens(),
            layer,
            max_chunk_words,
            inner.committed_logical_owner,
            inner.entries.is_empty(),
        )?;
        if pool.staging_page_limit().is_some() {
            return Err(error("BF16 committed trace requires resident fixed planes"));
        }
        committed_bf16_kv_trace_host_bytes(source, max_chunk_words)?;
        // Validate every page and plane before the first download/callback.
        let mut bindings = Vec::with_capacity(source.pages.len());
        for &page in &source.pages {
            let slot = pool
                .physical_slot(page)
                .ok_or_else(|| error("trace page is not resident"))?;
            let epoch = pool
                .slot_epoch(slot)
                .ok_or_else(|| error("trace page epoch absent"))?;
            for plane in 0..2 {
                let storage = pool
                    .plane_bf16_storage(plane)
                    .ok_or_else(|| error("trace plane not BF16"))?;
                let descriptor = pool.planes()[plane];
                let end = (slot as usize)
                    .checked_add(1)
                    .and_then(|n| n.checked_mul(descriptor.layer_count))
                    .and_then(|n| n.checked_mul(pool.page_tokens()))
                    .and_then(|n| n.checked_mul(descriptor.elements_per_token))
                    .ok_or_else(|| error("trace physical range overflow"))?;
                if end > storage.len() {
                    return Err(error("trace physical range exceeds storage"));
                }
            }
            bindings.push((page, slot, epoch));
        }
        let context = Rc::clone(&inner.context);
        let page_tokens = source.schema.page_size();
        for (page_index, (page, slot, slot_epoch)) in bindings.into_iter().enumerate() {
            let page_start = page_index * page_tokens;
            let page_end = (page_start + page_tokens).min(source.committed_tokens);
            for plane in 0..2 {
                let descriptor = source.schema.planes()[plane];
                let width = descriptor.elements_per_token;
                for start in (page_start..page_end).step_by(chunk_tokens) {
                    let end = (start + chunk_tokens).min(page_end);
                    let offset = ((slot as usize * descriptor.layer_count + layer) * page_tokens
                        + start % page_tokens)
                        * width;
                    // A failed owned download retains its host destination in the
                    // driver layer; poison the physical pool before submission.
                    inner.poisoned = true;
                    let words = inner.storage().download_bf16_words_range(
                        &context,
                        plane,
                        offset,
                        (end - start) * width,
                    )?;
                    inner.poisoned = false;
                    observe(CudaCommittedBf16KvChunk {
                        source,
                        plane,
                        layer,
                        positions: start..end,
                        page,
                        slot,
                        slot_epoch,
                        elements_per_token: width,
                        words: &words,
                    })?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_common::execution::{KvLayoutSchemaIdentity, StateSlot};
    fn source() -> CommittedKvSource {
        CommittedKvSource {
            owner: 7,
            generation: 2,
            incarnation: 1,
            state_slot: StateSlot::new(3),
            schema: KvLayoutSchemaIdentity::new(
                vec![
                    KvPlaneDescriptor::new("standard_gqa.key", 8, 2, KvElementType::Bf16),
                    KvPlaneDescriptor::new("standard_gqa.value", 8, 2, KvElementType::Bf16),
                ],
                4,
                32,
            ),
            committed_tokens: 5,
            positions: 0..5,
            pages: vec![KvPageId(4), KvPageId(9)],
        }
    }
    fn validate(
        s: &CommittedKvSource,
        cap: usize,
        owner: Option<u64>,
        idle: bool,
    ) -> Result<usize> {
        validate_export(s, s.schema.planes(), 4, 1, cap, owner, idle)
    }
    #[test]
    fn committed_bf16_trace_requires_full_bound_quiescent_frontier() {
        let mut s = source();
        assert_eq!(validate(&s, 16, Some(7), true).unwrap(), 2);
        assert!(validate(&s, 7, Some(7), true).is_err());
        assert!(validate(&s, 16, Some(8), true).is_err());
        assert!(validate(&s, 16, Some(7), false).is_err());
        s.positions = 4..5;
        assert!(validate(&s, 16, Some(7), true).is_err());
        s.positions = 0..5;
        s.pages[1] = s.pages[0];
        assert!(validate(&s, 16, Some(7), true).is_err());
    }
    #[test]
    fn committed_bf16_trace_checks_schema_incarnation_and_allows_generation_zero() {
        let mut s = source();
        assert!(validate_export(&s, s.schema.planes(), 8, 1, 16, Some(7), true).is_err());
        assert!(validate_export(&s, s.schema.planes(), 4, 2, 16, Some(7), true).is_err());
        assert!(validate(&s, usize::MAX, Some(7), true).is_err());
        // Generation is an opaque logical-authority value, including zero.
        s.generation = 0;
        assert_eq!(validate(&s, 16, Some(7), true).unwrap(), 2);
        s.incarnation = 0;
        assert!(validate(&s, 16, Some(7), true).is_err());
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use ferrule_common::execution::{KvLayoutSchemaIdentity, StateSlot};
    #[test]
    #[ignore = "explicit GPU grant; native committed-prefix readback and provisional exclusion"]
    fn committed_bf16_native_prefix_append_and_active_rejection() {
        let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
        let planes = [
            KvPlaneDescriptor::new("standard_gqa.key", 8, 2, KvElementType::Bf16),
            KvPlaneDescriptor::new("standard_gqa.value", 8, 2, KvElementType::Bf16),
        ];
        let mut pool = CudaPagedKvPool::new(Rc::clone(&ops), &planes, 4, 3).unwrap();
        pool.bind_committed_logical_owner(7).unwrap();
        let source = CommittedKvSource {
            owner: 7,
            state_slot: StateSlot::new(3),
            generation: 0,
            incarnation: 1,
            schema: KvLayoutSchemaIdentity::new(planes.to_vec(), 4, 32),
            committed_tokens: 5,
            positions: 0..5,
            pages: vec![KvPageId(4), KvPageId(9)],
        };
        let tx = pool
            .prepare(PagedKvPrepare {
                transaction: ExecutionTransactionId::new(1).unwrap(),
                new_pages: &source.pages,
                writable_pages: &[],
                cow_replacements: &[],
                protected_pages: &[],
            })
            .unwrap();
        assert!(
            pool.export_committed_bf16_kv(&source, 1, 16, |_| panic!("provisional callback"))
                .is_err()
        );
        ops.sync_stream().unwrap();
        assert_eq!(pool.commit(&mut Some(tx)).unwrap(), KvEndProgress::Complete);
        // Seed actual native pages with an independent position/plane/channel
        // word pattern. Export must not include unused tail tokens or layer 0.
        {
            let mut inner = pool.inner.borrow_mut();
            for (page_index, &page) in source.pages.iter().enumerate() {
                let slot = inner.storage().physical_slot(page).unwrap() as usize;
                for plane in 0..2 {
                    let words = (0..32)
                        .map(|i| 0x3e00 + (plane * 256 + page_index * 32 + i) as u16)
                        .collect::<Vec<_>>();
                    inner
                        .storage_mut()
                        .overwrite_bf16_words_range(&ops, plane, (slot * 2 + 1) * 32, &words)
                        .unwrap();
                }
            }
        }
        // A serialization failure after completed D2H is ordinary; no release
        // or retry is permitted for an actual Unknown readback instead.
        assert!(
            pool.export_committed_bf16_kv(&source, 1, 16, |_| Err(error(
                "injected writer rejection"
            )))
            .is_err()
        );
        assert!(!pool.has_unresolved_custody());
        let before = ops.counters();
        let mut outputs = [Vec::new(), Vec::new()];
        let mut chunks = 0;
        pool.export_committed_bf16_kv(&source, 1, 16, |chunk| {
            assert_eq!(chunk.source, &source);
            assert_eq!(chunk.words.len(), chunk.positions.len() * 8);
            assert!(chunk.words.len() <= 16);
            assert_eq!(outputs[chunk.plane].len() / 8, chunk.positions.start);
            outputs[chunk.plane].extend_from_slice(chunk.words);
            chunks += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(chunks, 6);
        for (plane, words) in outputs.iter().enumerate() {
            assert_eq!(
                words,
                &(0..40)
                    .map(|i| 0x3e00 + (plane * 256 + i) as u16)
                    .collect::<Vec<_>>()
            );
        }
        let after = ops.counters();
        assert_eq!(
            after.device_allocation_attempts,
            before.device_allocation_attempts
        );
        assert_eq!(after.kernel_launches, before.kernel_launches);
        pool.release(&source.pages).unwrap();
        assert!(
            pool.export_committed_bf16_kv(&source, 1, 16, |_| panic!("released callback"))
                .is_err()
        );
        pool.shutdown().unwrap();
    }
}
