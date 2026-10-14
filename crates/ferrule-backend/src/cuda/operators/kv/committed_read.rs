//! Physical half of an isolated logical committed read. No pointer or mutable
//! plane escapes this token; all pool clones see the same reader registration.
use super::*;
use crate::cuda::{CudaAsyncTransport, CudaTransferTicket};
use ferrule_common::host_kv::CommittedKvSource;

struct PageBinding {
    page: KvPageId,
    slot: u32,
    epoch: u64,
    base: [u64; 2],
    bytes: [usize; 2],
}

#[must_use = "retain until DMA and consumers quiesce"]
pub struct CudaCommittedRead {
    inner: Rc<RefCell<Inner>>,
    source: CommittedKvSource,
    nonce: u64,
    layer: usize,
    bindings: Vec<PageBinding>,
    submitted: bool,
    released: bool,
}
impl CudaPagedKvPool {
    /// Explicit trusted composition boundary. The first binding establishes the
    /// runtime-manager/pool pair; subsequent managers cannot reuse equal page IDs.
    /// It cannot infer which manager originally populated a legacy pool.
    pub fn bind_committed_logical_owner(&mut self, owner: u64) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        inner.available()?;
        inner.no_committed_reader()?;
        if owner == 0
            || !inner.entries.is_empty()
            || inner
                .committed_logical_owner
                .is_some_and(|bound| bound != owner)
        {
            return Err(error("foreign logical owner or nonquiescent pool binding"));
        }
        inner.committed_logical_owner = Some(owner);
        Ok(())
    }
    /// The caller supplies the logical authority's validated, live source.
    /// This does NOT certify runtime ownership; the runtime adapter retains that
    /// separate lease and explicitly binds this pool to its manager.
    pub fn acquire_committed_read(
        &mut self,
        source: &CommittedKvSource,
        layer: usize,
    ) -> Result<CudaCommittedRead> {
        source
            .validate_geometry()
            .map_err(|e| error(e.to_string()))?;
        let mut inner = self.inner.borrow_mut();
        inner.available()?;
        inner.no_committed_reader()?;
        if inner.committed_logical_owner != Some(source.owner) {
            return Err(error("committed pool is not bound to this logical manager"));
        }
        if !inner.entries.is_empty() {
            return Err(error(
                "committed read conflicts with active/prepared transaction",
            ));
        }
        let pool = inner.storage();
        if pool.staging_page_limit().is_some() {
            return Err(error(
                "chunked KV committed-read DMA is not supported; use transaction staging",
            ));
        }
        if source.schema.planes() != pool.planes()
            || source.schema.page_size() != pool.page_tokens()
            || layer >= pool.planes()[0].layer_count
        {
            return Err(error("committed read physical schema/layer mismatch"));
        }
        unique(&source.pages)?;
        let bindings = source
            .pages
            .iter()
            .map(|&page| {
                let slot = pool
                    .physical_slot(page)
                    .ok_or_else(|| error("committed page is not resident"))?;
                let mut base = [0; 2];
                let mut bytes = [0; 2];
                for plane in 0..2 {
                    let storage = pool
                        .plane_storage(plane)
                        .ok_or_else(|| error("committed read requires F32"))?;
                    base[plane] = storage.as_device_buffer().cu_deviceptr();
                    bytes[plane] = pool
                        .plane_storage_bytes(plane)
                        .ok_or_else(|| error("missing plane"))?;
                }
                Ok(PageBinding {
                    page,
                    slot,
                    epoch: pool
                        .slot_epoch(slot)
                        .ok_or_else(|| error("missing slot epoch"))?,
                    base,
                    bytes,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let nonce = inner.next_nonce;
        inner.next_nonce = nonce
            .checked_add(1)
            .ok_or_else(|| error("read nonce exhausted"))?;
        inner.committed_reader = Some(nonce);
        Ok(CudaCommittedRead {
            inner: self.inner.clone(),
            source: source.clone(),
            nonce,
            layer,
            bindings,
            submitted: false,
            released: false,
        })
    }
}
impl CudaCommittedRead {
    pub fn source(&self) -> &CommittedKvSource {
        &self.source
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    /// Ordinal/context equality alone does not identify the producer stream.
    pub fn validate_operator(&self, operators: &CudaOperators) -> Result<()> {
        if !std::ptr::eq(operators, self.inner.borrow().context.as_ref()) {
            return Err(error(
                "committed GQA producer belongs to another operator owner",
            ));
        }
        self.validate()
    }
    pub(in crate::cuda::operators::kv) fn context(&self) -> Rc<CudaOperators> {
        self.inner.borrow().context.clone()
    }
    pub(in crate::cuda::operators::kv) fn nonce(&self) -> u64 {
        self.nonce
    }
    pub(in crate::cuda::operators::kv) fn identity(&self) -> Rc<()> {
        self.inner.borrow().identity.clone()
    }
    pub(in crate::cuda::operators::kv) fn validate(&self) -> Result<()> {
        let inner = self.inner.borrow();
        inner.available()?;
        if self.released || inner.committed_reader != Some(self.nonce) || !inner.entries.is_empty()
        {
            return Err(error("stale committed physical reader"));
        }
        let pool = inner.storage();
        for b in &self.bindings {
            if pool.physical_slot(b.page) != Some(b.slot)
                || pool.slot_epoch(b.slot) != Some(b.epoch)
            {
                return Err(error("committed page mapping/epoch changed"));
            }
            for plane in 0..2 {
                let storage = pool
                    .plane_storage(plane)
                    .ok_or_else(|| error("missing F32 plane"))?;
                if storage.as_device_buffer().cu_deviceptr() != b.base[plane]
                    || pool.plane_storage_bytes(plane) != Some(b.bytes[plane])
                {
                    return Err(error("committed allocation changed"));
                }
            }
        }
        Ok(())
    }
    pub(in crate::cuda::operators::kv) fn download(
        &mut self,
        transport: &mut CudaAsyncTransport,
        plane: usize,
        start: usize,
        tokens: usize,
        producer: &CudaComputeEvent,
    ) -> Result<CudaTransferTicket> {
        self.validate()?;
        let end = start
            .checked_add(tokens)
            .ok_or_else(|| error("read range overflow"))?;
        let page_tokens = self.source.schema.page_size();
        if plane >= 2
            || tokens == 0
            || start < self.source.positions.start
            || end > self.source.positions.end
            || start / page_tokens != (end - 1) / page_tokens
        {
            return Err(error("chunk crosses committed range or page boundary"));
        }
        let inner = self.inner.borrow();
        let pool = inner.storage();
        let page = self.source.pages[start / page_tokens];
        let descriptor = pool.planes()[plane];
        let storage = pool
            .plane_storage(plane)
            .ok_or_else(|| error("missing F32 plane"))?;
        let slot = pool
            .physical_slot(page)
            .ok_or_else(|| error("page no longer resident"))?;
        let range = crate::cuda::operators::kv::read_protocol::committed_chunk_range(
            crate::cuda::operators::kv::read_protocol::CommittedChunk {
                slot,
                layer: self.layer,
                page_tokens,
                plane: descriptor,
                row: start % page_tokens,
                tokens,
            },
            storage.len(),
        )?;
        // Mark before the first driver call: an error can retain a transport slot.
        self.submitted = true;
        transport
            .submit_d2h_f32_after(storage, range.start, range.len(), producer)
            .map_err(|e| error(e.to_string()))
    }
    pub(in crate::cuda::operators::kv) fn release_after_quiescence(&mut self) -> Result<()> {
        self.validate()?;
        self.inner.borrow_mut().committed_reader = None;
        self.released = true;
        Ok(())
    }
}
impl Drop for CudaCommittedRead {
    fn drop(&mut self) {
        // Construction rejection before any source DMA is a proven no-submit.
        // Once submitted, only the placement/reaper can clear the registration.
        if !self.submitted && !self.released {
            let _ = self.release_after_quiescence();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_common::execution::{KvElementType, KvLayoutSchemaIdentity, StateSlot};

    fn fixture(ops: Rc<CudaOperators>) -> (CudaPagedKvPool, CommittedKvSource) {
        let planes = [
            KvPlaneDescriptor::new("standard_gqa.key", 2, 1, KvElementType::F32),
            KvPlaneDescriptor::new("standard_gqa.value", 2, 1, KvElementType::F32),
        ];
        let mut pool = CudaPagedKvPool::new(ops.clone(), &planes, 4, 2).unwrap();
        let tx = pool
            .prepare(PagedKvPrepare {
                transaction: ExecutionTransactionId::new(1).unwrap(),
                new_pages: &[KvPageId(1)],
                writable_pages: &[],
                cow_replacements: &[],
                protected_pages: &[],
            })
            .unwrap();
        ops.sync_stream().unwrap();
        assert_eq!(pool.commit(&mut Some(tx)).unwrap(), KvEndProgress::Complete);
        pool.bind_committed_logical_owner(7).unwrap();
        let source = CommittedKvSource {
            owner: 7,
            state_slot: StateSlot::new(0),
            generation: 1,
            incarnation: 1,
            schema: KvLayoutSchemaIdentity::new(planes.to_vec(), 4, 8),
            committed_tokens: 3,
            positions: 0..3,
            pages: vec![KvPageId(1)],
        };
        (pool, source)
    }

    #[test]
    #[ignore = "requires an explicitly selected idle CUDA device"]
    fn equal_ids_and_nonces_do_not_share_physical_reader_custody() {
        let ops = Rc::new(CudaOperators::new().unwrap());
        let (mut a, source) = fixture(ops.clone());
        let (mut b, other_source) = fixture(ops.clone());
        assert_eq!(source, other_source);
        let mut first = a.acquire_committed_read(&source, 0).unwrap();
        let second = b.acquire_committed_read(&source, 0).unwrap();
        assert_eq!(first.nonce(), second.nonce());
        assert!(!Rc::ptr_eq(&first.identity(), &second.identity()));
        first.validate_operator(&ops).unwrap();
        let foreign_ops = CudaOperators::new().unwrap();
        assert!(first.validate_operator(&foreign_ops).is_err());
        assert!(a.clone().acquire_committed_read(&source, 0).is_err());
        first.release_after_quiescence().unwrap();
        let current = a.acquire_committed_read(&source, 0).unwrap();
        assert_ne!(first.nonce(), current.nonce());
        assert!(first.release_after_quiescence().is_err());
        drop(first);
        assert!(a.clone().release(&source.pages).is_err());
        assert!(b.clone().release(&source.pages).is_err());
        // No DMA was submitted: dropping each construction token is a no-submit ACK.
        drop(current);
        a.release(&source.pages).unwrap();
        second.validate().unwrap();
        assert!(b.clone().shutdown().is_err());
        drop(second);
        b.release(&source.pages).unwrap();
        a.shutdown().unwrap();
        b.shutdown().unwrap();
    }

    #[test]
    #[ignore = "requires an explicitly selected idle CUDA device"]
    fn recycled_same_slot_epoch_rejects_release_without_consuming_pin() {
        let ops = Rc::new(CudaOperators::new().unwrap());
        let (mut pool, source) = fixture(ops);
        let mut read = pool.acquire_committed_read(&source, 0).unwrap();
        let before = read.bindings[0].epoch;
        let page = source.pages[0];
        let slot = read.bindings[0].slot;
        // Bypass the public guard to model an ABA regression. No read DMA has
        // been submitted, and the seeded allocation was positively fenced.
        let epoch = {
            let mut inner = pool.inner.borrow_mut();
            let storage = inner.storage_mut();
            assert_eq!(storage.detach_transaction_page(page), Some(slot));
            storage.recycle_transaction_slots(vec![slot]);
            assert_eq!(storage.take_transaction_slots(1).unwrap(), vec![slot]);
            assert_eq!(storage.install_transaction_slot(page, slot), None);
            storage.slot_epoch(slot).unwrap()
        };
        assert!(epoch > before);
        assert_eq!(pool.physical_slot(page), Some(slot));
        assert!(read.validate().is_err());
        assert!(read.release_after_quiescence().is_err());
        assert!(pool.clone().release(&source.pages).is_err());
        assert!(pool.clone().acquire_committed_read(&source, 0).is_err());
        // Test-only repair of the injected stale witness, not a public recovery API.
        read.bindings[0].epoch = epoch;
        read.release_after_quiescence().unwrap();
        pool.release(&source.pages).unwrap();
        pool.shutdown().unwrap();
    }
}
