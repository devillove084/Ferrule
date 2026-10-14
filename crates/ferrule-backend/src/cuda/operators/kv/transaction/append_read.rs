//! Internal transaction-prefix DMA. No pointer export or decoder enable flag.
//! Trusted composition must retain the runtime reservation and actual Q custody.
use super::prefix_read::{PrefixReadBinding, PrefixReadHandle};
use super::*;
use ferrule_common::host_kv::HostKvAppendReadWitness;

#[must_use = "finish through the owning transaction; drop is not an ACK"]
pub(crate) struct CudaAppendPrefix<'a> {
    pub(super) inner: Rc<RefCell<Inner>>,
    pub(super) transaction: CudaPagedKvTransaction,
    _borrow: std::marker::PhantomData<&'a CudaPagedKvTransaction>,
    pub(super) handle: PrefixReadHandle,
    witness: HostKvAppendReadWitness,
    layer: usize,
    allocation: [(u64, usize); 2],
}
fn allocation(storage: &CudaKvPagePool) -> Result<[(u64, usize); 2]> {
    let mut result = [(0, 0); 2];
    for (plane, entry) in result.iter_mut().enumerate() {
        let buffer = storage
            .plane_storage(plane)
            .ok_or_else(|| error("prefix read requires F32 planes"))?;
        *entry = (buffer.as_device_buffer().cu_deviceptr(), buffer.len());
    }
    Ok(result)
}
impl CudaPagedKvPool {
    /// Caller must pass a witness from the still-borrowed logical append
    /// reservation. Matching public metadata alone never authenticates runtime
    /// ownership. Kept crate-internal; the bridge requires borrowed logical authority.
    pub(crate) fn derive_append_prefix<'a>(
        &mut self,
        transaction: &'a CudaPagedKvTransaction,
        witness: &HostKvAppendReadWitness,
        layer: usize,
    ) -> Result<CudaAppendPrefix<'a>> {
        let mut inner = self.inner.borrow_mut();
        if inner.storage().staging_page_limit().is_some() {
            return Err(error(
                "chunked KV prefix DMA is not supported; use transaction staging",
            ));
        }
        let entry = inner.entry(transaction)?;
        entry.prefix_read.ensure_idle()?;
        if !entry.entered
            || entry.phase != Phase::Active
            || witness.transaction() != transaction.id
            || inner.committed_logical_owner != Some(witness.source().owner)
            || witness.source().schema.planes() != inner.storage().planes()
            || witness.source().schema.page_size() != inner.storage().page_tokens()
        {
            return Err(error(
                "prefix capability requires matching entered append custody",
            ));
        }
        entry.plan.validate(
            inner.storage().stats().allocated_slots,
            |p| inner.storage().physical_slot(p),
            |p| inner.storage().has_snapshot(p),
        )?;
        let allocations = allocation(inner.storage())?;
        let producer = match inner.context.record_compute_event() {
            Ok(fence) => fence,
            Err(error) => {
                inner.poisoned = true;
                return Err(error);
            }
        };
        let Inner {
            entries, storage, ..
        } = &mut *inner;
        let entry = entries
            .get_mut(&transaction.id)
            .expect("validated transaction");
        let storage = storage.as_ref().expect("owned storage");
        let handle = entry.prefix_read.derive(
            PrefixReadBinding {
                witness,
                layer,
                nonce: entry.nonce,
                view_epoch: entry.epoch,
            },
            entry
                .batch
                .as_ref()
                .ok_or_else(|| error("prefix read has no entered batch"))?,
            &entry.plan,
            |slot| storage.slot_epoch(slot),
            super::append_dma::AppendEvidence::Producer(producer),
        )?;
        Ok(CudaAppendPrefix {
            inner: self.inner.clone(),
            transaction: CudaPagedKvTransaction {
                id: transaction.id,
                identity: transaction.identity.clone(),
                nonce: transaction.nonce,
            },
            _borrow: std::marker::PhantomData,
            handle,
            witness: witness.clone(),
            layer,
            allocation: allocations,
        })
    }
}
impl CudaAppendPrefix<'_> {
    /// Only an authenticated backend view may detach the Rust handle borrow;
    /// the same Entry still pins every page and blocks leave/commit/abort.
    pub(super) fn into_owned(self) -> CudaAppendPrefix<'static> {
        CudaAppendPrefix {
            inner: self.inner,
            transaction: self.transaction,
            _borrow: std::marker::PhantomData,
            handle: self.handle,
            witness: self.witness,
            layer: self.layer,
            allocation: self.allocation,
        }
    }
    pub(super) fn adopt_resources(
        &mut self,
        resources: super::append_dma::AppendResources,
    ) -> Result<()> {
        self.validate()?;
        let mut inner = self.inner.borrow_mut();
        let entry = inner
            .entries
            .get_mut(&self.transaction.id)
            .expect("validated transaction");
        if entry.prefix_dma.is_some() {
            return Err(error("append resources already installed"));
        }
        entry.prefix_read.begin_submission(&self.handle)?;
        entry.prefix_dma = Some(resources);
        Ok(())
    }
    pub(super) fn finish_with_consumer(&mut self, consumer: CudaComputeEvent) -> Result<()> {
        self.validate()?;
        {
            let mut inner = self.inner.borrow_mut();
            let entry = inner
                .entries
                .get_mut(&self.transaction.id)
                .expect("validated transaction");
            let (dma, _) = entry
                .prefix_dma
                .as_ref()
                .ok_or_else(|| error("append resources absent"))?
                .evidence()?;
            entry.prefix_read.record_fences(
                &self.handle,
                dma,
                super::append_dma::AppendEvidence::Producer(consumer),
            )?;
        }
        if !self.finish()? {
            return Err(error("append output consumer still pending"));
        }
        Ok(())
    }
    pub(super) fn tail_range(&self, plane: usize) -> Result<std::ops::Range<usize>> {
        self.validate()?;
        let inner = self.inner.borrow();
        let entry = inner.entry(&self.transaction)?;
        let position = self.witness.source().committed_tokens;
        let page_size = self.witness.source().schema.page_size();
        let page = self.witness.pages()[position / page_size];
        let slot = *entry
            .plan
            .staged
            .get(&page)
            .ok_or_else(|| error("tail is not a staged write slot"))?;
        if plane >= 2 {
            return Err(error("invalid append plane"));
        }
        super::super::read_protocol::committed_chunk_range(
            super::super::read_protocol::CommittedChunk {
                slot,
                layer: self.layer,
                page_tokens: page_size,
                plane: inner.storage().planes()[plane],
                row: position % page_size,
                tokens: 1,
            },
            self.allocation[plane].1,
        )
    }
    pub(crate) fn validate(&self) -> Result<()> {
        let inner = self.inner.borrow();
        let entry = inner.entry(&self.transaction)?;
        if !entry.entered
            || entry.phase != Phase::Active
            || allocation(inner.storage())? != self.allocation
        {
            return Err(error("prefix transaction phase or allocation changed"));
        }
        entry.plan.validate(
            inner.storage().stats().allocated_slots,
            |p| inner.storage().physical_slot(p),
            |p| inner.storage().has_snapshot(p),
        )?;
        entry.prefix_read.validate(
            &self.handle,
            PrefixReadBinding {
                witness: &self.witness,
                layer: self.layer,
                nonce: entry.nonce,
                view_epoch: entry.epoch,
            },
            &entry.plan,
            |s| inner.storage().slot_epoch(s),
        )
    }
    /// Resolve only committed-prefix elements, never provisional tail writes.
    /// No device pointer or transferable buffer view escapes the capability.
    pub(crate) fn prefix_range(
        &self,
        plane: usize,
        start: usize,
        tokens: usize,
    ) -> Result<std::ops::Range<usize>> {
        self.validate()?;
        let source = self.witness.source();
        let end = start
            .checked_add(tokens)
            .ok_or_else(|| error("prefix extent overflow"))?;
        let page_size = source.schema.page_size();
        if plane >= 2
            || tokens == 0
            || start < source.positions.start
            || end > source.positions.end
            || start / page_size != (end - 1) / page_size
        {
            return Err(error("prefix chunk crosses frontier or page boundary"));
        }
        let inner = self.inner.borrow();
        let entry = inner.entry(&self.transaction)?;
        let page = source.pages[start / page_size];
        let original = entry
            .plan
            .original
            .get(&page)
            .copied()
            .flatten()
            .ok_or_else(|| error("prefix original slot absent"))?;
        let descriptor = inner.storage().planes()[plane];
        crate::cuda::operators::kv::read_protocol::committed_chunk_range(
            crate::cuda::operators::kv::read_protocol::CommittedChunk {
                slot: original,
                layer: self.layer,
                page_tokens: page_size,
                plane: descriptor,
                row: start % page_size,
                tokens,
            },
            self.allocation[plane].1,
        )
    }
    /// Start only after the trusted adapter has bound this owned, post-RoPE Q
    /// to the same query used by the tail. Budget and all resources stay in Entry.
    pub(crate) fn start_cpu_prefix(
        &mut self,
        operators: &CudaOperators,
        query: crate::cuda::operator_owner::CudaF32Buffer,
        config: crate::cpu::StreamingF32GqaConfig,
        budget: &ferrule_common::host_kv::placement::PlacementBudget,
    ) -> Result<()> {
        self.validate()?;
        let mut inner = self.inner.borrow_mut();
        if !std::ptr::eq(operators, inner.context.as_ref()) {
            return Err(error("append Q producer belongs to another operator owner"));
        }
        if inner.entry(&self.transaction)?.prefix_dma.is_some() {
            return Err(error("append DMA already started"));
        }
        let end = |base: u64, elements: usize| -> Result<u64> {
            let bytes = u64::try_from(elements)
                .ok()
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| error("append allocation byte extent overflow"))?;
            base.checked_add(bytes)
                .ok_or_else(|| error("append allocation address overflow"))
        };
        let query_start = query.as_device_buffer().cu_deviceptr();
        let query_end = end(query_start, query.len())?;
        for &(base, elements) in &self.allocation {
            if query_start < end(base, elements)? && base < query_end {
                return Err(error(
                    "append Q aliases resident or active-write KV allocation",
                ));
            }
        }
        let resources = super::append_dma::AppendResources::new(
            &inner.context,
            &self.witness,
            query,
            config,
            budget,
        )?;
        let entry = inner
            .entries
            .get_mut(&self.transaction.id)
            .expect("validated transaction");
        // Register both domains before ANY DMA. No second owner/journal is made.
        if let Err(e) = entry.prefix_read.begin_submission(&self.handle) {
            resources.release(); // positively no DMA submitted
            return Err(e);
        }
        entry.prefix_dma = Some(resources);
        Ok(())
    }
    pub(crate) fn poll_cpu_prefix(&mut self) -> Result<bool> {
        self.validate()?;
        let request = {
            let inner = self.inner.borrow();
            inner
                .entry(&self.transaction)?
                .prefix_dma
                .as_ref()
                .ok_or_else(|| error("append DMA not started"))?
                .request()?
        };
        let extent = match request {
            Some(super::append_stream::PrefixDownload::Key { start, tokens }) => {
                Some((0, self.prefix_range(0, start, tokens)?))
            }
            Some(super::append_stream::PrefixDownload::Value { start, tokens }) => {
                Some((1, self.prefix_range(1, start, tokens)?))
            }
            Some(super::append_stream::PrefixDownload::TailKey) => Some((0, self.tail_range(0)?)),
            Some(super::append_stream::PrefixDownload::TailValue) => Some((1, self.tail_range(1)?)),
            _ => None,
        };
        let mut inner = self.inner.borrow_mut();
        let ops = inner.context.clone();
        let Inner {
            entries, storage, ..
        } = &mut *inner;
        let source = extent.map(|(plane, range)| {
            (
                storage
                    .as_ref()
                    .expect("owned storage")
                    .plane_storage(plane)
                    .expect("validated F32"),
                range,
            )
        });
        let entry = entries
            .get_mut(&self.transaction.id)
            .expect("validated transaction");
        let result = entry
            .prefix_dma
            .as_mut()
            .expect("started DMA")
            .step(&ops, source);
        if result.is_err() {
            entry.prefix_read.mark_unknown(&self.handle)?;
        }
        result
    }
    /// Synchronous CPU consumer only. The adapter must merge the exact disjoint
    /// prefix/tail partials using the same Q, into separately charged output.
    /// An async consumer needs a distinct retained-resource extension, not this hook.
    pub(crate) fn consume_cpu_prefix(
        &mut self,
        consumer: impl FnOnce(&crate::cpu::StreamingF32GqaPartial) -> Result<()>,
    ) -> Result<()> {
        self.validate()?;
        let mut inner = self.inner.borrow_mut();
        let entry = inner
            .entries
            .get_mut(&self.transaction.id)
            .expect("validated transaction");
        let resources = entry
            .prefix_dma
            .as_mut()
            .ok_or_else(|| error("append DMA not started"))?;
        if let Err(e) = resources.consume(consumer) {
            entry.prefix_read.mark_unknown(&self.handle)?;
            return Err(e);
        }
        let (dma, cpu) = resources.evidence()?;
        entry.prefix_read.record_fences(&self.handle, dma, cpu)
    }
    /// Finish a no-submit registration or an exact completed DMA + synchronous
    /// consumer. A pending DMA is not an error/Unknown merely because polled early.
    pub(crate) fn finish(&mut self) -> Result<bool> {
        self.validate()?;
        let mut inner = self.inner.borrow_mut();
        let entry = inner
            .entries
            .get_mut(&self.transaction.id)
            .expect("validated transaction");
        if let Some(resources) = entry.prefix_dma.as_ref()
            && resources.evidence().is_err()
        {
            return Ok(false);
        }
        if !entry
            .prefix_read
            .finish(&self.handle, |f| f.is_complete())?
        {
            return Ok(false);
        }
        if let Some(resources) = entry.prefix_dma.take() {
            resources.release();
        }
        Ok(true)
    }
}
