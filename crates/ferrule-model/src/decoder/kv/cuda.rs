//! Thin CUDA adapter over the same decoder ownership ledger and sealed cohort.
use super::*;
use ferrule_backend::cuda::context::CudaOperators;
use ferrule_backend::cuda::operators::kv::{self as device, CudaF32GqaPlanes};
use std::rc::Rc;

pub type CudaPagedKvBackend = PagedKvBackend<CudaPagedKvPool>;

#[derive(Clone)]
pub struct CudaPagedKvPool {
    inner: device::CudaPagedKvPool,
    max_sequence_len: Option<usize>,
}

/// Physical authority plus the exact decoder projection captured at prepare.
pub struct CudaPagedKvTransaction {
    inner: Option<device::CudaPagedKvTransaction>,
    new_pages: BTreeSet<KvPageId>,
    writable_pages: BTreeSet<KvPageId>,
    protected_pages: BTreeSet<KvPageId>,
    cows: Box<[ferrule_common::execution::KvCowReplacement]>,
    batch: Option<PackedDecoderBatch>,
}
impl CudaPagedKvTransaction {
    fn handle(&self) -> Result<&device::CudaPagedKvTransaction> {
        self.inner
            .as_ref()
            .ok_or_else(|| kv_error("CUDA physical KV handle absent"))
    }
    fn handle_mut(&mut self) -> Result<&mut device::CudaPagedKvTransaction> {
        self.inner
            .as_mut()
            .ok_or_else(|| kv_error("CUDA physical KV handle absent"))
    }
}
impl CudaPagedKvPool {
    pub fn new(
        context: Rc<CudaOperators>,
        planes: impl IntoIterator<Item = KvPlaneDescriptor>,
        page_size: usize,
        max_slots: usize,
    ) -> Result<Self> {
        let planes: Vec<_> = planes.into_iter().collect();
        Ok(Self {
            inner: device::CudaPagedKvPool::new(context, &planes, page_size, max_slots)?,
            max_sequence_len: None,
        })
    }
    pub fn from_strategy(
        context: Rc<CudaOperators>,
        strategy: &impl DecoderKvPlaneStrategy,
        max_slots: usize,
    ) -> Result<Self> {
        if strategy.max_sequence_len() == 0 || strategy.max_sequence_len() > u32::MAX as usize {
            return Err(kv_error(
                "CUDA KV maximum sequence length must fit positive u32",
            ));
        }
        let mut pool = Self::new(
            context,
            strategy.planes().iter().copied(),
            strategy.page_size(),
            max_slots,
        )?;
        pool.max_sequence_len = Some(strategy.max_sequence_len());
        Ok(pool)
    }
    pub fn from_schema(
        context: Rc<CudaOperators>,
        schema: &impl DecoderKvPlaneStrategy,
        max_slots: usize,
    ) -> Result<Self> {
        Self::from_strategy(context, schema, max_slots)
    }
    pub fn planes(&self) -> Vec<KvPlaneDescriptor> {
        self.inner.planes()
    }
    pub fn page_size(&self) -> usize {
        self.inner.page_size()
    }
    pub fn physical_slot(&self, page: KvPageId) -> Option<u32> {
        self.inner.physical_slot(page)
    }
    pub fn free_slot_count(&self) -> usize {
        self.inner.capacity().free_pages
    }
    pub fn active_transaction_count(&self) -> usize {
        self.inner.capacity().active_transactions
    }
}
impl DecoderKvPageView for CudaPagedKvPool {
    fn page_status(&self, page: KvPageId) -> DecoderKvPageStatus {
        map_page_status(self.inner.page_status(page))
    }
}
impl PhysicalKvPool for CudaPagedKvPool {
    type SequenceState = super::super::GenericDecoderSequenceState;
    type Transaction = CudaPagedKvTransaction;
    type KvView = CudaKvView;

    fn configured_capacity(&self) -> usize {
        self.inner.capacity().physical_pages
    }
    fn physical_free_slots(&self) -> Option<usize> {
        Some(self.inner.capacity().free_pages)
    }
    fn configure_capacity(&mut self, max_pages: usize) -> Result<()> {
        self.inner.configure_capacity(max_pages)
    }
    fn prepare_failure_retains_custody(&self) -> bool {
        self.inner.has_unresolved_custody()
    }
    fn prepare(&mut self, request: DecoderKvPrepare<'_>) -> Result<Self::Transaction> {
        let mut protected_pages: BTreeSet<_> = request.protected_pages.iter().copied().collect();
        protected_pages.extend(request.new_pages.iter().copied());
        protected_pages.extend(request.writable_pages.iter().copied());
        protected_pages.extend(
            request
                .cow_replacements
                .iter()
                .flat_map(|cow| [cow.source, cow.replacement]),
        );
        let inner = self.inner.prepare(cpu::PagedKvPrepare {
            transaction: request.transaction,
            new_pages: request.new_pages,
            writable_pages: request.writable_pages,
            cow_replacements: request.cow_replacements,
            protected_pages: request.protected_pages,
        })?;
        Ok(CudaPagedKvTransaction {
            inner: Some(inner),
            new_pages: request.new_pages.iter().copied().collect(),
            writable_pages: request.writable_pages.iter().copied().collect(),
            protected_pages,
            cows: request.cow_replacements.into(),
            batch: None,
        })
    }
    fn enter(
        &mut self,
        tx: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
        _states: &mut [Self::SequenceState],
    ) -> Result<()> {
        if batch.page_size() != self.page_size()
            || batch.new_pages().iter().copied().collect::<BTreeSet<_>>() != tx.new_pages
            || batch
                .writable_pages()
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                != tx.writable_pages
            || batch.cow_replacements() != tx.cows.as_ref()
            || batch.sequences().iter().any(|s| {
                self.max_sequence_len
                    .is_some_and(|max| s.sequence_len() > max)
            })
            || tx.batch.as_ref().is_some_and(|old| old != batch)
        {
            return Err(kv_error(
                "CUDA batch differs from reserved page/COW/schema projection",
            ));
        }
        self.inner.enter(tx.handle_mut()?, backend_batch(batch))?;
        tx.batch = Some(batch.clone());
        Ok(())
    }
    fn active_view(
        &mut self,
        tx: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
    ) -> Result<Self::KvView> {
        if tx.batch.as_ref() != Some(batch) {
            return Err(kv_error("CUDA active view batch mismatch"));
        }
        Ok(CudaKvView {
            inner: self.inner.active_view(tx.handle_mut()?)?,
            batch: batch.clone(),
        })
    }
    fn leave(&mut self, tx: &mut Self::Transaction) -> Result<()> {
        self.inner.leave(tx.handle_mut()?)
    }
    fn preflight_commit(
        &self,
        pending: &PagedKvTransaction<Option<Self::Transaction>>,
    ) -> Result<()> {
        let tx = pending
            .payload()
            .as_ref()
            .ok_or_else(|| kv_error("CUDA physical reservation absent"))?;
        let mut writes = tx.writable_pages.clone();
        writes.extend(tx.new_pages.iter().copied());
        writes.extend(tx.cows.iter().map(|cow| cow.replacement));
        if pending.entered()
            || pending.batch() != tx.batch.as_ref()
            || pending.new_pages != tx.new_pages
            || pending.writable_pages != writes
            || pending.protected_pages != tx.protected_pages
            || pending.cow_replacements.as_ref() != tx.cows.as_ref()
        {
            return Err(kv_error(
                "CUDA ready mapping/protection/batch witness changed",
            ));
        }
        self.inner.preflight_commit(tx.handle()?)
    }
    fn commit(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        let tx = transaction
            .as_mut()
            .ok_or_else(|| kv_error("CUDA commit handle absent"))?;
        let progress = map_end_progress(self.inner.commit(&mut tx.inner)?);
        if progress == KvEndProgress::Complete {
            transaction.take();
        }
        Ok(progress)
    }
    fn abort(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        let tx = transaction
            .as_mut()
            .ok_or_else(|| kv_error("CUDA abort handle absent"))?;
        let progress = map_end_progress(self.inner.abort(&mut tx.inner)?);
        if progress == KvEndProgress::Complete {
            transaction.take();
        }
        Ok(progress)
    }
    fn release(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.inner.release(pages)
    }
    fn preempt(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.inner.preempt(pages)
    }
    fn restore(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.inner.restore(pages)
    }
    fn shutdown(&mut self) -> Result<()> {
        self.inner.shutdown()
    }
}
impl PhysicalKvPreparedPool for CudaPagedKvPool {
    fn preflight_prepared(&self, tx: &Self::Transaction) -> Result<()> {
        self.inner.preflight_prepared(tx.handle()?)
    }
    fn install_prepared(
        &mut self,
        tx: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        self.inner
            .commit_prepared(tx.handle_mut()?, generation)
            .map(map_end_progress)
    }
    fn poll_install_prepared(
        &mut self,
        tx: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        self.inner
            .poll_install_ack(tx.handle()?, generation)
            .map(map_end_progress)
    }
    fn abort_prepared(
        &mut self,
        tx: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        self.inner
            .abort_prepared(tx.handle_mut()?, generation)
            .map(map_end_progress)
    }
    fn publish_prepared(&mut self, tx: &Self::Transaction) -> Result<()> {
        self.inner.publish_prepared(tx.handle()?)
    }
    fn preflight_retirement(&self, tx: &Self::Transaction, pages: &[KvPageId]) -> Result<()> {
        self.inner.preflight_retirement(tx.handle()?, pages)
    }
    fn retire_prepared(
        &mut self,
        tx: &mut Self::Transaction,
        generation: u64,
        pages: &[KvPageId],
    ) -> Result<KvEndProgress> {
        self.inner
            .retire_prepared(tx.handle_mut()?, generation, pages)
            .map(map_end_progress)
    }
    fn finish_prepared(&mut self, mut tx: Self::Transaction) -> Result<Vec<KvPageId>> {
        self.inner.finish_prepared(&mut tx.inner)
    }
}

/// Call-scoped bindings used by the standard CUDA operator adapter.
pub struct CudaGqaPlanesMut<'a> {
    pub key: &'a mut ferrule_backend::cuda::context::CudaF32Buffer,
    pub value: &'a mut ferrule_backend::cuda::context::CudaF32Buffer,
    pub key_layout: device::PagedPlaneLayout,
    pub value_layout: device::PagedPlaneLayout,
    pub block_slots: &'a [i32],
    pub block_offsets: &'a [i32],
}

/// Typed GPU bindings backed by an Rc storage owner and a transaction read set.
#[derive(Clone)]
pub struct CudaKvView {
    inner: device::CudaKvView,
    batch: PackedDecoderBatch,
}
impl CudaKvView {
    pub const fn transaction(&self) -> ferrule_common::execution::ExecutionTransactionId {
        self.inner.transaction()
    }
    pub fn validate_batch(&self, batch: &PackedDecoderBatch) -> Result<()> {
        if batch != &self.batch {
            return Err(kv_error("CUDA KV view packed projection mismatch"));
        }
        self.inner.validate_batch(&backend_batch(batch))
    }
    pub fn physical_slot(&self, page: KvPageId) -> Option<u32> {
        self.inner.physical_slot(page)
    }
    /// Compatibility spelling for the standard CUDA operator's call-scoped
    /// binding. The outer Result validates custody; the closure owns its result.
    pub fn with_f32_gqa_planes_mut<R>(
        &mut self,
        context: &CudaOperators,
        local_layer: usize,
        batch: &PackedDecoderBatch,
        readset: &[KvPageId],
        run: impl FnOnce(CudaGqaPlanesMut<'_>) -> R,
    ) -> Result<R> {
        self.with_f32_gqa_planes(context, local_layer, batch, readset, |planes| {
            Ok(run(CudaGqaPlanesMut {
                key: planes.key,
                value: planes.value,
                key_layout: planes.layout,
                value_layout: planes.layout,
                block_slots: planes.block_slots,
                block_offsets: planes.block_offsets,
            }))
        })
    }
    /// Validates the full packed projection and declared read set before lending
    /// K/V buffers under an exclusive guard. Slots prefer shadow over resident.
    pub fn with_f32_gqa_planes<R>(
        &mut self,
        context: &CudaOperators,
        local_layer: usize,
        batch: &PackedDecoderBatch,
        readset: &[KvPageId],
        run: impl FnOnce(CudaF32GqaPlanes<'_>) -> Result<R>,
    ) -> Result<R> {
        self.validate_batch(batch)?;
        let reads = unique_page_set(readset, "CUDA operator read set")?;
        for page in &reads {
            if self.inner.physical_slot(*page).is_none() {
                return Err(kv_error("CUDA operator attempted an undeclared page read"));
            }
        }
        if batch
            .sequences()
            .iter()
            .flat_map(|sequence| sequence.block_table())
            .any(|page| !reads.contains(page))
        {
            return Err(kv_error(
                "CUDA operator read set differs from protected packed pages",
            ));
        }
        self.inner
            .with_f32_gqa_planes(context, local_layer, &backend_batch(batch), run)
    }
}
impl TransformerKvView for CudaKvView {
    fn append(&mut self, _request: KvAppendRequest<'_>) -> Result<()> {
        Err(kv_error(
            "CUDA KV host append unsupported; use GPU plane bindings",
        ))
    }
    fn history(
        &self,
        _layer: usize,
        _sequence: usize,
        _position: usize,
        _kv_heads: usize,
        _head_dim: usize,
    ) -> Result<KvHistory> {
        Err(kv_error(
            "CUDA KV host history unsupported; use GPU plane bindings",
        ))
    }
}
