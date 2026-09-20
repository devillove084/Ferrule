//! Guarded typed operator bindings; no raw CUDA handles cross this interface.
use super::*;
use crate::cuda::context::CudaF32Buffer;
use crate::cuda::operators::kv::PagedPlaneLayout;

/// References exist only while the view holds the pool's exclusive borrow guard.
/// Launch on the supplied owner compute stream using these validated packed
/// slots; only packed query rows are writable. Kernels must not stash pointers or
/// submit unfenced work on another stream.
pub struct CudaF32GqaPlanes<'a> {
    pub key: &'a mut CudaF32Buffer,
    pub value: &'a mut CudaF32Buffer,
    pub layout: PagedPlaneLayout,
    /// One physical slot per packed logical block-table entry, not logical IDs.
    pub block_slots: &'a [i32],
    /// Prefix offsets, including the terminal offset, in packed sequence order.
    pub block_offsets: &'a [i32],
}

#[derive(Clone)]
pub struct CudaKvView {
    pub(super) inner: Rc<RefCell<Inner>>,
    pub(super) id: ExecutionTransactionId,
    pub(super) nonce: u64,
    pub(super) epoch: u64,
}
impl CudaKvView {
    pub const fn transaction(&self) -> ExecutionTransactionId {
        self.id
    }
    fn entry<'a>(&self, inner: &'a Inner) -> Result<&'a Entry> {
        inner.available()?;
        let entry = inner
            .entries
            .get(&self.id)
            .ok_or_else(|| error("CUDA KV view transaction absent"))?;
        if entry.nonce != self.nonce
            || entry.epoch != self.epoch
            || !entry.entered
            || entry.phase != Phase::Active
        {
            return Err(error("CUDA KV view is stale or no longer entered"));
        }
        Ok(entry)
    }
    pub fn validate_batch(&self, batch: &PagedKvBatch) -> Result<()> {
        let inner = self
            .inner
            .try_borrow()
            .map_err(|_| error("CUDA KV storage already borrowed"))?;
        let entry = self.entry(&inner)?;
        if entry.batch.as_ref() != Some(batch) {
            return Err(error("CUDA KV view batch mismatch"));
        }
        Ok(())
    }
    pub fn physical_slot(&self, page: KvPageId) -> Option<u32> {
        let inner = self.inner.try_borrow().ok()?;
        self.entry(&inner).ok()?.plan.slot(page)
    }
    pub fn with_f32_gqa_planes<R>(
        &mut self,
        context: &CudaOperators,
        local_layer: usize,
        batch: &PagedKvBatch,
        run: impl FnOnce(CudaF32GqaPlanes<'_>) -> Result<R>,
    ) -> Result<R> {
        let mut inner = self
            .inner
            .try_borrow_mut()
            .map_err(|_| error("CUDA KV storage already borrowed"))?;
        if !std::ptr::eq(context, inner.context.as_ref()) {
            return Err(error("CUDA KV operator belongs to another owner/context"));
        }
        let entry = self.entry(&inner)?;
        if entry.batch.as_ref() != Some(batch) {
            return Err(error("CUDA KV operator packed batch mismatch"));
        }
        let (slots, offsets) = entry
            .plan
            .packed_slots(batch, inner.storage().page_tokens())?;
        let descriptor = inner.storage().planes()[0];
        let layout = PagedPlaneLayout {
            page_tokens: inner.storage().page_tokens(),
            elements_per_token: descriptor.elements_per_token,
            layer_index: local_layer,
            layer_count: descriptor.layer_count,
        };
        layout.validate()?;
        let (key, value) = inner.storage_mut().f32_transaction_planes()?;
        // Any submitted work, including a closure returning Err, remains owned by
        // the entered transaction. leave records its exact completion event.
        run(CudaF32GqaPlanes {
            key,
            value,
            layout,
            block_slots: &slots,
            block_offsets: &offsets,
        })
    }
}
