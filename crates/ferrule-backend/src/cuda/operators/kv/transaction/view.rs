//! Guarded typed operator bindings; no raw CUDA handles cross this interface.
use super::*;
use crate::cuda::operator_owner::CudaF32Buffer;
use crate::cuda::operators::kv::PagedPlaneLayout;

/// References exist only while the view holds the pool's exclusive borrow guard.
/// Launch on the supplied owner compute stream using these validated packed
/// slots; only packed query rows are writable. Kernels must not stash pointers or
/// submit unfenced work on another stream.
pub struct CudaF32GqaPlanes<'a> {
    pub append_transaction: super::append_bridge::CudaAppendTransaction,
    pub key: &'a mut CudaF32Buffer,
    pub value: &'a mut CudaF32Buffer,
    pub layout: PagedPlaneLayout,
    /// Kernel-local physical slots, not logical IDs. In chunked mode these are
    /// Entry staging coordinates, not `CudaKvView::physical_slot` coordinates.
    pub block_slots: &'a [i32],
    /// Prefix offsets, including the terminal offset, in packed sequence order.
    pub block_offsets: &'a [i32],
}

impl CudaF32GqaPlanes<'_> {
    /// Kernel ABI capacity, which can differ from logical pool capacity in
    /// chunked mode. Use with the callback's block_slots, never pool slot IDs.
    pub fn kernel_slot_capacity(&self) -> usize {
        self.key.len()
            / self
                .layout
                .checked_elements_per_page()
                .expect("validated layout")
    }
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
        // Existing plane callbacks expose whole mutable allocations, so even
        // disjoint tail writes must wait until this subordinate read is finished.
        inner.no_prefix_readers()?;
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
        super::plan::validate_plane_type(
            inner.storage().planes(),
            ferrule_common::execution::KvElementType::F32,
        )?;
        let (mut slots, offsets) = entry
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
        if inner.storage().staging_page_limit().is_some() {
            let entry = inner.entries.get_mut(&self.id).expect("validated");
            let staging = entry
                .staging
                .as_mut()
                .ok_or_else(|| error("chunk staging is absent"))?;
            for slot in &mut slots {
                *slot = *staging
                    .slots
                    .get(&(*slot as u32))
                    .ok_or_else(|| error("staging slot absent"))?;
            }
            let (key, value) = staging.planes.split_at_mut(1);
            return entry.completion.probe(|| {
                run(CudaF32GqaPlanes {
                    append_transaction: super::append_bridge::CudaAppendTransaction {
                        view: self.clone(),
                    },
                    key: &mut key[0],
                    value: &mut value[0],
                    layout,
                    block_slots: &slots,
                    block_offsets: &offsets,
                })
            });
        }
        let (key, value) = inner.storage_mut().f32_transaction_planes()?;
        // Any submitted work, including a closure returning Err, remains owned by
        // the entered transaction. leave records its exact completion event.
        run(CudaF32GqaPlanes {
            append_transaction: super::append_bridge::CudaAppendTransaction { view: self.clone() },
            key,
            value,
            layout,
            block_slots: &slots,
            block_offsets: &offsets,
        })
    }
}

impl CudaKvView {
    /// Borrow an authenticated BF16 cache binding. Slots, batch, epoch and COW
    /// writes come from this entered transaction, never from caller metadata.
    pub fn with_bf16_gqa_planes<R>(
        &mut self,
        context: &CudaOperators,
        local_layer: usize,
        batch: &PagedKvBatch,
        run: impl FnOnce(CudaBf16GqaPlanes<'_>) -> Result<R>,
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
        super::plan::validate_plane_type(
            inner.storage().planes(),
            ferrule_common::execution::KvElementType::Bf16,
        )?;
        super::plan::validate_storage_mode(
            inner.storage().planes(),
            inner.storage().staging_page_limit().is_some(),
        )?;
        let (slots, offsets) = entry
            .plan
            .packed_slots(batch, inner.storage().page_tokens())?;
        let descriptor = inner.storage().planes()[0];
        let layout = PagedPlaneLayout {
            page_tokens: inner.storage().page_tokens(),
            elements_per_token: descriptor.elements_per_token,
            layer_index: local_layer,
            layer_count: descriptor.layer_count,
        }
        .typed(ferrule_common::execution::KvElementType::Bf16);
        layout.validate()?;
        let capacity = inner.storage().stats().allocated_slots;
        let (key, value) = inner.storage_mut().bf16_plane_pair_mut(0, 1)?;
        // As for F32, leave fences any submitted work even if the closure errs.
        run(CudaBf16GqaPlanes::new(
            context, key, value, layout, capacity, batch, &slots, &offsets,
        )?)
    }
}
