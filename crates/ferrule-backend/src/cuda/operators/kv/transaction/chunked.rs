//! Bounded ABI-compatible staging belongs to the existing transaction Entry.
//! Authoritative chunks are never exposed by mutable plane callbacks.
use super::super::page_pool::KvHostSnapshot;
use super::*;
use crate::cuda::operator_owner::CudaF32Buffer;
use crate::cuda::runtime::DeviceBuffer;

pub(super) struct ChunkStaging {
    pub slots: BTreeMap<u32, i32>,
    pub planes: Vec<CudaF32Buffer>,
}

/// Payload ownership is not driver residency. Allocator grants/segments can be
/// rounded and their retirement is asynchronous, even after transaction leave.
#[derive(Debug, Clone, Copy)]
pub struct CudaKvPhysicalStats {
    pub independently_allocated: bool,
    pub device_payload_bytes: usize,
    pub entry_staging_payload_bytes: usize,
    pub host_snapshot_bytes: usize,
    pub pending_pinned_bytes: usize,
    /// Context-wide allocator metrics, including unrelated allocations.
    pub allocator: crate::cuda::CudaAllocatorMetrics,
}

#[derive(Debug, Clone, Copy)]
pub struct CudaKvReclaim {
    /// Successful trim return; context-wide, not attributed to a logical page.
    pub allocator_trimmed_bytes: usize,
    /// Actual successful driver-free byte counter delta, not slot reuse.
    pub driver_freed_bytes: u64,
}

impl CudaPagedKvPool {
    /// Opt-in independently trimmable F32 page storage. The existing kernel ABI
    /// operates on an Entry-owned compact staging chunk. `staging_page_limit`
    /// bounds all declared pages (including COW sources) per transaction. Only
    /// one transaction may be active in this initial path, making the scratch
    /// bound pool-wide. No fixed-capacity device planes are allocated.
    ///
    /// Ordinary prepare/enter/view/leave/commit and prepared leases are unchanged.
    /// Graph capture and legacy append-prefix/committed-read DMA are unsupported.
    pub fn new_chunked(
        context: Rc<CudaOperators>,
        planes: &[KvPlaneDescriptor],
        page_size: usize,
        max_slots: usize,
        staging_page_limit: usize,
    ) -> Result<Self> {
        validate_schema(planes, page_size, max_slots)?;
        let storage =
            CudaKvPagePool::new_chunked(planes, page_size, max_slots, staging_page_limit)?;
        Ok(Self::from_storage(context, storage))
    }

    /// Report a canceled/failed physical submission whose complete work cannot
    /// be fenced. This only quarantines; no public API can turn it into an ACK.
    /// Dropping the pool afterwards retains device and DMA operands indefinitely.
    pub fn mark_transaction_unknown(&mut self, tx: &CudaPagedKvTransaction) -> Result<()> {
        let inner = self.inner.borrow();
        inner.entry(tx)?.completion.cancel_unknown();
        Ok(())
    }

    pub fn physical_stats(&self) -> CudaKvPhysicalStats {
        let inner = self.inner.borrow();
        CudaKvPhysicalStats {
            independently_allocated: inner.storage().staging_page_limit().is_some(),
            device_payload_bytes: inner.storage().stats().allocated_bytes,
            entry_staging_payload_bytes: inner
                .entries
                .values()
                .filter_map(|e| e.staging.as_ref())
                .flat_map(|s| &s.planes)
                .map(|p| p.len() * 4)
                .sum(),
            host_snapshot_bytes: inner.storage().transaction_snapshot_count()
                * inner.storage().bytes_per_slot(),
            pending_pinned_bytes: inner.storage().pending_chunk_dma_bytes(),
            allocator: inner.context.allocator_metrics(),
        }
    }

    /// Owned complete snapshot after preempt; never accepts caller-authored
    /// contents or completion ACKs. Includes every plane, layer and page token.
    pub fn host_snapshot(&self, page: KvPageId) -> Result<KvHostSnapshot> {
        let inner = self.inner.borrow();
        inner.available()?;
        inner
            .storage()
            .snapshot(page)
            .cloned()
            .ok_or_else(|| error("page has no complete host snapshot"))
    }

    /// Nonblocking allocator reap/trim. May return zero while retirement events
    /// are pending. Call again later; this never converts Unknown into completion.
    pub fn trim_physical_allocations(&mut self) -> Result<CudaKvReclaim> {
        let inner = self.inner.borrow();
        inner.available()?;
        inner.no_committed_reader()?;
        if !inner.entries.is_empty() {
            return Err(error("physical trim requires no transaction custody"));
        }
        let before = inner.context.allocator_metrics().driver_free_bytes;
        let allocator_trimmed_bytes = inner.context.trim_device_allocator()?;
        let after = inner.context.allocator_metrics().driver_free_bytes;
        Ok(CudaKvReclaim {
            allocator_trimmed_bytes,
            driver_freed_bytes: after.saturating_sub(before),
        })
    }
}

impl Inner {
    pub(super) fn gather_chunked(&mut self, id: ExecutionTransactionId) -> Result<()> {
        let Self {
            context,
            entries,
            storage,
            ..
        } = self;
        let storage = storage.as_ref().expect("owned storage");
        let entry = entries.get_mut(&id).expect("registered Entry");
        if entry.staging.is_some() {
            return Ok(());
        }
        let slots = super::chunked_protocol::compact_slots(
            entry
                .plan
                .original
                .keys()
                .map(|page| entry.plan.slot(*page).expect("validated plan")),
            storage.staging_page_limit().expect("chunked pool"),
        )?;
        entry.staging = Some(ChunkStaging {
            slots,
            planes: Vec::new(),
        });
        let staging = entry.staging.as_mut().expect("installed before submission");
        for plane in 0..storage.planes().len() {
            let elements = storage.page_elements(plane).expect("schema");
            let len = elements
                .checked_mul(staging.slots.len())
                .ok_or_else(|| error("staging extent overflow"))?;
            // No initialization kernel until the allocation has Entry custody.
            let buffer = DeviceBuffer::uninitialized_dedicated(&context.stream_clone(), len)?;
            staging
                .planes
                .push(CudaF32Buffer::from_device_buffer(buffer));
            for (&slot, &index) in &staging.slots {
                context.copy_f32_range(
                    storage.chunk_plane(slot, plane)?,
                    0,
                    &mut staging.planes[plane],
                    index as usize * elements,
                    elements,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn scatter_chunked(&mut self, id: ExecutionTransactionId) -> Result<()> {
        let Self {
            context,
            entries,
            storage,
            ..
        } = self;
        let storage = storage.as_mut().expect("owned storage");
        let entry = entries.get_mut(&id).expect("registered Entry");
        let staging = entry
            .staging
            .as_ref()
            .ok_or_else(|| error("missing chunk staging"))?;
        // Only provisional writes may return to authoritative storage. Read-only
        // pages in staging cannot mutate originals, even through whole-plane APIs.
        for &slot in entry.plan.staged.values() {
            let index = staging.slots[&slot] as usize;
            for (plane, source) in staging.planes.iter().enumerate() {
                let elements = storage.page_elements(plane).expect("schema");
                context.copy_f32_range(
                    source,
                    index * elements,
                    storage.chunk_plane_mut(slot, plane)?,
                    0,
                    elements,
                )?;
            }
        }
        Ok(())
    }
}
