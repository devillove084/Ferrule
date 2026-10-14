//! Owner-local transactional standard-GQA storage. All clones share physical
//! pins; this journal is not a logical/global transaction coordinator.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use super::CudaKvPagePool;
use crate::cpu::{KvCapacity, KvEndProgress, KvPageStatus, PagedKvBatch, PagedKvPrepare};
use crate::cuda::operator_owner::{CudaComputeEvent, CudaOperators};
use ferrule_common::Result;
use ferrule_common::execution::{ExecutionTransactionId, KvPageId, KvPlaneDescriptor};

mod bf16;
mod chunked;
#[cfg(test)]
#[path = "../../../../tests/chunked_kv/entry_faults.rs"]
mod chunked_entry_faults;
mod chunked_protocol;
#[path = "committed_read.rs"]
mod committed_read;
mod lifecycle;
mod diagnostic;
pub use diagnostic::{CudaCommittedBf16KvChunk, committed_bf16_kv_trace_host_bytes};
pub use chunked::{CudaKvPhysicalStats, CudaKvReclaim};
pub use committed_read::CudaCommittedRead;
pub mod append_bridge;
mod plan;
// Internal DMA is connected; ordinary decoder dispatch still requires a trusted bridge.
#[allow(dead_code)]
mod append_dma;
#[cfg(test)]
#[path = "../../../../tests/host_kv/append_dma_cuda.rs"]
mod append_dma_cuda_tests;
#[allow(dead_code)]
mod append_read;
#[allow(dead_code)]
mod append_stream;
#[allow(dead_code)]
mod prefix_read;
#[cfg(test)]
mod prepare_tests;
mod prepared;
#[cfg(test)]
mod tests;
mod view;
pub use bf16::{CudaBf16GqaPlanes, CudaBf16GqaRows};
use plan::{Plan, error, unique, validate_schema};
pub use view::{CudaF32GqaPlanes, CudaKvView};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Active,
    Installed,
    Published,
    Retired,
    Aborting,
    Aborted,
}

struct Entry {
    nonce: u64,
    plan: Plan,
    batch: Option<PagedKvBatch>,
    entered: bool,
    epoch: u64,
    fence: Option<CudaComputeEvent>,
    completion: chunked_protocol::StickyCompletion,
    prefix_read: prefix_read::PrefixReadCustody<append_dma::AppendEvidence>,
    prefix_dma: Option<append_dma::AppendResources>,
    staging: Option<chunked::ChunkStaging>,
    phase: Phase,
    generation: Option<u64>,
    quarantine: Vec<u32>,
    retired: Vec<KvPageId>,
}

impl Entry {
    fn prefix_unknown(&self) -> bool {
        self.completion.is_unknown()
            || self.prefix_read.is_unknown()
            || self.prefix_dma.as_ref().is_some_and(|r| r.is_unknown())
    }
    fn quiescent(&self) -> Result<bool> {
        self.prefix_read.ensure_idle()?;
        if self.entered {
            return Err(error("CUDA KV transaction is still entered"));
        }
        self.completion.probe(|| {
            self.fence
                .as_ref()
                .ok_or_else(|| error("CUDA KV writes fence is unknown; custody retained"))?
                .is_complete()
        })
    }

    fn generation(&self, generation: u64) -> Result<()> {
        if generation == 0 || self.generation.is_some_and(|bound| bound != generation) {
            return Err(error("CUDA KV prepared generation mismatch"));
        }
        Ok(())
    }
}

struct Inner {
    identity: Rc<()>,
    context: Rc<CudaOperators>,
    storage: Option<CudaKvPagePool>,
    entries: BTreeMap<ExecutionTransactionId, Entry>,
    next_nonce: u64,
    committed_reader: Option<u64>,
    committed_logical_owner: Option<u64>,
    poisoned: bool,
    shutdown: bool,
}

impl Inner {
    fn storage(&self) -> &CudaKvPagePool {
        self.storage.as_ref().expect("owned CUDA KV storage")
    }
    fn storage_mut(&mut self) -> &mut CudaKvPagePool {
        self.storage.as_mut().expect("owned CUDA KV storage")
    }
    fn no_committed_reader(&self) -> Result<()> {
        if self.committed_reader.is_some() {
            return Err(error("CUDA pool has an isolated committed reader"));
        }
        Ok(())
    }
    fn no_prefix_readers(&self) -> Result<()> {
        for entry in self.entries.values() {
            entry.prefix_read.ensure_idle()?;
        }
        Ok(())
    }
    fn available(&self) -> Result<()> {
        if self.poisoned
            || self.shutdown
            || self.storage().has_chunk_dma()
            || self.entries.values().any(|e| e.prefix_unknown())
        {
            return Err(error("CUDA KV pool is quarantined or shut down"));
        }
        Ok(())
    }
    fn entry(&self, tx: &CudaPagedKvTransaction) -> Result<&Entry> {
        self.available()?;
        if !Rc::ptr_eq(&self.identity, &tx.identity) {
            return Err(error("foreign CUDA KV pool handle"));
        }
        let entry = self
            .entries
            .get(&tx.id)
            .ok_or_else(|| error("CUDA KV transaction is absent or finished"))?;
        if entry.nonce != tx.nonce {
            return Err(error("stale CUDA KV transaction nonce"));
        }
        Ok(entry)
    }
    fn validate_active(&self, tx: &CudaPagedKvTransaction) -> Result<&Entry> {
        let entry = self.entry(tx)?;
        entry.prefix_read.ensure_idle()?;
        if entry.phase != Phase::Active || entry.entered {
            return Err(error("CUDA KV reservation is not active and left"));
        }
        let storage = self.storage();
        entry.plan.validate(
            storage.stats().allocated_slots,
            |p| storage.physical_slot(p),
            |p| storage.has_snapshot(p),
        )?;
        Ok(entry)
    }
    fn unpinned(&self, pages: &[KvPageId], except: Option<ExecutionTransactionId>) -> Result<()> {
        self.available()?;
        self.no_committed_reader()?;
        for (id, entry) in &self.entries {
            if Some(*id) != except
                && pages
                    .iter()
                    .any(|page| entry.plan.original.contains_key(page))
            {
                return Err(error(
                    "CUDA KV page remains pinned by an active/prepared owner",
                ));
            }
        }
        Ok(())
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        if self.poisoned
            || self.storage().has_chunk_dma()
            || !self.entries.is_empty()
            || self.committed_reader.is_some()
        {
            // Unknown device completion (or abandoned prepared custody) must not
            // return allocations to the allocator, even after the last clone.
            std::mem::forget(self.storage.take());
            std::mem::forget(std::mem::take(&mut self.entries));
            std::mem::forget(self.context.clone());
        }
    }
}

/// Non-cloneable authority: pool identity, transaction ID, and monotonic nonce.
#[derive(Debug)]
pub struct CudaPagedKvTransaction {
    id: ExecutionTransactionId,
    identity: Rc<()>,
    nonce: u64,
}
impl CudaPagedKvTransaction {
    pub const fn id(&self) -> ExecutionTransactionId {
        self.id
    }
}

/// Logical capacity includes resident, provisional/shadow, and quarantined slots.
/// Fixed contiguous storage is the default; `new_chunked` opts into real release.
#[derive(Clone)]
pub struct CudaPagedKvPool {
    inner: Rc<RefCell<Inner>>,
}

impl CudaPagedKvPool {
    pub fn validate_schema(
        planes: &[KvPlaneDescriptor],
        page_size: usize,
        max_slots: usize,
    ) -> Result<()> {
        validate_schema(planes, page_size, max_slots)
    }
    pub fn new(
        context: Rc<CudaOperators>,
        planes: &[KvPlaneDescriptor],
        page_size: usize,
        max_slots: usize,
    ) -> Result<Self> {
        validate_schema(planes, page_size, max_slots)?;
        let storage = CudaKvPagePool::new(&context, planes, page_size, max_slots)?;
        Ok(Self::from_storage(context, storage))
    }

    fn from_storage(context: Rc<CudaOperators>, storage: CudaKvPagePool) -> Self {
        Self {
            inner: Rc::new(RefCell::new(Inner {
                identity: Rc::new(()),
                context,
                storage: Some(storage),
                entries: BTreeMap::new(),
                next_nonce: 1,
                committed_reader: None,
                committed_logical_owner: None,
                poisoned: false,
                shutdown: false,
            })),
        }
    }
    pub fn planes(&self) -> Vec<KvPlaneDescriptor> {
        self.inner.borrow().storage().planes().to_vec()
    }
    pub fn page_size(&self) -> usize {
        self.inner.borrow().storage().page_tokens()
    }
    pub fn physical_slot(&self, page: KvPageId) -> Option<u32> {
        self.inner.borrow().storage().physical_slot(page)
    }
    pub fn capacity(&self) -> KvCapacity {
        let inner = self.inner.borrow();
        let stats = inner.storage().stats();
        // Snapshots are accounted by the physical pool, not a second page ledger.
        KvCapacity {
            physical_pages: stats.allocated_slots,
            resident_pages: stats.resident_pages,
            preempted_pages: inner.storage().transaction_snapshot_count(),
            active_transactions: inner.entries.len(),
            free_pages: if inner.poisoned
                || inner.storage().has_chunk_dma()
                || inner.entries.values().any(|e| e.prefix_unknown())
            {
                0
            } else {
                stats.free_slots
            },
        }
    }
    /// An error is not a cleanup ACK while this is true. Available even when
    /// poisoned, so adapters retain logical custody despite no returned handle.
    pub fn has_unresolved_custody(&self) -> bool {
        let inner = self.inner.borrow();
        inner.poisoned
            || inner.storage().has_chunk_dma()
            || inner.entries.values().any(|e| e.prefix_unknown())
    }

    pub fn page_status(&self, page: KvPageId) -> KvPageStatus {
        let inner = self.inner.borrow();
        if inner.storage().physical_slot(page).is_some() {
            KvPageStatus::Resident
        } else if inner.storage().has_snapshot(page) {
            KvPageStatus::Preempted
        } else {
            KvPageStatus::Vacant
        }
    }
    pub fn configure_capacity(&mut self, capacity: usize) -> Result<()> {
        let inner = self.inner.borrow();
        inner.available()?;
        inner.no_committed_reader()?;
        if !inner.entries.is_empty()
            || inner.storage().stats().resident_pages != 0
            || inner.storage().transaction_snapshot_count() != 0
            || capacity != inner.storage().stats().allocated_slots
        {
            return Err(error(
                "CUDA KV capacity is fixed; reconfiguration requires an empty pool with the same bound",
            ));
        }
        Ok(())
    }
    pub fn prepare(&mut self, request: PagedKvPrepare<'_>) -> Result<CudaPagedKvTransaction> {
        self.prepare_with_submission(request, |inner, copies| {
            let context = inner.context.clone();
            for &(slot, source) in copies {
                if let Some(source) = source {
                    inner.storage_mut().copy_slot(&context, source, slot)?;
                } else {
                    bf16::clear_transaction_slot(inner, &context, slot)?;
                }
            }
            context.record_compute_event()
        })
    }

    // One reservation/failure path for production submission and deterministic
    // unit fault injection. The callback can never run before pins are saved.
    fn prepare_with_submission(
        &mut self,
        request: PagedKvPrepare<'_>,
        submit: impl FnOnce(&mut Inner, &[(u32, Option<u32>)]) -> Result<CudaComputeEvent>,
    ) -> Result<CudaPagedKvTransaction> {
        let mut inner = self.inner.borrow_mut();
        inner.available()?;
        inner.no_committed_reader()?;
        inner.no_prefix_readers()?;
        plan::validate_storage_mode(
            inner.storage().planes(),
            inner.storage().staging_page_limit().is_some(),
        )?;
        if inner.storage().planes()[0].element_type
            == ferrule_common::execution::KvElementType::Bf16
            && inner.context.stream_clone().context().is_capturing()
        {
            return Err(error(
                "transactional BF16 KV does not support graph capture",
            ));
        }
        if inner.entries.contains_key(&request.transaction) {
            return Err(error("duplicate CUDA KV transaction"));
        }
        let mut plan = Plan::new(
            request,
            |p| inner.storage().physical_slot(p),
            |p| inner.storage().has_snapshot(p),
        )?;
        if inner
            .entries
            .values()
            .any(|entry| plan.conflicts(&entry.plan, entry.phase != Phase::Active))
        {
            return Err(error(
                "CUDA KV read/write set conflicts with shared physical custody",
            ));
        }
        if let Some(limit) = inner.storage().staging_page_limit() {
            chunked_protocol::admit(plan.original.len(), limit, inner.entries.len())?;
            if inner.context.stream_clone().context().is_capturing() {
                return Err(error("chunked KV does not support graph capture"));
            }
        }
        let nonce = inner.next_nonce;
        let next = nonce
            .checked_add(1)
            .ok_or_else(|| error("CUDA KV nonce exhausted"))?;
        plan.bind_slots(
            inner
                .storage_mut()
                .take_transaction_slots(plan.copies.len())?,
        );
        let copies: Vec<_> = plan
            .staged
            .iter()
            .map(|(page, slot)| (*slot, plan.copies[page]))
            .collect();
        inner.entries.insert(
            request.transaction,
            Entry {
                nonce,
                plan,
                batch: None,
                entered: false,
                epoch: 0,
                fence: None,
                completion: Default::default(),
                prefix_read: prefix_read::PrefixReadCustody::default(),
                prefix_dma: None,
                staging: None,
                phase: Phase::Active,
                generation: None,
                quarantine: Vec::new(),
                retired: Vec::new(),
            },
        );
        inner.next_nonce = next;
        inner.poisoned = true;
        match submit(&mut inner, &copies) {
            Ok(fence) => {
                inner.poisoned = false;
                inner
                    .entries
                    .get_mut(&request.transaction)
                    .expect("reserved")
                    .fence = Some(fence)
            }
            Err(err) => {
                inner.poisoned = true;
                return Err(err);
            }
        }
        Ok(CudaPagedKvTransaction {
            id: request.transaction,
            identity: inner.identity.clone(),
            nonce,
        })
    }
    pub fn enter(&mut self, tx: &mut CudaPagedKvTransaction, batch: PagedKvBatch) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        let entry = inner.validate_active(tx)?;
        entry
            .plan
            .packed_slots(&batch, inner.storage().page_tokens())?;
        if entry.batch.as_ref().is_some_and(|old| old != &batch) {
            return Err(error("CUDA KV reentry changed packed batch"));
        }
        let epoch = entry
            .epoch
            .checked_add(1)
            .ok_or_else(|| error("CUDA KV view epoch exhausted"))?;
        let entry = inner.entries.get_mut(&tx.id).expect("validated");
        entry.epoch = epoch;
        entry.batch = Some(batch);
        entry.entered = true;
        entry.fence = None;
        if inner.storage().staging_page_limit().is_some() {
            inner.poisoned = true;
            if let Err(err) = inner.gather_chunked(tx.id) {
                inner.poisoned = true;
                return Err(err);
            }
            inner.poisoned = false;
        }
        Ok(())
    }
    pub fn active_view(&mut self, tx: &mut CudaPagedKvTransaction) -> Result<CudaKvView> {
        let inner = self.inner.borrow();
        let entry = inner.entry(tx)?;
        if !entry.entered || entry.phase != Phase::Active {
            return Err(error("CUDA KV view requires entered custody"));
        }
        Ok(CudaKvView {
            inner: self.inner.clone(),
            id: tx.id,
            nonce: tx.nonce,
            epoch: entry.epoch,
        })
    }
    pub fn leave(&mut self, tx: &mut CudaPagedKvTransaction) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        inner.no_prefix_readers()?;
        let entry = inner.entry(tx)?;
        if !entry.entered {
            return Err(error("CUDA KV transaction is not entered"));
        }
        inner.poisoned = true;
        if inner.storage().staging_page_limit().is_some() {
            inner.scatter_chunked(tx.id)?;
        }
        let fence = match inner.context.record_compute_event() {
            Ok(fence) => fence,
            Err(err) => {
                inner.poisoned = true;
                return Err(err);
            }
        };
        inner.poisoned = false;
        let entry = inner.entries.get_mut(&tx.id).expect("validated");
        entry.fence = Some(fence);
        entry.entered = false;
        Ok(())
    }
    /// Mapping validation is read-only and never consumes pending custody.
    pub fn preflight_commit(&self, tx: &CudaPagedKvTransaction) -> Result<()> {
        self.inner.borrow().validate_active(tx).map(|_| ())
    }
    pub fn commit(&mut self, tx: &mut Option<CudaPagedKvTransaction>) -> Result<KvEndProgress> {
        let handle = tx
            .as_ref()
            .ok_or_else(|| error("CUDA commit handle absent"))?;
        let mut inner = self.inner.borrow_mut();
        let entry = inner.validate_active(handle)?;
        if !entry.quiescent()? {
            return Ok(KvEndProgress::Pending);
        }
        let entry = inner.entries.remove(&handle.id).expect("validated");
        let mut old = Vec::new();
        for (page, slot) in entry.plan.staged {
            if let Some(slot) = inner.storage_mut().install_transaction_slot(page, slot) {
                old.push(slot);
            }
        }
        inner.storage_mut().recycle_transaction_slots(old);
        tx.take();
        Ok(KvEndProgress::Complete)
    }
    pub fn abort(&mut self, tx: &mut Option<CudaPagedKvTransaction>) -> Result<KvEndProgress> {
        let handle = tx
            .as_ref()
            .ok_or_else(|| error("CUDA abort handle absent"))?;
        let mut inner = self.inner.borrow_mut();
        let entry = inner.entry(handle)?;
        if entry.phase != Phase::Active {
            return Err(error("ordinary abort cannot consume a prepared lease"));
        }
        if !entry.quiescent()? {
            return Ok(KvEndProgress::Pending);
        }
        let entry = inner.entries.remove(&handle.id).expect("validated");
        inner
            .storage_mut()
            .recycle_transaction_slots(entry.plan.staged.into_values());
        tx.take();
        Ok(KvEndProgress::Complete)
    }
}
