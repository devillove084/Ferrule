//! Owner-local transactional standard-GQA storage. All clones share physical
//! pins; this journal is not a logical/global transaction coordinator.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use super::CudaKvPagePool;
use crate::cpu::{KvCapacity, KvEndProgress, KvPageStatus, PagedKvBatch, PagedKvPrepare};
use crate::cuda::context::{CudaComputeEvent, CudaOperators};
use ferrule_common::Result;
use ferrule_common::execution::{ExecutionTransactionId, KvPageId, KvPlaneDescriptor};

mod lifecycle;
mod plan;
#[cfg(test)]
mod prepare_tests;
mod prepared;
#[cfg(test)]
mod tests;
mod view;
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
    phase: Phase,
    generation: Option<u64>,
    quarantine: Vec<u32>,
    retired: Vec<KvPageId>,
}

impl Entry {
    fn quiescent(&self) -> Result<bool> {
        if self.entered {
            return Err(error("CUDA KV transaction is still entered"));
        }
        self.fence
            .as_ref()
            .ok_or_else(|| error("CUDA KV writes fence is unknown; custody retained"))?
            .is_complete()
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
    fn available(&self) -> Result<()> {
        if self.poisoned || self.shutdown {
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
        if self.poisoned || !self.entries.is_empty() {
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

/// Fixed capacity includes resident, provisional/shadow, and quarantined slots.
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
        Ok(Self {
            inner: Rc::new(RefCell::new(Inner {
                identity: Rc::new(()),
                context,
                storage: Some(storage),
                entries: BTreeMap::new(),
                next_nonce: 1,
                poisoned: false,
                shutdown: false,
            })),
        })
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
            free_pages: if inner.poisoned { 0 } else { stats.free_slots },
        }
    }
    /// An error is not a cleanup ACK while this is true. Available even when
    /// poisoned, so adapters retain logical custody despite no returned handle.
    pub fn has_unresolved_custody(&self) -> bool {
        self.inner.borrow().poisoned
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
                    inner.storage_mut().clear_transaction_slot(&context, slot)?;
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
                phase: Phase::Active,
                generation: None,
                quarantine: Vec::new(),
                retired: Vec::new(),
            },
        );
        inner.next_nonce = next;
        match submit(&mut inner, &copies) {
            Ok(fence) => {
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
        if !inner.entry(tx)?.entered {
            return Err(error("CUDA KV transaction is not entered"));
        }
        let fence = inner.context.record_compute_event()?;
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
