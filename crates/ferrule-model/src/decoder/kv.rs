use super::{
    DecoderKvBackend, DecoderKvCapacity, DecoderKvPageStatus, DecoderKvPageView, DecoderKvPrepare,
    DecoderSequenceState, KvEndProgress, PackedDecoderBatch,
};
use crate::transformer::{KvAppendRequest, KvHistory, KvView as TransformerKvView};
use ferrule_backend::cpu::{
    self, KvEndProgress as BackendKvEndProgress, KvPageStatus as BackendKvPageStatus,
    PagedKvHistory,
};
use ferrule_common::Result;
use ferrule_common::execution::{KvElementType, KvLayoutSchema, KvPageId, KvPlaneDescriptor};
use std::collections::{BTreeMap, BTreeSet};

pub use ferrule_backend::cpu::{CpuKvPlaneStorage, CpuPagedKvTransaction};

/// Storage-neutral ownership ledger for transactional paged KV.
///
/// Physical pools retain only typed planes, snapshots, recurrent checkpoints,
/// and backend descriptors. This ledger is the single source of truth for page
/// custody, COW/new-page provisional publication, preemption, and release.
#[derive(Debug)]
pub struct PagedKvOwnership<T> {
    capacity: usize,
    resident: BTreeSet<KvPageId>,
    preempted: BTreeSet<KvPageId>,
    pending: BTreeMap<ferrule_common::execution::ExecutionTransactionId, PagedKvTransaction<T>>,
}

#[derive(Debug)]
pub struct PagedKvTransaction<T> {
    new_pages: BTreeSet<KvPageId>,
    writable_pages: BTreeSet<KvPageId>,
    cow_replacements: Box<[ferrule_common::execution::KvCowReplacement]>,
    protected_pages: BTreeSet<KvPageId>,
    custody: Box<[super::DecoderKvSequenceCustody]>,
    batch: Option<PackedDecoderBatch>,
    entered: bool,
    payload: T,
}

impl<T> PagedKvOwnership<T> {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(kv_error("paged KV ownership capacity must be non-zero"));
        }
        Ok(Self {
            capacity,
            resident: BTreeSet::new(),
            preempted: BTreeSet::new(),
            pending: BTreeMap::new(),
        })
    }

    pub fn prepare(&mut self, request: DecoderKvPrepare<'_>, payload: T) -> Result<()> {
        if self.pending.contains_key(&request.transaction) {
            return Err(kv_error(format!(
                "paged KV transaction {} is already prepared",
                request.transaction.get()
            )));
        }
        let new_pages = unique_page_set(request.new_pages, "new")?;
        let mut writable_pages = unique_page_set(request.writable_pages, "writable")?;
        let protected_pages = unique_page_set(request.protected_pages, "protected")?;
        let mut replacements = BTreeSet::new();
        for cow in request.cow_replacements {
            if cow.source == cow.replacement
                || !self.resident.contains(&cow.source)
                || self.status(cow.replacement) != DecoderKvPageStatus::Vacant
                || !replacements.insert(cow.replacement)
            {
                return Err(kv_error(format!(
                    "invalid paged KV COW replacement {cow:?}"
                )));
            }
            writable_pages.insert(cow.replacement);
        }
        if new_pages
            .iter()
            .any(|page| self.status(*page) != DecoderKvPageStatus::Vacant)
        {
            return Err(kv_error("new paged KV page is not vacant"));
        }
        writable_pages.extend(new_pages.iter().copied());
        for (owner, active) in &self.pending {
            if let Some(page) = writable_pages.intersection(&active.protected_pages).next() {
                return Err(kv_error(format!(
                    "paged KV transaction {} cannot write page {} while transaction {} retains custody",
                    request.transaction.get(),
                    page.0,
                    owner.get()
                )));
            }
            if let Some(page) = protected_pages.intersection(&active.writable_pages).next() {
                return Err(kv_error(format!(
                    "paged KV transaction {} cannot read page {} while transaction {} writes it",
                    request.transaction.get(),
                    page.0,
                    owner.get()
                )));
            }
        }
        let provisional = new_pages.len().saturating_add(replacements.len());
        if self.resident.len().saturating_add(provisional) > self.capacity {
            return Err(kv_error("paged KV transaction exceeds physical capacity"));
        }
        self.pending.insert(
            request.transaction,
            PagedKvTransaction {
                new_pages,
                writable_pages,
                cow_replacements: request.cow_replacements.into(),
                protected_pages,
                custody: request.sequences.into(),
                batch: None,
                entered: false,
                payload,
            },
        );
        Ok(())
    }

    pub fn enter(
        &mut self,
        transaction: ferrule_common::execution::ExecutionTransactionId,
        batch: &PackedDecoderBatch,
    ) -> Result<&mut T> {
        let pending = self.pending.get_mut(&transaction).ok_or_else(|| {
            kv_error(format!(
                "paged KV transaction {} is not prepared",
                transaction.get()
            ))
        })?;
        if pending.entered {
            return Err(kv_error("paged KV transaction is already entered"));
        }
        match &pending.batch {
            Some(prepared) if prepared != batch => {
                return Err(kv_error(
                    "paged KV transaction entered with a different packed batch",
                ));
            }
            Some(_) => {}
            None => pending.batch = Some(batch.clone()),
        }
        pending.entered = true;
        Ok(&mut pending.payload)
    }

    pub fn contains_transaction(
        &self,
        transaction: ferrule_common::execution::ExecutionTransactionId,
    ) -> bool {
        self.pending.contains_key(&transaction)
    }

    pub fn transaction(
        &self,
        transaction: ferrule_common::execution::ExecutionTransactionId,
    ) -> Result<&PagedKvTransaction<T>> {
        self.pending
            .get(&transaction)
            .ok_or_else(|| kv_error("paged KV transaction is not prepared"))
    }

    pub fn transaction_mut(
        &mut self,
        transaction: ferrule_common::execution::ExecutionTransactionId,
    ) -> Result<&mut PagedKvTransaction<T>> {
        self.pending
            .get_mut(&transaction)
            .ok_or_else(|| kv_error("paged KV transaction is not prepared"))
    }

    pub fn payload_mut(
        &mut self,
        transaction: ferrule_common::execution::ExecutionTransactionId,
    ) -> Result<&mut T> {
        self.pending
            .get_mut(&transaction)
            .map(|pending| &mut pending.payload)
            .ok_or_else(|| kv_error("paged KV transaction is not prepared"))
    }
}

impl<T> PagedKvTransaction<T> {
    pub fn custody(&self) -> &[super::DecoderKvSequenceCustody] {
        &self.custody
    }

    pub fn batch(&self) -> Option<&PackedDecoderBatch> {
        self.batch.as_ref()
    }

    pub const fn entered(&self) -> bool {
        self.entered
    }

    pub fn payload(&self) -> &T {
        &self.payload
    }

    pub fn payload_mut(&mut self) -> &mut T {
        &mut self.payload
    }
}

impl<T> PagedKvOwnership<T> {
    pub fn retain_provisional_pages(
        &mut self,
        transaction: ferrule_common::execution::ExecutionTransactionId,
        retained: &BTreeSet<KvPageId>,
    ) -> Result<()> {
        let pending = self.transaction_mut(transaction)?;
        pending.new_pages.retain(|page| retained.contains(page));
        pending.cow_replacements = pending
            .cow_replacements
            .iter()
            .copied()
            .filter(|cow| retained.contains(&cow.replacement))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        pending
            .writable_pages
            .retain(|page| retained.contains(page) || pending.protected_pages.contains(page));
        Ok(())
    }

    pub fn leave(
        &mut self,
        transaction: ferrule_common::execution::ExecutionTransactionId,
    ) -> Result<()> {
        let pending = self.pending.get_mut(&transaction).ok_or_else(|| {
            kv_error(format!(
                "paged KV transaction {} is not prepared",
                transaction.get()
            ))
        })?;
        if !pending.entered {
            return Err(kv_error("paged KV transaction is not entered"));
        }
        pending.entered = false;
        Ok(())
    }

    pub fn commit(
        &mut self,
        transaction: ferrule_common::execution::ExecutionTransactionId,
    ) -> Result<T> {
        let pending = self.take_quiescent(transaction, "commit")?;
        self.resident.extend(pending.new_pages);
        self.resident
            .extend(pending.cow_replacements.iter().map(|cow| cow.replacement));
        Ok(pending.payload)
    }

    pub fn rollback(
        &mut self,
        transaction: ferrule_common::execution::ExecutionTransactionId,
    ) -> Result<T> {
        Ok(self.take_quiescent(transaction, "rollback")?.payload)
    }

    pub fn validate_release(&self, pages: &[KvPageId]) -> Result<()> {
        let pages = unique_page_set(pages, "release")?;
        self.ensure_unowned(&pages, "release")?;
        if pages
            .iter()
            .any(|page| !self.resident.contains(page) && !self.preempted.contains(page))
        {
            return Err(kv_error("cannot release an unknown paged KV page"));
        }
        Ok(())
    }

    pub fn release(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.validate_release(pages)?;
        let pages = unique_page_set(pages, "release")?;
        for page in pages {
            self.resident.remove(&page);
            self.preempted.remove(&page);
        }
        Ok(())
    }

    pub fn validate_preempt(&self, pages: &[KvPageId]) -> Result<()> {
        let pages = unique_page_set(pages, "preempt")?;
        self.ensure_unowned(&pages, "preempt")?;
        if pages.iter().any(|page| !self.resident.contains(page)) {
            return Err(kv_error("cannot preempt a non-resident paged KV page"));
        }
        Ok(())
    }

    pub fn preempt(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.validate_preempt(pages)?;
        let pages = unique_page_set(pages, "preempt")?;
        for page in pages {
            self.resident.remove(&page);
            self.preempted.insert(page);
        }
        Ok(())
    }

    pub fn validate_restore(&self, pages: &[KvPageId]) -> Result<()> {
        let pages = unique_page_set(pages, "restore")?;
        self.ensure_unowned(&pages, "restore")?;
        if pages.iter().any(|page| !self.preempted.contains(page))
            || self.resident.len().saturating_add(pages.len()) > self.capacity
        {
            return Err(kv_error("cannot restore paged KV pages"));
        }
        Ok(())
    }

    pub fn restore(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.validate_restore(pages)?;
        let pages = unique_page_set(pages, "restore")?;
        for page in pages {
            self.preempted.remove(&page);
            self.resident.insert(page);
        }
        Ok(())
    }

    pub fn status(&self, page: KvPageId) -> DecoderKvPageStatus {
        if self.resident.contains(&page) {
            DecoderKvPageStatus::Resident
        } else if self.preempted.contains(&page) {
            DecoderKvPageStatus::Preempted
        } else {
            DecoderKvPageStatus::Vacant
        }
    }

    pub fn capacity(&self) -> DecoderKvCapacity {
        DecoderKvCapacity {
            physical_pages: self.capacity,
            resident_pages: self.resident.len(),
            preempted_pages: self.preempted.len(),
            active_transactions: self.pending.len(),
            free_pages: self
                .capacity
                .saturating_sub(self.resident.len())
                .saturating_sub(
                    self.pending
                        .values()
                        .map(|pending| pending.new_pages.len() + pending.cow_replacements.len())
                        .sum::<usize>(),
                ),
        }
    }

    pub fn is_quiescent(&self) -> bool {
        self.pending.is_empty()
    }

    fn take_quiescent(
        &mut self,
        transaction: ferrule_common::execution::ExecutionTransactionId,
        operation: &str,
    ) -> Result<PagedKvTransaction<T>> {
        let pending = self.pending.get(&transaction).ok_or_else(|| {
            kv_error(format!(
                "paged KV transaction {} is not prepared",
                transaction.get()
            ))
        })?;
        if pending.entered {
            return Err(kv_error(format!(
                "cannot {operation} an entered paged KV transaction"
            )));
        }
        Ok(self
            .pending
            .remove(&transaction)
            .expect("validated paged KV transaction exists"))
    }

    fn ensure_unowned(&self, pages: &BTreeSet<KvPageId>, operation: &str) -> Result<()> {
        for (transaction, pending) in &self.pending {
            if let Some(page) = pages.intersection(&pending.protected_pages).next() {
                return Err(kv_error(format!(
                    "cannot {operation} page {} while transaction {} retains custody",
                    page.0,
                    transaction.get()
                )));
            }
        }
        Ok(())
    }
}

fn unique_page_set(pages: &[KvPageId], label: &str) -> Result<BTreeSet<KvPageId>> {
    let unique = pages.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != pages.len() {
        return Err(kv_error(format!("duplicate {label} paged KV page")));
    }
    Ok(unique)
}

/// Physical page-pool protocol used by model-neutral paged decoder backends.
pub trait PhysicalKvPool {
    type SequenceState: super::DecoderSequence;
    type Transaction;
    type KvView;

    fn configured_capacity(&self) -> usize;
    fn configure_capacity(&mut self, max_pages: usize) -> Result<()>;
    fn prepare(&mut self, request: DecoderKvPrepare<'_>) -> Result<Self::Transaction>;
    fn enter(
        &mut self,
        transaction: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
        states: &mut [Self::SequenceState],
    ) -> Result<()>;
    fn active_view(
        &mut self,
        transaction: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
    ) -> Result<Self::KvView>;
    fn proposal_view(
        &mut self,
        _transaction: ferrule_common::execution::ExecutionTransactionId,
        _state: &mut Self::SequenceState,
    ) -> Result<Self::KvView> {
        Err(kv_error("physical KV pool has no proposal view"))
    }
    fn leave(&mut self, transaction: &mut Self::Transaction) -> Result<()>;
    fn commit(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress>;
    fn abort(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress>;
    fn release(&mut self, pages: &[KvPageId]) -> Result<()>;
    fn preempt(&mut self, pages: &[KvPageId]) -> Result<()>;
    fn restore(&mut self, pages: &[KvPageId]) -> Result<()>;
    fn retain_provisional(
        &mut self,
        _transaction: &mut Self::Transaction,
        _sources: &[Self::SequenceState],
        _working_states: &mut [Self::SequenceState],
        _batch: &PackedDecoderBatch,
        _executed_rows: &[usize],
        _retained_rows: &[usize],
    ) -> Result<BTreeSet<KvPageId>> {
        Err(kv_error(
            "physical KV pool has no provisional-prefix contract",
        ))
    }
    fn shutdown(&mut self) -> Result<()>;
}

/// Generic, non-model transaction handle. All mutable transaction state remains
/// in PagedKvOwnership.
#[derive(Debug)]
pub struct PagedKvTransactionHandle {
    transaction: ferrule_common::execution::ExecutionTransactionId,
}

impl PagedKvTransactionHandle {
    pub const fn id(&self) -> ferrule_common::execution::ExecutionTransactionId {
        self.transaction
    }
}

/// Generic decoder backend owning the sole paged-KV transaction and page ledger.
pub struct PagedKvBackend<P>
where
    P: PhysicalKvPool,
{
    pool: P,
    ownership: Option<PagedKvOwnership<Option<P::Transaction>>>,
}

/// Standard CPU decoder backend over backend-owned typed physical storage.
pub type CpuPagedKvBackend = PagedKvBackend<CpuPagedKvPool>;

impl<P> PagedKvBackend<P>
where
    P: PhysicalKvPool,
{
    pub fn new(pool: P) -> Self {
        let capacity = pool.configured_capacity();
        Self {
            pool,
            ownership: (capacity != 0)
                .then(|| PagedKvOwnership::new(capacity).expect("non-zero physical capacity")),
        }
    }

    pub const fn pool(&self) -> &P {
        &self.pool
    }

    pub fn pool_mut(&mut self) -> &mut P {
        &mut self.pool
    }

    pub fn into_pool(self) -> P {
        self.pool
    }

    pub fn retain_provisional(
        &mut self,
        transaction: &mut PagedKvTransactionHandle,
        sources: &[P::SequenceState],
        working_states: &mut [P::SequenceState],
        executed_rows: &[usize],
        retained_rows: &[usize],
    ) -> Result<()> {
        self.validate_handle(transaction)?;
        if self
            .ownership()?
            .transaction(transaction.transaction)?
            .entered()
        {
            return Err(kv_error(
                "cannot retain a provisional prefix while paged KV is entered",
            ));
        }
        let batch = self
            .ownership()?
            .transaction(transaction.transaction)?
            .batch()
            .cloned()
            .ok_or_else(|| kv_error("provisional paged KV transaction has no entered batch"))?;
        let (pool, ownership) = (&mut self.pool, self.ownership.as_mut());
        let ownership = ownership.ok_or_else(|| kv_error("paged KV capacity is not configured"))?;
        let physical = ownership
            .payload_mut(transaction.transaction)?
            .as_mut()
            .ok_or_else(|| kv_error("paged KV transaction lost its physical reservation"))?;
        let retained = pool.retain_provisional(
            physical,
            sources,
            working_states,
            &batch,
            executed_rows,
            retained_rows,
        )?;
        ownership.retain_provisional_pages(transaction.transaction, &retained)
    }

    fn ownership(&self) -> Result<&PagedKvOwnership<Option<P::Transaction>>> {
        self.ownership
            .as_ref()
            .ok_or_else(|| kv_error("paged KV capacity is not configured"))
    }

    fn ownership_mut(&mut self) -> Result<&mut PagedKvOwnership<Option<P::Transaction>>> {
        self.ownership
            .as_mut()
            .ok_or_else(|| kv_error("paged KV capacity is not configured"))
    }

    fn validate_handle(&self, transaction: &PagedKvTransactionHandle) -> Result<()> {
        if self
            .ownership()?
            .contains_transaction(transaction.transaction)
        {
            Ok(())
        } else {
            Err(kv_error("paged KV transaction is no longer pending"))
        }
    }
}

impl<P> DecoderKvBackend for PagedKvBackend<P>
where
    P: PhysicalKvPool,
{
    type SequenceState = P::SequenceState;
    type Transaction = PagedKvTransactionHandle;
    type KvView = P::KvView;

    fn configure_capacity(&mut self, max_pages: usize) -> Result<()> {
        if self
            .ownership
            .as_ref()
            .is_some_and(|ownership| !ownership.is_quiescent())
        {
            return Err(kv_error(
                "cannot configure paged KV capacity while transactions are active",
            ));
        }
        self.pool.configure_capacity(max_pages)?;
        self.ownership = Some(PagedKvOwnership::new(max_pages)?);
        Ok(())
    }

    fn prepare(&mut self, request: DecoderKvPrepare<'_>) -> Result<Self::Transaction> {
        for snapshot in request.page_statuses {
            if DecoderKvBackend::page_status(self, snapshot.page) != snapshot.status {
                return Err(kv_error(format!(
                    "paged KV page {} changed from {:?} before prepare",
                    snapshot.page.0, snapshot.status
                )));
            }
        }
        if request.capacity != self.capacity() {
            return Err(kv_error("paged KV capacity changed before prepare"));
        }
        self.ownership_mut()?.prepare(request, None)?;
        match self.pool.prepare(request) {
            Ok(physical) => {
                *self.ownership_mut()?.payload_mut(request.transaction)? = Some(physical);
                Ok(PagedKvTransactionHandle {
                    transaction: request.transaction,
                })
            }
            Err(error) => {
                let _ = self.ownership_mut()?.rollback(request.transaction);
                Err(error)
            }
        }
    }

    fn enter(
        &mut self,
        transaction: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
        states: &mut [Self::SequenceState],
    ) -> Result<()> {
        self.validate_handle(transaction)?;
        validate_sequence_custody(
            self.ownership()?.transaction(transaction.transaction)?,
            batch,
            states,
        )?;
        self.ownership_mut()?
            .enter(transaction.transaction, batch)?;
        let (pool, ownership) = (&mut self.pool, self.ownership.as_mut());
        let physical = ownership
            .ok_or_else(|| kv_error("paged KV capacity is not configured"))?
            .payload_mut(transaction.transaction)?
            .as_mut()
            .ok_or_else(|| kv_error("paged KV transaction lost its physical reservation"))?;
        if let Err(error) = pool.enter(physical, batch, states) {
            let _ = self.ownership_mut()?.leave(transaction.transaction);
            return Err(error);
        }
        Ok(())
    }

    fn active_view(&mut self, transaction: &mut Self::Transaction) -> Result<Self::KvView> {
        self.validate_handle(transaction)?;
        let batch = self
            .ownership()?
            .transaction(transaction.transaction)?
            .batch()
            .cloned()
            .ok_or_else(|| kv_error("paged KV transaction has no entered batch"))?;
        let (pool, ownership) = (&mut self.pool, self.ownership.as_mut());
        let physical = ownership
            .ok_or_else(|| kv_error("paged KV capacity is not configured"))?
            .payload_mut(transaction.transaction)?
            .as_mut()
            .ok_or_else(|| kv_error("paged KV transaction lost its physical reservation"))?;
        pool.active_view(physical, &batch)
    }

    fn proposal_view(
        &mut self,
        transaction: ferrule_common::execution::ExecutionTransactionId,
        state: &mut Self::SequenceState,
    ) -> Result<Self::KvView> {
        if self
            .ownership
            .as_ref()
            .is_some_and(|ownership| ownership.contains_transaction(transaction))
        {
            return Err(kv_error(
                "proposal view transaction conflicts with pending paged KV custody",
            ));
        }
        self.pool.proposal_view(transaction, state)
    }

    fn leave(&mut self, transaction: &mut Self::Transaction) -> Result<()> {
        self.validate_handle(transaction)?;
        let (pool, ownership) = (&mut self.pool, self.ownership.as_mut());
        let physical = ownership
            .ok_or_else(|| kv_error("paged KV capacity is not configured"))?
            .payload_mut(transaction.transaction)?
            .as_mut()
            .ok_or_else(|| kv_error("paged KV transaction lost its physical reservation"))?;
        pool.leave(physical)?;
        self.ownership_mut()?.leave(transaction.transaction)
    }

    fn commit(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        let id = transaction
            .as_ref()
            .ok_or_else(|| kv_error("paged KV commit transaction is absent"))?
            .transaction;
        self.validate_handle(transaction.as_ref().expect("checked above"))?;
        if self.ownership()?.transaction(id)?.entered() {
            return Err(kv_error("cannot commit an entered paged KV transaction"));
        }
        let progress = {
            let (pool, ownership) = (&mut self.pool, self.ownership.as_mut());
            let physical = ownership
                .ok_or_else(|| kv_error("paged KV capacity is not configured"))?
                .payload_mut(id)?;
            pool.commit(physical)?
        };
        match progress {
            KvEndProgress::Pending => {}
            KvEndProgress::Complete => {
                self.ownership_mut()?.commit(id)?;
                transaction.take();
            }
            KvEndProgress::ConsumedRejected => {
                self.ownership_mut()?.rollback(id)?;
                transaction.take();
            }
        }
        Ok(progress)
    }

    fn rollback(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        let id = transaction
            .as_ref()
            .ok_or_else(|| kv_error("paged KV rollback transaction is absent"))?
            .transaction;
        self.validate_handle(transaction.as_ref().expect("checked above"))?;
        if self.ownership()?.transaction(id)?.entered() {
            return Err(kv_error("cannot roll back an entered paged KV transaction"));
        }
        let progress = {
            let (pool, ownership) = (&mut self.pool, self.ownership.as_mut());
            let physical = ownership
                .ok_or_else(|| kv_error("paged KV capacity is not configured"))?
                .payload_mut(id)?;
            pool.abort(physical)?
        };
        if !matches!(progress, KvEndProgress::Pending) {
            self.ownership_mut()?.rollback(id)?;
            transaction.take();
        }
        Ok(progress)
    }

    fn release(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.ownership()?.validate_release(pages)?;
        self.pool.release(pages)?;
        self.ownership_mut()?.release(pages)
    }

    fn preempt(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.ownership()?.validate_preempt(pages)?;
        self.pool.preempt(pages)?;
        self.ownership_mut()?.preempt(pages)
    }

    fn restore(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.ownership()?.validate_restore(pages)?;
        self.pool.restore(pages)?;
        self.ownership_mut()?.restore(pages)
    }

    fn capacity(&self) -> DecoderKvCapacity {
        self.ownership
            .as_ref()
            .map_or_else(DecoderKvCapacity::default, PagedKvOwnership::capacity)
    }

    fn page_status(&self, page: KvPageId) -> DecoderKvPageStatus {
        self.ownership
            .as_ref()
            .map_or(DecoderKvPageStatus::Vacant, |ownership| {
                ownership.status(page)
            })
    }

    fn shutdown(&mut self) -> Result<()> {
        if self
            .ownership
            .as_ref()
            .is_some_and(|ownership| !ownership.is_quiescent())
        {
            return Err(kv_error(
                "cannot shut down paged KV with active transaction custody",
            ));
        }
        self.pool.shutdown()
    }
}

impl<P> DecoderKvPageView for PagedKvBackend<P>
where
    P: PhysicalKvPool,
{
    fn page_status(&self, page: KvPageId) -> DecoderKvPageStatus {
        DecoderKvBackend::page_status(self, page)
    }
}

fn validate_sequence_custody<S, T>(
    transaction: &PagedKvTransaction<T>,
    batch: &PackedDecoderBatch,
    states: &[S],
) -> Result<()>
where
    S: super::DecoderSequence,
{
    if states.len() != batch.sequences().len() {
        return Err(kv_error(format!(
            "paged KV state count {} does not match packed sequence count {}",
            states.len(),
            batch.sequences().len()
        )));
    }
    if transaction.custody().is_empty() {
        return Ok(());
    }
    if transaction.custody().len() != batch.sequences().len() {
        return Err(kv_error(
            "paged KV sequence custody count changed before enter",
        ));
    }
    for (index, ((custody, sequence), state)) in transaction
        .custody()
        .iter()
        .zip(batch.sequences())
        .zip(states)
        .enumerate()
    {
        if custody.topology_id != sequence.topology_id()
            || custody.topology_id != state.topology_id()
            || custody.page_state_slot != sequence.page_state_slot()
            || custody.page_generation != sequence.page_generation()
            || custody.execution_generation != sequence.execution_generation()
            || custody.execution_generation != state.core().generation()
            || custody.context_len != sequence.context_len()
            || custody.context_len != state.core().position()
            || custody.query_len != sequence.query_len()
        {
            return Err(kv_error(format!(
                "paged KV sequence custody {index} changed before enter"
            )));
        }
    }
    Ok(())
}

/// Dtype-aware plane strategy accepted by generic physical pools.
pub trait DecoderKvPlaneStrategy: std::fmt::Debug + Send + Sync {
    fn planes(&self) -> &[KvPlaneDescriptor];
    fn page_size(&self) -> usize;
    fn max_sequence_len(&self) -> usize;
}

impl<T> DecoderKvPlaneStrategy for T
where
    T: KvLayoutSchema,
{
    fn planes(&self) -> &[KvPlaneDescriptor] {
        KvLayoutSchema::planes(self)
    }

    fn page_size(&self) -> usize {
        KvLayoutSchema::page_size(self)
    }

    fn max_sequence_len(&self) -> usize {
        KvLayoutSchema::max_sequence_len(self)
    }
}

/// Extension point for latent-attention layouts with model-defined plane roles.
pub trait MlaPlaneStrategy: DecoderKvPlaneStrategy {
    fn latent_plane(&self) -> usize;

    fn index_plane(&self) -> Option<usize> {
        None
    }

    fn compressed_plane(&self) -> Option<usize> {
        None
    }
}

/// Standard independent key/value planes for grouped-query attention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StandardGqaPlanes {
    page_size: usize,
    max_sequence_len: usize,
    planes: [KvPlaneDescriptor; 2],
}

impl StandardGqaPlanes {
    pub fn new(
        layer_count: usize,
        kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        max_sequence_len: usize,
        element_type: KvElementType,
    ) -> Result<Self> {
        if [layer_count, kv_heads, head_dim, page_size, max_sequence_len].contains(&0) {
            return Err(kv_error("standard GQA KV dimensions must be non-zero"));
        }
        let elements_per_token = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| kv_error("standard GQA elements per token overflow"))?;
        let planes = [
            KvPlaneDescriptor::new(
                "standard_gqa.key",
                elements_per_token,
                layer_count,
                element_type,
            ),
            KvPlaneDescriptor::new(
                "standard_gqa.value",
                elements_per_token,
                layer_count,
                element_type,
            ),
        ];
        for plane in planes {
            plane
                .checked_page_bytes(page_size)
                .ok_or_else(|| kv_error("standard GQA KV page size overflow"))?;
        }
        Ok(Self {
            page_size,
            max_sequence_len,
            planes,
        })
    }
}

impl KvLayoutSchema for StandardGqaPlanes {
    fn planes(&self) -> &[KvPlaneDescriptor] {
        &self.planes
    }

    fn page_size(&self) -> usize {
        self.page_size
    }

    fn max_sequence_len(&self) -> usize {
        self.max_sequence_len
    }
}

/// Decoder transaction adapter over backend-owned physical CPU storage.
#[derive(Clone)]
pub struct CpuPagedKvPool {
    inner: cpu::CpuPagedKvPool,
}

impl CpuPagedKvPool {
    pub fn new(
        planes: impl IntoIterator<Item = KvPlaneDescriptor>,
        page_size: usize,
        capacity: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: cpu::CpuPagedKvPool::new(planes, page_size, capacity)?,
        })
    }

    pub fn from_strategy(strategy: &impl DecoderKvPlaneStrategy, capacity: usize) -> Result<Self> {
        Self::new(
            strategy.planes().iter().copied(),
            strategy.page_size(),
            capacity,
        )
    }

    pub fn planes(&self) -> Vec<KvPlaneDescriptor> {
        self.inner.planes()
    }

    pub fn page_size(&self) -> usize {
        self.inner.page_size()
    }

    pub fn physical_slot(&self, page: KvPageId) -> Option<usize> {
        self.inner.physical_slot(page)
    }

    pub fn free_slot_count(&self) -> usize {
        self.inner.free_slot_count()
    }

    pub fn active_transaction_count(&self) -> usize {
        self.inner.active_transaction_count()
    }
}

impl DecoderKvPageView for CpuPagedKvPool {
    fn page_status(&self, page: KvPageId) -> DecoderKvPageStatus {
        map_page_status(self.inner.page_status(page))
    }
}

impl PhysicalKvPool for CpuPagedKvPool {
    type SequenceState = DecoderSequenceState<(), ()>;
    type Transaction = CpuPagedKvTransaction;
    type KvView = CpuKvView;

    fn configured_capacity(&self) -> usize {
        self.inner.capacity().physical_pages
    }

    fn configure_capacity(&mut self, max_pages: usize) -> Result<()> {
        self.inner.configure_capacity(max_pages)
    }

    fn prepare(&mut self, request: DecoderKvPrepare<'_>) -> Result<Self::Transaction> {
        self.inner.prepare(cpu::PagedKvPrepare {
            transaction: request.transaction,
            new_pages: request.new_pages,
            writable_pages: request.writable_pages,
            cow_replacements: request.cow_replacements,
            protected_pages: request.protected_pages,
        })
    }

    fn enter(
        &mut self,
        transaction: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
        _states: &mut [Self::SequenceState],
    ) -> Result<()> {
        self.inner.enter(transaction, backend_batch(batch))
    }

    fn active_view(
        &mut self,
        transaction: &mut Self::Transaction,
        _batch: &PackedDecoderBatch,
    ) -> Result<Self::KvView> {
        Ok(CpuKvView {
            inner: self.inner.active_view(transaction)?,
        })
    }

    fn leave(&mut self, transaction: &mut Self::Transaction) -> Result<()> {
        self.inner.leave(transaction)
    }

    fn commit(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        self.inner.commit(transaction).map(map_end_progress)
    }

    fn abort(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        self.inner.abort(transaction).map(map_end_progress)
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

/// Short-lived model transaction view over backend-owned typed pages.
#[derive(Clone)]
pub struct CpuKvView {
    inner: cpu::CpuKvView,
}

impl CpuKvView {
    pub const fn transaction(&self) -> ferrule_common::execution::ExecutionTransactionId {
        self.inner.transaction()
    }

    pub fn validate_batch(&self, batch: &PackedDecoderBatch) -> Result<()> {
        self.inner.validate_batch(&backend_batch(batch))
    }

    pub fn physical_slot(&self, page: KvPageId) -> Option<usize> {
        self.inner.physical_slot(page)
    }

    pub fn with_f32_page<R>(
        &self,
        page: KvPageId,
        plane: usize,
        read: impl FnOnce(&[f32]) -> R,
    ) -> Result<R> {
        self.inner.with_f32_page(page, plane, read)
    }

    pub fn with_bf16_page<R>(
        &self,
        page: KvPageId,
        plane: usize,
        read: impl FnOnce(&[u16]) -> R,
    ) -> Result<R> {
        self.inner.with_bf16_page(page, plane, read)
    }

    pub fn with_f32_page_mut<R>(
        &mut self,
        page: KvPageId,
        plane: usize,
        write: impl FnOnce(&mut [f32]) -> R,
    ) -> Result<R> {
        self.inner.with_f32_page_mut(page, plane, write)
    }

    pub fn with_bf16_page_mut<R>(
        &mut self,
        page: KvPageId,
        plane: usize,
        write: impl FnOnce(&mut [u16]) -> R,
    ) -> Result<R> {
        self.inner.with_bf16_page_mut(page, plane, write)
    }
}

impl TransformerKvView for CpuKvView {
    fn append(&mut self, request: KvAppendRequest<'_>) -> Result<()> {
        self.inner.append(
            request.layer,
            request.metadata.row_sequence_ids(),
            request.metadata.row_positions(),
            request.kv_heads,
            request.head_dim,
            request.key,
            request.value,
        )
    }

    fn history(
        &self,
        layer: usize,
        sequence: usize,
        through_position: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<KvHistory> {
        let history = self
            .inner
            .history(layer, sequence, through_position, kv_heads, head_dim)?;
        Ok(KvHistory {
            tokens: history.tokens,
            key: history.key,
            value: history.value,
        })
    }
}

fn backend_batch(batch: &PackedDecoderBatch) -> cpu::PagedKvBatch {
    cpu::PagedKvBatch {
        row_sequence_ids: batch.row_to_sequence().into(),
        row_positions: batch.positions().into(),
        sequences: batch
            .sequences()
            .iter()
            .map(|sequence| cpu::PagedKvSequence {
                sequence_len: sequence.sequence_len(),
                block_table: sequence.block_table().into(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    }
}

const fn map_page_status(status: BackendKvPageStatus) -> DecoderKvPageStatus {
    match status {
        BackendKvPageStatus::Vacant => DecoderKvPageStatus::Vacant,
        BackendKvPageStatus::Resident => DecoderKvPageStatus::Resident,
        BackendKvPageStatus::Preempted => DecoderKvPageStatus::Preempted,
    }
}

const fn map_end_progress(progress: BackendKvEndProgress) -> KvEndProgress {
    match progress {
        BackendKvEndProgress::Pending => KvEndProgress::Pending,
        BackendKvEndProgress::Complete => KvEndProgress::Complete,
        BackendKvEndProgress::ConsumedRejected => KvEndProgress::ConsumedRejected,
    }
}

fn kv_error(message: impl Into<String>) -> ferrule_common::Error {
    ferrule_common::Error::Execution {
        message: message.into(),
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use crate::decoder::{DecoderKvCapacity, DecoderKvPrepare};
    use ferrule_common::execution::ExecutionTransactionId;

    fn prepare<'a>(
        transaction: ExecutionTransactionId,
        new_pages: &'a [KvPageId],
        writable_pages: &'a [KvPageId],
        cow_replacements: &'a [ferrule_common::execution::KvCowReplacement],
        protected_pages: &'a [KvPageId],
        capacity: DecoderKvCapacity,
    ) -> DecoderKvPrepare<'a> {
        DecoderKvPrepare {
            transaction,
            sequences: &[],
            new_pages,
            writable_pages,
            cow_replacements,
            protected_pages,
            capacity,
            page_statuses: &[],
        }
    }

    #[test]
    fn standard_gqa_cpu_pool_preserves_descriptor_identity() {
        let strategy = StandardGqaPlanes::new(2, 3, 4, 8, 64, KvElementType::Bf16).unwrap();
        let descriptors = KvLayoutSchema::planes(&strategy).to_vec();
        assert!(
            descriptors
                .iter()
                .all(|plane| plane.element_type == KvElementType::Bf16)
        );

        let pool = CpuPagedKvPool::from_strategy(&strategy, 2).unwrap();
        assert_eq!(pool.planes(), descriptors);
    }

    #[test]
    fn ownership_publishes_new_and_cow_pages_only_on_commit() {
        let mut ownership = PagedKvOwnership::new(4).unwrap();
        let first = ExecutionTransactionId::new(1).unwrap();
        ownership
            .prepare(
                prepare(
                    first,
                    &[KvPageId(10)],
                    &[],
                    &[],
                    &[KvPageId(10)],
                    ownership.capacity(),
                ),
                "first",
            )
            .unwrap();
        assert_eq!(ownership.status(KvPageId(10)), DecoderKvPageStatus::Vacant);
        ownership.commit(first).unwrap();
        assert_eq!(
            ownership.status(KvPageId(10)),
            DecoderKvPageStatus::Resident
        );

        let cow = ferrule_common::execution::KvCowReplacement {
            source: KvPageId(10),
            replacement: KvPageId(11),
            logical_page: 0,
        };
        let second = ExecutionTransactionId::new(2).unwrap();
        ownership
            .prepare(
                prepare(
                    second,
                    &[],
                    &[],
                    &[cow],
                    &[KvPageId(10), KvPageId(11)],
                    ownership.capacity(),
                ),
                "second",
            )
            .unwrap();
        ownership.rollback(second).unwrap();
        assert_eq!(ownership.status(KvPageId(11)), DecoderKvPageStatus::Vacant);
    }

    #[test]
    fn ownership_blocks_page_lifecycle_while_transaction_retains_custody() {
        let mut ownership = PagedKvOwnership::new(2).unwrap();
        let seed = ExecutionTransactionId::new(3).unwrap();
        ownership
            .prepare(
                prepare(
                    seed,
                    &[KvPageId(20)],
                    &[],
                    &[],
                    &[KvPageId(20)],
                    ownership.capacity(),
                ),
                (),
            )
            .unwrap();
        ownership.commit(seed).unwrap();

        let reader = ExecutionTransactionId::new(4).unwrap();
        ownership
            .prepare(
                prepare(reader, &[], &[], &[], &[KvPageId(20)], ownership.capacity()),
                (),
            )
            .unwrap();
        assert!(ownership.validate_preempt(&[KvPageId(20)]).is_err());
        ownership.rollback(reader).unwrap();
        ownership.preempt(&[KvPageId(20)]).unwrap();
        assert_eq!(
            ownership.status(KvPageId(20)),
            DecoderKvPageStatus::Preempted
        );
        ownership.restore(&[KvPageId(20)]).unwrap();
        ownership.release(&[KvPageId(20)]).unwrap();
        assert_eq!(ownership.status(KvPageId(20)), DecoderKvPageStatus::Vacant);
    }
}
