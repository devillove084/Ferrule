//! Typed CPU paged KV physical storage and transactions.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::rc::Rc;

use ferrule_common::Result;
use ferrule_common::execution::{
    ExecutionTransactionId, KvCowReplacement, KvElementType, KvPageId, KvPlaneDescriptor,
};

use super::attention::{KvHistory, PagedKvHistory};
use super::operators::{bf16_rne_word, bf16_word_value, cpu_error};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPageStatus {
    Vacant,
    Resident,
    Preempted,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvCapacity {
    pub physical_pages: usize,
    pub resident_pages: usize,
    pub preempted_pages: usize,
    pub active_transactions: usize,
    pub free_pages: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvEndProgress {
    Pending,
    Complete,
    ConsumedRejected,
}

#[derive(Debug)]
struct CommittedLease {
    transaction: ExecutionTransactionId,
    pinned: BTreeSet<KvPageId>,
    phase: PreparedCommitPhase,
}

#[derive(Debug)]
enum PreparedCommitPhase {
    Installed,
    Published,
    Retired {
        pages: Vec<KvPageId>,
        slots: Vec<usize>,
    },
    Aborted {
        slots: Vec<usize>,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct PagedKvPrepare<'a> {
    pub transaction: ExecutionTransactionId,
    pub new_pages: &'a [KvPageId],
    pub writable_pages: &'a [KvPageId],
    pub cow_replacements: &'a [KvCowReplacement],
    pub protected_pages: &'a [KvPageId],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvSequence {
    pub sequence_len: usize,
    pub block_table: Box<[KvPageId]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvBatch {
    pub row_sequence_ids: Box<[usize]>,
    pub row_positions: Box<[usize]>,
    pub sequences: Box<[PagedKvSequence]>,
}

impl PagedKvBatch {
    pub fn validate(&self, page_size: usize) -> Result<()> {
        if page_size == 0
            || self.sequences.is_empty()
            || self.row_sequence_ids.is_empty()
            || self.row_sequence_ids.len() != self.row_positions.len()
            || self
                .row_sequence_ids
                .iter()
                .any(|&sequence| sequence >= self.sequences.len())
        {
            return Err(cpu_error("invalid CPU paged KV batch"));
        }
        for (row, (&sequence, &position)) in self
            .row_sequence_ids
            .iter()
            .zip(self.row_positions.iter())
            .enumerate()
        {
            let plan = &self.sequences[sequence];
            if position >= plan.sequence_len || position / page_size >= plan.block_table.len() {
                return Err(cpu_error(format!(
                    "CPU paged KV row {row} position {position} has no page"
                )));
            }
        }
        Ok(())
    }
}

/// One contiguous CPU allocation for a typed plane.
#[derive(Debug, Clone, PartialEq)]
pub enum CpuKvPlaneStorage {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}

impl CpuKvPlaneStorage {
    pub const fn element_type(&self) -> KvElementType {
        match self {
            Self::F32(_) => KvElementType::F32,
            Self::Bf16(_) => KvElementType::Bf16,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::Bf16(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn clear_range(&mut self, range: Range<usize>) {
        match self {
            Self::F32(values) => values[range].fill(0.0),
            Self::Bf16(values) => values[range].fill(0),
        }
    }

    fn snapshot_range(&self, range: Range<usize>) -> Self {
        match self {
            Self::F32(values) => Self::F32(values[range].to_vec()),
            Self::Bf16(values) => Self::Bf16(values[range].to_vec()),
        }
    }

    fn overwrite_range(&mut self, range: Range<usize>, source: &Self) -> Result<()> {
        match (self, source) {
            (Self::F32(destination), Self::F32(source)) if range.len() == source.len() => {
                destination[range].copy_from_slice(source);
                Ok(())
            }
            (Self::Bf16(destination), Self::Bf16(source)) if range.len() == source.len() => {
                destination[range].copy_from_slice(source);
                Ok(())
            }
            _ => Err(cpu_error("CPU KV snapshot dtype or length mismatch")),
        }
    }
}

#[derive(Debug)]
enum StagedPage {
    Physical {
        slot: usize,
    },
    Shadow {
        resident_slot: usize,
        planes: Vec<CpuKvPlaneStorage>,
    },
}

impl StagedPage {
    const fn physical_slot(&self) -> Option<usize> {
        match self {
            Self::Physical { slot } => Some(*slot),
            Self::Shadow { .. } => None,
        }
    }
}

#[derive(Debug)]
struct ActiveTransaction {
    incarnation: Rc<()>,
    entered: bool,
    batch: Option<PagedKvBatch>,
    pages: BTreeMap<KvPageId, StagedPage>,
    protected_pages: BTreeSet<KvPageId>,
}

#[derive(Debug)]
struct PoolInner {
    identity: Rc<()>,
    planes: Box<[KvPlaneDescriptor]>,
    page_size: usize,
    page_elements: Box<[usize]>,
    capacity: usize,
    storage: Vec<CpuKvPlaneStorage>,
    free_slots: Vec<usize>,
    resident: HashMap<KvPageId, usize>,
    preempted: HashMap<KvPageId, Vec<CpuKvPlaneStorage>>,
    active: HashMap<ExecutionTransactionId, ActiveTransaction>,
    committed: HashMap<u64, CommittedLease>,
    next_lease_nonce: u64,
    shutdown: bool,
}

impl PoolInner {
    fn new(planes: Vec<KvPlaneDescriptor>, page_size: usize, capacity: usize) -> Result<Self> {
        if planes.is_empty() || page_size == 0 {
            return Err(cpu_error("CPU paged KV pool requires planes and page size"));
        }
        let page_elements = planes
            .iter()
            .map(|plane| {
                plane
                    .checked_page_elements(page_size)
                    .filter(|elements| *elements != 0)
                    .ok_or_else(|| cpu_error("CPU KV plane page element count overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut inner = Self {
            identity: Rc::new(()),
            planes: planes.into_boxed_slice(),
            page_size,
            page_elements: page_elements.into_boxed_slice(),
            capacity: 0,
            storage: Vec::new(),
            free_slots: Vec::new(),
            resident: HashMap::new(),
            preempted: HashMap::new(),
            active: HashMap::new(),
            committed: HashMap::new(),
            next_lease_nonce: 1,
            shutdown: false,
        };
        inner.configure_capacity(capacity)?;
        Ok(inner)
    }

    fn configure_capacity(&mut self, capacity: usize) -> Result<()> {
        if capacity == 0 {
            return Err(cpu_error("CPU paged KV capacity must be non-zero"));
        }
        if !self.active.is_empty()
            || !self.committed.is_empty()
            || !self.resident.is_empty()
            || !self.preempted.is_empty()
        {
            return Err(cpu_error(
                "cannot reconfigure CPU paged KV while pages or transactions exist",
            ));
        }
        self.capacity = capacity;
        self.storage = self
            .planes
            .iter()
            .zip(self.page_elements.iter())
            .map(|(plane, page_elements)| {
                let elements = page_elements
                    .checked_mul(capacity)
                    .ok_or_else(|| cpu_error("CPU KV plane capacity overflow"))?;
                Ok(match plane.element_type {
                    KvElementType::F32 => CpuKvPlaneStorage::F32(vec![0.0; elements]),
                    KvElementType::Bf16 => CpuKvPlaneStorage::Bf16(vec![0; elements]),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.free_slots = (0..capacity).rev().collect();
        self.shutdown = false;
        Ok(())
    }

    fn slot_range(&self, slot: usize, plane: usize) -> Result<Range<usize>> {
        if slot >= self.capacity {
            return Err(cpu_error("CPU KV physical slot is out of range"));
        }
        let elements = *self
            .page_elements
            .get(plane)
            .ok_or_else(|| cpu_error("CPU KV plane index is out of range"))?;
        let start = slot
            .checked_mul(elements)
            .ok_or_else(|| cpu_error("CPU KV physical slot offset overflow"))?;
        Ok(start..start + elements)
    }

    fn clear_slot(&mut self, slot: usize) -> Result<()> {
        for plane in 0..self.storage.len() {
            let range = self.slot_range(slot, plane)?;
            self.storage[plane].clear_range(range);
        }
        Ok(())
    }

    fn snapshot_slot(&self, slot: usize) -> Result<Vec<CpuKvPlaneStorage>> {
        (0..self.storage.len())
            .map(|plane| Ok(self.storage[plane].snapshot_range(self.slot_range(slot, plane)?)))
            .collect()
    }

    fn overwrite_slot(&mut self, slot: usize, planes: &[CpuKvPlaneStorage]) -> Result<()> {
        if planes.len() != self.storage.len() {
            return Err(cpu_error("CPU KV snapshot plane count mismatch"));
        }
        for (plane, source) in planes.iter().enumerate() {
            let range = self.slot_range(slot, plane)?;
            self.storage[plane].overwrite_range(range, source)?;
        }
        Ok(())
    }

    fn allocate_slot(&mut self) -> Result<usize> {
        let slot = self
            .free_slots
            .pop()
            .ok_or_else(|| cpu_error("physical CPU KV page pool is exhausted"))?;
        self.clear_slot(slot)?;
        Ok(slot)
    }

    fn ensure_pages_available(&self, pages: &[KvPageId], operation: &str) -> Result<()> {
        self.ensure_pages_available_except(pages, operation, None)
    }

    fn ensure_pages_available_except(
        &self,
        pages: &[KvPageId],
        operation: &str,
        except_lease: Option<u64>,
    ) -> Result<()> {
        for page in pages {
            for (transaction, active) in &self.active {
                if active.pages.contains_key(page) || active.protected_pages.contains(page) {
                    return Err(cpu_error(format!(
                        "cannot {operation} CPU KV page {} while transaction {} retains custody",
                        page.0,
                        transaction.get()
                    )));
                }
            }
            for (nonce, lease) in &self.committed {
                if Some(*nonce) != except_lease && lease.pinned.contains(page) {
                    return Err(cpu_error(format!(
                        "cannot {operation} CPU KV page {} while prepared lease {} retains custody",
                        page.0, nonce
                    )));
                }
            }
        }
        Ok(())
    }

    fn next_lease_nonce(&mut self) -> Result<u64> {
        let nonce = self.next_lease_nonce;
        self.next_lease_nonce = self
            .next_lease_nonce
            .checked_add(1)
            .ok_or_else(|| cpu_error("CPU KV prepared lease nonce exhausted"))?;
        Ok(nonce)
    }

    fn validate_active_commit(&self, id: ExecutionTransactionId) -> Result<()> {
        let active = self
            .active
            .get(&id)
            .ok_or_else(|| cpu_error("CPU KV transaction is not active"))?;
        if active.entered {
            return Err(cpu_error("cannot commit an entered CPU KV transaction"));
        }
        for (page, staged) in &active.pages {
            match staged {
                StagedPage::Physical { .. }
                    if self.resident.contains_key(page) || self.preempted.contains_key(page) =>
                {
                    return Err(cpu_error(format!(
                        "CPU KV page {} became occupied before commit",
                        page.0
                    )));
                }
                StagedPage::Shadow { resident_slot, .. }
                    if self.resident.get(page) != Some(resident_slot) =>
                {
                    return Err(cpu_error(format!(
                        "CPU KV shadow page {} changed physical identity",
                        page.0
                    )));
                }
                _ => {}
            }
            if let StagedPage::Shadow {
                resident_slot,
                planes,
            } = staged
            {
                if planes.len() != self.storage.len() {
                    return Err(cpu_error("CPU KV snapshot plane count mismatch"));
                }
                for (plane, snapshot) in planes.iter().enumerate() {
                    if snapshot.element_type() != self.storage[plane].element_type()
                        || snapshot.len() != self.slot_range(*resident_slot, plane)?.len()
                    {
                        return Err(cpu_error("CPU KV snapshot dtype or length mismatch"));
                    }
                }
            }
        }
        Ok(())
    }

    fn commit_active(&mut self, id: ExecutionTransactionId) -> Result<BTreeSet<KvPageId>> {
        self.validate_active_commit(id)?;
        let active = self
            .active
            .remove(&id)
            .expect("validated CPU KV transaction exists");
        let pinned = active.protected_pages;
        for (page, staged) in active.pages {
            match staged {
                StagedPage::Physical { slot } => {
                    self.resident.insert(page, slot);
                }
                StagedPage::Shadow {
                    resident_slot,
                    planes,
                } => self
                    .overwrite_slot(resident_slot, &planes)
                    .expect("CPU KV shadow shapes validated before installation"),
            }
        }
        Ok(pinned)
    }

    fn validate_transaction(&self, transaction: &CpuPagedKvTransaction) -> Result<()> {
        if !Rc::ptr_eq(&self.identity, &transaction.identity) {
            return Err(cpu_error("CPU KV transaction belongs to another pool"));
        }
        Ok(())
    }

    fn committed_lease(&self, transaction: &CpuPagedKvTransaction) -> Result<&CommittedLease> {
        self.validate_transaction(transaction)?;
        let nonce = transaction
            .prepared_nonce
            .ok_or_else(|| cpu_error("CPU KV transaction has no prepared lease"))?;
        let lease = self
            .committed
            .get(&nonce)
            .ok_or_else(|| cpu_error("CPU KV prepared lease is absent or finished"))?;
        if lease.transaction != transaction.id {
            return Err(cpu_error("CPU KV prepared lease transaction mismatch"));
        }
        Ok(lease)
    }

    fn validate_prepared_retirement(
        &self,
        transaction: &CpuPagedKvTransaction,
        pages: &[KvPageId],
    ) -> Result<Vec<KvPageId>> {
        let pages = unique_pages(pages)?;
        let lease = self.committed_lease(transaction)?;
        match &lease.phase {
            PreparedCommitPhase::Installed | PreparedCommitPhase::Aborted { .. } => {
                return Err(cpu_error("CPU KV retirement precedes prepared publication"));
            }
            PreparedCommitPhase::Retired { pages: retired, .. } => {
                if retired != &pages {
                    return Err(cpu_error("CPU KV retirement retry changed its page set"));
                }
                return Ok(pages);
            }
            PreparedCommitPhase::Published => {}
        }
        if pages.iter().any(|page| !lease.pinned.contains(page)) {
            return Err(cpu_error(
                "CPU KV retirement page is not pinned by its prepared lease",
            ));
        }
        self.ensure_pages_available_except(&pages, "retire", transaction.prepared_nonce)?;
        for page in &pages {
            if !self.resident.contains_key(page) && !self.preempted.contains_key(page) {
                return Err(cpu_error(format!(
                    "CPU KV retirement page {} does not exist",
                    page.0
                )));
            }
        }
        Ok(pages)
    }
}

/// A prepared installation retains this non-cloneable authority until finish.
/// Dropping it never unpins shared storage; unknown completion must fail closed.
#[derive(Debug)]
pub struct CpuPagedKvTransaction {
    id: ExecutionTransactionId,
    identity: Rc<()>,
    prepared_nonce: Option<u64>,
}

impl CpuPagedKvTransaction {
    pub const fn id(&self) -> ExecutionTransactionId {
        self.id
    }
}

/// Fixed-capacity typed physical CPU page pool.
#[derive(Clone)]
pub struct CpuPagedKvPool {
    inner: Rc<RefCell<PoolInner>>,
}

impl CpuPagedKvPool {
    pub fn new(
        planes: impl IntoIterator<Item = KvPlaneDescriptor>,
        page_size: usize,
        capacity: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: Rc::new(RefCell::new(PoolInner::new(
                planes.into_iter().collect(),
                page_size,
                capacity,
            )?)),
        })
    }

    pub fn planes(&self) -> Vec<KvPlaneDescriptor> {
        self.inner.borrow().planes.to_vec()
    }

    pub fn page_size(&self) -> usize {
        self.inner.borrow().page_size
    }

    pub fn physical_slot(&self, page: KvPageId) -> Option<usize> {
        self.inner.borrow().resident.get(&page).copied()
    }

    pub fn free_slot_count(&self) -> usize {
        self.inner.borrow().free_slots.len()
    }

    pub fn active_transaction_count(&self) -> usize {
        let inner = self.inner.borrow();
        inner.active.len() + inner.committed.len()
    }

    pub fn page_status(&self, page: KvPageId) -> KvPageStatus {
        let inner = self.inner.borrow();
        if inner.resident.contains_key(&page) {
            KvPageStatus::Resident
        } else if inner.preempted.contains_key(&page) {
            KvPageStatus::Preempted
        } else {
            KvPageStatus::Vacant
        }
    }

    pub fn configure_capacity(&mut self, capacity: usize) -> Result<()> {
        self.inner.borrow_mut().configure_capacity(capacity)
    }

    pub fn prepare(&mut self, request: PagedKvPrepare<'_>) -> Result<CpuPagedKvTransaction> {
        let mut inner = self.inner.borrow_mut();
        if inner.shutdown {
            return Err(cpu_error("CPU paged KV pool is shut down"));
        }
        if inner.active.contains_key(&request.transaction)
            || inner
                .committed
                .values()
                .any(|lease| lease.transaction == request.transaction)
        {
            return Err(cpu_error(format!(
                "CPU KV transaction {} is already active",
                request.transaction.get()
            )));
        }
        let mut kinds = BTreeMap::new();
        for &page in request.new_pages {
            insert_kind(&mut kinds, page, PageKind::New)?;
        }
        for &page in request.writable_pages {
            insert_kind(&mut kinds, page, PageKind::Shadow)?;
        }
        for cow in request.cow_replacements {
            if cow.source == cow.replacement {
                return Err(cpu_error("CPU KV COW source equals replacement"));
            }
            insert_kind(
                &mut kinds,
                cow.replacement,
                PageKind::Cow { source: cow.source },
            )?;
        }
        let writer_pages = kinds.keys().copied().collect::<BTreeSet<_>>();
        let protected_pages = request
            .protected_pages
            .iter()
            .copied()
            .chain(writer_pages.iter().copied())
            .chain(request.cow_replacements.iter().map(|cow| cow.source))
            .collect::<BTreeSet<_>>();
        for (active_id, active) in &inner.active {
            if let Some(page) = writer_pages.iter().find(|page| {
                active.pages.contains_key(page) || active.protected_pages.contains(page)
            }) {
                return Err(cpu_error(format!(
                    "CPU KV transaction {} cannot write page {} while transaction {} retains custody",
                    request.transaction.get(),
                    page.0,
                    active_id.get()
                )));
            }
            if let Some(page) = protected_pages
                .iter()
                .find(|page| active.pages.contains_key(page))
            {
                return Err(cpu_error(format!(
                    "CPU KV transaction {} cannot read page {} while transaction {} writes it",
                    request.transaction.get(),
                    page.0,
                    active_id.get()
                )));
            }
        }
        for (nonce, lease) in &inner.committed {
            if let Some(page) = protected_pages.intersection(&lease.pinned).next() {
                return Err(cpu_error(format!(
                    "CPU KV transaction {} cannot access page {} while prepared lease {} retains custody",
                    request.transaction.get(),
                    page.0,
                    nonce
                )));
            }
        }
        let physical_count = kinds
            .values()
            .filter(|kind| !matches!(kind, PageKind::Shadow))
            .count();
        if physical_count > inner.free_slots.len() {
            return Err(cpu_error(format!(
                "CPU KV transaction {} needs {physical_count} pages but only {} are free",
                request.transaction.get(),
                inner.free_slots.len()
            )));
        }
        for (page, kind) in &kinds {
            match kind {
                PageKind::New | PageKind::Cow { .. } => {
                    if inner.resident.contains_key(page) || inner.preempted.contains_key(page) {
                        return Err(cpu_error(format!("CPU KV page {} already exists", page.0)));
                    }
                }
                PageKind::Shadow => {
                    if !inner.resident.contains_key(page) {
                        return Err(cpu_error(format!(
                            "CPU KV writable page {} is not resident",
                            page.0
                        )));
                    }
                }
            }
            if let PageKind::Cow { source } = kind
                && !inner.resident.contains_key(source)
            {
                return Err(cpu_error(format!(
                    "CPU KV COW source page {} is not resident",
                    source.0
                )));
            }
        }
        let mut pages = BTreeMap::new();
        for (page, kind) in kinds {
            let staged = match kind {
                PageKind::New => StagedPage::Physical {
                    slot: inner.allocate_slot()?,
                },
                PageKind::Cow { source } => {
                    let slot = inner.allocate_slot()?;
                    let snapshot = inner.snapshot_slot(inner.resident[&source])?;
                    inner.overwrite_slot(slot, &snapshot)?;
                    StagedPage::Physical { slot }
                }
                PageKind::Shadow => {
                    let resident_slot = inner.resident[&page];
                    StagedPage::Shadow {
                        resident_slot,
                        planes: inner.snapshot_slot(resident_slot)?,
                    }
                }
            };
            pages.insert(page, staged);
        }
        inner.active.insert(
            request.transaction,
            ActiveTransaction {
                incarnation: Rc::new(()),
                entered: false,
                batch: None,
                pages,
                protected_pages,
            },
        );
        Ok(CpuPagedKvTransaction {
            id: request.transaction,
            identity: inner.identity.clone(),
            prepared_nonce: None,
        })
    }

    pub fn enter(
        &mut self,
        transaction: &mut CpuPagedKvTransaction,
        batch: PagedKvBatch,
    ) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        inner.validate_transaction(transaction)?;
        batch.validate(inner.page_size)?;
        let active = inner.active.get_mut(&transaction.id).ok_or_else(|| {
            cpu_error(format!(
                "CPU KV transaction {} is not active",
                transaction.id.get()
            ))
        })?;
        if active.entered {
            return Err(cpu_error("CPU KV transaction is already entered"));
        }
        if let Some(prepared) = &active.batch {
            if prepared != &batch {
                return Err(cpu_error(
                    "CPU KV transaction entered with a different batch",
                ));
            }
        } else {
            active.batch = Some(batch);
        }
        active.entered = true;
        Ok(())
    }

    pub fn active_view(&mut self, transaction: &mut CpuPagedKvTransaction) -> Result<CpuKvView> {
        let inner = self.inner.borrow();
        inner.validate_transaction(transaction)?;
        let active = inner.active.get(&transaction.id).ok_or_else(|| {
            cpu_error(format!(
                "CPU KV transaction {} is not active",
                transaction.id.get()
            ))
        })?;
        if !active.entered || active.batch.is_none() {
            return Err(cpu_error("CPU KV transaction has no active batch"));
        }
        Ok(CpuKvView {
            inner: Rc::clone(&self.inner),
            transaction: transaction.id,
            incarnation: active.incarnation.clone(),
        })
    }

    pub fn leave(&mut self, transaction: &mut CpuPagedKvTransaction) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        inner.validate_transaction(transaction)?;
        let active = inner
            .active
            .get_mut(&transaction.id)
            .ok_or_else(|| cpu_error("CPU KV transaction is not active"))?;
        if !active.entered {
            return Err(cpu_error("CPU KV transaction is not entered"));
        }
        active.entered = false;
        Ok(())
    }

    pub fn preflight_commit(&self, transaction: &CpuPagedKvTransaction) -> Result<()> {
        let inner = self.inner.borrow();
        inner.validate_transaction(transaction)?;
        inner.validate_active_commit(transaction.id)
    }

    pub fn commit(
        &mut self,
        transaction: &mut Option<CpuPagedKvTransaction>,
    ) -> Result<KvEndProgress> {
        let handle = transaction
            .as_ref()
            .ok_or_else(|| cpu_error("CPU KV commit transaction is absent"))?;
        let mut inner = self.inner.borrow_mut();
        inner.validate_transaction(handle)?;
        inner.commit_active(handle.id)?;
        transaction.take();
        Ok(KvEndProgress::Complete)
    }

    /// Installs once, atomically transferring active pins to a shared-pool lease.
    /// Unlike ordinary `commit`, the exact handle must survive through retirement.
    pub fn commit_prepared(
        &mut self,
        transaction: &mut CpuPagedKvTransaction,
    ) -> Result<KvEndProgress> {
        let mut inner = self.inner.borrow_mut();
        inner.validate_transaction(transaction)?;
        if transaction.prepared_nonce.is_some() {
            return Err(cpu_error("CPU KV prepared commit already installed"));
        }
        inner.validate_active_commit(transaction.id)?;
        // Exhaustion must reject before any mappings change or active pins leave.
        let nonce = inner.next_lease_nonce()?;
        let pinned = inner.commit_active(transaction.id)?;
        inner.committed.insert(
            nonce,
            CommittedLease {
                transaction: transaction.id,
                pinned,
                phase: PreparedCommitPhase::Installed,
            },
        );
        transaction.prepared_nonce = Some(nonce);
        Ok(KvEndProgress::Complete)
    }

    /// Authorize retirement without releasing any physical pins.
    pub fn publish_prepared(&mut self, transaction: &CpuPagedKvTransaction) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        let lease = inner.committed_lease(transaction)?;
        if !matches!(lease.phase, PreparedCommitPhase::Installed) {
            return Err(cpu_error("CPU KV prepared publication already applied"));
        }
        inner
            .committed
            .get_mut(
                &transaction
                    .prepared_nonce
                    .expect("validated prepared nonce"),
            )
            .expect("validated prepared lease")
            .phase = PreparedCommitPhase::Published;
        Ok(())
    }

    pub fn preflight_retirement(
        &self,
        transaction: &CpuPagedKvTransaction,
        pages: &[KvPageId],
    ) -> Result<()> {
        self.inner
            .borrow()
            .validate_prepared_retirement(transaction, pages)
            .map(|_| ())
    }

    /// Detach the exact retirement set, retaining slots in quarantine until finish.
    /// Retries acknowledge the recorded set and never release a slot twice.
    pub fn retire_prepared(
        &mut self,
        transaction: &mut CpuPagedKvTransaction,
        pages: &[KvPageId],
    ) -> Result<KvEndProgress> {
        let mut inner = self.inner.borrow_mut();
        let pages = inner.validate_prepared_retirement(transaction, pages)?;
        if matches!(
            inner.committed_lease(transaction)?.phase,
            PreparedCommitPhase::Retired { .. }
        ) {
            return Ok(KvEndProgress::Complete);
        }
        let mut slots = Vec::with_capacity(pages.len());
        for page in &pages {
            if let Some(slot) = inner.resident.remove(page) {
                slots.push(slot);
            }
            inner.preempted.remove(page);
        }
        inner
            .committed
            .get_mut(
                &transaction
                    .prepared_nonce
                    .expect("validated prepared nonce"),
            )
            .expect("validated prepared lease")
            .phase = PreparedCommitPhase::Retired { pages, slots };
        Ok(KvEndProgress::Complete)
    }

    /// Discard staged writes without releasing their shared-pool custody. Retries
    /// acknowledge the same cleanup; only finish makes its slots reusable.
    pub fn abort_prepared(
        &mut self,
        transaction: &mut CpuPagedKvTransaction,
    ) -> Result<KvEndProgress> {
        let mut inner = self.inner.borrow_mut();
        inner.validate_transaction(transaction)?;
        if transaction.prepared_nonce.is_some() {
            return if matches!(
                inner.committed_lease(transaction)?.phase,
                PreparedCommitPhase::Aborted { .. }
            ) {
                Ok(KvEndProgress::Complete)
            } else {
                Err(cpu_error(
                    "cannot abort an installed CPU KV prepared commit",
                ))
            };
        }
        let active = inner
            .active
            .get(&transaction.id)
            .ok_or_else(|| cpu_error("CPU KV transaction is not active"))?;
        if active.entered {
            return Err(cpu_error("cannot abort an entered CPU KV transaction"));
        }
        let nonce = inner.next_lease_nonce()?;
        let active = inner
            .active
            .remove(&transaction.id)
            .expect("validated active transaction");
        let slots = active
            .pages
            .into_values()
            .filter_map(|staged| staged.physical_slot())
            .collect();
        inner.committed.insert(
            nonce,
            CommittedLease {
                transaction: transaction.id,
                pinned: active.protected_pages,
                phase: PreparedCommitPhase::Aborted { slots },
            },
        );
        transaction.prepared_nonce = Some(nonce);
        Ok(KvEndProgress::Complete)
    }

    /// Consume the exact pool/transaction/nonce once, after cleanup or retirement
    /// is ACKed. Returns retired logical IDs; errors preserve caller authority.
    pub fn finish_prepared(
        &mut self,
        transaction: &mut Option<CpuPagedKvTransaction>,
    ) -> Result<Vec<KvPageId>> {
        let handle = transaction
            .as_ref()
            .ok_or_else(|| cpu_error("CPU KV prepared finish transaction is absent"))?;
        let mut inner = self.inner.borrow_mut();
        if !matches!(
            inner.committed_lease(handle)?.phase,
            PreparedCommitPhase::Retired { .. } | PreparedCommitPhase::Aborted { .. }
        ) {
            return Err(cpu_error(
                "CPU KV prepared finish precedes cleanup/retirement ACK",
            ));
        }
        let lease = inner
            .committed
            .remove(&handle.prepared_nonce.expect("validated prepared nonce"))
            .expect("validated prepared lease");
        let (pages, slots) = match lease.phase {
            PreparedCommitPhase::Retired { pages, slots } => (pages, slots),
            PreparedCommitPhase::Aborted { slots } => (Vec::new(), slots),
            _ => unreachable!("validated terminal lease"),
        };
        inner.free_slots.extend(slots);
        transaction.take();
        Ok(pages)
    }

    pub fn abort(
        &mut self,
        transaction: &mut Option<CpuPagedKvTransaction>,
    ) -> Result<KvEndProgress> {
        let handle = transaction
            .as_ref()
            .ok_or_else(|| cpu_error("CPU KV abort transaction is absent"))?;
        let id = handle.id;
        let mut inner = self.inner.borrow_mut();
        inner.validate_transaction(handle)?;
        let active = inner
            .active
            .get(&id)
            .ok_or_else(|| cpu_error("CPU KV transaction is not active"))?;
        if active.entered {
            return Err(cpu_error("cannot abort an entered CPU KV transaction"));
        }
        let active = inner
            .active
            .remove(&id)
            .expect("validated CPU KV transaction exists");
        for staged in active.pages.into_values() {
            if let StagedPage::Physical { slot } = staged {
                inner.free_slots.push(slot);
            }
        }
        drop(inner);
        transaction.take();
        Ok(KvEndProgress::Complete)
    }

    pub fn release(&mut self, pages: &[KvPageId]) -> Result<()> {
        let pages = unique_pages(pages)?;
        let mut inner = self.inner.borrow_mut();
        inner.ensure_pages_available(&pages, "release")?;
        for page in &pages {
            if !inner.resident.contains_key(page) && !inner.preempted.contains_key(page) {
                return Err(cpu_error(format!("CPU KV page {} does not exist", page.0)));
            }
        }
        for page in pages {
            if let Some(slot) = inner.resident.remove(&page) {
                inner.free_slots.push(slot);
            }
            inner.preempted.remove(&page);
        }
        Ok(())
    }

    pub fn preempt(&mut self, pages: &[KvPageId]) -> Result<()> {
        let pages = unique_pages(pages)?;
        let mut inner = self.inner.borrow_mut();
        inner.ensure_pages_available(&pages, "preempt")?;
        let snapshots = pages
            .iter()
            .map(|page| {
                let slot =
                    inner.resident.get(page).copied().ok_or_else(|| {
                        cpu_error(format!("CPU KV page {} is not resident", page.0))
                    })?;
                Ok((*page, slot, inner.snapshot_slot(slot)?))
            })
            .collect::<Result<Vec<_>>>()?;
        for (page, slot, snapshot) in snapshots {
            inner.resident.remove(&page);
            inner.preempted.insert(page, snapshot);
            inner.free_slots.push(slot);
        }
        Ok(())
    }

    pub fn restore(&mut self, pages: &[KvPageId]) -> Result<()> {
        let pages = unique_pages(pages)?;
        let mut inner = self.inner.borrow_mut();
        inner.ensure_pages_available(&pages, "restore")?;
        if pages.len() > inner.free_slots.len() {
            return Err(cpu_error("CPU KV restore exceeds physical capacity"));
        }
        for page in &pages {
            if !inner.preempted.contains_key(page) {
                return Err(cpu_error(format!(
                    "CPU KV page {} is not preempted",
                    page.0
                )));
            }
        }
        for page in pages {
            let snapshot = inner
                .preempted
                .remove(&page)
                .expect("validated CPU KV snapshot exists");
            let slot = inner
                .free_slots
                .pop()
                .expect("restore capacity was validated");
            inner.overwrite_slot(slot, &snapshot)?;
            inner.resident.insert(page, slot);
        }
        Ok(())
    }

    /// Prepared leases count as active; quarantined slots are not free until finish.
    pub fn capacity(&self) -> KvCapacity {
        let inner = self.inner.borrow();
        KvCapacity {
            physical_pages: inner.capacity,
            resident_pages: inner.resident.len(),
            preempted_pages: inner.preempted.len(),
            active_transactions: inner.active.len() + inner.committed.len(),
            free_pages: inner.free_slots.len(),
        }
    }

    pub fn shutdown(&mut self) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        if !inner.active.is_empty() || !inner.committed.is_empty() {
            return Err(cpu_error(
                "cannot shut down CPU paged KV with active or prepared transactions",
            ));
        }
        inner.resident.clear();
        inner.preempted.clear();
        for storage in &mut inner.storage {
            match storage {
                CpuKvPlaneStorage::F32(values) => values.fill(0.0),
                CpuKvPlaneStorage::Bf16(values) => values.fill(0),
            }
        }
        inner.free_slots = (0..inner.capacity).rev().collect();
        inner.shutdown = true;
        Ok(())
    }
}

#[derive(Clone)]
pub struct CpuKvView {
    inner: Rc<RefCell<PoolInner>>,
    transaction: ExecutionTransactionId,
    // An old view must not revive when its numeric transaction ID is reused.
    incarnation: Rc<()>,
}

impl CpuKvView {
    pub const fn transaction(&self) -> ExecutionTransactionId {
        self.transaction
    }

    pub fn validate_batch(&self, batch: &PagedKvBatch) -> Result<()> {
        let inner = self.inner.borrow();
        let active = active_transaction(&inner, self.transaction, &self.incarnation)?;
        if active.batch.as_ref() != Some(batch) {
            return Err(cpu_error("CPU KV view is bound to a different batch"));
        }
        Ok(())
    }

    pub fn physical_slot(&self, page: KvPageId) -> Option<usize> {
        let inner = self.inner.borrow();
        let active = active_transaction(&inner, self.transaction, &self.incarnation).ok()?;
        validate_page_read(active, page).ok()?;
        active
            .pages
            .get(&page)
            .and_then(StagedPage::physical_slot)
            .or_else(|| inner.resident.get(&page).copied())
    }

    pub fn with_f32_page<R>(
        &self,
        page: KvPageId,
        plane: usize,
        read: impl FnOnce(&[f32]) -> R,
    ) -> Result<R> {
        let inner = self.inner.borrow();
        match page_storage(&inner, self.transaction, &self.incarnation, page, plane)? {
            PageStorageRef::F32(values) => Ok(read(values)),
            PageStorageRef::Bf16(_) => Err(cpu_error("requested CPU KV plane is not F32")),
        }
    }

    pub fn with_bf16_page<R>(
        &self,
        page: KvPageId,
        plane: usize,
        read: impl FnOnce(&[u16]) -> R,
    ) -> Result<R> {
        let inner = self.inner.borrow();
        match page_storage(&inner, self.transaction, &self.incarnation, page, plane)? {
            PageStorageRef::F32(_) => Err(cpu_error("requested CPU KV plane is not BF16")),
            PageStorageRef::Bf16(values) => Ok(read(values)),
        }
    }

    pub fn with_f32_page_mut<R>(
        &mut self,
        page: KvPageId,
        plane: usize,
        write: impl FnOnce(&mut [f32]) -> R,
    ) -> Result<R> {
        let mut inner = self.inner.borrow_mut();
        match writable_page_storage(&mut inner, self.transaction, &self.incarnation, page, plane)? {
            PageStorageMut::F32(values) => Ok(write(values)),
            PageStorageMut::Bf16(_) => Err(cpu_error("requested CPU KV plane is not F32")),
        }
    }

    pub fn with_bf16_page_mut<R>(
        &mut self,
        page: KvPageId,
        plane: usize,
        write: impl FnOnce(&mut [u16]) -> R,
    ) -> Result<R> {
        let mut inner = self.inner.borrow_mut();
        match writable_page_storage(&mut inner, self.transaction, &self.incarnation, page, plane)? {
            PageStorageMut::F32(_) => Err(cpu_error("requested CPU KV plane is not BF16")),
            PageStorageMut::Bf16(values) => Ok(write(values)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append(
        &mut self,
        layer: usize,
        row_sequence_ids: &[usize],
        row_positions: &[usize],
        kv_heads: usize,
        head_dim: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        let batch = active_transaction(&inner, self.transaction, &self.incarnation)?
            .batch
            .as_ref()
            .expect("active CPU KV transaction has a batch")
            .clone();
        if row_sequence_ids != batch.row_sequence_ids.as_ref()
            || row_positions != batch.row_positions.as_ref()
        {
            return Err(cpu_error(
                "CPU KV append metadata differs from active batch",
            ));
        }
        let width = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| cpu_error("CPU KV append width overflow"))?;
        validate_standard_gqa_planes(&inner, layer, width)?;
        let expected = row_positions
            .len()
            .checked_mul(width)
            .ok_or_else(|| cpu_error("CPU KV append size overflow"))?;
        if key.len() != expected || value.len() != expected {
            return Err(cpu_error("CPU KV append value length mismatch"));
        }
        for row in 0..row_positions.len() {
            let sequence = &batch.sequences[row_sequence_ids[row]];
            let position = row_positions[row];
            let page = sequence.block_table[position / inner.page_size];
            let token_in_page = position % inner.page_size;
            let source = row * width..(row + 1) * width;
            for (plane, values) in [(0, key), (1, value)] {
                let destination = token_plane_range(&inner, plane, layer, token_in_page, width)?;
                write_values(
                    writable_page_storage(
                        &mut inner,
                        self.transaction,
                        &self.incarnation,
                        page,
                        plane,
                    )?,
                    destination,
                    &values[source.clone()],
                )?;
            }
        }
        Ok(())
    }
}

impl PagedKvHistory for CpuKvView {
    fn history(
        &self,
        layer: usize,
        sequence: usize,
        through_position: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<KvHistory> {
        let inner = self.inner.borrow();
        let active = active_transaction(&inner, self.transaction, &self.incarnation)?;
        let batch = active
            .batch
            .as_ref()
            .expect("active CPU KV transaction has a batch");
        let sequence = batch
            .sequences
            .get(sequence)
            .ok_or_else(|| cpu_error("CPU KV history sequence is out of range"))?;
        if through_position >= sequence.sequence_len {
            return Err(cpu_error("CPU KV history position exceeds sequence length"));
        }
        let width = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| cpu_error("CPU KV history width overflow"))?;
        validate_standard_gqa_planes(&inner, layer, width)?;
        let tokens = through_position + 1;
        let capacity = tokens
            .checked_mul(width)
            .ok_or_else(|| cpu_error("CPU KV history size overflow"))?;
        let mut key = Vec::with_capacity(capacity);
        let mut value = Vec::with_capacity(capacity);
        for position in 0..tokens {
            let page = *sequence
                .block_table
                .get(position / inner.page_size)
                .ok_or_else(|| cpu_error("CPU KV history logical page is absent"))?;
            let token_in_page = position % inner.page_size;
            for (plane, destination) in [(0, &mut key), (1, &mut value)] {
                let range = token_plane_range(&inner, plane, layer, token_in_page, width)?;
                read_values(
                    page_storage(&inner, self.transaction, &self.incarnation, page, plane)?,
                    range,
                    destination,
                );
            }
        }
        Ok(KvHistory { tokens, key, value })
    }
}

fn active_transaction<'a>(
    inner: &'a PoolInner,
    transaction: ExecutionTransactionId,
    incarnation: &Rc<()>,
) -> Result<&'a ActiveTransaction> {
    let active = inner
        .active
        .get(&transaction)
        .ok_or_else(|| cpu_error("CPU KV view transaction is no longer active"))?;
    if !Rc::ptr_eq(&active.incarnation, incarnation) {
        return Err(cpu_error("CPU KV view has a stale transaction incarnation"));
    }
    if !active.entered {
        return Err(cpu_error("CPU KV view transaction is not entered"));
    }
    Ok(active)
}

fn validate_standard_gqa_planes(inner: &PoolInner, layer: usize, width: usize) -> Result<()> {
    if inner.planes.len() != 2 {
        return Err(cpu_error("standard GQA requires exactly two KV planes"));
    }
    for (index, plane) in inner.planes.iter().enumerate() {
        if plane.elements_per_token != width || layer >= plane.layer_count {
            return Err(cpu_error(format!(
                "CPU KV plane {index} cannot serve layer {layer} width {width}"
            )));
        }
    }
    Ok(())
}

fn token_plane_range(
    inner: &PoolInner,
    plane: usize,
    layer: usize,
    token_in_page: usize,
    width: usize,
) -> Result<Range<usize>> {
    let descriptor = inner
        .planes
        .get(plane)
        .ok_or_else(|| cpu_error("CPU KV plane index is out of range"))?;
    if token_in_page >= inner.page_size
        || layer >= descriptor.layer_count
        || descriptor.elements_per_token != width
    {
        return Err(cpu_error("CPU KV token range does not match typed plane"));
    }
    let token_index = layer
        .checked_mul(inner.page_size)
        .and_then(|base| base.checked_add(token_in_page))
        .ok_or_else(|| cpu_error("CPU KV token offset overflow"))?;
    let start = token_index
        .checked_mul(width)
        .ok_or_else(|| cpu_error("CPU KV element offset overflow"))?;
    Ok(start..start + width)
}

enum PageStorageRef<'a> {
    F32(&'a [f32]),
    Bf16(&'a [u16]),
}

enum PageStorageMut<'a> {
    F32(&'a mut [f32]),
    Bf16(&'a mut [u16]),
}

fn validate_page_read(active: &ActiveTransaction, page: KvPageId) -> Result<()> {
    if !active.pages.contains_key(&page) && !active.protected_pages.contains(&page) {
        return Err(cpu_error(format!(
            "CPU KV page {} is outside the transaction read set",
            page.0
        )));
    }
    Ok(())
}

fn page_storage<'a>(
    inner: &'a PoolInner,
    transaction: ExecutionTransactionId,
    incarnation: &Rc<()>,
    page: KvPageId,
    plane: usize,
) -> Result<PageStorageRef<'a>> {
    let active = active_transaction(inner, transaction, incarnation)?;
    validate_page_read(active, page)?;
    if let Some(StagedPage::Shadow { planes, .. }) = active.pages.get(&page) {
        return snapshot_ref(planes, plane);
    }
    let slot = active
        .pages
        .get(&page)
        .and_then(StagedPage::physical_slot)
        .or_else(|| inner.resident.get(&page).copied())
        .ok_or_else(|| cpu_error(format!("CPU KV page {} is not readable", page.0)))?;
    let range = inner.slot_range(slot, plane)?;
    let storage = inner
        .storage
        .get(plane)
        .ok_or_else(|| cpu_error("CPU KV plane index is out of range"))?;
    storage_ref(storage, range)
}

fn writable_page_storage<'a>(
    inner: &'a mut PoolInner,
    transaction: ExecutionTransactionId,
    incarnation: &Rc<()>,
    page: KvPageId,
    plane: usize,
) -> Result<PageStorageMut<'a>> {
    let (slot, shadow) = {
        let active = active_transaction(inner, transaction, incarnation)?;
        match active.pages.get(&page) {
            Some(StagedPage::Physical { slot }) => (Some(*slot), false),
            Some(StagedPage::Shadow { .. }) => (None, true),
            None => {
                return Err(cpu_error(format!(
                    "CPU KV page {} is not writable in transaction {}",
                    page.0,
                    transaction.get()
                )));
            }
        }
    };
    if shadow {
        let active = inner
            .active
            .get_mut(&transaction)
            .expect("validated active CPU KV transaction");
        let StagedPage::Shadow { planes, .. } = active
            .pages
            .get_mut(&page)
            .expect("validated CPU KV shadow page")
        else {
            unreachable!("validated CPU KV shadow page changed kind")
        };
        return snapshot_mut(planes, plane);
    }
    let range = inner.slot_range(slot.expect("physical page has a slot"), plane)?;
    storage_mut(&mut inner.storage[plane], range)
}

fn snapshot_ref(planes: &[CpuKvPlaneStorage], plane: usize) -> Result<PageStorageRef<'_>> {
    match planes
        .get(plane)
        .ok_or_else(|| cpu_error("CPU KV plane index is out of range"))?
    {
        CpuKvPlaneStorage::F32(values) => Ok(PageStorageRef::F32(values)),
        CpuKvPlaneStorage::Bf16(values) => Ok(PageStorageRef::Bf16(values)),
    }
}

fn snapshot_mut(planes: &mut [CpuKvPlaneStorage], plane: usize) -> Result<PageStorageMut<'_>> {
    match planes
        .get_mut(plane)
        .ok_or_else(|| cpu_error("CPU KV plane index is out of range"))?
    {
        CpuKvPlaneStorage::F32(values) => Ok(PageStorageMut::F32(values)),
        CpuKvPlaneStorage::Bf16(values) => Ok(PageStorageMut::Bf16(values)),
    }
}

fn storage_ref(storage: &CpuKvPlaneStorage, range: Range<usize>) -> Result<PageStorageRef<'_>> {
    match storage {
        CpuKvPlaneStorage::F32(values) => Ok(PageStorageRef::F32(&values[range])),
        CpuKvPlaneStorage::Bf16(values) => Ok(PageStorageRef::Bf16(&values[range])),
    }
}

fn storage_mut(storage: &mut CpuKvPlaneStorage, range: Range<usize>) -> Result<PageStorageMut<'_>> {
    match storage {
        CpuKvPlaneStorage::F32(values) => Ok(PageStorageMut::F32(&mut values[range])),
        CpuKvPlaneStorage::Bf16(values) => Ok(PageStorageMut::Bf16(&mut values[range])),
    }
}

fn write_values(storage: PageStorageMut<'_>, range: Range<usize>, values: &[f32]) -> Result<()> {
    if range.len() != values.len() {
        return Err(cpu_error("CPU KV write range length mismatch"));
    }
    match storage {
        PageStorageMut::F32(destination) => destination[range].copy_from_slice(values),
        PageStorageMut::Bf16(destination) => destination[range]
            .iter_mut()
            .zip(values)
            .for_each(|(destination, &value)| *destination = bf16_rne_word(value)),
    }
    Ok(())
}

fn read_values(storage: PageStorageRef<'_>, range: Range<usize>, destination: &mut Vec<f32>) {
    match storage {
        PageStorageRef::F32(values) => destination.extend_from_slice(&values[range]),
        PageStorageRef::Bf16(values) => {
            destination.extend(values[range].iter().copied().map(bf16_word_value));
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PageKind {
    New,
    Shadow,
    Cow { source: KvPageId },
}

fn insert_kind(
    kinds: &mut BTreeMap<KvPageId, PageKind>,
    page: KvPageId,
    kind: PageKind,
) -> Result<()> {
    if kinds.insert(page, kind).is_some() {
        return Err(cpu_error(format!(
            "CPU KV page {} has more than one mutation role",
            page.0
        )));
    }
    Ok(())
}

fn unique_pages(pages: &[KvPageId]) -> Result<Vec<KvPageId>> {
    let unique = pages.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != pages.len() {
        return Err(cpu_error("CPU KV page operation contains duplicates"));
    }
    Ok(unique.into_iter().collect())
}

#[cfg(test)]
mod prepared_commit_tests {
    use super::*;

    fn tx(id: u64) -> ExecutionTransactionId {
        ExecutionTransactionId::new(id).unwrap()
    }

    fn pool(capacity: usize) -> CpuPagedKvPool {
        CpuPagedKvPool::new(
            [
                KvPlaneDescriptor::new("key", 1, 1, KvElementType::F32),
                KvPlaneDescriptor::new("value", 1, 1, KvElementType::Bf16),
            ],
            4,
            capacity,
        )
        .unwrap()
    }

    fn prepare<'a>(id: u64, new_pages: &'a [KvPageId]) -> PagedKvPrepare<'a> {
        PagedKvPrepare {
            transaction: tx(id),
            new_pages,
            writable_pages: &[],
            cow_replacements: &[],
            protected_pages: &[],
        }
    }

    fn batch(page: KvPageId) -> PagedKvBatch {
        PagedKvBatch {
            row_sequence_ids: vec![0].into(),
            row_positions: vec![0].into(),
            sequences: vec![PagedKvSequence {
                sequence_len: 1,
                block_table: vec![page].into(),
            }]
            .into(),
        }
    }

    fn staged(pool: &mut CpuPagedKvPool, id: u64, page: KvPageId) -> CpuPagedKvTransaction {
        let mut handle = pool.prepare(prepare(id, &[page])).unwrap();
        pool.enter(&mut handle, batch(page)).unwrap();
        let mut view = pool.active_view(&mut handle).unwrap();
        view.with_f32_page_mut(page, 0, |v| v[0] = 3.25).unwrap();
        view.with_bf16_page_mut(page, 1, |v| v[0] = 0x4120).unwrap();
        pool.leave(&mut handle).unwrap();
        assert!(view.with_f32_page_mut(page, 0, |v| v.fill(99.0)).is_err());
        handle
    }

    fn snapshot(pool: &CpuPagedKvPool, page: KvPageId) -> Vec<CpuKvPlaneStorage> {
        let inner = pool.inner.borrow();
        inner.snapshot_slot(inner.resident[&page]).unwrap()
    }

    #[test]
    fn view_rejects_undeclared_resident_reads_and_mapping_queries() {
        let mut pool = pool(3);
        let resident = KvPageId(10);
        let staged_page = KvPageId(11);
        let mut seed = Some(staged(&mut pool, 1, resident));
        pool.commit(&mut seed).unwrap();
        let original = snapshot(&pool, resident);
        let mut writer = Some(pool.prepare(prepare(2, &[staged_page])).unwrap());
        pool.enter(writer.as_mut().unwrap(), batch(staged_page))
            .unwrap();
        let mut view = pool.active_view(writer.as_mut().unwrap()).unwrap();
        assert_eq!(
            view.with_f32_page(staged_page, 0, |v| v.to_vec()).unwrap(),
            vec![0.0; 4]
        );
        assert!(view.physical_slot(staged_page).is_some());
        let error = view
            .with_f32_page(resident, 0, |_| {
                panic!("undeclared F32 read escaped custody")
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("outside the transaction read set")
        );
        assert!(
            view.with_bf16_page(resident, 1, |_| panic!(
                "undeclared BF16 read escaped custody"
            ))
            .is_err()
        );
        assert!(
            view.with_f32_page_mut(resident, 0, |_| panic!("undeclared write escaped custody"))
                .is_err()
        );
        assert_eq!(view.physical_slot(resident), None);
        assert!(view.with_f32_page(staged_page, usize::MAX, |_| ()).is_err());
        assert!(
            view.with_f32_page_mut(staged_page, usize::MAX, |_| ())
                .is_err()
        );

        // A batch's block table is metadata, not authority to add a read set.
        let mut reader = Some(pool.prepare(prepare(3, &[])).unwrap());
        pool.enter(reader.as_mut().unwrap(), batch(resident))
            .unwrap();
        let undeclared = pool.active_view(reader.as_mut().unwrap()).unwrap();
        assert!(undeclared.history(0, 0, 0, 1, 1).is_err());
        assert_eq!(snapshot(&pool, resident), original);
        pool.leave(reader.as_mut().unwrap()).unwrap();
        pool.abort(&mut reader).unwrap();
        pool.leave(writer.as_mut().unwrap()).unwrap();
        pool.abort(&mut writer).unwrap();
        pool.release(&[resident]).unwrap();
    }

    #[test]
    fn view_protected_history_and_new_pages_remain_readable() {
        let mut pool = pool(2);
        let resident = KvPageId(10);
        let new_page = KvPageId(11);
        let mut seed = Some(staged(&mut pool, 1, resident));
        pool.commit(&mut seed).unwrap();
        let mut handle = Some(
            pool.prepare(PagedKvPrepare {
                protected_pages: &[resident],
                ..prepare(2, &[new_page])
            })
            .unwrap(),
        );
        pool.enter(
            handle.as_mut().unwrap(),
            PagedKvBatch {
                row_sequence_ids: vec![0].into(),
                row_positions: vec![4].into(),
                sequences: vec![PagedKvSequence {
                    sequence_len: 5,
                    block_table: vec![resident, new_page].into(),
                }]
                .into(),
            },
        )
        .unwrap();
        let mut view = pool.active_view(handle.as_mut().unwrap()).unwrap();
        assert_eq!(view.with_f32_page(resident, 0, |v| v[0]).unwrap(), 3.25);
        assert_eq!(view.with_bf16_page(resident, 1, |v| v[0]).unwrap(), 0x4120);
        assert!(view.physical_slot(resident).is_some());
        assert!(view.with_f32_page(resident, usize::MAX, |_| ()).is_err());
        view.append(0, &[0], &[4], 1, 1, &[4.5], &[20.0]).unwrap();
        let history = view.history(0, 0, 4, 1, 1).unwrap();
        assert_eq!(history.tokens, 5);
        assert_eq!(history.key, vec![3.25, 0.0, 0.0, 0.0, 4.5]);
        assert_eq!(history.value, vec![10.0, 0.0, 0.0, 0.0, 20.0]);
        pool.leave(handle.as_mut().unwrap()).unwrap();
        assert!(view.history(0, 0, 4, 1, 1).is_err());
        assert_eq!(view.physical_slot(resident), None);
        pool.abort(&mut handle).unwrap();
        pool.release(&[resident]).unwrap();
    }

    #[test]
    fn cow_and_shadow_read_sets_do_not_revive_stale_views_on_id_reuse() {
        for commit in [false, true] {
            let mut pool = pool(4);
            let source = KvPageId(10);
            let shadow = KvPageId(11);
            let replacement = KvPageId(12);
            let unrelated = KvPageId(13);
            for (id, page) in [(1, source), (2, shadow), (3, unrelated)] {
                let mut seed = Some(staged(&mut pool, id, page));
                pool.commit(&mut seed).unwrap();
            }
            let mut handle = Some(
                pool.prepare(PagedKvPrepare {
                    writable_pages: &[shadow],
                    cow_replacements: &[KvCowReplacement {
                        source,
                        replacement,
                        logical_page: 0,
                    }],
                    ..prepare(4, &[])
                })
                .unwrap(),
            );
            pool.enter(handle.as_mut().unwrap(), batch(replacement))
                .unwrap();
            let mut old_view = pool.active_view(handle.as_mut().unwrap()).unwrap();
            // COW sources and writable shadows are included by the reservation,
            // even when they are not explicitly repeated in protected_pages.
            assert_eq!(old_view.with_f32_page(source, 0, |v| v[0]).unwrap(), 3.25);
            assert_eq!(
                old_view.with_bf16_page(source, 1, |v| v[0]).unwrap(),
                0x4120
            );
            assert_eq!(
                old_view.with_f32_page(replacement, 0, |v| v[0]).unwrap(),
                3.25
            );
            old_view
                .with_f32_page_mut(shadow, 0, |v| v[0] = 7.0)
                .unwrap();
            assert_eq!(old_view.with_f32_page(shadow, 0, |v| v[0]).unwrap(), 7.0);
            assert!(old_view.with_f32_page(unrelated, 0, |_| ()).is_err());
            assert_eq!(old_view.physical_slot(unrelated), None);
            pool.leave(handle.as_mut().unwrap()).unwrap();
            assert!(pool.clone().release(&[source]).is_err());
            assert!(old_view.with_f32_page(source, 0, |_| ()).is_err());
            if commit {
                pool.commit(&mut handle).unwrap();
            } else {
                pool.abort(&mut handle).unwrap();
            }

            handle = Some(
                pool.prepare(PagedKvPrepare {
                    writable_pages: &[source],
                    ..prepare(4, &[])
                })
                .unwrap(),
            );
            pool.enter(handle.as_mut().unwrap(), batch(source)).unwrap();
            let new_view = pool.active_view(handle.as_mut().unwrap()).unwrap();
            assert_eq!(new_view.with_f32_page(source, 0, |v| v[0]).unwrap(), 3.25);
            let error = old_view
                .with_f32_page(source, 0, |_| panic!("stale view revived"))
                .unwrap_err();
            assert!(error.to_string().contains("stale transaction incarnation"));
            assert!(
                old_view
                    .with_f32_page_mut(source, 0, |_| panic!("stale writer revived"))
                    .is_err()
            );
            assert!(old_view.with_bf16_page(source, 1, |_| ()).is_err());
            assert!(old_view.validate_batch(&batch(source)).is_err());
            assert!(old_view.history(0, 0, 0, 1, 1).is_err());
            assert_eq!(old_view.physical_slot(source), None);
            assert_eq!(new_view.with_f32_page(source, 0, |v| v[0]).unwrap(), 3.25);
            pool.leave(handle.as_mut().unwrap()).unwrap();
            pool.abort(&mut handle).unwrap();
            let mut pages = vec![source, shadow, unrelated];
            if commit {
                pages.push(replacement);
            }
            pool.release(&pages).unwrap();
            assert_eq!(pool.free_slot_count(), 4);
        }
    }

    #[test]
    fn prepared_commit_pins_every_clone_until_exact_finish() {
        let mut pool = pool(3);
        let mut alias = pool.clone();
        let page = KvPageId(10);
        let mut handle = Some(staged(&mut pool, 1, page));
        pool.commit_prepared(handle.as_mut().unwrap()).unwrap();
        let slot = pool.physical_slot(page);
        let bytes = snapshot(&pool, page);
        let capacity = pool.capacity();

        assert!(alias.release(&[page]).is_err());
        assert!(alias.preempt(&[page]).is_err());
        assert!(alias.shutdown().is_err());
        assert!(alias.configure_capacity(3).is_err());
        assert!(
            alias
                .prepare(PagedKvPrepare {
                    writable_pages: &[page],
                    ..prepare(2, &[])
                })
                .is_err()
        );
        assert!(
            alias
                .prepare(PagedKvPrepare {
                    protected_pages: &[page],
                    ..prepare(2, &[])
                })
                .is_err()
        );
        assert!(
            alias
                .prepare(PagedKvPrepare {
                    cow_replacements: &[KvCowReplacement {
                        source: page,
                        replacement: KvPageId(11),
                        logical_page: 0,
                    }],
                    ..prepare(2, &[])
                })
                .is_err()
        );
        assert!(alias.prepare(prepare(1, &[KvPageId(20)])).is_err());
        assert!(alias.enter(handle.as_mut().unwrap(), batch(page)).is_err());
        assert!(alias.active_view(handle.as_mut().unwrap()).is_err());
        assert!(alias.commit(&mut handle).is_err());
        assert!(alias.abort(&mut handle).is_err());
        assert!(alias.commit_prepared(handle.as_mut().unwrap()).is_err());
        assert!(
            alias
                .retire_prepared(handle.as_mut().unwrap(), &[])
                .is_err()
        );
        assert!(alias.finish_prepared(&mut handle).is_err());
        assert!(handle.is_some());
        assert_eq!(alias.capacity(), capacity);
        assert_eq!(alias.physical_slot(page), slot);
        assert_eq!(snapshot(&alias, page), bytes);

        // Independent pages remain usable while this lease is pinned.
        let mut other = Some(alias.prepare(prepare(2, &[KvPageId(20)])).unwrap());
        alias.commit(&mut other).unwrap();
        pool.publish_prepared(handle.as_ref().unwrap()).unwrap();
        assert!(
            pool.preflight_retirement(handle.as_ref().unwrap(), &[KvPageId(20)])
                .is_err()
        );
        assert!(
            pool.retire_prepared(handle.as_mut().unwrap(), &[KvPageId(20)])
                .is_err()
        );
        assert_eq!(alias.page_status(KvPageId(20)), KvPageStatus::Resident);
        alias.release(&[KvPageId(20)]).unwrap();
        assert!(alias.publish_prepared(handle.as_ref().unwrap()).is_err());
        assert!(alias.finish_prepared(&mut handle).is_err());
        pool.retire_prepared(handle.as_mut().unwrap(), &[]).unwrap();
        assert!(alias.release(&[page]).is_err());
        assert_eq!(pool.finish_prepared(&mut handle).unwrap(), vec![]);
        assert!(handle.is_none());
        assert!(alias.finish_prepared(&mut handle).is_err());
        assert_eq!(snapshot(&alias, page), bytes);
        alias.preempt(&[page]).unwrap();
        alias.restore(&[page]).unwrap();
        assert_eq!(snapshot(&alias, page), bytes);
        alias.release(&[page]).unwrap();
        assert_eq!(alias.capacity().free_pages, 3);
        assert_eq!(alias.active_transaction_count(), 0);
        alias.shutdown().unwrap();
    }

    #[test]
    fn retirement_retries_keep_ids_and_slots_quarantined_until_finish() {
        let mut pool = pool(1);
        let mut alias = pool.clone();
        let page = KvPageId(10);
        let mut handle = Some(staged(&mut pool, 1, page));
        pool.commit_prepared(handle.as_mut().unwrap()).unwrap();
        pool.publish_prepared(handle.as_ref().unwrap()).unwrap();
        assert!(
            pool.retire_prepared(handle.as_mut().unwrap(), &[page, page])
                .is_err()
        );
        assert!(
            pool.retire_prepared(handle.as_mut().unwrap(), &[KvPageId(20)])
                .is_err()
        );
        pool.preflight_retirement(handle.as_ref().unwrap(), &[page])
            .unwrap();
        for _ in 0..3 {
            assert_eq!(
                pool.retire_prepared(handle.as_mut().unwrap(), &[page])
                    .unwrap(),
                KvEndProgress::Complete
            );
            assert_eq!(alias.physical_slot(page), None);
            assert_eq!(alias.free_slot_count(), 0);
            assert_eq!(alias.active_transaction_count(), 1);
        }
        assert!(pool.retire_prepared(handle.as_mut().unwrap(), &[]).is_err());
        assert!(alias.release(&[page]).is_err());
        assert!(alias.prepare(prepare(2, &[page])).is_err());
        assert!(alias.prepare(prepare(2, &[KvPageId(20)])).is_err());
        assert!(alias.shutdown().is_err());
        assert_eq!(pool.finish_prepared(&mut handle).unwrap(), vec![page]);
        assert!(pool.finish_prepared(&mut handle).is_err());
        assert_eq!(alias.free_slot_count(), 1);
        let mut reused = Some(alias.prepare(prepare(1, &[page])).unwrap());
        alias.abort(&mut reused).unwrap();
        assert_eq!(alias.free_slot_count(), 1);
    }

    #[test]
    fn dropping_any_unfinished_phase_keeps_shared_physical_custody() {
        for phase in 0..4 {
            let mut pool = pool(1);
            let mut alias = pool.clone();
            let page = KvPageId(10);
            let mut handle = staged(&mut pool, 1, page);
            if phase >= 1 {
                pool.commit_prepared(&mut handle).unwrap();
            }
            if phase >= 2 {
                pool.publish_prepared(&handle).unwrap();
            }
            if phase >= 3 {
                pool.retire_prepared(&mut handle, &[page]).unwrap();
            }
            drop(handle);
            drop(pool);
            assert_eq!(alias.active_transaction_count(), 1);
            assert_eq!(alias.free_slot_count(), 0);
            assert!(alias.release(&[page]).is_err());
            assert!(alias.preempt(&[page]).is_err());
            assert!(alias.shutdown().is_err());
            assert!(alias.configure_capacity(1).is_err());
            assert!(alias.prepare(prepare(1, &[])).is_err());
            assert!(alias.prepare(prepare(2, &[page])).is_err());
            assert!(alias.prepare(prepare(2, &[KvPageId(20)])).is_err());
        }
    }

    #[test]
    fn foreign_pool_authority_rejected_even_with_matching_id_and_nonce() {
        let (mut a, mut b) = (pool(1), pool(1));
        let page = KvPageId(10);
        let mut ha = Some(staged(&mut a, 1, page));
        let mut hb = Some(staged(&mut b, 1, page));
        assert!(b.commit_prepared(ha.as_mut().unwrap()).is_err());
        a.commit_prepared(ha.as_mut().unwrap()).unwrap();
        b.commit_prepared(hb.as_mut().unwrap()).unwrap();
        assert_eq!(
            ha.as_ref().unwrap().prepared_nonce,
            hb.as_ref().unwrap().prepared_nonce
        );
        assert!(b.publish_prepared(ha.as_ref().unwrap()).is_err());
        a.publish_prepared(ha.as_ref().unwrap()).unwrap();
        b.publish_prepared(hb.as_ref().unwrap()).unwrap();
        assert!(
            b.preflight_retirement(ha.as_ref().unwrap(), &[page])
                .is_err()
        );
        assert!(b.retire_prepared(ha.as_mut().unwrap(), &[page]).is_err());
        assert_eq!(snapshot(&a, page), snapshot(&b, page));
        a.retire_prepared(ha.as_mut().unwrap(), &[page]).unwrap();
        b.retire_prepared(hb.as_mut().unwrap(), &[page]).unwrap();
        assert!(b.finish_prepared(&mut ha).is_err());
        assert!(ha.is_some());
        assert_eq!(a.free_slot_count(), 0);
        assert_eq!(b.free_slot_count(), 0);
        a.finish_prepared(&mut ha).unwrap();
        b.finish_prepared(&mut hb).unwrap();
    }

    #[test]
    fn stale_nonce_cannot_finish_a_reused_transaction_id() {
        let mut pool = pool(1);
        let page = KvPageId(10);
        let mut handle = Some(staged(&mut pool, 1, page));
        pool.commit_prepared(handle.as_mut().unwrap()).unwrap();
        // Replay the private wire identity to exercise validation beneath the
        // public non-cloneable authority; safe callers cannot create this copy.
        let mut stale = Some(CpuPagedKvTransaction {
            id: handle.as_ref().unwrap().id,
            identity: handle.as_ref().unwrap().identity.clone(),
            prepared_nonce: handle.as_ref().unwrap().prepared_nonce,
        });
        pool.publish_prepared(handle.as_ref().unwrap()).unwrap();
        pool.retire_prepared(handle.as_mut().unwrap(), &[page])
            .unwrap();
        pool.finish_prepared(&mut handle).unwrap();
        handle = Some(staged(&mut pool, 1, page));
        pool.commit_prepared(handle.as_mut().unwrap()).unwrap();
        assert_ne!(
            stale.as_ref().unwrap().prepared_nonce,
            handle.as_ref().unwrap().prepared_nonce
        );
        assert!(pool.publish_prepared(stale.as_ref().unwrap()).is_err());
        assert!(
            pool.retire_prepared(stale.as_mut().unwrap(), &[page])
                .is_err()
        );
        assert!(pool.finish_prepared(&mut stale).is_err());
        assert!(stale.is_some());
        assert_eq!(pool.active_transaction_count(), 1);
        assert_eq!(pool.free_slot_count(), 0);
        pool.publish_prepared(handle.as_ref().unwrap()).unwrap();
        pool.retire_prepared(handle.as_mut().unwrap(), &[page])
            .unwrap();
        pool.finish_prepared(&mut handle).unwrap();
    }

    #[test]
    fn prepared_abort_keeps_cow_dependencies_and_slots_until_finish_or_after_drop() {
        for finish in [false, true] {
            let mut pool = pool(2);
            let mut alias = pool.clone();
            let source = KvPageId(10);
            let replacement = KvPageId(11);
            let mut seed = Some(staged(&mut pool, 1, source));
            pool.commit(&mut seed).unwrap();
            let before = snapshot(&pool, source);
            let mut handle = Some(
                pool.prepare(PagedKvPrepare {
                    cow_replacements: &[KvCowReplacement {
                        source,
                        replacement,
                        logical_page: 0,
                    }],
                    ..prepare(2, &[])
                })
                .unwrap(),
            );
            for _ in 0..2 {
                assert_eq!(
                    pool.abort_prepared(handle.as_mut().unwrap()).unwrap(),
                    KvEndProgress::Complete
                );
                assert_eq!(alias.free_slot_count(), 0);
                assert!(alias.release(&[source]).is_err());
                assert!(alias.preempt(&[source]).is_err());
                assert!(alias.shutdown().is_err());
                assert!(alias.prepare(prepare(2, &[])).is_err());
                assert!(alias.prepare(prepare(3, &[replacement])).is_err());
                assert!(pool.commit_prepared(handle.as_mut().unwrap()).is_err());
                assert!(pool.publish_prepared(handle.as_ref().unwrap()).is_err());
            }
            assert_eq!(snapshot(&alias, source), before);
            assert_eq!(alias.physical_slot(replacement), None);
            if finish {
                assert!(pool.finish_prepared(&mut handle).unwrap().is_empty());
                alias.release(&[source]).unwrap();
                assert_eq!(alias.free_slot_count(), 2);
                assert_eq!(alias.active_transaction_count(), 0);
                alias.shutdown().unwrap();
            } else {
                drop(handle);
                drop(pool);
                assert!(alias.release(&[source]).is_err());
                assert!(alias.shutdown().is_err());
                assert_eq!(alias.free_slot_count(), 0);
                assert_eq!(alias.active_transaction_count(), 1);
            }
        }
    }

    #[test]
    fn ordinary_commit_and_abort_do_not_leave_prepared_pins() {
        for commit in [false, true] {
            let mut pool = pool(1);
            let mut alias = pool.clone();
            let page = KvPageId(10);
            let mut handle = Some(staged(&mut pool, 1, page));
            if commit {
                pool.commit(&mut handle).unwrap();
                alias.release(&[page]).unwrap();
            } else {
                pool.abort(&mut handle).unwrap();
            }
            assert!(handle.is_none());
            assert_eq!(alias.active_transaction_count(), 0);
            assert_eq!(alias.free_slot_count(), 1);
            alias.shutdown().unwrap();
            alias.configure_capacity(1).unwrap();
        }
    }

    #[test]
    fn nonce_exhaustion_rejects_before_install_and_preserves_abort_custody() {
        let mut pool = pool(1);
        let page = KvPageId(10);
        let mut handle = Some(staged(&mut pool, 1, page));
        pool.inner.borrow_mut().next_lease_nonce = u64::MAX;
        for _ in 0..2 {
            assert!(pool.commit_prepared(handle.as_mut().unwrap()).is_err());
            assert_eq!(pool.physical_slot(page), None);
            assert_eq!(pool.active_transaction_count(), 1);
            assert_eq!(pool.free_slot_count(), 0);
            assert_eq!(handle.as_ref().unwrap().prepared_nonce, None);
        }
        pool.abort(&mut handle).unwrap();
        assert_eq!(pool.free_slot_count(), 1);
        // Exhausting prepared authority must not break the ordinary lifecycle.
        handle = Some(staged(&mut pool, 2, page));
        pool.commit(&mut handle).unwrap();
        pool.release(&[page]).unwrap();
    }

    #[test]
    fn retirement_cannot_bypass_another_active_or_prepared_cow_reader() {
        let mut pool = pool(2);
        let source = KvPageId(10);
        let replacement = KvPageId(11);
        let mut seed = Some(staged(&mut pool, 1, source));
        pool.commit(&mut seed).unwrap();
        let mut reader = Some(
            pool.prepare(PagedKvPrepare {
                protected_pages: &[source],
                ..prepare(2, &[])
            })
            .unwrap(),
        );
        let mut writer = Some(
            pool.prepare(PagedKvPrepare {
                cow_replacements: &[KvCowReplacement {
                    source,
                    replacement,
                    logical_page: 0,
                }],
                ..prepare(3, &[])
            })
            .unwrap(),
        );
        pool.commit_prepared(writer.as_mut().unwrap()).unwrap();
        pool.publish_prepared(writer.as_ref().unwrap()).unwrap();
        assert!(
            pool.retire_prepared(writer.as_mut().unwrap(), &[source])
                .is_err()
        );
        pool.commit_prepared(reader.as_mut().unwrap()).unwrap();
        assert!(
            pool.retire_prepared(writer.as_mut().unwrap(), &[source])
                .is_err()
        );
        assert_eq!(snapshot(&pool, source), snapshot(&pool, replacement));
        pool.publish_prepared(reader.as_ref().unwrap()).unwrap();
        pool.retire_prepared(reader.as_mut().unwrap(), &[]).unwrap();
        pool.finish_prepared(&mut reader).unwrap();
        pool.retire_prepared(writer.as_mut().unwrap(), &[source])
            .unwrap();
        assert_eq!(pool.free_slot_count(), 0);
        pool.finish_prepared(&mut writer).unwrap();
        pool.release(&[replacement]).unwrap();
        assert_eq!(pool.free_slot_count(), 2);
    }
}
