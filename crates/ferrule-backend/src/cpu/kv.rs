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
    entered: bool,
    batch: Option<PagedKvBatch>,
    pages: BTreeMap<KvPageId, StagedPage>,
    protected_pages: BTreeSet<KvPageId>,
}

#[derive(Debug)]
struct PoolInner {
    planes: Box<[KvPlaneDescriptor]>,
    page_size: usize,
    page_elements: Box<[usize]>,
    capacity: usize,
    storage: Vec<CpuKvPlaneStorage>,
    free_slots: Vec<usize>,
    resident: HashMap<KvPageId, usize>,
    preempted: HashMap<KvPageId, Vec<CpuKvPlaneStorage>>,
    active: HashMap<ExecutionTransactionId, ActiveTransaction>,
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
            planes: planes.into_boxed_slice(),
            page_size,
            page_elements: page_elements.into_boxed_slice(),
            capacity: 0,
            storage: Vec::new(),
            free_slots: Vec::new(),
            resident: HashMap::new(),
            preempted: HashMap::new(),
            active: HashMap::new(),
            shutdown: false,
        };
        inner.configure_capacity(capacity)?;
        Ok(inner)
    }

    fn configure_capacity(&mut self, capacity: usize) -> Result<()> {
        if capacity == 0 {
            return Err(cpu_error("CPU paged KV capacity must be non-zero"));
        }
        if !self.active.is_empty() || !self.resident.is_empty() || !self.preempted.is_empty() {
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
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct CpuPagedKvTransaction {
    id: ExecutionTransactionId,
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
        self.inner.borrow().active.len()
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
        if inner.active.contains_key(&request.transaction) {
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
            if let Some(page) = request
                .protected_pages
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
                entered: false,
                batch: None,
                pages,
                protected_pages: request.protected_pages.iter().copied().collect(),
            },
        );
        Ok(CpuPagedKvTransaction {
            id: request.transaction,
        })
    }

    pub fn enter(
        &mut self,
        transaction: &mut CpuPagedKvTransaction,
        batch: PagedKvBatch,
    ) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
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
        let active = inner.active.get(&transaction.id).ok_or_else(|| {
            cpu_error(format!(
                "CPU KV transaction {} is not active",
                transaction.id.get()
            ))
        })?;
        if !active.entered || active.batch.is_none() {
            return Err(cpu_error("CPU KV transaction has no active batch"));
        }
        drop(inner);
        Ok(CpuKvView {
            inner: Rc::clone(&self.inner),
            transaction: transaction.id,
        })
    }

    pub fn leave(&mut self, transaction: &mut CpuPagedKvTransaction) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
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

    pub fn commit(
        &mut self,
        transaction: &mut Option<CpuPagedKvTransaction>,
    ) -> Result<KvEndProgress> {
        let id = transaction
            .as_ref()
            .ok_or_else(|| cpu_error("CPU KV commit transaction is absent"))?
            .id;
        let mut inner = self.inner.borrow_mut();
        let active = inner
            .active
            .get(&id)
            .ok_or_else(|| cpu_error("CPU KV transaction is not active"))?;
        if active.entered {
            return Err(cpu_error("cannot commit an entered CPU KV transaction"));
        }
        for (page, staged) in &active.pages {
            match staged {
                StagedPage::Physical { .. }
                    if inner.resident.contains_key(page) || inner.preempted.contains_key(page) =>
                {
                    return Err(cpu_error(format!(
                        "CPU KV page {} became occupied before commit",
                        page.0
                    )));
                }
                StagedPage::Shadow { resident_slot, .. }
                    if inner.resident.get(page) != Some(resident_slot) =>
                {
                    return Err(cpu_error(format!(
                        "CPU KV shadow page {} changed physical identity",
                        page.0
                    )));
                }
                _ => {}
            }
        }
        let active = inner
            .active
            .remove(&id)
            .expect("validated CPU KV transaction exists");
        for (page, staged) in active.pages {
            match staged {
                StagedPage::Physical { slot } => {
                    inner.resident.insert(page, slot);
                }
                StagedPage::Shadow {
                    resident_slot,
                    planes,
                } => inner.overwrite_slot(resident_slot, &planes)?,
            }
        }
        drop(inner);
        transaction.take();
        Ok(KvEndProgress::Complete)
    }

    pub fn abort(
        &mut self,
        transaction: &mut Option<CpuPagedKvTransaction>,
    ) -> Result<KvEndProgress> {
        let id = transaction
            .as_ref()
            .ok_or_else(|| cpu_error("CPU KV abort transaction is absent"))?
            .id;
        let mut inner = self.inner.borrow_mut();
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

    pub fn capacity(&self) -> KvCapacity {
        let inner = self.inner.borrow();
        KvCapacity {
            physical_pages: inner.capacity,
            resident_pages: inner.resident.len(),
            preempted_pages: inner.preempted.len(),
            active_transactions: inner.active.len(),
            free_pages: inner.free_slots.len(),
        }
    }

    pub fn shutdown(&mut self) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        if !inner.active.is_empty() {
            return Err(cpu_error(
                "cannot shut down CPU paged KV with active transactions",
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
}

impl CpuKvView {
    pub const fn transaction(&self) -> ExecutionTransactionId {
        self.transaction
    }

    pub fn validate_batch(&self, batch: &PagedKvBatch) -> Result<()> {
        let inner = self.inner.borrow();
        let active = active_transaction(&inner, self.transaction)?;
        if active.batch.as_ref() != Some(batch) {
            return Err(cpu_error("CPU KV view is bound to a different batch"));
        }
        Ok(())
    }

    pub fn physical_slot(&self, page: KvPageId) -> Option<usize> {
        let inner = self.inner.borrow();
        let active = inner.active.get(&self.transaction)?;
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
        match page_storage(&inner, self.transaction, page, plane)? {
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
        match page_storage(&inner, self.transaction, page, plane)? {
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
        match writable_page_storage(&mut inner, self.transaction, page, plane)? {
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
        match writable_page_storage(&mut inner, self.transaction, page, plane)? {
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
        let batch = active_transaction(&inner, self.transaction)?
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
                    writable_page_storage(&mut inner, self.transaction, page, plane)?,
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
        let active = active_transaction(&inner, self.transaction)?;
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
                    page_storage(&inner, self.transaction, page, plane)?,
                    range,
                    destination,
                );
            }
        }
        Ok(KvHistory { tokens, key, value })
    }
}

fn active_transaction(
    inner: &PoolInner,
    transaction: ExecutionTransactionId,
) -> Result<&ActiveTransaction> {
    let active = inner
        .active
        .get(&transaction)
        .ok_or_else(|| cpu_error("CPU KV view transaction is no longer active"))?;
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

fn page_storage(
    inner: &PoolInner,
    transaction: ExecutionTransactionId,
    page: KvPageId,
    plane: usize,
) -> Result<PageStorageRef<'_>> {
    let active = active_transaction(inner, transaction)?;
    if let Some(StagedPage::Shadow { planes, .. }) = active.pages.get(&page) {
        return snapshot_ref(planes, plane);
    }
    let slot = active
        .pages
        .get(&page)
        .and_then(StagedPage::physical_slot)
        .or_else(|| inner.resident.get(&page).copied())
        .ok_or_else(|| cpu_error(format!("CPU KV page {} is not readable", page.0)))?;
    storage_ref(&inner.storage[plane], inner.slot_range(slot, plane)?)
}

fn writable_page_storage(
    inner: &mut PoolInner,
    transaction: ExecutionTransactionId,
    page: KvPageId,
    plane: usize,
) -> Result<PageStorageMut<'_>> {
    let (slot, shadow) = {
        let active = active_transaction(inner, transaction)?;
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
