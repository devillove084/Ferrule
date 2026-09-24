use super::DecoderSequence;
#[cfg(test)]
use super::DecoderSequenceState;
use super::{
    DecoderKvBackend, DecoderKvCapacity, DecoderKvCommitBackend, DecoderKvPageStatus,
    DecoderKvPageView, DecoderKvPrepare, KvCommitBinding, KvEndProgress, KvRankAck,
    PackedDecoderBatch,
};
use crate::transformer::{KvAppendRequest, KvHistory, KvView as TransformerKvView};
use ferrule_backend::cpu::{
    self, KvEndProgress as BackendKvEndProgress, KvPageStatus as BackendKvPageStatus,
    PagedKvHistory,
};
use ferrule_common::Result;
use ferrule_common::execution::{KvElementType, KvLayoutSchema, KvPageId, KvPlaneDescriptor};
use std::collections::{BTreeMap, BTreeSet};
use std::mem::ManuallyDrop;
use std::sync::Arc;

pub use ferrule_backend::cpu::CpuKvPlaneStorage;

#[cfg(feature = "cuda")]
mod cuda;
mod prepare_error;
mod prepared_cpu;
pub use prepare_error::KvPrepareQuiescenceUnknown;
#[cfg(test)]
mod prepare_tests;
#[cfg(feature = "cuda")]
pub use cuda::{
    CudaGqaPlanesMut, CudaKvView, CudaPagedKvBackend, CudaPagedKvPool, CudaPagedKvTransaction,
    TypedCudaPagedKvPool,
};

/// Borrowed input. A successful preflight transfers the non-cloneable handle
/// out of this slot; failure leaves all handles and the logical token untouched.
pub struct KvCommitOwner<'a, B: DecoderKvCommitBackend> {
    rank: ferrule_common::ParallelRankId,
    backend: &'a mut B,
    slot: &'a mut Option<B::Transaction>,
    sources: &'a [B::SequenceState],
}
impl<'a, B: DecoderKvCommitBackend> KvCommitOwner<'a, B> {
    pub fn new(
        rank: ferrule_common::ParallelRankId,
        backend: &'a mut B,
        slot: &'a mut Option<B::Transaction>,
        sources: &'a [B::SequenceState],
    ) -> Self {
        Self {
            rank,
            backend,
            slot,
            sources,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OwnerEnd {
    Ready,
    Installing,
    Installed,
    Aborting,
    Aborted,
    Retired,
}

/// Sealed owner capability. No public constructor, Clone, or mutable backend
/// accessor. The original ledger stays pinned even if an unfinished token drops.
///
/// ```compile_fail
/// use ferrule_model::decoder::{CpuPagedKvBackend, PhysicalKvCommitReady};
/// fn duplicate(token: PhysicalKvCommitReady<'_, CpuPagedKvBackend>) {
///     let _duplicate = token.clone();
/// }
/// ```
#[must_use = "retain physical custody until every owner ACK"]
pub struct PhysicalKvCommitReady<'a, B: DecoderKvCommitBackend> {
    owner: KvCommitOwner<'a, B>,
    transaction: ManuallyDrop<Option<B::Transaction>>,
    binding: KvCommitBinding,
    generation: u64,
    end: OwnerEnd,
}
impl<B: DecoderKvCommitBackend> PhysicalKvCommitReady<'_, B> {
    pub const fn binding(&self) -> &KvCommitBinding {
        &self.binding
    }
    pub const fn rank(&self) -> ferrule_common::ParallelRankId {
        self.owner.rank
    }
    pub const fn generation(&self) -> u64 {
        self.generation
    }
    fn ack(&self, progress: KvEndProgress) -> KvRankAck {
        KvRankAck::new(self.binding.clone(), self.rank(), self.generation, progress)
    }
    fn finish(&mut self) {
        let transaction = self.transaction.take().expect("sealed owner custody");
        self.owner.backend.finish_prepared(transaction);
    }
    fn preflight_install(&self) -> Result<()> {
        let transaction = self.transaction.as_ref().expect("sealed owner custody");
        self.owner.backend.preflight_commit_ready(
            transaction,
            &self.binding,
            self.owner.rank,
            self.owner.sources,
        )
    }
}

/// Contains every original input on failure, without automatic physical abort.
pub struct PrepareKvCommitError<'a, B: DecoderKvCommitBackend, L> {
    error: ferrule_common::Error,
    logical: L,
    owners: Vec<KvCommitOwner<'a, B>>,
}
impl<'a, B: DecoderKvCommitBackend, L> PrepareKvCommitError<'a, B, L> {
    pub fn into_parts(self) -> (ferrule_common::Error, L, Vec<KvCommitOwner<'a, B>>) {
        (self.error, self.logical, self.owners)
    }
}
impl<B: DecoderKvCommitBackend, L> std::fmt::Debug for PrepareKvCommitError<'_, B, L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrepareKvCommitError")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}
impl<B: DecoderKvCommitBackend, L> std::fmt::Display for PrepareKvCommitError<'_, B, L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.error, f)
    }
}
impl<B: DecoderKvCommitBackend, L> std::error::Error for PrepareKvCommitError<'_, B, L> {}

/// KV-only custody, not a decision coordinator. L must be the existing logical
/// owner's already-prepared reservation for this exact cohort. The callback at
/// publish is its infallible logical publish; returned pages are zero-refcount
/// quarantine pages, NOT every COW source (shared sources must remain resident).
///
/// Dropping unfinished custody leaks L and the physical handles intentionally:
/// unknown owner work must not be freed by RAII. Existing backend ledgers retain
/// their pins, and no caller can retrieve a handle to bypass this protocol.
///
/// ```compile_fail
/// use ferrule_model::decoder::{CpuPagedKvBackend, PreparedKvCommit};
/// fn duplicate(token: PreparedKvCommit<'_, CpuPagedKvBackend, ()>) {
///     let _duplicate = token.clone();
/// }
/// ```
#[must_use = "publish or explicitly abort sealed KV custody"]
pub struct PreparedKvCommit<'a, B: DecoderKvCommitBackend, L> {
    binding: KvCommitBinding,
    logical: Option<ManuallyDrop<L>>,
    owners: Vec<PhysicalKvCommitReady<'a, B>>,
}
impl<'a, B: DecoderKvCommitBackend, L> PreparedKvCommit<'a, B, L> {
    pub fn prepare_commit_ready(
        binding: KvCommitBinding,
        logical: L,
        owners: Vec<KvCommitOwner<'a, B>>,
    ) -> std::result::Result<Self, PrepareKvCommitError<'a, B, L>> {
        let check = || -> Result<()> {
            let ranks = owners
                .iter()
                .map(|owner| owner.rank)
                .collect::<BTreeSet<_>>();
            if owners.len() != binding.participants().len()
                || ranks.len() != owners.len()
                || ranks.iter().copied().ne(binding.participants().iter())
            {
                return Err(kv_error(
                    "KV ready requires the exact distinct participant set",
                ));
            }
            for owner in &owners {
                let transaction = owner
                    .slot
                    .as_ref()
                    .ok_or_else(|| kv_error("KV owner handle absent"))?;
                owner.backend.preflight_commit_ready(
                    transaction,
                    &binding,
                    owner.rank,
                    owner.sources,
                )?;
            }
            let first = owners[0]
                .backend
                .commit_batch(owners[0].slot.as_ref().expect("checked owner"))?;
            for owner in &owners[1..] {
                validate_commit_projection(
                    first,
                    owner
                        .backend
                        .commit_batch(owner.slot.as_ref().expect("checked owner"))?,
                )?;
            }
            Ok(())
        };
        if let Err(error) = check() {
            return Err(PrepareKvCommitError {
                error,
                logical,
                owners,
            });
        }
        // Allocate the cohort incarnation before consuming any inputs.
        let generation = match NEXT_PREPARED_KV.try_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |n| n.checked_add(1),
        ) {
            Ok(generation) => generation,
            Err(_) => {
                return Err(PrepareKvCommitError {
                    error: kv_error("KV custody generation exhausted"),
                    logical,
                    owners,
                });
            }
        };
        let owners = owners
            .into_iter()
            .map(|owner| {
                let transaction = ManuallyDrop::new(owner.slot.take());
                PhysicalKvCommitReady {
                    owner,
                    transaction,
                    binding: binding.clone(),
                    generation,
                    end: OwnerEnd::Ready,
                }
            })
            .collect();
        Ok(Self {
            binding,
            logical: Some(ManuallyDrop::new(logical)),
            owners,
        })
    }
    pub const fn binding(&self) -> &KvCommitBinding {
        &self.binding
    }
    pub fn owner(
        &self,
        rank: ferrule_common::ParallelRankId,
    ) -> Result<&PhysicalKvCommitReady<'a, B>> {
        self.owners
            .iter()
            .find(|owner| owner.rank() == rank)
            .ok_or_else(|| kv_error("unknown KV owner"))
    }
    pub fn page_status(
        &self,
        rank: ferrule_common::ParallelRankId,
        page: KvPageId,
    ) -> Result<DecoderKvPageStatus> {
        Ok(self.owner(rank)?.owner.backend.page_status(page))
    }
    pub fn capacity(&self, rank: ferrule_common::ParallelRankId) -> Result<DecoderKvCapacity> {
        Ok(self.owner(rank)?.owner.backend.capacity())
    }
    fn check(&self, expected: &KvCommitBinding) -> Result<()> {
        if &self.binding != expected || self.logical.is_none() {
            return Err(kv_error("stale or consumed KV commit token"));
        }
        Ok(())
    }
    fn owner_mut(
        &mut self,
        rank: ferrule_common::ParallelRankId,
    ) -> Result<&mut PhysicalKvCommitReady<'a, B>> {
        self.owners
            .iter_mut()
            .find(|owner| owner.rank() == rank)
            .ok_or_else(|| kv_error("unknown KV owner"))
    }
    /// Apply the external global Commit decision. Dispatch once; Err may mean a
    /// lost ACK, so even errors must be retried via poll_install_ack, not install.
    pub fn install_commit(
        &mut self,
        expected: &KvCommitBinding,
        rank: ferrule_common::ParallelRankId,
    ) -> Result<KvRankAck> {
        self.check(expected)?;
        if self
            .owners
            .iter()
            .any(|o| matches!(o.end, OwnerEnd::Aborting | OwnerEnd::Aborted))
        {
            return Err(kv_error("cannot reverse KV abort into commit"));
        }
        let owner = self.owner_mut(rank)?;
        if owner.end != OwnerEnd::Ready {
            return Err(kv_error("duplicate KV install"));
        }
        // This is deliberately separate from ready preparation. It proves that
        // the exact page mapping, COW source, shape, capacity and fence still
        // match immediately before the one physical install dispatch. A failed
        // revalidation leaves the token Ready and all custody untouched.
        owner.preflight_install()?;
        owner.end = OwnerEnd::Installing;
        let progress = match owner.owner.backend.install_commit(
            owner.transaction.as_mut().expect("owner custody"),
            owner.generation,
        ) {
            Ok(progress) => progress,
            Err(error) => {
                // The adapter may have lost its ACK after dispatch; never reset
                // to Ready and risk a duplicate install. Poll retains custody.
                owner.end = OwnerEnd::Installing;
                return Err(error);
            }
        };
        check_ack(progress)?;
        if progress == KvEndProgress::Complete {
            owner.end = OwnerEnd::Installed;
        }
        Ok(owner.ack(progress))
    }
    pub fn poll_install_ack(
        &mut self,
        expected: &KvCommitBinding,
        rank: ferrule_common::ParallelRankId,
    ) -> Result<KvRankAck> {
        self.check(expected)?;
        let owner = self.owner_mut(rank)?;
        if owner.end != OwnerEnd::Installing {
            return Err(kv_error("KV install ACK is not pending"));
        }
        let progress = match owner.owner.backend.poll_install_ack(
            owner.transaction.as_mut().expect("owner custody"),
            owner.generation,
        ) {
            Ok(progress) => progress,
            Err(error) => {
                owner.end = OwnerEnd::Installing;
                return Err(error);
            }
        };
        check_ack(progress)?;
        if progress == KvEndProgress::Complete {
            owner.end = OwnerEnd::Installed;
        }
        Ok(owner.ack(progress))
    }
    /// Apply the external Abort decision, before any install dispatch. Retry
    /// until every owner cleanup and fence is acknowledged, including lost ACKs.
    pub fn abort_prepared(
        &mut self,
        expected: &KvCommitBinding,
        rank: ferrule_common::ParallelRankId,
    ) -> Result<KvRankAck> {
        self.check(expected)?;
        if self
            .owners
            .iter()
            .any(|o| matches!(o.end, OwnerEnd::Installing | OwnerEnd::Installed))
        {
            return Err(kv_error("cannot reverse dispatched KV commit into abort"));
        }
        let owner = self.owner_mut(rank)?;
        if owner.end == OwnerEnd::Aborted {
            return Ok(owner.ack(KvEndProgress::Complete));
        }
        owner.end = OwnerEnd::Aborting;
        let progress = owner.owner.backend.abort_prepared(
            owner.transaction.as_mut().expect("owner custody"),
            owner.generation,
        )?;
        check_ack(progress)?;
        if progress == KvEndProgress::Complete {
            owner.end = OwnerEnd::Aborted;
        }
        Ok(owner.ack(progress))
    }
    pub fn publish_logical<R>(
        &mut self,
        expected: &KvCommitBinding,
        publish: impl FnOnce(L) -> (R, Vec<KvPageId>),
    ) -> Result<PreparedKvRetirement<'a, B, R>> {
        self.check(expected)?;
        if self
            .owners
            .iter()
            .any(|owner| owner.end != OwnerEnd::Installed)
        {
            return Err(kv_error(
                "logical publish requires ALL physical install/fence ACKs",
            ));
        }
        for owner in &mut self.owners {
            owner
                .owner
                .backend
                .publish_committed(owner.transaction.as_ref().expect("owner custody"));
        }
        let logical = ManuallyDrop::into_inner(self.logical.take().expect("logical custody"));
        let (retirement, pages) = publish(logical);
        Ok(PreparedKvRetirement {
            binding: self.binding.clone(),
            retirement: Some(ManuallyDrop::new(retirement)),
            pages,
            owners: std::mem::take(&mut self.owners),
            checked: false,
        })
    }
    pub fn abort_logical<R>(
        &mut self,
        expected: &KvCommitBinding,
        abort: impl FnOnce(L) -> R,
    ) -> Result<R> {
        self.check(expected)?;
        if self
            .owners
            .iter()
            .any(|owner| owner.end != OwnerEnd::Aborted)
        {
            return Err(kv_error(
                "logical abort requires ALL owner cleanup/fence ACKs",
            ));
        }
        for owner in &mut self.owners {
            owner.finish();
        }
        self.owners.clear();
        Ok(abort(ManuallyDrop::into_inner(
            self.logical.take().expect("logical custody"),
        )))
    }
}

/// Owns the existing logical quarantine token until every physical release ACK.
/// Failures are retryable. Dropping this token retains quarantine and ledger pins.
#[must_use = "retirement needs every owner/fence ACK before logical reuse"]
pub struct PreparedKvRetirement<'a, B: DecoderKvCommitBackend, R> {
    binding: KvCommitBinding,
    retirement: Option<ManuallyDrop<R>>,
    pages: Vec<KvPageId>,
    owners: Vec<PhysicalKvCommitReady<'a, B>>,
    checked: bool,
}
impl<B: DecoderKvCommitBackend, R> PreparedKvRetirement<'_, B, R> {
    pub fn pages(&self) -> &[KvPageId] {
        &self.pages
    }
    fn check(&self, expected: &KvCommitBinding) -> Result<()> {
        if &self.binding != expected || self.retirement.is_none() {
            return Err(kv_error("stale or consumed KV retirement token"));
        }
        Ok(())
    }
    pub fn retire_owner(
        &mut self,
        expected: &KvCommitBinding,
        rank: ferrule_common::ParallelRankId,
    ) -> Result<KvRankAck> {
        self.check(expected)?;
        if !self.checked {
            unique_page_set(&self.pages, "retirement")?;
            for owner in &self.owners {
                owner.owner.backend.preflight_retirement(
                    owner.transaction.as_ref().expect("owner custody"),
                    &self.pages,
                )?;
            }
            self.checked = true;
        }
        let owner = self
            .owners
            .iter_mut()
            .find(|owner| owner.rank() == rank)
            .ok_or_else(|| kv_error("unknown retirement owner"))?;
        if owner.end == OwnerEnd::Retired {
            return Ok(owner.ack(KvEndProgress::Complete));
        }
        let progress = owner.owner.backend.retire_prepared(
            owner.transaction.as_mut().expect("owner custody"),
            owner.generation,
            &self.pages,
        )?;
        check_ack(progress)?;
        if progress == KvEndProgress::Complete {
            owner.end = OwnerEnd::Retired;
        }
        Ok(owner.ack(progress))
    }
    pub fn finish_retirement(&mut self, expected: &KvCommitBinding) -> Result<R> {
        self.check(expected)?;
        if self
            .owners
            .iter()
            .any(|owner| owner.end != OwnerEnd::Retired)
        {
            return Err(kv_error("retirement requires ALL owner/fence ACKs"));
        }
        for owner in &mut self.owners {
            owner.finish();
        }
        self.owners.clear();
        Ok(ManuallyDrop::into_inner(
            self.retirement.take().expect("quarantine custody"),
        ))
    }
}
static NEXT_PREPARED_KV: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
fn check_ack(progress: KvEndProgress) -> Result<()> {
    if progress == KvEndProgress::ConsumedRejected {
        return Err(kv_error(
            "consumed rejection is not an ACK; KV custody retained",
        ));
    }
    Ok(())
}
fn validate_commit_projection(a: &PackedDecoderBatch, b: &PackedDecoderBatch) -> Result<()> {
    if a.page_size() != b.page_size()
        || a.new_pages() != b.new_pages()
        || a.writable_pages() != b.writable_pages()
        || a.cow_replacements() != b.cow_replacements()
        || a.protected_pages() != b.protected_pages()
        || a.row_to_sequence() != b.row_to_sequence()
        || a.positions() != b.positions()
        || a.sequences().len() != b.sequences().len()
    {
        return Err(kv_error(
            "KV participants disagree on page/COW/shape projection",
        ));
    }
    for (x, y) in a.sequences().iter().zip(b.sequences()) {
        if x.page_state_slot() != y.page_state_slot()
            || x.page_generation() != y.page_generation()
            || x.context_len() != y.context_len()
            || x.query_len() != y.query_len()
            || x.block_table() != y.block_table()
        {
            return Err(kv_error(
                "KV participants disagree on logical mapping/generation",
            ));
        }
    }
    Ok(())
}

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
    // Zero only for standalone ledger users, which do not issue backend handles.
    handle_nonce: u64,
    new_pages: BTreeSet<KvPageId>,
    writable_pages: BTreeSet<KvPageId>,
    cow_replacements: Box<[ferrule_common::execution::KvCowReplacement]>,
    protected_pages: BTreeSet<KvPageId>,
    custody: Box<[super::DecoderKvSequenceCustody]>,
    batch: Option<PackedDecoderBatch>,
    entered: bool,
    published: bool,
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
        let mut protected_pages = unique_page_set(request.protected_pages, "protected")?;
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
        if !new_pages.is_disjoint(&replacements) {
            return Err(kv_error("new and COW pages overlap"));
        }
        writable_pages.extend(new_pages.iter().copied());
        protected_pages.extend(writable_pages.iter().copied());
        protected_pages.extend(request.cow_replacements.iter().map(|cow| cow.source));
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
        if provisional > self.capacity().free_pages {
            return Err(kv_error("paged KV transaction exceeds physical capacity"));
        }
        self.pending.insert(
            request.transaction,
            PagedKvTransaction {
                handle_nonce: 0,
                new_pages,
                writable_pages,
                cow_replacements: request.cow_replacements.into(),
                protected_pages,
                custody: request.sequences.into(),
                batch: None,
                entered: false,
                published: false,
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
                        .filter(|pending| !pending.published)
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
    /// Optional physical free-slot witness. CUDA uses it so shadow slots are
    /// charged immediately; CPU/legacy pools retain their logical semantics.
    fn physical_free_slots(&self) -> Option<usize> {
        None
    }
    fn configure_capacity(&mut self, max_pages: usize) -> Result<()>;
    fn prepare(&mut self, request: DecoderKvPrepare<'_>) -> Result<Self::Transaction>;
    /// Whether a failed prepare may still own physical work without returning a
    /// handle. Asynchronous pools must report retention here or return
    /// `KvPrepareQuiescenceUnknown`, keeping pins until completion is proven.
    /// An untyped error with false guarantees no work or proven cleanup.
    fn prepare_failure_retains_custody(&self) -> bool {
        false
    }
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
    /// Optional legacy single-rank preflight. A default does NOT qualify the
    /// pool for the additive all-owner DecoderKvCommitBackend contract.
    fn preflight_commit(
        &self,
        _pending: &PagedKvTransaction<Option<Self::Transaction>>,
    ) -> Result<()> {
        Ok(())
    }
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

/// Additive physical prepared-commit contract for pools that can prove exact
/// device-side installation and cleanup. Ordinary/MLA pools do not implement
/// this trait and keep the legacy single-rank commit path unchanged.
pub trait PhysicalKvPreparedPool: PhysicalKvPool {
    fn preflight_prepared(&self, transaction: &Self::Transaction) -> Result<()>;
    fn install_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress>;
    fn poll_install_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress>;
    fn abort_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress>;
    fn publish_prepared(&mut self, transaction: &Self::Transaction) -> Result<()>;
    fn preflight_retirement(
        &self,
        transaction: &Self::Transaction,
        pages: &[KvPageId],
    ) -> Result<()>;
    fn retire_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
        pages: &[KvPageId],
    ) -> Result<KvEndProgress>;
    fn finish_prepared(&mut self, transaction: Self::Transaction) -> Result<Vec<KvPageId>>;
}

// Linear single-rank permission. Unadapted legacy pools keep their old Pending
// retry semantics; the distributed API dispatches once and polls ACKs instead.
struct SingleKvCommitReady(ferrule_common::execution::ExecutionTransactionId);
impl SingleKvCommitReady {
    fn install<P: PhysicalKvPool>(
        self,
        pool: &mut P,
        ownership: &mut PagedKvOwnership<Option<P::Transaction>>,
    ) -> Result<KvEndProgress> {
        pool.commit(ownership.payload_mut(self.0)?)
    }
}

/// Non-cloneable authority for one backend ledger and preparation incarnation.
/// All mutable transaction state remains in PagedKvOwnership.
///
/// ```compile_fail
/// use ferrule_model::decoder::PagedKvTransactionHandle;
/// use ferrule_common::execution::ExecutionTransactionId;
/// let forged = PagedKvTransactionHandle {
///     transaction: ExecutionTransactionId::new(1).unwrap(),
/// };
/// ```
#[derive(Debug)]
pub struct PagedKvTransactionHandle {
    transaction: ferrule_common::execution::ExecutionTransactionId,
    backend_identity: Arc<()>,
    nonce: u64,
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
    // Identity belongs to this ledger, never to the possibly shared physical pool.
    identity: Arc<()>,
    next_handle_nonce: u64,
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
            identity: Arc::new(()),
            next_handle_nonce: 1,
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
        if !Arc::ptr_eq(&self.identity, &transaction.backend_identity) {
            return Err(kv_error(
                "paged KV transaction belongs to another backend ledger",
            ));
        }
        let pending = self.ownership()?.transaction(transaction.transaction)?;
        if transaction.nonce == 0 || pending.handle_nonce != transaction.nonce {
            return Err(kv_error(
                "paged KV transaction handle has a stale preparation nonce",
            ));
        }
        Ok(())
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
        // A physical pool may report typed Unknown without exposing a separate
        // poisoned flag. A pending record with no physical payload is the
        // backend's durable witness that the previous prepare still owns
        // custody; never validate or redispatch a later request over it.
        if self.ownership.as_ref().is_some_and(|ownership| {
            ownership
                .pending
                .values()
                .any(|pending| pending.payload.is_none())
        }) {
            return Err(KvPrepareQuiescenceUnknown::wrap(
                request.transaction,
                kv_error("paged KV backend retains unresolved prepare custody"),
            ));
        }
        if self.pool.prepare_failure_retains_custody() {
            return Err(KvPrepareQuiescenceUnknown::wrap(
                request.transaction,
                kv_error("physical KV pool retains unresolved prepare custody"),
            ));
        }
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
        let nonce = self.next_handle_nonce;
        let next_nonce = nonce
            .checked_add(1)
            .ok_or_else(|| kv_error("paged KV handle nonce exhausted"))?;
        self.ownership_mut()?.prepare(request, None)?;
        self.next_handle_nonce = next_nonce;
        self.ownership_mut()?
            .transaction_mut(request.transaction)?
            .handle_nonce = nonce;
        match self.pool.prepare(request) {
            Ok(physical) => {
                *self.ownership_mut()?.payload_mut(request.transaction)? = Some(physical);
                Ok(PagedKvTransactionHandle {
                    transaction: request.transaction,
                    backend_identity: self.identity.clone(),
                    nonce,
                })
            }
            Err(error) => {
                if self.pool.prepare_failure_retains_custody()
                    || KvPrepareQuiescenceUnknown::from_error(&error).is_some()
                {
                    // The physical pool retained its entry but issued no public
                    // handle. Keep this exact logical record/pins, too; no
                    // cleanup or retirement ACK has occurred.
                    return Err(KvPrepareQuiescenceUnknown::wrap(request.transaction, error));
                }
                self.ownership_mut()?.rollback(request.transaction)?;
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
        self.pool
            .preflight_commit(self.ownership()?.transaction(id)?)?;
        let ready = SingleKvCommitReady(id);
        let progress = {
            let (pool, ownership) = (&mut self.pool, self.ownership.as_mut());
            ready.install(
                pool,
                ownership.ok_or_else(|| kv_error("paged KV capacity is not configured"))?,
            )?
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
        let mut capacity = self
            .ownership
            .as_ref()
            .map_or_else(DecoderKvCapacity::default, PagedKvOwnership::capacity);
        if let Some(free_slots) = self.pool.physical_free_slots() {
            capacity.free_pages = free_slots;
        }
        capacity
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
pub struct TypedCpuPagedKvPool<S> {
    state: std::marker::PhantomData<fn() -> S>,
    inner: cpu::CpuPagedKvPool,
    identity: std::rc::Rc<()>,
}

impl<S> Clone for TypedCpuPagedKvPool<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            identity: self.identity.clone(),
            state: std::marker::PhantomData,
        }
    }
}

/// CPU physical reservation and mapping/shape witness, not a second ownership
/// ledger. The backend's transaction handle cannot escape this adapter.
#[derive(Debug)]
pub struct CpuPagedKvTransaction {
    inner: Option<cpu::CpuPagedKvTransaction>,
    identity: std::rc::Rc<()>,
    id: ferrule_common::execution::ExecutionTransactionId,
    slots: BTreeMap<KvPageId, Option<usize>>,
    new_pages: BTreeSet<KvPageId>,
    writable_pages: BTreeSet<KvPageId>,
    cows: Box<[ferrule_common::execution::KvCowReplacement]>,
    capacity: usize,
    planes: Vec<KvPlaneDescriptor>,
    batch: Option<PackedDecoderBatch>,
    entered: bool,
    checked_slots: Option<BTreeMap<KvPageId, usize>>,
}
impl CpuPagedKvTransaction {
    pub const fn id(&self) -> ferrule_common::execution::ExecutionTransactionId {
        self.id
    }
    fn handle_mut(&mut self) -> Result<&mut cpu::CpuPagedKvTransaction> {
        self.inner
            .as_mut()
            .ok_or_else(|| kv_error("CPU physical KV handle absent"))
    }
}

pub type CpuPagedKvPool = TypedCpuPagedKvPool<super::GenericDecoderSequenceState>;

impl<S> TypedCpuPagedKvPool<S> {
    pub fn new(
        planes: impl IntoIterator<Item = KvPlaneDescriptor>,
        page_size: usize,
        capacity: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: cpu::CpuPagedKvPool::new(planes, page_size, capacity)?,
            state: std::marker::PhantomData,
            identity: std::rc::Rc::new(()),
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

impl<S> DecoderKvPageView for TypedCpuPagedKvPool<S> {
    fn page_status(&self, page: KvPageId) -> DecoderKvPageStatus {
        map_page_status(self.inner.page_status(page))
    }
}

impl<S: super::DecoderSequence> PhysicalKvPool for TypedCpuPagedKvPool<S> {
    type SequenceState = S;
    type Transaction = CpuPagedKvTransaction;
    type KvView = CpuKvView;

    fn configured_capacity(&self) -> usize {
        self.inner.capacity().physical_pages
    }

    fn configure_capacity(&mut self, max_pages: usize) -> Result<()> {
        self.inner.configure_capacity(max_pages)
    }

    fn prepare(&mut self, request: DecoderKvPrepare<'_>) -> Result<Self::Transaction> {
        let protected = request
            .protected_pages
            .iter()
            .copied()
            .chain(request.new_pages.iter().copied())
            .chain(request.writable_pages.iter().copied())
            .chain(
                request
                    .cow_replacements
                    .iter()
                    .flat_map(|cow| [cow.source, cow.replacement]),
            )
            .collect::<BTreeSet<_>>();
        let provisional = request
            .new_pages
            .iter()
            .copied()
            .chain(request.cow_replacements.iter().map(|cow| cow.replacement))
            .collect::<BTreeSet<_>>();
        let slots = protected
            .iter()
            .map(|page| (*page, self.inner.physical_slot(*page)))
            .collect::<BTreeMap<_, _>>();
        if slots
            .iter()
            .any(|(page, slot)| !provisional.contains(page) && slot.is_none())
        {
            return Err(kv_error("CPU protected/COW source page has no mapping"));
        }
        let protected = protected.into_iter().collect::<Vec<_>>();
        let inner = self.inner.prepare(cpu::PagedKvPrepare {
            transaction: request.transaction,
            new_pages: request.new_pages,
            writable_pages: request.writable_pages,
            cow_replacements: request.cow_replacements,
            protected_pages: &protected,
        })?;
        Ok(CpuPagedKvTransaction {
            inner: Some(inner),
            identity: self.identity.clone(),
            id: request.transaction,
            slots,
            new_pages: request.new_pages.iter().copied().collect(),
            writable_pages: request.writable_pages.iter().copied().collect(),
            cows: request.cow_replacements.into(),
            capacity: self.configured_capacity(),
            planes: self.planes(),
            batch: None,
            entered: false,
            checked_slots: None,
        })
    }

    fn enter(
        &mut self,
        transaction: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
        _states: &mut [Self::SequenceState],
    ) -> Result<()> {
        if !std::rc::Rc::ptr_eq(&self.identity, &transaction.identity) {
            return Err(kv_error("CPU transaction belongs to another pool"));
        }
        if batch.page_size() != self.page_size()
            || batch.new_pages().iter().copied().collect::<BTreeSet<_>>() != transaction.new_pages
            || batch
                .writable_pages()
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                != transaction.writable_pages
            || batch.cow_replacements() != transaction.cows.as_ref()
            || batch
                .sequences()
                .iter()
                .flat_map(|sequence| sequence.block_table())
                .any(|page| !transaction.slots.contains_key(page))
        {
            return Err(kv_error(
                "CPU batch differs from reserved mapping/COW shape",
            ));
        }
        self.inner
            .enter(transaction.handle_mut()?, backend_batch(batch))?;
        transaction.entered = true;
        transaction.batch = Some(batch.clone());
        transaction.checked_slots = None;
        Ok(())
    }

    fn active_view(
        &mut self,
        transaction: &mut Self::Transaction,
        _batch: &PackedDecoderBatch,
    ) -> Result<Self::KvView> {
        Ok(CpuKvView {
            inner: self.inner.active_view(transaction.handle_mut()?)?,
        })
    }

    fn leave(&mut self, transaction: &mut Self::Transaction) -> Result<()> {
        let view = self.inner.active_view(transaction.handle_mut()?)?;
        let mut checked = BTreeMap::new();
        for page in transaction.slots.keys() {
            let slot = view
                .physical_slot(*page)
                .ok_or_else(|| kv_error("CPU staged mapping absent"))?;
            if slot >= transaction.capacity {
                return Err(kv_error("CPU staged slot exceeds capacity"));
            }
            for (plane, descriptor) in transaction.planes.iter().enumerate() {
                let elements = descriptor
                    .checked_page_elements(self.page_size())
                    .ok_or_else(|| kv_error("CPU plane shape overflow"))?;
                let actual = match descriptor.element_type {
                    KvElementType::F32 => view.with_f32_page(*page, plane, |v| v.len())?,
                    KvElementType::Bf16 => view.with_bf16_page(*page, plane, |v| v.len())?,
                };
                if actual != elements {
                    return Err(kv_error("CPU staging shape changed"));
                }
            }
            checked.insert(*page, slot);
        }
        if checked.values().copied().collect::<BTreeSet<_>>().len() != checked.len() {
            return Err(kv_error("CPU staged pages alias a slot"));
        }
        self.inner.leave(transaction.handle_mut()?)?;
        transaction.entered = false;
        transaction.checked_slots = Some(checked);
        Ok(())
    }
    fn preflight_commit(
        &self,
        pending: &PagedKvTransaction<Option<Self::Transaction>>,
    ) -> Result<()> {
        let p = pending
            .payload()
            .as_ref()
            .ok_or_else(|| kv_error("CPU reservation absent"))?;
        if !std::rc::Rc::ptr_eq(&self.identity, &p.identity)
            || p.inner.is_none()
            || p.entered
            || pending.entered()
            || self.configured_capacity() != p.capacity
            || self.planes() != p.planes
            || pending.batch() != p.batch.as_ref()
            || pending.new_pages != p.new_pages
            || pending.cow_replacements.as_ref() != p.cows.as_ref()
        {
            return Err(kv_error(
                "CPU preflight identity/capacity/shape/fence mismatch",
            ));
        }
        let mut writes = p.writable_pages.clone();
        writes.extend(p.new_pages.iter().copied());
        writes.extend(p.cows.iter().map(|cow| cow.replacement));
        if pending.writable_pages != writes
            || pending
                .protected_pages
                .iter()
                .copied()
                .ne(p.slots.keys().copied())
        {
            return Err(kv_error(
                "CPU preflight protection/mutation mapping changed",
            ));
        }
        if p.batch.is_some() && p.checked_slots.is_none() {
            return Err(kv_error(
                "CPU staging lacks a quiescent shape/mapping witness",
            ));
        }
        if let Some(checked) = &p.checked_slots {
            if checked.keys().ne(p.slots.keys())
                || checked.values().any(|slot| *slot >= p.capacity)
                || checked.values().copied().collect::<BTreeSet<_>>().len() != checked.len()
            {
                return Err(kv_error("CPU staged mapping witness changed"));
            }
        }
        for (page, slot) in &p.slots {
            if self.inner.physical_slot(*page) != *slot
                || slot.is_some_and(|slot| slot >= p.capacity)
                || (slot.is_none() && self.inner.page_status(*page) != BackendKvPageStatus::Vacant)
            {
                return Err(kv_error("CPU page/COW source changed physical mapping"));
            }
        }
        let capacity = self.inner.capacity();
        if capacity.active_transactions == 0
            || capacity.resident_pages + capacity.free_pages + p.new_pages.len() + p.cows.len()
                > p.capacity
        {
            return Err(kv_error("CPU reservation exceeds capacity"));
        }
        // CPU copies/writes/leave are synchronous. Staging slots, dtype and length
        // are immutable in backend storage, and all views reject access after
        // leave. Also prove the exact physical handle is still active, not an
        // already-installed lease with an unchanged writable-page mapping.
        self.inner
            .preflight_commit(p.inner.as_ref().expect("checked physical handle"))
    }
    fn commit(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        let pending = transaction
            .as_mut()
            .ok_or_else(|| kv_error("CPU commit token absent"))?;
        if !std::rc::Rc::ptr_eq(&self.identity, &pending.identity) {
            return Err(kv_error("CPU commit belongs to another pool"));
        }
        let progress = self
            .inner
            .commit(&mut pending.inner)
            .map(map_end_progress)?;
        if progress == KvEndProgress::Complete {
            transaction.take();
        }
        Ok(progress)
    }
    fn abort(&mut self, transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        let pending = transaction
            .as_mut()
            .ok_or_else(|| kv_error("CPU abort token absent"))?;
        if !std::rc::Rc::ptr_eq(&self.identity, &pending.identity) {
            return Err(kv_error("CPU abort belongs to another pool"));
        }
        let progress = self.inner.abort(&mut pending.inner).map(map_end_progress)?;
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

/// One prepared protocol over the existing ledger for every opted-in pool.
impl<P: PhysicalKvPreparedPool> DecoderKvCommitBackend for PagedKvBackend<P> {
    fn preflight_commit_ready(
        &self,
        transaction: &Self::Transaction,
        binding: &KvCommitBinding,
        rank: ferrule_common::ParallelRankId,
        sources: &[Self::SequenceState],
    ) -> Result<()> {
        self.validate_handle(transaction)?;
        if transaction.id() != binding.transaction() || !binding.participants().contains(rank) {
            return Err(kv_error("KV binding/rank mismatch"));
        }
        let pending = self.ownership()?.transaction(transaction.id())?;
        if pending.entered || pending.published || pending.custody.is_empty() {
            return Err(kv_error(
                "readiness needs executed, quiescent sequence custody",
            ));
        }
        let batch = pending
            .batch()
            .ok_or_else(|| kv_error("ready batch absent"))?;
        for source in sources {
            source.core().begin_step()?;
        }
        validate_sequence_custody(pending, batch, sources)?;
        self.pool.preflight_commit(pending)?;
        self.pool.preflight_prepared(
            pending
                .payload()
                .as_ref()
                .ok_or_else(|| kv_error("physical reservation absent"))?,
        )
    }
    fn commit_batch(&self, transaction: &Self::Transaction) -> Result<&PackedDecoderBatch> {
        self.validate_handle(transaction)?;
        self.ownership()?
            .transaction(transaction.id())?
            .batch()
            .ok_or_else(|| kv_error("ready batch absent"))
    }
    fn install_commit(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        self.validate_handle(transaction)?;
        self.pool
            .preflight_commit(self.ownership()?.transaction(transaction.id())?)?;
        let physical = self
            .ownership
            .as_mut()
            .expect("validated ledger")
            .payload_mut(transaction.id())?
            .as_mut()
            .ok_or_else(|| kv_error("prepared physical custody absent"))?;
        self.pool.install_prepared(physical, generation)
    }
    fn poll_install_ack(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        self.validate_handle(transaction)?;
        let physical = self
            .ownership
            .as_mut()
            .expect("validated ledger")
            .payload_mut(transaction.id())?
            .as_mut()
            .ok_or_else(|| kv_error("prepared physical custody absent"))?;
        self.pool.poll_install_prepared(physical, generation)
    }
    fn abort_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        self.validate_handle(transaction)?;
        let physical = self
            .ownership
            .as_mut()
            .expect("validated ledger")
            .payload_mut(transaction.id())?
            .as_mut()
            .ok_or_else(|| kv_error("prepared physical custody absent"))?;
        self.pool.abort_prepared(physical, generation)
    }
    fn publish_committed(&mut self, transaction: &Self::Transaction) {
        self.validate_handle(transaction)
            .expect("exact KV authority before publication");
        let (pool, ownership) = (&mut self.pool, self.ownership.as_mut());
        let ledger = ownership.expect("prepared ledger");
        let pending = ledger
            .pending
            .get_mut(&transaction.id())
            .expect("prepared custody");
        assert!(!pending.entered && !pending.published);
        let physical = pending.payload.as_ref().expect("prepared physical custody");
        pool.publish_prepared(physical)
            .expect("exact installed lease remains pinned until finish");
        ledger.resident.extend(pending.new_pages.iter().copied());
        ledger
            .resident
            .extend(pending.cow_replacements.iter().map(|cow| cow.replacement));
        pending.published = true;
    }
    fn preflight_retirement(
        &self,
        transaction: &Self::Transaction,
        pages: &[KvPageId],
    ) -> Result<()> {
        self.validate_handle(transaction)?;
        let page_set = unique_page_set(pages, "retirement")?;
        let ledger = self.ownership()?;
        let pending = ledger.transaction(transaction.id())?;
        if !pending.published {
            return Err(kv_error("retirement before acknowledged publication"));
        }
        if page_set
            .iter()
            .any(|page| !ledger.resident.contains(page) && !ledger.preempted.contains(page))
        {
            return Err(kv_error("unknown retirement page"));
        }
        for (id, other) in &ledger.pending {
            if *id != transaction.id() && !page_set.is_disjoint(&other.protected_pages) {
                return Err(kv_error("retirement page pinned by another transaction"));
            }
        }
        let physical = pending
            .payload
            .as_ref()
            .ok_or_else(|| kv_error("prepared physical custody absent"))?;
        self.pool.preflight_retirement(physical, pages)
    }
    fn retire_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
        pages: &[KvPageId],
    ) -> Result<KvEndProgress> {
        self.preflight_retirement(transaction, pages)?;
        let (pool, ownership) = (&mut self.pool, self.ownership.as_mut());
        let physical = ownership
            .ok_or_else(|| kv_error("prepared ledger absent"))?
            .payload_mut(transaction.id())?
            .as_mut()
            .ok_or_else(|| kv_error("prepared physical custody absent"))?;
        // The physical lease journals the exact set. Keep ledger capacity charged
        // while detached slots remain quarantined awaiting the other owners' ACKs.
        pool.retire_prepared(physical, generation, pages)
    }
    fn finish_prepared(&mut self, transaction: Self::Transaction) {
        self.validate_handle(&transaction)
            .expect("exact KV authority before finish");
        let id = transaction.id();
        let (pool, ownership) = (&mut self.pool, self.ownership.as_mut());
        let ledger = ownership.expect("prepared ledger");
        let pending = ledger.pending.get_mut(&id).expect("prepared custody");
        assert!(!pending.entered);
        let physical = pending
            .payload_mut()
            .take()
            .expect("prepared custody retains its physical handle");
        let retired = pool
            .finish_prepared(physical)
            .expect("exact lease finished after all cleanup/retirement ACKs");
        for page in retired {
            ledger.resident.remove(&page);
            ledger.preempted.remove(&page);
        }
        ledger.pending.remove(&id).expect("prepared custody");
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

#[cfg(test)]
mod distributed_kv_tests {
    use super::*;
    use crate::decoder::{DecoderKvPageSnapshot, DecoderKvSequenceCustody};
    use ferrule_common::execution::{
        ExecutionBatch, ExecutionCapabilities, ExecutionSequence, ExecutionTransactionId,
        ForwardMode, ForwardPhase, KvBindingMode, KvBlockId, KvCowReplacement, KvReservationView,
        KvWriteSlot, LogitsRequest, LogitsRowPolicy, StateSlot,
    };
    use ferrule_common::{
        ParallelRankId, ParallelTopologyId, ParallelismPlan, ParticipantSet,
        ValidatedParallelTopology,
    };
    use std::cell::Cell;
    use std::num::NonZeroU32;
    use std::rc::Rc;

    type State = DecoderSequenceState<(), ()>;
    fn tx(n: u64) -> ExecutionTransactionId {
        ExecutionTransactionId::new(n).unwrap()
    }
    fn rank(n: u32) -> ParallelRankId {
        ParallelRankId::new(n)
    }
    fn scope(epoch: u32, generation: u64, id: u64, ranks: &[u32]) -> KvCommitBinding {
        let topology = ValidatedParallelTopology::new(
            ParallelTopologyId::new(epoch),
            2,
            rank(0),
            ParallelismPlan::validated(1, 2, 1, 1, 1, 1).unwrap(),
        )
        .unwrap();
        let participants = ParticipantSet::new(&topology, ranks.iter().copied().map(rank)).unwrap();
        KvCommitBinding::new(tx(id), topology.topology_id(), participants, generation).unwrap()
    }
    fn binding() -> KvCommitBinding {
        scope(1, 1, 2, &[0, 1])
    }
    #[derive(Default)]
    struct Faults {
        preflight: Cell<bool>,
        install_pending: Cell<bool>,
        install_unknown: Cell<bool>,
        abort_unknown: Cell<bool>,
        retirement_unknown: Cell<bool>,
        installs: Cell<usize>,
        polls: Cell<usize>,
        aborts: Cell<usize>,
        releases: Cell<usize>,
        retirement_attempts: Cell<usize>,
        retired: Cell<bool>,
    }
    struct FaultBackend {
        inner: CpuPagedKvBackend,
        faults: Rc<Faults>,
    }
    impl DecoderKvBackend for FaultBackend {
        type SequenceState = State;
        type Transaction = PagedKvTransactionHandle;
        type KvView = CpuKvView;
        fn configure_capacity(&mut self, n: usize) -> Result<()> {
            self.inner.configure_capacity(n)
        }
        fn prepare(&mut self, r: DecoderKvPrepare<'_>) -> Result<Self::Transaction> {
            self.inner.prepare(r)
        }
        fn enter(
            &mut self,
            t: &mut Self::Transaction,
            b: &PackedDecoderBatch,
            s: &mut [State],
        ) -> Result<()> {
            self.inner.enter(t, b, s)
        }
        fn active_view(&mut self, t: &mut Self::Transaction) -> Result<CpuKvView> {
            self.inner.active_view(t)
        }
        fn leave(&mut self, t: &mut Self::Transaction) -> Result<()> {
            self.inner.leave(t)
        }
        fn commit(&mut self, t: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
            DecoderKvBackend::commit(&mut self.inner, t)
        }
        fn rollback(&mut self, t: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
            self.inner.rollback(t)
        }
        fn release(&mut self, p: &[KvPageId]) -> Result<()> {
            self.inner.release(p)
        }
        fn preempt(&mut self, p: &[KvPageId]) -> Result<()> {
            self.inner.preempt(p)
        }
        fn restore(&mut self, p: &[KvPageId]) -> Result<()> {
            self.inner.restore(p)
        }
        fn capacity(&self) -> DecoderKvCapacity {
            self.inner.capacity()
        }
        fn page_status(&self, p: KvPageId) -> DecoderKvPageStatus {
            DecoderKvBackend::page_status(&self.inner, p)
        }
        fn shutdown(&mut self) -> Result<()> {
            self.inner.shutdown()
        }
    }
    impl DecoderKvCommitBackend for FaultBackend {
        fn preflight_commit_ready(
            &self,
            t: &Self::Transaction,
            b: &KvCommitBinding,
            r: ParallelRankId,
            s: &[State],
        ) -> Result<()> {
            if self.faults.preflight.get() {
                return Err(kv_error("injected owner preflight/fence failure"));
            }
            self.inner.preflight_commit_ready(t, b, r, s)
        }
        fn commit_batch(&self, t: &Self::Transaction) -> Result<&PackedDecoderBatch> {
            self.inner.commit_batch(t)
        }
        fn install_commit(&mut self, t: &mut Self::Transaction, g: u64) -> Result<KvEndProgress> {
            self.faults.installs.set(self.faults.installs.get() + 1);
            let progress = self.inner.install_commit(t, g)?;
            if self.faults.install_unknown.get() {
                return Err(kv_error("lost install ACK"));
            }
            if self.faults.install_pending.get() {
                return Ok(KvEndProgress::Pending);
            }
            Ok(progress)
        }
        fn poll_install_ack(
            &mut self,
            _t: &mut Self::Transaction,
            _g: u64,
        ) -> Result<KvEndProgress> {
            self.faults.polls.set(self.faults.polls.get() + 1);
            if self.faults.install_unknown.get() {
                return Err(kv_error("owner remains unknown"));
            }
            if self.faults.install_pending.get() {
                return Ok(KvEndProgress::Pending);
            }
            // The fault harness knows the synchronous inner install completed;
            // this is an ACK delivery simulation, never a second install call.
            Ok(KvEndProgress::Complete)
        }
        fn abort_prepared(&mut self, t: &mut Self::Transaction, g: u64) -> Result<KvEndProgress> {
            self.faults.aborts.set(self.faults.aborts.get() + 1);
            if self.faults.abort_unknown.get() {
                return Err(kv_error("cleanup owner unknown"));
            }
            self.inner.abort_prepared(t, g)
        }
        fn publish_committed(&mut self, t: &Self::Transaction) {
            self.inner.publish_committed(t)
        }
        fn preflight_retirement(&self, t: &Self::Transaction, p: &[KvPageId]) -> Result<()> {
            self.inner.preflight_retirement(t, p)
        }
        fn retire_prepared(
            &mut self,
            t: &mut Self::Transaction,
            g: u64,
            p: &[KvPageId],
        ) -> Result<KvEndProgress> {
            self.faults
                .retirement_attempts
                .set(self.faults.retirement_attempts.get() + 1);
            self.inner.retire_prepared(t, g, p)?;
            if !self.faults.retired.replace(true) {
                self.faults.releases.set(self.faults.releases.get() + 1);
            }
            if self.faults.retirement_unknown.get() {
                return Err(kv_error("release fence ACK unknown"));
            }
            Ok(KvEndProgress::Complete)
        }
        fn finish_prepared(&mut self, t: Self::Transaction) {
            self.inner.finish_prepared(t)
        }
    }
    struct Fixture {
        backend: FaultBackend,
        handle: Option<PagedKvTransactionHandle>,
        states: Vec<State>,
    }
    impl Fixture {
        fn new() -> Self {
            let strategy = StandardGqaPlanes::new(1, 1, 1, 4, 16, KvElementType::F32).unwrap();
            let pool = CpuPagedKvPool::from_strategy(&strategy, 4).unwrap();
            Self {
                backend: FaultBackend {
                    inner: PagedKvBackend::new(pool),
                    faults: Rc::new(Faults::default()),
                },
                handle: None,
                states: vec![State::new((), ())],
            }
        }
        fn execute(&mut self, id: u64, page: u32, cow: Option<KvCowReplacement>, value: f32) {
            let context = self.states[0].core().position() as u32;
            let new_pages = if cow.is_some() {
                vec![]
            } else {
                vec![KvPageId(page)]
            };
            let batch = ExecutionBatch::new(
                ForwardMode::Prefill,
                vec![17],
                vec![context],
                vec![Some(KvWriteSlot::new(page * 4 + context))],
                vec![LogitsRequest::None],
                vec![ExecutionSequence::new(
                    StateSlot::new(0),
                    ForwardPhase::Prefill,
                    0..1,
                    context,
                    context + 1,
                    0..1,
                )],
                vec![KvBlockId::new(page)],
            );
            let reservations = [KvReservationView {
                state_slot: StateSlot::new(3),
                execution_state_slot: StateSlot::new(0),
                positions: context as usize..context as usize + 1,
                newly_allocated: new_pages,
                generation: 7,
                execution_generation: self.states[0].core().generation(),
                cow_replacement: cow,
            }];
            let capabilities = ExecutionCapabilities {
                max_batch_tokens: 8,
                max_sequences: 2,
                max_prefill_query_tokens_per_sequence: 4,
                max_decode_query_tokens_per_sequence: 4,
                max_top_k: NonZeroU32::new(4),
                supports_prefill: true,
                supports_decode: true,
                supports_mixed: true,
                full_logits_width: NonZeroU32::new(4),
                kv_binding_mode: KvBindingMode::Paged,
                logits_row_policy: LogitsRowPolicy::Any,
            };
            let packed = PackedDecoderBatch::lower(
                &batch,
                &reservations,
                &self.states,
                &capabilities,
                4,
                &|page| self.backend.page_status(page),
            )
            .unwrap();
            let custody = packed
                .sequences()
                .iter()
                .map(|s| DecoderKvSequenceCustody {
                    source_index: s.state_index(),
                    topology_id: s.topology_id(),
                    page_state_slot: s.page_state_slot(),
                    page_generation: s.page_generation(),
                    execution_generation: s.execution_generation(),
                    context_len: s.context_len(),
                    query_len: s.query_len(),
                })
                .collect::<Vec<_>>();
            let statuses = packed
                .protected_pages()
                .iter()
                .copied()
                .map(|page| DecoderKvPageSnapshot {
                    page,
                    status: self.backend.page_status(page),
                })
                .collect::<Vec<_>>();
            let mut handle = self
                .backend
                .prepare(DecoderKvPrepare {
                    transaction: tx(id),
                    sequences: &custody,
                    new_pages: packed.new_pages(),
                    writable_pages: packed.writable_pages(),
                    cow_replacements: packed.cow_replacements(),
                    protected_pages: packed.protected_pages(),
                    capacity: self.backend.capacity(),
                    page_statuses: &statuses,
                })
                .unwrap();
            self.backend
                .enter(&mut handle, &packed, &mut self.states.clone())
                .unwrap();
            let mut view = self.backend.active_view(&mut handle).unwrap();
            if let Some(cow) = cow {
                assert_eq!(view.with_f32_page(cow.source, 0, |v| v[0]).unwrap(), 3.0);
                assert_eq!(
                    view.with_f32_page(cow.replacement, 0, |v| v[0]).unwrap(),
                    3.0
                );
            }
            view.with_f32_page_mut(KvPageId(page), 0, |v| v[context as usize] = value)
                .unwrap();
            if let Some(cow) = cow {
                assert_eq!(
                    view.with_f32_page(cow.source, 0, |v| v[context as usize])
                        .unwrap(),
                    0.0
                );
            }
            drop(view);
            self.backend.leave(&mut handle).unwrap();
            self.handle = Some(handle);
        }
        fn fresh() -> Self {
            let mut s = Self::new();
            s.execute(2, 10, None, 3.0);
            s
        }
        fn cow() -> Self {
            let mut s = Self::new();
            s.execute(1, 10, None, 3.0);
            assert_eq!(
                s.backend.commit(&mut s.handle).unwrap(),
                KvEndProgress::Complete
            );
            let step = s.states[0].core().begin_step().unwrap();
            s.states[0].core_mut().commit_step(step, 1).unwrap();
            s.execute(
                2,
                11,
                Some(KvCowReplacement {
                    source: KvPageId(10),
                    replacement: KvPageId(11),
                    logical_page: 0,
                }),
                9.0,
            );
            s
        }
    }
    fn owners<'a>(a: &'a mut Fixture, b: &'a mut Fixture) -> Vec<KvCommitOwner<'a, FaultBackend>> {
        vec![
            KvCommitOwner::new(rank(0), &mut a.backend, &mut a.handle, &a.states),
            KvCommitOwner::new(rank(1), &mut b.backend, &mut b.handle, &b.states),
        ]
    }
    struct Reservation(Rc<Cell<usize>>);
    impl Drop for Reservation {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    fn reservation() -> (Reservation, Rc<Cell<usize>>) {
        let count = Rc::new(Cell::new(0));
        (Reservation(count.clone()), count)
    }

    fn assert_foreign_ledger<T>(result: Result<T>) {
        match result {
            Err(error) => assert!(
                error.to_string().contains("another backend ledger"),
                "{error}"
            ),
            Ok(_) => panic!("foreign KV authority was accepted"),
        }
    }

    #[test]
    fn cpu_same_id_foreign_handles_reject_every_fallible_transaction_entry() {
        let (mut a, mut b) = (Fixture::fresh(), Fixture::fresh());
        let batch = b
            .backend
            .inner
            .commit_batch(b.handle.as_ref().unwrap())
            .unwrap()
            .clone();
        let mut working = b.states.clone();
        let before_a = a.backend.capacity();
        let before_b = b.backend.capacity();
        let free_a = a.backend.inner.pool().free_slot_count();
        let free_b = b.backend.inner.pool().free_slot_count();
        let foreign = a.handle.as_mut().unwrap();
        let backend = &mut b.backend.inner;
        assert_eq!(foreign.id(), b.handle.as_ref().unwrap().id());
        assert_foreign_ledger(backend.enter(foreign, &batch, &mut working));
        assert_foreign_ledger(backend.preflight_commit_ready(
            foreign,
            &binding(),
            rank(1),
            &b.states,
        ));
        assert_foreign_ledger(backend.commit_batch(foreign));
        assert_foreign_ledger(backend.install_commit(foreign, 1));
        assert_foreign_ledger(backend.poll_install_ack(foreign, 1));
        assert_foreign_ledger(backend.abort_prepared(foreign, 1));
        assert_foreign_ledger(backend.preflight_retirement(foreign, &[]));
        assert_foreign_ledger(backend.retire_prepared(foreign, 1, &[]));
        assert_foreign_ledger(backend.retain_provisional(
            foreign,
            &b.states,
            &mut working,
            &[1],
            &[1],
        ));
        backend
            .enter(b.handle.as_mut().unwrap(), &batch, &mut working)
            .unwrap();
        assert_foreign_ledger(backend.active_view(foreign));
        assert_foreign_ledger(backend.leave(foreign));
        let view = backend.active_view(b.handle.as_mut().unwrap()).unwrap();
        assert_eq!(view.with_f32_page(KvPageId(10), 0, |v| v[0]).unwrap(), 3.0);
        backend.leave(b.handle.as_mut().unwrap()).unwrap();
        assert_foreign_ledger(backend.commit(&mut a.handle));
        assert_foreign_ledger(backend.rollback(&mut a.handle));
        assert!(a.handle.is_some() && b.handle.is_some());
        assert_eq!(a.backend.capacity(), before_a);
        assert_eq!(b.backend.capacity(), before_b);
        assert_eq!(a.backend.inner.pool().free_slot_count(), free_a);
        assert_eq!(b.backend.inner.pool().free_slot_count(), free_b);
        assert_eq!(b.backend.inner.pool().physical_slot(KvPageId(10)), None);
        a.backend.rollback(&mut a.handle).unwrap();
        b.backend.rollback(&mut b.handle).unwrap();
    }

    #[test]
    fn cpu_swapped_same_id_handles_cannot_seal_foreign_owners() {
        let (mut a, mut b) = (Fixture::fresh(), Fixture::fresh());
        std::mem::swap(&mut a.handle, &mut b.handle);
        let error =
            PreparedKvCommit::prepare_commit_ready(binding(), (), cpu_owners(&mut a, &mut b))
                .err()
                .expect("foreign handles must not seal");
        let (error, (), inputs) = error.into_parts();
        assert!(error.to_string().contains("another backend ledger"));
        drop(inputs);
        assert!(a.handle.is_some() && b.handle.is_some());
        assert_eq!(a.backend.capacity().active_transactions, 1);
        assert_eq!(b.backend.capacity().active_transactions, 1);
        assert_eq!(a.backend.inner.pool().physical_slot(KvPageId(10)), None);
        assert_eq!(b.backend.inner.pool().physical_slot(KvPageId(10)), None);
        std::mem::swap(&mut a.handle, &mut b.handle);
        let mut ready =
            PreparedKvCommit::prepare_commit_ready(binding(), (), cpu_owners(&mut a, &mut b))
                .unwrap();
        ready.abort_prepared(&binding(), rank(0)).unwrap();
        ready.abort_prepared(&binding(), rank(1)).unwrap();
        ready.abort_logical(&binding(), drop).unwrap();
    }

    #[test]
    fn cpu_infallible_publish_and_finish_reject_foreign_authority_before_mutation() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let (mut a, mut b) = (Fixture::fresh(), Fixture::fresh());
        for fixture in [&mut a, &mut b] {
            fixture
                .backend
                .inner
                .install_commit(fixture.handle.as_mut().unwrap(), 1)
                .unwrap();
        }
        let foreign = a.handle.as_ref().unwrap();
        assert!(
            catch_unwind(AssertUnwindSafe(|| b
                .backend
                .inner
                .publish_committed(foreign)))
            .is_err()
        );
        assert_eq!(
            b.backend.page_status(KvPageId(10)),
            DecoderKvPageStatus::Vacant
        );
        assert!(
            !b.backend
                .inner
                .ownership()
                .unwrap()
                .transaction(tx(2))
                .unwrap()
                .published
        );
        assert_eq!(b.backend.inner.pool().active_transaction_count(), 1);
        for fixture in [&mut a, &mut b] {
            fixture
                .backend
                .inner
                .publish_committed(fixture.handle.as_ref().unwrap());
        }
        assert_foreign_ledger(
            b.backend
                .inner
                .preflight_retirement(a.handle.as_ref().unwrap(), &[KvPageId(10)]),
        );
        assert_foreign_ledger(b.backend.inner.retire_prepared(
            a.handle.as_mut().unwrap(),
            1,
            &[KvPageId(10)],
        ));
        assert!(b.backend.inner.pool().physical_slot(KvPageId(10)).is_some());
        for fixture in [&mut a, &mut b] {
            fixture
                .backend
                .inner
                .retire_prepared(fixture.handle.as_mut().unwrap(), 1, &[KvPageId(10)])
                .unwrap();
        }
        let before = b.backend.capacity();
        let foreign = a.handle.take().unwrap();
        assert!(
            catch_unwind(AssertUnwindSafe(|| b
                .backend
                .inner
                .finish_prepared(foreign)))
            .is_err()
        );
        assert_eq!(b.backend.capacity(), before);
        assert_eq!(b.backend.inner.pool().active_transaction_count(), 1);
        assert_eq!(b.backend.inner.pool().free_slot_count(), 3);
        assert_eq!(a.backend.inner.pool().active_transaction_count(), 1);
        b.backend.inner.finish_prepared(b.handle.take().unwrap());
        assert_eq!(b.backend.capacity().active_transactions, 0);
        assert_eq!(b.backend.inner.pool().free_slot_count(), 4);
    }

    fn replay_private_handle(handle: &PagedKvTransactionHandle) -> PagedKvTransactionHandle {
        // Simulate stale wire authority below the non-cloneable public boundary.
        PagedKvTransactionHandle {
            transaction: handle.transaction,
            backend_identity: handle.backend_identity.clone(),
            nonce: handle.nonce,
        }
    }

    #[test]
    fn cpu_handle_nonce_survives_same_id_reuse_and_capacity_reconfiguration() {
        let mut fixture = Fixture::fresh();
        let mut stale = Some(replay_private_handle(fixture.handle.as_ref().unwrap()));
        fixture.backend.rollback(&mut fixture.handle).unwrap();
        fixture.backend.configure_capacity(4).unwrap();
        fixture.execute(2, 10, None, 3.0);
        assert!(Arc::ptr_eq(
            &stale.as_ref().unwrap().backend_identity,
            &fixture.handle.as_ref().unwrap().backend_identity
        ));
        assert_ne!(
            stale.as_ref().unwrap().nonce,
            fixture.handle.as_ref().unwrap().nonce
        );
        let before = fixture.backend.capacity();
        let error = fixture.backend.commit(&mut stale).unwrap_err();
        assert!(error.to_string().contains("stale preparation nonce"));
        assert!(
            fixture
                .backend
                .inner
                .preflight_commit_ready(
                    stale.as_ref().unwrap(),
                    &binding(),
                    rank(0),
                    &fixture.states
                )
                .is_err()
        );
        assert!(
            fixture
                .backend
                .inner
                .install_commit(stale.as_mut().unwrap(), 1)
                .is_err()
        );
        assert!(fixture.backend.rollback(&mut stale).is_err());
        assert!(stale.is_some());
        assert_eq!(fixture.backend.capacity(), before);
        fixture.backend.rollback(&mut fixture.handle).unwrap();
    }

    #[test]
    fn cpu_backends_sharing_a_pool_have_distinct_ledger_authority() {
        let mut a = Fixture::fresh();
        let shared_pool = a.backend.inner.pool().clone();
        let mut old = Some(replay_private_handle(a.handle.as_ref().unwrap()));
        a.backend.rollback(&mut a.handle).unwrap();
        let mut b = Fixture::new();
        b.backend.inner = PagedKvBackend::new(shared_pool);
        b.execute(2, 10, None, 3.0);
        assert!(Rc::ptr_eq(
            &a.backend.inner.pool().identity,
            &b.backend.inner.pool().identity
        ));
        assert_eq!(
            old.as_ref().unwrap().nonce,
            b.handle.as_ref().unwrap().nonce
        );
        assert!(!Arc::ptr_eq(
            &old.as_ref().unwrap().backend_identity,
            &b.handle.as_ref().unwrap().backend_identity
        ));
        assert_foreign_ledger(b.backend.inner.commit(&mut old));
        assert_foreign_ledger(b.backend.inner.rollback(&mut old));
        assert_eq!(b.backend.capacity().active_transactions, 1);
        assert_eq!(b.backend.inner.pool().physical_slot(KvPageId(10)), None);
        b.backend.rollback(&mut b.handle).unwrap();
    }

    #[test]
    fn cpu_handle_nonce_exhaustion_does_not_reserve_physical_or_logical_pages() {
        let mut fixture = Fixture::new();
        fixture.backend.inner.next_handle_nonce = u64::MAX;
        let before = fixture.backend.capacity();
        for _ in 0..2 {
            let error = fixture
                .backend
                .prepare(DecoderKvPrepare {
                    new_pages: &[KvPageId(10)],
                    capacity: before,
                    ..clone_write_request(2, &[])
                })
                .unwrap_err();
            assert!(error.to_string().contains("handle nonce exhausted"));
            assert_eq!(fixture.backend.capacity(), before);
            assert_eq!(fixture.backend.inner.pool().free_slot_count(), 4);
            assert_eq!(fixture.backend.inner.pool().active_transaction_count(), 0);
        }
    }

    fn cpu_owners<'a>(
        a: &'a mut Fixture,
        b: &'a mut Fixture,
    ) -> Vec<KvCommitOwner<'a, CpuPagedKvBackend>> {
        vec![
            KvCommitOwner::new(rank(0), &mut a.backend.inner, &mut a.handle, &a.states),
            KvCommitOwner::new(rank(1), &mut b.backend.inner, &mut b.handle, &b.states),
        ]
    }

    fn clone_write_request<'a>(id: u64, writable: &'a [KvPageId]) -> DecoderKvPrepare<'a> {
        DecoderKvPrepare {
            transaction: tx(id),
            sequences: &[],
            new_pages: &[],
            writable_pages: writable,
            cow_replacements: &[],
            protected_pages: writable,
            capacity: DecoderKvCapacity::default(),
            page_statuses: &[],
        }
    }

    fn assert_clone_pinned(probe: &mut CpuPagedKvPool, pages: &[KvPageId]) {
        let before = pages
            .iter()
            .map(|p| probe.physical_slot(*p))
            .collect::<Vec<_>>();
        let free = probe.free_slot_count();
        assert!(PhysicalKvPool::release(probe, pages).is_err());
        assert!(PhysicalKvPool::preempt(probe, pages).is_err());
        assert!(PhysicalKvPool::shutdown(probe).is_err());
        assert!(PhysicalKvPool::configure_capacity(probe, 4).is_err());
        assert!(PhysicalKvPool::prepare(probe, clone_write_request(90, pages)).is_err());
        // A clone must not replace even a disjoint context under the leased ID.
        assert!(PhysicalKvPool::prepare(probe, clone_write_request(2, &[])).is_err());
        assert_eq!(probe.active_transaction_count(), 1);
        assert_eq!(probe.free_slot_count(), free);
        assert_eq!(
            pages
                .iter()
                .map(|p| probe.physical_slot(*p))
                .collect::<Vec<_>>(),
            before
        );
    }

    fn read_committed_page(probe: &mut CpuPagedKvPool, page: KvPageId) -> Vec<f32> {
        let mut reader = Some(
            probe
                .inner
                .prepare(cpu::PagedKvPrepare {
                    transaction: tx(99),
                    new_pages: &[],
                    writable_pages: &[],
                    cow_replacements: &[],
                    protected_pages: &[page],
                })
                .unwrap(),
        );
        probe
            .inner
            .enter(
                reader.as_mut().unwrap(),
                cpu::PagedKvBatch {
                    row_sequence_ids: vec![0].into(),
                    row_positions: vec![0].into(),
                    sequences: vec![cpu::PagedKvSequence {
                        sequence_len: 1,
                        block_table: vec![page].into(),
                    }]
                    .into(),
                },
            )
            .unwrap();
        let view = probe.inner.active_view(reader.as_mut().unwrap()).unwrap();
        let bytes = view.with_f32_page(page, 0, |v| v.to_vec()).unwrap();
        probe.inner.leave(reader.as_mut().unwrap()).unwrap();
        probe.inner.abort(&mut reader).unwrap();
        bytes
    }

    #[test]
    fn cpu_two_owner_install_keeps_precloned_pools_pinned_through_finish() {
        for cow in [false, true] {
            let (mut a, mut b) = if cow {
                (Fixture::cow(), Fixture::cow())
            } else {
                (Fixture::fresh(), Fixture::fresh())
            };
            let mut pa = a.backend.inner.pool().clone();
            let mut pb = b.backend.inner.pool().clone();
            let pages = if cow {
                vec![KvPageId(10), KvPageId(11)]
            } else {
                vec![KvPageId(10)]
            };
            let mut ready =
                PreparedKvCommit::prepare_commit_ready(binding(), (), cpu_owners(&mut a, &mut b))
                    .unwrap();
            ready.install_commit(&binding(), rank(0)).unwrap();
            assert_clone_pinned(&mut pa, &pages);
            assert!(ready.publish_logical(&binding(), |_| ((), vec![])).is_err());
            ready.install_commit(&binding(), rank(1)).unwrap();
            assert_clone_pinned(&mut pb, &pages);
            let mut retired = ready.publish_logical(&binding(), |_| ((), vec![])).unwrap();
            assert_clone_pinned(&mut pa, &pages);
            retired.retire_owner(&binding(), rank(0)).unwrap();
            assert!(retired.finish_retirement(&binding()).is_err());
            assert_clone_pinned(&mut pa, &pages);
            retired.retire_owner(&binding(), rank(1)).unwrap();
            assert_clone_pinned(&mut pa, &pages);
            retired.finish_retirement(&binding()).unwrap();
            drop(retired);
            drop(ready);
            for probe in [&mut pa, &mut pb] {
                assert_eq!(probe.active_transaction_count(), 0);
                assert_eq!(
                    read_committed_page(probe, KvPageId(10)),
                    vec![3.0, 0.0, 0.0, 0.0]
                );
                if cow {
                    assert_eq!(
                        read_committed_page(probe, KvPageId(11)),
                        vec![3.0, 9.0, 0.0, 0.0]
                    );
                }
                PhysicalKvPool::preempt(probe, &pages).unwrap();
                PhysicalKvPool::restore(probe, &pages).unwrap();
                PhysicalKvPool::release(probe, &pages).unwrap();
                assert_eq!(probe.free_slot_count(), 4);
                PhysicalKvPool::shutdown(probe).unwrap();
            }
            assert_eq!(a.backend.capacity().active_transactions, 0);
            assert_eq!(b.backend.capacity().active_transactions, 0);
        }
    }

    #[test]
    fn cpu_retired_cow_source_stays_quarantined_until_every_owner_finishes() {
        let (mut a, mut b) = (Fixture::cow(), Fixture::cow());
        let mut pa = a.backend.inner.pool().clone();
        let mut pb = b.backend.inner.pool().clone();
        let mut ready =
            PreparedKvCommit::prepare_commit_ready(binding(), (), cpu_owners(&mut a, &mut b))
                .unwrap();
        ready.install_commit(&binding(), rank(0)).unwrap();
        ready.install_commit(&binding(), rank(1)).unwrap();
        let mut retired = ready
            .publish_logical(&binding(), |_| ((), vec![KvPageId(10)]))
            .unwrap();
        retired.retire_owner(&binding(), rank(0)).unwrap();
        assert_eq!(pa.physical_slot(KvPageId(10)), None);
        assert_eq!(pa.free_slot_count(), 2);
        assert_clone_pinned(&mut pa, &[KvPageId(10), KvPageId(11)]);
        let request = DecoderKvPrepare {
            new_pages: &[KvPageId(10)],
            ..clone_write_request(90, &[])
        };
        assert!(PhysicalKvPool::prepare(&mut pa, request).is_err());
        assert!(retired.finish_retirement(&binding()).is_err());
        retired.retire_owner(&binding(), rank(1)).unwrap();
        assert_eq!(pb.free_slot_count(), 2);
        retired.finish_retirement(&binding()).unwrap();
        drop(retired);
        drop(ready);
        for probe in [&mut pa, &mut pb] {
            assert_eq!(probe.free_slot_count(), 3);
            assert_eq!(probe.active_transaction_count(), 0);
            assert_eq!(
                read_committed_page(probe, KvPageId(11)),
                vec![3.0, 9.0, 0.0, 0.0]
            );
            let mut reused = Some(PhysicalKvPool::prepare(probe, request).unwrap());
            PhysicalKvPool::abort(probe, &mut reused).unwrap();
            assert_eq!(probe.free_slot_count(), 3);
        }
        assert_eq!(
            a.backend.page_status(KvPageId(10)),
            DecoderKvPageStatus::Vacant
        );
        assert_eq!(a.backend.capacity().free_pages, 3);
        a.backend.release(&[KvPageId(11)]).unwrap();
        b.backend.release(&[KvPageId(11)]).unwrap();
        assert_eq!(a.backend.capacity().free_pages, 4);
        assert_eq!(pa.free_slot_count(), 4);
    }

    #[test]
    fn cpu_dropped_sealed_and_retirement_tokens_cannot_be_bypassed_by_clones() {
        for phase in 0..6 {
            let (mut a, mut b) = (Fixture::cow(), Fixture::cow());
            let mut pa = a.backend.inner.pool().clone();
            let mut pb = b.backend.inner.pool().clone();
            let mut ready =
                PreparedKvCommit::prepare_commit_ready(binding(), (), cpu_owners(&mut a, &mut b))
                    .unwrap();
            if phase >= 1 {
                ready.install_commit(&binding(), rank(0)).unwrap();
            }
            if phase >= 2 {
                ready.install_commit(&binding(), rank(1)).unwrap();
            }
            if phase >= 3 {
                let mut retired = ready
                    .publish_logical(&binding(), |_| ((), vec![KvPageId(10)]))
                    .unwrap();
                if phase >= 4 {
                    retired.retire_owner(&binding(), rank(0)).unwrap();
                }
                if phase >= 5 {
                    retired.retire_owner(&binding(), rank(1)).unwrap();
                }
                drop(retired);
            }
            drop(ready);
            assert!(a.handle.is_none() && b.handle.is_none());
            assert_eq!(a.backend.capacity().active_transactions, 1);
            assert_eq!(b.backend.capacity().active_transactions, 1);
            for probe in [&mut pa, &mut pb] {
                assert_clone_pinned(probe, &[KvPageId(10), KvPageId(11)]);
                assert_eq!(probe.free_slot_count(), 2);
                assert!(
                    PhysicalKvPool::prepare(
                        probe,
                        DecoderKvPrepare {
                            new_pages: &[KvPageId(10)],
                            ..clone_write_request(90, &[])
                        }
                    )
                    .is_err()
                );
            }
            // Even destroying the model ledger cannot remove shared physical pins.
            drop(a);
            drop(b);
            assert_clone_pinned(&mut pa, &[KvPageId(10), KvPageId(11)]);
            assert_clone_pinned(&mut pb, &[KvPageId(10), KvPageId(11)]);
        }
    }

    #[test]
    fn cpu_prepared_abort_preserves_clone_pins_until_all_cleanup_acks() {
        for abandon in [false, true] {
            let (mut a, mut b) = (Fixture::cow(), Fixture::cow());
            let mut pa = a.backend.inner.pool().clone();
            let mut pb = b.backend.inner.pool().clone();
            let mut ready =
                PreparedKvCommit::prepare_commit_ready(binding(), (), cpu_owners(&mut a, &mut b))
                    .unwrap();
            ready.abort_prepared(&binding(), rank(0)).unwrap();
            assert!(ready.abort_logical(&binding(), drop).is_err());
            assert_clone_pinned(&mut pa, &[KvPageId(10), KvPageId(11)]);
            assert_eq!(pa.free_slot_count(), 2);
            if abandon {
                drop(ready);
                drop(a);
                drop(b);
                assert_clone_pinned(&mut pa, &[KvPageId(10), KvPageId(11)]);
                assert_clone_pinned(&mut pb, &[KvPageId(10), KvPageId(11)]);
                assert_eq!(pa.free_slot_count(), 2);
            } else {
                ready.abort_prepared(&binding(), rank(1)).unwrap();
                assert_clone_pinned(&mut pb, &[KvPageId(10), KvPageId(11)]);
                ready.abort_logical(&binding(), drop).unwrap();
                drop(ready);
                assert_eq!(pa.free_slot_count(), 3);
                assert_eq!(pb.free_slot_count(), 3);
                assert_eq!(
                    read_committed_page(&mut pa, KvPageId(10)),
                    vec![3.0, 0.0, 0.0, 0.0]
                );
                a.backend.release(&[KvPageId(10)]).unwrap();
                b.backend.release(&[KvPageId(10)]).unwrap();
                assert_eq!(a.backend.capacity().free_pages, 4);
                assert_eq!(pa.free_slot_count(), 4);
            }
        }
    }

    #[test]
    fn cpu_ordinary_single_phase_cow_commit_releases_all_transaction_pins() {
        let mut a = Fixture::cow();
        let probe = a.backend.inner.pool().clone();
        a.backend.commit(&mut a.handle).unwrap();
        assert!(a.handle.is_none());
        assert_eq!(a.backend.capacity().active_transactions, 0);
        assert_eq!(probe.active_transaction_count(), 0);
        a.backend.release(&[KvPageId(10), KvPageId(11)]).unwrap();
        assert_eq!(probe.free_slot_count(), 4);
        a.backend.shutdown().unwrap();
    }

    #[test]
    fn two_owners_publish_only_after_both_physical_acks() {
        let (mut a, mut b) = (Fixture::fresh(), Fixture::fresh());
        let bind = binding();
        let (logical, dropped) = reservation();
        let visible = Cell::new(false);
        let mut ready =
            PreparedKvCommit::prepare_commit_ready(bind.clone(), logical, owners(&mut a, &mut b))
                .unwrap();
        assert_eq!(
            ready.page_status(rank(0), KvPageId(10)).unwrap(),
            DecoderKvPageStatus::Vacant
        );
        assert!(
            ready
                .publish_logical(&bind, |_| {
                    visible.set(true);
                    ((), vec![])
                })
                .is_err()
        );
        let ack = ready.install_commit(&bind, rank(0)).unwrap();
        assert_eq!(ack.binding(), &bind);
        assert_eq!(ack.rank(), rank(0));
        assert_eq!(ack.progress(), KvEndProgress::Complete);
        assert!(
            ready
                .publish_logical(&bind, |_| {
                    visible.set(true);
                    ((), vec![])
                })
                .is_err()
        );
        assert!(!visible.get());
        assert_eq!(dropped.get(), 0);
        ready.install_commit(&bind, rank(1)).unwrap();
        let mut retirement = ready
            .publish_logical(&bind, |logical| {
                drop(logical);
                visible.set(true);
                (77, vec![])
            })
            .unwrap();
        assert!(visible.get());
        assert_eq!(dropped.get(), 1);
        assert!(retirement.finish_retirement(&bind).is_err());
        retirement.retire_owner(&bind, rank(0)).unwrap();
        assert!(retirement.finish_retirement(&bind).is_err());
        retirement.retire_owner(&bind, rank(1)).unwrap();
        assert_eq!(retirement.finish_retirement(&bind).unwrap(), 77);
        assert!(retirement.finish_retirement(&bind).is_err());
        assert!(ready.install_commit(&bind, rank(0)).is_err());
        drop(retirement);
        drop(ready);
        assert!(a.handle.is_none() && b.handle.is_none());
        assert_eq!(a.backend.capacity().active_transactions, 0);
        assert_eq!(
            a.backend.page_status(KvPageId(10)),
            DecoderKvPageStatus::Resident
        );
    }

    #[test]
    fn late_preflight_failure_is_atomic_and_returns_all_custody() {
        let (mut a, mut b) = (Fixture::fresh(), Fixture::fresh());
        let faults = b.backend.faults.clone();
        faults.preflight.set(true);
        let (logical, dropped) = reservation();
        let error =
            PreparedKvCommit::prepare_commit_ready(binding(), logical, owners(&mut a, &mut b))
                .err()
                .expect("late owner must reject");
        let (_, logical, inputs) = error.into_parts();
        assert_eq!(dropped.get(), 0);
        faults.preflight.set(false);
        let mut ready = PreparedKvCommit::prepare_commit_ready(binding(), logical, inputs).unwrap();
        ready.abort_prepared(&binding(), rank(0)).unwrap();
        assert!(ready.abort_logical(&binding(), drop).is_err());
        ready.abort_prepared(&binding(), rank(1)).unwrap();
        ready.abort_logical(&binding(), drop).unwrap();
        drop(ready);
        assert_eq!(dropped.get(), 1);
        assert_eq!(a.backend.inner.pool().free_slot_count(), 4);
        assert_eq!(b.backend.faults.installs.get(), 0);
    }

    #[test]
    fn pending_and_lost_install_acks_are_polled_not_reinstalled() {
        for lost in [false, true] {
            let (mut a, mut b) = (Fixture::fresh(), Fixture::fresh());
            let faults = b.backend.faults.clone();
            faults.install_pending.set(!lost);
            faults.install_unknown.set(lost);
            let mut ready =
                PreparedKvCommit::prepare_commit_ready(binding(), (), owners(&mut a, &mut b))
                    .unwrap();
            ready.install_commit(&binding(), rank(0)).unwrap();
            let result = ready.install_commit(&binding(), rank(1));
            if lost {
                assert!(result.is_err());
            } else {
                assert_eq!(result.unwrap().progress(), KvEndProgress::Pending);
            }
            assert!(ready.install_commit(&binding(), rank(1)).is_err());
            assert!(ready.abort_prepared(&binding(), rank(0)).is_err());
            assert!(ready.publish_logical(&binding(), |_| ((), vec![])).is_err());
            assert_eq!(
                ready.page_status(rank(0), KvPageId(10)).unwrap(),
                DecoderKvPageStatus::Vacant
            );
            assert_eq!(ready.capacity(rank(1)).unwrap().active_transactions, 1);
            faults.install_pending.set(false);
            faults.install_unknown.set(false);
            assert_eq!(
                ready
                    .poll_install_ack(&binding(), rank(1))
                    .unwrap()
                    .progress(),
                KvEndProgress::Complete
            );
            assert_eq!(faults.installs.get(), 1);
            assert_eq!(faults.polls.get(), 1);
            let mut retired = ready.publish_logical(&binding(), |_| ((), vec![])).unwrap();
            for r in [0, 1] {
                retired.retire_owner(&binding(), rank(r)).unwrap();
            }
            retired.finish_retirement(&binding()).unwrap();
        }
    }

    #[test]
    fn stale_binding_and_duplicate_install_do_not_dispatch() {
        let (mut a, mut b) = (Fixture::fresh(), Fixture::fresh());
        let faults = a.backend.faults.clone();
        let mut ready =
            PreparedKvCommit::prepare_commit_ready(binding(), (), owners(&mut a, &mut b)).unwrap();
        for stale in [
            scope(2, 1, 2, &[0, 1]),
            scope(1, 2, 2, &[0, 1]),
            scope(1, 1, 3, &[0, 1]),
            scope(1, 1, 2, &[0]),
        ] {
            assert!(ready.install_commit(&stale, rank(0)).is_err());
        }
        assert!(ready.install_commit(&binding(), rank(2)).is_err());
        assert_eq!(faults.installs.get(), 0);
        ready.install_commit(&binding(), rank(0)).unwrap();
        assert!(ready.install_commit(&binding(), rank(0)).is_err());
        assert_eq!(faults.installs.get(), 1);
        ready.install_commit(&binding(), rank(1)).unwrap();
        let mut retired = ready.publish_logical(&binding(), |_| ((), vec![])).unwrap();
        for r in [0, 1] {
            retired.retire_owner(&binding(), rank(r)).unwrap();
        }
        retired.finish_retirement(&binding()).unwrap();
    }

    #[test]
    fn unknown_cleanup_keeps_logical_reservation_and_owner_pins_until_retry() {
        let (mut a, mut b) = (Fixture::fresh(), Fixture::fresh());
        let faults = b.backend.faults.clone();
        faults.abort_unknown.set(true);
        let probe = b.backend.inner.pool().clone();
        let (logical, dropped) = reservation();
        let mut ready =
            PreparedKvCommit::prepare_commit_ready(binding(), logical, owners(&mut a, &mut b))
                .unwrap();
        ready.abort_prepared(&binding(), rank(0)).unwrap();
        assert!(ready.abort_prepared(&binding(), rank(1)).is_err());
        assert!(ready.abort_logical(&binding(), drop).is_err());
        assert_eq!(dropped.get(), 0);
        assert_eq!(probe.free_slot_count(), 3);
        assert_eq!(ready.capacity(rank(0)).unwrap().active_transactions, 1);
        assert_eq!(
            ready
                .abort_prepared(&binding(), rank(0))
                .unwrap()
                .progress(),
            KvEndProgress::Complete
        );
        faults.abort_unknown.set(false);
        ready.abort_prepared(&binding(), rank(1)).unwrap();
        ready.abort_logical(&binding(), drop).unwrap();
        assert_eq!(dropped.get(), 1);
        assert_eq!(probe.free_slot_count(), 4);
    }

    #[test]
    fn cow_sources_and_retirement_wait_for_all_owner_fences() {
        for shared_source in [false, true] {
            let (mut a, mut b) = (Fixture::cow(), Fixture::cow());
            let probe = a.backend.inner.pool().clone();
            let faults = b.backend.faults.clone();
            faults.retirement_unknown.set(true);
            let mut ready =
                PreparedKvCommit::prepare_commit_ready(binding(), (), owners(&mut a, &mut b))
                    .unwrap();
            assert!(probe.physical_slot(KvPageId(10)).is_some());
            assert!(probe.physical_slot(KvPageId(11)).is_none());
            ready.install_commit(&binding(), rank(0)).unwrap();
            assert!(ready.publish_logical(&binding(), |_| ((), vec![])).is_err());
            assert!(probe.physical_slot(KvPageId(10)).is_some());
            ready.install_commit(&binding(), rank(1)).unwrap();
            let (quarantine, dropped) = reservation();
            let pages = if shared_source {
                vec![]
            } else {
                vec![KvPageId(10)]
            };
            let mut retired = ready
                .publish_logical(&binding(), |_| (quarantine, pages))
                .unwrap();
            retired.retire_owner(&binding(), rank(0)).unwrap();
            assert!(retired.retire_owner(&binding(), rank(1)).is_err());
            assert!(retired.finish_retirement(&binding()).is_err());
            assert_eq!(dropped.get(), 0);
            faults.retirement_unknown.set(false);
            retired.retire_owner(&binding(), rank(1)).unwrap();
            drop(retired.finish_retirement(&binding()).unwrap());
            assert_eq!(faults.releases.get(), 1);
            assert_eq!(faults.retirement_attempts.get(), 2);
            assert_eq!(dropped.get(), 1);
            assert_eq!(probe.physical_slot(KvPageId(10)).is_some(), shared_source);
            assert!(probe.physical_slot(KvPageId(11)).is_some());
        }
    }

    #[test]
    fn duplicate_and_missing_participants_fail_before_sealing() {
        for duplicate in [false, true] {
            let (mut a, mut b) = (Fixture::fresh(), Fixture::fresh());
            let mut inputs = owners(&mut a, &mut b);
            if duplicate {
                inputs[1].rank = rank(0);
            } else {
                inputs.pop();
            }
            let error = PreparedKvCommit::prepare_commit_ready(binding(), (), inputs)
                .err()
                .expect("invalid participant set");
            let (_, _, inputs) = error.into_parts();
            drop(inputs);
            assert!(a.handle.is_some() && b.handle.is_some());
            assert_eq!(a.backend.faults.installs.get(), 0);
            assert_eq!(b.backend.faults.installs.get(), 0);
            a.backend.rollback(&mut a.handle).unwrap();
            b.backend.rollback(&mut b.handle).unwrap();
        }
    }

    #[test]
    fn abandoned_unknown_owner_cannot_be_released_via_legacy_api() {
        let (mut a, mut b) = (Fixture::cow(), Fixture::cow());
        let faults = b.backend.faults.clone();
        faults.install_unknown.set(true);
        let (logical, dropped) = reservation();
        let mut ready =
            PreparedKvCommit::prepare_commit_ready(binding(), logical, owners(&mut a, &mut b))
                .unwrap();
        ready.install_commit(&binding(), rank(0)).unwrap();
        assert!(ready.install_commit(&binding(), rank(1)).is_err());
        drop(ready);
        assert_eq!(dropped.get(), 0);
        assert!(a.handle.is_none() && b.handle.is_none());
        assert!(a.backend.release(&[KvPageId(10)]).is_err());
        assert!(b.backend.shutdown().is_err());
        assert_eq!(b.backend.capacity().active_transactions, 1);
        assert_eq!(
            b.backend.page_status(KvPageId(11)),
            DecoderKvPageStatus::Vacant
        );
    }

    #[test]
    fn cpu_cow_preflight_rejects_stale_source_mapping_before_sealing() {
        let (mut a, mut b) = (Fixture::cow(), Fixture::cow());
        let physical = b
            .backend
            .inner
            .ownership
            .as_mut()
            .unwrap()
            .payload_mut(tx(2))
            .unwrap()
            .as_mut()
            .unwrap();
        physical.slots.insert(KvPageId(10), Some(999));
        let error =
            PreparedKvCommit::prepare_commit_ready(binding(), (), cpu_owners(&mut a, &mut b))
                .err()
                .expect("stale COW source mapping must reject before sealing");
        let (error, (), inputs) = error.into_parts();
        assert!(error.to_string().contains("physical mapping"));
        drop(inputs);
        assert!(a.handle.is_some() && b.handle.is_some());
        assert_eq!(a.backend.capacity().active_transactions, 1);
        assert_eq!(b.backend.capacity().active_transactions, 1);
        assert_eq!(a.backend.faults.installs.get(), 0);
        assert_eq!(b.backend.faults.installs.get(), 0);
        a.backend.rollback(&mut a.handle).unwrap();
        b.backend.rollback(&mut b.handle).unwrap();
    }

    #[test]
    fn cpu_preflight_rejects_stale_generation_mapping_shape_capacity_and_fence() {
        for fault in 0..5 {
            let (mut a, mut b) = (Fixture::fresh(), Fixture::fresh());
            if fault == 0 {
                b.states[0].core_mut().reset();
            } else {
                let physical = b
                    .backend
                    .inner
                    .ownership
                    .as_mut()
                    .unwrap()
                    .payload_mut(tx(2))
                    .unwrap()
                    .as_mut()
                    .unwrap();
                match fault {
                    1 => {
                        physical.slots.insert(KvPageId(10), Some(99));
                    }
                    2 => physical.planes[0].elements_per_token += 1,
                    3 => physical.capacity += 1,
                    4 => physical.checked_slots = None,
                    _ => unreachable!(),
                }
            }
            let error =
                PreparedKvCommit::prepare_commit_ready(binding(), (), owners(&mut a, &mut b))
                    .err()
                    .expect("preflight must reject");
            let (_, _, inputs) = error.into_parts();
            drop(inputs);
            assert_eq!(a.backend.faults.installs.get(), 0);
            assert!(a.handle.is_some() && b.handle.is_some());
            a.backend.rollback(&mut a.handle).unwrap();
            b.backend.rollback(&mut b.handle).unwrap();
        }
    }
}
