use super::{DecoderKvPageStatus, KvCommitProjection, PackedDecoderBatch};
use crate::execution::{ResolvedStage, SequenceTopologyId};
use ferrule_common::execution::{ExecutionTransactionId, KvCowReplacement, KvPageId, StateSlot};
use ferrule_common::{ContinuationId, DependencySet, ResidencyLeaseSet, Result};
/// Per-sequence KV custody captured before a packed batch is remapped into
/// transaction-local sequence order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderKvSequenceCustody {
    pub source_index: usize,
    pub topology_id: SequenceTopologyId,
    pub page_state_slot: StateSlot,
    pub page_generation: u64,
    pub execution_generation: u64,
    pub context_len: usize,
    pub query_len: usize,
}
/// Status observed for one page at the exact prepare boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderKvPageSnapshot {
    pub page: KvPageId,
    pub status: DecoderKvPageStatus,
}
/// Exact backend KV preparation request derived from one validated packed batch.
#[derive(Debug, Clone, Copy)]
pub struct DecoderKvPrepare<'a> {
    pub transaction: ExecutionTransactionId,
    pub sequences: &'a [DecoderKvSequenceCustody],
    pub new_pages: &'a [KvPageId],
    pub writable_pages: &'a [KvPageId],
    pub cow_replacements: &'a [KvCowReplacement],
    pub protected_pages: &'a [KvPageId],
    pub capacity: DecoderKvCapacity,
    pub page_statuses: &'a [DecoderKvPageSnapshot],
}
/// Structured terminal progress for KV backend custody.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvEndProgress {
    /// Physical work still owns the transaction and all protected dependencies.
    Pending,
    /// The transaction was consumed and the terminal transition completed.
    Complete,
    /// The transaction was consumed, but the backend rejected publication/rollback.
    /// The shell must release registry custody.
    ConsumedRejected,
}
/// Exact cohort identity supplied by the existing decision authority. This is
/// KV custody metadata, not a second transaction registry or global decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCommitBinding {
    transaction: ExecutionTransactionId,
    topology_id: ferrule_common::ParallelTopologyId,
    participants: ferrule_common::ParticipantSet,
    generation: u64,
}
impl KvCommitBinding {
    pub fn new(
        transaction: ExecutionTransactionId,
        topology_id: ferrule_common::ParallelTopologyId,
        participants: ferrule_common::ParticipantSet,
        generation: u64,
    ) -> Result<Self> {
        if generation == 0 || participants.is_empty() || participants.topology_id() != topology_id {
            return Err(ferrule_common::Error::Execution {
                message: "invalid KV commit binding".into(),
            });
        }
        Ok(Self {
            transaction,
            topology_id,
            participants,
            generation,
        })
    }
    pub const fn transaction(&self) -> ExecutionTransactionId {
        self.transaction
    }
    pub const fn topology_id(&self) -> ferrule_common::ParallelTopologyId {
        self.topology_id
    }
    pub const fn participants(&self) -> &ferrule_common::ParticipantSet {
        &self.participants
    }
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

/// Cohort-wide ACK observation; Pending and rejection never authorize publish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvRankAck {
    binding: KvCommitBinding,
    rank: ferrule_common::ParallelRankId,
    owner_generation: u64,
    progress: KvEndProgress,
}
impl KvRankAck {
    pub(super) const fn new(
        binding: KvCommitBinding,
        rank: ferrule_common::ParallelRankId,
        owner_generation: u64,
        progress: KvEndProgress,
    ) -> Self {
        Self {
            binding,
            rank,
            owner_generation,
            progress,
        }
    }
    pub const fn binding(&self) -> &KvCommitBinding {
        &self.binding
    }
    pub const fn rank(&self) -> ferrule_common::ParallelRankId {
        self.rank
    }
    pub const fn owner_generation(&self) -> u64 {
        self.owner_generation
    }
    pub const fn progress(&self) -> KvEndProgress {
        self.progress
    }
}

/// Additive all-owner KV adapter. No transport or global decision state lives
/// here. Complete means the operation AND all owner fences are quiescent;
/// ConsumedRejected is never a successful install/cleanup ACK.
///
/// Implementations retain the original transaction/ledger until finish_prepared.
/// CUDA must implement real mapping-generation, COW, capacity and fence checks;
/// it must not emulate preflight by calling commit. The CPU implementation is a
/// synchronous in-process reference, not a cross-process/CUDA KV implementation.
/// `PagedKvBackend<P>` opts in only through `PhysicalKvPreparedPool`; legacy MLA
/// pools retain their ordinary commit contract. CUDA F32 GQA uses the same
/// ledger/cohort with owner-local shadow slots and exact compute-stream fences.
///
/// Narrow prepared-KV participant used by the cohort protocol. This contract
/// carries only already-owned transaction custody and host commit metadata;
/// it does not imply execution, capacity inspection, or page lifecycle access.
///
/// Execution is not available through a participant-only bound:
/// ```compile_fail
/// use ferrule_model::decoder::{KvCommitParticipant, PackedDecoderBatch};
/// fn enter<P: KvCommitParticipant>(p: &mut P, tx: &mut P::Transaction,
///     batch: &PackedDecoderBatch, states: &mut [P::SequenceState]) {
///     p.enter(tx, batch, states).unwrap();
/// }
/// ```
/// Nor does sealing custody grant local capacity inspection:
/// ```compile_fail
/// use ferrule_model::decoder::{KvCommitParticipant, PreparedKvCommit};
/// use ferrule_common::ParallelRankId;
/// fn capacity<P: KvCommitParticipant>(token: &PreparedKvCommit<'_, P, ()>) {
///     token.capacity(ParallelRankId::new(0)).unwrap();
/// }
/// ```
/// ```compile_fail
/// use ferrule_model::decoder::{KvCommitParticipant, PreparedKvCommit};
/// use ferrule_common::{ParallelRankId, execution::KvPageId};
/// fn status<P: KvCommitParticipant>(token: &PreparedKvCommit<'_, P, ()>) {
///     token.page_status(ParallelRankId::new(0), KvPageId(0)).unwrap();
/// }
/// ```
pub trait KvCommitParticipant {
    type SequenceState;
    /// Linear transaction custody; never reconstructed from an addressing key.
    type Transaction;

    /// Read-only: validate the exact existing physical reservation and sources.
    fn preflight_commit_ready(
        &self,
        transaction: &Self::Transaction,
        binding: &KvCommitBinding,
        rank: ferrule_common::ParallelRankId,
        sources: &[Self::SequenceState],
    ) -> Result<()>;
    /// Describe the already-validated logical projection, in packed order.
    /// Rank-local sequence identities may differ; logical page generations may not.
    fn commit_projection(&self, transaction: &Self::Transaction) -> Result<KvCommitProjection>;
    /// Exactly one dispatch. All retries use poll_install_ack, including after Err.
    fn install_commit(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress>;
    fn poll_install_ack(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress>;
    /// Retry cleanup, including a lost owner ACK, without guessing unknown pages free.
    fn abort_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress>;
    /// Infallible after all install ACKs. Keep page pins until finish_prepared.
    fn publish_committed(&mut self, transaction: &Self::Transaction);
    fn preflight_retirement(
        &self,
        transaction: &Self::Transaction,
        pages: &[KvPageId],
    ) -> Result<()>;
    /// Retry-safe release. Complete includes every physical release fence.
    fn retire_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
        pages: &[KvPageId],
    ) -> Result<KvEndProgress>;
    /// Infallible after all cleanup/retirement ACKs; consumes ledger custody only.
    fn finish_prepared(&mut self, transaction: Self::Transaction);
}

/// Local-only capacity and page-state witness. Remote participants do not
/// implement this capability. The distinct method names preserve legacy method
/// lookup when callers import both this inspector and `DecoderKvBackend`.
pub trait KvCapacityInspector {
    fn inspect_capacity(&self) -> DecoderKvCapacity;
    fn inspect_page_status(&self, page: KvPageId) -> super::DecoderKvPageStatus;
}

impl<T: DecoderKvBackend + ?Sized> KvCapacityInspector for T {
    fn inspect_capacity(&self) -> DecoderKvCapacity {
        DecoderKvBackend::capacity(self)
    }

    fn inspect_page_status(&self, page: KvPageId) -> super::DecoderKvPageStatus {
        DecoderKvBackend::page_status(self, page)
    }
}

/// Compatibility adapter for local executable backends. Legacy MLA and
/// ordinary pools keep their existing public implementation; only opted-in
/// prepared pools implement this old trait and therefore bridge to the narrow
/// participant contract below.
impl<T: DecoderKvCommitBackend + ?Sized> KvCommitParticipant for T {
    type SequenceState = <T as DecoderKvBackend>::SequenceState;
    type Transaction = <T as DecoderKvBackend>::Transaction;

    fn preflight_commit_ready(
        &self,
        transaction: &Self::Transaction,
        binding: &KvCommitBinding,
        rank: ferrule_common::ParallelRankId,
        sources: &[Self::SequenceState],
    ) -> Result<()> {
        DecoderKvCommitBackend::preflight_commit_ready(self, transaction, binding, rank, sources)
    }
    fn commit_projection(&self, transaction: &Self::Transaction) -> Result<KvCommitProjection> {
        Ok(DecoderKvCommitBackend::commit_batch(self, transaction)?.commit_projection())
    }
    fn install_commit(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        DecoderKvCommitBackend::install_commit(self, transaction, generation)
    }
    fn poll_install_ack(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        DecoderKvCommitBackend::poll_install_ack(self, transaction, generation)
    }
    fn abort_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        DecoderKvCommitBackend::abort_prepared(self, transaction, generation)
    }
    fn publish_committed(&mut self, transaction: &Self::Transaction) {
        DecoderKvCommitBackend::publish_committed(self, transaction)
    }
    fn preflight_retirement(
        &self,
        transaction: &Self::Transaction,
        pages: &[KvPageId],
    ) -> Result<()> {
        DecoderKvCommitBackend::preflight_retirement(self, transaction, pages)
    }
    fn retire_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
        pages: &[KvPageId],
    ) -> Result<KvEndProgress> {
        DecoderKvCommitBackend::retire_prepared(self, transaction, generation, pages)
    }
    fn finish_prepared(&mut self, transaction: Self::Transaction) {
        DecoderKvCommitBackend::finish_prepared(self, transaction)
    }
}

/// Legacy executable prepared backend. Existing implementations are adapted
/// one way to `KvCommitParticipant`; ordinary backends do not opt in implicitly.
pub trait DecoderKvCommitBackend: DecoderKvBackend {
    /// Read-only: validate the exact existing physical reservation and sources.
    fn preflight_commit_ready(
        &self,
        transaction: &Self::Transaction,
        binding: &KvCommitBinding,
        rank: ferrule_common::ParallelRankId,
        sources: &[Self::SequenceState],
    ) -> Result<()>;
    /// Describe the already-validated logical projection, in packed order.
    /// Rank-local sequence identities may differ; logical page generations may not.
    fn commit_batch(&self, transaction: &Self::Transaction) -> Result<&PackedDecoderBatch>;
    /// Exactly one dispatch. All retries use poll_install_ack, including after Err.
    fn install_commit(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress>;
    fn poll_install_ack(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress>;
    /// Retry cleanup, including a lost owner ACK, without guessing unknown pages free.
    fn abort_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress>;
    /// Infallible after all install ACKs. Keep page pins until finish_prepared.
    fn publish_committed(&mut self, transaction: &Self::Transaction);
    fn preflight_retirement(
        &self,
        transaction: &Self::Transaction,
        pages: &[KvPageId],
    ) -> Result<()>;
    /// Retry-safe release. Complete includes every physical release fence.
    fn retire_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        generation: u64,
        pages: &[KvPageId],
    ) -> Result<KvEndProgress>;
    /// Infallible after all cleanup/retirement ACKs; consumes ledger custody only.
    fn finish_prepared(&mut self, transaction: Self::Transaction);
}

/// Capacity and lifecycle status of a decoder KV backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecoderKvCapacity {
    pub physical_pages: usize,
    pub resident_pages: usize,
    pub preempted_pages: usize,
    pub active_transactions: usize,
    pub free_pages: usize,
}
/// Model-neutral transactional KV backend.
pub trait DecoderKvBackend {
    type SequenceState;
    /// Non-cloneable backend transaction. Errors retain it and remain
    /// retryable; only `Complete` and `ConsumedRejected` consume it.
    type Transaction;
    /// Typed transaction-bound view passed to the forward executor.
    type KvView;
    fn configure_capacity(&mut self, max_pages: usize) -> Result<()>;
    /// A normal Err guarantees no unresolved prepare work. If device cleanup is
    /// unknown, return `KvPrepareQuiescenceUnknown` and retain backend custody;
    /// no returned handle must never be interpreted as a successful rollback.
    fn prepare(&mut self, request: DecoderKvPrepare<'_>) -> Result<Self::Transaction>;
    fn enter(
        &mut self,
        kv_transaction: &mut Self::Transaction,
        batch: &PackedDecoderBatch,
        states: &mut [Self::SequenceState],
    ) -> Result<()>;
    /// Reconstructs a typed view while the transaction is entered. The view may
    /// be short-lived; transaction custody remains in `Transaction` across waits.
    fn active_view(&mut self, kv_transaction: &mut Self::Transaction) -> Result<Self::KvView>;
    /// Constructs the typed committed-KV view used by one proposal call.
    fn proposal_view(
        &mut self,
        _transaction: ExecutionTransactionId,
        _state: &mut Self::SequenceState,
    ) -> Result<Self::KvView> {
        Err(ferrule_common::Error::Execution {
            message: "decoder KV backend has no proposal view".into(),
        })
    }
    fn leave(&mut self, kv_transaction: &mut Self::Transaction) -> Result<()>;
    /// On `Err`, the transaction must remain present and retryable. A consumed
    /// rejection is represented only by `ConsumedRejected`.
    fn commit(&mut self, kv_transaction: &mut Option<Self::Transaction>) -> Result<KvEndProgress>;
    /// On `Err`, the transaction must remain present and retryable. A consumed
    /// rejection is represented only by `ConsumedRejected`.
    fn rollback(&mut self, kv_transaction: &mut Option<Self::Transaction>)
    -> Result<KvEndProgress>;
    fn release(&mut self, pages: &[KvPageId]) -> Result<()>;
    fn preempt(&mut self, pages: &[KvPageId]) -> Result<()>;
    fn restore(&mut self, pages: &[KvPageId]) -> Result<()>;
    fn capacity(&self) -> DecoderKvCapacity;
    fn page_status(&self, page: KvPageId) -> super::DecoderKvPageStatus;
    /// Quiesces backend-owned resources. Implementations must reject shutdown
    /// while transaction custody remains active.
    fn shutdown(&mut self) -> Result<()>;
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecoderWait {
    Dependencies(DependencySet),
    ResolvedStage(ResolvedStage),
}
impl DecoderWait {
    fn dependencies(&self) -> Result<&DependencySet> {
        match self {
            Self::Dependencies(dependencies) => {
                dependencies.validate()?;
                Ok(dependencies)
            }
            Self::ResolvedStage(stage) => {
                stage
                    .dependencies()
                    .ok_or_else(|| ferrule_common::Error::Execution {
                        message: "a resource-free decoder stage cannot suspend".into(),
                    })
            }
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderContinuationWait {
    continuation_id: ContinuationId,
    wait: DecoderWait,
}
impl DecoderContinuationWait {
    pub fn new(continuation_id: ContinuationId, dependencies: DependencySet) -> Result<Self> {
        Self::from_wait(continuation_id, DecoderWait::Dependencies(dependencies))
    }
    pub fn from_wait(continuation_id: ContinuationId, wait: DecoderWait) -> Result<Self> {
        if continuation_id.is_zero() {
            return Err(ferrule_common::Error::Execution {
                message: "decoder continuation ID must be non-zero".into(),
            });
        }
        wait.dependencies()?;
        Ok(Self {
            continuation_id,
            wait,
        })
    }
    pub const fn continuation_id(&self) -> ContinuationId {
        self.continuation_id
    }
    pub fn dependencies(&self) -> &DependencySet {
        self.wait
            .dependencies()
            .expect("validated decoder wait retains dependencies")
    }
    pub const fn wait(&self) -> &DecoderWait {
        &self.wait
    }
    pub fn validate_resume_leases(&self, leases: &ResidencyLeaseSet) -> Result<()> {
        let required = self
            .dependencies()
            .iter()
            .filter_map(|dependency| dependency.materialization_key())
            .collect::<Vec<_>>();
        if leases.len() != required.len()
            || required
                .iter()
                .any(|key| leases.binding_for(*key).is_none())
        {
            return Err(ferrule_common::Error::Execution {
                message: format!(
                    "decoder continuation {} lease set does not exactly satisfy its dependency custody",
                    self.continuation_id.get()
                ),
            });
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderCancelProgress {
    Waiting,
    Complete,
}
