use super::{DecoderKvPageStatus, PackedDecoderBatch};
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
